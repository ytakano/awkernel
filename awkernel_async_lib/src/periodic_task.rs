//! Helpers for spawning logical periodic jobs on top of one Awkernel async task.
//!
//! A periodic task created by this module reuses a single runtime task. Each
//! iteration of the user-provided job body is treated as a logical periodic job
//! and is identified by a monotonically increasing `loop_index`.
//!
//! The runtime task is scheduled by `SchedulerType::GEDF(relative_deadline)`.
//! This module supplies the logical release time and absolute deadline for each
//! iteration so the GEDF scheduler and the baseline trace adapter can observe a
//! periodic job sequence without allocating a new runtime task for every job.
//!
//! When the `baseline_trace` feature is enabled, each periodic release can be
//! emitted as a loop-indexed `RunnableDeadline` row, and each completed
//! iteration is emitted as `PeriodicJobComplete(task, loop_index)`. These rows
//! are adapter-local trace evidence; the common scheduling model still treats
//! time as discrete values and does not depend on Awkernel's concrete timer
//! source.

use crate::{
    scheduler::SchedulerType,
    task::{self, GedfDeadlineHint, TaskResult},
};
use alloc::borrow::Cow;
use core::{future::Future, time::Duration};

/// Period and relative deadline parameters for a periodic GEDF task.
///
/// Both durations must be positive and convertible to microseconds. The period
/// determines the logical release spacing between consecutive iterations. The
/// relative deadline is stored in the runtime scheduler as
/// `SchedulerType::GEDF(relative_deadline_us)` and is also used to compute each
/// logical absolute deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicTaskSpec {
    /// Logical spacing between consecutive releases.
    pub period: Duration,

    /// Deadline relative to each logical release time.
    pub relative_deadline: Duration,
}

/// Context passed to one invocation of a periodic task body.
///
/// The same Awkernel runtime task calls the user job repeatedly. `loop_index`
/// distinguishes those logical jobs. `logical_release_time_us` and
/// `absolute_deadline_us` are expressed in the same discrete microsecond-scale
/// units used by `awkernel_lib::delay::uptime()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicJobContext {
    /// Zero-based logical job index.
    pub loop_index: u64,

    /// Logical release time for this iteration.
    pub logical_release_time_us: u64,

    /// Logical absolute deadline for this iteration.
    pub absolute_deadline_us: u64,

    /// Period configured for the task.
    pub period: Duration,

    /// Relative deadline configured for the task.
    pub relative_deadline: Duration,
}

/// Errors returned while creating or advancing a periodic task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicTaskError {
    /// The configured period was zero.
    ZeroPeriod,

    /// The configured relative deadline was zero.
    ZeroRelativeDeadline,

    /// A duration, release time, deadline, or loop index did not fit in `u64`.
    DurationOverflow,
}

/// Decision returned by a controlled periodic job body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicJobDisposition {
    /// Continue with the next logical periodic job.
    Continue,

    /// Complete the periodic runtime task after the current logical job.
    Complete,
}

impl PeriodicTaskSpec {
    /// Builds a validated periodic task specification.
    ///
    /// `period` and `relative_deadline` must be non-zero and representable in
    /// whole microseconds as `u64`. This constructor does not require
    /// `relative_deadline <= period`; deadlines larger than the period are
    /// accepted when the caller wants to model overlapping logical jobs at the
    /// trace level.
    pub fn new(period: Duration, relative_deadline: Duration) -> Result<Self, PeriodicTaskError> {
        if period.is_zero() {
            return Err(PeriodicTaskError::ZeroPeriod);
        }
        if relative_deadline.is_zero() {
            return Err(PeriodicTaskError::ZeroRelativeDeadline);
        }
        duration_to_us(period)?;
        duration_to_us(relative_deadline)?;
        Ok(Self {
            period,
            relative_deadline,
        })
    }

    fn period_us(self) -> Result<u64, PeriodicTaskError> {
        duration_to_us(self.period)
    }

    fn relative_deadline_us(self) -> Result<u64, PeriodicTaskError> {
        duration_to_us(self.relative_deadline)
    }
}

fn duration_to_us(duration: Duration) -> Result<u64, PeriodicTaskError> {
    u64::try_from(duration.as_micros()).map_err(|_| PeriodicTaskError::DurationOverflow)
}

fn deadline_hint(
    logical_release_time_us: u64,
    relative_deadline_us: u64,
    loop_index: u64,
) -> Result<GedfDeadlineHint, PeriodicTaskError> {
    let absolute_deadline = logical_release_time_us
        .checked_add(relative_deadline_us)
        .ok_or(PeriodicTaskError::DurationOverflow)?;
    Ok(GedfDeadlineHint {
        logical_release_time: logical_release_time_us,
        absolute_deadline,
        periodic_loop_index: Some(loop_index),
    })
}

fn wait_duration_until(logical_release_time_us: u64) -> Duration {
    let now = awkernel_lib::delay::uptime();
    Duration::from_micros(logical_release_time_us.saturating_sub(now))
}

/// Spawns a periodic GEDF task and returns its Awkernel runtime task id.
///
/// The first logical release is the current `uptime()` observed during spawn.
/// After each job body completes, the helper advances the logical release time
/// by `spec.period`, installs the next GEDF deadline hint, and sleeps until that
/// next release using an untraced sleep. If the job body returns `Err`, the
/// periodic task returns that error and stops.
///
/// The closure receives a [`PeriodicJobContext`] for each logical job. Use
/// `context.loop_index` to distinguish jobs that reuse the same runtime task.
///
/// # Example
///
/// ```
/// use std::{borrow::Cow, time::Duration};
///
/// use awkernel_async_lib::{
///     spawn_periodic_task, PeriodicJobContext, PeriodicTaskSpec,
/// };
///
/// let _ = async {
///     let spec = PeriodicTaskSpec::new(
///         Duration::from_millis(10),
///         Duration::from_millis(5),
///     )
///     .expect("periodic task parameters must be valid");
///
///     let task_id = spawn_periodic_task(
///         Cow::Borrowed("sensor-sampler"),
///         spec,
///         |context: PeriodicJobContext| async move {
///             let _loop_index = context.loop_index;
///             let _deadline_us = context.absolute_deadline_us;
///
///             // Do one logical job of periodic work here.
///             Ok::<(), Cow<'static, str>>(())
///         },
///     )
///     .expect("periodic task should spawn");
///
///     let _: u32 = task_id;
/// };
/// ```
pub fn spawn_periodic_task<F, Fut>(
    name: Cow<'static, str>,
    spec: PeriodicTaskSpec,
    mut job: F,
) -> Result<u32, PeriodicTaskError>
where
    F: FnMut(PeriodicJobContext) -> Fut + Send + 'static,
    Fut: Future<Output = TaskResult> + Send + 'static,
{
    spawn_periodic_task_controlled(name, spec, move |context| {
        let future = job(context);
        async move {
            future.await?;
            Ok(PeriodicJobDisposition::Continue)
        }
    })
}

/// Spawns a periodic GEDF task whose job body can stop the periodic loop.
///
/// This is useful for trace workloads and tests that need a finite periodic
/// sequence. `PeriodicJobDisposition::Complete` completes the runtime task
/// after the current logical job has emitted its `PeriodicJobComplete` trace
/// row.
pub fn spawn_periodic_task_controlled<F, Fut>(
    name: Cow<'static, str>,
    spec: PeriodicTaskSpec,
    mut job: F,
) -> Result<u32, PeriodicTaskError>
where
    F: FnMut(PeriodicJobContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<PeriodicJobDisposition, Cow<'static, str>>> + Send + 'static,
{
    let period_us = spec.period_us()?;
    let relative_deadline_us = spec.relative_deadline_us()?;
    let first_release_time_us = awkernel_lib::delay::uptime();
    let first_hint = deadline_hint(first_release_time_us, relative_deadline_us, 0)?;

    let future = async move {
        let mut loop_index = 0;
        let mut logical_release_time_us = first_release_time_us;

        loop {
            let hint = deadline_hint(logical_release_time_us, relative_deadline_us, loop_index)
                .map_err(|_| Cow::Borrowed("periodic task deadline overflow"))?;
            let context = PeriodicJobContext {
                loop_index,
                logical_release_time_us,
                absolute_deadline_us: hint.absolute_deadline,
                period: spec.period,
                relative_deadline: spec.relative_deadline,
            };

            let disposition = job(context).await;

            #[cfg(feature = "baseline_trace")]
            task::record_current_periodic_job_complete(loop_index);

            task::clear_current_gedf_deadline_hint();

            match disposition? {
                PeriodicJobDisposition::Continue => {}
                PeriodicJobDisposition::Complete => return Ok(()),
            }

            loop_index = loop_index
                .checked_add(1)
                .ok_or(Cow::Borrowed("periodic task loop index overflow"))?;
            logical_release_time_us = logical_release_time_us
                .checked_add(period_us)
                .ok_or(Cow::Borrowed("periodic task release time overflow"))?;
            let next_hint =
                deadline_hint(logical_release_time_us, relative_deadline_us, loop_index)
                    .map_err(|_| Cow::Borrowed("periodic task deadline overflow"))?;
            task::set_current_next_gedf_deadline_hint(next_hint);
            crate::sleep_untraced(wait_duration_until(logical_release_time_us)).await;
        }
    };

    let spawned = task::spawn_with_ids_and_gedf_hint(
        name,
        future,
        SchedulerType::GEDF(relative_deadline_us),
        None,
        Some(first_hint),
    );
    Ok(spawned.runtime_task_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_periodic_task_spec() {
        assert_eq!(
            PeriodicTaskSpec::new(Duration::ZERO, Duration::from_micros(1)),
            Err(PeriodicTaskError::ZeroPeriod)
        );
        assert_eq!(
            PeriodicTaskSpec::new(Duration::from_micros(1), Duration::ZERO),
            Err(PeriodicTaskError::ZeroRelativeDeadline)
        );
        assert!(
            PeriodicTaskSpec::new(Duration::from_micros(10), Duration::from_micros(20)).is_ok()
        );
    }

    #[test]
    fn builds_deadline_hint_with_loop_index() {
        let hint = deadline_hint(10, 5, 3).unwrap();
        assert_eq!(hint.logical_release_time, 10);
        assert_eq!(hint.absolute_deadline, 15);
        assert_eq!(hint.periodic_loop_index, Some(3));
    }

    #[test]
    fn periodic_job_disposition_values_are_distinct() {
        assert_ne!(
            PeriodicJobDisposition::Continue,
            PeriodicJobDisposition::Complete
        );
    }
}
