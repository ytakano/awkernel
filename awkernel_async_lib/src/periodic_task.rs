use crate::{
    scheduler::SchedulerType,
    task::{self, GedfDeadlineHint, TaskResult},
};
use alloc::borrow::Cow;
use core::{future::Future, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicTaskSpec {
    pub period: Duration,
    pub relative_deadline: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodicJobContext {
    pub loop_index: u64,
    pub logical_release_time_us: u64,
    pub absolute_deadline_us: u64,
    pub period: Duration,
    pub relative_deadline: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicTaskError {
    ZeroPeriod,
    ZeroRelativeDeadline,
    DurationOverflow,
}

impl PeriodicTaskSpec {
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

pub fn spawn_periodic_task<F, Fut>(
    name: Cow<'static, str>,
    spec: PeriodicTaskSpec,
    mut job: F,
) -> Result<u32, PeriodicTaskError>
where
    F: FnMut(PeriodicJobContext) -> Fut + Send + 'static,
    Fut: Future<Output = TaskResult> + Send + 'static,
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

            let result = job(context).await;

            #[cfg(feature = "baseline_trace")]
            task::record_current_periodic_job_complete(loop_index);

            task::clear_current_gedf_deadline_hint();

            if let Err(err) = result {
                return Err(err);
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
}
