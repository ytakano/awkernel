//! `JoinHandle` receives a return value of spawned a task.

#[cfg(all(
    feature = "baseline_trace",
    any(
        feature = "single_async_trace_vm",
        feature = "nested_spawn_trace_vm",
        feature = "multi_async_trace_vm",
        feature = "sleep_wakeup_trace_vm"
    )
))]
use crate::{baseline_trace, task};
use futures::channel::oneshot::{Canceled, Receiver};

pub struct JoinHandle<T> {
    #[cfg(all(
        feature = "baseline_trace",
        any(
            feature = "single_async_trace_vm",
            feature = "nested_spawn_trace_vm",
            feature = "multi_async_trace_vm",
            feature = "sleep_wakeup_trace_vm"
        )
    ))]
    child_task_id: u32,
    rx: Receiver<T>,
}

impl<T> JoinHandle<T> {
    pub fn new(child_task_id: u32, rx: Receiver<T>) -> Self {
        #[cfg(not(all(
            feature = "baseline_trace",
            any(
                feature = "single_async_trace_vm",
                feature = "nested_spawn_trace_vm",
                feature = "multi_async_trace_vm",
                feature = "sleep_wakeup_trace_vm"
            )
        )))]
        let _ = child_task_id;

        Self {
            #[cfg(all(
                feature = "baseline_trace",
                any(
                    feature = "single_async_trace_vm",
                    feature = "nested_spawn_trace_vm",
                    feature = "multi_async_trace_vm",
                    feature = "sleep_wakeup_trace_vm"
                )
            ))]
            child_task_id,
            rx,
        }
    }

    pub async fn join(self) -> Result<T, Canceled> {
        #[cfg(all(
            feature = "baseline_trace",
            any(
                feature = "single_async_trace_vm",
                feature = "nested_spawn_trace_vm",
                feature = "multi_async_trace_vm",
                feature = "sleep_wakeup_trace_vm"
            )
        ))]
        if let Some(waiter_task_id) = task::get_current_task(awkernel_lib::cpu::cpu_id()) {
            baseline_trace::record_task_trace(baseline_trace::TaskTraceEvent::JoinWait {
                waiter_task_id,
                child_task_id: self.child_task_id,
            });
        }

        self.rx.await
    }
}
