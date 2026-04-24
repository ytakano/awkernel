//! `JoinHandle` receives a return value of spawned a task.

#[cfg(feature = "baseline_trace")]
use crate::{baseline_trace, task};
use futures::channel::oneshot::{Canceled, Receiver};

pub struct JoinHandle<T> {
    #[cfg(feature = "baseline_trace")]
    child_trace_task_id: Option<u32>,
    rx: Receiver<T>,
}

impl<T> JoinHandle<T> {
    pub fn new(child_task_id: u32, rx: Receiver<T>) -> Self {
        #[cfg(not(feature = "baseline_trace"))]
        let _ = child_task_id;
        #[cfg(feature = "baseline_trace")]
        let child_trace_task_id = task::trace_task_id_of_runtime_task_id(child_task_id);

        Self {
            #[cfg(feature = "baseline_trace")]
            child_trace_task_id,
            rx,
        }
    }

    pub async fn join(self) -> Result<T, Canceled> {
        #[cfg(feature = "baseline_trace")]
        if let (Some(waiter_task_id), Some(child_task_id)) = (
            task::get_current_trace_task_id(awkernel_lib::cpu::cpu_id()),
            self.child_trace_task_id,
        ) {
            baseline_trace::record_task_trace(baseline_trace::TaskTraceEvent::JoinWait {
                waiter_task_id,
                child_task_id,
            });
        }

        self.rx.await
    }
}
