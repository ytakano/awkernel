//! Sleep a task.

use super::Cancel;
use crate::scheduler;
#[cfg(feature = "baseline_trace")]
use crate::{
    baseline_trace::{UnblockKind, WaitClass},
    task,
};
use alloc::sync::Arc;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::{task::Poll, time::Duration};
use futures::{future::FusedFuture, Future};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[must_use = "use `.await` to sleep"]
pub struct Sleep {
    state: Arc<Mutex<State>>,
    dur: Duration,
    #[cfg(feature = "baseline_trace")]
    trace_blocking: bool,
}

#[derive(Debug)]
pub enum State {
    Ready,
    Wait,
    Canceled,
    Finished,
}

impl Future for Sleep {
    type Output = State;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let mut node = MCSNode::new();
        let mut guard = self.state.lock(&mut node);

        match &*guard {
            State::Wait => Poll::Pending,
            State::Canceled => Poll::Ready(State::Canceled),
            State::Finished => Poll::Ready(State::Finished),
            State::Ready => {
                let state = self.state.clone();
                let waker = cx.waker().clone();

                *guard = State::Wait;

                #[cfg(feature = "baseline_trace")]
                let blocked_task_id = if self.trace_blocking {
                    task::record_current_task_block(WaitClass::Sleep)
                } else {
                    None
                };

                // Invoke `sleep_handler` after `self.dur` time.
                scheduler::sleep_task(
                    Box::new(move || {
                        let mut node = MCSNode::new();
                        let mut guard = state.lock(&mut node);
                        if let State::Wait = &*guard {
                            *guard = State::Finished;
                            #[cfg(feature = "baseline_trace")]
                            if let Some(task_id) = blocked_task_id {
                                task::record_task_unblock(
                                    task_id,
                                    WaitClass::Sleep,
                                    UnblockKind::Timeout,
                                );
                            }
                            waker.wake();
                        }
                    }),
                    self.dur,
                );

                Poll::Pending
            }
        }
    }
}

impl Cancel for Sleep {
    // Cancel sleep.
    fn cancel_unpin(&mut self) {
        let mut node = MCSNode::new();
        let mut guard = self.state.lock(&mut node);

        match &*guard {
            State::Ready | State::Wait => {
                *guard = State::Canceled;
            }
            _ => (),
        }
    }
}

impl Sleep {
    // Create a `Sleep`.
    pub(super) fn new(dur: Duration) -> Self {
        let state = Arc::new(Mutex::new(State::Ready));
        Self {
            state,
            dur,
            #[cfg(feature = "baseline_trace")]
            trace_blocking: true,
        }
    }

    pub(super) fn new_untraced(dur: Duration) -> Self {
        let state = Arc::new(Mutex::new(State::Ready));
        Self {
            state,
            dur,
            #[cfg(feature = "baseline_trace")]
            trace_blocking: false,
        }
    }
}

impl FusedFuture for Sleep {
    // Return true if the state is `Finished` or `Canceled`.
    fn is_terminated(&self) -> bool {
        let mut node = MCSNode::new();
        let guard = self.state.lock(&mut node);
        matches!(*guard, State::Finished | State::Canceled)
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        let mut node = MCSNode::new();
        let mut guard = self.state.lock(&mut node);
        if let State::Wait = &*guard {
            *guard = State::Canceled;
        }
    }
}
