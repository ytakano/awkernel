#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
use alloc::borrow::Cow;
#[cfg(any(
    feature = "single_async_trace_vm",
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm"
))]
use awkernel_async_lib::r#yield;
#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
use awkernel_async_lib::{scheduler::SchedulerType, spawn};
#[cfg(feature = "sleep_wakeup_trace_vm")]
use awkernel_async_lib::sleep;
use awkernel_async_lib::task::TaskResult;
#[cfg(feature = "sleep_wakeup_trace_vm")]
use core::time::Duration;

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
const TRACE_PRIORITY: u8 = 31;

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
fn trace_scheduler() -> SchedulerType {
    SchedulerType::PrioritizedFIFO(TRACE_PRIORITY)
}

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
fn join_canceled() -> Cow<'static, str> {
    Cow::Borrowed("workload trace child task was canceled")
}

#[cfg(any(feature = "nested_spawn_trace_vm", feature = "multi_async_trace_vm"))]
async fn yield_leaf() -> TaskResult {
    r#yield().await;
    Ok(())
}

#[cfg(feature = "single_async_trace_vm")]
pub(crate) async fn run_single_async() -> TaskResult {
    r#yield().await;
    Ok(())
}

#[cfg(feature = "nested_spawn_trace_vm")]
pub(crate) async fn run_nested_spawn() -> TaskResult {
    let child = spawn(
        "[Awkernel] nested_spawn trace child".into(),
        async {
            let grandchild = spawn(
                "[Awkernel] nested_spawn trace grandchild".into(),
                yield_leaf(),
                trace_scheduler(),
            )
            .await;

            grandchild.join().await.map_err(|_| join_canceled())?
        },
        trace_scheduler(),
    )
    .await;

    child.join().await.map_err(|_| join_canceled())?
}

#[cfg(feature = "multi_async_trace_vm")]
pub(crate) async fn run_multi_async() -> TaskResult {
    let child_a = spawn(
        "[Awkernel] multi_async trace child A".into(),
        yield_leaf(),
        trace_scheduler(),
    )
    .await;
    let child_b = spawn(
        "[Awkernel] multi_async trace child B".into(),
        yield_leaf(),
        trace_scheduler(),
    )
    .await;
    let child_c = spawn(
        "[Awkernel] multi_async trace child C".into(),
        yield_leaf(),
        trace_scheduler(),
    )
    .await;

    child_a.join().await.map_err(|_| join_canceled())??;
    child_b.join().await.map_err(|_| join_canceled())??;
    child_c.join().await.map_err(|_| join_canceled())?
}

#[cfg(feature = "sleep_wakeup_trace_vm")]
pub(crate) async fn run_sleep_wakeup() -> TaskResult {
    let child = spawn(
        "[Awkernel] sleep_wakeup trace child".into(),
        async {
            sleep(Duration::from_millis(1)).await;
            Ok(())
        },
        trace_scheduler(),
    )
    .await;

    child.join().await.map_err(|_| join_canceled())?
}
