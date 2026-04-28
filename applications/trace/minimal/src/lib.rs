#![no_std]

extern crate alloc;

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm",
    feature = "generic_trace_vm"
))]
use alloc::borrow::Cow;
#[cfg(feature = "generic_trace_vm")]
use alloc::{boxed::Box, vec::Vec};
#[cfg(any(
    feature = "single_async_trace_vm",
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "generic_trace_vm"
))]
use awkernel_async_lib::r#yield;
#[cfg(any(feature = "sleep_wakeup_trace_vm", feature = "generic_trace_vm"))]
use awkernel_async_lib::sleep;
use awkernel_async_lib::task::TaskResult;
#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm",
    feature = "generic_trace_vm"
))]
use awkernel_async_lib::{scheduler::SchedulerType, spawn};
#[cfg(any(feature = "sleep_wakeup_trace_vm", feature = "generic_trace_vm"))]
use core::time::Duration;
#[cfg(feature = "generic_trace_vm")]
use core::{future::Future, pin::Pin};

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm",
    feature = "generic_trace_vm"
))]
const TRACE_PRIORITY: u8 = 31;

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm",
    feature = "generic_trace_vm"
))]
fn trace_scheduler() -> SchedulerType {
    SchedulerType::PrioritizedFIFO(TRACE_PRIORITY)
}

#[cfg(any(
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm",
    feature = "generic_trace_vm"
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
pub async fn run_single_async() -> TaskResult {
    r#yield().await;
    Ok(())
}

#[cfg(feature = "nested_spawn_trace_vm")]
pub async fn run_nested_spawn() -> TaskResult {
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
pub async fn run_multi_async() -> TaskResult {
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
pub async fn run_sleep_wakeup() -> TaskResult {
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

#[cfg(feature = "generic_trace_vm")]
const DEFAULT_GENERIC_TRACE_SEED: u64 = 0x4d59_5df4_d0f3_3173;
#[cfg(feature = "generic_trace_vm")]
const MAX_TOTAL_TASKS: u8 = 8;
#[cfg(feature = "generic_trace_vm")]
const MAX_DEPTH: u8 = 3;
#[cfg(feature = "generic_trace_vm")]
const MAX_CHILDREN_PER_TASK: u8 = 2;
#[cfg(feature = "generic_trace_vm")]
const MAX_SLEEP_PER_TASK: u8 = 1;
#[cfg(feature = "generic_trace_vm")]
const MAX_YIELDS_PER_TASK: u8 = 2;

#[cfg(feature = "generic_trace_vm")]
type GenericWorkloadFuture = Pin<Box<dyn Future<Output = TaskResult> + Send>>;

#[cfg(feature = "generic_trace_vm")]
#[derive(Clone, Copy)]
struct GenericWorkloadState {
    seed: u64,
    depth: u8,
    remaining_descendants: u8,
}

#[cfg(feature = "generic_trace_vm")]
struct WorkloadRng {
    state: u64,
}

#[cfg(feature = "generic_trace_vm")]
impl WorkloadRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn below(&mut self, upper_exclusive: u8) -> u8 {
        if upper_exclusive <= 1 {
            0
        } else {
            (self.next_u64() % upper_exclusive as u64) as u8
        }
    }

    fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }
}

#[cfg(feature = "generic_trace_vm")]
fn parse_generic_trace_seed(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        raw.parse().ok()
    }
}

#[cfg(feature = "generic_trace_vm")]
fn generic_trace_seed() -> u64 {
    option_env!("GENERIC_TRACE_SEED")
        .and_then(parse_generic_trace_seed)
        .unwrap_or(DEFAULT_GENERIC_TRACE_SEED)
}

#[cfg(feature = "generic_trace_vm")]
async fn run_local_actions(rng: &mut WorkloadRng, force_sleep: bool) {
    let yield_count = rng.below(MAX_YIELDS_PER_TASK + 1);
    for _ in 0..yield_count {
        r#yield().await;
    }

    let sleep_count = if force_sleep || rng.bool() {
        MAX_SLEEP_PER_TASK
    } else {
        0
    };
    for _ in 0..sleep_count {
        let millis = 1 + rng.below(3) as u64;
        sleep(Duration::from_millis(millis)).await;
    }
}

#[cfg(feature = "generic_trace_vm")]
fn choose_child_count(state: GenericWorkloadState, rng: &mut WorkloadRng) -> u8 {
    if state.depth >= MAX_DEPTH || state.remaining_descendants == 0 {
        return 0;
    }

    let max_children = state.remaining_descendants.min(MAX_CHILDREN_PER_TASK);
    let min_children = if state.depth == 0 { 1 } else { 0 };
    min_children + rng.below(max_children - min_children + 1)
}

#[cfg(feature = "generic_trace_vm")]
fn generic_worker(state: GenericWorkloadState) -> GenericWorkloadFuture {
    Box::pin(async move {
        let mut rng = WorkloadRng::new(
            state.seed
                ^ ((state.depth as u64) << 48)
                ^ ((state.remaining_descendants as u64) << 32),
        );

        run_local_actions(&mut rng, state.depth == 0).await;

        let child_count = choose_child_count(state, &mut rng);
        let mut remaining_descendants = state.remaining_descendants - child_count;
        let mut children = Vec::new();

        for child_index in 0..child_count {
            let child_remaining = if state.depth + 1 < MAX_DEPTH && remaining_descendants > 0 {
                rng.below(remaining_descendants + 1)
            } else {
                0
            };
            remaining_descendants -= child_remaining;

            let child_state = GenericWorkloadState {
                seed: rng.next_u64() ^ ((child_index as u64) << 16),
                depth: state.depth + 1,
                remaining_descendants: child_remaining,
            };
            let child = spawn(
                "[Awkernel] generic_random trace worker".into(),
                generic_worker(child_state),
                trace_scheduler(),
            )
            .await;
            children.push(child);

            if rng.bool() {
                r#yield().await;
            }
        }

        run_local_actions(&mut rng, false).await;

        for child in children {
            child.join().await.map_err(|_| join_canceled())??;
        }

        Ok(())
    })
}

#[cfg(feature = "generic_trace_vm")]
pub async fn run_generic_random() -> TaskResult {
    generic_worker(GenericWorkloadState {
        seed: generic_trace_seed(),
        depth: 0,
        remaining_descendants: MAX_TOTAL_TASKS - 1,
    })
    .await
}
