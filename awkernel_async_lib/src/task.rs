//! Task structure and functions.
//!
//! - `Task` represents a task. This is handled as `Arc<Task>`.
//!     - `Task::wake()` and `Task::wake_by_ref()` call `Task::scheduler::wake_task()` to wake the task up.
//!     - `Task::info`, which type is `TaskInfo`, contains information of the task.
//! - `TaskInfo` represents information of task.
//! - `Tasks` is a set of tasks.

#[cfg(not(feature = "no_preempt"))]
mod preempt;

use crate::scheduler::{self, get_scheduler, pop_preemption_pending, Scheduler, SchedulerType};
use alloc::{
    borrow::Cow,
    collections::{btree_map, BTreeMap},
    sync::Arc,
};
use array_macro::array;
use awkernel_lib::{
    cpu::NUM_MAX_CPU,
    priority_queue::HIGHEST_PRIORITY,
    sync::mutex::{MCSNode, Mutex},
    unwind::catch_unwind,
};
#[cfg(target_pointer_width = "64")]
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

#[cfg(target_pointer_width = "32")]
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use core::task::{Context, Poll};
use futures::{
    future::{BoxFuture, Fuse, FusedFuture},
    task::{waker_ref, ArcWake},
    Future, FutureExt,
};

#[cfg(feature = "baseline_trace")]
use crate::baseline_trace::{
    self, BaselineTraceEvent, BaselineTraceSnapshot, SchedAndTaskDispatchTraceRecord,
};

#[cfg(feature = "baseline_trace")]
use crate::baseline_trace::{TaskTraceEvent, TaskTracePolicy, UnblockKind, WaitClass};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "no_preempt"))]
pub use preempt::{preemption, thread::deallocate_thread_pool, voluntary_preemption};

#[cfg(not(feature = "no_preempt"))]
use preempt::thread::PtrWorkerThreadContext;

/// Return type of futures taken by `awkernel_async_lib::task::spawn`.
pub type TaskResult = Result<(), Cow<'static, str>>;

static TASKS: Mutex<Tasks> = Mutex::new(Tasks::new()); // Set of tasks.
static RUNNING: [AtomicU32; NUM_MAX_CPU] = array![_ => AtomicU32::new(0); NUM_MAX_CPU]; // IDs of running tasks.
pub(crate) static MAX_TASK_PRIORITY: u64 = (1 << 56) - 1; // Maximum task priority.
#[cfg(target_pointer_width = "64")]
pub(crate) static NUM_TASK_IN_QUEUE: AtomicU64 = AtomicU64::new(0); // Number of tasks in the queue.

#[cfg(target_pointer_width = "32")]
pub(crate) static NUM_TASK_IN_QUEUE: AtomicU32 = AtomicU32::new(0); // Number of tasks in the queue.

static PREEMPTION_REQUEST: [AtomicBool; NUM_MAX_CPU] =
    array![_ => AtomicBool::new(false); NUM_MAX_CPU];

#[cfg(feature = "baseline_trace")]
fn trace_task_id_of_task(task: &Task) -> u32 {
    let mut node = MCSNode::new();
    let info = task.info.lock(&mut node);
    info.trace_task_id
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn record_current_task_join_target_ready() {
    let cpu_id = awkernel_lib::cpu::cpu_id();
    let runtime_task_id = match get_current_task(cpu_id) {
        Some(task_id) => task_id,
        None => return,
    };

    let task = {
        let mut node = MCSNode::new();
        let tasks = TASKS.lock(&mut node);
        tasks.id_to_task.get(&runtime_task_id).cloned()
    };

    let Some(task) = task else {
        return;
    };

    let trace_task_id = {
        let mut node = MCSNode::new();
        let info = task.info.lock(&mut node);
        info.trace_task_id
    };

    baseline_trace::record(
        BaselineTraceEvent::JoinTargetReady {
            task_id: trace_task_id,
        },
        baseline_snapshot(
            cpu_id,
            Some(trace_task_id),
            None,
            PREEMPTION_REQUEST[cpu_id].load(Ordering::Relaxed),
            None,
        ),
    );
    record_workload_task_trace(TaskTraceEvent::JoinTargetReady {
        task_id: trace_task_id,
    });
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn trace_task_id_of_runtime_task_id(runtime_task_id: u32) -> Option<u32> {
    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    tasks
        .id_to_task
        .get(&runtime_task_id)
        .map(|task| trace_task_id_of_task(task))
}

#[cfg(feature = "baseline_trace")]
fn baseline_current_task_id(cpu_id: usize) -> Option<u32> {
    let id = RUNNING[cpu_id].load(Ordering::Relaxed);
    (id != 0)
        .then_some(id)
        .and_then(trace_task_id_of_runtime_task_id)
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn get_current_trace_task_id(cpu_id: usize) -> Option<u32> {
    baseline_current_task_id(cpu_id)
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn record_current_task_block(wait_class: WaitClass) -> Option<u32> {
    let task_id = get_current_trace_task_id(awkernel_lib::cpu::cpu_id())?;
    record_workload_task_trace(TaskTraceEvent::Block {
        task_id,
        wait_class,
    });
    Some(task_id)
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn record_task_unblock(task_id: u32, wait_class: WaitClass, unblock_kind: UnblockKind) {
    record_workload_task_trace(TaskTraceEvent::Unblock {
        task_id,
        wait_class,
        unblock_kind,
    });
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn record_current_task_unblock(
    wait_class: WaitClass,
    unblock_kind: UnblockKind,
) -> Option<u32> {
    let task_id = get_current_trace_task_id(awkernel_lib::cpu::cpu_id())?;
    record_task_unblock(task_id, wait_class, unblock_kind);
    Some(task_id)
}

#[cfg(feature = "baseline_trace")]
fn baseline_runnable_ids(extra_runnable: Option<u32>) -> Vec<u32> {
    let mut runnable = Vec::new();

    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    for task in tasks.id_to_task.values() {
        let mut node = MCSNode::new();
        let info = task.info.lock(&mut node);
        if info.state == State::Runnable {
            runnable.push(info.trace_task_id);
        }
    }

    if let Some(task_id) = extra_runnable {
        if !runnable.iter().any(|&id| id == task_id) {
            runnable.push(task_id);
        }
    }

    runnable.sort_unstable();
    runnable
}

#[cfg(feature = "baseline_trace")]
fn baseline_choose_runnable_ids(chosen_task_id: u32) -> Vec<u32> {
    let mut runnable = baseline_runnable_ids(Some(chosen_task_id));

    runnable.retain(|&task_id| task_id != chosen_task_id);
    runnable.insert(0, chosen_task_id);
    runnable
}

#[cfg(feature = "baseline_trace")]
fn baseline_snapshot(
    cpu_id: usize,
    current: Option<u32>,
    extra_runnable: Option<u32>,
    need_resched: bool,
    dispatch_target: Option<u32>,
) -> BaselineTraceSnapshot {
    let num_cpu = awkernel_lib::cpu::num_cpu();
    let mut worker_current = Vec::new();
    let mut worker_need_resched = Vec::new();
    let mut worker_dispatch_target = Vec::new();

    for worker_cpu in 1..num_cpu {
        worker_current.push(if worker_cpu == cpu_id {
            current
        } else {
            baseline_current_task_id(worker_cpu)
        });
        worker_need_resched.push(if worker_cpu == cpu_id {
            need_resched
        } else {
            PREEMPTION_REQUEST[worker_cpu].load(Ordering::Relaxed)
        });
        worker_dispatch_target.push(if worker_cpu == cpu_id {
            dispatch_target
        } else {
            None
        });
    }

    BaselineTraceSnapshot {
        cpu_id,
        current,
        runnable: baseline_runnable_ids(extra_runnable),
        need_resched,
        dispatch_target,
        worker_current,
        worker_need_resched,
        worker_dispatch_target,
    }
}

#[cfg(feature = "baseline_trace")]
fn baseline_choose_snapshot(
    cpu_id: usize,
    task_id: u32,
    need_resched: bool,
) -> BaselineTraceSnapshot {
    let num_cpu = awkernel_lib::cpu::num_cpu();
    let mut worker_current = Vec::new();
    let mut worker_need_resched = Vec::new();
    let mut worker_dispatch_target = Vec::new();

    for worker_cpu in 1..num_cpu {
        worker_current.push(if worker_cpu == cpu_id {
            None
        } else {
            baseline_current_task_id(worker_cpu)
        });
        worker_need_resched.push(if worker_cpu == cpu_id {
            need_resched
        } else {
            PREEMPTION_REQUEST[worker_cpu].load(Ordering::Relaxed)
        });
        worker_dispatch_target.push(if worker_cpu == cpu_id {
            Some(task_id)
        } else {
            None
        });
    }

    BaselineTraceSnapshot {
        cpu_id,
        current: None,
        runnable: baseline_choose_runnable_ids(task_id),
        need_resched,
        dispatch_target: Some(task_id),
        worker_current,
        worker_need_resched,
        worker_dispatch_target,
    }
}

#[cfg(feature = "baseline_trace")]
fn record_baseline_runnable_projection(task_id: u32) {
    // Adapter-local projection: this records that TaskInfo became Runnable.
    // It is not a concrete scheduler-queue insertion observation.
    baseline_trace::record(
        BaselineTraceEvent::Wakeup { task_id },
        baseline_snapshot(0, baseline_current_task_id(0), Some(task_id), false, None),
    );
}

#[cfg(feature = "baseline_trace")]
pub(crate) struct BaselineDispatchProjection {
    dispatch: SchedAndTaskDispatchTraceRecord,
}

#[cfg(feature = "baseline_trace")]
pub(crate) fn capture_baseline_dispatch_projection(
    cpu_id: usize,
    runtime_task_id: u32,
) -> BaselineDispatchProjection {
    let task_id = trace_task_id_of_runtime_task_id(runtime_task_id).unwrap_or(runtime_task_id);
    let need_resched = PREEMPTION_REQUEST[cpu_id].load(Ordering::Relaxed);
    let dispatch = baseline_trace::capture_sched_and_task_dispatch(
        task_id,
        baseline_choose_snapshot(cpu_id, task_id, need_resched),
        baseline_snapshot(cpu_id, Some(task_id), None, false, None),
    );

    BaselineDispatchProjection { dispatch }
}

#[cfg(feature = "baseline_trace")]
fn emit_baseline_dispatch_projection(projection: BaselineDispatchProjection) {
    baseline_trace::emit_sched_and_task_dispatch(projection.dispatch);
}

#[cfg(feature = "baseline_trace")]
fn record_baseline_complete(cpu_id: usize, task_id: u32) {
    baseline_trace::record(
        BaselineTraceEvent::Complete { task_id },
        baseline_snapshot(cpu_id, None, None, true, None),
    );
}

#[cfg(feature = "baseline_trace")]
fn record_workload_task_trace(event: TaskTraceEvent) {
    baseline_trace::record_task_trace(event);
}

#[cfg(feature = "baseline_trace")]
fn record_baseline_stutter(cpu_id: usize) {
    baseline_trace::record(
        BaselineTraceEvent::Stutter,
        baseline_snapshot(
            cpu_id,
            baseline_current_task_id(cpu_id),
            None,
            PREEMPTION_REQUEST[cpu_id].load(Ordering::Relaxed),
            None,
        ),
    );
}

/// Task has ID, future, information, and a reference to a scheduler.
pub struct Task {
    pub id: u32,
    pub name: Cow<'static, str>,
    future: Mutex<Fuse<BoxFuture<'static, TaskResult>>>,
    pub info: Mutex<TaskInfo>,
    scheduler: &'static dyn Scheduler,
    pub priority: PriorityInfo,
}

impl Task {
    #[inline(always)]
    pub fn scheduler_name(&self) -> SchedulerType {
        self.scheduler.scheduler_name()
    }
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Task {}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Higher (larger) priority is greater.
        match self.priority.cmp(&other.priority) {
            core::cmp::Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

unsafe impl Sync for Task {}
unsafe impl Send for Task {}

impl ArcWake for Task {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let cloned = arc_self.clone();
        cloned.wake();
    }

    fn wake(self: Arc<Self>) {
        let panicked;

        {
            use State::*;

            let mut node = MCSNode::new();
            let mut info = self.info.lock(&mut node);

            match info.state {
                Running | Runnable | Preempted => {
                    info.need_sched = true;
                    return;
                }
                Terminated | Panicked => {
                    return;
                }
                Initialized | Waiting => {
                    info.state = Runnable;
                }
            }

            panicked = info.panicked;
        }

        NUM_TASK_IN_QUEUE.fetch_add(1, Ordering::Release);

        #[cfg(feature = "baseline_trace")]
        let trace_task_id = trace_task_id_of_task(&self);

        #[cfg(feature = "baseline_trace")]
        {
            // This lifecycle trace is emitted after the TaskInfo state becomes
            // Runnable and before scheduler::wake_task performs queue/preempt
            // handling. It projects release into the abstract runnable view.
            record_baseline_runnable_projection(trace_task_id);
            record_workload_task_trace(TaskTraceEvent::Runnable {
                task_id: trace_task_id,
            });
        }

        if panicked {
            scheduler::panicked::SCHEDULER.wake_task(self);
        } else {
            self.scheduler.wake_task(self);
        }

        // Notify the primary CPU to wake up workers.
        awkernel_lib::cpu::wake_cpu(0);
    }
}

/// Information of task.
pub struct TaskInfo {
    pub(crate) state: State,
    pub(crate) scheduler_type: SchedulerType,
    pub(crate) num_preempt: u64,
    last_executed_time: awkernel_lib::time::Time,
    absolute_deadline: Option<u64>,
    need_sched: bool,
    pub(crate) need_preemption: bool,
    panicked: bool,
    pub(crate) dag_info: Option<DagInfo>,
    #[cfg(feature = "baseline_trace")]
    trace_task_id: u32,
    #[cfg(feature = "baseline_trace")]
    completion_trace_recorded: bool,

    #[cfg(not(feature = "no_preempt"))]
    thread: Option<PtrWorkerThreadContext>,
}

impl TaskInfo {
    #[cfg(not(feature = "no_preempt"))]
    #[inline(always)]
    pub(crate) fn take_preempt_context(&mut self) -> Option<PtrWorkerThreadContext> {
        self.thread.take()
    }

    #[cfg(not(feature = "no_preempt"))]
    #[inline(always)]
    pub(crate) fn set_preempt_context(&mut self, ctx: PtrWorkerThreadContext) {
        assert!(self.thread.is_none());
        self.thread = Some(ctx)
    }

    #[inline(always)]
    pub fn get_state(&self) -> State {
        self.state
    }

    #[inline(always)]
    pub fn get_scheduler_type(&self) -> SchedulerType {
        if self.panicked {
            SchedulerType::Panicked
        } else {
            self.scheduler_type
        }
    }

    #[inline(always)]
    pub fn update_last_executed(&mut self) {
        self.last_executed_time = awkernel_lib::time::Time::now();
    }

    #[inline(always)]
    pub fn get_last_executed(&self) -> awkernel_lib::time::Time {
        self.last_executed_time
    }

    #[inline(always)]
    pub fn update_absolute_deadline(&mut self, deadline: u64) {
        self.absolute_deadline = Some(deadline);
    }

    #[inline(always)]
    pub fn get_absolute_deadline(&self) -> Option<u64> {
        self.absolute_deadline
    }

    #[inline(always)]
    pub fn get_num_preemption(&self) -> u64 {
        self.num_preempt
    }

    #[inline(always)]
    pub fn panicked(&self) -> bool {
        self.panicked
    }

    #[inline(always)]
    pub fn get_dag_info(&self) -> Option<DagInfo> {
        self.dag_info.clone()
    }
}

/// State of task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Initialized,
    Running,
    Runnable,
    Waiting,
    Preempted,
    Terminated,
    Panicked,
}

/// Tasks.
#[derive(Default)]
struct Tasks {
    candidate_id: u32, // Next candidate of task ID.
    id_to_task: BTreeMap<u32, Arc<Task>>,
}

#[derive(Clone)]
pub struct DagInfo {
    pub dag_id: u32,
    pub node_id: u32,
}

impl Tasks {
    const fn new() -> Self {
        Self {
            candidate_id: 1,
            id_to_task: BTreeMap::new(),
        }
    }

    fn spawn(
        &mut self,
        name: Cow<'static, str>,
        future: Fuse<BoxFuture<'static, TaskResult>>,
        scheduler: &'static dyn Scheduler,
        scheduler_type: SchedulerType,
        dag_info: Option<DagInfo>,
    ) -> SpawnedTask {
        let mut id = self.candidate_id;
        loop {
            if id == 0 {
                id += 1;
            }

            // Find an unused task ID.
            if let btree_map::Entry::Vacant(e) = self.id_to_task.entry(id) {
                #[cfg(feature = "baseline_trace")]
                let trace_task_id = baseline_trace::next_trace_task_id();

                let info = Mutex::new(TaskInfo {
                    scheduler_type,
                    state: State::Initialized,
                    num_preempt: 0,
                    last_executed_time: awkernel_lib::time::Time::now(),
                    absolute_deadline: None,
                    need_sched: false,
                    need_preemption: false,
                    panicked: false,
                    dag_info,
                    #[cfg(feature = "baseline_trace")]
                    trace_task_id,
                    #[cfg(feature = "baseline_trace")]
                    completion_trace_recorded: false,

                    #[cfg(not(feature = "no_preempt"))]
                    thread: None,
                });

                // Set the task priority.
                // If the scheduler implements dynamic priority scheduling, the task priority will be updated later.
                let task_priority = match scheduler_type {
                    SchedulerType::PrioritizedFIFO(priority)
                    | SchedulerType::PrioritizedRR(priority) => priority as u64,
                    _ => MAX_TASK_PRIORITY,
                };

                let task = Task {
                    name,
                    future: Mutex::new(future),
                    scheduler,
                    id,
                    info,
                    priority: PriorityInfo::new(scheduler.priority(), task_priority),
                };

                e.insert(Arc::new(task));
                self.candidate_id = id;

                return SpawnedTask {
                    runtime_task_id: id,
                    #[cfg(feature = "baseline_trace")]
                    trace_task_id,
                };
            } else {
                // The candidate task ID is already used.
                // Check next candidate.
                id += 1;
            }
        }
    }

    #[inline(always)]
    fn wake(&self, id: u32) {
        if let Some(task) = self.id_to_task.get(&id) {
            task.clone().wake();
        }
    }

    #[inline(always)]
    fn remove(&mut self, id: u32) {
        self.id_to_task.remove(&id);
    }
}

pub(crate) struct SpawnedTask {
    pub runtime_task_id: u32,
    #[cfg(feature = "baseline_trace")]
    pub trace_task_id: u32,
}

/// Spawn a detached task.
/// If you want to spawn tasks in non async functions,
/// use this function.
/// This function takes only futures that return `TaskResult`.
///
/// Use `awkernel_async_lib::spawn` in async functions instead of this.
/// `awkernel_async_lib::spawn` can take any future and joinable.
///
/// # Example
///
/// ```
/// use awkernel_async_lib::{scheduler::SchedulerType, task};
/// let task_id = task::spawn("example task".into(), async { Ok(()) }, SchedulerType::PrioritizedFIFO(0));
/// ```
pub fn spawn(
    name: Cow<'static, str>,
    future: impl Future<Output = TaskResult> + 'static + Send,
    sched_type: SchedulerType,
) -> u32 {
    spawn_with_ids(name, future, sched_type, None).runtime_task_id
}

/// Spawn a detached task with DAG information.
/// This function is similar to `spawn` but automatically sets DAG information
/// for the task, which is useful for DAG-based schedulers like GEDF.
///
/// # Example
///
/// ```ignore
/// use awkernel_async_lib::{scheduler::SchedulerType, task, dag::{create_dag, add_node_with_topic_edges_public, set_relative_deadline_public}};
/// use core::time::Duration;
/// let dag = create_dag();
/// let sink_node_idx = add_node_with_topic_edges_public(&dag, &[], &[]);
/// let deadline = Duration::from_millis(100);
/// set_relative_deadline_public(&dag, sink_node_idx, deadline);
/// let task_id = task::spawn_with_dag_info(
///     "dag task".into(),
///     async { Ok(()) },
///     SchedulerType::GEDF(0),
///     DagInfo { dag_id: 1, node_id: 0 }
/// );
/// ```
pub fn spawn_with_dag_info(
    name: Cow<'static, str>,
    future: impl Future<Output = TaskResult> + 'static + Send,
    sched_type: SchedulerType,
    dag_info: DagInfo,
) -> u32 {
    spawn_with_ids(name, future, sched_type, Some(dag_info)).runtime_task_id
}

pub fn inner_spawn(
    name: Cow<'static, str>,
    future: impl Future<Output = TaskResult> + 'static + Send,
    sched_type: SchedulerType,
    dag_info: Option<DagInfo>,
) -> u32 {
    spawn_with_ids(name, future, sched_type, dag_info).runtime_task_id
}

pub(crate) fn spawn_with_ids(
    name: Cow<'static, str>,
    future: impl Future<Output = TaskResult> + 'static + Send,
    sched_type: SchedulerType,
    dag_info: Option<DagInfo>,
) -> SpawnedTask {
    if let SchedulerType::PrioritizedFIFO(p) | SchedulerType::PrioritizedRR(p) = sched_type {
        if p > HIGHEST_PRIORITY {
            log::warn!(
                "Task priority should be between 0 and {HIGHEST_PRIORITY}. It is addressed as {HIGHEST_PRIORITY}."
            );
        }
    }

    let future = future.boxed();

    let scheduler = get_scheduler(sched_type);
    #[cfg(feature = "baseline_trace")]
    let parent_task_id = get_current_trace_task_id(awkernel_lib::cpu::cpu_id());

    let mut node = MCSNode::new();
    let mut tasks = TASKS.lock(&mut node);
    let spawned = tasks.spawn(name, future.fuse(), scheduler, sched_type, dag_info);
    let id = spawned.runtime_task_id;
    #[cfg(feature = "baseline_trace")]
    let child_trace_task_id = spawned.trace_task_id;
    let task = tasks.id_to_task.get(&id).cloned();
    drop(tasks);

    #[cfg(feature = "baseline_trace")]
    {
        record_workload_task_trace(TaskTraceEvent::Spawn {
            parent_task_id,
            child_task_id: child_trace_task_id,
            policy: TaskTracePolicy::from_scheduler_type(sched_type),
        });
    }

    if let Some(task) = task {
        task.wake();
    }

    spawned
}

/// Get the task ID currently running.
///
/// # Example
///
/// ```
/// if let Some(task_id) = awkernel_async_lib::task::get_current_task(1) { }
/// ```
#[inline(always)]
pub fn get_current_task(cpu_id: usize) -> Option<u32> {
    let id = RUNNING[cpu_id].load(Ordering::Relaxed);
    if id == 0 {
        None
    } else {
        Some(id)
    }
}

#[inline(always)]
pub fn set_current_task(cpu_id: usize, task_id: u32) {
    RUNNING[cpu_id].store(task_id, Ordering::Relaxed);
}

#[inline(always)]
fn get_next_task(execution_ensured: bool) -> Option<scheduler::ScheduledTask> {
    #[cfg(not(feature = "no_preempt"))]
    {
        if let Some(next) = preempt::get_next_task() {
            if execution_ensured {
                set_current_task(awkernel_lib::cpu::cpu_id(), next.id);
            }

            let scheduled = scheduler::ScheduledTask::new(next);

            #[cfg(feature = "baseline_trace")]
            {
                if execution_ensured {
                    let projection = capture_baseline_dispatch_projection(
                        awkernel_lib::cpu::cpu_id(),
                        scheduled.task.id,
                    );
                    return Some(scheduled.with_dispatch_projection(projection));
                }
            }

            return Some(scheduled);
        }
    }

    scheduler::get_next_task(execution_ensured)
}

#[cfg(feature = "perf")]
pub mod perf {
    use awkernel_lib::cpu::NUM_MAX_CPU;
    use core::ptr::{read_volatile, write_volatile};

    #[derive(Debug, Clone, PartialEq, Eq)]
    #[repr(u8)]
    pub enum PerfState {
        Boot = 0,
        Kernel,
        Task,
        ContextSwitch,
        Interrupt,
        Idle,
    }

    impl From<u8> for PerfState {
        fn from(value: u8) -> Self {
            match value {
                0 => Self::Boot,
                1 => Self::Kernel,
                2 => Self::Task,
                3 => Self::ContextSwitch,
                4 => Self::Interrupt,
                5 => Self::Idle,
                _ => panic!("From<u8> for PerfState::from: invalid value"),
            }
        }
    }

    static mut PERF_STATES: [u8; NUM_MAX_CPU] = [0; NUM_MAX_CPU];

    static mut START_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];

    static mut KERNEL_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut TASK_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut INTERRUPT_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut CONTEXT_SWITCH_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut IDLE_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut PERF_TIME: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];

    static mut KERNEL_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut TASK_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut INTERRUPT_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut CONTEXT_SWITCH_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut IDLE_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut PERF_WCET: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];

    static mut KERNEL_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut TASK_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut INTERRUPT_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut CONTEXT_SWITCH_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut IDLE_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];
    static mut PERF_COUNT: [u64; NUM_MAX_CPU] = [0; NUM_MAX_CPU];

    fn update_time_and_state(next_state: PerfState) {
        let end = awkernel_lib::delay::cpu_counter();
        let cpu_id = awkernel_lib::cpu::cpu_id();

        let state: PerfState = unsafe { read_volatile(&PERF_STATES[cpu_id]) }.into();
        if state == next_state {
            return;
        }

        let start = unsafe { read_volatile(&START_TIME[cpu_id]) };

        if start > 0 && start <= end {
            let diff = end - start;

            match state {
                PerfState::Kernel => unsafe {
                    let t = read_volatile(&KERNEL_TIME[cpu_id]);
                    write_volatile(&mut KERNEL_TIME[cpu_id], t + diff);
                    let c = read_volatile(&KERNEL_COUNT[cpu_id]);
                    write_volatile(&mut KERNEL_COUNT[cpu_id], c + 1);
                    let wcet = read_volatile(&KERNEL_WCET[cpu_id]);
                    write_volatile(&mut KERNEL_WCET[cpu_id], wcet.max(diff));
                },
                PerfState::Task => unsafe {
                    let t = read_volatile(&TASK_TIME[cpu_id]);
                    write_volatile(&mut TASK_TIME[cpu_id], t + diff);
                    let c = read_volatile(&TASK_COUNT[cpu_id]);
                    write_volatile(&mut TASK_COUNT[cpu_id], c + 1);
                    let wcet = read_volatile(&TASK_WCET[cpu_id]);
                    write_volatile(&mut TASK_WCET[cpu_id], wcet.max(diff));
                },
                PerfState::Interrupt => unsafe {
                    let t = read_volatile(&INTERRUPT_TIME[cpu_id]);
                    write_volatile(&mut INTERRUPT_TIME[cpu_id], t + diff);
                    let c = read_volatile(&INTERRUPT_COUNT[cpu_id]);
                    write_volatile(&mut INTERRUPT_COUNT[cpu_id], c + 1);
                    let wcet = read_volatile(&INTERRUPT_WCET[cpu_id]);
                    write_volatile(&mut INTERRUPT_WCET[cpu_id], wcet.max(diff));
                },
                PerfState::ContextSwitch => unsafe {
                    let t = read_volatile(&CONTEXT_SWITCH_TIME[cpu_id]);
                    write_volatile(&mut CONTEXT_SWITCH_TIME[cpu_id], t + diff);
                    let c = read_volatile(&CONTEXT_SWITCH_COUNT[cpu_id]);
                    write_volatile(&mut CONTEXT_SWITCH_COUNT[cpu_id], c + 1);
                    let wcet = read_volatile(&CONTEXT_SWITCH_WCET[cpu_id]);
                    write_volatile(&mut CONTEXT_SWITCH_WCET[cpu_id], wcet.max(diff));
                },
                PerfState::Idle => unsafe {
                    let t = read_volatile(&IDLE_TIME[cpu_id]);
                    write_volatile(&mut IDLE_TIME[cpu_id], t + diff);
                    let c = read_volatile(&IDLE_COUNT[cpu_id]);
                    write_volatile(&mut IDLE_COUNT[cpu_id], c + 1);
                    let wcet = read_volatile(&IDLE_WCET[cpu_id]);
                    write_volatile(&mut IDLE_WCET[cpu_id], wcet.max(diff));
                },
                PerfState::Boot => (),
            }
        }

        let cnt = awkernel_lib::delay::cpu_counter();

        unsafe {
            // Overhead of this.
            let t = read_volatile(&PERF_TIME[cpu_id]);
            write_volatile(&mut PERF_TIME[cpu_id], t + (cnt - end));
            let c = read_volatile(&PERF_COUNT[cpu_id]);
            write_volatile(&mut PERF_COUNT[cpu_id], c + 1);
            let wcet = read_volatile(&PERF_WCET[cpu_id]);
            write_volatile(&mut PERF_WCET[cpu_id], wcet.max(cnt - end));

            // State transition.
            write_volatile(&mut START_TIME[cpu_id], cnt);
            write_volatile(&mut PERF_STATES[cpu_id], next_state as u8);
        }
    }

    #[inline(always)]
    pub fn start_kernel() {
        update_time_and_state(PerfState::Kernel);
    }

    #[inline(always)]
    pub(crate) fn start_task() {
        update_time_and_state(PerfState::Task);
    }

    /// Return the previous state.
    #[inline(always)]
    pub fn start_interrupt() -> PerfState {
        let cpu_id = awkernel_lib::cpu::cpu_id();
        let previous: PerfState = unsafe { read_volatile(&PERF_STATES[cpu_id]) }.into();
        update_time_and_state(PerfState::Interrupt);
        previous
    }

    #[inline(always)]
    pub fn transition_to(next: PerfState) {
        match next {
            PerfState::Boot => unreachable!(),
            PerfState::Kernel => start_kernel(),
            PerfState::Task => start_task(),
            PerfState::ContextSwitch => start_context_switch(),
            PerfState::Interrupt => {
                start_interrupt();
            }
            PerfState::Idle => start_idle(),
        }
    }

    #[inline(always)]
    pub(crate) fn start_context_switch() {
        update_time_and_state(PerfState::ContextSwitch);
    }

    #[inline(always)]
    pub fn start_idle() {
        update_time_and_state(PerfState::Idle);
    }

    #[inline(always)]
    pub fn get_kernel_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&KERNEL_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_task_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&TASK_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_interrupt_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&INTERRUPT_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_context_switch_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&CONTEXT_SWITCH_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_idle_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&IDLE_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_perf_time(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&PERF_TIME[cpu_id]) }
    }

    #[inline(always)]
    pub fn get_ave_kernel_time(cpu_id: usize) -> Option<f64> {
        let total = get_kernel_time(cpu_id);
        let count = unsafe { read_volatile(&KERNEL_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_ave_task_time(cpu_id: usize) -> Option<f64> {
        let total = get_task_time(cpu_id);
        let count = unsafe { read_volatile(&TASK_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_ave_interrupt_time(cpu_id: usize) -> Option<f64> {
        let total = get_interrupt_time(cpu_id);
        let count = unsafe { read_volatile(&INTERRUPT_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_ave_context_switch_time(cpu_id: usize) -> Option<f64> {
        let total = get_context_switch_time(cpu_id);
        let count = unsafe { read_volatile(&CONTEXT_SWITCH_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_ave_idle_time(cpu_id: usize) -> Option<f64> {
        let total = get_idle_time(cpu_id);
        let count = unsafe { read_volatile(&IDLE_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_ave_perf_time(cpu_id: usize) -> Option<f64> {
        let total = get_perf_time(cpu_id);
        let count = unsafe { read_volatile(&PERF_COUNT[cpu_id]) };
        (count != 0).then_some((total as f64) / (count as f64))
    }

    #[inline(always)]
    pub fn get_kernel_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&KERNEL_WCET[cpu_id]) }
    }
    #[inline(always)]
    pub fn get_task_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&TASK_WCET[cpu_id]) }
    }
    #[inline(always)]
    pub fn get_idle_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&IDLE_WCET[cpu_id]) }
    }
    #[inline(always)]
    pub fn get_interrupt_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&INTERRUPT_WCET[cpu_id]) }
    }
    #[inline(always)]
    pub fn get_context_switch_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&CONTEXT_SWITCH_WCET[cpu_id]) }
    }
    #[inline(always)]
    pub fn get_perf_wcet(cpu_id: usize) -> u64 {
        unsafe { read_volatile(&PERF_WCET[cpu_id]) }
    }
}

pub fn run_main() {
    loop {
        #[cfg(feature = "perf")]
        perf::start_kernel();

        let cpu_id = awkernel_lib::cpu::cpu_id();
        if RUNNING[cpu_id].load(Ordering::Relaxed) == 0 {
            // Re-wake all preemption-pending tasks, because the preemption is no longer required.
            while let Some(p) = pop_preemption_pending(cpu_id) {
                p.scheduler.wake_task(p);
            }
        }

        if let Some(scheduled_task) = get_next_task(true) {
            #[cfg(feature = "baseline_trace")]
            let mut dispatch_projection = scheduled_task.dispatch_projection;
            let task = scheduled_task.task;

            PREEMPTION_REQUEST[cpu_id].store(false, Ordering::Relaxed);

            #[cfg(not(feature = "no_preempt"))]
            {
                // If the next task is a preempted task, then the current task will yield to the thread holding the next task.
                // After that, the current thread will be stored in the thread pool.
                let mut node = MCSNode::new();
                let mut info = task.info.lock(&mut node);

                if let Some(ctx) = info.take_preempt_context() {
                    info.update_last_executed();
                    drop(info);

                    #[cfg(feature = "baseline_trace")]
                    {
                        emit_baseline_dispatch_projection(
                            dispatch_projection
                                .take()
                                .expect("dispatch projection must be captured"),
                        );
                    }

                    #[cfg(feature = "perf")]
                    perf::start_context_switch();

                    unsafe { preempt::yield_and_pool(ctx) };

                    #[cfg(feature = "perf")]
                    perf::start_kernel();

                    continue;
                }
            }

            let w = waker_ref(&task);
            let mut ctx = Context::from_waker(&w);

            let result = {
                let cpu_id = awkernel_lib::cpu::cpu_id();
                let mut node = MCSNode::new();
                let Some(mut guard) = task.future.try_lock(&mut node) else {
                    // This task is running on another CPU,
                    // and re-schedule the task to avoid starvation just in case.
                    RUNNING[cpu_id].store(0, Ordering::Relaxed);
                    task.wake();
                    continue;
                };

                // Can remove this?
                if guard.is_terminated() {
                    RUNNING[cpu_id].store(0, Ordering::Relaxed);
                    continue;
                }

                {
                    let mut node = MCSNode::new();
                    let mut info = task.info.lock(&mut node);

                    if matches!(info.state, State::Terminated | State::Panicked) {
                        RUNNING[cpu_id].store(0, Ordering::Relaxed);
                        continue;
                    }

                    info.update_last_executed();
                }

                #[cfg(feature = "baseline_trace")]
                {
                    emit_baseline_dispatch_projection(
                        dispatch_projection
                            .take()
                            .expect("dispatch projection must be captured"),
                    );
                }

                // Use the primary memory allocator.
                #[cfg(not(feature = "std"))]
                unsafe {
                    awkernel_lib::heap::TALLOC.use_primary_cpu_id(cpu_id)
                };

                // This is unnecessary if the task is scheduled by PrioritizedFIFO. This remains for other schedulers.
                RUNNING[cpu_id].store(task.id, Ordering::Relaxed);

                // Invoke a task.
                catch_unwind(|| {
                    #[cfg(all(
                        any(target_arch = "aarch64", target_arch = "x86_64"),
                        not(feature = "std")
                    ))]
                    {
                        awkernel_lib::interrupt::enable();
                    }

                    #[cfg(feature = "perf")]
                    perf::start_task();

                    #[allow(clippy::let_and_return)]
                    let result = guard.poll_unpin(&mut ctx);

                    #[cfg(feature = "perf")]
                    perf::start_kernel();

                    #[cfg(all(
                        any(target_arch = "aarch64", target_arch = "x86_64"),
                        not(feature = "std")
                    ))]
                    {
                        awkernel_lib::interrupt::disable();
                    }

                    result
                })
            };

            let cpu_id = awkernel_lib::cpu::cpu_id();

            // If the primary memory allocator is available, it will be used.
            // If the primary memory allocator is exhausted, the backup allocator will be used.
            #[cfg(not(feature = "std"))]
            unsafe {
                awkernel_lib::heap::TALLOC.use_primary_then_backup_cpu_id(cpu_id)
            };

            let running_id = RUNNING[cpu_id].swap(0, Ordering::Relaxed);
            assert_eq!(running_id, task.id);

            let mut node = MCSNode::new();
            let mut info = task.info.lock(&mut node);

            match result {
                Ok(Poll::Pending) => {
                    // The task has not been terminated yet.
                    info.state = State::Waiting;

                    if info.need_sched {
                        info.need_sched = false;
                        drop(info);
                        task.clone().wake();
                    }
                }
                Ok(Poll::Ready(result)) => {
                    // The task has been terminated.

                    info.state = State::Terminated;
                    #[cfg(feature = "baseline_trace")]
                    let completion_trace = {
                        if info.completion_trace_recorded {
                            None
                        } else {
                            info.completion_trace_recorded = true;
                            Some(info.trace_task_id)
                        }
                    };
                    drop(info);

                    #[cfg(feature = "baseline_trace")]
                    if let Some(trace_task_id) = completion_trace {
                        record_baseline_complete(cpu_id, trace_task_id);
                        record_workload_task_trace(TaskTraceEvent::Complete {
                            task_id: trace_task_id,
                        });
                    }

                    #[cfg(all(
                        feature = "baseline_trace",
                        target_arch = "x86_64",
                        not(feature = "std")
                    ))]
                    if baseline_trace::take_dump_on_complete(task.id) {
                        baseline_trace::dump_to_console();
                        awkernel_lib::arch::x86_64::power::shutdown();
                    }

                    if let Err(msg) = result {
                        log::warn!("Task has been terminated but failed: {msg}");
                    }

                    let mut node = MCSNode::new();
                    let mut tasks = TASKS.lock(&mut node);

                    tasks.remove(task.id);
                }
                Err(_) => {
                    // Caught panic.
                    info.state = State::Panicked;
                    drop(info);

                    let mut node = MCSNode::new();
                    let mut tasks = TASKS.lock(&mut node);

                    tasks.remove(task.id);
                }
            }
        } else {
            #[cfg(feature = "perf")]
            perf::start_idle();

            #[cfg(feature = "baseline_trace")]
            record_baseline_stutter(cpu_id);

            awkernel_lib::cpu::sleep_cpu(None);
        }
    }
}

/// Execute runnable tasks.
///
/// # Safety
///
/// This function must be called from worker threads.
/// So, do not call this function in application code.
pub unsafe fn run() {
    #[cfg(not(feature = "std"))]
    preempt::init();

    run_main();
}

/// Wake `task_id` up.
#[inline(always)]
pub fn wake(task_id: u32) {
    let mut node = MCSNode::new();
    let gurad = TASKS.lock(&mut node);
    gurad.wake(task_id);
}

pub fn get_tasks() -> Vec<Arc<Task>> {
    let mut result = Vec::new();

    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    for (_, task) in tasks.id_to_task.iter() {
        result.push(task.clone());
    }

    result
}

#[derive(Debug)]
pub struct RunningTask {
    pub cpu_id: usize,
    pub task_id: u32,
}

pub fn get_tasks_running() -> Vec<RunningTask> {
    let mut tasks = Vec::new();
    let num_cpus = awkernel_lib::cpu::num_cpu();

    for (cpu_id, task) in RUNNING.iter().enumerate() {
        if cpu_id >= num_cpus {
            break;
        }

        let task_id = task.load(Ordering::Relaxed);
        tasks.push(RunningTask { cpu_id, task_id });
    }

    tasks
}

#[inline(always)]
pub fn get_num_preemption() -> usize {
    #[cfg(not(feature = "no_preempt"))]
    {
        preempt::get_num_preemption()
    }

    #[cfg(feature = "no_preempt")]
    {
        0
    }
}

#[inline(always)]
pub fn get_task(task_id: u32) -> Option<Arc<Task>> {
    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);
    tasks.id_to_task.get(&task_id).cloned()
}

#[inline(always)]
pub fn get_last_executed_by_task_id(task_id: u32) -> Option<awkernel_lib::time::Time> {
    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    tasks.id_to_task.get(&task_id).map(|task| {
        let mut node = MCSNode::new();
        let info = task.info.lock(&mut node);
        info.get_last_executed()
    })
}

#[inline(always)]
pub fn get_scheduler_type_by_task_id(task_id: u32) -> Option<SchedulerType> {
    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    tasks.id_to_task.get(&task_id).map(|task| {
        let mut node = MCSNode::new();
        let info = task.info.lock(&mut node);
        info.get_scheduler_type()
    })
}

#[inline(always)]
pub fn set_need_preemption(task_id: u32, cpu_id: usize) {
    let mut node = MCSNode::new();
    let tasks = TASKS.lock(&mut node);

    if let Some(task) = tasks.id_to_task.get(&task_id) {
        let mut node = MCSNode::new();
        let mut info = task.info.lock(&mut node);
        info.need_preemption = true;
    }

    PREEMPTION_REQUEST[cpu_id].store(true, Ordering::Release);
}

pub fn panicking() {
    let Some(task_id) = get_current_task(awkernel_lib::cpu::cpu_id()) else {
        return;
    };

    {
        let mut node = MCSNode::new();
        let tasks = TASKS.lock(&mut node);

        if let Some(task) = tasks.id_to_task.get(&task_id) {
            let mut node = MCSNode::new();
            let mut info = task.info.lock(&mut node);
            info.scheduler_type = SchedulerType::Panicked;
            info.panicked = true;
        } else {
            #[allow(clippy::needless_return)]
            return;
        }
    }

    #[cfg(not(feature = "no_preempt"))]
    unsafe {
        preempt::preemption();
    }
}

pub struct PriorityInfo {
    #[cfg(target_pointer_width = "64")]
    pub priority: AtomicU64,

    #[cfg(target_pointer_width = "32")]
    pub priority: AtomicU32,
}

impl PriorityInfo {
    fn new(scheduler_priority: u8, task_priority: u64) -> Self {
        PriorityInfo {
            #[cfg(target_pointer_width = "64")]
            priority: AtomicU64::new(Self::combine_priority(scheduler_priority, task_priority)),

            #[cfg(target_pointer_width = "32")]
            priority: AtomicU32::new(Self::combine_priority(scheduler_priority, task_priority)),
        }
    }

    #[cfg(target_pointer_width = "64")]
    pub fn update_priority_info(&self, scheduler_priority: u8, task_priority: u64) {
        self.priority.store(
            Self::combine_priority(scheduler_priority, task_priority),
            Ordering::Relaxed,
        );
    }

    #[cfg(target_pointer_width = "32")]
    pub fn update_priority_info(&self, scheduler_priority: u8, task_priority: u64) {
        self.priority.store(
            Self::combine_priority(scheduler_priority, task_priority),
            Ordering::Relaxed,
        );
    }

    #[cfg(target_pointer_width = "64")]
    fn combine_priority(scheduler_priority: u8, task_priority: u64) -> u64 {
        assert!(task_priority < (1 << 56), "Task priority exceeds 56 bits");
        ((scheduler_priority as u64) << 56) | (task_priority & ((1 << 56) - 1))
    }

    #[cfg(target_pointer_width = "32")]
    fn combine_priority(scheduler_priority: u8, task_priority: u64) -> u32 {
        let task_priority_32 = task_priority as u32;
        assert!(
            task_priority_32 < (1 << 24),
            "Task priority exceeds 24 bits for 32-bit"
        );
        ((scheduler_priority as u32) << 24) | (task_priority_32 & ((1 << 24) - 1))
    }
}

impl Clone for PriorityInfo {
    fn clone(&self) -> Self {
        let value = self.priority.load(Ordering::Relaxed);
        PriorityInfo {
            #[cfg(target_pointer_width = "64")]
            priority: AtomicU64::new(value),

            #[cfg(target_pointer_width = "32")]
            priority: AtomicU32::new(value),
        }
    }
}

impl PartialEq for PriorityInfo {
    fn eq(&self, other: &Self) -> bool {
        self.priority.load(Ordering::Relaxed) == other.priority.load(Ordering::Relaxed)
    }
}

impl Eq for PriorityInfo {}

impl PartialOrd for PriorityInfo {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityInfo {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.priority
            .load(Ordering::Relaxed)
            .cmp(&other.priority.load(Ordering::Relaxed))
    }
}

/// Wake workers up.
pub fn wake_workers() {
    let mut num_tasks = NUM_TASK_IN_QUEUE.load(Ordering::Relaxed);
    let num_cpu = awkernel_lib::cpu::num_cpu();

    for i in 1..num_cpu {
        if num_tasks == 0 {
            break;
        }

        if awkernel_lib::cpu::wake_cpu(i) {
            num_tasks -= 1;
        }
    }
}
