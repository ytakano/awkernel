use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use array_macro::array;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use awkernel_lib::{cpu::NUM_MAX_CPU, delay::cpu_counter};
use core::sync::atomic::{AtomicBool, AtomicU64};
use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(not(feature = "std"))]
use awkernel_lib::console;

#[cfg(not(feature = "std"))]
const SERIAL_PREFIX: &str = "BASELINE_TRACE:";
#[cfg(not(feature = "std"))]
const SERIAL_DONE_MARKER: &str = "BASELINE_TRACE_DONE";
#[cfg(not(feature = "std"))]
const SCHED_TRACE_BEGIN_MARKER: &str = "BEGIN_SCHED_TRACE";
#[cfg(not(feature = "std"))]
const SCHED_TRACE_END_MARKER: &str = "END_SCHED_TRACE";
#[cfg(not(feature = "std"))]
const TASK_TRACE_BEGIN_MARKER: &str = "BEGIN_TASK_TRACE";
#[cfg(not(feature = "std"))]
const TASK_TRACE_END_MARKER: &str = "END_TASK_TRACE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineTraceEvent {
    Wakeup { task_id: u32 },
    RequestResched { cpu_id: usize },
    HandleResched { cpu_id: usize },
    Choose { task_id: u32 },
    Dispatch { task_id: u32 },
    Complete { task_id: u32 },
    JoinTargetReady { task_id: u32 },
    Stutter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineTraceSnapshot {
    pub cpu_id: usize,
    pub current: Option<u32>,
    pub runnable: Vec<u32>,
    pub need_resched: bool,
    pub dispatch_target: Option<u32>,
    pub worker_current: Vec<Option<u32>>,
    pub worker_need_resched: Vec<bool>,
    pub worker_dispatch_target: Vec<Option<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BaselineTraceRecord {
    event_id: u64,
    tsc: u64,
    event: BaselineTraceEvent,
    snapshot: BaselineTraceSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskTraceEvent {
    Spawn {
        parent_task_id: Option<u32>,
        child_task_id: u32,
    },
    Runnable {
        task_id: u32,
    },
    Choose {
        task_id: u32,
    },
    Dispatch {
        task_id: u32,
    },
    Sleep {
        task_id: u32,
    },
    JoinWait {
        waiter_task_id: u32,
        child_task_id: u32,
    },
    JoinTargetReady {
        task_id: u32,
    },
    Complete {
        task_id: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskTraceRecord {
    pub event_id: u64,
    pub event: TaskTraceEvent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedAndTaskDispatchTraceRecord {
    sched_choose: BaselineTraceRecord,
    sched_dispatch: BaselineTraceRecord,
    task_choose: Option<TaskTraceRecord>,
    task_dispatch: Option<TaskTraceRecord>,
}

const TRACE_CAPACITY: usize = 4096;
const LIFECYCLE_TRACE_CAPACITY: usize = 512;

struct TraceBuffer {
    records: Vec<BaselineTraceRecord>,
    overflowed: bool,
}

impl TraceBuffer {
    const fn new() -> Self {
        Self {
            records: Vec::new(),
            overflowed: false,
        }
    }

    fn reset(&mut self) {
        self.records.clear();
        if self.records.capacity() < TRACE_CAPACITY {
            self.records
                .reserve(TRACE_CAPACITY - self.records.capacity());
        }
        self.overflowed = false;
    }

    fn push(&mut self, record: BaselineTraceRecord) {
        if self.records.len() >= TRACE_CAPACITY {
            self.overflowed = true;
            return;
        }

        self.records.push(record);
    }
}

struct TaskTraceBuffer {
    records: Vec<TaskTraceRecord>,
    overflowed: bool,
}

impl TaskTraceBuffer {
    const fn new() -> Self {
        Self {
            records: Vec::new(),
            overflowed: false,
        }
    }

    fn reset(&mut self) {
        self.records.clear();
        if self.records.capacity() < LIFECYCLE_TRACE_CAPACITY {
            self.records
                .reserve(LIFECYCLE_TRACE_CAPACITY - self.records.capacity());
        }
        self.overflowed = false;
    }

    fn push(&mut self, record: TaskTraceRecord) {
        if self.records.len() >= LIFECYCLE_TRACE_CAPACITY {
            self.overflowed = true;
            return;
        }

        self.records.push(record);
    }
}

static BASELINE_TRACE: [Mutex<TraceBuffer>; NUM_MAX_CPU] =
    array![_ => Mutex::new(TraceBuffer::new()); NUM_MAX_CPU];
static TASK_TRACE: Mutex<TaskTraceBuffer> = Mutex::new(TaskTraceBuffer::new());
static TRACE_EVENT_ID: AtomicU64 = AtomicU64::new(0);
static TRACE_TASK_ID: AtomicU32 = AtomicU32::new(1);
static WORKLOAD_ARTIFACT_ENABLED: AtomicBool = AtomicBool::new(false);
#[cfg(not(feature = "std"))]
static DUMP_ON_COMPLETE_TASK_ID: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn reset() {
    TRACE_EVENT_ID.store(0, core::sync::atomic::Ordering::Release);
    TRACE_TASK_ID.store(1, Ordering::Release);
    WORKLOAD_ARTIFACT_ENABLED.store(false, Ordering::Release);
    for trace in BASELINE_TRACE.iter() {
        let mut node = MCSNode::new();
        let mut trace = trace.lock(&mut node);
        trace.reset();
    }
    let mut node = MCSNode::new();
    let mut task_trace = TASK_TRACE.lock(&mut node);
    task_trace.reset();
}

#[inline(always)]
pub fn set_workload_artifact_enabled(enabled: bool) {
    WORKLOAD_ARTIFACT_ENABLED.store(enabled, Ordering::Release);
}

#[inline(always)]
pub fn enable_workload_trace_artifacts() {
    set_workload_artifact_enabled(true);
}

#[inline(always)]
fn workload_artifact_enabled() -> bool {
    WORKLOAD_ARTIFACT_ENABLED.load(Ordering::Acquire)
}

#[inline(always)]
pub(crate) fn next_trace_task_id() -> u32 {
    TRACE_TASK_ID.fetch_add(1, Ordering::Relaxed)
}

#[inline(always)]
fn next_event_id() -> u64 {
    TRACE_EVENT_ID.fetch_add(1, core::sync::atomic::Ordering::AcqRel)
}

#[inline(always)]
fn next_event_id_block(width: u64) -> u64 {
    TRACE_EVENT_ID.fetch_add(width, core::sync::atomic::Ordering::AcqRel)
}

#[inline(always)]
fn capture_record(
    event: BaselineTraceEvent,
    snapshot: BaselineTraceSnapshot,
) -> BaselineTraceRecord {
    let event_id = next_event_id();
    let tsc = cpu_counter();
    BaselineTraceRecord {
        event_id,
        tsc,
        event,
        snapshot,
    }
}

#[inline(always)]
fn capture_record_with_event_id(
    event_id: u64,
    event: BaselineTraceEvent,
    snapshot: BaselineTraceSnapshot,
) -> BaselineTraceRecord {
    let tsc = cpu_counter();
    BaselineTraceRecord {
        event_id,
        tsc,
        event,
        snapshot,
    }
}

#[inline(always)]
fn emit_record(record: BaselineTraceRecord) {
    let cpu_id = record.snapshot.cpu_id;
    let mut node = MCSNode::new();
    let mut trace = BASELINE_TRACE[cpu_id].lock(&mut node);
    trace.push(record);
}

#[inline(always)]
pub fn record(event: BaselineTraceEvent, snapshot: BaselineTraceSnapshot) {
    let record = capture_record(event, snapshot);
    emit_record(record);
}

#[inline(always)]
fn capture_task_record_with_event_id(
    event_id: u64,
    event: TaskTraceEvent,
) -> Option<TaskTraceRecord> {
    if workload_artifact_enabled() {
        Some(TaskTraceRecord { event_id, event })
    } else {
        None
    }
}

#[inline(always)]
fn capture_task_record(event: TaskTraceEvent) -> Option<TaskTraceRecord> {
    if workload_artifact_enabled() {
        Some(TaskTraceRecord {
            event_id: next_event_id(),
            event,
        })
    } else {
        None
    }
}

#[inline(always)]
pub(crate) fn capture_sched_and_task_dispatch(
    task_id: u32,
    choose_snapshot: BaselineTraceSnapshot,
    dispatch_snapshot: BaselineTraceSnapshot,
) -> SchedAndTaskDispatchTraceRecord {
    let choose_event_id = next_event_id_block(2);
    let dispatch_event_id = choose_event_id + 1;
    let sched_choose = capture_record_with_event_id(
        choose_event_id,
        BaselineTraceEvent::Choose { task_id },
        choose_snapshot,
    );
    let sched_dispatch = capture_record_with_event_id(
        dispatch_event_id,
        BaselineTraceEvent::Dispatch { task_id },
        dispatch_snapshot,
    );
    let task_choose =
        capture_task_record_with_event_id(choose_event_id, TaskTraceEvent::Choose { task_id });
    let task_dispatch =
        capture_task_record_with_event_id(dispatch_event_id, TaskTraceEvent::Dispatch { task_id });

    SchedAndTaskDispatchTraceRecord {
        sched_choose,
        sched_dispatch,
        task_choose,
        task_dispatch,
    }
}

#[inline(always)]
pub(crate) fn emit_sched_and_task_dispatch(record: SchedAndTaskDispatchTraceRecord) {
    emit_record(record.sched_choose);
    emit_record(record.sched_dispatch);
    emit_task_record(record.task_choose);
    emit_task_record(record.task_dispatch);
}

#[inline(always)]
fn emit_task_record(record: Option<TaskTraceRecord>) {
    if let Some(record) = record {
        let mut node = MCSNode::new();
        let mut trace = TASK_TRACE.lock(&mut node);
        trace.push(record);
    }
}

#[inline(always)]
pub fn record_task_trace(event: TaskTraceEvent) {
    let record = capture_task_record(event);
    emit_task_record(record);
}

fn merge_records(mut records: Vec<BaselineTraceRecord>) -> Vec<BaselineTraceRecord> {
    records.sort_by(|lhs, rhs| lhs.event_id.cmp(&rhs.event_id));
    records
}

#[inline(always)]
fn records() -> Vec<BaselineTraceRecord> {
    let mut merged = Vec::new();

    for trace in BASELINE_TRACE.iter() {
        let mut node = MCSNode::new();
        let trace = trace.lock(&mut node);
        merged.extend(trace.records.iter().cloned());
    }

    merge_records(merged)
}

#[inline(always)]
fn overflowed() -> bool {
    let row_overflow = BASELINE_TRACE.iter().any(|trace| {
        let mut node = MCSNode::new();
        let trace = trace.lock(&mut node);
        trace.overflowed
    });
    let mut node = MCSNode::new();
    let task_trace = TASK_TRACE.lock(&mut node);
    row_overflow || task_trace.overflowed
}

fn merge_task_trace_records(mut records: Vec<TaskTraceRecord>) -> Vec<TaskTraceRecord> {
    records.sort_by(|lhs, rhs| lhs.event_id.cmp(&rhs.event_id));
    records
}

#[inline(always)]
fn task_trace_records() -> Vec<TaskTraceRecord> {
    let mut node = MCSNode::new();
    let trace = TASK_TRACE.lock(&mut node);
    merge_task_trace_records(trace.records.clone())
}

fn render_lines() -> Vec<String> {
    records()
        .into_iter()
        .map(|record| {
            format!(
                "cpu={} event={} current={:?} runnable={:?} need_resched={} dispatch_target={:?}",
                record.snapshot.cpu_id,
                event_name(record.event),
                record.snapshot.current,
                record.snapshot.runnable,
                record.snapshot.need_resched,
                record.snapshot.dispatch_target
            )
        })
        .collect()
}

fn render_trace_rows_event(event: BaselineTraceEvent) -> (&'static str, Option<u32>, Option<u32>) {
    match event {
        BaselineTraceEvent::Wakeup { task_id } => ("Wakeup", Some(task_id), None),
        BaselineTraceEvent::RequestResched { cpu_id } => {
            ("RequestResched", Some(cpu_id as u32), None)
        }
        BaselineTraceEvent::HandleResched { cpu_id } => {
            ("HandleResched", Some(cpu_id as u32), None)
        }
        BaselineTraceEvent::Choose { task_id } => ("Choose", Some(1), Some(task_id)),
        BaselineTraceEvent::Dispatch { task_id } => ("Dispatch", Some(1), Some(task_id)),
        BaselineTraceEvent::Complete { task_id } => ("Complete", Some(task_id), None),
        BaselineTraceEvent::JoinTargetReady { task_id } => ("JoinTargetReady", Some(task_id), None),
        BaselineTraceEvent::Stutter => ("Stutter", None, None),
    }
}

fn render_trace_rows_option(value: Option<u32>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}

fn render_trace_rows_option_list(values: &[Option<u32>]) -> String {
    values
        .iter()
        .map(|value| render_trace_rows_option(*value))
        .collect::<Vec<_>>()
        .join(",")
}

fn render_trace_rows_bool_list(values: &[bool]) -> String {
    values
        .iter()
        .map(|value| {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn render_trace_rows_runnable(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn render_trace_rows_record(record: BaselineTraceRecord) -> String {
    let (event_tag, arg0, arg1) = render_trace_rows_event(record.event);
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        record.snapshot.cpu_id,
        event_tag,
        render_trace_rows_option(arg0),
        render_trace_rows_option(arg1),
        render_trace_rows_option(record.snapshot.current),
        render_trace_rows_runnable(&record.snapshot.runnable),
        if record.snapshot.need_resched {
            "true"
        } else {
            "false"
        },
        render_trace_rows_option(record.snapshot.dispatch_target),
        render_trace_rows_option_list(&record.snapshot.worker_current),
        render_trace_rows_bool_list(&record.snapshot.worker_need_resched),
        render_trace_rows_option_list(&record.snapshot.worker_dispatch_target)
    )
}

pub fn render_trace_rows_artifact_lines() -> Vec<String> {
    records()
        .into_iter()
        .map(render_trace_rows_record)
        .collect()
}

fn render_task_trace_kind(event: TaskTraceEvent) -> (&'static str, u32, Option<u32>) {
    match event {
        TaskTraceEvent::Spawn {
            parent_task_id,
            child_task_id,
        } => ("Spawn", child_task_id, parent_task_id),
        TaskTraceEvent::Runnable { task_id } => ("Runnable", task_id, None),
        TaskTraceEvent::Choose { task_id } => ("Choose", task_id, None),
        TaskTraceEvent::Dispatch { task_id } => ("Dispatch", task_id, None),
        TaskTraceEvent::Sleep { task_id } => ("Sleep", task_id, None),
        TaskTraceEvent::JoinWait {
            waiter_task_id,
            child_task_id,
        } => ("JoinWait", waiter_task_id, Some(child_task_id)),
        TaskTraceEvent::JoinTargetReady { task_id } => ("JoinTargetReady", task_id, None),
        TaskTraceEvent::Complete { task_id } => ("Complete", task_id, None),
    }
}

fn render_task_trace_record(record: TaskTraceRecord) -> String {
    let (kind, subject, related) = render_task_trace_kind(record.event);
    format!(
        "{}\t{}\t{}",
        kind,
        subject,
        render_trace_rows_option(related)
    )
}

fn render_task_trace_artifact_lines() -> Vec<String> {
    task_trace_records()
        .into_iter()
        .map(render_task_trace_record)
        .collect()
}

#[cfg(not(feature = "std"))]
pub fn arm_dump_on_complete(task_id: u32) {
    DUMP_ON_COMPLETE_TASK_ID.store(task_id, Ordering::Release);
}

#[cfg(not(feature = "std"))]
pub fn take_dump_on_complete(task_id: u32) -> bool {
    DUMP_ON_COMPLETE_TASK_ID
        .compare_exchange(task_id, 0, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(not(feature = "std"))]
pub fn dump_to_console() {
    if overflowed() {
        console::print("BASELINE_TRACE_OVERFLOW\r\n");
    }
    for line in render_lines() {
        console::print(&format!("{SERIAL_PREFIX} {line}\r\n"));
    }
    console::print(&format!("{SERIAL_DONE_MARKER}\r\n"));
    if workload_artifact_enabled() {
        console::print(&format!("{SCHED_TRACE_BEGIN_MARKER}\r\n"));
        for line in render_trace_rows_artifact_lines() {
            console::print(&format!("{line}\r\n"));
        }
        console::print(&format!("{SCHED_TRACE_END_MARKER}\r\n"));
        console::print(&format!("{TASK_TRACE_BEGIN_MARKER}\r\n"));
        for line in render_task_trace_artifact_lines() {
            console::print(&format!("{line}\r\n"));
        }
        console::print(&format!("{TASK_TRACE_END_MARKER}\r\n"));
    }
}

fn event_name(event: BaselineTraceEvent) -> &'static str {
    match event {
        BaselineTraceEvent::Wakeup { .. } => "EvWakeup",
        BaselineTraceEvent::RequestResched { .. } => "EvRequestResched",
        BaselineTraceEvent::HandleResched { .. } => "EvHandleResched",
        BaselineTraceEvent::Choose { .. } => "EvChoose",
        BaselineTraceEvent::Dispatch { .. } => "EvDispatch",
        BaselineTraceEvent::Complete { .. } => "EvComplete",
        BaselineTraceEvent::JoinTargetReady { .. } => "EvJoinTargetReady",
        BaselineTraceEvent::Stutter => "EvStutter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_snapshot(
        cpu_id: usize,
        current: Option<u32>,
        runnable: Vec<u32>,
        need_resched: bool,
        dispatch_target: Option<u32>,
    ) -> BaselineTraceSnapshot {
        BaselineTraceSnapshot {
            cpu_id,
            current,
            runnable,
            need_resched,
            dispatch_target,
            worker_current: vec![current],
            worker_need_resched: vec![need_resched],
            worker_dispatch_target: vec![dispatch_target],
        }
    }

    #[test]
    fn records_multiple_cpus_in_baseline_trace() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        record(
            BaselineTraceEvent::Wakeup { task_id: 1 },
            BaselineTraceSnapshot {
                cpu_id: 0,
                current: None,
                runnable: vec![1],
                need_resched: false,
                dispatch_target: None,
                worker_current: vec![None],
                worker_need_resched: vec![false],
                worker_dispatch_target: vec![None],
            },
        );
        record(
            BaselineTraceEvent::Stutter,
            BaselineTraceSnapshot {
                cpu_id: 1,
                current: None,
                runnable: vec![],
                need_resched: false,
                dispatch_target: None,
                worker_current: vec![None],
                worker_need_resched: vec![false],
                worker_dispatch_target: vec![None],
            },
        );

        let records = records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].event, BaselineTraceEvent::Wakeup { task_id: 1 });
        assert_eq!(records[1].event, BaselineTraceEvent::Stutter);
    }

    #[test]
    fn renders_line_oriented_trace() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        record(
            BaselineTraceEvent::Dispatch { task_id: 7 },
            BaselineTraceSnapshot {
                cpu_id: 1,
                current: Some(7),
                runnable: vec![],
                need_resched: false,
                dispatch_target: None,
                worker_current: vec![Some(7)],
                worker_need_resched: vec![false],
                worker_dispatch_target: vec![None],
            },
        );

        let lines = render_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("cpu=1"));
        assert!(lines[0].contains("event=EvDispatch"));
        assert!(lines[0].contains("current=Some(7)"));
    }

    #[test]
    fn renders_trace_rows_artifact() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        record(
            BaselineTraceEvent::Choose { task_id: 1 },
            BaselineTraceSnapshot {
                cpu_id: 1,
                current: None,
                runnable: vec![1],
                need_resched: true,
                dispatch_target: Some(1),
                worker_current: vec![None],
                worker_need_resched: vec![true],
                worker_dispatch_target: vec![Some(1)],
            },
        );
        record(
            BaselineTraceEvent::Dispatch { task_id: 1 },
            BaselineTraceSnapshot {
                cpu_id: 1,
                current: Some(1),
                runnable: vec![],
                need_resched: false,
                dispatch_target: None,
                worker_current: vec![Some(1)],
                worker_need_resched: vec![false],
                worker_dispatch_target: vec![None],
            },
        );

        let lines = render_trace_rows_artifact_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "1\tChoose\t1\t1\t-\t1\ttrue\t1\t-\ttrue\t1");
        assert_eq!(lines[1], "1\tDispatch\t1\t1\t1\t\tfalse\t-\t1\tfalse\t-");
    }

    #[test]
    fn dispatch_capture_reserves_contiguous_sched_event_ids() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();

        let pending = capture_sched_and_task_dispatch(
            7,
            test_snapshot(0, None, vec![7], true, Some(7)),
            test_snapshot(0, Some(7), vec![], false, None),
        );

        assert_eq!(
            pending.sched_choose.event_id + 1,
            pending.sched_dispatch.event_id
        );
        assert_eq!(
            pending.sched_choose.event,
            BaselineTraceEvent::Choose { task_id: 7 }
        );
        assert_eq!(
            pending.sched_dispatch.event,
            BaselineTraceEvent::Dispatch { task_id: 7 }
        );
        assert!(records().is_empty());

        emit_sched_and_task_dispatch(pending);

        let records = records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].event_id, 0);
        assert_eq!(records[1].event_id, 1);

        assert!(task_trace_records().is_empty());
    }

    #[test]
    fn delayed_dispatch_emit_keeps_sched_trace_order_by_reserved_event_id() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();

        let pending = capture_sched_and_task_dispatch(
            7,
            test_snapshot(0, None, vec![7], true, Some(7)),
            test_snapshot(0, Some(7), vec![], false, None),
        );
        record(
            BaselineTraceEvent::Complete { task_id: 42 },
            test_snapshot(1, None, vec![], true, None),
        );
        emit_sched_and_task_dispatch(pending);

        let lines = render_trace_rows_artifact_lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "0\tChoose\t1\t7\t-\t7\ttrue\t7\t-\ttrue\t7");
        assert_eq!(lines[1], "0\tDispatch\t1\t7\t7\t\tfalse\t-\t7\tfalse\t-");
        assert_eq!(lines[2], "1\tComplete\t42\t-\t-\t\ttrue\t-\t-\ttrue\t-");
    }

    #[test]
    fn dispatch_capture_pairs_sched_and_task_event_ids() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        set_workload_artifact_enabled(true);

        let pending = capture_sched_and_task_dispatch(
            7,
            test_snapshot(0, None, vec![7], true, Some(7)),
            test_snapshot(0, Some(7), vec![], false, None),
        );
        let task_choose = pending
            .task_choose
            .as_ref()
            .expect("task Choose trace must be present");
        let task_dispatch = pending
            .task_dispatch
            .as_ref()
            .expect("task Dispatch trace must be present");

        assert_eq!(pending.sched_choose.event_id, task_choose.event_id);
        assert_eq!(pending.sched_dispatch.event_id, task_dispatch.event_id);
        assert_eq!(
            pending.sched_choose.event_id + 1,
            pending.sched_dispatch.event_id
        );
    }

    #[test]
    fn delayed_dispatch_emit_keeps_task_trace_order_by_reserved_event_id() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        set_workload_artifact_enabled(true);

        let pending = capture_sched_and_task_dispatch(
            7,
            test_snapshot(0, None, vec![7], true, Some(7)),
            test_snapshot(0, Some(7), vec![], false, None),
        );
        record_task_trace(TaskTraceEvent::Runnable { task_id: 42 });
        emit_sched_and_task_dispatch(pending);

        let lines = render_task_trace_artifact_lines();
        assert_eq!(
            lines,
            vec!["Choose\t7\t-", "Dispatch\t7\t-", "Runnable\t42\t-"]
        );
    }

    #[test]
    fn join_target_ready_renders_task_trace_row() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        set_workload_artifact_enabled(true);

        record_task_trace(TaskTraceEvent::JoinTargetReady { task_id: 42 });

        let lines = render_task_trace_artifact_lines();
        assert_eq!(lines, vec!["JoinTargetReady\t42\t-"]);
    }

    #[test]
    fn merge_orders_by_event_id() {
        let merged = merge_records(vec![
            BaselineTraceRecord {
                event_id: 2,
                tsc: 12,
                event: BaselineTraceEvent::Dispatch { task_id: 2 },
                snapshot: BaselineTraceSnapshot {
                    cpu_id: 1,
                    current: Some(2),
                    runnable: vec![],
                    need_resched: false,
                    dispatch_target: None,
                    worker_current: vec![Some(2)],
                    worker_need_resched: vec![false],
                    worker_dispatch_target: vec![None],
                },
            },
            BaselineTraceRecord {
                event_id: 0,
                tsc: 10,
                event: BaselineTraceEvent::Wakeup { task_id: 1 },
                snapshot: BaselineTraceSnapshot {
                    cpu_id: 0,
                    current: None,
                    runnable: vec![1],
                    need_resched: false,
                    dispatch_target: None,
                    worker_current: vec![None],
                    worker_need_resched: vec![false],
                    worker_dispatch_target: vec![None],
                },
            },
            BaselineTraceRecord {
                event_id: 1,
                tsc: 10,
                event: BaselineTraceEvent::HandleResched { cpu_id: 1 },
                snapshot: BaselineTraceSnapshot {
                    cpu_id: 1,
                    current: None,
                    runnable: vec![1],
                    need_resched: true,
                    dispatch_target: None,
                    worker_current: vec![None],
                    worker_need_resched: vec![true],
                    worker_dispatch_target: vec![None],
                },
            },
        ]);

        assert_eq!(merged[0].event_id, 0);
        assert_eq!(merged[1].event_id, 1);
        assert_eq!(merged[2].event_id, 2);
        assert_eq!(merged[0].snapshot.cpu_id, 0);
        assert_eq!(merged[1].snapshot.cpu_id, 1);
        assert_eq!(merged[2].tsc, 12);
    }

    #[test]
    fn marks_overflow_once_capacity_is_exceeded() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();

        for task_id in 0..TRACE_CAPACITY as u32 {
            record(
                BaselineTraceEvent::Wakeup { task_id },
                BaselineTraceSnapshot {
                    cpu_id: 0,
                    current: None,
                    runnable: vec![task_id],
                    need_resched: false,
                    dispatch_target: None,
                    worker_current: vec![None],
                    worker_need_resched: vec![false],
                    worker_dispatch_target: vec![None],
                },
            );
        }

        assert!(!overflowed());

        record(
            BaselineTraceEvent::Wakeup {
                task_id: TRACE_CAPACITY as u32,
            },
            BaselineTraceSnapshot {
                cpu_id: 0,
                current: None,
                runnable: vec![TRACE_CAPACITY as u32],
                need_resched: false,
                dispatch_target: None,
                worker_current: vec![None],
                worker_need_resched: vec![false],
                worker_dispatch_target: vec![None],
            },
        );

        assert!(overflowed());
        assert_eq!(records().len(), TRACE_CAPACITY);
    }
}
