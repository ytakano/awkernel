use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use array_macro::array;
use awkernel_lib::{cpu::NUM_MAX_CPU, delay::cpu_counter};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::sync::atomic::AtomicU64;
#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
use core::sync::atomic::{AtomicU32, Ordering};
#[cfg(all(not(feature = "std"), feature = "handoff_trace_vm"))]
use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
use awkernel_lib::console;

pub const SERIAL_PREFIX: &str = "BASELINE_TRACE:";
pub const SERIAL_DONE_MARKER: &str = "BASELINE_TRACE_DONE";
pub const ROCQ_BEGIN_MARKER: &str = "BEGIN_ROCQ_TRACE";
pub const ROCQ_END_MARKER: &str = "END_ROCQ_TRACE";
pub const TRACE_ROWS_BEGIN_MARKER: &str = "BEGIN_TRACE_ROWS";
pub const TRACE_ROWS_END_MARKER: &str = "END_TRACE_ROWS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineTraceEvent {
    Wakeup { task_id: u32 },
    RequestResched { cpu_id: usize },
    HandleResched { cpu_id: usize },
    Choose { task_id: u32 },
    Dispatch { task_id: u32 },
    Complete { task_id: u32 },
    Stutter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineTraceSnapshot {
    pub cpu_id: usize,
    pub current: Option<u32>,
    pub runnable: Vec<u32>,
    pub need_resched: bool,
    pub dispatch_target: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineTraceRecord {
    pub event_id: u64,
    pub tsc: u64,
    pub event: BaselineTraceEvent,
    pub snapshot: BaselineTraceSnapshot,
}

const TRACE_CAPACITY: usize = 128;

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
            self.records.reserve(TRACE_CAPACITY - self.records.capacity());
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

static BASELINE_TRACE: [Mutex<TraceBuffer>; NUM_MAX_CPU] =
    array![_ => Mutex::new(TraceBuffer::new()); NUM_MAX_CPU];
static TRACE_EVENT_ID: AtomicU64 = AtomicU64::new(0);
#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
static DUMP_ON_COMPLETE_TASK_ID: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn reset() {
    TRACE_EVENT_ID.store(0, core::sync::atomic::Ordering::Release);
    for trace in BASELINE_TRACE.iter() {
        let mut node = MCSNode::new();
        let mut trace = trace.lock(&mut node);
        trace.reset();
    }
}

#[inline(always)]
pub fn record(event: BaselineTraceEvent, snapshot: BaselineTraceSnapshot) {
    let event_id = TRACE_EVENT_ID.fetch_add(1, core::sync::atomic::Ordering::AcqRel);
    let tsc = cpu_counter();
    let cpu_id = snapshot.cpu_id;
    let mut node = MCSNode::new();
    let mut trace = BASELINE_TRACE[cpu_id].lock(&mut node);
    trace.push(BaselineTraceRecord {
        event_id,
        tsc,
        event,
        snapshot,
    });
}

fn merge_records(mut records: Vec<BaselineTraceRecord>) -> Vec<BaselineTraceRecord> {
    records.sort_by(|lhs, rhs| lhs.event_id.cmp(&rhs.event_id));
    records
}

#[inline(always)]
pub fn records() -> Vec<BaselineTraceRecord> {
    let mut merged = Vec::new();

    for trace in BASELINE_TRACE.iter() {
        let mut node = MCSNode::new();
        let trace = trace.lock(&mut node);
        merged.extend(trace.records.iter().cloned());
    }

    merge_records(merged)
}

#[inline(always)]
pub fn overflowed() -> bool {
    BASELINE_TRACE.iter().any(|trace| {
        let mut node = MCSNode::new();
        let trace = trace.lock(&mut node);
        trace.overflowed
    })
}

pub fn render_lines() -> Vec<String> {
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

fn render_rocq_event(event: BaselineTraceEvent) -> String {
    match event {
        BaselineTraceEvent::Wakeup { task_id } => format!("EvWakeup {task_id}"),
        BaselineTraceEvent::RequestResched { cpu_id } => {
            format!("EvRequestResched {cpu_id}")
        }
        BaselineTraceEvent::HandleResched { cpu_id } => {
            format!("EvHandleResched {cpu_id}")
        }
        BaselineTraceEvent::Choose { task_id } => format!("EvChoose 1 {task_id}"),
        BaselineTraceEvent::Dispatch { task_id } => format!("EvDispatch 1 {task_id}"),
        BaselineTraceEvent::Complete { task_id } => format!("EvComplete {task_id}"),
        BaselineTraceEvent::Stutter => "EvStutter".to_string(),
    }
}

fn render_rocq_option(value: Option<u32>) -> String {
    match value {
        Some(v) => format!("(Some {v})"),
        None => "None".to_string(),
    }
}

fn render_rocq_list(values: &[u32]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

fn render_rocq_row(record: BaselineTraceRecord) -> String {
    format!(
        "mkAwkernelCapturedRow {} ({}) {} {} {} {}",
        record.snapshot.cpu_id,
        render_rocq_event(record.event),
        render_rocq_option(record.snapshot.current),
        render_rocq_list(&record.snapshot.runnable),
        if record.snapshot.need_resched {
            "true"
        } else {
            "false"
        },
        render_rocq_option(record.snapshot.dispatch_target)
    )
}

fn render_trace_rows_event(
    event: BaselineTraceEvent,
) -> (&'static str, Option<u32>, Option<u32>) {
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
        BaselineTraceEvent::Stutter => ("Stutter", None, None),
    }
}

fn render_trace_rows_option(value: Option<u32>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
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
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
        render_trace_rows_option(record.snapshot.dispatch_target)
    )
}

pub fn render_trace_rows_artifact_lines() -> Vec<String> {
    records()
        .into_iter()
        .map(render_trace_rows_record)
        .collect()
}

pub fn render_rocq_handoff_artifact_lines() -> Vec<String> {
    let records = records();
    let mut lines = vec![
        "From Stdlib Require Import List.".to_string(),
        "From RocqSched Require Import Operational.Common.Step.".to_string(),
        "From RocqSched Require Import Operational.Awkernel.CapturedTraceSyntax.".to_string(),
        "Import ListNotations.".to_string(),
        "".to_string(),
        "Definition awk_generated_handoff_rows : list AwkernelCapturedRow :=".to_string(),
    ];

    if records.is_empty() {
        lines.push("  [ ].".to_string());
        return lines;
    }

    for (idx, record) in records.into_iter().enumerate() {
        let prefix = if idx == 0 { "  [ " } else { "  ; " };
        lines.push(format!("{prefix}{}", render_rocq_row(record)));
    }
    lines.push("  ].".to_string());
    lines
}

#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
pub fn arm_dump_on_complete(task_id: u32) {
    DUMP_ON_COMPLETE_TASK_ID.store(task_id, Ordering::Release);
}

#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
pub fn take_dump_on_complete(task_id: u32) -> bool {
    DUMP_ON_COMPLETE_TASK_ID
        .compare_exchange(task_id, 0, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
pub fn dump_to_console() {
    if overflowed() {
        console::print("BASELINE_TRACE_OVERFLOW\r\n");
    }
    for line in render_lines() {
        console::print(&format!("{SERIAL_PREFIX} {line}\r\n"));
    }
    console::print(&format!("{SERIAL_DONE_MARKER}\r\n"));
    #[cfg(feature = "handoff_trace_vm")]
    {
        console::print(&format!("{TRACE_ROWS_BEGIN_MARKER}\r\n"));
        for line in render_trace_rows_artifact_lines() {
            console::print(&format!("{line}\r\n"));
        }
        console::print(&format!("{TRACE_ROWS_END_MARKER}\r\n"));
        console::print(&format!("{ROCQ_BEGIN_MARKER}\r\n"));
        for line in render_rocq_handoff_artifact_lines() {
            console::print(&format!("{line}\r\n"));
        }
        console::print(&format!("{ROCQ_END_MARKER}\r\n"));
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
        BaselineTraceEvent::Stutter => "EvStutter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

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
            },
        );

        let records = records();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].event,
            BaselineTraceEvent::Wakeup { task_id: 1 }
        );
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
            },
        );

        let lines = render_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("cpu=1"));
        assert!(lines[0].contains("event=EvDispatch"));
        assert!(lines[0].contains("current=Some(7)"));
    }

    #[test]
    fn renders_rocq_handoff_artifact() {
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
            },
        );
        record(
            BaselineTraceEvent::RequestResched { cpu_id: 1 },
            BaselineTraceSnapshot {
                cpu_id: 1,
                current: None,
                runnable: vec![1],
                need_resched: true,
                dispatch_target: None,
            },
        );

        let lines = render_rocq_handoff_artifact_lines();
        assert_eq!(lines[0], "From Stdlib Require Import List.");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Definition awk_generated_handoff_rows"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("mkAwkernelCapturedRow 0 (EvWakeup 1) None [1] false None"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("mkAwkernelCapturedRow 1 (EvRequestResched 1) None [1] true None"))
        );
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
            },
        );

        let lines = render_trace_rows_artifact_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "1\tChoose\t1\t1\t-\t1\ttrue\t1");
        assert_eq!(lines[1], "1\tDispatch\t1\t1\t1\t\tfalse\t-");
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
            },
        );

        assert!(overflowed());
        assert_eq!(records().len(), TRACE_CAPACITY);
    }
}
