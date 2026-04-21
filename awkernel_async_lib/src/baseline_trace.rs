use alloc::{format, string::{String, ToString}, vec, vec::Vec};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
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
    pub event: BaselineTraceEvent,
    pub snapshot: BaselineTraceSnapshot,
}

static BASELINE_TRACE: Mutex<Vec<BaselineTraceRecord>> = Mutex::new(Vec::new());
#[cfg(all(
    not(feature = "std"),
    any(feature = "baseline_trace_vm", feature = "handoff_trace_vm")
))]
static DUMP_ON_COMPLETE_TASK_ID: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn reset() {
    let mut node = MCSNode::new();
    let mut trace = BASELINE_TRACE.lock(&mut node);
    trace.clear();
}

#[inline(always)]
pub fn record(event: BaselineTraceEvent, snapshot: BaselineTraceSnapshot) {
    let mut node = MCSNode::new();
    let mut trace = BASELINE_TRACE.lock(&mut node);
    trace.push(BaselineTraceRecord { event, snapshot });
}

#[inline(always)]
pub fn records() -> Vec<BaselineTraceRecord> {
    let mut node = MCSNode::new();
    let trace = BASELINE_TRACE.lock(&mut node);
    trace.clone()
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
    for line in render_lines() {
        console::print(&format!("{SERIAL_PREFIX} {line}\r\n"));
    }
    console::print(&format!("{SERIAL_DONE_MARKER}\r\n"));
    #[cfg(feature = "handoff_trace_vm")]
    {
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
}
