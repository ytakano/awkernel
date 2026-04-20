use alloc::{format, string::String, vec::Vec};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
use awkernel_lib::console;

pub const SERIAL_PREFIX: &str = "BASELINE_TRACE:";
pub const SERIAL_DONE_MARKER: &str = "BASELINE_TRACE_DONE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineTraceEvent {
    Wakeup { task_id: u32 },
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
#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
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

#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
pub fn arm_dump_on_complete(task_id: u32) {
    DUMP_ON_COMPLETE_TASK_ID.store(task_id, Ordering::Release);
}

#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
pub fn take_dump_on_complete(task_id: u32) -> bool {
    DUMP_ON_COMPLETE_TASK_ID
        .compare_exchange(task_id, 0, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(all(not(feature = "std"), feature = "baseline_trace_vm"))]
pub fn dump_to_console() {
    for line in render_lines() {
        console::print(&format!("{SERIAL_PREFIX} {line}\r\n"));
    }
    console::print(&format!("{SERIAL_DONE_MARKER}\r\n"));
}

fn event_name(event: BaselineTraceEvent) -> &'static str {
    match event {
        BaselineTraceEvent::Wakeup { .. } => "EvWakeup",
        BaselineTraceEvent::Choose { .. } => "EvChoose",
        BaselineTraceEvent::Dispatch { .. } => "EvDispatch",
        BaselineTraceEvent::Complete { .. } => "EvComplete",
        BaselineTraceEvent::Stutter => "EvStutter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_multiple_cpus_in_baseline_trace() {
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
}
