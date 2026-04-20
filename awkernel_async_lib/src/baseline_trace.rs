use alloc::{format, string::String, vec::Vec};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};

pub const BASELINE_CPU_ID: usize = 0;

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

#[inline(always)]
pub fn reset() {
    let mut node = MCSNode::new();
    let mut trace = BASELINE_TRACE.lock(&mut node);
    trace.clear();
}

#[inline(always)]
pub fn record(event: BaselineTraceEvent, snapshot: BaselineTraceSnapshot) {
    if snapshot.cpu_id != BASELINE_CPU_ID {
        return;
    }

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
    fn records_only_baseline_cpu() {
        reset();
        record(
            BaselineTraceEvent::Wakeup { task_id: 1 },
            BaselineTraceSnapshot {
                cpu_id: BASELINE_CPU_ID,
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
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].event,
            BaselineTraceEvent::Wakeup { task_id: 1 }
        );
    }

    #[test]
    fn renders_line_oriented_trace() {
        reset();
        record(
            BaselineTraceEvent::Dispatch { task_id: 7 },
            BaselineTraceSnapshot {
                cpu_id: BASELINE_CPU_ID,
                current: Some(7),
                runnable: vec![],
                need_resched: false,
                dispatch_target: None,
            },
        );

        let lines = render_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("event=EvDispatch"));
        assert!(lines[0].contains("current=Some(7)"));
    }
}
