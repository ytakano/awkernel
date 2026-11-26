//! Executors for the forward pass.
//!
//! [`TaskPool`] splits each data-parallel loop of the engine over the
//! service task and a few helper tasks running on other CPUs.
//!
//! A loop of `n` items is cut into `chunks` pieces. Their indices are handed
//! out through one atomic *ticket* `(generation, chunks, next)`: whoever
//! claims an index with a CAS runs that chunk. The service task claims chunks
//! too, so a loop finishes even if no helper is scheduled (e.g. with a single
//! worker CPU, or while helpers are preempted); helpers only speed it up. The
//! service then waits only for chunks that helpers have claimed and are
//! running. Results do not depend on who runs which chunk.
//!
//! Wake-up latency of a sleeping task (tens of microseconds) is comparable to
//! one loop, so while an answer is generated the helpers spin waiting for the
//! next loop, yielding about every millisecond so that other tasks on their
//! CPUs still run. Between answers they sleep.
//!
//! Panics: a chunk that panics still counts as finished, and `par_for` does not
//! return or unwind past its frame before every claimed chunk has finished,
//! so no helper uses the closure after it is gone. A panic in a helper is
//! re-raised in the service task.

use alloc::{boxed::Box, format, sync::Arc};
use awkernel_async_lib::scheduler::SchedulerType;
use awkernel_lib::time::Instant;
use awkernel_llm::{exec::chunk, Executor, InlineExecutor};
use core::{
    cell::UnsafeCell,
    ops::Range,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    time::Duration,
};

/// Upper bound on helper tasks. Decoding is memory-bound; on the
/// development machine llm.asm stops scaling at about 4 threads.
const MAX_HELPERS: usize = 3;
/// Chunks per participating task (dynamic load balancing).
const CHUNKS_PER_TASK: usize = 4;
/// A spinning helper yields its CPU after this long.
const YIELD_INTERVAL: Duration = Duration::from_millis(1);
/// Sleep of an idle helper (no answer in progress).
const IDLE_SLEEP: Duration = Duration::from_millis(5);

const NEXT_MASK: u64 = 0xffff;

/// The executor used by the service task, and a handle to (de)activate the
/// helpers around an answer.
pub(crate) fn create() -> (Box<dyn Executor + Send>, Option<Arc<Shared>>) {
    let workers = awkernel_lib::cpu::num_cpu().saturating_sub(1); // CPU 0 runs no tasks
    let helpers = workers.saturating_sub(1).min(MAX_HELPERS);
    if helpers == 0 {
        return (Box::new(InlineExecutor), None);
    }

    let shared = Arc::new(Shared::new(helpers + 1));
    for i in 0..helpers {
        let s = shared.clone();
        let name = format!("[local_llm] helper {i}");
        awkernel_async_lib::task::spawn(
            name.into(),
            async move {
                helper(s).await;
                Ok(())
            },
            SchedulerType::PrioritizedFIFO(0),
        );
    }
    log::info!("[local_llm] service: {} helper tasks", helpers);
    (
        Box::new(TaskPool {
            shared: shared.clone(),
        }),
        Some(shared),
    )
}

/// The current loop. Written by the caller of `par_for` only while no chunk
/// of the previous loop is outstanding; read by a helper only while it holds
/// a claimed chunk of the current loop.
struct Job {
    f: *const (dyn Fn(Range<usize>) + Sync + 'static),
    n: usize,
    chunks: usize,
}

fn noop(_: Range<usize>) {}
static NOOP: fn(Range<usize>) = noop;

/// Keeps a value on its own cache line.
#[repr(align(64))]
struct CacheLine<T>(T);

pub(crate) struct Shared {
    /// `generation << 32 | chunks << 16 | next`. Read together with `job`
    /// and `active`, which share its line.
    ticket: CacheLine<AtomicU64>,
    /// Chunks of the current loop that have finished (polled by the caller;
    /// on its own line so that finishing does not disturb claiming).
    done: CacheLine<AtomicUsize>,
    job: UnsafeCell<Job>,
    /// An answer is being generated: helpers spin instead of sleeping.
    active: AtomicBool,
    /// A `par_for` is running (the protocol supports one caller at a time).
    in_use: AtomicBool,
    /// A chunk panicked in a helper.
    poisoned: AtomicBool,
    tasks: usize,
}

// SAFETY: `job` is only accessed under the ticket protocol described above,
// and `in_use` admits one `par_for` at a time.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    fn new(tasks: usize) -> Self {
        Self {
            ticket: CacheLine(AtomicU64::new(0)),
            done: CacheLine(AtomicUsize::new(0)),
            job: UnsafeCell::new(Job {
                f: &NOOP,
                n: 0,
                chunks: 0,
            }),
            active: AtomicBool::new(false),
            in_use: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            tasks,
        }
    }

    /// Helpers spin while active; call around each answer.
    pub(crate) fn set_active(&self, on: bool) {
        self.active.store(on, Ordering::Release);
    }

    /// Claims and runs chunks of the current loop until none is left.
    /// Returns true if at least one chunk was run.
    fn work(&self) -> bool {
        let mut worked = false;
        loop {
            let t = self.ticket.0.load(Ordering::Acquire);
            let chunks = (t >> 16) & NEXT_MASK;
            let next = t & NEXT_MASK;
            if next >= chunks {
                return worked;
            }
            if self
                .ticket
                .0
                .compare_exchange_weak(t, t + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // SAFETY: holding an unfinished chunk of the current loop, so the
            // caller cannot replace the job or leave `par_for` (which keeps
            // the closure alive).
            let (f, n, chunks) = unsafe {
                let job = &*self.job.get();
                (&*job.f, job.n, job.chunks)
            };
            let mut guard = ChunkDone {
                shared: self,
                finished: false,
            };
            f(range(n, chunks, next as usize));
            guard.finished = true;
            worked = true;
        }
    }
}

/// Counts a claimed chunk as done even if it panics.
struct ChunkDone<'a> {
    shared: &'a Shared,
    finished: bool,
}

impl Drop for ChunkDone<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.shared.poisoned.store(true, Ordering::Release);
        }
        self.shared.done.0.fetch_add(1, Ordering::Release);
    }
}

/// Ends the current loop in `par_for`, also when unwinding: stops further
/// claims and waits for every claimed chunk, so that no helper still runs the
/// closure when `par_for`'s frame (and the closure's captures) go away.
struct LoopEnd<'a> {
    shared: &'a Shared,
}

impl Drop for LoopEnd<'_> {
    fn drop(&mut self) {
        let s = self.shared;
        let mut t = s.ticket.0.load(Ordering::Acquire);
        let claimed = loop {
            let chunks = (t >> 16) & NEXT_MASK;
            let next = t & NEXT_MASK;
            if next >= chunks {
                break chunks;
            }
            let closed = (t & !NEXT_MASK) | chunks;
            match s
                .ticket
                .0
                .compare_exchange_weak(t, closed, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break next,
                Err(cur) => t = cur,
            }
        };
        while (s.done.0.load(Ordering::Acquire) as u64) < claimed {
            core::hint::spin_loop();
        }
        s.in_use.store(false, Ordering::Release);
    }
}

fn range(n: usize, chunks: usize, i: usize) -> Range<usize> {
    // Rows of large loops are split at multiples of 16 (one cache line of f32).
    let align = if n >= chunks * 64 { 16 } else { 1 };
    chunk(n, chunks, i, align)
}

pub(crate) struct TaskPool {
    shared: Arc<Shared>,
}

unsafe impl Executor for TaskPool {
    fn par_for(&self, n: usize, f: &(dyn Fn(Range<usize>) + Sync)) {
        if n == 0 {
            return;
        }
        let s = &*self.shared;
        let chunks = n.min(s.tasks * CHUNKS_PER_TASK).min(NEXT_MASK as usize);
        // One caller at a time; a concurrent caller runs its loop alone.
        if chunks <= 1 || s.in_use.swap(true, Ordering::Acquire) {
            f(0..n);
            return;
        }

        // SAFETY: all chunks of the previous loop have finished (`LoopEnd`
        // waited for them), so no helper reads `job` now. The lifetime of `f`
        // is erased; `f` outlives every use because `LoopEnd` waits for all
        // claimed chunks before this frame is left, even when unwinding.
        unsafe {
            let f: &(dyn Fn(Range<usize>) + Sync + 'static) = core::mem::transmute(f);
            *s.job.get() = Job { f, n, chunks };
        }
        s.done.0.store(0, Ordering::Relaxed);
        let generation = (s.ticket.0.load(Ordering::Relaxed) >> 32).wrapping_add(1) & 0xffff_ffff;
        s.ticket.0.store(
            (generation << 32) | ((chunks as u64) << 16),
            Ordering::Release,
        );

        let end = LoopEnd { shared: s };
        s.work();
        drop(end); // waits for the helpers' chunks

        if s.poisoned.load(Ordering::Acquire) {
            panic!("local_llm: a helper task panicked");
        }
    }
}

async fn helper(s: Arc<Shared>) {
    loop {
        if !s.active.load(Ordering::Acquire) {
            awkernel_async_lib::sleep(IDLE_SLEEP).await;
            continue;
        }
        let start = Instant::now();
        let mut spins = 0u32;
        while s.active.load(Ordering::Relaxed) {
            if !s.work() {
                core::hint::spin_loop();
            }
            spins = spins.wrapping_add(1);
            if spins % 256 == 0 && start.elapsed() >= YIELD_INTERVAL {
                break;
            }
        }
        // Let other tasks on this CPU run.
        awkernel_async_lib::r#yield().await;
    }
}
