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
//! next loop, yielding now and then so other tasks on their CPUs still run.
//! Between answers they sleep.

use alloc::{boxed::Box, format, sync::Arc};
use awkernel_async_lib::scheduler::SchedulerType;
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
/// Spin iterations without work before a helper yields its CPU.
const SPIN_BEFORE_YIELD: u32 = 20_000;
/// Sleep of an idle helper (no answer in progress).
const IDLE_SLEEP: Duration = Duration::from_millis(5);

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

/// The current loop. Written by the service task only while no chunk of the
/// previous loop is outstanding; read by a helper only while it holds a
/// claimed chunk of the current loop.
struct Job {
    f: *const (dyn Fn(Range<usize>) + Sync + 'static),
    n: usize,
    chunks: usize,
}

pub(crate) struct Shared {
    /// `generation << 32 | chunks << 16 | next`.
    ticket: AtomicU64,
    /// Chunks of the current loop that have finished.
    done: AtomicUsize,
    job: UnsafeCell<Job>,
    /// An answer is being generated: helpers spin instead of sleeping.
    active: AtomicBool,
    tasks: usize,
}

// SAFETY: `job` is only accessed under the ticket protocol described above.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

const NEXT_MASK: u64 = 0xffff;

fn noop(_: Range<usize>) {}
static NOOP: fn(Range<usize>) = noop;

impl Shared {
    fn new(tasks: usize) -> Self {
        Self {
            ticket: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            job: UnsafeCell::new(Job {
                f: &NOOP,
                n: 0,
                chunks: 0,
            }),
            active: AtomicBool::new(false),
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
            let t = self.ticket.load(Ordering::Acquire);
            let chunks = (t >> 16) & NEXT_MASK;
            let next = t & NEXT_MASK;
            if next >= chunks {
                return worked;
            }
            if self
                .ticket
                .compare_exchange_weak(t, t + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // SAFETY: holding an unfinished chunk of the current loop, so the
            // service task cannot replace the job or return from `par_for`
            // (which keeps the closure alive).
            let (f, n, chunks) = unsafe {
                let job = &*self.job.get();
                (&*job.f, job.n, job.chunks)
            };
            f(range(n, chunks, next as usize));
            self.done.fetch_add(1, Ordering::Release);
            worked = true;
        }
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
        if chunks <= 1 {
            f(0..n);
            return;
        }

        // SAFETY: all chunks of the previous loop have finished (we waited
        // for them), so no helper reads `job` now. The lifetime of `f` is
        // erased; `f` outlives every use because we wait for all chunks
        // before returning.
        unsafe {
            let f: &(dyn Fn(Range<usize>) + Sync + 'static) = core::mem::transmute(f);
            *s.job.get() = Job { f, n, chunks };
        }
        s.done.store(0, Ordering::Relaxed);
        let generation = (s.ticket.load(Ordering::Relaxed) >> 32).wrapping_add(1) & 0xffff_ffff;
        s.ticket.store(
            (generation << 32) | ((chunks as u64) << 16),
            Ordering::Release,
        );

        s.work();
        while s.done.load(Ordering::Acquire) < chunks {
            core::hint::spin_loop();
        }
    }
}

async fn helper(s: Arc<Shared>) {
    loop {
        if !s.active.load(Ordering::Acquire) {
            awkernel_async_lib::sleep(IDLE_SLEEP).await;
            continue;
        }
        let mut idle = 0u32;
        while idle < SPIN_BEFORE_YIELD {
            if s.work() {
                idle = 0;
            } else {
                idle += 1;
                core::hint::spin_loop();
            }
        }
        // Let other tasks on this CPU run (also between tokens).
        awkernel_async_lib::r#yield().await;
    }
}
