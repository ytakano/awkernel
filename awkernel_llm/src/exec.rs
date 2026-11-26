//! Data-parallel execution.
//!
//! The forward pass consists of independent row (matrix-vector product) or
//! head (attention) computations. It hands each such loop to an [`Executor`],
//! which decides how to split `0..n` and on which CPUs to run the pieces.
//! Every row/head is computed by the same code regardless of the split, so
//! results are bit-identical for any executor.

use core::{marker::PhantomData, ops::Range};

/// Runs a data-parallel loop.
///
/// # Safety
///
/// The engine writes to disjoint parts of shared buffers from `f`, relying on
/// this contract. An implementation must
///
/// - call `f` with ranges that are pairwise disjoint, lie within `0..n` and
///   together cover `0..n` exactly once, and
/// - not return from `par_for` before every call of `f` has returned.
pub unsafe trait Executor: Sync {
    /// Calls `f` on a partition of `0..n`.
    fn par_for(&self, n: usize, f: &(dyn Fn(Range<usize>) + Sync));
}

/// Runs the whole loop on the calling thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct InlineExecutor;

unsafe impl Executor for InlineExecutor {
    #[inline]
    fn par_for(&self, n: usize, f: &(dyn Fn(Range<usize>) + Sync)) {
        if n > 0 {
            f(0..n);
        }
    }
}

/// Returns the `i`-th of `parts` contiguous chunks of `0..n`, for executors.
///
/// Large loops (matrix rows) are split at multiples of 16 so that neighbouring
/// workers do not share a cache line of an `f32` output; small loops
/// (attention heads) are split per item so that every worker gets work.
pub fn split(n: usize, parts: usize, i: usize) -> Range<usize> {
    let align = if n >= parts * 64 { 16 } else { 1 };
    chunk(n, parts, i, align)
}

/// Returns the `i`-th of `parts` contiguous chunks of `0..n` whose boundaries
/// are multiples of `align` (except the end). Trailing chunks may be empty.
pub fn chunk(n: usize, parts: usize, i: usize, align: usize) -> Range<usize> {
    debug_assert!(parts > 0 && i < parts && align > 0);
    let units = n.div_ceil(align);
    let per = units / parts;
    let rem = units % parts;
    let start_u = i * per + i.min(rem);
    let end_u = start_u + per + usize::from(i < rem);
    (start_u * align).min(n)..(end_u * align).min(n)
}

/// A mutable slice that may be shared by the workers of one `par_for` call,
/// each of which accesses a disjoint sub-range.
pub(crate) struct SharedMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<T: Send> Sync for SharedMut<'_, T> {}
unsafe impl<T: Send> Send for SharedMut<'_, T> {}

impl<'a, T> SharedMut<'a, T> {
    pub(crate) fn new(s: &'a mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
            _marker: PhantomData,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    ///
    /// Ranges obtained concurrently must not overlap.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn range(&self, r: Range<usize>) -> &mut [T] {
        assert!(r.start <= r.end && r.end <= self.len);
        unsafe { core::slice::from_raw_parts_mut(self.ptr.add(r.start), r.end - r.start) }
    }
}

#[cfg(any(test, feature = "std"))]
mod thread_exec {
    use super::{split, Executor};
    use core::ops::Range;

    /// Splits each loop over `threads` OS threads (host only).
    ///
    /// Threads are spawned per call (`std::thread::scope`), which costs tens
    /// of microseconds per loop. It exists to check that results do not depend
    /// on the split, not for speed.
    #[derive(Debug, Clone, Copy)]
    pub struct ScopedThreadExecutor {
        pub threads: usize,
    }

    unsafe impl Executor for ScopedThreadExecutor {
        fn par_for(&self, n: usize, f: &(dyn Fn(Range<usize>) + Sync)) {
            let parts = self.threads.max(1);
            std::thread::scope(|s| {
                for i in 1..parts {
                    let r = split(n, parts, i);
                    if !r.is_empty() {
                        s.spawn(move || f(r));
                    }
                }
                let r = split(n, parts, 0);
                if !r.is_empty() {
                    f(r);
                }
            });
        }
    }
}

#[cfg(any(test, feature = "std"))]
pub use thread_exec::ScopedThreadExecutor;

#[cfg(test)]
mod tests {
    use super::{chunk, split};

    #[test]
    fn chunks_partition_range() {
        for n in [0, 1, 15, 16, 17, 100, 1024, 4096, 151_936] {
            for parts in 1..9 {
                let mut next = 0;
                for i in 0..parts {
                    let r = chunk(n, parts, i, 16);
                    assert_eq!(r.start, next.min(n));
                    assert!(r.start <= r.end);
                    if r.end < n {
                        assert_eq!(r.end % 16, 0);
                    }
                    next = r.end;
                }
                assert_eq!(next, n);
            }
        }
    }

    #[test]
    fn small_loops_use_all_parts() {
        for parts in 1..=8 {
            let busy = (0..parts)
                .filter(|&i| !split(16, parts, i).is_empty())
                .count();
            assert_eq!(busy, parts.min(16));
        }
    }
}
