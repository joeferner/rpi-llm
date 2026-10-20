//! Splitting the model's work across cores, without knowing what a core
//! is.
//!
//! The engine has no hardware in it, so it asks for parallelism through
//! [`Workers`]: run this job with each index from 0 to `count`, at once,
//! and return when all are done. The board implements it with the Pi's
//! four cores; the host tests with threads; [`Serial`] is one worker, the
//! default.
//!
//! What gets split is a matrix multiply's output rows, in contiguous
//! ranges, one per worker. Each row is the same dot product whichever
//! worker computes it, so the split changes no result: one worker or four
//! produce the same bits.

/// Something that runs a job on several workers at once.
pub trait Workers: Sync {
    /// How many workers there are: a job's indices are `0..count()`.
    fn count(&self) -> usize;

    /// Runs `job(i)` for every `i` in `0..count()`, concurrently, and
    /// returns once every call has returned.
    fn run(&self, job: &(dyn Fn(usize) + Sync));
}

/// One worker: the job runs on the calling core.
pub struct Serial;

impl Workers for Serial {
    fn count(&self) -> usize {
        1
    }

    fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        job(0);
    }
}

/// A buffer the workers share, each writing only its own part of it.
///
/// Made from a `&mut [T]` the caller holds for as long as the workers
/// run, which keeps everything else away from it meanwhile; the workers'
/// parts being disjoint is the caller's to guarantee, and every accessor
/// is `unsafe` for that reason.
pub(crate) struct Shared<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: see `Shared`: the workers only reach it through accessors whose
// callers promise disjoint parts, and the elements themselves must be
// sendable between workers.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// Shares `buf`.
    pub(crate) fn new(buf: &mut [T]) -> Self {
        Self {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
        }
    }

    /// Elements in the buffer.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Elements `start..start + len`, writable.
    ///
    /// A method rather than a field access in the workers' closure: a
    /// closure that named the raw pointer field would capture just that
    /// pointer, which is not `Sync`, instead of this wrapper, which is.
    ///
    /// # Safety
    ///
    /// The range must be inside the buffer, and no other worker may touch
    /// it while the returned slice lives.
    #[allow(clippy::mut_from_ref, reason = "disjointness is the caller's promise")]
    pub(crate) unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start + len <= self.len, "a shared slice out of bounds");
        // SAFETY: in bounds (checked); exclusive by the caller's promise.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

/// Runs `f(i)` for every `i` in `0..n`, the indices split into contiguous
/// ranges across `workers`.
pub(crate) fn for_each(workers: &dyn Workers, n: usize, f: impl Fn(usize) + Sync) {
    let per = n.div_ceil(workers.count().max(1));
    workers.run(&|w| {
        for i in (w * per).min(n)..((w + 1) * per).min(n) {
            f(i);
        }
    });
}

/// Sets `out[r] = row(r)` for every row, the rows dealt out to `workers`
/// in turn, `chunk` at a time: worker `w` takes chunks `w`, `w + n`,
/// `w + 2n`...
///
/// Interleaved rather than contiguous because of what the Pi 5's memory
/// does with four readers. Four cores each streaming their own quarter of
/// a matrix read 11.5 GB/s between them; taking turns through it a few
/// KiB at a time, so that at any moment they are reading neighbouring
/// addresses, they read 13.9 — as fast as one core alone (rpi-llm's
/// `examples/bandwidth.rs`). A matrix multiply sizes `chunk` to about
/// 4 KiB of its weights.
pub(crate) fn for_rows(
    workers: &dyn Workers,
    out: &mut [f32],
    chunk: usize,
    row: impl Fn(usize) -> f32 + Sync,
) {
    let rows = Shared::new(out);
    for_chunks(workers, rows.len(), chunk, |r| {
        // SAFETY: each row index goes to exactly one worker.
        unsafe { rows.slice(r, 1)[0] = row(r) };
    });
}

/// Runs `f(i)` for every `i` in `0..len`, the indices dealt out to
/// `workers` in turn, `chunk` at a time: the split [`for_rows`] uses, for
/// work that writes more than one output per index.
pub(crate) fn for_chunks(
    workers: &dyn Workers,
    len: usize,
    chunk: usize,
    f: impl Fn(usize) + Sync,
) {
    let chunk = chunk.max(1);
    let n = workers.count().max(1);
    workers.run(&|w| {
        let mut start = w * chunk;
        while start < len {
            for i in start..(start + chunk).min(len) {
                f(i);
            }
            start += n * chunk;
        }
    });
}
