//! The Pi 5's four cores as the engine's [`Workers`].
//!
//! Cores 1-3 are started once and then wait in `wfe` for work. A job is
//! published by core 0: it stores the job, bumps a generation counter
//! (release), and signals an event; each waiting core sees the new
//! generation (acquire), runs its index of the job, and counts itself
//! done. Core 0 runs index 0 itself, then waits for the other three. A
//! token runs about 225 of these, one per matrix multiply, so the
//! handoff has to cost microseconds, and does: no locks, no interrupts,
//! just the event register and three cache lines.
//!
//! Each shared word sits in its own cache line ([`CacheAligned`]): two
//! cores working on neighbouring atomics would otherwise keep clearing
//! each other's exclusive reservations.

use core::cell::UnsafeCell;
use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use rpi_hal::cpu::core_id;
use rpi_hal::multicore::{CacheAligned, Cores, Stack, signal_event, wait_for_event};
use rpi_llm_engine::parallel::Workers;

/// The job being run, as a pointer whose lifetime has been erased.
///
/// Only [`Pool::run`] writes it, before publishing a new generation, and
/// it is only read between that generation's publication and the reader's
/// done count — inside which `run` is still waiting, so the job it points
/// to is still alive.
struct Job(UnsafeCell<Option<*const (dyn Fn(usize) + Sync)>>);

// SAFETY: see `Job`: writes and reads are ordered by `GO` and `DONE`, and
// never overlap.
unsafe impl Sync for Job {}

static JOB: CacheAligned<Job> = CacheAligned(Job(UnsafeCell::new(None)));

/// Bumped by core 0 to start a job; the workers wait for it to change.
static GO: CacheAligned<AtomicU32> = CacheAligned(AtomicU32::new(0));

/// Workers that have finished the current job.
static DONE: CacheAligned<AtomicU32> = CacheAligned(AtomicU32::new(0));

/// The secondary cores' stacks. The engine's work on a core is a dot
/// product loop, so these stay shallow.
static mut STACKS: [Stack<0x10000>; 3] = [const { Stack::new() }; 3];

/// The four cores, as [`Workers`]. There is only one, made by
/// [`Pool::start`].
pub struct Pool(());

impl Pool {
    /// Starts cores 1-3 waiting for work.
    ///
    /// # Safety
    ///
    /// Call once: it steals the secondary cores.
    pub unsafe fn start() -> &'static Pool {
        static POOL: Pool = Pool(());
        // SAFETY: the caller calls this once, so the cores are stolen once
        // and each stack is handed to one spawn; `worker` never returns and
        // shares only the statics above.
        unsafe {
            let [s1, s2, s3] = &mut *addr_of_mut!(STACKS);
            let cores = Cores::steal();
            cores.core1.spawn(s1, worker);
            cores.core2.spawn(s2, worker);
            cores.core3.spawn(s3, worker);
        }
        &POOL
    }
}

impl Workers for Pool {
    fn count(&self) -> usize {
        4
    }

    fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        // SAFETY: no worker reads `JOB` until `GO` changes below, and every
        // worker finished reading the last one before `DONE` reached 3.
        // The lifetime is erased only for the span `run` waits out.
        unsafe {
            let job: *const (dyn Fn(usize) + Sync + '_) = job;
            *JOB.0.0.get() = Some(core::mem::transmute::<
                *const (dyn Fn(usize) + Sync + '_),
                *const (dyn Fn(usize) + Sync + 'static),
            >(job));
        }
        DONE.0.store(0, Ordering::Relaxed);
        let published = ticks();
        PUBLISHED.0.store(published, Ordering::Relaxed);
        GO.0.fetch_add(1, Ordering::Release);
        signal_event();

        job(0);
        let own = ticks();

        while DONE.0.load(Ordering::Acquire) < 3 {
            core::hint::spin_loop();
        }
        let all = ticks();

        STATS[0]
            .0
            .work
            .fetch_add(own - published, Ordering::Relaxed);
        STATS[0].0.wait.fetch_add(all - own, Ordering::Relaxed);
        JOBS_RUN.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// The ARM generic counter: a timebase every core shares, at the rate in
/// `CNTFRQ_EL0` — far finer than the System Timer's microsecond.
fn ticks() -> u64 {
    let t: u64;
    // SAFETY: reads a system register readable at EL1; no side effects.
    unsafe { core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) t, options(nomem, nostack)) };
    t
}

/// The generic counter's rate, in Hz.
fn tick_rate() -> u64 {
    let f: u64;
    // SAFETY: as in `ticks`.
    unsafe { core::arch::asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack)) };
    f
}

/// Where one core's time went, summed over every job, in counter ticks.
struct CoreStats {
    /// Core 0: publishing a job to finishing its own part. Workers: seeing
    /// the job to finishing their part.
    work: AtomicU64,
    /// Core 0: finishing its own part to the last worker finishing.
    /// Workers: the job's publication to seeing it — the wake-up.
    wait: AtomicU64,
}

static STATS: [CacheAligned<CoreStats>; 4] = [const {
    CacheAligned(CoreStats {
        work: AtomicU64::new(0),
        wait: AtomicU64::new(0),
    })
}; 4];

/// When core 0 published the current job, for the workers' wake-up time.
static PUBLISHED: CacheAligned<AtomicU64> = CacheAligned(AtomicU64::new(0));

/// Jobs run since the statistics were last reset.
static JOBS_RUN: CacheAligned<AtomicU64> = CacheAligned(AtomicU64::new(0));

/// Zeroes the statistics.
pub fn reset_stats() {
    for s in &STATS {
        s.0.work.store(0, Ordering::Relaxed);
        s.0.wait.store(0, Ordering::Relaxed);
    }
    JOBS_RUN.0.store(0, Ordering::Relaxed);
}

/// Where the cores' time went since [`reset_stats`]: per core, the
/// microseconds working and waiting (see [`CoreStats`]), and the jobs run.
pub fn stats() -> ([(u64, u64); 4], u64) {
    let per_us = tick_rate() / 1_000_000;
    let mut out = [(0, 0); 4];
    for (o, s) in out.iter_mut().zip(&STATS) {
        *o = (
            s.0.work.load(Ordering::Relaxed) / per_us,
            s.0.wait.load(Ordering::Relaxed) / per_us,
        );
    }
    (out, JOBS_RUN.0.load(Ordering::Relaxed))
}

/// Cores 1-3: wait for a job, run this core's index of it, count done.
extern "C" fn worker() -> ! {
    let id = core_id();
    let mut seen = 0;
    loop {
        loop {
            let go = GO.0.load(Ordering::Acquire);
            if go != seen {
                seen = go;
                break;
            }
            wait_for_event();
        }
        let woke = ticks();
        // SAFETY: `GO` was published after `JOB` was written, and `run`
        // is waiting for this core's done count, so the job is alive.
        unsafe {
            if let Some(job) = *JOB.0.0.get() {
                (*job)(id);
            }
        }
        let done = ticks();
        let published = PUBLISHED.0.load(Ordering::Relaxed);
        STATS[id]
            .0
            .wait
            .fetch_add(woke.saturating_sub(published), Ordering::Relaxed);
        STATS[id].0.work.fetch_add(done - woke, Ordering::Relaxed);
        DONE.0.fetch_add(1, Ordering::Release);
    }
}
