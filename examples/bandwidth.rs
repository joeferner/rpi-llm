#![no_std]
#![no_main]

//! How fast the Pi 5's memory can be read, by one core and by several at
//! once: the number that decides what splitting the model's matrix
//! multiplies across cores can gain.
//!
//! Generating a token reads every weight once, so one token's time is at
//! least the model's size over this bandwidth. One core already runs
//! SmolLM2 at about 7 GB/s; if all four together cannot read much faster
//! than that, four cores cannot make generation much faster either.
//!
//! The buffer is 256 MiB, far larger than the caches (512 KiB of L2 per
//! core, 2 MiB of L3), so every pass streams from DRAM. Each pass sums the
//! buffer with NEON loads, a cache line per iteration, four independent
//! accumulators deep; the sum is kept so the reads cannot be optimised
//! away. With several cores the buffer is split into equal contiguous
//! parts, all cores start together, and the time is until the last one
//! finishes.
//!
//! ```sh
//! make bandwidth DEVICE=/dev/ttyUSB0
//! ```

extern crate alloc;

use alloc::vec::Vec;
use core::arch::aarch64::*;
use core::fmt::Write;
use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::multicore::{CacheAligned, Cores, Stack, signal_event, wait_for_event};
use rpi_hal::{cpu::core_id, halt, pac, timer::Timer, uart::Uart};
use rpi_kickstart::heap;

/// The buffer's size.
const BUFFER: usize = 256 * 1024 * 1024;

/// Passes per core count; the best is reported.
const PASSES: usize = 3;

/// One cache line.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u64; 8]);

/// What one worker is asked to sum: `count` chunks of `chunk` lines, the
/// first at line `start` and each `step` lines after the last. A
/// contiguous part is one chunk; an interleaved split is many.
struct Job {
    start: AtomicUsize,
    chunk: AtomicUsize,
    step: AtomicUsize,
    count: AtomicUsize,
}

/// Each core's job, each in its own cache line.
static JOBS: [CacheAligned<Job>; 4] = [const {
    CacheAligned(Job {
        start: AtomicUsize::new(0),
        chunk: AtomicUsize::new(0),
        step: AtomicUsize::new(0),
        count: AtomicUsize::new(0),
    })
}; 4];

/// Bumped by core 0 to start a pass; workers wait for it to change.
static GO: CacheAligned<AtomicU32> = CacheAligned(AtomicU32::new(0));

/// Workers that have finished the current pass.
static DONE: CacheAligned<AtomicU32> = CacheAligned(AtomicU32::new(0));

/// Where the workers' sums go, so their reads count.
static SINK: CacheAligned<AtomicU64> = CacheAligned(AtomicU64::new(0));

/// The buffer's lines, for the workers.
static BASE: AtomicUsize = AtomicUsize::new(0);

static mut STACKS: [Stack<0x4000>; 3] = [const { Stack::new() }; 3];

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let p = unsafe { pac::Peripherals::steal() };
    let mut uart = Uart::init(&p.GPIO, p.UART0);
    let _ = writeln!(uart, "\r\nPANIC on core {}: {info}", core_id());
    halt();
}

/// Sums `lines` with NEON: each iteration loads one cache line as four
/// 16-byte vectors into four independent accumulators.
fn sum(lines: &[Line]) -> u64 {
    // SAFETY: NEON is always present on AArch64, and each `Line` is 64
    // bytes aligned to 64, exactly four aligned `uint64x2_t`s.
    unsafe {
        let mut acc = [vdupq_n_u64(0); 4];
        for line in lines {
            let v: &[uint64x2_t; 4] = &*(line as *const Line).cast();
            acc[0] = vaddq_u64(acc[0], v[0]);
            acc[1] = vaddq_u64(acc[1], v[1]);
            acc[2] = vaddq_u64(acc[2], v[2]);
            acc[3] = vaddq_u64(acc[3], v[3]);
        }
        let total = vaddq_u64(vaddq_u64(acc[0], acc[1]), vaddq_u64(acc[2], acc[3]));
        vgetq_lane_u64::<0>(total).wrapping_add(vgetq_lane_u64::<1>(total))
    }
}

/// Sums the job core `core` was given.
fn run_job(core: usize) -> u64 {
    let job = &JOBS[core].0;
    let base = BASE.load(Ordering::Acquire) as *const Line;
    let (start, chunk, step, count) = (
        job.start.load(Ordering::Acquire),
        job.chunk.load(Ordering::Acquire),
        job.step.load(Ordering::Acquire),
        job.count.load(Ordering::Acquire),
    );
    let mut total = 0u64;
    for k in 0..count {
        // SAFETY: core 0 set `BASE` to the buffer, which lives until the
        // program ends, and every chunk of every job is inside it.
        let lines = unsafe { core::slice::from_raw_parts(base.add(start + k * step), chunk) };
        total = total.wrapping_add(sum(lines));
    }
    total
}

/// Times reading the buffer on `n` cores, each taking every `n`th chunk of
/// `chunk` lines (one chunk each when `chunk` is a whole part), best of
/// [`PASSES`]. Returns the bytes read and the microseconds it took.
fn measure(n: usize, chunk: usize, lines: usize, timer: &Timer) -> (u64, u64) {
    let count = lines / (n * chunk);
    for (core, job) in JOBS.iter().enumerate() {
        let job = &job.0;
        job.start.store(core * chunk, Ordering::Relaxed);
        job.chunk.store(chunk, Ordering::Relaxed);
        job.step.store(n * chunk, Ordering::Relaxed);
        // Workers outside this pass get an empty job.
        job.count
            .store(if core < n { count } else { 0 }, Ordering::Relaxed);
    }

    let mut best = u64::MAX;
    for _ in 0..PASSES {
        DONE.0.store(0, Ordering::Relaxed);
        let started = timer.now_micros();
        GO.0.fetch_add(1, Ordering::Release);
        signal_event();
        SINK.0.fetch_add(run_job(0), Ordering::Relaxed);
        while DONE.0.load(Ordering::Acquire) < 3 {
            core::hint::spin_loop();
        }
        best = best.min(timer.now_micros() - started);
    }
    ((count * chunk * n * 64) as u64, best)
}

/// Prints one measurement as GB/s (10^9 bytes).
fn report(uart: &mut Uart, label: core::fmt::Arguments, (bytes, us): (u64, u64)) {
    // Bytes per microsecond is MB/s; in tenths of a GB/s.
    let tenths = bytes * 10 / us / 1000;
    let _ = writeln!(
        uart,
        "{label}: {} MiB in {us} us, {}.{} GB/s\r",
        bytes / (1024 * 1024),
        tenths / 10,
        tenths % 10,
    );
}

/// Cores 1-3: wait for a pass, sum the job, report done, repeat.
extern "C" fn worker() -> ! {
    let id = core_id();
    let mut seen = 0;
    loop {
        let go = loop {
            let go = GO.0.load(Ordering::Acquire);
            if go != seen {
                break go;
            }
            wait_for_event();
        };
        seen = go;
        SINK.0.fetch_add(run_job(id), Ordering::Relaxed);
        DONE.0.fetch_add(1, Ordering::Release);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kmain() -> ! {
    let p = unsafe { pac::Peripherals::steal() };
    let mut uart = Uart::init(&p.GPIO, p.UART0);
    let timer = Timer::new(p.SYSTMR);
    let mut mailbox = Mailbox::new(p.VCMAILBOX);
    let _ = writeln!(uart, "bandwidth: read throughput, 1 to 4 cores\r");

    if let Ok(max) = mailbox.max_clock_rate_hz(ClockId::Arm) {
        let _ = mailbox.set_clock_rate_hz(ClockId::Arm, max);
    }
    if let Ok(rate) = mailbox.measured_clock_rate_hz(ClockId::Arm) {
        let _ = writeln!(uart, "arm: {} MHz measured\r", rate / 1_000_000);
    }
    if let Err(e) = heap::init(&mut mailbox) {
        let _ = writeln!(uart, "heap: {e:?}\r");
        halt();
    }

    // Filled with distinct values, so every line is real memory the
    // reads have to fetch.
    let lines = BUFFER / 64;
    let mut buffer: Vec<Line> = Vec::with_capacity(lines);
    for i in 0..lines as u64 {
        buffer.push(Line([i; 8]));
    }
    BASE.store(buffer.as_ptr() as usize, Ordering::Release);
    let _ = writeln!(uart, "buffer: {} MiB\r", BUFFER / (1024 * 1024));

    // SAFETY: called once; each stack is a dedicated static handed to one
    // spawn; `worker` never returns and shares only the atomics above and
    // the buffer, which core 0 does not touch while a pass runs.
    unsafe {
        let stacks = &mut *addr_of_mut!(STACKS);
        let [s1, s2, s3] = stacks;
        let cores = Cores::steal();
        cores.core1.spawn(s1, worker);
        cores.core2.spawn(s2, worker);
        cores.core3.spawn(s3, worker);
    }

    // Contiguous parts: each core streams its own quarter (or half...).
    for n in 1..=4usize {
        let result = measure(n, lines / n, lines, &timer);
        report(&mut uart, format_args!("contiguous, {n} core(s)"), result);
    }

    // Interleaved: the cores take turns through the buffer a chunk at a
    // time, so at any moment they read nearby addresses.
    for n in [2usize, 4] {
        for chunk_bytes in [64usize, 1024, 4096, 64 * 1024, 1024 * 1024] {
            let result = measure(n, chunk_bytes / 64, lines, &timer);
            report(
                &mut uart,
                format_args!("interleaved {chunk_bytes} B chunks, {n} cores"),
                result,
            );
        }
    }

    let _ = writeln!(
        uart,
        "bandwidth: done (sink {:x})\r",
        SINK.0.load(Ordering::Relaxed)
    );
    halt();
}
