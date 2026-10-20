#![no_std]
#![no_main]

//! How fast the prompt's matrix multiply kernel (`quant::gemm4`) runs on
//! the Pi 5, alone, in cycles per 4x4 tile: four rows by four tokens over
//! one 32-value block, 32 `sdot`s and their scaling.
//!
//! Its instruction count puts the floor at about 25 cycles a tile (two
//! vector pipes, about 49 vector instructions), and the model's prompt
//! phase timing works out to about 38. The cases split that difference:
//!
//! - **L1-hot**: one quad of rows and eight tokens, a few KiB, over and
//!   over. What the kernel itself costs.
//! - **One core, real shapes**: SmolLM2's 2560x960 and 960x2560 matrices,
//!   with 8 tokens (the tokens fit in L1) and with 44 (the prompt's 41
//!   padded; for 2560 columns that is 127 KB of tokens, twice L1).
//! - **Four cores**: the same, split as the model splits them, a quad of
//!   rows at a time in turn.
//!
//! The weights are random but laid out exactly as a repacked model's, and
//! the tokens are quantized by the engine's own `quantize_batch`.
//!
//! ```sh
//! make kernel DEVICE=/dev/ttyUSB0
//! ```

extern crate alloc;

#[allow(
    dead_code,
    reason = "the board program's pool, of which this uses a part"
)]
#[path = "../src/cores.rs"]
mod cores;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write;
use core::hint::black_box;

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::{cpu::core_id, halt, pac, uart::Uart};
use rpi_kickstart::heap;
use rpi_llm_engine::model::Clock;
use rpi_llm_engine::parallel::{Serial, Workers};
use rpi_llm_engine::quant::{
    BLOCK, Batch, LANES, QUAD, Quad, QuadScales, f32_to_f16, gemm4, quantize_batch,
};

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let p = unsafe { pac::Peripherals::steal() };
    let mut uart = Uart::init(&p.GPIO, p.UART0);
    let _ = writeln!(uart, "\r\nPANIC on core {}: {info}", core_id());
    halt();
}

/// A pseudo-random stream, so the weights and tokens are not all alike.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

/// A repacked Q8_0 matrix: `rows / 4` quads of `cols / 32` blocks each.
struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<Quad>,
    scales: Vec<QuadScales>,
}

impl Matrix {
    fn random(rows: usize, cols: usize, noise: &mut Noise) -> Self {
        let n = rows / QUAD * (cols / BLOCK);
        let mut values = vec![Quad([0; QUAD * BLOCK]); n];
        for quad in &mut values {
            for v in &mut quad.0 {
                *v = noise.next() as i8;
            }
        }
        let scale = f32_to_f16(0.01);
        Self {
            rows,
            cols,
            values,
            scales: vec![QuadScales([scale; QUAD]); n],
        }
    }

    /// Quad `q`'s blocks and scales.
    fn quad(&self, q: usize) -> (&[Quad], &[QuadScales]) {
        let per = self.cols / BLOCK;
        (
            &self.values[q * per..(q + 1) * per],
            &self.scales[q * per..(q + 1) * per],
        )
    }
}

/// `tokens` random inputs of `cols` values, quantized as the model does.
fn batch(tokens: usize, cols: usize, noise: &mut Noise) -> Batch {
    let xs: Vec<f32> = (0..tokens * cols)
        .map(|_| noise.next() as f32 / (1 << 24) as f32 - 0.5)
        .collect();
    let mut out = Batch::default();
    quantize_batch(&Serial, &xs, cols, &mut out);
    out
}

/// Multiplies every quad of `m` in `quads` (each `step`th from `first`)
/// with `x`.
fn run(m: &Matrix, x: &Batch, first: usize, step: usize) {
    let mut out = [[0.0f32; QUAD]; 64];
    for q in (first..m.rows / QUAD).step_by(step) {
        let (values, scales) = m.quad(q);
        gemm4(values, scales, black_box(x), &mut out);
        black_box(&out);
    }
}

/// What one case printed needs: how long `reps` calls took, in counter
/// ticks, and how many 4x4 tiles each call computed.
fn report(uart: &mut Uart, label: &str, ticks: u64, reps: u64, tiles: u64, cores: u64, mhz: u64) {
    let rate = cores::Counter.ticks_per_second();
    let ns = ticks * 1_000_000_000 / rate / reps;
    // Each core's cycles, over its share of the tiles.
    let cycles_x10 = ticks * mhz * 1_000_000 / rate * cores * 10 / (reps * tiles);
    let _ = writeln!(
        uart,
        "{label}: {}.{:03} us a call, {}.{} cycles a tile per core\r",
        ns / 1000,
        ns % 1000,
        cycles_x10 / 10,
        cycles_x10 % 10,
    );
}

/// The best of three timings of `f` repeated `reps` times, in ticks.
fn time(reps: u64, f: impl Fn()) -> u64 {
    let mut best = u64::MAX;
    for _ in 0..3 {
        let start = cores::Counter.ticks();
        for _ in 0..reps {
            f();
        }
        best = best.min(cores::Counter.ticks() - start);
    }
    best
}

#[unsafe(no_mangle)]
pub extern "C" fn kmain() -> ! {
    let p = unsafe { pac::Peripherals::steal() };
    let mut uart = Uart::init(&p.GPIO, p.UART0);
    let mut mailbox = Mailbox::new(p.VCMAILBOX);
    let _ = writeln!(uart, "kernel: gemm4, cycles per 4x4 tile\r");

    if let Ok(max) = mailbox.max_clock_rate_hz(ClockId::Arm) {
        let _ = mailbox.set_clock_rate_hz(ClockId::Arm, max);
    }
    let mhz = mailbox
        .measured_clock_rate_hz(ClockId::Arm)
        .map_or(2400, |hz| u64::from(hz) / 1_000_000);
    let _ = writeln!(uart, "arm: {mhz} MHz measured\r");
    if let Err(e) = heap::init(&mut mailbox) {
        let _ = writeln!(uart, "heap: {e:?}\r");
        halt();
    }

    let mut noise = Noise(1);
    let tiles =
        |m: &Matrix, x: &Batch| (m.rows / QUAD * (m.cols / BLOCK) * (x.tokens() / LANES)) as u64;

    // The kernel alone: one quad, eight tokens, all in L1.
    let hot = Matrix::random(QUAD, 960, &mut noise);
    let x = batch(8, 960, &mut noise);
    let reps = 20_000;
    let ticks = time(reps, || run(&hot, &x, 0, 1));
    report(
        &mut uart,
        "L1-hot, 1 quad x 8 tokens, 960 columns",
        ticks,
        reps,
        tiles(&hot, &x),
        1,
        mhz,
    );

    let up = Matrix::random(2560, 960, &mut noise);
    let down = Matrix::random(960, 2560, &mut noise);

    // One core over whole matrices.
    for m in [&up, &down] {
        for tokens in [8, 44] {
            let x = batch(tokens, m.cols, &mut noise);
            let reps = 20;
            let ticks = time(reps, || run(m, &x, 0, 1));
            let label = format!("1 core, {}x{}, {tokens} tokens", m.rows, m.cols);
            report(&mut uart, &label, ticks, reps, tiles(m, &x), 1, mhz);
        }
    }

    // SAFETY: called once, before anything else uses the secondary cores.
    let pool = unsafe { cores::Pool::start() };
    for m in [&up, &down] {
        for tokens in [8, 44] {
            let x = batch(tokens, m.cols, &mut noise);
            let reps = 50;
            let ticks = time(reps, || pool.run(&|w| run(m, &x, w, 4)));
            let label = format!("4 cores, {}x{}, {tokens} tokens", m.rows, m.cols);
            report(&mut uart, &label, ticks, reps, tiles(m, &x), 4, mhz);
        }
    }

    let _ = writeln!(uart, "kernel: done\r");
    halt();
}
