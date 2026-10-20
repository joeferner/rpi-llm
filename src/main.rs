#![no_std]
#![no_main]

//! rpi-llm: a small language model on a bare-metal Raspberry Pi 5,
//! chatting over the serial console.
//!
//! What runs today: SmolLM2-360M-Instruct, a Q8_0 GGUF file, read off the
//! SD card into RAM with a progress line while it loads, checked against
//! the host's checksum, and parsed. Then one chat turn on one core — a
//! fixed request, the model's ChatML template around it — answered
//! greedily, streamed to the terminal, timed, and compared token for token
//! with what the same engine produced on the host.

extern crate alloc;

mod cores;
mod load;
mod term;

use alloc::vec::Vec;
use core::fmt::Write;

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::sd::{Sd, SdBlockDevice};
use rpi_hal::{halt, pac, timer::Timer, uart::Uart};
use rpi_kickstart::{console, heap, logln, storage};
use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::gguf::Gguf;
use rpi_llm_engine::model::{Clock, Phases, Transformer, Weights};
use static_cell::StaticCell;

/// The model, on the card.
const MODEL: &str = "models/smollm2-360m-instruct-q8_0.gguf";

/// The model file's FNV-1a, computed on the host over the file `make
/// models` fetches (its SHA-256 is in `models.sha256`).
const MODEL_FNV1A: u64 = 0x25b4_dd6c_3e5d_a561;

/// Positions the key/value cache holds. SmolLM2 was trained for 8192, but
/// the cache costs 80 KiB a position; 2048 is 160 MiB.
const CONTEXT: usize = 2048;

/// The chat turn: the model's ChatML template around one request, the same
/// prompt the engine's host tests use.
const PROMPT: &str = include_str!("../engine/tests/data/smollm2_story.prompt");

/// How many tokens the answer may run to.
const ANSWER_TOKENS: usize = 200;

/// What the engine generated from [`PROMPT`] on the host. The arithmetic is
/// IEEE `f32` with nothing fused and the same `libm` on both machines, so
/// the board has to produce exactly this.
const REFERENCE: &str = include_str!("../engine/tests/data/smollm2_story.engine.ids");

// The console borrows its sink for `'static`.
static SINK: StaticCell<term::Sink> = StaticCell::new();

/// The clock the console stamps each line with, in microseconds.
///
/// The System Timer's counters are read-only and free-running, so
/// stealing the peripheral here disturbs nothing.
fn uptime() -> u64 {
    Timer::new(unsafe { pac::Peripherals::steal() }.SYSTMR).now_micros()
}

// A program has exactly one panic handler, so it's the application's.
// It writes to a fresh UART rather than through the terminal: whatever
// held the terminal's lock may be what just panicked.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let peripherals = unsafe { pac::Peripherals::steal() };
    let mut uart = Uart::init(&peripherals.GPIO, peripherals.UART0);
    let _ = writeln!(uart, "\r\nPANIC: {info}");
    halt();
}

/// Stops with `what` on the console. For the failures there is no going
/// on from, which at this stage is every one of them.
fn fail(what: core::fmt::Arguments) -> ! {
    logln!("{what}");
    halt();
}

/// Where rpi-hal's boot code hands over, once the stack and `.bss` are
/// set up.
#[unsafe(no_mangle)]
pub extern "C" fn kmain() -> ! {
    let peripherals = unsafe { pac::Peripherals::steal() };
    let timer = Timer::new(peripherals.SYSTMR);
    let mut mailbox = Mailbox::new(peripherals.VCMAILBOX);

    term::init(Uart::init(&peripherals.GPIO, peripherals.UART0));
    console::init(SINK.init(term::Sink), uptime);
    logln!("rpi-llm: starting on the Raspberry Pi 5");

    match heap::init(&mut mailbox) {
        Ok(bytes) => logln!("heap: {} MiB", bytes / (1024 * 1024)),
        Err(e) => fail(format_args!("heap: {e:?}")),
    }

    // `steal_emmc` names whichever controller this chip's card slot is on.
    let emmc = unsafe { Sd::steal_emmc() };
    let sd = match Sd::init(&peripherals.GPIO, emmc, &mut mailbox, &timer) {
        Ok(sd) => sd,
        Err(e) => fail(format_args!("sd: {e:?}")),
    };
    let mut volume = match storage::mount(SdBlockDevice::new(sd, &timer)) {
        Ok(volume) => volume,
        Err(e) => fail(format_args!("storage: {e:?}")),
    };

    let (mut file, elapsed) = match load::load(&mut volume, MODEL, &timer) {
        Ok(loaded) => loaded,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    report_load(MODEL, file.bytes().len(), elapsed);
    report_checksum(MODEL, file.bytes(), &timer);

    let started = timer.now_micros();
    let gguf = match Gguf::parse_repacked(file.bytes_mut()) {
        Ok(gguf) => gguf,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    logln!(
        "model: Q8_0 tensors repacked in {} ms",
        (timer.now_micros() - started) / 1000
    );
    let bpe = match Bpe::from_gguf(&gguf) {
        Ok(bpe) => bpe,
        Err(e) => fail(format_args!("{MODEL}: tokenizer: {e:?}")),
    };
    let (config, weights) = match Weights::from_gguf(&gguf) {
        Ok(model) => model,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    logln!(
        "model: {} layers, dim {}, hidden {}, {} heads / {} kv heads, vocab {}, rope base {}",
        config.n_layers,
        config.dim,
        config.hidden_dim,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.rope_theta,
    );
    let mut transformer = Transformer::new(config, weights, CONTEXT);
    logln!("model: context {} positions", transformer.context());

    // SAFETY: the only call; nothing else here uses the secondary cores.
    let pool = unsafe { cores::Pool::start() };
    transformer.set_workers(pool);
    transformer.set_clock(&cores::Counter);
    logln!("cores: matrix multiplies split across 4 cores");

    full_speed(&mut mailbox);

    cores::reset_stats();
    let started = timer.now_micros();
    let ids = chat(&mut transformer, &bpe, &timer);
    let profile = transformer.profile();
    let rate = cores::Counter.ticks_per_second();
    report_phases("prompt", &profile.prefill, rate);
    report_phases("answer", &profile.forward, rate);
    let total_ms = (timer.now_micros() - started) / 1000;
    let (per_core, jobs) = cores::stats();
    logln!(
        "cores: {jobs} jobs in {total_ms} ms; core 0 worked {} ms, waited {} ms for the others",
        per_core[0].0 / 1000,
        per_core[0].1 / 1000
    );
    for (id, (work, wake)) in per_core.iter().enumerate().skip(1) {
        logln!(
            "cores: core {id} worked {} ms, spent {} ms waking up",
            work / 1000,
            wake / 1000
        );
    }
    check_against_reference(&ids);

    logln!("rpi-llm: done");
    halt();
}

/// Raises the ARM to the fastest clock the firmware offers, and logs what
/// it actually runs at.
///
/// It boots at 1.5 GHz of a possible 2.4. Only the measured rate is
/// trusted: the firmware's own report of a clock is what it set, which is
/// not always what is running.
fn full_speed(mailbox: &mut Mailbox) {
    let max = match mailbox.max_clock_rate_hz(ClockId::Arm) {
        Ok(max) => max,
        Err(e) => {
            logln!("arm: no maximum clock from the firmware: {e:?}");
            return;
        }
    };
    if let Err(e) = mailbox.set_clock_rate_hz(ClockId::Arm, max) {
        logln!("arm: setting {} MHz failed: {e:?}", max / 1_000_000);
    }
    match mailbox.measured_clock_rate_hz(ClockId::Arm) {
        Ok(rate) => logln!(
            "arm: {} MHz measured (max {} MHz)",
            rate / 1_000_000,
            max / 1_000_000
        ),
        Err(e) => logln!("arm: no measured clock: {e:?}"),
    }
}

/// Answers [`PROMPT`] greedily, streaming the answer to the terminal as
/// each token arrives, and returns the answer's tokens.
fn chat(transformer: &mut Transformer, bpe: &Bpe, timer: &Timer) -> Vec<u32> {
    let prompt = bpe.encode(PROMPT);
    let prompt_len = prompt.len();
    // Positions, not tokens: the last prompt position already produces the
    // first answer token.
    let steps = prompt_len + ANSWER_TOKENS - 1;
    let mut generation = Generation::new(prompt, steps, &[bpe.eos()]);
    logln!("chat: {prompt_len} prompt tokens, greedy, up to {ANSWER_TOKENS} answer tokens");
    term::write_bytes(PROMPT.as_bytes());

    let mut ids = Vec::new();
    let mut piece = Vec::new();
    let started = timer.now_micros();
    // The prompt, all but its last token, through the model together.
    generation.prefill(transformer);
    // When the first answer token arrived: the prompt's positions before it
    // and the answer's after it are timed separately, since a chat's
    // latency is the one and its speed the other.
    let mut first: Option<u64> = None;
    while let Some(step) = generation.step(transformer) {
        if !step.generated {
            continue;
        }
        first.get_or_insert_with(|| timer.now_micros());
        ids.push(step.next);
        piece.clear();
        bpe.decode(step.next, &mut piece);
        term::write_bytes(&piece);
    }
    let ended = timer.now_micros();
    term::write_bytes(b"\n");

    if let Some(first) = first {
        let prompt_s = (first - started) as f32 / 1e6;
        logln!(
            "chat: prompt {prompt_len} tokens in {prompt_s:.2} s, {:.1} tok/s",
            prompt_len as f32 / prompt_s
        );
        if ids.len() > 1 {
            let answer_s = (ended - first) as f32 / 1e6;
            logln!(
                "chat: answer {} tokens in {answer_s:.2} s, {:.1} tok/s",
                ids.len(),
                (ids.len() - 1) as f32 / answer_s
            );
        }
    }
    ids
}

/// Logs where the engine's calls spent their time: `prompt` for the
/// prefill batches, `answer` for the single-token forwards (which include
/// the prompt's last token).
fn report_phases(name: &str, p: &Phases, rate: u64) {
    let ms = |ticks: u64| ticks as f32 * 1000.0 / rate as f32;
    logln!(
        "phases: {name}: {} calls in {:.1} ms: quantize {:.1}, matmul {:.1}, attention {:.1}, other {:.1}",
        p.calls,
        ms(p.total),
        ms(p.quantize),
        ms(p.matmul),
        ms(p.attention),
        ms(p.other()),
    );
}

/// Logs whether `ids` is the answer the engine produced on the host.
fn check_against_reference(ids: &[u32]) {
    let expected: Vec<u32> = REFERENCE
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    match ids.iter().zip(&expected).position(|(a, b)| a != b) {
        Some(i) => logln!(
            "reference: token {i} differs: {}, the host engine has {}",
            ids[i],
            expected[i]
        ),
        None if ids.len() == expected.len() => {
            logln!("reference: all {} tokens match the host engine", ids.len());
        }
        None => logln!(
            "reference: {} tokens where the host engine has {}",
            ids.len(),
            expected.len()
        ),
    }
}

/// Logs how long a load took and the rate it ran at.
fn report_load(path: &str, bytes: usize, elapsed_us: u64) {
    let elapsed_us = elapsed_us.max(1);
    // Hundredths of a second, and tenths of a megabyte (10^6 bytes) per
    // second.
    let centis = elapsed_us / 10_000;
    let rate = bytes as u64 * 10 / elapsed_us;
    logln!(
        "{path}: {bytes} bytes in {}.{:02} s, {}.{} MB/s",
        centis / 100,
        centis % 100,
        rate / 10,
        rate % 10,
    );
}

/// Logs the FNV-1a checksum of a loaded file against the host's, and how
/// long it took.
fn report_checksum(path: &str, bytes: &[u8], timer: &Timer) {
    let started = timer.now_micros();
    let hash = load::fnv1a(bytes);
    let elapsed_ms = (timer.now_micros() - started) / 1000;
    let verdict = if hash == MODEL_FNV1A {
        "matches the host"
    } else {
        "DIFFERS from the host"
    };
    logln!("{path}: fnv1a {hash:016x}, {verdict} ({elapsed_ms} ms)");
}
