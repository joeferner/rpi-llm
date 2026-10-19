#![no_std]
#![no_main]

//! rpi-llm: a small language model on a bare-metal Raspberry Pi 5,
//! chatting over the serial console.
//!
//! What runs today: the model and its tokenizer read off the SD card into
//! RAM, with a progress line while they load, then checked — the
//! checkpoint's size against what its header implies, the tokenizer parsed
//! to its last byte, and a checksum of each to compare with the host's.
//! Then one greedy generation on one core, streamed to the terminal, timed,
//! and compared token for token with what llama2.c produced from the same
//! prompt.

extern crate alloc;

mod load;
mod term;

use alloc::vec::Vec;
use core::fmt::Write;

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::sd::{Sd, SdBlockDevice};
use rpi_hal::{halt, pac, timer::Timer, uart::Uart};
use rpi_kickstart::{console, heap, logln, storage};
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::llama2;
use rpi_llm_engine::model::{Transformer, Weights};
use rpi_llm_engine::tokenizer::{Tokenizer, printable};
use static_cell::StaticCell;

/// The checkpoint, on the card.
const MODEL: &str = "models/stories15M.bin";
/// Its tokenizer, on the card.
const TOKENIZER: &str = "models/tokenizer.bin";

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

    let (model, elapsed) = match load::load(&mut volume, MODEL, &timer) {
        Ok(loaded) => loaded,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    report_load(MODEL, model.bytes().len(), elapsed);
    let config = match llama2::Config::parse(model.bytes()) {
        Ok(config) => config,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    logln!(
        "model: dim {} hidden {} layers {} heads {} kv heads {} vocab {} context {}{}",
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.seq_len,
        if config.shared_classifier {
            ", shared classifier"
        } else {
            ""
        },
    );
    logln!(
        "model: {} weights, the size the header implies",
        config.weight_count()
    );
    report_checksum(MODEL, model.bytes(), &timer);

    let (tokenizer, elapsed) = match load::load(&mut volume, TOKENIZER, &timer) {
        Ok(loaded) => loaded,
        Err(e) => fail(format_args!("{TOKENIZER}: {e:?}")),
    };
    report_load(TOKENIZER, tokenizer.bytes().len(), elapsed);
    report_checksum(TOKENIZER, tokenizer.bytes(), &timer);
    let tokenizer = match Tokenizer::new(tokenizer.bytes(), config.vocab_size) {
        Ok(tokenizer) => tokenizer,
        Err(e) => fail(format_args!("{TOKENIZER}: {e:?}")),
    };
    logln!("tokenizer: {} tokens", config.vocab_size);

    let weights = match Weights::new(&config, model.bytes()) {
        Ok(weights) => weights,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    let mut transformer = Transformer::new(config, weights);

    full_speed(&mut mailbox);

    let ids = generate(&mut transformer, &tokenizer, PROMPT, &timer);
    check_against_reference(&ids);

    logln!("rpi-llm: done");
    halt();
}

/// What the model is asked to continue.
const PROMPT: &str = "Once upon a time";

/// Positions to run, prompt included: `run.c`'s `-n 256`.
const STEPS: usize = 256;

/// The tokens llama2.c's `run.c` produced for [`PROMPT`] greedily, the
/// same reference the engine's host test checks against.
const REFERENCE: &str = include_str!("../engine/tests/data/once_upon_a_time.ids");

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

/// Feeds `prompt` through the model and generates greedily from it,
/// streaming the text to the terminal as each token arrives. Returns the
/// tokens emitted, prompt tokens after `<s>` included, as `run.c` prints
/// them.
fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    timer: &Timer,
) -> Vec<u32> {
    let mut generation = Generation::new(tokenizer.encode(prompt, true, false), STEPS);
    logln!(
        "generate: \"{prompt}\", {} prompt tokens, greedy, up to {STEPS} positions",
        generation.prompt_len()
    );

    let mut ids = Vec::new();
    let started = timer.now_micros();
    // When the first generated token arrived, and how many have since:
    // the rate is over generated tokens only, as `run.c`'s is, so the
    // prompt's forced tokens do not inflate it.
    let mut first_generated: Option<u64> = None;
    let mut generated = 0u32;
    while let Some(step) = generation.step(transformer) {
        ids.push(step.next);
        let piece = tokenizer.decode(step.prev, step.next);
        if printable(piece) {
            term::write_bytes(piece);
        }
        if step.generated {
            match first_generated {
                None => first_generated = Some(timer.now_micros()),
                Some(_) => generated += 1,
            }
        }
    }
    let ended = timer.now_micros();
    term::write_bytes(b"\n");

    let total_s = (ended - started) as f32 / 1e6;
    logln!("generate: {} tokens in {total_s:.2} s", ids.len());
    if let Some(first) = first_generated
        && generated > 0
    {
        let rate = generated as f32 / ((ended - first) as f32 / 1e6);
        logln!("generate: {rate:.1} tok/s");
    }
    ids
}

/// Logs whether `ids` is the token sequence llama2.c produced.
fn check_against_reference(ids: &[u32]) {
    let mut expected = REFERENCE.lines().map(|line| line.parse::<u32>().unwrap());
    let mut count = 0;
    for (i, &id) in ids.iter().enumerate() {
        match expected.next() {
            Some(want) if want == id => count += 1,
            Some(want) => {
                logln!("reference: token {i} differs: {id}, llama2.c has {want}");
                return;
            }
            None => {
                logln!("reference: {} tokens, llama2.c stopped at {i}", ids.len());
                return;
            }
        }
    }
    let remaining = expected.count();
    if remaining == 0 {
        logln!("reference: all {count} tokens match llama2.c");
    } else {
        logln!("reference: stopped {remaining} tokens before llama2.c did");
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

/// Logs the FNV-1a checksum of a loaded file, and how long it took.
fn report_checksum(path: &str, bytes: &[u8], timer: &Timer) {
    let started = timer.now_micros();
    let hash = load::fnv1a(bytes);
    let elapsed_ms = (timer.now_micros() - started) / 1000;
    logln!("{path}: fnv1a {hash:016x} ({elapsed_ms} ms)");
}
