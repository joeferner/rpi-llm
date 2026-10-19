#![no_std]
#![no_main]

//! rpi-llm: a small language model on a bare-metal Raspberry Pi 5,
//! chatting over the serial console.
//!
//! What runs today is the first piece: the model and its tokenizer read off
//! the SD card into RAM, with a progress line while they load, then checked
//! — the checkpoint's size against what its header implies, the tokenizer
//! parsed to its last byte, and a checksum of each to compare with the
//! host's.

extern crate alloc;

mod load;
mod term;

use rpi_llm_engine::llama2;

use core::fmt::Write;

use rpi_hal::mailbox::Mailbox;
use rpi_hal::sd::{Sd, SdBlockDevice};
use rpi_hal::{halt, pac, timer::Timer, uart::Uart};
use rpi_kickstart::{console, heap, logln, storage};
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
    match llama2::check_tokenizer(tokenizer.bytes(), config.vocab_size) {
        Ok(summary) => logln!(
            "tokenizer: {} tokens, longest {} bytes",
            summary.tokens,
            summary.max_token_length
        ),
        Err(e) => fail(format_args!("{TOKENIZER}: {e:?}")),
    }
    report_checksum(TOKENIZER, tokenizer.bytes(), &timer);

    logln!("rpi-llm: model and tokenizer in RAM");
    halt();
}

/// Logs how long a load took and the rate it ran at.
fn report_load(path: &str, bytes: usize, elapsed_us: u64) {
    let elapsed_us = elapsed_us.max(1);
    // Hundredths of a second, and tenths of a megabyte (10^6 bytes) per
    // second: integer arithmetic on a soft-float build.
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
