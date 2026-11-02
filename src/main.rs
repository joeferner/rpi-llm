#![no_std]
#![no_main]

//! rpi-llm: a small language model on a bare-metal Raspberry Pi 5,
//! chatting over the serial console.
//!
//! SmolLM2-360M-Instruct, a Q8_0 GGUF file, is read off the SD card into
//! RAM with a progress line while it loads, checked against the host's
//! checksum, and parsed. If an NVMe SSD on the external PCIe connector has
//! the same file on its first FAT volume, it is read from there too, timed
//! and checked the same way, and that copy is the one used ([`ssd`]), so
//! the two loads can be compared. Then the chat: type a message at the terminal and
//! the reply streams back, greedily chosen on all four cores, with how
//! fast it came after it ([`chat`]).
//!
//! With the `benchmark` feature (`make benchmark`), one fixed request runs
//! first, timed and compared token for token with what the same engine
//! produced on the host ([`benchmark`]).

extern crate alloc;

#[cfg(feature = "benchmark")]
mod benchmark;
mod chat;
mod cores;
mod load;
mod ssd;
mod term;

use core::fmt::Write;

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::sd::{Sd, SdBlockDevice};
use rpi_hal::{halt, pac, timer::Timer, uart::Uart};
use rpi_kickstart::{console, heap, logln, storage};
use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::gguf::Gguf;
use rpi_llm_engine::model::{Transformer, Weights};
use static_cell::StaticCell;

/// The model, on the card.
const MODEL: &str = "models/smollm2-360m-instruct-q8_0.gguf";

/// The model file's FNV-1a, computed on the host over the file `make
/// models` fetches (its SHA-256 is in `models.sha256`).
const MODEL_FNV1A: u64 = 0x25b4_dd6c_3e5d_a561;

/// Positions the key/value cache holds. SmolLM2 was trained for 8192, but
/// the cache costs 80 KiB a position; 2048 is 160 MiB.
const CONTEXT: usize = 2048;

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

    logln!("loading from the SD card");
    let (mut file, elapsed) = match load::load(&mut volume, MODEL, &timer) {
        Ok(loaded) => loaded,
        Err(e) => fail(format_args!("{MODEL}: {e:?}")),
    };
    report_load(MODEL, file.bytes().len(), elapsed);
    report_checksum(MODEL, file.bytes(), &timer);

    // The same file off the SSD, if there is one, for the comparison. Its
    // copy replaces the card's only once it has loaded in full; anything
    // short of that leaves the card's in place.
    if let Some(mut ssd_volume) = ssd::mount(peripherals.PCIE1, &timer) {
        logln!("loading from the SSD");
        match load::load(&mut ssd_volume, MODEL, &timer) {
            Ok((ssd_file, elapsed)) => {
                report_load(MODEL, ssd_file.bytes().len(), elapsed);
                report_checksum(MODEL, ssd_file.bytes(), &timer);
                file = ssd_file;
            }
            Err(e) => logln!("ssd: {MODEL}: {e:?}"),
        }
        ssd::shut_down(ssd_volume, &timer);
    }

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

    #[cfg(feature = "benchmark")]
    benchmark::run(&mut transformer, &bpe, &timer, &mut mailbox);

    chat::run(&mut transformer, &bpe, &timer)
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
