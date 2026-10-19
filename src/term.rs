//! The serial terminal: one UART shared by the console's stamped lines and
//! the output that is not whole lines — the progress line now, streamed
//! tokens later.
//!
//! rpi-kickstart's console takes a sink and writes one complete line per
//! call, which is right for a log and wrong for a terminal UI: a progress
//! bar redraws one line in place, and a reply arrives a token at a time.
//! So the UART lives here behind a lock, the console is handed [`Sink`],
//! which writes through the same lock, and [`write`] is the way to put
//! anything else on the wire. Holding the lock for a whole write is what
//! keeps a log line from landing in the middle of a redraw.

use core::cell::RefCell;
use core::fmt::{self, Write};

use critical_section::Mutex;
use rpi_hal::uart::Uart;

/// The terminal's UART, once [`init`] has installed it.
static UART: Mutex<RefCell<Option<Uart>>> = Mutex::new(RefCell::new(None));

/// Installs `uart` as the terminal. Call once, before anything writes:
/// writes before then are dropped.
pub fn init(uart: Uart) {
    critical_section::with(|cs| *UART.borrow_ref_mut(cs) = Some(uart));
}

/// Writes `args` to the terminal as they are, with no stamp and no line
/// ending.
pub fn write(args: fmt::Arguments) {
    with(|uart| {
        let _ = uart.write_fmt(args);
    });
}

/// Writes raw bytes to the terminal, with each `\n` sent as `\r\n`.
///
/// For text that is not known to be whole UTF-8 — a token can be half of a
/// multi-byte character, the other half arriving with the next one — and
/// that uses bare newlines, as the model's output does. A serial terminal
/// does not translate line endings, so a bare `\n` there moves down a
/// line without returning to the start of it.
pub fn write_bytes(bytes: &[u8]) {
    with(|uart| {
        for &byte in bytes {
            if byte == b'\n' {
                uart.write_byte(b'\r');
            }
            uart.write_byte(byte);
        }
    });
}

/// Runs `f` on the terminal's UART with the lock held, so that several
/// writes go out together. Does nothing before [`init`].
fn with(f: impl FnOnce(&mut Uart)) {
    critical_section::with(|cs| {
        if let Some(uart) = UART.borrow_ref_mut(cs).as_mut() {
            f(uart);
        }
    });
}

/// The sink rpi-kickstart's console writes its lines to: the terminal.
pub struct Sink;

impl Write for Sink {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write(format_args!("{s}"));
        Ok(())
    }
}

/// A line redrawn in place to show a transfer's progress.
///
/// Each [`update`](Self::update) rewrites the line with a carriage return,
/// so the terminal shows one line that fills up rather than a column of
/// them; [`finish`](Self::finish) leaves it complete and moves on.
///
/// Redraws are rate-limited. The UART is the slowest thing in the loop it
/// reports on: one redraw is about 80 characters, 7 ms at 115200 baud, so
/// redrawing on every megabyte read would spend a noticeable share of the
/// load printing about it.
pub struct Progress<'a> {
    /// What is being transferred, shown at the start of the line.
    label: &'a str,
    /// The total, in bytes.
    total: u64,
    /// When the transfer started, in System Timer microseconds.
    started: u64,
    /// When the line was last drawn, in System Timer microseconds.
    drawn: Option<u64>,
}

/// How long [`Progress::update`] waits between redraws, in microseconds.
const REDRAW_US: u64 = 200_000;

/// Width of the bar, in characters.
const BAR: u64 = 24;

impl<'a> Progress<'a> {
    /// Starts reporting a transfer of `total` bytes, starting `now`.
    pub fn new(label: &'a str, total: u64, now: u64) -> Self {
        Self {
            label,
            total,
            started: now,
            drawn: None,
        }
    }

    /// Reports `done` bytes transferred by `now`, redrawing the line if the
    /// last redraw was long enough ago.
    pub fn update(&mut self, done: u64, now: u64) {
        if self.drawn.is_some_and(|at| now - at < REDRAW_US) {
            return;
        }
        self.drawn = Some(now);
        self.draw(done, now);
    }

    /// Draws the completed line, ends it, and returns the elapsed time in
    /// microseconds.
    pub fn finish(self, now: u64) -> u64 {
        self.draw(self.total, now);
        write(format_args!("\r\n"));
        now - self.started
    }

    /// Writes the line for `done` bytes at `now`, under one hold of the
    /// lock.
    ///
    /// All integer arithmetic, since nothing here needs more than tenths.
    fn draw(&self, done: u64, now: u64) {
        let elapsed = (now - self.started).max(1);
        let filled = (done * BAR).checked_div(self.total).unwrap_or(BAR);
        let percent = (done * 100).checked_div(self.total).unwrap_or(100);
        // Tenths of a megabyte (10^6 bytes) per second.
        let rate = done * 10 / elapsed;
        // Whole seconds left at the rate so far; unknown before any bytes.
        let left = (self.total - done)
            .checked_mul(elapsed)
            .and_then(|scaled| scaled.checked_div(done))
            .map(|us| us.div_ceil(1_000_000));

        with(|uart| {
            let _ = write!(
                uart,
                "\r{}: {} / {} MiB [",
                self.label,
                Tenths(done * 10 / MIB),
                Tenths(self.total * 10 / MIB),
            );
            for i in 0..BAR {
                let _ = uart.write_char(if i < filled { '#' } else { '.' });
            }
            let _ = write!(uart, "] {percent:>3}%  {} MB/s", Tenths(rate));
            let _ = match left {
                Some(seconds) if done < self.total => write!(uart, "  {seconds} s left "),
                // Padding over whatever an earlier, longer draw left behind.
                _ => uart.write_str("            "),
            };
        });
    }
}

/// One mebibyte.
const MIB: u64 = 1024 * 1024;

/// A count of tenths, displayed as a decimal with one place.
struct Tenths(u64);

impl fmt::Display for Tenths {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.0 / 10, self.0 % 10)
    }
}
