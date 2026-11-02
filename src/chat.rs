//! The chat: read a line at the terminal, stream the model's reply, show
//! how fast it came, repeat.
//!
//! The conversation is the engine's [`Chat`], which keeps it in the
//! key/value cache between turns, so a reply costs its own message rather
//! than everything said so far. Each reply ends with a footer:
//!
//! ```text
//! [<reply tokens> tok, <reply rate> tok/s, prompt <rate> tok/s, context <used>/2048]
//! ```
//!
//! — the reply's length and rate, how fast its prompt (this turn's new
//! tokens) went through, and how much of the cache the conversation fills.

use alloc::string::String;
use alloc::vec::Vec;

use rpi_hal::timer::Timer;
use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::chat::{self, Chat, ChatError, Stop};
use rpi_llm_engine::model::Transformer;

use crate::term;

/// The longest a reply may run, in tokens.
const MAX_REPLY: usize = 512;

/// Ctrl-C.
const INTERRUPT: u8 = 0x03;

/// Chats at the terminal, forever.
pub fn run(transformer: &mut Transformer, bpe: &Bpe, timer: &Timer) -> ! {
    let mut chat = Chat::new(chat::SYSTEM);
    let mut input = LineReader::default();
    let mut line = String::new();
    let mut piece = Vec::new();
    term::write(format_args!(
        "\r\nChat with SmolLM2-360M-Instruct. Type a message and press Enter;\r\n\
         Ctrl-C stops a reply, /reset starts a new conversation.\r\n"
    ));

    loop {
        term::write(format_args!("\r\n> "));
        input.read_line(&mut line);
        let message = line.trim();
        match message {
            "" => continue,
            "/reset" => {
                chat.reset();
                term::write(format_args!("(a new conversation)\r\n"));
                continue;
            }
            "/help" => {
                term::write(format_args!(
                    "Enter sends; Backspace, Ctrl-U and Ctrl-C edit the line.\r\n\
                     Ctrl-C while a reply is coming stops it.\r\n\
                     /reset starts a new conversation.\r\n"
                ));
                continue;
            }
            _ => {}
        }

        let started = timer.now_micros();
        let mut first = None;
        let reply = chat.reply(transformer, bpe, message, MAX_REPLY, |token| {
            first.get_or_insert_with(|| timer.now_micros());
            piece.clear();
            bpe.decode(token, &mut piece);
            term::write_bytes(&piece);
            // Anything else typed while the reply comes is dropped: the
            // prompt above has already been taken.
            term::try_read_byte() != Some(INTERRUPT)
        });
        let ended = timer.now_micros();

        let reply = match reply {
            Ok(reply) => reply,
            Err(ChatError::TooLong { tokens, context }) => {
                term::write(format_args!(
                    "(too long: {tokens} tokens, and the context holds {context})\r\n"
                ));
                continue;
            }
        };
        if reply.restarted {
            // Said after the reply, since it is known only once the turn
            // has run; the reply itself is unaffected.
            term::write(format_args!(
                "\r\n(the context was full: started a new conversation)"
            ));
        }
        match reply.stop {
            Stop::End => {}
            Stop::Limit => term::write(format_args!(" ...(cut off at {MAX_REPLY} tokens)")),
            Stop::Interrupted => term::write(format_args!(" ^C")),
            Stop::ContextFull => term::write(format_args!(
                "\r\n(the context is full: the next message starts a new conversation)"
            )),
        }
        footer(&reply, started, first, ended, &chat, transformer.context());
    }
}

/// Writes the line after a reply: its length and rate, the prompt's rate,
/// and how full the cache is.
fn footer(
    reply: &chat::Reply,
    started: u64,
    first: Option<u64>,
    ended: u64,
    chat: &Chat,
    context: usize,
) {
    let tenths = |count: usize, us: u64| count as u64 * 10_000_000 / us.max(1);
    let rate = |count: usize, us: u64| {
        let t = tenths(count, us);
        (t / 10, t % 10)
    };
    term::write(format_args!("\r\n["));
    term::write(format_args!("{} tok", reply.reply_tokens));
    if let Some(first) = first {
        // The first token comes with the prompt; the rate is the ones after.
        if reply.reply_tokens > 1 {
            let (whole, tenth) = rate(reply.reply_tokens - 1, ended - first);
            term::write(format_args!(", {whole}.{tenth} tok/s"));
        }
        let (whole, _) = rate(reply.prompt_tokens, first - started);
        term::write(format_args!(", prompt {whole} tok/s"));
    }
    term::write(format_args!(
        ", context {}/{context}]\r\n",
        chat.positions()
    ));
}

/// Reads lines at the terminal with basic editing.
///
/// Printable text (UTF-8 included) is echoed and kept; Backspace removes
/// the last character, Ctrl-U the whole line, and Ctrl-C drops it; Enter
/// ends it. Escape sequences — the arrow keys and the like — are read and
/// ignored, since there is nowhere to move to.
#[derive(Default)]
struct LineReader {
    /// Whether the last byte was a carriage return, so that a line feed
    /// right after it (a terminal sending `\r\n` for Enter) isn't a
    /// second, empty line.
    after_cr: bool,
}

/// Where an escape sequence being skipped has got to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Escape {
    /// Not in one.
    None,
    /// After `ESC`.
    Started,
    /// After `ESC [`: parameters until a final byte, `0x40..=0x7e`.
    Csi,
    /// After `ESC O`: one more byte.
    Ss3,
}

impl LineReader {
    /// Reads one line into `line`, echoing as it goes.
    fn read_line(&mut self, line: &mut String) {
        line.clear();
        let mut escape = Escape::None;
        // The bytes of a UTF-8 character still arriving.
        let mut partial: Vec<u8> = Vec::new();
        loop {
            let byte = term::read_byte();
            let after_cr = core::mem::replace(&mut self.after_cr, byte == b'\r');
            match escape {
                Escape::Started => {
                    escape = match byte {
                        b'[' => Escape::Csi,
                        b'O' => Escape::Ss3,
                        _ => Escape::None,
                    };
                    continue;
                }
                Escape::Csi => {
                    if (0x40..=0x7e).contains(&byte) {
                        escape = Escape::None;
                    }
                    continue;
                }
                Escape::Ss3 => {
                    escape = Escape::None;
                    continue;
                }
                Escape::None => {}
            }
            match byte {
                b'\n' if after_cr => {}
                b'\r' | b'\n' => {
                    term::write(format_args!("\r\n"));
                    return;
                }
                0x7f | 0x08 => {
                    if line.pop().is_some() {
                        term::write(format_args!("\x08 \x08"));
                    }
                }
                0x15 => {
                    for _ in line.chars() {
                        term::write(format_args!("\x08 \x08"));
                    }
                    line.clear();
                }
                INTERRUPT => {
                    term::write(format_args!("^C\r\n"));
                    line.clear();
                    return;
                }
                0x1b => escape = Escape::Started,
                0x20..=0x7e => {
                    line.push(byte as char);
                    term::write_bytes(&[byte]);
                }
                0x80.. => {
                    partial.push(byte);
                    if let Ok(text) = core::str::from_utf8(&partial) {
                        line.push_str(text);
                        term::write_bytes(&partial);
                        partial.clear();
                    } else if partial.len() >= 4 {
                        // Not UTF-8 after all; drop it.
                        partial.clear();
                    }
                }
                _ => {}
            }
        }
    }
}
