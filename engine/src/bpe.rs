//! GPT-2-style byte-level BPE, the tokenizer GGUF calls `gpt2`, with the
//! `smollm` pre-tokenizer, as llama.cpp implements them.
//!
//! Encoding a text runs in four stages, each llama.cpp's:
//!
//! 1. Special tokens (`<|im_start|>` and the other control tokens) are cut
//!    out of the text first and become their own ids; only the text
//!    between them is tokenized further.
//! 2. Pre-tokenization splits that text into words: first every number
//!    character on its own (`\p{N}`), then the GPT-2 pattern
//!    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!    within each piece — written out by hand, as llama.cpp's
//!    `unicode_regex_split_custom_gpt2` is.
//! 3. Each word's UTF-8 bytes are mapped to printable characters (GPT-2's
//!    `bytes_to_unicode`: a space becomes `Ġ`, a newline `Ċ`), which is the
//!    alphabet the vocabulary and merges are written in.
//! 4. The word is merged pair by pair, always the adjacent pair whose
//!    merge ranks earliest in `tokenizer.ggml.merges`, until none merges.
//!
//! Unicode classes come from `core`'s `char` methods: `is_alphabetic` for
//! `\p{L}`, `is_numeric` for `\p{N}`, `is_whitespace` for `\s`. Those match
//! llama.cpp's tables everywhere in ASCII and in ordinary text, and differ
//! only at the edges of the categories (a combining mark `core` calls
//! alphabetic, a letter-number like `Ⅻ`).

use alloc::string::String;
use alloc::vec::Vec;

use crate::gguf::Gguf;

/// GGUF's token type for a control token, which is matched whole in text
/// and never printed.
const CONTROL: i32 = 3;

/// Why a tokenizer could not be read out of a file.
#[derive(Debug, PartialEq, Eq)]
pub enum BpeError {
    /// A metadata key the tokenizer needs is missing or the wrong type.
    Metadata(&'static str),
    /// `tokenizer.ggml.model` is not `gpt2`.
    Model,
    /// A merge is not two tokens separated by a space.
    Merge(usize),
}

/// A byte-level BPE vocabulary.
pub struct Bpe<'a> {
    /// Each token's text, in the byte-to-character alphabet.
    tokens: Vec<&'a str>,
    /// Token ids, sorted by text, for lookup.
    sorted: Vec<u32>,
    /// Each merge's two halves, in rank order.
    merges: Vec<(&'a str, &'a str)>,
    /// Merge ranks, sorted by the pair, for lookup.
    merges_sorted: Vec<u32>,
    /// The control tokens, longest text first: the order to match them in.
    specials: Vec<u32>,
    /// Whether each token is a control token.
    control: Vec<bool>,
    /// Beginning of sequence.
    bos: u32,
    /// End of sequence: for a chat model, the end of a turn.
    eos: u32,
}

impl<'a> Bpe<'a> {
    /// Reads the tokenizer out of a GGUF file's metadata.
    pub fn from_gguf(gguf: &Gguf<'a>) -> Result<Self, BpeError> {
        if gguf.string("tokenizer.ggml.model") != Some("gpt2") {
            return Err(BpeError::Model);
        }
        let tokens: Vec<&str> = gguf
            .array("tokenizer.ggml.tokens")
            .and_then(|a| a.strings().map(|s| s.collect()))
            .ok_or(BpeError::Metadata("tokenizer.ggml.tokens"))?;
        let types: Vec<i32> = gguf
            .array("tokenizer.ggml.token_type")
            .and_then(|a| a.i32s().map(|s| s.collect()))
            .ok_or(BpeError::Metadata("tokenizer.ggml.token_type"))?;
        let merges = gguf
            .array("tokenizer.ggml.merges")
            .and_then(|a| a.strings())
            .ok_or(BpeError::Metadata("tokenizer.ggml.merges"))?
            .enumerate()
            .map(|(i, m)| m.split_once(' ').ok_or(BpeError::Merge(i)))
            .collect::<Result<Vec<_>, _>>()?;
        let id = |key: &'static str| {
            gguf.uint(key)
                .map(|v| v as u32)
                .ok_or(BpeError::Metadata(key))
        };

        let mut sorted: Vec<u32> = (0..tokens.len() as u32).collect();
        sorted.sort_by_key(|&i| tokens[i as usize]);
        let mut merges_sorted: Vec<u32> = (0..merges.len() as u32).collect();
        merges_sorted.sort_by_key(|&i| merges[i as usize]);
        let control: Vec<bool> = types.iter().map(|&t| t == CONTROL).collect();
        // An empty text would match everywhere; no real control token has
        // one, but a file that did would otherwise never finish encoding.
        let mut specials: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| control[i as usize] && !tokens[i as usize].is_empty())
            .collect();
        specials.sort_by_key(|&i| core::cmp::Reverse(tokens[i as usize].len()));

        Ok(Self {
            bos: id("tokenizer.ggml.bos_token_id")?,
            eos: id("tokenizer.ggml.eos_token_id")?,
            tokens,
            sorted,
            merges,
            merges_sorted,
            specials,
            control,
        })
    }

    /// The beginning-of-sequence token.
    pub fn bos(&self) -> u32 {
        self.bos
    }

    /// The end-of-sequence token; for a chat model, the end of a turn.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Whether `token` is a control token, which [`Self::decode`] renders
    /// as nothing.
    pub fn is_control(&self, token: u32) -> bool {
        self.control[token as usize]
    }

    /// The id of the token whose text is `text`, in the byte alphabet.
    pub fn id(&self, text: &str) -> Option<u32> {
        self.sorted
            .binary_search_by(|&i| self.tokens[i as usize].cmp(text))
            .ok()
            .map(|at| self.sorted[at])
    }

    /// The rank of the merge `left right`, if there is one.
    fn rank(&self, left: &str, right: &str) -> Option<u32> {
        self.merges_sorted
            .binary_search_by(|&i| self.merges[i as usize].cmp(&(left, right)))
            .ok()
            .map(|at| self.merges_sorted[at])
    }

    /// Encodes `text`, with any control token's text in it becoming that
    /// token. No `<s>` is added: this tokenizer's models do not want one.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in what is left, longest first
            // where two start at the same place.
            let next = self
                .specials
                .iter()
                .filter_map(|&id| rest.find(self.tokens[id as usize]).map(|at| (at, id)))
                .min_by_key(|&(at, _)| at);
            match next {
                Some((at, id)) => {
                    self.encode_plain(&rest[..at], &mut ids);
                    ids.push(id);
                    rest = &rest[at + self.tokens[id as usize].len()..];
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    rest = "";
                }
            }
        }
        ids
    }

    /// Encodes text with no special tokens in it, appending to `ids`.
    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        let chars: Vec<char> = text.chars().collect();
        let mut encoded = String::new();
        let mut symbols: Vec<(usize, usize)> = Vec::new();
        for (start, end) in pre_tokenize(&chars) {
            // The word's bytes, in the byte alphabet, one symbol per byte.
            encoded.clear();
            symbols.clear();
            let mut utf8 = [0u8; 4];
            for c in &chars[start..end] {
                for &byte in c.encode_utf8(&mut utf8).as_bytes() {
                    let at = encoded.len();
                    encoded.push(byte_to_char(byte));
                    symbols.push((at, encoded.len()));
                }
            }

            // Merge the earliest-ranked adjacent pair until none merges;
            // of equal ranks, the leftmost.
            loop {
                let best = symbols
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, pair)| {
                        let (l, r) = (pair[0], pair[1]);
                        self.rank(&encoded[l.0..l.1], &encoded[r.0..r.1])
                            .map(|rank| (rank, i))
                    })
                    .min();
                let Some((_, i)) = best else { break };
                symbols[i].1 = symbols[i + 1].1;
                symbols.remove(i + 1);
            }

            for &(s, e) in &symbols {
                match self.id(&encoded[s..e]) {
                    Some(id) => ids.push(id),
                    // A merged symbol missing from the vocabulary: llama.cpp
                    // falls back to its single characters.
                    None => ids.extend(
                        encoded[s..e]
                            .char_indices()
                            .filter_map(|(i, c)| self.id(&encoded[s + i..s + i + c.len_utf8()])),
                    ),
                }
            }
        }
    }

    /// Appends the bytes `token` stands for to `out`; nothing for a control
    /// token.
    pub fn decode(&self, token: u32, out: &mut Vec<u8>) {
        if self.is_control(token) {
            return;
        }
        let mut utf8 = [0u8; 4];
        for c in self.tokens[token as usize].chars() {
            match char_to_byte(c) {
                Some(byte) => out.push(byte),
                // Not in the byte alphabet, which a byte-level vocabulary's
                // ordinary tokens never are: pass it through as text.
                None => out.extend_from_slice(c.encode_utf8(&mut utf8).as_bytes()),
            }
        }
    }
}

/// A character's classes, as the GPT-2 pattern sees them.
#[derive(Clone, Copy, Default)]
struct Class {
    letter: bool,
    number: bool,
    whitespace: bool,
    /// Whether there is a character at all: false past the end of the
    /// piece being split.
    present: bool,
}

impl Class {
    fn of(c: char) -> Self {
        Self {
            letter: c.is_alphabetic(),
            number: c.is_numeric(),
            whitespace: c.is_whitespace(),
            present: true,
        }
    }

    /// `[^\s\p{L}\p{N}]`.
    fn other(self) -> bool {
        self.present && !(self.whitespace || self.letter || self.number)
    }
}

/// Splits `chars` into words, as `(start, end)` indices: the `smollm`
/// pre-tokenizer.
fn pre_tokenize(chars: &[char]) -> Vec<(usize, usize)> {
    // Stage one, `\p{N}`: every number character is a piece of its own, and
    // the text between them is kept as pieces too.
    let mut pieces = Vec::new();
    let mut start = 0;
    for (i, c) in chars.iter().enumerate() {
        if c.is_numeric() {
            if start < i {
                pieces.push((start, i));
            }
            pieces.push((i, i + 1));
            start = i + 1;
        }
    }
    if start < chars.len() {
        pieces.push((start, chars.len()));
    }

    // Stage two, the GPT-2 pattern within each piece.
    let mut words = Vec::new();
    for (begin, end) in pieces {
        let class = |pos: usize| {
            if pos < end {
                Class::of(chars[pos])
            } else {
                Class::default()
            }
        };
        let mut pos = begin;
        let mut word = |to: usize, pos: &mut usize| {
            words.push((*pos, to));
            *pos = to;
        };
        while pos < end {
            let c = chars[pos];

            // 's|'t|'re|'ve|'m|'ll|'d
            if c == '\'' && pos + 1 < end {
                let n = chars[pos + 1];
                if matches!(n, 's' | 't' | 'm' | 'd') {
                    word(pos + 2, &mut pos);
                    continue;
                }
                if pos + 2 < end {
                    let nn = chars[pos + 2];
                    if matches!((n, nn), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        word(pos + 3, &mut pos);
                        continue;
                    }
                }
            }

            // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: an optional
            // space, then a run of one class.
            let lead = c == ' ';
            let first = class(pos + usize::from(lead));
            let same: Option<fn(Class) -> bool> = if first.letter {
                Some(|k| k.letter)
            } else if first.number {
                Some(|k| k.number)
            } else if first.other() {
                Some(Class::other)
            } else {
                None
            };
            if let Some(same) = same {
                let mut to = pos + usize::from(lead);
                while same(class(to)) {
                    to += 1;
                }
                word(to, &mut pos);
                continue;
            }

            let mut spaces = 0;
            while class(pos + spaces).whitespace {
                spaces += 1;
            }
            // `\s+(?!\S)`: a run of whitespace before more text leaves its
            // last character to start the next word (` word`).
            if spaces > 1 && pos + spaces < end {
                word(pos + spaces - 1, &mut pos);
                continue;
            }
            // `\s+`
            if spaces > 0 {
                word(pos + spaces, &mut pos);
                continue;
            }
            word(pos + 1, &mut pos);
        }
    }
    words
}

/// GPT-2's `bytes_to_unicode`: the printable bytes stand for themselves,
/// and the other 68 are given the code points from 256 up, in order.
fn byte_to_char(byte: u8) -> char {
    if printable(byte) {
        return char::from(byte);
    }
    let rank = (0..byte).filter(|&b| !printable(b)).count() as u32;
    char::from_u32(256 + rank).unwrap()
}

/// The inverse of [`byte_to_char`].
fn char_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    if code < 256 {
        let byte = code as u8;
        return printable(byte).then_some(byte);
    }
    let rank = (code - 256) as usize;
    (0..=255u8).filter(|&b| !printable(b)).nth(rank)
}

/// Whether GPT-2's byte alphabet keeps `byte` as its own character: `!`
/// to `~`, `¡` to `¬`, and `®` to `ÿ`.
fn printable(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        pre_tokenize(&chars)
            .into_iter()
            .map(|(s, e)| chars[s..e].iter().collect())
            .collect()
    }

    #[test]
    fn byte_alphabet_round_trips() {
        for b in 0..=255u8 {
            assert_eq!(char_to_byte(byte_to_char(b)), Some(b));
        }
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
    }

    #[test]
    fn pre_tokenizer_splits_like_llama_cpp() {
        assert_eq!(words("Hello world"), ["Hello", " world"]);
        assert_eq!(
            words("I have 25 apples"),
            ["I", " have", " ", "2", "5", " apples"]
        );
        assert_eq!(words("don't they're"), ["don", "'t", " they", "'re"]);
        assert_eq!(words("a   b"), ["a", "  ", " b"]);
        assert_eq!(words("end  "), ["end", "  "]);
        assert_eq!(words("x\n\ny"), ["x", "\n", "\n", "y"]);
        assert_eq!(words("hi!!! ok"), ["hi", "!!!", " ok"]);
    }
}
