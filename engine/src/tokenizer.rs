//! The byte-pair-encoding tokenizer, as llama2.c's `run.c` implements
//! SentencePiece's: text to token ids and back, over `tokenizer.bin`.

use alloc::vec::Vec;

use crate::llama2::{FormatError, check_tokenizer};

/// The id of `<s>`, the beginning-of-sequence token. The model also emits
/// it to end one story and start the next.
pub const BOS: u32 = 1;
/// The id of `</s>`, the end-of-sequence token.
pub const EOS: u32 = 2;

/// A vocabulary: each token's text and merge score, plus the order that
/// finds a text's id by binary search.
pub struct Tokenizer<'a> {
    /// Each token's bytes, borrowed from the tokenizer file.
    pieces: Vec<&'a [u8]>,
    /// Each token's merge score: higher merges first.
    scores: Vec<f32>,
    /// Token ids, sorted by their bytes.
    sorted: Vec<u32>,
    /// The longest piece, in bytes.
    max_token_length: usize,
}

impl<'a> Tokenizer<'a> {
    /// Reads `vocab_size` tokens from a `tokenizer.bin`.
    pub fn new(file: &'a [u8], vocab_size: usize) -> Result<Self, FormatError> {
        let summary = check_tokenizer(file, vocab_size)?;

        // `check_tokenizer` has walked these exact offsets already, so the
        // reads below are in bounds.
        let read_i32 = |at: usize| i32::from_le_bytes(file[at..at + 4].try_into().unwrap());
        let mut pieces = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut at = 4;
        for _ in 0..vocab_size {
            scores.push(f32::from_bits(read_i32(at) as u32));
            let len = read_i32(at + 4) as usize;
            at += 8;
            pieces.push(&file[at..at + len]);
            at += len;
        }

        let mut sorted: Vec<u32> = (0..vocab_size as u32).collect();
        // `run.c` sorts with `strcmp`, which on these NUL-free pieces is
        // the same byte order a slice comparison gives.
        sorted.sort_by(|&a, &b| pieces[a as usize].cmp(pieces[b as usize]));

        Ok(Self {
            pieces,
            scores,
            sorted,
            max_token_length: summary.max_token_length,
        })
    }

    /// The id of the token whose text is exactly `text`.
    fn lookup(&self, text: &[u8]) -> Option<u32> {
        self.sorted
            .binary_search_by(|&id| self.pieces[id as usize].cmp(text))
            .ok()
            .map(|i| self.sorted[i])
    }

    /// Encodes `text` as tokens, with `<s>` first if `bos`, and `</s>` last
    /// if `eos`.
    ///
    /// `run.c`'s algorithm step for step: prefix a space (SentencePiece's
    /// `add_dummy_prefix`), map each UTF-8 code point to its token or, for
    /// one the vocabulary lacks, each of its bytes to a byte token, then
    /// repeatedly merge the adjacent pair whose merged text has the highest
    /// score until no pair merges.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty()
            && let Some(space) = self.lookup(b" ")
        {
            tokens.push(space);
        }

        let mut buffer = [0u8; 4];
        for c in text.chars() {
            let bytes = c.encode_utf8(&mut buffer).as_bytes();
            match self.lookup(bytes) {
                Some(id) => tokens.push(id),
                // Byte fallback: ids 3-258 are the bytes 0-255, after
                // `<unk>`, `<s>` and `</s>`.
                None => tokens.extend(bytes.iter().map(|&b| u32::from(b) + 3)),
            }
        }

        let mut merged = Vec::with_capacity(2 * self.max_token_length);
        loop {
            let mut best: Option<(f32, u32, usize)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                merged.clear();
                merged.extend_from_slice(self.pieces[tokens[i] as usize]);
                merged.extend_from_slice(self.pieces[tokens[i + 1] as usize]);
                if let Some(id) = self.lookup(&merged) {
                    let score = self.scores[id as usize];
                    // Strictly greater, so the leftmost of equal scores
                    // wins, as in `run.c`.
                    if best.is_none_or(|(best, _, _)| score > best) {
                        best = Some((score, id, i));
                    }
                }
            }
            let Some((_, id, i)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }

        if eos {
            tokens.push(EOS);
        }
        tokens
    }

    /// The bytes `token` stands for, following `prev`.
    ///
    /// Two adjustments, both `run.c`'s: after `<s>` the leading space a
    /// piece carries is dropped, and a byte token (`<0x0A>`) is the one
    /// byte it names rather than its six characters of text.
    pub fn decode(&self, prev: u32, token: u32) -> &[u8] {
        let mut piece = self.pieces[token as usize];
        if prev == BOS && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        if let Some(byte) = byte_token(piece) {
            return &BYTES[byte as usize..byte as usize + 1];
        }
        piece
    }
}

/// The byte a piece like `<0x0A>` names, if it is one.
fn byte_token(piece: &[u8]) -> Option<u8> {
    let hex = piece.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    if hex.len() != 2 {
        return None;
    }
    let digit = |c: u8| (c as char).to_digit(16);
    Some((digit(hex[0])? * 16 + digit(hex[1])?) as u8)
}

/// Every byte value, so [`Tokenizer::decode`] can return a one-byte slice
/// for a byte token.
static BYTES: [u8; 256] = {
    let mut bytes = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        bytes[i] = i as u8;
        i += 1;
    }
    bytes
};

/// Whether `run.c`'s `safe_printf` would print `piece`: anything but a
/// single byte that is neither printable ASCII nor whitespace. For the
/// terminal, which a stray control byte would otherwise disturb.
pub fn printable(piece: &[u8]) -> bool {
    match piece {
        [] => false,
        // C's `isprint` is 0x20-0x7e, and its `isspace` includes the
        // vertical tab (0x0b), which Rust's ASCII whitespace does not.
        [byte] => (0x20..=0x7e).contains(byte) || byte.is_ascii_whitespace() || *byte == 0x0b,
        _ => true,
    }
}
