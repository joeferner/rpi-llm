//! The engine against llama2.c itself: greedy generation from the same
//! prompt has to produce the same tokens.
//!
//! The expected output in `tests/data/` came from llama2.c's `run.c`
//! (commit 350e04f), built with `gcc -O3` and run as
//!
//! ```text
//! run stories15M.bin -z tokenizer.bin -t 0 -n 256 -i "Once upon a time"
//! ```
//!
//! with one line added to print each token's id to stderr as it is
//! emitted. `-t 0` is greedy decoding, so the output is a function of the
//! weights and the arithmetic alone.
//!
//! Needs the model files in the repository's `models/` directory
//! (`make models` fetches them) and skips, saying so, without them.

use std::path::PathBuf;

use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::llama2::Config;
use rpi_llm_engine::model::{Transformer, Weights};
use rpi_llm_engine::tokenizer::{Tokenizer, printable};

/// The repository's `models/` directory.
fn models() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

/// Reads `name` from `models/` into memory aligned for `f32`s, as the
/// board's loader does; `None` if it is not there.
fn read_model_file(name: &str) -> Option<(Vec<u32>, usize)> {
    let bytes = std::fs::read(models().join(name)).ok()?;
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    // SAFETY: `words` has at least `bytes.len()` bytes, and `u32` has no
    // invalid bit patterns.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast(), bytes.len());
    }
    Some((words, bytes.len()))
}

/// The first `len` bytes of `words`.
fn as_bytes(words: &[u32], len: usize) -> &[u8] {
    // SAFETY: in bounds by construction in `read_model_file`.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), len) }
}

#[test]
fn greedy_generation_matches_llama2_c() {
    let (Some((model, model_len)), Some((tok, tok_len))) = (
        read_model_file("stories15M.bin"),
        read_model_file("tokenizer.bin"),
    ) else {
        eprintln!("skipped: no stories15M.bin and tokenizer.bin in models/ (`make models`)");
        return;
    };
    let checkpoint = as_bytes(&model, model_len);
    let config = Config::parse(checkpoint).unwrap();
    let weights = Weights::new(&config, checkpoint).unwrap();
    let mut transformer = Transformer::new(config, weights);
    let tokenizer = Tokenizer::new(as_bytes(&tok, tok_len), config.vocab_size).unwrap();

    let prompt = tokenizer.encode("Once upon a time", true, false);
    assert_eq!(prompt, [1, 9038, 2501, 263, 931], "the prompt's tokens");

    let mut generation = Generation::new(prompt, 256);
    let mut ids = Vec::new();
    let mut text = Vec::new();
    while let Some(step) = generation.step(&mut transformer) {
        ids.push(step.next);
        let piece = tokenizer.decode(step.prev, step.next);
        if printable(piece) {
            text.extend_from_slice(piece);
        }
    }

    let expected: Vec<u32> = include_str!("data/once_upon_a_time.ids")
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    if let Some(first) = ids.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "token {first} differs: got {}, llama2.c has {}",
            ids[first], expected[first]
        );
    }
    assert_eq!(ids.len(), expected.len(), "how many tokens before <s>");

    // `run.c` ends its output with a newline of its own.
    let expected_text = include_str!("data/once_upon_a_time.txt");
    assert_eq!(
        String::from_utf8(text).unwrap(),
        expected_text.strip_suffix('\n').unwrap()
    );
}
