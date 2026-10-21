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

mod common;

use common::read_model_file;
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::llama2::Config;
use rpi_llm_engine::model::{Transformer, Weights};
use rpi_llm_engine::tokenizer::{BOS, Tokenizer, printable};

#[test]
fn greedy_generation_matches_llama2_c() {
    let (Some(model), Some(tok)) = (
        read_model_file("stories15M.bin"),
        read_model_file("tokenizer.bin"),
    ) else {
        eprintln!("skipped: no stories15M.bin and tokenizer.bin in models/ (`make models`)");
        return;
    };
    let checkpoint = model.bytes();
    let header = Config::parse(checkpoint).unwrap();
    let (config, weights) = Weights::from_llama2(&header, checkpoint).unwrap();
    let mut transformer = Transformer::new(config, weights, 256);
    let tokenizer = Tokenizer::new(tok.bytes(), config.vocab_size).unwrap();

    let prompt = tokenizer.encode("Once upon a time", true, false);
    assert_eq!(prompt, [1, 9038, 2501, 263, 931], "the prompt's tokens");

    let mut generation = Generation::new(prompt, 256, &[BOS]);
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

/// `prefill` on the f32 path leaves the same bits as feeding one token at
/// a time, and `prefill_logits` computes the same logits: llama2.c's own
/// story, 150 tokens in, so batches of 64 and their boundaries are
/// covered.
#[test]
fn prefill_computes_the_same_bits_as_one_token_at_a_time() {
    let Some(model) = read_model_file("stories15M.bin") else {
        eprintln!("skipped: no stories15M.bin in models/ (`make models`)");
        return;
    };
    let checkpoint = model.bytes();
    let header = Config::parse(checkpoint).unwrap();
    let mut tokens = vec![BOS];
    tokens.extend(
        include_str!("data/once_upon_a_time.ids")
            .lines()
            .map(|l| l.parse::<u32>().unwrap())
            .take(149),
    );
    let last = tokens.len() - 1;

    let (config, weights) = Weights::from_llama2(&header, checkpoint).unwrap();
    let mut one_by_one = Transformer::new(config, weights, 256);
    for (pos, &token) in tokens[..last].iter().enumerate() {
        one_by_one.forward(token, pos);
    }
    let expected: Vec<u32> = one_by_one
        .forward(tokens[last], last)
        .iter()
        .map(|v| v.to_bits())
        .collect();

    let (config, weights) = Weights::from_llama2(&header, checkpoint).unwrap();
    let mut batched = Transformer::new(config, weights, 256);
    batched.prefill(&tokens[..last], 0);
    let got: Vec<u32> = batched
        .forward(tokens[last], last)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert!(got == expected, "the logits after prefill differ");

    // The last token in the batch too.
    let (config, weights) = Weights::from_llama2(&header, checkpoint).unwrap();
    let mut batched = Transformer::new(config, weights, 256);
    let got: Vec<u32> = batched
        .prefill_logits(&tokens, 0)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert!(got == expected, "the logits from prefill_logits differ");
}
