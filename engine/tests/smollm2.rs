//! The engine against llama.cpp on SmolLM2-360M-Instruct (Q8_0 GGUF).
//!
//! The expected outputs in `tests/data/smollm2_*` came from llama.cpp
//! (commit 0ce8cd1) on x86-64:
//!
//! - `smollm2_tokenizer.json`: `llama-tokenize --stdin --ids --no-escape`
//!   on each text. Tokenizing involves no arithmetic, so these must match
//!   exactly.
//! - `smollm2_story.ids`: `llama-simple`, greedy, one thread, f32 key/value
//!   cache, flash attention off, with lines added to print the prompt's
//!   token ids (`P`) and each generated one (`G`). The prompt is the
//!   model's ChatML template around one user message.
//!
//! The model is compared token by token with llama.cpp's choices fed back
//! in (teacher forcing), not by letting both run free. The arithmetic is
//! the same but its summation order is not (llama.cpp's AVX2 kernels add
//! a dot product's blocks in vector lanes), so the logits differ in their
//! last digits — and where two tokens are within that of each other, the
//! two pick differently and then write different texts from there on. On
//! this prompt that happens at the sixth token ("there" against "in", 0.02
//! apart on logits of 18). Fed llama.cpp's own tokens, the model agrees
//! with its choice at 195 of 200 positions, and the other five are ties
//! of that size, which is what a real error in the arithmetic could not
//! produce.
//!
//! Needs `smollm2-360m-instruct-q8_0.gguf` in the repository's `models/`
//! directory (`make models`) and skips, saying so, without it.

mod common;

use common::{Threads, read_model_file};
use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::gguf::Gguf;
use rpi_llm_engine::model::{Transformer, Weights};
use rpi_llm_engine::sample::argmax;

const MODEL: &str = "smollm2-360m-instruct-q8_0.gguf";

/// The prompt `smollm2_story.ids` was generated from. A file of its own
/// because the board program checks itself against the same story.
const STORY_PROMPT: &str = include_str!("data/smollm2_story.prompt");

/// The least share of positions at which the model has to pick what
/// llama.cpp picked.
const MIN_AGREEMENT: f32 = 0.97;

/// How far below the model's own top logit llama.cpp's choice may be where
/// they differ: the size of a near-tie, an order of magnitude under what a
/// real disagreement looks like.
const MAX_GAP: f32 = 0.1;

fn load() -> Option<common::Aligned> {
    let file = read_model_file(MODEL);
    if file.is_none() {
        eprintln!("skipped: no {MODEL} in models/ (`make models`)");
    }
    file
}

#[test]
fn tokenizer_matches_llama_cpp() {
    let Some(file) = load() else { return };
    let gguf = Gguf::parse(file.bytes()).unwrap();
    let bpe = Bpe::from_gguf(&gguf).unwrap();

    let cases: serde_json::Value =
        serde_json::from_str(include_str!("data/smollm2_tokenizer.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(bpe.encode(text), expected, "tokenizing {text:?}");

        // And back: decoding the ids gives the text again.
        let mut decoded = Vec::new();
        for &id in &expected {
            bpe.decode(id, &mut decoded);
        }
        assert_eq!(
            String::from_utf8(decoded).unwrap(),
            text,
            "decoding {text:?}"
        );
    }
}

/// The `P` (prompt) and `G` (generated) ids llama.cpp printed.
fn story_reference() -> (Vec<u32>, Vec<u32>) {
    let (mut prompt, mut generated) = (Vec::new(), Vec::new());
    for line in include_str!("data/smollm2_story.ids").lines() {
        let (kind, id) = line.split_once(' ').unwrap();
        let id: u32 = id.parse().unwrap();
        match kind {
            "P" => prompt.push(id),
            _ => generated.push(id),
        }
    }
    (prompt, generated)
}

#[test]
fn model_agrees_with_llama_cpp_token_by_token() {
    let Some(mut file) = load() else { return };
    let gguf = Gguf::parse_repacked(file.bytes_mut()).unwrap();
    let bpe = Bpe::from_gguf(&gguf).unwrap();
    let (config, weights) = Weights::from_gguf(&gguf).unwrap();
    let (prompt, expected) = story_reference();
    assert_eq!(bpe.encode(STORY_PROMPT), prompt, "the chat prompt's tokens");

    let mut sequence = prompt.clone();
    sequence.extend_from_slice(&expected);
    let mut transformer = Transformer::new(config, weights, sequence.len());

    let mut agree = 0;
    for pos in 0..sequence.len() - 1 {
        let logits = transformer.forward(sequence[pos], pos);
        if pos + 1 < prompt.len() {
            continue;
        }
        let theirs = sequence[pos + 1] as usize;
        let ours = argmax(logits) as usize;
        if ours == theirs {
            agree += 1;
            continue;
        }
        let gap = logits[ours] - logits[theirs];
        let text = |id: usize| {
            let mut bytes = Vec::new();
            bpe.decode(id as u32, &mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        };
        eprintln!(
            "token {}: {:?} where llama.cpp has {:?}, {gap:.4} apart",
            pos + 1 - prompt.len(),
            text(ours),
            text(theirs),
        );
        assert!(gap < MAX_GAP, "a disagreement of {gap}, not a near-tie");
    }

    let share = agree as f32 / expected.len() as f32;
    eprintln!("{agree} of {} tokens agree with llama.cpp", expected.len());
    assert!(
        share >= MIN_AGREEMENT,
        "only {agree} of {} agree",
        expected.len()
    );
}

/// The engine's own greedy generation from the story prompt, as recorded
/// from it: what the board has to reproduce exactly.
///
/// The arithmetic is IEEE `f32` with no fused operations and the same
/// `libm` on both machines, so x86-64 and the Pi 5's A76 compute the same
/// bits, and this — unlike the comparison with llama.cpp — is a test that
/// passes or fails exactly. It also pins the engine's numerics: a change
/// that reorders a sum changes these tokens, and has to say so by updating
/// the file.
#[test]
fn free_running_generation_matches_the_recorded_engine_output() {
    let Some(mut file) = load() else { return };
    let gguf = Gguf::parse_repacked(file.bytes_mut()).unwrap();
    let bpe = Bpe::from_gguf(&gguf).unwrap();
    let (config, weights) = Weights::from_gguf(&gguf).unwrap();

    let prompt = bpe.encode(STORY_PROMPT);
    // Positions, not tokens: the last prompt position already produces the
    // first generated token.
    let steps = prompt.len() + STORY_TOKENS - 1;
    let mut transformer = Transformer::new(config, weights, steps);
    let mut generation = Generation::new(prompt, steps, &[bpe.eos()]);
    let mut ids = Vec::new();
    while let Some(step) = generation.step(&mut transformer) {
        if step.generated {
            ids.push(step.next);
        }
    }

    let recorded: Vec<u32> = include_str!("data/smollm2_story.engine.ids")
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    if ids != recorded {
        let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
        panic!("generated:\n{}", ids.join("\n"));
    }
}

/// How many tokens the recorded engine output runs to.
const STORY_TOKENS: usize = 200;

/// Splitting the matrix multiplies across workers changes nothing: four
/// threads give the same logits, bit for bit, as one, over the whole
/// prompt.
#[test]
fn four_workers_compute_the_same_bits_as_one() {
    let Some(mut file) = load() else { return };
    let gguf = Gguf::parse_repacked(file.bytes_mut()).unwrap();
    let bpe = Bpe::from_gguf(&gguf).unwrap();
    let prompt = bpe.encode(STORY_PROMPT);

    let (config, weights) = Weights::from_gguf(&gguf).unwrap();
    let mut serial = Transformer::new(config, weights, prompt.len());
    let (config, weights) = Weights::from_gguf(&gguf).unwrap();
    let mut parallel = Transformer::new(config, weights, prompt.len());
    let threads = Threads(4);
    parallel.set_workers(&threads);

    for (pos, &token) in prompt.iter().enumerate() {
        let one: Vec<u32> = serial
            .forward(token, pos)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let four: Vec<u32> = parallel
            .forward(token, pos)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert!(one == four, "position {pos}: the logits differ");
    }
}

/// Feeding tokens together through `prefill` leaves the model in exactly
/// the state feeding them one at a time does: the next token's logits are
/// the same bits. Over 150 tokens, so the batches of 64 and their
/// boundaries are covered, on one worker and on four.
#[test]
fn prefill_computes_the_same_bits_as_one_token_at_a_time() {
    let Some(mut file) = load() else { return };
    let gguf = Gguf::parse_repacked(file.bytes_mut()).unwrap();
    let bpe = Bpe::from_gguf(&gguf).unwrap();
    let mut tokens = bpe.encode(STORY_PROMPT);
    tokens.extend(
        include_str!("data/smollm2_story.engine.ids")
            .lines()
            .map(|l| l.parse::<u32>().unwrap()),
    );
    tokens.truncate(150);
    let last = tokens.len() - 1;

    let (config, weights) = Weights::from_gguf(&gguf).unwrap();
    let mut one_by_one = Transformer::new(config, weights, tokens.len());
    for (pos, &token) in tokens[..last].iter().enumerate() {
        one_by_one.forward(token, pos);
    }
    let expected: Vec<u32> = one_by_one
        .forward(tokens[last], last)
        .iter()
        .map(|v| v.to_bits())
        .collect();

    let threads = Threads(4);
    for workers in [None, Some(&threads)] {
        let (config, weights) = Weights::from_gguf(&gguf).unwrap();
        let mut batched = Transformer::new(config, weights, tokens.len());
        if let Some(workers) = workers {
            batched.set_workers(workers);
        }
        batched.prefill(&tokens[..last], 0);
        let got: Vec<u32> = batched
            .forward(tokens[last], last)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert!(
            got == expected,
            "the logits after prefill differ ({} workers)",
            if workers.is_some() { 4 } else { 1 }
        );
    }
}
