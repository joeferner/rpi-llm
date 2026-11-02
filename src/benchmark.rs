//! The benchmark, with the `benchmark` feature: one fixed chat request,
//! answered greedily, timed phase by phase, and compared token for token
//! with what the same engine produced on the host. Runs before the chat.
//!
//! These are the numbers the README's performance table comes from.

use alloc::vec::Vec;

use rpi_hal::mailbox::{ClockId, Mailbox};
use rpi_hal::timer::Timer;
use rpi_kickstart::logln;
use rpi_llm_engine::bpe::Bpe;
use rpi_llm_engine::generate::Generation;
use rpi_llm_engine::model::{Clock, Phases, Transformer};

use crate::{cores, term};

/// The chat turn: the model's ChatML template around one request, the same
/// prompt the engine's host tests use.
const PROMPT: &str = include_str!("../engine/tests/data/smollm2_story.prompt");

/// How many tokens the answer may run to.
const ANSWER_TOKENS: usize = 200;

/// What the engine generated from [`PROMPT`] on the host. The arithmetic is
/// IEEE `f32` with nothing fused and the same `libm` on both machines, so
/// the board has to produce exactly this.
const REFERENCE: &str = include_str!("../engine/tests/data/smollm2_story.engine.ids");

/// Runs the benchmark and logs its results. Leaves the model's profile
/// and the cores' statistics zeroed for whatever runs after.
pub fn run(transformer: &mut Transformer, bpe: &Bpe, timer: &Timer, mailbox: &mut Mailbox) {
    cores::reset_stats();
    transformer.reset_profile();
    let started = timer.now_micros();
    let ids = answer(transformer, bpe, timer, mailbox);
    let profile = transformer.profile();
    let rate = cores::Counter.ticks_per_second();
    report_phases("prompt", &profile.prefill, rate);
    report_phases("answer", &profile.forward, rate);
    let total_ms = (timer.now_micros() - started) / 1000;
    let (per_core, jobs) = cores::stats();
    logln!(
        "cores: {jobs} jobs in {total_ms} ms; core 0 worked {} ms, waited {} ms for the others",
        per_core[0].0 / 1000,
        per_core[0].1 / 1000
    );
    for (id, (work, wake)) in per_core.iter().enumerate().skip(1) {
        logln!(
            "cores: core {id} worked {} ms, spent {} ms waking up",
            work / 1000,
            wake / 1000
        );
    }
    check_against_reference(&ids);
    cores::reset_stats();
    transformer.reset_profile();
}

/// Logs what the ARM is running at `when`: the measured clock, the SoC's
/// temperature, and the firmware's throttling word (bits 0-3 now, 16-19
/// since boot: under-voltage, frequency capped, throttled, soft
/// temperature limit).
///
/// The prompt is the run's hardest burst, all four cores in `sdot` at
/// once, and one run's prompt phases all slowed down together by about
/// 40% while the answer's ran at full speed: what a clock cut for the
/// burst would look like.
fn arm_state(mailbox: &mut Mailbox, when: &str) {
    let mhz = mailbox
        .measured_clock_rate_hz(ClockId::Arm)
        .map_or(0, |hz| hz / 1_000_000);
    let temp = mailbox.temperature_millicelsius().unwrap_or(0);
    let throttled = mailbox.throttled().unwrap_or(u32::MAX);
    logln!(
        "arm: {when}: {mhz} MHz measured, {}.{} C, throttled {throttled:#x}",
        temp / 1000,
        temp % 1000 / 100
    );
}

/// Answers [`PROMPT`] greedily, streaming the answer to the terminal as
/// each token arrives, and returns the answer's tokens.
fn answer(
    transformer: &mut Transformer,
    bpe: &Bpe,
    timer: &Timer,
    mailbox: &mut Mailbox,
) -> Vec<u32> {
    let prompt = bpe.encode(PROMPT);
    let prompt_len = prompt.len();
    // Positions, not tokens: the last prompt position already produces the
    // first answer token.
    let steps = prompt_len + ANSWER_TOKENS - 1;
    let mut generation = Generation::new(prompt, steps, &[bpe.eos()]);
    logln!("benchmark: {prompt_len} prompt tokens, greedy, up to {ANSWER_TOKENS} answer tokens");
    term::write_bytes(PROMPT.as_bytes());

    let mut ids = Vec::new();
    let mut piece = Vec::new();
    arm_state(mailbox, "before the prompt");
    let mut started = timer.now_micros();
    // The whole prompt through the model together; the first step then
    // returns the answer's first token.
    generation.prefill(transformer);
    // Read straight after, while the burst's state lasts; its own time is
    // taken out of the prompt's.
    let probe = timer.now_micros();
    arm_state(mailbox, "after the prompt");
    started += timer.now_micros() - probe;
    // When the first answer token arrived: the prompt's positions before it
    // and the answer's after it are timed separately, since a chat's
    // latency is the one and its speed the other.
    let mut first: Option<u64> = None;
    while let Some(step) = generation.step(transformer) {
        if !step.generated {
            continue;
        }
        first.get_or_insert_with(|| timer.now_micros());
        ids.push(step.next);
        piece.clear();
        bpe.decode(step.next, &mut piece);
        term::write_bytes(&piece);
    }
    let ended = timer.now_micros();
    term::write_bytes(b"\n");

    if let Some(first) = first {
        let prompt_s = (first - started) as f32 / 1e6;
        logln!(
            "benchmark: prompt {prompt_len} tokens in {prompt_s:.2} s, {:.1} tok/s",
            prompt_len as f32 / prompt_s
        );
        if ids.len() > 1 {
            let answer_s = (ended - first) as f32 / 1e6;
            logln!(
                "benchmark: answer {} tokens in {answer_s:.2} s, {:.1} tok/s",
                ids.len(),
                (ids.len() - 1) as f32 / answer_s
            );
        }
    }
    ids
}

/// Logs where the engine's calls spent their time: `prompt` for the
/// prefill batches, `answer` for the single-token forwards.
fn report_phases(name: &str, p: &Phases, rate: u64) {
    let ms = |ticks: u64| ticks as f32 * 1000.0 / rate as f32;
    logln!(
        "phases: {name}: {} calls in {:.1} ms: quantize {:.1}, matmul {:.1}, attention {:.1}, other {:.1}",
        p.calls,
        ms(p.total),
        ms(p.quantize),
        ms(p.matmul),
        ms(p.attention),
        ms(p.other()),
    );
}

/// Logs whether `ids` is the answer the engine produced on the host.
fn check_against_reference(ids: &[u32]) {
    let expected: Vec<u32> = REFERENCE
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    match ids.iter().zip(&expected).position(|(a, b)| a != b) {
        Some(i) => logln!(
            "reference: token {i} differs: {}, the host engine has {}",
            ids[i],
            expected[i]
        ),
        None if ids.len() == expected.len() => {
            logln!("reference: all {} tokens match the host engine", ids.len());
        }
        None => logln!(
            "reference: {} tokens where the host engine has {}",
            ids.len(),
            expected.len()
        ),
    }
}
