//! Choosing the next token from the logits.

/// The index of the largest logit, the first of any that tie: greedy
/// decoding, `run.c`'s `sample_argmax`, and what its `-t 0` runs.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &logit) in logits.iter().enumerate().skip(1) {
        if logit > logits[best] {
            best = i;
        }
    }
    best as u32
}
