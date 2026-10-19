//! Running a prompt forward and sampling what follows: `run.c`'s
//! `generate` loop, one token per call so the caller can print and time
//! each one as it arrives.

use alloc::vec::Vec;

use crate::model::Transformer;
use crate::sample::argmax;
use crate::tokenizer::BOS;

/// One step of a generation: the token that was fed in and the one that
/// came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// The token the model was given, which [`Tokenizer::decode`] needs to
    /// render `next` correctly.
    ///
    /// [`Tokenizer::decode`]: crate::tokenizer::Tokenizer::decode
    pub prev: u32,
    /// The token that follows it.
    pub next: u32,
    /// Whether `next` was chosen by the model, or was the next prompt
    /// token being fed through. Only chosen tokens count toward a
    /// generation rate.
    pub generated: bool,
}

/// A generation in progress: a prompt being fed through, then the model's
/// own tokens, greedily chosen.
pub struct Generation {
    prompt: Vec<u32>,
    token: u32,
    pos: usize,
    steps: usize,
}

impl Generation {
    /// Starts a generation of at most `steps` positions (prompt included,
    /// as in `run.c`) from `prompt`, which must start with `<s>`.
    ///
    /// # Panics
    ///
    /// If `prompt` is empty.
    pub fn new(prompt: Vec<u32>, steps: usize) -> Self {
        let token = *prompt.first().expect("a prompt has at least <s>");
        Self {
            prompt,
            token,
            pos: 0,
            steps,
        }
    }

    /// The prompt's length, in tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    /// Runs one position: `None` once `steps` positions have run or the
    /// model has emitted `<s>`, which ends the story it was telling.
    pub fn step(&mut self, transformer: &mut Transformer) -> Option<Step> {
        if self.pos >= self.steps {
            return None;
        }
        let logits = transformer.forward(self.token, self.pos);
        let (next, generated) = match self.prompt.get(self.pos + 1) {
            Some(&forced) => (forced, false),
            None => (argmax(logits), true),
        };
        self.pos += 1;
        if next == BOS {
            self.steps = self.pos;
            return None;
        }
        let step = Step {
            prev: self.token,
            next,
            generated,
        };
        self.token = next;
        Some(step)
    }
}
