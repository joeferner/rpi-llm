//! Running a prompt forward and sampling what follows: `run.c`'s
//! `generate` loop, one token per call so the caller can print and time
//! each one as it arrives.

use alloc::vec::Vec;

use crate::model::Transformer;
use crate::sample::argmax;

/// One step of a generation: the token that was fed in and the one that
/// came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// The token the model was given, which llama2.c's
    /// [`Tokenizer::decode`] needs to render `next` correctly.
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
    stop: Vec<u32>,
}

impl Generation {
    /// Starts a generation of at most `steps` positions (prompt included,
    /// as in `run.c`) from `prompt`, ending early when the model chooses
    /// one of the `stop` tokens — `<s>` for llama2.c's story models, which
    /// start the next story with it, or the end-of-turn token for a chat
    /// model. A stop token ends the generation without being emitted.
    ///
    /// # Panics
    ///
    /// If `prompt` is empty.
    pub fn new(prompt: Vec<u32>, steps: usize, stop: &[u32]) -> Self {
        let token = *prompt.first().expect("a prompt has at least one token");
        Self {
            prompt,
            token,
            pos: 0,
            steps,
            stop: stop.to_vec(),
        }
    }

    /// The prompt's length, in tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    /// Runs every prompt token but the last through the model together
    /// ([`Transformer::prefill`]), instead of one [`Self::step`] each.
    ///
    /// The cache ends up holding the same bits either way, so the answer is
    /// the same; it just arrives sooner. The prompt tokens fed this way are
    /// not returned as steps: the next `step` feeds the last prompt token
    /// and returns the first answer token.
    ///
    /// # Panics
    ///
    /// If called after the first `step`.
    pub fn prefill(&mut self, transformer: &mut Transformer) {
        assert_eq!(self.pos, 0, "prefill comes before the first step");
        let last = self.prompt.len() - 1;
        if last > 0 {
            transformer.prefill(&self.prompt[..last], 0);
        }
        self.pos = last;
        self.token = self.prompt[last];
    }

    /// Runs one position: `None` once `steps` positions have run or the
    /// model has chosen a stop token.
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
        if generated && self.stop.contains(&next) {
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
