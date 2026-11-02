//! A conversation with a ChatML model, kept in the key/value cache from
//! one turn to the next.
//!
//! Each turn only runs its own new tokens through the model: the earlier
//! turns are already in the cache at the positions before it, so a reply
//! costs its message, not the whole conversation so far. The template is
//! SmolLM2's:
//!
//! ```text
//! <|im_start|>system
//! You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
//! <|im_start|>user
//! {message}<|im_end|>
//! <|im_start|>assistant
//! {reply}<|im_end|>
//! <|im_start|>user
//! ...
//! ```
//!
//! A reply ends with a token chosen but never fed back in: `<|im_end|>`,
//! where the model ended its turn, or the one a reply was cut off at. So
//! every turn after the first starts with that token — then `<|im_end|>`
//! too, for a reply that was cut off — closing the previous reply in the
//! same batch as the new message rather than costing a forward pass of
//! its own.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::bpe::Bpe;
use crate::model::Transformer;
use crate::sample::argmax;

/// The system message SmolLM2-Instruct was trained with.
pub const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// A conversation in progress: where in the cache it has reached.
pub struct Chat {
    /// The system message the conversation opens with.
    system: String,
    /// Positions of the cache the conversation fills so far.
    pos: usize,
    /// The token the last reply ended with, chosen but not yet in the
    /// cache; the next turn feeds it first. `None` in a new conversation.
    pending: Option<u32>,
}

/// Why a reply ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The model ended its turn.
    End,
    /// The reply reached the caller's token limit.
    Limit,
    /// The caller asked it to stop (its callback returned `false`).
    Interrupted,
    /// The cache is full. The next turn starts a new conversation.
    ContextFull,
}

/// What a reply was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reply {
    /// Tokens the turn's prompt ran through the model.
    pub prompt_tokens: usize,
    /// Tokens of reply generated.
    pub reply_tokens: usize,
    /// Why it ended.
    pub stop: Stop,
    /// Whether the conversation started over for this turn: the cache had
    /// no room left for the message and a reply.
    pub restarted: bool,
}

/// Why a turn couldn't be taken at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatError {
    /// The message, with the system message and the template around it,
    /// doesn't fit in the cache even in a new conversation.
    TooLong {
        /// Tokens it would take.
        tokens: usize,
        /// Positions the cache has.
        context: usize,
    },
}

/// Positions a turn keeps free for its reply before it will start: with
/// fewer, the conversation starts over rather than cut the reply short.
const MIN_REPLY_ROOM: usize = 64;

impl Chat {
    /// A new conversation that opens with `system`.
    pub fn new(system: &str) -> Self {
        Self {
            system: system.into(),
            pos: 0,
            pending: None,
        }
    }

    /// Positions of the cache the conversation fills so far.
    pub fn positions(&self) -> usize {
        self.pos
    }

    /// Forgets the conversation: the next turn opens a new one.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.pending = None;
    }

    /// The tokens of the next turn for `message`, in the template: the
    /// system message first in a new conversation, and otherwise the
    /// previous reply closed first.
    pub fn turn_tokens(&self, bpe: &Bpe, message: &str) -> Vec<u32> {
        let ask = format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n");
        match self.pending {
            None => bpe.encode(&format!(
                "<|im_start|>system\n{}<|im_end|>\n{ask}",
                self.system
            )),
            Some(last) => {
                let mut tokens = alloc::vec![last];
                let close = if last == bpe.eos() { "" } else { "<|im_end|>" };
                tokens.extend(bpe.encode(&format!("{close}\n{ask}")));
                tokens
            }
        }
    }

    /// Answers `message` greedily, handing each reply token to `on_token`
    /// as it is chosen, and stopping after `max_tokens` of them, when the
    /// model ends its turn, or when `on_token` returns `false`.
    ///
    /// The first call to `on_token` comes as soon as the prompt has run,
    /// which is what a chat's latency is.
    ///
    /// # Errors
    ///
    /// [`ChatError::TooLong`] if the message can't fit in the cache.
    pub fn reply(
        &mut self,
        transformer: &mut Transformer,
        bpe: &Bpe,
        message: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Reply, ChatError> {
        let context = transformer.context();
        let mut tokens = self.turn_tokens(bpe, message);
        let mut restarted = false;
        if self.pos + tokens.len() + MIN_REPLY_ROOM > context && self.pos > 0 {
            self.reset();
            restarted = true;
            tokens = self.turn_tokens(bpe, message);
        }
        if tokens.len() + MIN_REPLY_ROOM > context {
            return Err(ChatError::TooLong {
                tokens: tokens.len(),
                context,
            });
        }

        let prompt_tokens = tokens.len();
        let mut next = argmax(transformer.prefill_logits(&tokens, self.pos));
        self.pos += prompt_tokens;

        // `next` is always the chosen token not yet fed: the loop feeds it
        // only once it has decided to go on.
        let mut reply_tokens = 0;
        let stop = loop {
            if next == bpe.eos() {
                break Stop::End;
            }
            reply_tokens += 1;
            if !on_token(next) {
                break Stop::Interrupted;
            }
            if reply_tokens >= max_tokens {
                break Stop::Limit;
            }
            // Room for this token and for the next turn to close the
            // reply; past that, the conversation is over.
            if self.pos + 2 >= context {
                break Stop::ContextFull;
            }
            next = argmax(transformer.forward(next, self.pos));
            self.pos += 1;
        };
        self.pending = Some(next);
        if stop == Stop::ContextFull {
            self.reset();
        }
        Ok(Reply {
            prompt_tokens,
            reply_tokens,
            stop,
            restarted,
        })
    }
}

/// What [`Chat::reply`] handed to `on_token`, collected: for tests and
/// for callers that want the reply whole.
pub fn collect(
    chat: &mut Chat,
    transformer: &mut Transformer,
    bpe: &Bpe,
    message: &str,
    max_tokens: usize,
) -> Result<(Vec<u32>, Reply), ChatError> {
    let mut tokens = Vec::new();
    let reply = chat.reply(transformer, bpe, message, max_tokens, |t| {
        tokens.push(t);
        true
    })?;
    Ok((tokens, reply))
}
