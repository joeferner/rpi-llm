//! The llama2.c file formats: a model checkpoint (`stories15M.bin`) and its
//! tokenizer (`tokenizer.bin`), as Karpathy's `run.c` reads them.
//!
//! Only the parsing and the checks that the files are what they claim to
//! be. Nothing here runs the model yet.

/// A checkpoint's header: seven little-endian `i32`s ahead of the weights.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Width of the residual stream.
    pub dim: usize,
    /// Width of the feed-forward layer.
    pub hidden_dim: usize,
    /// Transformer layers.
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads (fewer than `n_heads` under grouped-query attention).
    pub n_kv_heads: usize,
    /// Tokens in the vocabulary.
    pub vocab_size: usize,
    /// Longest sequence the model was trained on, and so the context.
    pub seq_len: usize,
    /// Whether the output classifier reuses the token embedding table.
    /// llama2.c says so with a positive vocabulary size; a negative one
    /// means a separate classifier follows the other weights.
    pub shared_classifier: bool,
}

/// Bytes in a checkpoint's header.
pub const HEADER_BYTES: usize = 7 * 4;

/// Why a file did not parse as what it was meant to be.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "the fields are read through `Debug`, in the failure message"
)]
pub enum FormatError {
    /// Shorter than its header.
    Truncated,
    /// A header field is zero, negative where it cannot be, or inconsistent
    /// with another (heads that do not divide the width).
    BadHeader,
    /// The weights after the header are not the size the header implies.
    WrongSize {
        /// The bytes the header calls for, header included.
        expected: usize,
        /// The bytes the file has.
        actual: usize,
    },
    /// The tokenizer ran out of bytes partway through a token, or a
    /// token's length is negative.
    BadToken {
        /// Which token, counting from 0.
        index: usize,
    },
    /// The tokenizer had bytes left over after the last token.
    TrailingBytes {
        /// How many.
        count: usize,
    },
}

impl Config {
    /// Parses the header at the start of `checkpoint` and checks the rest
    /// of the file is exactly the weights it describes.
    ///
    /// The size check is the strong one: every weight tensor's shape comes
    /// from the header, so a file whose length matches to the byte has
    /// both the right header and all of its weights.
    pub fn parse(checkpoint: &[u8]) -> Result<Self, FormatError> {
        let header = checkpoint
            .get(..HEADER_BYTES)
            .ok_or(FormatError::Truncated)?;
        let field = |i: usize| i32::from_le_bytes(header[4 * i..4 * i + 4].try_into().unwrap());
        let positive = |value: i32| usize::try_from(value).ok().filter(|&v| v > 0);

        let vocab = field(5);
        let config = Self {
            dim: positive(field(0)).ok_or(FormatError::BadHeader)?,
            hidden_dim: positive(field(1)).ok_or(FormatError::BadHeader)?,
            n_layers: positive(field(2)).ok_or(FormatError::BadHeader)?,
            n_heads: positive(field(3)).ok_or(FormatError::BadHeader)?,
            n_kv_heads: positive(field(4)).ok_or(FormatError::BadHeader)?,
            vocab_size: positive(vocab.saturating_abs()).ok_or(FormatError::BadHeader)?,
            seq_len: positive(field(6)).ok_or(FormatError::BadHeader)?,
            shared_classifier: vocab > 0,
        };
        if !config.dim.is_multiple_of(config.n_heads)
            || !config.n_heads.is_multiple_of(config.n_kv_heads)
        {
            return Err(FormatError::BadHeader);
        }

        let expected = HEADER_BYTES + 4 * config.weight_count();
        if checkpoint.len() != expected {
            return Err(FormatError::WrongSize {
                expected,
                actual: checkpoint.len(),
            });
        }
        Ok(config)
    }

    /// Width of one attention head.
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// How many `f32`s follow the header, in `run.c`'s
    /// `memory_map_weights` order.
    pub fn weight_count(&self) -> usize {
        let (dim, hidden, layers) = (self.dim, self.hidden_dim, self.n_layers);
        let kv_dim = self.n_kv_heads * self.head_size();

        let embedding = self.vocab_size * dim;
        let per_layer = dim // rms_att_weight
            + dim * dim // wq
            + 2 * dim * kv_dim // wk, wv
            + dim * dim // wo
            + dim // rms_ffn_weight
            + 3 * dim * hidden; // w1, w2, w3
        let final_norm = dim;
        // `freq_cis_real` and `freq_cis_imag`: precomputed RoPE tables from
        // an older version of the format, still in the file, and skipped
        // by `run.c`, which computes RoPE itself.
        let rope_tables = self.seq_len * self.head_size();
        let classifier = if self.shared_classifier {
            0
        } else {
            self.vocab_size * dim
        };

        embedding + layers * per_layer + final_norm + rope_tables + classifier
    }
}

/// What a tokenizer file holds, as far as the checks here go.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerSummary {
    /// The longest token, in bytes, as the file's header declares it.
    pub max_token_length: usize,
    /// The tokens read.
    pub tokens: usize,
}

/// Walks `tokenizer` — an `i32` maximum length, then per token an `f32`
/// merge score, an `i32` length and that many bytes — and checks it holds
/// exactly `vocab_size` tokens with nothing left over.
pub fn check_tokenizer(
    tokenizer: &[u8],
    vocab_size: usize,
) -> Result<TokenizerSummary, FormatError> {
    let read_i32 = |at: usize| {
        tokenizer
            .get(at..at + 4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
    };

    let max_token_length = read_i32(0)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or(FormatError::Truncated)?;

    let mut at = 4;
    for index in 0..vocab_size {
        // The score is skipped; only its four bytes matter here.
        let len = read_i32(at + 4)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(FormatError::BadToken { index })?;
        at += 8;
        if at + len > tokenizer.len() {
            return Err(FormatError::BadToken { index });
        }
        at += len;
    }
    if at != tokenizer.len() {
        return Err(FormatError::TrailingBytes {
            count: tokenizer.len() - at,
        });
    }

    Ok(TokenizerSummary {
        max_token_length,
        tokens: vocab_size,
    })
}
