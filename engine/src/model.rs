//! The Llama transformer's forward pass, over weights already in memory:
//! llama2.c checkpoints (f32) and GGUF files (Q8_0), the same block either
//! way.
//!
//! The f32 arithmetic is written in llama2.c's `run.c` order on purpose:
//! each sum accumulates left to right, multiply and add are separate
//! operations, and nothing is reassociated. Floating-point addition is not
//! associative, so that is what lets this produce the same logits as that
//! reference to the bit (the math library aside), and what makes "same
//! tokens as llama2.c" a test that can pass exactly rather than a
//! tolerance to argue about. The Q8_0 path follows ggml's reference
//! kernels the same way (see [`crate::quant`]). Faster versions come
//! later, measured against these.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::attention::{self, Floats, dot};
use crate::gguf::{Gguf, Repacked, Tensor, TensorType};
use crate::llama2::{self, FormatError, HEADER_BYTES};
use crate::parallel::{Serial, Shared, Workers, for_chunks, for_each, for_rows};
use crate::quant::{self, BLOCK, QUAD, Quad, QuadScales, Quantized};

/// A model's shape and the constants its arithmetic needs.
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
    /// The longest context the model was trained for.
    pub seq_len: usize,
    /// RoPE's base frequency.
    pub rope_theta: f32,
    /// The epsilon inside RMSNorm's square root.
    pub norm_eps: f32,
}

impl Config {
    /// Width of one attention head.
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Width of the keys and values: one head's worth per key/value head.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_size()
    }
}

impl From<&llama2::Config> for Config {
    /// llama2.c's format has no field for either constant: `run.c` uses
    /// 10000 and 1e-5.
    fn from(c: &llama2::Config) -> Self {
        Self {
            dim: c.dim,
            hidden_dim: c.hidden_dim,
            n_layers: c.n_layers,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            vocab_size: c.vocab_size,
            seq_len: c.seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        }
    }
}

/// Why weights could not be read out of a file.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The llama2.c checkpoint itself is malformed.
    Format(FormatError),
    /// A metadata key the model needs is missing or the wrong type.
    Metadata(&'static str),
    /// A tensor the model needs is missing.
    MissingTensor(String),
    /// A tensor is not the shape the metadata implies.
    Shape(String),
    /// A tensor is stored in a type this engine does not compute with.
    Unsupported(String),
    /// An f32 tensor does not start on a 4-byte boundary, so it cannot be
    /// read in place.
    Misaligned(String),
}

impl From<FormatError> for ModelError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

/// A weight matrix of `rows` rows of `cols` columns, row-major, in
/// whichever form the file stores it.
#[derive(Clone, Copy)]
pub enum Matrix<'a> {
    /// Plain `f32`s.
    F32 {
        /// The elements.
        data: &'a [f32],
        /// Elements per row.
        cols: usize,
    },
    /// Q8_0, repacked four rows at a time (see [`crate::quant`]): each quad
    /// of rows is `cols / 32` [`Quad`]s.
    Q8_0 {
        /// Every quad's blocks, quad by quad.
        values: &'a [Quad],
        /// Their scales, in the same order.
        scales: &'a [QuadScales],
        /// Elements per row.
        cols: usize,
    },
}

impl Matrix<'_> {
    /// Elements per row.
    pub fn cols(&self) -> usize {
        match *self {
            Self::F32 { cols, .. } | Self::Q8_0 { cols, .. } => cols,
        }
    }

    /// Row `i`, as `f32`s, into `out`: an embedding lookup.
    fn row_into(&self, i: usize, out: &mut [f32]) {
        match *self {
            Self::F32 { data, cols } => out.copy_from_slice(&data[i * cols..(i + 1) * cols]),
            Self::Q8_0 {
                values,
                scales,
                cols,
            } => {
                let per_row = cols / BLOCK;
                let quad = i / QUAD * per_row..(i / QUAD + 1) * per_row;
                quant::dequantize_row(&values[quad.clone()], &scales[quad], i % QUAD, out);
            }
        }
    }
}

/// One transformer block's weights.
pub struct Layer<'a> {
    /// RMSNorm before attention, `(dim,)`.
    rms_att: &'a [f32],
    /// Query projection, `(dim, dim)`.
    wq: Matrix<'a>,
    /// Key projection, `(kv_dim, dim)`.
    wk: Matrix<'a>,
    /// Value projection, `(kv_dim, dim)`.
    wv: Matrix<'a>,
    /// Attention output projection, `(dim, dim)`.
    wo: Matrix<'a>,
    /// RMSNorm before the feed-forward layer, `(dim,)`.
    rms_ffn: &'a [f32],
    /// The gate projection, `(hidden_dim, dim)`.
    w1: Matrix<'a>,
    /// The down projection, `(dim, hidden_dim)`.
    w2: Matrix<'a>,
    /// The up projection, `(hidden_dim, dim)`.
    w3: Matrix<'a>,
}

/// The model's weights: views into the file's bytes. Nothing is copied.
pub struct Weights<'a> {
    /// `(vocab_size, dim)`.
    token_embedding: Matrix<'a>,
    /// The transformer blocks.
    layers: Vec<Layer<'a>>,
    /// The last RMSNorm, `(dim,)`.
    rms_final: &'a [f32],
    /// `(vocab_size, dim)`: the embedding table again when shared.
    classifier: Matrix<'a>,
}

impl<'a> Weights<'a> {
    /// Views the weights in a llama2.c `checkpoint`, which
    /// [`llama2::Config::parse`] has already accepted, in `run.c`'s
    /// `memory_map_weights` layout.
    pub fn from_llama2(
        c: &llama2::Config,
        checkpoint: &'a [u8],
    ) -> Result<(Config, Self), ModelError> {
        let config = Config::from(c);
        let body = &checkpoint[HEADER_BYTES..];
        if !(body.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
            return Err(FormatError::Misaligned.into());
        }
        // SAFETY: aligned (checked above), in bounds (the slice is the
        // body's own length in `f32`s), and every bit pattern is a valid
        // `f32`. The file is little-endian, as is every target this runs on.
        let floats: &'a [f32] =
            unsafe { core::slice::from_raw_parts(body.as_ptr().cast(), body.len() / 4) };

        let (dim, hidden, n_layers) = (config.dim, config.hidden_dim, config.n_layers);
        let kv_dim = config.kv_dim();
        let mut rest = floats;
        let mut take = |n: usize| {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };
        let f32s = |data, cols| Matrix::F32 { data, cols };

        let token_embedding = take(config.vocab_size * dim);
        // llama2.c stores each kind of tensor for every layer together.
        let rms_att = take(n_layers * dim);
        let wq = take(n_layers * dim * dim);
        let wk = take(n_layers * dim * kv_dim);
        let wv = take(n_layers * dim * kv_dim);
        let wo = take(n_layers * dim * dim);
        let rms_ffn = take(n_layers * dim);
        let w1 = take(n_layers * dim * hidden);
        let w2 = take(n_layers * hidden * dim);
        let w3 = take(n_layers * dim * hidden);
        let rms_final = take(dim);
        // The obsolete RoPE tables `run.c` skips over.
        take(config.seq_len * config.head_size());
        let classifier = if c.shared_classifier {
            token_embedding
        } else {
            take(config.vocab_size * dim)
        };

        let per = |t: &'a [f32], l: usize, n: usize| &t[l * n..(l + 1) * n];
        let layers = (0..n_layers)
            .map(|l| Layer {
                rms_att: per(rms_att, l, dim),
                wq: f32s(per(wq, l, dim * dim), dim),
                wk: f32s(per(wk, l, dim * kv_dim), dim),
                wv: f32s(per(wv, l, dim * kv_dim), dim),
                wo: f32s(per(wo, l, dim * dim), dim),
                rms_ffn: per(rms_ffn, l, dim),
                w1: f32s(per(w1, l, dim * hidden), dim),
                w2: f32s(per(w2, l, hidden * dim), hidden),
                w3: f32s(per(w3, l, dim * hidden), dim),
            })
            .collect();

        Ok((
            config,
            Self {
                token_embedding: f32s(token_embedding, dim),
                layers,
                rms_final,
                classifier: f32s(classifier, dim),
            },
        ))
    }

    /// Views the weights of a `llama`-architecture GGUF model, and reads
    /// its shape from the metadata.
    ///
    /// Matrices may be F32 or Q8_0, norms must be F32. The output layer is
    /// `output.weight` if there is one, and the embedding table otherwise
    /// (tied weights, as in SmolLM2).
    ///
    /// The RoPE pairing needs no handling here: llama.cpp's converter
    /// permutes the query and key weights so that GGUF `llama` models
    /// rotate adjacent pairs, as llama2.c's checkpoints do.
    ///
    /// Takes the file as [`Gguf::parse_repacked`] returns it, since Q8_0
    /// tensors are computed with in the repacked layout.
    pub fn from_gguf(gguf: &Repacked<'a>) -> Result<(Config, Self), ModelError> {
        let arch = gguf
            .string("general.architecture")
            .ok_or(ModelError::Metadata("general.architecture"))?;
        if arch != "llama" {
            return Err(ModelError::Metadata("general.architecture"));
        }
        let uint = |key: &'static str| {
            gguf.uint(key)
                .map(|v| v as usize)
                .ok_or(ModelError::Metadata(key))
        };
        let float = |key: &'static str| {
            gguf.float(key)
                .map(|v| v as f32)
                .ok_or(ModelError::Metadata(key))
        };
        let tokens = gguf
            .array("tokenizer.ggml.tokens")
            .ok_or(ModelError::Metadata("tokenizer.ggml.tokens"))?;

        let config = Config {
            dim: uint("llama.embedding_length")?,
            hidden_dim: uint("llama.feed_forward_length")?,
            n_layers: uint("llama.block_count")?,
            n_heads: uint("llama.attention.head_count")?,
            n_kv_heads: uint("llama.attention.head_count_kv")?,
            vocab_size: tokens.len,
            seq_len: uint("llama.context_length")?,
            rope_theta: gguf
                .float("llama.rope.freq_base")
                .map_or(10000.0, |v| v as f32),
            norm_eps: float("llama.attention.layer_norm_rms_epsilon")?,
        };
        let (dim, hidden, kv_dim) = (config.dim, config.hidden_dim, config.kv_dim());

        let matrix = |name: &str, rows: usize, cols: usize| -> Result<Matrix<'a>, ModelError> {
            let t = tensor(gguf, name)?;
            if t.n_dims != 2 || t.dims[0] as usize != cols || t.dims[1] as usize != rows {
                return Err(ModelError::Shape(name.into()));
            }
            match t.kind {
                TensorType::F32 => Ok(Matrix::F32 {
                    data: f32_view(t)?,
                    cols,
                }),
                TensorType::Q8_0 if cols.is_multiple_of(BLOCK) && rows.is_multiple_of(QUAD) => {
                    let (values, scales) = quant::repacked(t.data)
                        .ok_or_else(|| ModelError::Misaligned(name.into()))?;
                    Ok(Matrix::Q8_0 {
                        values,
                        scales,
                        cols,
                    })
                }
                _ => Err(ModelError::Unsupported(name.into())),
            }
        };
        let vector = |name: &str, len: usize| -> Result<&'a [f32], ModelError> {
            let t = tensor(gguf, name)?;
            if t.elements() as usize != len || t.kind != TensorType::F32 {
                return Err(ModelError::Shape(name.into()));
            }
            f32_view(t)
        };

        let token_embedding = matrix("token_embd.weight", config.vocab_size, dim)?;
        let layers = (0..config.n_layers)
            .map(|l| {
                let name = |part: &str| format!("blk.{l}.{part}.weight");
                Ok(Layer {
                    rms_att: vector(&name("attn_norm"), dim)?,
                    wq: matrix(&name("attn_q"), dim, dim)?,
                    wk: matrix(&name("attn_k"), kv_dim, dim)?,
                    wv: matrix(&name("attn_v"), kv_dim, dim)?,
                    wo: matrix(&name("attn_output"), dim, dim)?,
                    rms_ffn: vector(&name("ffn_norm"), dim)?,
                    w1: matrix(&name("ffn_gate"), hidden, dim)?,
                    w2: matrix(&name("ffn_down"), dim, hidden)?,
                    w3: matrix(&name("ffn_up"), hidden, dim)?,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let rms_final = vector("output_norm.weight", dim)?;
        let classifier = match gguf.tensor("output.weight") {
            Some(_) => matrix("output.weight", config.vocab_size, dim)?,
            None => token_embedding,
        };

        Ok((
            config,
            Self {
                token_embedding,
                layers,
                rms_final,
                classifier,
            },
        ))
    }
}

/// The tensor called `name`, or the error naming it.
fn tensor<'a, 'g>(gguf: &'g Gguf<'a>, name: &str) -> Result<&'g Tensor<'a>, ModelError> {
    gguf.tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))
}

/// An F32 tensor's data as `f32`s, in place.
fn f32_view<'a>(t: &Tensor<'a>) -> Result<&'a [f32], ModelError> {
    if !(t.data.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
        return Err(ModelError::Misaligned(t.name.into()));
    }
    // SAFETY: aligned (checked above), in bounds (the data's own length in
    // `f32`s), every bit pattern is a valid `f32`, and GGUF is
    // little-endian like every target this runs on.
    Ok(unsafe { core::slice::from_raw_parts(t.data.as_ptr().cast(), t.data.len() / 4) })
}

/// A model ready to run: its shape, its weights, and the activations and
/// key/value cache one sequence needs.
pub struct Transformer<'a> {
    config: Config,
    weights: Weights<'a>,
    /// The positions the key/value cache holds.
    context: usize,
    /// The residual stream, `(dim,)`.
    x: Vec<f32>,
    /// Inside a residual branch, `(dim,)`.
    xb: Vec<f32>,
    /// A second branch buffer, `(dim,)`.
    xb2: Vec<f32>,
    /// The feed-forward layer's hidden activations, `(hidden_dim,)`.
    hb: Vec<f32>,
    /// The same, for the gate's second projection, `(hidden_dim,)`.
    hb2: Vec<f32>,
    /// The query, `(dim,)`.
    q: Vec<f32>,
    /// Attention scores, `(n_heads, context)`.
    att: Vec<f32>,
    /// The output, `(vocab_size,)`.
    logits: Vec<f32>,
    /// Keys for every layer and position so far, `(n_layers, context,
    /// kv_dim)`, aligned for attention's NEON loads.
    key_cache: Floats,
    /// Values, laid out as `key_cache`.
    value_cache: Floats,
    /// What every matrix multiply needs besides its operands.
    mm: MatMul<'a>,
    /// [`Self::prefill`]'s buffers.
    batch: BatchBuffers,
}

/// A source of time for [`Transformer::set_clock`]: the engine has no
/// hardware of its own, so whoever runs it supplies one.
pub trait Clock: Sync {
    /// A free-running count, in [`Self::ticks_per_second`] units.
    fn ticks(&self) -> u64;
    /// How fast [`Self::ticks`] counts.
    fn ticks_per_second(&self) -> u64;
}

/// Where a run of [`Transformer::forward`] or [`Transformer::prefill`]
/// calls spent its time, in clock ticks.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phases {
    /// Calls: tokens for `forward`, batches for `prefill`.
    pub calls: u64,
    /// Everything, start to end of each call.
    pub total: u64,
    /// Quantizing matrix multiplies' inputs.
    pub quantize: u64,
    /// The matrix multiplies themselves, on all the workers.
    pub matmul: u64,
    /// Attention, on all the workers.
    pub attention: u64,
}

impl Phases {
    /// What `total` holds besides the phases named: the per-token steps
    /// (RMSNorm, RoPE, SwiGLU, the residual adds) and buffer setup.
    pub fn other(&self) -> u64 {
        self.total
            .saturating_sub(self.quantize + self.matmul + self.attention)
    }
}

/// [`Phases`] for prompts and for single tokens, kept apart.
#[derive(Clone, Copy, Debug, Default)]
pub struct Profile {
    /// [`Transformer::prefill`]'s batches.
    pub prefill: Phases,
    /// [`Transformer::forward`]'s tokens.
    pub forward: Phases,
}

/// A matrix multiply's scratch space and the workers it is split across.
struct MatMul<'a> {
    /// The input of a Q8_0 matrix multiply, quantized.
    scratch: Quantized,
    /// The inputs of a batched Q8_0 matrix multiply, quantized.
    batch: quant::Batch,
    /// What times the phases, if anything does.
    clock: Option<&'a dyn Clock>,
    /// The phases' times so far.
    profile: Profile,
    /// Whether the time being spent is a prefill's.
    in_prefill: bool,
    /// Who computes the output rows.
    workers: &'a dyn Workers,
}

impl MatMul<'_> {
    /// The clock's time, or 0 with no clock.
    fn now(&self) -> u64 {
        self.clock.map_or(0, |c| c.ticks())
    }

    /// The phases the time being spent belongs to.
    fn phases(&mut self) -> &mut Phases {
        if self.in_prefill {
            &mut self.profile.prefill
        } else {
            &mut self.profile.forward
        }
    }
}

impl<'a> Transformer<'a> {
    /// A model over `weights`, with its buffers allocated and zeroed, and
    /// room in its key/value cache for `context` positions or the model's
    /// own context length, whichever is less.
    ///
    /// The cap matters: the cache is `2 * n_layers * kv_dim` floats a
    /// position, and at SmolLM2's 8192-token context that is 671 MB.
    pub fn new(config: Config, weights: Weights<'a>, context: usize) -> Self {
        let context = context.min(config.seq_len);
        let cache = config.n_layers * context * config.kv_dim();
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * context],
            logits: vec![0.0; config.vocab_size],
            key_cache: Floats::zeroed(cache),
            value_cache: Floats::zeroed(cache),
            batch: BatchBuffers::default(),
            mm: MatMul {
                scratch: Quantized::default(),
                batch: quant::Batch::default(),
                clock: None,
                profile: Profile::default(),
                in_prefill: false,
                workers: &Serial,
            },
            context,
            config,
            weights,
        }
    }

    /// The model's shape.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The positions the key/value cache holds.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Splits every matrix multiply's rows across `workers` from now on.
    /// The results are the same bits as on one worker (see
    /// [`crate::parallel`]).
    pub fn set_workers(&mut self, workers: &'a dyn Workers) {
        self.mm.workers = workers;
    }

    /// Times every call's phases with `clock` from now on (see
    /// [`Self::profile`]). Without a clock nothing is timed.
    pub fn set_clock(&mut self, clock: &'a dyn Clock) {
        self.mm.clock = Some(clock);
    }

    /// Where the calls since the last [`Self::reset_profile`] spent their
    /// time, if a clock is set.
    pub fn profile(&self) -> Profile {
        self.mm.profile
    }

    /// Zeroes [`Self::profile`].
    pub fn reset_profile(&mut self) {
        self.mm.profile = Profile::default();
    }

    /// Runs `token` through the model at position `pos` and returns the
    /// logits for the token after it.
    ///
    /// Positions must be fed in order from 0: each call writes its key and
    /// value into the cache at `pos` and attends over every position up to
    /// it.
    ///
    /// # Panics
    ///
    /// If `pos` is at or past [`Self::context`], or `token` is outside the
    /// vocabulary.
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let c = self.config;
        let w = &self.weights;
        let (dim, ctx) = (c.dim, self.context);
        let head_size = c.head_size();
        let kv_dim = c.kv_dim();
        let kv_mul = c.n_heads / c.n_kv_heads;
        assert!(pos < ctx, "position {pos} is past the context");
        let started = self.mm.now();

        w.token_embedding.row_into(token as usize, &mut self.x);
        // The RoPE angles depend on the position and the pair's place in
        // its head, not the layer: computed once here, used by all of them.
        let mut angles = [(0.0f32, 0.0f32); MAX_HEAD_SIZE / 2];
        rope_angles(c.rope_theta, head_size, pos, &mut angles[..head_size / 2]);

        for (l, layer) in w.layers.iter().enumerate() {
            rmsnorm(&mut self.xb, &self.x, layer.rms_att, c.norm_eps);

            // This position's key and value go straight into the cache.
            let base = l * ctx * kv_dim;
            let at = base + pos * kv_dim;
            matmul(&mut self.q, &self.xb, &layer.wq, &mut self.mm, Input::New);
            matmul(
                &mut self.key_cache[at..at + kv_dim],
                &self.xb,
                &layer.wk,
                &mut self.mm,
                Input::Same,
            );
            matmul(
                &mut self.value_cache[at..at + kv_dim],
                &self.xb,
                &layer.wv,
                &mut self.mm,
                Input::Same,
            );

            // RoPE: rotate each adjacent pair of the query, and of the key
            // while inside it, by an angle set by the position and the
            // pair's place in its head.
            let k = &mut self.key_cache[at..at + kv_dim];
            for i in (0..dim).step_by(2) {
                let (fcr, fci) = angles[(i % head_size) / 2];
                rotate(&mut self.q[i..i + 2], fcr, fci);
                if i < kv_dim {
                    rotate(&mut k[i..i + 2], fcr, fci);
                }
            }

            // Attention over every position so far, the heads split across
            // the workers. Each head reads the shared query and cache and
            // writes only its own scores and its own slice of `xb`, and its
            // arithmetic is the same whichever worker runs it.
            let attention_started = self.mm.now();
            let scale = libm::sqrtf(head_size as f32);
            let scores = Shared::new(&mut self.att);
            let outputs = Shared::new(&mut self.xb);
            let (queries, keys, values) = (&self.q, &*self.key_cache, &*self.value_cache);
            for_each(self.mm.workers, c.n_heads, |h| {
                let q = &queries[h * head_size..(h + 1) * head_size];
                // SAFETY: head `h`'s scores and its slice of `xb` are
                // disjoint from every other head's, and each head runs on
                // one worker.
                let (att, xb) = unsafe {
                    (
                        scores.slice(h * ctx, pos + 1),
                        outputs.slice(h * head_size, head_size),
                    )
                };
                let at = base + (h / kv_mul) * head_size;
                attention::scores(q, &keys[at..], kv_dim, scale, att);
                softmax(att);
                attention::weighted_sum(att, &values[at..], kv_dim, xb);
            });
            let attention_ended = self.mm.now();
            self.mm.phases().attention += attention_ended - attention_started;

            matmul(&mut self.xb2, &self.xb, &layer.wo, &mut self.mm, Input::New);
            for (x, &d) in self.x.iter_mut().zip(&self.xb2) {
                *x += d;
            }

            // The feed-forward layer: w2(silu(w1(x)) * w3(x)).
            rmsnorm(&mut self.xb, &self.x, layer.rms_ffn, c.norm_eps);
            matmul(&mut self.hb, &self.xb, &layer.w1, &mut self.mm, Input::New);
            matmul(
                &mut self.hb2,
                &self.xb,
                &layer.w3,
                &mut self.mm,
                Input::Same,
            );
            // SwiGLU, its `expf`s split across the workers: each element
            // is the same function of its own inputs wherever it runs.
            let gates = Shared::new(&mut self.hb);
            let ups = &self.hb2;
            for_each(self.mm.workers, ups.len(), |i| {
                // SAFETY: each element goes to exactly one worker.
                let h = unsafe { &mut gates.slice(i, 1)[0] };
                *h = swiglu(*h, ups[i]);
            });
            matmul(&mut self.xb, &self.hb, &layer.w2, &mut self.mm, Input::New);
            for (x, &d) in self.x.iter_mut().zip(&self.xb) {
                *x += d;
            }
        }

        // `run.c` normalises `x` in place; a copy of the input keeps the
        // borrow checker satisfied and changes no arithmetic.
        self.xb.copy_from_slice(&self.x);
        rmsnorm(&mut self.x, &self.xb, w.rms_final, c.norm_eps);
        matmul(
            &mut self.logits,
            &self.x,
            &w.classifier,
            &mut self.mm,
            Input::New,
        );

        let ended = self.mm.now();
        let phases = self.mm.phases();
        phases.total += ended - started;
        phases.calls += 1;
        &self.logits
    }

    /// The most tokens [`Self::prefill`] runs through the model together; a
    /// longer run goes through in batches of this many.
    pub const MAX_BATCH: usize = 64;

    /// Runs `tokens` through the model at positions `pos..pos + len`,
    /// together, filling the key/value cache with exactly what feeding them
    /// one at a time through [`Self::forward`] would put there — the same
    /// bits — and computing no logits.
    ///
    /// For a prompt: everything but its last token goes through here, and
    /// the last through `forward`, whose logits choose the first answer
    /// token. Together is faster because each weight is read from memory
    /// once for the whole batch instead of once per token, which turns the
    /// prompt from memory-bound into compute-bound work.
    ///
    /// # Panics
    ///
    /// If the run goes past [`Self::context`], or a token is outside the
    /// vocabulary.
    pub fn prefill(&mut self, tokens: &[u32], pos: usize) {
        for (i, batch) in tokens.chunks(Self::MAX_BATCH).enumerate() {
            self.forward_batch(batch, pos + i * Self::MAX_BATCH);
        }
    }

    /// [`Self::prefill`] for one batch of at most [`Self::MAX_BATCH`].
    ///
    /// The single-token [`Self::forward`] with every per-token buffer
    /// widened to the batch: per-token steps (RMSNorm, RoPE, the residual
    /// adds, SwiGLU) loop over the tokens doing exactly what `forward` does
    /// for one, matrix multiplies take every token's row at once
    /// ([`matmul_batch`]), and attention runs each (token, head) pair as a
    /// job of its own, each token attending over the positions up to its
    /// own.
    fn forward_batch(&mut self, tokens: &[u32], pos0: usize) {
        let c = self.config;
        let w = &self.weights;
        let n = tokens.len();
        let (dim, hidden, ctx) = (c.dim, c.hidden_dim, self.context);
        let head_size = c.head_size();
        let kv_dim = c.kv_dim();
        let kv_mul = c.n_heads / c.n_kv_heads;
        assert!(pos0 + n <= ctx, "positions past the context");
        self.mm.in_prefill = true;
        let started = self.mm.now();

        // The batch's buffers, kept between calls rather than allocated
        // and zeroed for each: every element is written before it is read.
        let mut buffers = core::mem::take(&mut self.batch);
        buffers.reserve(n, &c, ctx);
        let half = head_size / 2;
        let BatchBuffers {
            x,
            xb,
            xb2,
            q,
            hb,
            hb2,
            att,
            angles,
        } = &mut buffers;
        let (x, xb, xb2, q) = (
            &mut x[..n * dim],
            &mut xb[..n * dim],
            &mut xb2[..n * dim],
            &mut q[..n * dim],
        );
        let (hb, hb2) = (&mut hb[..n * hidden], &mut hb2[..n * hidden]);
        let workers = self.mm.workers;

        for (t, &token) in tokens.iter().enumerate() {
            w.token_embedding
                .row_into(token as usize, &mut x[t * dim..(t + 1) * dim]);
            // Each token's RoPE angles, once for every layer.
            rope_angles(
                c.rope_theta,
                head_size,
                pos0 + t,
                &mut angles[t * half..(t + 1) * half],
            );
        }
        let angles = &angles[..n * half];

        for (l, layer) in w.layers.iter().enumerate() {
            // The per-token steps run token by token across the workers:
            // each token's arithmetic is exactly `forward`'s, wherever it
            // runs.
            per_token(workers, xb, dim, |t, xb| {
                rmsnorm(xb, &x[t * dim..(t + 1) * dim], layer.rms_att, c.norm_eps);
            });

            // The batch's keys and values go straight into the cache: its
            // positions are consecutive, so their rows are too.
            let base = l * ctx * kv_dim;
            let at = base + pos0 * kv_dim;
            matmul_batch(q, xb, &layer.wq, &mut self.mm, Input::New);
            matmul_batch(
                &mut self.key_cache[at..at + n * kv_dim],
                xb,
                &layer.wk,
                &mut self.mm,
                Input::Same,
            );
            matmul_batch(
                &mut self.value_cache[at..at + n * kv_dim],
                xb,
                &layer.wv,
                &mut self.mm,
                Input::Same,
            );

            // RoPE, each token at its own position.
            let keys = Shared::new(&mut self.key_cache[at..at + n * kv_dim]);
            per_token(workers, q, dim, |t, qt| {
                // SAFETY: token `t`'s key row is its own.
                let k = unsafe { keys.slice(t * kv_dim, kv_dim) };
                let angles = &angles[t * half..(t + 1) * half];
                for i in (0..dim).step_by(2) {
                    let (fcr, fci) = angles[(i % head_size) / 2];
                    rotate(&mut qt[i..i + 2], fcr, fci);
                    if i < kv_dim {
                        rotate(&mut k[i..i + 2], fcr, fci);
                    }
                }
            });

            // Attention: every (token, head) pair a job of its own, each
            // the same arithmetic `forward` does for that head at that
            // token's position.
            let attention_started = self.mm.now();
            let scale = libm::sqrtf(head_size as f32);
            let scores = Shared::new(&mut att[..n * c.n_heads * ctx]);
            let outputs = Shared::new(xb);
            let (queries, keys, values) = (&*q, &*self.key_cache, &*self.value_cache);
            // Dealt out one pair at a time rather than in contiguous runs:
            // a later token attends over more positions, and contiguous
            // runs would give the last worker the batch's longest ones.
            for_chunks(workers, n * c.n_heads, 1, |i| {
                let (t, h) = (i / c.n_heads, i % c.n_heads);
                let pos = pos0 + t;
                let qh = &queries[t * dim + h * head_size..t * dim + (h + 1) * head_size];
                // SAFETY: pair `i`'s scores and its slice of `xb` are its
                // own, and each pair runs on one worker.
                let (att, out) = unsafe {
                    (
                        scores.slice(i * ctx, pos + 1),
                        outputs.slice(t * dim + h * head_size, head_size),
                    )
                };
                let at = base + (h / kv_mul) * head_size;
                attention::scores(qh, &keys[at..], kv_dim, scale, att);
                softmax(att);
                attention::weighted_sum(att, &values[at..], kv_dim, out);
            });
            let attention_ended = self.mm.now();
            self.mm.phases().attention += attention_ended - attention_started;

            matmul_batch(xb2, xb, &layer.wo, &mut self.mm, Input::New);

            // The residual add and the feed-forward RMSNorm, per token.
            let normed = Shared::new(xb);
            per_token(workers, x, dim, |t, xt| {
                for (x, &d) in xt.iter_mut().zip(&xb2[t * dim..(t + 1) * dim]) {
                    *x += d;
                }
                // SAFETY: token `t`'s row of `xb` is its own.
                let out = unsafe { normed.slice(t * dim, dim) };
                rmsnorm(out, xt, layer.rms_ffn, c.norm_eps);
            });

            matmul_batch(hb, xb, &layer.w1, &mut self.mm, Input::New);
            matmul_batch(hb2, xb, &layer.w3, &mut self.mm, Input::Same);
            per_token(workers, hb, hidden, |t, gates| {
                for (h, &g) in gates.iter_mut().zip(&hb2[t * hidden..(t + 1) * hidden]) {
                    *h = swiglu(*h, g);
                }
            });
            matmul_batch(xb, hb, &layer.w2, &mut self.mm, Input::New);
            per_token(workers, x, dim, |t, xt| {
                for (x, &d) in xt.iter_mut().zip(&xb[t * dim..(t + 1) * dim]) {
                    *x += d;
                }
            });
        }
        self.batch = buffers;

        let ended = self.mm.now();
        let phases = self.mm.phases();
        phases.total += ended - started;
        phases.calls += 1;
        self.mm.in_prefill = false;
    }
}

/// `out = weight * x / rms(x)`, `run.c`'s `rmsnorm`.
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += eps;
    ss = 1.0 / libm::sqrtf(ss);
    for ((o, &w), &v) in out.iter_mut().zip(weight).zip(x) {
        *o = w * (ss * v);
    }
}

/// Runs `f(t, row)` for every token `t` of a batch, `row` being token
/// `t`'s `width` elements of `rows`, the tokens split across `workers`.
fn per_token(
    workers: &dyn Workers,
    rows: &mut [f32],
    width: usize,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    let tokens = rows.len() / width;
    let rows = Shared::new(rows);
    for_each(workers, tokens, |t| {
        // SAFETY: each token's row goes to exactly one worker.
        f(t, unsafe { rows.slice(t * width, width) });
    });
}

/// The buffers [`Transformer::prefill`] works in, kept between calls and
/// grown to the largest batch seen.
#[derive(Default)]
struct BatchBuffers {
    /// The residual stream, `(tokens, dim)`.
    x: Vec<f32>,
    /// Inside a residual branch, `(tokens, dim)`.
    xb: Vec<f32>,
    /// A second branch buffer, `(tokens, dim)`.
    xb2: Vec<f32>,
    /// The queries, `(tokens, dim)`.
    q: Vec<f32>,
    /// The feed-forward layer's hidden activations, `(tokens, hidden_dim)`.
    hb: Vec<f32>,
    /// The same, for the gate's second projection.
    hb2: Vec<f32>,
    /// Attention scores, `(tokens, n_heads, context)`.
    att: Vec<f32>,
    /// Each token's RoPE angles, `(tokens, head_size / 2)`.
    angles: Vec<(f32, f32)>,
}

impl BatchBuffers {
    /// Grows the buffers to hold a batch of `n` tokens, if they do not
    /// already. Their contents are left as they are: every element a batch
    /// reads, it has written first.
    fn reserve(&mut self, n: usize, c: &Config, context: usize) {
        let grow = |v: &mut Vec<f32>, len: usize| {
            if v.len() < len {
                v.resize(len, 0.0);
            }
        };
        grow(&mut self.x, n * c.dim);
        grow(&mut self.xb, n * c.dim);
        grow(&mut self.xb2, n * c.dim);
        grow(&mut self.q, n * c.dim);
        grow(&mut self.hb, n * c.hidden_dim);
        grow(&mut self.hb2, n * c.hidden_dim);
        grow(&mut self.att, n * c.n_heads * context);
        let half = c.head_size() / 2;
        if self.angles.len() < n * half {
            self.angles.resize(n * half, (0.0, 0.0));
        }
    }
}

/// The largest head this engine computes RoPE for, which sizes its table
/// of angles: SmolLM2's heads are 64 wide, stories15M's 48.
const MAX_HEAD_SIZE: usize = 256;

/// RoPE's rotation for every pair of a head at position `pos`, as
/// `(cos, sin)`: `out[p]` for the pair starting at element `2p`.
///
/// The same expressions `run.c` evaluates per pair, per layer — computed
/// once per position instead of once per layer, which gives the same bits
/// and saves all but one layer's `powf`, `cosf` and `sinf`.
///
/// # Panics
///
/// If `head_size` is more than [`MAX_HEAD_SIZE`].
fn rope_angles(theta: f32, head_size: usize, pos: usize, out: &mut [(f32, f32)]) {
    assert!(
        head_size <= MAX_HEAD_SIZE,
        "a head wider than {MAX_HEAD_SIZE}"
    );
    for (p, angle) in out.iter_mut().enumerate() {
        let head_dim = 2 * p;
        let freq = 1.0 / libm::powf(theta, head_dim as f32 / head_size as f32);
        let val = pos as f32 * freq;
        *angle = (libm::cosf(val), libm::sinf(val));
    }
}

/// SwiGLU for one element: `silu(gate) * up`, in `run.c`'s operation
/// order.
fn swiglu(gate: f32, up: f32) -> f32 {
    let mut val = gate;
    val *= 1.0 / (1.0 + libm::expf(-val));
    val *= up;
    val
}

/// In-place softmax, max-subtracted for stability.
fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(x[0], f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Whether a matrix multiply's input is new, or the one the previous
/// matrix multiply was given.
///
/// Several matrices take the same input — the query, key and value
/// projections; the feed-forward layer's gate and up projections — and
/// quantizing it once for all of them gives every one the same quantized
/// input it would have computed itself.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    /// Quantize it.
    New,
    /// Reuse the previous call's quantization: the caller promises the
    /// input is the same values.
    Same,
}

/// How many bytes of weights each worker takes at a time when a matrix's
/// rows are dealt out: see [`for_rows`] for why it is a few KiB.
const CHUNK_BYTES: usize = 4096;

/// `out = w @ x`, one output per row of `w`, the rows dealt out across the
/// workers about [`CHUNK_BYTES`] of weights at a time.
///
/// Where nearly all the time goes. An F32 matrix is the plain loop llama2.c
/// has; a Q8_0 matrix quantizes `x` once, as ggml does, and every worker
/// takes its quads' dot products with [`quant::dot4`] against that.
fn matmul(out: &mut [f32], x: &[f32], w: &Matrix, mm: &mut MatMul, input: Input) {
    let started = mm.now();
    let mut quantizing = 0;
    match *w {
        Matrix::F32 { data, cols } => {
            let chunk = CHUNK_BYTES / (cols * 4);
            for_rows(mm.workers, out, chunk, |i| {
                dot(&data[i * cols..(i + 1) * cols], x)
            });
        }
        Matrix::Q8_0 {
            values,
            scales,
            cols,
        } => {
            if input == Input::New {
                quant::quantize(x, &mut mm.scratch);
                quantizing = mm.now() - started;
            }
            let per_row = cols / BLOCK;
            let scratch = &mm.scratch;
            let quads = out.len() / QUAD;
            let outputs = Shared::new(out);
            for_chunks(mm.workers, quads, quads_per_chunk(cols), |q| {
                let quad = q * per_row..(q + 1) * per_row;
                let sums = quant::dot4(&values[quad.clone()], &scales[quad], scratch);
                // SAFETY: quad `q` goes to exactly one worker, and its
                // outputs are rows `4q..4q + 4`.
                unsafe { outputs.slice(QUAD * q, QUAD) }.copy_from_slice(&sums);
            });
        }
    }
    mm.book(started, quantizing);
}

impl MatMul<'_> {
    /// Books a matrix multiply that began at `started` and spent
    /// `quantizing` of its time quantizing its input.
    fn book(&mut self, started: u64, quantizing: u64) {
        let elapsed = self.now() - started;
        let phases = self.phases();
        phases.quantize += quantizing;
        phases.matmul += elapsed - quantizing;
    }
}

/// How many quads of `cols`-wide Q8_0 rows make up about [`CHUNK_BYTES`]
/// of weights, at least one: a row's values are a byte each, and its scales
/// a sixteenth as much again in a separate array.
fn quads_per_chunk(cols: usize) -> usize {
    (CHUNK_BYTES / (QUAD * cols)).max(1)
}

/// [`matmul`] for a batch: `xs` holds one input vector per token, and
/// `out` gets token `t`'s output for row `r` at `t * rows + r`.
///
/// The rows are dealt out to the workers as in [`matmul`], and each worker
/// computes all of a row's tokens while the row is in hand — for Q8_0 a
/// quad of rows at a time with [`quant::gemm4`], whose results are the same
/// bits [`quant::dot4`] gives one token at a time.
fn matmul_batch(out: &mut [f32], xs: &[f32], w: &Matrix, mm: &mut MatMul, input: Input) {
    let cols = w.cols();
    let n = xs.len() / cols;
    let rows = out.len() / n;
    let outputs = Shared::new(out);
    let started = mm.now();
    let mut quantizing = 0;
    match *w {
        Matrix::F32 { data, cols } => {
            for_chunks(mm.workers, rows, CHUNK_BYTES / (cols * 4), |r| {
                let row = &data[r * cols..(r + 1) * cols];
                for t in 0..n {
                    // SAFETY: row `r` goes to exactly one worker, and
                    // `t * rows + r` is that row's output for token `t`.
                    unsafe {
                        outputs.slice(t * rows + r, 1)[0] = dot(row, &xs[t * cols..(t + 1) * cols])
                    };
                }
            });
        }
        Matrix::Q8_0 {
            values,
            scales,
            cols,
        } => {
            if input == Input::New {
                quant::quantize_batch(mm.workers, xs, cols, &mut mm.batch);
                quantizing = mm.now() - started;
            }
            let per_row = cols / BLOCK;
            let batch = &mm.batch;
            for_chunks(mm.workers, rows / QUAD, quads_per_chunk(cols), |q| {
                let quad = q * per_row..(q + 1) * per_row;
                let mut sums = [[0.0f32; QUAD]; Transformer::MAX_BATCH];
                quant::gemm4(&values[quad.clone()], &scales[quad], batch, &mut sums);
                for (t, sums) in sums.iter().enumerate().take(n) {
                    // SAFETY: quad `q` goes to exactly one worker, and
                    // token `t`'s outputs for its rows are
                    // `t * rows + 4q..t * rows + 4q + 4`.
                    unsafe { outputs.slice(t * rows + QUAD * q, QUAD) }.copy_from_slice(sums);
                }
            });
        }
    }
    mm.book(started, quantizing);
}

/// Rotates the pair `v` by the angle whose cosine and sine are given.
fn rotate(v: &mut [f32], cos: f32, sin: f32) {
    let (v0, v1) = (v[0], v[1]);
    v[0] = v0 * cos - v1 * sin;
    v[1] = v0 * sin + v1 * cos;
}
