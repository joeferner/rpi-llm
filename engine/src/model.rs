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

use crate::gguf::{Gguf, Repacked, Tensor, TensorType};
use crate::llama2::{self, FormatError, HEADER_BYTES};
use crate::parallel::{Serial, Shared, Workers, for_each, for_rows};
use crate::quant::{self, BLOCK, Block, Quantized};

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
    /// Q8_0, repacked (see [`crate::quant`]): `cols / 32` blocks per row.
    Q8_0 {
        /// Every block's values, row by row.
        values: &'a [Block],
        /// Every block's `f16` scale, in the same order.
        scales: &'a [u16],
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
                let row = cols / BLOCK * i..cols / BLOCK * (i + 1);
                quant::dequantize(&values[row.clone()], &scales[row], out);
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
                TensorType::Q8_0 if cols.is_multiple_of(BLOCK) => {
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
    /// kv_dim)`.
    key_cache: Vec<f32>,
    /// Values, laid out as `key_cache`.
    value_cache: Vec<f32>,
    /// What every matrix multiply needs besides its operands.
    mm: MatMul<'a>,
}

/// A matrix multiply's scratch space and the workers it is split across.
struct MatMul<'a> {
    /// The input of a Q8_0 matrix multiply, quantized.
    scratch: Quantized,
    /// Who computes the output rows.
    workers: &'a dyn Workers,
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
            key_cache: vec![0.0; cache],
            value_cache: vec![0.0; cache],
            mm: MatMul {
                scratch: Quantized::default(),
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

        w.token_embedding.row_into(token as usize, &mut self.x);

        for (l, layer) in w.layers.iter().enumerate() {
            rmsnorm(&mut self.xb, &self.x, layer.rms_att, c.norm_eps);

            // This position's key and value go straight into the cache.
            let base = l * ctx * kv_dim;
            let at = base + pos * kv_dim;
            matmul(&mut self.q, &self.xb, &layer.wq, &mut self.mm);
            matmul(
                &mut self.key_cache[at..at + kv_dim],
                &self.xb,
                &layer.wk,
                &mut self.mm,
            );
            matmul(
                &mut self.value_cache[at..at + kv_dim],
                &self.xb,
                &layer.wv,
                &mut self.mm,
            );

            // RoPE: rotate each adjacent pair of the query, and of the key
            // while inside it, by an angle set by the position and the
            // pair's place in its head.
            let k = &mut self.key_cache[at..at + kv_dim];
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / libm::powf(c.rope_theta, head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let (fcr, fci) = (libm::cosf(val), libm::sinf(val));
                rotate(&mut self.q[i..i + 2], fcr, fci);
                if i < kv_dim {
                    rotate(&mut k[i..i + 2], fcr, fci);
                }
            }

            // Attention over every position so far, the heads split across
            // the workers. Each head reads the shared query and cache and
            // writes only its own scores and its own slice of `xb`, and its
            // arithmetic is the same whichever worker runs it.
            let scale = libm::sqrtf(head_size as f32);
            let scores = Shared::new(&mut self.att);
            let outputs = Shared::new(&mut self.xb);
            let (queries, keys, values) = (&self.q, &self.key_cache, &self.value_cache);
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
                let kv_head = (h / kv_mul) * head_size;
                for (t, score) in att.iter_mut().enumerate() {
                    let at = base + t * kv_dim + kv_head;
                    *score = dot(q, &keys[at..at + head_size]) / scale;
                }
                softmax(att);

                xb.fill(0.0);
                for (t, &a) in att.iter().enumerate() {
                    let at = base + t * kv_dim + kv_head;
                    for (out, &v) in xb.iter_mut().zip(&values[at..at + head_size]) {
                        *out += a * v;
                    }
                }
            });

            matmul(&mut self.xb2, &self.xb, &layer.wo, &mut self.mm);
            for (x, &d) in self.x.iter_mut().zip(&self.xb2) {
                *x += d;
            }

            // The feed-forward layer: w2(silu(w1(x)) * w3(x)).
            rmsnorm(&mut self.xb, &self.x, layer.rms_ffn, c.norm_eps);
            matmul(&mut self.hb, &self.xb, &layer.w1, &mut self.mm);
            matmul(&mut self.hb2, &self.xb, &layer.w3, &mut self.mm);
            for (h, &g) in self.hb.iter_mut().zip(&self.hb2) {
                let mut val = *h;
                val *= 1.0 / (1.0 + libm::expf(-val));
                val *= g;
                *h = val;
            }
            matmul(&mut self.xb, &self.hb, &layer.w2, &mut self.mm);
            for (x, &d) in self.x.iter_mut().zip(&self.xb) {
                *x += d;
            }
        }

        // `run.c` normalises `x` in place; a copy of the input keeps the
        // borrow checker satisfied and changes no arithmetic.
        self.xb.copy_from_slice(&self.x);
        rmsnorm(&mut self.x, &self.xb, w.rms_final, c.norm_eps);
        matmul(&mut self.logits, &self.x, &w.classifier, &mut self.mm);
        &self.logits
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

/// `out = w @ x`, one output per row of `w`, the rows split across the
/// workers.
///
/// Where nearly all the time goes. An F32 matrix is the plain loop llama2.c
/// has; a Q8_0 matrix quantizes `x` once, as ggml does, and every worker
/// takes its rows' dot products with [`quant::dot`] against that.
fn matmul(out: &mut [f32], x: &[f32], w: &Matrix, mm: &mut MatMul) {
    match *w {
        Matrix::F32 { data, cols } => {
            for_rows(mm.workers, out, |i| dot(&data[i * cols..(i + 1) * cols], x));
        }
        Matrix::Q8_0 {
            values,
            scales,
            cols,
        } => {
            quant::quantize(x, &mut mm.scratch);
            let per_row = cols / BLOCK;
            let scratch = &mm.scratch;
            for_rows(mm.workers, out, |i| {
                let row = i * per_row..(i + 1) * per_row;
                quant::dot(&values[row.clone()], &scales[row], scratch)
            });
        }
    }
}

/// A left-to-right dot product.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        sum += a * b;
    }
    sum
}

/// Rotates the pair `v` by the angle whose cosine and sine are given.
fn rotate(v: &mut [f32], cos: f32, sin: f32) {
    let (v0, v1) = (v[0], v[1]);
    v[0] = v0 * cos - v1 * sin;
    v[1] = v0 * sin + v1 * cos;
}
