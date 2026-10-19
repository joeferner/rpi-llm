//! The Llama 2 transformer's forward pass, as llama2.c's `run.c` computes
//! it, over a checkpoint already in memory.
//!
//! The arithmetic is written in `run.c`'s order on purpose: each sum
//! accumulates left to right, multiply and add are separate operations,
//! and nothing is reassociated. Floating-point addition is not
//! associative, so that is what lets this produce the same logits as the
//! reference to the bit (the math library aside), and what makes "same
//! tokens as llama2.c" a test that can pass exactly rather than a
//! tolerance to argue about. Faster versions come later, measured against
//! this one.

use alloc::vec;
use alloc::vec::Vec;

use crate::llama2::{Config, FormatError, HEADER_BYTES};

/// The model's weights: typed views into the checkpoint's bytes, in
/// `run.c`'s `memory_map_weights` layout. Nothing is copied.
pub struct Weights<'a> {
    /// `(vocab_size, dim)`.
    token_embedding: &'a [f32],
    /// `(n_layers, dim)`.
    rms_att: &'a [f32],
    /// `(n_layers, dim, dim)`.
    wq: &'a [f32],
    /// `(n_layers, kv_dim, dim)`.
    wk: &'a [f32],
    /// `(n_layers, kv_dim, dim)`.
    wv: &'a [f32],
    /// `(n_layers, dim, dim)`.
    wo: &'a [f32],
    /// `(n_layers, dim)`.
    rms_ffn: &'a [f32],
    /// `(n_layers, hidden_dim, dim)`.
    w1: &'a [f32],
    /// `(n_layers, dim, hidden_dim)`.
    w2: &'a [f32],
    /// `(n_layers, hidden_dim, dim)`.
    w3: &'a [f32],
    /// `(dim,)`.
    rms_final: &'a [f32],
    /// `(vocab_size, dim)`: the embedding table again when shared.
    classifier: &'a [f32],
}

impl<'a> Weights<'a> {
    /// Views the weights in `checkpoint`, which [`Config::parse`] has
    /// already accepted.
    ///
    /// # Errors
    ///
    /// [`FormatError::Misaligned`] unless the bytes after the header start
    /// on a 4-byte boundary: they are read in place as `f32`s.
    pub fn new(config: &Config, checkpoint: &'a [u8]) -> Result<Self, FormatError> {
        let body = &checkpoint[HEADER_BYTES..];
        if !(body.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
            return Err(FormatError::Misaligned);
        }
        // SAFETY: aligned (checked above), in bounds (the slice is the
        // body's own length in `f32`s), and every bit pattern is a valid
        // `f32`. The file is little-endian, as is every target this runs on.
        let floats: &'a [f32] =
            unsafe { core::slice::from_raw_parts(body.as_ptr().cast(), body.len() / 4) };

        let (dim, hidden, layers) = (config.dim, config.hidden_dim, config.n_layers);
        let kv_dim = config.n_kv_heads * config.head_size();
        let mut rest = floats;
        let mut take = |n: usize| {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head
        };

        let token_embedding = take(config.vocab_size * dim);
        let rms_att = take(layers * dim);
        let wq = take(layers * dim * dim);
        let wk = take(layers * dim * kv_dim);
        let wv = take(layers * dim * kv_dim);
        let wo = take(layers * dim * dim);
        let rms_ffn = take(layers * dim);
        let w1 = take(layers * dim * hidden);
        let w2 = take(layers * hidden * dim);
        let w3 = take(layers * dim * hidden);
        let rms_final = take(dim);
        // The obsolete RoPE tables `run.c` skips over.
        take(config.seq_len * config.head_size());
        let classifier = if config.shared_classifier {
            token_embedding
        } else {
            take(config.vocab_size * dim)
        };

        Ok(Self {
            token_embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            classifier,
        })
    }
}

/// A model ready to run: its shape, its weights, and the activations and
/// key/value cache one sequence needs.
pub struct Transformer<'a> {
    config: Config,
    weights: Weights<'a>,
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
    /// Attention scores, `(n_heads, seq_len)`.
    att: Vec<f32>,
    /// The output, `(vocab_size,)`.
    logits: Vec<f32>,
    /// Keys for every layer and position so far, `(n_layers, seq_len,
    /// kv_dim)`.
    key_cache: Vec<f32>,
    /// Values, laid out as `key_cache`.
    value_cache: Vec<f32>,
}

impl<'a> Transformer<'a> {
    /// A model over `weights`, with its buffers allocated and zeroed.
    pub fn new(config: Config, weights: Weights<'a>) -> Self {
        let kv_dim = config.n_kv_heads * config.head_size();
        let cache = config.n_layers * config.seq_len * kv_dim;
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; cache],
            value_cache: vec![0.0; cache],
            config,
            weights,
        }
    }

    /// The model's shape.
    pub fn config(&self) -> &Config {
        &self.config
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
    /// If `pos` is at or past the context length, or `token` is outside
    /// the vocabulary.
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let c = self.config;
        let w = &self.weights;
        let (dim, hidden) = (c.dim, c.hidden_dim);
        let head_size = c.head_size();
        let kv_dim = c.n_kv_heads * head_size;
        let kv_mul = c.n_heads / c.n_kv_heads;
        assert!(pos < c.seq_len, "position {pos} is past the context");

        let token = token as usize;
        self.x
            .copy_from_slice(&w.token_embedding[token * dim..(token + 1) * dim]);

        for l in 0..c.n_layers {
            rmsnorm(&mut self.xb, &self.x, &w.rms_att[l * dim..(l + 1) * dim]);

            // This position's key and value go straight into the cache.
            let layer = l * c.seq_len * kv_dim;
            let at = layer + pos * kv_dim;
            matmul(&mut self.q, &self.xb, layer_of(w.wq, l, dim * dim));
            matmul(
                &mut self.key_cache[at..at + kv_dim],
                &self.xb,
                layer_of(w.wk, l, dim * kv_dim),
            );
            matmul(
                &mut self.value_cache[at..at + kv_dim],
                &self.xb,
                layer_of(w.wv, l, dim * kv_dim),
            );

            // RoPE: rotate each pair of the query, and of the key while
            // inside it, by an angle set by the position and the pair's
            // place in its head.
            let k = &mut self.key_cache[at..at + kv_dim];
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / libm::powf(10000.0, head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let (fcr, fci) = (libm::cosf(val), libm::sinf(val));
                rotate(&mut self.q[i..i + 2], fcr, fci);
                if i < kv_dim {
                    rotate(&mut k[i..i + 2], fcr, fci);
                }
            }

            // Attention, head by head, over every position so far.
            let scale = libm::sqrtf(head_size as f32);
            for h in 0..c.n_heads {
                let q = &self.q[h * head_size..(h + 1) * head_size];
                let att = &mut self.att[h * c.seq_len..h * c.seq_len + pos + 1];
                let kv_head = (h / kv_mul) * head_size;
                for (t, score) in att.iter_mut().enumerate() {
                    let base = layer + t * kv_dim + kv_head;
                    let k = &self.key_cache[base..base + head_size];
                    *score = dot(q, k) / scale;
                }
                softmax(att);

                let xb = &mut self.xb[h * head_size..(h + 1) * head_size];
                xb.fill(0.0);
                for (t, &a) in att.iter().enumerate() {
                    let base = layer + t * kv_dim + kv_head;
                    let v = &self.value_cache[base..base + head_size];
                    for (out, &v) in xb.iter_mut().zip(v) {
                        *out += a * v;
                    }
                }
            }

            matmul(&mut self.xb2, &self.xb, layer_of(w.wo, l, dim * dim));
            for (x, &d) in self.x.iter_mut().zip(&self.xb2) {
                *x += d;
            }

            // The feed-forward layer: w2(silu(w1(x)) * w3(x)).
            rmsnorm(&mut self.xb, &self.x, &w.rms_ffn[l * dim..(l + 1) * dim]);
            matmul(&mut self.hb, &self.xb, layer_of(w.w1, l, dim * hidden));
            matmul(&mut self.hb2, &self.xb, layer_of(w.w3, l, dim * hidden));
            for (h, &g) in self.hb.iter_mut().zip(&self.hb2) {
                let mut val = *h;
                val *= 1.0 / (1.0 + libm::expf(-val));
                val *= g;
                *h = val;
            }
            matmul(&mut self.xb, &self.hb, layer_of(w.w2, l, hidden * dim));
            for (x, &d) in self.x.iter_mut().zip(&self.xb) {
                *x += d;
            }
        }

        // `run.c` normalises `x` in place; a copy of the input keeps the
        // borrow checker satisfied and changes no arithmetic.
        self.xb.copy_from_slice(&self.x);
        rmsnorm(&mut self.x, &self.xb, w.rms_final);
        matmul(&mut self.logits, &self.x, w.classifier);
        &self.logits
    }
}

/// Layer `l`'s slice of a tensor stacked `size` elements per layer.
fn layer_of(tensor: &[f32], l: usize, size: usize) -> &[f32] {
    &tensor[l * size..(l + 1) * size]
}

/// `out = weight * x / rms(x)`, `run.c`'s `rmsnorm`.
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
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

/// `out = w @ x` for `w` of shape `(out.len(), x.len())`, row-major.
///
/// Where nearly all the time goes, and where the faster versions will go:
/// this one is the plain loop `run.c` has, kept as the reference they are
/// checked against.
fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot(&w[i * n..(i + 1) * n], x);
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
