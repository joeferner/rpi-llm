//! Q8_0, ggml's 8-bit block quantization, and the half-precision floats
//! its scales are stored in.
//!
//! A Q8_0 block is 32 values as `i8`s sharing one scale, an `f16`: the
//! value is `scale * q`. 34 bytes for 32 weights, a little over a byte
//! each.
//!
//! ggml multiplies a Q8_0 matrix by a vector without dequantizing the
//! matrix. It quantizes the *vector* to Q8_0 as well, then each block's
//! dot product is a sum of 32 integer products, scaled once by both
//! blocks' scales. [`quantize`] and the kernels here are that scheme
//! exactly as ggml's reference C code (`quantize_row_q8_0_ref`,
//! `ggml_vec_dot_q8_0_q8_0_generic`) writes it, so the results are the
//! ones llama.cpp computes, up to the order its SIMD paths add the
//! per-block products in.
//!
//! # The repacked layout
//!
//! A file's 34-byte blocks put every block's 32 values at an odd offset,
//! and the board's target compiles with `strict-align`: told nothing
//! about alignment, the compiler must assume an unaligned vector load
//! could fault, and builds each block a byte at a time instead. So a
//! tensor is rearranged in place before use ([`repack`]): all its values
//! first, then all its scales — the same bytes, in the same space.
//!
//! The values are also interleaved four rows at a time, the layout
//! llama.cpp's ARM kernels use: a [`Quad`] holds one block of four
//! consecutive rows, in 4-byte groups — row 0's first four values, row 1's,
//! row 2's, row 3's, then each row's next four. One `sdot` by element
//! multiplies all four rows' groups by the same four values of the input,
//! so its four lanes come out as four rows' sums with nothing left to
//! reduce. [`dot4`] uses that for one token (generation) and [`gemm4`] for
//! four tokens at a time (the prompt).
//!
//! # The same bits everywhere
//!
//! However the work is arranged, every (row, token) result is computed the
//! same way: per block, the integer sum of its 32 products (exact, in any
//! order), times the weight block's scale times the input block's scale,
//! added to a running total in block order. So the one-token and batched
//! kernels, the scalar and NEON versions, and any split across cores all
//! produce identical results, which the tests check.

use alloc::vec::Vec;

use crate::parallel::{Shared, Workers, for_each};

/// Values in one block.
pub const BLOCK: usize = 32;
/// Bytes in one block as a file stores it: the `f16` scale, then the 32
/// `i8`s.
pub const BLOCK_BYTES: usize = 2 + BLOCK;
/// Rows interleaved together in the repacked layout.
pub const QUAD: usize = 4;

/// One block's 32 values, aligned so a vector load of either half is a
/// single aligned instruction.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Block(pub [i8; BLOCK]);

/// One block of four interleaved rows (or tokens): for each 4-byte group
/// `k` of the block, the four rows' values `4k..4k + 4`, row by row — 16
/// bytes a group, eight groups.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Quad(pub [i8; QUAD * BLOCK]);

/// One block's scales for the four rows of a [`Quad`], as `f16` bits.
/// Aligned so the four load as one.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct QuadScales(pub [u16; QUAD]);

/// An `f16`, as its bits, widened to `f32` (exactly: every `f16` is an
/// `f32`).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: renormalise into the `f32` exponent range.
        (0, _) => {
            let shift = man.leading_zeros() - 21;
            let man = (man << shift) & 0x3ff;
            let exp = 127 - 15 + 1 - shift;
            sign | (exp << 23) | (man << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// The `f16` nearest `x` (ties to even), as its bits: ggml's
/// `GGML_FP32_TO_FP16`, and what F16C's conversion instruction does.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal in `f16`, or too small to be anything but zero.
        if e < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = man & ((1 << shift) - 1);
        let mut m = man >> shift;
        if rem > half || (rem == half && m & 1 == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    // A carry out of the mantissa rolls into the exponent, which is the
    // right answer, up to and including overflow to infinity.
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Rearranges a Q8_0 tensor of `cols`-wide rows in place: every [`Quad`]
/// of values first, four rows' block `b` at a time, then every
/// [`QuadScales`]. See the module documentation.
///
/// Returns `false`, leaving the tensor as it was, unless it is whole
/// blocks and its rows come in fours. The scratch copy is the size of the
/// tensor.
pub fn repack(tensor: &mut [u8], cols: usize) -> bool {
    if cols == 0 || !cols.is_multiple_of(BLOCK) {
        return false;
    }
    let per_row = cols / BLOCK;
    let row_bytes = per_row * BLOCK_BYTES;
    if !tensor.len().is_multiple_of(row_bytes) {
        return false;
    }
    let rows = tensor.len() / row_bytes;
    if !rows.is_multiple_of(QUAD) {
        return false;
    }

    let original = tensor.to_vec();
    let (values, scales) = tensor.split_at_mut(rows * cols);
    for q in 0..rows / QUAD {
        for b in 0..per_row {
            let quad = (q * per_row + b) * QUAD * BLOCK;
            for j in 0..QUAD {
                let block =
                    &original[((QUAD * q + j) * per_row + b) * BLOCK_BYTES..][..BLOCK_BYTES];
                let scale = ((q * per_row + b) * QUAD + j) * 2;
                scales[scale..scale + 2].copy_from_slice(&block[..2]);
                for k in 0..BLOCK / 4 {
                    let to = quad + k * 16 + j * 4;
                    values[to..to + 4].copy_from_slice(&block[2 + 4 * k..2 + 4 * k + 4]);
                }
            }
        }
    }
    true
}

/// A repacked Q8_0 tensor's quads and their scales, viewed in place.
///
/// `None` unless the tensor's parts are aligned for them, which a GGUF
/// tensor in a buffer aligned to at least 32 always is (the format aligns
/// each tensor to 32, and the values take a multiple of 128 bytes).
pub fn repacked(tensor: &[u8]) -> Option<(&[Quad], &[QuadScales])> {
    let quads = tensor.len() / (QUAD * BLOCK_BYTES);
    let (values, scales) = tensor.split_at(quads * QUAD * BLOCK);
    if !(values.as_ptr() as usize).is_multiple_of(align_of::<Quad>())
        || !(scales.as_ptr() as usize).is_multiple_of(align_of::<QuadScales>())
    {
        return None;
    }
    // SAFETY: both alignments checked above, both lengths are whole
    // elements of the slices they come from, and any bytes are valid `i8`s
    // and `u16`s (little-endian, as every target this runs on).
    unsafe {
        Some((
            core::slice::from_raw_parts(values.as_ptr().cast(), quads),
            core::slice::from_raw_parts(scales.as_ptr().cast(), quads),
        ))
    }
}

/// Row `j` (0-3) of a quad row — four rows' `values` and `scales`, one
/// [`Quad`] per block — dequantized into `out`: an embedding lookup.
pub fn dequantize_row(values: &[Quad], scales: &[QuadScales], j: usize, out: &mut [f32]) {
    for ((quad, s), out) in values
        .iter()
        .zip(scales)
        .zip(out.as_chunks_mut::<BLOCK>().0)
    {
        let d = f16_to_f32(s.0[j]);
        for k in 0..BLOCK / 4 {
            for i in 0..4 {
                out[4 * k + i] = d * f32::from(quad.0[k * 16 + j * 4 + i]);
            }
        }
    }
}

/// A vector quantized to Q8_0: the scales, already widened from their
/// `f16`s, and the values as aligned [`Block`]s.
#[derive(Default)]
pub struct Quantized {
    /// The scales, four to an aligned group so a vector of four loads in
    /// one instruction; the last group padded with zeros.
    scales: Vec<Scales>,
    values: Vec<Block>,
}

/// Four scales, aligned for a single vector load.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct Scales([f32; 4]);

impl Quantized {
    /// Block `b`'s scale.
    fn scale(&self, b: usize) -> f32 {
        self.scales[b / 4].0[b % 4]
    }

    /// Every block's scale, in order.
    #[cfg_attr(not(test), allow(dead_code))]
    fn scale_iter(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.values.len()).map(|b| self.scale(b))
    }
}

/// Quantizes `x` to Q8_0 into `out`, reusing its storage.
///
/// Each block's scale is its largest magnitude over 127, rounded to `f16`
/// for storage as ggml does; the values are rounded with the unrounded
/// scale's reciprocal, half away from zero (C's `roundf`).
///
/// # Panics
///
/// If `x.len()` is not a multiple of [`BLOCK`].
pub fn quantize(x: &[f32], out: &mut Quantized) {
    assert!(x.len().is_multiple_of(BLOCK), "Q8_0 works in blocks of 32");
    out.scales.clear();
    out.values.clear();
    for (b, block) in x.as_chunks::<BLOCK>().0.iter().enumerate() {
        let (scale, values) = quantize_block(block);
        if b % 4 == 0 {
            out.scales.push(Scales::default());
        }
        out.scales[b / 4].0[b % 4] = scale;
        out.values.push(values);
    }
}

/// One block of [`quantize`]: its scale (already rounded through `f16`)
/// and its values.
fn quantize_block(block: &[f32; BLOCK]) -> (f32, Block) {
    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let values = block.map(|v| round_away(v * id).clamp(-128, 127) as i8);
    (f16_to_f32(f32_to_f16(d)), Block(values))
}

/// `v` rounded to the nearest integer, halves away from zero: C's `roundf`,
/// which ggml's quantizer uses.
///
/// On AArch64 that is one instruction, `fcvtas`, whose rounding rule is
/// exactly `roundf`'s; `libm`'s `roundf` is a software routine, and it was
/// a large part of the time the prompt spent quantizing.
#[inline(always)]
fn round_away(v: f32) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        let r: i32;
        // SAFETY: a register-to-register conversion, no memory touched.
        unsafe {
            core::arch::asm!(
                "fcvtas {r:w}, {v:s}",
                v = in(vreg) v,
                r = lateout(reg) r,
                options(pure, nomem, nostack),
            );
        }
        r
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        libm::roundf(v) as i32
    }
}

/// Four rows' dot products with one quantized vector: the rows are one
/// quad row of a repacked matrix (`values` and `scales`, one entry per
/// block). Generation's kernel.
pub fn dot4(values: &[Quad], scales: &[QuadScales], x: &Quantized) -> [f32; QUAD] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the Cortex-A76 (and every core this crate runs on as
        // AArch64) implements the dot-product extension.
        unsafe { neon::dot4(values, scales, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot4_scalar(values, scales, x)
    }
}

/// [`dot4`], one multiply at a time.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot4_scalar(values: &[Quad], scales: &[QuadScales], x: &Quantized) -> [f32; QUAD] {
    let mut sums = [0.0f32; QUAD];
    for (j, sum) in sums.iter_mut().enumerate() {
        for (b, (quad, s)) in values.iter().zip(scales).enumerate() {
            let xs = &x.values[b].0;
            let mut sumi = 0i32;
            for k in 0..BLOCK / 4 {
                for i in 0..4 {
                    sumi += i32::from(quad.0[k * 16 + j * 4 + i]) * i32::from(xs[4 * k + i]);
                }
            }
            *sum += sumi as f32 * (f16_to_f32(s.0[j]) * x.scale(b));
        }
    }
    sums
}

/// Tokens a batch is processed in groups of.
pub const LANES: usize = 4;

/// Several vectors quantized to Q8_0 together: a batch of tokens' inputs to
/// one matrix multiply, interleaved four tokens at a time the way a
/// [`Quad`] interleaves four rows, block-major so that one weight block
/// meets every token group's matching block.
///
/// The token count is rounded up to [`LANES`]; the padding tokens are
/// zeros, and their results are thrown away.
#[derive(Default)]
pub struct Batch {
    /// Tokens, padding included: a multiple of [`LANES`].
    tokens: usize,
    /// Token group `g`'s block `b` is at `b * groups + g`.
    values: Vec<Quad>,
    /// Their four scales, at the same index.
    scales: Vec<Scales>,
}

impl Batch {
    /// Tokens in the batch, padding included.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Quantizes `xs`, consecutive vectors of `n` values each, into `out`, the
/// vectors split across `workers`.
///
/// Each vector is quantized exactly as [`quantize`] does it, so its blocks
/// and scales are the ones a single-token matrix multiply would use.
///
/// # Panics
///
/// If `n` is not a multiple of [`BLOCK`] or `xs` not a whole number of
/// vectors.
pub fn quantize_batch(workers: &dyn Workers, xs: &[f32], n: usize, out: &mut Batch) {
    assert!(xs.len().is_multiple_of(n), "whole vectors");
    let real = xs.len() / n;
    let tokens = real.next_multiple_of(LANES);
    let groups = tokens / LANES;
    let blocks = n / BLOCK;
    out.tokens = tokens;
    // Zeroed, for the padding tokens; the real ones overwrite their parts.
    out.values.clear();
    out.values.resize(blocks * groups, Quad([0; QUAD * BLOCK]));
    out.scales.clear();
    out.scales.resize(blocks * groups, Scales::default());

    // Four tokens share each quad and each group of scales, each writing
    // its own bytes and lane, so the workers reach them as bytes and
    // lanes: never two `&mut` to one whole quad.
    // SAFETY: `Quad` and `Scales` are plain arrays of `i8` and `f32`
    // (`repr(C)`), so their storage is exactly that many of those.
    let (value_bytes, scale_lanes) = unsafe {
        (
            core::slice::from_raw_parts_mut(
                out.values.as_mut_ptr().cast::<i8>(),
                out.values.len() * QUAD * BLOCK,
            ),
            core::slice::from_raw_parts_mut(
                out.scales.as_mut_ptr().cast::<f32>(),
                out.scales.len() * 4,
            ),
        )
    };
    let values = Shared::new(value_bytes);
    let scales = Shared::new(scale_lanes);
    for_each(workers, real, |t| {
        let (g, m) = (t / LANES, t % LANES);
        let x = &xs[t * n..(t + 1) * n];
        for (b, block) in x.as_chunks::<BLOCK>().0.iter().enumerate() {
            let (scale, q) = quantize_block(block);
            let at = b * groups + g;
            for k in 0..BLOCK / 4 {
                // SAFETY: token `t` alone writes bytes `m * 4..m * 4 + 4`
                // of each 16-byte group of this quad, and lane `m` of its
                // scales; each token runs on one worker.
                unsafe {
                    values
                        .slice(at * QUAD * BLOCK + k * 16 + m * 4, 4)
                        .copy_from_slice(&q.0[4 * k..4 * k + 4]);
                }
            }
            // SAFETY: as above.
            unsafe { scales.slice(at * 4 + m, 1)[0] = scale };
        }
    });
}

/// Four rows' dot products with every vector in `x`, into `out` (one
/// `[f32; 4]` of the four rows' results per token, padding included). The
/// prompt's kernel.
///
/// Each (row, token) result is the same bits [`dot4`] gives for that token
/// alone.
///
/// # Panics
///
/// If `out` is shorter than `x.tokens()`.
pub fn gemm4(values: &[Quad], scales: &[QuadScales], x: &Batch, out: &mut [[f32; QUAD]]) {
    assert!(out.len() >= x.tokens, "a result for every token");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: as in `dot4`.
        unsafe { neon::gemm4(values, scales, x, out) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemm4_scalar(values, scales, x, out)
    }
}

/// [`gemm4`], one token, one row and one multiply at a time.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn gemm4_scalar(values: &[Quad], scales: &[QuadScales], x: &Batch, out: &mut [[f32; QUAD]]) {
    let groups = x.tokens / LANES;
    for (t, o) in out.iter_mut().enumerate().take(x.tokens) {
        let (g, m) = (t / LANES, t % LANES);
        for (j, sum) in o.iter_mut().enumerate() {
            *sum = 0.0;
            for (b, (quad, s)) in values.iter().zip(scales).enumerate() {
                let at = b * groups + g;
                let xs = &x.values[at].0;
                let mut sumi = 0i32;
                for k in 0..BLOCK / 4 {
                    for i in 0..4 {
                        sumi += i32::from(quad.0[k * 16 + j * 4 + i])
                            * i32::from(xs[k * 16 + m * 4 + i]);
                    }
                }
                *sum += sumi as f32 * (f16_to_f32(s.0[j]) * x.scales[at].0[m]);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{Batch, Block, Quad, QuadScales, Quantized, Scales};

    /// A [`Quad`]'s eight 16-byte groups, through a pointer to an aligned
    /// type, so each load is one `ldr q` even under `strict-align`.
    #[inline(always)]
    fn groups(quad: &Quad) -> &[int8x16_t; 8] {
        // SAFETY: `Quad` is 128 bytes aligned to 16, exactly eight
        // `int8x16_t`s, and any bytes are valid ones.
        unsafe { &*(quad as *const Quad).cast() }
    }

    /// A [`Block`]'s two 16-byte halves, likewise.
    #[inline(always)]
    fn halves(block: &Block) -> &[int8x16_t; 2] {
        // SAFETY: `Block` is 32 bytes aligned to 16.
        unsafe { &*(block as *const Block).cast() }
    }

    /// Four rows' `f16` scales widened to `f32`s, one load and one
    /// instruction.
    #[inline(always)]
    unsafe fn widen(s: &QuadScales) -> float32x4_t {
        let out: float32x4_t;
        // SAFETY: `QuadScales` is four `u16`s aligned to 8: one `ldr d`;
        // `fcvtl` only reads and writes registers.
        unsafe {
            let bits: uint16x4_t = *(s as *const QuadScales).cast();
            core::arch::asm!(
                "fcvtl {out:v}.4s, {bits:v}.4h",
                out = lateout(vreg) out,
                bits = in(vreg) bits,
                options(pure, nomem, nostack),
            );
        }
        out
    }

    /// [`super::dot4`] with `sdot` by element.
    ///
    /// Per block: the token's 32 values as two vectors, and eight `sdot`s,
    /// each multiplying the four rows' 4-byte group `k` by the token's
    /// group `k` (a lane of one of the two vectors). The four lanes of the
    /// result are the four rows' integer sums, scaled as a vector and added
    /// into the four rows' running totals — lane by lane the scalar
    /// version's operations, in its order.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn dot4(values: &[Quad], scales: &[QuadScales], x: &Quantized) -> [f32; 4] {
        let mut acc = vdupq_n_f32(0.0);
        unsafe {
            for (b, (quad, s)) in values.iter().zip(scales).enumerate() {
                let w = groups(quad);
                let [xl, xh] = *halves(&x.values[b]);
                let mut sums = vdupq_n_s32(0);
                sums = vdotq_laneq_s32::<0>(sums, w[0], xl);
                sums = vdotq_laneq_s32::<1>(sums, w[1], xl);
                sums = vdotq_laneq_s32::<2>(sums, w[2], xl);
                sums = vdotq_laneq_s32::<3>(sums, w[3], xl);
                sums = vdotq_laneq_s32::<0>(sums, w[4], xh);
                sums = vdotq_laneq_s32::<1>(sums, w[5], xh);
                sums = vdotq_laneq_s32::<2>(sums, w[6], xh);
                sums = vdotq_laneq_s32::<3>(sums, w[7], xh);
                let scale = vmulq_f32(widen(s), vdupq_n_f32(x.scale(b)));
                acc = vaddq_f32(acc, vmulq_f32(vcvtq_f32_s32(sums), scale));
            }
        }
        let mut out = [0.0f32; 4];
        // SAFETY: four `f32`s into a four-`f32` array.
        unsafe { vst1q_f32(out.as_mut_ptr(), acc) };
        out
    }

    /// [`super::gemm4`] with `sdot` by element: four rows by eight tokens
    /// at a time, and four by four for a leftover group.
    ///
    /// Eight tokens rather than four because of `sdot`'s latency: one
    /// group's 32 `sdot`s per block are four chains (a token's accumulator
    /// each) eight deep, too few to keep both vector pipes busy, and they
    /// waited. Two groups make eight chains, and each weight load and
    /// scale conversion serves twice the work.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn gemm4(
        values: &[Quad],
        scales: &[QuadScales],
        x: &Batch,
        out: &mut [[f32; 4]],
    ) {
        let groups_n = x.tokens / 4;
        let mut g = 0;
        unsafe {
            while g + 2 <= groups_n {
                tile::<2>(values, scales, x, g, out);
                g += 2;
            }
            if g < groups_n {
                tile::<1>(values, scales, x, g, out);
            }
        }
    }

    /// Four rows by token groups `first..first + G`.
    ///
    /// Per block: the four rows' quad, each group's quad, and 32 `sdot`s per
    /// group — group `k` of the rows times group `k` of token `m` (lane `m`)
    /// into token `m`'s accumulator, whose lanes are the rows. Nothing is
    /// reduced, and each (row, token) result is scaled and added block by
    /// block exactly as [`dot4`] does.
    #[inline(always)]
    unsafe fn tile<const G: usize>(
        values: &[Quad],
        scales: &[QuadScales],
        x: &Batch,
        first: usize,
        out: &mut [[f32; 4]],
    ) {
        let groups_n = x.tokens / 4;
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; G];
            for (b, (quad, s)) in values.iter().zip(scales).enumerate() {
                let w = groups(quad);
                let at = b * groups_n + first;
                let mut sums = [[vdupq_n_s32(0); 4]; G];
                for (k, &wk) in w.iter().enumerate() {
                    for (h, sums) in sums.iter_mut().enumerate() {
                        let t = groups(&x.values[at + h])[k];
                        sums[0] = vdotq_laneq_s32::<0>(sums[0], wk, t);
                        sums[1] = vdotq_laneq_s32::<1>(sums[1], wk, t);
                        sums[2] = vdotq_laneq_s32::<2>(sums[2], wk, t);
                        sums[3] = vdotq_laneq_s32::<3>(sums[3], wk, t);
                    }
                }
                let dw = widen(s);
                for (h, (acc, sums)) in acc.iter_mut().zip(&sums).enumerate() {
                    let dx: float32x4_t = *(&x.scales[at + h] as *const Scales).cast();
                    let dxs = [
                        vdupq_laneq_f32::<0>(dx),
                        vdupq_laneq_f32::<1>(dx),
                        vdupq_laneq_f32::<2>(dx),
                        vdupq_laneq_f32::<3>(dx),
                    ];
                    for m in 0..4 {
                        acc[m] = vaddq_f32(
                            acc[m],
                            vmulq_f32(vcvtq_f32_s32(sums[m]), vmulq_f32(dw, dxs[m])),
                        );
                    }
                }
            }
            for (h, acc) in acc.iter().enumerate() {
                for (m, acc) in acc.iter().enumerate() {
                    vst1q_f32(out[4 * (first + h) + m].as_mut_ptr(), *acc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_every_finite_value() {
        for h in 0..=0xffffu16 {
            let exp = (h >> 10) & 0x1f;
            if exp == 0x1f {
                continue;
            }
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#06x}");
        }
    }

    #[test]
    fn f32_to_f16_ties_to_even() {
        // 1 + 2^-11 is exactly halfway between 1 and the next f16 up; the
        // even neighbour is 1.
        assert_eq!(f32_to_f16(1.0 + f32::powi(2.0, -11)), 0x3c00);
        // 1 + 3 * 2^-11 is halfway between two odd-even neighbours; even is
        // the upper one.
        assert_eq!(f32_to_f16(1.0 + 3.0 * f32::powi(2.0, -11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "rounds up to infinity");
        assert_eq!(
            f32_to_f16(f32::powi(2.0, -24)),
            0x0001,
            "smallest subnormal"
        );
    }

    const COLS: usize = 160;
    const ROWS: usize = 8;

    /// A `ROWS` by `COLS` matrix of made-up weights.
    fn weights() -> Vec<f32> {
        (0..ROWS * COLS).map(|i| (i as f32 * 0.11).cos()).collect()
    }

    /// `w`, quantized row by row and laid out as a file stores it, in a
    /// buffer aligned to 32 as a GGUF tensor in an aligned file is.
    fn file_tensor(w: &[f32]) -> (Vec<[u64; 4]>, usize) {
        let mut bytes = Vec::new();
        let mut q = Quantized::default();
        for row in w.as_chunks::<COLS>().0 {
            quantize(row, &mut q);
            for (scale, block) in q.scale_iter().zip(&q.values) {
                bytes.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                bytes.extend(block.0.iter().map(|&v| v as u8));
            }
        }
        let mut buf = vec![[0u64; 4]; bytes.len().div_ceil(32)];
        // SAFETY: `buf` has at least `bytes.len()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr().cast(), bytes.len());
        }
        (buf, bytes.len())
    }

    fn bytes_mut(buf: &mut [[u64; 4]], len: usize) -> &mut [u8] {
        // SAFETY: `buf` holds `len` initialised bytes.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), len) }
    }

    #[test]
    fn repack_keeps_every_value_and_scale() {
        let w = weights();
        let (mut buf, len) = file_tensor(&w);
        let file = bytes_mut(&mut buf, len).to_vec();
        let tensor = bytes_mut(&mut buf, len);
        assert!(repack(tensor, COLS));
        let (values, scales) = repacked(tensor).unwrap();

        let per_row = COLS / BLOCK;
        for row in 0..ROWS {
            let (q, j) = (row / QUAD, row % QUAD);
            for b in 0..per_row {
                let block = &file[(row * per_row + b) * BLOCK_BYTES..][..BLOCK_BYTES];
                assert_eq!(
                    scales[q * per_row + b].0[j],
                    u16::from_le_bytes([block[0], block[1]])
                );
                for v in 0..BLOCK {
                    let (k, i) = (v / 4, v % 4);
                    assert_eq!(
                        values[q * per_row + b].0[k * 16 + j * 4 + i] as u8,
                        block[2 + v]
                    );
                }
            }
        }
    }

    #[test]
    fn repack_refuses_rows_that_are_not_in_fours() {
        let mut tensor = vec![0u8; 3 * BLOCK_BYTES];
        assert!(!repack(&mut tensor, BLOCK));
    }

    #[test]
    fn kernels_agree_with_each_other_and_with_floats() {
        let w = weights();
        let (mut buf, len) = file_tensor(&w);
        let tensor = bytes_mut(&mut buf, len);
        assert!(repack(tensor, COLS));
        let (values, scales) = repacked(tensor).unwrap();
        let per_row = COLS / BLOCK;

        // Ten tokens: two full groups of four, which the NEON kernel takes
        // together, and a padded one it takes alone.
        let xs: Vec<f32> = (0..10 * COLS).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut batch = Batch::default();
        quantize_batch(&crate::parallel::Serial, &xs, COLS, &mut batch);
        let mut q = Quantized::default();

        for quad in 0..ROWS / QUAD {
            let rows = &values[quad * per_row..(quad + 1) * per_row];
            let row_scales = &scales[quad * per_row..(quad + 1) * per_row];
            let mut batched = vec![[0.0f32; QUAD]; batch.tokens()];
            gemm4(rows, row_scales, &batch, &mut batched);

            for (t, x) in xs.as_chunks::<COLS>().0.iter().enumerate() {
                quantize(x, &mut q);
                let one = dot4(rows, row_scales, &q);
                assert_eq!(
                    one.map(f32::to_bits),
                    dot4_scalar(rows, row_scales, &q).map(f32::to_bits)
                );
                assert_eq!(
                    one.map(f32::to_bits),
                    batched[t].map(f32::to_bits),
                    "token {t}"
                );
                for j in 0..QUAD {
                    let row = &w[(QUAD * quad + j) * COLS..][..COLS];
                    let exact: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
                    assert!((exact - one[j]).abs() < 0.1, "{exact} vs {}", one[j]);
                }
            }

            for j in 0..QUAD {
                let mut back = vec![0.0; COLS];
                dequantize_row(rows, row_scales, j, &mut back);
                let row = &w[(QUAD * quad + j) * COLS..][..COLS];
                for (a, b) in row.iter().zip(&back) {
                    assert!((a - b).abs() < 0.01);
                }
            }
        }
    }
}
