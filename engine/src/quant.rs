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
//! blocks' scales. [`quantize`] and [`dot`] are that scheme exactly as
//! ggml's reference C code (`quantize_row_q8_0_ref`,
//! `ggml_vec_dot_q8_0_q8_0_generic`) writes it, so the results are the
//! ones llama.cpp computes, up to the order its SIMD paths add the
//! per-block products in.
//!
//! # The repacked layout
//!
//! A file's 34-byte blocks put every block's 32 values at an odd offset,
//! and the board's target compiles with `strict-align`: told nothing
//! about alignment, the compiler must assume an unaligned vector load
//! could fault, and builds each block a byte at a time instead. That made
//! the whole model several times slower. So a tensor is rearranged in
//! place before use ([`repack`]): all its values first, as 16-byte-aligned
//! [`Block`]s the compiler can load directly, then all its scales — the
//! same bytes, in the same space.

use alloc::vec::Vec;

/// Values in one block.
pub const BLOCK: usize = 32;
/// Bytes in one block as a file stores it: the `f16` scale, then the 32
/// `i8`s.
pub const BLOCK_BYTES: usize = 2 + BLOCK;

/// One block's 32 values, aligned so a vector load of either half is a
/// single aligned instruction.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Block(pub [i8; BLOCK]);

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

/// Rearranges a Q8_0 tensor's file-format blocks in place: every block's
/// values first, then every block's scale. See the module documentation.
///
/// The scratch copy is the size of the tensor.
///
/// # Panics
///
/// If `tensor` is not a whole number of blocks.
pub fn repack(tensor: &mut [u8]) {
    assert!(
        tensor.len().is_multiple_of(BLOCK_BYTES),
        "not whole Q8_0 blocks"
    );
    let blocks = tensor.len() / BLOCK_BYTES;
    let original = tensor.to_vec();
    let (values, scales) = tensor.split_at_mut(blocks * BLOCK);
    for (i, block) in original.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        scales[2 * i..2 * i + 2].copy_from_slice(&block[..2]);
        values[BLOCK * i..BLOCK * (i + 1)].copy_from_slice(&block[2..]);
    }
}

/// A repacked Q8_0 tensor's values and scales, viewed in place.
///
/// `None` unless the values start on a 16-byte boundary, which a GGUF
/// tensor in a buffer aligned to at least 32 always does (the format
/// aligns each tensor to 32).
pub fn repacked(tensor: &[u8]) -> Option<(&[Block], &[u16])> {
    let blocks = tensor.len() / BLOCK_BYTES;
    let (values, scales) = tensor.split_at(blocks * BLOCK);
    if !(values.as_ptr() as usize).is_multiple_of(align_of::<Block>())
        || !(scales.as_ptr() as usize).is_multiple_of(align_of::<u16>())
    {
        return None;
    }
    // SAFETY: both alignments checked above, both lengths are whole
    // elements of the slices they come from, and any bytes are valid `i8`s
    // and `u16`s (little-endian, as every target this runs on).
    unsafe {
        Some((
            core::slice::from_raw_parts(values.as_ptr().cast(), blocks),
            core::slice::from_raw_parts(scales.as_ptr().cast(), blocks),
        ))
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

/// Four blocks' scales, aligned for a single vector load.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct Scales([f32; 4]);

impl Quantized {
    /// Block `b`'s scale.
    fn scale(&self, b: usize) -> f32 {
        self.scales[b / 4].0[b % 4]
    }

    /// Every block's scale, in order.
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
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        if b % 4 == 0 {
            out.scales.push(Scales::default());
        }
        out.scales[b / 4].0[b % 4] = f16_to_f32(f32_to_f16(d));
        out.values
            .push(Block(block.map(|v| libm::roundf(v * id) as i8)));
    }
}

/// The dot product of one repacked Q8_0 row (its blocks' values and
/// scales) with a quantized vector of the same length.
///
/// The per-block products are added in block order, one at a time, as
/// ggml's reference does; only the integer sums inside a block are
/// vectorised (where the target allows), and those are exact in any
/// order. So every implementation of this returns the same bits.
pub fn dot(values: &[Block], scales: &[u16], x: &Quantized) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the Cortex-A76 (and every core this crate runs on as
        // AArch64) implements the dot-product extension.
        unsafe { neon::dot(values, scales, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_scalar(values, scales, x)
    }
}

/// [`dot`], one multiply at a time.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_scalar(values: &[Block], scales: &[u16], x: &Quantized) -> f32 {
    let mut sum = 0.0f32;
    for ((w, &dw), (xv, dx)) in values
        .iter()
        .zip(scales)
        .zip(x.values.iter().zip(x.scale_iter()))
    {
        let mut sumi = 0i32;
        for (&a, &b) in w.0.iter().zip(&xv.0) {
            sumi += i32::from(a) * i32::from(b);
        }
        sum += sumi as f32 * (f16_to_f32(dw) * dx);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::{Block, Quantized};

    /// Loads one aligned half of a block.
    ///
    /// Through a pointer to a 16-byte-aligned type, so the compiler knows
    /// the alignment and emits one `ldr q` even under `strict-align`.
    #[inline(always)]
    fn half(block: &Block, upper: bool) -> int8x16_t {
        let halves: &[int8x16_t; 2] =
            // SAFETY: `Block` is 32 bytes aligned to 16, exactly two
            // `int8x16_t`s, and any bytes are a valid `int8x16_t`.
            unsafe { &*(block as *const Block).cast() };
        halves[usize::from(upper)]
    }

    /// One block's integer dot product: two `sdot`s, then a horizontal
    /// add of the four lanes.
    #[inline(always)]
    unsafe fn block_sum(w: &Block, x: &Block) -> int32x4_t {
        unsafe {
            let acc = vdotq_s32(vdupq_n_s32(0), half(w, false), half(x, false));
            vdotq_s32(acc, half(w, true), half(x, true))
        }
    }

    /// Four `f16`s widened to `f32`s with one instruction.
    #[inline(always)]
    unsafe fn widen(h: &[u16]) -> float32x4_t {
        let out: float32x4_t;
        // SAFETY: `h` has at least four elements (the caller's loop
        // bound); `fcvtl` only reads and writes registers.
        unsafe {
            let bits = vld1_u16(h.as_ptr());
            core::arch::asm!(
                "fcvtl {out:v}.4s, {bits:v}.4h",
                out = lateout(vreg) out,
                bits = in(vreg) bits,
                options(pure, nomem, nostack),
            );
        }
        out
    }

    /// [`super::dot`] with `sdot`, four blocks at a time.
    ///
    /// Each group's four integer sums are reduced into one vector, scaled
    /// as a vector (lane by lane, the same IEEE operations the scalar code
    /// does), then added into the total one lane at a time, in order.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn dot(values: &[Block], scales: &[u16], x: &Quantized) -> f32 {
        let n = values.len();
        let mut sum = 0.0f32;
        let mut b = 0;
        unsafe {
            while b + 4 <= n {
                let s0 = block_sum(&values[b], &x.values[b]);
                let s1 = block_sum(&values[b + 1], &x.values[b + 1]);
                let s2 = block_sum(&values[b + 2], &x.values[b + 2]);
                let s3 = block_sum(&values[b + 3], &x.values[b + 3]);
                // [Σs0, Σs1, Σs2, Σs3]: exact, and well inside `f32`'s
                // 24-bit integer range (|Σ| ≤ 32 * 127 * 127).
                let sums = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
                let dw = widen(&scales[b..b + 4]);
                // `b` is a multiple of four here, so this is one aligned
                // group: a single `ldr q`.
                let dx: float32x4_t = *(&x.scales[b / 4] as *const super::Scales).cast();
                let p = vmulq_f32(vcvtq_f32_s32(sums), vmulq_f32(dw, dx));
                sum += vgetq_lane_f32::<0>(p);
                sum += vgetq_lane_f32::<1>(p);
                sum += vgetq_lane_f32::<2>(p);
                sum += vgetq_lane_f32::<3>(p);
                b += 4;
            }
            while b < n {
                let sumi = vaddvq_s32(block_sum(&values[b], &x.values[b]));
                sum += sumi as f32 * (super::f16_to_f32(scales[b]) * x.scale(b));
                b += 1;
            }
        }
        sum
    }
}

/// Dequantizes one repacked Q8_0 row into `out`: what an embedding lookup
/// does.
pub fn dequantize(values: &[Block], scales: &[u16], out: &mut [f32]) {
    for ((w, &d), out) in values
        .iter()
        .zip(scales)
        .zip(out.as_chunks_mut::<BLOCK>().0)
    {
        let d = f16_to_f32(d);
        for (o, &q) in out.iter_mut().zip(&w.0) {
            *o = d * f32::from(q);
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

    /// `w`, quantized and laid out as a file stores it.
    fn file_blocks(w: &[f32]) -> Vec<u8> {
        let mut q = Quantized::default();
        quantize(w, &mut q);
        let mut bytes = Vec::new();
        for (scale, block) in q.scale_iter().zip(&q.values) {
            bytes.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
            bytes.extend(block.0.iter().map(|&v| v as u8));
        }
        bytes
    }

    /// `bytes` copied into a buffer aligned to 32, as a GGUF tensor in an
    /// aligned file is.
    fn aligned(bytes: &[u8]) -> Vec<[u64; 4]> {
        let mut buf = vec![[0u64; 4]; bytes.len().div_ceil(32)];
        // SAFETY: `buf` has at least `bytes.len()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr().cast(), bytes.len());
        }
        buf
    }

    #[test]
    fn repack_keeps_every_value_and_scale() {
        let w: Vec<f32> = (0..96).map(|i| (i as f32 * 0.11).cos()).collect();
        let file = file_blocks(&w);
        let mut buf = aligned(&file);
        // SAFETY: `buf` holds `file.len()` initialised bytes.
        let tensor =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), file.len()) };
        repack(tensor);
        let (values, scales) = repacked(tensor).unwrap();
        for (i, block) in file.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
            assert_eq!(scales[i], u16::from_le_bytes([block[0], block[1]]));
            assert!(
                values[i]
                    .0
                    .iter()
                    .zip(&block[2..])
                    .all(|(&a, &b)| a as u8 == b)
            );
        }
    }

    #[test]
    fn quantize_then_dot_matches_the_float_dot_closely() {
        let x: Vec<f32> = (0..160).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..160).map(|i| (i as f32 * 0.11).cos()).collect();
        let file = file_blocks(&w);
        let mut buf = aligned(&file);
        // SAFETY: `buf` holds `file.len()` initialised bytes.
        let tensor =
            unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), file.len()) };
        repack(tensor);
        let (values, scales) = repacked(tensor).unwrap();

        let mut qx = Quantized::default();
        quantize(&x, &mut qx);
        let exact: f32 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
        let approx = dot(values, scales, &qx);
        assert!((exact - approx).abs() < 0.1, "{exact} vs {approx}");
        assert_eq!(
            approx.to_bits(),
            dot_scalar(values, scales, &qx).to_bits(),
            "every implementation adds in the same order"
        );

        let mut back = vec![0.0; 160];
        dequantize(values, scales, &mut back);
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
