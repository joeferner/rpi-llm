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

use alloc::vec::Vec;

/// Values in one block.
pub const BLOCK: usize = 32;
/// Bytes in one block: the `f16` scale, then the 32 `i8`s.
pub const BLOCK_BYTES: usize = 2 + BLOCK;

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

/// A vector quantized to Q8_0, kept as separate scales and values rather
/// than interleaved blocks: the layout the dot product wants.
#[derive(Default)]
pub struct Quantized {
    /// One scale per block, already widened from its `f16`.
    scales: Vec<f32>,
    /// The values, `BLOCK` per scale.
    values: Vec<i8>,
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
    for block in x.as_chunks::<BLOCK>().0 {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.scales.push(f16_to_f32(f32_to_f16(d)));
        out.values
            .extend(block.iter().map(|&v| libm::roundf(v * id) as i8));
    }
}

/// The dot product of one Q8_0 row (`row.len() / BLOCK_BYTES` blocks, as
/// stored in the file) with a quantized vector of the same length.
pub fn dot(row: &[u8], x: &Quantized) -> f32 {
    let mut sum = 0.0f32;
    for ((block, &dx), xs) in row
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(&x.scales)
        .zip(x.values.as_chunks::<BLOCK>().0)
    {
        let dw = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut sumi = 0i32;
        for (&w, &v) in block[2..].iter().zip(xs) {
            sumi += i32::from(w as i8) * i32::from(v);
        }
        sum += sumi as f32 * (dw * dx);
    }
    sum
}

/// Dequantizes one Q8_0 row into `out`: what an embedding lookup does.
pub fn dequantize(row: &[u8], out: &mut [f32]) {
    for (block, out) in row
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK>().0)
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (o, &q) in out.iter_mut().zip(&block[2..]) {
            *o = d * f32::from(q as i8);
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

    #[test]
    fn quantize_then_dot_matches_the_float_dot_closely() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).cos()).collect();
        // Quantize the weights into file-format blocks.
        let mut qw = Quantized::default();
        quantize(&w, &mut qw);
        let mut row = Vec::new();
        for (scale, values) in qw.scales.iter().zip(qw.values.as_chunks::<BLOCK>().0) {
            row.extend_from_slice(&f32_to_f16(*scale).to_le_bytes());
            row.extend(values.iter().map(|&v| v as u8));
        }
        let mut qx = Quantized::default();
        quantize(&x, &mut qx);

        let exact: f32 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
        let approx = dot(&row, &qx);
        assert!((exact - approx).abs() < 0.05, "{exact} vs {approx}");

        let mut back = vec![0.0; 64];
        dequantize(&row, &mut back);
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
