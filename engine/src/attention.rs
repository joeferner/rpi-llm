//! Attention's two loops over the cached positions, for one head: the
//! query's score against every key, and the scores' weighted sum of the
//! values.
//!
//! Both are `run.c`'s loops. The scalar versions are those loops as they
//! are; the NEON ones compute the same bits, because what they vectorize
//! across is independent work rather than a sum's terms:
//!
//! - Scores run four positions at once, a lane each. Four keys' next four
//!   elements are loaded and transposed so that each lane gets its own
//!   key's, and each lane then multiplies and adds them in the order the
//!   scalar dot product does.
//! - The weighted sum runs four of the head's elements at once, a lane
//!   each, every lane adding up the positions in order.
//!
//! Multiply and add stay separate instructions, as in the scalar code; a
//! fused multiply-add rounds once instead of twice, and would give
//! different bits.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

/// Four `f32`s aligned to 16 bytes: the unit [`Floats`] is stored in.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct Lanes([f32; 4]);

/// A zeroed `f32` buffer whose start is aligned to 16 bytes, so that the
/// NEON loops can load four elements with one `ldr q`; the target's
/// `strict-align` turns a load the compiler cannot prove aligned into
/// single bytes. The key/value cache is one.
pub(crate) struct Floats {
    lanes: Vec<Lanes>,
    len: usize,
}

impl Floats {
    /// `len` zeroes.
    pub(crate) fn zeroed(len: usize) -> Self {
        Self {
            lanes: vec![Lanes::default(); len.div_ceil(4)],
            len,
        }
    }
}

impl Deref for Floats {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        // SAFETY: `lanes` holds at least `len` contiguous `f32`s.
        unsafe { core::slice::from_raw_parts(self.lanes.as_ptr().cast(), self.len) }
    }
}

impl DerefMut for Floats {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as in `deref`, and borrowed mutably through `self`.
        unsafe { core::slice::from_raw_parts_mut(self.lanes.as_mut_ptr().cast(), self.len) }
    }
}

/// `out[s] = dot(q, key s) / scale` for every position `s` in
/// `0..out.len()`, key `s` being `q.len()` elements at `keys[s * stride..]`.
///
/// # Panics
///
/// If `keys` is too short to hold every position's key.
pub(crate) fn scores(q: &[f32], keys: &[f32], stride: usize, scale: f32, out: &mut [f32]) {
    check(keys, stride, q.len(), out.len());
    #[allow(unused_mut, reason = "only the NEON version does part of the work")]
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    if neon::fits(keys, stride, q.len()) {
        // SAFETY: checked in bounds above, and aligned by `fits`.
        done = unsafe { neon::scores(q, keys, stride, scale, out) };
    }
    for (s, score) in out.iter_mut().enumerate().skip(done) {
        *score = dot(q, &keys[s * stride..s * stride + q.len()]) / scale;
    }
}

/// `out = sum over s of att[s] * value s`, added up in order of `s` from
/// zero, value `s` being `out.len()` elements at `values[s * stride..]`.
///
/// # Panics
///
/// If `values` is too short to hold every position's value.
pub(crate) fn weighted_sum(att: &[f32], values: &[f32], stride: usize, out: &mut [f32]) {
    check(values, stride, out.len(), att.len());
    #[cfg(target_arch = "aarch64")]
    if neon::fits(values, stride, out.len()) {
        // SAFETY: checked in bounds above, and aligned by `fits`.
        unsafe { neon::weighted_sum(att, values, stride, out) };
        return;
    }
    out.fill(0.0);
    let width = out.len();
    for (s, &a) in att.iter().enumerate() {
        let at = s * stride;
        for (o, &v) in out.iter_mut().zip(&values[at..at + width]) {
            *o += a * v;
        }
    }
}

/// Panics unless `rows` holds `count` rows of `width` at `stride`.
fn check(rows: &[f32], stride: usize, width: usize, count: usize) {
    assert!(
        count == 0 || (count - 1) * stride + width <= rows.len(),
        "attention past the end of the cache"
    );
}

/// A left-to-right dot product.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        sum += a * b;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// Whether `rows` suits the NEON loops: each row starts 16-byte
    /// aligned, and is a whole number of 16-element (one cache line)
    /// pieces wide.
    pub(super) fn fits(rows: &[f32], stride: usize, width: usize) -> bool {
        (rows.as_ptr() as usize).is_multiple_of(16)
            && stride.is_multiple_of(4)
            && width.is_multiple_of(16)
    }

    /// Elements `4c..4c + 4` of the row at `row`, one aligned load.
    ///
    /// # Safety
    ///
    /// They must be in bounds, and `row` 16-byte aligned.
    #[inline(always)]
    unsafe fn load(row: *const f32, c: usize) -> float32x4_t {
        unsafe { *row.cast::<float32x4_t>().add(c) }
    }

    /// The 4x4 transpose: lane `j` of result `i` is lane `i` of `r[j]`.
    #[inline(always)]
    fn transpose(r: [float32x4_t; 4]) -> [float32x4_t; 4] {
        // SAFETY: register-only shuffles.
        unsafe {
            let t0 = vreinterpretq_f64_f32(vtrn1q_f32(r[0], r[1]));
            let t1 = vreinterpretq_f64_f32(vtrn2q_f32(r[0], r[1]));
            let t2 = vreinterpretq_f64_f32(vtrn1q_f32(r[2], r[3]));
            let t3 = vreinterpretq_f64_f32(vtrn2q_f32(r[2], r[3]));
            [
                vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)),
                vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)),
                vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)),
                vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)),
            ]
        }
    }

    /// [`super::scores`] for the positions in whole fours; returns how
    /// many it did, the rest left to the scalar loop.
    ///
    /// # Safety
    ///
    /// `keys` must hold every position's key and suit [`fits`].
    pub(super) unsafe fn scores(
        q: &[f32],
        keys: &[f32],
        stride: usize,
        scale: f32,
        out: &mut [f32],
    ) -> usize {
        let mut s = 0;
        unsafe {
            // Eight positions at a time: two independent chains of adds
            // for the core to overlap, each one add's latency long per
            // element.
            while s + 8 <= out.len() {
                group::<2>(q, keys, stride, scale, &mut out[s..s + 8], s);
                s += 8;
            }
            while s + 4 <= out.len() {
                group::<1>(q, keys, stride, scale, &mut out[s..s + 4], s);
                s += 4;
            }
        }
        s
    }

    /// Scores for the `4 * G` positions from `first` into `out`.
    #[inline(always)]
    unsafe fn group<const G: usize>(
        q: &[f32],
        keys: &[f32],
        stride: usize,
        scale: f32,
        out: &mut [f32],
        first: usize,
    ) {
        // SAFETY: loads inside the caller's bounds, aligned by its promise;
        // stores of four `f32`s into four-`f32` pieces of `out`.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); G];
            let base = keys.as_ptr();
            for c in 0..q.len() / 4 {
                let qs = &q[4 * c..4 * c + 4];
                for (g, acc) in acc.iter_mut().enumerate() {
                    let s = first + 4 * g;
                    let cols = transpose(core::array::from_fn(|j| {
                        load(base.add((s + j) * stride), c)
                    }));
                    for (col, &qi) in cols.iter().zip(qs) {
                        *acc = vaddq_f32(*acc, vmulq_n_f32(*col, qi));
                    }
                }
            }
            let scale = vdupq_n_f32(scale);
            for (g, acc) in acc.iter().enumerate() {
                vst1q_f32(out[4 * g..4 * g + 4].as_mut_ptr(), vdivq_f32(*acc, scale));
            }
        }
    }

    /// [`super::weighted_sum`]: sixteen of the head's elements at a time,
    /// in four accumulators, through every position.
    ///
    /// # Safety
    ///
    /// `values` must hold every position's value and suit [`fits`].
    pub(super) unsafe fn weighted_sum(att: &[f32], values: &[f32], stride: usize, out: &mut [f32]) {
        let base = values.as_ptr();
        // SAFETY: loads inside the caller's bounds, aligned by its promise;
        // stores of four `f32`s into four-`f32` pieces of `out`.
        unsafe {
            for piece in 0..out.len() / 16 {
                let mut acc = [vdupq_n_f32(0.0); 4];
                for (s, &a) in att.iter().enumerate() {
                    let row = base.add(s * stride);
                    for (k, acc) in acc.iter_mut().enumerate() {
                        *acc = vaddq_f32(*acc, vmulq_n_f32(load(row, 4 * piece + k), a));
                    }
                }
                for (k, acc) in acc.iter().enumerate() {
                    let at = 16 * piece + 4 * k;
                    vst1q_f32(out[at..at + 4].as_mut_ptr(), *acc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Floats spread over several orders of magnitude, both signs: sums
    /// of them round differently in any other order.
    fn noise(len: usize, seed: u32) -> Floats {
        let mut f = Floats::zeroed(len);
        let mut state = seed;
        for v in f.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mantissa = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
            *v = mantissa * [0.01, 1.0, 100.0][(state % 3) as usize];
        }
        f
    }

    #[test]
    fn scores_and_sums_are_the_scalar_loops_bits() {
        let (head, stride) = (64, 320);
        for positions in [1, 3, 4, 7, 8, 13, 41, 100] {
            let cache = noise(positions * stride, 1);
            let q = noise(head, 2);
            let att = noise(positions, 3);
            for at in [0, 64, 256] {
                let rows = &cache[at..];
                let mut got = vec![0.0; positions];
                scores(&q, rows, stride, 8.0, &mut got);
                for (s, &g) in got.iter().enumerate() {
                    let want = dot(&q, &rows[s * stride..s * stride + head]) / 8.0;
                    assert_eq!(g.to_bits(), want.to_bits(), "score {s} of {positions}");
                }

                let mut got = vec![0.0; head];
                weighted_sum(&att, rows, stride, &mut got);
                let mut want = vec![0.0f32; head];
                for (s, &a) in att.iter().enumerate() {
                    for (w, &v) in want.iter_mut().zip(&rows[s * stride..]) {
                        *w += a * v;
                    }
                }
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "element {i} of {positions}");
                }
            }
        }
    }
}
