//! Q8_0 matrix-vector products.
//!
//! A Q8_0 row of `K` elements (`K % 32 == 0`) is stored as
//!
//! ```text
//! [K x i8 weights in -127..=127][K/32 x f16 scales]
//! ```
//!
//! so its size is `K + K/16` bytes. Element `j` is `q[j] * scale[j / 32]`.
//!
//! As in llm.asm `FAST_Q8=1`, the activation vector is quantized to Q8_0 too
//! (one f32 scale per 32 elements), so each 32-element block is an exact
//! integer dot product followed by one multiply by `w_scale * x_scale`.

use crate::cpu::{Backend, Impl};

pub const BLOCK: usize = 32;
const INV_127: f32 = 1.0 / 127.0;

/// Bytes per row of a Q8_0 matrix with `k` columns.
#[inline]
pub const fn row_stride(k: usize) -> usize {
    k + k / 16
}

/// Converts an IEEE 754 binary16 value to f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalize the mantissa.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn scale(row: &[u8], k: usize, b: usize) -> f32 {
    let o = k + 2 * b;
    f16_to_f32(u16::from_le_bytes([row[o], row[o + 1]]))
}

/// Dequantizes one row (`row.len() == row_stride(k)`) into `out[..k]`.
pub fn dequant_row(row: &[u8], k: usize, out: &mut [f32]) {
    assert!(row.len() >= row_stride(k) && out.len() >= k && k.is_multiple_of(BLOCK));
    for b in 0..k / BLOCK {
        let s = scale(row, k, b);
        for j in 0..BLOCK {
            out[b * BLOCK + j] = (row[b * BLOCK + j] as i8) as f32 * s;
        }
    }
}

/// Quantizes `x` into `xq` (int8) and `xs` (one f32 scale per 32 elements).
///
/// `scale = max|x| / 127`, `q = round_half_even(x * 127 / max|x|)`, matching
/// llm.asm `quantize_x_q8_0`.
pub fn quantize(backend: Backend, x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
    let k = x.len();
    assert!(k.is_multiple_of(BLOCK) && xq.len() >= k && xs.len() >= k / BLOCK);
    match backend.imp() {
        // SAFETY: an AVX2 `Backend` is only created after CPU detection.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::quantize(x, xq, xs) },
        _ => quantize_scalar(x, xq, xs),
    }
}

pub fn quantize_scalar(x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
    for (b, blk) in x.as_chunks::<BLOCK>().0.iter().enumerate() {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        xs[b] = amax * INV_127;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (j, v) in blk.iter().enumerate() {
            let q = libm::rintf(v * inv) as i32;
            xq[b * BLOCK + j] = q.clamp(-127, 127) as i8;
        }
    }
}

/// Computes `out[i] = dot(W[row0 + i], x)` for `i in 0..out.len()`, where `W`
/// is a Q8_0 matrix with `k` columns and `x` is given as quantized blocks.
pub fn matvec(
    backend: Backend,
    w: &[u8],
    k: usize,
    xq: &[i8],
    xs: &[f32],
    out: &mut [f32],
    row0: usize,
) {
    let stride = row_stride(k);
    assert!(k.is_multiple_of(BLOCK) && xq.len() >= k && xs.len() >= k / BLOCK);
    assert!((row0 + out.len()) * stride <= w.len());
    match backend.imp() {
        // SAFETY: an AVX2 `Backend` is only created after CPU detection; the
        // bounds were checked above.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::matvec(w, k, xq, xs, out, row0) },
        _ => matvec_scalar(w, k, xq, xs, out, row0),
    }
}

pub fn matvec_scalar(w: &[u8], k: usize, xq: &[i8], xs: &[f32], out: &mut [f32], row0: usize) {
    let stride = row_stride(k);
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[(row0 + i) * stride..][..stride];
        let mut sum = 0.0f32;
        for b in 0..k / BLOCK {
            let mut isum = 0i32;
            for j in 0..BLOCK {
                isum += (row[b * BLOCK + j] as i8) as i32 * xq[b * BLOCK + j] as i32;
            }
            sum += isum as f32 * (scale(row, k, b) * xs[b]);
        }
        *o = sum;
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{row_stride, BLOCK, INV_127};
    use core::arch::x86_64::*;

    /// AVX2 version of [`super::quantize_scalar`]. `vcvtps2dq` rounds half to
    /// even under the default MXCSR, as `rintf` does.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2; `x.len() % 32 == 0`, `xq.len() >= x.len()`,
    /// `xs.len() >= x.len() / 32`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn quantize(x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        // `packs` interleaves the 128-bit lanes; this restores element order.
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let min = _mm256_set1_epi8(-127);

        for b in 0..x.len() / BLOCK {
            let p = x.as_ptr().add(b * BLOCK);
            let v0 = _mm256_loadu_ps(p);
            let v1 = _mm256_loadu_ps(p.add(8));
            let v2 = _mm256_loadu_ps(p.add(16));
            let v3 = _mm256_loadu_ps(p.add(24));

            let m = _mm256_max_ps(
                _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
                _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask)),
            );
            let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
            let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            let m = _mm_max_ss(m, _mm_shuffle_ps::<1>(m, m));
            let amax = _mm_cvtss_f32(m);

            *xs.get_unchecked_mut(b) = amax * INV_127;
            let inv = _mm256_set1_ps(if amax > 0.0 { 127.0 / amax } else { 0.0 });

            let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, inv));
            let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, inv));
            let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, inv));
            let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, inv));
            let q = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
            let q = _mm256_max_epi8(_mm256_permutevar8x32_epi32(q, perm), min);
            _mm256_storeu_si256(xq.as_mut_ptr().add(b * BLOCK) as *mut __m256i, q);
        }
    }

    /// One 32-element block: `int_dot(w, x)` as 8 partial f32 sums.
    #[inline(always)]
    unsafe fn block_dot(w: *const u8, x: *const i8, ones: __m256i) -> __m256 {
        let xv = _mm256_loadu_si256(x as *const __m256i);
        let wv = _mm256_loadu_si256(w as *const __m256i);
        // maddubs needs one unsigned operand: |x| * (w * sign(x)) = x * w.
        // |x|, |w| <= 127, so the pairwise i16 sums (<= 32258) never saturate.
        let ax = _mm256_sign_epi8(xv, xv);
        let sw = _mm256_sign_epi8(wv, xv);
        let p16 = _mm256_maddubs_epi16(ax, sw);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones))
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    /// AVX2 version of [`super::matvec_scalar`], following llm.asm
    /// `matmul_q8_0_range` (FAST_Q8=1): rows are processed one at a time, four
    /// blocks (128 bytes) per iteration with four independent accumulators,
    /// and the next 128 bytes after the current row's weights are
    /// prefetched, which streams through the scales into the next row.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2/FMA/F16C and the caller must have checked
    /// the bounds asserted in [`super::matvec`].
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn matvec(
        w: &[u8],
        k: usize,
        xq: &[i8],
        xs: &[f32],
        out: &mut [f32],
        row0: usize,
    ) {
        let stride = row_stride(k);
        let nb = k / BLOCK;
        let nb4 = nb & !3;
        let ones = _mm256_set1_epi16(1);
        let xq = xq.as_ptr();
        let xs = xs.as_ptr();

        for (i, o) in out.iter_mut().enumerate() {
            let row = w.as_ptr().add((row0 + i) * stride);
            let scales = row.add(k);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut b = 0;
            while b < nb4 {
                let off = b * BLOCK;
                // Prefetch may point past the matrix; it never faults.
                _mm_prefetch::<_MM_HINT_T0>(scales.wrapping_add(off) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(scales.wrapping_add(off + 64) as *const i8);

                // Combined scales w_s * x_s of the 4 blocks, in both lanes.
                let sw = _mm_cvtph_ps(_mm_loadl_epi64(scales.add(2 * b) as *const __m128i));
                let s = _mm_mul_ps(sw, _mm_loadu_ps(xs.add(b)));
                let s = _mm256_set_m128(s, s);

                let d0 = block_dot(row.add(off), xq.add(off), ones);
                let d1 = block_dot(row.add(off + 32), xq.add(off + 32), ones);
                let d2 = block_dot(row.add(off + 64), xq.add(off + 64), ones);
                let d3 = block_dot(row.add(off + 96), xq.add(off + 96), ones);
                acc0 = _mm256_fmadd_ps(d0, _mm256_permute_ps::<0x00>(s), acc0);
                acc1 = _mm256_fmadd_ps(d1, _mm256_permute_ps::<0x55>(s), acc1);
                acc2 = _mm256_fmadd_ps(d2, _mm256_permute_ps::<0xAA>(s), acc2);
                acc3 = _mm256_fmadd_ps(d3, _mm256_permute_ps::<0xFF>(s), acc3);
                b += 4;
            }
            while b < nb {
                let off = b * BLOCK;
                let h = u16::from_le_bytes([*scales.add(2 * b), *scales.add(2 * b + 1)]);
                let sw = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)));
                let s = _mm256_set1_ps(sw * *xs.add(b));
                acc0 = _mm256_fmadd_ps(block_dot(row.add(off), xq.add(off), ones), s, acc0);
                b += 1;
            }

            *o = hsum(_mm256_add_ps(
                _mm256_add_ps(acc0, acc1),
                _mm256_add_ps(acc2, acc3),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    fn random(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Builds a Q8_0 matrix from f32 values (like export_weights.py).
    fn make_matrix(rows: usize, k: usize, seed: u32) -> Vec<u8> {
        let vals = random(rows * k, seed);
        let mut w = Vec::new();
        for r in 0..rows {
            let mut q = vec![0i8; k];
            let mut s = vec![0f32; k / 32];
            quantize_scalar(&vals[r * k..(r + 1) * k], &mut q, &mut s);
            w.extend(q.iter().map(|&v| v as u8));
            for sc in s {
                // f32 -> f16 for normal values in range (truncating is fine here).
                let b = sc.to_bits();
                let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
                let h =
                    ((b >> 16) & 0x8000) as u16 | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16;
                w.extend(h.to_le_bytes());
            }
        }
        w
    }

    #[test]
    fn backends_agree() {
        let be = Backend::detect();
        if !be.is_avx2() {
            eprintln!("skipped: no AVX2");
            return;
        }
        // k = 1024 uses only 4-block steps, 1056 also the 1-block tail.
        for (k, rows) in [(1024, 37), (1056, 20), (32, 5)] {
            let w = make_matrix(rows, k, k as u32);
            let x = random(k, 99);
            let (mut q0, mut s0) = (vec![0i8; k], vec![0f32; k / 32]);
            let (mut q1, mut s1) = (vec![0i8; k], vec![0f32; k / 32]);
            quantize_scalar(&x, &mut q0, &mut s0);
            quantize(be, &x, &mut q1, &mut s1);
            assert_eq!(q0, q1);
            assert_eq!(s0, s1);

            let mut o0 = vec![0f32; rows - 3];
            let mut o1 = vec![0f32; rows - 3];
            matvec_scalar(&w, k, &q0, &s0, &mut o0, 3);
            matvec(be, &w, k, &q0, &s0, &mut o1, 3);
            for (a, b) in o0.iter().zip(&o1) {
                assert!(
                    (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                    "k={k}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn quantize_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.37).collect();
        let mut xq = vec![0i8; 64];
        let mut xs = vec![0f32; 2];
        quantize_scalar(&x, &mut xq, &mut xs);
        for (i, v) in x.iter().enumerate() {
            let r = xq[i] as f32 * xs[i / 32];
            assert!((r - v).abs() <= xs[i / 32] * 0.5 + 1e-6, "{i}: {r} vs {v}");
        }
        assert!(xq.iter().all(|&q| q >= -127));
    }
}
