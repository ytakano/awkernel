//! Element-wise and reduction kernels on f32 vectors.
//!
//! Except for attention (`dot`/`axpy`/`softmax`, which grow with the context
//! length) these are O(dim) per call and a small part of the runtime compared
//! with the Q8_0 products in [`crate::q8`].

use crate::cpu::{Backend, Impl};

/// `out = x / rms(x) * w`, `rms(x) = sqrt(mean(x^2) + eps)`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let inv = inv_rms(x, eps);
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

/// In-place [`rmsnorm`].
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let inv = inv_rms(x, eps);
    for (v, &g) in x.iter_mut().zip(w) {
        *v = *v * inv * g;
    }
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    1.0 / libm::sqrtf(ss / x.len() as f32 + eps)
}

/// In-place softmax.
pub fn softmax(backend: Backend, x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum = match backend.imp() {
        // SAFETY: an AVX2 `Backend` is only created after CPU detection.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::exp_sub_sum(x, max) },
        _ => {
            let mut sum = 0.0f32;
            for v in x.iter_mut() {
                *v = libm::expf(*v - max);
                sum += *v;
            }
            sum
        }
    };
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `gate[i] = silu(gate[i]) * up[i]`, `silu(x) = x / (1 + exp(-x))`.
pub fn silu_mul(backend: Backend, gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    match backend.imp() {
        // SAFETY: see `softmax`.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::silu_mul(gate, up) },
        _ => {
            for (g, &u) in gate.iter_mut().zip(up) {
                *g = silu(*g) * u;
            }
        }
    }
}

/// `x / (1 + exp(-x))`, written so that `exp` never overflows.
#[inline]
fn silu(x: f32) -> f32 {
    let e = libm::expf(-x.abs());
    let num = if x < 0.0 { e } else { 1.0 };
    x * num / (1.0 + e)
}

/// `dst += src`.
pub fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Split-half (NeoX) rotary embedding of consecutive heads.
pub fn rope(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in x.chunks_exact_mut(head_dim) {
        let (a, b) = h.split_at_mut(half);
        for i in 0..half {
            let (x1, x2) = (a[i], b[i]);
            a[i] = x1 * cos[i] - x2 * sin[i];
            b[i] = x2 * cos[i] + x1 * sin[i];
        }
    }
}

/// Dot product.
#[inline]
pub fn dot(backend: Backend, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    match backend.imp() {
        // SAFETY: see `softmax`.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::dot(a, b) },
        _ => a.iter().zip(b).map(|(x, y)| x * y).sum(),
    }
}

/// `dst += alpha * src`.
#[inline]
pub fn axpy(backend: Backend, dst: &mut [f32], alpha: f32, src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    match backend.imp() {
        // SAFETY: see `softmax`.
        #[cfg(target_arch = "x86_64")]
        Impl::Avx2 => unsafe { avx2::axpy(dst, alpha, src) },
        _ => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += alpha * s;
            }
        }
    }
}

/// Index of the maximum (first one on ties).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    /// 8-wide `exp`: range reduction `x = n ln2 + r`, `|r| <= ln2/2`, and a
    /// degree-6 Taylor polynomial (relative error ~1e-7), as llm.asm
    /// `fast_expf_avx2`. Inputs are clamped to `[-87.34, 88.38]`.
    #[inline(always)]
    unsafe fn exp8(x: __m256) -> __m256 {
        let x = _mm256_min_ps(x, _mm256_set1_ps(88.376_26));
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.336_54));
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_ps(x, _mm256_set1_ps(core::f32::consts::LOG2_E)),
        );
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);

        let mut p = _mm256_set1_ps(1.0 / 720.0);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

        // 2^n for n in [-126, 127].
        let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvtps_epi32(n),
            _mm256_set1_epi32(127),
        ));
        _mm256_mul_ps(p, _mm256_castsi256_ps(e))
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    /// `x[i] = exp(x[i] - max)`, returns the sum.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_sub_sum(x: &mut [f32], max: f32) -> f32 {
        let m = _mm256_set1_ps(max);
        let mut acc = _mm256_setzero_ps();
        let (chunks, rest) = x.as_chunks_mut::<8>();
        for c in chunks {
            let v = exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), m));
            _mm256_storeu_ps(c.as_mut_ptr(), v);
            acc = _mm256_add_ps(acc, v);
        }
        let mut sum = hsum(acc);
        for v in rest {
            *v = libm::expf(*v - max);
            sum += *v;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn silu_mul(gate: &mut [f32], up: &[f32]) {
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let one = _mm256_set1_ps(1.0);
        let n8 = gate.len() / 8 * 8;
        for i in (0..n8).step_by(8) {
            let x = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            // e = exp(-|x|) in (0, 1]; silu = x * (x < 0 ? e : 1) / (1 + e).
            let e = exp8(_mm256_sub_ps(
                _mm256_setzero_ps(),
                _mm256_and_ps(x, abs_mask),
            ));
            let num = _mm256_blendv_ps(one, e, x);
            let s = _mm256_div_ps(_mm256_mul_ps(x, num), _mm256_add_ps(one, e));
            _mm256_storeu_ps(gate.as_mut_ptr().add(i), _mm256_mul_ps(s, u));
        }
        for i in n8..gate.len() {
            gate[i] = super::silu(gate[i]) * up[i];
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n16 = a.len() / 16 * 16;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for i in (0..n16).step_by(16) {
            let (pa, pb) = (a.as_ptr().add(i), b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)), acc1);
        }
        let mut sum = hsum(_mm256_add_ps(acc0, acc1));
        for i in n16..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy(dst: &mut [f32], alpha: f32, src: &[f32]) {
        let n8 = dst.len() / 8 * 8;
        let a = _mm256_set1_ps(alpha);
        for i in (0..n8).step_by(8) {
            let d = dst.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                d,
                _mm256_fmadd_ps(a, _mm256_loadu_ps(src.as_ptr().add(i)), _mm256_loadu_ps(d)),
            );
        }
        for i in n8..dst.len() {
            dst[i] += alpha * src[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, rel: f32) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-30)
    }

    #[test]
    fn softmax_sums_to_one() {
        for be in [Backend::SCALAR, Backend::detect()] {
            let mut x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
            x.push(-1000.0);
            softmax(be, &mut x);
            assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(x[37] < 1e-30);
        }
    }

    #[test]
    fn silu_matches_definition() {
        for be in [Backend::SCALAR, Backend::detect()] {
            let xs: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 3.3).collect();
            let mut g = xs.clone();
            silu_mul(be, &mut g, &vec![2.0; xs.len()]);
            for (&v, &got) in xs.iter().zip(&g) {
                let want = v / (1.0 + (-v as f64).exp() as f32) * 2.0;
                assert!(
                    close(got, want, 1e-6) || (got - want).abs() < 1e-7,
                    "{v}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn exp_accuracy() {
        let be = Backend::detect();
        let mut x: Vec<f32> = (0..2001).map(|i| -80.0 + i as f32 * 0.08).collect();
        let want: Vec<f32> = x.iter().map(|&v| (v as f64).exp() as f32).collect();
        softmax(be, &mut x);
        // Undo the normalization by comparing ratios to the last element.
        let last = *x.last().unwrap();
        let wl = *want.last().unwrap();
        for (g, w) in x.iter().zip(&want) {
            assert!(close(g / last, w / wl, 2e-6), "{g} vs {w}");
        }
    }

    #[test]
    fn dot_axpy_agree() {
        let be = Backend::detect();
        let a: Vec<f32> = (0..133).map(|i| (i as f32 * 0.7).cos()).collect();
        let b: Vec<f32> = (0..133).map(|i| (i as f32 * 0.3).sin()).collect();
        assert!(close(dot(be, &a, &b), dot(Backend::SCALAR, &a, &b), 1e-5));
        let mut d0 = b.clone();
        let mut d1 = b.clone();
        axpy(Backend::SCALAR, &mut d0, 0.25, &a);
        axpy(be, &mut d1, 0.25, &a);
        for (x, y) in d0.iter().zip(&d1) {
            assert!(close(*x, *y, 1e-6));
        }
    }

    #[test]
    fn rope_rotates_pairs() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        let a = 0.5f32;
        rope(&mut x, 4, &[a.cos(), 1.0], &[a.sin(), 0.0]);
        assert!((x[0] - a.cos()).abs() < 1e-6 && (x[2] - a.sin()).abs() < 1e-6);
        assert_eq!((x[1], x[3]), (0.0, 1.0));
    }
}
