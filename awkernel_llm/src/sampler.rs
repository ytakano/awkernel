//! Next-token selection: greedy, or temperature + nucleus (top-p) sampling.

use crate::{kernels, try_zeroed_vec, Backend, Error};
use alloc::vec::Vec;

pub struct Sampler {
    backend: Backend,
    temperature: f32,
    top_p: f32,
    rng: u64,
    /// Scratch `[vocab]`, allocated only when sampling is enabled.
    probs: Vec<f32>,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    /// `temperature <= 0` selects greedy decoding (argmax).
    pub fn new(
        backend: Backend,
        vocab: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<Self, Error> {
        if !temperature.is_finite() || !top_p.is_finite() {
            return Err(Error::BadConfig("temperature/top_p must be finite"));
        }
        let sampling = temperature > 0.0;
        let mut cand = Vec::new();
        if sampling {
            cand.try_reserve_exact(vocab)
                .map_err(|_| Error::OutOfMemory)?;
        }
        Ok(Self {
            backend,
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            // xorshift must not start at 0.
            rng: seed ^ 0x2545_F491_4F6C_DD1D,
            probs: if sampling {
                try_zeroed_vec(vocab)?
            } else {
                Vec::new()
            },
            cand,
        })
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.sample_except(logits, None)
    }

    /// Like [`Sampler::sample`], but never returns `banned`.
    pub fn sample_except(&mut self, logits: &[f32], banned: Option<u32>) -> u32 {
        if self.is_greedy() {
            return argmax_except(logits, banned);
        }

        let r = self.next_f32();
        let probs = &mut self.probs[..logits.len()];
        let inv_t = 1.0 / self.temperature;
        for (p, &l) in probs.iter_mut().zip(logits) {
            *p = l * inv_t;
        }
        if let Some(b) = banned {
            probs[b as usize] = f32::NEG_INFINITY;
        }
        kernels::softmax(self.backend, probs);

        if self.top_p <= 0.0 || self.top_p >= 1.0 {
            return sample_cdf(probs, r);
        }

        // Nucleus sampling. Tokens with p < (1 - top_p) / (n - 1) can never be
        // in the nucleus, so only the survivors are sorted: O(V + m log m)
        // with m << V in practice.
        let cutoff = (1.0 - self.top_p) / (probs.len() - 1).max(1) as f32;
        self.cand.clear();
        self.cand.extend(
            probs
                .iter()
                .enumerate()
                .filter(|(_, &p)| p >= cutoff)
                .map(|(i, &p)| (p, i as u32)),
        );
        if self.cand.is_empty() {
            // Only possible with NaN/inf logits or top_p < 1/V.
            return argmax_except(logits, banned);
        }
        self.cand
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

        let mut cum = 0.0f32;
        let mut last = self.cand.len().saturating_sub(1);
        for (i, &(p, _)) in self.cand.iter().enumerate() {
            cum += p;
            if cum > self.top_p {
                last = i;
                break;
            }
        }
        let target = r * cum;
        let mut acc = 0.0f32;
        for &(p, id) in &self.cand[..=last] {
            acc += p;
            if target < acc {
                return id;
            }
        }
        self.cand.get(last).map_or(0, |c| c.1)
    }

    /// Uniform in `[0, 1)` (xorshift64*).
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax_except(logits: &[f32], banned: Option<u32>) -> u32 {
    let best = kernels::argmax(logits) as u32;
    match banned {
        Some(b) if b == best => {
            let mut best = usize::from(b == 0);
            for (i, &v) in logits.iter().enumerate() {
                if i != b as usize && v > logits[best] {
                    best = i;
                }
            }
            best as u32
        }
        _ => best,
    }
}

fn sample_cdf(probs: &[f32], r: f32) -> u32 {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_first_max() {
        let mut s = Sampler::new(Backend::SCALAR, 4, 0.0, 0.9, 1).unwrap();
        assert_eq!(s.sample(&[0.0, 3.0, 3.0, 1.0]), 1);
    }

    #[test]
    fn banned_token_is_skipped() {
        let mut s = Sampler::new(Backend::SCALAR, 4, 0.0, 0.9, 1).unwrap();
        assert_eq!(s.sample_except(&[0.0, 3.0, 2.0, 1.0], Some(1)), 2);
        assert_eq!(s.sample_except(&[5.0, 3.0], Some(0)), 1);
        let mut s = Sampler::new(Backend::SCALAR, 3, 1.0, 1.0, 1).unwrap();
        for _ in 0..50 {
            assert_ne!(s.sample_except(&[9.0, 0.0, 0.0], Some(0)), 0);
        }
    }

    #[test]
    fn top_p_stays_in_nucleus() {
        let mut s = Sampler::new(Backend::detect(), 5, 1.0, 0.5, 7).unwrap();
        let logits = [10.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..100 {
            assert_eq!(s.sample(&logits), 0);
        }
    }
}
