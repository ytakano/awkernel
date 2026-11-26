//! One decoding step of Qwen3: activations, KV cache and the forward pass.
//!
//! Per layer:
//!
//! ```text
//! xb = rmsnorm(x);  q|k|v = Wqkv xb;  q,k = rmsnorm per head;  rope(q, k)
//! cache[pos] = k, v;  att = softmax(q k^T / sqrt(hd)) v   (GQA)
//! x += Wo att;  xb = rmsnorm(x);  g|u = Wgu xb;  x += Wdown (silu(g) * u)
//! ```
//!
//! and finally `logits = Wembed^T rmsnorm(x)` (tied embeddings).
//!
//! Every matrix-vector product and the attention over heads go through the
//! [`Executor`]; everything else runs on the calling thread.

use crate::{
    cpu::Backend,
    exec::{Executor, SharedMut},
    kernels,
    model::{mul, Config, Model},
    q8::{self, row_stride},
    try_zeroed_vec, Error,
};
use alloc::vec::Vec;

/// Activations and KV cache for one sequence.
pub struct Transformer {
    cfg: Config,
    max_seq: usize,
    backend: Backend,

    /// Residual stream `[dim]`.
    x: Vec<f32>,
    /// Normalized input of a sub-block / output of a projection `[dim]`.
    xb: Vec<f32>,
    /// `q | k | v` of the current token `[q_dim + 2 * kv_dim]`.
    qkv: Vec<f32>,
    /// Attention output `[q_dim]`.
    att_out: Vec<f32>,
    /// Attention scores, one row per head `[n_heads, max_seq]`.
    att: Vec<f32>,
    /// `gate | up` `[2 * hidden_dim]`.
    hb: Vec<f32>,
    logits: Vec<f32>,
    /// Quantized activation (`max(dim, q_dim, hidden_dim)` elements).
    xq: Vec<i8>,
    xs: Vec<f32>,

    inv_freq: Vec<f64>,
    cos: Vec<f32>,
    sin: Vec<f32>,

    /// `[n_layers, n_kv_heads, max_seq, head_dim]`: each head's history is
    /// contiguous, so attention streams through memory.
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

impl Transformer {
    /// Allocates the state for contexts of up to `max_seq` tokens.
    ///
    /// The KV cache dominates: `2 * n_layers * kv_dim * max_seq * 4` bytes
    /// (224 KiB per token for Qwen3-0.6B, i.e. 448 MiB for 2048 tokens).
    pub fn new(cfg: &Config, max_seq: usize, backend: Backend) -> Result<Self, Error> {
        if max_seq == 0 || max_seq > cfg.max_seq_len {
            return Err(Error::BadContextLength);
        }
        let c = cfg;
        let half = c.head_dim / 2;
        let kv_len = mul(mul(c.n_layers, c.kv_dim())?, max_seq)?;
        let max_k = c.dim.max(c.q_dim()).max(c.hidden_dim);

        let mut inv_freq = try_zeroed_vec::<f64>(half)?;
        // inv_freq[i] = theta^(-2i / head_dim), in f64 as in llm.asm.
        let ln_theta = libm::log(c.rope_theta as f64);
        for (i, f) in inv_freq.iter_mut().enumerate() {
            *f = libm::exp(-ln_theta * (2 * i) as f64 / c.head_dim as f64);
        }

        Ok(Self {
            cfg: *c,
            max_seq,
            backend,
            x: try_zeroed_vec(c.dim)?,
            xb: try_zeroed_vec(c.dim)?,
            qkv: try_zeroed_vec(c.q_dim() + 2 * c.kv_dim())?,
            att_out: try_zeroed_vec(c.q_dim())?,
            att: try_zeroed_vec(c.n_heads * max_seq)?,
            hb: try_zeroed_vec(2 * c.hidden_dim)?,
            logits: try_zeroed_vec(c.vocab_size)?,
            xq: try_zeroed_vec(max_k)?,
            xs: try_zeroed_vec(max_k / q8::BLOCK)?,
            inv_freq,
            cos: try_zeroed_vec(half)?,
            sin: try_zeroed_vec(half)?,
            k_cache: try_zeroed_vec(kv_len)?,
            v_cache: try_zeroed_vec(kv_len)?,
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Logits of the last step run with `want_logits`.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Heap bytes held by this state.
    pub fn memory_bytes(&self) -> usize {
        4 * (self.x.len()
            + self.xb.len()
            + self.qkv.len()
            + self.att_out.len()
            + self.att.len()
            + self.hb.len()
            + self.logits.len()
            + self.xs.len()
            + self.cos.len()
            + self.sin.len()
            + self.k_cache.len()
            + self.v_cache.len())
            + self.xq.len()
            + 8 * self.inv_freq.len()
    }

    /// Runs `token` at position `pos`, appending its keys/values to the cache.
    /// Positions `0..pos` must have been run before (in order). The logits are
    /// computed only if `want_logits` (the LM head is ~25% of the work).
    pub fn forward(
        &mut self,
        model: &Model<'_>,
        token: u32,
        pos: usize,
        want_logits: bool,
        exec: &dyn Executor,
    ) -> Result<(), Error> {
        let c = self.cfg;
        if token as usize >= c.vocab_size {
            return Err(Error::BadToken(token));
        }
        if pos >= self.max_seq {
            return Err(Error::BadContextLength);
        }
        let be = self.backend;
        let (dim, hd, hidden) = (c.dim, c.head_dim, c.hidden_dim);
        let (q_dim, kv_dim) = (c.q_dim(), c.kv_dim());
        let eps = c.rms_eps;

        // Embedding lookup.
        let stride = row_stride(dim);
        let row = &model.embed[token as usize * stride..][..stride];
        q8::dequant_row(row, dim, &mut self.x);

        // RoPE angles for this position, shared by all layers.
        for i in 0..hd / 2 {
            let a = pos as f64 * self.inv_freq[i];
            self.cos[i] = libm::cos(a) as f32;
            self.sin[i] = libm::sin(a) as f32;
        }

        for (l, lw) in model.layers.iter().enumerate() {
            let norms = model.layer_norms(l);

            // Attention block.
            kernels::rmsnorm(&mut self.xb, &self.x, norms.input, eps);
            matvec(
                be,
                exec,
                lw.qkv,
                dim,
                &self.xb,
                &mut self.xq,
                &mut self.xs,
                &mut self.qkv,
            );

            let (q, kv) = self.qkv.split_at_mut(q_dim);
            let (k, v) = kv.split_at_mut(kv_dim);
            for h in q.chunks_exact_mut(hd) {
                kernels::rmsnorm_inplace(h, norms.q, eps);
            }
            for h in k.chunks_exact_mut(hd) {
                kernels::rmsnorm_inplace(h, norms.k, eps);
            }
            kernels::rope(q, hd, &self.cos, &self.sin);
            kernels::rope(k, hd, &self.cos, &self.sin);

            let layer_base = l * c.n_kv_heads * self.max_seq * hd;
            for h in 0..c.n_kv_heads {
                let o = layer_base + (h * self.max_seq + pos) * hd;
                self.k_cache[o..o + hd].copy_from_slice(&k[h * hd..][..hd]);
                self.v_cache[o..o + hd].copy_from_slice(&v[h * hd..][..hd]);
            }

            self.attention(exec, layer_base, pos);

            matvec(
                be,
                exec,
                lw.o,
                q_dim,
                &self.att_out,
                &mut self.xq,
                &mut self.xs,
                &mut self.xb,
            );
            kernels::add_assign(&mut self.x, &self.xb);

            // MLP block.
            kernels::rmsnorm(&mut self.xb, &self.x, norms.post, eps);
            matvec(
                be,
                exec,
                lw.gate_up,
                dim,
                &self.xb,
                &mut self.xq,
                &mut self.xs,
                &mut self.hb,
            );
            let (gate, up) = self.hb.split_at_mut(hidden);
            kernels::silu_mul(be, gate, up);
            matvec(
                be,
                exec,
                lw.down,
                hidden,
                gate,
                &mut self.xq,
                &mut self.xs,
                &mut self.xb,
            );
            kernels::add_assign(&mut self.x, &self.xb);
        }

        if want_logits {
            kernels::rmsnorm_inplace(&mut self.x, model.final_norm(), eps);
            matvec(
                be,
                exec,
                model.embed,
                dim,
                &self.x,
                &mut self.xq,
                &mut self.xs,
                &mut self.logits,
            );
        }
        Ok(())
    }

    /// Grouped-query attention for the token at `pos`, in parallel over KV
    /// heads. The query heads sharing a KV head are handled together, so each
    /// cached key/value row is read once per layer:
    /// O(n_heads * (pos + 1) * head_dim) flops and
    /// O(n_kv_heads * (pos + 1) * head_dim) floats of KV traffic.
    fn attention(&mut self, exec: &dyn Executor, layer_base: usize, pos: usize) {
        let c = self.cfg;
        let hd = c.head_dim;
        let group = c.n_heads / c.n_kv_heads;
        let max_seq = self.max_seq;
        let scale = 1.0 / libm::sqrtf(hd as f32);
        let be = self.backend;

        let q = &self.qkv[..c.q_dim()];
        let k_cache = &self.k_cache;
        let v_cache = &self.v_cache;
        let att = SharedMut::new(&mut self.att);
        let out = SharedMut::new(&mut self.att_out);

        exec.par_for(c.n_kv_heads, &|kv_heads| {
            for kvh in kv_heads {
                let base = layer_base + kvh * max_seq * hd;
                let h0 = kvh * group;
                // SAFETY: the query heads of KV head `kvh` belong to exactly
                // one worker, and their score rows / outputs are contiguous.
                let scores = unsafe { att.range(h0 * max_seq..(h0 + group) * max_seq) };
                let outs = unsafe { out.range(h0 * hd..(h0 + group) * hd) };

                for t in 0..=pos {
                    let kt = &k_cache[base + t * hd..][..hd];
                    for g in 0..group {
                        let qh = &q[(h0 + g) * hd..][..hd];
                        scores[g * max_seq + t] = kernels::dot(be, qh, kt) * scale;
                    }
                }
                for g in 0..group {
                    kernels::softmax(be, &mut scores[g * max_seq..g * max_seq + pos + 1]);
                }
                outs.fill(0.0);
                for t in 0..=pos {
                    let vt = &v_cache[base + t * hd..][..hd];
                    for g in 0..group {
                        let p = scores[g * max_seq + t];
                        kernels::axpy(be, &mut outs[g * hd..(g + 1) * hd], p, vt);
                    }
                }
            }
        });
    }
}

/// `out = W x` for a Q8_0 matrix `W` with `k` columns and `out.len()` rows.
/// `x` is quantized once on the calling thread, then rows are split over the
/// executor.
#[allow(clippy::too_many_arguments)]
fn matvec(
    be: Backend,
    exec: &dyn Executor,
    w: &[u8],
    k: usize,
    x: &[f32],
    xq: &mut [i8],
    xs: &mut [f32],
    out: &mut [f32],
) {
    q8::quantize(be, &x[..k], xq, xs);
    let (xq, xs) = (&xq[..k], &xs[..k / q8::BLOCK]);
    let out_s = SharedMut::new(out);
    exec.par_for(out_s.len(), &|rows| {
        let row0 = rows.start;
        // SAFETY: the executor hands out disjoint row ranges.
        let o = unsafe { out_s.range(rows) };
        q8::matvec(be, w, k, xq, xs, o, row0);
    });
}
