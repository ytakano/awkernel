//! `.asmbin` v2 parser.
//!
//! Layout (all sections 64-byte aligned, little endian), as written by
//! llm.asm `scripts/export_weights.py --scale-dtype fp16`:
//!
//! ```text
//! [0, 128)            header (see `parse_header`)
//! off_tokenizer       tokenizer section (see `tokenizer`)
//! off_embed           embed_tokens [vocab, dim] Q8_0, tied with the LM head
//! off_layers + l*S    layer l: in_norm[dim] q_norm[hd] k_norm[hd] post_norm[dim] (f32),
//!                     then (each 64-byte aligned) q, k, v, o, gate, up, down (Q8_0)
//! off_final_norm      final norm [dim] f32
//! ```
//!
//! A Q8_0 matrix `[M, K]` stores each row as `K` int8 weights followed by
//! `K/32` FP16 block scales; see [`crate::q8`].
//!
//! The weight matrices are borrowed (zero-copy). The small f32 norm vectors are
//! copied so that they can be used as `&[f32]` regardless of the alignment of
//! the input buffer.

use crate::{q8::row_stride, tokenizer::Tokenizer, try_zeroed_vec, Error};
use alloc::vec::Vec;

pub const HEADER_SIZE: usize = 128;
pub const MAGIC: &[u8; 4] = b"ASML";
pub const VERSION_V2: u32 = 2;
pub const FLAG_TIED_EMB: u32 = 1;
pub const FLAG_QK_NORM: u32 = 2;
pub const FLAG_FP16_SCALES: u32 = 4;

/// Model hyper-parameters from the header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    /// Maximum context length the export was made for.
    pub max_seq_len: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// End-of-sequence tokens (`<|im_end|>`, `<|endoftext|>`).
    pub eos: [u32; 2],
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos.contains(&tok)
    }
}

/// Weights of one transformer layer.
#[derive(Debug, Clone, Copy)]
pub struct Layer<'a> {
    /// `q_proj`, `k_proj`, `v_proj` stored back to back: `[q_dim + 2*kv_dim, dim]`.
    pub qkv: &'a [u8],
    /// `o_proj`: `[dim, q_dim]`.
    pub o: &'a [u8],
    /// `gate_proj`, `up_proj` stored back to back: `[2*hidden_dim, dim]`.
    pub gate_up: &'a [u8],
    /// `down_proj`: `[dim, hidden_dim]`.
    pub down: &'a [u8],
}

/// A parsed model. Borrows the file contents.
pub struct Model<'a> {
    pub config: Config,
    pub tokenizer: Tokenizer<'a>,
    /// `embed_tokens` / LM head: `[vocab, dim]`.
    pub embed: &'a [u8],
    pub layers: Vec<Layer<'a>>,
    /// Per layer: `in_norm[dim] | q_norm[hd] | k_norm[hd] | post_norm[dim]`.
    norms: Vec<f32>,
    final_norm: Vec<f32>,
}

/// Norm weights of one layer.
pub struct LayerNorms<'n> {
    pub input: &'n [f32],
    pub q: &'n [f32],
    pub k: &'n [f32],
    pub post: &'n [f32],
}

impl<'a> Model<'a> {
    /// Parses and validates `data`. Only the header, the offsets and the small
    /// norm vectors are read; the weights are not touched.
    pub fn parse(data: &'a [u8]) -> Result<Self, Error> {
        let h = parse_header(data)?;
        let c = h.config;

        let tokenizer = Tokenizer::parse(
            data,
            h.off_tokenizer,
            c.vocab_size,
            h.ht_mask as usize,
            h.off_embed,
        )?;

        let dim_stride = row_stride(c.dim);
        let embed = section(data, h.off_embed, mul(c.vocab_size, dim_stride)?, "embed")?;

        let norms_per_layer = add(mul(2, c.dim)?, mul(2, c.head_dim)?)?;
        let mut norms = try_zeroed_vec::<f32>(mul(norms_per_layer, c.n_layers)?)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.n_layers)
            .map_err(|_| Error::OutOfMemory)?;

        for l in 0..c.n_layers {
            let base = add(h.off_layers, mul(l, h.layer_stride)?)?;
            let norm_bytes = norms_per_layer * 4;
            let src = section(data, base, norm_bytes, "layer norms")?;
            read_f32s(
                src,
                &mut norms[l * norms_per_layer..(l + 1) * norms_per_layer],
            );

            let mut cur = base + norm_bytes;
            let mut take =
                |rows: usize, k: usize, what: &'static str| -> Result<(usize, usize), Error> {
                    cur = align64(cur);
                    let start = cur;
                    let len = mul(rows, row_stride(k))?;
                    cur = add(cur, len)?;
                    // Every matrix must stay inside this layer's stride.
                    if cur > add(base, h.layer_stride)? {
                        return Err(Error::BadShape(what));
                    }
                    Ok((start, len))
                };

            let q = take(c.q_dim(), c.dim, "q_proj")?;
            let k = take(c.kv_dim(), c.dim, "k_proj")?;
            let v = take(c.kv_dim(), c.dim, "v_proj")?;
            let o = take(c.dim, c.q_dim(), "o_proj")?;
            let gate = take(c.hidden_dim, c.dim, "gate_proj")?;
            let up = take(c.hidden_dim, c.dim, "up_proj")?;
            let down = take(c.dim, c.hidden_dim, "down_proj")?;

            // The fused QKV and gate+up products need the matrices back to back.
            if q.0 + q.1 != k.0 || k.0 + k.1 != v.0 {
                return Err(Error::BadShape("q/k/v not contiguous"));
            }
            if gate.0 + gate.1 != up.0 {
                return Err(Error::BadShape("gate/up not contiguous"));
            }

            layers.push(Layer {
                qkv: section(data, q.0, q.1 + k.1 + v.1, "qkv")?,
                o: section(data, o.0, o.1, "o_proj")?,
                gate_up: section(data, gate.0, gate.1 + up.1, "gate_up")?,
                down: section(data, down.0, down.1, "down_proj")?,
            });
        }

        let mut final_norm = try_zeroed_vec::<f32>(c.dim)?;
        read_f32s(
            section(data, h.off_final_norm, c.dim * 4, "final norm")?,
            &mut final_norm,
        );

        Ok(Self {
            config: c,
            tokenizer,
            embed,
            layers,
            norms,
            final_norm,
        })
    }

    pub fn layer_norms(&self, l: usize) -> LayerNorms<'_> {
        let c = &self.config;
        let per = 2 * c.dim + 2 * c.head_dim;
        let n = &self.norms[l * per..(l + 1) * per];
        let (input, rest) = n.split_at(c.dim);
        let (q, rest) = rest.split_at(c.head_dim);
        let (k, post) = rest.split_at(c.head_dim);
        LayerNorms { input, q, k, post }
    }

    pub fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }
}

struct Header {
    config: Config,
    ht_mask: u32,
    off_tokenizer: usize,
    off_embed: usize,
    off_layers: usize,
    layer_stride: usize,
    off_final_norm: usize,
}

fn parse_header(data: &[u8]) -> Result<Header, Error> {
    if data.len() < HEADER_SIZE {
        return Err(Error::Truncated("header"));
    }
    if &data[0..4] != MAGIC {
        return Err(Error::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let f32_at = |o: usize| f32::from_bits(u32_at(o));
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
    let usize_at = |o: usize| usize::try_from(u64_at(o)).map_err(|_| Error::Truncated("offset"));

    let version = u32_at(4);
    if version != VERSION_V2 {
        return Err(Error::UnsupportedVersion(version));
    }
    let flags = u32_at(60);
    let required = FLAG_TIED_EMB | FLAG_QK_NORM | FLAG_FP16_SCALES;
    if flags & required != required {
        return Err(Error::UnsupportedFlags(flags));
    }

    let config = Config {
        dim: u32_at(8) as usize,
        hidden_dim: u32_at(12) as usize,
        n_layers: u32_at(16) as usize,
        n_heads: u32_at(20) as usize,
        n_kv_heads: u32_at(24) as usize,
        head_dim: u32_at(28) as usize,
        vocab_size: u32_at(32) as usize,
        max_seq_len: u32_at(36) as usize,
        rms_eps: f32_at(40),
        rope_theta: f32_at(44),
        eos: [u32_at(48), u32_at(52)],
    };
    let c = &config;

    // Q8_0 rows are made of 32-element blocks; the kernels also assume this.
    for (v, what) in [
        (c.dim, "dim"),
        (c.hidden_dim, "hidden_dim"),
        (c.head_dim, "head_dim"),
    ] {
        if v == 0 || v % 32 != 0 {
            return Err(Error::BadShape(what));
        }
    }
    if c.n_layers == 0 || c.n_heads == 0 || c.n_kv_heads == 0 || c.vocab_size == 0 {
        return Err(Error::BadShape("zero dimension"));
    }
    if !c.n_heads.is_multiple_of(c.n_kv_heads) {
        return Err(Error::BadShape("n_heads % n_kv_heads"));
    }
    if c.vocab_size > u32::MAX as usize || c.eos.iter().any(|&e| e as usize >= c.vocab_size) {
        return Err(Error::BadShape("vocab/eos"));
    }
    if !(c.rms_eps.is_finite() && c.rms_eps > 0.0 && c.rope_theta.is_finite() && c.rope_theta > 1.0)
    {
        return Err(Error::BadShape("eps/theta"));
    }

    Ok(Header {
        config,
        ht_mask: u32_at(56),
        off_tokenizer: usize_at(64)?,
        off_embed: usize_at(72)?,
        off_layers: usize_at(80)?,
        layer_stride: usize_at(88)?,
        off_final_norm: usize_at(96)?,
    })
}

pub(crate) fn section<'a>(
    data: &'a [u8],
    off: usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8], Error> {
    let end = add(off, len).map_err(|_| Error::Truncated(what))?;
    data.get(off..end).ok_or(Error::Truncated(what))
}

fn read_f32s(src: &[u8], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
        *d = f32::from_le_bytes(*s);
    }
}

fn align64(n: usize) -> usize {
    (n + 63) & !63
}

pub(crate) fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::BadShape("overflow"))
}

pub(crate) fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::BadShape("overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic_and_short_input() {
        assert_eq!(
            Model::parse(&[0u8; 10]).err(),
            Some(Error::Truncated("header"))
        );
        assert_eq!(Model::parse(&[0u8; 128]).err(), Some(Error::BadMagic));

        let mut h = [0u8; 128];
        h[0..4].copy_from_slice(MAGIC);
        h[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(Model::parse(&h).err(), Some(Error::UnsupportedVersion(1)));
    }
}
