//! Byte-level BPE tokenizer and the Qwen3 chat template.
//!
//! The tokenizer section of `.asmbin` (offsets relative to its start):
//!
//! ```text
//! [0, 32)       u64 offsets of byte_to_tok, tok_offsets, tok_bytes, merge table
//! byte_to_tok   u32[256]          base token of each raw byte
//! tok_offsets   u32[vocab + 1]    token i is tok_bytes[off[i]..off[i+1]]
//! tok_bytes     u8[..]            raw UTF-8 bytes of every token
//! merge table   {u64 key, u32 merged, u32 rank}[mask + 1]
//!               open addressing, key = (left << 32) | right,
//!               slot = (key * FIB) >> (64 - bits), linear probing,
//!               rank == u32::MAX marks an empty slot
//! ```
//!
//! Encoding follows llm.asm exactly: map every byte to its base token, then
//! repeatedly merge the adjacent pair with the lowest rank (leftmost on ties).
//! There is no regex pre-tokenization, so results can differ from Hugging Face
//! for some inputs, but they match the llm.asm reference.

use crate::{
    model::{add, mul, section},
    Error,
};
use alloc::vec::Vec;

const FIB_MULT: u64 = 0x9E37_79B9_7F4A_7C15;
const EMPTY_RANK: u32 = u32::MAX;
const HT_ENTRY: usize = 16;

// Qwen3 special tokens used by the chat template.
pub const TOK_ENDOFTEXT: u32 = 151_643;
pub const TOK_IM_START: u32 = 151_644;
pub const TOK_IM_END: u32 = 151_645;
pub const TOK_THINK_START: u32 = 151_667;
pub const TOK_THINK_END: u32 = 151_668;
pub const TOK_USER: u32 = 872;
pub const TOK_ASSISTANT: u32 = 77_091;
/// `"\n"`.
pub const TOK_NL: u32 = 198;
/// `"\n\n"`.
pub const TOK_NL2: u32 = 271;
pub const TOK_TOOL_CALL: u32 = 151_657;
pub const TOK_TOOL_CALL_END: u32 = 151_658;
pub const TOK_TOOL_RESPONSE: u32 = 151_665;
pub const TOK_TOOL_RESPONSE_END: u32 = 151_666;

/// Special tokens recognized by [`Tokenizer::encode_with_specials`].
const SPECIALS: [u32; 9] = [
    TOK_ENDOFTEXT,
    TOK_IM_START,
    TOK_IM_END,
    TOK_THINK_START,
    TOK_THINK_END,
    TOK_TOOL_CALL,
    TOK_TOOL_CALL_END,
    TOK_TOOL_RESPONSE,
    TOK_TOOL_RESPONSE_END,
];

pub struct Tokenizer<'a> {
    byte_to_tok: [u32; 256],
    offsets: &'a [u8],
    bytes: &'a [u8],
    table: &'a [u8],
    mask: u64,
    shift: u32,
    vocab_size: usize,
    max_token_len: usize,
}

impl<'a> Tokenizer<'a> {
    /// Parses the tokenizer section at `off`, which must end at or before `limit`.
    pub(crate) fn parse(
        data: &'a [u8],
        off: usize,
        vocab_size: usize,
        mask: usize,
        limit: usize,
    ) -> Result<Self, Error> {
        let data = data.get(..limit).ok_or(Error::Truncated("tokenizer"))?;
        let hdr = section(data, off, 32, "tokenizer header")?;
        let rel = |i: usize| -> Result<usize, Error> {
            let v = u64::from_le_bytes(hdr[i * 8..i * 8 + 8].try_into().unwrap());
            add(
                off,
                usize::try_from(v).map_err(|_| Error::Truncated("tokenizer"))?,
            )
        };

        let b2t = section(data, rel(0)?, 256 * 4, "byte_to_tok")?;
        let offsets = section(data, rel(1)?, mul(vocab_size + 1, 4)?, "tok_offsets")?;

        let n_bytes = u32_at(offsets, vocab_size) as usize;
        let bytes = section(data, rel(2)?, n_bytes, "tok_bytes")?;

        let slots = mask.checked_add(1).ok_or(Error::BadShape("ht mask"))?;
        if !slots.is_power_of_two() || slots < 2 {
            return Err(Error::BadShape("ht mask"));
        }
        let table = section(data, rel(3)?, mul(slots, HT_ENTRY)?, "merge table")?;

        let mut byte_to_tok = [0u32; 256];
        for (i, t) in byte_to_tok.iter_mut().enumerate() {
            *t = u32_at(b2t, i);
            if *t as usize >= vocab_size {
                return Err(Error::BadShape("byte_to_tok"));
            }
        }

        // Offsets must be monotonic so that `token_bytes` never panics.
        let mut prev = 0;
        let mut max_token_len = 0;
        for i in 0..=vocab_size {
            let o = u32_at(offsets, i);
            if o < prev || o as usize > n_bytes {
                return Err(Error::BadShape("tok_offsets"));
            }
            max_token_len = max_token_len.max((o - prev) as usize);
            prev = o;
        }

        // A lookup stops at the first empty slot; a table without one would
        // make every lookup scan all slots.
        if !table
            .as_chunks::<HT_ENTRY>()
            .0
            .iter()
            .any(|e| e[12..16] == EMPTY_RANK.to_le_bytes())
        {
            return Err(Error::BadShape("merge table full"));
        }

        Ok(Self {
            byte_to_tok,
            offsets,
            bytes,
            table,
            mask: mask as u64,
            shift: 64 - slots.trailing_zeros(),
            vocab_size,
            max_token_len,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Length in bytes of the longest token.
    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    /// Raw bytes of a token (may be a partial UTF-8 sequence). Empty for
    /// out-of-range ids.
    pub fn token_bytes(&self, tok: u32) -> &'a [u8] {
        let t = tok as usize;
        if t >= self.vocab_size {
            return &[];
        }
        let s = u32_at(self.offsets, t) as usize;
        let e = u32_at(self.offsets, t + 1) as usize;
        &self.bytes[s..e]
    }

    /// Looks up the merge of `(a, b)`: `(rank, merged token)`.
    fn merge(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let key = ((a as u64) << 32) | b as u64;
        let mut slot = key.wrapping_mul(FIB_MULT) >> self.shift;
        // `parse` checked that an empty slot exists, so this terminates; the
        // bound is only a second line of defence.
        for _ in 0..=self.mask {
            let e = &self.table[slot as usize * HT_ENTRY..][..HT_ENTRY];
            let rank = u32::from_le_bytes(e[12..16].try_into().unwrap());
            if rank == EMPTY_RANK {
                return None;
            }
            if u64::from_le_bytes(e[0..8].try_into().unwrap()) == key {
                let merged = u32::from_le_bytes(e[8..12].try_into().unwrap());
                return ((merged as usize) < self.vocab_size).then_some((rank, merged));
            }
            slot = (slot + 1) & self.mask;
        }
        None
    }

    /// Appends the BPE encoding of `text` to `out`.
    ///
    /// The rank of every adjacent pair is cached, so each merge costs one
    /// linear scan for the minimum plus two table lookups: O(n^2) time in the
    /// worst case for n input bytes and O(n) extra memory.
    pub fn encode(&self, text: &[u8], out: &mut Vec<u32>) -> Result<(), Error> {
        let mut toks = Vec::new();
        toks.try_reserve_exact(text.len())
            .map_err(|_| Error::OutOfMemory)?;
        toks.extend(text.iter().map(|&b| self.byte_to_tok[b as usize]));

        // pairs[i] = merge of (toks[i], toks[i + 1]); rank EMPTY_RANK if none.
        let mut pairs = Vec::new();
        pairs
            .try_reserve_exact(toks.len().saturating_sub(1))
            .map_err(|_| Error::OutOfMemory)?;
        pairs.extend(
            toks.windows(2)
                .map(|w| self.merge(w[0], w[1]).unwrap_or((EMPTY_RANK, 0))),
        );

        loop {
            let mut best = EMPTY_RANK;
            let mut best_i = usize::MAX;
            for (i, &(rank, _)) in pairs.iter().enumerate() {
                if rank < best {
                    best = rank;
                    best_i = i;
                }
            }
            if best_i == usize::MAX {
                break;
            }

            let i = best_i;
            toks[i] = pairs[i].1;
            toks.remove(i + 1);
            pairs.remove(i);
            let lookup = |a: u32, b: u32| self.merge(a, b).unwrap_or((EMPTY_RANK, 0));
            if i > 0 {
                pairs[i - 1] = lookup(toks[i - 1], toks[i]);
            }
            if i < pairs.len() {
                pairs[i] = lookup(toks[i], toks[i + 1]);
            }
        }

        out.try_reserve(toks.len())
            .map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&toks);
        Ok(())
    }

    /// Appends one user turn of the Qwen3 chat template:
    ///
    /// ```text
    /// <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    /// ```
    ///
    /// followed by an empty `<think>\n\n</think>\n\n` block unless `think`.
    pub fn encode_user_turn(
        &self,
        text: &[u8],
        think: bool,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        out.try_reserve(12).map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&[TOK_IM_START, TOK_USER, TOK_NL]);
        self.encode(text, out)?;
        out.try_reserve(9).map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&[TOK_IM_END, TOK_NL, TOK_IM_START, TOK_ASSISTANT, TOK_NL]);
        if !think {
            out.extend_from_slice(&[TOK_THINK_START, TOK_NL2, TOK_THINK_END, TOK_NL2]);
        }
        Ok(())
    }
}

impl Tokenizer<'_> {
    /// Like [`Tokenizer::encode`], but the texts of the chat special tokens
    /// (`<|im_start|>`, `<tool_call>`, ...) become those tokens, as the
    /// Hugging Face tokenizer does. Only for trusted text such as our own
    /// system prompt: in user or tool text this would allow injecting turns.
    pub fn encode_with_specials(&self, text: &[u8], out: &mut Vec<u32>) -> Result<(), Error> {
        let mut start = 0;
        let mut i = 0;
        while i < text.len() {
            let special = (text[i] == b'<')
                .then(|| {
                    SPECIALS.iter().find(|&&t| {
                        let b = self.token_bytes(t);
                        !b.is_empty() && text[i..].starts_with(b)
                    })
                })
                .flatten();
            match special {
                Some(&t) => {
                    self.encode(&text[start..i], out)?;
                    out.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
                    out.push(t);
                    i += self.token_bytes(t).len();
                    start = i;
                }
                None => i += 1,
            }
        }
        self.encode(&text[start..], out)
    }

    /// Appends a tool result as the next user turn, followed by the
    /// assistant header (Qwen3 template):
    ///
    /// ```text
    /// <|im_start|>user\n<tool_response>\n{text}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
    /// ```
    ///
    /// plus the empty think block unless `think`. Special-token texts inside
    /// `text` are *not* interpreted.
    ///
    /// The tool output is cut to `max_tokens` tokens (followed by
    /// `... (truncated)`).
    pub fn encode_tool_response_turn(
        &self,
        text: &[u8],
        think: bool,
        max_tokens: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        out.try_reserve(16).map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&[TOK_IM_START, TOK_USER, TOK_NL, TOK_TOOL_RESPONSE]);
        let mut body = Vec::new();
        body.try_reserve(text.len() + 2)
            .map_err(|_| Error::OutOfMemory)?;
        body.push(b'\n');
        // No token is shorter than one byte, and long outputs would make the
        // O(n^2) BPE slow: cut the bytes first (at a UTF-8 boundary).
        let mut cut = text.len().min(max_tokens.saturating_mul(8));
        while cut > 0 && cut < text.len() && (text[cut] & 0xc0) == 0x80 {
            cut -= 1;
        }
        body.extend_from_slice(&text[..cut]);
        body.push(b'\n');
        let start = out.len();
        self.encode(&body, out)?;
        if out.len() - start > max_tokens || cut < text.len() {
            out.truncate(start + max_tokens.min(out.len() - start));
            self.encode(b"\n... (truncated)\n", out)?;
        }
        out.try_reserve(14).map_err(|_| Error::OutOfMemory)?;
        out.extend_from_slice(&[
            TOK_TOOL_RESPONSE_END,
            TOK_IM_END,
            TOK_NL,
            TOK_IM_START,
            TOK_ASSISTANT,
            TOK_NL,
        ]);
        if !think {
            out.extend_from_slice(&[TOK_THINK_START, TOK_NL2, TOK_THINK_END, TOK_NL2]);
        }
        Ok(())
    }
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}
