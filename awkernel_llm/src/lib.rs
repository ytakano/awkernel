//! # awkernel_llm
//!
//! A `no_std` + `alloc` inference engine for **Qwen3-0.6B** stored in the
//! `.asmbin` v2 format of [llm.asm](https://github.com/alesha-pro/llm.asm)
//! (`scripts/export_weights.py --scale-dtype fp16`).
//!
//! The crate does not depend on Awkernel. Its inputs are the model bytes
//! (borrowed, zero-copy), heap memory for activations and the KV cache, and an
//! [`Executor`] that runs data-parallel loops. This lets the same code run as a
//! host program (unit tests, `examples/llm_cli.rs`) and inside Awkernel.
//!
//! Layering:
//!
//! - [`model`] parses and validates the `.asmbin` file.
//! - [`tokenizer`] is the byte-level BPE encoder/decoder.
//! - [`q8`] and [`kernels`] are the numeric kernels (scalar and AVX2).
//! - [`transformer`] owns the activations and KV cache and runs one forward step.
//! - [`sampler`] picks the next token.
//! - [`chat`] drives a multi-turn chat one step at a time, so a caller can
//!   yield between tokens.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

pub mod agent;
pub mod chat;
pub mod cpu;
pub mod exec;
pub mod kernels;
pub mod model;
pub mod q8;
pub mod router;
pub mod sampler;
pub mod tokenizer;
pub mod tools;
pub mod transformer;
pub mod utf8;

pub use agent::{Agent, AgentConfig, AgentEvent, AgentStop, Summary};
pub use chat::{Chat, ChatConfig, Event, StopReason};
pub use cpu::Backend;
pub use exec::{Executor, InlineExecutor};
pub use model::{Config, Model};

/// Errors returned by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The file does not start with `"ASML"`.
    BadMagic,
    /// Only `.asmbin` v2 (FP16 block scales) is supported.
    UnsupportedVersion(u32),
    /// A required header flag is missing (tied embeddings, QK-norm, FP16 scales).
    UnsupportedFlags(u32),
    /// A header field is inconsistent with this engine (e.g. not a multiple of 32).
    BadShape(&'static str),
    /// A section lies (partly) outside the file.
    Truncated(&'static str),
    /// A heap allocation failed.
    OutOfMemory,
    /// The requested context length is zero or exceeds the model's maximum.
    BadContextLength,
    /// A token id is not smaller than the vocabulary size.
    BadToken(u32),
    /// The prompt does not fit into the context.
    PromptTooLong,
    /// `step()` was called while no turn is in progress.
    Idle,
    /// An invalid configuration value.
    BadConfig(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Allocates a zero-initialized vector, reporting allocation failure instead of
/// aborting (the KV cache alone is hundreds of MiB).
pub(crate) fn try_zeroed_vec<T: Copy + Default>(n: usize) -> Result<alloc::vec::Vec<T>, Error> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, T::default());
    Ok(v)
}
