//! Decides with the model itself whether a question needs a tool.
//!
//! In a chat, the 0.6B model often answers questions about the system
//! without calling a tool (it guesses). Asked directly "which of these
//! tools does this question need?", it does better. The router keeps a
//! separate, small KV cache holding a fixed classification prompt with a few
//! examples. Per question it only processes the question and reads the
//! logits of the first token of each label (`task`, `interrupt`, ...,
//! `none`); nothing is generated. With ~30 question tokens this costs about
//! as much as generating 30 tokens.

use crate::{
    cpu::Backend,
    exec::Executor,
    model::Model,
    tokenizer::{
        TOK_ASSISTANT, TOK_IM_END, TOK_IM_START, TOK_NL, TOK_NL2, TOK_THINK_END, TOK_THINK_START,
        TOK_USER,
    },
    tools::ToolSpec,
    transformer::Transformer,
    Error,
};
use alloc::{string::String, vec::Vec};
use core::fmt::Write;

/// Label for questions that need no tool.
const NONE: &str = "none";

/// A router cache size that fits the catalog prompt (~430 tokens) and a
/// question of ~150 tokens (143 MiB for Qwen3-0.6B).
pub const DEFAULT_LEN: usize = 640;

/// The router's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// No tool is needed.
    None,
    /// The tool with this index (in the list given to [`Router::new`]).
    Tool(usize),
}

pub struct Router {
    tf: Transformer,
    /// Classification prompt up to (excluding) the question.
    prefix: Vec<u32>,
    /// How much of `prefix` is in the cache.
    primed: usize,
    /// First token of each label: tools, then `none`.
    labels: Vec<u32>,
    /// Question turn being processed.
    query: Vec<u32>,
    next: usize,
}

impl Router {
    /// Builds the classification prompt for `tools` and allocates a cache of
    /// `max_seq` tokens (224 KiB per token for Qwen3-0.6B).
    pub fn new(
        model: &Model<'_>,
        tools: &[ToolSpec<'_>],
        no_tool_examples: &[&str],
        backend: Backend,
        max_seq: usize,
    ) -> Result<Self, Error> {
        let tok = &model.tokenizer;

        // Labels must start with different tokens.
        let mut labels = Vec::new();
        for name in tools.iter().map(|t| t.name).chain([NONE]) {
            let mut ids = Vec::new();
            tok.encode(name.as_bytes(), &mut ids)?;
            let first = *ids.first().ok_or(Error::BadConfig("empty tool name"))?;
            if labels.contains(&first) {
                return Err(Error::BadConfig(
                    "tool names must start with different tokens",
                ));
            }
            labels.push(first);
        }

        let mut system = String::from(
            "Decide which tool is needed to answer the user's question about the Awkernel \
             operating system. Reply with one word:\n",
        );
        for t in tools {
            let _ = writeln!(system, "- {}: {}", t.name, t.description);
        }
        let _ = write!(
            system,
            "- {NONE}: the question is not about the current state of this system: general \
             knowledge, math, chat, or an explanation of a concept (for example what an IRQ or a \
             network interface is in general)."
        );

        let mut prefix = Vec::new();
        prefix.extend_from_slice(&[TOK_IM_START]);
        tok.encode(b"system", &mut prefix)?;
        prefix.push(TOK_NL);
        tok.encode(system.as_bytes(), &mut prefix)?;
        prefix.extend_from_slice(&[TOK_IM_END, TOK_NL]);

        // Few-shot examples, interleaved so that no label dominates the end.
        let mut examples: Vec<(&str, &str)> = Vec::new();
        let rounds = tools
            .iter()
            .map(|t| t.examples.len())
            .max()
            .unwrap_or(0)
            .max(no_tool_examples.len());
        for i in 0..rounds {
            for t in tools {
                if let Some(q) = t.examples.get(i) {
                    examples.push((q, t.name));
                }
            }
            if let Some(q) = no_tool_examples.get(i) {
                examples.push((q, NONE));
            }
        }
        for (q, label) in examples {
            question_turn(model, q, &mut prefix)?;
            tok.encode(label.as_bytes(), &mut prefix)?;
            prefix.extend_from_slice(&[TOK_IM_END, TOK_NL]);
        }

        if prefix.len() + 64 > max_seq {
            return Err(Error::PromptTooLong);
        }
        Ok(Self {
            tf: Transformer::new(&model.config, max_seq, backend)?,
            prefix,
            primed: 0,
            labels,
            query: Vec::new(),
            next: 0,
        })
    }

    /// Number of labels (tools + `none`).
    pub fn labels(&self) -> usize {
        self.labels.len()
    }

    /// Tokens of the classification prompt.
    pub fn prompt_len(&self) -> usize {
        self.prefix.len()
    }

    /// Whether the prompt has been processed.
    pub fn is_primed(&self) -> bool {
        self.primed == self.prefix.len()
    }

    /// Processes one token of the classification prompt; returns `true` when
    /// it is done. Run it while idle so that the first question is fast.
    pub fn prime_step(&mut self, model: &Model<'_>, exec: &dyn Executor) -> Result<bool, Error> {
        if self.is_primed() {
            return Ok(true);
        }
        self.tf
            .forward(model, self.prefix[self.primed], self.primed, false, exec)?;
        self.primed += 1;
        Ok(self.is_primed())
    }

    /// Starts classifying `question`. Returns `false` if it is too long for
    /// the router (the caller should then let the chat model decide).
    pub fn start(&mut self, model: &Model<'_>, question: &str) -> Result<bool, Error> {
        self.query.clear();
        self.next = 0;
        let max_len = model.tokenizer.max_token_len().max(1);
        if self.prefix.len() + question.len() / max_len + 16 > self.tf.max_seq() {
            return Ok(false);
        }
        question_turn(model, question, &mut self.query)?;
        Ok(self.prefix.len() + self.query.len() <= self.tf.max_seq())
    }

    /// Runs one forward pass (priming first if needed). Returns the decision
    /// after the last question token.
    pub fn step(&mut self, model: &Model<'_>, exec: &dyn Executor) -> Result<Option<Route>, Error> {
        if !self.is_primed() {
            self.prime_step(model, exec)?;
            return Ok(None);
        }
        let Some(&tok) = self.query.get(self.next) else {
            return Err(Error::Idle);
        };
        let last = self.next + 1 == self.query.len();
        // The question goes right after the prompt; earlier questions are
        // simply overwritten.
        self.tf
            .forward(model, tok, self.prefix.len() + self.next, last, exec)?;
        self.next += 1;
        if !last {
            return Ok(None);
        }
        let logits = self.tf.logits();
        let mut best = 0;
        for (i, &l) in self.labels.iter().enumerate() {
            if logits[l as usize] > logits[self.labels[best] as usize] {
                best = i;
            }
        }
        Ok(Some(if best + 1 == self.labels.len() {
            Route::None
        } else {
            Route::Tool(best)
        }))
    }
}

/// `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
fn question_turn(model: &Model<'_>, q: &str, out: &mut Vec<u32>) -> Result<(), Error> {
    out.extend_from_slice(&[TOK_IM_START, TOK_USER, TOK_NL]);
    model.tokenizer.encode(q.as_bytes(), out)?;
    out.extend_from_slice(&[
        TOK_IM_END,
        TOK_NL,
        TOK_IM_START,
        TOK_ASSISTANT,
        TOK_NL,
        TOK_THINK_START,
        TOK_NL2,
        TOK_THINK_END,
        TOK_NL2,
    ]);
    Ok(())
}
