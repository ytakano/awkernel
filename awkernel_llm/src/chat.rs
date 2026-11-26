//! Multi-turn chat, driven one forward step at a time.
//!
//! [`Chat::start_turn`] tokenizes a user message; each [`Chat::step`] then
//! runs **at most one** forward pass and reports what happened, so a caller
//! (an async task) can yield, print or cancel between tokens. The KV cache is
//! kept across turns; only the new turn is prefilled.
//!
//! The cache contents follow llm.asm `--chat`: every assistant turn is closed
//! with `<|im_end|>\n`, and the cache is reset when the next turn would come
//! within 64 tokens of the context limit. To keep the one-forward-per-step
//! rule, the closing tokens (and a generated token that was not fed yet) are
//! not run when a turn ends but carried over and prefilled at the start of the
//! next turn.

use crate::{
    cpu::Backend,
    exec::Executor,
    model::Model,
    sampler::Sampler,
    tokenizer::{TOK_IM_END, TOK_NL},
    transformer::Transformer,
    Error,
};
use alloc::vec::Vec;

/// Room kept for the answer when deciding whether a turn still fits.
const ANSWER_RESERVE: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct ChatConfig {
    /// Context length (KV cache size) in tokens.
    pub max_seq: usize,
    /// Maximum tokens generated per turn.
    pub max_new_tokens: usize,
    /// Qwen3 reasoning mode (`<think>...</think>`).
    pub think: bool,
    /// `<= 0` for greedy decoding.
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            max_seq: 2048,
            max_new_tokens: 512,
            think: false,
            temperature: 0.0,
            top_p: 0.95,
            seed: 42,
        }
    }
}

/// Result of one [`Chat::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// One prompt token was processed; `remaining` are left.
    Prefill { remaining: usize },
    /// A new token was generated (it is fed back in the next step).
    Token(u32),
    /// The turn has ended and the chat is idle.
    Done(StopReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model produced an end-of-sequence token.
    Eos,
    /// `max_new_tokens` were generated.
    MaxTokens,
    /// The context is full; the next turn starts from an empty cache.
    ContextFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// `pending[next..]` are still to be fed.
    Prefill {
        next: usize,
    },
    /// `tok` was emitted and must be fed next.
    Feed {
        tok: u32,
    },
}

pub struct Chat<'m> {
    model: &'m Model<'m>,
    tf: Transformer,
    sampler: Sampler,
    cfg: ChatConfig,
    /// Number of tokens in the KV cache.
    pos: usize,
    /// Cache length when the current turn started (restored on cancel).
    turn_start: usize,
    /// Tokens of the previous turn that still have to be fed (closing tokens).
    carry: Vec<u32>,
    /// `carry` followed by the current turn's prompt.
    pending: Vec<u32>,
    carry_len: usize,
    phase: Phase,
    generated: usize,
}

impl<'m> Chat<'m> {
    /// Allocates the KV cache and buffers (see [`Transformer::new`]).
    pub fn new(model: &'m Model<'m>, cfg: ChatConfig, backend: Backend) -> Result<Self, Error> {
        let c = &model.config;
        Ok(Self {
            model,
            tf: Transformer::new(c, cfg.max_seq, backend)?,
            sampler: Sampler::new(backend, c.vocab_size, cfg.temperature, cfg.top_p, cfg.seed)?,
            cfg,
            pos: 0,
            turn_start: 0,
            carry: Vec::new(),
            pending: Vec::new(),
            carry_len: 0,
            phase: Phase::Idle,
            generated: 0,
        })
    }

    pub fn model(&self) -> &'m Model<'m> {
        self.model
    }

    pub fn config(&self) -> &ChatConfig {
        &self.cfg
    }

    pub fn transformer(&self) -> &Transformer {
        &self.tf
    }

    /// Tokens currently in the KV cache.
    pub fn context_len(&self) -> usize {
        self.pos
    }

    pub fn is_idle(&self) -> bool {
        self.phase == Phase::Idle
    }

    /// Prompt tokens of the current turn (without carried-over tokens).
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.pending[self.carry_len..]
    }

    /// Tokens generated in the current (or last) turn.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Forgets the conversation.
    pub fn clear(&mut self) {
        self.pos = 0;
        self.turn_start = 0;
        self.carry.clear();
        self.pending.clear();
        self.carry_len = 0;
        self.phase = Phase::Idle;
    }

    /// Enables or disables reasoning mode for the following turns.
    pub fn set_think(&mut self, think: bool) {
        self.cfg.think = think;
    }

    /// Starts a user turn. Returns `true` if the conversation was reset
    /// because the context was nearly full.
    ///
    /// Tokenization is O(n^2) in the text length; texts that cannot fit into
    /// the context are rejected before tokenizing.
    pub fn start_turn(&mut self, text: &str) -> Result<bool, Error> {
        self.cancel();
        // Every token is at most `max_token_len` bytes, so this is a lower
        // bound of the token count.
        let max_len = self.model.tokenizer.max_token_len().max(1);
        if text.len() / max_len >= self.tf.max_seq() {
            return Err(Error::PromptTooLong);
        }
        self.begin_pending()?;
        self.model.tokenizer.encode_user_turn(
            text.as_bytes(),
            self.cfg.think,
            &mut self.pending,
        )?;
        self.start_tokens()
    }

    /// Starts a turn from raw token ids (no chat template).
    pub fn start_raw(&mut self, tokens: &[u32]) -> Result<bool, Error> {
        self.cancel();
        if let Some(&t) = tokens
            .iter()
            .find(|&&t| t as usize >= self.model.config.vocab_size)
        {
            return Err(Error::BadToken(t));
        }
        self.begin_pending()?;
        self.pending
            .try_reserve(tokens.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.extend_from_slice(tokens);
        self.start_tokens()
    }

    /// `pending = carry`.
    fn begin_pending(&mut self) -> Result<(), Error> {
        self.pending.clear();
        self.carry_len = 0;
        self.pending
            .try_reserve(self.carry.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.extend_from_slice(&self.carry);
        self.carry_len = self.carry.len();
        Ok(())
    }

    fn start_tokens(&mut self) -> Result<bool, Error> {
        let max_seq = self.tf.max_seq();
        let mut n = self.pending.len();
        let mut reset = false;
        if self.pos > 0 && self.pos + n + ANSWER_RESERVE >= max_seq {
            // Start a fresh conversation; the carried closing tokens belong to
            // the old one.
            self.pending.drain(..self.carry_len);
            self.carry_len = 0;
            self.carry.clear();
            self.pos = 0;
            n = self.pending.len();
            reset = true;
        }
        if n == self.carry_len || n + 1 > max_seq {
            // Empty prompt, or no room for even one generated token.
            self.pending.truncate(self.carry_len);
            return Err(Error::PromptTooLong);
        }
        self.carry.clear();
        self.turn_start = self.pos;
        self.generated = 0;
        self.phase = Phase::Prefill { next: 0 };
        Ok(reset)
    }

    /// Stops the current turn without running any forward pass.
    ///
    /// During prefill the cache is rolled back to the start of the turn, so
    /// the partial user message is forgotten. During generation the partial
    /// answer is kept and closed at the start of the next turn.
    pub fn cancel(&mut self) {
        match self.phase {
            Phase::Idle => {}
            Phase::Prefill { .. } => {
                self.pos = self.turn_start;
                self.carry.clear();
                self.carry
                    .extend_from_slice(&self.pending[..self.carry_len]);
            }
            Phase::Feed { tok } => self.set_carry(Some(tok)),
        }
        self.phase = Phase::Idle;
    }

    /// Runs at most one forward pass. On error the turn is abandoned (as by
    /// [`Chat::cancel`]).
    pub fn step(&mut self, exec: &dyn Executor) -> Result<Event, Error> {
        let r = self.step_inner(exec);
        if r.is_err() {
            self.cancel();
        }
        r
    }

    fn step_inner(&mut self, exec: &dyn Executor) -> Result<Event, Error> {
        match self.phase {
            Phase::Idle => Err(Error::Idle),
            Phase::Prefill { next } => {
                let last = next + 1 == self.pending.len();
                let want_logits = last && self.cfg.max_new_tokens > 0;
                self.feed(self.pending[next], want_logits, exec)?;
                if !last {
                    self.phase = Phase::Prefill { next: next + 1 };
                    return Ok(Event::Prefill {
                        remaining: self.pending.len() - next - 1,
                    });
                }
                if !want_logits {
                    return Ok(self.finish(StopReason::MaxTokens, None));
                }
                let tok = self.sampler.sample(self.tf.logits());
                Ok(self.emit(tok))
            }
            Phase::Feed { tok } => {
                if self.pos >= self.tf.max_seq() {
                    self.phase = Phase::Idle;
                    return Ok(Event::Done(StopReason::ContextFull));
                }
                if self.generated >= self.cfg.max_new_tokens {
                    // Feeding `tok` is deferred to the next turn.
                    return Ok(self.finish(StopReason::MaxTokens, Some(tok)));
                }
                self.feed(tok, true, exec)?;
                let next = self.sampler.sample(self.tf.logits());
                Ok(self.emit(next))
            }
        }
    }

    fn emit(&mut self, tok: u32) -> Event {
        if self.model.config.is_eos(tok) {
            return self.finish(StopReason::Eos, None);
        }
        self.generated += 1;
        self.phase = Phase::Feed { tok };
        Event::Token(tok)
    }

    fn finish(&mut self, reason: StopReason, unfed: Option<u32>) -> Event {
        self.set_carry(unfed);
        self.phase = Phase::Idle;
        Event::Done(reason)
    }

    /// Closing tokens fed at the start of the next turn: the last generated
    /// token if it was not fed, then `<|im_end|>\n`.
    fn set_carry(&mut self, unfed: Option<u32>) {
        self.carry.clear();
        self.carry.extend(unfed);
        self.carry.extend_from_slice(&[TOK_IM_END, TOK_NL]);
    }

    fn feed(&mut self, tok: u32, want_logits: bool, exec: &dyn Executor) -> Result<(), Error> {
        self.tf
            .forward(self.model, tok, self.pos, want_logits, exec)?;
        self.pos += 1;
        Ok(())
    }

    /// Bytes of a token for display. Special tokens such as `<think>` are
    /// stored as their text.
    pub fn token_bytes(&self, tok: u32) -> &'m [u8] {
        self.model.tokenizer.token_bytes(tok)
    }
}
