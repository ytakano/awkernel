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
//!
//! A system turn ([`Chat::set_system_prompt`]) is put at the start of the
//! cache once and kept by [`Chat::clear`] and context resets, so it is not
//! processed again. [`Chat::mark`] and [`Chat::rollback`] return the cache to
//! an earlier position (the entries before it are still valid), which lets a
//! caller replace a turn by a shorter one.

use crate::{
    cpu::Backend,
    exec::Executor,
    model::Model,
    sampler::Sampler,
    tokenizer::{TOK_IM_END, TOK_NL, TOK_TOOL_CALL, TOK_TOOL_CALL_END},
    transformer::Transformer,
    Error,
};
use alloc::vec::Vec;

/// Room kept for the answer when deciding whether a user turn still fits.
const ANSWER_RESERVE: usize = 64;
/// Room kept after a tool response (the answer, or another tool call).
const TOOL_RESERVE: usize = 256;

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
    /// Tool output is cut to this many tokens.
    pub max_tool_tokens: usize,
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
            max_tool_tokens: 600,
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
    /// The model finished a tool call (`</tool_call>`), or a turn started by
    /// [`Chat::start_turn_with_tool_call`] was fed. Run the tool and continue
    /// with [`Chat::start_tool_response`].
    ToolCall,
    /// A turn started by [`Chat::start_append`] was fed.
    Appended,
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

/// A position in the conversation, see [`Chat::mark`].
#[derive(Debug, Clone)]
pub struct Mark {
    pos: usize,
    pub(crate) carry: Vec<u32>,
    epoch: u64,
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
    /// System turn put at the start of every conversation.
    system: Vec<u32>,
    /// `system` is in the cache at `0..system.len()`.
    system_cached: bool,
    /// Index in `pending` after which `system` has been fed (this turn).
    system_end: Option<usize>,
    /// The current turn only feeds tokens: when done, report this reason and
    /// carry these tokens into the next turn.
    prefill_only: Option<(StopReason, [u32; 2], usize)>,
    /// Never generate `<tool_call>`.
    ban_tool_calls: bool,
    /// Incremented whenever earlier cache contents are discarded (clear,
    /// reset, new system prompt), which invalidates marks.
    epoch: u64,
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
            system: Vec::new(),
            system_cached: false,
            system_end: None,
            prefill_only: None,
            ban_tool_calls: false,
            epoch: 0,
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

    /// Context length.
    pub fn max_seq(&self) -> usize {
        self.tf.max_seq()
    }

    pub fn is_idle(&self) -> bool {
        self.phase == Phase::Idle
    }

    /// Prompt tokens of the current turn, without carried-over tokens but
    /// including the system turn if this turn starts a conversation.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.pending[self.carry_len..]
    }

    /// Tokens generated in the current (or last) turn.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Cache position of an empty conversation (after the system turn, if it
    /// is cached).
    fn base(&self) -> usize {
        if self.system_cached {
            self.system.len()
        } else {
            0
        }
    }

    /// Forgets the conversation (the cached system turn is kept).
    pub fn clear(&mut self) {
        self.cancel();
        self.pos = self.base();
        self.turn_start = self.pos;
        self.carry.clear();
        self.pending.clear();
        self.carry_len = 0;
        self.epoch += 1;
    }

    /// Sets the system prompt (trusted text, see
    /// [`crate::tokenizer::Tokenizer::encode_with_specials`]): the complete
    /// system turn including `<|im_start|>system\n` and `<|im_end|>\n`, e.g.
    /// from [`crate::tools::system_prompt`]. It is put at the start of every
    /// new conversation; the current one is cleared.
    pub fn set_system_prompt(&mut self, turn: &str) -> Result<(), Error> {
        let mut system = Vec::new();
        self.model
            .tokenizer
            .encode_with_specials(turn.as_bytes(), &mut system)?;
        if system.len() + ANSWER_RESERVE >= self.tf.max_seq() {
            return Err(Error::PromptTooLong);
        }
        self.system = system;
        self.system_cached = false;
        self.clear();
        Ok(())
    }

    /// Enables or disables reasoning mode for the following turns.
    pub fn set_think(&mut self, think: bool) {
        self.cfg.think = think;
    }

    /// Limit of generated tokens per turn, from the next turn on.
    pub fn set_max_new_tokens(&mut self, n: usize) {
        self.cfg.max_new_tokens = n;
    }

    /// Whether the model may start a tool call (`<tool_call>` is never
    /// sampled when disabled).
    pub fn set_tool_calls_allowed(&mut self, allowed: bool) {
        self.ban_tool_calls = !allowed;
    }

    /// The current position, to return to with [`Chat::rollback`]. Call it
    /// between turns. A mark can be used once.
    pub fn mark(&self) -> Mark {
        debug_assert!(self.is_idle());
        Mark {
            pos: self.pos,
            carry: self.carry.clone(),
            epoch: self.epoch,
        }
    }

    /// Returns the conversation to `mark`, forgetting everything after it.
    /// Costs no computation. Fails (after cancelling the current turn) if the
    /// cache was cleared, reset or rolled back since the mark was taken.
    pub fn rollback(&mut self, mark: &Mark) -> bool {
        self.rollback_keep(mark, 0)
    }

    /// Like [`Chat::rollback`], but keeps the first `keep` tokens after the
    /// mark (the carried tokens and whatever followed them), which must still
    /// be what the caller wants there.
    pub fn rollback_keep(&mut self, mark: &Mark, keep: usize) -> bool {
        self.cancel();
        let start = mark.pos.max(self.base());
        if mark.epoch != self.epoch || start + keep > self.pos {
            return false;
        }
        self.pos = start + keep;
        self.turn_start = self.pos;
        self.carry.clear();
        if keep == 0 {
            self.carry.extend_from_slice(&mark.carry);
        }
        // Positions after the mark will be overwritten: older marks are void.
        self.epoch += 1;
        true
    }

    /// Starts a user turn. Returns `true` if the conversation was reset
    /// because the context was nearly full.
    ///
    /// Tokenization is O(n^2) in the text length; texts that cannot fit into
    /// the context are rejected before tokenizing.
    pub fn start_turn(&mut self, text: &str) -> Result<bool, Error> {
        self.cancel();
        self.check_text_len(text)?;
        self.begin_pending()?;
        self.model.tokenizer.encode_user_turn(
            text.as_bytes(),
            self.cfg.think,
            &mut self.pending,
        )?;
        self.start_tokens(ANSWER_RESERVE)
    }

    /// Starts a user turn in which the assistant calls the tool `name`
    /// (without arguments) instead of generating, e.g. when a router has
    /// already decided that the tool is needed. The turn ends with
    /// [`StopReason::ToolCall`] without generating anything.
    pub fn start_turn_with_tool_call(&mut self, text: &str, name: &str) -> Result<bool, Error> {
        self.cancel();
        self.check_text_len(text)?;
        self.begin_pending()?;
        let tok = &self.model.tokenizer;
        tok.encode_user_turn(text.as_bytes(), self.cfg.think, &mut self.pending)?;
        self.pending
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.push(TOK_TOOL_CALL);
        let mut call = Vec::new();
        call.try_reserve(name.len() + 40)
            .map_err(|_| Error::OutOfMemory)?;
        call.extend_from_slice(b"\n{\"name\": \"");
        call.extend_from_slice(name.as_bytes());
        call.extend_from_slice(b"\", \"arguments\": {}}\n");
        tok.encode(&call, &mut self.pending)?;
        self.pending
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.push(TOK_TOOL_CALL_END);
        self.prefill_only = Some((StopReason::ToolCall, [TOK_IM_END, TOK_NL], 2));
        self.start_tokens(ANSWER_RESERVE)
    }

    /// Continues after [`StopReason::ToolCall`] with the tool's output as a
    /// `<tool_response>` turn, cut to `max_tool_tokens` and to the room left
    /// in the context ([`Chat::tool_room`]). Returns `true` if the
    /// conversation was reset because the context was full (the tool call is
    /// then lost).
    pub fn start_tool_response(&mut self, output: &str) -> Result<bool, Error> {
        self.cancel();
        let max_tokens = self.cfg.max_tool_tokens.min(self.tool_room()).max(16);
        self.begin_pending()?;
        self.model.tokenizer.encode_tool_response_turn(
            output.as_bytes(),
            self.cfg.think,
            max_tokens,
            &mut self.pending,
        )?;
        self.start_tokens(TOOL_RESERVE)
    }

    /// Tokens of tool output that still fit before the next tool response
    /// would force a reset (template overhead and an answer reserved).
    pub fn tool_room(&self) -> usize {
        const OVERHEAD: usize = 32;
        self.tf
            .max_seq()
            .saturating_sub(self.pos + self.carry.len() + OVERHEAD + TOOL_RESERVE)
    }

    /// Feeds `tokens` without generating; the turn ends with
    /// [`StopReason::Appended`]. `tokens` must form complete turns (ending
    /// with `<|im_end|>\n`).
    pub fn start_append(&mut self, tokens: &[u32]) -> Result<bool, Error> {
        let reset = self.start_raw(tokens)?;
        self.prefill_only = Some((StopReason::Appended, [0; 2], 0));
        Ok(reset)
    }

    /// Feeds the system turn ahead of the first question (prefill only,
    /// ends with [`StopReason::Appended`]). Returns `false` if there is
    /// nothing to do (no system prompt, already cached, or a conversation is
    /// in progress).
    pub fn start_system(&mut self) -> Result<bool, Error> {
        if self.system.is_empty() || self.system_cached || self.pos != 0 || !self.is_idle() {
            return Ok(false);
        }
        self.begin_pending()?;
        if self.carry_len != 0 {
            return Ok(false);
        }
        self.pending
            .try_reserve(self.system.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.extend_from_slice(&self.system);
        self.system_end = Some(self.system.len());
        self.prefill_only = Some((StopReason::Appended, [0; 2], 0));
        self.turn_start = 0;
        self.generated = 0;
        self.phase = Phase::Prefill { next: 0 };
        Ok(true)
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
        self.start_tokens(ANSWER_RESERVE)
    }

    fn check_text_len(&self, text: &str) -> Result<(), Error> {
        // Every token is at most `max_token_len` bytes, so this is a lower
        // bound of the token count.
        let max_len = self.model.tokenizer.max_token_len().max(1);
        if text.len() / max_len >= self.tf.max_seq() {
            return Err(Error::PromptTooLong);
        }
        Ok(())
    }

    /// `pending = carry`.
    fn begin_pending(&mut self) -> Result<(), Error> {
        self.prefill_only = None;
        self.pending.clear();
        self.carry_len = 0;
        self.pending
            .try_reserve(self.carry.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.pending.extend_from_slice(&self.carry);
        self.carry_len = self.carry.len();
        Ok(())
    }

    fn start_tokens(&mut self, reserve: usize) -> Result<bool, Error> {
        let max_seq = self.tf.max_seq();
        let mut n = self.pending.len();
        let mut reset = false;
        if n == self.carry_len {
            self.prefill_only = None;
            return Err(Error::PromptTooLong);
        }
        if self.pos > self.base() && self.pos + n + reserve >= max_seq {
            // Start a fresh conversation (keeping the cached system turn); the
            // carried closing tokens belong to the old one.
            self.pending.drain(..self.carry_len);
            self.carry_len = 0;
            self.carry.clear();
            self.pos = self.base();
            self.epoch += 1;
            n = self.pending.len();
            reset = true;
        }
        self.system_end = None;
        if self.pos == 0 && !self.system.is_empty() {
            // A new conversation starts with the system turn.
            self.pending
                .try_reserve(self.system.len())
                .map_err(|_| Error::OutOfMemory)?;
            self.pending
                .splice(self.carry_len..self.carry_len, self.system.iter().copied());
            self.system_end = Some(self.carry_len + self.system.len());
            n = self.pending.len();
        }
        if self.pos + n + 1 > max_seq {
            // No room for even one generated token.
            self.pending.truncate(self.carry_len);
            self.prefill_only = None;
            self.system_end = None;
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
                // A system turn fed in this turn stays cached.
                self.pos = self.turn_start.max(self.base());
                self.carry.clear();
                self.carry
                    .extend_from_slice(&self.pending[..self.carry_len]);
            }
            Phase::Feed { tok } => self.set_carry(Some(tok)),
        }
        self.prefill_only = None;
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
                let generate = self.prefill_only.is_none() && self.cfg.max_new_tokens > 0;
                let want_logits = last && generate;
                self.feed(self.pending[next], want_logits, exec)?;
                if self.system_end == Some(next + 1) {
                    self.system_cached = true;
                }
                if !last {
                    self.phase = Phase::Prefill { next: next + 1 };
                    return Ok(Event::Prefill {
                        remaining: self.pending.len() - next - 1,
                    });
                }
                if let Some((reason, carry, n)) = self.prefill_only.take() {
                    self.carry.clear();
                    self.carry.extend_from_slice(&carry[..n]);
                    self.phase = Phase::Idle;
                    return Ok(Event::Done(reason));
                }
                if !want_logits {
                    return Ok(self.finish(StopReason::MaxTokens, None));
                }
                let tok = self.sample();
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
                let next = self.sample();
                Ok(self.emit(next))
            }
        }
    }

    fn sample(&mut self) -> u32 {
        let banned = self.ban_tool_calls.then_some(TOK_TOOL_CALL);
        self.sampler.sample_except(self.tf.logits(), banned)
    }

    fn emit(&mut self, tok: u32) -> Event {
        if self.model.config.is_eos(tok) {
            return self.finish(StopReason::Eos, None);
        }
        if tok == TOK_TOOL_CALL_END {
            // Closed later, as for the other stop reasons: `</tool_call>`
            // and `<|im_end|>\n` are fed at the start of the tool response.
            self.generated += 1;
            return self.finish(StopReason::ToolCall, Some(tok));
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
