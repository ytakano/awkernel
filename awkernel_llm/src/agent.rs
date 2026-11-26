//! A question-answering loop that lets the model call tools, like a coding
//! agent: generate, run the requested tool, feed its output back, repeat
//! until the model answers.
//!
//! [`Agent::step`] runs at most one forward pass, so the caller can yield,
//! print and cancel between steps. One question goes through:
//!
//! 1. **Routing** (optional): the [`Router`] decides from the question alone
//!    whether a tool is needed. If so, the tool call is put into the turn
//!    directly, so the model cannot skip it and guess.
//! 2. **Generation**: the model answers, possibly calling (more) tools. At
//!    most `max_tool_steps` calls are run; after that `<tool_call>` can no
//!    longer be generated, which forces an answer. A repeated call is
//!    answered with a reminder instead of the output again.
//! 3. **Compaction**: the conversation is rolled back to before the question
//!    and replaced by `question, tool call(s) with "(omitted)" results,
//!    answer`. Tool outputs (hundreds of tokens each) do not pile up in the
//!    context, the model cannot answer later questions from stale outputs,
//!    and the history still shows that the facts came from tools.
//!
//! Cancelling forgets the question completely.

use crate::{
    chat::{Chat, Event, Mark, StopReason},
    exec::Executor,
    router::{Route, Router},
    tokenizer::{
        TOK_ASSISTANT, TOK_IM_END, TOK_IM_START, TOK_NL, TOK_NL2, TOK_THINK_END, TOK_THINK_START,
        TOK_TOOL_CALL, TOK_TOOL_CALL_END, TOK_TOOL_RESPONSE, TOK_TOOL_RESPONSE_END, TOK_USER,
    },
    tools::{parse_tool_call, system_prompt, ToolSpec},
    Error,
};
use alloc::{format, string::String, vec::Vec};

#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    /// Tool calls run per question.
    pub max_tool_steps: usize,
    /// Tokens generated per question (all rounds together).
    pub max_new_tokens: usize,
    /// Before a question, the conversation is cleared if fewer tokens than
    /// this are free.
    pub headroom: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_steps: 3,
            max_new_tokens: 512,
            headroom: 1024,
        }
    }
}

/// Less room than this for tool output is not worth another call.
const MIN_TOOL_ROOM: usize = 128;

/// Result of one [`Agent::step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// Work without visible output (routing, prefill, a hidden tool call,
    /// compaction).
    Busy,
    /// A token of the answer, to be shown.
    Token(u32),
    /// The tool `name` is about to be run (`routed`: chosen by the router).
    ToolCall { name: String, routed: bool },
    /// The answer is complete. Compaction follows; keep stepping until
    /// [`AgentEvent::Idle`].
    Answered(Summary),
    /// Nothing to do.
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStop {
    /// The model finished its answer.
    Answer,
    /// The token budget was used up.
    MaxTokens,
    /// The context was full.
    ContextFull,
    /// The model wrote a tool call that could not be parsed and stopped.
    MalformedCall,
    /// [`Agent::cancel`] was called.
    Cancelled,
    /// The engine failed.
    Error(Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub stop: AgentStop,
    pub tool_calls: usize,
    /// Tool chosen by the router, if any.
    pub routed: Option<usize>,
    /// Prompt tokens processed (question, tool outputs).
    pub prompt_tokens: usize,
    /// Answer tokens shown.
    pub answer_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Priming,
    /// The system turn is being fed.
    PrimingSystem,
    Routing,
    Chat,
    /// The answer was reported; compact the history next.
    Compact,
    /// The compacted history is being fed.
    Replay,
}

/// Runs a tool by name; `None` if there is no such tool.
pub type RunTool<'a> = dyn FnMut(&str) -> Option<String> + 'a;

pub struct Agent<'m> {
    chat: Chat<'m>,
    router: Option<Router>,
    tools: Vec<ToolSpec<'static>>,
    cfg: AgentConfig,
    phase: Phase,

    question: String,
    mark: Option<Mark>,
    /// Tools called for this question (`None`: unparsable call).
    calls: Vec<Option<usize>>,
    /// Text of the tool call being generated.
    call: Option<Vec<u8>>,
    /// Answer tokens of the final round.
    answer: Vec<u32>,
    summary: Summary,
    restarted: bool,
}

impl<'m> Agent<'m> {
    pub fn new(chat: Chat<'m>, cfg: AgentConfig) -> Self {
        Self {
            chat,
            router: None,
            tools: Vec::new(),
            cfg,
            phase: Phase::Idle,
            question: String::new(),
            mark: None,
            calls: Vec::new(),
            call: None,
            answer: Vec::new(),
            summary: empty_summary(),
            restarted: false,
        }
    }

    pub fn chat(&self) -> &Chat<'m> {
        &self.chat
    }

    /// Direct access, e.g. for a self test before tools are set.
    pub fn chat_mut(&mut self) -> &mut Chat<'m> {
        &mut self.chat
    }

    pub fn tools(&self) -> &[ToolSpec<'static>] {
        &self.tools
    }

    pub fn is_idle(&self) -> bool {
        self.phase == Phase::Idle
    }

    /// Sets the tools (and the system prompt describing them) and, if
    /// `router` is given, the router. Clears the conversation.
    pub fn set_tools(
        &mut self,
        tools: &[ToolSpec<'static>],
        preamble: Option<&str>,
        router: Option<Router>,
    ) -> Result<(), Error> {
        self.cancel();
        if tools.is_empty() {
            self.tools.clear();
            self.router = None;
            self.chat.clear();
            return Ok(());
        }
        if router
            .as_ref()
            .is_some_and(|r| r.labels() != tools.len() + 1)
        {
            return Err(Error::BadConfig("the router was built for other tools"));
        }
        self.chat
            .set_system_prompt(&system_prompt(tools, preamble))?;
        self.tools.clear();
        self.tools.extend_from_slice(tools);
        self.router = router;
        Ok(())
    }

    /// Processes the system prompt and the router prompt ahead of the first
    /// question. Keep stepping until [`AgentEvent::Idle`].
    pub fn warm_up(&mut self) {
        if self.phase == Phase::Idle {
            self.phase = Phase::Priming;
        }
    }

    /// Forgets the conversation.
    pub fn clear(&mut self) {
        self.cancel();
        self.chat.clear();
    }

    pub fn set_think(&mut self, on: bool) {
        self.chat.set_think(on);
    }

    /// Starts answering `question`.
    pub fn ask(&mut self, question: &str) -> Result<(), Error> {
        self.cancel();
        // Room for the question, tool outputs and the answer.
        if self.chat.max_seq() - self.chat.context_len() < self.cfg.headroom {
            self.chat.clear();
        }
        self.question.clear();
        self.question.push_str(question);
        self.restarted = false;
        self.begin_question();
        match &mut self.router {
            Some(r) => {
                if r.start(self.chat.model(), question)? {
                    self.phase = Phase::Routing;
                    Ok(())
                } else {
                    self.start_chat(None)
                }
            }
            None => self.start_chat(None),
        }
    }

    fn begin_question(&mut self) {
        self.mark = Some(self.chat.mark());
        self.calls.clear();
        self.call = None;
        self.answer.clear();
        self.summary = empty_summary();
    }

    /// Stops the current question and forgets it.
    pub fn cancel(&mut self) {
        match self.phase {
            Phase::Routing | Phase::Chat | Phase::Compact => {
                self.chat.cancel();
                if let Some(m) = self.mark.take() {
                    self.chat.rollback(&m);
                }
            }
            Phase::Replay | Phase::PrimingSystem => {
                // Keep the conversation consistent: drop the partial feed.
                self.chat.cancel();
            }
            Phase::Idle | Phase::Priming => {}
        }
        self.phase = Phase::Idle;
    }

    fn start_chat(&mut self, route: Option<usize>) -> Result<(), Error> {
        let budget = self
            .cfg
            .max_new_tokens
            .saturating_sub(self.summary.answer_tokens);
        self.chat.set_max_new_tokens(budget);
        self.chat
            .set_tool_calls_allowed(self.cfg.max_tool_steps > 0);
        match route {
            Some(i) => {
                self.summary.routed = Some(i);
                let name = self.tools[i].name;
                self.chat.start_turn_with_tool_call(&self.question, name)?;
                self.call = Some(Vec::from(&b"\n{\"name\": \""[..]));
                if let Some(c) = &mut self.call {
                    c.extend_from_slice(name.as_bytes());
                    c.extend_from_slice(b"\", \"arguments\": {}}\n");
                }
            }
            None => {
                self.chat.start_turn(&self.question)?;
            }
        }
        self.phase = Phase::Chat;
        Ok(())
    }

    /// Runs at most one forward pass. `run_tool` is called (synchronously)
    /// when the model calls a tool.
    pub fn step(&mut self, exec: &dyn Executor, run_tool: &mut RunTool<'_>) -> AgentEvent {
        match self.step_inner(exec, run_tool) {
            Ok(ev) => ev,
            Err(e) => {
                // Errors in background work (priming, compaction) are not
                // reported: the answer, if any, was reported already.
                let background = matches!(
                    self.phase,
                    Phase::Priming | Phase::PrimingSystem | Phase::Compact | Phase::Replay
                );
                self.cancel();
                if background {
                    AgentEvent::Idle
                } else {
                    self.summary.stop = AgentStop::Error(e);
                    AgentEvent::Answered(self.summary)
                }
            }
        }
    }

    fn step_inner(
        &mut self,
        exec: &dyn Executor,
        run_tool: &mut RunTool<'_>,
    ) -> Result<AgentEvent, Error> {
        match self.phase {
            Phase::Idle => Ok(AgentEvent::Idle),
            Phase::Priming => {
                if let Some(r) = &mut self.router {
                    if !r.prime_step(self.chat.model(), exec)? {
                        return Ok(AgentEvent::Busy);
                    }
                }
                self.phase = if self.chat.start_system()? {
                    Phase::PrimingSystem
                } else {
                    Phase::Idle
                };
                Ok(AgentEvent::Busy)
            }
            Phase::PrimingSystem => match self.chat.step(exec)? {
                Event::Done(_) => {
                    self.phase = Phase::Idle;
                    Ok(AgentEvent::Idle)
                }
                _ => Ok(AgentEvent::Busy),
            },
            Phase::Routing => {
                let router = self.router.as_mut().ok_or(Error::Idle)?;
                match router.step(self.chat.model(), exec)? {
                    None => Ok(AgentEvent::Busy),
                    Some(Route::None) => {
                        self.start_chat(None)?;
                        Ok(AgentEvent::Busy)
                    }
                    Some(Route::Tool(i)) => {
                        self.start_chat(Some(i))?;
                        Ok(AgentEvent::Busy)
                    }
                }
            }
            Phase::Chat => self.step_chat(exec, run_tool),
            Phase::Compact => {
                self.compact()?;
                Ok(AgentEvent::Busy)
            }
            Phase::Replay => match self.chat.step(exec)? {
                Event::Done(_) => {
                    self.phase = Phase::Idle;
                    Ok(AgentEvent::Idle)
                }
                _ => Ok(AgentEvent::Busy),
            },
        }
    }

    fn step_chat(
        &mut self,
        exec: &dyn Executor,
        run_tool: &mut RunTool<'_>,
    ) -> Result<AgentEvent, Error> {
        match self.chat.step(exec)? {
            Event::Prefill { .. } => {
                self.summary.prompt_tokens += 1;
                Ok(AgentEvent::Busy)
            }
            Event::Token(t) => {
                if t == TOK_TOOL_CALL {
                    self.call = Some(Vec::new());
                    return Ok(AgentEvent::Busy);
                }
                if let Some(c) = &mut self.call {
                    c.extend_from_slice(self.chat.token_bytes(t));
                    return Ok(AgentEvent::Busy);
                }
                self.answer.push(t);
                self.summary.answer_tokens += 1;
                Ok(AgentEvent::Token(t))
            }
            Event::Done(StopReason::ToolCall) => self.tool_call(run_tool),
            Event::Done(reason) => {
                // The last prompt token of a turn that ends immediately.
                if self.call.is_some() {
                    // A call that was not closed: run it if it parses.
                    let text = self.call_text();
                    if parse_tool_call(&text).is_some() {
                        return self.tool_call(run_tool);
                    }
                    self.call = None;
                    return Ok(self.answered(AgentStop::MalformedCall));
                }
                let stop = match reason {
                    StopReason::Eos => AgentStop::Answer,
                    StopReason::MaxTokens => AgentStop::MaxTokens,
                    StopReason::ContextFull => AgentStop::ContextFull,
                    _ => AgentStop::Answer,
                };
                Ok(self.answered(stop))
            }
        }
    }

    fn call_text(&self) -> String {
        String::from_utf8_lossy(self.call.as_deref().unwrap_or(&[])).into_owned()
    }

    /// Runs the tool the model asked for and continues with its output.
    fn tool_call(&mut self, run_tool: &mut RunTool<'_>) -> Result<AgentEvent, Error> {
        let text = self.call_text();
        self.call = None;
        // Text before the call is not part of the final answer.
        self.answer.clear();
        self.summary.tool_calls += 1;
        let routed = self.summary.routed.is_some() && self.calls.is_empty();

        let idx = parse_tool_call(&text).and_then(|n| self.tools.iter().position(|t| t.name == n));
        let repeated = idx.is_some() && self.calls.contains(&idx);
        self.calls.push(idx);

        let (name, output) = match idx {
            None => (
                String::from("?"),
                format!(
                    "error: invalid tool call. Available tools: {}.",
                    self.tools
                        .iter()
                        .map(|t| t.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ),
            Some(i) if repeated => (
                String::from(self.tools[i].name),
                String::from(
                    "The output of this tool is already shown above. Answer the question now.",
                ),
            ),
            Some(i) => {
                let name = self.tools[i].name;
                let out =
                    run_tool(name).unwrap_or_else(|| format!("error: {name} is not available."));
                (String::from(name), out)
            }
        };

        // After the last allowed call (or a repeat), or when another tool
        // response would not fit, the model must answer.
        let more = self.summary.tool_calls < self.cfg.max_tool_steps
            && !repeated
            && self.chat.tool_room() >= 2 * MIN_TOOL_ROOM;
        self.chat.set_tool_calls_allowed(more);
        let budget = self
            .cfg
            .max_new_tokens
            .saturating_sub(self.summary.answer_tokens);
        self.chat.set_max_new_tokens(budget);

        if self.chat.start_tool_response(&output)? {
            // The context overflowed and was cleared: the question is gone.
            // Ask it once more from an empty conversation.
            if self.restarted {
                return Ok(self.answered(AgentStop::ContextFull));
            }
            self.restarted = true;
            self.chat.clear();
            self.begin_question();
            self.start_chat(None)?;
        }
        Ok(AgentEvent::ToolCall { name, routed })
    }

    fn answered(&mut self, stop: AgentStop) -> AgentEvent {
        self.summary.stop = stop;
        self.restarted = false;
        self.phase = Phase::Compact;
        AgentEvent::Answered(self.summary)
    }

    /// Replaces the question's turns by the compact history (see the module
    /// documentation). The question turn (and a routed first call) is
    /// already in the cache in exactly this form, so only what follows it is
    /// fed again. Without tool calls (and without reasoning) the cache
    /// already holds the compact form, so nothing is done.
    fn compact(&mut self) -> Result<(), Error> {
        let Some(mark) = self.mark.take() else {
            self.phase = Phase::Idle;
            return Ok(());
        };
        let think = self.chat.config().think;
        if self.calls.is_empty() && !think {
            self.phase = Phase::Idle;
            return Ok(());
        }

        let tok = &self.chat.model().tokenizer;
        let mut h = Vec::new();
        h.extend_from_slice(&mark.carry);
        tok.encode_user_turn(self.question.as_bytes(), think, &mut h)?;
        let routed = self.summary.routed.is_some();
        let mut keep = h.len();
        for (k, i) in self.calls.iter().enumerate() {
            let Some(i) = *i else { continue };
            h.push(TOK_TOOL_CALL);
            let call = format!(
                "\n{{\"name\": \"{}\", \"arguments\": {{}}}}\n",
                self.tools[i].name
            );
            tok.encode(call.as_bytes(), &mut h)?;
            h.push(TOK_TOOL_CALL_END);
            if k == 0 && routed {
                // Fed as part of the question turn.
                keep = h.len();
            }
            h.extend_from_slice(&[
                TOK_IM_END,
                TOK_NL,
                TOK_IM_START,
                TOK_USER,
                TOK_NL,
                TOK_TOOL_RESPONSE,
            ]);
            tok.encode(b"\n(omitted)\n", &mut h)?;
            h.extend_from_slice(&[
                TOK_TOOL_RESPONSE_END,
                TOK_IM_END,
                TOK_NL,
                TOK_IM_START,
                TOK_ASSISTANT,
                TOK_NL,
            ]);
            if !think {
                h.extend_from_slice(&[TOK_THINK_START, TOK_NL2, TOK_THINK_END, TOK_NL2]);
            }
        }
        // Reasoning is not kept in the history (as in the Qwen3 template).
        let answer = match self.answer.iter().rposition(|&t| t == TOK_THINK_END) {
            Some(i) => &self.answer[i + 1..],
            None => &self.answer[..],
        };
        let skip = answer
            .iter()
            .take_while(|&&t| tok.token_bytes(t).iter().all(|&b| b == b'\n'))
            .count();
        h.extend_from_slice(&answer[skip..]);
        h.extend_from_slice(&[TOK_IM_END, TOK_NL]);

        if !self.chat.rollback_keep(&mark, keep) {
            // The conversation was cleared meanwhile; keep what is there.
            self.phase = Phase::Idle;
            return Ok(());
        }
        match self.chat.start_append(&h[keep..]) {
            Ok(_) => self.phase = Phase::Replay,
            Err(_) => {
                // Too long for the context: start over.
                self.chat.clear();
                self.phase = Phase::Idle;
            }
        }
        Ok(())
    }
}

fn empty_summary() -> Summary {
    Summary {
        stop: AgentStop::Answer,
        tool_calls: 0,
        routed: None,
        prompt_tokens: 0,
        answer_tokens: 0,
    }
}
