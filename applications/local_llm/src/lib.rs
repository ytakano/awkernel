//! Local LLM chat service.
//!
//! Runs Qwen3-0.6B (`awkernel_llm`) on the model found in the bootloader's
//! ramdisk. One service task owns the model state; other code (the shell)
//! submits requests with [`submit`] and may wait with [`is_busy`] and stop a
//! generation with [`cancel`]. Answers are streamed to the console.
//!
//! The service task performs one forward pass per scheduling step and yields
//! between them, so other tasks of the same priority keep running.

#![no_std]

extern crate alloc;

mod executor;
pub mod tools;

/// Descriptions of the shell commands offered as tools.
pub use awkernel_llm::tools::catalog;

use alloc::{boxed::Box, collections::VecDeque, format, string::String};
use awkernel_async_lib::scheduler::SchedulerType;
use awkernel_lib::{
    console,
    sync::mutex::{MCSNode, Mutex},
    time::Instant,
};
use awkernel_llm::{
    router::Router, utf8::Utf8Stream, Agent, AgentConfig, AgentEvent, AgentStop, Backend, Chat,
    ChatConfig, Executor, Model,
};
use core::{
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::Duration,
};

/// Context length (KV cache: 224 KiB per token, 448 MiB in total).
pub const CONTEXT_LEN: usize = 2048;
/// Maximum number of tokens generated per answer.
pub const MAX_NEW_TOKENS: usize = 512;
/// Tool calls allowed per question (agent loop budget).
pub const MAX_TOOL_STEPS: usize = 2;

/// Upper bound for a prompt in bytes (tokenization is O(n^2)).
pub const MAX_PROMPT_BYTES: usize = 4096;

const SERVICE_NAME: &str = "[local_llm] service";

/// A request to the service task.
#[derive(Debug)]
pub enum Request {
    /// Answer a user message (continuing the conversation).
    Chat(String),
    /// Forget the conversation.
    Clear,
    /// Enable/disable Qwen3 reasoning (`<think>`) for the following turns.
    Think(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    /// The service has not started (feature disabled or not yet spawned).
    NotStarted = 0,
    Loading = 1,
    Ready = 2,
    /// No model or not enough memory; see the log.
    Unavailable = 3,
}

static STATUS: AtomicU8 = AtomicU8::new(Status::NotStarted as u8);
static BUSY: AtomicBool = AtomicBool::new(false);
static CANCEL: AtomicBool = AtomicBool::new(false);
static REQUESTS: Mutex<VecDeque<Request>> = Mutex::new(VecDeque::new());

pub fn status() -> Status {
    match STATUS.load(Ordering::Acquire) {
        1 => Status::Loading,
        2 => Status::Ready,
        3 => Status::Unavailable,
        _ => Status::NotStarted,
    }
}

/// True while a chat request is queued or being answered.
pub fn is_busy() -> bool {
    BUSY.load(Ordering::Acquire)
}

/// Stops the answer being generated (if any).
pub fn cancel() {
    if is_busy() {
        CANCEL.store(true, Ordering::Release);
    }
}

/// Queues a request.
pub fn submit(req: Request) -> Result<(), &'static str> {
    match status() {
        Status::Ready => {}
        Status::Loading => return Err("the model is still loading"),
        Status::NotStarted | Status::Unavailable => return Err("the LLM is not available"),
    }
    if let Request::Chat(text) = &req {
        if text.len() > MAX_PROMPT_BYTES {
            return Err("the prompt is too long");
        }
    }
    // Every request keeps the service busy until it has been handled, so the
    // shell prints its prompt after the request's output.
    if BUSY.swap(true, Ordering::AcqRel) {
        return Err("the previous request is still running");
    }
    // A Ctrl-C from now on applies to this request (and a stale one from the
    // previous request is dropped).
    CANCEL.store(false, Ordering::Release);
    let mut node = MCSNode::new();
    let mut queue = REQUESTS.lock(&mut node);
    queue.push_back(req);
    Ok(())
}

/// Starts the service task.
pub async fn run() {
    STATUS.store(Status::Loading as u8, Ordering::Release);
    awkernel_async_lib::spawn(
        SERVICE_NAME.into(),
        service(),
        SchedulerType::PrioritizedFIFO(0),
    )
    .await;
}

/// Marks the service unavailable if its task ends, including by a panic
/// (task panics are caught), so the shell does not wait for it forever.
struct ServiceGuard;

impl Drop for ServiceGuard {
    fn drop(&mut self) {
        STATUS.store(Status::Unavailable as u8, Ordering::Release);
        BUSY.store(false, Ordering::Release);
    }
}

async fn service() {
    let _guard = ServiceGuard;
    let Some(mut agent) = load() else {
        STATUS.store(Status::Unavailable as u8, Ordering::Release);
        return;
    };
    let (exec, pool) = executor::create();
    let set_active = |on: bool| {
        if let Some(p) = &pool {
            p.set_active(on);
        }
    };

    #[cfg(feature = "selftest")]
    {
        set_active(true);
        selftest::run(agent.chat_mut(), exec.as_ref()).await;
        set_active(false);
    }

    // Tools are registered by the shell during boot, before this task runs.
    let mut tools_version = usize::MAX;
    set_active(true);
    update_tools(&mut agent, &mut tools_version, exec.as_ref()).await;
    set_active(false);

    STATUS.store(Status::Ready as u8, Ordering::Release);
    log::info!("{SERVICE_NAME}: ready. Try (llm \"Hello\") in the shell.");

    loop {
        let req = {
            let mut node = MCSNode::new();
            let mut queue = REQUESTS.lock(&mut node);
            queue.pop_front()
        };
        let Some(req) = req else {
            // Requests come from the shell at human speed, so polling is
            // cheap and keeps the interface lock-free.
            awkernel_async_lib::sleep(Duration::from_millis(50)).await;
            continue;
        };
        match req {
            Request::Chat(text) => {
                set_active(true);
                update_tools(&mut agent, &mut tools_version, exec.as_ref()).await;
                answer(&mut agent, &text, exec.as_ref()).await;
                // The answer is complete: let the shell continue while the
                // history is compacted (one more second of work).
                BUSY.store(false, Ordering::Release);
                run_until_idle(&mut agent, exec.as_ref()).await;
                set_active(false);
                continue;
            }
            Request::Clear => {
                agent.clear();
                console::print("[llm] conversation cleared\r\n");
            }
            Request::Think(on) => {
                agent.set_think(on);
                console::print(if on {
                    "[llm] think: on\r\n"
                } else {
                    "[llm] think: off\r\n"
                });
            }
        }
        BUSY.store(false, Ordering::Release);
    }
}

fn load() -> Option<Agent<'static>> {
    let Some(data) = awkernel_lib::ramdisk::get() else {
        log::warn!("{SERVICE_NAME}: no ramdisk (build with RAMDISK=<model.asmbin>)");
        return None;
    };
    let model = match Model::parse(data) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("{SERVICE_NAME}: invalid model in the ramdisk: {e}");
            return None;
        }
    };
    // The service lives as long as the kernel.
    let model: &'static Model<'static> = Box::leak(Box::new(model));

    let backend = select_backend();
    if !backend.is_avx2() {
        log::warn!("{SERVICE_NAME}: AVX2/FMA/F16C not available; using the slow scalar kernels");
    }
    let cfg = ChatConfig {
        max_seq: CONTEXT_LEN,
        max_new_tokens: MAX_NEW_TOKENS,
        ..Default::default()
    };
    match Chat::new(model, cfg, backend) {
        Ok(chat) => {
            let c = &model.config;
            log::info!(
                "{SERVICE_NAME}: Qwen3 {} layers, dim {}, vocab {}, model {} MiB, state {} MiB, backend {}",
                c.n_layers,
                c.dim,
                c.vocab_size,
                data.len() >> 20,
                chat.transformer().memory_bytes() >> 20,
                backend.name()
            );
            let agent_cfg = AgentConfig {
                max_tool_steps: MAX_TOOL_STEPS,
                max_new_tokens: MAX_NEW_TOKENS,
                ..Default::default()
            };
            Some(Agent::new(chat, agent_cfg))
        }
        Err(e) => {
            log::warn!("{SERVICE_NAME}: failed to allocate the model state: {e}");
            None
        }
    }
}

/// AVX2 is used only if this CPU supports it *and* the kernel enabled AVX on
/// every CPU (the service task may migrate, and the preemption handler must
/// save the AVX state).
fn select_backend() -> Backend {
    let backend = Backend::detect();
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    if backend.is_avx2() && !awkernel_lib::arch::x86_64::xsave::is_avx_enabled() {
        return Backend::SCALAR;
    }
    backend
}

/// Sets up the tools (system prompt and router) if tools were registered
/// since the last call, and processes both prompts ahead of the first
/// question. Clears the conversation, which only happens at boot.
async fn update_tools(
    agent: &mut Agent<'static>,
    version: &mut usize,
    exec: &(dyn Executor + Send),
) {
    let v = tools::version();
    if v == *version {
        return;
    }
    *version = v;
    let specs = tools::seal();
    if specs.is_empty() {
        return;
    }
    let model = agent.chat().model();
    let backend = agent.chat().transformer().backend();
    let router = match Router::new(
        model,
        &specs,
        catalog::NO_TOOL_EXAMPLES,
        backend,
        awkernel_llm::router::DEFAULT_LEN,
    ) {
        Ok(r) => Some(r),
        Err(e) => {
            log::warn!("{SERVICE_NAME}: no tool router: {e}");
            None
        }
    };
    if let Err(e) = agent.set_tools(&specs, Some(catalog::PREAMBLE), router) {
        log::warn!("{SERVICE_NAME}: cannot use the tools: {e}");
        return;
    }
    let start = Instant::now();
    agent.warm_up();
    run_until_idle(agent, exec).await;
    log::info!(
        "{SERVICE_NAME}: tools {:?} ready in {:.1} s",
        specs.iter().map(|t| t.name).collect::<alloc::vec::Vec<_>>(),
        start.elapsed().as_secs_f64()
    );
}

/// Steps the agent until it is idle (background work), yielding between steps.
async fn run_until_idle(agent: &mut Agent<'static>, exec: &(dyn Executor + Send)) {
    let mut run = |name: &str| tools::run(name);
    let mut told = false;
    while agent.step(exec, &mut run) != AgentEvent::Idle {
        if !told && is_busy() {
            // A new request is waiting for this background work.
            console::print("[llm] finishing the previous answer...\r\n");
            told = true;
        }
        awkernel_async_lib::r#yield().await;
    }
}

/// Answers `question`, streaming the text to the console. The model may
/// call tools (see `awkernel_llm::agent`); each call is shown as one line.
async fn answer(agent: &mut Agent<'static>, question: &str, exec: &(dyn Executor + Send)) {
    if let Err(e) = agent.ask(question) {
        console::print(&format!("[llm] error: {e}\r\n"));
        return;
    }

    let start = Instant::now();
    let mut first: Option<Instant> = None;
    let mut utf8 = Utf8Stream::new();
    let mut out = String::new();
    let mut run = |name: &str| tools::run(name);

    let summary = loop {
        if CANCEL.swap(false, Ordering::AcqRel) {
            agent.cancel();
            break None;
        }
        match agent.step(exec, &mut run) {
            AgentEvent::Busy => {}
            AgentEvent::Token(tok) => {
                first.get_or_insert_with(Instant::now);
                out.clear();
                utf8.push(agent.chat().token_bytes(tok), &mut out);
                print_text(&out);
            }
            AgentEvent::ToolCall { name, routed } => {
                console::print(&format!(
                    "[llm] -> {name}(){}\r\n",
                    if routed { "" } else { " (by the model)" }
                ));
            }
            AgentEvent::Answered(s) => break Some(s),
            AgentEvent::Idle => {
                console::print("\r\n[llm] internal error: the agent stopped\r\n");
                return;
            }
        }
        awkernel_async_lib::r#yield().await;
    };

    out.clear();
    utf8.finish(&mut out);
    print_text(&out);

    let end = Instant::now();
    let first = first.unwrap_or(end);
    let Some(s) = summary else {
        console::print("\r\n[llm] (cancelled; the question was forgotten)\r\n");
        return;
    };
    let stop = match s.stop {
        AgentStop::Answer => String::new(),
        AgentStop::MaxTokens => String::from(" (max tokens)"),
        AgentStop::ContextFull => String::from(" (context full)"),
        AgentStop::MalformedCall => String::from(" (malformed tool call)"),
        AgentStop::Cancelled => String::from(" (cancelled)"),
        AgentStop::Error(e) => format!(" (error: {e})"),
    };
    let tools = match s.tool_calls {
        0 => String::from("no tool used"),
        1 => String::from("1 tool call"),
        n => format!("{n} tool calls"),
    };
    let prefill = first - start;
    let decode = end - first;
    console::print(&format!(
        "\r\n[llm] {tools}; first token after {:.2} s ({} prompt tok), answer {} tok {:.2} s ({:.1} tok/s), ctx {}/{}{stop}\r\n",
        prefill.as_secs_f64(),
        s.prompt_tokens,
        s.answer_tokens,
        decode.as_secs_f64(),
        rate(s.answer_tokens.saturating_sub(1), decode),
        agent.chat().context_len(),
        CONTEXT_LEN,
    ));
}

fn rate(n: usize, d: Duration) -> f64 {
    let s = d.as_secs_f64();
    if s > 0.0 {
        n as f64 / s
    } else {
        0.0
    }
}

/// Prints text, converting `\n` to `\r\n` for serial terminals.
fn print_text(s: &str) {
    let mut rest = s;
    while let Some(i) = rest.find('\n') {
        console::print(&rest[..i]);
        console::print("\r\n");
        rest = &rest[i + 1..];
    }
    if !rest.is_empty() {
        console::print(rest);
    }
}

#[cfg(feature = "selftest")]
mod selftest {
    use super::*;
    use alloc::vec::Vec;
    use awkernel_llm::Event;

    const PROMPT: &str = "Explain how a CPU cache works in 3 concise sentences.";

    /// `asm-llm --print-tokens` for `PROMPT`.
    const REF_PROMPT: &[u32] = &[
        151644, 872, 198, 840, 20772, 1246, 264, 13940, 6500, 4278, 304, 220, 18, 63594, 22870, 13,
        151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
    ];

    /// Greedy prefix shared by llm.asm and its NumPy reference.
    const REF_GEN: &[u32] = &[
        32, 13940, 6500, 374, 264, 2613, 3311, 315, 4938, 1483, 311, 3553, 13814, 24758, 821, 323,
        11221,
    ];

    pub(super) async fn run(chat: &mut Chat<'static>, exec: &(dyn Executor + Send)) {
        if let Err(e) = chat.start_turn(PROMPT) {
            log::error!("local_llm selftest: {e}");
            return;
        }
        let prompt_ok = chat.prompt_tokens() == REF_PROMPT;
        let start = Instant::now();
        let mut gen = Vec::new();
        while gen.len() < REF_GEN.len() {
            match chat.step(exec) {
                Ok(Event::Token(t)) => gen.push(t),
                Ok(Event::Prefill { .. }) => {}
                Ok(Event::Done(_)) | Err(_) => break,
            }
            awkernel_async_lib::r#yield().await;
        }
        let elapsed = start.elapsed();
        chat.cancel();
        chat.clear();

        let gen_ok = gen == REF_GEN;
        log::info!(
            "local_llm selftest: prompt_tokens={} gen_tokens={} ({} tokens in {:.2} s) {}",
            if prompt_ok { "match" } else { "MISMATCH" },
            if gen_ok { "match" } else { "MISMATCH" },
            REF_PROMPT.len() + gen.len(),
            elapsed.as_secs_f64(),
            if prompt_ok && gen_ok { "PASS" } else { "FAIL" }
        );
        if !gen_ok {
            log::info!("local_llm selftest: got {gen:?}");
        }
    }
}
