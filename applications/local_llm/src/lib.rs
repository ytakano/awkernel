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

use alloc::{boxed::Box, collections::VecDeque, format, string::String};
use awkernel_async_lib::scheduler::SchedulerType;
use awkernel_lib::{
    console,
    sync::mutex::{MCSNode, Mutex},
    time::Instant,
};
use awkernel_llm::{
    utf8::Utf8Stream, Backend, Chat, ChatConfig, Event, Executor, Model, StopReason,
};
use core::{
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::Duration,
};

/// Context length (KV cache: 224 KiB per token, 448 MiB in total).
pub const CONTEXT_LEN: usize = 2048;
/// Maximum number of tokens generated per answer.
pub const MAX_NEW_TOKENS: usize = 512;
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
    let Some(chat) = load() else {
        STATUS.store(Status::Unavailable as u8, Ordering::Release);
        return;
    };
    let mut chat = chat;
    let (exec, pool) = executor::create();
    let set_active = |on: bool| {
        if let Some(p) = &pool {
            p.set_active(on);
        }
    };

    #[cfg(feature = "selftest")]
    {
        set_active(true);
        selftest::run(&mut chat, exec.as_ref()).await;
        set_active(false);
    }

    STATUS.store(Status::Ready as u8, Ordering::Release);
    log::info!("{SERVICE_NAME}: ready. Try (llm \"Hello\") in the shell.");

    loop {
        let req = {
            let mut node = MCSNode::new();
            let mut queue = REQUESTS.lock(&mut node);
            queue.pop_front()
        };
        match req {
            // Requests come from the shell at human speed, so polling is
            // cheap and keeps the interface lock-free.
            None => {
                awkernel_async_lib::sleep(Duration::from_millis(50)).await;
            }
            Some(req) => {
                match req {
                    Request::Chat(text) => {
                        set_active(true);
                        answer(&mut chat, &text, exec.as_ref()).await;
                        set_active(false);
                    }
                    Request::Clear => {
                        chat.clear();
                        console::print("[llm] conversation cleared\r\n");
                    }
                    Request::Think(on) => {
                        chat.set_think(on);
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
    }
}

fn load() -> Option<Chat<'static>> {
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
            Some(chat)
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

/// Generates one answer, streaming it to the console.
async fn answer(chat: &mut Chat<'static>, prompt: &str, exec: &(dyn Executor + Send)) {
    match chat.start_turn(prompt) {
        Ok(true) => console::print("[llm] context full; the conversation was reset\r\n"),
        Ok(false) => {}
        Err(e) => {
            console::print(&format!("[llm] error: {e}\r\n"));
            return;
        }
    }
    // Prompt tokens actually processed (fewer if cancelled during prefill;
    // more if tokens carried over from the previous turn were fed first).
    let mut n_prompt = 0;

    let start = Instant::now();
    let mut first: Option<Instant> = None;
    let mut utf8 = Utf8Stream::new();
    let mut out = String::new();
    let mut stop = None;

    loop {
        if CANCEL.swap(false, Ordering::AcqRel) {
            chat.cancel();
            break;
        }
        match chat.step(exec) {
            Ok(Event::Prefill { .. }) => n_prompt += 1,
            Ok(Event::Token(tok)) => {
                if first.is_none() {
                    first = Some(Instant::now());
                    n_prompt += 1; // the last prompt token
                }
                out.clear();
                utf8.push(chat.token_bytes(tok), &mut out);
                print_text(&out);
            }
            Ok(Event::Done(reason)) => {
                if first.is_none() {
                    // The turn ended right after the last prompt token.
                    first = Some(Instant::now());
                    n_prompt += 1;
                }
                stop = Some(reason);
                break;
            }
            Err(e) => {
                console::print(&format!("\r\n[llm] error: {e}\r\n"));
                break;
            }
        }
        awkernel_async_lib::r#yield().await;
    }

    out.clear();
    utf8.finish(&mut out);
    print_text(&out);

    let end = Instant::now();
    let prefill = first.unwrap_or(end) - start;
    let decode = first.map(|f| end - f).unwrap_or_default();
    let n_gen = chat.generated();
    let reason = match stop {
        Some(StopReason::Eos) => "",
        Some(StopReason::MaxTokens) => " (max tokens)",
        Some(StopReason::ContextFull) => " (context full)",
        None => " (cancelled)",
    };
    console::print(&format!(
        "\r\n[llm] prompt {n_prompt} tok {:.2} s ({:.1} tok/s), answer {n_gen} tok {:.2} s ({:.1} tok/s), ctx {}/{}{reason}\r\n",
        prefill.as_secs_f64(),
        rate(n_prompt, prefill),
        decode.as_secs_f64(),
        // The first answer token comes out of the prefill.
        rate(n_gen.saturating_sub(1), decode),
        chat.context_len(),
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
