//! Tools the model may call while answering (agent loop).
//!
//! **Registration is the allowlist**: the model can only run tools registered
//! here, by exact name, without arguments. The shell registers its read-only
//! commands `(task)`, `(interrupt)` and `(ifconfig)` at boot. The registry is
//! sealed when the service sets up its tools, so nothing can be added later.
//! As a second line of defence, commands that change the system are refused
//! by name.
//!
//! Tool contract: a tool must only read state, return quickly (it runs on
//! the service task while helper tasks wait on other CPUs, and cannot be
//! cancelled), and must not panic: the kernel's panic handler demotes the
//! panicking task to the `Panicked` scheduler even if the unwind is caught,
//! which would leave the LLM service permanently starved.

use alloc::{string::String, vec::Vec};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use awkernel_llm::tools::ToolSpec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A tool without parameters. `run` returns its output as text.
#[derive(Clone, Copy)]
pub struct Tool {
    /// Name, description and example questions (see
    /// `awkernel_llm::tools::catalog`).
    pub spec: ToolSpec<'static>,
    pub run: fn() -> String,
}

/// Commands that must never be run by the model.
pub const FORBIDDEN: [&str; 3] = ["reboot", "shutdown", "netdump"];

/// Tool output given to the model is cut to this many bytes (the engine
/// also limits it in tokens; numbers tokenize into many tokens).
pub const MAX_OUTPUT: usize = 2048;

static TOOLS: Mutex<Vec<Tool>> = Mutex::new(Vec::new());
/// Incremented on every registration, so the service can rebuild its
/// system prompt.
static VERSION: AtomicUsize = AtomicUsize::new(0);
static SEALED: AtomicBool = AtomicBool::new(false);

/// Registers a tool. Fails for forbidden or duplicate names, and after the
/// service has set up its tools.
pub fn register(tool: Tool) -> Result<(), &'static str> {
    let name = tool.spec.name;
    if FORBIDDEN.contains(&name) {
        return Err("this command must not be called by the LLM");
    }
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err("invalid tool name");
    }
    let mut node = MCSNode::new();
    let mut tools = TOOLS.lock(&mut node);
    if SEALED.load(Ordering::Acquire) {
        return Err("tools can no longer be registered");
    }
    if tools.iter().any(|t| t.spec.name == name) {
        return Err("the tool is already registered");
    }
    tools.push(tool);
    VERSION.fetch_add(1, Ordering::Release);
    Ok(())
}

pub(crate) fn version() -> usize {
    VERSION.load(Ordering::Acquire)
}

/// Seals the registry and returns the registered tools' specifications.
pub(crate) fn seal() -> Vec<ToolSpec<'static>> {
    let mut node = MCSNode::new();
    let tools = TOOLS.lock(&mut node);
    SEALED.store(true, Ordering::Release);
    tools.iter().map(|t| t.spec).collect()
}

/// Runs the tool `name` and returns its output prepared for the model
/// (`\r\n` → `\n`, at most [`MAX_OUTPUT`] bytes). `None` if there is no such
/// tool.
pub(crate) fn run(name: &str) -> Option<String> {
    let run = {
        let mut node = MCSNode::new();
        let tools = TOOLS.lock(&mut node);
        tools.iter().find(|t| t.spec.name == name)?.run
    };
    // Called without holding the lock: tools take locks of their own.
    let mut out = run().replace("\r\n", "\n");
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    if out.len() > MAX_OUTPUT {
        let mut cut = MAX_OUTPUT;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("\n... (truncated)");
    }
    Some(out)
}
