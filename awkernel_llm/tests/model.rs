//! End-to-end checks against the llm.asm reference. They need the exported
//! model and are skipped unless `AWKERNEL_LLM_MODEL` points to it:
//!
//! ```sh
//! AWKERNEL_LLM_MODEL=.../qwen3-0.6b-q8_0.asmbin cargo test --release -p awkernel_llm
//! ```

use awkernel_llm::{
    exec::ScopedThreadExecutor,
    tokenizer::{TOK_IM_END, TOK_NL},
    transformer::Transformer,
    Backend, Chat, ChatConfig, Event, InlineExecutor, Model, StopReason,
};

const PROMPT: &str = "Explain how a CPU cache works in 3 concise sentences.";

/// `asm-llm --print-tokens` for `PROMPT` (chat template, no think).
const REF_PROMPT_TOKENS: &[u32] = &[
    151644, 872, 198, 840, 20772, 1246, 264, 13940, 6500, 4278, 304, 220, 18, 63594, 22870, 13,
    151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
];

/// The greedy continuation shared by llm.asm (Q8 activations) and the llm.asm
/// NumPy reference (f32 activations). They diverge at token 17, which is a
/// near tie, so only this prefix is a stable target.
const REF_GEN_PREFIX: &[u32] = &[
    32, 13940, 6500, 374, 264, 2613, 3311, 315, 4938, 1483, 311, 3553, 13814, 24758, 821, 323,
    11221,
];

fn model_bytes() -> Option<Vec<u8>> {
    let path = std::env::var("AWKERNEL_LLM_MODEL").ok()?;
    Some(std::fs::read(path).expect("AWKERNEL_LLM_MODEL is not readable"))
}

fn greedy(model: &Model<'_>, backend: Backend, n: usize) -> (Vec<u32>, Vec<u32>) {
    let cfg = ChatConfig {
        max_seq: 256,
        max_new_tokens: n,
        ..Default::default()
    };
    let mut chat = Chat::new(model, cfg, backend).unwrap();
    chat.start_turn(PROMPT).unwrap();
    let prompt = chat.prompt_tokens().to_vec();
    let mut gen = Vec::new();
    loop {
        match chat.step(&InlineExecutor).unwrap() {
            Event::Prefill { .. } => {}
            Event::Token(t) => gen.push(t),
            Event::Done(_) => break,
        }
    }
    (prompt, gen)
}

#[test]
fn matches_reference_scalar() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let (prompt, gen) = greedy(&model, Backend::SCALAR, REF_GEN_PREFIX.len());
    assert_eq!(prompt, REF_PROMPT_TOKENS);
    assert_eq!(gen, REF_GEN_PREFIX);
}

#[test]
fn matches_reference_detected_backend() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let (_, gen) = greedy(&model, Backend::detect(), REF_GEN_PREFIX.len());
    assert_eq!(gen, REF_GEN_PREFIX);
}

/// Splitting rows/heads over threads must not change a single bit.
#[test]
fn executor_does_not_change_results() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let run = |threads: usize| {
        let mut tf = Transformer::new(&model.config, 16, Backend::detect()).unwrap();
        let exec = ScopedThreadExecutor { threads };
        for (pos, &t) in REF_PROMPT_TOKENS[..4].iter().enumerate() {
            tf.forward(&model, t, pos, pos == 3, &exec).unwrap();
        }
        tf.logits().to_vec()
    };
    let one = run(1);
    for threads in [2, 3, 7] {
        assert!(one == run(threads), "threads = {threads}");
    }
}

/// Runs the current turn to completion; returns generated tokens and the
/// stop reason (also for prefill-only turns).
fn run_turn(chat: &mut Chat<'_>) -> (Vec<u32>, StopReason) {
    let mut gen = Vec::new();
    loop {
        match chat.step(&InlineExecutor).unwrap() {
            Event::Prefill { .. } => {}
            Event::Token(t) => gen.push(t),
            Event::Done(r) => return (gen, r),
        }
    }
}

/// A second turn must see exactly the cache llm.asm builds:
/// `prompt1 gen1 <|im_end|> \n prompt2`, even though the closing tokens are
/// fed lazily at the start of turn 2.
#[test]
fn multi_turn_cache_matches_flat_sequence() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let cfg = ChatConfig {
        max_seq: 256,
        max_new_tokens: 6,
        ..Default::default()
    };
    let be = Backend::detect();

    let mut chat = Chat::new(&model, cfg, be).unwrap();
    chat.start_turn("Hi").unwrap();
    let p1 = chat.prompt_tokens().to_vec();
    let (g1, r1) = run_turn(&mut chat);
    assert_eq!(r1, StopReason::MaxTokens);
    assert_eq!(g1.len(), 6);
    // The last generated token has not been fed yet.
    assert_eq!(chat.context_len(), p1.len() + 5);

    chat.start_turn("Thanks").unwrap();
    let p2 = chat.prompt_tokens().to_vec();
    let (g2, _) = run_turn(&mut chat);

    let mut flat = p1.clone();
    flat.extend(&g1);
    flat.extend([TOK_IM_END, TOK_NL]);
    flat.extend(&p2);
    let mut chat2 = Chat::new(&model, cfg, be).unwrap();
    chat2.start_raw(&flat).unwrap();
    let (g2_flat, _) = run_turn(&mut chat2);
    assert_eq!(g2, g2_flat);
}

#[test]
fn cancel_and_limits() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let be = Backend::detect();

    // Cancelling during prefill forgets the partial turn.
    let mut chat = Chat::new(
        &model,
        ChatConfig {
            max_seq: 128,
            ..Default::default()
        },
        be,
    )
    .unwrap();
    chat.start_turn("Hello there").unwrap();
    for _ in 0..3 {
        chat.step(&InlineExecutor).unwrap();
    }
    chat.cancel();
    assert_eq!(chat.context_len(), 0);
    assert!(chat.is_idle());

    // max_new_tokens = 0 generates nothing.
    let cfg = ChatConfig {
        max_seq: 128,
        max_new_tokens: 0,
        ..Default::default()
    };
    let mut chat = Chat::new(&model, cfg, be).unwrap();
    chat.start_turn("Hi").unwrap();
    let (g, r) = run_turn(&mut chat);
    assert!(g.is_empty());
    assert_eq!(r, StopReason::MaxTokens);

    // Bad token ids and over-long prompts are rejected up front.
    assert!(chat.start_raw(&[u32::MAX]).is_err());
    assert!(chat.start_raw(&[]).is_err());
    assert!(chat.prompt_tokens().is_empty());
    let long = "a".repeat(1 << 20);
    assert!(chat.start_turn(&long).is_err());

    // Each step runs at most one forward pass.
    let mut chat = Chat::new(
        &model,
        ChatConfig {
            max_seq: 128,
            max_new_tokens: 3,
            ..Default::default()
        },
        be,
    )
    .unwrap();
    for text in ["Hi", "Again"] {
        chat.start_turn(text).unwrap();
        loop {
            let before = chat.context_len();
            let e = chat.step(&InlineExecutor).unwrap();
            assert!(chat.context_len() <= before + 1);
            if matches!(e, Event::Done(_)) {
                break;
            }
        }
    }
}

/// Rolling back to a mark restores the cache exactly: the same question
/// asked again gives the same answer.
#[test]
fn rollback_restores_the_conversation() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let cfg = ChatConfig {
        max_seq: 256,
        max_new_tokens: 5,
        ..Default::default()
    };
    let mut chat = Chat::new(&model, cfg, Backend::detect()).unwrap();
    chat.start_turn("Hi").unwrap();
    run_turn(&mut chat);

    let mark = chat.mark();
    let len = chat.context_len();
    chat.start_turn("Name a color.").unwrap();
    let (first, _) = run_turn(&mut chat);
    assert!(chat.rollback(&mark));
    assert_eq!(chat.context_len(), len);
    chat.start_turn("Name a color.").unwrap();
    let (again, _) = run_turn(&mut chat);
    assert_eq!(first, again);

    // A mark taken before a clear is invalid.
    chat.clear();
    assert!(!chat.rollback(&mark));
}

/// The system turn stays cached across `clear()`.
#[test]
fn system_prompt_is_kept() {
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let cfg = ChatConfig {
        max_seq: 512,
        max_new_tokens: 4,
        ..Default::default()
    };
    let mut chat = Chat::new(&model, cfg, Backend::detect()).unwrap();
    chat.set_system_prompt("<|im_start|>system\nYou are terse.<|im_end|>\n")
        .unwrap();
    assert!(chat.start_system().unwrap());
    let (_, r) = run_turn(&mut chat);
    assert_eq!(r, StopReason::Appended);
    let sys_len = chat.context_len();
    assert!(sys_len > 5);

    chat.start_turn("Hi").unwrap();
    run_turn(&mut chat);
    chat.clear();
    assert_eq!(chat.context_len(), sys_len);
    // Not fed again.
    assert!(!chat.start_system().unwrap());
}

/// The agent runs a tool, answers, and compacts the history to
/// question + call + "(omitted)" + answer.
#[test]
fn agent_calls_tools_and_compacts() {
    use awkernel_llm::{
        router::Router,
        tools::catalog::{IFCONFIG, INTERRUPT, NO_TOOL_EXAMPLES, PREAMBLE, TASK},
        Agent, AgentConfig, AgentEvent, AgentStop,
    };
    let Some(data) = model_bytes() else {
        eprintln!("skipped: AWKERNEL_LLM_MODEL is not set");
        return;
    };
    let model = Model::parse(&data).unwrap();
    let be = Backend::detect();
    let chat = Chat::new(&model, ChatConfig::default(), be).unwrap();
    let mut agent = Agent::new(chat, AgentConfig::default());
    let tools = [TASK, INTERRUPT, IFCONFIG];
    let router = Router::new(
        &model,
        &tools,
        NO_TOOL_EXAMPLES,
        be,
        awkernel_llm::router::DEFAULT_LEN,
    )
    .unwrap();
    agent
        .set_tools(&tools, Some(PREAMBLE), Some(router))
        .unwrap();
    let exec = ScopedThreadExecutor { threads: 4 };

    let calls = std::cell::RefCell::new(Vec::new());
    let mut run = |name: &str| {
        calls.borrow_mut().push(name.to_string());
        (name == "ifconfig").then(|| "MAC address: 52:54:00:12:34:56".to_string())
    };
    agent.warm_up();
    while agent.step(&exec, &mut run) != AgentEvent::Idle {}
    let base = agent.chat().context_len();

    agent
        .ask("What is the MAC address of the network interface?")
        .unwrap();
    let mut answer = Vec::new();
    let summary = loop {
        match agent.step(&exec, &mut run) {
            AgentEvent::Token(t) => answer.extend_from_slice(agent.chat().token_bytes(t)),
            AgentEvent::Answered(s) => break s,
            _ => {}
        }
    };
    let during = agent.chat().context_len();
    while agent.step(&exec, &mut run) != AgentEvent::Idle {}
    let after = agent.chat().context_len();

    assert_eq!(summary.stop, AgentStop::Answer);
    assert_eq!(*calls.borrow(), ["ifconfig"]);
    assert!(String::from_utf8_lossy(&answer).contains("52:54:00:12:34:56"));
    // The compacted history is shorter than the turns with the tool output
    // but keeps question, call and answer.
    assert!(
        after > base + 20 && after <= during,
        "{base} {after} {during}"
    );

    // Cancelling forgets the question.
    agent.ask("How many tasks are there?").unwrap();
    for _ in 0..3 {
        agent.step(&exec, &mut run);
    }
    agent.cancel();
    assert_eq!(agent.chat().context_len(), after);
}
