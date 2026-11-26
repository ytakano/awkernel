//! Evaluates tool use (host only), with the same `Agent`, tool catalog and
//! router as the kernel.
//!
//! Each system question is asked against two recorded system states
//! (fixtures A and B with different values), so an answer only counts if it
//! contains the facts of the state the tool returned. Facts are counted only
//! when the expected tool was called. Questions with even index form the
//! "dev" split (used for tuning), odd ones the "test" split.
//!
//! ```sh
//! cargo run --release -p awkernel_llm --features std --example tool_eval -- \
//!     -m qwen3-0.6b-q8_0.asmbin [-t 4] [--no-router] [-v]
//! ```

use awkernel_llm::{
    exec::ScopedThreadExecutor,
    router::Router,
    tools::catalog::{IFCONFIG, INTERRUPT, NO_TOOL_EXAMPLES, PREAMBLE, TASK},
    Agent, AgentConfig, AgentEvent, AgentStop, Backend, Chat, ChatConfig, Executor, InlineExecutor,
    Model,
};
use std::time::Instant;

/// Tool outputs for system states A and B, made with the same formatting
/// functions as the kernel (`tools::report`).
fn tool_output(name: &str, b: bool) -> Option<String> {
    use awkernel_llm::tools::report::*;
    let row = |id, name, state, preemptions| TaskRow {
        id,
        name,
        state,
        preemptions,
    };
    Some(match name {
        "task" if !b => tasks(&TaskReport {
            uptime_us: 5_123_456,
            total_preemptions: 0,
            cpus: &[
                (0, None),
                (1, Some(11)),
                (2, Some(10)),
                (3, Some(12)),
                (4, Some(13)),
            ],
            tasks: &[
                row(2, "[Awkernel] buffered logger service", "Waiting", 0),
                row(3, "[Awkernel] network service", "Waiting", 0),
                row(4, "[Awkernel] display service", "Waiting", 0),
                row(5, "[Awkernel] TCP garbage collector", "Waiting", 0),
                row(6, "[Awkernel] Network interface poller", "Waiting", 0),
                row(
                    7,
                    "[Awkernel] network service:igb-0000:00:02.0: IRQ = 64",
                    "Waiting",
                    0,
                ),
                row(
                    8,
                    "[Awkernel] network service:igb-0000:00:02.0: tick = 200 [msec]",
                    "Waiting",
                    0,
                ),
                row(9, "[Awkernel] shell", "Waiting", 0),
                row(10, "[local_llm] service", "Running", 0),
                row(11, "[local_llm] helper 0", "Running", 0),
                row(12, "[local_llm] helper 1", "Running", 0),
                row(13, "[local_llm] helper 2", "Running", 0),
            ],
        }),
        "task" => tasks(&TaskReport {
            uptime_us: 7_654_321,
            total_preemptions: 12,
            cpus: &[(0, None), (1, Some(14)), (2, Some(10)), (3, Some(9))],
            tasks: &[
                row(2, "[Awkernel] buffered logger service", "Waiting", 0),
                row(3, "[Awkernel] network service", "Waiting", 0),
                row(4, "[Awkernel] display service", "Waiting", 0),
                row(5, "[Awkernel] TCP garbage collector", "Waiting", 2),
                row(6, "[Awkernel] Network interface poller", "Waiting", 0),
                row(
                    7,
                    "[Awkernel] network service:igb-0000:00:02.0: IRQ = 65",
                    "Waiting",
                    0,
                ),
                row(
                    8,
                    "[Awkernel] network service:igb-0000:00:02.0: tick = 200 [msec]",
                    "Waiting",
                    0,
                ),
                row(9, "[Awkernel] shell", "Running", 3),
                row(10, "[local_llm] service", "Running", 0),
                row(11, "[local_llm] helper 0", "Waiting", 0),
                row(12, "[local_llm] helper 1", "Waiting", 0),
                row(13, "[local_llm] helper 2", "Waiting", 0),
                row(14, "[app] sensor driver", "Running", 7),
                row(15, "[app] logger", "Waiting", 0),
            ],
        }),
        "interrupt" => {
            let irq = |irq, name| IrqRow { irq, name };
            if b {
                interrupts(&[
                    irq(36, "serial port (awkernel_shell)"),
                    irq(65, "igb-0000:00:02.0"),
                    irq(66, "virtio-net-0000:00:03.0"),
                    irq(253, "local timer"),
                    irq(255, "preemption"),
                ])
            } else {
                interrupts(&[
                    irq(36, "serial port (awkernel_shell)"),
                    irq(64, "igb-0000:00:02.0"),
                    irq(253, "local timer"),
                    irq(255, "preemption"),
                ])
            }
        }
        "ifconfig" => {
            let addr = [String::from("10.0.2.15/24")];
            netifs(&[if b {
                NetIf {
                    name: "igb-0000:00:02.0",
                    ipv4: &addr,
                    gateway: Some("10.0.2.2"),
                    mac: [0x52, 0x54, 0x00, 0xab, 0xcd, 0xef],
                    link: Link::Down,
                    speed_mbps: 100,
                    irqs: &[65],
                    poll_mode: false,
                }
            } else {
                NetIf {
                    name: "igb-0000:00:02.0",
                    ipv4: &[],
                    gateway: None,
                    mac: [0x52, 0x54, 0x00, 0x12, 0x34, 0x56],
                    link: Link::UpFullDuplex,
                    speed_mbps: 1000,
                    irqs: &[64],
                    poll_mode: false,
                }
            }])
        }
        _ => return None,
    })
}

/// Facts: every group must match (any string of a group).
type Facts = &'static [&'static [&'static str]];

struct Case {
    q: &'static str,
    tool: Option<&'static str>,
    a: Facts,
    b: Facts,
}

const fn sys(q: &'static str, tool: &'static str, a: Facts, b: Facts) -> Case {
    Case {
        q,
        tool: Some(tool),
        a,
        b,
    }
}

const fn gen(q: &'static str, f: Facts) -> Case {
    Case {
        q,
        tool: None,
        a: f,
        b: &[],
    }
}

const CASES: &[Case] = &[
    sys(
        "Which tasks are running on this system?",
        "task",
        &[&["local_llm"]],
        &[&["sensor"]],
    ),
    sys("How many tasks are there?", "task", &[&["12"]], &[&["14"]]),
    sys(
        "Which task is running on CPU 1?",
        "task",
        &[&["helper 0", "11"]],
        &[&["sensor", "14"]],
    ),
    sys(
        "Is the shell task running or waiting?",
        "task",
        &[&["aiting"]],
        &[&["unning"]],
    ),
    sys(
        "What is the uptime?",
        "task",
        &[&["5.1", "5123456"]],
        &[&["7.6", "7654321"]],
    ),
    sys("Which task has ID 7?", "task", &[&["igb"]], &[&["igb"]]),
    sys(
        "実行中のタスクを教えてください。",
        "task",
        &[&["local_llm"]],
        &[&["sensor"]],
    ),
    sys(
        "CPU 1 では何が動いていますか？",
        "task",
        &[&["helper 0", "11"]],
        &[&["sensor", "14"]],
    ),
    sys("タスクはいくつありますか？", "task", &[&["12"]], &[&["14"]]),
    sys(
        "What state is the TCP garbage collector in?",
        "task",
        &[&["aiting"]],
        &[&["aiting"]],
    ),
    sys(
        "Show me what each CPU is doing.",
        "task",
        &[&["helper"]],
        &[&["sensor"]],
    ),
    sys(
        "How many preemptions happened in total?",
        "task",
        &[&["0", "zero", "no "]],
        &[&["12"]],
    ),
    sys(
        "Which IRQ does the network card igb use?",
        "interrupt",
        &[&["64"]],
        &[&["65"]],
    ),
    sys(
        "List the interrupt handlers.",
        "interrupt",
        &[&["timer"]],
        &[&["virtio"]],
    ),
    sys(
        "How many interrupt handlers are registered?",
        "interrupt",
        &[&["4", "four"]],
        &[&["5", "five"]],
    ),
    sys(
        "What is IRQ 253 used for?",
        "interrupt",
        &[&["timer"]],
        &[&["timer"]],
    ),
    sys(
        "Which interrupt number is used for preemption?",
        "interrupt",
        &[&["255"]],
        &[&["255"]],
    ),
    sys(
        "割り込みハンドラはいくつ登録されていますか？",
        "interrupt",
        &[&["4", "４", "四"]],
        &[&["5", "５", "五"]],
    ),
    sys(
        "IRQ 36 は何に使われていますか？",
        "interrupt",
        &[&["serial", "シリアル"]],
        &[&["serial", "シリアル"]],
    ),
    sys(
        "Is there an interrupt handler for virtio?",
        "interrupt",
        &[&["No", "no", "not"]],
        &[&["Yes", "yes", "66"]],
    ),
    sys(
        "What is the MAC address of the network interface?",
        "ifconfig",
        &[&["52:54:00:12:34:56"]],
        &[&["52:54:00:ab:cd:ef"]],
    ),
    sys(
        "Is the network link up?",
        "ifconfig",
        &[&["up", "Up"]],
        &[&["down", "Down"]],
    ),
    sys(
        "What is the link speed?",
        "ifconfig",
        &[&["1000"]],
        &[&["100 Mbps", "100Mbps"]],
    ),
    sys(
        "What is the IPv4 address of this machine?",
        "ifconfig",
        &[&["not", "no ", "No ", "none", "None", "empty"]],
        &[&["10.0.2.15"]],
    ),
    sys(
        "What is the name of the network interface?",
        "ifconfig",
        &[&["igb-0000:00:02.0"]],
        &[&["igb-0000:00:02.0"]],
    ),
    sys(
        "ネットワークインターフェースの MAC アドレスは？",
        "ifconfig",
        &[&["52:54:00:12:34:56"]],
        &[&["52:54:00:ab:cd:ef"]],
    ),
    sys(
        "リンク速度はいくつですか？",
        "ifconfig",
        &[&["1000"]],
        &[&["100 Mbps", "100Mbps", "100 M"]],
    ),
    sys(
        "What is the default gateway?",
        "ifconfig",
        &[&["None", "none", "no ", "not"]],
        &[&["10.0.2.2"]],
    ),
    sys(
        "Is the network interface in poll mode?",
        "ifconfig",
        &[&["No", "no", "false", "not"]],
        &[&["No", "no", "false", "not"]],
    ),
    sys(
        "Is the network cable connected?",
        "ifconfig",
        &[&["Yes", "yes", "up", "Up", "connected"]],
        &[&["No", "no", "not", "down", "Down"]],
    ),
    gen("What is 17 times 3?", &[&["51"]]),
    gen("What is the capital of France?", &[&["Paris"]]),
    gen("What is an IRQ?", &[&["nterrupt"]]),
    gen(
        "What does a network interface do?",
        &[&["network", "Network"]],
    ),
    gen(
        "Explain what a task scheduler is in one sentence.",
        &[&["schedul", "Schedul"]],
    ),
    gen("日本の首都はどこですか？", &[&["東京"]]),
    gen(
        "What is a MAC address?",
        &[&["ardware", "dentifier", "unique"]],
    ),
    gen("Write a haiku about the sea.", &[]),
    gen("What is 2 to the power of 10?", &[&["1024"]]),
    gen("Translate 'good morning' to Japanese.", &[&["おはよう"]]),
    gen("What is Rust?", &[&["programming", "language"]]),
    gen("How do interrupts differ from polling?", &[&["nterrupt"]]),
];

#[derive(Default, Clone, Copy)]
struct Score {
    cases: usize,
    sys: usize,
    /// System questions whose first tool call was the expected tool.
    right_tool: usize,
    /// System questions answered without any tool.
    no_call: usize,
    /// General questions that triggered a tool call.
    false_call: usize,
    gen: usize,
    /// Facts correct (system: and the expected tool was called).
    facts: usize,
    seconds: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut threads = 1;
    let mut use_router = true;
    let mut verbose = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" => path = args.next(),
            "-t" => threads = args.next().unwrap().parse().unwrap(),
            "--no-router" => use_router = false,
            "-v" => verbose = true,
            "--dump-tools" => {
                for name in ["task", "interrupt", "ifconfig"] {
                    for b in [false, true] {
                        println!(
                            "== {name} ({})\n{}",
                            if b { 'B' } else { 'A' },
                            tool_output(name, b).unwrap()
                        );
                    }
                }
                return;
            }
            _ => panic!("unknown argument {a}"),
        }
    }
    let data = std::fs::read(path.expect("-m <model>")).unwrap();
    let model = Model::parse(&data).unwrap();
    let backend = Backend::detect();
    let chat = Chat::new(&model, ChatConfig::default(), backend).unwrap();
    let mut agent = Agent::new(chat, AgentConfig::default());
    let tools = [TASK, INTERRUPT, IFCONFIG];
    let router = use_router.then(|| {
        Router::new(
            &model,
            &tools,
            NO_TOOL_EXAMPLES,
            backend,
            awkernel_llm::router::DEFAULT_LEN,
        )
        .unwrap()
    });
    if let Some(r) = &router {
        println!("router prompt: {} tokens", r.prompt_len());
    }
    agent.set_tools(&tools, Some(PREAMBLE), router).unwrap();

    let threaded = ScopedThreadExecutor { threads };
    let exec: &dyn Executor = if threads > 1 {
        &threaded
    } else {
        &InlineExecutor
    };
    let mut none = |_: &str| None;
    agent.warm_up();
    while agent.step(exec, &mut none) != AgentEvent::Idle {}
    println!("system prompt: {} tokens", agent.chat().context_len());

    let mut scores = [Score::default(); 2]; // dev, test
    for (i, case) in CASES.iter().enumerate() {
        let fixtures: &[(char, Facts)] = if case.tool.is_some() {
            &[('A', case.a), ('B', case.b)]
        } else {
            &[('-', case.a)]
        };
        for &(fx, facts) in fixtures {
            let t0 = Instant::now();
            let mut run = |name: &str| tool_output(name, fx == 'B');
            agent.clear();
            agent.ask(case.q).unwrap();
            let mut answer = Vec::new();
            let mut calls = Vec::new();
            let summary = loop {
                match agent.step(exec, &mut run) {
                    AgentEvent::Token(t) => answer.extend_from_slice(agent.chat().token_bytes(t)),
                    AgentEvent::ToolCall { name, routed } => {
                        calls.push(if routed { format!("{name}*") } else { name })
                    }
                    AgentEvent::Answered(s) => break s,
                    _ => {}
                }
            };
            while agent.step(exec, &mut run) != AgentEvent::Idle {}
            let answer = String::from_utf8_lossy(&answer).trim().to_string();

            let first = calls.first().map(|c| c.trim_end_matches('*'));
            let right_tool = case.tool.is_some() && first == case.tool;
            let facts_ok = facts.iter().all(|g| g.iter().any(|f| answer.contains(f)));
            let s = &mut scores[i % 2];
            s.cases += 1;
            s.seconds += t0.elapsed().as_secs_f64();
            match case.tool {
                Some(_) => {
                    s.sys += 1;
                    s.right_tool += right_tool as usize;
                    s.no_call += calls.is_empty() as usize;
                    s.facts += (right_tool && facts_ok) as usize;
                }
                None => {
                    s.gen += 1;
                    s.false_call += (!calls.is_empty()) as usize;
                    s.facts += facts_ok as usize;
                }
            }
            let mark = |b: bool| if b { "ok" } else { "NG" };
            let good = facts_ok && (case.tool.is_none() || right_tool);
            let stop = if summary.stop == AgentStop::Answer {
                String::new()
            } else {
                format!(" {:?}", summary.stop)
            };
            println!(
                "[{fx}] tool:{} facts:{} {q}  calls={calls:?}{stop}",
                if case.tool.is_some() {
                    mark(right_tool)
                } else {
                    mark(calls.is_empty())
                },
                mark(good),
                q = case.q,
            );
            if verbose || !good {
                let short: String = answer.replace('\n', " ").chars().take(200).collect();
                println!("      {short}");
            }
        }
    }

    for (name, s) in [("dev", scores[0]), ("test", scores[1])] {
        println!(
            "{name}: router={use_router} right tool {}/{} (no call {}), false calls {}/{}, facts {}/{}, {:.1} s/case",
            s.right_tool,
            s.sys,
            s.no_call,
            s.false_call,
            s.gen,
            s.facts,
            s.cases,
            s.seconds / s.cases as f64
        );
    }
}
