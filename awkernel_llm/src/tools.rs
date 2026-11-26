//! Tool calling in the Qwen3 chat format.
//!
//! The tools are described in the system turn ([`system_prompt`]). The model
//! answers a question that needs one with
//!
//! ```text
//! <tool_call>
//! {"name": "task", "arguments": {}}
//! </tool_call>
//! ```
//!
//! at which point [`crate::Chat`] stops with
//! [`crate::StopReason::ToolCall`]. The caller extracts the name with
//! [`parse_tool_call`], runs the tool, and continues with
//! [`crate::Chat::start_tool_response`].
//!
//! Only tools without parameters are supported.

use alloc::string::String;
use core::fmt::Write;

/// A tool without parameters.
#[derive(Debug, Clone, Copy)]
pub struct ToolSpec<'a> {
    pub name: &'a str,
    pub description: &'a str,
    /// Example questions that need this tool (used by the router).
    pub examples: &'a [&'a str],
}

/// The system turn describing `tools`, exactly as the Qwen3 chat template
/// renders it (`<|im_start|>system ... <|im_end|>\n`). `preamble` is an
/// optional system message placed before the tool description.
///
/// Encode it with [`crate::tokenizer::Tokenizer::encode_with_specials`]
/// (done by [`crate::Chat::set_system_prompt`]).
pub fn system_prompt(tools: &[ToolSpec<'_>], preamble: Option<&str>) -> String {
    let mut s = String::from("<|im_start|>system\n");
    if let Some(p) = preamble {
        s.push_str(p);
        s.push_str("\n\n");
    }
    s.push_str(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for t in tools {
        // `tool | tojson` in the template (Python json.dumps separators).
        let _ = write!(
            s,
            "\n{{\"type\": \"function\", \"function\": {{\"name\": \"{}\", \"description\": \"{}\", \
             \"parameters\": {{\"type\": \"object\", \"properties\": {{}}, \"required\": []}}}}}}",
            Json(t.name),
            Json(t.description)
        );
    }
    s.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
    );
    s
}

/// A string escaped as the contents of a JSON string literal.
struct Json<'a>(&'a str);

impl core::fmt::Display for Json<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for c in self.0.chars() {
            match c {
                '"' => f.write_str("\\\"")?,
                '\\' => f.write_str("\\\\")?,
                '\n' => f.write_str("\\n")?,
                '\r' => f.write_str("\\r")?,
                '\t' => f.write_str("\\t")?,
                c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
                c => f.write_char(c)?,
            }
        }
        Ok(())
    }
}

/// Extracts the function name from the text of a tool call (what the model
/// wrote between `<tool_call>` and `</tool_call>`), e.g.
/// `{"name": "task", "arguments": {}}`. Returns `None` unless the text is an
/// object with a `"name"` string. Escapes in the name are not supported.
pub fn parse_tool_call(text: &str) -> Option<&str> {
    let s = text.trim();
    if !s.starts_with('{') {
        return None;
    }
    let mut rest = &s[1..];
    // Find the key "name" at the top level of the object. Keys and values
    // other than `name` are skipped as strings or balanced brackets.
    loop {
        rest = rest.trim_start();
        let (key, after) = string_literal(rest)?;
        rest = after.trim_start().strip_prefix(':')?.trim_start();
        if key == "name" {
            let (name, _) = string_literal(rest)?;
            return Some(name);
        }
        rest = skip_value(rest)?.trim_start();
        rest = rest.strip_prefix(',')?;
    }
}

/// `"..."` at the start of `s`: (contents, rest). No escapes.
fn string_literal(s: &str) -> Option<(&str, &str)> {
    let s = s.strip_prefix('"')?;
    let end = s.find(['"', '\\'])?;
    if s.as_bytes()[end] == b'\\' {
        return None;
    }
    Some((&s[..end], &s[end + 1..]))
}

/// Skips one JSON value; returns the rest.
fn skip_value(s: &str) -> Option<&str> {
    if s.starts_with('"') {
        let mut escaped = false;
        for (i, c) in s.char_indices().skip(1) {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => return Some(&s[i + 1..]),
                _ => {}
            }
        }
        return None;
    }
    if s.starts_with(['{', '[']) {
        let mut depth = 0usize;
        let mut in_str = false;
        let mut escaped = false;
        for (i, c) in s.char_indices() {
            if in_str {
                match c {
                    _ if escaped => escaped = false,
                    '\\' => escaped = true,
                    '"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&s[i + 1..]);
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    // Number, true, false, null.
    let end = s.find([',', '}']).unwrap_or(s.len());
    (end > 0).then(|| &s[end..])
}

/// Descriptions of the Awkernel shell commands offered as tools. Kept here
/// so that the kernel and the host evaluation (`examples/tool_eval.rs`) use
/// the same text.
pub mod catalog {
    use super::ToolSpec;

    pub const TASK: ToolSpec<'static> = ToolSpec {
        name: "task",
        description: "Show the tasks (processes and services) of Awkernel: ID, state \
                      (Running/Waiting), number of preemptions, last executed time and name of \
                      each task, the task running on each CPU, the uptime and the total number \
                      of preemptions. Use it for any question about the tasks, services, CPUs, \
                      uptime or preemptions of this system.",
        examples: &["What is running on CPU 3?", "Show me the list of services."],
    };

    pub const INTERRUPT: ToolSpec<'static> = ToolSpec {
        name: "interrupt",
        description: "Show the interrupt handlers (IRQs) registered in Awkernel: IRQ number and \
                      name of each handler. Use it for any question about interrupts or IRQs.",
        examples: &[
            "Which interrupt number does the serial port use?",
            "割り込みハンドラの一覧は？",
        ],
    };

    pub const IFCONFIG: ToolSpec<'static> = ToolSpec {
        name: "ifconfig",
        description: "Show the network interfaces of Awkernel: name, IPv4 address, gateway, MAC \
                      address, link status (up/down, i.e. whether the cable is connected) and \
                      speed, IRQs and poll mode. Use it for any question about the network of \
                      this system.",
        examples: &[
            "What is the gateway address?",
            "ネットワークのリンクは繋がっている？",
        ],
    };

    /// Questions that need no tool (router examples).
    pub const NO_TOOL_EXAMPLES: &[&str] = &[
        "What is an interrupt?",
        "What is 12 times 7?",
        "日本の首都はどこ？",
    ];

    /// System message before the tool descriptions.
    pub const PREAMBLE: &str = "You are an assistant running inside the Awkernel operating \
                                system. Use the tools to answer questions about this system.";
}

/// Tool outputs written for a small model: facts that need counting or
/// matching (how many tasks, which IRQ belongs to the network card, whether
/// the link is up) are computed here and stated in a summary first, so the
/// model only has to repeat them. The kernel fills in the data; the host
/// evaluation uses the same functions.
pub mod report {
    use alloc::{format, string::String, vec::Vec};
    use core::fmt::Write;

    pub struct TaskRow<'a> {
        pub id: u32,
        pub name: &'a str,
        /// `Running`, `Waiting`, ...
        pub state: &'a str,
        pub preemptions: u64,
    }

    pub struct TaskReport<'a> {
        pub uptime_us: u64,
        pub total_preemptions: u64,
        /// `(cpu, running task id)` for every CPU.
        pub cpus: &'a [(usize, Option<u32>)],
        pub tasks: &'a [TaskRow<'a>],
    }

    /// Output of the `task` tool.
    pub fn tasks(r: &TaskReport<'_>) -> String {
        let name_of = |id: u32| r.tasks.iter().find(|t| t.id == id).map_or("?", |t| t.name);
        let mut states: Vec<(&str, usize)> = Vec::new();
        for t in r.tasks {
            match states.iter_mut().find(|(s, _)| *s == t.state) {
                Some((_, n)) => *n += 1,
                None => states.push((t.state, 1)),
            }
        }
        let mut s = String::from("Summary:\n");
        let _ = write!(s, "- There are {} tasks", r.tasks.len());
        for (i, (state, n)) in states.iter().enumerate() {
            let _ = write!(
                s,
                "{}{n} {}",
                if i == 0 { ": " } else { ", " },
                state.to_ascii_lowercase()
            );
        }
        s.push_str(".\n- Running tasks: ");
        let running: Vec<String> = r
            .tasks
            .iter()
            .filter(|t| t.state == "Running")
            .map(|t| format!("{} \"{}\"", t.id, t.name))
            .collect();
        s.push_str(if running.is_empty() { "none" } else { "" });
        s.push_str(&running.join(", "));
        s.push_str(".\n");
        for (cpu, task) in r.cpus {
            match task {
                Some(id) => {
                    let _ = writeln!(s, "- CPU {cpu} is running task {id} \"{}\".", name_of(*id));
                }
                None => {
                    let _ = writeln!(s, "- CPU {cpu} is idle.");
                }
            }
        }
        let _ = writeln!(
            s,
            "- Uptime: {}.{} seconds ({} us).\n- Total preemptions: {}.",
            r.uptime_us / 1_000_000,
            r.uptime_us / 100_000 % 10,
            r.uptime_us,
            r.total_preemptions
        );
        s.push_str("All tasks (ID \"name\": state, preemptions):\n");
        for t in r.tasks {
            let _ = writeln!(
                s,
                "- {} \"{}\": {}, {} preemptions",
                t.id, t.name, t.state, t.preemptions
            );
        }
        s
    }

    pub struct IrqRow<'a> {
        pub irq: u16,
        pub name: &'a str,
    }

    /// What kind of device a handler name refers to.
    fn irq_kind(name: &str) -> Option<&'static str> {
        const NICS: [&str; 6] = ["igb", "ixgbe", "e1000", "virtio-net", "genet", "rtl8"];
        if NICS.iter().any(|n| name.contains(n)) {
            Some("network interface card")
        } else if name.contains("serial") || name.contains("uart") {
            Some("serial port")
        } else if name.contains("timer") {
            Some("timer")
        } else if name.contains("preemption") {
            Some("task preemption by the scheduler")
        } else {
            None
        }
    }

    /// Output of the `interrupt` tool.
    pub fn interrupts(rows: &[IrqRow<'_>]) -> String {
        let mut s = format!("Summary:\n- There are {} interrupt handlers.\n", rows.len());
        for r in rows {
            let _ = write!(s, "- IRQ {} is \"{}\"", r.irq, r.name);
            if let Some(k) = irq_kind(r.name) {
                let _ = write!(s, " ({k})");
            }
            s.push_str(".\n");
        }
        s
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Link {
        Up,
        UpFullDuplex,
        UpHalfDuplex,
        Down,
        Unknown,
    }

    pub struct NetIf<'a> {
        pub name: &'a str,
        /// `"10.0.2.15/24"`, ...
        pub ipv4: &'a [String],
        pub gateway: Option<&'a str>,
        pub mac: [u8; 6],
        pub link: Link,
        pub speed_mbps: u64,
        pub irqs: &'a [u16],
        pub poll_mode: bool,
    }

    /// Output of the `ifconfig` tool.
    pub fn netifs(ifs: &[NetIf<'_>]) -> String {
        let mut s = format!(
            "Summary:\n- There {} {} network interface{}.\n",
            if ifs.len() == 1 { "is" } else { "are" },
            ifs.len(),
            if ifs.len() == 1 { "" } else { "s" }
        );
        for i in ifs {
            let m = i.mac;
            let _ = write!(
                s,
                "Interface \"{}\":\n- MAC address: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}\n",
                i.name, m[0], m[1], m[2], m[3], m[4], m[5]
            );
            if i.ipv4.is_empty() {
                s.push_str("- IPv4 address: none (no address is assigned)\n");
            } else {
                let _ = writeln!(s, "- IPv4 address: {}", i.ipv4.join(", "));
            }
            let _ = writeln!(s, "- Default gateway: {}", i.gateway.unwrap_or("none"));
            let link = match i.link {
                Link::Up => "up (the cable is connected)",
                Link::UpFullDuplex => "up (the cable is connected), full duplex",
                Link::UpHalfDuplex => "up (the cable is connected), half duplex",
                Link::Down => "down (the cable is not connected)",
                Link::Unknown => "unknown",
            };
            let _ = writeln!(s, "- Link: {link}, speed {} Mbps", i.speed_mbps);
            let irqs: Vec<String> = i.irqs.iter().map(|q| format!("{q}")).collect();
            let _ = writeln!(s, "- IRQs used by this interface: {}", irqs.join(", "));
            let _ = writeln!(s, "- Poll mode: {}", if i.poll_mode { "yes" } else { "no" });
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_calls() {
        assert_eq!(
            parse_tool_call("\n{\"name\": \"task\", \"arguments\": {}}\n"),
            Some("task")
        );
        assert_eq!(
            parse_tool_call(
                "{\"arguments\": {\"a\": [1, {\"b\": \"}\"}]}, \"name\": \"ifconfig\"}"
            ),
            Some("ifconfig")
        );
        assert_eq!(
            parse_tool_call("{\"x\": 1, \"name\":\"interrupt\"}"),
            Some("interrupt")
        );
        assert_eq!(parse_tool_call("task"), None);
        assert_eq!(parse_tool_call("{\"arguments\": {}}"), None);
        assert_eq!(parse_tool_call("{\"name\": 3}"), None);
        assert_eq!(parse_tool_call("{\"name\": \"a\\\"b\"}"), None);
        assert_eq!(parse_tool_call("{\"name"), None);
    }

    #[test]
    fn reports_state_counts() {
        use report::*;
        let tasks = [
            TaskRow {
                id: 2,
                name: "logger",
                state: "Waiting",
                preemptions: 0,
            },
            TaskRow {
                id: 3,
                name: "shell",
                state: "Running",
                preemptions: 1,
            },
            TaskRow {
                id: 4,
                name: "net",
                state: "Waiting",
                preemptions: 0,
            },
        ];
        let t = report::tasks(&TaskReport {
            uptime_us: 5_123_456,
            total_preemptions: 1,
            cpus: &[(0, None), (1, Some(3))],
            tasks: &tasks,
        });
        assert!(
            t.contains("There are 3 tasks: 2 waiting, 1 running."),
            "{t}"
        );
        assert!(t.contains("CPU 1 is running task 3 \"shell\"."));
        assert!(t.contains("Uptime: 5.1 seconds"));
        let i = report::interrupts(&[IrqRow {
            irq: 64,
            name: "igb-0000:00:02.0",
        }]);
        assert!(i.contains("IRQ 64 is \"igb-0000:00:02.0\" (network interface card)."));
        let n = report::netifs(&[NetIf {
            name: "igb0",
            ipv4: &[],
            gateway: None,
            mac: [0x52, 0x54, 0, 0x12, 0x34, 0x56],
            link: Link::Down,
            speed_mbps: 100,
            irqs: &[65],
            poll_mode: false,
        }]);
        assert!(n.contains("MAC address: 52:54:00:12:34:56"));
        assert!(n.contains("Link: down (the cable is not connected)"));
        assert!(n.contains("IPv4 address: none"));
    }

    #[test]
    fn system_prompt_matches_template() {
        let p = system_prompt(
            &[ToolSpec {
                name: "task",
                description: "List \"tasks\".",
                examples: &[],
            }],
            None,
        );
        assert!(p.starts_with("<|im_start|>system\n# Tools\n\n"));
        assert!(p.contains(
            "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"task\", \
             \"description\": \"List \\\"tasks\\\".\", \"parameters\": {\"type\": \"object\", \
             \"properties\": {}, \"required\": []}}}\n</tools>"
        ));
        assert!(p.ends_with("</tool_call><|im_end|>\n"));
    }
}
