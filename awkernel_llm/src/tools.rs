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
