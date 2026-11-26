//! Host driver for checking the engine against llm.asm.
//!
//! ```sh
//! cargo run --release -p awkernel_llm --features std --example llm_cli -- \
//!     -m qwen3-0.6b-q8_0.asmbin -p "Explain how a CPU cache works." -n 64 --print-tokens
//! ```
//!
//! `--print-tokens` prints `PROMPT_TOKENS:` / `GEN_TOKENS:` to stderr in the
//! same format as llm.asm, and `--dump-logits` writes the f32 logits after
//! prefill, so both can be diffed against the reference.

use awkernel_llm::{
    exec::ScopedThreadExecutor, utf8::Utf8Stream, Backend, Chat, ChatConfig, Event, Executor,
    InlineExecutor, Model,
};
use std::{io::Write, time::Instant};

fn usage() -> ! {
    eprintln!(
        "usage: llm_cli -m <model.asmbin> [-p <prompt>] [-n <max_new>] [-c <ctx>] [-t <threads>]\n\
         \x20              [--think] [--scalar] [--temp <f>] [--topp <f>] [--seed <n>]\n\
         \x20              [--print-tokens] [--dump-logits <path>] [--tokens <csv>]"
    );
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_path = None;
    let mut prompt = String::from("Explain how a CPU cache works in 3 concise sentences.");
    let mut cfg = ChatConfig {
        max_new_tokens: 128,
        ..Default::default()
    };
    let mut threads = 1usize;
    let mut backend = Backend::detect();
    let mut print_tokens = false;
    let mut dump_logits: Option<String> = None;
    let mut raw_tokens: Option<Vec<u32>> = None;

    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "-m" => model_path = Some(val()),
            "-p" => prompt = val(),
            "-n" => cfg.max_new_tokens = val().parse().unwrap_or_else(|_| usage()),
            "-c" => cfg.max_seq = val().parse().unwrap_or_else(|_| usage()),
            "-t" => threads = val().parse().unwrap_or_else(|_| usage()),
            "--temp" => cfg.temperature = val().parse().unwrap_or_else(|_| usage()),
            "--topp" => cfg.top_p = val().parse().unwrap_or_else(|_| usage()),
            "--seed" => cfg.seed = val().parse().unwrap_or_else(|_| usage()),
            "--think" => cfg.think = true,
            "--scalar" => backend = Backend::SCALAR,
            "--print-tokens" => print_tokens = true,
            "--dump-logits" => dump_logits = Some(val()),
            "--tokens" => {
                raw_tokens = Some(
                    val()
                        .split(',')
                        .map(|t| t.trim().parse().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            _ => usage(),
        }
    }
    let model_path = model_path.unwrap_or_else(|| usage());

    let data = std::fs::read(&model_path).expect("failed to read the model");
    let model = Model::parse(&data).expect("invalid model");
    let mut chat = Chat::new(&model, cfg, backend).expect("failed to allocate");
    eprintln!(
        "[llm_cli] backend={} threads={threads} ctx={} state={} MiB",
        backend.name(),
        cfg.max_seq,
        chat.transformer().memory_bytes() >> 20
    );

    let threaded = ScopedThreadExecutor { threads };
    let exec: &dyn Executor = if threads > 1 {
        &threaded
    } else {
        &InlineExecutor
    };

    match &raw_tokens {
        Some(t) => chat.start_raw(t).map(|_| ()),
        None => chat.start_turn(&prompt).map(|_| ()),
    }
    .expect("prompt");
    let prompt_toks = chat.prompt_tokens().to_vec();

    let mut out = std::io::stdout().lock();
    let mut utf8 = Utf8Stream::new();
    let mut text = String::new();
    let mut gen = Vec::new();
    let t0 = Instant::now();
    let mut t_first = None;

    loop {
        match chat.step(exec).expect("step") {
            Event::Prefill { .. } => {}
            Event::Token(tok) => {
                if t_first.is_none() {
                    t_first = Some(Instant::now());
                    if let Some(p) = &dump_logits {
                        let bytes: Vec<u8> = chat
                            .transformer()
                            .logits()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
                        std::fs::write(p, bytes).expect("dump logits");
                    }
                }
                gen.push(tok);
                text.clear();
                utf8.push(chat.token_bytes(tok), &mut text);
                out.write_all(text.as_bytes()).unwrap();
                out.flush().unwrap();
            }
            Event::Done(reason) => {
                text.clear();
                utf8.finish(&mut text);
                writeln!(out, "{text}").unwrap();
                eprintln!("[llm_cli] stop: {reason:?}");
                break;
            }
        }
    }

    let t_end = Instant::now();
    let t_first = t_first.unwrap_or(t_end);
    let prefill = (t_first - t0).as_secs_f64();
    let decode = (t_end - t_first).as_secs_f64();
    // The first token comes out of prefill; the rest each cost one step.
    let n_decode = gen.len().saturating_sub(1).max(1);
    eprintln!(
        "[llm_cli] prefill: {} tok in {:.2} ms ({:.2} tok/s)",
        prompt_toks.len(),
        prefill * 1e3,
        prompt_toks.len() as f64 / prefill
    );
    eprintln!(
        "[llm_cli] decode : {} tok in {:.2} ms ({:.2} tok/s)",
        gen.len(),
        decode * 1e3,
        n_decode as f64 / decode.max(1e-9)
    );
    if print_tokens {
        let csv = |v: &[u32]| {
            v.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        eprintln!("PROMPT_TOKENS:{}", csv(&prompt_toks));
        eprintln!("GEN_TOKENS:{}", csv(&gen));
    }
}
