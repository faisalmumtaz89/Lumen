//! Check the tokenizer against a corpus of reference ids.
//!
//! The single-prompt check (`tokenizer-check`) proves one prompt. This one covers
//! the cases where a hand-written pre-tokenizer typically diverges from the
//! regex it is imitating: runs of spaces, leading and trailing whitespace,
//! newlines, contractions, punctuation, multi-byte characters, and the special
//! tokens themselves.
//!
//! Usage: `tokenizer-corpus-check <checkpoint-dir> <corpus.json>`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::pipeline::render_prompt;
use lumen_image::tokenizer::Tokenizer;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let ckpt = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: tokenizer-corpus-check <checkpoint-dir> <corpus.json>");
        std::process::exit(2)
    }));
    let corpus = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: tokenizer-corpus-check <checkpoint-dir> <corpus.json>");
        std::process::exit(2)
    }));

    let tok = Tokenizer::from_files_with_added(
        &ckpt.join("processor/vocab.json"),
        &ckpt.join("processor/merges.txt"),
        Some(&ckpt.join("processor/added_tokens.json")),
    )
    .map_err(|e| format!("{e}"))?;

    let raw = std::fs::read(&corpus).map_err(|e| format!("{}: {e}", corpus.display()))?;
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&raw).map_err(|e| format!("corpus parse: {e}"))?;

    let mut failures = 0usize;
    for case in &cases {
        let prompt = case["prompt"].as_str().unwrap_or("");
        let want: Vec<u32> = case["template_ids"]
            .as_array()
            .ok_or("case has no template_ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect();
        let got = tok
            .encode(&render_prompt(prompt))
            .map_err(|e| e.to_string())?;
        if got != want {
            failures += 1;
            let at = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            println!(
                "  MISMATCH {prompt:?}\n    at {at}: ours={:?} ref={:?}\n    lengths ours={} ref={}",
                got.get(at),
                want.get(at),
                got.len(),
                want.len()
            );
        }
    }
    println!(
        "{} of {} cases matched",
        cases.len() - failures,
        cases.len()
    );
    if failures > 0 {
        return Err(format!("{failures} case(s) diverged from the reference"));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
