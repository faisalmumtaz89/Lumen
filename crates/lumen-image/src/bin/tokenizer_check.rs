//! Our BPE must reproduce the prompt token ids the oracle recorded.
//!
//! The oracle's `prompt_input_ids` are what the reference tokenizer produced for
//! the rendered template. Reproducing them exactly is the whole requirement —
//! one id off and the text encoder is being asked a different question.
//!
//! Usage: `tokenizer-check <checkpoint-dir> <oracle-dir> [tag]`

use std::path::PathBuf;
use std::process::ExitCode;

use lumen_image::npy;
use lumen_image::tokenizer::Tokenizer;

const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: tokenizer-check <checkpoint-dir> <oracle-dir> [tag]");
        std::process::exit(2)
    };
    let ckpt = PathBuf::from(args.next().unwrap_or_else(usage));
    let oracle = PathBuf::from(args.next().unwrap_or_else(usage));
    let tag = args.next().unwrap_or_else(|| "smoke".to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| "A red apple on a wooden table, studio lighting".to_string());

    let tok = Tokenizer::from_files_with_added(
        &ckpt.join("processor/vocab.json"),
        &ckpt.join("processor/merges.txt"),
        Some(&ckpt.join("processor/added_tokens.json")),
    )
    .map_err(|e| format!("{e}"))?;
    println!("vocabulary: {} tokens", tok.vocab_size());

    // The pipeline's text-to-image template, verbatim.
    let template = format!(
        "<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let ours = tok.encode(&template).map_err(|e| e.to_string())?;

    let want = npy::load(&oracle.join(format!("prompt_input_ids_{tag}.npy")))
        .map_err(|e| format!("{e}"))?;
    let want_ids: Vec<u32> = want.data.iter().map(|&v| v as u32).collect();

    println!("ours {} ids, oracle {} ids", ours.len(), want_ids.len());
    if ours != want_ids {
        let first = ours
            .iter()
            .zip(&want_ids)
            .position(|(a, b)| a != b)
            .unwrap_or(ours.len().min(want_ids.len()));
        for i in first.saturating_sub(2)..(first + 4).min(ours.len().max(want_ids.len())) {
            let a = ours.get(i).copied();
            let b = want_ids.get(i).copied();
            let mark = if a == b { " " } else { "*" };
            let text = a.map(|x| tok.decode(&[x])).unwrap_or_default();
            println!("  {mark} [{i:3}] ours={a:?} oracle={b:?}  {text:?}");
        }
        return Err(format!("first divergence at index {first}"));
    }
    println!("tokenizer matches the oracle exactly ({} ids)", ours.len());
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
