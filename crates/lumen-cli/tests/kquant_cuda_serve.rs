//! A K-quant artifact served on CUDA keeps its Q4_K / Q5_K / Q6_K planes native:
//! the load prints the plane counters with the host-dequant catch-all at zero
//! for all three schemes and no F16 cache built for an F32-resident plane,
//! and the embedding is gathered from its native plane.
//!
//! Requires a CUDA GPU and a K-quant `.lbc` (a Qwen3.8-27B Q4_K_M or Q5_K_M
//! conversion) at `LUMEN_KQUANT_LBC`:
//!
//!   LUMEN_KQUANT_LBC=/path/to/model.lbc cargo test --release -p lumen-cli --features cuda \
//!       --test kquant_cuda_serve -- --ignored

use std::process::Command;

#[test]
#[ignore = "requires a CUDA GPU and a K-quant artifact at LUMEN_KQUANT_LBC"]
fn k_quant_artifact_serves_its_planes_natively() {
    let lbc = std::env::var("LUMEN_KQUANT_LBC").expect("LUMEN_KQUANT_LBC");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args([
            "run",
            "--model",
            &lbc,
            "--cuda",
            "--prompt",
            "The capital of France is",
            // the chat template turns the prompt into a question; the 27B answers it
            // in a short sentence, so the budget covers the sentence
            "--max-tokens",
            "16",
            "--temperature",
            "0",
            "--repetition-penalty",
            "1.0",
            "--presence-penalty",
            "0",
            "--frequency-penalty",
            "0",
        ])
        .env("LUMEN_CUDA_VERBOSE", "1")
        .output()
        .expect("spawn lumen");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "lumen run failed: {stderr}");
    assert!(stdout.contains("Paris"), "unexpected completion: {stdout}");

    let counters = stderr
        .lines()
        .find(|l| l.starts_with("[CUDA] K-quant planes:"))
        .unwrap_or_else(|| panic!("no K-quant plane counters line in:\n{stderr}"));
    assert!(
        counters.contains("host-dequant catch-all Q4_K=0 Q5_K=0 Q6_K=0"),
        "a K-quant plane went through the host-dequant catch-all: {counters}"
    );
    assert!(
        counters.contains("F16 caches built for F32-resident planes=0"),
        "an F16 cache was built: {counters}"
    );
    let after = counters.split("native ").nth(1).expect("native section");
    let native: usize = ["Q4_K=", "Q5_K=", "Q6_K="]
        .iter()
        .map(|key| {
            let field = after.split(key).nth(1).unwrap_or("0");
            field
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(0)
        })
        .sum();
    assert!(native > 0, "no plane was served natively: {counters}");
    assert!(
        stderr.lines().any(|l| {
            l.starts_with("[CUDA] embed_token_q4_k: ACTIVE")
                || l.starts_with("[CUDA] embed_token_q5_k: ACTIVE")
                || l.starts_with("[CUDA] embed_token_q6_k: ACTIVE")
        }),
        "the embedding was not gathered from its native K-quant plane:\n{stderr}"
    );
}
