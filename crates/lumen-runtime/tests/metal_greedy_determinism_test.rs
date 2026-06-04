//! DET-001 regression guard: Metal greedy decode (temperature=0) MUST be
//! byte-deterministic across repeated identical generations.
//!
//! BACKGROUND
//! ----------
//! DET-001 was three intra-kernel cross-threadgroup races in the Metal Qwen3.5
//! path that made greedy decode non-deterministic at a low, scheduler-timing
//! dependent rate. All three are now fixed BIT-FOR-BIT (default behavior, no env
//! toggle):
//!   1. `ssm_conv1d_state_update` split (gdn_advanced.msl / gdn.rs) -- the batched
//!      prefill conv_state circular-buffer update was racing the conv1d reads; it
//!      is now a separate barriered dispatch.
//!   2. `fused_rope_kv_mha` (attention.msl) -- the in-place RoPE write-back to the
//!      shared `k_vec` raced across GQA Q-head threadgroups; the RoPE'd K now goes
//!      only to the per-position K cache (race-free).
//!   3. `deinterleave_norm_assemble` (gdn_core.msl) -- the assembled K/V were
//!      written into `qkv_buf`, aliasing the still-in-flight qgate read under GQA;
//!      K/V are now normalized in their own buffers and copied in after a barrier.
//!
//! SCOPE OF THIS TEST vs THE SHELL HARNESS
//! ---------------------------------------
//! Races #1 and #3 require the *Qwen3.5* GDN recurrence + Q+gate fusion kernels,
//! which the tiny synthetic test model does not contain; race #2 requires a GQA
//! ratio > 1. The synthetic model here is configured GQA (num_heads > num_kv_heads)
//! so it drives the real `metal/decode_single_cb.rs` single-command-buffer decode
//! loop and the shared activation buffers (`x_buf`, `qkv_buf`, `attn_out_buf`,
//! KV cache) through repeated greedy steps. This is a genuine guard against gross
//! decode non-determinism / buffer-aliasing regressions in that shared path, but it
//! does NOT exercise the GDN/qgate kernels.
//!
//! The AUTHORITATIVE, full-architecture DET-001 regression (all three races, real
//! Qwen3.5-9B Q8/Q4) is the checked-in harness:
//!     scripts/metal_determinism_regression.sh
//! which loads the real model and asserts 1 distinct md5 / N. CI / release sign-off
//! runs that script; this Rust test is the cheap, in-tree structural complement.
//!
//! Gated `#[ignore]` because it needs a real Metal GPU. Run with:
//!   cargo test -p lumen-runtime --test metal_greedy_determinism_test -- --ignored --nocapture
#![cfg(target_os = "macos")]

use lumen_format::test_model::{generate_test_model_q8_0, TestModelQ8Config};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::metal::MetalF32Backend;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write a tiny GQA Q8 synthetic model (num_heads > num_kv_heads so the decode
/// path uses the same GQA-share structure that race #2 lived in).
fn write_gqa_q8_model() -> std::path::PathBuf {
    let cfg = TestModelQ8Config {
        num_heads: 4,
        num_kv_heads: 2,
        ..TestModelQ8Config::default()
    };
    let data = generate_test_model_q8_0(&cfg);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_det_gqa_q8_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model_q8.lbc");
    std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
    path
}

/// One greedy generation through the REAL Metal decode path -> token ids.
/// Mirrors the CLI wiring (set_global_tensors -> set_*_raw -> init ->
/// preload_weights -> engine.generate).
fn generate_tokens(model_path: &std::path::Path, prompt: &[u32], max_new: usize) -> Vec<u32> {
    let provider = SyncWeightProvider::open(model_path).expect("open model");
    let hyper = provider.lbc().header.hyperparams.clone();

    let mut backend = MetalF32Backend::new().expect("Metal backend must be available on macOS");
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    if !provider.output_proj_raw.is_empty() {
        backend.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if !provider.embedding_raw.is_empty() {
        backend.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    if provider.weight_tying {
        backend.set_weight_tying(true);
    }
    backend.init(&hyper).expect("Metal init failed");
    backend.preload_weights(&provider).expect("Metal preload_weights failed");

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F16, // Metal KV cache is F16-only.
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        hyper,
    );

    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };
    let result = engine
        .generate(prompt, &provider, &backend, &StopCondition::MaxTokens(max_new), &sampling)
        .expect("generate failed");
    result.tokens
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn metal_greedy_decode_is_byte_deterministic_across_repeats() {
    let model = write_gqa_q8_model();
    let prompt: Vec<u32> = vec![1, 5, 9, 13, 17, 21, 25, 29]; // 8-token prompt -> batched prefill
    let max_new = 24usize;

    // Reference run.
    let reference = generate_tokens(&model, &prompt, max_new);
    assert!(!reference.is_empty(), "reference generation produced no tokens");

    // N repeats MUST match the reference token-for-token. A divergence is a
    // DET-001-class non-determinism regression in the shared decode CB.
    const N: usize = 20;
    for i in 0..N {
        let again = generate_tokens(&model, &prompt, max_new);
        assert_eq!(
            again, reference,
            "DET-001 REGRESSION: greedy decode diverged on repeat {i} \
             (ref={reference:?} got={again:?}). The Metal single-CB decode path is \
             non-deterministic -- check the three DET-001 race fixes \
             (fused_rope_kv_mha k_vec, deinterleave_norm_assemble aliasing, \
             ssm_conv1d_state_update split)."
        );
    }

    let _ = std::fs::remove_dir_all(model.parent().unwrap());
}
