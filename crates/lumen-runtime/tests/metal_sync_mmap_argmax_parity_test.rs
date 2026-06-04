//! CALL-SITE regression guard: Metal-Q8 first-token argmax
//! MUST be identical for the SyncWeightProvider path and the MmapWeightProvider
//! path.
//!
//! Background (root cause): the Metal GPU-resident batched prefill
//! (`metal/prefill.rs`) uploads layer weights to a unified buffer in their RAW
//! native-quant layout, then reads the per-subtensor byte OFFSETS from the
//! `LayerView` returned for that layer. On the fallback path it MUST call
//! `weights.get_layer_raw(layer)` (raw offsets that match the raw buffer), NOT
//! `weights.get_layer_blocking(layer)` (which on a SyncWeightProvider rebuilds
//! the blob to F32 with DIFFERENT offsets). Mmap's `try_get_layer` returns a
//! pre-built raw view so it was never affected; a SyncWeightProvider's
//! `try_get_layer` returns `None`, so the fallback fired and — before the fix —
//! fed F32 offsets to the raw GPU buffer, collapsing argmax to the pad token
//! (`[PAD248319]`).
//!
//! This test drives the REAL `metal/prefill.rs` code path through
//! `InferenceEngine::generate` for both providers and asserts the generated
//! tokens match. If `metal/prefill.rs` is reverted to `get_layer_blocking`, the
//! sync path diverges (pad-token garbage) and this test FAILS — that is the
//! counterfactual this guard exists to catch.
//!
//! Gated `#[ignore]` because it needs a real Metal GPU. It uses the tiny
//! synthetic Q8 model (no large-model download, ~kilobytes), so when run on
//! macOS it is cheap. Run with:
//!   cargo test -p lumen-runtime --test metal_sync_mmap_argmax_parity_test -- --ignored --nocapture
#![cfg(target_os = "macos")]

use lumen_format::test_model::{generate_test_model_q8_0, TestModelQ8Config};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::metal::MetalF32Backend;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_q8_model() -> std::path::PathBuf {
    let data = generate_test_model_q8_0(&TestModelQ8Config::default());
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_metal_argmax_parity_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model_q8.lbc");
    std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
    path
}

/// Wire a `MetalF32Backend` from the provider's globals exactly as the CLI
/// `create_backend` + `run_with_sync` do (set_global_tensors -> set_*_raw ->
/// set_weight_tying -> init -> preload_weights), then return it ready for
/// `engine.generate`. `embedding_raw`/`output_proj_raw`/`embedding_quant`/etc.
/// are read straight off the provider so both sync and mmap feed identical
/// global tensors — isolating the layer-view OFFSET source as the only variable.
macro_rules! wire_metal_backend {
    ($provider:expr, $hyper:expr) => {{
        let mut backend = MetalF32Backend::new()
            .expect("Metal backend must be available on macOS");
        backend.set_global_tensors(
            $provider.embedding.clone(),
            $provider.final_norm.clone(),
            $provider.output_proj.clone(),
        );
        if !$provider.output_proj_raw.is_empty() {
            backend.set_output_proj_raw(
                $provider.output_proj_raw.clone(),
                $provider.output_proj_quant,
            );
        }
        if !$provider.embedding_raw.is_empty() {
            backend.set_embedding_raw(
                $provider.embedding_raw.clone(),
                $provider.embedding_quant,
            );
        }
        if $provider.weight_tying {
            backend.set_weight_tying(true);
        }
        backend.init(&$hyper).expect("Metal init failed");
        backend
            .preload_weights(&$provider)
            .expect("Metal preload_weights failed");
        backend
    }};
}

fn engine_for(hyper: lumen_format::hyperparams::ModelHyperparams) -> InferenceEngine {
    InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            // Metal KV cache is F16-only (validate_kv_precision).
            kv_precision: KvPrecision::F16,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        hyper,
    )
}

#[ignore = "needs a real Metal GPU; opt-in call-site regression guard"]
#[test]
fn metal_q8_first_token_argmax_identical_for_sync_and_mmap() {
    let path = write_q8_model();

    // ---- mmap provider (the path that always worked) ----
    let mmap = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
    let mmap_hyper = mmap.lbc().header.hyperparams.clone();
    let mmap_backend = wire_metal_backend!(mmap, mmap_hyper);

    // ---- sync provider (the path that produced [PAD248319] before the fix) ----
    let sync = SyncWeightProvider::open(&path).unwrap();
    let sync_hyper = sync.lbc().header.hyperparams.clone();
    let sync_backend = wire_metal_backend!(sync, sync_hyper);

    // Greedy decode is deterministic; compare the full short sequence (the first
    // token is the argmax that collapsed to PAD in the bug, the rest confirm the
    // whole prefill->decode chain agrees).
    let prompt_tokens = vec![0u32, 1, 2, 3];
    let stop = StopCondition::MaxTokens(6);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };

    let mmap_engine = engine_for(mmap_hyper);
    let mmap_result = mmap_engine
        .generate(&prompt_tokens, &mmap, &mmap_backend, &stop, &sampling)
        .expect("mmap generate failed");

    let sync_engine = engine_for(sync_hyper);
    let sync_result = sync_engine
        .generate(&prompt_tokens, &sync, &sync_backend, &stop, &sampling)
        .expect("sync generate failed");

    eprintln!("mmap tokens: {:?}", mmap_result.tokens);
    eprintln!("sync tokens: {:?}", sync_result.tokens);

    assert!(!mmap_result.tokens.is_empty(), "mmap must generate at least one token");

    // The headline guard: first-token argmax identical between providers.
    assert_eq!(
        sync_result.tokens.first(),
        mmap_result.tokens.first(),
        "Metal-Q8 FIRST-TOKEN argmax differs between sync and mmap providers — \
         this is the [PAD248319] regression (metal/prefill.rs reverted to \
         get_layer_blocking would cause exactly this).",
    );

    // Stronger: the entire greedy sequence must match (prefill + decode chain).
    assert_eq!(
        sync_result.tokens, mmap_result.tokens,
        "Metal-Q8 greedy generation differs between sync and mmap providers",
    );
}
