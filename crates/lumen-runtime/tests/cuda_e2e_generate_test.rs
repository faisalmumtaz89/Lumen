//! CUDA end-to-end generation test.
//!
//! Runs the full `InferenceEngine::generate()` loop with a synthetic test model
//! on the CUDA backend and compares output tokens against the CPU naive backend.
//! This proves the entire decode pipeline works end-to-end: embed -> (compute_layer
//! x num_layers -> advance_kv -> compute_final -> sample) x num_tokens.
//!
//! All tests use greedy sampling (temperature=0) for deterministic comparison.
//! The CUDA backend uses the streaming/CPU path (gpu_resident=false), so the
//! engine drives per-layer forward_pass + compute_final + CPU-side sampling.
//!
//! Gated behind `--features cuda`. Will fail on macOS (no NVIDIA GPU).

#![cfg(feature = "cuda")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write synthetic LBC to a temp file and return (provider, CPU backend, CUDA backend).
///
/// Both backends are initialized with identical global tensors and hyperparams.
/// Returns Err if CUDA device initialization fails (no GPU).
fn setup_cpu_and_cuda(
    config: &TestModelConfig,
) -> Result<(SyncWeightProvider, NaiveF32Backend, CudaBackend), lumen_runtime::RuntimeError> {
    let lbc_data = generate_test_model(config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_e2e_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path)?;
    let hp = provider.lbc().header.hyperparams;

    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cpu.init(&hp)?;

    let mut cuda = CudaBackend::new(0)?;
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp)?;

    Ok((provider, cpu, cuda))
}

/// Run `InferenceEngine::generate()` with a given backend, returning the token sequence.
fn run_generate(
    provider: &SyncWeightProvider,
    backend: &dyn ComputeBackend,
    prompt: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let hp = provider.lbc().header.hyperparams;
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: hp.max_seq_len as usize,
            collect_per_layer_timings: false,
        },
        hp,
    );

    let stop = StopCondition::MaxTokens(max_tokens);
    let sampling = SamplingParams {
        temperature: 0.0,
        seed: Some(42),
    };

    engine
        .generate(prompt, provider, backend, &stop, &sampling)
        .unwrap()
        .tokens
}

// ---------------------------------------------------------------------------
// Test 1: F32 test model, 10 tokens, CUDA vs CPU
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_f32_generate_matches_cpu() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let prompt = vec![0u32, 1, 2];
    let max_tokens = 10;

    let cpu_tokens = run_generate(&provider, &cpu, &prompt, max_tokens);
    let cuda_tokens = run_generate(&provider, &cuda, &prompt, max_tokens);

    assert_eq!(cpu_tokens.len(), max_tokens, "CPU should generate {max_tokens} tokens");
    assert_eq!(cuda_tokens.len(), max_tokens, "CUDA should generate {max_tokens} tokens");
    assert_eq!(
        cpu_tokens, cuda_tokens,
        "CUDA must produce identical tokens to CPU (greedy, F32)\n\
         CPU:  {cpu_tokens:?}\n\
         CUDA: {cuda_tokens:?}"
    );

    eprintln!("cuda_e2e_f32_generate_matches_cpu: tokens={cpu_tokens:?}");
}

// ---------------------------------------------------------------------------
// Test 2: F32 test model, single-token prompt
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_f32_single_token_prompt() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let prompt = vec![0u32];
    let max_tokens = 5;

    let cpu_tokens = run_generate(&provider, &cpu, &prompt, max_tokens);
    let cuda_tokens = run_generate(&provider, &cuda, &prompt, max_tokens);

    assert_eq!(cpu_tokens.len(), max_tokens);
    assert_eq!(cuda_tokens.len(), max_tokens);
    assert_eq!(
        cpu_tokens, cuda_tokens,
        "CUDA must match CPU with single-token prompt\n\
         CPU:  {cpu_tokens:?}\n\
         CUDA: {cuda_tokens:?}"
    );

    // Verify tokens are within vocab range.
    let vocab_size = provider.lbc().header.hyperparams.vocab_size;
    for &tok in &cuda_tokens {
        assert!(tok < vocab_size, "token {tok} >= vocab_size {vocab_size}");
    }

    eprintln!("cuda_e2e_f32_single_token_prompt: tokens={cpu_tokens:?}");
}

// ---------------------------------------------------------------------------
// Test 3: Multi-token stress test (20 tokens) to exercise KV cache growth
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_f32_multi_token_stress() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let prompt = vec![0u32, 1, 2];
    let max_tokens = 20;

    let cpu_tokens = run_generate(&provider, &cpu, &prompt, max_tokens);
    let cuda_tokens = run_generate(&provider, &cuda, &prompt, max_tokens);

    assert_eq!(cpu_tokens.len(), max_tokens, "CPU should generate {max_tokens} tokens");
    assert_eq!(cuda_tokens.len(), max_tokens, "CUDA should generate {max_tokens} tokens");
    assert_eq!(
        cpu_tokens, cuda_tokens,
        "CUDA must match CPU over 20 tokens (KV cache stress test)\n\
         CPU:  {cpu_tokens:?}\n\
         CUDA: {cuda_tokens:?}"
    );

    eprintln!(
        "cuda_e2e_f32_multi_token_stress: first 10 tokens={:?}",
        &cpu_tokens[..10]
    );
}

// ---------------------------------------------------------------------------
// Test 4: Determinism -- two CUDA runs produce identical output
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_deterministic() {
    let config = TestModelConfig::default();
    let (provider, _cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let prompt = vec![0u32, 1, 2];
    let max_tokens = 10;

    let run1 = run_generate(&provider, &cuda, &prompt, max_tokens);

    // Second CUDA backend instance with identical setup.
    let (provider2, _cpu2, cuda2) = setup_cpu_and_cuda(&config).unwrap();
    let run2 = run_generate(&provider2, &cuda2, &prompt, max_tokens);

    assert_eq!(
        run1, run2,
        "Two CUDA runs must produce identical tokens (determinism)\n\
         run1: {run1:?}\n\
         run2: {run2:?}"
    );

    eprintln!("cuda_e2e_deterministic: tokens={run1:?}");
}

// ---------------------------------------------------------------------------
// Test 5: Metrics sanity -- prefill + decode metrics are populated
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_metrics_populated() {
    let config = TestModelConfig::default();
    let (provider, _cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: hp.max_seq_len as usize,
            collect_per_layer_timings: false,
        },
        hp,
    );

    let prompt = vec![0u32, 1, 2];
    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams {
        temperature: 0.0,
        seed: Some(42),
    };

    let result = engine
        .generate(&prompt, &provider, &cuda, &stop, &sampling)
        .unwrap();

    assert_eq!(result.tokens.len(), 5);
    assert_eq!(result.metrics.prompt_tokens, 3);
    assert_eq!(result.metrics.generated_tokens, 5);
    assert!(
        result.metrics.total_time.as_nanos() > 0,
        "total_time should be > 0"
    );
    assert!(
        result.metrics.prefill_time.as_nanos() > 0,
        "prefill_time should be > 0"
    );
    assert!(
        result.metrics.decode_time.as_nanos() > 0,
        "decode_time should be > 0"
    );

    eprintln!(
        "cuda_e2e_metrics_populated: decode={:.1} tok/s, prefill={:.1} tok/s",
        result.metrics.decode_tokens_per_sec,
        result.metrics.prefill_tokens_per_sec,
    );
}

// ---------------------------------------------------------------------------
// Test 6: Larger model config (4 layers, wider dims) to stress multi-layer
// ---------------------------------------------------------------------------

#[test]
fn cuda_e2e_larger_model_matches_cpu() {
    let config = TestModelConfig {
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 64,
        max_seq_len: 128,
        seed: 123,
    };

    let (provider, cpu, cuda) = match setup_cpu_and_cuda(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA e2e test (no GPU?): {e}");
            return;
        }
    };

    let prompt = vec![0u32, 5, 10, 15];
    let max_tokens = 10;

    let cpu_tokens = run_generate(&provider, &cpu, &prompt, max_tokens);
    let cuda_tokens = run_generate(&provider, &cuda, &prompt, max_tokens);

    assert_eq!(cpu_tokens.len(), max_tokens);
    assert_eq!(cuda_tokens.len(), max_tokens);
    assert_eq!(
        cpu_tokens, cuda_tokens,
        "CUDA must match CPU on larger model (4 layers, GQA 4Q/2KV)\n\
         CPU:  {cpu_tokens:?}\n\
         CUDA: {cuda_tokens:?}"
    );

    eprintln!("cuda_e2e_larger_model_matches_cpu: tokens={cpu_tokens:?}");
}
