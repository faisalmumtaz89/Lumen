//! End-to-end integration test: generate synthetic model → run inference → verify output.

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::compute::naive::NaiveF32Backend;
use lumen_runtime::compute::simd::SimdF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::metal::MetalF32Backend;
use lumen_runtime::weight::provider_async::AsyncWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
#[cfg(unix)]
use lumen_runtime::storage::MmapConfig;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn setup_test_model() -> (SyncWeightProvider, NaiveF32Backend) {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);

    // Use unique path per call to avoid parallel test interference
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_e2e_test_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path).unwrap();

    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();

    (provider, backend)
}

fn setup_test_model_simd() -> (SyncWeightProvider, SimdF32Backend) {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_e2e_simd_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path).unwrap();

    let mut backend = SimdF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();

    (provider, backend)
}

#[test]
fn e2e_deterministic_generation() {
    let (provider, backend) = setup_test_model();

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: true,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams {
        temperature: 0.0, // greedy for determinism
        seed: Some(42),
        ..Default::default()
    };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();

    assert_eq!(result.tokens.len(), 5, "should generate exactly 5 tokens");
    assert!(result.metrics.generated_tokens == 5);
    assert!(result.metrics.prompt_tokens == 3);

    // Run again — must produce identical output (determinism check)
    let (provider2, backend2) = setup_test_model();
    let engine2 = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: true,
        },
        provider2.lbc().header.hyperparams.clone(),
    );

    let result2 = engine2.generate(&prompt_tokens, &provider2, &backend2, &stop, &sampling).unwrap();
    assert_eq!(result.tokens, result2.tokens, "greedy generation must be deterministic");

    eprintln!("Generated tokens: {:?}", result.tokens);
    eprintln!("{}", result.metrics.summary());
}

#[test]
fn e2e_temperature_sampling() {
    let (provider, backend) = setup_test_model();

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(10);
    let sampling = SamplingParams {
        temperature: 0.8,
        seed: Some(12345),
        ..Default::default()
    };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();
    assert_eq!(result.tokens.len(), 10);

    // All tokens should be within vocab range
    let vocab_size = provider.lbc().header.hyperparams.vocab_size;
    for &tok in &result.tokens {
        assert!(tok < vocab_size, "token {tok} >= vocab_size {vocab_size}");
    }

    eprintln!("Temperature sampling tokens: {:?}", result.tokens);
}

#[test]
fn e2e_eos_stop() {
    let (provider, backend) = setup_test_model();

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    // Use a greedy run to find what the first token will be, then set that as EOS
    let prompt = vec![0u32];
    let sampling = SamplingParams { temperature: 0.0, seed: None, ..Default::default() };
    let result = engine.generate(
        &prompt,
        &provider,
        &backend,
        &StopCondition::MaxTokens(1),
        &sampling,
    ).unwrap();
    let first_token = result.tokens[0];

    // Now set that token as EOS — should stop after 1 token
    let stop = StopCondition::MaxTokensOrEos {
        max_tokens: 100,
        eos_tokens: vec![first_token],
    };
    let result = engine.generate(&prompt, &provider, &backend, &stop, &sampling).unwrap();
    assert_eq!(result.tokens.len(), 1);
    assert_eq!(result.tokens[0], first_token);
}

#[test]
#[cfg(unix)]
fn e2e_mmap_deterministic() {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_e2e_mmap_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: true,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();
    assert_eq!(result.tokens.len(), 5);

    // Must match the sync provider output (same deterministic output)
    let (sync_provider, sync_backend) = setup_test_model();
    let sync_engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        sync_provider.lbc().header.hyperparams.clone(),
    );
    let sync_result = sync_engine.generate(&prompt_tokens, &sync_provider, &sync_backend, &stop, &sampling).unwrap();
    assert_eq!(result.tokens, sync_result.tokens, "mmap and sync providers must produce identical output");
}

#[test]
fn e2e_empty_prompt_error() {
    let (provider, backend) = setup_test_model();
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams { temperature: 0.0, seed: None, ..Default::default() };
    let result = engine.generate(&[], &provider, &backend, &stop, &sampling);
    assert!(result.is_err(), "empty prompt should produce an error");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("empty prompt"), "error should mention empty prompt: {err_msg}");
}

#[test]
fn e2e_kv_cache_overflow() {
    let (provider, backend) = setup_test_model();
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 5, // Very small: prompt(3) + generate can overflow
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2]; // 3 tokens uses 3 KV slots
    let stop = StopCondition::MaxTokens(10); // Try to generate 10 more
    let sampling = SamplingParams { temperature: 0.0, seed: None, ..Default::default() };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling);
    // Should either error with KvCache overflow, or generate fewer tokens
    // With max_seq_len=5, prompt=3, it can generate at most 2 tokens before overflow
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("KV cache") || msg.contains("exceed"), "unexpected error: {msg}");
        }
        Ok(r) => {
            // If it didn't error, it should have stopped before overflowing
            assert!(r.tokens.len() <= 2, "should generate at most 2 tokens with max_seq_len=5 and 3 prompt tokens");
        }
    }
}

#[test]
fn e2e_per_layer_timing_count() {
    let (provider, backend) = setup_test_model();
    let num_layers = provider.lbc().header.hyperparams.num_layers as usize;
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: true,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(3);
    let sampling = SamplingParams { temperature: 0.0, seed: None, ..Default::default() };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();

    // Each token (prompt + generated) processes all layers
    // prompt_tokens=3, generated=3, but the first generated token reuses the
    // last prompt forward pass for logits, so forward passes = prompt_tokens + generated_tokens - 1
    // Actually: the engine does forward_pass for each prompt token (3) + forward_pass for each
    // generated token except the first (which gets logits from the last prompt forward).
    // Looking at engine.rs: prefill does forward_pass for each prompt token (3 passes),
    // then decode does forward_pass for each generated token after the first (2 passes).
    // Total forward passes = 3 + (3-1) = 5, each producing num_layers timing entries.
    let total_tokens_with_forward = prompt_tokens.len() + result.tokens.len() - 1;
    let expected = num_layers * total_tokens_with_forward;
    assert_eq!(
        result.metrics.per_layer_timings.len(), expected,
        "expected {} timing entries (num_layers={} * tokens={}), got {}",
        expected, num_layers, total_tokens_with_forward, result.metrics.per_layer_timings.len()
    );
}

#[test]
fn e2e_io_metrics_populated() {
    let (provider, backend) = setup_test_model();
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();

    // I/O metrics must be populated — the sync backend reads layer weights from disk
    assert!(result.metrics.io.bytes_read > 0, "bytes_read should be > 0, got {}", result.metrics.io.bytes_read);
    assert!(result.metrics.io.read_ops > 0, "read_ops should be > 0, got {}", result.metrics.io.read_ops);
    assert!(result.metrics.io.duration.as_nanos() > 0, "io duration should be > 0");

    // Bandwidth should be computable (non-NaN, non-infinite)
    let bw = result.metrics.io.read_bandwidth_gibs();
    assert!(bw.is_finite(), "bandwidth should be finite, got {bw}");
}

#[test]
fn e2e_single_token_prompt() {
    let (provider, backend) = setup_test_model();
    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        provider.lbc().header.hyperparams.clone(),
    );

    let prompt_tokens = vec![0]; // Single token
    let stop = StopCondition::MaxTokens(3);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };

    let result = engine.generate(&prompt_tokens, &provider, &backend, &stop, &sampling).unwrap();
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.metrics.prompt_tokens, 1);

    // Verify all generated tokens are within vocab range
    let vocab_size = provider.lbc().header.hyperparams.vocab_size;
    for &tok in &result.tokens {
        assert!(tok < vocab_size, "token {tok} >= vocab_size {vocab_size}");
    }
}

#[test]
fn e2e_async_matches_sync() {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_e2e_async_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let prompt_tokens = vec![0, 1, 2];
    let stop = StopCondition::MaxTokens(5);
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };

    // Generate with sync provider.
    let sync_provider = SyncWeightProvider::open(&path).unwrap();
    let mut sync_backend = NaiveF32Backend::new();
    sync_backend.set_global_tensors(
        sync_provider.embedding.clone(),
        sync_provider.final_norm.clone(),
        sync_provider.output_proj.clone(),
    );
    sync_backend.init(&sync_provider.lbc().header.hyperparams).unwrap();

    let rt_config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 64,
        collect_per_layer_timings: false,
    };
    let sync_engine = InferenceEngine::new(rt_config.clone(), sync_provider.lbc().header.hyperparams);
    let sync_result = sync_engine.generate(&prompt_tokens, &sync_provider, &sync_backend, &stop, &sampling).unwrap();

    // Generate with async provider.
    let async_provider = AsyncWeightProvider::open(&path).unwrap();
    let mut async_backend = NaiveF32Backend::new();
    async_backend.set_global_tensors(
        async_provider.embedding.clone(),
        async_provider.final_norm.clone(),
        async_provider.output_proj.clone(),
    );
    async_backend.init(&async_provider.lbc().header.hyperparams).unwrap();

    let async_engine = InferenceEngine::new(rt_config, async_provider.lbc().header.hyperparams);
    let async_result = async_engine.generate(&prompt_tokens, &async_provider, &async_backend, &stop, &sampling).unwrap();

    assert_eq!(
        sync_result.tokens, async_result.tokens,
        "async and sync providers must produce identical output"
    );
    assert_eq!(async_result.tokens.len(), 5);
}

// ==================== SIMD backend e2e tests ====================

/// Helper to run generation with both naive and SIMD backends on the same test model,
/// returning both token sequences for comparison.
fn run_both_backends(
    prompt: &[u32],
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
) -> (Vec<u32>, Vec<u32>) {
    let rt_config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 128,
        collect_per_layer_timings: false,
    };
    let sampling = SamplingParams { temperature, seed, ..Default::default() };
    let stop = StopCondition::MaxTokens(max_tokens);

    // Naive backend
    let (naive_provider, naive_backend) = setup_test_model();
    let naive_engine = InferenceEngine::new(
        rt_config.clone(),
        naive_provider.lbc().header.hyperparams,
    );
    let naive_result = naive_engine
        .generate(prompt, &naive_provider, &naive_backend, &stop, &sampling)
        .unwrap();

    // SIMD backend
    let (simd_provider, simd_backend) = setup_test_model_simd();
    let simd_engine = InferenceEngine::new(
        rt_config,
        simd_provider.lbc().header.hyperparams,
    );
    let simd_result = simd_engine
        .generate(prompt, &simd_provider, &simd_backend, &stop, &sampling)
        .unwrap();

    (naive_result.tokens, simd_result.tokens)
}

#[test]
fn e2e_simd_matches_naive() {
    let (naive_tokens, simd_tokens) = run_both_backends(
        &[0, 1, 2], // prompt
        10,          // max_tokens
        0.0,         // greedy
        Some(42),    // seed
    );

    assert_eq!(naive_tokens.len(), 10);
    assert_eq!(simd_tokens.len(), 10);
    assert_eq!(
        naive_tokens, simd_tokens,
        "SIMD backend must produce identical tokens to naive backend (greedy)\n\
         naive: {naive_tokens:?}\n\
         simd:  {simd_tokens:?}"
    );

    eprintln!("e2e_simd_matches_naive: tokens={naive_tokens:?}");
}

#[test]
fn e2e_simd_temperature_deterministic() {
    let (naive_tokens, simd_tokens) = run_both_backends(
        &[0, 1, 2],  // prompt
        10,           // max_tokens
        0.8,          // temperature
        Some(12345),  // seed
    );

    assert_eq!(naive_tokens.len(), 10);
    assert_eq!(simd_tokens.len(), 10);
    assert_eq!(
        naive_tokens, simd_tokens,
        "SIMD backend must produce identical tokens to naive backend (temperature=0.8)\n\
         naive: {naive_tokens:?}\n\
         simd:  {simd_tokens:?}"
    );

    eprintln!("e2e_simd_temperature_deterministic: tokens={naive_tokens:?}");
}

#[test]
fn e2e_simd_long_generation() {
    let (naive_tokens, simd_tokens) = run_both_backends(
        &[0, 1, 2], // prompt
        50,          // max_tokens — catches drift/accumulation errors
        0.0,         // greedy
        Some(42),    // seed
    );

    assert_eq!(naive_tokens.len(), 50);
    assert_eq!(simd_tokens.len(), 50);
    assert_eq!(
        naive_tokens, simd_tokens,
        "SIMD backend must produce identical tokens to naive over 50 tokens (drift check)\n\
         naive: {naive_tokens:?}\n\
         simd:  {simd_tokens:?}"
    );

    eprintln!("e2e_simd_long_generation: first 10 tokens={:?}", &naive_tokens[..10]);
}

#[test]
fn e2e_simd_single_token_prompt() {
    let (naive_tokens, simd_tokens) = run_both_backends(
        &[0],     // single token prompt
        5,        // max_tokens
        0.0,      // greedy
        Some(42), // seed
    );

    assert_eq!(naive_tokens.len(), 5);
    assert_eq!(simd_tokens.len(), 5);
    assert_eq!(
        naive_tokens, simd_tokens,
        "SIMD backend must match naive with single-token prompt\n\
         naive: {naive_tokens:?}\n\
         simd:  {simd_tokens:?}"
    );

    eprintln!("e2e_simd_single_token_prompt: tokens={naive_tokens:?}");
}

// ==================== Metal backend e2e tests ====================

#[cfg(target_os = "macos")]
fn setup_test_model_metal() -> (SyncWeightProvider, MetalF32Backend) {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_e2e_metal_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path).unwrap();

    let mut backend = MetalF32Backend::new().unwrap();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();

    (provider, backend)
}

#[test]
#[cfg(target_os = "macos")]
fn e2e_metal_matches_naive() {
    let rt_config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 128,
        collect_per_layer_timings: false,
    };
    let prompt = vec![0u32, 1, 2];
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };
    let stop = StopCondition::MaxTokens(10);

    // Naive backend
    let (naive_provider, naive_backend) = setup_test_model();
    let naive_engine = InferenceEngine::new(
        rt_config.clone(),
        naive_provider.lbc().header.hyperparams,
    );
    let naive_result = naive_engine
        .generate(&prompt, &naive_provider, &naive_backend, &stop, &sampling)
        .unwrap();

    // Metal backend
    let (metal_provider, metal_backend) = setup_test_model_metal();
    let metal_engine = InferenceEngine::new(
        rt_config,
        metal_provider.lbc().header.hyperparams,
    );
    let metal_result = metal_engine
        .generate(&prompt, &metal_provider, &metal_backend, &stop, &sampling)
        .unwrap();

    assert_eq!(naive_result.tokens.len(), 10);
    assert_eq!(metal_result.tokens.len(), 10);
    assert_eq!(
        naive_result.tokens, metal_result.tokens,
        "Metal backend must produce identical tokens to naive backend (greedy)\n\
         naive: {:?}\n\
         metal: {:?}",
        naive_result.tokens, metal_result.tokens,
    );

    eprintln!("e2e_metal_matches_naive: tokens={:?}", naive_result.tokens);
}

#[test]
#[cfg(target_os = "macos")]
fn e2e_metal_single_token_prompt() {
    let rt_config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 128,
        collect_per_layer_timings: false,
    };
    let prompt = vec![0u32];
    let sampling = SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() };
    let stop = StopCondition::MaxTokens(5);

    // Naive backend
    let (naive_provider, naive_backend) = setup_test_model();
    let naive_engine = InferenceEngine::new(
        rt_config.clone(),
        naive_provider.lbc().header.hyperparams,
    );
    let naive_result = naive_engine
        .generate(&prompt, &naive_provider, &naive_backend, &stop, &sampling)
        .unwrap();

    // Metal backend
    let (metal_provider, metal_backend) = setup_test_model_metal();
    let metal_engine = InferenceEngine::new(
        rt_config,
        metal_provider.lbc().header.hyperparams,
    );
    let metal_result = metal_engine
        .generate(&prompt, &metal_provider, &metal_backend, &stop, &sampling)
        .unwrap();

    assert_eq!(naive_result.tokens.len(), 5);
    assert_eq!(metal_result.tokens.len(), 5);
    assert_eq!(
        naive_result.tokens, metal_result.tokens,
        "Metal backend must match naive with single-token prompt\n\
         naive: {:?}\n\
         metal: {:?}",
        naive_result.tokens, metal_result.tokens,
    );

    eprintln!("e2e_metal_single_token_prompt: tokens={:?}", naive_result.tokens);
}
