//! CUDA prefill integration tests.
//!
//! Verifies that the CUDA backend's batched prefill produces the same output
//! as token-at-a-time processing through the CPU naive backend. Both backends
//! process the same deterministic weights and inputs.
//!
//! Tests exercise the engine's prefill dispatch path (`caps.batched_prefill = true`),
//! which calls `backend.prefill()` instead of processing tokens one at a time.
//!
//! These tests require a CUDA-capable GPU. They are gated behind
//! `--features cuda` and will fail on macOS (no NVIDIA GPU).

#![cfg(feature = "cuda")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend};
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Set up a test model and return (provider, CPU backend, CUDA backend).
fn setup_backends() -> Result<(SyncWeightProvider, NaiveF32Backend, CudaBackend), RuntimeError> {
    let config = TestModelConfig::default();
    let lbc_data = generate_test_model(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_prefill_test_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path)?;
    let hp = provider.lbc().header.hyperparams;

    // CPU naive backend
    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cpu.init(&hp)?;

    // CUDA backend
    let mut cuda = CudaBackend::new(0)?;
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp)?;

    Ok((provider, cpu, cuda))
}

/// Compare two f32 slices element-wise with a tolerance, printing mismatches.
fn assert_f32_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance {
            if mismatches < 5 {
                eprintln!(
                    "  {label}[{i}]: CUDA={a:.6}, CPU={e:.6}, diff={diff:.2e}"
                );
            }
            mismatches += 1;
        }
        max_diff = max_diff.max(diff);
    }
    assert_eq!(
        mismatches, 0,
        "{label}: {mismatches} elements exceed tolerance {tolerance:.1e} (max_diff={max_diff:.2e})"
    );
}

/// Run CPU token-at-a-time prefill to produce reference hidden state.
fn cpu_prefill_reference(
    provider: &SyncWeightProvider,
    cpu: &NaiveF32Backend,
    prompt_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let hp = provider.lbc().header.hyperparams;
    let hidden_dim = hp.hidden_dim as usize;
    let num_layers = hp.num_layers as usize;

    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })?;

    let mut last_x: Option<ActivationBuffer> = None;
    for &token_id in prompt_tokens {
        let mut x = cpu.embed_token(token_id)?;
        let seq_pos = kv.seq_len();
        for layer in 0..num_layers {
            let layer_view = provider.get_layer_blocking(layer)?;
            let mut kv_view = kv.view_mut(layer)?;
            cpu.compute_layer(layer, &mut x, &layer_view, Some(&mut kv_view), seq_pos)?;
            kv.commit_view(kv_view)?;
        }
        kv.advance_seq_len()?;
        last_x = Some(x);
    }

    let x = last_x.unwrap();
    let mut result = vec![0.0f32; hidden_dim];
    x.read_f32_into(&mut result);
    Ok(result)
}

#[test]
fn test_prefill_matches_token_at_a_time_4_tokens() {
    let (provider, cpu, cuda) = setup_backends().expect("failed to set up backends");
    let prompt = vec![0, 1, 2, 3];

    // CPU reference: token-at-a-time
    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt)
        .expect("CPU prefill failed");

    // CUDA prefill
    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    let cuda_hidden = cuda
        .prefill(&prompt, &provider, &mut kv)
        .expect("CUDA prefill failed");

    assert_f32_close("prefill_4tok", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_prefill_matches_token_at_a_time_8_tokens() {
    let (provider, cpu, cuda) = setup_backends().expect("failed to set up backends");
    let prompt: Vec<u32> = (0..8).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt)
        .expect("CPU prefill failed");

    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    let cuda_hidden = cuda
        .prefill(&prompt, &provider, &mut kv)
        .expect("CUDA prefill failed");

    assert_f32_close("prefill_8tok", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_prefill_matches_token_at_a_time_16_tokens() {
    let (provider, cpu, cuda) = setup_backends().expect("failed to set up backends");
    let prompt: Vec<u32> = (0..16).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt)
        .expect("CPU prefill failed");

    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    let cuda_hidden = cuda
        .prefill(&prompt, &provider, &mut kv)
        .expect("CUDA prefill failed");

    assert_f32_close("prefill_16tok", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_prefill_single_token() {
    let (provider, cpu, cuda) = setup_backends().expect("failed to set up backends");
    let prompt = vec![5];

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt)
        .expect("CPU prefill failed");

    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    let cuda_hidden = cuda
        .prefill(&prompt, &provider, &mut kv)
        .expect("CUDA prefill failed");

    assert_f32_close("prefill_1tok", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_prefill_empty_prompt_returns_error() {
    let (provider, _cpu, cuda) = setup_backends().expect("failed to set up backends");
    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    let result = cuda.prefill(&[], &provider, &mut kv);
    assert!(result.is_err(), "empty prompt should return error");
}

#[test]
fn test_prefill_kv_cache_advances_correctly() {
    let (provider, _cpu, cuda) = setup_backends().expect("failed to set up backends");
    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;
    let prompt: Vec<u32> = (0..6).collect();

    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 64,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap();

    assert_eq!(kv.seq_len(), 0);

    cuda.prefill(&prompt, &provider, &mut kv)
        .expect("CUDA prefill failed");

    // After prefill of 6 tokens, seq_len should be 6.
    assert_eq!(
        kv.seq_len(),
        6,
        "KV cache seq_len should equal number of prefilled tokens"
    );
}

#[test]
fn test_prefill_caps_reports_true() {
    let (_provider, _cpu, cuda) = setup_backends().expect("failed to set up backends");
    let caps = cuda.caps();
    assert!(
        caps.batched_prefill,
        "CUDA backend should report batched_prefill=true"
    );
}

#[test]
fn test_engine_uses_prefill_path() {
    // Verify the engine dispatches through the prefill path (not token-at-a-time)
    // when the backend reports batched_prefill=true.
    let (provider, _cpu, cuda) = setup_backends().expect("failed to set up backends");
    let hp = provider.lbc().header.hyperparams;

    let engine = InferenceEngine::new(
        RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        },
        hp,
    );

    let prompt = vec![0, 1, 2, 3];
    let stop = StopCondition::MaxTokens(3);
    let sampling = SamplingParams {
        temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    };

    let result = engine
        .generate(&prompt, &provider, &cuda, &stop, &sampling)
        .expect("generate failed");

    // Verify we got 3 generated tokens.
    assert_eq!(
        result.tokens.len(),
        3,
        "should generate exactly 3 tokens"
    );

    // Verify metrics report prompt tokens.
    assert_eq!(
        result.metrics.prompt_tokens,
        4,
        "metrics should report 4 prompt tokens"
    );
}
