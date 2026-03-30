//! CUDA GEMM kernel tests for the batched prefill path.
//!
//! Exercises `gemm_f32` and `gemm_f32_residual` with various M, N, K dimensions
//! including non-multiples of the 32x32 tile size. Verifies correctness against
//! a CPU reference implementation.
//!
//! These tests require a CUDA-capable GPU. They are gated behind
//! `--features cuda` and will fail on macOS (no NVIDIA GPU).

#![cfg(feature = "cuda")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend};
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

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

/// Set up backends with a custom TestModelConfig.
fn setup_backends_custom(
    config: &TestModelConfig,
) -> Result<(SyncWeightProvider, NaiveF32Backend, CudaBackend), RuntimeError> {
    let lbc_data = generate_test_model(config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_gemm_test_{id}"));
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
        max_seq_len: 256,
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

/// Helper: run CUDA prefill and return hidden state.
fn cuda_prefill(
    provider: &SyncWeightProvider,
    cuda: &CudaBackend,
    prompt_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;

    let mut kv = KvCache::new(KvCacheConfig {
        max_seq_len: 256,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })?;

    cuda.prefill(prompt_tokens, provider, &mut kv)
}

// ---------- Test: default config (hidden=8, head=4, inter=16, vocab=32) ----------
// All dimensions are non-multiples of 32. This is the primary regression test
// for the MISALIGNED_ADDRESS bug: the GEMM kernel must correctly zero-pad
// shared memory tiles when M, N, K < 32.

#[test]
fn test_gemm_default_config_batch_1() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt = vec![0];

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("default_batch1", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_gemm_default_config_batch_4() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt = vec![0, 1, 2, 3];

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("default_batch4", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: batch > 32 (M not a multiple of TILE_M) ----------
// With batch=33, the GEMM grid has 2 tile-rows. The second tile covers
// rows 32..63, but only row 32 is valid (M=33). Threads for rows 33-63
// must produce zeros and NOT write to C.

#[test]
fn test_gemm_batch_33_non_tile_multiple() {
    let config = TestModelConfig {
        vocab_size: 64, // need at least 33 distinct tokens
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..33).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("batch33", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: batch exactly 32 (tile-aligned) ----------

#[test]
fn test_gemm_batch_32_tile_aligned() {
    let config = TestModelConfig {
        vocab_size: 64,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..32).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("batch32", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: dimensions that are multiples of 32 ----------
// hidden_dim=32, head_dim=16, inter=64 -- all tile-aligned.
// This verifies the kernel works correctly in the "happy path".

#[test]
fn test_gemm_tile_aligned_dimensions() {
    let config = TestModelConfig {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 64,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..8).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("aligned_dims", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: large batch (pp128) with small dimensions ----------
// 128 tokens with hidden_dim=8. GEMM grid: (1, 4). Four tile-rows,
// all fully utilized. Tests that multi-row grids work correctly.

#[test]
fn test_gemm_pp128_small_model() {
    let config = TestModelConfig {
        vocab_size: 256,
        max_seq_len: 256,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..128).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("pp128_small", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: asymmetric Q vs KV dimensions ----------
// num_heads=4, num_kv_heads=1: q_dim=64, kv_dim=16. The WQ GEMM has
// N=64 (2 tiles), while WK/WV GEMM has N=16 (1 partial tile).
// This tests that different N dimensions within the same prefill call
// all produce correct results.

#[test]
fn test_gemm_asymmetric_qkv() {
    let config = TestModelConfig {
        num_heads: 4,
        num_kv_heads: 1,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 64,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..8).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("asym_qkv", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: odd dimensions (primes) ----------
// hidden_dim=13, head_dim=13, inter_dim=17. None of these are multiples
// of 32, and none are even multiples of 4. Stress-tests the zero-padding
// logic in the GEMM kernel.

#[test]
fn test_gemm_prime_dimensions() {
    let config = TestModelConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 13,
        hidden_dim: 13,
        intermediate_dim: 17,
        vocab_size: 32,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..7).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("prime_dims", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: batch=1 with tile-aligned dimensions ----------
// M=1, N=32, K=32. Only 1 row of the output tile is valid.
// All other rows (1-31) should produce no writes.

#[test]
fn test_gemm_single_token_aligned() {
    let config = TestModelConfig {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 64,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt = vec![5];

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("single_aligned", &cuda_hidden, &cpu_hidden, 1e-4);
}

// ---------- Test: residual GEMM correctness ----------
// The output projection (wo) uses gemm_f32_residual which adds the
// input residual. Run with different batch sizes to exercise the
// residual path at various grid configurations.

#[test]
fn test_gemm_residual_batch_5() {
    let config = TestModelConfig::default();
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..5).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("residual_batch5", &cuda_hidden, &cpu_hidden, 1e-4);
}

#[test]
fn test_gemm_residual_batch_17() {
    let config = TestModelConfig {
        vocab_size: 64,
        ..TestModelConfig::default()
    };
    let (provider, cpu, cuda) = setup_backends_custom(&config).expect("setup failed");
    let prompt: Vec<u32> = (0..17).collect();

    let cpu_hidden = cpu_prefill_reference(&provider, &cpu, &prompt).expect("CPU failed");
    let cuda_hidden = cuda_prefill(&provider, &cuda, &prompt).expect("CUDA failed");

    assert_f32_close("residual_batch17", &cuda_hidden, &cpu_hidden, 1e-4);
}
