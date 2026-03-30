//! CUDA compute_layer and compute_final integration tests.
//!
//! Compares CUDA backend output against the CPU naive backend for a synthetic
//! test model. Both backends process the same deterministic weights and inputs,
//! so their outputs should match within floating-point tolerance.
//!
//! These tests require a CUDA-capable GPU. They are gated behind
//! `--features cuda` and will fail on macOS (no NVIDIA GPU).

#![cfg(feature = "cuda")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
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
    let dir = std::env::temp_dir().join(format!("lumen_cuda_layer_test_{id}"));
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
    assert!(
        mismatches == 0,
        "{label}: {mismatches}/{} elements exceed tolerance {tolerance:.1e} (max_diff={max_diff:.2e})",
        actual.len()
    );
    eprintln!("  {label}: max_diff={max_diff:.2e} (within tolerance {tolerance:.1e})");
}

#[test]
fn test_cuda_compute_final_matches_cpu() {
    let (provider, cpu, cuda) = match setup_backends() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA test (no GPU?): {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;

    // Create a deterministic activation buffer (embed token 0).
    let x = cpu.embed_token(0).unwrap();

    // CPU compute_final
    let cpu_logits = cpu.compute_final(&x).unwrap();

    // CUDA compute_final
    let cuda_logits = cuda.compute_final(&x).unwrap();

    eprintln!("=== compute_final comparison ===");
    assert_eq!(
        cpu_logits.data.len(),
        cuda_logits.data.len(),
        "logits length mismatch"
    );
    assert_eq!(
        cpu_logits.data.len(),
        hp.vocab_size as usize,
        "logits should have vocab_size elements"
    );

    assert_f32_close("logits", &cuda_logits.data, &cpu_logits.data, 1e-3);

    // Verify argmax matches (most important: same top prediction).
    let cpu_argmax = cpu_logits.argmax();
    let cuda_argmax = cuda_logits.argmax();
    assert_eq!(
        cpu_argmax, cuda_argmax,
        "compute_final argmax mismatch: CPU={cpu_argmax}, CUDA={cuda_argmax}"
    );
    eprintln!("  argmax: CPU={cpu_argmax}, CUDA={cuda_argmax} (match)");
}

#[test]
fn test_cuda_compute_layer_matches_cpu() {
    let (provider, cpu, cuda) = match setup_backends() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA test (no GPU?): {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;
    let hidden_dim = hp.hidden_dim as usize;
    let num_layers = hp.num_layers as usize;

    // Create KV caches for both backends.
    let kv_config = KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let mut cpu_kv = KvCache::new(kv_config.clone()).unwrap();
    let mut cuda_kv = KvCache::new(kv_config).unwrap();

    // Process token 0 through both backends.
    let mut cpu_x = cpu.embed_token(0).unwrap();
    let mut cuda_x = cuda.embed_token(0).unwrap();

    eprintln!("=== compute_layer comparison (token 0) ===");
    eprintln!("  model: {num_layers} layers, hidden_dim={hidden_dim}");

    for layer_idx in 0..num_layers {
        let layer_view = provider.get_layer_blocking(layer_idx).unwrap();
        let seq_pos = cpu_kv.seq_len();

        // CPU compute_layer
        {
            let mut kv_view = cpu_kv.view_mut(layer_idx).unwrap();
            cpu.compute_layer(layer_idx, &mut cpu_x, &layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cpu_kv.commit_view(kv_view).unwrap();
        }

        // CUDA compute_layer
        {
            let mut kv_view = cuda_kv.view_mut(layer_idx).unwrap();
            cuda.compute_layer(layer_idx, &mut cuda_x, &layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cuda_kv.commit_view(kv_view).unwrap();
        }

        // Compare layer outputs.
        let cpu_vals = cpu_x.as_f32_slice();
        let cuda_vals = cuda_x.as_f32_slice();
        assert_f32_close(
            &format!("layer_{layer_idx}"),
            cuda_vals,
            cpu_vals,
            1e-3,
        );
    }

    // Advance KV cache seq_len for both.
    cpu_kv.advance_seq_len().unwrap();
    cuda_kv.advance_seq_len().unwrap();

    // Run compute_final on both and compare.
    let cpu_logits = cpu.compute_final(&cpu_x).unwrap();
    let cuda_logits = cuda.compute_final(&cuda_x).unwrap();

    eprintln!("=== compute_final after layers ===");
    assert_f32_close("final_logits", &cuda_logits.data, &cpu_logits.data, 1e-3);

    let cpu_argmax = cpu_logits.argmax();
    let cuda_argmax = cuda_logits.argmax();
    assert_eq!(
        cpu_argmax, cuda_argmax,
        "final argmax mismatch: CPU={cpu_argmax}, CUDA={cuda_argmax}"
    );
    eprintln!("  final argmax: CPU={cpu_argmax}, CUDA={cuda_argmax} (match)");
}

#[test]
fn test_cuda_multi_token_decode() {
    let (provider, cpu, cuda) = match setup_backends() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA test (no GPU?): {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;
    let num_layers = hp.num_layers as usize;

    let kv_config = KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let mut cpu_kv = KvCache::new(kv_config.clone()).unwrap();
    let mut cuda_kv = KvCache::new(kv_config).unwrap();

    eprintln!("=== multi-token decode (3 tokens) ===");

    // Process 3 tokens to verify KV cache accumulation works correctly.
    let tokens = [0u32, 1, 2];
    for (tok_idx, &token_id) in tokens.iter().enumerate() {
        let mut cpu_x = cpu.embed_token(token_id).unwrap();
        let mut cuda_x = cuda.embed_token(token_id).unwrap();

        for layer_idx in 0..num_layers {
            let layer_view = provider.get_layer_blocking(layer_idx).unwrap();
            let seq_pos = cpu_kv.seq_len();

            {
                let mut kv_view = cpu_kv.view_mut(layer_idx).unwrap();
                cpu.compute_layer(layer_idx, &mut cpu_x, &layer_view, Some(&mut kv_view), seq_pos)
                    .unwrap();
                cpu_kv.commit_view(kv_view).unwrap();
            }

            {
                let mut kv_view = cuda_kv.view_mut(layer_idx).unwrap();
                cuda.compute_layer(layer_idx, &mut cuda_x, &layer_view, Some(&mut kv_view), seq_pos)
                    .unwrap();
                cuda_kv.commit_view(kv_view).unwrap();
            }
        }

        cpu_kv.advance_seq_len().unwrap();
        cuda_kv.advance_seq_len().unwrap();

        let cpu_logits = cpu.compute_final(&cpu_x).unwrap();
        let cuda_logits = cuda.compute_final(&cuda_x).unwrap();

        assert_f32_close(
            &format!("token_{tok_idx}_logits"),
            &cuda_logits.data,
            &cpu_logits.data,
            1e-3,
        );

        let cpu_argmax = cpu_logits.argmax();
        let cuda_argmax = cuda_logits.argmax();
        assert_eq!(
            cpu_argmax, cuda_argmax,
            "token {tok_idx} argmax mismatch: CPU={cpu_argmax}, CUDA={cuda_argmax}"
        );
        eprintln!("  token {tok_idx}: argmax CPU={cpu_argmax}, CUDA={cuda_argmax} (match)");
    }
}
