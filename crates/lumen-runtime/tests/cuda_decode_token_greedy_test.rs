//! Differential test: `decode_token_greedy(t)` MUST return the same token
//! IDs as `argmax(decode_token(t).logits)` for the same input token stream.
//!
//! This proves the new `decode_token_greedy` method (returns u32 directly via
//! GPU argmax) is behaviourally equivalent to the existing `decode_token`
//! method (returns Logits Vec, with the engine doing CPU argmax). Both run
//! the same kernels and the same GPU argmax internally; the only difference
//! is the final step where greedy skips the one-hot Vec allocation and
//! returns u32 directly.
//!
//! Method: spin up TWO preloaded CUDA backends on the same synthetic model.
//! Drive backend A through `decode_token` + CPU argmax. Drive backend B
//! through `decode_token_greedy`. Feed the previous token back as the next
//! input on each side (mirroring the engine.generate() greedy loop). Assert
//! the 50-token sequences match byte-for-byte.
//!
//! Gated behind `--features cuda`. Skips gracefully when no NVIDIA GPU.

#![cfg(feature = "cuda")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write synthetic LBC to a temp file and return the provider.
fn write_and_open(lbc_data: &[u8], label: &str) -> SyncWeightProvider {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_greedy_{label}_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(lbc_data).unwrap();
    }
    SyncWeightProvider::open(&path).unwrap()
}

/// Try to create a CudaBackend, returning None when no NVIDIA GPU.
fn try_cuda_backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("Skipping: no CUDA GPU available: {e}");
            None
        }
    }
}

/// Build a preloaded CUDA backend bound to `provider`.
fn make_preloaded_backend(provider: &SyncWeightProvider) -> Option<CudaBackend> {
    let hp = provider.lbc().header.hyperparams;
    let mut backend = try_cuda_backend()?;
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&hp).expect("init");
    backend.preload_weights(provider).expect("preload");
    Some(backend)
}

/// CPU argmax over a Logits Vec; returns 0 on empty input.
fn cpu_argmax(values: &[f32]) -> u32 {
    let mut best_idx: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Differential test: `decode_token_greedy` MUST agree with
/// `argmax(decode_token().data)` on every step of a 50-token decode loop.
///
/// Two preloaded CUDA backends are seeded with identical weights. We feed
/// the same prompt to both and then drive each through 50 single-token
/// decode steps, using the previous token as the next input (mirroring the
/// engine.generate() greedy loop). After every step, assert the tokens match.
#[test]
fn decode_token_greedy_matches_decode_token_argmax() {
    // Small synthetic model: fast and exercises both prefill + decode paths.
    let config = TestModelConfig {
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 256,
        max_seq_len: 128,
        seed: 42,
    };
    let lbc_data = generate_test_model(&config);
    let provider = write_and_open(&lbc_data, "diff");

    // Two independent preloaded backends with identical state.
    let backend_a = match make_preloaded_backend(&provider) {
        Some(b) => b,
        None => return, // No GPU available
    };
    let backend_b = match make_preloaded_backend(&provider) {
        Some(b) => b,
        None => return,
    };

    // Cap must now advertise gpu_argmax=true on both backends.
    assert!(
        backend_a.caps().gpu_argmax,
        "preloaded CUDA backend must advertise gpu_argmax=true"
    );
    assert!(backend_b.caps().gpu_argmax);

    let hp = provider.lbc().header.hyperparams;
    let max_seq_len = hp.max_seq_len as usize;
    let num_layers = hp.num_layers as usize;
    let num_kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;

    // Two independent KV caches (one per backend).
    let kv_cfg = KvCacheConfig {
        max_seq_len,
        num_layers,
        num_kv_heads,
        head_dim,
        precision: KvPrecision::F32,
    };
    let mut kv_a = KvCache::new(kv_cfg.clone()).expect("kv_a");
    let mut kv_b = KvCache::new(kv_cfg).expect("kv_b");

    // Seed both backends with a tiny prompt by driving them through
    // decode_token / decode_token_greedy from the same starting token.
    // The synthetic test_model is decode-friendly: the first call advances
    // KV and produces an output token, which we feed back next step.
    let mut next_token_a: u32 = 1;
    let mut next_token_b: u32 = 1;

    let num_steps = 50;
    for step in 0..num_steps {
        // Side A: full Logits + CPU argmax.
        let logits = backend_a
            .decode_token(next_token_a, &provider, &mut kv_a)
            .expect("decode_token A");
        let token_a = cpu_argmax(&logits.data);

        // Side B: GPU argmax (4-byte readback).
        let token_b = backend_b
            .decode_token_greedy(next_token_b, &provider, &mut kv_b)
            .expect("decode_token_greedy B");

        assert_eq!(
            token_a, token_b,
            "step {step}: decode_token_greedy must match argmax(decode_token.data)\n\
             argmax(logits)         = {token_a}\n\
             decode_token_greedy()  = {token_b}\n\
             input token A          = {next_token_a}\n\
             input token B          = {next_token_b}"
        );

        // Feed the matched token back for the next step.
        next_token_a = token_a;
        next_token_b = token_b;
    }

    eprintln!(
        "decode_token_greedy_matches_decode_token_argmax: {num_steps} tokens matched"
    );
}

/// Smoke test: when weights are NOT preloaded, the trait method must error
/// (because gpu_argmax cap is gated on `is_preloaded`).
#[test]
fn decode_token_greedy_errors_without_preload() {
    let config = TestModelConfig {
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 32,
        intermediate_dim: 64,
        vocab_size: 256,
        max_seq_len: 128,
        seed: 7,
    };
    let lbc_data = generate_test_model(&config);
    let provider = write_and_open(&lbc_data, "noprealod");
    let hp = provider.lbc().header.hyperparams;

    let mut backend = match try_cuda_backend() {
        Some(b) => b,
        None => return,
    };
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&hp).unwrap();
    // NOTE: deliberately skipping preload_weights here.

    // Without preload, the cap must NOT advertise gpu_argmax.
    assert!(
        !backend.caps().gpu_argmax,
        "unpreloaded CUDA backend must NOT advertise gpu_argmax"
    );

    // Direct call MUST error out (matches decode_token's contract).
    let kv_cfg = KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let mut kv = KvCache::new(kv_cfg).unwrap();
    let res = backend.decode_token_greedy(0, &provider, &mut kv);
    assert!(
        res.is_err(),
        "decode_token_greedy without preload must return Err"
    );
}
