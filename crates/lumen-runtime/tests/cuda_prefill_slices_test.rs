//! Slice boundaries do not change what a prefill computes: a prompt gives the
//! same hidden state whether it arrives in one call or in two calls cut at
//! another point, since the KV cache and the GDN recurrent state carry every
//! earlier token either way. Lengths sit on both sides of the 2,048-token
//! slice, with tails of one token, five tokens (the scalar attention path) and
//! a full slice, on an attention-only model and on the GDN/attention hybrid.
//!
//! Its own binary because the prefill attention route is chosen once per
//! process: the tests pin the exact-F32 route production takes
//! (`LUMEN_CUDA_ATTN_PRECISE=3`) before any backend exists. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_prefill_slices_test
#![cfg(feature = "cuda")]

mod common;

use common::gdn_hybrid::{build_gdn_hybrid_lbc_with, gdn_model_hyperparams_with};
use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const CONTEXT: usize = 8192;
const CASES: [(usize, usize); 5] = [
    (2048, 700),
    (2049, 1000),
    (2048 + 5, 2040),
    (2048 + 300, 700),
    (4097, 2500),
];

fn open(lbc: &[u8], label: &str) -> SyncWeightProvider {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "lumen_prefill_slices_{label}_{}_{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.lbc");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(lbc)
        .unwrap();
    SyncWeightProvider::open(&path).expect("open provider")
}

fn backend(provider: &SyncWeightProvider) -> Option<CudaBackend> {
    let hp = provider.lbc().header.hyperparams;
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: no CUDA GPU: {e}");
            return None;
        }
    };
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp).expect("init");
    cuda.preload_weights(provider).expect("preload");
    Some(cuda)
}

/// Every case in one call and in two calls, on the given model bytes.
fn check(lbc: &[u8], label: &str) {
    std::env::set_var("LUMEN_CUDA_ATTN_PRECISE", "3");
    let provider = open(lbc, label);
    let (Some(cuda_a), Some(cuda_b)) = (backend(&provider), backend(&provider)) else {
        return;
    };
    assert_eq!(
        lumen_runtime::runtime_defaults::attn_precise_selected(),
        3,
        "the exact-F32 attention route"
    );
    let hp = provider.lbc().header.hyperparams;
    let kv_cfg = KvCacheConfig {
        max_seq_len: CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let rel_l2 = |a: &[f32], b: &[f32]| {
        let d = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        d / a.iter().map(|&x| x * x).sum::<f32>().sqrt()
    };
    for (total, cut) in CASES {
        let prompt: Vec<u32> = (0..total)
            .map(|i| ((i * 7919 + 13) % hp.vocab_size as usize) as u32)
            .collect();
        let one_call = |cuda: &CudaBackend| {
            let mut kv = KvCache::new(kv_cfg.clone()).unwrap();
            let hidden = cuda
                .prefill(&prompt, &provider, &mut kv)
                .unwrap_or_else(|e| panic!("{label}: prefill of {total} tokens in one call: {e}"));
            assert_eq!(kv.seq_len(), total);
            cuda.reset_recurrent_state();
            hidden
        };
        let first = one_call(&cuda_a);
        let again = one_call(&cuda_a);
        assert_eq!(
            first, again,
            "{label}: prefill of {total} tokens is not deterministic"
        );

        let mut kv_b = KvCache::new(kv_cfg.clone()).unwrap();
        cuda_b
            .prefill(&prompt[..cut], &provider, &mut kv_b)
            .unwrap_or_else(|e| panic!("{label}: prefill of {total} tokens, first {cut}: {e}"));
        let two_calls = cuda_b
            .prefill(&prompt[cut..], &provider, &mut kv_b)
            .unwrap_or_else(|e| {
                panic!("{label}: prefill of {total} tokens, the rest after {cut}: {e}")
            });
        assert_eq!(kv_b.seq_len(), total);
        cuda_b.reset_recurrent_state();

        let distance = rel_l2(&first, &two_calls);
        println!(
            "{label}: prefill {total} tokens, one call vs cut at {cut}: rel L2 {distance:.3e}"
        );
        assert!(
            distance <= 1e-3,
            "{label}: prefill of {total} tokens cut at {cut}: rel L2 {distance:.3e} > 1e-3"
        );
    }
}

#[test]
fn attention_model_prefill_result_does_not_depend_on_slice_boundaries() {
    let config = TestModelConfig {
        max_seq_len: CONTEXT as u32,
        ..TestModelConfig::default()
    };
    check(&generate_test_model(&config), "attention");
}

#[test]
fn gdn_hybrid_prefill_result_does_not_depend_on_slice_boundaries() {
    check(
        &build_gdn_hybrid_lbc_with(gdn_model_hyperparams_with(CONTEXT as u32)),
        "gdn-hybrid",
    );
}
