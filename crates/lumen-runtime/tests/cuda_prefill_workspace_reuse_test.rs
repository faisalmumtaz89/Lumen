//! The batched prefill's workspace and the grouped MoE prefill scratch are
//! allocated once at load and lent to every request, so nothing a request
//! computes may depend on what an earlier request left in them. Here every
//! buffer of them is filled with 0xFF bytes (NaN as F32) before each request,
//! after a longer and a shorter one, and the hidden state, the KV cache and the
//! GDN state must match a fresh backend's bit for bit. Each model states which
//! optional parts (score block, Q+gate pair, grouped MoE scratch) it has, so a
//! route the test names cannot drop out unseen.
#![cfg(all(feature = "cuda", feature = "test-state-snapshot"))]

mod common;

use common::gdn_hybrid::{
    build_gdn_hybrid_lbc_with, build_qwen35_cpu_lbc, gdn_model_hyperparams_with, Qwen35Fixture,
};
use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_format::QuantScheme;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::{CudaBackend, PoisonedParts, StateSnapshot};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const CONTEXT: usize = 8192;

fn open(lbc: &[u8], label: &str) -> SyncWeightProvider {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "lumen_prefill_reuse_{label}_{}_{id}",
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

fn kv(provider: &SyncWeightProvider) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Everything a request leaves: its hidden output and the decode state, as bits.
fn outcome(hidden: &[f32], snap: &StateSnapshot) -> Vec<Vec<u32>> {
    let mut all = vec![bits(hidden), bits(&snap.x)];
    for (k, v) in &snap.kv {
        all.push(bits(k));
        all.push(bits(v));
    }
    all.extend(snap.h_states.iter().map(|h| bits(h)));
    all.extend(snap.conv_states.iter().map(|c| bits(c)));
    all.push(snap.conv_positions.clone());
    all
}

fn prompt(len: usize, vocab: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 7919 + 13 + seed) % vocab) as u32)
        .collect()
}

/// `parts`: the optional workspace parts the model must have, so the poisoning
/// covers the routes the test names.
fn check(lbc: &[u8], label: &str, parts: PoisonedParts) {
    let provider = open(lbc, label);
    let Some(reused) = backend(&provider) else {
        return;
    };
    let vocab = provider.lbc().header.hyperparams.vocab_size as usize;
    // Longer than one slice, then short, then a one-slice prompt, then one token.
    for (len, seed) in [(3000, 0), (17, 1), (2048, 2), (1, 3)] {
        let p = prompt(len, vocab, seed);
        // A backend whose workspace no request has touched: the reference.
        let fresh = backend(&provider).unwrap();
        let mut kv_fresh = kv(&provider);
        let h = fresh
            .prefill(&p, &provider, &mut kv_fresh)
            .unwrap_or_else(|e| panic!("{label}: fresh prefill of {len}: {e}"));
        let want = outcome(&h, &fresh.snapshot_state(&kv_fresh).unwrap());

        assert_eq!(
            reused.poison_prefill_workspace().unwrap(),
            parts,
            "{label}: the workspace parts the poisoning filled"
        );
        let mut kv_reused = kv(&provider);
        reused.reset_recurrent_state();
        let h = reused
            .prefill(&p, &provider, &mut kv_reused)
            .unwrap_or_else(|e| panic!("{label}: prefill of {len} after poisoning: {e}"));
        let got = outcome(&h, &reused.snapshot_state(&kv_reused).unwrap());
        assert!(
            got == want,
            "{label}: a {len}-token prefill read a workspace row it did not write"
        );
        println!("{label}: {len}-token prefill identical after poisoning");
    }
}

#[test]
fn an_attention_model_reads_no_stale_workspace_row() {
    let config = TestModelConfig {
        max_seq_len: CONTEXT as u32,
        ..TestModelConfig::default()
    };
    check(
        &generate_test_model(&config),
        "attention",
        PoisonedParts::default(),
    );
}

#[test]
fn a_gdn_hybrid_reads_no_stale_workspace_row() {
    check(
        &build_gdn_hybrid_lbc_with(gdn_model_hyperparams_with(CONTEXT as u32)),
        "gdn-hybrid",
        PoisonedParts::default(),
    );
}

/// Every optional part of the workspace a Qwen3.5-shaped model has.
const QWEN35_PARTS: PoisonedParts = PoisonedParts {
    score_block: true,
    q_gate: true,
    grouped_moe: true,
};

/// Gated attention with Q/K norms (the tiled attention's full-context score
/// block and the Q+gate pair) and MoE layers of 64 Q8_0 experts (the grouped
/// MoE route), the other matrices F32.
#[test]
fn a_qwen35_shaped_model_reads_no_stale_workspace_row() {
    let fx = Qwen35Fixture {
        expert_quant: Some(QuantScheme::Q8_0),
        routed_experts: Some(64),
        ..Qwen35Fixture::default()
    };
    check(
        &build_qwen35_cpu_lbc(&fx, CONTEXT as u32),
        "qwen35",
        QWEN35_PARTS,
    );
}

/// The same model with its other matrices in the quantised schemes, so the
/// dequant scratch is exercised too.
#[test]
fn a_quantised_qwen35_shaped_model_reads_no_stale_workspace_row() {
    let fx = Qwen35Fixture {
        mixed_quant: true,
        expert_quant: Some(QuantScheme::Q8_0),
        routed_experts: Some(64),
        ..Qwen35Fixture::default()
    };
    check(
        &build_qwen35_cpu_lbc(&fx, CONTEXT as u32),
        "qwen35-quant",
        QWEN35_PARTS,
    );
}
