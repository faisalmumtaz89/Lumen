//! Slice boundaries do not change what a Metal prefill computes: a prompt gives the
//! same hidden state whether it arrives in one call or in two calls cut at another
//! point, since the KV cache carries every earlier token either way. Lengths sit on
//! both sides of the 2,048-token slice. The GDN recurrent state across slices is
//! covered on a real model by `metal_warm_append_test`.
//!
//!   cargo test --release -p lumen-runtime --test metal_prefill_slices_test
#![cfg(target_os = "macos")]

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{ComputeBackend, MetalF32Backend};

const CONTEXT: usize = 8192;
const CASES: [(usize, usize); 6] = [
    (2048, 700),
    (2049, 1000),
    (2048 + 5, 2040),
    (2048 + 300, 700),
    (4096, 2048),
    (4097, 2500),
];

#[test]
fn prefill_result_does_not_depend_on_slice_boundaries() {
    let lbc = generate_test_model(&TestModelConfig {
        max_seq_len: CONTEXT as u32,
        ..TestModelConfig::default()
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.lbc");
    std::fs::write(&path, lbc).unwrap();
    let provider = SyncWeightProvider::open(&path).expect("open provider");
    let hp = provider.lbc().header.hyperparams;
    let backend = || {
        let mut metal = MetalF32Backend::new().expect("Metal");
        metal.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        metal.init(&hp).expect("init");
        metal
    };
    let kv = || {
        KvCache::new(KvCacheConfig {
            max_seq_len: CONTEXT,
            num_layers: hp.num_layers as usize,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: KvPrecision::F32,
        })
        .unwrap()
    };
    let (metal_a, metal_b) = (backend(), backend());
    for (total, cut) in CASES {
        let prompt: Vec<u32> = (0..total)
            .map(|i| ((i * 7919 + 13) % hp.vocab_size as usize) as u32)
            .collect();
        let one_call = || {
            let mut kv = kv();
            let hidden = metal_a
                .prefill(&prompt, &provider, &mut kv)
                .unwrap_or_else(|e| panic!("prefill of {total} tokens in one call: {e}"));
            assert_eq!(kv.seq_len(), total);
            hidden
        };
        let first = one_call();
        assert_eq!(
            first,
            one_call(),
            "prefill of {total} tokens is not deterministic"
        );

        let mut kv_b = kv();
        metal_b
            .prefill(&prompt[..cut], &provider, &mut kv_b)
            .unwrap_or_else(|e| panic!("prefill of {total} tokens, first {cut}: {e}"));
        let two_calls = metal_b
            .prefill(&prompt[cut..], &provider, &mut kv_b)
            .unwrap_or_else(|e| panic!("prefill of {total} tokens, the rest after {cut}: {e}"));
        assert_eq!(kv_b.seq_len(), total);
        assert_eq!(
            first, two_calls,
            "prefill of {total} tokens differs from the same prompt cut at {cut}"
        );
    }
}
