//! CUDA on a real Qwen3.x model (`#[ignore]`): a prefill that continues a sequence whose last
//! tokens were decoded starts from the state those decode steps left. Prefilled to a cut,
//! decoded for a few tokens and prefilled for the rest, the sequence leaves recurrent states
//! within 0.05 (relative L2) of the whole sequence prefilled in one call, the route's rounding
//! (measured on Qwen3.8-27B: up to 1.1e-3 on Q4_0, 2.7e-2 on NVFP4); continued from that point
//! with its recurrent states zeroed, it lands at least four times farther away (measured: 7.4
//! times or more), so the comparison can see the state it covers. The conv rings and the last
//! hidden row are printed alongside. The rests reach past the 2,048-token prefill slice.
//! Requires a CUDA GPU:
//!
//!   LUMEN_WARM_APPEND_MODEL=/path/to/model.lbc cargo test --release -p lumen-runtime \
//!     --features test-state-snapshot --test cuda_warm_append_test -- --ignored --nocapture
#![cfg(feature = "test-state-snapshot")]

use lumen_format::quantization::QuantScheme;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::{CudaBackend, StateSnapshot};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::path::PathBuf;

const CONTEXT: usize = 8192;

/// The backend as `lumen-server` builds it for CUDA, with an F32 KV store (the one the state
/// snapshot reads) and the context capped at `CONTEXT`.
fn backend(provider: &SyncWeightProvider) -> CudaBackend {
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = hp.max_seq_len.min(CONTEXT as u32);
    let mut cuda = CudaBackend::new(0).expect("CUDA device 0");
    cuda.set_kv_precision(KvPrecision::F32).expect("F32 KV");
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    let embedding_raw = matches!(
        provider.embedding_quant,
        QuantScheme::Q8_0
            | QuantScheme::Q4_0
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q4_K
            | QuantScheme::Q5_K
            | QuantScheme::Q6_K
    ) && !provider.embedding_raw.is_empty();
    if embedding_raw {
        cuda.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    let head_raw = matches!(
        provider.output_proj_quant,
        QuantScheme::Q8_0
            | QuantScheme::Q4_0
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q6_K
            | QuantScheme::Nvfp4
    ) && !provider.output_proj_raw.is_empty();
    if head_raw {
        cuda.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if provider.weight_tying {
        cuda.set_weight_tying(true);
    }
    cuda.init(&hp).expect("init");
    cuda.preload_weights(provider).expect("preload");
    cuda
}

fn kv_cache(provider: &SyncWeightProvider) -> KvCache {
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

/// Token ids of an ordinary vocabulary range.
fn tokens(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (1000 + (i * 7919 + 104_729) % 90_000) as u32)
        .collect()
}

/// Relative L2 distance of `got` from `want`, over every value of every slice.
fn rel_l2<'a>(
    got: impl Iterator<Item = &'a Vec<f32>>,
    want: impl Iterator<Item = &'a Vec<f32>>,
) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (g, w) in got.zip(want) {
        for (&x, &y) in g.iter().zip(w) {
            d += f64::from(x - y).powi(2);
            n += f64::from(y).powi(2);
        }
    }
    (d / n).sqrt()
}

/// Distances of the recurrent states, the conv rings and the last hidden row.
fn distances(got: &StateSnapshot, want: &StateSnapshot) -> [f64; 3] {
    assert_eq!(got.seq_len, want.seq_len);
    assert_eq!(got.conv_positions, want.conv_positions);
    [
        rel_l2(got.h_states.iter(), want.h_states.iter()),
        rel_l2(got.conv_states.iter(), want.conv_states.iter()),
        rel_l2([&got.x].into_iter(), [&want.x].into_iter()),
    ]
}

#[test]
#[ignore]
fn a_prefill_continues_the_state_decode_left() {
    let path = PathBuf::from(
        std::env::var("LUMEN_WARM_APPEND_MODEL").expect("set LUMEN_WARM_APPEND_MODEL to an LBC"),
    );
    let provider = SyncWeightProvider::open(&path).expect("open the model");
    let (a, b) = (backend(&provider), backend(&provider));
    let all = tokens(CONTEXT);
    for (cut, decoded, rest) in [(600, 1, 40), (600, 64, 400), (2047, 2, 2100)] {
        let total = cut + decoded + rest;
        let tag = format!("prefill {cut}, decode {decoded}, prefill {rest}");

        let mut kv_a = kv_cache(&provider);
        a.prefill(&all[..total], &provider, &mut kv_a)
            .expect("prefill in one call");
        let one_call = a.snapshot_state(&kv_a).expect("snapshot");
        a.reset_recurrent_state();

        let mut kv_b = kv_cache(&provider);
        b.prefill(&all[..cut], &provider, &mut kv_b)
            .expect("prefill to the cut");
        for &t in &all[cut..cut + decoded] {
            b.decode_token(t, &provider, &mut kv_b).expect("decode");
        }
        let mut zeroed = b.snapshot_state(&kv_b).expect("snapshot");
        b.prefill(&all[cut + decoded..total], &provider, &mut kv_b)
            .expect("prefill of the rest");
        let continued = b.snapshot_state(&kv_b).expect("snapshot");
        b.reset_recurrent_state();

        zeroed.h_states.iter_mut().for_each(|h| h.fill(0.0));
        let mut kv_b = kv_cache(&provider);
        b.restore_state(&zeroed, &mut kv_b).expect("restore");
        b.prefill(&all[cut + decoded..total], &provider, &mut kv_b)
            .expect("prefill of the rest from zeroed recurrent states");
        let from_zero = b.snapshot_state(&kv_b).expect("snapshot");
        b.reset_recurrent_state();

        let near = distances(&continued, &one_call);
        let far = distances(&from_zero, &one_call);
        let show = |d: [f64; 3]| format!("{:.3e} {:.3e} {:.3e}", d[0], d[1], d[2]);
        println!(
            "{tag}: h, conv, x rel L2 {}; from zeroed recurrent states {}",
            show(near),
            show(far)
        );
        assert!(
            near[0] <= 0.05,
            "{tag}: recurrent states {:.3e} from one call",
            near[0]
        );
        assert!(
            far[0] >= 4.0 * near[0],
            "{tag}: zeroed recurrent states land {:.3e} away, continued ones {:.3e}",
            far[0],
            near[0]
        );
    }
}
