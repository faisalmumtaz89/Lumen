//! A bit-for-bit record of what the batched prefill leaves, so a change to how
//! the prefill stores its intermediates can be shown to leave every bit as it
//! was. For each model, KV precision and prompt length the test prefills a
//! prompt, decodes four tokens and prefills a second prompt after them, and
//! digests the hidden states, the logits and, on an F32 KV store (the one
//! state snapshots cover), the KV cache and the GDN state at each step; on a
//! 16-bit store the decode logits, which read the cache, stand for it. `LUMEN_PREFILL_ORACLE=capture` writes the digests to
//! `LUMEN_PREFILL_ORACLE_FILE`; `check` compares them with that file and names
//! every case that differs. `LUMEN_PREFILL_ORACLE_LAYOUT=unshared` prefills
//! through a workspace that gives every value its own region, so a record
//! captured that way checks the shared layout against it.
#![cfg(all(feature = "cuda", feature = "test-state-snapshot"))]

mod common;

use common::gdn_hybrid::{
    build_gdn_hybrid_lbc_with, build_qwen35_cpu_lbc, gdn_model_hyperparams_with, Qwen35Fixture,
};
use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_format::QuantScheme;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::{CudaBackend, StateSnapshot};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::io::Write;

const CONTEXT: usize = 8192;
/// Around the batched kernels' 16-row tile, the 512-row attention block and
/// the 2048-row slice, and one prompt of several slices.
const LENGTHS: [usize; 9] = [15, 16, 511, 512, 513, 2047, 2048, 2049, 3000];
const CONTINUATION: usize = 513;

fn provider(lbc: &[u8], label: &str) -> SyncWeightProvider {
    let dir = std::env::temp_dir().join(format!(
        "lumen_prefill_oracle_{label}_{}",
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

fn backend(provider: &SyncWeightProvider, precision: KvPrecision) -> CudaBackend {
    let hp = provider.lbc().header.hyperparams;
    let mut cuda = CudaBackend::new(0).expect("a CUDA GPU");
    cuda.set_kv_precision(precision).unwrap();
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp).expect("init");
    if std::env::var("LUMEN_PREFILL_ORACLE_LAYOUT").as_deref() == Ok("unshared") {
        cuda.set_unshared_prefill_workspace()
            .expect("before preload");
    }
    cuda.preload_weights(provider).expect("preload");
    cuda
}

fn kv(provider: &SyncWeightProvider, precision: KvPrecision) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision,
    })
    .unwrap()
}

fn prompt(len: usize, vocab: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 7919 + 13 + seed) % vocab) as u32)
        .collect()
}

fn digest_f32(h: &mut DefaultHasher, v: &[f32]) {
    h.write_usize(v.len());
    for x in v {
        h.write_u32(x.to_bits());
    }
}

fn digest_state(h: &mut DefaultHasher, snap: &StateSnapshot) {
    digest_f32(h, &snap.x);
    for (k, v) in &snap.kv {
        digest_f32(h, k);
        digest_f32(h, v);
    }
    for s in snap.h_states.iter().chain(&snap.conv_states) {
        digest_f32(h, s);
    }
    for p in &snap.conv_positions {
        h.write_u32(*p);
    }
}

/// One line per step of one case: `<case> <step> <digest>`.
fn run(lbc: &[u8], label: &str, lines: &mut Vec<String>) {
    let provider = provider(lbc, label);
    let vocab = provider.lbc().header.hyperparams.vocab_size as usize;
    for precision in [KvPrecision::F32, KvPrecision::Bf16, KvPrecision::F16] {
        for (n, &len) in LENGTHS.iter().enumerate() {
            let case = format!("{label}/{precision:?}/{len}");
            let cuda = backend(&provider, precision);
            let mut kv = kv(&provider, precision);
            let mut step = |name: &str, f: &dyn Fn(&mut DefaultHasher)| {
                let mut h = DefaultHasher::new();
                f(&mut h);
                lines.push(format!("{case} {name} {:016x}", h.finish()));
            };
            let hidden = cuda
                .prefill(&prompt(len, vocab, n), &provider, &mut kv)
                .unwrap_or_else(|e| panic!("{case}: prefill: {e}"));
            let snap = (precision == KvPrecision::F32).then(|| cuda.snapshot_state(&kv).unwrap());
            step("prefill", &|h| {
                digest_f32(h, &hidden);
                if let Some(snap) = &snap {
                    digest_state(h, snap);
                }
            });
            let mut token = (len % vocab) as u32;
            for d in 0..4 {
                let logits = cuda
                    .decode_token(token, &provider, &mut kv)
                    .unwrap_or_else(|e| panic!("{case}: decode {d}: {e}"));
                step(&format!("decode{d}"), &|h| digest_f32(h, &logits.data));
                token = (token + 1) % vocab as u32;
            }
            let hidden = cuda
                .prefill(&prompt(CONTINUATION, vocab, n + 100), &provider, &mut kv)
                .unwrap_or_else(|e| panic!("{case}: continuation: {e}"));
            let snap = (precision == KvPrecision::F32).then(|| cuda.snapshot_state(&kv).unwrap());
            step("continuation", &|h| {
                digest_f32(h, &hidden);
                if let Some(snap) = &snap {
                    digest_state(h, snap);
                }
            });
        }
    }
}

#[test]
#[ignore = "needs a GPU, LUMEN_PREFILL_ORACLE=capture|check and LUMEN_PREFILL_ORACLE_FILE"]
fn prefill_leaves_the_recorded_bits() {
    let mode = std::env::var("LUMEN_PREFILL_ORACLE").expect("LUMEN_PREFILL_ORACLE");
    let file = std::env::var("LUMEN_PREFILL_ORACLE_FILE").expect("LUMEN_PREFILL_ORACLE_FILE");
    let mut lines = Vec::new();
    run(
        &generate_test_model(&TestModelConfig {
            max_seq_len: CONTEXT as u32,
            ..TestModelConfig::default()
        }),
        "attention",
        &mut lines,
    );
    run(
        &build_gdn_hybrid_lbc_with(gdn_model_hyperparams_with(CONTEXT as u32)),
        "gdn-hybrid",
        &mut lines,
    );
    for (label, mixed_quant) in [("qwen35", false), ("qwen35-quant", true)] {
        let fx = Qwen35Fixture {
            mixed_quant,
            expert_quant: Some(QuantScheme::Q8_0),
            routed_experts: Some(64),
            ..Qwen35Fixture::default()
        };
        run(
            &build_qwen35_cpu_lbc(&fx, CONTEXT as u32),
            label,
            &mut lines,
        );
    }
    match mode.as_str() {
        "capture" => {
            std::fs::write(&file, lines.join("\n") + "\n").unwrap();
            println!("captured {} steps to {file}", lines.len());
        }
        "check" => {
            let want: Vec<String> = std::fs::read_to_string(&file)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect();
            assert_eq!(want.len(), lines.len(), "step count");
            let differ: Vec<_> = want
                .iter()
                .zip(&lines)
                .filter(|(w, g)| w != g)
                .map(|(w, _)| w.rsplit_once(' ').unwrap().0.to_owned())
                .collect();
            assert!(differ.is_empty(), "differ from the record: {differ:?}");
            println!("{} steps match the record", lines.len());
        }
        other => panic!("LUMEN_PREFILL_ORACLE={other}: capture or check"),
    }
}
