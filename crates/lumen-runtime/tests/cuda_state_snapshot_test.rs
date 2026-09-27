//! A decode resumed from a restored state snapshot is bit-identical to the
//! uninterrupted decode it was taken from, on the GDN/attention hybrid: after
//! a prefill and again part-way through decoding, into a second backend whose
//! own state was dirtied by another prompt first. Restoring and snapshotting
//! again returns the same snapshot. A snapshot with one conv ring position
//! moved decodes differently, so the comparison can see the state it claims
//! to cover. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features test-state-snapshot --test cuda_state_snapshot_test
#![cfg(feature = "test-state-snapshot")]

mod common;

use common::gdn_hybrid::{build_gdn_hybrid_lbc_with, gdn_model_hyperparams_with};
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::io::Write;

const CONTEXT: usize = 256;
const PROMPT: usize = 40;
const STEPS: usize = 16;
const SPLIT: usize = 6;

fn open(dir: &std::path::Path, lbc: &[u8]) -> SyncWeightProvider {
    let path = dir.join("model.lbc");
    std::fs::write(&path, lbc).unwrap();
    SyncWeightProvider::open(&path).expect("open provider")
}

fn backend(provider: &SyncWeightProvider) -> Option<CudaBackend> {
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
    cuda.init(&provider.lbc().header.hyperparams).expect("init");
    cuda.preload_weights(provider).expect("preload");
    Some(cuda)
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

fn prompt(len: usize, salt: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 7919 + salt) % vocab) as u32)
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Logits of the prompt's last token, from the hidden row the engine hands to
/// `compute_final`.
fn first_logits(cuda: &CudaBackend, x: &[f32]) -> Vec<u32> {
    let mut buf = ActivationBuffer::zeros(x.len(), ComputeDtype::F32);
    buf.write_f32_from(x);
    let logits = cuda.compute_final(&buf).expect("compute_final");
    logits.data.iter().map(|v| v.to_bits()).collect()
}

/// `steps` greedy decode steps from `token`: every step's logits as bits, and
/// the token the last step chose.
fn decode(
    cuda: &CudaBackend,
    provider: &SyncWeightProvider,
    kv: &mut KvCache,
    mut token: u32,
    steps: usize,
) -> (Vec<Vec<u32>>, u32) {
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let logits = cuda.decode_token(token, provider, kv).expect("decode");
        token = argmax(&logits.data);
        out.push(logits.data.iter().map(|v| v.to_bits()).collect());
    }
    (out, token)
}

#[test]
fn restored_state_decodes_bit_identically() {
    let dir = tempfile::tempdir().unwrap();
    let provider = open(
        dir.path(),
        &build_gdn_hybrid_lbc_with(gdn_model_hyperparams_with(CONTEXT as u32)),
    );
    let (Some(a), Some(b)) = (backend(&provider), backend(&provider)) else {
        return;
    };
    let vocab = provider.lbc().header.hyperparams.vocab_size as usize;

    // Uninterrupted: prefill, snapshot, decode SPLIT steps, snapshot, decode the rest.
    let mut kv_a = kv_cache(&provider);
    let x = a
        .prefill(&prompt(PROMPT, 13, vocab), &provider, &mut kv_a)
        .expect("prefill");
    let after_prefill = a.snapshot_state(&kv_a).expect("snapshot after prefill");
    assert_eq!(
        after_prefill.x, x,
        "the snapshot holds the prefill's last row"
    );
    let logits0 = first_logits(&a, &x);
    let t0 = argmax(
        &logits0
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect::<Vec<_>>(),
    );
    let (head, t_split) = decode(&a, &provider, &mut kv_a, t0, SPLIT);
    let mid = a.snapshot_state(&kv_a).expect("snapshot mid-decode");
    assert_eq!(mid.seq_len, PROMPT + SPLIT);
    assert_eq!(mid.decode_token_count, SPLIT);
    let (tail, _) = decode(&a, &provider, &mut kv_a, t_split, STEPS - SPLIT);

    // Dirty the second backend with another prompt and a few decode steps.
    let mut kv_dirty = kv_cache(&provider);
    b.prefill(&prompt(PROMPT - 7, 101, vocab), &provider, &mut kv_dirty)
        .expect("dirty prefill");
    decode(&b, &provider, &mut kv_dirty, 3, 5);

    let mut kv_b = kv_cache(&provider);
    b.restore_state(&after_prefill, &mut kv_b)
        .expect("restore after prefill");
    assert_eq!(kv_b.seq_len(), PROMPT);
    assert_eq!(
        b.snapshot_state(&kv_b)
            .expect("snapshot of the restored state"),
        after_prefill,
        "restore then snapshot returns the snapshot"
    );
    let (all, _) = decode(&b, &provider, &mut kv_b, t0, STEPS);
    assert_eq!(
        all[..SPLIT],
        head[..],
        "restored after prefill: decode differs"
    );
    assert_eq!(
        all[SPLIT..],
        tail[..],
        "restored after prefill: decode differs"
    );

    let mut kv_b = kv_cache(&provider);
    b.restore_state(&mid, &mut kv_b)
        .expect("restore mid-decode");
    let (rest, _) = decode(&b, &provider, &mut kv_b, t_split, STEPS - SPLIT);
    assert_eq!(rest, tail, "restored mid-decode: decode differs");

    // Negative control: one GDN ring position moved.
    let mut moved = after_prefill.clone();
    moved.conv_positions[0] = (moved.conv_positions[0] + 1) % 3;
    let mut kv_b = kv_cache(&provider);
    b.restore_state(&moved, &mut kv_b).expect("restore moved");
    let (moved_decode, _) = decode(&b, &provider, &mut kv_b, t0, STEPS);
    assert_ne!(
        moved_decode[..SPLIT],
        head[..],
        "a moved conv ring position must change the decode"
    );

    // A ring position past the ring is refused.
    let mut out_of_ring = after_prefill.clone();
    out_of_ring.conv_positions[0] = 3;
    let err = b
        .restore_state(&out_of_ring, &mut kv_cache(&provider))
        .expect_err("restore with a ring position past the ring");
    assert!(err.to_string().contains("conv ring slots"), "{err}");

    // A host cache that already holds tokens is refused before anything is written.
    let err = b
        .restore_state(&after_prefill, &mut kv_b)
        .expect_err("restore into a non-empty host cache");
    assert!(err.to_string().contains("empty host KV cache"), "{err}");
}

// ---------------------------------------------------------------------------
// Real-model fixtures. Two ignored tools, run on a GPU host against a real
// artifact, freeze decode states and decode from them, so two builds can be
// compared on one state (decode speed and logits):
//
//   LUMEN_STATE_MODEL=<model.lbc> LUMEN_STATE_CASES=<cases.json> LUMEN_STATE_DIR=<dir> \
//     cargo test --release -p lumen-runtime --features test-state-snapshot \
//     --test cuda_state_snapshot_test -- --ignored --exact capture_state_fixtures
//   LUMEN_STATE_MODEL=... LUMEN_STATE_DIR=<dir> LUMEN_STATE_OUT=<results.json> \
//     [LUMEN_STATE_REPS=<n>] ... --exact decode_state_fixtures
//
// `cases.json` is `{"cases": [{"name": "P128", "steps": [{"prefill": [ids]},
// {"decode": [ids]}, ...]}]}`: prefill slices and teacher-forced decode steps
// in order. Each case is saved to `<dir>/<name>/` as `meta.json` and
// `state.bin` (little-endian f32: every attention layer's K then V, every GDN
// state, every conv ring, then the hidden row, at the lengths `meta.json`
// records).
// ---------------------------------------------------------------------------

use lumen_format::quantization::QuantScheme;
use lumen_runtime::cuda::StateSnapshot;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Device and host KV capacity of the real-model tools.
const REAL_CONTEXT: usize = 4096;
/// Greedy tokens decoded from each fixture.
const DECODE_TOKENS: usize = 256;
/// Steps the moved-ring negative control decodes.
const CHECK_STEPS: usize = 32;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

/// The backend as `lumen-server` builds it for CUDA: F32 KV, the raw global
/// planes CUDA takes as stored, the context capped at `REAL_CONTEXT`.
fn real_backend(provider: &SyncWeightProvider) -> CudaBackend {
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = hp.max_seq_len.min(REAL_CONTEXT as u32);
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

fn real_kv(provider: &SyncWeightProvider) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: REAL_CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    })
    .unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("an array of token ids")
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("a token id")).expect("a u32 token id"))
        .collect()
}

/// FNV-1a over a step's logits bits: equal hashes stand for bit-equal logits.
fn fnv(logits: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in logits.iter().flat_map(|v| v.to_bits().to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

fn logits_of_row(cuda: &CudaBackend, x: &[f32]) -> Vec<f32> {
    let mut buf = ActivationBuffer::zeros(x.len(), ComputeDtype::F32);
    buf.write_f32_from(x);
    cuda.compute_final(&buf).expect("compute_final").data
}

/// The first token and the hashes of the prompt's logits and of `steps`
/// greedy decode steps after it, with each step's chosen token.
fn logits_chain(
    cuda: &CudaBackend,
    provider: &SyncWeightProvider,
    kv: &mut KvCache,
    x: &[f32],
    steps: usize,
) -> (Vec<String>, Vec<u32>) {
    let logits = logits_of_row(cuda, x);
    let mut hashes = vec![fnv(&logits)];
    let mut tokens = vec![argmax(&logits)];
    for _ in 0..steps {
        let logits = cuda
            .decode_token(*tokens.last().unwrap(), provider, kv)
            .expect("decode")
            .data;
        hashes.push(fnv(&logits));
        tokens.push(argmax(&logits));
    }
    (hashes, tokens)
}

fn save(dir: &Path, name: &str, snap: &StateSnapshot, meta: Value) {
    let case = dir.join(name);
    std::fs::create_dir_all(&case).unwrap();
    let mut bin = std::io::BufWriter::new(std::fs::File::create(case.join("state.bin")).unwrap());
    let blocks = snap
        .kv
        .iter()
        .flat_map(|(k, v)| [k, v])
        .chain(&snap.h_states)
        .chain(&snap.conv_states)
        .chain([&snap.x]);
    for block in blocks {
        for v in block {
            bin.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    bin.flush().unwrap();
    let mut meta = meta;
    meta["seq_len"] = json!(snap.seq_len);
    meta["decode_token_count"] = json!(snap.decode_token_count);
    meta["conv_positions"] = json!(snap.conv_positions);
    meta["kv_lens"] = json!(snap.kv.iter().map(|(k, _)| k.len()).collect::<Vec<_>>());
    meta["h_lens"] = json!(snap.h_states.iter().map(Vec::len).collect::<Vec<_>>());
    meta["conv_lens"] = json!(snap.conv_states.iter().map(Vec::len).collect::<Vec<_>>());
    meta["x_len"] = json!(snap.x.len());
    std::fs::write(
        case.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

fn load(case: &Path) -> (StateSnapshot, Value) {
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(case.join("meta.json")).unwrap()).unwrap();
    let bytes = std::fs::read(case.join("state.bin")).unwrap();
    let mut floats = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()));
    let lens = |key: &str| -> Vec<usize> {
        meta[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect()
    };
    let mut take = |n: usize| -> Vec<f32> { floats.by_ref().take(n).collect() };
    let kv = lens("kv_lens")
        .into_iter()
        .map(|n| (take(n), take(n)))
        .collect();
    let h_states = lens("h_lens").into_iter().map(&mut take).collect();
    let conv_states = lens("conv_lens").into_iter().map(&mut take).collect();
    let x = take(meta["x_len"].as_u64().unwrap() as usize);
    assert!(
        floats.next().is_none(),
        "{}: state.bin longer than meta.json",
        case.display()
    );
    let snap = StateSnapshot {
        seq_len: meta["seq_len"].as_u64().unwrap() as usize,
        kv,
        h_states,
        conv_states,
        conv_positions: meta["conv_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap() as u32)
            .collect(),
        x,
        decode_token_count: meta["decode_token_count"].as_u64().unwrap() as usize,
    };
    let stored: usize = snap
        .kv
        .iter()
        .map(|(k, v)| k.len() + v.len())
        .sum::<usize>()
        + snap.h_states.iter().map(Vec::len).sum::<usize>()
        + snap.conv_states.iter().map(Vec::len).sum::<usize>()
        + snap.x.len();
    assert_eq!(
        stored * 4,
        bytes.len(),
        "{}: state.bin is short",
        case.display()
    );
    (snap, meta)
}

#[test]
#[ignore = "needs a GPU, a real artifact and LUMEN_STATE_* paths"]
fn capture_state_fixtures() {
    let provider = SyncWeightProvider::open(&env_path("LUMEN_STATE_MODEL")).expect("model");
    let cases: Value =
        serde_json::from_str(&std::fs::read_to_string(env_path("LUMEN_STATE_CASES")).unwrap())
            .unwrap();
    let dir = env_path("LUMEN_STATE_DIR");
    let cuda = real_backend(&provider);
    for case in cases["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        cuda.reset_recurrent_state();
        let mut kv = real_kv(&provider);
        let mut x = Vec::new();
        for step in case["steps"].as_array().expect("steps") {
            if let Some(p) = step.get("prefill") {
                x = cuda.prefill(&ids(p), &provider, &mut kv).expect("prefill");
            } else {
                for t in ids(&step["decode"]) {
                    cuda.decode_token(t, &provider, &mut kv)
                        .expect("forced decode");
                }
                x = cuda.snapshot_state(&kv).expect("snapshot").x;
            }
        }
        let snap = cuda.snapshot_state(&kv).expect("snapshot");
        assert_eq!(snap.x, x);
        let (hashes, tokens) = logits_chain(&cuda, &provider, &mut kv, &snap.x, DECODE_TOKENS);
        println!(
            "{name}: seq_len {} first tokens {:?}",
            snap.seq_len,
            &tokens[..8]
        );
        save(
            &dir,
            name,
            &snap,
            json!({"name": name, "steps": case["steps"], "check_logits_fnv": hashes,
                   "check_tokens": tokens}),
        );
    }
}

#[test]
#[ignore = "needs a GPU, a real artifact and LUMEN_STATE_* paths"]
fn decode_state_fixtures() {
    let provider = SyncWeightProvider::open(&env_path("LUMEN_STATE_MODEL")).expect("model");
    let dir = env_path("LUMEN_STATE_DIR");
    let reps: usize = std::env::var("LUMEN_STATE_REPS")
        .map(|r| r.parse().expect("LUMEN_STATE_REPS"))
        .unwrap_or(5);
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.json").is_file())
        .collect();
    cases.sort();
    let loaded: Vec<(StateSnapshot, Value)> = cases.iter().map(|c| load(c)).collect();
    let cuda = real_backend(&provider);
    let restore = |snap: &StateSnapshot| {
        let mut kv = real_kv(&provider);
        cuda.restore_state(snap, &mut kv).expect("restore");
        kv
    };

    // Per fixture: whether this build's restored decode reproduces the
    // capturing build's logits bit for bit (recorded, and asserted after the
    // timing, so a build that differs is still timed), and this build's own checks: two restored
    // decodes agree, and a moved ring position changes the decode.
    let mut results = Vec::new();
    for (snap, meta) in &loaded {
        let name = meta["name"].as_str().unwrap();
        let mut kv = restore(snap);
        let (hashes, tokens) = logits_chain(&cuda, &provider, &mut kv, &snap.x, DECODE_TOKENS);
        let check: Vec<String> = serde_json::from_value(meta["check_logits_fnv"].clone()).unwrap();
        let first_diff =
            (0..hashes.len().max(check.len())).find(|&i| hashes.get(i) != check.get(i));
        let mut kv = restore(snap);
        let (again, _) = logits_chain(&cuda, &provider, &mut kv, &snap.x, DECODE_TOKENS);
        assert_eq!(again, hashes, "{name}: two restored decodes differ");
        let mut moved = snap.clone();
        moved.conv_positions[0] = (moved.conv_positions[0] + 1) % 3;
        let mut kv = restore(&moved);
        let (moved_hashes, _) = logits_chain(&cuda, &provider, &mut kv, &snap.x, CHECK_STEPS);
        assert_ne!(
            moved_hashes[1..],
            hashes[1..=CHECK_STEPS],
            "{name}: a moved ring position went unseen"
        );
        results.push(json!({"name": name, "seq_len": snap.seq_len,
                            "matches_capture": first_diff.is_none(),
                            "first_step_differing_from_capture": first_diff,
                            "logits_fnv": hashes, "tokens": tokens, "decode_s": []}));
    }

    // Speed: the production greedy path, fixtures interleaved within each
    // repetition, after one untimed pass over every fixture. Every
    // repetition's tokens must equal the argmax of the logits checked above,
    // so the timed path is the checked one.
    for rep in 0..=reps {
        for ((snap, _), result) in loaded.iter().zip(results.iter_mut()) {
            let mut kv = restore(snap);
            let mut token = u32::try_from(result["tokens"][0].as_u64().unwrap()).unwrap();
            let mut greedy = vec![token];
            let start = Instant::now();
            for _ in 0..DECODE_TOKENS {
                token = cuda
                    .decode_token_greedy(token, &provider, &mut kv)
                    .expect("greedy decode");
                greedy.push(token);
            }
            let secs = start.elapsed().as_secs_f64();
            assert_eq!(
                json!(greedy),
                result["tokens"],
                "{}: greedy decode differs from the checked logits' argmax",
                result["name"]
            );
            if rep > 0 {
                result["decode_s"].as_array_mut().unwrap().push(json!(secs));
            }
        }
    }
    for r in &results {
        let times: Vec<f64> = serde_json::from_value(r["decode_s"].clone()).unwrap();
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "{}: seq_len {} matches capture {} median {:.2} tok/s over {} reps",
            r["name"],
            r["seq_len"],
            r["matches_capture"],
            DECODE_TOKENS as f64 / sorted[sorted.len() / 2],
            times.len()
        );
    }
    std::fs::write(
        env_path("LUMEN_STATE_OUT"),
        serde_json::to_string_pretty(&json!({"decode_tokens": DECODE_TOKENS, "fixtures": results}))
            .unwrap(),
    )
    .unwrap();
    let differing: Vec<_> = results
        .iter()
        .filter(|r| r["matches_capture"] != json!(true))
        .map(|r| r["name"].clone())
        .collect();
    assert!(
        differing.is_empty(),
        "restored decode differs from capture on {differing:?}; timings are in LUMEN_STATE_OUT"
    );
}
