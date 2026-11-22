//! The CPU backends on a Qwen3.5-shaped hybrid (`common::gdn_hybrid::build_qwen35_cpu_lbc`):
//! two GDN layers with distinct weights, gated full attention with partial RoPE, dense and MoE
//! FFNs, in F32 and in every stored format the CPU decodes.
//!
//! - SIMD against the naive reference under one forced token history, within a tolerance set
//!   from the gap the two backends already show on the Llama path.
//! - Determinism: two naive runs, and SIMD with two and with seven threads, bit for bit.
//! - Providers: sync (after its rebuilding cache was filled), mmap and async give the same
//!   tokens and logits through the session's forward loop.
//! - Prompt cache: warm append (below and across the suffix threshold), exact match, a shorter
//!   prefix and a divergent prompt each equal a cold session, with F32 and F16 KV.
//! - Session save right after prefill and load into a fresh backend continue like the
//!   uninterrupted session; a file without recurrent state is refused.
//! - Preparation: a bad last layer fails preparation before anything changes; an unprepared
//!   backend refuses to compute, save and restore; the SIMD Llama path refuses a tensor it
//!   cannot read, which the naive one decodes.

mod common;

use common::gdn_hybrid::{build_qwen35_cpu_lbc, Qwen35Fixture};
use lumen_format::test_model::{
    generate_test_model, generate_test_model_q4_0, TestModelConfig, TestModelQ4Config,
};
use lumen_runtime::compute::cpu_simd::SimdF32Backend;
use lumen_runtime::kv::disk::{save_atomic, serialize_hyperparams_le, ModelFingerprint};
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::session::BenchTop2;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_async::AsyncWeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{
    ComputeBackend, NaiveF32Backend, RuntimeConfig, SamplingParams, Session, WeightProvider,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static UNIQ: AtomicU64 = AtomicU64::new(0);

const MAX_SEQ: u32 = 96;

fn write_lbc(bytes: &[u8]) -> PathBuf {
    let id = UNIQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cpu_qwen35_{}_{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.lbc");
    std::fs::write(&path, bytes).unwrap();
    path
}

fn fixture(fx: Qwen35Fixture) -> PathBuf {
    write_lbc(&build_qwen35_cpu_lbc(&fx, MAX_SEQ))
}

/// `backend` initialised from `provider` and, with `prepare`, prepared on it.
fn ready<B: ComputeBackend>(mut backend: B, provider: &SyncWeightProvider, prepare: bool) -> B {
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();
    if prepare {
        backend.preload_weights(provider).unwrap();
    }
    backend
}

fn kv_cache(provider: &SyncWeightProvider, precision: KvPrecision) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision,
    })
    .unwrap()
}

/// One token through every layer, as the forward loop runs it; the logits.
fn step(
    backend: &dyn ComputeBackend,
    w: &dyn WeightProvider,
    kv: &mut KvCache,
    token: u32,
) -> Vec<f32> {
    let mut x = backend.embed_token(token).unwrap();
    let pos = kv.seq_len();
    for layer in 0..w.num_layers() {
        let view = w.get_layer_raw(layer).unwrap();
        let mut v = kv.view_mut(layer).unwrap();
        backend
            .compute_layer(layer, &mut x, &view, Some(&mut v), pos)
            .unwrap();
        kv.commit_view(v).unwrap();
    }
    kv.advance_seq_len().unwrap();
    backend.compute_final(&x).unwrap().data
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// The logits of `tokens` fed one by one from a fresh cache.
fn logits_of(
    backend: &dyn ComputeBackend,
    provider: &SyncWeightProvider,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    backend.reset_recurrent_state();
    let mut kv = kv_cache(provider, KvPrecision::F32);
    tokens
        .iter()
        .map(|&t| step(backend, provider, &mut kv, t))
        .collect()
}

/// A prompt and the naive reference's greedy continuation of `n` tokens.
fn greedy_history(
    backend: &dyn ComputeBackend,
    provider: &SyncWeightProvider,
    n: usize,
) -> Vec<u32> {
    backend.reset_recurrent_state();
    let mut kv = kv_cache(provider, KvPrecision::F32);
    let mut tokens = vec![1u32, 7, 3];
    let mut logits = Vec::new();
    for &t in &tokens {
        logits = step(backend, provider, &mut kv, t);
    }
    for _ in 0..n {
        let t = argmax(&logits);
        tokens.push(t);
        logits = step(backend, provider, &mut kv, t);
    }
    tokens
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

/// The largest per-step relative L2 distance between SIMD and naive logits over
/// the naive backend's own 32-token greedy history.
fn simd_gap(path: &Path) -> f64 {
    let provider = SyncWeightProvider::open(path).unwrap();
    let naive = ready(NaiveF32Backend::new(), &provider, true);
    let simd = ready(SimdF32Backend::new(), &provider, true);
    let tokens = greedy_history(&naive, &provider, 32);
    let a = logits_of(&naive, &provider, &tokens);
    let b = logits_of(&simd, &provider, &tokens);
    a.iter()
        .zip(&b)
        .map(|(n, s)| rel_l2(s, n))
        .fold(0.0, f64::max)
}

/// The SIMD backend approximates the naive one where its kernels do (aarch64 quantizes the
/// activations of a Q8_0 matrix and the output head to Q8_0). The gap is measured on a Llama
/// test model of the fixture's width, vocabulary and head size, and the hybrid must stay
/// within 4x of it (and of 1e-4 where the Llama gap is smaller, e.g. on x86 without the Q8
/// head).
#[test]
fn simd_tracks_the_naive_reference_on_one_history() {
    let llama = write_lbc(&generate_test_model(&TestModelConfig {
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 256,
        hidden_dim: 64,
        intermediate_dim: 64,
        vocab_size: 64,
        max_seq_len: MAX_SEQ,
        seed: 42,
    }));
    let llama_gap = simd_gap(&llama);
    let limit = (4.0 * llama_gap).max(1e-4);
    for mixed_quant in [false, true] {
        let gap = simd_gap(&fixture(Qwen35Fixture {
            mixed_quant,
            ..Default::default()
        }));
        eprintln!(
            "mixed_quant={mixed_quant}: SIMD-naive gap {gap:.3e} (Llama path {llama_gap:.3e})"
        );
        assert!(
            gap <= limit,
            "mixed_quant={mixed_quant}: gap {gap:.3e} > {limit:.3e}"
        );
    }
}

/// Two naive runs over the same history, the recurrent state reset in between, and SIMD with
/// two and with seven threads, agree bit for bit.
#[test]
fn runs_are_deterministic() {
    for mixed_quant in [false, true] {
        let path = fixture(Qwen35Fixture {
            mixed_quant,
            ..Default::default()
        });
        let provider = SyncWeightProvider::open(&path).unwrap();
        let naive = ready(NaiveF32Backend::new(), &provider, true);
        let tokens = greedy_history(&naive, &provider, 16);
        assert_eq!(
            logits_of(&naive, &provider, &tokens),
            logits_of(&naive, &provider, &tokens),
            "naive, mixed_quant={mixed_quant}"
        );
        let two = ready(SimdF32Backend::with_threads(2), &provider, true);
        let seven = ready(SimdF32Backend::with_threads(7), &provider, true);
        assert_eq!((two.thread_count(), seven.thread_count()), (2, 7));
        assert_eq!(
            logits_of(&two, &provider, &tokens),
            logits_of(&seven, &provider, &tokens),
            "SIMD threads, mixed_quant={mixed_quant}"
        );
    }
}

fn config(precision: KvPrecision) -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: lumen_runtime::pipeline::PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: precision,
        max_seq_len: MAX_SEQ as usize,
        collect_per_layer_timings: false,
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        repetition_penalty: Some(1.0),
        seed: Some(1),
        ..Default::default()
    }
}

fn session(provider_hp: lumen_format::ModelHyperparams, precision: KvPrecision) -> Session {
    let mut s = Session::new(config(precision), provider_hp, greedy()).unwrap();
    s.set_bench_top2(true);
    s
}

/// `n` greedy tokens and what the top-2 surface saw for each.
fn generate(
    s: &mut Session,
    backend: &dyn ComputeBackend,
    w: &dyn WeightProvider,
    n: usize,
) -> (Vec<u32>, Vec<BenchTop2>) {
    let tokens = (0..n).map(|_| s.next_token(backend, w).unwrap()).collect();
    (tokens, s.take_bench_top2())
}

/// The session's forward loop reads every provider's layers as stored: the sync provider
/// with its dequantising cache filled first, mmap and async give one result.
#[test]
fn providers_agree_through_the_forward_loop() {
    let path = fixture(Qwen35Fixture {
        mixed_quant: true,
        ..Default::default()
    });
    let sync = SyncWeightProvider::open(&path).unwrap();
    for layer in 0..sync.num_layers() {
        sync.get_layer_blocking(layer).unwrap();
    }
    assert!(
        sync.try_get_layer(0).is_some(),
        "the rebuilt cache is filled"
    );
    let mmap = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
    let asyn = AsyncWeightProvider::open(&path).unwrap();
    let hp = sync.lbc().header.hyperparams;
    let backend = ready(NaiveF32Backend::new(), &sync, true);
    let prompt = [2u32, 9, 4, 11];
    let mut results = Vec::new();
    for w in [&sync as &dyn WeightProvider, &mmap, &asyn] {
        let mut s = session(hp, KvPrecision::F32);
        s.extend(&prompt, &backend, w).unwrap();
        results.push(generate(&mut s, &backend, w, 12));
    }
    assert_eq!(results[0], results[1], "sync vs mmap");
    assert_eq!(results[1], results[2], "mmap vs async");
}

/// Every prompt-cache route ends where a cold session over the same history does.
#[test]
fn prompt_cache_routes_equal_a_cold_session() {
    let path = fixture(Qwen35Fixture::default());
    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;
    let backend = ready(NaiveF32Backend::new(), &provider, true);
    assert!(backend.caps().gdn);
    let threshold = 4;
    for precision in [KvPrecision::F32, KvPrecision::F16] {
        let cold = |history: &[u32]| {
            let mut s = session(hp, precision);
            s.extend(history, &backend, &provider).unwrap();
            generate(&mut s, &backend, &provider, 6)
        };
        let mut warm = session(hp, precision);
        let base = [3u32, 1, 4, 1, 5];
        warm.extend_with_cache(&base, &backend, &provider, threshold)
            .unwrap();
        let (gen, _) = generate(&mut warm, &backend, &provider, 5);
        let mut history: Vec<u32> = base.iter().copied().chain(gen).collect();
        // appends below and across the threshold, after generation
        for extra in [&[9u32, 2, 6][..], &[5, 3, 5, 8, 9, 7, 9, 3, 2, 3]] {
            history.extend_from_slice(extra);
            warm.extend_with_cache(&history, &backend, &provider, threshold)
                .unwrap();
            let got = generate(&mut warm, &backend, &provider, 6);
            assert_eq!(
                got,
                cold(&history),
                "{precision:?}: append of {}",
                extra.len()
            );
            history.extend_from_slice(&got.0);
        }
        // the history exactly, then a shorter prefix, then a divergent tail
        warm.extend_with_cache(&history, &backend, &provider, threshold)
            .unwrap();
        assert_eq!(
            generate(&mut warm, &backend, &provider, 6),
            cold(&history),
            "{precision:?}: exact"
        );
        let shorter = &history[..history.len() - 4];
        warm.extend_with_cache(shorter, &backend, &provider, threshold)
            .unwrap();
        assert_eq!(
            generate(&mut warm, &backend, &provider, 6),
            cold(shorter),
            "{precision:?}: shorter"
        );
        let mut divergent = history[..history.len() - 2].to_vec();
        divergent.extend_from_slice(&[60, 61, 62]);
        warm.extend_with_cache(&divergent, &backend, &provider, threshold)
            .unwrap();
        assert_eq!(
            generate(&mut warm, &backend, &provider, 6),
            cold(&divergent),
            "{precision:?}: divergent"
        );
    }
}

fn fingerprint(provider: &SyncWeightProvider) -> ModelFingerprint {
    let hp = provider.lbc().header.hyperparams;
    ModelFingerprint::from_live_model(
        &serialize_hyperparams_le(&hp),
        &[],
        provider.output_proj_quant.to_u8() as u32,
        lumen_format::LBC_VERSION,
    )
}

/// Saved right after prefill and loaded into a fresh backend, a session continues exactly
/// like the one that never stopped; a file that carries no recurrent state is refused.
#[test]
fn session_save_after_prefill_restores_on_a_fresh_backend() {
    let path = fixture(Qwen35Fixture {
        mixed_quant: true,
        ..Default::default()
    });
    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;
    let fp = fingerprint(&provider);
    let prompt = [8u32, 6, 7, 5, 3, 0, 9];
    let file = path.with_file_name("session.kv");

    let backend = ready(NaiveF32Backend::new(), &provider, true);
    let mut s = session(hp, KvPrecision::F16);
    s.extend(&prompt, &backend, &provider).unwrap();
    s.save_to_disk(&file, &backend, &provider, &fp).unwrap();
    let uninterrupted = generate(&mut s, &backend, &provider, 10);

    let fresh = ready(NaiveF32Backend::new(), &provider, true);
    let mut r = Session::load_from_disk(&file, config(KvPrecision::F16), hp, greedy(), &fresh, &fp)
        .unwrap();
    r.set_bench_top2(true);
    assert_eq!(generate(&mut r, &fresh, &provider, 10), uninterrupted);

    let bare = path.with_file_name("bare.kv");
    save_atomic(
        s.kv(),
        &s.tokens()[..s.kv().seq_len()],
        &bare,
        0,
        &fp,
        None,
        None,
    )
    .unwrap();
    let err = Session::load_from_disk(&bare, config(KvPrecision::F16), hp, greedy(), &fresh, &fp)
        .err()
        .expect("a file without recurrent state is refused");
    assert!(
        err.to_string().contains("carries no recurrent state"),
        "{err}"
    );
}

/// A bad last layer fails preparation, so nothing has run: the session is empty and the
/// backend reports no recurrent state.
#[test]
fn a_bad_last_layer_fails_preparation_before_any_change() {
    let path = fixture(Qwen35Fixture {
        bad_last_layer: true,
        ..Default::default()
    });
    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;
    let mut backend = ready(NaiveF32Backend::new(), &provider, false);
    let err = backend.preload_weights(&provider).unwrap_err().to_string();
    assert!(
        err.contains(&format!("layer {}", hp.num_layers - 1)),
        "{err}"
    );
    assert!(err.contains("attn_q_norm"), "{err}");
    assert!(backend.gdn_layout().is_none() && !backend.caps().gdn);
    let mut s = session(hp, KvPrecision::F32);
    assert!(s.extend(&[1, 2], &backend, &provider).is_err());
    assert_eq!((s.kv().seq_len(), s.token_count()), (0, 0));
}

/// Unprepared, a backend computes nothing (not even a plain layer ahead of the hybrid
/// ones), saves nothing and restores nothing, whatever the file holds.
#[test]
fn an_unprepared_backend_refuses_compute_save_and_restore() {
    let path = fixture(Qwen35Fixture {
        plain_first: true,
        ..Default::default()
    });
    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;
    let fp = fingerprint(&provider);
    let file = path.with_file_name("pending.kv");

    // a checkpoint with pending logits and no recurrent section
    let prepared = ready(SimdF32Backend::new(), &provider, true);
    let mut s = session(hp, KvPrecision::F32);
    s.extend(&[4, 5, 6], &prepared, &provider).unwrap();
    let pending = vec![0.5f32; hp.vocab_size as usize];
    save_atomic(s.kv(), s.tokens(), &file, 0, &fp, None, Some(&pending)).unwrap();

    for unprepared in [
        Box::new(ready(NaiveF32Backend::new(), &provider, false)) as Box<dyn ComputeBackend>,
        Box::new(ready(SimdF32Backend::new(), &provider, false)),
    ] {
        let b = unprepared.as_ref();
        assert!(b.gdn_layout().is_none());
        let mut s = session(hp, KvPrecision::F32);
        let err = s.extend(&[1, 2, 3], b, &provider).unwrap_err().to_string();
        assert!(err.contains("preload_weights"), "{err}");
        assert_eq!(s.kv().seq_len(), 0);
        let (k0, v0) = s.kv().layer_raw_bytes(0).unwrap();
        assert!(
            !k0.is_empty() && k0.len() == v0.len(),
            "the layer's KV was not returned"
        );
        assert!(
            k0.iter().chain(v0).all(|&b| b == 0),
            "the plain layer wrote KV"
        );

        let saved = path.with_file_name("unprepared.kv");
        assert!(s.save_to_disk(&saved, b, &provider, &fp).is_err());
        assert!(!saved.exists(), "a refused save writes no file");

        let err = Session::load_from_disk(&file, config(KvPrecision::F32), hp, greedy(), b, &fp)
            .err()
            .expect("restore into an unprepared backend is refused");
        assert!(err.to_string().contains("preload_weights"), "{err}");
    }
}

/// On the Llama path the naive backend decodes a Q4_0 model's stored rows to the same logits
/// as the sync provider's F32 dequantisation of them; the SIMD path, which reads only F32
/// and Q8_0 there, refuses the model at preparation instead of misreading it.
#[test]
fn the_llama_path_decodes_or_refuses_a_q4_0_model() {
    let path = write_lbc(&generate_test_model_q4_0(&TestModelQ4Config::default()));
    let provider = SyncWeightProvider::open(&path).unwrap();
    let naive = ready(NaiveF32Backend::new(), &provider, true);
    let forward = |raw: bool| {
        let mut kv = kv_cache(&provider, KvPrecision::F32);
        let mut logits = Vec::new();
        for token in [3u32, 1, 4] {
            let mut x = naive.embed_token(token).unwrap();
            let pos = kv.seq_len();
            for layer in 0..provider.num_layers() {
                let view = if raw {
                    provider.get_layer_raw(layer).unwrap()
                } else {
                    provider.get_layer_blocking(layer).unwrap()
                };
                let mut v = kv.view_mut(layer).unwrap();
                naive
                    .compute_layer(layer, &mut x, &view, Some(&mut v), pos)
                    .unwrap();
                kv.commit_view(v).unwrap();
            }
            kv.advance_seq_len().unwrap();
            logits.push(naive.compute_final(&x).unwrap().data);
        }
        logits
    };
    assert_eq!(forward(true), forward(false));

    let mut simd = ready(SimdF32Backend::new(), &provider, false);
    let err = simd.preload_weights(&provider).unwrap_err().to_string();
    assert!(
        err.contains("is Q4_0; this path reads F32 and Q8_0"),
        "{err}"
    );
}
