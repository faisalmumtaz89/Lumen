//! A session saved at any point of a generation and loaded again continues exactly like one
//! that never stopped, on the CPU backends and the GDN-hybrid fixture
//! (`common::gdn_hybrid::build_qwen35_cpu_lbc`). Runs, all from one prompt:
//!
//! - A: never saved, the prompt then `k + M` tokens;
//! - C: saved after `k` tokens, then continued in the same session;
//! - B: saved after `k` tokens, loaded into a fresh backend and continued;
//! - B': as B, loaded into a backend that first ran another prompt.
//!
//! On the greedy and penalised-greedy routes B, B' and C emit A's tokens, and their final
//! files (KV, recurrent state, tokens, next-token logits) are byte-identical to A's. On the
//! sampled route C does too (a save consumes no randomness); a loaded session reseeds its
//! sampler, so B and B' are compared with A reseeded by `set_sampling` after `k` tokens.
//!
//! Also: back-to-back saves, a load re-saved unchanged, the full-context and
//! fed-history-without-logits refusals, and the refusal of a file without logits written
//! after generation by an earlier version.

mod common;

use common::gdn_hybrid::{build_qwen35_cpu_lbc, Qwen35Fixture};
use lumen_runtime::compute::cpu_simd::SimdF32Backend;
use lumen_runtime::kv::disk::{
    serialize_hyperparams_le, DiskKvHeader, ModelFingerprint, RecurrentState, HEADER_SIZE,
    PENDING_LOGITS_META_BYTES,
};
use lumen_runtime::kv::{KvCache, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{ComputeBackend, NaiveF32Backend, RuntimeConfig, SamplingParams, Session};
use std::path::Path;
use tempfile::TempDir;

const MAX_SEQ: u32 = 96;
const PROMPT: &[u32] = &[3, 1, 4, 1, 5, 9, 2, 6];
const OTHER: &[u32] = &[7, 7, 1, 8, 2, 8];
const M: usize = 16;

/// A temporary directory holding the fixture model, for it and the test's session files.
fn model() -> (TempDir, SyncWeightProvider) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.lbc");
    std::fs::write(
        &path,
        build_qwen35_cpu_lbc(&Qwen35Fixture::default(), MAX_SEQ),
    )
    .unwrap();
    let provider = SyncWeightProvider::open(&path).unwrap();
    (dir, provider)
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Naive,
    Simd,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Route {
    Greedy,
    Penalised,
    Sampled,
}

fn backend(kind: Kind, provider: &SyncWeightProvider) -> Box<dyn ComputeBackend> {
    fn prepared<B: ComputeBackend + 'static>(
        mut b: B,
        provider: &SyncWeightProvider,
    ) -> Box<dyn ComputeBackend> {
        b.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        b.init(&provider.lbc().header.hyperparams).unwrap();
        b.preload_weights(provider).unwrap();
        Box::new(b)
    }
    match kind {
        Kind::Naive => prepared(NaiveF32Backend::new(), provider),
        Kind::Simd => prepared(SimdF32Backend::new(), provider),
    }
}

fn sampling(route: Route) -> SamplingParams {
    match route {
        Route::Greedy => SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.0),
            seed: Some(1),
            ..Default::default()
        },
        Route::Penalised => SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.3),
            seed: Some(1),
            ..Default::default()
        },
        Route::Sampled => SamplingParams {
            temperature: 0.8,
            repetition_penalty: Some(1.0),
            seed: Some(7),
            ..Default::default()
        },
    }
}

fn config(max_seq_len: usize) -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: lumen_runtime::pipeline::PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len,
        collect_per_layer_timings: false,
    }
}

fn fingerprint(provider: &SyncWeightProvider) -> ModelFingerprint {
    ModelFingerprint::from_live_model(
        &serialize_hyperparams_le(&provider.lbc().header.hyperparams),
        &[],
        provider.output_proj_quant.to_u8() as u32,
        lumen_format::LBC_VERSION,
    )
}

/// A session on `b` that has read `prompt`.
fn start(
    provider: &SyncWeightProvider,
    b: &dyn ComputeBackend,
    route: Route,
    prompt: &[u32],
    max_seq_len: usize,
) -> Session {
    b.reset_recurrent_state();
    let hp = provider.lbc().header.hyperparams;
    let mut s = Session::new(config(max_seq_len), hp, sampling(route)).unwrap();
    s.extend(prompt, b, provider).unwrap();
    s
}

fn generate(s: &mut Session, b: &dyn ComputeBackend, p: &SyncWeightProvider, n: usize) -> Vec<u32> {
    (0..n).map(|_| s.next_token(b, p).unwrap()).collect()
}

fn save(s: &mut Session, b: &dyn ComputeBackend, p: &SyncWeightProvider, path: &Path) -> Vec<u8> {
    s.save_to_disk(path, b, p, &fingerprint(p)).unwrap();
    std::fs::read(path).unwrap()
}

fn load(path: &Path, b: &dyn ComputeBackend, p: &SyncWeightProvider, route: Route) -> Session {
    let hp = p.lbc().header.hyperparams;
    Session::load_from_disk(
        path,
        config(MAX_SEQ as usize),
        hp,
        sampling(route),
        b,
        &fingerprint(p),
    )
    .unwrap()
}

/// `a` and `b` equal except for the header's save time (bytes 44..48); otherwise the first
/// differing offset.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let mask = |i: usize| (44..48).contains(&i);
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| !mask(i) && a[i] != b[i])
}

fn case(kind: Kind, route: Route, k: usize) {
    let (d, p) = model();
    let tag = format!("{kind:?} {route:?} k={k}");
    let max = MAX_SEQ as usize;

    // A: never saved.
    let a_b = backend(kind, &p);
    let mut a = start(&p, &*a_b, route, PROMPT, max);
    let a_tokens = generate(&mut a, &*a_b, &p, k + M);
    let a_file = save(&mut a, &*a_b, &p, &d.path().join("a.kv"));

    // C: saved, continued live.
    let c_b = backend(kind, &p);
    let mut c = start(&p, &*c_b, route, PROMPT, max);
    let mut c_tokens = generate(&mut c, &*c_b, &p, k);
    save(&mut c, &*c_b, &p, &d.path().join("c_k.kv"));
    c_tokens.extend(generate(&mut c, &*c_b, &p, M));
    assert_eq!(c_tokens, a_tokens, "{tag}: C differs from A");
    let c_file = save(&mut c, &*c_b, &p, &d.path().join("c.kv"));
    assert_eq!(
        first_difference(&c_file, &a_file),
        None,
        "{tag}: C's final file differs from A's"
    );

    // What B continues with: A's tokens, or on the sampled route A reseeded at the save.
    let (want, want_file) = if route == Route::Sampled {
        let s_b = backend(kind, &p);
        let mut s = start(&p, &*s_b, route, PROMPT, max);
        generate(&mut s, &*s_b, &p, k);
        s.set_sampling(sampling(route));
        let tokens = generate(&mut s, &*s_b, &p, M);
        (tokens, save(&mut s, &*s_b, &p, &d.path().join("s.kv")))
    } else {
        (a_tokens[k..].to_vec(), a_file)
    };

    // B: saved, loaded into a fresh backend, continued.
    let saved = d.path().join("b_k.kv");
    {
        let b_b = backend(kind, &p);
        let mut b = start(&p, &*b_b, route, PROMPT, max);
        generate(&mut b, &*b_b, &p, k);
        save(&mut b, &*b_b, &p, &saved);
    }
    let want_history: Vec<u32> = PROMPT.iter().chain(&a_tokens[..k]).copied().collect();
    for used in [false, true] {
        let r_b = backend(kind, &p);
        if used {
            let mut other = start(&p, &*r_b, route, OTHER, max);
            generate(&mut other, &*r_b, &p, 8);
        }
        let mut r = load(&saved, &*r_b, &p, route);
        assert_eq!(r.tokens(), &want_history[..], "{tag} used={used}: history");
        assert_eq!(
            r.kv().seq_len(),
            want_history.len(),
            "{tag} used={used}: KV length"
        );
        let cont = generate(&mut r, &*r_b, &p, M);
        assert_eq!(cont, want, "{tag} used={used}: continuation");
        let r_file = save(&mut r, &*r_b, &p, &d.path().join("r.kv"));
        assert_eq!(
            first_difference(&r_file, &want_file),
            None,
            "{tag} used={used}: final file differs from the reference's"
        );
    }
}

fn all_k(kind: Kind, route: Route) {
    for k in [0, 1, 2, 3, 8] {
        case(kind, route, k);
    }
}

#[test]
fn naive_greedy_resumes_like_a_session_that_never_stopped() {
    all_k(Kind::Naive, Route::Greedy);
}

#[test]
fn naive_penalised_resumes_like_a_session_that_never_stopped() {
    all_k(Kind::Naive, Route::Penalised);
}

#[test]
fn naive_save_takes_no_randomness_from_a_sampled_session() {
    all_k(Kind::Naive, Route::Sampled);
}

#[test]
fn simd_greedy_resumes_like_a_session_that_never_stopped() {
    all_k(Kind::Simd, Route::Greedy);
}

#[test]
fn simd_penalised_resumes_like_a_session_that_never_stopped() {
    all_k(Kind::Simd, Route::Penalised);
}

#[test]
fn simd_save_takes_no_randomness_from_a_sampled_session() {
    all_k(Kind::Simd, Route::Sampled);
}

/// With the top-2 surface armed, a save records nothing: the records around it equal those
/// of a run that never saved.
#[test]
fn a_save_records_no_top2() {
    let (d, p) = model();
    let run = |saved: bool| {
        let b = backend(Kind::Naive, &p);
        let mut s = start(&p, &*b, Route::Greedy, PROMPT, MAX_SEQ as usize);
        s.set_bench_top2(true);
        let mut tokens = generate(&mut s, &*b, &p, 3);
        if saved {
            save(&mut s, &*b, &p, &d.path().join("t.kv"));
        }
        tokens.extend(generate(&mut s, &*b, &p, M));
        (tokens, s.take_bench_top2())
    };
    assert_eq!(run(true), run(false));
}

/// Two saves in a row write the same bytes; a save at 3 and another 5 tokens later both
/// continue as the unsaved run; a loaded file re-saved at once is unchanged.
#[test]
fn repeated_saves_agree() {
    let (d, p) = model();
    let max = MAX_SEQ as usize;
    let a_b = backend(Kind::Naive, &p);
    let mut a = start(&p, &*a_b, Route::Greedy, PROMPT, max);
    let a_tokens = generate(&mut a, &*a_b, &p, 3 + 5 + M);

    let b = backend(Kind::Naive, &p);
    let mut s = start(&p, &*b, Route::Greedy, PROMPT, max);
    generate(&mut s, &*b, &p, 3);
    let first = save(&mut s, &*b, &p, &d.path().join("x1.kv"));
    let again = save(&mut s, &*b, &p, &d.path().join("x2.kv"));
    assert_eq!(
        first_difference(&first, &again),
        None,
        "back-to-back saves differ"
    );
    generate(&mut s, &*b, &p, 5);
    save(&mut s, &*b, &p, &d.path().join("y.kv"));
    let rest = generate(&mut s, &*b, &p, M);
    assert_eq!(rest, &a_tokens[8..], "continuing after two saves");

    for (file, at) in [("x1.kv", 3), ("y.kv", 8)] {
        let r_b = backend(Kind::Naive, &p);
        let mut r = load(&d.path().join(file), &*r_b, &p, Route::Greedy);
        let resaved = save(&mut r, &*r_b, &p, &d.path().join("re.kv"));
        assert_eq!(
            first_difference(&resaved, &std::fs::read(d.path().join(file)).unwrap()),
            None,
            "{file}: re-save"
        );
        let cont = generate(&mut r, &*r_b, &p, M);
        assert_eq!(cont, &a_tokens[at..at + M], "{file}: continuation");
    }
}

type Snapshot = (
    Vec<u32>,
    usize,
    Vec<u8>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
    Vec<u32>,
);

/// Tokens, KV length and bytes, and the backend's recurrent state.
fn snapshot(s: &Session, b: &dyn ComputeBackend) -> Snapshot {
    let mut kv = Vec::new();
    for layer in 0..s.kv().config().num_layers {
        let (k, v) = s.kv().layer_raw_bytes(layer).unwrap();
        kv.extend_from_slice(k);
        kv.extend_from_slice(v);
    }
    let mut rec = RecurrentState::zeroed(b.gdn_layout().unwrap());
    let mut scratch = KvCache::new(s.kv().config().clone()).unwrap();
    b.sync_kv_to_cpu(&mut scratch, Some(&mut rec)).unwrap();
    let RecurrentState {
        h_states,
        conv_states,
        conv_positions,
        ..
    } = rec;
    (
        s.tokens().to_vec(),
        s.kv().seq_len(),
        kv,
        h_states,
        conv_states,
        conv_positions,
    )
}

/// With the KV full and one token sampled, a save is refused before anything changes; with
/// one free slot it succeeds and the loaded session stops at the same point as the unsaved one.
#[test]
fn a_full_context_is_refused_before_anything_changes() {
    let (d, p) = model();
    let full = PROMPT.len();
    let b = backend(Kind::Naive, &p);
    let mut s = start(&p, &*b, Route::Greedy, PROMPT, full);
    generate(&mut s, &*b, &p, 1);
    let before = snapshot(&s, &*b);
    let path = d.path().join("full.kv");
    let err = s
        .save_to_disk(&path, &*b, &p, &fingerprint(&p))
        .unwrap_err()
        .to_string();
    assert!(err.contains("context"), "{err}");
    assert!(!path.exists(), "a refused save wrote {path:?}");
    assert!(
        snapshot(&s, &*b) == before,
        "a refused save changed the session"
    );

    let room = full + 1;
    let a_b = backend(Kind::Naive, &p);
    let mut a = start(&p, &*a_b, Route::Greedy, PROMPT, room);
    let a_tokens = generate(&mut a, &*a_b, &p, 2);
    let a_err = a.next_token(&*a_b, &p).unwrap_err().to_string();

    let b = backend(Kind::Naive, &p);
    let mut s = start(&p, &*b, Route::Greedy, PROMPT, room);
    let mut tokens = generate(&mut s, &*b, &p, 1);
    s.save_to_disk(&path, &*b, &p, &fingerprint(&p)).unwrap();
    let r_b = backend(Kind::Naive, &p);
    let hp = p.lbc().header.hyperparams;
    let mut r = Session::load_from_disk(
        &path,
        config(room),
        hp,
        sampling(Route::Greedy),
        &*r_b,
        &fingerprint(&p),
    )
    .unwrap();
    tokens.extend(generate(&mut r, &*r_b, &p, 1));
    assert_eq!(tokens, a_tokens);
    assert_eq!(r.next_token(&*r_b, &p).unwrap_err().to_string(), a_err);
}

/// A fed history without the next token's logits (after `truncate_to`) is refused.
#[test]
fn a_history_without_logits_is_refused() {
    let (d, p) = model();
    let b = backend(Kind::Naive, &p);
    let mut s = start(&p, &*b, Route::Greedy, PROMPT, MAX_SEQ as usize);
    s.truncate_to(4);
    let path = d.path().join("t.kv");
    let err = s
        .save_to_disk(&path, &*b, &p, &fingerprint(&p))
        .unwrap_err()
        .to_string();
    assert!(err.contains("extend it before saving"), "{err}");
    assert!(!path.exists());
}

/// A file holding tokens but no logits, as an earlier version wrote after generation, is
/// refused on load with what to do.
#[test]
fn an_earlier_versions_after_generation_file_is_refused() {
    let (d, p) = model();
    let b = backend(Kind::Naive, &p);
    let mut s = start(&p, &*b, Route::Greedy, PROMPT, MAX_SEQ as usize);
    let path = d.path().join("old.kv");
    // The earlier writer's file: this save without its trailing logits section
    // (its metadata, then the f32 logits) and with the flag cleared.
    let mut bytes = save(&mut s, &*b, &p, &path);
    let vocab = p.lbc().header.hyperparams.vocab_size as usize;
    bytes.truncate(bytes.len() - PENDING_LOGITS_META_BYTES - 4 * vocab);
    let mut header = DiskKvHeader::from_bytes(bytes[..HEADER_SIZE].try_into().unwrap()).unwrap();
    assert_eq!(header.has_pending_logits, 1);
    header.has_pending_logits = 0;
    bytes[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let r_b = backend(Kind::Naive, &p);
    let hp = p.lbc().header.hyperparams;
    let err = Session::load_from_disk(
        &path,
        config(MAX_SEQ as usize),
        hp,
        sampling(Route::Greedy),
        &*r_b,
        &fingerprint(&p),
    )
    .err()
    .expect("a file without logits is refused")
    .to_string();
    assert!(err.contains("earlier Lumen version"), "{err}");
    assert!(err.contains("delete the file"), "{err}");
}
