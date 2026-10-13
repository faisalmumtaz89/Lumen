//! Metal on a real Qwen3.5 model (`#[ignore]`): a session saved after `k` generated tokens and
//! loaded again continues exactly like one that never stopped, on the pipelined greedy route
//! and on the penalised logits route, and the file holds the GDN state decode actually
//! updates. Runs, all from one prompt: A never saved; C saved and continued live; B loaded
//! into a fresh backend; B' loaded into a backend that first ran another prompt. B, B' and C
//! emit A's tokens and their final files equal A's byte for byte (save time aside); a file
//! saved after two or more generated tokens differs from the one saved after the prompt in
//! every GDN layer's state. On the sampled route a loaded session reseeds its sampler, so B
//! and B' are compared with A reseeded by `set_sampling` after `k` tokens. A greedy session
//! switched to the penalised route and saved before its next token continues like one
//! switched without saving.
//!
//! LUMEN_QWEN35_9B_Q8=/path/to/qwen3-5-9b-Q8_0.lbc cargo test --release -p lumen-runtime \
//!   --test metal_session_resume_test -- --ignored --nocapture
#![cfg(target_os = "macos")]

use lumen_format::QuantScheme;
use lumen_runtime::kv::disk::{load_into, serialize_hyperparams_le, ModelFingerprint};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::{ComputeBackend, MetalF32Backend, RuntimeConfig, SamplingParams, Session};
use std::path::{Path, PathBuf};

const CONTEXT: usize = 512;
const PROMPT: &[u32] = &[
    248045, 846, 198, 3710, 369, 279, 6511, 315, 9625, 30, 248046, 198, 248045, 74455, 198,
];
const OTHER: &[u32] = &[248045, 846, 198, 9707, 11, 1246, 525, 498, 30, 248046, 198];
const M: usize = 32;

fn metal(provider: &MmapWeightProvider) -> MetalF32Backend {
    if std::env::var_os("LUMEN_METAL_MMAP_ONLY").is_none() {
        std::env::set_var("LUMEN_METAL_MMAP_ONLY", "1");
    }
    let mut m = MetalF32Backend::new().expect("Metal");
    m.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    let raw = [
        QuantScheme::Q8_0,
        QuantScheme::Q4_0,
        QuantScheme::F16,
        QuantScheme::Bf16,
    ];
    if raw.contains(&provider.output_proj_quant) && !provider.output_proj_raw.is_empty() {
        m.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if raw[..3].contains(&provider.embedding_quant) && !provider.embedding_raw.is_empty() {
        m.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    if provider.weight_tying {
        m.set_weight_tying(true);
    }
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = CONTEXT as u32;
    m.init(&hp).unwrap();
    m.preload_weights(provider).unwrap();
    m
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Greedy,
    Penalised,
    Sampled,
}

fn sampling(route: Route) -> SamplingParams {
    let (temperature, penalty) = match route {
        Route::Greedy => (0.0, 1.0),
        Route::Penalised => (0.0, 1.3),
        Route::Sampled => (0.8, 1.0),
    };
    SamplingParams {
        temperature,
        repetition_penalty: Some(penalty),
        seed: Some(1),
        ..Default::default()
    }
}

fn config() -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: lumen_runtime::pipeline::PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F16,
        max_seq_len: CONTEXT,
        collect_per_layer_timings: false,
    }
}

struct Model {
    provider: MmapWeightProvider,
    fp: ModelFingerprint,
}

impl Model {
    fn hp(&self) -> lumen_format::ModelHyperparams {
        let mut hp = self.provider.lbc().header.hyperparams;
        hp.max_seq_len = CONTEXT as u32;
        hp
    }

    fn start(&self, b: &MetalF32Backend, route: Route, prompt: &[u32]) -> Session {
        b.reset_recurrent_state();
        let mut s = Session::new(config(), self.hp(), sampling(route)).unwrap();
        s.extend(prompt, b, &self.provider).unwrap();
        s
    }

    fn generate(&self, s: &mut Session, b: &MetalF32Backend, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| s.next_token(b, &self.provider).unwrap())
            .collect()
    }

    fn save(&self, s: &mut Session, b: &MetalF32Backend, path: &Path) -> Vec<u8> {
        s.save_to_disk(path, b, &self.provider, &self.fp).unwrap();
        std::fs::read(path).unwrap()
    }

    fn load(&self, path: &Path, b: &MetalF32Backend, route: Route) -> Session {
        Session::load_from_disk(path, config(), self.hp(), sampling(route), b, &self.fp).unwrap()
    }
}

/// `a` and `b` equal except for the header's save time (bytes 44..48); otherwise the first
/// differing offset.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| !(44..48).contains(&i) && a[i] != b[i])
}

fn h_states(path: &Path) -> Vec<Vec<u8>> {
    load_into(path, None, None, None, None)
        .unwrap()
        .recurrent
        .expect("a GDN model saves its recurrent state")
        .h_states
}

fn route_cases(model: &Model, route: Route, dir: &Path) {
    let mut after_prompt: Option<PathBuf> = None;
    for k in [0usize, 1, 2, 3, 8] {
        let tag = format!("{route:?} k={k}");
        let x = metal(&model.provider);

        let mut a = model.start(&x, route, PROMPT);
        let a_tokens = model.generate(&mut a, &x, k + M);
        let a_file = model.save(&mut a, &x, &dir.join("a.kv"));

        let mut c = model.start(&x, route, PROMPT);
        let mut c_tokens = model.generate(&mut c, &x, k);
        let saved = dir.join(format!("{route:?}_{k}.kv"));
        model.save(&mut c, &x, &saved);
        c_tokens.extend(model.generate(&mut c, &x, M));
        assert_eq!(c_tokens, a_tokens, "{tag}: C differs from A");
        let c_file = model.save(&mut c, &x, &dir.join("c.kv"));
        assert_eq!(
            first_difference(&c_file, &a_file),
            None,
            "{tag}: C's final file"
        );

        if k == 0 {
            after_prompt = Some(saved.clone());
        } else if k >= 2 {
            let (h0, hk) = (h_states(after_prompt.as_ref().unwrap()), h_states(&saved));
            let layers = x.gdn_layout().unwrap().num_gdn_layers as usize;
            assert_eq!(
                (h0.len(), hk.len()),
                (layers, layers),
                "{tag}: GDN layers saved"
            );
            let unchanged = h0.iter().zip(&hk).filter(|(p, q)| p == q).count();
            assert_eq!(
                unchanged, 0,
                "{tag}: GDN state saved after {k} tokens equals the prompt's in {unchanged} layers"
            );
        }

        let (want, want_file) = if let Route::Sampled = route {
            let mut s = model.start(&x, route, PROMPT);
            model.generate(&mut s, &x, k);
            s.set_sampling(sampling(route));
            let tokens = model.generate(&mut s, &x, M);
            (tokens, model.save(&mut s, &x, &dir.join("s.kv")))
        } else {
            (a_tokens[k..].to_vec(), a_file)
        };
        let want_history: Vec<u32> = PROMPT.iter().chain(&a_tokens[..k]).copied().collect();
        let fresh = metal(&model.provider);
        let mut other = model.start(&x, route, OTHER);
        model.generate(&mut other, &x, 8);
        for (name, b) in [("B", &fresh), ("B'", &x)] {
            let mut r = model.load(&saved, b, route);
            assert_eq!(r.tokens(), &want_history[..], "{tag} {name}: history");
            let cont = model.generate(&mut r, b, M);
            assert_eq!(cont, want, "{tag} {name}: continuation");
            let r_file = model.save(&mut r, b, &dir.join("r.kv"));
            assert_eq!(
                first_difference(&r_file, &want_file),
                None,
                "{tag} {name}: final file"
            );
        }
        let reference = if let Route::Sampled = route {
            "A reseeded after k tokens"
        } else {
            "A"
        };
        eprintln!(
            "  {tag}: C matches A, B and B' match {reference}, over {M} tokens and final files"
        );
    }
}

fn route_switch_case(model: &Model, dir: &Path) {
    let k = 3;
    let x = metal(&model.provider);

    let mut a = model.start(&x, Route::Greedy, PROMPT);
    model.generate(&mut a, &x, k);
    a.set_sampling(sampling(Route::Penalised));
    let a_tokens = model.generate(&mut a, &x, M);
    let a_file = model.save(&mut a, &x, &dir.join("a.kv"));

    let mut c = model.start(&x, Route::Greedy, PROMPT);
    model.generate(&mut c, &x, k);
    c.set_sampling(sampling(Route::Penalised));
    let saved = dir.join("switch.kv");
    model.save(&mut c, &x, &saved);
    assert_eq!(model.generate(&mut c, &x, M), a_tokens, "switch: C");
    let c_file = model.save(&mut c, &x, &dir.join("c.kv"));
    assert_eq!(
        first_difference(&c_file, &a_file),
        None,
        "switch: C's final file"
    );

    let fresh = metal(&model.provider);
    let mut r = model.load(&saved, &fresh, Route::Penalised);
    assert_eq!(model.generate(&mut r, &fresh, M), a_tokens, "switch: B");
    let r_file = model.save(&mut r, &fresh, &dir.join("r.kv"));
    assert_eq!(
        first_difference(&r_file, &a_file),
        None,
        "switch: B's final file"
    );
    eprintln!("  switch k={k}: A, B, C agree over {M} tokens; final files equal");
}

#[test]
#[ignore = "needs Apple Silicon and a real Qwen3.5 LBC in LUMEN_QWEN35_9B_Q8"]
fn metal_sessions_resume_like_a_session_that_never_stopped() {
    let path = PathBuf::from(
        std::env::var("LUMEN_QWEN35_9B_Q8").expect("set LUMEN_QWEN35_9B_Q8 to an LBC"),
    );
    let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();
    let fp = ModelFingerprint::from_live_model(
        &serialize_hyperparams_le(&provider.lbc().header.hyperparams),
        &[],
        provider.output_proj_quant.to_u8() as u32,
        lumen_format::LBC_VERSION,
    );
    let model = Model { provider, fp };
    let dir = tempfile::tempdir().unwrap();
    for route in [Route::Greedy, Route::Penalised, Route::Sampled] {
        route_cases(&model, route, dir.path());
    }
    route_switch_case(&model, dir.path());
}
