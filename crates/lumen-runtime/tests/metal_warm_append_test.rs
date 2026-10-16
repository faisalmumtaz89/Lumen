//! Metal on a real Qwen3.5 model (`#[ignore]`): a batched prefill that extends a session after
//! generation continues from the state the decode left. A session that generated `k` tokens and
//! then appends a suffix in one batch is compared with a session loaded from a file saved after
//! the same `k` tokens, appending the same suffix: their files after the append, the next `M`
//! greedy tokens and their files after those are equal byte for byte, for `k` of 1, 2, 7 and 64.
//! The append is also run on a backend whose earlier, longer prefill had already sized its
//! scratch, and gives the same file as on a fresh backend.
//!
//! LUMEN_QWEN35_9B_Q8=/path/to/qwen3-5-9b-Q8_0.lbc cargo test --release -p lumen-runtime \
//!   --test metal_warm_append_test -- --ignored --nocapture
#![cfg(target_os = "macos")]

use lumen_format::QuantScheme;
use lumen_runtime::kv::disk::{serialize_hyperparams_le, ModelFingerprint};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::{ComputeBackend, MetalF32Backend, RuntimeConfig, SamplingParams, Session};
use std::path::{Path, PathBuf};

const CONTEXT: usize = 2048;
/// Greedy tokens compared after the append.
const M: usize = 8;
const PROMPT_LEN: usize = 600;
const SUFFIX_LEN: usize = 400;

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

fn config() -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: lumen_runtime::pipeline::PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F16,
        max_seq_len: CONTEXT,
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

/// Token ids of an ordinary vocabulary range, deterministic per `salt`.
fn tokens(len: usize, salt: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (1000 + (i * 7919 + salt * 104_729) % 90_000) as u32)
        .collect()
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

    fn start(&self, b: &MetalF32Backend, prompt: &[u32]) -> Session {
        b.reset_recurrent_state();
        let mut s = Session::new(config(), self.hp(), greedy()).unwrap();
        s.extend(prompt, b, &self.provider).unwrap();
        s
    }

    fn generate(&self, s: &mut Session, b: &MetalF32Backend, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| s.next_token(b, &self.provider).unwrap())
            .collect()
    }

    /// Append `full`'s tail past the session's tokens in one batched prefill.
    fn append(&self, s: &mut Session, b: &MetalF32Backend, full: &[u32]) {
        let reused = s.token_count();
        let r = s
            .extend_with_cache(full, b, &self.provider, Session::DEFAULT_SUFFIX_THRESHOLD)
            .unwrap();
        assert!(!r.used_single_token_path && !r.fell_back_to_cold);
        assert_eq!(r.reused_prefix_len, reused);
    }

    fn save(&self, s: &mut Session, b: &MetalF32Backend, path: &Path) -> Vec<u8> {
        s.save_to_disk(path, b, &self.provider, &self.fp).unwrap();
        std::fs::read(path).unwrap()
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

#[test]
#[ignore]
fn a_batched_append_after_generation_continues_the_decoded_state() {
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
    let prompt = tokens(PROMPT_LEN, 1);
    let suffix = tokens(SUFFIX_LEN, 2);

    let x = metal(&model.provider);
    for k in [1usize, 2, 7, 64] {
        let tag = format!("k={k}");
        let mut a = model.start(&x, &prompt);
        let generated = model.generate(&mut a, &x, k);
        let saved = dir.path().join("generated.kv");
        model.save(&mut a, &x, &saved);
        let mut full = prompt.clone();
        full.extend(&generated);
        full.extend(&suffix);

        let mut live = model.start(&x, &prompt);
        assert_eq!(model.generate(&mut live, &x, k), generated, "{tag}");
        model.append(&mut live, &x, &full);
        let live_file = model.save(&mut live, &x, &dir.path().join("live.kv"));
        let live_next = model.generate(&mut live, &x, M);
        let live_after = model.save(&mut live, &x, &dir.path().join("live_after.kv"));

        let mut loaded =
            Session::load_from_disk(&saved, config(), model.hp(), greedy(), &x, &model.fp).unwrap();
        model.append(&mut loaded, &x, &full);
        let loaded_file = model.save(&mut loaded, &x, &dir.path().join("loaded.kv"));
        let loaded_next = model.generate(&mut loaded, &x, M);
        let loaded_after = model.save(&mut loaded, &x, &dir.path().join("loaded_after.kv"));

        assert_eq!(
            first_difference(&live_file, &loaded_file),
            None,
            "{tag}: the file after the append"
        );
        assert_eq!(live_next, loaded_next, "{tag}: the tokens after the append");
        assert_eq!(
            first_difference(&live_after, &loaded_after),
            None,
            "{tag}: the file after the next tokens"
        );
    }

    // A fresh backend sizes the append's scratch itself: the same file as on a backend whose
    // earlier prefill of the whole sequence had sized it already.
    let mut full = prompt.clone();
    full.extend(&suffix);
    let fresh = metal(&model.provider);
    let mut s = model.start(&fresh, &prompt);
    model.append(&mut s, &fresh, &full);
    let fresh_file = model.save(&mut s, &fresh, &dir.path().join("fresh.kv"));
    drop((s, fresh));
    let sized = metal(&model.provider);
    model.start(&sized, &full);
    let mut s = model.start(&sized, &prompt);
    model.append(&mut s, &sized, &full);
    let sized_file = model.save(&mut s, &sized, &dir.path().join("sized.kv"));
    assert_eq!(
        first_difference(&fresh_file, &sized_file),
        None,
        "the append on a fresh backend"
    );
}
