//! S5 — CLI-facing session save/resume integration tests.
//!
//! Per the test contract:
//! > - `--session-resume nonexistent.kv` → clear error
//! > - `--session-save <path>` writes valid v2 file
//! > - `--session-resume <path> --session-save <path>` round-trip
//! > - ALL 3 weight quants (BF16, Q8, Q4)
//! > - ARGMAX-token-identical resume vs continuous session (extended
//! >   to CLI)
//!
//! These tests exercise `InferenceEngine::generate_with_session` (the entry
//! point the CLI's `run_generation` dispatcher calls) on the synthetic test
//! model, which gives us a CI-runnable, deterministic substrate. The
//! synthetic model is F32-only — the multi-quant gate is staged as a
//! hardware test (`#[ignore]`) that operators run against a real BF16/Q8/Q4
//! LBC file at `LUMEN_QWEN35_9B_PATH`.
//!
//! The argmax-token bit-identity contract is the
//! lesson: float-tolerance comparison is unreliable on the near-tie
//! landscape of real LLM logits (Qwen3.5-9B sees sub-1e-4 margins on common
//! prompts), so we compare integer token IDs.

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::disk::{
    serialize_hyperparams_le, DiskKvHeader, ModelFingerprint, HEADER_SIZE,
};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{ComputeBackend, NaiveF32Backend, RuntimeConfig, Session};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};

static UNIQ: AtomicU64 = AtomicU64::new(0);

/// Construct a temp dir + synthetic test model file. Returns `(dir, path)`.
fn synthetic_model_path() -> (std::path::PathBuf, std::path::PathBuf) {
    let id = UNIQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("lumen_session_{pid}_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model.lbc");
    let bytes = generate_test_model(&TestModelConfig::default());
    std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
    (dir, path)
}

/// Build a CPU-naive backend wired with the provider's global tensors and
/// initialized for the given hyperparams.
fn make_backend(provider: &SyncWeightProvider) -> NaiveF32Backend {
    let hp = provider.lbc().header.hyperparams;
    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&hp).unwrap();
    backend
}

/// Build the standard `RuntimeConfig` used by all tests in this module.
fn make_config() -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 128,
        collect_per_layer_timings: false,
    }
}

fn greedy_sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    }
}

/// Build a real `ModelFingerprint` for the live provider via the S4
/// production constructor. Mirrors what the CLI's `LiveModel::fingerprint`
/// helper does internally.
fn live_fingerprint(provider: &SyncWeightProvider) -> ModelFingerprint {
    let hp = provider.lbc().header.hyperparams;
    let hp_bytes = serialize_hyperparams_le(&hp);
    // Vocab bytes: synthetic model has no real tokenizer, use empty.
    let vocab: &[u8] = &[];
    let quant_tag = provider.output_proj_quant.to_u8() as u32;
    ModelFingerprint::from_live_model(
        &hp_bytes,
        vocab,
        quant_tag,
        lumen_format::LBC_VERSION,
    )
}

/// `Session::load_from_disk` on a nonexistent path produces a
/// clear, actionable error.
///
/// The CLI surfaces this as `--session-resume <path>: <inner-error>`. The
/// inner error MUST mention the path and an actionable hint; this is the
/// "clear error" contract from.
#[test]
fn s5_load_from_disk_nonexistent_path_errors_clearly() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;
    let fingerprint = live_fingerprint(&provider);

    let bogus = dir.join("does_not_exist.kv");
    let result = Session::load_from_disk(
        &bogus,
        make_config(),
        hp,
        greedy_sampling(),
        &backend,
        &fingerprint,
    );
    let err = match result {
        Ok(_) => panic!("load_from_disk on nonexistent path must error"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    // The IO error must reach the caller (not be silently swallowed).
    assert!(
        msg.to_ascii_lowercase().contains("no such file")
            || msg.to_ascii_lowercase().contains("not found")
            || msg.to_ascii_lowercase().contains("cannot find"),
        "expected actionable not-found message, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `Session::save_to_disk` writes a valid v2 disk-KV file.
///
/// Validates:
/// - File exists at the specified path after save.
/// - First four bytes are the v2 magic.
/// - Fifth-eighth bytes are version 2 (NOT v1).
/// - `has_recurrent_state` and `has_pending_logits` byte flags are
///   correctly populated (synthetic model + non-empty prompt → no GDN
///   state but pending_logits ARE present).
#[test]
fn s5_save_to_disk_writes_valid_v2_file() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;
    let fingerprint = live_fingerprint(&provider);

    let mut session = Session::new(make_config(), hp, greedy_sampling()).unwrap();
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    session.extend(&prompt, &backend, &provider).unwrap();

    let save_path = dir.join("session.kv");
    session
        .save_to_disk(&save_path, &backend, &fingerprint)
        .expect("save_to_disk must succeed on CPU naive");
    assert!(save_path.exists(), "save_to_disk must create the file");

    // Read the header back and validate v2-ness.
    let mut f = std::fs::File::open(&save_path).unwrap();
    let mut header_buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut header_buf).unwrap();
    let header = DiskKvHeader::from_bytes(&header_buf)
        .expect("header MUST parse as v2");
    assert_eq!(header.version, 2, "expected v2, got v{}", header.version);
    assert_eq!(
        header.seq_len, prompt.len() as u64,
        "header.seq_len must match prompt"
    );
    // CPU naive has no GDN; recurrent state byte must be 0.
    assert_eq!(header.has_recurrent_state, 0);
    // pending_logits byte must be 1 since `extend` sets them.
    assert_eq!(
        header.has_pending_logits, 1,
        "extend leaves pending_logits set; save must persist them"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// round-trip via `--session-resume <path> --session-save <path>`.
///
/// Builds two sessions:
/// 1. Continuous: prefill + sample 16 tokens in one go.
/// 2. Round-trip: prefill + save → load → save + sample 16 tokens.
///
/// The round-trip session saves AGAIN at the end to validate the
/// re-save-after-load path (the realistic CLI usage where the same path
/// is used for both `--session-resume` and `--session-save` so the next
/// invocation can continue). The argmax-token comparison is the
/// contract.
#[test]
fn s5_session_resume_save_round_trip_argmax_identical() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;
    let fingerprint = live_fingerprint(&provider);
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    // Continuous session.
    let mut sess_cont = Session::new(make_config(), hp, greedy_sampling()).unwrap();
    sess_cont.extend(&prompt, &backend, &provider).unwrap();
    let mut continuous_tokens: Vec<u32> = Vec::with_capacity(16);
    for _ in 0..16 {
        continuous_tokens.push(sess_cont.next_token(&backend, &provider).unwrap());
    }

    // Round-trip session: prefill + save.
    let mut sess_a = Session::new(make_config(), hp, greedy_sampling()).unwrap();
    sess_a.extend(&prompt, &backend, &provider).unwrap();
    let save_path = dir.join("round_trip.kv");
    sess_a
        .save_to_disk(&save_path, &backend, &fingerprint)
        .unwrap();
    drop(sess_a);

    // Load + sample + re-save (simulates the --resume + --save round trip).
    let mut sess_r = Session::load_from_disk(
        &save_path,
        make_config(),
        hp,
        greedy_sampling(),
        &backend,
        &fingerprint,
    )
    .unwrap();
    let mut resumed_tokens: Vec<u32> = Vec::with_capacity(16);
    for _ in 0..16 {
        resumed_tokens.push(sess_r.next_token(&backend, &provider).unwrap());
    }
    // Re-save after generation — this is the realistic CLI flow.
    sess_r
        .save_to_disk(&save_path, &backend, &fingerprint)
        .expect("re-save must succeed");

    assert_eq!(
        resumed_tokens, continuous_tokens,
        "S5 round-trip: argmax tokens MUST match position-for-position"
    );

    // The re-saved file must still be a valid v2 header.
    let mut f = std::fs::File::open(&save_path).unwrap();
    let mut buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut buf).unwrap();
    let h = DiskKvHeader::from_bytes(&buf).unwrap();
    assert_eq!(h.version, 2);
    // After full generation the saved seq_len equals `kv.seq_len`, which is
    // `prompt + generated - 1` because the LAST `next_token` call was a
    // Path A (samples pending_logits and pushes the token, defers the
    // K/V write to the next decode step). The Session::save_to_disk
    // invariant repair trims tokens to match. The deferred
    // token IS recoverable on resume via the saved pending_logits.
    let expected_kv_seq = (prompt.len() + 16 - 1) as u64;
    assert!(
        h.seq_len == expected_kv_seq || h.seq_len == (prompt.len() + 16) as u64,
        "header.seq_len must be either {expected_kv_seq} (Path-A-tail) or {} (Path B tail), got {}",
        prompt.len() + 16,
        h.seq_len,
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// fingerprint mismatch is rejected on load.
///
/// Saves with one fingerprint, attempts to load with another. The loader
/// MUST reject before reading the K/V payload to avoid silent corruption
/// . This is the safety net that protects against
/// "I changed models without clearing my session dir".
#[test]
fn s5_fingerprint_mismatch_rejected_on_load() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;

    // Save with one fingerprint.
    let fp_a = live_fingerprint(&provider);
    let mut sess = Session::new(make_config(), hp, greedy_sampling()).unwrap();
    sess.extend(&[1u32, 2, 3, 4], &backend, &provider).unwrap();
    let path = dir.join("fp_mismatch.kv");
    sess.save_to_disk(&path, &backend, &fp_a).unwrap();

    // Load with a different fingerprint.
    let fp_b = ModelFingerprint {
        model_hash: [0xCDu8; 32], // different
        weight_quant_tag: fp_a.weight_quant_tag,
        lumen_format_version: fp_a.lumen_format_version,
    };
    let result = Session::load_from_disk(
        &path,
        make_config(),
        hp,
        greedy_sampling(),
        &backend,
        &fp_b,
    );
    assert!(
        result.is_err(),
        "fingerprint mismatch must be rejected on load"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `engine.generate_with_session` produces argmax-identical
/// output to `engine.generate` for the same prompt + seed + sampling.
///
/// This is the integration-level gate for the CLI's `run_generation`
/// dispatcher: when no session flags are active, the dispatch falls
/// through to `engine.generate`; when they ARE active, it uses
/// `generate_with_session`. Both paths must produce identical tokens
/// for the same input.
#[test]
fn s5_generate_with_session_matches_generate() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;
    let prompt: Vec<u32> = vec![5, 6, 7, 8, 9, 10];

    let engine = InferenceEngine::new(make_config(), hp);
    let stop = StopCondition::MaxTokens(8);
    let sampling = greedy_sampling();

    // Path A: legacy generate.
    let a_result = engine
        .generate(&prompt, &provider, &backend, &stop, &sampling)
        .unwrap();

    // Path B: session-driven generate (fresh session, no resume).
    let mut session = Session::new(make_config(), hp, sampling.clone()).unwrap();
    let b_result = engine
        .generate_with_session(&mut session, &prompt, &provider, &backend, &stop)
        .unwrap();

    assert_eq!(
        a_result.tokens, b_result.tokens,
        "generate vs generate_with_session must produce identical token sequences"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// empty prompt on a resumed session continues generation
/// from the cached state without re-prefilling.
///
/// This is the "interactive continue" CLI pattern: `lumen run model
/// --session-resume foo.kv` with no `--prompt`. The first
/// `generate_with_session` call sees an empty prompt + a non-empty
/// session and must dispatch the no-op `extend_with_cache` path (N4 from
/// S3), then proceed to decode from the cached pending_logits (Path A).
#[test]
fn s5_resume_continues_from_cache_on_empty_prompt() {
    let (dir, model_path) = synthetic_model_path();
    let provider = SyncWeightProvider::open(&model_path).unwrap();
    let backend = make_backend(&provider);
    let hp = provider.lbc().header.hyperparams;
    let fingerprint = live_fingerprint(&provider);
    let prompt: Vec<u32> = vec![10, 20, 25, 30];

    // Build a session, run prefill, save.
    let mut sess_a = Session::new(make_config(), hp, greedy_sampling()).unwrap();
    sess_a.extend(&prompt, &backend, &provider).unwrap();
    let save_path = dir.join("continue.kv");
    sess_a.save_to_disk(&save_path, &backend, &fingerprint).unwrap();
    // Sample 8 tokens continuously for the baseline.
    let mut continuous: Vec<u32> = Vec::with_capacity(8);
    for _ in 0..8 {
        continuous.push(sess_a.next_token(&backend, &provider).unwrap());
    }
    drop(sess_a);

    // Resume the saved session and generate via `generate_with_session`
    // with EMPTY `prompt_tokens`. The engine must NOT re-prefill (N4
    // no-op path) and must produce the same tokens as the continuous
    // session would.
    let mut sess_r = Session::load_from_disk(
        &save_path,
        make_config(),
        hp,
        greedy_sampling(),
        &backend,
        &fingerprint,
    )
    .unwrap();
    let engine = InferenceEngine::new(make_config(), hp);
    let r_result = engine
        .generate_with_session(
            &mut sess_r,
            &[],
            &provider,
            &backend,
            &StopCondition::MaxTokens(8),
        )
        .unwrap();
    assert_eq!(
        r_result.tokens, continuous,
        "empty-prompt resume must continue from cache and match continuous"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// multi-quant hardware gate (BF16 / Q8 / Q4 via Metal).
///
/// All 3 weight quants must produce argmax-identical
/// resume. The synthetic test model is F32-only, so this `#[ignore]`-d
/// test runs against a real Qwen3.5-9B file staged at
/// `LUMEN_QWEN35_9B_BF16`, `LUMEN_QWEN35_9B_Q8`, `LUMEN_QWEN35_9B_Q4`.
///
/// Run with:
/// ```
/// LUMEN_QWEN35_9B_BF16=/path/to/bf16.lbc \
/// LUMEN_QWEN35_9B_Q8=/path/to/q8.lbc \
/// LUMEN_QWEN35_9B_Q4=/path/to/q4.lbc \
/// cargo test --release -p lumen-runtime --tests s5_multi_quant_hardware_gate -- --ignored
/// ```
///
/// Each model path is optional; the test skips with `[skip] ...` if a
/// given env var is unset.
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn s5_multi_quant_hardware_gate() {
    use lumen_runtime::storage::MmapConfig;
    use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
    use lumen_runtime::MetalF32Backend;

    let prompt: Vec<u32> = (1u32..=16u32).collect();
    let cases = [
        ("BF16", "LUMEN_QWEN35_9B_BF16"),
        ("Q8_0", "LUMEN_QWEN35_9B_Q8"),
        ("Q4_0", "LUMEN_QWEN35_9B_Q4"),
    ];
    let mut ran_any = false;

    for (label, env_name) in &cases {
        let model_path = match std::env::var(env_name) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("[skip {label}] {env_name} not set");
                continue;
            }
        };
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("[skip {label}] model file not found: {model_path}");
            continue;
        }
        ran_any = true;
        eprintln!("[gate {label}] {model_path}");

        let provider = MmapWeightProvider::open(
            std::path::Path::new(&model_path),
            MmapConfig::default(),
        )
        .unwrap();
        let hp = provider.lbc().header.hyperparams;

        // Build the Metal backend.
        let mut metal = MetalF32Backend::new().expect("Metal backend must construct");
        ComputeBackend::set_global_tensors(
            &mut metal,
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        ComputeBackend::init(&mut metal, &hp).unwrap();

        let config = RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 2,
            kv_precision: KvPrecision::F16,
            max_seq_len: 256,
            collect_per_layer_timings: false,
        };
        let sampling = SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        };

        // Continuous session.
        let mut sess_cont = Session::new(config.clone(), hp, sampling.clone()).unwrap();
        sess_cont.extend(&prompt, &metal, &provider).unwrap();
        let mut continuous: Vec<u32> = Vec::with_capacity(16);
        for _ in 0..16 {
            continuous.push(sess_cont.next_token(&metal, &provider).unwrap());
        }
        drop(sess_cont);

        // Save + restore + sample.
        let dir = std::env::temp_dir()
            .join(format!("lumen_session_metal_{label}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.kv");
        let hp_bytes = serialize_hyperparams_le(&hp);
        let fingerprint = ModelFingerprint::from_live_model(
            &hp_bytes,
            &[],
            provider.output_proj_quant.to_u8() as u32,
            lumen_format::LBC_VERSION,
        );
        let mut sess_a = Session::new(config.clone(), hp, sampling.clone()).unwrap();
        sess_a.extend(&prompt, &metal, &provider).unwrap();
        sess_a.save_to_disk(&path, &metal, &fingerprint).unwrap();

        let mut sess_r = Session::load_from_disk(
            &path,
            config.clone(),
            hp,
            sampling.clone(),
            &metal,
            &fingerprint,
        )
        .unwrap();
        let mut resumed: Vec<u32> = Vec::with_capacity(16);
        for _ in 0..16 {
            resumed.push(sess_r.next_token(&metal, &provider).unwrap());
        }

        assert_eq!(
            resumed, continuous,
            "S5 multi-quant hardware gate {label}: argmax MUST match",
        );
        eprintln!("[pass {label}] 16/16 argmax tokens identical");
        let _ = std::fs::remove_dir_all(&dir);
    }

    if !ran_any {
        eprintln!(
            "[skip] none of LUMEN_QWEN35_9B_BF16 / _Q8 / _Q4 are set; \
             multi-quant gate did not run"
        );
    }
}
