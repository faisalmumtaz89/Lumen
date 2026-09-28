//! Token limit + reasoning budget (forced-close) end-to-end engine tests.
//!
//! Boots a `lumen-server` engine worker on the CPU-naive backend with a tiny
//! synthetic model + the byte-identity tokenizer, then submits `JobRequest`s
//! that exercise the decode-loop control:
//!
//!   * thinking-OFF (the default) is deterministic and stops at exactly
//!     `max_tokens`.
//!   * thinking-ON: `max_tokens` bounds reasoning and answer together, as both
//!     APIs define it. A small `reasoning_budget` FORCE-CLOSES the `<think>`
//!     block at the budget (the synthetic model never emits `</think>` on its
//!     own) by injecting `</think>\n\n`, and the answer uses what remains; a
//!     limit reached while reasoning ends the reply there.
//!
//! The synthetic model has random weights, so the exact token ids are
//! arbitrary but DETERMINISTIC under temp 0 + fixed seed. The tests assert on
//! counts and the injected close marker, not on specific token values.

use std::sync::Arc;
use std::time::Duration;

use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::RuntimeConfig;

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_server::{
    EngineHandle, EngineWorker, FinishReason, IdentityByteTokenizer, JobRequest, ModelInfo,
    TokenEvent, Tokenize,
};

const MAX_SEQ_LEN: usize = 256;

/// Spawn an engine worker on the CPU-naive backend with a synthetic model.
fn boot_engine() -> EngineHandle {
    let cfg = TestModelConfig {
        vocab_size: 256,
        max_seq_len: MAX_SEQ_LEN as u32,
        ..TestModelConfig::default()
    };
    let bytes = generate_test_model(&cfg);
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("test_model.lbc");
    std::fs::write(&path, &bytes).unwrap();
    // Keep the tempdir alive for the process lifetime (leak is fine in a test).
    std::mem::forget(tmp);

    let provider = SyncWeightProvider::open(&path).unwrap();
    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();
    let hyperparams = provider.lbc().header.hyperparams;
    let runtime_cfg = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: MAX_SEQ_LEN,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: "lumen-test:reasoning".into(),
        owned_by: "lumen-test".into(),
        created: 0,
        context_length: MAX_SEQ_LEN,
    };
    let tokenizer: Arc<dyn Tokenize> = Arc::new(IdentityByteTokenizer::default());
    EngineWorker::spawn(
        runtime_cfg,
        hyperparams,
        Box::new(backend),
        Arc::new(provider),
        tokenizer,
        model_info,
        8,
    )
}

/// A drained job: the ordered decoded text fragments, their token ids, the
/// finish reason, and the reported completion-token count.
struct Drained {
    fragments: Vec<String>,
    token_ids: Vec<u32>,
    finish: FinishReason,
    completion_tokens: usize,
}

impl Drained {
    fn full_text(&self) -> String {
        self.fragments.concat()
    }
    /// Count of `TokenEvent::Token` events (each decode step OR the single
    /// forced-close injection counts as one event).
    fn token_event_count(&self) -> usize {
        self.fragments.len()
    }
}

fn job(max_tokens: usize, enable_thinking: bool, reasoning_budget: usize) -> JobRequest {
    JobRequest {
        prompt_tokens: vec![104, 105], // "h", "i"
        max_tokens,
        stop_text: Vec::new(),
        eos_token_ids: Vec::new(),
        ignore_eos: false,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..SamplingParams::default()
        },
        suffix_threshold: 32,
        enable_thinking,
        reasoning_budget,
    }
}

async fn drain(handle: &EngineHandle, req: JobRequest) -> Drained {
    let mut rx = handle.submit(req, 256).await.expect("submit");
    let mut fragments = Vec::new();
    let mut token_ids = Vec::new();
    let mut finish = FinishReason::Stop;
    let mut completion_tokens = 0usize;
    loop {
        match tokio::time::timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(TokenEvent::Token {
                token_id,
                delta_text,
            })) => {
                token_ids.push(token_id);
                fragments.push(delta_text);
            }
            Ok(Some(TokenEvent::Done {
                finish_reason,
                completion_tokens: c,
                ..
            })) => {
                finish = finish_reason;
                completion_tokens = c;
                break;
            }
            Ok(Some(TokenEvent::Error(e))) => panic!("engine error: {e}"),
            Ok(Some(TokenEvent::PrefillDone { .. })) => {}
            Ok(Some(TokenEvent::BenchTokenIds { .. })) => {}
            Ok(None) => break,
            Err(_) => panic!("timed out draining job"),
        }
    }
    Drained {
        fragments,
        token_ids,
        finish,
        completion_tokens,
    }
}

// =========================================================================
// Thinking-OFF byte-identity (the hard requirement)
// =========================================================================

/// Two thinking-off greedy requests with the same seed must produce the
/// EXACT same token sequence (determinism) AND stop at exactly
/// `max_tokens`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thinking_off_is_deterministic_and_budget_exact() {
    let handle = boot_engine();
    let a = drain(&handle, job(12, false, 0)).await;
    let b = drain(&handle, job(12, false, 0)).await;

    // Determinism: identical token streams.
    assert_eq!(
        a.token_ids, b.token_ids,
        "thinking-off greedy must be deterministic"
    );
    assert_eq!(a.full_text(), b.full_text());

    // Exactly max_tokens tokens, finish_reason == Length.
    assert_eq!(
        a.completion_tokens, 12,
        "thinking-off must emit exactly max_tokens"
    );
    assert_eq!(a.token_event_count(), 12);
    assert_eq!(a.finish, FinishReason::Length);

    // No forced-close marker ever appears on the thinking-off path.
    assert!(
        !a.full_text().contains("</think>"),
        "thinking-off must NEVER inject </think>"
    );
}

/// An EOS token ends the answer where it appears and renders nothing; with
/// `ignore_eos` it still renders nothing but decoding carries on to the
/// budget, and the token counts toward it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ignore_eos_keeps_decoding_and_renders_no_eos_text() {
    let handle = boot_engine();
    let free = drain(&handle, job(12, false, 0)).await;
    // An ASCII token, so the byte tokenizer renders it whole and the text
    // comparison below is byte-exact.
    let eos = *free
        .token_ids
        .iter()
        .find(|&&t| t < 0x80)
        .expect("the greedy stream has an ASCII token to use as EOS");
    let first = free.token_ids.iter().position(|&t| t == eos).unwrap();

    let mut stop = job(12, false, 0);
    stop.eos_token_ids = vec![eos];
    let stopped = drain(&handle, stop).await;
    assert_eq!(stopped.finish, FinishReason::Stop);
    assert_eq!(stopped.token_ids, free.token_ids[..first].to_vec());
    assert_eq!(stopped.completion_tokens, first + 1);

    let mut go = job(12, false, 0);
    go.eos_token_ids = vec![eos];
    go.ignore_eos = true;
    let ignored = drain(&handle, go).await;
    assert_eq!(ignored.finish, FinishReason::Length);
    assert_eq!(ignored.completion_tokens, 12);
    let kept: Vec<usize> = (0..12).filter(|&i| free.token_ids[i] != eos).collect();
    assert!(
        kept.len() < 12,
        "the EOS token must be skipped, not emitted"
    );
    assert_eq!(
        ignored.token_ids,
        kept.iter().map(|&i| free.token_ids[i]).collect::<Vec<_>>()
    );
    assert_eq!(
        ignored.full_text(),
        kept.iter()
            .map(|&i| free.fragments[i].as_str())
            .collect::<String>()
    );
}

/// `reasoning_budget` is ignored entirely when thinking is off: a request with
/// thinking-off + a (meaningless) reasoning_budget behaves identically to one
/// with reasoning_budget = 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thinking_off_ignores_reasoning_budget() {
    let handle = boot_engine();
    let with_budget = drain(&handle, job(10, false, 4)).await;
    let no_budget = drain(&handle, job(10, false, 0)).await;
    assert_eq!(
        with_budget.token_ids, no_budget.token_ids,
        "reasoning_budget must be a no-op when thinking is off"
    );
    assert_eq!(with_budget.completion_tokens, 10);
    assert!(!with_budget.full_text().contains("</think>"));
}

/// `max_tokens == 0` degenerate input: the pre-Part-4 loop emitted ZERO tokens
/// with finish_reason == Stop. Part 4 must preserve that exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_tokens_zero_emits_nothing() {
    let handle = boot_engine();
    let d = drain(&handle, job(0, false, 0)).await;
    assert_eq!(d.completion_tokens, 0, "max_tokens=0 must emit no tokens");
    assert_eq!(d.token_event_count(), 0);
    assert_eq!(d.finish, FinishReason::Stop);
}

// =========================================================================
// Thinking-ON: one token limit, reasoning included
// =========================================================================

/// With thinking ON and a small reasoning_budget, the synthetic model (which
/// never emits `</think>` on its own) is FORCE-CLOSED at the budget: the stream
/// contains an injected `</think>` marker, and the answer then gets the rest of
/// `max_tokens`. completion_tokens counts decode steps (the injected close is
/// emitted as bytes but is not one), so it is exactly `max_tokens`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thinking_on_forced_close_at_reasoning_budget() {
    let handle = boot_engine();
    let d = drain(&handle, job(11, true, 6)).await;
    assert!(
        d.full_text().contains("</think>"),
        "forced-close must inject </think>; got: {:?}",
        d.full_text()
    );
    assert_eq!(d.completion_tokens, 11, "reasoning + answer = max_tokens");
    assert_eq!(d.finish, FinishReason::Length);
}

/// The reasoning budget moves tokens between reasoning and answer, never past
/// `max_tokens`: a longer trace leaves a shorter answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_tokens_includes_reasoning() {
    let handle = boot_engine();
    let small = drain(&handle, job(15, true, 4)).await;
    let large = drain(&handle, job(15, true, 10)).await;
    assert_eq!(small.completion_tokens, 15);
    assert_eq!(large.completion_tokens, 15);
    // Answer tokens: the Token events after the injected close.
    let answer = |d: &Drained| {
        let close = d.fragments.iter().position(|f| f.contains("</think>"));
        d.fragments.len() - 1 - close.expect("forced close")
    };
    assert_eq!(answer(&small), 15 - 4);
    assert_eq!(answer(&large), 15 - 10);
}

/// A limit reached while reasoning ends the reply there: no forced close, no
/// answer, finish Length (Anthropic `max_tokens`, OpenAI `length`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limit_reached_while_reasoning_ends_the_reply() {
    let handle = boot_engine();
    let d = drain(&handle, job(4, true, 100_000)).await;
    assert!(
        !d.full_text().contains("</think>"),
        "no forced close before the budget"
    );
    assert_eq!(d.completion_tokens, 4);
    assert_eq!(d.finish, FinishReason::Length);
}
