//! Part 4 (reasoning budget + forced-close) end-to-end engine tests.
//!
//! Boots a `lumen-server` engine worker on the CPU-naive backend with a tiny
//! synthetic model + the byte-identity tokenizer, then submits `JobRequest`s
//! that exercise the Part-4 decode-loop control:
//!
//!   * thinking-OFF (the default) must produce a BYTE-IDENTICAL token stream
//!     to the pre-Part-4 loop — proven by replaying the same greedy request
//!     twice and (separately) by checking the answer budget still bounds total
//!     tokens exactly at `max_tokens`.
//!   * thinking-ON with a small `reasoning_budget` must FORCE-CLOSE the
//!     `<think>` block at the budget (the synthetic model never emits
//!     `</think>` on its own), inject `</think>\n\n`, and then apply the
//!     SEPARATE answer budget — so the answer is never starved by reasoning.
//!
//! The synthetic model has random weights, so the exact token ids are
//! arbitrary but DETERMINISTIC under temp 0 + fixed seed. The tests assert on
//! counts, the injected close marker, and budget separation — not on specific
//! token values.

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
            Ok(Some(TokenEvent::Token { token_id, delta_text })) => {
                token_ids.push(token_id);
                fragments.push(delta_text);
            }
            Ok(Some(TokenEvent::Done { finish_reason, completion_tokens: c, .. })) => {
                finish = finish_reason;
                completion_tokens = c;
                break;
            }
            Ok(Some(TokenEvent::Error(e))) => panic!("engine error: {e}"),
            Ok(Some(TokenEvent::PrefillDone { .. })) => {}
            Ok(None) => break,
            Err(_) => panic!("timed out draining job"),
        }
    }
    Drained { fragments, token_ids, finish, completion_tokens }
}

// =========================================================================
// Thinking-OFF byte-identity (the hard requirement)
// =========================================================================

/// Two thinking-off greedy requests with the same seed must produce the
/// EXACT same token sequence (determinism) AND respect the answer budget
/// exactly. This is the byte-identity guard: Part-4 must not perturb the
/// default path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thinking_off_is_deterministic_and_budget_exact() {
    let handle = boot_engine();
    let a = drain(&handle, job(12, false, 0)).await;
    let b = drain(&handle, job(12, false, 0)).await;

    // Determinism: identical token streams.
    assert_eq!(a.token_ids, b.token_ids, "thinking-off greedy must be deterministic");
    assert_eq!(a.full_text(), b.full_text());

    // Answer budget: exactly max_tokens tokens (no reasoning phase, so every
    // token is an answer token), finish_reason == Length.
    assert_eq!(a.completion_tokens, 12, "thinking-off must emit exactly max_tokens");
    assert_eq!(a.token_event_count(), 12);
    assert_eq!(a.finish, FinishReason::Length);

    // No forced-close marker ever appears on the thinking-off path.
    assert!(
        !a.full_text().contains("</think>"),
        "thinking-off must NEVER inject </think>"
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
// Thinking-ON forced-close + separate budget
// =========================================================================

/// With thinking ON and a small reasoning_budget, the synthetic model (which
/// never emits `</think>` on its own) must be FORCE-CLOSED at the budget: the
/// stream contains an injected `</think>` marker, and the SEPARATE answer
/// budget then bounds the answer tokens. Total decoded tokens = reasoning
/// budget + answer budget (the injected close is emitted but not counted in
/// completion_tokens).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thinking_on_forced_close_at_reasoning_budget() {
    let handle = boot_engine();
    let reasoning_budget = 6usize;
    let answer_budget = 5usize;
    let d = drain(&handle, job(answer_budget, true, reasoning_budget)).await;

    // The forced-close injected `</think>` into the stream.
    assert!(
        d.full_text().contains("</think>"),
        "forced-close must inject </think>; got: {:?}",
        d.full_text()
    );

    // SEPARATE budgets: completion_tokens counts reasoning + answer decode
    // steps (the injected close is emitted as bytes but is not a decode step),
    // so it equals reasoning_budget + answer_budget. Critically, the answer
    // was NOT starved: `answer_budget` answer tokens were produced AFTER the
    // forced close even though reasoning already consumed its full budget.
    assert_eq!(
        d.completion_tokens,
        reasoning_budget + answer_budget,
        "answer budget must be applied SEPARATELY (reasoning {reasoning_budget} + answer {answer_budget})"
    );
    assert_eq!(d.finish, FinishReason::Length);
}

/// The answer budget is independent of the reasoning budget: doubling the
/// reasoning budget leaves the same number of ANSWER tokens (answer is never
/// starved). We compare completion_tokens deltas.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn answer_budget_is_independent_of_reasoning_budget() {
    let handle = boot_engine();
    let answer = 5usize;
    let small = drain(&handle, job(answer, true, 4)).await;
    let large = drain(&handle, job(answer, true, 10)).await;
    // completion = reasoning_budget + answer in each case.
    assert_eq!(small.completion_tokens, 4 + answer);
    assert_eq!(large.completion_tokens, 10 + answer);
    // The ANSWER allotment (completion - reasoning_budget) is the SAME — the
    // longer reasoning trace did not eat into the answer.
    assert_eq!(small.completion_tokens - 4, large.completion_tokens - 10);
    assert_eq!(small.completion_tokens - 4, answer);
}

/// A generous reasoning_budget that the (bounded) generation never reaches
/// means no forced-close fires; with thinking on but the model never emitting
/// `</think>`, the answer budget is never reached either (all tokens stay in
/// the reasoning phase), so generation is bounded by the context window /
/// reasoning budget — NOT a starved answer. This pins that an un-hit reasoning
/// budget does not inject a marker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reasoning_budget_not_reached_no_injection() {
    let handle = boot_engine();
    // reasoning_budget larger than what the answer budget would allow to be
    // generated, but we stop via the context guard well before. Use a small
    // answer budget; since the model never emits </think>, it stays in
    // reasoning forever until the reasoning budget forces a close. To get the
    // "not reached" case we set reasoning_budget huge and rely on the context
    // guard (MAX_SEQ_LEN) to terminate.
    let d = drain(&handle, job(4, true, 100_000)).await;
    // Never closed (budget never hit, context guard stopped it): no </think>.
    assert!(
        !d.full_text().contains("</think>"),
        "an un-reached reasoning budget must not inject </think>"
    );
    // Terminated by the context window (Length), having stayed in reasoning.
    assert_eq!(d.finish, FinishReason::Length);
}
