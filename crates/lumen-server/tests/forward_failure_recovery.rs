//! A forward pass that fails part-way leaves KV and recurrent state partly advanced; the worker
//! must not continue from it. On a GDN model served by the CPU backend, a request whose prefill
//! or decode step fails at a later layer is answered with the error, and the next request then
//! produces what a fresh worker produces.

#[path = "../../lumen-runtime/tests/common/gdn_hybrid.rs"]
#[allow(dead_code)]
mod gdn_hybrid;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gdn_hybrid::{build_qwen35_cpu_lbc, Qwen35Fixture};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::cache::{
    CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider,
};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{RuntimeConfig, RuntimeError};
use lumen_server::{
    EngineHandle, EngineWorker, FinishReason, IdentityByteTokenizer, JobRequest, ModelInfo,
    TokenEvent, Tokenize,
};

const MAX_SEQ_LEN: usize = 96;
const FAILING_LAYER: usize = 2;
/// Reads of `FAILING_LAYER` in the decode test's warm append before the one that fails.
const DECODE_FAILURE_READS: usize = 4;

/// A provider that fails one read of `FAILING_LAYER`: the one made when `reads_left` is 0
/// (`usize::MAX`: none).
struct FlakyLayer {
    inner: SyncWeightProvider,
    reads_left: Arc<AtomicUsize>,
}

impl WeightProvider for FlakyLayer {
    fn prefetch_layer(
        &self,
        layer: usize,
        priority: PrefetchPriority,
    ) -> Result<PrefetchHandle, RuntimeError> {
        self.inner.prefetch_layer(layer, priority)
    }
    fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        self.inner.get_layer_blocking(layer)
    }
    fn get_layer_raw(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        if layer == FAILING_LAYER {
            match self.reads_left.load(Ordering::SeqCst) {
                0 => {
                    self.reads_left.store(usize::MAX, Ordering::SeqCst);
                    return Err(RuntimeError::LayerUnavailable {
                        layer,
                        reason: "injected read failure".into(),
                    });
                }
                usize::MAX => {}
                n => self.reads_left.store(n - 1, Ordering::SeqCst),
            }
        }
        self.inner.get_layer_raw(layer)
    }
    fn try_get_layer(&self, layer: usize) -> Option<LayerView> {
        self.inner.try_get_layer(layer)
    }
    fn release_layer_hint(&self, layer: usize) {
        self.inner.release_layer_hint(layer)
    }
    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }
}

fn boot(path: &std::path::Path, reads_left: Arc<AtomicUsize>) -> EngineHandle {
    let provider = SyncWeightProvider::open(path).unwrap();
    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    let hyperparams = provider.lbc().header.hyperparams;
    backend.init(&hyperparams).unwrap();
    backend.preload_weights(&provider).unwrap();
    assert!(backend.caps().gdn);
    let runtime_cfg = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: MAX_SEQ_LEN,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: "lumen-test:qwen35-cpu".into(),
        owned_by: "lumen-test".into(),
        created: 0,
        context_length: MAX_SEQ_LEN,
    };
    let tokenizer: Arc<dyn Tokenize> = Arc::new(IdentityByteTokenizer::default());
    EngineWorker::spawn(
        runtime_cfg,
        hyperparams,
        Box::new(backend),
        Arc::new(FlakyLayer {
            inner: provider,
            reads_left,
        }),
        tokenizer,
        model_info,
        8,
    )
}

fn job(prompt: &[u32]) -> JobRequest {
    JobRequest {
        prompt_tokens: prompt.to_vec(),
        max_tokens: 6,
        stop_text: Vec::new(),
        eos_token_ids: Vec::new(),
        ignore_eos: false,
        sampling: SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.0),
            seed: Some(42),
            ..SamplingParams::default()
        },
        suffix_threshold: 32,
        enable_thinking: false,
        reasoning_budget: 0,
        response_prefix: String::new(),
    }
}

fn never() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(usize::MAX))
}

fn lbc(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("model.lbc");
    std::fs::write(
        &path,
        build_qwen35_cpu_lbc(&Qwen35Fixture::default(), MAX_SEQ_LEN as u32),
    )
    .unwrap();
    path
}

/// What a job produced: its tokens, the prompt prefix its prefill reused (`None` when prefill
/// did not finish), and how it ended.
type Drained = (Vec<u32>, Option<usize>, Result<FinishReason, String>);

async fn drain(handle: &EngineHandle, req: JobRequest) -> Drained {
    let mut rx = handle.submit(req, 256).await.expect("submit");
    let (mut ids, mut reused) = (Vec::new(), None);
    loop {
        match tokio::time::timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(TokenEvent::PrefillDone {
                reused_prefix_len, ..
            })) => reused = Some(reused_prefix_len),
            Ok(Some(TokenEvent::Token { token_id, .. })) => ids.push(token_id),
            Ok(Some(TokenEvent::Done { finish_reason, .. })) => {
                return (ids, reused, Ok(finish_reason))
            }
            Ok(Some(TokenEvent::Error(e))) => return (ids, reused, Err(e)),
            Ok(Some(_)) => {}
            Ok(None) => panic!("channel closed without Done or Error"),
            Err(_) => panic!("timed out draining the job"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_request_after_a_failed_forward_pass_matches_a_fresh_worker() {
    let dir = tempfile::tempdir().unwrap();
    let path = lbc(&dir);
    let reads_left = never();
    let handle = boot(&path, Arc::clone(&reads_left));

    let first = [5u32, 9, 2, 7];
    let (gen, _, end) = drain(&handle, job(&first)).await;
    assert!(end.is_ok(), "{end:?}");
    // A warm append whose prefill fails at a later layer, after the earlier
    // GDN layers have advanced.
    let mut extended: Vec<u32> = first.iter().copied().chain(gen).collect();
    extended.extend_from_slice(&[11, 13]);
    reads_left.store(0, Ordering::SeqCst);
    let (_, _, end) = drain(&handle, job(&extended)).await;
    let err = end.expect_err("the failing prefill is answered with an error");
    assert!(err.contains("injected read failure"), "{err}");

    let (after, reused, end) = drain(&handle, job(&extended)).await;
    assert!(end.is_ok(), "{end:?}");
    assert_eq!(reused, Some(0), "the request after the failure starts cold");
    let fresh = boot(&path, never());
    let (want, _, end) = drain(&fresh, job(&extended)).await;
    assert!(end.is_ok(), "{end:?}");
    assert_eq!(after, want);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_conversation_after_a_failed_decode_step_matches_a_fresh_worker() {
    let dir = tempfile::tempdir().unwrap();
    let path = lbc(&dir);
    let reads_left = never();
    let handle = boot(&path, Arc::clone(&reads_left));

    let first = [5u32, 9, 2, 7];
    let (gen, _, end) = drain(&handle, job(&first)).await;
    assert!(end.is_ok(), "{end:?}");
    // A warm append whose prefill succeeds and whose decode fails part-way.
    let mut extended: Vec<u32> = first.iter().copied().chain(gen).collect();
    extended.extend_from_slice(&[11, 13]);
    reads_left.store(DECODE_FAILURE_READS, Ordering::SeqCst);
    let (partial, reused, end) = drain(&handle, job(&extended)).await;
    assert!(reused > Some(0), "the failing request is a warm append");
    let err = end.expect_err("the failing decode step is answered with an error");
    assert!(err.contains("injected read failure"), "{err}");
    assert!(
        !partial.is_empty(),
        "the failure came before any token was decoded"
    );

    // The client continues the conversation from the partial reply, a strict
    // extension of the failed request: a kept session would reuse it.
    let mut next: Vec<u32> = extended.iter().copied().chain(partial).collect();
    next.extend_from_slice(&[17, 19]);
    let (after, reused, end) = drain(&handle, job(&next)).await;
    assert!(end.is_ok(), "{end:?}");
    assert_eq!(reused, Some(0), "the request after the failure starts cold");
    let fresh = boot(&path, never());
    let (want, _, end) = drain(&fresh, job(&next)).await;
    assert!(end.is_ok(), "{end:?}");
    assert_eq!(after, want);
}
