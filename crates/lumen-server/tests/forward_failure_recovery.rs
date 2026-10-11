//! A forward pass that fails part-way leaves KV and recurrent state partly advanced; the worker
//! must not continue from it. On a GDN model served by the CPU backend, a request whose prefill
//! fails at a later layer is answered with the error, and the next request then produces what a
//! fresh worker produces.

#[path = "../../lumen-runtime/tests/common/gdn_hybrid.rs"]
#[allow(dead_code)]
mod gdn_hybrid;

use std::sync::atomic::{AtomicBool, Ordering};
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

/// A provider whose `FAILING_LAYER` cannot be read while `fail` is set.
struct FlakyLayer {
    inner: SyncWeightProvider,
    fail: Arc<AtomicBool>,
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
        if layer == FAILING_LAYER && self.fail.load(Ordering::SeqCst) {
            return Err(RuntimeError::LayerUnavailable {
                layer,
                reason: "injected read failure".into(),
            });
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

fn boot(path: &std::path::Path, fail: Arc<AtomicBool>) -> EngineHandle {
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
            fail,
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

async fn drain(handle: &EngineHandle, req: JobRequest) -> (Vec<u32>, Result<FinishReason, String>) {
    let mut rx = handle.submit(req, 256).await.expect("submit");
    let mut ids = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(TokenEvent::Token { token_id, .. })) => ids.push(token_id),
            Ok(Some(TokenEvent::Done { finish_reason, .. })) => return (ids, Ok(finish_reason)),
            Ok(Some(TokenEvent::Error(e))) => return (ids, Err(e)),
            Ok(Some(_)) => {}
            Ok(None) => panic!("channel closed without Done or Error"),
            Err(_) => panic!("timed out draining the job"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_request_after_a_failed_forward_pass_matches_a_fresh_worker() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.lbc");
    std::fs::write(
        &path,
        build_qwen35_cpu_lbc(&Qwen35Fixture::default(), MAX_SEQ_LEN as u32),
    )
    .unwrap();
    let fail = Arc::new(AtomicBool::new(false));
    let handle = boot(&path, Arc::clone(&fail));

    let first = [5u32, 9, 2, 7];
    let (gen, end) = drain(&handle, job(&first)).await;
    assert!(end.is_ok(), "{end:?}");
    // A warm append whose prefill fails at a later layer, after the earlier
    // GDN layers have advanced.
    let mut extended: Vec<u32> = first.iter().copied().chain(gen).collect();
    extended.extend_from_slice(&[11, 13]);
    fail.store(true, Ordering::SeqCst);
    let (_, end) = drain(&handle, job(&extended)).await;
    let err = end.expect_err("the failing prefill is answered with an error");
    assert!(err.contains("injected read failure"), "{err}");
    fail.store(false, Ordering::SeqCst);

    let (after, end) = drain(&handle, job(&extended)).await;
    assert!(end.is_ok(), "{end:?}");
    let fresh = boot(&path, Arc::new(AtomicBool::new(false)));
    let (want, end) = drain(&fresh, job(&extended)).await;
    assert!(end.is_ok(), "{end:?}");
    assert_eq!(after, want);
}
