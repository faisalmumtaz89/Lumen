//! A worker on the CPU-naive backend with a synthetic model, for the
//! fault-injection tests (each test binary sets its environment before the
//! first job, and the module reads it once).

use std::sync::Arc;
use std::time::Duration;

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::RuntimeConfig;
use lumen_server::{
    EngineHandle, EngineWorker, FinishReason, IdentityByteTokenizer, JobRequest, ModelInfo,
    TokenEvent, Tokenize,
};

const MAX_SEQ_LEN: usize = 256;

pub fn boot_engine() -> EngineHandle {
    let cfg = TestModelConfig {
        vocab_size: 256,
        max_seq_len: MAX_SEQ_LEN as u32,
        ..TestModelConfig::default()
    };
    let bytes = generate_test_model(&cfg);
    let tmp = tempfile::tempdir().expect("temp dir");
    let path = tmp.path().join("test_model.lbc");
    std::fs::write(&path, &bytes).unwrap();
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
        id: "lumen-test:fault".into(),
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

pub fn job(max_tokens: usize) -> JobRequest {
    JobRequest {
        prompt_tokens: vec![104, 105],
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
        enable_thinking: false,
        reasoning_budget: 0,
    }
}

/// The token ids emitted, then how the job ended: `Ok(finish)` or the error.
pub async fn drain(
    handle: &EngineHandle,
    req: JobRequest,
) -> (Vec<u32>, Result<FinishReason, String>) {
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
