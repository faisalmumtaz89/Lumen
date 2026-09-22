//! The exclusive device lease: a lease evicts the text backend, requests are
//! refused while it is held, and dropping it restores service.
//!
//! These are protocol tests, not device tests. The backend here is the naive
//! CPU one, so "evicted" means the engine dropped its `Box<dyn ComputeBackend>`
//! and rebuilt it from the factory — the same code path a CUDA backend takes,
//! with the device-memory effects a CPU backend does not have. What is under
//! test is the ordering (evict before use, restore before release), the
//! refusal while leased, and that a generation's `Drop` always restores.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, Logits};
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCacheView, KvPrecision};
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::cache::LayerView;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::RuntimeConfig;

use lumen_server::{
    BackendFactory, EngineWorker, IdentityByteTokenizer, JobRequest, ModelInfo, Tokenize,
};

const MODEL_ID: &str = "lumen-test:lease";

/// The naive CPU backend with its drop counted, so a test can see that an
/// eviction freed the backend rather than only flagged it. Only the methods
/// the naive backend implements itself are forwarded; the rest are the
/// trait's defaults on both.
struct CountedBackend {
    inner: NaiveF32Backend,
    dropped: Arc<AtomicUsize>,
}

impl Drop for CountedBackend {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::AcqRel);
    }
}

impl ComputeBackend for CountedBackend {
    fn init(&mut self, hyperparams: &lumen_format::ModelHyperparams) -> Result<(), RuntimeError> {
        self.inner.init(hyperparams)
    }
    fn compute_layer(
        &self,
        layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        self.inner.compute_layer(layer_idx, x, weights, kv, seq_pos)
    }
    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        self.inner.compute_final(x)
    }
    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        self.inner.embed_token(token_id)
    }
    fn set_global_tensors(
        &mut self,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        output_proj: Vec<f32>,
    ) {
        self.inner
            .set_global_tensors(embedding, final_norm, output_proj)
    }
}

/// Builds a naive CPU backend. A restored backend is indistinguishable from
/// the first one, which is what makes it a usable stand-in for CUDA here:
/// the lease protocol cannot tell them apart, and neither can a wrong
/// implementation hide behind the difference.
struct TestFactory {
    globals: (Vec<f32>, Vec<f32>, Vec<f32>),
    hyperparams: lumen_format::ModelHyperparams,
    /// While set, `build` fails, standing in for a device allocation that
    /// does not come back.
    fail: AtomicBool,
    /// Backends dropped so far, the first one included.
    dropped: Arc<AtomicUsize>,
    /// Backends this factory built and initialised.
    built: AtomicUsize,
}

impl BackendFactory for TestFactory {
    fn device(&self) -> usize {
        0
    }

    fn build(&self) -> Result<Box<dyn ComputeBackend>, String> {
        if self.fail.load(Ordering::Acquire) {
            return Err("the test factory was told to fail".to_string());
        }
        let mut b = CountedBackend {
            inner: NaiveF32Backend::new(),
            dropped: Arc::clone(&self.dropped),
        };
        b.set_global_tensors(
            self.globals.0.clone(),
            self.globals.1.clone(),
            self.globals.2.clone(),
        );
        b.init(&self.hyperparams)
            .map_err(|e| format!("test backend init: {e}"))?;
        self.built.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(b))
    }
}

/// A worker over a synthetic model, with a rebuildable backend. The factory
/// is returned so a test can make its next build fail.
fn spawn_worker() -> (
    lumen_server::EngineHandle,
    Arc<TestFactory>,
    tempfile::TempDir,
) {
    let (provider, tmp) = test_model();
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut backend = CountedBackend {
        inner: NaiveF32Backend::new(),
        dropped: Arc::clone(&dropped),
    };
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();
    let hyperparams = provider.lbc().header.hyperparams;
    let tokenizer: Arc<dyn Tokenize> = Arc::new(IdentityByteTokenizer::default());
    let factory = Arc::new(TestFactory {
        globals: (
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        ),
        hyperparams,
        fail: AtomicBool::new(false),
        dropped,
        built: AtomicUsize::new(0),
    });
    let handle = EngineWorker::spawn_rebuildable(
        runtime_cfg(),
        hyperparams,
        Box::new(backend),
        Arc::clone(&factory) as Arc<dyn BackendFactory>,
        Arc::new(provider),
        tokenizer,
        model_info(),
        4,
    );
    (handle, factory, tmp)
}

/// The synthetic 96-position model, written to a temp dir and opened.
fn test_model() -> (SyncWeightProvider, tempfile::TempDir) {
    let cfg = TestModelConfig {
        max_seq_len: 96,
        ..TestModelConfig::default()
    };
    let bytes = generate_test_model(&cfg);
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = tmp.path().join("test_model.lbc");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
    (SyncWeightProvider::open(&path).unwrap(), tmp)
}

fn runtime_cfg() -> RuntimeConfig {
    RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len: 96,
        collect_per_layer_timings: false,
    }
}

fn model_info() -> ModelInfo {
    ModelInfo {
        id: MODEL_ID.into(),
        owned_by: "lumen-test".into(),
        created: 0,
        context_length: 96,
    }
}

/// The synthetic model's vocab is 32, so every id must be inside it.
fn prompt_tokens() -> Vec<u32> {
    vec![3, 7, 11]
}

fn job() -> JobRequest {
    use lumen_runtime::engine::SamplingParams;
    JobRequest {
        prompt_tokens: prompt_tokens(),
        max_tokens: 2,
        stop_text: Vec::new(),
        eos_token_ids: Vec::new(),
        ignore_eos: false,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(1),
            ..SamplingParams::default()
        },
        suffix_threshold: 32,
        enable_thinking: false,
        reasoning_budget: 0,
    }
}

/// Drain a job to completion. Returns `Ok` if it produced output, `Err` with
/// the server's message if it was refused.
async fn run_job(handle: &lumen_server::EngineHandle) -> Result<(), String> {
    let mut rx = handle
        .submit(job(), 8)
        .await
        .map_err(|e| format!("{e:?}"))?;
    let mut saw_text = false;
    while let Some(ev) = rx.recv().await {
        match ev {
            lumen_server::TokenEvent::Token { .. } => saw_text = true,
            lumen_server::TokenEvent::Error(m) => return Err(m),
            lumen_server::TokenEvent::Done { .. } => break,
            _ => {}
        }
    }
    if saw_text {
        Ok(())
    } else {
        Err("no tokens".to_string())
    }
}

/// A lease evicts the backend; text requests are refused while it is held and
/// served again once it drops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lease_refuses_text_and_restores_it() {
    let (handle, factory, _tmp) = spawn_worker();
    // Serving to begin with.
    run_job(&handle).await.expect("the first job is served");

    let h = handle.clone();
    let lease = tokio::task::spawn_blocking(move || h.try_exclusive())
        .await
        .unwrap()
        .expect("a rebuildable worker grants a lease");
    // The backend is gone by the time the lease is granted: the guard's
    // device memory is free for the holder, not merely marked for later.
    assert_eq!(
        factory.dropped.load(Ordering::Acquire),
        1,
        "the lease is granted only after the backend was dropped"
    );
    // Refused at submission, with the retryable shape and the evicted
    // message — not admitted and then failed by the worker.
    match handle.submit(job(), 8).await {
        Err(lumen_server::ServerError::EngineUnavailable(m))
            if m == lumen_server::EVICTED_MESSAGE => {}
        other => panic!(
            "a job submitted while leased must be refused as EngineUnavailable(EVICTED_MESSAGE), \
             got {:?}",
            other.map(|_| ())
        ),
    }

    drop(lease);
    // The restore completed inside `drop`: the factory has built the
    // replacement before any job could be submitted.
    assert_eq!(
        factory.built.load(Ordering::Acquire),
        1,
        "the backend is rebuilt before the lease's drop returns"
    );
    run_job(&handle)
        .await
        .expect("the engine is served again after the lease drops");
    assert_eq!(factory.dropped.load(Ordering::Acquire), 1);
}

/// `try_exclusive` refuses on a worker that cannot rebuild, so a
/// plain `spawn` deployment never loses its model to a lease it cannot undo.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_without_a_factory_refuses_to_lease() {
    let (provider, _tmp) = test_model();
    let mut backend = NaiveF32Backend::new();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&provider.lbc().header.hyperparams).unwrap();
    let hyperparams = provider.lbc().header.hyperparams;
    let handle = EngineWorker::spawn(
        runtime_cfg(),
        hyperparams,
        Box::new(backend),
        Arc::new(provider),
        Arc::new(IdentityByteTokenizer::default()) as Arc<dyn Tokenize>,
        model_info(),
        4,
    );
    let h1 = handle.clone();
    let refused = tokio::task::spawn_blocking(move || h1.try_exclusive())
        .await
        .unwrap();
    assert!(
        refused.is_err(),
        "a non-rebuildable worker must not hand out a lease"
    );
    drop(refused);
    run_job(&handle).await.expect("service is unaffected");
}

/// The retry a text request runs after a failed restore blocks nothing and
/// needs no spare blocking thread, so an embedder on a current-thread
/// runtime whose one blocking thread the engine already holds still gets the
/// refusal and, once the backend can be built again, the recovery.
#[test]
fn a_failed_restore_is_retried_on_a_current_thread_runtime() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    rt.block_on(async {
        // The engine worker is the runtime's one blocking thread from here on.
        let (handle, factory, _tmp) = spawn_worker();
        run_job(&handle).await.expect("served before the lease");
        // The lease itself must block (it waits for the eviction), so it runs
        // on a plain thread rather than the runtime's blocking pool.
        let h = handle.clone();
        let lease = std::thread::spawn(move || h.try_exclusive())
            .join()
            .unwrap()
            .expect("lease");
        factory.fail.store(true, Ordering::Release);
        std::thread::spawn(move || drop(lease)).join().unwrap();
        let refused = tokio::time::timeout(Duration::from_secs(5), handle.submit(job(), 8))
            .await
            .expect("the retry needs no spare blocking thread");
        match refused {
            Err(lumen_server::ServerError::EngineUnavailable(m))
                if m.starts_with(lumen_server::RESTORE_FAILED_PREFIX) => {}
            other => panic!("refused at admission, got {:?}", other.map(|_| ())),
        }
        factory.fail.store(false, Ordering::Release);
        run_job(&handle)
            .await
            .expect("recovered on a current-thread runtime");
    });
}

/// A held lease admits only one generation: a second acquisition waits for
/// the first to drop rather than racing it for the device.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_lease_waits_for_the_first() {
    let (handle, factory, _tmp) = spawn_worker();
    let h0 = handle.clone();
    let first = tokio::task::spawn_blocking(move || h0.try_exclusive())
        .await
        .unwrap()
        .expect("first lease");

    // The second acquisition must block while the first is held. Run it on a
    // thread and confirm it does not complete, then release and confirm it
    // does.
    let h = handle.clone();
    let asking = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&asking);
    let second = tokio::task::spawn_blocking(move || {
        flag.store(true, Ordering::Release);
        let g = h.try_exclusive();
        let acquired = g.is_ok();
        drop(g);
        acquired
    });
    while !asking.load(Ordering::Acquire) {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !second.is_finished(),
        "a second lease must not be granted while the first is held"
    );
    assert_eq!(factory.dropped.load(Ordering::Acquire), 1);
    drop(first);
    let acquired = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .expect("the second lease is granted once the first drops")
        .unwrap();
    assert!(acquired);
    // The first lease's restore ran before the second's eviction: a second
    // backend was built and dropped, which a lease granted alongside the
    // first (its eviction finding no backend) would not have done.
    assert_eq!(
        factory.dropped.load(Ordering::Acquire),
        2,
        "the second lease evicted the backend the first lease restored"
    );
    run_job(&handle).await.expect("service is restored");
}

/// A restore that fails leaves the worker evicted but alive: requests are
/// answered with an error rather than run on a missing backend, and the next
/// request's own retry brings service back once the backend can be built.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_restore_keeps_the_worker_answering() {
    let (handle, factory, _tmp) = spawn_worker();
    run_job(&handle).await.expect("served before the lease");

    let h = handle.clone();
    let lease = tokio::task::spawn_blocking(move || h.try_exclusive())
        .await
        .unwrap()
        .expect("lease");
    factory.fail.store(true, Ordering::Release);
    tokio::task::spawn_blocking(move || drop(lease))
        .await
        .unwrap();
    // Evicted for good: refused at admission with the retryable shape — the
    // rebuild is retried for the request and its failure reported before any
    // stream could start — not a hang and not a dead worker.
    let refused = tokio::time::timeout(Duration::from_secs(5), handle.submit(job(), 8))
        .await
        .expect("an evicted worker answers rather than hangs");
    match refused {
        Err(lumen_server::ServerError::EngineUnavailable(m))
            if m.starts_with(lumen_server::RESTORE_FAILED_PREFIX) && m.contains("retries it") => {}
        other => panic!(
            "a permanently evicted worker refuses at admission, got {:?}",
            other.map(|_| ())
        ),
    }

    // A lease granted meanwhile finds nothing to evict and its restore fails
    // again; the worker keeps answering.
    let h = handle.clone();
    let lease = tokio::task::spawn_blocking(move || h.try_exclusive())
        .await
        .unwrap()
        .expect("a lease is granted while evicted");
    tokio::task::spawn_blocking(move || drop(lease))
        .await
        .unwrap();
    match handle.submit(job(), 8).await {
        Err(lumen_server::ServerError::EngineUnavailable(m))
            if m.starts_with(lumen_server::RESTORE_FAILED_PREFIX) => {}
        other => panic!(
            "still evicted after the second failed restore, got {:?}",
            other.map(|_| ())
        ),
    }

    factory.fail.store(false, Ordering::Release);
    run_job(&handle)
        .await
        .expect("the next request's retry brings service back");
    assert_eq!(factory.dropped.load(Ordering::Acquire), 1);
}

/// A lease requested while a job is running is granted only after that job
/// has finished on the live backend: the job completes normally, and the
/// eviction follows it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lease_during_a_job_lets_the_job_finish() {
    let (handle, _factory, _tmp) = spawn_worker();
    // More tokens than the channel holds, so the job is still running — blocked
    // on its channel — until this test reads.
    let mut long = job();
    long.max_tokens = 64;
    let mut rx = handle.submit(long, 8).await.expect("the job is admitted");
    // The lease is requested only once the job is provably running: the first
    // token is read before asking. The worker then parks on the full channel
    // (16 events of 64 — the pool channel's fixed capacity; the 8 passed to
    // `submit` is advisory) for as long as nothing else is read, so the pause
    // below lets the request reach the inbox while the job is still inside
    // the backend, whatever the scheduling of the blocking thread.
    let mut tokens = 0usize;
    loop {
        match rx.recv().await.expect("the stream is open") {
            lumen_server::TokenEvent::Token { .. } => {
                tokens += 1;
                break;
            }
            lumen_server::TokenEvent::PrefillDone { .. } => {}
            other => panic!("the job must reach its first token, got {other:?}"),
        }
    }
    let h = handle.clone();
    let lease = tokio::task::spawn_blocking(move || h.try_exclusive());
    // Confirms the request entered the lease protocol while
    // the job is parked: admission closes before the eviction is queued
    // behind the job.
    tokio::time::timeout(Duration::from_secs(5), async {
        while !handle.is_leased() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the lease request closes admission while the job is parked");
    assert!(
        !lease.is_finished(),
        "the lease must not be granted while the job is parked mid-stream"
    );
    // After `tokens` reads the worker has produced at most `tokens + 16` and
    // is still inside the job while `tokens + 16 < 64`; the assertion is
    // confined to that range, where a granted lease could only mean an
    // eviction under the running job.
    while let Some(ev) = rx.recv().await {
        match ev {
            lumen_server::TokenEvent::Token { .. } => {
                tokens += 1;
                if tokens <= 32 {
                    assert!(
                        !lease.is_finished(),
                        "the lease must not be granted while the job is still running"
                    );
                }
            }
            lumen_server::TokenEvent::Error(m) => panic!("the running job must not fail: {m}"),
            lumen_server::TokenEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert_eq!(
        tokens, 64,
        "the job admitted before the lease ran to completion"
    );
    let lease = lease
        .await
        .unwrap()
        .expect("the lease is granted once the job is done");
    tokio::task::spawn_blocking(move || drop(lease))
        .await
        .unwrap();
    run_job(&handle).await.expect("service is restored");
}
