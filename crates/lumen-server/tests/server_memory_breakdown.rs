//! `/debug/memory_breakdown` endpoint contract tests.
//!
//! These tests verify the server-side instrumentation that the
//! soak harness uses to attribute long-session RSS growth to one of the
//! seven hypothesis classes named in.
//!
//! Test matrix (5 tests):
//!
//! 1. `default_path_returns_404` — without `LUMEN_SERVER_DEBUG_MEM`,
//!    the endpoint is invisible (404).  This is the byte-identical-default
//!    invariant.
//! 2. `enabled_returns_zero_snapshot_before_jobs` — with the env var set,
//!    a server that has not yet served a request returns the all-zero
//!    snapshot (except for `engine_inbox_capacity`).
//! 3. `enabled_zero_disables_endpoint` — `LUMEN_SERVER_DEBUG_MEM=0`
//!    behaves like unset (404).
//! 4. `snapshot_updates_after_request` — after one completed chat
//!    request the snapshot reflects the new `kv_seq_len`,
//!    `session_tokens_len`, `update_count >= 1`, and a non-zero
//!    `last_update_unix` epoch timestamp.
//! 5. `snapshot_handle_clone_shares_arc` — Clones of `EngineHandle` see
//!    the same snapshot (Arc-shared) and do NOT double-allocate the
//!    breakdown buffer.
//!
//! Background workload uses the synthetic test model + CPU naive backend
//! (matches `server_integration.rs` pattern) so the test stays
//! Metal-independent and runs on every CI platform.

// parity: keep the breakdown-test binary on the same allocator
// as the soak-test binary; the breakdown counters track per-component
// state, so any mimalloc-induced shift in those counters surfaces here.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::{Request, Uri};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;
use serde_json::Value;

use lumen_format::test_model::{generate_test_model, TestModelConfig};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::RuntimeConfig;

use lumen_server::{
    build_router, EngineHandle, EngineWorker, IdentityByteTokenizer, ModelInfo, Tokenize,
};

const MAX_TOKENS: usize = 4;
const MODEL_ID: &str = "lumen-test:memory-breakdown";
const INBOX_SIZE: usize = 4;
const ENV_KEY: &str = "LUMEN_SERVER_DEBUG_MEM";

/// Process-wide mutex serialising env var manipulation across the tests
/// in this binary.  Each `#[tokio::test]` in this file mutates the
/// `LUMEN_SERVER_DEBUG_MEM` env var, so they MUST run serially or one
/// test's `remove_var` will race against another's `set_var`.
///
/// We pay the cost of serial execution because the methodology brief
/// names the env var verbatim — switching to a router-builder argument
/// would fork that contract.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that acquires `ENV_LOCK`, sets `LUMEN_SERVER_DEBUG_MEM` to
/// the supplied value (or removes it when `None`), and restores the
/// previous state on Drop.  Built on `Mutex` rather than `parking_lot`
/// so this test file has no extra dev-dependency.
struct EnvGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl EnvGuard {
    /// Acquire the lock, capture the current `LUMEN_SERVER_DEBUG_MEM`
    /// value, set the new value (or remove on `None`).
    fn set(value: Option<&str>) -> Self {
        // If a prior test panicked while holding the lock, the mutex is
        // poisoned.  We don't care about the poisoning semantics for env
        // var serialisation — we just want exclusive access — so the
        // poisoned guard is unwrapped.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var(ENV_KEY).ok();
        match value {
            Some(v) => unsafe { std::env::set_var(ENV_KEY, v) },
            None => unsafe { std::env::remove_var(ENV_KEY) },
        }
        Self { _guard, prior }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prior.take() {
            Some(v) => unsafe { std::env::set_var(ENV_KEY, v) },
            None => unsafe { std::env::remove_var(ENV_KEY) },
        }
    }
}

/// Boot the server with the synthetic test model and return
/// (address, hyper client, engine handle, tempdir keeper).
///
/// The caller is responsible for managing the env var lifetime — these
/// tests deliberately set and unset `LUMEN_SERVER_DEBUG_MEM` around the
/// request to verify gating, and the env var lookup is performed
/// per-handler-invocation so no caching wraps the test.
async fn boot() -> (
    SocketAddr,
    Client<HttpConnector, Full<bytes::Bytes>>,
    EngineHandle,
    tempfile::TempDir,
) {
    let cfg = TestModelConfig {
        vocab_size: 256,
        // bumped from default 64 to 96 to absorb the 19-byte
        // Qwen3.5 `enable_thinking=false` empty-think tail appended by
        // `render_chat_prompt`. Must match runtime_cfg.max_seq_len.
        max_seq_len: 96,
        ..TestModelConfig::default()
    };
    let bytes = generate_test_model(&cfg);
    let tmp = tempfile::tempdir().expect("create temp dir");
    let path = tmp.path().join("test_model.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
    }
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
        max_seq_len: 96,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: MODEL_ID.into(),
        owned_by: "lumen-test".into(),
        created: 0,
        context_length: 96,
    };
    let tokenizer: Arc<dyn Tokenize> = Arc::new(IdentityByteTokenizer::default());
    let handle = EngineWorker::spawn(
        runtime_cfg,
        hyperparams,
        Box::new(backend),
        Arc::new(provider),
        tokenizer,
        model_info,
        INBOX_SIZE,
    );

    let handle_for_app = handle.clone();
    let app = build_router(handle_for_app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Give axum a moment to start listening.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let client: Client<HttpConnector, Full<bytes::Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();
    (addr, client, handle, tmp)
}

/// GET a URL and return (status, body).  Distinct from
/// `server_integration::get_json` because that helper asserts success;
/// we explicitly need to observe the 404 path.
async fn get_status_and_body(
    client: &Client<HttpConnector, Full<bytes::Bytes>>,
    uri: Uri,
) -> (hyper::StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(bytes::Bytes::new()))
        .unwrap();
    let resp = client.request(req).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

/// Send one chat-completion to populate session state, ignore the body.
async fn one_chat(
    client: &Client<HttpConnector, Full<bytes::Bytes>>,
    addr: SocketAddr,
) {
    let uri: Uri = format!("http://{addr}/v1/chat/completions").parse().unwrap();
    let body = serde_json::json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 42,
        "stream": false,
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(bytes::Bytes::from(body_bytes)))
        .unwrap();
    let resp = client.request(req).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(
        status.is_success(),
        "chat completion failed: {} body={}",
        status,
        String::from_utf8_lossy(&body),
    );
}

// =====================================================================
// Test 1 — default path returns 404
// =====================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn default_path_returns_404() {
    let _env = EnvGuard::set(None);
    let (addr, client, _h, _tmp) = boot().await;
    let uri: Uri = format!("http://{addr}/debug/memory_breakdown").parse().unwrap();
    let (status, body) = get_status_and_body(&client, uri).await;
    assert_eq!(
        status,
        hyper::StatusCode::NOT_FOUND,
        "default path must 404; got {} body={}",
        status,
        String::from_utf8_lossy(&body),
    );
}

// =====================================================================
// Test 2 — enabled, zero snapshot before first job
// =====================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enabled_returns_zero_snapshot_before_jobs() {
    let _env = EnvGuard::set(Some("1"));
    let (addr, client, _h, _tmp) = boot().await;
    let uri: Uri = format!("http://{addr}/debug/memory_breakdown").parse().unwrap();
    let (status, body) = get_status_and_body(&client, uri).await;
    assert_eq!(status, hyper::StatusCode::OK, "enabled must return 200");
    let v: Value = serde_json::from_slice(&body).expect("body is JSON");

    // engine_inbox_capacity is set at spawn time to INBOX_SIZE; everything
    // else stays zero until process_job runs at least once.
    assert_eq!(v["engine_inbox_capacity"].as_u64().unwrap(), INBOX_SIZE as u64);
    assert_eq!(v["update_count"].as_u64().unwrap(), 0);
    assert_eq!(v["last_update_unix"].as_u64().unwrap(), 0);
    assert_eq!(v["kv_used_bytes"].as_u64().unwrap(), 0);
    assert_eq!(v["kv_seq_len"].as_u64().unwrap(), 0);
    assert_eq!(v["session_tokens_len"].as_u64().unwrap(), 0);
}

// =====================================================================
// Test 3 — env=0 disables endpoint (treated same as unset)
// =====================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enabled_zero_disables_endpoint() {
    let _env = EnvGuard::set(Some("0"));
    let (addr, client, _h, _tmp) = boot().await;
    let uri: Uri = format!("http://{addr}/debug/memory_breakdown").parse().unwrap();
    let (status, _body) = get_status_and_body(&client, uri).await;
    assert_eq!(
        status,
        hyper::StatusCode::NOT_FOUND,
        "LUMEN_SERVER_DEBUG_MEM=0 must 404",
    );
}

// =====================================================================
// Test 4 — snapshot updates after a request
// =====================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_updates_after_request() {
    let _env = EnvGuard::set(Some("1"));
    let (addr, client, _h, _tmp) = boot().await;

    // Drive one request through.
    one_chat(&client, addr).await;

    // Then sample the breakdown.
    let uri: Uri = format!("http://{addr}/debug/memory_breakdown").parse().unwrap();
    let (status, body) = get_status_and_body(&client, uri).await;
    assert_eq!(status, hyper::StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).expect("body is JSON");

    let update_count = v["update_count"].as_u64().unwrap();
    let last_update = v["last_update_unix"].as_u64().unwrap();
    let kv_seq_len = v["kv_seq_len"].as_u64().unwrap();
    let session_tokens_len = v["session_tokens_len"].as_u64().unwrap();
    let pending_logits_bytes = v["session_pending_logits_bytes"].as_u64().unwrap();

    assert!(update_count >= 1, "update_count must be >= 1 after a job, got {update_count}: full body={}", String::from_utf8_lossy(&body));
    assert!(last_update > 0, "last_update_unix must be a real epoch ts, got {last_update}");
    // Identity byte tokenizer + "x" prompt + chat template adds a handful of
    // tokens; whatever the exact count, it must be > 0 after a job.
    assert!(session_tokens_len > 0, "session_tokens_len must be > 0 after a job");
    // KV seq_len matches the token count for the simple identity tokenizer
    // path (no chat template).
    assert!(kv_seq_len > 0, "kv_seq_len must be > 0 after a job");
    // pending_logits is vocab*4 by the conservative steady-state estimate;
    // the synthetic test model has vocab_size=256 so the value is exactly 1024.
    assert_eq!(
        pending_logits_bytes, 256 * 4,
        "session_pending_logits_bytes must equal vocab*4, got {pending_logits_bytes}",
    );
}

// =====================================================================
// Test 5 — handle clones share the snapshot Arc
// =====================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_handle_clone_shares_arc() {
    let (_addr, _client, handle, _tmp) = boot().await;
    let clone = handle.clone();
    // Both handles must see the same default-zero snapshot value
    // (modulo engine_inbox_capacity, which is constant).
    let a = handle.memory_breakdown_snapshot();
    let b = clone.memory_breakdown_snapshot();
    assert_eq!(a, b, "cloned EngineHandle must share breakdown snapshot");
    // And the underlying Arc must be the SAME Arc (so the post-job
    // refresh on the worker thread is visible to both handles).
    assert!(
        Arc::ptr_eq(&handle.breakdown_arc(), &clone.breakdown_arc()),
        "breakdown Arc must be shared between handle clones, not duplicated",
    );
}
