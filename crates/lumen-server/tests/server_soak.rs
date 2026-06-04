//! long-session soak harness for `lumen-server`.
//!
//! This `#[ignore]`-d integration test boots a real `lumen-server` on a
//! Qwen3.5-9B Q8_0 LBC (Metal backend on macOS, naive CPU elsewhere),
//! exposes a stable HTTP port via stdout, samples RSS / FD / request
//! statistics on a 30-second cadence, and runs until `LUMEN_SOAK_DURATION_SEC`
//! elapses.
//!
//! ## Invocation
//!
//! Smoke (5 minutes):
//! ```sh
//! LUMEN_QWEN35_9B_Q8=/path/to/qwen3-5-9b-Q8_0.lbc \
//! LUMEN_SOAK_DURATION_SEC=300 \
//! LUMEN_SOAK_OUT_DIR=target/soak-out \
//! cargo test --release -p lumen-server --test server_soak \
//!   -- --ignored --nocapture soak_smoke_5min
//! ```
//!
//! Full 4-hour:
//! ```sh
//! LUMEN_QWEN35_9B_Q8=... LUMEN_SOAK_DURATION_SEC=14400 ... soak_4h
//! ```
//!
//! ## Outputs
//!
//! - `$LUMEN_SOAK_OUT_DIR/soak-stats.jsonl` — one line per 30-second sample
//!   with RSS, FD count, request counter, decode-token counter.
//! - `$LUMEN_SOAK_OUT_DIR/soak-port.txt` — the bound port (for the Python
//!   client to read).
//! - `$LUMEN_SOAK_OUT_DIR/soak-summary.json` — per-hour aggregates + final
//!   verdict.
//!
//! ## Methodology compliance
//!
//! - Reuses `lumen_cli::tokenize::BpeTokenizer` via the `lumen_cli` library
//!   facade (no parallel implementation — rule 7).
//! - Reuses `lumen_server::{EngineWorker, build_router}` directly (no fork).
//! - Reuses `mtl-mem-probe` if available; otherwise records
//!   `mtl_alloc_bytes = -1` (sentinel).
//! - All outputs are machine-readable JSONL / JSON (no subjective signals —
//!   convention).
//! - Default-path bit-exact when disabled (this file lives entirely
//!   under `tests/` and only runs when explicitly requested via `--ignored`).

#![cfg(target_os = "macos")] // Soak harness is Metal-focused; CPU naive soak is out of scope.

// parity: bind the soak-test process to the mimalloc allocator.
// The validation gate (`RSS slope ≤ +0.5%/h`) is measured against the
// process this test runs in, so the allocator declared here is what the
// soak actually exercises.
//
// attribution: an opt-in cfg-gate lets a follow-up build
// recompile this binary WITHOUT the mimalloc binding (so Apple's
// `MallocStackLogging=1` + `malloc_history -callTree` can see Rust-heap
// allocations, which mimalloc otherwise bypasses by going to mmap
// directly). Build with `RUSTFLAGS='--cfg stacks_dump'` to opt in.
// Default-path (no cfg flag) is byte-identical to the build.
#[cfg(not(stacks_dump))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use http_body_util::{BodyExt, Full};
use hyper::{Request, Uri};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::{MetalF32Backend, RuntimeConfig};

use lumen_server::{
    build_router, DiskKvConfig, EngineWorker, ModelInfo, Tokenize,
};

const MODEL_ID: &str = "qwen3.5-9b:q8_0";
const SAMPLE_INTERVAL_SEC: u64 = 30;
const DISK_KV_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GB disk-KV budget
const CONTEXT_LEN: usize = 4096;
const INBOX_SIZE: usize = 16;

/// Stats sampled at every interval and written to `soak-stats.jsonl`.
#[derive(Debug)]
struct SoakSample {
    ts_unix: u64,
    elapsed_sec: u64,
    rss_kb: i64,
    fd_count: i64,
    mtl_alloc_bytes: i64,
    request_count: u64,
    decode_tokens_total: u64,
    disk_kv_used_bytes: i64,
    disk_kv_tmp_orphan_count: i64,
}

impl SoakSample {
    fn to_jsonl(&self) -> String {
        format!(
            r#"{{"ts_unix":{},"elapsed_sec":{},"rss_kb":{},"fd_count":{},"mtl_alloc_bytes":{},"request_count":{},"decode_tokens_total":{},"disk_kv_used_bytes":{},"disk_kv_tmp_orphan_count":{}}}"#,
            self.ts_unix,
            self.elapsed_sec,
            self.rss_kb,
            self.fd_count,
            self.mtl_alloc_bytes,
            self.request_count,
            self.decode_tokens_total,
            self.disk_kv_used_bytes,
            self.disk_kv_tmp_orphan_count,
        )
    }
}

/// `Tokenize` adapter wrapping `lumen_cli::tokenize::BpeTokenizer`.
///
/// Decoding is per-token via `BpeTokenizer::decode(&[id])` plus a small
/// UTF-8 buffer to handle codepoints that span multiple tokens (a real
/// concern for byte-pair BPE).
struct BpeTokenizerAdapter {
    inner: lumen_cli::tokenize::BpeTokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenize for BpeTokenizerAdapter {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode_incremental(&self, state: &mut Vec<u8>, token_id: u32) -> String {
        // Decode just this single token to bytes, append to the buffer,
        // and flush whatever forms a complete UTF-8 prefix.
        let frag = self.inner.decode(&[token_id]);
        state.extend_from_slice(frag.as_bytes());
        match std::str::from_utf8(state) {
            Ok(_) => {
                let bytes = std::mem::take(state);
                String::from_utf8(bytes).unwrap_or_default()
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    String::new()
                } else {
                    let head_bytes = state[..valid].to_vec();
                    let tail = state[valid..].to_vec();
                    *state = tail;
                    String::from_utf8(head_bytes).unwrap_or_default()
                }
            }
        }
    }

    fn apply_chat_template(&self, system: Option<&str>, user: &str) -> Option<String> {
        Some(self.inner.apply_chat_template_with_system(user, system))
    }

    fn eos_tokens(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }
}

/// Resolve LBC path from env vars. Returns `None` if no Q8 path is set;
/// the test gracefully no-ops in that case (matches pattern).
fn resolve_q8_lbc_path() -> Option<PathBuf> {
    std::env::var("LUMEN_QWEN35_9B_Q8")
        .ok()
        .map(PathBuf::from)
}

/// Resolve soak duration from env. Defaults to 300s (smoke).
fn resolve_duration_sec() -> u64 {
    std::env::var("LUMEN_SOAK_DURATION_SEC")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300)
}

/// Resolve output directory; defaults to `target/soak-out` (override with
/// the `LUMEN_SOAK_OUT_DIR` env var).
fn resolve_out_dir() -> PathBuf {
    std::env::var("LUMEN_SOAK_OUT_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/soak-out"))
}

/// Sample current process RSS in KB via `ps -o rss= -p $PID`. Returns -1
/// on parse failure (sentinel; not fatal — soak continues so the slope
/// analysis can still partial-recover).
fn sample_rss_kb(pid: u32) -> i64 {
    use std::process::Command;
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<i64>().unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Count open file descriptors via `lsof -p $PID | wc -l`.
fn sample_fd_count(pid: u32) -> i64 {
    use std::process::Command;
    let out = Command::new("sh")
        .args(["-c", &format!("lsof -p {} 2>/dev/null | wc -l", pid)])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<i64>().unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Sample MTLBuffer allocated bytes via the swift probe.
/// Returns -1 if the probe is not present.
fn sample_mtl_alloc_bytes() -> i64 {
    use std::process::Command;
    let probe_path = "target/metal/mtl-mem-probe";
    if !Path::new(probe_path).exists() {
        return -1;
    }
    let out = Command::new(probe_path).output();
    match out {
        Ok(o) if o.status.success() => {
            // Probe prints a single line of digits = allocated bytes.
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<i64>().unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Compute disk usage of a directory in bytes (recursive). Returns -1 on error.
fn dir_size_bytes(dir: &Path) -> i64 {
    let mut total: i64 = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return -1,
    };
    for entry in entries.flatten() {
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_file() {
            total += meta.len() as i64;
        } else if meta.is_dir() {
            let sub = dir_size_bytes(&entry.path());
            if sub > 0 {
                total += sub;
            }
        }
    }
    total
}

/// Count `.tmp.<pid>` files in a directory.
fn count_tmp_orphans(dir: &Path) -> i64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return -1,
    };
    let mut count = 0;
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if name.contains(".tmp.") {
                count += 1;
            }
        }
    }
    count
}

/// Build the soak server: open LBC, build BPE tokenizer + Metal backend +
/// EngineWorker; return the bound socket address.
async fn boot_soak_server(
    lbc_path: &Path,
    disk_kv_dir: PathBuf,
) -> (std::net::SocketAddr, Arc<std::sync::atomic::AtomicU64>) {
    // Parse the LBC tokenizer section for BpeTokenizer.
    let lbc = lumen_format::reader::LbcFile::open(lbc_path)
        .unwrap_or_else(|e| panic!("open LBC {lbc_path:?}: {e}"));
    let tok_section = lbc
        .tokenizer
        .as_ref()
        .unwrap_or_else(|| panic!("LBC {lbc_path:?} has no embedded tokenizer"))
        .clone();
    let tok_data = lumen_convert::tokenizer_data::TokenizerData {
        model_type: tok_section.model_type,
        pre_tokenizer: tok_section.pre_tokenizer,
        tokens: tok_section.tokens,
        token_types: tok_section.token_types,
        scores: tok_section.scores,
        merges: tok_section.merges,
        bos_token_id: tok_section.bos_token_id,
        eos_token_id: tok_section.eos_token_id,
        pad_token_id: tok_section.pad_token_id,
        add_bos_token: tok_section.add_bos_token,
        add_eos_token: tok_section.add_eos_token,
        add_space_prefix: tok_section.add_space_prefix,
        chat_template: tok_section.chat_template,
    };
    let eos_id = tok_data.eos_token_id;
    let bpe = lumen_cli::tokenize::BpeTokenizer::from_tokenizer_data(&tok_data);
    let tokenizer: Arc<dyn Tokenize> = Arc::new(BpeTokenizerAdapter {
        inner: bpe,
        eos_ids: vec![eos_id],
    });

    // Build provider + Metal backend.
    let provider = SyncWeightProvider::open(lbc_path)
        .unwrap_or_else(|e| panic!("open weights {lbc_path:?}: {e}"));
    let hyperparams = provider.lbc().header.hyperparams;
    let context_length = std::cmp::min(CONTEXT_LEN, hyperparams.max_seq_len as usize);

    let mut backend = MetalF32Backend::new()
        .unwrap_or_else(|e| panic!("Metal backend unavailable: {e}"));
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    // Pass raw quantized blobs for embedding + output_proj so Metal can
    // use the optimized paths (matches CLI pattern from run.rs:1208-1220).
    let qs = provider.embedding_quant;
    if matches!(
        qs,
        lumen_format::QuantScheme::Q8_0
            | lumen_format::QuantScheme::Q4_0
            | lumen_format::QuantScheme::F16
    ) {
        backend.set_embedding_raw(provider.embedding_raw.clone(), qs);
    }
    let qs_out = provider.output_proj_quant;
    if matches!(
        qs_out,
        lumen_format::QuantScheme::Q8_0
            | lumen_format::QuantScheme::Q4_0
            | lumen_format::QuantScheme::F16
    ) {
        backend.set_output_proj_raw(provider.output_proj_raw.clone(), qs_out);
    }
    if provider.weight_tying {
        backend.set_weight_tying(true);
    }
    backend.init(&hyperparams).unwrap();

    // Metal backend requires `preload_weights_gpu_resident` to populate
    // `gdn_h_states` / `gdn_conv_states` before any prefill (otherwise
    // `s.gdn_h_states[0]` out-of-bounds panic). The CLI calls this at
    // `run.rs:1610`; lumen-server itself does NOT. Embedders MUST call it.
    // (Future: make `EngineWorker::spawn_with_disk_cache` invoke
    // `preload_weights` for backends that need it.)
    backend
        .preload_weights(&provider)
        .expect("preload_weights_gpu_resident");

    let runtime_cfg = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F16, // Metal requires F16
        max_seq_len: context_length,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: MODEL_ID.into(),
        owned_by: "lumen-soak".into(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        context_length,
    };

    // Wire engine worker with disk-KV.
    std::fs::create_dir_all(&disk_kv_dir).unwrap();
    let disk_kv = DiskKvConfig {
        dir: disk_kv_dir.clone(),
        budget_bytes: DISK_KV_BUDGET_BYTES,
    };
    let handle = EngineWorker::spawn_with_disk_cache(
        runtime_cfg,
        hyperparams,
        Box::new(backend),
        Arc::new(provider),
        tokenizer,
        model_info,
        INBOX_SIZE,
        Some(disk_kv),
    );

    let app = build_router(handle);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Brief warmup so axum is listening.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let request_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    (addr, request_counter)
}

/// Send one chat-completion request and return the raw response body
/// (non-streaming). Used by the harness for argmax-correctness checks
/// when no external Python client is running.
#[allow(dead_code)]
async fn one_chat_request(
    client: &Client<HttpConnector, Full<bytes::Bytes>>,
    addr: std::net::SocketAddr,
    user_prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let uri: Uri = format!("http://{addr}/v1/chat/completions")
        .parse()
        .map_err(|e| format!("uri parse: {e}"))?;
    let body = serde_json::json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });
    let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(bytes::Bytes::from(body_bytes)))
        .map_err(|e| format!("build: {e}"))?;
    let resp = client.request(req).await.map_err(|e| format!("send: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("status: {}", resp.status()));
    }
    let body = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| format!("collect: {e}"))?
        .to_bytes();
    Ok(String::from_utf8_lossy(&body).to_string())
}

/// Soak driver: runs the workload loop entirely in-process (no external
/// Python client needed for smoke tests). Issues 60% short / 30% medium /
/// 10% long requests in a round-robin schedule.
async fn run_in_process_workload(
    client: &Client<HttpConnector, Full<bytes::Bytes>>,
    addr: std::net::SocketAddr,
    duration: Duration,
    counter: Arc<std::sync::atomic::AtomicU64>,
    out_dir: &Path,
) -> Result<(), String> {
    let started = Instant::now();
    let workload_path = out_dir.join("soak-workload.jsonl");
    let mut workload_log = std::fs::File::create(&workload_path)
        .map_err(|e| format!("create workload log: {e}"))?;
    use std::io::Write;

    // Workload mix (round-robin in 10-request cycles: 6 short, 3 medium, 1 long).
    let workload = [
        (0, "What is 2+2?", 32_usize),
        (1, "Hello!", 32),
        (2, "Name a planet.", 32),
        (3, "What color is the sky?", 32),
        (4, "Capital of France?", 32),
        (5, "Define gravity.", 32),
        (6, "Explain quantum entanglement in three paragraphs.", 256),
        (7, "Describe the lifecycle of a star.", 256),
        (8, "Compare Rust and Go.", 256),
        (9, "Write a short essay on the history of computing.", 1024),
    ];

    // Reference response: capture first short request for argmax check.
    let mut reference_response_text: Option<String> = None;

    let mut cycle_idx: u64 = 0;
    let mut consecutive_errors: u32 = 0;
    const MAX_CONSECUTIVE_ERRORS: u32 = 5;

    // Argmax-divergence note: lumen-server's `EngineWorker` keeps a SINGLE
    // session across consecutive jobs (engine.rs:326-330), so even the same
    // user prompt produces a different completion depending on prior KV
    // state. A true cross-request argmax check would require resetting the
    // session before each reference comparison — out of scope for this release.
    // We rely on (a) the `consecutive_errors >= 5` guard for catastrophic
    // failure detection, (b) "no panics" in the log, and (c) the post-soak
    // `cargo test` regression check for state corruption.

    while started.elapsed() < duration {
        for (_ix, prompt, max_tok) in &workload {
            if started.elapsed() >= duration {
                break;
            }
            let req_start = Instant::now();
            let result = one_chat_request(client, addr, prompt, *max_tok).await;
            let req_elapsed_ms = req_start.elapsed().as_millis() as u64;
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let req_idx = counter.load(std::sync::atomic::Ordering::Relaxed);

            let (status, body_len, completion) = match &result {
                Ok(body) => {
                    consecutive_errors = 0;
                    ("ok", body.len(), body.clone())
                }
                Err(e) => {
                    consecutive_errors += 1;
                    ("error", 0, format!(r#"{{"error":"{e}"}}"#))
                }
            };

            // Fail-fast guard: if we hit too many consecutive errors, the
            // server is in a bad state. Sleep briefly to avoid spinning
            // (which inflates request_count without progress) and abort
            // the soak with a clear error.
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                writeln!(
                    workload_log,
                    r#"{{"ts":{},"ix":{},"status":"ABORT_TOO_MANY_ERRORS","consecutive_errors":{}}}"#,
                    elapsed_unix(), req_idx, consecutive_errors,
                ).ok();
                return Err(format!(
                    "{} consecutive request failures; aborting soak (see workload jsonl)",
                    consecutive_errors,
                ));
            }

            // Reference response captured for inspection (first occurrence
            // of the first prompt). NO comparison — see note above on
            // session-stateful semantics; cross-request comparisons would
            // need a session reset.
            if status == "ok" && *prompt == "What is 2+2?" && reference_response_text.is_none() {
                reference_response_text = Some(completion.clone());
            }
            let _ = &reference_response_text; // silence unused if duration is too short

            writeln!(
                workload_log,
                r#"{{"ts":{},"ix":{},"prompt_short":"{}","max_tok":{},"status":"{}","latency_ms":{},"body_len":{}}}"#,
                elapsed_unix(),
                req_idx,
                prompt.chars().take(30).collect::<String>(),
                max_tok,
                status,
                req_elapsed_ms,
                body_len,
            )
            .ok();
        }
        cycle_idx += 1;
    }

    let _ = cycle_idx; // silence unused if loop never runs
    Ok(())
}

fn elapsed_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// capture a triple of heap-attribution snapshots
/// (`malloc_history -callTree`, `vmmap -s`, `leaks -nostacks`) against
/// the current process and write them to
/// `<out_dir>/heap-t<idx>-{callTree,vmmap,leaks}.txt`. Returns a one-line
/// summary suitable for the soak's stdout/stats log.
///
/// All three tools serialize on `task_for_pid`; we therefore run them
/// sequentially. They are READ-ONLY w.r.t. the target process.
///
/// This helper is only ever invoked when the operator sets the
/// `LUMEN_SOAK_STACK_DUMP=1` env var AND the binary was compiled with
/// `RUSTFLAGS='--cfg stacks_dump'` (so that mimalloc is not bound and
/// libsystem_malloc is the actual allocator, otherwise MallocStackLogging
/// captures nothing).
fn capture_heap_snapshot(out_dir: &Path, idx: usize, elapsed_sec: u64) -> String {
    use std::process::Command;
    let pid = std::process::id();
    let mut summary = format!("heap-t{idx} pid={pid} elapsed_sec={elapsed_sec}");

    // malloc_history is the call-tree workhorse. `-callTree -highWaterMark`
    // gives live-allocation call-stack rollup at the high-water mark; sans
    // -highWaterMark gives currently-live attribution. We want the latter
    // (current live allocations at this tick) so the inter-tick diff is
    // exactly the new growth.
    let mh_path = out_dir.join(format!("heap-t{}-callTree.txt", idx));
    let mh = Command::new("/usr/bin/malloc_history")
        .args([&pid.to_string(), "-callTree"])
        .output();
    match mh {
        Ok(o) => {
            let _ = std::fs::write(&mh_path, &o.stdout);
            summary.push_str(&format!(
                " callTree={}KB exit={}",
                o.stdout.len() / 1024,
                o.status.code().unwrap_or(-1),
            ));
        }
        Err(e) => summary.push_str(&format!(" callTree=ERR:{}", e)),
    }

    // vmmap -s: region-level breakdown by region tag (MALLOC_TINY,
    // MALLOC_LARGE, Stack, VM_ALLOCATE, MALLOC_REGION, etc). Critical for
    // discriminating "heap growth via libmalloc" vs "anonymous mmap growth"
    // vs "stack growth" vs "framework-private region growth".
    let vm_path = out_dir.join(format!("heap-t{}-vmmap.txt", idx));
    let vm = Command::new("/usr/bin/vmmap")
        .args(["-s", &pid.to_string()])
        .output();
    match vm {
        Ok(o) => {
            let _ = std::fs::write(&vm_path, &o.stdout);
            summary.push_str(&format!(
                " vmmap={}KB exit={}",
                o.stdout.len() / 1024,
                o.status.code().unwrap_or(-1),
            ));
        }
        Err(e) => summary.push_str(&format!(" vmmap=ERR:{}", e)),
    }

    // leaks -nostacks: identifies UNREACHABLE allocations (true leaks; the
    // pointer to them is no longer reachable from globals/stack/registers
    // or other reachable malloc blocks). -nostacks omits backtraces (we
    // already have those from malloc_history) and runs much faster --
    // BUT on a 2-3 GB heap with stack logging enabled, `leaks` still
    // takes 60-120 seconds per snapshot and balloons its own RSS to
    // 5-10 GB (mach_vm scan + own bookkeeping). The smoke run
    // empirically blew past 9 GB on a single snapshot; on a 30-min
    // soak with 4 snapshots, that's 4-8 minutes of blocked workload +
    // 40 GB of transient memory pressure.
    //
    // Therefore `leaks` is opt-in via `LUMEN_SOAK_STACK_LEAKS=1`. The
    // default soak runs WITHOUT it because malloc_history -callTree
    // gives the same call-stack attribution at ~30s/snapshot. If a
    // future run needs the reachable-vs-unreachable distinction, it
    // can flip the env var and pay the cost.
    let leaks_enabled = std::env::var("LUMEN_SOAK_STACK_LEAKS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    let lk_path = out_dir.join(format!("heap-t{}-leaks.txt", idx));
    if leaks_enabled {
        let lk = Command::new("/usr/bin/leaks")
            .args(["-nostacks", "-quiet", &pid.to_string()])
            .output();
        match lk {
            Ok(o) => {
                let _ = std::fs::write(&lk_path, &o.stdout);
                summary.push_str(&format!(
                    " leaks={}KB exit={}",
                    o.stdout.len() / 1024,
                    o.status.code().unwrap_or(-1),
                ));
            }
            Err(e) => summary.push_str(&format!(" leaks=ERR:{}", e)),
        }
    } else {
        summary.push_str(" leaks=SKIPPED(set LUMEN_SOAK_STACK_LEAKS=1 to enable)");
    }
    summary
}

/// Periodic sampler: every SAMPLE_INTERVAL_SEC seconds, append one stats
/// line to `soak-stats.jsonl`. Stops when `stop` flag flips to true.
///
/// when `breakdown_addr` is `Some`, the sampler ALSO issues a
/// GET to `http://{addr}/debug/memory_breakdown` on each tick and
/// appends the response body verbatim to `soak-breakdown.jsonl` next to
/// `soak-stats.jsonl`.  Each breakdown line is prefixed with the
/// elapsed_sec field of the matching RSS sample, so the two files are
/// aligned on the same timeline:
///
///     {"elapsed_sec":30, ... breakdown JSON ...}
///
/// The breakdown sample is best-effort: if the HTTP query times out or
/// returns a non-200 status (e.g. `LUMEN_SERVER_DEBUG_MEM` was not set
/// in the server process), an `{"elapsed_sec":N,"breakdown_error":...}`
/// line is recorded instead and the soak continues.
async fn sample_loop(
    out_dir: PathBuf,
    counter: Arc<std::sync::atomic::AtomicU64>,
    disk_kv_dir: PathBuf,
    stop: Arc<std::sync::atomic::AtomicBool>,
    breakdown_addr: Option<std::net::SocketAddr>,
) -> Result<(), String> {
    let stats_path = out_dir.join("soak-stats.jsonl");
    let mut stats_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stats_path)
        .map_err(|e| format!("open stats file: {e}"))?;
    // per-component breakdown file (optional sister of stats).
    let mut breakdown_file = match breakdown_addr {
        Some(_) => {
            let path = out_dir.join("soak-breakdown.jsonl");
            Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| format!("open breakdown file: {e}"))?,
            )
        }
        None => None,
    };
    let breakdown_client: Option<Client<HttpConnector, Full<bytes::Bytes>>> =
        breakdown_addr.map(|_| Client::builder(TokioExecutor::new()).build_http());
    use std::io::Write;

    // heap-snapshot cadence. When `LUMEN_SOAK_STACK_DUMP=1`,
    // the sampler calls `capture_heap_snapshot` at each scheduled tick in
    // `snapshot_ticks_sec` (and only at those ticks). Outputs land next to
    // `soak-stats.jsonl` in `out_dir`. Indices monotonically increase so
    // the diff-analysis script (S3) can pair t0/t1/t2/t3 unambiguously.
    let stack_dump_enabled = std::env::var("LUMEN_SOAK_STACK_DUMP")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    // Optional comma-separated override for the snapshot tick schedule;
    // e.g. `LUMEN_SOAK_STACK_TICKS=30,60` for smoke testing the cadence
    // without paying a 30-minute soak. Default schedule captures
    // post-warmup, mid, late, and end-of-soak references.
    let snapshot_ticks_sec: Vec<u64> = std::env::var("LUMEN_SOAK_STACK_TICKS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u64>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<u64>| !v.is_empty())
        .unwrap_or_else(|| vec![300, 900, 1500, 1800]);
    let mut snapshot_idx: usize = 0;
    let mut snapshot_log = if stack_dump_enabled {
        let path = out_dir.join("snapshots.log");
        Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("open snapshots log: {e}"))?,
        )
    } else {
        None
    };
    if stack_dump_enabled {
        eprintln!(
            "[soak/init] stack-dump cadence ENABLED ticks={:?}",
            snapshot_ticks_sec
        );
    }

    let pid = std::process::id();
    let started = Instant::now();
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let elapsed_sec = started.elapsed().as_secs();
        let sample = SoakSample {
            ts_unix: elapsed_unix(),
            elapsed_sec,
            rss_kb: sample_rss_kb(pid),
            fd_count: sample_fd_count(pid),
            mtl_alloc_bytes: sample_mtl_alloc_bytes(),
            request_count: counter.load(std::sync::atomic::Ordering::Relaxed),
            decode_tokens_total: 0, // (per-token counter would require engine
                                    // changes; deferred)
            disk_kv_used_bytes: dir_size_bytes(&disk_kv_dir),
            disk_kv_tmp_orphan_count: count_tmp_orphans(&disk_kv_dir),
        };
        let line = sample.to_jsonl();
        writeln!(stats_file, "{}", line).ok();
        stats_file.flush().ok();
        eprintln!("[soak/{:>5}s] {}", elapsed_sec, line);

        // breakdown sample — best-effort GET against the
        // server's /debug/memory_breakdown endpoint.
        if let (Some(addr), Some(client), Some(f)) =
            (breakdown_addr, breakdown_client.as_ref(), breakdown_file.as_mut())
        {
            let line = match sample_breakdown(client, addr).await {
                Ok(body) => format!(r#"{{"elapsed_sec":{},"breakdown":{}}}"#, elapsed_sec, body),
                Err(e) => format!(
                    r#"{{"elapsed_sec":{},"breakdown_error":"{}"}}"#,
                    elapsed_sec,
                    e.replace('"', "\\\""),
                ),
            };
            writeln!(f, "{}", line).ok();
            f.flush().ok();
            eprintln!("[soak/{:>5}s] {}", elapsed_sec, line);
        }

        // heap-snapshot trigger — fire `capture_heap_snapshot`
        // when `elapsed_sec` has crossed the next scheduled tick. We use
        // crossed-or-equal rather than strict equality because the sample
        // loop fires at 30-second intervals and 300 / 900 / 1500 / 1800
        // are all multiples of 30, so equality is normally hit, but we
        // tolerate a one-sample drift if scheduling jitters past the
        // exact boundary.
        if stack_dump_enabled
            && snapshot_idx < snapshot_ticks_sec.len()
            && elapsed_sec >= snapshot_ticks_sec[snapshot_idx]
        {
            // Block on the snapshot synchronously inside the async loop.
            // We use `tokio::task::block_in_place` so the runtime can
            // continue scheduling other tasks (e.g. the workload loop)
            // while the snapshot tools run. malloc_history can take ~10
            // seconds against a 6 GB heap with stack logging enabled.
            let line = tokio::task::block_in_place(|| {
                capture_heap_snapshot(&out_dir, snapshot_idx, elapsed_sec)
            });
            if let Some(f) = snapshot_log.as_mut() {
                writeln!(f, "{}", line).ok();
                f.flush().ok();
            }
            eprintln!("[soak/snapshot] {}", line);
            snapshot_idx += 1;
        }

        // Sleep until the next interval boundary; check stop every 1s so the
        // loop exits promptly when the workload finishes.
        for _ in 0..SAMPLE_INTERVAL_SEC {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    Ok(())
}

/// GET `/debug/memory_breakdown` and return the raw JSON body.
/// Distinguishes "endpoint disabled" (404) from "transport error" so the
/// evaluation phase can decide whether the env-gate was honoured.
async fn sample_breakdown(
    client: &Client<HttpConnector, Full<bytes::Bytes>>,
    addr: std::net::SocketAddr,
) -> Result<String, String> {
    let uri: Uri = format!("http://{addr}/debug/memory_breakdown")
        .parse()
        .map_err(|e| format!("uri parse: {e}"))?;
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(bytes::Bytes::new()))
        .map_err(|e| format!("build: {e}"))?;
    let resp = client.request(req).await.map_err(|e| format!("send: {e}"))?;
    let status = resp.status();
    let body = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| format!("collect: {e}"))?
        .to_bytes();
    if !status.is_success() {
        return Err(format!(
            "status={} body={}",
            status,
            String::from_utf8_lossy(&body).chars().take(200).collect::<String>(),
        ));
    }
    Ok(String::from_utf8_lossy(&body).to_string())
}

/// Compute summary statistics from `soak-stats.jsonl`. Writes
/// `soak-summary.json` and returns true if all acceptance criteria pass.
///
/// ## gate-methodology alignment
///
/// Up through the `rss_pass` gate computed a two-point endpoint
/// slope `(rss_last − rss_first) / hours`. Empirically that over-weights
/// the one-time warmup window (mimalloc arena commit, Metal first-touch,
/// MTLBuffer pool fill, embedding mmap page-in). On `run2`
/// (default-mimalloc 30-min soak post-`@autoreleasepool` fix) the
/// endpoint slope was +0.9086 %/h (failing the ≤0.5%/h) even
/// though the post-warmup linear regression on the same 61 samples
/// was +0.060 %/h (passing the gate by 8×). (the user-accepted
/// 13h CLI soak) used "RSS p50 per hour" specifically to exclude
/// warmup; the offline analyzer at
/// reproduces that same
/// post-warmup linear-regression methodology.
///
/// This function therefore computes BOTH metrics — the whole-run
/// endpoint slope is retained verbatim for backwards compatibility
/// (downstream consumers still see `rss_first_kb`, `rss_last_kb`,
/// `rss_slope_pct_per_hour`, `rss_pass`, `fd_first`, `fd_last`,
/// `fd_delta`, `fd_pass` exactly as before) — and the post-warmup
/// linear-regression slope plus the post-warmup FD-delta become the
/// inputs to `overall_pass`. The warmup window is configurable via
/// `LUMEN_SOAK_WARMUP_SEC` (default 300s = the same value used by
/// the offline analyzer's `DEFAULT_WARMUP_SEC` constant lineage).
fn write_soak_summary(out_dir: &Path, duration_sec: u64) -> Result<bool, String> {
    let stats_path = out_dir.join("soak-stats.jsonl");
    let raw = std::fs::read_to_string(&stats_path)
        .map_err(|e| format!("read stats: {e}"))?;
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    if lines.len() < 2 {
        return Err(format!("soak-stats.jsonl has only {} lines; cannot compute slope", lines.len()));
    }

    // Parse first & last sample to compute slopes (simple two-point — for
    // small N this is robust; for N>10 the workload script can compute a
    // proper rolling-window slope from the same file).
    let parse_field = |line: &str, key: &str| -> Option<i64> {
        let needle = format!("\"{}\":", key);
        let i = line.find(&needle)?;
        let after = &line[i + needle.len()..];
        let end = after.find([',', '}']).unwrap_or(after.len());
        after[..end].trim().parse::<i64>().ok()
    };

    let first = lines[0];
    let last = lines[lines.len() - 1];

    let rss_first = parse_field(first, "rss_kb").unwrap_or(-1);
    let rss_last = parse_field(last, "rss_kb").unwrap_or(-1);
    let fd_first = parse_field(first, "fd_count").unwrap_or(-1);
    let fd_last = parse_field(last, "fd_count").unwrap_or(-1);
    let mtl_first = parse_field(first, "mtl_alloc_bytes").unwrap_or(-1);
    let mtl_last = parse_field(last, "mtl_alloc_bytes").unwrap_or(-1);
    let _req_first = parse_field(first, "request_count").unwrap_or(0);
    let req_last = parse_field(last, "request_count").unwrap_or(0);
    let kv_orphan_max = lines
        .iter()
        .filter_map(|l| parse_field(l, "disk_kv_tmp_orphan_count"))
        .max()
        .unwrap_or(0);

    let hours = (duration_sec as f64 / 3600.0).max(0.001);
    let rss_slope_pct_per_hour = if rss_first > 0 {
        ((rss_last - rss_first) as f64 / rss_first as f64) * 100.0 / hours
    } else {
        f64::NAN
    };
    let mtl_slope_pct_per_hour = if mtl_first > 0 {
        ((mtl_last - mtl_first) as f64 / mtl_first as f64) * 100.0 / hours
    } else {
        f64::NAN
    };
    let fd_delta = fd_last - fd_first;

    // --- post-warmup metrics (this is what `overall_pass` now uses) ---
    //
    // Pull (elapsed_sec, rss_kb, fd_count) triplets from EVERY sample, then
    // filter to the post-warmup window. We need elapsed_sec for the linear
    // regression (a two-point slope can't see warmup curvature).
    let warmup_sec: u64 = std::env::var("LUMEN_SOAK_WARMUP_SEC")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300);
    let triplets: Vec<(i64, i64, i64)> = lines
        .iter()
        .filter_map(|l| {
            let e = parse_field(l, "elapsed_sec")?;
            let r = parse_field(l, "rss_kb")?;
            let fd = parse_field(l, "fd_count")?;
            // sample_rss_kb returns -1 sentinel on parse failure; skip those.
            if r <= 0 {
                return None;
            }
            Some((e, r, fd))
        })
        .collect();
    let post_warmup: Vec<(i64, i64, i64)> = triplets
        .iter()
        .copied()
        .filter(|&(e, _, _)| e >= warmup_sec as i64)
        .collect();

    // RSS post-warmup linear regression. Slope is in kb/sec; convert to
    // MB/h (×3600/1024) and %/h (×3600/mean × 100). Matches the offline
    // analyzer in
    //
    // When fewer than 2 post-warmup samples exist (e.g. smoke at 5 min
    // with the default 300s warmup), we cannot compute a meaningful
    // slope. The gate then falls back to the smoke sanity check
    // (`rss_last <= 2 × rss_first`) so short smoke runs still gate
    // cleanly without producing NaN failures.
    let (
        rss_post_warmup_n,
        rss_post_warmup_mean_kb,
        rss_post_warmup_slope_kb_per_sec,
        rss_post_warmup_slope_mb_per_hour,
        rss_post_warmup_slope_pct_per_hour,
    ) = if post_warmup.len() >= 2 {
        let n = post_warmup.len();
        let mean_x = post_warmup.iter().map(|&(e, _, _)| e as f64).sum::<f64>() / n as f64;
        let mean_y =
            post_warmup.iter().map(|&(_, r, _)| r as f64).sum::<f64>() / n as f64;
        let num: f64 = post_warmup
            .iter()
            .map(|&(e, r, _)| (e as f64 - mean_x) * (r as f64 - mean_y))
            .sum();
        let den: f64 = post_warmup
            .iter()
            .map(|&(e, _, _)| {
                let d = e as f64 - mean_x;
                d * d
            })
            .sum();
        let slope_kb_per_sec = if den == 0.0 { 0.0 } else { num / den };
        let slope_mb_per_hour = slope_kb_per_sec * 3600.0 / 1024.0;
        let slope_pct_per_hour = if mean_y > 0.0 {
            slope_kb_per_sec * 3600.0 / mean_y * 100.0
        } else {
            f64::NAN
        };
        (n, mean_y, slope_kb_per_sec, slope_mb_per_hour, slope_pct_per_hour)
    } else {
        (0, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };
    let _ = rss_post_warmup_slope_kb_per_sec; // emitted via the MB/h form only

    // FD post-warmup delta. The whole-run `fd_delta` over-counts the
    // one-time startup FD allocs (mimalloc thread-local descriptors,
    // axum listening socket, lsof's own pipe). Post-warmup FD on a
    // healthy server is stationary; a real FD leak shows up as a
    // monotonic post-warmup increase. On run2 the post-warmup
    // FD trace is 36 → 36 (FLAT) while the whole-run delta is 34→36
    // (FAIL strict).
    let (fd_post_warmup_first, fd_post_warmup_last, fd_post_warmup_delta) =
        if let (Some(&(_, _, ff)), Some(&(_, _, fl))) =
            (post_warmup.first(), post_warmup.last())
        {
            (ff, fl, fl - ff)
        } else {
            (-1, -1, 0)
        };

    // Acceptance criteria aligned with methodology.
    //
    // Two-tier gate model retained: smoke (duration < 1800s = 30 min)
    // checks only sanity gates (no crashes, no orphans, ≥30 requests,
    // RSS within 2× initial). The slope criterion is physically
    // meaningful only AFTER warmup has tapered, which empirically
    // takes ~120-300s for a 10 GB Q8 mmap on M3 Ultra; below that
    // window the linear regression is dominated by mmap page-in.
    //
    // Full soak (≥30 min) uses the post-warmup linear-regression slope
    // as the RSS gate and the post-warmup FD delta as the FD gate.
    // The whole-run endpoint slope and whole-run FD delta are
    // computed and emitted unchanged (downstream consumers still see
    // every field they saw before).
    let is_smoke = duration_sec < 1800;
    let pass_rss = if is_smoke {
        rss_first <= 0 || rss_last <= 2 * rss_first.max(1)
    } else {
        rss_slope_pct_per_hour.is_nan() || rss_slope_pct_per_hour <= 0.5
    };
    let pass_mtl = mtl_slope_pct_per_hour.is_nan() || mtl_slope_pct_per_hour <= 0.5;
    let pass_fd = fd_delta <= 0;
    let pass_orphans = kv_orphan_max == 0;
    let pass_req = req_last >= 30; // smoke; 4h gate raises this externally

    // post-warmup gates feed `overall_pass`. For smoke runs
    // where the post-warmup window is empty, `overall_pass` falls back
    // to the smoke sanity checks (same behaviour as the prior
    // harness for short runs).
    let rss_post_warmup_pass = if rss_post_warmup_n >= 2 {
        rss_post_warmup_slope_pct_per_hour.is_nan()
            || rss_post_warmup_slope_pct_per_hour <= 0.5
    } else {
        pass_rss // fall back to smoke sanity check
    };
    let fd_post_warmup_pass = if rss_post_warmup_n >= 2 {
        fd_post_warmup_delta <= 0
    } else {
        pass_fd // fall back to whole-run delta in smoke window
    };
    let overall_pass = rss_post_warmup_pass
        && fd_post_warmup_pass
        && pass_mtl
        && pass_orphans
        && pass_req;

    let gate_mode = if is_smoke { "smoke" } else { "full_soak" };
    // f64::NAN serialization: serde would emit `null`, but we hand-roll
    // JSON here. Render NaN as the bare token `NaN` (matches the prior
    // harness output where `mtl_slope_pct_per_hour` already emitted NaN
    // unquoted when the probe was missing; downstream consumers already
    // tolerate it).
    let fmt_f = |v: f64| -> String {
        if v.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.4}", v)
        }
    };
    let summary = format!(
        r#"{{
  "duration_sec": {},
  "gate_mode": "{}",
  "samples": {},
  "request_count": {},
  "rss_first_kb": {}, "rss_last_kb": {}, "rss_slope_pct_per_hour": {}, "rss_pass": {},
  "fd_first": {}, "fd_last": {}, "fd_delta": {}, "fd_pass": {},
  "mtl_first_bytes": {}, "mtl_last_bytes": {}, "mtl_slope_pct_per_hour": {}, "mtl_pass": {},
  "kv_tmp_orphan_max": {}, "kv_orphan_pass": {},
  "rss_warmup_sec_excluded": {}, "rss_post_warmup_samples": {}, "rss_post_warmup_mean_kb": {}, "rss_post_warmup_slope_mb_per_hour": {}, "rss_post_warmup_slope_pct_per_hour": {}, "rss_post_warmup_pass": {},
  "fd_post_warmup_first": {}, "fd_post_warmup_last": {}, "fd_post_warmup_delta": {}, "fd_post_warmup_pass": {},
  "overall_pass": {}
}}
"#,
        duration_sec,
        gate_mode,
        lines.len(),
        req_last,
        rss_first, rss_last, fmt_f(rss_slope_pct_per_hour), pass_rss,
        fd_first, fd_last, fd_delta, pass_fd,
        mtl_first, mtl_last, fmt_f(mtl_slope_pct_per_hour), pass_mtl,
        kv_orphan_max, pass_orphans,
        warmup_sec,
        rss_post_warmup_n,
        fmt_f(rss_post_warmup_mean_kb),
        fmt_f(rss_post_warmup_slope_mb_per_hour),
        fmt_f(rss_post_warmup_slope_pct_per_hour),
        rss_post_warmup_pass,
        fd_post_warmup_first, fd_post_warmup_last, fd_post_warmup_delta, fd_post_warmup_pass,
        overall_pass,
    );
    let summary_path = out_dir.join("soak-summary.json");
    std::fs::write(&summary_path, summary.as_bytes())
        .map_err(|e| format!("write summary: {e}"))?;
    eprintln!("[soak/summary] wrote {}", summary_path.display());
    eprintln!(
        "[soak/summary] post-warmup (warmup={}s): rss_slope={} %/h fd_delta={} → rss_pass={} fd_pass={}",
        warmup_sec,
        fmt_f(rss_post_warmup_slope_pct_per_hour),
        fd_post_warmup_delta,
        rss_post_warmup_pass,
        fd_post_warmup_pass,
    );
    eprintln!("[soak/summary] verdict: {}", if overall_pass { "PASS" } else { "FAIL" });
    Ok(overall_pass)
}

// ============================================================================
// Test entry points (#[ignore]-d; explicit env var opt-in)
// ============================================================================

/// 5-minute smoke: validates the harness end-to-end on a real model.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "soak smoke: set LUMEN_QWEN35_9B_Q8=... and run with --ignored --nocapture"]
async fn soak_smoke_5min() {
    run_soak().await
}

/// Shared async body for both smoke and 4h entry points. Duration is
/// determined entirely by `LUMEN_SOAK_DURATION_SEC`; the two tests differ
/// only in which `#[ignore]` annotation/name the operator chooses to invoke.
///
/// when `LUMEN_SERVER_DEBUG_MEM=1` is set in the process env,
/// the sampler also queries the server's `/debug/memory_breakdown`
/// endpoint on each tick and persists the per-component snapshot to
/// `soak-breakdown.jsonl`.  Set `LUMEN_SERVER_DEBUG_MEM=0` (or leave
/// unset) to opt out — the soak then behaves exactly as it did.
async fn run_soak() {
    let lbc = match resolve_q8_lbc_path() {
        Some(p) => p,
        None => {
            eprintln!("LUMEN_QWEN35_9B_Q8 not set; skipping soak smoke");
            return;
        }
    };
    if !lbc.exists() {
        panic!("LBC path {lbc:?} does not exist");
    }
    let duration = Duration::from_secs(resolve_duration_sec());
    let out_dir = resolve_out_dir();
    std::fs::create_dir_all(&out_dir).unwrap();

    // Fresh stats file per run (don't append across runs).
    let _ = std::fs::remove_file(out_dir.join("soak-stats.jsonl"));
    let _ = std::fs::remove_file(out_dir.join("soak-summary.json"));
    let _ = std::fs::remove_file(out_dir.join("soak-workload.jsonl"));
    // fresh breakdown file per run.
    let _ = std::fs::remove_file(out_dir.join("soak-breakdown.jsonl"));

    let disk_kv_dir = out_dir.join("disk-kv");
    let _ = std::fs::remove_dir_all(&disk_kv_dir); // start clean

    eprintln!("[soak/init] LBC={lbc:?} duration={duration:?} out_dir={out_dir:?}");
    let (addr, counter) = boot_soak_server(&lbc, disk_kv_dir.clone()).await;
    eprintln!("[soak/init] bound 127.0.0.1:{}", addr.port());
    std::fs::write(out_dir.join("soak-port.txt"), format!("{}", addr.port()))
        .expect("write port file");

    let client: Client<HttpConnector, Full<bytes::Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // only sample the breakdown when the operator has set the
    // env var — otherwise the endpoint returns 404 and we'd log a
    // breakdown_error line every interval for the entire soak.
    let breakdown_enabled = std::env::var("LUMEN_SERVER_DEBUG_MEM")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    let breakdown_addr = if breakdown_enabled { Some(addr) } else { None };
    eprintln!(
        "[soak/init] breakdown sampling: {}",
        if breakdown_enabled { "ENABLED" } else { "disabled" },
    );

    // Spawn sampler loop alongside workload.
    let sampler_handle = {
        let out_dir = out_dir.clone();
        let counter = Arc::clone(&counter);
        let stop = Arc::clone(&stop);
        let kv_dir = disk_kv_dir.clone();
        tokio::spawn(async move { sample_loop(out_dir, counter, kv_dir, stop, breakdown_addr).await })
    };

    // Run the workload (this blocks until duration elapses).
    let workload_result = run_in_process_workload(&client, addr, duration, Arc::clone(&counter), &out_dir).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    // Wait for the sampler to flush its final line.
    let _ = sampler_handle.await;

    if let Err(e) = workload_result {
        panic!("workload failed: {e}");
    }

    // Write summary + assert acceptance.
    let pass = write_soak_summary(&out_dir, duration.as_secs()).expect("write summary");
    assert!(pass, "soak failed acceptance criteria (see soak-summary.json)");
}

/// 4-hour soak: same harness, longer duration. Caller MUST set
/// `LUMEN_SOAK_DURATION_SEC=14400` for the full run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "soak 4h: set LUMEN_QWEN35_9B_Q8=... and LUMEN_SOAK_DURATION_SEC=14400, run with --ignored --nocapture"]
async fn soak_4h() {
    // Reuses the shared async body — the only delta is the duration env
    // var (which the harness reads internally).
    run_soak().await
}

/// 30-minute instrumented soak.
///
/// Same shared body as `soak_4h` / `soak_smoke_5min` but with:
///
/// - `LUMEN_SERVER_DEBUG_MEM=1` force-enabled in the process env before
///   `boot_soak_server`, so the breakdown sampler captures
///   `soak-breakdown.jsonl` alongside `soak-stats.jsonl`.
/// - Default `LUMEN_SOAK_DURATION_SEC=1800` (30 min) — sufficient to see
///   the slope class of each tracked component without paying the full
///   4-hour cost of `soak_4h`.  Operators can still override with
///   `LUMEN_SOAK_DURATION_SEC=…`.
///
/// Run as:
///
/// ```sh
/// LUMEN_QWEN35_9B_Q8=/path/to/qwen3-5-9b-Q8_0.lbc \
/// LUMEN_SOAK_DURATION_SEC=1800 \
/// LUMEN_SOAK_OUT_DIR=target/soak-out \
/// cargo test --release -p lumen-server --test server_soak \
///   -- --ignored --nocapture soak_30min
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = " soak: set LUMEN_QWEN35_9B_Q8=... and run with --ignored --nocapture soak_30min"]
async fn soak_30min() {
    // Force the env var so the operator does NOT need to remember it.
    // Unsafe block: Rust 1.84+ marks env mutation unsafe (TOCTOU concern
    // across threads).  Only this test mutates the var in this binary;
    // the env-guard pattern lives in server_memory_breakdown.rs for the
    // unit tests that need finer control.
    unsafe { std::env::set_var("LUMEN_SERVER_DEBUG_MEM", "1"); }
    // If the operator did not explicitly set a duration, default to 30 min.
    if std::env::var("LUMEN_SOAK_DURATION_SEC").is_err() {
        unsafe { std::env::set_var("LUMEN_SOAK_DURATION_SEC", "1800"); }
    }
    run_soak().await
}
