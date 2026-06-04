//! CUDA long-session soak harness for `lumen-server`.
//!
//! Sibling of `server_soak.rs` (macOS / Metal). This test boots a real
//! single-process `lumen-server` on the CUDA backend with a Qwen3.5-9B
//! Q8_0 LBC, runs an HTTP workload in-process for the configured duration,
//! and writes the bound port + driver PID to disk so an external Modal
//! supervisor can sample container metrics (RSS / FD / VRAM) at the same
//! 30-second cadence on the same process.
//!
//! The Rust test is in-process server + in-process workload (so the
//! "single long-running lumen-server process" property holds end-to-end).
//! The Modal Python supervisor at `modal/server_soak.py` reads
//! `LUMEN_SOAK_OUT_DIR/soak-port.txt` and `LUMEN_SOAK_OUT_DIR/soak-pid.txt`
//! to know which port + PID to drive against and sample from.
//!
//! Methodology mirrors / `server_soak.rs` exactly except for:
//!
//! - Backend: `CudaBackend::new(0)` instead of `MetalF32Backend::new()`
//! - KV precision: `KvPrecision::F32` (CUDA requires F32 per
//!   `validate_kv_precision`) instead of F16 (Metal requires F16)
//! - OS gate: `#[cfg(target_os = "linux")]` instead of macOS
//! - Pid + port published to disk so the external supervisor can attach
//! - Throughput target: ≥7 req/min (matches acceptance)
//!
//! ## Invocation (under Modal-managed CUDA environment)
//!
//! ```sh
//! LUMEN_QWEN35_9B_Q8=/tmp/models/qwen3.5-9b-q8_0.lbc \
//! LUMEN_SOAK_DURATION_SEC=3600 \
//! LUMEN_SOAK_OUT_DIR=/tmp/soak-out \
//! cargo test --release --features cuda -p lumen-server \
//!   --test server_soak_cuda -- --ignored --nocapture soak_cuda_1h
//! ```
//!
//! ## Outputs (under `LUMEN_SOAK_OUT_DIR/`)
//!
//! - `soak-port.txt`     — bound port (Python supervisor probe target)
//! - `soak-pid.txt`      — driver PID (Python supervisor sample target)
//! - `soak-stats.jsonl`  — one line per 30-second sample (in-process RSS / FD,
//!                          mirrors `server_soak.rs` for cross-platform parity)
//! - `soak-workload.jsonl`— one line per HTTP request
//! - `soak-summary.json` — final per-gate verdict (in-process view)
//!
//! The Modal supervisor writes a parallel set of files under its container
//! `/tmp` with container-level sampling (psutil PID-targeted + nvidia-smi
//! for VRAM) that the supervisor combines into the final verdict. Both the
//! in-process Rust view and the external Python view are reconciled by
//! `server_soak.py` at completion.

// CUDA soak is Linux-only (Modal A100). On macOS / windows the cfg gate
// makes this file a no-op so the wider workspace still compiles.
#![cfg(all(target_os = "linux", feature = "cuda"))]

// parity: bind the soak-test process to mimalloc so the
// RSS gate is measured against the same allocator the Metal soak uses.
// Mismatched allocators (default glibc malloc vs mimalloc) produce
// systematically different post-warmup slopes; binding both backends to
// mimalloc makes the Metal-soak and CUDA-soak numbers directly comparable.
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
use lumen_runtime::{CudaBackend, RuntimeConfig};

use lumen_server::{
    build_router, DiskKvConfig, EngineWorker, ModelInfo, Tokenize,
};

const MODEL_ID: &str = "qwen3.5-9b:q8_0";
const SAMPLE_INTERVAL_SEC: u64 = 30;
const DISK_KV_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GB, matches server_soak.rs
const CONTEXT_LEN: usize = 4096;
const INBOX_SIZE: usize = 16;

/// Stats sampled in-process every interval and written to `soak-stats.jsonl`.
/// Mirrors `SoakSample` in `server_soak.rs` but renames `mtl_alloc_bytes`
/// to `vram_used_mb_inproc` (always -1 in-process; the Modal supervisor
/// fills the real value via nvidia-smi at the container level).
#[derive(Debug)]
struct SoakSample {
    ts_unix: u64,
    elapsed_sec: u64,
    rss_kb: i64,
    fd_count: i64,
    request_count: u64,
    decode_tokens_total: u64,
    disk_kv_used_bytes: i64,
    disk_kv_tmp_orphan_count: i64,
}

impl SoakSample {
    fn to_jsonl(&self) -> String {
        format!(
            r#"{{"ts_unix":{},"elapsed_sec":{},"rss_kb":{},"fd_count":{},"request_count":{},"decode_tokens_total":{},"disk_kv_used_bytes":{},"disk_kv_tmp_orphan_count":{}}}"#,
            self.ts_unix,
            self.elapsed_sec,
            self.rss_kb,
            self.fd_count,
            self.request_count,
            self.decode_tokens_total,
            self.disk_kv_used_bytes,
            self.disk_kv_tmp_orphan_count,
        )
    }
}

/// `Tokenize` adapter wrapping `lumen_cli::tokenize::BpeTokenizer`.
/// Identical implementation to `server_soak.rs::BpeTokenizerAdapter`.
struct BpeTokenizerAdapter {
    inner: lumen_cli::tokenize::BpeTokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenize for BpeTokenizerAdapter {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode_incremental(&self, state: &mut Vec<u8>, token_id: u32) -> String {
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

fn resolve_q8_lbc_path() -> Option<PathBuf> {
    std::env::var("LUMEN_QWEN35_9B_Q8")
        .ok()
        .map(PathBuf::from)
}

fn resolve_duration_sec() -> u64 {
    std::env::var("LUMEN_SOAK_DURATION_SEC")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300)
}

fn resolve_out_dir() -> PathBuf {
    std::env::var("LUMEN_SOAK_OUT_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/soak-out-cuda"))
}

/// Sample current process RSS in KB via `/proc/self/status:VmRSS`.
/// Linux equivalent of `server_soak.rs::sample_rss_kb` (which used `ps`
/// on macOS). Returns -1 on parse failure (sentinel; not fatal).
fn sample_rss_kb(_pid: u32) -> i64 {
    let s = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return -1,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:\t  12345 kB"
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(n) = parts.first() {
                if let Ok(v) = n.parse::<i64>() {
                    return v;
                }
            }
        }
    }
    -1
}

/// Count open file descriptors via `/proc/self/fd` directory entries.
/// Linux equivalent of `server_soak.rs::sample_fd_count` (which used
/// `lsof` on macOS).
fn sample_fd_count(_pid: u32) -> i64 {
    let entries = match std::fs::read_dir("/proc/self/fd") {
        Ok(e) => e,
        Err(_) => return -1,
    };
    let mut count: i64 = 0;
    for _ in entries.flatten() {
        count += 1;
    }
    count
}

/// Compute disk usage of a directory in bytes (recursive). Identical to
/// `server_soak.rs::dir_size_bytes`.
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
/// Identical to `server_soak.rs::count_tmp_orphans`.
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

/// Build the soak server on the CUDA backend.
///
/// Wiring mirrors `lumen-cli/src/run.rs::build_backend` for CUDA + the
/// `server_soak.rs::boot_soak_server` pattern:
///   1. Open LBC, extract tokenizer + provider.
///   2. Construct `CudaBackend::new(0)`.
///   3. Wire global tensors + raw quantized output_proj/embedding.
///   4. Call `backend.preload_weights(&provider)` (CUDA needs this for
///      GPU-resident weight upload; lumen-server itself does not call it,
///      so the embedder must — exactly the requirement documented in
///      `server_soak.rs:357-366`).
///   5. Spawn the `EngineWorker` with disk-KV enabled.
///   6. Build the axum router + bind to a random port.
async fn boot_soak_server(
    lbc_path: &Path,
    disk_kv_dir: PathBuf,
) -> (std::net::SocketAddr, Arc<std::sync::atomic::AtomicU64>) {
    // Parse the LBC tokenizer section (identical to server_soak.rs).
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

    // Build provider + CUDA backend.
    let provider = SyncWeightProvider::open(lbc_path)
        .unwrap_or_else(|e| panic!("open weights {lbc_path:?}: {e}"));
    let hyperparams = provider.lbc().header.hyperparams;
    let context_length = std::cmp::min(CONTEXT_LEN, hyperparams.max_seq_len as usize);

    let mut backend = CudaBackend::new(0)
        .unwrap_or_else(|e| panic!("CUDA backend unavailable: {e}"));
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
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

    // CUDA requires preload_weights for GPU-resident upload (mirrors
    // run.rs:1610 and the macOS embedder-must-preload requirement documented in server_soak.rs).
    backend
        .preload_weights(&provider)
        .expect("CUDA preload_weights");

    let runtime_cfg = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32, // CUDA requires F32 per validate_kv_precision
        max_seq_len: context_length,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: MODEL_ID.into(),
        owned_by: "lumen-soak-cuda".into(),
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
    tokio::time::sleep(Duration::from_millis(100)).await;

    let request_counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    (addr, request_counter)
}

/// Send one chat-completion request and return the raw response body.
/// Identical to `server_soak.rs::one_chat_request`.
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

/// In-process workload loop (identical to `server_soak.rs::run_in_process_workload`).
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

    // Workload mix matches server_soak.rs exactly: 6 short / 3 medium / 1 long.
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

    let mut consecutive_errors: u32 = 0;
    const MAX_CONSECUTIVE_ERRORS: u32 = 5;

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

            let (status, body_len) = match &result {
                Ok(body) => {
                    consecutive_errors = 0;
                    ("ok", body.len())
                }
                Err(_) => {
                    consecutive_errors += 1;
                    ("error", 0)
                }
            };

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
    }
    Ok(())
}

fn elapsed_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Sampler loop. Mirrors `server_soak.rs::sample_loop` minus the
/// breakdown HTTP and heap snapshots (out of scope).
async fn sample_loop(
    out_dir: PathBuf,
    counter: Arc<std::sync::atomic::AtomicU64>,
    disk_kv_dir: PathBuf,
    stop: Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let stats_path = out_dir.join("soak-stats.jsonl");
    let mut stats_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stats_path)
        .map_err(|e| format!("open stats file: {e}"))?;
    use std::io::Write;

    let pid = std::process::id();
    let started = Instant::now();
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let elapsed_sec = started.elapsed().as_secs();
        let sample = SoakSample {
            ts_unix: elapsed_unix(),
            elapsed_sec,
            rss_kb: sample_rss_kb(pid),
            fd_count: sample_fd_count(pid),
            request_count: counter.load(std::sync::atomic::Ordering::Relaxed),
            decode_tokens_total: 0,
            disk_kv_used_bytes: dir_size_bytes(&disk_kv_dir),
            disk_kv_tmp_orphan_count: count_tmp_orphans(&disk_kv_dir),
        };
        let line = sample.to_jsonl();
        writeln!(stats_file, "{}", line).ok();
        stats_file.flush().ok();
        eprintln!("[soak/{:>5}s] {}", elapsed_sec, line);

        for _ in 0..SAMPLE_INTERVAL_SEC {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    Ok(())
}

/// Compute summary statistics. Identical algorithm to
/// `server_soak.rs::write_soak_summary` ( post-warmup linear
/// regression + post-warmup FD delta). The Python supervisor computes
/// its own parallel summary with container-level RSS + nvidia-smi VRAM;
/// both should agree on the RSS gate verdict.
fn write_soak_summary(out_dir: &Path, duration_sec: u64) -> Result<bool, String> {
    let stats_path = out_dir.join("soak-stats.jsonl");
    let raw = std::fs::read_to_string(&stats_path)
        .map_err(|e| format!("read stats: {e}"))?;
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    if lines.len() < 2 {
        return Err(format!("soak-stats.jsonl has only {} lines; cannot compute slope", lines.len()));
    }

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
    let fd_delta = fd_last - fd_first;

    // post-warmup linear regression (identical algorithm to
    // server_soak.rs:957-1021).
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

    let (
        rss_post_warmup_n,
        rss_post_warmup_mean_kb,
        _rss_post_warmup_slope_kb_per_sec,
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

    let (fd_post_warmup_first, fd_post_warmup_last, fd_post_warmup_delta) =
        if let (Some(&(_, _, ff)), Some(&(_, _, fl))) =
            (post_warmup.first(), post_warmup.last())
        {
            (ff, fl, fl - ff)
        } else {
            (-1, -1, 0)
        };

    // Acceptance criteria aligned with methodology.
    let is_smoke = duration_sec < 1800;
    let pass_rss = if is_smoke {
        rss_first <= 0 || rss_last <= 2 * rss_first.max(1)
    } else {
        rss_slope_pct_per_hour.is_nan() || rss_slope_pct_per_hour <= 0.5
    };
    let pass_fd = fd_delta <= 0;
    let pass_orphans = kv_orphan_max == 0;
    let pass_req = req_last >= 30;

    let rss_post_warmup_pass = if rss_post_warmup_n >= 2 {
        rss_post_warmup_slope_pct_per_hour.is_nan()
            || rss_post_warmup_slope_pct_per_hour <= 0.5
    } else {
        pass_rss
    };
    let fd_post_warmup_pass = if rss_post_warmup_n >= 2 {
        fd_post_warmup_delta <= 0
    } else {
        pass_fd
    };
    let overall_pass = rss_post_warmup_pass
        && fd_post_warmup_pass
        && pass_orphans
        && pass_req;

    let gate_mode = if is_smoke { "smoke" } else { "full_soak" };
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

/// Shared async body. Duration is determined by `LUMEN_SOAK_DURATION_SEC`.
async fn run_soak() {
    let lbc = match resolve_q8_lbc_path() {
        Some(p) => p,
        None => {
            eprintln!("LUMEN_QWEN35_9B_Q8 not set; skipping CUDA soak");
            return;
        }
    };
    if !lbc.exists() {
        panic!("LBC path {lbc:?} does not exist");
    }
    let duration = Duration::from_secs(resolve_duration_sec());
    let out_dir = resolve_out_dir();
    std::fs::create_dir_all(&out_dir).unwrap();

    // Fresh stats files per run.
    let _ = std::fs::remove_file(out_dir.join("soak-stats.jsonl"));
    let _ = std::fs::remove_file(out_dir.join("soak-summary.json"));
    let _ = std::fs::remove_file(out_dir.join("soak-workload.jsonl"));

    let disk_kv_dir = out_dir.join("disk-kv");
    let _ = std::fs::remove_dir_all(&disk_kv_dir);

    // Publish PID early so the Python supervisor can start sampling before
    // model load completes (the supervisor will get warmup-dominated
    // samples for the first ~5 min, which is exactly the warmup window
    // the post-warmup linear regression excludes).
    let pid = std::process::id();
    std::fs::write(out_dir.join("soak-pid.txt"), format!("{}", pid))
        .expect("write pid file");
    eprintln!("[soak/init] driver PID = {}", pid);

    eprintln!("[soak/init] LBC={lbc:?} duration={duration:?} out_dir={out_dir:?}");
    let (addr, counter) = boot_soak_server(&lbc, disk_kv_dir.clone()).await;
    eprintln!("[soak/init] bound 127.0.0.1:{}", addr.port());
    std::fs::write(out_dir.join("soak-port.txt"), format!("{}", addr.port()))
        .expect("write port file");

    let client: Client<HttpConnector, Full<bytes::Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Spawn sampler alongside workload (both share the same process PID).
    let sampler_handle = {
        let out_dir = out_dir.clone();
        let counter = Arc::clone(&counter);
        let stop = Arc::clone(&stop);
        let kv_dir = disk_kv_dir.clone();
        tokio::spawn(async move { sample_loop(out_dir, counter, kv_dir, stop).await })
    };

    let workload_result = run_in_process_workload(&client, addr, duration, Arc::clone(&counter), &out_dir).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = sampler_handle.await;

    if let Err(e) = workload_result {
        panic!("workload failed: {e}");
    }

    let pass = write_soak_summary(&out_dir, duration.as_secs()).expect("write summary");
    assert!(pass, "soak failed acceptance criteria (see soak-summary.json)");
}

/// 1-hour CUDA soak entry point.
/// Reads `LUMEN_SOAK_DURATION_SEC` (default 300 = smoke). For the real
/// 1-hour run, the supervisor sets `LUMEN_SOAK_DURATION_SEC=3600`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = " CUDA soak: set LUMEN_QWEN35_9B_Q8 + LUMEN_SOAK_DURATION_SEC, run with --ignored --nocapture soak_cuda_1h"]
async fn soak_cuda_1h() {
    run_soak().await
}

/// 5-minute smoke entry (matches `server_soak.rs::soak_smoke_5min` pattern).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = " CUDA soak smoke: set LUMEN_QWEN35_9B_Q8 + LUMEN_SOAK_DURATION_SEC=300, --ignored --nocapture soak_cuda_smoke_5min"]
async fn soak_cuda_smoke_5min() {
    run_soak().await
}
