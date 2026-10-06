//! `lumen-server` standalone binary.
//!
//! Boots an OpenAI/Anthropic-compatible HTTP server on the Lumen runtime.
//! Follows the same backend-wiring pattern as
//! `crates/lumen-server/tests/server_soak.rs` (Metal) and
//! `crates/lumen-server/tests/server_soak_cuda.rs` (CUDA) so the production
//! bin and the integration/soak harnesses share one wiring template.
//!
//! See `README.md` "HTTP server" for endpoint documentation. The bin is
//! intentionally a thin embedder around the `lumen-server` library:
//! everything serious (routing, SSE, tool-call streaming, channel pool)
//! already lives in `crates/lumen-server/src/`.

// Match the rest of the production binaries on mimalloc.
#[cfg(not(feature = "system-allocator"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lumen_format::reader::LbcFile;
use lumen_format::QuantScheme;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::cache::WeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
#[cfg(feature = "cuda")]
use lumen_runtime::CudaBackend;
#[cfg(target_os = "macos")]
use lumen_runtime::MetalF32Backend;
use lumen_runtime::RuntimeConfig;

#[cfg(feature = "image")]
use lumen_server::build_router_with_images;
use lumen_server::{build_router, AllowedOrigins, EngineWorker, ModelInfo, Tokenize};

// ---------------------------------------------------------------------------
// CLI parsing — manual, matches `lumen-cli/src/run.rs` style (no `clap` dep).
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum BackendChoice {
    Auto,
    Cuda,
    Metal,
    Cpu,
}

#[derive(Debug)]
struct Args {
    model: String,
    quant: Option<String>,
    host: String,
    port: u16,
    /// `--allow-origin`: the web pages whose requests are served.
    origins: AllowedOrigins,
    context_len: usize,
    backend: BackendChoice,
    backend_device: usize,
    /// `--kv-precision`; `None` resolves from `LUMEN_KV_PRECISION`, then the backend default.
    kv_precision: Option<KvPrecision>,
    inbox_size: usize,
    log_level: String,
    /// Force the heavyweight `SyncWeightProvider` (pread-into-Vec, full CPU
    /// copy) instead of the default zero-copy `MmapWeightProvider` on the Metal
    /// path. Default `false` → mmap/no-copy residency (drops the ~10 GB
    /// redundant GPU private weight copy). The sync path is the guarded
    /// fallback; both are proven byte-identical on Metal by
    /// `metal_sync_mmap_argmax_parity_test`. CUDA/CPU always use sync.
    sync_provider: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: String::new(),
            quant: None,
            host: "127.0.0.1".to_string(),
            port: 8000,
            origins: AllowedOrigins::default(),
            context_len: 8192,
            backend: BackendChoice::Auto,
            backend_device: 0,
            kv_precision: None,
            inbox_size: 16,
            log_level: "info".to_string(),
            sync_provider: false,
        }
    }
}

fn print_help() {
    println!(
        "\
lumen-server - OpenAI / Anthropic-compatible HTTP server for Lumen

USAGE:
    lumen-server [OPTIONS] [MODEL:QUANT]
    lumen-server [OPTIONS] --model <MODEL> [--quant <Q>]
    lumen-server [OPTIONS] qwen-image
                           (images only, from `lumen pull qwen-image`; a --features image build)
    LUMEN_IMAGE_LBI=<dir> LUMEN_IMAGE_CKPT=<dir> lumen-server [OPTIONS]
                           (images only from a checkpoint converted by hand; a --features image build)

MODEL (positional or --model):
    MODEL:QUANT            Registry name with an optional quant tag, e.g.
                           `lumen-server qwen3.5-9b:q4_0`. Equivalent to
                           `--model qwen3.5-9b --quant q4_0`. A bare
                           `lumen-server qwen3.5-9b` uses the default quant.
    --model <ID|PATH>      Registry name (e.g. qwen3.5-9b, qwen3.5-moe-35b-a3b)
                           OR direct path to a .lbc file. Registry-name
                           resolution requires the LBC to be cached under
                           ~/.cache/lumen/ (run `lumen pull <name>` first).

OPTIONS:
    --quant <Q>            Quantization tag when --model is a registry name
                           (q8_0, q4_0, bf16; q4_k_m, q5_k_m for qwen3.8-27b,
                           served as stored on CUDA only — on Apple Silicon the
                           cached artifact is larger than the q8_0 one).
                           Default: q8_0
    --host <HOST>          Listen host. Default: 127.0.0.1
    --port <N>             Listen port. Default: 8000
    --allow-origin <ORIGIN>
                           Serve requests whose Origin header is ORIGIN, as
                           the client sends it (https://app.example.com,
                           chrome-extension://<id>, tauri://localhost);
                           repeatable. Web pages, browser extensions and
                           web-view apps send one and are refused unless
                           allowed; curl and the SDKs send none. Adds no
                           CORS headers.
    --context-len <N>      Max sequence length (KV cache size).
                           Capped at the model's native max_seq_len.
                           Default: 8192
    --backend <B>          cuda | metal | cpu
                           Default: auto (Metal on macOS, CUDA if available, else CPU)
    --backend-device <N>   GPU device ordinal (CUDA only). Default: 0
    --kv-precision <P>     KV cache storage: f16 | bf16 | f32. Default: LUMEN_KV_PRECISION,
                           else Metal f16, CUDA f32, CPU f32. On CUDA, f16 and bf16
                           halve the cache's bytes and its attention reads; bf16 is
                           CUDA only.
    --inbox-size <N>       Engine inbox capacity (in-flight job queue depth).
                           Default: 16
    --log-level <LEVEL>    error | warn | info | debug. Default: info
    --sync                 Use the legacy SyncWeightProvider (full CPU weight
                           copy) instead of the default zero-copy mmap provider
                           on Metal. Default is mmap/no-copy residency, which
                           drops the ~10 GB redundant GPU private weight copy.
                           --sync is the guarded fallback (both paths
                           are byte-identical on Metal). CUDA/CPU always use sync.
    -h, --help             Print this help
    -V, --version          Print version

ENVIRONMENT VARIABLES (image endpoint, `--features image` builds):
    LUMEN_IMAGE_LBI=<dir>  Converted image components (transformer.lbi,
                           vae.lbi, text_encoder.lbi). With LUMEN_IMAGE_CKPT
                           and no model, the server serves images only; a
                           model as well is refused (one model per process).
                           Not needed for `lumen-server qwen-image`, which
                           serves what `lumen pull qwen-image` cached.
    LUMEN_IMAGE_CKPT=<dir> The source checkpoint, for processor/vocab.json,
                           processor/merges.txt and processor/added_tokens.json.
    LUMEN_IMAGE_MODEL_ID=<id>
                           The model id the endpoint reports and accepts
                           (default Qwen-Image-2.1).
    LUMEN_IMAGE_DEVICE=cuda|gpu|cpu
                           Where generations run (default cuda). On CUDA the
                           transformer and VAE normally stay resident on the
                           device between generations.
    LUMEN_IMAGE_PIN_TEXT_ENCODER=1
                           Keep the text encoder's weights in page-locked host
                           memory (12.9 GiB, held for the server's lifetime) so
                           each generation loads them faster. CUDA only.

ENVIRONMENT VARIABLES (CUDA backend):
    LUMEN_CUDA_VERBOSE=1
                           Print the CUDA backend's start-up details: kernel
                           cache, memory use and the routes chosen by default.
                           Warnings and errors print without it.
    LUMEN_CUDA_DECODE_DELAY_US=<N>
                           Per-decode-step CPU sleep in microseconds, applied
                           after `cudaDeviceSynchronize` in the CUDA decode
                           paths. `lumen-server` defaults to `50` — an empirical
                           mitigation for decode non-determinism observed under
                           heavy MoE Q4 concurrency (not a root-caused fix); the
                           `lumen run` CLI defaults to `0`. Set `=0` to disable
                           on the server. Cost <=1% TPOT.

EXAMPLES:
    # Positional model:quant, auto-detect backend, default port 8000
    lumen-server qwen3.5-9b:q4_0

    # Explicit flags (equivalent to the above for q8_0)
    lumen-server --model qwen3.5-9b --quant q8_0

    # CUDA, custom port
    lumen-server qwen3.5-9b:q8_0 --backend cuda --port 9000

    # Direct file path
    lumen-server --model /path/to/qwen3-5-9b-Q8_0.lbc --port 8080

    # Images only (a --features image build): the model `lumen pull qwen-image` cached
    lumen-server qwen-image

    # Images only from a checkpoint converted by hand, beside a text server on 8000
    LUMEN_IMAGE_LBI=/path/to/lbi LUMEN_IMAGE_CKPT=/path/to/ckpt lumen-server --port 8001

ENDPOINTS (text server):
    GET  /v1/models                  OpenAI-style model list
    POST /v1/chat/completions        OpenAI chat completion (SSE optional)
    POST /v1/completions             OpenAI text completion (SSE optional)
    POST /v1/messages                Anthropic messages (SSE optional)
    POST /v1/messages/count_tokens   Anthropic token count

ENDPOINTS (image-only server):
    GET  /v1/models                  The image model
    POST /v1/images/generations      Text to image
"
    );
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut args = Args::default();
    // Tracks whether `--quant` was passed explicitly. A positional `model:quant`
    // tag only sets the quant when `--quant` was NOT given, so an explicit
    // `--quant` always wins regardless of argument order.
    let mut quant_explicit = false;
    let mut origins = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model" => {
                i += 1;
                args.model = raw.get(i).ok_or("--model requires a value")?.clone();
            }
            "--quant" => {
                i += 1;
                args.quant = Some(raw.get(i).ok_or("--quant requires a value")?.clone());
                quant_explicit = true;
            }
            "--host" => {
                i += 1;
                args.host = raw.get(i).ok_or("--host requires a value")?.clone();
            }
            "--allow-origin" => {
                i += 1;
                origins.push(raw.get(i).ok_or("--allow-origin requires a value")?.clone());
            }
            "--port" => {
                i += 1;
                let v = raw.get(i).ok_or("--port requires a value")?;
                args.port = v
                    .parse()
                    .map_err(|_| format!("--port must be u16, got {v}"))?;
            }
            "--context-len" => {
                i += 1;
                let v = raw.get(i).ok_or("--context-len requires a value")?;
                args.context_len = v
                    .parse()
                    .map_err(|_| format!("--context-len must be usize, got {v}"))?;
            }
            "--backend" => {
                i += 1;
                let v = raw.get(i).ok_or("--backend requires a value")?;
                args.backend = match v.to_ascii_lowercase().as_str() {
                    "cuda" => BackendChoice::Cuda,
                    "metal" => BackendChoice::Metal,
                    "cpu" => BackendChoice::Cpu,
                    "auto" => BackendChoice::Auto,
                    other => {
                        return Err(format!(
                            "--backend must be cuda|metal|cpu|auto, got {other}"
                        ))
                    }
                };
            }
            "--backend-device" => {
                i += 1;
                let v = raw.get(i).ok_or("--backend-device requires a value")?;
                args.backend_device = v
                    .parse()
                    .map_err(|_| format!("--backend-device must be usize, got {v}"))?;
            }
            "--inbox-size" => {
                i += 1;
                let v = raw.get(i).ok_or("--inbox-size requires a value")?;
                args.inbox_size = v
                    .parse()
                    .map_err(|_| format!("--inbox-size must be usize, got {v}"))?;
            }
            "--log-level" => {
                i += 1;
                args.log_level = raw.get(i).ok_or("--log-level requires a value")?.clone();
            }
            "--kv-precision" => {
                i += 1;
                let v = raw.get(i).ok_or("--kv-precision requires a value")?;
                args.kv_precision = Some(parse_kv_precision(v)?);
            }
            "--sync" => {
                args.sync_provider = true;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!(
                    "lumen-server {}",
                    option_env!("LUMEN_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
                );
                std::process::exit(0);
            }
            // Positional `model:quant`. Only a token that does NOT start with
            // `-` and only when `--model` has not yet been set is treated as the
            // model spec. This mirrors `lumen run <model>:<quant>`
            // (`lumen-cli/src/run.rs`): split on the LAST `:` — name before, tag
            // after — and a trailing `:` (empty tag) means "no quant". A second
            // bare positional, or a bare token after `--model`, leaves
            // `args.model` already set and so falls through to the error below.
            other if !other.starts_with('-') && args.model.is_empty() => {
                // A direct file path (contains `/`/`\`, or ends with `.lbc`/`.gguf`)
                // is taken verbatim and never split on an internal `:` — matching
                // `lumen run` and `resolve_model_path`'s path-first rule. Only a
                // registry-style `name:quant` token is split on the last `:`.
                let looks_like_path = other.contains('/')
                    || other.contains('\\')
                    || other.ends_with(".lbc")
                    || other.ends_with(".gguf");
                if looks_like_path {
                    args.model = other.to_string();
                } else {
                    match other.rfind(':') {
                        // `name:tag` with a non-empty tag → set both (tag only when
                        // `--quant` wasn't given, so an explicit `--quant` wins).
                        Some(p) if !other[p + 1..].is_empty() => {
                            args.model = other[..p].to_string();
                            if !quant_explicit {
                                args.quant = Some(other[p + 1..].to_string());
                            }
                        }
                        // Trailing `:` with an empty tag → name only (strip the `:`).
                        Some(p) => args.model = other[..p].to_string(),
                        // No `:` at all → the whole token is the name.
                        None => args.model = other.to_string(),
                    }
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    // Whether a model is required is decided in `run`, once the image endpoint's
    // configuration is known: a server with image config and no `--model` serves
    // images only. Here we only finish parsing.
    args.origins = AllowedOrigins::from_values(&origins)?;
    Ok(args)
}

// ---------------------------------------------------------------------------
// Model resolution: registry name -> ~/.cache/lumen/<key>-<QUANT>.lbc, or
// direct file path. Mirrors `lumen-cli/src/cache.rs::lbc_path`. We
// deliberately do NOT auto-download here — the production server should not
// kick off a ~10 GB HuggingFace fetch on its first request. Operators run
// `lumen pull <model>:<quant>` once at provisioning time.
// ---------------------------------------------------------------------------

fn cache_dir() -> PathBuf {
    if let Ok(val) = std::env::var("LUMEN_CACHE_DIR") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    // Mirror `lumen-cli/src/cache.rs::cache_dir`: on macOS the CLI uses
    // `dirs::cache_dir()` which resolves to `~/Library/Caches/lumen/`, not
    // `~/.cache/lumen/`, and elsewhere to `$XDG_CACHE_HOME/lumen/` when that
    // is set. The server must look in the SAME place or operators have to
    // symlink. Implemented inline (no `dirs` dep on the server crate) by
    // matching the platform manually.
    #[cfg(not(target_os = "macos"))]
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if xdg.starts_with('/') {
            return PathBuf::from(xdg).join("lumen");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        #[cfg(target_os = "macos")]
        {
            let macos = PathBuf::from(&home)
                .join("Library")
                .join("Caches")
                .join("lumen");
            if macos.is_dir() {
                return macos;
            }
        }
        return PathBuf::from(home).join(".cache").join("lumen");
    }
    PathBuf::from(".cache").join("lumen")
}

/// Resolve a `--model` value to a concrete `.lbc` path.
///
/// File-path heuristic matches `lumen-cli/src/run.rs::resolve_model_path`:
/// the value is treated as a path if it contains `/`, `\`, or ends with
/// `.lbc`/`.gguf`. Otherwise it is taken as a registry name or alias and a
/// cached LBC is looked up by `~/.cache/lumen/<key>-<QUANT>.lbc`, with the
/// key [`registry_key`] gives.
fn resolve_model_path(model: &str, quant_arg: Option<&str>) -> Result<PathBuf, String> {
    let looks_like_path = model.contains('/')
        || model.contains('\\')
        || model.ends_with(".lbc")
        || model.ends_with(".gguf");
    if looks_like_path {
        let path = PathBuf::from(model);
        if !path.exists() {
            return Err(format!("model file not found: {model}"));
        }
        return Ok(path);
    }

    // Registry-style name. Strip any `:quant` tag the user might have passed
    // alongside `--quant` (give --quant priority if both are set).
    let (name, tag_quant) = match model.rfind(':') {
        Some(p) if !model[p + 1..].is_empty() => (&model[..p], Some(&model[p + 1..])),
        _ => (model, None),
    };
    // The quant fallback must match model_registry.toml's [meta]
    // default_quant. lumen-cli prefers that default for a bare name when its
    // LBC is cached (falling back to a sole cached quant); the server always
    // uses it.
    let quant = quant_arg
        .map(str::to_owned)
        .or_else(|| tag_quant.map(|s| s.to_owned()))
        .unwrap_or_else(|| "q8_0".to_owned())
        .to_uppercase();

    let key = registry_key(name);

    // Lookup priority mirrors `lumen-cli/src/cache.rs::cached_lbc`:
    //   1. On macOS, prefer `<key>-<QUANT>-metal.lbc` (produced by
    //      `lumen convert --target metal`). Required for MoE Q4_0 whose
    //      K-quant FFN experts must be upcast to Q8_0 for Metal (the
    //      generic LBC otherwise emits gibberish on the Metal backend).
    //   2. Fall back to `<key>-<QUANT>.lbc`.
    let cache = cache_dir();
    #[cfg(target_os = "macos")]
    {
        let metal_path = cache.join(format!("{key}-{quant}-metal.lbc"));
        if metal_path.is_file() {
            return Ok(metal_path);
        }
    }
    let path = cache.join(format!("{key}-{quant}.lbc"));
    if !path.is_file() {
        return Err(format!(
            "model not cached: {}\n\
             Run `lumen pull {}:{}` (or `lumen pull {}:{}`) first, or pass \
             --model with a direct .lbc path.",
            path.display(),
            name,
            quant.to_lowercase(),
            key,
            quant.to_lowercase(),
        ));
    }
    Ok(path)
}

/// The cache key `lumen pull` stores a registry name or alias under
/// (`qwen3.5-moe` -> `qwen3-5-moe-35b-a3b`). A name the registry does not know
/// keeps the dot-to-dash normalization of its keys (`my.model` -> `my-model`).
fn registry_key(name: &str) -> String {
    lumen_cli::registry::load_registry()
        .resolve(name)
        .map_or_else(|| name.replace('.', "-"), |entry| entry.key.clone())
}

/// The registry key of `name` when it names the image model (a registry entry
/// with a checkpoint), which is served from `<cache>/<key>/` rather than from
/// an `.lbc`.
fn image_model_key(name: &str) -> Option<String> {
    lumen_cli::registry::load_registry()
        .resolve(name)
        .filter(|entry| entry.checkpoint.is_some())
        .map(|entry| entry.key.clone())
}

/// The model id reported on the wire (`/v1/models`, and echoed in responses).
/// A registry name is reported as given; a model given as a file path is reduced
/// to the file's name without its directory or extension, so a server launched
/// by path does not expose the operator's home directory and user name (a
/// request to `/v1/models` would otherwise return, e.g.,
/// `/Users/alice/models/qwen3-5-9b-Q8_0.lbc`). Path detection matches the
/// positional argument parser above.
fn model_public_id(model: &str) -> String {
    let looks_like_path = model.contains('/')
        || model.contains('\\')
        || model.ends_with(".lbc")
        || model.ends_with(".gguf");
    if !looks_like_path {
        return model.to_string();
    }
    std::path::Path::new(model)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| model.to_string())
}

// ---------------------------------------------------------------------------
// Tokenizer adapter — wraps `lumen_cli::tokenize::BpeTokenizer` to implement
// the `lumen_server::Tokenize` trait. Same shape as the soak harness adapter
// at `tests/server_soak.rs::BpeTokenizerAdapter`.
// ---------------------------------------------------------------------------

struct BpeTokenizerAdapter {
    inner: lumen_cli::tokenize::BpeTokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenize for BpeTokenizerAdapter {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode_incremental(&self, state: &mut Vec<u8>, token_id: u32) -> String {
        let frag_bytes = self.inner.decode_bytes(&[token_id]);
        state.extend_from_slice(&frag_bytes);
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

    fn decode_id_bytes(&self, token_id: u32) -> Vec<u8> {
        self.inner.decode_bytes(&[token_id])
    }

    fn apply_chat_template(&self, system: Option<&str>, user: &str) -> Option<String> {
        // Soak/bench harness: reasoning off (closed think tail), byte-identical
        // to the pre-reasoning-control template.
        Some(
            self.inner
                .apply_chat_template_with_system(user, system, false),
        )
    }

    fn chat_template(&self) -> Option<&str> {
        self.inner.chat_template()
    }

    fn eos_tokens(&self) -> Vec<u32> {
        self.eos_ids.clone()
    }
}

// ---------------------------------------------------------------------------
// Backend selection + wiring. The Metal and CUDA branches mirror
// `tests/server_soak.rs::boot_soak_server` and
// `tests/server_soak_cuda.rs::boot_soak_server` line-for-line so any wiring
// bug fixed in one place is fixed in the other.
// ---------------------------------------------------------------------------

fn select_backend(choice: BackendChoice) -> BackendChoice {
    match choice {
        BackendChoice::Auto => {
            #[cfg(target_os = "macos")]
            {
                BackendChoice::Metal
            }
            #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
            {
                if lumen_runtime::runtime_defaults::nvidia_gpu_present() {
                    BackendChoice::Cuda
                } else {
                    BackendChoice::Cpu
                }
            }
            #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
            {
                BackendChoice::Cpu
            }
        }
        other => other,
    }
}

/// Provider-agnostic view of the global tensors the backend wiring needs.
/// Borrowed from whichever concrete provider the server opened. `SyncWeightProvider`
/// and `MmapWeightProvider` expose the exact same public global fields; this
/// bundle lets `wire_global_tensors_and_raw` be written once for both.
struct WeightGlobals<'a> {
    embedding: &'a [f32],
    final_norm: &'a [f32],
    output_proj: &'a [f32],
    embedding_raw: &'a [u8],
    embedding_quant: QuantScheme,
    output_proj_raw: &'a [u8],
    output_proj_quant: QuantScheme,
    weight_tying: bool,
}

/// The server's weight provider — either the legacy full-copy `SyncWeightProvider`
/// or the zero-copy `MmapWeightProvider`. Both implement `WeightProvider`
/// (`Send + Sync`), so either can back the engine's `Arc<dyn WeightProvider>`.
/// The Metal default is `Mmap` (no-copy residency, ~10 GB lighter); `--sync`,
/// CUDA, and CPU use `Sync`. Both are proven byte-identical on the Metal
/// GPU-resident path by `metal_sync_mmap_argmax_parity_test`.
#[derive(Clone)]
enum ServerWeights {
    Sync(Arc<SyncWeightProvider>),
    Mmap(Arc<MmapWeightProvider>),
}

impl ServerWeights {
    fn lbc(&self) -> &LbcFile {
        match self {
            ServerWeights::Sync(p) => p.lbc(),
            ServerWeights::Mmap(p) => p.lbc(),
        }
    }

    fn globals(&self) -> WeightGlobals<'_> {
        match self {
            ServerWeights::Sync(p) => WeightGlobals {
                embedding: &p.embedding,
                final_norm: &p.final_norm,
                output_proj: &p.output_proj,
                embedding_raw: &p.embedding_raw,
                embedding_quant: p.embedding_quant,
                output_proj_raw: &p.output_proj_raw,
                output_proj_quant: p.output_proj_quant,
                weight_tying: p.weight_tying,
            },
            ServerWeights::Mmap(p) => WeightGlobals {
                embedding: &p.embedding,
                final_norm: &p.final_norm,
                output_proj: &p.output_proj,
                embedding_raw: &p.embedding_raw,
                embedding_quant: p.embedding_quant,
                output_proj_raw: &p.output_proj_raw,
                output_proj_quant: p.output_proj_quant,
                weight_tying: p.weight_tying,
            },
        }
    }

    /// Borrow as `&dyn WeightProvider` for `preload_weights` (the CUDA and Metal backends).
    #[cfg(any(feature = "cuda", target_os = "macos"))]
    fn as_dyn(&self) -> &dyn WeightProvider {
        match self {
            ServerWeights::Sync(p) => &**p,
            ServerWeights::Mmap(p) => &**p,
        }
    }

    /// Consume into the `Arc<dyn WeightProvider>` the engine worker owns.
    fn into_arc(self) -> Arc<dyn WeightProvider> {
        match self {
            ServerWeights::Sync(p) => p,
            ServerWeights::Mmap(p) => p,
        }
    }
}

/// Which raw global planes a backend takes as stored (the rest are handed over as the
/// provider's F32 dequant).
#[derive(Default, Clone, Copy)]
struct RawAcceptance {
    /// Drop the F32 copy of a plane the backend takes raw.
    skip_f32_when_raw: bool,
    q6k_head: bool,
    bf16_head: bool,
    nvfp4_head: bool,
    bf16_embedding: bool,
    kquant_embedding: bool,
}

fn wire_global_tensors_and_raw(
    backend: &mut dyn ComputeBackend,
    g: &WeightGlobals<'_>,
    accept: RawAcceptance,
) {
    let RawAcceptance {
        skip_f32_when_raw,
        q6k_head: accept_q6k_head,
        bf16_head: accept_bf16_head,
        nvfp4_head: accept_nvfp4_head,
        bf16_embedding: accept_bf16_embedding,
        kquant_embedding: accept_kquant_embedding,
    } = accept;
    // When a native-quant raw blob is present for a scheme the backend uploads
    // directly (Q8_0/Q4_0/F16), `init()` builds the GPU buffer straight from the
    // raw bytes and never needs the F32 dequant. Materializing that F32 here via
    // `.to_vec()` then allocates a large heap copy (~3.8 GB each for
    // embedding/output_proj on Qwen3.5-9B) that the backend holds resident (CUDA)
    // or frees unused at init (Metal, backend_impl.rs:67/102) — a multi-GB
    // boot-time memory spike either way.
    //
    // `skip_f32_when_raw` passes an empty Vec instead, but only where that is
    // runtime-validated: Metal (true, byte-identical, ~1.6 GB lower peak).
    // CUDA (false) awaits a hardware run — statically its CPU embed fallback
    // is unreachable after init(), so the cost is ~8 GB of unread host heap,
    // not a live dependency. The raw upload itself is gated only on
    // `*_has_raw`, so both backends always receive it.
    // BF16 embedding raw is CUDA-only for now: CUDA's bf16 embed path is
    // hardware-validated; Metal's bf16 embed pipelines are wired but the raw
    // path is not hardware-validated, and CPU's set_embedding_raw is a
    // no-op. Both keep the F32 dequant copy.
    // A K-quant embedding is gathered natively by CUDA from the stored planes
    // (`accept_kquant_embedding`); Metal and CPU have no K-quant gather.
    let embedding_has_raw = (matches!(
        g.embedding_quant,
        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16
    ) || (accept_bf16_embedding
        && g.embedding_quant == QuantScheme::Bf16)
        || (accept_kquant_embedding
            && matches!(
                g.embedding_quant,
                QuantScheme::Q4_K | QuantScheme::Q5_K | QuantScheme::Q6_K
            )))
        && !g.embedding_raw.is_empty();
    // Q6_K (source-fidelity head) is CUDA-only: the CUDA backend splits the
    // superblocks into dp4a planes; Metal/CPU have no Q6_K head kernel.
    // NVFP4 is CUDA-only too: the provider holds no F32 head for it, so the
    // raw plane is the backend's only copy of the head.
    // Bf16 matches the CLI allow-list (run.rs): Metal and CUDA serve a raw
    // BF16 head natively; leaving it out silently doubles the head's memory
    // via the F32 fallback and skips the native BF16 dispatch. The CPU
    // backend has no raw BF16 head path, so forwarding there would only
    // clone a multi-GB buffer to discard it.
    let output_proj_has_raw = (matches!(
        g.output_proj_quant,
        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16
    ) || (accept_bf16_head && g.output_proj_quant == QuantScheme::Bf16)
        || (accept_q6k_head && g.output_proj_quant == QuantScheme::Q6_K)
        || (accept_nvfp4_head && g.output_proj_quant == QuantScheme::Nvfp4))
        && !g.output_proj_raw.is_empty();
    backend.set_global_tensors(
        if skip_f32_when_raw && embedding_has_raw {
            Vec::new()
        } else {
            g.embedding.to_vec()
        },
        g.final_norm.to_vec(),
        if skip_f32_when_raw && output_proj_has_raw {
            Vec::new()
        } else {
            g.output_proj.to_vec()
        },
    );
    if embedding_has_raw {
        backend.set_embedding_raw(g.embedding_raw.to_vec(), g.embedding_quant);
    }
    if output_proj_has_raw {
        backend.set_output_proj_raw(g.output_proj_raw.to_vec(), g.output_proj_quant);
    }
    if g.weight_tying {
        backend.set_weight_tying(true);
    }
}

// ---------------------------------------------------------------------------
// Top-level boot: parse args, open LBC, build backend, spawn worker, serve.
// ---------------------------------------------------------------------------

async fn run(args: Args) -> Result<(), String> {
    // Read and validate the image endpoint's configuration before the text
    // model loads, so a misconfiguration costs seconds rather than a full
    // model load per attempt.
    #[cfg(feature = "fault-injection")]
    lumen_server::fault::validate()?;
    // The registry key of the image model when `--model` names it (`lumen-server
    // qwen-image`), whose files `lumen pull` put under `<cache>/<key>/`.
    let image_key = (!args.model.is_empty())
        .then(|| image_model_key(&args.model))
        .flatten();
    #[cfg(feature = "image")]
    let image_settings = image_settings_from_env()?;
    #[cfg(feature = "image")]
    let image_dirs = image_dirs_from_env()?;
    #[cfg(feature = "image")]
    if image_dirs.is_none() && image_key.is_none() {
        if let Some(name) = image_settings.set.first() {
            return Err(format!(
                "{name} is set but LUMEN_IMAGE_LBI and LUMEN_IMAGE_CKPT are not"
            ));
        }
    }
    #[cfg(not(feature = "image"))]
    for name in [
        "LUMEN_IMAGE_LBI",
        "LUMEN_IMAGE_CKPT",
        "LUMEN_IMAGE_MODEL_ID",
        "LUMEN_IMAGE_DEVICE",
        "LUMEN_IMAGE_PIN_TEXT_ENCODER",
    ] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "{name} is set but this lumen-server was built without the image endpoint \
                 (`--features image`)"
            ));
        }
    }
    #[cfg(not(feature = "image"))]
    if image_key.is_some() {
        return Err(format!(
            "{} makes images, and this lumen-server was built without the image endpoint \
             (`--features image`)",
            args.model
        ));
    }
    // One model per process. With the image endpoint configured and no `--model`, serve
    // images only: skip the text engine and serve just the image routes (plus `/v1/models`
    // for the image model). With `--model`, the text engine builds below — the image
    // endpoint cannot also run in that process. With neither, there is nothing to serve.
    #[cfg(feature = "image")]
    if args.model.is_empty() {
        let (lbi_dir, checkpoint_dir) = image_dirs.ok_or_else(|| {
            "a model is required: pass `MODEL:QUANT` or --model, or set LUMEN_IMAGE_LBI and \
             LUMEN_IMAGE_CKPT to serve images only (try --help)"
                .to_string()
        })?;
        let config = image_config(lbi_dir, checkpoint_dir, image_settings)?;
        let state = build_image_state(config)?;
        let app = build_router_with_images(state, args.origins.clone());
        return serve(app, &args.host, args.port).await;
    }
    // The image model by its registry name: what `lumen pull` cached, served the same way.
    #[cfg(feature = "image")]
    if let Some(key) = &image_key {
        if image_dirs.is_some() {
            return Err(format!(
                "{} names the image model and LUMEN_IMAGE_LBI and LUMEN_IMAGE_CKPT are set as \
                 well; pass one or the other",
                args.model
            ));
        }
        if let Some(quant) = &args.quant {
            return Err(format!(
                "{} comes in one form and takes no quantization (got {quant:?})",
                args.model
            ));
        }
        let checkpoint_dir = cache_dir().join(key);
        let lbi_dir = checkpoint_dir.join("lbi");
        if !lbi_dir.is_dir() {
            return Err(format!(
                "model not cached: {}\nRun `lumen pull {}` first.",
                lbi_dir.display(),
                args.model
            ));
        }
        let config = image_config(lbi_dir, checkpoint_dir, image_settings)?;
        let state = build_image_state(config)?;
        let app = build_router_with_images(state, args.origins.clone());
        return serve(app, &args.host, args.port).await;
    }
    // A text model and the image endpoint cannot share one process: the two exceed the
    // card, so they run as separate servers.
    #[cfg(feature = "image")]
    if image_dirs.is_some() {
        return Err(
            "a text model and the image endpoint cannot run in one process; start two \
                    servers — one with --model, one with LUMEN_IMAGE_LBI and LUMEN_IMAGE_CKPT \
                    and no --model"
                .to_string(),
        );
    }
    #[cfg(not(feature = "image"))]
    if args.model.is_empty() {
        return Err("a model is required: pass `MODEL:QUANT` or --model (try --help)".to_string());
    }

    let lbc_path = resolve_model_path(&args.model, args.quant.as_deref())?;
    eprintln!("[lumen-server] model: {}", lbc_path.display());

    // Open weights + extract tokenizer from the LBC's embedded tokenizer
    // section. Same pattern as `tests/server_soak.rs:289-317`.
    let lbc = lumen_format::reader::LbcFile::open(&lbc_path)
        .map_err(|e| format!("open LBC {lbc_path:?}: {e}"))?;

    // A scheme with no kernels is refused first: the refusal is decided from
    // what `LbcFile::open` parsed — the header, the index and the tokenizer
    // section — before the tokenizer is built and before any weight provider opens. The artifact parses,
    // so without this it would reach the provider and fail as a missing kernel or, worse, a misread plane.
    // Admission is backend-dependent: the planar schemes are served by the CUDA kernels only, and only
    // while the kill switch leaves them on; the switch is published the same way the CLI publishes it.
    // The backend is resolved here, before admission and the weight-provider choice, which both depend on
    // it. A backend this build cannot construct is refused for that reason first, with the message the
    // backend construction below gives, before any weight provider opens and before admission could name
    // the kill switch instead of the build.
    lumen_runtime::runtime_defaults::publish_cuda_nvfp4_admission();
    let backend_choice = select_backend(args.backend);
    if matches!(backend_choice, BackendChoice::Cuda) && !cfg!(feature = "cuda") {
        return Err("--backend cuda requires building with --features cuda".to_string());
    }
    if matches!(backend_choice, BackendChoice::Metal) && !cfg!(target_os = "macos") {
        return Err("--backend metal is only supported on macOS".to_string());
    }
    let admission_backend = match backend_choice {
        BackendChoice::Cuda => lumen_format::serving_rules::ServingBackend::Cuda,
        BackendChoice::Metal => lumen_format::serving_rules::ServingBackend::Metal,
        BackendChoice::Cpu | BackendChoice::Auto => {
            lumen_format::serving_rules::ServingBackend::Cpu
        }
    };
    if let Some(scheme) = lumen_format::serving_rules::unservable_scheme(&lbc, admission_backend) {
        return Err(lumen_format::serving_rules::no_serving_kernels_message(
            scheme,
            admission_backend,
        ));
    }

    let tok_section = lbc
        .tokenizer
        .as_ref()
        .ok_or_else(|| format!("LBC {lbc_path:?} has no embedded tokenizer section"))?
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
    let bpe = lumen_cli::tokenize::BpeTokenizer::from_tokenizer_data(&tok_data);
    let eos_ids = bpe.stop_token_ids.clone();
    let tokenizer: Arc<dyn Tokenize> = Arc::new(BpeTokenizerAdapter {
        inner: bpe,
        eos_ids,
    });

    eprintln!("[lumen-server] backend: {backend_choice:?}");

    // CtInt4G32 has CUDA kernels only; no other backend can serve the packed
    // planes. Check the lightweight header/index HERE — per-tensor slices,
    // not just the primary scheme — before the provider reads multi-GB
    // weight data, so an unsupported combination fails fast instead of
    // OOM-ing. A no-CUDA build fails here too, not after global expansion.
    if lbc.uses_quant(lumen_format::quantization::QuantScheme::CtInt4G32)
        && !(cfg!(feature = "cuda") && matches!(backend_choice, BackendChoice::Cuda))
    {
        return Err("this model (CtInt4G32) requires the CUDA backend \
                    (a lumen-server build with the `cuda` feature)"
            .into());
    }

    // Provider selection: the Metal GPU-resident path and CUDA NVFP4 both use
    // the zero-copy `MmapWeightProvider`.
    //
    // Metal engages the no-copy unified-buffer path (`LUMEN_METAL_MMAP_ONLY=1`),
    // avoiding the ~10 GB redundant CPU copy + GPU private weight copy that
    // `SyncWeightProvider` (pread-into-Vec, non-page-aligned) forces.
    //
    // CUDA NVFP4 uses it to skip a redundant per-layer host copy at preload:
    // `SyncWeightProvider` preads each layer into a freshly allocated host buffer
    // before the device copy, whereas serving the mapped bytes straight into the
    // htod copy skips that, so the one-time preload and cold startup are faster.
    // Both providers serve the same raw file bytes through `get_layer_raw` (the
    // same range + subtensor offsets, with no per-scheme dequant), so the bytes
    // reaching the GPU are identical: the Q8 case is unit-tested
    // (`get_layer_raw_is_byte_identical_across_sync_and_mmap_for_q8`) and NVFP4 is
    // identical by the same mechanism, verified end-to-end (generated images are
    // byte-identical).
    //
    // `--sync`, other CUDA quants, and CPU keep `SyncWeightProvider` (the guarded
    // fallback). Mmap and sync are proven byte-identical on the Metal path by
    // `metal_sync_mmap_argmax_parity_test`.
    let cuda_nvfp4 = matches!(backend_choice, BackendChoice::Cuda)
        && lbc.uses_quant(lumen_format::quantization::QuantScheme::Nvfp4);
    let use_mmap_provider =
        !args.sync_provider && (matches!(backend_choice, BackendChoice::Metal) || cuda_nvfp4);
    let provider: ServerWeights = if use_mmap_provider {
        // Enable Metal's no-copy unified-buffer residency path. Metal only: it is
        // that backend's zero-copy GPU-resident route (gpu_resident.rs probe),
        // whereas CUDA reads the mapped layers into ordinary htod copies. We set
        // the env BEFORE the backend constructor / `preload_weights` reads it. An
        // explicit operator override (e.g. `LUMEN_METAL_MMAP_ONLY=0`) still wins
        // because we only set it when unset.
        if matches!(backend_choice, BackendChoice::Metal)
            && std::env::var_os("LUMEN_METAL_MMAP_ONLY").is_none()
        {
            // SAFETY: single-threaded boot, before any worker thread is spawned.
            unsafe {
                std::env::set_var("LUMEN_METAL_MMAP_ONLY", "1");
            }
        }
        let mmap_config = MmapConfig {
            prefetch_window: 2,
            advise_sequential: true,
            release_with_dontneed: true,
        };
        eprintln!("[lumen-server] weights: mmap/no-copy (zero CPU copy; --sync to force legacy)");
        ServerWeights::Mmap(Arc::new(
            MmapWeightProvider::open(&lbc_path, mmap_config)
                .map_err(|e| format!("open weights (mmap) {lbc_path:?}: {e}"))?,
        ))
    } else {
        eprintln!("[lumen-server] weights: sync (full CPU copy)");
        ServerWeights::Sync(Arc::new(
            SyncWeightProvider::open(&lbc_path)
                .map_err(|e| format!("open weights (sync) {lbc_path:?}: {e}"))?,
        ))
    };

    // Plumb model-aware defaults into the runtime BEFORE
    // any backend constructor or kernel dispatch fires its first env-var
    // read. The two setters are idempotent and cheap (one atomic store
    // each); they let the operator run a BF16 dense LBC against
    // `lumen-server` with NO `LUMEN_CUDA_BF16_GEMMEX=1` and the runtime
    // still picks the right path for that cell. Operator explicit env vars
    // remain authoritative — model-aware defaults only apply when the env
    // is unset. F3 typo validator already ran in `main()` BEFORE this; the
    // CUDA backend constructor (`CudaBackend::new`) is invoked downstream
    // and is when the cached env-or-default helpers latch.
    lumen_runtime::runtime_defaults::set_path_is_server(true);
    lumen_runtime::runtime_defaults::set_model_dense_quant(provider.globals().output_proj_quant);
    // PRIMARY (bulk) quant — the body attn/FFN scheme, NOT output_proj:
    // output_proj is Q8_0 for BOTH 27B-q4 and 27B-q8, so only the primary
    // scheme tells them apart.
    lumen_runtime::runtime_defaults::set_model_primary_quant(
        provider.lbc().header.quantization.scheme,
    );
    // Model-size discriminator for the per-class defaults (9B = 32 layers,
    // 27B = 64).
    lumen_runtime::runtime_defaults::set_model_block_count(
        provider.lbc().header.hyperparams.num_layers,
    );
    // Feed the MoE flag alongside the dense-quant hint so
    // the Q8-only flag resolvers (`q8_split_default` and the chain that
    // delegates to it) stay OFF on MoE 35B-A3B. Without this gate, MoE
    // Q8 MoE emits 1 valid token then 159 `[PAD248319]` per prompt.
    lumen_runtime::runtime_defaults::set_model_is_moe(
        provider.lbc().header.hyperparams.num_experts.unwrap_or(0) > 0,
    );

    let hyperparams = provider.lbc().header.hyperparams;
    let model_max_seq_len = hyperparams.max_seq_len as usize;
    let context_length = std::cmp::min(args.context_len, model_max_seq_len);

    // Right-size the KV cache to `--context-len`.
    //
    // The backend's `init` allocates the GPU KV cache for
    // `hyperparams.max_seq_len` tokens. Mirroring the CLI's pattern
    // (`effective_max_seq_len` in `lumen-cli/src/run.rs`), we pass a capped
    // hyperparams snapshot so the backend honours `--context-len` instead of
    // the model's native max_seq_len. Without this cap the server reserves
    // KV for the native 262144 context regardless of `--context-len` (9B
    // Q8: ~68.7 GB → premature prefill OOM at ~6.4k tokens).
    // `RuntimeConfig` already carries `max_seq_len: context_length` for the
    // prompt-length guard; this aligns the actual KV allocation with it.
    let mut hyperparams_capped = hyperparams;
    hyperparams_capped.max_seq_len = context_length as u32;

    if context_length < model_max_seq_len {
        eprintln!(
            "[lumen-server] context_length: {context_length} (KV cache sized to \
             this; model native max_seq_len is {model_max_seq_len})"
        );
    } else {
        eprintln!("[lumen-server] context_length: {context_length}");
    }

    // Build the concrete backend, wire global tensors + raw quantized blobs,
    // call init(), then preload_weights() (required for GPU-resident upload
    // on Metal AND CUDA; `tests/server_soak.rs:357-366` documents the requirement).

    let (backend, kv_precision): (Box<dyn ComputeBackend>, KvPrecision) = match backend_choice {
        BackendChoice::Metal => {
            #[cfg(target_os = "macos")]
            {
                let mut metal = MetalF32Backend::new()
                    .map_err(|e| format!("Metal backend unavailable: {e}"))?;
                // Metal: skip the unused F32 dequant when the native-quant raw
                // is present — runtime-validated byte-identical (see fn doc).
                wire_global_tensors_and_raw(
                    &mut metal,
                    &provider.globals(),
                    RawAcceptance {
                        skip_f32_when_raw: true,
                        bf16_head: true,
                        ..RawAcceptance::default()
                    },
                );
                metal
                    .init(&hyperparams_capped)
                    .map_err(|e| format!("Metal init: {e}"))?;
                metal
                    .preload_weights(provider.as_dyn())
                    .map_err(|e| format!("Metal preload_weights: {e}"))?;
                // Metal holds F16 only; a requested F32 or BF16 is refused by
                // validate_kv_precision at the engine.
                (
                    Box::new(metal),
                    resolve_kv_precision(args.kv_precision, KvPrecision::F16),
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("--backend metal is only supported on macOS".to_string());
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let kv_precision = resolve_kv_precision(args.kv_precision, KvPrecision::F32);
                let mut cuda = CudaBackend::new(args.backend_device).map_err(|e| {
                    format!(
                        "CUDA backend unavailable (device {}): {e}",
                        args.backend_device
                    )
                })?;
                // The store is chosen before init(): the backend compiles the store's
                // kernels as a group and allocates every cache in it.
                cuda.set_kv_precision(kv_precision)
                    .map_err(|e| format!("CUDA KV precision: {e}"))?;
                // Skip the unused F32 dequant of a plane CUDA takes raw: the raw plane
                // is the backend's copy and the CPU embed fallback is statically
                // unreachable after init(), so the F32 would be only ~8 GB of unread
                // host heap. Runtime-validated byte-identical (as Metal already relies on).
                wire_global_tensors_and_raw(
                    &mut cuda,
                    &provider.globals(),
                    RawAcceptance {
                        skip_f32_when_raw: true,
                        q6k_head: true,
                        bf16_head: true,
                        nvfp4_head: true,
                        bf16_embedding: true,
                        kquant_embedding: true,
                    },
                );
                cuda.init(&hyperparams_capped)
                    .map_err(|e| format!("CUDA init: {e}"))?;
                cuda.preload_weights(provider.as_dyn())
                    .map_err(|e| format!("CUDA preload_weights: {e}"))?;
                (Box::new(cuda), kv_precision)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = args.backend_device;
                return Err("--backend cuda requires building with --features cuda".to_string());
            }
        }
        BackendChoice::Cpu => {
            let mut cpu = NaiveF32Backend::new();
            // CPU: keep the F32 dequant (skip=false) — the CPU backend reads it
            // directly and does not build a GPU buffer from the raw.
            wire_global_tensors_and_raw(&mut cpu, &provider.globals(), RawAcceptance::default());
            cpu.init(&hyperparams_capped)
                .map_err(|e| format!("CPU init: {e}"))?;
            // The CPU cache stores F32 or F16; the engine refuses any other precision.
            (
                Box::new(cpu),
                resolve_kv_precision(args.kv_precision, KvPrecision::F32),
            )
        }
        BackendChoice::Auto => {
            return Err("internal: select_backend should have resolved Auto".to_string());
        }
    };

    let runtime_cfg = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision,
        max_seq_len: context_length,
        collect_per_layer_timings: false,
    };
    let model_info = ModelInfo {
        id: model_public_id(&args.model),
        owned_by: "lumen".to_string(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        context_length,
    };

    // Spawn the engine worker. The handle is what the router consumes.
    // Pass the capped hyperparams so the worker's session sees the same
    // `max_seq_len` the backend KV was sized for (the engine only reads
    // `vocab_size`/`num_layers` from this, and `Session` sizes its KV from
    // `RuntimeConfig.max_seq_len`, but passing capped keeps the snapshot
    // internally consistent — no native-262144 value leaks downstream).
    let handle = EngineWorker::spawn(
        runtime_cfg,
        hyperparams_capped,
        backend,
        provider.into_arc(),
        tokenizer,
        model_info,
        args.inbox_size,
    );

    // One model per process: the image-only path returned earlier, and a text model with
    // image config was refused above, so by here the server is text-only.
    let app = build_router(handle, args.origins.clone());

    // `log_level` is captured for future structured-log wiring; today the bin logs via
    // `eprintln!` only, so we acknowledge the value to avoid an unused-field warning.
    let _ = &args.log_level;
    serve(app, &args.host, args.port).await
}

/// Bind the listener and serve `app` until shutdown; the device is released when the
/// process exits. Shutdown is graceful on Ctrl-C (SIGINT): in-flight requests finish and
/// any running image generation is asked to stop. Shared by the text-only and
/// image-only serving paths.
async fn serve(app: axum::Router, host: &str, port: u16) -> Result<(), String> {
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| format!("bind {bind_addr}: {e}"))?;
    let local = listener
        .local_addr()
        .map_err(|e| format!("local_addr: {e}"))?;
    eprintln!("[lumen-server] listening on http://{local}");
    eprintln!("[lumen-server] try: curl http://{local}/v1/models");
    eprintln!("[lumen-server] (Ctrl-C to stop)");
    let shutdown = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            eprintln!("[lumen-server] ctrl_c watcher failed: {e}");
        }
        eprintln!("[lumen-server] shutdown requested");
        #[cfg(feature = "image")]
        lumen_server::router_image::request_shutdown();
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| format!("axum serve: {e}"))?;
    eprintln!("[lumen-server] stopped");
    Ok(())
}

/// Open the image endpoint's model containers and build its router state. On the GPU
/// (`config.use_gpu`) the transformer and VAE are loaded resident and stay on the device
/// across generations; on the CPU each generation runs on the host.
#[cfg(feature = "image")]
fn build_image_state(
    config: lumen_server::router_image::ImageConfig,
) -> Result<Arc<lumen_server::router_image::ImageState>, String> {
    let resident = if config.use_gpu {
        let paths = lumen_image::pipeline::PipelinePaths::from_roots(
            &config.lbi_dir,
            &config.checkpoint_dir,
        );
        let mut sources = lumen_image::pipeline::GpuSources::open(&paths)
            .map_err(|e| format!("the image endpoint cannot open its model: {e}"))?;
        if config.pin_text_encoder {
            sources.pin_text_encoder().map_err(|e| {
                format!(
                    "LUMEN_IMAGE_PIN_TEXT_ENCODER=1, but the text encoder cannot be \
                     page-locked: {e}"
                )
            })?;
        }
        Some(std::sync::Mutex::new(
            lumen_image::pipeline::GpuResident::load(sources).map_err(|e| {
                format!(
                    "the image endpoint cannot keep its transformer and VAE loaded \
                     on CUDA device {}: {e}",
                    lumen_image::pipeline::GPU_DEVICE
                )
            })?,
        ))
    } else {
        None
    };
    eprintln!(
        "[lumen-server] /v1/images/generations enabled: model {} on {} from {} + {}{}{}",
        config.model_id,
        if config.use_gpu { "cuda" } else { "cpu" },
        config.lbi_dir.display(),
        config.checkpoint_dir.display(),
        if config.use_gpu {
            "; the transformer and VAE stay loaded between generations"
        } else {
            ""
        },
        if config.pin_text_encoder {
            "; the text encoder loads from page-locked host memory"
        } else {
            ""
        },
    );
    Ok(Arc::new(lumen_server::router_image::ImageState {
        config,
        resident,
    }))
}

/// The image endpoint's settings from `LUMEN_IMAGE_MODEL_ID`, `LUMEN_IMAGE_DEVICE`
/// and `LUMEN_IMAGE_PIN_TEXT_ENCODER`, with the names of those that are set. An
/// empty or unknown value is refused rather than silently defaulted.
#[cfg(feature = "image")]
struct ImageSettings {
    model_id: String,
    use_gpu: bool,
    pin_text_encoder: bool,
    set: Vec<&'static str>,
}

/// One `LUMEN_IMAGE_*` variable as set, with paths untouched (a directory name
/// may end in a space); only the enumerated values are trimmed at their match.
#[cfg(feature = "image")]
fn image_var(name: &'static str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) if v.trim().is_empty() => Err(format!("{name} is set but empty")),
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

#[cfg(feature = "image")]
fn image_settings_from_env() -> Result<ImageSettings, String> {
    let mut set = Vec::new();
    let mut var = |name: &'static str| -> Result<Option<String>, String> {
        let value = image_var(name)?;
        if value.is_some() {
            set.push(name);
        }
        Ok(value)
    };
    let model_id = var("LUMEN_IMAGE_MODEL_ID")?
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "Qwen-Image-2.1".to_string());
    let use_gpu = match var("LUMEN_IMAGE_DEVICE")?.as_deref().map(str::trim) {
        None | Some("cuda") | Some("gpu") => true,
        Some("cpu") => false,
        Some(other) => {
            return Err(format!(
                "LUMEN_IMAGE_DEVICE={other:?} is not one of cuda, gpu, cpu"
            ))
        }
    };
    let pin_text_encoder = match var("LUMEN_IMAGE_PIN_TEXT_ENCODER")?
        .as_deref()
        .map(str::trim)
    {
        None | Some("0") => false,
        Some("1") if use_gpu => true,
        Some("1") => return Err(
            "LUMEN_IMAGE_PIN_TEXT_ENCODER=1 applies to the CUDA path, not LUMEN_IMAGE_DEVICE=cpu"
                .to_string(),
        ),
        Some(other) => {
            return Err(format!(
                "LUMEN_IMAGE_PIN_TEXT_ENCODER={other:?} is not 0 or 1"
            ))
        }
    };
    Ok(ImageSettings {
        model_id,
        use_gpu,
        pin_text_encoder,
        set,
    })
}

/// The converted components' directory and the checkpoint directory from
/// `LUMEN_IMAGE_LBI` and `LUMEN_IMAGE_CKPT`, which are set together or not at all.
#[cfg(feature = "image")]
fn image_dirs_from_env() -> Result<Option<(std::path::PathBuf, std::path::PathBuf)>, String> {
    match (
        image_var("LUMEN_IMAGE_LBI")?,
        image_var("LUMEN_IMAGE_CKPT")?,
    ) {
        (None, None) => Ok(None),
        (Some(lbi), Some(ckpt)) => Ok(Some((lbi.into(), ckpt.into()))),
        (Some(_), None) => Err("LUMEN_IMAGE_LBI is set but LUMEN_IMAGE_CKPT is not".into()),
        (None, Some(_)) => Err("LUMEN_IMAGE_CKPT is set but LUMEN_IMAGE_LBI is not".into()),
    }
}

/// The image endpoint's configuration for a converted checkpoint: the CUDA
/// device is required when the endpoint runs on it, and the checkpoint's files
/// are opened and checked once, the way a generation opens them.
#[cfg(feature = "image")]
fn image_config(
    lbi_dir: std::path::PathBuf,
    checkpoint_dir: std::path::PathBuf,
    settings: ImageSettings,
) -> Result<lumen_server::router_image::ImageConfig, String> {
    if settings.use_gpu {
        match lumen_runtime::cuda::ffi::device_count() {
            Ok(0) => {
                return Err(
                    "the image endpoint runs on CUDA (LUMEN_IMAGE_DEVICE) but no CUDA device \
                     is available; set LUMEN_IMAGE_DEVICE=cpu for the CPU reference"
                        .to_string(),
                )
            }
            Ok(_) => {}
            Err(e) => {
                return Err(format!(
                    "the image endpoint runs on CUDA (LUMEN_IMAGE_DEVICE) but CUDA failed to initialise: {e}"
                ))
            }
        }
    }
    lumen_image::pipeline::PipelinePaths::from_roots(&lbi_dir, &checkpoint_dir)
        .check(settings.use_gpu)
        .map_err(|e| {
            format!(
                "the image endpoint cannot start: {e} ({} must hold \
                 transformer.lbi, vae.lbi and text_encoder.lbi; {} must hold \
                 processor/vocab.json, merges.txt and added_tokens.json{})",
                lbi_dir.display(),
                checkpoint_dir.display(),
                if settings.use_gpu {
                    "; the CUDA device must have the memory a 2048x2048 generation uses"
                } else {
                    ""
                }
            )
        })?;
    Ok(lumen_server::router_image::ImageConfig {
        lbi_dir,
        checkpoint_dir,
        model_id: settings.model_id,
        use_gpu: settings.use_gpu,
        pin_text_encoder: settings.pin_text_encoder,
    })
}

/// `--kv-precision` values: `f16` / `bf16` / `f32` (case-insensitive).
fn parse_kv_precision(value: &str) -> Result<KvPrecision, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "half" => Ok(KvPrecision::F16),
        "bf16" | "bfloat16" => Ok(KvPrecision::Bf16),
        "f32" | "fp32" | "float" => Ok(KvPrecision::F32),
        other => Err(format!(
            "--kv-precision must be f16, bf16 or f32 (got '{other}')"
        )),
    }
}

/// The KV cache storage: the flag, else `LUMEN_KV_PRECISION`, else the
/// backend's default. A malformed variable is said once and ignored.
fn resolve_kv_precision(flag: Option<KvPrecision>, default: KvPrecision) -> KvPrecision {
    if let Some(p) = flag {
        return p;
    }
    if let Ok(raw) = std::env::var("LUMEN_KV_PRECISION") {
        match parse_kv_precision(&raw) {
            Ok(p) => return p,
            Err(e) => eprintln!("[lumen-server] LUMEN_KV_PRECISION ignored: {e}"),
        }
    }
    default
}

fn main() -> ExitCode {
    // run the env-var typo validator BEFORE any other env
    // read in the binary. Two passes:
    //  (a) names that start with `LUMEN_` but are NOT in the allowlist
    //      (mis-spelled suffix, e.g. `LUMEN_CUDA_GDN_REGISTER_RESIDENT`
    //      with the trailing `T` missing); and
    //  (b) names that do NOT start with `LUMEN_` but suffix-match a
    //      canonical `LUMEN_CUDA_*` / `LUMEN_METAL_*` / `LUMEN_SERVER_*`
    //      entry (missing prefix, e.g. `GDN_REGISTER_RESIDENT` — the literal
    //      bug).
    // Each warning surfaces the closest canonical name so the operator
    // sees "did you mean LUMEN_CUDA_GDN_REGISTER_RESIDENT?" at a glance.
    let _warnings = lumen_runtime::runtime_defaults::validate_lumen_env_vars();
    lumen_runtime::runtime_defaults::mark_validator_ran();
    // A name an earlier release read and this one does not is an error, not a
    // warning: the configuration the operator asked for would silently not apply.
    let removed = lumen_runtime::runtime_defaults::removed_lumen_env_vars_set();
    if !removed.is_empty() {
        for line in &removed {
            eprintln!("[lumen] ERROR: {line}");
        }
        eprintln!("[lumen] refusing to start with a removed env var set");
        std::process::exit(2);
    }
    lumen_runtime::runtime_defaults::set_build_identity(
        option_env!("LUMEN_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION")),
    );

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("lumen-server: {e}");
            eprintln!("Run `lumen-server --help` for usage.");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("lumen-server: tokio runtime build failed: {e}");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lumen-server: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    /// Build the `&[String]` `parse_args` expects from string literals.
    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn registry_names_resolve_to_their_cache_keys() {
        assert_eq!(super::registry_key("qwen3.5-moe"), "qwen3-5-moe-35b-a3b");
        assert_eq!(
            super::registry_key("qwen3.5-moe-35b-a3b"),
            "qwen3-5-moe-35b-a3b"
        );
        assert_eq!(super::registry_key("qwen3.8-27b"), "qwen3-8-27b");
        assert_eq!(super::registry_key("qwen3.5-9b"), "qwen3-5-9b");
        assert_eq!(super::registry_key("qwen3-5-9b"), "qwen3-5-9b");
        assert_eq!(super::registry_key("my.model"), "my-model");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_cache_dir_honours_xdg_cache_home_like_the_cli() {
        // Process-global variables: set, read, restore before asserting.
        let cache = std::env::var("LUMEN_CACHE_DIR").ok();
        let xdg = std::env::var("XDG_CACHE_HOME").ok();
        std::env::remove_var("LUMEN_CACHE_DIR");
        std::env::set_var("XDG_CACHE_HOME", "/var/tmp/lumen-xdg-test");
        let with_xdg = super::cache_dir();
        std::env::set_var("XDG_CACHE_HOME", "relative/is/ignored");
        let relative = super::cache_dir();
        match xdg {
            Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
            None => std::env::remove_var("XDG_CACHE_HOME"),
        }
        if let Some(v) = cache {
            std::env::set_var("LUMEN_CACHE_DIR", v);
        }
        assert_eq!(
            with_xdg,
            std::path::PathBuf::from("/var/tmp/lumen-xdg-test/lumen")
        );
        assert!(relative.ends_with(".cache/lumen"), "{}", relative.display());
    }

    #[test]
    fn only_the_image_model_has_an_image_key() {
        assert_eq!(
            super::image_model_key("qwen-image").as_deref(),
            Some("qwen-image-2-1")
        );
        assert_eq!(
            super::image_model_key("qwen-image-2-1").as_deref(),
            Some("qwen-image-2-1")
        );
        for text in ["qwen3.5-9b", "qwen3.8-27b", "qwen3.5-moe", "my.model"] {
            assert_eq!(super::image_model_key(text), None, "{text}");
        }
    }

    #[test]
    fn model_public_id_reports_a_name_never_a_path() {
        // A registry name is reported verbatim — its dots are not an extension.
        assert_eq!(super::model_public_id("qwen3.8-27b"), "qwen3.8-27b");
        assert_eq!(super::model_public_id("qwen3.5-9b"), "qwen3.5-9b");
        assert_eq!(
            super::model_public_id("qwen3-5-moe-35b-a3b"),
            "qwen3-5-moe-35b-a3b"
        );
        // A path is reduced to the file's name — no directory, no extension.
        assert_eq!(
            super::model_public_id("/Users/alice/models/qwen3-5-9b-Q8_0.lbc"),
            "qwen3-5-9b-Q8_0"
        );
        assert_eq!(super::model_public_id("/home/bob/m.gguf"), "m");
        assert_eq!(super::model_public_id("./sub/dir/qwen.lbc"), "qwen");
        // A bare filename has no directory to expose but still drops the extension.
        assert_eq!(super::model_public_id("model.lbc"), "model");
    }

    #[test]
    fn allow_origin_takes_origins_and_refuses_anything_else() {
        let origins = [
            "m",
            "--allow-origin",
            "https://a.example",
            "--allow-origin",
            "http://b.example:8080",
        ];
        assert!(parse_args(&argv(&origins)).is_ok());
        let err = parse_args(&argv(&["m", "--allow-origin", "https://a.example/"])).unwrap_err();
        assert!(err.starts_with("--allow-origin takes one origin"), "{err}");
        let err = parse_args(&argv(&["m", "--allow-origin"])).unwrap_err();
        assert_eq!(err, "--allow-origin requires a value");
    }

    #[test]
    fn kv_precision_flag_accepts_bf16() {
        let a = parse_args(&argv(&["m", "--kv-precision", "bf16"])).expect("parse");
        assert_eq!(a.kv_precision, Some(lumen_runtime::kv::KvPrecision::Bf16));
        assert!(parse_args(&argv(&["m", "--kv-precision", "int8"])).is_err());
    }

    #[test]
    fn positional_model_quant_sets_both() {
        // `lumen-server qwen3.5-9b:q4_0` ⇒ model=qwen3.5-9b, quant=Some("q4_0").
        let a = parse_args(&argv(&["qwen3.5-9b:q4_0"])).expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant.as_deref(), Some("q4_0"));
    }

    #[test]
    fn positional_path_with_colon_is_not_split() {
        // A direct file path containing a `:` must be taken verbatim, NOT split
        // into a bogus model/quant — matching `lumen run`'s path-first rule.
        let a = parse_args(&argv(&["./ckpt:final/model.lbc"])).expect("parse");
        assert_eq!(a.model, "./ckpt:final/model.lbc");
        assert_eq!(a.quant, None);
        // The `.lbc` extension alone (no slash) also marks it a path.
        let b = parse_args(&argv(&["weird:name.lbc"])).expect("parse");
        assert_eq!(b.model, "weird:name.lbc");
        assert_eq!(b.quant, None);
    }

    #[test]
    fn bare_positional_sets_model_no_quant() {
        // `lumen-server qwen3.5-9b` ⇒ model set, quant defaults (None here;
        // resolve_model_path fills q8_0 downstream).
        let a = parse_args(&argv(&["qwen3.5-9b"])).expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant, None);
    }

    #[test]
    fn trailing_colon_means_no_quant() {
        // Split on the LAST `:`; a trailing `:` (empty tag) ⇒ just the name.
        let a = parse_args(&argv(&["qwen3.5-9b:"])).expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant, None);
    }

    #[test]
    fn explicit_quant_overrides_positional_tag_either_order() {
        // --quant after the positional wins.
        let a = parse_args(&argv(&["qwen3.5-9b:q4_0", "--quant", "q8_0"])).expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant.as_deref(), Some("q8_0"));
        // --quant before the positional also wins (no clobber).
        let b = parse_args(&argv(&["--quant", "q8_0", "qwen3.5-9b:q4_0"])).expect("parse");
        assert_eq!(b.model, "qwen3.5-9b");
        assert_eq!(b.quant.as_deref(), Some("q8_0"));
    }

    #[test]
    fn positional_then_flags_still_parse() {
        // Positional spec composes with other flags.
        let a = parse_args(&argv(&[
            "qwen3.5-9b:q8_0",
            "--backend",
            "cuda",
            "--port",
            "9000",
        ]))
        .expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant.as_deref(), Some("q8_0"));
        assert_eq!(a.port, 9000);
    }

    #[test]
    fn explicit_flags_still_work() {
        // Backward compatibility: --model/--quant unchanged.
        let a = parse_args(&argv(&["--model", "qwen3.5-9b", "--quant", "q8_0"])).expect("parse");
        assert_eq!(a.model, "qwen3.5-9b");
        assert_eq!(a.quant.as_deref(), Some("q8_0"));
    }

    #[test]
    fn last_colon_split_keeps_namespaced_names() {
        // rfind(':') splits on the LAST colon, so a name containing a colon
        // keeps everything before the final `:` as the model.
        let a = parse_args(&argv(&["org:model:q4_0"])).expect("parse");
        assert_eq!(a.model, "org:model");
        assert_eq!(a.quant.as_deref(), Some("q4_0"));
    }

    #[test]
    fn second_bare_positional_is_error() {
        // A second bare token (model already set) is rejected.
        let e = parse_args(&argv(&["qwen3.5-9b:q4_0", "qwen3.8-27b"]));
        assert!(e.is_err(), "expected error, got {e:?}");
    }

    #[test]
    fn bare_token_after_model_flag_is_error() {
        // `--model X` then a bare positional is rejected (model already set).
        let e = parse_args(&argv(&["--model", "qwen3.5-9b", "qwen3.8-27b"]));
        assert!(e.is_err(), "expected error, got {e:?}");
    }

    #[test]
    fn unknown_flag_is_still_error() {
        // A leading-dash unknown flag is NOT treated as a positional.
        let e = parse_args(&argv(&["--bogus"]));
        assert!(e.is_err(), "expected error, got {e:?}");
    }

    #[test]
    fn no_model_parses_ok_requirement_deferred_to_run() {
        // The model requirement moved out of parsing into `run`: with the image endpoint
        // configured and no `--model`, the server serves images only, so whether a model
        // is required depends on the image config that `run` reads. Parsing an empty argv
        // now succeeds with an empty model; `run` turns that into an error only when no
        // image endpoint is configured (validated end-to-end against the built server).
        let args = parse_args(&argv(&[])).expect("empty argv now parses");
        assert!(
            args.model.is_empty(),
            "expected an empty model, got {:?}",
            args.model
        );
    }

    /// Records the output head the wiring hands over.
    #[cfg(feature = "cuda")]
    #[derive(Default)]
    struct HeadRecorder {
        output_proj_len: usize,
        output_proj_raw: Option<(usize, lumen_format::QuantScheme)>,
    }

    #[cfg(feature = "cuda")]
    impl lumen_runtime::compute::ComputeBackend for HeadRecorder {
        fn init(
            &mut self,
            _: &lumen_format::ModelHyperparams,
        ) -> Result<(), lumen_runtime::RuntimeError> {
            unreachable!()
        }
        fn compute_layer(
            &self,
            _: usize,
            _: &mut lumen_runtime::compute::ActivationBuffer,
            _: &lumen_runtime::weight::cache::LayerView,
            _: Option<&mut lumen_runtime::kv::KvCacheView>,
            _: usize,
        ) -> Result<(), lumen_runtime::RuntimeError> {
            unreachable!()
        }
        fn compute_final(
            &self,
            _: &lumen_runtime::compute::ActivationBuffer,
        ) -> Result<lumen_runtime::compute::Logits, lumen_runtime::RuntimeError> {
            unreachable!()
        }
        fn embed_token(
            &self,
            _: u32,
        ) -> Result<lumen_runtime::compute::ActivationBuffer, lumen_runtime::RuntimeError> {
            unreachable!()
        }
        fn set_global_tensors(&mut self, _: Vec<f32>, _: Vec<f32>, output_proj: Vec<f32>) {
            self.output_proj_len = output_proj.len();
        }
        fn set_output_proj_raw(&mut self, raw: Vec<u8>, quant: lumen_format::QuantScheme) {
            self.output_proj_raw = Some((raw.len(), quant));
        }
    }

    /// The provider holds no F32 copy of an NVFP4 head, so the raw plane must
    /// reach CUDA or its init finds no output projection at all.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_takes_an_nvfp4_head_raw() {
        use lumen_format::QuantScheme;
        let raw = vec![0u8; 1156];
        let globals = super::WeightGlobals {
            embedding: &[0.0; 64],
            final_norm: &[1.0; 8],
            output_proj: &[],
            embedding_raw: &[],
            embedding_quant: QuantScheme::F32,
            output_proj_raw: &raw,
            output_proj_quant: QuantScheme::Nvfp4,
            weight_tying: false,
        };
        let mut backend = HeadRecorder::default();
        super::wire_global_tensors_and_raw(
            &mut backend,
            &globals,
            super::RawAcceptance {
                skip_f32_when_raw: true,
                q6k_head: true,
                bf16_head: true,
                nvfp4_head: true,
                bf16_embedding: true,
                kquant_embedding: true,
            },
        );
        assert_eq!(backend.output_proj_len, 0);
        assert_eq!(
            backend.output_proj_raw,
            Some((raw.len(), QuantScheme::Nvfp4))
        );
    }
}
