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
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::cache::WeightProvider;
use lumen_runtime::storage::MmapConfig;
use lumen_format::reader::LbcFile;
use lumen_runtime::RuntimeConfig;
#[cfg(target_os = "macos")]
use lumen_runtime::MetalF32Backend;
#[cfg(feature = "cuda")]
use lumen_runtime::CudaBackend;
use lumen_format::QuantScheme;

use lumen_server::{build_router, EngineWorker, ModelInfo, Tokenize};

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
    context_len: usize,
    backend: BackendChoice,
    backend_device: usize,
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
            context_len: 8192,
            backend: BackendChoice::Auto,
            backend_device: 0,
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
    lumen-server [OPTIONS] --model <MODEL>

REQUIRED:
    --model <ID|PATH>      Registry name (e.g. qwen3.5-9b, qwen3.5-moe-35b-a3b)
                           OR direct path to a .lbc file. Registry-name
                           resolution requires the LBC to be cached under
                           ~/.cache/lumen/ (run `lumen pull <name>` first).

OPTIONS:
    --quant <Q>            Quantization tag when --model is a registry name
                           (q8_0, q4_0, bf16). Default: q8_0
    --host <HOST>          Listen host. Default: 127.0.0.1
    --port <N>             Listen port. Default: 8000
    --context-len <N>      Max sequence length (KV cache size).
                           Capped at the model's native max_seq_len.
                           Default: 8192
    --backend <B>          cuda | metal | cpu
                           Default: auto (Metal on macOS, CUDA if available, else CPU)
    --backend-device <N>   GPU device ordinal (CUDA only). Default: 0
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

ENVIRONMENT VARIABLES (CUDA backend):
    LUMEN_CUDA_DECODE_DELAY_US=<N>
                           Per-decode-step CPU sleep in microseconds, applied
                           after `cudaDeviceSynchronize` in the CUDA decode
                           paths. `lumen-server` defaults to `50` (auto-applied)
                           to mitigate a CUDA-scheduler timing race under heavy
                           MoE Q4 concurrency; the `lumen run` CLI defaults to
                           `0`. Set `=0` to disable on the server. Cost <=1% TPOT.

EXAMPLES:
    # Auto-detect backend, default port 8000
    lumen-server --model qwen3.5-9b --quant q8_0

    # CUDA, custom port
    lumen-server --model qwen3.5-9b --quant q8_0 --backend cuda --port 9000

    # Direct file path
    lumen-server --model /path/to/qwen3-5-9b-Q8_0.lbc --port 8080

ENDPOINTS:
    GET  /v1/models                  OpenAI-style model list
    POST /v1/chat/completions        OpenAI chat completion (SSE optional)
    POST /v1/completions             OpenAI text completion (SSE optional)
    POST /v1/messages                Anthropic messages (SSE optional)
"
    );
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut args = Args::default();
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
            }
            "--host" => {
                i += 1;
                args.host = raw.get(i).ok_or("--host requires a value")?.clone();
            }
            "--port" => {
                i += 1;
                let v = raw.get(i).ok_or("--port requires a value")?;
                args.port = v.parse().map_err(|_| format!("--port must be u16, got {v}"))?;
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
                    other => return Err(format!("--backend must be cuda|metal|cpu|auto, got {other}")),
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
            "--sync" => {
                args.sync_provider = true;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("lumen-server {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if args.model.is_empty() {
        return Err("--model is required (try --help)".to_string());
    }
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
    // `~/.cache/lumen/`. The server must look in the SAME place or
    // operators have to symlink. Implemented inline (no `dirs` dep on the
    // server crate) by matching the platform manually.
    if let Ok(home) = std::env::var("HOME") {
        #[cfg(target_os = "macos")]
        {
            let macos = PathBuf::from(&home).join("Library").join("Caches").join("lumen");
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
/// `.lbc`/`.gguf`. Otherwise it is taken as a registry-style name and a
/// cached LBC is looked up by `~/.cache/lumen/<key>-<QUANT>.lbc`, with the
/// dot-to-dash key normalization the CLI's registry uses
/// (e.g. `qwen3.5-9b` -> `qwen3-5-9b`).
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
    let quant = quant_arg
        .map(str::to_owned)
        .or_else(|| tag_quant.map(|s| s.to_owned()))
        .unwrap_or_else(|| "q8_0".to_owned())
        .to_uppercase();

    // Registry uses `qwen3-5-9b` (dot-to-dash) as the canonical key; the
    // README + lumen-cli accept `qwen3.5-9b` as an alias. Normalize.
    let key = name.replace('.', "-");

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
                if std::path::Path::new("/dev/nvidia0").exists() {
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
enum ServerWeights {
    Sync(SyncWeightProvider),
    Mmap(MmapWeightProvider),
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

    /// Borrow as `&dyn WeightProvider` for `preload_weights`.
    fn as_dyn(&self) -> &dyn WeightProvider {
        match self {
            ServerWeights::Sync(p) => p,
            ServerWeights::Mmap(p) => p,
        }
    }

    /// Consume into the `Arc<dyn WeightProvider>` the engine worker owns.
    fn into_arc(self) -> Arc<dyn WeightProvider> {
        match self {
            ServerWeights::Sync(p) => Arc::new(p),
            ServerWeights::Mmap(p) => Arc::new(p),
        }
    }
}

fn wire_global_tensors_and_raw(
    backend: &mut dyn ComputeBackend,
    g: &WeightGlobals<'_>,
) {
    backend.set_global_tensors(
        g.embedding.to_vec(),
        g.final_norm.to_vec(),
        g.output_proj.to_vec(),
    );
    if matches!(
        g.embedding_quant,
        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16
    ) && !g.embedding_raw.is_empty()
    {
        backend.set_embedding_raw(g.embedding_raw.to_vec(), g.embedding_quant);
    }
    if matches!(
        g.output_proj_quant,
        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16
    ) && !g.output_proj_raw.is_empty()
    {
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
    let lbc_path = resolve_model_path(&args.model, args.quant.as_deref())?;
    eprintln!("[lumen-server] model: {}", lbc_path.display());

    // Open weights + extract tokenizer from the LBC's embedded tokenizer
    // section. Same pattern as `tests/server_soak.rs:289-317`.
    let lbc = lumen_format::reader::LbcFile::open(&lbc_path)
        .map_err(|e| format!("open LBC {lbc_path:?}: {e}"))?;
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
    let tokenizer: Arc<dyn Tokenize> = Arc::new(BpeTokenizerAdapter { inner: bpe, eos_ids });

    // Resolve the backend first — the weight-provider choice depends on it.
    let backend_choice = select_backend(args.backend);
    eprintln!("[lumen-server] backend: {backend_choice:?}");

    // Provider selection: the Metal GPU-resident path defaults
    // to the zero-copy `MmapWeightProvider` so the no-copy unified-buffer path
    // (`LUMEN_METAL_MMAP_ONLY=1`) engages and we avoid the ~10 GB redundant CPU
    // copy + GPU private weight copy that `SyncWeightProvider` (pread-into-Vec,
    // non-page-aligned) forces. `--sync`, CUDA, and CPU keep `SyncWeightProvider`
    // (CUDA/CPU read F32-dequantized layers; the sync path is also the
    // guarded fallback). Mmap and sync are proven byte-identical on the
    // Metal path by `metal_sync_mmap_argmax_parity_test`.
    let use_mmap_provider =
        matches!(backend_choice, BackendChoice::Metal) && !args.sync_provider;
    let provider: ServerWeights = if use_mmap_provider {
        // Enable the no-copy residency path. We set the env BEFORE the backend
        // constructor / `preload_weights` reads it (gpu_resident.rs probe). An
        // explicit operator override (e.g. `LUMEN_METAL_MMAP_ONLY=0`) still wins
        // because we only set it when unset.
        if std::env::var_os("LUMEN_METAL_MMAP_ONLY").is_none() {
            // SAFETY: single-threaded boot, before any worker thread is spawned.
            unsafe { std::env::set_var("LUMEN_METAL_MMAP_ONLY", "1"); }
        }
        let mmap_config = MmapConfig {
            prefetch_window: 2,
            advise_sequential: true,
            release_with_dontneed: true,
        };
        eprintln!("[lumen-server] weights: mmap/no-copy (zero CPU copy; --sync to force legacy)");
        ServerWeights::Mmap(
            MmapWeightProvider::open(&lbc_path, mmap_config)
                .map_err(|e| format!("open weights (mmap) {lbc_path:?}: {e}"))?,
        )
    } else {
        eprintln!("[lumen-server] weights: sync (full CPU copy)");
        ServerWeights::Sync(
            SyncWeightProvider::open(&lbc_path)
                .map_err(|e| format!("open weights (sync) {lbc_path:?}: {e}"))?,
        )
    };

    // Plumb model-aware defaults into the runtime BEFORE
    // any backend constructor or kernel dispatch fires its first env-var
    // read. The two setters are idempotent and cheap (one atomic store
    // each); they let the operator run a BF16 dense LBC against
    // `lumen-server` with NO `LUMEN_CUDA_BF16_GEMMEX=1` and NO
    // `LUMEN_CUDA_DECODE_GRAPH=1` / `_QGATE` / `_TILED` and the runtime
    // still picks the right path for that cell. Operator explicit env vars
    // remain authoritative — model-aware defaults only apply when the env
    // is unset. F3 typo validator already ran in `main()` BEFORE this; the
    // CUDA backend constructor (`CudaBackend::new`) is invoked downstream
    // and is when the cached env-or-default helpers latch.
    lumen_runtime::runtime_defaults::set_path_is_server(true);
    lumen_runtime::runtime_defaults::set_model_dense_quant(provider.globals().output_proj_quant);
    // Feed the MoE flag alongside the dense-quant hint so
    // the Q8-only flag resolvers (`q8_split_default` and the chain that
    // delegates to it) stay OFF on MoE 30B-A3B. Without this gate, MoE
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
                wire_global_tensors_and_raw(&mut metal, &provider.globals());
                metal
                    .init(&hyperparams_capped)
                    .map_err(|e| format!("Metal init: {e}"))?;
                metal
                    .preload_weights(provider.as_dyn())
                    .map_err(|e| format!("Metal preload_weights: {e}"))?;
                // KvPrecision::F16 — Metal requires F16.
                (Box::new(metal), KvPrecision::F16)
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("--backend metal is only supported on macOS".to_string());
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let mut cuda = CudaBackend::new(args.backend_device)
                    .map_err(|e| format!("CUDA backend unavailable (device {}): {e}", args.backend_device))?;
                wire_global_tensors_and_raw(&mut cuda, &provider.globals());
                cuda.init(&hyperparams_capped)
                    .map_err(|e| format!("CUDA init: {e}"))?;
                cuda.preload_weights(provider.as_dyn())
                    .map_err(|e| format!("CUDA preload_weights: {e}"))?;
                // KvPrecision::F32 — CUDA requires F32 per validate_kv_precision.
                (Box::new(cuda), KvPrecision::F32)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = args.backend_device;
                return Err(
                    "--backend cuda requires building with --features cuda".to_string(),
                );
            }
        }
        BackendChoice::Cpu => {
            let mut cpu = NaiveF32Backend::new();
            wire_global_tensors_and_raw(&mut cpu, &provider.globals());
            cpu.init(&hyperparams_capped)
                .map_err(|e| format!("CPU init: {e}"))?;
            (Box::new(cpu), KvPrecision::F32)
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
        id: args.model.clone(),
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

    // Bind and serve. Graceful shutdown on Ctrl-C / SIGTERM.
    let app = build_router(handle);
    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| format!("bind {bind_addr}: {e}"))?;
    let local = listener.local_addr().map_err(|e| format!("local_addr: {e}"))?;
    eprintln!("[lumen-server] listening on http://{local}");
    eprintln!("[lumen-server] try: curl http://{local}/v1/models");
    eprintln!("[lumen-server] (Ctrl-C to stop)");

    // `log_level` is captured for future structured-log wiring; today the
    // bin logs via `eprintln!` only, so we acknowledge the value to avoid
    // an unused-variable warning without committing to a specific log crate.
    let _ = &args.log_level;

    let shutdown = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            eprintln!("[lumen-server] ctrl_c watcher failed: {e}");
        }
        eprintln!("[lumen-server] shutdown requested");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| format!("axum serve: {e}"))?;
    eprintln!("[lumen-server] stopped");
    Ok(())
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
