//! Benchmark runner — executes configs and collects results.

use crate::config::{BackendChoice, BenchConfig, BenchPipelineMode, ModelSpec};
use crate::results::{BenchResult, BenchSummary};

use lumen_format::large_model::{self, LargeModelConfig};
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::cpu_simd::SimdF32Backend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::engine::{GenerationResult, InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_async::AsyncWeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;

use lumen_format::hyperparams::ModelHyperparams;

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Create and initialize the appropriate compute backend based on config.
fn create_backend(
    use_simd: bool,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    hyperparams: &ModelHyperparams,
) -> Result<Box<dyn ComputeBackend>, BenchError> {
    if use_simd {
        let mut backend = SimdF32Backend::new();
        backend.set_global_tensors(embedding, final_norm, output_proj);
        backend.init(hyperparams)?;
        Ok(Box::new(backend))
    } else {
        let mut backend = NaiveF32Backend::new();
        backend.set_global_tensors(embedding, final_norm, output_proj);
        backend.init(hyperparams)?;
        Ok(Box::new(backend))
    }
}

/// Errors from the benchmark runner.
#[derive(Debug)]
pub enum BenchError {
    Io(std::io::Error),
    Runtime(lumen_runtime::RuntimeError),
    Config(String),
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Runtime(e) => write!(f, "runtime error: {e}"),
            Self::Config(s) => write!(f, "config error: {s}"),
        }
    }
}

impl std::error::Error for BenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Runtime(e) => Some(e),
            Self::Config(_) => None,
        }
    }
}

impl From<std::io::Error> for BenchError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

impl From<lumen_runtime::RuntimeError> for BenchError {
    fn from(e: lumen_runtime::RuntimeError) -> Self { Self::Runtime(e) }
}

/// Resolve a `ModelSpec` to a path, generating the model if needed.
pub fn ensure_model(spec: &ModelSpec) -> Result<PathBuf, BenchError> {
    match spec {
        ModelSpec::Path(p) => {
            if !p.exists() {
                return Err(BenchError::Config(format!("model not found: {}", p.display())));
            }
            Ok(p.clone())
        }
        ModelSpec::Generate { size, output_dir } => {
            std::fs::create_dir_all(output_dir)?;
            let filename = format!("bench_{size}.lbc");
            let path = output_dir.join(&filename);
            if path.exists() {
                eprintln!("  Using existing model: {}", path.display());
                return Ok(path);
            }

            let config = match size.as_str() {
                "256mb" => LargeModelConfig::bench_256mb(),
                "1gb" => LargeModelConfig::bench_1gb(),
                "4gb" => LargeModelConfig::bench_4gb(),
                "7b" => LargeModelConfig::llama_7b(),
                _ => return Err(BenchError::Config(format!("unknown size: {size}"))),
            };

            let est = config.estimated_file_size();
            eprintln!(
                "  Generating {size} model (~{})...",
                large_model::format_size(est)
            );

            let file = std::fs::File::create(&path)?;
            let w = BufWriter::with_capacity(8 * 1024 * 1024, file);
            large_model::generate_large_model(w, &config)?;

            eprintln!("  Written: {}", path.display());
            Ok(path)
        }
    }
}

/// Run a single benchmark config for all iterations.
pub fn run_bench(config: &BenchConfig) -> Result<BenchSummary, BenchError> {
    let model_path = ensure_model(&config.model)?;

    // Generate prompt tokens (just sequential IDs — we're benchmarking I/O, not quality)
    let prompt_tokens: Vec<u32> = (0..config.prompt_length as u32).collect();

    let pipeline_mode = match config.pipeline_mode {
        BenchPipelineMode::MinMem => PipelineMode::MinMem,
        BenchPipelineMode::Perf => PipelineMode::Perf,
    };

    let sampling = SamplingParams {
        temperature: config.temperature,
        seed: Some(config.seed),
        ..Default::default()
    };
    let stop = StopCondition::MaxTokens(config.generate_length);

    let mut results = Vec::with_capacity(config.bench_iters);

    // Warmup iterations (results discarded)
    for i in 0..config.warmup_iters {
        eprintln!("  Warmup {}/{}...", i + 1, config.warmup_iters);
        run_single_iteration(
            &model_path,
            config,
            pipeline_mode,
            &prompt_tokens,
            &stop,
            &sampling,
        )?;
    }

    // Measured iterations
    for i in 0..config.bench_iters {
        eprintln!("  Iteration {}/{}...", i + 1, config.bench_iters);
        let result = run_single_iteration(
            &model_path,
            config,
            pipeline_mode,
            &prompt_tokens,
            &stop,
            &sampling,
        )?;
        results.push(result);
    }

    Ok(BenchSummary::new(config.label.clone(), results))
}

fn run_single_iteration(
    model_path: &Path,
    config: &BenchConfig,
    pipeline_mode: PipelineMode,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
) -> Result<BenchResult, BenchError> {
    match config.backend {
        BackendChoice::Mmap => run_mmap_iteration(
            model_path, config, pipeline_mode, prompt_tokens, stop, sampling,
        ),
        BackendChoice::Sync => run_sync_iteration(
            model_path, config, pipeline_mode, prompt_tokens, stop, sampling,
        ),
        BackendChoice::AsyncSync => run_async_iteration(
            model_path, config, pipeline_mode, prompt_tokens, stop, sampling,
        ),
    }
}

/// Convert inference metrics into a BenchResult.
///
/// This is the shared extraction logic for all backend variants. Before this
/// refactor, the same ~20 lines were duplicated three times across
/// run_mmap_iteration, run_async_iteration, and run_sync_iteration.
fn metrics_to_bench_result(
    result: &GenerationResult,
    initial_residency: f64,
) -> BenchResult {
    let m = &result.metrics;
    let total_stall: Duration = m.per_layer_timings.iter().map(|t| t.stall_time).sum();
    let total_compute: Duration = m.per_layer_timings.iter().map(|t| t.total_time()).sum();
    let stall_fraction = if total_compute.is_zero() {
        0.0
    } else {
        total_stall.as_secs_f64() / total_compute.as_secs_f64()
    };

    BenchResult {
        total_time: m.total_time,
        prefill_time: m.prefill_time,
        decode_time: m.decode_time,
        tpot_ms: m.tpot_ms,
        bytes_read: m.io.bytes_read,
        read_ops: m.io.read_ops,
        bandwidth_gibs: m.io.read_bandwidth_gibs(),
        weight_cache_hit_rate: m.weight_cache_hit_rate,
        initial_residency,
        final_residency: 0.0,
        total_stall_time: total_stall,
        stall_fraction,
        prompt_tokens: m.prompt_tokens,
        generated_tokens: m.generated_tokens,
    }
}

/// Run inference with a given weight provider and collect a BenchResult.
///
/// This generic helper eliminates the duplication across the three backend
/// iteration functions. Each backend only needs to open its provider,
/// extract the global tensors, and call this function.
fn run_iteration_with_provider(
    provider: &dyn lumen_runtime::WeightProvider,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    hyperparams: &ModelHyperparams,
    config: &BenchConfig,
    pipeline_mode: PipelineMode,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    initial_residency: f64,
) -> Result<BenchResult, BenchError> {
    let backend = create_backend(
        config.use_simd,
        embedding,
        final_norm,
        output_proj,
        hyperparams,
    )?;

    let rt_config = RuntimeConfig {
        pipeline_mode,
        prefetch_distance: config.prefetch_distance,
        kv_precision: KvPrecision::F32,
        max_seq_len: hyperparams.max_seq_len as usize,
        collect_per_layer_timings: true,
    };

    let engine = InferenceEngine::new(rt_config, *hyperparams);
    let result = engine.generate(prompt_tokens, provider, backend.as_ref(), stop, sampling)?;

    Ok(metrics_to_bench_result(&result, initial_residency))
}

fn run_mmap_iteration(
    model_path: &Path,
    config: &BenchConfig,
    pipeline_mode: PipelineMode,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
) -> Result<BenchResult, BenchError> {
    let mmap_config = MmapConfig {
        prefetch_window: config.prefetch_distance,
        advise_sequential: true,
        release_with_dontneed: pipeline_mode == PipelineMode::MinMem,
    };
    let provider = MmapWeightProvider::open(model_path, mmap_config)?;

    // Cold start: purge page cache
    #[cfg(unix)]
    let initial_residency = if config.cold_start {
        lumen_runtime::storage::purge_file_cache(model_path)?;
        0.0
    } else {
        0.0
    };
    #[cfg(not(unix))]
    let initial_residency = 0.0;

    let hp = provider.lbc().header.hyperparams;
    run_iteration_with_provider(
        &provider,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hp, config, pipeline_mode, prompt_tokens, stop, sampling, initial_residency,
    )
}

fn run_async_iteration(
    model_path: &Path,
    config: &BenchConfig,
    pipeline_mode: PipelineMode,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
) -> Result<BenchResult, BenchError> {
    let provider = AsyncWeightProvider::open(model_path)?;
    let hp = provider.lbc().header.hyperparams;
    run_iteration_with_provider(
        &provider,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hp, config, pipeline_mode, prompt_tokens, stop, sampling, 0.0,
    )
}

fn run_sync_iteration(
    model_path: &Path,
    config: &BenchConfig,
    pipeline_mode: PipelineMode,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
) -> Result<BenchResult, BenchError> {
    let provider = SyncWeightProvider::open(model_path)?;
    let hp = provider.lbc().header.hyperparams;
    run_iteration_with_provider(
        &provider,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hp, config, pipeline_mode, prompt_tokens, stop, sampling, 0.0,
    )
}
