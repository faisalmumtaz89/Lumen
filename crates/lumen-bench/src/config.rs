//! Benchmark configuration types.

use std::path::PathBuf;

/// How to obtain the model for benchmarking.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// Use an existing LBC file at this path.
    Path(PathBuf),
    /// Generate a synthetic model of the given size.
    Generate {
        /// One of: "256mb", "1gb", "4gb", "7b".
        size: String,
        /// Directory to write the generated model.
        output_dir: PathBuf,
    },
}

/// Which storage backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Sync,
    Mmap,
    AsyncSync,
}

/// Pipeline mode for the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchPipelineMode {
    MinMem,
    Perf,
}

/// Full benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub model: ModelSpec,
    pub backend: BackendChoice,
    pub pipeline_mode: BenchPipelineMode,
    pub prefetch_distance: usize,
    pub prompt_length: usize,
    pub generate_length: usize,
    pub cold_start: bool,
    pub warmup_iters: usize,
    pub bench_iters: usize,
    pub temperature: f32,
    pub seed: u64,
    /// Use SIMD compute backend instead of naive.
    pub use_simd: bool,
    /// Human-readable label for this config.
    pub label: String,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            model: ModelSpec::Generate {
                size: "1gb".to_string(),
                output_dir: std::env::temp_dir().join("lumen_bench"),
            },
            backend: BackendChoice::Mmap,
            pipeline_mode: BenchPipelineMode::MinMem,
            prefetch_distance: 2,
            prompt_length: 128,
            generate_length: 32,
            cold_start: false,
            warmup_iters: 2,
            bench_iters: 7,
            temperature: 0.0, // greedy for deterministic comparison
            seed: 42,
            use_simd: false,
            label: "default".to_string(),
        }
    }
}
