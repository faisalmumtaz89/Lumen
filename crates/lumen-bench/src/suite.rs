//! Preset benchmark suites.

use crate::config::{BackendChoice, BenchConfig, BenchPipelineMode, ModelSpec};
use std::path::Path;

/// Minimal experiment: 1 model, mmap, cold vs warm (2 configs, quick validation).
pub fn minimal_experiment(model_dir: &Path, size: &str) -> Vec<BenchConfig> {
    let model = ModelSpec::Generate {
        size: size.to_string(),
        output_dir: model_dir.to_path_buf(),
    };

    vec![
        BenchConfig {
            model: model.clone(),
            backend: BackendChoice::Mmap,
            pipeline_mode: BenchPipelineMode::MinMem,
            cold_start: true,
            label: format!("{size}-mmap-cold"),
            ..Default::default()
        },
        BenchConfig {
            model,
            backend: BackendChoice::Mmap,
            pipeline_mode: BenchPipelineMode::MinMem,
            cold_start: false,
            label: format!("{size}-mmap-warm"),
            ..Default::default()
        },
    ]
}

/// Async comparison: all 3 backends (sync, mmap, async) with cold/warm.
pub fn async_comparison_suite(model_dir: &Path, size: &str) -> Vec<BenchConfig> {
    let model = ModelSpec::Generate {
        size: size.to_string(),
        output_dir: model_dir.to_path_buf(),
    };

    let mut configs = Vec::new();

    for &backend in &[BackendChoice::Sync, BackendChoice::Mmap, BackendChoice::AsyncSync] {
        let backend_name = match backend {
            BackendChoice::Sync => "sync",
            BackendChoice::Mmap => "mmap",
            BackendChoice::AsyncSync => "async",
        };

        for &cold in &[true, false] {
            let cold_name = if cold { "cold" } else { "warm" };

            // Skip cold start for sync backend (no page cache control).
            if cold && backend == BackendChoice::Sync {
                continue;
            }

            configs.push(BenchConfig {
                model: model.clone(),
                backend,
                pipeline_mode: BenchPipelineMode::MinMem,
                prefetch_distance: 2,
                cold_start: cold,
                label: format!("{size}-{backend_name}-{cold_name}"),
                ..Default::default()
            });
        }
    }

    configs
}

/// Full hypothesis validation: sweep backends, pipeline modes, and cold/warm.
pub fn ssd_hypothesis_suite(model_dir: &Path, size: &str) -> Vec<BenchConfig> {
    let model = ModelSpec::Generate {
        size: size.to_string(),
        output_dir: model_dir.to_path_buf(),
    };

    let mut configs = Vec::new();

    for &backend in &[BackendChoice::Sync, BackendChoice::Mmap, BackendChoice::AsyncSync] {
        let backend_name = match backend {
            BackendChoice::Sync => "sync",
            BackendChoice::Mmap => "mmap",
            BackendChoice::AsyncSync => "async",
        };

        for &mode in &[BenchPipelineMode::MinMem, BenchPipelineMode::Perf] {
            let mode_name = match mode {
                BenchPipelineMode::MinMem => "minmem",
                BenchPipelineMode::Perf => "perf",
            };

            for &cold in &[true, false] {
                let cold_name = if cold { "cold" } else { "warm" };

                // Skip cold start for sync backend (no page cache control)
                if cold && backend == BackendChoice::Sync {
                    continue;
                }

                for &prefetch in &[1, 2, 4] {
                    configs.push(BenchConfig {
                        model: model.clone(),
                        backend,
                        pipeline_mode: mode,
                        prefetch_distance: prefetch,
                        cold_start: cold,
                        label: format!("{size}-{backend_name}-{mode_name}-{cold_name}-pf{prefetch}"),
                        ..Default::default()
                    });
                }
            }
        }
    }

    configs
}
