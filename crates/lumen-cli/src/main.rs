//! Lumen CLI -- command-line interface for LLM inference.

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::compute::naive::NaiveF32Backend;
use lumen_runtime::compute::simd::SimdF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::metal::MetalF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::AccelerateBatchBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_async::AsyncWeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use lumen_format::quantization::QuantScheme;

use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "run" => run_inference(&args[2..]),
        "generate-test-model" => generate_test_model_cmd(&args[2..]),
        "bench" => bench_cmd(&args[2..]),
        "purge" => purge_cmd(&args[2..]),
        "convert" => convert_cmd(&args[2..]),
        "--help" | "-h" | "help" => print_usage(),
        "--version" | "-V" => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}

// ---- generate-test-model ----

fn generate_test_model_cmd(args: &[String]) {
    use lumen_format::large_model::{self, LargeModelConfig};
    use lumen_format::test_model::{generate_test_model, TestModelConfig};
    use std::io::{BufWriter, Write};

    let mut output_path = "test_model.lbc".to_string();
    let mut size: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                i += 1;
                output_path = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --output requires a path");
                    std::process::exit(1);
                }).clone();
            }
            "--size" => {
                i += 1;
                size = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --size requires a value (256mb, 1gb, 4gb, 7b)");
                    std::process::exit(1);
                }).clone());
            }
            other if !other.starts_with('-') && i == 0 => {
                output_path = other.to_string();
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    match size {
        Some(sz) => {
            let config = match sz.as_str() {
                "256mb" => LargeModelConfig::bench_256mb(),
                "1gb" => LargeModelConfig::bench_1gb(),
                "4gb" => LargeModelConfig::bench_4gb(),
                "7b" => LargeModelConfig::llama_7b(),
                _ => {
                    eprintln!("Error: unknown size '{sz}'. Use: 256mb, 1gb, 4gb, 7b");
                    std::process::exit(1);
                }
            };

            let est = config.estimated_file_size();
            println!("Generating {sz} model (~{})...", large_model::format_size(est));
            println!("  Layers: {}", config.num_layers);
            println!("  Hidden dim: {}", config.hidden_dim);
            println!("  Heads: {} (KV: {})", config.num_heads, config.num_kv_heads);
            println!("  Head dim: {}", config.head_dim);
            println!("  Intermediate dim: {}", config.intermediate_dim);
            println!("  Vocab size: {}", config.vocab_size);
            println!("  Layer blob size: {}", large_model::format_size(config.layer_blob_size()));

            let file = std::fs::File::create(&output_path).unwrap_or_else(|e| {
                eprintln!("Error creating {output_path}: {e}");
                std::process::exit(1);
            });
            let w = BufWriter::with_capacity(8 * 1024 * 1024, file);
            large_model::generate_large_model(w, &config).unwrap_or_else(|e| {
                eprintln!("Error generating model: {e}");
                std::process::exit(1);
            });

            let actual_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
            println!("Written: {} ({})", output_path, large_model::format_size(actual_size));
        }
        None => {
            // Default: generate the tiny test model
            let config = TestModelConfig::default();
            let data = generate_test_model(&config);

            let mut f = std::fs::File::create(&output_path).unwrap_or_else(|e| {
                eprintln!("Error creating {output_path}: {e}");
                std::process::exit(1);
            });
            f.write_all(&data).unwrap_or_else(|e| {
                eprintln!("Error writing {output_path}: {e}");
                std::process::exit(1);
            });

            println!("Generated test model: {output_path}");
            println!("  Layers: {}", config.num_layers);
            println!("  Heads: {} (KV: {})", config.num_heads, config.num_kv_heads);
            println!("  Hidden dim: {}", config.hidden_dim);
            println!("  Intermediate dim: {}", config.intermediate_dim);
            println!("  Head dim: {}", config.head_dim);
            println!("  Vocab size: {}", config.vocab_size);
            println!("  Max seq len: {}", config.max_seq_len);
            println!("  Size: {} bytes", data.len());
        }
    }
}

// ---- purge ----

fn purge_cmd(args: &[String]) {
    let mut model_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --model requires a path");
                    std::process::exit(1);
                }).clone());
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Error: --model is required");
        eprintln!("USAGE: lumen purge --model <path.lbc>");
        std::process::exit(1);
    });

    let path = Path::new(&model_path);
    if !path.exists() {
        eprintln!("Error: file not found: {model_path}");
        std::process::exit(1);
    }

    #[cfg(unix)]
    {
        println!("Purging page cache for: {model_path}");
        lumen_runtime::storage::purge_file_cache(path).unwrap_or_else(|e| {
            eprintln!("Error purging cache: {e}");
            std::process::exit(1);
        });
        println!("Done. File evicted from page cache (best-effort on macOS).");
    }

    #[cfg(not(unix))]
    {
        eprintln!("Page cache purge is not supported on this platform.");
        std::process::exit(1);
    }
}

// ---- bench ----

fn bench_cmd(args: &[String]) {
    use lumen_bench::config::{BackendChoice, BenchConfig, BenchPipelineMode, ModelSpec};
    use lumen_bench::output;
    use lumen_bench::runner;
    use lumen_bench::suite;

    let mut suite_name = "minimal".to_string();
    let mut size = "1gb".to_string();
    let mut backend = "mmap".to_string();
    let mut prefetch: usize = 2;
    let mut mode = "minmem".to_string();
    let mut cold_start = false;
    let mut iters: usize = 3;
    let mut prompt_len: usize = 128;
    let mut gen_len: usize = 32;
    let mut output_dir = std::env::temp_dir().join("lumen_bench");
    let mut json = false;
    let mut use_simd = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--suite" => { i += 1; suite_name = parse_arg(args, i, "--suite"); }
            "--size" => { i += 1; size = parse_arg(args, i, "--size"); }
            "--backend" => { i += 1; backend = parse_arg(args, i, "--backend"); }
            "--prefetch" => { i += 1; prefetch = parse_arg_num(args, i, "--prefetch"); }
            "--mode" => { i += 1; mode = parse_arg(args, i, "--mode"); }
            "--cold-start" => { cold_start = true; }
            "--iters" => { i += 1; iters = parse_arg_num(args, i, "--iters"); }
            "--prompt-len" => { i += 1; prompt_len = parse_arg_num(args, i, "--prompt-len"); }
            "--gen-len" => { i += 1; gen_len = parse_arg_num(args, i, "--gen-len"); }
            "--output-dir" => { i += 1; output_dir = PathBuf::from(parse_arg(args, i, "--output-dir")); }
            "--json" => { json = true; }
            "--simd" => { use_simd = true; }
            "--help" | "-h" => { print_bench_usage(); return; }
            other => {
                eprintln!("Unknown bench option: {other}");
                print_bench_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut configs: Vec<BenchConfig> = match suite_name.as_str() {
        "minimal" => suite::minimal_experiment(&output_dir, &size),
        "async-comparison" => suite::async_comparison_suite(&output_dir, &size),
        "ssd-hypothesis" => suite::ssd_hypothesis_suite(&output_dir, &size),
        "custom" => {
            let backend_choice = match backend.as_str() {
                "sync" => BackendChoice::Sync,
                "mmap" => BackendChoice::Mmap,
                "async" => BackendChoice::AsyncSync,
                _ => {
                    eprintln!("Error: --backend must be 'sync', 'mmap', or 'async'");
                    std::process::exit(1);
                }
            };
            let pipeline_mode = match mode.as_str() {
                "minmem" => BenchPipelineMode::MinMem,
                "perf" => BenchPipelineMode::Perf,
                _ => {
                    eprintln!("Error: --mode must be 'minmem' or 'perf'");
                    std::process::exit(1);
                }
            };
            vec![BenchConfig {
                model: ModelSpec::Generate {
                    size: size.clone(),
                    output_dir: output_dir.clone(),
                },
                backend: backend_choice,
                pipeline_mode,
                prefetch_distance: prefetch,
                prompt_length: prompt_len,
                generate_length: gen_len,
                cold_start,
                warmup_iters: 1,
                bench_iters: iters,
                temperature: 0.0,
                seed: 42,
                use_simd,
                label: format!("{size}-{backend}-{mode}-{}{}", if cold_start { "cold" } else { "warm" }, if use_simd { "-simd" } else { "" }),
            }]
        }
        _ => {
            eprintln!("Unknown suite: {suite_name}. Use: minimal, async-comparison, ssd-hypothesis, custom");
            std::process::exit(1);
        }
    };

    // Apply --simd flag to all configs from preset suites
    if use_simd {
        for c in &mut configs {
            c.use_simd = true;
            if !c.label.ends_with("-simd") {
                c.label = format!("{}-simd", c.label);
            }
        }
    }

    println!("lumen bench — {} config(s)\n", configs.len());

    let mut summaries = Vec::new();
    for (idx, config) in configs.iter().enumerate() {
        println!("[{}/{}] {}", idx + 1, configs.len(), config.label);
        match runner::run_bench(config) {
            Ok(summary) => summaries.push(summary),
            Err(e) => {
                eprintln!("  ERROR: {e}");
                continue;
            }
        }
    }

    println!("\n=== Results ===\n");
    if json {
        output::print_json(&summaries);
    } else {
        output::print_table(&summaries);
    }
}

fn parse_arg(args: &[String], i: usize, name: &str) -> String {
    args.get(i).unwrap_or_else(|| {
        eprintln!("Error: {name} requires a value");
        std::process::exit(1);
    }).clone()
}

fn parse_arg_num<T: std::str::FromStr>(args: &[String], i: usize, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let val = parse_arg(args, i, name);
    val.parse().unwrap_or_else(|e| {
        eprintln!("Error: {name} must be a valid number, got '{val}': {e}");
        std::process::exit(1);
    })
}

// ---- run inference ----

fn run_inference(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut tokens_str: Option<String> = None;
    let mut max_tokens: usize = 10;
    let mut temperature: f32 = 1.0;
    let mut seed: u64 = 42;
    let mut use_sync = false;
    let mut use_async = false;
    let mut use_simd = false;
    let mut use_metal = false;
    let mut use_accelerate = false;
    let mut gpu_resident = true;  // Default: GPU-resident when --metal
    let mut option_a = false;
    let mut threads: usize = 0;
    let mut profile = false;
    let mut context_len: Option<usize> = None;
    let mut verbose_routing = false;
    let mut routing_bias: Option<f32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --model requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--tokens" => {
                i += 1;
                tokens_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --tokens requires token IDs");
                    std::process::exit(1);
                }).clone());
            }
            "--max-tokens" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --max-tokens requires a number");
                    std::process::exit(1);
                });
                max_tokens = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --max-tokens must be a positive integer, got: {val}");
                    std::process::exit(1);
                });
            }
            "--temperature" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --temperature requires a number");
                    std::process::exit(1);
                });
                temperature = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --temperature must be a number, got: {val}");
                    std::process::exit(1);
                });
            }
            "--seed" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --seed requires a number");
                    std::process::exit(1);
                });
                seed = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --seed must be an integer, got: {val}");
                    std::process::exit(1);
                });
            }
            "--threads" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --threads requires a number");
                    std::process::exit(1);
                });
                threads = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --threads must be a non-negative integer, got: {val}");
                    std::process::exit(1);
                });
            }
            "--sync" => {
                use_sync = true;
            }
            "--async" => {
                use_async = true;
            }
            "--simd" => {
                use_simd = true;
            }
            "--metal" => {
                use_metal = true;
            }
            "--accelerate" => {
                use_accelerate = true;
            }
            "--gpu-resident" => {
                gpu_resident = true;
            }
            "--no-gpu-resident" | "--streaming" => {
                gpu_resident = false;
            }
            "--option-a" => {
                option_a = true;
            }
            "--profile" => {
                profile = true;
            }
            "--verbose-routing" => {
                verbose_routing = true;
            }
            "--routing-bias" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --routing-bias requires a float value");
                    std::process::exit(1);
                });
                routing_bias = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --routing-bias must be a number, got: {val}");
                    std::process::exit(1);
                }));
            }
            "--context-len" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --context-len requires a number");
                    std::process::exit(1);
                });
                context_len = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --context-len must be a positive integer, got: {val}");
                    std::process::exit(1);
                }));
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_run_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Error: --model is required");
        print_run_usage();
        std::process::exit(1);
    });
    let tokens_str = tokens_str.unwrap_or_else(|| {
        eprintln!("Error: --tokens is required");
        print_run_usage();
        std::process::exit(1);
    });

    let prompt_tokens: Vec<u32> = tokens_str
        .split_whitespace()
        .map(|s| {
            s.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("Error: token IDs must be non-negative integers, got: {s}");
                std::process::exit(1);
            })
        })
        .collect();

    if prompt_tokens.is_empty() {
        eprintln!("Error: --tokens must contain at least one token ID");
        std::process::exit(1);
    }

    let path = Path::new(&model_path);
    if !path.exists() {
        eprintln!("Error: model file not found: {model_path}");
        std::process::exit(1);
    }

    let sampling = SamplingParams {
        temperature,
        seed: Some(seed),
    };
    let stop = StopCondition::MaxTokens(max_tokens);

    // GPU-resident only applies to Metal backend
    if !use_metal {
        gpu_resident = false;
    }

    if use_async {
        run_with_async(path, use_simd, use_metal, use_accelerate, threads, profile, &prompt_tokens, &stop, &sampling, context_len);
    } else if use_sync {
        run_with_sync(path, use_simd, use_metal, use_accelerate, threads, profile, &prompt_tokens, &stop, &sampling, context_len);
    } else {
        run_with_mmap(path, use_simd, use_metal, use_accelerate, gpu_resident, option_a, threads, profile, verbose_routing, routing_bias, &prompt_tokens, &stop, &sampling, context_len);
    }
}

/// Compute effective max_seq_len: user override, or right-sized for the actual
/// prompt + generation length to avoid GPU memory pressure from oversized KV caches.
///
/// Models with large context windows (e.g. 128K) allocate KV caches proportional
/// to max_seq_len. On GPU, this causes severe performance degradation due to memory
/// pressure / TLB thrashing (empirically measured: Llama 3.2 at context=8192 is 28%
/// slower than context=256 for pp128, because KV cache grows from 8 MB to 537 MB).
///
/// Default: min(model_max, max(prompt_len + max_gen + 256 headroom, 512)).
/// This ensures KV cache is right-sized for actual usage while leaving headroom
/// for multi-turn conversations. Use --context-len to override.
fn effective_max_seq_len(model_max: usize, user_override: Option<usize>, prompt_len: usize, max_gen: usize) -> usize {
    let effective = match user_override {
        Some(n) => n.min(model_max),
        None => {
            // Right-size: actual usage + 256 headroom, minimum 512
            let needed = prompt_len + max_gen + 256;
            let capped = needed.max(512).min(model_max);
            capped
        }
    };
    if effective < model_max {
        eprintln!("  Context length: {} (model supports {}, use --context-len {} to increase)",
            effective, model_max, model_max);
    }
    effective
}

/// Create and initialize the appropriate compute backend.
///
/// `threads` controls the SIMD backend thread pool size:
///   0 = auto-detect (all available cores), N = use exactly N threads.
#[allow(clippy::too_many_arguments)]
fn create_backend(
    use_simd: bool,
    #[allow(unused_variables)]
    use_metal: bool,
    threads: usize,
    profile: bool,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    hyperparams: &lumen_format::hyperparams::ModelHyperparams,
    #[allow(unused_variables)]
    output_proj_raw: Vec<u8>,
    #[allow(unused_variables)]
    output_proj_quant: QuantScheme,
    #[allow(unused_variables)]
    embedding_raw: Vec<u8>,
    #[allow(unused_variables)]
    embedding_quant: QuantScheme,
    #[allow(unused_variables)]
    weight_tying: bool,
) -> Box<dyn ComputeBackend> {
    #[cfg(target_os = "macos")]
    if use_metal {
        let mut backend = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        println!("Metal GPU backend: {}", backend.device_name());
        backend.set_global_tensors(embedding, final_norm, output_proj);
        if matches!(output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !output_proj_raw.is_empty() {
            println!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                output_proj_quant, output_proj_raw.len(), output_proj_raw.len() as f64 / 1048576.0);
            backend.set_output_proj_q8(output_proj_raw, output_proj_quant);
        }
        if matches!(embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !embedding_raw.is_empty() {
            println!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                embedding_quant, embedding_raw.len(), embedding_raw.len() as f64 / 1048576.0);
            backend.set_embedding_raw(embedding_raw, embedding_quant);
        }
        if weight_tying {
            println!("  weight_tying: output_proj shares embedding storage");
            backend.set_weight_tying(true);
        }
        backend.init(hyperparams).unwrap_or_else(|e| {
            eprintln!("Error: Metal initialization failed: {e}");
            std::process::exit(1);
        });
        if profile { backend.set_profile(true); }
        return Box::new(backend);
    }

    #[cfg(not(target_os = "macos"))]
    if use_metal {
        eprintln!("Error: --metal is only supported on macOS");
        std::process::exit(1);
    }

    if use_simd {
        let mut backend = SimdF32Backend::with_threads(threads);
        backend.set_global_tensors(embedding, final_norm, output_proj);
        backend.init(hyperparams).unwrap();
        if profile { backend.set_profile(true); }
        Box::new(backend)
    } else {
        let mut backend = NaiveF32Backend::new();
        backend.set_global_tensors(embedding, final_norm, output_proj);
        backend.init(hyperparams).unwrap();
        if profile { backend.set_profile(true); }
        Box::new(backend)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_with_async(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
) {
    let provider = AsyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, _ => 1024 });

    let backend = create_backend(
        use_simd,
        use_metal,
        threads,
        profile,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &provider.lbc().header.hyperparams,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 2,
        kv_precision: KvPrecision::F32,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let engine = InferenceEngine::new(config, provider.lbc().header.hyperparams);
    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate, use_metal,
        &provider.lbc().header.hyperparams,
        &provider.embedding, &provider.final_norm, &provider.output_proj,
        prompt_tokens, stop, sampling, profile,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_with_sync(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
) {
    let provider = SyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, _ => 1024 });

    let backend = create_backend(
        use_simd,
        use_metal,
        threads,
        profile,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &provider.lbc().header.hyperparams,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision: KvPrecision::F32,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let engine = InferenceEngine::new(config, provider.lbc().header.hyperparams);
    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate, use_metal,
        &provider.lbc().header.hyperparams,
        &provider.embedding, &provider.final_norm, &provider.output_proj,
        prompt_tokens, stop, sampling, profile,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_with_mmap(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_accelerate: bool,
    gpu_resident: bool,
    option_a: bool,
    threads: usize,
    profile: bool,
    verbose_routing: bool,
    routing_bias: Option<f32>,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
) {
    let mmap_config = MmapConfig {
        prefetch_window: 2,
        advise_sequential: true,
        release_with_dontneed: true,
    };
    let provider = MmapWeightProvider::open(path, mmap_config).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, _ => 1024 });

    // Create hyperparams with capped max_seq_len for Metal init.
    // This ensures KV cache, RoPE tables, and scratch buffers use the capped value.
    let mut hyperparams_capped = provider.lbc().header.hyperparams;
    hyperparams_capped.max_seq_len = max_seq_len as u32;

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 2,
        kv_precision: KvPrecision::F32,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let engine = InferenceEngine::new(config, hyperparams_capped);

    // Metal path: use concrete MetalF32Backend for both batched prefill + decode
    #[cfg(target_os = "macos")]
    if use_metal {
        println!("Prompt tokens: {prompt_tokens:?}");
        println!("Running inference with Metal GPU batched prefill...\n");

        let mut metal = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        println!("Metal GPU backend: {}", metal.device_name());
        metal.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.output_proj_raw.is_empty() {
            println!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                provider.output_proj_quant, provider.output_proj_raw.len(), provider.output_proj_raw.len() as f64 / 1048576.0);
            metal.set_output_proj_q8(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.embedding_raw.is_empty() {
            println!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                provider.embedding_quant, provider.embedding_raw.len(), provider.embedding_raw.len() as f64 / 1048576.0);
            metal.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            metal.set_weight_tying(true);
        }
        // Configure MoE expert caching for streaming mode (non-GPU-resident).
        // Default capacity: 4 experts per MoE layer (covers ~40-50% activations
        // based on heavy-tailed expert distribution research).
        if hyperparams_capped.num_experts.unwrap_or(0) > 0 && !gpu_resident {
            let num_layers = hyperparams_capped.num_layers as usize;
            let cache_capacity = num_layers * 4; // 4 hot experts/layer
            metal.configure_expert_cache(path, cache_capacity);
            println!(
                "MoE expert cache: {} experts/layer x {} layers = {} capacity",
                4, num_layers, cache_capacity,
            );
        }
        // Enable router diagnostics before init so per-layer buffers are allocated.
        if verbose_routing {
            metal.configure_router_debug(true);
            println!("Router diagnostics: enabled (--verbose-routing)");
        }
        // Configure cache-conditional routing bias.
        if let Some(lambda) = routing_bias {
            metal.configure_routing_bias(lambda);
            println!("Routing bias \u{03bb}={lambda} (cache-conditional routing active)");
        }
        metal.init(&hyperparams_capped).unwrap_or_else(|e| {
            eprintln!("Error: Metal initialization failed: {e}");
            std::process::exit(1);
        });
        if profile { metal.set_profile(true); }

        // Configure MoE Option A dispatch (streaming + GPU-resident).
        // When enabled, only top-K experts are dispatched per token instead of all 8.
        // Option A now works in GPU-resident mode via two-CB split per MoE layer.
        if option_a {
            metal.configure_option_a(true);
            let top_k = hyperparams_capped.num_active_experts.unwrap_or(2) as usize;
            if !gpu_resident {
                // Streaming mode: warmup profiler to populate expert cache.
                metal.configure_expert_warmup(8, top_k);
                println!("Option A enabled: top-{} expert dispatch (streaming mode)", top_k);
            } else {
                println!("Option A enabled: top-{} expert dispatch (GPU-resident mode)", top_k);
            }
        }

        // Pre-load all layer weights into GPU-resident Metal buffers.
        // Eliminates TLB misses and page table walks from mmap access.
        if gpu_resident {
            metal.preload_weights_gpu_resident(&provider).unwrap_or_else(|e| {
                eprintln!("Error: GPU-resident preload failed: {e}");
                std::process::exit(1);
            });
        }

        match engine.generate_with_metal_prefill(prompt_tokens, &provider, &metal, stop, sampling) {
            Ok(result) => {
                println!("Generated tokens: {:?}", result.tokens);
                println!("\n--- Metrics ---");
                println!("{}", result.metrics.summary());
                metal.print_profile();
                // Print MoE expert caching stats.
                if let Some(summary) = metal.expert_profiler_summary() {
                    println!("\n--- MoE Expert Profiler ---");
                    println!("Total record() calls: {}", summary.total_tokens);
                    for (i, entropy) in summary.per_layer_entropy.iter().enumerate() {
                        println!("  Layer {i}: entropy={entropy:.3}");
                    }
                    if !summary.global_top_experts.is_empty() {
                        println!("Top 10 global experts:");
                        for &(layer, eid, freq) in summary.global_top_experts.iter().take(10) {
                            println!("  layer={layer} expert={eid} freq={freq:.3}");
                        }
                    }
                }
                if let Some(stats) = metal.expert_cache_stats() {
                    println!("\n--- MoE Expert Cache ---");
                    println!("  cached: {}/{} experts ({} bytes)",
                        stats.cached_experts, stats.capacity, stats.cached_bytes);
                    println!("  hits: {}, misses: {}, hit_rate: {:.1}%",
                        stats.total_hits, stats.total_misses, stats.hit_rate * 100.0);
                }
                // MoE expert I/O statistics.
                let (disk, cache, blob) = metal.expert_io_stats();
                if disk + cache + blob > 0 {
                    let total = disk + cache + blob;
                    println!("\n--- MoE Expert I/O ---");
                    println!("  from_cache: {} bytes ({:.1}%)",
                        cache, cache as f64 / total as f64 * 100.0);
                    println!("  from_disk:  {} bytes ({:.1}%)",
                        disk, disk as f64 / total as f64 * 100.0);
                    println!("  from_blob:  {} bytes ({:.1}%)",
                        blob, blob as f64 / total as f64 * 100.0);
                    println!("  total:      {} bytes", total);
                }
                // Router diagnostics summary.
                if let Some(summary) = metal.router_debug_summary() {
                    println!("\n{summary}");
                }
            }
            Err(e) => {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let backend = create_backend(
        use_simd,
        use_metal,
        threads,
        profile,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hyperparams_capped,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate, use_metal,
        &hyperparams_capped,
        &provider.embedding, &provider.final_norm, &provider.output_proj,
        prompt_tokens, stop, sampling, profile,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    engine: &InferenceEngine,
    weights: &dyn WeightProvider,
    backend: &dyn ComputeBackend,
    #[allow(unused_variables)]
    use_accelerate: bool,
    #[allow(unused_variables)]
    use_metal: bool,
    #[allow(unused_variables)]
    hyperparams: &lumen_format::hyperparams::ModelHyperparams,
    #[allow(unused_variables)]
    embedding: &[f32],
    #[allow(unused_variables)]
    final_norm: &[f32],
    #[allow(unused_variables)]
    output_proj: &[f32],
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    _profile: bool,
) {
    println!("Prompt tokens: {prompt_tokens:?}");

    // Metal batched prefill is handled in the run_with_* functions directly,
    // not here, to avoid creating a second Metal backend.

    #[cfg(target_os = "macos")]
    if use_accelerate {
        println!("Running inference with Accelerate batched prefill...\n");
        let mut accel = AccelerateBatchBackend::new(
            hyperparams,
            prompt_tokens.len(),
            embedding.to_vec(),
            final_norm.to_vec(),
        );
        match engine.generate_with_prefill(prompt_tokens, weights, backend, &mut accel, stop, sampling) {
            Ok(result) => {
                println!("Generated tokens: {:?}", result.tokens);
                println!("\n--- Metrics ---");
                println!("{}", result.metrics.summary());
                backend.print_profile();
            }
            Err(e) => {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    #[cfg(not(target_os = "macos"))]
    if use_accelerate {
        eprintln!("Error: --accelerate is only supported on macOS");
        std::process::exit(1);
    }

    println!("Running inference...\n");

    match engine.generate(prompt_tokens, weights, backend, stop, sampling) {
        Ok(result) => {
            println!("Generated tokens: {:?}", result.tokens);
            println!("\n--- Metrics ---");
            println!("{}", result.metrics.summary());
            backend.print_profile();
        }
        Err(e) => {
            eprintln!("Inference error: {e}");
            std::process::exit(1);
        }
    }
}

// ---- convert ----

fn convert_cmd(args: &[String]) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut dequantize = false;
    let mut requant: Option<QuantScheme> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --input requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--output" | "-o" => {
                i += 1;
                output_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --output requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--dequantize" => {
                dequantize = true;
            }
            "--requant" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --requant requires a value (e.g. q4_0)");
                    std::process::exit(1);
                });
                requant = match val.to_lowercase().as_str() {
                    "q4_0" | "q4" => Some(QuantScheme::Q4_0),
                    "q8_0" | "q8" => Some(QuantScheme::Q8_0),
                    other => {
                        eprintln!("Error: unsupported requant target: {other} (supported: q4_0, q8_0)");
                        std::process::exit(1);
                    }
                };
            }
            "--help" | "-h" => {
                print_convert_usage();
                return;
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_convert_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.unwrap_or_else(|| {
        eprintln!("Error: --input is required");
        print_convert_usage();
        std::process::exit(1);
    });

    let output_path = output_path.unwrap_or_else(|| {
        // Default: same name with .lbc extension
        let p = Path::new(&input_path);
        p.with_extension("lbc")
            .to_string_lossy()
            .into_owned()
    });

    let input = Path::new(&input_path);
    if !input.exists() {
        eprintln!("Error: input file not found: {input_path}");
        std::process::exit(1);
    }

    let opts = ConvertOptions {
        alignment: 128 * 1024,
        dequantize_to_f32: dequantize,
        requant_to: requant,
    };

    println!("Converting: {input_path} -> {output_path}");

    match convert_gguf_to_lbc(input, Path::new(&output_path), &opts) {
        Ok(stats) => {
            println!("{stats}");
            println!("Done.");
        }
        Err(e) => {
            eprintln!("Conversion error: {e}");
            std::process::exit(1);
        }
    }
}

// ---- Help text ----

fn print_usage() {
    println!(
        "\
lumen - GPU-resident LLM inference engine

USAGE:
    lumen <COMMAND> [OPTIONS]

COMMANDS:
    run                   Run inference on a model
    generate-test-model   Generate a synthetic model (LBC file)
    bench                 Run benchmarks (I/O, throughput, cold/warm)
    purge                 Evict a model file from the OS page cache
    convert               Convert a model to LBC format
    help                  Print this help message

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version"
    );
}

fn print_run_usage() {
    println!(
        "\
USAGE:
    lumen run --model <path.lbc> --tokens \"0 1 2\" [OPTIONS]

OPTIONS:
    --model <path>        Path to LBC model file (required)
    --tokens <ids>        Space-separated token IDs (required)
    --max-tokens <n>      Max tokens to generate (default: 10)
    --temperature <f>     Sampling temperature (default: 1.0, 0=greedy)
    --seed <n>            Random seed (default: 42)
    --sync                Use sync file backend instead of mmap
    --async               Use async I/O backend (background prefetch thread)
    --simd                Use SIMD-accelerated compute backend
    --threads <n>         Thread count for SIMD backend (default: 0 = auto-detect)
    --metal               Use Metal GPU compute backend (macOS only)
    --accelerate          Use Accelerate AMX batched prefill (macOS only, use with --simd)
    --gpu-resident        Pre-load all weights into GPU Metal buffers (DEFAULT with --metal)
    --no-gpu-resident     Disable GPU-resident mode, use SSD-streaming (alias: --streaming)
    --option-a            MoE: dispatch only top-K experts per token (streaming + GPU-resident)
    --routing-bias <f>    MoE: cache-conditional routing bias lambda (default: 0.0 = disabled)
    --context-len <n>     Max context length for KV cache (default: auto-sized to prompt + generation + headroom)
    --profile             Print per-operation timing breakdown after inference
    --verbose-routing     Print per-layer MoE router diagnostics (entropy, expert selection)"
    );
}

fn print_bench_usage() {
    println!(
        "\
USAGE:
    lumen bench [OPTIONS]

OPTIONS:
    --suite <name>       Preset suite: minimal | async-comparison | ssd-hypothesis | custom (default: minimal)
    --size <spec>        Model size: 256mb | 1gb | 4gb | 7b (default: 1gb)
    --backend <name>     Storage backend: sync | mmap | async (default: mmap)
    --prefetch <n>       Prefetch distance (default: 2)
    --mode <name>        Pipeline mode: minmem | perf (default: minmem)
    --cold-start         Purge page cache before each iteration
    --iters <n>          Measured iterations (default: 3)
    --prompt-len <n>     Prompt token count (default: 128)
    --gen-len <n>        Tokens to generate (default: 32)
    --output-dir <path>  Directory for models and results
    --json               Output as JSON
    --simd               Use SIMD-accelerated compute backend"
    );
}

fn print_convert_usage() {
    println!(
        "\
USAGE:
    lumen convert --input <model.gguf> [--output <model.lbc>] [OPTIONS]

OPTIONS:
    --input <path>       Path to input GGUF model file (required)
    --output <path>      Path to output LBC file (default: input with .lbc extension)
    --dequantize         Dequantize all tensors to F32 (larger but compatible)
    --requant <scheme>   Requantize weights to target scheme during conversion
                         Supported: q4_0, q8_0"
    );
}
