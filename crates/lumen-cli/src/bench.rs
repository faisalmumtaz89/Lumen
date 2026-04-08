use crate::help::print_bench_usage;
use crate::run::{parse_arg, parse_arg_num};

use std::path::PathBuf;

pub(crate) fn generate_test_model_cmd(args: &[String]) {
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

pub(crate) fn purge_cmd(args: &[String]) {
    use std::path::Path;

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
            "--help" | "-h" => {
                println!("USAGE: lumen purge --model <path.lbc>");
                return;
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

pub(crate) fn bench_cmd(args: &[String]) {
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
