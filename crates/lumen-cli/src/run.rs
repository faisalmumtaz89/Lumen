use crate::help::print_run_usage;

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::cpu_simd::SimdF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::metal::MetalF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::AccelerateBatchBackend;
#[cfg(feature = "cuda")]
use lumen_runtime::CudaBackend;
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

use std::path::Path;

pub(crate) fn parse_arg(args: &[String], i: usize, name: &str) -> String {
    args.get(i).unwrap_or_else(|| {
        eprintln!("Error: {name} requires a value");
        std::process::exit(1);
    }).clone()
}

pub(crate) fn parse_arg_num<T: std::str::FromStr>(args: &[String], i: usize, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let val = parse_arg(args, i, name);
    val.parse().unwrap_or_else(|e| {
        eprintln!("Error: {name} must be a valid number, got '{val}': {e}");
        std::process::exit(1);
    })
}

/// Levenshtein edit distance between two strings (standard 2-row DP).
fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }

    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0usize; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost)
                .min(prev[j + 1] + 1)
                .min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Suggest similar model names for a mistyped input using edit distance and
/// prefix matching against the registry's canonical keys and aliases.
fn suggest_models(input: &str, registry: &crate::registry::Registry) -> Vec<String> {
    let input_lower = input.to_lowercase();
    let mut candidates: Vec<(usize, String)> = Vec::new();

    // Check canonical keys.
    for entry in registry.list() {
        let key_lower = entry.key.to_lowercase();
        let dist = levenshtein(&input_lower, &key_lower);
        if dist <= 3 || key_lower.starts_with(&input_lower) || input_lower.starts_with(&key_lower) {
            let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            candidates.push((dist, format!("  {:<20} {} ({})", entry.key, entry.display_name, quants.join(", "))));
        }
    }

    // Check aliases.
    for alias in registry.alias_keys() {
        let alias_lower = alias.to_lowercase();
        let dist = levenshtein(&input_lower, &alias_lower);
        if dist <= 3 || alias_lower.starts_with(&input_lower) || input_lower.starts_with(&alias_lower) {
            // Resolve to display the canonical entry info.
            if let Some(entry) = registry.resolve(alias) {
                let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                let line = format!("  {:<20} {} ({})", alias, entry.display_name, quants.join(", "));
                // Avoid duplicates (alias might resolve to same model already added).
                if !candidates.iter().any(|(_, l)| l == &line) {
                    candidates.push((dist, line));
                }
            }
        }
    }

    // Sort by edit distance (closest first).
    candidates.sort_by_key(|(dist, _)| *dist);
    candidates.into_iter().map(|(_, line)| line).collect()
}

pub(crate) fn run_inference(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut tokens_str: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut system_str: Option<String> = None;
    let mut max_tokens: usize = usize::MAX; // unlimited by default, stops at EOS
    let mut temperature: f32 = 0.8;
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    let mut use_sync = false;
    let mut use_async = false;
    let mut use_simd = false;
    let mut use_metal = false;
    let mut use_accelerate = false;
    let mut use_cuda = false;
    let mut explicitly_chose_backend = false;
    let mut cuda_device: usize = 0;
    let mut gpu_resident = true;  // Default: GPU-resident when --metal
    let mut option_a = false;
    let mut threads: usize = 0;
    let mut profile = false;
    let mut context_len: Option<usize> = None;
    let mut verbose = false;
    let mut verbose_routing = false;
    let mut routing_bias: Option<f32> = None;
    let mut positional_args: Vec<String> = Vec::new();

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
            "--prompt" => {
                i += 1;
                prompt_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --prompt requires text");
                    std::process::exit(1);
                }).clone());
            }
            "--system" => {
                i += 1;
                system_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --system requires text");
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
                explicitly_chose_backend = true;
            }
            "--metal" => {
                use_metal = true;
                explicitly_chose_backend = true;
            }
            "--accelerate" => {
                use_accelerate = true;
                explicitly_chose_backend = true;
            }
            "--cuda" => {
                use_cuda = true;
                explicitly_chose_backend = true;
            }
            "--cuda-device" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --cuda-device requires a device ordinal (e.g. 0, 1)");
                    std::process::exit(1);
                });
                cuda_device = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --cuda-device must be a non-negative integer, got: {val}");
                    std::process::exit(1);
                });
                use_cuda = true; // --cuda-device implies --cuda
                explicitly_chose_backend = true;
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
                verbose = true; // profiling implies verbose diagnostics
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
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--help" | "-h" => {
                print_run_usage();
                return;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option: {other}");
                    print_run_usage();
                    std::process::exit(1);
                }
                // Positional argument -- collect it.
                positional_args.push(other.to_string());
            }
        }
        i += 1;
    }

    // Resolve positional args: [MODEL] [PROMPT...]
    // Flags take precedence; conflicts between flags and positionals are errors.
    if !positional_args.is_empty() {
        if model_path.is_none() {
            model_path = Some(positional_args.remove(0));
        } else if !positional_args.is_empty() {
            // --model was set via flag. Check if first positional looks like a model name.
            let first = &positional_args[0];
            let reg = crate::registry::load_registry();
            if reg.resolve(first).is_some() {
                eprintln!("Error: model specified twice — --model {} and '{}'", model_path.as_ref().unwrap(), first);
                eprintln!("Use one or the other, not both.");
                std::process::exit(1);
            }
        }
        // Remaining positionals become the prompt.
        if !positional_args.is_empty() {
            if prompt_str.is_some() {
                eprintln!("Error: --prompt flag and positional prompt are mutually exclusive.");
                eprintln!("Use either:  lumen run <model> \"prompt\"");
                eprintln!("         or: lumen run --model <model> --prompt \"prompt\"");
                std::process::exit(1);
            }
            prompt_str = Some(positional_args.join(" "));
        }
    }

    // Determine the model string for error messages before resolution.
    let model_str = model_path.clone().unwrap_or_default();

    let model_path = match model_path {
        Some(p) => p,
        None => {
            // If --prompt is provided without a model, use the default from registry.
            if prompt_str.is_some() {
                let registry = crate::registry::load_registry();
                let default = registry.default_model();
                eprintln!("No model specified. Using default: {} ({})",
                    default.display_name, default.key);
                default.key.clone()
            } else {
                eprintln!("Model name is required.\n");
                let registry = crate::registry::load_registry();
                eprintln!("Available models:");
                for entry in registry.list() {
                    let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                    quants.sort();
                    let tags: Vec<String> = quants.iter().map(|q| format!("{}:{}", entry.key, q.to_lowercase())).collect();
                    eprintln!("  {}", tags.join(", "));
                }
                eprintln!("\nUsage: lumen run <model>:<quant> \"your prompt\"");
                eprintln!("Example: lumen run qwen2.5-3b:q8_0 \"What is 2+2?\"");
                std::process::exit(1);
            }
        }
    };

    // Mutual exclusivity: --tokens and --prompt cannot both be provided.
    if tokens_str.is_some() && prompt_str.is_some() {
        eprintln!("Error: --prompt and --tokens are mutually exclusive");
        std::process::exit(1);
    }

    // Dual-mode --model: if the value looks like a file path (contains / or \,
    // or ends with .lbc/.gguf), treat it as a direct path. Otherwise, try to
    // resolve it as a preset name from the model registry.
    let model_path = resolve_model_path(&model_path, verbose);

    let path = Path::new(&model_path);
    if !path.exists() {
        eprintln!("Error: model file not found: {model_path}");
        std::process::exit(1);
    }

    // Resolve prompt tokens: either from --tokens (raw IDs) or --prompt (text -> tokenizer).
    let (prompt_tokens, tokenizer) = if let Some(ref tokens) = tokens_str {
        // --tokens mode: parse integer IDs (existing behavior, unchanged).
        let ids: Vec<u32> = tokens
            .split_whitespace()
            .map(|s| {
                s.parse::<u32>().unwrap_or_else(|_| {
                    eprintln!("Error: token IDs must be non-negative integers, got: {s}");
                    std::process::exit(1);
                })
            })
            .collect();
        if ids.is_empty() {
            eprintln!("Error: --tokens must contain at least one token ID");
            std::process::exit(1);
        }
        (ids, None) // No tokenizer needed for --tokens mode
    } else if let Some(ref prompt) = prompt_str {
        if prompt.is_empty() {
            eprintln!("Error: --prompt must not be empty");
            std::process::exit(1);
        }
        // Load tokenizer from LBC file header (targeted seek, not full-file read).
        let lbc = lumen_format::reader::LbcFile::open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error parsing model file: {e}");
                std::process::exit(1);
            });
        let tok_section = lbc.tokenizer.unwrap_or_else(|| {
            eprintln!("Error: This model has no embedded tokenizer (LBC v2).");
            eprintln!("Re-convert with: lumen convert --input model.gguf --output model.lbc");
            std::process::exit(1);
        });
        // Build TokenizerData from TokenizerSection (move fields, no clone).
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
        let tokenizer = crate::tokenize::BpeTokenizer::from_tokenizer_data(&tok_data);
        let templated = tokenizer.apply_chat_template_with_system(prompt, system_str.as_deref());
        let ids = tokenizer.encode(&templated);
        if ids.is_empty() {
            eprintln!("Error: prompt produced no tokens after tokenization");
            std::process::exit(1);
        }
        if verbose { eprintln!("Tokenized prompt: {} tokens", ids.len()); }
        (ids, Some(tokenizer))
    } else {
        eprintln!("No prompt provided.\n");
        eprintln!("Usage:");
        eprintln!("  lumen run {} \"What is the capital of France?\"", model_str);
        eprintln!("  lumen run --model {} --prompt \"Hello\"", model_str);
        std::process::exit(1);
    };

    // Backend auto-detection: if no backend flag was provided, pick the best
    // available backend for this platform.
    if !explicitly_chose_backend && !use_sync && !use_async {
        #[cfg(target_os = "macos")]
        {
            use_metal = true;
            if verbose { eprintln!("Auto-detected backend: Metal (Apple Silicon GPU)"); }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Check for NVIDIA GPU presence.
            if std::path::Path::new("/dev/nvidia0").exists() {
                #[cfg(feature = "cuda")]
                {
                    use_cuda = true;
                    if verbose { eprintln!("Auto-detected backend: CUDA (NVIDIA GPU)"); }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    use_simd = true;
                    if verbose {
                        eprintln!("NVIDIA GPU detected but CUDA support not compiled in.");
                        eprintln!("Falling back to CPU (SIMD) backend.");
                        eprintln!("Rebuild with: cargo build --release --features cuda");
                    }
                }
            } else {
                use_simd = true;
                if verbose {
                    eprintln!("No GPU detected. Using CPU (SIMD) backend.");
                    eprintln!("For faster inference:");
                    eprintln!("  macOS:  rebuild and use --metal");
                    eprintln!("  NVIDIA: rebuild with --features cuda");
                    eprintln!("  Modal:  modal run examples/cuda_inference.py");
                }
            }
        }
    }

    let sampling = SamplingParams {
        temperature,
        seed: Some(seed),
    };
    let stop = if let Some(ref tok) = tokenizer {
        let eos_ids = tok.stop_token_ids.clone();
        if eos_ids.is_empty() {
            StopCondition::MaxTokens(max_tokens)
        } else {
            StopCondition::MaxTokensOrEos { max_tokens, eos_tokens: eos_ids }
        }
    } else {
        StopCondition::MaxTokens(max_tokens)
    };

    // GPU-resident mode for Metal and CUDA backends.
    if !use_metal && !use_cuda {
        gpu_resident = false;
    }

    // Determine backend display name for the banner.
    let backend_name = if use_metal {
        "Metal"
    } else if use_cuda {
        "CUDA"
    } else if use_accelerate {
        "Accelerate"
    } else if use_simd {
        "SIMD"
    } else {
        "CPU"
    };

    // Extract short model name from file path for banner display.
    let model_display = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&model_path)
        .to_string();

    if use_async {
        run_with_async(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, threads, profile, verbose, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name);
    } else if use_sync {
        run_with_sync(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, threads, profile, verbose, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name);
    } else {
        run_with_mmap(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, gpu_resident, option_a, threads, profile, verbose, verbose_routing, routing_bias, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name);
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
fn effective_max_seq_len(model_max: usize, user_override: Option<usize>, prompt_len: usize, max_gen: usize, verbose: bool) -> usize {
    let effective = match user_override {
        Some(n) => n.min(model_max),
        None => {
            // Right-size: actual usage + 256 headroom, minimum 512
            let needed = prompt_len.saturating_add(max_gen).saturating_add(256);
            let capped = needed.max(512).min(model_max);
            capped
        }
    };
    if verbose && effective < model_max {
        eprintln!("  Context length: {} (model supports {}, use --context-len {} to increase)",
            effective, model_max, model_max);
    }
    effective
}

/// Resolve a `--model` value to a file path.
///
/// Supports three forms:
/// - File path: `/path/to/model.lbc` or `model.gguf` (returned as-is)
/// - Preset with quant tag: `qwen2.5-3b:q4_0` (downloads specific quant)
/// - Preset without tag: `qwen2.5-3b` (errors with available quants if multiple exist,
///   or auto-selects if only one quant is available)
fn resolve_model_path(value: &str, verbose: bool) -> String {
    // Heuristic: looks like a file path if it contains path separators or
    // ends with a known model file extension.
    let looks_like_path = value.contains('/')
        || value.contains('\\')
        || value.ends_with(".lbc")
        || value.ends_with(".gguf");

    if looks_like_path {
        return value.to_owned();
    }

    // Parse model:quant tag syntax (e.g., "qwen2.5-3b:q4_0").
    let (model_name, explicit_quant) = if let Some(colon_pos) = value.rfind(':') {
        let name = &value[..colon_pos];
        let tag = &value[colon_pos + 1..];
        if tag.is_empty() {
            (value, None) // trailing colon, treat as no tag
        } else {
            (name, Some(tag.to_uppercase()))
        }
    } else {
        (value, None)
    };

    // Try resolving as a preset name.
    let reg = crate::registry::load_registry();
    let entry = match reg.resolve(model_name) {
        Some(e) => e.clone(),
        None => {
            // Not a known preset -- show helpful error with suggestions.
            let suggestions = suggest_models(model_name, &reg);
            if !suggestions.is_empty() {
                eprintln!("Error: unknown model '{}'\n", model_name);
                eprintln!("Did you mean?");
                for s in &suggestions {
                    eprintln!("{s}");
                }
            } else {
                eprintln!("Error: unknown model '{}'\n", model_name);
                eprintln!("Available models:");
                for entry in reg.list() {
                    let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                    eprintln!("  {:<20} {} ({})", entry.key, entry.display_name, quants.join(", "));
                }
            }
            eprintln!("\nRun 'lumen models' to see all available models.");
            std::process::exit(1);
        }
    };

    // Determine quantization.
    let quant = if let Some(ref q) = explicit_quant {
        // User specified explicit quant tag.
        if !entry.gguf_files.contains_key(q.as_str()) {
            let available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            eprintln!("Error: quantization '{}' not available for {}\n", q, entry.display_name);
            eprintln!("Available:");
            for a in &available {
                eprintln!("  {}:{}", model_name, a.to_lowercase());
            }
            std::process::exit(1);
        }
        q.as_str().to_owned()
    } else if entry.gguf_files.len() == 1 {
        // Only one quant available — auto-select it.
        entry.gguf_files.keys().next().unwrap().clone()
    } else {
        // Multiple quants — require explicit choice.
        eprintln!("Multiple quantizations available for {}:\n", entry.display_name);
        let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
        quants.sort();
        for q in &quants {
            eprintln!("  {}:{}", model_name, q.to_lowercase());
        }
        eprintln!("\nSpecify one: lumen run {}:<quant> \"your prompt\"", model_name);
        std::process::exit(1);
    };
    let quant = quant.as_str();

    // Check cache first.
    if let Some(cached) = crate::cache::cached_lbc(&entry.key, quant) {
        if verbose { eprintln!("Using cached model: {}", cached.display()); }
        return cached.to_string_lossy().into_owned();
    }

    // Not cached -- attempt download + convert.
    let gguf_source = match entry.gguf_files.get(quant) {
        Some(src) => src.clone(),
        None => {
            let available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            eprintln!("Error: no {quant} GGUF available for {}", entry.display_name);
            eprintln!("Available quantizations: {}", available.join(", "));
            std::process::exit(1);
        }
    };

    #[cfg(feature = "download")]
    {
        if verbose { eprintln!("Model '{}' not cached. Downloading {}...", value, entry.display_name); }
        let gguf_path = resolve_download_gguf(&gguf_source.repo, &gguf_source.file, verbose);
        let lbc_out = crate::cache::lbc_path(&entry.key, quant);
        resolve_convert_to_lbc(&gguf_path, &lbc_out, verbose);
        if verbose { eprintln!("Ready: {}", lbc_out.display()); }
        return lbc_out.to_string_lossy().into_owned();
    }

    #[cfg(not(feature = "download"))]
    {
        let _ = gguf_source; // suppress unused warning
        eprintln!("Error: model '{}' is not cached.", value);
        eprintln!("Download it first with: lumen pull {}", value);
        eprintln!("Or pass a direct file path: --model /path/to/model.lbc");
        std::process::exit(1);
    }
}

#[cfg(feature = "download")]
fn resolve_download_gguf(repo: &str, filename: &str, verbose: bool) -> std::path::PathBuf {
    if let Some(existing) = crate::cache::cached_gguf(filename) {
        if verbose { eprintln!("GGUF already downloaded: {}", existing.display()); }
        return existing;
    }

    crate::cache::ensure_cache_dir().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    // Auto-download without confirmation in --model resolution (interactive
    // users who want a confirmation prompt should use `lumen pull` instead).
    crate::download::download_gguf(repo, filename, &crate::cache::cache_dir(), true)
        .unwrap_or_else(|e| {
            eprintln!("Download failed: {e}");
            std::process::exit(1);
        })
}

#[cfg(feature = "download")]
fn resolve_convert_to_lbc(gguf_path: &std::path::Path, lbc_out: &std::path::Path, verbose: bool) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let opts = ConvertOptions {
        alignment: 128 * 1024,
        dequantize_to_f32: false,
        requant_to: None,
    };

    if verbose { eprintln!("Converting to LBC: {} -> {}", gguf_path.display(), lbc_out.display()); }
    match convert_gguf_to_lbc(gguf_path, lbc_out, &opts) {
        Ok(stats) => { if verbose { eprintln!("{stats}"); } }
        Err(e) => {
            eprintln!("Conversion failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Decode and print generated tokens as text when a tokenizer is available.
/// Only the generated text goes to stdout (for piping). No headers or decoration.
fn print_generated_text(tokens: &[u32], tokenizer: Option<&crate::tokenize::BpeTokenizer>) {
    if let Some(tok) = tokenizer {
        let stop_ids = &tok.stop_token_ids;
        let clean: Vec<u32> = tokens.iter().copied().filter(|t| !stop_ids.contains(t)).collect();
        let text = tok.decode(&clean);
        println!("{text}");
    }
}

/// Print the Lumen ASCII banner with inference metrics.
///
/// Only prints when stderr is a TTY (not piped or redirected), so scripted
/// usage and benchmarks get clean output.
fn print_banner(model_name: &str, backend: &str, metrics_summary: &str) {
    // Check if stderr is a TTY. We use stderr because all banner output goes
    // there (eprintln), leaving stdout clean for token output.
    #[cfg(unix)]
    {
        extern "C" { fn isatty(fd: std::ffi::c_int) -> std::ffi::c_int; }
        // stderr = fd 2
        if unsafe { isatty(2) } == 0 {
            return;
        }
    }

    let sep = "\u{2500}".repeat(46);
    eprintln!();
    eprintln!(" _   _   _ __  __ ___ _  _ ");
    eprintln!("| | | | | |  \\/  | __| \\| |");
    eprintln!("| |_| |_| | |\\/| | _|| .` |");
    eprintln!("|____\\___/|_|  |_|___|_|\\_|");
    eprintln!();
    eprintln!(" Rust LLM Inference Engine");
    eprintln!();
    eprintln!("{sep}");
    eprintln!("  Source    github.com/faisalmumtaz89/Lumen");
    eprintln!("  Engine    Lumen v{} (Rust + {backend})", env!("CARGO_PKG_VERSION"));
    eprintln!("  Model     {model_name}");
    // Indent each line of the metrics summary for alignment within the banner.
    for line in metrics_summary.lines() {
        eprintln!("  {line}");
    }
    eprintln!("{sep}");
}

/// Select CPU or Metal backend based on platform and flags.
///
/// Extracted so that both `#[cfg(feature = "cuda")]` and `#[cfg(not(feature = "cuda"))]`
/// branches can call it without duplicating the platform-specific Metal logic.
fn create_cpu_or_metal_backend(
    #[allow(unused_variables)] use_metal: bool,
    use_simd: bool,
    threads: usize,
    verbose: bool,
) -> Box<dyn ComputeBackend> {
    #[cfg(target_os = "macos")]
    if use_metal {
        let metal = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        if verbose { eprintln!("Metal GPU backend: {}", metal.device_name()); }
        return Box::new(metal);
    }

    #[cfg(not(target_os = "macos"))]
    if use_metal {
        eprintln!("Error: --metal is only supported on macOS");
        std::process::exit(1);
    }

    if use_simd {
        Box::new(SimdF32Backend::with_threads(threads))
    } else {
        Box::new(NaiveF32Backend::new())
    }
}

/// Create and initialize the appropriate compute backend.
///
/// All backends are configured through the `ComputeBackend` trait after boxing,
/// so the setup logic (global tensors, quant data, init, profile) is shared.
///
/// `threads` controls the SIMD backend thread pool size:
///   0 = auto-detect (all available cores), N = use exactly N threads.
#[allow(clippy::too_many_arguments)]
fn create_backend(
    use_simd: bool,
    #[allow(unused_variables)]
    use_metal: bool,
    #[allow(unused_variables)]
    use_cuda: bool,
    #[allow(unused_variables)]
    cuda_device: usize,
    threads: usize,
    profile: bool,
    verbose: bool,
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
    // Construct the concrete backend and box it. All subsequent setup goes
    // through the ComputeBackend trait, keeping the logic backend-agnostic.
    #[allow(unused_mut)]
    let mut backend: Box<dyn ComputeBackend> = {
        // CUDA backend (cross-platform, requires --features cuda at build time).
        #[cfg(feature = "cuda")]
        if use_cuda {
            let cuda = CudaBackend::new(cuda_device).unwrap_or_else(|e| {
                eprintln!("Error: CUDA backend unavailable: {e}");
                std::process::exit(1);
            });
            Box::new(cuda)
        } else {
            create_cpu_or_metal_backend(use_metal, use_simd, threads, verbose)
        }

        #[cfg(not(feature = "cuda"))]
        {
            if use_cuda {
                eprintln!("Error: --cuda requires building with --features cuda");
                std::process::exit(1);
            }
            create_cpu_or_metal_backend(use_metal, use_simd, threads, verbose)
        }
    };

    // Backend-agnostic setup via trait methods.
    backend.set_global_tensors(embedding, final_norm, output_proj);

    if matches!(output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !output_proj_raw.is_empty() {
        if verbose {
            eprintln!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                output_proj_quant, output_proj_raw.len(), output_proj_raw.len() as f64 / 1048576.0);
        }
        backend.set_output_proj_raw(output_proj_raw, output_proj_quant);
    }
    if matches!(embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !embedding_raw.is_empty() {
        if verbose {
            eprintln!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                embedding_quant, embedding_raw.len(), embedding_raw.len() as f64 / 1048576.0);
        }
        backend.set_embedding_raw(embedding_raw, embedding_quant);
    }
    if weight_tying {
        if verbose { eprintln!("  weight_tying: output_proj shares embedding storage"); }
        backend.set_weight_tying(true);
    }

    backend.init(hyperparams).unwrap_or_else(|e| {
        eprintln!("Error: backend initialization failed: {e}");
        std::process::exit(1);
    });
    if profile { backend.set_profile(true); }

    backend
}

#[allow(clippy::too_many_arguments)]
fn run_with_async(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
) {
    let provider = AsyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

    let backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
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
        &engine, &provider, backend.as_ref(), use_accelerate,
        &provider.lbc().header.hyperparams,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_with_sync(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
) {
    let provider = SyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

    let backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
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
        &engine, &provider, backend.as_ref(), use_accelerate,
        &provider.lbc().header.hyperparams,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_with_mmap(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    gpu_resident: bool,
    option_a: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    verbose_routing: bool,
    routing_bias: Option<f32>,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
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
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

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
        if verbose {
            eprintln!("Prompt tokens: {prompt_tokens:?}");
            eprintln!("Running inference with Metal GPU batched prefill...\n");
        }

        let mut metal = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        if verbose { eprintln!("Metal GPU backend: {}", metal.device_name()); }
        metal.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.output_proj_raw.is_empty() {
            if verbose {
                eprintln!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                    provider.output_proj_quant, provider.output_proj_raw.len(), provider.output_proj_raw.len() as f64 / 1048576.0);
            }
            metal.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.embedding_raw.is_empty() {
            if verbose {
                eprintln!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                    provider.embedding_quant, provider.embedding_raw.len(), provider.embedding_raw.len() as f64 / 1048576.0);
            }
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
            if verbose {
                eprintln!(
                    "MoE expert cache: {} experts/layer x {} layers = {} capacity",
                    4, num_layers, cache_capacity,
                );
            }
        }
        // Enable router diagnostics before init so per-layer buffers are allocated.
        if verbose_routing {
            metal.configure_router_debug(true);
            if verbose { eprintln!("Router diagnostics: enabled (--verbose-routing)"); }
        }
        // Configure cache-conditional routing bias.
        if let Some(lambda) = routing_bias {
            metal.configure_routing_bias(lambda);
            if verbose { eprintln!("Routing bias \u{03bb}={lambda} (cache-conditional routing active)"); }
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
                if verbose { eprintln!("Option A enabled: top-{} expert dispatch (streaming mode)", top_k); }
            } else {
                if verbose { eprintln!("Option A enabled: top-{} expert dispatch (GPU-resident mode)", top_k); }
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

        match engine.generate(prompt_tokens, &provider, &metal as &dyn ComputeBackend, stop, sampling) {
            Ok(result) => {
                print_generated_text(&result.tokens, tokenizer);
                if verbose {
                    eprintln!("Generated tokens: {:?}", result.tokens);
                    eprintln!("\n--- Metrics ---");
                    eprintln!("{}", result.metrics.summary());
                    metal.print_profile();
                    // Print MoE expert caching stats.
                    if let Some(summary) = metal.expert_profiler_summary() {
                        eprintln!("\n--- MoE Expert Profiler ---");
                        eprintln!("Total record() calls: {}", summary.total_tokens);
                        for (i, entropy) in summary.per_layer_entropy.iter().enumerate() {
                            eprintln!("  Layer {i}: entropy={entropy:.3}");
                        }
                        if !summary.global_top_experts.is_empty() {
                            eprintln!("Top 10 global experts:");
                            for &(layer, eid, freq) in summary.global_top_experts.iter().take(10) {
                                eprintln!("  layer={layer} expert={eid} freq={freq:.3}");
                            }
                        }
                    }
                    if let Some(stats) = metal.expert_cache_stats() {
                        eprintln!("\n--- MoE Expert Cache ---");
                        eprintln!("  cached: {}/{} experts ({} bytes)",
                            stats.cached_experts, stats.capacity, stats.cached_bytes);
                        eprintln!("  hits: {}, misses: {}, hit_rate: {:.1}%",
                            stats.total_hits, stats.total_misses, stats.hit_rate * 100.0);
                    }
                    // MoE expert I/O statistics.
                    let (disk, cache, blob) = metal.expert_io_stats();
                    if disk + cache + blob > 0 {
                        let total = disk + cache + blob;
                        eprintln!("\n--- MoE Expert I/O ---");
                        eprintln!("  from_cache: {} bytes ({:.1}%)",
                            cache, cache as f64 / total as f64 * 100.0);
                        eprintln!("  from_disk:  {} bytes ({:.1}%)",
                            disk, disk as f64 / total as f64 * 100.0);
                        eprintln!("  from_blob:  {} bytes ({:.1}%)",
                            blob, blob as f64 / total as f64 * 100.0);
                        eprintln!("  total:      {} bytes", total);
                    }
                    // Router diagnostics summary.
                    if let Some(summary) = metal.router_debug_summary() {
                        eprintln!("\n{summary}");
                    }
                    print_banner(model_display, backend_name, &result.metrics.summary());
                }
            }
            Err(e) => {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    #[allow(unused_mut)]
    let mut backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
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

    // GPU-resident preload for CUDA: loads all layer weights to GPU memory.
    // Enables native quantized kernel paths (dp4a, HGEMV) and batched prefill.
    if gpu_resident && use_cuda {
        backend.as_mut().preload_weights(&provider).unwrap_or_else(|e| {
            eprintln!("Error: CUDA GPU-resident preload failed: {e}");
            std::process::exit(1);
        });
    }

    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate,
        &hyperparams_capped,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
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
    hyperparams: &lumen_format::hyperparams::ModelHyperparams,
    #[allow(unused_variables)]
    embedding: &[f32],
    #[allow(unused_variables)]
    final_norm: &[f32],
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    verbose: bool,
    model_display: &str,
    backend_name: &str,
) {
    if verbose { eprintln!("Prompt tokens: {prompt_tokens:?}"); }

    // Metal batched prefill is handled in the run_with_* functions directly,
    // not here, to avoid creating a second Metal backend.

    #[cfg(target_os = "macos")]
    if use_accelerate {
        if verbose { eprintln!("Running inference with Accelerate batched prefill...\n"); }
        let mut accel = AccelerateBatchBackend::new(
            hyperparams,
            prompt_tokens.len(),
            embedding.to_vec(),
            final_norm.to_vec(),
        );
        match engine.generate_with_prefill(prompt_tokens, weights, backend, &mut accel, stop, sampling) {
            Ok(result) => {
                print_generated_text(&result.tokens, tokenizer);
                if verbose {
                    eprintln!("Generated tokens: {:?}", result.tokens);
                    eprintln!("\n--- Metrics ---");
                    eprintln!("{}", result.metrics.summary());
                    backend.print_profile();
                    print_banner(model_display, backend_name, &result.metrics.summary());
                }
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

    if verbose { eprintln!("Running inference...\n"); }

    match engine.generate(prompt_tokens, weights, backend, stop, sampling) {
        Ok(result) => {
            print_generated_text(&result.tokens, tokenizer);
            if verbose {
                eprintln!("Generated tokens: {:?}", result.tokens);
                eprintln!("\n--- Metrics ---");
                eprintln!("{}", result.metrics.summary());
                backend.print_profile();
                print_banner(model_display, backend_name, &result.metrics.summary());
            }
        }
        Err(e) => {
            eprintln!("Inference error: {e}");
            std::process::exit(1);
        }
    }
}
