//! Lumen CLI -- command-line interface for LLM inference.

// Route Rust-heap allocations through mimalloc instead of the macOS
// system allocator. Addresses the libmalloc page-retention pattern that
// surfaced as a +154 MB/h RSS slope under sustained server-style load.
// Allocator switch is per-binary (declared here for `lumen`); library
// crates do not depend on mimalloc so external consumers can pick their
// own.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod bench;
pub mod cache;
mod convert;
#[allow(unused)]
mod download;
mod help;
pub mod registry;
mod run;
pub mod tokenize;

fn main() {
    // env-var typo validator. Catches `GDN_REGISTER_RESIDENT=1`
    // (missing LUMEN_CUDA_ prefix - the literal bug) and the truncated form
    // `LUMEN_CUDA_GDN_REGISTER_RESIDENT` with a missing trailing character
    // (mis-spelled suffix). Emits one stderr
    // WARNING per suspect name with the closest canonical matches as
    // suggestions. The CLI path explicitly leaves
    // `set_path_is_server(false)` (the default) so
    // `LUMEN_CUDA_DECODE_DELAY_US` defaults to `0` µs — CLI is
    // fork-deterministic. The server bin
    // (`lumen-server::bin::main`) calls `set_path_is_server(true)`.
    let _warnings = lumen_runtime::runtime_defaults::validate_lumen_env_vars();
    lumen_runtime::runtime_defaults::mark_validator_ran();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        help::print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "run" => run::run_inference(&args[2..]),
        "pull" => pull_cmd(&args[2..]),
        "models" => models_cmd(),
        "generate-test-model" => bench::generate_test_model_cmd(&args[2..]),
        "bench" => bench::bench_cmd(&args[2..]),
        "purge" => bench::purge_cmd(&args[2..]),
        "convert" => convert::convert_cmd(&args[2..]),
        "--help" | "-h" | "help" => help::print_usage(),
        "--version" | "-V" => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            eprintln!("Unknown command: {other}");
            help::print_usage();
            std::process::exit(1);
        }
    }
}

/// Download and convert a model from the registry.
///
/// Usage: `lumen pull <model-name> [--quant Q8_0] [--yes]`
fn pull_cmd(args: &[String]) {
    let reg = registry::load_registry();

    // Parse arguments: positional model name, optional --quant and --yes.
    let mut model_name: Option<&str> = None;
    let mut quant_override: Option<String> = None;
    let mut skip_confirm = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--quant" => {
                i += 1;
                quant_override = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --quant requires a value (e.g. Q8_0, Q4_0, F16)");
                    std::process::exit(1);
                }).clone());
            }
            "--yes" | "-y" => {
                skip_confirm = true;
            }
            "--help" | "-h" => {
                help::print_pull_usage();
                return;
            }
            other if other.starts_with('-') => {
                eprintln!("Unknown option: {other}");
                help::print_pull_usage();
                std::process::exit(1);
            }
            name => {
                if model_name.is_some() {
                    eprintln!("Error: unexpected argument: {name}");
                    help::print_pull_usage();
                    std::process::exit(1);
                }
                model_name = Some(name);
            }
        }
        i += 1;
    }

    let model_name = model_name.unwrap_or_else(|| {
        eprintln!("Usage: lumen pull <model>:<quant> [--yes]");
        eprintln!("\nAvailable models:");
        for entry in reg.list() {
            let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            quants.sort();
            let tags: Vec<String> = quants.iter().map(|q| format!("{}:{}", entry.key, q.to_lowercase())).collect();
            eprintln!("  {}", tags.join(", "));
        }
        std::process::exit(1);
    });

    // Parse model:quant tag syntax (e.g., "qwen3.5-9b:q4_0").
    let (resolved_name, tag_quant) = if let Some(pos) = model_name.rfind(':') {
        let name = &model_name[..pos];
        let tag = &model_name[pos + 1..];
        if tag.is_empty() { (model_name, None) } else { (name, Some(tag.to_uppercase())) }
    } else {
        (model_name, None)
    };

    let entry = reg.resolve(resolved_name).unwrap_or_else(|| {
        eprintln!("Unknown model: {resolved_name}");
        eprintln!("\nAvailable models:");
        for e in reg.list() {
            eprintln!("  {}", e.key);
        }
        std::process::exit(1);
    });

    // Quant priority: colon tag > --quant flag > auto-select (single) > error (multiple)
    let quant_owned: String;
    let quant: &str = if let Some(ref tq) = tag_quant {
        quant_owned = tq.clone();
        &quant_owned
    } else if let Some(ref qo) = quant_override {
        quant_owned = qo.clone();
        &quant_owned
    } else if entry.gguf_files.len() == 1 {
        quant_owned = entry.gguf_files.keys().next().unwrap().clone();
        &quant_owned
    } else {
        // Multiple quants — require explicit choice.
        eprintln!("Multiple quantizations available for {}:\n", entry.display_name);
        let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
        quants.sort();
        for q in &quants {
            eprintln!("  {}:{}", resolved_name, q.to_lowercase());
        }
        eprintln!("\nSpecify one: lumen pull {}:<quant>", resolved_name);
        std::process::exit(1);
    };

    // Check if LBC is already cached.
    if let Some(lbc_path) = cache::cached_lbc(&entry.key, quant) {
        println!("Already cached: {}", lbc_path.display());
        return;
    }

    // Validate the requested quant exists in the registry for this model.
    let gguf_source = entry.gguf_files.get(quant).unwrap_or_else(|| {
        let available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
        eprintln!("No {quant} GGUF available for {}", entry.display_name);
        eprintln!("Available quantizations: {}", available.join(", "));
        std::process::exit(1);
    });

    // Download GGUF -- every shard for multi-shard sources, the single file
    // for legacy single-shard. The primary (first) shard path is what the
    // converter is pointed at; the multi-shard reader auto-discovers siblings
    // from there.
    let gguf_path = pull_download_gguf_shards(gguf_source, skip_confirm);

    // Convert GGUF to LBC.
    let lbc_out = cache::lbc_path(&entry.key, quant);
    pull_convert_to_lbc(&gguf_path, &lbc_out);

    println!("\nReady: {}", lbc_out.display());
    println!("Run with: lumen run {}:{} \"Hello\"", resolved_name, quant.to_lowercase());
}

/// Download every shard listed in a [`registry::GgufSource`], returning the
/// path to the primary (first) shard. For single-file sources this is a
/// single download; for multi-shard sources this fetches every sibling shard
/// sequentially so the converter can ingest the full set.
#[cfg(feature = "download")]
fn pull_download_gguf_shards(src: &registry::GgufSource, skip_confirm: bool) -> std::path::PathBuf {
    cache::ensure_cache_dir().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    if src.is_multi_shard() {
        eprintln!(
            "Multi-shard model: {} shard(s) to fetch from {}",
            src.files.len(),
            src.repo
        );
    }

    let mut primary: Option<std::path::PathBuf> = None;
    for (idx, filename) in src.files.iter().enumerate() {
        let path = pull_download_gguf(&src.repo, filename, skip_confirm);
        if idx == 0 {
            primary = Some(path);
        }
    }
    primary.unwrap_or_else(|| {
        eprintln!("Internal error: GgufSource had no shard files");
        std::process::exit(1);
    })
}

#[cfg(not(feature = "download"))]
fn pull_download_gguf_shards(_src: &registry::GgufSource, _skip_confirm: bool) -> std::path::PathBuf {
    eprintln!("Error: download support is not compiled in.");
    eprintln!("Rebuild with: cargo build --release --features download");
    std::process::exit(1);
}

/// Download a GGUF file, returning its path. Exits on failure.
#[cfg(feature = "download")]
fn pull_download_gguf(repo: &str, filename: &str, skip_confirm: bool) -> std::path::PathBuf {
    // Check if GGUF is already cached.
    if let Some(existing) = cache::cached_gguf(filename) {
        eprintln!("GGUF already downloaded: {}", existing.display());
        return existing;
    }

    cache::ensure_cache_dir().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    download::download_gguf(repo, filename, &cache::cache_dir(), skip_confirm)
        .unwrap_or_else(|e| {
            eprintln!("Download failed: {e}");
            std::process::exit(1);
        })
}

#[cfg(not(feature = "download"))]
fn pull_download_gguf(_repo: &str, _filename: &str, _skip_confirm: bool) -> std::path::PathBuf {
    eprintln!("Error: download support is not compiled in.");
    eprintln!("Rebuild with: cargo build --release --features download");
    std::process::exit(1);
}

/// Convert a GGUF file to LBC. Exits on failure.
fn pull_convert_to_lbc(gguf_path: &std::path::Path, lbc_out: &std::path::Path) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let opts = ConvertOptions {
        alignment: 128 * 1024,
        dequantize_to_f32: false,
        requant_to: None,
        target: crate::convert::default_target_for_host(),
    };

    eprintln!("Converting to LBC: {} -> {}", gguf_path.display(), lbc_out.display());

    match convert_gguf_to_lbc(gguf_path, lbc_out, &opts) {
        Ok(stats) => {
            eprintln!("{stats}");
        }
        Err(e) => {
            eprintln!("Conversion failed: {e}");
            std::process::exit(1);
        }
    }
}

/// List cached models and available models from the registry.
fn models_cmd() {
    let cached = cache::list_cached();
    if cached.is_empty() {
        println!("No cached models.");
        println!("Download one with: lumen pull <model-name>");
        println!();

        let reg = registry::load_registry();
        println!("Available models:");
        for entry in reg.list() {
            let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            println!("  {:<20} {} ({})", entry.key, entry.display_name, quants.join(", "));
        }
        return;
    }

    println!("Cached models:\n");
    for (name, _path, size) in &cached {
        println!("  {:<40} {}", name, cache::format_size(*size));
    }

    // Also show available (not yet cached) models.
    let reg = registry::load_registry();
    let cached_stems: Vec<&str> = cached.iter().map(|(name, _, _)| name.as_str()).collect();
    let mut available = Vec::new();
    for entry in reg.list() {
        for quant in entry.gguf_files.keys() {
            let stem = format!("{}-{}", entry.key, quant);
            if !cached_stems.contains(&stem.as_str()) {
                available.push((entry.key.clone(), entry.display_name.clone(), quant.clone()));
            }
        }
    }
    if !available.is_empty() {
        println!("\nAvailable to download:");
        for (key, display, quant) in &available {
            println!("  {:<20} {} {}", key, display, quant);
        }
        println!("\nDownload with: lumen pull <model-name> [--quant Q8_0]");
    }
}
