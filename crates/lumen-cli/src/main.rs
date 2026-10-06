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
#[cfg(test)]
mod build_script_tests;
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
            println!(
                "lumen {}",
                option_env!("LUMEN_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
            );
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
                quant_override = Some(
                    args.get(i)
                        .unwrap_or_else(|| {
                            eprintln!("Error: --quant requires a value (e.g. Q8_0, Q4_0, BF16)");
                            std::process::exit(1);
                        })
                        .clone(),
                );
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
            eprintln!("  {}", entry.pull_tags().join(", "));
        }
        std::process::exit(1);
    });

    // Parse model:quant tag syntax (e.g., "qwen3.5-9b:q4_0").
    let (resolved_name, tag_quant) = registry::split_model_tag(model_name);

    let entry = reg.resolve(resolved_name).unwrap_or_else(|| {
        eprintln!("Unknown model: {resolved_name}");
        eprintln!("\nAvailable models:");
        for e in reg.list() {
            eprintln!("  {}", e.key);
        }
        std::process::exit(1);
    });

    if let Some(checkpoint) = &entry.checkpoint {
        if tag_quant.is_some() || quant_override.is_some() {
            eprintln!(
                "{} comes in one form and takes no quantization: lumen pull {resolved_name}",
                entry.display_name
            );
            std::process::exit(1);
        }
        pull_image(entry, checkpoint, resolved_name, skip_confirm);
        return;
    }

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
        eprintln!(
            "Multiple quantizations available for {}:\n",
            entry.display_name
        );
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
        let mut available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
        available.sort();
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
    println!(
        "Run with: lumen run {}:{} \"Write a haiku about light\"",
        resolved_name,
        quant.to_lowercase()
    );
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
fn pull_download_gguf_shards(
    _src: &registry::GgufSource,
    _skip_confirm: bool,
) -> std::path::PathBuf {
    eprintln!("Error: download support is not compiled in.");
    eprintln!("Rebuild with: cargo build --release --features download");
    std::process::exit(1);
}

/// Download a GGUF file, returning its path. Exits on failure.
#[cfg(feature = "download")]
fn pull_download_gguf(repo: &str, filename: &str, skip_confirm: bool) -> std::path::PathBuf {
    // Check if GGUF is already cached. Reclaim first even on a hit: the
    // per-process partial files crashed downloads left behind are otherwise
    // never cleaned once the file is published (every later call returns
    // here). Cheap: one read_dir; deletion needs stale mtime + liveness.
    if let Ok((_, local_name)) = download::split_repo_path(filename) {
        download::reclaim_stale_parts(&cache::cache_dir(), &local_name);
    }
    if let Some(existing) = cache::cached_gguf(filename) {
        eprintln!("GGUF already downloaded: {}", existing.display());
        return existing;
    }

    cache::ensure_cache_dir().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    download::download_gguf(repo, filename, &cache::cache_dir(), skip_confirm).unwrap_or_else(|e| {
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

/// Download the image model's pinned checkpoint and convert it into the
/// `.lbi` files the image-only `lumen-server` serves. Exits on failure.
///
/// The checkpoint lands in `<cache>/<key>/` in its own layout and the `.lbi`
/// in `<cache>/<key>/lbi/`. Once the conversion has been read back, the
/// component directories are removed: only `processor/` is read at run time.
/// A later pull downloads only what is missing, so an interrupted one is
/// continued, and a complete one is reported as cached. One pull of the
/// model runs at a time per cache: a second waits for the first.
#[cfg(feature = "download")]
fn pull_image(
    entry: &registry::ModelEntry,
    checkpoint: &registry::Checkpoint,
    name: &str,
    skip_confirm: bool,
) {
    use std::io::Write;

    if !cfg!(feature = "cuda") {
        eprintln!(
            "{} makes images on NVIDIA CUDA, and this lumen was built without CUDA; nothing was \
             downloaded.",
            entry.display_name
        );
        std::process::exit(1);
    }
    let dir = cache::image_checkpoint_dir(&entry.key);
    let lbi_dir = cache::image_lbi_dir(&entry.key);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
        eprintln!("Error: failed to create {}: {e}", dir.display());
        std::process::exit(1);
    });
    let _lock = lock_pull(&dir, &entry.display_name).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    // A conversion that died left its staging files; no other pull holds
    // the lock, so they are nobody's.
    lumen_image::convert::remove_staging(&lbi_dir).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    if cache::cached_image(&entry.key) {
        // An earlier pull may have stopped between publishing the conversion
        // and removing what it converted.
        if let Some(removed) = remove_image_sources(&dir) {
            eprintln!(
                "Removed the downloaded checkpoint files ({}) an earlier pull left behind.",
                cache::format_size(removed)
            );
        }
        println!("Already cached: {}", lbi_dir.display());
        return;
    }
    let converted = cache::IMAGE_SERVED_FILES
        .iter()
        .filter(|f| f.starts_with("lbi/"))
        .all(|f| std::fs::metadata(dir.join(f)).is_ok_and(|m| m.is_file() && m.len() > 0));
    // With the conversion done, only the tokenizer files can be missing.
    let needed: Vec<&registry::CheckpointFile> = checkpoint
        .files
        .iter()
        .filter(|f| !converted || f.path.starts_with("processor/"))
        .collect();
    let to_download: Vec<&registry::CheckpointFile> = needed
        .iter()
        .copied()
        .filter(|f| {
            !std::fs::metadata(dir.join(&f.path)).is_ok_and(|m| m.is_file() && m.len() == f.size)
        })
        .collect();
    let bytes: u64 = to_download.iter().map(|f| f.size).sum();
    if !to_download.is_empty() && !skip_confirm {
        let disk = if converted {
            format!("{} on disk", cache::format_size(bytes))
        } else {
            format!(
                "up to {} free during the conversion, {} kept",
                cache::format_size(conversion_peak_bytes(checkpoint)),
                cache::format_size(checkpoint.total_bytes())
            )
        };
        eprint!(
            "Download {} from {} ({}; {})? [Y/n] ",
            entry.display_name,
            checkpoint.repo,
            cache::format_size(bytes),
            disk
        );
        std::io::stderr().flush().ok();
        let mut input = String::new();
        // The end of the input is no answer: a pull with no terminal and no
        // --yes must not download.
        let answered = matches!(std::io::stdin().read_line(&mut input), Ok(n) if n > 0);
        let answer = input.trim();
        if !answered
            || !(answer.is_empty()
                || answer.eq_ignore_ascii_case("y")
                || answer.eq_ignore_ascii_case("yes"))
        {
            eprintln!("Download declined.");
            std::process::exit(1);
        }
    }
    for file in to_download {
        let dest_dir = dir.join(file.path.rsplit_once('/').map_or("", |(d, _)| d));
        std::fs::create_dir_all(&dest_dir).unwrap_or_else(|e| {
            eprintln!("Error: failed to create {}: {e}", dest_dir.display());
            std::process::exit(1);
        });
        let path = dir.join(&file.path);
        if path.exists() {
            // A file of another size is not the pinned one.
            std::fs::remove_file(&path).unwrap_or_else(|e| {
                eprintln!("Error: failed to remove {}: {e}", path.display());
                std::process::exit(1);
            });
        }
        download::download_pinned(&checkpoint.repo, &checkpoint.revision, file, &dest_dir)
            .unwrap_or_else(|e| {
                eprintln!("Download failed: {e}");
                std::process::exit(1);
            });
    }
    if !converted {
        eprintln!(
            "Converting to LBI: {} -> {}",
            dir.display(),
            lbi_dir.display()
        );
        let reports =
            lumen_image::convert::convert_checkpoint(&dir, &lbi_dir).unwrap_or_else(|e| {
                eprintln!("Conversion failed: {e}");
                std::process::exit(1);
            });
        for report in &reports {
            eprintln!(
                "  {:14} {:4} tensors  {}",
                report.component,
                report.tensor_count,
                cache::format_size(report.total_bytes)
            );
        }
    }
    if let Some(removed) = remove_image_sources(&dir) {
        eprintln!(
            "Removed the downloaded checkpoint files ({}); the converted files and processor/ are what the server reads.",
            cache::format_size(removed)
        );
    }
    println!("\nReady: {}", lbi_dir.display());
    println!("Serve with: lumen-server {name}");
}

/// The most disk a pull of `checkpoint` holds at once: every downloaded file
/// stays until the last component is converted, and a component's `.lbi` is
/// assembled from a blob file of the same size before the two are joined.
/// The `.lbi` hold the same tensor bytes the checkpoint does.
#[cfg(feature = "download")]
fn conversion_peak_bytes(checkpoint: &registry::Checkpoint) -> u64 {
    let component_bytes = |component: &str| -> u64 {
        checkpoint
            .files
            .iter()
            .filter(|f| f.path.starts_with(component) && f.path[component.len()..].starts_with('/'))
            .map(|f| f.size)
            .sum()
    };
    let mut converted = 0u64;
    let mut peak = 0u64;
    for component in lumen_image::convert::COMPONENTS {
        let bytes = component_bytes(component);
        peak = peak.max(converted + 2 * bytes);
        converted += bytes;
    }
    checkpoint.total_bytes() + peak
}

/// Remove the component directories a pull downloaded into `dir`, once their
/// conversion is published; the bytes removed, or None when there were none.
/// Exits on a failure to remove.
#[cfg(feature = "download")]
fn remove_image_sources(dir: &std::path::Path) -> Option<u64> {
    let mut removed = None;
    for component in lumen_image::convert::COMPONENTS {
        let comp_dir = dir.join(component);
        let Ok(entries) = std::fs::read_dir(&comp_dir) else {
            continue;
        };
        let bytes: u64 = entries
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        std::fs::remove_dir_all(&comp_dir).unwrap_or_else(|e| {
            eprintln!("Error: failed to remove {}: {e}", comp_dir.display());
            std::process::exit(1);
        });
        removed = Some(removed.unwrap_or(0) + bytes);
    }
    removed
}

/// Hold `dir`'s pull lock for the rest of the pull, waiting while another
/// lumen process pulls the same model into it: its downloads, conversion and
/// cleanup must not interleave with this one's. The lock file is never
/// written: it is opened without truncation and must be a regular file, so a
/// symbolic link planted in a shared cache is refused rather than followed
/// (a hard link is the same file and the same lock), and a file system
/// without locks is refused rather than pulled into unlocked.
#[cfg(feature = "download")]
fn lock_pull(dir: &std::path::Path, display_name: &str) -> Result<std::fs::File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    let path = dir.join("pull.lock");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .map_err(|e| match e.raw_os_error() {
            Some(libc::ELOOP) => format!("{} is a symbolic link; remove it", path.display()),
            _ => format!("failed to open {}: {e}", path.display()),
        })?;
    let meta = file
        .metadata()
        .map_err(|e| format!("failed to inspect {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    // SAFETY: flock on a descriptor this function owns.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(file);
    }
    let e = std::io::Error::last_os_error();
    if e.raw_os_error() != Some(libc::EWOULDBLOCK) {
        return Err(format!(
            "cannot lock {}: {e}; a pull needs a file system with file locks",
            path.display()
        ));
    }
    eprintln!("Waiting for another pull of {display_name} to finish...");
    // SAFETY: as above; a signal ends the wait early, so wait again.
    while unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::Interrupted {
            return Err(format!("failed to lock {}: {e}", path.display()));
        }
    }
    Ok(file)
}

#[cfg(not(feature = "download"))]
fn pull_image(
    _entry: &registry::ModelEntry,
    _checkpoint: &registry::Checkpoint,
    _name: &str,
    _skip_confirm: bool,
) {
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

    eprintln!(
        "Converting to LBC: {} -> {}",
        gguf_path.display(),
        lbc_out.display()
    );

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
    let reg = registry::load_registry();
    let mut cached = cache::list_cached();
    for entry in reg.list().into_iter().filter(|e| e.checkpoint.is_some()) {
        if cache::cached_image(&entry.key) {
            let lbi_dir = cache::image_lbi_dir(&entry.key);
            let size = std::fs::read_dir(&lbi_dir)
                .map(|rd| {
                    rd.flatten()
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);
            cached.push((entry.key.clone(), lbi_dir, size));
        }
    }
    if cached.is_empty() {
        println!("No cached models.");
        println!("Download one with: lumen pull <model-name>");
        println!();

        println!("Available models:");
        for entry in reg.list() {
            println!(
                "  {:<20} {} ({})",
                entry.key,
                entry.display_name,
                entry.variants()
            );
        }
        return;
    }

    println!("Cached models:\n");
    for (name, _path, size) in &cached {
        println!("  {:<40} {}", name, cache::format_size(*size));
    }

    // Also show available (not yet cached) models.
    let cached_stems: Vec<&str> = cached.iter().map(|(name, _, _)| name.as_str()).collect();
    let mut available = Vec::new();
    for entry in reg.list() {
        if entry.checkpoint.is_some() {
            if !cached_stems.contains(&entry.key.as_str()) {
                available.push((
                    entry.key.clone(),
                    entry.display_name.clone(),
                    entry.variants(),
                ));
            }
            continue;
        }
        let mut quants: Vec<&String> = entry.gguf_files.keys().collect();
        quants.sort();
        for quant in quants {
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

#[cfg(all(test, feature = "download"))]
mod pull_image_tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-pull-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_disk_peak_counts_every_source_plus_the_finished_and_staged_lbi() {
        // Qwen-Image-2.1 at the pinned commit: 33,120,164,845 bytes in all;
        // transformer 14,230,315,061, vae 1,350,991,591, text_encoder
        // 17,534,408,800. The peak is during the text encoder's conversion:
        // every source, the two finished .lbi, and the text encoder's blob
        // and part files, 83,770,289,097 bytes.
        let reg = registry::load_registry();
        let ckpt = reg
            .resolve("qwen-image")
            .unwrap()
            .checkpoint
            .as_ref()
            .unwrap();
        assert_eq!(conversion_peak_bytes(ckpt), 83_770_289_097);
    }

    #[test]
    fn removing_sources_keeps_the_processor_and_reports_what_went() {
        let dir = scratch("sources");
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(dir.join("transformer/config.json"), b"12345").unwrap();
        std::fs::create_dir_all(dir.join("processor")).unwrap();
        std::fs::write(dir.join("processor/vocab.json"), b"{}").unwrap();
        assert_eq!(remove_image_sources(&dir), Some(5));
        assert!(!dir.join("transformer").exists());
        assert!(dir.join("processor/vocab.json").is_file());
        assert_eq!(remove_image_sources(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_pull_waits_for_the_first() {
        let dir = scratch("lock");
        let first = lock_pull(&dir, "Test").expect("locks are supported here");
        assert_eq!(
            std::fs::read(dir.join("pull.lock")).unwrap(),
            b"",
            "the lock file is empty"
        );
        let dir2 = dir.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let second = std::thread::spawn(move || {
            let lock = lock_pull(&dir2, "Test");
            tx.send(std::time::Instant::now()).unwrap();
            lock
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            rx.try_recv().is_err(),
            "the second pull must not get the lock while the first holds it"
        );
        let released = std::time::Instant::now();
        drop(first);
        let got = rx.recv().unwrap();
        assert!(
            got >= released,
            "the second pull got the lock before the first released it"
        );
        assert!(second.join().unwrap().is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_symbolic_link_at_the_lock_path_is_refused_and_a_hard_link_is_the_same_lock() {
        let dir = scratch("lock-link");
        let target = dir.join("victim");
        std::fs::write(&target, b"do not truncate").unwrap();
        std::os::unix::fs::symlink(&target, dir.join("pull.lock")).unwrap();
        let err = lock_pull(&dir, "Test").unwrap_err();
        assert!(err.contains("symbolic link"), "{err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"do not truncate");
        std::fs::remove_file(dir.join("pull.lock")).unwrap();
        // A hard link (a backup made with `cp -al`, say) is the same file: the
        // lock is taken through it, nothing is written, and a pull through
        // the other name waits on the same lock.
        std::fs::hard_link(&target, dir.join("pull.lock")).unwrap();
        let held = lock_pull(&dir, "Test").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"do not truncate");
        let other = dir.join("other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::hard_link(&target, other.join("pull.lock")).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let lock = lock_pull(&other, "Test");
            tx.send(()).unwrap();
            lock
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            rx.try_recv().is_err(),
            "the other name must wait on the same lock"
        );
        drop(held);
        rx.recv().unwrap();
        assert!(waiter.join().unwrap().is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }
}
