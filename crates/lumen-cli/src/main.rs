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
mod image;
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
        "image" => image::image_cmd(&args[2..]),
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

/// `lumen pull` for the image model: [`fetch_image`], then where the model is
/// and what to run. Exits on failure.
#[cfg(feature = "download")]
fn pull_image(
    entry: &registry::ModelEntry,
    checkpoint: &registry::Checkpoint,
    name: &str,
    skip_confirm: bool,
) {
    if !cfg!(feature = "cuda") {
        eprintln!(
            "Error: {} makes images on NVIDIA CUDA, and this lumen was built without CUDA; \
             nothing was downloaded.",
            entry.display_name
        );
        std::process::exit(1);
    }
    let dir = cache::image_checkpoint_dir(&entry.key);
    let mut stdin = std::io::stdin().lock();
    let answers = (!skip_confirm).then_some(&mut stdin as &mut dyn std::io::BufRead);
    match fetch_image(&dir, checkpoint, &entry.display_name, answers) {
        Ok(true) => println!(
            "Already cached: {}",
            cache::image_lbi_dir(&entry.key).display()
        ),
        Ok(false) => {
            println!("\nReady: {}", cache::image_lbi_dir(&entry.key).display());
            println!("Make a picture: lumen image \"A red apple on a wooden table\"");
            println!("Serve it:       lumen-server {name}");
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

/// Download the image model's pinned checkpoint into `dir`, its directory in
/// the cache, and convert it into the `.lbi` files the image-only
/// `lumen-server` serves; true when it was cached already. With `answers`,
/// the user confirms a download first, answering from it.
///
/// The checkpoint keeps its own layout under `dir` and the `.lbi` go in
/// `dir/lbi/`. `dir` and everything in it must be this user's alone (see
/// [`yours_alone`]). Every file the pull downloads or reuses is checked
/// against its pinned size and SHA-256, and a conversion is reused only when
/// it passes the checks the image server starts with. Once the conversion is
/// published, the checkpoint's files outside `processor/`, the only part read
/// at run time, are removed. A later pull downloads only what is missing, so
/// an interrupted one is continued. One pull of the model writes to a cache at
/// a time: a second waits for the first. On a cache mounted read-only, a pull
/// only reports a model that is cached there.
#[cfg(feature = "download")]
fn fetch_image(
    dir: &std::path::Path,
    checkpoint: &registry::Checkpoint,
    display_name: &str,
    answers: Option<&mut dyn std::io::BufRead>,
) -> Result<bool, String> {
    let lbi_dir = dir.join("lbi");
    let root = dir
        .parent()
        .expect("a model's directory is inside the cache");
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o755)
            .create(root)
            .map_err(|e| format!("failed to create {}: {e}", root.display()))?;
    }
    ensure_dir(dir)?;
    yours_alone(dir)?;
    let lock = lock_pull(dir, display_name)?;
    // Under the lock: another pull adds and removes files here as it goes.
    all_yours_alone(dir)?;
    for sub in ["lbi", "processor"]
        .into_iter()
        .chain(lumen_image::convert::COMPONENTS)
    {
        only_dir(&dir.join(sub))?;
    }
    // A conversion that died left its staging files; no other pull holds
    // the lock, so they are nobody's. A pull that cannot write leaves them,
    // and what follows, to the next one that can.
    if lock.is_some() {
        lumen_image::convert::remove_staging(&lbi_dir).map_err(|e| e.to_string())?;
    }
    let converted = converted(dir, &lbi_dir);
    // With the conversion done, only the tokenizer files are needed.
    let needed: Vec<&registry::CheckpointFile> = checkpoint
        .files
        .iter()
        .filter(|f| !converted || f.path.starts_with("processor/"))
        .collect();
    let needed_bytes: u64 = needed.iter().map(|f| f.size).sum();
    let mut to_download = Vec::new();
    for file in needed {
        if !reusable(dir, file)? {
            to_download.push(file);
        }
    }
    if converted && to_download.is_empty() {
        // An earlier pull may have stopped between publishing the conversion
        // and removing what it converted.
        if lock.is_some() {
            if let Some(removed) = remove_image_sources(dir, checkpoint)? {
                eprintln!(
                    "Removed the downloaded checkpoint files ({}) an earlier pull left behind.",
                    cache::format_size(removed)
                );
            }
        }
        return Ok(true);
    }
    if lock.is_none() {
        return Err(format!(
            "{} is on a file system mounted read-only and does not hold the image model; \
             set LUMEN_CACHE_DIR to a directory you can write",
            dir.display()
        ));
    }
    let bytes: u64 = to_download.iter().map(|f| f.size).sum();
    let disk = if converted {
        String::new()
    } else {
        format!(
            "; about {} of disk at the peak, {} kept",
            cache::format_size(conversion_peak_bytes(checkpoint)),
            cache::format_size(checkpoint.total_bytes())
        )
    };
    let what = format!(
        "{display_name} from {} ({}{disk})",
        checkpoint.repo,
        cache::format_size(bytes)
    );
    let asked = answers.is_some();
    if let Some(mut answers) = answers.filter(|_| !to_download.is_empty()) {
        if !download::confirm(&format!("Download {what}?"), &mut answers)
            .map_err(|e| e.to_string())?
        {
            return Err(download::DownloadError::UserDeclined.to_string());
        }
    }
    // The disk the download and the conversion need is checked before the
    // first byte; the containers of a conversion the server would refuse are
    // removed next, so their room counts as free.
    let free_needed = if converted {
        bytes
    } else {
        conversion_peak_bytes(checkpoint) - (needed_bytes - bytes)
    };
    let removable: u64 = if converted {
        0
    } else {
        lumen_image::convert::COMPONENTS
            .iter()
            .filter_map(|c| std::fs::symlink_metadata(lbi_dir.join(format!("{c}.lbi"))).ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    };
    let available = available_bytes(dir)?.saturating_add(removable);
    if available < free_needed {
        return Err(format!(
            "{display_name} needs {} free in {}; {} is available. Free some space or set \
             LUMEN_CACHE_DIR to a larger disk",
            cache::format_size(free_needed),
            dir.display(),
            cache::format_size(available)
        ));
    }
    if !asked && !to_download.is_empty() {
        eprintln!("Downloading {what}");
    }
    if !converted {
        // A conversion the server would refuse is made again, as a whole set.
        for component in lumen_image::convert::COMPONENTS {
            remove_if_present(&lbi_dir.join(format!("{component}.lbi")))?;
        }
    }
    for file in to_download {
        let (sub, _) = file
            .path
            .split_once('/')
            .expect("checkpoint paths name a directory");
        let dest_dir = dir.join(sub);
        ensure_dir(&dest_dir)?;
        // A file there is not the pinned one.
        remove_if_present(&dir.join(&file.path))?;
        download::download_pinned(&checkpoint.repo, &checkpoint.revision, file, &dest_dir)
            .map_err(|e| format!("download failed: {e}"))?;
    }
    if !converted {
        ensure_dir(&lbi_dir)?;
        eprintln!(
            "Converting to LBI: {} -> {}",
            dir.display(),
            lbi_dir.display()
        );
        lumen_image::convert::convert_checkpoint(dir, &lbi_dir, |report| {
            eprintln!(
                "  {:14} {:4} tensors  {}",
                report.component,
                report.tensor_count,
                cache::format_size(report.total_bytes)
            );
        })
        .map_err(|e| format!("conversion failed: {e}"))?;
    }
    if let Some(removed) = remove_image_sources(dir, checkpoint)? {
        eprintln!(
            "Removed the downloaded checkpoint files ({}); the converted files and \
             processor/ are what the server reads.",
            cache::format_size(removed)
        );
    }
    Ok(false)
}

/// `path` as a directory of the image cache, created writable by its owner
/// alone when missing, also when another pull creates it at the same moment;
/// see [`only_dir`].
#[cfg(feature = "download")]
fn ensure_dir(path: &std::path::Path) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;
    match std::fs::DirBuilder::new().mode(0o755).create(path) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => {
            Err(format!("failed to create {}: {e}", path.display()))
        }
        _ => only_dir(path),
    }
}

/// Refuse anything but a directory at `path`, if there is something: a
/// symbolic link there would send the pull's writes and removals elsewhere.
#[cfg(feature = "download")]
fn only_dir(path: &std::path::Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(m) if !m.is_dir() => Err(format!(
            "{} is not a directory (a symbolic link or a file); to keep the image model \
             elsewhere, set LUMEN_CACHE_DIR there instead of linking",
            path.display()
        )),
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(format!("failed to inspect {}: {e}", path.display()))
        }
        _ => Ok(()),
    }
}

/// Refuse `path` when another user owns it, or can write it as a directory:
/// they could otherwise plant or swap, while a pull runs, what sends its
/// writes and removals outside the cache or what it converts. A link is
/// judged by itself, not by its target.
#[cfg(feature = "download")]
fn yours_alone(path: &std::path::Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::symlink_metadata(path)
        .map_err(|e| format!("failed to inspect {}: {e}", path.display()))?;
    // SAFETY: geteuid has no failure mode.
    if m.uid() != unsafe { libc::geteuid() } {
        return Err(format!(
            "{} belongs to another user; the image model's cache directory must be yours alone",
            path.display()
        ));
    }
    if m.is_dir() && m.mode() & 0o022 != 0 {
        return Err(format!(
            "{} can be written by its group or by others; make it yours alone: chmod go-w {}",
            path.display(),
            path.display()
        ));
    }
    Ok(())
}

/// [`yours_alone`] for `path` and everything in it, links not followed.
#[cfg(feature = "download")]
fn all_yours_alone(path: &std::path::Path) -> Result<(), String> {
    yours_alone(path)?;
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.is_dir()) {
        for entry in std::fs::read_dir(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?
        {
            let entry = entry.map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            all_yours_alone(&entry.path())?;
        }
    }
    Ok(())
}

/// Whether `file` is at its path under `dir` as pinned: a regular file of
/// its size and SHA-256.
#[cfg(feature = "download")]
fn reusable(dir: &std::path::Path, file: &registry::CheckpointFile) -> Result<bool, String> {
    let path = dir.join(&file.path);
    Ok(
        std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_file() && m.len() == file.size)
            && download::compute_sha256(&path).map_err(|e| e.to_string())? == file.sha256,
    )
}

/// Whether `lbi_dir` holds a conversion the image server on CUDA accepts: the
/// three converted files, regular files, pass the checks it starts with.
#[cfg(feature = "download")]
fn converted(dir: &std::path::Path, lbi_dir: &std::path::Path) -> bool {
    lumen_image::convert::COMPONENTS.iter().all(|component| {
        std::fs::symlink_metadata(lbi_dir.join(format!("{component}.lbi")))
            .is_ok_and(|m| m.is_file())
    }) && lumen_image::pipeline::PipelinePaths::from_roots(lbi_dir, dir)
        .check_components(true)
        .is_ok()
}

/// Remove the file or link at `path`, if there is one.
#[cfg(feature = "download")]
fn remove_if_present(path: &std::path::Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(format!("failed to remove {}: {e}", path.display()))
        }
        _ => Ok(()),
    }
}

/// The bytes an unprivileged user may still write on the file system that
/// holds `dir`.
#[cfg(feature = "download")]
fn available_bytes(dir: &std::path::Path) -> Result<u64, String> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).expect("a path holds no NUL");
    // SAFETY: a NUL-terminated path, and a statvfs the call fills in.
    let mut fs: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut fs) } != 0 {
        return Err(format!(
            "cannot read the free space of {}: {}",
            dir.display(),
            std::io::Error::last_os_error()
        ));
    }
    #[allow(clippy::useless_conversion)]
    Ok(u64::from(fs.f_bavail).saturating_mul(u64::from(fs.f_frsize)))
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

/// Remove the checkpoint files a pull downloads into `dir` outside
/// `processor/`, once their conversion is published, then each component
/// directory if that leaves it empty; the bytes removed, or None when there
/// were none. Nothing else is touched.
#[cfg(feature = "download")]
fn remove_image_sources(
    dir: &std::path::Path,
    checkpoint: &registry::Checkpoint,
) -> Result<Option<u64>, String> {
    let mut removed = None;
    for file in checkpoint
        .files
        .iter()
        .filter(|f| !f.path.starts_with("processor/"))
    {
        let path = dir.join(&file.path);
        if let Ok(m) = std::fs::symlink_metadata(&path) {
            remove_if_present(&path)?;
            removed = Some(removed.unwrap_or(0) + m.len());
        }
    }
    for component in lumen_image::convert::COMPONENTS {
        // A directory holding anything else stays.
        let _ = std::fs::remove_dir(dir.join(component));
    }
    Ok(removed)
}

/// Whether the file system holding `dir` is mounted read-only.
#[cfg(feature = "download")]
fn mounted_read_only(dir: &std::path::Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).expect("a path holds no NUL");
    // SAFETY: a NUL-terminated path, and a statvfs the call fills in.
    let mut fs: libc::statvfs = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::statvfs(path.as_ptr(), &mut fs) };
    status == 0 && fs.f_flag & libc::ST_RDONLY != 0
}

/// Hold `dir`'s pull lock for the rest of the pull, waiting while another
/// lumen process pulls the same model into it: its downloads, conversion and
/// cleanup must not interleave with this one's. The lock file is created
/// readable by its owner alone. It is never written: it is opened without
/// truncation and must be a regular file, so a symbolic link there is refused
/// rather than followed (a hard link is the same file and the same lock), and
/// a file system without locks is refused rather than pulled into unlocked.
/// On a file system mounted read-only there is no lock to take, and the pull
/// writes nothing there: it reports a model that is cached and otherwise stops
/// before downloading. A lock file that alone refuses writing is refused.
#[cfg(feature = "download")]
fn lock_pull(dir: &std::path::Path, display_name: &str) -> Result<Option<std::fs::File>, String> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    let path = dir.join("pull.lock");
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
    {
        Ok(file) => file,
        Err(e) => {
            return match e.raw_os_error() {
                Some(libc::EROFS) if mounted_read_only(dir) => Ok(None),
                Some(libc::ELOOP) => {
                    Err(format!("{} is a symbolic link; remove it", path.display()))
                }
                _ => Err(format!("failed to open {}: {e}", path.display())),
            }
        }
    };
    let meta = file
        .metadata()
        .map_err(|e| format!("failed to inspect {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    // SAFETY: flock on a descriptor this function owns.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(file));
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
    Ok(Some(file))
}

#[cfg(not(feature = "download"))]
fn fetch_image(
    _dir: &std::path::Path,
    _checkpoint: &registry::Checkpoint,
    _display_name: &str,
    _answers: Option<&mut dyn std::io::BufRead>,
) -> Result<bool, String> {
    Err(
        "download support is not compiled in; rebuild with: cargo build --release --features cuda"
            .to_string(),
    )
}

#[cfg(not(feature = "download"))]
fn pull_image(
    _entry: &registry::ModelEntry,
    _checkpoint: &registry::Checkpoint,
    _name: &str,
    _skip_confirm: bool,
) {
    eprintln!("Error: download support is not compiled in.");
    eprintln!("Rebuild with: cargo build --release --features cuda");
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

    /// `path` and its missing parents, as the pull creates them whatever the
    /// umask.
    fn mkdirs(path: &std::path::Path) {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o755)
            .create(path)
            .unwrap();
    }

    fn no_files() -> registry::Checkpoint {
        registry::Checkpoint {
            repo: "org/repo".to_owned(),
            revision: "abc".to_owned(),
            files: Vec::new(),
        }
    }

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
    fn removing_sources_takes_only_the_checkpoint_files_outside_the_tokenizer() {
        let dir = scratch("sources");
        let checkpoint = registry::Checkpoint {
            repo: "org/repo".to_owned(),
            revision: "abc".to_owned(),
            files: [
                "transformer/config.json",
                "vae/config.json",
                "processor/vocab.json",
            ]
            .iter()
            .map(|path| registry::CheckpointFile {
                path: (*path).to_owned(),
                size: 5,
                sha256: String::new(),
            })
            .collect(),
        };
        for d in ["transformer", "vae", "processor"] {
            std::fs::create_dir_all(dir.join(d)).unwrap();
        }
        std::fs::write(dir.join("transformer/config.json"), b"12345").unwrap();
        std::fs::write(dir.join("vae/config.json"), b"123").unwrap();
        std::fs::write(dir.join("vae/notes.txt"), b"mine").unwrap();
        std::fs::write(dir.join("processor/vocab.json"), b"{}").unwrap();

        assert_eq!(remove_image_sources(&dir, &checkpoint), Ok(Some(8)));
        assert!(!dir.join("transformer").exists(), "emptied, so removed");
        assert_eq!(std::fs::read(dir.join("vae/notes.txt")).unwrap(), b"mine");
        assert!(!dir.join("vae/config.json").exists());
        assert!(dir.join("processor/vocab.json").is_file());
        assert_eq!(remove_image_sources(&dir, &checkpoint), Ok(None));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_cache_directory_is_created_or_accepted_but_never_a_link() {
        let dir = scratch("cache-dir");
        let made = dir.join("made");
        assert_eq!(ensure_dir(&made), Ok(()));
        assert!(made.is_dir());
        assert_eq!(
            ensure_dir(&made),
            Ok(()),
            "an existing directory is accepted"
        );
        assert_eq!(
            only_dir(&dir.join("absent")),
            Ok(()),
            "nothing there is fine"
        );
        let elsewhere = dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let link = dir.join("link");
        std::os::unix::fs::symlink(&elsewhere, &link).unwrap();
        assert!(ensure_dir(&link).unwrap_err().contains("not a directory"));
        assert!(only_dir(&link).unwrap_err().contains("not a directory"));
        std::fs::write(dir.join("file"), b"x").unwrap();
        assert!(only_dir(&dir.join("file"))
            .unwrap_err()
            .contains("not a directory"));
        // Two pulls creating the same directory at once both go on.
        let race = dir.join("race");
        let other = race.clone();
        let t = std::thread::spawn(move || ensure_dir(&other));
        assert_eq!(ensure_dir(&race), Ok(()));
        assert_eq!(t.join().unwrap(), Ok(()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_model_directory_others_can_write_or_with_a_linked_subdirectory_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let set = |p: &std::path::Path, mode: u32| {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap()
        };
        let root = scratch("owner");
        let dir = root.join("model");
        assert_eq!(ensure_dir(&dir), Ok(()));
        assert_eq!(
            yours_alone(&dir),
            Ok(()),
            "created writable by its owner alone"
        );
        set(&dir, 0o775);
        assert!(yours_alone(&dir).unwrap_err().contains("chmod go-w"));
        set(&dir, 0o757);
        assert!(yours_alone(&dir).unwrap_err().contains("chmod go-w"));
        assert!(fetch_image(&dir, &no_files(), "Test", None)
            .unwrap_err()
            .contains("chmod go-w"));
        assert!(
            !dir.join("pull.lock").exists(),
            "no lock file is made in a directory others can write"
        );
        set(&dir, 0o755);
        for sub in [
            "lbi",
            "processor",
            "transformer",
            "vae",
            "text_encoder",
            "vae/work",
        ] {
            mkdirs(&dir.join(sub));
            set(&dir.join(sub), 0o775);
            assert!(
                fetch_image(&dir, &no_files(), "Test", None)
                    .unwrap_err()
                    .contains("chmod go-w"),
                "{sub}"
            );
            set(&dir.join(sub), 0o755);
        }
        std::fs::remove_dir(dir.join("vae/work")).unwrap();
        for sub in ["lbi", "processor", "transformer", "vae", "text_encoder"] {
            std::fs::remove_dir(dir.join(sub)).unwrap();
        }
        // A link in place of the model's directory itself.
        let target = root.join("elsewhere-model");
        mkdirs(&target);
        let linked = root.join("linked-model");
        std::os::unix::fs::symlink(&target, &linked).unwrap();
        assert!(fetch_image(&linked, &no_files(), "Test", None)
            .unwrap_err()
            .contains("not a directory"));
        assert_eq!(
            std::fs::read_dir(&target).unwrap().count(),
            0,
            "nothing written through the link"
        );
        for sub in ["lbi", "processor", "transformer", "vae", "text_encoder"] {
            let elsewhere = root.join(format!("elsewhere-{sub}"));
            mkdirs(&elsewhere);
            std::fs::write(elsewhere.join("vae.lbi"), b"not the cache's").unwrap();
            std::fs::write(elsewhere.join("vae.lbi.part"), b"not the cache's").unwrap();
            std::os::unix::fs::symlink(&elsewhere, dir.join(sub)).unwrap();
            assert!(fetch_image(&dir, &no_files(), "Test", None)
                .unwrap_err()
                .contains("not a directory"));
            assert!(
                elsewhere.join("vae.lbi").is_file() && elsewhere.join("vae.lbi.part").is_file(),
                "nothing removed through the link at {sub}"
            );
            std::fs::remove_file(dir.join(sub)).unwrap();
        }
        // SAFETY: geteuid has no failure mode.
        if unsafe { libc::geteuid() } != 0 {
            assert!(yours_alone(std::path::Path::new("/"))
                .unwrap_err()
                .contains("belongs to another user"));
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// A small checkpoint pinned to its own bytes: one F32 tensor per
    /// component and tokenizer files, every one already downloaded, so the
    /// pull converts without the network.
    fn local_checkpoint(dir: &std::path::Path) -> registry::Checkpoint {
        let mut files = Vec::new();
        let mut add = |path: &str, bytes: &[u8]| {
            let full = dir.join(path);
            mkdirs(full.parent().unwrap());
            std::fs::write(&full, bytes).unwrap();
            files.push(registry::CheckpointFile {
                path: path.to_owned(),
                size: bytes.len() as u64,
                sha256: download::compute_sha256(&full).unwrap(),
            });
        };
        for component in lumen_image::convert::COMPONENTS {
            add(&format!("{component}/config.json"), br#"{"probe":true}"#);
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&[1, 2, 3, 4]);
            add(&format!("{component}/model.safetensors"), &bytes);
        }
        for name in ["vocab.json", "merges.txt", "added_tokens.json"] {
            add(&format!("processor/{name}"), b"{}");
        }
        registry::Checkpoint {
            repo: "org/repo".to_owned(),
            revision: "abc".to_owned(),
            files,
        }
    }

    #[test]
    fn a_pull_with_every_file_present_converts_and_keeps_only_what_the_server_reads() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch("pull");
        // A cache directory the user's group can write, as umask 002 leaves it.
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o775)).unwrap();
        let dir = root.join("qwen-image-2-1");
        let checkpoint = local_checkpoint(&dir);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        // A file the user's group can write, as umask 002 leaves it, and a
        // link of the user's to a directory of root's.
        std::fs::set_permissions(
            dir.join("processor/vocab.json"),
            std::fs::Permissions::from_mode(0o664),
        )
        .unwrap();
        std::os::unix::fs::symlink("/", dir.join("root")).unwrap();
        // What an earlier, dead conversion and the user left there.
        let lbi = dir.join("lbi");
        mkdirs(&lbi);
        std::fs::write(lbi.join("vae.lbi"), b"garbage").unwrap();
        let (part, tmp) = lumen_image::lbi::staging_paths(&lbi.join("text_encoder.lbi"));
        std::fs::write(&part, b"stale").unwrap();
        std::fs::write(&tmp, b"stale").unwrap();
        std::fs::write(dir.join("vae/notes.txt"), b"mine").unwrap();

        // Every file is there, so nothing is asked: a "no" is never read.
        let mut no = &b"n\n"[..];
        assert_eq!(
            fetch_image(&dir, &checkpoint, "Test", Some(&mut no)),
            Ok(false)
        );

        let mut left: Vec<String> = std::fs::read_dir(&lbi)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["text_encoder.lbi", "transformer.lbi", "vae.lbi"]);
        for component in lumen_image::convert::COMPONENTS {
            let file = lumen_image::lbi::LbiFile::open(&lbi.join(format!("{component}.lbi")));
            assert_eq!(file.unwrap().len(), 1, "{component} converted again");
        }
        assert!(!dir.join("transformer").exists() && !dir.join("text_encoder").exists());
        assert_eq!(std::fs::read(dir.join("vae/notes.txt")).unwrap(), b"mine");
        assert!(!dir.join("vae/model.safetensors").exists());
        assert!(dir.join("processor/vocab.json").is_file());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_failed_conversion_leaves_no_converted_file_from_before() {
        let root = scratch("pull-fails");
        let dir = root.join("qwen-image-2-1");
        let mut checkpoint = local_checkpoint(&dir);
        let source = dir.join("transformer/model.safetensors");
        std::fs::write(&source, b"not safetensors").unwrap();
        let pin = checkpoint
            .files
            .iter_mut()
            .find(|f| f.path == "transformer/model.safetensors")
            .unwrap();
        pin.size = std::fs::metadata(&source).unwrap().len();
        pin.sha256 = download::compute_sha256(&source).unwrap();
        mkdirs(&dir.join("lbi"));
        std::fs::write(dir.join("lbi/vae.lbi"), b"garbage").unwrap();

        let err = fetch_image(&dir, &checkpoint, "Test", None).unwrap_err();
        assert!(err.contains("conversion failed"), "{err}");
        assert!(!dir.join("lbi/vae.lbi").exists());
        assert!(source.is_file(), "sources stay for the next pull");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_first_pull_creates_the_cache_and_the_model_directory() {
        let root = scratch("fresh");
        let dir = root.join("home/.cache/lumen/qwen-image-2-1");
        let err = fetch_image(&dir, &no_files(), "Test", None).unwrap_err();
        assert!(err.contains("conversion failed"), "{err}");
        assert!(dir.join("lbi").is_dir());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_pull_asks_before_downloading_what_is_missing_or_not_as_pinned() {
        let root = scratch("pull-asks");
        let dir = root.join("qwen-image-2-1");
        let checkpoint = local_checkpoint(&dir);
        let declined = Err(download::DownloadError::UserDeclined.to_string());
        mkdirs(&dir.join("lbi"));
        std::fs::write(dir.join("lbi/vae.lbi"), b"refused by the server").unwrap();
        let (part, _) = lumen_image::lbi::staging_paths(&dir.join("lbi/text_encoder.lbi"));
        std::fs::write(&part, b"a dead conversion's").unwrap();
        let mut no = &b"n\n"[..];
        std::fs::write(dir.join("processor/vocab.json"), b"[]").unwrap();
        assert_eq!(
            fetch_image(&dir, &checkpoint, "Test", Some(&mut no)),
            declined,
            "a file with other bytes"
        );
        std::fs::write(dir.join("processor/vocab.json"), b"{}").unwrap();
        std::fs::remove_file(dir.join("transformer/model.safetensors")).unwrap();
        let mut no = &b"n\n"[..];
        assert_eq!(
            fetch_image(&dir, &checkpoint, "Test", Some(&mut no)),
            declined,
            "a missing file"
        );
        assert_eq!(
            std::fs::read(dir.join("lbi/vae.lbi")).unwrap(),
            b"refused by the server",
            "a declined pull keeps the conversion"
        );
        assert!(
            !part.exists(),
            "a dead conversion's staging file is removed"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_pull_refuses_a_download_the_disk_cannot_hold_before_its_first_byte() {
        let root = scratch("pull-disk");
        let dir = root.join("qwen-image-2-1");
        let mut checkpoint = local_checkpoint(&dir);
        checkpoint.files.push(registry::CheckpointFile {
            path: "transformer/huge.safetensors".to_owned(),
            size: 1 << 60,
            sha256: "0".repeat(64),
        });
        let err = fetch_image(&dir, &checkpoint, "Test", None).unwrap_err();
        assert!(
            err.contains("free in") && err.contains("LUMEN_CACHE_DIR"),
            "{err}"
        );
        assert!(!dir.join("transformer/huge.safetensors").exists());
        assert!(!dir.join("lbi").exists(), "nothing converted");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_pull_waits_while_another_pull_of_the_model_runs() {
        let root = scratch("pull-waits");
        let dir = root.join("qwen-image-2-1");
        let checkpoint = local_checkpoint(&dir);
        let other = lock_pull(&dir, "Test").expect("locks are supported here");
        // The holder's work in progress, which the waiting pull must not judge.
        mkdirs(&dir.join("vae/work"));
        std::fs::set_permissions(
            dir.join("vae/work"),
            std::os::unix::fs::PermissionsExt::from_mode(0o777),
        )
        .unwrap();
        let waiting = dir.clone();
        let pull = std::thread::spawn(move || fetch_image(&waiting, &checkpoint, "Test", None));
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            !pull.is_finished() && !dir.join("lbi").exists(),
            "the pull must not start while another holds the model"
        );
        std::fs::remove_dir(dir.join("vae/work")).unwrap();
        drop(other);
        assert_eq!(pull.join().unwrap(), Ok(false));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_file_is_reused_only_with_its_pinned_size_and_hash() {
        let dir = scratch("reusable");
        std::fs::create_dir_all(dir.join("processor")).unwrap();
        let pinned = registry::CheckpointFile {
            path: "processor/vocab.json".to_owned(),
            size: 18,
            sha256: download::sha256_of_reader(&mut &b"stored model bytes"[..]).unwrap(),
        };
        let path = dir.join(&pinned.path);
        assert_eq!(reusable(&dir, &pinned), Ok(false), "absent");
        std::fs::write(&path, b"stored model bytes").unwrap();
        assert_eq!(reusable(&dir, &pinned), Ok(true));
        std::fs::write(&path, b"stored model bytez").unwrap();
        assert_eq!(reusable(&dir, &pinned), Ok(false), "same size, other bytes");
        std::fs::write(&path, b"stored model bytes!").unwrap();
        assert_eq!(reusable(&dir, &pinned), Ok(false), "other size");
        // A link of the file's length to a file with its bytes.
        std::fs::write(
            dir.join("processor/abcdefghijklmnopqr"),
            b"stored model bytes",
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink("abcdefghijklmnopqr", &path).unwrap();
        assert_eq!(reusable(&dir, &pinned), Ok(false), "a link is not the file");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_conversion_is_reused_only_when_the_server_would_accept_it() {
        let dir = scratch("converted");
        let lbi = dir.join("lbi");
        std::fs::create_dir_all(&lbi).unwrap();
        assert!(!converted(&dir, &lbi), "nothing converted");
        for component in lumen_image::convert::COMPONENTS {
            std::fs::write(lbi.join(format!("{component}.lbi")), b"garbage").unwrap();
        }
        assert!(!converted(&dir, &lbi), "files that do not open");
        for component in lumen_image::convert::COMPONENTS {
            let path = lbi.join(format!("{component}.lbi"));
            lumen_image::lbi::LbiWriter::create(&path, serde_json::json!({}))
                .unwrap()
                .finish()
                .unwrap();
        }
        assert!(
            !converted(&dir, &lbi),
            "containers that open but hold no model"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_pull_waits_for_the_first() {
        let dir = scratch("lock");
        assert!(
            !mounted_read_only(&dir),
            "a scratch directory can be written"
        );
        let first = lock_pull(&dir, "Test").expect("locks are supported here");
        assert_eq!(
            std::fs::read(dir.join("pull.lock")).unwrap(),
            b"",
            "the lock file is empty"
        );
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join("pull.lock"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "no other user can open it to hold it");
        }
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
        // Readable by others, as an existing lock file may be.
        std::fs::set_permissions(&target, std::os::unix::fs::PermissionsExt::from_mode(0o644))
            .unwrap();
        std::os::unix::fs::symlink(&target, dir.join("pull.lock")).unwrap();
        let err = lock_pull(&dir, "Test").unwrap_err();
        assert!(err.contains("is a symbolic link; remove it"), "{err}");
        // A named pipe there neither blocks the open nor passes for the lock.
        let pipe_dir = dir.join("pipe");
        std::fs::create_dir_all(&pipe_dir).unwrap();
        let fifo = std::ffi::CString::new(pipe_dir.join("pull.lock").to_str().unwrap()).unwrap();
        // SAFETY: a NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        assert!(lock_pull(&pipe_dir, "Test")
            .unwrap_err()
            .contains("not a regular file"));
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
