//! Download GGUF models from HuggingFace with integrity verification.
//!
//! All code in this module is gated behind `#[cfg(feature = "download")]`.
//! When the feature is disabled, only the `sanitize_filename` function
//! (used for path traversal prevention) is available.

/// Split a registry-declared GGUF path into `(url_path, local_basename)`.
///
/// Registry entries may nest shards under a repo subdirectory (e.g.
/// `"Qwen_Qwen3.6-27B-bf16/Qwen_Qwen3.6-27B-bf16-00001-of-00002.gguf"`).
/// The subdirectory is used only on the URL side; locally every shard is
/// cached flat under its basename so multi-shard siblings stay adjacent
/// (which is what the multi-shard reader's auto-discovery expects).
///
/// Every path segment is individually validated with [`sanitize_filename`]
/// (rejects `".."`, null bytes, control characters); backslashes and empty
/// segments (leading/trailing/double `/`) are rejected outright, so path
/// traversal cannot reach the filesystem or the URL.
pub fn split_repo_path(path: &str) -> Result<(String, String), String> {
    if path.contains('\\') {
        return Err(format!("path contains backslash: {path:?}"));
    }
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() > 4 {
        return Err(format!(
            "path nests too deep ({} segments): {path:?}",
            segments.len()
        ));
    }
    for seg in &segments {
        if seg.is_empty() {
            return Err(format!("path contains empty segment: {path:?}"));
        }
        sanitize_filename(seg)?;
    }
    let basename = segments[segments.len() - 1];
    Ok((path.to_owned(), basename.to_owned()))
}

/// Validate that a filename is safe for use as a cache key.
///
/// Rejects filenames containing path traversal sequences, directory separators,
/// null bytes, or control characters. Returns `Ok(())` if safe.
pub fn sanitize_filename(filename: &str) -> Result<(), String> {
    if filename.is_empty() {
        return Err("filename is empty".to_owned());
    }
    if filename.contains("..") {
        return Err(format!("filename contains path traversal: {filename:?}"));
    }
    if filename.contains('/') || filename.contains('\\') {
        return Err(format!(
            "filename contains directory separator: {filename:?}"
        ));
    }
    if filename.contains('\0') {
        return Err(format!("filename contains null byte: {filename:?}"));
    }
    // Reject control characters (0x00..0x1F, 0x7F).
    if filename.bytes().any(|b| b < 0x20 || b == 0x7F) {
        return Err(format!("filename contains control character: {filename:?}"));
    }
    Ok(())
}

#[cfg(feature = "download")]
mod inner {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    use super::sanitize_filename;

    /// Errors that can occur during download.
    #[derive(Debug)]
    pub enum DownloadError {
        /// User declined the download confirmation.
        UserDeclined,
        /// Network or I/O error.
        Io(String),
        /// Invalid filename (path traversal, etc.).
        InvalidFilename(String),
    }

    impl std::fmt::Display for DownloadError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                DownloadError::UserDeclined => write!(f, "download declined by user"),
                DownloadError::Io(msg) => write!(f, "{msg}"),
                DownloadError::InvalidFilename(msg) => write!(f, "invalid filename: {msg}"),
            }
        }
    }

    /// Get the file size from HuggingFace via a HEAD request.
    ///
    /// HF returns a 302 redirect to CDN; ureq follows it and we read Content-Length.
    fn get_remote_size(url: &str) -> Result<Option<u64>, DownloadError> {
        // ureq's HEAD doesn't read a body, but we need Content-Length from the
        // final response (after redirects). Use a GET with range 0-0 to get
        // Content-Range which tells us the total size.
        // Actually, ureq follows redirects by default. Let's try HEAD first.
        let resp = ureq::head(url)
            .call()
            .map_err(|e| DownloadError::Io(format!("HEAD request failed for {url}: {e}")))?;

        if let Some(cl) = resp.header("content-length") {
            if let Ok(size) = cl.parse::<u64>() {
                return Ok(Some(size));
            }
        }

        Ok(None)
    }

    /// Prompt the user for [Y/n] confirmation.
    ///
    /// Returns `true` if the user accepts (Enter or Y/y), `false` otherwise.
    fn confirm_download(
        repo: &str,
        filename: &str,
        size: Option<u64>,
    ) -> Result<bool, DownloadError> {
        let size_str = match size {
            Some(s) => crate::cache::format_size(s),
            None => "unknown size".to_owned(),
        };
        eprint!("Download {filename} from {repo} ({size_str})? [Y/n] ");
        std::io::stderr().flush().ok();

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| DownloadError::Io(format!("failed to read confirmation: {e}")))?;

        let trimmed = input.trim();
        Ok(trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("y")
            || trimmed.eq_ignore_ascii_case("yes"))
    }

    /// Download a GGUF file from HuggingFace.
    ///
    /// The file is downloaded to a `.part` temporary file, then atomically renamed
    /// to the final path after SHA-256 is computed and stored in a `.sha256` sidecar.
    ///
    /// If the final file already exists and is non-empty, this is a cache hit and
    /// the existing path is returned immediately.
    ///
    /// # Arguments
    /// - `repo`: HuggingFace repo (e.g. `"bartowski/Qwen2.5-3B-Instruct-GGUF"`)
    /// - `filename`: GGUF filename, optionally nested under a repo
    ///   subdirectory (e.g. `"Qwen2.5-3B-Instruct-Q8_0.gguf"` or
    ///   `"subdir/model-00001-of-00002.gguf"`). The subdirectory applies to
    ///   the download URL only; the local cache file is always the flat
    ///   basename so multi-shard siblings stay adjacent.
    /// - `dest_dir`: Directory to download into (typically the cache dir)
    /// - `skip_confirm`: If true, skip the `[Y/n]` prompt
    pub fn download_gguf(
        repo: &str,
        filename: &str,
        dest_dir: &Path,
        skip_confirm: bool,
    ) -> Result<PathBuf, DownloadError> {
        // Validate (traversal-safe) and split into URL path + local basename.
        let (url_path, local_name) =
            super::split_repo_path(filename).map_err(DownloadError::InvalidFilename)?;
        let filename = local_name.as_str();

        let final_path = dest_dir.join(filename);
        // The staging name carries the PID: two concurrent first-time
        // downloads of the same file must not clobber each other's .part
        // before the atomic rename. The .sha256 sidecar keeps its stable
        // name BY DESIGN: it is shared last-writer-wins metadata whose
        // content is identical for both racers (hash of the same URL's
        // bytes, same basename), written after the winner's rename, and
        // write-only in production (only its unit test reads it back).
        let sha_path = dest_dir.join(format!("{filename}.sha256"));
        // Reclaim BEFORE the cache-hit return: after one racer succeeds,
        // every future call takes the cache-hit fast path, so litter from
        // a SIGKILLed racer would otherwise never be reclaimed. The scan
        // is a small read_dir plus one libc::kill per stale candidate —
        // cheap since the subprocess-based check was replaced.
        reclaim_stale_parts(dest_dir, filename);

        // Cache hit: file already exists and is non-empty.
        if final_path.is_file() {
            if let Ok(meta) = std::fs::metadata(&final_path) {
                if meta.len() > 0 {
                    eprintln!("Cache hit: {}", final_path.display());
                    return Ok(final_path);
                }
            }
        }

        // Build the HuggingFace download URL (uses the full repo path, which
        // may include a subdirectory; the local file is the flat basename).
        let url = format!("https://huggingface.co/{repo}/resolve/main/{url_path}");

        // Get file size for confirmation and progress bar.
        let size = get_remote_size(&url)?;

        // Confirm with user unless --yes was passed.
        if !skip_confirm && !confirm_download(repo, filename, size)? {
            return Err(DownloadError::UserDeclined);
        }

        // Ensure dest dir exists.
        std::fs::create_dir_all(dest_dir).map_err(|e| {
            DownloadError::Io(format!("failed to create {}: {e}", dest_dir.display()))
        })?;

        // Start the download.
        eprintln!("Downloading: {url}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| DownloadError::Io(format!("GET request failed: {e}")))?;

        // Get content length from the actual response (might differ from HEAD due to CDN).
        let content_length = resp
            .header("content-length")
            .and_then(|cl| cl.parse::<u64>().ok())
            .or(size);

        // Set up progress bar.
        let pb = if let Some(total) = content_length {
            let pb = indicatif::ProgressBar::new(total);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                    .progress_chars("=>-"),
            );
            pb
        } else {
            let pb = indicatif::ProgressBar::new_spinner();
            pb.set_style(
                indicatif::ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
                    .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
            );
            pb
        };

        // Download to .part file. The guard removes OUR pid-named staging
        // file on every early-error return (network, write, flush, hash,
        // rename); it is defused only after the atomic rename succeeds —
        // the old fixed name self-overwrote, so without this the PID
        // scheme would turn each aborted multi-GB pull into invisible
        // litter no lumen command can reclaim.
        // Exclusive creation with a collision-retried nonce: PIDs are NOT
        // unique across PID namespaces (two containers sharing a cache
        // volume can both be namespace-local PID 1, giving both the same
        // pid-named path — a truncating create would resurrect the exact
        // clobber race, this time publishing a silently partial FINAL).
        // `create_new` (O_EXCL) makes the filesystem the arbiter; on a
        // name collision we retry with a fresh nonce rather than truncate.
        let (part_path, mut file) = create_exclusive_staging(dest_dir, filename)?;
        let own_dev_ino = {
            use std::os::unix::fs::MetadataExt;
            let m = file
                .metadata()
                .map_err(|e| DownloadError::Io(format!("fstat error on staging: {e}")))?;
            (m.dev(), m.ino())
        };
        let mut part_guard = StagingGuard {
            path: part_path.clone(),
            dev_ino: own_dev_ino,
            armed: true,
        };
        let mut reader = resp.into_reader();

        let mut buf = vec![0u8; 64 * 1024]; // 64 KB buffer
        let mut total_written: u64 = 0;

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| DownloadError::Io(format!("read error during download: {e}")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| DownloadError::Io(format!("write error: {e}")))?;
            total_written += n as u64;
            pb.set_position(total_written);
        }

        file.flush()
            .map_err(|e| DownloadError::Io(format!("flush error: {e}")))?;

        pb.finish_with_message("download complete");

        // Verify size if known.
        if let Some(expected) = content_length {
            if total_written != expected {
                // Guard cleans up the .part file on return.
                return Err(DownloadError::Io(format!(
                    "size mismatch: expected {expected} bytes, got {total_written} bytes"
                )));
            }
        }

        // Hash through OUR OWN file descriptor, never by reopening the
        // pathname: after an unlink (e.g. a reclaimer that judged this
        // transfer stalled) the NAME can be reused by a fresh exclusive
        // create, and a pathname reopen would hash — and then rename —
        // someone else's in-progress bytes.
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|e| DownloadError::Io(format!("seek error before hashing: {e}")))?;
        let hash = sha256_of_reader(&mut file)?;

        // Identity check before the by-path rename: the pathname must
        // still be OUR inode. If it is not (unlinked and possibly reused),
        // renaming would publish a stranger's partial file — disarm the
        // guard (the path is not ours to delete) and fail cleanly; a
        // retry re-downloads.
        {
            use std::os::unix::fs::MetadataExt;
            let path_dev_ino = std::fs::metadata(&part_path)
                .map(|m| (m.dev(), m.ino()))
                .ok();
            if path_dev_ino != Some(own_dev_ino) {
                part_guard.armed = false;
                return Err(DownloadError::Io(format!(
                    "staging file {} was unlinked or replaced during the \
                     download (a reclaimer judged this transfer stalled, or \
                     the cache dir was cleaned) — retry the download",
                    part_path.display()
                )));
            }
        }
        // The fd stays open through the rename: keeping it open prevents
        // inode recycling from blurring the identity we just verified.
        let file_kept_open = file;

        // Atomic rename FIRST: .part -> final. The sidecar follows, so a
        // published file is never newer than its sidecar by more than one
        // racer's window (contents are identical per URL either way).
        //
        // Rename FAILURE is cleaned up here, explicitly, while our fd is
        // still open: a `?` would drop `file_kept_open` before the guard's
        // Drop ran (reverse declaration order), letting the freed inode be
        // recycled and the guard's identity check pass on a stranger's
        // file. With the fd held, a path whose (dev, ino) matches ours IS
        // ours, so the delete is safe.
        if let Err(e) = std::fs::rename(&part_path, &final_path) {
            use std::os::unix::fs::MetadataExt;
            part_guard.armed = false;
            let still_ours = std::fs::metadata(&part_path)
                .map(|m| (m.dev(), m.ino()) == own_dev_ino)
                .unwrap_or(false);
            if still_ours {
                let _ = std::fs::remove_file(&part_path);
            }
            drop(file_kept_open);
            return Err(DownloadError::Io(format!(
                "failed to rename {} -> {}: {e}",
                part_path.display(),
                final_path.display()
            )));
        }
        part_guard.armed = false;
        drop(file_kept_open);

        // Write SHA-256 sidecar (shared name, last-writer-wins by design).
        std::fs::write(&sha_path, format!("{hash}  {filename}\n")).map_err(|e| {
            DownloadError::Io(format!("failed to write {}: {e}", sha_path.display()))
        })?;

        eprintln!("Saved: {} (SHA-256: {hash})", final_path.display());
        Ok(final_path)
    }

    /// Compute SHA-256 hash of a file. Returns the hex-encoded digest.
    pub fn compute_sha256(path: &Path) -> Result<String, DownloadError> {
        let mut file = std::fs::File::open(path).map_err(|e| {
            DownloadError::Io(format!(
                "failed to open {} for hashing: {e}",
                path.display()
            ))
        })?;
        sha256_of_reader(&mut file)
    }

    /// Streaming SHA-256 over an already-open reader — used by the
    /// download path to hash through its own descriptor (a pathname
    /// reopen could read a reused name's bytes after an unlink).
    pub fn sha256_of_reader<R: Read>(reader: &mut R) -> Result<String, DownloadError> {
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| DownloadError::Io(format!("read error during hashing: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex_encode(&hasher.finalize()))
    }

    /// Removes the caller's PID-named staging file unless defused (set
    /// `armed = false` after the atomic rename). Covers every early-error
    /// return and panic unwind in the download path.
    struct StagingGuard {
        path: std::path::PathBuf,
        /// (device, inode) of OUR staging file, captured at creation: the
        /// guard must never delete a stranger's file that reused our
        /// pathname after a reclaimer unlinked ours mid-download. The
        /// check-to-unlink window inside Drop is a microsecond-class
        /// TOCTOU (a replacement landing between metadata and remove_file)
        /// — an absolute guarantee needs serialized cleanup, which is
        /// deliberately out of scope; the residual is ledgered.
        dev_ino: (u64, u64),
        armed: bool,
    }

    impl Drop for StagingGuard {
        fn drop(&mut self) {
            if self.armed {
                use std::os::unix::fs::MetadataExt;
                let still_ours = std::fs::metadata(&self.path)
                    .map(|m| (m.dev(), m.ino()) == self.dev_ino)
                    .unwrap_or(false);
                if still_ours {
                    let _ = std::fs::remove_file(&self.path);
                }
            }
        }
    }

    /// Exclusive staging creation: opens `{base}.part`-style paths with
    /// `create_new` (O_EXCL), retrying with a fresh nonce on collision so
    /// two writers can never share (and truncate) one staging file — PIDs
    /// alone are not unique across PID namespaces. The final path shape is
    /// `{filename}.{pid}-{nonce}.part`.
    pub fn create_exclusive_staging(
        dest_dir: &Path,
        filename: &str,
    ) -> Result<(std::path::PathBuf, std::fs::File), DownloadError> {
        // Built by joining onto dest_dir — never by string-mangling the
        // full path, which breaks valid non-UTF-8 Unix cache directories.
        for attempt in 0u32..16 {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(attempt)
                .wrapping_add(attempt);
            let candidate =
                dest_dir.join(format!("{filename}.{}-{nonce}.part", std::process::id()));
            match std::fs::OpenOptions::new()
                .read(true) // the SAME fd is later re-read for hashing
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(f) => return Ok((candidate, f)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(DownloadError::Io(format!(
                        "failed to create staging {}: {e}",
                        candidate.display()
                    )))
                }
            }
        }
        Err(DownloadError::Io(
            "could not create a unique staging file after 16 attempts".into(),
        ))
    }

    /// Best-effort reclamation of `{filename}.<pid>[-<nonce>].part`
    /// stragglers from crashed runs. Deletion requires BOTH a stale mtime
    /// (>60s grace — a live writer refreshes mtime on every chunk, in any
    /// PID namespace) AND either ESRCH in our namespace or >24h staleness
    /// (pid numbers are namespace-local, so a foreign container's live
    /// writer can look dead here; mtime freshness is the cross-namespace
    /// protection). EPERM means alive under another user and keeps.
    /// Legacy fixed-name `{filename}.part` litter is age-gated at >1h —
    /// same mtime-freshness rationale.
    pub fn reclaim_stale_parts(dest_dir: &std::path::Path, filename: &str) {
        let Ok(entries) = std::fs::read_dir(dest_dir) else {
            return;
        };
        let prefix = format!("{filename}.");
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(rest) = name.strip_prefix(&prefix) else {
                continue;
            };
            let Some(pid) = rest.strip_suffix(".part") else {
                // Legacy pre-PID litter: exactly `{filename}.part`.
                // Reclaim only when stale by mtime (an old-binary
                // download could still be writing it; the old scheme
                // self-overwrote anyway).
                if rest == "part" {
                    let stale = entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age.as_secs() > 3600);
                    if stale {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
                continue;
            };
            // Accept both the bare `{pid}` form (never emitted by any
            // released binary — accepted defensively) and the
            // `{pid}-{nonce}` form; liveness keys on the pid component
            // only.
            let pid = pid.split('-').next().unwrap_or(pid);
            if pid.chars().all(|c| c.is_ascii_digit()) && !pid.is_empty() {
                // Deletion rule, safe across users AND PID namespaces:
                //   age < 60s          -> keep (grace: a live writer's
                //                         mtime refreshes on every 64KB
                //                         chunk, in any namespace)
                //   ESRCH && age > 60s -> reclaim (dead in OUR namespace,
                //                         provably not writing)
                //   age > 24h          -> reclaim regardless of liveness
                //                         (a foreign namespace's pid can
                //                         alias a live local process; no
                //                         real download goes 24h without
                //                         touching mtime)
                //   otherwise          -> keep
                // libc::kill is silent and EPERM (alive under another
                // user) keeps. Pure pid-liveness is NOT sufficient: pid
                // numbers are namespace-local, so a foreign container's
                // live writer could look ESRCH-dead here — the mtime
                // grace is the cross-namespace protection.
                let Ok(pid_num) = pid.parse::<i32>() else {
                    continue;
                };
                let Some(age) = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .map(|d| d.as_secs())
                else {
                    continue;
                };
                if age < 60 {
                    continue;
                }
                let esrch = unsafe { libc::kill(pid_num, 0) } == -1
                    && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
                if esrch || age > 24 * 3600 {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Verify a cached file against its `.sha256` sidecar.
    ///
    /// Returns `Ok(true)` if the hash matches, `Ok(false)` if it doesn't,
    /// or `Err` if the sidecar is missing or unreadable.
    pub fn verify_sha256(file_path: &Path) -> Result<bool, DownloadError> {
        let sha_path = file_path.with_extension(format!(
            "{}.sha256",
            file_path.extension().and_then(|e| e.to_str()).unwrap_or("")
        ));

        let expected = std::fs::read_to_string(&sha_path).map_err(|e| {
            DownloadError::Io(format!("failed to read {}: {e}", sha_path.display()))
        })?;

        // Format is "<hash>  <filename>\n" (GNU coreutils style).
        let expected_hash = expected.split_whitespace().next().unwrap_or("");

        let actual_hash = compute_sha256(file_path)?;

        Ok(expected_hash == actual_hash)
    }

    /// Encode bytes as lowercase hex string.
    fn hex_encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(feature = "download")]
pub use inner::*;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- sanitize_filename tests (always available, no feature) --

    #[test]
    fn sanitize_rejects_empty() {
        assert!(sanitize_filename("").is_err());
    }

    #[test]
    fn sanitize_rejects_path_traversal() {
        assert!(sanitize_filename("../etc/passwd").is_err());
        assert!(sanitize_filename("foo/../bar.gguf").is_err());
        assert!(sanitize_filename("..").is_err());
    }

    #[test]
    fn sanitize_rejects_directory_separators() {
        assert!(sanitize_filename("path/to/file.gguf").is_err());
        assert!(sanitize_filename("path\\to\\file.gguf").is_err());
    }

    #[test]
    fn sanitize_rejects_null_bytes() {
        assert!(sanitize_filename("file\0.gguf").is_err());
    }

    #[test]
    fn sanitize_rejects_control_chars() {
        assert!(sanitize_filename("file\n.gguf").is_err());
        assert!(sanitize_filename("file\t.gguf").is_err());
        assert!(sanitize_filename("\x01file.gguf").is_err());
        assert!(sanitize_filename("file\x7F.gguf").is_err());
    }

    #[test]
    fn sanitize_accepts_valid_filenames() {
        assert!(sanitize_filename("model.Q8_0.gguf").is_ok());
        assert!(sanitize_filename("Qwen2.5-3B-Instruct-Q8_0.gguf").is_ok());
        assert!(sanitize_filename("tinyllama-1.1b-chat-v1.0.Q4_0.gguf").is_ok());
        assert!(sanitize_filename("Meta-Llama-3.1-8B-Instruct.f16.gguf").is_ok());
    }

    #[test]
    fn sanitize_accepts_dots_in_filenames() {
        // Single dots are fine, only ".." is rejected.
        assert!(sanitize_filename("file.name.with.dots.gguf").is_ok());
        assert!(sanitize_filename(".hidden-file.gguf").is_ok());
    }

    // -- split_repo_path tests --

    #[test]
    fn split_accepts_flat_filename() {
        let (url, local) = split_repo_path("Qwen_Qwen3.5-9B-Q8_0.gguf").unwrap();
        assert_eq!(url, "Qwen_Qwen3.5-9B-Q8_0.gguf");
        assert_eq!(local, "Qwen_Qwen3.5-9B-Q8_0.gguf");
    }

    #[test]
    fn split_accepts_nested_shard() {
        let (url, local) =
            split_repo_path("Qwen_Qwen3.6-27B-bf16/Qwen_Qwen3.6-27B-bf16-00001-of-00002.gguf")
                .unwrap();
        assert_eq!(
            url,
            "Qwen_Qwen3.6-27B-bf16/Qwen_Qwen3.6-27B-bf16-00001-of-00002.gguf"
        );
        assert_eq!(local, "Qwen_Qwen3.6-27B-bf16-00001-of-00002.gguf");
    }

    #[test]
    fn split_rejects_traversal_and_malformed() {
        assert!(split_repo_path("../etc/passwd").is_err());
        assert!(split_repo_path("subdir/../escape.gguf").is_err());
        assert!(split_repo_path("/abs/path.gguf").is_err()); // leading slash -> empty segment
        assert!(split_repo_path("trailing/").is_err()); // trailing slash -> empty segment
        assert!(split_repo_path("double//slash.gguf").is_err()); // empty middle segment
        assert!(split_repo_path("back\\slash.gguf").is_err());
        assert!(split_repo_path("a/b/c/d/e.gguf").is_err()); // too deep
        assert!(split_repo_path("").is_err());
    }

    // -- download feature tests --

    #[cfg(feature = "download")]
    mod download_tests {
        use super::super::inner::*;
        use std::io::Write;

        #[test]
        fn compute_sha256_known_value() {
            // SHA-256 of "hello world\n" = a948904f2f0f479b8f8564...
            let dir =
                std::env::temp_dir().join(format!("lumen-test-sha256-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join("test-hello.txt");
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"hello world\n").unwrap();
            drop(f);

            let hash = compute_sha256(&path).unwrap();
            assert_eq!(
                hash,
                "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
            );

            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn verify_sha256_roundtrip() {
            let dir = std::env::temp_dir()
                .join(format!("lumen-test-sha256-verify-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join("test-verify.gguf");
            let sha_path = dir.join("test-verify.gguf.sha256");

            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"test content for sha256 verification")
                .unwrap();
            drop(f);

            // Compute hash and write sidecar.
            let hash = compute_sha256(&path).unwrap();
            std::fs::write(&sha_path, format!("{hash}  test-verify.gguf\n")).unwrap();

            // Verify should succeed.
            assert!(verify_sha256(&path).unwrap());

            // Tamper with file.
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"tampered content").unwrap();
            drop(f);

            // Verify should fail.
            assert!(!verify_sha256(&path).unwrap());

            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&sha_path);
        }

        #[test]
        fn download_gguf_rejects_traversal() {
            let dir =
                std::env::temp_dir().join(format!("lumen-test-traversal-{}", std::process::id()));
            let result = download_gguf("some/repo", "../etc/passwd", &dir, true);
            assert!(result.is_err());
            if let Err(DownloadError::InvalidFilename(msg)) = result {
                assert!(
                    msg.contains("path traversal"),
                    "expected traversal error, got: {msg}"
                );
            } else {
                panic!("expected InvalidFilename error");
            }
        }
    }
    #[cfg(feature = "download")]
    #[test]
    fn exclusive_staging_write_then_hash_via_same_fd() {
        // Regression for the EBADF cold-download failure: the staging fd is
        // opened read+write, written, seeked to 0, and hashed through the
        // SAME descriptor — the exact production flow.
        use std::io::{Seek, Write};
        let dir = std::env::temp_dir().join(format!("lumen-staging-fd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (path, mut f) = super::create_exclusive_staging(&dir, "m.gguf").unwrap();
        f.write_all(b"lumen staging bytes").unwrap();
        f.flush().unwrap();
        f.seek(std::io::SeekFrom::Start(0)).unwrap();
        let h = super::sha256_of_reader(&mut f).unwrap();
        assert_eq!(h.len(), 64, "hex sha256 expected");
        // Same-fd hash must match the by-path hash of the same bytes.
        let h2 = super::compute_sha256(&path).unwrap();
        assert_eq!(h, h2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
