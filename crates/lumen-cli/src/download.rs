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

    /// Where model files are fetched from. The field is private to this
    /// module and production has exactly one constructor, so the download
    /// path cannot be pointed anywhere else without the test-only one.
    mod base_url {
        pub(crate) struct BaseUrl(String);

        impl BaseUrl {
            pub(crate) fn hugging_face() -> Self {
                Self("https://huggingface.co".to_string())
            }

            #[cfg(test)]
            pub(crate) fn local(origin: String) -> Self {
                Self(origin)
            }

            pub(crate) fn as_str(&self) -> &str {
                &self.0
            }
        }
    }
    pub(crate) use base_url::BaseUrl;

    /// A read or write that makes no progress for this long is a stalled
    /// transfer, not a slow one.
    const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    /// A request for the bytes with no content coding applied. A server that
    /// honors the header does not touch any `Content-Length` it sends; one
    /// that encodes anyway is caught by [`reject_encoded_response`], since
    /// the crate is built without transparent decompression. A secure URL is
    /// never followed to a plaintext one.
    pub(crate) fn stored_bytes_request(method: &str, url: &str) -> ureq::Request {
        let secure = url
            .get(..8)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("https://"));
        ureq::AgentBuilder::new()
            .https_only(secure)
            .timeout_read(STALL_TIMEOUT)
            .timeout_write(STALL_TIMEOUT)
            .build()
            .request(method, url)
            .set("Accept-Encoding", "identity")
    }

    /// Encoded bytes would pass the length check and be published as the
    /// model, so a response is accepted only when every `Content-Encoding`
    /// value it carries is a bare `identity` and every `Transfer-Encoding`
    /// value is `chunked`. Lists are refused as written, and a header ureq
    /// parses but cannot render as text is refused rather than ignored; a
    /// line ureq cannot parse at all never reaches this check.
    fn reject_encoded_response(resp: &ureq::Response) -> Result<(), DownloadError> {
        let refuse = |header: &str, values: &[&str]| {
            let what = if values.is_empty() {
                "an unreadable".to_string()
            } else {
                format!("{values:?}")
            };
            DownloadError::Io(format!(
                "server sent {what} {header} for {}; refusing to store encoded bytes as the model",
                resp.get_url()
            ))
        };
        for (header, shown, allowed) in [
            ("content-encoding", "Content-Encoding", "identity"),
            ("transfer-encoding", "Transfer-Encoding", "chunked"),
        ] {
            let present = resp
                .headers_names()
                .iter()
                .any(|n| n.eq_ignore_ascii_case(header));
            if !present {
                continue;
            }
            let values = resp.all(header);
            if values.is_empty() || !values.iter().all(|v| v.eq_ignore_ascii_case(allowed)) {
                return Err(refuse(shown, &values));
            }
        }
        Ok(())
    }

    pub(crate) fn model_url(base_url: &str, repo: &str, url_path: &str) -> String {
        format!("{base_url}/{repo}/resolve/main/{url_path}")
    }

    /// Get the file size via a HEAD request. HF answers with a 302 to its
    /// CDN; ureq follows it and the final response carries Content-Length.
    fn get_remote_size(url: &str) -> Result<Option<u64>, DownloadError> {
        let resp = stored_bytes_request("HEAD", url)
            .call()
            .map_err(|e| DownloadError::Io(format!("HEAD request failed for {url}: {e}")))?;
        reject_encoded_response(&resp)?;

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
    /// The file is downloaded to a `.part` temporary file whose full byte count
    /// is verified, then hashed, then atomically renamed to the final path; the
    /// `.sha256` sidecar is written after the rename (so a published file may
    /// briefly exist without its sidecar — harmless, as the sidecar is
    /// write-only metadata that no load path consults).
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
        download_from(
            &BaseUrl::hugging_face(),
            repo,
            filename,
            dest_dir,
            skip_confirm,
        )
    }

    pub(crate) fn download_from(
        base_url: &BaseUrl,
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
        // name BY DESIGN: it is shared last-writer-wins metadata, written
        // after the winner's rename, and write-only in production (only its
        // unit test reads it back). Because the cache keys on the flattened
        // basename while the hash is of the source URL (repo + path), two
        // different sources sharing a basename can leave a sidecar whose hash
        // does not match the resident file — harmless, since no load path
        // consults it; correctness rests on the atomic rename publishing only
        // fully-verified bytes.
        let sha_path = dest_dir.join(format!("{filename}.sha256"));
        // Reclaim BEFORE the cache-hit return: after one racer succeeds,
        // every future call takes the cache-hit fast path, so litter from
        // a SIGKILLed racer would otherwise never be reclaimed. The scan
        // is a small read_dir plus one libc::kill per stale candidate — cheap.
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

        // The URL uses the full repo path, which may include a subdirectory;
        // the local file is the flat basename.
        let url = model_url(base_url.as_str(), repo, &url_path);

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
        let resp = stored_bytes_request("GET", &url)
            .call()
            .map_err(|e| DownloadError::Io(format!("GET request failed: {e}")))?;
        reject_encoded_response(&resp)?;

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
        // create_exclusive_staging captures the inode from the fd it just
        // O_EXCL-created and returns it, so the guard here is armed with the
        // identity it will check on Drop without a second stat. The fstat
        // failure window lives inside that helper, and its only outcome is a
        // bounded, self-healing leak (the .part is left for reclaim, never
        // deleted by path unverified) — not a wrong-file deletion.
        let (part_path, mut file, own_dev_ino) = create_exclusive_staging(dest_dir, filename)?;
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

        // Verify the full byte count before publishing. The guard cleans up
        // the .part file on an error return.
        verify_complete_transfer(content_length, total_written)?;

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

        // Atomic rename FIRST: .part -> final, then the sidecar. The rename
        // publishes only fully size- and hash-verified bytes, so the final
        // file is correct the instant it appears. The sidecar write that
        // follows is best-effort write-only metadata; a crash or write
        // failure between the two can leave the final without a current
        // sidecar indefinitely, which is harmless because no load path reads
        // it (the cache hit checks only that the file exists and is nonempty).
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
    /// `{filename}.{pid}-{nonce}.part`. Returns the path, the read+write fd,
    /// and the fd's `(dev, ino)` so the caller can arm its cleanup guard
    /// atomically — no window between the exclusive create and the armed
    /// guard. A failed stat on the fresh fd (near-impossible) leaves the
    /// `.part` for `reclaim_stale_parts` to sweep rather than deleting it by
    /// path unverified, which could not confirm the file is still ours.
    pub fn create_exclusive_staging(
        dest_dir: &Path,
        filename: &str,
    ) -> Result<(std::path::PathBuf, std::fs::File, (u64, u64)), DownloadError> {
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
                Ok(f) => {
                    use std::os::unix::fs::MetadataExt;
                    let m = f
                        .metadata()
                        .map_err(|e| DownloadError::Io(format!("fstat error on staging: {e}")))?;
                    return Ok((candidate, f, (m.dev(), m.ino())));
                }
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

    /// Decide whether a finished transfer is safe to publish. A clean EOF is
    /// indistinguishable from a complete transfer, so a connection-close
    /// truncation with no authoritative length would hash and publish a
    /// partial model that the sidecar then certifies. `content_length` is the
    /// GET length or the HEAD fallback; for HuggingFace it is always present.
    /// When neither reports a size we cannot detect truncation, so we refuse
    /// to publish rather than risk a silently partial model.
    pub(crate) fn verify_complete_transfer(
        content_length: Option<u64>,
        total_written: u64,
    ) -> Result<(), DownloadError> {
        match content_length {
            Some(expected) if total_written != expected => Err(DownloadError::Io(format!(
                "size mismatch: expected {expected} bytes, got {total_written} bytes"
            ))),
            None => Err(DownloadError::Io(format!(
                "server reported no Content-Length (HEAD or GET) for this download, \
                 so a truncated transfer cannot be detected; refusing to publish \
                 {total_written} unverified bytes — retry, or fetch from a source \
                 that reports a size"
            ))),
            _ => Ok(()),
        }
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
        let (path, mut f, _dev_ino) = super::create_exclusive_staging(&dir, "m.gguf").unwrap();
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

    #[cfg(feature = "download")]
    #[test]
    fn verify_complete_transfer_rejects_short_and_unknown() {
        // A known length that matches publishes.
        assert!(super::verify_complete_transfer(Some(100), 100).is_ok());
        // A short transfer against a known length is a size mismatch.
        let e = super::verify_complete_transfer(Some(100), 40).unwrap_err();
        assert!(format!("{e}").contains("size mismatch"), "got {e}");
        // No authoritative length: refuse to publish rather than certify a
        // possibly-truncated model.
        let e = super::verify_complete_transfer(None, 40).unwrap_err();
        assert!(
            format!("{e}").contains("no Content-Length"),
            "unknown-length transfer must be refused, got {e}"
        );
    }

    /// A stand-in for HF's topology: an origin that answers every request
    /// with a 302 to a CDN on a different authority, and a CDN that serves
    /// `BODY` with the given extra headers. Requests are accepted until
    /// `expected` heads were seen or a deadline passes, and every read or
    /// write on the wire is bounded, so a broken premise or a silent peer
    /// fails the count instead of hanging. Returns the origin base URL.
    #[cfg(feature = "download")]
    fn serve_like_hf(
        expected: usize,
        head_extra: &'static str,
        get_extra: &'static str,
    ) -> (super::BaseUrl, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{BufRead, BufReader, Write};
        let origin = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let cdn = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let origin_base = format!("http://{}", origin.local_addr().unwrap());
        let base = format!("http://{}", cdn.local_addr().unwrap());
        origin.set_nonblocking(true).unwrap();
        cdn.set_nonblocking(true).unwrap();
        let handle = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut heads = Vec::new();
            while heads.len() < expected && std::time::Instant::now() < deadline {
                let stream = match origin.accept().or_else(|_| cdn.accept()) {
                    Ok((stream, _)) => stream,
                    Err(_) => {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    }
                };
                stream.set_nonblocking(false).unwrap();
                let wire = Some(std::time::Duration::from_secs(2));
                stream.set_read_timeout(wire).unwrap();
                stream.set_write_timeout(wire).unwrap();
                let mut reader = BufReader::new(stream);
                let mut head = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() {
                        head.clear();
                        break;
                    }
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    head.push_str(&line);
                }
                if head.is_empty() {
                    continue;
                }
                let is_head = head.starts_with("HEAD ");
                let response = if !head.contains(" /cdn/") {
                    format!("HTTP/1.1 302 Found\r\nLocation: {base}/cdn/m.gguf\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                } else {
                    let extra = if is_head { head_extra } else { get_extra };
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                        BODY.len()
                    )
                };
                let stream = reader.get_mut();
                let _ = stream.write_all(response.as_bytes());
                if !is_head && head.contains(" /cdn/") {
                    let _ = stream.write_all(BODY);
                }
                heads.push(head.to_ascii_lowercase());
            }
            heads
        });
        (super::BaseUrl::local(origin_base), handle)
    }

    #[cfg(feature = "download")]
    const BODY: &[u8] = b"stored model bytes";

    #[cfg(feature = "download")]
    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[cfg(feature = "download")]
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-dl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The real download path against the stand-in: both hops of HEAD and
    /// GET ask for stored bytes, and the published file is byte-exact.
    #[cfg(feature = "download")]
    #[test]
    fn download_asks_for_stored_bytes_across_the_redirect() {
        let (base, server) = serve_like_hf(4, "", "");
        let dir = scratch_dir("ok");
        let path = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), BODY);
        assert_eq!(entries(&dir), vec!["m.gguf", "m.gguf.sha256"]);
        let heads = server.join().unwrap();
        assert_eq!(
            heads.len(),
            4,
            "each method: origin hop + cross-authority CDN hop"
        );
        for head in heads {
            assert!(
                head.lines().any(|l| l == "accept-encoding: identity"),
                "request must ask for stored bytes, got:\n{head}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A server that encodes anyway is refused on the GET, before any byte
    /// is published.
    #[cfg(feature = "download")]
    #[test]
    fn encoded_get_is_refused() {
        let (base, server) = serve_like_hf(4, "", "Content-Encoding: br\r\n");
        let dir = scratch_dir("enc-get");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(
            format!("{err}").contains("[\"br\"] Content-Encoding"),
            "got {err}"
        );
        assert!(
            entries(&dir).is_empty(),
            "nothing may be left behind, got {:?}",
            entries(&dir)
        );
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Only bare `identity` values pass: a duplicate header whose first value
    /// is identity, a value ureq cannot render, a list, and a coded transfer
    /// encoding are all refused.
    #[cfg(feature = "download")]
    #[test]
    fn duplicate_or_unrenderable_content_encoding_is_refused() {
        for (tag, extra) in [
            (
                "dup",
                "Content-Encoding: identity\r\nContent-Encoding: gzip\r\n",
            ),
            ("raw", "Content-Encoding: gzip\u{e9}\r\n"),
            ("list", "Content-Encoding: identity, gzip\r\n"),
            ("transfer", "Transfer-Encoding: gzip, chunked\r\n"),
        ] {
            let (base, server) = serve_like_hf(4, "", extra);
            let dir = scratch_dir(tag);
            let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
            assert!(format!("{err}").contains("-Encoding"), "{tag}: got {err}");
            assert!(
                entries(&dir).is_empty(),
                "{tag}: nothing may be left behind"
            );
            assert_eq!(server.join().unwrap().len(), 4, "{tag}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[cfg(feature = "download")]
    #[test]
    fn hf_url_is_https_huggingface() {
        assert_eq!(
            super::model_url(
                super::BaseUrl::hugging_face().as_str(),
                "org/repo",
                "sub/m.gguf"
            ),
            "https://huggingface.co/org/repo/resolve/main/sub/m.gguf"
        );
    }

    /// The guard can only see an encoding ureq leaves in place: were ureq's
    /// gzip feature back on, it would decode this response and drop the
    /// header, the guard would pass, and the message here would not appear.
    #[cfg(feature = "download")]
    #[test]
    fn transparent_decompression_is_off() {
        let (base, server) = serve_like_hf(4, "", "Content-Encoding: gzip\r\n");
        let dir = scratch_dir("gzip-off");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(
            format!("{err}").contains("[\"gzip\"] Content-Encoding"),
            "got {err}"
        );
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same refusal on the HEAD, whose length would describe encoded bytes.
    #[cfg(feature = "download")]
    #[test]
    fn encoded_head_is_refused() {
        let (base, server) = serve_like_hf(2, "Content-Encoding: gzip\r\n", "");
        let dir = scratch_dir("enc-head");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(
            format!("{err}").contains("[\"gzip\"] Content-Encoding"),
            "got {err}"
        );
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
