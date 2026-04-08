//! Download GGUF models from HuggingFace with integrity verification.
//!
//! All code in this module is gated behind `#[cfg(feature = "download")]`.
//! When the feature is disabled, only the `sanitize_filename` function
//! (used for path traversal prevention) is available.

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
        return Err(format!("filename contains directory separator: {filename:?}"));
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
    fn confirm_download(repo: &str, filename: &str, size: Option<u64>) -> Result<bool, DownloadError> {
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
        Ok(trimmed.is_empty() || trimmed.eq_ignore_ascii_case("y") || trimmed.eq_ignore_ascii_case("yes"))
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
    /// - `filename`: GGUF filename (e.g. `"Qwen2.5-3B-Instruct-Q8_0.gguf"`)
    /// - `dest_dir`: Directory to download into (typically the cache dir)
    /// - `skip_confirm`: If true, skip the `[Y/n]` prompt
    pub fn download_gguf(
        repo: &str,
        filename: &str,
        dest_dir: &Path,
        skip_confirm: bool,
    ) -> Result<PathBuf, DownloadError> {
        // Sanitize filename to prevent path traversal.
        sanitize_filename(filename).map_err(DownloadError::InvalidFilename)?;

        let final_path = dest_dir.join(filename);
        let part_path = dest_dir.join(format!("{filename}.part"));
        let sha_path = dest_dir.join(format!("{filename}.sha256"));

        // Cache hit: file already exists and is non-empty.
        if final_path.is_file() {
            if let Ok(meta) = std::fs::metadata(&final_path) {
                if meta.len() > 0 {
                    eprintln!("Cache hit: {}", final_path.display());
                    return Ok(final_path);
                }
            }
        }

        // Build the HuggingFace download URL.
        let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");

        // Get file size for confirmation and progress bar.
        let size = get_remote_size(&url)?;

        // Confirm with user unless --yes was passed.
        if !skip_confirm {
            if !confirm_download(repo, filename, size)? {
                return Err(DownloadError::UserDeclined);
            }
        }

        // Ensure dest dir exists.
        std::fs::create_dir_all(dest_dir)
            .map_err(|e| DownloadError::Io(format!("failed to create {}: {e}", dest_dir.display())))?;

        // Start the download.
        eprintln!("Downloading: {url}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| DownloadError::Io(format!("GET request failed: {e}")))?;

        // Get content length from the actual response (might differ from HEAD due to CDN).
        let content_length = resp.header("content-length")
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

        // Download to .part file.
        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(&part_path)
            .map_err(|e| DownloadError::Io(format!("failed to create {}: {e}", part_path.display())))?;

        let mut buf = vec![0u8; 64 * 1024]; // 64 KB buffer
        let mut total_written: u64 = 0;

        loop {
            let n = reader.read(&mut buf)
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
        drop(file);

        pb.finish_with_message("download complete");

        // Verify size if known.
        if let Some(expected) = content_length {
            if total_written != expected {
                // Clean up .part file.
                let _ = std::fs::remove_file(&part_path);
                return Err(DownloadError::Io(format!(
                    "size mismatch: expected {expected} bytes, got {total_written} bytes"
                )));
            }
        }

        // Compute SHA-256 of the downloaded file.
        let hash = compute_sha256(&part_path)?;

        // Write SHA-256 sidecar.
        std::fs::write(&sha_path, format!("{hash}  {filename}\n"))
            .map_err(|e| DownloadError::Io(format!("failed to write {}: {e}", sha_path.display())))?;

        // Atomic rename: .part -> final.
        std::fs::rename(&part_path, &final_path)
            .map_err(|e| DownloadError::Io(format!(
                "failed to rename {} -> {}: {e}",
                part_path.display(),
                final_path.display()
            )))?;

        eprintln!("Saved: {} (SHA-256: {hash})", final_path.display());
        Ok(final_path)
    }

    /// Compute SHA-256 hash of a file. Returns the hex-encoded digest.
    pub fn compute_sha256(path: &Path) -> Result<String, DownloadError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| DownloadError::Io(format!("failed to open {} for hashing: {e}", path.display())))?;

        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];

        loop {
            let n = file.read(&mut buf)
                .map_err(|e| DownloadError::Io(format!("read error during hashing: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        let digest = hasher.finalize();
        Ok(hex_encode(&digest))
    }

    /// Verify a cached file against its `.sha256` sidecar.
    ///
    /// Returns `Ok(true)` if the hash matches, `Ok(false)` if it doesn't,
    /// or `Err` if the sidecar is missing or unreadable.
    pub fn verify_sha256(file_path: &Path) -> Result<bool, DownloadError> {
        let sha_path = file_path.with_extension(
            format!(
                "{}.sha256",
                file_path.extension().and_then(|e| e.to_str()).unwrap_or("")
            ),
        );

        let expected = std::fs::read_to_string(&sha_path)
            .map_err(|e| DownloadError::Io(format!("failed to read {}: {e}", sha_path.display())))?;

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

    // -- sanitize_filename tests (always available, no feature gate) --

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

    // -- download feature tests --

    #[cfg(feature = "download")]
    mod download_tests {
        use super::super::inner::*;
        use std::io::Write;

        #[test]
        fn compute_sha256_known_value() {
            // SHA-256 of "hello world\n" = a948904f2f0f479b8f8564...
            let dir = std::env::temp_dir().join("lumen-test-sha256");
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
            let dir = std::env::temp_dir().join("lumen-test-sha256-verify");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join("test-verify.gguf");
            let sha_path = dir.join("test-verify.gguf.sha256");

            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"test content for sha256 verification").unwrap();
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
            let dir = std::env::temp_dir().join("lumen-test-traversal");
            let result = download_gguf("some/repo", "../etc/passwd", &dir, true);
            assert!(result.is_err());
            if let Err(DownloadError::InvalidFilename(msg)) = result {
                assert!(msg.contains("path traversal"), "expected traversal error, got: {msg}");
            } else {
                panic!("expected InvalidFilename error");
            }
        }
    }
}
