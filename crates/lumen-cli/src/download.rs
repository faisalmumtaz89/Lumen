//! Download GGUF models from HuggingFace with integrity verification.
//!
//! All code in this module is gated behind `#[cfg(feature = "download")]`.
//! When the feature is disabled, only the `sanitize_filename` function
//! (used for path traversal prevention) is available.

/// Split a registry-declared GGUF path into `(url_path, local_basename)`.
///
/// Registry entries may nest shards under a repo subdirectory (e.g.
/// `"BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"`).
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

    /// The proxy a request goes through, with the variable that named it.
    pub(crate) type Route = Option<(ureq::Proxy, &'static str)>;

    /// A request for the bytes with no content coding applied. A server that
    /// honors the header does not touch any `Content-Length` it sends; one
    /// that encodes anyway is caught by [`reject_unusable_response`], since
    /// the crate is built without transparent decompression. A secure URL is
    /// never followed to a plaintext one. The request goes through `route`'s
    /// proxy, redirects included.
    pub(crate) fn stored_bytes_request(method: &str, url: &str, route: &Route) -> ureq::Request {
        let secure = url
            .get(..8)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("https://"));
        let mut agent = ureq::AgentBuilder::new()
            .https_only(secure)
            .timeout_read(STALL_TIMEOUT)
            .timeout_write(STALL_TIMEOUT);
        if let Some((proxy, _)) = route {
            agent = agent.proxy(proxy.clone());
        }
        agent
            .build()
            .request(method, url)
            .set("Accept-Encoding", "identity")
    }

    /// " through the proxy in <variable>" for a proxied route, else nothing:
    /// the suffix a failed request or a broken transfer through a proxy
    /// carries.
    fn via(route: &Route) -> String {
        route
            .as_ref()
            .map(|(_, name)| format!(" through the proxy in {name}"))
            .unwrap_or_default()
    }

    /// The route the environment sets for `url` ([`env_proxy`]).
    pub(crate) fn env_route(url: &str) -> Result<Route, DownloadError> {
        env_proxy(url, |name| std::env::var(name).ok())
    }

    /// The proxy the environment sets for `url`, chosen as curl chooses it:
    /// `https_proxy` for an https URL or `http_proxy` for an http one, else
    /// `all_proxy`, each read lowercase first, then uppercase; an empty value
    /// counts as unset. A loopback host, or one that `no_proxy` covers, is
    /// reached directly. An HTTP or SOCKS proxy is accepted, its user name and
    /// password percent-decoded as curl reads them. `var` reads one
    /// environment variable.
    fn env_proxy(url: &str, var: impl Fn(&str) -> Option<String>) -> Result<Route, DownloadError> {
        let parsed = url::Url::parse(url)
            .map_err(|e| DownloadError::Io(format!("invalid URL {url}: {e}")))?;
        let host = parsed
            .host_str()
            .unwrap_or_default()
            .trim_end_matches('.')
            .to_ascii_lowercase();
        let set = |name: &str| var(name).filter(|value| !value.is_empty());
        let loopback = host == "localhost"
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        let no_proxy = set("no_proxy").or_else(|| set("NO_PROXY"));
        if loopback || no_proxy.is_some_and(|list| no_proxy_covers(&list, &host)) {
            return Ok(None);
        }
        let names = if parsed.scheme() == "https" {
            ["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
        } else {
            ["http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"]
        };
        let Some((name, value)) = names
            .into_iter()
            .find_map(|name| set(name).map(|value| (name, value)))
        else {
            return Ok(None);
        };
        // The value is never echoed: it may carry a password.
        ureq::Proxy::new(proxy_spec(&value))
            .map(|proxy| Some((proxy, name)))
            .map_err(|e| DownloadError::Io(format!("{name} is not a proxy URL lumen can use: {e}")))
    }

    /// `value` in the form ureq parses: the user name and password
    /// percent-decoded (a password holding `@` is written `%40`, as curl
    /// requires), a user name without a password given an empty one (ureq
    /// refuses the bare name; curl sends `name:`), and `socks5h` spelled
    /// `socks5`, whose connections ureq already resolve on the proxy. A part
    /// that does not decode is kept.
    fn proxy_spec(value: &str) -> String {
        let (scheme, rest) = match value.split_once("://") {
            Some(("socks5h", rest)) => ("socks5://", rest),
            Some((scheme, rest)) => (&value[..scheme.len() + 3], rest),
            None => ("", value),
        };
        let Some((userinfo, address)) = rest.rsplit_once('@') else {
            return format!("{scheme}{rest}");
        };
        let decode = |part: &str| {
            percent_encoding::percent_decode_str(part)
                .decode_utf8()
                .map_or_else(|_| part.to_string(), |text| text.into_owned())
        };
        let userinfo = match userinfo.split_once(':') {
            Some((user, password)) => format!("{}:{}", decode(user), decode(password)),
            None => format!("{}:", decode(userinfo)),
        };
        format!("{scheme}{userinfo}@{address}")
    }

    /// Whether a `no_proxy` list (entries split by commas or whitespace)
    /// covers `host`, given lowercase and without a trailing dot: `*` covers
    /// every host, and any other entry covers the host it names and that
    /// host's subdomains, written with or without a leading `.` or `*.`.
    fn no_proxy_covers(list: &str, host: &str) -> bool {
        list.split(|c: char| c == ',' || c.is_ascii_whitespace())
            .map(|entry| {
                entry
                    .trim_start_matches("*.")
                    .trim_start_matches('.')
                    .trim_end_matches('.')
                    .to_ascii_lowercase()
            })
            .filter(|entry| !entry.is_empty())
            .any(|entry| {
                entry == "*"
                    || host == entry
                    || host
                        .strip_suffix(entry.as_str())
                        .is_some_and(|rest| rest.ends_with('.'))
            })
    }

    /// Run a ureq call or header read with a panic turned into `Err`.
    /// Unwinding is safe to assert: the closures only send a request built
    /// for that call or read a `Response`, so no shared state is left
    /// half-updated. The
    /// panic hook is left alone — it is process-global, and replacing it
    /// would also swallow the failure text of any other thread (including
    /// this crate's own tests) — so the parser's one panic line still
    /// prints before the refusal.
    fn fenced<T>(op: impl FnOnce() -> T) -> Result<T, ()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).map_err(|_| ())
    }

    /// Every value of a response header, or a refusal when a line ureq
    /// accepted cannot be sliced (see [`call_for_stored_bytes`]).
    fn header_values(resp: &ureq::Response, name: &str) -> Result<Vec<String>, DownloadError> {
        fenced(|| resp.all(name).into_iter().map(str::to_owned).collect()).map_err(|_| {
            DownloadError::Io(format!(
                "the {name} header of the response from {} could not be read safely; refusing the response",
                resp.get_url()
            ))
        })
    }

    /// Issue a stored-bytes request and hand back only a usable response.
    /// A header line with no colon is accepted by ureq's parser, and its
    /// value accessors then index past the end of the line — during response
    /// construction for some headers, on later lookups for others — so both
    /// the call and every value read are fenced: a panic is a refusal, not
    /// an abort of the process on a hostile origin.
    fn call_for_stored_bytes(
        method: &str,
        url: &str,
        route: &Route,
    ) -> Result<ureq::Response, DownloadError> {
        let request = stored_bytes_request(method, url, route);
        let outcome = fenced(|| request.call());
        let resp = match outcome {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                // ureq words a 401 or 407 from the proxy as "Provided proxy
                // credentials are incorrect", also when none were sent.
                let why = match &e {
                    ureq::Error::Transport(t) if t.kind() == ureq::ErrorKind::ProxyUnauthorized => {
                        "the proxy requires authentication; lumen sends the user name and \
                         password in the proxy URL (user:password@host, percent-encoded) with \
                         Basic authentication"
                            .to_string()
                    }
                    _ => e.to_string(),
                };
                return Err(DownloadError::Io(format!(
                    "{method} request failed for {url}{}: {why}",
                    via(route)
                )));
            }
            Err(_) => {
                return Err(DownloadError::Io(format!(
                    "the response from {url} could not be read safely (a header line the parser cannot slice, or an internal error); refusing the response"
                )))
            }
        };
        reject_unusable_response(&resp)?;
        Ok(resp)
    }

    /// A response is stored only when it is a complete 200 whose every
    /// `Content-Encoding` value is a bare `identity` and every
    /// `Transfer-Encoding` value is `chunked`. The value count is checked
    /// against the number of header lines carrying that name, so a readable
    /// `identity` on one line cannot hide an unreadable value on another;
    /// lists are refused as written. Every `Content-Length` line, one or
    /// many, must be ASCII digits that fit in a u64 (no sign, no other bytes;
    /// ureq trims surrounding whitespace, Unicode included, before the value
    /// is seen) and all must be byte-identical, since the first gates the
    /// completion check; a length that fails to parse would otherwise fall
    /// back to the HEAD's advisory size. Encoded or partial bytes would
    /// otherwise pass the length check and be published as the model. Out of
    /// reach: any header line whose name holds a byte that is not a token
    /// character (a space before the colon, an obsolete folded continuation
    /// starting with a space or tab, a high byte) is dropped by ureq before
    /// any header view exists.
    fn reject_unusable_response(resp: &ureq::Response) -> Result<(), DownloadError> {
        if resp.status() != 200 {
            return Err(DownloadError::Io(format!(
                "server answered {} {} for {}; only a complete 200 response is stored",
                resp.status(),
                resp.status_text(),
                resp.get_url()
            )));
        }
        let refuse = |header: &str, values: &[String]| {
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
        let names = resp.headers_names();
        let length_lines = names
            .iter()
            .filter(|n| n.eq_ignore_ascii_case("content-length"))
            .count();
        if length_lines > 0 {
            let lengths = header_values(resp, "content-length")?;
            // ASCII digits only (ureq has already trimmed Unicode whitespace such as
            // U+00A0 off the value, so the bytes are checked, not the trimmed text)
            // and representable: 2^64 is all digits and would parse to nothing.
            let is_plain_integer =
                |l: &String| l.bytes().all(|b| b.is_ascii_digit()) && l.parse::<u64>().is_ok();
            if lengths.len() != length_lines
                || !lengths.iter().all(is_plain_integer)
                || lengths.iter().any(|l| l != &lengths[0])
            {
                return Err(refuse("Content-Length", &lengths));
            }
        }
        for (header, shown, allowed) in [
            ("content-encoding", "Content-Encoding", "identity"),
            ("transfer-encoding", "Transfer-Encoding", "chunked"),
        ] {
            let lines = names
                .iter()
                .filter(|n| n.eq_ignore_ascii_case(header))
                .count();
            if lines == 0 {
                continue;
            }
            let values = header_values(resp, header)?;
            if values.len() != lines || !values.iter().all(|v| v.eq_ignore_ascii_case(allowed)) {
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
    /// The HEAD stores nothing and its size is only advisory (a fallback
    /// for a GET without a length), so anything but a clean, complete 200
    /// makes the size unknown rather than failing the download; the GET
    /// answers to every rule on its own.
    ///
    /// The size is not asked for through any proxy. Through an HTTP proxy
    /// ureq 2 reads the tunnel's CONNECT answer as the HEAD's own bodiless
    /// response and pools its socket, which clears the socket's read and write
    /// timeouts, so a stalled TLS handshake would hang the pull with no limit.
    /// A SOCKS route has no such hang; one rule keeps every proxied pull alike:
    /// the prompt shows an unknown size, and the GET must carry its own
    /// Content-Length, since there is no HEAD size to fall back on.
    fn get_remote_size(url: &str, route: &Route) -> Result<Option<u64>, DownloadError> {
        if route.is_some() {
            return Ok(None);
        }
        let unknown = |why: String| {
            eprintln!("Size unknown before download ({why}); the GET's own length decides.");
            Ok(None)
        };
        let request = stored_bytes_request("HEAD", url, route);
        let resp = match fenced(|| request.call()) {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => return unknown(format!("HEAD failed: {e}")),
            Err(()) => return unknown("HEAD response could not be read safely".to_string()),
        };
        if let Err(e) = reject_unusable_response(&resp) {
            return unknown(format!("HEAD unusable: {e}"));
        }
        let values = match header_values(&resp, "content-length") {
            Ok(values) => values,
            Err(e) => return unknown(format!("HEAD unusable: {e}")),
        };
        Ok(values.first().and_then(|cl| cl.parse::<u64>().ok()))
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

        // The route every request of this download takes; an unusable proxy
        // setting fails here, before anything is fetched.
        let route = env_route(&url)?;

        // Get file size for confirmation and progress bar.
        let size = get_remote_size(&url, &route)?;

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
        let resp = call_for_stored_bytes("GET", &url, &route)?;

        // Get content length from the actual response (might differ from HEAD due to CDN).
        let content_length = header_values(&resp, "content-length")?
            .first()
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
            let n = reader.read(&mut buf).map_err(|e| {
                DownloadError::Io(format!("read error during download{}: {e}", via(&route)))
            })?;
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

    #[cfg(test)]
    mod registry_files {
        //! requires network access: HEADs the registry's K-quant files and checks the advertised
        //! Content-Length against the sizes pinned below (README.md rounds them to 16.2 / 19.5 GiB)
        use super::*;

        #[test]
        #[ignore = "requires network access to huggingface.co"]
        fn k_quant_registry_files_resolve_to_their_pinned_sizes() {
            let reg = crate::registry::load_registry();
            let entry = reg.resolve("qwen3.8-27b").expect("qwen3.8-27b");
            for (key, size) in [("Q4_K_M", 17_442_399_968u64), ("Q5_K_M", 20_923_877_088u64)] {
                let src = &entry.gguf_files[key];
                let url = model_url("https://huggingface.co", &src.repo, src.file());
                let got = get_remote_size(&url, &None)
                    .expect("HEAD")
                    .expect("Content-Length");
                assert_eq!(got, size, "{key}: {url}");
            }
        }
    }

    #[cfg(test)]
    mod proxy_env {
        use super::*;

        const HF: &str = "https://huggingface.co/Qwen/x/resolve/main/m.gguf";

        /// For tests that send requests: `.invalid` never resolves (RFC 6761),
        /// so a regression that bypasses the proxy fails at name lookup
        /// instead of reaching a real host.
        const OFFLINE: &str = "https://huggingface.invalid/Qwen/x/resolve/main/m.gguf";

        fn route(url: &str, vars: &[(&str, &str)]) -> Result<Route, DownloadError> {
            env_proxy(url, |name| {
                vars.iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| v.to_string())
            })
        }

        fn proxy(url: &str, vars: &[(&str, &str)]) -> Result<Option<ureq::Proxy>, DownloadError> {
            route(url, vars).map(|route| route.map(|(proxy, _)| proxy))
        }

        fn http(addr: &str) -> Option<ureq::Proxy> {
            Some(ureq::Proxy::new(addr).unwrap())
        }

        #[test]
        fn an_https_url_takes_https_proxy_then_all_proxy() {
            assert_eq!(proxy(HF, &[]).unwrap(), None);
            let both = [("https_proxy", "http://a:1"), ("all_proxy", "http://b:2")];
            assert_eq!(proxy(HF, &both).unwrap(), http("http://a:1"));
            assert_eq!(
                proxy(HF, &[("ALL_PROXY", "http://b:2")]).unwrap(),
                http("http://b:2")
            );
            assert_eq!(
                proxy(HF, &[("HTTPS_PROXY", "proxy.corp:3128")]).unwrap(),
                http("http://proxy.corp:3128")
            );
            let cased = [
                ("https_proxy", "http://lower:1"),
                ("HTTPS_PROXY", "http://upper:2"),
            ];
            assert_eq!(proxy(HF, &cased).unwrap(), http("http://lower:1"));
        }

        #[test]
        fn a_scheme_takes_only_its_own_proxy_variable() {
            let only_http = [("http_proxy", "http://a:1"), ("HTTP_PROXY", "http://a:1")];
            assert_eq!(proxy(HF, &only_http).unwrap(), None);
            let only_https = [("https_proxy", "http://a:1")];
            assert_eq!(
                proxy("http://mirror.test/m.gguf", &only_https).unwrap(),
                None
            );
            assert_eq!(
                proxy("http://mirror.test/m.gguf", &only_http).unwrap(),
                http("http://a:1")
            );
        }

        #[test]
        fn an_empty_value_counts_as_unset() {
            let vars = [("https_proxy", ""), ("HTTPS_PROXY", "http://a:1")];
            assert_eq!(proxy(HF, &vars).unwrap(), http("http://a:1"));
            let vars = [("https_proxy", "http://a:1"), ("no_proxy", "")];
            assert_eq!(proxy(HF, &vars).unwrap(), http("http://a:1"));
        }

        #[test]
        fn no_proxy_covers_a_host_and_its_subdomains() {
            let covered = |list: &str, host: &str| {
                let vars = [("https_proxy", "http://a:1"), ("no_proxy", list)];
                proxy(&format!("https://{host}/m"), &vars)
                    .unwrap()
                    .is_none()
            };
            assert!(covered("*", "huggingface.co"));
            assert!(covered("huggingface.co", "huggingface.co"));
            assert!(covered("huggingface.co", "cdn-lfs.huggingface.co"));
            assert!(covered(".huggingface.co", "cdn-lfs.huggingface.co"));
            assert!(covered("*.huggingface.co", "cdn-lfs.huggingface.co"));
            assert!(covered("internal, HuggingFace.co.", "huggingface.co"));
            assert!(covered("internal huggingface.co", "huggingface.co"));
            assert!(!covered("face.co", "huggingface.co"));
            assert!(!covered("cdn.huggingface.co", "huggingface.co"));
            assert!(!covered("internal,,", "huggingface.co"));
            let upper = [
                ("https_proxy", "http://a:1"),
                ("NO_PROXY", "huggingface.co"),
            ];
            assert_eq!(proxy(HF, &upper).unwrap(), None);
        }

        #[test]
        fn a_loopback_host_is_never_proxied() {
            let vars = [("http_proxy", "http://a:1"), ("https_proxy", "http://a:1")];
            for url in [
                "http://127.0.0.1:8080/m",
                "http://localhost:8080/m",
                "http://[::1]:8080/m",
                "https://127.0.0.2/m",
            ] {
                assert_eq!(proxy(url, &vars).unwrap(), None, "{url}");
            }
        }

        #[test]
        fn an_unusable_setting_is_refused_without_echoing_its_value() {
            let bad = proxy(HF, &[("HTTPS_PROXY", "ftp://user:secret@h:21")])
                .unwrap_err()
                .to_string();
            assert!(
                bad.contains("HTTPS_PROXY") && !bad.contains("secret"),
                "{bad}"
            );
        }

        #[test]
        fn the_route_names_the_variable_it_came_from() {
            let vars = [("HTTPS_PROXY", "http://a:1"), ("all_proxy", "http://b:2")];
            assert_eq!(
                route(HF, &vars).unwrap().map(|(_, name)| name),
                Some("HTTPS_PROXY")
            );
            let vars = [("ALL_PROXY", "http://b:2")];
            assert_eq!(
                route(HF, &vars).unwrap().map(|(_, name)| name),
                Some("ALL_PROXY")
            );
        }

        #[test]
        fn a_socks_proxy_is_accepted_and_socks5h_resolves_on_the_proxy() {
            let socks = |value: &str| proxy(HF, &[("all_proxy", value)]).unwrap();
            let want = Some(ureq::Proxy::new("socks5://u:p@h:1080").unwrap());
            assert_eq!(socks("socks5://u:p@h:1080"), want);
            assert_eq!(socks("socks5h://u:p@h:1080"), want);
            assert_eq!(
                socks("socks4a://h:1080"),
                Some(ureq::Proxy::new("socks4a://h:1080").unwrap())
            );
        }

        /// A proxy on loopback that answers one connection with `answer` and
        /// hands over the request head it read: the CONNECT for an https URL,
        /// the request itself for an http one.
        fn recording_proxy(
            answer: &'static [u8],
        ) -> (std::net::SocketAddr, std::sync::mpsc::Receiver<String>) {
            use std::io::{Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let Ok((mut sock, _)) = listener.accept() else {
                    return;
                };
                sock.set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && sock.read(&mut byte).is_ok_and(|n| n == 1) {
                    head.push(byte[0]);
                }
                tx.send(String::from_utf8_lossy(&head).into_owned())
                    .unwrap();
                sock.write_all(answer).ok();
            });
            (addr, rx)
        }

        const FORBIDDEN: &[u8] = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n";

        fn head_from(heads: &std::sync::mpsc::Receiver<String>) -> String {
            heads
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the proxy was never contacted")
        }

        #[test]
        fn credentials_reach_the_proxy_as_curl_sends_them() {
            // Each value with the Basic credentials curl sends for it.
            for (value, sent) in [
                ("http://jdoe:P%40ss%3Aw%2Fd@{addr}", "amRvZTpQQHNzOncvZA=="), // jdoe:P@ss:w/d
                ("http://tok@{addr}", "dG9rOg=="),                             // tok:
                ("http://u%3Ap@{addr}", "dTpwOg=="),                           // u:p:
                ("dom%5Cjdoe:pw@{addr}", "ZG9tXGpkb2U6cHc="),                  // dom\jdoe:pw
                ("http://jdoe:100%zz@{addr}", "amRvZToxMDAleno="),             // jdoe:100%zz
            ] {
                let (addr, heads) = recording_proxy(FORBIDDEN);
                let value = value.replace("{addr}", &addr.to_string());
                let route = route(OFFLINE, &[("https_proxy", &value)]).unwrap();
                let err = call_for_stored_bytes("GET", OFFLINE, &route)
                    .unwrap_err()
                    .to_string();
                let head = head_from(&heads);
                let header = format!("Proxy-Authorization: basic {sent}");
                assert!(head.lines().any(|line| line == header), "{value}: {head}");
                // A 403 is a refusal, not a request for credentials.
                assert!(!err.contains("requires authentication"), "{err}");
            }
        }

        #[test]
        fn a_proxy_that_wants_authentication_says_so() {
            let (addr, _heads) = recording_proxy(
                b"HTTP/1.1 407 Proxy Authentication Required\r\n\
                  Proxy-Authenticate: Basic realm=\"corp\"\r\nContent-Length: 0\r\n\r\n",
            );
            let route = route(OFFLINE, &[("https_proxy", &format!("http://{addr}"))]).unwrap();
            let err = call_for_stored_bytes("GET", OFFLINE, &route)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("through the proxy in https_proxy: the proxy requires authentication"),
                "{err}"
            );
            assert!(!err.contains("incorrect"), "{err}");
        }

        /// Serializes the tests that set proxy variables in the process
        /// environment, which a download reads.
        static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

        /// Run `f` with exactly `vars` set among the proxy variables, and
        /// restore the environment before returning.
        fn with_proxy_env<T>(vars: &[(&str, String)], f: impl FnOnce() -> T) -> T {
            const NAMES: [&str; 8] = [
                "https_proxy",
                "HTTPS_PROXY",
                "http_proxy",
                "HTTP_PROXY",
                "all_proxy",
                "ALL_PROXY",
                "no_proxy",
                "NO_PROXY",
            ];
            let _guard = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let saved: Vec<_> = NAMES
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect();
            for name in NAMES {
                std::env::remove_var(name);
            }
            for (name, value) in vars {
                std::env::set_var(name, value);
            }
            let out = f();
            for (name, value) in saved {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            out
        }

        fn empty_dir(tag: &str) -> std::path::PathBuf {
            let dir =
                std::env::temp_dir().join(format!("lumen-proxy-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        #[test]
        fn a_download_goes_through_the_proxy_the_environment_names() {
            let (addr, heads) = recording_proxy(FORBIDDEN);
            let dir = empty_dir("env");
            let result = with_proxy_env(&[("HTTPS_PROXY", format!("http://{addr}"))], || {
                let base = BaseUrl::local("https://hf.invalid".to_string());
                download_from(&base, "org/repo", "m.gguf", &dir, true)
            });
            let err = result.unwrap_err().to_string();
            let head = head_from(&heads);
            assert!(
                head.starts_with("CONNECT hf.invalid:443 HTTP/1.1\r\n"),
                "{head}"
            );
            assert!(err.contains("through the proxy in HTTPS_PROXY"), "{err}");
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn a_transfer_that_breaks_through_a_proxy_names_the_variable() {
            let (addr, heads) =
                recording_proxy(b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\npartial");
            let dir = empty_dir("broken");
            let base = BaseUrl::local("http://mirror.invalid".to_string());
            let result = with_proxy_env(&[("http_proxy", format!("http://{addr}"))], || {
                download_from(&base, "org/repo", "m.gguf", &dir, true)
            });
            let err = result.unwrap_err().to_string();
            let head = head_from(&heads);
            assert!(
                head.starts_with(
                    "GET http://mirror.invalid/org/repo/resolve/main/m.gguf HTTP/1.1\r\n"
                ),
                "{head}"
            );
            assert!(
                err.contains("read error during download through the proxy in http_proxy"),
                "{err}"
            );
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
            std::fs::remove_dir_all(&dir).ok();
        }

        const STORED: &[u8] = b"stored model bytes";

        /// A complete response for STORED, as an origin sends it.
        const COMPLETE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 18\r\n\r\nstored model bytes";

        #[test]
        fn a_download_through_an_http_proxy_completes() {
            let (addr, heads) = recording_proxy(COMPLETE);
            let dir = empty_dir("http-done");
            let base = BaseUrl::local("http://mirror.invalid".to_string());
            let result = with_proxy_env(&[("http_proxy", format!("http://{addr}"))], || {
                download_from(&base, "org/repo", "m.gguf", &dir, true)
            });
            let path = result.unwrap();
            assert!(head_from(&heads).starts_with(
                "GET http://mirror.invalid/org/repo/resolve/main/m.gguf HTTP/1.1\r\n"
            ));
            assert_eq!(std::fs::read(&path).unwrap(), STORED);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A SOCKS5 proxy on loopback that requires a user name and password
        /// (RFC 1929) and answers the tunnelled request with `answer`; hands
        /// over the user name, password and target it was sent.
        fn recording_socks5(
            answer: &'static [u8],
        ) -> (
            std::net::SocketAddr,
            std::sync::mpsc::Receiver<(String, String, String)>,
        ) {
            use std::io::{Read, Write};
            fn take(sock: &mut std::net::TcpStream, n: usize) -> Vec<u8> {
                let mut bytes = vec![0u8; n];
                sock.read_exact(&mut bytes).unwrap();
                bytes
            }
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let Ok((mut sock, _)) = listener.accept() else {
                    return;
                };
                sock.set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let greeting = take(&mut sock, 2);
                let methods = take(&mut sock, greeting[1] as usize);
                assert!(methods.contains(&2), "no username/password method offered");
                sock.write_all(&[5, 2]).unwrap();
                let len = take(&mut sock, 2)[1] as usize;
                let user = String::from_utf8(take(&mut sock, len)).unwrap();
                let len = take(&mut sock, 1)[0] as usize;
                let password = String::from_utf8(take(&mut sock, len)).unwrap();
                sock.write_all(&[1, 0]).unwrap();
                let request = take(&mut sock, 4);
                assert_eq!(request[3], 3, "the target is sent as a name");
                let len = take(&mut sock, 1)[0] as usize;
                let host = String::from_utf8(take(&mut sock, len)).unwrap();
                let port = u16::from_be_bytes(take(&mut sock, 2).try_into().unwrap());
                sock.write_all(&[5, 0, 0, 1, 0, 0, 0, 0, 0, 0]).unwrap();
                let mut head = Vec::new();
                while !head.ends_with(b"\r\n\r\n") {
                    head.push(take(&mut sock, 1)[0]);
                }
                tx.send((user, password, format!("{host}:{port}"))).unwrap();
                sock.write_all(answer).unwrap();
            });
            (addr, rx)
        }

        #[test]
        fn a_download_through_socks5_completes_with_the_decoded_credentials() {
            let (addr, sent) = recording_socks5(COMPLETE);
            let dir = empty_dir("socks-done");
            let base = BaseUrl::local("http://mirror.invalid".to_string());
            let value = format!("socks5://jdoe:P%40ss%3Aw%2Fd@{addr}");
            let result = with_proxy_env(&[("all_proxy", value)], || {
                download_from(&base, "org/repo", "m.gguf", &dir, true)
            });
            let path = result.unwrap();
            let (user, password, target) = sent
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the proxy was never contacted");
            assert_eq!(
                (user.as_str(), password.as_str(), target.as_str()),
                ("jdoe", "P@ss:w/d", "mirror.invalid:80")
            );
            assert_eq!(std::fs::read(&path).unwrap(), STORED);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A proxy that accepts connections and never answers: a request
        /// through it would stall.
        fn silent_proxy() -> (std::net::TcpListener, Route) {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let proxy = ureq::Proxy::new(format!("http://{addr}")).unwrap();
            (listener, Some((proxy, "https_proxy")))
        }

        #[test]
        fn the_advisory_size_is_not_asked_for_through_a_proxy() {
            let (listener, route) = silent_proxy();
            listener.set_nonblocking(true).unwrap();
            let started = std::time::Instant::now();
            assert_eq!(get_remote_size(OFFLINE, &route).unwrap(), None);
            assert!(started.elapsed() < std::time::Duration::from_secs(1));
            assert!(listener.accept().is_err(), "the proxy was contacted");
        }

        #[test]
        fn a_failure_through_a_proxy_names_the_variable() {
            let addr = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap()
            };
            let route = Some((
                ureq::Proxy::new(format!("http://{addr}")).unwrap(),
                "https_proxy",
            ));
            let err = call_for_stored_bytes("GET", OFFLINE, &route)
                .unwrap_err()
                .to_string();
            // Refused by the dead proxy, not a name lookup of the target.
            assert!(err.contains("Connection Failed"), "{err}");
            assert!(err.contains("through the proxy in https_proxy"), "{err}");
        }
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
        let (url, local) = split_repo_path("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf").unwrap();
        assert_eq!(url, "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf");
        assert_eq!(local, "Qwen3.8-27B-BF16-00001-of-00002.gguf");
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
        get_status: &'static str,
    ) -> (super::BaseUrl, std::thread::JoinHandle<Vec<String>>) {
        serve_like_hf_full(expected, head_extra, "200 OK", get_extra, get_status, false)
    }

    #[cfg(feature = "download")]
    fn serve_like_hf_with_head_status(
        expected: usize,
        head_extra: &'static str,
        head_status: &'static str,
    ) -> (super::BaseUrl, std::thread::JoinHandle<Vec<String>>) {
        serve_like_hf_full(expected, head_extra, head_status, "", "200 OK", false)
    }

    #[cfg(feature = "download")]
    fn serve_like_hf_full(
        expected: usize,
        head_extra: &'static str,
        head_status: &'static str,
        get_extra: &'static str,
        get_status: &'static str,
        extra_replaces_length: bool,
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
                    let (status, extra) = if is_head {
                        (head_status, head_extra)
                    } else {
                        (get_status, get_extra)
                    };
                    let length = if extra_replaces_length && !is_head {
                        String::new()
                    } else {
                        format!("Content-Length: {}\r\n", BODY.len())
                    };
                    format!("HTTP/1.1 {status}\r\n{length}{extra}Connection: close\r\n\r\n")
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
        let (base, server) = serve_like_hf(4, "", "", "200 OK");
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
        let (base, server) = serve_like_hf(4, "", "Content-Encoding: br\r\n", "200 OK");
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
            let (base, server) = serve_like_hf(4, "", extra, "200 OK");
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
        let (base, server) = serve_like_hf(4, "", "Content-Encoding: gzip\r\n", "200 OK");
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

    /// A readable `identity` cannot vouch for a second, unreadable value on
    /// another line of the same header: the value count is held against the
    /// line count.
    #[cfg(feature = "download")]
    #[test]
    fn identity_beside_an_unreadable_value_is_refused() {
        let (base, server) = serve_like_hf(
            4,
            "",
            "Content-Encoding: identity\r\nContent-Encoding: gzip\u{e9}\r\n",
            "200 OK",
        );
        let dir = scratch_dir("mixed");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(
            format!("{err}").contains("[\"identity\"] Content-Encoding"),
            "got {err}"
        );
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A header line with no colon is refused as malformed rather than
    /// aborting the process inside the header parser.
    #[cfg(feature = "download")]
    #[test]
    fn colonless_header_line_is_refused_not_a_panic() {
        let (base, server) = serve_like_hf(4, "", "Transfer-Encoding\r\n", "200 OK");
        let dir = scratch_dir("te");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(
            format!("{err}").contains("could not be read safely"),
            "got {err}"
        );
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An unsolicited partial response is refused: its Content-Length
    /// describes the part, and the completion check would accept it.
    #[cfg(feature = "download")]
    #[test]
    fn partial_content_is_refused() {
        let (base, server) = serve_like_hf(
            4,
            "",
            "Content-Range: bytes 0-17/4096\r\n",
            "206 Partial Content",
        );
        let dir = scratch_dir("206");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(format!("{err}").contains("answered 206"), "got {err}");
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two `Content-Length` lines that disagree are refused: the first one
    /// gates the completion check, so a second must not be able to differ.
    #[cfg(feature = "download")]
    #[test]
    fn conflicting_content_lengths_are_refused() {
        // A different number, and the same number spelled differently: the
        // lines must be byte-identical, not merely numerically equal.
        for (tag, second) in [
            ("cl-conflict", "Content-Length: 4096\r\n"),
            ("cl-spelling", "Content-Length: 018\r\n"),
        ] {
            let (base, server) = serve_like_hf(4, "", second, "200 OK");
            let dir = scratch_dir(tag);
            let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
            assert!(
                format!("{err}").contains("Content-Length"),
                "{tag}: got {err}"
            );
            assert!(
                entries(&dir).is_empty(),
                "{tag}: nothing may be left behind"
            );
            assert_eq!(server.join().unwrap().len(), 4, "{tag}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// A readable `Content-Length` beside one ureq cannot render is refused
    /// by the line count, since the unreadable line never reaches the values.
    #[cfg(feature = "download")]
    #[test]
    fn unreadable_second_content_length_is_refused() {
        let (base, server) = serve_like_hf(4, "", "Content-Length: 18\u{e9}\r\n", "200 OK");
        let dir = scratch_dir("cl-unreadable");
        let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
        assert!(format!("{err}").contains("Content-Length"), "got {err}");
        assert!(entries(&dir).is_empty(), "nothing may be left behind");
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A single `Content-Length` line that is not one representable integer (a
    /// comma list, garbage, a sign, a value past u64) is refused: its parse would otherwise fail and
    /// the download would fall back to the HEAD's advisory size.
    #[cfg(feature = "download")]
    #[test]
    fn single_unparseable_content_length_is_refused() {
        for (tag, extra) in [
            ("cl-list", "Content-Length: 18, 18\r\n"),
            ("cl-garbage", "Content-Length: 18abc\r\n"),
            ("cl-signed", "Content-Length: +18\r\n"),
            ("cl-overflow", "Content-Length: 18446744073709551616\r\n"),
        ] {
            let (base, server) = serve_like_hf_full(4, "", "200 OK", extra, "200 OK", true);
            let dir = scratch_dir(tag);
            let err = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap_err();
            assert!(
                format!("{err}").contains("Content-Length"),
                "{tag}: got {err}"
            );
            assert!(
                entries(&dir).is_empty(),
                "{tag}: nothing may be left behind"
            );
            assert_eq!(server.join().unwrap().len(), 4, "{tag}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// A HEAD the origin answers oddly (no content, a server error, or a line
    /// ureq cannot slice) costs only the advisory size; the download still
    /// completes.
    #[cfg(feature = "download")]
    #[test]
    fn odd_head_only_loses_the_advisory_size() {
        for (tag, head_extra, head_status) in [
            ("head-204", "", "204 No Content"),
            ("head-500", "", "500 Internal Server Error"),
            (
                "head-colonless",
                "Content-Length-Hint\r\nContent-Length\r\n",
                "200 OK",
            ),
        ] {
            let (base, server) = serve_like_hf_with_head_status(4, head_extra, head_status);
            let dir = scratch_dir(tag);
            let path = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), BODY, "{tag}");
            assert_eq!(server.join().unwrap().len(), 4, "{tag}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// An encoded HEAD only loses its advisory size; the GET, answering to
    /// every rule on its own, still publishes the file.
    #[cfg(feature = "download")]
    #[test]
    fn encoded_head_size_is_ignored() {
        let (base, server) = serve_like_hf(4, "Content-Encoding: gzip\r\n", "", "200 OK");
        let dir = scratch_dir("enc-head");
        let path = super::download_from(&base, "org/repo", "m.gguf", &dir, true).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), BODY);
        assert_eq!(server.join().unwrap().len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }
}
