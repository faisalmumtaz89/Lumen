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
    #[cfg(not(test))]
    const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    #[cfg(test)]
    const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

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

    /// Attempts in a row that add nothing to what this run has held of the
    /// file, after which a download stops.
    const ATTEMPTS: u32 = 6;

    /// The wait before a retry: doubled after each attempt that added
    /// nothing (1, 2, 4, 8 and 16 s), back to the start after one that did.
    #[cfg(not(test))]
    const BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
    #[cfg(test)]
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(10);

    /// Why a request, or an attempt at the rest of a file, ended early.
    enum Failure {
        /// A dropped, refused or stalled connection, or a busy server: another
        /// attempt may get further. Said in plain words.
        Transient(String),
        /// The server cannot continue from the bytes kept: the next attempt
        /// starts from the first byte.
        Restart(String),
        /// Anything another attempt would meet again.
        Final(DownloadError),
    }

    impl std::fmt::Display for Failure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Failure::Transient(reason) | Failure::Restart(reason) => f.write_str(reason),
                Failure::Final(e) => e.fmt(f),
            }
        }
    }

    const CANNOT_CONTINUE: &str = "the server could not continue from the bytes already downloaded";

    /// Whether an I/O error is the network's: a refused, reset, dropped or
    /// stalled connection, or no route to the host.
    fn is_network(e: &std::io::Error) -> bool {
        use std::io::ErrorKind;
        matches!(
            e.kind(),
            ErrorKind::ConnectionRefused
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::NotConnected
                | ErrorKind::BrokenPipe
                | ErrorKind::TimedOut
                | ErrorKind::WouldBlock
                | ErrorKind::UnexpectedEof
        ) || matches!(
            e.raw_os_error(),
            Some(libc::ENETUNREACH | libc::EHOSTUNREACH | libc::ENETDOWN)
        )
    }

    /// A network error on an open connection, in plain words; None for any
    /// other I/O error.
    fn network_failure(e: &std::io::Error, route: &Route) -> Option<String> {
        if !is_network(e) {
            return None;
        }
        let via = via(route);
        Some(match e.kind() {
            std::io::ErrorKind::UnexpectedEof => closed_early(route),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => {
                format!("no data arrived{via} for {} s", STALL_TIMEOUT.as_secs())
            }
            _ => format!("the connection{via} was lost"),
        })
    }

    fn closed_early(route: &Route) -> String {
        format!(
            "the connection{} closed before the file was complete",
            via(route)
        )
    }

    /// A failed request as a [`Failure`]: one a later attempt may get past
    /// in plain words, anything else with the request's own error.
    fn request_failure(
        method: &str,
        url: &str,
        route: &Route,
        ranged: bool,
        e: &ureq::Error,
    ) -> Failure {
        let proxy = route.as_ref().map(|(_, name)| *name);
        let transient = match e {
            ureq::Error::Status(416, _) if ranged => {
                return Failure::Restart(CANNOT_CONTINUE.to_string())
            }
            ureq::Error::Status(code, _) if *code == 408 || *code == 429 || *code >= 500 => {
                Some(format!("the server answered {code}{}", via(route)))
            }
            ureq::Error::Status(..) => None,
            ureq::Error::Transport(t) => {
                let io = std::error::Error::source(t)
                    .and_then(|source| source.downcast_ref::<std::io::Error>());
                match (t.kind(), proxy) {
                    (ureq::ErrorKind::Dns, Some(name)) => {
                        Some(format!("could not look up the proxy in {name}"))
                    }
                    (ureq::ErrorKind::Dns, None) => {
                        Some("could not look up the server's address".to_string())
                    }
                    (ureq::ErrorKind::ProxyConnect, Some(name)) => Some(format!(
                        "the proxy in {name} did not open a connection to the server"
                    )),
                    (ureq::ErrorKind::ConnectionFailed, _) if io.is_some_and(is_network) => {
                        Some(match proxy {
                            Some(name) => {
                                format!(
                                    "could not connect to the server through the proxy in {name}"
                                )
                            }
                            None => "could not connect to the server".to_string(),
                        })
                    }
                    (ureq::ErrorKind::Io, _) => io.and_then(|io| network_failure(io, route)),
                    _ => None,
                }
            }
        };
        if let Some(reason) = transient {
            return Failure::Transient(reason);
        }
        // ureq words a 401 or 407 from the proxy as "Provided proxy
        // credentials are incorrect", also when none were sent.
        let why = match e {
            ureq::Error::Transport(t) if t.kind() == ureq::ErrorKind::ProxyUnauthorized => {
                "the proxy requires authentication; lumen sends the user name and \
                 password in the proxy URL (user:password@host, percent-encoded) with \
                 Basic authentication"
                    .to_string()
            }
            _ => e.to_string(),
        };
        Failure::Final(DownloadError::Io(format!(
            "{method} request failed for {url}{}: {why}",
            via(route)
        )))
    }

    /// Issue a stored-bytes request and hand back only a usable response.
    /// With `resume`, the request asks for the rest of the file from the
    /// byte given, provided the file still has the ETag given.
    /// A header line with no colon is accepted by ureq's parser, and its
    /// value accessors then index past the end of the line — during response
    /// construction for some headers, on later lookups for others — so both
    /// the call and every value read are fenced: a panic is a refusal, not
    /// an abort of the process on a hostile origin.
    fn call_for_stored_bytes(
        method: &str,
        url: &str,
        route: &Route,
        resume: Option<(u64, &str)>,
    ) -> Result<ureq::Response, Failure> {
        let mut request = stored_bytes_request(method, url, route);
        if let Some((offset, etag)) = resume {
            request = request
                .set("Range", &format!("bytes={offset}-"))
                .set("If-Range", etag);
        }
        let resp = match fenced(|| request.call()) {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => return Err(request_failure(method, url, route, resume.is_some(), &e)),
            Err(_) => {
                return Err(Failure::Final(DownloadError::Io(format!(
                    "the response from {url} could not be read safely (a header line the parser cannot slice, or an internal error); refusing the response"
                ))))
            }
        };
        reject_unusable_response(&resp, resume.is_some()).map_err(Failure::Final)?;
        Ok(resp)
    }

    /// A response is stored only when it is a complete 200 (or, for a
    /// request for the rest of a file, a 206) whose every
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
    fn reject_unusable_response(resp: &ureq::Response, ranged: bool) -> Result<(), DownloadError> {
        if resp.status() != 200 && !(ranged && resp.status() == 206) {
            let stored = if ranged {
                "a 200 or 206"
            } else {
                "a complete 200"
            };
            return Err(DownloadError::Io(format!(
                "server answered {} {} for {}; only {stored} response is stored",
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

    pub(crate) fn model_url(base_url: &str, repo: &str, revision: &str, url_path: &str) -> String {
        format!("{base_url}/{repo}/resolve/{revision}/{url_path}")
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
        if let Err(e) = reject_unusable_response(&resp, false) {
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
        let read = std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| DownloadError::Io(format!("failed to read confirmation: {e}")))?;
        // The end of the input is no answer: a pull with no terminal and no
        // --yes must not download.
        if read == 0 {
            return Ok(false);
        }

        let trimmed = input.trim();
        Ok(trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("y")
            || trimmed.eq_ignore_ascii_case("yes"))
    }

    /// Download a GGUF file from HuggingFace.
    ///
    /// The file is downloaded to `{filename}.{machine}-{uid}.partial`, locked
    /// for the length of the download so a second download of the same file
    /// by the same user on the same machine waits for this one. A dropped or
    /// stalled connection is retried, continuing from the byte reached when
    /// the server identifies the file by a strong ETag; a download that stops
    /// keeps those bytes, and the next one continues from them. The full byte
    /// count is verified, then the file is hashed and atomically renamed to
    /// the final path; the `.sha256` sidecar is written after the rename (so
    /// a published file may briefly exist without its sidecar — harmless, as
    /// the sidecar is write-only metadata that no load path consults).
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

    /// Download one file of a checkpoint pinned to a commit: `file.path` at
    /// `revision` of `repo`, stored under its basename in `dest_dir` only when
    /// the bytes received have the size and SHA-256 pinned, so an upload that
    /// replaced the file is refused. The caller has confirmed the download.
    pub fn download_pinned(
        repo: &str,
        revision: &str,
        file: &crate::registry::CheckpointFile,
        dest_dir: &Path,
    ) -> Result<PathBuf, DownloadError> {
        download_file(
            &BaseUrl::hugging_face(),
            repo,
            revision,
            &file.path,
            dest_dir,
            true,
            Some(file),
        )
    }

    pub(crate) fn download_from(
        base_url: &BaseUrl,
        repo: &str,
        filename: &str,
        dest_dir: &Path,
        skip_confirm: bool,
    ) -> Result<PathBuf, DownloadError> {
        download_file(
            base_url,
            repo,
            "main",
            filename,
            dest_dir,
            skip_confirm,
            None,
        )
    }

    pub(crate) fn download_file(
        base_url: &BaseUrl,
        repo: &str,
        revision: &str,
        filename: &str,
        dest_dir: &Path,
        skip_confirm: bool,
        pinned: Option<&crate::registry::CheckpointFile>,
    ) -> Result<PathBuf, DownloadError> {
        // Validate (traversal-safe) and split into URL path + local basename.
        let (url_path, local_name) =
            super::split_repo_path(filename).map_err(DownloadError::InvalidFilename)?;
        let filename = local_name.as_str();

        let final_path = dest_dir.join(filename);
        // The .sha256 sidecar keeps its stable name BY DESIGN: it is shared
        // last-writer-wins metadata, written after the rename that publishes
        // the file, and write-only in production (only its unit test reads it
        // back). Because the cache keys on the flattened basename while the
        // hash is of the source URL (repo + path), two different sources
        // sharing a basename can leave a sidecar whose hash does not match the
        // resident file — harmless, since no load path consults it;
        // correctness rests on the atomic rename publishing only
        // fully-verified bytes.
        let sha_path = dest_dir.join(format!("{filename}.sha256"));
        // Reclaim BEFORE the cache-hit return: every call after the file is
        // published takes the cache-hit fast path, so the per-process partial
        // files crashed downloads left behind would otherwise never be
        // reclaimed. The scan is a small read_dir plus one libc::kill per
        // stale candidate — cheap.
        reclaim_stale_parts(dest_dir, filename);

        if is_published(&final_path) {
            eprintln!("Cache hit: {}", final_path.display());
            return Ok(final_path);
        }

        // The URL uses the full repo path, which may include a subdirectory;
        // the local file is the flat basename.
        let url = model_url(base_url.as_str(), repo, revision, &url_path);

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

        let Some(mut partial) = Partial::lock(dest_dir, filename, &final_path)? else {
            eprintln!("Cache hit: {}", final_path.display());
            return Ok(final_path);
        };
        if partial.have > 0 {
            eprintln!(
                "Resuming {filename}: {} already downloaded.",
                partial.done()
            );
        }

        eprintln!("Downloading: {url}");
        let pb = indicatif::ProgressBar::no_length();
        show_total(&pb, partial.total.or(size));
        pb.set_position(partial.have);

        // The most of each version of the file (by its strong ETag) this run
        // has held, the attempts in a row that added nothing, and the wait
        // before the next attempt.
        let mut held: Vec<(String, u64)> = partial
            .etag
            .iter()
            .map(|etag| (etag.clone(), partial.have))
            .collect();
        let mut idle = 0;
        let mut wait = BACKOFF;
        while partial.total != Some(partial.have) {
            let restarts = partial.restarts;
            let failure = match attempt(&url, &route, size, &mut partial, &pb) {
                Ok(()) => match verify_complete_transfer(partial.total, partial.have) {
                    Ok(()) => break,
                    Err(e) => Failure::Final(e),
                },
                Err(failure) => failure,
            };
            // What the attempt ended holding, before a restart drops it.
            let done = partial.done();
            let reason = match failure {
                Failure::Transient(reason) => reason,
                Failure::Restart(reason) => {
                    partial.restart(None, None)?;
                    reason
                }
                Failure::Final(e) => {
                    pb.abandon();
                    if partial.kept() {
                        return Err(DownloadError::Io(format!(
                            "{e}. The {done} downloaded so far is kept in {}; running the same \
                             command again continues from there.",
                            partial.path.display()
                        )));
                    }
                    partial.drop_unless_kept();
                    return Err(e);
                }
            };
            // Progress is more of one version of the file than this run has
            // held of it. A start from the first byte counts only for the
            // first version the run holds: a server that sends a version whole
            // again, or another version each time, could else be asked
            // without end.
            let restarted = partial.restarts != restarts;
            let progress = match &partial.etag {
                Some(etag) if partial.have > 0 => match held.iter_mut().find(|(v, _)| v == etag) {
                    Some((_, most)) => {
                        let more = !restarted && partial.have > *most;
                        if more {
                            *most = partial.have;
                        }
                        more
                    }
                    None => {
                        let first = held.is_empty();
                        held.push((etag.clone(), partial.have));
                        first
                    }
                },
                _ => false,
            };
            if progress {
                idle = 0;
                wait = BACKOFF;
            } else {
                idle += 1;
            }
            if idle == ATTEMPTS {
                pb.abandon();
                let next = if partial.kept() {
                    "resume"
                } else {
                    "try again"
                };
                let done = partial.done();
                partial.drop_unless_kept();
                return Err(DownloadError::Io(format!(
                    "stopped at {done} after {ATTEMPTS} attempts without progress ({reason}). \
                     Run the same command again to {next}."
                )));
            }
            pb.suspend(|| {
                eprintln!(
                    "Download interrupted at {done}: {reason}. Retrying in {} s.",
                    wait.as_secs()
                )
            });
            std::thread::sleep(wait);
            if idle > 0 {
                wait *= 2;
            }
        }
        pb.finish_with_message("download complete");

        // Hash through the descriptor that wrote the bytes; the lock keeps
        // every other lumen process away from them.
        use std::io::Seek;
        partial
            .file
            .seek(std::io::SeekFrom::Start(0))
            .map_err(|e| DownloadError::Io(format!("seek error before hashing: {e}")))?;
        let hash = sha256_of_reader(&mut partial.file)?;
        if !partial.holds_path() {
            return Err(partial.replaced());
        }
        if let Some(pinned) = pinned {
            if partial.have != pinned.size || hash != pinned.sha256 {
                let _ = partial.forget_source();
                let _ = std::fs::remove_file(&partial.path);
                return Err(DownloadError::Io(format!(
                    "{url}: received {} bytes with SHA-256 {hash}, not the pinned {} bytes with \
                     SHA-256 {}; nothing was stored",
                    partial.have, pinned.size, pinned.sha256
                )));
            }
        }

        // Atomic rename FIRST: .partial -> final, then the sidecar, with the
        // lock still held, so a process waiting for it finds the published
        // file. The rename publishes only fully size-verified bytes, so the
        // final file is correct the instant it appears. The sidecar write
        // that follows is best-effort write-only metadata; a crash or write
        // failure between the two can leave the final without a current
        // sidecar indefinitely, which is harmless because no load path reads
        // it (the cache hit checks only that the file exists and is nonempty).
        std::fs::rename(&partial.path, &final_path).map_err(|e| {
            DownloadError::Io(format!(
                "failed to rename {} -> {}: {e}",
                partial.path.display(),
                final_path.display()
            ))
        })?;
        let _ = partial.forget_source();
        drop(partial);

        // Write SHA-256 sidecar (shared name, last-writer-wins by design).
        std::fs::write(&sha_path, format!("{hash}  {filename}\n")).map_err(|e| {
            DownloadError::Io(format!("failed to write {}: {e}", sha_path.display()))
        })?;
        // What other downloads of the file kept is of no use from now on.
        reclaim_stale_parts(dest_dir, filename);

        eprintln!("Saved: {} (SHA-256: {hash})", final_path.display());
        Ok(final_path)
    }

    /// Options to read and write a file whose path may not be a symbolic
    /// link: in a cache directory others can write to, one could otherwise
    /// point the download at a file of the user's. Nor does the open wait
    /// on a named pipe planted there (a plain file ignores the flag).
    fn no_follow() -> std::fs::OpenOptions {
        use std::os::unix::fs::OpenOptionsExt;
        let mut options = std::fs::OpenOptions::new();
        options
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        options
    }

    /// `{filename}.{machine}-{uid}`, or `{filename}.{uid}` where the system
    /// keeps no identity for the machine, the name the files a download keeps
    /// start with. They are shared only by one user's downloads on one
    /// machine, as far as machines have identities of their own: a network
    /// file system may keep locks per machine, so a lock would exclude
    /// nothing between machines, and in a cache several users share another
    /// user's files may be writable but not removable. The machine is named
    /// by a hash of its stable identity, never by its host name, which a
    /// container gets anew each run; a container has no identity of its own,
    /// and its processes share the locks of the host's kernel. Machines that
    /// share an identity (containers, or clones that kept one) share the
    /// name, and on a cache between them only locks that work between
    /// machines keep their downloads apart.
    fn shared_stem(filename: &str) -> String {
        // SAFETY: getuid has no failure mode.
        let uid = unsafe { libc::getuid() };
        match machine_id() {
            Some(id) => {
                let digest = Sha256::digest(format!("lumen partial download {id}"));
                format!("{filename}.{}-{uid}", &hex_encode(&digest)[..16])
            }
            None => format!("{filename}.{uid}"),
        }
    }

    /// The machine's hardware UUID.
    #[cfg(target_os = "macos")]
    fn machine_id() -> Option<String> {
        let mut id = [0u8; 16];
        let wait = libc::timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        // SAFETY: gethostuuid writes the 16 bytes of a uuid_t into `id`.
        (unsafe { libc::gethostuuid(id.as_mut_ptr(), &wait) } == 0).then(|| hex_encode(&id))
    }

    /// The machine ID systemd keeps, when there is one: 32 hex digits
    /// (absent, empty or "uninitialized" in most containers).
    #[cfg(not(target_os = "macos"))]
    fn machine_id() -> Option<String> {
        let id = std::fs::read_to_string("/etc/machine-id").ok()?;
        let id = id.trim();
        (id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())).then(|| id.to_string())
    }

    /// Why a partial that exists cannot be opened for this download, when a
    /// file of its own would do instead; None when nothing in the directory
    /// can be written (a read-only file system).
    fn unusable(e: &std::io::Error) -> Option<String> {
        match e.raw_os_error() {
            Some(libc::EROFS) => None,
            Some(libc::ELOOP) => Some("it is a symbolic link".to_string()),
            _ if e.kind() == std::io::ErrorKind::PermissionDenied => {
                Some("this user may not write it".to_string())
            }
            _ => Some(e.to_string()),
        }
    }

    /// Whether `path` is a published (nonempty) file.
    fn is_published(path: &Path) -> bool {
        std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0)
    }

    /// The record at `path`, the file's full length and strong ETag, when
    /// what is there holds one. A named pipe planted there reads as empty.
    fn read_record(path: &Path) -> Option<(u64, String)> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .ok()?;
        let mut text = String::new();
        (&file).take(1024).read_to_string(&mut text).ok()?;
        let (total, etag) = text.trim_end().split_once(' ')?;
        let total = total.parse::<u64>().ok()?;
        is_strong_etag(etag).then(|| (total, etag.to_string()))
    }

    /// The bytes of one download kept so far, locked by this process. They
    /// are kept in `{filename}.{machine}-{uid}.partial`, and while they can be
    /// continued, the same name with `.meta` added records the file's full
    /// length and the strong ETag it was sent with: without both, no later
    /// response can be matched to them. A download that cannot use that
    /// partial safely goes to a file of its own, with no record, that no
    /// later run continues.
    struct Partial {
        file: std::fs::File,
        path: PathBuf,
        /// The record's path; None for a download in a file of its own.
        meta: Option<PathBuf>,
        /// Bytes in the file, all of one version of the source.
        have: u64,
        /// The file's full length, once a response has said.
        total: Option<u64>,
        /// The version's strong ETag, once a response has sent one.
        etag: Option<String>,
        /// How often the file was started over: an attempt that leaves this
        /// alone added to the bytes kept before it.
        restarts: u64,
    }

    impl Partial {
        /// Lock the partial download of `final_path`, waiting while another
        /// lumen process downloads the same file, and read what it holds;
        /// None when the file was published meanwhile. The lock belongs to
        /// the partial file and ends when the file closes, also when its
        /// process dies. Its holder publishes by renaming the file, so a
        /// waiter that wakes holding a file no longer at the path opens the
        /// path again. When the partial cannot be used safely (a file system
        /// without locks, a partial this user may not write, or one that is
        /// not a plain file with a single name), the download goes to a file
        /// of its own, as downloads did before they could be continued.
        fn lock(
            dest_dir: &Path,
            filename: &str,
            final_path: &Path,
        ) -> Result<Option<Self>, DownloadError> {
            use std::os::unix::fs::MetadataExt;
            use std::os::unix::io::AsRawFd;
            let stem = shared_stem(filename);
            let path = dest_dir.join(format!("{stem}.partial"));
            let failed = |what: &str, e: std::io::Error| {
                DownloadError::Io(format!("failed to {what} {}: {e}", path.display()))
            };
            let own = |why: String| Self::own(dest_dir, filename, &why).map(Some);
            let mut told = false;
            let (file, have) = loop {
                if is_published(final_path) {
                    return Ok(None);
                }
                let (file, created) = match no_follow().create_new(true).open(&path) {
                    Ok(file) => (file, true),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        match no_follow().open(&path) {
                            Ok(file) => (file, false),
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                            Err(e) => match unusable(&e) {
                                Some(why) => {
                                    return own(format!(
                                        "{} cannot be used ({why})",
                                        path.display()
                                    ))
                                }
                                None => return Err(failed("open", e)),
                            },
                        }
                    }
                    Err(e) => return Err(failed("create", e)),
                };
                let unusable = || {
                    own(format!(
                        "{} is not a plain file with a single name",
                        path.display()
                    ))
                };
                if !file.metadata().map_err(|e| failed("inspect", e))?.is_file() {
                    return unusable();
                }
                // SAFETY: flock on a descriptor this function owns.
                if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                    let e = std::io::Error::last_os_error();
                    let code = e.raw_os_error().unwrap_or_default();
                    if matches!(code, libc::ENOLCK | libc::ENOSYS)
                        || code == libc::ENOTSUP
                        || code == libc::EOPNOTSUPP
                    {
                        // No download on this file system writes the shared
                        // partial, so one this call created goes again.
                        if created {
                            let _ = std::fs::remove_file(&path);
                        }
                        return own(format!(
                            "{} does not support file locks",
                            dest_dir.display()
                        ));
                    }
                    if code != libc::EWOULDBLOCK {
                        return Err(failed("lock", e));
                    }
                    if !told {
                        eprintln!("Waiting for another download of {filename} to finish...");
                        told = true;
                    }
                    // SAFETY: as above; a signal ends the wait early, so wait again.
                    while unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
                        let e = std::io::Error::last_os_error();
                        if e.kind() != std::io::ErrorKind::Interrupted {
                            return Err(failed("lock", e));
                        }
                    }
                }
                let held = file.metadata().map_err(|e| failed("inspect", e))?;
                let named = std::fs::symlink_metadata(&path).ok();
                if !named.is_some_and(|m| (m.dev(), m.ino()) == (held.dev(), held.ino())) {
                    continue;
                }
                // Another name for the file, a hard link, would have the
                // download empty and rewrite a file of the user's.
                if held.nlink() != 1 {
                    return unusable();
                }
                break (file, held.len());
            };
            let meta = dest_dir.join(format!("{stem}.partial.meta"));
            let (total, etag) =
                read_record(&meta).map_or((None, None), |(total, etag)| (Some(total), Some(etag)));
            let mut partial = Self {
                file,
                path,
                meta: Some(meta),
                have,
                total,
                etag,
                restarts: 0,
            };
            if partial.resumable() {
                use std::io::Seek;
                partial
                    .file
                    .seek(std::io::SeekFrom::Start(have))
                    .map_err(|e| partial.failed("seek in", e))?;
            } else {
                partial.restart(None, None)?;
            }
            Ok(Some(partial))
        }

        /// A download in a file of its own, `{filename}.{pid}-{nonce}.part`,
        /// which reclaim_stale_parts removes once its process is gone. It
        /// retries like any other, but no later run continues it: nothing
        /// guards it against a second writer.
        fn own(dest_dir: &Path, filename: &str, why: &str) -> Result<Self, DownloadError> {
            eprintln!("Note: {why}, so if this download stops, the next run starts it over.");
            for attempt in 0u32..16 {
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(attempt, |d| d.subsec_nanos())
                    .wrapping_add(attempt);
                let path = dest_dir.join(format!("{filename}.{}-{nonce}.part", std::process::id()));
                match no_follow().create_new(true).open(&path) {
                    Ok(file) => {
                        return Ok(Self {
                            file,
                            path,
                            meta: None,
                            have: 0,
                            total: None,
                            etag: None,
                            restarts: 0,
                        })
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => {
                        return Err(DownloadError::Io(format!(
                            "failed to create {}: {e}",
                            path.display()
                        )))
                    }
                }
            }
            Err(DownloadError::Io(format!(
                "could not create a file of its own for {filename} in {}",
                dest_dir.display()
            )))
        }

        fn failed(&self, what: &str, e: std::io::Error) -> DownloadError {
            DownloadError::Io(format!("failed to {what} {}: {e}", self.path.display()))
        }

        /// Whether a later response can be matched to the bytes kept: they
        /// are some or all of a file whose length and strong ETag are known.
        fn resumable(&self) -> bool {
            self.etag.is_some()
                && self
                    .total
                    .is_some_and(|total| (1..=total).contains(&self.have))
        }

        /// Whether a later run can continue the bytes kept.
        fn kept(&self) -> bool {
            self.meta.is_some() && self.resumable() && self.holds_path()
        }

        /// Whether the path still names the file this download writes. The
        /// path and its record are acted on by name, and someone may remove
        /// the file while the download runs, after which another download of
        /// the same file makes a new one there.
        fn holds_path(&self) -> bool {
            use std::os::unix::fs::MetadataExt;
            let held = self.file.metadata().map(|m| (m.dev(), m.ino()));
            let named = std::fs::symlink_metadata(&self.path).map(|m| (m.dev(), m.ino()));
            matches!((held, named), (Ok(held), Ok(named)) if held == named)
        }

        fn replaced(&self) -> DownloadError {
            DownloadError::Io(format!(
                "{} was removed or replaced during the download, so nothing was published; \
                 run the same command again",
                self.path.display()
            ))
        }

        /// Empty the file for a transfer from the first byte of the version
        /// `total` and `etag` describe, and record them when both are known.
        /// The file is emptied before the record changes, so no record ever
        /// vouches for bytes of another version, and the old record is
        /// removed before a new one is created, so a link planted in its
        /// place is never written through.
        fn restart(
            &mut self,
            total: Option<u64>,
            etag: Option<String>,
        ) -> Result<(), DownloadError> {
            use std::io::Seek;
            if self.meta.is_some() && !self.holds_path() {
                return Err(self.replaced());
            }
            self.file.set_len(0).map_err(|e| self.failed("empty", e))?;
            self.file
                .seek(std::io::SeekFrom::Start(0))
                .map_err(|e| self.failed("seek in", e))?;
            self.have = 0;
            self.total = total;
            self.etag = etag;
            self.restarts += 1;
            self.forget_source()?;
            match (&self.meta, self.total, &self.etag) {
                (Some(meta), Some(total), Some(etag)) => no_follow()
                    .create_new(true)
                    .open(meta)
                    .and_then(|mut file| file.write_all(format!("{total} {etag}\n").as_bytes()))
                    .map_err(|e| {
                        DownloadError::Io(format!("failed to write {}: {e}", meta.display()))
                    }),
                _ => Ok(()),
            }
        }

        fn append(&mut self, bytes: &[u8]) -> Result<(), DownloadError> {
            self.file
                .write_all(bytes)
                .map_err(|e| DownloadError::Io(format!("write error: {e}")))?;
            self.have += bytes.len() as u64;
            Ok(())
        }

        /// Remove the record of the file's length and ETag.
        fn forget_source(&self) -> Result<(), DownloadError> {
            let Some(meta) = &self.meta else {
                return Ok(());
            };
            match std::fs::remove_file(meta) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(DownloadError::Io(
                    format!("failed to remove {}: {e}", meta.display()),
                )),
                _ => Ok(()),
            }
        }

        /// On a failure: keep bytes a later run can continue, remove the rest,
        /// and leave alone a path that names another download's file.
        fn drop_unless_kept(&self) {
            if !self.kept() && self.holds_path() {
                let _ = std::fs::remove_file(&self.path);
                let _ = self.forget_source();
            }
        }

        /// "1.2 GB of 15.0 GB", or the bytes alone while the length is unknown.
        fn done(&self) -> String {
            let have = crate::cache::format_size(self.have);
            match self.total {
                Some(total) => format!("{have} of {}", crate::cache::format_size(total)),
                None => have,
            }
        }
    }

    /// A strong entity tag: quoted, not weak (`W/`), not empty, and made of
    /// the visible ASCII an entity tag may hold (RFC 9110), which is also
    /// what a request header can carry. Only these can vouch that two
    /// responses carry bytes of the same file.
    fn is_strong_etag(tag: &str) -> bool {
        tag.len() > 2
            && tag.starts_with('"')
            && tag.ends_with('"')
            && tag[1..tag.len() - 1]
                .bytes()
                .all(|b| b == b'!' || (b'#'..=b'~').contains(&b))
    }

    /// The response's ETag, when it sent one and that one is strong.
    fn strong_etag(resp: &ureq::Response) -> Result<Option<String>, DownloadError> {
        Ok(match header_values(resp, "etag")?.as_slice() {
            [tag] if is_strong_etag(tag) => Some(tag.clone()),
            _ => None,
        })
    }

    /// The response's `Content-Length`, which [`reject_unusable_response`]
    /// has checked is one plain number.
    fn content_length(resp: &ureq::Response) -> Result<Option<u64>, DownloadError> {
        Ok(header_values(resp, "content-length")?
            .first()
            .and_then(|length| length.parse::<u64>().ok()))
    }

    /// Whether a 206 carries exactly the rest of the file `partial` holds
    /// the start of: from its next byte to the last byte of the same length,
    /// under the same ETag. The ETag is required: a CDN may serve a range
    /// whatever the request's If-Range says (Hugging Face's does), so only
    /// the 206's own ETag shows the bytes are of the file already kept.
    fn continues(resp: &ureq::Response, partial: &Partial) -> Result<bool, DownloadError> {
        let (Some(total), Some(etag)) = (partial.total, &partial.etag) else {
            return Ok(false);
        };
        let rest = format!("bytes {}-{}/{total}", partial.have, total - 1);
        Ok(header_values(resp, "content-range")? == [rest]
            && content_length(resp)?.map_or(true, |length| length == total - partial.have)
            && strong_etag(resp)?.as_ref() == Some(etag))
    }

    /// Show the bytes against `total`, or on their own while it is unknown.
    fn show_total(pb: &indicatif::ProgressBar, total: Option<u64>) {
        match total {
            Some(total) => {
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                        .progress_chars("=>-"),
                );
                pb.set_length(total);
            }
            None => {
                pb.set_style(
                    indicatif::ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
                );
                pb.unset_length();
            }
        }
    }

    /// One request for the rest of the file: the bytes after those `partial`
    /// holds when a response can be matched to them, else the whole file.
    /// Appends what arrives until the body ends.
    fn attempt(
        url: &str,
        route: &Route,
        advisory: Option<u64>,
        partial: &mut Partial,
        pb: &indicatif::ProgressBar,
    ) -> Result<(), Failure> {
        let etag = partial.etag.clone().filter(|_| partial.resumable());
        let resume = etag.as_deref().map(|etag| (partial.have, etag));
        let resp = call_for_stored_bytes("GET", url, route, resume)?;
        if resp.status() == 206 {
            if !continues(&resp, partial).map_err(Failure::Final)? {
                return Err(Failure::Restart(CANNOT_CONTINUE.to_string()));
            }
        } else {
            if resume.is_some() {
                pb.suspend(|| eprintln!("The server sent the whole file again; starting over."));
            }
            let total = content_length(&resp).map_err(Failure::Final)?.or(advisory);
            let etag = strong_etag(&resp).map_err(Failure::Final)?;
            partial.restart(total, etag).map_err(Failure::Final)?;
            show_total(pb, total);
        }
        pb.set_position(partial.have);
        let mut reader = resp.into_reader();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf).map_err(|e| {
                // ureq's chunk decoder reports a body cut at the edge of a
                // chunk as invalid input.
                if e.kind() == std::io::ErrorKind::InvalidInput {
                    return Failure::Transient(closed_early(route));
                }
                match network_failure(&e, route) {
                    Some(reason) => Failure::Transient(reason),
                    None => Failure::Final(DownloadError::Io(format!(
                        "read error during download{}: {e}",
                        via(route)
                    ))),
                }
            })?;
            if n == 0 {
                break;
            }
            partial.append(&buf[..n]).map_err(Failure::Final)?;
            pb.set_position(partial.have);
        }
        // A body that ends early without a network error: one sent without a
        // length or chunked framing, closed by a dropped connection.
        if partial.total.is_some_and(|total| partial.have < total) {
            return Err(Failure::Transient(closed_early(route)));
        }
        Ok(())
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
    /// download path to hash the bytes through the descriptor that wrote
    /// them.
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

    /// Best-effort reclamation of the `{filename}.<pid>[-<nonce>].part`
    /// files a download in a file of its own (and every download of an
    /// earlier version) writes, left behind by runs that crashed or were
    /// stopped. Deletion requires BOTH a stale mtime
    /// (>60s grace — a live writer refreshes mtime on every chunk, in any
    /// PID namespace) AND either ESRCH in our namespace or >24h staleness
    /// (pid numbers are namespace-local, so a foreign container's live
    /// writer can look dead here; mtime freshness is the cross-namespace
    /// protection). EPERM means alive under another user and keeps.
    /// Legacy fixed-name `{filename}.part` litter is age-gated at >1h —
    /// same mtime-freshness rationale. Once the file is published, no
    /// download continues a `.partial` of it, whichever machine or user kept
    /// it, since every later pull is a cache hit: one not written for a
    /// minute is removed with its record, and a record whose partial is gone
    /// goes too (a download still writing a removed one stops before it
    /// publishes).
    pub fn reclaim_stale_parts(dest_dir: &std::path::Path, filename: &str) {
        let Ok(entries) = std::fs::read_dir(dest_dir) else {
            return;
        };
        let published = is_published(&dest_dir.join(filename));
        let prefix = format!("{filename}.");
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(rest) = name.strip_prefix(&prefix) else {
                continue;
            };
            if rest.ends_with(".partial") {
                let idle = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age.as_secs() > 60);
                if published && idle {
                    let _ = std::fs::remove_file(entry.path());
                    let _ = std::fs::remove_file(dest_dir.join(format!("{name}.meta")));
                }
                continue;
            }
            if let Some(partial) = name
                .strip_suffix(".meta")
                .filter(|p| p.ends_with(".partial"))
            {
                if published && std::fs::symlink_metadata(dest_dir.join(partial)).is_err() {
                    let _ = std::fs::remove_file(entry.path());
                }
                continue;
            }
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
                let url = model_url("https://huggingface.co", &src.repo, "main", src.file());
                let got = get_remote_size(&url, &None)
                    .expect("HEAD")
                    .expect("Content-Length");
                assert_eq!(got, size, "{key}: {url}");
            }
        }

        #[test]
        #[ignore = "requires network access to huggingface.co"]
        fn image_checkpoint_files_resolve_to_their_pinned_sizes() {
            let reg = crate::registry::load_registry();
            let entry = reg.resolve("qwen-image").expect("qwen-image");
            let ckpt = entry.checkpoint.as_ref().expect("checkpoint");
            for file in &ckpt.files {
                let url = model_url(
                    "https://huggingface.co",
                    &ckpt.repo,
                    &ckpt.revision,
                    &file.path,
                );
                let got = get_remote_size(&url, &None)
                    .expect("HEAD")
                    .expect("Content-Length");
                assert_eq!(got, file.size, "{}: {url}", file.path);
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

        /// A proxy on loopback that answers `connections` connections with
        /// `answer` each and hands over the request heads it read: the CONNECT
        /// for an https URL, the request itself for an http one.
        fn recording_proxy(
            answer: &'static [u8],
            connections: u32,
        ) -> (std::net::SocketAddr, std::sync::mpsc::Receiver<String>) {
            use std::io::{Read, Write};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                for _ in 0..connections {
                    let Ok((mut sock, _)) = listener.accept() else {
                        return;
                    };
                    sock.set_read_timeout(Some(std::time::Duration::from_secs(5)))
                        .unwrap();
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    while !head.ends_with(b"\r\n\r\n") && sock.read(&mut byte).is_ok_and(|n| n == 1)
                    {
                        head.push(byte[0]);
                    }
                    let _ = tx.send(String::from_utf8_lossy(&head).into_owned());
                    sock.write_all(answer).ok();
                }
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
                let (addr, heads) = recording_proxy(FORBIDDEN, 1);
                let value = value.replace("{addr}", &addr.to_string());
                let route = route(OFFLINE, &[("https_proxy", &value)]).unwrap();
                let err = call_for_stored_bytes("GET", OFFLINE, &route, None)
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
                1,
            );
            let route = route(OFFLINE, &[("https_proxy", &format!("http://{addr}"))]).unwrap();
            let err = call_for_stored_bytes("GET", OFFLINE, &route, None)
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

        /// A proxy that refuses to open a connection is asked again, like any
        /// failure another attempt may get past, and the download then stops
        /// with the reason, naming the variable.
        #[test]
        fn a_download_goes_through_the_proxy_the_environment_names() {
            let (addr, heads) = recording_proxy(FORBIDDEN, ATTEMPTS);
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
            assert!(
                err.contains("(the proxy in HTTPS_PROXY did not open a connection to the server)"),
                "{err}"
            );
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// Every attempt breaks the same way; with no ETag nothing is kept,
        /// so the download stops without leaving a file behind.
        #[test]
        fn a_transfer_that_breaks_through_a_proxy_names_the_variable() {
            let (addr, heads) = recording_proxy(
                b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\npartial",
                ATTEMPTS,
            );
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
            assert_eq!(
                err,
                "stopped at 7 B of 64 B after 6 attempts without progress (the connection \
                 through the proxy in http_proxy closed before the file was complete). Run the \
                 same command again to try again."
            );
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
            std::fs::remove_dir_all(&dir).ok();
        }

        const STORED: &[u8] = b"stored model bytes";

        /// A complete response for STORED, as an origin sends it.
        const COMPLETE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 18\r\n\r\nstored model bytes";

        #[test]
        fn a_download_through_an_http_proxy_completes() {
            let (addr, heads) = recording_proxy(COMPLETE, 1);
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
            let err = call_for_stored_bytes("GET", OFFLINE, &route, None)
                .unwrap_err()
                .to_string();
            // Refused by the dead proxy, not a name lookup of the target.
            assert_eq!(
                err,
                "could not connect to the server through the proxy in https_proxy"
            );
        }
    }

    #[cfg(test)]
    mod resume {
        //! Downloads that are cut, stopped, changed or run twice at once,
        //! against a server on loopback.
        use super::*;
        use std::sync::{Arc, Mutex};

        const FILE: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        /// Another version of FILE, the same length and different from the
        /// first byte, so a file mixed from the two equals neither.
        const NEW: &[u8] = b"9876543210ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const V1: &str = "\"v1\"";
        const V2: &str = "\"v2\"";
        /// A strong ETag as long as the ones Hugging Face's CDN sends.
        const LONG: &str = "\"a44b0e06af0c97cece312f6dca52b3639d038d1a74a2e194b22e159f9ccdbb21\"";

        /// A 200 for all of `body`, cut after `sent` bytes.
        fn whole(etag: Option<&str>, body: &[u8], sent: usize) -> Vec<u8> {
            let etag = etag
                .map(|tag| format!("ETag: {tag}\r\n"))
                .unwrap_or_default();
            let mut out = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{etag}Connection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            out.extend_from_slice(&body[..sent]);
            out
        }

        /// A 206 with the given `Content-Range`, ETag and body.
        fn part(range: &str, etag: Option<&str>, body: &[u8]) -> Vec<u8> {
            let etag = etag
                .map(|tag| format!("ETag: {tag}\r\n"))
                .unwrap_or_default();
            let mut out = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: {range}\r\nContent-Length: {}\r\n\
                 {etag}Connection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            out.extend_from_slice(body);
            out
        }

        /// A 206 for the rest of `body` from `from`.
        fn rest(etag: &str, body: &[u8], from: usize) -> Vec<u8> {
            let range = format!("bytes {from}-{}/{}", body.len() - 1, body.len());
            part(&range, Some(etag), &body[from..])
        }

        fn status(line: &str) -> Vec<u8> {
            format!("HTTP/1.1 {line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .into_bytes()
        }

        /// The byte a request asks for the rest of the file from.
        fn range_from(head: &str) -> Option<usize> {
            head.lines().find_map(|line| {
                line.strip_prefix("range: bytes=")?
                    .strip_suffix('-')?
                    .parse()
                    .ok()
            })
        }

        /// A server on loopback that answers each request with what `respond`
        /// returns for its head (lowercased), each connection on a thread of
        /// its own, and closes the connection; a HEAD gets the head of the
        /// response only. It takes `connections` connections, or what arrives
        /// in ten seconds, refuses any more, and hands back the heads.
        fn serve(
            connections: usize,
            respond: impl Fn(&str) -> Vec<u8> + Send + Sync + 'static,
        ) -> (BaseUrl, std::thread::JoinHandle<Vec<String>>) {
            use std::io::{BufRead, BufReader};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let base = BaseUrl::local(format!("http://{}", listener.local_addr().unwrap()));
            listener.set_nonblocking(true).unwrap();
            let respond = Arc::new(respond);
            let heads = Arc::new(Mutex::new(Vec::new()));
            let server = std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                let mut handlers = Vec::new();
                while handlers.len() < connections && std::time::Instant::now() < deadline {
                    let Ok((stream, _)) = listener.accept() else {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    };
                    let (respond, heads) = (respond.clone(), heads.clone());
                    handlers.push(std::thread::spawn(move || {
                        stream.set_nonblocking(false).unwrap();
                        let wire = Some(std::time::Duration::from_secs(5));
                        stream.set_read_timeout(wire).unwrap();
                        stream.set_write_timeout(wire).unwrap();
                        let mut reader = BufReader::new(stream);
                        let mut head = String::new();
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                                break;
                            }
                            head.push_str(&line.to_ascii_lowercase());
                        }
                        heads.lock().unwrap().push(head.clone());
                        let mut response = respond(&head);
                        if head.starts_with("head ") {
                            let end = response
                                .windows(4)
                                .position(|w| w == b"\r\n\r\n")
                                .map_or(response.len(), |at| at + 4);
                            response.truncate(end);
                        }
                        let _ = reader.get_mut().write_all(&response);
                    }));
                }
                drop(listener);
                for handler in handlers {
                    handler.join().unwrap();
                }
                let heads = heads.lock().unwrap().clone();
                heads
            });
            (base, server)
        }

        fn gets(heads: &[String]) -> Vec<&String> {
            heads
                .iter()
                .filter(|head| head.starts_with("get "))
                .collect()
        }

        fn scratch(tag: &str) -> PathBuf {
            let dir =
                std::env::temp_dir().join(format!("lumen-resume-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        fn names(dir: &Path) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }

        /// This user's partial of m.gguf on this machine, and its record.
        fn kept() -> String {
            format!("{}.partial", shared_stem("m.gguf"))
        }

        fn record() -> String {
            format!("{}.meta", kept())
        }

        /// `names`, sorted as `names()` lists a directory.
        fn listed(names: &[&str]) -> Vec<String> {
            let mut names: Vec<String> = names.iter().map(|name| name.to_string()).collect();
            names.sort();
            names
        }

        /// Leave the bytes and record a stopped download of FILE leaves.
        fn seed(dir: &Path, bytes: &[u8], meta: Option<&str>) {
            std::fs::write(dir.join(kept()), bytes).unwrap();
            if let Some(meta) = meta {
                std::fs::write(dir.join(record()), meta).unwrap();
            }
        }

        fn pull(base: &BaseUrl, dir: &Path) -> Result<PathBuf, DownloadError> {
            download_from(base, "org/repo", "m.gguf", dir, true)
        }

        #[test]
        fn a_cut_transfer_continues_from_the_byte_it_reached() {
            let (base, server) = serve(3, |head| match range_from(head) {
                None => whole(Some(V1), FILE, 10),
                Some(from) => rest(V1, FILE, from),
            });
            let dir = scratch("cut");
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 2, "{heads:?}");
            assert!(
                gets[1].contains("\r\nrange: bytes=10-\r\n")
                    && gets[1].contains("\r\nif-range: \"v1\"\r\n"),
                "{}",
                gets[1]
            );
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn a_stopped_download_keeps_its_bytes_and_the_next_run_continues() {
            // Cut at byte 10, then busy until the retries run out.
            let (base, server) = serve(2 + ATTEMPTS as usize, |head| match range_from(head) {
                None => whole(Some(LONG), FILE, 10),
                Some(_) => status("503 Service Unavailable"),
            });
            let dir = scratch("stopped");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 10 B of 36 B after 6 attempts without progress (the server \
                 answered 503). Run the same command again to resume."
            );
            assert_eq!(names(&dir), listed(&[&kept(), &record()]));
            assert_eq!(
                std::fs::read_to_string(dir.join(record())).unwrap(),
                format!("36 {LONG}\n")
            );
            assert_eq!(gets(&server.join().unwrap()).len(), 1 + ATTEMPTS as usize);

            let (base, server) = serve(2, |head| match range_from(head) {
                None => whole(Some(LONG), FILE, FILE.len()),
                Some(from) => rest(LONG, FILE, from),
            });
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 1, "{heads:?}");
            assert_eq!(range_from(gets[0]), Some(10), "{}", gets[0]);
            assert!(
                gets[0].contains(&format!("\r\nif-range: {LONG}\r\n")),
                "{}",
                gets[0]
            );
            std::fs::remove_dir_all(&dir).ok();
        }

        /// As Hugging Face's CDN does: the range is served whatever If-Range
        /// says, under the ETag of the file as it is now.
        #[test]
        fn a_changed_file_starts_over_when_the_server_ignores_if_range() {
            let (base, server) = serve(3, |head| match range_from(head) {
                None => whole(Some(V2), NEW, NEW.len()),
                Some(from) => rest(V2, NEW, from),
            });
            let dir = scratch("changed");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), NEW);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 2, "{heads:?}");
            assert_eq!(range_from(gets[0]), Some(10));
            assert_eq!(range_from(gets[1]), None);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// As the HTTP standard has it: a changed file is sent whole.
        #[test]
        fn a_changed_file_sent_whole_replaces_the_bytes_kept() {
            let (base, server) = serve(2, |_| whole(Some(V2), NEW, NEW.len()));
            let dir = scratch("changed-whole");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), NEW);
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 1, "{heads:?}");
            assert_eq!(range_from(gets[0]), Some(10));
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn an_answer_that_does_not_continue_the_bytes_kept_starts_over() {
            let rest_of_file = &FILE[10..];
            for (tag, answer) in [
                // Each answer is wrong in one respect only.
                ("start", part("bytes 0-25/36", Some(V1), &FILE[..26])),
                ("total", part("bytes 10-35/40", Some(V1), rest_of_file)),
                ("length", part("bytes 10-35/36", Some(V1), &[b'x'; 30])),
                ("no-etag", part("bytes 10-35/36", None, rest_of_file)),
                (
                    "weak",
                    part("bytes 10-35/36", Some("W/\"v1\""), rest_of_file),
                ),
                ("unsatisfiable", status("416 Range Not Satisfiable")),
            ] {
                let (base, server) = serve(3, move |head| match range_from(head) {
                    None => whole(Some(V1), FILE, FILE.len()),
                    Some(_) => answer.clone(),
                });
                let dir = scratch(tag);
                seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
                let path = pull(&base, &dir).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                let heads = server.join().unwrap();
                let gets = gets(&heads);
                assert_eq!(gets.len(), 2, "{tag}: {heads:?}");
                assert_eq!(range_from(gets[1]), None, "{tag}");
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A body sent without a length ends at a clean close, so only the
        /// length the HEAD gave shows that it was cut.
        #[test]
        fn a_body_without_a_length_that_ends_early_is_continued() {
            let (base, server) = serve(3, |head| match range_from(head) {
                _ if head.starts_with("head ") => whole(Some(V1), FILE, FILE.len()),
                None => {
                    let mut out =
                        format!("HTTP/1.1 200 OK\r\nETag: {V1}\r\nConnection: close\r\n\r\n")
                            .into_bytes();
                    out.extend_from_slice(&FILE[..10]);
                    out
                }
                Some(from) => rest(V1, FILE, from),
            });
            let dir = scratch("no-length");
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 2, "{heads:?}");
            assert_eq!(range_from(gets[1]), Some(10));
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A 206 sent in chunks has no length of its own; its range and tag
        /// are enough to continue.
        #[test]
        fn a_chunked_answer_continues_the_bytes_kept() {
            let (base, server) = serve(3, |head| match range_from(head) {
                None => whole(Some(V1), FILE, 10),
                Some(from) => {
                    let body = &FILE[from..];
                    let mut out = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-35/36\r\n\
                         ETag: {V1}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n\
                         {:x}\r\n",
                        body.len()
                    )
                    .into_bytes();
                    out.extend_from_slice(body);
                    out.extend_from_slice(b"\r\n0\r\n\r\n");
                    out
                }
            });
            let dir = scratch("chunked");
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 2, "{heads:?}");
            assert_eq!(range_from(gets[1]), Some(10));
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A chunked body cut at the edge of a chunk, before a chunk's line
        /// ending or inside a chunk is continued like any other cut body.
        #[test]
        fn a_chunked_body_cut_anywhere_is_continued() {
            for (tag, cut) in [
                ("between", "a\r\n0123456789\r\n"),
                ("before-line-end", "a\r\n0123456789"),
                ("inside", "a\r\n01234"),
            ] {
                let first = std::sync::atomic::AtomicBool::new(true);
                let (base, server) = serve(3, move |head| match range_from(head) {
                    _ if head.starts_with("head ") => whole(Some(V1), FILE, FILE.len()),
                    None if first.swap(false, std::sync::atomic::Ordering::SeqCst) => format!(
                        "HTTP/1.1 200 OK\r\nETag: {V1}\r\nTransfer-Encoding: chunked\r\n\
                         Connection: close\r\n\r\n{cut}"
                    )
                    .into_bytes(),
                    None => whole(Some(V1), FILE, FILE.len()),
                    Some(from) => rest(V1, FILE, from),
                });
                let dir = scratch(&format!("chunk-cut-{tag}"));
                let path = pull(&base, &dir).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                assert_eq!(gets(&server.join().unwrap()).len(), 2, "{tag}");
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A file that changed on the server starts over, and the bytes of
        /// the new version count as progress from then on, however few they
        /// are next to the bytes of the old version that were dropped.
        #[test]
        fn after_a_new_start_each_continued_attempt_is_progress() {
            let (base, server) = serve(10, |head| {
                let old = head.contains("\r\nif-range: \"v1\"\r\n");
                match range_from(head) {
                    _ if head.starts_with("head ") => whole(Some(V2), NEW, NEW.len()),
                    Some(from) if old => rest(V2, NEW, from),
                    None => whole(Some(V2), NEW, 5),
                    Some(from) => {
                        let mut cut = rest(V2, NEW, from);
                        cut.truncate(cut.len() - NEW.len().saturating_sub(from + 5));
                        cut
                    }
                }
            });
            let dir = scratch("new-start");
            seed(&dir, &FILE[..30], Some("36 \"v1\"\n"));
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), NEW);
            let heads = server.join().unwrap();
            assert_eq!(gets(&heads).len(), 9, "{heads:?}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A server that sends the whole file whatever is asked, and is cut
        /// a little further each time, never lets the download continue, so
        /// it stops after the fixed number of attempts.
        #[test]
        fn a_server_that_always_starts_over_is_not_asked_without_end() {
            let sent = std::sync::atomic::AtomicUsize::new(10);
            let (base, server) = serve(2 + ATTEMPTS as usize, move |head| {
                if head.starts_with("head ") {
                    return whole(Some(V1), FILE, FILE.len());
                }
                whole(
                    Some(V1),
                    FILE,
                    sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                )
            });
            let dir = scratch("always-whole");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 16 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to resume."
            );
            assert_eq!(gets(&server.join().unwrap()).len(), 1 + ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A server that continues the file one time and sends it whole the
        /// next, cut at the same bytes each time, never gives the download
        /// more than it held before, so it stops after the fixed number of
        /// attempts.
        #[test]
        fn a_server_that_continues_only_at_times_is_not_asked_without_end() {
            let (base, server) = serve(3 + ATTEMPTS as usize, |head| match range_from(head) {
                _ if head.starts_with("head ") => whole(Some(V1), FILE, FILE.len()),
                Some(10) => {
                    let mut cut = rest(V1, FILE, 10);
                    cut.truncate(cut.len() - (FILE.len() - 20));
                    cut
                }
                _ => whole(Some(V1), FILE, 10),
            });
            let dir = scratch("sometimes");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 20 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to resume."
            );
            assert_eq!(gets(&server.join().unwrap()).len(), 2 + ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A server whose ETag for the file alternates between two values, as
        /// servers behind one address can when each makes its own: it
        /// continues each version from byte 10 to byte 20, then sends the
        /// other whole, cut at byte 10. Each version is judged by the most of
        /// it the run has held, so the download stops after the fixed number
        /// of attempts.
        #[test]
        fn a_server_that_alternates_two_versions_is_not_asked_without_end() {
            let (base, server) = serve(5 + ATTEMPTS as usize, |head| {
                if head.starts_with("head ") {
                    return whole(Some(V1), FILE, FILE.len());
                }
                let asked = if head.contains("\r\nif-range: \"v1\"\r\n") {
                    Some(V1)
                } else if head.contains("\r\nif-range: \"v2\"\r\n") {
                    Some(V2)
                } else {
                    None
                };
                match (range_from(head), asked) {
                    (Some(10), Some(tag)) => {
                        let mut cut = rest(tag, FILE, 10);
                        cut.truncate(cut.len() - (FILE.len() - 20));
                        cut
                    }
                    (Some(_), Some(tag)) => whole(Some(if tag == V1 { V2 } else { V1 }), FILE, 10),
                    _ => whole(Some(V1), FILE, 10),
                }
            });
            let dir = scratch("two-versions");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 20 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to resume."
            );
            assert_eq!(gets(&server.join().unwrap()).len(), 4 + ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A server that sends another version of the file with every answer,
        /// each cut: the download holds a new version each time, but only the
        /// first counts, so it stops after the fixed number of attempts.
        #[test]
        fn a_new_version_in_every_answer_is_not_asked_without_end() {
            let answers = std::sync::atomic::AtomicUsize::new(0);
            let (base, server) = serve(2 + ATTEMPTS as usize, move |head| {
                if head.starts_with("head ") {
                    return whole(Some(V1), FILE, FILE.len());
                }
                let n = answers.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                whole(Some(&format!("\"v{n}\"")), FILE, 10)
            });
            let dir = scratch("every-answer-new");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 10 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to resume."
            );
            assert_eq!(gets(&server.join().unwrap()).len(), 1 + ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// The bytes a run starts with count as held: a server that sends the
        /// whole file again, cut where the bytes kept end, adds nothing.
        #[test]
        fn bytes_a_run_starts_with_count_as_held() {
            let (base, server) = serve(1 + ATTEMPTS as usize, |_| whole(Some(V1), FILE, 10));
            let dir = scratch("held-at-start");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 10 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to resume."
            );
            assert_eq!(gets(&server.join().unwrap()).len(), ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// An attempt that brings no bytes of the file is not progress, even
        /// the first one.
        #[test]
        fn an_attempt_that_brings_no_bytes_is_not_progress() {
            let (base, server) = serve(1 + ATTEMPTS as usize, |_| whole(Some(V1), FILE, 0));
            let dir = scratch("no-bytes");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 0 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to try again."
            );
            assert!(names(&dir).is_empty(), "{:?}", names(&dir));
            assert_eq!(gets(&server.join().unwrap()).len(), ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn without_a_strong_etag_a_cut_transfer_starts_over() {
            for (tag, etag) in [
                ("none", None),
                ("weak", Some("W/\"v1\"")),
                ("empty", Some("\"\"")),
                ("unquoted", Some("v1\"")),
                ("two", Some("\"v1\"\r\nETag: \"v1\"")),
            ] {
                let cut = std::sync::atomic::AtomicBool::new(true);
                let (base, server) = serve(3, move |head| {
                    let first_get = head.starts_with("get ")
                        && cut.swap(false, std::sync::atomic::Ordering::SeqCst);
                    whole(etag, FILE, if first_get { 10 } else { FILE.len() })
                });
                let dir = scratch(&format!("etag-{tag}"));
                let path = pull(&base, &dir).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                let heads = server.join().unwrap();
                let gets = gets(&heads);
                assert_eq!(gets.len(), 2, "{tag}: {heads:?}");
                assert_eq!(range_from(gets[1]), None, "{tag}");
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        #[test]
        fn a_download_that_cannot_be_continued_leaves_nothing_when_it_stops() {
            let (base, server) = serve(1 + ATTEMPTS as usize, |_| whole(None, FILE, 10));
            let dir = scratch("no-etag-stops");
            let started = std::time::Instant::now();
            let err = pull(&base, &dir).unwrap_err().to_string();
            // Five waits between six attempts, each twice the one before.
            assert!(started.elapsed() >= BACKOFF * 31, "{:?}", started.elapsed());
            assert_eq!(
                err,
                "stopped at 10 B of 36 B after 6 attempts without progress (the connection \
                 closed before the file was complete). Run the same command again to try again."
            );
            assert!(names(&dir).is_empty(), "{:?}", names(&dir));
            assert_eq!(gets(&server.join().unwrap()).len(), ATTEMPTS as usize);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn bytes_no_record_vouches_for_are_not_continued() {
            for (tag, meta) in [
                ("no-record", None),
                ("weak", Some("36 W/\"v1\"\n")),
                ("shorter", Some("5 \"v1\"\n")),
                ("garbled", Some("36\n")),
            ] {
                let (base, server) = serve(2, |_| whole(Some(V1), FILE, FILE.len()));
                let dir = scratch(&format!("record-{tag}"));
                // Longer than the file, so bytes left past its end would show.
                seed(&dir, &[b'x'; 50], meta);
                let path = pull(&base, &dir).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                let heads = server.join().unwrap();
                let gets = gets(&heads);
                assert_eq!(gets.len(), 1, "{tag}: {heads:?}");
                assert_eq!(range_from(gets[0]), None, "{tag}");
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A record whose ETag holds a byte no request header can carry, as a
        /// damaged or planted record could, is not continued: asking with it
        /// would fail the same way on every run.
        #[test]
        fn a_record_with_a_tag_no_request_can_carry_is_not_continued() {
            let (base, server) = serve(2, |_| whole(Some(V1), FILE, FILE.len()));
            let dir = scratch("unsendable");
            seed(&dir, &NEW[..10], Some("36 \"v\u{1}1\"\n"));
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 1, "{heads:?}");
            assert_eq!(range_from(gets[0]), None);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn a_partial_that_is_already_complete_is_published_without_fetching() {
            // Room for one request past the HEAD, so a GET would be answered
            // and seen; the test takes it itself when none came.
            let (base, server) = serve(2, |_| whole(Some(V1), FILE, FILE.len()));
            let dir = scratch("complete");
            seed(&dir, FILE, Some("36 \"v1\"\n"));
            let path = pull(&base, &dir).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            let addr = base.as_str().trim_start_matches("http://").to_string();
            if let Ok(mut probe) = std::net::TcpStream::connect(addr) {
                let _ = probe.write_all(b"HEAD /probe HTTP/1.1\r\n\r\n");
            }
            assert!(gets(&server.join().unwrap()).is_empty());
            std::fs::remove_dir_all(&dir).ok();
        }

        /// Where the shared partial cannot be used safely, the download goes
        /// to a file of its own: it still completes, still continues a cut
        /// transfer in the same run, and leaves the file in the way alone.
        #[test]
        fn a_partial_that_cannot_be_used_safely_is_left_alone() {
            use std::os::unix::fs::PermissionsExt;
            // Root may write any file, so only another user sees the unwritable case.
            let root = unsafe { libc::geteuid() } == 0;
            for tag in ["hard-link", "symbolic-link", "pipe", "unwritable"] {
                if tag == "unwritable" && root {
                    continue;
                }
                let (base, server) = serve(3, |head| match range_from(head) {
                    None => whole(Some(V1), FILE, 10),
                    Some(from) => rest(V1, FILE, from),
                });
                let dir = scratch(&format!("unusable-{tag}"));
                let other = dir.join("other");
                std::fs::write(&other, b"someone else's bytes").unwrap();
                let partial = dir.join(kept());
                match tag {
                    "hard-link" => std::fs::hard_link(&other, &partial).unwrap(),
                    "symbolic-link" => std::os::unix::fs::symlink(&other, &partial).unwrap(),
                    "pipe" => {
                        let name = std::ffi::CString::new(partial.to_str().unwrap()).unwrap();
                        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o644) }, 0);
                    }
                    _ => {
                        std::fs::copy(&other, &partial).unwrap();
                        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o444))
                            .unwrap();
                    }
                }
                let path = pull(&base, &dir).unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                assert_eq!(
                    std::fs::read(&other).unwrap(),
                    b"someone else's bytes",
                    "{tag}"
                );
                if tag != "pipe" {
                    assert_eq!(
                        std::fs::read(&partial).unwrap(),
                        b"someone else's bytes",
                        "{tag}"
                    );
                }
                assert_eq!(
                    names(&dir),
                    listed(&["m.gguf", &kept(), "m.gguf.sha256", "other"]),
                    "{tag}"
                );
                let heads = server.join().unwrap();
                let gets = gets(&heads);
                assert_eq!(gets.len(), 2, "{tag}: {heads:?}");
                assert_eq!(range_from(gets[1]), Some(10), "{tag}");
                std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o644)).ok();
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A link or a named pipe planted where the record is kept is removed,
        /// never written through or waited on: the download completes and the
        /// file a link points at is left alone.
        #[test]
        fn a_link_or_pipe_in_place_of_the_record_is_not_followed() {
            for tag in ["hard-link", "symbolic-link", "pipe"] {
                let (base, server) = serve(2, |_| whole(Some(V1), FILE, FILE.len()));
                let dir = scratch(&format!("record-{tag}"));
                let other = dir.join("other");
                std::fs::write(&other, b"36 \"v1\"\n").unwrap();
                let record = dir.join(record());
                match tag {
                    "hard-link" => std::fs::hard_link(&other, &record).unwrap(),
                    "symbolic-link" => std::os::unix::fs::symlink(&other, &record).unwrap(),
                    _ => {
                        let name = std::ffi::CString::new(record.to_str().unwrap()).unwrap();
                        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o644) }, 0);
                    }
                }
                let (done, finished) = std::sync::mpsc::channel();
                let (base_url, dir_path) = (base.as_str().to_string(), dir.clone());
                std::thread::spawn(move || {
                    let _ = done.send(pull(&BaseUrl::local(base_url), &dir_path));
                });
                let path = finished
                    .recv_timeout(std::time::Duration::from_secs(20))
                    .expect("the download waited on the record")
                    .unwrap();
                assert_eq!(std::fs::read(&path).unwrap(), FILE, "{tag}");
                assert_eq!(std::fs::read(&other).unwrap(), b"36 \"v1\"\n", "{tag}");
                assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256", "other"], "{tag}");
                server.join().unwrap();
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A download in a file of its own that stops leaves nothing and says
        /// the next run starts over.
        #[test]
        fn a_download_in_a_file_of_its_own_leaves_nothing_when_it_stops() {
            let (base, server) = serve(2 + ATTEMPTS as usize, |head| match range_from(head) {
                None => whole(Some(V1), FILE, 10),
                Some(_) => status("503 Service Unavailable"),
            });
            let dir = scratch("own-stops");
            let other = dir.join("other");
            std::fs::write(&other, b"x").unwrap();
            std::fs::hard_link(&other, dir.join(kept())).unwrap();
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(
                err,
                "stopped at 10 B of 36 B after 6 attempts without progress (the server \
                 answered 503). Run the same command again to try again."
            );
            assert_eq!(names(&dir), listed(&[&kept(), "other"]));
            server.join().unwrap();
            std::fs::remove_dir_all(&dir).ok();
        }

        /// The per-process files of downloads that are gone are removed; a
        /// live one, a fresh one and the shared partial are not.
        #[test]
        fn only_files_of_downloads_that_are_gone_are_reclaimed() {
            let dir = scratch("reclaim");
            let gone = 2_147_483_646; // above any pid_max: no such process
            let old = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
            let live = std::process::id();
            for (name, when) in [
                (format!("m.gguf.{gone}-1.part"), old),
                (
                    format!("m.gguf.{gone}-2.part"),
                    std::time::SystemTime::now(),
                ),
                (format!("m.gguf.{live}-3.part"), old),
                ("m.gguf.part".to_string(), old),
                (kept(), old),
                (record(), old),
                (format!("n.gguf.{gone}-4.part"), old),
            ] {
                let file = std::fs::File::create(dir.join(name)).unwrap();
                file.set_modified(when).unwrap();
            }
            reclaim_stale_parts(&dir, "m.gguf");
            assert_eq!(
                names(&dir),
                listed(&[
                    &format!("m.gguf.{gone}-2.part"),
                    &format!("m.gguf.{live}-3.part"),
                    &kept(),
                    &record(),
                    &format!("n.gguf.{gone}-4.part"),
                ])
            );
            std::fs::remove_dir_all(&dir).ok();
        }

        /// G1's stall: part of the body, then nothing for longer than the
        /// stall timeout while the connection stays open.
        #[test]
        fn a_stalled_transfer_continues_from_the_byte_it_reached() {
            use std::io::{BufRead, BufReader};
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let base = BaseUrl::local(format!("http://{}", listener.local_addr().unwrap()));
            let server = std::thread::spawn(move || {
                let mut heads = Vec::new();
                let mut held = Vec::new();
                for stream in listener.incoming().take(3) {
                    let mut reader = BufReader::new(stream.unwrap());
                    let mut head = String::new();
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        head.push_str(&line.to_ascii_lowercase());
                    }
                    let answer = match range_from(&head) {
                        _ if head.starts_with("head ") => whole(Some(V1), FILE, 0),
                        None => whole(Some(V1), FILE, 10),
                        Some(from) => rest(V1, FILE, from),
                    };
                    let mut stream = reader.into_inner();
                    stream.write_all(&answer).unwrap();
                    if range_from(&head).is_none() && head.starts_with("get ") {
                        // Say nothing more, and keep the connection open.
                        held.push(stream);
                    }
                    heads.push(head);
                }
                heads
            });
            let dir = scratch("stall");
            let started = std::time::Instant::now();
            let path = pull(&base, &dir).unwrap();
            assert!(
                started.elapsed() >= STALL_TIMEOUT,
                "{:?}",
                started.elapsed()
            );
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            let heads = server.join().unwrap();
            let gets = gets(&heads);
            assert_eq!(gets.len(), 2, "{heads:?}");
            assert_eq!(range_from(gets[1]), Some(10));
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A waiter that wakes after the holder published, to find a new
        /// partial another download has since made, holds the published file:
        /// it must look again, not continue or empty what it holds.
        #[test]
        fn a_waiter_that_wakes_to_another_partial_leaves_the_published_file_alone() {
            use std::os::unix::io::AsRawFd;
            let (base, server) = serve(1, |_| whole(Some(V1), FILE, FILE.len()));
            let dir = scratch("waiter");
            let partial = dir.join(kept());
            std::fs::write(&partial, FILE).unwrap();
            let held = std::fs::File::open(&partial).unwrap();
            assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX) }, 0);
            let (base, dir2) = (base, dir.clone());
            let waiter = std::thread::spawn(move || pull(&base, &dir2));
            // The waiter asks the size, then blocks on the lock.
            assert_eq!(server.join().unwrap().len(), 1);
            std::thread::sleep(std::time::Duration::from_millis(200));
            // As the holder does: publish by renaming the locked file; then a
            // third download makes a new partial before the lock is let go.
            std::fs::rename(&partial, dir.join("m.gguf")).unwrap();
            std::fs::write(&partial, b"").unwrap();
            drop(held);
            let path = waiter.join().unwrap().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// Once the file is published, what downloads of it by other
        /// machines or users kept is of no use: a partial not written for a
        /// minute goes with its record, and so does a record without its
        /// partial. Before, or while a partial is being written, nothing goes.
        #[test]
        fn what_other_downloads_kept_goes_once_the_file_is_published() {
            let dir = scratch("kept-by-others");
            let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
            let other_machine = format!("m.gguf.0123456789abcdef-{}.partial", unsafe {
                libc::getuid()
            });
            // Another user's, on a machine with no identity.
            let container = format!("m.gguf.{}.partial", unsafe { libc::getuid() } + 1);
            let written_now = "m.gguf.fedcba9876543210-7.partial".to_string();
            let record_alone = "m.gguf.00000000000000aa-9.partial.meta".to_string();
            let names_kept = [
                other_machine.clone(),
                format!("{other_machine}.meta"),
                container.clone(),
                format!("{container}.meta"),
                written_now.clone(),
                format!("{written_now}.meta"),
                record_alone.clone(),
                kept(),
                record(),
            ];
            for name in &names_kept {
                let file = std::fs::File::create(dir.join(name)).unwrap();
                if name != &written_now {
                    file.set_modified(old).unwrap();
                }
            }
            reclaim_stale_parts(&dir, "m.gguf");
            let mut before: Vec<&str> = names_kept.iter().map(String::as_str).collect();
            assert_eq!(names(&dir), listed(&before));

            std::fs::write(dir.join("m.gguf"), FILE).unwrap();
            reclaim_stale_parts(&dir, "m.gguf");
            before.retain(|name| name.starts_with(written_now.as_str()));
            before.push("m.gguf");
            assert_eq!(names(&dir), listed(&before));
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A download publishes, then sweeps what other downloads of the file
        /// kept.
        #[test]
        fn a_download_reclaims_what_others_kept_once_it_publishes() {
            let (base, server) = serve(2, |_| whole(Some(V1), FILE, FILE.len()));
            let dir = scratch("reclaim-after-publish");
            // Another user's, on a machine with no identity.
            let other = format!("m.gguf.{}.partial", unsafe { libc::getuid() } + 1);
            let container = dir.join(&other);
            std::fs::write(&container, &FILE[..5]).unwrap();
            std::fs::write(dir.join(format!("{other}.meta")), "36 \"v1\"\n").unwrap();
            let file = std::fs::File::options()
                .write(true)
                .open(&container)
                .unwrap();
            file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(600))
                .unwrap();
            pull(&base, &dir).unwrap();
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            server.join().unwrap();
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A download is published only when its length is known and every
        /// byte of it arrived: with no length anywhere, or with more bytes than
        /// the length, nothing is published.
        #[test]
        fn a_download_whose_length_does_not_hold_is_not_published() {
            let (base, server) = serve(2, |_| {
                let mut out = format!("HTTP/1.1 200 OK\r\nETag: {V1}\r\nConnection: close\r\n\r\n")
                    .into_bytes();
                out.extend_from_slice(FILE);
                out
            });
            let dir = scratch("no-length-anywhere");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert!(err.contains("no Content-Length"), "{err}");
            assert!(names(&dir).is_empty(), "{:?}", names(&dir));
            server.join().unwrap();
            std::fs::remove_dir_all(&dir).ok();

            let (base, server) = serve(2, |head| {
                let from = range_from(head).unwrap_or(0);
                let mut body = FILE[from..].to_vec();
                body.extend_from_slice(b"XXXX");
                let mut out = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-35/36\r\n\
                     ETag: {V1}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n\
                     {:x}\r\n",
                    body.len()
                )
                .into_bytes();
                out.extend_from_slice(&body);
                out.extend_from_slice(b"\r\n0\r\n\r\n");
                out
            });
            let dir = scratch("past-the-end");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert_eq!(err, "size mismatch: expected 36 bytes, got 40 bytes");
            assert!(names(&dir).is_empty(), "{:?}", names(&dir));
            server.join().unwrap();
            std::fs::remove_dir_all(&dir).ok();
        }

        /// An answer that continues the bytes kept is stored only if it is
        /// unencoded, like any other; the bytes kept stay for the next run.
        #[test]
        fn an_encoded_answer_that_continues_the_bytes_kept_is_refused() {
            let (base, server) = serve(2, |head| {
                let from = range_from(head).unwrap_or(0);
                let encoded = vec![0x1f_u8; FILE.len() - from];
                let mut out = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-35/36\r\n\
                     Content-Length: {}\r\nContent-Encoding: gzip\r\nETag: {V1}\r\n\
                     Connection: close\r\n\r\n",
                    encoded.len()
                )
                .into_bytes();
                out.extend_from_slice(&encoded);
                out
            });
            let dir = scratch("encoded-206");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert!(
                err.contains("server sent [\"gzip\"] Content-Encoding")
                    && err.ends_with("running the same command again continues from there."),
                "{err}"
            );
            assert_eq!(std::fs::read(dir.join(kept())).unwrap(), &FILE[..10]);
            assert_eq!(names(&dir), listed(&[&kept(), &record()]));
            server.join().unwrap();
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A download whose partial is removed while it runs, and made anew by
        /// a second download of the same file, never publishes the second
        /// one's bytes, nor removes or rewrites its partial or record, whether
        /// it then finishes, fails or has to start over; the second finishes
        /// the file.
        #[test]
        fn a_partial_removed_while_its_download_runs_is_left_to_the_next() {
            use std::io::{BufRead, BufReader};
            for case in ["finishes", "fails", "starts-over"] {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                let base = BaseUrl::local(format!("http://{}", listener.local_addr().unwrap()));
                let (to_first, first_waits) = std::sync::mpsc::channel::<()>();
                let (to_second, second_waits) = std::sync::mpsc::channel::<()>();
                let server = std::thread::spawn(move || {
                    let mut gets = 0;
                    let mut handlers = Vec::new();
                    let mut waits = [first_waits, second_waits].into_iter();
                    let connections = if case == "finishes" { 5 } else { 6 };
                    for stream in listener.incoming().take(connections) {
                        let mut reader = BufReader::new(stream.unwrap());
                        let mut head = String::new();
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                                break;
                            }
                            head.push_str(&line.to_ascii_lowercase());
                        }
                        let mut stream = reader.into_inner();
                        if head.starts_with("head ") {
                            stream.write_all(&whole(Some(V1), FILE, 0)).unwrap();
                            continue;
                        }
                        gets += 1;
                        let answer = match range_from(&head) {
                            // The first download's own answer and the second's:
                            // part of the file, the rest once the test says.
                            None => {
                                let sent = if gets == 1 { 10 } else { 5 };
                                stream.write_all(&whole(Some(V1), FILE, sent)).unwrap();
                                let wait = waits.next().unwrap();
                                handlers.push(std::thread::spawn(move || {
                                    wait.recv().unwrap();
                                    if sent == 10 && case == "finishes" {
                                        let _ = stream.write_all(&FILE[10..]);
                                    }
                                }));
                                continue;
                            }
                            // The first download asks again.
                            Some(10) if case == "fails" => status("404 Not Found"),
                            Some(10) => whole(Some(V2), NEW, NEW.len()),
                            Some(from) => rest(V1, FILE, from),
                        };
                        stream.write_all(&answer).unwrap();
                    }
                    for handler in handlers {
                        handler.join().unwrap();
                    }
                });
                let dir = scratch(&format!("removed-{case}"));
                let partial = dir.join(kept());
                let grown_to = |len: u64| {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                    while std::fs::metadata(&partial).map_or(true, |m| m.len() != len) {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "{case}: never {len} B"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                };
                std::thread::scope(|s| {
                    let first = s.spawn(|| pull(&base, &dir));
                    grown_to(10);
                    std::fs::remove_file(&partial).unwrap();
                    let second = s.spawn(|| pull(&base, &dir));
                    grown_to(5);
                    to_first.send(()).unwrap();
                    let err = first.join().unwrap().unwrap_err().to_string();
                    let replaced = format!(
                        "{} was removed or replaced during the download, so nothing was \
                         published; run the same command again",
                        partial.display()
                    );
                    if case == "fails" {
                        assert!(err.contains("404") && !err.contains("kept in"), "{err}");
                    } else {
                        assert_eq!(err, replaced, "{case}");
                    }
                    assert!(!dir.join("m.gguf").exists(), "{case}");
                    assert_eq!(std::fs::read(&partial).unwrap(), &FILE[..5], "{case}");
                    assert_eq!(
                        std::fs::read_to_string(dir.join(record())).unwrap(),
                        "36 \"v1\"\n",
                        "{case}"
                    );
                    to_second.send(()).unwrap();
                    let path = second.join().unwrap().unwrap();
                    assert_eq!(std::fs::read(&path).unwrap(), FILE, "{case}");
                });
                server.join().unwrap();
                assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"], "{case}");
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// A pull sweeps what dead downloads of the same file left, also
        /// when it finds the file published and fetches nothing.
        #[test]
        fn a_pull_reclaims_what_dead_downloads_left() {
            let dir = scratch("reclaim-on-pull");
            std::fs::write(dir.join("m.gguf"), FILE).unwrap();
            let left = dir.join("m.gguf.2147483646-1.part");
            let file = std::fs::File::create(&left).unwrap();
            file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(7200))
                .unwrap();
            let base = BaseUrl::local("http://127.0.0.1:9".to_string());
            assert_eq!(pull(&base, &dir).unwrap(), dir.join("m.gguf"));
            assert_eq!(names(&dir), ["m.gguf"]);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn a_failure_another_attempt_cannot_change_stops_at_once() {
            let (base, server) = serve(2, |_| status("404 Not Found"));
            let dir = scratch("404");
            let err = pull(&base, &dir).unwrap_err().to_string();
            assert!(err.contains("404"), "{err}");
            assert!(names(&dir).is_empty(), "{:?}", names(&dir));
            assert_eq!(gets(&server.join().unwrap()).len(), 1);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A failure no other attempt can get past keeps the bytes a later
        /// run can continue, and says where they are.
        #[test]
        fn a_failure_that_stops_at_once_keeps_what_can_be_continued() {
            let (base, server) = serve(2, |head| {
                if head.starts_with("head ") {
                    whole(Some(V1), FILE, FILE.len())
                } else {
                    status("403 Forbidden")
                }
            });
            let dir = scratch("403-kept");
            seed(&dir, &FILE[..10], Some("36 \"v1\"\n"));
            let err = pull(&base, &dir).unwrap_err().to_string();
            let tail = format!(
                "403. The 10 B of 36 B downloaded so far is kept in {}; running the same \
                 command again continues from there.",
                dir.join(kept()).display()
            );
            assert!(err.ends_with(&tail), "{err}");
            assert_eq!(names(&dir), listed(&[&kept(), &record()]));
            assert_eq!(std::fs::read(dir.join(kept())).unwrap(), &FILE[..10]);
            assert_eq!(gets(&server.join().unwrap()).len(), 1);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A download in a file of its own leaves a file reclaim_stale_parts
        /// knows: removed once stale, kept while its download may be running.
        #[test]
        fn a_file_of_its_own_is_reclaimed_once_stale() {
            let dir = scratch("own-reclaim");
            let stale = Partial::own(&dir, "m.gguf", "a test").unwrap();
            let fresh = Partial::own(&dir, "m.gguf", "a test").unwrap();
            // This process is alive, so only its age makes the file stale.
            stale
                .file
                .set_modified(
                    std::time::SystemTime::now() - std::time::Duration::from_secs(25 * 3600),
                )
                .unwrap();
            reclaim_stale_parts(&dir, "m.gguf");
            let fresh_name = fresh.path.file_name().unwrap().to_string_lossy();
            assert_eq!(names(&dir), [fresh_name]);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// The files a download keeps are named for this machine, by a hash
        /// of its identity, and for this user, so only downloads a lock can
        /// exclude share them.
        #[test]
        fn kept_files_are_named_for_this_machine_and_user() {
            let uid = unsafe { libc::getuid() };
            let stem = shared_stem("m.gguf");
            let identified = cfg!(target_os = "macos")
                || std::fs::read_to_string("/etc/machine-id").is_ok_and(|id| {
                    let id = id.trim();
                    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
                });
            if identified {
                let machine = stem
                    .strip_prefix("m.gguf.")
                    .and_then(|rest| rest.strip_suffix(&format!("-{uid}")))
                    .unwrap_or_else(|| panic!("{stem}"));
                assert!(
                    machine.len() == 16
                        && machine
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{stem}"
                );
            } else {
                assert_eq!(stem, format!("m.gguf.{uid}"));
            }
            assert_eq!(shared_stem("m.gguf"), stem);
        }

        #[test]
        fn a_second_download_of_the_same_file_waits_for_the_first() {
            let (release, released) = std::sync::mpsc::channel::<()>();
            let released = Mutex::new(released);
            let (seen, heads_seen) = std::sync::mpsc::channel::<String>();
            let seen = Mutex::new(seen);
            let (base, server) = serve(3, move |head| {
                seen.lock().unwrap().send(head.to_string()).unwrap();
                if head.starts_with("get ") {
                    let released = released.lock().unwrap();
                    released
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                }
                whole(Some(V1), FILE, FILE.len())
            });
            let dir = scratch("waits");
            let next = || {
                heads_seen
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap()
            };
            let (first, second) = std::thread::scope(|s| {
                let first = s.spawn(|| pull(&base, &dir));
                assert!(next().starts_with("head "));
                // The first download holds the lock from before it asks for
                // the file until after it publishes it.
                assert!(next().starts_with("get "));
                let second = s.spawn(|| pull(&base, &dir));
                assert!(next().starts_with("head "));
                std::thread::sleep(std::time::Duration::from_millis(200));
                release.send(()).unwrap();
                (first.join().unwrap(), second.join().unwrap())
            });
            let path = first.unwrap();
            assert_eq!(second.unwrap(), path);
            assert_eq!(std::fs::read(&path).unwrap(), FILE);
            assert_eq!(names(&dir), ["m.gguf", "m.gguf.sha256"]);
            assert_eq!(gets(&server.join().unwrap()).len(), 1);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn failures_are_retried_only_when_another_attempt_may_get_past_them() {
            let transient = |url: &str| match call_for_stored_bytes("GET", url, &None, None) {
                Err(Failure::Transient(reason)) => reason,
                Err(other) => panic!("{url}: {other}"),
                Ok(_) => panic!("{url}: answered"),
            };
            let refused = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap()
            };
            assert_eq!(
                transient(&format!("http://{refused}/m")),
                "could not connect to the server"
            );
            assert_eq!(
                transient("http://lumen.invalid/m"),
                "could not look up the server's address"
            );

            let (base, server) = serve(5, |head| {
                let code = head.split(' ').nth(1).unwrap().trim_start_matches('/');
                status(&format!("{code} Answer"))
            });
            for code in ["408", "429", "500", "503"] {
                assert_eq!(
                    transient(&format!("{}/{code}", base.as_str())),
                    format!("the server answered {code}")
                );
            }
            match call_for_stored_bytes("GET", &format!("{}/404", base.as_str()), &None, None) {
                Err(Failure::Final(e)) => assert!(e.to_string().contains("404"), "{e}"),
                Err(other) => panic!("retried: {other}"),
                Ok(_) => panic!("answered"),
            }
            assert_eq!(server.join().unwrap().len(), 5);

            // A TLS handshake the server answers in plain text is not the
            // network's doing, so it is not retried.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            std::thread::spawn(move || {
                if let Ok((mut sock, _)) = listener.accept() {
                    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                }
            });
            match call_for_stored_bytes("GET", &format!("https://{addr}/m"), &None, None) {
                Err(Failure::Final(e)) => assert!(e.to_string().contains("tls"), "{e}"),
                Err(other) => panic!("retried: {other}"),
                Ok(_) => panic!("answered"),
            }

            // A tunnel the proxy opened and that closes in the TLS handshake.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            std::thread::spawn(move || {
                use std::io::Read;
                if let Ok((mut sock, _)) = listener.accept() {
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    while !head.ends_with(b"\r\n\r\n") && sock.read(&mut byte).is_ok_and(|n| n == 1)
                    {
                        head.push(byte[0]);
                    }
                    let _ = sock.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n");
                    let _ = sock.read(&mut [0u8; 512]);
                }
            });
            let route: Route = Some((
                ureq::Proxy::new(format!("http://{addr}")).unwrap(),
                "HTTPS_PROXY",
            ));
            match call_for_stored_bytes("GET", "https://huggingface.invalid/m.gguf", &route, None) {
                Err(Failure::Transient(reason)) => assert_eq!(
                    reason,
                    "could not connect to the server through the proxy in HTTPS_PROXY"
                ),
                Err(other) => panic!("not retried: {other}"),
                Ok(_) => panic!("answered"),
            }

            // A read-only file system fails the download; the rest of what
            // stops an existing partial being opened sends it to a file of
            // its own.
            let why = |code| unusable(&std::io::Error::from_raw_os_error(code));
            assert_eq!(why(libc::EROFS), None);
            assert_eq!(why(libc::ELOOP).unwrap(), "it is a symbolic link");
            assert_eq!(why(libc::EACCES).unwrap(), "this user may not write it");

            let words = |e: std::io::Error| network_failure(&e, &None);
            use std::io::ErrorKind;
            for kind in [ErrorKind::WouldBlock, ErrorKind::TimedOut] {
                assert_eq!(
                    words(kind.into()).unwrap(),
                    format!("no data arrived for {} s", STALL_TIMEOUT.as_secs())
                );
            }
            assert_eq!(
                words(ErrorKind::UnexpectedEof.into()).unwrap(),
                "the connection closed before the file was complete"
            );
            for kind in [
                ErrorKind::ConnectionRefused,
                ErrorKind::ConnectionReset,
                ErrorKind::ConnectionAborted,
                ErrorKind::NotConnected,
                ErrorKind::BrokenPipe,
            ] {
                assert_eq!(
                    words(kind.into()).unwrap(),
                    "the connection was lost",
                    "{kind:?}"
                );
            }
            for code in [libc::ENETUNREACH, libc::EHOSTUNREACH, libc::ENETDOWN] {
                assert_eq!(
                    words(std::io::Error::from_raw_os_error(code)).unwrap(),
                    "the connection was lost",
                    "{code}"
                );
            }
            assert_eq!(words(ErrorKind::InvalidData.into()), None);
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

    const BODY_SHA256: &str = "ed09a051f6ed65022ae62dc8c2c657896406047015a45c02cadb9ec4548b72b7";

    #[cfg(feature = "download")]
    fn pinned(size: u64, sha256: &str) -> crate::registry::CheckpointFile {
        crate::registry::CheckpointFile {
            path: "sub/m.gguf".to_string(),
            size,
            sha256: sha256.to_string(),
        }
    }

    /// A pinned file is asked for at its revision and published under its
    /// basename once the bytes received have the pinned size and hash.
    #[cfg(feature = "download")]
    #[test]
    fn pinned_download_publishes_the_file_at_its_revision() {
        let (base, server) = serve_like_hf(4, "", "", "200 OK");
        let dir = scratch_dir("pinned-ok");
        let file = pinned(BODY.len() as u64, BODY_SHA256);
        let path = super::download_file(
            &base,
            "org/repo",
            "abc123",
            &file.path,
            &dir,
            true,
            Some(&file),
        )
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), BODY);
        assert_eq!(entries(&dir), vec!["m.gguf", "m.gguf.sha256"]);
        let heads = server.join().unwrap();
        assert!(
            heads
                .iter()
                .any(|h| h.starts_with("get /org/repo/resolve/abc123/sub/m.gguf http/1.1")),
            "the file must be asked for at the pinned revision, got:\n{heads:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Bytes with another hash or another size are not the pinned file:
    /// refused, with nothing left in the directory.
    #[cfg(feature = "download")]
    #[test]
    fn pinned_download_refuses_other_bytes() {
        for (tag, file) in [
            ("pinned-hash", pinned(BODY.len() as u64, &"0".repeat(64))),
            ("pinned-size", pinned(BODY.len() as u64 + 1, BODY_SHA256)),
        ] {
            let (base, _server) = serve_like_hf(4, "", "", "200 OK");
            let dir = scratch_dir(tag);
            let err = super::download_file(
                &base,
                "org/repo",
                "abc123",
                &file.path,
                &dir,
                true,
                Some(&file),
            )
            .unwrap_err();
            assert!(
                format!("{err}").contains("not the pinned"),
                "{tag}: got {err}"
            );
            assert!(
                entries(&dir).is_empty(),
                "{tag}: nothing may be left behind, got {:?}",
                entries(&dir)
            );
            std::fs::remove_dir_all(&dir).ok();
        }
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
                "main",
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
