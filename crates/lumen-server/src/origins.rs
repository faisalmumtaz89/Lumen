//! Requests from web pages.
//!
//! A browser adds an `Origin` header to every request a page makes other than
//! a plain GET or HEAD, and a page can neither remove nor change it (Fetch
//! Standard, "append a request `Origin` header"). Such a request is refused
//! unless the page's origin is allowed: otherwise any site the user visits
//! could have the machine generate text and images, including by DNS
//! rebinding, where the page's own name is made to point at this machine.
//! Browser extensions and apps built on a web view send one too, and need
//! theirs allowed; clients that are not web pages (curl, the SDKs, editors)
//! send none and are served as before.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{FromRequest, Request, State};
use axum::http::header;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::ServerError;

/// The origins whose requests are served: none unless listed.
/// No value allows every origin: the server sends no CORS headers, so no page
/// could read its answers, and every page could still have it run.
#[derive(Clone, Debug, Default)]
pub struct AllowedOrigins {
    listed: Vec<String>,
}

impl AllowedOrigins {
    /// From `--allow-origin` values, each one origin as the client sends it,
    /// `scheme://host[:port]`.
    pub fn from_values(values: &[String]) -> Result<Self, String> {
        let mut allowed = Self::default();
        for value in values {
            if !is_origin(value) {
                return Err(format!(
                    "--allow-origin takes one origin, such as https://app.example.com or \
                     chrome-extension://<id> (no path, no trailing slash, no wildcard); got \
                     {value:?}"
                ));
            }
            allowed.listed.push(value.clone());
        }
        Ok(allowed)
    }

    /// Browsers send the scheme and host in lowercase, so case is not compared.
    fn allows(&self, origin: &str) -> bool {
        self.listed
            .iter()
            .any(|listed| listed.eq_ignore_ascii_case(origin))
    }
}

/// `scheme://host[:port]`: a scheme, then a host with no path, query,
/// fragment or wildcard. `null`, the origin of sandboxed frames and local
/// files, is not one: any site can open such a frame.
fn is_origin(value: &str) -> bool {
    let Some((scheme, rest)) = value.split_once("://") else {
        return false;
    };
    scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
        && !rest.is_empty()
        && rest
            .bytes()
            .all(|b| b.is_ascii_graphic() && !b"/?#*".contains(&b))
}

/// Refuse a request that carries an `Origin` header `allowed` does not list,
/// before any route sees it. A value that is not visible ASCII is refused
/// like any other the list does not name.
pub async fn refuse_unlisted_origins(
    State(allowed): State<Arc<AllowedOrigins>>,
    request: Request,
    next: Next,
) -> Response {
    let refused = request
        .headers()
        .get_all(header::ORIGIN)
        .iter()
        .find(|origin| !allowed.allows(origin.to_str().unwrap_or_default()));
    let Some(origin) = refused else {
        return next.run(request).await;
    };
    let value = origin.to_str().unwrap_or_default();
    let message = if is_origin(value) {
        format!(
            "requests from web pages are refused unless their origin is allowed; to allow this \
             one, start lumen-server with --allow-origin {value}"
        )
    } else {
        format!(
            "requests from web pages are refused unless their origin is allowed, and this one's \
             origin, {:?}, cannot be",
            String::from_utf8_lossy(origin.as_bytes())
        )
    };
    // The body is read as the routes read theirs: answering while the client
    // is still sending it would reset the connection before the refusal
    // arrives.
    let _ = Bytes::from_request(request, &()).await;
    ServerError::Forbidden(message).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(list: &[&str]) -> Vec<String> {
        list.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn an_origin_is_taken_and_anything_else_is_refused() {
        for good in [
            "http://localhost:3000",
            "https://app.example.com",
            "https://app.example.com:8443",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
            "moz-extension://6f2c8f7e-3b1d-4c55-9a4e-0d1e2f3a4b5c",
            "tauri://localhost",
            "http://[::1]:8000",
        ] {
            assert!(
                AllowedOrigins::from_values(&values(&[good])).is_ok(),
                "{good}"
            );
        }
        for bad in [
            "",
            "null",
            "*",
            "http://*",
            "https://*.example.com",
            "localhost:3000",
            "://host",
            "3http://host",
            "http://",
            "http://host/",
            "http://host/path",
            "http://host?x",
            "http://host#x",
            "http://ho st",
        ] {
            let err = AllowedOrigins::from_values(&values(&[bad])).unwrap_err();
            assert!(
                err.starts_with("--allow-origin takes one origin"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn only_the_listed_origins_are_allowed_whatever_their_case() {
        let allowed =
            AllowedOrigins::from_values(&values(&["https://App.Example.com", "http://x:1"]))
                .unwrap();
        assert!(allowed.allows("https://app.example.com"));
        assert!(allowed.allows("http://x:1"));
        assert!(!allowed.allows("https://app.example.com:443"));
        assert!(!allowed.allows("http://x:2"));
        assert!(!allowed.allows(""));
        assert!(!AllowedOrigins::default().allows("http://localhost:3000"));
    }
}
