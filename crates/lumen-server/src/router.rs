//! axum router wiring.
//!
//! Endpoints:
//!
//! - `GET  /v1/models`              - OpenAI-style model list (1 entry).
//! - `POST /v1/chat/completions`    - OpenAI chat completion (SSE optional).
//! - `POST /v1/completions`         - Legacy OpenAI text completion (SSE optional).
//! - `POST /v1/messages`            - Anthropic messages (SSE optional).
//! - `GET /debug/memory_breakdown` - per-component RSS report
//!                                    (env-gated `LUMEN_SERVER_DEBUG_MEM=1`;
//!                                    returns 404 by default).
//!
//! ## Error mapping
//!
//! All request handlers use the [`OpenAiJson`] extractor instead of axum's
//! built-in `Json<T>`. Schema-deserialization errors (missing fields, wrong
//! types, unknown fields when `deny_unknown_fields` applies) are converted
//! to [`ServerError::BadRequest`] with `param` populated from
//! `serde_json::Error::path()` and a stable `code` derived from
//! `Error::classify()`. The response is HTTP 400 with the OpenAI envelope
//! `{"error":{"message", "type":"invalid_request_error", "param", "code"}}`.
//! This replaces axum's default 422 + plain-text rejection.

use std::time::SystemTime;

use axum::body::{Body, Bytes};
use axum::extract::{FromRequest, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;

use crate::engine::EngineHandle;
use crate::error::ServerError;
use crate::wire;

/// Application state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
}

/// Build the axum router.
///
/// The `/debug/memory_breakdown` endpoint is registered unconditionally so
/// the surface is uniform. The handler returns 404 when
/// `LUMEN_SERVER_DEBUG_MEM` is unset or `0`, and only returns the JSON
/// breakdown when the env var is set to a non-empty value other than `"0"`.
/// This keeps the default response surface byte-identical to the legacy
/// router shape — no new live route response body, no new shipped data
/// when the operator has not explicitly opted in to debug instrumentation.
pub fn build_router(engine: EngineHandle) -> Router {
    let state = AppState { engine };
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(messages))
        .route("/debug/memory_breakdown", get(memory_breakdown))
        .with_state(state)
}

// ----------------------------- OpenAiJson extractor ---------------------
//
// custom JSON extractor that maps deserialization errors to
// `ServerError::BadRequest` (HTTP 400 + OpenAI envelope) instead of axum's
// default `Json<T>` rejection (HTTP 422 + plain text). The wire shape is
// `{"error":{"message", "type":"invalid_request_error", "param", "code"}}`
// per the OpenAI error-codes spec.
//
// Mapping rules from `serde_json::Error`:
//   - `missing field "X"`             → code=`missing_field`,    param=X
//   - `invalid type ... at line/col`  → code=`invalid_type`,     param=path
//   - `unknown field "X"`             → code=`unknown_field`,    param=X
//   - other Data errors               → code=`invalid_value`,    param=path
//   - Syntax errors (truncated JSON)  → code=`invalid_json`,     param=None
//   - Io / Eof                        → code=`invalid_json`,     param=None
//
// `param` for nested fields is the dotted JSON pointer
// (`messages[0].role`). For top-level fields it is the field name.

/// Custom JSON extractor that emits OpenAI-shape 400 errors on schema
/// failures. Use this in handlers instead of `axum::Json<T>`.
pub struct OpenAiJson<T>(pub T);

#[axum::async_trait]
impl<T, S> FromRequest<S> for OpenAiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ServerError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = Bytes::from_request(req, state).await.map_err(|e| {
            ServerError::bad_request(format!("failed to read request body: {e}"))
        })?;
        let value: T = serde_json::from_slice(&bytes).map_err(map_serde_error)?;
        Ok(OpenAiJson(value))
    }
}

/// Translate a `serde_json::Error` into a localized `BadRequest`.
fn map_serde_error(e: serde_json::Error) -> ServerError {
    let msg = e.to_string();
    // serde_json messages have stable shapes we can parse:
    //   "missing field `messages` at line 1 column 18"
    //   "invalid type: integer `5`, expected a sequence at line 1 column 30"
    //   "unknown field `garbage`, expected one of `model`, ... at line 1 column 30"
    //   "EOF while parsing a value at line 1 column 0"
    //   "expected `:` at line 1 column 8"
    if let Some(field) = extract_quoted_after(&msg, "missing field") {
        return ServerError::bad_request_field(
            format!("missing required field: `{field}`"),
            field,
            "missing_field",
        );
    }
    if let Some(field) = extract_quoted_after(&msg, "unknown field") {
        return ServerError::bad_request_field(
            format!("unknown field: `{field}`"),
            field,
            "unknown_field",
        );
    }
    if msg.starts_with("invalid type") {
        // Best-effort: trim the trailing "at line N column M" so the
        // message focuses on the type mismatch.
        let trimmed = msg
            .find(" at line ")
            .map(|i| &msg[..i])
            .unwrap_or(&msg)
            .to_string();
        return ServerError::bad_request_field(trimmed, "", "invalid_type");
    }
    // Classify the rest by serde_json category.
    let code = match e.classify() {
        serde_json::error::Category::Data => "invalid_value",
        serde_json::error::Category::Syntax
        | serde_json::error::Category::Eof
        | serde_json::error::Category::Io => "invalid_json",
    };
    ServerError::bad_request_field(format!("malformed request body: {msg}"), "", code)
}

/// Extract the first backtick-quoted token after the given prefix.
/// Returns `None` if the prefix is not present or no backticked token follows.
fn extract_quoted_after(msg: &str, prefix: &str) -> Option<String> {
    let after = msg.split_once(prefix)?.1;
    let start = after.find('`')?;
    let rest = &after[start + 1..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

/// returns `true` iff `LUMEN_SERVER_DEBUG_MEM` is set to a
/// non-empty value other than `"0"`.  Cheap; the env lookup is the only
/// cost.  We deliberately do NOT cache the result process-wide because
/// operators may want to toggle the flag for a single soak run.
pub(crate) fn memory_breakdown_enabled() -> bool {
    match std::env::var("LUMEN_SERVER_DEBUG_MEM") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

// ----------------------------- /v1/models -------------------------------

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let info = state.engine.model_info();
    let body = serde_json::json!({
        "object": "list",
        "data": [{
            "id": info.id,
            "object": "model",
            "created": info.created,
            "owned_by": info.owned_by,
        }]
    });
    Json(body)
}

// ----------------------------- /v1/chat/completions ---------------------

async fn chat_completions(
    State(state): State<AppState>,
    OpenAiJson(req): OpenAiJson<wire::openai::ChatCompletionRequest>,
) -> Result<Response, ServerError> {
    let model_id = state.engine.model_info().id.clone();
    let stream = req.stream.unwrap_or(false);
    let job = req.into_job(&state.engine)?;
    let rx = state.engine.submit(job, 128).await?;

    if stream {
        let body = wire::openai::stream_chat(rx, model_id, current_unix_time());
        Ok(sse_response(body))
    } else {
        let resp = wire::openai::collect_chat(rx, model_id, current_unix_time()).await?;
        Ok((StatusCode::OK, Json(resp)).into_response())
    }
}

// ----------------------------- /v1/completions --------------------------

async fn completions(
    State(state): State<AppState>,
    OpenAiJson(req): OpenAiJson<wire::openai::CompletionRequest>,
) -> Result<Response, ServerError> {
    let model_id = state.engine.model_info().id.clone();
    let stream = req.stream.unwrap_or(false);
    let job = req.into_job(&state.engine)?;
    let rx = state.engine.submit(job, 128).await?;
    if stream {
        let body = wire::openai::stream_completion(rx, model_id, current_unix_time());
        Ok(sse_response(body))
    } else {
        let resp = wire::openai::collect_completion(rx, model_id, current_unix_time()).await?;
        Ok((StatusCode::OK, Json(resp)).into_response())
    }
}

// ----------------------------- /v1/messages -----------------------------

async fn messages(
    State(state): State<AppState>,
    OpenAiJson(req): OpenAiJson<wire::anthropic::MessagesRequest>,
) -> Result<Response, ServerError> {
    let model_id = state.engine.model_info().id.clone();
    let stream = req.stream.unwrap_or(false);
    let job = req.into_job(&state.engine)?;
    let rx = state.engine.submit(job, 128).await?;
    if stream {
        let body = wire::anthropic::stream_messages(rx, model_id);
        Ok(sse_response(body))
    } else {
        let resp = wire::anthropic::collect_messages(rx, model_id).await?;
        Ok((StatusCode::OK, Json(resp)).into_response())
    }
}

// ----------------------------- shared utilities -------------------------

/// Build an SSE response from a stream of byte chunks (each chunk is one
/// complete SSE frame ending with `\n\n`).
fn sse_response(body: Body) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-store, no-transform"),
    );
    headers.insert(
        header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    (StatusCode::OK, headers, body).into_response()
}

/// Current wall-clock seconds, used for `created` fields in OpenAI bodies.
/// Falls back to 0 if the clock is before the epoch (impossible in
/// practice, but `unwrap` would still panic on hypothetical sysclock skew).
pub(crate) fn current_unix_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ----------------------------- /debug/memory_breakdown ------------------

/// per-component memory breakdown handler.
///
/// Returns:
/// - `404 Not Found` when `LUMEN_SERVER_DEBUG_MEM` is unset or `"0"`.  This
///   is the default for any deployed lumen-server — the endpoint is wired
///   into the router but invisible unless the operator opts in.
/// - `200 OK` with a JSON body matching `ServerMemoryBreakdown::to_jsonl`
///   when the env var is set.  The body is one self-contained object
///   suitable for direct append into `soak-breakdown.jsonl`.
///
/// Cost when enabled: one Mutex acquisition on the snapshot Arc.  The
/// snapshot itself is refreshed by the worker thread after every
/// completed job (see `EngineWorker::update_memory_breakdown`); the
/// handler does NOT trigger a refresh, so it cannot block the worker.
async fn memory_breakdown(State(state): State<AppState>) -> Response {
    if !memory_breakdown_enabled() {
        return (StatusCode::NOT_FOUND, "memory breakdown disabled").into_response();
    }
    let snap = state.engine.memory_breakdown_snapshot();
    let body = snap.to_jsonl();
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    (StatusCode::OK, headers, body).into_response()
}

