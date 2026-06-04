//! Server error type.
//!
//! Maps internal failures to HTTP responses with structured JSON bodies that
//! match the OpenAI / Anthropic error shape clients expect to see.
//!
//! the OpenAI envelope is now the full four-field shape
//! `{"error":{"message", "type", "param", "code"}}` per the OpenAI spec
//! (https://platform.openai.com/docs/guides/error-codes#api-errors).
//! `param` and `code` are emitted as JSON `null` when unknown, never
//! omitted, so the wire shape is byte-stable across both schema-level
//! (extractor) and semantic-level (wire layer) failures.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use thiserror::Error;

/// All errors a request handler can surface.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Malformed input: missing fields, wrong types, schema mismatches.
    /// `param` is the JSON pointer to the offending field (e.g. `"messages"`,
    /// `"messages[0].role"`), `None` when not localizable.
    /// `code` is a stable machine-readable tag (e.g. `"missing_field"`,
    /// `"invalid_type"`, `"unknown_field"`), `None` when generic.
    #[error("{message}")]
    BadRequest {
        message: String,
        param: Option<String>,
        code: Option<String>,
    },

    /// Model not found in the registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Underlying runtime error while running inference.
    #[error("runtime error: {0}")]
    Runtime(String),

    /// The engine worker has shut down or never started.
    #[error("engine unavailable: {0}")]
    EngineUnavailable(String),

    /// Internal server bug, panic, or unrecoverable failure.
    #[error("internal error: {0}")]
    Internal(String),
}

impl ServerError {
    /// Construct a generic `BadRequest` with no `param` / `code` (legacy
    /// callers + semantic wire-layer errors that don't have field info).
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest {
            message: msg.into(),
            param: None,
            code: None,
        }
    }

    /// Construct a localized `BadRequest` (extractor path: knows the field
    /// name + a stable code tag).
    pub fn bad_request_field(
        msg: impl Into<String>,
        param: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self::BadRequest {
            message: msg.into(),
            param: Some(param.into()),
            code: Some(code.into()),
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound(_) => StatusCode::NOT_FOUND,
            Self::EngineUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Runtime(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::BadRequest { .. } => "invalid_request_error",
            Self::ModelNotFound(_) => "not_found_error",
            Self::Runtime(_) => "api_error",
            Self::EngineUnavailable(_) => "overloaded_error",
            Self::Internal(_) => "api_error",
        }
    }

    /// `param` field of the OpenAI envelope (JSON `null` when unknown).
    fn envelope_param(&self) -> Value {
        match self {
            Self::BadRequest { param: Some(p), .. } => Value::String(p.clone()),
            _ => Value::Null,
        }
    }

    /// `code` field of the OpenAI envelope (JSON `null` when unknown).
    fn envelope_code(&self) -> Value {
        match self {
            Self::BadRequest { code: Some(c), .. } => Value::String(c.clone()),
            _ => Value::Null,
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = json!({
            "error": {
                "message": self.to_string(),
                "type": self.error_type(),
                "param": self.envelope_param(),
                "code": self.envelope_code(),
            }
        });
        (status, Json(body)).into_response()
    }
}

impl From<lumen_runtime::RuntimeError> for ServerError {
    fn from(e: lumen_runtime::RuntimeError) -> Self {
        Self::Runtime(e.to_string())
    }
}
