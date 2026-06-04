//! `lumen-server` -- HTTP front-end for the Lumen runtime.
//!
//! Provides three wire formats over the same engine:
//!
//! - OpenAI-compatible (`POST /v1/chat/completions`, `POST /v1/completions`)
//! - Anthropic-compatible (`POST /v1/messages`)
//! - Legacy completion (`POST /v1/completions`)
//!
//! ## Architecture
//!
//! ```text
//!   axum router
//!         |
//!         v
//!   per-request handler  --(JobRequest mpsc)-->  EngineWorker
//!         ^                                            |
//!         |                                            v
//!         +-----------(TokenEvent mpsc)----  Session::stream
//! ```
//!
//! The engine is owned by a single tokio task. HTTP handlers send a
//! `JobRequest` to that task and receive a `TokenEvent` stream back. The
//! worker holds the runtime types (`Session`, `Backend`, `WeightProvider`)
//! that are `!Send`-friendly under `tokio::task::spawn_blocking`, so
//! handlers stay cheap and concurrent while the compute path remains
//! sequential -- pure axum + tokio, single-worker mpsc owning the engine.
//!
//! ## Streaming safety
//!
//! [`SseSafeEmitter`] holds bytes back until they form a valid UTF-8
//! boundary AND the tool-call streaming parser has decided whether they
//! belong to a function call body or to user-visible text. The `text`
//! field of an emitter delta is therefore safe to send as a single SSE
//! `data:` field without splitting a multi-byte codepoint or leaking a
//! partial `<tool_call>` marker.
//!
//! ## What this crate does NOT do
//!
//! - It does not own a tokenizer. The runtime takes token-id slices; the
//!   tokenizer is owned by the embedder (a CLI binary, an integration
//!   test, an external process). Callers pass [`engine::Tokenize`]
//!   implementations into [`engine::EngineWorker::new`].
//! - It does not own a weight provider. The embedder constructs one and
//!   moves it into the worker.
//! - It does not enforce auth. Add a `tower::Layer` if you need bearer
//!   tokens.

pub mod engine;
pub mod error;
pub mod router;
pub mod sse;
pub mod tokenstop;
pub mod wire;

pub use engine::{
    CancellationFlag, CancellationGuard, DiskKvConfig, EngineHandle, EngineWorker, FinishReason,
    IdentityByteTokenizer, JobRequest, JobResponseChannel, ModelInfo, PooledReceiver, TokenEvent,
    Tokenize,
};
pub use error::ServerError;
pub use router::{build_router, AppState};
pub use sse::SseSafeEmitter;
pub use tokenstop::StopMatcher;
