//! Wire-format encoders.
//!
//! Each submodule owns the request DTO, the SSE state machine, and the
//! non-streaming response shape for one external API.

pub mod anthropic;
pub mod openai;
