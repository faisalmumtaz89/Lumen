//! Host models of the native prefill, shared by its GPU suites.
#![allow(dead_code)]

pub mod attn;
pub mod gdn;
#[cfg(feature = "test-prefill-dump")]
pub mod layer;
pub mod producers;
