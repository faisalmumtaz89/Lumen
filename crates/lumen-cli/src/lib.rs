//! `lumen_cli` library facade — re-exports the `tokenize`, `registry` and
//! `cache` modules so integration tests and downstream crates (lumen-server,
//! its soak harness, benchmark drivers, etc.) can reuse the project-canonical
//! `BpeTokenizer`, model registry and model cache without forking them.
//!
//! The CLI binary (`lumen`, declared at `[[bin]]` in `Cargo.toml`) keeps its
//! command surface private; only the modules listed here are re-exported.
//!
//! This file exists for the single purpose of single-source-of-truth
//! tokenizer, registry and cache reuse — see the no-parallel-implementations convention (no parallel implementations).
//!
//! When the CLI binary is built, both this library AND the binary are
//! produced; the binary continues to declare its own `mod` items in
//! `main.rs`. The two compilation units coexist without conflict.

pub mod cache;
pub mod registry;
pub mod tokenize;
