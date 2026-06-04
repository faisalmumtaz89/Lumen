//! Post-generation evaluation gates.
//!
//! This module hosts cheap, observable correctness gates that run once
//! per validated (model, quant) configuration. Today it covers semantic
//! coherence; future gates (e.g. determinism cross-trial compare from
//! `lumen-bench`) live alongside.
//!
//! These are observation hooks — they MUST NOT touch the live
//! prefill/decode paths.

pub mod coherence;

pub use coherence::{coherence_score, CoherenceVerdict};
