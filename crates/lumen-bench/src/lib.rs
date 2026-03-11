//! Benchmark harness for Lumen inference validation.
//!
//! Provides configurable benchmark suites, a runner that handles model
//! generation and page cache control, and both tabular and JSON output.

pub mod config;
pub mod output;
pub mod results;
pub mod runner;
pub mod suite;
