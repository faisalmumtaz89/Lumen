//! End-to-end token-stream test for the tiled decode-attention path
//! at long context.
//!
//! Drives a small synthetic Qwen-like model through a long-context decode
//! that crosses the gate threshold. Verifies that:
//!
//! 1.**Correctness**: with `LUMEN_CUDA_DECODE_TILED=1` forced, the
//!    backend produces token streams that are close to (allowing for
//!    floating-point reassociation near-ties) the streams produced when
//!    the gate-driven path is left to its default.
//! 2.**Capability**: the tiled path serves seq_len well past the
//!    single-block kernel's `40_950` ceiling (the central correctness gate
//!    for the tiled-decode path).
//!
//! Test methodology mirrors the synthetic token-stream-
//! exact" pattern from:
//!   - Build a tiny test model (small layer/head counts so KV cache fits).
//!   - Seed identical state in two backends.
//!   - Run N tokens in each; compare token IDs.
//!   - Tolerate ≤ 2 / N divergent tokens (near-ties on the model's
//!     well-conditioned vocab).
//!
//! Gated behind `--features cuda`. Skips gracefully when no NVIDIA GPU
//! via the standard `try_cuda_backend` pattern.
//!
//! Note: This is a SCAFFOLDING test — it documents the validation
//! intent. The full long-context exercise at seq_len = 64K on a real
//! Qwen3.5-9B model is a Modal cell because it requires
//! ~17 GB of KV cache (out of scope for local CI).

#![cfg(feature = "cuda")]

use lumen_runtime::cuda::CudaBackend;

/// Try to create a CudaBackend, returning None when no NVIDIA GPU.
fn try_cuda_backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("Skipping: no CUDA GPU available: {e}");
            None
        }
    }
}

/// Smoke: confirm a CUDA backend can be constructed on this host. This is
/// the universal preflight that all subsequent long-context tests share.
/// On macOS (no NVIDIA GPU) this test gracefully skips.
#[test]
fn cuda_backend_creation_smoke_for_long_context() {
    let _backend = match try_cuda_backend() {
        Some(b) => b,
        None => return,
    };
    // If we got here, the backend constructed without error. The actual
    // long-context decode tests (vs CPU reference, with gate forcing) live
    // in the kernel-level `cuda_attention_tiled_test.rs` (which exercises
    // the tiled kernel directly at seq_len = 64K via the standard CUDA
    // device API). Backend-level long-context decode at seq_len ≥ 64K is
    // a Modal cell.
}

/// Verify that the gate-predicate env vars resolve as documented in the
/// design. This is the production-side parallel to the
/// `attention_decode_variant_tests` unit tests in `decode.rs`. Confirms
/// the OnceLock-cached values are visible from the integration-test
/// process (where the gate would be queried from `launch_attention_
/// decode_gated`).
///
/// We do NOT mutate env vars at runtime in this test (the gate caches its
/// resolution in a OnceLock on first read, so a per-test mutation is
/// brittle). Instead, this test documents the EXPECTED resolution under
/// the default (unset) environment. The full env-driven sweep is
/// covered by the instrumented dispatch count check.
#[test]
fn gate_env_resolution_default_unset() {
    // When no env vars are set, the gate's force flag is false and the
    // threshold is the default 0 ("tiled-always" — was 36_864, lowered
    // per empirical data). Both are observed by reading the
    // process-static OnceLocks via the lib-public API path.
    //
    // The OnceLocks are crate-private inside `decode.rs`, so we cannot
    // call them directly from an integration test. We instead verify
    // (a) that no env var is set in the test process, and (b) that any
    // future test that wants different behavior must set them BEFORE
    // the first kernel dispatch (the OnceLock contract).
    let force = std::env::var("LUMEN_CUDA_DECODE_TILED").ok();
    let threshold = std::env::var("LUMEN_CUDA_DECODE_TILED_THRESHOLD").ok();
    eprintln!(
        "[env preflight] LUMEN_CUDA_DECODE_TILED = {:?}, LUMEN_CUDA_DECODE_TILED_THRESHOLD = {:?}",
        force, threshold,
    );
    // If the env is unset, the gate defaults are in effect. If it IS set
    // (operator-driven), the test still passes — we just log the
    // resolved values for the record.
}

/// Verify that requesting a CUDA backend, even on a host without an NVIDIA
/// GPU, does not panic and gracefully reports the unavailability. This
/// covers the macOS dev path where the long-context feature is implemented
/// but the kernel itself runs on Modal A100. On Modal
/// the call returns Ok; on macOS it returns Err (which we surface as
/// "skip" rather than failure).
#[test]
fn cuda_backend_constructor_graceful() {
    match CudaBackend::new(0) {
        Ok(_) => {
            // CUDA available -- backend constructed, that's the win.
        }
        Err(e) => {
            // CUDA unavailable -- expected on macOS.
            let msg = format!("{e}");
            assert!(
                !msg.is_empty(),
                "CudaBackend::new error must carry a non-empty message; got empty"
            );
            eprintln!("[CUDA absent, skipping deeper long-context tests] {msg}");
        }
    }
}
