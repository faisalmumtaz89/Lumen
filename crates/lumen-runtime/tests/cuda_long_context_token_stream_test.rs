//! Backend-construction preflights for long-context decode on CUDA.
//!
//! Decode attention runs one kernel at every context the cache holds, with
//! fixed scratch and no route switch; its correctness at long context is
//! held by the kernel-level suites (`cuda_attention_decode_test.rs` to
//! 32,768 keys, the fixture and shape suites) and by the end-to-end
//! token-stream tests on real models. What this file holds is the two
//! preflights those suites assume: the backend constructs where a GPU is
//! present, and reports its absence without panicking where one is not.
//!
//! Gated behind `--features cuda`. Skips gracefully when no NVIDIA GPU
//! via the standard `try_cuda_backend` pattern.

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
    // The backend constructed without error. The long-context decode tests
    // themselves are the kernel-level suites named in the file header and
    // the end-to-end token-stream tests on real models.
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
