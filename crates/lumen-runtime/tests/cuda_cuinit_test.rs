//! CUDA cuInit integration tests.
//!
//! Validates that the CUDA backend correctly initializes the driver API
//! before querying devices. These tests require an NVIDIA GPU with CUDA
//! drivers installed; they are gated behind `#[cfg(feature = "cuda")]` and
//! skipped gracefully on machines without CUDA hardware.
//!
//! # Running
//!
//! ```sh
//! cargo test --release -p lumen-runtime --features cuda --test cuda_cuinit_test
//! ```

#![cfg(feature = "cuda")]

use lumen_runtime::cuda::ffi;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::error::RuntimeError;

/// Verify `device_count()` calls `cuInit(0)` before `cuDeviceGetCount`.
///
/// Before the fix, this would fail with `CUDA_ERROR_NOT_INITIALIZED` on
/// fresh containers (e.g. Modal) where no prior CUDA call had been made.
/// After the fix, `device_count()` calls `cuInit(0)` first (idempotent).
///
/// On machines without CUDA hardware, `cuInit` succeeds but returns 0
/// devices -- that is a valid outcome, not an error.
#[test]
fn device_count_succeeds_after_cuinit() {
    match ffi::device_count() {
        Ok(count) => {
            // cuInit + cuDeviceGetCount succeeded. On GPU machines count > 0;
            // on CPU-only machines count == 0 (both are valid).
            eprintln!("[cuda_cuinit_test] device_count = {count}");
        }
        Err(RuntimeError::Compute(msg)) if msg.contains("not found") => {
            // No CUDA driver installed (libcuda.so / libcuda.dylib missing).
            // This is expected on macOS dev machines. Skip gracefully.
            eprintln!("[cuda_cuinit_test] No CUDA driver: {msg} -- skipping");
        }
        Err(e) => {
            // Unexpected error -- the cuInit fix should prevent
            // CUDA_ERROR_NOT_INITIALIZED. Fail the test.
            panic!("device_count() failed unexpectedly: {e}");
        }
    }
}

/// Verify `device_count()` is idempotent -- calling it multiple times
/// must not fail even though `cuInit(0)` is called each time.
#[test]
fn device_count_idempotent() {
    // First call initializes the driver.
    let first = ffi::device_count();
    // Second call must produce the same result (cuInit is idempotent).
    let second = ffi::device_count();

    match (&first, &second) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "device_count must be stable across calls");
        }
        (Err(_), Err(_)) => {
            // Both failed (no driver) -- consistent behavior, acceptable.
        }
        _ => {
            panic!(
                "device_count() was inconsistent: first={first:?}, second={second:?}"
            );
        }
    }
}

/// Verify `CudaBackend::new(0)` succeeds on GPU machines.
///
/// This exercises the full initialization path: cuInit -> device count check
/// -> CudaContext creation -> stream creation -> backend construction.
///
/// On CPU-only machines (no CUDA driver or 0 devices), the test skips
/// gracefully.
#[test]
fn cuda_backend_new_succeeds() {
    // Check if we have a GPU first.
    let count = match ffi::device_count() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[cuda_cuinit_test] No CUDA driver -- skipping backend test");
            return;
        }
    };
    if count == 0 {
        eprintln!("[cuda_cuinit_test] 0 CUDA devices -- skipping backend test");
        return;
    }

    // Must succeed: cuInit was already called by device_count, and
    // CudaContext::new also calls it internally.
    let backend = CudaBackend::new(0);
    assert!(
        backend.is_ok(),
        "CudaBackend::new(0) failed: {:?}",
        backend.err()
    );
}

/// Verify that requesting a non-existent device produces a clear error.
#[test]
fn cuda_backend_invalid_device_id() {
    let count = match ffi::device_count() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[cuda_cuinit_test] No CUDA driver -- skipping");
            return;
        }
    };
    if count == 0 {
        eprintln!("[cuda_cuinit_test] 0 CUDA devices -- skipping");
        return;
    }

    // Request device ordinal beyond available count.
    let result = CudaBackend::new(count + 100);
    assert!(
        result.is_err(),
        "CudaBackend::new({}) should fail with only {} device(s)",
        count + 100,
        count
    );
}
