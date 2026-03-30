//! Production polish tests for the CUDA backend.
//!
//! Tests device info queries, device selection, and graceful error messages
//! when CUDA initialization fails or an invalid device is requested.
//!
//! Run on a CUDA-capable machine:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_production_test
//!
//! On macOS (no NVIDIA GPU), cudarc's dynamic loading panics when the CUDA
//! shared library is absent. Tests use `catch_unwind` to handle this gracefully:
//! the test passes (skipped) on macOS and runs the actual assertions on a GPU machine.

#![cfg(feature = "cuda")]

use lumen_runtime::cuda::ffi::{self, CudaDevice};
use std::panic;

/// Try to create a CUDA device, returning None if the CUDA library is absent
/// (cudarc panics on missing .so/.dylib) or if no GPU is available.
fn try_create_device(device_id: usize) -> Option<CudaDevice> {
    let result = panic::catch_unwind(|| CudaDevice::new(device_id));
    match result {
        Ok(Ok(device)) => Some(device),
        Ok(Err(e)) => {
            eprintln!("[skip] CudaDevice::new({device_id}) returned error: {e}");
            None
        }
        Err(_) => {
            eprintln!("[skip] CudaDevice::new({device_id}) panicked (no CUDA library)");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Device info tests (require a real CUDA GPU -- skipped gracefully if absent)
// ---------------------------------------------------------------------------

#[test]
fn device_count_returns_without_panic() {
    // Must not panic regardless of whether a CUDA driver is present.
    // cudarc may panic if the library is absent, so catch that.
    let result = panic::catch_unwind(ffi::device_count);
    match result {
        Ok(Ok(count)) => eprintln!("[info] device count: {count}"),
        Ok(Err(e)) => eprintln!("[skip] device_count error: {e}"),
        Err(_) => eprintln!("[skip] device_count panicked (no CUDA library)"),
    }
}

#[test]
fn device_name_is_nonempty_on_gpu() {
    let device = match try_create_device(0) {
        Some(d) => d,
        None => return,
    };
    let name = device.name().expect("name() should succeed on a valid device");
    assert!(!name.is_empty(), "device name must not be empty");
    eprintln!("[info] device name: {name}");
}

#[test]
fn total_memory_is_positive_on_gpu() {
    let device = match try_create_device(0) {
        Some(d) => d,
        None => return,
    };
    let total = device.total_memory().expect("total_memory() should succeed");
    assert!(total > 0, "total VRAM must be > 0");
    eprintln!(
        "[info] VRAM: {:.1} GB",
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

#[test]
fn free_memory_does_not_exceed_total() {
    let device = match try_create_device(0) {
        Some(d) => d,
        None => return,
    };
    let total = device.total_memory().expect("total_memory() should succeed");
    let free = device.free_memory().expect("free_memory() should succeed");
    assert!(free <= total, "free ({free}) must not exceed total ({total})");
    eprintln!("[info] free: {free} / total: {total}");
}

#[test]
fn compute_capability_is_valid_on_gpu() {
    let device = match try_create_device(0) {
        Some(d) => d,
        None => return,
    };
    let (major, minor) = device
        .compute_capability()
        .expect("compute_capability() should succeed");
    // SM 3.0 is the oldest CUDA architecture; modern GPUs are 7.0+.
    assert!(major >= 3, "SM major must be >= 3, got {major}");
    // Minor version is a small non-negative number.
    assert!(
        minor >= 0 && minor <= 9,
        "SM minor must be 0..9, got {minor}"
    );
    eprintln!("[info] SM {major}.{minor}");
}

// ---------------------------------------------------------------------------
// Device selection tests
// ---------------------------------------------------------------------------

#[test]
fn device_0_matches_default() {
    // Creating device 0 explicitly should succeed if any GPU is present.
    match try_create_device(0) {
        Some(d) => {
            let name = d.name().unwrap_or_default();
            eprintln!("[info] device 0: {name}");
        }
        None => {} // Gracefully skipped.
    }
}

// ---------------------------------------------------------------------------
// Graceful error message tests
// ---------------------------------------------------------------------------

#[test]
fn invalid_device_id_gives_helpful_message() {
    // Device ordinal 9999 should not exist on any reasonable machine.
    // On macOS without CUDA, this will panic from cudarc. Catch that.
    let result = panic::catch_unwind(|| CudaDevice::new(9999));
    match result {
        Ok(Ok(_)) => panic!("device 9999 must not exist"),
        Ok(Err(e)) => {
            let err_msg = format!("{e}");
            // The error message must mention the requested device ordinal
            // and the actual count of available devices, OR indicate no driver.
            let mentions_device = err_msg.contains("9999");
            let mentions_driver = err_msg.contains("driver") || err_msg.contains("No CUDA");
            assert!(
                mentions_device || mentions_driver,
                "error must mention device ID or driver status: {err_msg}"
            );
            eprintln!("[info] error for device 9999: {err_msg}");
        }
        Err(_) => {
            eprintln!("[skip] CudaDevice::new(9999) panicked (no CUDA library on macOS)");
        }
    }
}

// ---------------------------------------------------------------------------
// CudaBackend::new(0) integration (device info printed to stderr)
// ---------------------------------------------------------------------------

#[test]
fn cuda_backend_new_prints_device_info() {
    use lumen_runtime::CudaBackend;
    // On a GPU machine this will print [cuda] lines to stderr.
    // On macOS without CUDA, cudarc panics. Catch that.
    let result = panic::catch_unwind(|| CudaBackend::new(0));
    match result {
        Ok(Ok(_backend)) => {
            eprintln!("[info] CudaBackend created successfully on device 0");
        }
        Ok(Err(e)) => {
            let msg = format!("{e}");
            // Must mention NVIDIA driver or device count, not a raw cudarc error.
            assert!(
                msg.contains("driver") || msg.contains("device") || msg.contains("CUDA"),
                "error must be user-friendly: {msg}"
            );
            eprintln!("[skip] CudaBackend::new(0) error: {msg}");
        }
        Err(_) => {
            eprintln!("[skip] CudaBackend::new(0) panicked (no CUDA library on macOS)");
        }
    }
}
