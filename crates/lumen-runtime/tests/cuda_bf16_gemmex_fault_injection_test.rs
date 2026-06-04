//! End-to-end fault-injection tests for the BF16 GemmEx wrapper
//!
//! These tests close the only acknowledged-gap 's audit: the
//! `arm_runtime_fallback_sets_flag_and_is_idempotent` unit test verifies
//! the *flag-arming* mechanism in isolation, and the
//! `cuda_bf16_gemmex_fallback_test` integration suite verifies the
//! *probe-on-construction* mechanism + `LUMEN_CUDA_BF16_GEMMEX=0`
//! env-opt-out path;**this**suite drives the wrapper's
//! `Bf16LaunchOutcome::CublasFailure -> arm_bf16_gemmex_runtime_fallback
//! -> launch_legacy_matvec_bf16` fall-through arm under a real BF16
//! matvec dispatch on A100 hardware.
//!
//! Mechanism: the cfg-gated `inject_next_bf16_cublas_failure()` helper
//! (lives in `crates/lumen-runtime/src/cuda/backend_impl.rs` behind
//! `#[cfg(any(test, feature = "test-fault-injection"))]`) flips a
//! one-shot atomic. On the next call into `launch_hgemv_bf16` (or its
//! residual sibling), the inject check at the top of that function
//! atomically swaps the flag back to false and returns
//! `Ok(Bf16LaunchOutcome::CublasFailure(CUBLAS_STATUS_NOT_INITIALIZED))`
//! *without* dispatching cuBLAS. The wrapper at
//! `launch_bf16_matvec_with_fallback` then arms the process-wide
//! runtime-fallback flag (`BF16_GEMMEX_FALLBACK_ARMED` -> true), emits
//! the once-only `BF16_GEMMEX_RUNTIME_WARNING` eprintln, and
//! re-dispatches the matvec via `launch_legacy_matvec_bf16` -- exactly
//! the path that runs on a real cuBLAS runtime failure.
//!
//! The production wrappers and call sites at
//! `crates/lumen-runtime/src/cuda/backend_impl.rs:5559,6002,6571`
//! remain byte-identical regardless of whether the feature is enabled:
//! the inject check is itself `#[cfg]`-gated inside the `unsafe fn
//! launch_hgemv_bf16` body, so release builds without the feature
//! compile that branch away in its entirety.
//!
//! On macOS dev hosts (no NVIDIA driver) every test bails out cleanly
//! via `try_cuda_backend`; on Modal A100 they all execute and the
//! end-to-end wrapper path is exercised under a real BF16 matvec.
//!
//! Tests serialize through a process-wide mutex because they manipulate
//! the same process-static atomics (`BF16_GEMMEX_AVAILABLE`,
//! `BF16_GEMMEX_FALLBACK_ARMED`, `BF16_INJECT_NEXT_CUBLAS_FAILURE`,
//! `BF16_GEMMEX_RUNTIME_WARNING`). The Modal validation harness also
//! runs with `--test-threads=1` as belt-and-suspenders.
//!
//! Run on Modal A100:
//!     cargo test --release -p lumen-runtime \
//!         --features "cuda test-fault-injection" \
//!         --test cuda_bf16_gemmex_fault_injection_test \
//!         -- --test-threads=1 --nocapture
//!
//! Run locally on macOS (skips cleanly):
//!     cargo test -p lumen-runtime --features cuda \
//!         --test cuda_bf16_gemmex_fault_injection_test

#![cfg(feature = "cuda")]
// The four tests in this file depend on the cfg-gated injection helper.
// Without the `test-fault-injection` feature the helper does not exist
// at all (it is `#[cfg(any(test, feature = "test-fault-injection"))]`
// in `backend_impl.rs`), so the file is compiled as an empty test
// binary in that configuration. We declare the dependency at module
// level so `cargo test --features cuda` (without the inject feature)
// produces a runnable but no-op test binary, while `cargo test
// --features "cuda test-fault-injection"` exposes the four real tests.
#![cfg(feature = "test-fault-injection")]

use lumen_format::ModelHyperparams;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::{
    bf16_gemmex_fallback_armed_for_tests,
    bf16_gemmex_runtime_warning_emitted_for_tests,
    inject_next_bf16_cublas_failure,
    reset_bf16_gemmex_state_for_tests,
    CudaBackend,
};
use std::sync::{Mutex, OnceLock};

/// Returns the process-wide serialization mutex. Required because all
/// four tests in this file mutate the same process-static BF16 GemmEx
/// atomics; running concurrently would interleave their state changes
/// and produce false negatives.
fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Try constructing a CUDA backend on device 0. Returns `None` on hosts
/// without a NVIDIA driver (macOS dev boxes, CI runners without GPUs).
/// cudarc 0.19's fallback dynamic loader panics rather than returning
/// `Err` when `libcuda.{dylib,so}` cannot be loaded; wrap in
/// `catch_unwind` so these tests skip cleanly on dev hosts. Mirrors the
/// pattern used by `cuda_bf16_gemmex_fallback_test.rs`.
fn try_cuda_backend() -> Option<CudaBackend> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CudaBackend::new(0)
    })) {
        Ok(Ok(b)) => Some(b),
        Ok(Err(_)) | Err(_) => None,
    }
}

/// Build a minimal `ModelHyperparams` for a tiny test model that
/// exercises the BF16 wrapper without requiring a multi-GB weight
/// payload. The values here are arbitrary -- they only need to be
/// internally consistent (head_dim * num_heads = hidden_dim etc.) so
/// that `CudaBackend::init` succeeds and compiles the full kernel set.
/// The actual BF16 matvec dispatched by the test does NOT use these
/// hyperparams; it uses the (in_dim, out_dim) the test itself provides.
fn minimal_test_hyperparams() -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        hidden_dim: 128,
        intermediate_dim: 256,
        vocab_size: 1024,
        max_seq_len: 512,
        rope_params: None,
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-6,
        rotary_dim: None,
        rope_neox: false,
    }
}

/// Convert an f32 slice to BF16 row-major bytes via truncation
/// (round-to-zero, the same rounding the `f32_to_bf16` kernel uses).
/// Used by tests to build small synthetic weight matrices.
fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        // BF16 = top 16 bits of the IEEE 754 f32 representation.
        let bits = v.to_bits();
        let bf16 = (bits >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// Construct a backend + init it. Returns None if no CUDA device is
/// available (test skips cleanly on macOS). Resets the BF16 GemmEx
/// state machine so the test starts from a known baseline regardless
/// of which other tests ran first.
fn setup_initialized_backend() -> Option<CudaBackend> {
    let mut backend = try_cuda_backend()?;
    reset_bf16_gemmex_state_for_tests();
    let hp = minimal_test_hyperparams();
    match backend.init(&hp) {
        Ok(()) => Some(backend),
        Err(e) => {
            eprintln!("skipping: init failed ({e}) -- likely missing CUDA SDK");
            None
        }
    }
}

/// Test 1: forced cuBLAS failure during the FIRST BF16 matvec dispatch
/// triggers the legacy-kernel fall-through. The in-flight "request"
/// (here a single matvec) must complete, the runtime-fallback flag
/// must be armed, the once-only warning must be emitted, AND the
/// legacy-kernel output must numerically match the GemmEx-success
/// output for the same inputs.
///
/// This is the acceptance criterion (2) end-to-end verification:
/// "A per-call cuBLAS failure inside `launch_hgemv_bf16` does NOT
/// propagate up; it sets a backend-level fallback flag, logs a single
/// warning (eprintln-once), and re-runs the same step on the legacy
/// kernel."
#[test]
fn bf16_runtime_failure_triggers_legacy_fallback_and_request_survives() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let backend = match setup_initialized_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };

    // Small but non-trivial matvec: 32x16. Inputs all 1.0, weights
    // identity-like. The exact values don't matter -- we just need both
    // the GemmEx path and the legacy path to compute identically.
    let in_dim = 16usize;
    let out_dim = 32usize;
    let weight_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let weight_bf16 = f32_to_bf16_bytes(&weight_f32);
    let input_f32: Vec<f32> = (0..in_dim).map(|i| 1.0 + (i as f32) * 0.1).collect();

    // Pass 1: baseline GemmEx path (no injection). Captures the
    // expected output the legacy kernel must match.
    let baseline = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test1_baseline",
        )
        .expect("baseline GemmEx dispatch must succeed");
    assert_eq!(
        baseline.len(),
        out_dim,
        "baseline output must have out_dim elements",
    );
    assert!(
        !bf16_gemmex_fallback_armed_for_tests(),
        "baseline dispatch must not arm runtime fallback",
    );

    // Pass 2: inject a CUBLAS failure -- next dispatch must take the
    // legacy fall-through arm WITHOUT propagating an error to the
    // caller, AND must arm the runtime fallback flag, AND must emit
    // the once-only warning. The returned output must equal the
    // baseline within BF16 tolerance.
    inject_next_bf16_cublas_failure();
    let fallback = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test1_after_inject",
        )
        .expect(
            "BF16 matvec must not error out after a per-call cuBLAS \
             failure -- the wrapper must fall through to the legacy \
             matvec_bf16 kernel and the in-flight request must survive",
        );
    assert_eq!(fallback.len(), out_dim);
    assert!(
        bf16_gemmex_fallback_armed_for_tests(),
        "wrapper must arm the runtime-fallback flag after a cuBLAS \
         failure (BF16_GEMMEX_FALLBACK_ARMED must be true)",
    );
    assert!(
        bf16_gemmex_runtime_warning_emitted_for_tests(),
        "wrapper must emit the once-only runtime-fallback warning \
         via BF16_GEMMEX_RUNTIME_WARNING.get_or_init",
    );

    // Numerical agreement: both GemmEx and the legacy kernel compute
    // the same mathematical op on the same BF16 weights. Differences
    // are bounded by BF16 rounding in the accumulator order; 1e-2
    // absolute tolerance on small magnitudes is the established BF16
    // numerical tolerance for matvecs of this shape.
    for (i, (a, b)) in baseline.iter().zip(fallback.iter()).enumerate() {
        let diff = (a - b).abs();
        let mag = a.abs().max(b.abs()).max(1.0);
        assert!(
            diff <= 1e-2 * mag,
            "baseline[{i}]={a} vs fallback[{i}]={b}, diff={diff} \
             exceeds BF16 tolerance",
        );
    }

    // Pass 3: subsequent dispatches must take the legacy path silently
    // (the gate `bf16_gemmex_enabled()` is now closed because
    // FALLBACK_ARMED == true; no new injection is needed). The output
    // must still match the baseline within tolerance.
    let post_fallback = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test1_post_fallback",
        )
        .expect("post-fallback dispatch must succeed on the legacy path");
    assert!(
        bf16_gemmex_fallback_armed_for_tests(),
        "runtime-fallback flag must remain armed (monotonic)",
    );
    for (i, (a, b)) in baseline.iter().zip(post_fallback.iter()).enumerate() {
        let diff = (a - b).abs();
        let mag = a.abs().max(b.abs()).max(1.0);
        assert!(
            diff <= 1e-2 * mag,
            "baseline[{i}]={a} vs post_fallback[{i}]={b}, diff={diff}",
        );
    }
}

/// Test 2: forced cuBLAS failure during a BF16 matvec with residual
/// (`output = W^T * input + residual`) triggers the legacy-kernel
/// fall-through on the residual variant. Verifies the per-call fault
/// injection covers both `launch_hgemv_bf16` and
/// `launch_hgemv_bf16_residual` arms (the inject check is duplicated
/// in both functions; this test exercises the residual one).
#[test]
fn bf16_inject_residual_variant_triggers_legacy_fallback() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let backend = match setup_initialized_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };

    let in_dim = 16usize;
    let out_dim = 32usize;
    let weight_f32: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f32) * 0.005)
        .collect();
    let weight_bf16 = f32_to_bf16_bytes(&weight_f32);
    let input_f32: Vec<f32> = (0..in_dim).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let residual_f32: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.03).collect();

    let baseline = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, Some(&residual_f32),
            "fault_inject_test2_residual_baseline",
        )
        .expect("residual baseline GemmEx dispatch must succeed");
    assert!(
        !bf16_gemmex_fallback_armed_for_tests(),
        "residual baseline dispatch must not arm runtime fallback",
    );

    inject_next_bf16_cublas_failure();
    let fallback = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, Some(&residual_f32),
            "fault_inject_test2_residual_after_inject",
        )
        .expect(
            "BF16 residual matvec must not error out after a per-call \
             cuBLAS failure -- the wrapper must fall through to the \
             legacy matvec_bf16_residual kernel and the in-flight \
             request must survive",
        );
    assert!(
        bf16_gemmex_fallback_armed_for_tests(),
        "residual wrapper must arm the runtime-fallback flag",
    );

    for (i, (a, b)) in baseline.iter().zip(fallback.iter()).enumerate() {
        let diff = (a - b).abs();
        let mag = a.abs().max(b.abs()).max(1.0);
        assert!(
            diff <= 1e-2 * mag,
            "residual baseline[{i}]={a} vs fallback[{i}]={b}, diff={diff}",
        );
    }
}

/// Test 3: the inject flag is one-shot. A single call to
/// `inject_next_bf16_cublas_failure` triggers at most one synthetic
/// CublasFailure. The very next dispatch after the injection consumed
/// it must take the production code path (gate composition; in this
/// test, since FALLBACK_ARMED is now true, that path goes to legacy
/// silently -- not via the inject seam, which has cleared its flag).
///
/// This is the load-bearing invariant that protects tests from cascade
/// failures: a single inject affects exactly one call, no more.
#[test]
fn bf16_inject_clears_after_single_use() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let backend = match setup_initialized_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };

    let in_dim = 16usize;
    let out_dim = 32usize;
    let weight_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.01).collect();
    let weight_bf16 = f32_to_bf16_bytes(&weight_f32);
    let input_f32 = vec![1.0f32; in_dim];

    // Inject once; first dispatch consumes the inject -> CublasFailure ->
    // arms fallback -> falls through to legacy.
    inject_next_bf16_cublas_failure();
    let _first = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test3_first",
        )
        .expect("first dispatch must survive the injected failure");
    assert!(
        bf16_gemmex_fallback_armed_for_tests(),
        "first dispatch must arm the fallback flag",
    );

    // Second dispatch: no new injection. The wrapper's gate composition
    // sees FALLBACK_ARMED == true and routes to legacy without
    // entering the GemmEx code path (so the inject check is never
    // reached). The dispatch must succeed. Even if a stray re-trigger
    // of the inject flag happened, the second call would have taken
    // the same fall-through path. The assertion that matters is:
    // dispatch succeeds AND no panic/error/inject-cascade occurs.
    let _second = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test3_second",
        )
        .expect("second dispatch (no fresh inject) must succeed via legacy path");
    assert!(
        bf16_gemmex_fallback_armed_for_tests(),
        "fallback flag must remain armed across subsequent dispatches",
    );

    // Stronger one-shot evidence: reset the FALLBACK_ARMED flag so the
    // gate would otherwise re-open the GemmEx path, then dispatch.
    // If the inject flag were sticky (not one-shot), this would
    // re-trigger a CublasFailure and re-arm. Instead, the inject was
    // consumed on the first dispatch (via the `swap(false, ...)` in
    // the inject check) and this dispatch must take the GemmEx path
    // successfully without arming fallback.
    reset_bf16_gemmex_state_for_tests();
    assert!(
        !bf16_gemmex_fallback_armed_for_tests(),
        "state reset must clear the fallback flag",
    );
    let _third = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "fault_inject_test3_third_post_reset",
        )
        .expect("post-reset dispatch must succeed via GemmEx (no inject left)");
    assert!(
        !bf16_gemmex_fallback_armed_for_tests(),
        "fallback flag must NOT be re-armed because the inject flag \
         was already consumed on the first dispatch (one-shot \
         semantics) -- if this fails, the inject is sticky and the \
         test guard fn is broken",
    );
}

/// Test 4: the once-only runtime warning is emitted at most once
/// across multiple subsequent inject + arm cycles. The OnceLock
/// `get_or_init` contract guarantees the eprintln body runs exactly
/// once per process; this test exercises multiple arming paths and
/// asserts the warning OnceLock has been set (it must remain set --
/// observability is monotonic-on once observed).
#[test]
fn bf16_warning_emitted_exactly_once_across_repeated_failures() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let backend = match setup_initialized_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device available");
            return;
        }
    };

    let in_dim = 16usize;
    let out_dim = 32usize;
    let weight_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.01).collect();
    let weight_bf16 = f32_to_bf16_bytes(&weight_f32);
    let input_f32 = vec![1.0f32; in_dim];

    // First inject -> first arming -> warning OnceLock gets set.
    inject_next_bf16_cublas_failure();
    let _ = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "warning_emitted_first",
        )
        .expect("first dispatch must survive injected failure");
    assert!(
        bf16_gemmex_runtime_warning_emitted_for_tests(),
        "first injected failure must emit the runtime warning",
    );

    // Reset just the FALLBACK_ARMED flag (not the warning OnceLock --
    // OnceLocks cannot be reset) and inject again. The next dispatch
    // will trigger another `arm_bf16_gemmex_runtime_fallback` call,
    // but the OnceLock `get_or_init` will NOT re-execute the eprintln
    // body. The warning observer remains true; the OnceLock state is
    // monotonic-on.
    reset_bf16_gemmex_state_for_tests();
    // The reset helper does NOT touch the warning OnceLock (it can't);
    // BF16_GEMMEX_RUNTIME_WARNING remains set across the reset and
    // subsequent rearms. This is the contract we are asserting.
    inject_next_bf16_cublas_failure();
    let _ = backend
        .dispatch_bf16_matvec_for_tests(
            &weight_bf16, &input_f32, out_dim, in_dim, None,
            "warning_emitted_second_arm_attempt",
        )
        .expect("second dispatch with re-injected failure must survive");
    assert!(
        bf16_gemmex_runtime_warning_emitted_for_tests(),
        "warning OnceLock must remain set across subsequent armings",
    );
    // No way to portably assert "exactly one eprintln" without a
    // capturing stderr helper; the OnceLock's `is_some()` plus the
    // `get_or_init` contract together prove at-most-once. The
    // contract is enforced by the language ( `OnceLock::get_or_init`
    // runs the closure at most once for the entire process
    // lifetime).
}
