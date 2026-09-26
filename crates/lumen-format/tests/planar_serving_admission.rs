//! Planar-scheme admission per serving backend: CUDA with the kill switch on accepts NVFP4 and FP8, and
//! the CPU backend, a build without CUDA, Metal and CUDA with the kill switch off each refuse by name.
//!
//! This drives the predicate the binaries call (`scheme_has_no_serving_kernels`) through every
//! configuration, so what is tested is the admission decision itself and not a mock of it. The expected
//! answers are stated independently of the implementation: the two planar schemes are refused everywhere
//! except CUDA with the switch on, and no other scheme is refused by this rule on any backend. (CtInt4G32
//! is CUDA-only as well, but the binaries refuse it by their own check after admission.)
//!
//! Pure logic, no GPU needed:
//!
//! ```text
//! cargo test --release -p lumen-format --test planar_serving_admission
//! ```
//!
//! A build without the `cuda` feature is the `Cpu` case here off macOS, because that is what such a build
//! resolves its automatic default to there (on macOS the default is Metal). `unservable_scheme` over a converted artifact is covered
//! in `lumen-convert`, and the end-to-end refusal through the binaries by the `planar_scheme_admission`
//! tests in `lumen-cli` and `lumen-server`; this pins the decision itself.

use lumen_format::serving_rules::{
    cuda_planar_kernels_enabled, no_serving_kernels_message, scheme_has_no_serving_kernels,
    set_cuda_planar_kernels_enabled, ServingBackend,
};
use lumen_format::QuantScheme;
use std::sync::{Mutex, MutexGuard};

/// The planar switch is process-wide and the test harness runs tests on parallel threads, so every test
/// that reads or sets it holds this lock for its whole body.
static FLAG_LOCK: Mutex<()> = Mutex::new(());

fn lock_flag() -> MutexGuard<'static, ()> {
    FLAG_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The predicate must be exhaustive in BOTH directions: every scheme on every backend has a decided answer,
/// and the two planar schemes are exactly the ones whose answer depends on the backend.
#[test]
fn every_scheme_on_every_backend_has_a_decided_answer() {
    let _flag = lock_flag();
    set_cuda_planar_kernels_enabled(true);
    let all = [
        QuantScheme::F32,
        QuantScheme::F16,
        QuantScheme::Bf16,
        QuantScheme::Q8_0,
        QuantScheme::Q4_0,
        QuantScheme::Q4_1,
        QuantScheme::Q4_K,
        QuantScheme::Q5_0,
        QuantScheme::Q5_K,
        QuantScheme::Q6_K,
        QuantScheme::Q2_K,
        QuantScheme::Q3_K,
        QuantScheme::CtInt4G32,
        QuantScheme::Nvfp4,
        QuantScheme::Fp8E4M3,
    ];
    for q in all {
        for b in [
            ServingBackend::Cuda,
            ServingBackend::Metal,
            ServingBackend::Cpu,
        ] {
            let refused = scheme_has_no_serving_kernels(q, b);
            // The two planar schemes are refused everywhere EXCEPT CUDA with the switch on.
            let planar = matches!(q, QuantScheme::Nvfp4 | QuantScheme::Fp8E4M3);
            let expect = planar && b != ServingBackend::Cuda;
            assert_eq!(refused, expect, "{q:?} on {b:?}");
        }
    }
}

/// CUDA with the switch on accepts; CPU, a build without CUDA, Metal, and CUDA with the kill switch off
/// refuse.
#[test]
fn cuda_accepts_and_cpu_no_cuda_and_kill_switch_refuse() {
    let _flag = lock_flag();
    set_cuda_planar_kernels_enabled(true);
    assert!(
        !scheme_has_no_serving_kernels(QuantScheme::Nvfp4, ServingBackend::Cuda),
        "CUDA with the switch on must ACCEPT"
    );
    assert!(
        !scheme_has_no_serving_kernels(QuantScheme::Fp8E4M3, ServingBackend::Cuda),
        "CUDA with the switch on must ACCEPT FP8 too"
    );
    assert!(
        scheme_has_no_serving_kernels(QuantScheme::Nvfp4, ServingBackend::Cpu),
        "the CPU path must REFUSE"
    );
    // The no-CUDA build resolves to Cpu, so it is this arm.
    assert!(
        scheme_has_no_serving_kernels(QuantScheme::Fp8E4M3, ServingBackend::Cpu),
        "a no-CUDA build must REFUSE"
    );
    assert!(
        scheme_has_no_serving_kernels(QuantScheme::Nvfp4, ServingBackend::Metal),
        "Metal must REFUSE"
    );

    // The kill switch: CUDA stops accepting, and everything else is unchanged.
    set_cuda_planar_kernels_enabled(false);
    assert!(
        scheme_has_no_serving_kernels(QuantScheme::Nvfp4, ServingBackend::Cuda),
        "the kill switch must make CUDA REFUSE"
    );
    assert!(
        scheme_has_no_serving_kernels(QuantScheme::Fp8E4M3, ServingBackend::Cuda),
        "the kill switch covers BOTH planar schemes"
    );
    // On CUDA the switch is the only reason to refuse, so the refusal names it.
    let message = no_serving_kernels_message(QuantScheme::Nvfp4, ServingBackend::Cuda);
    assert!(
        message.contains("Nvfp4") && message.contains("LUMEN_CUDA_NVFP4=0"),
        "the CUDA refusal must name the scheme and the switch: {message}"
    );
    // Restore, so the switch's default cannot leak into another test in this binary.
    set_cuda_planar_kernels_enabled(true);
    assert!(cuda_planar_kernels_enabled(), "the switch is back on");
}

/// The refusal names the scheme, which is the whole point of refusing at admission rather than letting it
/// surface later as a missing kernel or a misread plane.
#[test]
fn the_refusal_names_the_scheme() {
    let nvfp4 = no_serving_kernels_message(QuantScheme::Nvfp4, ServingBackend::Cpu);
    assert!(
        nvfp4.contains("Nvfp4") && nvfp4.contains("is served on CUDA only"),
        "the message must name NVFP4 and where it is served: {nvfp4}"
    );
    let fp8 = no_serving_kernels_message(QuantScheme::Fp8E4M3, ServingBackend::Metal);
    assert!(
        fp8.contains("Fp8E4M3") && fp8.contains("Metal"),
        "the message must name FP8 and the backend: {fp8}"
    );
    assert_ne!(nvfp4, fp8, "the two schemes must not share one message");
}

/// A NON-planar artifact is admitted on every backend, so the planar rule cannot refuse what already served.
#[test]
fn an_ordinary_scheme_is_admitted_everywhere() {
    for b in [
        ServingBackend::Cuda,
        ServingBackend::Metal,
        ServingBackend::Cpu,
    ] {
        assert!(
            !scheme_has_no_serving_kernels(QuantScheme::Q4_K, b),
            "a Q4_K artifact must be admitted on {b:?}"
        );
        assert!(
            !scheme_has_no_serving_kernels(QuantScheme::Q8_0, b),
            "a Q8_0 artifact must be admitted on {b:?}"
        );
    }
}
