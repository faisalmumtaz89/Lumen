//! A/B benchmark: custom matvec_f32 kernel vs cuBLAS SGEMV.
//!
//! Measures latency, GFLOPS, and effective memory bandwidth for both the custom
//! NVRTC-compiled `matvec_f32` kernel and cuBLAS SGEMV across four dimension
//! configurations that reflect real transformer workloads:
//!
//!   - 4096x4096   (8B hidden projection)
//!   - 14336x4096  (8B FFN gate/up)
//!   - 2048x2048   (1B hidden projection)
//!   - 8192x2048   (1B FFN gate/up)
//!
//! Also verifies that both implementations produce matching output within 1e-4.
//!
//! Run via Modal:
//!     modal run modal/bench_cublas_ab.py
//!
//! Run locally (requires NVIDIA GPU):
//!     cargo test -p lumen-runtime --features cuda --release \
//!         --test cuda_cublas_ab_test -- --ignored --nocapture

#![cfg(feature = "cuda")]

use cudarc::cublas::{CudaBlas, Gemv, GemvConfig, sys};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

const CORRECTNESS_TOL: f32 = 1e-4;

/// Deterministic pseudo-random f32 values via LCG.
fn lcg_f32_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Dimension configuration for one benchmark case.
struct DimConfig {
    label: &'static str,
    out_dim: usize,
    in_dim: usize,
}

/// Results from a single benchmark run.
struct BenchResult {
    us_per_op: f64,
    gflops: f64,
    gb_per_s: f64,
}

/// Run the custom matvec_f32 kernel for `iterations` and return timing stats.
fn bench_custom(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    func: &cudarc::driver::CudaFunction,
    w_gpu: &CudaSlice<f32>,
    x_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    warmup: usize,
    iterations: usize,
) -> BenchResult {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(func)
                .arg(w_gpu)
                .arg(x_gpu)
                .arg(&mut out_gpu)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(func)
                .arg(w_gpu)
                .arg(x_gpu)
                .arg(&mut out_gpu)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();

    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let flops = 2.0 * out_dim as f64 * in_dim as f64;
    let gflops = (flops * iterations as f64) / elapsed.as_secs_f64() / 1e9;
    let bytes = (out_dim * in_dim * 4 + in_dim * 4 + out_dim * 4) as f64;
    let gb_per_s = (bytes * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    BenchResult { us_per_op, gflops, gb_per_s }
}

/// Run cuBLAS SGEMV for `iterations` and return timing stats.
fn bench_cublas(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    blas: &CudaBlas,
    w_gpu: &CudaSlice<f32>,
    x_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    warmup: usize,
    iterations: usize,
) -> BenchResult {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let gemv_cfg = GemvConfig {
        trans: sys::cublasOperation_t::CUBLAS_OP_T,
        m: in_dim as i32,
        n: out_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        incx: 1,
        beta: 0.0f32,
        incy: 1,
    };

    // Warmup
    for _ in 0..warmup {
        unsafe { blas.gemv(gemv_cfg, w_gpu, x_gpu, &mut out_gpu) }.unwrap();
    }
    stream.synchronize().unwrap();

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { blas.gemv(gemv_cfg, w_gpu, x_gpu, &mut out_gpu) }.unwrap();
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();

    let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let flops = 2.0 * out_dim as f64 * in_dim as f64;
    let gflops = (flops * iterations as f64) / elapsed.as_secs_f64() / 1e9;
    let bytes = (out_dim * in_dim * 4 + in_dim * 4 + out_dim * 4) as f64;
    let gb_per_s = (bytes * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    BenchResult { us_per_op, gflops, gb_per_s }
}

/// Run the custom kernel once and return the output vector.
fn run_custom_once(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    func: &cudarc::driver::CudaFunction,
    w_gpu: &CudaSlice<f32>,
    x_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(func)
            .arg(w_gpu)
            .arg(x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Run cuBLAS SGEMV once and return the output vector.
fn run_cublas_once(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    blas: &CudaBlas,
    w_gpu: &CudaSlice<f32>,
    x_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let cfg = GemvConfig {
        trans: sys::cublasOperation_t::CUBLAS_OP_T,
        m: in_dim as i32,
        n: out_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        incx: 1,
        beta: 0.0f32,
        incy: 1,
    };
    unsafe { blas.gemv(cfg, w_gpu, x_gpu, &mut out_gpu) }.unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Read an environment variable as usize, falling back to the default.
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A/B benchmark across four dimension configurations.
///
/// Reports per-config: custom and cuBLAS latency, GFLOPS, GB/s, and speedup.
/// Verifies correctness (custom vs cuBLAS within 1e-4) for each config.
///
/// Environment variables:
///   LUMEN_AB_WARMUP     -- warmup iterations per kernel (default 100)
///   LUMEN_AB_ITERATIONS -- measured iterations per kernel (default 10000)
#[test]
#[ignore]
fn bench_cublas_ab_comparison() {
    let warmup = env_usize("LUMEN_AB_WARMUP", 100);
    let iterations = env_usize("LUMEN_AB_ITERATIONS", 10000);

    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    // Compile custom kernel
    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_f32.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let configs = [
        DimConfig { label: "8B hidden (4096x4096)", out_dim: 4096, in_dim: 4096 },
        DimConfig { label: "8B FFN gate/up (14336x4096)", out_dim: 14336, in_dim: 4096 },
        DimConfig { label: "1B hidden (2048x2048)", out_dim: 2048, in_dim: 2048 },
        DimConfig { label: "1B FFN gate/up (8192x2048)", out_dim: 8192, in_dim: 2048 },
    ];

    eprintln!();
    eprintln!("======================================================================");
    eprintln!("  cuBLAS SGEMV vs Custom matvec_f32 -- A/B Comparison");
    eprintln!("======================================================================");
    eprintln!("  Warmup:     {} iterations per kernel", warmup);
    eprintln!("  Measured:   {} iterations per kernel", iterations);
    eprintln!();

    // A10G theoretical: 31.2 TFLOPS FP32, 600 GB/s HBM bandwidth
    eprintln!("  A10G specs: 31.2 TFLOPS FP32, 600 GB/s HBM bandwidth");
    eprintln!();

    let mut all_correct = true;

    for config in &configs {
        let out_dim = config.out_dim;
        let in_dim = config.in_dim;
        let n_elements = out_dim * in_dim;

        // Generate deterministic data
        let weight = lcg_f32_vec(42, n_elements);
        let x = lcg_f32_vec(123, in_dim);

        let w_gpu = stream.clone_htod(&weight).unwrap();
        let x_gpu = stream.clone_htod(&x).unwrap();

        // --- Correctness check ---
        let custom_out = run_custom_once(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim);
        let cublas_out = run_cublas_once(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim);

        let mut max_diff: f32 = 0.0;
        let mut max_diff_idx: usize = 0;
        for i in 0..out_dim {
            let diff = (custom_out[i] - cublas_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        let correct = max_diff <= CORRECTNESS_TOL;
        if !correct {
            all_correct = false;
        }

        // --- Performance benchmark ---
        let custom_result =
            bench_custom(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim, warmup, iterations);
        let cublas_result =
            bench_cublas(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim, warmup, iterations);

        let speedup = custom_result.us_per_op / cublas_result.us_per_op;

        // Compute memory size for reference
        let weight_mb = (n_elements * 4) as f64 / (1024.0 * 1024.0);

        eprintln!("----------------------------------------------------------------------");
        eprintln!("  {} ({:.1} MB weight matrix)", config.label, weight_mb);
        eprintln!("----------------------------------------------------------------------");
        eprintln!(
            "  Correctness:  {} (max_diff={:.2e} at [{}], tol={:.0e})",
            if correct { "PASS" } else { "FAIL" },
            max_diff,
            max_diff_idx,
            CORRECTNESS_TOL,
        );
        eprintln!();
        eprintln!("  {:>12} {:>10} {:>10} {:>10}", "", "us/op", "GFLOPS", "GB/s");
        eprintln!(
            "  {:>12} {:>10.1} {:>10.1} {:>10.1}",
            "Custom:", custom_result.us_per_op, custom_result.gflops, custom_result.gb_per_s,
        );
        eprintln!(
            "  {:>12} {:>10.1} {:>10.1} {:>10.1}",
            "cuBLAS:", cublas_result.us_per_op, cublas_result.gflops, cublas_result.gb_per_s,
        );
        eprintln!();
        eprintln!("  Speedup: {:.2}x (cuBLAS vs Custom)", speedup);
        eprintln!(
            "  BW utilization: Custom {:.0}%, cuBLAS {:.0}% (of 600 GB/s)",
            custom_result.gb_per_s / 600.0 * 100.0,
            cublas_result.gb_per_s / 600.0 * 100.0,
        );
        eprintln!();
    }

    eprintln!("======================================================================");
    eprintln!(
        "  Overall correctness: {}",
        if all_correct { "ALL PASS" } else { "SOME FAILED" },
    );
    eprintln!("======================================================================");

    assert!(all_correct, "Correctness check failed for one or more dimension configs");
}
