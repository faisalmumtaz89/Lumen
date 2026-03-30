//! Integration tests for cuBLAS GEMV vs custom matvec_f32 kernel.
//!
//! Verifies that cuBLAS SGEMV produces the same output as the custom NVRTC
//! matvec_f32 kernel within 1e-4 tolerance. Tests both plain and residual
//! variants across multiple matrix sizes.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_cublas_test

#![cfg(feature = "cuda")]

use cudarc::cublas::{CudaBlas, Gemv, GemvConfig, sys};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const TOLERANCE: f32 = 1e-4;

/// CPU reference: y = W * x (row-major W: [out_dim, in_dim])
fn cpu_matvec(weight: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0f32;
        for col in 0..in_dim {
            sum += weight[row * in_dim + col] * x[col];
        }
        y[row] = sum;
    }
    y
}

/// CPU reference: y = W * x + residual
fn cpu_matvec_residual(
    weight: &[f32],
    x: &[f32],
    residual: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut y = cpu_matvec(weight, x, out_dim, in_dim);
    for i in 0..out_dim {
        y[i] += residual[i];
    }
    y
}

/// Run the custom matvec_f32 kernel and return the output.
fn run_custom_kernel(
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
    let launch_cfg = LaunchConfig {
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
            .launch(launch_cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Run cuBLAS SGEMV and return the output.
fn run_cublas_gemv(
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

/// Run cuBLAS SGEMV with residual (beta=1.0) and return the output.
fn run_cublas_gemv_residual(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    blas: &CudaBlas,
    w_gpu: &CudaSlice<f32>,
    x_gpu: &CudaSlice<f32>,
    residual_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    // Copy residual into output buffer.
    stream.memcpy_dtod(residual_gpu, &mut out_gpu).unwrap();
    let cfg = GemvConfig {
        trans: sys::cublasOperation_t::CUBLAS_OP_T,
        m: in_dim as i32,
        n: out_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        incx: 1,
        beta: 1.0f32,
        incy: 1,
    };
    unsafe { blas.gemv(cfg, w_gpu, x_gpu, &mut out_gpu) }.unwrap();
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Assert two vectors match within tolerance, with descriptive error messages.
fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{label}: max_diff={max_diff} at [{max_idx}] (custom={}, cublas={}, tol={tol})",
        a[max_idx],
        b[max_idx],
    );
}

// ---------------------------------------------------------------------------
// cuBLAS GEMV vs custom kernel: small matrix
// ---------------------------------------------------------------------------

#[test]
fn cublas_vs_custom_small() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let out_dim = 4;
    let in_dim = 8;
    let weight: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.01 - 0.15).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let expected = cpu_matvec(&weight, &x, out_dim, in_dim);

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();

    let custom = run_custom_kernel(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim);
    let cublas = run_cublas_gemv(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim);

    assert_close("custom_vs_cpu", &custom, &expected, TOLERANCE);
    assert_close("cublas_vs_cpu", &cublas, &expected, TOLERANCE);
    assert_close("cublas_vs_custom", &cublas, &custom, TOLERANCE);
}

// ---------------------------------------------------------------------------
// cuBLAS GEMV vs custom kernel: realistic transformer dimensions
// ---------------------------------------------------------------------------

#[test]
fn cublas_vs_custom_4096x4096() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let out_dim = 4096;
    let in_dim = 4096;
    // Pseudo-random weights via simple LCG.
    let mut seed = 42u64;
    let mut rng = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let weight: Vec<f32> = (0..out_dim * in_dim).map(|_| rng()).collect();
    let x: Vec<f32> = (0..in_dim).map(|_| rng()).collect();

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();

    let custom = run_custom_kernel(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim);
    let cublas = run_cublas_gemv(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim);

    assert_close("cublas_vs_custom_4k", &cublas, &custom, TOLERANCE);
}

// ---------------------------------------------------------------------------
// cuBLAS GEMV vs custom kernel: non-square (inter_dim x hidden_dim)
// ---------------------------------------------------------------------------

#[test]
fn cublas_vs_custom_11008x4096() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let out_dim = 11008;
    let in_dim = 4096;
    let mut seed = 123u64;
    let mut rng = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let weight: Vec<f32> = (0..out_dim * in_dim).map(|_| rng()).collect();
    let x: Vec<f32> = (0..in_dim).map(|_| rng()).collect();

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();

    let custom = run_custom_kernel(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim);
    let cublas = run_cublas_gemv(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim);

    assert_close("cublas_vs_custom_11k_4k", &cublas, &custom, TOLERANCE);
}

// ---------------------------------------------------------------------------
// cuBLAS GEMV with residual
// ---------------------------------------------------------------------------

#[test]
fn cublas_residual_vs_cpu() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let out_dim = 2048;
    let in_dim = 4096;
    let mut seed = 77u64;
    let mut rng = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let weight: Vec<f32> = (0..out_dim * in_dim).map(|_| rng()).collect();
    let x: Vec<f32> = (0..in_dim).map(|_| rng()).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| rng()).collect();
    let expected = cpu_matvec_residual(&weight, &x, &residual, out_dim, in_dim);

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let r_gpu = stream.clone_htod(&residual).unwrap();

    let cublas = run_cublas_gemv_residual(&stream, &blas, &w_gpu, &x_gpu, &r_gpu, out_dim, in_dim);
    assert_close("cublas_residual_vs_cpu", &cublas, &expected, TOLERANCE);
}

// ---------------------------------------------------------------------------
// Edge case: very small dimensions
// ---------------------------------------------------------------------------

#[test]
fn cublas_vs_custom_1x1() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let weight = vec![3.0f32];
    let x = vec![7.0f32];
    let expected = vec![21.0f32];

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();

    let custom = run_custom_kernel(&stream, &func, &w_gpu, &x_gpu, 1, 1);
    let cublas = run_cublas_gemv(&stream, &blas, &w_gpu, &x_gpu, 1, 1);

    assert_close("custom_1x1", &custom, &expected, TOLERANCE);
    assert_close("cublas_1x1", &cublas, &expected, TOLERANCE);
}

// ---------------------------------------------------------------------------
// Edge case: non-4-aligned in_dim (forces scalar path in custom kernel)
// ---------------------------------------------------------------------------

#[test]
fn cublas_vs_custom_non_aligned() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    let src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_f32").unwrap();

    let out_dim = 17;
    let in_dim = 13; // Not a multiple of 4
    let weight: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.003 - 0.3).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.07 + 0.1).collect();
    let expected = cpu_matvec(&weight, &x, out_dim, in_dim);

    let w_gpu = stream.clone_htod(&weight).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();

    let custom = run_custom_kernel(&stream, &func, &w_gpu, &x_gpu, out_dim, in_dim);
    let cublas = run_cublas_gemv(&stream, &blas, &w_gpu, &x_gpu, out_dim, in_dim);

    assert_close("custom_non_aligned", &custom, &expected, TOLERANCE);
    assert_close("cublas_non_aligned", &cublas, &expected, TOLERANCE);
    assert_close("cublas_vs_custom_non_aligned", &cublas, &custom, TOLERANCE);
}
