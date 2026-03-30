//! Integration tests for the batched RMSNorm CUDA kernel (prefill_norm.cu).
//!
//! Verifies that `rmsnorm_batched` produces identical output to applying
//! single-row `cpu_rmsnorm` independently on each row. Covers batch=1
//! through batch=256, dim=2048, and edge cases (zero input, uniform values).
//!
//! Requires a CUDA-capable GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_prefill_norm_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// CPU reference: rmsnorm(x, weight, eps) -> out for a single [dim] vector.
fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

/// CPU reference for batched RMSNorm: apply single-row rmsnorm to each row.
fn cpu_rmsnorm_batched(x: &[f32], weight: &[f32], eps: f32, batch: usize, dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), batch * dim);
    assert_eq!(weight.len(), dim);
    let mut out = vec![0.0f32; batch * dim];
    for b in 0..batch {
        let row_in = &x[b * dim..(b + 1) * dim];
        let row_out = cpu_rmsnorm(row_in, weight, eps);
        out[b * dim..(b + 1) * dim].copy_from_slice(&row_out);
    }
    out
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// Compile the batched RMSNorm kernel and return the function handle.
fn compile_rmsnorm_batched(
    ctx: &std::sync::Arc<CudaContext>,
) -> cudarc::driver::CudaFunction {
    let src = lumen_runtime::cuda::shaders::PREFILL_NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for prefill_norm.cu");
    let module = ctx.load_module(ptx).expect("Failed to load prefill_norm module");
    module
        .load_function("rmsnorm_batched")
        .expect("Failed to load rmsnorm_batched function")
}

/// Block size for RMSNorm: min(dim, 1024) rounded down to a multiple of 32.
fn rmsnorm_block_size(dim: usize) -> u32 {
    let bs = dim.min(1024);
    let bs = (bs / 32) * 32;
    bs.max(32) as u32
}

/// Shared memory bytes for RMSNorm: (block_size / 32) * 4.
fn rmsnorm_shared_bytes(block_size: u32) -> u32 {
    (block_size / 32) * 4
}

/// Launch rmsnorm_batched on GPU and return the result as a host Vec.
fn gpu_rmsnorm_batched(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    func: &cudarc::driver::CudaFunction,
    x: &[f32],
    weight: &[f32],
    eps: f32,
    batch: usize,
    dim: usize,
) -> Vec<f32> {
    let x_gpu = stream.clone_htod(x).unwrap();
    let w_gpu = stream.clone_htod(weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(batch * dim).unwrap();

    let block_size = rmsnorm_block_size(dim);
    let shared_bytes = rmsnorm_shared_bytes(block_size);
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;

    unsafe {
        stream
            .launch_builder(func)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut out_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(cfg)
    }
    .expect("rmsnorm_batched launch failed");

    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Assert that two f32 slices match within tolerance, printing first N mismatches.
fn assert_f32_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance {
            if mismatches < 5 {
                eprintln!(
                    "  {label}[{i}]: GPU={a:.8}, CPU={e:.8}, diff={diff:.2e}"
                );
            }
            mismatches += 1;
        }
        max_diff = max_diff.max(diff);
    }
    assert!(
        mismatches == 0,
        "{label}: {mismatches} mismatches out of {} (max_diff={max_diff:.2e}, tol={tolerance:.2e})",
        actual.len()
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Core test: batch=128, dim=2048. Matches 128 sequential rmsnorm calls.
#[test]
fn test_rmsnorm_batched_128x2048() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 128;
    let dim = 2048;
    let eps = 1e-5f32;

    // Deterministic input: each element = sin(index) to avoid trivial zeros.
    let x: Vec<f32> = (0..batch * dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 * 0.001).cos()).collect();

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("batch128_dim2048", &actual, &expected, 1e-5);
}

/// batch=1: degenerate case, should behave identically to single-row rmsnorm.
#[test]
fn test_rmsnorm_batched_single_row() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 1;
    let dim = 2048;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let weight: Vec<f32> = vec![1.0f32; dim];

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("batch1_dim2048", &actual, &expected, 1e-5);
}

/// batch=256: maximum batch size per acceptance criteria.
#[test]
fn test_rmsnorm_batched_256() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 256;
    let dim = 2048;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..batch * dim)
        .map(|i| ((i as f32) * 0.007).sin() * 2.0)
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.002).sin()).collect();

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("batch256_dim2048", &actual, &expected, 1e-5);
}

/// Small dim=64 (exactly 2 warps with block_size=64): tests min warp reduction path.
#[test]
fn test_rmsnorm_batched_small_dim() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 32;
    let dim = 64;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.05).collect();
    let weight: Vec<f32> = vec![1.0f32; dim];

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("batch32_dim64", &actual, &expected, 1e-5);
}

/// Large dim=4096 (Llama 8B hidden_dim): ensures multi-pass loop works.
#[test]
fn test_rmsnorm_batched_large_dim() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 64;
    let dim = 4096;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..batch * dim)
        .map(|i| ((i as f32) * 0.003).cos())
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.8 + (i as f32 * 0.0005).sin()).collect();

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("batch64_dim4096", &actual, &expected, 1e-5);
}

/// Zero input: rmsnorm(0, ...) = 0 (the scale is 1/sqrt(eps), but 0*anything=0).
#[test]
fn test_rmsnorm_batched_zero_input() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 4;
    let dim = 256;
    let eps = 1e-5f32;

    let x = vec![0.0f32; batch * dim];
    let weight = vec![1.0f32; dim];

    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    // All outputs should be exactly 0.0 since x[i]=0 and 0*anything=0.
    for (i, &v) in actual.iter().enumerate() {
        assert!(
            v == 0.0,
            "zero_input[{i}]: expected 0.0, got {v}"
        );
    }
}

/// Uniform input: all elements = 1.0. Tests that the reduction sum is correct.
#[test]
fn test_rmsnorm_batched_uniform() {
    let (ctx, stream) = create_context();
    let func = compile_rmsnorm_batched(&ctx);

    let batch = 8;
    let dim = 512;
    let eps = 1e-5f32;

    let x = vec![1.0f32; batch * dim];
    let weight = vec![1.0f32; dim];

    let expected = cpu_rmsnorm_batched(&x, &weight, eps, batch, dim);
    let actual = gpu_rmsnorm_batched(&stream, &func, &x, &weight, eps, batch, dim);

    assert_f32_close("uniform_input", &actual, &expected, 1e-5);
}

/// Verify that the standalone prefill_norm.cu kernel produces identical output
/// to the rmsnorm_batched in prefill_kernels.cu (the existing monolithic file).
/// This ensures the extraction into a separate compilation unit is correct.
#[test]
fn test_rmsnorm_batched_matches_prefill_kernels() {
    let (ctx, stream) = create_context();

    // Compile from standalone file.
    let standalone_func = compile_rmsnorm_batched(&ctx);

    // Compile from monolithic prefill_kernels.cu.
    let prefill_src = lumen_runtime::cuda::shaders::PREFILL_KERNEL_SOURCE;
    let prefill_ptx =
        compile_ptx(prefill_src).expect("NVRTC compile failed for prefill_kernels.cu");
    let prefill_module = ctx
        .load_module(prefill_ptx)
        .expect("Failed to load prefill_kernels module");
    let monolithic_func = prefill_module
        .load_function("rmsnorm_batched")
        .expect("Failed to load rmsnorm_batched from prefill_kernels");

    let batch = 64;
    let dim = 2048;
    let eps = 1e-5f32;

    let x: Vec<f32> = (0..batch * dim)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.001).cos()).collect();

    let standalone_out =
        gpu_rmsnorm_batched(&stream, &standalone_func, &x, &weight, eps, batch, dim);
    let monolithic_out =
        gpu_rmsnorm_batched(&stream, &monolithic_func, &x, &weight, eps, batch, dim);

    // Both GPU implementations must produce bit-identical output (same PTX code path).
    for (i, (&s, &m)) in standalone_out.iter().zip(monolithic_out.iter()).enumerate() {
        assert!(
            s == m,
            "standalone vs monolithic[{i}]: {s} != {m}"
        );
    }
}
