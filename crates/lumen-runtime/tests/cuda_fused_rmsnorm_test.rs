//! Integration tests for the fused RMSNorm + MatVec kernels.
//!
//! Validates that the two-pass fused approach (compute_rms_scale + fused_norm_matvec)
//! produces output matching the separate rmsnorm + matvec within tolerance.
//!
//! Tests cover:
//! - compute_rms_scale: scalar matches CPU reference
//! - fused_norm_matvec_f32: output matches separate rmsnorm + matvec_f32
//! - fused_norm_dual_matvec_f32: gate+up match separate paths
//! - Edge cases: dim=32 (min warp), dim=4096 (model scale), non-aligned dim
//!
//! Requires a CUDA-capable GPU (run on Modal):
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_fused_rmsnorm_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// CPU reference: RMSNorm(x, weight, eps) -> Vec<f32>
fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

/// CPU reference: RMS scale = 1 / sqrt(mean(x^2) + eps)
fn cpu_rms_scale(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    1.0 / (ms + eps).sqrt()
}

/// CPU reference: matvec out[i] = sum_j(weight[i * dim + j] * x[j])
fn cpu_matvec(weight: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row = &weight[i * in_dim..(i + 1) * in_dim];
        out[i] = row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum();
    }
    out
}

/// Generate deterministic pseudo-random data for reproducible tests.
fn deterministic_data(n: usize, seed: u32) -> Vec<f32> {
    let mut vals = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        // Simple LCG: state = (a * state + c) mod m
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [-1.0, 1.0] range
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        vals.push(f);
    }
    vals
}

// ---------------------------------------------------------------------------
// compute_rms_scale tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_rms_scale_basic() {
    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).expect("Failed to load module");
    let func = module
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");

    let dim = 128usize;
    let eps: f32 = 1e-5;
    let x = deterministic_data(dim, 42);
    let expected = cpu_rms_scale(&x, eps);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();

    let block_size = dim.min(1024) as u32;
    let block_size = (block_size / 32) * 32;
    let block_size = block_size.max(32);
    let shared_bytes = (block_size / 32) * 4;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&x_gpu)
            .arg(&mut scale_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(cfg)
    }
    .expect("compute_rms_scale launch failed");
    stream.synchronize().unwrap();

    let result = stream.clone_dtoh(&scale_gpu).unwrap();
    let rel_err = ((result[0] - expected) / expected).abs();
    assert!(
        rel_err < 1e-5,
        "compute_rms_scale: got {}, expected {}, rel_err={}",
        result[0],
        expected,
        rel_err,
    );
}

#[test]
fn test_compute_rms_scale_large_dim() {
    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).expect("Failed to load module");
    let func = module
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");

    let dim = 4096usize;
    let eps: f32 = 1e-5;
    let x = deterministic_data(dim, 99);
    let expected = cpu_rms_scale(&x, eps);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();

    let block_size = dim.min(1024) as u32;
    let block_size = (block_size / 32) * 32;
    let shared_bytes = (block_size / 32) * 4;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&x_gpu)
            .arg(&mut scale_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(cfg)
    }
    .expect("compute_rms_scale launch failed");
    stream.synchronize().unwrap();

    let result = stream.clone_dtoh(&scale_gpu).unwrap();
    let rel_err = ((result[0] - expected) / expected).abs();
    assert!(
        rel_err < 1e-5,
        "compute_rms_scale dim=4096: got {}, expected {}, rel_err={}",
        result[0],
        expected,
        rel_err,
    );
}

// ---------------------------------------------------------------------------
// fused_norm_matvec_f32 tests
// ---------------------------------------------------------------------------

/// Test that fused_norm_matvec_f32 matches separate rmsnorm + matvec.
#[test]
fn test_fused_norm_matvec_f32_matches_separate() {
    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).expect("Failed to load module");
    let rms_func = module
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");
    let fused_func = module
        .load_function("fused_norm_matvec_f32")
        .expect("Failed to load fused_norm_matvec_f32");

    let dim = 256usize;
    let out_dim = 128usize;
    let eps: f32 = 1e-5;

    let x = deterministic_data(dim, 1);
    let norm_weight = deterministic_data(dim, 2);
    let weight = deterministic_data(out_dim * dim, 3);

    // CPU reference: rmsnorm then matvec.
    let normed = cpu_rmsnorm(&x, &norm_weight, eps);
    let expected = cpu_matvec(&weight, &normed, out_dim, dim);

    // GPU: fused path.
    let x_gpu = stream.clone_htod(&x).unwrap();
    let nw_gpu = stream.clone_htod(&norm_weight).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    // Pass 1: compute_rms_scale.
    let block_size = dim.min(1024) as u32;
    let block_size = ((block_size / 32) * 32).max(32);
    let shared_bytes = (block_size / 32) * 4;
    let rms_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&x_gpu)
            .arg(&mut scale_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(rms_cfg)
    }
    .expect("compute_rms_scale launch failed");

    // Pass 2: fused_norm_matvec_f32.
    let fused_cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    unsafe {
        stream
            .launch_builder(&fused_func)
            .arg(&x_gpu)
            .arg(&scale_gpu)
            .arg(&nw_gpu)
            .arg(&w_gpu)
            .arg(&mut out_gpu)
            .arg(&dim_u32)
            .arg(&out_dim_u32)
            .launch(fused_cfg)
    }
    .expect("fused_norm_matvec_f32 launch failed");
    stream.synchronize().unwrap();

    let result = stream.clone_dtoh(&out_gpu).unwrap();
    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        let denom = expected[i].abs().max(1e-7);
        assert!(
            err / denom < 1e-4,
            "fused_norm_matvec_f32 mismatch at [{}]: got {}, expected {}, err={}",
            i,
            result[i],
            expected[i],
            err,
        );
    }
}

/// Test fused path at model-scale dimensions (4096 -> 4096).
#[test]
fn test_fused_norm_matvec_f32_model_scale() {
    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).expect("Failed to load module");
    let rms_func = module
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");
    let fused_func = module
        .load_function("fused_norm_matvec_f32")
        .expect("Failed to load fused_norm_matvec_f32");

    let dim = 4096usize;
    let out_dim = 4096usize;
    let eps: f32 = 1e-5;

    let x = deterministic_data(dim, 10);
    let norm_weight = deterministic_data(dim, 20);
    let weight = deterministic_data(out_dim * dim, 30);

    let normed = cpu_rmsnorm(&x, &norm_weight, eps);
    let expected = cpu_matvec(&weight, &normed, out_dim, dim);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let nw_gpu = stream.clone_htod(&norm_weight).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let block_size = dim.min(1024) as u32;
    let block_size = ((block_size / 32) * 32).max(32);
    let shared_bytes = (block_size / 32) * 4;
    let dim_u32 = dim as u32;
    let out_dim_u32 = out_dim as u32;

    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&x_gpu)
            .arg(&mut scale_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
    }
    .expect("compute_rms_scale launch failed");

    unsafe {
        stream
            .launch_builder(&fused_func)
            .arg(&x_gpu)
            .arg(&scale_gpu)
            .arg(&nw_gpu)
            .arg(&w_gpu)
            .arg(&mut out_gpu)
            .arg(&dim_u32)
            .arg(&out_dim_u32)
            .launch(LaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .expect("fused_norm_matvec_f32 launch failed");
    stream.synchronize().unwrap();

    let result = stream.clone_dtoh(&out_gpu).unwrap();
    let max_err = result
        .iter()
        .zip(expected.iter())
        .map(|(r, e)| (r - e).abs())
        .fold(0.0f32, f32::max);
    let max_val = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_err / max_val.max(1e-7) < 1e-4,
        "fused_norm_matvec_f32 4096x4096: max_err={}, max_val={}",
        max_err,
        max_val,
    );
}

// ---------------------------------------------------------------------------
// fused_norm_dual_matvec_f32 tests
// ---------------------------------------------------------------------------

/// Test that fused dual matvec matches separate rmsnorm + gate_matvec + up_matvec.
#[test]
fn test_fused_norm_dual_matvec_f32_matches_separate() {
    let (ctx, stream) = create_context();
    let src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).expect("Failed to load module");
    let rms_func = module
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");
    let dual_func = module
        .load_function("fused_norm_dual_matvec_f32")
        .expect("Failed to load fused_norm_dual_matvec_f32");

    let dim = 256usize;
    let out_dim = 512usize;
    let eps: f32 = 1e-5;

    let x = deterministic_data(dim, 100);
    let norm_weight = deterministic_data(dim, 200);
    let w_gate = deterministic_data(out_dim * dim, 300);
    let w_up = deterministic_data(out_dim * dim, 400);

    // CPU reference.
    let normed = cpu_rmsnorm(&x, &norm_weight, eps);
    let expected_gate = cpu_matvec(&w_gate, &normed, out_dim, dim);
    let expected_up = cpu_matvec(&w_up, &normed, out_dim, dim);

    // GPU: fused dual path.
    let x_gpu = stream.clone_htod(&x).unwrap();
    let nw_gpu = stream.clone_htod(&norm_weight).unwrap();
    let wg_gpu = stream.clone_htod(&w_gate).unwrap();
    let wu_gpu = stream.clone_htod(&w_up).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();
    let mut gate_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut up_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let block_size = dim.min(1024) as u32;
    let block_size = ((block_size / 32) * 32).max(32);
    let shared_bytes = (block_size / 32) * 4;
    let dim_u32 = dim as u32;
    let out_dim_u32 = out_dim as u32;

    // Pass 1: compute_rms_scale.
    unsafe {
        stream
            .launch_builder(&rms_func)
            .arg(&x_gpu)
            .arg(&mut scale_gpu)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
    }
    .expect("compute_rms_scale launch failed");

    // Pass 2: fused dual matvec.
    unsafe {
        stream
            .launch_builder(&dual_func)
            .arg(&x_gpu)
            .arg(&scale_gpu)
            .arg(&nw_gpu)
            .arg(&wg_gpu)
            .arg(&wu_gpu)
            .arg(&mut gate_gpu)
            .arg(&mut up_gpu)
            .arg(&dim_u32)
            .arg(&out_dim_u32)
            .launch(LaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .expect("fused_norm_dual_matvec_f32 launch failed");
    stream.synchronize().unwrap();

    let gate_result = stream.clone_dtoh(&gate_gpu).unwrap();
    let up_result = stream.clone_dtoh(&up_gpu).unwrap();

    for i in 0..out_dim {
        let gate_err = (gate_result[i] - expected_gate[i]).abs();
        let gate_denom = expected_gate[i].abs().max(1e-7);
        assert!(
            gate_err / gate_denom < 1e-4,
            "dual gate mismatch at [{}]: got {}, expected {}, err={}",
            i,
            gate_result[i],
            expected_gate[i],
            gate_err,
        );

        let up_err = (up_result[i] - expected_up[i]).abs();
        let up_denom = expected_up[i].abs().max(1e-7);
        assert!(
            up_err / up_denom < 1e-4,
            "dual up mismatch at [{}]: got {}, expected {}, err={}",
            i,
            up_result[i],
            expected_up[i],
            up_err,
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmark test (latency comparison: fused vs separate)
// ---------------------------------------------------------------------------

/// Micro-benchmark: compare fused vs separate rmsnorm+matvec latency.
///
/// Run with: cargo test --features cuda --test cuda_fused_rmsnorm_test -- --nocapture --ignored
#[test]
#[ignore]
fn bench_fused_vs_separate() {
    use std::time::Instant;

    let (ctx, stream) = create_context();

    // Compile fused kernels.
    let fused_src = lumen_runtime::cuda::shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE;
    let fused_ptx = compile_ptx(fused_src).expect("NVRTC compile failed (fused)");
    let fused_mod = ctx.load_module(fused_ptx).expect("Failed to load fused module");
    let rms_func = fused_mod
        .load_function("compute_rms_scale")
        .expect("Failed to load compute_rms_scale");
    let fused_func = fused_mod
        .load_function("fused_norm_matvec_f32")
        .expect("Failed to load fused_norm_matvec_f32");

    // Compile separate kernels.
    let norm_src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let norm_ptx = compile_ptx(norm_src).expect("NVRTC compile failed (norm)");
    let norm_mod = ctx.load_module(norm_ptx).expect("Failed to load norm module");
    let norm_func = norm_mod
        .load_function("rmsnorm")
        .expect("Failed to load rmsnorm");

    let mv_src = lumen_runtime::cuda::shaders::MATVEC_F32_KERNEL_SOURCE;
    let mv_ptx = compile_ptx(mv_src).expect("NVRTC compile failed (matvec)");
    let mv_mod = ctx.load_module(mv_ptx).expect("Failed to load matvec module");
    let mv_func = mv_mod
        .load_function("matvec_f32")
        .expect("Failed to load matvec_f32");

    let dim = 4096usize;
    let out_dim = 4096usize;
    let eps: f32 = 1e-5;
    let iterations = std::env::var("LUMEN_BENCH_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000u32);

    let x = deterministic_data(dim, 42);
    let norm_weight = deterministic_data(dim, 43);
    let weight = deterministic_data(out_dim * dim, 44);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let nw_gpu = stream.clone_htod(&norm_weight).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut normed_gpu: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut scale_gpu: CudaSlice<f32> = stream.alloc_zeros(1).unwrap();
    let mut out_fused_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let norm_block = dim.min(1024) as u32;
    let norm_block = ((norm_block / 32) * 32).max(32);
    let norm_shared = (norm_block / 32) * 4;
    let dim_u32 = dim as u32;
    let out_dim_u32 = out_dim as u32;

    // Warm up.
    for _ in 0..10 {
        stream.synchronize().unwrap();
    }

    // -- Benchmark: separate rmsnorm + matvec --
    stream.synchronize().unwrap();
    let start_sep = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&norm_func)
                .arg(&x_gpu)
                .arg(&nw_gpu)
                .arg(&mut normed_gpu)
                .arg(&eps)
                .arg(&dim_u32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (norm_block, 1, 1),
                    shared_mem_bytes: norm_shared,
                })
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&mv_func)
                .arg(&w_gpu)
                .arg(&normed_gpu)
                .arg(&mut out_gpu)
                .arg(&out_dim_u32)
                .arg(&dim_u32)
                .launch(LaunchConfig {
                    grid_dim: (out_dim as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let elapsed_sep = start_sep.elapsed();

    // -- Benchmark: fused (compute_rms_scale + fused_norm_matvec) --
    stream.synchronize().unwrap();
    let start_fused = Instant::now();
    for _ in 0..iterations {
        unsafe {
            stream
                .launch_builder(&rms_func)
                .arg(&x_gpu)
                .arg(&mut scale_gpu)
                .arg(&eps)
                .arg(&dim_u32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (norm_block, 1, 1),
                    shared_mem_bytes: norm_shared,
                })
        }
        .unwrap();
        unsafe {
            stream
                .launch_builder(&fused_func)
                .arg(&x_gpu)
                .arg(&scale_gpu)
                .arg(&nw_gpu)
                .arg(&w_gpu)
                .arg(&mut out_fused_gpu)
                .arg(&dim_u32)
                .arg(&out_dim_u32)
                .launch(LaunchConfig {
                    grid_dim: (out_dim as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let elapsed_fused = start_fused.elapsed();

    let sep_us = elapsed_sep.as_micros() as f64 / iterations as f64;
    let fused_us = elapsed_fused.as_micros() as f64 / iterations as f64;
    let speedup = sep_us / fused_us;

    println!("\n=== Fused RMSNorm+MatVec Benchmark (dim={dim}, out_dim={out_dim}) ===");
    println!("  Iterations:     {iterations}");
    println!("  Separate:       {sep_us:.2} us/iter (rmsnorm + matvec_f32)");
    println!("  Fused:          {fused_us:.2} us/iter (compute_rms_scale + fused_norm_matvec_f32)");
    println!("  Speedup:        {speedup:.3}x");
    println!(
        "  Saved per iter: {:.2} us ({:.1}%)",
        sep_us - fused_us,
        (1.0 - fused_us / sep_us) * 100.0
    );
}
