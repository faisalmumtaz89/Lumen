//! Integration tests for Q8_0 v3 vectorized matrix-vector multiply CUDA kernel.
//!
//! Tests `matvec_q8_0_v3` and `matvec_q8_0_v3_residual` against CPU reference
//! and v1 kernel output. Requires a CUDA-capable GPU (run on Modal).
//!
//! The v3 kernel uses vectorized 128-bit loads (int for quants, float4 for x)
//! while maintaining the same NR=2 deferred-reduction architecture as v1.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_matvec_q8_v3_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

/// NR=2 rows per block, 128 threads per block (mirrors kernel constants).
const NR: u32 = 2;
const BLOCK_DIM: u32 = 128;

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// Build the launch config for the NR=2 multi-row Q8_0 matvec kernel.
fn q8_launch_config(out_dim: u32) -> LaunchConfig {
    let grid = (out_dim + NR - 1) / NR;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Q8_0 encoding helpers (CPU-side)
// ---------------------------------------------------------------------------

/// Convert an f32 value to IEEE 754 half-precision (16-bit) as raw bits.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp == 0xff {
        let f16_frac = if frac != 0 { 0x200u32 } else { 0 };
        return ((sign << 15) | (0x1f << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return ((sign << 15) | (0x1f << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let m = frac | 0x800000;
        let shift = 1 - new_exp;
        let f16_frac = m >> (13 + shift);
        return ((sign << 15) | f16_frac) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Encode a row of f32 values into Q8_0 blocks.
fn encode_q8_0(values: &[f32]) -> Vec<u8> {
    let block_count = (values.len() + 31) / 32;
    let mut bytes = Vec::with_capacity(block_count * 34);

    for block_idx in 0..block_count {
        let start = block_idx * 32;
        let end = (start + 32).min(values.len());
        let block_values = &values[start..end];

        let amax = block_values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let scale_bits = f32_to_f16_bits(scale);
        bytes.push((scale_bits & 0xff) as u8);
        bytes.push((scale_bits >> 8) as u8);

        for i in 0..32 {
            if i < block_values.len() {
                let q = (block_values[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                bytes.push(q as u8);
            } else {
                bytes.push(0u8);
            }
        }
    }

    bytes
}

/// Convert f16 bits back to f32 (CPU reference).
fn f16_bits_to_f32_cpu(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;

    if exp == 0 {
        if frac == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        let f = frac as f32 / 1024.0;
        let v = f * 6.103515625e-05;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        if frac != 0 {
            return f32::NAN;
        }
        return if sign != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    let f32_exp = (exp as u32).wrapping_sub(15).wrapping_add(127);
    let f32_frac = (frac as u32) << 13;
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_frac;
    f32::from_bits(f32_bits)
}

/// CPU Q8_0 matrix-vector multiply reference.
fn cpu_matvec_q8_0(weight_bytes: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let blocks_per_row = (in_dim + 31) / 32;
    let row_bytes = blocks_per_row * 34;

    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_start = row * row_bytes;
        let mut sum = 0.0f32;

        for b in 0..blocks_per_row {
            let block_start = row_start + b * 34;
            let scale_bits = (weight_bytes[block_start] as u16)
                | ((weight_bytes[block_start + 1] as u16) << 8);
            let scale = f16_bits_to_f32_cpu(scale_bits);

            let base_idx = b * 32;
            let elems = 32.min(in_dim - base_idx);

            let mut block_sum = 0.0f32;
            for j in 0..elems {
                let q = weight_bytes[block_start + 2 + j] as i8;
                block_sum += (q as f32) * x[base_idx + j];
            }
            sum += scale * block_sum;
        }

        out[row] = sum;
    }
    out
}

/// Simple LCG pseudo-random number generator for deterministic test data.
fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

// ---------------------------------------------------------------------------
// Correctness tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q8_0_v3_small() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_q8_0_v3.cu");
    let module = ctx.load_module(ptx).expect("Failed to load matvec_q8_0_v3 module");
    let func = module
        .load_function("matvec_q8_0_v3")
        .expect("Failed to load matvec_q8_0_v3");

    let out_dim = 4usize;
    let in_dim = 32usize;

    let weight_f32: Vec<Vec<f32>> = vec![
        (0..32).map(|i| (i as f32 + 1.0) * 0.25).collect(),
        (0..32).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect(),
        {
            let mut v = vec![0.0f32; 32];
            v[0] = 1.0;
            v
        },
        (0..32).map(|i| (i as f32 + 1.0) * 0.025).collect(),
    ];
    let x = vec![1.0f32; 32];

    let mut weight_bytes = Vec::new();
    for row in &weight_f32 {
        weight_bytes.extend_from_slice(&encode_q8_0(row));
    }

    let expected = cpu_matvec_q8_0(&weight_bytes, &x, out_dim, in_dim);

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .expect("matvec_q8_0_v3 launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(
            (result[i] - expected[i]).abs() < 0.5,
            "matvec_q8_0_v3_small[{i}]: GPU {}, CPU {}, diff {}",
            result[i],
            expected[i],
            (result[i] - expected[i]).abs()
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_v3_realistic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v3").unwrap();

    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let mut rng_state = 42u64;

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();

    let expected = cpu_matvec_q8_0(&weight_bytes, &x, out_dim, in_dim);

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    let tolerance = 1.0;
    let mut max_err = 0.0f32;
    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < tolerance,
            "matvec_q8_0_v3_realistic[{i}]: GPU {}, CPU {}, err {} > tol {}",
            result[i],
            expected[i],
            err,
            tolerance
        );
    }
    eprintln!(
        "matvec_q8_0_v3 4096x4096: max error = {:.6} (tol {})",
        max_err, tolerance
    );
}

#[test]
fn test_cuda_matvec_q8_0_v3_residual() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v3_residual").unwrap();

    let out_dim = 4usize;
    let in_dim = 64usize;

    let mut rng_state = 123u64;

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();

    let matvec_result = cpu_matvec_q8_0(&weight_bytes, &x, out_dim, in_dim);
    let expected: Vec<f32> = matvec_result
        .iter()
        .zip(residual.iter())
        .map(|(&m, &r)| m + r)
        .collect();

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let residual_gpu = stream.clone_htod(&residual).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&residual_gpu)
            .arg(&mut out_gpu)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .expect("matvec_q8_0_v3_residual launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.5,
            "matvec_q8_0_v3_residual[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_v3_zero_weights() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v3").unwrap();

    let out_dim = 2usize;
    let in_dim = 32usize;

    let weight_f32 = vec![0.0f32; in_dim];
    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        weight_bytes.extend_from_slice(&encode_q8_0(&weight_f32));
    }

    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) + 1.0).collect();

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(
            result[i].abs() < 1e-6,
            "matvec_q8_0_v3_zeros[{i}]: expected ~0, got {}",
            result[i]
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_v3_odd_out_dim() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v3").unwrap();

    let out_dim = 5usize;
    let in_dim = 128usize;

    let mut rng_state = 999u64;

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
    let expected = cpu_matvec_q8_0(&weight_bytes, &x, out_dim, in_dim);

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    let tolerance = 0.5;
    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < tolerance,
            "matvec_q8_0_v3_odd[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

// ---------------------------------------------------------------------------
// Differential test: v1 vs v3 must produce identical output
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q8_0_v1_v3_match() {
    let (ctx, stream) = create_context();

    let src_v1 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx_v1 = compile_ptx(src_v1).expect("NVRTC compile failed v1");
    let module_v1 = ctx.load_module(ptx_v1).unwrap();
    let func_v1 = module_v1.load_function("matvec_q8_0").unwrap();

    let src_v3 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx_v3 = compile_ptx(src_v3).expect("NVRTC compile failed v3");
    let module_v3 = ctx.load_module(ptx_v3).unwrap();
    let func_v3 = module_v3.load_function("matvec_q8_0_v3").unwrap();

    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let mut rng_state = 42u64;

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng_state)).collect();

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_v1: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_v3: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = q8_launch_config(out_dim as u32);

    unsafe {
        stream
            .launch_builder(&func_v1)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_v1)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    unsafe {
        stream
            .launch_builder(&func_v3)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_v3)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result_v1 = stream.clone_dtoh(&out_v1).unwrap();
    let result_v3 = stream.clone_dtoh(&out_v3).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..out_dim {
        let diff = (result_v1[i] - result_v3[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "v1 vs v3 mismatch at [{i}]: v1={}, v3={}, diff={}",
            result_v1[i],
            result_v3[i],
            diff
        );
    }
    eprintln!("v1 vs v3 max diff: {:.6e} (tol 1e-3)", max_diff);
}

// ---------------------------------------------------------------------------
// Benchmark: v1 vs v3 comparison
// ---------------------------------------------------------------------------

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Run a kernel benchmark loop and report structured results.
fn bench_kernel_loop<F>(
    stream: &cudarc::driver::CudaStream,
    warmup: usize,
    iterations: usize,
    mut launch: F,
) -> (f64, std::time::Duration)
where
    F: FnMut(),
{
    for _ in 0..warmup {
        launch();
    }
    stream.synchronize().unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        launch();
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();

    let mean_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    (mean_us, elapsed)
}

#[test]
#[ignore]
fn bench_kernel_matvec_q8_0_v1_vs_v3() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);

    // Test multiple matrix sizes to understand scaling behavior.
    let sizes: Vec<(usize, usize)> = vec![
        (4096, 4096),   // typical hidden_dim for 7B models
        (11008, 4096),  // FFN gate/up for Llama 7B
        (4096, 11008),  // FFN down for Llama 7B
        (32000, 4096),  // output projection (vocab_size x hidden_dim)
    ];

    let (ctx, stream) = create_context();

    let src_v1 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx_v1 = compile_ptx(src_v1).expect("NVRTC compile v1 failed");
    let module_v1 = ctx.load_module(ptx_v1).unwrap();
    let func_v1 = module_v1.load_function("matvec_q8_0").unwrap();

    let src_v3 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V3_KERNEL_SOURCE;
    let ptx_v3 = compile_ptx(src_v3).expect("NVRTC compile v3 failed");
    let module_v3 = ctx.load_module(ptx_v3).unwrap();
    let func_v3 = module_v3.load_function("matvec_q8_0_v3").unwrap();

    eprintln!();
    eprintln!("=== Q8_0 MatVec: v1 vs v3 (vectorized loads) ===");
    eprintln!("  Iterations per size: {iterations}");
    eprintln!();
    eprintln!(
        "  {:>12} {:>12}  {:>10} {:>10}  {:>8} {:>8}  {:>7}",
        "Size", "Weight MB", "v1 us", "v3 us", "v1 GB/s", "v3 GB/s", "Speedup"
    );
    eprintln!("  {}", "-".repeat(82));

    for &(out_dim, in_dim) in &sizes {
        let blocks_per_row = (in_dim + 31) / 32;
        let row_bytes = blocks_per_row * 34;
        let total_bytes = out_dim * row_bytes;

        // Generate Q8_0 weight data.
        let mut weight_bytes: Vec<u8> = Vec::with_capacity(total_bytes);
        let mut rng_state = 42u64;
        for _ in 0..out_dim {
            for _ in 0..blocks_per_row {
                weight_bytes.push(0x1E); // f16 scale ~ 0.01
                weight_bytes.push(0x21);
                for _ in 0..32 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let q = ((rng_state >> 33) as i32 % 127) as i8;
                    weight_bytes.push(q as u8);
                }
            }
        }

        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

        let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
        let w_gpu = stream.clone_htod(&weight_i8).unwrap();
        let x_gpu = stream.clone_htod(&x).unwrap();
        let mut out_v1: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
        let mut out_v3: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

        let cfg = q8_launch_config(out_dim as u32);

        // Benchmark v1
        let (v1_us, v1_elapsed) = bench_kernel_loop(
            &stream, 100, iterations,
            || {
                unsafe {
                    stream
                        .launch_builder(&func_v1)
                        .arg(&w_gpu)
                        .arg(&x_gpu)
                        .arg(&mut out_v1)
                        .arg(&(out_dim as u32))
                        .arg(&(in_dim as u32))
                        .launch(cfg)
                }
                .unwrap();
            },
        );

        // Benchmark v3
        let (v3_us, v3_elapsed) = bench_kernel_loop(
            &stream, 100, iterations,
            || {
                unsafe {
                    stream
                        .launch_builder(&func_v3)
                        .arg(&w_gpu)
                        .arg(&x_gpu)
                        .arg(&mut out_v3)
                        .arg(&(out_dim as u32))
                        .arg(&(in_dim as u32))
                        .launch(cfg)
                }
                .unwrap();
            },
        );

        // Memory traffic: read Q8_0 weight + read x (f32) + write out (f32)
        let bytes_per_op = total_bytes + in_dim * 4 + out_dim * 4;
        let v1_bw = (bytes_per_op as f64 * iterations as f64) / v1_elapsed.as_secs_f64() / 1e9;
        let v3_bw = (bytes_per_op as f64 * iterations as f64) / v3_elapsed.as_secs_f64() / 1e9;
        let speedup = v1_us / v3_us;
        let weight_mb = total_bytes as f64 / (1024.0 * 1024.0);

        eprintln!(
            "  {:>5}x{:<5} {:>9.1} MB  {:>10.3} {:>10.3}  {:>7.1} {:>7.1}  {:>6.2}x",
            out_dim, in_dim, weight_mb, v1_us, v3_us, v1_bw, v3_bw, speedup
        );

        // Verify correctness: v1 and v3 must match.
        stream.synchronize().unwrap();
        let result_v1 = stream.clone_dtoh(&out_v1).unwrap();
        let result_v3 = stream.clone_dtoh(&out_v3).unwrap();
        let mut max_diff = 0.0f32;
        for i in 0..out_dim {
            let diff = (result_v1[i] - result_v3[i]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-3,
            "v1 vs v3 mismatch for {}x{}: max_diff={}",
            out_dim, in_dim, max_diff
        );
    }

    eprintln!();

    // Also compute theoretical bandwidth limit for the roofline analysis.
    // A10G: 600 GB/s HBM bandwidth, 31.2 TFLOPS FP32.
    eprintln!("  Roofline reference (A10G):");
    eprintln!("    Peak HBM bandwidth: 600 GB/s");
    eprintln!("    Peak FP32 compute:  31.2 TFLOPS");
    eprintln!("    Q8_0 arithmetic intensity: ~2 FLOP/byte (bandwidth-bound)");
    eprintln!();
    eprintln!("  If v3 bandwidth approaches 500+ GB/s, the kernel is near-optimal");
    eprintln!("  for memory bandwidth. Further improvement requires algorithmic");
    eprintln!("  changes (e.g., fused operations to reduce memory passes).");
}

/// Roofline analysis: measure time vs theoretical bandwidth and compute limits.
///
/// For each matrix size, computes:
/// - Measured time
/// - Time at theoretical bandwidth limit (all bytes at peak BW)
/// - Time at theoretical compute limit (all FLOPs at peak FP32)
/// - Whether the kernel is bandwidth-bound or compute-bound
#[test]
#[ignore]
fn bench_kernel_q8_0_roofline() {
    let iterations = env_usize("LUMEN_BENCH_ITERATIONS", 10000);

    let sizes: Vec<(usize, usize)> = vec![
        (4096, 4096),
        (11008, 4096),
        (4096, 11008),
    ];

    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    eprintln!();
    eprintln!("=== Q8_0 MatVec Roofline Analysis ===");
    eprintln!();
    eprintln!(
        "  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Size", "Measured", "BW limit", "Compute", "BW util", "Bottleneck"
    );
    eprintln!(
        "  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "", "(us)", "(us)", "(us)", "(%)", ""
    );
    eprintln!("  {}", "-".repeat(72));

    // A10G theoretical limits (adjust for your GPU).
    let peak_bw_gb_s = 600.0;    // GB/s HBM bandwidth
    let peak_flops = 31.2e12;     // FP32 peak FLOPS

    for &(out_dim, in_dim) in &sizes {
        let blocks_per_row = (in_dim + 31) / 32;
        let row_bytes = blocks_per_row * 34;
        let total_weight_bytes = out_dim * row_bytes;

        let mut weight_bytes: Vec<u8> = Vec::with_capacity(total_weight_bytes);
        let mut rng_state = 42u64;
        for _ in 0..out_dim {
            for _ in 0..blocks_per_row {
                weight_bytes.push(0x1E);
                weight_bytes.push(0x21);
                for _ in 0..32 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let q = ((rng_state >> 33) as i32 % 127) as i8;
                    weight_bytes.push(q as u8);
                }
            }
        }

        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

        let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
        let w_gpu = stream.clone_htod(&weight_i8).unwrap();
        let x_gpu = stream.clone_htod(&x).unwrap();
        let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

        let cfg = q8_launch_config(out_dim as u32);

        let (measured_us, _) = bench_kernel_loop(
            &stream, 100, iterations,
            || {
                unsafe {
                    stream
                        .launch_builder(&func)
                        .arg(&w_gpu)
                        .arg(&x_gpu)
                        .arg(&mut out_gpu)
                        .arg(&(out_dim as u32))
                        .arg(&(in_dim as u32))
                        .launch(cfg)
                }
                .unwrap();
            },
        );

        // Bandwidth limit: time to transfer all data at peak bandwidth.
        let total_bytes = total_weight_bytes + in_dim * 4 + out_dim * 4;
        let bw_limit_us = (total_bytes as f64 / (peak_bw_gb_s * 1e3)) * 1e6;

        // Compute limit: time to execute all FLOPs at peak throughput.
        let flops = 2.0 * out_dim as f64 * in_dim as f64;
        let compute_limit_us = (flops / peak_flops) * 1e6;

        let bw_utilization = (bw_limit_us / measured_us) * 100.0;
        let bottleneck = if bw_limit_us > compute_limit_us {
            "Bandwidth"
        } else {
            "Compute"
        };

        eprintln!(
            "  {:>5}x{:<5}  {:>10.3}  {:>10.3}  {:>10.3}  {:>9.1}%  {:>12}",
            out_dim, in_dim, measured_us, bw_limit_us, compute_limit_us,
            bw_utilization, bottleneck
        );
    }

    eprintln!();
    eprintln!("  BW util% = (theoretical min time at peak BW) / (measured time) * 100");
    eprintln!("  Higher = better. 100% = fully utilizing memory bandwidth.");
    eprintln!("  Q8_0 matvec is bandwidth-bound: weight read dominates.");
}
