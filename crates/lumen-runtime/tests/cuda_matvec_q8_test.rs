//! Integration tests for Q8_0 quantized matrix-vector multiply CUDA kernels.
//!
//! Tests `matvec_q8_0` and `matvec_q8_0_residual` against CPU reference
//! implementations. Requires a CUDA-capable GPU (run on Modal).
//!
//! The kernel uses multi-row deferred-reduction (NR=2, 128 threads per block).
//! Launch config: grid = ceil(out_dim/2), block = 128.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_matvec_q8_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

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
        // Inf or NaN
        let f16_frac = if frac != 0 { 0x200u32 } else { 0 };
        return ((sign << 15) | (0x1f << 10) | f16_frac) as u16;
    }

    // Re-bias exponent from f32 bias (127) to f16 bias (15).
    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow to infinity.
        return ((sign << 15) | (0x1f << 10)) as u16;
    }
    if new_exp <= 0 {
        // Subnormal or zero.
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        // Denormalized: shift mantissa.
        let m = frac | 0x800000;
        let shift = 1 - new_exp;
        let f16_frac = m >> (13 + shift);
        return ((sign << 15) | f16_frac) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Encode a row of f32 values into Q8_0 blocks.
///
/// Returns the raw byte representation: for each block of 32 values,
/// 2 bytes f16 scale (LE) + 32 bytes int8 quants.
fn encode_q8_0(values: &[f32]) -> Vec<u8> {
    let block_count = (values.len() + 31) / 32;
    let mut bytes = Vec::with_capacity(block_count * 34);

    for block_idx in 0..block_count {
        let start = block_idx * 32;
        let end = (start + 32).min(values.len());
        let block_values = &values[start..end];

        // Find the absolute max to compute scale.
        let amax = block_values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write f16 scale (little-endian).
        let scale_bits = f32_to_f16_bits(scale);
        bytes.push((scale_bits & 0xff) as u8);
        bytes.push((scale_bits >> 8) as u8);

        // Write 32 int8 quants.
        for i in 0..32 {
            if i < block_values.len() {
                let q = (block_values[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                bytes.push(q as u8);
            } else {
                // Padding for tail block.
                bytes.push(0u8);
            }
        }
    }

    bytes
}

/// Convert f16 bits back to f32 (CPU reference, mirrors the CUDA device function).
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

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU Q8_0 matrix-vector multiply: out[row] = sum over blocks of (scale * dot(quants, x_slice)).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q8_0_small() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_q8_0.cu");
    let module = ctx.load_module(ptx).expect("Failed to load matvec_q8_0 module");
    let func = module
        .load_function("matvec_q8_0")
        .expect("Failed to load matvec_q8_0");

    // 4x32 matrix: 4 output rows, 32 input elements (1 Q8_0 block per row).
    // in_dim must be a multiple of 32 for the kernel's nb = in_dim >> 5.
    let out_dim = 4usize;
    let in_dim = 32usize;

    // Create known f32 weight values, then encode to Q8_0.
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

    // Encode all rows into contiguous Q8_0 bytes.
    let mut weight_bytes = Vec::new();
    for row in &weight_f32 {
        weight_bytes.extend_from_slice(&encode_q8_0(row));
    }

    let expected = cpu_matvec_q8_0(&weight_bytes, &x, out_dim, in_dim);

    // Upload to GPU as raw bytes (i8 for cudarc compatibility).
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
    .expect("matvec_q8_0 launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(
            (result[i] - expected[i]).abs() < 0.5,
            "matvec_q8_0_small[{i}]: GPU {}, CPU {}, diff {}",
            result[i],
            expected[i],
            (result[i] - expected[i]).abs()
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_realistic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    // Realistic dimensions: 4096x4096 (typical hidden_dim for 7B models).
    let out_dim = 4096usize;
    let in_dim = 4096usize;

    // Deterministic pseudo-random data using a simple LCG.
    let mut rng_state = 42u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Map to [-1.0, 1.0]
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    // Generate weight values and encode to Q8_0.
    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();

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

    // Q8_0 quantization introduces error. For 4096 element dot products with
    // values in [-1, 1], the accumulated quantization error is bounded.
    // Use a tolerance proportional to in_dim: each of 128 blocks contributes
    // up to ~0.01 quantization error per element.
    let tolerance = 1.0;
    let mut max_err = 0.0f32;
    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < tolerance,
            "matvec_q8_0_realistic[{i}]: GPU {}, CPU {}, err {} > tol {}",
            result[i],
            expected[i],
            err,
            tolerance
        );
    }
    eprintln!(
        "matvec_q8_0 4096x4096: max error = {:.6} (tol {})",
        max_err, tolerance
    );
}

#[test]
fn test_cuda_matvec_q8_0_residual() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_residual").unwrap();

    let out_dim = 4usize;
    let in_dim = 64usize;

    // Deterministic pseudo-random data.
    let mut rng_state = 123u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| next_f32(&mut rng_state)).collect();

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
    .expect("matvec_q8_0_residual launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.5,
            "matvec_q8_0_residual[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_aligned_dim() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    // in_dim=64 is 2 Q8_0 blocks per row.
    let out_dim = 2usize;
    let in_dim = 64usize;

    let mut rng_state = 77u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
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

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.5,
            "matvec_q8_0_aligned[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_zero_weights() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    let out_dim = 2usize;
    let in_dim = 32usize;

    // All-zero weights: output should be zero regardless of input.
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
            "matvec_q8_0_zeros[{i}]: expected ~0, got {}",
            result[i]
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_odd_out_dim() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0").unwrap();

    // Odd out_dim tests the NR=2 boundary handling (last block has only 1 valid row).
    let out_dim = 5usize;
    let in_dim = 128usize;

    let mut rng_state = 999u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
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
            "matvec_q8_0_odd[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

// ---------------------------------------------------------------------------
// V2 kernel tests (shared memory x + vectorized loads)
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q8_0_v2_small() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V2_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_q8_0_v2.cu");
    let module = ctx.load_module(ptx).expect("Failed to load matvec_q8_0_v2 module");
    let func = module
        .load_function("matvec_q8_0_v2")
        .expect("Failed to load matvec_q8_0_v2");

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
    .expect("matvec_q8_0_v2 launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(
            (result[i] - expected[i]).abs() < 0.5,
            "matvec_q8_0_v2_small[{i}]: GPU {}, CPU {}, diff {}",
            result[i],
            expected[i],
            (result[i] - expected[i]).abs()
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_v2_realistic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V2_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v2").unwrap();

    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let mut rng_state = 42u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();

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
            "matvec_q8_0_v2_realistic[{i}]: GPU {}, CPU {}, err {} > tol {}",
            result[i],
            expected[i],
            err,
            tolerance
        );
    }
    eprintln!(
        "matvec_q8_0_v2 4096x4096: max error = {:.6} (tol {})",
        max_err, tolerance
    );
}

#[test]
fn test_cuda_matvec_q8_0_v2_residual() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V2_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("matvec_q8_0_v2_residual").unwrap();

    let out_dim = 4usize;
    let in_dim = 64usize;

    let mut rng_state = 123u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
    let residual: Vec<f32> = (0..out_dim).map(|_| next_f32(&mut rng_state)).collect();

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
    .expect("matvec_q8_0_v2_residual launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.5,
            "matvec_q8_0_v2_residual[{i}]: GPU {}, CPU {}, err {}",
            result[i],
            expected[i],
            err
        );
    }
}

#[test]
fn test_cuda_matvec_q8_0_v1_v2_match() {
    // Differential test: v1 and v2 must produce identical output.
    let (ctx, stream) = create_context();

    let src_v1 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_KERNEL_SOURCE;
    let ptx_v1 = compile_ptx(src_v1).expect("NVRTC compile failed v1");
    let module_v1 = ctx.load_module(ptx_v1).unwrap();
    let func_v1 = module_v1.load_function("matvec_q8_0").unwrap();

    let src_v2 = lumen_runtime::cuda::shaders::MATVEC_Q8_0_V2_KERNEL_SOURCE;
    let ptx_v2 = compile_ptx(src_v2).expect("NVRTC compile failed v2");
    let module_v2 = ctx.load_module(ptx_v2).unwrap();
    let func_v2 = module_v2.load_function("matvec_q8_0_v2").unwrap();

    let out_dim = 4096usize;
    let in_dim = 4096usize;

    let mut rng_state = 42u64;
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut weight_bytes = Vec::new();
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();
        weight_bytes.extend_from_slice(&encode_q8_0(&row));
    }

    let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut rng_state)).collect();

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_v1: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_v2: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

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
            .launch_builder(&func_v2)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_v2)
            .arg(&(out_dim as u32))
            .arg(&(in_dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result_v1 = stream.clone_dtoh(&out_v1).unwrap();
    let result_v2 = stream.clone_dtoh(&out_v2).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..out_dim {
        let diff = (result_v1[i] - result_v2[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "v1 vs v2 mismatch at [{i}]: v1={}, v2={}, diff={}",
            result_v1[i],
            result_v2[i],
            diff
        );
    }
    eprintln!("v1 vs v2 max diff: {:.6e} (tol 1e-3)", max_diff);
}
