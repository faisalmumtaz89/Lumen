//! Integration tests for the Q4_0 quantized matrix-vector multiply CUDA kernel.
//!
//! Tests `matvec_q4_0` and `matvec_q4_0_residual` against a CPU reference
//! implementation of Q4_0 dequantization + dot product.
//!
//! Requires a CUDA-capable GPU (run on Modal):
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_matvec_q4_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// Q4_0 block layout constants (GGML standard).
const Q4_0_BLOCK_BYTES: usize = 18; // 2 bytes f16 scale + 16 bytes packed nibbles
const Q4_0_GROUP_SIZE: usize = 32; // elements per block

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

// ---------------------------------------------------------------------------
// CPU reference: Q4_0 quantization + dequantization
// ---------------------------------------------------------------------------

/// Convert f32 to IEEE 754 half-precision bits (truncation, not rounding).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp == 0xff {
        // Inf/NaN
        let f16_frac = if frac != 0 { 0x200 } else { 0 };
        return ((sign << 15) | 0x7c00 | f16_frac) as u16;
    }
    if exp == 0 {
        // Zero/subnormal f32 -> zero in f16
        return (sign << 15) as u16;
    }

    let f16_exp = exp - 127 + 15;
    if f16_exp >= 31 {
        // Overflow -> Inf
        return ((sign << 15) | 0x7c00) as u16;
    }
    if f16_exp <= 0 {
        // Underflow -> zero
        return (sign << 15) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((f16_exp as u32) << 10) | f16_frac) as u16
}

/// Quantize a row of f32 values into Q4_0 blocks.
///
/// The row length must be a multiple of 32. Returns the packed byte
/// representation matching the GGML Q4_0 format.
fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len() % Q4_0_GROUP_SIZE == 0,
        "Q4_0 requires input length to be a multiple of 32"
    );

    let num_blocks = values.len() / Q4_0_GROUP_SIZE;
    let mut data = Vec::with_capacity(num_blocks * Q4_0_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let base = block_idx * Q4_0_GROUP_SIZE;
        let group = &values[base..base + Q4_0_GROUP_SIZE];

        // Find the absmax to compute the scale.
        let amax = group
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        // Scale: maps [-amax, amax] to [-8, 7] (approximately).
        // Q4_0 stores unsigned nibbles 0..15, centered at 8.
        // dequant: val = scale * (nibble - 8)
        // So quant: nibble = clamp(round(val / scale) + 8, 0, 15)
        let scale = if amax == 0.0 { 0.0 } else { amax / 8.0 };

        // Write f16 scale (little-endian).
        let scale_bits = f32_to_f16_bits(scale);
        data.push((scale_bits & 0xff) as u8);
        data.push((scale_bits >> 8) as u8);

        // Quantize 32 values into 16 nibble-pair bytes.
        for i in 0..16 {
            let v0 = group[2 * i];
            let v1 = group[2 * i + 1];

            let q0 = if scale == 0.0 {
                8u8
            } else {
                ((v0 / scale + 8.0).round() as i32).clamp(0, 15) as u8
            };
            let q1 = if scale == 0.0 {
                8u8
            } else {
                ((v1 / scale + 8.0).round() as i32).clamp(0, 15) as u8
            };

            // Pack: low nibble = element 2*i, high nibble = element 2*i+1.
            data.push(q0 | (q1 << 4));
        }
    }

    data
}

/// f16 bits to f32 (CPU reference, matching the CUDA device function).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let f = frac as f32 / 1024.0;
        let v = f * 6.103515625e-05;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        let f32_bits = (sign << 31) | 0x7f800000 | if frac != 0 { 0x400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }

    let f32_exp = exp + 127 - 15;
    let f32_frac = frac << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
}

/// CPU reference: Q4_0 dequantize + matrix-vector multiply.
///
/// `weight_q4` is the packed Q4_0 data for a [out_dim, in_dim] weight matrix.
/// `x` is the input vector of length in_dim.
/// Returns the output vector of length out_dim.
fn cpu_matvec_q4_0(weight_q4: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let num_blocks_per_row = in_dim / Q4_0_GROUP_SIZE;
    let row_bytes = num_blocks_per_row * Q4_0_BLOCK_BYTES;

    let mut out = vec![0.0f32; out_dim];

    for row in 0..out_dim {
        let row_start = row * row_bytes;
        let mut sum = 0.0f32;

        for b in 0..num_blocks_per_row {
            let block_start = row_start + b * Q4_0_BLOCK_BYTES;

            // Read f16 scale.
            let scale_bits = weight_q4[block_start] as u16
                | ((weight_q4[block_start + 1] as u16) << 8);
            let scale = f16_to_f32(scale_bits);

            let x_base = b * Q4_0_GROUP_SIZE;

            // Dequantize 32 nibbles and dot with x.
            for i in 0..16 {
                let byte_val = weight_q4[block_start + 2 + i];
                let nibble_lo = byte_val & 0x0f;
                let nibble_hi = (byte_val >> 4) & 0x0f;

                let dq_lo = scale * (nibble_lo as f32 - 8.0);
                let dq_hi = scale * (nibble_hi as f32 - 8.0);

                sum += dq_lo * x[x_base + 2 * i] + dq_hi * x[x_base + 2 * i + 1];
            }
        }

        out[row] = sum;
    }

    out
}

// ---------------------------------------------------------------------------
// Helper: load and compile the Q4_0 kernel module
// ---------------------------------------------------------------------------

fn compile_q4_module(
    ctx: &Arc<CudaContext>,
) -> (Arc<cudarc::driver::CudaModule>, cudarc::driver::CudaFunction, cudarc::driver::CudaFunction)
{
    let src = lumen_runtime::cuda::shaders::MATVEC_Q4_0_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_q4_0.cu");
    let module = ctx.load_module(ptx).expect("Failed to load Q4_0 module");
    let matvec = module
        .load_function("matvec_q4_0")
        .expect("Failed to load matvec_q4_0");
    let matvec_res = module
        .load_function("matvec_q4_0_residual")
        .expect("Failed to load matvec_q4_0_residual");
    (module, matvec, matvec_res)
}

// ---------------------------------------------------------------------------
// Test 1: Small matrix — verify basic correctness
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q4_0_small() {
    let (ctx, stream) = create_context();
    let (_module, func, _func_res) = compile_q4_module(&ctx);

    // 4x32 weight matrix (out_dim=4, in_dim=32 — minimum one Q4_0 block per row).
    let out_dim = 4usize;
    let in_dim = 32usize;

    // Generate predictable weights: row i has all values = (i+1)*0.5.
    let mut weight_f32 = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            weight_f32[row * in_dim + col] = (row as f32 + 1.0) * 0.5;
        }
    }

    // Quantize each row independently.
    let mut weight_q4 = Vec::new();
    for row in 0..out_dim {
        let row_data = &weight_f32[row * in_dim..(row + 1) * in_dim];
        weight_q4.extend_from_slice(&quantize_q4_0(row_data));
    }

    // Input vector: x[j] = j * 0.1.
    let x: Vec<f32> = (0..in_dim).map(|j| j as f32 * 0.1).collect();

    // CPU reference.
    let expected = cpu_matvec_q4_0(&weight_q4, &x, out_dim, in_dim);

    // Upload to GPU.
    let weight_gpu = stream.clone_htod(&weight_q4).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .expect("matvec_q4_0 launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(
            (result[i] - expected[i]).abs() < 1e-3,
            "matvec_q4_0_small[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Realistic 4096x4096 dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q4_0_realistic() {
    let (ctx, stream) = create_context();
    let (_module, func, _func_res) = compile_q4_module(&ctx);

    let out_dim = 4096usize;
    let in_dim = 4096usize;

    // Generate weights with a simple pattern for reproducibility.
    // Use a deterministic PRNG-like pattern: val = sin(row * 37 + col * 13) * 0.3.
    let mut weight_f32 = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            weight_f32[row * in_dim + col] =
                ((row as f32 * 37.0 + col as f32 * 13.0) * 0.01).sin() * 0.3;
        }
    }

    // Quantize.
    let mut weight_q4 = Vec::with_capacity(out_dim * (in_dim / Q4_0_GROUP_SIZE) * Q4_0_BLOCK_BYTES);
    for row in 0..out_dim {
        let row_data = &weight_f32[row * in_dim..(row + 1) * in_dim];
        weight_q4.extend_from_slice(&quantize_q4_0(row_data));
    }

    // Input vector.
    let x: Vec<f32> = (0..in_dim).map(|j| (j as f32 * 0.001).cos()).collect();

    // CPU reference.
    let expected = cpu_matvec_q4_0(&weight_q4, &x, out_dim, in_dim);

    // Upload to GPU.
    let weight_gpu = stream.clone_htod(&weight_q4).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .expect("matvec_q4_0 launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    // With quantization noise + f16 scale + f32 accumulation across 4096 elements,
    // allow slightly wider tolerance than for F32 kernels.
    let mut max_err = 0.0f32;
    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 0.05,
            "matvec_q4_0_realistic[{i}]: GPU {}, CPU {}, err {err}",
            result[i],
            expected[i]
        );
    }
    println!("matvec_q4_0 4096x4096: max error = {max_err:.6}");
}

// ---------------------------------------------------------------------------
// Test 3: Fused residual variant
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q4_0_residual() {
    let (ctx, stream) = create_context();
    let (_module, _func, func_res) = compile_q4_module(&ctx);

    let out_dim = 128usize;
    let in_dim = 256usize;

    // Generate weights.
    let mut weight_f32 = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            weight_f32[row * in_dim + col] = ((row + col) as f32 * 0.007).sin() * 0.5;
        }
    }

    let mut weight_q4 = Vec::new();
    for row in 0..out_dim {
        let row_data = &weight_f32[row * in_dim..(row + 1) * in_dim];
        weight_q4.extend_from_slice(&quantize_q4_0(row_data));
    }

    let x: Vec<f32> = (0..in_dim).map(|j| (j as f32) * 0.01 - 1.0).collect();
    let residual: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.1).collect();

    // CPU reference: matvec + residual.
    let matvec_out = cpu_matvec_q4_0(&weight_q4, &x, out_dim, in_dim);
    let expected: Vec<f32> = matvec_out
        .iter()
        .zip(residual.iter())
        .map(|(&m, &r)| m + r)
        .collect();

    // Upload to GPU.
    let weight_gpu = stream.clone_htod(&weight_q4).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let residual_gpu = stream.clone_htod(&residual).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    unsafe {
        stream
            .launch_builder(&func_res)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&residual_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .expect("matvec_q4_0_residual launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.05,
            "matvec_q4_0_residual[{i}]: GPU {}, CPU {}, err {err}",
            result[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Zero weights — verify no NaN/inf
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q4_0_zero_weights() {
    let (ctx, stream) = create_context();
    let (_module, func, _func_res) = compile_q4_module(&ctx);

    let out_dim = 4usize;
    let in_dim = 64usize;

    // All-zero weights -> all-zero Q4_0 blocks (scale=0, nibbles=8).
    let weight_f32 = vec![0.0f32; out_dim * in_dim];
    let mut weight_q4 = Vec::new();
    for row in 0..out_dim {
        let row_data = &weight_f32[row * in_dim..(row + 1) * in_dim];
        weight_q4.extend_from_slice(&quantize_q4_0(row_data));
    }

    let x: Vec<f32> = (0..in_dim).map(|j| j as f32).collect();

    let weight_gpu = stream.clone_htod(&weight_q4).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .expect("matvec_q4_0 zero launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        assert!(result[i].is_finite(), "output[{i}] should be finite, got {}", result[i]);
        assert!(
            result[i].abs() < 1e-6,
            "zero weights should produce near-zero output, got {}",
            result[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: Multiple Q4_0 blocks per row — verify block boundary handling
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_matvec_q4_0_multi_block() {
    let (ctx, stream) = create_context();
    let (_module, func, _func_res) = compile_q4_module(&ctx);

    // 8 rows, 128 columns = 4 Q4_0 blocks per row.
    let out_dim = 8usize;
    let in_dim = 128usize;

    let mut weight_f32 = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for col in 0..in_dim {
            // Place different magnitudes in each block to test boundary handling.
            let block = col / Q4_0_GROUP_SIZE;
            weight_f32[row * in_dim + col] = ((row + 1) * (block + 1)) as f32 * 0.1;
        }
    }

    let mut weight_q4 = Vec::new();
    for row in 0..out_dim {
        let row_data = &weight_f32[row * in_dim..(row + 1) * in_dim];
        weight_q4.extend_from_slice(&quantize_q4_0(row_data));
    }

    let x = vec![1.0f32; in_dim];

    let expected = cpu_matvec_q4_0(&weight_q4, &x, out_dim, in_dim);

    let weight_gpu = stream.clone_htod(&weight_q4).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&weight_gpu)
            .arg(&x_gpu)
            .arg(&mut out_gpu)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(cfg)
    }
    .expect("matvec_q4_0 multi-block launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..out_dim {
        let err = (result[i] - expected[i]).abs();
        assert!(
            err < 0.05,
            "matvec_q4_0_multi_block[{i}]: GPU {}, CPU {}, err {err}",
            result[i],
            expected[i]
        );
    }
}
