//! Differential test for the scale-halfword `matvec_q8_aligned_q8_1_hw[_residual]`
//! kernel variant against the production `matvec_q8_aligned_q8_1[_residual]`.
//!
//! Both kernels are compiled via NVRTC, run on the same Q8Aligned weight and
//! Q8_1 input data, and their F32 outputs are compared element-by-element.
//! The only structural difference is the scale-load pattern (halfword cast vs
//! byte-OR-shift); the outputs must match within a tight tolerance (1e-3) at
//! realistic matrix dimensions.
//!
//!   cargo test --release -p lumen-runtime --features cuda \
//!     --test cuda_matvec_q8_aligned_q8_1_hw_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// Kernel constants must match the .cu defines.
const NR: u32 = 2;
const BLOCK_DIM: u32 = 128;
const Q8_BLOCK_SIZE: usize = 32;
const Q8_ALIGNED_BYTES: usize = 36;
const Q8_1_BYTES: usize = 36;

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

fn launch_config(out_dim: u32) -> LaunchConfig {
    let grid = (out_dim + NR - 1) / NR;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

// IEEE 754 f32 -> f16 (mirrors the helper used in the other CUDA tests).
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

// Encode a contiguous F32 slice into Q8Aligned (36-byte) blocks.
// Layout: [f16 scale][2 byte pad][32 int8 quants].
fn encode_q8_aligned(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % Q8_BLOCK_SIZE == 0, "in_dim must be multiple of 32");
    let block_count = values.len() / Q8_BLOCK_SIZE;
    let mut bytes = Vec::with_capacity(block_count * Q8_ALIGNED_BYTES);
    for block_idx in 0..block_count {
        let start = block_idx * Q8_BLOCK_SIZE;
        let block_values = &values[start..start + Q8_BLOCK_SIZE];
        let amax = block_values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let scale_bits = f32_to_f16_bits(scale);
        bytes.push((scale_bits & 0xff) as u8);
        bytes.push((scale_bits >> 8) as u8);
        bytes.push(0);
        bytes.push(0);
        for &v in block_values {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

// Encode a contiguous F32 slice into Q8_1 (36-byte) blocks.
// Layout: [f16 scale][f16 weighted sum][32 int8 quants].
fn encode_q8_1(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % Q8_BLOCK_SIZE == 0, "in_dim must be multiple of 32");
    let block_count = values.len() / Q8_BLOCK_SIZE;
    let mut bytes = Vec::with_capacity(block_count * Q8_1_BYTES);
    for block_idx in 0..block_count {
        let start = block_idx * Q8_BLOCK_SIZE;
        let block_values = &values[start..start + Q8_BLOCK_SIZE];
        let amax = block_values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut quants = [0i8; 32];
        let mut qsum: i32 = 0;
        for i in 0..32 {
            let q = (block_values[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            quants[i] = q;
            qsum += q as i32;
        }
        let weighted_sum = scale * (qsum as f32);
        let scale_bits = f32_to_f16_bits(scale);
        let sum_bits = f32_to_f16_bits(weighted_sum);
        bytes.push((scale_bits & 0xff) as u8);
        bytes.push((scale_bits >> 8) as u8);
        bytes.push((sum_bits & 0xff) as u8);
        bytes.push((sum_bits >> 8) as u8);
        for &q in quants.iter() {
            bytes.push(q as u8);
        }
    }
    bytes
}

fn deterministic_random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        out.push(v);
    }
    out
}

/// Run one of the two kernel variants and return F32 output.
unsafe fn run_kernel(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    func: &cudarc::driver::CudaFunction,
    weight_gpu: &CudaSlice<i8>,
    x_gpu: &CudaSlice<i8>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    stream
        .launch_builder(func)
        .arg(weight_gpu)
        .arg(x_gpu)
        .arg(&mut out_gpu)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(launch_config(out_dim_u32))
        .expect("kernel launch failed");
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

/// Run the residual variant and return F32 output.
unsafe fn run_kernel_residual(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    func: &cudarc::driver::CudaFunction,
    weight_gpu: &CudaSlice<i8>,
    x_gpu: &CudaSlice<i8>,
    residual_gpu: &CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    stream
        .launch_builder(func)
        .arg(weight_gpu)
        .arg(x_gpu)
        .arg(residual_gpu)
        .arg(&mut out_gpu)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(launch_config(out_dim_u32))
        .expect("kernel launch failed");
    stream.synchronize().unwrap();
    stream.clone_dtoh(&out_gpu).unwrap()
}

#[test]
fn test_q8_aligned_q8_1_hw_matches_reference() {
    let (ctx, stream) = create_context();

    let src_ref = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE;
    let src_hw = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_Q8_1_HW_KERNEL_SOURCE;
    let ptx_ref = compile_ptx(src_ref).expect("NVRTC compile failed for reference kernel");
    let ptx_hw = compile_ptx(src_hw).expect("NVRTC compile failed for hw variant");
    let mod_ref = ctx.load_module(ptx_ref).expect("load reference module");
    let mod_hw = ctx.load_module(ptx_hw).expect("load hw module");
    let f_ref = mod_ref
        .load_function("matvec_q8_aligned_q8_1")
        .expect("load matvec_q8_aligned_q8_1");
    let f_hw = mod_hw
        .load_function("matvec_q8_aligned_q8_1_hw")
        .expect("load matvec_q8_aligned_q8_1_hw");

    // Representative GEMV dimensions for a 7-9B model decode step.
    let out_dim = 4096usize;
    let in_dim = 4096usize;

    // Generate deterministic weight rows and a single x vector.
    let mut weight_bytes = Vec::new();
    for row in 0..out_dim {
        let row_f32 = deterministic_random_vec(in_dim, 0x9E37_79B9_u64.wrapping_add(row as u64));
        weight_bytes.extend_from_slice(&encode_q8_aligned(&row_f32));
    }
    let x_f32 = deterministic_random_vec(in_dim, 0xC0FF_EE_u64);
    let x_q8_1 = encode_q8_1(&x_f32);

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let x_i8: Vec<i8> = x_q8_1.iter().map(|&b| b as i8).collect();

    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_i8).unwrap();

    let out_ref =
        unsafe { run_kernel(&stream, &f_ref, &weight_gpu, &x_gpu, out_dim, in_dim) };
    let out_hw =
        unsafe { run_kernel(&stream, &f_hw, &weight_gpu, &x_gpu, out_dim, in_dim) };

    // Both kernels do the same math; their outputs must be bit-identical or
    // very close (NVCC scheduling can reorder FP additions in the reduction
    // tree, so we use a small relative-tolerance band).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..out_dim {
        let abs = (out_ref[i] - out_hw[i]).abs();
        let rel = abs / out_ref[i].abs().max(1e-6);
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_abs < 1e-3 && max_rel < 1e-4,
        "hw variant diverges: max_abs={max_abs}, max_rel={max_rel}",
    );
}

#[test]
fn test_q8_aligned_q8_1_hw_residual_matches_reference() {
    let (ctx, stream) = create_context();

    let src_ref = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE;
    let src_hw = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_Q8_1_HW_KERNEL_SOURCE;
    let ptx_ref = compile_ptx(src_ref).expect("NVRTC compile failed for reference kernel");
    let ptx_hw = compile_ptx(src_hw).expect("NVRTC compile failed for hw variant");
    let mod_ref = ctx.load_module(ptx_ref).expect("load reference module");
    let mod_hw = ctx.load_module(ptx_hw).expect("load hw module");
    let f_ref = mod_ref
        .load_function("matvec_q8_aligned_q8_1_residual")
        .expect("load reference residual fn");
    let f_hw = mod_hw
        .load_function("matvec_q8_aligned_q8_1_hw_residual")
        .expect("load hw residual fn");

    let out_dim = 2048usize;
    let in_dim = 2048usize;

    let mut weight_bytes = Vec::new();
    for row in 0..out_dim {
        let row_f32 = deterministic_random_vec(in_dim, 0xDEAD_BEEF_u64.wrapping_add(row as u64));
        weight_bytes.extend_from_slice(&encode_q8_aligned(&row_f32));
    }
    let x_f32 = deterministic_random_vec(in_dim, 0x1234_5678_u64);
    let x_q8_1 = encode_q8_1(&x_f32);
    let residual: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 1e-3).collect();

    let weight_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let x_i8: Vec<i8> = x_q8_1.iter().map(|&b| b as i8).collect();

    let weight_gpu = stream.clone_htod(&weight_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_i8).unwrap();
    let residual_gpu = stream.clone_htod(&residual).unwrap();

    let out_ref = unsafe {
        run_kernel_residual(
            &stream,
            &f_ref,
            &weight_gpu,
            &x_gpu,
            &residual_gpu,
            out_dim,
            in_dim,
        )
    };
    let out_hw = unsafe {
        run_kernel_residual(
            &stream,
            &f_hw,
            &weight_gpu,
            &x_gpu,
            &residual_gpu,
            out_dim,
            in_dim,
        )
    };

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..out_dim {
        let abs = (out_ref[i] - out_hw[i]).abs();
        let rel = abs / out_ref[i].abs().max(1e-6);
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_abs < 1e-3 && max_rel < 1e-4,
        "hw residual variant diverges: max_abs={max_abs}, max_rel={max_rel}",
    );
}
