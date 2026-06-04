//! Differential test for the dp4a-mmvq Q8 split matvec kernel against the
//! production split kernel (matvec_q8_split_q8_1).
//!
//! Both kernels consume the SAME 34-byte-per-block SoA layout. The only
//! difference is the thread→K mapping: the dp4a-mmvq kernel uses the
//! `kbx = tid/(qi/vdr)`, `kqs = vdr*(tid%(qi/vdr))` pattern with
//! blocks_per_iter=32 (4 threads per block, vdr=2), giving K-trip≥4 for the
//! 4096-hidden shapes that dominate Qwen3.5-9B decode. The production kernel
//! uses vdr=8 (1 thread per block) with K-trip=1.
//!
//! Numerical equivalence is exact in the dp4a integer accumulator domain;
//! we tolerate 1e-3 absolute error on the final F32 to cover f16 scale
//! quantization rounding (same tolerance as `cuda_q8_split_decode_test`).
//!
//! Requires a CUDA-capable GPU.
//!
//!   cargo test --release -p lumen-runtime --features cuda \
//!       --test cuda_q8_split_4thread_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// Compile a kernel source with SM80 + use_fast_math (matches Lumen's prod
/// `compile_and_load_with_arch_fast_math`).
fn compile_sm80(source: &str) -> cudarc::nvrtc::Ptx {
    let opts = CompileOptions {
        arch: Some("compute_80"),
        options: vec!["--use_fast_math".to_string()],
        ..Default::default()
    };
    compile_ptx_with_opts(source, opts).expect("compile_ptx_with_opts SM80")
}

const Q8_BLOCK: usize = 32;
const Q8_RAW_BYTES: usize = 34;

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

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

fn encode_q8_raw(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % Q8_BLOCK == 0, "encoder requires whole blocks");
    let n_blocks = values.len() / Q8_BLOCK;
    let mut out = vec![0u8; n_blocks * Q8_RAW_BYTES];
    for b in 0..n_blocks {
        let blk = &values[b * Q8_BLOCK..(b + 1) * Q8_BLOCK];
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let scale_bits = f32_to_f16_bits(scale);
        let block_dst = &mut out[b * Q8_RAW_BYTES..(b + 1) * Q8_RAW_BYTES];
        block_dst[0] = (scale_bits & 0xff) as u8;
        block_dst[1] = (scale_bits >> 8) as u8;
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            block_dst[2 + i] = q as u8;
        }
    }
    out
}

fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Run the dp4a-mmvq split kernel against the production split kernel
/// for one (out_dim, in_dim) shape. Both consume the same Split byte layout
/// produced by `repack_q8_raw_to_split`; only the matvec kernels differ.
fn run_4thread_vs_split(out_dim: usize, in_dim: usize, seed: u64, tol: f32) {
    let (ctx, stream) = create_context();

    let raw_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;
    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE;
    let lc_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_4THREAD_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE;

    let raw_module = ctx
        .load_module(compile_sm80(raw_src))
        .expect("Q8Raw dp4a load");
    let split_module = ctx
        .load_module(compile_sm80(split_src))
        .expect("split load");
    let lc_module = ctx
        .load_module(compile_sm80(lc_src))
        .expect("4thread load");
    let repack_module = ctx
        .load_module(compile_sm80(repack_src))
        .expect("repack load");

    let split_fn = split_module
        .load_function("matvec_q8_split_q8_1")
        .expect("split fn");
    let split_res_fn = split_module
        .load_function("matvec_q8_split_q8_1_residual")
        .expect("split res fn");
    let lc_fn = lc_module
        .load_function("matvec_q8_split_q8_1_4thread")
        .expect("4thread fn");
    let lc_res_fn = lc_module
        .load_function("matvec_q8_split_q8_1_4thread_residual")
        .expect("4thread res fn");
    let repack_fn = repack_module
        .load_function("repack_q8_raw_to_split")
        .expect("repack fn");
    let quant_fn = raw_module
        .load_function("quantize_f32_to_q8_1")
        .expect("q8_1 quantize fn");

    let mut rng = seed;
    let mut weight_bytes: Vec<u8> =
        Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_RAW_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_bytes.extend_from_slice(&encode_q8_raw(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
    let residual_f32: Vec<f32> = (0..out_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    let weight_raw_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();
    let residual_gpu = stream.clone_htod(&residual_f32).unwrap();

    let nb = in_dim / Q8_BLOCK;
    let q8_1_bytes = nb * 36;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(q8_1_bytes).unwrap();

    let split_bytes = out_dim * nb * 34;
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(split_bytes).unwrap();

    // Repack: Q8Raw -> Q8Split.
    {
        let total_blocks = (out_dim * nb) as u32;
        let block_size = 256u32;
        let grid_size = (total_blocks + block_size - 1) / block_size;
        let nb_u32 = nb as u32;
        let out_dim_u32 = out_dim as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&repack_fn)
                .arg(&weight_raw_gpu)
                .arg(&mut split_buf)
                .arg(&nb_u32)
                .arg(&out_dim_u32)
                .launch(cfg)
        }
        .expect("repack_q8_raw_to_split launch");
    }

    // Quantize x to Q8_1.
    {
        let in_dim_u32 = in_dim as u32;
        let quant_grid = nb as u32;
        let cfg = LaunchConfig {
            grid_dim: (quant_grid, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&quant_fn)
                .arg(&x_gpu)
                .arg(&mut q8_1_buf)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("quantize_f32_to_q8_1 launch");
    }

    // Baseline: production split kernel.
    let mut out_split: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 2u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&split_fn)
                .arg(&split_buf)
                .arg(&q8_1_buf)
                .arg(&mut out_split)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_split_q8_1 launch");
    }

    // Candidate: dp4a-mmvq split kernel.
    let mut out_4thread: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 2u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&lc_fn)
                .arg(&split_buf)
                .arg(&q8_1_buf)
                .arg(&mut out_4thread)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_split_q8_1_4thread launch");
    }

    // Residual variants.
    let mut out_split_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_4thread_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 2u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&split_res_fn)
                .arg(&split_buf)
                .arg(&q8_1_buf)
                .arg(&residual_gpu)
                .arg(&mut out_split_res)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_split_q8_1_residual launch");
        unsafe {
            stream
                .launch_builder(&lc_res_fn)
                .arg(&split_buf)
                .arg(&q8_1_buf)
                .arg(&residual_gpu)
                .arg(&mut out_4thread_res)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_split_q8_1_4thread_residual launch");
    }

    stream.synchronize().unwrap();

    let host_split = stream.clone_dtoh(&out_split).unwrap();
    let host_4thread = stream.clone_dtoh(&out_4thread).unwrap();
    let host_split_res = stream.clone_dtoh(&out_split_res).unwrap();
    let host_4thread_res = stream.clone_dtoh(&out_4thread_res).unwrap();

    let mut max_err = 0.0f32;
    for i in 0..out_dim {
        let err = (host_4thread[i] - host_split[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < tol,
            "4thread vs split mismatch at row {i}: lc={} split={} err={} tol={}",
            host_4thread[i],
            host_split[i],
            err,
            tol,
        );
    }
    let mut max_err_res = 0.0f32;
    for i in 0..out_dim {
        let err = (host_4thread_res[i] - host_split_res[i]).abs();
        if err > max_err_res {
            max_err_res = err;
        }
        assert!(
            err < tol,
            "4thread+residual vs split mismatch at row {i}: lc={} split={} err={} tol={}",
            host_4thread_res[i],
            host_split_res[i],
            err,
            tol,
        );
    }
    eprintln!(
        "Q8 4thread vs split [{out_dim}x{in_dim}]: max_err={:.3e} max_err_res={:.3e} (tol={})",
        max_err, max_err_res, tol,
    );
}

// NOTE: smallest valid shape requires nb >= 2 because the SPLIT layout puts
// the quants stream at offset `2*nb` within the row, and the kernel does
// `(int*)(row_base + 2*nb + ...)` which requires 4-byte alignment. nb=1
// produces a +2 offset (XID 13 "Misaligned Address"). The production
// matvec_q8_split_q8_1 kernel has the same constraint -- see the "Alignment
// contract enforced by the host" comment block.

#[test]
fn cuda_q8_4thread_medium_64x256() {
    // nb = 256/32 = 8, alignment OK. NR=2 boundary stress.
    run_4thread_vs_split(64, 256, 91, 1e-3);
}

#[test]
fn cuda_q8_4thread_qwen_shape_4096x4096() {
    // The load-bearing FFN gate/up shape (and QKV combined).
    run_4thread_vs_split(4096, 4096, 123, 1e-3);
}

#[test]
fn cuda_q8_4thread_ffn_down_4096x12288() {
    // FFN down projection: 12288 in_dim, K-trip = 12 in reference pattern.
    run_4thread_vs_split(4096, 12288, 7777, 1e-3);
}

#[test]
fn cuda_q8_4thread_kv_shape_1024x4096() {
    run_4thread_vs_split(1024, 4096, 314, 1e-3);
}

#[test]
fn cuda_q8_4thread_gate_up_fused_8192x4096() {
    // Combined gate+up fused-Q shape (qwen35 doubled Q+gate proj).
    run_4thread_vs_split(8192, 4096, 2024, 1e-3);
}
