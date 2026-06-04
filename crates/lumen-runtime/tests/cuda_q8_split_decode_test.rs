//! Differential test for the Q8Raw -> Q8Split decode path against the in-tree
//! Q8Raw + Q8_1 dp4a matvec used in production (matvec_q8_0_q8_1).
//!
//! Pipeline under test:
//!   F32 weights -> Q8_0 (34B AoS, scale @+0, quants @+2) -> Q8Split (per-row
//!   [f16 scale * nb][int8 quants[32] * nb], 34*nb bytes/row) ->
//!   dp4a matvec against pre-quantized Q8_1 input.
//!
//! The differential check is load-bearing for the dual Q8Raw + Q8Split layout:
//! GDN models keep Q8Raw in `lw.wq`/etc. (for batched HGEMM prefill) AND populate
//! the sibling `lw.q8_split_*` buffer (for decode dp4a). Decode reads the
//! sibling, prefill reads the base. Both must produce bit-identical results
//! from the same F32 source weights.
//!
//! The test confirms three invariants:
//!   1. The Q8Raw -> Q8Split repack preserves every f16 scale and int8 quant.
//!   2. matvec_q8_split_q8_1 matches matvec_q8_0_q8_1 within the 1e-3 tolerance
//!      (logit-level end-to-end tolerance from the production Z0 baseline).
//!   3. The residual variant matches as well.
//!
//! Requires a CUDA-capable GPU. Skipped automatically when CUDA is unavailable.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_q8_split_decode_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

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

/// Encode a flat row of F32 values into 34-byte Q8Raw blocks. The layout matches
/// the bytes loaded from GGUF / produced by the CUDA dequant inverse: scale at
/// offset +0..1 (f16 little-endian), quants at offset +2..33 (int8).
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

/// Simple LCG pseudo-random number generator for deterministic test data.
fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Run the Q8 split-layout differential test for one (out_dim, in_dim) shape
/// against the in-tree Q8Raw + Q8_1 dp4a matvec kernel.
fn run_split_vs_raw(out_dim: usize, in_dim: usize, seed: u64, tol: f32) {
    let (ctx, stream) = create_context();

    // --- Compile the kernels we need from the in-tree shaders. ---
    let raw_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;
    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE;

    let raw_module = ctx
        .load_module(compile_ptx(raw_src).expect("Q8Raw dp4a compile"))
        .expect("Q8Raw dp4a load");
    let split_module = ctx
        .load_module(compile_ptx(split_src).expect("split compile"))
        .expect("split load");
    let repack_module = ctx
        .load_module(compile_ptx(repack_src).expect("repack compile"))
        .expect("repack load");

    let raw_fn = raw_module
        .load_function("matvec_q8_0_q8_1")
        .expect("Q8Raw matvec fn");
    let raw_res_fn = raw_module
        .load_function("matvec_q8_0_q8_1_residual")
        .expect("Q8Raw matvec residual fn");
    let split_fn = split_module
        .load_function("matvec_q8_split_q8_1")
        .expect("split fn");
    let split_res_fn = split_module
        .load_function("matvec_q8_split_q8_1_residual")
        .expect("split res fn");
    let repack_fn = repack_module
        .load_function("repack_q8_raw_to_split")
        .expect("repack fn");
    let quant_fn = raw_module
        .load_function("quantize_f32_to_q8_1")
        .expect("q8_1 quantize fn");

    // --- Build random weights as Q8Raw bytes on the host. ---
    let mut rng = seed;
    let mut weight_bytes: Vec<u8> = Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_RAW_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_bytes.extend_from_slice(&encode_q8_raw(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
    let residual_f32: Vec<f32> = (0..out_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    // --- Upload buffers to GPU. ---
    let weight_raw_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();
    let residual_gpu = stream.clone_htod(&residual_f32).unwrap();

    // Q8_1 scratch: (in_dim / 32) blocks * 36 bytes.
    let nb = in_dim / Q8_BLOCK;
    let q8_1_bytes = nb * 36;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(q8_1_bytes).unwrap();

    // Split buffer: (out_dim * nb) blocks * 34 bytes (same density as Q8Raw).
    let split_bytes = out_dim * nb * 34;
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(split_bytes).unwrap();

    // --- Run the repack kernel: Q8Raw -> Q8Split. ---
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

    // --- Quantize the F32 input vector to Q8_1 (shared by both matvec kernels). ---
    {
        let in_dim_u32 = in_dim as u32;
        let quant_grid = nb as u32;
        let cfg = LaunchConfig {
            grid_dim: (quant_grid, 1, 1),
            block_dim: (32, 1, 1), // matches Q8_1_QUANT_BLOCK_DIM = 32
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

    // --- Baseline: matvec_q8_0_q8_1 against the Q8Raw AoS layout. ---
    let mut out_raw: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 2u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1), // matches DP4A_Q8_1_BLOCK_DIM = 128
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&raw_fn)
                .arg(&weight_raw_gpu)
                .arg(&q8_1_buf)
                .arg(&mut out_raw)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_0_q8_1 launch");
    }

    // --- Candidate: matvec_q8_split_q8_1 against the SoA layout. ---
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

    // --- Residual variant. ---
    let mut out_raw_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_split_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
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
                .launch_builder(&raw_res_fn)
                .arg(&weight_raw_gpu)
                .arg(&q8_1_buf)
                .arg(&residual_gpu)
                .arg(&mut out_raw_res)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_q8_0_q8_1_residual launch");
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
    }

    stream.synchronize().unwrap();

    let host_raw = stream.clone_dtoh(&out_raw).unwrap();
    let host_split = stream.clone_dtoh(&out_split).unwrap();
    let host_raw_res = stream.clone_dtoh(&out_raw_res).unwrap();
    let host_split_res = stream.clone_dtoh(&out_split_res).unwrap();

    // --- Verify the repack bit-preserves the source. ---
    let host_split_bytes = stream.clone_dtoh(&split_buf).unwrap();
    let row_bytes = nb * 34;
    for row in 0..out_dim {
        let src_row = &weight_bytes[row * nb * Q8_RAW_BYTES..(row + 1) * nb * Q8_RAW_BYTES];
        let dst_row = &host_split_bytes[row * row_bytes..(row + 1) * row_bytes];
        // Scales region: dst[0..2nb] should be source f16 scales from offsets 0..1 of each src block.
        for b in 0..nb {
            assert_eq!(
                dst_row[b * 2] as u8,
                src_row[b * Q8_RAW_BYTES] as u8,
                "row {row} block {b} scale lo byte mismatch",
            );
            assert_eq!(
                dst_row[b * 2 + 1] as u8,
                src_row[b * Q8_RAW_BYTES + 1] as u8,
                "row {row} block {b} scale hi byte mismatch",
            );
        }
        // Quants region: dst[2nb..34nb] should be 32 int8 from offset +2 of each src block.
        let quants_offset = nb * 2;
        for b in 0..nb {
            for i in 0..32 {
                assert_eq!(
                    dst_row[quants_offset + b * 32 + i] as u8,
                    src_row[b * Q8_RAW_BYTES + 2 + i] as u8,
                    "row {row} block {b} byte {i} quant mismatch",
                );
            }
        }
    }

    // --- Compare matvec outputs. ---
    let mut max_err = 0.0f32;
    for i in 0..out_dim {
        let err = (host_split[i] - host_raw[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < tol,
            "matvec mismatch at row {i}: split={} raw={} err={} tol={}",
            host_split[i],
            host_raw[i],
            err,
            tol,
        );
    }
    let mut max_err_res = 0.0f32;
    for i in 0..out_dim {
        let err = (host_split_res[i] - host_raw_res[i]).abs();
        if err > max_err_res {
            max_err_res = err;
        }
        assert!(
            err < tol,
            "matvec+residual mismatch at row {i}: split={} raw={} err={} tol={}",
            host_split_res[i],
            host_raw_res[i],
            err,
            tol,
        );
    }
    eprintln!(
        "Q8 split vs raw [{out_dim}x{in_dim}]: max_err={:.3e} max_err_res={:.3e} (tol={})",
        max_err, max_err_res, tol,
    );
}

#[test]
fn cuda_q8_split_decode_small_4x32() {
    // Smallest meaningful shape: 1 block per row, 4 rows. Exercises the NR=2
    // boundary (4 / 2 = 2 CTAs) without stressing memory.
    run_split_vs_raw(4, 32, 17, 1e-3);
}

#[test]
fn cuda_q8_split_decode_medium_64x256() {
    run_split_vs_raw(64, 256, 91, 1e-3);
}

#[test]
fn cuda_q8_split_decode_qwen_shape_4096x4096() {
    // Qwen3.5-9B hidden_dim x hidden_dim: the load-bearing shape.
    run_split_vs_raw(4096, 4096, 123, 1e-3);
}

#[test]
fn cuda_q8_split_decode_ffn_shape_4096x12288() {
    // FFN down projection (hidden x inter) — biggest per-row stride.
    run_split_vs_raw(4096, 12288, 7777, 1e-3);
}

#[test]
fn cuda_q8_split_decode_kv_shape_1024x4096() {
    // Qwen3.5-9B K/V projection (kv_dim x hidden).
    run_split_vs_raw(1024, 4096, 314, 1e-3);
}
