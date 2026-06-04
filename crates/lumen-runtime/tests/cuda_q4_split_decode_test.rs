//! Differential test for the Q4Raw -> Q4Split decode path against the in-tree
//! Q4Raw + Q8_1 dp4a matvec used in production (matvec_q4_0_dp4a).
//!
//! Pipeline under test:
//!   F32 weights -> Q4_0 (18B AoS, scale @+0, nibbles @+2) -> Q4Split (per-row
//!   [f16 scale * nb][nibble[16] * nb], 18*nb bytes/row) ->
//!   dp4a matvec against pre-quantized Q8_1 input.
//!
//! Mirror of cuda_q8_split_decode_test.rs for Q4_0 weights. The split sibling
//! is decode-only: GDN models keep Q4Raw in `lw.wq` (for HGEMM prefill) AND
//! populate `lw.q4_split_*` (for decode dp4a). Both paths must produce
//! bit-identical results from the same F32 source weights.
//!
//! Requires a CUDA-capable GPU. Skipped automatically when CUDA is unavailable.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_q4_split_decode_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const Q4_BLOCK: usize = 32;
const Q4_RAW_BYTES: usize = 18;

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

/// Encode a flat row of F32 values into 18-byte Q4Raw blocks using the GGML
/// de-interleaved layout: lo nibble of byte i = element i, hi nibble = element
/// i+16. Scale at offset +0..1 (f16 little-endian), nibbles at +2..17.
fn encode_q4_raw(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % Q4_BLOCK == 0, "encoder requires whole blocks");
    let n_blocks = values.len() / Q4_BLOCK;
    let mut out = vec![0u8; n_blocks * Q4_RAW_BYTES];
    for b in 0..n_blocks {
        let blk = &values[b * Q4_BLOCK..(b + 1) * Q4_BLOCK];
        // Q4_0: signed nibble range [-8, 7], maxabs / -8 -> scale.
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { -amax / 8.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let scale_bits = f32_to_f16_bits(scale);
        let block_dst = &mut out[b * Q4_RAW_BYTES..(b + 1) * Q4_RAW_BYTES];
        block_dst[0] = (scale_bits & 0xff) as u8;
        block_dst[1] = (scale_bits >> 8) as u8;
        // Quantize to nibbles 0..15 (centered at 8).
        let mut nibbles = [0u8; 32];
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * inv_scale + 8.5).floor().clamp(0.0, 15.0) as u8;
            nibbles[i] = q;
        }
        // De-interleaved packing: lo nibble of byte i = element i, hi = element i+16.
        for i in 0..16 {
            let lo = nibbles[i] & 0x0f;
            let hi = nibbles[i + 16] & 0x0f;
            block_dst[2 + i] = lo | (hi << 4);
        }
    }
    out
}

/// Simple LCG pseudo-random number generator for deterministic test data.
fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Run the Q4 split-layout differential test for one (out_dim, in_dim) shape
/// against the in-tree Q4Raw + Q8_1 dp4a matvec kernel.
fn run_split_vs_raw(out_dim: usize, in_dim: usize, seed: u64, tol: f32) {
    let (ctx, stream) = create_context();

    // --- Compile the kernels we need from the in-tree shaders. ---
    let raw_src = lumen_runtime::cuda::shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE;
    let q8_1_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;
    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q4_SPLIT_Q8_1_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q4_RAW_TO_SPLIT_KERNEL_SOURCE;

    let raw_module = ctx
        .load_module(compile_ptx(raw_src).expect("Q4Raw dp4a compile"))
        .expect("Q4Raw dp4a load");
    let q8_1_module = ctx
        .load_module(compile_ptx(q8_1_src).expect("Q8_1 quantize compile"))
        .expect("Q8_1 quantize load");
    let split_module = ctx
        .load_module(compile_ptx(split_src).expect("Q4 split compile"))
        .expect("Q4 split load");
    let repack_module = ctx
        .load_module(compile_ptx(repack_src).expect("Q4 repack compile"))
        .expect("Q4 repack load");

    let raw_fn = raw_module
        .load_function("matvec_q4_0_dp4a")
        .expect("Q4Raw matvec fn");
    let raw_res_fn = raw_module
        .load_function("matvec_q4_0_dp4a_residual")
        .expect("Q4Raw matvec residual fn");
    let split_fn = split_module
        .load_function("matvec_q4_split_q8_1")
        .expect("Q4 split fn");
    let split_res_fn = split_module
        .load_function("matvec_q4_split_q8_1_residual")
        .expect("Q4 split residual fn");
    let repack_fn = repack_module
        .load_function("repack_q4_raw_to_split")
        .expect("Q4 repack fn");
    let quant_fn = q8_1_module
        .load_function("quantize_f32_to_q8_1")
        .expect("Q8_1 quantize fn");

    // --- Build random weights as Q4Raw bytes on the host. ---
    let mut rng = seed;
    let mut weight_bytes: Vec<u8> = Vec::with_capacity(out_dim * (in_dim / Q4_BLOCK) * Q4_RAW_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_bytes.extend_from_slice(&encode_q4_raw(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
    let residual_f32: Vec<f32> = (0..out_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    // --- Upload buffers to GPU. ---
    let weight_raw_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();
    let residual_gpu = stream.clone_htod(&residual_f32).unwrap();

    // Q8_1 scratch: (in_dim / 32) blocks * 36 bytes.
    let nb = in_dim / Q4_BLOCK;
    let q8_1_bytes = nb * 36;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(q8_1_bytes).unwrap();

    // Split buffer: (out_dim * nb) blocks * 18 bytes (same density as Q4Raw).
    let split_bytes = out_dim * nb * 18;
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(split_bytes).unwrap();

    // --- Run the repack kernel: Q4Raw -> Q4Split. ---
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
        .expect("repack_q4_raw_to_split launch");
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

    // --- Baseline: matvec_q4_0_dp4a against the Q4Raw AoS layout (NR=4). ---
    let mut out_raw: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 4u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1), // matches DP4A_Q4_BLOCK_DIM = 256
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
        .expect("matvec_q4_0_dp4a launch");
    }

    // --- Candidate: matvec_q4_split_q8_1 against the SoA layout (NR=4). ---
    let mut out_split: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 4u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
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
        .expect("matvec_q4_split_q8_1 launch");
    }

    // --- Residual variant. ---
    let mut out_raw_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    let mut out_split_res: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 4u32;
        let grid = (out_dim_u32 + nr - 1) / nr;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
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
        .expect("matvec_q4_0_dp4a_residual launch");
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
        .expect("matvec_q4_split_q8_1_residual launch");
    }

    stream.synchronize().unwrap();

    let host_raw = stream.clone_dtoh(&out_raw).unwrap();
    let host_split = stream.clone_dtoh(&out_split).unwrap();
    let host_raw_res = stream.clone_dtoh(&out_raw_res).unwrap();
    let host_split_res = stream.clone_dtoh(&out_split_res).unwrap();

    // --- Verify the repack bit-preserves the source. ---
    let host_split_bytes = stream.clone_dtoh(&split_buf).unwrap();
    let row_bytes = nb * 18;
    for row in 0..out_dim {
        let src_row = &weight_bytes[row * nb * Q4_RAW_BYTES..(row + 1) * nb * Q4_RAW_BYTES];
        let dst_row = &host_split_bytes[row * row_bytes..(row + 1) * row_bytes];
        // Scales region: dst[0..2nb] should be source f16 scales from offsets 0..1 of each src block.
        for b in 0..nb {
            assert_eq!(
                dst_row[b * 2] as u8,
                src_row[b * Q4_RAW_BYTES] as u8,
                "row {row} block {b} scale lo byte mismatch",
            );
            assert_eq!(
                dst_row[b * 2 + 1] as u8,
                src_row[b * Q4_RAW_BYTES + 1] as u8,
                "row {row} block {b} scale hi byte mismatch",
            );
        }
        // Nibbles region: dst[2nb..18nb] should be 16 bytes from offset +2 of each src block.
        let nibble_offset = nb * 2;
        for b in 0..nb {
            for i in 0..16 {
                assert_eq!(
                    dst_row[nibble_offset + b * 16 + i] as u8,
                    src_row[b * Q4_RAW_BYTES + 2 + i] as u8,
                    "row {row} block {b} byte {i} nibble mismatch",
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
        "Q4 split vs raw [{out_dim}x{in_dim}]: max_err={:.3e} max_err_res={:.3e} (tol={})",
        max_err, max_err_res, tol,
    );
}

#[test]
fn cuda_q4_split_decode_small_4x32() {
    // Smallest meaningful shape: 1 block per row, 4 rows. Exercises the NR=4
    // boundary (4 / 4 = 1 CTA).
    run_split_vs_raw(4, 32, 17, 1e-3);
}

#[test]
fn cuda_q4_split_decode_medium_64x256() {
    run_split_vs_raw(64, 256, 91, 1e-3);
}

#[test]
fn cuda_q4_split_decode_qwen_shape_4096x4096() {
    // Qwen3.5-9B hidden_dim x hidden_dim.
    run_split_vs_raw(4096, 4096, 123, 1e-3);
}

#[test]
fn cuda_q4_split_decode_ffn_shape_4096x12288() {
    // FFN down projection (hidden x inter) — biggest per-row stride.
    run_split_vs_raw(4096, 12288, 7777, 1e-3);
}

#[test]
fn cuda_q4_split_decode_kv_shape_1024x4096() {
    // Qwen3.5-9B K/V projection (kv_dim x hidden).
    run_split_vs_raw(1024, 4096, 314, 1e-3);
}
