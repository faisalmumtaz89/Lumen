//! Differential test for the output-proj specialised large-NR matvec kernels
//! (`matvec_q8_split_output_proj_nr{16,32,64,128}`) against the in-tree NR=2
//! split kernel (`matvec_q8_split_q8_1`).
//!
//! The four specialised variants share a templated body and only differ in
//! their `NR` constant. They reuse the same per-row split weight layout and
//! the same Q8_1 input layout, so a correct implementation must produce
//! bit-near-identical outputs to the NR=2 baseline (the only differences come
//! from `dp4a` accumulation order, which is identical here because each row's
//! accumulation is independent of the NR value).
//!
//! Requires a CUDA-capable GPU. Skipped automatically when CUDA is unavailable.
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_output_proj_nr_decode_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// Compile each kernel source with SM 80 (A100) targeting and --use_fast_math
// to match production loader behaviour (`load_fn_sm80_fast_math`). NVRTC's
// default arch is too low to enable __dp4a, which all of these kernels use.
fn compile_for_a100(src: &str) -> cudarc::nvrtc::Ptx {
    let opts = CompileOptions {
        arch: Some("compute_80"),
        options: vec!["--use_fast_math".to_string()],
        ..Default::default()
    };
    compile_ptx_with_opts(src, opts).expect("nvrtc compile")
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
    assert!(values.len() % Q8_BLOCK == 0);
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

/// Compare large-NR output_proj kernel outputs against the NR=2 split baseline
/// at relative tolerance `rel_tol`. The NR=2 kernel is itself differentially
/// validated against `matvec_q8_0_q8_1` in `cuda_q8_split_decode_test.rs`, so a
/// match here is a transitive proof that the specialised kernel is correct.
fn run_nr_vs_nr2(out_dim: usize, in_dim: usize, seed: u64, rel_tol: f32) {
    let (ctx, stream) = create_context();

    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE;
    let nr_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE;
    let raw_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;

    let split_module = ctx
        .load_module(compile_for_a100(split_src))
        .expect("split load");
    let nr_module = ctx
        .load_module(compile_for_a100(nr_src))
        .expect("nr load");
    let repack_module = ctx
        .load_module(compile_for_a100(repack_src))
        .expect("repack load");
    let raw_module = ctx
        .load_module(compile_for_a100(raw_src))
        .expect("raw load");

    let split_fn = split_module
        .load_function("matvec_q8_split_q8_1")
        .expect("split fn");
    let nr8_fn = nr_module
        .load_function("matvec_q8_split_output_proj_nr8")
        .expect("nr8 fn");
    let nr16_fn = nr_module
        .load_function("matvec_q8_split_output_proj_nr16")
        .expect("nr16 fn");
    let nr32_fn = nr_module
        .load_function("matvec_q8_split_output_proj_nr32")
        .expect("nr32 fn");
    let nr64_fn = nr_module
        .load_function("matvec_q8_split_output_proj_nr64")
        .expect("nr64 fn");
    let nr128_fn = nr_module
        .load_function("matvec_q8_split_output_proj_nr128")
        .expect("nr128 fn");
    let repack_fn = repack_module
        .load_function("repack_q8_raw_to_split")
        .expect("repack fn");
    let quant_fn = raw_module
        .load_function("quantize_f32_to_q8_1")
        .expect("quantize fn");

    // Build random weights.
    let mut rng = seed;
    let mut weight_bytes: Vec<u8> =
        Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_RAW_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_bytes.extend_from_slice(&encode_q8_raw(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    // Upload.
    let weight_raw_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();

    let nb = in_dim / Q8_BLOCK;
    let q8_1_bytes = nb * 36;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(q8_1_bytes).unwrap();

    let split_bytes = out_dim * nb * 34;
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(split_bytes).unwrap();

    // Repack Q8Raw -> Q8Split.
    {
        let total_blocks = (out_dim * nb) as u32;
        let block_size = 256u32;
        let grid_size = total_blocks.div_ceil(block_size);
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
        .expect("repack launch");
    }

    // Quantize x to Q8_1.
    {
        let in_dim_u32 = in_dim as u32;
        let cfg = LaunchConfig {
            grid_dim: (nb as u32, 1, 1),
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
        .expect("quantize launch");
    }

    // Baseline: NR=2 split kernel.
    let mut out_baseline: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
    {
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let nr = 2u32;
        let grid = out_dim_u32.div_ceil(nr);
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
                .arg(&mut out_baseline)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("NR=2 split launch");
    }

    // Each candidate kernel.
    let candidates: [(u32, &cudarc::driver::CudaFunction); 5] = [
        (8, &nr8_fn),
        (16, &nr16_fn),
        (32, &nr32_fn),
        (64, &nr64_fn),
        (128, &nr128_fn),
    ];

    stream.synchronize().unwrap();
    let host_baseline = stream.clone_dtoh(&out_baseline).unwrap();

    for (nr, func) in candidates.iter() {
        let mut out_candidate: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let grid = out_dim_u32.div_ceil(*nr);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(*func)
                .arg(&split_buf)
                .arg(&q8_1_buf)
                .arg(&mut out_candidate)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .unwrap_or_else(|e| panic!("NR={nr} launch: {e}"));

        stream.synchronize().unwrap();
        let host_candidate = stream.clone_dtoh(&out_candidate).unwrap();

        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        for i in 0..out_dim {
            let b = host_baseline[i];
            let c = host_candidate[i];
            let abs_err = (c - b).abs();
            let denom = b.abs().max(1e-6);
            let rel_err = abs_err / denom;
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
            assert!(
                rel_err < rel_tol || abs_err < 1e-5,
                "NR={nr} mismatch at row {i}: baseline={b} candidate={c} abs={abs_err:.3e} rel={rel_err:.3e}",
            );
        }
        eprintln!(
            "NR={nr} vs NR=2 [{out_dim}x{in_dim}]: max_abs_err={:.3e} max_rel_err={:.3e}",
            max_abs_err, max_rel_err,
        );
    }
}

#[test]
fn cuda_output_proj_nr_small_256x128() {
    // Smallest shape that exercises NR=128: out_dim=256 (=2*128), in_dim=128
    // (4 Q8 blocks per row). Verifies kernels for all four NR variants on a
    // shape divisible by 128.
    run_nr_vs_nr2(256, 128, 0xDEADBEEFu64, 1e-3);
}

#[test]
fn cuda_output_proj_nr_ragged_300x128() {
    // Ragged shape: 300 rows is NOT a multiple of 16/32/64/128. Confirms the
    // in-kernel early-exit (`if target_row >= out_dim`) handles tail blocks.
    run_nr_vs_nr2(300, 128, 0x12345678u64, 1e-3);
}

#[test]
fn cuda_output_proj_nr_qwen35_shape_2048x4096() {
    // Surrogate for Qwen3.5-9B output_proj at production aspect ratio.
    // We use 2048 instead of 248320 to keep the test fast; the kernel logic
    // is independent of out_dim once nb is fixed (nb = 4096/32 = 128 here).
    run_nr_vs_nr2(2048, 4096, 0xABCDEF01u64, 1e-3);
}
