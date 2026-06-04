//! Microbenchmark: AoS NR8 (NR=8) Q8 aligned matvec vs production AoS
//! (NR=2, vdr=8), production SPLIT (NR=2, vdr=8), and SPLIT NR8 (NR=8)
//! across Qwen3.5-9B production shapes.
//!
//! Reports per-shape median microsecond timings and per-baseline speedup
//! ratios.
//!
//! `#[ignore]`-gated so it doesn't run in CI but can be invoked on Modal:
//!
//!   cargo test --release -p lumen-runtime --features cuda \
//!       --test cuda_q8_aos_nr8_microbench -- --ignored --nocapture
//!
//! Output (parseable):
//!   [MICROBENCH] shape=NAME out=N in=M aos_prod=NN.NNNus split_prod=NN.NNNus split_full=NN.NNNus aos_full=NN.NNNus aos_full/aos_prod=N.NNNx aos_full/split_prod=N.NNNx aos_full/split_full=N.NNNx

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::driver::result::event;
use cudarc::driver::sys as cuda_sys;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

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
const Q8_ALIGNED_BYTES: usize = 36;
const NUM_WARMUP: usize = 20;
const NUM_TRIALS: usize = 50;

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

fn encode_q8_aligned(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % Q8_BLOCK == 0);
    let n_blocks = values.len() / Q8_BLOCK;
    let mut out = vec![0u8; n_blocks * Q8_ALIGNED_BYTES];
    for b in 0..n_blocks {
        let blk = &values[b * Q8_BLOCK..(b + 1) * Q8_BLOCK];
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let scale_bits = f32_to_f16_bits(scale);
        let block_dst = &mut out[b * Q8_ALIGNED_BYTES..(b + 1) * Q8_ALIGNED_BYTES];
        block_dst[0] = (scale_bits & 0xff) as u8;
        block_dst[1] = (scale_bits >> 8) as u8;
        // Bytes 2-3 pad.
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            block_dst[4 + i] = q as u8;
        }
    }
    out
}

fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

struct Setup {
    #[allow(dead_code)]
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    aos_prod_fn: cudarc::driver::CudaFunction,
    split_prod_fn: cudarc::driver::CudaFunction,
    split_full_fn: cudarc::driver::CudaFunction,
    aos_full_fn: cudarc::driver::CudaFunction,
    aligned_buf: CudaSlice<i8>,
    split_buf: CudaSlice<i8>,
    q8_1_buf: CudaSlice<i8>,
    out_buf: CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
}

fn build_setup(out_dim: usize, in_dim: usize, seed: u64) -> Setup {
    let (ctx, stream) = create_context();

    let aligned_src = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE;
    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE;
    let split_full_src =
        lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_NR8_KERNEL_SOURCE;
    let aos_full_src = lumen_runtime::cuda::shaders::MATVEC_Q8_ALIGNED_NR8_KERNEL_SOURCE;
    let raw_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE;

    let aligned_module = ctx.load_module(compile_sm80(aligned_src)).expect("aligned load");
    let split_module = ctx.load_module(compile_sm80(split_src)).expect("split load");
    let split_full_module = ctx
        .load_module(compile_sm80(split_full_src))
        .expect("split_full load");
    let aos_full_module = ctx
        .load_module(compile_sm80(aos_full_src))
        .expect("aos_full load");
    let raw_module = ctx.load_module(compile_sm80(raw_src)).expect("raw load");
    let repack_module = ctx.load_module(compile_sm80(repack_src)).expect("repack load");

    let aos_prod_fn = aligned_module
        .load_function("matvec_q8_aligned_q8_1")
        .expect("aos_prod fn");
    let split_prod_fn = split_module
        .load_function("matvec_q8_split_q8_1")
        .expect("split_prod fn");
    let split_full_fn = split_full_module
        .load_function("matvec_q8_split_q8_1_nr8")
        .expect("split_full fn");
    let aos_full_fn = aos_full_module
        .load_function("matvec_q8_aligned_nr8")
        .expect("aos_full fn");
    let quant_fn = raw_module
        .load_function("quantize_f32_to_q8_1")
        .expect("quant fn");
    let repack_fn = repack_module
        .load_function("repack_q8_raw_to_split")
        .expect("repack fn");

    let mut rng = seed;
    let mut weight_raw_bytes: Vec<u8> =
        Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_RAW_BYTES);
    let mut weight_aligned_bytes: Vec<u8> =
        Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_ALIGNED_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_raw_bytes.extend_from_slice(&encode_q8_raw(&row));
        weight_aligned_bytes.extend_from_slice(&encode_q8_aligned(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    let weight_raw_i8: Vec<i8> = weight_raw_bytes.iter().map(|&b| b as i8).collect();
    let weight_aligned_i8: Vec<i8> = weight_aligned_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let aligned_buf = stream.clone_htod(&weight_aligned_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();

    let nb = in_dim / Q8_BLOCK;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(nb * 36).unwrap();
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(out_dim * nb * Q8_RAW_BYTES).unwrap();
    let out_buf: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    // Repack Q8Raw -> SPLIT.
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
        .expect("repack launch");
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
        .expect("quant launch");
    }
    stream.synchronize().unwrap();

    Setup {
        ctx,
        stream,
        aos_prod_fn,
        split_prod_fn,
        split_full_fn,
        aos_full_fn,
        aligned_buf,
        split_buf,
        q8_1_buf,
        out_buf,
        out_dim,
        in_dim,
    }
}

unsafe fn launch_aos_prod(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 1) / 2; // NR=2
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.aos_prod_fn)
        .arg(&s.aligned_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("aos_prod launch");
}

unsafe fn launch_split_prod(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 1) / 2; // NR=2
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.split_prod_fn)
        .arg(&s.split_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("split_prod launch");
}

unsafe fn launch_split_full(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 7) / 8; // NR=8
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.split_full_fn)
        .arg(&s.split_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("split_full launch");
}

unsafe fn launch_aos_full(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 7) / 8; // NR=8
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.aos_full_fn)
        .arg(&s.aligned_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("aos_full launch");
}

fn time_kernel<F>(s: &mut Setup, mut launch: F) -> Vec<f32>
where
    F: FnMut(&mut Setup),
{
    let mut events = Vec::with_capacity(NUM_TRIALS);
    for _ in 0..NUM_TRIALS {
        let e0 = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
            .expect("event create start");
        let e1 = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
            .expect("event create end");
        events.push((e0, e1));
    }

    for _ in 0..NUM_WARMUP {
        launch(s);
    }
    s.stream.synchronize().unwrap();

    let raw_stream = s.stream.cu_stream();
    let mut times_ms: Vec<f32> = Vec::with_capacity(NUM_TRIALS);
    for (e0, e1) in &events {
        unsafe {
            event::record(*e0, raw_stream).expect("event record start");
            launch(s);
            event::record(*e1, raw_stream).expect("event record end");
        }
    }
    s.stream.synchronize().unwrap();
    for (e0, e1) in &events {
        let ms = unsafe { event::elapsed(*e0, *e1).expect("event elapsed") };
        times_ms.push(ms);
    }
    for (e0, e1) in &events {
        unsafe {
            let _ = event::destroy(*e0);
            let _ = event::destroy(*e1);
        }
    }
    times_ms
}

fn median(vals: &mut Vec<f32>) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

fn bench_shape(name: &str, out_dim: usize, in_dim: usize) {
    let mut s = build_setup(out_dim, in_dim, 0xC0FFEE);
    let mut t_aos_prod = time_kernel(&mut s, |s| unsafe { launch_aos_prod(s) });
    let mut t_split_prod = time_kernel(&mut s, |s| unsafe { launch_split_prod(s) });
    let mut t_split_full = time_kernel(&mut s, |s| unsafe { launch_split_full(s) });
    let mut t_aos_full = time_kernel(&mut s, |s| unsafe { launch_aos_full(s) });
    let med_aos_prod = median(&mut t_aos_prod) * 1000.0; // ms -> us
    let med_split_prod = median(&mut t_split_prod) * 1000.0;
    let med_split_full = median(&mut t_split_full) * 1000.0;
    let med_aos_full = median(&mut t_aos_full) * 1000.0;
    // Speedups are larger-is-better for aos_full (ratio of baseline / candidate).
    let f_vs_aos = med_aos_prod / med_aos_full;
    let f_vs_split = med_split_prod / med_aos_full;
    let f_vs_split_full = med_split_full / med_aos_full;
    println!(
        "[MICROBENCH] shape={name:<13} out={out_dim:<7} in={in_dim:<7} aos_prod={:8.3}us split_prod={:8.3}us split_full={:8.3}us aos_full={:8.3}us aos_full/aos_prod={:.3}x aos_full/split_prod={:.3}x aos_full/split_full={:.3}x",
        med_aos_prod, med_split_prod, med_split_full, med_aos_full,
        f_vs_aos, f_vs_split, f_vs_split_full,
    );
}

#[test]
#[ignore]
fn microbench_aos_nr8_vs_all() {
    // Qwen3.5-9B production shapes hit during decode.
    bench_shape("4096x4096",    4096,  4096);   // FFN gate, up; QKV separated; GDN ssm_out
    bench_shape("1024x4096",    1024,  4096);   // K, V proj
    bench_shape("4096x12288",   4096, 12288);   // FFN down
    bench_shape("12288x4096",  12288,  4096);   // FFN gate+up SEPARATED, each side
    bench_shape("8192x4096",    8192,  4096);   // Qwen3.5 attn_q (Q+gate fused)
    bench_shape("248320x4096", 248320, 4096);   // output_proj (full vocab)
}
