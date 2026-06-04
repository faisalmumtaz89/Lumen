//! Microbenchmark: dp4a-mmvq Q8 split matvec vs production Q8 split matvec.
//!
//! Times both kernels via cudaEvent across the canonical FFN/QKV shapes from
//! Qwen3.5-9B. Per-shape median of NUM_TRIALS measurements after NUM_WARMUP
//! warmup iterations.
//!
//! This is an `#[ignore]`-gated test so it doesn't run in CI but can be
//! invoked explicitly on Modal A100-80GB:
//!
//!   cargo test --release -p lumen-runtime --features cuda \
//!       --test cuda_q8_4thread_microbench -- --ignored --nocapture
//!
//! Output (parseable JSON-ish lines):
//!   [MICROBENCH] shape=4096x4096   split=NN.NNNus  4thread=NN.NNNus  speedup=N.NNx
//!
//! These numbers are read by the Modal harness and folded into the
//! benchmark report. The matchup is deterministic (fixed LCG seed) so absolute
//! us-per-call values are comparable across reruns.

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

fn lcg_next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

struct Setup {
    #[allow(dead_code)]
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    split_fn: cudarc::driver::CudaFunction,
    lc_fn: cudarc::driver::CudaFunction,
    split_buf: CudaSlice<i8>,
    q8_1_buf: CudaSlice<i8>,
    out_buf: CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
}

fn build_setup(out_dim: usize, in_dim: usize, seed: u64) -> Setup {
    let (ctx, stream) = create_context();

    let split_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE;
    let lc_src = lumen_runtime::cuda::shaders::MATVEC_Q8_SPLIT_Q8_1_4THREAD_KERNEL_SOURCE;
    let raw_src = lumen_runtime::cuda::shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE;
    let repack_src = lumen_runtime::cuda::shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE;

    let split_module = ctx
        .load_module(compile_sm80(split_src))
        .expect("split load");
    let lc_module = ctx
        .load_module(compile_sm80(lc_src))
        .expect("lc load");
    let raw_module = ctx
        .load_module(compile_sm80(raw_src))
        .expect("raw load");
    let repack_module = ctx
        .load_module(compile_sm80(repack_src))
        .expect("repack load");

    let split_fn = split_module
        .load_function("matvec_q8_split_q8_1")
        .expect("split fn");
    let lc_fn = lc_module
        .load_function("matvec_q8_split_q8_1_4thread")
        .expect("lc fn");
    let quant_fn = raw_module
        .load_function("quantize_f32_to_q8_1")
        .expect("quant fn");
    let repack_fn = repack_module
        .load_function("repack_q8_raw_to_split")
        .expect("repack fn");

    let mut rng = seed;
    let mut weight_bytes: Vec<u8> =
        Vec::with_capacity(out_dim * (in_dim / Q8_BLOCK) * Q8_RAW_BYTES);
    for _ in 0..out_dim {
        let row: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();
        weight_bytes.extend_from_slice(&encode_q8_raw(&row));
    }
    let x_f32: Vec<f32> = (0..in_dim).map(|_| lcg_next_f32(&mut rng)).collect();

    let weight_raw_i8: Vec<i8> = weight_bytes.iter().map(|&b| b as i8).collect();
    let weight_raw_gpu = stream.clone_htod(&weight_raw_i8).unwrap();
    let x_gpu = stream.clone_htod(&x_f32).unwrap();

    let nb = in_dim / Q8_BLOCK;
    let mut q8_1_buf: CudaSlice<i8> = stream.alloc_zeros(nb * 36).unwrap();
    let mut split_buf: CudaSlice<i8> = stream.alloc_zeros(out_dim * nb * 34).unwrap();
    let out_buf: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

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
        .expect("repack launch");
    }
    // Quantize input.
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
        split_fn,
        lc_fn,
        split_buf,
        q8_1_buf,
        out_buf,
        out_dim,
        in_dim,
    }
}

unsafe fn launch_split(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 1) / 2;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.split_fn)
        .arg(&s.split_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("split launch");
}

unsafe fn launch_4thread(s: &mut Setup) {
    let out_dim_u32 = s.out_dim as u32;
    let in_dim_u32 = s.in_dim as u32;
    let grid = (out_dim_u32 + 1) / 2;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    s.stream
        .launch_builder(&s.lc_fn)
        .arg(&s.split_buf)
        .arg(&s.q8_1_buf)
        .arg(&mut s.out_buf)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .expect("lc launch");
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

    // Warmup.
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
    let mut times_split = time_kernel(&mut s, |s| unsafe { launch_split(s) });
    let mut times_4thread = time_kernel(&mut s, |s| unsafe { launch_4thread(s) });
    let med_split = median(&mut times_split) * 1000.0;  // ms -> us
    let med_4thread = median(&mut times_4thread) * 1000.0;
    let speedup = med_split / med_4thread;
    println!(
        "[MICROBENCH] shape={name:<13} out={out_dim:<7} in={in_dim:<7} split={:8.3}us  4thread={:8.3}us  speedup={:.3}x",
        med_split, med_4thread, speedup,
    );
}

#[test]
#[ignore]
fn microbench_4thread_vs_split_all_shapes() {
    // Qwen3.5-9B production shapes hit during decode.
    // hidden_dim = 4096, kv_dim = 1024, inter_dim = 12288, qkv fused = 8192
    bench_shape("4096x4096",    4096,  4096);   // FFN gate, up; QKV separated
    bench_shape("4096x12288",   4096, 12288);   // FFN down
    bench_shape("12288x4096",  12288,  4096);   // FFN gate+up (separate)
    bench_shape("8192x4096",    8192,  4096);   // Qwen3.5 attn_q (Q+gate fused)
    bench_shape("1024x4096",    1024,  4096);   // K, V proj
    bench_shape("248320x4096", 248320, 4096);   // output_proj (full vocab)
}
