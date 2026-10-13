//! The f16 header (`d`, `s`) of the Q8_1 blocks the quantizers write, checked against the
//! values the kernel quantized, for each quantizer helper's file compiled both ways production
//! compiles it: to PTX that the driver JIT-compiles, and to a cubin for the device's own
//! architecture, the route `ffi::sass_arch_for` takes when NVRTC is newer than the driver.
//! ptxas 12.2 through 13.0 lowered a byte store of a 16-bit f16 register as an f16-to-integer
//! conversion, which replaced the low byte of every scale and sum.
//!
//!   * `quantize_f32_to_q8_1` (`matvec_dp4a_q8_1.cu`, compute_80, fast math): `s = d * sum(q)`;
//!   * `quantize_q8_1_rawsum` (`mmv_q.cu`) and `quantize_q8_1_moe` (`mmv_q_moe.cu`), on the
//!     dp4a target `ffi::dp4a_arch_for` picks: `s = sum(x)`;
//!   * `rmsnorm_to_q8_1` (`rmsnorm_q8_1.cu`, NVRTC's default target): `s = d * sum(q)` over
//!     the normalized vector;
//!   * `gdn_prefill_norm_gate_q8` (`gdn.cu`, NVRTC's default target): `s = sum(x)` over the
//!     gated vector it also writes in F32.
//!
//! Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_q8_1_header_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use lumen_runtime::cuda::shaders;
use std::sync::Arc;

const DIM: usize = 4096;

#[derive(Clone, Copy, Debug)]
enum Route {
    Ptx,
    Cubin,
}

/// The PTX target production compiles a module for.
enum Target {
    Default,
    Compute80,
    Dp4a,
}

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// `ffi::dp4a_arch_for` over the loaded NVRTC's targets: compute_61 when listed, else the
/// highest listed target at or below the device.
fn dp4a_target(cc: i32) -> i32 {
    use cudarc::nvrtc::sys;
    let mut count = 0;
    // SAFETY: the buffer is sized to the reported count before the second call writes it.
    let archs = unsafe {
        sys::nvrtcGetNumSupportedArchs(&mut count).result().unwrap();
        let mut archs = vec![0; count as usize];
        sys::nvrtcGetSupportedArchs(archs.as_mut_ptr())
            .result()
            .unwrap();
        archs
    };
    if archs.contains(&61) {
        return 61;
    }
    archs
        .into_iter()
        .filter(|&a| (61..=cc).contains(&a))
        .max()
        .expect("no dp4a target for this device")
}

/// `name` from `src`, compiled for its production `target` (PTX route) or for the device's own
/// `sm_XY` (cubin route), with the production options.
fn load(
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
    target: Target,
    fast_math: bool,
    route: Route,
) -> CudaFunction {
    let (major, minor) = ctx.compute_capability().unwrap();
    let arch = match (route, target) {
        (Route::Cubin, _) => Some(format!("sm_{major}{minor}")),
        (Route::Ptx, Target::Default) => None,
        (Route::Ptx, Target::Compute80) => Some("compute_80".to_string()),
        (Route::Ptx, Target::Dp4a) => Some(format!("compute_{}", dp4a_target(major * 10 + minor))),
    };
    let mut options: Vec<String> = arch
        .map(|a| format!("--gpu-architecture={a}"))
        .into_iter()
        .collect();
    if fast_math {
        options.push("--use_fast_math".to_string());
    }
    let image = match route {
        Route::Ptx => compile_ptx_with_opts(
            src,
            CompileOptions {
                options,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("NVRTC PTX compile of {name} failed: {e:?}")),
        Route::Cubin => Ptx::from_binary(cubin(src, &options)),
    };
    ctx.load_module(image)
        .and_then(|m| m.load_function(name))
        .unwrap_or_else(|e| panic!("load {name} ({route:?}): {e:?}"))
}

fn cubin(src: &str, options: &[String]) -> Vec<u8> {
    use cudarc::nvrtc::{result, sys};
    let source = std::ffi::CString::new(src).unwrap();
    let program = result::create_program(&source, None).unwrap();
    // SAFETY: `program` is live until the destroy below; the buffer is sized by the size query.
    unsafe {
        result::compile_program(program, options).expect("NVRTC cubin compile");
        let mut size = 0usize;
        sys::nvrtcGetCUBINSize(program, &mut size).result().unwrap();
        let mut bytes = vec![0u8; size];
        sys::nvrtcGetCUBIN(program, bytes.as_mut_ptr().cast())
            .result()
            .unwrap();
        result::destroy_program(program).unwrap();
        bytes
    }
}

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn rand_unit(s: &mut u64) -> f32 {
    ((rng_next(s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0
}

fn input(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..DIM).map(|_| rand_unit(&mut s) * 0.5).collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Every block's header against the host's values: `d` within f16 rounding of
/// `amax / 127` of `values` (the vector the kernel quantized), `s` within f16 rounding of
/// `sum(block)` (`raw_sum`) or of `d * sum(q)`.
fn check(kernel: &str, route: Route, out: &[u8], values: &[f64], raw_sum: bool) {
    let tol = 2f64.powi(-10);
    for (b, (blk, vals)) in out.chunks(36).zip(values.chunks(32)).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let s = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let amax = vals.iter().fold(0f64, |m, v| m.max(v.abs()));
        let want_d = amax / 127.0;
        assert!(
            (d - want_d).abs() <= want_d * tol,
            "{kernel} ({route:?}) block {b}: d {d} != amax/127 {want_d}"
        );
        let want_s = if raw_sum {
            vals.iter().sum::<f64>()
        } else {
            want_d * blk[4..].iter().map(|&q| q as i8 as f64).sum::<f64>()
        };
        assert!(
            (s - want_s).abs() <= want_s.abs() * tol + 1e-3,
            "{kernel} ({route:?}) block {b}: s {s} != {want_s}"
        );
    }
}

/// A one-warp-per-block quantizer `(x, vy, dim)` over `input(7)`.
fn quantize_case(
    src: &str,
    name: &str,
    target: Target,
    fast_math: bool,
    raw_sum: bool,
    route: Route,
) {
    let (ctx, stream) = create_context();
    let f = load(&ctx, src, name, target, fast_math, route);
    let x = input(7);
    let x_gpu = stream.clone_htod(&x).unwrap();
    let mut out = stream.alloc_zeros::<u8>(DIM / 32 * 36).unwrap();
    let dim = DIM as u32;
    let cfg = LaunchConfig {
        grid_dim: ((DIM / 32) as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&x_gpu)
            .arg(&mut out)
            .arg(&dim)
            .launch(cfg)
            .unwrap();
    }
    let values: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    check(
        name,
        route,
        &stream.clone_dtoh(&out).unwrap(),
        &values,
        raw_sum,
    );
}

fn rmsnorm_case(route: Route) {
    let (ctx, stream) = create_context();
    let f = load(
        &ctx,
        shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
        "rmsnorm_to_q8_1",
        Target::Default,
        false,
        route,
    );
    let x = input(11);
    let mut s = 13;
    let w: Vec<f32> = (0..DIM).map(|_| 0.5 + rand_unit(&mut s).abs()).collect();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&w).unwrap();
    let mut out = stream.alloc_zeros::<u8>(DIM / 32 * 36).unwrap();
    let (eps, dim) = (1e-6f32, DIM as u32);
    let block = 1024u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block / 32 * 4,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut out)
            .arg(&eps)
            .arg(&dim)
            .launch(cfg)
            .unwrap();
    }
    let ms = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / DIM as f64;
    let rms = 1.0 / (ms + eps as f64).sqrt();
    let normed: Vec<f64> = x
        .iter()
        .zip(&w)
        .map(|(&v, &g)| v as f64 * rms * g as f64)
        .collect();
    check(
        "rmsnorm_to_q8_1",
        route,
        &stream.clone_dtoh(&out).unwrap(),
        &normed,
        false,
    );
}

/// `gdn_prefill_norm_gate_q8` at the launch production uses: one CTA per head; the header is
/// checked against the F32 vector the kernel writes beside the blocks.
fn gdn_case(route: Route) {
    let (ctx, stream) = create_context();
    let f = load(
        &ctx,
        shaders::GDN_KERNEL_SOURCE,
        "gdn_prefill_norm_gate_q8",
        Target::Default,
        false,
        route,
    );
    let (heads, head_dim) = (32u32, 128u32);
    let raw = input(17);
    let gate = input(19);
    let mut s = 23;
    let scale: Vec<f32> = (0..DIM).map(|_| 0.5 + rand_unit(&mut s).abs()).collect();
    let raw_gpu = stream.clone_htod(&raw).unwrap();
    let gate_gpu = stream.clone_htod(&gate).unwrap();
    let scale_gpu = stream.clone_htod(&scale).unwrap();
    let mut gated = stream.alloc_zeros::<f32>(DIM).unwrap();
    let mut out = stream.alloc_zeros::<u8>(DIM / 32 * 36).unwrap();
    let eps = 1e-6f32;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: (head_dim.div_ceil(32) + 1) * 4,
    };
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&raw_gpu)
            .arg(&gate_gpu)
            .arg(&scale_gpu)
            .arg(&mut gated)
            .arg(&mut out)
            .arg(&heads)
            .arg(&head_dim)
            .arg(&eps)
            .arg(&heads)
            .launch(cfg)
            .unwrap();
    }
    let values: Vec<f64> = stream
        .clone_dtoh(&gated)
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    check(
        "gdn_prefill_norm_gate_q8",
        route,
        &stream.clone_dtoh(&out).unwrap(),
        &values,
        true,
    );
}

fn dp4a_q8_1(route: Route) {
    quantize_case(
        shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
        "quantize_f32_to_q8_1",
        Target::Compute80,
        true,
        false,
        route,
    );
}

fn rawsum(route: Route) {
    quantize_case(
        shaders::MMV_Q_DP4A_KERNEL_SOURCE,
        "quantize_q8_1_rawsum",
        Target::Dp4a,
        false,
        true,
        route,
    );
}

fn moe(route: Route) {
    quantize_case(
        shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
        "quantize_q8_1_moe",
        Target::Dp4a,
        false,
        true,
        route,
    );
}

#[test]
fn quantize_f32_to_q8_1_writes_its_header_via_ptx() {
    dp4a_q8_1(Route::Ptx);
}

#[test]
fn quantize_f32_to_q8_1_writes_its_header_via_cubin() {
    dp4a_q8_1(Route::Cubin);
}

#[test]
fn quantize_q8_1_rawsum_writes_its_header_via_ptx() {
    rawsum(Route::Ptx);
}

#[test]
fn quantize_q8_1_rawsum_writes_its_header_via_cubin() {
    rawsum(Route::Cubin);
}

#[test]
fn quantize_q8_1_moe_writes_its_header_via_ptx() {
    moe(Route::Ptx);
}

#[test]
fn quantize_q8_1_moe_writes_its_header_via_cubin() {
    moe(Route::Cubin);
}

#[test]
fn rmsnorm_to_q8_1_writes_its_header_via_ptx() {
    rmsnorm_case(Route::Ptx);
}

#[test]
fn rmsnorm_to_q8_1_writes_its_header_via_cubin() {
    rmsnorm_case(Route::Cubin);
}

#[test]
fn gdn_prefill_norm_gate_q8_writes_its_header_via_ptx() {
    gdn_case(Route::Ptx);
}

#[test]
fn gdn_prefill_norm_gate_q8_writes_its_header_via_cubin() {
    gdn_case(Route::Cubin);
}
