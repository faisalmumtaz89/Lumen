//! The f16 header (`d`, `s`) of the Q8_1 blocks the decode quantizers write, checked against
//! values computed on the host, for each quantizer compiled both ways production compiles it:
//! to PTX that the driver JIT-compiles, and to a cubin for the device's own architecture, the
//! route `ffi::sass_arch_for` takes when NVRTC is newer than the driver. Some ptxas versions
//! lowered a byte store of an f16 register as an f16-to-integer conversion, which zeroed the
//! low byte of every scale below 1 and truncated the sums.
//!
//!   * `quantize_f32_to_q8_1` (`matvec_dp4a_q8_1.cu`, compute_80, fast math): `s = d * sum(q)`;
//!   * `rmsnorm_to_q8_1` (`rmsnorm_q8_1.cu`, NVRTC's default target): `s = d * sum(q)` over
//!     the normalized vector;
//!   * `quantize_q8_1_rawsum` (`mmv_q.cu`, the dp4a target): `s = sum(x)`.
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

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// `name` from `src`, compiled for `arch` (PTX route) or for the device's own `sm_XY` (cubin
/// route), with the production options.
fn load(
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
    arch: Option<&'static str>,
    fast_math: bool,
    route: Route,
) -> CudaFunction {
    let image = match route {
        Route::Ptx => compile_ptx_with_opts(
            src,
            CompileOptions {
                arch,
                options: if fast_math {
                    vec!["--use_fast_math".to_string()]
                } else {
                    Vec::new()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("NVRTC PTX compile of {name} failed: {e:?}")),
        Route::Cubin => {
            let (major, minor) = ctx.compute_capability().unwrap();
            Ptx::from_binary(cubin(src, &format!("sm_{major}{minor}"), fast_math))
        }
    };
    ctx.load_module(image)
        .and_then(|m| m.load_function(name))
        .unwrap_or_else(|e| panic!("load {name} ({route:?}): {e:?}"))
}

fn cubin(src: &str, arch: &str, fast_math: bool) -> Vec<u8> {
    use cudarc::nvrtc::{result, sys};
    let source = std::ffi::CString::new(src).unwrap();
    let mut options = vec![format!("--gpu-architecture={arch}")];
    if fast_math {
        options.push("--use_fast_math".to_string());
    }
    let program = result::create_program(&source, None).unwrap();
    // SAFETY: `program` is live until the destroy below; the buffer is sized by the size query.
    unsafe {
        result::compile_program(program, &options).expect("NVRTC cubin compile");
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

fn input(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..DIM).map(|_| rand_unit(&mut s) * 0.5).collect()
}

fn quantize_case(route: Route, raw_sum: bool) {
    let (ctx, stream) = create_context();
    let (src, name, arch, fast_math) = if raw_sum {
        let (major, minor) = ctx.compute_capability().unwrap();
        let cc = major * 10 + minor;
        assert!(cc >= 61, "dp4a needs compute capability 6.1");
        (
            shaders::MMV_Q_DP4A_KERNEL_SOURCE,
            "quantize_q8_1_rawsum",
            Some("compute_61"),
            false,
        )
    } else {
        (
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "quantize_f32_to_q8_1",
            Some("compute_80"),
            true,
        )
    };
    let f = load(&ctx, src, name, arch, fast_math, route);
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
        None,
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

#[test]
fn quantize_f32_to_q8_1_writes_its_header_via_ptx() {
    quantize_case(Route::Ptx, false);
}

#[test]
fn quantize_f32_to_q8_1_writes_its_header_via_cubin() {
    quantize_case(Route::Cubin, false);
}

#[test]
fn quantize_q8_1_rawsum_writes_its_header_via_ptx() {
    quantize_case(Route::Ptx, true);
}

#[test]
fn quantize_q8_1_rawsum_writes_its_header_via_cubin() {
    quantize_case(Route::Cubin, true);
}

#[test]
fn rmsnorm_to_q8_1_writes_its_header_via_ptx() {
    rmsnorm_case(Route::Ptx);
}

#[test]
fn rmsnorm_to_q8_1_writes_its_header_via_cubin() {
    rmsnorm_case(Route::Cubin);
}
