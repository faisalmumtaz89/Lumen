//! Integration tests for the CtInt4G32 CUDA kernels (`matvec_ct4_q8_1`,
//! `matvec_ct4_q8_1_residual`, `dequant_ct4_to_f16`) against CPU references
//! computed from the same decode blocks and the same Q8_1 activation bytes.
//!
//! Requires a CUDA-capable GPU (SM80+ for the dp4a kernels):
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_ct4_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

// The dp4a inline asm needs an explicit SM target; a plain NVRTC-default
// compile emits PTX the driver rejects (CUDA_ERROR_INVALID_PTX). Mirrors
// the shim in `cuda_matvec_q8_aligned_q8_1_hw_test.rs`. `--use_fast_math`
// matches the production loader (`load_fn_sm80_fast_math`) so the test
// exercises the same FMAD/FTZ code generation that ships.
fn compile_ptx(src: &str) -> Result<cudarc::nvrtc::Ptx, String> {
    compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some("compute_80"),
            use_fast_math: Some(true),
            ..Default::default()
        },
    )
    .map_err(|e| format!("{e:?}"))
}

/// Round-half-to-even for the Q8_1 quantizer reference
/// (`f32::round_ties_even` needs a newer toolchain than the workspace MSRV).
fn round_ties_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r as i64 % 2 != 0 {
        r - v.signum()
    } else {
        r
    }
}

const BLOCK_BYTES: usize = 20;
const GROUP: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 36;

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        let v = (frac as f32) * 6.103_515_6e-5 / 1024.0;
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return f32::NAN;
    }
    f32::from_bits((sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13))
}

/// Round-nearest-even f32 -> f16 bits (matches PTX `cvt.rn.f16.f32` for
/// normal values, which is all these tests generate).
fn f32_to_f16_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = bits & 0x7f_ffff;
    if val == 0.0 {
        return sign << 15;
    }
    assert!((1..31).contains(&exp), "test values must be f16-normal");
    let mut f16 = ((exp as u32) << 10) | (frac >> 13);
    let round_bits = frac & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (f16 & 1) == 1) {
        f16 += 1;
    }
    (sign << 15) | f16 as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// One synthetic decode block: bf16 scale, zero-point, and 32 4-bit values.
struct Ct4Block {
    d_bits: u16,
    zp: u8,
    q: [u8; 32],
}

impl Ct4Block {
    fn to_bytes(&self) -> [u8; BLOCK_BYTES] {
        let mut out = [0u8; BLOCK_BYTES];
        out[0..2].copy_from_slice(&self.d_bits.to_le_bytes());
        out[2] = self.zp;
        for i in 0..16 {
            out[4 + i] = (self.q[i] & 0xF) | ((self.q[i + 16] & 0xF) << 4);
        }
        out
    }
}

fn random_blocks(out_dim: usize, in_dim: usize, seed: u64) -> Vec<Ct4Block> {
    let mut s = seed;
    let n_blocks = out_dim * in_dim / GROUP;
    (0..n_blocks)
        .map(|_| {
            // bf16 scales in ~[0.004, 0.07]: realistic and exactly stored.
            let d = 0.004 + (rng_next(&mut s) % 64) as f32 * 0.001;
            let d_bits = ((d.to_bits() + 0x8000) >> 16) as u16;
            let zp = (rng_next(&mut s) % 16) as u8;
            let mut q = [0u8; 32];
            for v in &mut q {
                *v = (rng_next(&mut s) % 16) as u8;
            }
            Ct4Block { d_bits, zp, q }
        })
        .collect()
}

/// Quantize x to Q8_1 blocks on CPU with the engine's exact field math
/// (f16 scale, f16 sum = scale * sum(q)).
fn quantize_q8_1(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % GROUP, 0);
    let mut out = Vec::with_capacity(x.len() / GROUP * Q8_1_BLOCK_BYTES);
    for block in x.chunks_exact(GROUP) {
        let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let q: Vec<i8> = block
            .iter()
            .map(|v| round_ties_even(v * inv).clamp(-127.0, 127.0) as i8)
            .collect();
        let qsum: f32 = q.iter().map(|&v| v as f32).sum();
        let d_bits = f32_to_f16_rne(scale);
        let s_bits = f32_to_f16_rne(f16_bits_to_f32(d_bits) * qsum);
        out.extend_from_slice(&d_bits.to_le_bytes());
        out.extend_from_slice(&s_bits.to_le_bytes());
        out.extend(q.iter().map(|&v| v as u8));
    }
    out
}

/// CPU ground truth: exact integer dot and activation sum per block (the
/// zero-point term must NOT go through the Q8_1 block's f16 sum field —
/// its rounding is signal-scale after the zp multiply).
fn cpu_matvec_ct4(blocks: &[Ct4Block], q8_1: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let nb = in_dim / GROUP;
    let mut out = vec![0.0f32; out_dim];
    for (row, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for b in 0..nb {
            let blk = &blocks[row * nb + b];
            let d = bf16_bits_to_f32(blk.d_bits);
            let xb = &q8_1[b * Q8_1_BLOCK_BYTES..(b + 1) * Q8_1_BLOCK_BYTES];
            let x_scale = f16_bits_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let mut dot = 0i32;
            let mut xsum = 0i32;
            for i in 0..GROUP {
                dot += blk.q[i] as i32 * (xb[4 + i] as i8) as i32;
                xsum += (xb[4 + i] as i8) as i32;
            }
            acc += d * x_scale * (dot - blk.zp as i32 * xsum) as f32;
        }
        *o = acc;
    }
    out
}

fn compile_module(
    ctx: &Arc<CudaContext>,
) -> (
    Arc<cudarc::driver::CudaModule>,
    cudarc::driver::CudaFunction,
    cudarc::driver::CudaFunction,
    cudarc::driver::CudaFunction,
) {
    let src = lumen_runtime::cuda::shaders::MATVEC_CT4_G32_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_ct4_g32.cu");
    let module = ctx.load_module(ptx).expect("Failed to load ct4 module");
    let mv = module.load_function("matvec_ct4_q8_1").unwrap();
    let mv_res = module.load_function("matvec_ct4_q8_1_residual").unwrap();
    let dq = module.load_function("dequant_ct4_to_f16").unwrap();
    (module, mv, mv_res, dq)
}

fn run_case(out_dim: usize, in_dim: usize, seed: u64, with_residual: bool) {
    let (ctx, stream) = create_context();
    let (_m, mv, mv_res, _dq) = compile_module(&ctx);

    let blocks = random_blocks(out_dim, in_dim, seed);
    let weight_bytes: Vec<u8> = blocks.iter().flat_map(|b| b.to_bytes()).collect();
    let mut s = seed ^ 0x9e37;
    let x: Vec<f32> = (0..in_dim)
        .map(|_| ((rng_next(&mut s) % 512) as f32 - 256.0) / 256.0)
        .collect();
    let q8_1 = quantize_q8_1(&x);
    let residual: Vec<f32> = (0..out_dim)
        .map(|_| ((rng_next(&mut s) % 512) as f32 - 256.0) / 64.0)
        .collect();

    let mut expected = cpu_matvec_ct4(&blocks, &q8_1, out_dim, in_dim);
    if with_residual {
        for (e, r) in expected.iter_mut().zip(&residual) {
            *e += r;
        }
    }

    let w_gpu = stream.clone_htod(&weight_bytes).unwrap();
    let x_gpu = stream.clone_htod(&q8_1).unwrap();
    let r_gpu = stream.clone_htod(&residual).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_u32 = out_dim as u32;
    let in_u32 = in_dim as u32;
    unsafe {
        if with_residual {
            stream
                .launch_builder(&mv_res)
                .arg(&w_gpu)
                .arg(&x_gpu)
                .arg(&r_gpu)
                .arg(&mut out_gpu)
                .arg(&out_u32)
                .arg(&in_u32)
                .launch(cfg)
                .unwrap();
        } else {
            stream
                .launch_builder(&mv)
                .arg(&w_gpu)
                .arg(&x_gpu)
                .arg(&mut out_gpu)
                .arg(&out_u32)
                .arg(&in_u32)
                .launch(cfg)
                .unwrap();
        }
    }
    let got = stream.clone_dtoh(&out_gpu).unwrap();
    let ref_norm = expected
        .iter()
        .fold(0.0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() / ref_norm < 1e-5,
            "row {i}: got {g}, expected {e} (out_dim={out_dim}, in_dim={in_dim})"
        );
    }
}

#[test]
fn test_ct4_matvec_small() {
    run_case(4, 32, 1, false);
}

#[test]
fn test_ct4_matvec_large_random() {
    run_case(256, 2048, 2, false);
}

#[test]
fn test_ct4_matvec_residual() {
    run_case(64, 512, 3, true);
}

#[test]
fn test_ct4_dequant_f16_matches_cpu() {
    let (ctx, stream) = create_context();
    let (_m, _mv, _mv_res, dq) = compile_module(&ctx);
    let (out_dim, in_dim) = (16usize, 256usize);
    let blocks = random_blocks(out_dim, in_dim, 7);
    let weight_bytes: Vec<u8> = blocks.iter().flat_map(|b| b.to_bytes()).collect();
    let n = out_dim * in_dim;

    let w_gpu = stream.clone_htod(&weight_bytes).unwrap();
    let mut out_gpu: CudaSlice<u8> = stream.alloc_zeros(n * 2).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&dq)
            .arg(&w_gpu)
            .arg(&mut out_gpu)
            .arg(&n_u32)
            .launch(cfg)
            .unwrap();
    }
    let got = stream.clone_dtoh(&out_gpu).unwrap();
    for e in 0..n {
        let blk = &blocks[e / GROUP];
        let v = bf16_bits_to_f32(blk.d_bits) * (blk.q[e % GROUP] as f32 - blk.zp as f32);
        let expect_bits = f32_to_f16_rne(v);
        let got_bits = u16::from_le_bytes([got[e * 2], got[e * 2 + 1]]);
        // v may be exactly 0 (q == zp) which f32_to_f16_rne handles; all
        // other values are f16-normal by construction.
        assert_eq!(
            got_bits, expect_bits,
            "element {e}: got {got_bits:#06x}, expected {expect_bits:#06x} (v={v})"
        );
    }
}
