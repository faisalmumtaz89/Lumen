//! The FP8 E4M3 decode matvec (`matvec_fp8_f32`) and its residual twin (`matvec_fp8_f32_residual`, which
//! computes `y = W*x + residual` for the attention output projection) against an independent host reference.
//!
//! Every representative FP8 matrix shape must agree with the reference within `max_abs < 1e-3` and
//! `rel_l2 <= 1e-4`, with both sides finite and the reference not identically zero. The residual twin must match the reference plus the residual, and
//! differ from the plain kernel by exactly the residual.
//!
//! The reference decodes with `lumen_format::planar_dequant::dequantize_fp8` and multiplies in f64, so it
//! shares no code with the kernel.
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_fp8_matvec_test
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use lumen_format::planar_dequant::dequantize_fp8;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::MATVEC_FP8_KERNEL_SOURCE;

/// Five FP8 projection shapes with realistic `weight_scale` values, plus a square 5120 x 5120 cell at a
/// large `weight_scale` (the shape the residual test uses).
const CELLS: &[(u32, u32, f32)] = &[
    (1024, 5120, 6.25e-4),
    (5120, 6144, 3.0e-3),
    (6144, 5120, 5.5e-4),
    (10240, 5120, 9.5e-4),
    (12288, 5120, 1.1e-3),
    (5120, 5120, 3.3e-3), // a large FP8 weight_scale
];

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

fn reference_dot(weights: &[u8], scale: f32, x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; out_dim];
    for row in 0..out_dim {
        let w = dequantize_fp8(&weights[row * in_dim..(row + 1) * in_dim], scale);
        let mut acc = 0.0f64;
        for (wi, xi) in w.iter().zip(x.iter()) {
            acc += (*wi as f64) * (*xi as f64);
        }
        out[row] = acc;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run(
    dev: &CudaDevice,
    weights: &[u8],
    scale: f32,
    x: &[f32],
    residual: Option<&[f32]>,
    out_dim: u32,
    in_dim: u32,
) -> Vec<f32> {
    // The kernel takes ONE plane: weight | global_scale(F32 LE), the converter's order
    // (`convert_hf.rs::lower_fp8`). Building it here exercises the real layout rather than a simplified
    // one, and it is what the production dispatch passes.
    let mut plane = Vec::with_capacity(weights.len() + 8);
    plane.extend_from_slice(weights);
    plane.extend_from_slice(&scale.to_le_bytes());
    // The activation scale a converted slice may carry after its planes: NaN, so a kernel that read it
    // would fail every comparison here.
    plane.extend_from_slice(&f32::NAN.to_le_bytes());
    let d_w = dev.htod_copy(&plane).expect("htod plane");
    let d_x = dev.htod_copy(x).expect("htod x");
    let mut d_out = dev.alloc_zeros::<f32>(out_dim as usize).expect("alloc out");
    let m = dev
        .compile_and_load(MATVEC_FP8_KERNEL_SOURCE)
        .expect("compile");
    const THREADS: u32 = 128;
    let grid = out_dim.div_ceil(THREADS / 32).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    match residual {
        Some(r) => {
            let d_r = dev.htod_copy(r).expect("htod r");
            let f = m
                .load_function("matvec_fp8_f32_residual")
                .expect("matvec_fp8_f32_residual");
            let mut b = dev.stream.launch_builder(&f);
            b.arg(&d_w)
                .arg(&d_x)
                .arg(&d_r)
                .arg(&mut d_out)
                .arg(&out_dim)
                .arg(&in_dim);
            unsafe { b.launch(cfg) }.expect("launch residual");
        }
        None => {
            let f = m.load_function("matvec_fp8_f32").expect("matvec_fp8_f32");
            let mut b = dev.stream.launch_builder(&f);
            b.arg(&d_w)
                .arg(&d_x)
                .arg(&mut d_out)
                .arg(&out_dim)
                .arg(&in_dim);
            unsafe { b.launch(cfg) }.expect("launch");
        }
    }
    dev.stream.synchronize().expect("sync");
    dev.dtoh_copy(&d_out).expect("dtoh")
}

/// The per-cell accuracy check, made non-vacuous.
///
/// A plain `max_abs`/`rel_l2` comparison silently passes NaN: `abs(NaN - w) > max_abs` is false, so
/// `max_abs` stays 0, and `den > 0.0` with `den = NaN` is false, so `rel_l2` falls back to 0. An all-NaN
/// kernel output would therefore satisfy both bounds. Every value is asserted finite, on BOTH sides, before
/// the bounds are computed, and an identically zero reference is refused.
fn check_cell(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    // Both sides finite, everywhere; without this an all-NaN kernel output passes the bounds below.
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            g.is_finite(),
            "{label}: device output [{k}] is {} — a non-finite result passes an unsigned diff silently",
            g
        );
        assert!(
            w.is_finite(),
            "{label}: reference [{k}] is {w} — the test DATA is wrong, not the kernel"
        );
    }
    let mut max_abs = 0.0f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (*g as f64 - *w).abs();
        max_abs = max_abs.max(d);
        num += d * d;
        den += w * w;
    }
    let rel_l2 = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let out_scale = got.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    // A reference that is identically zero would make rel_l2 meaningless, so refuse that too.
    assert!(
        den > 0.0,
        "{label}: the reference is identically zero — the cell proves nothing"
    );
    eprintln!(
        "  {label:<22} max_abs={max_abs:.3e}  rel_l2={rel_l2:.3e}  (|out|max={out_scale:.4})"
    );
    assert!(max_abs < 1e-3, "{label}: max_abs {max_abs:e} >= 1e-3");
    assert!(rel_l2 <= 1e-4, "{label}: rel_l2 {rel_l2:e} > 1e-4");
}

/// Test weights that are FINITE E4M3 codes only, and scale codes that are never NaN.
///
/// A ModelOpt export's FP8 weights contain no NaN code, so a random byte stream is the wrong model of it: 2 of
/// 256 E4M3 codes are NaN, so every 5120-wide row would contain one and the output would be NaN by
/// construction — testing the propagation of a value the artifact cannot contain.
fn finite_e4m3(byte: u8) -> u8 {
    if byte == 0x7F || byte == 0xFF {
        0x38 // 1.0, a neutral finite code
    } else {
        byte
    }
}

#[test]
fn fp8_matvec_matches_the_reference_on_every_shape() {
    let dev = CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU");
    let mut rng = Lcg(0x0FF8_2718_2818_2845);
    for &(out_dim, in_dim, scale) in CELLS {
        let (o, i) = (out_dim as usize, in_dim as usize);
        let weights: Vec<u8> = (0..o * i).map(|_| finite_e4m3(rng.next_u8())).collect();
        let x: Vec<f32> = (0..i).map(|k| ((k as f32) * 0.001 - 1.0) * 0.5).collect();
        let want = reference_dot(&weights, scale, &x, o, i);
        let got = run(&dev, &weights, scale, &x, None, out_dim, in_dim);
        check_cell(&format!("{out_dim}x{in_dim}"), &got, &want);
    }
}

/// The residual twin on a 5120 x 5120 cell at a large FP8 `weight_scale`.
#[test]
fn fp8_matvec_residual_twin_adds_the_residual() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let mut rng = Lcg(0x_51_20_20_27_18_28_45);
    let (out_dim, in_dim, scale) = (5120u32, 5120u32, 3.3e-3f32);
    let (o, i) = (out_dim as usize, in_dim as usize);
    let weights: Vec<u8> = (0..o * i).map(|_| finite_e4m3(rng.next_u8())).collect();
    let x: Vec<f32> = (0..i).map(|k| ((k as f32) * 0.0007 - 0.5) * 0.25).collect();
    let residual: Vec<f32> = (0..o).map(|k| (k as f32) * 0.01 - 5.0).collect();

    // The reference: the same dot, plus the residual, in f64.
    let base = reference_dot(&weights, scale, &x, o, i);
    let want: Vec<f64> = base
        .iter()
        .zip(residual.iter())
        .map(|(b, r)| b + (*r as f64))
        .collect();
    let got = run(&dev, &weights, scale, &x, Some(&residual), out_dim, in_dim);
    check_cell("5120x5120 + residual", &got, &want);

    // And the residual is actually ADDED, not ignored: the two arms must differ by exactly the residual.
    let plain = run(&dev, &weights, scale, &x, None, out_dim, in_dim);
    let mut worst = 0.0f32;
    for k in 0..o {
        worst = worst.max(((got[k] - plain[k]) - residual[k]).abs());
    }
    assert!(
        worst < 1e-3,
        "residual not applied faithfully: worst delta {worst:e}"
    );
}

/// A hand-computed cell: one row of 8 weights, scale 1.0, x all ones, so the result is the sum of the
/// decoded codes — checkable without the reference.
#[test]
fn fp8_matvec_hand_computed_cell() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    // E4M3: 0x38=1.0, 0x40=2.0, 0x3C=1.5, 0x00=+0.0, 0x80=-0.0, 0xB8=-1.0, 0x7E=448.0, 0x01=2^-9
    let codes: Vec<u8> = vec![0x38, 0x40, 0x3C, 0x00, 0x80, 0xB8, 0x7E, 0x01];
    let x = vec![1.0f32; 8];
    let got = run(&dev, &codes, 1.0, &x, None, 1, 8);
    // 1.0 + 2.0 + 1.5 + 0.0 + (-0.0) + (-1.0) + 448.0 + (1/512)
    let expect = 1.0f64 + 2.0 + 1.5 + 0.0 + (-0.0) + (-1.0) + 448.0 + (1.0 / 512.0);
    assert!(
        (got[0] as f64 - expect).abs() < 1e-3,
        "device {} vs hand-computed {expect}",
        got[0]
    );
}
