//! The CUDA NVFP4 and FP8 plane decoders (`dequant_nvfp4_to_f32`, `dequant_fp8_to_f32`, the kernels the
//! planar prefill route runs) against the host decoder in `lumen_format::planar_dequant`, bit for bit.
//!
//! Every output is compared as raw `u32` bits with no tolerance: one differing bit fails, and a NaN must
//! decode to a NaN.
//!
//! Coverage:
//!   * all 16 E2M1 codes, including the two zero codes (both +0.0 in the host decoder) and the subnormal 0x1;
//!   * all 256 E4M3 codes; `0x7F`/`0xFF` asserted `is_nan()`; `+0.0` and `-0.0` bit-pinned;
//!   * block scales at the edges — 0.0, 448.0 (the largest finite E4M3), the smallest normal, the smallest
//!     subnormal and 1.0 — under the smallest and largest global scale of the reference-vector fixture and
//!     a wide global scale that drives the decoded magnitude towards the F16 bound;
//!   * all-zero planes and alternating nibbles (0x0F/0xF0 patterns, which catch a nibble-order swap);
//!   * multi-group NVFP4 planes (65 and 1000 groups), so each thread's group index, the block-scale offset
//!     and the global scale after the block scales are checked past the first group.
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_planar_decode_exact_test
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use lumen_format::planar_dequant::{dequantize_fp8, dequantize_nvfp4, NVFP4_GROUP};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::{DEQUANT_FP8_KERNEL_SOURCE, DEQUANT_NVFP4_KERNEL_SOURCE};

/// The global-scale range spanned by the three `*.globalscale.f32` files of the reference-vector fixture that
/// `lumen_format::planar_dequant`'s tests decode: 3.252466e-05 .. 3.8680798e-04.
const GS_MIN: f32 = 3.252_466e-05;
const GS_MAX: f32 = 3.868_079_8e-04;
/// A deliberately WIDE global scale, used with the largest block scale and the largest codes so the decoded
/// magnitude actually approaches the F16 bound instead of sitting at ~1.04. The fixture's own scales are
/// tame; this is what makes the bound assertion mean something.
const GS_WIDE: f32 = 24.0;

/// The F16 bound on a decoded value. At the fixture's largest global scale the decoded magnitude is
/// `e2m1_max * e4m3_max * gs = 6.0 * 448.0 * 3.8680798e-04 = 1.04`, so the bound is not tight here; it is
/// asserted as the format's actual F16 maximum so an out-of-range value from a bad scale still fails.
const F16_MAX: f32 = 65504.0;

fn bits(v: f32) -> u32 {
    v.to_bits()
}

/// Compare two f32 by BITS, with NaN matching NaN (a NaN's payload can differ between an ALU path and a
/// hand-built one, and the oracle's NaN is `f32::NAN`, so compare `is_nan()` first).
fn bit_eq(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || bits(a) == bits(b)
}

fn first_bit_diff(got: &[f32], want: &[f32], label: &str) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{label}: length {} != {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if !bit_eq(*g, *w) {
            return Some(format!(
                "{label}: index {i}: device {:#010x} ({g:e}) vs oracle {:#010x} ({w:e})",
                bits(*g),
                bits(*w)
            ));
        }
    }
    None
}

fn device() -> CudaDevice {
    CudaDevice::new(0).expect("CUDA device 0 — this test needs a GPU")
}

/// Run the NVFP4 decoder over `n_blocks` 16-weight blocks.
fn run_nvfp4(
    dev: &CudaDevice,
    packed: &[u8],
    scales: &[u8],
    global: f32,
    n_blocks: u32,
) -> Vec<f32> {
    // The kernel takes ONE plane: weight | block_scale | global_scale(F32 LE), the converter's order
    // (`convert_hf.rs::lower_nvfp4`). Building it here makes the test exercise the real layout, including
    // the scale after the block scales, rather than a simplified one.
    let mut plane = Vec::with_capacity(packed.len() + scales.len() + 8);
    plane.extend_from_slice(packed);
    plane.extend_from_slice(scales);
    plane.extend_from_slice(&global.to_le_bytes());
    // The activation scale a converted slice may carry after its planes: NaN, so a kernel that read it
    // would fail every comparison here.
    plane.extend_from_slice(&f32::NAN.to_le_bytes());
    let d_plane = dev.htod_copy(&plane).expect("htod plane");
    let mut d_out = dev
        .alloc_zeros::<f32>((n_blocks as usize) * NVFP4_GROUP)
        .expect("alloc out");
    let m = dev
        .compile_and_load(DEQUANT_NVFP4_KERNEL_SOURCE)
        .expect("compile nvfp4 kernel");
    let f = m
        .load_function("dequant_nvfp4_to_f32")
        .expect("dequant_nvfp4_to_f32");
    // One thread per 4 weights, as the prefill launcher sizes it (`launch_dequant_plane_to_f32`).
    let n_elements = n_blocks * NVFP4_GROUP as u32;
    let block = 256u32;
    let grid = (n_elements / 4).div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&d_plane).arg(&mut d_out).arg(&n_elements);
    unsafe { b.launch(cfg) }.expect("launch nvfp4");
    dev.stream.synchronize().expect("sync");
    dev.dtoh_copy(&d_out).expect("dtoh out")
}

fn run_fp8(dev: &CudaDevice, weights: &[u8], scale: f32) -> Vec<f32> {
    let n = weights.len() as u32;
    // FP8's plane is weight[n] | global_scale(F32 LE), the same rule
    // (`convert_hf.rs::lower_fp8`).
    let mut plane = Vec::with_capacity(weights.len() + 8);
    plane.extend_from_slice(weights);
    plane.extend_from_slice(&scale.to_le_bytes());
    // The activation scale a converted slice may carry after its planes: NaN, so a kernel that read it
    // would fail every comparison here.
    plane.extend_from_slice(&f32::NAN.to_le_bytes());
    let d_w = dev.htod_copy(&plane).expect("htod plane");
    let mut d_out = dev.alloc_zeros::<f32>(n as usize).expect("alloc out");
    let m = dev
        .compile_and_load(DEQUANT_FP8_KERNEL_SOURCE)
        .expect("compile fp8 kernel");
    let f = m
        .load_function("dequant_fp8_to_f32")
        .expect("dequant_fp8_to_f32");
    let threads = 256u32;
    let grid = n.div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&d_w).arg(&mut d_out).arg(&n);
    unsafe { b.launch(cfg) }.expect("launch fp8");
    dev.stream.synchronize().expect("sync");
    dev.dtoh_copy(&d_out).expect("dtoh out")
}

/// All 16 E2M1 codes, under a scale of 1.0 so the code's own value is what is compared.
#[test]
fn nvfp4_decode_all_16_e2m1_codes_bit_exact() {
    let dev = device();
    // One block of 16 weights: byte i holds code i in its LOW nibble, high nibble 0 (also +0.0).
    let mut packed = vec![0u8; 8];
    for i in 0..8u8 {
        packed[i as usize] = i; // low nibble = code i
    }
    let mut packed2 = vec![0u8; 8];
    for i in 0..8u8 {
        packed2[i as usize] = (8 + i) << 4; // high nibble = codes 8..15
    }
    // E4M3 0x38 = 1.0 (exp 7 -> 2^0), so the block scale is exactly 1.0.
    let scales = vec![0x38u8];
    let global = 1.0f32;

    let got = run_nvfp4(&dev, &packed, &scales, global, 1);
    let want = dequantize_nvfp4(&packed, &scales[..], global).expect("oracle");
    if let Some(e) = first_bit_diff(&got, &want, "e2m1 low nibbles") {
        panic!("{e}");
    }
    let got2 = run_nvfp4(&dev, &packed2, &scales, global, 1);
    let want2 = dequantize_nvfp4(&packed2, &scales[..], global).expect("oracle");
    if let Some(e) = first_bit_diff(&got2, &want2, "e2m1 high nibbles") {
        panic!("{e}");
    }
    // Indexing, so the assertions below are about the right element: each byte carries its code in the LOW
    // nibble and 0x0 in the high one, and the decoder emits (lo, hi) per byte. So code 0x1 — the subnormal —
    // lands at output index 2, not index 1.
    assert_eq!(
        bits(got[0]),
        bits(0.0),
        "byte 0 low nibble (code 0x0) must be +0.0"
    );
    assert_eq!(
        bits(got[1]),
        bits(0.0),
        "byte 0 high nibble (code 0x0) must be +0.0"
    );
    assert_eq!(
        bits(got[2]),
        bits(0.5),
        "byte 1 low nibble (code 0x1, subnormal) must be 0.5"
    );
    assert_eq!(
        bits(got[3]),
        bits(0.0),
        "byte 1 high nibble (code 0x0) must be +0.0"
    );
    assert!(got.iter().all(|v| v.abs() <= F16_MAX), "F16 bound");
}

/// The edge block scales under the fixture's smallest and largest global scale and the wide one.
#[test]
fn nvfp4_decode_edge_block_and_global_scales_bit_exact() {
    let dev = device();
    // Alternating nibbles: 0x0F and 0xF0 across the block, so a nibble-order swap cannot pass.
    let packed: Vec<u8> = vec![0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0];
    let all_zero = vec![0u8; 8];
    // 0x00 = +0.0; 0x7E = 448.0; 0x08 = smallest normal (2^-6); 0x01 = smallest subnormal; 0x38 = 1.0.
    let edge_scales: [u8; 5] = [0x00, 0x7E, 0x08, 0x01, 0x38];
    for s in edge_scales {
        for gs in [GS_MIN, GS_MAX, GS_WIDE] {
            let scales = vec![s];
            let got = run_nvfp4(&dev, &packed, &scales, gs, 1);
            let want = dequantize_nvfp4(&packed, &scales[..], gs).expect("oracle");
            if let Some(e) = first_bit_diff(&got, &want, &format!("scale {s:#04x} gs {gs:e}")) {
                panic!("{e}");
            }
            let got0 = run_nvfp4(&dev, &all_zero, &scales, gs, 1);
            let want0 = dequantize_nvfp4(&all_zero, &scales[..], gs).expect("oracle");
            if let Some(e) = first_bit_diff(&got0, &want0, &format!("all-zero scale {s:#04x}")) {
                panic!("{e}");
            }
            assert!(
                got.iter().all(|v| v.abs() <= F16_MAX),
                "F16 bound at scale {s:#04x} gs {gs:e}: max {:e}",
                got.iter().fold(0.0f32, |m, v| m.max(v.abs()))
            );
        }
    }
}

/// Planes of many 16-weight groups, each with its own block scale: every group is decoded with its own
/// scale and the global scale read after the block scales, as in a real matrix.
#[test]
fn nvfp4_decode_many_groups_bit_exact() {
    let dev = device();
    // A fixed linear congruential sequence, so every nibble code and a spread of finite block scales occur.
    let mut state = 0x2545_f491u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 24) as u8
    };
    for n_blocks in [65u32, 1000] {
        let packed: Vec<u8> = (0..n_blocks * 8).map(|_| next()).collect();
        // Block scales from 0x01 (the smallest subnormal) to 0x7E (448.0), never a NaN code.
        let scales: Vec<u8> = (0..n_blocks).map(|_| 1 + next() % 0x7E).collect();
        for gs in [GS_MIN, GS_MAX] {
            let got = run_nvfp4(&dev, &packed, &scales, gs, n_blocks);
            let want = dequantize_nvfp4(&packed, &scales, gs).expect("oracle");
            if let Some(e) = first_bit_diff(&got, &want, &format!("{n_blocks} groups gs {gs:e}")) {
                panic!("{e}");
            }
        }
    }
}

/// All 256 E4M3 codes in one plane — the 254 finite ones, the two NaNs and both signed zeros — under four
/// scales.
#[test]
fn fp8_decode_all_e4m3_codes_bit_exact() {
    let dev = device();
    let all: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
    for scale in [1.0f32, GS_MIN, GS_MAX, GS_WIDE] {
        let got = run_fp8(&dev, &all, scale);
        let want = dequantize_fp8(&all, scale);
        if let Some(e) = first_bit_diff(&got, &want, &format!("fp8 scale {scale:e}")) {
            panic!("{e}");
        }
        // The two NaNs, explicitly.
        for nan_code in [0x7Fu8, 0xFFu8] {
            assert!(
                got[nan_code as usize].is_nan(),
                "{nan_code:#04x} must decode to NaN, got {:e}",
                got[nan_code as usize]
            );
        }
        // -0.0 preserved, bit for bit.
        assert_eq!(bits(got[0x80]), bits(-0.0), "0x80 must be -0.0");
        assert_eq!(bits(got[0x00]), bits(0.0), "0x00 must be +0.0");
        // The largest finite value is 448.0 at scale 1.0.
        if scale == 1.0 {
            assert_eq!(got[0x7E], 448.0, "0x7E must be 448.0");
        }
    }
}

/// Every E2M1 code, filling a whole block in both nibbles, at every edge block scale and every global
/// scale: the decode arithmetic across the scale range, not merely the happy path.
#[test]
fn nvfp4_decode_every_code_at_every_edge_scale_bit_exact() {
    let dev = device();
    for code in 0u8..16 {
        for s in [0x00u8, 0x7E, 0x08, 0x01, 0x38] {
            let packed = vec![code, code, code, code, code, code, code, code];
            let scales = vec![s];
            for gs in [GS_MIN, GS_MAX, GS_WIDE] {
                let got = run_nvfp4(&dev, &packed, &scales, gs, 1);
                let want = dequantize_nvfp4(&packed, &scales[..], gs).expect("oracle");
                if let Some(e) = first_bit_diff(
                    &got,
                    &want,
                    &format!("code {code:#03x} scale {s:#04x} gs {gs:e}"),
                ) {
                    panic!("{e}");
                }
            }
        }
    }
}

/// Every E4M3 byte decoded alone, as a one-element plane, at scale 1.0.
#[test]
fn fp8_decode_every_byte_alone_bit_exact() {
    let dev = device();
    for b in 0u16..=255 {
        let w = vec![b as u8];
        let got = run_fp8(&dev, &w, 1.0);
        let want = dequantize_fp8(&w, 1.0);
        if let Some(e) = first_bit_diff(&got, &want, &format!("byte {b:#04x}")) {
            panic!("{e}");
        }
    }
}
