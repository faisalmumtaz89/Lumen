//! A Q8_0 plane resident only in the split (SoA) layout dequantizes to the same F16
//! and F32 tiles as its raw AoS plane: `dequant_q8_split_to_f16` / `_to_f32` against
//! `dequant_q8_0_to_f16` / `_to_f32` on the same random planes, bit for bit, over the
//! 27B's projection shapes plus a short odd shape. This is what lets the split-clone
//! pass release the raw plane.
//!
//! Requires a CUDA GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_q8_split_dequant_test

#![cfg(feature = "cuda")]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use lumen_runtime::cuda::shaders::{
    DEQUANT_Q8_0_KERNEL_SOURCE, DEQUANT_Q8_SPLIT_KERNEL_SOURCE,
    REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE,
};
use std::sync::Arc;

fn compile(ctx: &Arc<CudaContext>, src: &str) -> Ptx {
    // The highest target this device can load (a PTX for a newer target compiles but cannot load).
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let cc = (major * 10 + minor) as u32;
    let mut last = String::new();
    for (arch, arch_cc) in [
        ("compute_121", 121u32),
        ("compute_120", 120),
        ("compute_110", 110),
        ("compute_103", 103),
        ("compute_100", 100),
        ("compute_90", 90),
        ("compute_89", 89),
        ("compute_88", 88),
        ("compute_87", 87),
        ("compute_86", 86),
        ("compute_80", 80),
        ("compute_75", 75),
        ("compute_72", 72),
        ("compute_70", 70),
    ] {
        if arch_cc > cc {
            continue;
        }
        match compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some(arch),
                ..Default::default()
            },
        ) {
            Ok(p) => return p,
            Err(e) => last = format!("{arch}: {e}"),
        }
    }
    panic!("no compile target for cc {cc}: {last}");
}

fn load(ctx: &Arc<CudaContext>, ptx: Ptx, names: &[&str]) -> (Arc<CudaModule>, Vec<CudaFunction>) {
    let module = ctx.load_module(ptx).expect("load module");
    let fns = names
        .iter()
        .map(|n| {
            module
                .load_function(n)
                .unwrap_or_else(|e| panic!("{n}: {e}"))
        })
        .collect();
    (module, fns)
}

fn rng_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// A random Q8_0 plane [rows x in_dim]: f16 scale + 32 int8 per block, 34 bytes.
fn random_q8_plane(rows: usize, in_dim: usize, seed: u64) -> Vec<u8> {
    let nb = in_dim / 32;
    let mut out = Vec::with_capacity(rows * nb * 34);
    let mut st = seed;
    for _ in 0..rows * nb {
        // scale: a random f16 bit pattern with a finite value, sign either way, including
        // subnormals (exponent 0) so the convert path is exercised end to end
        let mut bits = (rng_next(&mut st) & 0xffff) as u16;
        if (bits & 0x7c00) == 0x7c00 {
            bits &= 0x7bff; // no inf / NaN scales
        }
        out.extend_from_slice(&bits.to_le_bytes());
        for _ in 0..32 {
            out.push((rng_next(&mut st) & 0xff) as u8);
        }
    }
    out
}

fn launch_config(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test]
fn split_layout_dequant_is_bit_identical_to_the_raw_dequant() {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream: Arc<CudaStream> = ctx.default_stream();
    let (_m0, raw_fns) = load(
        &ctx,
        compile(&ctx, DEQUANT_Q8_0_KERNEL_SOURCE),
        &["dequant_q8_0_to_f16", "dequant_q8_0_to_f32"],
    );
    let (_m1, split_fns) = load(
        &ctx,
        compile(&ctx, DEQUANT_Q8_SPLIT_KERNEL_SOURCE),
        &["dequant_q8_split_to_f16", "dequant_q8_split_to_f32"],
    );
    let (_m2, repack) = load(
        &ctx,
        compile(&ctx, REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE),
        &["repack_q8_raw_to_split"],
    );

    // (rows, in_dim): the 27B's GDN qkv / z-gate / ssm_out / attention projections, and a
    // short odd-row shape whose nb is even but small.
    let shapes = [
        (10240usize, 5120usize),
        (6144, 5120),
        (5120, 6144),
        (1024, 5120),
        (7, 64),
    ];
    for (i, &(rows, in_dim)) in shapes.iter().enumerate() {
        let nb = in_dim / 32;
        assert_eq!(nb % 2, 0, "the split layout needs an even block count");
        let n = rows * in_dim;
        let raw_host = random_q8_plane(rows, in_dim, 0x5eed_0000 + i as u64);
        let raw = stream.clone_htod(&raw_host).unwrap();
        let mut split = stream.alloc_zeros::<u8>(raw_host.len()).unwrap();

        let total_blocks = (rows * nb) as u32;
        let nb_u32 = nb as u32;
        let rows_u32 = rows as u32;
        unsafe {
            stream
                .launch_builder(&repack[0])
                .arg(&raw)
                .arg(&mut split)
                .arg(&nb_u32)
                .arg(&rows_u32)
                .launch(launch_config(total_blocks as usize))
                .unwrap();
        }

        let n_u32 = n as u32;
        let in_dim_u32 = in_dim as u32;
        let mut f16_raw = stream.alloc_zeros::<u16>(n).unwrap();
        let mut f16_split = stream.alloc_zeros::<u16>(n).unwrap();
        let mut f32_raw = stream.alloc_zeros::<f32>(n).unwrap();
        let mut f32_split = stream.alloc_zeros::<f32>(n).unwrap();
        unsafe {
            stream
                .launch_builder(&raw_fns[0])
                .arg(&raw)
                .arg(&mut f16_raw)
                .arg(&n_u32)
                .launch(launch_config(n))
                .unwrap();
            stream
                .launch_builder(&raw_fns[1])
                .arg(&raw)
                .arg(&mut f32_raw)
                .arg(&n_u32)
                .launch(launch_config(n))
                .unwrap();
            stream
                .launch_builder(&split_fns[0])
                .arg(&split)
                .arg(&mut f16_split)
                .arg(&n_u32)
                .arg(&in_dim_u32)
                .launch(launch_config(n))
                .unwrap();
            stream
                .launch_builder(&split_fns[1])
                .arg(&split)
                .arg(&mut f32_split)
                .arg(&n_u32)
                .arg(&in_dim_u32)
                .launch(launch_config(n))
                .unwrap();
        }
        stream.synchronize().unwrap();
        let (a16, b16): (Vec<u16>, Vec<u16>) = (
            stream.clone_dtoh(&f16_raw).unwrap(),
            stream.clone_dtoh(&f16_split).unwrap(),
        );
        let (a32, b32): (Vec<f32>, Vec<f32>) = (
            stream.clone_dtoh(&f32_raw).unwrap(),
            stream.clone_dtoh(&f32_split).unwrap(),
        );
        let bad16 = a16.iter().zip(&b16).filter(|(x, y)| x != y).count();
        let bad32 = a32
            .iter()
            .zip(&b32)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        // the plane is not degenerate: some scales are non-zero and so are some tiles
        let nonzero = a32.iter().filter(|v| **v != 0.0).count();
        println!(
            "[q8 split dequant] [{rows} x {in_dim}]: f16 mismatches {bad16}, f32 mismatches {bad32}, non-zero {nonzero}/{n}"
        );
        assert!(
            nonzero > n / 2,
            "[{rows} x {in_dim}] degenerate fixture: non-zero {nonzero} <= {}",
            n / 2
        );
        assert_eq!(
            bad16, 0,
            "[{rows} x {in_dim}] F16 tile differs between raw and split layouts"
        );
        assert_eq!(
            bad32, 0,
            "[{rows} x {in_dim}] F32 tile differs between raw and split layouts"
        );
    }
}
