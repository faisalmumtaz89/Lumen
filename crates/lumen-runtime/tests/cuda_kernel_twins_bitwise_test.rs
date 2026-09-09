//! Device-side byte-identity tests for kernel twins that claim the ORIGINAL kernel's output
//! bytes from a different launch geometry:
//!
//!   * `rmsnorm_to_q8_1_cta5` (ceil(blocks/warps) CTAs, one Q8_1 block per warp, the reduction
//!     repeated per CTA) against `rmsnorm_to_q8_1` (one block), at dims 2048 / 4096 / 5120 and
//!     a dim whose block count does not divide by the warp count;
//!   * `matvec_q4_0_dp4a_t160` (160 threads at K=5120) against `matvec_q4_0_dp4a` (256).
//!
//! Random inputs, the production compile options (the norm kernels at NVRTC's default target with
//! no options, the dp4a family at the explicit `compute_80` target with the raw `--use_fast_math`), and a
//! bit-for-bit comparison of every output byte. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_kernel_twins_bitwise_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// The norm kernels as production loads them (`decode::load_fn` -> `ffi::compile_and_load`):
/// NVRTC's default target, no options.
fn compile_norm(src: &str) -> cudarc::nvrtc::Ptx {
    compile_ptx(src).unwrap_or_else(|e| panic!("NVRTC compile failed: {e:?}"))
}

/// The dp4a family as production loads it (`decode::load_fn_sm80_fast_math` ->
/// `ffi::compile_and_load_with_arch_fast_math`): the explicit `compute_80` target and the raw
/// `--use_fast_math` flag (cudarc's `use_fast_math` field would add only `--fmad=true`).
fn compile_dp4a(src: &str) -> cudarc::nvrtc::Ptx {
    compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some("compute_80"),
            options: vec!["--use_fast_math".to_string()],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("NVRTC compile failed (compute_80): {e:?}"))
}

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

/// A uniform value in [-1, 1) with a full 24-bit mantissa, so a product of two such values is
/// NOT exactly representable in f32 and `sum_sq += val * val` differs between a fused and an
/// unfused multiply-add: a test input coarse enough to make `val * val` exact (9 bits) cannot
/// tell the two apart (review 2026-09-09, MAJOR-3).
fn rand_unit(s: &mut u64) -> f32 {
    ((rng_next(s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0
}

/// f16 bits of a positive f32 in the normal range (round-nearest-even on the fraction).
fn f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "test values must be f16-normal");
    let frac = bits & 0x7f_ffff;
    let mut f16 = ((exp as u32) << 10) | (frac >> 13);
    let round_bits = frac & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (f16 & 1) == 1) {
        f16 += 1;
    }
    f16 as u16
}

// ── rmsnorm_to_q8_1 vs rmsnorm_to_q8_1_cta5 ────────────────────────────────────────────────

/// The production block size (`decode::rmsnorm_block_size`) and the cta5 grid
/// (`decode::rmsnorm_q8_1_cta5_grid`), restated so the test stands on its own.
fn rmsnorm_block_size(dim: usize) -> u32 {
    (((dim.min(1024)) / 32) * 32).max(32) as u32
}

fn cta5_grid(dim: usize, block_size: u32) -> u32 {
    ((dim / 32) as u32)
        .div_ceil((block_size / 32).max(1))
        .max(1)
}

fn rmsnorm_q8_1_case(dim: usize, seed: u64) {
    let (ctx, stream) = create_context();
    let ptx = compile_norm(lumen_runtime::cuda::shaders::RMSNORM_Q8_1_KERNEL_SOURCE);
    let module = ctx.load_module(ptx).expect("load rmsnorm_q8_1 module");
    let single = module.load_function("rmsnorm_to_q8_1").unwrap();
    let cta5 = module.load_function("rmsnorm_to_q8_1_cta5").unwrap();

    let mut s = seed;
    let x: Vec<f32> = (0..dim).map(|_| rand_unit(&mut s) * 3.0).collect();
    let w: Vec<f32> = (0..dim).map(|_| 0.5 + rand_unit(&mut s).abs()).collect();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&w).unwrap();
    let eps = 1e-6f32;
    let dim_u32 = dim as u32;
    let block_size = rmsnorm_block_size(dim);
    let shared = (block_size / 32) * 4;
    let out_bytes = dim / 32 * 36;

    let mut outs: Vec<Vec<u8>> = Vec::new();
    for (f, grid) in [(&single, 1u32), (&cta5, cta5_grid(dim, block_size))] {
        // Poisoned, not zeroed: a block the twin never writes shows up as a mismatch.
        let poison = vec![0xA5u8; out_bytes];
        let mut out_gpu: CudaSlice<u8> = stream.clone_htod(&poison).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(&x_gpu)
                .arg(&w_gpu)
                .arg(&mut out_gpu)
                .arg(&eps)
                .arg(&dim_u32)
                .launch(cfg)
                .unwrap();
        }
        outs.push(stream.clone_dtoh(&out_gpu).unwrap());
    }
    assert!(
        outs[0].iter().any(|&b| b != 0xA5),
        "the single-block kernel wrote nothing at dim {dim}"
    );
    for (i, (a, b)) in outs[0].iter().zip(&outs[1]).enumerate() {
        assert_eq!(
            a, b,
            "dim {dim}: Q8_1 byte {i} (block {}, offset {}) differs: cta5 {b:#04x} != single {a:#04x}",
            i / 36,
            i % 36
        );
    }
}

#[test]
fn rmsnorm_to_q8_1_cta5_is_bitwise_the_single_block_kernel_at_5120() {
    rmsnorm_q8_1_case(5120, 1);
}

#[test]
fn rmsnorm_to_q8_1_cta5_is_bitwise_the_single_block_kernel_at_4096() {
    rmsnorm_q8_1_case(4096, 2);
}

#[test]
fn rmsnorm_to_q8_1_cta5_is_bitwise_the_single_block_kernel_at_2048() {
    rmsnorm_q8_1_case(2048, 3);
}

#[test]
fn rmsnorm_to_q8_1_cta5_covers_a_partial_final_cta() {
    // 5152 / 32 = 161 blocks over 32 warps: six CTAs, the last one owning a single block.
    rmsnorm_q8_1_case(5152, 4);
}

// ── matvec_q4_0_dp4a vs matvec_q4_0_dp4a_t160 ──────────────────────────────────────────────

const Q4_BLOCK_BYTES: usize = 18;
const Q8_1_BLOCK_BYTES: usize = 36;

fn random_q4_rows(out_dim: usize, in_dim: usize, s: &mut u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(out_dim * in_dim / 32 * Q4_BLOCK_BYTES);
    for _ in 0..out_dim * in_dim / 32 {
        let d = 0.002 + (rng_next(s) % 64) as f32 * 0.0005;
        bytes.extend_from_slice(&f16_bits(d).to_le_bytes());
        for _ in 0..16 {
            bytes.push((rng_next(s) % 256) as u8);
        }
    }
    bytes
}

fn random_q8_1(in_dim: usize, s: &mut u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(in_dim / 32 * Q8_1_BLOCK_BYTES);
    for _ in 0..in_dim / 32 {
        let q: Vec<i8> = (0..32)
            .map(|_| (rng_next(s) % 255) as i32 as i8)
            .map(|v| v.max(-127))
            .collect();
        let scale = 0.004 + (rng_next(s) % 64) as f32 * 0.001;
        let sum: i32 = q.iter().map(|&v| v as i32).sum();
        let weighted = scale * sum as f32;
        bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
        // the sum field may be negative: sign bit on top of the magnitude's f16
        let mag = f16_bits(weighted.abs().max(1e-4));
        let sum_bits = if weighted < 0.0 { mag | 0x8000 } else { mag };
        bytes.extend_from_slice(&sum_bits.to_le_bytes());
        bytes.extend(q.iter().map(|&v| v as u8));
    }
    bytes
}

fn q4_exactk_case(out_dim: usize, in_dim: usize, seed: u64) {
    let (ctx, stream) = create_context();
    let ptx = compile_dp4a(lumen_runtime::cuda::shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE);
    let module = ctx.load_module(ptx).expect("load matvec_q4_0_dp4a module");
    let full = module.load_function("matvec_q4_0_dp4a").unwrap();
    let t160 = module.load_function("matvec_q4_0_dp4a_t160").unwrap();

    let mut s = seed;
    let w = random_q4_rows(out_dim, in_dim, &mut s);
    let x = random_q8_1(in_dim, &mut s);
    let w_gpu = stream.clone_htod(&w).unwrap();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let out_u32 = out_dim as u32;
    let in_u32 = in_dim as u32;
    let grid = (out_dim as u32).div_ceil(4);

    let mut outs: Vec<Vec<f32>> = Vec::new();
    for (f, threads) in [(&full, 256u32), (&t160, 160u32)] {
        let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(&w_gpu)
                .arg(&x_gpu)
                .arg(&mut out_gpu)
                .arg(&out_u32)
                .arg(&in_u32)
                .launch(cfg)
                .unwrap();
        }
        outs.push(stream.clone_dtoh(&out_gpu).unwrap());
    }
    assert!(
        outs[0].iter().any(|v| *v != 0.0),
        "the 256-thread kernel produced all zeros"
    );
    for (i, (a, b)) in outs[0].iter().zip(&outs[1]).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {i}: t160 {b} != 256-thread {a} (out_dim={out_dim}, in_dim={in_dim})"
        );
    }
}

#[test]
fn matvec_q4_0_dp4a_t160_is_bitwise_the_256_thread_kernel_at_k5120() {
    q4_exactk_case(64, 5120, 21);
}

#[test]
fn matvec_q4_0_dp4a_t160_is_bitwise_at_k5120_with_a_ragged_row_count() {
    // 67 rows: the last CTA owns three rows and hits the out_dim guard.
    q4_exactk_case(67, 5120, 22);
}

// ── rmsnorm + rmsnorm_to_q8_1 vs rmsnorm_to_q8_1_cta5_normed ───────────────────────────────

fn rmsnorm_dual_case(dim: usize, seed: u64) {
    let (ctx, stream) = create_context();
    let q8_mod = ctx
        .load_module(compile_norm(
            lumen_runtime::cuda::shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
        ))
        .expect("load rmsnorm_q8_1 module");
    let norm_mod = ctx
        .load_module(compile_norm(
            lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE,
        ))
        .expect("load norm module");
    let plain = norm_mod.load_function("rmsnorm").unwrap();
    let single = q8_mod.load_function("rmsnorm_to_q8_1").unwrap();
    let dual = q8_mod.load_function("rmsnorm_to_q8_1_cta5_normed").unwrap();

    let mut s = seed;
    let x: Vec<f32> = (0..dim).map(|_| rand_unit(&mut s) * 3.0).collect();
    let w: Vec<f32> = (0..dim).map(|_| 0.5 + rand_unit(&mut s).abs()).collect();
    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&w).unwrap();
    let eps = 1e-6f32;
    let dim_u32 = dim as u32;
    let block_size = rmsnorm_block_size(dim);
    let shared = (block_size / 32) * 4;
    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared,
    };

    // the pair the dual kernel replaces
    let mut normed_ref: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();
    unsafe {
        stream
            .launch_builder(&plain)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut normed_ref)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(one)
            .unwrap();
    }
    let out_bytes = dim / 32 * 36;
    let mut q8_ref: CudaSlice<u8> = stream.clone_htod(&vec![0xA5u8; out_bytes]).unwrap();
    unsafe {
        stream
            .launch_builder(&single)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut q8_ref)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(one)
            .unwrap();
    }

    // the dual kernel
    let mut normed_dual: CudaSlice<f32> = stream.clone_htod(&vec![f32::NAN; dim]).unwrap();
    // different poison from q8_ref: two kernels that both wrote nothing must not compare equal
    let mut q8_dual: CudaSlice<u8> = stream.clone_htod(&vec![0x5Au8; out_bytes]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (cta5_grid(dim, block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared,
    };
    unsafe {
        stream
            .launch_builder(&dual)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut normed_dual)
            .arg(&mut q8_dual)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(cfg)
            .unwrap();
    }
    let (nr, nd) = (
        stream.clone_dtoh(&normed_ref).unwrap(),
        stream.clone_dtoh(&normed_dual).unwrap(),
    );
    for (i, (a, b)) in nr.iter().zip(&nd).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "dim {dim}: normed[{i}] dual {b} != plain {a}"
        );
    }
    let (qr, qd) = (
        stream.clone_dtoh(&q8_ref).unwrap(),
        stream.clone_dtoh(&q8_dual).unwrap(),
    );
    assert!(
        qr.iter().any(|&b| b != 0xA5),
        "the single-block kernel wrote nothing at dim {dim}"
    );
    assert!(
        qd.iter().any(|&b| b != 0x5A),
        "the dual kernel wrote nothing at dim {dim}"
    );
    assert!(
        nd.iter().all(|v| !v.is_nan()),
        "the dual kernel left part of normed unwritten at dim {dim}"
    );
    for (i, (a, b)) in qr.iter().zip(&qd).enumerate() {
        assert_eq!(
            a, b,
            "dim {dim}: Q8_1 byte {i} dual {b:#04x} != single {a:#04x}"
        );
    }
}

#[test]
fn rmsnorm_to_q8_1_cta5_normed_is_bitwise_both_kernels_it_replaces_at_5120() {
    rmsnorm_dual_case(5120, 31);
}

#[test]
fn rmsnorm_to_q8_1_cta5_normed_is_bitwise_both_kernels_it_replaces_at_2048() {
    rmsnorm_dual_case(2048, 32);
}

#[test]
fn rmsnorm_to_q8_1_cta5_normed_covers_a_partial_final_cta() {
    // 5152 / 32 = 161 blocks over 32 warps: six CTAs, the last one owning a single block; the
    // plain rmsnorm writes all 5152 normalized values and so must the dual kernel.
    rmsnorm_dual_case(5152, 33);
}
