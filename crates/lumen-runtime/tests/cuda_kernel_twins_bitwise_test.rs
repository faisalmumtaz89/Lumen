//! Device-side byte-identity tests for kernel twins that claim the ORIGINAL kernel's output
//! bytes from a different launch geometry or a different compile target:
//!
//!   * `rmsnorm_to_q8_1_cta5` (ceil(blocks/warps) CTAs, one Q8_1 block per warp, the reduction
//!     repeated per CTA) against `rmsnorm_to_q8_1` (one block), at dims 2048 / 4096 / 5120 and
//!     a dim whose block count does not divide by the warp count;
//!   * `rmsnorm_to_q8_1_cta5_normed` (the dual-output launch) against the plain `rmsnorm` and
//!     `rmsnorm_to_q8_1` pair it replaces, at dims 2048 / 4096 / 5120 / 5152.
//!
//! Random inputs, the production compile options (NVRTC's default target with no options), and
//! a bit-for-bit comparison of every output byte. The decode-attention kernel's cross-target
//! identity lives in `cuda_attention_decode_fixture_test.rs`. Requires a CUDA GPU:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_kernel_twins_bitwise_test
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// The norm kernels as production loads them (`decode::load_fn` -> `ffi::compile_and_load`):
/// NVRTC's default target, no options.
fn compile_norm(src: &str) -> cudarc::nvrtc::Ptx {
    compile_ptx(src).unwrap_or_else(|e| panic!("NVRTC compile failed: {e:?}"))
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
/// tell the two apart.
fn rand_unit(s: &mut u64) -> f32 {
    ((rng_next(s) & 0xff_ffff) as f32 / 8_388_608.0) - 1.0
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
fn rmsnorm_to_q8_1_cta5_normed_is_bitwise_both_kernels_it_replaces_at_4096() {
    // Qwen3.5-9B's hidden width; a 9B Q4_0 artifact with F32 GDN alpha/beta takes the dual there.
    rmsnorm_dual_case(4096, 34);
}

#[test]
fn rmsnorm_to_q8_1_cta5_normed_covers_a_partial_final_cta() {
    // 5152 / 32 = 161 blocks over 32 warps: six CTAs, the last one owning a single block; the
    // plain rmsnorm writes all 5152 normalized values and so must the dual kernel.
    rmsnorm_dual_case(5152, 33);
}
