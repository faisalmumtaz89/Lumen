//! Tests for the batched prefill RoPE CUDA kernel.
//!
//! Verifies that `rope_apply_batched` with batch=128 produces identical output
//! to 128 sequential calls to `rope_apply`, each at position pos_start + t.
//!
//! Requires a CUDA-capable GPU:
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_prefill_rope_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn create_context() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

/// CPU reference RoPE for a single token (interleaved pair layout).
///
/// Applies rotation in-place to `vec` which has layout [num_heads * head_dim].
fn cpu_rope_apply(
    vec: &mut [f32],
    pos: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;
    for h in 0..num_heads {
        let head_start = h * head_dim;
        for d in 0..half_dim {
            let freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let i0 = head_start + 2 * d;
            let i1 = head_start + 2 * d + 1;
            let x0 = vec[i0];
            let x1 = vec[i1];
            vec[i0] = x0 * cos_a - x1 * sin_a;
            vec[i1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// Run the single-token rope_apply kernel 128 times sequentially, returning
/// the concatenated Q and K results as reference output.
fn run_sequential_rope(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_flat: &[f32],      // [batch, q_dim]
    k_flat: &[f32],      // [batch, kv_dim]
    pos_start: u32,
    batch: usize,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let src = lumen_runtime::cuda::shaders::ROPE_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for rope.cu");
    let module = ctx.load_module(ptx).expect("load rope module");
    let func = module.load_function("rope_apply").expect("load rope_apply");

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let half_dim = head_dim / 2;
    let total_q_pairs = num_q_heads * half_dim;
    let total_k_pairs = num_kv_heads * half_dim;
    let max_pairs = total_q_pairs.max(total_k_pairs);
    let block_dim = 256u32;
    let grid_dim = max_pairs.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut q_out = q_flat.to_vec();
    let mut k_out = k_flat.to_vec();

    for t in 0..batch {
        let q_row = &q_out[t * q_dim..(t + 1) * q_dim];
        let k_row = &k_out[t * kv_dim..(t + 1) * kv_dim];
        let mut q_gpu = stream.clone_htod(q_row).unwrap();
        let mut k_gpu = stream.clone_htod(k_row).unwrap();
        let pos = pos_start + t as u32;

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&mut q_gpu)
                .arg(&mut k_gpu)
                .arg(&pos)
                .arg(&num_q_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&theta)
                .launch(cfg)
        }
        .expect("rope_apply launch failed");

        stream.synchronize().unwrap();
        let q_result = stream.clone_dtoh(&q_gpu).unwrap();
        let k_result = stream.clone_dtoh(&k_gpu).unwrap();
        q_out[t * q_dim..(t + 1) * q_dim].copy_from_slice(&q_result);
        k_out[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&k_result);
    }

    (q_out, k_out)
}

/// Run the batched rope_apply_batched kernel once on the full [batch, dim] matrices.
fn run_batched_rope(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_flat: &[f32],
    k_flat: &[f32],
    pos_start: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let src = lumen_runtime::cuda::shaders::PREFILL_ROPE_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for prefill_rope.cu");
    let module = ctx.load_module(ptx).expect("load prefill_rope module");
    let func = module
        .load_function("rope_apply_batched")
        .expect("load rope_apply_batched");

    let half_dim = head_dim / 2;
    let total_q_pairs = num_q_heads * half_dim;
    let total_work = batch * total_q_pairs;
    let block_dim = 256u32;
    let grid_dim = total_work.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut q_gpu = stream.clone_htod(q_flat).unwrap();
    let mut k_gpu = stream.clone_htod(k_flat).unwrap();

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut q_gpu)
            .arg(&mut k_gpu)
            .arg(&pos_start)
            .arg(&batch)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&theta)
            .launch(cfg)
    }
    .expect("rope_apply_batched launch failed");

    stream.synchronize().unwrap();
    let q_result = stream.clone_dtoh(&q_gpu).unwrap();
    let k_result = stream.clone_dtoh(&k_gpu).unwrap();
    (q_result, k_result)
}

/// Compare two f32 slices element-wise with tolerance, printing first N mismatches.
fn assert_f32_close(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut mismatches = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tol {
            if mismatches < 5 {
                eprintln!(
                    "  {label}[{i}]: batched={a:.8}, sequential={e:.8}, diff={diff:.2e}"
                );
            }
            mismatches += 1;
        }
        max_diff = max_diff.max(diff);
    }
    assert_eq!(
        mismatches, 0,
        "{label}: {mismatches} elements exceed tolerance {tol:.1e} (max_diff={max_diff:.2e})"
    );
}

/// Generate deterministic test data: non-trivial values that exercise the rotation.
fn generate_test_data(len: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(len);
    // Simple LCG PRNG for reproducibility without rand crate.
    let mut state = seed as u64;
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-1.0, 1.0].
        let val = ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
        data.push(val);
    }
    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Core acceptance test: batch=128 batched kernel matches 128 sequential calls.
///
/// Uses 4 Q heads, 4 KV heads, head_dim=8, theta=10000, pos_start=0.
/// This tests positions 0..127, covering the full standard prefill range.
#[test]
fn test_batched_rope_128_matches_sequential() {
    let (ctx, stream) = create_context();

    let batch: usize = 128;
    let num_q_heads: u32 = 4;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 8;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 0;

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 42);
    let k_data = generate_test_data(batch * kv_dim, 137);

    let (q_seq, k_seq) = run_sequential_rope(
        &ctx,
        &stream,
        &q_data,
        &k_data,
        pos_start,
        batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        theta,
    );

    let (q_bat, k_bat) = run_batched_rope(
        &ctx,
        &stream,
        &q_data,
        &k_data,
        pos_start,
        batch as u32,
        num_q_heads,
        num_kv_heads,
        head_dim,
        theta,
    );

    // Both kernels compute powf/cosf/sinf identically on the same GPU, so
    // results should be bit-identical. Use 1e-6 to allow for any reordering.
    assert_f32_close("Q batch=128", &q_bat, &q_seq, 1e-6);
    assert_f32_close("K batch=128", &k_bat, &k_seq, 1e-6);
}

/// Test with non-zero pos_start (continuation after prior context).
#[test]
fn test_batched_rope_nonzero_pos_start() {
    let (ctx, stream) = create_context();

    let batch: usize = 32;
    let num_q_heads: u32 = 2;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 4;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 50;

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 99);
    let k_data = generate_test_data(batch * kv_dim, 200);

    let (q_seq, k_seq) = run_sequential_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch,
        num_q_heads, num_kv_heads, head_dim, theta,
    );
    let (q_bat, k_bat) = run_batched_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch as u32,
        num_q_heads, num_kv_heads, head_dim, theta,
    );

    assert_f32_close("Q pos_start=50", &q_bat, &q_seq, 1e-6);
    assert_f32_close("K pos_start=50", &k_bat, &k_seq, 1e-6);
}

/// Test GQA: num_q_heads > num_kv_heads.
/// Only the first num_kv_heads * half_dim pairs should be rotated in K.
#[test]
fn test_batched_rope_gqa() {
    let (ctx, stream) = create_context();

    let batch: usize = 16;
    let num_q_heads: u32 = 8;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 8;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 0;

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 555);
    let k_data = generate_test_data(batch * kv_dim, 666);

    let (q_seq, k_seq) = run_sequential_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch,
        num_q_heads, num_kv_heads, head_dim, theta,
    );
    let (q_bat, k_bat) = run_batched_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch as u32,
        num_q_heads, num_kv_heads, head_dim, theta,
    );

    assert_f32_close("Q GQA 8q/2kv", &q_bat, &q_seq, 1e-6);
    assert_f32_close("K GQA 8q/2kv", &k_bat, &k_seq, 1e-6);
}

/// Test batch=1 (degenerate case matching single-token behavior).
#[test]
fn test_batched_rope_single_token() {
    let (ctx, stream) = create_context();

    let batch: usize = 1;
    let num_q_heads: u32 = 4;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 8;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 7;

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 12345);
    let k_data = generate_test_data(batch * kv_dim, 67890);

    let (q_seq, k_seq) = run_sequential_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch,
        num_q_heads, num_kv_heads, head_dim, theta,
    );
    let (q_bat, k_bat) = run_batched_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch as u32,
        num_q_heads, num_kv_heads, head_dim, theta,
    );

    assert_f32_close("Q batch=1", &q_bat, &q_seq, 1e-6);
    assert_f32_close("K batch=1", &k_bat, &k_seq, 1e-6);
}

/// Verify RoPE preserves vector magnitude for every pair (rotation invariant).
#[test]
fn test_batched_rope_preserves_magnitude() {
    let (ctx, stream) = create_context();

    let batch: usize = 64;
    let num_q_heads: u32 = 4;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 8;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 0;

    let q_dim = (num_q_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 77);
    let k_data = generate_test_data(batch * (num_kv_heads * head_dim) as usize, 88);

    let (q_result, _) = run_batched_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch as u32,
        num_q_heads, num_kv_heads, head_dim, theta,
    );

    // Check every pair in every token and head.
    let half_dim = (head_dim / 2) as usize;
    for t in 0..batch {
        for h in 0..num_q_heads as usize {
            for d in 0..half_dim {
                let base_before = t * q_dim + h * head_dim as usize + 2 * d;
                let x0_before = q_data[base_before];
                let x1_before = q_data[base_before + 1];
                let mag_before = (x0_before * x0_before + x1_before * x1_before).sqrt();

                let x0_after = q_result[base_before];
                let x1_after = q_result[base_before + 1];
                let mag_after = (x0_after * x0_after + x1_after * x1_after).sqrt();

                assert!(
                    (mag_before - mag_after).abs() < 1e-5,
                    "Magnitude not preserved at t={t} h={h} d={d}: before={mag_before}, after={mag_after}"
                );
            }
        }
    }
}

/// Cross-validate batched GPU kernel against CPU reference implementation.
#[test]
fn test_batched_rope_matches_cpu_reference() {
    let (ctx, stream) = create_context();

    let batch: usize = 128;
    let num_q_heads: u32 = 4;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 8;
    let theta: f32 = 10000.0;
    let pos_start: u32 = 0;

    let q_dim = (num_q_heads * head_dim) as usize;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_data = generate_test_data(batch * q_dim, 314);
    let k_data = generate_test_data(batch * kv_dim, 159);

    // GPU batched result.
    let (q_gpu, k_gpu) = run_batched_rope(
        &ctx, &stream, &q_data, &k_data, pos_start, batch as u32,
        num_q_heads, num_kv_heads, head_dim, theta,
    );

    // CPU reference: apply rope to each token independently.
    let mut q_cpu = q_data.clone();
    let mut k_cpu = k_data.clone();
    for t in 0..batch {
        let pos = pos_start as usize + t;
        cpu_rope_apply(
            &mut q_cpu[t * q_dim..(t + 1) * q_dim],
            pos,
            num_q_heads as usize,
            head_dim as usize,
            theta,
        );
        cpu_rope_apply(
            &mut k_cpu[t * kv_dim..(t + 1) * kv_dim],
            pos,
            num_kv_heads as usize,
            head_dim as usize,
            theta,
        );
    }

    // GPU powf/cosf/sinf may differ slightly from CPU libm, so use 1e-5.
    assert_f32_close("Q vs CPU", &q_gpu, &q_cpu, 1e-5);
    assert_f32_close("K vs CPU", &k_gpu, &k_cpu, 1e-5);
}
