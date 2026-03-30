//! Integration test for CUDA attention_decode kernel.
//!
//! This test requires a CUDA-capable GPU and is gated behind the `cuda` feature.
//! It will fail on macOS (no NVIDIA GPU). Run on Modal or a CUDA machine:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_attention_test
//!
//! Test plan:
//! 1. Tiny attention: 1 head, head_dim=4, seq_len=3, hand-verified expected output
//! 2. GQA: 4 Q heads, 2 KV heads (gqa_ratio=2), seq_len=8, head_dim=8, vs CPU ref
//! 3. Single position: seq_len=1, output must equal V[0] exactly (softmax of 1 = 1.0)

#![cfg(feature = "cuda")]

use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::shaders::ATTENTION_KERNEL_SOURCE;

use cudarc::driver::{LaunchConfig, PushKernelArg};

/// CPU reference implementation of single-token multi-head attention with GQA.
///
/// q:           [num_heads * head_dim]
/// k_cache:     [num_kv_heads, max_seq_len, head_dim]  (only first seq_len positions valid)
/// v_cache:     same layout
/// Returns:     [num_heads * head_dim]
fn cpu_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let gqa_ratio = num_heads / num_kv_heads;
    let mut out = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let kv_base = kv_h * max_seq_len * head_dim;

        // Compute QK scores
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cache[kv_base + t * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax: max-subtract for numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        // Weighted V sum
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..seq_len {
                acc += scores[t] * v_cache[kv_base + t * head_dim + d];
            }
            out[h * head_dim + d] = acc;
        }
    }

    out
}

/// Compute the block size for the attention kernel: min(seq_len, 256), rounded
/// up to the nearest multiple of 32 (warp size), minimum 32.
fn attention_block_size(seq_len: u32) -> u32 {
    let bs = (seq_len as usize).min(256);
    let bs = ((bs + 31) / 32) * 32;
    bs.max(32) as u32
}

/// Launch attention_decode kernel and return output on host.
fn run_attention_kernel(
    device: &CudaDevice,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Vec<f32> {
    let module = device
        .compile_and_load(ATTENTION_KERNEL_SOURCE)
        .expect("Failed to compile attention kernel");
    let func = module
        .load_function("attention_decode")
        .expect("Failed to load attention_decode function");

    let q_gpu = device.htod_copy(q).expect("Failed to upload Q");
    let k_gpu = device.htod_copy(k_cache).expect("Failed to upload K cache");
    let v_gpu = device.htod_copy(v_cache).expect("Failed to upload V cache");
    let mut out_gpu = device
        .alloc_zeros::<f32>((num_heads * head_dim) as usize)
        .expect("Failed to allocate output");

    let block_size = attention_block_size(seq_len);

    // Dynamic shared memory: 8 floats for warp reductions + seq_len floats for scores
    let shared_bytes = (8 + seq_len) * 4;

    let launch_cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    // SAFETY: All buffers are correctly sized for the kernel parameters.
    // q_gpu: num_heads * head_dim elements
    // k_gpu/v_gpu: num_kv_heads * max_seq_len * head_dim elements
    // out_gpu: num_heads * head_dim elements
    // Shared memory: (8 + seq_len) * sizeof(float) bytes
    unsafe {
        device
            .stream
            .launch_builder(&func)
            .arg(&q_gpu)
            .arg(&k_gpu)
            .arg(&v_gpu)
            .arg(&mut out_gpu)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&seq_len)
            .arg(&max_seq_len)
            .arg(&scale)
            .launch(launch_cfg)
    }
    .expect("Kernel launch failed");

    device.synchronize().expect("Synchronize failed");
    device.dtoh_copy(&out_gpu).expect("Failed to download output")
}

#[test]
fn test_cuda_attention_tiny() {
    // 1 head, head_dim=4, seq_len=3, max_seq_len=8
    let num_heads = 1u32;
    let num_kv_heads = 1u32;
    let head_dim = 4u32;
    let seq_len = 3u32;
    let max_seq_len = 8u32;
    let scale = 1.0 / (head_dim as f32).sqrt(); // 0.5

    // Q vector for the single head
    let q = vec![1.0, 0.0, 1.0, 0.0];

    // K cache: 1 KV head, 8 max positions, head_dim=4 (only first 3 valid)
    let mut k_cache = vec![0.0f32; (max_seq_len * head_dim) as usize];
    // Position 0: [1, 0, 0, 0]
    k_cache[0] = 1.0;
    // Position 1: [0, 1, 0, 1]
    k_cache[4 + 1] = 1.0;
    k_cache[4 + 3] = 1.0;
    // Position 2: [1, 1, 1, 1]
    k_cache[8] = 1.0;
    k_cache[9] = 1.0;
    k_cache[10] = 1.0;
    k_cache[11] = 1.0;

    // V cache: same layout
    let mut v_cache = vec![0.0f32; (max_seq_len * head_dim) as usize];
    // Position 0: [10, 0, 0, 0]
    v_cache[0] = 10.0;
    // Position 1: [0, 20, 0, 0]
    v_cache[4 + 1] = 20.0;
    // Position 2: [0, 0, 30, 0]
    v_cache[8 + 2] = 30.0;

    // CPU reference
    let expected = cpu_attention(
        &q,
        &k_cache,
        &v_cache,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        seq_len as usize,
        max_seq_len as usize,
        scale,
    );

    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let result = run_attention_kernel(
        &device,
        &q,
        &k_cache,
        &v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
        scale,
    );

    assert_eq!(result.len(), expected.len());
    for i in 0..result.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "Mismatch at index {i}: GPU={}, CPU={}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_attention_gqa() {
    // 4 Q heads, 2 KV heads -> gqa_ratio=2
    // Heads 0,1 share KV head 0; heads 2,3 share KV head 1
    let num_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 8u32;
    let seq_len = 8u32;
    let max_seq_len = 16u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Deterministic pseudo-random data using a simple LCG
    let mut seed = 42u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let q: Vec<f32> = (0..(num_heads * head_dim)).map(|_| next_f32()).collect();

    // K cache: [num_kv_heads, max_seq_len, head_dim]
    let k_cache: Vec<f32> = (0..(num_kv_heads * max_seq_len * head_dim))
        .map(|_| next_f32())
        .collect();
    let v_cache: Vec<f32> = (0..(num_kv_heads * max_seq_len * head_dim))
        .map(|_| next_f32())
        .collect();

    let expected = cpu_attention(
        &q,
        &k_cache,
        &v_cache,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        seq_len as usize,
        max_seq_len as usize,
        scale,
    );

    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let result = run_attention_kernel(
        &device,
        &q,
        &k_cache,
        &v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
        scale,
    );

    assert_eq!(result.len(), expected.len());
    for i in 0..result.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "GQA mismatch at index {i}: GPU={}, CPU={}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_attention_single_position() {
    // seq_len=1: softmax of a single score = 1.0, so output = V[0] exactly
    let num_heads = 2u32;
    let num_kv_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 1u32;
    let max_seq_len = 4u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // K cache: [2, 4, 4] -- only position 0 matters
    let mut k_cache = vec![0.0f32; (num_kv_heads * max_seq_len * head_dim) as usize];
    // Head 0 position 0
    k_cache[0] = 0.5;
    k_cache[1] = 0.5;
    k_cache[2] = 0.5;
    k_cache[3] = 0.5;
    // Head 1 position 0
    let h1_base = (max_seq_len * head_dim) as usize;
    k_cache[h1_base] = -0.5;
    k_cache[h1_base + 1] = -0.5;
    k_cache[h1_base + 2] = -0.5;
    k_cache[h1_base + 3] = -0.5;

    // V cache: same layout
    let mut v_cache = vec![0.0f32; (num_kv_heads * max_seq_len * head_dim) as usize];
    // Head 0 position 0: [100, 200, 300, 400]
    v_cache[0] = 100.0;
    v_cache[1] = 200.0;
    v_cache[2] = 300.0;
    v_cache[3] = 400.0;
    // Head 1 position 0: [10, 20, 30, 40]
    v_cache[h1_base] = 10.0;
    v_cache[h1_base + 1] = 20.0;
    v_cache[h1_base + 2] = 30.0;
    v_cache[h1_base + 3] = 40.0;

    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let result = run_attention_kernel(
        &device,
        &q,
        &k_cache,
        &v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
        scale,
    );

    // With seq_len=1, softmax(score) = 1.0 regardless of score value.
    // Output for head 0 = V[0, 0] = [100, 200, 300, 400]
    // Output for head 1 = V[1, 0] = [10, 20, 30, 40]
    let expected = vec![100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0];
    assert_eq!(result.len(), expected.len());
    for i in 0..result.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "Single-position mismatch at {i}: GPU={}, expected={}",
            result[i],
            expected[i]
        );
    }
}
