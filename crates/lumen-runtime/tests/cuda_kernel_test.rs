//! Integration tests for CUDA elementwise kernels.
//!
//! Tests RMSNorm, SwiGLU, residual_add, softmax, and RoPE kernels against
//! CPU reference implementations. Requires a CUDA-capable GPU (run on Modal).
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_kernel_test

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::driver::{CudaContext, CudaSlice};

/// Compile CUDA source, allocate device buffers, launch kernel, read back.
/// All tests follow this pattern using cudarc directly (the integration test
/// exercises kernel correctness, not the CudaDevice wrapper).

fn create_context() -> (std::sync::Arc<CudaContext>, std::sync::Arc<cudarc::driver::CudaStream>) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

// ---------------------------------------------------------------------------
// RMSNorm tests
// ---------------------------------------------------------------------------

/// CPU reference: rmsnorm(x, weight, eps) -> out
fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * scale * wi).collect()
}

#[test]
fn test_cuda_rmsnorm_basic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for norm.cu");
    let module = ctx.load_module(ptx).expect("Failed to load norm module");
    let func = module.load_function("rmsnorm").expect("Failed to load rmsnorm");

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps: f32 = 1e-5;
    let dim: u32 = 4;
    let expected = cpu_rmsnorm(&x, &weight, eps);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(4).unwrap();

    // Single block, 32 threads (minimum warp), shared_mem = 1 warp * 4 bytes.
    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) as u32 * 4,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&x_gpu)
            .arg(&w_gpu)
            .arg(&mut out_gpu)
            .arg(&eps)
            .arg(&dim)
            .launch(cfg)
    }
    .expect("rmsnorm launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "rmsnorm[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_rmsnorm_with_weights() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rmsnorm").unwrap();

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![0.5f32, 2.0, 0.1, 3.0];
    let eps: f32 = 1e-5;
    let dim: u32 = 4;
    let expected = cpu_rmsnorm(&x, &weight, eps);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(4).unwrap();

    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) as u32 * 4,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&x_gpu).arg(&w_gpu).arg(&mut out_gpu)
            .arg(&eps).arg(&dim)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "rmsnorm_weighted[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_rmsnorm_zeros() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rmsnorm").unwrap();

    let x = vec![0.0f32; 4];
    let weight = vec![1.0f32; 4];
    let eps: f32 = 1e-5;
    let dim: u32 = 4;

    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(4).unwrap();

    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) as u32 * 4,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&x_gpu).arg(&w_gpu).arg(&mut out_gpu)
            .arg(&eps).arg(&dim)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..4 {
        assert!(result[i].is_finite(), "rmsnorm_zeros[{i}] should be finite");
        assert_eq!(result[i], 0.0, "rmsnorm of zero input should be zero");
    }
}

#[test]
fn test_cuda_rmsnorm_large_dim() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::NORM_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rmsnorm").unwrap();

    let dim = 4096usize;
    let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let weight: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let eps: f32 = 1e-5;
    let expected = cpu_rmsnorm(&x, &weight, eps);

    let x_gpu = stream.clone_htod(&x).unwrap();
    let w_gpu = stream.clone_htod(&weight).unwrap();
    let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();

    // Use 256 threads for realistic workload.
    let block_dim = 256u32;
    let num_warps = block_dim / 32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: num_warps * 4,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&x_gpu).arg(&w_gpu).arg(&mut out_gpu)
            .arg(&eps).arg(&(dim as u32))
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&out_gpu).unwrap();

    for i in 0..dim {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "rmsnorm_large[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// SwiGLU tests
// ---------------------------------------------------------------------------

/// CPU reference: swiglu_inplace(gate, up) -> gate (modified)
fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect()
}

#[test]
fn test_cuda_swiglu() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for activations.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("swiglu_inplace").unwrap();

    let gate = vec![0.0f32, 1.0, -1.0, 2.5, -3.0];
    let up = vec![1.0f32, 1.0, 1.0, 2.0, 0.5];
    let n: u32 = 5;
    let expected = cpu_swiglu(&gate, &up);

    let mut gate_gpu = stream.clone_htod(&gate).unwrap();
    let up_gpu = stream.clone_htod(&up).unwrap();

    let block_dim = 256u32;
    let grid_dim = n.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut gate_gpu)
            .arg(&up_gpu)
            .arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&gate_gpu).unwrap();

    for i in 0..n as usize {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "swiglu[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_swiglu_known_values() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("swiglu_inplace").unwrap();

    // silu(0) * 1 = 0, silu(1) * 1 = 0.7310586, silu(-1) * 1 = -0.2689414
    let gate = vec![0.0f32, 1.0, -1.0];
    let up = vec![1.0f32, 1.0, 1.0];
    let n: u32 = 3;

    let mut gate_gpu = stream.clone_htod(&gate).unwrap();
    let up_gpu = stream.clone_htod(&up).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut gate_gpu).arg(&up_gpu).arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&gate_gpu).unwrap();

    assert!((result[0] - 0.0).abs() < 1e-6, "silu(0)*1 = {}", result[0]);
    assert!(
        (result[1] - 0.7310586).abs() < 1e-5,
        "silu(1)*1 = {}",
        result[1]
    );
    assert!(
        (result[2] - (-0.2689414)).abs() < 1e-5,
        "silu(-1)*1 = {}",
        result[2]
    );
}

// ---------------------------------------------------------------------------
// Residual add tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_residual_add() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("residual_add").unwrap();

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let residual = vec![10.0f32, 20.0, 30.0, 40.0];
    let n: u32 = 4;

    let mut x_gpu = stream.clone_htod(&x).unwrap();
    let res_gpu = stream.clone_htod(&residual).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut x_gpu).arg(&res_gpu).arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&x_gpu).unwrap();

    let expected = vec![11.0f32, 22.0, 33.0, 44.0];
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-6,
            "residual_add[{i}]: GPU {}, expected {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_residual_add_large() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("residual_add").unwrap();

    let dim = 4096usize;
    let x: Vec<f32> = (0..dim).map(|i| i as f32).collect();
    let residual: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5).collect();
    let n = dim as u32;

    let mut x_gpu = stream.clone_htod(&x).unwrap();
    let res_gpu = stream.clone_htod(&residual).unwrap();

    let block_dim = 256u32;
    let grid_dim = n.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut x_gpu).arg(&res_gpu).arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&x_gpu).unwrap();

    for i in 0..dim {
        let expected = x[i] + residual[i];
        assert!(
            (result[i] - expected).abs() < 1e-4,
            "residual_add_large[{i}]: GPU {}, expected {}",
            result[i],
            expected
        );
    }
}

// ---------------------------------------------------------------------------
// Softmax tests
// ---------------------------------------------------------------------------

/// CPU reference: softmax with max-subtraction
fn cpu_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[test]
fn test_cuda_softmax() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("softmax_inplace").unwrap();

    let scores = vec![1.0f32, 2.0, 3.0];
    let n: u32 = 3;
    let expected = cpu_softmax(&scores);

    let mut scores_gpu = stream.clone_htod(&scores).unwrap();

    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) * 4,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut scores_gpu).arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&scores_gpu).unwrap();

    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum should be 1.0, got {}",
        sum
    );
    for i in 0..3 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "softmax[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
    assert!(result[2] > result[1], "softmax ordering: [2] > [1]");
    assert!(result[1] > result[0], "softmax ordering: [1] > [0]");
}

#[test]
fn test_cuda_softmax_large_values() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::ACTIVATIONS_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("softmax_inplace").unwrap();

    // Large values that would overflow without max-subtraction.
    let scores = vec![1000.0f32, 1001.0, 1002.0];
    let n: u32 = 3;
    let expected = cpu_softmax(&scores);

    let mut scores_gpu = stream.clone_htod(&scores).unwrap();

    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) * 4,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut scores_gpu).arg(&n)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&scores_gpu).unwrap();

    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum with large values: {}",
        sum
    );
    assert!(
        result.iter().all(|v| v.is_finite()),
        "softmax should produce finite values for large inputs"
    );
    for i in 0..3 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "softmax_large[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// RoPE tests
// ---------------------------------------------------------------------------

/// CPU reference: apply_rope for a single vector, computing cos/sin from theta.
///
/// Matches the CUDA kernel's internal angle computation:
///   freq = 1 / theta^(2d / head_dim)
///   angle = pos * freq
///   (x0', x1') = (x0*cos - x1*sin, x0*sin + x1*cos)
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
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let idx0 = head_start + 2 * i;
            let idx1 = head_start + 2 * i + 1;
            let v0 = vec[idx0];
            let v1 = vec[idx1];
            vec[idx0] = v0 * cos_val - v1 * sin_val;
            vec[idx1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

/// Launch the rope_apply kernel on both Q and K vectors.
///
/// The CUDA kernel signature is:
///   rope_apply(q, k, pos, num_q_heads, num_kv_heads, head_dim, theta_base)
/// It processes both Q and K in a single launch, computing cos/sin internally.
fn run_rope_kernel(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    q: &[f32],
    k: &[f32],
    pos: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let src = lumen_runtime::cuda::shaders::ROPE_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for rope.cu");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("rope_apply").unwrap();

    let mut q_gpu = stream.clone_htod(q).unwrap();
    let mut k_gpu = stream.clone_htod(k).unwrap();

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

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut q_gpu)
            .arg(&mut k_gpu)
            .arg(&pos)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&theta)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let q_result = stream.clone_dtoh(&q_gpu).unwrap();
    let k_result = stream.clone_dtoh(&k_gpu).unwrap();
    (q_result, k_result)
}

#[test]
fn test_cuda_rope_position_zero() {
    let (ctx, stream) = create_context();

    // 1 head, head_dim=4
    let head_dim = 4u32;
    let num_heads = 1u32;
    let pos = 0u32;
    let theta = 10000.0f32;

    let q = vec![1.0f32, 0.0, 0.0, 1.0];
    let k = vec![0.5f32, 0.5, 0.5, 0.5];
    let mut q_expected = q.clone();
    let mut k_expected = k.clone();
    cpu_rope_apply(&mut q_expected, pos as usize, num_heads as usize, head_dim as usize, theta);
    cpu_rope_apply(&mut k_expected, pos as usize, num_heads as usize, head_dim as usize, theta);

    let (q_result, k_result) = run_rope_kernel(
        &ctx, &stream, &q, &k, pos, num_heads, num_heads, head_dim, theta,
    );

    // At position 0, angles are 0 -> cos=1, sin=0 -> no change.
    for i in 0..4 {
        assert!(
            (q_result[i] - q_expected[i]).abs() < 1e-6,
            "rope_pos0 q[{i}]: GPU {}, CPU {}",
            q_result[i],
            q_expected[i]
        );
        assert!(
            (k_result[i] - k_expected[i]).abs() < 1e-6,
            "rope_pos0 k[{i}]: GPU {}, CPU {}",
            k_result[i],
            k_expected[i]
        );
    }
}

#[test]
fn test_cuda_rope_nonzero_position() {
    let (ctx, stream) = create_context();

    let head_dim = 4u32;
    let num_heads = 1u32;
    let pos = 1u32;
    let theta = 10000.0f32;

    let q = vec![1.0f32, 0.0, 0.0, 1.0];
    let k = vec![1.0f32, 1.0, 1.0, 1.0];
    let mut q_expected = q.clone();
    let mut k_expected = k.clone();
    cpu_rope_apply(&mut q_expected, pos as usize, num_heads as usize, head_dim as usize, theta);
    cpu_rope_apply(&mut k_expected, pos as usize, num_heads as usize, head_dim as usize, theta);

    let (q_result, _k_result) = run_rope_kernel(
        &ctx, &stream, &q, &k, pos, num_heads, num_heads, head_dim, theta,
    );

    // Verify against CPU reference.
    for i in 0..4 {
        assert!(
            (q_result[i] - q_expected[i]).abs() < 1e-5,
            "rope_pos1[{i}]: GPU {}, CPU {}",
            q_result[i],
            q_expected[i]
        );
    }

    // RoPE must preserve vector magnitude for each pair.
    let mag_before = (1.0f32 * 1.0 + 0.0 * 0.0).sqrt();
    let mag_after = (q_result[0] * q_result[0] + q_result[1] * q_result[1]).sqrt();
    assert!(
        (mag_before - mag_after).abs() < 1e-5,
        "RoPE must preserve magnitude: before={}, after={}",
        mag_before,
        mag_after
    );
}

#[test]
fn test_cuda_rope_multi_head() {
    let (ctx, stream) = create_context();

    let head_dim = 4u32;
    let num_q_heads = 4u32;
    let num_kv_heads = 4u32;
    let pos = 3u32;
    let theta = 10000.0f32;

    // 4 heads * 4 dims = 16 elements
    let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let k: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.2).collect();
    let mut q_expected = q.clone();
    let mut k_expected = k.clone();
    cpu_rope_apply(&mut q_expected, pos as usize, num_q_heads as usize, head_dim as usize, theta);
    cpu_rope_apply(&mut k_expected, pos as usize, num_kv_heads as usize, head_dim as usize, theta);

    let (q_result, k_result) = run_rope_kernel(
        &ctx, &stream, &q, &k, pos, num_q_heads, num_kv_heads, head_dim, theta,
    );

    for i in 0..16 {
        assert!(
            (q_result[i] - q_expected[i]).abs() < 1e-4,
            "rope_multi q[{i}]: GPU {}, CPU {}",
            q_result[i],
            q_expected[i]
        );
        assert!(
            (k_result[i] - k_expected[i]).abs() < 1e-4,
            "rope_multi k[{i}]: GPU {}, CPU {}",
            k_result[i],
            k_expected[i]
        );
    }
}
