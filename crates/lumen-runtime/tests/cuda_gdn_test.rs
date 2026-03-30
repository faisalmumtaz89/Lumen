//! Integration tests for CUDA GatedDeltaNet (GDN) kernels.
//!
//! Tests each GDN kernel against CPU reference implementations with known
//! inputs and expected outputs. Requires a CUDA-capable GPU (run on Modal).
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_gdn_test

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

fn create_context() -> (
    std::sync::Arc<CudaContext>,
    std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let ctx = CudaContext::new(0).expect("No CUDA GPU available");
    let stream = ctx.default_stream();
    (ctx, stream)
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: ssm_conv1d_decode
///
/// Circular-buffer causal convolution matching the Metal/CUDA kernel exactly.
fn cpu_conv1d_decode(
    conv_state: &mut [f32], // [buf_slots, conv_dim]
    input: &[f32],          // [conv_dim]
    weight: &[f32],         // [conv_dim, kernel_size]
    conv_dim: usize,
    kernel_size: usize,
    state_pos: usize,
) -> Vec<f32> {
    let buf_slots = kernel_size - 1;
    let mut output = vec![0.0f32; conv_dim];

    for gid in 0..conv_dim {
        let mut sum = 0.0f32;
        // Taps 0..buf_slots-1: read from circular buffer
        for tap in 0..buf_slots {
            let slot = (state_pos + tap) % buf_slots;
            sum += weight[gid * kernel_size + tap] * conv_state[slot * conv_dim + gid];
        }
        // Tap buf_slots: current input
        sum += weight[gid * kernel_size + buf_slots] * input[gid];
        output[gid] = sum;
        // Update circular buffer
        conv_state[state_pos * conv_dim + gid] = input[gid];
    }
    output
}

/// CPU reference: gdn_compute_gates
fn cpu_gdn_compute_gates(
    dt_bias: &[f32],
    ssm_a: &[f32],
    beta_proj: &[f32],
    alpha_proj: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = dt_bias.len();
    let mut alpha_out = vec![0.0f32; n];
    let mut beta_out = vec![0.0f32; n];

    for h in 0..n {
        let sp_input = alpha_proj[h] + dt_bias[h];
        let sp = if sp_input > 20.0 {
            sp_input
        } else {
            (1.0f32 + sp_input.exp()).ln()
        };
        let gate = ssm_a[h] * sp;
        alpha_out[h] = gate.exp();
        beta_out[h] = 1.0 / (1.0 + (-beta_proj[h]).exp());
    }
    (alpha_out, beta_out)
}

/// CPU reference: l2_normalize_heads
fn cpu_l2_normalize_heads(x: &mut [f32], num_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &x[start..end];
        let ss: f32 = head.iter().map(|&v| v * v).sum();
        let norm = ss.sqrt();
        let scale = if norm > eps { 1.0 / norm } else { 1.0 / eps };
        for i in start..end {
            x[i] *= scale;
        }
    }
}

/// CPU reference: gdn_state_update (delta rule with separate output)
///
/// Implements the exact same algorithm as the CUDA kernel:
///   1. s_decayed = alpha * s_old
///   2. retrieval = s_decayed^T @ k
///   3. delta = beta * (v - retrieval)
///   4. s_new = s_decayed + outer(k, delta)
///   5. output = s_new @ (q * scale)  where scale = 1/sqrt(key_dim)
fn cpu_gdn_state_update(
    h_state: &mut [f32], // [num_heads, val_dim, key_dim] transposed layout
    k_norm: &[f32],      // [num_kv_heads * key_dim]
    v: &[f32],           // [num_heads * val_dim]
    alpha: &[f32],       // [num_heads]
    beta: &[f32],        // [num_heads]
    q_norm: &[f32],      // [num_kv_heads * key_dim]
    num_heads: usize,
    val_dim: usize,
    key_dim: usize,
    num_kv_heads: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * val_dim];
    let q_scale = 1.0 / (key_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_head = h % num_kv_heads;
        let a = alpha[h];
        let b = beta[h];

        for vj in 0..val_dim {
            let v_val = v[h * val_dim + vj];
            let h_base = h * val_dim * key_dim + vj * key_dim;

            // Phase 1: Decay + retrieval
            let mut retrieval = 0.0f32;
            for ki in 0..key_dim {
                let h_decayed = a * h_state[h_base + ki];
                h_state[h_base + ki] = h_decayed;
                retrieval += h_decayed * k_norm[kv_head * key_dim + ki];
            }

            // Phase 2: Delta update + output
            let v_delta = b * (v_val - retrieval);
            let mut my_out = 0.0f32;
            for ki in 0..key_dim {
                let h_updated =
                    h_state[h_base + ki] + k_norm[kv_head * key_dim + ki] * v_delta;
                h_state[h_base + ki] = h_updated;
                my_out += h_updated * q_norm[kv_head * key_dim + ki] * q_scale;
            }
            output[h * val_dim + vj] = my_out;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// ssm_conv1d_decode tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_conv1d_decode_basic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed for gdn.cu");
    let module = ctx.load_module(ptx).expect("Failed to load GDN module");
    let func = module
        .load_function("ssm_conv1d_decode")
        .expect("Failed to load ssm_conv1d_decode");

    // Small test: conv_dim=4, kernel_size=3 (buf_slots=2)
    let conv_dim: u32 = 4;
    let kernel_size: u32 = 3;
    let state_pos: u32 = 0;

    // conv_state: [buf_slots=2, conv_dim=4]
    let conv_state_cpu = vec![
        1.0f32, 2.0, 3.0, 4.0, // slot 0
        5.0, 6.0, 7.0, 8.0, // slot 1
    ];
    let input = vec![9.0f32, 10.0, 11.0, 12.0];
    // weight: [conv_dim=4, kernel_size=3]
    let weight = vec![
        0.1f32, 0.2, 0.3, // gid=0: tap0, tap1, tap2(current)
        0.4, 0.5, 0.6, // gid=1
        0.7, 0.8, 0.9, // gid=2
        1.0, 1.1, 1.2, // gid=3
    ];

    let expected = cpu_conv1d_decode(
        &mut conv_state_cpu.clone(),
        &input,
        &weight,
        conv_dim as usize,
        kernel_size as usize,
        state_pos as usize,
    );

    // GPU execution
    let mut conv_state_gpu = stream.clone_htod(&conv_state_cpu).unwrap();
    let input_gpu = stream.clone_htod(&input).unwrap();
    let weight_gpu = stream.clone_htod(&weight).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(conv_dim as usize).unwrap();

    let block_dim = 256u32;
    let grid_dim = conv_dim.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut conv_state_gpu)
            .arg(&input_gpu)
            .arg(&weight_gpu)
            .arg(&mut output_gpu)
            .arg(&conv_dim)
            .arg(&kernel_size)
            .arg(&state_pos)
            .launch(cfg)
    }
    .expect("ssm_conv1d_decode launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&output_gpu).unwrap();

    for i in 0..conv_dim as usize {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "conv1d_decode[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }

    // Also verify conv_state was updated
    let state_result = stream.clone_dtoh(&conv_state_gpu).unwrap();
    let mut expected_state = conv_state_cpu.clone();
    cpu_conv1d_decode(
        &mut expected_state,
        &input,
        &weight,
        conv_dim as usize,
        kernel_size as usize,
        state_pos as usize,
    );
    for i in 0..expected_state.len() {
        assert!(
            (state_result[i] - expected_state[i]).abs() < 1e-5,
            "conv_state[{i}]: GPU {}, CPU {}",
            state_result[i],
            expected_state[i]
        );
    }
}

#[test]
fn test_cuda_conv1d_decode_nonzero_state_pos() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("ssm_conv1d_decode").unwrap();

    // conv_dim=8, kernel_size=4 (buf_slots=3), state_pos=2
    let conv_dim: u32 = 8;
    let kernel_size: u32 = 4;
    let state_pos: u32 = 2;

    let conv_state_cpu: Vec<f32> = (0..3 * 8).map(|i| (i as f32) * 0.1).collect();
    let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let weight: Vec<f32> = (0..8 * 4).map(|i| (i as f32) * 0.05).collect();

    let expected = cpu_conv1d_decode(
        &mut conv_state_cpu.clone(),
        &input,
        &weight,
        conv_dim as usize,
        kernel_size as usize,
        state_pos as usize,
    );

    let mut conv_state_gpu = stream.clone_htod(&conv_state_cpu).unwrap();
    let input_gpu = stream.clone_htod(&input).unwrap();
    let weight_gpu = stream.clone_htod(&weight).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(conv_dim as usize).unwrap();

    let block_dim = 256u32;
    let grid_dim = conv_dim.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut conv_state_gpu)
            .arg(&input_gpu)
            .arg(&weight_gpu)
            .arg(&mut output_gpu)
            .arg(&conv_dim)
            .arg(&kernel_size)
            .arg(&state_pos)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&output_gpu).unwrap();

    for i in 0..conv_dim as usize {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "conv1d_nonzero[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

// ---------------------------------------------------------------------------
// gdn_compute_gates tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_gdn_compute_gates_basic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).expect("NVRTC compile failed");
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("gdn_compute_gates").unwrap();

    let num_heads: u32 = 4;

    // dt_bias: typical small positive values
    let dt_bias = vec![0.5f32, 1.0, -0.5, 2.0];
    // ssm_a: negative values (-exp(A_log))
    let ssm_a = vec![-0.5f32, -1.0, -0.036, -72.0];
    // beta_proj: pre-sigmoid values
    let beta_proj = vec![0.0f32, 2.0, -2.0, 5.0];
    // alpha_proj: gk_proj output
    let alpha_proj = vec![1.0f32, 0.5, 3.0, -1.0];

    let (expected_alpha, expected_beta) =
        cpu_gdn_compute_gates(&dt_bias, &ssm_a, &beta_proj, &alpha_proj);

    let dt_bias_gpu = stream.clone_htod(&dt_bias).unwrap();
    let ssm_a_gpu = stream.clone_htod(&ssm_a).unwrap();
    let beta_proj_gpu = stream.clone_htod(&beta_proj).unwrap();
    let alpha_proj_gpu = stream.clone_htod(&alpha_proj).unwrap();
    let mut alpha_out_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads as usize).unwrap();
    let mut beta_out_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads as usize).unwrap();

    let block_dim = 256u32;
    let grid_dim = num_heads.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&dt_bias_gpu)
            .arg(&ssm_a_gpu)
            .arg(&beta_proj_gpu)
            .arg(&alpha_proj_gpu)
            .arg(&mut alpha_out_gpu)
            .arg(&mut beta_out_gpu)
            .arg(&num_heads)
            .launch(cfg)
    }
    .expect("gdn_compute_gates launch failed");

    stream.synchronize().unwrap();
    let alpha_result = stream.clone_dtoh(&alpha_out_gpu).unwrap();
    let beta_result = stream.clone_dtoh(&beta_out_gpu).unwrap();

    for h in 0..num_heads as usize {
        assert!(
            (alpha_result[h] - expected_alpha[h]).abs() < 1e-5,
            "alpha[{h}]: GPU {}, CPU {}",
            alpha_result[h],
            expected_alpha[h]
        );
        // Alpha must be in (0, 1) since gate is negative
        assert!(
            alpha_result[h] > 0.0 && alpha_result[h] <= 1.0,
            "alpha[{h}] = {} should be in (0, 1]",
            alpha_result[h]
        );
    }

    for h in 0..num_heads as usize {
        assert!(
            (beta_result[h] - expected_beta[h]).abs() < 1e-5,
            "beta[{h}]: GPU {}, CPU {}",
            beta_result[h],
            expected_beta[h]
        );
        // Beta = sigmoid(x) must be in (0, 1)
        assert!(
            beta_result[h] > 0.0 && beta_result[h] < 1.0,
            "beta[{h}] = {} should be in (0, 1)",
            beta_result[h]
        );
    }
}

#[test]
fn test_cuda_gdn_compute_gates_large_softplus() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("gdn_compute_gates").unwrap();

    // Test the softplus large-value branch (sp_input > 20 => sp ~= sp_input)
    let num_heads: u32 = 2;
    let dt_bias = vec![10.0f32, 15.0];
    let ssm_a = vec![-0.5f32, -1.0];
    let beta_proj = vec![0.0f32, 100.0]; // sigmoid(100) ~= 1.0
    let alpha_proj = vec![15.0f32, 10.0]; // 15+10=25 > 20, 10+15=25 > 20

    let (expected_alpha, expected_beta) =
        cpu_gdn_compute_gates(&dt_bias, &ssm_a, &beta_proj, &alpha_proj);

    let dt_bias_gpu = stream.clone_htod(&dt_bias).unwrap();
    let ssm_a_gpu = stream.clone_htod(&ssm_a).unwrap();
    let beta_proj_gpu = stream.clone_htod(&beta_proj).unwrap();
    let alpha_proj_gpu = stream.clone_htod(&alpha_proj).unwrap();
    let mut alpha_out_gpu: CudaSlice<f32> = stream.alloc_zeros(2).unwrap();
    let mut beta_out_gpu: CudaSlice<f32> = stream.alloc_zeros(2).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&dt_bias_gpu)
            .arg(&ssm_a_gpu)
            .arg(&beta_proj_gpu)
            .arg(&alpha_proj_gpu)
            .arg(&mut alpha_out_gpu)
            .arg(&mut beta_out_gpu)
            .arg(&num_heads)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let alpha_result = stream.clone_dtoh(&alpha_out_gpu).unwrap();
    let beta_result = stream.clone_dtoh(&beta_out_gpu).unwrap();

    for h in 0..2 {
        assert!(
            (alpha_result[h] - expected_alpha[h]).abs() < 1e-5,
            "alpha_large[{h}]: GPU {}, CPU {}",
            alpha_result[h],
            expected_alpha[h]
        );
        assert!(
            (beta_result[h] - expected_beta[h]).abs() < 1e-5,
            "beta_large[{h}]: GPU {}, CPU {}",
            beta_result[h],
            expected_beta[h]
        );
    }

    // sigmoid(100) should be very close to 1.0
    assert!(
        (beta_result[1] - 1.0).abs() < 1e-6,
        "sigmoid(100) should be ~1.0, got {}",
        beta_result[1]
    );
}

// ---------------------------------------------------------------------------
// l2_normalize_heads tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_l2_normalize_basic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("l2_normalize_heads").unwrap();

    let num_heads: u32 = 2;
    let head_dim: u32 = 4;
    let eps: f32 = 1e-12;

    // Two heads: [3, 4, 0, 0] and [1, 1, 1, 1]
    let x = vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let mut expected = x.clone();
    cpu_l2_normalize_heads(&mut expected, num_heads as usize, head_dim as usize, eps);

    let mut x_gpu = stream.clone_htod(&x).unwrap();

    let block_dim = 32u32; // One warp per head
    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) * 4,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut x_gpu)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)
    }
    .expect("l2_normalize_heads launch failed");

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&x_gpu).unwrap();

    // Head 0: [3,4,0,0] -> norm=5, [0.6, 0.8, 0, 0]
    assert!(
        (result[0] - 0.6).abs() < 1e-5,
        "l2_norm[0]: GPU {}, expected 0.6",
        result[0]
    );
    assert!(
        (result[1] - 0.8).abs() < 1e-5,
        "l2_norm[1]: GPU {}, expected 0.8",
        result[1]
    );

    // Verify unit length per head
    for h in 0..num_heads as usize {
        let start = h * head_dim as usize;
        let head_norm: f32 = result[start..start + head_dim as usize]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (head_norm - 1.0).abs() < 1e-5,
            "head {h} norm should be 1.0, got {}",
            head_norm
        );
    }

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "l2_norm[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_l2_normalize_large_dim() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("l2_normalize_heads").unwrap();

    // Realistic GDN dimensions: 16 heads, 128 dim
    let num_heads: u32 = 16;
    let head_dim: u32 = 128;
    let eps: f32 = 1e-12;

    let x: Vec<f32> = (0..num_heads as usize * head_dim as usize)
        .map(|i| ((i as f32) * 0.01 + 0.5).sin())
        .collect();
    let mut expected = x.clone();
    cpu_l2_normalize_heads(&mut expected, num_heads as usize, head_dim as usize, eps);

    let mut x_gpu = stream.clone_htod(&x).unwrap();

    let block_dim = 128u32; // 4 warps for head_dim=128
    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) * 4,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut x_gpu)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&x_gpu).unwrap();

    // Verify unit length per head
    for h in 0..num_heads as usize {
        let start = h * head_dim as usize;
        let head_norm: f32 = result[start..start + head_dim as usize]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (head_norm - 1.0).abs() < 1e-4,
            "head {h} norm should be 1.0, got {}",
            head_norm
        );
    }

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "l2_norm_large[{i}]: GPU {}, CPU {}",
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_l2_normalize_zero_head() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("l2_normalize_heads").unwrap();

    // All-zero head should produce zeros (eps prevents division by zero)
    let num_heads: u32 = 1;
    let head_dim: u32 = 4;
    let eps: f32 = 1e-12;

    let x = vec![0.0f32; 4];
    let mut x_gpu = stream.clone_htod(&x).unwrap();

    let block_dim = 32u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim / 32) * 4,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut x_gpu)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let result = stream.clone_dtoh(&x_gpu).unwrap();

    for i in 0..4 {
        assert!(
            result[i].is_finite(),
            "l2_norm_zero[{i}] should be finite, got {}",
            result[i]
        );
        assert_eq!(result[i], 0.0, "l2_norm of zero should be zero");
    }
}

// ---------------------------------------------------------------------------
// gdn_state_update tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_gdn_state_update_basic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("gdn_state_update").unwrap();

    // Small test: 2 heads, val_dim=2, key_dim=2, 1 kv_head (GQA 2:1)
    let num_heads: u32 = 2;
    let val_dim: u32 = 2;
    let key_dim: u32 = 2;
    let num_kv_heads: u32 = 1;

    // h_state: [num_heads=2, val_dim=2, key_dim=2] = 8 floats
    let h_state_cpu = vec![
        // head 0, vj=0: [1.0, 2.0], vj=1: [3.0, 4.0]
        1.0f32, 2.0, 3.0, 4.0, // head 1, vj=0: [5.0, 6.0], vj=1: [7.0, 8.0]
        5.0, 6.0, 7.0, 8.0,
    ];

    // k_norm: [1 * 2] (1 kv_head)
    let k_norm = vec![0.6f32, 0.8];
    // v: [2 * 2] (2 heads, no GQA)
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    // alpha: decay factors
    let alpha = vec![0.9f32, 0.8];
    // beta: mixing rates
    let beta = vec![0.3f32, 0.5];
    // q_norm: [1 * 2] (1 kv_head)
    let q_norm = vec![0.5f32, 0.5];

    let mut expected_state = h_state_cpu.clone();
    let expected_output = cpu_gdn_state_update(
        &mut expected_state,
        &k_norm,
        &v,
        &alpha,
        &beta,
        &q_norm,
        num_heads as usize,
        val_dim as usize,
        key_dim as usize,
        num_kv_heads as usize,
    );

    // GPU execution
    let mut h_state_gpu = stream.clone_htod(&h_state_cpu).unwrap();
    let k_norm_gpu = stream.clone_htod(&k_norm).unwrap();
    let v_gpu = stream.clone_htod(&v).unwrap();
    let alpha_gpu = stream.clone_htod(&alpha).unwrap();
    let beta_gpu = stream.clone_htod(&beta).unwrap();
    let q_norm_gpu = stream.clone_htod(&q_norm).unwrap();
    let mut output_gpu: CudaSlice<f32> =
        stream.alloc_zeros(num_heads as usize * val_dim as usize).unwrap();

    // One block per head, val_dim threads per block
    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (val_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut h_state_gpu)
            .arg(&k_norm_gpu)
            .arg(&v_gpu)
            .arg(&alpha_gpu)
            .arg(&beta_gpu)
            .arg(&q_norm_gpu)
            .arg(&mut output_gpu)
            .arg(&num_heads)
            .arg(&val_dim)
            .arg(&key_dim)
            .arg(&num_kv_heads)
            .launch(cfg)
    }
    .expect("gdn_state_update launch failed");

    stream.synchronize().unwrap();
    let output_result = stream.clone_dtoh(&output_gpu).unwrap();
    let state_result = stream.clone_dtoh(&h_state_gpu).unwrap();

    // Verify output
    for i in 0..expected_output.len() {
        assert!(
            (output_result[i] - expected_output[i]).abs() < 1e-4,
            "state_update output[{i}]: GPU {}, CPU {}",
            output_result[i],
            expected_output[i]
        );
    }

    // Verify h_state was updated correctly
    for i in 0..expected_state.len() {
        assert!(
            (state_result[i] - expected_state[i]).abs() < 1e-4,
            "state_update h_state[{i}]: GPU {}, CPU {}",
            state_result[i],
            expected_state[i]
        );
    }
}

#[test]
fn test_cuda_gdn_state_update_realistic() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("gdn_state_update").unwrap();

    // Realistic GDN dimensions: 4 heads, val_dim=32, key_dim=32, 2 kv_heads (GQA 2:1)
    let num_heads: u32 = 4;
    let val_dim: u32 = 32;
    let key_dim: u32 = 32;
    let num_kv_heads: u32 = 2;

    let state_size = num_heads as usize * val_dim as usize * key_dim as usize;
    let h_state_cpu: Vec<f32> = (0..state_size).map(|i| (i as f32) * 0.001).collect();
    let k_norm: Vec<f32> = (0..num_kv_heads as usize * key_dim as usize)
        .map(|i| ((i as f32) * 0.1).sin() * 0.1)
        .collect();
    let v: Vec<f32> = (0..num_heads as usize * val_dim as usize)
        .map(|i| ((i as f32) * 0.05).cos() * 0.5)
        .collect();
    let alpha = vec![0.95f32, 0.9, 0.85, 0.8]; // realistic decay
    let beta = vec![0.1f32, 0.2, 0.15, 0.25]; // realistic mixing
    let q_norm: Vec<f32> = (0..num_kv_heads as usize * key_dim as usize)
        .map(|i| ((i as f32) * 0.07).sin() * 0.1)
        .collect();

    let mut expected_state = h_state_cpu.clone();
    let expected_output = cpu_gdn_state_update(
        &mut expected_state,
        &k_norm,
        &v,
        &alpha,
        &beta,
        &q_norm,
        num_heads as usize,
        val_dim as usize,
        key_dim as usize,
        num_kv_heads as usize,
    );

    let mut h_state_gpu = stream.clone_htod(&h_state_cpu).unwrap();
    let k_norm_gpu = stream.clone_htod(&k_norm).unwrap();
    let v_gpu = stream.clone_htod(&v).unwrap();
    let alpha_gpu = stream.clone_htod(&alpha).unwrap();
    let beta_gpu = stream.clone_htod(&beta).unwrap();
    let q_norm_gpu = stream.clone_htod(&q_norm).unwrap();
    let mut output_gpu: CudaSlice<f32> =
        stream.alloc_zeros(num_heads as usize * val_dim as usize).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (val_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut h_state_gpu)
            .arg(&k_norm_gpu)
            .arg(&v_gpu)
            .arg(&alpha_gpu)
            .arg(&beta_gpu)
            .arg(&q_norm_gpu)
            .arg(&mut output_gpu)
            .arg(&num_heads)
            .arg(&val_dim)
            .arg(&key_dim)
            .arg(&num_kv_heads)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let output_result = stream.clone_dtoh(&output_gpu).unwrap();
    let state_result = stream.clone_dtoh(&h_state_gpu).unwrap();

    // Verify output
    for i in 0..expected_output.len() {
        assert!(
            (output_result[i] - expected_output[i]).abs() < 1e-3,
            "realistic output[{i}]: GPU {}, CPU {}",
            output_result[i],
            expected_output[i]
        );
    }

    // Verify state
    for i in 0..expected_state.len() {
        assert!(
            (state_result[i] - expected_state[i]).abs() < 1e-3,
            "realistic h_state[{i}]: GPU {}, CPU {}",
            state_result[i],
            expected_state[i]
        );
    }
}

#[test]
fn test_cuda_gdn_state_update_zero_alpha() {
    let (ctx, stream) = create_context();

    let src = lumen_runtime::cuda::shaders::GDN_KERNEL_SOURCE;
    let ptx = compile_ptx(src).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("gdn_state_update").unwrap();

    // alpha=0 means complete state decay (no memory of past)
    let num_heads: u32 = 1;
    let val_dim: u32 = 2;
    let key_dim: u32 = 2;
    let num_kv_heads: u32 = 1;

    let h_state_cpu = vec![100.0f32, 200.0, 300.0, 400.0]; // large initial state
    let k_norm = vec![1.0f32, 0.0]; // unit vector along dim 0
    let v = vec![1.0f32, 2.0];
    let alpha = vec![0.0f32]; // total decay
    let beta = vec![1.0f32]; // full mixing
    let q_norm = vec![1.0f32, 0.0];

    let mut expected_state = h_state_cpu.clone();
    let expected_output = cpu_gdn_state_update(
        &mut expected_state,
        &k_norm,
        &v,
        &alpha,
        &beta,
        &q_norm,
        num_heads as usize,
        val_dim as usize,
        key_dim as usize,
        num_kv_heads as usize,
    );

    let mut h_state_gpu = stream.clone_htod(&h_state_cpu).unwrap();
    let k_norm_gpu = stream.clone_htod(&k_norm).unwrap();
    let v_gpu = stream.clone_htod(&v).unwrap();
    let alpha_gpu = stream.clone_htod(&alpha).unwrap();
    let beta_gpu = stream.clone_htod(&beta).unwrap();
    let q_norm_gpu = stream.clone_htod(&q_norm).unwrap();
    let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(2).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (val_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&mut h_state_gpu)
            .arg(&k_norm_gpu)
            .arg(&v_gpu)
            .arg(&alpha_gpu)
            .arg(&beta_gpu)
            .arg(&q_norm_gpu)
            .arg(&mut output_gpu)
            .arg(&num_heads)
            .arg(&val_dim)
            .arg(&key_dim)
            .arg(&num_kv_heads)
            .launch(cfg)
    }
    .unwrap();

    stream.synchronize().unwrap();
    let output_result = stream.clone_dtoh(&output_gpu).unwrap();

    for i in 0..expected_output.len() {
        assert!(
            (output_result[i] - expected_output[i]).abs() < 1e-4,
            "zero_alpha output[{i}]: GPU {}, CPU {}",
            output_result[i],
            expected_output[i]
        );
    }

    // With alpha=0, old state is completely erased
    let state_result = stream.clone_dtoh(&h_state_gpu).unwrap();
    for i in 0..expected_state.len() {
        assert!(
            (state_result[i] - expected_state[i]).abs() < 1e-4,
            "zero_alpha h_state[{i}]: GPU {}, CPU {}",
            state_result[i],
            expected_state[i]
        );
    }
}
