// GatedDeltaNet (Linear Attention) kernel tests
// Extracted from mod.rs for modularity.

use crate::metal::*;
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::ffi::MTLSize;

#[test]
fn test_gated_delta_net_state_update_beta_one() {
    // With beta=1.0 and h_state starting at zero, after one step:
    // h_state[h, ki, vj] = outer(k, v)[ki, vj] = k[ki] * v[vj]
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gated_delta_net_state_update").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 2;
    let key_dim: u32 = 4;
    let val_dim: u32 = 4;
    let n_kv_heads: u32 = 2;
    let state_size = (n_heads * key_dim * val_dim) as usize;

    let h_state = vec![0.0f32; state_size];
    let k_norm: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let v_tokens: Vec<f32> = (1..=8).map(|x| x as f32 * 0.1).collect();
    let beta = vec![1.0f32; n_heads as usize];

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let beta_buf = backend.upload_f32(&beta).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&h_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&beta_buf, 0, 3);
    enc.set_bytes(&n_heads.to_le_bytes(), 4);
    enc.set_bytes(&key_dim.to_le_bytes(), 5);
    enc.set_bytes(&val_dim.to_le_bytes(), 6);
    enc.set_bytes(&n_kv_heads.to_le_bytes(), 7);
    enc.dispatch_threads(
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; state_size];
    h_buf.read_f32(&mut result);

    // h_state is now transposed: [n_heads, val_dim, key_dim]
    for h in 0..n_heads as usize {
        for ki in 0..key_dim as usize {
            for vj in 0..val_dim as usize {
                let expected = k_norm[h * key_dim as usize + ki]
                    * v_tokens[h * val_dim as usize + vj];
                let actual = result[h * (val_dim as usize * key_dim as usize)
                    + vj * key_dim as usize + ki];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "state_update beta=1: h={h} ki={ki} vj={vj}: got {actual}, expected {expected}"
                );
            }
        }
    }
    eprintln!("gated_delta_net_state_update (beta=1.0): PASS");
}

#[test]
fn test_gated_delta_net_state_update_beta_zero() {
    // With beta=0.0, h_state should be unchanged
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gated_delta_net_state_update").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 2;
    let key_dim: u32 = 4;
    let val_dim: u32 = 4;
    let n_kv_heads: u32 = 2;
    let state_size = (n_heads * key_dim * val_dim) as usize;

    let h_state: Vec<f32> = (0..state_size).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let h_state_copy = h_state.clone();
    let k_norm = vec![99.0f32; (n_heads * key_dim) as usize];
    let v_tokens = vec![99.0f32; (n_kv_heads * val_dim) as usize];
    let beta = vec![0.0f32; n_heads as usize];

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let beta_buf = backend.upload_f32(&beta).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&h_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&beta_buf, 0, 3);
    enc.set_bytes(&n_heads.to_le_bytes(), 4);
    enc.set_bytes(&key_dim.to_le_bytes(), 5);
    enc.set_bytes(&val_dim.to_le_bytes(), 6);
    enc.set_bytes(&n_kv_heads.to_le_bytes(), 7);
    enc.dispatch_threads(
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; state_size];
    h_buf.read_f32(&mut result);

    for i in 0..state_size {
        assert!(
            (result[i] - h_state_copy[i]).abs() < 1e-6,
            "state_update beta=0: index {i}: got {}, expected {}",
            result[i], h_state_copy[i]
        );
    }
    eprintln!("gated_delta_net_state_update (beta=0.0): PASS");
}

#[test]
fn test_gated_delta_net_state_update_beta_half() {
    // With beta=0.5, h = 0.5*h_old + 0.5*outer(k,v)
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gated_delta_net_state_update").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 1;
    let key_dim: u32 = 2;
    let val_dim: u32 = 2;
    let n_kv_heads: u32 = 1;

    // h_state is transposed: [val_dim, key_dim]
    // Logical H[ki,vj]: H[0,0]=1, H[0,1]=2, H[1,0]=3, H[1,1]=4
    // Transposed storage: h_state[vj,ki]: [H[0,0],H[1,0],H[0,1],H[1,1]] = [1,3,2,4]
    let h_state = vec![1.0f32, 3.0, 2.0, 4.0];
    let k_norm = vec![1.0f32, 0.0];
    let v_tokens = vec![2.0f32, 3.0];
    let beta = vec![0.5f32];
    // outer(k,v) = [[2, 3], [0, 0]] (logical H[ki,vj])
    // new H[ki,vj] = 0.5 * old + 0.5 * outer = [1.5, 2.5, 1.5, 2.0] (logical)
    // Transposed: [H[0,0],H[1,0],H[0,1],H[1,1]] = [1.5, 1.5, 2.5, 2.0]

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let beta_buf = backend.upload_f32(&beta).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&h_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&beta_buf, 0, 3);
    enc.set_bytes(&n_heads.to_le_bytes(), 4);
    enc.set_bytes(&key_dim.to_le_bytes(), 5);
    enc.set_bytes(&val_dim.to_le_bytes(), 6);
    enc.set_bytes(&n_kv_heads.to_le_bytes(), 7);
    enc.dispatch_threads(
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
        MTLSize::new(key_dim as u64, val_dim as u64, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    h_buf.read_f32(&mut result);

    // Transposed layout expected: [H_new[0,0], H_new[1,0], H_new[0,1], H_new[1,1]]
    let expected = [1.5f32, 1.5, 2.5, 2.0];
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "state_update beta=0.5: index {i}: got {}, expected {}",
            result[i], expected[i]
        );
    }
    eprintln!("gated_delta_net_state_update (beta=0.5): PASS");
}

#[test]
fn test_gated_delta_net_output_identity_state() {
    // h_state = identity matrix per head -> output should equal q
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gated_delta_net_output").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 2;
    let key_dim: u32 = 4;
    let val_dim: u32 = 4;

    // Transposed layout: h_state[h, vj, ki] — identity means h_state[h,i,i] = 1
    let mut h_state = vec![0.0f32; (n_heads * key_dim * val_dim) as usize];
    for h in 0..n_heads as usize {
        for i in 0..key_dim.min(val_dim) as usize {
            h_state[h * (val_dim as usize * key_dim as usize) + i * key_dim as usize + i] = 1.0;
        }
    }

    let q_norm: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let out_size = (n_heads * val_dim) as usize;

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let q_buf = backend.upload_f32(&q_norm).unwrap();
    let out_buf = backend.device.new_buffer(out_size * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&h_buf, 0, 0);
    enc.set_buffer(&q_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&n_heads.to_le_bytes(), 3);
    enc.set_bytes(&key_dim.to_le_bytes(), 4);
    enc.set_bytes(&val_dim.to_le_bytes(), 5);
    let num_tg = (n_heads * val_dim) as u64;
    enc.dispatch_threadgroups(
        MTLSize::new(num_tg, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; out_size];
    out_buf.read_f32(&mut result);

    for i in 0..out_size {
        assert!(
            (result[i] - q_norm[i]).abs() < 1e-5,
            "delta_net_output identity: index {i}: got {}, expected {}",
            result[i], q_norm[i]
        );
    }
    eprintln!("gated_delta_net_output (identity state): PASS");
}

#[test]
fn test_gated_delta_net_output_known_state() {
    // Known h_state, verify dot product correctness
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gated_delta_net_output").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 1;
    let key_dim: u32 = 2;
    let val_dim: u32 = 2;

    // Logical H[ki,vj] = [[1, 2], [3, 4]]
    // Transposed storage h_state[vj,ki] = [H[0,0],H[1,0],H[0,1],H[1,1]] = [1,3,2,4]
    let h_state = vec![1.0f32, 3.0, 2.0, 4.0];
    let q_norm = vec![1.0f32, 1.0];
    // output[0] = H[0,0]*1 + H[1,0]*1 = 1+3 = 4
    // output[1] = H[0,1]*1 + H[1,1]*1 = 2+4 = 6

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let q_buf = backend.upload_f32(&q_norm).unwrap();
    let out_buf = backend.device.new_buffer(8).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&h_buf, 0, 0);
    enc.set_buffer(&q_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&n_heads.to_le_bytes(), 3);
    enc.set_bytes(&key_dim.to_le_bytes(), 4);
    enc.set_bytes(&val_dim.to_le_bytes(), 5);
    enc.dispatch_threadgroups(
        MTLSize::new(2, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 2];
    out_buf.read_f32(&mut result);

    assert!(
        (result[0] - 4.0).abs() < 1e-5,
        "delta_net_output known: [0] got {}, expected 4.0", result[0]
    );
    assert!(
        (result[1] - 6.0).abs() < 1e-5,
        "delta_net_output known: [1] got {}, expected 6.0", result[1]
    );
    eprintln!("gated_delta_net_output (known state): PASS");
}

#[test]
fn test_l2_normalize_heads() {
    // Verify L2 normalization produces unit-norm vectors
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("l2_normalize_heads").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 2;
    let head_dim: u32 = 4;
    let eps: f32 = 1e-12;

    // Head 0: [3, 4, 0, 0] -> norm=5
    // Head 1: [1, 1, 1, 1] -> norm=2
    let x = vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let x_buf = backend.upload_f32(&x).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&x_buf, 0, 0);
    enc.set_bytes(&n_heads.to_le_bytes(), 1);
    enc.set_bytes(&head_dim.to_le_bytes(), 2);
    enc.set_bytes(&eps.to_le_bytes(), 3);
    enc.dispatch_threadgroups(
        MTLSize::new(n_heads as u64, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 8];
    x_buf.read_f32(&mut result);

    let norm0: f32 = result[0..4].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm0 - 1.0).abs() < 1e-5, "l2_norm head 0: norm={norm0}");
    assert!((result[0] - 0.6).abs() < 1e-5, "l2_norm [0]: got {}", result[0]);
    assert!((result[1] - 0.8).abs() < 1e-5, "l2_norm [1]: got {}", result[1]);

    let norm1: f32 = result[4..8].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm1 - 1.0).abs() < 1e-5, "l2_norm head 1: norm={norm1}");
    for i in 4..8 {
        assert!((result[i] - 0.5).abs() < 1e-5, "l2_norm [{i}]: got {}", result[i]);
    }
    eprintln!("l2_normalize_heads: PASS");
}

#[test]
fn test_l2_normalize_heads_zero_vector() {
    // Zero vector should not produce NaN
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("l2_normalize_heads").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 1;
    let head_dim: u32 = 4;
    let eps: f32 = 1e-6;
    let x = vec![0.0f32; 4];
    let x_buf = backend.upload_f32(&x).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&x_buf, 0, 0);
    enc.set_bytes(&n_heads.to_le_bytes(), 1);
    enc.set_bytes(&head_dim.to_le_bytes(), 2);
    enc.set_bytes(&eps.to_le_bytes(), 3);
    enc.dispatch_threadgroups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    x_buf.read_f32(&mut result);

    for i in 0..4 {
        assert!(result[i].is_finite(), "l2_norm zero [{i}]: got {} (not finite)", result[i]);
    }
    eprintln!("l2_normalize_heads (zero vector): PASS");
}

#[test]
fn test_ssm_conv1d_decode() {
    // Causal 1D convolution with kernel_size=4, uniform weights
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("ssm_conv1d_decode").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let dim: u32 = 4;
    let kernel_size: u32 = 4;
    let state_pos: u32 = 0;

    let conv_state = vec![
        1.0f32, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ];
    let input_val = vec![4.0f32; 4];
    let kernel_w = vec![1.0f32; 16];

    let input_buf = backend.upload_f32(&input_val).unwrap();
    let state_buf = backend.upload_f32(&conv_state).unwrap();
    let kernel_buf = backend.upload_f32(&kernel_w).unwrap();
    let output_buf = backend.device.new_buffer(16).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&input_buf, 0, 0);
    enc.set_buffer(&state_buf, 0, 1);
    enc.set_buffer(&kernel_buf, 0, 2);
    enc.set_buffer(&output_buf, 0, 3);
    enc.set_bytes(&dim.to_le_bytes(), 4);
    enc.set_bytes(&kernel_size.to_le_bytes(), 5);
    enc.set_bytes(&state_pos.to_le_bytes(), 6);
    enc.dispatch_threadgroups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(4, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    output_buf.read_f32(&mut result);

    for i in 0..4 {
        assert!((result[i] - 10.0).abs() < 1e-5, "conv1d [{i}]: got {}, expected 10.0", result[i]);
    }

    let mut state_result = vec![0.0f32; 12];
    state_buf.read_f32(&mut state_result);
    for i in 0..4 {
        assert!((state_result[i] - 4.0).abs() < 1e-5, "conv1d state [{i}]: got {}", state_result[i]);
    }
    eprintln!("ssm_conv1d_decode: PASS");
}

#[test]
fn test_ssm_conv1d_decode_weighted() {
    // Non-uniform weights with circular buffer wrap-around
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("ssm_conv1d_decode").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let dim: u32 = 2;
    let kernel_size: u32 = 3;
    let state_pos: u32 = 1;

    let conv_state = vec![10.0f32, 20.0, 30.0, 40.0];
    let input_val = vec![50.0f32, 60.0];
    let kernel_w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // output[0] = 1*30 + 2*10 + 3*50 = 200
    // output[1] = 4*40 + 5*20 + 6*60 = 620

    let input_buf = backend.upload_f32(&input_val).unwrap();
    let state_buf = backend.upload_f32(&conv_state).unwrap();
    let kernel_buf = backend.upload_f32(&kernel_w).unwrap();
    let output_buf = backend.device.new_buffer(8).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&input_buf, 0, 0);
    enc.set_buffer(&state_buf, 0, 1);
    enc.set_buffer(&kernel_buf, 0, 2);
    enc.set_buffer(&output_buf, 0, 3);
    enc.set_bytes(&dim.to_le_bytes(), 4);
    enc.set_bytes(&kernel_size.to_le_bytes(), 5);
    enc.set_bytes(&state_pos.to_le_bytes(), 6);
    enc.dispatch_threads(
        MTLSize::new(dim as u64, 1, 1),
        MTLSize::new(dim as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 2];
    output_buf.read_f32(&mut result);

    assert!((result[0] - 200.0).abs() < 1e-4, "conv1d weighted [0]: got {}", result[0]);
    assert!((result[1] - 620.0).abs() < 1e-4, "conv1d weighted [1]: got {}", result[1]);
    eprintln!("ssm_conv1d_decode (weighted, wrap-around): PASS");
}

#[test]
fn test_sigmoid_gate_correctness() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("sigmoid_gate").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let input = vec![0.0f32, 1.0, -1.0, 10.0, -10.0];
    let dim = input.len() as u32;
    let input_buf = backend.upload_f32(&input).unwrap();
    let output_buf = backend.device.new_buffer(input.len() * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&input_buf, 0, 0);
    enc.set_buffer(&output_buf, 0, 1);
    enc.set_bytes(&dim.to_le_bytes(), 2);
    enc.dispatch_threads(
        MTLSize::new(dim as u64, 1, 1),
        MTLSize::new(dim as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; input.len()];
    output_buf.read_f32(&mut result);

    for i in 0..input.len() {
        let expected = 1.0f32 / (1.0 + (-input[i]).exp());
        assert!(
            (result[i] - expected).abs() < 1e-5,
            "sigmoid [{i}]: got {}, expected {}", result[i], expected
        );
    }
    eprintln!("sigmoid_gate: PASS");
}

#[test]
fn test_silu_elementwise_mul_correctness() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("silu_elementwise_mul").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let alpha = vec![0.0f32, 1.0, -1.0, 2.0];
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let dim = 4u32;

    let alpha_buf = backend.upload_f32(&alpha).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let output_buf = backend.device.new_buffer(16).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&alpha_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&output_buf, 0, 2);
    enc.set_bytes(&dim.to_le_bytes(), 3);
    enc.dispatch_threads(
        MTLSize::new(4, 1, 1),
        MTLSize::new(4, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    output_buf.read_f32(&mut result);

    for i in 0..4 {
        let a = alpha[i];
        let sig = 1.0f32 / (1.0 + (-a).exp());
        let expected = a * sig * x[i];
        assert!(
            (result[i] - expected).abs() < 1e-5,
            "silu_mul [{i}]: got {}, expected {}", result[i], expected
        );
    }
    eprintln!("silu_elementwise_mul: PASS");
}

#[test]
fn test_gated_delta_net_roundtrip() {
    // Full roundtrip: state_update then output, verify mathematical consistency.
    // h = outer(k, v), output = h^T @ q = v * dot(k, q)
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let update_func = lib.get_function("gated_delta_net_state_update").unwrap();
    let update_pso = backend.device.new_compute_pipeline_state(&update_func).unwrap();
    let output_func = lib.get_function("gated_delta_net_output").unwrap();
    let output_pso = backend.device.new_compute_pipeline_state(&output_func).unwrap();

    let n_heads: u32 = 1;
    let key_dim: u32 = 4;
    let val_dim: u32 = 4;
    let n_kv_heads: u32 = 1;

    let h_state = vec![0.0f32; 16];
    let k_norm = vec![1.0f32, 0.5, 0.0, -0.5];
    let v_tokens = vec![2.0f32, 3.0, 4.0, 5.0];
    let beta = vec![1.0f32];
    let q_norm = vec![1.0f32, 1.0, 0.0, 0.0];

    // dot(k, q) = 1*1 + 0.5*1 + 0*0 + (-0.5)*0 = 1.5
    // output = v * 1.5 = [3.0, 4.5, 6.0, 7.5]

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let beta_buf = backend.upload_f32(&beta).unwrap();
    let q_buf = backend.upload_f32(&q_norm).unwrap();
    let out_buf = backend.device.new_buffer(16).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();

    {
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&update_pso);
        enc.set_buffer(&h_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&v_buf, 0, 2);
        enc.set_buffer(&beta_buf, 0, 3);
        enc.set_bytes(&n_heads.to_le_bytes(), 4);
        enc.set_bytes(&key_dim.to_le_bytes(), 5);
        enc.set_bytes(&val_dim.to_le_bytes(), 6);
        enc.set_bytes(&n_kv_heads.to_le_bytes(), 7);
        enc.dispatch_threads(
            MTLSize::new(key_dim as u64, val_dim as u64, 1),
            MTLSize::new(key_dim as u64, val_dim as u64, 1),
        );
        enc.end_encoding();
    }

    {
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&output_pso);
        enc.set_buffer(&h_buf, 0, 0);
        enc.set_buffer(&q_buf, 0, 1);
        enc.set_buffer(&out_buf, 0, 2);
        enc.set_bytes(&n_heads.to_le_bytes(), 3);
        enc.set_bytes(&key_dim.to_le_bytes(), 4);
        enc.set_bytes(&val_dim.to_le_bytes(), 5);
        enc.dispatch_threadgroups(
            MTLSize::new(4, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
    }

    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    out_buf.read_f32(&mut result);

    let expected = [3.0f32, 4.5, 6.0, 7.5];
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "roundtrip [{i}]: got {}, expected {}", result[i], expected[i]
        );
    }
    eprintln!("gated_delta_net roundtrip (update+output): PASS");
}

/// Test gdn_state_output_norm: delta rule state update + output + RMSNorm.
/// h_state layout: [n_heads, val_dim, key_dim] (transposed for coalesced access).
///
/// Setup: 1 head, key_dim=val_dim=4, alpha=1.0 (no decay), beta=1.0
/// Initial h_state = zeros.
///
/// Step 1 token: k=[1,0,0,0], v=[0,1,0,0], q=[1,0,0,0]
///   retrieval = H^T @ k = 0 (h_state is zero)
///   v_delta = beta*(v - retrieval) = [0,1,0,0]
///   H_new = H + k x v_delta = outer([1,0,0,0], [0,1,0,0]):
///     H[0,:] = [0, 1, 0, 0]   (all others zero)
///   raw_out[vj] = sum_ki(H[ki][vj] * q[ki])
///   raw_out = [0, 1, 0, 0]
///
/// RMSNorm: scale = [1,1,1,1], eps~0
///   sum(x^2) = 1, mean = 0.25, rms = sqrt(0.25) = 0.5
///   normalized = [0/0.5, 1/0.5, 0/0.5, 0/0.5] = [0, 2, 0, 0]
#[test]
fn test_gdn_state_output_norm_simple() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gdn_state_output_norm").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 1;
    let key_dim: u32 = 4;
    let val_dim: u32 = 4;
    let n_kv_heads: u32 = 1;
    let eps: f32 = 1e-6;  // small epsilon to avoid divide by zero

    let h_state = vec![0.0f32; (n_heads * key_dim * val_dim) as usize];
    let k_norm = vec![1.0f32, 0.0, 0.0, 0.0];   // k = e_0
    let v_tokens = vec![0.0f32, 1.0, 0.0, 0.0]; // v = e_1
    let alpha = vec![1.0f32];  // no decay
    let beta = vec![1.0f32];   // full update
    let q_norm = vec![1.0f32, 0.0, 0.0, 0.0];   // q = e_0
    let scale = vec![1.0f32, 1.0, 1.0, 1.0];    // identity scale

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let a_buf = backend.upload_f32(&alpha).unwrap();
    let b_buf = backend.upload_f32(&beta).unwrap();
    let q_buf = backend.upload_f32(&q_norm).unwrap();
    let s_buf = backend.upload_f32(&scale).unwrap();
    let out_buf = backend.device.new_buffer((val_dim * 4) as usize).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    {
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&h_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&v_buf, 0, 2);
        enc.set_buffer(&a_buf, 0, 3);
        enc.set_buffer(&b_buf, 0, 4);
        enc.set_buffer(&q_buf, 0, 5);
        enc.set_buffer(&s_buf, 0, 6);
        enc.set_buffer(&out_buf, 0, 7);
        enc.set_bytes(&n_heads.to_le_bytes(), 8);
        enc.set_bytes(&key_dim.to_le_bytes(), 9);
        enc.set_bytes(&val_dim.to_le_bytes(), 10);
        enc.set_bytes(&n_kv_heads.to_le_bytes(), 11);
        enc.set_bytes(&eps.to_le_bytes(), 12);
        let scale_n_heads: u32 = 1;
        enc.set_bytes(&scale_n_heads.to_le_bytes(), 13);
        enc.dispatch_threadgroups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(val_dim as u64, 1, 1),
        );
        enc.end_encoding();
    }
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; val_dim as usize];
    out_buf.read_f32(&mut result);

    // raw_out = [0, 1, 0, 0]; rms = 0.5; normalized = [0, 2, 0, 0]
    let expected = [0.0f32, 2.0, 0.0, 0.0];
    eprintln!("gdn_state_output_norm result: {:?}", result);
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-3,
            "gdn_state_output_norm [{i}]: got {}, expected {}", result[i], expected[i]
        );
    }
    eprintln!("test_gdn_state_output_norm_simple: PASS");
}

/// Test gdn_state_output_norm delta rule: retrieval from existing state.
///
/// Setup: 1 head, key_dim=val_dim=2, alpha=1.0, beta=1.0
/// Pre-load h_state: H = [[1,0],[0,1]] (identity)
///
/// Token: k=[1,0], v=[2,3], q=[1,0]
///   retrieval[vj] = sum_ki(H[ki][vj] * k[ki])
///     retrieval[0] = H[0][0]*k[0] + H[1][0]*k[1] = 1*1 + 0*0 = 1
///     retrieval[1] = H[0][1]*k[0] + H[1][1]*k[1] = 0*1 + 1*0 = 0
///   v_delta[vj] = beta*(v[vj] - retrieval[vj])
///     v_delta[0] = 1*(2 - 1) = 1
///     v_delta[1] = 1*(3 - 0) = 3
///   H_new[ki][vj] = H[ki][vj] + k[ki]*v_delta[vj]
///     H_new[0][0] = 1 + 1*1 = 2, H_new[0][1] = 0 + 1*3 = 3
///     H_new[1][0] = 0 + 0*1 = 0, H_new[1][1] = 1 + 0*3 = 1
///   raw_out[vj] = sum_ki(H_new[ki][vj] * q[ki])
///     raw_out[0] = H_new[0][0]*1 + H_new[1][0]*0 = 2
///     raw_out[1] = H_new[0][1]*1 + H_new[1][1]*0 = 3
///   rms = sqrt((4+9)/2 + eps) ≈ sqrt(6.5)
///   normalized = [2/sqrt(6.5), 3/sqrt(6.5)] * scale[1,1]
#[test]
fn test_gdn_state_output_norm_delta_rule() {
    let backend = MetalF32Backend::new().unwrap();
    let lib = backend.device.new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("gdn_state_output_norm").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let n_heads: u32 = 1;
    let key_dim: u32 = 2;
    let val_dim: u32 = 2;
    let n_kv_heads: u32 = 1;
    let eps: f32 = 0.0;

    // H = identity [[1,0],[0,1]] stored in transposed layout h_state[h][vj][ki]
    // Storage: h_state + h * val_dim * key_dim + vj * key_dim + ki
    // h_state[0,0]=1, h_state[0,1]=0, h_state[1,0]=0, h_state[1,1]=1
    // (identity is symmetric, so data is the same)
    let h_state = vec![1.0f32, 0.0, 0.0, 1.0];
    let k_norm = vec![1.0f32, 0.0];
    let v_tokens = vec![2.0f32, 3.0];
    let alpha = vec![1.0f32];
    let beta = vec![1.0f32];
    let q_norm = vec![1.0f32, 0.0];
    let scale = vec![1.0f32, 1.0];

    let h_buf = backend.upload_f32(&h_state).unwrap();
    let k_buf = backend.upload_f32(&k_norm).unwrap();
    let v_buf = backend.upload_f32(&v_tokens).unwrap();
    let a_buf = backend.upload_f32(&alpha).unwrap();
    let b_buf = backend.upload_f32(&beta).unwrap();
    let q_buf = backend.upload_f32(&q_norm).unwrap();
    let s_buf = backend.upload_f32(&scale).unwrap();
    let out_buf = backend.device.new_buffer((val_dim * 4) as usize).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    {
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&h_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&v_buf, 0, 2);
        enc.set_buffer(&a_buf, 0, 3);
        enc.set_buffer(&b_buf, 0, 4);
        enc.set_buffer(&q_buf, 0, 5);
        enc.set_buffer(&s_buf, 0, 6);
        enc.set_buffer(&out_buf, 0, 7);
        enc.set_bytes(&n_heads.to_le_bytes(), 8);
        enc.set_bytes(&key_dim.to_le_bytes(), 9);
        enc.set_bytes(&val_dim.to_le_bytes(), 10);
        enc.set_bytes(&n_kv_heads.to_le_bytes(), 11);
        enc.set_bytes(&eps.to_le_bytes(), 12);
        let scale_n_heads: u32 = 1;
        enc.set_bytes(&scale_n_heads.to_le_bytes(), 13);
        enc.dispatch_threadgroups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(val_dim as u64, 1, 1),
        );
        enc.end_encoding();
    }
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; val_dim as usize];
    out_buf.read_f32(&mut result);

    // raw_out = [2, 3], rms = sqrt((4+9)/2) = sqrt(6.5) ≈ 2.5495
    let rms = (6.5f32).sqrt();
    let expected = [2.0f32 / rms, 3.0f32 / rms];
    eprintln!("gdn_state_output_norm delta rule result: {:?}", result);
    eprintln!("expected: {:?}", expected);
    for i in 0..2 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-3,
            "gdn_delta_rule [{i}]: got {}, expected {}", result[i], expected[i]
        );
    }
    eprintln!("test_gdn_state_output_norm_delta_rule: PASS");
}
