use crate::metal::*;
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::ffi::{MTLSize, MetalBuffer};

#[test]
fn test_metal_backend_creation() {
    let backend = MetalF32Backend::new();
    assert!(backend.is_ok(), "Should create Metal backend on macOS");
    let backend = backend.unwrap();
    let name = backend.device_name();
    assert!(!name.is_empty());
    eprintln!("Metal backend device: {name}");
}

#[test]
fn test_metal_matmul_correctness() {
    // Test that GPU matmul matches CPU matmul
    let backend = MetalF32Backend::new().unwrap();

    // Compile pipelines manually for this test
    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("matmul_f32").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    // 2x3 matrix * 3-vector
    let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0f32, 0.0, 1.0];

    let w_buf = backend.upload_f32(&w).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let out_buf = backend.device.new_buffer(8).unwrap(); // 2 floats

    let in_dim = 3u32;
    let tg_size = 32u64;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&w_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&in_dim.to_le_bytes(), 3);
    enc.dispatch_threadgroups(
        MTLSize::new(2, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 2];
    out_buf.read_f32(&mut result);
    assert_eq!(result, vec![4.0, 10.0], "GPU matmul should match CPU: [1,2,3]*[1,0,1]=4, [4,5,6]*[1,0,1]=10");
}

#[test]
fn test_metal_rmsnorm_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("rmsnorm").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;
    let dim = 4u32;

    let x_buf = backend.upload_f32(&x).unwrap();
    let w_buf = backend.upload_f32(&w).unwrap();
    let out_buf = backend.device.new_buffer(16).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&x_buf, 0, 0);
    enc.set_buffer(&w_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&dim.to_le_bytes(), 3);
    enc.set_bytes(&eps.to_le_bytes(), 4);
    enc.dispatch_threadgroups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 4];
    out_buf.read_f32(&mut result);

    // Reference: ms = (1+4+9+16)/4 = 7.5, scale = 1/sqrt(7.5+1e-5)
    let expected_scale = 1.0 / (7.5f32 + 1e-5).sqrt();
    for (i, &v) in result.iter().enumerate() {
        let expected = x[i] * expected_scale;
        assert!(
            (v - expected).abs() < 1e-4,
            "rmsnorm[{i}]: GPU={v}, expected={expected}"
        );
    }
}

#[test]
fn test_metal_softmax_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("softmax").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let data = vec![1.0f32, 2.0, 3.0];
    let data_buf = backend.upload_f32(&data).unwrap();
    let len = 3u32;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&data_buf, 0, 0);
    enc.set_bytes(&len.to_le_bytes(), 1);
    enc.dispatch_threadgroups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 3];
    data_buf.read_f32(&mut result);

    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum should be ~1.0, got {sum}"
    );
    assert!(result[2] > result[1], "softmax ordering");
    assert!(result[1] > result[0], "softmax ordering");
}

#[test]
fn test_metal_swiglu_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("swiglu").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let gate = vec![0.0f32, 1.0, -1.0];
    let up = vec![1.0f32, 1.0, 1.0];
    let dim = 3u32;

    let gate_buf = backend.upload_f32(&gate).unwrap();
    let up_buf = backend.upload_f32(&up).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&gate_buf, 0, 0);
    enc.set_buffer(&up_buf, 0, 1);
    enc.set_bytes(&dim.to_le_bytes(), 2);
    enc.dispatch_threads(
        MTLSize::new(3, 1, 1),
        MTLSize::new(3, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 3];
    gate_buf.read_f32(&mut result);

    assert!((result[0] - 0.0).abs() < 1e-6, "swiglu(0)*1 = 0");
    assert!((result[1] - 0.7310586).abs() < 1e-4, "swiglu(1)*1 ~ 0.731, got {}", result[1]);
    assert!((result[2] - (-0.2689414)).abs() < 1e-4, "swiglu(-1)*1 ~ -0.269, got {}", result[2]);
}

/// Helper: convert f16 (IEEE 754 half-precision) bits (u16) to f32.
fn f16_to_f32_bits(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized: convert to normalized f32
        let mut e = exp;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e = e.wrapping_sub(1);
        }
        f &= 0x3FF;
        let f32_exp = (e as i32 + 127 - 15 + 1) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13));
    }
    if exp == 0x1F {
        if frac != 0 {
            return f32::NAN;
        }
        return if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY };
    }

    let f32_exp = (exp as i32 + 127 - 15) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
}

/// Helper: convert a slice of f32 to f16 bits (u16).
fn f32_slice_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

/// Helper: convert a slice of f16 bits (u16) to f32.
fn f16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| f16_to_f32_bits(b)).collect()
}

/// Helper: convert f32 to f16 (IEEE 754 half-precision) as u16 bits.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xFF {
        let f16_frac = if frac != 0 { 0x200u16 } else { 0 };
        return ((sign << 15) | 0x7C00 | f16_frac as u32) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    let f16_frac = (frac >> 13) as u16;
    ((sign << 15) as u16) | ((new_exp as u16) << 10) | f16_frac
}

#[test]
fn test_metal_dequant_matmul_q8_0_correctness() {
    // Test the Q8_0 fused dequant + matmul kernel.
    //
    // Setup: 2x64 weight matrix (2 output rows, 64 input elements).
    // 64 elements = 2 Q8_0 blocks per row.
    //
    // Row 0: scale=0.5, values all 2  -> dequant = 1.0 for all elements
    // Row 1: scale=1.0, values [0,1,2,...,31, 0,1,2,...,31]
    //
    // Input x = [1.0; 64]  (all ones)
    //
    // Expected:
    //   out[0] = sum of 1.0 * 1.0 = 64.0
    //   out[1] = 1.0 * (0+1+...+31) * 2 blocks = 2 * 496 = 992.0

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("dequant_matmul_q8_0").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let in_dim: usize = 64;
    let out_dim: usize = 2;
    let q8_group_size: usize = 32;
    let q8_block_size: usize = 34;
    let num_blocks_per_row = in_dim / q8_group_size; // 2
    let row_bytes = num_blocks_per_row * q8_block_size; // 68

    let mut w_q8 = vec![0u8; out_dim * row_bytes]; // 136 bytes

    // Row 0: 2 blocks, each with scale=0.5, all int8 values = 2
    let scale_half_bits = f32_to_f16_bits(0.5);
    for b in 0..num_blocks_per_row {
        let block_start = 0 * row_bytes + b * q8_block_size;
        w_q8[block_start] = (scale_half_bits & 0xFF) as u8;
        w_q8[block_start + 1] = (scale_half_bits >> 8) as u8;
        for j in 0..q8_group_size {
            w_q8[block_start + 2 + j] = 2u8;
        }
    }

    // Row 1: 2 blocks, each with scale=1.0, values=[0,1,2,...,31]
    let scale_one_bits: u16 = 0x3C00;
    for b in 0..num_blocks_per_row {
        let block_start = 1 * row_bytes + b * q8_block_size;
        w_q8[block_start] = (scale_one_bits & 0xFF) as u8;
        w_q8[block_start + 1] = (scale_one_bits >> 8) as u8;
        for j in 0..q8_group_size {
            w_q8[block_start + 2 + j] = j as u8;
        }
    }

    let x = vec![1.0f32; in_dim];

    let w_buf = backend.device.new_buffer_with_bytes(&w_q8).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let out_buf = backend.device.new_buffer(out_dim * 4).unwrap();

    let in_dim_u32 = in_dim as u32;
    let tg_size = 32u64;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&w_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
    enc.dispatch_threadgroups(
        MTLSize::new(out_dim as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; out_dim];
    out_buf.read_f32(&mut result);

    // Row 0: scale=0.5, all values=2 -> each element = 0.5*2 = 1.0
    //         dot with x=[1.0; 64] -> sum = 64.0
    let expected_0 = 64.0f32;
    assert!(
        (result[0] - expected_0).abs() < 0.1,
        "Q8_0 matmul row 0: GPU={}, expected={expected_0}", result[0]
    );

    // Row 1: scale=1.0, values=[0..31, 0..31] -> elements = [0,1,...,31,0,1,...,31]
    //         dot with x=[1.0; 64] -> sum = 2 * (0+1+...+31) = 2 * 496 = 992.0
    let expected_1 = 992.0f32;
    assert!(
        (result[1] - expected_1).abs() < 0.1,
        "Q8_0 matmul row 1: GPU={}, expected={expected_1}", result[1]
    );

    eprintln!("Q8_0 dequant matmul: out[0]={}, out[1]={} (expected {expected_0}, {expected_1})",
        result[0], result[1]);
}

#[test]
fn test_metal_dequant_matmul_q8_0_negative_values() {
    // Test with negative int8 values (critical for correctness).
    // Row 0: scale=2.0, values all -1 -> dequant = -2.0 per element
    // Input x = [1.0; 32]
    // Expected: out[0] = 32 * (-2.0) = -64.0

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("dequant_matmul_q8_0").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let in_dim: usize = 32;
    let out_dim: usize = 1;
    let q8_block_size: usize = 34;

    let mut w_q8 = vec![0u8; q8_block_size];

    // scale=2.0 in f16 = 0x4000
    let scale_bits = f32_to_f16_bits(2.0);
    w_q8[0] = (scale_bits & 0xFF) as u8;
    w_q8[1] = (scale_bits >> 8) as u8;
    // All values = -1 (0xFF as i8 = -1)
    for j in 0..32 {
        w_q8[2 + j] = 0xFF;
    }

    let x = vec![1.0f32; in_dim];

    let w_buf = backend.device.new_buffer_with_bytes(&w_q8).unwrap();
    let x_buf = backend.upload_f32(&x).unwrap();
    let out_buf = backend.device.new_buffer(out_dim * 4).unwrap();

    let in_dim_u32 = in_dim as u32;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&w_buf, 0, 0);
    enc.set_buffer(&x_buf, 0, 1);
    enc.set_buffer(&out_buf, 0, 2);
    enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
    enc.dispatch_threadgroups(
        MTLSize::new(out_dim as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; out_dim];
    out_buf.read_f32(&mut result);

    // scale=2.0, all i8=-1 -> dequant = 2.0 * (-1) = -2.0
    // dot with x=[1.0; 32] -> sum = 32 * (-2.0) = -64.0
    let expected = -64.0f32;
    assert!(
        (result[0] - expected).abs() < 0.1,
        "Q8_0 matmul neg: GPU={}, expected={expected}", result[0]
    );

    eprintln!("Q8_0 dequant matmul (negative): out[0]={} (expected {expected})", result[0]);
}

#[test]
fn test_metal_write_kv_cache_correctness() {
    // Verify write_kv_cache writes K at row-major [max_seq_len, kv_dim]
    // and V at transposed [kv_dim, max_seq_len], both as f16.
    //
    // Setup: kv_dim=4, write to seq_pos=2 in a cache sized for 4 positions.
    // After write: K[2*4..3*4] should contain K values (row-major, f16).
    // V[d*4+2] should contain V[d] for each d (transposed, f16).
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("write_kv_cache").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let kv_dim: usize = 4;
    let max_seq_len: usize = 4;
    let seq_pos: u32 = 2;

    let k_new = vec![1.0f32, 2.0, 3.0, 4.0];
    let v_new = vec![5.0f32, 6.0, 7.0, 8.0];

    let k_new_buf = backend.upload_f32(&k_new).unwrap();
    let v_new_buf = backend.upload_f32(&v_new).unwrap();

    // Pre-fill cache with zeros (f16 = 2 bytes per element)
    let k_cache_buf = backend.device.new_buffer(max_seq_len * kv_dim * 2).unwrap();
    let v_cache_buf = backend.device.new_buffer(max_seq_len * kv_dim * 2).unwrap();
    k_cache_buf.write_u16(&vec![0u16; max_seq_len * kv_dim]);
    v_cache_buf.write_u16(&vec![0u16; max_seq_len * kv_dim]);

    let kv_dim_u32 = kv_dim as u32;
    let max_seq_len_u32 = max_seq_len as u32;

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&k_new_buf, 0, 0);
    enc.set_buffer(&v_new_buf, 0, 1);
    enc.set_buffer(&k_cache_buf, 0, 2);
    enc.set_buffer(&v_cache_buf, 0, 3);
    enc.set_bytes(&kv_dim_u32.to_le_bytes(), 4);
    enc.set_bytes(&seq_pos.to_le_bytes(), 5);
    enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 6);
    enc.dispatch_threadgroups(
        MTLSize::new(kv_dim as u64, 1, 1),
        MTLSize::new(kv_dim as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    // Read back as u16 (f16 bits) and convert to f32 for comparison
    let mut k_result_u16 = vec![0u16; max_seq_len * kv_dim];
    let mut v_result_u16 = vec![0u16; max_seq_len * kv_dim];
    k_cache_buf.read_u16(&mut k_result_u16);
    v_cache_buf.read_u16(&mut v_result_u16);
    let k_result = f16_slice_to_f32(&k_result_u16);
    let v_result = f16_slice_to_f32(&v_result_u16);

    // K: row-major [max_seq_len, kv_dim]. Position 2 should have our data.
    let start = seq_pos as usize * kv_dim;
    assert_eq!(&k_result[0..start], &vec![0.0f32; start][..], "K before write pos should be zero");
    assert_eq!(&k_result[start..start + kv_dim], &k_new[..], "K at write pos should match input");

    // V: transposed [kv_dim, max_seq_len]. v_cache[d * max_seq_len + seq_pos] = v_new[d]
    for d in 0..kv_dim {
        let v_idx = d * max_seq_len + seq_pos as usize;
        assert_eq!(v_result[v_idx], v_new[d],
            "V transposed: v_cache[{d}*{max_seq_len}+{seq_pos}] = {} should be {}", v_result[v_idx], v_new[d]);
    }

    eprintln!("write_kv_cache: K[{start}..{}] = {:?}", start + kv_dim, &k_result[start..start + kv_dim]);
}

/// Upload f32 data as f16 (half) to a Metal buffer.
/// Converts each f32 to f16 bits and creates a buffer with 2 bytes per element.
fn upload_as_f16(backend: &MetalF32Backend, data: &[f32]) -> MetalBuffer {
    let f16_data = f32_slice_to_f16(data);
    let buf = backend.device.new_buffer(f16_data.len() * 2).unwrap();
    buf.write_u16(&f16_data);
    buf
}

/// Transpose V cache from [seq_len, kv_dim] to [kv_dim, max_seq_len] layout.
fn transpose_v_cache(v_row_major: &[f32], seq_len: usize, kv_dim: usize, max_seq_len: usize) -> Vec<f32> {
    let mut v_transposed = vec![0.0f32; kv_dim * max_seq_len];
    for t in 0..seq_len {
        for d in 0..kv_dim {
            v_transposed[d * max_seq_len + t] = v_row_major[t * kv_dim + d];
        }
    }
    v_transposed
}

#[test]
fn test_metal_multi_head_attention_single_head() {
    // Test the fused multi_head_attention kernel with a simple single-head case.
    //
    // Q = [1, 0, 0, 0], head_dim=4, seq_len=2
    // K cache = [[1,0,0,0], [0,1,0,0]]
    // V cache = [[10,20,30,40], [50,60,70,80]] (stored transposed)
    //
    // score[0] = dot(Q, K[0]) * scale = 1.0 * 0.5 = 0.5
    // score[1] = dot(Q, K[1]) * scale = 0.0 * 0.5 = 0.0
    // softmax([0.5, 0.0]) -> [w0, w1], output = w0*V[0] + w1*V[1]

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("multi_head_attention").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_heads: u32 = 1;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 4;
    let kv_dim: u32 = 4;
    let seq_len: u32 = 2;
    let max_seq_len: u32 = 2;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let k_cache = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v_cache_row = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let v_cache = transpose_v_cache(&v_cache_row, seq_len as usize, kv_dim as usize, max_seq_len as usize);

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend.device.new_buffer(head_dim as usize * 4).unwrap();
    let scores_buf = backend.device.new_buffer((num_heads * seq_len) as usize * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&q_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&out_buf, 0, 3);
    enc.set_buffer(&scores_buf, 0, 4);
    enc.set_bytes(&num_heads.to_le_bytes(), 5);
    enc.set_bytes(&num_kv_heads.to_le_bytes(), 6);
    enc.set_bytes(&head_dim.to_le_bytes(), 7);
    enc.set_bytes(&kv_dim.to_le_bytes(), 8);
    enc.set_bytes(&seq_len.to_le_bytes(), 9);
    enc.set_bytes(&scale.to_le_bytes(), 10);
    enc.set_bytes(&max_seq_len.to_le_bytes(), 11);
    enc.dispatch_threadgroups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; head_dim as usize];
    out_buf.read_f32(&mut result);

    let w0 = (0.5f32).exp() / ((0.5f32).exp() + 1.0f32);
    let w1 = 1.0 - w0;
    for d in 0..4 {
        let expected = w0 * v_cache_row[d] + w1 * v_cache_row[4 + d];
        assert!(
            (result[d] - expected).abs() < 0.01,
            "MHA out[{d}]: GPU={}, expected={expected}", result[d]
        );
    }
    eprintln!("multi_head_attention (1 head): {:?}", result);
}

#[test]
fn test_metal_multi_head_attention_gqa() {
    // Test GQA: 4 query heads sharing 2 KV heads (gqa_ratio=2).
    // head_dim=2, kv_dim=4, seq_len=1.
    // Heads 0,1 -> kv_head 0; Heads 2,3 -> kv_head 1.
    // With seq_len=1, output = V[kv_head].

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("multi_head_attention").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_heads: u32 = 4;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 2;
    let kv_dim: u32 = 4;
    let seq_len: u32 = 1;
    let max_seq_len: u32 = 1;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let k_cache = vec![1.0f32, 0.0, 0.0, 1.0];
    let v_cache_row = vec![10.0f32, 20.0, 30.0, 40.0];
    let v_cache = transpose_v_cache(&v_cache_row, seq_len as usize, kv_dim as usize, max_seq_len as usize);

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend.device.new_buffer((num_heads * head_dim) as usize * 4).unwrap();
    let scores_buf = backend.device.new_buffer((num_heads * seq_len) as usize * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&q_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&out_buf, 0, 3);
    enc.set_buffer(&scores_buf, 0, 4);
    enc.set_bytes(&num_heads.to_le_bytes(), 5);
    enc.set_bytes(&num_kv_heads.to_le_bytes(), 6);
    enc.set_bytes(&head_dim.to_le_bytes(), 7);
    enc.set_bytes(&kv_dim.to_le_bytes(), 8);
    enc.set_bytes(&seq_len.to_le_bytes(), 9);
    enc.set_bytes(&scale.to_le_bytes(), 10);
    enc.set_bytes(&max_seq_len.to_le_bytes(), 11);
    enc.dispatch_threadgroups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; (num_heads * head_dim) as usize];
    out_buf.read_f32(&mut result);

    // seq_len=1 -> softmax([score]) = [1.0], output = V[kv_head]
    let expected = [10.0f32, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0];
    for i in 0..8 {
        assert!(
            (result[i] - expected[i]).abs() < 0.01,
            "MHA GQA out[{i}]: GPU={}, expected={}", result[i], expected[i]
        );
    }
    eprintln!("multi_head_attention (GQA 4q/2kv): {:?}", result);
}

#[test]
fn test_metal_multi_head_attention_uniform_scores() {
    // Q=[0,0] -> all dot products = 0 -> uniform attention -> output = mean(V)

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("multi_head_attention").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    let num_heads: u32 = 1;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 2;
    let kv_dim: u32 = 2;
    let seq_len: u32 = 3;
    let max_seq_len: u32 = 3;
    let scale: f32 = 1.0;

    let q = vec![0.0f32, 0.0];
    let k_cache = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v_cache_row = vec![3.0f32, 6.0, 9.0, 12.0, 15.0, 18.0];
    let v_cache = transpose_v_cache(&v_cache_row, seq_len as usize, kv_dim as usize, max_seq_len as usize);

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend.device.new_buffer(head_dim as usize * 4).unwrap();
    let scores_buf = backend.device.new_buffer((num_heads * seq_len) as usize * 4).unwrap();

    let cmd = backend.queue.new_command_buffer().unwrap();
    let enc = cmd.new_compute_encoder().unwrap();
    enc.set_pipeline_state(&pso);
    enc.set_buffer(&q_buf, 0, 0);
    enc.set_buffer(&k_buf, 0, 1);
    enc.set_buffer(&v_buf, 0, 2);
    enc.set_buffer(&out_buf, 0, 3);
    enc.set_buffer(&scores_buf, 0, 4);
    enc.set_bytes(&num_heads.to_le_bytes(), 5);
    enc.set_bytes(&num_kv_heads.to_le_bytes(), 6);
    enc.set_bytes(&head_dim.to_le_bytes(), 7);
    enc.set_bytes(&kv_dim.to_le_bytes(), 8);
    enc.set_bytes(&seq_len.to_le_bytes(), 9);
    enc.set_bytes(&scale.to_le_bytes(), 10);
    enc.set_bytes(&max_seq_len.to_le_bytes(), 11);
    enc.dispatch_threadgroups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; head_dim as usize];
    out_buf.read_f32(&mut result);

    // Uniform attention: output = mean(V)
    let expected_0 = (3.0 + 9.0 + 15.0) / 3.0;
    let expected_1 = (6.0 + 12.0 + 18.0) / 3.0;
    assert!(
        (result[0] - expected_0).abs() < 0.01,
        "MHA uniform out[0]: GPU={}, expected={expected_0}", result[0]
    );
    assert!(
        (result[1] - expected_1).abs() < 0.01,
        "MHA uniform out[1]: GPU={}, expected={expected_1}", result[1]
    );
    eprintln!("multi_head_attention (uniform): {:?}", result);
}

#[test]
fn test_flash_decode_matches_original_mha() {
    // Verify flash decode produces the same output as the original
    // multi_head_attention kernel for a non-trivial case.
    //
    // Setup: 2 query heads, 1 KV head (GQA ratio=2), head_dim=4, seq_len=4
    // Q = [1,0,0,0] for head 0, [0,1,0,0] for head 1
    // K cache: 4 vectors, V cache: 4 vectors (transposed)
    // This exercises GQA, softmax, and the full pipeline.

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let mha_func = lib.get_function("multi_head_attention").unwrap();
    let mha_pso = backend.device.new_compute_pipeline_state(&mha_func).unwrap();
    let fd_func = lib.get_function("flash_decode_attention").unwrap();
    let fd_pso = backend.device.new_compute_pipeline_state(&fd_func).unwrap();
    let fr_func = lib.get_function("flash_decode_reduce").unwrap();
    let fr_pso = backend.device.new_compute_pipeline_state(&fr_func).unwrap();

    let num_heads: u32 = 2;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 4;
    let kv_dim: u32 = 4; // num_kv_heads * head_dim
    let seq_len: u32 = 4;
    let max_seq_len: u32 = 4;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    // Q: head 0 = [1,0,0,0], head 1 = [0,1,0,0]
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0];
    // K cache: 4 positions (row-major)
    let k_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,  // pos 0: aligns with head 0
        0.0, 1.0, 0.0, 0.0,  // pos 1: aligns with head 1
        0.5, 0.5, 0.0, 0.0,  // pos 2: partial alignment
        0.0, 0.0, 1.0, 0.0,  // pos 3: orthogonal to both
    ];
    // V cache: 4 positions (stored transposed)
    let v_data_row: Vec<f32> = vec![
        10.0, 20.0, 30.0, 40.0,  // pos 0
        50.0, 60.0, 70.0, 80.0,  // pos 1
        1.0,  2.0,  3.0,  4.0,   // pos 2
        5.0,  6.0,  7.0,  8.0,   // pos 3
    ];
    let v_data = transpose_v_cache(&v_data_row, seq_len as usize, kv_dim as usize, max_seq_len as usize);

    let q_buf = backend.upload_f32(&q_data).unwrap();
    let k_buf = upload_as_f16(&backend, &k_data);
    let v_buf = upload_as_f16(&backend, &v_data);
    let out_buf_mha = backend.device.new_buffer((num_heads * head_dim) as usize * 4).unwrap();
    let out_buf_fd  = backend.device.new_buffer((num_heads * head_dim) as usize * 4).unwrap();
    let scores_buf = backend.device.new_buffer((num_heads * seq_len) as usize * 4).unwrap();

    // Run original MHA
    let queue = backend.device.new_command_queue().unwrap();
    {
        let cmd = queue.new_command_buffer().unwrap();
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&mha_pso);
        enc.set_buffer(&q_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&v_buf, 0, 2);
        enc.set_buffer(&out_buf_mha, 0, 3);
        enc.set_buffer(&scores_buf, 0, 4);
        enc.set_bytes(&num_heads.to_le_bytes(), 5);
        enc.set_bytes(&num_kv_heads.to_le_bytes(), 6);
        enc.set_bytes(&head_dim.to_le_bytes(), 7);
        enc.set_bytes(&kv_dim.to_le_bytes(), 8);
        enc.set_bytes(&seq_len.to_le_bytes(), 9);
        enc.set_bytes(&scale.to_le_bytes(), 10);
        enc.set_bytes(&max_seq_len.to_le_bytes(), 11);
        enc.dispatch_threadgroups(
            MTLSize::new(num_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();
    }

    let mut mha_result = vec![0.0f32; (num_heads * head_dim) as usize];
    out_buf_mha.read_f32(&mut mha_result);

    // Run flash decode (tile_size=2, so 2 tiles of 2 positions each)
    let tile_kv: u32 = 2;
    let num_tiles: u32 = (seq_len + tile_kv - 1) / tile_kv;
    let partial_stride = head_dim + 2;
    let partial_size = (num_heads * num_tiles * partial_stride) as usize;
    let partial_buf = backend.device.new_buffer(partial_size * 4).unwrap();

    {
        let cmd = queue.new_command_buffer().unwrap();

        // Phase 1: flash_decode_attention
        let enc = cmd.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&fd_pso);
        enc.set_buffer(&q_buf, 0, 0);
        enc.set_buffer(&k_buf, 0, 1);
        enc.set_buffer(&v_buf, 0, 2);
        enc.set_buffer(&partial_buf, 0, 3);
        enc.set_bytes(&num_heads.to_le_bytes(), 4);
        enc.set_bytes(&num_kv_heads.to_le_bytes(), 5);
        enc.set_bytes(&head_dim.to_le_bytes(), 6);
        enc.set_bytes(&kv_dim.to_le_bytes(), 7);
        enc.set_bytes(&seq_len.to_le_bytes(), 8);
        enc.set_bytes(&scale.to_le_bytes(), 9);
        enc.set_bytes(&tile_kv.to_le_bytes(), 10);
        enc.set_bytes(&num_tiles.to_le_bytes(), 11);
        enc.set_bytes(&max_seq_len.to_le_bytes(), 12);
        let total_tgs = (num_heads * num_tiles) as u64;
        enc.dispatch_threadgroups(
            MTLSize::new(total_tgs, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();

        // Phase 2: flash_decode_reduce
        let enc2 = cmd.new_compute_encoder().unwrap();
        enc2.set_pipeline_state(&fr_pso);
        enc2.set_buffer(&partial_buf, 0, 0);
        enc2.set_buffer(&out_buf_fd, 0, 1);
        enc2.set_bytes(&num_heads.to_le_bytes(), 2);
        enc2.set_bytes(&head_dim.to_le_bytes(), 3);
        enc2.set_bytes(&num_tiles.to_le_bytes(), 4);
        enc2.dispatch_threadgroups(
            MTLSize::new(num_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc2.end_encoding();

        cmd.commit_and_wait();
    }

    let mut fd_result = vec![0.0f32; (num_heads * head_dim) as usize];
    out_buf_fd.read_f32(&mut fd_result);

    eprintln!("MHA result:  {:?}", mha_result);
    eprintln!("Flash result: {:?}", fd_result);

    // Compare: both should produce the same output within tolerance
    for i in 0..(num_heads * head_dim) as usize {
        assert!(
            (mha_result[i] - fd_result[i]).abs() < 0.01,
            "Flash decode mismatch at [{}]: MHA={}, Flash={}",
            i, mha_result[i], fd_result[i]
        );
    }
}

// ========================================================================
// Bandwidth measurement harness for matvec kernel
// ========================================================================

/// Measure effective GPU memory bandwidth of the dequant_matmul_q8_0_4row kernel
/// for a given matrix shape.
///
/// Returns (bandwidth_gb_s, elapsed_ms) for reporting.
fn measure_matvec_bandwidth(in_dim: u32, out_dim: u32, iterations: u32) -> (f64, f64) {
    let backend = MetalF32Backend::new().expect("No Metal device");

    let lib = backend.device
        .new_library_with_source(METAL_SHADER_SOURCE).unwrap();
    let func = lib.get_function("dequant_matmul_q8_0_4row").unwrap();
    let pso = backend.device.new_compute_pipeline_state(&func).unwrap();

    // Q8_0 layout: each block = 2 bytes scale (f16) + 32 bytes int8 = 34 bytes
    let num_blocks = in_dim / 32;
    let row_bytes = num_blocks as usize * 34;
    let total_weight_bytes = out_dim as usize * row_bytes;

    // Fill weight buffer with pseudo-random data (correctness irrelevant for BW test).
    // Use a simple LCG to fill bytes -- faster than rand crate and avoids dependency.
    let mut weight_data = vec![0u8; total_weight_bytes];
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for chunk in weight_data.chunks_mut(8) {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bytes = rng_state.to_le_bytes();
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = bytes[i];
        }
    }
    // Ensure scale bytes are valid f16 (avoid NaN/Inf which might cause GPU issues).
    // Set every block's scale to f16 1.0 = 0x3C00.
    for row in 0..out_dim as usize {
        for blk in 0..num_blocks as usize {
            let offset = row * row_bytes + blk * 34;
            weight_data[offset] = 0x00; // low byte of f16 1.0
            weight_data[offset + 1] = 0x3C; // high byte of f16 1.0
        }
    }

    let w_buf = backend.device.new_buffer_with_bytes(&weight_data).unwrap();
    let x_data = vec![1.0f32; in_dim as usize];
    let x_buf = backend.upload_f32(&x_data).unwrap();
    let out_buf = backend.device.new_buffer(out_dim as usize * 4).unwrap();

    let n_tg = ((out_dim as u64) + 3) / 4;

    // Warmup: 10 iterations to prime caches and GPU clocks
    let warmup_cb = backend.queue.new_command_buffer().unwrap();
    for _ in 0..10 {
        let enc = warmup_cb.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(&x_buf, 0, 1);
        enc.set_buffer(&out_buf, 0, 2);
        enc.set_bytes(&in_dim.to_le_bytes(), 3);
        enc.set_bytes(&out_dim.to_le_bytes(), 4);
        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }
    warmup_cb.commit_and_wait();

    // Timed run: encode all iterations into a single command buffer
    let cb = backend.queue.new_command_buffer().unwrap();
    for _ in 0..iterations {
        let enc = cb.new_compute_encoder().unwrap();
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(&x_buf, 0, 1);
        enc.set_buffer(&out_buf, 0, 2);
        enc.set_bytes(&in_dim.to_le_bytes(), 3);
        enc.set_bytes(&out_dim.to_le_bytes(), 4);
        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }
    let start = std::time::Instant::now();
    cb.commit_and_wait();
    let elapsed = start.elapsed();

    let total_bytes_read = total_weight_bytes as f64 * iterations as f64;
    let elapsed_s = elapsed.as_secs_f64();
    let bandwidth_gb_s = total_bytes_read / elapsed_s / 1e9;

    (bandwidth_gb_s, elapsed_s * 1000.0)
}

/// Bandwidth measurement: QKV projection (hidden_dim x hidden_dim)
/// Matrix: 2048x2048 Q8_0 (~4.5 MB, fits in L2 cache -- shows cache-hot BW)
#[test]
fn bench_matvec_bandwidth_qkv() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 2048;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: QKV Projection ===");
    println!("Matrix: {}x{} Q8_0 ({:.1} MB weights)", out_dim, in_dim, weight_mb);
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!("Note: {:.1} MB likely fits in L2 cache -- expect high BW", weight_mb);
    println!("========================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: FFN gate/up projection (hidden_dim -> ffn_dim)
/// Matrix: 5632x2048 Q8_0 (~12.3 MB, fits in L2 cache)
#[test]
fn bench_matvec_bandwidth_ffn_gate_up() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 5632;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: FFN Gate/Up Projection ===");
    println!("Matrix: {}x{} Q8_0 ({:.1} MB weights)", out_dim, in_dim, weight_mb);
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!("Note: {:.1} MB likely fits in L2 cache -- expect high BW", weight_mb);
    println!("================================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: FFN down projection (ffn_dim -> hidden_dim)
/// Matrix: 2048x5632 Q8_0 (~12.3 MB, fits in L2 cache)
#[test]
fn bench_matvec_bandwidth_ffn_down() {
    let in_dim: u32 = 5632;
    let out_dim: u32 = 2048;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: FFN Down Projection ===");
    println!("Matrix: {}x{} Q8_0 ({:.1} MB weights)", out_dim, in_dim, weight_mb);
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!("Note: {:.1} MB likely fits in L2 cache -- expect high BW", weight_mb);
    println!("=============================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: Output projection (hidden_dim -> vocab_size)
/// Matrix: 32000x2048 Q8_0 (~69.6 MB, EXCEEDS L2 cache)
/// This is the critical DRAM bandwidth measurement since the buffer is
/// larger than the L2 cache on most Apple Silicon chips.
#[test]
fn bench_matvec_bandwidth_output_proj() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 32000;
    let iterations = 100;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: Output Projection (DRAM) ===");
    println!("Matrix: {}x{} Q8_0 ({:.1} MB weights)", out_dim, in_dim, weight_mb);
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!("Note: {:.1} MB exceeds typical L2 cache -- measures true DRAM bandwidth", weight_mb);
    println!("===================================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Combined bandwidth summary across all decode-path matrix sizes.
/// Runs all four shapes and prints a comparison table.
#[test]
fn bench_matvec_bandwidth_summary() {
    let configs: &[(u32, u32, &str)] = &[
        (2048, 2048, "QKV proj (hidden x hidden)"),
        (2048, 5632, "FFN gate/up (hidden -> ffn)"),
        (5632, 2048, "FFN down (ffn -> hidden)"),
        (2048, 32000, "Output proj (hidden -> vocab)"),
    ];

    println!("\n======================================================================");
    println!("  Matvec Bandwidth Summary (dequant_matmul_q8_0_4row)");
    println!("======================================================================");
    println!("{:<35} {:>8} {:>10} {:>10}",
        "Shape", "MB", "GB/s");
    println!("{:-<35} {:->8} {:->10}", "", "", "");

    for &(in_dim, out_dim, label) in configs {
        let iterations = if out_dim >= 32000 { 100 } else { 200 };
        let (bw, _elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);
        let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;

        println!("{:<35} {:>7.1} {:>9.1}",
            label, weight_mb, bw);
    }

    println!("======================================================================");
    println!("  Large shapes exceed L2 cache and measure true DRAM bandwidth.");
    println!("======================================================================\n");
}


