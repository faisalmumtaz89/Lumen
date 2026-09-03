use crate::metal::ffi::{MTLSize, MetalBuffer};
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::*;

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
    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    enc.dispatch_threadgroups(MTLSize::new(2, 1, 1), MTLSize::new(tg_size, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 2];
    out_buf.read_f32(&mut result);
    assert_eq!(
        result,
        vec![4.0, 10.0],
        "GPU matmul should match CPU: [1,2,3]*[1,0,1]=4, [4,5,6]*[1,0,1]=10"
    );
}

#[test]
fn test_metal_rmsnorm_correctness() {
    let backend = MetalF32Backend::new().unwrap();

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    enc.dispatch_threads(MTLSize::new(3, 1, 1), MTLSize::new(3, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; 3];
    gate_buf.read_f32(&mut result);

    assert!((result[0] - 0.0).abs() < 1e-6, "swiglu(0)*1 = 0");
    assert!(
        (result[1] - 0.7310586).abs() < 1e-4,
        "swiglu(1)*1 ~ 0.731, got {}",
        result[1]
    );
    assert!(
        (result[2] - (-0.2689414)).abs() < 1e-4,
        "swiglu(-1)*1 ~ -0.269, got {}",
        result[2]
    );
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
        return if sign != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
        let block_start = b * q8_block_size; // row 0 starts at offset 0
        w_q8[block_start] = (scale_half_bits & 0xFF) as u8;
        w_q8[block_start + 1] = (scale_half_bits >> 8) as u8;
        for j in 0..q8_group_size {
            w_q8[block_start + 2 + j] = 2u8;
        }
    }

    // Row 1: 2 blocks, each with scale=1.0, values=[0,1,2,...,31]
    let scale_one_bits: u16 = 0x3C00;
    for b in 0..num_blocks_per_row {
        let block_start = row_bytes + b * q8_block_size; // row 1 starts at row_bytes
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
        "Q8_0 matmul row 0: GPU={}, expected={expected_0}",
        result[0]
    );

    // Row 1: scale=1.0, values=[0..31, 0..31] -> elements = [0,1,...,31,0,1,...,31]
    //         dot with x=[1.0; 64] -> sum = 2 * (0+1+...+31) = 2 * 496 = 992.0
    let expected_1 = 992.0f32;
    assert!(
        (result[1] - expected_1).abs() < 0.1,
        "Q8_0 matmul row 1: GPU={}, expected={expected_1}",
        result[1]
    );

    eprintln!(
        "Q8_0 dequant matmul: out[0]={}, out[1]={} (expected {expected_0}, {expected_1})",
        result[0], result[1]
    );
}

#[test]
fn test_metal_dequant_matmul_q8_0_negative_values() {
    // Test with negative int8 values (critical for correctness).
    // Row 0: scale=2.0, values all -1 -> dequant = -2.0 per element
    // Input x = [1.0; 32]
    // Expected: out[0] = 32 * (-2.0) = -64.0

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    enc.dispatch_threadgroups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; out_dim];
    out_buf.read_f32(&mut result);

    // scale=2.0, all i8=-1 -> dequant = 2.0 * (-1) = -2.0
    // dot with x=[1.0; 32] -> sum = 32 * (-2.0) = -64.0
    let expected = -64.0f32;
    assert!(
        (result[0] - expected).abs() < 0.1,
        "Q8_0 matmul neg: GPU={}, expected={expected}",
        result[0]
    );

    eprintln!(
        "Q8_0 dequant matmul (negative): out[0]={} (expected {expected})",
        result[0]
    );
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    assert_eq!(
        &k_result[0..start],
        &vec![0.0f32; start][..],
        "K before write pos should be zero"
    );
    assert_eq!(
        &k_result[start..start + kv_dim],
        &k_new[..],
        "K at write pos should match input"
    );

    // V: transposed [kv_dim, max_seq_len]. v_cache[d * max_seq_len + seq_pos] = v_new[d]
    for d in 0..kv_dim {
        let v_idx = d * max_seq_len + seq_pos as usize;
        assert_eq!(
            v_result[v_idx], v_new[d],
            "V transposed: v_cache[{d}*{max_seq_len}+{seq_pos}] = {} should be {}",
            v_result[v_idx], v_new[d]
        );
    }

    eprintln!(
        "write_kv_cache: K[{start}..{}] = {:?}",
        start + kv_dim,
        &k_result[start..start + kv_dim]
    );
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
fn transpose_v_cache(
    v_row_major: &[f32],
    seq_len: usize,
    kv_dim: usize,
    max_seq_len: usize,
) -> Vec<f32> {
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    let v_cache = transpose_v_cache(
        &v_cache_row,
        seq_len as usize,
        kv_dim as usize,
        max_seq_len as usize,
    );

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend.device.new_buffer(head_dim as usize * 4).unwrap();
    let scores_buf = backend
        .device
        .new_buffer((num_heads * seq_len) as usize * 4)
        .unwrap();

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
    enc.dispatch_threadgroups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
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
            "MHA out[{d}]: GPU={}, expected={expected}",
            result[d]
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    let v_cache = transpose_v_cache(
        &v_cache_row,
        seq_len as usize,
        kv_dim as usize,
        max_seq_len as usize,
    );

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend
        .device
        .new_buffer((num_heads * head_dim) as usize * 4)
        .unwrap();
    let scores_buf = backend
        .device
        .new_buffer((num_heads * seq_len) as usize * 4)
        .unwrap();

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
    enc.dispatch_threadgroups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; (num_heads * head_dim) as usize];
    out_buf.read_f32(&mut result);

    // seq_len=1 -> softmax([score]) = [1.0], output = V[kv_head]
    let expected = [10.0f32, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0];
    for i in 0..8 {
        assert!(
            (result[i] - expected[i]).abs() < 0.01,
            "MHA GQA out[{i}]: GPU={}, expected={}",
            result[i],
            expected[i]
        );
    }
    eprintln!("multi_head_attention (GQA 4q/2kv): {:?}", result);
}

#[test]
fn test_metal_multi_head_attention_uniform_scores() {
    // Q=[0,0] -> all dot products = 0 -> uniform attention -> output = mean(V)

    let backend = MetalF32Backend::new().unwrap();

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
    let v_cache = transpose_v_cache(
        &v_cache_row,
        seq_len as usize,
        kv_dim as usize,
        max_seq_len as usize,
    );

    let q_buf = backend.upload_f32(&q).unwrap();
    let k_buf = upload_as_f16(&backend, &k_cache);
    let v_buf = upload_as_f16(&backend, &v_cache);
    let out_buf = backend.device.new_buffer(head_dim as usize * 4).unwrap();
    let scores_buf = backend
        .device
        .new_buffer((num_heads * seq_len) as usize * 4)
        .unwrap();

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
    enc.dispatch_threadgroups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit_and_wait();

    let mut result = vec![0.0f32; head_dim as usize];
    out_buf.read_f32(&mut result);

    // Uniform attention: output = mean(V)
    let expected_0 = (3.0 + 9.0 + 15.0) / 3.0;
    let expected_1 = (6.0 + 12.0 + 18.0) / 3.0;
    assert!(
        (result[0] - expected_0).abs() < 0.01,
        "MHA uniform out[0]: GPU={}, expected={expected_0}",
        result[0]
    );
    assert!(
        (result[1] - expected_1).abs() < 0.01,
        "MHA uniform out[1]: GPU={}, expected={expected_1}",
        result[1]
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
    let mha_func = lib.get_function("multi_head_attention").unwrap();
    let mha_pso = backend
        .device
        .new_compute_pipeline_state(&mha_func)
        .unwrap();
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
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    // K cache: 4 positions (row-major)
    let k_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // pos 0: aligns with head 0
        0.0, 1.0, 0.0, 0.0, // pos 1: aligns with head 1
        0.5, 0.5, 0.0, 0.0, // pos 2: partial alignment
        0.0, 0.0, 1.0, 0.0, // pos 3: orthogonal to both
    ];
    // V cache: 4 positions (stored transposed)
    let v_data_row: Vec<f32> = vec![
        10.0, 20.0, 30.0, 40.0, // pos 0
        50.0, 60.0, 70.0, 80.0, // pos 1
        1.0, 2.0, 3.0, 4.0, // pos 2
        5.0, 6.0, 7.0, 8.0, // pos 3
    ];
    let v_data = transpose_v_cache(
        &v_data_row,
        seq_len as usize,
        kv_dim as usize,
        max_seq_len as usize,
    );

    let q_buf = backend.upload_f32(&q_data).unwrap();
    let k_buf = upload_as_f16(&backend, &k_data);
    let v_buf = upload_as_f16(&backend, &v_data);
    let out_buf_mha = backend
        .device
        .new_buffer((num_heads * head_dim) as usize * 4)
        .unwrap();
    let out_buf_fd = backend
        .device
        .new_buffer((num_heads * head_dim) as usize * 4)
        .unwrap();
    let scores_buf = backend
        .device
        .new_buffer((num_heads * seq_len) as usize * 4)
        .unwrap();

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
        enc.dispatch_threadgroups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
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
        enc.dispatch_threadgroups(MTLSize::new(total_tgs, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();

        // Phase 2: flash_decode_reduce
        let enc2 = cmd.new_compute_encoder().unwrap();
        enc2.set_pipeline_state(&fr_pso);
        enc2.set_buffer(&partial_buf, 0, 0);
        enc2.set_buffer(&out_buf_fd, 0, 1);
        enc2.set_bytes(&num_heads.to_le_bytes(), 2);
        enc2.set_bytes(&head_dim.to_le_bytes(), 3);
        enc2.set_bytes(&num_tiles.to_le_bytes(), 4);
        enc2.dispatch_threadgroups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
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
            i,
            mha_result[i],
            fd_result[i]
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

    let lib = backend
        .device
        .new_library_with_source(METAL_SHADER_SOURCE)
        .unwrap();
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
// Perf benchmark, not a correctness test: asserts a GPU throughput floor
// (`bw > 50 GB/s`) that is not a stable invariant under the default
// multithreaded runner (N-way GPU contention drops cache-hot bandwidth
// below the floor). Ignored from the default `--lib` gate; run the perf
// pass explicitly with `cargo test -p lumen-runtime --lib -- --ignored`.
#[test]
#[ignore = "perf benchmark: GPU bandwidth floor is contention-sensitive; run with --ignored"]
fn bench_matvec_bandwidth_qkv() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 2048;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: QKV Projection ===");
    println!(
        "Matrix: {}x{} Q8_0 ({:.1} MB weights)",
        out_dim, in_dim, weight_mb
    );
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!(
        "Note: {:.1} MB likely fits in L2 cache -- expect high BW",
        weight_mb
    );
    println!("========================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: FFN gate/up projection (hidden_dim -> ffn_dim)
/// Matrix: 5632x2048 Q8_0 (~12.3 MB, fits in L2 cache)
// Perf benchmark (see `bench_matvec_bandwidth_qkv`): contention-sensitive
// GPU throughput floor, ignored from the default gate; run with `--ignored`.
#[test]
#[ignore = "perf benchmark: GPU bandwidth floor is contention-sensitive; run with --ignored"]
fn bench_matvec_bandwidth_ffn_gate_up() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 5632;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: FFN Gate/Up Projection ===");
    println!(
        "Matrix: {}x{} Q8_0 ({:.1} MB weights)",
        out_dim, in_dim, weight_mb
    );
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!(
        "Note: {:.1} MB likely fits in L2 cache -- expect high BW",
        weight_mb
    );
    println!("================================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: FFN down projection (ffn_dim -> hidden_dim)
/// Matrix: 2048x5632 Q8_0 (~12.3 MB, fits in L2 cache)
// Perf benchmark (see `bench_matvec_bandwidth_qkv`): contention-sensitive
// GPU throughput floor, ignored from the default gate; run with `--ignored`.
#[test]
#[ignore = "perf benchmark: GPU bandwidth floor is contention-sensitive; run with --ignored"]
fn bench_matvec_bandwidth_ffn_down() {
    let in_dim: u32 = 5632;
    let out_dim: u32 = 2048;
    let iterations = 200;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: FFN Down Projection ===");
    println!(
        "Matrix: {}x{} Q8_0 ({:.1} MB weights)",
        out_dim, in_dim, weight_mb
    );
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!(
        "Note: {:.1} MB likely fits in L2 cache -- expect high BW",
        weight_mb
    );
    println!("=============================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Bandwidth measurement: Output projection (hidden_dim -> vocab_size)
/// Matrix: 32000x2048 Q8_0 (~69.6 MB, EXCEEDS L2 cache)
/// This is the critical DRAM bandwidth measurement since the buffer is
/// larger than the L2 cache on most Apple Silicon chips.
// Perf benchmark (see `bench_matvec_bandwidth_qkv`): contention-sensitive
// GPU throughput floor, ignored from the default gate; run with `--ignored`.
#[test]
#[ignore = "perf benchmark: GPU bandwidth floor is contention-sensitive; run with --ignored"]
fn bench_matvec_bandwidth_output_proj() {
    let in_dim: u32 = 2048;
    let out_dim: u32 = 32000;
    let iterations = 100;
    let (bw, elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);

    let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;
    println!("\n=== Matvec Bandwidth: Output Projection (DRAM) ===");
    println!(
        "Matrix: {}x{} Q8_0 ({:.1} MB weights)",
        out_dim, in_dim, weight_mb
    );
    println!("Iterations: {}", iterations);
    println!("Elapsed: {:.2} ms", elapsed_ms);
    println!("Effective bandwidth: {:.1} GB/s", bw);
    println!(
        "Note: {:.1} MB exceeds typical L2 cache -- measures true DRAM bandwidth",
        weight_mb
    );
    println!("===================================================\n");

    assert!(bw > 50.0, "Bandwidth too low: {:.1} GB/s", bw);
}

/// Combined bandwidth summary across all decode-path matrix sizes.
/// Runs all four shapes and prints a comparison table.
// Perf benchmark (see `bench_matvec_bandwidth_qkv`): pure measurement table,
// no correctness assertion, but adds GPU contention; ignored from the default
// gate to keep it deterministic. Run with `--ignored` for the perf pass.
#[test]
#[ignore = "perf benchmark: measurement-only; run with --ignored"]
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
    println!("{:<35} {:>8} {:>10}", "Shape", "MB", "GB/s");
    println!("{:-<35} {:->8} {:->10}", "", "", "");

    for &(in_dim, out_dim, label) in configs {
        let iterations = if out_dim >= 32000 { 100 } else { 200 };
        let (bw, _elapsed_ms) = measure_matvec_bandwidth(in_dim, out_dim, iterations);
        let weight_mb = (out_dim as usize * (in_dim / 32) as usize * 34) as f64 / 1e6;

        println!("{:<35} {:>7.1} {:>9.1}", label, weight_mb, bw);
    }

    println!("======================================================================");
    println!("  Large shapes exceed L2 cache and measure true DRAM bandwidth.");
    println!("======================================================================\n");
}

// ---- Metal `validate_kv_precision` tests ------------------------------

#[test]
fn metal_validate_kv_precision_accepts_f16() {
    use crate::compute::ComputeBackend;
    use crate::kv::KvPrecision;
    let backend = MetalF32Backend::new().unwrap();
    assert!(
        backend.validate_kv_precision(KvPrecision::F16).is_ok(),
        "Metal must accept F16 KV precision (its hardcoded layout)",
    );
}

#[test]
fn metal_validate_kv_precision_rejects_f32() {
    use crate::compute::ComputeBackend;
    use crate::error::RuntimeError;
    use crate::kv::KvPrecision;
    let backend = MetalF32Backend::new().unwrap();
    let result = backend.validate_kv_precision(KvPrecision::F32);
    assert!(
        matches!(result, Err(RuntimeError::Unsupported(_))),
        "Metal must reject F32 KV with Unsupported error, got {result:?}",
    );
    if let Err(RuntimeError::Unsupported(msg)) = result {
        assert!(
            msg.contains("F16-only") && msg.contains("Metal"),
            "error message should be actionable: {msg}",
        );
    }
}

#[test]
fn metal_validate_kv_precision_rejects_int_quantized() {
    use crate::compute::ComputeBackend;
    use crate::error::RuntimeError;
    use crate::kv::KvPrecision;
    let backend = MetalF32Backend::new().unwrap();
    for p in [KvPrecision::Int8, KvPrecision::Int4] {
        let result = backend.validate_kv_precision(p);
        assert!(
            matches!(result, Err(RuntimeError::Unsupported(_))),
            "Metal must reject {p:?} KV with Unsupported error",
        );
    }
}

/// A full-attention layer with a zero-length or wrong-length `wo` must be
/// rejected at every Metal load path: the Wo matvec derives its geometry
/// from hyperparams and would otherwise read `hidden` rows x `q_dim`
/// columns of in-bounds bytes starting at wo's recorded offset — silently
/// consuming the following tensors' bytes (`w_gate` onward in this
/// fixture) as output weights. The zero-length case exercises the
/// mandatory-presence rule, the wrong-length case the wo geometry rule;
/// both drive a real file through provider open + init + prefill, so the
/// validation call inside the load path itself is load-bearing.
#[test]
fn bad_wo_rejected_at_load() {
    use crate::compute::ComputeBackend;
    use crate::weight::provider_sync::SyncWeightProvider;
    use lumen_format::header::LbcHeader;
    use lumen_format::hyperparams::{ModelHyperparams, RopeParams};
    use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
    use lumen_format::writer::{write_lbc, GlobalTensors};

    let (hidden, inter, heads, kv_heads, head_dim, vocab) =
        (64usize, 128usize, 2u32, 2u32, 32u32, 256usize);
    let q_dim = heads as usize * head_dim as usize;
    let kv_dim = kv_heads as usize * head_dim as usize;

    let q8_bytes = |n: usize| vec![0u8; n / 32 * 34];
    let norm_bytes = |n: usize| {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0f32.to_le_bytes());
        }
        v
    };
    // (0, presence rule) and (1024 = nonzero wrong length, geometry rule):
    // the second case is what keeps the wo-geometry line itself
    // load-bearing — presence alone would also catch a zero length.
    for (wo_len, want) in [(0u64, "empty wk/wv/wo"), (1024u64, "wo is 1024 bytes")] {
        let mut blob = Vec::new();
        let mut offset = 0u64;
        let add = |data: Vec<u8>, quant: QuantScheme, blob: &mut Vec<u8>, offset: &mut u64| {
            let s = TensorSlice {
                offset: *offset,
                length: data.len() as u64,
                quant,
            };
            blob.extend_from_slice(&data);
            *offset += s.length;
            s
        };
        let wq = add(
            q8_bytes(q_dim * hidden),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        let wk = add(
            q8_bytes(kv_dim * hidden),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        let wv = add(
            q8_bytes(kv_dim * hidden),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        // The defect under test: wo's index entry disagrees with the dispatch
        // geometry (no wo bytes are appended, so a nonzero length aliases the
        // following w_gate bytes).
        let wo = TensorSlice {
            offset,
            length: wo_len,
            quant: QuantScheme::Q8_0,
        };
        let w_gate = add(
            q8_bytes(inter * hidden),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        let w_up = add(
            q8_bytes(inter * hidden),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        let w_down = add(
            q8_bytes(hidden * inter),
            QuantScheme::Q8_0,
            &mut blob,
            &mut offset,
        );
        let attn_norm = add(norm_bytes(hidden), QuantScheme::F32, &mut blob, &mut offset);
        let ffn_norm = add(norm_bytes(hidden), QuantScheme::F32, &mut blob, &mut offset);

        let subtensors = SubtensorOffsets {
            wq,
            wk,
            wv,
            wo,
            bq: None,
            bk: None,
            bv: None,
            w_gate,
            w_up,
            w_down,
            attn_norm,
            ffn_norm,
            router_weight: None,
            experts: None,
            shared_expert_gate: None,
            shared_expert_up: None,
            shared_expert_down: None,
            attn_gate: None,
            attn_post_norm: None,
            ssm_a: None,
            ssm_conv1d: None,
            ssm_dt: None,
            ssm_beta: None,
            ssm_alpha: None,
            ssm_norm: None,
            ssm_out: None,
            attn_q_norm: None,
            attn_k_norm: None,
            ffn_gate_inp_shexp: None,
            layer_type: Some(0),
        };
        let layer_indices = vec![LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors,
        }];

        let hp = ModelHyperparams {
            num_layers: 1,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            hidden_dim: hidden as u32,
            intermediate_dim: inter as u32,
            vocab_size: vocab as u32,
            max_seq_len: 512,
            rope_params: Some(RopeParams::default()),
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-5,
            rotary_dim: None,
            rope_neox: false,
            gdn: None,
        };
        let qd = QuantizationDescriptor {
            scheme: QuantScheme::Q8_0,
            group_size: QuantGroupSize::Group(32),
            block_byte_size: 34,
            scale_offset_in_block: Some(0),
        };
        let mut header = LbcHeader::new(hp, qd);
        header.embedding.quant = QuantScheme::Q8_0;
        header.output_proj.quant = QuantScheme::Q8_0;
        header.final_norm.quant = QuantScheme::F32;
        let globals = GlobalTensors {
            embedding: q8_bytes(vocab * hidden),
            final_norm: norm_bytes(hidden),
            output_proj: q8_bytes(vocab * hidden),
        };
        let mut bytes = Vec::new();
        write_lbc(
            &mut bytes,
            &header,
            &layer_indices,
            &globals,
            &[blob.as_slice()],
            None,
        )
        .unwrap();

        let dir = std::env::temp_dir().join(format!("lumen_zero_wo_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_wo.lbc");
        std::fs::write(&path, &bytes).unwrap();

        let provider = SyncWeightProvider::open(&path).unwrap();
        let hp = provider.lbc().header.hyperparams;

        // Site 1 — resident preload.
        {
            let mut backend = MetalF32Backend::new().unwrap();
            backend.set_global_tensors(
                provider.embedding.clone(),
                provider.final_norm.clone(),
                provider.output_proj.clone(),
            );
            backend.init(&hp).unwrap();
            let err = match backend.preload_weights_gpu_resident(&provider) {
                Ok(()) => {
                    panic!("wo_len {wo_len}: resident preload must fail Metal load validation")
                }
                Err(e) => e,
            };
            assert!(
                err.to_string().contains(want),
                "wo_len {wo_len}: preload rejection must carry {want:?}: {err}"
            );
        }

        // Site 2 — streaming create_layer_buffer, reached via prefill.
        let mut backend = MetalF32Backend::new().unwrap();
        backend.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        backend.init(&hp).unwrap();
        let mut kv = crate::kv::KvCache::new(crate::kv::KvCacheConfig {
            max_seq_len: hp.max_seq_len as usize,
            num_layers: hp.num_layers as usize,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: crate::kv::KvPrecision::F32,
        })
        .unwrap();
        let result = backend.prefill(
            &[1, 2, 3],
            &provider as &dyn crate::weight::cache::WeightProvider,
            &mut kv,
        );
        std::fs::remove_dir_all(&dir).ok();
        let err = match result {
            Ok(_) => panic!("wo_len {wo_len}: model must fail Metal load validation"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains(want),
            "wo_len {wo_len}: rejection must carry {want:?}: {err}"
        );
    }
}

/// Which guard rule a MoE layer trips, or none.
#[derive(Clone, Copy)]
enum MoeDefect {
    /// wo's index length disagrees with the dispatch geometry.
    WoGeometry,
    /// The fused QKV tensors sit in a K-quant scheme Metal has no kernels
    /// for (uniform across wq/wk/wv, so only the scheme is at fault).
    KQuantDense,
    /// The layer carries fewer experts than the header declares.
    ExpertCount,
    /// Well-formed.
    Sound,
}

impl MoeDefect {
    fn refusal(self) -> [&'static str; 2] {
        match self {
            Self::WoGeometry => ["wo is", "the dispatch expects"],
            Self::KQuantDense => ["tensor 'wq' is Q2_K", "no dense DECODE dispatch kernels"],
            Self::ExpertCount => ["MoE layer carries 1 experts", "the model header"],
            Self::Sound => ["", ""],
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::WoGeometry => "wo",
            Self::KQuantDense => "kquant",
            Self::ExpertCount => "experts",
            Self::Sound => "sound",
        }
    }
}

/// A one-layer, two-expert MoE LBC carrying the given defect and nothing
/// else. Returns the path and the number of experts the header declares.
fn write_moe_layer_lbc(dir: &std::path::Path, defect: MoeDefect) -> (std::path::PathBuf, usize) {
    use lumen_format::header::LbcHeader;
    use lumen_format::hyperparams::{ModelHyperparams, RopeParams};
    use lumen_format::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
    use lumen_format::writer::{write_lbc, GlobalTensors};

    let (hidden, inter, heads, kv_heads, head_dim, vocab, num_experts) =
        (64usize, 128usize, 2u32, 2u32, 32u32, 256usize, 2usize);
    let q_dim = heads as usize * head_dim as usize;
    let kv_dim = kv_heads as usize * head_dim as usize;
    let q8_bytes = |n: usize| vec![0u8; n / 32 * 34];
    let f32_bytes = |n: usize| {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0f32.to_le_bytes());
        }
        v
    };
    let mut blob = Vec::new();
    let mut offset = 0u64;
    let mut add = |data: Vec<u8>, quant: QuantScheme| {
        let s = TensorSlice {
            offset,
            length: data.len() as u64,
            quant,
        };
        blob.extend_from_slice(&data);
        offset += s.length;
        s
    };
    let qkv_quant = match defect {
        MoeDefect::KQuantDense => QuantScheme::Q2_K,
        _ => QuantScheme::Q8_0,
    };
    let wq = add(q8_bytes(q_dim * hidden), qkv_quant);
    let wk = add(q8_bytes(kv_dim * hidden), qkv_quant);
    let wv = add(q8_bytes(kv_dim * hidden), qkv_quant);
    let wo = match defect {
        MoeDefect::WoGeometry => TensorSlice {
            length: 1024,
            ..add(Vec::new(), QuantScheme::Q8_0)
        },
        _ => add(q8_bytes(hidden * q_dim), QuantScheme::Q8_0),
    };
    let attn_norm = add(f32_bytes(hidden), QuantScheme::F32);
    let ffn_norm = add(f32_bytes(hidden), QuantScheme::F32);
    let router_weight = add(f32_bytes(num_experts * hidden), QuantScheme::F32);
    let zero = TensorSlice {
        offset: 0,
        length: 0,
        quant: QuantScheme::F32,
    };
    let present = match defect {
        MoeDefect::ExpertCount => 1,
        _ => num_experts,
    };
    let experts: Vec<ExpertSlice> = (0..present)
        .map(|_| ExpertSlice {
            gate: add(q8_bytes(inter * hidden), QuantScheme::Q8_0),
            up: add(q8_bytes(inter * hidden), QuantScheme::Q8_0),
            down: add(q8_bytes(hidden * inter), QuantScheme::Q8_0),
        })
        .collect();
    let subtensors = SubtensorOffsets {
        wq,
        wk,
        wv,
        wo,
        bq: None,
        bk: None,
        bv: None,
        w_gate: zero,
        w_up: zero,
        w_down: zero,
        attn_norm,
        ffn_norm,
        router_weight: Some(router_weight),
        experts: Some(experts),
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate: None,
        attn_post_norm: None,
        ssm_a: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: Some(0),
    };
    let layer_indices = vec![LayerIndex {
        layer_offset_bytes: 0,
        layer_length_bytes: blob.len() as u64,
        subtensors,
    }];
    let hp = ModelHyperparams {
        num_layers: 1,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        hidden_dim: hidden as u32,
        intermediate_dim: inter as u32,
        vocab_size: vocab as u32,
        max_seq_len: 512,
        rope_params: Some(RopeParams::default()),
        num_experts: Some(num_experts as u32),
        num_active_experts: Some(1),
        norm_eps: 1e-5,
        rotary_dim: None,
        rope_neox: false,
        gdn: None,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::Q8_0,
        group_size: QuantGroupSize::Group(32),
        block_byte_size: 34,
        scale_offset_in_block: Some(0),
    };
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::Q8_0;
    header.output_proj.quant = QuantScheme::Q8_0;
    header.final_norm.quant = QuantScheme::F32;
    let globals = GlobalTensors {
        embedding: q8_bytes(vocab * hidden),
        final_norm: f32_bytes(hidden),
        output_proj: q8_bytes(vocab * hidden),
    };
    let mut bytes = Vec::new();
    write_lbc(
        &mut bytes,
        &header,
        &layer_indices,
        &globals,
        &[blob.as_slice()],
        None,
    )
    .unwrap();
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("moe_layer_{}.lbc", defect.tag()));
    std::fs::write(&path, &bytes).unwrap();
    (path, num_experts)
}

/// The engine's streaming `compute_layer` over a cache that already holds
/// every expert of layer 0, which is the only way the expert-cache partial
/// buffer is reached. The cache preconditions are asserted so an empty cache
/// cannot let the call fall through to the full layer buffer.
fn compute_layer_over_warmed_cache(
    provider: &crate::weight::provider_sync::SyncWeightProvider,
    path: &std::path::Path,
    num_experts: usize,
) -> (Result<(), crate::RuntimeError>, MetalF32Backend) {
    use crate::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
    use crate::weight::cache::WeightProvider;
    let hp = provider.lbc().header.hyperparams;
    let mut backend = MetalF32Backend::new().unwrap();
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.configure_expert_cache(path, num_experts);
    backend.init(&hp).unwrap();
    {
        let mut cache = backend.expert_cache.as_ref().unwrap().lock().unwrap();
        let zero = lumen_format::index::TensorSlice {
            offset: 0,
            length: 0,
            quant: lumen_format::quantization::QuantScheme::Q8_0,
        };
        let slices = lumen_format::index::ExpertSlice {
            gate: zero,
            up: zero,
            down: zero,
        };
        for e in 0..num_experts as u32 {
            assert!(
                cache
                    .insert((0, e), vec![0u8; 34], slices.clone())
                    .is_none(),
                "expert {e} must be cached without eviction"
            );
        }
        assert!(
            (0..num_experts as u32).all(|e| cache.contains(&(0, e))),
            "every expert of layer 0 must be cached for the partial path"
        );
    }
    let layer_view = provider.get_layer_raw(0).unwrap();
    let mut x = ActivationBuffer::zeros(hp.hidden_dim as usize, ComputeDtype::F32);
    let result = backend.compute_layer(0, &mut x, &layer_view, None, 0);
    (result, backend)
}

/// Every Metal load site refuses each of the three rules the Metal sites
/// wire, on a MoE layer: the resident preload, the streaming
/// `create_layer_buffer` reached by prefill, and the expert-cache
/// `create_partial_layer_buffer` reached by the engine's streaming
/// `compute_layer` once every expert of the layer is cached. The third site
/// is defence in depth for the shipped 256-expert model, whose expert cache
/// never holds a whole layer; a decode pass over a warmed cache on a
/// small-expert-count MoE is where it is live. A defect must be refused by
/// name at all three sites, and a sound layer over the same warmed cache
/// must create the partial buffer and not the full one, which pins the
/// branch itself rather than only the cache state it reads.
#[test]
fn moe_layer_defects_rejected_at_every_metal_load_site() {
    use crate::compute::ComputeBackend;
    use crate::weight::cache::WeightProvider;
    use crate::weight::provider_sync::SyncWeightProvider;

    for defect in [
        MoeDefect::WoGeometry,
        MoeDefect::KQuantDense,
        MoeDefect::ExpertCount,
    ] {
        let dir =
            std::env::temp_dir().join(format!("lumen_moe_{}_{}", defect.tag(), std::process::id()));
        let (path, num_experts) = write_moe_layer_lbc(&dir, defect);
        let provider = SyncWeightProvider::open(&path).unwrap();
        let hp = provider.lbc().header.hyperparams;
        let [want, want2] = defect.refusal();
        let check = |site: &str, result: Result<(), crate::RuntimeError>| {
            let err = match result {
                Ok(()) => panic!("{site}: {want:?} must fail Metal load validation"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains(want) && err.contains(want2),
                "{site}: refusal must carry {want:?} and {want2:?}: {err}"
            );
        };
        let fresh = || {
            let mut backend = MetalF32Backend::new().unwrap();
            backend.set_global_tensors(
                provider.embedding.clone(),
                provider.final_norm.clone(),
                provider.output_proj.clone(),
            );
            backend
        };
        let kv = || {
            crate::kv::KvCache::new(crate::kv::KvCacheConfig {
                max_seq_len: hp.max_seq_len as usize,
                num_layers: hp.num_layers as usize,
                num_kv_heads: hp.num_kv_heads as usize,
                head_dim: hp.head_dim as usize,
                precision: crate::kv::KvPrecision::F32,
            })
            .unwrap()
        };

        // Site 1 — resident preload.
        {
            let mut backend = fresh();
            backend.init(&hp).unwrap();
            check("preload", backend.preload_weights_gpu_resident(&provider));
        }
        // Site 2 — streaming create_layer_buffer via prefill.
        {
            let mut backend = fresh();
            backend.init(&hp).unwrap();
            let mut kv = kv();
            check(
                "prefill",
                backend
                    .prefill(&[1, 2, 3], &provider as &dyn WeightProvider, &mut kv)
                    .map(|_| ()),
            );
        }
        // Site 3 — expert-cache partial buffer via compute_layer.
        let (result, _backend) = compute_layer_over_warmed_cache(&provider, &path, num_experts);
        check("partial", result);
        std::fs::remove_dir_all(&dir).ok();
    }

    // Positive control: a sound layer over the same warmed cache creates the
    // partial buffer and not the full one, then stops at attention for want
    // of a KV view — so the branch the defects were refused on is the
    // partial one.
    let dir = std::env::temp_dir().join(format!("lumen_moe_sound_{}", std::process::id()));
    let (path, num_experts) = write_moe_layer_lbc(&dir, MoeDefect::Sound);
    let provider = SyncWeightProvider::open(&path).unwrap();
    let (result, backend) = compute_layer_over_warmed_cache(&provider, &path, num_experts);
    std::fs::remove_dir_all(&dir).ok();
    let err = result.expect_err("a sound layer must pass validation and stop at attention");
    assert!(
        err.to_string()
            .contains("KV cache view required for attention"),
        "sound layer must reach attention: {err}"
    );
    let scratch = backend.scratch.lock().unwrap();
    let s = scratch.as_ref().unwrap();
    assert!(
        s.moe_partial_buf_cache[0].is_some(),
        "partial buffer must be bound"
    );
    assert!(
        s.layer_buf_cache[0].is_none(),
        "full layer buffer must not be bound"
    );
}

/// A `--target generic` LBC keeps K-quant layer tensors intact (correct for
/// CUDA/CPU); the Metal cache prefers the `-metal` variant but falls back to
/// this generic artifact when none exists, and the Metal dense dispatch has
/// no K-quant kernels — without `validate_layer_quants` those bytes feed an
/// F32-reading pipeline and produce silent gibberish (documented at the
/// cache lookup and the validator itself). This is a BENIGN-reachable
/// correctness guard, so its wiring must stay pinned: this test drives a
/// K-quant dense tensor through the two dense-reachable Metal load sites
/// (resident preload and the streaming create_layer_buffer via prefill) and
/// asserts each refuses; the expert-cache partial site is covered by
/// `moe_layer_defects_rejected_at_every_metal_load_site`. Removing either
/// `validate_layer_quants` call turns the
/// clean refusal into silent-wrong output (or a later, differently-worded
/// failure) and fails this test.
#[test]
fn metal_kquant_dense_tensor_rejected_at_both_load_sites() {
    use crate::compute::ComputeBackend;
    use crate::weight::provider_sync::SyncWeightProvider;
    use lumen_format::header::LbcHeader;
    use lumen_format::hyperparams::{ModelHyperparams, RopeParams};
    use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
    use lumen_format::writer::{write_lbc, GlobalTensors};

    let (hidden, inter, heads, kv_heads, head_dim, vocab) =
        (64usize, 128usize, 2u32, 2u32, 32u32, 256usize);
    let q_dim = heads as usize * head_dim as usize;
    let kv_dim = kv_heads as usize * head_dim as usize;

    let q8_bytes = |n: usize| vec![0u8; n / 32 * 34];
    let norm_bytes = |n: usize| {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0f32.to_le_bytes());
        }
        v
    };

    let mut blob = Vec::new();
    let mut offset = 0u64;
    let add = |data: Vec<u8>, quant: QuantScheme, blob: &mut Vec<u8>, offset: &mut u64| {
        let s = TensorSlice {
            offset: *offset,
            length: data.len() as u64,
            quant,
        };
        blob.extend_from_slice(&data);
        *offset += s.length;
        s
    };
    let wq = add(
        q8_bytes(q_dim * hidden),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    let wk = add(
        q8_bytes(kv_dim * hidden),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    let wv = add(
        q8_bytes(kv_dim * hidden),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    let wo = add(
        q8_bytes(q_dim * hidden),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    // The defect under test: a K-quant (Q2_K) dense FFN tensor — exactly what
    // `--target generic` preserves and Metal cannot dispatch. The validator
    // reads only the quant scheme, so the appended bytes stand in for the
    // Q2_K blob (84 bytes / 256-element superblock).
    let w_gate = add(
        vec![0u8; inter * hidden / 256 * 84],
        QuantScheme::Q2_K,
        &mut blob,
        &mut offset,
    );
    let w_up = add(
        q8_bytes(inter * hidden),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    let w_down = add(
        q8_bytes(hidden * inter),
        QuantScheme::Q8_0,
        &mut blob,
        &mut offset,
    );
    let attn_norm = add(norm_bytes(hidden), QuantScheme::F32, &mut blob, &mut offset);
    let ffn_norm = add(norm_bytes(hidden), QuantScheme::F32, &mut blob, &mut offset);

    let subtensors = SubtensorOffsets {
        wq,
        wk,
        wv,
        wo,
        bq: None,
        bk: None,
        bv: None,
        w_gate,
        w_up,
        w_down,
        attn_norm,
        ffn_norm,
        router_weight: None,
        experts: None,
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate: None,
        attn_post_norm: None,
        ssm_a: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: Some(0),
    };
    let layer_indices = vec![LayerIndex {
        layer_offset_bytes: 0,
        layer_length_bytes: blob.len() as u64,
        subtensors,
    }];

    let hp = ModelHyperparams {
        num_layers: 1,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim,
        hidden_dim: hidden as u32,
        intermediate_dim: inter as u32,
        vocab_size: vocab as u32,
        max_seq_len: 512,
        rope_params: Some(RopeParams::default()),
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-5,
        rotary_dim: None,
        rope_neox: false,
        gdn: None,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::Q8_0,
        group_size: QuantGroupSize::Group(32),
        block_byte_size: 34,
        scale_offset_in_block: Some(0),
    };
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::Q8_0;
    header.output_proj.quant = QuantScheme::Q8_0;
    header.final_norm.quant = QuantScheme::F32;
    let globals = GlobalTensors {
        embedding: q8_bytes(vocab * hidden),
        final_norm: norm_bytes(hidden),
        output_proj: q8_bytes(vocab * hidden),
    };
    let mut bytes = Vec::new();
    write_lbc(
        &mut bytes,
        &header,
        &layer_indices,
        &globals,
        &[blob.as_slice()],
        None,
    )
    .unwrap();

    let dir = std::env::temp_dir().join(format!("lumen_kquant_metal_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("generic_kquant.lbc");
    std::fs::write(&path, &bytes).unwrap();

    let provider = SyncWeightProvider::open(&path).unwrap();
    let hp = provider.lbc().header.hyperparams;
    let want = "no dense DECODE dispatch kernels";

    // Site 1 — resident preload (the production path; gpu_resident=true).
    {
        let mut backend = MetalF32Backend::new().unwrap();
        backend.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        backend.init(&hp).unwrap();
        let err = backend
            .preload_weights_gpu_resident(&provider)
            .expect_err("resident preload must refuse a K-quant dense tensor on Metal");
        assert!(
            err.to_string().contains(want),
            "preload rejection must carry {want:?}: {err}"
        );
    }

    // Site 2 — streaming create_layer_buffer, reached via prefill.
    {
        let mut backend = MetalF32Backend::new().unwrap();
        backend.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        backend.init(&hp).unwrap();
        let mut kv = crate::kv::KvCache::new(crate::kv::KvCacheConfig {
            max_seq_len: hp.max_seq_len as usize,
            num_layers: hp.num_layers as usize,
            num_kv_heads: hp.num_kv_heads as usize,
            head_dim: hp.head_dim as usize,
            precision: crate::kv::KvPrecision::F32,
        })
        .unwrap();
        let err = backend
            .prefill(
                &[1, 2, 3],
                &provider as &dyn crate::weight::cache::WeightProvider,
                &mut kv,
            )
            .expect_err("streaming prefill must refuse a K-quant dense tensor on Metal");
        assert!(
            err.to_string().contains(want),
            "prefill rejection must carry {want:?}: {err}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The partial layer buffer covers `blob[0..non_expert_byte_end]`, so every
/// non-expert slice must be able to extend it and expert slices must not.
/// One assertion per field: a mutant that drops any single field from the
/// extent scan fails exactly that field's case.
#[test]
fn non_expert_byte_end_spans_every_optional_slice_and_excludes_experts() {
    use lumen_format::index::{ExpertSlice, SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::QuantScheme;
    let sl = |offset: u64, length: u64| TensorSlice {
        offset,
        length,
        quant: QuantScheme::Q8_0,
    };
    let base = || SubtensorOffsets {
        wq: sl(0, 10),
        wk: sl(10, 10),
        wv: sl(20, 10),
        wo: sl(30, 10),
        bq: None,
        bk: None,
        bv: None,
        w_gate: sl(40, 10),
        w_up: sl(50, 10),
        w_down: sl(60, 10),
        attn_norm: sl(70, 10),
        ffn_norm: sl(80, 10),
        router_weight: None,
        experts: Some(vec![ExpertSlice {
            gate: sl(20_000, 100),
            up: sl(20_100, 100),
            down: sl(20_200, 100),
        }]),
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate: None,
        attn_post_norm: None,
        ssm_a: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: Some(0),
    };
    // Mandatory only: the prefix ends at the last mandatory slice, not at the
    // experts far beyond it.
    assert_eq!(MetalF32Backend::non_expert_byte_end(&base()), 90);
    type Set = fn(&mut SubtensorOffsets, TensorSlice);
    let setters: [(&str, Set); 19] = [
        ("bq", |st, s| st.bq = Some(s)),
        ("bk", |st, s| st.bk = Some(s)),
        ("bv", |st, s| st.bv = Some(s)),
        ("router_weight", |st, s| st.router_weight = Some(s)),
        ("shared_expert_gate", |st, s| {
            st.shared_expert_gate = Some(s)
        }),
        ("shared_expert_up", |st, s| st.shared_expert_up = Some(s)),
        ("shared_expert_down", |st, s| {
            st.shared_expert_down = Some(s)
        }),
        ("attn_gate", |st, s| st.attn_gate = Some(s)),
        ("attn_post_norm", |st, s| st.attn_post_norm = Some(s)),
        ("ssm_a", |st, s| st.ssm_a = Some(s)),
        ("ssm_conv1d", |st, s| st.ssm_conv1d = Some(s)),
        ("ssm_dt", |st, s| st.ssm_dt = Some(s)),
        ("ssm_beta", |st, s| st.ssm_beta = Some(s)),
        ("ssm_alpha", |st, s| st.ssm_alpha = Some(s)),
        ("ssm_norm", |st, s| st.ssm_norm = Some(s)),
        ("ssm_out", |st, s| st.ssm_out = Some(s)),
        ("attn_q_norm", |st, s| st.attn_q_norm = Some(s)),
        ("attn_k_norm", |st, s| st.attn_k_norm = Some(s)),
        ("ffn_gate_inp_shexp", |st, s| {
            st.ffn_gate_inp_shexp = Some(s)
        }),
    ];
    for (name, set) in setters {
        let mut st = base();
        set(&mut st, sl(10_000, 100));
        assert_eq!(
            MetalF32Backend::non_expert_byte_end(&st),
            10_100,
            "{name} must extend the non-expert prefix"
        );
    }
    let mandatory: [(&str, Set); 9] = [
        ("wq", |st, s| st.wq = s),
        ("wk", |st, s| st.wk = s),
        ("wv", |st, s| st.wv = s),
        ("wo", |st, s| st.wo = s),
        ("w_gate", |st, s| st.w_gate = s),
        ("w_up", |st, s| st.w_up = s),
        ("w_down", |st, s| st.w_down = s),
        ("attn_norm", |st, s| st.attn_norm = s),
        ("ffn_norm", |st, s| st.ffn_norm = s),
    ];
    for (name, set) in mandatory {
        let mut st = base();
        set(&mut st, sl(10_000, 100));
        assert_eq!(
            MetalF32Backend::non_expert_byte_end(&st),
            10_100,
            "{name} must extend the non-expert prefix"
        );
    }
}
