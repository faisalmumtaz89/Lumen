//! Tests for F16 CUDA kernels: matvec correctness and KV cache round-trip.
//!
//! GPU tests require `--features cuda` and a CUDA-capable GPU.
//! CPU reference tests validate the f16 conversion logic and expected
//! matvec output without requiring GPU hardware.

// ============================================================================
// CPU reference tests (always run, no GPU required)
// ============================================================================

/// IEEE 754 f32 -> f16 conversion (CPU reference, matches kv_cache_f16.cu).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let frac = bits & 0x7f_ffff;

    if exp > 15 {
        if exp == 128 && frac != 0 {
            // NaN
            let mut h = ((sign << 15) | (0x1f << 10) | (frac >> 13)) as u16;
            if h & 0x3ff == 0 {
                h |= 1;
            }
            h
        } else {
            // Inf or overflow
            ((sign << 15) | (0x1f << 10)) as u16
        }
    } else if exp < -24 {
        // Flush to zero
        (sign << 15) as u16
    } else if exp < -14 {
        // Subnormal
        let frac_with_implicit = frac | 0x80_0000;
        let shift = (-1 - exp) as u32;
        let rounded = frac_with_implicit >> (shift + 13);
        let remainder = frac_with_implicit & ((1 << (shift + 13)) - 1);
        let halfway = 1u32 << (shift + 12);
        let mut rounded = rounded;
        if remainder > halfway || (remainder == halfway && (rounded & 1) != 0) {
            rounded += 1;
        }
        ((sign << 15) | rounded) as u16
    } else {
        // Normalized
        let f16_exp = (exp + 15) as u32;
        let mut rounded_frac = frac >> 13;
        let remainder = frac & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && (rounded_frac & 1) != 0) {
            rounded_frac += 1;
            if rounded_frac > 0x3ff {
                let f16_exp = f16_exp + 1;
                if f16_exp > 30 {
                    return ((sign << 15) | (0x1f << 10)) as u16;
                }
                return ((sign << 15) | (f16_exp << 10)) as u16;
            }
        }
        ((sign << 15) | (f16_exp << 10) | rounded_frac) as u16
    }
}

/// IEEE 754 f16 -> f32 conversion (CPU reference, matches matvec_f16.cu).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;

    let f = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: normalize
            let mut shift = 0u32;
            let mut tmp = frac;
            while (tmp & 0x400) == 0 {
                tmp <<= 1;
                shift += 1;
            }
            let frac = tmp & 0x3ff;
            let exp = 1u32.wrapping_sub(shift);
            (sign << 31) | (((exp.wrapping_add(127).wrapping_sub(15)) & 0xff) << 23) | (frac << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        // Normalized
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f)
}

/// CPU reference matvec: out = W_f16 * x
fn cpu_matvec_f16(weight_f16: &[u16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let w = f16_bits_to_f32(weight_f16[row * in_dim + j]);
            sum += w * x[j];
        }
        out[row] = sum;
    }
    out
}

#[test]
fn test_f16_roundtrip_basic_values() {
    let test_values: &[f32] = &[
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        0.333333, 2.0, 100.0, 0.001,
        65504.0,  // max f16 normal
        -65504.0, // min f16 normal
    ];

    for &v in test_values {
        let h = f32_to_f16_bits(v);
        let back = f16_bits_to_f32(h);
        let tol = if v == 0.0 { 0.0 } else { (v.abs() * 1e-3).max(6e-8) };
        assert!(
            (v - back).abs() <= tol,
            "f16 roundtrip failed: {v} -> 0x{h:04x} -> {back} (diff={})",
            (v - back).abs()
        );
    }
}

#[test]
fn test_f16_special_values() {
    // Positive infinity
    let h_inf = f32_to_f16_bits(f32::INFINITY);
    assert_eq!(h_inf, 0x7C00, "f16 +inf should be 0x7C00");
    assert!(f16_bits_to_f32(0x7C00).is_infinite());

    // Negative infinity
    let h_neg_inf = f32_to_f16_bits(f32::NEG_INFINITY);
    assert_eq!(h_neg_inf, 0xFC00, "f16 -inf should be 0xFC00");

    // NaN
    let h_nan = f32_to_f16_bits(f32::NAN);
    assert!(f16_bits_to_f32(h_nan).is_nan(), "f16 NaN should round-trip");

    // Zero
    let h_zero = f32_to_f16_bits(0.0);
    assert_eq!(h_zero, 0x0000);
    assert_eq!(f16_bits_to_f32(0x0000), 0.0);

    // Negative zero
    let h_neg_zero = f32_to_f16_bits(-0.0);
    assert_eq!(h_neg_zero, 0x8000);
}

#[test]
fn test_f16_subnormal() {
    // Smallest positive subnormal f16 = 2^-24 ~ 5.96e-8
    let smallest_sub = f16_bits_to_f32(0x0001);
    assert!(smallest_sub > 0.0, "smallest subnormal should be positive");
    assert!(smallest_sub < 1e-6, "smallest subnormal should be tiny: {smallest_sub}");

    // Largest subnormal f16 = (1023/1024) * 2^-14 ~ 6.098e-5
    let largest_sub = f16_bits_to_f32(0x03FF);
    assert!(largest_sub > 0.0);
    assert!(largest_sub < 1e-3);
}

#[test]
fn test_f16_overflow() {
    // Values > 65504 should overflow to infinity
    let h = f32_to_f16_bits(100000.0);
    assert!(f16_bits_to_f32(h).is_infinite(), "100000.0 should overflow to f16 inf");

    // Values very close to zero should flush
    let h = f32_to_f16_bits(1e-30);
    assert_eq!(f16_bits_to_f32(h), 0.0, "1e-30 should underflow to f16 zero");
}

#[test]
fn test_cpu_matvec_f16_identity() {
    // 3x3 identity matrix in f16, multiplied by [1.0, 2.0, 3.0]
    let dim = 3;
    let mut weight_f16 = vec![0u16; dim * dim];
    for i in 0..dim {
        weight_f16[i * dim + i] = f32_to_f16_bits(1.0);
    }
    let x = vec![1.0f32, 2.0, 3.0];

    let out = cpu_matvec_f16(&weight_f16, &x, dim, dim);
    for i in 0..dim {
        assert!(
            (out[i] - x[i]).abs() < 1e-3,
            "identity matvec[{i}]: expected {}, got {}",
            x[i], out[i]
        );
    }
}

#[test]
fn test_cpu_matvec_f16_known_output() {
    // 2x3 matrix: [[1, 2, 3], [4, 5, 6]] * [1, 0, 1] = [4, 10]
    let out_dim = 2;
    let in_dim = 3;
    let weight_f32 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight_f16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let x = vec![1.0f32, 0.0, 1.0];

    let out = cpu_matvec_f16(&weight_f16, &x, out_dim, in_dim);
    assert!((out[0] - 4.0).abs() < 1e-3, "row 0: expected 4.0, got {}", out[0]);
    assert!((out[1] - 10.0).abs() < 1e-3, "row 1: expected 10.0, got {}", out[1]);
}

#[test]
fn test_cpu_matvec_f16_residual() {
    // Verify matvec + residual: out = W*x + residual
    let dim = 4;
    let mut weight_f16 = vec![0u16; dim * dim];
    for i in 0..dim {
        weight_f16[i * dim + i] = f32_to_f16_bits(2.0); // 2x identity
    }
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let residual = vec![10.0f32, 20.0, 30.0, 40.0];

    let matvec_out = cpu_matvec_f16(&weight_f16, &x, dim, dim);
    let with_residual: Vec<f32> = matvec_out.iter().zip(residual.iter()).map(|(m, r)| m + r).collect();

    let expected = [12.0f32, 24.0, 36.0, 48.0]; // 2*x + residual
    for i in 0..dim {
        assert!(
            (with_residual[i] - expected[i]).abs() < 1e-2,
            "residual[{i}]: expected {}, got {}",
            expected[i], with_residual[i]
        );
    }
}

#[test]
fn test_kv_cache_f16_roundtrip() {
    // Simulate KV cache write+read: f32 -> f16 -> f32
    let num_kv_heads = 2;
    let max_seq_len = 8;
    let head_dim = 4;

    // Create test data for one token's K values
    let data: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();

    // Simulate cache as [num_kv_heads, max_seq_len, head_dim] f16 bits
    let cache_size = num_kv_heads * max_seq_len * head_dim;
    let mut cache = vec![0u16; cache_size];

    // Write at position 3
    let pos = 3;
    for head in 0..num_kv_heads {
        for d in 0..head_dim {
            let data_idx = head * head_dim + d;
            let cache_idx = head * max_seq_len * head_dim + pos * head_dim + d;
            cache[cache_idx] = f32_to_f16_bits(data[data_idx]);
        }
    }

    // Read back
    for head in 0..num_kv_heads {
        for d in 0..head_dim {
            let data_idx = head * head_dim + d;
            let cache_idx = head * max_seq_len * head_dim + pos * head_dim + d;
            let readback = f16_bits_to_f32(cache[cache_idx]);
            let expected = data[data_idx];
            assert!(
                (readback - expected).abs() < 1e-3,
                "KV cache roundtrip head={head} d={d}: expected {expected}, got {readback}"
            );
        }
    }
}

#[test]
fn test_kv_cache_f16_multiple_positions() {
    // Write multiple positions and verify all are independently correct
    let num_kv_heads = 1;
    let max_seq_len = 16;
    let head_dim = 8;

    let cache_size = num_kv_heads * max_seq_len * head_dim;
    let mut cache = vec![0u16; cache_size];

    // Write 5 positions with distinct data
    for pos in 0..5 {
        let data: Vec<f32> = (0..head_dim)
            .map(|d| (pos as f32) * 10.0 + (d as f32))
            .collect();
        for d in 0..head_dim {
            let cache_idx = pos * head_dim + d;
            cache[cache_idx] = f32_to_f16_bits(data[d]);
        }
    }

    // Read all positions back and verify
    for pos in 0..5 {
        for d in 0..head_dim {
            let expected = (pos as f32) * 10.0 + (d as f32);
            let cache_idx = pos * head_dim + d;
            let readback = f16_bits_to_f32(cache[cache_idx]);
            assert!(
                (readback - expected).abs() < 0.05,
                "pos={pos} d={d}: expected {expected}, got {readback}"
            );
        }
    }
}

#[test]
fn test_f16_precision_boundary() {
    // Test values near the precision boundary of f16 (~3.3 decimal digits)
    // f16 can represent integers exactly up to 2048
    for i in 0..2048 {
        let v = i as f32;
        let h = f32_to_f16_bits(v);
        let back = f16_bits_to_f32(h);
        assert_eq!(v, back, "integer {v} should round-trip exactly in f16");
    }
}

// ============================================================================
// GPU tests (require `--features cuda` and a CUDA-capable GPU)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_tests {
    use super::*;
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

    /// Helper: convert f32 values to contiguous f16 bits stored as u16.
    fn f32_slice_to_f16_u16(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_f16_bits(v)).collect()
    }

    #[test]
    fn test_gpu_matvec_f16_identity() {
        let (ctx, stream) = create_context();

        let src = lumen_runtime::cuda::shaders::MATVEC_F16_KERNEL_SOURCE;
        let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_f16.cu");
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("matvec_f16").unwrap();

        let dim = 64usize;
        let mut weight_f32 = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weight_f32[i * dim + i] = 1.0;
        }
        let weight_f16 = f32_slice_to_f16_u16(&weight_f32);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();

        let gpu_w = stream.clone_htod(&weight_f16).unwrap();
        let gpu_x = stream.clone_htod(&x).unwrap();
        let mut gpu_out: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (dim as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dim_u32 = dim as u32;
        let in_dim_u32 = dim as u32;

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&gpu_w)
                .arg(&gpu_x)
                .arg(&mut gpu_out)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_f16 launch failed");

        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&gpu_out).unwrap();
        for i in 0..dim {
            assert!(
                (out[i] - x[i]).abs() < 1e-3,
                "GPU identity matvec[{i}]: expected {}, got {}",
                x[i], out[i]
            );
        }
    }

    #[test]
    fn test_gpu_matvec_f16_vs_cpu_reference() {
        let (ctx, stream) = create_context();

        let src = lumen_runtime::cuda::shaders::MATVEC_F16_KERNEL_SOURCE;
        let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_f16.cu");
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("matvec_f16").unwrap();

        let out_dim = 128usize;
        let in_dim = 256usize;

        let mut seed = 42u64;
        let next_f32 = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };

        let weight_f32: Vec<f32> = (0..out_dim * in_dim).map(|_| next_f32(&mut seed)).collect();
        let x: Vec<f32> = (0..in_dim).map(|_| next_f32(&mut seed)).collect();

        let weight_f16_u16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let cpu_out = cpu_matvec_f16(&weight_f16_u16, &x, out_dim, in_dim);

        let gpu_w = stream.clone_htod(&weight_f16_u16).unwrap();
        let gpu_x = stream.clone_htod(&x).unwrap();
        let mut gpu_out: CudaSlice<f32> = stream.alloc_zeros(out_dim).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (out_dim as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&gpu_w)
                .arg(&gpu_x)
                .arg(&mut gpu_out)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_f16 launch failed");

        stream.synchronize().unwrap();
        let gpu_result = stream.clone_dtoh(&gpu_out).unwrap();

        for i in 0..out_dim {
            let diff = (gpu_result[i] - cpu_out[i]).abs();
            let tol = cpu_out[i].abs() * 1e-3 + 1e-4;
            assert!(
                diff < tol,
                "GPU vs CPU mismatch at [{i}]: gpu={}, cpu={}, diff={diff}",
                gpu_result[i], cpu_out[i]
            );
        }
    }

    #[test]
    fn test_gpu_matvec_f16_residual() {
        let (ctx, stream) = create_context();

        let src = lumen_runtime::cuda::shaders::MATVEC_F16_KERNEL_SOURCE;
        let ptx = compile_ptx(src).expect("NVRTC compile failed for matvec_f16.cu");
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("matvec_f16_residual").unwrap();

        let dim = 64usize;
        let mut weight_f32 = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weight_f32[i * dim + i] = 2.0;
        }
        let weight_f16 = f32_slice_to_f16_u16(&weight_f32);
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        let residual: Vec<f32> = (0..dim).map(|i| (i as f32) * 10.0).collect();

        let gpu_w = stream.clone_htod(&weight_f16).unwrap();
        let gpu_x = stream.clone_htod(&x).unwrap();
        let res_gpu = stream.clone_htod(&residual).unwrap();
        let mut gpu_out: CudaSlice<f32> = stream.alloc_zeros(dim).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (dim as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dim_u32 = dim as u32;
        let in_dim_u32 = dim as u32;

        // matvec_f16_residual(weight, x, out, residual, out_dim, in_dim)
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&gpu_w)
                .arg(&gpu_x)
                .arg(&mut gpu_out)
                .arg(&res_gpu)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(cfg)
        }
        .expect("matvec_f16_residual launch failed");

        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&gpu_out).unwrap();
        for i in 0..dim {
            let expected = 2.0 * x[i] + residual[i];
            assert!(
                (out[i] - expected).abs() < 0.1,
                "residual matvec[{i}]: expected {expected}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_gpu_kv_cache_f16_write_read() {
        let (ctx, stream) = create_context();

        let src = lumen_runtime::cuda::shaders::KV_CACHE_F16_KERNEL_SOURCE;
        let ptx = compile_ptx(src).expect("NVRTC compile failed for kv_cache_f16.cu");
        let module = ctx.load_module(ptx).unwrap();
        let write_func = module.load_function("kv_cache_write_f16").unwrap();
        let read_func = module.load_function("kv_cache_read_f16").unwrap();

        let num_kv_heads = 4u32;
        let max_seq_len = 32u32;
        let head_dim = 64u32;
        let total_f16_elems = (num_kv_heads * max_seq_len * head_dim) as usize;

        // Allocate cache as u16 (f16 bits)
        let mut cache: CudaSlice<u16> = stream.alloc_zeros(total_f16_elems).unwrap();

        // Write data at position 5
        let pos = 5u32;
        let total_data_elems = (num_kv_heads * head_dim) as usize;
        let data: Vec<f32> = (0..total_data_elems)
            .map(|i| (i as f32) * 0.01 - 1.0)
            .collect();
        let gpu_data = stream.clone_htod(&data).unwrap();

        let write_cfg = LaunchConfig {
            grid_dim: ((total_data_elems as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&write_func)
                .arg(&mut cache)
                .arg(&gpu_data)
                .arg(&pos)
                .arg(&num_kv_heads)
                .arg(&max_seq_len)
                .arg(&head_dim)
                .launch(write_cfg)
        }
        .expect("kv_cache_write_f16 launch failed");

        // Read back for each head
        for head in 0..num_kv_heads {
            let count = 1u32;
            let read_elems = (count * head_dim) as usize;
            let mut gpu_readback: CudaSlice<f32> = stream.alloc_zeros(read_elems).unwrap();

            let read_cfg = LaunchConfig {
                grid_dim: ((read_elems as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                stream
                    .launch_builder(&read_func)
                    .arg(&cache)
                    .arg(&mut gpu_readback)
                    .arg(&head)
                    .arg(&pos)
                    .arg(&count)
                    .arg(&max_seq_len)
                    .arg(&head_dim)
                    .launch(read_cfg)
            }
            .expect("kv_cache_read_f16 launch failed");

            stream.synchronize().unwrap();
            let readback = stream.clone_dtoh(&gpu_readback).unwrap();
            for d in 0..head_dim as usize {
                let data_idx = (head as usize) * (head_dim as usize) + d;
                let expected = data[data_idx];
                let tol = expected.abs() * 1e-3 + 1e-4;
                assert!(
                    (readback[d] - expected).abs() < tol,
                    "KV read head={head} d={d}: expected {expected}, got {}",
                    readback[d]
                );
            }
        }
    }
}
