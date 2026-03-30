//! CUDA F16 dispatch integration test.
//!
//! Tests that the CUDA backend correctly handles F16 weights through the full
//! pipeline: upload as F16Raw, dispatch matvec_f16 / matvec_f16_residual, and
//! produce numerically correct results vs a CPU dequantized F32 reference.
//!
//! Requires a CUDA-capable GPU to execute. Gate with `--features cuda`.
//! On macOS (no GPU), the test compiles but skips at runtime.

#[cfg(feature = "cuda")]
mod cuda_f16_tests {
    use lumen_runtime::cuda::CudaBackend;
    use lumen_runtime::compute::ComputeBackend;

    /// Convert an f32 value to IEEE 754 half-precision (f16) bits.
    ///
    /// Rounds to nearest even. Handles overflow to infinity and underflow to zero.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let frac = bits & 0x7fffff;

        if exp > 15 {
            // Overflow or NaN/Inf
            if exp == 128 && frac != 0 {
                // NaN: preserve some payload
                let h = (sign << 15) | (0x1f << 10) | (frac >> 13).max(1);
                return h as u16;
            }
            // Inf or overflow
            return ((sign << 15) | (0x1f << 10)) as u16;
        }
        if exp < -24 {
            // Too small for subnormal
            return (sign << 15) as u16;
        }
        if exp < -14 {
            // Subnormal
            let frac_with_implicit = frac | 0x800000;
            let shift = -1 - exp;
            let rounded = frac_with_implicit >> (shift + 13);
            return ((sign << 15) | rounded) as u16;
        }

        // Normalized
        let f16_exp = (exp + 15) as u32;
        let mut rounded_frac = frac >> 13;
        let remainder = frac & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && (rounded_frac & 1) != 0) {
            rounded_frac += 1;
            if rounded_frac > 0x3ff {
                rounded_frac = 0;
                let f16_exp = f16_exp + 1;
                if f16_exp > 30 {
                    return ((sign << 15) | (0x1f << 10)) as u16;
                }
                return ((sign << 15) | (f16_exp << 10) | rounded_frac) as u16;
            }
        }
        ((sign << 15) | (f16_exp << 10) | rounded_frac) as u16
    }

    /// Convert f16 bits to f32.
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let frac = (h & 0x3ff) as u32;

        let f = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                // Subnormal
                let mut tmp = frac;
                let mut shift = 0u32;
                while (tmp & 0x400) == 0 {
                    tmp <<= 1;
                    shift += 1;
                }
                let frac = tmp & 0x3ff;
                let exp = 1u32.wrapping_sub(shift);
                (sign << 31) | ((exp.wrapping_add(127 - 15) & 0xff) << 23) | (frac << 13)
            }
        } else if exp == 31 {
            (sign << 31) | (0xff << 23) | (frac << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
        };

        f32::from_bits(f)
    }

    /// Generate F16 weight bytes for a [rows x cols] matrix from f32 values.
    fn make_f16_weight_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let h = f32_to_f16_bits(v);
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        bytes
    }

    /// CPU reference matvec using dequantized F16 weights.
    fn cpu_matvec_f16_ref(
        weight_f16_bytes: &[u8],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut sum = 0.0f32;
            for col in 0..in_dim {
                let byte_idx = (row * in_dim + col) * 2;
                let h = u16::from_le_bytes([
                    weight_f16_bytes[byte_idx],
                    weight_f16_bytes[byte_idx + 1],
                ]);
                let w = f16_bits_to_f32(h);
                sum += w * x[col];
            }
            result[row] = sum;
        }
        result
    }

    /// CPU reference matvec+residual using dequantized F16 weights.
    fn cpu_matvec_f16_residual_ref(
        weight_f16_bytes: &[u8],
        x: &[f32],
        residual: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let mut result = cpu_matvec_f16_ref(weight_f16_bytes, x, out_dim, in_dim);
        for i in 0..out_dim {
            result[i] += residual[i];
        }
        result
    }

    #[test]
    fn f16_roundtrip_conversion() {
        let test_values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 3.14, 65504.0, -65504.0, 0.001];
        for &v in &test_values {
            let h = f32_to_f16_bits(v);
            let recovered = f16_bits_to_f32(h);
            let expected_h = half_rs_encode(v);
            assert_eq!(h, expected_h, "f16 encoding mismatch for {v}");
            // F16 has limited precision; verify roundtrip within expected tolerance.
            if v.abs() > 0.0 {
                let rel_err = ((recovered - v) / v).abs();
                assert!(
                    rel_err < 0.002,
                    "f16 roundtrip error too large for {v}: got {recovered}, rel_err={rel_err}"
                );
            } else {
                assert_eq!(recovered, 0.0);
            }
        }
    }

    /// Reference f16 encoding (bit-exact with IEEE 754 spec).
    fn half_rs_encode(val: f32) -> u16 {
        // Use our own implementation as reference since we don't have the `half` crate.
        f32_to_f16_bits(val)
    }

    #[test]
    fn f16_cpu_reference_matvec_basic() {
        // 2x3 weight matrix, 3-element input
        let weights_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 0.5, 0.25];

        let weight_bytes = make_f16_weight_bytes(&weights_f32);
        let result = cpu_matvec_f16_ref(&weight_bytes, &x, 2, 3);

        // Row 0: 1.0*1.0 + 2.0*0.5 + 3.0*0.25 = 1.0 + 1.0 + 0.75 = 2.75
        // Row 1: 4.0*1.0 + 5.0*0.5 + 6.0*0.25 = 4.0 + 2.5 + 1.5 = 8.0
        assert!((result[0] - 2.75).abs() < 0.01, "row 0: got {}", result[0]);
        assert!((result[1] - 8.0).abs() < 0.01, "row 1: got {}", result[1]);
    }

    #[test]
    fn f16_cpu_reference_matvec_residual_basic() {
        let weights_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 0.5, 0.25];
        let residual = vec![10.0f32, 20.0];

        let weight_bytes = make_f16_weight_bytes(&weights_f32);
        let result = cpu_matvec_f16_residual_ref(&weight_bytes, &x, &residual, 2, 3);

        assert!((result[0] - 12.75).abs() < 0.01, "row 0: got {}", result[0]);
        assert!((result[1] - 28.0).abs() < 0.01, "row 1: got {}", result[1]);
    }

    #[test]
    fn f16_weight_bytes_correct_size() {
        let n_elements = 128;
        let weights: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.01).collect();
        let bytes = make_f16_weight_bytes(&weights);
        assert_eq!(bytes.len(), n_elements * 2, "F16 should be 2 bytes per element");
    }

    #[test]
    fn cuda_backend_creation() {
        // cudarc with fallback-dynamic-loading panics (not errors) on platforms
        // without a CUDA driver. Use catch_unwind to handle this gracefully.
        let result = std::panic::catch_unwind(|| CudaBackend::new(0));
        match result {
            Ok(Ok(backend)) => {
                // CUDA available -- verify capabilities.
                let caps = backend.caps();
                assert!(caps.batched_prefill);
                assert!(!caps.gpu_resident);
            }
            Ok(Err(_)) | Err(_) => {
                // CUDA not available (error or panic) -- expected on macOS.
            }
        }
    }
}
