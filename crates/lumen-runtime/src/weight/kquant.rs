//! Host dequantisation of the GGML K-quant block formats, shared by the weight
//! providers (the F32 copy of a K-quant embedding or output head) and the CUDA backend
//! (the layer host-dequant catch-all, the Q5_K `ssm_out` plane build of a non-K-quant
//! artifact, the `LUMEN_CUDA_Q6K_HEAD=0` head fallback, and the reference the K-quant
//! kernels are held to — bit-identically for the F16 dequant tiles and the embedding
//! gathers, within a tolerance for the matvecs).

use crate::error::RuntimeError;
use lumen_format::quantization::QuantScheme;

/// Host IEEE f16 bits -> f32 (exact, subnormals and the special values
/// included); the conversion the K-quant references take by name.
pub fn host_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let v = (frac as f32) / 16_777_216.0; // 2^-24, the f16 subnormal scale
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac != 0 {
            f32::NAN
        } else if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    // rebias 15 -> 127 as one addition: `exp` is 1..=30 here, and `exp - 15` would
    // underflow the unsigned value for every scale below 1.0
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
}

/// Decode K-quant scales from 12 packed bytes into 8 scale + 8 min arrays.
///
/// Used by Q4_K and Q5_K. The 12 bytes encode 8 6-bit scales and 8 6-bit mins
/// in the standard packed layout: low 6 bits from bytes 0..7, high 2 bits from bytes 8..11.
fn decode_k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 63;
        m[j] = scales[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

/// Dequantize a K-quant weight buffer to F32.
///
/// Supports every K-quant scheme an LBC stores: the GGML blocks Q6_K, Q4_K, Q5_K,
/// Q2_K and Q3_K. Mixed-quant GGUFs (e.g. bartowski/mradermacher Q4_0 with imatrix)
/// commonly use Q5_K or Q6_K for sensitive per-layer tensors alongside Q4_0.
///
/// All implementations match the reference layout exactly (same as
/// lumen-convert::dequant).
pub fn dequant_kquant_to_f32(
    raw: &[u8],
    scheme: QuantScheme,
    n_elements: usize,
) -> Result<Vec<f32>, RuntimeError> {
    // the plane must hold every element: a short plane is an error here, never a zero tail
    let block_bytes = match scheme {
        QuantScheme::Q6_K => 210,
        QuantScheme::Q4_K => 144,
        QuantScheme::Q5_K => 176,
        QuantScheme::Q2_K => 84,
        QuantScheme::Q3_K => 110,
        other => {
            return Err(RuntimeError::Compute(format!(
                "{other:?} planes have no host dequant. Re-convert the model with \
                 --requant q8_0 or --requant q4_0.",
            )));
        }
    };
    let held = raw.len() / block_bytes * 256;
    if held < n_elements {
        return Err(RuntimeError::Compute(format!(
            "{scheme:?} plane: {} bytes hold {held} elements, {n_elements} required",
            raw.len()
        )));
    }
    let mut out = vec![0.0f32; n_elements];

    match scheme {
        QuantScheme::Q6_K => {
            // Q6_K: 256 elements per block, 210 bytes per block.
            // Layout: [128B ql, 64B qh, 16B scales, 2B f16_d]
            let n_blocks = raw.len() / block_bytes;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_bytes..];
                let ql = &bp[0..128];
                let qh = &bp[128..192];
                let scales = &bp[192..208];
                let d_bits = u16::from_le_bytes([bp[208], bp[209]]);
                let d = host_f16_to_f32(d_bits);

                let mut idx = 0usize;
                for half in 0..2usize {
                    let ql_ptr = &ql[64 * half..];
                    let qh_ptr = &qh[32 * half..];
                    let sc_ptr = &scales[8 * half..];

                    // Group 0: low nibbles of ql[0..32], qh bits [0..1]
                    for j in 0..32 {
                        if written + idx >= n_elements {
                            break;
                        }
                        let q_lo = ql_ptr[j] & 0x0F;
                        let q_hi = (qh_ptr[j] & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 1: low nibbles of ql[32..64], qh bits [2..3]
                    // (order MUST match ggml's dequantize_row_q6_K).
                    for j in 0..32 {
                        if written + idx >= n_elements {
                            break;
                        }
                        let q_lo = ql_ptr[32 + j] & 0x0F;
                        let q_hi = ((qh_ptr[j] >> 2) & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[2 + j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 2: high nibbles of ql[0..32], qh bits [4..5]
                    for j in 0..32 {
                        if written + idx >= n_elements {
                            break;
                        }
                        let q_lo = (ql_ptr[j] >> 4) & 0x0F;
                        let q_hi = ((qh_ptr[j] >> 4) & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[4 + j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 3: high nibbles of ql[32..64], qh bits [6..7]
                    for j in 0..32 {
                        if written + idx >= n_elements {
                            break;
                        }
                        let q_lo = (ql_ptr[32 + j] >> 4) & 0x0F;
                        let q_hi = ((qh_ptr[j] >> 6) & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[6 + j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                }
                written += idx;
            }
        }
        QuantScheme::Q4_K => {
            // Q4_K: 256 elements per block, 144 bytes per block.
            // Layout: [2B f16 d, 2B f16 dmin, 12B scales, 128B qs]
            let n_blocks = raw.len() / block_bytes;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_bytes..];
                let d = host_f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
                let dmin = host_f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
                let (sc, m_arr) = decode_k_scales(&bp[4..16]);
                let qs = &bp[16..144];

                // 4 groups of 64 values (2 sub-blocks each)
                for group in 0..4 {
                    let is = group * 2;
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * m_arr[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * m_arr[is + 1] as f32;
                    let qs_offset = group * 32;

                    // First 32 values: low nibbles
                    for l in 0..32 {
                        if written >= n_elements {
                            break;
                        }
                        out[written] = d1 * (qs[qs_offset + l] & 0x0F) as f32 - m1;
                        written += 1;
                    }
                    // Second 32 values: high nibbles
                    for l in 0..32 {
                        if written >= n_elements {
                            break;
                        }
                        out[written] = d2 * ((qs[qs_offset + l] >> 4) & 0x0F) as f32 - m2;
                        written += 1;
                    }
                }
            }
        }
        QuantScheme::Q5_K => {
            // Q5_K: 256 elements per block, 176 bytes per block.
            // Layout: [2B f16 d, 2B f16 dmin, 12B scales, 32B qh, 128B qs]
            let n_blocks = raw.len() / block_bytes;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_bytes..];
                let d = host_f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
                let dmin = host_f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
                let (sc, m_arr) = decode_k_scales(&bp[4..16]);
                let qh = &bp[16..48];
                let qs = &bp[48..176];

                // 4 groups of 64 values
                for group in 0..4 {
                    let is = group * 2;
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * m_arr[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * m_arr[is + 1] as f32;
                    let qs_offset = group * 32;
                    let u1 = group * 2;
                    let u2 = u1 + 1;

                    // First 32 values: low nibbles + high bit
                    for l in 0..32 {
                        if written >= n_elements {
                            break;
                        }
                        let h_bit = (qh[l] >> u1) & 1;
                        out[written] = d1 * ((qs[qs_offset + l] & 0x0F) | (h_bit << 4)) as f32 - m1;
                        written += 1;
                    }
                    // Second 32 values: high nibbles + high bit
                    for l in 0..32 {
                        if written >= n_elements {
                            break;
                        }
                        let h_bit = (qh[l] >> u2) & 1;
                        out[written] =
                            d2 * (((qs[qs_offset + l] >> 4) & 0x0F) | (h_bit << 4)) as f32 - m2;
                        written += 1;
                    }
                }
            }
        }
        QuantScheme::Q2_K => {
            // Q2_K: 256 elements per block, 84 bytes per block.
            // Layout: [16B scales, 64B qs, 2B f16 d, 2B f16 dmin]
            //
            // qs traversal MUST follow GGML's `dequantize_row_q2_K`: two
            // 128-value groups, four shift passes (0,2,4,6) over the same 32 qs
            // bytes per group, two 16-value runs (`q[l]`, `q[l+16]`) per pass.
            // A naive linear byte scan corrupts ~74% of any real Q2_K block
            // (agrees only on degenerate uniform blocks). See the matching
            // converter fix in lumen-convert/src/dequant.rs::dequantize_q2_k.
            let n_blocks = raw.len() / block_bytes;
            let mut written = 0usize;
            'blocks: for b in 0..n_blocks {
                let bp = &raw[b * block_bytes..];
                let scales = &bp[0..16];
                let qs = &bp[16..80];
                let d = host_f16_to_f32(u16::from_le_bytes([bp[80], bp[81]]));
                let dmin = host_f16_to_f32(u16::from_le_bytes([bp[82], bp[83]]));

                let mut q_off = 0usize;
                let mut is = 0usize;
                for _group in 0..2 {
                    let mut shift = 0u8;
                    for _j in 0..4 {
                        let sc0 = scales[is];
                        is += 1;
                        let dl0 = d * (sc0 & 0x0F) as f32;
                        let ml0 = dmin * ((sc0 >> 4) & 0x0F) as f32;
                        for l in 0..16usize {
                            if written >= n_elements {
                                break 'blocks;
                            }
                            out[written] = dl0 * (((qs[q_off + l] >> shift) & 3) as f32) - ml0;
                            written += 1;
                        }
                        let sc1 = scales[is];
                        is += 1;
                        let dl1 = d * (sc1 & 0x0F) as f32;
                        let ml1 = dmin * ((sc1 >> 4) & 0x0F) as f32;
                        for l in 0..16usize {
                            if written >= n_elements {
                                break 'blocks;
                            }
                            out[written] = dl1 * (((qs[q_off + l + 16] >> shift) & 3) as f32) - ml1;
                            written += 1;
                        }
                        shift += 2;
                    }
                    q_off += 32;
                }
            }
        }
        QuantScheme::Q3_K => {
            // Q3_K: 256 elements per block, 110 bytes per block.
            // Layout: [32B hmask, 64B qs (2-bit low), 12B scales (6-bit packed), 2B f16 d]
            let n_blocks = raw.len() / block_bytes;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_bytes..];
                let hmask = &bp[0..32];
                let qs = &bp[32..96];
                let scale_bytes = &bp[96..108];
                let d = host_f16_to_f32(u16::from_le_bytes([bp[108], bp[109]]));

                // Decode 16 6-bit scales from 12 bytes (standard packed layout)
                let mut sc_arr = [0u8; 16];
                for j in 0..4 {
                    sc_arr[j] = scale_bytes[j] & 0x0F;
                    sc_arr[j + 4] = (scale_bytes[j] >> 4) & 0x0F;
                }
                for j in 0..4 {
                    sc_arr[j + 8] = scale_bytes[4 + j] & 0x0F;
                    sc_arr[j + 12] = (scale_bytes[4 + j] >> 4) & 0x0F;
                }
                for (j, sc) in sc_arr.iter_mut().enumerate() {
                    let byte_idx = 8 + j / 4;
                    let bit_shift = 2 * (j % 4);
                    *sc |= ((scale_bytes[byte_idx] >> bit_shift) & 3) << 4;
                }

                // GGML `dequantize_row_q3_K` traversal: same grouped/shifted
                // scheme as Q2_K (two 128-value groups; four shift passes over
                // the same 32 qs bytes; `q[l]`/`q[l+16]` runs; hmask selector
                // `m` advances per pass). The naive linear scan corrupts ~82%
                // of values. See the converter fix in
                // lumen-convert/src/dequant.rs::dequantize_q3_k.
                let mut q_off = 0usize;
                let mut is = 0usize;
                let mut hbit = 1u8;
                'q3blocks: for _group in 0..2 {
                    let mut shift = 0u8;
                    for _j in 0..4 {
                        let scale0 = d * (sc_arr[is] as i8 as f32 - 32.0);
                        is += 1;
                        for l in 0..16usize {
                            if written >= n_elements {
                                break 'q3blocks;
                            }
                            let q_lo = (qs[q_off + l] >> shift) & 3;
                            let h = u8::from((hmask[l] & hbit) != 0);
                            out[written] = scale0 * ((q_lo | (h << 2)) as i32 - 4) as f32;
                            written += 1;
                        }
                        let scale1 = d * (sc_arr[is] as i8 as f32 - 32.0);
                        is += 1;
                        for l in 0..16usize {
                            if written >= n_elements {
                                break 'q3blocks;
                            }
                            let q_lo = (qs[q_off + l + 16] >> shift) & 3;
                            let h = u8::from((hmask[l + 16] & hbit) != 0);
                            out[written] = scale1 * ((q_lo | (h << 2)) as i32 - 4) as f32;
                            written += 1;
                        }
                        shift += 2;
                        hbit <<= 1;
                    }
                    q_off += 32;
                }
            }
        }
        other => unreachable!("{other:?} was refused above"),
    }

    Ok(out)
}

#[cfg(test)]
mod f16_tests {
    use super::host_f16_to_f32;

    #[test]
    fn host_f16_to_f32_matches_ieee() {
        // normals below and above 1.0 (the sub-unit ones are every K-quant scale in
        // practice), the extremes, subnormals and the specials
        for (bits, want) in [
            (0x3800u16, 0.5f32),
            (0x3C00, 1.0),
            (0x4000, 2.0),
            (0xC000, -2.0),
            (0x0400, 6.103_515_6e-5),
            (0x7BFF, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x8001, -5.960_464_5e-8),
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x7C00, f32::INFINITY),
            (0xFC00, f32::NEG_INFINITY),
        ] {
            let got = host_f16_to_f32(bits);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{bits:#06x}: got {got} want {want}"
            );
        }
        assert!(host_f16_to_f32(0x7C01).is_nan());
        // every normal encoding round-trips through f32 exactly
        for bits in (0x0400u16..0x7C00).chain(0x8400..0xFC00) {
            let v = host_f16_to_f32(bits);
            assert!(v.is_finite(), "{bits:#06x}");
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = ((bits >> 10) & 0x1f) as i32 - 15;
            let frac = 1.0 + (bits & 0x3ff) as f32 / 1024.0;
            assert_eq!(v, sign * frac * 2f32.powi(exp), "{bits:#06x}");
        }
    }
}
