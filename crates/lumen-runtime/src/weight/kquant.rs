//! Host dequantisation of every block format an LBC stores. The CPU backends
//! decode weight rows with [`dequant_row_into`]; the K-quant wrapper
//! [`dequant_kquant_to_f32`] is shared by the weight providers (the F32 copy of a
//! K-quant embedding or output head) and the CUDA backend
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

/// Decode the 16 six-bit Q3_K scales from the block's 12 packed scale bytes.
///
/// Scale `i` takes its low four bits from one nibble of bytes 0..8 and its high
/// two bits from a 2-bit field of bytes 8..12:
///
/// - scales 0..4   low nibble of bytes 0..4
/// - scales 4..8   low nibble of bytes 4..8
/// - scales 8..12  high nibble of bytes 0..4
/// - scales 12..16 high nibble of bytes 4..8
/// - scale `i`     high two bits from byte `8 + i % 4`, at bit position `2 * (i / 4)`
///
/// Each decoded value is a 6-bit magnitude; the dequantiser reads it as `sc - 32`.
fn decode_q3_k_scales(scale_bytes: &[u8]) -> [u8; 16] {
    let mut sc = [0u8; 16];
    for j in 0..4 {
        sc[j] = scale_bytes[j] & 0x0F;
        sc[j + 4] = scale_bytes[j + 4] & 0x0F;
        sc[j + 8] = scale_bytes[j] >> 4;
        sc[j + 12] = scale_bytes[j + 4] >> 4;
    }
    for (i, v) in sc.iter_mut().enumerate() {
        *v |= ((scale_bytes[8 + i % 4] >> (2 * (i / 4))) & 3) << 4;
    }
    sc
}

/// Elements and bytes of one storage block of `scheme`, for every scheme the
/// host decodes; `None` for the rest (NVFP4, FP8, the CUDA-only CT planes).
pub fn block_geometry(scheme: QuantScheme) -> Option<(usize, usize)> {
    Some(match scheme {
        QuantScheme::F32 => (1, 4),
        QuantScheme::F16 | QuantScheme::Bf16 => (1, 2),
        QuantScheme::Q8_0 => (32, 34),
        QuantScheme::Q4_0 => (32, 18),
        QuantScheme::Q4_1 => (32, 20),
        QuantScheme::Q5_0 => (32, 22),
        QuantScheme::Q2_K => (256, 84),
        QuantScheme::Q3_K => (256, 110),
        QuantScheme::Q4_K => (256, 144),
        QuantScheme::Q5_K => (256, 176),
        QuantScheme::Q6_K => (256, 210),
        _ => return None,
    })
}

/// Bytes of one `cols`-element row of `scheme`; `None` when the host cannot
/// decode the scheme or a row does not end on a block boundary.
pub fn row_bytes(scheme: QuantScheme, cols: usize) -> Option<usize> {
    let (elems, bytes) = block_geometry(scheme)?;
    (cols % elems == 0).then(|| cols / elems * bytes)
}

/// Decode row `row` of a row-major matrix of `cols`-element rows stored as
/// `scheme` in `bytes` into `out` (`cols` floats). A scheme the host cannot
/// decode, a row that splits a block, or storage too short for the row is an
/// error, never a partial write.
pub fn dequant_row_into(
    out: &mut [f32],
    bytes: &[u8],
    row: usize,
    cols: usize,
    scheme: QuantScheme,
) -> Result<(), RuntimeError> {
    let (elems, block) = block_geometry(scheme)
        .ok_or_else(|| RuntimeError::Compute(format!("{scheme:?} has no host decoder")))?;
    let rb = row_bytes(scheme, cols).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "{scheme:?} rows of {cols} elements split a {elems}-element block"
        ))
    })?;
    if out.len() != cols {
        return Err(RuntimeError::Compute(format!(
            "{scheme:?} row decode: output holds {} elements, row has {cols}",
            out.len()
        )));
    }
    let start = row * rb;
    let src = bytes.get(start..start + rb).ok_or_else(|| {
        RuntimeError::Compute(format!(
            "{scheme:?} row {row} needs bytes {start}..{}, storage holds {}",
            start + rb,
            bytes.len()
        ))
    })?;
    for (dst, b) in out.chunks_exact_mut(elems).zip(src.chunks_exact(block)) {
        decode_block(scheme, b, dst);
    }
    Ok(())
}

/// Decode one storage block `b` of `scheme` into `out` (the block's elements).
/// The caller has checked the scheme and both lengths.
fn decode_block(scheme: QuantScheme, b: &[u8], out: &mut [f32]) {
    match scheme {
        QuantScheme::F32 => out[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        QuantScheme::F16 => out[0] = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        QuantScheme::Bf16 => {
            out[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
        }
        QuantScheme::Q8_0 => {
            let d = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for (o, &q) in out.iter_mut().zip(&b[2..34]) {
                *o = d * (q as i8) as f32;
            }
        }
        // The 4- and 5-bit blocks put the low nibbles in elements 0..16 and the
        // high nibbles in 16..32 (GGML's de-interleaved order).
        QuantScheme::Q4_0 => {
            let d = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for i in 0..16 {
                let byte = b[2 + i];
                out[i] = d * ((byte & 0x0F) as i32 - 8) as f32;
                out[16 + i] = d * ((byte >> 4) as i32 - 8) as f32;
            }
        }
        QuantScheme::Q4_1 => {
            let d = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let m = host_f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            for i in 0..16 {
                let byte = b[4 + i];
                out[i] = d * (byte & 0x0F) as f32 + m;
                out[16 + i] = d * (byte >> 4) as f32 + m;
            }
        }
        // qh bit j is the high bit of element j.
        QuantScheme::Q5_0 => {
            let d = host_f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            for i in 0..16 {
                let byte = b[6 + i];
                let lo = (byte & 0x0F) as u32 | (((qh >> i) & 1) << 4);
                let hi = (byte >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
                out[i] = d * (lo as f32 - 16.0);
                out[16 + i] = d * (hi as f32 - 16.0);
            }
        }
        QuantScheme::Q6_K => decode_q6_k(b, out),
        QuantScheme::Q4_K => decode_q4_k(b, out),
        QuantScheme::Q5_K => decode_q5_k(b, out),
        QuantScheme::Q2_K => decode_q2_k(b, out),
        QuantScheme::Q3_K => decode_q3_k(b, out),
        other => unreachable!("{other:?} has no block geometry"),
    }
}

/// Q6_K: 256 elements in 210 bytes, `[128B ql, 64B qh, 16B scales, 2B f16 d]`.
/// The group order MUST match ggml's `dequantize_row_q6_K`.
fn decode_q6_k(bp: &[u8], out: &mut [f32]) {
    let ql = &bp[0..128];
    let qh = &bp[128..192];
    let scales = &bp[192..208];
    let d = host_f16_to_f32(u16::from_le_bytes([bp[208], bp[209]]));
    let mut idx = 0usize;
    for half in 0..2usize {
        let ql_ptr = &ql[64 * half..];
        let qh_ptr = &qh[32 * half..];
        let sc_ptr = &scales[8 * half..];
        // Group 0: low nibbles of ql[0..32], qh bits [0..1]
        for j in 0..32 {
            let q = ((ql_ptr[j] & 0x0F) | ((qh_ptr[j] & 3) << 4)) as i32 - 32;
            out[idx] = d * (sc_ptr[j / 16] as i8 as f32) * q as f32;
            idx += 1;
        }
        // Group 1: low nibbles of ql[32..64], qh bits [2..3]
        for j in 0..32 {
            let q = ((ql_ptr[32 + j] & 0x0F) | (((qh_ptr[j] >> 2) & 3) << 4)) as i32 - 32;
            out[idx] = d * (sc_ptr[2 + j / 16] as i8 as f32) * q as f32;
            idx += 1;
        }
        // Group 2: high nibbles of ql[0..32], qh bits [4..5]
        for j in 0..32 {
            let q = (((ql_ptr[j] >> 4) & 0x0F) | (((qh_ptr[j] >> 4) & 3) << 4)) as i32 - 32;
            out[idx] = d * (sc_ptr[4 + j / 16] as i8 as f32) * q as f32;
            idx += 1;
        }
        // Group 3: high nibbles of ql[32..64], qh bits [6..7]
        for j in 0..32 {
            let q = (((ql_ptr[32 + j] >> 4) & 0x0F) | (((qh_ptr[j] >> 6) & 3) << 4)) as i32 - 32;
            out[idx] = d * (sc_ptr[6 + j / 16] as i8 as f32) * q as f32;
            idx += 1;
        }
    }
}

/// Q4_K: 256 elements in 144 bytes, `[2B f16 d, 2B f16 dmin, 12B scales, 128B qs]`,
/// four groups of 64 values (two sub-blocks each).
fn decode_q4_k(bp: &[u8], out: &mut [f32]) {
    let d = host_f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
    let dmin = host_f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
    let (sc, m_arr) = decode_k_scales(&bp[4..16]);
    let qs = &bp[16..144];
    for group in 0..4 {
        let is = group * 2;
        let d1 = d * sc[is] as f32;
        let m1 = dmin * m_arr[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * m_arr[is + 1] as f32;
        let qs_offset = group * 32;
        for l in 0..32 {
            out[group * 64 + l] = d1 * (qs[qs_offset + l] & 0x0F) as f32 - m1;
            out[group * 64 + 32 + l] = d2 * ((qs[qs_offset + l] >> 4) & 0x0F) as f32 - m2;
        }
    }
}

/// Q5_K: 256 elements in 176 bytes, `[2B f16 d, 2B f16 dmin, 12B scales, 32B qh, 128B qs]`.
fn decode_q5_k(bp: &[u8], out: &mut [f32]) {
    let d = host_f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
    let dmin = host_f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
    let (sc, m_arr) = decode_k_scales(&bp[4..16]);
    let qh = &bp[16..48];
    let qs = &bp[48..176];
    for group in 0..4 {
        let is = group * 2;
        let d1 = d * sc[is] as f32;
        let m1 = dmin * m_arr[is] as f32;
        let d2 = d * sc[is + 1] as f32;
        let m2 = dmin * m_arr[is + 1] as f32;
        let qs_offset = group * 32;
        let u1 = group * 2;
        let u2 = u1 + 1;
        for l in 0..32 {
            let h1 = (qh[l] >> u1) & 1;
            out[group * 64 + l] = d1 * ((qs[qs_offset + l] & 0x0F) | (h1 << 4)) as f32 - m1;
            let h2 = (qh[l] >> u2) & 1;
            out[group * 64 + 32 + l] =
                d2 * (((qs[qs_offset + l] >> 4) & 0x0F) | (h2 << 4)) as f32 - m2;
        }
    }
}

/// Q2_K: 256 elements in 84 bytes, `[16B scales, 64B qs, 2B f16 d, 2B f16 dmin]`.
///
/// The qs traversal MUST follow GGML's `dequantize_row_q2_K`: two 128-value
/// groups, four shift passes (0,2,4,6) over the same 32 qs bytes per group, two
/// 16-value runs (`q[l]`, `q[l+16]`) per pass. A naive linear byte scan corrupts
/// ~74% of any real Q2_K block. See the matching converter decoder in
/// lumen-convert/src/dequant.rs::dequantize_q2_k.
fn decode_q2_k(bp: &[u8], out: &mut [f32]) {
    let scales = &bp[0..16];
    let qs = &bp[16..80];
    let d = host_f16_to_f32(u16::from_le_bytes([bp[80], bp[81]]));
    let dmin = host_f16_to_f32(u16::from_le_bytes([bp[82], bp[83]]));
    let mut idx = 0usize;
    let mut q_off = 0usize;
    let mut is = 0usize;
    for _group in 0..2 {
        let mut shift = 0u8;
        for _j in 0..4 {
            for run in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0F) as f32;
                let ml = dmin * ((sc >> 4) & 0x0F) as f32;
                for l in 0..16usize {
                    out[idx] = dl * (((qs[q_off + 16 * run + l] >> shift) & 3) as f32) - ml;
                    idx += 1;
                }
            }
            shift += 2;
        }
        q_off += 32;
    }
}

/// Q3_K: 256 elements in 110 bytes,
/// `[32B hmask, 64B qs (2-bit low), 12B scales (6-bit packed), 2B f16 d]`.
///
/// GGML `dequantize_row_q3_K` traversal: the grouped/shifted scheme of Q2_K, with
/// the hmask selector advancing per pass. The converter reads the same traversal
/// in lumen-convert/src/dequant.rs::dequantize_q3_k.
fn decode_q3_k(bp: &[u8], out: &mut [f32]) {
    let hmask = &bp[0..32];
    let qs = &bp[32..96];
    let d = host_f16_to_f32(u16::from_le_bytes([bp[108], bp[109]]));
    let sc_arr = decode_q3_k_scales(&bp[96..108]);
    let mut idx = 0usize;
    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut hbit = 1u8;
    for _group in 0..2 {
        let mut shift = 0u8;
        for _j in 0..4 {
            for run in 0..2 {
                let scale = d * (f32::from(sc_arr[is]) - 32.0);
                is += 1;
                for l in 0..16usize {
                    let q_lo = (qs[q_off + 16 * run + l] >> shift) & 3;
                    let h = u8::from((hmask[16 * run + l] & hbit) != 0);
                    out[idx] = scale * ((q_lo | (h << 2)) as i32 - 4) as f32;
                    idx += 1;
                }
            }
            shift += 2;
            hbit <<= 1;
        }
        q_off += 32;
    }
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
    if !matches!(
        scheme,
        QuantScheme::Q6_K
            | QuantScheme::Q4_K
            | QuantScheme::Q5_K
            | QuantScheme::Q2_K
            | QuantScheme::Q3_K
    ) {
        return Err(RuntimeError::Compute(format!(
            "{scheme:?} planes have no host dequant. Re-convert the model with \
             --requant q8_0 or --requant q4_0.",
        )));
    }
    // the plane must hold every element: a short plane is an error here, never a zero tail
    let block_bytes = row_bytes(scheme, 256).expect("K-quant block geometry");
    let held = raw.len() / block_bytes * 256;
    if held < n_elements {
        return Err(RuntimeError::Compute(format!(
            "{scheme:?} plane: {} bytes hold {held} elements, {n_elements} required",
            raw.len()
        )));
    }
    let n_blocks = n_elements.div_ceil(256);
    let mut out = vec![0.0f32; n_blocks * 256];
    dequant_row_into(&mut out, raw, 0, n_blocks * 256, scheme)?;
    out.truncate(n_elements);
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

#[cfg(test)]
mod q3_k_scale_tests {
    use super::decode_q3_k_scales;

    /// The 16 scales GGML's `dequantize_row_q3_K` derives, expressed the way the
    /// reference does: three little-endian u32 words masked and recombined, rather
    /// than the per-index byte reads the dequantiser uses.
    fn reference_scales(packed: &[u8; 12]) -> [u8; 16] {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let a0 = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
        let a1 = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
        let tmp = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&((a0 & KMASK2) | ((tmp & KMASK1) << 4)).to_le_bytes());
        out[4..8].copy_from_slice(&((a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4)).to_le_bytes());
        out[8..12]
            .copy_from_slice(&(((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)).to_le_bytes());
        out[12..16]
            .copy_from_slice(&(((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)).to_le_bytes());
        out
    }

    #[test]
    fn q3_k_scales_match_two_hand_decoded_blocks() {
        // Block 1: every scale byte distinct, all four 2-bit fields of each high
        // byte in use. Scale 0 is the low nibble of byte 0 (2) plus bits 0..2 of
        // byte 8 (0x1B & 3 = 3) shifted to bits 4..6, so 3 * 16 + 2 = 50. Scale 11
        // is the high nibble of byte 3 (7) plus bits 4..6 of byte 11
        // (0x4E >> 4 & 3 = 0), so 7.
        let block1 = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x1B, 0x2C, 0x3D, 0x4E,
        ];
        assert_eq!(
            decode_q3_k_scales(&block1),
            [50, 4, 22, 40, 42, 60, 62, 48, 17, 35, 53, 7, 9, 11, 13, 31]
        );

        // Block 2: the 16 nibble sources hold the 16 distinct values 0..16, and each
        // high byte's four 2-bit fields are a permutation of 0..4 with the four
        // permutations distinct, so every scale names both its nibble and its field.
        let block2 = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xE4, 0x1B, 0x4E, 0xB1,
        ];
        assert_eq!(
            decode_q3_k_scales(&block2),
            [1, 51, 37, 23, 25, 43, 61, 15, 32, 18, 4, 54, 56, 10, 28, 46]
        );
    }

    #[test]
    fn q3_k_scales_match_the_reference_shuffle() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..4096 {
            let mut packed = [0u8; 12];
            for chunk in packed.chunks_mut(8) {
                let bytes = next().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            assert_eq!(
                decode_q3_k_scales(&packed),
                reference_scales(&packed),
                "packed scales {packed:02x?}"
            );
        }
    }
}

#[cfg(test)]
mod row_decode_tests {
    use super::*;

    fn f16(v: f32) -> [u8; 2] {
        // exact for the scales used below (powers of two and small halves)
        let bits: u16 = match v {
            0.5 => 0x3800,
            1.0 => 0x3C00,
            2.0 => 0x4000,
            -0.25 => 0xB400,
            _ => unreachable!(),
        };
        bits.to_le_bytes()
    }

    /// A 3-row matrix of 64-element rows (two blocks per row), each block with
    /// its own scale, decoded row by row against the values the blocks encode.
    fn check(
        scheme: QuantScheme,
        block: impl Fn(usize, &mut Vec<u8>),
        want: impl Fn(usize, usize) -> f32,
    ) {
        let mut bytes = Vec::new();
        for b in 0..6 {
            block(b, &mut bytes);
        }
        for row in 0..3 {
            let mut out = vec![0.0f32; 64];
            dequant_row_into(&mut out, &bytes, row, 64, scheme).unwrap();
            for (j, &v) in out.iter().enumerate() {
                assert_eq!(
                    v,
                    want(row * 2 + j / 32, j % 32),
                    "{scheme:?} row {row} element {j}"
                );
            }
        }
        // the last row one byte short, and a row past the end
        let mut out = vec![0.0f32; 64];
        assert!(dequant_row_into(&mut out, &bytes[..bytes.len() - 1], 2, 64, scheme).is_err());
        assert!(dequant_row_into(&mut out, &bytes, 3, 64, scheme).is_err());
        // a row that splits a block, and a wrong output length
        if block_geometry(scheme).unwrap().0 > 1 {
            assert!(dequant_row_into(&mut out[..48], &bytes, 0, 48, scheme).is_err());
        }
        assert!(dequant_row_into(&mut out[..32], &bytes, 0, 64, scheme).is_err());
    }

    const SCALES: [f32; 6] = [1.0, 2.0, 0.5, -0.25, 1.0, 2.0];

    /// A scrambled permutation of 0..32, so a misrouted nibble or high bit shows.
    const CODES: [u8; 32] = [
        28, 9, 19, 10, 29, 5, 7, 22, 0, 14, 8, 15, 23, 24, 21, 13, 25, 27, 6, 16, 26, 18, 11, 3,
        17, 2, 1, 31, 12, 4, 30, 20,
    ];

    #[test]
    fn q8_0_rows() {
        let code = |b: usize, j: usize| (j as i32 * 7 + b as i32 * 13) % 255 - 127;
        check(
            QuantScheme::Q8_0,
            |b, out| {
                out.extend_from_slice(&f16(SCALES[b]));
                out.extend((0..32).map(|j| code(b, j) as i8 as u8));
            },
            |b, j| SCALES[b] * code(b, j) as f32,
        );
    }

    #[test]
    fn q4_0_rows() {
        let code = |b: usize, j: usize| CODES[(j + b) % 32] & 0x0F;
        check(
            QuantScheme::Q4_0,
            |b, out| {
                out.extend_from_slice(&f16(SCALES[b]));
                out.extend((0..16).map(|i| code(b, i) | (code(b, i + 16) << 4)));
            },
            |b, j| SCALES[b] * (code(b, j) as i32 - 8) as f32,
        );
    }

    #[test]
    fn q4_1_rows() {
        let code = |b: usize, j: usize| CODES[(j + 3 * b) % 32] & 0x0F;
        let min = |b: usize| SCALES[5 - b];
        check(
            QuantScheme::Q4_1,
            |b, out| {
                out.extend_from_slice(&f16(SCALES[b]));
                out.extend_from_slice(&f16(min(b)));
                out.extend((0..16).map(|i| code(b, i) | (code(b, i + 16) << 4)));
            },
            |b, j| SCALES[b] * code(b, j) as f32 + min(b),
        );
    }

    /// GGML Q5_0: element j < 16 reads the low nibble of qs[j], element j >= 16
    /// the high nibble of qs[j - 16]; qh bit j is the high bit of element j;
    /// value = scale * (code - 16).
    #[test]
    fn q5_0_rows() {
        let code = |b: usize, j: usize| CODES[(j + 5 * b) % 32];
        check(
            QuantScheme::Q5_0,
            |b, out| {
                out.extend_from_slice(&f16(SCALES[b]));
                let qh: u32 = (0..32).map(|j| (((code(b, j) >> 4) & 1) as u32) << j).sum();
                out.extend_from_slice(&qh.to_le_bytes());
                out.extend((0..16).map(|i| (code(b, i) & 0x0F) | ((code(b, i + 16) & 0x0F) << 4)));
            },
            |b, j| SCALES[b] * (code(b, j) as f32 - 16.0),
        );
    }

    #[test]
    fn float_rows() {
        let val = |b: usize, j: usize| SCALES[b] * (j as f32 - 9.5);
        check(
            QuantScheme::F32,
            |b, out| {
                if b % 2 == 0 {
                    for half in 0..2 {
                        out.extend((0..32).flat_map(|j| val(b + half, j).to_le_bytes()));
                    }
                }
            },
            val,
        );
        check(
            QuantScheme::Bf16,
            |b, out| {
                if b % 2 == 0 {
                    for half in 0..2 {
                        out.extend((0..32).flat_map(|j| {
                            ((val(b + half, j).to_bits() >> 16) as u16).to_le_bytes()
                        }));
                    }
                }
            },
            val,
        );
    }

    /// Every row of a K-quant matrix decodes to the matching slice of the whole
    /// plane: the row stride is a whole number of 256-element blocks.
    #[test]
    fn kquant_rows_match_the_plane() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for scheme in [
            QuantScheme::Q2_K,
            QuantScheme::Q3_K,
            QuantScheme::Q4_K,
            QuantScheme::Q5_K,
            QuantScheme::Q6_K,
        ] {
            let (rows, cols) = (3, 512);
            let rb = row_bytes(scheme, cols).unwrap();
            let mut bytes: Vec<u8> = (0..rows * rb)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect();
            // keep every f16 scale finite and small: 0x3C00-ish halves
            let (_, block) = block_geometry(scheme).unwrap();
            let d_offsets: &[usize] = match scheme {
                QuantScheme::Q6_K => &[208],
                QuantScheme::Q3_K => &[108],
                QuantScheme::Q2_K => &[80, 82],
                _ => &[0, 2],
            };
            for blk in bytes.chunks_exact_mut(block) {
                for &o in d_offsets {
                    blk[o] &= 0x0F;
                    blk[o + 1] = 0x34;
                }
            }
            let plane = dequant_kquant_to_f32(&bytes, scheme, rows * cols).unwrap();
            for row in 0..rows {
                let mut out = vec![0.0f32; cols];
                dequant_row_into(&mut out, &bytes, row, cols, scheme).unwrap();
                assert_eq!(
                    out,
                    plane[row * cols..(row + 1) * cols],
                    "{scheme:?} row {row}"
                );
                assert!(out.iter().all(|v| v.is_finite()));
            }
            let mut out = vec![0.0f32; cols];
            assert!(
                dequant_row_into(&mut out, &bytes[..rows * rb - 1], rows - 1, cols, scheme)
                    .is_err()
            );
            assert!(dequant_row_into(&mut out[..128], &bytes, 0, 128, scheme).is_err());
        }
    }

    #[test]
    fn schemes_without_a_host_decoder_are_refused() {
        let mut out = vec![0.0f32; 32];
        for scheme in [
            QuantScheme::Nvfp4,
            QuantScheme::Fp8E4M3,
            QuantScheme::CtInt4G32,
        ] {
            assert!(block_geometry(scheme).is_none());
            assert!(dequant_row_into(&mut out, &[0u8; 1024], 0, 32, scheme).is_err());
        }
    }
}
