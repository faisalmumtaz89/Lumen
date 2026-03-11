//! Dequantization and quantization functions for GGUF tensor data.
//!
//! Pure functions with no state.

use crate::gguf::GgmlType;
use crate::convert::ConvertError;

// ---------------------------------------------------------------------------
// F16/BF16 -> F32 conversion
// ---------------------------------------------------------------------------

/// Convert a half-precision (IEEE 754 binary16) value to f32.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;

    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal
        let f = frac as f32 / 1024.0;
        let v = f * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    let e = (exp as i32) - 15;
    let f = 1.0 + frac as f32 / 1024.0;
    let v = f * 2.0f32.powi(e);
    if sign == 1 { -v } else { v }
}

/// Convert a bfloat16 value to f32 (zero-extend the lower 16 bits).
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert a buffer of F16 values (as raw bytes) to F32 bytes.
pub(crate) fn convert_f16_bytes_to_f32(src: &[u8]) -> Vec<u8> {
    let count = src.len() / 2;
    let mut out = Vec::with_capacity(count * 4);
    for chunk in src.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = f16_to_f32(bits);
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Convert a buffer of BF16 values (as raw bytes) to F32 bytes.
pub(crate) fn convert_bf16_bytes_to_f32(src: &[u8]) -> Vec<u8> {
    let count = src.len() / 2;
    let mut out = Vec::with_capacity(count * 4);
    for chunk in src.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = bf16_to_f32(bits);
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Ensure global tensor data is F32. Convert F16/BF16/quantized to F32.
///
/// Global tensors (embedding, final_norm, output_proj) MUST always be F32 in
/// the LBC file because weight providers read them via `bytes_to_f32` /
/// `read_f32_tensor`. This function always dequantizes quantized formats
/// (Q8_0, Q4_K, etc.) regardless of the `--dequantize` flag, since that flag
/// only controls per-layer weight tensors.
pub(crate) fn ensure_f32_global(
    data: Vec<u8>,
    ggml_type: GgmlType,
    tensor_name: &str,
    n_elements: u64,
) -> Result<Vec<u8>, ConvertError> {
    match ggml_type {
        GgmlType::F32 => Ok(data),
        GgmlType::F16 => Ok(convert_f16_bytes_to_f32(&data)),
        GgmlType::BF16 => Ok(convert_bf16_bytes_to_f32(&data)),
        other => {
            // Always dequantize quantized globals to F32
            dequantize_to_f32_bytes(&data, other, n_elements, tensor_name)
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantization kernels
// ---------------------------------------------------------------------------

/// Dequantize Q8_0: block_size=32, type_size=34.
/// Layout: [2 bytes f16 scale] [32 bytes int8 values]
pub(crate) fn dequantize_q8_0(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 34;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        for i in 0..32 {
            if written >= n {
                break;
            }
            let q = src[offset + 2 + i] as i8;
            let val = scale * q as f32;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q4_0: block_size=32, type_size=18.
/// Layout: [2 bytes f16 scale] [16 bytes packed 4-bit nibbles]
/// GGML de-interleaved order: indices 0-15 use lo nibbles, indices 16-31 use hi nibbles.
pub(crate) fn dequantize_q4_0(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 18;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        // De-interleaved: first 16 elements from lo nibbles (indices 0-15)
        for i in 0..16 {
            if written >= n {
                break;
            }
            let byte = src[offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let val = scale * lo as f32;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        // Then 16 elements from hi nibbles (indices 16-31)
        for i in 0..16 {
            if written >= n {
                break;
            }
            let byte = src[offset + 2 + i];
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            let val = scale * hi as f32;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q4_1: block_size=32, type_size=20.
/// Layout: [2 bytes f16 scale] [2 bytes f16 min] [16 bytes packed 4-bit nibbles]
/// GGML de-interleaved order: indices 0-15 use lo nibbles, indices 16-31 use hi nibbles.
pub(crate) fn dequantize_q4_1(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 20;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([src[offset + 2], src[offset + 3]]));
        // De-interleaved: first 16 elements from lo nibbles (indices 0-15)
        for i in 0..16 {
            if written >= n {
                break;
            }
            let byte = src[offset + 4 + i];
            let lo = (byte & 0x0F) as f32;
            let val = scale * lo + min;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        // Then 16 elements from hi nibbles (indices 16-31)
        for i in 0..16 {
            if written >= n {
                break;
            }
            let byte = src[offset + 4 + i];
            let hi = ((byte >> 4) & 0x0F) as f32;
            let val = scale * hi + min;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q5_0: block_size=32, type_size=22.
/// Layout: [2 bytes f16 scale] [4 bytes high-bit array] [16 bytes packed 4-bit nibbles]
pub(crate) fn dequantize_q5_0(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 22;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let qh_bytes = &src[offset + 2..offset + 6]; // 4 bytes of high bits
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);
        let qs = &src[offset + 6..offset + 22]; // 16 bytes packed nibbles
        for j in 0..32 {
            if written >= n {
                break;
            }
            let nibble = if j < 16 {
                qs[j] & 0x0F
            } else {
                (qs[j - 16] >> 4) & 0x0F
            };
            let high_bit = ((qh >> j) & 1) as u8;
            let combined = (nibble | (high_bit << 4)) as f32 - 16.0;
            let val = scale * combined;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        offset += block_size;
    }
    out
}

/// Decode the 12 bytes of 6-bit scales used by Q4_K, Q5_K.
/// Returns 8 pairs of (sc, m).
pub(crate) fn decode_k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
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

/// Dequantize Q4_K: block_size=256, type_size=144.
/// Layout: [2 bytes f16 d] [2 bytes f16 dmin] [12 bytes scales] [128 bytes qs]
pub(crate) fn dequantize_q4_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 144;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let d = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([src[offset + 2], src[offset + 3]]));
        let scales_slice = &src[offset + 4..offset + 16];
        let (sc, m_arr) = decode_k_scales(scales_slice);
        let qs = &src[offset + 16..offset + 144];

        // Process 4 groups of 64 values (2 sub-blocks each)
        for group in 0..4 {
            let is = group * 2; // sub-block pair index
            let d1 = d * sc[is] as f32;
            let m1 = dmin * m_arr[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * m_arr[is + 1] as f32;
            let qs_offset = group * 32;

            // First 32 values: low nibbles
            for l in 0..32 {
                if written >= n {
                    break;
                }
                let val = d1 * (qs[qs_offset + l] & 0x0F) as f32 - m1;
                out.extend_from_slice(&val.to_le_bytes());
                written += 1;
            }
            // Second 32 values: high nibbles
            for l in 0..32 {
                if written >= n {
                    break;
                }
                let val = d2 * ((qs[qs_offset + l] >> 4) & 0x0F) as f32 - m2;
                out.extend_from_slice(&val.to_le_bytes());
                written += 1;
            }
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q5_K: block_size=256, type_size=176.
/// Layout: [2 bytes f16 d] [2 bytes f16 dmin] [12 bytes scales] [32 bytes qh] [128 bytes qs]
pub(crate) fn dequantize_q5_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 176;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let d = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([src[offset + 2], src[offset + 3]]));
        let scales_slice = &src[offset + 4..offset + 16];
        let (sc, m_arr) = decode_k_scales(scales_slice);
        let qh = &src[offset + 16..offset + 48]; // 32 bytes high bits
        let qs = &src[offset + 48..offset + 176]; // 128 bytes 4-bit values

        // Process 4 groups of 64 values
        for group in 0..4 {
            let is = group * 2;
            let d1 = d * sc[is] as f32;
            let m1 = dmin * m_arr[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * m_arr[is + 1] as f32;
            let qs_offset = group * 32;
            let u1 = group * 2; // high-bit shift for first sub-block
            let u2 = u1 + 1; // high-bit shift for second sub-block

            // First 32 values: low nibbles + high bit
            for l in 0..32 {
                if written >= n {
                    break;
                }
                let h_bit = (qh[l] >> u1) & 1;
                let val = d1 * ((qs[qs_offset + l] & 0x0F) | (h_bit << 4)) as f32 - m1;
                out.extend_from_slice(&val.to_le_bytes());
                written += 1;
            }
            // Second 32 values: high nibbles + high bit
            for l in 0..32 {
                if written >= n {
                    break;
                }
                let h_bit = (qh[l] >> u2) & 1;
                let val = d2 * (((qs[qs_offset + l] >> 4) & 0x0F) | (h_bit << 4)) as f32 - m2;
                out.extend_from_slice(&val.to_le_bytes());
                written += 1;
            }
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q6_K: block_size=256, type_size=210.
/// Layout: [128 bytes ql] [64 bytes qh] [16 bytes scales (int8)] [2 bytes f16 d]
/// Note: d is at the END of the block.
pub(crate) fn dequantize_q6_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 210;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let ql = &src[offset..offset + 128];
        let qh = &src[offset + 128..offset + 192];
        let scales = &src[offset + 192..offset + 208];
        let d = f16_to_f32(u16::from_le_bytes([src[offset + 208], src[offset + 209]]));

        // Q6_K dequantization following the ggml reference layout.
        // Each block has 256 values split into two halves of 128.
        // Each half uses 64 bytes of ql, 32 bytes of qh, and 8 scales.
        // Within each half, values are arranged as 4 groups of 32:
        //   group 0: ql[0..32] low nibble,  qh[0..32] bits [0..1]
        //   group 1: ql[0..32] high nibble, qh[0..32] bits [2..3]
        //   group 2: ql[32..64] low nibble, qh[0..32] bits [4..5]
        //   group 3: ql[32..64] high nibble, qh[0..32] bits [6..7]
        // Each group of 32 uses 2 consecutive scales (16 values per scale).
        let mut idx = 0usize;
        for half in 0..2 {
            let ql_ptr = &ql[64 * half..];
            let qh_ptr = &qh[32 * half..];
            let sc_ptr = &scales[8 * half..];

            // Group 0: low nibbles of ql[0..32], qh bits [0..1]
            for j in 0..32 {
                if written + idx >= n {
                    break;
                }
                let q_lo = ql_ptr[j] & 0x0F;
                let q_hi = (qh_ptr[j] & 3) << 4;
                let q = (q_lo | q_hi) as i32 - 32;
                let sc = sc_ptr[j / 16] as i8 as f32;
                let val = d * sc * q as f32;
                out.extend_from_slice(&val.to_le_bytes());
                idx += 1;
            }
            // Group 1: high nibbles of ql[0..32], qh bits [2..3]
            for j in 0..32 {
                if written + idx >= n {
                    break;
                }
                let q_lo = (ql_ptr[j] >> 4) & 0x0F;
                let q_hi = ((qh_ptr[j] >> 2) & 3) << 4;
                let q = (q_lo | q_hi) as i32 - 32;
                let sc = sc_ptr[2 + j / 16] as i8 as f32;
                let val = d * sc * q as f32;
                out.extend_from_slice(&val.to_le_bytes());
                idx += 1;
            }
            // Group 2: low nibbles of ql[32..64], qh bits [4..5]
            for j in 0..32 {
                if written + idx >= n {
                    break;
                }
                let q_lo = ql_ptr[32 + j] & 0x0F;
                let q_hi = ((qh_ptr[j] >> 4) & 3) << 4;
                let q = (q_lo | q_hi) as i32 - 32;
                let sc = sc_ptr[4 + j / 16] as i8 as f32;
                let val = d * sc * q as f32;
                out.extend_from_slice(&val.to_le_bytes());
                idx += 1;
            }
            // Group 3: high nibbles of ql[32..64], qh bits [6..7]
            for j in 0..32 {
                if written + idx >= n {
                    break;
                }
                let q_lo = (ql_ptr[32 + j] >> 4) & 0x0F;
                let q_hi = ((qh_ptr[j] >> 6) & 3) << 4;
                let q = (q_lo | q_hi) as i32 - 32;
                let sc = sc_ptr[6 + j / 16] as i8 as f32;
                let val = d * sc * q as f32;
                out.extend_from_slice(&val.to_le_bytes());
                idx += 1;
            }
        }
        written += idx;
        offset += block_size;
    }
    out
}

/// Dequantize Q2_K: block_size=256, type_size=84.
/// Layout: [16 bytes scales] [16 bytes zero-points] [64 bytes qs] [2 bytes f16 d] [2 bytes f16 dmin]
///
/// scales: 16 x uint8. Each byte has a 4-bit scale (low) and 4-bit min (high)
/// packed per 16-value sub-block.
/// qs: 64 bytes, 2-bit packed (4 values per byte).
pub(crate) fn dequantize_q2_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 84;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scales = &src[offset..offset + 16];
        let qs = &src[offset + 16..offset + 80];
        let d = f16_to_f32(u16::from_le_bytes([src[offset + 80], src[offset + 81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([src[offset + 82], src[offset + 83]]));

        // 16 sub-blocks of 16 values = 256 values
        // Each scales[j] byte: low 4 bits = scale, high 4 bits = min
        let mut q_idx = 0usize;
        for &scale_byte in scales.iter().take(16) {
            let sc = d * (scale_byte & 0x0F) as f32;
            let m = dmin * ((scale_byte >> 4) & 0x0F) as f32;
            // 16 values, 2-bit each = 4 bytes
            for _l in 0..4 {
                if written >= n {
                    break;
                }
                let byte = qs[q_idx];
                q_idx += 1;
                for shift in [0, 2, 4, 6] {
                    if written >= n {
                        break;
                    }
                    let q = ((byte >> shift) & 3) as f32;
                    let val = sc * q - m;
                    out.extend_from_slice(&val.to_le_bytes());
                    written += 1;
                }
            }
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q3_K: block_size=256, type_size=110.
/// Layout: [32 bytes hmask] [64 bytes qs (2-bit low)] [12 bytes scales (6-bit packed)] [2 bytes f16 d]
pub(crate) fn dequantize_q3_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 110;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let hmask = &src[offset..offset + 32];
        let qs = &src[offset + 32..offset + 96];
        let scale_bytes = &src[offset + 96..offset + 108];
        let d = f16_to_f32(u16::from_le_bytes([src[offset + 108], src[offset + 109]]));

        // Decode 16 6-bit scales from 12 bytes
        // The packing is the same as K-quant scales but for 16 values from 12 bytes:
        // Q3_K encodes 16 signed 6-bit scales into 12 bytes (GGML layout):
        //   scales[0..3]  = low nibbles of aux8[0..3]
        //   scales[4..7]  = high nibbles of aux8[0..3]
        //   scales[8..11] = low nibbles of aux8[4..7]
        //   scales[12..15]= high nibbles of aux8[4..7]
        //   high 2 bits:    scales[j] |= ((aux8[8 + j/4] >> (2*(j%4))) & 3) << 4
        //   convert signed: scales[j] = (scales[j] as i8 - 32)

        let mut sc_arr = [0u8; 16];
        // Low nibbles and high nibbles from first 8 bytes
        for j in 0..4 {
            sc_arr[j] = scale_bytes[j] & 0x0F;
            sc_arr[j + 4] = (scale_bytes[j] >> 4) & 0x0F;
        }
        for j in 0..4 {
            sc_arr[j + 8] = scale_bytes[4 + j] & 0x0F;
            sc_arr[j + 12] = (scale_bytes[4 + j] >> 4) & 0x0F;
        }
        // High 2 bits from bytes 8..12
        for (j, sc) in sc_arr.iter_mut().enumerate() {
            let byte_idx = 8 + j / 4;
            let bit_shift = 2 * (j % 4);
            *sc |= ((scale_bytes[byte_idx] >> bit_shift) & 3) << 4;
        }

        // Process 256 values in 16 sub-blocks of 16
        // hmask: 32 bytes = 256 bits, one per value (the high bit)
        // qs: 64 bytes = 256 2-bit values (4 per byte)
        let mut q_byte_idx = 0usize;
        let mut q_bit_shift = 0u8;
        let mut val_idx = 0usize;
        for &sc_val in &sc_arr {
            let scale = d * (sc_val as i8 as f32 - 32.0);
            for _l in 0..16 {
                if written >= n {
                    break;
                }
                // Get 2-bit low value from qs
                let q_lo = (qs[q_byte_idx] >> q_bit_shift) & 3;
                q_bit_shift += 2;
                if q_bit_shift >= 8 {
                    q_bit_shift = 0;
                    q_byte_idx += 1;
                }
                // Get high bit from hmask
                let hmask_byte = val_idx / 8;
                let hmask_bit = val_idx % 8;
                let h = (hmask[hmask_byte] >> hmask_bit) & 1;
                // Combine: 3-bit value = q_lo | (h << 2), then subtract 4 to center
                let q = (q_lo | (h << 2)) as i32 - 4;
                let val = scale * q as f32;
                out.extend_from_slice(&val.to_le_bytes());
                written += 1;
                val_idx += 1;
            }
        }
        offset += block_size;
    }
    out
}


// ---------------------------------------------------------------------------
// F32 -> Q4_0 quantization (for requantization during conversion)
// ---------------------------------------------------------------------------

/// Convert f32 to f16 bits (truncation, no rounding -- matches common quantization practice).
pub(crate) fn f32_to_f16_bits_convert(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        let f16_frac = if frac != 0 { 0x200 } else { 0 }; // NaN or Inf
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow -> Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        // Underflow -> zero (subnormals not handled for quantization scale)
        return (sign << 15) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Quantize F32 bytes to Q4_0 format.
/// Input: raw F32 bytes (little-endian), n_elements must be divisible by 32.
/// Output: Q4_0 bytes ([f16 scale][16 bytes nibbles] per 32-element block).
pub(crate) fn quantize_f32_to_q4_0(f32_bytes: &[u8], n_elements: usize) -> Vec<u8> {
    assert!(n_elements % 32 == 0, "Q4_0 quantization requires elements divisible by 32");
    let num_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);

    for block in 0..num_blocks {
        let base = block * 32;
        // Read 32 f32 values
        let mut vals = [0.0f32; 32];
        for i in 0..32 {
            let off = (base + i) * 4;
            vals[i] = f32::from_le_bytes([
                f32_bytes[off], f32_bytes[off + 1], f32_bytes[off + 2], f32_bytes[off + 3],
            ]);
        }

        // Find max absolute value for scale
        let mut amax = 0.0f32;
        for &v in &vals {
            amax = amax.max(v.abs());
        }

        // Scale: map [-amax, amax] to [-8, 7] (4-bit signed range with offset)
        // Q4_0 stores nibbles as unsigned 0..15, dequant subtracts 8: value = (nibble - 8) * scale
        // So nibble = round(value / scale) + 8, clamped to [0, 15]
        let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 }; // 7 because max positive nibble-8 = 7
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write f16 scale
        let scale_f16 = f32_to_f16_bits_convert(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize and pack into nibbles (GGML de-interleaved order)
        // vals[0..16] -> lo nibbles of bytes 0..15
        // vals[16..32] -> hi nibbles of bytes 0..15
        for i in 0..16 {
            let q_lo = ((vals[i] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let q_hi = ((vals[i + 16] * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(q_lo | (q_hi << 4));
        }
    }

    out
}

/// Quantize F32 bytes to Q8_0 format.
/// Input: raw F32 bytes (little-endian), n_elements must be divisible by 32.
/// Output: Q8_0 bytes ([f16 scale][32 x i8 quants] per 32-element block = 34 bytes/block).
pub(crate) fn quantize_f32_to_q8_0(f32_bytes: &[u8], n_elements: usize) -> Vec<u8> {
    assert!(n_elements % 32 == 0, "Q8_0 quantization requires elements divisible by 32");
    let num_blocks = n_elements / 32;
    // Q8_0 block = 2 bytes (f16 scale) + 32 bytes (i8 quants) = 34 bytes
    let mut out = Vec::with_capacity(num_blocks * 34);

    for block in 0..num_blocks {
        let base = block * 32;
        // Read 32 f32 values
        let mut vals = [0.0f32; 32];
        for i in 0..32 {
            let off = (base + i) * 4;
            vals[i] = f32::from_le_bytes([
                f32_bytes[off], f32_bytes[off + 1], f32_bytes[off + 2], f32_bytes[off + 3],
            ]);
        }

        // Find max absolute value for scale
        let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        // Write f16 scale
        let scale_f16 = f32_to_f16_bits_convert(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize to i8
        for i in 0..32 {
            let q = (vals[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// MXFP4 dequantization (GGML type 39)
// ---------------------------------------------------------------------------

/// MXFP4 kvalues lookup table: maps a 4-bit index to an int8 representative value.
/// From ggml: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Convert an E8M0 (8-bit exponent, 0-bit mantissa) byte to an f32 scale factor.
/// E8M0 represents a pure power-of-two value.  The "half" variant used by ggml
/// for MXFP4 dequantization constructs an IEEE-754 float from the raw exponent:
///   - For e < 2:  result = f32::from_bits(0x00200000 << e)  (subnormal range)
///   - For e >= 2: result = f32::from_bits((e - 1) << 23)    (normal range)
pub(crate) fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits: u32 = if e < 2 {
        0x00200000_u32 << (e as u32)
    } else {
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// Dequantize MXFP4 data to F32.
///
/// MXFP4 block format (17 bytes per 32 elements):
///   - 1 byte: E8M0 scale exponent
///   - 16 bytes: packed 4-bit quantized indices (2 per byte)
///
/// Layout: for byte qs[j], the low nibble (qs[j] & 0x0F) maps to output[j],
/// and the high nibble (qs[j] >> 4) maps to output[j + 16].
pub(crate) fn dequantize_mxfp4(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    assert!(n % 32 == 0, "MXFP4 dequantization requires elements divisible by 32");
    let num_blocks = n / 32;
    let mut out = Vec::with_capacity(n * 4);

    let block_bytes = 17; // 1 byte E8M0 + 16 bytes nibbles
    for i in 0..num_blocks {
        let base = i * block_bytes;
        let e = src[base];
        let d = e8m0_to_f32_half(e);

        // First pass: low nibbles -> positions 0..16
        // Second pass: high nibbles -> positions 16..32
        let mut vals = [0.0f32; 32];
        for j in 0..16 {
            let byte = src[base + 1 + j];
            let x0 = KVALUES_MXFP4[(byte & 0x0F) as usize];
            let x1 = KVALUES_MXFP4[(byte >> 4) as usize];
            vals[j] = (x0 as f32) * d;
            vals[j + 16] = (x1 as f32) * d;
        }

        for v in &vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    out
}

/// Dequantize a tensor from any supported GGML format to F32 bytes.
/// Returns the raw F32 bytes (little-endian).
pub(crate) fn dequantize_to_f32_bytes(
    src: &[u8],
    ggml_type: GgmlType,
    n_elements: u64,
    tensor_name: &str,
) -> Result<Vec<u8>, ConvertError> {
    match ggml_type {
        GgmlType::F32 => Ok(src.to_vec()),
        GgmlType::F16 => Ok(convert_f16_bytes_to_f32(src)),
        GgmlType::BF16 => Ok(convert_bf16_bytes_to_f32(src)),
        GgmlType::Q8_0 => Ok(dequantize_q8_0(src, n_elements)),
        GgmlType::Q4_0 => Ok(dequantize_q4_0(src, n_elements)),
        GgmlType::Q4_1 => Ok(dequantize_q4_1(src, n_elements)),
        GgmlType::Q5_0 => Ok(dequantize_q5_0(src, n_elements)),
        GgmlType::Q4_K => Ok(dequantize_q4_k(src, n_elements)),
        GgmlType::Q5_K => Ok(dequantize_q5_k(src, n_elements)),
        GgmlType::Q6_K => Ok(dequantize_q6_k(src, n_elements)),
        GgmlType::Q2_K => Ok(dequantize_q2_k(src, n_elements)),
        GgmlType::Q3_K => Ok(dequantize_q3_k(src, n_elements)),
        GgmlType::MXFP4 => Ok(dequantize_mxfp4(src, n_elements)),
        other => Err(ConvertError::UnsupportedTensorType {
            tensor: tensor_name.to_string(),
            ggml_type: format!("{other:?} (dequantization not supported)"),
        }),
    }
}
