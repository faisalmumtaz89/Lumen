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

/// Dequantize Q8_1: block_size=32, type_size=36.
/// Layout: [2 bytes f16 scale] [2 bytes f16 min] [32 bytes int8 values]
/// Dequant: val[i] = f32(scale) * f32(qs[i]) + f32(min)
pub(crate) fn dequantize_q8_1(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 36;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([src[offset + 2], src[offset + 3]]));
        for i in 0..32 {
            if written >= n {
                break;
            }
            let q = src[offset + 4 + i] as i8;
            let val = scale * q as f32 + min;
            out.extend_from_slice(&val.to_le_bytes());
            written += 1;
        }
        offset += block_size;
    }
    out
}

/// Dequantize Q5_1: block_size=32, type_size=24.
/// Layout: [2 bytes f16 scale] [2 bytes f16 min] [4 bytes high-bit array] [16 bytes packed 4-bit nibbles]
/// Dequant: val[i] = f32(scale) * (nibble | (high_bit << 4)) + f32(min)
pub(crate) fn dequantize_q5_1(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = Vec::with_capacity(n * 4);
    let block_size = 24;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scale = f16_to_f32(u16::from_le_bytes([src[offset], src[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([src[offset + 2], src[offset + 3]]));
        let qh_bytes = &src[offset + 4..offset + 8]; // 4 bytes of high bits
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);
        let qs = &src[offset + 8..offset + 24]; // 16 bytes packed nibbles
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
            let combined = (nibble | (high_bit << 4)) as f32;
            let val = scale * combined + min;
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
/// Layout: [16 bytes scales] [64 bytes qs] [2 bytes f16 d] [2 bytes f16 dmin]
///
/// scales: 16 x uint8. Each byte has a 4-bit scale (low) and 4-bit min (high)
/// packed per 16-value sub-block.
/// qs: 64 bytes, 2-bit packed.
///
/// CRITICAL — the qs traversal order is GGML's `dequantize_row_q2_K`, NOT a
/// naive linear "byte 0 -> values 0..3, byte 1 -> values 4..7" scan. GGML
/// processes the 256 values as two 128-value groups; within a group it makes
/// four passes (shift = 0,2,4,6) over the SAME 32 qs bytes, emitting two runs
/// of 16 values per pass (`q[l]` then `q[l+16]`) and consuming two scale
/// sub-blocks per pass. A naive linear scan happens to agree only when every
/// quant and every scale in the block is identical (the degenerate case the
/// old unit test used), which is why this bug shipped: it corrupts ~74% of the
/// values in any real Q2_K block. The 27B (Qwen3.6) GGUF is Q2_K, so the
/// converter dequant->Q8_0 produced corrupted weights -> token salad; the 9B
/// is Q8_0-sourced and never reaches this code, so it is byte-unaffected.
///
/// Reference (ggml-quants.c, stable across llama.cpp builds incl. b9430):
/// ```c
/// const uint8_t * q = x[i].qs; int is = 0; float dl, ml;
/// for (int n = 0; n < QK_K; n += 128) {
///     int shift = 0;
///     for (int j = 0; j < 4; ++j) {
///         uint8_t sc = x[i].scales[is++];
///         dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
///         for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l]    >> shift) & 3) - ml;
///         sc = x[i].scales[is++];
///         dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
///         for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l+16] >> shift) & 3) - ml;
///         shift += 2;
///     }
///     q += 32;
/// }
/// ```
pub(crate) fn dequantize_q2_k(src: &[u8], n_elements: u64) -> Vec<u8> {
    let n = n_elements as usize;
    let mut out = vec![0u8; n * 4];
    let block_size = 84;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n && offset + block_size <= src.len() {
        let scales = &src[offset..offset + 16];
        let qs = &src[offset + 16..offset + 80];
        let d = f16_to_f32(u16::from_le_bytes([src[offset + 80], src[offset + 81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([src[offset + 82], src[offset + 83]]));

        // Block-local output index (0..256). The caller may request fewer than
        // 256 elements on the final block; `written < n` guards every store.
        let mut y_local = 0usize;
        let mut q_off = 0usize; // qs byte offset for the current 128-value group
        let mut is = 0usize; // scale sub-block cursor
        // Two groups of 128 values; q pointer advances 32 bytes per group.
        for _group in 0..2 {
            let mut shift = 0u8;
            for _j in 0..4 {
                let sc0 = scales[is];
                is += 1;
                let dl0 = d * (sc0 & 0x0F) as f32;
                let ml0 = dmin * ((sc0 >> 4) & 0x0F) as f32;
                for l in 0..16usize {
                    if written >= n {
                        return out;
                    }
                    let q = ((qs[q_off + l] >> shift) & 3) as f32;
                    let val = dl0 * q - ml0;
                    let o = (written) * 4;
                    out[o..o + 4].copy_from_slice(&val.to_le_bytes());
                    written += 1;
                    y_local += 1;
                }
                let sc1 = scales[is];
                is += 1;
                let dl1 = d * (sc1 & 0x0F) as f32;
                let ml1 = dmin * ((sc1 >> 4) & 0x0F) as f32;
                for l in 0..16usize {
                    if written >= n {
                        return out;
                    }
                    let q = ((qs[q_off + l + 16] >> shift) & 3) as f32;
                    let val = dl1 * q - ml1;
                    let o = (written) * 4;
                    out[o..o + 4].copy_from_slice(&val.to_le_bytes());
                    written += 1;
                    y_local += 1;
                }
                shift += 2;
            }
            q_off += 32;
        }
        let _ = y_local;
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

        // Decode 16 6-bit scales from 12 bytes using GGML's EXACT `aux`
        // uint32 shuffle (ggml-quants.c `dequantize_row_q3_K`). The previous
        // nibble-by-nibble decode produced a DIFFERENT 16-scale permutation
        // than ggml (verified against llama.cpp reference activations: it gave the wrong ffn_out;
        // the aux shuffle reproduces llama's ffn_out-0 = 0.005 byte-for-byte).
        // This corrupted every Q3_K tensor (e.g. all ffn_down on the 27B).
        //
        // Reference:
        //   const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        //   memcpy(aux, scales, 12);
        //   uint32_t tmp = aux[2];
        //   aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        //   aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        //   aux[0] = (aux[0]      & kmask2) | (((tmp >> 0) & kmask1) << 4);
        //   aux[1] = (aux[1]      & kmask2) | (((tmp >> 2) & kmask1) << 4);
        //   scales = (const int8_t*)aux;  // 16 bytes, each used as (sc - 32)
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let a0 = u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
        let a1 = u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]);
        let tmp = u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]);
        let out0 = (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        let out1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let out2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        let out3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        let mut sc_arr = [0u8; 16];
        sc_arr[0..4].copy_from_slice(&out0.to_le_bytes());
        sc_arr[4..8].copy_from_slice(&out1.to_le_bytes());
        sc_arr[8..12].copy_from_slice(&out2.to_le_bytes());
        sc_arr[12..16].copy_from_slice(&out3.to_le_bytes());

        // qs / hmask traversal follows GGML's `dequantize_row_q3_K`, which is
        // the SAME grouped/shifted scheme as Q2_K (NOT a linear byte scan): two
        // 128-value groups; within a group, four passes (shift = 0,2,4,6) over
        // the same 32 qs bytes, two 16-value runs (`q[l]`, `q[l+16]`) per pass,
        // consuming two scale sub-blocks per pass. The hmask high-bit selector
        // `m` advances ONCE per pass (`m <<= 1`), so all 32 values of a pass
        // share the same hmask bit position; the hmask BYTE is the in-run index
        // `l` (0..31). The naive linear scan corrupts ~82% of values on a real
        // block (matches only the degenerate uniform case the old test used).
        //
        // Reference (ggml-quants.c):
        // ```c
        // const uint8_t * q = x[i].qs; const uint8_t * hm = x[i].hmask; uint8_t m = 1;
        // for (int n = 0; n < QK_K; n += 128) {
        //   int shift = 0;
        //   for (int j = 0; j < 4; ++j) {
        //     int8_t sc = scales[is++]; float dl = d_all * (sc - 32);
        //     for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l]    >> shift) & 3) - ((hm[l]    & m) ? 0 : 4));
        //     sc = scales[is++]; dl = d_all * (sc - 32);
        //     for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
        //     shift += 2; m <<= 1;
        //   }
        //   q += 32;
        // }
        // ```
        // `(int8_t)(q2 & 3) - (h ? 0 : 4)` is algebraically `(q_lo | (h<<2)) - 4`.
        let mut q_off = 0usize; // qs byte offset for the current 128-value group
        let mut is = 0usize; // scale sub-block cursor
        let mut hbit = 1u8; // hmask high-bit selector, advances per pass
        for _group in 0..2 {
            let mut shift = 0u8;
            for _j in 0..4 {
                let scale0 = d * (sc_arr[is] as i8 as f32 - 32.0);
                is += 1;
                for l in 0..16usize {
                    if written >= n {
                        return out;
                    }
                    let q_lo = (qs[q_off + l] >> shift) & 3;
                    let h = u8::from((hmask[l] & hbit) != 0);
                    let q = (q_lo | (h << 2)) as i32 - 4;
                    let val = scale0 * q as f32;
                    out.extend_from_slice(&val.to_le_bytes());
                    written += 1;
                }
                let scale1 = d * (sc_arr[is] as i8 as f32 - 32.0);
                is += 1;
                for l in 0..16usize {
                    if written >= n {
                        return out;
                    }
                    let q_lo = (qs[q_off + l + 16] >> shift) & 3;
                    let h = u8::from((hmask[l + 16] & hbit) != 0);
                    let q = (q_lo | (h << 2)) as i32 - 4;
                    let val = scale1 * q as f32;
                    out.extend_from_slice(&val.to_le_bytes());
                    written += 1;
                }
                shift += 2;
                hbit <<= 1;
            }
            q_off += 32;
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
        // f16 subnormal range. Flushing these to zero silently destroys any
        // Q8_0/Q4_0 block whose per-block scale (amax/127 or amax/7) lands
        // below the smallest NORMAL f16 (2^-14 ≈ 6.10e-5). For Q2_K/Q4_K
        // tensors upcast to Q8_0 (Metal path), small-magnitude weight blocks
        // produce exactly such scales, so flush-to-zero wiped ~50-90% of the
        // blocks in low-magnitude rows -> token salad on the 27B.
        //
        // f16 subnormals represent values down to 2^-24 ≈ 5.96e-8 with a
        // 10-bit fraction and implicit exponent 2^-14. Encode them properly:
        //   value = frac/1024 * 2^-14, where frac is the 10-bit mantissa.
        // `new_exp <= 0` means the unbiased f32 exponent E = exp-127 satisfies
        // E <= -15. The subnormal fraction is the full mantissa (with the
        // implicit leading 1) right-shifted by (1 - new_exp) bits.
        if new_exp < -10 {
            // Below the smallest representable subnormal (2^-24): true underflow.
            return (sign << 15) as u16;
        }
        // Restore the implicit leading 1, then shift into the 10-bit subnormal
        // field. shift = 14 (frac is 23-bit) - 10 (target) + (1 - new_exp).
        let mantissa = frac | 0x0080_0000; // 24-bit significand (1.fff...)
        let shift = (14 - new_exp) as u32; // new_exp in [-10, 0] -> shift in [14, 24]
        let f16_frac = mantissa >> shift; // truncation, consistent with normal path
        return ((sign << 15) | (f16_frac & 0x3FF)) as u16;
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
        // 7 because max positive nibble-8 = 7. Clamp to the f16 max normal
        // (65504): the scale is STORED as f16, so amax > ~4.59e5 would make
        // amax/7 overflow f16 to Inf and the whole block dequantize to NaN
        // (reproduced 2026-06-10 with a synthetic high-amplitude block). With the
        // clamp, out-of-range values saturate at the nibble bounds instead.
        let scale = if amax == 0.0 { 1.0 } else { (amax / 7.0).min(65504.0) };
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
        GgmlType::Q8_1 => Ok(dequantize_q8_1(src, n_elements)),
        GgmlType::Q4_0 => Ok(dequantize_q4_0(src, n_elements)),
        GgmlType::Q4_1 => Ok(dequantize_q4_1(src, n_elements)),
        GgmlType::Q5_0 => Ok(dequantize_q5_0(src, n_elements)),
        GgmlType::Q5_1 => Ok(dequantize_q5_1(src, n_elements)),
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

// ---------------------------------------------------------------------------
// F32 -> Q3_K quantization (Q3_K_M variant)
// ---------------------------------------------------------------------------
//
// Q3_K block: 256 elements, 110 bytes.
// Layout (matches dequantize_q3_k above):
//   [32B hmask][64B qs 2-bit low][12B scales 6-bit packed][2B f16 d]
//   16 sub-blocks of 16 elements each share a 6-bit signed scale ([-32,31],
//   stored unsigned 0..63 = scale + 32). Per element: 3-bit signed value
//   [-4,3] stored unsigned 0..7; low 2 bits in qs, high 1 bit in hmask.
//   Super-block scale d (f16) multiplies the sub-block scale.
//   Dequant: value = d * (sc6 - 32) * (q3 - 4)
//
// Encoder mirrors the GGML Q3_K reference (`quantize_row_q3_K_impl` in
// `ggml-quants.c`):
//   1. Per 16-element sub-block, find float scale via make_qx_quants
//      (rmse_type=1, nmax=4): scale that maximizes <q,x>^2/<q,q> for clamped q.
//   2. super-d = max_abs(sub_scale_float) / 32  (encoded as f16)
//      Sign convention: iscale = -32 / max_abs => super-d = -max_abs / 32.
//   3. int_scale_per_sb = round(iscale * sub_scale_float), clamped [-32, 31].
//   4. Per element: q = round(x / (super_d * int_scale)), clamped [-4, 3].
//
// Currently exercised by the `q3_k_encoder_tests` module only -- Q3_K is not
// yet a runtime target. The encoder is preserved for future round-trip
// requantisation against Q3_K-quantised GGUFs.
#[cfg(test)]
pub(crate) fn quantize_f32_to_q3_k(f32_bytes: &[u8], n_elements: usize) -> Vec<u8> {
    assert!(n_elements % 256 == 0, "Q3_K quantization requires elements divisible by 256");
    let num_blocks = n_elements / 256;
    let mut out = vec![0u8; num_blocks * 110];
    let mut vals = [0.0f32; 256];

    for blk in 0..num_blocks {
        let base = blk * 256;
        for i in 0..256 {
            let off = (base + i) * 4;
            vals[i] = f32::from_le_bytes([
                f32_bytes[off], f32_bytes[off + 1],
                f32_bytes[off + 2], f32_bytes[off + 3],
            ]);
        }

        // 1) Per sub-block float scale.
        let mut sub_scale = [0.0f32; 16];
        let mut max_abs_scale = 0.0f32;
        for sb in 0..16 {
            let s = make_qx_quants_q3(&vals[sb * 16..(sb + 1) * 16]);
            sub_scale[sb] = s;
            let a = s.abs();
            if a > max_abs_scale { max_abs_scale = a; }
        }

        // 2) Super-block iscale (GGML Q3_K sign convention).
        let (super_d, iscale) = if max_abs_scale > 0.0 {
            (-max_abs_scale / 32.0, -32.0 / max_abs_scale)
        } else {
            (0.0, 0.0)
        };

        // 3) Quantize sub-block scales to int [-32, 31], store unsigned 0..63.
        let mut sc6 = [0u8; 16];
        let mut int_scale = [0i32; 16];
        for sb in 0..16 {
            let q = ((iscale * sub_scale[sb]).round() as i32).clamp(-32, 31);
            int_scale[sb] = q;
            sc6[sb] = (q + 32) as u8;
        }

        // 4) Per-element quantization.
        let mut q3 = [0u8; 256];
        for sb in 0..16 {
            let eff = super_d * int_scale[sb] as f32;
            let inv_eff = if eff.abs() > 1e-30 { 1.0 / eff } else { 0.0 };
            for k in 0..16 {
                let idx = sb * 16 + k;
                let mut q = (vals[idx] * inv_eff).round() as i32;
                if q < -4 { q = -4; }
                if q > 3 { q = 3; }
                q3[idx] = (q + 4) as u8;
            }
        }

        // 5) Encode into GGML Q3_K block layout. The qs/hmask placement is the
        // exact inverse of `dequantize_q3_k`'s GGML traversal: emit `q3` in the
        // decoder's (group, pass, run, l) order and write each code's 2 low bits
        // + high bit to the position the decoder reads them from:
        //   qs    byte = group*32 + run*16 + l ,  2-bit shift = 2*pass
        //   hmask byte = run*16 + l ,             bit = group*4 + pass
        // (Previously this packed qs/hmask sequentially, matching the OLD linear
        // decoder; that mismatched GGML and only round-tripped against the buggy
        // dequant.)
        let bp = &mut out[blk * 110..(blk + 1) * 110];

        let mut p = 0usize; // sequential q3 index in decoder emit order
        for group in 0..2usize {
            for pass in 0..4usize {
                let shift = (2 * pass) as u8;
                for run in 0..2usize {
                    for l in 0..16usize {
                        let code = q3[p];
                        let qs_byte = group * 32 + run * 16 + l;
                        bp[32 + qs_byte] |= (code & 3) << shift;
                        let hmask_byte = run * 16 + l;
                        let hmask_bit = (group * 4 + pass) as u8;
                        bp[hmask_byte] |= ((code >> 2) & 1) << hmask_bit;
                        p += 1;
                    }
                }
            }
        }

        // scales (12B): exact INVERSE of the GGML aux uint32 decode shuffle
        // (see `dequantize_q3_k`). The decode reconstructs 16 6-bit scales as:
        //   s[0..4]   low4 = b[0..4]&0xF       high2 = (b[8..12]>>0)&3
        //   s[4..8]   low4 = b[4..8]&0xF       high2 = (b[8..12]>>2)&3
        //   s[8..12]  low4 = (b[0..4]>>4)&0xF  high2 = (b[8..12]>>4)&3
        //   s[12..16] low4 = (b[4..8]>>4)&0xF  high2 = (b[8..12]>>6)&3
        // so the inverse pack is:
        let sb_scale_bytes = &mut bp[96..108];
        for k in 0..4 {
            sb_scale_bytes[k]     = (sc6[k]     & 0x0F) | ((sc6[k + 8]  & 0x0F) << 4);
            sb_scale_bytes[4 + k] = (sc6[4 + k] & 0x0F) | ((sc6[12 + k] & 0x0F) << 4);
            sb_scale_bytes[8 + k] = ((sc6[k]      >> 4) & 3)
                | (((sc6[4 + k]  >> 4) & 3) << 2)
                | (((sc6[8 + k]  >> 4) & 3) << 4)
                | (((sc6[12 + k] >> 4) & 3) << 6);
        }

        // f16 d
        let d_f16 = f32_to_f16_bits_convert(super_d);
        bp[108] = d_f16 as u8;
        bp[109] = (d_f16 >> 8) as u8;
    }

    out
}

/// GGML `make_qx_quants` reference (n=16, nmax=4, rmse_type=1): returns the
/// per-block float scale d that maximizes <q_clamped, x>^2 / <q_clamped,
/// q_clamped> where q = round(x / d), clamped to [-4, 3]. Returns d (not 1/d).
#[cfg(test)]
fn make_qx_quants_q3(x: &[f32]) -> f32 {
    const NMAX: i32 = 4;
    let mut amax = 0.0f32;
    let mut max_signed = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax { amax = av; max_signed = v; }
    }
    if amax == 0.0 { return 0.0; }

    // Initial iscale (per the GGML Q3_K reference encoder): -nmax / max(x).
    let iscale_initial = -(NMAX as f32) / max_signed;
    let mut best_score = q3_iscale_score(x, iscale_initial, NMAX);
    let mut best_iscale = iscale_initial;

    // Sweep small perturbations: is in [-9..9] except 0.
    for is in -9..=9i32 {
        if is == 0 { continue; }
        let iscale = -(NMAX as f32 + 0.1f32 * is as f32) / max_signed;
        let s = q3_iscale_score(x, iscale, NMAX);
        if s > best_score {
            best_score = s;
            best_iscale = iscale;
        }
    }
    if best_iscale == 0.0 { 0.0 } else { 1.0 / best_iscale }
}

#[cfg(test)]
#[inline]
fn q3_iscale_score(x: &[f32], iscale: f32, nmax: i32) -> f32 {
    let mut sum_xq = 0.0f32;
    let mut sum_qq = 0.0f32;
    for &v in x {
        let l = ((iscale * v).round() as i32).clamp(-nmax, nmax - 1) as f32;
        sum_xq += v * l;
        sum_qq += l * l;
    }
    if sum_qq > 0.0 { sum_xq * sum_xq / sum_qq } else { 0.0 }
}

#[cfg(test)]
mod q3_k_encoder_tests {
    use super::*;

    fn vec_to_f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
        out
    }

    fn dequant_to_vec(q3_bytes: &[u8], n: usize) -> Vec<f32> {
        let dq = dequantize_q3_k(q3_bytes, n as u64);
        (0..n).map(|i| f32::from_le_bytes([
            dq[i*4], dq[i*4+1], dq[i*4+2], dq[i*4+3]
        ])).collect()
    }

    #[test]
    fn q3_k_zero_input_roundtrips_zero() {
        let n = 256;
        let zeros = vec![0.0f32; n];
        let q3 = quantize_f32_to_q3_k(&vec_to_f32_bytes(&zeros), n);
        assert_eq!(q3.len(), 110);
        let rt = dequant_to_vec(&q3, n);
        for &v in &rt { assert_eq!(v, 0.0); }
    }

    #[test]
    fn q3_k_block_size_is_110_bytes() {
        let n = 256;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let q3 = quantize_f32_to_q3_k(&vec_to_f32_bytes(&v), n);
        assert_eq!(q3.len(), 110, "1 super-block must be exactly 110 bytes");
    }

    #[test]
    fn q3_k_multi_block_size() {
        let n = 1024; // 4 super-blocks
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let q3 = quantize_f32_to_q3_k(&vec_to_f32_bytes(&v), n);
        assert_eq!(q3.len(), 4 * 110, "4 super-blocks must be 440 bytes");
    }

    #[test]
    fn q3_k_snr_on_smooth_signal_meets_floor() {
        // Smooth input is easier than i.i.d. Gaussian. We expect Q3_K to
        // achieve >= 20 dB SNR here; this guards the encoder from gross bugs.
        let n = 4096;
        let v: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / n as f32;
            (t * 8.0 * std::f32::consts::PI).sin() * 1.5
        }).collect();
        let q3 = quantize_f32_to_q3_k(&vec_to_f32_bytes(&v), n);
        let rt = dequant_to_vec(&q3, n);
        let mut sig = 0.0f32;
        let mut err = 0.0f32;
        for i in 0..n {
            sig += v[i] * v[i];
            let e = v[i] - rt[i];
            err += e * e;
        }
        let snr_db = 10.0 * (sig / err).log10();
        assert!(
            snr_db >= 20.0,
            "Q3_K SNR on smooth signal: {:.2} dB (must be >= 20 dB)",
            snr_db,
        );
    }
}
