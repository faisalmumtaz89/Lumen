//! GPU-resident layer weight buffers for the CUDA backend.
//!
//! Each transformer layer's weight tensors are uploaded to GPU memory once
//! (during weight preloading or on first use) and cached for the lifetime
//! of the inference session. This avoids per-token host-to-device transfers.
//!
//! Supports F32, F16, Q8_0, and Q4_0 quantization. F16 weights are uploaded
//! as raw bytes (2 bytes per element) and dispatched via cuBLAS HGEMM for
//! half-bandwidth matvec. Quantized weights are uploaded as raw bytes and
//! dequantized on-the-fly by the GPU kernel. Norm weights (attn_norm, ffn_norm)
//! are always F32 regardless of model quantization.

use crate::error::RuntimeError;
use crate::weight::cache::LayerView;
use cudarc::driver::CudaSlice;
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;

use super::ffi::CudaDevice;

/// A weight buffer on GPU: F32, F16 raw bytes, Q8_0 raw bytes, or Q4_0 raw bytes.
///
/// The dispatch logic in `backend_impl` inspects this enum to select the
/// correct dispatch path: cuBLAS SGEMV for F32, cuBLAS HGEMM for F16,
/// or custom NVRTC kernels for quantized formats.
pub enum GpuWeightBuf {
    /// Unquantized F32 weights, dispatched via cuBLAS SGEMV.
    F32(CudaSlice<f32>),
    /// IEEE 754 half-precision weights as raw bytes (2 bytes per element).
    /// Dispatched via custom `matvec_f16` kernel (dequant f16->f32 on the fly).
    F16Raw(CudaSlice<u8>),
    /// Raw Q8_0 bytes on GPU (34 bytes per block of 32 elements).
    /// Dequantized on-the-fly by `matvec_q8_0`.
    Q8Raw(CudaSlice<u8>),
    /// Repacked Q8_0 with 36-byte aligned blocks (2B scale + 2B pad + 32B quants).
    /// Quant data at offset +4 is 4-byte aligned, enabling native int* loads
    /// in the dp4a kernel (8 instructions vs 56 byte-level ops per block).
    /// Created by `repack_q8_to_aligned()` during `preload_weights()`.
    Q8Aligned(CudaSlice<u8>),
    /// Raw Q4_0 bytes on GPU (18 bytes per block of 32 elements).
    /// Dequantized on-the-fly by `matvec_q4_0`.
    Q4Raw(CudaSlice<u8>),
    /// Repacked Q4_0 with 20-byte aligned blocks (2B scale + 2B pad + 16B nibbles).
    /// Nibble data at offset +4 is 4-byte aligned, enabling native int* loads
    /// in the dp4a kernel (4 int loads vs 16 byte loads per block).
    /// Created by `repack_q4_to_aligned()` during `preload_weights()`.
    Q4Aligned(CudaSlice<u8>),
}

/// Per-layer weight buffers resident on GPU.
///
/// Projection weight fields hold `GpuWeightBuf` which may be F32, Q8_0, or Q4_0.
/// Norm weights are always F32 (even in quantized models, norms stay F32).
///
/// Shapes (row-major):
/// - `wq`: `[num_heads * head_dim, hidden_dim]`
/// - `wk`: `[num_kv_heads * head_dim, hidden_dim]`
/// - `wv`: `[num_kv_heads * head_dim, hidden_dim]`
/// - `wo`: `[hidden_dim, num_heads * head_dim]`
/// - `attn_norm`: `[hidden_dim]`
/// - `ffn_norm`: `[hidden_dim]`
/// - `w_gate`: `[inter_dim, hidden_dim]`
/// - `w_up`: `[inter_dim, hidden_dim]`
/// - `w_down`: `[hidden_dim, inter_dim]`
pub struct LayerWeightsGpu {
    pub wq: GpuWeightBuf,
    pub wk: GpuWeightBuf,
    pub wv: GpuWeightBuf,
    pub wo: GpuWeightBuf,
    pub attn_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub w_gate: GpuWeightBuf,
    pub w_up: GpuWeightBuf,
    pub w_down: GpuWeightBuf,
    /// Pre-dequanted F16 caches for cuBLAS HGEMM prefill (tensor core path).
    /// Populated by `dequant_layer_q8_to_f16()` after weight upload.
    /// Each cache stores [out_dim * in_dim * 2] bytes (f16 per element).
    /// None for weights that are already F32 or F16 (use cuBLAS directly).
    pub wq_f16: Option<CudaSlice<u8>>,
    pub wk_f16: Option<CudaSlice<u8>>,
    pub wv_f16: Option<CudaSlice<u8>>,
    pub wo_f16: Option<CudaSlice<u8>>,
    pub w_gate_f16: Option<CudaSlice<u8>>,
    pub w_up_f16: Option<CudaSlice<u8>>,
    pub w_down_f16: Option<CudaSlice<u8>>,

    // --- GDN-specific weights (None for standard attention layers) ---
    /// Layer type: 0 = softmax attention (default), 1 = GDN.
    pub layer_type: u8,
    /// SSM Conv1D weight: [conv_dim * kernel_size] F32.
    pub ssm_conv1d: Option<CudaSlice<f32>>,
    /// SSM dt_bias: [num_heads] F32.
    pub ssm_dt_bias: Option<CudaSlice<f32>>,
    /// SSM A (decay): [num_heads] F32 (stores -exp(A_log)).
    pub ssm_a: Option<CudaSlice<f32>>,
    /// SSM alpha projection weight.
    pub ssm_alpha: Option<GpuWeightBuf>,
    /// SSM beta projection weight.
    pub ssm_beta: Option<GpuWeightBuf>,
    /// SSM norm weight (tiled from [head_dim] to [value_dim]).
    pub ssm_norm_tiled: Option<CudaSlice<f32>>,
    /// SSM output projection weight.
    pub ssm_out: Option<GpuWeightBuf>,
    /// Attention gate weight (for GDN gated output).
    pub attn_gate: Option<GpuWeightBuf>,

    // --- GDN-specific F16 caches (for HGEMM prefill on tensor cores) ---
    /// F16 cache for SSM output projection (None if not Q8_0/Q4_0 or OOM).
    pub ssm_out_f16: Option<CudaSlice<u8>>,
    /// F16 cache for attention gate projection (None if not Q8_0/Q4_0 or OOM).
    pub attn_gate_f16: Option<CudaSlice<u8>>,
    /// F16 cache for SSM alpha projection (None if not Q8_0/Q4_0 or OOM).
    pub ssm_alpha_f16: Option<CudaSlice<u8>>,
    /// F16 cache for SSM beta projection (None if not Q8_0/Q4_0 or OOM).
    pub ssm_beta_f16: Option<CudaSlice<u8>>,
}

/// Reinterpret raw LE bytes as an f32 slice.
///
/// Only valid on little-endian platforms with F32 quantization. The bytes must
/// have length divisible by 4 and must represent contiguous LE f32 values.
#[cfg(target_endian = "little")]
fn bytes_as_f32(bytes: &[u8]) -> Result<&[f32], RuntimeError> {
    if bytes.len() % 4 != 0 {
        return Err(RuntimeError::Compute(format!(
            "F32 weight bytes not 4-byte aligned: {} bytes",
            bytes.len(),
        )));
    }
    if bytes.is_empty() {
        return Ok(&[]);
    }
    let ptr = bytes.as_ptr();
    // Global allocators guarantee at least pointer-size alignment (>= 4 on all
    // platforms), and mmap returns page-aligned memory. Debug-assert to catch
    // exotic edge cases. Skipped for empty slices whose pointer is undefined.
    debug_assert_eq!(
        ptr.align_offset(std::mem::align_of::<f32>()),
        0,
        "weight bytes not 4-byte aligned"
    );
    // SAFETY: bytes is contiguous LE f32 data on a LE platform. Alignment
    // verified above. Length is bytes.len() / 4 elements.
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const f32, bytes.len() / 4) })
}

#[cfg(target_endian = "big")]
fn bytes_as_f32(bytes: &[u8]) -> Result<&[f32], RuntimeError> {
    // Big-endian platforms would need byte-swapping. Not supported yet.
    Err(RuntimeError::Compute(
        "F32 weight loading requires little-endian platform".into(),
    ))
}

/// Estimate the number of elements from raw byte length and quant scheme.
///
/// Returns 0 if the scheme is unknown or the byte length doesn't make sense.
fn estimate_quant_elements(byte_len: usize, scheme: QuantScheme) -> usize {
    // (block_size_elements, block_size_bytes) for each K-quant scheme.
    let (bs_elem, bs_bytes) = match scheme {
        QuantScheme::Q4_K => (256, 144),
        QuantScheme::Q5_K => (256, 176),
        QuantScheme::Q6_K => (256, 210),
        QuantScheme::Q2_K => (256, 84),
        QuantScheme::Q3_K => (256, 110),
        _ => return 0,
    };
    let n_blocks = byte_len / bs_bytes;
    n_blocks * bs_elem
}

/// Decode K-quant scales from 12 packed bytes into 8 scale + 8 min arrays.
///
/// Used by Q4_K and Q5_K. The 12 bytes encode 8 6-bit scales and 8 6-bit mins
/// in the GGML packed layout: low 6 bits from bytes 0..7, high 2 bits from bytes 8..11.
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
/// Supports all K-quant schemes stored in LBC files: Q6_K, Q4_K, Q5_K, Q2_K,
/// Q3_K. Mixed-quant GGUFs (e.g. bartowski/mradermacher Q4_0 with imatrix)
/// commonly use Q5_K or Q6_K for sensitive per-layer tensors alongside Q4_0.
///
/// All implementations match the GGML reference layout exactly (same as
/// lumen-convert::dequant).
fn dequant_kquant_to_f32(
    raw: &[u8],
    scheme: QuantScheme,
    n_elements: usize,
    f16_to_f32: fn(u16) -> f32,
) -> Result<Vec<f32>, RuntimeError> {
    let mut out = vec![0.0f32; n_elements];

    match scheme {
        QuantScheme::Q6_K => {
            // Q6_K: 256 elements per block, 210 bytes per block.
            // Layout: [128B ql, 64B qh, 16B scales, 2B f16_d]
            let block_size = 210;
            let n_blocks = raw.len() / block_size;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_size..];
                let ql = &bp[0..128];
                let qh = &bp[128..192];
                let scales = &bp[192..208];
                let d_bits = u16::from_le_bytes([bp[208], bp[209]]);
                let d = f16_to_f32(d_bits);

                let mut idx = 0usize;
                for half in 0..2usize {
                    let ql_ptr = &ql[64 * half..];
                    let qh_ptr = &qh[32 * half..];
                    let sc_ptr = &scales[8 * half..];

                    // Group 0: low nibbles of ql[0..32], qh bits [0..1]
                    for j in 0..32 {
                        if written + idx >= n_elements { break; }
                        let q_lo = ql_ptr[j] & 0x0F;
                        let q_hi = (qh_ptr[j] & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 1: high nibbles of ql[0..32], qh bits [2..3]
                    for j in 0..32 {
                        if written + idx >= n_elements { break; }
                        let q_lo = (ql_ptr[j] >> 4) & 0x0F;
                        let q_hi = ((qh_ptr[j] >> 2) & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[2 + j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 2: low nibbles of ql[32..64], qh bits [4..5]
                    for j in 0..32 {
                        if written + idx >= n_elements { break; }
                        let q_lo = ql_ptr[32 + j] & 0x0F;
                        let q_hi = ((qh_ptr[j] >> 4) & 3) << 4;
                        let q = (q_lo | q_hi) as i32 - 32;
                        let sc = sc_ptr[4 + j / 16] as i8 as f32;
                        out[written + idx] = d * sc * q as f32;
                        idx += 1;
                    }
                    // Group 3: high nibbles of ql[32..64], qh bits [6..7]
                    for j in 0..32 {
                        if written + idx >= n_elements { break; }
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
            let block_size = 144;
            let n_blocks = raw.len() / block_size;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_size..];
                let d = f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
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
                        if written >= n_elements { break; }
                        out[written] = d1 * (qs[qs_offset + l] & 0x0F) as f32 - m1;
                        written += 1;
                    }
                    // Second 32 values: high nibbles
                    for l in 0..32 {
                        if written >= n_elements { break; }
                        out[written] = d2 * ((qs[qs_offset + l] >> 4) & 0x0F) as f32 - m2;
                        written += 1;
                    }
                }
            }
        }
        QuantScheme::Q5_K => {
            // Q5_K: 256 elements per block, 176 bytes per block.
            // Layout: [2B f16 d, 2B f16 dmin, 12B scales, 32B qh, 128B qs]
            let block_size = 176;
            let n_blocks = raw.len() / block_size;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_size..];
                let d = f16_to_f32(u16::from_le_bytes([bp[0], bp[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([bp[2], bp[3]]));
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
                        if written >= n_elements { break; }
                        let h_bit = (qh[l] >> u1) & 1;
                        out[written] = d1 * ((qs[qs_offset + l] & 0x0F) | (h_bit << 4)) as f32 - m1;
                        written += 1;
                    }
                    // Second 32 values: high nibbles + high bit
                    for l in 0..32 {
                        if written >= n_elements { break; }
                        let h_bit = (qh[l] >> u2) & 1;
                        out[written] = d2 * (((qs[qs_offset + l] >> 4) & 0x0F) | (h_bit << 4)) as f32 - m2;
                        written += 1;
                    }
                }
            }
        }
        QuantScheme::Q2_K => {
            // Q2_K: 256 elements per block, 84 bytes per block.
            // Layout: [16B scales, 64B qs, 2B f16 d, 2B f16 dmin]
            let block_size = 84;
            let n_blocks = raw.len() / block_size;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_size..];
                let scales = &bp[0..16];
                let qs = &bp[16..80];
                let d = f16_to_f32(u16::from_le_bytes([bp[80], bp[81]]));
                let dmin = f16_to_f32(u16::from_le_bytes([bp[82], bp[83]]));

                // 16 sub-blocks of 16 values = 256 values
                let mut q_idx = 0usize;
                for &scale_byte in scales.iter().take(16) {
                    let sc = d * (scale_byte & 0x0F) as f32;
                    let m = dmin * ((scale_byte >> 4) & 0x0F) as f32;
                    // 16 values, 2-bit each = 4 bytes
                    for _ in 0..4 {
                        if written >= n_elements { break; }
                        let byte = qs[q_idx];
                        q_idx += 1;
                        for shift in [0, 2, 4, 6] {
                            if written >= n_elements { break; }
                            let q = ((byte >> shift) & 3) as f32;
                            out[written] = sc * q - m;
                            written += 1;
                        }
                    }
                }
            }
        }
        QuantScheme::Q3_K => {
            // Q3_K: 256 elements per block, 110 bytes per block.
            // Layout: [32B hmask, 64B qs (2-bit low), 12B scales (6-bit packed), 2B f16 d]
            let block_size = 110;
            let n_blocks = raw.len() / block_size;
            let mut written = 0usize;
            for b in 0..n_blocks {
                let bp = &raw[b * block_size..];
                let hmask = &bp[0..32];
                let qs = &bp[32..96];
                let scale_bytes = &bp[96..108];
                let d = f16_to_f32(u16::from_le_bytes([bp[108], bp[109]]));

                // Decode 16 6-bit scales from 12 bytes (GGML packed layout)
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

                // Process 256 values in 16 sub-blocks of 16
                let mut q_byte_idx = 0usize;
                let mut q_bit_shift = 0u8;
                let mut val_idx = 0usize;
                for &sc_val in &sc_arr {
                    let scale = d * (sc_val as i8 as f32 - 32.0);
                    for _ in 0..16 {
                        if written >= n_elements { break; }
                        let q_lo = (qs[q_byte_idx] >> q_bit_shift) & 3;
                        q_bit_shift += 2;
                        if q_bit_shift >= 8 {
                            q_bit_shift = 0;
                            q_byte_idx += 1;
                        }
                        let hmask_byte = val_idx / 8;
                        let hmask_bit = val_idx % 8;
                        let h = (hmask[hmask_byte] >> hmask_bit) & 1;
                        let q = (q_lo | (h << 2)) as i32 - 4;
                        out[written] = scale * q as f32;
                        written += 1;
                        val_idx += 1;
                    }
                }
            }
        }
        other => {
            return Err(RuntimeError::Compute(format!(
                "CUDA weight upload: {other:?} dequant not implemented. \
                 Re-convert the model with --requant q4_0 to convert K-quant \
                 tensors to a supported format.",
            )));
        }
    }

    Ok(out)
}

/// Upload a single tensor to GPU as the appropriate format based on its quantization.
///
/// - `QuantScheme::F32`: reinterpret bytes as f32, upload as `GpuWeightBuf::F32`.
/// - `QuantScheme::F16`: upload raw bytes as `GpuWeightBuf::F16Raw`.
/// - `QuantScheme::Q8_0`: upload raw bytes as `GpuWeightBuf::Q8Raw`.
/// - `QuantScheme::Q4_0`: upload raw bytes as `GpuWeightBuf::Q4Raw`.
/// - `QuantScheme::Q4_1`, `Q5_0`, `Bf16`: dequant to F32 on host.
/// - K-quants (`Q6_K`, `Q5_K`, `Q4_K`, `Q3_K`, `Q2_K`): dequant to F32 on host.
///   The F32 buffer gets an F16 cache via `dequant_layer_q8_to_f16()` for HGEMV.
fn upload_tensor(
    device: &CudaDevice,
    weights: &LayerView,
    name: &str,
    slice: &lumen_format::index::TensorSlice,
) -> Result<GpuWeightBuf, RuntimeError> {
    let raw = weights.subtensor_bytes(slice)?;
    match slice.quant {
        QuantScheme::F32 => {
            let f32_data = bytes_as_f32(raw)?;
            let gpu_buf = device.htod_copy(f32_data)?;
            Ok(GpuWeightBuf::F32(gpu_buf))
        }
        QuantScheme::F16 => {
            // Upload raw F16 bytes (2 bytes per element). The custom matvec_f16
            // kernel dequantizes f16→f32 on the fly.
            let gpu_buf = device.htod_copy(raw)?;
            Ok(GpuWeightBuf::F16Raw(gpu_buf))
        }
        QuantScheme::Q8_0 => {
            let gpu_buf = device.htod_copy(raw)?;
            Ok(GpuWeightBuf::Q8Raw(gpu_buf))
        }
        QuantScheme::Q4_0 => {
            let gpu_buf = device.htod_copy(raw)?;
            Ok(GpuWeightBuf::Q4Raw(gpu_buf))
        }
        QuantScheme::Q4_1 => {
            // Q4_1 is used for some tensors in "Q4_0" GGUFs (e.g. w_down).
            // Dequantize to F32 on host and upload as F32.
            // Q4_1 block: 20 bytes = f16 scale (2B) + f16 min (2B) + 16 nibble-pair bytes (32 elements).
            let n_blocks = raw.len() / 20;
            let n_elements = n_blocks * 32;
            let mut f32_data = vec![0.0f32; n_elements];

            // Inline f16→f32 conversion (no half crate dependency needed)
            fn f16_to_f32(bits: u16) -> f32 {
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1f) as u32;
                let frac = (bits & 0x3ff) as u32;
                if exp == 0 {
                    if frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
                    let v = (frac as f32) * 6.103515625e-05 / 1024.0;
                    return if sign == 1 { -v } else { v };
                }
                if exp == 31 {
                    return if frac != 0 { f32::NAN } else if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
                }
                let f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
                f32::from_bits(f32_bits)
            }

            for b in 0..n_blocks {
                let bp = &raw[b * 20..];
                let scale = f16_to_f32((bp[0] as u16) | ((bp[1] as u16) << 8));
                let min = f16_to_f32((bp[2] as u16) | ((bp[3] as u16) << 8));
                for i in 0..16 {
                    let byte = bp[4 + i];
                    let lo = (byte & 0x0F) as f32;
                    let hi = ((byte >> 4) & 0x0F) as f32;
                    f32_data[b * 32 + 2 * i] = scale * lo + min;
                    f32_data[b * 32 + 2 * i + 1] = scale * hi + min;
                }
            }
            let gpu_buf = device.htod_copy(&f32_data)?;
            Ok(GpuWeightBuf::F32(gpu_buf))
        }
        QuantScheme::Q5_0 => {
            // Q5_0: 22 bytes per block of 32 elements.
            // Layout: f16 scale (2B) + 4B high-bits + 16B low-nibbles.
            let n_blocks = raw.len() / 22;
            let n_elements = n_blocks * 32;
            let mut f32_data = vec![0.0f32; n_elements];

            fn f16_to_f32_q5(bits: u16) -> f32 {
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1f) as u32;
                let frac = (bits & 0x3ff) as u32;
                if exp == 0 {
                    if frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
                    let v = (frac as f32) * 6.103515625e-05 / 1024.0;
                    return if sign == 1 { -v } else { v };
                }
                if exp == 31 {
                    return if frac != 0 { f32::NAN } else if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
                }
                f32::from_bits((sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13))
            }

            for b in 0..n_blocks {
                let bp = &raw[b * 22..];
                let scale = f16_to_f32_q5((bp[0] as u16) | ((bp[1] as u16) << 8));
                // 4 bytes of high bits (1 bit per element, packed as u32)
                let qh = (bp[2] as u32) | ((bp[3] as u32) << 8)
                    | ((bp[4] as u32) << 16) | ((bp[5] as u32) << 24);
                let qs = &bp[6..]; // 16 nibble-pair bytes
                for i in 0..16 {
                    let byte = qs[i];
                    let lo_nibble = (byte & 0x0F) as u32;
                    let hi_nibble = ((byte >> 4) & 0x0F) as u32;
                    let lo_hi = (qh >> (2 * i)) & 1;
                    let hi_hi = (qh >> (2 * i + 1)) & 1;
                    let lo_val = (lo_nibble | (lo_hi << 4)) as f32 - 16.0;
                    let hi_val = (hi_nibble | (hi_hi << 4)) as f32 - 16.0;
                    f32_data[b * 32 + 2 * i] = scale * lo_val;
                    f32_data[b * 32 + 2 * i + 1] = scale * hi_val;
                }
            }
            eprintln!("[CUDA] upload {name}: Q5_0 dequant to F32 ({n_elements} elements)");
            let gpu_buf = device.htod_copy(&f32_data)?;
            Ok(GpuWeightBuf::F32(gpu_buf))
        }
        QuantScheme::Bf16 => {
            // BF16: 2 bytes per element. Dequantize to F32 on host.
            let n_elements = raw.len() / 2;
            let mut f32_data = vec![0.0f32; n_elements];
            for i in 0..n_elements {
                let bits = (raw[i * 2] as u16) | ((raw[i * 2 + 1] as u16) << 8);
                // BF16 -> F32: just shift left by 16 bits.
                f32_data[i] = f32::from_bits((bits as u32) << 16);
            }
            eprintln!("[CUDA] upload {name}: BF16 dequant to F32 ({n_elements} elements)");
            let gpu_buf = device.htod_copy(&f32_data)?;
            Ok(GpuWeightBuf::F32(gpu_buf))
        }
        other => {
            // Catch-all for K-quant and other unsupported quant schemes:
            // dequantize to F32 on host and upload as F32. This handles Q4_K,
            // Q5_K, Q6_K, Q2_K, Q3_K, and any future schemes the converter emits.
            // The F32 buffer will get an F16 cache via dequant_layer_q8_to_f16(),
            // so decode uses the fast HGEMV path (not the slow scalar matvec).
            let n_elements = estimate_quant_elements(raw.len(), other);
            if n_elements == 0 {
                return Err(RuntimeError::Compute(format!(
                    "CUDA weight upload: {name} uses {other:?} with {len} bytes, \
                     cannot determine element count",
                    len = raw.len(),
                )));
            }

            fn f16_to_f32_generic(bits: u16) -> f32 {
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1f) as u32;
                let frac = (bits & 0x3ff) as u32;
                if exp == 0 {
                    if frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
                    let v = (frac as f32) * 6.103515625e-05 / 1024.0;
                    return if sign == 1 { -v } else { v };
                }
                if exp == 31 {
                    return if frac != 0 { f32::NAN } else if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
                }
                f32::from_bits((sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13))
            }

            let f32_data = dequant_kquant_to_f32(raw, other, n_elements, f16_to_f32_generic)?;
            eprintln!("[CUDA] upload {name}: {other:?} dequant to F32 ({n_elements} elements)");
            let gpu_buf = device.htod_copy(&f32_data)?;
            Ok(GpuWeightBuf::F32(gpu_buf))
        }
    }
}

/// Upload a norm tensor to GPU as F32.
///
/// Norm weights (attn_norm, ffn_norm) are always stored as F32, even in
/// quantized models. Returns an error if the norm tensor is not F32.
fn upload_norm_tensor(
    device: &CudaDevice,
    weights: &LayerView,
    name: &str,
    slice: &lumen_format::index::TensorSlice,
) -> Result<CudaSlice<f32>, RuntimeError> {
    if slice.quant != QuantScheme::F32 {
        return Err(RuntimeError::Compute(format!(
            "CUDA norm upload: {name} uses {:?}, norms must be F32",
            slice.quant,
        )));
    }
    let raw = weights.subtensor_bytes(slice)?;
    let f32_data = bytes_as_f32(raw)?;
    device.htod_copy(f32_data)
}

/// Upload a single layer's weight tensors from a `LayerView` to GPU memory.
///
/// Extracts each subtensor's raw bytes and uploads according to quant scheme:
/// - F32 tensors are reinterpreted and uploaded as f32 buffers.
/// - F16 tensors are reinterpreted and uploaded as half::f16 buffers.
/// - Q8_0 tensors are uploaded as raw byte buffers (dequantized by GPU kernels).
/// - Q4_0 tensors are uploaded as raw byte buffers (dequantized by GPU kernels).
/// - Norm weights are always F32.
///
/// Returns an error if any tensor uses an unsupported scheme or if CUDA upload fails.
pub fn upload_layer_weights(
    device: &CudaDevice,
    weights: &LayerView,
    _hp: &ModelHyperparams,
) -> Result<LayerWeightsGpu, RuntimeError> {
    let subs = &weights.subtensors;

    Ok(LayerWeightsGpu {
        wq: upload_tensor(device, weights, "wq", &subs.wq)?,
        wk: upload_tensor(device, weights, "wk", &subs.wk)?,
        wv: upload_tensor(device, weights, "wv", &subs.wv)?,
        wo: upload_tensor(device, weights, "wo", &subs.wo)?,
        attn_norm: upload_norm_tensor(device, weights, "attn_norm", &subs.attn_norm)?,
        // Prefer attn_post_norm when ffn_norm sentinel is absent (length=0).
        // Qwen3.5 GDN layers use post_attention_norm as the FFN pre-norm;
        // ffn_norm is left as a zero-sentinel (offset=0, length=0) in the LBC.
        ffn_norm: {
            let ffn_norm_slice = if subs.ffn_norm.length == 0 {
                subs.attn_post_norm.as_ref().unwrap_or(&subs.ffn_norm)
            } else {
                &subs.ffn_norm
            };
            upload_norm_tensor(device, weights, "ffn_norm", ffn_norm_slice)?
        },
        w_gate: upload_tensor(device, weights, "w_gate", &subs.w_gate)?,
        w_up: upload_tensor(device, weights, "w_up", &subs.w_up)?,
        w_down: upload_tensor(device, weights, "w_down", &subs.w_down)?,
        // F16 caches start as None; populated by dequant_layer_q8_to_f16().
        wq_f16: None,
        wk_f16: None,
        wv_f16: None,
        wo_f16: None,
        w_gate_f16: None,
        w_up_f16: None,
        w_down_f16: None,
        // GDN fields: populated only for GDN layers (layer_type == 1).
        layer_type: subs.layer_type.unwrap_or(0),
        ssm_conv1d: match &subs.ssm_conv1d {
            Some(s) if s.quant == QuantScheme::F32 => {
                let raw = weights.subtensor_bytes(s)?;
                Some(device.htod_copy(bytes_as_f32(raw)?)?)
            }
            _ => None,
        },
        ssm_dt_bias: match &subs.ssm_dt {
            Some(s) if s.quant == QuantScheme::F32 => {
                let raw = weights.subtensor_bytes(s)?;
                Some(device.htod_copy(bytes_as_f32(raw)?)?)
            }
            _ => None,
        },
        ssm_a: match &subs.ssm_a {
            Some(s) if s.quant == QuantScheme::F32 => {
                let raw = weights.subtensor_bytes(s)?;
                Some(device.htod_copy(bytes_as_f32(raw)?)?)
            }
            _ => None,
        },
        ssm_alpha: match &subs.ssm_alpha {
            Some(s) => Some(upload_tensor(device, weights, "ssm_alpha", s)?),
            None => None,
        },
        ssm_beta: match &subs.ssm_beta {
            Some(s) => Some(upload_tensor(device, weights, "ssm_beta", s)?),
            None => None,
        },
        ssm_norm_tiled: match &subs.ssm_norm {
            Some(s) if s.quant == QuantScheme::F32 => {
                let raw = weights.subtensor_bytes(s)?;
                let norm_f32 = bytes_as_f32(raw)?;
                let head_dim = norm_f32.len();
                // Tile from [head_dim] to [value_dim = num_heads * head_dim]
                let num_heads = 16 * 2; // GDN: group_count=16 * GQA ratio 2 = 32 (NOT model's num_kv_heads)
                let value_dim = num_heads * head_dim;
                let mut tiled = vec![0.0f32; value_dim];
                for h in 0..num_heads {
                    tiled[h * head_dim..(h + 1) * head_dim].copy_from_slice(norm_f32);
                }
                Some(device.htod_copy(&tiled)?)
            }
            _ => None,
        },
        ssm_out: match &subs.ssm_out {
            Some(s) => Some(upload_tensor(device, weights, "ssm_out", s)?),
            None => None,
        },
        attn_gate: match &subs.attn_gate {
            Some(s) => Some(upload_tensor(device, weights, "attn_gate", s)?),
            None => None,
        },
        // GDN F16 caches start as None; populated by dequant_layer_q8_to_f16().
        ssm_out_f16: None,
        attn_gate_f16: None,
        ssm_alpha_f16: None,
        ssm_beta_f16: None,
    })
}

/// Dequantize a Q8_0 GPU buffer to F16 using the `dequant_q8_0_to_f16` kernel.
///
/// Returns an F16 buffer of `num_elements * 2` bytes. The kernel is dispatched
/// once per weight during `preload_weights()`, making the cost amortized over
/// all subsequent prefill calls (which use cuBLAS HGEMM on the cached F16 data).
pub fn dequant_q8_to_f16_gpu(
    device: &CudaDevice,
    kernel: &cudarc::driver::CudaFunction,
    q8_buf: &CudaSlice<u8>,
    num_elements: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    use cudarc::driver::{LaunchConfig as CudarcLaunchConfig, PushKernelArg};

    // Allocate F16 output: 2 bytes per element
    let mut f16_buf: CudaSlice<u8> = device.alloc_zeros(num_elements * 2)?;

    let block_size = 256u32;
    let grid_size = ((num_elements as u32) + block_size - 1) / block_size;
    let num_elems_u32 = num_elements as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(kernel)
            .arg(q8_buf)
            .arg(&mut f16_buf)
            .arg(&num_elems_u32)
            .launch(launch_cfg)
    }
    .map_err(|e| RuntimeError::Compute(format!("dequant_q8_0_to_f16 launch: {e}")))?;

    device.synchronize()?;
    Ok(f16_buf)
}

/// Pre-dequant all Q8_0 projection weights in a layer to F16 for HGEMM.
///
/// Populates the `wX_f16` fields. F32 and F16 weights are skipped (already usable).
/// Pre-dequant all Q8_0 projection weights in a layer to F16 for HGEMM.
///
/// Populates the `wX_f16` fields. F32 and F16 weights are skipped (already usable).
pub fn dequant_layer_q8_to_f16(
    device: &CudaDevice,
    kernel: &cudarc::driver::CudaFunction,
    q4_kernel: &cudarc::driver::CudaFunction,
    kernels: &super::decode::KernelSet,
    layer: &mut LayerWeightsGpu,
    hp: &ModelHyperparams,
) -> Result<(), RuntimeError> {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    // For GDN layers (layer_type == 1):
    // F16 caches are REQUIRED for Q4_0 weights. Without them, Q4_0 falls through
    // to the slow scalar matvec_q4_0 kernel (~22 tok/s vs ~56 tok/s with HGEMV).
    // For Q8_0, the dp4a kernel handles GDN dispatch efficiently, but F16 caches
    // are still beneficial for the FFN block which runs after GDN attention.
    //
    // GDN wq is fused [qkv_dim, hidden_dim], so we compute element counts from
    // actual buffer sizes rather than model hyperparams (which give q_dim, not qkv_dim).
    let is_gdn = layer.layer_type == 1;

    // Process each weight: Q8_0 uses q8 dequant kernel, Q4_0 uses q4 dequant kernel.
    // F16Raw weights are already in the right format for HGEMM -- no dequant needed.
    // For F32 weights (e.g. Q4_1/Q6_K dequanted to F32), create F16 via f32_to_f16_vec.
    let f32_to_f16_fn = &kernels.f32_to_f16_vec;
    let dequant_weight = |w: &GpuWeightBuf, n: usize| -> Result<Option<CudaSlice<u8>>, RuntimeError> {
        match w {
            GpuWeightBuf::Q8Raw(q8) => Ok(Some(dequant_q8_to_f16_gpu(device, kernel, q8, n)?)),
            GpuWeightBuf::Q4Raw(q4) => Ok(Some(dequant_q8_to_f16_gpu(device, q4_kernel, q4, n)?)),
            GpuWeightBuf::F32(f32_buf) => {
                // F32 weights (possibly from Q4_1/Q6_K host dequant) -- convert to F16 for HGEMV.
                use cudarc::driver::PushKernelArg;
                if f32_buf.len() == 0 { return Ok(None); }
                let mut f16_buf: CudaSlice<u8> = device.alloc_zeros(n * 2)?;
                let n_u32 = n as u32;
                let block = 256u32;
                let grid = (n_u32 + block - 1) / block;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    device.stream
                        .launch_builder(f32_to_f16_fn)
                        .arg(f32_buf)
                        .arg(&mut f16_buf)
                        .arg(&n_u32)
                        .launch(cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("f32_to_f16 for HGEMV: {e}")))?;
                device.synchronize()?;
                Ok(Some(f16_buf))
            }
            GpuWeightBuf::Q4Aligned(_) => Ok(None), // Q4Aligned uses dp4a path, no F16 cache needed
            _ => Ok(None), // F16Raw already in the right format for HGEMM -- no dequant needed
        }
    };

    // Helper: compute element count from buffer type and dimensions.
    // For GDN layers, wq is fused [qkv_dim, hidden_dim] so we derive
    // the element count from the buffer byte size instead of model dims.
    let buf_elements = |w: &GpuWeightBuf| -> usize {
        match w {
            GpuWeightBuf::Q8Raw(q8) | GpuWeightBuf::Q8Aligned(q8) => {
                // Q8_0: 34 bytes per block of 32 elements
                (q8.len() / 34) * 32
            }
            GpuWeightBuf::Q4Raw(q4) => {
                // Q4_0: 18 bytes per block of 32 elements
                (q4.len() / 18) * 32
            }
            GpuWeightBuf::Q4Aligned(q4a) => {
                // Q4Aligned: 20 bytes per block of 32 elements
                (q4a.len() / 20) * 32
            }
            GpuWeightBuf::F32(f32_buf) => f32_buf.len(),
            GpuWeightBuf::F16Raw(f16_buf) => f16_buf.len() / 2,
        }
    };

    if is_gdn {
        // GDN layer: skip ALL F16 dequant to avoid OOM on A100-80GB.
        // Qwen3.5-9B Q8_0 weights (~10 GB) + 24 GDN layers × 8 weights × ~100 MB F16 each
        // = ~19 GB F16 caches → exceeds available VRAM after standard layers + CUDA overhead.
        // Attempted 3 times (C26, C27, C29) with graceful OOM — always fails at layer 13-22.
        // GDN GEMMs use dequant+SGEMM fallback. Decode uses dp4a/scalar.
        return Ok(());
    } else {
        // Standard attention layer: use model-dimension-based element counts.
        // These are critical for performance -- OOM here is a hard error.
        layer.wq_f16 = dequant_weight(&layer.wq, q_dim * hidden)?;
        layer.wk_f16 = dequant_weight(&layer.wk, kv_dim * hidden)?;
        layer.wv_f16 = dequant_weight(&layer.wv, kv_dim * hidden)?;
        layer.wo_f16 = dequant_weight(&layer.wo, hidden * q_dim)?;
        layer.w_gate_f16 = dequant_weight(&layer.w_gate, inter * hidden)?;
        layer.w_up_f16 = dequant_weight(&layer.w_up, inter * hidden)?;
        layer.w_down_f16 = dequant_weight(&layer.w_down, hidden * inter)?;
    }

    Ok(())
}

/// Repack a single Q8_0 GPU buffer from 34-byte blocks to 36-byte aligned blocks.
///
/// Runs the `repack_q8_0_to_aligned36` kernel on the GPU. The aligned layout
/// inserts 2 bytes of padding after the f16 scale so quant data starts at offset +4
/// (4-byte aligned), enabling native `int*` loads in the dp4a kernel.
///
/// `num_elements` is the total number of quantized elements (must be a multiple of 32).
/// Returns the new 36-byte-aligned buffer.
pub fn repack_q8_to_aligned(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    q8_buf: &CudaSlice<u8>,
    num_elements: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    use cudarc::driver::{LaunchConfig as CudarcLaunchConfig, PushKernelArg};

    let num_blocks = num_elements / 32;
    // Allocate aligned output: 36 bytes per block
    let mut aligned_buf: CudaSlice<u8> = device.alloc_zeros(num_blocks * 36)?;

    let block_size = 256u32;
    let grid_size = ((num_blocks as u32) + block_size - 1) / block_size;
    let num_blocks_u32 = num_blocks as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(repack_kernel)
            .arg(q8_buf)
            .arg(&mut aligned_buf)
            .arg(&num_blocks_u32)
            .launch(launch_cfg)
    }
    .map_err(|e| RuntimeError::Compute(format!("repack_q8_0_to_aligned36 launch: {e}")))?;

    device.synchronize()?;
    Ok(aligned_buf)
}

/// Repack all Q8_0 projection weights in a layer to 36-byte aligned format.
///
/// Replaces each `Q8Raw` weight with a `Q8Aligned` weight. F32, F16, and Q4_0
/// weights are left untouched. The original Q8_0 buffer is dropped (freed)
/// when replaced.
///
/// This runs once during `preload_weights()`, making the cost amortized over
/// all subsequent decode calls which benefit from native int* loads.
pub fn repack_layer_q8_to_aligned(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layer: &mut LayerWeightsGpu,
    hp: &ModelHyperparams,
) -> Result<(), RuntimeError> {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let is_gdn = layer.layer_type == 1;

    /// Repack a single weight buffer if it is Q8Raw.
    fn repack_weight(
        device: &CudaDevice,
        repack_kernel: &cudarc::driver::CudaFunction,
        w: &mut GpuWeightBuf,
        num_elements: usize,
    ) -> Result<(), RuntimeError> {
        // Take ownership: temporarily replace with a dummy to avoid double-borrow.
        let old = std::mem::replace(w, GpuWeightBuf::F32(device.alloc_zeros(0)?));
        match old {
            GpuWeightBuf::Q8Raw(q8_buf) => {
                let aligned = repack_q8_to_aligned(device, repack_kernel, &q8_buf, num_elements)?;
                *w = GpuWeightBuf::Q8Aligned(aligned);
                // q8_buf is dropped here, freeing the original 34-byte buffer.
                Ok(())
            }
            other => {
                // Not Q8_0 -- put it back.
                *w = other;
                Ok(())
            }
        }
    }

    // For GDN layers: skip QKV repack (wq is fused QKV used by GDN dispatch directly).
    if !is_gdn {
        repack_weight(device, repack_kernel, &mut layer.wq, q_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wk, kv_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wv, kv_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wo, hidden * q_dim)?;
    }
    // FFN weights: skip aligned repack for GDN models to save GPU memory.
    // The F16 HGEMV path handles FFN dispatch via F16 caches instead.
    if !is_gdn {
        repack_weight(device, repack_kernel, &mut layer.w_gate, inter * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.w_up, inter * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.w_down, hidden * inter)?;
    }

    Ok(())
}

/// Repack a single Q4_0 GPU buffer from 18-byte blocks to 20-byte aligned blocks.
///
/// Runs the `repack_q4_0_to_aligned20` kernel on the GPU. The aligned layout
/// inserts 2 bytes of padding after the f16 scale so nibble data starts at offset +4
/// (4-byte aligned), enabling native `int*` loads in the dp4a kernel (4 loads
/// vs 16 byte loads per block).
///
/// `num_elements` is the total number of quantized elements (must be a multiple of 32).
/// Returns the new 20-byte-aligned buffer.
pub fn repack_q4_to_aligned(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    q4_buf: &CudaSlice<u8>,
    num_elements: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    use cudarc::driver::{LaunchConfig as CudarcLaunchConfig, PushKernelArg};

    let num_blocks = num_elements / 32;
    // Allocate aligned output: 20 bytes per block
    let mut aligned_buf: CudaSlice<u8> = device.alloc_zeros(num_blocks * 20)?;

    let block_size = 256u32;
    let grid_size = ((num_blocks as u32) + block_size - 1) / block_size;
    let num_blocks_u32 = num_blocks as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream
            .launch_builder(repack_kernel)
            .arg(q4_buf)
            .arg(&mut aligned_buf)
            .arg(&num_blocks_u32)
            .launch(launch_cfg)
    }
    .map_err(|e| RuntimeError::Compute(format!("repack_q4_0_to_aligned20 launch: {e}")))?;

    device.synchronize()?;
    Ok(aligned_buf)
}

/// Repack all Q4_0 projection weights in a layer to 20-byte aligned format.
///
/// Replaces each `Q4Raw` weight with a `Q4Aligned` weight. Other formats
/// are left untouched. The original Q4_0 buffer is dropped (freed)
/// when replaced.
///
/// This runs once during `preload_weights()`, making the cost amortized over
/// all subsequent decode calls which benefit from aligned int* loads.
pub fn repack_layer_q4_to_aligned(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layer: &mut LayerWeightsGpu,
    hp: &ModelHyperparams,
) -> Result<(), RuntimeError> {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let is_gdn = layer.layer_type == 1;

    /// Repack a single weight buffer if it is Q4Raw.
    fn repack_weight(
        device: &CudaDevice,
        repack_kernel: &cudarc::driver::CudaFunction,
        w: &mut GpuWeightBuf,
        num_elements: usize,
    ) -> Result<(), RuntimeError> {
        let old = std::mem::replace(w, GpuWeightBuf::F32(device.alloc_zeros(0)?));
        match old {
            GpuWeightBuf::Q4Raw(q4_buf) => {
                let aligned = repack_q4_to_aligned(device, repack_kernel, &q4_buf, num_elements)?;
                *w = GpuWeightBuf::Q4Aligned(aligned);
                // q4_buf is dropped here, freeing the original 18-byte buffer.
                Ok(())
            }
            other => {
                // Not Q4_0 -- put it back.
                *w = other;
                Ok(())
            }
        }
    }

    // For GDN layers: skip QKV repack (wq is fused QKV used by GDN dispatch directly).
    if !is_gdn {
        repack_weight(device, repack_kernel, &mut layer.wq, q_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wk, kv_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wv, kv_dim * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.wo, hidden * q_dim)?;
    }
    if !is_gdn {
        repack_weight(device, repack_kernel, &mut layer.w_gate, inter * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.w_up, inter * hidden)?;
        repack_weight(device, repack_kernel, &mut layer.w_down, hidden * inter)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_as_f32_valid() {
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = bytes_as_f32(&bytes).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn bytes_as_f32_misaligned_length() {
        let bytes = vec![0u8; 5]; // Not divisible by 4
        assert!(bytes_as_f32(&bytes).is_err());
    }

    #[test]
    fn bytes_as_f32_empty() {
        let bytes: Vec<u8> = vec![];
        let result = bytes_as_f32(&bytes).unwrap();
        assert_eq!(result.len(), 0);
    }

}
