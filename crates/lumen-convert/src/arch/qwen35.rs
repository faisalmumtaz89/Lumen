//! Qwen3.5 (dense) converter: hybrid GDN + full-attention with dense FFN.

use super::ArchConverter;
use crate::convert::ConvertError;
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufFile};
use crate::tensor_names::*;
use crate::tensor_io::*;
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

use super::qwen35_moe::is_qwen35moe_full_attention_layer;

pub(crate) struct Qwen35Converter;

impl ArchConverter for Qwen35Converter {
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
    ) -> Result<LayerShape, ConvertError> {
        compute_layer_shape_qwen35(gguf, layer, dequantize, requant_to)
    }

    fn write_layer_blob<R: Read + Seek>(
        &self,
        blob: &mut Vec<u8>,
        reader: &mut R,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
    ) -> Result<(), ConvertError> {
        write_qwen35_layer_blob(blob, reader, gguf, layer, dequantize, requant_to)
    }

    fn layer_kind_label(&self, layer: usize) -> String {
        let kind = if is_qwen35moe_full_attention_layer(layer) {
            "full_attn"
        } else {
            "linear_attn"
        };
        format!("{}, dense", kind)
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 (dense) layer shape computation
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a single Qwen3.5 (dense) layer.
///
/// Same hybrid GDN + full-attention architecture as Qwen3.5-MoE, but with
/// dense FFN (ffn_gate/ffn_up/ffn_down) instead of MoE (router + experts + shared expert).
fn compute_layer_shape_qwen35(
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Helper to compute a TensorSlice for a given tensor.
    let compute_slice = |gguf: &GgufFile, name: &str, blob_offset: &mut u64, dequantize: bool|
        -> Result<TensorSlice, ConvertError>
    {
        let tensor = gguf.find_tensor(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_string()))?;
        let is_norm = name.contains("norm");

        // Check if requantization applies
        if let Some(target) = requant_to {
            if is_norm || dequantize {
                // Norms stay F32
                let n_elements = tensor.n_elements();
                let size = n_elements * 4;
                let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
                *blob_offset += size;
                return Ok(slice);
            }
            let src_quant = tensor.ggml_type.to_lbc_quant();
            if src_quant == Some(target) {
                // Already in target format
                let size = tensor.byte_size().unwrap_or(0);
                let slice = TensorSlice { offset: *blob_offset, length: size, quant: target };
                *blob_offset += size;
                return Ok(slice);
            }
            // Compute size for target quant
            let n_elements = tensor.n_elements() as usize;
            assert!(n_elements % 32 == 0,
                "quantization requires elements divisible by 32, got {n_elements} for {name}");
            let (size, quant) = match target {
                QuantScheme::Q8_0 => {
                    // Q8_0: 34 bytes per 32 elements
                    let num_blocks = n_elements / 32;
                    ((num_blocks * 34) as u64, QuantScheme::Q8_0)
                }
                QuantScheme::Q4_0 => {
                    // Q4_0: 18 bytes per 32 elements
                    let num_blocks = n_elements / 32;
                    ((num_blocks * 18) as u64, QuantScheme::Q4_0)
                }
                _ => {
                    // Unsupported target: F32
                    (n_elements as u64 * 4, QuantScheme::F32)
                }
            };
            let slice = TensorSlice { offset: *blob_offset, length: size, quant };
            *blob_offset += size;
            return Ok(slice);
        }

        if dequantize {
            let n_elements = tensor.n_elements();
            let size = n_elements * 4;
            let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
            *blob_offset += size;
            Ok(slice)
        } else {
            let quant = tensor.ggml_type.to_lbc_quant()
                .ok_or_else(|| ConvertError::UnsupportedTensorType {
                    tensor: name.to_string(),
                    ggml_type: format!("{:?}", tensor.ggml_type),
                })?;
            let size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
                tensor: name.to_string(),
                ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
            })?;
            let slice = TensorSlice { offset: *blob_offset, length: size, quant };
            *blob_offset += size;
            Ok(slice)
        }
    };

    // Helper for optional tensors.
    let try_compute_opt_slice = |gguf: &GgufFile, layer: usize, suffix: &str, blob_offset: &mut u64, dequantize: bool|
        -> Result<Option<TensorSlice>, ConvertError>
    {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let force_dequant = !dequantize && tensor.ggml_type.to_lbc_quant().is_none();
            if force_dequant {
                if tensor.ggml_type == GgmlType::MXFP4 {
                    eprintln!("  Note: dequantizing {} ({:?} -> F32)", name, tensor.ggml_type);
                    Ok(Some(compute_slice(gguf, &name, blob_offset, /*dequantize=*/ true)?))
                } else {
                    eprintln!("  Warning: skipping {} (unsupported GGML type {:?})", name, tensor.ggml_type);
                    Ok(None)
                }
            } else {
                Ok(Some(compute_slice(gguf, &name, blob_offset, dequantize)?))
            }
        } else {
            Ok(None)
        }
    };

    // Attention projections: different tensor layout per layer type.
    let (wq, wk, wv, wo);
    if is_full_attn {
        wq = compute_slice(gguf, &layer_tensor_name(layer, ATTN_Q), &mut blob_size, dequantize)?;
        wk = compute_slice(gguf, &layer_tensor_name(layer, ATTN_K), &mut blob_size, dequantize)?;
        wv = compute_slice(gguf, &layer_tensor_name(layer, ATTN_V), &mut blob_size, dequantize)?;
        wo = compute_slice(gguf, &layer_tensor_name(layer, ATTN_OUTPUT), &mut blob_size, dequantize)?;
    } else {
        // Linear attention: fused QKV stored in wq slot; wk/wv/wo left as zero sentinel
        let z = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };
        wq = compute_slice(gguf, &layer_tensor_name(layer, ATTN_QKV), &mut blob_size, dequantize)?;
        wk = z; wv = z; wo = z;
    }

    // Pre-attention norm (always present)
    let attn_norm = compute_slice(gguf, &layer_tensor_name(layer, ATTN_NORM), &mut blob_size, dequantize)?;

    // Post-attention norm (present in all Qwen3.5 layers)
    let attn_post_norm = try_compute_opt_slice(gguf, layer, ATTN_POST_NORM, &mut blob_size, dequantize)?;

    // FFN norm (present in all layers)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    let ffn_norm = if gguf.find_tensor(&ffn_norm_name).is_some() {
        compute_slice(gguf, &ffn_norm_name, &mut blob_size, dequantize)?
    } else {
        TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 }
    };

    // Attention gate (full attention layers only)
    let attn_gate = try_compute_opt_slice(gguf, layer, ATTN_GATE_WEIGHT, &mut blob_size, dequantize)?;

    // SSM tensors (linear attention layers only) — never requantized.
    // ssm_a/dt/conv1d are F32 scalars; ssm_alpha/beta are Q8_0 gate matrices.
    // Bypasses requant_to to preserve original format (GPU kernels expect specific quant).
    // IMPORTANT: ssm_alpha/beta MUST be Q8_0 or F32 — the GDN runtime hardcodes Q8_0 matvec
    // kernels for these tensors. If the source is F16/BF16 (e.g. from a BF16→F16 GGUF),
    // we force-requantize them to Q8_0.
    let try_compute_ssm_slice = |gguf: &GgufFile, layer: usize, suffix: &str, blob_offset: &mut u64|
        -> Result<Option<TensorSlice>, ConvertError>
    {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            if dequantize {
                let n_elements = tensor.n_elements();
                let size = n_elements * 4;
                let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
                *blob_offset += size;
                Ok(Some(slice))
            } else {
                let quant = tensor.ggml_type.to_lbc_quant()
                    .ok_or_else(|| ConvertError::UnsupportedTensorType {
                        tensor: name.to_string(),
                        ggml_type: format!("{:?}", tensor.ggml_type),
                    })?;
                // Force ssm_alpha/beta to Q8_0 if they are F16/BF16 — runtime expects Q8_0.
                let is_alpha_or_beta = suffix == SSM_ALPHA || suffix == SSM_BETA;
                if is_alpha_or_beta && matches!(quant, QuantScheme::F16 | QuantScheme::Bf16) {
                    let n_elements = tensor.n_elements() as usize;
                    assert!(n_elements % 32 == 0,
                        "Q8_0 requires elements divisible by 32, got {n_elements} for {name}");
                    let num_blocks = n_elements / 32;
                    let size = (num_blocks * 34) as u64; // Q8_0: 34 bytes per 32 elements
                    let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::Q8_0 };
                    *blob_offset += size;
                    Ok(Some(slice))
                } else {
                    let size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
                        tensor: name.to_string(),
                        ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                    })?;
                    let slice = TensorSlice { offset: *blob_offset, length: size, quant };
                    *blob_offset += size;
                    Ok(Some(slice))
                }
            }
        } else {
            Ok(None)
        }
    };
    let ssm_a = try_compute_ssm_slice(gguf, layer, SSM_A, &mut blob_size)?;
    let ssm_conv1d = try_compute_ssm_slice(gguf, layer, SSM_CONV1D, &mut blob_size)?;
    let ssm_dt = try_compute_ssm_slice(gguf, layer, SSM_DT, &mut blob_size)?;
    let ssm_beta = try_compute_ssm_slice(gguf, layer, SSM_BETA, &mut blob_size)?;
    let ssm_alpha = try_compute_ssm_slice(gguf, layer, SSM_ALPHA, &mut blob_size)?;
    let ssm_norm = try_compute_ssm_slice(gguf, layer, SSM_NORM, &mut blob_size)?;
    let ssm_out = try_compute_opt_slice(gguf, layer, SSM_OUT, &mut blob_size, requant_to.is_none())?;  // force F32 unless requant handles it

    // Dense FFN weights (present in all layers)
    let w_gate = compute_slice(gguf, &layer_tensor_name(layer, FFN_GATE), &mut blob_size, dequantize)?;
    let w_up = compute_slice(gguf, &layer_tensor_name(layer, FFN_UP), &mut blob_size, dequantize)?;
    let w_down = compute_slice(gguf, &layer_tensor_name(layer, FFN_DOWN), &mut blob_size, dequantize)?;

    // Optional bias tensors
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    // Per-head Q/K RMSNorm weights (full attention layers only, always F32)
    let attn_q_norm = try_compute_opt_slice(gguf, layer, ATTN_Q_NORM, &mut blob_size, /*dequantize=*/ true)?;
    let attn_k_norm = try_compute_opt_slice(gguf, layer, ATTN_K_NORM, &mut blob_size, /*dequantize=*/ true)?;

    let layer_type = if is_full_attn { Some(0u8) } else { Some(1u8) };

    let subtensors = SubtensorOffsets {
        wq, wk, wv, wo,
        bq, bk, bv,
        w_gate, w_up, w_down,
        attn_norm, ffn_norm,
        router_weight: None,
        experts: None,
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate, attn_post_norm,
        ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
        attn_q_norm, attn_k_norm, ffn_gate_inp_shexp: None,
        layer_type,
    };

    Ok(LayerShape {
        blob_size,
        index: LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors,
        },
    })
}

// ---------------------------------------------------------------------------
// Qwen3.5 (dense) layer blob writing
// ---------------------------------------------------------------------------

/// Write a Qwen3.5 (dense) layer blob.
///
/// Same hybrid GDN + full-attention layout as Qwen3.5-MoE, but with dense FFN
/// (ffn_gate/ffn_up/ffn_down) instead of MoE (router + experts + shared expert).
/// Tensor order must match `compute_layer_shape_qwen35()`.
fn write_qwen35_layer_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
) -> Result<(), ConvertError> {
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Attention projections: layout differs by layer type
    if is_full_attn {
        // Full attention: separate Q/K/V/output tensors
        for suffix in &ATTN_TENSOR_SUFFIXES {
            append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, suffix), dequantize, requant_to)?;
        }
    } else {
        // Linear attention: fused QKV tensor only (stored in wq slot in index)
        append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, ATTN_QKV), dequantize, requant_to)?;
    }

    // Pre-attention norm
    append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, ATTN_NORM), dequantize, requant_to)?;

    // Post-attention norm (if present)
    let post_norm_name = layer_tensor_name(layer, ATTN_POST_NORM);
    if gguf.find_tensor(&post_norm_name).is_some() {
        append_tensor_to_blob_requant(blob, reader, gguf, &post_norm_name, dequantize, requant_to)?;
    }

    // FFN norm (if present)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    if gguf.find_tensor(&ffn_norm_name).is_some() {
        append_tensor_to_blob_requant(blob, reader, gguf, &ffn_norm_name, dequantize, requant_to)?;
    }

    // Attention gate (if present)
    let attn_gate_name = layer_tensor_name(layer, ATTN_GATE_WEIGHT);
    if gguf.find_tensor(&attn_gate_name).is_some() {
        append_tensor_to_blob_requant(blob, reader, gguf, &attn_gate_name, dequantize, requant_to)?;
    }

    // SSM tensors (if present) — preserve original precision, never requantize.
    // ssm_a/dt/conv1d are small F32 scalars read as float* in GPU kernels.
    // ssm_alpha/beta MUST be Q8_0 — the GDN runtime hardcodes Q8_0 matvec kernels.
    // If the source is F16/BF16, force-requantize to Q8_0 via dequant→F32→Q8_0.
    for suffix in &[SSM_A, SSM_CONV1D, SSM_DT, SSM_BETA, SSM_ALPHA, SSM_NORM] {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let is_alpha_or_beta = *suffix == SSM_ALPHA || *suffix == SSM_BETA;
            let src_quant = tensor.ggml_type.to_lbc_quant();
            if is_alpha_or_beta && matches!(src_quant, Some(QuantScheme::F16) | Some(QuantScheme::Bf16)) {
                // Force-requantize F16/BF16 alpha/beta to Q8_0
                append_tensor_to_blob_requant(blob, reader, gguf, &name, false, Some(QuantScheme::Q8_0))?;
            } else {
                append_tensor_to_blob_requant(blob, reader, gguf, &name, dequantize, /*requant_to=*/ None)?;
            }
        }
    }
    {
        let name = layer_tensor_name(layer, SSM_OUT);
        if gguf.find_tensor(&name).is_some() {
            // SSM_OUT: requantize to target if specified, else force F32 (Q5_K unsupported)
            append_tensor_to_blob_requant(blob, reader, gguf, &name, requant_to.is_none(), requant_to)?;
        }
    }

    // Dense FFN weights
    append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, FFN_GATE), dequantize, requant_to)?;
    append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, FFN_UP), dequantize, requant_to)?;
    append_tensor_to_blob_requant(blob, reader, gguf, &layer_tensor_name(layer, FFN_DOWN), dequantize, requant_to)?;

    // Optional bias tensors (always F32)
    for bias_suffix in &[ATTN_Q_BIAS, ATTN_K_BIAS, ATTN_V_BIAS] {
        let name = layer_tensor_name(layer, bias_suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let data = read_tensor_data(reader, gguf, tensor)?;
            let f32_data = dequantize_to_f32_bytes(
                &data, tensor.ggml_type, tensor.n_elements(), &name,
            )?;
            blob.extend_from_slice(&f32_data);
        }
    }

    // Per-head Q/K RMSNorm weights (always dequantized to F32)
    for suffix in &[ATTN_Q_NORM, ATTN_K_NORM] {
        let name = layer_tensor_name(layer, suffix);
        if gguf.find_tensor(&name).is_some() {
            append_tensor_to_blob(blob, reader, gguf, &name, /*dequantize=*/ true)?;
        }
    }

    Ok(())
}
