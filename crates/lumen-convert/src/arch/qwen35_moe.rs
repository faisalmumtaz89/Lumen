//! Qwen3.5-MoE converter: hybrid GDN + full-attention with stacked expert tensors.

use super::ArchConverter;
use super::gdn_gates::{compute_ssm_slices, write_ssm_tensors};
use crate::convert::ConvertError;
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufFile};
use crate::tensor_names::*;
use crate::tensor_io::*;
use lumen_format::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek, SeekFrom};

pub(crate) struct Qwen35MoeConverter {
    pub(crate) num_experts: u32,
}

impl ArchConverter for Qwen35MoeConverter {
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        _requant_to: Option<QuantScheme>,
    ) -> Result<LayerShape, ConvertError> {
        compute_layer_shape_qwen35moe(gguf, layer, self.num_experts, dequantize)
    }

    fn write_layer_blob<R: Read + Seek>(
        &self,
        blob: &mut Vec<u8>,
        reader: &mut R,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        _requant_to: Option<QuantScheme>,
    ) -> Result<(), ConvertError> {
        write_qwen35moe_layer_blob(blob, reader, gguf, layer, self.num_experts, dequantize)
    }

    fn layer_kind_label(&self, layer: usize) -> String {
        let kind = if is_qwen35moe_full_attention_layer(layer) {
            "full_attn"
        } else {
            "linear_attn"
        };
        format!("{}, {} experts", kind, self.num_experts)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Returns true if this layer index is a "full attention" layer in the Qwen3.5-MoE pattern.
/// Full attention layers occur at indices 3, 7, 11, 15, ... (every 4th, 0-indexed, starting at 3).
pub(crate) fn is_qwen35moe_full_attention_layer(layer: usize) -> bool {
    layer >= 3 && (layer - 3) % 4 == 0
}

/// Read a sub-range of a stacked tensor (e.g. ffn_gate_exps[expert_idx]) and append to blob.
/// The stacked tensor has shape [num_experts, rows, cols]. Each expert slice is
/// rows*cols elements, contiguous in memory.
fn append_stacked_expert_slice<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    tensor_name: &str,
    expert_idx: usize,
    num_experts: u64,
    dequantize: bool,
) -> Result<(), ConvertError> {
    let tensor = gguf.find_tensor(tensor_name)
        .ok_or_else(|| ConvertError::MissingTensor(tensor_name.to_string()))?;
    let total_elements = tensor.n_elements();
    let expert_elements = total_elements / num_experts;
    let base_offset = gguf.tensor_data_offset(tensor);
    if dequantize {
        let total_size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: tensor_name.to_string(),
            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
        })?;
        let expert_byte_size = total_size / num_experts;
        let expert_offset = base_offset + (expert_idx as u64) * expert_byte_size;
        reader.seek(SeekFrom::Start(expert_offset))?;
        let mut buf = vec![0u8; expert_byte_size as usize];
        reader.read_exact(&mut buf)?;
        let f32_data = dequantize_to_f32_bytes(&buf, tensor.ggml_type, expert_elements, tensor_name)?;
        blob.extend_from_slice(&f32_data);
    } else if tensor.ggml_type == GgmlType::Q4_1 {
        // Q4_1 has no dedicated GPU kernel -- requantize to Q4_0.
        let total_size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: tensor_name.to_string(),
            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
        })?;
        let expert_byte_size = total_size / num_experts;
        let expert_offset = base_offset + (expert_idx as u64) * expert_byte_size;
        reader.seek(SeekFrom::Start(expert_offset))?;
        let mut buf = vec![0u8; expert_byte_size as usize];
        reader.read_exact(&mut buf)?;
        let f32_data = dequantize_to_f32_bytes(&buf, tensor.ggml_type, expert_elements, tensor_name)?;
        let n_elems = expert_elements as usize;
        let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
        blob.extend_from_slice(&q4_data);
    } else {
        let total_size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: tensor_name.to_string(),
            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
        })?;
        let expert_byte_size = total_size / num_experts;
        let expert_offset = base_offset + (expert_idx as u64) * expert_byte_size;
        reader.seek(SeekFrom::Start(expert_offset))?;
        let mut buf = vec![0u8; expert_byte_size as usize];
        reader.read_exact(&mut buf)?;
        blob.extend_from_slice(&buf);
    }
    Ok(())
}

/// Compute TensorSlice for a single expert within a stacked tensor.
///
/// `num_experts` must be the model's declared expert count (from GGUF metadata),
/// not inferred from tensor dims -- different models store experts as the first or
/// last dimension (e.g. Mixtral: [num_experts, ...], Qwen3.5: [..., num_experts]).
fn compute_stacked_slice(
    gguf: &GgufFile,
    tensor_name: &str,
    blob_offset: &mut u64,
    dequantize: bool,
    num_experts: u64,
) -> Result<TensorSlice, ConvertError> {
    let tensor = gguf.find_tensor(tensor_name)
        .ok_or_else(|| ConvertError::MissingTensor(tensor_name.to_string()))?;

    if dequantize {
        let expert_elements = tensor.n_elements() / num_experts;
        let size = expert_elements * 4;
        let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
        *blob_offset += size;
        Ok(slice)
    } else if tensor.ggml_type == GgmlType::Q4_1 {
        // Q4_1 has no dedicated GPU kernel -- requantize to Q4_0.
        let expert_elements = tensor.n_elements() / num_experts;
        assert!(expert_elements % 32 == 0, "Q4_1->Q4_0 requires elements divisible by 32, got {expert_elements}");
        let size = ((expert_elements as usize / 32) * 18) as u64;
        let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::Q4_0 };
        *blob_offset += size;
        Ok(slice)
    } else {
        let quant = tensor.ggml_type.to_lbc_quant()
            .ok_or_else(|| ConvertError::UnsupportedTensorType {
                tensor: tensor_name.to_string(),
                ggml_type: format!("{:?}", tensor.ggml_type),
            })?;
        let total_size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: tensor_name.to_string(),
            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
        })?;
        let size = total_size / num_experts;
        let slice = TensorSlice { offset: *blob_offset, length: size, quant };
        *blob_offset += size;
        Ok(slice)
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5-MoE layer shape computation
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a Qwen3.5-MoE layer.
///
/// Qwen3.5-MoE layer blob layout:
///   [Wq | Wk | Wv | Wo]                     -- attention projections
///   [attn_norm]                               -- pre-attention RMSNorm
///   [attn_post_norm]                          -- post-attention RMSNorm (if present)
///   [ffn_norm]                                -- pre-FFN RMSNorm (if present)
///   [attn_gate]                               -- output gate weight (full attn layers only)
///   [SSM tensors: a, conv1d, dt, beta, alpha, norm, out]  -- linear attn layers only
///   [router_weight]                           -- expert router
///   [expert_0: gate | up | down]             -- de-stacked per-expert FFN weights
///   [expert_N: gate | up | down]
///   [shared_expert: gate | up | down]        -- shared/always-on expert
///   [optional: bq | bk | bv]                 -- QKV biases
fn compute_layer_shape_qwen35moe(
    gguf: &GgufFile,
    layer: usize,
    num_experts: u32,
    dequantize: bool,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Helper to compute a TensorSlice for a given tensor.
    let compute_slice = |gguf: &GgufFile, name: &str, blob_offset: &mut u64, dequantize: bool|
        -> Result<TensorSlice, ConvertError>
    {
        let tensor = gguf.find_tensor(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_string()))?;
        if dequantize {
            let n_elements = tensor.n_elements();
            let size = n_elements * 4;
            let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q4_1 {
            // Q4_1 has no dedicated GPU kernel -- requantize to Q4_0.
            let n_elements = tensor.n_elements();
            assert!(n_elements % 32 == 0, "Q4_1->Q4_0 requires elements divisible by 32, got {n_elements}");
            let size = ((n_elements as usize / 32) * 18) as u64;
            let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::Q4_0 };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q8_1 {
            // Q8_1 has no LBC QuantScheme -- requantize to Q8_0.
            let n_elements = tensor.n_elements();
            assert!(n_elements % 32 == 0, "Q8_1->Q8_0 requires elements divisible by 32, got {n_elements}");
            let size = ((n_elements as usize / 32) * 34) as u64;
            let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::Q8_0 };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q5_1 {
            // Q5_1 has no LBC QuantScheme -- dequantize to F32.
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
    // Returns None if the tensor is absent. For tensors with no direct LBC
    // mapping but a known dequant path (Q8_1, Q5_1, MXFP4, etc.), forces
    // dequantization to F32 instead of silently skipping.
    let try_compute_opt_slice = |gguf: &GgufFile, layer: usize, suffix: &str, blob_offset: &mut u64, dequantize: bool|
        -> Result<Option<TensorSlice>, ConvertError>
    {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let force_dequant = !dequantize && tensor.ggml_type.to_lbc_quant().is_none();
            if force_dequant {
                if tensor.ggml_type.has_dequant_path() {
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
    // Full attention layers (3,7,11,...): separate attn_q/k/v/output.weight
    // Linear attention layers (0,1,2,...): fused attn_qkv.weight (stored in wq slot), no wk/wv/wo
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

    // Post-attention norm (present in all Qwen3.5-MoE layers)
    let attn_post_norm = try_compute_opt_slice(gguf, layer, ATTN_POST_NORM, &mut blob_size, dequantize)?;

    // FFN norm (present in all layers with MoE FFN)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    let ffn_norm = if gguf.find_tensor(&ffn_norm_name).is_some() {
        compute_slice(gguf, &ffn_norm_name, &mut blob_size, dequantize)?
    } else {
        // If ffn_norm is missing, use a zero-length sentinel
        TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 }
    };

    // Attention gate (full attention layers only)
    let attn_gate = try_compute_opt_slice(gguf, layer, ATTN_GATE_WEIGHT, &mut blob_size, dequantize)?;

    // SSM tensors (linear attention layers only) — ssm_alpha/beta MUST be Q8_0.
    // Shared logic in gdn_gates handles force-requant from F32/F16/BF16 to Q8_0.
    let ssm = compute_ssm_slices(gguf, layer, &mut blob_size, dequantize)?;
    let ssm_a = ssm.ssm_a;
    let ssm_conv1d = ssm.ssm_conv1d;
    let ssm_dt = ssm.ssm_dt;
    let ssm_beta = ssm.ssm_beta;
    let ssm_alpha = ssm.ssm_alpha;
    let ssm_norm = ssm.ssm_norm;
    let ssm_out = try_compute_opt_slice(gguf, layer, SSM_OUT, &mut blob_size, true)?;  // force F32: Q5_K unsupported at runtime

    // Router weight (always F32)
    let router_weight = compute_slice(gguf, &layer_tensor_name(layer, FFN_GATE_INP), &mut blob_size, /*dequantize=*/ true)?;

    // De-stacked per-expert FFN weights
    let gate_exps_name = layer_tensor_name(layer, FFN_GATE_EXPS);
    let up_exps_name = layer_tensor_name(layer, FFN_UP_EXPS);
    let down_exps_name = layer_tensor_name(layer, FFN_DOWN_EXPS);

    let mut expert_slices = Vec::with_capacity(num_experts as usize);
    for _e in 0..num_experts as usize {
        let gate = compute_stacked_slice(gguf, &gate_exps_name, &mut blob_size, dequantize, num_experts.into())?;
        let up = compute_stacked_slice(gguf, &up_exps_name, &mut blob_size, dequantize, num_experts.into())?;
        let down = compute_stacked_slice(gguf, &down_exps_name, &mut blob_size, dequantize, num_experts.into())?;
        expert_slices.push(ExpertSlice { gate, up, down });
    }

    // Shared expert weights: requantize to Q4_0 for efficient runtime dispatch.
    // The source formats (MXFP4 for gate/up, Q6_K for down) lack direct LBC mappings,
    // so we dequantize to F32 then requantize to Q4_0.  Q4_0 byte size = (n_elements / 32) * 18.
    let try_compute_slice_q4 = |gguf: &GgufFile, layer: usize, suffix: &str, blob_offset: &mut u64|
        -> Result<Option<TensorSlice>, ConvertError>
    {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let n_elements = tensor.n_elements();
            assert!(n_elements % 32 == 0,
                "Q4_0 requires elements divisible by 32, got {n_elements} for {name}");
            let q4_size = (n_elements as u64 / 32) * 18;
            let slice = TensorSlice { offset: *blob_offset, length: q4_size, quant: QuantScheme::Q4_0 };
            *blob_offset += q4_size;
            Ok(Some(slice))
        } else {
            Ok(None)
        }
    };
    let shared_expert_gate = try_compute_slice_q4(gguf, layer, FFN_GATE_SHEXP, &mut blob_size)?;
    let shared_expert_up = try_compute_slice_q4(gguf, layer, FFN_UP_SHEXP, &mut blob_size)?;
    let shared_expert_down = try_compute_slice_q4(gguf, layer, FFN_DOWN_SHEXP, &mut blob_size)?;

    // Optional bias tensors
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    // Per-head Q/K RMSNorm weights (full attention layers only, always F32)
    let attn_q_norm = try_compute_opt_slice(gguf, layer, ATTN_Q_NORM, &mut blob_size, /*dequantize=*/ true)?;
    let attn_k_norm = try_compute_opt_slice(gguf, layer, ATTN_K_NORM, &mut blob_size, /*dequantize=*/ true)?;

    // Shared expert gate input weight: sigmoid(dot(ffn_gate_inp_shexp, input)) gates shared expert output.
    // Always F32 (small: [hidden_dim] = 2048 elements = 8 KB).
    let ffn_gate_inp_shexp = try_compute_opt_slice(gguf, layer, FFN_GATE_INP_SHEXP, &mut blob_size, /*dequantize=*/ true)?;

    let zero_slice = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };

    let layer_type = if is_full_attn { Some(0u8) } else { Some(1u8) };

    let subtensors = SubtensorOffsets {
        wq, wk, wv, wo,
        bq, bk, bv,
        w_gate: zero_slice, w_up: zero_slice, w_down: zero_slice,
        attn_norm, ffn_norm,
        router_weight: Some(router_weight),
        experts: Some(expert_slices),
        shared_expert_gate, shared_expert_up, shared_expert_down,
        attn_gate, attn_post_norm,
        ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
        attn_q_norm, attn_k_norm, ffn_gate_inp_shexp,
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
// Qwen3.5-MoE layer blob writing
// ---------------------------------------------------------------------------

/// Write a Qwen3.5-MoE layer blob, de-stacking experts from stacked tensors.
fn write_qwen35moe_layer_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    num_experts: u32,
    dequantize: bool,
) -> Result<(), ConvertError> {
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Attention projections: layout differs by layer type
    if is_full_attn {
        // Full attention: separate Q/K/V/output tensors
        for suffix in &ATTN_TENSOR_SUFFIXES {
            append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, suffix), dequantize)?;
        }
    } else {
        // Linear attention: fused QKV tensor only (stored in wq slot in index)
        append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, ATTN_QKV), dequantize)?;
    }

    // Pre-attention norm
    append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, ATTN_NORM), dequantize)?;

    // Post-attention norm (if present)
    let post_norm_name = layer_tensor_name(layer, ATTN_POST_NORM);
    if gguf.find_tensor(&post_norm_name).is_some() {
        append_tensor_to_blob(blob, reader, gguf, &post_norm_name, dequantize)?;
    }

    // FFN norm (if present)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    if gguf.find_tensor(&ffn_norm_name).is_some() {
        append_tensor_to_blob(blob, reader, gguf, &ffn_norm_name, dequantize)?;
    }

    // Attention gate (if present)
    let attn_gate_name = layer_tensor_name(layer, ATTN_GATE_WEIGHT);
    if gguf.find_tensor(&attn_gate_name).is_some() {
        append_tensor_to_blob(blob, reader, gguf, &attn_gate_name, dequantize)?;
    }

    // SSM tensors (if present) — shared GDN gate logic handles force-requant
    // of ssm_alpha/beta to Q8_0 when source is F32/F16/BF16.
    write_ssm_tensors(blob, reader, gguf, layer, dequantize)?;
    {
        let name = layer_tensor_name(layer, SSM_OUT);
        if gguf.find_tensor(&name).is_some() {
            append_tensor_to_blob(blob, reader, gguf, &name, true)?;  // force F32: Q5_K unsupported
        }
    }

    // Router weight (always F32)
    append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, FFN_GATE_INP), /*dequantize=*/ true)?;

    // De-stack expert weights from stacked tensors
    let gate_exps_name = layer_tensor_name(layer, FFN_GATE_EXPS);
    let up_exps_name = layer_tensor_name(layer, FFN_UP_EXPS);
    let down_exps_name = layer_tensor_name(layer, FFN_DOWN_EXPS);

    for e in 0..num_experts as usize {
        append_stacked_expert_slice(blob, reader, gguf, &gate_exps_name, e, num_experts.into(), dequantize)?;
        append_stacked_expert_slice(blob, reader, gguf, &up_exps_name, e, num_experts.into(), dequantize)?;
        append_stacked_expert_slice(blob, reader, gguf, &down_exps_name, e, num_experts.into(), dequantize)?;
    }

    // Shared expert weights (if present) -- dequantize to F32 then requantize to Q4_0.
    // The source formats (MXFP4 for gate/up, Q6_K for down) lack direct LBC mappings.
    // Requantizing to Q4_0 gives 8x smaller weights vs F32, eliminating the perf regression.
    for suffix in &[FFN_GATE_SHEXP, FFN_UP_SHEXP, FFN_DOWN_SHEXP] {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let data = read_tensor_data(reader, gguf, tensor)?;
            let f32_data = dequantize_to_f32_bytes(
                &data, tensor.ggml_type, tensor.n_elements(), &name,
            )?;
            let n_elems = tensor.n_elements() as usize;
            let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
            eprintln!("  Requantized {} to Q4_0 ({} -> {} bytes, {} elements)",
                name, f32_data.len(), q4_data.len(), n_elems);
            blob.extend_from_slice(&q4_data);
        }
    }

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

    // Shared expert gate input weight (always F32)
    {
        let name = layer_tensor_name(layer, FFN_GATE_INP_SHEXP);
        if gguf.find_tensor(&name).is_some() {
            append_tensor_to_blob(blob, reader, gguf, &name, /*dequantize=*/ true)?;
        }
    }

    Ok(())
}
