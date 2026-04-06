//! Generic MoE (Mixtral-style) converter with per-expert tensor files.

use super::ArchConverter;
use crate::convert::ConvertError;
use crate::dequant::*;
use crate::gguf::GgufFile;
use crate::tensor_names::*;
use crate::tensor_io::*;
use lumen_format::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

pub(crate) struct MoeConverter {
    pub(crate) num_experts: u32,
}

impl ArchConverter for MoeConverter {
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        _requant_to: Option<QuantScheme>,
    ) -> Result<LayerShape, ConvertError> {
        compute_layer_shape_moe(gguf, layer, self.num_experts, dequantize)
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
        write_moe_layer_blob(blob, reader, gguf, layer, self.num_experts, dequantize)
    }

    fn layer_kind_label(&self, _layer: usize) -> String {
        format!("MoE: {} experts", self.num_experts)
    }
}

// ---------------------------------------------------------------------------
// MoE layer shape computation
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a single MoE layer by inspecting its tensors in the GGUF file.
///
/// MoE layer blob layout:
///   [Wq | Wk | Wv | Wo]                  -- attention projections
///   [attn_norm | ffn_norm]                -- norms
///   [router_weight]                       -- expert router
///   [expert_0: gate | up | down]          -- per-expert FFN weights
///   [expert_1: gate | up | down]
///   ...
///   [expert_N: gate | up | down]
///   [optional: bq | bk | bv]             -- QKV biases
fn compute_layer_shape_moe(
    gguf: &GgufFile,
    layer: usize,
    num_experts: u32,
    dequantize: bool,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;

    // Helper to compute a TensorSlice for a given tensor
    let compute_slice = |gguf: &GgufFile, name: &str, blob_offset: &mut u64, dequantize: bool|
        -> Result<TensorSlice, ConvertError>
    {
        let tensor = gguf
            .find_tensor(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_string()))?;

        if dequantize {
            let n_elements = tensor.n_elements();
            let size = n_elements * 4;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::F32,
            };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == crate::gguf::GgmlType::Q4_1 {
            // Q4_1 has no dedicated GPU kernel -- requantize to Q4_0.
            let n_elements = tensor.n_elements();
            assert!(n_elements % 32 == 0, "Q4_1->Q4_0 requires elements divisible by 32, got {n_elements} for {name}");
            let size = ((n_elements as usize / 32) * 18) as u64;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::Q4_0,
            };
            *blob_offset += size;
            Ok(slice)
        } else {
            let quant = tensor
                .ggml_type
                .to_lbc_quant()
                .ok_or_else(|| ConvertError::UnsupportedTensorType {
                    tensor: name.to_string(),
                    ggml_type: format!("{:?}", tensor.ggml_type),
                })?;
            let size = tensor.byte_size().ok_or_else(|| {
                ConvertError::UnsupportedTensorType {
                    tensor: name.to_string(),
                    ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                }
            })?;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant,
            };
            *blob_offset += size;
            Ok(slice)
        }
    };

    // Attention projections
    let wq = compute_slice(gguf, &layer_tensor_name(layer, ATTN_Q), &mut blob_size, dequantize)?;
    let wk = compute_slice(gguf, &layer_tensor_name(layer, ATTN_K), &mut blob_size, dequantize)?;
    let wv = compute_slice(gguf, &layer_tensor_name(layer, ATTN_V), &mut blob_size, dequantize)?;
    let wo = compute_slice(gguf, &layer_tensor_name(layer, ATTN_OUTPUT), &mut blob_size, dequantize)?;

    // Norms (always included as F32 in this path since they are small)
    let attn_norm = compute_slice(gguf, &layer_tensor_name(layer, ATTN_NORM), &mut blob_size, dequantize)?;
    let ffn_norm = compute_slice(gguf, &layer_tensor_name(layer, FFN_NORM), &mut blob_size, dequantize)?;

    // Router weight -- always force to F32.  The Metal `moe_router_softmax`
    // kernel declares `device const float* gate_weight`, so the blob must
    // contain F32 regardless of the source GGUF type (commonly F16/BF16).
    // The router is tiny (num_experts * hidden_dim = 32 K params for
    // Mixtral 8x7B), so the cost is negligible.
    let router_weight = compute_slice(gguf, &layer_tensor_name(layer, FFN_GATE_INP), &mut blob_size, /*dequantize=*/ true)?;

    // Per-expert FFN weights
    let mut expert_slices = Vec::with_capacity(num_experts as usize);
    for e in 0..num_experts as usize {
        let gate = compute_slice(gguf, &expert_tensor_name(layer, "gate", e), &mut blob_size, dequantize)?;
        let up = compute_slice(gguf, &expert_tensor_name(layer, "up", e), &mut blob_size, dequantize)?;
        let down = compute_slice(gguf, &expert_tensor_name(layer, "down", e), &mut blob_size, dequantize)?;
        expert_slices.push(ExpertSlice { gate, up, down });
    }

    // Optional bias tensors
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    // MoE layers use zero-length sentinel slices for the dense FFN fields
    let zero_slice = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };

    let subtensors = SubtensorOffsets {
        wq, wk, wv, wo,
        bq, bk, bv,
        w_gate: zero_slice,
        w_up: zero_slice,
        w_down: zero_slice,
        attn_norm, ffn_norm,
        router_weight: Some(router_weight),
        experts: Some(expert_slices),
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
        layer_type: None,
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
// MoE layer blob writing
// ---------------------------------------------------------------------------

/// Write a MoE layer blob in the order expected by the format:
///   [Wq | Wk | Wv | Wo] [attn_norm | ffn_norm] [router] [experts...] [biases]
fn write_moe_layer_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    num_experts: u32,
    dequantize: bool,
) -> Result<(), ConvertError> {
    // Attention projections
    for suffix in &ATTN_TENSOR_SUFFIXES {
        append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, suffix), dequantize)?;
    }

    // Norms
    for suffix in &NORM_TENSOR_SUFFIXES {
        append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, suffix), dequantize)?;
    }

    // Router weight -- always dequantize to F32 (see compute_layer_shape_moe).
    append_tensor_to_blob(blob, reader, gguf, &layer_tensor_name(layer, FFN_GATE_INP), /*dequantize=*/ true)?;

    // Per-expert FFN weights
    for e in 0..num_experts as usize {
        append_tensor_to_blob(blob, reader, gguf, &expert_tensor_name(layer, "gate", e), dequantize)?;
        append_tensor_to_blob(blob, reader, gguf, &expert_tensor_name(layer, "up", e), dequantize)?;
        append_tensor_to_blob(blob, reader, gguf, &expert_tensor_name(layer, "down", e), dequantize)?;
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

    Ok(())
}
