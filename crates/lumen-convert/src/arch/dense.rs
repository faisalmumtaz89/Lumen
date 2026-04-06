//! Dense LLaMA-family converter (LLaMA, Mistral, Qwen2, InternLM2, etc.)

use super::ArchConverter;
use crate::convert::ConvertError;
use crate::dequant::*;
use crate::gguf::GgufFile;
use crate::tensor_names::*;
use crate::tensor_io::*;
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

pub(crate) struct DenseConverter;

impl ArchConverter for DenseConverter {
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
    ) -> Result<LayerShape, ConvertError> {
        if let Some(target) = requant_to {
            compute_layer_shape_requant(gguf, layer, target)
        } else {
            compute_layer_shape_dense(gguf, layer, dequantize)
        }
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
        write_dense_layer_blob(blob, reader, gguf, layer, dequantize, requant_to)
    }

    fn layer_kind_label(&self, _layer: usize) -> String {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Dense layer shape computation
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a single dense layer by inspecting its tensors in the GGUF file.
/// When `dequantize` is true, computes blob_size as n_elements * 4 (F32) and sets
/// all subtensor quant schemes to F32.
fn compute_layer_shape_dense(
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;
    let mut slices = Vec::with_capacity(9);

    for suffix in &LAYER_TENSOR_SUFFIXES {
        let name = layer_tensor_name(layer, suffix);
        let tensor = gguf
            .find_tensor(&name)
            .ok_or_else(|| ConvertError::MissingTensor(name.clone()))?;

        if dequantize {
            let n_elements = tensor.n_elements();
            let size = n_elements * 4; // F32 = 4 bytes per element
            slices.push(TensorSlice {
                offset: blob_size,
                length: size,
                quant: QuantScheme::F32,
            });
            blob_size += size;
        } else if tensor.ggml_type == crate::gguf::GgmlType::Q4_1 {
            // Q4_1 has no dedicated GPU kernel (neither Metal nor CUDA).
            // Requantize to Q4_0 during conversion so the runtime can use
            // native Q4_0 dequant-matvec kernels.
            let n_elements = tensor.n_elements();
            assert!(n_elements % 32 == 0, "Q4_0 requires elements divisible by 32, got {n_elements} for {name}");
            let size = ((n_elements as usize / 32) * 18) as u64; // Q4_0: 18 bytes per block
            slices.push(TensorSlice {
                offset: blob_size,
                length: size,
                quant: QuantScheme::Q4_0,
            });
            blob_size += size;
        } else {
            let quant = tensor
                .ggml_type
                .to_lbc_quant()
                .ok_or_else(|| ConvertError::UnsupportedTensorType {
                    tensor: name.clone(),
                    ggml_type: format!("{:?}", tensor.ggml_type),
                })?;
            let size = tensor.byte_size().ok_or_else(|| {
                ConvertError::UnsupportedTensorType {
                    tensor: name,
                    ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                }
            })?;

            slices.push(TensorSlice {
                offset: blob_size,
                length: size,
                quant,
            });
            blob_size += size;
        }
    }

    // Optional bias tensors (appended after weight/norm data)
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    let subtensors = SubtensorOffsets {
        wq: slices[0],
        wk: slices[1],
        wv: slices[2],
        wo: slices[3],
        bq, bk, bv,
        w_gate: slices[4],
        w_up: slices[5],
        w_down: slices[6],
        attn_norm: slices[7],
        ffn_norm: slices[8],
        router_weight: None,
        experts: None,
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
            layer_offset_bytes: 0, // StreamingLbcWriter fixes these
            layer_length_bytes: blob_size,
            subtensors,
        },
    })
}

// ---------------------------------------------------------------------------
// Dense requantization layer shape
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a single layer when requantizing to a target scheme.
/// Weight tensors are sized for the target scheme; norm tensors remain F32.
fn compute_layer_shape_requant(
    gguf: &GgufFile,
    layer: usize,
    target: QuantScheme,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;
    let mut slices = Vec::with_capacity(9);

    for suffix in &LAYER_TENSOR_SUFFIXES {
        let name = layer_tensor_name(layer, suffix);
        let tensor = gguf
            .find_tensor(&name)
            .ok_or_else(|| ConvertError::MissingTensor(name.clone()))?;

        let is_norm = suffix.contains("norm");
        let n_elements = tensor.n_elements();

        if is_norm {
            // Norm tensors stay as F32
            let size = n_elements * 4;
            slices.push(TensorSlice {
                offset: blob_size,
                length: size,
                quant: QuantScheme::F32,
            });
            blob_size += size;
        } else {
            // Weight tensors use target quant scheme
            let size = match target {
                QuantScheme::Q4_0 => {
                    // Q4_0: 18 bytes per 32-element block
                    let n = n_elements as usize;
                    assert!(n % 32 == 0, "Q4_0 requires elements divisible by 32, got {n} for {name}");
                    ((n / 32) * 18) as u64
                }
                QuantScheme::Q8_0 => {
                    // Q8_0: 34 bytes per 32-element block
                    let n = n_elements as usize;
                    assert!(n % 32 == 0, "Q8_0 requires elements divisible by 32, got {n} for {name}");
                    ((n / 32) * 34) as u64
                }
                QuantScheme::F32 => n_elements * 4,
                _ => n_elements * 4, // fallback to F32
            };
            slices.push(TensorSlice {
                offset: blob_size,
                length: size,
                quant: target,
            });
            blob_size += size;
        }
    }

    // Optional bias tensors (always F32, appended after weight/norm data)
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    let subtensors = SubtensorOffsets {
        wq: slices[0],
        wk: slices[1],
        wv: slices[2],
        wo: slices[3],
        bq, bk, bv,
        w_gate: slices[4],
        w_up: slices[5],
        w_down: slices[6],
        attn_norm: slices[7],
        ffn_norm: slices[8],
        router_weight: None,
        experts: None,
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
            layer_offset_bytes: 0, // StreamingLbcWriter fixes these
            layer_length_bytes: blob_size,
            subtensors,
        },
    })
}

// ---------------------------------------------------------------------------
// Dense layer blob writing
// ---------------------------------------------------------------------------

fn write_dense_layer_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
) -> Result<(), ConvertError> {
    for suffix in &LAYER_TENSOR_SUFFIXES {
        let name = layer_tensor_name(layer, suffix);
        let tensor = gguf.find_tensor(&name).ok_or_else(|| {
            ConvertError::MissingTensor(name.clone())
        })?;
        let data = read_tensor_data(reader, gguf, tensor)?;
        if let Some(target_quant) = requant_to {
            // Requantization: dequant -> F32 -> target quant
            // Norm tensors stay as F32 (small, no benefit from quantizing)
            let is_norm = suffix.contains("norm");
            if is_norm {
                let f32_data = dequantize_to_f32_bytes(
                    &data, tensor.ggml_type, tensor.n_elements(), &name,
                )?;
                blob.extend_from_slice(&f32_data);
            } else {
                let f32_data = dequantize_to_f32_bytes(
                    &data, tensor.ggml_type, tensor.n_elements(), &name,
                )?;
                let n_elems = tensor.n_elements() as usize;
                match target_quant {
                    QuantScheme::Q4_0 => {
                        let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
                        blob.extend_from_slice(&q4_data);
                    }
                    _ => {
                        blob.extend_from_slice(&f32_data);
                    }
                }
            }
        } else if dequantize {
            let f32_data = dequantize_to_f32_bytes(
                &data,
                tensor.ggml_type,
                tensor.n_elements(),
                &name,
            )?;
            blob.extend_from_slice(&f32_data);
        } else if tensor.ggml_type == crate::gguf::GgmlType::Q4_1 {
            // Q4_1 has no dedicated GPU kernel (neither Metal nor CUDA).
            // Requantize to Q4_0: dequant Q4_1 -> F32 -> quantize Q4_0.
            let f32_data = dequantize_to_f32_bytes(
                &data, tensor.ggml_type, tensor.n_elements(), &name,
            )?;
            let n_elems = tensor.n_elements() as usize;
            let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
            eprintln!("    Requantized Q4_1 -> Q4_0: {name} ({} -> {} bytes)",
                data.len(), q4_data.len());
            blob.extend_from_slice(&q4_data);
        } else {
            blob.extend_from_slice(&data);
        }
    }

    // Append optional bias tensors (always F32, order: bq, bk, bv)
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
