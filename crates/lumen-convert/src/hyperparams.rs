//! Hyperparameter extraction and quantization scheme detection.

use crate::convert::ConvertError;
use crate::gguf::GgufFile;
use crate::tensor_names::*;
use crate::tensor_io::{layer_tensor_name, expert_tensor_name};
use lumen_format::hyperparams::{ModelHyperparams, RopeParams, RopeScalingType};
use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};

// ---------------------------------------------------------------------------
// Hyperparameter extraction
// ---------------------------------------------------------------------------

fn get_required_u32(gguf: &GgufFile, key: &str) -> Result<u32, ConvertError> {
    gguf.get_u32(key)
        .ok_or_else(|| ConvertError::MissingMetadata(key.to_string()))
}

pub(crate) fn extract_hyperparams(gguf: &GgufFile) -> Result<(ModelHyperparams, String), ConvertError> {
    let arch = gguf
        .get_string("general.architecture")
        .ok_or_else(|| ConvertError::MissingMetadata("general.architecture".into()))?
        .to_string();

    // Support LLaMA family + hybrid MoE architectures
    match arch.as_str() {
        "llama" | "mistral" | "internlm2" | "xverse" | "exaone" | "qwen2" => {}
        "qwen35" | "qwen35moe" | "qwen3_5_moe" | "qwen3.5_moe" => {}
        other => return Err(ConvertError::UnsupportedArchitecture(other.into())),
    }

    let prefix = &arch;

    let num_heads = get_required_u32(gguf, &format!("{prefix}.attention.head_count"))?;
    let hidden_dim = get_required_u32(gguf, &format!("{prefix}.embedding_length"))?;

    // Prefer explicit key_length metadata (required for qwen35moe where head_dim != hidden_dim/num_heads).
    let head_dim = gguf
        .get_u32(&format!("{prefix}.attention.key_length"))
        .unwrap_or(hidden_dim / num_heads);

    let num_kv_heads = gguf
        .get_u32(&format!("{prefix}.attention.head_count_kv"))
        .unwrap_or(num_heads);

    let num_layers = get_required_u32(gguf, &format!("{prefix}.block_count"))?;
    // MoE models (e.g. Qwen3.5-35B-A3B) use expert_feed_forward_length for the per-expert
    // inter_dim and may not have a feed_forward_length field at all.
    // Dense models use feed_forward_length. Try both, preferring expert_feed_forward_length
    // for MoE models so that if BOTH fields exist, we get the per-expert (not shared-expert) dim.
    let num_experts_hint = gguf.get_u32(&format!("{prefix}.expert_count")).is_some()
        || gguf.get_u32(&format!("{prefix}.expert_feed_forward_length")).is_some();
    let intermediate_dim = if num_experts_hint {
        gguf.get_u32(&format!("{prefix}.expert_feed_forward_length"))
            .or_else(|| gguf.get_u32(&format!("{prefix}.feed_forward_length")))
    } else {
        gguf.get_u32(&format!("{prefix}.feed_forward_length"))
    }
    .ok_or_else(|| ConvertError::MissingMetadata(format!("{prefix}.feed_forward_length")))?;

    // Vocab size: try metadata first, fall back to token_embd.weight dims[0]
    let vocab_size = if let Some(tokens) = gguf.get_string_array("tokenizer.ggml.tokens") {
        tokens.len() as u32
    } else if let Some(embd) = gguf.find_tensor(EMBEDDING_NAME) {
        embd.dims.first().copied().unwrap_or(0) as u32
    } else {
        return Err(ConvertError::MissingMetadata(
            "vocab_size (no tokenizer.ggml.tokens or token_embd.weight)".into(),
        ));
    };

    let max_seq_len = gguf
        .get_u32(&format!("{prefix}.context_length"))
        .unwrap_or(4096);

    let rope_theta = gguf
        .get_f32(&format!("{prefix}.rope.freq_base"))
        .unwrap_or(10000.0);
    let rope_scaling_factor = gguf
        .get_f32(&format!("{prefix}.rope.scaling.factor"))
        .unwrap_or(1.0);
    let rope_scaling_type = match gguf.get_string(&format!("{prefix}.rope.scaling.type")) {
        Some("linear") => RopeScalingType::Linear,
        Some("yarn") => RopeScalingType::Yarn,
        _ => RopeScalingType::None,
    };

    let norm_eps = gguf
        .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);

    let num_experts = gguf.get_u32(&format!("{prefix}.expert_count"));
    let num_active_experts = gguf.get_u32(&format!("{prefix}.expert_used_count"));

    // Partial RoPE: some models (e.g. Qwen3.5) only rotate a subset of head dimensions.
    // GGUF key: {arch}.rope.dimension_count. None/0 = full head_dim (most models).
    let rotary_dim = gguf.get_u32(&format!("{prefix}.rope.dimension_count"))
        .filter(|&v| v > 0 && v < head_dim);
    if let Some(d) = rotary_dim {
        if d > 255 {
            return Err(ConvertError::MissingMetadata(
                format!("{prefix}.rope.dimension_count={d} exceeds u8 wire limit (255)")));
        }
    }

    let hp = ModelHyperparams {
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        max_seq_len,
        rope_params: Some(RopeParams {
            theta: rope_theta,
            scaling_factor: rope_scaling_factor,
            scaling_type: rope_scaling_type,
        }),
        num_experts,
        num_active_experts,
        norm_eps,
        rotary_dim,
        rope_neox: matches!(arch.as_str(), "qwen2" | "qwen35" | "qwen35moe" | "qwen3_5_moe" | "qwen3.5_moe"),
    };

    Ok((hp, arch))
}

// ---------------------------------------------------------------------------
// Quantization scheme detection
// ---------------------------------------------------------------------------

/// Determine the dominant quantization scheme from the first layer's weight tensors.
/// Norm tensors are excluded (they are typically F32 regardless).
pub(crate) fn detect_quant_scheme(gguf: &GgufFile, num_layers: u32) -> QuantScheme {
    if num_layers == 0 {
        return QuantScheme::F32;
    }

    // Check the first layer's attention Q weight as representative
    let weight_suffixes = [ATTN_Q, ATTN_K, ATTN_V, ATTN_OUTPUT, FFN_GATE, FFN_UP, FFN_DOWN];

    for suffix in &weight_suffixes {
        let name = layer_tensor_name(0, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            if let Some(quant) = tensor.ggml_type.to_lbc_quant() {
                return quant;
            }
        }
    }

    // For MoE models, dense FFN tensors may not exist. Try expert 0 gate.
    let expert_0_gate = expert_tensor_name(0, "gate", 0);
    if let Some(tensor) = gguf.find_tensor(&expert_0_gate) {
        if let Some(quant) = tensor.ggml_type.to_lbc_quant() {
            return quant;
        }
    }

    // For stacked-expert MoE models (e.g. Qwen3.5-MoE), try ffn_gate_exps.
    let stacked_gate = layer_tensor_name(0, FFN_GATE_EXPS);
    if let Some(tensor) = gguf.find_tensor(&stacked_gate) {
        if let Some(quant) = tensor.ggml_type.to_lbc_quant() {
            return quant;
        }
    }

    QuantScheme::F32
}

/// Build QuantizationDescriptor from a QuantScheme.
pub(crate) fn quant_descriptor_for(scheme: QuantScheme) -> QuantizationDescriptor {
    match scheme {
        QuantScheme::F32 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        },
        QuantScheme::F16 | QuantScheme::Bf16 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 2,
            scale_offset_in_block: None,
        },
        QuantScheme::Q8_0 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(32),
            block_byte_size: 34,
            scale_offset_in_block: Some(32),
        },
        QuantScheme::Q4_0 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(32),
            block_byte_size: 18,
            scale_offset_in_block: Some(16),
        },
        QuantScheme::Q4_1 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(32),
            block_byte_size: 20,
            scale_offset_in_block: Some(16),
        },
        QuantScheme::Q4_K => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(256),
            block_byte_size: 144,
            scale_offset_in_block: None,
        },
        QuantScheme::Q5_0 => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(32),
            block_byte_size: 22,
            scale_offset_in_block: Some(20),
        },
        QuantScheme::Q5_K => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(256),
            block_byte_size: 176,
            scale_offset_in_block: None,
        },
        QuantScheme::Q6_K => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(256),
            block_byte_size: 210,
            scale_offset_in_block: None,
        },
        QuantScheme::Q2_K => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(256),
            block_byte_size: 84,
            scale_offset_in_block: None,
        },
        QuantScheme::Q3_K => QuantizationDescriptor {
            scheme,
            group_size: QuantGroupSize::Group(256),
            block_byte_size: 110,
            scale_offset_in_block: None,
        },
    }
}
