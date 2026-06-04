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

    // Supported architectures: Qwen3.5 (dense GatedDeltaNet) and Qwen3.5 MoE.
    match arch.as_str() {
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

    let metadata_block_count = get_required_u32(gguf, &format!("{prefix}.block_count"))?;
    // Some GGUF producers count auxiliary/MTP transformer blocks in
    // `block_count` even though those blocks are not part of the main
    // backbone. Concretely, `convert_hf_to_gguf.py --outtype bf16` for
    // Qwen3.5-9B writes `qwen35.block_count = 33` because it counts the
    // single MTP "Next-N" head (blk.32, identified by `nextn.eh_proj.weight`,
    // `nextn.enorm.weight`, `nextn.hnorm.weight`, `nextn.shared_head_norm.weight`)
    // alongside the 32 real layers. Trusting the metadata then causes the
    // converter to iterate up to layer 32 and crash with
    // `missing tensor: blk.32.attn_qkv.weight` because the schedule says
    // layer 32 should be a linear-attn layer but the MTP head actually
    // ships full-attn tensors (attn_q/k/v/output).
    //
    // Cross-check by counting `blk.N` indices that contain a REQUIRED
    // attention weight (`attn_q.weight` OR `attn_qkv.weight`) AND lack
    // any `nextn.*` MTP marker tensor. This handles:
    //   - overshoot from MTP heads (Qwen3.5 NextN)
    //   - simple off-by-one in `block_count` metadata
    //   - undercount in `block_count`
    let real_layers = real_main_layer_count(gguf);
    let num_layers = match real_layers {
        Some(observed) if observed != metadata_block_count => {
            eprintln!(
                "  WARNING: {prefix}.block_count metadata says {metadata_block_count} but \
                 observed {observed} real (non-MTP) blk.* layers in tensor list. \
                 Using observed value.",
            );
            observed
        }
        Some(observed) => observed,
        None => metadata_block_count,
    };
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
        rope_neox: matches!(arch.as_str(), "qwen35" | "qwen35moe" | "qwen3_5_moe" | "qwen3.5_moe"),
    };

    Ok((hp, arch))
}

/// Return the number of REAL (non-MTP) backbone layers detected in the GGUF
/// tensor list, or `None` if no `blk.N.*` tensors are present.
///
/// A `blk.N` index counts as a real layer iff:
///   1. It exposes at least one of `attn_q.weight` (full-attn) or
///      `attn_qkv.weight` (fused/linear-attn). These are the load-bearing
///      attention weights every backbone layer must carry.
///   2. It does NOT contain any `nextn.*` marker tensor
///      (`nextn.eh_proj.weight`, `nextn.enorm.weight`, `nextn.hnorm.weight`,
///      `nextn.shared_head_norm.weight`). Those tensors mark Multi-Token
///      Prediction "Next-N" heads (Qwen3.5 / DeepSeek-V3 style) which are
///      auxiliary speculative-decode helpers, not main transformer layers.
///
/// The Qwen3.5-9B BF16 GGUF produced by `convert_hf_to_gguf.py --outtype bf16`
/// is the motivating case: 32 real layers (0..31) plus 1 MTP head at blk.32,
/// declared as `block_count = 33`. Without this filter, the converter would
/// loop past the real layers and fail on the wrong attention tensor.
///
/// Returns the number of *consecutive* real layers starting from 0. If there
/// is a gap (e.g. blk.5 is missing but blk.6 exists), only the consecutive
/// prefix is returned — that's the contract our per-layer arch dispatchers
/// rely on.
fn real_main_layer_count(gguf: &GgufFile) -> Option<u32> {
    use std::collections::{HashMap, HashSet};

    // Collect, per-layer, the set of suffixes (everything after `blk.N.`).
    let mut per_layer: HashMap<u32, HashSet<String>> = HashMap::new();
    let mut max_idx: Option<u32> = None;
    for tensor in &gguf.tensors {
        let name = &tensor.name;
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<u32>() {
                    let suffix = rest[dot + 1..].to_string();
                    per_layer.entry(idx).or_default().insert(suffix);
                    max_idx = Some(match max_idx {
                        Some(m) => m.max(idx),
                        None => idx,
                    });
                }
            }
        }
    }
    let max_idx = max_idx?;

    // Walk from 0 upward and stop at the first index that is either missing
    // or is an MTP head.
    let mut real_count: u32 = 0;
    for idx in 0..=max_idx {
        let Some(suffixes) = per_layer.get(&idx) else {
            // Gap: stop counting. The arch dispatcher iterates 0..n_layers
            // consecutively so we can't safely extend past a missing layer.
            break;
        };
        let has_attn = suffixes.contains("attn_q.weight")
            || suffixes.contains("attn_qkv.weight");
        let has_nextn = suffixes.iter().any(|s| s.starts_with("nextn."));
        if !has_attn || has_nextn {
            break;
        }
        real_count += 1;
    }

    Some(real_count)
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
