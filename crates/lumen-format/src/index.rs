//! Layer and expert index tables for the LBC format.
//!
//! Sit after the header and before the payload. Provide byte offsets so the
//! runtime can seek directly to any layer or expert blob.

use crate::quantization::QuantScheme;

/// Per-expert FFN weight slices within a MoE layer blob.
///
/// Each expert has its own gate, up, and down projection weights.
/// Offsets are relative to the layer blob start, like all other subtensor slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSlice {
    pub gate: TensorSlice,
    pub up: TensorSlice,
    pub down: TensorSlice,
}

/// Sub-tensor byte ranges within a layer blob.
///
/// All offsets are relative to the layer blob start. The runtime can read
/// the entire layer as one I/O operation, then extract individual tensors.
///
/// For dense models, `router_weight` and `experts` are `None`, and the
/// standard `w_gate`/`w_up`/`w_down` fields are populated.
///
/// For MoE models, `router_weight` and `experts` are populated, and
/// `w_gate`/`w_up`/`w_down` are zero-length sentinel slices (offset=0, length=0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtensorOffsets {
    // -- Attention weights --
    /// Wq (query projection).
    pub wq: TensorSlice,
    /// Wk (key projection).
    pub wk: TensorSlice,
    /// Wv (value projection).
    pub wv: TensorSlice,
    /// Wo (output projection).
    pub wo: TensorSlice,

    // -- MLP weights (dense models) --
    /// W_gate. Zero-length sentinel for MoE layers.
    pub w_gate: TensorSlice,
    /// W_up. Zero-length sentinel for MoE layers.
    pub w_up: TensorSlice,
    /// W_down. Zero-length sentinel for MoE layers.
    pub w_down: TensorSlice,

    // -- QKV biases (Qwen2-family models) --
    /// Bias for query projection (None for models without QKV bias, e.g., LLaMA).
    pub bq: Option<TensorSlice>,
    /// Bias for key projection.
    pub bk: Option<TensorSlice>,
    /// Bias for value projection.
    pub bv: Option<TensorSlice>,

    // -- Normalization --
    pub attn_norm: TensorSlice,
    pub ffn_norm: TensorSlice,

    // -- MoE fields (None for dense layers) --
    /// Router weight for expert selection. Shape: [num_experts, hidden_dim].
    pub router_weight: Option<TensorSlice>,
    /// Per-expert FFN weight slices, one per expert.
    pub experts: Option<Vec<ExpertSlice>>,

    // -- Shared expert (MoE models with a shared/always-on expert) --
    /// Shared expert gate projection.
    pub shared_expert_gate: Option<TensorSlice>,
    /// Shared expert up projection.
    pub shared_expert_up: Option<TensorSlice>,
    /// Shared expert down projection.
    pub shared_expert_down: Option<TensorSlice>,

    // -- Extended attention fields (hybrid models) --
    /// Attention output gate weight (e.g. Qwen3.5-MoE attn_output_gate).
    pub attn_gate: Option<TensorSlice>,
    /// Post-attention RMSNorm weight.
    pub attn_post_norm: Option<TensorSlice>,

    // -- SSM / linear attention fields (hybrid models like Qwen3.5-MoE GatedDeltaNet) --
    /// SSM A matrix (no-scan mode).
    pub ssm_a: Option<TensorSlice>,
    /// Short convolution kernel (conv_kernel_dim=4 typically).
    pub ssm_conv1d: Option<TensorSlice>,
    /// Delta time projection.
    pub ssm_dt: Option<TensorSlice>,
    /// Beta mixing coefficient.
    pub ssm_beta: Option<TensorSlice>,
    /// Alpha coefficient.
    pub ssm_alpha: Option<TensorSlice>,
    /// SSM normalization weight.
    pub ssm_norm: Option<TensorSlice>,
    /// SSM output projection.
    pub ssm_out: Option<TensorSlice>,

    // -- Per-head Q/K normalization (Qwen3.5 full-attention layers) --
    /// Per-head Q RMSNorm weight. Shape: [head_dim] F32, shared across all heads.
    pub attn_q_norm: Option<TensorSlice>,
    /// Per-head K RMSNorm weight. Shape: [head_dim] F32, shared across all heads.
    pub attn_k_norm: Option<TensorSlice>,

    // -- Shared expert gating (MoE layers with a shared/always-on expert) --
    /// Sigmoid gate weight for the shared expert. Shape: [hidden_dim] F32.
    /// Applied as: shared_out *= sigmoid(dot(ffn_gate_inp_shexp, input))
    pub ffn_gate_inp_shexp: Option<TensorSlice>,

    // -- Layer type discriminator --
    /// 0 = standard/full attention (default), 1 = linear attention (GatedDeltaNet).
    /// `None` means legacy LBC file without layer type info (treat as 0).
    pub layer_type: Option<u8>,
}

/// A (offset, length) pair identifying a tensor within a layer blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSlice {
    /// Relative to layer blob start.
    pub offset: u64,
    pub length: u64,
    pub quant: QuantScheme,
}

/// Index entry for a single transformer layer.
///
/// File-level byte range for the layer blob plus sub-tensor offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerIndex {
    pub layer_offset_bytes: u64,
    pub layer_length_bytes: u64,
    pub subtensors: SubtensorOffsets,
}

impl LayerIndex {
    /// Validates that all sub-tensor slices fit within the layer blob.
    pub fn validate(&self, layer_idx: usize) -> Result<(), crate::FormatError> {
        let len = self.layer_length_bytes;

        let validate_slice = |name: &'static str, slice: &TensorSlice| -> Result<(), crate::FormatError> {
            let end = slice.offset.checked_add(slice.length).ok_or(
                crate::FormatError::LayerOutOfBounds {
                    layer: layer_idx,
                    tensor_name: name,
                    offset: slice.offset,
                    length: slice.length,
                    file_size: len,
                },
            )?;
            if end > len {
                return Err(crate::FormatError::LayerOutOfBounds {
                    layer: layer_idx,
                    tensor_name: name,
                    offset: slice.offset,
                    length: slice.length,
                    file_size: len,
                });
            }
            Ok(())
        };

        let slices = [
            ("wq", &self.subtensors.wq),
            ("wk", &self.subtensors.wk),
            ("wv", &self.subtensors.wv),
            ("wo", &self.subtensors.wo),
            ("w_gate", &self.subtensors.w_gate),
            ("w_up", &self.subtensors.w_up),
            ("w_down", &self.subtensors.w_down),
            ("attn_norm", &self.subtensors.attn_norm),
            ("ffn_norm", &self.subtensors.ffn_norm),
        ];

        for (name, slice) in slices {
            validate_slice(name, slice)?;
        }

        // Optional bias slices (Qwen2-family models)
        let bias_slices: [(&str, &Option<TensorSlice>); 3] = [
            ("bq", &self.subtensors.bq),
            ("bk", &self.subtensors.bk),
            ("bv", &self.subtensors.bv),
        ];
        for (name, opt_slice) in bias_slices {
            if let Some(slice) = opt_slice {
                validate_slice(name, slice)?;
            }
        }

        // Shared expert slices
        let shared_expert_slices: [(&str, &Option<TensorSlice>); 3] = [
            ("shared_expert_gate", &self.subtensors.shared_expert_gate),
            ("shared_expert_up", &self.subtensors.shared_expert_up),
            ("shared_expert_down", &self.subtensors.shared_expert_down),
        ];
        for (name, opt_slice) in shared_expert_slices {
            if let Some(slice) = opt_slice {
                validate_slice(name, slice)?;
            }
        }

        // Extended attention fields
        let attn_ext_slices: [(&str, &Option<TensorSlice>); 2] = [
            ("attn_gate", &self.subtensors.attn_gate),
            ("attn_post_norm", &self.subtensors.attn_post_norm),
        ];
        for (name, opt_slice) in attn_ext_slices {
            if let Some(slice) = opt_slice {
                validate_slice(name, slice)?;
            }
        }

        // SSM / linear attention fields
        let ssm_slices: [(&str, &Option<TensorSlice>); 7] = [
            ("ssm_a", &self.subtensors.ssm_a),
            ("ssm_conv1d", &self.subtensors.ssm_conv1d),
            ("ssm_dt", &self.subtensors.ssm_dt),
            ("ssm_beta", &self.subtensors.ssm_beta),
            ("ssm_alpha", &self.subtensors.ssm_alpha),
            ("ssm_norm", &self.subtensors.ssm_norm),
            ("ssm_out", &self.subtensors.ssm_out),
        ];
        for (name, opt_slice) in ssm_slices {
            if let Some(slice) = opt_slice {
                validate_slice(name, slice)?;
            }
        }

        // MoE fields
        if let Some(ref router) = self.subtensors.router_weight {
            validate_slice("router_weight", router)?;
        }
        if let Some(ref experts) = self.subtensors.experts {
            for (e, expert) in experts.iter().enumerate() {
                // Use static string names for expert slices to satisfy the 'static
                // lifetime requirement. We can only validate a fixed number of experts
                // with static names; for the rest, reuse a generic name.
                let gate_name: &'static str = match e {
                    0 => "expert_0_gate", 1 => "expert_1_gate",
                    2 => "expert_2_gate", 3 => "expert_3_gate",
                    4 => "expert_4_gate", 5 => "expert_5_gate",
                    6 => "expert_6_gate", 7 => "expert_7_gate",
                    _ => "expert_N_gate",
                };
                let up_name: &'static str = match e {
                    0 => "expert_0_up", 1 => "expert_1_up",
                    2 => "expert_2_up", 3 => "expert_3_up",
                    4 => "expert_4_up", 5 => "expert_5_up",
                    6 => "expert_6_up", 7 => "expert_7_up",
                    _ => "expert_N_up",
                };
                let down_name: &'static str = match e {
                    0 => "expert_0_down", 1 => "expert_1_down",
                    2 => "expert_2_down", 3 => "expert_3_down",
                    4 => "expert_4_down", 5 => "expert_5_down",
                    6 => "expert_6_down", 7 => "expert_7_down",
                    _ => "expert_N_down",
                };
                validate_slice(gate_name, &expert.gate)?;
                validate_slice(up_name, &expert.up)?;
                validate_slice(down_name, &expert.down)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantScheme;

    fn make_slice(offset: u64, length: u64) -> TensorSlice {
        TensorSlice { offset, length, quant: QuantScheme::F32 }
    }

    fn valid_index(blob_size: u64) -> LayerIndex {
        let s = make_slice(0, 10);
        LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors: SubtensorOffsets {
                wq: s, wk: s, wv: s, wo: s,
                bq: None, bk: None, bv: None,
                w_gate: s, w_up: s, w_down: s,
                attn_norm: s, ffn_norm: s,
                router_weight: None,
                experts: None,
                shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
                attn_gate: None, attn_post_norm: None,
                ssm_a: None, ssm_conv1d: None, ssm_dt: None,
                ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
                attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
                layer_type: None,
            },
        }
    }

    #[test]
    fn validate_all_slices_within_bounds() {
        let idx = valid_index(100);
        idx.validate(0).unwrap();
    }

    #[test]
    fn validate_slice_exceeds_bounds() {
        let mut idx = valid_index(100);
        idx.subtensors.wq = make_slice(90, 20); // 90+20=110 > 100
        let err = idx.validate(0).unwrap_err();
        match err {
            crate::FormatError::LayerOutOfBounds { tensor_name, .. } => {
                assert_eq!(tensor_name, "wq");
            }
            _ => panic!("expected LayerOutOfBounds"),
        }
    }

    #[test]
    fn validate_offset_length_overflow() {
        let mut idx = valid_index(100);
        idx.subtensors.wk = make_slice(u64::MAX, 1);
        assert!(idx.validate(0).is_err());
    }

    #[test]
    fn validate_zero_length_and_exact_boundary() {
        // Zero-length slices are valid
        let mut idx = valid_index(100);
        idx.subtensors.wq = make_slice(50, 0);
        idx.validate(0).unwrap();

        // Exact boundary is valid
        idx.subtensors.wk = make_slice(90, 10); // 90+10=100 == blob_size
        idx.validate(0).unwrap();
    }

    #[test]
    fn validate_moe_fields_within_bounds() {
        let s = make_slice(0, 10);
        let expert = ExpertSlice {
            gate: make_slice(0, 20),
            up: make_slice(20, 20),
            down: make_slice(40, 20),
        };
        let idx = LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: 200,
            subtensors: SubtensorOffsets {
                wq: s, wk: s, wv: s, wo: s,
                bq: None, bk: None, bv: None,
                w_gate: make_slice(0, 0), w_up: make_slice(0, 0), w_down: make_slice(0, 0),
                attn_norm: s, ffn_norm: s,
                router_weight: Some(make_slice(60, 10)),
                experts: Some(vec![expert.clone(), expert]),
                shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
                attn_gate: None, attn_post_norm: None,
                ssm_a: None, ssm_conv1d: None, ssm_dt: None,
                ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
                attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
                layer_type: None,
            },
        };
        idx.validate(0).unwrap();
    }

    #[test]
    fn validate_moe_expert_exceeds_bounds() {
        let s = make_slice(0, 10);
        let bad_expert = ExpertSlice {
            gate: make_slice(0, 10),
            up: make_slice(10, 10),
            down: make_slice(90, 20), // 90+20=110 > 100
        };
        let idx = LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: 100,
            subtensors: SubtensorOffsets {
                wq: s, wk: s, wv: s, wo: s,
                bq: None, bk: None, bv: None,
                w_gate: make_slice(0, 0), w_up: make_slice(0, 0), w_down: make_slice(0, 0),
                attn_norm: s, ffn_norm: s,
                router_weight: Some(make_slice(0, 5)),
                experts: Some(vec![bad_expert]),
                shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
                attn_gate: None, attn_post_norm: None,
                ssm_a: None, ssm_conv1d: None, ssm_dt: None,
                ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
                attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
                layer_type: None,
            },
        };
        let err = idx.validate(0).unwrap_err();
        match err {
            crate::FormatError::LayerOutOfBounds { tensor_name, .. } => {
                assert!(tensor_name.contains("down"), "expected expert down tensor, got: {tensor_name}");
            }
            _ => panic!("expected LayerOutOfBounds"),
        }
    }

    #[test]
    fn validate_moe_router_exceeds_bounds() {
        let s = make_slice(0, 10);
        let idx = LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: 100,
            subtensors: SubtensorOffsets {
                wq: s, wk: s, wv: s, wo: s,
                bq: None, bk: None, bv: None,
                w_gate: make_slice(0, 0), w_up: make_slice(0, 0), w_down: make_slice(0, 0),
                attn_norm: s, ffn_norm: s,
                router_weight: Some(make_slice(90, 20)), // 90+20=110 > 100
                experts: None,
                shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
                attn_gate: None, attn_post_norm: None,
                ssm_a: None, ssm_conv1d: None, ssm_dt: None,
                ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
                attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
                layer_type: None,
            },
        };
        let err = idx.validate(0).unwrap_err();
        match err {
            crate::FormatError::LayerOutOfBounds { tensor_name, .. } => {
                assert_eq!(tensor_name, "router_weight");
            }
            _ => panic!("expected LayerOutOfBounds"),
        }
    }
}
