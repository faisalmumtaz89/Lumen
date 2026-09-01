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

impl SubtensorOffsets {
    /// True if any nonempty slice in this layer carries `scheme`. Used for
    /// backend-capability checks: LBC permits per-tensor quantization, so
    /// the header's primary scheme alone cannot prove a scheme is absent.
    pub fn uses_quant(&self, scheme: QuantScheme) -> bool {
        let hit = |t: &TensorSlice| t.length > 0 && t.quant == scheme;
        let opt = |t: &Option<TensorSlice>| t.as_ref().is_some_and(hit);
        hit(&self.wq)
            || hit(&self.wk)
            || hit(&self.wv)
            || hit(&self.wo)
            || hit(&self.w_gate)
            || hit(&self.w_up)
            || hit(&self.w_down)
            || hit(&self.attn_norm)
            || hit(&self.ffn_norm)
            || opt(&self.bq)
            || opt(&self.bk)
            || opt(&self.bv)
            || opt(&self.router_weight)
            || opt(&self.shared_expert_gate)
            || opt(&self.shared_expert_up)
            || opt(&self.shared_expert_down)
            || opt(&self.attn_gate)
            || opt(&self.attn_post_norm)
            || opt(&self.ssm_a)
            || opt(&self.ssm_conv1d)
            || opt(&self.ssm_dt)
            || opt(&self.ssm_beta)
            || opt(&self.ssm_alpha)
            || opt(&self.ssm_norm)
            || opt(&self.ssm_out)
            || opt(&self.attn_q_norm)
            || opt(&self.attn_k_norm)
            || opt(&self.ffn_gate_inp_shexp)
            || self.experts.as_ref().is_some_and(|es| {
                es.iter()
                    .any(|e| hit(&e.gate) || hit(&e.up) || hit(&e.down))
            })
    }
}

/// A (offset, length) pair identifying a tensor within a layer blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSlice {
    /// Relative to layer blob start.
    pub offset: u64,
    pub length: u64,
    pub quant: QuantScheme,
}

impl SubtensorOffsets {
    /// Every present sub-tensor slice with a stable diagnostic name,
    /// including optional fields and per-expert slices. Backends use this to
    /// validate that a layer's quant schemes are dispatchable before upload.
    pub fn named_slices(&self) -> Vec<(String, &TensorSlice)> {
        let mut out: Vec<(String, &TensorSlice)> = vec![
            ("wq".into(), &self.wq),
            ("wk".into(), &self.wk),
            ("wv".into(), &self.wv),
            ("wo".into(), &self.wo),
            ("w_gate".into(), &self.w_gate),
            ("w_up".into(), &self.w_up),
            ("w_down".into(), &self.w_down),
            ("attn_norm".into(), &self.attn_norm),
            ("ffn_norm".into(), &self.ffn_norm),
        ];
        let opts: [(&str, &Option<TensorSlice>); 19] = [
            ("bq", &self.bq),
            ("bk", &self.bk),
            ("bv", &self.bv),
            ("router_weight", &self.router_weight),
            ("shared_expert_gate", &self.shared_expert_gate),
            ("shared_expert_up", &self.shared_expert_up),
            ("shared_expert_down", &self.shared_expert_down),
            ("attn_gate", &self.attn_gate),
            ("attn_post_norm", &self.attn_post_norm),
            ("ssm_a", &self.ssm_a),
            ("ssm_conv1d", &self.ssm_conv1d),
            ("ssm_dt", &self.ssm_dt),
            ("ssm_beta", &self.ssm_beta),
            ("ssm_alpha", &self.ssm_alpha),
            ("ssm_norm", &self.ssm_norm),
            ("ssm_out", &self.ssm_out),
            ("attn_q_norm", &self.attn_q_norm),
            ("attn_k_norm", &self.attn_k_norm),
            ("ffn_gate_inp_shexp", &self.ffn_gate_inp_shexp),
        ];
        for (name, slice) in opts {
            if let Some(s) = slice {
                out.push((name.to_string(), s));
            }
        }
        if let Some(experts) = &self.experts {
            for (i, e) in experts.iter().enumerate() {
                out.push((format!("expert[{i}].gate"), &e.gate));
                out.push((format!("expert[{i}].up"), &e.up));
                out.push((format!("expert[{i}].down"), &e.down));
            }
        }
        out
    }
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
    /// Validates that every sub-tensor slice fits within the layer blob.
    ///
    /// Drives off [`SubtensorOffsets::named_slices`] — the single
    /// enumeration of every loader-consumed slice — so a field added to
    /// the struct is bounds-checked here the moment it is wired for quant
    /// dispatch, with no parallel list to fall out of sync.
    pub fn validate(&self, layer_idx: usize) -> Result<(), crate::FormatError> {
        let len = self.layer_length_bytes;
        for (name, slice) in self.subtensors.named_slices() {
            let end = slice.offset.checked_add(slice.length);
            if end.map_or(true, |end| end > len) {
                return Err(crate::FormatError::LayerOutOfBounds {
                    layer: layer_idx,
                    tensor_name: name,
                    offset: slice.offset,
                    length: slice.length,
                    file_size: len,
                });
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
        TensorSlice {
            offset,
            length,
            quant: QuantScheme::F32,
        }
    }

    fn valid_index(blob_size: u64) -> LayerIndex {
        let s = make_slice(0, 10);
        LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors: SubtensorOffsets {
                wq: s,
                wk: s,
                wv: s,
                wo: s,
                bq: None,
                bk: None,
                bv: None,
                w_gate: s,
                w_up: s,
                w_down: s,
                attn_norm: s,
                ffn_norm: s,
                router_weight: None,
                experts: None,
                shared_expert_gate: None,
                shared_expert_up: None,
                shared_expert_down: None,
                attn_gate: None,
                attn_post_norm: None,
                ssm_a: None,
                ssm_conv1d: None,
                ssm_dt: None,
                ssm_beta: None,
                ssm_alpha: None,
                ssm_norm: None,
                ssm_out: None,
                attn_q_norm: None,
                attn_k_norm: None,
                ffn_gate_inp_shexp: None,
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

    /// `attn_q_norm`, `attn_k_norm` and `ffn_gate_inp_shexp` are parsed
    /// from the wire and bound raw into the Metal device buffer, but a
    /// hand-rolled slice list once left them unchecked. Driving `validate`
    /// off `named_slices` closes that; assert each is now caught.
    #[test]
    fn validate_optional_norm_and_shexp_fields_bounded() {
        type Setter = fn(&mut SubtensorOffsets, TensorSlice);
        let cases: [(Setter, &str); 3] = [
            (|st, s| st.attn_q_norm = Some(s), "attn_q_norm"),
            (|st, s| st.attn_k_norm = Some(s), "attn_k_norm"),
            (
                |st, s| st.ffn_gate_inp_shexp = Some(s),
                "ffn_gate_inp_shexp",
            ),
        ];
        for (set, name) in cases {
            let mut idx = valid_index(100);
            set(&mut idx.subtensors, make_slice(90, 20)); // 110 > 100
            match idx.validate(0).unwrap_err() {
                crate::FormatError::LayerOutOfBounds { tensor_name, .. } => {
                    assert_eq!(tensor_name, name);
                }
                other => panic!("expected LayerOutOfBounds for {name}, got {other:?}"),
            }
        }
    }

    /// Every optional slice populated in-bounds must pass — guards against
    /// over-rejecting a rich MoE+GDN layer once `validate` covers all
    /// fields enumerated by `named_slices`.
    #[test]
    fn validate_fully_populated_layer_within_bounds() {
        let s = make_slice(0, 10);
        let mut idx = valid_index(1000);
        let st = &mut idx.subtensors;
        for f in [
            &mut st.bq,
            &mut st.bk,
            &mut st.bv,
            &mut st.router_weight,
            &mut st.shared_expert_gate,
            &mut st.shared_expert_up,
            &mut st.shared_expert_down,
            &mut st.attn_gate,
            &mut st.attn_post_norm,
            &mut st.ssm_a,
            &mut st.ssm_conv1d,
            &mut st.ssm_dt,
            &mut st.ssm_beta,
            &mut st.ssm_alpha,
            &mut st.ssm_norm,
            &mut st.ssm_out,
            &mut st.attn_q_norm,
            &mut st.attn_k_norm,
            &mut st.ffn_gate_inp_shexp,
        ] {
            *f = Some(s);
        }
        st.experts = Some(vec![ExpertSlice {
            gate: s,
            up: s,
            down: s,
        }]);
        // Every field validate visits is enumerated by named_slices, so a
        // clean pass proves coverage without over-rejection.
        idx.validate(0).unwrap();
        assert!(idx.subtensors.named_slices().len() >= 20);
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
                wq: s,
                wk: s,
                wv: s,
                wo: s,
                bq: None,
                bk: None,
                bv: None,
                w_gate: make_slice(0, 0),
                w_up: make_slice(0, 0),
                w_down: make_slice(0, 0),
                attn_norm: s,
                ffn_norm: s,
                router_weight: Some(make_slice(60, 10)),
                experts: Some(vec![expert.clone(), expert]),
                shared_expert_gate: None,
                shared_expert_up: None,
                shared_expert_down: None,
                attn_gate: None,
                attn_post_norm: None,
                ssm_a: None,
                ssm_conv1d: None,
                ssm_dt: None,
                ssm_beta: None,
                ssm_alpha: None,
                ssm_norm: None,
                ssm_out: None,
                attn_q_norm: None,
                attn_k_norm: None,
                ffn_gate_inp_shexp: None,
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
                wq: s,
                wk: s,
                wv: s,
                wo: s,
                bq: None,
                bk: None,
                bv: None,
                w_gate: make_slice(0, 0),
                w_up: make_slice(0, 0),
                w_down: make_slice(0, 0),
                attn_norm: s,
                ffn_norm: s,
                router_weight: Some(make_slice(0, 5)),
                experts: Some(vec![bad_expert]),
                shared_expert_gate: None,
                shared_expert_up: None,
                shared_expert_down: None,
                attn_gate: None,
                attn_post_norm: None,
                ssm_a: None,
                ssm_conv1d: None,
                ssm_dt: None,
                ssm_beta: None,
                ssm_alpha: None,
                ssm_norm: None,
                ssm_out: None,
                attn_q_norm: None,
                attn_k_norm: None,
                ffn_gate_inp_shexp: None,
                layer_type: None,
            },
        };
        let err = idx.validate(0).unwrap_err();
        match err {
            crate::FormatError::LayerOutOfBounds { tensor_name, .. } => {
                assert!(
                    tensor_name.contains("down"),
                    "expected expert down tensor, got: {tensor_name}"
                );
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
                wq: s,
                wk: s,
                wv: s,
                wo: s,
                bq: None,
                bk: None,
                bv: None,
                w_gate: make_slice(0, 0),
                w_up: make_slice(0, 0),
                w_down: make_slice(0, 0),
                attn_norm: s,
                ffn_norm: s,
                router_weight: Some(make_slice(90, 20)), // 90+20=110 > 100
                experts: None,
                shared_expert_gate: None,
                shared_expert_up: None,
                shared_expert_down: None,
                attn_gate: None,
                attn_post_norm: None,
                ssm_a: None,
                ssm_conv1d: None,
                ssm_dt: None,
                ssm_beta: None,
                ssm_alpha: None,
                ssm_norm: None,
                ssm_out: None,
                attn_q_norm: None,
                attn_k_norm: None,
                ffn_gate_inp_shexp: None,
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
