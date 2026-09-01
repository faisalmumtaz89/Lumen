//! Model hyperparameters stored in the LBC header.
//!
//! Needed by both the runtime (layer count, head dimensions) and the compute
//! backend (buffer allocation, kernel dispatch).

/// Core model hyperparameters.
///
/// For MoE models, `num_experts` and `num_active_experts` are set;
/// for dense models they are `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelHyperparams {
    pub num_layers: u32,
    pub num_heads: u32,
    /// For grouped-query attention; equals `num_heads` for standard MHA.
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// Embedding size.
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_params: Option<RopeParams>,
    /// `None` for dense models.
    pub num_experts: Option<u32>,
    /// `None` for dense models.
    pub num_active_experts: Option<u32>,
    /// Typically 1e-5 or 1e-6.
    pub norm_eps: f32,
    /// Number of dimensions to apply rotary embedding to per head.
    /// `None` = full `head_dim` (default for most models).
    /// `Some(n)` = partial RoPE, only rotate first `n` dims (e.g. Qwen3.5: 64 of 256).
    pub rotary_dim: Option<u32>,
    /// NeoX-style (half-split) RoPE: pairs at (d, d+half_rot) instead of interleaved (2d, 2d+1).
    /// True for Qwen2, Qwen3.5 architectures. False for Llama, Mistral.
    pub rope_neox: bool,
    /// Gated-DeltaNet (linear-attention / SSM) dimensions, carried from GGUF
    /// metadata (`{arch}.ssm.*`). `None` for models without GDN layers OR for
    /// older (v3) LBC files that predate this field — in both cases the runtime
    /// falls back to the Qwen3.5-9B defaults via [`ModelHyperparams::gdn_dims`].
    pub gdn: Option<GdnDims>,
}

/// Gated-DeltaNet (GDN) per-model dimensions.
///
/// These come from GGUF SSM metadata and differ from the standard attention
/// head counts. The mapping from GGUF keys is:
/// - `{arch}.ssm.time_step_rank` -> `num_v_heads` (state / V heads)
/// - `{arch}.ssm.group_count`     -> `num_k_heads` (Q and K pre-repeat heads)
/// - `{arch}.ssm.state_size`      -> `head_dim`
/// - `{arch}.ssm.conv_kernel`     -> `conv_kernel`
///
/// Known shapes:
/// - Qwen3.5-9B:  num_v_heads=32, num_k_heads=16, head_dim=128, conv_kernel=4
///   => v_dim=4096, qk_dim=2048, qkv_dim=8192
/// - Qwen3.6-27B: num_v_heads=48, num_k_heads=16, head_dim=128, conv_kernel=4
///   => v_dim=6144, qk_dim=2048, qkv_dim=10240
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDims {
    /// Number of state / V heads (`ssm.time_step_rank`). 32 for 9B, 48 for 27B.
    pub num_v_heads: u32,
    /// Number of Q/K heads before GQA repeat (`ssm.group_count`). 16 for both.
    pub num_k_heads: u32,
    /// Per-head dimension (`ssm.state_size`). 128 for both.
    pub head_dim: u32,
    /// Conv1d kernel size (`ssm.conv_kernel`). 4 for both.
    pub conv_kernel: u32,
}

impl GdnDims {
    /// Qwen3.5-9B default GDN shape. Used whenever `ModelHyperparams.gdn` is
    /// `None` so that 9B models (and v3 LBC files) stay byte-identical.
    pub const QWEN35_9B: GdnDims = GdnDims {
        num_v_heads: 32,
        num_k_heads: 16,
        head_dim: 128,
        conv_kernel: 4,
    };

    /// V projection dimension: `num_v_heads * head_dim` (4096 for 9B, 6144 for 27B).
    pub fn v_dim(&self) -> u32 {
        self.num_v_heads * self.head_dim
    }

    /// Q (and K) projection dimension: `num_k_heads * head_dim` (2048 for both).
    pub fn qk_dim(&self) -> u32 {
        self.num_k_heads * self.head_dim
    }

    /// Fused QKV dimension: `2 * qk_dim + v_dim` (8192 for 9B, 10240 for 27B).
    pub fn qkv_dim(&self) -> u32 {
        2 * self.qk_dim() + self.v_dim()
    }
}

/// RoPE (Rotary Position Embedding) configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    /// Base frequency (commonly 10000.0).
    pub theta: f32,
    /// 1.0 = no scaling.
    pub scaling_factor: f32,
    pub scaling_type: RopeScalingType,
}

/// RoPE scaling variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeScalingType {
    #[default]
    None,
    Linear,
    /// Neural Tangent Kernel-aware scaling.
    Ntk,
    /// Yet another RoPE extensioN.
    Yarn,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            theta: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RopeScalingType::None,
        }
    }
}

/// Compile-time guard: `DIM_BOUND` is only sound while every three-factor
/// u32 dimension product stays within u32. A free const so rustc always
/// evaluates it (an associated const is only checked when referenced).
const _DIM_BOUND_KEEPS_U32_DIMS_TOTAL: () = assert!(
    3 * (ModelHyperparams::DIM_BOUND as u64) * (ModelHyperparams::DIM_BOUND as u64)
        <= u32::MAX as u64
);

impl ModelHyperparams {
    /// Upper bound on every geometry-feeding dimension field. Real models
    /// sit far below it (head_dim 256, hidden 5120, intermediate 17408);
    /// anything above is a malformed or hostile header. The value is
    /// chosen so that the u32 DIMENSION arithmetic the tree already
    /// performs is total: every dimension accessor product — including
    /// the (2*num_k_heads + num_v_heads) * head_dim composite — is
    /// <= 3 * 2^30 < 2^32 (compile-time-pinned above), and every
    /// per-tensor byte expectation fits u64 with orders of magnitude to
    /// spare. Deeper state/aggregate products (GDN h-state
    /// v_heads*head_dim^2, KV-cache totals) can still exceed u32/u64 on
    /// hostile-but-bounded headers and are separately capped or ledgered
    /// — the bound does NOT claim totality for them; those live behind
    /// the `HOSTILE-HEADER-KERNEL-CAPS` tracker entry (kernel-deep u32
    /// and KV-total products; mostly hostile-header-only, with per-item
    /// reachability stated there). Raising this bound
    /// requires widening the u32 dimension sites first; the free
    /// `_DIM_BOUND_KEEPS_U32_DIMS_TOTAL` const above makes a silent
    /// rebound fail the build.
    pub const DIM_BOUND: u32 = 1 << 15;
    /// Vocab is the one dimension legitimately far larger than the rest
    /// (248320 today); it still gets a generous ceiling.
    pub const VOCAB_BOUND: u32 = 1 << 24;
    /// Sequence length feeds KV-cache geometry only (real max 262144).
    pub const SEQ_BOUND: u32 = 1 << 20;
    /// Layer count bounds every per-layer aggregate.
    pub const LAYER_BOUND: u32 = 1 << 12;
    /// Expert count bounds per-layer expert banks and repack allocations.
    /// 256 is also the sizing assumption baked into the reader's
    /// per-layer index budget (MAX_LAYER_INDEX_ENTRY_SIZE) — the two
    /// must move together.
    pub const EXPERT_BOUND: u32 = 256;

    /// Reject headers whose dimension fields are zero, beyond the sane
    /// bounds, or mutually inconsistent — BEFORE any consumer derives
    /// geometry from them. Zero has no served meaning for any of these
    /// fields (and several consumers divide by them); oversized or
    /// inconsistent fields exist only in malformed or hostile headers.
    pub fn validate_bounds(&self) -> Result<(), String> {
        let field = |name: &str, v: u32, bound: u32| -> Result<(), String> {
            if v == 0 || v > bound {
                return Err(format!(
                    "malformed hyperparams: {name} = {v} is outside [1, {bound}]"
                ));
            }
            Ok(())
        };
        field("num_layers", self.num_layers, Self::LAYER_BOUND)?;
        field("num_heads", self.num_heads, Self::DIM_BOUND)?;
        field("num_kv_heads", self.num_kv_heads, Self::DIM_BOUND)?;
        field("head_dim", self.head_dim, Self::DIM_BOUND)?;
        field("hidden_dim", self.hidden_dim, Self::DIM_BOUND)?;
        field("intermediate_dim", self.intermediate_dim, Self::DIM_BOUND)?;
        field("vocab_size", self.vocab_size, Self::VOCAB_BOUND)?;
        field("max_seq_len", self.max_seq_len, Self::SEQ_BOUND)?;
        if self.num_kv_heads > self.num_heads || self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "malformed hyperparams: num_heads = {} is not a positive multiple of num_kv_heads = {} (GQA grouping requires it)",
                self.num_heads, self.num_kv_heads
            ));
        }
        match (self.num_experts, self.num_active_experts) {
            (Some(e), active) => {
                field("num_experts", e, Self::EXPERT_BOUND)?;
                if let Some(a) = active {
                    if a == 0 || a > e {
                        return Err(format!(
                            "malformed hyperparams: num_active_experts = {a} is outside [1, num_experts = {e}]"
                        ));
                    }
                }
            }
            (None, Some(a)) => {
                return Err(format!(
                    "malformed hyperparams: num_active_experts = {a} without num_experts"
                ));
            }
            (None, None) => {}
        }
        if let Some(gd) = self.gdn {
            field("ssm num_v_heads", gd.num_v_heads, Self::DIM_BOUND)?;
            field("ssm num_k_heads", gd.num_k_heads, Self::DIM_BOUND)?;
            field("ssm head_dim", gd.head_dim, Self::DIM_BOUND)?;
            // The rolling conv buffer holds conv_kernel - 1 slots and is
            // indexed modulo that count: a kernel of 1 divides by zero.
            if gd.conv_kernel < 2 || gd.conv_kernel > Self::DIM_BOUND {
                return Err(format!(
                    "malformed hyperparams: ssm conv_kernel = {} is outside [2, {}]",
                    gd.conv_kernel,
                    Self::DIM_BOUND
                ));
            }
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }

    /// Resolved Gated-DeltaNet dimensions for this model.
    ///
    /// Returns the explicit [`GdnDims`] carried in `self.gdn` when present, or
    /// the Qwen3.5-9B default ([`GdnDims::QWEN35_9B`]) when `None`. The default
    /// fallback guarantees that 9B models and v3 LBC files (which never stored
    /// GDN dims) keep their exact historical shape, so their GPU buffers and
    /// kernel dispatches remain byte-identical.
    pub fn gdn_dims(&self) -> GdnDims {
        self.gdn.unwrap_or(GdnDims::QWEN35_9B)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ModelHyperparams {
        ModelHyperparams {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            max_seq_len: 512,
            rope_params: None,
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-5,
            rotary_dim: None,
            rope_neox: false,
            gdn: Some(GdnDims {
                num_v_heads: 2,
                num_k_heads: 1,
                head_dim: 16,
                conv_kernel: 4,
            }),
        }
    }

    /// Every bounded field rejects 0 and bound+1 and accepts its bound —
    /// table-driven so a field can neither be dropped from the gate nor
    /// silently rebounded without failing here.
    #[test]
    fn bounds_table() {
        assert!(base().validate_bounds().is_ok());
        type Set = fn(&mut ModelHyperparams, u32);
        let cases: [(&str, Set, u32); 12] = [
            (
                "num_layers",
                |h, v| h.num_layers = v,
                ModelHyperparams::LAYER_BOUND,
            ),
            (
                "num_heads",
                |h, v| {
                    h.num_heads = v;
                    h.num_kv_heads = 1;
                },
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "num_kv_heads",
                |h, v| {
                    h.num_heads = ModelHyperparams::DIM_BOUND;
                    h.num_kv_heads = v;
                },
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "head_dim",
                |h, v| h.head_dim = v,
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "hidden_dim",
                |h, v| h.hidden_dim = v,
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "intermediate_dim",
                |h, v| h.intermediate_dim = v,
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "vocab_size",
                |h, v| h.vocab_size = v,
                ModelHyperparams::VOCAB_BOUND,
            ),
            (
                "max_seq_len",
                |h, v| h.max_seq_len = v,
                ModelHyperparams::SEQ_BOUND,
            ),
            (
                "num_experts",
                |h, v| h.num_experts = Some(v),
                ModelHyperparams::EXPERT_BOUND,
            ),
            (
                "ssm num_v_heads",
                |h, v| h.gdn.as_mut().unwrap().num_v_heads = v,
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "ssm num_k_heads",
                |h, v| h.gdn.as_mut().unwrap().num_k_heads = v,
                ModelHyperparams::DIM_BOUND,
            ),
            (
                "ssm head_dim",
                |h, v| h.gdn.as_mut().unwrap().head_dim = v,
                ModelHyperparams::DIM_BOUND,
            ),
        ];
        for (name, set, bound) in cases {
            let mut h = base();
            set(&mut h, bound);
            assert!(h.validate_bounds().is_ok(), "{name} at bound must pass");
            let mut h = base();
            set(&mut h, 0);
            assert!(h.validate_bounds().is_err(), "{name} = 0 must fail");
            let mut h = base();
            set(&mut h, bound + 1);
            let err = h.validate_bounds().unwrap_err();
            assert!(err.contains(name), "{name} over bound: {err}");
        }
        // conv_kernel has a floor of 2 (rolling buffer of conv_kernel - 1
        // slots), so it gets its own three-point case.
        let mut h = base();
        h.gdn.as_mut().unwrap().conv_kernel = ModelHyperparams::DIM_BOUND;
        assert!(h.validate_bounds().is_ok());
        let mut h = base();
        h.gdn.as_mut().unwrap().conv_kernel = 1;
        assert!(h.validate_bounds().unwrap_err().contains("conv_kernel"));
        let mut h = base();
        h.gdn.as_mut().unwrap().conv_kernel = ModelHyperparams::DIM_BOUND + 1;
        assert!(h.validate_bounds().unwrap_err().contains("conv_kernel"));
        // Relations: GQA divisibility, kv <= heads, active <= experts,
        // active-without-experts.
        let mut h = base();
        h.num_heads = 3;
        assert!(h.validate_bounds().unwrap_err().contains("multiple"));
        let mut h = base();
        h.num_kv_heads = 8;
        assert!(h.validate_bounds().is_err());
        let mut h = base();
        h.num_experts = Some(4);
        h.num_active_experts = Some(5);
        assert!(h
            .validate_bounds()
            .unwrap_err()
            .contains("num_active_experts"));
        let mut h = base();
        h.num_active_experts = Some(2);
        assert!(h
            .validate_bounds()
            .unwrap_err()
            .contains("without num_experts"));
    }
}
