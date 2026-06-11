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

impl ModelHyperparams {
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
