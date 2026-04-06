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
}
