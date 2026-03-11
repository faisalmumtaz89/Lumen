//! GGUF tensor name constants and suffix arrays.

// ---------------------------------------------------------------------------
// Global tensor names
// ---------------------------------------------------------------------------

pub(crate) const EMBEDDING_NAME: &str = "token_embd.weight";
pub(crate) const FINAL_NORM_NAME: &str = "output_norm.weight";
pub(crate) const OUTPUT_PROJ_NAME: &str = "output.weight";

// ---------------------------------------------------------------------------
// Per-layer tensor name suffixes (dense)
// ---------------------------------------------------------------------------

pub(crate) const ATTN_Q: &str = "attn_q.weight";
pub(crate) const ATTN_K: &str = "attn_k.weight";
pub(crate) const ATTN_V: &str = "attn_v.weight";
pub(crate) const ATTN_OUTPUT: &str = "attn_output.weight";
pub(crate) const FFN_GATE: &str = "ffn_gate.weight";
pub(crate) const FFN_UP: &str = "ffn_up.weight";
pub(crate) const FFN_DOWN: &str = "ffn_down.weight";
pub(crate) const ATTN_NORM: &str = "attn_norm.weight";
pub(crate) const FFN_NORM: &str = "ffn_norm.weight";

// ---------------------------------------------------------------------------
// Optional per-layer bias tensors (Qwen2 and similar architectures)
// ---------------------------------------------------------------------------

pub(crate) const ATTN_Q_BIAS: &str = "attn_q.bias";
pub(crate) const ATTN_K_BIAS: &str = "attn_k.bias";
pub(crate) const ATTN_V_BIAS: &str = "attn_v.bias";

// ---------------------------------------------------------------------------
// MoE tensor name patterns
// ---------------------------------------------------------------------------

// Router: blk.{L}.ffn_gate_inp.weight  -- shape [num_experts, hidden_dim]
// Per-expert: blk.{L}.ffn_gate.{E}.weight, blk.{L}.ffn_up.{E}.weight, blk.{L}.ffn_down.{E}.weight
pub(crate) const FFN_GATE_INP: &str = "ffn_gate_inp.weight";

// ---------------------------------------------------------------------------
// Stacked expert tensor names (Qwen3.5-MoE uses stacked format instead of per-expert)
// ---------------------------------------------------------------------------

// Shape: [num_experts, intermediate_dim, hidden_dim] for gate/up,
//        [num_experts, hidden_dim, intermediate_dim] for down.
pub(crate) const FFN_GATE_EXPS: &str = "ffn_gate_exps.weight";
pub(crate) const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
pub(crate) const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";

// ---------------------------------------------------------------------------
// Shared expert tensor names (Qwen3.5-MoE)
// ---------------------------------------------------------------------------

pub(crate) const FFN_GATE_SHEXP: &str = "ffn_gate_shexp.weight";
pub(crate) const FFN_UP_SHEXP: &str = "ffn_up_shexp.weight";
pub(crate) const FFN_DOWN_SHEXP: &str = "ffn_down_shexp.weight";

// ---------------------------------------------------------------------------
// Extended attention tensor names (Qwen3.5-MoE full attention layers)
// ---------------------------------------------------------------------------

pub(crate) const ATTN_Q_NORM: &str = "attn_q_norm.weight";
pub(crate) const ATTN_K_NORM: &str = "attn_k_norm.weight";
pub(crate) const ATTN_GATE_WEIGHT: &str = "attn_gate.weight";
pub(crate) const ATTN_POST_NORM: &str = "post_attention_norm.weight";

// Shared expert gating (Qwen3.5-MoE): sigmoid(dot(ffn_gate_inp_shexp, input)) gates shared expert output.
pub(crate) const FFN_GATE_INP_SHEXP: &str = "ffn_gate_inp_shexp.weight";
// Fused QKV weight used by Qwen3.5-MoE linear attention layers
pub(crate) const ATTN_QKV: &str = "attn_qkv.weight";

// ---------------------------------------------------------------------------
// SSM / linear attention tensor names (Qwen3.5-MoE GatedDeltaNet layers)
// ---------------------------------------------------------------------------

pub(crate) const SSM_A: &str = "ssm_a";           // no .weight suffix in GGUF
pub(crate) const SSM_CONV1D: &str = "ssm_conv1d.weight";
pub(crate) const SSM_DT: &str = "ssm_dt.bias";   // .bias not .weight in GGUF
pub(crate) const SSM_BETA: &str = "ssm_beta.weight";
pub(crate) const SSM_ALPHA: &str = "ssm_alpha.weight";
pub(crate) const SSM_NORM: &str = "ssm_norm.weight";
pub(crate) const SSM_OUT: &str = "ssm_out.weight";

// ---------------------------------------------------------------------------
// Suffix arrays
// ---------------------------------------------------------------------------

/// All per-layer tensor suffixes in the order they appear in the LBC layer blob.
/// For dense layers only. MoE layers use a different path.
pub(crate) const LAYER_TENSOR_SUFFIXES: [&str; 9] = [
    ATTN_Q, ATTN_K, ATTN_V, ATTN_OUTPUT, FFN_GATE, FFN_UP, FFN_DOWN, ATTN_NORM, FFN_NORM,
];

/// Attention tensor suffixes shared between dense and MoE layers.
pub(crate) const ATTN_TENSOR_SUFFIXES: [&str; 4] = [ATTN_Q, ATTN_K, ATTN_V, ATTN_OUTPUT];

/// Norm tensor suffixes shared between dense and MoE layers.
pub(crate) const NORM_TENSOR_SUFFIXES: [&str; 2] = [ATTN_NORM, FFN_NORM];
