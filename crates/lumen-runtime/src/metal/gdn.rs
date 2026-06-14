//! GatedDeltaNet (linear attention) encode functions for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! These are methods on MetalF32Backend that encode GDN-specific GPU dispatch sequences.

use crate::error::RuntimeError;
use crate::metal::ffi::{
    MetalBuffer, MetalCommandBuffer, MetalComputeEncoder, MTLSize,
};
use lumen_format::quantization::QuantScheme;
use super::{MetalPipelines, MetalScratch, CachedLayerMeta, MetalF32Backend};
use super::graph_reorder;
use super::ffi::MetalSharedEvent;
use std::cell::RefCell;

/// Per-prefill dual-queue context for the GDN branch-overlap path
/// (`LUMEN_METAL_GDN_DUAL_QUEUE=1`). Set by the prefill driver before the layer
/// loop and cleared after; read by `encode_batched_gdn_prefill` to route GDN
/// layers through the dual-queue variant. Thread-local because prefill runs
/// single-threaded per request and this avoids threading the context through
/// the many-arg `encode_layer_batched` call chain (matching the codebase's
/// existing thread-local profile-state pattern). Holds owned aux CB + 3 events
/// for the lifetime of one prefill; `ord` is the running 1-based GDN ordinal.
pub(crate) struct DualQueueCtx {
    pub aux_cmd: MetalCommandBuffer,
    pub ev_norm_ready: MetalSharedEvent,
    pub ev_ab_ready: MetalSharedEvent,
    pub ev_gate_ready: MetalSharedEvent,
    pub ord: u64,
}

thread_local! {
    static DUAL_QUEUE_CTX: RefCell<Option<DualQueueCtx>> = const { RefCell::new(None) };
}

/// Install the dual-queue context for the current prefill (thread-local).
pub(crate) fn dual_queue_ctx_set(ctx: DualQueueCtx) {
    DUAL_QUEUE_CTX.with(|c| *c.borrow_mut() = Some(ctx));
}

/// Take (remove) the dual-queue context, returning it so the driver can commit
/// the aux CB and keep the events alive until the main CB completes.
pub(crate) fn dual_queue_ctx_take() -> Option<DualQueueCtx> {
    DUAL_QUEUE_CTX.with(|c| c.borrow_mut().take())
}

/// True if a dual-queue context is currently installed.
pub(crate) fn dual_queue_ctx_active() -> bool {
    DUAL_QUEUE_CTX.with(|c| c.borrow().is_some())
}

impl MetalF32Backend {
    /// Encode full GatedDeltaNet (linear attention) layer for single-token decode.
    ///
    /// Replaces the standard softmax attention block (RoPE, KV cache write,
    /// multi-head attention, Wo projection) for layers with layer_type=1.
    /// The GDN block produces the same shape output (residual addition + post-norm)
    /// so the downstream FFN block (MoE or dense) is unchanged.
    ///
    /// Dispatch sequence:
    ///   1. RMSNorm(x_buf, attn_norm) -> normed_buf
    ///   2. QKV matvec: normed_buf @ attn_qkv^T -> qkv_buf
    ///   3. ssm_conv1d_decode on k and v portions of qkv_buf
    ///   4. gdn_compute_gates: softplus(dt), exp(decay), sigmoid(beta) -> alpha, beta
    ///   5. l2_normalize_heads on q and k (in-place in qkv_buf)
    ///   6. gated_delta_net_state_update_v2: h = alpha*h + beta*outer(k,v)
    ///   7. gated_delta_net_output: o = q @ h -> gdn_output_buf
    ///   8. ssm_l2_norm_scale: L2-norm + learned scale -> gdn_normed_out_buf
    ///   9. attn_gate: gate[4096] = sigmoid(attn_gate_weight * x_norm[2048]) -> gdn_gate_sigmoid_buf
    ///  10. sigmoid_mul: sigmoid(gate[4096]) * normed_out_buf[4096] -> gdn_gate_sigmoid_buf
    ///  11. ssm_out projection: gdn_gate_sigmoid_buf[4096] @ ssm_out^T -> gdn_ssm_proj_buf[2048]
    ///  12. add_residual: x_buf += gdn_ssm_proj_buf
    ///  13. Copy x_buf -> attn_proj_buf (for FFN norm compatibility)
    /// Returns the new conv position for this GDN layer (to be written back
    /// to `s.gdn_conv_positions[gdn_idx]` by the caller).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gdn_layer_decode(
        cmd: &MetalCommandBuffer,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        meta: &CachedLayerMeta,
        gdn_idx: usize,
    ) -> Result<u32, RuntimeError> {
        let hidden_dim = s.hidden_dim;
        // GDN uses a different head/dim layout than full-attention layers, taken
        // from the resolved SSM dims (9B {32,16,128,4} default, 27B {48,16,128,4}).
        //
        // Reference (Qwen3_5MoeGatedDeltaNet in transformers):
        //   Q+K: num_k_heads heads × head_dim each (repeated_interleave to num_v_heads before GDN)
        //   V:   num_v_heads  heads × head_dim each (no repeat)
        //   conv_dim  = 2*qk_dim + value_dim  (9B = 8192, 27B = 10240)
        //   value_dim = num_v_heads*head_dim — gate and output projection output size
        let num_heads = s.gdn_num_v_heads;   // ssm.time_step_rank = num_v_heads (state/V heads)
        let num_kv_heads = s.gdn_num_k_heads; // ssm.group_count = num_k_heads (Q and K pre-repeat heads)
        let head_dim = s.gdn_head_dim;       // ssm.state_size
        let qk_dim = num_kv_heads * head_dim; // Q and K each: num_k_heads * head_dim (9B = 2048)
        let value_dim = num_heads * head_dim; // V: num_v_heads * head_dim (9B = 4096, 27B = 6144)
        let q_dim = value_dim;               // alias: gate/output projection size = value_dim
        let qkv_dim = 2 * qk_dim + value_dim; // Q + K + V (9B = 8192, 27B = 10240)
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;
        let matmul_tg_size = s.matmul_tg_size;
        let conv_kernel_size = s.gdn_conv_kernel_size;

        // Byte offsets within qkv_conv_buf: Q[0..qk_dim) | K[qk_dim..2*qk_dim) | V[2*qk_dim..)
        let _q_byte_off: u64 = 0;  // Q starts at byte 0
        let k_byte_off: u64 = (qk_dim as u64) * 4;             // = 8192 bytes
        let v_byte_off: u64 = ((qk_dim + qk_dim) as u64) * 4;  // = 16384 bytes

        // Validate required SSM offsets
        let ssm_conv1d_off = meta.ssm_conv1d_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_conv1d_off".into())
        })?;
        let ssm_dt_off = meta.ssm_dt_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_dt_off".into())
        })?;
        let ssm_a_off = meta.ssm_a_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_a_off".into())
        })?;
        let ssm_beta_off = meta.ssm_beta_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_beta_off".into())
        })?;
        let ssm_alpha_off = meta.ssm_alpha_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_alpha_off".into())
        })?;
        let ssm_norm_off = meta.ssm_norm_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_norm_off".into())
        })?;
        let ssm_out_off = meta.ssm_out_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_out_off".into())
        })?;
        let ssm_out_quant = meta.ssm_out_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_out_quant. Re-convert model.".into())
        })?;
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_quant. Re-convert model.".into())
        })?;
        let attn_norm_off = meta.attn_norm_off;

        // Get GDN pipeline states (must be compiled for GDN model)
        let pso_conv1d = pipelines.ssm_conv1d_decode.as_ref().ok_or_else(|| {
            RuntimeError::Compute("ssm_conv1d_decode pipeline not compiled".into())
        })?;
        let pso_l2_norm = pipelines.l2_normalize_heads.as_ref().ok_or_else(|| {
            RuntimeError::Compute("l2_normalize_heads pipeline not compiled".into())
        })?;
        let pso_compute_gates = pipelines.gdn_compute_gates.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_compute_gates pipeline not compiled".into())
        })?;

        // Get GDN scratch buffers
        let alpha_buf = s.gdn_alpha_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_alpha_buf not allocated".into())
        })?;
        let beta_buf = s.gdn_beta_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_beta_buf not allocated".into())
        })?;
        let ssm_proj_buf = s.gdn_ssm_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_ssm_proj_buf not allocated".into())
        })?;
        let gate_sigmoid_buf = s.gdn_gate_sigmoid_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_gate_sigmoid_buf not allocated".into())
        })?;
        let normed_out_buf = s.gdn_normed_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_normed_out_buf not allocated".into())
        })?;
        let alpha_raw_buf = s.gdn_alpha_raw_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_alpha_raw_buf not allocated".into())
        })?;
        let beta_raw_buf = s.gdn_beta_raw_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_beta_raw_buf not allocated".into())
        })?;
        let qkv_conv_buf = s.gdn_qkv_conv_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_qkv_conv_buf not allocated".into())
        })?;

        let h_state_buf = &s.gdn_h_states[gdn_idx];
        let conv_state_buf = &s.gdn_conv_states[gdn_idx];
        let conv_pos = s.gdn_conv_positions[gdn_idx];

        // ================================================================
        // Step 1: RMSNorm(x_buf, attn_norm) -> normed_buf
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for attn_norm".into())
            })?;
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Step 2: Fused QKV matvec: normed_buf @ attn_qkv^T -> qkv_buf
        // ================================================================
        {
            let wq_off = meta.wq_off;
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for QKV matvec".into())
            })?;
            let tg = match meta.wq_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg); 64u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                QuantScheme::Bf16 => { enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, wq_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.qkv_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) {
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            }
            let n_tg = match meta.wq_quant { QuantScheme::Q8_0 => ((qkv_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2, QuantScheme::Bf16 => ((qkv_dim as u64) + 1) / 2, _ => qkv_dim as u64 };
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
        }

        // ================================================================
        // Step 3: ssm_conv1d_decode on ALL 8192 QKV channels.
        // ssm_conv1d.weight is [4, 8192] = conv on all QKV channels at once.
        // Output goes to gdn_qkv_conv_buf; downstream steps read Q/K/V
        // portions from it at the appropriate byte offsets.
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for conv1d".into())
            })?;
            enc.set_pipeline_state(pso_conv1d);
            enc.set_buffer(&s.qkv_buf, 0, 0);               // input: all QKV (8192)
            enc.set_buffer(conv_state_buf, 0, 1);             // conv state for all channels
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2);     // weights [8192, 4]
            enc.set_buffer(qkv_conv_buf, 0, 3);               // output: conv'd QKV (8192)
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);   // dim = 8192
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            let conv_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), 1, 1),
                MTLSize::new(conv_tg, 1, 1),
            );
            enc.end_encoding();
        }
        // Compute next conv position (circular buffer, wraps around kernel_size-1).
        // The caller must write this back to s.gdn_conv_positions[gdn_idx].
        let buf_slots = (conv_kernel_size - 1) as u32;
        let new_conv_pos = (conv_pos + 1) % buf_slots;

        // ================================================================
        // Step 3b: SiLU activation on conv1d output.
        // Qwen3.5 GDN algorithm: SiLU is applied to the entire QKV after conv1d.
        // ================================================================
        {
            let pso_silu = pipelines.silu_inplace.as_ref().ok_or_else(|| {
                RuntimeError::Compute("silu_inplace pipeline not compiled".into())
            })?;
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for SiLU".into())
            })?;
            enc.set_pipeline_state(pso_silu);
            enc.set_buffer(qkv_conv_buf, 0, 0);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 1);
            let silu_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(silu_tg), 1, 1),
                MTLSize::new(silu_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Step 4a: alpha_raw = Q8_0 matvec(ssm_alpha.weight, normed_buf)
        //   ssm_alpha.weight: [32(out), 2048(in)] Q8_0
        //   Input: normed_buf [2048 = hidden_dim]
        //   Output: alpha_raw_buf [32 = num_heads]
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for alpha_raw matvec".into())
            })?;
            enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg);
            enc.set_buffer(layer_buf, ssm_alpha_off, 0);   // weight Q8_0 [32, 2048]
            enc.set_buffer(&s.normed_buf, 0, 1);            // input [2048]
            enc.set_buffer(alpha_raw_buf, 0, 2);             // output [32]
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);  // in_dim = 2048
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // out_dim = 32
            let n_tg = ((num_heads as u64) + 7) / 8;
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
        }

        // ================================================================
        // Step 4b: beta_raw = Q8_0 matvec(ssm_beta.weight, normed_buf)
        //   ssm_beta.weight: [32(out), 2048(in)] Q8_0
        //   Input: normed_buf [2048 = hidden_dim]
        //   Output: beta_raw_buf [32 = num_heads]
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for beta_raw matvec".into())
            })?;
            enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg);
            enc.set_buffer(layer_buf, ssm_beta_off, 0);    // weight Q8_0 [32, 2048]
            enc.set_buffer(&s.normed_buf, 0, 1);            // input [2048]
            enc.set_buffer(beta_raw_buf, 0, 2);              // output [32]
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);  // in_dim = 2048
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // out_dim = 32
            let n_tg = ((num_heads as u64) + 7) / 8;
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
        }

        // ================================================================
        // Step 4c: Compute gates (alpha, beta) from SSM parameters
        //   dt = softplus(ssm_dt_bias)
        //   alpha = exp(-exp(ssm_a) * dt) * sigmoid(alpha_raw)
        //   beta = sigmoid(beta_raw)
        // Now using F32 matvec outputs instead of raw Q8_0 weight bytes.
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for compute_gates".into())
            })?;
            enc.set_pipeline_state(pso_compute_gates);
            enc.set_buffer(layer_buf, ssm_dt_off, 0);      // ssm_dt_bias [32] F32
            enc.set_buffer(layer_buf, ssm_a_off, 1);        // ssm_a [32] F32
            enc.set_buffer(beta_raw_buf, 0, 2);              // beta_raw [32] F32 (from matvec)
            enc.set_buffer(alpha_raw_buf, 0, 3);             // alpha_raw [32] F32 (from matvec)
            enc.set_buffer(alpha_buf, 0, 4);                  // alpha output [32]
            enc.set_buffer(beta_buf, 0, 5);                   // beta output [32]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
            let gates_tg = 256u64.min(num_heads as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((num_heads as u64).div_ceil(gates_tg), 1, 1),
                MTLSize::new(gates_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Step 5: L2-normalize Q and K per-head.
        // Q: in qkv_conv_buf at offset 0, num_kv_heads=16 heads of head_dim=128
        // K: in qkv_conv_buf at k_byte_off, num_kv_heads=16 heads of head_dim=128
        // Both Q and K have num_kv_heads=16 heads from the conv output.
        // The GDN kernel applies them as GQA (repeated to num_heads=32 virtually).
        // L2 normalize operates in-place.
        // ================================================================
        {
            // Normalize Q (16 pre-repeat heads, 128 dim each)
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for L2 norm Q".into())
            })?;
            enc.set_pipeline_state(pso_l2_norm);
            enc.set_buffer(qkv_conv_buf, 0, 0);                // Q at offset 0
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 1);  // 16 heads (not 32)
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 2);
            let l2_eps: f32 = 1e-12;
            enc.set_bytes(&l2_eps.to_le_bytes(), 3);
            let l2_tg = 256u64.min(head_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new(num_kv_heads as u64, 1, 1),  // 16 groups
                MTLSize::new(l2_tg, 1, 1),
            );
            enc.end_encoding();
        }
        {
            // Normalize K (16 KV heads, 128 dim each)
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for L2 norm K".into())
            })?;
            enc.set_pipeline_state(pso_l2_norm);
            enc.set_buffer(qkv_conv_buf, k_byte_off, 0);       // K at k_byte_off
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 1);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 2);
            let l2_eps: f32 = 1e-12;
            enc.set_bytes(&l2_eps.to_le_bytes(), 3);
            let l2_tg = 256u64.min(head_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(l2_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Steps 6+7+8 (fused): Delta-rule StateUpdate + Output + SSMNorm
        // Q: num_kv_heads=16 pre-repeat heads (kernel tile: q[h] = q_norm[h * n_kv / n_heads])
        // K: num_kv_heads=16 heads (kernel tile: k[h] = k_norm[h * n_kv / n_heads])
        // V: num_heads=32 heads directly (kernel uses h for V, no GQA)
        // ================================================================
        {
            let pso_fused = pipelines.gdn_state_output_norm.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_state_output_norm pipeline not compiled".into())
            })?;
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for state+output+norm".into())
            })?;
            enc.set_pipeline_state(pso_fused);
            enc.set_buffer(h_state_buf, 0, 0);            // h_state [n_heads=32 * val_dim * key_dim] (transposed layout)
            enc.set_buffer(qkv_conv_buf, k_byte_off, 1);  // k_norm [n_kv_heads=16 * key_dim]
            enc.set_buffer(qkv_conv_buf, v_byte_off, 2);  // v_tokens [n_heads=32 * val_dim]
            enc.set_buffer(alpha_buf, 0, 3);               // alpha [n_heads=32]
            enc.set_buffer(beta_buf, 0, 4);                // beta [n_heads=32]
            enc.set_buffer(qkv_conv_buf, 0, 5);            // q_norm [n_kv_heads=16 * key_dim]
            enc.set_buffer(layer_buf, ssm_norm_off, 6);    // scale [1 * val_dim] (shared)
            enc.set_buffer(normed_out_buf, 0, 7);          // output [n_heads=32 * val_dim]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);   // key_dim = 128
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 10);  // val_dim = 128
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 11);  // n_kv_heads=16 for Q/K GQA
            enc.set_bytes(&eps.to_le_bytes(), 12);  // RMSNorm eps (model norm_eps)
            enc.set_bytes(&(1u32).to_le_bytes(), 13);  // scale_n_heads=1: ssm_norm is [128], shared
            {
                // One threadgroup per state-head (32), one thread per val_dim element
                let tg = head_dim as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            }
            enc.end_encoding();
        }

        // ================================================================
        // Step 9: Attention gate: gate[4096] = sigmoid(attn_gate_weight * x_norm[2048])
        //   attn_gate_weight shape: [q_dim(4096), hidden_dim(2048)]
        //   Input: x_norm (s.normed_buf) [hidden_dim(2048)]
        //   Output: gate_sigmoid_buf [q_dim(4096)]
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for attn_gate matmul".into())
            })?;
            let tg = match attn_gate_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg); 64u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                QuantScheme::Bf16 => { enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, attn_gate_off, 0);      // weights [q_dim x hidden_dim]
            enc.set_buffer(&s.normed_buf, 0, 1);               // input x_norm [hidden_dim=2048]
            enc.set_buffer(gate_sigmoid_buf, 0, 2);             // output [q_dim=4096]
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);  // in_dim = hidden_dim = 2048
            if matches!(attn_gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) {
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);   // out_dim = q_dim = 4096
            }
            let n_tg = match attn_gate_quant { QuantScheme::Q8_0 => ((q_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((q_dim as u64) + 1) / 2, QuantScheme::F16 => ((q_dim as u64) + 1) / 2, QuantScheme::Bf16 => ((q_dim as u64) + 1) / 2, _ => q_dim as u64 };
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
        }
        // Apply SiLU-gated output: output[i] = silu(gate[i]) * normed_out[i]
        // = gate[i] * sigmoid(gate[i]) * normed_out[i]
        {
            let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
            })?;
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for SiLU-gated multiply".into())
            })?;
            enc.set_pipeline_state(pso_silu_mul);
            enc.set_buffer(gate_sigmoid_buf, 0, 0);    // gate = raw gate [4096] -> sigmoid applied
            enc.set_buffer(normed_out_buf, 0, 1);       // x = normed GDN output [4096]
            enc.set_buffer(gate_sigmoid_buf, 0, 2);     // output overwrites gate_sigmoid_buf [4096]
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
            let mul_tg = 256u64.min(q_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((q_dim as u64).div_ceil(mul_tg), 1, 1),
                MTLSize::new(mul_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Step 11: SSM output projection: ssm_proj_buf = ssm_out_weight * gate_sigmoid_buf
        //   ssm_out_weight shape: [hidden_dim(2048), q_dim(4096)] F32/Q4_0
        //   Input: gate_sigmoid_buf [q_dim(4096)] (gated normed output)
        //   Output: ssm_proj_buf [hidden_dim(2048)]
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for ssm_out proj".into())
            })?;
            let tg = match ssm_out_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg); 64u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                QuantScheme::Bf16 => { enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, ssm_out_off, 0);          // weights
            enc.set_buffer(gate_sigmoid_buf, 0, 1);              // input [4096] (gated)
            enc.set_buffer(ssm_proj_buf, 0, 2);                  // output [2048]
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);     // in_dim = q_dim = 4096
            if matches!(ssm_out_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) {
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);  // out_dim = 2048
            }
            let n_tg = match ssm_out_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Bf16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
        }

        // ================================================================
        // Step 12: Residual: x_buf += ssm_proj_buf
        // ================================================================
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create encoder for residual".into())
            })?;
            enc.set_pipeline_state(&pipelines.add_residual);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(ssm_proj_buf, 0, 1);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
            let res_tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(res_tg), 1, 1),
                MTLSize::new(res_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ================================================================
        // Step 13: Copy x_buf -> attn_proj_buf for FFN norm compatibility
        // The FFN RMSNorm reads from attn_proj_buf. After the residual,
        // x_buf contains the correct post-attention state.
        // ================================================================
        {
            let blit = cmd.new_blit_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN: failed to create blit encoder".into())
            })?;
            blit.copy_from_buffer(&s.x_buf, 0, &s.attn_proj_buf, 0, (hidden_dim * 4) as u64);
            blit.end_encoding();
        }

        Ok(new_conv_pos)
    }

    /// Fused single-encoder variant of encode_gdn_layer_decode.
    ///
    /// All dispatches go through the caller-provided encoder (no encoder
    /// create/end cycles). The blit copy is replaced with a compute
    /// copy_buffer dispatch to avoid forcing an encoder boundary.
    pub(crate) fn encode_gdn_layer_decode_fused(
        enc: &MetalComputeEncoder,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        meta: &CachedLayerMeta,
        gdn_idx: usize,
    ) -> Result<u32, RuntimeError> {
        let hidden_dim = s.hidden_dim;
        // Resolved SSM dims (9B {32,16,128,4} default, 27B {48,16,128,4}).
        // Q and K each have num_k_heads pre-repeat heads; V has num_v_heads heads.
        // qkv_dim = 2*qk_dim + value_dim (9B = 8192, 27B = 10240).
        let num_heads = s.gdn_num_v_heads;    // num_v_heads = state/V heads
        let num_kv_heads = s.gdn_num_k_heads; // num_k_heads = Q and K pre-repeat heads
        let head_dim = s.gdn_head_dim;
        let qk_dim = num_kv_heads * head_dim; // Q and K each: num_k_heads * head_dim
        let value_dim = num_heads * head_dim; // V: num_v_heads * head_dim (= inner_size)
        let q_dim = value_dim;                // alias for gate/output projection size
        let qkv_dim = 2 * qk_dim + value_dim;
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;
        let conv_kernel_size = s.gdn_conv_kernel_size;

        // Layout: Q[0..qk_dim) | K[qk_dim..2*qk_dim) | V[2*qk_dim..)
        let _q_byte_off: u64 = 0;
        let k_byte_off: u64 = (qk_dim as u64) * 4;             // 8192 bytes
        let v_byte_off: u64 = ((qk_dim + qk_dim) as u64) * 4;  // 16384 bytes

        let ssm_conv1d_off = meta.ssm_conv1d_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_conv1d_off".into())
        })?;
        let ssm_dt_off = meta.ssm_dt_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_dt_off".into())
        })?;
        let ssm_a_off = meta.ssm_a_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_a_off".into())
        })?;
        let ssm_beta_off = meta.ssm_beta_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_beta_off".into())
        })?;
        let ssm_alpha_off = meta.ssm_alpha_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_alpha_off".into())
        })?;
        let ssm_norm_off = meta.ssm_norm_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_norm_off".into())
        })?;
        let ssm_out_off = meta.ssm_out_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_out_off".into())
        })?;
        let ssm_out_quant = meta.ssm_out_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_out_quant. Re-convert model.".into())
        })?;
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_quant. Re-convert model.".into())
        })?;
        let attn_norm_off = meta.attn_norm_off;

        // Prefer fused Conv1D+SiLU (eliminates 1 dispatch + 1 barrier), fall back to separate.
        let pso_conv1d_silu = pipelines.ssm_conv1d_silu_decode.as_ref();
        let pso_conv1d = pipelines.ssm_conv1d_decode.as_ref().ok_or_else(|| {
            RuntimeError::Compute("ssm_conv1d_decode pipeline not compiled".into())
        })?;
        // Prefer fused dual alpha+beta+gates kernel (eliminates 2 dispatches + 1 barrier).
        let pso_dual_gates = pipelines.dequant_matmul_q8_0_dual_gates_nr2.as_ref();
        let pso_compute_gates = pipelines.gdn_compute_gates.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_compute_gates pipeline not compiled".into())
        })?;
        // Fused pipelines
        let pso_l2_norm_qk = pipelines.l2_normalize_qk.as_ref().ok_or_else(|| {
            RuntimeError::Compute("l2_normalize_qk pipeline not compiled".into())
        })?;
        // Fused StateUpdate+Output+Norm mega-kernel
        let pso_state_output_norm = pipelines.gdn_state_output_norm.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_state_output_norm pipeline not compiled".into())
        })?;
        // Fused L2+state+output+norm (eliminates l2_normalize_qk dispatch + barrier)
        let pso_state_output_norm_l2 = pipelines.gdn_state_output_norm_l2.as_ref();
        // Fused Conv1D+SiLU + L2 + state + output + norm (eliminates conv1d dispatch + barrier)
        let pso_conv_l2_state = pipelines.gdn_state_output_norm_l2_conv.as_ref();
        // Full GDN decode megakernel: Conv+SiLU + inline gates + L2 + state + output + norm
        let pso_megakernel = pipelines.gdn_decode_megakernel.as_ref();
        // Simdgroup-parallel GDN state update (high-occupancy variant)
        let pso_state_l2_sg = pipelines.gdn_state_output_l2_sg.as_ref();
        let pso_norm_scale = pipelines.gdn_decode_norm_scale.as_ref();
        // Fused SiLU-gated matvec+residual+copy (eliminates silu_elementwise_mul dispatch + barrier)
        let pso_silu_matvec = pipelines.dequant_matmul_q8_0_silu_deferred_residual_copy_nr2.as_ref();
        let pso_silu_matvec_q4 = pipelines.dequant_matmul_q4_0_silu_deferred_residual_copy_nr2.as_ref();

        let alpha_buf = s.gdn_alpha_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_alpha_buf not allocated".into())
        })?;
        let beta_buf = s.gdn_beta_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_beta_buf not allocated".into())
        })?;
        // gdn_out_buf used as raw output buffer by gdn_state_output_l2_sg (pre-norm)
        let gdn_out_buf = s.gdn_output_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_output_buf not allocated".into())
        })?;
        let ssm_proj_buf = s.gdn_ssm_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_ssm_proj_buf not allocated".into())
        })?;
        let gate_sigmoid_buf = s.gdn_gate_sigmoid_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_gate_sigmoid_buf not allocated".into())
        })?;
        let normed_out_buf = s.gdn_normed_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_normed_out_buf not allocated".into())
        })?;
        let alpha_raw_buf = s.gdn_alpha_raw_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_alpha_raw_buf not allocated".into())
        })?;
        let beta_raw_buf = s.gdn_beta_raw_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_beta_raw_buf not allocated".into())
        })?;
        let qkv_conv_buf = s.gdn_qkv_conv_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("gdn_qkv_conv_buf not allocated".into())
        })?;

        let h_state_buf = &s.gdn_h_states[gdn_idx];
        let conv_state_buf = &s.gdn_conv_states[gdn_idx];
        let conv_pos = s.gdn_conv_positions[gdn_idx];

        // === GROUP 0 (Q8_0 only): Separate RMSNorm ===
        // For Q8_0, unfuse RMSNorm from matvecs: a single RMSNorm dispatch writes normed_buf,
        // then all Q8_0 matvecs in Group 1 read normed_buf. This eliminates redundant norm weight
        // loads (32KB per TG * 6144 TGs = 192MB per GDN layer * 24 layers = 4.6GB per token).
        // Q4_0 keeps fused RMSNorm+matvec (different kernel, different tradeoff).
        let q8_unfused = matches!(meta.wq_quant, QuantScheme::Q8_0);
        if q8_unfused {
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(norm_tg_size, 1, 1));
        }

        // === PARALLEL GROUP 1 ===
        // Q8_0: matvecs read normed_buf (RMSNorm already done above).
        // Q4_0: fused RMSNorm+matvec kernels read x_buf + norm weights inline.
        // Alpha/beta+gates: always fused (only 16 TGs, overhead negligible).

        // Step 1+2: QKV matvec
        {
            let wq_off = meta.wq_off;
            if q8_unfused {
                // Q8_0 unfused: read normed_buf, use dequant_matmul_q8_0_2sg (64 threads, 8 rows/TG)
                enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg);
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.qkv_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                let n_tg = ((qkv_dim as u64) + 7) / 8;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(64, 1, 1));
            } else if matches!(meta.wq_quant, QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) {
                // Q4_0/F16/BF16 fused: RMSNorm + matvec in one kernel
                match meta.wq_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                    QuantScheme::Bf16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_bf16_deferred_nr2),
                    _ => unreachable!(),
                }
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(&s.qkv_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                enc.set_buffer(layer_buf, attn_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                let n_tg = ((qkv_dim as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            } else {
                // Fallback: separate RMSNorm + matvec (non-quantized)
                enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                enc.set_buffer(&s.x_buf, 0, 0);
                enc.set_buffer(layer_buf, attn_norm_off, 1);
                enc.set_buffer(&s.normed_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(norm_tg_size, 1, 1));
                let tg = match meta.wq_quant {
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); s.matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.qkv_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if meta.wq_quant == QuantScheme::Q4_0 { enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4); }
                let n_tg = match meta.wq_quant { QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2, _ => qkv_dim as u64 };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            }
        }

        // Step 1+4a+4b+4c (fused): RMSNorm + dual alpha/beta matvec + gate transforms
        // Fused kernel reads x_buf ONCE, computes both alpha_raw and beta_raw dot products,
        // then applies gate transforms inline (softplus+exp for alpha, sigmoid for beta).
        // Eliminates 2 dispatches (separate alpha/beta matvecs) and the compute_gates dispatch.
        // KEEP fused even for Q8_0: only 16 TGs, redundant norm load overhead negligible.
        if let Some(pso_dual) = pso_dual_gates {
            enc.set_pipeline_state(pso_dual);
            enc.set_buffer(layer_buf, ssm_alpha_off, 0);  // alpha weights Q8_0 [num_heads, hidden_dim]
            enc.set_buffer(layer_buf, ssm_beta_off, 1);   // beta weights Q8_0 [num_heads, hidden_dim]
            enc.set_buffer(&s.x_buf, 0, 2);               // input [hidden_dim]
            enc.set_buffer(alpha_buf, 0, 3);               // alpha output [num_heads] (post-gates)
            enc.set_buffer(beta_buf, 0, 4);                // beta output [num_heads] (post-gates)
            enc.set_buffer(layer_buf, attn_norm_off, 5);   // RMSNorm weights
            enc.set_buffer(layer_buf, ssm_a_off, 6);       // ssm_a [num_heads] = -exp(A_log)
            enc.set_buffer(layer_buf, ssm_dt_off, 7);      // dt_bias [num_heads]
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 8);  // K = hidden_dim
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 9);   // N = num_heads
            enc.set_bytes(&eps.to_le_bytes(), 10);
            let n_tg = ((num_heads as u64) + 1) / 2;
            enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
        } else {
            // Fallback: separate alpha_raw + beta_raw matvecs (compute_gates in Group 2)
            enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2);
            enc.set_buffer(layer_buf, ssm_alpha_off, 0);
            enc.set_buffer(&s.x_buf, 0, 1);
            enc.set_buffer(alpha_raw_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, attn_norm_off, 5);
            enc.set_bytes(&eps.to_le_bytes(), 6);
            {
                let n_tg = ((num_heads as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            }

            enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2);
            enc.set_buffer(layer_buf, ssm_beta_off, 0);
            enc.set_buffer(&s.x_buf, 0, 1);
            enc.set_buffer(beta_raw_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, attn_norm_off, 5);
            enc.set_bytes(&eps.to_le_bytes(), 6);
            {
                let n_tg = ((num_heads as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            }
        }

        // Step 1+9: Attn gate matvec
        {
            if q8_unfused {
                // Q8_0 unfused: read normed_buf, use dequant_matmul_q8_0_2sg (64 threads, 8 rows/TG)
                enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg);
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(gate_sigmoid_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);
                let n_tg = ((q_dim as u64) + 7) / 8;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(64, 1, 1));
            } else if matches!(attn_gate_quant, QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) {
                // Q4_0/F16/BF16 fused: RMSNorm + matvec in one kernel
                match attn_gate_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                    QuantScheme::Bf16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_bf16_deferred_nr2),
                    _ => unreachable!(),
                }
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(gate_sigmoid_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);
                enc.set_buffer(layer_buf, attn_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                let n_tg = ((q_dim as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            } else {
                // Non-quantized fallback: gate reads normed_buf (needs separate RMSNorm)
                // Only reached if the non-fused QKV path above already wrote normed_buf.
                let tg = match attn_gate_quant {
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); s.matmul_tg_size },
                };
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(gate_sigmoid_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if attn_gate_quant == QuantScheme::Q4_0 { enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4); }
                let n_tg = match attn_gate_quant { QuantScheme::Q4_0 => ((q_dim as u64) + 1) / 2, _ => q_dim as u64 };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
            }
        }

        // Barrier: QKV -> qkv_buf, alpha/beta -> alpha/beta_buf (fused) or alpha_raw/beta_raw (fallback),
        // gate -> gate_sigmoid all ready

        // === GROUP 2: Conv1D + L2 + StateUpdate + Output + Norm ===
        //
        // Tier 0 (best): Conv1D+SiLU (separate) + l2_sg (4096 TGs) + norm_scale
        //   Uses simdgroup-parallel state update for higher GPU occupancy.
        //
        // Tier 1: gdn_decode_megakernel (1 dispatch) -- when dual_gates unavailable.
        // Tier 2: gdn_state_output_norm_l2_conv (1 dispatch) -- conv-fused fallback.
        // Tier 3: separate Conv1D+SiLU then L2+State+Output+Norm (multiple dispatches).

        let buf_slots = (conv_kernel_size - 1) as u32;
        let new_conv_pos = (conv_pos + 1) % buf_slots;

        let use_high_occupancy = pso_state_l2_sg.is_some() && pso_norm_scale.is_some();

        if use_high_occupancy {
            // === Tier 0: HIGH-OCCUPANCY path ===
            // Conv1D+SiLU (separate) -> simdgroup state update -> norm_scale
            // Gates: need post-gate alpha/beta. If dual_gates was NOT used, compute_gates now.
            if pso_dual_gates.is_none() {
                enc.set_pipeline_state(pso_compute_gates);
                enc.set_buffer(layer_buf, ssm_dt_off, 0);
                enc.set_buffer(layer_buf, ssm_a_off, 1);
                enc.set_buffer(beta_raw_buf, 0, 2);
                enc.set_buffer(alpha_raw_buf, 0, 3);
                enc.set_buffer(alpha_buf, 0, 4);
                enc.set_buffer(beta_buf, 0, 5);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
                {
                    let gates_tg = 256u64.min(num_heads as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((num_heads as u64).div_ceil(gates_tg), 1, 1),
                        MTLSize::new(gates_tg, 1, 1),
                    );
                }
            }

            // Conv1D+SiLU on all QKV channels (separate dispatch, reads qkv_buf -> qkv_conv_buf)
            if let Some(pso_fused) = pso_conv1d_silu {
                enc.set_pipeline_state(pso_fused);
            } else {
                enc.set_pipeline_state(pso_conv1d);
            }
            enc.set_buffer(&s.qkv_buf, 0, 0);
            enc.set_buffer(conv_state_buf, 0, 1);
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2);
            enc.set_buffer(qkv_conv_buf, 0, 3);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            let conv_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), 1, 1),
                MTLSize::new(conv_tg, 1, 1),
            );

            // Barrier: Conv1D+SiLU -> qkv_conv_buf, gates -> alpha/beta all ready

            if pso_conv1d_silu.is_none() {
                let pso_silu = pipelines.silu_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("silu_inplace pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_silu);
                enc.set_buffer(qkv_conv_buf, 0, 0);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 1);
                let silu_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(silu_tg), 1, 1),
                    MTLSize::new(silu_tg, 1, 1),
                );
            }

            // Simdgroup-parallel state update with high GPU occupancy.
            // Each simdgroup (32 threads) handles one (head, val_col) pair via simd_sum.
            let pso_sg = pso_state_l2_sg.unwrap();
            enc.set_pipeline_state(pso_sg);
            enc.set_buffer(h_state_buf, 0, 0);              // h_state [n_heads * val_dim * key_dim]
            enc.set_buffer(qkv_conv_buf, k_byte_off, 1);    // k_raw [n_kv_heads * key_dim] (UN-normalized)
            enc.set_buffer(qkv_conv_buf, v_byte_off, 2);    // v_tokens [n_heads * val_dim]
            enc.set_buffer(alpha_buf, 0, 3);                 // alpha [n_heads]
            enc.set_buffer(beta_buf, 0, 4);                  // beta [n_heads]
            enc.set_buffer(qkv_conv_buf, 0, 5);              // q_raw [n_kv_heads * key_dim] (UN-normalized)
            enc.set_buffer(gdn_out_buf, 0, 6);               // raw_out [n_heads * val_dim] (pre-norm)
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 8);   // key_dim = 128
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);   // val_dim = 128
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 10);
            {
                let l2_eps: f32 = 1e-12;
                enc.set_bytes(&l2_eps.to_le_bytes(), 11);
                // grid: (1, val_dim=128, n_heads=32) = 4096 TGs, threadgroup: (32, 1, 1)
                enc.dispatch_threadgroups(
                    MTLSize::new(1, head_dim as u64, num_heads as u64),
                    MTLSize::new(32, 1, 1),
                );
            }

            // RMSNorm + learned scale on raw output -> normed_out_buf
            let pso_ns = pso_norm_scale.unwrap();
            enc.set_pipeline_state(pso_ns);
            enc.set_buffer(gdn_out_buf, 0, 0);              // raw_out [n_heads * val_dim]
            enc.set_buffer(layer_buf, ssm_norm_off, 1);      // scale [val_dim]
            enc.set_buffer(normed_out_buf, 0, 2);            // output [n_heads * val_dim]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4); // val_dim = 128
            enc.set_bytes(&eps.to_le_bytes(), 5);
            enc.set_bytes(&(1u32).to_le_bytes(), 6);         // scale_n_heads = 1
            enc.dispatch_threadgroups(
                MTLSize::new(num_heads as u64, 1, 1),
                MTLSize::new(head_dim as u64, 1, 1),
            );
        } else if pso_dual_gates.is_none() && pso_megakernel.is_some() {
            // Tier 1: Full megakernel (32 TGs) - inline gates from alpha_raw/beta_raw
            let pso = pso_megakernel.unwrap();
            enc.set_pipeline_state(pso);
            enc.set_buffer(h_state_buf, 0, 0);              // h_state [n_heads * val_dim * key_dim]
            enc.set_buffer(&s.qkv_buf, 0, 1);               // qkv_raw [qkv_dim] from matvec (pre-conv)
            enc.set_buffer(conv_state_buf, 0, 2);            // conv_state circular buffer (R/W)
            enc.set_buffer(layer_buf, ssm_conv1d_off, 3);    // conv_weight [qkv_dim * kernel_size]
            enc.set_buffer(alpha_raw_buf, 0, 4);             // alpha_raw [n_heads] (pre-gate)
            enc.set_buffer(beta_raw_buf, 0, 5);              // beta_raw [n_heads] (pre-gate)
            enc.set_buffer(layer_buf, ssm_a_off, 6);         // ssm_a [n_heads]
            enc.set_buffer(layer_buf, ssm_dt_off, 7);        // dt_bias [n_heads]
            enc.set_buffer(layer_buf, ssm_norm_off, 8);      // scale [val_dim]
            enc.set_buffer(normed_out_buf, 0, 9);            // output [n_heads * val_dim]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 10);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 11);   // key_dim = 128
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 12);   // val_dim = 128
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 13);
            enc.set_bytes(&eps.to_le_bytes(), 14);           // RMSNorm eps
            enc.set_bytes(&(1u32).to_le_bytes(), 15);        // scale_n_heads = 1
            {
                let l2_eps: f32 = 1e-12;
                enc.set_bytes(&l2_eps.to_le_bytes(), 16);    // L2 norm eps
                enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 17);   // qk_dim = 2048
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 18);  // qkv_dim = 8192
                enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 19);
                enc.set_bytes(&conv_pos.to_le_bytes(), 20);  // state_pos
                let tg = head_dim as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            }
        } else if pso_conv_l2_state.is_some() {
            // Tier 2: Conv-fused kernel (32 TGs) - needs post-gate alpha/beta
            if pso_dual_gates.is_none() {
                enc.set_pipeline_state(pso_compute_gates);
                enc.set_buffer(layer_buf, ssm_dt_off, 0);
                enc.set_buffer(layer_buf, ssm_a_off, 1);
                enc.set_buffer(beta_raw_buf, 0, 2);
                enc.set_buffer(alpha_raw_buf, 0, 3);
                enc.set_buffer(alpha_buf, 0, 4);
                enc.set_buffer(beta_buf, 0, 5);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
                {
                    let gates_tg = 256u64.min(num_heads as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((num_heads as u64).div_ceil(gates_tg), 1, 1),
                        MTLSize::new(gates_tg, 1, 1),
                    );
                }
            }

            let pso = pso_conv_l2_state.unwrap();
            enc.set_pipeline_state(pso);
            enc.set_buffer(h_state_buf, 0, 0);              // h_state [n_heads * val_dim * key_dim]
            enc.set_buffer(&s.qkv_buf, 0, 1);               // qkv_raw [qkv_dim] from matvec (pre-conv)
            enc.set_buffer(conv_state_buf, 0, 2);            // conv_state circular buffer (R/W)
            enc.set_buffer(layer_buf, ssm_conv1d_off, 3);    // conv_weight [qkv_dim * kernel_size]
            enc.set_buffer(alpha_buf, 0, 4);                 // alpha [n_heads] (post-gates)
            enc.set_buffer(beta_buf, 0, 5);                  // beta [n_heads] (post-gates)
            enc.set_buffer(layer_buf, ssm_norm_off, 6);      // scale [val_dim] (shared)
            enc.set_buffer(normed_out_buf, 0, 7);            // output [n_heads * val_dim]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);    // key_dim = 128
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 10);   // val_dim = 128
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 11);
            enc.set_bytes(&eps.to_le_bytes(), 12);           // RMSNorm eps
            enc.set_bytes(&(1u32).to_le_bytes(), 13);        // scale_n_heads = 1
            {
                let l2_eps: f32 = 1e-12;
                enc.set_bytes(&l2_eps.to_le_bytes(), 14);    // L2 norm eps
                enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 15);   // qk_dim = 2048
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 16);  // qkv_dim = 8192
                enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 17);
                enc.set_bytes(&conv_pos.to_le_bytes(), 18);  // state_pos
                let tg = head_dim as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            }
        } else {
            // Tier 3 fallback: separate Conv1D+SiLU then L2+State+Output+Norm
            if pso_dual_gates.is_none() {
                enc.set_pipeline_state(pso_compute_gates);
                enc.set_buffer(layer_buf, ssm_dt_off, 0);
                enc.set_buffer(layer_buf, ssm_a_off, 1);
                enc.set_buffer(beta_raw_buf, 0, 2);
                enc.set_buffer(alpha_raw_buf, 0, 3);
                enc.set_buffer(alpha_buf, 0, 4);
                enc.set_buffer(beta_buf, 0, 5);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
                {
                    let gates_tg = 256u64.min(num_heads as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((num_heads as u64).div_ceil(gates_tg), 1, 1),
                        MTLSize::new(gates_tg, 1, 1),
                    );
                }
            }

            if let Some(pso_fused) = pso_conv1d_silu {
                enc.set_pipeline_state(pso_fused);
            } else {
                enc.set_pipeline_state(pso_conv1d);
            }
            enc.set_buffer(&s.qkv_buf, 0, 0);
            enc.set_buffer(conv_state_buf, 0, 1);
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2);
            enc.set_buffer(qkv_conv_buf, 0, 3);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            let conv_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), 1, 1),
                MTLSize::new(conv_tg, 1, 1),
            );

            // Barrier: Conv1D+SiLU -> qkv_conv_buf, gates -> alpha/beta all ready

            if pso_conv1d_silu.is_none() {
                let pso_silu = pipelines.silu_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("silu_inplace pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_silu);
                enc.set_buffer(qkv_conv_buf, 0, 0);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 1);
                let silu_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(silu_tg), 1, 1),
                    MTLSize::new(silu_tg, 1, 1),
                );
            }

            if pso_state_l2_sg.is_some() && pso_norm_scale.is_some() {
                // High-occupancy path: simdgroup-parallel state update + separate norm.
                // Each simdgroup (32 threads) handles one (head, val_col) pair via simd_sum
                // reductions with state in registers.
                let pso_sg = pso_state_l2_sg.unwrap();
                enc.set_pipeline_state(pso_sg);
                enc.set_buffer(h_state_buf, 0, 0);              // h_state [n_heads * val_dim * key_dim]
                enc.set_buffer(qkv_conv_buf, k_byte_off, 1);    // k_raw [n_kv_heads * key_dim] (UN-normalized)
                enc.set_buffer(qkv_conv_buf, v_byte_off, 2);    // v_tokens [n_heads * val_dim]
                enc.set_buffer(alpha_buf, 0, 3);                 // alpha [n_heads]
                enc.set_buffer(beta_buf, 0, 4);                  // beta [n_heads]
                enc.set_buffer(qkv_conv_buf, 0, 5);              // q_raw [n_kv_heads * key_dim] (UN-normalized)
                enc.set_buffer(gdn_out_buf, 0, 6);               // raw_out [n_heads * val_dim] (pre-norm)
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 8);   // key_dim = 128
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);   // val_dim = 128
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 10);
                {
                    let l2_eps: f32 = 1e-12;
                    enc.set_bytes(&l2_eps.to_le_bytes(), 11);
                    // grid: (1, val_dim=128, n_heads=32) = 4096 TGs, threadgroup: (32, 1, 1)
                    enc.dispatch_threadgroups(
                        MTLSize::new(1, head_dim as u64, num_heads as u64),
                        MTLSize::new(32, 1, 1),
                    );
                }

                // RMSNorm + learned scale on raw output -> normed_out_buf
                let pso_ns = pso_norm_scale.unwrap();
                enc.set_pipeline_state(pso_ns);
                enc.set_buffer(gdn_out_buf, 0, 0);              // raw_out [n_heads * val_dim]
                enc.set_buffer(layer_buf, ssm_norm_off, 1);      // scale [val_dim]
                enc.set_buffer(normed_out_buf, 0, 2);            // output [n_heads * val_dim]
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4); // val_dim = 128
                enc.set_bytes(&eps.to_le_bytes(), 5);
                enc.set_bytes(&(1u32).to_le_bytes(), 6);         // scale_n_heads = 1
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );
            } else if let Some(pso_l2_fused) = pso_state_output_norm_l2 {
                enc.set_pipeline_state(pso_l2_fused);
                enc.set_buffer(h_state_buf, 0, 0);
                enc.set_buffer(qkv_conv_buf, k_byte_off, 1);
                enc.set_buffer(qkv_conv_buf, v_byte_off, 2);
                enc.set_buffer(alpha_buf, 0, 3);
                enc.set_buffer(beta_buf, 0, 4);
                enc.set_buffer(qkv_conv_buf, 0, 5);
                enc.set_buffer(layer_buf, ssm_norm_off, 6);
                enc.set_buffer(normed_out_buf, 0, 7);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 8);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 10);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 11);
                enc.set_bytes(&eps.to_le_bytes(), 12);
                enc.set_bytes(&(1u32).to_le_bytes(), 13);
                {
                    let l2_eps: f32 = 1e-12;
                    enc.set_bytes(&l2_eps.to_le_bytes(), 14);
                    let tg = head_dim as u64;
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
            } else {
                enc.set_pipeline_state(pso_l2_norm_qk);
                enc.set_buffer(qkv_conv_buf, 0, 0);
                enc.set_buffer(qkv_conv_buf, k_byte_off, 1);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 2);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4);
                {
                    let l2_eps: f32 = 1e-12;
                    enc.set_bytes(&l2_eps.to_le_bytes(), 5);
                    let l2_tg = 256u64.min(head_dim as u64).max(1);
                    let total_heads = (num_kv_heads + num_kv_heads) as u64;
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_heads, 1, 1),
                        MTLSize::new(l2_tg, 1, 1),
                    );
                }

                enc.set_pipeline_state(pso_state_output_norm);
                enc.set_buffer(h_state_buf, 0, 0);
                enc.set_buffer(qkv_conv_buf, k_byte_off, 1);
                enc.set_buffer(qkv_conv_buf, v_byte_off, 2);
                enc.set_buffer(alpha_buf, 0, 3);
                enc.set_buffer(beta_buf, 0, 4);
                enc.set_buffer(qkv_conv_buf, 0, 5);
                enc.set_buffer(layer_buf, ssm_norm_off, 6);
                enc.set_buffer(normed_out_buf, 0, 7);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 8);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 10);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 11);
                enc.set_bytes(&eps.to_le_bytes(), 12);
                enc.set_bytes(&(1u32).to_le_bytes(), 13);
                {
                    let tg = head_dim as u64;
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
            }
        }


        // Steps 10+11+12+13: SiLU-gated output + SSMOut matvec + residual + copy
        // Prefer fused SiLU+matvec (eliminates silu_elementwise_mul dispatch + barrier).
        // gate_sigmoid_buf holds raw gate values; normed_out_buf holds GDN normed output.
        {
            match ssm_out_quant {
                QuantScheme::Q8_0 if pso_silu_matvec.is_some() => {
                    // Fused: silu(gate) * normed_out computed inline during matvec x-load
                    let pso = pso_silu_matvec.unwrap();
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);   // weights
                    enc.set_buffer(normed_out_buf, 0, 1);         // x = normed GDN output
                    enc.set_buffer(&s.x_buf, 0, 2);              // accum (R/W)
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);  // in_dim = 4096
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);       // copy_dst
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // out_dim = 2048
                    enc.set_buffer(gate_sigmoid_buf, 0, 6);       // gate values
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                }
                QuantScheme::Q8_0 => {
                    // Fallback: separate silu_mul then matvec
                    let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_silu_mul);
                    enc.set_buffer(gate_sigmoid_buf, 0, 0);
                    enc.set_buffer(normed_out_buf, 0, 1);
                    enc.set_buffer(gate_sigmoid_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(q_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );

                    let pso = pipelines.dequant_matmul_q8_0_deferred_residual_copy_nr2.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("dequant_matmul_q8_0_deferred_residual_copy_nr2 pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(&s.x_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                }
                QuantScheme::Q4_0 if pso_silu_matvec_q4.is_some() => {
                    // Fused: silu(gate) * normed_out computed inline during Q4_0 matvec x-load
                    let pso = pso_silu_matvec_q4.unwrap();
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);   // weights
                    enc.set_buffer(normed_out_buf, 0, 1);         // x = normed GDN output
                    enc.set_buffer(&s.x_buf, 0, 2);              // accum (R/W)
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);  // in_dim
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);       // copy_dst
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // out_dim
                    enc.set_buffer(gate_sigmoid_buf, 0, 6);       // gate values
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                }
                QuantScheme::Q4_0 => {
                    // Fallback: separate silu_mul then Q4_0 matvec
                    let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_silu_mul);
                    enc.set_buffer(gate_sigmoid_buf, 0, 0);
                    enc.set_buffer(normed_out_buf, 0, 1);
                    enc.set_buffer(gate_sigmoid_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(q_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );

                    let pso = pipelines.dequant_matmul_q4_0_deferred_residual_copy_nr2.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("dequant_matmul_q4_0_deferred_residual_copy_nr2 pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(&s.x_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                }
                QuantScheme::F16 => {
                    // F16: separate silu_mul then F16 matvec with residual
                    let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_silu_mul);
                    enc.set_buffer(gate_sigmoid_buf, 0, 0);
                    enc.set_buffer(normed_out_buf, 0, 1);
                    enc.set_buffer(gate_sigmoid_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(q_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );

                    // F16 matvec -> ssm_proj_buf, then manual residual+copy
                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(ssm_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));

                    // residual_add_copy: x_buf += ssm_proj_buf, attn_proj_buf = x_buf
                    let pso_residual_copy = pipelines.residual_add_copy.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("residual_add_copy pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_residual_copy);
                    enc.set_buffer(&s.x_buf, 0, 0);
                    enc.set_buffer(ssm_proj_buf, 0, 1);
                    enc.set_buffer(&s.attn_proj_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    let rc_tg = 256u64.min(hidden_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(rc_tg), 1, 1),
                        MTLSize::new(rc_tg, 1, 1),
                    );
                }
                QuantScheme::Bf16 => {
                    // BF16: separate silu_mul then BF16 matvec with residual (mirror F16)
                    let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_silu_mul);
                    enc.set_buffer(gate_sigmoid_buf, 0, 0);
                    enc.set_buffer(normed_out_buf, 0, 1);
                    enc.set_buffer(gate_sigmoid_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(q_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );

                    // BF16 matvec -> ssm_proj_buf
                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(ssm_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));

                    // residual_add_copy: x_buf += ssm_proj_buf, attn_proj_buf = x_buf
                    let pso_residual_copy = pipelines.residual_add_copy.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("residual_add_copy pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_residual_copy);
                    enc.set_buffer(&s.x_buf, 0, 0);
                    enc.set_buffer(ssm_proj_buf, 0, 1);
                    enc.set_buffer(&s.attn_proj_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    let rc_tg = 256u64.min(hidden_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(rc_tg), 1, 1),
                        MTLSize::new(rc_tg, 1, 1),
                    );
                }
                _ => {
                    // F32 fallback: separate silu_mul + matvec + residual_add_copy
                    let pso_silu_mul = pipelines.silu_elementwise_mul.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("silu_elementwise_mul pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_silu_mul);
                    enc.set_buffer(gate_sigmoid_buf, 0, 0);
                    enc.set_buffer(normed_out_buf, 0, 1);
                    enc.set_buffer(gate_sigmoid_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(q_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );

                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(ssm_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(hidden_dim as u64, 1, 1),
                        MTLSize::new(s.matmul_tg_size, 1, 1),
                    );

                    let pso_residual_copy = pipelines.residual_add_copy.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("residual_add_copy pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso_residual_copy);
                    enc.set_buffer(&s.x_buf, 0, 0);
                    enc.set_buffer(ssm_proj_buf, 0, 1);
                    enc.set_buffer(&s.attn_proj_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    let fused_tg = 256u64.min(hidden_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );
                }
            }
        }

        Ok(new_conv_pos)
    }

    /// Encode batched GDN prefill for T tokens.
    ///
    /// Replaces the token-by-token GDN prefill loop with batched GPU dispatches.
    /// All projections use batched GEMM; the recurrent state update is sequential
    /// inside the fused kernel (gdn_prefill_state_output_norm).
    ///
    /// Dispatch sequence (15 steps):
    ///  1. Batched RMSNorm: x_buf[T, 4096] -> normed_buf[T, 4096]
    ///  2. Batched QKV GEMM: normed_buf[T, 4096] x attn_qkv[4096, 8192] -> qkv_buf[T, 8192]
    ///  3. Batched Conv1D: qkv_buf[T, 8192] -> qkv_conv_buf[T, 8192] (fallback: per-token)
    ///  4. Batched SiLU: in-place on qkv_conv_buf[T * 8192]
    ///  5. Batched Alpha GEMM: normed_buf[T, 4096] x ssm_alpha[4096, 32] -> alpha_raw[T, 32]
    ///  6. Batched Beta GEMM: normed_buf[T, 4096] x ssm_beta[4096, 32] -> beta_raw[T, 32]
    ///  7. Batched Compute Gates: alpha_raw/beta_raw -> alpha/beta[T, 32]
    ///  8. Batched L2 Normalize Q: in qkv_conv_buf, 16 heads per token
    ///  9. Batched L2 Normalize K: in qkv_conv_buf, 16 heads per token
    /// 10. Fused GDN State+Output+Norm: sequential state update across T tokens
    /// 11. Batched Attn Gate GEMM: normed_buf[T, 4096] x attn_gate[4096, 4096] -> gate_buf[T, 4096]
    /// 12. Batched Sigmoid Gate Multiply: sigmoid(gate) * gdn_output -> gate_buf[T, 4096]
    /// 13. Batched SSM Out GEMM: gate_buf[T, 4096] x ssm_out[4096, 4096] -> proj_buf[T, 4096]
    /// 14. Batched Residual Add: x_buf[T, 4096] += proj_buf[T, 4096]
    /// 15. Copy x_buf -> attn_proj_buf (blit)
    ///
    /// Returns the final conv position for this GDN layer.
    ///
    /// `_outer_enc` is reserved for API symmetry with
    /// `encode_layer_batched_concurrent(cmd, Some(outer_enc), ...)`. The GDN
    /// prefill currently manages its own encoder lifecycle internally
    /// (the multi-phase RMSNorm + QKV + alpha/beta + conv1d + state
    /// update flow uses several discrete encoders separated by
    /// resource-scoped barriers), so the outer encoder cannot be
    /// shared today. Callers that own a whole-prefill outer encoder
    /// pass it through so the signature matches the concurrent-encoder path, and the
    /// future "GDN-into-outer-encoder" refactor can light it
    /// up without churning call sites.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batched_gdn_prefill(
        cmd: &MetalCommandBuffer,
        _outer_enc: Option<&MetalComputeEncoder>,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        meta: &CachedLayerMeta,
        gdn_idx: usize,
        x_buf: &MetalBuffer,       // [T, hidden_dim] batched input (R/W: residual written back)
        normed_buf: &MetalBuffer,  // [T, hidden_dim] batched RMSNorm output
        qkv_buf: &MetalBuffer,    // [T, qkv_dim] batched QKV matvec output
        attn_out_buf: &MetalBuffer, // [T, q_dim] reused for pre-computed attn gate
        scratch_buf: &MetalBuffer,  // [T, max(inter,q_dim)] scratch for batched alpha/beta raw
        attn_proj_buf: &MetalBuffer, // [T, hidden_dim] output for FFN norm
        batch_size: usize,
    ) -> Result<u32, RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let matmul_tg_size = s.matmul_tg_size;
        // Resolved SSM dims (9B {32,16,128,4} default, 27B {48,16,128,4}).
        // qkv_dim = 2*qk_dim + value_dim (9B = 8192, 27B = 10240).
        let num_heads = s.gdn_num_v_heads;
        let num_kv_heads = s.gdn_num_k_heads;
        let head_dim = s.gdn_head_dim;
        let qk_dim = num_kv_heads * head_dim;
        let value_dim = num_heads * head_dim;
        let q_dim = value_dim;
        let qkv_dim = 2 * qk_dim + value_dim;
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;
        let conv_kernel_size = s.gdn_conv_kernel_size;

        let ssm_conv1d_off = meta.ssm_conv1d_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_conv1d_off".into())
        })?;
        let ssm_dt_off = meta.ssm_dt_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_dt_off".into())
        })?;
        let ssm_a_off = meta.ssm_a_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_a_off".into())
        })?;
        let ssm_beta_off = meta.ssm_beta_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_beta_off".into())
        })?;
        let ssm_alpha_off = meta.ssm_alpha_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_alpha_off".into())
        })?;
        let ssm_norm_off = meta.ssm_norm_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_norm_off".into())
        })?;
        let ssm_out_off = meta.ssm_out_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing ssm_out_off".into())
        })?;
        let ssm_out_quant = meta.ssm_out_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing ssm_out_quant. Re-convert model.".into())
        })?;
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_quant. Re-convert model.".into())
        })?;
        let attn_norm_off = meta.attn_norm_off;

        let h_state_buf = &s.gdn_h_states[gdn_idx];
        let conv_state_buf = &s.gdn_conv_states[gdn_idx];
        let conv_pos = s.gdn_conv_positions[gdn_idx];

        let buf_slots = (conv_kernel_size - 1) as u32;
        let tok_bytes_hidden = (hidden_dim * 4) as u64;
        let tok_bytes_qkv = (qkv_dim * 4) as u64;
        let tok_bytes_qdim = (q_dim * 4) as u64;

        let gate_all_buf = attn_out_buf;

        // ================================================================
        // route GDN dispatches through de-aliased buffers + a
        // concurrent compute encoder + resource-scoped barriers so Apple's
        // whole-MTLBuffer hazard tracker can issue producer/consumer
        // chains in parallel where the dependencies allow.
        //
        // Activation conditions (ALL required):
        //   1. `graph_reorder::gdn_concurrent_encoder_enabled()` returns true (env
        //      `LUMEN_METAL_GDN_CONCURRENT_ENCODER=1` or defaults active).
        //   2. All five de-aliased buffers are present in scratch (allocated
        //      in `prefill_encode.rs:127-141` when the env was set at
        //      `ensure_batch_buffers` time).
        //   3. Deep-profile is OFF (`is_gdn_deep_enabled() == false`); the
        //      deep-profile path splits encoders per phase and defeats the
        //      concurrent scheduler.
        //
        // When ANY condition is false, the function falls back to the
        // legacy code path: serial encoder, scope(1) barriers, shared
        // `qkv_buf` / `scratch_buf` aliasing. The legacy path is exactly
        // byte-equivalent to the prior production default.
        //
        // Specification (consistent with retro):
        //   * Phase 1 QKV stays in `qkv_buf` (size qkv_dim, used only by
        //     Phase 1 and conv1d input — no aliasing conflict).
        //   * Phase 1 alpha gets a dedicated MTLBuffer (`batch_gdn_alpha_buf`).
        //   * Phase 1 beta gets a dedicated MTLBuffer (`batch_gdn_beta_buf`).
        //   * Conv1d+SiLU output gets a dedicated MTLBuffer
        //     (`batch_gdn_conv_out_buf`). Phase 2a reads from it.
        //   * Phase 2a raw_out gets a dedicated MTLBuffer
        //     (`batch_gdn_raw_out_buf`). Phase 2b reads from it.
        //   * Phase 2b ssm_out gets a dedicated MTLBuffer
        //     (`batch_gdn_ssm_in_buf`). Phase 3 reads from it.
        //
        // In legacy mode the same `(buf, offset)` pairs point into
        // `qkv_buf` / `scratch_buf` at the legacy offsets (alpha at 0,
        // beta at alpha_all_bytes, conv_out at 2*alpha_all_bytes, raw_out
        // at q_dim*batch_size*4, ssm_in at 0).
        let concurrent_encoder_active = graph_reorder::gdn_concurrent_encoder_enabled()
            && !super::profile::is_gdn_deep_enabled()
            && s.batch_gdn_raw_out_buf.is_some()
            && s.batch_gdn_ssm_in_buf.is_some()
            && s.batch_gdn_alpha_buf.is_some()
            && s.batch_gdn_beta_buf.is_some()
            && s.batch_gdn_conv_out_buf.is_some();
        let concurrent_encoder_validate = graph_reorder::gdn_concurrent_encoder_validate_serial();

        // Pre-compute byte offsets used by both modes (legacy: into shared
        // qkv_buf/scratch_buf; concurrent-encoder mode: ignored, dedicated buffers used at 0).
        let alpha_all_bytes = (batch_size * num_heads * 4) as u64;
        let conv_out_off_legacy = 2 * alpha_all_bytes;
        let raw_out_off_legacy = (batch_size * q_dim * 4) as u64;

        // (buf, offset) accessor helpers — return per-role buffer in concurrent-encoder mode
        // mode and shared-buffer offsets in legacy mode.
        let alpha_role_buf: &MetalBuffer = if concurrent_encoder_active {
            s.batch_gdn_alpha_buf.as_ref().unwrap()
        } else {
            scratch_buf
        };
        let alpha_role_off: u64 = if concurrent_encoder_active { 0 } else { 0 };
        let beta_role_buf: &MetalBuffer = if concurrent_encoder_active {
            s.batch_gdn_beta_buf.as_ref().unwrap()
        } else {
            scratch_buf
        };
        let beta_role_off: u64 = if concurrent_encoder_active { 0 } else { alpha_all_bytes };
        let conv_out_role_buf: &MetalBuffer = if concurrent_encoder_active {
            s.batch_gdn_conv_out_buf.as_ref().unwrap()
        } else {
            scratch_buf
        };
        let conv_out_role_off: u64 = if concurrent_encoder_active { 0 } else { conv_out_off_legacy };
        let raw_out_role_buf: &MetalBuffer = if concurrent_encoder_active {
            s.batch_gdn_raw_out_buf.as_ref().unwrap()
        } else {
            qkv_buf
        };
        let raw_out_role_off: u64 = if concurrent_encoder_active { 0 } else { raw_out_off_legacy };
        let ssm_in_role_buf: &MetalBuffer = if concurrent_encoder_active {
            s.batch_gdn_ssm_in_buf.as_ref().unwrap()
        } else {
            qkv_buf
        };
        let ssm_in_role_off: u64 = if concurrent_encoder_active { 0 } else { 0 };

        // Barrier helper: in concurrent-encoder production mode emit resource-scoped
        // `memoryBarrierWithResources:count:`, in legacy / validate mode
        // emit whole-MTLBuffer `memoryBarrierWithScope:1`. Resource-
        // scoped barriers are no-ops on Apple if the listed resources
        // don't have outstanding writes, so they're safe to over-scope.
        let emit_phase_barrier = |enc: &MetalComputeEncoder, bufs: &[&MetalBuffer]| {
            if concurrent_encoder_active && !concurrent_encoder_validate {
                enc.memory_barrier_with_resources(bufs);
            } else {
                enc.memory_barrier_with_scope(1);
            }
        };

        // ================================================================
        // PHASE 1: Pre-compute token-independent projections
        // ================================================================

        // Phase 1: Merged RMSNorm + QKV GEMM + Gate GEMM + Alpha/Beta matvec + Conv1d + SiLU
        //
        // The encoder is held `mut` (rather than `let enc =`) so the
        // optional MPSGraph BF16 QKV path can end this encoder, encode the
        // matmul into the command buffer, and reopen a fresh encoder for
        // the remaining sub-ops. When the MPSGraph path is OFF
        // the encoder is created once and torn down at Phase 1 end exactly
        // as before — bit-exact behaviour when disabled.
        //
        // GPU-time census (no-op unless LUMEN_METAL_PROFILE_GDN=1): when GDN deep
        // profiling is on, `concurrent_encoder_active` is forced false (guard
        // above), so `enc` is always a serial `new_compute_encoder`. At each
        // phase boundary the census helper ends the encoder and reopens a
        // fresh one; the FFI split hook commits+waits the just-finished CB and
        // records its TRUE GPU wall time (GPUEndTime-GPUStartTime) under the
        // section label set just before the reopen. This yields a
        // contamination-free per-PHASE GPU-time table (Phase1-bundle =
        // RMSNorm+QKV+gate+alpha/beta+conv+L2 vs Phase2a-recurrence vs
        // Phase2b-normgate vs Phase3-ssm_out), adjudicating the recurrence-vs-
        // GEMM-bundle question that subskip cannot answer (real data flows;
        // no value corruption => immune to MoE-routing contamination).
        let census = super::profile::is_gdn_deep_enabled();
        {
            // Label Phase 1 BEFORE opening its encoder: the Phase-1
            // `new_compute_encoder()` below triggers the FFI split that
            // promotes this label to in-flight, so the Phase-1 CB's GPU wall
            // is correctly attributed to it (the prior in-flight label, set by
            // prefill_encode as "gdn/batched_prefill", is recorded for whatever
            // preceded the GDN block).
            if census {
                super::profile::set_section("gdn/p1a_rmsnorm+qkv_gemm");
            }
            #[allow(unused_assignments)]
            let mut enc = if concurrent_encoder_active && !concurrent_encoder_validate {
                cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN prefill: phase 1 concurrent encoder".into())
                })?
            } else {
                cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN prefill: phase 1 encoder".into())
                })?
            };

            // Phase 1a: Batched RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
            enc.set_buffer(x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            // Phase 1a → Phase 1b barrier: RMSNorm wrote normed_buf,
            // downstream QKV / gate / alpha / beta GEMMs all read it.
            emit_phase_barrier(&enc, &[normed_buf]);

            // BF16 GDN qkv-proj + attn-gate-proj paired-dispatch fast path.
            //
            // When `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED=1` AND both QKV and
            // attn_gate are BF16 AND the per-layer repacked buffer is allocated,
            // a single `tiled_matmul_bf16_k64_qkv_gate_paired` dispatch produces
            // both `qkv_buf` and `gate_all_buf` from one shared X load. The
            // path-taken flag also skips the attn_gate dispatch further below.
            let bf16_paired_taken: bool = if matches!(meta.wq_quant, QuantScheme::Bf16)
                && matches!(attn_gate_quant, QuantScheme::Bf16)
                && super::graph_reorder::bf16_gdn_qkv_gate_paired_enabled()
                && gdn_idx < s.repacked_gdn_qkv_gate_bf16.len()
                && hidden_dim % 64 == 0
                && qkv_dim % 32 == 0
                && q_dim % 32 == 0
            {
                let maybe_buf = s.repacked_gdn_qkv_gate_bf16.get(gdn_idx).and_then(|o| o.as_ref());
                if let Some(packed_buf) = maybe_buf {
                    let gemm_aligned = batch_size % 32 == 0;
                    if gemm_aligned {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_qkv_gate_paired_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_qkv_gate_paired);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(packed_buf, 0, 0);              // W: concat-stripe BF16
                    enc.set_buffer(normed_buf, 0, 1);              // X: shared input
                    enc.set_buffer(qkv_buf, 0, 2);                 // Y_qkv
                    enc.set_buffer(gate_all_buf, 0, 3);            // Y_gate
                    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 4);  // M
                    enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 5);    // N_qkv
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 6);      // N_gate
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 7); // K
                    let n_total = (qkv_dim + q_dim) as u64;
                    enc.dispatch_threadgroups(
                        MTLSize::new(n_total.div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    true
                } else { false }
            } else { false };

            // QKV: tiled GEMM dispatch [batch_size, hidden_dim] @ [hidden_dim, qkv_dim] -> [batch_size, qkv_dim]
            // Tiled GEMM reads weight tiles once across all output rows (vs GEMV reading full weights per row).
            if bf16_paired_taken {
                // QKV produced by the paired dispatch above.
            } else if matches!(meta.wq_quant, QuantScheme::Q8_0) {
                let gemm_aligned = batch_size % 32 == 0 && qkv_dim % 32 == 0 && hidden_dim % 32 == 0;
                if gemm_aligned && hidden_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, meta.wq_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(qkv_buf, 0, 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);    // N
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K
                // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 1): skip QKV-proj GEMM.
                if super::graph_reorder::gdn_subskip() & 1 == 0 {
                    enc.dispatch_threadgroups(
                        MTLSize::new((qkv_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            } else if matches!(meta.wq_quant, QuantScheme::Bf16) {
                // Optional MPSGraph BF16 GEMM path for the QKV
                // projection (qkv_dim=8192, hidden_dim=4096 — the largest
                // per-layer BF16 matmul). Gated by
                // `LUMEN_METAL_BF16_MPS=1` AND M >= 32 AND N >= 4096.
                // When taken: end current encoder, encode MPSGraph into
                // the command buffer, reopen a fresh encoder so the
                // remaining Phase 1 sub-ops (Gate / Alpha / Beta / Conv1d /
                // L2) continue in a new context. Byte-equivalent to the
                // legacy path when the env var is unset.
                let mps_eligible = super::graph_reorder::bf16_mps_enabled()
                    && batch_size >= 32
                    && qkv_dim >= 4096
                    && hidden_dim >= 4096;
                let mps_taken = if mps_eligible {
                    if let Some(ctx) = super::mps_graph_ffi::get() {
                        // End current encoder, encode MPSGraph into
                        // the same CB. MPSGraph encoder is independent
                        // of compute encoders so this is safe. If
                        // MPSGraph internally commits (large graphs),
                        // the FFI rebinds `cmd` to the new root CB.
                        enc.end_encoding();
                        super::mps_graph_ffi::encode_bf16_matmul_into_cb(
                            ctx, cmd,
                            normed_buf,  // X [M=batch_size, K=hidden_dim]
                            layer_buf,
                            meta.wq_off, // W offset into the layer blob
                            qkv_buf,     // Y [M, N=qkv_dim]
                            batch_size as u32,
                            hidden_dim as u32,
                            qkv_dim as u32,
                        ).map_err(|reason| {
                            RuntimeError::Compute(format!(
                                "MPSGraph BF16 QKV failed ({reason}); fall back via LUMEN_METAL_BF16_MPS=0"
                            ))
                        })?;
                        // Reopen an encoder of the same flavour (concurrent
                        // vs serial) used for Phase 1.
                        enc = if concurrent_encoder_active && !concurrent_encoder_validate {
                            cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "GDN prefill: phase 1 concurrent encoder reopen after MPSGraph QKV".into()
                                )
                            })?
                        } else {
                            cmd.new_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "GDN prefill: phase 1 encoder reopen after MPSGraph QKV".into()
                                )
                            })?
                        };
                        true
                    } else { false }
                } else { false };
                if !mps_taken {
                    // BF16 tiled prefill GEMM (mirror Q8 structure with BF16 kernels)
                    // optional env opt-in to force non-K64 tile (4 KB shmem,
                    // 4× barriers per K-loop iteration vs K64). Default OFF preserves
                    // reference byte-identical behaviour.
                    let force_nok64 = super::graph_reorder::bf16_gdn_tile_nok64_enabled();
                    let gemm_aligned = batch_size % 32 == 0 && qkv_dim % 32 == 0 && hidden_dim % 32 == 0;
                    if !force_nok64 && hidden_dim % 64 == 0 && batch_size <= 4096 {
                        if gemm_aligned {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    } else {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    enc.set_buffer(layer_buf, meta.wq_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(qkv_buf, 0, 2);
                    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                    enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);    // N
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K
                    enc.dispatch_threadgroups(
                        MTLSize::new((qkv_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            } else if matches!(meta.wq_quant, QuantScheme::Q4_0) {
                // Use k64 kernel when K (hidden_dim) is 64-aligned and batch fits — halves
                // K-loop barriers vs non-k64 variant (applied to GDN prefill).
                if hidden_dim % 64 == 0 && batch_size <= 256 {
                    let gemm_aligned = batch_size % 32 == 0 && qkv_dim % 32 == 0 && hidden_dim % 32 == 0;
                    if gemm_aligned {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                    enc.set_threadgroup_memory_length(4096, 0);
                }
                enc.set_buffer(layer_buf, meta.wq_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(qkv_buf, 0, 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);    // N
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5); // K
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                for t in 0..batch_size {
                    let normed_t_off = (t as u64) * tok_bytes_hidden;
                    let qkv_t_off = (t as u64) * tok_bytes_qkv;
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, meta.wq_off, 0);
                    enc.set_buffer(normed_buf, normed_t_off, 1);
                    enc.set_buffer(qkv_buf, qkv_t_off, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(qkv_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                }
            }

            // GPU-time census split: close the RMSNorm+QKV-GEMM CB (records its GPU
            // wall under p1a), label the attn-gate GEMM. No-op when off.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p1b_attngate_gemm");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: p1b attngate encoder reopen".into())
                })?;
            }

            // Gate: tiled GEMM dispatch [batch_size, hidden_dim] @ [hidden_dim, q_dim] -> [batch_size, q_dim]
            // paired path produces gate_all_buf via the QKV dispatch above; skip this block.
            if bf16_paired_taken {
                // gate_all_buf produced by the paired dispatch.
            } else if matches!(attn_gate_quant, QuantScheme::Q8_0) {
                let gemm_aligned = batch_size % 32 == 0 && q_dim % 32 == 0 && hidden_dim % 32 == 0;
                if gemm_aligned && hidden_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_all_buf, 0, 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);       // N
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K
                // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 64): skip attn-gate GEMM.
                if super::graph_reorder::gdn_subskip() & 64 == 0 {
                    enc.dispatch_threadgroups(
                        MTLSize::new((q_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            } else if matches!(attn_gate_quant, QuantScheme::Bf16) {
                // BF16 tiled gate GEMM
                let force_nok64 = super::graph_reorder::bf16_gdn_tile_nok64_enabled();
                let gemm_aligned = batch_size % 32 == 0 && q_dim % 32 == 0 && hidden_dim % 32 == 0;
                if !force_nok64 && hidden_dim % 64 == 0 && batch_size <= 4096 {
                    if gemm_aligned {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                } else {
                    enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
                    enc.set_threadgroup_memory_length(4096, 0);
                }
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_all_buf, 0, 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);       // N
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else if matches!(attn_gate_quant, QuantScheme::Q4_0) {
                // Use k64 kernel when K (hidden_dim) is 64-aligned — halves K-loop barriers.
                if hidden_dim % 64 == 0 && batch_size <= 256 {
                    let gemm_aligned = batch_size % 32 == 0 && q_dim % 32 == 0 && hidden_dim % 32 == 0;
                    if gemm_aligned {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                    enc.set_threadgroup_memory_length(4096, 0);
                }
                enc.set_buffer(layer_buf, attn_gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_all_buf, 0, 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);       // N
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5); // K
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                for t in 0..batch_size {
                    let normed_t_off = (t as u64) * tok_bytes_hidden;
                    let gate_t_off = (t as u64) * tok_bytes_qdim;
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, attn_gate_off, 0);
                    enc.set_buffer(normed_buf, normed_t_off, 1);
                    enc.set_buffer(gate_all_buf, gate_t_off, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(q_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                }
            }
            // GPU-time census split: close the attn-gate GEMM CB, label the
            // alpha/beta + gates block. No-op when off.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p1c_alpha_beta_gates");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: p1c alpha/beta encoder reopen".into())
                })?;
            }
            // Alpha/Beta projections via tiled GEMM + separate batched gate computation.
            // Tiled GEMM amortizes weight loads across batch_size output rows (vs GEMV per-row).
            // alpha/beta outputs go to dedicated MTLBuffers
            // (`alpha_role_buf` / `beta_role_buf`) so the gate compute and
            // Phase 2a v3 chunked kernel don't pay scratch_buf whole-buffer
            // hazard cost. In legacy mode the role buffers point into the
            // shared `scratch_buf` at offsets `alpha_role_off` / `beta_role_off`.
            {
                // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 2): skip alpha/beta GEMMs.
                let skip_ab = super::graph_reorder::gdn_subskip() & 2 != 0;
                // Alpha GEMM: [batch_size, hidden_dim] @ [hidden_dim, num_heads] -> [batch_size, num_heads]
                let gemm_aligned = batch_size % 32 == 0 && num_heads % 32 == 0 && hidden_dim % 32 == 0;
                if gemm_aligned && hidden_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, ssm_alpha_off, 0);   // weights [num_heads, hidden_dim] Q8_0
                enc.set_buffer(normed_buf, 0, 1);               // input [batch_size, hidden_dim]
                enc.set_buffer(alpha_role_buf, alpha_role_off, 2); // output [batch_size, num_heads] (raw alpha)
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // N = 32
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K = 4096
                if !skip_ab {
                enc.dispatch_threadgroups(
                    MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
                }

                // Beta GEMM: [batch_size, hidden_dim] @ [hidden_dim, num_heads] -> [batch_size, num_heads]
                if gemm_aligned && hidden_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, ssm_beta_off, 0);    // weights [num_heads, hidden_dim] Q8_0
                enc.set_buffer(normed_buf, 0, 1);               // input [batch_size, hidden_dim]
                enc.set_buffer(beta_role_buf, beta_role_off, 2); // output [batch_size, num_heads] (raw beta)
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // N = 32
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K = 4096
                if !skip_ab {
                enc.dispatch_threadgroups(
                    MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
                }

                // Batched gate computation: transform raw alpha/beta to gated values
                // alpha_out = exp(ssm_a * softplus(alpha_raw + dt_bias))  -- decay in (0,1)
                // beta_out  = sigmoid(beta_raw)                          -- mixing rate
                // Phase 1 alpha/beta GEMMs → batched gate compute barrier:
                // gate compute reads alpha_role and beta_role.
                emit_phase_barrier(&enc, &[alpha_role_buf, beta_role_buf]);
                let pso_gates_batched = pipelines.gdn_compute_gates_batched.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("gdn_compute_gates_batched pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_gates_batched);
                enc.set_buffer(layer_buf, ssm_dt_off, 0);             // dt_bias [n_heads]
                enc.set_buffer(layer_buf, ssm_a_off, 1);              // ssm_a [n_heads]
                enc.set_buffer(beta_role_buf, beta_role_off, 2);      // beta_raw [T * n_heads] (input)
                enc.set_buffer(alpha_role_buf, alpha_role_off, 3);    // alpha_raw [T * n_heads] (input)
                enc.set_buffer(alpha_role_buf, alpha_role_off, 4);    // alpha_out [T * n_heads] (overwrite in-place)
                enc.set_buffer(beta_role_buf, beta_role_off, 5);      // beta_out [T * n_heads] (overwrite in-place)
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
                let total_gates = (num_heads * batch_size) as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(total_gates.div_ceil(256), 1, 1),
                    MTLSize::new(256u64.min(total_gates), 1, 1),
                );
            }

            // GPU-time census split: close the alpha/beta+gates CB, label conv1d
            // (+L2, which follows it in the same block). No-op when off.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p1d_conv1d_silu+l2");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: p1d conv1d encoder reopen".into())
                })?;
            }

            // Batched conv1d: qkv_buf[T, qkv_dim] -> conv_out_role_buf
            // Needs barrier: reads from qkv_buf written by batched QKV matvec above.
            // Phase 1 QKV GEMM → conv1d barrier: conv1d reads qkv_buf.
            emit_phase_barrier(&enc, &[qkv_buf]);

            // Fused conv1d+SiLU+L2(Q/K) path collapses the
            // conv1d -> BARRIER -> L2 dispatch boundary into one kernel. The
            // standalone L2 dispatch is the dominant GDN-block stall (subskip
            // attribution: removing L2 saves ~200ms; an equally-correct
            // barrier-free L2-SG kernel recovers ~0 — the cost is the dispatch
            // epoch boundary, not L2 compute). Bit-identical math.
            // Default OFF (env LUMEN_METAL_GDN_CONV_L2_FUSED=1); requires the
            // fused + V-range pipelines and the state-update pipeline.
            let conv_l2_fused = super::graph_reorder::gdn_conv_l2_fused_enabled()
                && super::graph_reorder::gdn_subskip() & (4 | 8) == 0   // diagnostic skips force legacy
                && pipelines.conv1d_silu_l2_qk_fused.is_some()
                && pipelines.conv1d_silu_vrange.is_some()
                && pipelines.ssm_conv1d_state_update.is_some();

            if conv_l2_fused {
                let pso_fused = pipelines.conv1d_silu_l2_qk_fused.as_ref().unwrap();
                let pso_vrange = pipelines.conv1d_silu_vrange.as_ref().unwrap();
                let pso_su = pipelines.ssm_conv1d_state_update.as_ref().unwrap();
                let v_base = (2 * qk_dim) as u32;   // V region channel base
                let v_count = value_dim as u32;     // V region channel count

                // (a) Fused conv+SiLU+L2 for Q/K. Grid (num_kv_heads, T), 128 thr/TG.
                enc.set_pipeline_state(pso_fused);
                enc.set_buffer(qkv_buf, 0, 0);
                enc.set_buffer(conv_state_buf, 0, 1);
                enc.set_buffer(layer_buf, ssm_conv1d_off, 2);
                enc.set_buffer(conv_out_role_buf, conv_out_role_off, 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
                enc.set_bytes(&conv_pos.to_le_bytes(), 6);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                enc.set_bytes(&(0u32).to_le_bytes(), 10);             // q_offset
                enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 11);    // k_offset
                enc.dispatch_threadgroups(
                    MTLSize::new(num_kv_heads as u64, batch_size as u64, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );

                // (b) conv+SiLU for V (no L2). Independent of Q/K — same encoder,
                // no barrier between (a) and (b): disjoint output ranges.
                let v_tg = 256u64.min(v_count as u64).max(1);
                enc.set_pipeline_state(pso_vrange);
                enc.set_buffer(qkv_buf, 0, 0);
                enc.set_buffer(conv_state_buf, 0, 1);
                enc.set_buffer(layer_buf, ssm_conv1d_off, 2);
                enc.set_buffer(conv_out_role_buf, conv_out_role_off, 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
                enc.set_bytes(&conv_pos.to_le_bytes(), 6);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
                enc.set_bytes(&v_base.to_le_bytes(), 8);
                enc.set_bytes(&v_count.to_le_bytes(), 9);
                enc.dispatch_threadgroups(
                    MTLSize::new((v_count as u64).div_ceil(v_tg), batch_size as u64, 1),
                    MTLSize::new(v_tg, 1, 1),
                );

                // (c) conv_state circular-buffer update. Reads qkv_buf, writes
                // conv_state — independent of conv_out, so it is OFF the
                // conv_out -> recurrence critical spine. Barrier on conv_state
                // (the fused/V kernels READ conv_state; this WRITES it -> RAW).
                emit_phase_barrier(&enc, &[conv_state_buf]);
                let su_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.set_pipeline_state(pso_su);
                enc.set_buffer(qkv_buf, 0, 0);
                enc.set_buffer(conv_state_buf, 0, 1);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 2);
                enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 3);
                enc.set_bytes(&conv_pos.to_le_bytes(), 4);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(su_tg), 1, 1),
                    MTLSize::new(su_tg, 1, 1),
                );
                // conv_out (Q/K L2-normalized, V SiLU'd) is fully produced.
                // The L2->Phase2a barrier below (on conv_out, alpha, beta) is
                // the only join the recurrence needs.
            } else {

            // Fused Conv1d + SiLU -- token-parallel variant dispatches (dim_blocks, T) TGs
            // Falls back to serial kernel if parallel pipeline unavailable.
            // conv1d output goes to `conv_out_role_buf` (dedicated
            // when the concurrent encoder is active; into `scratch_buf` at `conv_out_role_off` in
            // legacy mode).
            let pso_conv1d_silu_par = pipelines.ssm_conv1d_silu_prefill_parallel.as_ref();

            enc.set_buffer(qkv_buf, 0, 0);                          // input [T * qkv_dim]
            enc.set_buffer(conv_state_buf, 0, 1);                    // conv_state circular buffer (R/W)
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2);            // kernel weights
            enc.set_buffer(conv_out_role_buf, conv_out_role_off, 3); // output [T * qkv_dim] (SiLU-activated)
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            let used_parallel_conv1d = pso_conv1d_silu_par.is_some();
            // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 4): skip conv1d.
            let skip_conv1d = super::graph_reorder::gdn_subskip() & 4 != 0;
            if skip_conv1d {
                // attribution only: do not dispatch conv1d (garbage output).
            } else if let Some(pso_par) = pso_conv1d_silu_par {
                // Parallel: (ceil(dim/TG_SIZE), T) TGs -- each handles one (channel_block, token)
                // Determinism fix: this kernel NO LONGER writes conv_state (it
                // is pure-read on conv_state). The circular-buffer update is a
                // SEPARATE barriered dispatch below, eliminating the cross-
                // threadgroup conv_state read-write race that was the prefill
                // non-determinism source.
                let conv_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.set_pipeline_state(pso_par);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), batch_size as u64, 1),
                    MTLSize::new(conv_tg, 1, 1),
                );
            } else {
                // Fallback: serial kernel -- ceil(dim/256) TGs, each loops over T.
                // The serial kernel is race-free and updates conv_state itself
                // (within each per-channel thread, reads precede the write).
                let pso_conv1d_silu = pipelines.ssm_conv1d_silu_prefill.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("ssm_conv1d_silu_prefill pipeline not compiled".into())
                })?;
                let conv_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.set_pipeline_state(pso_conv1d_silu);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), 1, 1),
                    MTLSize::new(conv_tg, 1, 1),
                );
            }

            // Determinism fix: race-free conv_state circular-buffer update for
            // the parallel path. Barrier first so EVERY conv1d read of the OLD
            // conv_state retires before these writes. Byte-identical to the update
            // block that used to live (racily) inside the parallel conv1d kernel.
            if used_parallel_conv1d {
                if let Some(pso_su) = pipelines.ssm_conv1d_state_update.as_ref() {
                    emit_phase_barrier(&enc, &[conv_state_buf]);
                    let su_tg = 256u64.min(qkv_dim as u64).max(1);
                    enc.set_pipeline_state(pso_su);
                    enc.set_buffer(qkv_buf, 0, 0);            // input [T * qkv_dim]
                    enc.set_buffer(conv_state_buf, 0, 1);     // conv_state (write)
                    enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 2);
                    enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 3);
                    enc.set_bytes(&conv_pos.to_le_bytes(), 4);
                    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((qkv_dim as u64).div_ceil(su_tg), 1, 1),
                        MTLSize::new(su_tg, 1, 1),
                    );
                } else {
                    return Err(RuntimeError::Compute(
                        "ssm_conv1d_state_update pipeline not compiled (DET-001 fix requires it)".into()));
                }
            }

            // L2 normalize Q and K in-place within conv_out (after SiLU).
            // conv1d → L2 barrier: L2 reads conv_out.
            emit_phase_barrier(&enc, &[conv_out_role_buf]);
            {
                // K2 Wave-1: simdgroup-per-head L2 (no threadgroup barriers),
                // gated by LUMEN_METAL_GDN_L2_SG=1, requires head_dim%32==0.
                let use_l2_sg = super::graph_reorder::gdn_l2_sg_enabled()
                    && head_dim % 32 == 0
                    && pipelines.l2_normalize_qk_strided_sg.is_some();
                let pso_l2 = if use_l2_sg {
                    pipelines.l2_normalize_qk_strided_sg.as_ref().unwrap()
                } else {
                    pipelines.l2_normalize_qk_strided.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("l2_normalize_qk_strided pipeline not compiled".into())
                    })?
                };
                enc.set_pipeline_state(pso_l2);
                enc.set_buffer(conv_out_role_buf, conv_out_role_off, 0);  // conv_out [T, qkv_dim]
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 1);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);    // stride
                enc.set_bytes(&(0u32).to_le_bytes(), 5);              // q_offset = 0
                enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 6);     // k_offset = qk_dim
                // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 8): skip L2.
                if super::graph_reorder::gdn_subskip() & 8 == 0 {
                    if use_l2_sg {
                        // SG_PER_TG=4 simdgroups/TG (128 threads); one SG per (token,head).
                        const SG_PER_TG: u64 = 4;
                        let total_heads = (num_kv_heads * batch_size) as u64;
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_heads.div_ceil(SG_PER_TG), 1, 1),
                            MTLSize::new(32 * SG_PER_TG, 1, 1),
                        );
                    } else {
                        enc.dispatch_threadgroups(
                            MTLSize::new((num_kv_heads * batch_size) as u64, 1, 1),
                            MTLSize::new(head_dim as u64, 1, 1),
                        );
                    }
                }
            }
            } // end else (legacy conv+state+L2 path)

            // ================================================================
            // PHASE 2: GDN state update (v3 chunked simdgroup-parallel kernel)
            // Continues in the same encoder as Phase 1
            // ================================================================

            let pso_v3 = pipelines.gdn_prefill_fused_v3_chunked.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_prefill_fused_v3_chunked pipeline not compiled".into())
            })?;
            let pso_norm_gate = pipelines.gdn_prefill_norm_gate.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_prefill_norm_gate pipeline not compiled".into())
            })?;

            // L2 → Phase 2a barrier: Phase 2a reads conv_out (L2-normalised Q/K),
            // alpha_role, beta_role. All must be retired before Phase 2a starts.
            emit_phase_barrier(&enc, &[conv_out_role_buf, alpha_role_buf, beta_role_buf]);

            // GPU-time census split: close Phase-1 CB (records its GPU wall under the
            // p1 label), label Phase 2a, reopen serial encoder. No-op otherwise.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p2a_recurrence");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: phase 2a encoder reopen".into())
                })?;
            }

            // Phase 2a: v3 chunked state update (4x unrolled, simdgroup-parallel)
            // reads conv_out/alpha/beta from role buffers,
            // writes raw_out to `raw_out_role_buf` (dedicated when concurrent encoder is
            // active; legacy `qkv_buf` at `raw_out_off_legacy`).
            //
            // NSG4 geometry opt-in (`LUMEN_METAL_GDN_PHASE2A_NSG4=1`): swaps the
            // (1, val_dim, n_heads) grid of 32-thread TGs for a (val_dim/4,
            // n_heads, 1) grid of 128-thread TGs (4 simdgroups/TG sharing Q/K
            // fetches via L1). Bit-identical kernel body; the reference-token
            // gate is the validator. Wired here for the MoE GDN path (the dense
            // 9B used different layer counts/FFN share — the MoE
            // GDN is 72% of prefill so a small per-layer win compounds 30x).
            // Chunk-parallel delta-rule: replaces the O(T)-serial
            // recurrence with O(T/C) serial chunks. Requires head_dim==128 (the
            // GDN_CS geometry: 32 lanes x 4 key-elems; MT=32 value-tile).
            let use_chunkscan = graph_reorder::gdn_prefill_chunked_enabled()
                && pipelines.gdn_prefill_chunkscan.is_some()
                && head_dim == 128;
            let use_nsg4 = !use_chunkscan
                && graph_reorder::gdn_phase2a_nsg4_enabled()
                && pipelines.gdn_prefill_fused_v3_chunked_nsg4.is_some()
                && head_dim % 4 == 0;
            if use_chunkscan {
                enc.set_pipeline_state(pipelines.gdn_prefill_chunkscan.as_ref().unwrap());
            } else if use_nsg4 {
                enc.set_pipeline_state(pipelines.gdn_prefill_fused_v3_chunked_nsg4.as_ref().unwrap());
            } else {
                enc.set_pipeline_state(pso_v3);
            }
            enc.set_buffer(h_state_buf, 0, 0);                          // h_state [n_heads * val_dim * key_dim] (transposed layout)
            enc.set_buffer(conv_out_role_buf, conv_out_role_off, 1);     // conv_out_all [T, qkv_dim]
            enc.set_buffer(alpha_role_buf, alpha_role_off, 2);           // alpha_all [T, n_heads]
            enc.set_buffer(beta_role_buf, beta_role_off, 3);             // beta_all [T, n_heads]
            enc.set_buffer(raw_out_role_buf, raw_out_role_off, 4);       // raw_out [T, q_dim]
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 5);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 6);          // key_dim
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 7);          // val_dim (per head)
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 9);        // T
            enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 10);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 11);
            // DIAG: skip the Phase 2a state-update dispatch to attribute its GPU
            // cost (no-op when LUMEN_METAL_GDN_DIAG_SKIP is unset; garbage output).
            if graph_reorder::gdn_diag_skip() != 1 {
                if use_chunkscan {
                    let chunk_c = graph_reorder::gdn_prefill_chunk_c();
                    enc.set_bytes(&chunk_c.to_le_bytes(), 12);          // chunk_C
                    // K_tg threadgroup memory: C * key_dim floats.
                    let k_tg_bytes = (chunk_c as u64) * (head_dim as u64) * 4;
                    enc.set_threadgroup_memory_length(k_tg_bytes, 0);
                    // grid (n_heads, val_dim_per_head/MT=32, 1); TG (32, 4, 1).
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, (head_dim as u64) / 32, 1),
                        MTLSize::new(32, 4, 1),
                    );
                } else if use_nsg4 {
                    // (val_dim/NSG, n_heads, 1) grid of (32, NSG=4, 1) threads.
                    enc.dispatch_threadgroups(
                        MTLSize::new((head_dim as u64) / 4, num_heads as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );
                } else {
                    enc.dispatch_threadgroups(
                        MTLSize::new(1, head_dim as u64, num_heads as u64),
                        MTLSize::new(32, 1, 1),
                    );
                }
            }
            // Phase 2a → Phase 2b barrier: Phase 2b reads `raw_out_role` and
            // `gate_all_buf` (= attn_out_buf, written by Phase 1 attn-gate
            // GEMM long earlier). retro's original bug was the
            // attn_out_buf omission — pre-empt it by including it here.
            emit_phase_barrier(&enc, &[raw_out_role_buf, attn_out_buf]);

            // GPU-time census split: close Phase-2a CB (records recurrence GPU wall),
            // label Phase 2b, reopen serial encoder. No-op otherwise.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p2b_norm_gate");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: phase 2b encoder reopen".into())
                })?;
            }

            // Phase 2b: RMSNorm + SiLU gate on raw output
            //
            // Fusion of norm_gate INTO the state kernel is NOT feasible.
            //
            // The state kernel (v3_chunked) is parallelized as grid (1, val_dim=128, n_heads=32)
            // with 32 threads per TG. Each TG owns a single (head, vj) pair and writes one
            // scalar per token: raw_out[t, h, vj].
            //
            // RMSNorm requires sum-of-squares across all val_dim=128 elements for a given
            // (token, head). Those 128 values live in 128 SEPARATE threadgroups. Metal has
            // no cross-threadgroup synchronization within a single dispatch, so the reduction
            // cannot be performed inside the state kernel without fundamentally restructuring
            // its parallelism.
            //
            // Restructuring to put all 128 vj values in one TG would require each TG to hold
            // 128 * 128 = 16384 state floats (64KB), destroying occupancy. The current design
            // uses 4096 TGs of 32 threads each, which is optimal for GPU saturation.
            //
            // This is a hard architectural constraint, not a performance tradeoff. The barrier
            // between Phase 2a and 2b is load-bearing.
            // reads raw_out and gate_all_buf, writes ssm_in_role_buf.
            enc.set_pipeline_state(pso_norm_gate);
            enc.set_buffer(raw_out_role_buf, raw_out_role_off, 0);    // raw_out [T, q_dim]
            enc.set_buffer(gate_all_buf, 0, 1);                       // gate_all [T, q_dim]
            enc.set_buffer(layer_buf, ssm_norm_off, 2);               // norm_scale
            enc.set_buffer(ssm_in_role_buf, ssm_in_role_off, 3);      // ssm_out [T, q_dim] (output, consumed by Phase 3)
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 5);       // val_dim
            enc.set_bytes(&eps.to_le_bytes(), 6);
            enc.set_bytes(&(1u32).to_le_bytes(), 7);                  // scale_n_heads
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 8);     // T
            // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 16): skip norm_gate.
            if super::graph_reorder::gdn_subskip() & 16 == 0 {
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, batch_size as u64, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );
            }
            // Phase 2b → Phase 3 barrier: Phase 3 ssm_out GEMM reads
            // `ssm_in_role` as X-input.
            emit_phase_barrier(&enc, &[ssm_in_role_buf]);

            // GPU-time census split: close Phase-2b CB (records norm_gate GPU wall),
            // label Phase 3, reopen serial encoder. No-op otherwise.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p3_ssm_out_gemm+residual");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: phase 3 encoder reopen".into())
                })?;
            }

            // Phase 3: Fused SSM Out GEMM + residual add
            // attn_proj_buf[m,n] = SSM_Out_GEMM(qkv_buf)[m,n] + x_buf[m,n]
            // Eliminates separate residual_add_copy dispatch + barrier.
            // The FFN down projection later does: x_buf = Down * gate + attn_proj_buf,
            // so x_buf is correctly updated for the next layer without an extra copy.

            // DIAG: one-shot print of the ACTUAL ssm_out dispatch (quant,
            // dims, aligned-variant) to settle whether the census's ~38ms/call
            // is the kernel the microbench measured at 1.5ms. No-op unless deep
            // profiling is on; prints only on gdn_idx==0.
            if census && gdn_idx == 0 {
                eprintln!(
                    "[ssm-out-diag] ssm_out_quant={:?} wq_quant={:?} attn_gate_quant={:?} \
                     M(batch)={batch_size} N(hidden)={hidden_dim} K(q_dim)={q_dim} aligned={}",
                    ssm_out_quant, meta.wq_quant, attn_gate_quant,
                    batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0 && q_dim % 64 == 0,
                );
                if matches!(ssm_out_quant, QuantScheme::F32) {
                    eprintln!(
                        "[ssm-out-diag] *** F32 ssm_out => PER-TOKEN matvec fallback: \
                         {batch_size} dispatches/layer x 30 layers = {} tiny GEMVs ***",
                        batch_size * 30
                    );
                }
            }
            // Phase 3 ssm_out X-input comes from `ssm_in_role_buf`
            // (dedicated when the concurrent encoder is active; legacy `qkv_buf` at offset 0).
            if matches!(ssm_out_quant, QuantScheme::Q8_0) {
                let gemm_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
                if gemm_aligned && q_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, ssm_out_off, 0);                  // W_q8 weights [hidden_dim, q_dim]
                enc.set_buffer(ssm_in_role_buf, ssm_in_role_off, 1);         // X input [batch_size, q_dim]
                enc.set_buffer(attn_proj_buf, 0, 2);                          // Y output [batch_size, hidden_dim]
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);         // M
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);         // N
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);              // K
                enc.set_buffer(x_buf, 0, 6);                                  // R residual [batch_size, hidden_dim]
                // K2 DIAG (no-op unless LUMEN_METAL_GDN_SUBSKIP bit 32): skip ssm_out GEMM.
                if super::graph_reorder::gdn_subskip() & 32 == 0 {
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            } else if matches!(ssm_out_quant, QuantScheme::Q4_0) {
                // Fused Q4_0 tiled GEMM + residual: attn_proj_buf = GEMM(ssm_in) + x_buf
                // Use k64 residual variant when K (q_dim) is 64-aligned — halves K-loop barriers.
                if q_dim % 64 == 0 && batch_size <= 256 {
                    let gemm_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
                    if gemm_aligned {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_residual_batched);
                    enc.set_threadgroup_memory_length(4096, 0);
                }
                enc.set_buffer(layer_buf, ssm_out_off, 0);                   // W_q4 weights [hidden_dim, q_dim]
                enc.set_buffer(ssm_in_role_buf, ssm_in_role_off, 1);          // X input [batch_size, q_dim]
                enc.set_buffer(attn_proj_buf, 0, 2);                          // Y output [batch_size, hidden_dim]
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);         // M
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);         // N
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);              // K
                enc.set_buffer(x_buf, 0, 6);                                  // R residual [batch_size, hidden_dim]
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else if matches!(ssm_out_quant, QuantScheme::Bf16) {
                // Optional MPSGraph BF16 GEMM path for the ssm_out
                // projection (hidden_dim=4096, q_dim=4096 — second-largest
                // per-layer BF16 matmul). Gated by `LUMEN_METAL_BF16_MPS=1`
                // AND M >= 32 AND N >= 4096. The fused-residual contract
                // is recovered by dispatching `add_residual_batched` after
                // the matmul (`attn_proj_buf += x_buf`).
                let mps_eligible_ssm = super::graph_reorder::bf16_mps_enabled()
                    && batch_size >= 32
                    && hidden_dim >= 4096
                    && q_dim >= 4096;
                let mps_taken_ssm = if mps_eligible_ssm {
                    if let Some(ctx) = super::mps_graph_ffi::get() {
                        // End current encoder, encode MPSGraph matmul,
                        // open a fresh encoder for the residual add + the
                        // remainder of Phase 3 + FFN-norm.
                        enc.end_encoding();
                        super::mps_graph_ffi::encode_bf16_matmul_into_cb(
                            ctx, cmd,
                            ssm_in_role_buf,  // X [M, K=q_dim] at offset 0
                            layer_buf,
                            meta.ssm_out_off.unwrap(),
                            attn_proj_buf,    // Y [M, N=hidden_dim]
                            batch_size as u32,
                            q_dim as u32,
                            hidden_dim as u32,
                        ).map_err(|reason| {
                            RuntimeError::Compute(format!(
                                "MPSGraph BF16 ssm_out failed ({reason}); fall back via LUMEN_METAL_BF16_MPS=0"
                            ))
                        })?;
                        // Reopen encoder of the same flavour and dispatch
                        // the F32 residual add to recover fused-residual
                        // semantics (`attn_proj_buf += x_buf`).
                        enc = if concurrent_encoder_active && !concurrent_encoder_validate {
                            cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "GDN prefill: phase 3 concurrent encoder reopen after MPSGraph ssm_out".into()
                                )
                            })?
                        } else {
                            cmd.new_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "GDN prefill: phase 3 encoder reopen after MPSGraph ssm_out".into()
                                )
                            })?
                        };
                        enc.set_pipeline_state(&pipelines.add_residual_batched);
                        enc.set_buffer(attn_proj_buf, 0, 0);  // dst: F32 [batch_size, hidden_dim]
                        enc.set_buffer(x_buf, 0, 1);          // src: F32 residual
                        let total_elems = (batch_size * hidden_dim) as u32;
                        enc.set_bytes(&total_elems.to_le_bytes(), 2);
                        let tg = 256u64;
                        enc.dispatch_threadgroups(
                            MTLSize::new((total_elems as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                        true
                    } else { false }
                } else { false };
                if !mps_taken_ssm {
                    // BF16 tiled GEMM + residual: attn_proj_buf = GEMM(ssm_in) + x_buf
                    // optional env opt-in to force non-K64 tile (4 KB shmem,
                    // 4× barriers per K-loop iteration vs K64). Default OFF preserves
                    // reference byte-identical behaviour.
                    let force_nok64 = super::graph_reorder::bf16_gdn_tile_nok64_enabled();
                    if !force_nok64 && q_dim % 64 == 0 && batch_size <= 4096 {
                        let gemm_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
                        if gemm_aligned {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    } else {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    enc.set_buffer(layer_buf, ssm_out_off, 0);                  // W_bf16 weights [hidden_dim, q_dim]
                    enc.set_buffer(ssm_in_role_buf, ssm_in_role_off, 1);         // X input [batch_size, q_dim]
                    enc.set_buffer(attn_proj_buf, 0, 2);                          // Y output [batch_size, hidden_dim]
                    enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);         // M
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);         // N
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);              // K
                    enc.set_buffer(x_buf, 0, 6);                                  // R residual [batch_size, hidden_dim]
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            } else if graph_reorder::gdn_ssm_out_f32_batched_enabled() {
                // F32 ssm_out BATCHED tiled GEMM + fused residual.
                //
                // The MoE-35B ssm_out weight is stored F32; the legacy fallback
                // below ran a PER-TOKEN matvec loop (1239 dispatches/layer × 30
                // GDN layers = 37,170 tiny GEMVs), measured at ~38ms/layer =
                // ~1143ms = 60% of prefill (clean GPU census). This replaces
                // it with ONE `tiled_matmul_bytes_f32_residual` dispatch:
                // simdgroup-MMA tiled GEMM (F32 weights+activations cast to half
                // for the MMA inputs, F32 accumulate) with the residual add
                // (attn_proj = ssm_out·X^T + x_buf) fused at writeback — same
                // contract as the Q8/Q4/Bf16 tiled-residual paths. FP-order +
                // half-input precision diverge from the per-token F32 matvec, so
                // the quality suite is the validator (validated PRISTINE×3,
                // permits non-byte-identical output). Env
                // `LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED=0` reverts
                // to the legacy per-token loop.
                //
                // Buffer convention (gemm_residual_f16.msl:198):
                //   buf0 = W (F32 byte-encoded) [N=hidden, K=q_dim]
                //   buf1 = X input [M=batch, K=q_dim]   (ssm_in_role_buf)
                //   buf2 = Y output [M, N=hidden]        (attn_proj_buf)
                //   buf6 = R residual [M, N=hidden]      (x_buf)
                enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32_residual);
                enc.set_buffer(layer_buf, ssm_out_off, 0);            // W F32 [hidden, q_dim]
                enc.set_buffer(ssm_in_role_buf, ssm_in_role_off, 1);   // X [batch, q_dim]
                enc.set_buffer(attn_proj_buf, 0, 2);                    // Y [batch, hidden]
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);   // M
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);   // N
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);        // K
                enc.set_buffer(x_buf, 0, 6);                            // R residual [batch, hidden]
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // Fallback: per-token SSM Out matvec + separate residual_add_copy (F32 only).
                // this F32 fallback retains scratch_buf-aliasing semantics —
                // any user running with F32 weights pays legacy serialisation.
                for t in 0..batch_size {
                    let silu_out_t_off = ssm_in_role_off + (t as u64) * tok_bytes_qdim;
                    let proj_t_off = (t as u64) * tok_bytes_hidden;
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(ssm_in_role_buf, silu_out_t_off, 1);
                    enc.set_buffer(normed_buf, proj_t_off, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(hidden_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                }
                // F32 fallback inner barrier: needs normed_buf to be retired
                // before residual_add_copy reads it.
                emit_phase_barrier(&enc, &[normed_buf]);

                // Fallback path: separate residual add + copy (only for F32)
                let pso_residual_copy = pipelines.residual_add_copy.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("residual_add_copy pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_residual_copy);
                enc.set_buffer(x_buf, 0, 0);        // dst (residual, R/W)
                enc.set_buffer(normed_buf, 0, 1);    // src (SSM proj output)
                enc.set_buffer(attn_proj_buf, 0, 2); // copy_dst
                let total_residual_elems = (batch_size * hidden_dim) as u32;
                enc.set_bytes(&total_residual_elems.to_le_bytes(), 3);
                {
                    let fused_tg = 256u64;
                    enc.dispatch_threadgroups(
                        MTLSize::new((total_residual_elems as u64).div_ceil(fused_tg), 1, 1),
                        MTLSize::new(fused_tg, 1, 1),
                    );
                }
            }

            // Phase 3 → FFN-norm barrier: FFN-norm reads attn_proj_buf,
            // which was written by the fused residual GEMM (or
            // residual_add_copy in F32 fallback). In legacy mode the
            // attn_proj_buf is not aliased; in concurrent-encoder mode we still need an
            // explicit barrier so the concurrent encoder serialises against
            // the residual write.
            emit_phase_barrier(&enc, &[attn_proj_buf]);

            // GPU-time census split: close the Phase-3 ssm_out GEMM CB (records its
            // GPU wall), label the FFN-norm, reopen serial encoder. No-op off.
            if census {
                enc.end_encoding();
                super::profile::set_section("gdn/p3b_ffn_rmsnorm");
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("GDN census: ffn-norm encoder reopen".into())
                })?;
            }

            // FFN RMSNorm: fused into same encoder (saves 1 encoder boundary per layer).
            // Reads attn_proj_buf (written by fused residual GEMM or residual_add_copy above)
            // and writes normed_buf for the FFN gate+up dispatch.
            let ffn_norm_off = meta.ffn_norm_off;
            enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
            enc.set_buffer(attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );

            enc.end_encoding();
        }

        // Conv position after batched prefill: advance by batch_size tokens.
        // The circular buffer has buf_slots = kernel_size - 1 slots.
        // Each token writes to one slot and advances the position by 1.
        // After T tokens: new_pos = (old_pos + T) % buf_slots.
        let new_conv_pos = (conv_pos + batch_size as u32) % buf_slots;

        Ok(new_conv_pos)
    }

    /// Dual-queue GDN prefill: bit-identical to `encode_batched_gdn_prefill`'s
    /// Q8 path, but splits the per-layer branchy DAG across two command queues
    /// coordinated by `MetalSharedEvent` so the independent branches overlap.
    ///
    /// Main CB (`cmd`, main queue):
    ///   E0 RMSNorm(x->normed) ; signal(norm_ready,ord)
    ///   branch A: QKV-GEMM -> conv1d+SiLU -> conv_state_update -> L2
    ///   wait(ab_ready,ord)
    ///   recurrence (Phase 2a state update)
    ///   wait(gate_ready,ord)
    ///   join tail: gated-RMSNorm(Phase 2b) -> ssm_out-GEMM+residual(Phase 3) -> FFN-RMSNorm
    ///
    /// Aux CB (`aux_cmd`, aux queue):
    ///   wait(norm_ready,ord)
    ///   branch B: alpha-GEMM, beta-GEMM -> compute_gates
    ///   signal(ab_ready,ord)
    ///   branch C: attn-gate-GEMM
    ///   signal(gate_ready,ord)
    ///
    /// Each kernel dispatch (pipeline, set_buffer/set_bytes order, grid/TG, FP
    /// accumulation order) is byte-identical to the single-encoder Q8 path; only
    /// the encoder/CB/queue placement changes. signal/wait are CB-granularity:
    /// the caller must NOT have an encoder open across them — this function ends
    /// each encoder before the signal/wait.
    ///
    /// `ord` is the GDN ordinal (1-based) used as the monotonic event value on
    /// all three per-prefill events. Preconditions (caller-checked): Q8 wq,
    /// attn_gate, ssm_out; de-aliased role buffers present; parallel conv1d
    /// pipeline present. Returns the new conv position.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batched_gdn_prefill_dual_queue(
        cmd: &MetalCommandBuffer,
        aux_cmd: &MetalCommandBuffer,
        ev_norm_ready: &super::ffi::MetalSharedEvent,
        ev_ab_ready: &super::ffi::MetalSharedEvent,
        ev_gate_ready: &super::ffi::MetalSharedEvent,
        ord: u64,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        meta: &CachedLayerMeta,
        gdn_idx: usize,
        x_buf: &MetalBuffer,
        normed_buf: &MetalBuffer,
        qkv_buf: &MetalBuffer,
        attn_out_buf: &MetalBuffer,
        attn_proj_buf: &MetalBuffer,
        batch_size: usize,
    ) -> Result<u32, RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let num_heads = s.gdn_num_v_heads;
        let num_kv_heads = s.gdn_num_k_heads;
        let head_dim = s.gdn_head_dim;
        let qk_dim = num_kv_heads * head_dim;
        let value_dim = num_heads * head_dim;
        let q_dim = value_dim;
        let qkv_dim = 2 * qk_dim + value_dim;
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;
        let conv_kernel_size = s.gdn_conv_kernel_size;

        let ssm_conv1d_off = meta.ssm_conv1d_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_conv1d_off".into())
        })?;
        let ssm_dt_off = meta.ssm_dt_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_dt_off".into())
        })?;
        let ssm_a_off = meta.ssm_a_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_a_off".into())
        })?;
        let ssm_beta_off = meta.ssm_beta_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_beta_off".into())
        })?;
        let ssm_alpha_off = meta.ssm_alpha_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_alpha_off".into())
        })?;
        let ssm_norm_off = meta.ssm_norm_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_norm_off".into())
        })?;
        let ssm_out_off = meta.ssm_out_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing ssm_out_off".into())
        })?;
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing attn_gate_off".into())
        })?;
        let attn_norm_off = meta.attn_norm_off;
        let ffn_norm_off = meta.ffn_norm_off;

        let conv_state_buf = &s.gdn_conv_states[gdn_idx];
        let conv_pos = s.gdn_conv_positions[gdn_idx];
        let h_state_buf = &s.gdn_h_states[gdn_idx];

        let buf_slots = (conv_kernel_size - 1) as u32;
        let gate_all_buf = attn_out_buf;

        // De-aliased role buffers (dedicated MTLBuffers; required precondition).
        let alpha_role_buf = s.batch_gdn_alpha_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing batch_gdn_alpha_buf".into())
        })?;
        let beta_role_buf = s.batch_gdn_beta_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing batch_gdn_beta_buf".into())
        })?;
        let conv_out_role_buf = s.batch_gdn_conv_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing batch_gdn_conv_out_buf".into())
        })?;
        let raw_out_role_buf = s.batch_gdn_raw_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing batch_gdn_raw_out_buf".into())
        })?;
        let ssm_in_role_buf = s.batch_gdn_ssm_in_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("GDN dual-queue: missing batch_gdn_ssm_in_buf".into())
        })?;

        // Resource-scoped barrier helper (concurrent encoders, same as the
        // production concurrent path: resource-scoped barriers within an
        // encoder; cross-encoder/CB hazards are handled by the events + Metal's
        // automatic boundary hazard tracking).
        let barrier = |enc: &MetalComputeEncoder, bufs: &[&MetalBuffer]| {
            enc.memory_barrier_with_resources(bufs);
        };

        // ===============================================================
        // MAIN CB — E0: RMSNorm(x -> normed)
        // ===============================================================
        {
            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: E0 encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
            enc.set_buffer(x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }
        // normed_buf is now produced on the main queue; signal aux to start B/C.
        cmd.encode_signal_event(ev_norm_ready, ord);

        // ===============================================================
        // AUX CB — wait(norm) ; branch B (alpha,beta,gates) ; signal(ab) ;
        //          branch C (attn-gate GEMM) ; signal(gate)
        // ===============================================================
        aux_cmd.encode_wait_for_event(ev_norm_ready, ord);
        {
            let enc = aux_cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: aux branch-B encoder".into())
            })?;
            // Alpha GEMM (Q8 k64): [T,hidden] @ [hidden,num_heads] -> alpha_role
            let ab_aligned = batch_size % 32 == 0 && num_heads % 32 == 0 && hidden_dim % 32 == 0;
            let pso_ab = if ab_aligned && hidden_dim % 64 == 0 {
                &pipelines.dequant_tiled_matmul_q8_0_k64_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64
            };
            enc.set_pipeline_state(pso_ab);
            enc.set_threadgroup_memory_length(8192, 0);
            enc.set_buffer(layer_buf, ssm_alpha_off, 0);
            enc.set_buffer(normed_buf, 0, 1);
            enc.set_buffer(alpha_role_buf, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
            // Beta GEMM (Q8 k64): -> beta_role
            enc.set_pipeline_state(pso_ab);
            enc.set_threadgroup_memory_length(8192, 0);
            enc.set_buffer(layer_buf, ssm_beta_off, 0);
            enc.set_buffer(normed_buf, 0, 1);
            enc.set_buffer(beta_role_buf, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
            // compute-gates: alpha/beta -> gated (in-place). Barrier: reads alpha/beta.
            barrier(&enc, &[alpha_role_buf, beta_role_buf]);
            let pso_gates = pipelines.gdn_compute_gates_batched.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_compute_gates_batched pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso_gates);
            enc.set_buffer(layer_buf, ssm_dt_off, 0);
            enc.set_buffer(layer_buf, ssm_a_off, 1);
            enc.set_buffer(beta_role_buf, 0, 2);
            enc.set_buffer(alpha_role_buf, 0, 3);
            enc.set_buffer(alpha_role_buf, 0, 4);
            enc.set_buffer(beta_role_buf, 0, 5);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            let total_gates = (num_heads * batch_size) as u64;
            enc.dispatch_threadgroups(
                MTLSize::new(total_gates.div_ceil(256), 1, 1),
                MTLSize::new(256u64.min(total_gates), 1, 1),
            );
            enc.end_encoding();
        }
        // Branch B (alpha,beta,gates) ready -> unblock main's recurrence.
        aux_cmd.encode_signal_event(ev_ab_ready, ord);
        {
            // Branch C: attn-gate GEMM (Q8 k64) -> gate_all_buf
            let enc = aux_cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: aux branch-C encoder".into())
            })?;
            let c_aligned = batch_size % 32 == 0 && q_dim % 32 == 0 && hidden_dim % 32 == 0;
            let pso_c = if c_aligned && hidden_dim % 64 == 0 {
                &pipelines.dequant_tiled_matmul_q8_0_k64_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64
            };
            enc.set_pipeline_state(pso_c);
            enc.set_threadgroup_memory_length(8192, 0);
            enc.set_buffer(layer_buf, attn_gate_off, 0);
            enc.set_buffer(normed_buf, 0, 1);
            enc.set_buffer(gate_all_buf, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new((q_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        }
        // Branch C (attn-gate) ready -> unblock main's gated-RMSNorm.
        aux_cmd.encode_signal_event(ev_gate_ready, ord);

        // ===============================================================
        // MAIN CB — branch A: QKV-GEMM -> conv1d+SiLU -> conv_state_update -> L2
        // ===============================================================
        {
            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: branch-A encoder".into())
            })?;
            // QKV GEMM (Q8 k64): normed -> qkv_buf
            let qkv_aligned = batch_size % 32 == 0 && qkv_dim % 32 == 0 && hidden_dim % 32 == 0;
            let pso_qkv = if qkv_aligned && hidden_dim % 64 == 0 {
                &pipelines.dequant_tiled_matmul_q8_0_k64_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64
            };
            enc.set_pipeline_state(pso_qkv);
            enc.set_threadgroup_memory_length(8192, 0);
            enc.set_buffer(layer_buf, meta.wq_off, 0);
            enc.set_buffer(normed_buf, 0, 1);
            enc.set_buffer(qkv_buf, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
            // conv1d+SiLU: qkv_buf -> conv_out_role. Barrier: reads qkv_buf.
            barrier(&enc, &[qkv_buf]);
            let pso_conv1d_par = pipelines.ssm_conv1d_silu_prefill_parallel.as_ref().ok_or_else(|| {
                RuntimeError::Compute(
                    "GDN dual-queue requires ssm_conv1d_silu_prefill_parallel pipeline".into())
            })?;
            let conv_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.set_pipeline_state(pso_conv1d_par);
            enc.set_buffer(qkv_buf, 0, 0);
            enc.set_buffer(conv_state_buf, 0, 1);
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2);
            enc.set_buffer(conv_out_role_buf, 0, 3);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), batch_size as u64, 1),
                MTLSize::new(conv_tg, 1, 1),
            );
            // conv_state circular-buffer update (race-free; reads qkv_buf, writes conv_state).
            let pso_su = pipelines.ssm_conv1d_state_update.as_ref().ok_or_else(|| {
                RuntimeError::Compute("ssm_conv1d_state_update pipeline not compiled".into())
            })?;
            barrier(&enc, &[conv_state_buf]);
            let su_tg = 256u64.min(qkv_dim as u64).max(1);
            enc.set_pipeline_state(pso_su);
            enc.set_buffer(qkv_buf, 0, 0);
            enc.set_buffer(conv_state_buf, 0, 1);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 2);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&conv_pos.to_le_bytes(), 4);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new((qkv_dim as u64).div_ceil(su_tg), 1, 1),
                MTLSize::new(su_tg, 1, 1),
            );
            // L2 normalize q/k in conv_out (in-place). Barrier: reads conv_out.
            barrier(&enc, &[conv_out_role_buf]);
            let pso_l2 = pipelines.l2_normalize_qk_strided.as_ref().ok_or_else(|| {
                RuntimeError::Compute("l2_normalize_qk_strided pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso_l2);
            enc.set_buffer(conv_out_role_buf, 0, 0);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 1);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(0u32).to_le_bytes(), 5);
            enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 6);
            enc.dispatch_threadgroups(
                MTLSize::new((num_kv_heads * batch_size) as u64, 1, 1),
                MTLSize::new(head_dim as u64, 1, 1),
            );
            enc.end_encoding();
        }

        // Main must not run recurrence until branch B (alpha/beta/gates) retires.
        cmd.encode_wait_for_event(ev_ab_ready, ord);

        // ===============================================================
        // MAIN CB — recurrence (Phase 2a state update)
        // ===============================================================
        {
            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: recurrence encoder".into())
            })?;
            let pso_v3 = pipelines.gdn_prefill_fused_v3_chunked.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_prefill_fused_v3_chunked pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso_v3);
            enc.set_buffer(h_state_buf, 0, 0);
            enc.set_buffer(conv_out_role_buf, 0, 1);
            enc.set_buffer(alpha_role_buf, 0, 2);
            enc.set_buffer(beta_role_buf, 0, 3);
            enc.set_buffer(raw_out_role_buf, 0, 4);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 5);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 6);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 7);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 9);
            enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 10);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 11);
            enc.dispatch_threadgroups(
                MTLSize::new(1, head_dim as u64, num_heads as u64),
                MTLSize::new(32, 1, 1),
            );
            enc.end_encoding();
        }

        // Main must not run gated-RMSNorm until branch C (attn-gate) retires.
        cmd.encode_wait_for_event(ev_gate_ready, ord);

        // ===============================================================
        // MAIN CB — join tail: Phase 2b (gated-RMSNorm) -> Phase 3
        //           (ssm_out GEMM + residual) -> FFN-RMSNorm
        // ===============================================================
        {
            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN dual-queue: join-tail encoder".into())
            })?;
            // Phase 2b: gated-RMSNorm(raw_out, gate_all -> ssm_in)
            let pso_norm_gate = pipelines.gdn_prefill_norm_gate.as_ref().ok_or_else(|| {
                RuntimeError::Compute("gdn_prefill_norm_gate pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso_norm_gate);
            enc.set_buffer(raw_out_role_buf, 0, 0);
            enc.set_buffer(gate_all_buf, 0, 1);
            enc.set_buffer(layer_buf, ssm_norm_off, 2);
            enc.set_buffer(ssm_in_role_buf, 0, 3);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&eps.to_le_bytes(), 6);
            enc.set_bytes(&(1u32).to_le_bytes(), 7);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 8);
            enc.dispatch_threadgroups(
                MTLSize::new(num_heads as u64, batch_size as u64, 1),
                MTLSize::new(head_dim as u64, 1, 1),
            );
            // Phase 2b -> Phase 3 barrier: ssm_out GEMM reads ssm_in.
            barrier(&enc, &[ssm_in_role_buf]);
            // Phase 3: ssm_out GEMM (Q8 k64 residual) + residual add -> attn_proj_buf
            let p3_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
            let pso_ssm = if p3_aligned && q_dim % 64 == 0 {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched
            };
            enc.set_pipeline_state(pso_ssm);
            enc.set_threadgroup_memory_length(8192, 0);
            enc.set_buffer(layer_buf, ssm_out_off, 0);
            enc.set_buffer(ssm_in_role_buf, 0, 1);
            enc.set_buffer(attn_proj_buf, 0, 2);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);
            enc.set_buffer(x_buf, 0, 6);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
            // Phase 3 -> FFN-norm barrier: FFN-norm reads attn_proj_buf.
            barrier(&enc, &[attn_proj_buf]);
            // FFN RMSNorm: attn_proj_buf -> normed_buf
            enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
            enc.set_buffer(attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        let new_conv_pos = (conv_pos + batch_size as u32) % buf_slots;
        Ok(new_conv_pos)
    }

    /// Reset GatedDeltaNet recurrent state for a new sequence.
    ///
    /// Must be called between sequences to clear the h_state matrices and
    /// conv1d circular buffers. Without this, the GDN state from the previous
    /// sequence leaks into the new one, producing incorrect output.
    pub fn reset_gdn_state(&self) {
        let mut scratch_guard = self.scratch.lock().unwrap();
        if let Some(ref mut s) = *scratch_guard {
            if s.gdn_num_layers > 0 {
                // GDN-specific dimensions — must NOT use s.num_heads / s.head_dim / s.kv_dim
                // which are full-attention hyperparams. Use the resolved SSM dims
                // (9B {32,16,128,4} default, 27B {48,16,128,4}). These MUST match
                // the sizes used at allocation so reset zeroes the exact buffers.
                let gdn_num_v_heads = s.gdn_num_v_heads; // ssm.time_step_rank
                let gdn_num_k_heads = s.gdn_num_k_heads; // ssm.group_count
                let gdn_head_dim = s.gdn_head_dim;       // ssm.state_size
                let conv_kernel_size = s.gdn_conv_kernel_size;
                let h_state_size = gdn_num_v_heads * gdn_head_dim * gdn_head_dim;
                let gdn_qkv_dim = 2 * gdn_num_k_heads * gdn_head_dim + gdn_num_v_heads * gdn_head_dim;
                let conv_state_size = (conv_kernel_size - 1) * gdn_qkv_dim;

                for h_buf in &s.gdn_h_states {
                    h_buf.write_f32(&vec![0.0f32; h_state_size]);
                }
                for c_buf in &s.gdn_conv_states {
                    c_buf.write_f32(&vec![0.0f32; conv_state_size]);
                }
                for pos in &mut s.gdn_conv_positions {
                    *pos = 0;
                }
            }
        }
    }
}
