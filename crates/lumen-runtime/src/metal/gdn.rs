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
        // GDN uses different head/dim layout than full-attention layers.
        // GGUF: ssm.time_step_rank=32, ssm.state_size=128, ssm.group_count=16, ssm.inner_size=4096
        //
        // Reference (Qwen3_5MoeGatedDeltaNet in transformers):
        //   Q+K: num_kv_heads=16 heads × 128 each (repeated_interleave to 32 before GDN)
        //   V:   num_v_heads=32  heads × 128 each (no repeat)
        //   conv_dim = Q(2048) + K(2048) + V(4096) = 8192
        //   value_dim (inner_size) = 4096 — gate and output projection output size
        let num_heads = 32usize;      // ssm.time_step_rank = num_v_heads (state/V heads)
        let num_kv_heads = 16usize;   // ssm.group_count = num_k_heads (Q and K pre-repeat heads)
        let head_dim = 128usize;      // ssm.state_size
        let qk_dim = 2048usize;       // Q and K each: num_kv_heads * head_dim = 16 * 128
        let value_dim = 4096usize;    // V: num_heads * head_dim = 32 * 128 = inner_size
        let q_dim = value_dim;        // alias: gate/output projection size = value_dim = 4096
        let qkv_dim = 8192usize;      // Q(2048) + K(2048) + V(4096) = 8192
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
        let ssm_out_quant = meta.ssm_out_quant.unwrap_or(QuantScheme::F32);
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
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
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, wq_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.qkv_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            }
            let n_tg = match meta.wq_quant { QuantScheme::Q8_0 => ((qkv_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2, _ => qkv_dim as u64 };
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
        // Reference: llama.cpp applies SiLU to the entire QKV after conv1d.
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
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, attn_gate_off, 0);      // weights [q_dim x hidden_dim]
            enc.set_buffer(&s.normed_buf, 0, 1);               // input x_norm [hidden_dim=2048]
            enc.set_buffer(gate_sigmoid_buf, 0, 2);             // output [q_dim=4096]
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);  // in_dim = hidden_dim = 2048
            if matches!(attn_gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 4);   // out_dim = q_dim = 4096
            }
            let n_tg = match attn_gate_quant { QuantScheme::Q8_0 => ((q_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((q_dim as u64) + 1) / 2, QuantScheme::F16 => ((q_dim as u64) + 1) / 2, _ => q_dim as u64 };
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
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, ssm_out_off, 0);          // weights
            enc.set_buffer(gate_sigmoid_buf, 0, 1);              // input [4096] (gated)
            enc.set_buffer(ssm_proj_buf, 0, 2);                  // output [2048]
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);     // in_dim = q_dim = 4096
            if matches!(ssm_out_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);  // out_dim = 2048
            }
            let n_tg = match ssm_out_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
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
        // Reference layout: Q(2048) + K(2048) + V(4096) = 8192
        // Q and K each have num_kv_heads=16 pre-repeat heads; V has num_heads=32 heads.
        let num_heads = 32usize;      // num_v_heads = state/V heads
        let num_kv_heads = 16usize;   // num_k_heads = Q and K pre-repeat heads
        let head_dim = 128usize;
        let qk_dim = 2048usize;       // Q and K each: 16 * 128
        let value_dim = 4096usize;    // V: 32 * 128 (= inner_size)
        let q_dim = value_dim;        // alias for gate/output projection size
        let qkv_dim = 8192usize;
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
        let ssm_out_quant = meta.ssm_out_quant.unwrap_or(QuantScheme::F32);
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN layer missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
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
            enc.memory_barrier_with_scope(1);
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
            } else if matches!(meta.wq_quant, QuantScheme::Q4_0 | QuantScheme::F16) {
                // Q4_0/F16 fused: RMSNorm + matvec in one kernel
                match meta.wq_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
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
                enc.memory_barrier_with_scope(1);
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
            } else if matches!(attn_gate_quant, QuantScheme::Q4_0 | QuantScheme::F16) {
                // Q4_0/F16 fused: RMSNorm + matvec in one kernel
                match attn_gate_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
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
        enc.memory_barrier_with_scope(1);

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
            enc.memory_barrier_with_scope(1);

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
                enc.memory_barrier_with_scope(1);
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
            enc.memory_barrier_with_scope(1); // raw_out ready

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
                enc.memory_barrier_with_scope(1); // gates ready
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
            enc.memory_barrier_with_scope(1);

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
                enc.memory_barrier_with_scope(1);
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
                enc.memory_barrier_with_scope(1); // raw_out ready

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
                enc.memory_barrier_with_scope(1);

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

        enc.memory_barrier_with_scope(1); // normed_out_buf ready

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
                    enc.memory_barrier_with_scope(1);

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
                    enc.memory_barrier_with_scope(1);

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
                    enc.memory_barrier_with_scope(1);

                    // F16 matvec -> ssm_proj_buf, then manual residual+copy
                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(ssm_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                    let n_tg = ((hidden_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    enc.memory_barrier_with_scope(1);

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
                    enc.memory_barrier_with_scope(1);

                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(gate_sigmoid_buf, 0, 1);
                    enc.set_buffer(ssm_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(hidden_dim as u64, 1, 1),
                        MTLSize::new(s.matmul_tg_size, 1, 1),
                    );
                    enc.memory_barrier_with_scope(1);

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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batched_gdn_prefill(
        cmd: &MetalCommandBuffer,
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
        let num_heads = 32usize;
        let num_kv_heads = 16usize;
        let head_dim = 128usize;
        let qk_dim = 2048usize;
        let value_dim = 4096usize;
        let q_dim = value_dim;
        let qkv_dim = 8192usize;
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
        let ssm_out_quant = meta.ssm_out_quant.unwrap_or(QuantScheme::F32);
        let attn_gate_off = meta.attn_gate_off.ok_or_else(|| {
            RuntimeError::Compute("GDN batched prefill: missing attn_gate_off".into())
        })?;
        let attn_gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
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
        // PHASE 1: Pre-compute token-independent projections
        // ================================================================

        // Phase 1: Merged RMSNorm + QKV GEMM + Gate GEMM + Alpha/Beta matvec + Conv1d + SiLU
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("GDN prefill: phase 1 encoder".into())
            })?;

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
            enc.memory_barrier_with_scope(1);

            // QKV: tiled GEMM dispatch [batch_size, hidden_dim] @ [hidden_dim, qkv_dim] -> [batch_size, qkv_dim]
            // Tiled GEMM reads weight tiles once across all output rows (vs GEMV reading full weights per row).
            if matches!(meta.wq_quant, QuantScheme::Q8_0) {
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
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
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

            // Gate: tiled GEMM dispatch [batch_size, hidden_dim] @ [hidden_dim, q_dim] -> [batch_size, q_dim]
            if matches!(attn_gate_quant, QuantScheme::Q8_0) {
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
            // Alpha/Beta projections via tiled GEMM + separate batched gate computation.
            // Tiled GEMM amortizes weight loads across batch_size output rows (vs GEMV per-row).
            // Output: scratch_buf[0..T*num_heads] = alpha (decay), [T*num_heads..2*T*num_heads] = beta (sigmoid)
            let alpha_all_bytes = (batch_size * num_heads * 4) as u64;
            {
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
                enc.set_buffer(scratch_buf, 0, 2);              // output [batch_size, num_heads] (raw alpha)
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // N = 32
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K = 4096
                enc.dispatch_threadgroups(
                    MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );

                // Beta GEMM: [batch_size, hidden_dim] @ [hidden_dim, num_heads] -> [batch_size, num_heads]
                if gemm_aligned && hidden_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, ssm_beta_off, 0);    // weights [num_heads, hidden_dim] Q8_0
                enc.set_buffer(normed_buf, 0, 1);               // input [batch_size, hidden_dim]
                enc.set_buffer(scratch_buf, alpha_all_bytes, 2); // output [batch_size, num_heads] (raw beta)
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);   // N = 32
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);  // K = 4096
                enc.dispatch_threadgroups(
                    MTLSize::new((num_heads as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );

                // Batched gate computation: transform raw alpha/beta to gated values
                // alpha_out = exp(ssm_a * softplus(alpha_raw + dt_bias))  -- decay in (0,1)
                // beta_out  = sigmoid(beta_raw)                          -- mixing rate
                enc.memory_barrier_with_scope(1);
                let pso_gates_batched = pipelines.gdn_compute_gates_batched.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("gdn_compute_gates_batched pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_gates_batched);
                enc.set_buffer(layer_buf, ssm_dt_off, 0);          // dt_bias [n_heads]
                enc.set_buffer(layer_buf, ssm_a_off, 1);           // ssm_a [n_heads]
                enc.set_buffer(scratch_buf, alpha_all_bytes, 2);   // beta_raw [T * n_heads] (input)
                enc.set_buffer(scratch_buf, 0, 3);                 // alpha_raw [T * n_heads] (input)
                enc.set_buffer(scratch_buf, 0, 4);                 // alpha_out [T * n_heads] (overwrite in-place)
                enc.set_buffer(scratch_buf, alpha_all_bytes, 5);   // beta_out [T * n_heads] (overwrite in-place)
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 6);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
                let total_gates = (num_heads * batch_size) as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(total_gates.div_ceil(256), 1, 1),
                    MTLSize::new(256u64.min(total_gates), 1, 1),
                );
            }

            // Batched conv1d: qkv_buf[T, qkv_dim] -> scratch_buf conv_out region
            // Needs barrier: reads from qkv_buf written by batched QKV matvec above
            enc.memory_barrier_with_scope(1);

            // Fused Conv1d + SiLU -- token-parallel variant dispatches (dim_blocks, T) TGs
            // Falls back to serial kernel if parallel pipeline unavailable
            let pso_conv1d_silu_par = pipelines.ssm_conv1d_silu_prefill_parallel.as_ref();

            // scratch_buf layout:
            //   [0 .. T*num_heads*4)                     = alpha_raw[T, num_heads]
            //   [T*num_heads*4 .. 2*T*num_heads*4)       = beta_raw[T, num_heads]
            //   [2*T*num_heads*4 .. 2*T*num_heads*4 + T*qkv_dim*4) = conv_out[T, qkv_dim]
            let alpha_all_bytes = (batch_size * num_heads * 4) as u64;
            let conv_out_off = 2 * alpha_all_bytes;

            enc.set_buffer(qkv_buf, 0, 0);        // input [T * qkv_dim]
            enc.set_buffer(conv_state_buf, 0, 1);  // conv_state circular buffer (R/W)
            enc.set_buffer(layer_buf, ssm_conv1d_off, 2); // kernel weights
            enc.set_buffer(scratch_buf, conv_out_off, 3); // output [T * qkv_dim] (SiLU-activated)
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(conv_kernel_size as u32).to_le_bytes(), 5);
            enc.set_bytes(&conv_pos.to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            if let Some(pso_par) = pso_conv1d_silu_par {
                // Parallel: (ceil(dim/TG_SIZE), T) TGs -- each handles one (channel_block, token)
                let conv_tg = 256u64.min(qkv_dim as u64).max(1);
                enc.set_pipeline_state(pso_par);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(conv_tg), batch_size as u64, 1),
                    MTLSize::new(conv_tg, 1, 1),
                );
            } else {
                // Fallback: serial kernel -- ceil(dim/256) TGs, each loops over T
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

            // L2 normalize Q and K in-place within conv_out (after SiLU)
            enc.memory_barrier_with_scope(1);
            {
                let alpha_all_bytes_inner = (batch_size * num_heads * 4) as u64;
                let conv_out_off_inner = 2 * alpha_all_bytes_inner;
                let pso_l2 = pipelines.l2_normalize_qk_strided.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("l2_normalize_qk_strided pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_l2);
                enc.set_buffer(scratch_buf, conv_out_off_inner, 0);  // conv_out [T, qkv_dim]
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 1);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 2);
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);
                enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);    // stride
                enc.set_bytes(&(0u32).to_le_bytes(), 5);              // q_offset = 0
                enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 6);     // k_offset = qk_dim
                enc.dispatch_threadgroups(
                    MTLSize::new((num_kv_heads * batch_size) as u64, 1, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );
            }

            // scratch_buf layout (finalized):
            //   [0 .. T*num_heads*4)                     = alpha_all[T, num_heads] (precomputed gates)
            //   [T*num_heads*4 .. 2*T*num_heads*4)       = beta_all[T, num_heads] (precomputed gates)
            //   [2*T*num_heads*4 .. ...)                  = conv_out[T, qkv_dim] (SiLU+L2-normalized Q/K)
            let alpha_all_bytes = (batch_size * num_heads * 4) as u64;
            let conv_out_off = 2 * alpha_all_bytes;
            let raw_out_off = (batch_size * q_dim * 4) as u64;

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

            enc.memory_barrier_with_scope(1);

            // Phase 2a: v3 chunked state update (4x unrolled, simdgroup-parallel)
            enc.set_pipeline_state(pso_v3);
            enc.set_buffer(h_state_buf, 0, 0);                       // h_state [n_heads * val_dim * key_dim] (transposed layout)
            enc.set_buffer(scratch_buf, conv_out_off, 1);             // conv_out_all [T, qkv_dim]
            enc.set_buffer(scratch_buf, 0, 2);                        // alpha_all [T, n_heads]
            enc.set_buffer(scratch_buf, alpha_all_bytes, 3);          // beta_all [T, n_heads]
            enc.set_buffer(qkv_buf, raw_out_off, 4);                  // raw_out [T, q_dim] (upper half of qkv_buf)
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 5);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 6);       // key_dim
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 7);       // val_dim
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 9);     // T
            enc.set_bytes(&(qk_dim as u32).to_le_bytes(), 10);
            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 11);
            enc.dispatch_threadgroups(
                MTLSize::new(1, head_dim as u64, num_heads as u64),
                MTLSize::new(32, 1, 1),
            );
            enc.memory_barrier_with_scope(1);

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
            enc.set_pipeline_state(pso_norm_gate);
            enc.set_buffer(qkv_buf, raw_out_off, 0);                  // raw_out [T, q_dim]
            enc.set_buffer(gate_all_buf, 0, 1);                       // gate_all [T, q_dim]
            enc.set_buffer(layer_buf, ssm_norm_off, 2);               // norm_scale
            enc.set_buffer(qkv_buf, 0, 3);                            // ssm_out [T, q_dim] (output)
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 5);       // val_dim
            enc.set_bytes(&eps.to_le_bytes(), 6);
            enc.set_bytes(&(1u32).to_le_bytes(), 7);                  // scale_n_heads
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 8);     // T
            enc.dispatch_threadgroups(
                MTLSize::new(num_heads as u64, batch_size as u64, 1),
                MTLSize::new(head_dim as u64, 1, 1),
            );
            enc.memory_barrier_with_scope(1);

            // Phase 3: Fused SSM Out GEMM + residual add
            // attn_proj_buf[m,n] = SSM_Out_GEMM(qkv_buf)[m,n] + x_buf[m,n]
            // Eliminates separate residual_add_copy dispatch + barrier.
            // The FFN down projection later does: x_buf = Down * gate + attn_proj_buf,
            // so x_buf is correctly updated for the next layer without an extra copy.

            if matches!(ssm_out_quant, QuantScheme::Q8_0) {
                let gemm_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
                if gemm_aligned && q_dim % 64 == 0 {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned);
                } else {
                    enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched);
                }
                enc.set_threadgroup_memory_length(8192, 0);
                enc.set_buffer(layer_buf, ssm_out_off, 0);   // W_q8 weights [hidden_dim, q_dim]
                enc.set_buffer(qkv_buf, 0, 1);               // X input [batch_size, q_dim]
                enc.set_buffer(attn_proj_buf, 0, 2);          // Y output [batch_size, hidden_dim]
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);  // N
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);       // K
                enc.set_buffer(x_buf, 0, 6);                  // R residual [batch_size, hidden_dim]
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else if matches!(ssm_out_quant, QuantScheme::Q4_0) {
                // Fused Q4_0 tiled GEMM + residual: attn_proj_buf = GEMM(qkv_buf) + x_buf
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
                enc.set_buffer(layer_buf, ssm_out_off, 0);   // W_q4 weights [hidden_dim, q_dim]
                enc.set_buffer(qkv_buf, 0, 1);               // X input [batch_size, q_dim]
                enc.set_buffer(attn_proj_buf, 0, 2);          // Y output [batch_size, hidden_dim]
                enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3);  // M
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);  // N
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);       // K
                enc.set_buffer(x_buf, 0, 6);                  // R residual [batch_size, hidden_dim]
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(32), (batch_size as u64).div_ceil(32), 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // Fallback: per-token SSM Out matvec + separate residual_add_copy (F32 only)
                for t in 0..batch_size {
                    let silu_out_t_off = (t as u64) * tok_bytes_qdim;
                    let proj_t_off = (t as u64) * tok_bytes_hidden;
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, ssm_out_off, 0);
                    enc.set_buffer(qkv_buf, silu_out_t_off, 1);
                    enc.set_buffer(normed_buf, proj_t_off, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(hidden_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                }
                enc.memory_barrier_with_scope(1);

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
                // which are full-attention hyperparams. GDN uses different per-model constants.
                const GDN_NUM_HEADS: usize = 32;    // ssm.time_step_rank
                const GDN_HEAD_DIM: usize = 128;    // ssm.state_size
                const GDN_QKV_DIM: usize = 8192;    // Q(2048)+K(2048)+V(4096)
                let conv_kernel_size = s.gdn_conv_kernel_size;
                let h_state_size = GDN_NUM_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
                let conv_state_size = (conv_kernel_size - 1) * GDN_QKV_DIM;

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
