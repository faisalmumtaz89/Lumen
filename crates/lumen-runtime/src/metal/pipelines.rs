//! Metal compute pipeline compilation.

use super::MetalF32Backend;
use super::ffi::MetalFunctionConstantValues;
use super::shaders::METAL_SHADER_SOURCE;
use super::types::MetalPipelines;
use crate::error::RuntimeError;

impl MetalF32Backend {
    /// Compile all Metal shader pipelines.
    pub(super) fn compile_pipelines(&self) -> Result<MetalPipelines, RuntimeError> {
        let lib = self
            .device
            .new_library_with_source(METAL_SHADER_SOURCE)
            .map_err(RuntimeError::Compute)?;

        macro_rules! make_pipeline {
            ($name:expr) => {{
                let func = lib.get_function($name).ok_or_else(|| {
                    RuntimeError::Compute(format!("Metal kernel '{}' not found", $name))
                })?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        // Create a pipeline specialized with function constants for aligned GEMM.
        // BC_M=false, BC_N=false, BC_K=false: the Metal compiler dead-code-eliminates
        // all boundary checks, producing a faster kernel for aligned dimensions.
        macro_rules! make_aligned_pipeline {
            ($name:expr) => {{
                let fcv = MetalFunctionConstantValues::new();
                fcv.set_bool(false, 10); // FC_BC_M = false (M aligned to TILE_M)
                fcv.set_bool(false, 11); // FC_BC_N = false (N aligned to TILE_N)
                fcv.set_bool(false, 12); // FC_BC_K = false (K aligned to TILE_K)
                let func = lib.get_function_with_constants($name, &fcv)
                    .map_err(RuntimeError::Compute)?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        // Create a pipeline for kernels that use function constants, with BC_M/N/K=true
        // (boundary-checked fallback). Required because once a kernel declares
        // [[function_constant]] attributes, plain newFunctionWithName: no longer works.
        macro_rules! make_bc_pipeline {
            ($name:expr) => {{
                let fcv = MetalFunctionConstantValues::new();
                fcv.set_bool(true, 10);  // FC_BC_M = true (boundary checks enabled)
                fcv.set_bool(true, 11);  // FC_BC_N = true
                fcv.set_bool(true, 12);  // FC_BC_K = true
                let func = lib.get_function_with_constants($name, &fcv)
                    .map_err(RuntimeError::Compute)?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        Ok(MetalPipelines {
            matmul_f32: make_pipeline!("matmul_f32"),
            matmul_f32_deferred: make_pipeline!("matmul_f32_deferred"),
            matmul_bytes_f32: make_pipeline!("matmul_bytes_f32"),
            // F16 decode kernels
            matmul_f16_deferred_nr2: make_pipeline!("matmul_f16_deferred_nr2"),
            matmul_f16_deferred_residual_nr2: make_pipeline!("matmul_f16_deferred_residual_nr2"),
            matmul_f16_deferred_bias_nr2: make_pipeline!("matmul_f16_deferred_bias_nr2"),
            dequant_matmul_q8_0: make_pipeline!("dequant_matmul_q8_0"),
            rmsnorm: make_pipeline!("rmsnorm"),
            rmsnorm_bytes: make_pipeline!("rmsnorm_bytes"),
            rope: make_pipeline!("rope"),
            rope_neox: lib.get_function("rope_neox")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            swiglu: make_pipeline!("swiglu"),
            softmax: make_pipeline!("softmax"),
            attention_scores: make_pipeline!("attention_scores"),
            attention_output: make_pipeline!("attention_output"),
            write_kv_cache: make_pipeline!("write_kv_cache"),
            fused_rope_kv_write: make_pipeline!("fused_rope_kv_write"),
            fused_rope_kv_mha: make_pipeline!("fused_rope_kv_mha"),
            fused_rope_neox_kv_write: lib.get_function("fused_rope_neox_kv_write")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            multi_head_attention: make_pipeline!("multi_head_attention"),
            flash_decode_attention: make_pipeline!("flash_decode_attention"),
            flash_decode_reduce: make_pipeline!("flash_decode_reduce"),
            add_residual: make_pipeline!("add_residual"),
            embed_token: make_pipeline!("embed_token"),
            embed_token_q8_0: make_pipeline!("embed_token_q8_0"),
            embed_token_q4_0: make_pipeline!("embed_token_q4_0"),
            embed_token_f16: make_pipeline!("embed_token_f16"),
            dequant_matmul_q8_0_residual: make_pipeline!("dequant_matmul_q8_0_residual"),
            dequant_matmul_q8_0_multirow: make_pipeline!("dequant_matmul_q8_0_multirow"),
            dequant_matmul_q8_0_residual_multirow: make_pipeline!("dequant_matmul_q8_0_residual_multirow"),
            dequant_matmul_q8_0_4row: make_pipeline!("dequant_matmul_q8_0_4row"),
            dequant_matmul_q8_0_residual_4row: make_pipeline!("dequant_matmul_q8_0_residual_4row"),
            dequant_matmul_q8_0_8row: make_pipeline!("dequant_matmul_q8_0_8row"),
            dequant_matmul_q8_0_residual_8row: make_pipeline!("dequant_matmul_q8_0_residual_8row"),
            dequant_matmul_q8_0_deferred: make_pipeline!("dequant_matmul_q8_0_deferred"),
            dequant_matmul_q8_0_deferred_residual: make_pipeline!("dequant_matmul_q8_0_deferred_residual"),
            dequant_matmul_q8_0_deferred_bias: make_pipeline!("dequant_matmul_q8_0_deferred_bias"),
            dequant_matmul_q8_0_deferred_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_nr2"),
            dequant_matmul_q8_0_deferred_residual_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_residual_nr2"),
            dequant_matmul_q8_0_deferred_bias_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_bias_nr2"),
            // MLX-style 2-SG kernels
            dequant_matmul_q8_0_2sg: make_pipeline!("dequant_matmul_q8_0_2sg"),
            dequant_matmul_q8_0_2sg_residual: make_pipeline!("dequant_matmul_q8_0_2sg_residual"),
            ffn_fused_gate_up_swiglu_q8_0_2sg: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0_2sg"),
            // Q4_0 decode kernels
            dequant_matmul_q4_0: make_pipeline!("dequant_matmul_q4_0"),
            dequant_matmul_q4_0_residual: make_pipeline!("dequant_matmul_q4_0_residual"),
            dequant_matmul_q4_0_4row: make_pipeline!("dequant_matmul_q4_0_4row"),
            dequant_matmul_q4_0_residual_4row: make_pipeline!("dequant_matmul_q4_0_residual_4row"),
            dequant_matmul_q4_0_deferred: make_pipeline!("dequant_matmul_q4_0_deferred"),
            dequant_matmul_q4_0_deferred_residual: make_pipeline!("dequant_matmul_q4_0_deferred_residual"),
            dequant_matmul_q4_0_deferred_bias: make_pipeline!("dequant_matmul_q4_0_deferred_bias"),
            dequant_matmul_q4_0_deferred_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_nr2"),
            dequant_matmul_q4_0_deferred_residual_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_residual_nr2"),
            dequant_matmul_q4_0_deferred_bias_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_bias_nr2"),
            dequant_tiled_matmul_q8_0_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q8_0_residual_batched"),
            // Q4_0 batched prefill kernels
            dequant_tiled_matmul_q4_0: make_pipeline!("dequant_tiled_matmul_q4_0"),
            dequant_tiled_matmul_q4_0_residual_batched: make_pipeline!("dequant_tiled_matmul_q4_0_residual_batched"),
            dequant_tiled_matmul_q4_0_splitk: make_pipeline!("dequant_tiled_matmul_q4_0_splitk"),
            // Q4_1 kernels
            dequant_tiled_matmul_q4_1: make_pipeline!("dequant_tiled_matmul_q4_1"),
            dequant_tiled_matmul_q4_1_residual_batched: make_pipeline!("dequant_tiled_matmul_q4_1_residual_batched"),
            dequant_matmul_q4_1_deferred: make_pipeline!("dequant_matmul_q4_1_deferred"),
            tiled_matmul_bytes_f32_residual: make_pipeline!("tiled_matmul_bytes_f32_residual"),
            tiled_matmul_f16: make_pipeline!("tiled_matmul_f16"),
            tiled_matmul_f16_residual: make_pipeline!("tiled_matmul_f16_residual"),
            tiled_matmul_f16_k64: make_bc_pipeline!("tiled_matmul_f16_k64"),
            tiled_matmul_f16_k64_residual: make_bc_pipeline!("tiled_matmul_f16_k64_residual"),
            matmul_bytes_f32_residual: make_pipeline!("matmul_bytes_f32_residual"),
            copy_buffer: make_pipeline!("copy_buffer"),
            add_write: make_pipeline!("add_write"),

            // Split-K GEMM kernels
            dequant_tiled_matmul_q8_0_splitk: make_pipeline!("dequant_tiled_matmul_q8_0_splitk"),
            reduce_splitk: make_pipeline!("reduce_splitk"),

            // K64 GEMM variants
            dequant_tiled_matmul_q8_0_k64: make_bc_pipeline!("dequant_tiled_matmul_q8_0_k64"),
            dequant_tiled_matmul_q8_0_k64_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q8_0_k64_residual_batched"),
            dequant_tiled_matmul_q4_0_k64: make_bc_pipeline!("dequant_tiled_matmul_q4_0_k64"),
            dequant_tiled_matmul_q4_0_k64_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q4_0_k64_residual_batched"),

            // Function-constant-specialized aligned GEMM variants (no boundary checks)
            dequant_tiled_matmul_q8_0_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0"),
            dequant_tiled_matmul_q8_0_k64_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_k64"),
            dequant_tiled_matmul_q8_0_k64_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_k64_residual_batched"),
            dequant_tiled_matmul_q8_0_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_residual_batched"),
            dequant_tiled_matmul_q4_0_k64_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q4_0_k64"),
            dequant_tiled_matmul_q4_0_k64_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q4_0_k64_residual_batched"),
            tiled_matmul_f16_k64_aligned: make_aligned_pipeline!("tiled_matmul_f16_k64"),
            tiled_matmul_f16_k64_residual_aligned: make_aligned_pipeline!("tiled_matmul_f16_k64_residual"),

            // Batched prefill kernels
            tiled_matmul_f32: make_pipeline!("tiled_matmul_f32"),
            tiled_matmul_bytes_f32: make_pipeline!("tiled_matmul_bytes_f32"),
            dequant_tiled_matmul_q8_0: make_bc_pipeline!("dequant_tiled_matmul_q8_0"),
            rmsnorm_batched: make_pipeline!("rmsnorm_batched"),
            rmsnorm_batched_bytes: make_pipeline!("rmsnorm_batched_bytes"),
            rope_batched: make_pipeline!("rope_batched"),
            rope_batched_neox: lib.get_function("rope_batched_neox")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            add_residual_batched: make_pipeline!("add_residual_batched"),
            swiglu_batched: make_pipeline!("swiglu_batched"),
            embed_tokens_batched: make_pipeline!("embed_tokens_batched"),
            embed_tokens_batched_q8_0: make_pipeline!("embed_tokens_batched_q8_0"),
            embed_tokens_batched_q4_0: make_pipeline!("embed_tokens_batched_q4_0"),
            embed_tokens_batched_f16: make_pipeline!("embed_tokens_batched_f16"),
            kv_cache_write_batched: make_pipeline!("kv_cache_write_batched"),
            v_cache_write_batched: make_pipeline!("v_cache_write_batched"),
            attention_scores_batched: make_pipeline!("attention_scores_batched"),
            softmax_batched: make_pipeline!("softmax_batched"),
            attention_output_batched: make_pipeline!("attention_output_batched"),
            attention_scores_tiled: make_pipeline!("attention_scores_tiled"),
            attention_output_tiled: make_pipeline!("attention_output_tiled"),
            rmsnorm_dequant_matmul_q8_0_deferred_nr2: make_pipeline!("rmsnorm_dequant_matmul_q8_0_deferred_nr2"),
            rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2: make_pipeline!("rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2"),
            rmsnorm_dequant_matmul_q4_0_deferred_nr2: make_pipeline!("rmsnorm_dequant_matmul_q4_0_deferred_nr2"),
            rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2: make_pipeline!("rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2"),
            // Fused RMSNorm + F16 matvec NR2
            rmsnorm_matmul_f16_deferred_nr2: make_pipeline!("rmsnorm_matmul_f16_deferred_nr2"),
            rmsnorm_matmul_f16_deferred_residual_nr2: make_pipeline!("rmsnorm_matmul_f16_deferred_residual_nr2"),
            rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred"),
            rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row"),
            rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred"),
            rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row"),
            rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred"),
            ffn_fused_gate_up_swiglu_q8_0: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0"),
            ffn_fused_gate_up_swiglu_q8_0_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0_deferred"),
            ffn_fused_gate_up_swiglu_q4_0: make_pipeline!("ffn_fused_gate_up_swiglu_q4_0"),
            ffn_fused_gate_up_swiglu_q4_0_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q4_0_deferred"),
            ffn_fused_gate_up_swiglu_q4_1_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q4_1_deferred"),
            ffn_fused_gate_up_swiglu_f16_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_f16_deferred"),
            argmax: make_pipeline!("argmax"),
            bias_add: make_pipeline!("bias_add"),
            bias_add_batched: make_pipeline!("bias_add_batched"),
            deinterleave_qkv: make_pipeline!("deinterleave_qkv"),

            // MoE pipeline states.
            // Option to provide clear runtime error if Metal compilation fails.
            moe_router_softmax: lib.get_function("moe_router_softmax")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_router_softmax_batched: lib.get_function("moe_router_softmax_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_router_softmax_biased: lib.get_function("moe_router_softmax_biased")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum: lib.get_function("moe_expert_accum")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum_batched: lib.get_function("moe_expert_accum_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum_option_a: lib.get_function("moe_expert_accum_option_a")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Batched MoE expert FFN kernels.
            moe_batched_gate_up_swiglu_q4_0: lib.get_function("moe_batched_gate_up_swiglu_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_gate_up_swiglu_q4_1: lib.get_function("moe_batched_gate_up_swiglu_q4_1")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_gate_up_swiglu_q8_0: lib.get_function("moe_batched_gate_up_swiglu_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q4_0: lib.get_function("moe_batched_down_accum_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q4_1: lib.get_function("moe_batched_down_accum_q4_1")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q8_0: lib.get_function("moe_batched_down_accum_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q8_0: lib.get_function("moe_batched_down_accum_shared_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q8_0_se_q4_0: lib.get_function("moe_batched_down_accum_shared_q8_0_se_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q4_0: lib.get_function("moe_batched_down_accum_shared_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            sigmoid_scale_add: lib.get_function("sigmoid_scale_add")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // GatedDeltaNet (linear attention) pipeline states.
            ssm_conv1d_decode: lib.get_function("ssm_conv1d_decode")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_heads: lib.get_function("l2_normalize_heads")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            sigmoid_gate: lib.get_function("sigmoid_gate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            silu_elementwise_mul: lib.get_function("silu_elementwise_mul")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gated_delta_net_state_update: lib.get_function("gated_delta_net_state_update")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gated_delta_net_output: lib.get_function("gated_delta_net_output")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Additional GDN pipeline states for full forward pass.
            gated_delta_net_state_update_v2: lib.get_function("gated_delta_net_state_update_v2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_compute_gates: lib.get_function("gdn_compute_gates")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            elementwise_mul_f32: lib.get_function("elementwise_mul_f32")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_l2_norm_scale: lib.get_function("ssm_l2_norm_scale")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused element-wise kernels for GDN dispatch reduction.
            sigmoid_mul_fused: lib.get_function("sigmoid_mul_fused")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            residual_add_copy: lib.get_function("residual_add_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_qk: lib.get_function("l2_normalize_qk")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // SiLU activation (in-place) for post-conv1d GDN activation.
            silu_inplace: lib.get_function("silu_inplace")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused Conv1D + SiLU for GDN decode.
            ssm_conv1d_silu_decode: lib.get_function("ssm_conv1d_silu_decode")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Q+gate de-interleave for Qwen3.5 full-attention layers.
            deinterleave_qgate: lib.get_function("deinterleave_qgate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Per-head RMSNorm for Q and K (Qwen3.5 full-attention layers).
            rmsnorm_per_head: lib.get_function("rmsnorm_per_head")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Sigmoid-scale for shared expert gating.
            sigmoid_scale_buffer: lib.get_function("sigmoid_scale_buffer")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Batched sigmoid-scale-add for shared expert gating during prefill.
            sigmoid_scale_add_batched: lib.get_function("sigmoid_scale_add_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused GDN mega-kernels for further dispatch reduction.
            gdn_state_output_norm: lib.get_function("gdn_state_output_norm")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q8_0_deferred_residual_copy: lib.get_function("dequant_matmul_q8_0_deferred_residual_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q8_0_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q8_0_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q4_0_deferred_residual_copy: lib.get_function("dequant_matmul_q4_0_deferred_residual_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q4_0_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q4_0_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused deinterleave+norm+assemble for full-attention Q+gate layers
            deinterleave_norm_assemble: lib.get_function("deinterleave_norm_assemble")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused L2-normalize + state-update + output + RMSNorm
            gdn_state_output_norm_l2: lib.get_function("gdn_state_output_norm_l2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Simdgroup-parallel state update (4096 TGs)
            gdn_state_output_l2_sg: lib.get_function("gdn_state_output_l2_sg")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // RMSNorm + scale for decode (pairs with gdn_state_output_l2_sg)
            gdn_decode_norm_scale: lib.get_function("gdn_decode_norm_scale")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused Conv1D+SiLU + L2-normalize + state-update + output + RMSNorm
            gdn_state_output_norm_l2_conv: lib.get_function("gdn_state_output_norm_l2_conv")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Full GDN decode megakernel: Conv1D+SiLU + inline gates + L2 + state + output + RMSNorm
            gdn_decode_megakernel: lib.get_function("gdn_decode_megakernel")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused SiLU-gated Q8_0 matvec + residual + copy
            dequant_matmul_q8_0_silu_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q8_0_silu_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused SiLU-gated Q4_0 matvec + residual + copy
            dequant_matmul_q4_0_silu_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q4_0_silu_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused dual alpha+beta RMSNorm+matvec+gates for GDN decode
            dequant_matmul_q8_0_dual_gates_nr2: lib.get_function("dequant_matmul_q8_0_dual_gates_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Batched GDN prefill kernels
            gdn_prefill_state_output_norm: lib.get_function("gdn_prefill_state_output_norm")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused: lib.get_function("gdn_prefill_fused")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused_v2: lib.get_function("gdn_prefill_fused_v2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused_v3_chunked: lib.get_function("gdn_prefill_fused_v3_chunked")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_norm_gate: lib.get_function("gdn_prefill_norm_gate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_prefill: lib.get_function("ssm_conv1d_prefill")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_silu_prefill: lib.get_function("ssm_conv1d_silu_prefill")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_silu_prefill_parallel: lib.get_function("ssm_conv1d_silu_prefill_parallel")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_heads_batched: lib.get_function("l2_normalize_heads_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_qk_strided: lib.get_function("l2_normalize_qk_strided")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_compute_gates_batched: lib.get_function("gdn_compute_gates_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_batched_matvec_q8_0: lib.get_function("dequant_batched_matvec_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_batched_matvec_q8_0_dual: lib.get_function("dequant_batched_matvec_q8_0_dual")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
        })
    }
}
