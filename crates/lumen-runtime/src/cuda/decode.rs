//! CUDA kernel compilation and dispatch helpers for the decode path.
//!
//! Provides `KernelSet` (all compiled CUDA function handles) and helper
//! functions for launching individual kernels. The backend_impl module
//! orchestrates these into the full compute_layer/compute_final pipeline.

use crate::error::RuntimeError;
use super::ffi::CudaDevice;
use super::shaders;
use cudarc::driver::CudaFunction;

/// kernel-load chatter throttle.
///
/// `compile_all_kernels` writes a `[CUDA] <kernel>: OK | FAILED: <reason>`
/// line for every kernel it tries to load. On a fresh A100/H100 boot some
/// 20 dp4a / fused-down PTX modules emit `CUDA_ERROR_INVALID_PTX` and fall
/// back cleanly to v1 paths; the FAILED lines are *expected* behaviour, not
/// regressions, so the noise drowns the actual generation output in
/// production scripts.
///
/// The helper routes those lines through stderr only when the operator
/// explicitly opts in via `LUMEN_CUDA_VERBOSE=1` (or the equivalent
/// truthy values `true`/`yes`/`on`). The default is silent, but the
/// FALLBACK PATH IS UNCHANGED — failing kernels still become `None` and
/// callers still dispatch to the alternate kernel. Operators investigating
/// a load failure re-run with `LUMEN_CUDA_VERBOSE=1` to see the per-kernel
/// trace, and the long-term fix (root cause: dp4a PTX validity on
/// SM 80) is tracked as.
#[inline]
fn cuda_verbose() -> bool {
    // Read once per process; OS env var reads are not on a hot path here
    // (every call site is inside the one-shot `compile_all_kernels`).
    static CHECKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECKED.get_or_init(|| {
        std::env::var("LUMEN_CUDA_VERBOSE")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

macro_rules! cuda_log {
    ($($arg:tt)*) => {
        if crate::cuda::decode::cuda_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// unconditional log used by one-shot tracers
/// (the corresponding gate flag is checked at the call site).
#[inline]
pub(crate) fn cuda_log_force(msg: String) {
    if crate::cuda::decode::cuda_verbose() {
        eprintln!("{msg}");
    }
}

/// All compiled CUDA kernel functions needed for single-token decode.
///
/// Compiled once during `init()` via NVRTC and reused across all layers.
pub(crate) struct KernelSet {
    // Normalization
    pub(crate) rmsnorm: CudaFunction,
    #[allow(dead_code)] // Compiled but not yet dispatched; kept for per-head norm path.
    pub(crate) rmsnorm_per_head: CudaFunction,

    // F32 matrix-vector multiply (superseded by fused_norm_matvec_f32; kept for fallback).
    #[allow(dead_code)]
    pub(crate) matvec_f32: CudaFunction,
    #[allow(dead_code)]
    pub(crate) matvec_f32_residual: CudaFunction,

    // F16 matrix-vector multiply (custom NVRTC kernel, fallback for non-cuBLAS path)
    pub(crate) matvec_f16: CudaFunction,
    pub(crate) matvec_f16_residual: CudaFunction,

    // BF16 matrix-vector multiply (custom NVRTC kernel, decode path for BF16 weights).
    // Optional: compilation succeeds on all targets, but the kernel uses a 16-bit
    // left-shift bit-cast rather than the SM_80 cvt.f32.bf16 PTX intrinsic, so it
    // compiles and runs on any compute capability. `Option` is reserved for future
    // SM-gated variants.
    pub(crate) matvec_bf16: CudaFunction,
    pub(crate) matvec_bf16_residual: CudaFunction,

    // F32 <-> F16 conversion kernels (for cuBLAS HGEMM activation conversion)
    pub(crate) f32_to_f16_vec: CudaFunction,
    // Vectorized F32->F16: 4 elements/thread with float4 loads + uint2 stores.
    // Used in prefill GEMM where conversion sizes are large and aligned.
    pub(crate) f32_to_f16_vec4: Option<CudaFunction>,
    #[allow(dead_code)] // Compiled but superseded by in-kernel F16 reads; kept for fallback.
    pub(crate) f16_to_f32_vec: CudaFunction,

    // F32 -> BF16 conversion kernels (for cuBLAS GemmEx CUDA_R_16BF prefill).
    // Required to lift BF16 prefill off the per-row matvec_bf16 fallback onto
    // tensor-core HGEMM (cublasGemmEx CUDA_R_16BF). Optional because the BF16
    // weight path itself is auto-detected from the LBC header; if BF16 weights
    // are not in the model, the kernel is loaded but never dispatched.
    pub(crate) f32_to_bf16_vec: CudaFunction,
    pub(crate) f32_to_bf16_vec4: Option<CudaFunction>,

    // Q8_0 matrix-vector multiply (on-the-fly dequantization)
    pub(crate) matvec_q8_0: CudaFunction,
    pub(crate) matvec_q8_0_residual: CudaFunction,

    // Q4_0 matrix-vector multiply (on-the-fly 4-bit dequantization)
    pub(crate) matvec_q4_0: CudaFunction,
    pub(crate) matvec_q4_0_residual: CudaFunction,

    // Rotary position embeddings
    pub(crate) rope_apply: CudaFunction,

    // Elementwise activations
    pub(crate) swiglu_inplace: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    /// Fused residual add + copy: dst[i] = a[i] + b[i].
    /// Replaces residual_add (in-place) + memcpy_dtod in graph decode pipeline.
    /// Saves 1 graph node per layer (36 fewer nodes for 3B model).
    pub(crate) residual_add_copy: CudaFunction,

    // Multi-head attention
    pub(crate) attention_decode: CudaFunction,

    // Tiled streaming-softmax decode-attention kernel.
    //
    // Closes the single-block `attention_decode` kernel's `seq_len <= 40_950`
    // ceiling by streaming the softmax over fixed-size KV tiles (T_C=128) using
    // Dao 2022 online-softmax mechanics. Per-CTA shmem is constant in seq_len
    // (~1.6 KB at head_dim=256); no dynamic-shmem opt-in required.
    //
    // Optional: NVRTC compile may fail on extremely old drivers; the
    // capability gate at `decode::attention_decode_variant` ignores the
    // tiled path when this field is `None`, falling back to the single-block
    // kernel. (At `seq_len > 40_950` the single-block kernel cannot launch,
    // so a missing tiled kernel surfaces as a runtime error from
    // `attention_decode_can_launch`.)
    pub(crate) attention_decode_tiled: Option<CudaFunction>,

    // Tiled GEMM for batched prefill (superseded by cuBLAS HGEMM; kept for fallback).
    #[allow(dead_code)]
    pub(crate) gemm_f32: CudaFunction,
    #[allow(dead_code)]
    pub(crate) gemm_f32_residual: CudaFunction,

    // Fused RMSNorm + MatVec (two-pass: rms_scale scalar + inline norm)
    pub(crate) compute_rms_scale: CudaFunction,
    pub(crate) fused_norm_matvec_f32: CudaFunction,
    pub(crate) fused_norm_dual_matvec_f32: CudaFunction,

    // Batched prefill kernels
    pub(crate) embed_batch_f32: CudaFunction,
    pub(crate) embed_batch_q8_0: CudaFunction,
    pub(crate) embed_batch_f16: CudaFunction,
    pub(crate) embed_batch_q4_0: CudaFunction,
    pub(crate) rmsnorm_batched: CudaFunction,
    pub(crate) rope_apply_batched: CudaFunction,
    pub(crate) rope_apply_neox: CudaFunction,
    pub(crate) rope_apply_batched_neox: CudaFunction,
    pub(crate) bias_add_batched: CudaFunction,
    pub(crate) bias_add: CudaFunction,
    pub(crate) kv_cache_write_batch: CudaFunction,
    pub(crate) swiglu_batched: CudaFunction,
    pub(crate) residual_add_batched: CudaFunction,
    pub(crate) extract_row: CudaFunction,
    pub(crate) scatter_row: CudaFunction,

    // Flash Attention v2 for causal prefill (online softmax)
    pub(crate) flash_attention_v2: CudaFunction,
    pub(crate) flash_attention_br4: CudaFunction,

    // Q8_0 dequantization for batched prefill GEMM (replaces per-row matvec fallback)
    pub(crate) dequant_q8_0_to_f16: CudaFunction,
    pub(crate) dequant_q8_0_to_f32: CudaFunction,
    pub(crate) dequant_q4_0_to_f16: CudaFunction,
    pub(crate) dequant_q4_0_to_f32: CudaFunction,

    // Q8_0 dp4a INT8 dot product kernels (native __dp4a(), SM 6.1+).
    // On-the-fly x quantization to int8 for full integer dot product pipeline.
    // Optional: compilation requires SM 6.1+; falls back to scalar matvec_q8_0 if unavailable.
    pub(crate) matvec_q8_0_dp4a: Option<CudaFunction>,
    pub(crate) matvec_q8_0_dp4a_residual: Option<CudaFunction>,

    // MMQ-style Q8_0 batched matmul (dp4a path).
    // MMQ Q8_0 inner math (INT8xINT8->INT32->F32-scale) for batched
    // T-token prefill activations. Used to close the qkv_pre_conv 5.85e-2 drift
    // identified. Optional: requires SM 6.1+ for __dp4a().
    pub(crate) mmq_q8_0_batched: Option<CudaFunction>,

    // MMQ-style Q8_0 batched matmul WITH RESIDUAL ADD (out = residual +
    // W @ x). Used by `launch_gemm_residual` to route the ssm_out projection
    // (GDN-block exit, 4096->2048) through the MMQ math, closing the residual
    // `linear_attn_out` drift (~0.226 max-abs after's qkv-projection fix).
    // Same dp4a kernel class as `mmq_q8_0_batched`; only the final store site
    // differs (load residual, add MMQ result, store sum).
    pub(crate) mmq_q8_0_batched_residual: Option<CudaFunction>,

    // Q8_0 native warp-cooperative: scalar dequant+FMA, no x-quantization.
    // 2 warps per row, deferred reduction. Reads 1.0625 bytes/elem.
    // Superseded by dp4a/HGEMV paths; kept for fallback.
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_native: Option<CudaFunction>,
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_native_residual: Option<CudaFunction>,

    // Q8_0 aligned dp4a kernels: 36-byte blocks with 4-byte-aligned quant data.
    // Uses native int* loads (8 instructions) instead of byte-level packing (56 ops).
    // Optional: compilation requires SM 6.1+; dispatch requires Q8Aligned weight buffers.
    pub(crate) matvec_q8_0_aligned: Option<CudaFunction>,
    pub(crate) matvec_q8_0_aligned_residual: Option<CudaFunction>,

    // Repack Q8_0 from 34-byte to 36-byte aligned blocks (one-time during preload).
    pub(crate) repack_q8_0_to_aligned36: Option<CudaFunction>,

    // Q8_0 v4: dp4a + cooperative x quantization + K-tiling. NR=4 rows/block.
    // DISABLED: regressive (75% lane waste, 30% slower than v1). Compiled but not dispatched.
    // Kept for future redesign experiments. See 8.
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_v4: Option<CudaFunction>,
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_v4_residual: Option<CudaFunction>,

    // Q8_0 shared-memory matvec: x-vector cached in shmem, scalar dequant+FMA.
    // NR=2 rows/block, 256 threads. PRIMARY Q8_0 decode path — reads native Q8_0
    // (1.0625 B/elem) instead of HGEMV's pre-dequanted F16 (2 B/elem).
    // Requires dynamic shared memory = in_dim * 4 bytes.
    pub(crate) matvec_q8_0_smem: Option<CudaFunction>,
    pub(crate) matvec_q8_0_smem_residual: Option<CudaFunction>,

    // Q4_0 shared-memory matvec: x-vector cached in shmem, nibble unpack+FMA.
    // NR=2 rows/block, 256 threads. PRIMARY Q4_0 decode path — reads native Q4_0
    // (0.5625 B/elem) instead of HGEMV's pre-dequanted F16 (2 B/elem).
    pub(crate) matvec_q4_0_smem: Option<CudaFunction>,
    pub(crate) matvec_q4_0_smem_residual: Option<CudaFunction>,

    // GDN (GatedDeltaNet) kernels for Qwen3.5 hybrid layers.
    pub(crate) ssm_conv1d_decode: Option<CudaFunction>,
    pub(crate) gdn_compute_gates: Option<CudaFunction>,
    pub(crate) l2_normalize_heads: Option<CudaFunction>,
    pub(crate) gdn_state_update: Option<CudaFunction>,
    pub(crate) silu_inplace: Option<CudaFunction>,
    pub(crate) silu_elementwise_mul: Option<CudaFunction>,

    // GDN fused decode megakernels (8 launches -> 2 per GDN layer).
    // gdn_decode_megakernel: fuses conv1d+silu, gates, L2 norm, state update.
    // gdn_rmsnorm_silu_gate: fuses RMSNorm + SiLU(gate) * normed.
    pub(crate) gdn_decode_megakernel: Option<CudaFunction>,
    pub(crate) gdn_rmsnorm_silu_gate: Option<CudaFunction>,

    // graph-compatible variant of gdn_decode_megakernel.
    // Identical math but state_pos is read from a device pointer instead of a
    // scalar arg, enabling CUDA graph capture for the GDN decode path. Paired
    // with `advance_conv_position` (compiled in graph::GraphKernelSet) which
    // increments the GPU-resident counter inside the captured graph.
    pub(crate) gdn_decode_megakernel_graph: Option<CudaFunction>,

    // GDN two-launch decode kernel pair (env-gated alternative to gdn_decode_megakernel).
    // gdn_phase123_register_resident: conv1d+silu+gates+L2-norm; writes Q/K/V/alpha/beta
    // to device buffers (Phases 1-3 of the existing megakernel, no Phase 4).
    // gdn_phase4_register_resident: register-resident delta-rule. Grid
    // (num_heads, 1, ceil(head_dim/4)). Each warp owns one column; each
    // lane keeps 4 state rows in registers (s_shard[4]). 1 R + 1 W per
    // h_state element vs 2 R + 2 W in `gdn_decode_megakernel` Phase 4.
    // Env gate: `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1` (requires `LUMEN_CUDA_GDN_MEGA=1`
    // not to be falsy; the two-launch pair structurally replaces the megakernel
    // Phase 4 only, leaving Phases 1-3 logically equivalent).
    pub(crate) gdn_phase123_register_resident: Option<CudaFunction>,
    pub(crate) gdn_phase4_register_resident: Option<CudaFunction>,

    /// CUDA graph-capturable variant of `gdn_phase123_register_resident`.
    /// Identical math; reads `state_pos` from a device pointer instead of a
    /// host-scalar arg, so the dispatch's kernel-launch parameters are
    /// identical across replays. Enables graph capture under
    /// `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1` (previously silently disabled). Pairs with
    /// the existing `advance_conv_position` kernel for per-replay GPU-side
    /// state position advancement.
    pub(crate) gdn_phase123_register_resident_graph: Option<CudaFunction>,

    // GDN Phase-4 coalesced-access variant (env-gated default OFF).
    // Identical math to `gdn_phase4_register_resident`; lane-ownership change from
    // `ki = lane*4 + r` to `ki = lane + r*32` produces a single 128B
    // coalesced LDG.E.32 per warp per `r` instead of 4 strided sectors.
    // Gated by `LUMEN_CUDA_GDN_PHASE4_COAL=1` (requires `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1`).
    pub(crate) gdn_phase4_register_resident_coal: Option<CudaFunction>,

    // GDN fused prefill kernels (eliminate per-token loop over decode kernels).
    // ssm_conv1d_silu_prefill: batched conv1d+SiLU across T tokens.
    // gdn_compute_gates_batched: batched gate computation for T * num_heads.
    // l2_normalize_qk_strided: batched L2 norm for Q and K across T tokens.
    // gdn_prefill_fused_v3: warp-parallel fused state update (4x unrolled).
    // gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output.
    pub(crate) ssm_conv1d_silu_prefill: Option<CudaFunction>,
    pub(crate) gdn_compute_gates_batched: Option<CudaFunction>,
    pub(crate) l2_normalize_qk_strided: Option<CudaFunction>,
    /// two-step rsqrtf L2-norm variant (eps=1e-6, rsqrtf(fmaxf(ss,eps^2))).
    /// Env-gated by LUMEN_CUDA_L2NORM_RSQRTF=1. Default OFF preserves current behavior.
    pub(crate) l2_normalize_qk_strided_rsqrtf: Option<CudaFunction>,
    pub(crate) gdn_prefill_fused_v3: Option<CudaFunction>,
    pub(crate) gdn_prefill_norm_gate: Option<CudaFunction>,

    /// RMSNorm + SiLU-gate variant. Differences vs the
    /// original above:
    ///   1. Variance step uses `rsqrtf(mean + eps)` (one hardware op) instead
    ///      of `1.0f / sqrtf(mean + eps)` (two IEEE ops).
    ///   2. Cross-warp reduction uses a block-wide warp-shuffle SUM pattern.
    /// Env-gated by `LUMEN_CUDA_RMSNORM_RSQRTF=1`; default OFF is byte-identical
    /// to the original kernel above on Lumen's actual inputs.
    pub(crate) gdn_prefill_norm_gate_rsqrtf: Option<CudaFunction>,

    // GDN F64-internal-accumulator variants (env-gated default OFF via
    // `LUMEN_CUDA_GDN_F64_ACCUM=1`). Address cumulative F32-ULP
    // non-associativity drift bisected.
    pub(crate) l2_normalize_qk_strided_f64accum: Option<CudaFunction>,
    pub(crate) gdn_prefill_fused_v3_f64accum: Option<CudaFunction>,
    pub(crate) gdn_prefill_norm_gate_f64accum: Option<CudaFunction>,
    pub(crate) gdn_phase4_register_resident_f64accum: Option<CudaFunction>,
    pub(crate) gdn_rmsnorm_silu_gate_f64accum: Option<CudaFunction>,

    // Tensor-core Flash Attention (WMMA via inline PTX, SM 80+).
    // 16x16 query tiles via mma.sync.aligned.m16n8k16 for QK^T and PV.
    // Optional: compilation requires SM 8.0+; falls back to scalar flash_attention_br4 if unavailable.
    pub(crate) flash_attention_wmma: Option<CudaFunction>,

    // Flash Attention 2 with mask block-skip (P1-3, long-context prefill).
    // Br=4 rows per block, Bc=64 KV positions per tile. The kernel iterates
    // only the lower-triangular tiles of (Q-tile, KV-tile); upper-triangular
    // tiles past the causal boundary are never visited, saving O(seq_len^2/2)
    // FLOPs at long contexts. Companion Split-K kernels emit per-slice
    // (O, m, l) tuples for sequences large enough that a single CTA per
    // (Q, head) underutilises the SM (seq_len > ~4096).
    //
    // Env gate: `LUMEN_CUDA_FA2_BLOCKSKIP=1` selects the new FA2 path at
    // prefill dispatch in place of `flash_attention_wmma`/`flash_attention_br4`.
    // Default OFF -- enabled per-session via the CLI/server `--fa2-blockskip` flag.
    pub(crate) flash_attention_fa2_causal: Option<CudaFunction>,
    pub(crate) flash_attention_fa2_splitk_partial: Option<CudaFunction>,
    pub(crate) flash_attention_fa2_splitk_reduce: Option<CudaFunction>,

    // GPU-side argmax: finds index of max value in logits buffer.
    // Single block of 1024 threads, reads back 4 bytes instead of vocab_size*4.
    pub(crate) argmax_f32: CudaFunction,

    // Fused F16 decode kernels (dispatch count reduction for F16 HGEMV path).
    // fused_rmsnorm_f16: RMSNorm + F32->F16 output in one kernel.
    // Replaces rmsnorm + f32_to_f16_vec at 2 sites/layer (attn_norm, ffn_norm).
    pub(crate) fused_rmsnorm_f16: Option<CudaFunction>,
    // swiglu_f32_to_f16: SwiGLU activation + F32->F16 output in one kernel.
    // Replaces swiglu_inplace + f32_to_f16_vec at 1 site/layer (FFN).
    pub(crate) swiglu_f32_to_f16: Option<CudaFunction>,
    // fused_residual_rmsnorm_f16: Residual add + RMSNorm + F16 output in one kernel.
    // Fuses end of layer L (residual add) with start of layer L+1 (attn RMSNorm).
    // Saves 1 dispatch per inter-layer boundary (35 fewer for 36-layer models).
    pub(crate) fused_residual_rmsnorm_f16: Option<CudaFunction>,
    // fused_residual_rmsnorm_f32: Residual add + RMSNorm (F32 output) in one kernel.
    // For Q8_0/Q4_0 inter-layer: fuses residual_add_copy + rmsnorm.
    // Saves 1 dispatch per inter-layer boundary (47 fewer for 48-layer models).
    #[allow(dead_code)] // Compiled but not yet wired into dispatch; kept for future inter-layer fusion.
    pub(crate) fused_residual_rmsnorm_f32: Option<CudaFunction>,
    // fused_residual_rms_scale: Residual add + compute_rms_scale (scalar output).
    // For fused_glu_gemv inter-layer: fuses residual_add_copy + compute_rms_scale.
    // Saves 1 dispatch per inter-layer boundary.
    #[allow(dead_code)] // Compiled but not yet wired into dispatch; kept for future inter-layer fusion.
    pub(crate) fused_residual_rms_scale: Option<CudaFunction>,

    // Fused F16 prefill kernels (dispatch count reduction for F16 HGEMM path).
    // fused_rmsnorm_f16_batched: Batched RMSNorm + F32->F16 output.
    // Replaces rmsnorm_batched + f32_to_f16_vec at 2 sites/layer (attn_norm, ffn_norm).
    // Saves 64 dispatches per prefill on 32-layer models.
    #[allow(dead_code)] // Compiled but launch wrapper removed; kept for future prefill fusion.
    pub(crate) fused_rmsnorm_f16_batched: Option<CudaFunction>,
    // swiglu_f32_to_f16_batched: Batched SwiGLU + F32->F16 output.
    // Replaces swiglu_batched + f32_to_f16_vec at 1 site/layer (FFN).
    // Saves 32 dispatches per prefill on 32-layer models.
    #[allow(dead_code)] // Compiled but launch wrapper removed; kept for future prefill fusion.
    pub(crate) swiglu_f32_to_f16_batched: Option<CudaFunction>,

    // Q8_0 dequant-in-register HGEMV: F16 x-vector in shmem, NR=4.
    // Covers in_dim up to 24576 (shmem = in_dim * 2 bytes vs smem kernel's * 4).
    // Primary Q8_0 path for 12288 < in_dim <= 24576 (e.g. FFN down on 8B+).
    pub(crate) hgemv_q8_0: Option<CudaFunction>,
    pub(crate) hgemv_q8_0_residual: Option<CudaFunction>,

    // Q4_0 dequant-in-register HGEMV: F16 x-vector in shmem, NR=4.
    // Covers in_dim up to 24576 (shmem = in_dim * 2 bytes).
    // Primary Q4_0 path for 12288 < in_dim <= 24576.
    pub(crate) hgemv_q4_0: Option<CudaFunction>,
    pub(crate) hgemv_q4_0_residual: Option<CudaFunction>,

    // dp4a with pre-quantized Q8_1 input.
    // quantize_f32_to_q8_1: pre-quantize x once per token.
    // matvec_q8_0_q8_1: dp4a dot product with native int* input loads.
    // NR=2 rows/block, 128 threads, NO shmem for input. SM 6.1+.
    pub(crate) quantize_f32_to_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_0_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_0_q8_1_residual: Option<CudaFunction>,

    // Q4_0 dp4a kernels: native Q4_0 weights + pre-quantized Q8_1 input.
    // Unpacks nibbles to sequential int8 for dp4a, includes zero-point correction.
    // NR=4 rows/block, 256 threads. SM 6.1+.
    pub(crate) matvec_q4_0_dp4a: Option<CudaFunction>,
    pub(crate) matvec_q4_0_dp4a_residual: Option<CudaFunction>,

    // Q4Aligned + Q8_1 input dp4a kernels (NR=4).
    // Combines aligned int* nibble loads (20-byte blocks) with pre-quantized Q8_1 input.
    // 4 int* loads vs 16 byte loads per block. Dispatch priority: this > matvec_q4_0_dp4a.
    pub(crate) matvec_q4_aligned_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q4_aligned_q8_1_residual: Option<CudaFunction>,

    // Repack Q4_0 from 18-byte to 20-byte aligned blocks (one-time during preload).
    pub(crate) repack_q4_0_to_aligned20: Option<CudaFunction>,

    // Optimal Q8Aligned + Q8_1 input dp4a kernels (NR=2).
    // Combines aligned int* weight loads with pre-quantized Q8_1 input.
    // Both sides use native int* loads (zero byte-packing), dp4a, NR=2.
    // Dispatch priority: this > matvec_q8_0_aligned (on-the-fly quant).
    pub(crate) matvec_q8_aligned_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_q8_1_residual: Option<CudaFunction>,

    // split-layout integration: (opt-in via env vars; default OFF).
    //
    // Halfword (16-bit) scale-load variant of matvec_q8_aligned_q8_1. Selected
    // when `LUMEN_CUDA_Q8_SCALE_HW=1` is set at session start. Numerically
    // equivalent; reduces ALU per block iter by ~5 instr.
    pub(crate) matvec_q8_aligned_q8_1_hw: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_q8_1_hw_residual: Option<CudaFunction>,

    // Q8 split (SoA) matvec against pre-quantized Q8_1 input (dp4a, NR=2).
    // Consumes the per-row split layout produced by repack_q8_raw_to_split.
    // Selected when `LUMEN_CUDA_Q8_SPLIT=1` is set AND a sibling buffer is
    // populated on the relevant `LayerWeightsGpu.q8_split_*` field.
    pub(crate) matvec_q8_split_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_split_q8_1_residual: Option<CudaFunction>,

    // 4-threads-per-block split-Q8 matvec on the SAME byte layout.
    // K-loop trip 4+ (vs Lumen's 1) addresses the warp scheduler
    // starvation root cause observed at Q8 decode shapes.
    // Selected when `LUMEN_CUDA_Q8_SPLIT_4THREAD=1` is set AND the kernel loaded.
    pub(crate) matvec_q8_split_q8_1_4thread: Option<CudaFunction>,
    pub(crate) matvec_q8_split_q8_1_4thread_residual: Option<CudaFunction>,

    // NR=8 split-Q8 matvec: 4-threads-per-block thread mapping with NR=8
    // (rows/CTA). The single parameter-combination not tested in any prior
    // Lumen revision: 4-threads-per-block + K-trip=4+ + vdr=2 STACKED with NR=8
    // (vs the 4thread variant which kept NR=2).
    // Selected when `LUMEN_CUDA_Q8_SPLIT_NR8=1` is set AND the kernel loaded.
    // Mutually exclusive with `use_q8_split_4thread_dispatch` -- NR8 takes
    // priority if both env vars are set.
    pub(crate) matvec_q8_split_q8_1_nr8: Option<CudaFunction>,
    pub(crate) matvec_q8_split_q8_1_nr8_residual: Option<CudaFunction>,

    // AoS NR=8 matvec: 4-threads-per-block thread mapping (vdr=2,
    // blocks_per_iter=32, NR=8) applied to Lumen's AoS 36-byte block layout
    // (the same parameters on the SPLIT/SoA layout regressed -4.85% e2e).
    // Selected when `LUMEN_CUDA_Q8_AOS_NR8=1` is set AND the kernel
    // loaded. Operates on the AoS dispatch path (`launch_matvec_preq8_1`),
    // NOT the SPLIT path. Default OFF (default-off contract).
    pub(crate) matvec_q8_aligned_nr8: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_nr8_residual: Option<CudaFunction>,

    // Q4 split (SoA) matvec against pre-quantized Q8_1 input (dp4a, NR=4).
    pub(crate) matvec_q4_split_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q4_split_q8_1_residual: Option<CudaFunction>,

    // Q8 split matvec dedicated to the final output projection (configurable NR).
    // The shader exports nr8/nr16/nr32/nr64/nr128 instantiations; we load each
    // as its own optional handle so the host can pick at dispatch time via
    // `LUMEN_CUDA_OUTPUT_PROJ_NR={8,16,32,64,128}`. The legacy
    // `matvec_q8_split_output_proj` field continues to alias the nr32 variant
    // so existing dispatch sites keep working.
    pub(crate) matvec_q8_split_output_proj: Option<CudaFunction>,
    pub(crate) matvec_q8_split_output_proj_nr8: Option<CudaFunction>,
    pub(crate) matvec_q8_split_output_proj_nr16: Option<CudaFunction>,
    pub(crate) matvec_q8_split_output_proj_nr64: Option<CudaFunction>,
    pub(crate) matvec_q8_split_output_proj_nr128: Option<CudaFunction>,

    // One-time repack kernels (run during preload_weights, not on hot path).
    pub(crate) repack_q8_raw_to_split: Option<CudaFunction>,
    pub(crate) repack_q4_raw_to_split: Option<CudaFunction>,

    // TILE: tile-grouped matvec kernels + one-time
    // tile-layout repack kernels. Same byte density as the SPLIT layout but
    // colocates 8 scales and 8 quant/nibble blocks within a single 144 B
    // (Q4) / 272 B (Q8) tile to improve L1-sector locality. Opt-in via
    // `LUMEN_CUDA_Q8_TILE=1` / `LUMEN_CUDA_Q4_TILE=1`; default OFF (Hard
    // Rule 12). `Option<CudaFunction>` so a failed NVRTC compile falls back
    // to the SPLIT / Aligned / Raw dispatch in `launch_matvec_preq8_1_tile`.
    pub(crate) matvec_q8_tile_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_tile_q8_1_residual: Option<CudaFunction>,
    pub(crate) matvec_q4_tile_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q4_tile_q8_1_residual: Option<CudaFunction>,
    pub(crate) repack_q8_raw_to_tile: Option<CudaFunction>,
    pub(crate) repack_q4_raw_to_tile: Option<CudaFunction>,

    // split-layout integration: runtime feature flags (computed in CudaBackend::init
    // from LUMEN_CUDA_* env vars and kernel availability). Stored on `KernelSet`
    // so the `launch_matvec_*` free functions can consult them without taking
    // an extra parameter (matches the patch's `use_q8_cpasync` convention).
    /// `LUMEN_CUDA_Q8_SCALE_HW=1` AND matvec_q8_aligned_q8_1_hw loaded.
    pub(crate) use_q8_scale_hw: bool,
    /// `LUMEN_CUDA_Q8_SPLIT=1` AND matvec_q8_split_q8_1 loaded.
    /// When true AND the layer's `q8_split_*` sibling is `Some`, dispatch
    /// routes to `matvec_q8_split_q8_1` instead of the Q8Aligned/Q8Raw path.
    pub(crate) use_q8_split_dispatch: bool,
    /// `LUMEN_CUDA_Q8_SPLIT_4THREAD=1` AND matvec_q8_split_q8_1_4thread loaded.
    /// When true, the SPLIT dispatch uses the dp4a-mmvq kernel variant
    /// (K-trip=4 thread mapping) instead of the Lumen-native split kernel.
    /// Has no effect when `use_q8_split_dispatch` is false.
    pub(crate) use_q8_split_4thread_dispatch: bool,
    /// `LUMEN_CUDA_Q8_SPLIT_NR8=1` AND matvec_q8_split_q8_1_nr8 loaded.
    /// When true, the SPLIT dispatch uses the dp4a-mmvq kernel variant
    /// with NR=8 rows/CTA and the 4-thread mapping. Takes priority over
    /// `use_q8_split_4thread_dispatch` when both env vars are set.
    /// Default OFF (default-off contract).
    pub(crate) use_q8_split_nr8_dispatch: bool,
    /// `LUMEN_CUDA_Q8_AOS_NR8=1` AND matvec_q8_aligned_nr8 loaded.
    /// When true, the AoS dispatch (`launch_matvec_preq8_1`) uses the
    /// dp4a-mmvq kernel (NR=8 + 4-thread mapping) on the 36-byte block layout. Default OFF
    /// (default-off contract). Independent of `use_q8_split_dispatch` —
    /// the AoS NR8 operates whenever the dispatch path lands on
    /// `launch_matvec_preq8_1` (i.e., when split is OFF, or fallback).
    pub(crate) use_q8_aos_nr8_dispatch: bool,
    /// `LUMEN_CUDA_Q4_SPLIT=1` AND matvec_q4_split_q8_1 loaded.
    pub(crate) use_q4_split_dispatch: bool,
    /// `LUMEN_CUDA_Q8_TILE=1` AND matvec_q8_tile_q8_1 loaded.
    /// When true AND the layer's `q8_tile_*` sibling is `Some`, dispatch
    /// routes to `matvec_q8_tile_q8_1` instead of the SPLIT / Aligned / Raw
    /// paths. Checked BEFORE `use_q8_split_dispatch` in
    /// `launch_matvec_preq8_1_tile`.
    pub(crate) use_q8_tile_dispatch: bool,
    /// `LUMEN_CUDA_Q4_TILE=1` AND matvec_q4_tile_q8_1 loaded.
    pub(crate) use_q4_tile_dispatch: bool,

    /// `LUMEN_CUDA_FA2_BLOCKSKIP=1` AND flash_attention_fa2_causal loaded.
    /// Routes prefill attention to the FA2 block-skip kernel in place of the
    /// existing wmma/br4 dispatch when the env var is set at session start.
    /// Has / P1-3 lineage; default OFF (default-off contract).
    pub(crate) use_fa2_blockskip_dispatch: bool,

    // Fused gate+up+SwiGLU GEMV with inline RMSNorm for single-token decode.
    // Reads the input vector ONCE, computes gate and up projections simultaneously,
    // and applies SwiGLU inline. Eliminates 2-4 kernel launches per layer.
    //
    // Q8_0 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q8_0: Option<CudaFunction>,
    // Q4_0 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q4_0: Option<CudaFunction>,
    // F16 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_f16: Option<CudaFunction>,
    // Q8_0 variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q8_0_hg: Option<CudaFunction>,
    // Q4_0 variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q4_0_hg: Option<CudaFunction>,
    // Q8Aligned variant (F32 x in shmem): 36-byte blocks, hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q8_aligned: Option<CudaFunction>,
    // Q8Aligned variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q8_aligned_hg: Option<CudaFunction>,

    // Fused down projection: inline F32->Q8_1 quantize + dp4a matvec.
    // Eliminates separate quantize_f32_to_q8_1 dispatch for down projection.
    // matvec_q8_aligned_f32: reads F32 (from fused_glu output), quantizes inline, dp4a.
    pub(crate) matvec_q8_aligned_f32: Option<CudaFunction>,
    #[allow(dead_code)] // Compiled but not yet dispatched; kept for residual fusion path.
    pub(crate) matvec_q8_aligned_f32_residual: Option<CudaFunction>,
    // matvec_q8_aligned_f32_swiglu: fuses SwiGLU + quantize + dp4a (3 dispatches -> 1).
    pub(crate) matvec_q8_aligned_f32_swiglu: Option<CudaFunction>,
    #[allow(dead_code)] // Compiled but not yet dispatched; kept for residual fusion path.
    pub(crate) matvec_q8_aligned_f32_swiglu_residual: Option<CudaFunction>,

    // Fused down projection for Q4Aligned: inline F32->Q8_1 quantize + dp4a.
    // Same approach as Q8Aligned fused down but for Q4Aligned weights (20-byte blocks,
    // __byte_perm nibble unpack, zero-point correction).
    // matvec_q4_aligned_f32: reads F32, quantizes inline, dp4a against Q4Aligned.
    pub(crate) matvec_q4_aligned_f32: Option<CudaFunction>,
    #[allow(dead_code)] // Compiled but not yet dispatched; kept for residual fusion path.
    pub(crate) matvec_q4_aligned_f32_residual: Option<CudaFunction>,
    // matvec_q4_aligned_f32_swiglu: fuses SwiGLU + quantize + dp4a (3 dispatches -> 1).
    pub(crate) matvec_q4_aligned_f32_swiglu: Option<CudaFunction>,
    #[allow(dead_code)] // Compiled but not yet dispatched; kept for residual fusion path.
    pub(crate) matvec_q4_aligned_f32_swiglu_residual: Option<CudaFunction>,

    // Fused RMSNorm + Q8_1 quantization (dispatch count reduction for Q8_0 dp4a path).
    // rmsnorm_to_q8_1: RMSNorm + Q8_1 quantize in one kernel.
    // Replaces rmsnorm + quantize_f32_to_q8_1 at 2 sites/layer (attn_norm, ffn_norm).
    pub(crate) rmsnorm_to_q8_1: Option<CudaFunction>,
    // fused_residual_rmsnorm_q8_1: Residual add + RMSNorm + Q8_1 quantize.
    // For Q8_0 inter-layer boundaries: fuses residual_add_copy + rmsnorm + quantize_f32_to_q8_1.
    pub(crate) fused_residual_rmsnorm_q8_1: Option<CudaFunction>,

    // Qwen3.5 Q+gate fusion kernels (full-attention layers only).
    // deinterleave_qgate: Split [Q_h0, gate_h0, Q_h1, gate_h1, ...] -> Q + gate.
    pub(crate) deinterleave_qgate: Option<CudaFunction>,
    // sigmoid_mul: sigmoid(gate) * x -> out (for gating attention output).
    pub(crate) sigmoid_mul: Option<CudaFunction>,
    // rmsnorm_per_head_inplace: Per-head RMSNorm with shared [head_dim] weight across heads.
    pub(crate) rmsnorm_per_head_inplace: Option<CudaFunction>,

    // MoE top-K router kernel.
    // One CTA per token; softmax + top-K + renormalize fused.
    // Optional: NVRTC compile can fail on extremely old drivers; backend gates
    // MoE dispatch on this kernel being present (per the caps).
    pub(crate) moe_router_softmax: Option<CudaFunction>,

    // MoE weighted accumulation (dense top-K layout; per-expert path default).
    // Computes x = residual + Σ_k expert_weights[k] * expert_outputs[k].
    pub(crate) moe_expert_accum_option_a: Option<CudaFunction>,

    // MoE batched-expert FFN kernels.
    // Single-launch processing of all top_k active experts; eliminates
    // K-fold launch overhead vs the per-expert path. Gated behind
    // LUMEN_CUDA_MOE_BATCHED=1 env var (default OFF).
    pub(crate) moe_batched_gate_up_swiglu_q8_0: Option<CudaFunction>,
    pub(crate) moe_batched_down_accum_q8_0: Option<CudaFunction>,

    // V2 kernels: cooperative-CTA-per-row-tile pattern matching
    // the dense fused_glu_gemv_q8_0 optimization. Gated behind
    // LUMEN_CUDA_MOE_BATCHED_V2=1 (under MOE_BATCHED=1). All None on driver-load
    // failure -> v2 path silently disables, falling back to v1 batched.
    pub(crate) moe_router_logits_v2: Option<CudaFunction>,
    pub(crate) moe_router_softmax_finalize_v2: Option<CudaFunction>,
    pub(crate) moe_router_fused_v2: Option<CudaFunction>,
    pub(crate) moe_router_fused_atomic_v2: Option<CudaFunction>,
    pub(crate) moe_batched_gate_up_swiglu_q8_0_v2: Option<CudaFunction>,
    pub(crate) moe_batched_down_v2: Option<CudaFunction>,
    pub(crate) moe_batched_gate_up_swiglu_q8_0_v3: Option<CudaFunction>,
    pub(crate) moe_batched_down_v3: Option<CudaFunction>,
    // fused persistent gate+up+SwiGLU+down+accum kernel (Q8_0).
    // Eliminates the HBM round-trip on swiglu_buf by keeping the K-expert
    // SwiGLU intermediate in shmem. One launch replaces the
    // (gate_up_v3 + down_v3 + accum_option_a) trio. Gated behind
    // `LUMEN_CUDA_MOE_FUSED_PERSISTENT=1` (default OFF; opt-in).
    pub(crate) moe_batched_persistent_gate_up_swiglu_down_accum_q8_0:
        Option<CudaFunction>,
    // fused FFN-norm + router single-launch kernel.
    // Replaces (standalone `rmsnorm` writing `normed_out` + `moe_router_fused_atomic_v2`)
    // pair with one launch. Saves ~1 µs/layer launch overhead + the global write/read
    // of `normed_out` is replaced by per-CTA shmem reuse. Gated behind
    // `LUMEN_CUDA_MOE_FUSED_NORM_ROUTER=1` (default-on when V2 path is active).
    pub(crate) moe_router_rmsnorm_atomic_v3: Option<CudaFunction>,

    // fused-topK MoE router (sigmoid + top-K + renorm in one launch).
    // One kernel replaces `moe_router_softmax_finalize_v2` (the 8.7% TPOT
    // second launch) with a warp-parallel sigmoid+top-K+renorm. Gated behind
    // `LUMEN_CUDA_TOPK_MOE_FUSED=1` (default OFF). Three instantiations cover
    // n_experts ∈ {64, 128, 256}; non-power-of-two falls back to V2.
    pub(crate) topk_moe_fused_64_no_bias: Option<CudaFunction>,
    pub(crate) topk_moe_fused_128_no_bias: Option<CudaFunction>,
    pub(crate) topk_moe_fused_256_no_bias: Option<CudaFunction>,

    // Q8_1-activation x {Q8_0,Q4_0}-weight matvec with dp4a INT8
    // dot-product, plus the `quantize_q8_1` activation pre-pass. Replaces the
    // 4 Lumen matvec kernels identified as the dominant per-token
    // cost (63.3% TPOT). Gated behind `LUMEN_CUDA_MMV_Q_DP4A=1` (default OFF).
    // NVRTC failure leaves the existing `matvec_q8_0_smem` / `matvec_q4_0` /
    // batched MoE kernels as the fallback (byte-identical to).
    pub(crate) quantize_q8_1_rawsum: Option<CudaFunction>,
    pub(crate) mul_mat_vec_q_q8_0: Option<CudaFunction>,
    pub(crate) mul_mat_vec_q_q4_0: Option<CudaFunction>,
    // batched MoE FFN matvec (Q8_0/Q4_0 weights, fused gate+up+SwiGLU + down).
    // Replaces moe_batched_gate_up_swiglu_q8_0_v3 + moe_batched_down_v3 when
    // LUMEN_CUDA_MMV_Q_MOE_DP4A=1 (default OFF). NVRTC failure leaves v3 as
    // fallback (byte-identical to).
    pub(crate) quantize_q8_1_moe: Option<CudaFunction>,
    pub(crate) quantize_q8_1_moe_swiglu: Option<CudaFunction>,
    pub(crate) mmv_q_moe_gate_up_swiglu_q8_0: Option<CudaFunction>,
    pub(crate) mmv_q_moe_down_q8_0: Option<CudaFunction>,
    pub(crate) mmv_q_moe_gate_up_swiglu_q4_0: Option<CudaFunction>,
    pub(crate) mmv_q_moe_down_q4_0: Option<CudaFunction>,

    // BF16 output_proj matvec (replaces cuBLAS HGEMV at the
    // largest BF16 decode call, 16.7% TPOT measured).
    // Env-gated `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=1` (default OFF). NVRTC
    // failure leaves cuBLAS HGEMV path as fallback (byte-identical).
    pub(crate) mul_mat_vec_f_bf16: Option<CudaFunction>,

    // MoE per-expert FFN kernels.
    // One launch per (expert, token) pair; K launches per MoE FFN per token.
    pub(crate) moe_expert_gate_up_swiglu_q8_0: Option<CudaFunction>,
    pub(crate) moe_expert_down_q8_0: Option<CudaFunction>,

    // MoE BF16 per-expert + batched FFN kernels.
    // BF16 weights are plain row-major (no block structure); each element is
    // 2 bytes. Per-expert kernels are the reference implementation; batched kernels
    // process all K experts in one launch.
    pub(crate) moe_expert_gate_up_swiglu_bf16: Option<CudaFunction>,
    pub(crate) moe_expert_down_bf16: Option<CudaFunction>,
    pub(crate) moe_batched_gate_up_swiglu_bf16: Option<CudaFunction>,
    pub(crate) moe_batched_down_accum_bf16: Option<CudaFunction>,
    // cooperative-CTA-per-row-tile BF16 kernels (V3 pattern; port of
    // the Q8 `*_q8_0_v3` / `moe_batched_down_v3` kernels). High-occupancy
    // replacement for the V1 one-thread-per-row batched kernels above. F32
    // activation preserved (P3-coherent). Gated by `LUMEN_CUDA_BF16_MOE_V3`.
    pub(crate) moe_batched_gate_up_swiglu_bf16_v3: Option<CudaFunction>,
    pub(crate) moe_batched_down_bf16_v3: Option<CudaFunction>,

    // MoE per-expert FFN kernels — Q4_0 variant.
    // Identical dispatch contract to the Q8_0 pair above; selected by the
    // dispatch site based on `meta.expert_gate_quant == QuantScheme::Q4_0`.
    pub(crate) moe_expert_gate_up_swiglu_q4_0: Option<CudaFunction>,
    pub(crate) moe_expert_down_q4_0: Option<CudaFunction>,

    // MoE batched-expert FFN kernels — Q4_0 variant.
    // V1 simple (one thread per row, all K experts in one launch via gridDim.y).
    // V2 cooperative (NR=2 row-tile, BLOCK_DIM_V2=256 threads).
    pub(crate) moe_batched_gate_up_swiglu_q4_0: Option<CudaFunction>,
    pub(crate) moe_batched_down_accum_q4_0: Option<CudaFunction>,
    pub(crate) moe_batched_gate_up_swiglu_q4_0_v2: Option<CudaFunction>,
    pub(crate) moe_batched_down_v2_q4_0: Option<CudaFunction>,
    // V3 cooperative (NR=4 row-tile, BLOCK_DIM_V2=256). Ports the proven
    // Q8/BF16 V3 high-occupancy geometry to Q4_0 (opt-in LUMEN_CUDA_MOE_Q4_V3).
    pub(crate) moe_batched_gate_up_swiglu_q4_0_v3: Option<CudaFunction>,
    pub(crate) moe_batched_down_q4_0_v3: Option<CudaFunction>,
    // V3b: high-MLP element-cooperative (1 row/CTA, all threads stride the
    // contraction) — fixes the V3 ~7%-of-peak-bandwidth occupancy stall on Q4's
    // small inter_dim (opt-in LUMEN_CUDA_MOE_Q4_V3B under MOE_Q4_V3).
    pub(crate) moe_batched_gate_up_swiglu_q4_0_v3b: Option<CudaFunction>,
    pub(crate) moe_batched_down_q4_0_v3b: Option<CudaFunction>,

    // MoE shared-expert auxiliary kernels.
    // The shared expert FFN reuses matvec_q4_0 + swiglu_inplace for the heavy
    // projections; these three add the F32 dot + sigmoid-gated accumulate
    // path required by Qwen3.5-MoE's always-on shared expert.
    pub(crate) moe_shared_dot_f32: Option<CudaFunction>,
    pub(crate) moe_shared_sigmoid_gated_accum: Option<CudaFunction>,
    pub(crate) moe_shared_residual_accum: Option<CudaFunction>,
    // fused shared-expert FFN kernels (collapses 5-6 launches to 3).
    pub(crate) fused_glu_gemv_q4_0_prenormed_no_norm: Option<CudaFunction>,
    pub(crate) moe_shared_down_q4_0_sigmoid_accum: Option<CudaFunction>,
    pub(crate) moe_shared_down_q4_0_residual_accum: Option<CudaFunction>,
}

/// Compile all CUDA kernels via NVRTC and return the kernel function handles.
///
/// Each .cu source is compiled into a separate PTX module. This avoids
/// symbol conflicts between kernels that define identically-named device
/// helper functions (e.g., `warp_reduce_sum` appears in multiple .cu files).
pub(crate) fn compile_all_kernels(device: &CudaDevice) -> Result<KernelSet, RuntimeError> {
    let load_fn = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load(source)?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load CUDA kernel '{name}': {e}"))
        })
    };

    // For kernels needing SM 80+ features (dp4a, WMMA tensor cores).
    let load_fn_sm80 = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load_with_arch(source, "compute_80")?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load SM80 CUDA kernel '{name}': {e}"))
        })
    };

    // For kernels using inline-PTX dp4a (sm_61+) — avoids the
    // sm_80 PTX JIT issue observed in this build/driver env where every
    // compute_XX target produces PTX that cuModuleLoadData rejects.
    // compute_61 PTX loads successfully and __dp4a is emulated via PTX `dp4a`
    // instruction directly (mmq_q8_0.cu uses inline `dp4a.s32.s32`).
    #[allow(dead_code)]
    let load_fn_sm61 = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load_with_arch(source, "compute_61")?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load SM61 CUDA kernel '{name}': {e}"))
        })
    };

    // For dp4a kernels: SM 80+ with --use_fast_math (--fmad=true --ftz=true
    // --prec-div=false --prec-sqrt=false). Accelerates scale multiplication
    // in bandwidth-bound GEMV kernels where full FP precision is unnecessary.
    let load_fn_sm80_fast_math = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load_with_arch_fast_math(source, "compute_80")?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load SM80 fast_math CUDA kernel '{name}': {e}"))
        })
    };

    let kernels = KernelSet {
        rmsnorm: load_fn(shaders::NORM_KERNEL_SOURCE, "rmsnorm")?,
        rmsnorm_per_head: load_fn(shaders::NORM_KERNEL_SOURCE, "rmsnorm_per_head")?,
        matvec_f32: load_fn(shaders::MATVEC_F32_KERNEL_SOURCE, "matvec_f32")?,
        matvec_f32_residual: load_fn(
            shaders::MATVEC_F32_KERNEL_SOURCE,
            "matvec_f32_residual",
        )?,
        matvec_f16: load_fn(shaders::MATVEC_F16_KERNEL_SOURCE, "matvec_f16")?,
        matvec_f16_residual: load_fn(
            shaders::MATVEC_F16_KERNEL_SOURCE,
            "matvec_f16_residual",
        )?,
        matvec_bf16: {
            let f = load_fn(shaders::MATVEC_BF16_KERNEL_SOURCE, "matvec_bf16")?;
            cuda_log!("[CUDA] matvec_bf16: OK");
            f
        },
        matvec_bf16_residual: {
            let f = load_fn(shaders::MATVEC_BF16_KERNEL_SOURCE, "matvec_bf16_residual")?;
            cuda_log!("[CUDA] matvec_bf16_residual: OK");
            f
        },
        f32_to_f16_vec: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f32_to_f16_vec")?,
        f32_to_f16_vec4: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f32_to_f16_vec4").ok(),
        f16_to_f32_vec: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f16_to_f32_vec")?,
        f32_to_bf16_vec: {
            let f = load_fn(shaders::CONVERT_BF16_KERNEL_SOURCE, "f32_to_bf16_vec")?;
            cuda_log!("[CUDA] f32_to_bf16_vec: OK");
            f
        },
        f32_to_bf16_vec4: load_fn(shaders::CONVERT_BF16_KERNEL_SOURCE, "f32_to_bf16_vec4").ok(),
        matvec_q8_0: load_fn(shaders::MATVEC_Q8_0_KERNEL_SOURCE, "matvec_q8_0")?,
        matvec_q8_0_residual: load_fn(
            shaders::MATVEC_Q8_0_KERNEL_SOURCE,
            "matvec_q8_0_residual",
        )?,
        // v2, v3 removed (dead code — never dispatched, wastes compile time + GPU memory)
        matvec_q4_0: load_fn(shaders::MATVEC_Q4_0_KERNEL_SOURCE, "matvec_q4_0")?,
        matvec_q4_0_residual: load_fn(
            shaders::MATVEC_Q4_0_KERNEL_SOURCE,
            "matvec_q4_0_residual",
        )?,
        rope_apply: load_fn(shaders::ROPE_KERNEL_SOURCE, "rope_apply")?,
        swiglu_inplace: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "swiglu_inplace")?,
        residual_add: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "residual_add")?,
        residual_add_copy: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "residual_add_copy")?,
        attention_decode: load_fn(shaders::ATTENTION_KERNEL_SOURCE, "attention_decode")?,
        // Tiled streaming-softmax decode-attention kernel.
        // Optional: log a warning if NVRTC compile fails so the gate sees
        // the unavailability and operators learn the long-context path is
        // disabled on this device.
        attention_decode_tiled: match load_fn(
            shaders::ATTENTION_DECODE_TILED_KERNEL_SOURCE,
            "attention_decode_tiled",
        ) {
            Ok(f) => Some(f),
            Err(e) => {
                cuda_log!(
                    "[CUDA] attention_decode_tiled: FAILED ({e}); \
                     long-context decode (seq_len > {}) will error at dispatch",
                    ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN
                );
                None
            }
        },
        gemm_f32: load_fn(shaders::GEMM_F32_KERNEL_SOURCE, "gemm_f32")?,
        gemm_f32_residual: load_fn(shaders::GEMM_F32_KERNEL_SOURCE, "gemm_f32_residual")?,
        compute_rms_scale: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "compute_rms_scale",
        )?,
        fused_norm_matvec_f32: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "fused_norm_matvec_f32",
        )?,
        fused_norm_dual_matvec_f32: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "fused_norm_dual_matvec_f32",
        )?,
        embed_batch_f32: load_fn(shaders::PREFILL_KERNEL_SOURCE, "embed_batch_f32")?,
        embed_batch_q8_0: load_fn(shaders::PREFILL_KERNEL_SOURCE, "embed_batch_q8_0")?,
        embed_batch_f16: load_fn(shaders::PREFILL_EMBED_KERNEL_SOURCE, "embed_batch_f16")?,
        embed_batch_q4_0: load_fn(shaders::PREFILL_EMBED_KERNEL_SOURCE, "embed_batch_q4_0")?,
        rmsnorm_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "rmsnorm_batched")?,
        rope_apply_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "rope_apply_batched")?,
        rope_apply_neox: load_fn(shaders::ROPE_KERNEL_SOURCE, "rope_apply_neox")?,
        rope_apply_batched_neox: load_fn(shaders::PREFILL_KERNEL_SOURCE, "rope_apply_batched_neox")?,
        bias_add_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "bias_add_batched")?,
        bias_add: load_fn(shaders::PREFILL_KERNEL_SOURCE, "bias_add")?,
        kv_cache_write_batch: load_fn(
            shaders::PREFILL_KERNEL_SOURCE,
            "kv_cache_write_batch",
        )?,
        swiglu_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "swiglu_batched")?,
        residual_add_batched: load_fn(
            shaders::PREFILL_KERNEL_SOURCE,
            "residual_add_batched",
        )?,
        extract_row: load_fn(shaders::PREFILL_KERNEL_SOURCE, "extract_row")?,
        scatter_row: load_fn(shaders::PREFILL_KERNEL_SOURCE, "scatter_row")?,
        flash_attention_v2: load_fn(
            shaders::FLASH_ATTENTION_KERNEL_SOURCE,
            "flash_attention_causal_v2",
        )?,
        flash_attention_br4: load_fn(
            shaders::FLASH_ATTENTION_KERNEL_SOURCE,
            "flash_attention_causal_br4",
        )?,
        dequant_q8_0_to_f16: load_fn(
            shaders::DEQUANT_Q8_0_KERNEL_SOURCE,
            "dequant_q8_0_to_f16",
        )?,
        dequant_q8_0_to_f32: load_fn(
            shaders::DEQUANT_Q8_0_KERNEL_SOURCE,
            "dequant_q8_0_to_f32",
        )?,
        dequant_q4_0_to_f16: load_fn(
            shaders::DEQUANT_Q4_0_F16_KERNEL_SOURCE,
            "dequant_q4_0_to_f16",
        )?,
        dequant_q4_0_to_f32: load_fn(
            shaders::DEQUANT_Q4_0_F16_KERNEL_SOURCE,
            "dequant_q4_0_to_f32",
        )?,
        matvec_q8_0_native: match load_fn(
            shaders::MATVEC_Q8_0_NATIVE_KERNEL_SOURCE,
            "matvec_q8_0_native",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q8_0 native warp kernel: FAILED: {e}"); None }
        },
        matvec_q8_0_native_residual: match load_fn(
            shaders::MATVEC_Q8_0_NATIVE_KERNEL_SOURCE,
            "matvec_q8_0_native_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q8_0 native warp residual: FAILED: {e}"); None }
        },
        matvec_q8_0_dp4a: match load_fn_sm80(
            shaders::MATVEC_Q8_0_DP4A_KERNEL_SOURCE,
            "matvec_q8_0_dp4a",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] dp4a kernel: FAILED (fallback to v1): {e}"); None }
        },
        matvec_q8_0_dp4a_residual: match load_fn_sm80(
            shaders::MATVEC_Q8_0_DP4A_KERNEL_SOURCE,
            "matvec_q8_0_dp4a_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] dp4a_residual: FAILED: {e}"); None }
        },

        // MMQ-style batched Q8_0 matmul via dp4a.
        // Uses load_fn_sm80 (compute_80) which is required since dp4a requires
        // sm_61+. The kernel itself uses INLINE PTX `dp4a.s32.s32` instead of
        // the __dp4a intrinsic — this avoids the NVRTC `Unresolved extern
        // function '_Z6__dp4aiii'` failure that affects existing dp4a kernels
        // in this build environment and instead emits the PTX dp4a opcode
        // directly, which the driver JIT accepts on compute_80.
        mmq_q8_0_batched: match load_fn_sm80(
            shaders::MMQ_Q8_0_KERNEL_SOURCE,
            "mmq_q8_0_batched",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] MMQ Q8_0 batched: FAILED: {e}"); None }
        },
        // residual variant. Same kernel source compiles both entry points.
        mmq_q8_0_batched_residual: match load_fn_sm80(
            shaders::MMQ_Q8_0_KERNEL_SOURCE,
            "mmq_q8_0_batched_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] MMQ Q8_0 batched_residual: FAILED: {e}"); None }
        },
        matvec_q8_0_aligned: match load_fn_sm80(
            shaders::MATVEC_Q8_0_ALIGNED_KERNEL_SOURCE,
            "matvec_q8_0_aligned",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q8_0 aligned dp4a kernel: FAILED: {e}"); None }
        },
        matvec_q8_0_aligned_residual: match load_fn_sm80(
            shaders::MATVEC_Q8_0_ALIGNED_KERNEL_SOURCE,
            "matvec_q8_0_aligned_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q8_0 aligned dp4a residual: FAILED: {e}"); None }
        },
        repack_q8_0_to_aligned36: match load_fn(
            shaders::REPACK_Q8_ALIGNED_KERNEL_SOURCE,
            "repack_q8_0_to_aligned36",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] repack_q8_0_to_aligned36: FAILED: {e}"); None }
        },
        matvec_q8_0_v4: load_fn_sm80(
            shaders::MATVEC_Q8_0_V4_KERNEL_SOURCE,
            "matvec_q8_0_v4",
        ).ok(),
        matvec_q8_0_v4_residual: load_fn_sm80(
            shaders::MATVEC_Q8_0_V4_KERNEL_SOURCE,
            "matvec_q8_0_v4_residual",
        ).ok(),
        flash_attention_wmma: match load_fn_sm80(
            shaders::FLASH_ATTENTION_WMMA_KERNEL_SOURCE,
            "flash_attention_wmma",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] WMMA flash attention: FAILED: {e}"); None }
        },
        flash_attention_fa2_causal: match load_fn(
            shaders::FLASH_ATTENTION_FA2_KERNEL_SOURCE,
            "flash_attention_fa2_causal",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] FA2 block-skip: FAILED: {e}"); None }
        },
        flash_attention_fa2_splitk_partial: match load_fn(
            shaders::FLASH_ATTENTION_FA2_KERNEL_SOURCE,
            "flash_attention_fa2_splitk_partial",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] FA2 Split-K partial: FAILED: {e}"); None }
        },
        flash_attention_fa2_splitk_reduce: match load_fn(
            shaders::FLASH_ATTENTION_FA2_KERNEL_SOURCE,
            "flash_attention_fa2_splitk_reduce",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] FA2 Split-K reduce: FAILED: {e}"); None }
        },
        argmax_f32: load_fn(shaders::ARGMAX_KERNEL_SOURCE, "argmax_f32")?,
        // Q8_0 shared-memory matvec (PRIMARY Q8_0 decode path)
        matvec_q8_0_smem: match load_fn(
            shaders::MATVEC_Q8_0_SMEM_KERNEL_SOURCE,
            "matvec_q8_0_smem",
        ) {
            Ok(f) => { cuda_log!("[CUDA] Q8_0 smem matvec: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] Q8_0 smem matvec: FAILED: {e}"); None }
        },
        matvec_q8_0_smem_residual: match load_fn(
            shaders::MATVEC_Q8_0_SMEM_KERNEL_SOURCE,
            "matvec_q8_0_smem_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q8_0 smem matvec residual: FAILED: {e}"); None }
        },
        // Q4_0 shared-memory matvec (PRIMARY Q4_0 decode path)
        matvec_q4_0_smem: match load_fn(
            shaders::MATVEC_Q4_0_SMEM_KERNEL_SOURCE,
            "matvec_q4_0_smem",
        ) {
            Ok(f) => { cuda_log!("[CUDA] Q4_0 smem matvec: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] Q4_0 smem matvec: FAILED: {e}"); None }
        },
        matvec_q4_0_smem_residual: match load_fn(
            shaders::MATVEC_Q4_0_SMEM_KERNEL_SOURCE,
            "matvec_q4_0_smem_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] Q4_0 smem matvec residual: FAILED: {e}"); None }
        },
        // GDN kernels (for Qwen3.5 hybrid layers)
        ssm_conv1d_decode: load_fn(shaders::GDN_KERNEL_SOURCE, "ssm_conv1d_decode").ok(),
        gdn_compute_gates: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_compute_gates").ok(),
        l2_normalize_heads: load_fn(shaders::GDN_KERNEL_SOURCE, "l2_normalize_heads").ok(),
        gdn_state_update: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_state_update").ok(),
        silu_inplace: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "silu_inplace").ok(),
        silu_elementwise_mul: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "silu_elementwise_mul").ok(),
        // GDN fused decode megakernels (8 -> 2 launches per GDN layer)
        gdn_decode_megakernel: match load_fn(
            shaders::GDN_MEGAKERNEL_SOURCE,
            "gdn_decode_megakernel",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_decode_megakernel: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_decode_megakernel: FAILED: {e}"); None }
        },
        gdn_rmsnorm_silu_gate: match load_fn(
            shaders::GDN_MEGAKERNEL_SOURCE,
            "gdn_rmsnorm_silu_gate",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_rmsnorm_silu_gate: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_rmsnorm_silu_gate: FAILED: {e}"); None }
        },
        // graph-compatible megakernel (state_pos via device pointer).
        gdn_decode_megakernel_graph: match load_fn(
            shaders::GDN_MEGAKERNEL_SOURCE,
            "gdn_decode_megakernel_graph",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_decode_megakernel_graph: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_decode_megakernel_graph: FAILED: {e}"); None }
        },
        // GDN two-launch decode kernel pair (env-gated alternative)
        gdn_phase123_register_resident: match load_fn(
            shaders::GDN_REGISTER_RESIDENT_KERNEL_SOURCE,
            "gdn_phase123_register_resident",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_phase123_register_resident: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_phase123_register_resident: FAILED: {e}"); None }
        },
        gdn_phase4_register_resident: match load_fn(
            shaders::GDN_REGISTER_RESIDENT_KERNEL_SOURCE,
            "gdn_phase4_register_resident",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_phase4_register_resident: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_phase4_register_resident: FAILED: {e}"); None }
        },
        // CUDA graph-capturable variant (device-pointer state_pos)
        gdn_phase123_register_resident_graph: match load_fn(
            shaders::GDN_REGISTER_RESIDENT_KERNEL_SOURCE,
            "gdn_phase123_register_resident_graph",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_phase123_register_resident_graph: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_phase123_register_resident_graph: FAILED: {e}"); None }
        },
        // GDN Phase 4 coalesced variant (env-gated default OFF)
        gdn_phase4_register_resident_coal: match load_fn(
            shaders::GDN_REGISTER_RESIDENT_KERNEL_SOURCE,
            "gdn_phase4_register_resident_coal",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_phase4_register_resident_coal: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_phase4_register_resident_coal: FAILED: {e}"); None }
        },
        // GDN fused prefill kernels
        ssm_conv1d_silu_prefill: load_fn(shaders::GDN_KERNEL_SOURCE, "ssm_conv1d_silu_prefill").ok(),
        gdn_compute_gates_batched: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_compute_gates_batched").ok(),
        l2_normalize_qk_strided: load_fn(shaders::GDN_KERNEL_SOURCE, "l2_normalize_qk_strided").ok(),
        l2_normalize_qk_strided_rsqrtf: load_fn(shaders::GDN_KERNEL_SOURCE, "l2_normalize_qk_strided_rsqrtf").ok(),
        gdn_prefill_fused_v3: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_prefill_fused_v3").ok(),
        gdn_prefill_norm_gate: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_prefill_norm_gate").ok(),
        // RMSNorm + SiLU-gate variant. Loaded best-effort;
        // engaged only when LUMEN_CUDA_RMSNORM_RSQRTF=1 is set.
        gdn_prefill_norm_gate_rsqrtf: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_prefill_norm_gate_rsqrtf").ok(),
        // accumulator variants. Default-OFF; loaded best-effort.
        l2_normalize_qk_strided_f64accum: match load_fn(
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            "l2_normalize_qk_strided_f64accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] l2_normalize_qk_strided_f64accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] l2_normalize_qk_strided_f64accum: FAILED: {e}"); None }
        },
        gdn_prefill_fused_v3_f64accum: match load_fn(
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            "gdn_prefill_fused_v3_f64accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_prefill_fused_v3_f64accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_prefill_fused_v3_f64accum: FAILED: {e}"); None }
        },
        gdn_prefill_norm_gate_f64accum: match load_fn(
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            "gdn_prefill_norm_gate_f64accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_prefill_norm_gate_f64accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_prefill_norm_gate_f64accum: FAILED: {e}"); None }
        },
        gdn_phase4_register_resident_f64accum: match load_fn(
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            "gdn_phase4_register_resident_f64accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_phase4_register_resident_f64accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_phase4_register_resident_f64accum: FAILED: {e}"); None }
        },
        gdn_rmsnorm_silu_gate_f64accum: match load_fn(
            shaders::GDN_F64ACCUM_KERNEL_SOURCE,
            "gdn_rmsnorm_silu_gate_f64accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] gdn_rmsnorm_silu_gate_f64accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] gdn_rmsnorm_silu_gate_f64accum: FAILED: {e}"); None }
        },
        // Fused F16 decode kernels (dispatch count reduction)
        fused_rmsnorm_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_rmsnorm_f16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_rmsnorm_f16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_rmsnorm_f16: FAILED: {e}"); None }
        },
        swiglu_f32_to_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "swiglu_f32_to_f16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] swiglu_f32_to_f16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] swiglu_f32_to_f16: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_residual_rmsnorm_f16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_residual_rmsnorm_f16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_residual_rmsnorm_f16: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_f32: match load_fn(
            shaders::NORM_KERNEL_SOURCE,
            "fused_residual_rmsnorm_f32",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_residual_rmsnorm_f32: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_residual_rmsnorm_f32: FAILED: {e}"); None }
        },
        fused_residual_rms_scale: match load_fn(
            shaders::NORM_KERNEL_SOURCE,
            "fused_residual_rms_scale",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_residual_rms_scale: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_residual_rms_scale: FAILED: {e}"); None }
        },
        // Fused F16 prefill kernels (dispatch count reduction for batched HGEMM)
        fused_rmsnorm_f16_batched: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_rmsnorm_f16_batched",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_rmsnorm_f16_batched: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_rmsnorm_f16_batched: FAILED: {e}"); None }
        },
        swiglu_f32_to_f16_batched: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "swiglu_f32_to_f16_batched",
        ) {
            Ok(f) => { cuda_log!("[CUDA] swiglu_f32_to_f16_batched: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] swiglu_f32_to_f16_batched: FAILED: {e}"); None }
        },
        // Q8_0 dequant-in-register HGEMV (F16 x-vector shmem, NR=4)
        hgemv_q8_0: match load_fn(
            shaders::HGEMV_Q8_0_KERNEL_SOURCE,
            "hgemv_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] hgemv_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] hgemv_q8_0: FAILED: {e}"); None }
        },
        hgemv_q8_0_residual: match load_fn(
            shaders::HGEMV_Q8_0_KERNEL_SOURCE,
            "hgemv_q8_0_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] hgemv_q8_0_residual: FAILED: {e}"); None }
        },
        // Q4_0 dequant-in-register HGEMV (F16 x-vector shmem, NR=4)
        hgemv_q4_0: match load_fn(
            shaders::HGEMV_Q4_0_KERNEL_SOURCE,
            "hgemv_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] hgemv_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] hgemv_q4_0: FAILED: {e}"); None }
        },
        hgemv_q4_0_residual: match load_fn(
            shaders::HGEMV_Q4_0_KERNEL_SOURCE,
            "hgemv_q4_0_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] hgemv_q4_0_residual: FAILED: {e}"); None }
        },
        // dp4a with pre-quantized Q8_1 input
        // Compiled with --use_fast_math for accelerated scale multiplication.
        quantize_f32_to_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "quantize_f32_to_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] quantize_f32_to_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] quantize_f32_to_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_0_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "matvec_q8_0_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_0_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_0_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_0_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "matvec_q8_0_q8_1_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] matvec_q8_0_q8_1_residual: FAILED: {e}"); None }
        },
        // Q4_0 dp4a with pre-quantized Q8_1 input (NR=4, nibble unpack + dp4a)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_0_dp4a: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE,
            "matvec_q4_0_dp4a",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_0_dp4a: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_0_dp4a: FAILED: {e}"); None }
        },
        matvec_q4_0_dp4a_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE,
            "matvec_q4_0_dp4a_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_0_dp4a_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_0_dp4a_residual: FAILED: {e}"); None }
        },
        // Q4Aligned + Q8_1 input dp4a (NR=4, aligned int* nibble loads)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_aligned_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q4_aligned_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_q8_1: FAILED: {e}"); None }
        },
        matvec_q4_aligned_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q4_aligned_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_q8_1_residual: FAILED: {e}"); None }
        },
        // Repack Q4_0 to 20-byte aligned blocks
        repack_q4_0_to_aligned20: match load_fn(
            shaders::REPACK_Q4_ALIGNED_KERNEL_SOURCE,
            "repack_q4_0_to_aligned20",
        ) {
            Ok(f) => Some(f),
            Err(e) => { cuda_log!("[CUDA] repack_q4_0_to_aligned20: FAILED: {e}"); None }
        },
        // Optimal Q8Aligned + Q8_1 input dp4a (NR=2, both sides aligned)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q8_aligned_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_aligned_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_residual: FAILED: {e}"); None }
        },
        // split-layout integration: NVRTC compile of split-layout, scale-HW, and
        // output_proj split kernels. All `Option<CudaFunction>` so the failure
        // path is FALLBACK (caller dispatches to existing Q8Raw/Q8Aligned/Q4Raw/
        // Q4Aligned paths) rather than hard error -- preserves default-off contract
        // (clean revert) when the env vars are unset.
        matvec_q8_aligned_q8_1_hw: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_HW_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1_hw",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_hw: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_hw: FAILED: {e}"); None }
        },
        matvec_q8_aligned_q8_1_hw_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_HW_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1_hw_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_hw_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_q8_1_hw_residual: FAILED: {e}"); None }
        },
        matvec_q8_split_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE,
            "matvec_q8_split_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_split_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE,
            "matvec_q8_split_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_residual: FAILED: {e}"); None }
        },
        matvec_q8_split_q8_1_4thread: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_4THREAD_KERNEL_SOURCE,
            "matvec_q8_split_q8_1_4thread",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_4thread: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_4thread: FAILED: {e}"); None }
        },
        matvec_q8_split_q8_1_4thread_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_4THREAD_KERNEL_SOURCE,
            "matvec_q8_split_q8_1_4thread_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_4thread_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_4thread_residual: FAILED: {e}"); None }
        },
        // NR=8 split-Q8 matvec (4-threads-per-block, NR=8 rows/CTA).
        matvec_q8_split_q8_1_nr8: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_NR8_KERNEL_SOURCE,
            "matvec_q8_split_q8_1_nr8",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_nr8: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_nr8: FAILED: {e}"); None }
        },
        matvec_q8_split_q8_1_nr8_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_Q8_1_NR8_KERNEL_SOURCE,
            "matvec_q8_split_q8_1_nr8_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_nr8_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_q8_1_nr8_residual: FAILED: {e}"); None }
        },
        // AoS NR=8 matvec (4-threads-per-block thread mapping on Lumen's
        // 36-byte aligned block layout).
        matvec_q8_aligned_nr8: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_NR8_KERNEL_SOURCE,
            "matvec_q8_aligned_nr8",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_nr8: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_nr8: FAILED: {e}"); None }
        },
        matvec_q8_aligned_nr8_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_NR8_KERNEL_SOURCE,
            "matvec_q8_aligned_nr8_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_nr8_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_nr8_residual: FAILED: {e}"); None }
        },
        matvec_q4_split_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_SPLIT_Q8_1_KERNEL_SOURCE,
            "matvec_q4_split_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_split_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_split_q8_1: FAILED: {e}"); None }
        },
        matvec_q4_split_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_SPLIT_Q8_1_KERNEL_SOURCE,
            "matvec_q4_split_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_split_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_split_q8_1_residual: FAILED: {e}"); None }
        },
        matvec_q8_split_output_proj: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE,
            "matvec_q8_split_output_proj_nr32",
        ) {
            // The .cu file exports nr16/nr32/nr64/nr128 variants -- pick nr32 as
            // a reasonable middle ground for Qwen3.5-9B's 248320x4096 shape.
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_output_proj (nr32): OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr32: FAILED: {e}"); None }
        },
        // explicit NR=8/16/64/128 handles for `LUMEN_CUDA_OUTPUT_PROJ_NR`.
        // Failure to load is non-fatal; dispatch falls back to nr32 above.
        matvec_q8_split_output_proj_nr8: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE,
            "matvec_q8_split_output_proj_nr8",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr8: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr8: FAILED: {e}"); None }
        },
        matvec_q8_split_output_proj_nr16: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE,
            "matvec_q8_split_output_proj_nr16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr16: FAILED: {e}"); None }
        },
        matvec_q8_split_output_proj_nr64: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE,
            "matvec_q8_split_output_proj_nr64",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr64: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr64: FAILED: {e}"); None }
        },
        matvec_q8_split_output_proj_nr128: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE,
            "matvec_q8_split_output_proj_nr128",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr128: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_split_output_proj_nr128: FAILED: {e}"); None }
        },
        repack_q8_raw_to_split: match load_fn(
            shaders::REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE,
            "repack_q8_raw_to_split",
        ) {
            Ok(f) => { cuda_log!("[CUDA] repack_q8_raw_to_split: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] repack_q8_raw_to_split: FAILED: {e}"); None }
        },
        repack_q4_raw_to_split: match load_fn(
            shaders::REPACK_Q4_RAW_TO_SPLIT_KERNEL_SOURCE,
            "repack_q4_raw_to_split",
        ) {
            Ok(f) => { cuda_log!("[CUDA] repack_q4_raw_to_split: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] repack_q4_raw_to_split: FAILED: {e}"); None }
        },
        // TILE: tile-grouped Q8 / Q4 matvec kernels + one-time repack
        // kernels. Failures are non-fatal -- caller falls back to SPLIT /
        // Aligned / Raw via `launch_matvec_preq8_1_tile`.
        matvec_q8_tile_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_TILE_Q8_1_KERNEL_SOURCE,
            "matvec_q8_tile_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_tile_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_tile_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_tile_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_TILE_Q8_1_KERNEL_SOURCE,
            "matvec_q8_tile_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_tile_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_tile_q8_1_residual: FAILED: {e}"); None }
        },
        matvec_q4_tile_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_TILE_Q8_1_KERNEL_SOURCE,
            "matvec_q4_tile_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_tile_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_tile_q8_1: FAILED: {e}"); None }
        },
        matvec_q4_tile_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_TILE_Q8_1_KERNEL_SOURCE,
            "matvec_q4_tile_q8_1_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_tile_q8_1_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_tile_q8_1_residual: FAILED: {e}"); None }
        },
        repack_q8_raw_to_tile: match load_fn(
            shaders::REPACK_Q8_TILE_KERNEL_SOURCE,
            "repack_q8_raw_to_tile",
        ) {
            Ok(f) => { cuda_log!("[CUDA] repack_q8_raw_to_tile: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] repack_q8_raw_to_tile: FAILED: {e}"); None }
        },
        repack_q4_raw_to_tile: match load_fn(
            shaders::REPACK_Q4_TILE_KERNEL_SOURCE,
            "repack_q4_raw_to_tile",
        ) {
            Ok(f) => { cuda_log!("[CUDA] repack_q4_raw_to_tile: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] repack_q4_raw_to_tile: FAILED: {e}"); None }
        },
        // Fused gate+up+SwiGLU GEMV with inline RMSNorm
        fused_glu_gemv_q8_0: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q8_0: FAILED: {e}"); None }
        },
        fused_glu_gemv_q4_0: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0: FAILED: {e}"); None }
        },
        fused_glu_gemv_f16: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_f16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_f16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_f16: FAILED: {e}"); None }
        },
        fused_glu_gemv_q8_0_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_0_hg",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q8_0_hg: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q8_0_hg: FAILED: {e}"); None }
        },
        fused_glu_gemv_q4_0_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q4_0_hg",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0_hg: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0_hg: FAILED: {e}"); None }
        },
        // Fused gate+up+SwiGLU GEMV for Q8Aligned (36-byte blocks)
        fused_glu_gemv_q8_aligned: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_aligned",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q8_aligned: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q8_aligned: FAILED: {e}"); None }
        },
        fused_glu_gemv_q8_aligned_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_aligned_hg",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q8_aligned_hg: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q8_aligned_hg: FAILED: {e}"); None }
        },
        // Fused down projection: inline F32->Q8_1 quantize + dp4a matvec
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q8_aligned_f32: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_f32: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_f32: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_residual: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_swiglu: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_swiglu",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_swiglu: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_swiglu: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_swiglu_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_swiglu_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_swiglu_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q8_aligned_f32_swiglu_residual: FAILED: {e}"); None }
        },
        // Fused down projection for Q4Aligned: inline F32->Q8_1 quantize + dp4a.
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_aligned_f32: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_f32: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_f32: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_residual: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_swiglu: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_swiglu",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_swiglu: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_swiglu: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_swiglu_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_swiglu_residual",
        ) {
            Ok(f) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_swiglu_residual: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] matvec_q4_aligned_f32_swiglu_residual: FAILED: {e}"); None }
        },
        // Fused RMSNorm + Q8_1 quantization (dispatch count reduction for Q8_0 dp4a path)
        rmsnorm_to_q8_1: match load_fn(
            shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
            "rmsnorm_to_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] rmsnorm_to_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] rmsnorm_to_q8_1: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_q8_1: match load_fn(
            shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
            "fused_residual_rmsnorm_q8_1",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_residual_rmsnorm_q8_1: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_residual_rmsnorm_q8_1: FAILED: {e}"); None }
        },
        // Qwen3.5 Q+gate fusion kernels (full-attention layers)
        deinterleave_qgate: match load_fn(
            shaders::QGATE_FUSION_KERNEL_SOURCE,
            "deinterleave_qgate",
        ) {
            Ok(f) => { cuda_log!("[CUDA] deinterleave_qgate: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] deinterleave_qgate: FAILED: {e}"); None }
        },
        sigmoid_mul: match load_fn(
            shaders::QGATE_FUSION_KERNEL_SOURCE,
            "sigmoid_mul",
        ) {
            Ok(f) => { cuda_log!("[CUDA] sigmoid_mul: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] sigmoid_mul: FAILED: {e}"); None }
        },
        rmsnorm_per_head_inplace: match load_fn(
            shaders::QGATE_FUSION_KERNEL_SOURCE,
            "rmsnorm_per_head_inplace",
        ) {
            Ok(f) => { cuda_log!("[CUDA] rmsnorm_per_head_inplace: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] rmsnorm_per_head_inplace: FAILED: {e}"); None }
        },
        // MoE top-K router + accumulator (per-expert path mandatory; batched opt-in).
        moe_router_softmax: match load_fn(
            shaders::MOE_ROUTER_KERNEL_SOURCE,
            "moe_router_softmax",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_softmax: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_softmax: FAILED: {e}"); None }
        },
        moe_expert_accum_option_a: match load_fn(
            shaders::MOE_ACCUM_KERNEL_SOURCE,
            "moe_expert_accum_option_a",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_accum_option_a: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_accum_option_a: FAILED: {e}"); None }
        },
        // Sub-phase F: batched-expert FFN kernels (opt-in via env var).
        // Batched kernels are INCLUDED in this revision.
        moe_batched_gate_up_swiglu_q8_0: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0: FAILED: {e}"); None }
        },
        moe_batched_down_accum_q8_0: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_down_accum_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_accum_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_accum_q8_0: FAILED: {e}"); None }
        },
        // V2 kernels: cooperative-CTA-per-row-tile MoE path.
        // Gated behind LUMEN_CUDA_MOE_BATCHED_V2=1 (under MOE_BATCHED=1).
        moe_router_logits_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_router_logits_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_logits_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_logits_v2: FAILED: {e}"); None }
        },
        moe_router_softmax_finalize_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_router_softmax_finalize_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_softmax_finalize_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_softmax_finalize_v2: FAILED: {e}"); None }
        },
        moe_router_fused_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_router_fused_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_fused_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_fused_v2: FAILED: {e}"); None }
        },
        moe_router_fused_atomic_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_router_fused_atomic_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_fused_atomic_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_fused_atomic_v2: FAILED: {e}"); None }
        },
        moe_batched_gate_up_swiglu_q8_0_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q8_0_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0_v2: FAILED: {e}"); None }
        },
        moe_batched_down_v2: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_down_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_v2: FAILED: {e}"); None }
        },
        moe_batched_gate_up_swiglu_q8_0_v3: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q8_0_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q8_0_v3: FAILED: {e}"); None }
        },
        moe_batched_down_v3: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_down_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_v3: FAILED: {e}"); None }
        },
        // fused persistent gate+up+SwiGLU+down+accum.
        moe_batched_persistent_gate_up_swiglu_down_accum_q8_0: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_batched_persistent_gate_up_swiglu_down_accum_q8_0",
        ) {
            Ok(f) => {
                cuda_log!(
                    "[CUDA] moe_batched_persistent_gate_up_swiglu_down_accum_q8_0: OK"
                );
                Some(f)
            }
            Err(e) => {
                cuda_log!(
                    "[CUDA] moe_batched_persistent_gate_up_swiglu_down_accum_q8_0: FAILED: {e}"
                );
                None
            }
        },
        // fused FFN-norm + router single-launch kernel.
        // Replaces 2 launches (standalone `rmsnorm` + `moe_router_fused_atomic_v2`)
        // with 1. Gated behind `LUMEN_CUDA_MOE_FUSED_NORM_ROUTER=1` (default-on
        // when MOE_BATCHED + V2 are active). NVRTC-load-failure silently disables.
        moe_router_rmsnorm_atomic_v3: match load_fn(
            shaders::MOE_BATCHED_KERNEL_SOURCE,
            "moe_router_rmsnorm_atomic_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_router_rmsnorm_atomic_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_router_rmsnorm_atomic_v3: FAILED: {e}"); None }
        },
        // fused-topK MoE router (sigmoid+top-K+renorm).
        // Three instantiations cover all production num_experts. NVRTC failure
        // disables the fused router; the V2 finalize remains the fallback.
        topk_moe_fused_64_no_bias: match load_fn(
            shaders::TOPK_MOE_FUSED_KERNEL_SOURCE,
            "topk_moe_fused_64_no_bias",
        ) {
            Ok(f) => { cuda_log!("[CUDA] topk_moe_fused_64_no_bias: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] topk_moe_fused_64_no_bias: FAILED: {e}"); None }
        },
        topk_moe_fused_128_no_bias: match load_fn(
            shaders::TOPK_MOE_FUSED_KERNEL_SOURCE,
            "topk_moe_fused_128_no_bias",
        ) {
            Ok(f) => { cuda_log!("[CUDA] topk_moe_fused_128_no_bias: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] topk_moe_fused_128_no_bias: FAILED: {e}"); None }
        },
        topk_moe_fused_256_no_bias: match load_fn(
            shaders::TOPK_MOE_FUSED_KERNEL_SOURCE,
            "topk_moe_fused_256_no_bias",
        ) {
            Ok(f) => { cuda_log!("[CUDA] topk_moe_fused_256_no_bias: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] topk_moe_fused_256_no_bias: FAILED: {e}"); None }
        },
        // dispatch the Q-quant decode matvec + quantize_q8_1 kernels.
        // Uses load_fn_sm61 (PTX JIT workaround) because compute_80 PTX
        // from NVRTC fails JIT in this build env; compute_61 PTX is accepted
        // and `dp4a.s32.s32` PTX inline asm is sm_61+ compatible.
        quantize_q8_1_rawsum: match load_fn_sm61(
            shaders::MMV_Q_DP4A_KERNEL_SOURCE,
            "quantize_q8_1_rawsum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] quantize_q8_1_rawsum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] quantize_q8_1_rawsum: FAILED: {e}"); None }
        },
        mul_mat_vec_q_q8_0: match load_fn_sm61(
            shaders::MMV_Q_DP4A_KERNEL_SOURCE,
            "mul_mat_vec_q_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mul_mat_vec_q_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mul_mat_vec_q_q8_0: FAILED: {e}"); None }
        },
        mul_mat_vec_q_q4_0: match load_fn_sm61(
            shaders::MMV_Q_DP4A_KERNEL_SOURCE,
            "mul_mat_vec_q_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mul_mat_vec_q_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mul_mat_vec_q_q4_0: FAILED: {e}"); None }
        },
        // mmv_q_moe kernels (batched MoE FFN matvec).
        // Same sm_61 workaround.
        quantize_q8_1_moe: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "quantize_q8_1_moe",
        ) {
            Ok(f) => { cuda_log!("[CUDA] quantize_q8_1_moe: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] quantize_q8_1_moe: FAILED: {e}"); None }
        },
        quantize_q8_1_moe_swiglu: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "quantize_q8_1_moe_swiglu",
        ) {
            Ok(f) => { cuda_log!("[CUDA] quantize_q8_1_moe_swiglu: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] quantize_q8_1_moe_swiglu: FAILED: {e}"); None }
        },
        mmv_q_moe_gate_up_swiglu_q8_0: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "mmv_q_moe_gate_up_swiglu_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mmv_q_moe_gate_up_swiglu_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mmv_q_moe_gate_up_swiglu_q8_0: FAILED: {e}"); None }
        },
        mmv_q_moe_down_q8_0: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "mmv_q_moe_down_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mmv_q_moe_down_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mmv_q_moe_down_q8_0: FAILED: {e}"); None }
        },
        mmv_q_moe_gate_up_swiglu_q4_0: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "mmv_q_moe_gate_up_swiglu_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mmv_q_moe_gate_up_swiglu_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mmv_q_moe_gate_up_swiglu_q4_0: FAILED: {e}"); None }
        },
        mmv_q_moe_down_q4_0: match load_fn_sm61(
            shaders::MMV_Q_MOE_DP4A_KERNEL_SOURCE,
            "mmv_q_moe_down_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mmv_q_moe_down_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mmv_q_moe_down_q4_0: FAILED: {e}"); None }
        },
        // BF16 output_proj matvec dispatch.
        // Uses load_fn_sm61 (compute_80 PTX JIT workaround).
        // BF16 -> F32 conversion is via bit-shift (the upper 16 bits of an
        // IEEE F32), so no nv_bfloat16 intrinsics are needed and compute_61
        // PTX is sufficient.
        mul_mat_vec_f_bf16: match load_fn_sm61(
            shaders::MMV_F_BF16_KERNEL_SOURCE,
            "mul_mat_vec_f_bf16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] mul_mat_vec_f_bf16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] mul_mat_vec_f_bf16: FAILED: {e}"); None }
        },
        // Sub-phase B: per-expert FFN kernels (default dispatch path).
        moe_expert_gate_up_swiglu_q8_0: match load_fn(
            shaders::MOE_EXPERT_KERNEL_SOURCE,
            "moe_expert_gate_up_swiglu_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_q8_0: FAILED: {e}"); None }
        },
        moe_expert_down_q8_0: match load_fn(
            shaders::MOE_EXPERT_KERNEL_SOURCE,
            "moe_expert_down_q8_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_down_q8_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_down_q8_0: FAILED: {e}"); None }
        },
        // MoE BF16 per-expert + batched FFN kernels.
        // Per-expert kernels mirror moe_expert_q8_0 but read plain BF16 row-
        // major weights; batched kernels collapse K (expert, token) launches
        // to one. NVRTC-load-failure silently disables (caller falls back to
        // Unsupported on BF16 dispatch).
        moe_expert_gate_up_swiglu_bf16: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_expert_gate_up_swiglu_bf16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_bf16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_bf16: FAILED: {e}"); None }
        },
        moe_expert_down_bf16: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_expert_down_bf16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_down_bf16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_down_bf16: FAILED: {e}"); None }
        },
        moe_batched_gate_up_swiglu_bf16: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_bf16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_bf16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_bf16: FAILED: {e}"); None }
        },
        moe_batched_down_accum_bf16: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_batched_down_accum_bf16",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_accum_bf16: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_accum_bf16: FAILED: {e}"); None }
        },
        // cooperative-CTA BF16 V3 kernels.
        moe_batched_gate_up_swiglu_bf16_v3: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_bf16_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_bf16_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_bf16_v3: FAILED: {e}"); None }
        },
        moe_batched_down_bf16_v3: match load_fn(
            shaders::MOE_BATCHED_BF16_KERNEL_SOURCE,
            "moe_batched_down_bf16_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_bf16_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_bf16_v3: FAILED: {e}"); None }
        },
        // MoE per-expert FFN kernels — Q4_0 variant.
        moe_expert_gate_up_swiglu_q4_0: match load_fn(
            shaders::MOE_EXPERT_Q4_0_KERNEL_SOURCE,
            "moe_expert_gate_up_swiglu_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_gate_up_swiglu_q4_0: FAILED: {e}"); None }
        },
        moe_expert_down_q4_0: match load_fn(
            shaders::MOE_EXPERT_Q4_0_KERNEL_SOURCE,
            "moe_expert_down_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_expert_down_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_expert_down_q4_0: FAILED: {e}"); None }
        },
        // MoE batched-expert FFN kernels — Q4_0 variant.
        moe_batched_gate_up_swiglu_q4_0: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0: FAILED: {e}"); None }
        },
        moe_batched_down_accum_q4_0: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_down_accum_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_accum_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_accum_q4_0: FAILED: {e}"); None }
        },
        moe_batched_gate_up_swiglu_q4_0_v2: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q4_0_v2",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v2: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v2: FAILED: {e}"); None }
        },
        moe_batched_down_v2_q4_0: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_down_v2_q4_0",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_v2_q4_0: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_v2_q4_0: FAILED: {e}"); None }
        },
        // Q4_0 V3 cooperative-CTA (NR=4) kernels.
        moe_batched_gate_up_swiglu_q4_0_v3: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q4_0_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v3: FAILED: {e}"); None }
        },
        moe_batched_down_q4_0_v3: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_down_q4_0_v3",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_q4_0_v3: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_q4_0_v3: FAILED: {e}"); None }
        },
        // V3b: high-MLP element-cooperative Q4_0 kernels.
        moe_batched_gate_up_swiglu_q4_0_v3b: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_gate_up_swiglu_q4_0_v3b",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v3b: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_gate_up_swiglu_q4_0_v3b: FAILED: {e}"); None }
        },
        moe_batched_down_q4_0_v3b: match load_fn(
            shaders::MOE_BATCHED_Q4_0_KERNEL_SOURCE,
            "moe_batched_down_q4_0_v3b",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_batched_down_q4_0_v3b: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_batched_down_q4_0_v3b: FAILED: {e}"); None }
        },
        // FIX: MoE shared-expert auxiliary kernels.
        // The shared expert reuses matvec_q4_0 + swiglu_inplace for the heavy
        // projections; these three add the F32 dot + sigmoid-gated accumulate.
        moe_shared_dot_f32: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "moe_shared_dot_f32",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_shared_dot_f32: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_shared_dot_f32: FAILED: {e}"); None }
        },
        moe_shared_sigmoid_gated_accum: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "moe_shared_sigmoid_gated_accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_shared_sigmoid_gated_accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_shared_sigmoid_gated_accum: FAILED: {e}"); None }
        },
        moe_shared_residual_accum: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "moe_shared_residual_accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_shared_residual_accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_shared_residual_accum: FAILED: {e}"); None }
        },
        // fused shared-expert FFN kernels (NVRTC; silent-disable on fail).
        fused_glu_gemv_q4_0_prenormed_no_norm: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "fused_glu_gemv_q4_0_prenormed_no_norm",
        ) {
            Ok(f) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0_prenormed_no_norm: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] fused_glu_gemv_q4_0_prenormed_no_norm: FAILED: {e}"); None }
        },
        moe_shared_down_q4_0_sigmoid_accum: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "moe_shared_down_q4_0_sigmoid_accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_shared_down_q4_0_sigmoid_accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_shared_down_q4_0_sigmoid_accum: FAILED: {e}"); None }
        },
        moe_shared_down_q4_0_residual_accum: match load_fn(
            shaders::MOE_SHARED_ACCUM_KERNEL_SOURCE,
            "moe_shared_down_q4_0_residual_accum",
        ) {
            Ok(f) => { cuda_log!("[CUDA] moe_shared_down_q4_0_residual_accum: OK"); Some(f) }
            Err(e) => { cuda_log!("[CUDA] moe_shared_down_q4_0_residual_accum: FAILED: {e}"); None }
        },
        // split-layout integration: feature flags. Start OFF; flipped on in
        // `CudaBackend::init()` after reading LUMEN_CUDA_* env vars and
        // verifying the corresponding kernel(s) loaded successfully.
        use_q8_scale_hw: false,
        use_q8_split_dispatch: false,
        use_q8_split_4thread_dispatch: false,
        use_q8_split_nr8_dispatch: false,
        use_q8_aos_nr8_dispatch: false,
        use_q4_split_dispatch: false,
        // TILE feature flags. Same handling as SPLIT.
        use_q8_tile_dispatch: false,
        use_q4_tile_dispatch: false,
        // FA2 block-skip dispatch flag (default-off contract: default OFF, env-gated).
        use_fa2_blockskip_dispatch: false,
    };

    // Raise the attention_decode kernel's per-block dynamic shared-memory cap
    // from the static 48 KB ceiling to ATTN_DECODE_EXTENDED_SHMEM_BYTES so the
    // decode seq_len ceiling rises from ~12 K to ~40 K. Best-effort: GPUs that
    // cannot service the request keep the default cap and only long-context
    // decode is affected.
    opt_in_attention_decode_dyn_shmem(&[&kernels.attention_decode])?;

    Ok(kernels)
}

/// Shared memory bytes needed for the RMSNorm kernel.
///
/// The kernel uses (block_size / 32) floats for per-warp partial sums.
pub(crate) fn rmsnorm_shared_bytes(block_size: u32) -> u32 {
    (block_size / 32) * 4
}

/// Shared memory bytes needed for the attention_decode kernel.
///
/// Layout: 8 floats for warp partial reduction + seq_len floats for scores.
///
/// A100 SM 8.0 has a per-block static shared-memory ceiling of 48 KB. When
/// `seq_len > ATTN_DECODE_DEFAULT_SHMEM_MAX_SEQ_LEN` (12280), the kernel
/// needs `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)` to opt
/// into the SM-8.0 extended limit. We perform this opt-in at module load
/// via [`opt_in_attention_decode_dyn_shmem`]; the practical ceiling then
/// becomes [`ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN`] (40950).
pub(crate) fn attention_shared_bytes(seq_len: u32) -> u32 {
    (8 + seq_len) * 4
}

/// Per-block shared memory the `attention_decode` kernel uses without
/// dynamic-shared opt-in (CUDA's default cap on SM 8.0+).
pub(crate) const ATTN_DECODE_DEFAULT_SHMEM_BYTES: u32 = 49_152;

/// Per-block shared memory the kernel CAN use after
/// `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`. 163 KB is the
/// SM 8.0 maximum (164 KB device limit minus 1 KB system reserve); we
/// round down to 163,840 so the kernel never touches the reserved region.
pub(crate) const ATTN_DECODE_EXTENDED_SHMEM_BYTES: u32 = 163_840;

/// Maximum `seq_len` reachable using the default 48 KB shared-memory cap.
pub(crate) const ATTN_DECODE_DEFAULT_SHMEM_MAX_SEQ_LEN: u32 =
    (ATTN_DECODE_DEFAULT_SHMEM_BYTES / 4) - 8;

/// Maximum `seq_len` reachable after the kernel has opted in to the
/// extended dynamic-shared-memory budget (163 KB on SM 8.0+).
pub(crate) const ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN: u32 = 40_950;

/// Inclusive ceiling on `seq_len` for the single-block `attention_decode`
/// kernel.
///
/// Kept as a public crate API even after because it documents the
/// structural invariant that drives the gate threshold default
/// ([`ATTN_DECODE_TILED_DEFAULT_THRESHOLD`] sits ~10% below this value)
/// and is the documented operator-facing constant in the tiled-decode
/// acceptance criterion. Used in tests to verify the gate cutover falls
/// inside the single-block kernel's serviceable range.
#[allow(dead_code)]
pub(crate) const fn attention_decode_max_seq_len() -> u32 {
    ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN
}

/// Predicate: can the single-block `attention_decode` kernel honor
/// `attention_shared_bytes(seq_len)` after the extended-shmem opt-in?
///
/// Kept as a public crate API even after. The new gate predicate
/// `attention_decode_variant` supersedes this for kernel-selection decisions,
/// but `attention_decode_can_launch` remains the documented invariant for the
/// single-block kernel's serviceable range. Test code and diagnostic logging
/// reference it; production dispatch is via the gate.
#[allow(dead_code)]
pub(crate) fn attention_decode_can_launch(seq_len: u32) -> bool {
    seq_len <= attention_decode_max_seq_len()
}

/// Raise the `attention_decode` kernel's dynamic shared-memory limit from
/// the default 48 KB to [`ATTN_DECODE_EXTENDED_SHMEM_BYTES`]. Failure is
/// non-fatal: the kernel keeps its default cap and only the long-context
/// decode path is affected.
pub(crate) fn opt_in_attention_decode_dyn_shmem(
    fns: &[&CudaFunction],
) -> Result<(), RuntimeError> {
    use cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    for f in fns {
        if let Err(e) = f.set_attribute(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            ATTN_DECODE_EXTENDED_SHMEM_BYTES as i32,
        ) {
            let msg = format!("{e}");
            if msg.contains("INVALID_VALUE") {
                eprintln!(
                    "[lumen-cuda] attention_decode dyn-shmem opt-in declined ({msg}); \
                     long-context decode capped at seq_len <= {}",
                    ATTN_DECODE_DEFAULT_SHMEM_MAX_SEQ_LEN
                );
                continue;
            }
            return Err(RuntimeError::Compute(format!(
                "set_attribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) on attention_decode: {e}",
            )));
        }
    }
    Ok(())
}

// ===========================================================================
// Tiled streaming-softmax decode-attention gate
// ===========================================================================
//
// Closes: long-context decode at
// `seq_len > 40_950` (the single-block kernel's SM-8.0 extended-shmem
// ceiling, see `ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN` above).
//
// Two paths cohabit:
//   - Single-block `attention_decode` (existing, fast): seq_len <= 40_950.
//   - Tiled `attention_decode_tiled` (NEW): seq_len up to KV cache cap.
//
// Routing is decided by `attention_decode_variant(seq_len, force_tiled)`,
// a pure function of two inputs. The host wraps every `attention_decode`
// launch with this gate. There are FIVE such launch points in the
// production CUDA path:
//   1. backend_impl.rs decode primary entry
//   2. backend_impl.rs per-layer decode body
//   3. backend_impl.rs CUDA graph capture body (eager-fallback when Tiled;
//      graph fast path preserved at seq_len <= threshold)
//   4. prefill.rs   per-token prefill fallback launcher
//   5. prefill_attention.rs sequential per-token prefill launcher
//
// User decisions:
//   - Tiled-decode acceptance criterion amended (physics-grounded gate; enforces).
//   - Split-K decode: conditional, deferred to a future revision.
//   - Graph capture at seq_len > threshold: eager-fallback (no tiled-graph kernel).
//   - `LUMEN_CUDA_DECODE_TILED` default: opt-in only (FORCE-mode for
//     A/B benching). Auto-routing happens via the THRESHOLD-based predicate
//     below: when `seq_len > LUMEN_CUDA_DECODE_TILED_THRESHOLD` (default
//     `ATTN_DECODE_TILED_DEFAULT_THRESHOLD = 36_864`, i.e. ~10% below the
//     40_950 ceiling), the tiled path is selected automatically. will
//     refine the production default empirically.

/// KV tile width for `attention_decode_tiled` (must match `T_C` in the kernel).
pub(crate) const ATTN_DECODE_TILED_T_C: u32 = 128;

/// Block dim for `attention_decode_tiled` (must match `BLOCK_DIM` in the kernel).
pub(crate) const ATTN_DECODE_TILED_BLOCK_DIM: u32 = 128;

/// Shared memory bytes for `attention_decode_tiled`.
///
/// Layout (CONSTANT in seq_len): `partial[8] + q_row[head_dim] + s_tile[T_C]`
/// floats. At `head_dim = 256`: `(8 + 256 + 128) * 4 = 1568 bytes` — well
/// under the 48 KB default shmem cap; no `cuFuncSetAttribute` opt-in needed.
pub(crate) const fn attention_decode_tiled_shared_bytes(head_dim: u32) -> u32 {
    (8 + head_dim + ATTN_DECODE_TILED_T_C) * 4
}

/// Whether the tiled decode kernel can serve the given `head_dim`.
///
/// **addition**: the tiled kernel's per-thread loop unrolls
/// `head_dim` over `BLOCK_DIM = 128` threads, so it requires
/// `head_dim % ATTN_DECODE_TILED_BLOCK_DIM == 0` (and `head_dim >=
/// ATTN_DECODE_TILED_BLOCK_DIM` so each thread has at least one
/// element). Production Qwen3.5-9B uses `head_dim = 256` which
/// satisfies both. Tiny test models (e.g. `head_dim = 4`, used by
/// `crates/lumen-runtime/tests/cuda_e2e_generate_test.rs`) do NOT
/// satisfy this and must fall back to SingleBlock.
///
/// Used by [`launch_attention_decode_gated`] as a hardware-compat
/// guard AFTER the pure-predicate gate selects Tiled — if the gate
/// says Tiled but `head_dim` is incompatible, the dispatch silently
/// falls back to SingleBlock instead of failing the launch. This
/// preserves the "tiled-always" default while keeping small
/// test models working.
pub(crate) const fn attention_decode_tiled_supports_head_dim(head_dim: u32) -> bool {
    head_dim >= ATTN_DECODE_TILED_BLOCK_DIM
        && head_dim % ATTN_DECODE_TILED_BLOCK_DIM == 0
}

/// Default threshold at which the gate auto-routes to the tiled kernel.
///
/// **(2026-05-25)**: lowered from the prior value of 36_864
/// to 0 ("tiled-always") based on empirical data showing the tiled
/// kernel is universally 1.16x-1.50x FASTER than single-block at every
/// measured seq_len from 4_096 through 36_864, monotone in seq_len, across
/// all 3 quants (BF16/Q4_0/Q8_0) on Qwen3.5-9B / A100-80GB PCIe. The
/// earlier prediction ("tiled slowdown at 4K significant
/// 10-25%") was empirically refuted; the conservative-headroom argument
/// that motivated the 36_864 value no longer applies, and keeping it
/// would mask a 16-50% free decode speedup for all callers at short
/// context.
///
/// Operators that want the prior behaviour can opt out by setting
/// `LUMEN_CUDA_DECODE_TILED_THRESHOLD=4294967295` (u32::MAX), which makes
/// the gate's `seq_len > threshold` check effectively always-false; the
/// single-block kernel then serves every seq_len up to the
/// ceiling at [`ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN`] = 40_950 (above
/// which the gate auto-promotes Tiled regardless, because single-block
/// structurally cannot launch).
///
/// Default 0 means the tiled streaming-softmax decode path is always engaged,
/// so it serves sequence lengths past the single-block shared-memory ceiling.
pub(crate) const ATTN_DECODE_TILED_DEFAULT_THRESHOLD: u32 = 0;

/// Which decode-attention kernel variant to dispatch for this `seq_len`.
///
/// Decision rule (binding ADR + user sign-off):
///   1. If `force_tiled = true` → Tiled (FORCE-mode opt-in, for A/B benching).
///   2. Else if `seq_len > threshold` → Tiled (auto-route above threshold).
///   3. Else → SingleBlock (fast path for short decode).
///
/// `threshold` is the resolved per-process value of
/// `LUMEN_CUDA_DECODE_TILED_THRESHOLD` (default
/// [`ATTN_DECODE_TILED_DEFAULT_THRESHOLD`]).
///
/// PURE function of its three inputs — no global state, no env reads.
/// Env-var resolution lives in [`decode_tiled_threshold`] and
/// [`decode_tiled_force_enabled`], which cache their results in
/// process-static [`OnceLock`]s read once per backend lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionDecodeVariant {
    SingleBlock,
    Tiled,
}

/// Pure gate predicate. Unit-testable; mirrors the established pattern of
/// `attention_decode_can_launch` (existing function with same purpose).
pub(crate) fn attention_decode_variant(
    seq_len: u32,
    force_tiled: bool,
    threshold: u32,
) -> AttentionDecodeVariant {
    if force_tiled {
        AttentionDecodeVariant::Tiled
    } else if seq_len > threshold {
        AttentionDecodeVariant::Tiled
    } else {
        AttentionDecodeVariant::SingleBlock
    }
}

/// Resolved-once value of `LUMEN_CUDA_DECODE_TILED_THRESHOLD`.
///
/// Mirrors the `bf16_gemmex_env_force_off()` pattern: env is read
/// exactly once into the OnceLock the first time this function is called
/// (typically at first decode dispatch); all subsequent calls return the
/// cached value without a syscall.
///
/// Accepts any `u32`-parseable value. Empty / unparseable / unset → default
/// [`ATTN_DECODE_TILED_DEFAULT_THRESHOLD`]. Out-of-range values (e.g.
/// `0` or values above the model's max_seq_len) are passed through
/// faithfully — the predicate handles edge cases.
pub(crate) fn decode_tiled_threshold() -> u32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LUMEN_CUDA_DECODE_TILED_THRESHOLD")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(ATTN_DECODE_TILED_DEFAULT_THRESHOLD)
    })
}

/// Resolved-once value of `LUMEN_CUDA_DECODE_TILED` (FORCE-mode opt-in).
///
/// Returns `true` when the env var is set to a truthy value
/// (`1` / `true` / `yes` / `on`, case-insensitive). When `true`, the gate
/// dispatches the tiled kernel at ALL seq_lens — used for paired
/// A/B benching against the single-block kernel in the regime where both
/// can launch.
///
/// Default `false` (opt-in only). Auto-routing happens via the
/// THRESHOLD-based predicate, not this flag.
pub(crate) fn decode_tiled_force_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LUMEN_CUDA_DECODE_TILED")
            .ok()
            .map(|v| {
                let s = v.trim().to_ascii_lowercase();
                matches!(s.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod attention_decode_variant_tests {
    //! Gate-predicate boundary tests covering the boundary check
    //! refinement.
    //!
    //! These tests are hardware-independent (the predicate is pure Rust
    //! and reads no global state). They run on macOS via `cargo test --lib`.
    //! On Linux + CUDA they additionally execute as part of the lib test
    //! suite during `cargo test --release -p lumen-runtime --features cuda`.

    use super::{
        attention_decode_variant, AttentionDecodeVariant,
        ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN, ATTN_DECODE_TILED_DEFAULT_THRESHOLD,
    };

    /// Default production threshold (0 = "tiled-always" —
    /// empirical data showed tiled is 1.16x-1.50x faster than single-block
    /// at every measured seq_len). Tests use this constant directly so any
    /// future change to the default lights up in CI.
    const DEFAULT_THRESHOLD: u32 = ATTN_DECODE_TILED_DEFAULT_THRESHOLD;

    /// Legacy threshold (36_864) retained as a fixed constant for
    /// shape-validation tests below; the gate predicate is exercised at
    /// non-default thresholds so the bench harness + opt-out path remain
    /// covered after the default flip to 0.
    const LEGACY_THRESHOLD: u32 = 36_864;

    /// Sanity: the default threshold sits strictly below the single-block
    /// shmem ceiling so the gate cutover always has headroom. With the
    /// default of 0 ("tiled-always"), this still holds trivially
    /// (0 < 40_950) — the assertion is preserved so any future bump to a
    /// near-ceiling value is caught immediately.
    #[test]
    fn default_threshold_below_single_block_ceiling() {
        assert!(
            DEFAULT_THRESHOLD < ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN,
            "DEFAULT_THRESHOLD ({DEFAULT_THRESHOLD}) must be < ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN ({ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN}) so the auto-routing cutover happens before the single-block kernel's shmem cap"
        );
    }

    /// with the default-threshold flip to 0, all seq_len > 0
    /// auto-route to Tiled. Verify the new contract at a representative
    /// short-context seq_len.
    #[test]
    fn gate_default_threshold_zero_routes_tiled_at_short_ctx() {
        let v = attention_decode_variant(32_768, false, DEFAULT_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::Tiled);
    }

    /// `seq_len = 32_768` (well below LEGACY_THRESHOLD = 36_864): SingleBlock
    /// when not forced. Exercises the gate-predicate's `seq_len <= threshold`
    /// branch with the non-default LEGACY value to keep the opt-out path
    /// covered after the default flip.
    #[test]
    fn gate_below_threshold_picks_single_block() {
        let v = attention_decode_variant(32_768, false, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::SingleBlock);
    }

    /// `seq_len = threshold` (the gate uses `>` so equal goes to SingleBlock):
    /// at the boundary, single-block is selected when not forced. Uses
    /// LEGACY_THRESHOLD so the strict-equality boundary is exercised at a
    /// non-zero value (the boundary case at threshold=0 is degenerate; see
    /// `gate_zero_seq_len_handled` for the seq_len=0 corner).
    #[test]
    fn gate_at_exact_threshold_picks_single_block() {
        let v = attention_decode_variant(LEGACY_THRESHOLD, false, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::SingleBlock);
    }

    /// `seq_len = threshold + 1`: first seq_len that auto-routes to tiled.
    #[test]
    fn gate_one_past_threshold_picks_tiled() {
        let v = attention_decode_variant(LEGACY_THRESHOLD + 1, false, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::Tiled);
    }

    /// `seq_len = 131_072` (far past threshold): Tiled even without force.
    #[test]
    fn gate_far_past_threshold_picks_tiled() {
        let v = attention_decode_variant(131_072, false, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::Tiled);
    }

    /// `force_tiled = true` overrides the threshold (FORCE-mode opt-in for
    /// A/B benching at short seq_len). Uses LEGACY_THRESHOLD to make the
    /// override semantically meaningful (at DEFAULT=0 the gate already picks
    /// Tiled, so force_tiled is a no-op; LEGACY=36_864 makes force_tiled the
    /// load-bearing flag here).
    #[test]
    fn gate_force_tiled_overrides_below_threshold() {
        let v = attention_decode_variant(32_768, true, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::Tiled);
    }

    /// `force_tiled = true` at long context: idempotent (tiled is already the
    /// auto-routed choice; force is a no-op here but must not regress).
    #[test]
    fn gate_force_tiled_idempotent_above_threshold() {
        let v = attention_decode_variant(131_072, true, LEGACY_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::Tiled);
    }

    /// `seq_len = 0` (degenerate): defaults to SingleBlock under both
    /// LEGACY_THRESHOLD (`0 <= 36_864`) and the new DEFAULT=0
    /// (`0 > 0` is false, so SingleBlock). Decode never actually calls with
    /// seq_len = 0 in production (KV cache is appended BEFORE attention
    /// dispatch, so seq_len >= 1 always), but the predicate must handle the
    /// degenerate case without panicking. Pass 1
    /// boundary table.)
    #[test]
    fn gate_zero_seq_len_handled() {
        let v = attention_decode_variant(0, false, DEFAULT_THRESHOLD);
        assert_eq!(v, AttentionDecodeVariant::SingleBlock);
        let v_legacy = attention_decode_variant(0, false, LEGACY_THRESHOLD);
        assert_eq!(v_legacy, AttentionDecodeVariant::SingleBlock);
    }

    /// u32::MAX is the documented opt-out value — operators who
    /// want the prior single-block-default behaviour set the env var to
    /// u32::MAX, and the gate then NEVER auto-routes Tiled below the
    /// 40_950 structural ceiling (above which the gate still must promote
    /// Tiled because single-block cannot launch — verified separately by
    /// the bench harness running `LUMEN_CUDA_DECODE_TILED_THRESHOLD=
    /// 4294967295` at seq_len > 40_950 and observing the auto-promotion
    /// in production callers; this unit test verifies the predicate alone
    /// honours u32::MAX strictly per `seq_len > threshold`).
    #[test]
    fn gate_opt_out_threshold_u32_max_picks_single_block_at_short_ctx() {
        let v = attention_decode_variant(32_768, false, u32::MAX);
        assert_eq!(v, AttentionDecodeVariant::SingleBlock);
        let v_at_legacy = attention_decode_variant(36_864, false, u32::MAX);
        assert_eq!(v_at_legacy, AttentionDecodeVariant::SingleBlock);
    }

    /// Custom-threshold test: when the operator overrides via env var, the
    /// predicate honours the new boundary precisely. (Models the real
    /// resolution path from `decode_tiled_threshold()`.)
    #[test]
    fn gate_custom_threshold_honoured() {
        // Operator sets threshold = 10_000 (e.g. for stress-testing the tiled
        // path at medium seq_len without forcing every dispatch).
        let custom = 10_000u32;
        assert_eq!(
            attention_decode_variant(9_999, false, custom),
            AttentionDecodeVariant::SingleBlock
        );
        assert_eq!(
            attention_decode_variant(10_000, false, custom),
            AttentionDecodeVariant::SingleBlock
        );
        assert_eq!(
            attention_decode_variant(10_001, false, custom),
            AttentionDecodeVariant::Tiled
        );
    }
}

#[cfg(test)]
mod attention_decode_tiled_head_dim_tests {
    //! hardware-compat-guard tests for the tiled kernel's
    //! `head_dim % BLOCK_DIM == 0` requirement. See
    //! [`attention_decode_tiled_supports_head_dim`].

    use super::{attention_decode_tiled_supports_head_dim, ATTN_DECODE_TILED_BLOCK_DIM};

    /// Production Qwen3.5-9B uses head_dim = 256 = 2 * BLOCK_DIM. PASS.
    #[test]
    fn supports_qwen3_5_head_dim_256() {
        assert!(attention_decode_tiled_supports_head_dim(256));
    }

    /// head_dim = BLOCK_DIM (= 128) is the minimum supported. PASS.
    #[test]
    fn supports_head_dim_equal_to_block_dim() {
        assert!(attention_decode_tiled_supports_head_dim(ATTN_DECODE_TILED_BLOCK_DIM));
    }

    /// head_dim = 384 = 3 * BLOCK_DIM. PASS.
    #[test]
    fn supports_multiple_of_block_dim() {
        assert!(attention_decode_tiled_supports_head_dim(384));
        assert!(attention_decode_tiled_supports_head_dim(512));
        assert!(attention_decode_tiled_supports_head_dim(1024));
    }

    /// head_dim = 4 (tiny test model, `TestModelConfig::default()` in
    /// `crates/lumen-format/src/test_model.rs`): NOT supported.
    /// dispatch must fall back to SingleBlock.
    #[test]
    fn rejects_tiny_test_head_dim() {
        assert!(!attention_decode_tiled_supports_head_dim(4));
        assert!(!attention_decode_tiled_supports_head_dim(8));
        assert!(!attention_decode_tiled_supports_head_dim(16));
        assert!(!attention_decode_tiled_supports_head_dim(32));
        assert!(!attention_decode_tiled_supports_head_dim(64));
    }

    /// head_dim = 127 (one below BLOCK_DIM): NOT supported.
    #[test]
    fn rejects_just_below_block_dim() {
        assert!(!attention_decode_tiled_supports_head_dim(ATTN_DECODE_TILED_BLOCK_DIM - 1));
    }

    /// head_dim = 129 (just above BLOCK_DIM but not a multiple): NOT supported.
    #[test]
    fn rejects_just_above_block_dim_non_multiple() {
        assert!(!attention_decode_tiled_supports_head_dim(ATTN_DECODE_TILED_BLOCK_DIM + 1));
    }

    /// head_dim = 0 (degenerate): NOT supported (`0 % anything == 0` but
    /// `0 >= 128` is false).
    #[test]
    fn rejects_zero() {
        assert!(!attention_decode_tiled_supports_head_dim(0));
    }
}

#[cfg(test)]
mod attention_decode_tiled_const_tests {
    //! Compile-time invariant tests for the tiled kernel constants.
    //!
    //! These guard against accidental shmem-budget drift if T_C or BLOCK_DIM
    //! are ever retuned — recompile is required, and we want CI to surface
    //! the change loudly so the operator can re-validate.

    use super::{
        attention_decode_tiled_shared_bytes, ATTN_DECODE_TILED_BLOCK_DIM,
        ATTN_DECODE_TILED_T_C,
    };

    /// At Qwen3.5-9B's `head_dim = 256`, tiled shmem must be small (< 4 KB)
    /// so we never need a `cuFuncSetAttribute` opt-in for the tiled kernel
    #[test]
    fn tiled_shmem_fits_default_cap_at_head_dim_256() {
        let bytes = attention_decode_tiled_shared_bytes(256);
        // (8 + 256 + 128) * 4 = 1568.
        assert_eq!(bytes, 1568);
        // Default per-block dyn-shmem cap on SM 6.0+ is 48 KB (49152).
        assert!(
            bytes < 49152,
            "tiled shmem {bytes} bytes must fit default 48 KB cap"
        );
    }

    /// Even at head_dim = 1024 (well above any realistic model), the tiled
    /// shmem still fits.
    #[test]
    fn tiled_shmem_fits_default_cap_at_head_dim_1024() {
        let bytes = attention_decode_tiled_shared_bytes(1024);
        // (8 + 1024 + 128) * 4 = 4640.
        assert_eq!(bytes, 4640);
        assert!(bytes < 49152);
    }

    /// T_C must equal BLOCK_DIM for the kernel's "one lane per tile position"
    /// stride assumption to hold (saves an `if (j < T_C)` outer guard in
    /// Phase A). If they are ever decoupled, the kernel's stride loop in
    /// Phases A/C still works (it walks T_C in `block_size`-strided
    /// chunks) but the dispatch will be sub-optimal.
    #[test]
    fn tc_equals_block_dim_for_one_lane_per_position() {
        assert_eq!(ATTN_DECODE_TILED_T_C, ATTN_DECODE_TILED_BLOCK_DIM);
    }
}

/// Block size for RMSNorm: use min(dim, 1024), rounded down to a multiple of 32.
///
/// The kernel runs as a single block, so all threads must fit in one block.
pub(crate) fn rmsnorm_block_size(dim: usize) -> u32 {
    let bs = dim.min(1024);
    // Round down to nearest multiple of 32 (warp size).
    let bs = (bs / 32) * 32;
    bs.max(32) as u32
}

/// Block size for F32/Q4_0 matvec: 256 threads (one block per output row).
pub(crate) fn matvec_block_size() -> u32 {
    256
}

/// Q8_0 matvec constants matching matvec_q8_0.cu kernel defines.
///
/// The Q8_0 kernel uses NR=2 rows per thread block with 128 threads (4 warps).
/// Grid must be ceil(out_dim / NR), not out_dim.
pub(crate) const Q8_0_NR: u32 = 2;
pub(crate) const Q8_0_BLOCK_DIM: u32 = 128;

/// Grid size for Q8_0 matvec: ceil(out_dim / NR) blocks.
pub(crate) fn matvec_q8_0_grid(out_dim: u32) -> u32 {
    (out_dim + Q8_0_NR - 1) / Q8_0_NR
}

/// Smem Q8_0/Q4_0 matvec constants (must match matvec_q8_0_smem.cu / matvec_q4_0_smem.cu).
///
/// NR=2 rows per block, 256 threads (8 warps). Shared memory = in_dim * 4 bytes.
pub(crate) const SMEM_NR: u32 = 2;
pub(crate) const SMEM_BLOCK_DIM: u32 = 256;

/// Grid size for smem matvec: ceil(out_dim / NR) blocks.
pub(crate) fn matvec_smem_grid(out_dim: u32) -> u32 {
    (out_dim + SMEM_NR - 1) / SMEM_NR
}

/// Shared memory bytes for smem matvec: in_dim floats for x-vector cache.
pub(crate) fn matvec_smem_shared_bytes(in_dim: u32) -> u32 {
    in_dim * 4
}

/// Block size for attention: min(seq_len, 256), rounded up to multiple of 32.
///
/// Must be at least 32 (one warp) for the reduction helpers to work correctly.
pub(crate) fn attention_block_size(seq_len: usize) -> u32 {
    let bs = seq_len.min(256);
    // Round up to nearest multiple of 32.
    let bs = ((bs + 31) / 32) * 32;
    bs.max(32) as u32
}

/// Block size for fused norm+matvec F32: 256 threads (matches FUSED_BLOCK_SIZE define).
pub(crate) fn fused_norm_matvec_block_size() -> u32 {
    256
}

// ------------------------------------------------------------------
// Flash Attention v2 constants and helpers (must match flash_attention.cu)
// ------------------------------------------------------------------

/// KV tile size for flash attention (must match FA_BC in flash_attention.cu).
pub(crate) const FA_BC: u32 = 32;

/// Query tile size for Br=4 variant (must match FA_BR in flash_attention.cu).
pub(crate) const FA_BR: u32 = 4;

/// Block size for flash_attention_causal_v2 (Br=1).
///
/// One thread block per (head, query_token). 128 threads provides 4 warps
/// for cooperative dot products and V accumulation.
pub(crate) fn flash_attention_v2_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_causal_v2.
///
/// Layout: partial[8] + q_row[head_dim] + s_tile[FA_BC].
pub(crate) fn flash_attention_v2_shared_bytes(head_dim: u32) -> u32 {
    (8 + head_dim + FA_BC) * 4
}

/// Block size for flash_attention_causal_br4 (Br=4).
///
/// 128 threads = 4 warps, one warp per query row.
pub(crate) fn flash_attention_br4_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_causal_br4.
///
/// Layout: q_rows[4][head_dim] + s_tiles[4][FA_BC].
pub(crate) fn flash_attention_br4_shared_bytes(head_dim: u32) -> u32 {
    FA_BR * (head_dim + FA_BC) * 4
}

// ------------------------------------------------------------------
// WMMA Flash Attention constants and helpers (must match flash_attention_wmma.cu)
// ------------------------------------------------------------------

/// Query tile rows for WMMA variant (must match FA_TC_BR in flash_attention_wmma.cu).
pub(crate) const FA_TC_BR: u32 = 16;

/// KV tile columns for WMMA variant (must match FA_TC_BC in flash_attention_wmma.cu).
pub(crate) const FA_TC_BC: u32 = 16;

/// Block size for flash_attention_wmma (128 threads = 4 warps).
pub(crate) fn flash_attention_wmma_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_wmma.
///
/// Layout:
/// half Q_sh[BR * head_dim] = BR * hd * 2 bytes
/// half KV_sh[BC * head_dim] = BC * hd * 2 bytes (reused for K then V)
/// float S_sh[BR * BC] = BR * BC * 4 bytes
/// half P_sh[BR * BC] = BR * BC * 2 bytes
/// float O_acc[BR * head_dim] = BR * hd * 4 bytes
/// float rowmax[BR] = BR * 4 bytes
/// float rowsum[BR] = BR * 4 bytes
pub(crate) fn flash_attention_wmma_shared_bytes(head_dim: u32) -> u32 {
    let br = FA_TC_BR;
    let bc = FA_TC_BC;
    let hd = head_dim;

    let q_sh = br * hd * 2;         // half Q_sh[BR][hd]
    let kv_sh = bc * hd * 2;        // half KV_sh[BC][hd]
    let s_sh = br * bc * 4;         // float S_sh[BR][BC]
    let p_sh = br * bc * 2;         // half P_sh[BR][BC]
    let o_acc = br * hd * 4;        // float O_acc[BR][hd]
    let rowmax = br * 4;            // float rowmax[BR]
    let rowsum = br * 4;            // float rowsum[BR]

    q_sh + kv_sh + s_sh + p_sh + o_acc + rowmax + rowsum
}

// ------------------------------------------------------------------
// FA2 block-skip constants and helpers (must match flash_attention_fa2.cu)
// ------------------------------------------------------------------

/// KV tile size for FA2 block-skip kernel (must match `FA2_BC`).
pub(crate) const FA2_BC: u32 = 64;

/// Query rows per block in `flash_attention_fa2_causal` (must match `FA2_BR`).
pub(crate) const FA2_BR: u32 = 4;

/// Block size for `flash_attention_fa2_causal` (128 threads = 4 warps).
pub(crate) fn flash_attention_fa2_block_size() -> u32 {
    128
}

/// Shared memory bytes for `flash_attention_fa2_causal`.
///
/// Layout: q_rows[FA2_BR][head_dim] + s_tiles[FA2_BR][FA2_BC] floats.
pub(crate) fn flash_attention_fa2_shared_bytes(head_dim: u32) -> u32 {
    FA2_BR * (head_dim + FA2_BC) * 4
}

/// Block size for `flash_attention_fa2_splitk_partial` (one warp).
pub(crate) fn flash_attention_fa2_splitk_partial_block_size() -> u32 {
    32
}

/// Shared memory bytes for `flash_attention_fa2_splitk_partial`.
///
/// Layout: q_shmem[head_dim] + s_tile[FA2_BC] floats.
pub(crate) fn flash_attention_fa2_splitk_partial_shared_bytes(head_dim: u32) -> u32 {
    (head_dim + FA2_BC) * 4
}

/// Block size for `flash_attention_fa2_splitk_reduce`.
pub(crate) fn flash_attention_fa2_splitk_reduce_block_size() -> u32 {
    256
}

/// Shared memory bytes for `flash_attention_fa2_splitk_reduce`.
///
/// Layout: m_arr[N] + l_arr[N] + rescale[N] + scratch[1] floats.
pub(crate) fn flash_attention_fa2_splitk_reduce_shared_bytes(num_splits: u32) -> u32 {
    (3 * num_splits + 1) * 4
}

/// Recommended Split-K KV slice size. The single-kernel block-skip path is
/// preferred for short contexts; Split-K kicks in when seq_len is large
/// enough that a single (q_idx, head) block underutilises the SM.
pub(crate) const FA2_SPLITK_SLICE: u32 = 1024;

/// Minimum seq_len that triggers Split-K dispatch.
pub(crate) const FA2_SPLITK_MIN_SEQ: u32 = 4096;

// ------------------------------------------------------------------
// HGEMV (dequant-in-register) constants and helpers
// (must match hgemv_q8_0.cu / hgemv_q4_0.cu)
// ------------------------------------------------------------------

/// Rows per thread block for HGEMV kernels.
pub(crate) const HGEMV_NR: u32 = 4;

/// Threads per block for HGEMV kernels.
pub(crate) const HGEMV_BLOCK_DIM: u32 = 256;

/// Grid size for HGEMV: ceil(out_dim / NR) blocks.
pub(crate) fn hgemv_grid(out_dim: u32) -> u32 {
    (out_dim + HGEMV_NR - 1) / HGEMV_NR
}

/// Shared memory bytes for HGEMV: in_dim * 2 (F16 x-vector cache).
pub(crate) fn hgemv_shared_bytes(in_dim: u32) -> u32 {
    in_dim * 2
}

/// Maximum shared memory (bytes) per block on A100. Default limit before
/// requesting dynamic shared memory extension via cudaFuncSetAttribute.
/// 48 KB = 49152 bytes. HGEMV uses in_dim * 2 bytes for F16 x-vector,
/// so covers up to in_dim = 24576.
pub(crate) const HGEMV_SHMEM_LIMIT: u32 = 49152;

// ------------------------------------------------------------------
// dp4a Q8_1 constants and helpers
// (must match matvec_dp4a_q8_1.cu defines)
// ------------------------------------------------------------------

/// Q8_1 block size in bytes: [f16 scale, f16 sum, 32 × int8] = 36 bytes.
pub(crate) const Q8_1_BLOCK_BYTES: u32 = 36;

/// Elements per Q8_1 block (same as Q8_0).
pub(crate) const Q8_1_BLOCK_SIZE: u32 = 32;

/// Threads per block for the Q8_1 quantization kernel (1 warp per block).
pub(crate) const Q8_1_QUANT_BLOCK_DIM: u32 = 32;

/// Threads per block for the dp4a Q8_1 matvec kernel (4 warps).
pub(crate) const DP4A_Q8_1_BLOCK_DIM: u32 = 128;

/// Size of the Q8_1 buffer in bytes for a given input dimension.
///
/// Each group of 32 F32 elements becomes one 36-byte Q8_1 block.
pub(crate) fn q8_1_buffer_bytes(in_dim: u32) -> u32 {
    let num_blocks = in_dim / Q8_1_BLOCK_SIZE;
    num_blocks * Q8_1_BLOCK_BYTES
}

/// Grid size for the Q8_1 quantization kernel: one warp per block.
pub(crate) fn q8_1_quant_grid(in_dim: u32) -> u32 {
    in_dim / Q8_1_BLOCK_SIZE
}

/// Grid size for the dp4a Q8_1 matvec: ceil(out_dim / MV_NR) blocks.
/// MV_NR=2: each block processes 2 output rows.
/// Used by Q8_0, Q8Aligned, and Q8_1 kernels (NR=2).
pub(crate) fn dp4a_q8_1_grid(out_dim: u32) -> u32 {
    (out_dim + 1) / 2
}

/// Threads per block for Q4_0 dp4a kernels (NR=4, 8 warps).
/// Q4 kernels use NR=4 with 256 threads for better x-vector amortization:
/// 4x weight bandwidth amortization per x-vector load vs 2x for Q8.
/// Q4 blocks are smaller (18-20 bytes vs 34-36 for Q8), so NR=4 doesn't
/// cause register spill despite processing 4 rows per block.
pub(crate) const DP4A_Q4_BLOCK_DIM: u32 = 256;

/// Grid size for Q4_0 dp4a matvec: ceil(out_dim / 4) blocks.
/// NR=4: each block processes 4 output rows.
pub(crate) fn dp4a_q4_grid(out_dim: u32) -> u32 {
    (out_dim + 3) / 4
}

// ------------------------------------------------------------------
// Fused GLU GEMV constants and helpers
// (must match fused_glu_gemv.cu defines)
// ------------------------------------------------------------------

/// Rows per thread block for fused GLU GEMV (must match NR in fused_glu_gemv.cu).
pub(crate) const FUSED_GLU_NR: u32 = 2;

/// Threads per block for fused GLU GEMV (must match BLOCK_DIM in fused_glu_gemv.cu).
pub(crate) const FUSED_GLU_BLOCK_DIM: u32 = 256;

/// Grid size for fused GLU GEMV: ceil(inter_dim / NR) blocks.
pub(crate) fn fused_glu_grid(inter_dim: u32) -> u32 {
    (inter_dim + FUSED_GLU_NR - 1) / FUSED_GLU_NR
}

/// Shared memory bytes for fused GLU GEMV (F32 x-vector variant):
/// hidden_dim * 4 bytes for the normed x-vector.
pub(crate) fn fused_glu_shared_bytes_f32(hidden_dim: u32) -> u32 {
    hidden_dim * 4
}

/// Shared memory bytes for fused GLU GEMV (F16 x-vector variant):
/// hidden_dim * 2 bytes for the normed F16 x-vector.
pub(crate) fn fused_glu_shared_bytes_f16(hidden_dim: u32) -> u32 {
    hidden_dim * 2
}

/// Maximum shared memory per block (48KB on A100).
/// F32 variant covers hidden_dim <= 12288 (12288 * 4 = 49152).
/// F16 variant covers hidden_dim <= 24576 (24576 * 2 = 49152).
pub(crate) const FUSED_GLU_SHMEM_LIMIT: u32 = 49152;
