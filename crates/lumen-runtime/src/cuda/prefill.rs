//! Batched prefill pipeline for the CUDA backend.
//!
//! Replaces token-at-a-time prefill (128 * 32 * 15 = 61,440 kernel launches
//! for pp128 on 8B) with batched operations (~700 launches). F32 projections
//! use cuBLAS SGEMM; Q8_0 weights use cublasGemmEx HGEMM via pre-dequanted
//! F16 caches (tensor core path); native F16 weights use cublasGemmEx HGEMM
//! directly (no dequant needed -- already in the right format). Q4_0 falls
//! back to per-row matvec. Attention remains sequential per token since each
//! token's causal mask differs.
//!
//! cuBLAS SGEMM for row-major matrices:
//!
//! Our weight W is [out_dim, in_dim] row-major. Activation A is [batch, in_dim]
//! row-major. Output C is [batch, out_dim] row-major. We want C = A * W^T.
//!
//! cuBLAS is column-major. Row-major M[R,C] looks like col-major M_cm[C,R].
//! So W_cm is [in_dim, out_dim], A_cm is [in_dim, batch], C_cm is [out_dim, batch].
//!
//! C_cm = W_cm^T * A_cm => transa=T, transb=N
//! m=out_dim, n=batch, k=in_dim
//! A(cublas)=W, lda=in_dim, B(cublas)=A, ldb=in_dim, C(cublas)=C, ldc=out_dim

use cudarc::cublas::{Gemm, GemmConfig, sys as cublas_sys};
use cudarc::driver::{CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};

use crate::error::RuntimeError;

use super::decode::{
    AttentionDecodeVariant, KernelSet, attention_block_size, attention_shared_bytes,
    attention_decode_tiled_shared_bytes, attention_decode_tiled_supports_head_dim,
    attention_decode_variant, decode_tiled_force_enabled,
    decode_tiled_threshold, matvec_block_size, rmsnorm_block_size, rmsnorm_shared_bytes,
    ATTN_DECODE_TILED_BLOCK_DIM,
};
use super::ffi::CudaDevice;
use super::gpu_buffers::GpuWeightBuf;
use super::kv_cache::KvCacheGpu;
use super::types::LaunchConfig;

/// Pre-allocated GPU scratch buffers for the batched prefill path.
///
/// Sized for a specific `batch` (prompt length). Allocated once per prefill
/// call and reused across all layers. All buffers are [batch, dim] matrices.
pub(crate) struct PrefillScratch {
    /// Activation matrix: [batch, hidden_dim].
    pub x: CudaSlice<f32>,
    /// Normalized activation: [batch, hidden_dim].
    pub normed: CudaSlice<f32>,
    /// Q projection: [batch, q_dim].
    pub q: CudaSlice<f32>,
    /// K projection: [batch, kv_dim].
    pub k: CudaSlice<f32>,
    /// V projection: [batch, kv_dim].
    pub v: CudaSlice<f32>,
    /// Attention output: [batch, q_dim] (filled token-at-a-time).
    pub attn_out: CudaSlice<f32>,
    /// Output projection + residual: [batch, hidden_dim].
    pub attn_proj: CudaSlice<f32>,
    /// Gate FFN: [batch, inter_dim].
    pub gate: CudaSlice<f32>,
    /// Up FFN: [batch, inter_dim].
    pub up: CudaSlice<f32>,
    /// Down projection: [batch, hidden_dim].
    pub down: CudaSlice<f32>,
    /// Token IDs on GPU: [batch].
    pub token_ids_gpu: CudaSlice<u32>,
    /// Single-token Q for attention: [q_dim].
    /// Used by `launch_attention_for_token` (sequential prefill fallback).
    #[allow(dead_code)]
    pub q_single: CudaSlice<f32>,
    /// Single-token attention output: [q_dim].
    /// Used by `launch_attention_for_token` (sequential prefill fallback).
    #[allow(dead_code)]
    pub attn_out_single: CudaSlice<f32>,
    /// F32 scratch for dequantized Q8_0 weights: [max_weight_elements].
    ///
    /// Sized to hold the largest projection weight matrix in F32 format.
    /// Reused across all projections within a layer. Replaces the per-row
    /// matvec fallback with: dequant Q8_0 -> F32 scratch -> cuBLAS SGEMM.
    pub dequant_f32: CudaSlice<f32>,
    /// F16 activation scratch for HGEMM input conversion: [batch * max_in_dim * 2] bytes.
    ///
    /// Used to convert F32 activations to F16 before `cublasGemmEx` HGEMM.
    /// Reused across all projections. Only allocated when F16 weight caches exist.
    pub activation_f16: CudaSlice<u8>,
    /// F16 scratch for dequantized Q8_0/Q4_0 weights: [max_weight_elements * 2] bytes.
    ///
    /// Sized to hold the largest projection weight matrix in F16 format.
    /// Reused across all projections within a layer. Enables the HGEMM path
    /// (312 TFLOPS tensor cores) for quantized weights that lack a persistent
    /// F16 cache (e.g., GDN layers). Replaces the slow dequant->F32->SGEMM
    /// path (19.5 TFLOPS) with dequant->F16->HGEMM.
    pub dequant_f16: CudaSlice<u8>,
}

/// Pre-allocated GPU scratch buffers for GDN batched prefill.
///
/// Holds batched projections and per-token outputs specific to GDN layers.
/// Allocated once per prefill call when the model has GDN layers.
pub(crate) struct GdnPrefillScratch {
    /// Batched QKV output: [batch, qkv_dim].
    pub qkv: CudaSlice<f32>,
    /// Batched alpha raw projection: [batch, num_heads].
    pub alpha_raw: CudaSlice<f32>,
    /// Batched beta raw projection: [batch, num_heads].
    pub beta_raw: CudaSlice<f32>,
    /// Batched gate projection: [batch, value_dim].
    pub gate: CudaSlice<f32>,
    /// Batched GDN output (per-token gated output): [batch, value_dim].
    pub gdn_out: CudaSlice<f32>,
    /// Batched conv1d + SiLU output: [batch, qkv_dim].
    pub conv_out: CudaSlice<f32>,
    /// Raw output from gdn_prefill_fused_v3: [batch, num_heads * val_dim].
    pub raw_out: CudaSlice<f32>,
    /// Computed alpha gates: [batch, num_heads]. Output of gdn_compute_gates_batched.
    pub alpha_out: CudaSlice<f32>,
    /// Computed beta gates: [batch, num_heads]. Output of gdn_compute_gates_batched.
    pub beta_out: CudaSlice<f32>,
}

/// Allocate GDN prefill scratch buffers.
pub(crate) fn alloc_gdn_prefill_scratch(
    device: &super::ffi::CudaDevice,
    batch: usize,
    qkv_dim: usize,
    num_heads: usize,
    value_dim: usize,
) -> Result<GdnPrefillScratch, RuntimeError> {
    Ok(GdnPrefillScratch {
        qkv: device.alloc_zeros(batch * qkv_dim)?,
        alpha_raw: device.alloc_zeros(batch * num_heads)?,
        beta_raw: device.alloc_zeros(batch * num_heads)?,
        gate: device.alloc_zeros(batch * value_dim)?,
        gdn_out: device.alloc_zeros(batch * value_dim)?,
        conv_out: device.alloc_zeros(batch * qkv_dim)?,
        raw_out: device.alloc_zeros(batch * value_dim)?,
        alpha_out: device.alloc_zeros(batch * num_heads)?,
        beta_out: device.alloc_zeros(batch * num_heads)?,
    })
}

/// Allocate prefill scratch buffers for the given batch size and model dimensions.
///
/// The `dequant_f32` buffer is sized to hold the largest projection weight matrix
/// in F32 format across ALL kernels that share this scratch: standard attention
/// (wq/wk/wv/wo), FFN (w_gate/w_up/w_down), AND -- when present -- GDN fused
/// QKV (out_dim=gdn_qkv_dim, in_dim=hidden_dim) and GDN SSM output projection
/// (out_dim=hidden_dim, in_dim=gdn_value_dim). The `gdn_qkv_dim` and
/// `gdn_value_dim` parameters are optional; pass `Some(...)` for models with
/// GDN layers (Qwen3.5 family) and `None` for pure-attention models.
///
/// previously the scratch was sized only from standard attention/FFN
/// dimensions, which understimated the requirement for GDN models whose
/// `qkv_dim = 2*qk_dim + value_dim` can exceed `inter_dim`. Qwen3.5-9B never
/// tripped this because its dims happen to fit (`inter_dim=12288 >
/// qkv_dim=8192`); Qwen3.5-35B-A3B fails at the first GDN prefill because
/// `qkv_dim * hidden_dim = 8192 * 2048 = 16.8M` exceeds `inter_dim *
/// hidden_dim = 6144 * 2048 = 12.6M`.
pub(crate) fn alloc_prefill_scratch(
    device: &CudaDevice,
    batch: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
    gdn_qkv_dim: Option<usize>,
    gdn_value_dim: Option<usize>,
) -> Result<PrefillScratch, RuntimeError> {
    // Maximum weight matrix size across all projections that share this scratch.
    //
    // Standard attention/FFN terms always apply. GDN terms are included when
    // the model declares GDN layers (`gdn_qkv_dim.is_some()`); otherwise the
    // GDN candidates evaluate to 0 and are dominated by the attention/FFN terms.
    let gdn_qkv = gdn_qkv_dim.unwrap_or(0);
    let gdn_value = gdn_value_dim.unwrap_or(0);
    let max_weight_elems = [
        q_dim * hidden_dim,         // wq
        kv_dim * hidden_dim,        // wk, wv
        hidden_dim * q_dim,         // wo
        inter_dim * hidden_dim,     // w_gate, w_up
        hidden_dim * inter_dim,     // w_down
        gdn_qkv * hidden_dim,       // GDN fused QKV projection (out=qkv_dim, in=hidden_dim)
        hidden_dim * gdn_value,     // GDN SSM output projection (out=hidden_dim, in=value_dim)
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    // F16 activation buffer must cover the largest activation row across all
    // kernels: standard attention/FFN inputs, plus GDN value-side inputs when
    // present (e.g. ssm_out reads activations of width `value_dim`).
    let max_in_dim = [
        hidden_dim, inter_dim, q_dim, gdn_value,
    ]
    .into_iter()
    .max()
    .unwrap_or(hidden_dim);

    Ok(PrefillScratch {
        x: device.alloc_zeros(batch * hidden_dim)?,
        normed: device.alloc_zeros(batch * hidden_dim)?,
        q: device.alloc_zeros(batch * q_dim)?,
        k: device.alloc_zeros(batch * kv_dim)?,
        v: device.alloc_zeros(batch * kv_dim)?,
        attn_out: device.alloc_zeros(batch * q_dim)?,
        attn_proj: device.alloc_zeros(batch * hidden_dim)?,
        gate: device.alloc_zeros(batch * inter_dim)?,
        up: device.alloc_zeros(batch * inter_dim)?,
        down: device.alloc_zeros(batch * hidden_dim)?,
        token_ids_gpu: device.alloc_zeros(batch)?,
        q_single: device.alloc_zeros(q_dim)?,
        attn_out_single: device.alloc_zeros(q_dim)?,
        dequant_f32: device.alloc_zeros(max_weight_elems)?,
        // F16 activation: batch * max_in_dim * 2 bytes. max_in_dim covers all
        // kernels that convert activations to F16 prior to cublasGemmEx HGEMM.
        activation_f16: device.alloc_zeros(batch * max_in_dim * 2)?,
        // F16 dequant scratch: max_weight_elems * 2 bytes (F16). Enables HGEMM for Q8_0/Q4_0
        // weights without persistent F16 caches (GDN layers).
        dequant_f16: device.alloc_zeros(max_weight_elems * 2)?,
    })
}

/// Batch embed tokens into [batch, hidden_dim] on GPU.
///
/// # Safety
///
/// All token IDs must be < vocab_size. `token_ids_gpu` must have `batch` elements.
/// `output` must have `batch * hidden_dim` elements. Embedding buffers must be valid.
pub(crate) unsafe fn launch_embed_batch(
    device: &CudaDevice,
    kernels: &KernelSet,
    embedding_f32: &CudaSlice<f32>,
    embedding_q8: Option<&CudaSlice<u8>>,
    embedding_f16: Option<&CudaSlice<u8>>,
    embedding_q4: Option<&CudaSlice<u8>>,
    token_ids_gpu: &CudaSlice<u32>,
    output: &mut CudaSlice<f32>,
    batch: usize,
    hidden_dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * hidden_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_u32 = batch as u32;
    let hd = hidden_dim as u32;

    // Dispatch priority: F16 > Q4_0 > Q8_0 > F32 (same order as embed_token_gpu)
    if let Some(emb_f16) = embedding_f16 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_f16)
            .arg(emb_f16)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_f16 launch: {e}")))?;
    } else if let Some(emb_q4) = embedding_q4 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_q4_0)
            .arg(emb_q4)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_q4_0 launch: {e}")))?;
    } else if let Some(emb_q8) = embedding_q8 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_q8_0)
            .arg(emb_q8)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_q8_0 launch: {e}")))?;
    } else {
        device
            .stream
            .launch_builder(&kernels.embed_batch_f32)
            .arg(embedding_f32)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_f32 launch: {e}")))?;
    }
    Ok(())
}

/// Batched RMSNorm: normalize each row of [batch, dim] independently.
///
/// # Safety
///
/// `x` and `out` must have `batch * dim` elements. `weight` must have `dim` elements.
pub(crate) unsafe fn launch_rmsnorm_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    eps: f32,
    batch: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let block_size = rmsnorm_block_size(dim);
    let shared_bytes = rmsnorm_shared_bytes(block_size);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.rmsnorm_batched)
        .arg(x)
        .arg(weight)
        .arg(out)
        .arg(&eps)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("rmsnorm_batched launch: {e}")))?;
    Ok(())
}

/// Batched GEMM projection: out = input * weight^T.
///
/// For F32 weights, uses cuBLAS SGEMM directly. For Q8_0 weights with a
/// pre-dequanted F16 cache, uses cublasGemmEx HGEMM (tensor cores). For Q8_0
/// without F16 cache, dequantizes to the F32 scratch buffer then calls cuBLAS
/// SGEMM. For native F16 weights (`F16Raw`), uses cublasGemmEx HGEMM directly
/// (no dequant needed -- weights are already F16). For Q4_0, falls back to
/// per-row matvec.
///
/// cuBLAS column-major mapping for row-major data:
/// Row-major W[out_dim, in_dim] = col-major W_cm[in_dim, out_dim]
/// Row-major A[batch, in_dim] = col-major A_cm[in_dim, batch]
/// Row-major C[batch, out_dim] = col-major C_cm[out_dim, batch]
/// C_cm = W_cm^T * A_cm
/// cublasSgemm(T, N, out_dim, batch, in_dim, 1.0, W, in_dim, A, in_dim, 0.0, C, out_dim)
///
/// # Safety
///
/// `input` must be [batch, in_dim]. `weight` must be [out_dim, in_dim].
/// `output` must be [batch, out_dim]. `dequant_scratch` must have at least
/// `out_dim * in_dim` elements (only used for Q8_0 weights).
pub(crate) unsafe fn launch_gemm_projection(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    dequant_scratch: &mut CudaSlice<f32>,
    activation_f16: &mut CudaSlice<u8>,
    dequant_f16: &mut CudaSlice<u8>,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // Validate buffer sizes before launching GPU work.
    let input_needed = batch * in_dim;
    let output_needed = batch * out_dim;
    if input.len() < input_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm {label}: input buffer too small: have {} elements, \
             need {} (batch={batch}, in_dim={in_dim})",
            input.len(),
            input_needed,
        )));
    }
    if output.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm {label}: output buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            output.len(),
            output_needed,
        )));
    }

    // diagnostic: when LUMEN_CUDA_PREFILL_F32=1, bypass the HGEMM-F16
    // tensor-core fast path AND the F16-cache Q8/Q4 paths; force the SGEMM-F32
    // fallback paths in the `match weight` block. Used to test whether HGEMM-F16
    // rounding accounts for the L0 drift on the Q8 projection.
    let force_f32 = std::env::var("LUMEN_CUDA_PREFILL_F32").is_ok();

    // when LUMEN_CUDA_Q8_PROJ_MMQ=1, also bypass the F16-cache
    // fast path so that Q8Raw weights are routed to mmq_q8_0_batched (MMQ-style
    // INT8xINT8->INT32->F32-scale dp4a math). The F16 cache would otherwise
    // short-circuit before the match arm.
    let q8_proj_mmq = std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").is_ok();
    let weight_is_q8raw = matches!(weight, GpuWeightBuf::Q8Raw(_));
    let prefer_mmq_over_f16cache = q8_proj_mmq && weight_is_q8raw && in_dim % 32 == 0
        && kernels.mmq_q8_0_batched.is_some();

    // `ssm_alpha` and `ssm_beta` weights are stored as F32 in the
    // GGUF source and an F32 SGEMM dispatch is the canonical reference path.
    // Lumen's MoE converter force-requantizes them to Q8_0 at LBC creation
    // time (see `crates/lumen-convert/src/arch/gdn_gates.rs`), so the runtime
    // path runs them through HGEMM-F16 / MMQ-Q8 — both of which introduce
    // ~0.4-3.4% per-element rounding noise from the requant step. When
    // LUMEN_CUDA_GDN_AB_F32=1 (or the combined gate LUMEN_CUDA_NORM_RSQRTF_BUNDLE=1) is
    // set AND the projection label is `gdn_alpha` or `gdn_beta`, dequantize
    // the Q8 weight to F32 scratch and route through cuBLAS SGEMM, matching
    // the F32 SGEMM data path bit-for-bit (modulo SGEMM accumulator order).
    let gdn_ab_f32 = (std::env::var("LUMEN_CUDA_GDN_AB_F32").is_ok()
        || std::env::var("LUMEN_CUDA_NORM_RSQRTF_BUNDLE").is_ok())
        && (label == "gdn_alpha" || label == "gdn_beta");
    let force_alpha_beta_f32 = gdn_ab_f32 && weight_is_q8raw;

    // Fast path: HGEMM with pre-dequanted F16 weights (tensor core, 312 TFLOPS on A100).
    // Converts F32 activations to F16 on the fly, uses cublasGemmEx with F16 inputs
    // and F32 compute/accumulate for numerical stability.
    let f16_cache_active = !force_f32 && !prefer_mmq_over_f16cache
        && !force_alpha_beta_f32
        && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        static HGEMM_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[CUDA] Prefill HGEMM: ACTIVE (tensor core path)");
        }

        // Step 1: Convert F32 activation to F16 via vectorized kernel (4 elems/thread).
        launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

        // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
        launch_cublas_hgemm(
            device, w_f16, activation_f16, output,
            out_dim, batch, in_dim, 0.0, label,
        )?;
        return Ok(());
    }

    match weight {
        GpuWeightBuf::F32(w_f32) => {
            let weight_needed = out_dim * in_dim;
            if w_f32.len() < weight_needed {
                return Err(RuntimeError::Compute(format!(
                    "sgemm {label}: weight buffer too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    w_f32.len(),
                    weight_needed,
                )));
            }
            // cuBLAS SGEMM: C_cm[out_dim, batch] = W_cm^T[out_dim, in_dim] * A_cm[in_dim, batch]
            // transa = T (transpose W_cm to get [out_dim, in_dim])
            // transb = N (A_cm is already [in_dim, batch])
            // m = out_dim, n = batch, k = in_dim
            // lda = in_dim (leading dim of W_cm[in_dim, out_dim])
            // ldb = in_dim (leading dim of A_cm[in_dim, batch])
            // ldc = out_dim (leading dim of C_cm[out_dim, batch])
            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 0.0f32,
                ldc: out_dim as i32,
            };
            device
                .blas
                .gemm(cfg, w_f32, input, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!("cuBLAS SGEMM {label}: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // when LUMEN_CUDA_Q8_PROJ_MMQ=1, route Q8 projection
            // through the MMQ INT8 dp4a math (INT8xINT8->INT32->F32-scale).
            // This is the path required to close the qkv_pre_conv 5.85e-2
            // max-abs drift that HGEMM-F16 and SGEMM-F32 paths could not close.
            // Default OFF preserves byte-identical behaviour vs main; ENV-ON
            // routes to mmq_q8_0_batched (modulo dp4a/MMA microarchitecture
            // differences).
            // when LUMEN_CUDA_GDN_AB_F32=1 (or NORM_RSQRTF_BUNDLE=1), bypass MMQ
            // for `gdn_alpha` / `gdn_beta` projections specifically. This
            // forces the Q8 -> F32-dequant -> SGEMM-F32 path, restoring the
            // F32 SGEMM data path for these tensors (the GGUF source weight
            // is F32).
            let use_mmq = std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").is_ok()
                && !force_alpha_beta_f32
                && kernels.mmq_q8_0_batched.is_some()
                && in_dim % 32 == 0;
            if use_mmq {
                static MMQ_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !MMQ_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: Prefill Q8 MMQ: ACTIVE ({label}, batch={batch}, \
                         out_dim={out_dim}, in_dim={in_dim})"
                    );
                }
                launch_mmq_q8_0_batched(
                    device, kernels, w_q8, input, output,
                    out_dim, in_dim, batch, label,
                )?;
                return Ok(());
            }
            // Dequantize Q8_0 weights to F16 scratch, then cuBLAS HGEMM (tensor cores).
            //
            // This replaces the old F32 SGEMM path (19.5 TFLOPS) with:
            // 1. One dequant_q8_0_to_f16 kernel (num_elements threads)
            // 2. One f32_to_f16 conversion of activations
            // 3. One cublasGemmEx HGEMM call (312 TFLOPS on A100)
            //
            // Critical for GDN layers where F16 weight caches are skipped (OOM concern)
            // but HGEMM tensor cores provide 16x higher throughput than SGEMM.
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: when LUMEN_CUDA_PREFILL_F32 is set, take the
            // SGEMM-F32 fallback even if the F16 scratch is large enough. This
            // tests whether HGEMM-F16 rounding accounts for the L0 drift on
            // the Q8 projection (qkv_pre_conv) path.
            // also force F32 SGEMM for `gdn_alpha`/`gdn_beta` to
            // restore the F32-GGUF path for these projections.
            if force_f32 || force_alpha_beta_f32 || dequant_f16.len() < f16_bytes_needed {
                if force_alpha_beta_f32 {
                    static AB_F32_LOGGED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !AB_F32_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[CUDA]: GDN alpha/beta forced through F32 SGEMM \
                             (F32 SGEMM on dequantized GDN alpha/beta; bypasses MMQ/HGEMM)"
                        );
                    }
                }
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if dequant_scratch.len() < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        dequant_scratch.len(),
                        num_elements,
                    )));
                }
                launch_dequant_q8_0_to_f32(
                    device, kernels, w_q8, dequant_scratch, num_elements, label,
                )?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 0.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM fallback (dequant Q8_0) {label}: {e}"
                        ))
                    })?;
            } else {
                static Q8_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !Q8_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[CUDA] Prefill Q8_0 HGEMM: ACTIVE (dequant->F16->tensor core path)");
                }

                // Step 1: Dequantize Q8_0 -> F16 in scratch buffer.
                launch_dequant_q8_0_to_f16(
                    device, kernels, w_q8, dequant_f16, num_elements, label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device, dequant_f16, activation_f16, output,
                    out_dim, batch, in_dim, 0.0, label,
                )?;
            }
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            // Native F16 weights: cublasGemmEx HGEMM directly (no dequant needed).
            // Convert F32 activations to F16, then HGEMM with F16 weights.
            // note: when LUMEN_CUDA_PREFILL_F32=1, this branch CANNOT
            // take an F32 fallback because the source weight is already F16.
            if force_f32 {
                static F16_FORCED_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !F16_FORCED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: LUMEN_CUDA_PREFILL_F32=1 but weight is F16Raw \
                         ({label}); HGEMM-F16 path still taken (weight class is F16)."
                    );
                }
            }
            static F16_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !F16_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[CUDA] Prefill HGEMM F16Raw: ACTIVE (native F16 tensor core path)");
            }

            // Step 1: Convert F32 activation to F16 via vectorized kernel.
            launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

            // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
            launch_cublas_hgemm(
                device, w_f16, activation_f16, output,
                out_dim, batch, in_dim, 0.0, label,
            )?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            // Dequantize Q4_0 weights to F16 scratch, then cuBLAS HGEMM (tensor cores).
            //
            // Same pattern as Q8Raw HGEMM: replaces `batch` sequential matvec launches with:
            // 1. One dequant_q4_0_to_f16 kernel (num_elements threads)
            // 2. One f32_to_f16 conversion of activations
            // 3. One cublasGemmEx HGEMM call (312 TFLOPS on A100)
            //
            // Critical for GDN layers where F16 caches are skipped to save GPU memory.
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || dequant_f16.len() < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if dequant_scratch.len() < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        dequant_scratch.len(),
                        num_elements,
                    )));
                }
                launch_dequant_q4_0_to_f32(
                    device, kernels, w_q4, dequant_scratch, num_elements, label,
                )?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 0.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM fallback (dequant Q4_0) {label}: {e}"
                        ))
                    })?;
            } else {
                static Q4_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !Q4_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[CUDA] Prefill Q4_0 HGEMM: ACTIVE (dequant->F16->tensor core path)");
                }

                // Step 1: Dequantize Q4_0 -> F16 in scratch buffer.
                launch_dequant_q4_0_to_f16(
                    device, kernels, w_q4, dequant_f16, num_elements, label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device, dequant_f16, activation_f16, output,
                    out_dim, batch, in_dim, 0.0, label,
                )?;
            }
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill: fall back to per-row matvec
            // (aligned format is optimized for single-token decode, not GEMM).
            for row in 0..batch {
                let in_offset = row * in_dim;
                let out_offset = row * out_dim;
                launch_matvec_slice(
                    device, kernels, weight, input, output, in_offset, out_offset,
                    out_dim, in_dim, label,
                )?;
            }
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Q4Aligned should not appear in prefill -- aligned repack is skipped
            // for GDN models, and non-GDN models use F16 HGEMM (routed above).
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight in batched prefill GEMM {label} -- unexpected"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // BF16 prefill: cuBLAS `cublasGemmEx` with CUDA_R_16BF inputs and
            // F32 output (CUBLAS_COMPUTE_32F). Tensor-core BF16 mma.sync path
            // on SM_80+ (A100 312 TFLOPS). Replaces the previous per-row
            // matvec_bf16 fallback which was ~42x slower than llama.cpp on
            // batched prefill.
            //
            // BF16 has the same 2 B/elem footprint as F16 — we reuse the F16
            // activation scratch buffer (`activation_f16`) for the BF16
            // activation conversion. The buffer is `CudaSlice<u8>`, so the
            // interpretation is determined by the cuBLAS `cudaDataType_t`.
            let activation_needed = batch * in_dim * 2;
            if activation_f16.len() < activation_needed {
                return Err(RuntimeError::Compute(format!(
                    "bf16_gemm {label}: activation scratch too small: have {} bytes, \
                     need {} (batch={batch}, in_dim={in_dim})",
                    activation_f16.len(),
                    activation_needed,
                )));
            }
            static BF16_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !BF16_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[CUDA] Prefill BF16 GemmEx: ACTIVE (native BF16 tensor core path)");
            }

            // Step 1: Convert F32 activation to BF16 via vectorized kernel.
            launch_f32_to_bf16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

            // Step 2: cublasGemmEx (BF16 weight + BF16 activation -> F32 output).
            launch_cublas_gemm_bf16(
                device, w_bf16, activation_f16, output,
                out_dim, batch, in_dim, 0.0, label,
            )?;
        }
        // split-layout: / TILE: prefill never dispatches against
        // Q8Split/Q4Split/Q8Tile/Q4Tile siblings. These are decode-only
        // reorganizations; the prefill path operates on the original AoS
        // Q8Raw/Q4Raw via dequant->F16->cuBLAS HGEMM. If we somehow get
        // here the caller has confused decode/prefill dispatch.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "prefill GEMM {label}: Q8Split/Q4Split/Q8Tile/Q4Tile sibling \
                 routed to prefill; prefill must use the original Q8Raw/Q4Raw \
                 buffer (dequant->HGEMM path)",
            )));
        }
    }
    Ok(())
}

/// Batched GEMM with fused residual: out = input * weight^T + residual.
///
/// For F32 weights, copies the residual into the output buffer first, then
/// calls cuBLAS SGEMM with beta=1.0 to accumulate C = alpha*A*B^T + beta*C.
/// For Q8_0 weights with F16 cache, uses cublasGemmEx HGEMM with beta=1.0.
/// For Q8_0 without F16 cache, dequantizes to F32 scratch then SGEMM.
/// For native F16 weights (`F16Raw`), uses cublasGemmEx HGEMM directly with
/// beta=1.0 (no dequant needed). For Q4_0, falls back to per-row matvec +
/// residual.
///
/// # Safety
///
/// `input` must be [batch, in_dim]. `weight` must be [out_dim, in_dim].
/// `residual` and `output` must be [batch, out_dim]. `dequant_scratch` must
/// have at least `out_dim * in_dim` elements (only used for Q8_0 weights).
pub(crate) unsafe fn launch_gemm_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    input: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    dequant_scratch: &mut CudaSlice<f32>,
    activation_f16: &mut CudaSlice<u8>,
    dequant_f16: &mut CudaSlice<u8>,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // Validate buffer sizes before launching GPU work.
    let input_needed = batch * in_dim;
    let output_needed = batch * out_dim;
    if input.len() < input_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: input buffer too small: have {} elements, \
             need {} (batch={batch}, in_dim={in_dim})",
            input.len(),
            input_needed,
        )));
    }
    if residual.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: residual buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            residual.len(),
            output_needed,
        )));
    }
    if output.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: output buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            output.len(),
            output_needed,
        )));
    }

    // diagnostic: when LUMEN_CUDA_PREFILL_F32=1, bypass HGEMM-F16 paths
    // and use the SGEMM-F32 fallbacks. Used to test the HGEMM-F16-precision
    // hypothesis on the linear_attn_out (ssm_out residual) GEMM.
    let force_f32 = std::env::var("LUMEN_CUDA_PREFILL_F32").is_ok();

    // same env gate as's projection MMQ (LUMEN_CUDA_Q8_PROJ_MMQ=1)
    // also covers the residual GEMM site (ssm_out). When the env is set AND the
    // weight is Q8Raw AND the kernel loaded, prefer the MMQ residual path over
    // the F16-cache fast path so the MMQ INT8 dp4a math is taken on the GDN-
    // block exit projection (`launch_gemm_residual` is called with
    // label="gdn_ssm_out" from backend_impl.rs:5012).
    //
    // closed `qkv_pre_conv` drift 7700x; closes the residual
    // `linear_attn_out` drift (~0.226 max-abs persists with alone) by
    // extending the same MMQ math to this second Q8 projection site.
    //
    // Diagnostic sub-gate: LUMEN_CUDA_Q8_SSM_OUT_MMQ_OFF=1 disables ONLY the
    // residual MMQ arm (used to A/B isolate the prior vs
    //  Default-on when parent env is set.
    let q8_proj_mmq = std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").is_ok();
    let ssm_out_mmq_off = std::env::var("LUMEN_CUDA_Q8_SSM_OUT_MMQ_OFF").is_ok();
    let q8_residual_mmq = q8_proj_mmq && !ssm_out_mmq_off;
    let weight_is_q8raw = matches!(weight, GpuWeightBuf::Q8Raw(_));
    let prefer_mmq_over_f16cache = q8_residual_mmq && weight_is_q8raw && in_dim % 32 == 0
        && kernels.mmq_q8_0_batched_residual.is_some();

    // diagnostic: log weight variant + gate decision on FIRST entry to
    // each label, to confirm MMQ residual dispatch actually fires on
    // `gdn_ssm_out` at L0 (not just `wo` at dense-attn layers). Strictly
    // diagnostic; remove or env-gate after release sign-off.
    if q8_proj_mmq {
        static GEMM_RES_TRACE: std::sync::Once = std::sync::Once::new();
        GEMM_RES_TRACE.call_once(|| {
            let variant = match weight {
                GpuWeightBuf::F32(_) => "F32",
                GpuWeightBuf::Q8Raw(_) => "Q8Raw",
                GpuWeightBuf::Q4Raw(_) => "Q4Raw",
                GpuWeightBuf::F16Raw(_) => "F16Raw",
                _ => "OTHER",
            };
            eprintln!(
                "[CUDA] trace[1st call]: launch_gemm_residual label={label} weight={variant} \
                 in_dim={in_dim} out_dim={out_dim} f16_cache={} prefer_mmq={} q8_residual_mmq={}",
                weight_f16_cache.is_some(),
                prefer_mmq_over_f16cache,
                q8_residual_mmq,
            );
        });
    }

    // Fast path: HGEMM residual with pre-dequanted F16 weights.
    // Copy residual to output first, then HGEMM with beta=1.0.
    let f16_cache_active = !force_f32 && !prefer_mmq_over_f16cache && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        // Copy residual -> output for beta=1.0 accumulation.
        device
            .stream
            .memcpy_dtod(residual, output)
            .map_err(|e| RuntimeError::Compute(format!(
                "dtod residual copy {label}: {e}"
            )))?;

        // Convert F32 activation to F16 via vectorized kernel.
        launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

        // cublasGemmEx with beta=1.0 for residual accumulation.
        launch_cublas_hgemm(
            device, w_f16, activation_f16, output,
            out_dim, batch, in_dim, 1.0, label,
        )?;
        return Ok(());
    }

    match weight {
        GpuWeightBuf::F32(w_f32) => {
            let weight_needed = out_dim * in_dim;
            if w_f32.len() < weight_needed {
                return Err(RuntimeError::Compute(format!(
                    "sgemm_residual {label}: weight buffer too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    w_f32.len(),
                    weight_needed,
                )));
            }
            // Copy residual -> output so SGEMM can accumulate with beta=1.0.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dtod residual copy for {label}: {e}"
                    ))
                })?;

            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 1.0f32,
                ldc: out_dim as i32,
            };
            device
                .blas
                .gemm(cfg, w_f32, input, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!("cuBLAS SGEMM+residual {label}: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // when LUMEN_CUDA_Q8_PROJ_MMQ=1 (and the sub-gate to disable
            // ONLY this site is NOT set), route Q8 residual GEMM through the
            // MMQ kernel (INT8xINT8->INT32->F32-scale math) with fused residual
            // add. Default OFF preserves byte-identical behavior vs main;
            // ENV-ON closes the `linear_attn_out` ~0.226 max-abs drift that
            // survives's projection-only fix.
            let use_mmq = q8_residual_mmq
                && kernels.mmq_q8_0_batched_residual.is_some()
                && in_dim % 32 == 0;
            if use_mmq {
                static MMQ_RES_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !MMQ_RES_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: Prefill Q8 MMQ+residual: ACTIVE ({label}, batch={batch}, \
                         out_dim={out_dim}, in_dim={in_dim})"
                    );
                }
                launch_mmq_q8_0_batched_residual(
                    device, kernels, w_q8, input, residual, output,
                    out_dim, in_dim, batch, label,
                )?;
                return Ok(());
            }
            // Dequantize Q8_0 -> F16, then cuBLAS HGEMM with fused residual (tensor cores).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || dequant_f16.len() < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if dequant_scratch.len() < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        dequant_scratch.len(),
                        num_elements,
                    )));
                }
                launch_dequant_q8_0_to_f32(
                    device, kernels, w_q8, dequant_scratch, num_elements, label,
                )?;
                device
                    .stream
                    .memcpy_dtod(residual, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0 fallback) {label}: {e}"
                        ))
                    })?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 1.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM+residual fallback (dequant Q8_0) {label}: {e}"
                        ))
                    })?;
            } else {
                // Step 1: Copy residual -> output for beta=1.0 accumulation.
                device
                    .stream
                    .memcpy_dtod(residual, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0) {label}: {e}"
                        ))
                    })?;

                // Step 2: Dequantize Q8_0 -> F16 in scratch buffer.
                launch_dequant_q8_0_to_f16(
                    device, kernels, w_q8, dequant_f16, num_elements, label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device, dequant_f16, activation_f16, output,
                    out_dim, batch, in_dim, 1.0, label,
                )?;
            }
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            // Native F16 weights: cublasGemmEx HGEMM with residual (no dequant needed).
            // Copy residual -> output, convert F32 activation to F16, then HGEMM beta=1.0.
            // note: when LUMEN_CUDA_PREFILL_F32=1, this branch CANNOT
            // take an F32 fallback because the source weight is already F16.
            if force_f32 {
                static F16_FORCED_RES_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !F16_FORCED_RES_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: LUMEN_CUDA_PREFILL_F32=1 but residual weight is F16Raw \
                         ({label}); HGEMM-F16 path still taken."
                    );
                }
            }

            // Step 1: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| RuntimeError::Compute(format!(
                    "dtod residual copy F16Raw {label}: {e}"
                )))?;

            // Step 2: Convert F32 activation to F16 via vectorized kernel.
            launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

            // Step 3: cublasGemmEx with beta=1.0 for residual accumulation.
            launch_cublas_hgemm(
                device, w_f16, activation_f16, output,
                out_dim, batch, in_dim, 1.0, label,
            )?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            // Dequantize Q4_0 -> F16, then cuBLAS HGEMM with fused residual (tensor cores).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || dequant_f16.len() < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if dequant_scratch.len() < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        dequant_scratch.len(),
                        num_elements,
                    )));
                }
                launch_dequant_q4_0_to_f32(
                    device, kernels, w_q4, dequant_scratch, num_elements, label,
                )?;
                device
                    .stream
                    .memcpy_dtod(residual, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q4_0 fallback) {label}: {e}"
                        ))
                    })?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 1.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM+residual fallback (dequant Q4_0) {label}: {e}"
                        ))
                    })?;
            } else {
                // Step 1: Copy residual -> output for beta=1.0 accumulation.
                device
                    .stream
                    .memcpy_dtod(residual, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q4_0) {label}: {e}"
                        ))
                    })?;

                // Step 2: Dequantize Q4_0 -> F16 in scratch buffer.
                launch_dequant_q4_0_to_f16(
                    device, kernels, w_q4, dequant_f16, num_elements, label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device, dequant_f16, activation_f16, output,
                    out_dim, batch, in_dim, 1.0, label,
                )?;
            }
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill residual: fall back to per-row matvec.
            for row in 0..batch {
                let in_offset = row * in_dim;
                let res_offset = row * out_dim;
                let out_offset = row * out_dim;
                launch_matvec_residual_slice(
                    device, kernels, weight, input, residual, output, in_offset,
                    res_offset, out_offset, out_dim, in_dim, label,
                )?;
            }
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight in batched prefill GEMM residual {label} -- unexpected"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // BF16 prefill + residual: cuBLAS `cublasGemmEx` with beta=1.0.
            // Copies residual -> output, converts F32 activation -> BF16, then
            // invokes the tensor-core BF16 GemmEx (CUDA_R_16BF, F32 acc) which
            // accumulates W * A + residual into output.
            // replaces the per-row matvec_bf16_residual fallback (the 42x gap).
            let activation_needed = batch * in_dim * 2;
            if activation_f16.len() < activation_needed {
                return Err(RuntimeError::Compute(format!(
                    "bf16_gemm_residual {label}: activation scratch too small: have {} bytes, \
                     need {} (batch={batch}, in_dim={in_dim})",
                    activation_f16.len(),
                    activation_needed,
                )));
            }

            // Step 1: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| RuntimeError::Compute(format!(
                    "dtod residual copy Bf16Raw {label}: {e}"
                )))?;

            // Step 2: Convert F32 activation to BF16 via vectorized kernel.
            launch_f32_to_bf16_fast(device, kernels, input, activation_f16, batch * in_dim, label)?;

            // Step 3: cublasGemmEx BF16 with beta=1.0 for residual accumulation.
            launch_cublas_gemm_bf16(
                device, w_bf16, activation_f16, output,
                out_dim, batch, in_dim, 1.0, label,
            )?;
        }
        // split-layout: / TILE: prefill never dispatches against
        // Q8Split/Q4Split/Q8Tile/Q4Tile siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "prefill residual GEMM {label}: Q8Split/Q4Split/Q8Tile/Q4Tile \
                 sibling routed to prefill; prefill must use the original \
                 Q8Raw/Q4Raw buffer",
            )));
        }
    }
    Ok(())
}

/// Launch batched RoPE for Q and K matrices.
///
/// # Safety
///
/// `q` must be [batch, q_dim]. `k` must be [batch, kv_dim].
pub(crate) unsafe fn launch_rope_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &mut CudaSlice<f32>,
    k: &mut CudaSlice<f32>,
    pos_start: usize,
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
    rope_neox: bool,
    rotary_dim: u32,
) -> Result<(), RuntimeError> {
    // For partial RoPE (rotary_dim > 0), only rotate first rotary_dim dims per head.
    let actual_rot = if rotary_dim > 0 && (rotary_dim as usize) < head_dim { rotary_dim as usize } else { head_dim };
    let half_rot = actual_rot / 2;
    let total_q_pairs = num_q_heads * half_rot;
    let total_work = batch * total_q_pairs;
    let config = LaunchConfig::for_elements(total_work);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let pos_start_u32 = pos_start as u32;
    let batch_u32 = batch as u32;
    let nqh = num_q_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;

    let rope_fn = if rope_neox {
        &kernels.rope_apply_batched_neox
    } else {
        &kernels.rope_apply_batched
    };
    device
        .stream
        .launch_builder(rope_fn)
        .arg(&mut *q)
        .arg(&mut *k)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nqh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&theta)
        .arg(&rotary_dim)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("rope_apply_batched launch: {e}")))?;
    Ok(())
}

/// Write a batch of K/V data to the GPU KV cache at positions pos_start..pos_start+batch-1.
///
/// # Safety
///
/// `data` must be [batch, num_kv_heads * head_dim]. Cache must have capacity.
pub(crate) unsafe fn launch_kv_cache_write_batch(
    device: &CudaDevice,
    kernels: &KernelSet,
    cache: &mut CudaSlice<f32>,
    data: &CudaSlice<f32>,
    pos_start: usize,
    batch: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<(), RuntimeError> {
    let kv_dim = num_kv_heads * head_dim;
    let total = batch * kv_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let pos_start_u32 = pos_start as u32;
    let batch_u32 = batch as u32;
    let nkvh = num_kv_heads as u32;
    let msl = max_seq_len as u32;
    let hd = head_dim as u32;
    device
        .stream
        .launch_builder(&kernels.kv_cache_write_batch)
        .arg(cache)
        .arg(data)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nkvh)
        .arg(&msl)
        .arg(&hd)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("kv_cache_write_batch launch: {e}")))?;
    Ok(())
}

/// Batched SwiGLU: gate = silu(gate) * up for [batch, inter_dim].
///
/// # Safety
///
/// `gate` and `up` must have `batch * inter_dim` elements.
pub(crate) unsafe fn launch_swiglu_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    batch: usize,
    inter_dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * inter_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_u32 = total as u32;
    device
        .stream
        .launch_builder(&kernels.swiglu_batched)
        .arg(gate)
        .arg(up)
        .arg(&total_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("swiglu_batched launch: {e}")))?;
    Ok(())
}

/// Batched residual add: x += residual for [batch, dim].
///
/// # Safety
///
/// `x` and `residual` must have `batch * dim` elements.
pub(crate) unsafe fn launch_residual_add_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &mut CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    batch: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_u32 = total as u32;
    device
        .stream
        .launch_builder(&kernels.residual_add_batched)
        .arg(x)
        .arg(residual)
        .arg(&total_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("residual_add_batched launch: {e}")))?;
    Ok(())
}

/// Extract a single row from a [batch, dim] matrix into a [dim] vector.
///
/// # Safety
///
/// `matrix` must have at least `(row_idx + 1) * dim` elements.
/// `output` must have `dim` elements.
pub(crate) unsafe fn launch_extract_row(
    device: &CudaDevice,
    kernels: &KernelSet,
    matrix: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    row_idx: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(dim);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_u32 = row_idx as u32;
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.extract_row)
        .arg(matrix)
        .arg(output)
        .arg(&row_u32)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("extract_row launch: {e}")))?;
    Ok(())
}

/// Write a [dim] vector into row `row_idx` of a [batch, dim] matrix on GPU.
///
/// # Safety
///
/// `matrix` must have at least `(row_idx + 1) * dim` elements.
/// `input` must have `dim` elements.
#[allow(dead_code)]
pub(crate) unsafe fn launch_scatter_row(
    device: &CudaDevice,
    kernels: &KernelSet,
    matrix: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    row_idx: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(dim);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_u32 = row_idx as u32;
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.scatter_row)
        .arg(matrix)
        .arg(input)
        .arg(&row_u32)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("scatter_row launch: {e}")))?;
    Ok(())
}

/// Sequential attention for one token during batched prefill.
///
/// Extracts token's Q from the batched Q matrix, runs the existing
/// attention_decode kernel against the KV cache (which contains positions
/// 0..seq_len-1), then scatter-writes the result back into attn_out_batch.
///
/// # Safety
///
/// `q_batch` must be [batch, q_dim]. `q_single` and `attn_out_single` must
/// be [q_dim]. `attn_out_batch` must be [batch, q_dim]. KV cache must have
/// valid data for `seq_len` positions.
#[allow(dead_code)]
pub(crate) unsafe fn launch_attention_for_token(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    attn_out_batch: &mut CudaSlice<f32>,
    q_single: &mut CudaSlice<f32>,
    attn_out_single: &mut CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    token_idx: usize,
    q_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<(), RuntimeError> {
    // Extract this token's Q vector from the batch.
    launch_extract_row(device, kernels, q_batch, q_single, token_idx, q_dim)?;

    // Run decode-attention for this single token. gate: routes to
    // the tiled streaming-softmax kernel at long context (seq_len > threshold,
    // default 0 = "tiled-always") or when LUMEN_CUDA_DECODE_TILED=1
    // forces it. Operators can set `LUMEN_CUDA_DECODE_TILED_THRESHOLD=
    // 4294967295` to opt out (force single-block below the 40_950 ceiling).
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let sl = seq_len as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    launch_attention_decode_gated(
        device,
        kernels,
        q_single as &CudaSlice<f32>,
        &kv_cache.k_cache,
        &kv_cache.v_cache,
        &mut *attn_out_single,
        nh,
        nkvh,
        hd,
        sl,
        msl,
        scale,
    )
    .map_err(|e| {
        RuntimeError::Compute(format!("attention_decode prefill t={token_idx}: {e}"))
    })?;

    // Scatter-write attn_out_single back into the batch matrix at row token_idx.
    launch_scatter_row(
        device, kernels, attn_out_batch, &*attn_out_single, token_idx, q_dim,
    )?;

    Ok(())
}

/// Launch the tiled streaming-softmax decode-attention kernel.
///
/// Closes the single-block `attention_decode` kernel's `seq_len <= 40_950`
/// ceiling by streaming the softmax over fixed-size KV tiles (T_C=128) using
/// Dao 2022 online-softmax mechanics. Per-CTA shared memory is constant in
/// `seq_len` (~1.6 KB at head_dim=256); no `cuFuncSetAttribute` opt-in
/// required.
///
/// Grid: `(num_heads, 1, 1)` — one CTA per query head. Block: 128 threads
/// (4 warps). Mirrors the single-block kernel's grid topology so the gate
/// dispatch can swap kernels without changing call shape.
///
/// Errors when the tiled kernel is unavailable (NVRTC compile failed at
/// backend init) — at `seq_len > ATTN_DECODE_EXTENDED_SHMEM_MAX_SEQ_LEN`
/// the single-block kernel cannot serve, so a missing tiled kernel is an
/// operator-visible error rather than a silent fallback.
///
/// # Safety
///
/// Same buffer constraints as `attention_decode`:
///   - `q` has `num_heads * head_dim` elements.
///   - `k_cache`, `v_cache` have `num_kv_heads * max_seq_len * head_dim` elements.
///   - `attn_out` has `num_heads * head_dim` elements.
///   - `seq_len <= max_seq_len`.
///   - `head_dim` must be divisible by `ATTN_DECODE_TILED_BLOCK_DIM` (128)
///     for the kernel's per-thread output slot addressing to be exact.
///     Qwen3.5-9B uses head_dim=256 (256 % 128 == 0). Models with head_dim
///     that is not a multiple of 128 will fail this guard.
pub(crate) unsafe fn launch_attention_decode_tiled(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    // Host-side invariant guard (mirror of the kernel's documented invariant):
    // head_dim must be divisible by block_size for the output-slot addressing
    // to be exact. Pass 3 refinement #1.)
    if head_dim % ATTN_DECODE_TILED_BLOCK_DIM != 0 {
        return Err(RuntimeError::Compute(format!(
            "attention_decode_tiled: head_dim ({head_dim}) must be divisible by \
             BLOCK_DIM ({ATTN_DECODE_TILED_BLOCK_DIM}); production Qwen3.5-9B \
             has head_dim=256 which is supported"
        )));
    }

    let kernel = kernels.attention_decode_tiled.as_ref().ok_or_else(|| {
        RuntimeError::Compute(
            "attention_decode_tiled: kernel not available (NVRTC compile failed at backend init); \
             long-context decode (seq_len > 40_950) cannot proceed on this device"
                .into(),
        )
    })?;

    let shared_bytes = attention_decode_tiled_shared_bytes(head_dim);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    device
        .stream
        .launch_builder(kernel)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(attn_out)
        .arg(&num_heads)
        .arg(&num_kv_heads)
        .arg(&head_dim)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_tiled launch: {e}")))?;

    Ok(())
}

/// Gate-and-dispatch the appropriate decode-attention kernel for `seq_len`.
///
/// Single source of truth for the kernel selection logic, used at all five
/// production launch sites:
///   1. `backend_impl.rs::compute_layer_gpu` — primary decode entry
///   2. `backend_impl.rs::compute_layer_decode` per-layer body
///   3. `backend_impl.rs` CUDA graph capture body (eager-fallback gate
///      lives at the caller; this wrapper is bypassed when graph capture
///      is active and the variant is Tiled — see caller `:4810`)
///   4. `prefill.rs::launch_attention_for_token` per-token prefill fallback
///   5. `prefill_attention.rs::prefill_attention_sequential` per-token prefill
///
/// Returns the variant chosen so callers (notably the CUDA graph site) can
/// branch on it for eager-fallback decisions WITHOUT re-evaluating the gate.
///
/// # Safety
///
/// Same buffer / shape constraints as the underlying kernels. The wrapper
/// does not re-check them — it forwards directly to the chosen launcher.
pub(crate) unsafe fn launch_attention_decode_gated(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<AttentionDecodeVariant, RuntimeError> {
    // FA2 split-K decode path. Reuses the existing prefill FA2
    // splitk kernels (`flash_attention_fa2_splitk_partial` + `..._reduce`)
    // for the 16 dense-attn layers of Qwen3.5-MoE. The partial kernel is
    // single-warp-per-block and handles batch=1 natively (one (head, q_idx,
    // split) block per partial work-item); the reduce kernel merges across
    // splits. Algorithm follows the standard FA-tile (head_dim=128, block=128)
    // formulation except Lumen's F32 accumulators don't truncate to F16.
    //
    // Gated by `LUMEN_CUDA_FA2_ATTN=1` (default OFF). When ON AND both
    // partial+reduce kernels are loaded, ALL dense-attn decode calls route
    // through FA2 splitk regardless of seq_len. The variant returned is
    // `Tiled` (the closest semantic match: FA2 splitk also streams the
    // softmax over KV tiles).
    let fa2_attn_on = super::moe::fa2_attn_enabled();
    let fa2_partial_ok = kernels.flash_attention_fa2_splitk_partial.is_some();
    let fa2_reduce_ok = kernels.flash_attention_fa2_splitk_reduce.is_some();
    // one-time activation log (only when the gate flips ON).
    if fa2_attn_on {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[CUDA] LUMEN_CUDA_FA2_ATTN=1: dense-attn decode routes through FA2 splitk (partial_loaded={fa2_partial_ok} reduce_loaded={fa2_reduce_ok})"
            );
        }
    }
    if fa2_attn_on && fa2_partial_ok && fa2_reduce_ok {
        // Use FA2_SPLITK_SLICE = 1024 KV positions per split (matches
        // prefill-path constant). For decode the typical seq_len at the
        // dense-attn layers is small (start-of-generation: 16-256; mid:
        // 1-4K), so num_splits is usually 1-4.
        let slice_size = super::decode::FA2_SPLITK_SLICE;
        launch_flash_attention_fa2_splitk_decode(
            device,
            kernels,
            q,
            k_cache,
            v_cache,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            max_seq_len,
            scale,
            slice_size,
        )?;
        return Ok(AttentionDecodeVariant::Tiled);
    }

    let force_tiled = decode_tiled_force_enabled();
    let threshold = decode_tiled_threshold();
    let mut variant = attention_decode_variant(seq_len, force_tiled, threshold);

    // hardware-compat guard: the tiled kernel requires
    // `head_dim % BLOCK_DIM == 0` and `head_dim >= BLOCK_DIM`.
    // Production Qwen3.5-9B uses `head_dim = 256` which satisfies both.
    // Tiny test models (e.g. `head_dim = 4` in
    // `tests/cuda_e2e_generate_test.rs::TestModelConfig::default()`) do
    // NOT — fall back to SingleBlock instead of failing the launch with
    // `head_dim must be divisible by BLOCK_DIM`. This is a strict
    // superset of the prior default-threshold-36_864 behaviour
    // (tiny test models with seq_len << 36_864 already routed
    // SingleBlock); the default of 0 made them route Tiled at
    // seq_len > 0, which previously broke at launch.
    if variant == AttentionDecodeVariant::Tiled && !attention_decode_tiled_supports_head_dim(head_dim) {
        variant = AttentionDecodeVariant::SingleBlock;
    }

    match variant {
        AttentionDecodeVariant::SingleBlock => {
            // Single-block fast path (existing kernel, byte-identical to the
            // the prior dispatch when force_tiled = false and
            // seq_len <= threshold).
            let block_size = attention_block_size(seq_len as usize);
            let shared_bytes = attention_shared_bytes(seq_len);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            device
                .stream
                .launch_builder(&kernels.attention_decode)
                .arg(q)
                .arg(k_cache)
                .arg(v_cache)
                .arg(attn_out)
                .arg(&num_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&seq_len)
                .arg(&max_seq_len)
                .arg(&scale)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("attention_decode launch: {e}"))
                })?;
        }
        AttentionDecodeVariant::Tiled => {
            launch_attention_decode_tiled(
                device,
                kernels,
                q,
                k_cache,
                v_cache,
                attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                scale,
            )?;
        }
    }

    Ok(variant)
}

/// Launch the Q8_0 -> F32 dequantization kernel.
///
/// Dequantizes `num_elements` from the Q8_0 raw buffer into a contiguous F32 buffer.
/// Grid: ceil(num_elements / 256), Block: 256. Each thread dequantizes one element.
///
/// # Safety
///
/// `q8_data` must contain enough Q8_0 blocks for `num_elements` (i.e.,
/// ceil(num_elements / 32) * 34 bytes). `f32_out` must have `num_elements` elements.
/// launch the MMQ-style Q8_0 batched matmul kernel.
///
/// Computes `out[t, r] = sum_k(dequant(weight_q8[r, k]) * x[t, k])` via the
/// MMQ path: per-token Q8_1 activation quantization, INT8 dp4a inner products,
/// F32 scale at K-block granularity. INT32-exact intra-block sums, F32 scale
/// applied only at sum-time.
///
/// `in_dim` must be a multiple of 32 (Q8_0 block size).
///
/// # Safety
///
/// `weight_q8` must contain `out_dim * (in_dim/32) * 34` bytes of Q8_0 data.
/// `x` must have `batch * in_dim` F32 elements. `out` must have `batch *
/// out_dim` F32 elements.
pub(crate) unsafe fn launch_mmq_q8_0_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight_q8: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "mmq_q8_0_batched {label}: in_dim ({in_dim}) must be a multiple of 32"
        )));
    }
    let f = kernels.mmq_q8_0_batched.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!(
            "mmq_q8_0_batched {label}: kernel not loaded (requires SM 6.1+ for __dp4a)"
        ))
    })?;
    const NR: usize = 2;
    let grid_x = (out_dim + NR - 1) / NR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_x as u32, batch as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let batch_u32 = batch as u32;
    device
        .stream
        .launch_builder(f)
        .arg(weight_q8)
        .arg(x)
        .arg(out)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .arg(&batch_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("mmq_q8_0_batched {label}: {e}")))?;
    Ok(())
}

/// MMQ Q8_0 batched matmul WITH RESIDUAL ADD: `out = residual + W @ x`.
///
/// Same MMQ INT8 dp4a path as `launch_mmq_q8_0_batched`, except the kernel
/// fuses the residual add into the final store site. Used by
/// `launch_gemm_residual`'s Q8Raw path when `LUMEN_CUDA_Q8_PROJ_MMQ=1`, to
/// route the `ssm_out` (4096 -> 2048) GEMM through the MMQ INT8 path.
/// Closes the residual `linear_attn_out` drift (~0.226 max-abs) that survives
/// 's projection-only MMQ fix.
///
/// `in_dim` must be a multiple of 32 (Q8_0 block size).
///
/// # Safety
///
/// `weight_q8` must contain `out_dim * (in_dim/32) * 34` bytes of Q8_0 data.
/// `x` must have `batch * in_dim` F32 elements. `residual` and `out` must
/// have `batch * out_dim` F32 elements. `out` and `residual` may NOT alias
/// (the kernel reads `residual[idx]` and writes `out[idx]` in distinct
/// warps; aliasing would create a read-after-write hazard).
pub(crate) unsafe fn launch_mmq_q8_0_batched_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight_q8: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "mmq_q8_0_batched_residual {label}: in_dim ({in_dim}) must be a multiple of 32"
        )));
    }
    let f = kernels.mmq_q8_0_batched_residual.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!(
            "mmq_q8_0_batched_residual {label}: kernel not loaded \
             (requires SM 6.1+ for dp4a + the MMQ residual variant)"
        ))
    })?;
    const NR: usize = 2;
    let grid_x = (out_dim + NR - 1) / NR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_x as u32, batch as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let batch_u32 = batch as u32;
    device
        .stream
        .launch_builder(f)
        .arg(weight_q8)
        .arg(x)
        .arg(residual)
        .arg(out)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .arg(&batch_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("mmq_q8_0_batched_residual {label}: {e}")))?;
    Ok(())
}

unsafe fn launch_dequant_q8_0_to_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    q8_data: &CudaSlice<u8>,
    f32_out: &mut CudaSlice<f32>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_0_to_f32)
        .arg(q8_data)
        .arg(f32_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("dequant_q8_0_to_f32 {label}: {e}"))
        })?;
    Ok(())
}

/// Dequantize Q4_0 weights to F32 scratch buffer for cuBLAS SGEMM.
///
/// Each thread dequantizes one element: reads the block's F16 scale and the
/// element's 4-bit nibble, computes `scale * (nibble - 8)`.
///
/// # Safety
///
/// `q4_data` must contain valid Q4_0 blocks. `f32_out` must have at least
/// `num_elements` elements.
unsafe fn launch_dequant_q4_0_to_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_data: &CudaSlice<u8>,
    f32_out: &mut CudaSlice<f32>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q4_0_to_f32)
        .arg(q4_data)
        .arg(f32_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("dequant_q4_0_to_f32 {label}: {e}"))
        })?;
    Ok(())
}

/// Dequantize Q8_0 weights to F16 scratch buffer for cuBLAS HGEMM.
///
/// Each thread dequantizes one element from Q8_0 format to F16.
/// Enables the tensor core HGEMM path (312 TFLOPS on A100) for Q8_0 weights
/// that lack a persistent F16 cache (e.g., GDN layers).
///
/// # Safety
///
/// `q8_data` must contain valid Q8_0 blocks. `f16_out` must have at least
/// `num_elements * 2` bytes.
unsafe fn launch_dequant_q8_0_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    q8_data: &CudaSlice<u8>,
    f16_out: &mut CudaSlice<u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_0_to_f16)
        .arg(q8_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("dequant_q8_0_to_f16 {label}: {e}"))
        })?;
    Ok(())
}

/// Dequantize Q4_0 weights to F16 scratch buffer for cuBLAS HGEMM.
///
/// Each thread dequantizes one element from Q4_0 format to F16: reads the
/// block's F16 scale and the element's 4-bit nibble, computes
/// `f16(scale * (nibble - 8))`.
///
/// # Safety
///
/// `q4_data` must contain valid Q4_0 blocks. `f16_out` must have at least
/// `num_elements * 2` bytes.
unsafe fn launch_dequant_q4_0_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_data: &CudaSlice<u8>,
    f16_out: &mut CudaSlice<u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q4_0_to_f16)
        .arg(q4_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("dequant_q4_0_to_f16 {label}: {e}"))
        })?;
    Ok(())
}

/// Per-row matvec dispatch for non-F32 fallback path.
///
/// Dispatches the appropriate matvec kernel for a single row within a batched matrix,
/// using element offsets to address the correct row of input and output.
///
/// # Safety
///
/// Input and output slices must be large enough for the given offsets + dimensions.
unsafe fn launch_matvec_slice(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    _in_offset: usize,
    _out_offset: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // For the non-F32 fallback, we use try_slice to get views at the correct offsets.
    let in_view = input
        .try_slice(_in_offset.._in_offset + in_dim)
        .ok_or_else(|| RuntimeError::Compute(format!(
            "matvec_slice input slice out of bounds: offset={_in_offset} dim={in_dim}",
        )))?;
    let mut out_view = output
        .try_slice_mut(_out_offset.._out_offset + out_dim)
        .ok_or_else(|| RuntimeError::Compute(format!(
            "matvec_slice output slice out of bounds: offset={_out_offset} dim={out_dim}",
        )))?;

    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::F32(_) => unreachable!("F32 uses cuBLAS SGEMM path"),
        GpuWeightBuf::F16Raw(w_f16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_f16)
                .arg(w_f16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec F16 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels.matvec_q8_0_aligned.as_ref()
                .or(kernels.matvec_q8_0_dp4a.as_ref())
                .unwrap_or(&kernels.matvec_q8_0);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(q8a_fn)
                .arg(w_q8a)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q8_0 aligned {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // Q8_0 prefill: dp4a -> v1 fallback.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8_fn = kernels.matvec_q8_0_dp4a.as_ref()
                .unwrap_or(&kernels.matvec_q8_0);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let shmem = 0u32;
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(q8_fn)
                .arg(w_q8)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q8_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0)
                .arg(w_q4)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q4_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned in per-row matvec prefill {label} -- should route through F16 HGEMM"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_bf16)
                .arg(w_bf16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec BF16 {label} prefill: {e}"))
                })?;
        }
        // split-layout: / TILE: prefill never dispatches against
        // Q8Split/Q4Split/Q8Tile/Q4Tile siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec prefill fallback {label}: Q8Split/Q4Split/Q8Tile/Q4Tile \
                 sibling routed to prefill",
            )));
        }
    }
    Ok(())
}

/// Launch Flash Attention v2 (Br=1) for all tokens in a prefill batch.
///
/// Processes all query tokens against the KV cache in a SINGLE kernel launch
/// with causal masking. Each thread block handles one (head, token) pair.
///
/// Grid: (num_heads, batch, 1) -- one block per (head, query_token)
/// Block: (128, 1, 1) -- 128 threads (4 warps)
///
/// Replaces the sequential extract_row -> attention_decode -> scatter_row loop
/// that required 3 * batch kernel launches per layer.
///
/// # Arguments
///
/// * `q_batch` - Batched Q vectors `[batch, num_heads * head_dim]`
/// * `kv_cache` - GPU KV cache with data for positions 0..pos_start+batch-1
/// * `attn_out` - Output buffer `[batch, num_heads * head_dim]`
/// * `batch` - Number of query tokens
/// * `num_heads` - Number of Q attention heads
/// * `num_kv_heads` - Number of KV attention heads (for GQA)
/// * `head_dim` - Dimension per attention head
/// * `pos_start` - Position of first query token in the sequence
///
/// # Safety
///
/// * `q_batch` must have `batch * num_heads * head_dim` elements
/// * `attn_out` must have `batch * num_heads * head_dim` elements
/// * KV cache must contain valid data for `pos_start + batch` positions
#[allow(dead_code)]
pub(crate) unsafe fn launch_flash_attention_v2(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{flash_attention_v2_block_size, flash_attention_v2_shared_bytes};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_v2: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(), needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_v2: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(), needed,
        )));
    }

    let block_size = flash_attention_v2_block_size();
    let shared_bytes = flash_attention_v2_shared_bytes(head_dim as u32);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, batch as u32, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_v2)
        .arg(q_batch)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_v2 launch: {e}")))?;

    Ok(())
}

/// Launch Flash Attention Br=4 for all tokens in a prefill batch.
///
/// Processes 4 query tokens per thread block using warp-level parallelism.
/// Each of the 4 warps independently handles one query row, avoiding
/// block-level syncs between queries for higher throughput.
///
/// Grid: (num_heads, ceil(batch / 4), 1)
/// Block: (128, 1, 1) -- 4 warps of 32 threads
///
/// Preferred over `launch_flash_attention_v2` when batch >= 4, as it
/// processes 4x more queries per block with the same thread count.
///
/// # Arguments
///
/// Same as `launch_flash_attention_v2`.
///
/// # Safety
///
/// Same as `launch_flash_attention_v2`.
pub(crate) unsafe fn launch_flash_attention_br4(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{FA_BR, flash_attention_br4_block_size, flash_attention_br4_shared_bytes};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(), needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(), needed,
        )));
    }

    let block_size = flash_attention_br4_block_size();
    let shared_bytes = flash_attention_br4_shared_bytes(head_dim as u32);
    let q_tiles = (batch as u32 + FA_BR - 1) / FA_BR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, q_tiles, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_br4)
        .arg(q_batch)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_br4 launch: {e}")))?;

    Ok(())
}

/// Launch WMMA tensor-core Flash Attention for all tokens in a prefill batch.
///
/// Uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for QK^T and PV
/// matrix multiplies, providing up to 16x throughput over scalar F32 on A100.
///
/// Grid: (num_heads, ceil(batch / 16), 1)
/// Block: (128, 1, 1) -- 4 warps of 32 threads
///
/// # Arguments
///
/// Same as `launch_flash_attention_br4`.
///
/// # Safety
///
/// Same as `launch_flash_attention_br4`.
pub(crate) unsafe fn launch_flash_attention_wmma(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{FA_TC_BR, flash_attention_wmma_block_size, flash_attention_wmma_shared_bytes};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_wmma: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(), needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_wmma: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(), needed,
        )));
    }

    let block_size = flash_attention_wmma_block_size();
    let shared_bytes = flash_attention_wmma_shared_bytes(head_dim as u32);
    let q_tiles = (batch as u32 + FA_TC_BR - 1) / FA_TC_BR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, q_tiles, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let wmma_fn = kernels.flash_attention_wmma.as_ref()
        .ok_or_else(|| RuntimeError::Compute(
            "flash_attention_wmma: kernel not available (SM 8.0+ required)".into()
        ))?;

    device
        .stream
        .launch_builder(wmma_fn)
        .arg(q_batch)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_wmma launch: {e}")))?;

    Ok(())
}

/// Launch FA2 with mask block-skip (single-kernel mode) for an entire batch.
///
/// Each thread block processes FA2_BR=4 query rows for one (head, q_tile).
/// The kernel iterates only the lower-triangular tiles of (Q-tile, KV-tile),
/// saving O(seq_len^2 / 2) FLOPs vs the un-skipped implementation when
/// `pos_start + batch` is large.
///
/// Use `launch_flash_attention_fa2_splitk` instead when `pos_start + batch`
/// exceeds the SM occupancy threshold (typically > 4096).
///
/// # Safety
///
/// Same buffer constraints as `launch_flash_attention_br4`.
pub(crate) unsafe fn launch_flash_attention_fa2(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{
        FA2_BR, flash_attention_fa2_block_size, flash_attention_fa2_shared_bytes,
    };

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_fa2: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(),
            needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_fa2: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(),
            needed,
        )));
    }

    let fa2_fn = kernels
        .flash_attention_fa2_causal
        .as_ref()
        .ok_or_else(|| {
            RuntimeError::Compute(
                "flash_attention_fa2: kernel not available (compile failure)".into(),
            )
        })?;

    let block_size = flash_attention_fa2_block_size();
    let shared_bytes = flash_attention_fa2_shared_bytes(head_dim as u32);
    let q_tiles = ((batch as u32) + FA2_BR - 1) / FA2_BR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, q_tiles, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(fa2_fn)
        .arg(q_batch)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_fa2 launch: {e}")))?;

    Ok(())
}

/// Launch the Split-K FA2 pipeline (partial + reduce).
///
/// Allocates scratch buffers for partial (O, m, l) tuples on the supplied
/// device. The partial kernel is launched once per (head, q_idx, split); the
/// reduce kernel merges across splits into `attn_out`.
///
/// `slice_size` is the KV tile slice per split (typically 1024). `causal_max`
/// is the largest causal upper bound across the batch (= pos_start + batch);
/// the number of splits derives from `ceil(causal_max / slice_size)`.
///
/// # Safety
///
/// Same buffer constraints as `launch_flash_attention_fa2`. The Split-K path
/// allocates its own scratch buffers each call -- callers expecting graph
/// capture should hold scratch externally and use the partial / reduce
/// launchers directly. Documented limitation: this convenience wrapper
/// performs runtime allocations.
/// FA2 split-K decode dispatch.
///
/// Variant of `launch_flash_attention_fa2_splitk` for the decode path. Takes
/// raw `k_cache`/`v_cache` slices (decode site has them split out) and the
/// `seq_len` known at the dispatch point (vs prefill's `pos_start + batch`).
/// batch is always 1 for decode.
///
/// The kernels (`flash_attention_fa2_splitk_partial` + `..._reduce`) are
/// already loaded and validated under `LUMEN_CUDA_FA2_BLOCKSKIP=1` prefill
/// dispatch. This dispatch site brings them online for decode
/// of the dense-attn layers in Qwen3.5-MoE.
///
/// # Safety
///
/// `q` has `num_heads * head_dim` F32 elements.
/// `k_cache`, `v_cache` have `num_kv_heads * max_seq_len * head_dim` F32 each,
/// with `seq_len <= max_seq_len` valid KV positions written.
/// `attn_out` has `num_heads * head_dim` F32 elements.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_flash_attention_fa2_splitk_decode(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
    slice_size: u32,
) -> Result<(), RuntimeError> {
    use super::decode::{
        flash_attention_fa2_splitk_partial_block_size,
        flash_attention_fa2_splitk_partial_shared_bytes,
        flash_attention_fa2_splitk_reduce_block_size,
        flash_attention_fa2_splitk_reduce_shared_bytes,
    };

    let q_dim = (num_heads * head_dim) as usize;
    let needed = q_dim; // batch=1
    if q.len() < needed || attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "fa2_splitk_decode: q/attn_out too small (have {}/{}, need {})",
            q.len(), attn_out.len(), needed,
        )));
    }
    let partial_fn = kernels
        .flash_attention_fa2_splitk_partial
        .as_ref()
        .ok_or_else(|| RuntimeError::Compute(
            "fa2_splitk_decode: partial kernel not available".into()
        ))?;
    let reduce_fn = kernels
        .flash_attention_fa2_splitk_reduce
        .as_ref()
        .ok_or_else(|| RuntimeError::Compute(
            "fa2_splitk_decode: reduce kernel not available".into()
        ))?;

    // For decode, the causal upper bound is `seq_len` itself (the current
    // token attends to KV positions [0, seq_len-1] inclusive plus its OWN
    // KV write at index seq_len-1). The partial kernel computes
    // q_pos = pos_start + q_idx, causal_excl = q_pos + 1. For batch=1
    // decode at seq_len=N, the FA2 partial expects pos_start=N-1 so that
    // q_pos=N-1 → causal_excl=N. Lumen's decode dispatch passes seq_len AS
    // the "max KV index reachable" so we translate: pos_start = seq_len - 1.
    let causal_max = seq_len;
    if causal_max == 0 || slice_size == 0 {
        return Err(RuntimeError::Compute(
            "fa2_splitk_decode: seq_len or slice_size is zero".into()
        ));
    }
    let num_splits = ((causal_max + slice_size - 1) / slice_size).max(1);
    // batch=1 → partial scratch sized for (num_splits, 1, num_heads).
    let partial_o_len = (num_splits as usize) * (num_heads as usize) * (head_dim as usize);
    let partial_ml_len = (num_splits as usize) * (num_heads as usize);
    let mut partial_o = device
        .stream
        .alloc_zeros::<f32>(partial_o_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk_decode: alloc partial_o: {e}")))?;
    let mut partial_m = device
        .stream
        .alloc_zeros::<f32>(partial_ml_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk_decode: alloc partial_m: {e}")))?;
    let mut partial_l = device
        .stream
        .alloc_zeros::<f32>(partial_ml_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk_decode: alloc partial_l: {e}")))?;

    // pos_start = seq_len - 1 (single-token decode: q_pos = seq_len - 1).
    let pos_start: u32 = seq_len.saturating_sub(1);
    let batch_u32: u32 = 1;
    let partial_block = flash_attention_fa2_splitk_partial_block_size();
    let partial_shmem = flash_attention_fa2_splitk_partial_shared_bytes(head_dim);
    let partial_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads, batch_u32, num_splits),
        block_dim: (partial_block, 1, 1),
        shared_mem_bytes: partial_shmem,
    };
    device
        .stream
        .launch_builder(partial_fn)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut partial_o)
        .arg(&mut partial_m)
        .arg(&mut partial_l)
        .arg(&batch_u32)
        .arg(&num_heads)
        .arg(&num_kv_heads)
        .arg(&head_dim)
        .arg(&pos_start)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&slice_size)
        .arg(&num_splits)
        .launch(partial_cfg)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk_decode partial: {e}")))?;

    let reduce_block = flash_attention_fa2_splitk_reduce_block_size();
    let reduce_shmem = flash_attention_fa2_splitk_reduce_shared_bytes(num_splits);
    let reduce_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads, batch_u32, 1),
        block_dim: (reduce_block, 1, 1),
        shared_mem_bytes: reduce_shmem,
    };
    device
        .stream
        .launch_builder(reduce_fn)
        .arg(&partial_o)
        .arg(&partial_m)
        .arg(&partial_l)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&num_heads)
        .arg(&head_dim)
        .arg(&num_splits)
        .launch(reduce_cfg)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk_decode reduce: {e}")))?;
    Ok(())
}

pub(crate) unsafe fn launch_flash_attention_fa2_splitk(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv_cache: &KvCacheGpu,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
    slice_size: u32,
) -> Result<(), RuntimeError> {
    use super::decode::{
        flash_attention_fa2_splitk_partial_block_size,
        flash_attention_fa2_splitk_partial_shared_bytes,
        flash_attention_fa2_splitk_reduce_block_size,
        flash_attention_fa2_splitk_reduce_shared_bytes,
    };

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed || attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "fa2_splitk: q/attn_out too small (have {}/{}, need {})",
            q_batch.len(),
            attn_out.len(),
            needed,
        )));
    }

    let partial_fn = kernels
        .flash_attention_fa2_splitk_partial
        .as_ref()
        .ok_or_else(|| {
            RuntimeError::Compute("fa2_splitk: partial kernel not available".into())
        })?;
    let reduce_fn = kernels
        .flash_attention_fa2_splitk_reduce
        .as_ref()
        .ok_or_else(|| {
            RuntimeError::Compute("fa2_splitk: reduce kernel not available".into())
        })?;

    let causal_max = (pos_start + batch) as u32;
    if causal_max == 0 || slice_size == 0 {
        return Err(RuntimeError::Compute(
            "fa2_splitk: causal_max or slice_size is zero".into(),
        ));
    }
    let num_splits = ((causal_max + slice_size - 1) / slice_size).max(1);

    // Allocate per-call scratch. partial_o is [splits, batch, heads, head_dim]
    // and partial_ml is [splits, batch, heads].
    let partial_o_len = (num_splits as usize) * batch * num_heads * head_dim;
    let partial_ml_len = (num_splits as usize) * batch * num_heads;
    let mut partial_o = device
        .stream
        .alloc_zeros::<f32>(partial_o_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk: alloc partial_o: {e}")))?;
    let mut partial_m = device
        .stream
        .alloc_zeros::<f32>(partial_ml_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk: alloc partial_m: {e}")))?;
    let mut partial_l = device
        .stream
        .alloc_zeros::<f32>(partial_ml_len)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk: alloc partial_l: {e}")))?;

    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let batch_u32 = batch as u32;

    // Partial launch
    let partial_block = flash_attention_fa2_splitk_partial_block_size();
    let partial_shmem = flash_attention_fa2_splitk_partial_shared_bytes(hd);
    let partial_cfg = CudarcLaunchConfig {
        grid_dim: (nh, batch_u32, num_splits),
        block_dim: (partial_block, 1, 1),
        shared_mem_bytes: partial_shmem,
    };
    device
        .stream
        .launch_builder(partial_fn)
        .arg(q_batch)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(&mut partial_o)
        .arg(&mut partial_m)
        .arg(&mut partial_l)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .arg(&slice_size)
        .arg(&num_splits)
        .launch(partial_cfg)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk partial launch: {e}")))?;

    // Reduce launch
    let reduce_block = flash_attention_fa2_splitk_reduce_block_size();
    let reduce_shmem = flash_attention_fa2_splitk_reduce_shared_bytes(num_splits);
    let reduce_cfg = CudarcLaunchConfig {
        grid_dim: (nh, batch_u32, 1),
        block_dim: (reduce_block, 1, 1),
        shared_mem_bytes: reduce_shmem,
    };
    device
        .stream
        .launch_builder(reduce_fn)
        .arg(&partial_o)
        .arg(&partial_m)
        .arg(&partial_l)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&hd)
        .arg(&num_splits)
        .launch(reduce_cfg)
        .map_err(|e| RuntimeError::Compute(format!("fa2_splitk reduce launch: {e}")))?;

    Ok(())
}

/// Per-row matvec + residual dispatch for non-F32 fallback path.
///
/// # Safety
///
/// Same constraints as `launch_matvec_slice`, plus residual slice validity.
unsafe fn launch_matvec_residual_slice(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    in_offset: usize,
    res_offset: usize,
    out_offset: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let in_view = input
        .try_slice(in_offset..in_offset + in_dim)
        .ok_or_else(|| RuntimeError::Compute(format!(
            "matvec_res input slice out of bounds: offset={in_offset} dim={in_dim}",
        )))?;
    let res_view = residual
        .try_slice(res_offset..res_offset + out_dim)
        .ok_or_else(|| RuntimeError::Compute(format!(
            "matvec_res residual slice out of bounds: offset={res_offset} dim={out_dim}",
        )))?;
    let mut out_view = output
        .try_slice_mut(out_offset..out_offset + out_dim)
        .ok_or_else(|| RuntimeError::Compute(format!(
            "matvec_res output slice out of bounds: offset={out_offset} dim={out_dim}",
        )))?;

    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::F32(_) => unreachable!("F32 uses cuBLAS SGEMM path"),
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned residual prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels.matvec_q8_0_aligned_residual.as_ref()
                .or(kernels.matvec_q8_0_dp4a_residual.as_ref())
                .unwrap_or(&kernels.matvec_q8_0_residual);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(q8a_fn)
                .arg(w_q8a)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q8_0 aligned {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // Q8_0 residual prefill: dp4a -> v1 fallback.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8_fn = kernels.matvec_q8_0_dp4a_residual.as_ref()
                .unwrap_or(&kernels.matvec_q8_0_residual);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let shmem = 0u32;
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(q8_fn)
                .arg(w_q8)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q8_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_f16_residual)
                .arg(w_f16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&res_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res F16 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0_residual)
                .arg(w_q4)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q4_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned in per-row matvec+res prefill {label} -- should route through F16 HGEMM"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_bf16_residual)
                .arg(w_bf16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&res_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res BF16 {label} prefill: {e}"))
                })?;
        }
        // split-layout: / TILE: prefill never dispatches against
        // Q8Split/Q4Split/Q8Tile/Q4Tile siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec+residual prefill fallback {label}: Q8Split/Q4Split/Q8Tile/Q4Tile \
                 sibling routed to prefill",
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Shared helpers: F32->F16 fast conversion and cuBLAS HGEMM wrapper.
// ============================================================================

/// Fast F32->F16 conversion using vectorized kernel (4 elems/thread) when available,
/// falling back to scalar kernel (1 elem/thread) otherwise.
///
/// # Safety
///
/// `src` must have at least `num_elements` F32 values. `dst` must have at least
/// `num_elements * 2` bytes (F16 = 2 bytes each).
pub(crate) unsafe fn launch_f32_to_f16_fast(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let n = num_elements as u32;

    // Prefer vectorized kernel (4 elems/thread): 4x fewer threads, coalesced loads.
    if let Some(ref vec4_fn) = kernels.f32_to_f16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_f16_vec4 {label}: {e}",
            )))?;
    } else {
        // Fallback: scalar kernel (still uses hardware PTX cvt.rn.f16.f32).
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_f16_vec)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_f16_vec {label}: {e}",
            )))?;
    }
    Ok(())
}

/// Wrapper for cublasGemmEx HGEMM: F16 weight + F16 activation -> F32 output.
///
/// Computes C = alpha * W^T * A + beta * C where W is F16 [out_dim, in_dim],
/// A is F16 [batch, in_dim], and C is F32 [batch, out_dim].
///
/// Uses `CUBLAS_COMPUTE_32F_FAST_16F` for maximum tensor core throughput (312 TFLOPS A100).
///
/// # Safety
///
/// `w_f16` must be [out_dim * in_dim] F16 elements. `a_f16` must be [batch * in_dim] F16 elements.
/// `output` must be [batch * out_dim] F32 elements.
unsafe fn launch_cublas_hgemm(
    device: &CudaDevice,
    w_f16: &CudaSlice<u8>,
    a_f16: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    batch: usize,
    in_dim: usize,
    beta_val: f32,
    label: &str,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = beta_val;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = a_f16.device_ptr(&device.stream);
    let (c_ptr, _) = output.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,
        batch as i32,
        in_dim as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMM {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// F32 -> BF16 bulk conversion. Prefers vectorized (4 elems/thread) variant;
/// falls back to scalar. Used to convert activations before
/// `launch_cublas_gemm_bf16`.
///
/// # Safety
///
/// `src` must have at least `num_elements` F32 values. `dst` must have at
/// least `num_elements * 2` bytes (BF16 = 2 bytes each).
pub(crate) unsafe fn launch_f32_to_bf16_fast(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let n = num_elements as u32;

    if let Some(ref vec4_fn) = kernels.f32_to_bf16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16_vec4 {label}: {e}",
            )))?;
    } else {
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_bf16_vec)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16_vec {label}: {e}",
            )))?;
    }
    Ok(())
}

/// Wrapper for `cublasGemmEx` BF16: BF16 weight + BF16 activation -> F32 output.
///
/// Computes C = alpha * W^T * A + beta * C where W is BF16 [out_dim, in_dim],
/// A is BF16 [batch, in_dim], and C is F32 [batch, out_dim].
///
/// Uses `CUBLAS_COMPUTE_32F` with `CUDA_R_16BF` inputs. cuBLAS automatically
/// selects the tensor-core BF16 path on SM_80+ (A100 312 TFLOPS via mma.sync
/// bf16.bf16.f32). On older GPUs it falls back to software BF16; this path is
/// only invoked when the LBC header declares BF16 weights, which the runtime
/// already gates on Ampere+ capability via the matvec_bf16 SM_80 PTX path.
///
/// F32 accumulator preserves numerical equivalence with the per-row
/// matvec_bf16 fallback this replaces — both accumulate dot products in F32.
///
/// # Safety
///
/// `w_bf16` must be `out_dim * in_dim` BF16 elements (2 bytes each).
/// `a_bf16` must be `batch * in_dim` BF16 elements.
/// `output` must be `batch * out_dim` F32 elements.
unsafe fn launch_cublas_gemm_bf16(
    device: &CudaDevice,
    w_bf16: &CudaSlice<u8>,
    a_bf16: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    batch: usize,
    in_dim: usize,
    beta_val: f32,
    label: &str,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = beta_val;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_bf16.device_ptr(&device.stream);
    let (a_ptr, _) = a_bf16.device_ptr(&device.stream);
    let (c_ptr, _) = output.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,
        batch as i32,
        in_dim as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx BF16 GEMM {label}: status={status:?}",
        )));
    }
    Ok(())
}

