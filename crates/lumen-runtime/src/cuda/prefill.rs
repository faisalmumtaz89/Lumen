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
//! C_cm = W_cm^T * A_cm  =>  transa=T, transb=N
//!   m=out_dim, n=batch, k=in_dim
//!   A(cublas)=W, lda=in_dim, B(cublas)=A, ldb=in_dim, C(cublas)=C, ldc=out_dim

use cudarc::cublas::{Gemm, GemmConfig, sys as cublas_sys};
use cudarc::driver::{CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};

use crate::error::RuntimeError;

use super::decode::{
    KernelSet, attention_block_size, attention_shared_bytes,
    matvec_block_size, rmsnorm_block_size, rmsnorm_shared_bytes,
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
    })
}

/// Allocate prefill scratch buffers for the given batch size and model dimensions.
///
/// The `dequant_f32` buffer is sized to hold the largest projection weight matrix
/// in F32 format. This is max(q_dim * hidden_dim, kv_dim * hidden_dim,
/// inter_dim * hidden_dim, hidden_dim * inter_dim) = inter_dim * hidden_dim
/// for standard transformer architectures where inter_dim > q_dim.
pub(crate) fn alloc_prefill_scratch(
    device: &CudaDevice,
    batch: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
) -> Result<PrefillScratch, RuntimeError> {
    // Maximum weight matrix size across all projections.
    let max_weight_elems = [
        q_dim * hidden_dim,      // wq
        kv_dim * hidden_dim,     // wk, wv
        hidden_dim * q_dim,      // wo
        inter_dim * hidden_dim,  // w_gate, w_up
        hidden_dim * inter_dim,  // w_down
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

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
        // F16 activation: batch * max_in_dim * 2 bytes. max_in_dim = max(hidden_dim, inter_dim, q_dim).
        activation_f16: device.alloc_zeros(batch * [hidden_dim, inter_dim, q_dim].into_iter().max().unwrap_or(hidden_dim) * 2)?,
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
///   Row-major W[out_dim, in_dim] = col-major W_cm[in_dim, out_dim]
///   Row-major A[batch, in_dim]   = col-major A_cm[in_dim, batch]
///   Row-major C[batch, out_dim]  = col-major C_cm[out_dim, batch]
///   C_cm = W_cm^T * A_cm
///   cublasSgemm(T, N, out_dim, batch, in_dim, 1.0, W, in_dim, A, in_dim, 0.0, C, out_dim)
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

    // Fast path: HGEMM with pre-dequanted F16 weights (tensor core, 312 TFLOPS on A100).
    // Converts F32 activations to F16 on the fly, uses cublasGemmEx with F16 inputs
    // and F32 compute/accumulate for numerical stability.
    if let Some(w_f16) = weight_f16_cache {
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
            //   transa = T (transpose W_cm to get [out_dim, in_dim])
            //   transb = N (A_cm is already [in_dim, batch])
            //   m = out_dim, n = batch, k = in_dim
            //   lda = in_dim (leading dim of W_cm[in_dim, out_dim])
            //   ldb = in_dim (leading dim of A_cm[in_dim, batch])
            //   ldc = out_dim (leading dim of C_cm[out_dim, batch])
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
            // Dequantize Q8_0 weights to F32 scratch, then cuBLAS SGEMM.
            //
            // This replaces `batch` sequential matvec kernel launches with:
            //   1. One dequant_q8_0_to_f32 kernel (num_elements threads)
            //   2. One cuBLAS SGEMM call
            //
            // For pp128 on Llama 8B (out_dim=4096, in_dim=4096):
            //   Before: 128 matvec launches * 7 projections * 32 layers = 28,672 launches
            //   After:  2 launches * 7 projections * 32 layers = 448 launches
            let num_elements = out_dim * in_dim;
            if dequant_scratch.len() < num_elements {
                return Err(RuntimeError::Compute(format!(
                    "sgemm {label}: dequant scratch too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    dequant_scratch.len(),
                    num_elements,
                )));
            }

            // Step 1: Dequantize Q8_0 -> F32 in scratch buffer.
            launch_dequant_q8_0_to_f32(
                device, kernels, w_q8, dequant_scratch, num_elements, label,
            )?;

            // Step 2: cuBLAS SGEMM with the dequantized F32 weights.
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
                        "cuBLAS SGEMM (dequant Q8_0) {label}: {e}"
                    ))
                })?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            // Native F16 weights: cublasGemmEx HGEMM directly (no dequant needed).
            // Convert F32 activations to F16, then HGEMM with F16 weights.
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
            // Dequantize Q4_0 weights to F32 scratch, then cuBLAS SGEMM.
            //
            // Same pattern as Q8Raw: replaces `batch` sequential matvec launches with:
            //   1. One dequant_q4_0_to_f32 kernel (num_elements threads)
            //   2. One cuBLAS SGEMM call
            //
            // Critical for GDN layers where F16 caches are skipped to save GPU memory.
            let num_elements = out_dim * in_dim;
            if dequant_scratch.len() < num_elements {
                return Err(RuntimeError::Compute(format!(
                    "sgemm {label}: dequant scratch too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    dequant_scratch.len(),
                    num_elements,
                )));
            }

            // Step 1: Dequantize Q4_0 -> F32 in scratch buffer.
            launch_dequant_q4_0_to_f32(
                device, kernels, w_q4, dequant_scratch, num_elements, label,
            )?;

            // Step 2: cuBLAS SGEMM with the dequantized F32 weights.
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
                        "cuBLAS SGEMM (dequant Q4_0) {label}: {e}"
                    ))
                })?;
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

    // Fast path: HGEMM residual with pre-dequanted F16 weights.
    // Copy residual to output first, then HGEMM with beta=1.0.
    if let Some(w_f16) = weight_f16_cache {
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
            // Dequantize Q8_0 -> F32, then cuBLAS SGEMM with fused residual.
            let num_elements = out_dim * in_dim;
            if dequant_scratch.len() < num_elements {
                return Err(RuntimeError::Compute(format!(
                    "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    dequant_scratch.len(),
                    num_elements,
                )));
            }

            // Step 1: Dequantize Q8_0 -> F32 in scratch buffer.
            launch_dequant_q8_0_to_f32(
                device, kernels, w_q8, dequant_scratch, num_elements, label,
            )?;

            // Step 2: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dtod residual copy (dequant Q8_0) {label}: {e}"
                    ))
                })?;

            // Step 3: cuBLAS SGEMM with beta=1.0.
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
                        "cuBLAS SGEMM+residual (dequant Q8_0) {label}: {e}"
                    ))
                })?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            // Native F16 weights: cublasGemmEx HGEMM with residual (no dequant needed).
            // Copy residual -> output, convert F32 activation to F16, then HGEMM beta=1.0.

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
            // Dequantize Q4_0 -> F32, then cuBLAS SGEMM with fused residual.
            let num_elements = out_dim * in_dim;
            if dequant_scratch.len() < num_elements {
                return Err(RuntimeError::Compute(format!(
                    "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    dequant_scratch.len(),
                    num_elements,
                )));
            }

            // Step 1: Dequantize Q4_0 -> F32 in scratch buffer.
            launch_dequant_q4_0_to_f32(
                device, kernels, w_q4, dequant_scratch, num_elements, label,
            )?;

            // Step 2: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dtod residual copy (dequant Q4_0) {label}: {e}"
                    ))
                })?;

            // Step 3: cuBLAS SGEMM with beta=1.0.
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
                        "cuBLAS SGEMM+residual (dequant Q4_0) {label}: {e}"
                    ))
                })?;
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
) -> Result<(), RuntimeError> {
    let half_dim = head_dim / 2;
    let total_q_pairs = num_q_heads * half_dim;
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

    device
        .stream
        .launch_builder(&kernels.rope_apply_batched)
        .arg(&mut *q)
        .arg(&mut *k)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nqh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&theta)
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

    // Run attention_decode for this single token.
    let block_size = attention_block_size(seq_len);
    let shared_bytes = attention_shared_bytes(seq_len as u32);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let sl = seq_len as u32;
    let msl = kv_cache.max_seq_len as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.attention_decode)
        .arg(q_single as &CudaSlice<f32>)
        .arg(&kv_cache.k_cache)
        .arg(&kv_cache.v_cache)
        .arg(&mut *attn_out_single)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&sl)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("attention_decode prefill t={token_idx}: {e}"))
        })?;

    // Scatter-write attn_out_single back into the batch matrix at row token_idx.
    launch_scatter_row(
        device, kernels, attn_out_batch, &*attn_out_single, token_idx, q_dim,
    )?;

    Ok(())
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
    }
    Ok(())
}

/// Launch Flash Attention v2 (Br=1) for all tokens in a prefill batch.
///
/// Processes all query tokens against the KV cache in a SINGLE kernel launch
/// with causal masking. Each thread block handles one (head, token) pair.
///
/// Grid: (num_heads, batch, 1)  --  one block per (head, query_token)
/// Block: (128, 1, 1)           --  128 threads (4 warps)
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
/// Block: (128, 1, 1)  --  4 warps of 32 threads
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
/// Block: (128, 1, 1)  --  4 warps of 32 threads
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

/// HGEMM projection using pre-converted F16 activations (no F32->F16 conversion needed).
///
/// This is used when the activation is already in F16 format (e.g., from fused_rmsnorm_f16_batched).
/// Skips the F32->F16 conversion step entirely, saving one kernel dispatch per projection.
///
/// # Safety
///
/// `activation_f16` must already contain valid F16 data for [batch, in_dim].
/// `weight_f16` must be [out_dim, in_dim] F16. `output` must be [batch, out_dim] F32.
pub(crate) unsafe fn launch_hgemm_with_f16_input(
    device: &CudaDevice,
    weight_f16: &CudaSlice<u8>,
    activation_f16: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    launch_cublas_hgemm(
        device, weight_f16, activation_f16, output,
        out_dim, batch, in_dim, 0.0, label,
    )
}

/// Batched fused RMSNorm -> F16 output.
///
/// Replaces the two-dispatch sequence: rmsnorm_batched + f32_to_f16_vec.
/// Each threadblock handles one row of the [batch, dim] input matrix.
///
/// # Safety
///
/// `x` must be [batch, dim] F32. `weight` must be [dim] F32. `out_f16` must be [batch * dim * 2] bytes.
pub(crate) unsafe fn launch_fused_rmsnorm_f16_batched(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out_f16: &mut CudaSlice<u8>,
    eps: f32,
    batch: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let func = kernels.fused_rmsnorm_f16_batched.as_ref()
        .ok_or_else(|| RuntimeError::Compute(
            "fused_rmsnorm_f16_batched kernel not available".into()
        ))?;

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
        .launch_builder(func)
        .arg(x)
        .arg(weight)
        .arg(out_f16)
        .arg(&eps)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "fused_rmsnorm_f16_batched launch: {e}"
        )))?;
    Ok(())
}

/// Batched fused SwiGLU -> F16 output.
///
/// Replaces the two-dispatch sequence: swiglu_batched + f32_to_f16_vec.
/// Computes SwiGLU and writes both F32 (in-place to gate) and F16 (to out_f16).
///
/// # Safety
///
/// `gate` and `up` must be [batch * inter_dim] F32. `out_f16` must be [batch * inter_dim * 2] bytes.
pub(crate) unsafe fn launch_fused_swiglu_f16_batched(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out_f16: &mut CudaSlice<u8>,
    batch: usize,
    inter_dim: usize,
) -> Result<(), RuntimeError> {
    let func = kernels.swiglu_f32_to_f16_batched.as_ref()
        .ok_or_else(|| RuntimeError::Compute(
            "swiglu_f32_to_f16_batched kernel not available".into()
        ))?;

    let total = (batch * inter_dim) as u32;
    let block_size = 256u32;
    let grid_size = (total + block_size - 1) / block_size;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(func)
        .arg(gate)
        .arg(up)
        .arg(out_f16)
        .arg(&total)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "swiglu_f32_to_f16_batched launch: {e}"
        )))?;
    Ok(())
}
