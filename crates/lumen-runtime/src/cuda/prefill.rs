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

use cudarc::cublas::{sys as cublas_sys, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};

use crate::error::RuntimeError;

use super::decode::{
    attention_block_size, attention_decode_tiled_shared_bytes,
    attention_decode_tiled_supports_head_dim, attention_decode_variant, attention_shared_bytes,
    decode_tiled_force_enabled, decode_tiled_threshold, matvec_block_size, rmsnorm_block_size,
    rmsnorm_shared_bytes, AttentionDecodeVariant, KernelSet, ATTN_DECODE_TILED_BLOCK_DIM,
};
use super::ffi::CudaDevice;
use super::gpu_buffers::GpuWeightBuf;
use super::kv_cache::{KvRef, KvView};
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
    /// Score block of the tiled SGEMM prefill attention: `group ×
    /// min(batch, 512) × (pos_start + batch)` F32, reused across every KV
    /// head and every layer. Allocated once here, before the layer loop
    /// writes any KV, so an out-of-memory failure cannot land after the GPU
    /// KV caches have advanced past the host's. `None` when the path is off,
    /// its kernel did not load, or the allocation failed — the dispatcher
    /// then takes the scalar attention, which needs no score scratch.
    pub attn_scores: Option<CudaSlice<f32>>,
    /// Q+gate fusion temporaries: the fused projection `[batch, q_dim * 2]`
    /// and the deinterleaved gate `[batch, q_dim]`. Present when a layer fuses
    /// Q and gate.
    pub q_gate: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
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
    qgate_fused: bool,
) -> Result<PrefillScratch, RuntimeError> {
    // Maximum weight matrix size across all projections that share this scratch.
    //
    // Standard attention/FFN terms always apply. GDN terms are included when
    // the model declares GDN layers (`gdn_qkv_dim.is_some()`); otherwise the
    // GDN candidates evaluate to 0 and are dominated by the attention/FFN terms.
    let gdn_qkv = gdn_qkv_dim.unwrap_or(0);
    let gdn_value = gdn_value_dim.unwrap_or(0);
    let max_weight_elems = [
        q_dim * hidden_dim, // wq
        // wq with Q+gate fusion projects out = q_dim * 2 through this scratch.
        if qgate_fused {
            2 * q_dim * hidden_dim
        } else {
            0
        },
        kv_dim * hidden_dim,    // wk, wv
        hidden_dim * q_dim,     // wo
        inter_dim * hidden_dim, // w_gate, w_up
        hidden_dim * inter_dim, // w_down
        gdn_qkv * hidden_dim,   // GDN fused QKV projection (out=qkv_dim, in=hidden_dim)
        hidden_dim * gdn_value, // GDN SSM output projection (out=hidden_dim, in=value_dim)
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    // F16 activation buffer must cover the largest activation row across all
    // kernels: standard attention/FFN inputs, plus GDN value-side inputs when
    // present (e.g. ssm_out reads activations of width `value_dim`).
    let max_in_dim = [hidden_dim, inter_dim, q_dim, gdn_value]
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
        attn_scores: None,
        q_gate: if qgate_fused {
            Some((
                device.alloc_zeros(batch * q_dim * 2)?,
                device.alloc_zeros(batch * q_dim)?,
            ))
        } else {
            None
        },
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

/// The tiled prefill attention's score block, or `None` when the device
/// refuses it: the refusal is printed once and costs that route, not the
/// request, since the attention then runs on the scalar kernel. `None` in means a
/// geometry the route cannot host (see [`attn_score_block_elems`]).
pub(crate) fn alloc_attn_score_block(
    device: &CudaDevice,
    elems: Option<usize>,
) -> Option<CudaSlice<f32>> {
    let elems = elems.filter(|&n| n > 0)?;
    match device.alloc_zeros::<f32>(elems) {
        Ok(buf) => Some(buf),
        Err(e) => {
            warn_attn_scores_alloc_failed(elems, &e);
            None
        }
    }
}

/// Tokens per prefill slice. A prompt longer than this runs through the
/// layers in slices, each through one scratch sized for the slice, so the
/// per-token scratch is bounded by this and not by the prompt: on the 27B
/// dense model it measures about 0.55 MiB per token over a fixed ~0.6 GiB, so a
/// slice takes at most ~1.8 GiB, which fits beside the weights and a 16k-token
/// KV cache on a 32 GB card. The tiled attention's score block still grows with the prompt
/// (`group × min(slice, 512) × keys` F32: ~190 MiB at 16k keys on the 27B).
/// Prompts up to this length run as one slice, exactly as before.
pub(crate) const PREFILL_SLICE_TOKENS: usize = 2048;

/// Element count of the tiled prefill attention's score block for one
/// prefill: `group × min(batch, ATTN_PREFILL_SGEMM_ROWS) × (pos_start +
/// batch)` F32. The block is reused across every KV head and every layer, so
/// this is the path's whole footprint. `None` when the geometry cannot host
/// the path (no KV heads, or a query-head count that is not a multiple of
/// them -- `launch_flash_attention_sgemm` rejects both) or when the product
/// overflows `usize`.
pub(crate) fn attn_score_block_elems(
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    pos_start: usize,
) -> Option<usize> {
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return None;
    }
    let group = num_heads / num_kv_heads;
    let rows_max = batch.min(ATTN_PREFILL_SGEMM_ROWS);
    let kv_total = pos_start.checked_add(batch)?;
    group.checked_mul(rows_max)?.checked_mul(kv_total)
}

/// One line, once per process, when the score block cannot be allocated. The
/// prefill still runs -- the dispatcher falls back to the scalar attention --
/// so this is not an error, and a device that is short of memory must not
/// print it once per prefill.
fn warn_attn_scores_alloc_failed(elems: usize, err: &RuntimeError) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[CUDA] tiled prefill attention unavailable: its {:.1} MiB score block \
             ({elems} floats) could not be allocated ({err}); exact-F32 prefill \
             attention runs on the scalar kernel",
            (elems as f64 * 4.0) / (1024.0 * 1024.0)
        );
    });
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
    embedding_bf16: Option<&CudaSlice<u8>>,
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

    // Dispatch priority: BF16 > F16 > Q4_0 > Q8_0 > F32 (same order as
    // embed_token_gpu)
    if let Some(emb_bf16) = embedding_bf16 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_bf16)
            .arg(emb_bf16)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_bf16 launch: {e}")))?;
    } else if let Some(emb_f16) = embedding_f16 {
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

/// MoE BF16 fidelity routing: is this projection a native-BF16 weight on a MoE
/// model that should bypass the F16-cache fast path?
///
/// Empirical (2026-06-09, GQ-001 MoE-bf16 precision sweep): the F16-cache path
/// (`launch_cublas_hgemm`: F16 weights + `FAST_16F` reduced-precision
/// accumulation) makes the MoE BF16 model emit WRONG arithmetic — e.g. it reads
/// the input "1000 - 37" as "100 - 37" = 63, and degenerates into repetition —
/// whereas the native BF16 arm (`launch_cublas_gemm_bf16`: BF16 weights + full
/// `CUBLAS_COMPUTE_32F` accumulation) yields the correct answer (963). Routing
/// MoE BF16 to the native arm moves GQ-001 from 13/15 FAIL to 14/15 PASS (the
/// residual is a cosmetic near-tie intermediate slip inherent to the 7-bit BF16
/// mantissa; final answers are correct). AND-gated on `model_is_moe()` so DENSE
/// BF16 models keep the F16-cache fast path BYTE-IDENTICAL. The env override
/// `LUMEN_CUDA_MOE_BF16_NATIVE=0` restores the F16-cache path for MoE BF16 too.
fn moe_bf16_native_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("LUMEN_CUDA_MOE_BF16_NATIVE").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE")
        )
    })
}

fn moe_bf16_native_path(weight: &GpuWeightBuf) -> bool {
    matches!(weight, GpuWeightBuf::Bf16Raw(_))
        && crate::runtime_defaults::model_is_moe()
        && moe_bf16_native_enabled()
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
    let prefer_mmq_over_f16cache =
        q8_proj_mmq && weight_is_q8raw && in_dim % 32 == 0 && kernels.mmq_q8_0_batched.is_some();

    // `ssm_alpha` and `ssm_beta` weights are stored as F32 in the
    // GGUF source and an F32 SGEMM dispatch is the canonical reference path.
    // Default conversions force-requantize them to Q8_0 at LBC creation
    // time (see `crates/lumen-convert/src/arch/gdn_gates.rs`; source-fidelity,
    // HF-import, and `--dequantize` non-Metal artifacts keep F32), so the
    // default runtime path runs them through HGEMM-F16 / MMQ-Q8 — both of which introduce
    // ~0.4-3.4% per-element rounding noise from the requant step. When the
    // `gdn_alpha`/`gdn_beta` projection is routed through F32 SGEMM, it matches
    // that canonical F32 path (modulo SGEMM accumulator order).
    //
    // Default-ON for MoE BF16 (`model_is_moe_bf16()`): empirically this F32
    // alpha/beta path is what kills the BF16 GQ-001 arith-05 repetition that the
    // `moe_bf16_native` main-projection routing alone leaves — taking MoE bf16 to
    // 14/15 PASS. It is gated to BF16 models because the same lever REGRESSES MoE
    // q8 (adds a DD-REP). Env override: `LUMEN_CUDA_GDN_AB_F32=0` forces OFF, any
    // other value forces ON.
    let gdn_ab_f32_on = match std::env::var("LUMEN_CUDA_GDN_AB_F32").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") => false,
        Some(_) => true,
        None => crate::runtime_defaults::model_is_moe_bf16(),
    };
    let gdn_ab_f32 = gdn_ab_f32_on && (label == "gdn_alpha" || label == "gdn_beta");
    let force_alpha_beta_f32 = gdn_ab_f32 && weight_is_q8raw;

    // Fast path: HGEMM with pre-dequanted F16 weights (tensor core, 312 TFLOPS on A100).
    // Converts F32 activations to F16 on the fly, uses cublasGemmEx with F16 inputs
    // and F32 compute/accumulate for numerical stability.
    let f16_cache_active = !force_f32
        && !prefer_mmq_over_f16cache
        && !force_alpha_beta_f32
        && !moe_bf16_native_path(weight)
        && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        static HGEMM_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[CUDA] Prefill HGEMM: ACTIVE (tensor core path)");
        }

        // Step 1: Convert F32 activation to F16 via vectorized kernel (4 elems/thread).
        launch_f32_to_f16_fast(
            device,
            kernels,
            input,
            activation_f16,
            batch * in_dim,
            label,
        )?;

        // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
        launch_cublas_hgemm(
            device,
            w_f16,
            activation_f16,
            output,
            out_dim,
            batch,
            in_dim,
            0.0,
            label,
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
                .map_err(|e| RuntimeError::Compute(format!("cuBLAS SGEMM {label}: {e}")))?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // when LUMEN_CUDA_Q8_PROJ_MMQ=1, route Q8 projection
            // through the MMQ INT8 dp4a math (INT8xINT8->INT32->F32-scale).
            // This is the path required to close the qkv_pre_conv 5.85e-2
            // max-abs drift that HGEMM-F16 and SGEMM-F32 paths could not close.
            // Default OFF preserves byte-identical behaviour vs main; ENV-ON
            // routes to mmq_q8_0_batched (modulo dp4a/MMA microarchitecture
            // differences).
            // when LUMEN_CUDA_GDN_AB_F32=1, bypass MMQ
            // for `gdn_alpha` / `gdn_beta` projections specifically. This
            // forces the Q8 -> F32-dequant -> SGEMM-F32 path, restoring the
            // F32 SGEMM data path for these tensors (the GGUF source weight
            // is F32).
            // MoE models DEFAULT to the MMQ (INT8 dp4a) Q8 projection path.
            // llama.cpp computes Q8 matmuls in INT8 dp4a; lumen's default
            // dequant->F16->HGEMM path drifts ~5.85e-2 max-abs from that, which
            // is harmless for dense models but the 256-expert top-K router
            // AMPLIFIES it into flipped expert selection -> degenerate math on
            // Qwen3.5-MoE-35B-A3B (q8 writes "17x20=140" instead of 340). MMQ
            // restores llama-matching INT8 numerics so the products are correct.
            // Dense q8 keeps the faster HGEMM path (no router to amplify drift).
            // Env `LUMEN_CUDA_Q8_PROJ_MMQ=0|1` overrides the per-model default.
            let mmq_enabled = match std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").ok().as_deref() {
                Some(v) => !matches!(v, "0" | "false" | "no"),
                None => crate::runtime_defaults::model_is_moe(),
            };
            let use_mmq = mmq_enabled
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
                    device, kernels, w_q8, input, output, out_dim, in_dim, batch, label,
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
                    device,
                    kernels,
                    w_q8,
                    dequant_scratch,
                    num_elements,
                    label,
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
                    device,
                    kernels,
                    w_q8,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device,
                    dequant_f16,
                    activation_f16,
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    0.0,
                    label,
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
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
            launch_cublas_hgemm(
                device,
                w_f16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
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
                    device,
                    kernels,
                    w_q4,
                    dequant_scratch,
                    num_elements,
                    label,
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
                    device,
                    kernels,
                    w_q4,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device,
                    dequant_f16,
                    activation_f16,
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    0.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::Ct4Raw(w_ct4) => {
            // CtInt4G32 prefill: dequantize to F16 scratch, then HGEMM —
            // the same pattern as the Q4Raw arm above. No F32 fallback
            // exists for this format (its dequant kernel emits F16 only).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || dequant_f16.len() < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: CtInt4G32 requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    dequant_f16.len(),
                )));
            }
            launch_dequant_ct4_to_f16(device, kernels, w_ct4, dequant_f16, num_elements, label)?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                dequant_f16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill: fall back to per-row matvec
            // (aligned format is optimized for single-token decode, not GEMM).
            for row in 0..batch {
                let in_offset = row * in_dim;
                let out_offset = row * out_dim;
                launch_matvec_slice(
                    device, kernels, weight, input, output, in_offset, out_offset, out_dim, in_dim,
                    label,
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
            launch_f32_to_bf16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 2: cublasGemmEx (BF16 weight + BF16 activation -> F32 output).
            launch_cublas_gemm_bf16(
                device,
                w_bf16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        // split-layout: prefill never dispatches against Q8Split/Q4Split
        // siblings. These are decode-only reorganizations; the prefill path
        // operates on the original AoS Q8Raw/Q4Raw via dequant->F16->cuBLAS
        // HGEMM. If we somehow get here the caller has confused decode/prefill
        // dispatch.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "prefill GEMM {label}: Q8Split/Q4Split sibling \
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
    let q8_proj_mmq = std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").is_ok();
    let q8_residual_mmq = q8_proj_mmq;
    let weight_is_q8raw = matches!(weight, GpuWeightBuf::Q8Raw(_));
    let prefer_mmq_over_f16cache = q8_residual_mmq
        && weight_is_q8raw
        && in_dim % 32 == 0
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
    let f16_cache_active = !force_f32
        && !prefer_mmq_over_f16cache
        && !moe_bf16_native_path(weight)
        && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        // Copy residual -> output for beta=1.0 accumulation.
        device
            .stream
            .memcpy_dtod(residual, output)
            .map_err(|e| RuntimeError::Compute(format!("dtod residual copy {label}: {e}")))?;

        // Convert F32 activation to F16 via vectorized kernel.
        launch_f32_to_f16_fast(
            device,
            kernels,
            input,
            activation_f16,
            batch * in_dim,
            label,
        )?;

        // cublasGemmEx with beta=1.0 for residual accumulation.
        launch_cublas_hgemm(
            device,
            w_f16,
            activation_f16,
            output,
            out_dim,
            batch,
            in_dim,
            1.0,
            label,
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
            device.stream.memcpy_dtod(residual, output).map_err(|e| {
                RuntimeError::Compute(format!("dtod residual copy for {label}: {e}"))
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
            device.blas.gemm(cfg, w_f32, input, output).map_err(|e| {
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
            let use_mmq =
                q8_residual_mmq && kernels.mmq_q8_0_batched_residual.is_some() && in_dim % 32 == 0;
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
                    device, kernels, w_q8, input, residual, output, out_dim, in_dim, batch, label,
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
                    device,
                    kernels,
                    w_q8,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                device.stream.memcpy_dtod(residual, output).map_err(|e| {
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
                device.stream.memcpy_dtod(residual, output).map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy (dequant Q8_0) {label}: {e}"))
                })?;

                // Step 2: Dequantize Q8_0 -> F16 in scratch buffer.
                launch_dequant_q8_0_to_f16(
                    device,
                    kernels,
                    w_q8,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device,
                    dequant_f16,
                    activation_f16,
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    1.0,
                    label,
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
            device.stream.memcpy_dtod(residual, output).map_err(|e| {
                RuntimeError::Compute(format!("dtod residual copy F16Raw {label}: {e}"))
            })?;

            // Step 2: Convert F32 activation to F16 via vectorized kernel.
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 3: cublasGemmEx with beta=1.0 for residual accumulation.
            launch_cublas_hgemm(
                device,
                w_f16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
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
                    device,
                    kernels,
                    w_q4,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                device.stream.memcpy_dtod(residual, output).map_err(|e| {
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
                device.stream.memcpy_dtod(residual, output).map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy (dequant Q4_0) {label}: {e}"))
                })?;

                // Step 2: Dequantize Q4_0 -> F16 in scratch buffer.
                launch_dequant_q4_0_to_f16(
                    device,
                    kernels,
                    w_q4,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device,
                    dequant_f16,
                    activation_f16,
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    1.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::Ct4Raw(w_ct4) => {
            // CtInt4G32 prefill: dequantize to F16 scratch, then HGEMM —
            // the same pattern as the Q4Raw arm above. No F32 fallback
            // exists for this format (its dequant kernel emits F16 only).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || dequant_f16.len() < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: CtInt4G32 requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    dequant_f16.len(),
                )));
            }
            // beta=1.0 accumulates into `output`, so it must hold the residual first.
            device.stream.memcpy_dtod(residual, output).map_err(|e| {
                RuntimeError::Compute(format!("dtod residual copy (CtInt4G32) {label}: {e}"))
            })?;
            launch_dequant_ct4_to_f16(device, kernels, w_ct4, dequant_f16, num_elements, label)?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                dequant_f16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill residual: fall back to per-row matvec.
            for row in 0..batch {
                let in_offset = row * in_dim;
                let res_offset = row * out_dim;
                let out_offset = row * out_dim;
                launch_matvec_residual_slice(
                    device, kernels, weight, input, residual, output, in_offset, res_offset,
                    out_offset, out_dim, in_dim, label,
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
            device.stream.memcpy_dtod(residual, output).map_err(|e| {
                RuntimeError::Compute(format!("dtod residual copy Bf16Raw {label}: {e}"))
            })?;

            // Step 2: Convert F32 activation to BF16 via vectorized kernel.
            launch_f32_to_bf16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 3: cublasGemmEx BF16 with beta=1.0 for residual accumulation.
            launch_cublas_gemm_bf16(
                device,
                w_bf16,
                activation_f16,
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        // split-layout: prefill never dispatches against Q8Split/Q4Split
        // siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "prefill residual GEMM {label}: Q8Split/Q4Split \
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
    let actual_rot = if rotary_dim > 0 && (rotary_dim as usize) < head_dim {
        rotary_dim as usize
    } else {
        head_dim
    };
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
    kv: &KvView<'_>,
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
    let msl = kv.seq_stride as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    launch_attention_decode_gated(
        device,
        kernels,
        q_single as &CudaSlice<f32>,
        KvRef::F32 { k: kv.k, v: kv.v },
        None,
        &mut *attn_out_single,
        nh,
        nkvh,
        hd,
        sl,
        msl,
        scale,
    )
    .map_err(|e| RuntimeError::Compute(format!("attention_decode prefill t={token_idx}: {e}")))?;

    // Scatter-write attn_out_single back into the batch matrix at row token_idx.
    launch_scatter_row(
        device,
        kernels,
        attn_out_batch,
        &*attn_out_single,
        token_idx,
        q_dim,
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

/// Upper bound on the split-K decode-attention split count. Single source for
/// the scratch sizing (`GpuScratch::attn_splitk`); the partial-pass grid uses
/// [`attn_splitk_chunks`] for the token's actual context.
pub const ATTN_SPLITK_S_MAX: u32 = 32;
const _: () = assert!(crate::runtime_defaults::ATTN_SPLITK_FIXED_CHUNKS <= ATTN_SPLITK_S_MAX);

/// The split count for a decode step over `seq_len` KV positions: one chunk
/// per [`crate::runtime_defaults::ATTN_SPLITK_CHUNK_POSITIONS`] positions
/// (`LUMEN_CUDA_ATTN_SPLITK_CHUNK` overrides), at least 1, at most
/// [`ATTN_SPLITK_S_MAX`]; or the fixed
/// [`crate::runtime_defaults::ATTN_SPLITK_FIXED_CHUNKS`] at every context
/// when `LUMEN_CUDA_ATTN_SPLITK_SCALE=0`. The target picks the count, not
/// the span: the kernel divides `seq_len` evenly into the count, so a chunk
/// walks `seq_len` divided by it — 65 and 64 at a context of 129, 119 at
/// 1300, and 384 at 12280, where the cap binds; past the cap the count is
/// pinned and each chunk's walk grows a tile at every further multiple, the
/// price of bounded scratch (the tiled route's one CTA walks all of it). A
/// fixed count starved a long context (24 query heads × 4 chunks on a
/// 170-SM card); scaling with the context keeps every chunk's serial walk
/// bounded up to the cap. A count of 1 means the caller takes the tiled kernel when it
/// loaded: one chunk plus a merge is the tiled walk with an extra launch.
pub fn attn_splitk_chunks(seq_len: u32) -> u32 {
    if !crate::runtime_defaults::attn_splitk_scale_with_context() {
        return crate::runtime_defaults::ATTN_SPLITK_FIXED_CHUNKS;
    }
    attn_splitk_chunk_count(
        seq_len,
        crate::runtime_defaults::attn_splitk_chunk_positions(),
    )
}

/// Pure arithmetic behind [`attn_splitk_chunks`] (separated for unit
/// testing), with the target positions per chunk resolved by the caller.
fn attn_splitk_chunk_count(seq_len: u32, chunk_positions: u32) -> u32 {
    seq_len
        .div_ceil(chunk_positions)
        .clamp(1, ATTN_SPLITK_S_MAX)
}

/// Split-K shape eligibility: the accumulator slots cover `head_dim` exactly
/// (same 128-lane slot addressing as the tiled kernel, 8 slots max).
fn attention_decode_splitk_supports_head_dim(head_dim: u32) -> bool {
    head_dim % ATTN_DECODE_TILED_BLOCK_DIM == 0 && head_dim <= 1024
}

// ---------------------------------------------------------------------------
// GQA-shared split-K pair (`LUMEN_CUDA_ATTN_SPLITK_GQA6`)
// ---------------------------------------------------------------------------

/// Query heads per KV head the GQA-shared partial pass is specialised for:
/// it forms all six of a group's scores from one register-resident K row.
pub const ATTN_SPLITK_GQA6_GQA_RATIO: u32 = 6;

/// Head dimension the GQA-shared partial pass is specialised for: 256 is two
/// `float4` per lane across a 32-lane warp, both in the QK phase and in the
/// two dimensions each of the 128 threads owns in the PV phase.
pub const ATTN_SPLITK_GQA6_HEAD_DIM: u32 = 256;

/// KV positions per chunk of the GQA-shared partial pass. The whole chunk's
/// scores for one query head live in one warp's lanes, so a chunk cannot
/// exceed 32; 16 is what the geometry measured fastest at, and it is what
/// puts enough CTAs on the card (4 KV heads x 69 chunks = 276 at a context of
/// 1,100, against 170 SMs).
pub const ATTN_SPLITK_GQA6_CHUNK: u32 = 16;

/// Upper bound on the GQA-shared split count, and with it the scratch: the
/// partial pass writes `num_heads * S * head_dim` floats (24 MiB at this bound
/// for a 24-head, 256-dimension model, 24.2 MiB with each chunk's running max
/// and sum), covering 16,384 KV positions at the
/// shipped chunk length. `LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS` lowers the
/// bound a process serves ([`attn_splitk_gqa6_chunk_bound`]); nothing raises
/// it past this constant. Through v0.30.0 the bound was 256 (4,096 positions),
/// past which a running generation fell back to the per-query-head pair,
/// which reads every K and V row once per query head: at 6,144 keys that
/// fallback cost 1.6 ms of a 13.1 ms decode token on the RTX 5090.
pub const ATTN_SPLITK_GQA6_S_MAX: u32 = 1024;

/// CTAs per query head in the GQA-shared merge: each owns one block's worth
/// of the head's dimensions, one dimension per thread. Splitting the head
/// this way halves each CTA's accumulation and doubles the grid, which one
/// CTA per head leaves at 24 on a 170-SM card; the price is evaluating the S
/// rescale factors once per tile instead of once per head. The merge measures
/// 4.1 µs at a context of 1,100.
pub const ATTN_SPLITK_GQA6_DIM_TILES: u32 = ATTN_SPLITK_GQA6_HEAD_DIM / ATTN_DECODE_TILED_BLOCK_DIM;

// The kernel's safety rests on these three, and each would fail silently:
// a chunk longer than a warp leaves the softmax normalising only the first
// 32 positions while the PV loop folds the rest in un-normalised; tiles that
// do not cover the head exactly either drop dimensions or write past the
// head into its neighbour's output row; and shared memory over the default
// cap needs an opt-in the launcher does not perform.
const _: () = assert!(ATTN_SPLITK_GQA6_CHUNK <= 32);
const _: () =
    assert!(ATTN_SPLITK_GQA6_S_MAX == crate::runtime_defaults::ATTN_SPLITK_GQA6_MAX_CHUNKS_DEFAULT);
const _: () =
    assert!(ATTN_SPLITK_GQA6_DIM_TILES * ATTN_DECODE_TILED_BLOCK_DIM == ATTN_SPLITK_GQA6_HEAD_DIM);
const _: () = assert!(attn_splitk_gqa6_partial_shared_bytes() <= 49152);
const _: () = assert!(attn_splitk_gqa6_merge_shared_bytes(ATTN_SPLITK_GQA6_S_MAX) <= 49152);

/// The longest context the GQA-shared pair can serve at the compile-time
/// bound. Past it the chunk count would exceed the scratch bound and a chunk
/// would outgrow the 32 lanes its softmax runs on, so the caller keeps the
/// per-query-head pair instead — nothing is truncated.
pub const fn attn_splitk_gqa6_max_seq_len() -> u32 {
    ATTN_SPLITK_GQA6_S_MAX * ATTN_SPLITK_GQA6_CHUNK
}

/// The split-count bound this process serves: `LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS`
/// clamped to `1..=ATTN_SPLITK_GQA6_S_MAX`, the constant when unset or unparsable.
/// The eligibility test and the scratch allocator read the same bound, so a
/// lowered bound shrinks the scratch and hands longer contexts to the
/// per-query-head pair exactly where the dispatcher stops.
pub fn attn_splitk_gqa6_chunk_bound() -> u32 {
    crate::runtime_defaults::attn_splitk_gqa6_max_chunks().clamp(1, ATTN_SPLITK_GQA6_S_MAX)
}

/// The longest context the GQA-shared pair serves in this process:
/// [`attn_splitk_gqa6_chunk_bound`] chunks of [`ATTN_SPLITK_GQA6_CHUNK`].
pub fn attn_splitk_gqa6_served_seq_len() -> u32 {
    attn_splitk_gqa6_chunk_bound() * ATTN_SPLITK_GQA6_CHUNK
}

/// The GQA-shared split count for `seq_len`: one chunk per
/// [`ATTN_SPLITK_GQA6_CHUNK`] positions. The kernels then divide `seq_len`
/// evenly into that count, and `ceil(n / ceil(n / c)) <= c` guarantees the
/// span they walk never exceeds the chunk length.
pub fn attn_splitk_gqa6_chunks(seq_len: u32) -> u32 {
    seq_len.div_ceil(ATTN_SPLITK_GQA6_CHUNK).max(1)
}

/// Shape eligibility for the GQA-shared pair: any model whose query heads
/// come in groups of [`ATTN_SPLITK_GQA6_GQA_RATIO`] per KV head at
/// [`ATTN_SPLITK_GQA6_HEAD_DIM`] dimensions, up to
/// [`attn_splitk_gqa6_max_seq_len`] KV positions. The kernels index by group,
/// not by an absolute head count, so 24 query heads over 4 KV heads and 12
/// over 2 are the same work per CTA and differ only in grid height.
pub fn attention_decode_splitk_gqa6_supports(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> bool {
    attention_decode_splitk_gqa6_supports_within(
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        attn_splitk_gqa6_chunk_bound(),
    )
}

/// [`attention_decode_splitk_gqa6_supports`] with the chunk bound explicit: a
/// pure function of its arguments, so a test can state the bound it means
/// without touching the process environment.
pub const fn attention_decode_splitk_gqa6_supports_within(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    chunk_bound: u32,
) -> bool {
    num_kv_heads != 0
        && num_heads == num_kv_heads * ATTN_SPLITK_GQA6_GQA_RATIO
        && head_dim == ATTN_SPLITK_GQA6_HEAD_DIM
        && seq_len >= 1
        && seq_len <= chunk_bound * ATTN_SPLITK_GQA6_CHUNK
}

/// Dynamic shared bytes for the GQA-shared partial pass: the six staged Q
/// rows, the staged V tile, the `[6][C]` score block, and the per-head
/// (m, l) pair. 22'960 B at the shipped chunk length, inside the 48 KiB
/// default cap, so no shared-memory opt-in is needed.
pub const fn attn_splitk_gqa6_partial_shared_bytes() -> u32 {
    (ATTN_SPLITK_GQA6_GQA_RATIO * ATTN_SPLITK_GQA6_HEAD_DIM
        + ATTN_SPLITK_GQA6_CHUNK * ATTN_SPLITK_GQA6_HEAD_DIM
        + ATTN_SPLITK_GQA6_GQA_RATIO * ATTN_SPLITK_GQA6_CHUNK
        + 12)
        * 4
}

/// Dynamic shared bytes for the GQA-shared merge: one rescale factor per
/// chunk plus the four-warp reduction scratch.
pub const fn attn_splitk_gqa6_merge_shared_bytes(chunks: u32) -> u32 {
    (chunks + 4) * 4
}

/// Launch the split-K decode-attention pair (`LUMEN_CUDA_ATTN_SPLITK`):
/// sequence-parallel partial pass on a `num_heads * S` grid, then a
/// `num_heads`-CTA merge. Shares the tiled kernel's input/output buffer
/// contract but additionally requires `head_dim <= 1024` (enforced by
/// `attention_decode_splitk_supports_head_dim`); `scratch` holds the
/// per-chunk (m, l, o) triples sized for `ATTN_SPLITK_S_MAX`.
///
/// # Safety
///
/// Same input/output buffer contract as `launch_attention_decode_tiled`,
/// plus correctly sized split-K scratch and `head_dim <= 1024`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_splitk(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    let s: u32 = attn_splitk_chunks(seq_len);
    if !attention_decode_splitk_supports_head_dim(head_dim) {
        return Err(RuntimeError::Compute(format!(
            "attention_decode_splitk: unsupported head_dim ({head_dim}); \
             requires head_dim % {ATTN_DECODE_TILED_BLOCK_DIM} == 0 and <= 1024"
        )));
    }
    let (partial_fn, merge_fn) = match (
        kernels.attention_decode_splitk_partial.as_ref(),
        kernels.attention_decode_splitk_merge.as_ref(),
    ) {
        (Some(p), Some(m)) => (p, m),
        _ => {
            return Err(RuntimeError::Compute(
                "attention_decode_splitk: kernels not available".into(),
            ))
        }
    };
    let shared_bytes = attention_decode_tiled_shared_bytes(head_dim);
    let (m_part, l_part, o_part) = scratch;
    device
        .stream
        .launch_builder(partial_fn)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut *m_part)
        .arg(&mut *l_part)
        .arg(&mut *o_part)
        .arg(&num_heads)
        .arg(&num_kv_heads)
        .arg(&head_dim)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads * s, 1, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: shared_bytes,
        })
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_splitk_partial: {e}")))?;
    device
        .stream
        .launch_builder(merge_fn)
        .arg(&*m_part)
        .arg(&*l_part)
        .arg(&*o_part)
        .arg(attn_out)
        .arg(&num_heads)
        .arg(&head_dim)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_splitk_merge: {e}")))?;
    Ok(())
}

/// Launch the GQA-shared split-K decode-attention pair
/// (`LUMEN_CUDA_ATTN_SPLITK_GQA6`): a partial pass on a
/// `(S, num_kv_heads)` grid, then a `(num_heads, dim_tiles)` merge. Shares
/// the sibling pair's buffer contract and its `(m, l, o)` scratch, but with
/// the larger split count [`attn_splitk_gqa6_chunks`] returns — the caller
/// must have sized the scratch for it.
///
/// # Safety
///
/// Same buffer / shape constraints as the underlying kernels. The shape must
/// already satisfy [`attention_decode_splitk_gqa6_supports`] and the scratch
/// must hold `num_heads * S * head_dim` floats; the caller checks both.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_splitk_gqa6(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    let (partial_fn, merge_fn) = match (
        kernels.attention_decode_splitk_partial_gqa6.as_ref(),
        kernels.attention_decode_splitk_merge_gqa6.as_ref(),
    ) {
        (Some(p), Some(m)) => (p, m),
        _ => {
            return Err(RuntimeError::Compute(
                "attention_decode_splitk_gqa6: kernels not available".into(),
            ))
        }
    };
    let s: u32 = attn_splitk_gqa6_chunks(seq_len);
    let chunk = ATTN_SPLITK_GQA6_CHUNK;
    let (m_part, l_part, o_part) = scratch;
    device
        .stream
        .launch_builder(partial_fn)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut *m_part)
        .arg(&mut *l_part)
        .arg(&mut *o_part)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&s)
        .arg(&chunk)
        .launch(CudarcLaunchConfig {
            grid_dim: (s, num_kv_heads, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: attn_splitk_gqa6_partial_shared_bytes(),
        })
        .map_err(|e| {
            RuntimeError::Compute(format!("attention_decode_splitk_partial_gqa6_f32: {e}"))
        })?;
    device
        .stream
        .launch_builder(merge_fn)
        .arg(&*m_part)
        .arg(&*l_part)
        .arg(&*o_part)
        .arg(attn_out)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads, ATTN_SPLITK_GQA6_DIM_TILES, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: attn_splitk_gqa6_merge_shared_bytes(s),
        })
        .map_err(|e| {
            RuntimeError::Compute(format!("attention_decode_splitk_merge_gqa6_f32: {e}"))
        })?;
    Ok(())
}

/// Name the split-K decode-attention pair on its first dispatch.
///
/// The chunk count moves with the context, so the line reports the count of
/// the dispatch that emitted it plus the policy that produced it — `scaled`
/// (count grows with `seq_len`, capped) or `fixed`
/// (`LUMEN_CUDA_ATTN_SPLITK_SCALE=0`).
fn announce_splitk_route(head_dim: u32, seq_len: u32) {
    static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    super::decode::announce_route_once(&SEEN, || {
        let policy = if crate::runtime_defaults::attn_splitk_scale_with_context() {
            "scaled"
        } else {
            "fixed"
        };
        let chunks = attn_splitk_chunks(seq_len);
        format!(
            "[CUDA] attention_decode_splitk_partial: ACTIVE (chunks={chunks} {policy}, \
             head_dim={head_dim}, merge=attention_decode_splitk_merge)"
        )
    });
}

/// Name the GQA-shared split-K pair on its first dispatch.
///
/// The chunk length is fixed, so the line reports the count the dispatch that
/// emitted it derived from its context, alongside the geometry the pair is
/// specialised for.
fn announce_splitk_gqa6_route(num_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32) {
    static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    super::decode::announce_route_once(&SEEN, || {
        let chunks = attn_splitk_gqa6_chunks(seq_len);
        format!(
            "[CUDA] attention_decode_splitk_partial_gqa6_f32: ACTIVE (kv=f32, \
             q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, \
             seq_len={seq_len}, chunks={chunks}, chunk={chunk}, block={block}, \
             merge=attention_decode_splitk_merge_gqa6_f32)",
            chunk = ATTN_SPLITK_GQA6_CHUNK,
            block = ATTN_DECODE_TILED_BLOCK_DIM,
        )
    });
}

/// Why a loaded GQA-shared pair will not serve a dispatch.
///
/// Each variant is a branch of [`launch_attention_decode_gated`] that sends
/// the token somewhere else, and each gets its own announcement latch, so an
/// operator running one process through several shapes hears every distinct
/// reason rather than only the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplitKGqa6Exclusion {
    ForcedTiled,
    SingleBlockThreshold,
    NoScratch,
    ShippingPairAbsent,
    HeadDim,
    OneChunkContext,
    Shape,
    ScratchTooSmall,
}

impl SplitKGqa6Exclusion {
    /// Every variant, for the tests to walk. Maintained by hand: the assert
    /// below catches a list that has fallen behind the enum's last variant,
    /// and the exhaustive match in `latch` refuses a variant with no latch;
    /// a variant added with its arms but left out of this list is not caught.
    const ALL: [Self; 8] = [
        Self::ForcedTiled,
        Self::SingleBlockThreshold,
        Self::NoScratch,
        Self::ShippingPairAbsent,
        Self::HeadDim,
        Self::OneChunkContext,
        Self::Shape,
        Self::ScratchTooSmall,
    ];

    /// This reason's own announcement latch.
    ///
    /// An exhaustive match rather than an index into an array: a variant
    /// added to the enum then fails to compile here, where an index would
    /// have compiled and panicked out of bounds the first time the new
    /// reason was announced — inside a diagnostic, under a flag, which is
    /// the worst place to learn about it.
    fn latch(self) -> &'static std::sync::OnceLock<()> {
        static FORCED_TILED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static SINGLE_BLOCK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static NO_SCRATCH: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static PAIR_ABSENT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static HEAD_DIM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static ONE_CHUNK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static SHAPE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        static SCRATCH_TOO_SMALL: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        match self {
            Self::ForcedTiled => &FORCED_TILED,
            Self::SingleBlockThreshold => &SINGLE_BLOCK,
            Self::NoScratch => &NO_SCRATCH,
            Self::ShippingPairAbsent => &PAIR_ABSENT,
            Self::HeadDim => &HEAD_DIM,
            Self::OneChunkContext => &ONE_CHUNK,
            Self::Shape => &SHAPE,
            Self::ScratchTooSmall => &SCRATCH_TOO_SMALL,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::ForcedTiled => "LUMEN_CUDA_DECODE_TILED forces the tiled kernel",
            Self::SingleBlockThreshold => {
                "the decode-attention threshold selected the single-block kernel"
            }
            Self::NoScratch => "this dispatch site supplies no split-K scratch",
            Self::ShippingPairAbsent => "the split-K pair this route rides did not load",
            Self::HeadDim => "head_dim is outside the split-K route's range",
            Self::OneChunkContext => "a one-chunk context hands off to the tiled kernel",
            Self::Shape => {
                "the pair serves 6 query heads per KV head at head_dim 256, \
                 up to its chunk bound of KV positions (16384 at the shipped \
                 bound; LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS lowers it)"
            }
            Self::ScratchTooSmall => {
                "the split-K scratch is sized for the per-query-head split count"
            }
        }
    }
}

// Catches an `ALL` that has fallen behind the enum's last variant. It cannot
// catch a variant appended after that one — `latch`'s exhaustive match is what
// closes that case.
const _: () =
    assert!(SplitKGqa6Exclusion::ALL.len() == SplitKGqa6Exclusion::ScratchTooSmall as usize + 1);

/// Which exclusion applies to this dispatch, or `None` when the GQA-shared
/// pair serves it.
///
/// The arms are in the order [`launch_attention_decode_gated`] takes its
/// decisions, so one call before any route is chosen covers every way out —
/// including the ones that never enter the split-K block.
///
/// PURE function of its inputs — the caller resolves `shipping_chunks` (which
/// reads the environment) and passes it in.
#[allow(clippy::too_many_arguments)]
fn splitk_gqa6_exclusion_reason(
    force_tiled: bool,
    variant: AttentionDecodeVariant,
    scratch_o_floats: Option<usize>,
    splitk_pair_loaded: bool,
    tiled_loaded: bool,
    shipping_chunks: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    chunk_bound: u32,
) -> Option<SplitKGqa6Exclusion> {
    use SplitKGqa6Exclusion as X;
    if force_tiled {
        return Some(X::ForcedTiled);
    }
    if variant != AttentionDecodeVariant::Tiled {
        return Some(X::SingleBlockThreshold);
    }
    let Some(o_floats) = scratch_o_floats else {
        return Some(X::NoScratch);
    };
    if !splitk_pair_loaded {
        return Some(X::ShippingPairAbsent);
    }
    if !attention_decode_splitk_supports_head_dim(head_dim) {
        return Some(X::HeadDim);
    }
    if shipping_chunks <= 1 && tiled_loaded {
        return Some(X::OneChunkContext);
    }
    if !attention_decode_splitk_gqa6_supports_within(
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        chunk_bound,
    ) {
        return Some(X::Shape);
    }
    let needed =
        (num_heads as usize) * (attn_splitk_gqa6_chunks(seq_len) as usize) * (head_dim as usize);
    if o_floats < needed {
        return Some(X::ScratchTooSmall);
    }
    None
}

/// Say once per reason why an eligible-looking dispatch declined the
/// GQA-shared pair after the loader built it (by default on the measured
/// cell, or under `LUMEN_CUDA_ATTN_SPLITK_GQA6=1`).
///
/// Short contexts and every geometry the specialised kernels do not serve
/// are expected exclusions, not failures — but an operator who expects the
/// pair and sees no ACTIVE line needs the reason without reading the
/// dispatcher.
///
/// The line carries no kernel symbol at all — not the route that declined and
/// not the one that serves the token instead. A census harvests route names by
/// substring, so either would let a declined dispatch be counted as one that
/// happened. Which route serves the token depends on the reason: the tiled or
/// single-block kernel when the split-K route is not entered at all, the
/// per-query-head pair when it is and only this pass is declined; whichever it
/// is announces itself on its own line.
fn announce_splitk_gqa6_excluded(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    reason: SplitKGqa6Exclusion,
) {
    // One latch per reason: a process that meets several shapes reports each
    // of them once, not just whichever came first.
    let why = reason.message();
    super::decode::announce_route_once(reason.latch(), || {
        format!(
            "[CUDA] split-K GQA-shared pair: EXCLUDED ({why}; \
             q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, \
             seq_len={seq_len})"
        )
    });
}

/// Gate-and-dispatch the appropriate decode-attention kernel for `seq_len`.
///
/// Single source of truth for the kernel selection logic, used at the
/// launch sites:
///   1. `backend_impl.rs::compute_layer_gpu` — primary decode entry
///      (supplies split-K scratch)
///   2. `backend_impl.rs::compute_layer` per-layer body
///      (supplies split-K scratch)
///   3. `prefill.rs::launch_attention_for_token` per-token prefill fallback
///   4. `prefill_attention.rs::prefill_attention_sequential` per-token prefill
///
/// Returns the variant chosen (informational — current callers discard it;
/// the selector's own gate is the single decision point).
///
/// # Safety
///
/// Same buffer / shape constraints as the underlying kernels. The wrapper
/// does not re-check them — it forwards directly to the chosen launcher.
pub(crate) unsafe fn launch_attention_decode_gated(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    kv: KvRef<'_>,
    splitk_scratch: Option<&mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<AttentionDecodeVariant, RuntimeError> {
    // The store's type picks the router: each router's kernels take exactly
    // the bytes that store holds, so a half store can never reach an F32
    // kernel, nor the reverse.
    let variant = match kv {
        KvRef::F32 { k, v } => launch_attention_decode_routed(
            device,
            kernels,
            q,
            k,
            v,
            splitk_scratch,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            max_seq_len,
            scale,
        )?,
        KvRef::F16 { k, v } => launch_attention_decode_routed_f16(
            device,
            kernels,
            q,
            k,
            v,
            splitk_scratch,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            max_seq_len,
            scale,
        )?,
    };
    if let Some((dir, lengths)) = attention_dump_config() {
        if lengths.contains(&seq_len) {
            dump_attention_call(
                device,
                dir,
                q,
                kv,
                attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                scale,
                variant,
            )?;
        }
    }
    Ok(variant)
}

/// `LUMEN_CUDA_ATTN_DUMP=<dir>:<seq_len>[,<seq_len>...]`, parsed once: the
/// directory the decode-attention dump writes into and the sequence lengths
/// it writes at. `None` when unset or malformed (a malformed value is said
/// once and ignored, so a typo can never stall a decode).
fn attention_dump_config() -> Option<&'static (std::path::PathBuf, Vec<u32>)> {
    static CONFIG: std::sync::OnceLock<Option<(std::path::PathBuf, Vec<u32>)>> =
        std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let raw = std::env::var("LUMEN_CUDA_ATTN_DUMP").ok()?;
            let parsed = raw.rsplit_once(':').and_then(|(dir, lengths)| {
                let lengths: Vec<u32> = lengths
                    .split(',')
                    .map(|n| n.trim().parse::<u32>().ok())
                    .collect::<Option<_>>()?;
                (!dir.is_empty() && !lengths.is_empty())
                    .then(|| (std::path::PathBuf::from(dir), lengths))
            });
            if parsed.is_none() {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_ATTN_DUMP={raw:?}: want <dir>:<seq_len>[,<seq_len>...]; ignored"
                );
            }
            parsed
        })
        .as_ref()
}

/// Write one decode-attention call to `dir`: `attn-<seq_len>-<call>.json`
/// (shape, scale, the route that served) beside the raw little-endian F32
/// files `.q.f32` (`[num_heads, head_dim]`), `.k.f32` and `.v.f32` (the live
/// `[num_kv_heads, seq_len, head_dim]` region of the cache) and `.out.f32`
/// (the route's output, `[num_heads, head_dim]`). The call counter runs over
/// the process, so the attention layers of one token appear in order. A
/// diagnostic for replaying real activations through a reference; it copies
/// the cache to the host and is not for a measured run.
#[allow(clippy::too_many_arguments)]
fn dump_attention_call(
    device: &CudaDevice,
    dir: &std::path::Path,
    q: &CudaSlice<f32>,
    kv: KvRef<'_>,
    attn_out: &CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
    variant: AttentionDecodeVariant,
) -> Result<(), RuntimeError> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CALL: AtomicU32 = AtomicU32::new(0);
    let call = CALL.fetch_add(1, Ordering::Relaxed);
    let io = |e: std::io::Error| RuntimeError::Compute(format!("attention dump: {e}"));
    std::fs::create_dir_all(dir).map_err(io)?;
    let stem = dir.join(format!("attn-{seq_len}-{call:04}"));
    let (nh, nkv, hd, sl, msl) = (
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        seq_len as usize,
        max_seq_len as usize,
    );
    let write_f32 = |suffix: &str, data: &[f32]| -> Result<(), RuntimeError> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(stem.with_extension(suffix), bytes).map_err(io)
    };
    let q_host: Vec<f32> = device.dtoh_copy_view(&q.slice(0..nh * hd))?;
    write_f32("q.f32", &q_host)?;
    // The cache region as stored: F32 words, or the half bit patterns the
    // kernel read (`.k.f16` / `.v.f16`, 16-bit little-endian). A replay of a
    // half dump must widen exactly; the storage rounding already happened.
    let kv_dtype = match kv {
        KvRef::F32 { k, v } => {
            for (name, cache) in [("k.f32", k), ("v.f32", v)] {
                let mut host = Vec::with_capacity(nkv * sl * hd);
                for kv_h in 0..nkv {
                    let base = kv_h * msl * hd;
                    let region: Vec<f32> =
                        device.dtoh_copy_view(&cache.slice(base..base + sl * hd))?;
                    host.extend_from_slice(&region);
                }
                write_f32(name, &host)?;
            }
            "f32"
        }
        KvRef::F16 { k, v } => {
            for (name, cache) in [("k.f16", k), ("v.f16", v)] {
                let mut bytes = Vec::with_capacity(nkv * sl * hd * 2);
                for kv_h in 0..nkv {
                    let base = kv_h * msl * hd;
                    let region: Vec<u16> =
                        device.dtoh_copy_view(&cache.slice(base..base + sl * hd))?;
                    for x in &region {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                std::fs::write(stem.with_extension(name), bytes).map_err(io)?;
            }
            "f16"
        }
    };
    let out_host: Vec<f32> = device.dtoh_copy_view(&attn_out.slice(0..nh * hd))?;
    write_f32("out.f32", &out_host)?;
    let route = match variant {
        AttentionDecodeVariant::SingleBlock => "attention_decode",
        AttentionDecodeVariant::Tiled => "attention_decode_tiled",
        AttentionDecodeVariant::SplitK => "attention_decode_splitk_partial",
        AttentionDecodeVariant::SplitKGqa6 => "attention_decode_splitk_partial_gqa6_f32",
        AttentionDecodeVariant::TiledF16 => "attention_decode_tiled_f16",
        AttentionDecodeVariant::SplitKGqa6F16 => "attention_decode_splitk_partial_gqa6_f16",
        AttentionDecodeVariant::SplitKF16 => "attention_decode_splitk_partial_f16",
    };
    let engine = crate::runtime_defaults::build_identity();
    let meta = format!(
        "{{\n \"format\": \"lumen-attn-dump@1\",\n \"engine\": \"{engine}\",\n \"kv_dtype\": \"{kv_dtype}\",\n \"call\": {call},\n \"route\": \"{route}\",\n \"num_heads\": {num_heads},\n \"num_kv_heads\": {num_kv_heads},\n \"head_dim\": {head_dim},\n \"seq_len\": {seq_len},\n \"max_seq_len\": {max_seq_len},\n \"scale\": {scale:e}\n}}\n"
    );
    std::fs::write(stem.with_extension("json"), meta).map_err(io)
}

/// The split-K choice the decode routers share: one function decides, from
/// the selector's variant, the knobs, the loaded F32 kernels and the shape,
/// whether this dispatch takes the GQA-shared pair, the per-query-head pair
/// or neither — and both routers launch that choice with their own store's
/// kernels. One decision, so the two stores can never diverge on availability
/// or shape (a half store's twins load as a group, but the F32 pairs they
/// merge with are optional and are checked here for both).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SplitKChoice {
    Gqa6,
    PerHead,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_attention_splitk_choice(
    kernels: &KernelSet,
    variant: AttentionDecodeVariant,
    force_tiled: bool,
    scratch_o_floats: Option<usize>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Option<SplitKChoice> {
    // Split-K upgrade: applies only to the AUTO Tiled selection — an explicit
    // `LUMEN_CUDA_DECODE_TILED=1` still forces the tiled kernel, and a
    // SingleBlock selection (including the threshold opt-out) is untouched.
    // Requires caller-supplied scratch, both kernels, and an eligible shape;
    // anything else falls through to the existing selection.
    if variant != AttentionDecodeVariant::Tiled || force_tiled {
        return None;
    }
    let scratch_o_floats = scratch_o_floats?;
    if !(kernels.attention_decode_splitk_partial.is_some()
        && kernels.attention_decode_splitk_merge.is_some()
        && attention_decode_splitk_supports_head_dim(head_dim)
        // One chunk plus a merge is the tiled walk with an extra launch, so a
        // one-chunk context hands off to the tiled kernel, but only when that
        // kernel is there to take it.
        && (attn_splitk_chunks(seq_len) > 1 || kernels.attention_decode_tiled.is_none()))
    {
        return None;
    }
    // GQA-shared upgrade within the split-K route: same inputs, same scratch,
    // same output — one CTA per (KV head, chunk) instead of per (query head,
    // chunk). Only for the geometry the kernels are specialised for and a
    // context the chunk cap covers; every other shape keeps the pair below,
    // so nothing is silently truncated.
    let gqa6_loaded = kernels.attention_decode_splitk_partial_gqa6.is_some()
        && kernels.attention_decode_splitk_merge_gqa6.is_some();
    if gqa6_loaded
        && attention_decode_splitk_gqa6_supports(num_heads, num_kv_heads, head_dim, seq_len)
        && scratch_o_floats
            >= (num_heads as usize)
                * (attn_splitk_gqa6_chunks(seq_len) as usize)
                * (head_dim as usize)
    {
        return Some(SplitKChoice::Gqa6);
    }
    Some(SplitKChoice::PerHead)
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_routed(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    splitk_scratch: Option<&mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<AttentionDecodeVariant, RuntimeError> {
    let force_tiled = decode_tiled_force_enabled();
    let threshold = decode_tiled_threshold();
    let mut variant = attention_decode_variant(seq_len, force_tiled, threshold);

    // Say once why a loaded GQA-shared pair will not serve this dispatch.
    // Asked here, before any route is taken, so that every way out reaches
    // the operator who set the flag — including the ones that never enter
    // the split-K block below.
    let gqa6_loaded = kernels.attention_decode_splitk_partial_gqa6.is_some()
        && kernels.attention_decode_splitk_merge_gqa6.is_some();
    let scratch_o_floats = splitk_scratch.as_ref().map(|s| s.2.len());
    if gqa6_loaded {
        if let Some(reason) = splitk_gqa6_exclusion_reason(
            force_tiled,
            variant,
            scratch_o_floats,
            kernels.attention_decode_splitk_partial.is_some()
                && kernels.attention_decode_splitk_merge.is_some(),
            kernels.attention_decode_tiled.is_some(),
            attn_splitk_chunks(seq_len),
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            attn_splitk_gqa6_chunk_bound(),
        ) {
            announce_splitk_gqa6_excluded(num_heads, num_kv_heads, head_dim, seq_len, reason);
        }
    }

    match decode_attention_splitk_choice(
        kernels,
        variant,
        force_tiled,
        scratch_o_floats,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    ) {
        Some(SplitKChoice::Gqa6) => {
            let scratch = splitk_scratch.expect("a split-K choice implies scratch");
            launch_attention_decode_splitk_gqa6(
                device,
                kernels,
                q,
                k_cache,
                v_cache,
                scratch,
                attn_out,
                num_heads,
                num_kv_heads,
                seq_len,
                max_seq_len,
                scale,
            )?;
            announce_splitk_gqa6_route(num_heads, num_kv_heads, head_dim, seq_len);
            return Ok(AttentionDecodeVariant::SplitKGqa6);
        }
        Some(SplitKChoice::PerHead) => {
            let scratch = splitk_scratch.expect("a split-K choice implies scratch");
            launch_attention_decode_splitk(
                device,
                kernels,
                q,
                k_cache,
                v_cache,
                scratch,
                attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                scale,
            )?;
            announce_splitk_route(head_dim, seq_len);
            return Ok(AttentionDecodeVariant::SplitK);
        }
        None => {}
    }

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
    if variant == AttentionDecodeVariant::Tiled
        && !attention_decode_tiled_supports_head_dim(head_dim)
    {
        variant = AttentionDecodeVariant::SingleBlock;
    }

    match variant {
        // Both split-K variants return early from the upgrade block above;
        // the selector never produces them here.
        AttentionDecodeVariant::SplitK | AttentionDecodeVariant::SplitKGqa6 => unreachable!(),
        // The half routes belong to the half router; this one takes F32 bytes.
        AttentionDecodeVariant::TiledF16
        | AttentionDecodeVariant::SplitKGqa6F16
        | AttentionDecodeVariant::SplitKF16 => unreachable!(),
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
                .map_err(|e| RuntimeError::Compute(format!("attention_decode launch: {e}")))?;
            {
                static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                super::decode::announce_route_once(&SEEN, || {
                    format!("[CUDA] attention_decode: ACTIVE (head_dim={head_dim})")
                });
            }
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
            {
                static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                super::decode::announce_route_once(&SEEN, || {
                    let codegen = kernels.attention_decode_tiled_codegen;
                    let name = super::decode::attention_decode_tiled_route_name(codegen);
                    let target = super::decode::attn_tiled_codegen_target(codegen);
                    format!("[CUDA] {name}: ACTIVE (head_dim={head_dim}, target={target})")
                });
            }
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
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_0_to_f32 {label}: {e}")))?;
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
        .map_err(|e| RuntimeError::Compute(format!("dequant_q4_0_to_f32 {label}: {e}")))?;
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
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_0_to_f16 {label}: {e}")))?;
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
        .map_err(|e| RuntimeError::Compute(format!("dequant_q4_0_to_f16 {label}: {e}")))?;
    Ok(())
}

/// Launch `dequant_ct4_to_f16`: CtInt4G32 decode blocks -> F16 image.
///
/// # Safety
///
/// Same contract as `launch_dequant_q4_0_to_f16`.
unsafe fn launch_dequant_ct4_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    ct4_data: &CudaSlice<u8>,
    f16_out: &mut CudaSlice<u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let dequant_fn = kernels.dequant_ct4_to_f16.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!("dequant_ct4_to_f16 {label}: kernel unavailable"))
    })?;
    // 32 elements per 20-byte block; the kernel indexes blocks straight from
    // the element id, so a short weight buffer would read out of bounds.
    let expected_w = num_elements / 32 * 20;
    let needed_out = num_elements * 2;
    if num_elements % 32 != 0
        || ct4_data.len() != expected_w
        || f16_out.len() < needed_out
        || u32::try_from(num_elements).is_err()
    {
        return Err(RuntimeError::Compute(format!(
            "dequant_ct4_to_f16 {label}: shape mismatch: {num_elements} elements, \
             weight {} bytes (expected {expected_w}), out {} bytes (need {needed_out})",
            ct4_data.len(),
            f16_out.len(),
        )));
    }
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(dequant_fn)
        .arg(ct4_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_ct4_to_f16 {label}: {e}")))?;
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
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_slice input slice out of bounds: offset={_in_offset} dim={in_dim}",
            ))
        })?;
    let mut out_view = output
        .try_slice_mut(_out_offset.._out_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_slice output slice out of bounds: offset={_out_offset} dim={out_dim}",
            ))
        })?;

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
        GpuWeightBuf::Ct4Raw(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_slice {label}: CtInt4G32 is served by the prefill \
                 HGEMM path, not the per-row fallback"
            )));
        }
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
                .map_err(|e| RuntimeError::Compute(format!("matvec F16 {label} prefill: {e}")))?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels
                .matvec_q8_0_aligned
                .as_ref()
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
            let q8_fn = kernels
                .matvec_q8_0_dp4a
                .as_ref()
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
                .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 {label} prefill: {e}")))?;
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
                .map_err(|e| RuntimeError::Compute(format!("matvec Q4_0 {label} prefill: {e}")))?;
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
                .map_err(|e| RuntimeError::Compute(format!("matvec BF16 {label} prefill: {e}")))?;
        }
        // split-layout: prefill never dispatches against Q8Split/Q4Split
        // siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec prefill fallback {label}: Q8Split/Q4Split \
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
    kv: &KvView<'_>,
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
            q_batch.len(),
            needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_v2: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(),
            needed,
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
    let msl = kv.seq_stride as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_v2)
        .arg(q_batch)
        .arg(kv.k)
        .arg(kv.v)
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

/// Query rows per block of the tiled SGEMM prefill attention: bounds the score
/// scratch to `group * 512 * kv_len` floats.
pub(crate) const ATTN_PREFILL_SGEMM_ROWS: usize = 512;

/// Threads per block of `attn_softmax_causal_rows`. The kernel folds the row
/// max and the row sum across exactly `SMX_THREADS / 32` warps in a fixed
/// order and sizes its shared `part[]` from that count, so this value and
/// `SMX_THREADS` in `attn_softmax_causal.cu` must agree; the kernel carries a
/// `static_assert` on the warp count.
pub(crate) const ATTN_SOFTMAX_CAUSAL_THREADS: u32 = 128;

/// Query rows of the block starting at `qb` in a `batch`-row prefill: the
/// block size the tiled SGEMM attention walks with, `ATTN_PREFILL_SGEMM_ROWS`
/// except for the last (partial) block. `qb < batch` is the caller's loop
/// condition; `qb >= batch` yields 0, which would not terminate the walk.
pub(crate) fn attn_sgemm_block_rows(batch: usize, qb: usize) -> usize {
    batch.saturating_sub(qb).min(ATTN_PREFILL_SGEMM_ROWS)
}

/// Tiled prefill attention in exact F32: for each block of query rows and
/// each KV head, S = Q·Kᵀ over that head's query group by one strided-batched
/// SGEMM (the same K for every head of the group), a causal row softmax, then
/// O = P·V by a second strided-batched SGEMM. Same inputs and outputs as
/// `launch_flash_attention_br4`.
///
/// # Exact F32
///
/// Both products run as `cublasSgemmStridedBatched` with F32 operands, F32
/// scalars and an F32 accumulator, on the handle's default math mode
/// (`CUBLAS_DEFAULT_MATH`, which cuBLAS documents as using "compute and
/// intermediate storage precisions with at least the same number of mantissa
/// and exponent bits as requested"). TF32 carries ten mantissa bits against
/// F32's twenty-three, so it is outside what that mode may select for an F32
/// request; cuBLAS documents TF32 acceleration of single-precision routines as
/// what the `CUBLAS_TF32_TENSOR_OP_MATH` math mode enables. Nothing in the
/// engine calls `cublasSetMathMode`, and the only documented environment lever,
/// `NVIDIA_TF32_OVERRIDE=0`, can only take TF32 away. **Calling
/// `cublasSetMathMode` with `CUBLAS_TF32_TENSOR_OP_MATH` (or building this
/// through `cublasGemmEx` with a `_FAST_TF32` compute type) would silently
/// break the exact-F32 contract these two calls stand on, and with it the
/// exact-F32 prefill attention.**
///
/// # Evaluation order
///
/// The arithmetic is exact F32 but not the scalar kernel's order: the softmax
/// row is normalised before P·V and reduced over the whole row, where the
/// scalar kernel accumulates unnormalised weighted values online and
/// normalises afterwards. Greedy output can therefore differ from the scalar
/// kernel where two candidates sit within rounding of each other, and cuBLAS
/// guarantees bit-wise reproducibility only within a toolkit version on a
/// given architecture and SM count.
///
/// Dispatched on the fused Q+gate prefill path only (`attn_q_norm` present —
/// every artifact today's converter produces: qwen35 / qwen35moe). A layer
/// without per-head q/k norms takes the plain prefill dispatch, which runs the
/// scalar kernel.
///
/// `scores` is the block `alloc_prefill_scratch` sized for this prefill; it
/// is never allocated here (see the check below for why).
///
/// Both GEMMs batch over the query group with `stride_a = 0`, which points
/// every instance at the one K (or V) that group shares. cuBLAS's reference
/// for `cublas<t>gemmStridedBatched` states the requirement on the batch
/// offsets as "The unit for the offset is number of elements and must not be
/// zero", while NVIDIA's own strided-batched example broadcasts an input with
/// a zero stride; the sentence reads as a constraint on the output offsets,
/// which the same page spells out ("Matrices C[i] should not overlap;
/// otherwise, undefined behavior is expected") and which this call honours:
/// `stride_c` is `rows * s_ld` for Q·Kᵀ and `head_dim` for P·V, so no two
/// instances write the same element. A and B are read-only, so their
/// instances may alias. Measured correct on CUDA 13.1 / sm_120 (md5-identical
/// greedy output against the scalar kernel). If a future cuBLAS rejects a
/// zero input stride, the replacement is one GEMM per group head — `group`
/// times the launches per block and KV head, no extra memory.
///
/// # Safety
///
/// `q` holds `batch * num_heads * head_dim` floats, `attn_out` at least as
/// many, and the KV cache holds positions `0..pos_start + batch`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_flash_attention_sgemm(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    kv: &KvView<'_>,
    attn_out: &mut CudaSlice<f32>,
    scores: &mut Option<CudaSlice<f32>>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use cudarc::cublas::StridedBatchedConfig;
    // Buffer sizes first, before any allocation or launch (as in
    // `launch_flash_attention_br4`).
    let needed_io = batch * num_heads * head_dim;
    if q.len() < needed_io {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_sgemm: q too small: have {} elements, \
             need {needed_io} (batch={batch}, q_dim={})",
            q.len(),
            num_heads * head_dim,
        )));
    }
    if attn_out.len() < needed_io {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_sgemm: attn_out too small: have {} elements, \
             need {needed_io} (batch={batch}, q_dim={})",
            attn_out.len(),
            num_heads * head_dim,
        )));
    }
    let softmax_fn = kernels.attn_softmax_causal.as_ref().ok_or_else(|| {
        RuntimeError::Compute("attention prefill sgemm: softmax kernel unavailable".into())
    })?;
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: {num_heads} query heads over {num_kv_heads} KV heads"
        )));
    }
    let group = num_heads / num_kv_heads;
    let q_dim = num_heads * head_dim;
    let max_seq = kv.seq_stride;
    let kv_total = pos_start + batch;
    if kv_total > max_seq {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: {kv_total} positions exceed the KV cache's {max_seq}"
        )));
    }
    let needed =
        attn_score_block_elems(batch, num_heads, num_kv_heads, pos_start).ok_or_else(|| {
            RuntimeError::Compute("attention prefill sgemm: score block size overflows".into())
        })?;
    // Never allocated here. The layer loop writes each layer's KV and
    // advances that cache's length BEFORE it reaches this dispatch, while the
    // host-side KV length only advances after the last layer; an allocation
    // that failed inside the loop would abort the prefill with the two out of
    // step. `alloc_prefill_scratch` therefore sizes the block up front, ahead
    // of the first KV write, and the dispatcher only routes here when it is
    // present. This check is the guard on that contract, not a fallback.
    let have_block = match scores.as_ref() {
        Some(s) => s.len() >= needed,
        None => false,
    };
    if !have_block {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: no score block for batch {batch} at position \
             {pos_start} ({needed} floats needed); the block is sized before the \
             layer loop and this route runs only when it exists"
        )));
    }
    let s_buf = scores
        .as_mut()
        .expect("score block presence checked immediately above");
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut qb = 0usize;
    while qb < batch {
        let rows = attn_sgemm_block_rows(batch, qb);
        let kv_len = pos_start + qb + rows; // causal bound for this block
        let s_ld = kv_len;
        for kv_h in 0..num_kv_heads {
            let h0 = kv_h * group;
            let k_view = kv.k.slice(kv_h * max_seq * head_dim..);
            let v_view = kv.v.slice(kv_h * max_seq * head_dim..);
            let q_view = q.slice(qb * q_dim + h0 * head_dim..);
            // S[g][i][j] = sum_d Q[i][h0+g][d] * K[j][d]: in cuBLAS's column-major
            // terms C(kv_len x rows) = K(head_dim x kv_len)^T * Q(head_dim x rows).
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: kv_len as i32,
                    n: rows as i32,
                    k: head_dim as i32,
                    alpha: 1.0f32,
                    lda: head_dim as i32,
                    ldb: q_dim as i32,
                    beta: 0.0f32,
                    ldc: s_ld as i32,
                },
                batch_size: group as i32,
                stride_a: 0,
                stride_b: head_dim as i64,
                stride_c: (rows * s_ld) as i64,
            };
            device
                .blas
                .gemm_strided_batched(cfg, &k_view, &q_view, s_buf)
                .map_err(|e| RuntimeError::Compute(format!("attention prefill sgemm QK: {e}")))?;
            let rows_u32 = rows as u32;
            let kv_u32 = kv_len as u32;
            let ld_u32 = s_ld as u32;
            let p0 = (pos_start + qb) as u32;
            device
                .stream
                .launch_builder(softmax_fn)
                .arg(&mut *s_buf)
                .arg(&rows_u32)
                .arg(&kv_u32)
                .arg(&ld_u32)
                .arg(&p0)
                .arg(&scale)
                .launch(CudarcLaunchConfig {
                    grid_dim: (rows as u32, group as u32, 1),
                    block_dim: (ATTN_SOFTMAX_CAUSAL_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("attn_softmax_causal_rows: {e}")))?;
            // O[i][h0+g][d] = sum_j P[g][i][j] * V[j][d]: column-major
            // C(head_dim x rows) = V(head_dim x kv_len) * P(kv_len x rows).
            let mut o_view = attn_out.slice_mut(qb * q_dim + h0 * head_dim..);
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: head_dim as i32,
                    n: rows as i32,
                    k: kv_len as i32,
                    alpha: 1.0f32,
                    lda: head_dim as i32,
                    ldb: s_ld as i32,
                    beta: 0.0f32,
                    ldc: q_dim as i32,
                },
                batch_size: group as i32,
                stride_a: 0,
                stride_b: (rows * s_ld) as i64,
                stride_c: head_dim as i64,
            };
            device
                .blas
                .gemm_strided_batched(cfg, &v_view, &*s_buf, &mut o_view)
                .map_err(|e| RuntimeError::Compute(format!("attention prefill sgemm PV: {e}")))?;
        }
        qb += rows;
    }
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
    kv: &KvView<'_>,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{flash_attention_br4_block_size, flash_attention_br4_shared_bytes, FA_BR};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(),
            needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(),
            needed,
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
    let msl = kv.seq_stride as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_br4)
        .arg(q_batch)
        .arg(kv.k)
        .arg(kv.v)
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

#[allow(clippy::too_many_arguments)]
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
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res input slice out of bounds: offset={in_offset} dim={in_dim}",
            ))
        })?;
    let res_view = residual
        .try_slice(res_offset..res_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res residual slice out of bounds: offset={res_offset} dim={out_dim}",
            ))
        })?;
    let mut out_view = output
        .try_slice_mut(out_offset..out_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res output slice out of bounds: offset={out_offset} dim={out_dim}",
            ))
        })?;

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
        GpuWeightBuf::Ct4Raw(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_slice_residual {label}: CtInt4G32 is served by the \
                 prefill HGEMM path, not the per-row fallback"
            )));
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned residual prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels
                .matvec_q8_0_aligned_residual
                .as_ref()
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
            let q8_fn = kernels
                .matvec_q8_0_dp4a_residual
                .as_ref()
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
        // split-layout: prefill never dispatches against Q8Split/Q4Split
        // siblings.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec+residual prefill fallback {label}: Q8Split/Q4Split \
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
            .map_err(|e| RuntimeError::Compute(format!("f32_to_f16_vec4 {label}: {e}",)))?;
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
            .map_err(|e| RuntimeError::Compute(format!("f32_to_f16_vec {label}: {e}",)))?;
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
            .map_err(|e| RuntimeError::Compute(format!("f32_to_bf16_vec4 {label}: {e}",)))?;
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
            .map_err(|e| RuntimeError::Compute(format!("f32_to_bf16_vec {label}: {e}",)))?;
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
pub(crate) unsafe fn launch_cublas_gemm_bf16(
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

#[cfg(test)]
mod attn_splitk_chunk_tests {
    //! Split-count and partition tests for the split-K decode-attention
    //! pair. Both are hardware-independent: the count is pure Rust over the
    //! resolved target, and the partition mirrors the bounds the partial
    //! kernel derives from `seq_len` and the count.

    use super::{attn_splitk_chunk_count, ATTN_SPLITK_S_MAX};
    use crate::runtime_defaults::ATTN_SPLITK_CHUNK_POSITIONS;

    /// The count at the shipping target: one chunk per 128 KV positions,
    /// never below 1, never above the scratch bound.
    #[test]
    fn chunk_count_at_the_default_target() {
        for (seq_len, want) in [
            (0u32, 1u32),
            (1, 1),
            (127, 1),
            (128, 1),
            (129, 2),
            (512, 4),
            (1300, 11),
            (4096, 32),
            (4097, 32),
            (u32::MAX, 32),
        ] {
            assert_eq!(
                attn_splitk_chunk_count(seq_len, ATTN_SPLITK_CHUNK_POSITIONS),
                want,
                "seq_len={seq_len}"
            );
        }
    }

    /// A larger target yields proportionally fewer chunks, and the cap still
    /// binds from above.
    #[test]
    fn chunk_count_follows_the_target() {
        assert_eq!(attn_splitk_chunk_count(1300, 256), 6);
        assert_eq!(attn_splitk_chunk_count(1300, 1300), 1);
        assert_eq!(attn_splitk_chunk_count(1300, u32::MAX), 1);
        assert_eq!(attn_splitk_chunk_count(u32::MAX, 128), ATTN_SPLITK_S_MAX);
    }

    /// The chunks the partial kernel derives cover [0, seq_len) exactly:
    /// chunk c walks [c * span, min((c + 1) * span, seq_len)) with
    /// span = seq_len divided by the count, rounded up. No gap, no overlap,
    /// no empty chunk, at every context routed to the pair.
    #[test]
    fn chunks_partition_the_context_without_gap_or_overlap() {
        for seq_len in 1u32..20_000 {
            let count = attn_splitk_chunk_count(seq_len, ATTN_SPLITK_CHUNK_POSITIONS);
            if count == 1 {
                continue; // one chunk: the whole context, nothing to partition
            }
            let span = seq_len.div_ceil(count);
            let mut next = 0u32;
            for chunk in 0..count {
                let p0 = chunk * span;
                let p1 = (p0 + span).min(seq_len);
                assert_eq!(p0, next, "seq_len={seq_len} chunk={chunk}: gap or overlap");
                assert!(p0 < p1, "seq_len={seq_len} chunk={chunk}: empty chunk");
                next = p1;
            }
            assert_eq!(next, seq_len, "seq_len={seq_len}: chunks left a tail");
        }
    }
}

#[cfg(test)]
mod attn_splitk_gqa6_tests {
    //! Shape and exclusion-reason tests for the GQA-shared split-K pair.
    //! Hardware-independent: both predicates are pure Rust, and the reason
    //! function takes the environment-derived shipping split count as an
    //! argument rather than resolving it.

    use super::{
        attention_decode_splitk_gqa6_supports, attention_decode_splitk_gqa6_supports_within,
        attn_splitk_gqa6_chunk_bound, attn_splitk_gqa6_chunks, attn_splitk_gqa6_max_seq_len,
        attn_splitk_gqa6_served_seq_len, splitk_gqa6_exclusion_reason, SplitKGqa6Exclusion,
        ATTN_SPLITK_GQA6_CHUNK, ATTN_SPLITK_GQA6_DIM_TILES, ATTN_SPLITK_GQA6_HEAD_DIM,
        ATTN_SPLITK_GQA6_S_MAX,
    };
    use crate::cuda::decode::AttentionDecodeVariant;

    /// The Qwen3.8-27B full-attention shape, and the scratch a 4096-position
    /// cache gets: enough for every eligible context.
    const HEADS: u32 = 24;
    const KV: u32 = 4;
    const HD: u32 = ATTN_SPLITK_GQA6_HEAD_DIM;
    fn ample_scratch() -> Option<usize> {
        Some((HEADS * ATTN_SPLITK_GQA6_S_MAX * HD) as usize)
    }

    /// Every argument set to a value that takes the route, so each test
    /// perturbs exactly one thing.
    fn reason_for_eligible() -> Option<SplitKGqa6Exclusion> {
        splitk_gqa6_exclusion_reason(
            false,
            AttentionDecodeVariant::Tiled,
            ample_scratch(),
            true,
            true,
            9, // the shipping count at 1,100 positions
            HEADS,
            KV,
            HD,
            1100,
            ATTN_SPLITK_GQA6_S_MAX,
        )
    }

    #[test]
    fn the_eligible_dispatch_has_no_exclusion_reason() {
        assert_eq!(reason_for_eligible(), None);
    }

    #[test]
    fn every_way_out_of_the_route_has_its_own_reason() {
        use SplitKGqa6Exclusion as X;
        let cases: [(&str, X, Option<X>); 8] = [
            (
                "forced tiled",
                X::ForcedTiled,
                splitk_gqa6_exclusion_reason(
                    true,
                    AttentionDecodeVariant::Tiled,
                    ample_scratch(),
                    true,
                    true,
                    9,
                    HEADS,
                    KV,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "single-block threshold",
                X::SingleBlockThreshold,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::SingleBlock,
                    ample_scratch(),
                    true,
                    true,
                    9,
                    HEADS,
                    KV,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "no scratch (a prefill dispatch site)",
                X::NoScratch,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    None,
                    true,
                    true,
                    9,
                    HEADS,
                    KV,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "shipping pair absent",
                X::ShippingPairAbsent,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    ample_scratch(),
                    false,
                    true,
                    9,
                    HEADS,
                    KV,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "head_dim the split-K route declines",
                X::HeadDim,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    ample_scratch(),
                    true,
                    true,
                    9,
                    HEADS,
                    KV,
                    100,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                // A context short enough that the shipping split count is
                // one: the tiled kernel takes the token without the dispatch
                // ever entering the split-K block.
                "one-chunk context",
                X::OneChunkContext,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    ample_scratch(),
                    true,
                    true,
                    1,
                    HEADS,
                    KV,
                    HD,
                    64,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "geometry the kernels do not serve",
                X::Shape,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    ample_scratch(),
                    true,
                    true,
                    9,
                    32,
                    4,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
            (
                "scratch sized for the other split count",
                X::ScratchTooSmall,
                splitk_gqa6_exclusion_reason(
                    false,
                    AttentionDecodeVariant::Tiled,
                    Some((HEADS * 32 * HD) as usize),
                    true,
                    true,
                    9,
                    HEADS,
                    KV,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX,
                ),
            ),
        ];
        for (what, want, got) in cases {
            assert_eq!(got, Some(want), "{what}");
        }
        // Every variant is reachable, and each has its own latch slot.
        let reached: Vec<X> = cases.iter().map(|(_, w, _)| *w).collect();
        for x in SplitKGqa6Exclusion::ALL {
            assert!(
                reached.contains(&x),
                "{x:?} is unreachable from the dispatcher"
            );
        }
        for (i, a) in SplitKGqa6Exclusion::ALL.iter().enumerate() {
            for b in SplitKGqa6Exclusion::ALL.iter().skip(i + 1) {
                assert!(
                    !std::ptr::eq(a.latch(), b.latch()),
                    "{a:?} and {b:?} share a latch, so one would silence the other"
                );
            }
        }
    }

    /// A one-chunk context with no tiled kernel to hand off to stays on the
    /// split-K route, so the GQA-shared pair may still take it.
    #[test]
    fn a_one_chunk_context_without_a_tiled_kernel_is_not_excluded() {
        assert_eq!(
            splitk_gqa6_exclusion_reason(
                false,
                AttentionDecodeVariant::Tiled,
                ample_scratch(),
                true,
                false,
                1,
                HEADS,
                KV,
                HD,
                64,
                ATTN_SPLITK_GQA6_S_MAX,
            ),
            None
        );
    }

    #[test]
    fn the_supported_scope_is_six_query_heads_per_kv_head_at_head_dim_256() {
        for (heads, kv) in [(24, 4), (12, 2), (6, 1), (48, 8)] {
            assert!(
                attention_decode_splitk_gqa6_supports_within(
                    heads,
                    kv,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX
                ),
                "{heads} query heads over {kv} KV heads is a 6:1 group"
            );
        }
        for (heads, kv) in [(24, 8), (32, 4), (24, 3), (24, 0)] {
            assert!(
                !attention_decode_splitk_gqa6_supports_within(
                    heads,
                    kv,
                    HD,
                    1100,
                    ATTN_SPLITK_GQA6_S_MAX
                ),
                "{heads} over {kv} is not a 6:1 group"
            );
        }
        for hd in [64, 128, 192, 512] {
            assert!(!attention_decode_splitk_gqa6_supports_within(
                HEADS,
                KV,
                hd,
                1100,
                ATTN_SPLITK_GQA6_S_MAX
            ));
        }
    }

    #[test]
    fn the_served_bound_follows_the_env_inside_the_compile_time_bound() {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS");
        assert_eq!(attn_splitk_gqa6_chunk_bound(), ATTN_SPLITK_GQA6_S_MAX);
        assert_eq!(
            attn_splitk_gqa6_served_seq_len(),
            attn_splitk_gqa6_max_seq_len()
        );
        std::env::set_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS", "256");
        assert_eq!(attn_splitk_gqa6_chunk_bound(), 256);
        assert_eq!(attn_splitk_gqa6_served_seq_len(), 4096, "the v0.30.0 bound");
        assert!(attention_decode_splitk_gqa6_supports(HEADS, KV, HD, 4096));
        assert!(
            !attention_decode_splitk_gqa6_supports(HEADS, KV, HD, 4097),
            "past the lowered bound the per-query-head pair serves"
        );
        std::env::set_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS", "99999");
        assert_eq!(
            attn_splitk_gqa6_chunk_bound(),
            ATTN_SPLITK_GQA6_S_MAX,
            "nothing raises the bound past the constant"
        );
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS");
    }

    /// One past the served bound is the shape exclusion — the message the
    /// operator reads when a generation crosses it.
    #[test]
    fn one_past_the_served_bound_is_the_shape_exclusion() {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS");
        let cap = attn_splitk_gqa6_max_seq_len();
        assert_eq!(cap, 16_384);
        assert_eq!(
            splitk_gqa6_exclusion_reason(
                false,
                AttentionDecodeVariant::Tiled,
                ample_scratch(),
                true,
                true,
                super::attn_splitk_chunks(cap + 1),
                HEADS,
                KV,
                HD,
                cap + 1,
                ATTN_SPLITK_GQA6_S_MAX,
            ),
            Some(SplitKGqa6Exclusion::Shape)
        );
        assert_eq!(
            splitk_gqa6_exclusion_reason(
                false,
                AttentionDecodeVariant::Tiled,
                ample_scratch(),
                true,
                true,
                super::attn_splitk_chunks(cap),
                HEADS,
                KV,
                HD,
                cap,
                ATTN_SPLITK_GQA6_S_MAX,
            ),
            None
        );
    }

    #[test]
    fn the_context_bound_is_the_scratch_bound() {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS");
        let cap = attn_splitk_gqa6_max_seq_len();
        assert_eq!(cap, ATTN_SPLITK_GQA6_S_MAX * ATTN_SPLITK_GQA6_CHUNK);
        assert_eq!(cap, 16_384);
        assert!(attention_decode_splitk_gqa6_supports_within(
            HEADS,
            KV,
            HD,
            cap,
            ATTN_SPLITK_GQA6_S_MAX
        ));
        assert!(!attention_decode_splitk_gqa6_supports_within(
            HEADS,
            KV,
            HD,
            cap + 1,
            ATTN_SPLITK_GQA6_S_MAX
        ));
        assert!(!attention_decode_splitk_gqa6_supports_within(
            HEADS,
            KV,
            HD,
            0,
            ATTN_SPLITK_GQA6_S_MAX
        ));
        assert_eq!(attn_splitk_gqa6_chunks(cap), ATTN_SPLITK_GQA6_S_MAX);
    }

    /// The kernels give one warp to a query head's whole chunk and stage that
    /// many V rows in shared, so the walk must never outgrow the chunk.
    #[test]
    fn no_eligible_context_makes_a_chunk_outgrow_its_warp() {
        for seq_len in 1..=attn_splitk_gqa6_max_seq_len() {
            let s = attn_splitk_gqa6_chunks(seq_len);
            assert!(s <= ATTN_SPLITK_GQA6_S_MAX, "seq_len={seq_len}: S={s}");
            let span = seq_len.div_ceil(s);
            assert!(
                span <= ATTN_SPLITK_GQA6_CHUNK && span <= 32,
                "seq_len={seq_len}: span={span}"
            );
        }
    }

    /// The kernel repeats the group size, head dimension and block width as
    /// its own `#define`s, and NVRTC never sees the Rust constants. The two
    /// are bound here rather than by a `const _`, because the shader is a
    /// string: a retune on one side that misses the other fails this test
    /// instead of launching a kernel whose shared-memory layout disagrees
    /// with the bytes the host requested.
    #[test]
    fn the_shader_defines_match_the_host_constants() {
        let src = crate::cuda::shaders::ATTENTION_DECODE_SPLITK_GQA6_KERNEL_SOURCE;
        for (name, want) in [
            ("GQA6_HD", ATTN_SPLITK_GQA6_HEAD_DIM),
            ("GQA6_G", super::ATTN_SPLITK_GQA6_GQA_RATIO),
            ("GQA6_BLOCK", super::ATTN_DECODE_TILED_BLOCK_DIM),
        ] {
            let line = src
                .lines()
                .find(|l| l.starts_with(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("the shader no longer defines {name}"));
            let got: u32 = line
                .split_whitespace()
                .nth(2)
                .and_then(|v| v.trim_end_matches('u').parse().ok())
                .unwrap_or_else(|| panic!("cannot read a value out of `{line}`"));
            assert_eq!(got, want, "#define {name} disagrees with the host constant");
        }
    }

    /// The merge's CTAs must tile the head exactly, one dimension per thread.
    #[test]
    fn the_merge_tiles_cover_the_head_exactly() {
        assert_eq!(
            ATTN_SPLITK_GQA6_DIM_TILES * super::ATTN_DECODE_TILED_BLOCK_DIM,
            ATTN_SPLITK_GQA6_HEAD_DIM
        );
    }
}

// ---------------------------------------------------------------------------
// The half store's readers and writers. Every function here takes `u16`
// buffers and the kernels in `KernelSet::kv_f16`, which exist only when the
// backend was built for a half store; nothing here can be reached with F32
// bytes.
// ---------------------------------------------------------------------------

/// Dynamic shared bytes of `attention_decode_splitk_partial_gqa6_f16`: the Q
/// block as F32, the V tile as halves, the score block and the (m, l) slots.
pub const fn attn_splitk_gqa6_partial_shared_bytes_f16() -> u32 {
    (ATTN_SPLITK_GQA6_GQA_RATIO * ATTN_SPLITK_GQA6_HEAD_DIM
        + ATTN_SPLITK_GQA6_CHUNK * ATTN_SPLITK_GQA6_HEAD_DIM / 2
        + ATTN_SPLITK_GQA6_GQA_RATIO * ATTN_SPLITK_GQA6_CHUNK
        + 2 * ATTN_SPLITK_GQA6_GQA_RATIO)
        * 4
}
const _: () = assert!(attn_splitk_gqa6_partial_shared_bytes_f16() == 14_768);

/// Route a decode step over a half store: the same admission sequence as the
/// F32 router, each route replaced by its half twin. The force knob is
/// honoured; the split-K upgrade applies to the automatic tiled selection with
/// the same scratch, kernel and shape conditions; the GQA-shared twin serves
/// the pair's geometry and window and the per-query-head twin every other
/// shape the F32 router hands to that pair; the tiled twin serves the rest.
/// The single-block route has no half twin, so a threshold that would select
/// it is refused at init (the store is never built) and here as a defence.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_routed_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    splitk_scratch: Option<&mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<AttentionDecodeVariant, RuntimeError> {
    let f16 = kernels.kv_f16.as_ref().ok_or_else(|| {
        RuntimeError::Compute("16-bit KV cache dispatched without its kernels".into())
    })?;
    let force_tiled = decode_tiled_force_enabled();
    let threshold = decode_tiled_threshold();
    let variant = attention_decode_variant(seq_len, force_tiled, threshold);

    // The same one-time explanation the F32 router gives when a loaded
    // GQA-shared pair will not serve this dispatch.
    let gqa6_loaded = kernels.attention_decode_splitk_partial_gqa6.is_some()
        && kernels.attention_decode_splitk_merge_gqa6.is_some();
    let scratch_o_floats = splitk_scratch.as_ref().map(|s| s.2.len());
    if gqa6_loaded {
        if let Some(reason) = splitk_gqa6_exclusion_reason(
            force_tiled,
            variant,
            scratch_o_floats,
            kernels.attention_decode_splitk_partial.is_some()
                && kernels.attention_decode_splitk_merge.is_some(),
            kernels.attention_decode_tiled.is_some(),
            attn_splitk_chunks(seq_len),
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            attn_splitk_gqa6_chunk_bound(),
        ) {
            announce_splitk_gqa6_excluded(num_heads, num_kv_heads, head_dim, seq_len, reason);
        }
    }

    match decode_attention_splitk_choice(
        kernels,
        variant,
        force_tiled,
        scratch_o_floats,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    ) {
        Some(SplitKChoice::Gqa6) => {
            let scratch = splitk_scratch.expect("a split-K choice implies scratch");
            launch_attention_decode_splitk_gqa6_f16(
                device,
                kernels,
                f16,
                q,
                k_cache,
                v_cache,
                scratch,
                attn_out,
                num_heads,
                num_kv_heads,
                seq_len,
                max_seq_len,
                scale,
            )?;
            announce_splitk_gqa6_route_f16(num_heads, num_kv_heads, head_dim, seq_len);
            return Ok(AttentionDecodeVariant::SplitKGqa6F16);
        }
        Some(SplitKChoice::PerHead) => {
            let scratch = splitk_scratch.expect("a split-K choice implies scratch");
            launch_attention_decode_splitk_f16(
                device,
                kernels,
                f16,
                q,
                k_cache,
                v_cache,
                scratch,
                attn_out,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                max_seq_len,
                scale,
            )?;
            announce_splitk_route_f16(head_dim, seq_len);
            return Ok(AttentionDecodeVariant::SplitKF16);
        }
        None => {}
    }

    match variant {
        AttentionDecodeVariant::SingleBlock => Err(RuntimeError::Compute(format!(
            "16-bit KV cache: the single-block decode-attention route has no half twin \
             (LUMEN_CUDA_DECODE_TILED_THRESHOLD={threshold} selected it at seq_len {seq_len}); \
             unset the threshold or use --kv-precision f32"
        ))),
        _ => {
            if !attention_decode_tiled_supports_head_dim(head_dim) {
                return Err(RuntimeError::Compute(format!(
                    "16-bit KV cache: no half reader serves head_dim {head_dim}"
                )));
            }
            announce_tiled_route_f16(head_dim, seq_len);
            launch_attention_decode_tiled_f16(
                device,
                f16,
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
            Ok(AttentionDecodeVariant::TiledF16)
        }
    }
}

/// The per-query-head split-K pair on a half store: the half partial, the
/// shipped F32 merge, the same split count and scratch as the F32 pair.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_splitk_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    f16: &super::decode::KvF16Kernels,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    let s: u32 = attn_splitk_chunks(seq_len);
    let merge_fn = kernels
        .attention_decode_splitk_merge
        .as_ref()
        .ok_or_else(|| {
            RuntimeError::Compute("attention_decode_splitk_f16: merge not available".into())
        })?;
    let shared_bytes = attention_decode_tiled_shared_bytes(head_dim);
    let (m_part, l_part, o_part) = scratch;
    device
        .stream
        .launch_builder(&f16.splitk_partial)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut *m_part)
        .arg(&mut *l_part)
        .arg(&mut *o_part)
        .arg(&num_heads)
        .arg(&num_kv_heads)
        .arg(&head_dim)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads * s, 1, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: shared_bytes,
        })
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_splitk_partial_f16: {e}")))?;
    device
        .stream
        .launch_builder(merge_fn)
        .arg(&*m_part)
        .arg(&*l_part)
        .arg(&*o_part)
        .arg(attn_out)
        .arg(&num_heads)
        .arg(&head_dim)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_splitk_merge: {e}")))?;
    Ok(())
}

fn announce_splitk_route_f16(head_dim: u32, seq_len: u32) {
    static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    super::decode::announce_route_once(&SEEN, || {
        let chunks = attn_splitk_chunks(seq_len);
        format!(
            "[CUDA] attention_decode_splitk_partial_f16: ACTIVE (kv=f16, chunks={chunks}, \
             head_dim={head_dim}, seq_len={seq_len}, merge=attention_decode_splitk_merge)"
        )
    });
}

/// The GQA-shared pair on a half store: the half partial, the shipped F32
/// merge, the same split count and scratch as the F32 pair.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_splitk_gqa6_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    f16: &super::decode::KvF16Kernels,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    scratch: &mut (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>),
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    let merge_fn = kernels
        .attention_decode_splitk_merge_gqa6
        .as_ref()
        .ok_or_else(|| {
            RuntimeError::Compute("attention_decode_splitk_gqa6_f16: merge not available".into())
        })?;
    let s: u32 = attn_splitk_gqa6_chunks(seq_len);
    let chunk = ATTN_SPLITK_GQA6_CHUNK;
    let (m_part, l_part, o_part) = scratch;
    device
        .stream
        .launch_builder(&f16.splitk_partial_gqa6)
        .arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(&mut *m_part)
        .arg(&mut *l_part)
        .arg(&mut *o_part)
        .arg(&seq_len)
        .arg(&max_seq_len)
        .arg(&scale)
        .arg(&s)
        .arg(&chunk)
        .launch(CudarcLaunchConfig {
            grid_dim: (s, num_kv_heads, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: attn_splitk_gqa6_partial_shared_bytes_f16(),
        })
        .map_err(|e| {
            RuntimeError::Compute(format!("attention_decode_splitk_partial_gqa6_f16: {e}"))
        })?;
    device
        .stream
        .launch_builder(merge_fn)
        .arg(&*m_part)
        .arg(&*l_part)
        .arg(&*o_part)
        .arg(attn_out)
        .arg(&s)
        .launch(CudarcLaunchConfig {
            grid_dim: (num_heads, ATTN_SPLITK_GQA6_DIM_TILES, 1),
            block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
            shared_mem_bytes: attn_splitk_gqa6_merge_shared_bytes(s),
        })
        .map_err(|e| {
            RuntimeError::Compute(format!("attention_decode_splitk_merge_gqa6_f32: {e}"))
        })?;
    Ok(())
}

/// The tiled kernel on a half store: the F32 launcher's geometry and shared
/// bytes (the V tile is not staged, so nothing shrinks).
#[allow(clippy::too_many_arguments)]
unsafe fn launch_attention_decode_tiled_f16(
    device: &CudaDevice,
    f16: &super::decode::KvF16Kernels,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    attn_out: &mut CudaSlice<f32>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    scale: f32,
) -> Result<(), RuntimeError> {
    if head_dim % ATTN_DECODE_TILED_BLOCK_DIM != 0 {
        return Err(RuntimeError::Compute(format!(
            "attention_decode_tiled_f16: head_dim ({head_dim}) must be divisible by \
             BLOCK_DIM ({ATTN_DECODE_TILED_BLOCK_DIM})"
        )));
    }
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (ATTN_DECODE_TILED_BLOCK_DIM, 1, 1),
        shared_mem_bytes: attention_decode_tiled_shared_bytes(head_dim),
    };
    device
        .stream
        .launch_builder(&f16.tiled)
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
        .map_err(|e| RuntimeError::Compute(format!("attention_decode_tiled_f16 launch: {e}")))?;
    Ok(())
}

fn announce_splitk_gqa6_route_f16(num_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32) {
    static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    super::decode::announce_route_once(&SEEN, || {
        let chunks = attn_splitk_gqa6_chunks(seq_len);
        format!(
            "[CUDA] attention_decode_splitk_partial_gqa6_f16: ACTIVE (kv=f16, \
             q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, \
             seq_len={seq_len}, chunks={chunks}, chunk={chunk}, block={block}, \
             merge=attention_decode_splitk_merge_gqa6_f32)",
            chunk = ATTN_SPLITK_GQA6_CHUNK,
            block = ATTN_DECODE_TILED_BLOCK_DIM,
        )
    });
}

fn announce_tiled_route_f16(head_dim: u32, seq_len: u32) {
    static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    super::decode::announce_route_once(&SEEN, || {
        format!(
            "[CUDA] attention_decode_tiled_f16: ACTIVE (kv=f16, head_dim={head_dim}, \
             seq_len={seq_len}, block={block}, tile={tile})",
            block = ATTN_DECODE_TILED_BLOCK_DIM,
            tile = ATTN_DECODE_TILED_BLOCK_DIM,
        )
    });
}

/// `kv_cache_write_batch_f16`: `batch` tokens of F32 K or V into a half
/// store, rounding on the way in and counting what does not fit.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_kv_cache_write_batch_f16(
    device: &CudaDevice,
    f16: &super::decode::KvF16Kernels,
    cache: &mut CudaSlice<u16>,
    data: &CudaSlice<f32>,
    overflow: &mut CudaSlice<u32>,
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
        .launch_builder(&f16.write_batch)
        .arg(cache)
        .arg(data)
        .arg(overflow)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nkvh)
        .arg(&msl)
        .arg(&hd)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("kv_cache_write_batch_f16 launch: {e}")))?;
    Ok(())
}

/// `kv_cache_widen_f16` for K and V: positions `0..count` of every head,
/// widened into `[num_kv_heads, count, head_dim]` F32 buffers — the F32 cache
/// layout with a position stride of `count`, which is what the returned view
/// says.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_kv_widen_f16<'a>(
    device: &CudaDevice,
    f16: &super::decode::KvF16Kernels,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    out: &'a mut (CudaSlice<f32>, CudaSlice<f32>),
    num_kv_heads: usize,
    count: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<KvView<'a>, RuntimeError> {
    let total = num_kv_heads * count * head_dim;
    if out.0.len() < total || out.1.len() < total {
        return Err(RuntimeError::Compute(format!(
            "kv widen: {total} floats needed, buffers hold {} and {}",
            out.0.len(),
            out.1.len()
        )));
    }
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkvh = num_kv_heads as u32;
    let cnt = count as u32;
    let msl = max_seq_len as u32;
    let hd = head_dim as u32;
    for (cache, dst, which) in [(k_cache, &mut out.0, "K"), (v_cache, &mut out.1, "V")] {
        device
            .stream
            .launch_builder(&f16.widen)
            .arg(cache)
            .arg(dst)
            .arg(&nkvh)
            .arg(&cnt)
            .arg(&msl)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| {
                RuntimeError::Compute(format!("kv_cache_widen_f16 {which} launch: {e}"))
            })?;
    }
    Ok(KvView {
        k: &out.0,
        v: &out.1,
        seq_stride: count,
    })
}
