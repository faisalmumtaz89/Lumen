//! Block-causal attention for the DiT.
//!
//! Two implementations of `softmax(Q Kᵀ / √d) V` under the same mask:
//!
//! - [`fused_block_causal_attention`] is the production path: one kernel
//!   (`flash_attn.cu`) that keeps the scores in tensor-core tiles and runs an
//!   online softmax, so nothing of size `seq²` is ever written. It is tiled
//!   for 128-wide heads, the model's width.
//! - [`block_causal_attention`] is the unfused form: two batched cuBLAS
//!   products with the masked softmax as an elementwise pass in between. It
//!   materialises the scores (`heads · seq² · 4` bytes, 2.2 GB at 1024²) and
//!   exists as the reference the fused kernel is checked against
//!   (`cuda-ops-check`), for any head width.
//!
//! The mask is applied exactly in both: `text_count` is the length of the
//! leading run of text positions, and every text position precedes every
//! image position, so an image query sees every key while a text query is
//! causal over the text prefix.

use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

use super::launch::DevVec;
use super::{blas, ImageKernels};

/// The fused kernel's fixed geometry: `FA_D`, `FA_BQ` and `FA_THREADS` in
/// `flash_attn.cu`, which cannot read these. A disagreement leaves rows
/// unwritten, which `cuda-ops-check`'s production-shape case fails.
pub const FLASH_HEAD_DIM: usize = 128;
const FLASH_BQ: u32 = 64;
const FLASH_THREADS: u32 = 128;

/// `out[seq, heads, head_dim] = softmax(masked(Q Kᵀ / √d)) V`.
///
/// `q`, `k`, `v` and the returned buffer are all `[seq, heads, head_dim]`
/// row-major, the layout the projections and the per-head norms already produce.
/// `text_count` is the length of the leading run of text positions. The mask is
/// fully determined by it: every text position precedes every image position, so
/// an image query sees every key while a text query is causal over the text
/// prefix.
///
/// # Safety
///
/// `q`, `k` and `v` must each hold `seq * heads * head_dim` f32 elements.
pub unsafe fn block_causal_attention(
    dev: &CudaDevice,
    kernels: &ImageKernels,
    q: &DevVec,
    k: &DevVec,
    v: &DevVec,
    text_count: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<DevVec, RuntimeError> {
    if seq == 0 || heads == 0 || head_dim == 0 {
        return Err(RuntimeError::Compute(
            "block_causal_attention: empty input".to_string(),
        ));
    }
    let expect = seq * heads * head_dim;
    for (name, buf) in [("q", q), ("k", k), ("v", v)] {
        if buf.len != expect {
            return Err(RuntimeError::Compute(format!(
                "block_causal_attention: {name} has {} elements, expected {seq}x{heads}x{head_dim}",
                buf.len
            )));
        }
    }
    if seq > blas::SOFTMAX_MAX_SEQ {
        return Err(RuntimeError::Compute(format!(
            "block_causal_attention: seq {seq} exceeds the {} the softmax kernel supports",
            blas::SOFTMAX_MAX_SEQ
        )));
    }

    let text_count = u32::try_from(text_count)
        .map_err(|_| RuntimeError::Compute("attention: text count exceeds u32".to_string()))?;

    // No head permute is needed: a head `h` of the stored `[seq, heads,
    // head_dim]` is the strided sub-tensor `q[(i * heads + h) * head_dim + d]`,
    // whose row stride is `heads * head_dim` and whose batch stride is
    // `head_dim` — both legal cuBLAS leading dimensions.
    let q16 = to_bf16(dev, &kernels.f32_to_bf16_trunc, &q.buf)?;
    let k16 = to_bf16(dev, &kernels.f32_to_bf16_trunc, &k.buf)?;
    let v16 = to_bf16(dev, &kernels.f32_to_bf16_trunc, &v.buf)?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scores = unsafe { batched_qk(dev, &q16, &k16, seq, heads, head_dim, scale)? };

    let rows = heads * seq;
    let probs = unsafe {
        blas::mask_softmax_rows(
            dev,
            &kernels.mask_softmax_rows,
            &scores,
            rows,
            seq,
            text_count,
        )?
    };
    drop(scores);

    let out = unsafe { batched_pv(dev, &probs, &v16, seq, heads, head_dim)? };
    Ok(DevVec {
        buf: out,
        len: expect,
    })
}

/// [`block_causal_attention`] on the fused kernel: the same mask, the scores
/// never leaving the tensor-core tiles, and the result rounded to bf16 for
/// the `to_out` projection that consumes it.
///
/// `q` and `k` arrive already converted (the per-head norm writes them as
/// bf16); `v` is the projection's f32 output and is truncated here. The head
/// width is fixed at 128 by the kernel's tiling.
///
/// # Safety
///
/// `q` and `k` must hold `seq * heads * 128` bf16 elements and `v` as many
/// f32 elements.
pub unsafe fn fused_block_causal_attention(
    dev: &CudaDevice,
    kernels: &ImageKernels,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &DevVec,
    text_count: usize,
    seq: usize,
    heads: usize,
) -> Result<blas::Bf16Activation, RuntimeError> {
    if seq == 0 || heads == 0 {
        return Err(RuntimeError::Compute(
            "fused attention: empty input".to_string(),
        ));
    }
    let expect = seq * heads * FLASH_HEAD_DIM;
    if q.len() != expect || k.len() != expect || v.len != expect {
        return Err(RuntimeError::Compute(format!(
            "fused attention: q/k/v hold {}/{}/{} elements, expected {seq}x{heads}x{FLASH_HEAD_DIM}",
            q.len(),
            k.len(),
            v.len
        )));
    }
    if text_count > seq {
        return Err(RuntimeError::Compute(format!(
            "fused attention: text prefix {text_count} exceeds the sequence {seq}"
        )));
    }
    let text_count = text_count as u32;
    let v16 = to_bf16(dev, &kernels.f32_to_bf16_trunc, &v.buf)?;

    // Safety: the output is written in full by the kernel: every row below
    // `seq` of every head.
    let mut out: CudaSlice<u16> = unsafe { dev.alloc_uninit::<u16>(expect) }
        .map_err(|e| RuntimeError::Compute(format!("fused attention: alloc: {e}")))?;
    // The softmax runs in log2 units: scale by 1/√d and by log2(e) at once.
    let scale_log2 = std::f32::consts::LOG2_E / (FLASH_HEAD_DIM as f32).sqrt();
    let (seq_u, heads_u) = (seq as u32, heads as u32);
    let grid_x = seq_u.div_ceil(FLASH_BQ);
    // Safety: the grid is one block per 64 query rows per head, and the
    // kernel guards rows and keys past `seq`.
    unsafe {
        dev.stream
            .launch_builder(&kernels.flash_attn)
            .arg(q)
            .arg(k)
            .arg(&v16)
            .arg(&mut out)
            .arg(&seq_u)
            .arg(&heads_u)
            .arg(&text_count)
            .arg(&scale_log2)
            .launch(cudarc::driver::LaunchConfig {
                grid_dim: (grid_x, heads_u, 1),
                block_dim: (FLASH_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| RuntimeError::Compute(format!("flash_attn_bf16: {e}")))?;
    }
    blas::Bf16Activation::from_bits(out, seq, heads * FLASH_HEAD_DIM)
}

/// Convert an f32 buffer to bf16 bits through the shared conversion kernel.
pub fn to_bf16(
    dev: &CudaDevice,
    convert: &CudaFunction,
    src: &CudaSlice<f32>,
) -> Result<CudaSlice<u16>, RuntimeError> {
    // Safety: the conversion writes every element before the buffer is read.
    let dst: CudaSlice<u16> = unsafe { dev.alloc_uninit::<u16>(src.len()) }
        .map_err(|e| RuntimeError::Compute(format!("attention: bf16 alloc: {e}")))?;
    unsafe { blas::convert_f32_to_bf16(dev, convert, src, &dst)? };
    Ok(dst)
}

/// `scores[h] = Q[h] K[h]ᵀ · scale`, batched over heads, all head-contiguous.
///
/// # Safety
///
/// `q` and `k` must each hold `heads * seq * head_dim` bf16 elements.
unsafe fn batched_qk(
    dev: &CudaDevice,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    seq: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<CudaSlice<f32>, RuntimeError> {
    // cuBLAS writes every element (beta = 0), so zeroing is wasted work.
    let mut out: CudaSlice<f32> = unsafe { dev.alloc_uninit::<f32>(heads * seq * seq) }
        .map_err(|e| RuntimeError::Compute(format!("attention QK: alloc: {e}")))?;
    let alpha = scale;
    let beta = 0.0f32;
    // The guards live until the call has been issued on the stream.
    let (q_ptr, _q_guard) = q.device_ptr(&dev.stream);
    let (k_ptr, _k_guard) = k.device_ptr(&dev.stream);
    let (o_ptr, _o_guard) = out.device_ptr_mut(&dev.stream);

    // Row-major C[i, j] = sum_d q[i, d] * k[j, d]. Read as column-major with
    // C_cm[j, i] it is C_cm = k_cmᵀ q_cm, where k[i, d] row-major is k_cm[d, i]
    // — so OP_T on k, OP_N on q, and (m, n, k) = (seq, seq, head_dim).
    //
    // The leading dimension is the column-major row stride, which for a
    // `[seq, heads, head_dim]` row-major buffer is `heads * head_dim`; the batch
    // stride between heads is one `head_dim`.
    let lda = (heads * head_dim) as i64;
    let batch = head_dim as i64;
    let so = (seq * seq) as i64;
    let status = cudarc::cublas::sys::cublasGemmStridedBatchedEx(
        *dev.blas.handle(),
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        seq as i32,
        seq as i32,
        head_dim as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        k_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        lda as i32,
        batch,
        q_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        lda as i32,
        batch,
        &beta as *const f32 as *const std::ffi::c_void,
        o_ptr as *mut std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
        seq as i32,
        so,
        heads as i32,
        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "attention QK batched GEMM: status={status:?}"
        )));
    }
    drop((_q_guard, _k_guard, _o_guard));
    Ok(out)
}

/// `out[h] = probs[h] · V[h]`, batched over heads, bf16 operands and f32
/// accumulation.
///
/// # Safety
///
/// `probs` holds `heads * seq * seq` bf16 and `v` holds `heads * seq * head_dim`.
unsafe fn batched_pv(
    dev: &CudaDevice,
    probs: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<CudaSlice<f32>, RuntimeError> {
    let mut out: CudaSlice<f32> = unsafe { dev.alloc_uninit::<f32>(heads * seq * head_dim) }
        .map_err(|e| RuntimeError::Compute(format!("attention PV: alloc: {e}")))?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // The guards live until the call has been issued on the stream.
    let (s_ptr, _s_guard) = probs.device_ptr(&dev.stream);
    let (v_ptr, _v_guard) = v.device_ptr(&dev.stream);
    let (o_ptr, _o_guard) = out.device_ptr_mut(&dev.stream);

    // out[i, d] = sum_j s[i, j] * v[j, d]. Column-major out_cm[d, i] =
    // sum_j v_cm[d, j] s_cm[j, i], so out_cm = v_cm s_cm with neither operand
    // transposed, (m, n, k) = (head_dim, seq, seq).
    let ss = (seq * seq) as i64;
    // V is `[seq, heads, head_dim]` row-major, so head `h` starts at offset `h *
    // head_dim` and advances `heads * head_dim` per position — the same strided
    // view as q and k.
    let ldv = (heads * head_dim) as i64;
    let batchv = head_dim as i64;
    let ldo = (heads * head_dim) as i64;
    let batcho = head_dim as i64;
    let status = cudarc::cublas::sys::cublasGemmStridedBatchedEx(
        *dev.blas.handle(),
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        head_dim as i32,
        seq as i32,
        seq as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        v_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        ldv as i32,
        batchv,
        s_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        seq as i32,
        ss,
        &beta as *const f32 as *const std::ffi::c_void,
        o_ptr as *mut std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
        ldo as i32,
        batcho,
        heads as i32,
        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "attention PV batched GEMM: status={status:?}"
        )));
    }
    drop((_s_guard, _v_guard, _o_guard));
    Ok(out)
}
