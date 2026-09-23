//! Tensor-core GEMMs for the DiT, via cuBLAS.
//!
//! The DiT's projections are large and dense — `[4114, 4096] x [4096, 12288]`
//! at 1024x1024 — and they are ~86% of its FLOPs. Weights and activations are
//! both bf16, with f32 accumulation.
//!
//! Every DiT linear is bias-free (`DitGpu::linear` passes no bias), so there is
//! no epilogue here and `beta` is always 0.
//!
//! # Layout
//!
//! cuBLAS is column-major and our tensors are row-major, which is handled by
//! reading the row-major buffer as its own transpose:
//!
//! ```text
//! row-major W[N, K] = column-major W_cm[K, N]
//! row-major A[M, K] = column-major A_cm[K, M]
//! row-major C[M, N] = column-major C_cm[N, M]
//! C_cm = W_cm^T * A_cm
//! cublasGemmEx(OP_T, OP_N, N, M, K, W, K, A, K, C, N)
//! ```
//!
//! The leading dimensions are the row-major row strides: `K` for both `W[N,K]`
//! and `A[M,K]`, and `N` for `C[M,N]`. No weight preprocessing is needed — the
//! bf16 bytes the container already holds are laid out exactly as cuBLAS wants.

use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::error::RuntimeError;

/// Threads per block in the f32 -> bf16 conversion; each thread converts one
/// element.
const CONVERT_THREADS: usize = 256;

/// An `[m, k]` bf16 activation, the operand every projection multiplies.
///
/// Several projections may read one activation — `to_q`, `to_k` and `to_v`
/// share the block input, the MLP's gate and up share theirs.
pub struct Bf16Activation {
    /// The `[m, k]` activation as bf16 bits, row-major.
    pub bits: CudaSlice<u16>,
    m: usize,
    k: usize,
}

impl Bf16Activation {
    /// The activation's width.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The activation's row count.
    pub fn m(&self) -> usize {
        self.m
    }

    /// An `[m, k]` activation a kernel has already rounded to bf16.
    pub fn from_bits(bits: CudaSlice<u16>, m: usize, k: usize) -> Result<Self, RuntimeError> {
        if m == 0 || k == 0 || bits.len() != m * k {
            return Err(RuntimeError::Compute(format!(
                "bf16 activation has {} elements, expected {m}x{k}",
                bits.len()
            )));
        }
        Ok(Self { bits, m, k })
    }
}

/// `out[M, N] = a[M, K] * w[N, K]^T` for bf16 weights and a bf16 activation,
/// f32 accumulation and output.
///
/// # Safety
///
/// `w` must hold `n * k` bf16 elements.
pub unsafe fn gemm_bf16(
    dev: &CudaDevice,
    w: &CudaSlice<u16>,
    a: &Bf16Activation,
    n: usize,
) -> Result<CudaSlice<f32>, RuntimeError> {
    gemm_ex(
        dev,
        w,
        a,
        n,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
    )
}

/// [`gemm_bf16`] with the output written as bf16, rounded to nearest even
/// from the f32 accumulator: what a bf16 `nn.Linear` produces, through the
/// same cuBLAS call.
///
/// # Safety
///
/// `w` must hold `n * k` bf16 elements.
pub unsafe fn gemm_bf16_out(
    dev: &CudaDevice,
    w: &CudaSlice<u16>,
    a: &Bf16Activation,
    n: usize,
) -> Result<Bf16Activation, RuntimeError> {
    let m = a.m;
    let bits = gemm_ex(
        dev,
        w,
        a,
        n,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
    )?;
    Bf16Activation::from_bits(bits, m, n)
}

/// The GEMM both entry points share, writing `C` as `c_type`, whose element
/// size `T` must match.
unsafe fn gemm_ex<T: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    w: &CudaSlice<u16>,
    a: &Bf16Activation,
    n: usize,
    c_type: cudarc::cublas::sys::cudaDataType_t,
) -> Result<CudaSlice<T>, RuntimeError> {
    let (m, k) = (a.m, a.k);
    check_shape(m, n, k)?;
    let a_bf16 = &a.bits;
    // Uninitialized: cuBLAS writes every element (beta = 0).
    let mut out: CudaSlice<T> = unsafe { dev.alloc_uninit::<T>(m * n) }
        .map_err(|e| RuntimeError::Compute(format!("gemm_bf16: alloc out: {e}")))?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // The guards live until the call has been issued on the stream.
    let (w_ptr, _w_guard) = w.device_ptr(&dev.stream);
    let (a_ptr, _a_guard) = a_bf16.device_ptr(&dev.stream);
    let (c_ptr, _c_guard) = out.device_ptr_mut(&dev.stream);

    let status = cudarc::cublas::sys::cublasGemmEx(
        *dev.blas.handle(),
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        n as i32,
        m as i32,
        k as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        k as i32,
        a_ptr as *const std::ffi::c_void,
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
        k as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        c_type,
        n as i32,
        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx bf16 GEMM [{m}x{n}x{k}]: status={status:?}"
        )));
    }
    drop((_w_guard, _a_guard, _c_guard));
    Ok(out)
}

/// Convert an f32 device buffer to bf16 bits.
///
/// # Safety
///
/// `src` must hold at least `dst.len()` f32 elements.
pub unsafe fn convert_f32_to_bf16(
    dev: &CudaDevice,
    convert: &CudaFunction,
    src: &CudaSlice<f32>,
    dst: &CudaSlice<u16>,
) -> Result<(), RuntimeError> {
    // The kernel indexes with `unsigned int` like every other kernel in this
    // crate, so the length must fit one. At 1024x1024 the largest activation
    // is 51M.
    let n = dst.len();
    if n == 0 {
        return Ok(());
    }
    if n > u32::MAX as usize {
        return Err(RuntimeError::Compute(format!(
            "f32 to bf16 conversion: {n} elements exceeds the u32 kernel limit"
        )));
    }
    let n32 = n as u32;
    let grid = n.div_ceil(CONVERT_THREADS) as u32;
    dev.stream
        .launch_builder(convert)
        .arg(src)
        .arg(dst)
        .arg(&n32)
        .launch(cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (CONVERT_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| RuntimeError::Compute(format!("f32 to bf16 conversion: {e}")))?;
    Ok(())
}

/// cuBLAS takes `i32` dimensions, so a shape past `i32::MAX` would wrap and
/// silently compute the wrong thing. Our largest is `4096x12288`.
fn check_shape(m: usize, n: usize, k: usize) -> Result<(), RuntimeError> {
    if m == 0 || n == 0 || k == 0 {
        return Err(RuntimeError::Compute(format!(
            "gemm_bf16: degenerate shape {m}x{n}x{k}"
        )));
    }
    for (name, v) in [("m", m), ("n", n), ("k", k)] {
        if v > i32::MAX as usize {
            return Err(RuntimeError::Compute(format!(
                "gemm_bf16: {name}={v} exceeds the i32 cuBLAS limit"
            )));
        }
    }
    Ok(())
}

/// The largest sequence the softmax kernel's shared row buffer holds. Above
/// this the launcher refuses rather than reading past the array.
pub const SOFTMAX_MAX_SEQ: usize = 8192;

/// Threads per softmax block: the kernel reduces through eight warp slots
/// (`part_max[8]`).
const SOFTMAX_THREADS: u32 = 256;

/// Block-causal softmax over the last axis, f32 scores in, bf16 probabilities
/// out.
///
/// `scores` is `[rows, seq]` with `rows = heads * seq`; the mask is derived
/// in-kernel from `text_count`, the length of the text prefix.
///
/// # Safety
///
/// `scores` must hold `rows * seq` elements.
pub unsafe fn mask_softmax_rows(
    dev: &CudaDevice,
    f: &CudaFunction,
    scores: &CudaSlice<f32>,
    rows: usize,
    seq: usize,
    text_count: u32,
) -> Result<CudaSlice<u16>, RuntimeError> {
    if seq == 0 || rows == 0 {
        return Err(RuntimeError::Compute(
            "mask_softmax_rows: empty input".to_string(),
        ));
    }
    if seq > SOFTMAX_MAX_SEQ {
        return Err(RuntimeError::Compute(format!(
            "mask_softmax_rows: seq {seq} exceeds the {SOFTMAX_MAX_SEQ} shared-memory limit"
        )));
    }
    if text_count as usize > seq {
        return Err(RuntimeError::Compute(format!(
            "mask_softmax_rows: text_count {text_count} exceeds seq {seq}"
        )));
    }
    if scores.len() != rows * seq {
        return Err(RuntimeError::Compute(format!(
            "mask_softmax_rows: scores has {} elements, expected {rows}x{seq}",
            scores.len()
        )));
    }
    let probs: CudaSlice<u16> = unsafe { dev.alloc_uninit::<u16>(rows * seq) }
        .map_err(|e| RuntimeError::Compute(format!("mask_softmax_rows: alloc: {e}")))?;
    let smem = (seq * std::mem::size_of::<f32>()) as u32;
    dev.stream
        .launch_builder(f)
        .arg(scores)
        .arg(&probs)
        .arg(&(seq as u32))
        .arg(&text_count)
        .launch(cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (SOFTMAX_THREADS, 1, 1),
            shared_mem_bytes: smem,
        })
        .map_err(|e| RuntimeError::Compute(format!("mask_softmax_rows: {e}")))?;
    Ok(probs)
}
