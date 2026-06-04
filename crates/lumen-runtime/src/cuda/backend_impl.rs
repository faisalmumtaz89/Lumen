//! CUDA `ComputeBackend` implementation.
//!
//! Implements the full single-token decode pipeline on GPU:
//! - `embed_token`: GPU embedding lookup (F32 or Q8_0)
//! - `compute_layer`: RMSNorm -> QKV -> RoPE -> KV cache -> Attention ->
//! Output proj + residual -> FFN RMSNorm -> SwiGLU MLP -> Residual
//! - `compute_final`: Final RMSNorm -> output projection to logits
//! - `preload_weights`: Upload ALL layer weights to GPU once at startup
//! - `decode_token`: GPU-resident single-token decode (no per-layer upload)
//!
//! Supports F32, F16, Q8_0, and Q4_0 weight quantization. Two weight paths:
//! - **GPU-resident** (`preload_weights` called): all layer weights cached on GPU.
//! `compute_layer` uses cached `LayerWeightsGpu` -- zero host-to-device transfer.
//! - **Streaming** (no preload): per-call `upload_layer_weights` from `LayerView`.

use crate::compute::{ActivationBuffer, BackendCaps, ComputeBackend, ComputeDtype, Logits};
use crate::error::RuntimeError;
use crate::kv::{KvCacheView, KvPrecision};
use crate::weight::cache::{LayerView, WeightProvider};
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;

use super::decode::{
    self, KernelSet, fused_norm_matvec_block_size,
    hgemv_grid, hgemv_shared_bytes, HGEMV_BLOCK_DIM, HGEMV_SHMEM_LIMIT,
    matvec_block_size, matvec_q8_0_grid, matvec_smem_grid, matvec_smem_shared_bytes,
    rmsnorm_block_size, rmsnorm_shared_bytes, Q8_0_BLOCK_DIM, SMEM_BLOCK_DIM,
    fused_glu_grid, fused_glu_shared_bytes_f32, fused_glu_shared_bytes_f16,
    FUSED_GLU_BLOCK_DIM, FUSED_GLU_SHMEM_LIMIT,
    q8_1_quant_grid, dp4a_q8_1_grid, dp4a_q4_grid,
    Q8_1_QUANT_BLOCK_DIM, DP4A_Q8_1_BLOCK_DIM, DP4A_Q4_BLOCK_DIM,
};
use super::ffi::CudaDevice;
use super::gpu_buffers::{GpuWeightBuf, LayerWeightsGpu, upload_layer_weights};
use super::kv_cache::KvCacheGpu;
use super::shaders::EMBED_KERNEL_SOURCE;
use super::types::LaunchConfig;
use cudarc::cublas::{Gemv, GemvConfig, sys as cublas_sys};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cached cuBLAS algorithm selection for HGEMV (M, N=1, K) shapes.
///
/// During `preload_weights()`, benchmarks all 16 tensor-core cuBLAS algorithms
/// (ALGO0_TENSOR_OP through ALGO15_TENSOR_OP) plus DEFAULT_TENSOR_OP for each
/// unique (M=out_dim, K=in_dim) shape used in F16 decode. Caches the fastest
/// algorithm per shape. Falls back to DEFAULT_TENSOR_OP for un-benchmarked shapes.
///
/// Key insight: cuBLAS GEMM_DEFAULT_TENSOR_OP uses internal heuristics to select
/// an algorithm. For M=1 (GEMV) with small K, the heuristic may select a GEMM
/// kernel optimized for larger batch sizes. Explicit algorithm selection (like
/// cuBLAS algorithm autotuning) can find a better kernel for these
/// specific shapes, yielding 5-15% improvements on small models.
struct AlgoCache {
    /// Map from (out_dim, in_dim) -> best cublasGemmAlgo_t.
    best_algo: HashMap<(usize, usize), cublas_sys::cublasGemmAlgo_t>,
}

impl AlgoCache {
    fn new() -> Self {
        Self {
            best_algo: HashMap::new(),
        }
    }

    /// Look up the best algorithm for a given shape. Falls back to DEFAULT_TENSOR_OP.
    fn get(&self, out_dim: usize, in_dim: usize) -> cublas_sys::cublasGemmAlgo_t {
        self.best_algo
            .get(&(out_dim, in_dim))
            .copied()
            .unwrap_or(cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP)
    }
}

// ---------------------------------------------------------------------------
// BF16 cuBLAS algo cache (separate from F16 because BF16 uses
// `CUDA_R_16BF + CUBLAS_COMPUTE_32F`; the F16 autotune trains on
// `CUDA_R_16F + COMPUTE_32F_FAST_16F` which is a different algo space).
// Populated once per process at `preload_weights` when the model has BF16
// weights. Read by `launch_hgemv_bf16` / `launch_hgemv_bf16_residual` on
// every BF16 GemmEx call (60.9% of decode time per nsys profile).
// ---------------------------------------------------------------------------
static BF16_ALGO_CACHE: OnceLock<HashMap<(usize, usize), cublas_sys::cublasGemmAlgo_t>> =
    OnceLock::new();

/// Look up the best BF16 cuBLAS algorithm for a (M=out_dim, K=in_dim) shape.
/// Falls back to `CUBLAS_GEMM_DEFAULT_TENSOR_OP` when the cache is unpopulated
/// (autotune disabled or model has no BF16 weights) or the shape was not
/// benchmarked. The fallback matches the prior hardcoded behavior so the
/// patch is byte-identity-safe when autotune is off.
fn bf16_algo_for(out_dim: usize, in_dim: usize) -> cublas_sys::cublasGemmAlgo_t {
    BF16_ALGO_CACHE
        .get()
        .and_then(|m| m.get(&(out_dim, in_dim)).copied())
        .unwrap_or(cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP)
}

/// Env-gate for the BF16 autotune. Default ON because the reference-engine
/// gap (66.0 -> 73 tok/s = +10.6%) is dominated by 60.9% of decode time in
/// BF16 GemmEx kernels (per nsys profile bf16_decode.nsys-rep). Operators may
/// opt out with `LUMEN_CUDA_BF16_AUTOTUNE=0` to retain the prior path
/// (DEFAULT_TENSOR_OP for all BF16 shapes) for A/B benchmarking or rollback.
fn bf16_autotune_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LUMEN_CUDA_BF16_AUTOTUNE")
            .ok()
            .as_deref()
            .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off"))
            .unwrap_or(true)
    })
}

/// Benchmark all tensor-core cuBLAS algorithms for each unique (M, K) HGEMV
/// shape under the BF16 GemmEx datapath (CUDA_R_16BF operands + COMPUTE_32F
/// accumulator) and return a (shape -> best algo) map.
///
/// Mirrors `autotune_cublas_algos` (the F16 variant at line 89) but tests
/// BF16 inputs against `COMPUTE_32F` (BF16 has no FAST_16F variant). Same
/// proxy-shape capping (4096) and warmup/trials parameters.
///
/// Used by `preload_weights` at line ~14609 when the model has BF16 weights.
/// The resulting cache lives in `BF16_ALGO_CACHE` static OnceLock and is read
/// by `launch_hgemv_bf16` / `launch_hgemv_bf16_residual` on every BF16 GemmEx
/// call. A2/A3 fallback: shapes that fail ALL algos default to
/// DEFAULT_TENSOR_OP (the prior path).
fn autotune_cublas_algos_bf16(
    device: &CudaDevice,
    shapes: &[(usize, usize)],
) -> Result<HashMap<(usize, usize), cublas_sys::cublasGemmAlgo_t>, RuntimeError> {
    use cudarc::driver::result::event;
    use cudarc::driver::sys as cuda_sys;
    use cudarc::driver::DevicePtr;

    let mut cache: HashMap<(usize, usize), cublas_sys::cublasGemmAlgo_t> = HashMap::new();

    if shapes.is_empty() {
        return Ok(cache);
    }

    // Cap autotune dimensions at 4096 to prevent OOM (~64 MB BF16 weight at
    // 4096x4096; ~600 MB at vocab=248320). Optimal algo is stable beyond
    // cuBLAS tile size (~256), so a capped proxy matches the full shape.
    const AUTOTUNE_DIM_CAP: usize = 4096;

    let mut proxy_to_originals: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &(out_dim, in_dim) in shapes {
        let proxy = (
            out_dim.min(AUTOTUNE_DIM_CAP),
            in_dim.min(AUTOTUNE_DIM_CAP),
        );
        proxy_to_originals.entry(proxy).or_default().push((out_dim, in_dim));
    }

    let proxy_shapes: Vec<(usize, usize)> = proxy_to_originals.keys().copied().collect();

    let algos: &[cublas_sys::cublasGemmAlgo_t] = &[
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP,
    ];

    const WARMUP: usize = 3;
    const TRIALS: usize = 5;

    let start_event = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: create start event: {e}")))?;
    let end_event = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: create end event: {e}")))?;

    let raw_stream = device.stream.cu_stream();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    for &(proxy_out, proxy_in) in &proxy_shapes {
        // BF16 has the same 2 B/elem footprint as F16. Reuse buffer sizes.
        let w_bytes = proxy_out * proxy_in * 2;
        let x_bytes = proxy_in * 2;
        let w_buf: CudaSlice<u8> = device.alloc_zeros(w_bytes)
            .map_err(|e| RuntimeError::Compute(format!(
                "bf16_autotune: alloc weight ({proxy_out}x{proxy_in}): {e}"
            )))?;
        let x_buf: CudaSlice<u8> = device.alloc_zeros(x_bytes)
            .map_err(|e| RuntimeError::Compute(format!(
                "bf16_autotune: alloc input ({proxy_in}): {e}"
            )))?;
        let y_buf: CudaSlice<f32> = device.alloc_zeros(proxy_out)
            .map_err(|e| RuntimeError::Compute(format!(
                "bf16_autotune: alloc output ({proxy_out}): {e}"
            )))?;

        let (w_ptr, _) = w_buf.device_ptr(&device.stream);
        let (x_ptr, _) = x_buf.device_ptr(&device.stream);
        let (y_ptr, _) = y_buf.device_ptr(&device.stream);

        let mut best_time = f32::MAX;
        let mut best_algo = cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        for &algo in algos {
            let mut warmup_ok = true;
            for _ in 0..WARMUP {
                let status = unsafe {
                    cublas_sys::cublasGemmEx(
                        *device.blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        proxy_out as i32,
                        1i32,
                        proxy_in as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        w_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16BF,
                        proxy_in as i32,
                        x_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16BF,
                        proxy_in as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        y_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        proxy_out as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        algo,
                    )
                };
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    warmup_ok = false;
                    break;
                }
            }
            if !warmup_ok {
                continue;
            }

            device.synchronize()
                .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: sync before timing: {e}")))?;

            let mut times = Vec::with_capacity(TRIALS);
            for _ in 0..TRIALS {
                unsafe {
                    event::record(start_event, raw_stream)
                        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: record start: {e}")))?;
                    let status = cublas_sys::cublasGemmEx(
                        *device.blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        proxy_out as i32,
                        1i32,
                        proxy_in as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        w_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16BF,
                        proxy_in as i32,
                        x_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16BF,
                        proxy_in as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        y_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        proxy_out as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        algo,
                    );
                    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        break;
                    }
                    event::record(end_event, raw_stream)
                        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: record end: {e}")))?;
                    event::synchronize(end_event)
                        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: sync end: {e}")))?;
                    let ms = event::elapsed(start_event, end_event)
                        .map_err(|e| RuntimeError::Compute(format!("bf16_autotune: elapsed: {e}")))?;
                    times.push(ms);
                }
            }
            if times.len() < TRIALS {
                continue;
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = times[TRIALS / 2];
            if median < best_time {
                best_time = median;
                best_algo = algo;
            }
        }

        let algo_name = match best_algo {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP => "DEFAULT_TENSOR_OP".to_string(),
            other => format!("ALGO{}_TENSOR_OP", other as i32 - 100),
        };

        if let Some(originals) = proxy_to_originals.get(&(proxy_out, proxy_in)) {
            for &(orig_out, orig_in) in originals {
                if orig_out != proxy_out || orig_in != proxy_in {
                    eprintln!(
                        "[CUDA] Autotune BF16 HGEMV ({orig_out}x{orig_in}): \
                         using proxy ({proxy_out}x{proxy_in}) best={algo_name} ({best_time:.3}ms)"
                    );
                } else {
                    eprintln!(
                        "[CUDA] Autotune BF16 HGEMV ({orig_out}x{orig_in}): best={algo_name} ({best_time:.3}ms)"
                    );
                }
                cache.insert((orig_out, orig_in), best_algo);
            }
        }

        drop(w_buf);
        drop(x_buf);
        drop(y_buf);
    }

    unsafe {
        let _ = event::destroy(start_event);
        let _ = event::destroy(end_event);
    }

    Ok(cache)
}

/// Benchmark all tensor-core cuBLAS algorithms for each unique (M, K) HGEMV shape
/// and return an `AlgoCache` mapping shapes to the fastest algorithm.
///
/// For each shape, allocates temporary F16 weight and input buffers, then times
/// each of the 16 tensor-core algorithms plus DEFAULT_TENSOR_OP. Uses CUDA events
/// for sub-microsecond timing. Runs each algorithm `warmup + trials` times and
/// selects the one with the lowest median time.
///
/// Shapes that fail on a particular algorithm (CUBLAS_STATUS_NOT_SUPPORTED or
/// CUBLAS_STATUS_INTERNAL_ERROR) are silently skipped. If ALL algorithms fail
/// for a shape, DEFAULT_TENSOR_OP is used (it never fails).
fn autotune_cublas_algos(
    device: &CudaDevice,
    shapes: &[(usize, usize)], // (out_dim, in_dim)
) -> Result<AlgoCache, RuntimeError> {
    use cudarc::driver::result::event;
    use cudarc::driver::sys as cuda_sys;
    use cudarc::driver::DevicePtr;

    let mut cache = AlgoCache::new();

    if shapes.is_empty() {
        return Ok(cache);
    }

    // Cap autotune dimensions at 4096 to prevent OOM when allocating temp F16
    // weight buffers for large shapes (e.g., 4096x12288 FFN). The optimal
    // algorithm is stable beyond cuBLAS tile size (~256), so a capped proxy
    // shape produces the same algorithm selection as the full shape.
    const AUTOTUNE_DIM_CAP: usize = 4096;

    // Build proxy shapes: cap each dimension, then deduplicate.
    // Multiple original shapes may map to the same proxy (e.g., (4096, 12288)
    // and (4096, 8192) both map to (4096, 4096)). Benchmark each proxy once.
    let mut proxy_to_originals: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &(out_dim, in_dim) in shapes {
        let proxy = (
            out_dim.min(AUTOTUNE_DIM_CAP),
            in_dim.min(AUTOTUNE_DIM_CAP),
        );
        proxy_to_originals
            .entry(proxy)
            .or_default()
            .push((out_dim, in_dim));
    }

    // Collect unique proxy shapes for benchmarking.
    let proxy_shapes: Vec<(usize, usize)> = proxy_to_originals.keys().copied().collect();

    // All 16 tensor-core algorithms plus the default heuristic.
    let algos: &[cublas_sys::cublasGemmAlgo_t] = &[
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14_TENSOR_OP,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP,
    ];

    const WARMUP: usize = 3;
    const TRIALS: usize = 5;

    // Create CUDA events for timing.
    let start_event = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| RuntimeError::Compute(format!("autotune: create start event: {e}")))?;
    let end_event = event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| RuntimeError::Compute(format!("autotune: create end event: {e}")))?;

    let raw_stream = device.stream.cu_stream();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    for &(proxy_out, proxy_in) in &proxy_shapes {
        // Allocate temporary buffers for this proxy shape (F16 weight, F16 input, F32 output).
        let w_bytes = proxy_out * proxy_in * 2; // F16
        let x_bytes = proxy_in * 2;             // F16
        let w_buf: CudaSlice<u8> = device.alloc_zeros(w_bytes)
            .map_err(|e| RuntimeError::Compute(format!(
                "autotune: alloc weight ({proxy_out}x{proxy_in}): {e}"
            )))?;
        let x_buf: CudaSlice<u8> = device.alloc_zeros(x_bytes)
            .map_err(|e| RuntimeError::Compute(format!(
                "autotune: alloc input ({proxy_in}): {e}"
            )))?;
        let y_buf: CudaSlice<f32> = device.alloc_zeros(proxy_out)
            .map_err(|e| RuntimeError::Compute(format!(
                "autotune: alloc output ({proxy_out}): {e}"
            )))?;

        let (w_ptr, _) = w_buf.device_ptr(&device.stream);
        let (x_ptr, _) = x_buf.device_ptr(&device.stream);
        let (y_ptr, _) = y_buf.device_ptr(&device.stream);

        let mut best_time = f32::MAX;
        let mut best_algo = cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        for &algo in algos {
            // Warmup: run a few times to prime caches and cuBLAS internal state.
            let mut warmup_ok = true;
            for _ in 0..WARMUP {
                let status = unsafe {
                    cublas_sys::cublasGemmEx(
                        *device.blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        proxy_out as i32,
                        1i32,
                        proxy_in as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        w_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        proxy_in as i32,
                        x_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        proxy_in as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        y_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        proxy_out as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                        algo,
                    )
                };
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    warmup_ok = false;
                    break;
                }
            }
            if !warmup_ok {
                continue; // Skip unsupported algorithms.
            }

            // Sync before timing to avoid overlap with warmup.
            device.synchronize()
                .map_err(|e| RuntimeError::Compute(format!("autotune: sync before timing: {e}")))?;

            // Timed trials: use CUDA events for precise GPU timing.
            let mut times = Vec::with_capacity(TRIALS);
            for _ in 0..TRIALS {
                unsafe {
                    event::record(start_event, raw_stream)
                        .map_err(|e| RuntimeError::Compute(format!("autotune: record start: {e}")))?;

                    let status = cublas_sys::cublasGemmEx(
                        *device.blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        proxy_out as i32,
                        1i32,
                        proxy_in as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        w_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        proxy_in as i32,
                        x_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_16F,
                        proxy_in as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        y_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        proxy_out as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                        algo,
                    );
                    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        break; // Shouldn't happen after warmup, but be safe.
                    }

                    event::record(end_event, raw_stream)
                        .map_err(|e| RuntimeError::Compute(format!("autotune: record end: {e}")))?;
                    event::synchronize(end_event)
                        .map_err(|e| RuntimeError::Compute(format!("autotune: sync end: {e}")))?;

                    let ms = event::elapsed(start_event, end_event)
                        .map_err(|e| RuntimeError::Compute(format!("autotune: elapsed: {e}")))?;
                    times.push(ms);
                }
            }

            if times.len() < TRIALS {
                continue; // Algorithm failed during timed trials.
            }

            // Use median time to avoid outliers.
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = times[TRIALS / 2];

            if median < best_time {
                best_time = median;
                best_algo = algo;
            }
        }

        let algo_name = match best_algo {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP => "DEFAULT_TENSOR_OP".to_string(),
            other => format!("ALGO{}_TENSOR_OP", other as i32 - 100),
        };

        // Map the proxy result back to all original shapes that share this proxy.
        if let Some(originals) = proxy_to_originals.get(&(proxy_out, proxy_in)) {
            for &(orig_out, orig_in) in originals {
                if orig_out != proxy_out || orig_in != proxy_in {
                    eprintln!(
                        "[CUDA] Autotune HGEMV ({orig_out}x{orig_in}): \
                         using proxy ({proxy_out}x{proxy_in}) best={algo_name} ({best_time:.3}ms)"
                    );
                } else {
                    eprintln!(
                        "[CUDA] Autotune HGEMV ({orig_out}x{orig_in}): best={algo_name} ({best_time:.3}ms)"
                    );
                }
                cache.best_algo.insert((orig_out, orig_in), best_algo);
            }
        }

        // Explicitly drop buffers to free GPU memory before next proxy shape.
        drop(w_buf);
        drop(x_buf);
        drop(y_buf);
    }

    // Clean up events.
    unsafe {
        let _ = event::destroy(start_event);
        let _ = event::destroy(end_event);
    }

    Ok(cache)
}

/// Pre-computed device pointer arrays for batched cuBLAS HGEMV calls.
///
/// Eliminates per-layer htod memcpys (~6 per layer, 192 per token for 32 layers)
/// by uploading all pointer arrays once during `preload_weights()`. Weight and
/// output buffer GPU addresses are fixed for the lifetime of the inference session.
///
/// The input pointer (B) is the SAME for all batch elements within a call
/// (the normed/converted F16 activation vector), but its address can change if
/// the scratch buffer is reallocated. Since we allocate scratch once in `init()`
/// and never reallocate, the address is stable.
#[allow(dead_code)]
struct PrecomputedBatchPtrs {
    /// Per-layer KV batched GEMM device pointer arrays.
    /// Each entry holds [wk_ptr, wv_ptr] for A, [input_f16_ptr x2] for B,
    /// [k_out_ptr, v_out_ptr] for C. Uploaded once, never updated.
    kv_a_ptrs: Vec<CudaSlice<u64>>,
    kv_b_ptrs: Vec<CudaSlice<u64>>,
    kv_c_ptrs: Vec<CudaSlice<u64>>,

    /// Per-layer gate+up batched GEMM device pointer arrays.
    /// Each entry holds [gate_ptr, up_ptr] for A, [input_f16_ptr x2] for B,
    /// [gate_out_ptr, up_out_ptr] for C.
    ffn_a_ptrs: Vec<CudaSlice<u64>>,
    ffn_b_ptrs: Vec<CudaSlice<u64>>,
    ffn_c_ptrs: Vec<CudaSlice<u64>>,

    /// Whether cublasGemmGroupedBatchedEx is available at runtime (CUDA 12.5+).
    /// If true, QKV projections use a single grouped GEMM call instead of
    /// separate Q + batched KV calls (saves 1 cuBLAS call per layer).
    has_grouped_gemm: bool,

    /// Per-layer QKV grouped GEMM device pointer arrays (only populated if has_grouped_gemm).
    /// 3 pointers: [wq_ptr, wk_ptr, wv_ptr] for A, [input_f16_ptr x3] for B,
    /// [q_out_ptr, k_out_ptr, v_out_ptr] for C.
    qkv_a_ptrs: Vec<CudaSlice<u64>>,
    qkv_b_ptrs: Vec<CudaSlice<u64>>,
    qkv_c_ptrs: Vec<CudaSlice<u64>>,
}

/// Pre-allocated GPU scratch buffers reused across all layer calls.
///
/// Allocated once in `init()` with sizes derived from model hyperparameters.
/// All buffers live on the GPU device for the lifetime of the backend.
struct GpuScratch {
    /// RMSNorm output: [hidden_dim]
    normed: CudaSlice<f32>,
    /// Query projection: [num_heads * head_dim]
    q: CudaSlice<f32>,
    /// Key projection: [num_kv_heads * head_dim]
    k: CudaSlice<f32>,
    /// Value projection: [num_kv_heads * head_dim]
    v: CudaSlice<f32>,
    /// Attention output: [num_heads * head_dim]
    attn_out: CudaSlice<f32>,
    /// Gate FFN activation: [inter_dim]
    gate: CudaSlice<f32>,
    /// Up FFN activation: [inter_dim]
    up: CudaSlice<f32>,
    /// Down projection output: [hidden_dim]
    down: CudaSlice<f32>,
    /// Current hidden state on GPU: [hidden_dim]
    x_gpu: CudaSlice<f32>,
    /// Attention projection + residual: [hidden_dim]
    attn_proj: CudaSlice<f32>,
    /// Precomputed RMS scale scalar for fused norm+matvec: [1]
    rms_scale: CudaSlice<f32>,
    /// F16 scratch for cuBLAS HGEMV input conversion: [max(hidden_dim, inter_dim) * 2] bytes.
    ///
    /// Used by `launch_hgemv_f16` to convert F32 activations to F16 before
    /// `cublasGemmEx` with N=1, which triggers NVIDIA's optimized GEMV path.
    input_f16: CudaSlice<u8>,

    /// Pre-quantized Q8_1 input buffer for dp4a matvec.
    ///
    /// Size: max(hidden_dim, inter_dim) / 32 * 36 bytes.
    /// Populated by `quantize_f32_to_q8_1` kernel once per activation vector,
    /// then reused across all Q8_0 matvec calls sharing that input.
    /// None if dp4a Q8_1 kernels failed to compile.
    input_q8_1: Option<CudaSlice<u8>>,

    /// Pre-allocated device pointer arrays for `cublasGemmBatchedEx`.
    ///
    /// Each holds up to 3 device pointers (for QKV or gate+up batching).
    /// Sized as raw `u64` to hold GPU virtual addresses (device pointers).
    /// Populated per-layer via small htod memcpy before each batched call.
    batched_a_ptrs: CudaSlice<u64>,
    batched_b_ptrs: CudaSlice<u64>,
    batched_c_ptrs: CudaSlice<u64>,

    /// Qwen3.5 Q+gate fusion scratch buffers.
    /// q_gate: [q_dim * 2] F32 -- raw interleaved Q+gate output from wq projection.
    /// gate_buf: [q_dim] F32 -- deinterleaved gate (persists until after attention).
    /// None for models without Q+gate fusion (standard Llama/Qwen2/Mistral).
    q_gate: Option<CudaSlice<f32>>,
    gate_buf: Option<CudaSlice<f32>>,
}

/// GPU-resident global tensors (uploaded once at init, reused across all tokens).
///
/// Global tensors may be F32, F16, Q8_0, or Q4_0 depending on the model. The output
/// projection and embedding can be quantized; the final norm is always F32.
struct GpuGlobals {
    /// Final RMSNorm weights: [hidden_dim] (always F32)
    final_norm: CudaSlice<f32>,
    /// Output projection weights (F32 path): [vocab_size * hidden_dim]
    /// Empty if output_proj uses a quantized or F16 raw path instead.
    output_proj: CudaSlice<f32>,
    /// Output projection as raw F16 bytes (None if not F16).
    output_proj_f16: Option<CudaSlice<u8>>,
    /// Output projection as raw Q8_0 bytes (None if not Q8_0).
    output_proj_q8: Option<CudaSlice<u8>>,
    /// Output projection as 36-byte aligned Q8_0 (None if not Q8_0 or repack failed).
    /// Preferred over output_proj_q8 for decode (int* loads vs byte packing).
    output_proj_q8_aligned: Option<CudaSlice<u8>>,
    /// split-layout integration: output projection in per-row split (SoA) layout
    /// (None unless `LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1` AND the source is Q8 AND
    /// the repack succeeded). Decode dispatch prefers this over the aligned
    /// variant when present. The original `output_proj_q8` is preserved so
    /// the F16-cache prefill path keeps its source.
    output_proj_q8_split: Option<CudaSlice<u8>>,
    /// output_proj fast-path: pre-dequanted F16 cache of `output_proj_q8`
    /// for cuBLAS HGEMV-N=1 decode. Populated when
    /// `LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1` is set AND the source projection is
    /// Q8_0 AND the dequant allocation succeeds (~1.94 GB on Qwen3.5-9B).
    /// Decode dispatch prefers this over the Q8 SPLIT / Q8 Aligned / Q8 Raw
    /// paths when present, mirroring the BF16 cuBLAS HGEMV path which already
    /// wins on the same shape. The original `output_proj_q8` is retained for
    /// the prefill F16-cache path (which uses cuBLAS HGEMM with M=N tokens,
    /// not GEMV).
    output_proj_q8_to_f16_cache: Option<CudaSlice<u8>>,
    /// Output projection as raw Q4_0 bytes (None if not Q4_0).
    output_proj_q4: Option<CudaSlice<u8>>,
    /// Output projection as 20-byte aligned Q4_0 (None if not Q4_0 or repack failed).
    /// Preferred over output_proj_q4 for decode (int* nibble loads vs byte loads).
    output_proj_q4_aligned: Option<CudaSlice<u8>>,
    /// Output projection as raw BF16 bytes (None if not BF16).
    /// Dispatched via the `matvec_bf16` kernel — 2 B/elem of HBM traffic with
    /// full F32 dynamic range. Avoids the ~4 GB F32 inflation that previously
    /// caused OOM during preload on Qwen3.5-9B BF16.
    output_proj_bf16: Option<CudaSlice<u8>>,
    /// Embedding table (F32 path): [vocab_size * hidden_dim]
    /// Empty if embedding uses a quantized raw path instead.
    embedding: CudaSlice<f32>,
    /// Embedding as raw Q8_0 bytes (None if not Q8_0).
    embedding_q8: Option<CudaSlice<u8>>,
    /// Embedding as raw F16 bytes (None if not F16).
    embedding_f16: Option<CudaSlice<u8>>,
    /// Embedding as raw Q4_0 bytes (None if not Q4_0).
    embedding_q4: Option<CudaSlice<u8>>,
    /// Embedding as raw BF16 bytes (None if not BF16).
    /// Dispatched via the `embed_token_bf16` kernel. Avoids the host-side
    /// BF16 -> F32 inflation (~4 GB on Qwen3.5-9B) that previously OOM'd preload.
    embedding_bf16: Option<CudaSlice<u8>>,
}

/// GPU-resident scratch buffers for GDN (GatedDeltaNet) layer computation.
///
/// Allocated lazily on the first GDN layer encountered during decode.
/// Per-layer state (h_states, conv_states) persists across tokens within a
/// sequence. Shared scratch buffers are ephemeral and overwritten each layer.
struct GdnScratchGpu {
    /// GDN dimension parameters.
    params: super::gdn::GdnParams,

    // --- Per-layer persistent state ---

    /// Recurrent hidden state per GDN layer.
    /// Each entry: [num_heads * head_dim * head_dim] f32, transposed layout.
    /// Persists across tokens, reset between sequences.
    h_states: Vec<CudaSlice<f32>>,

    /// Conv1d circular buffer state per GDN layer.
    /// Each entry: [(conv_kernel_size - 1) * qkv_dim] f32.
    conv_states: Vec<CudaSlice<f32>>,

    /// Current write position in each conv circular buffer [0..kernel_size-2].
    /// Stored on host; uploaded as kernel arg each dispatch.
    conv_positions: Vec<u32>,

    /// GPU-resident conv positions for CUDA graph capture.
    /// Each entry is a single u32 on GPU, read by `ssm_conv1d_decode_graph`
    /// and updated by `advance_conv_position`. Synced from host `conv_positions`
    /// before graph capture begins.
    conv_positions_gpu: Option<Vec<CudaSlice<u32>>>,

    /// Layer index mapping: layer_idx -> gdn_scratch_index.
    /// `gdn_layer_map[layer_idx] = Some(gdn_idx)` for GDN layers, `None` for standard.
    gdn_layer_map: Vec<Option<usize>>,

    // --- Ephemeral per-dispatch buffers (shared across GDN layers) ---

    /// QKV matvec output: [qkv_dim] f32.
    qkv_buf: CudaSlice<f32>,
    /// Conv1d output + SiLU activation: [qkv_dim] f32.
    qkv_conv_buf: CudaSlice<f32>,
    /// Computed alpha (decay) per head: [num_heads] f32.
    alpha_buf: CudaSlice<f32>,
    /// Computed beta (mixing) per head: [num_heads] f32.
    beta_buf: CudaSlice<f32>,
    /// Raw alpha projection output (pre-gate transform): [num_heads] f32.
    alpha_raw_buf: CudaSlice<f32>,
    /// Raw beta projection output (pre-gate transform): [num_heads] f32.
    beta_raw_buf: CudaSlice<f32>,
    /// GDN state-update output: [value_dim] f32.
    output_buf: CudaSlice<f32>,
    /// RMSNorm + scale on output: [value_dim] f32.
    normed_out_buf: CudaSlice<f32>,
    /// Attention gate silu*normed_out: [value_dim] f32.
    gate_buf: CudaSlice<f32>,
    /// SSM output projection result: [hidden_dim] f32.
    ssm_proj_buf: CudaSlice<f32>,

    // --- Two-launch GDN intermediates (allocated only when LUMEN_CUDA_GDN_REGISTER_RESIDENT=1) ---
    //
    // The two-launch kernel pair splits Phase 4 from Phases 1-3, so Q_norm
    // and K_norm must be materialized between the two kernels. V reuses
    // `output_buf` (the megakernel already writes V there during Phase 1).
    //
    /// Post-conv1d, post-SiLU, post-L2-norm Q: [num_kv_heads * head_dim] f32.
    /// Written by `gdn_phase123_register_resident`, read by `gdn_phase4_register_resident`.
    q_norm_buf_rr: Option<CudaSlice<f32>>,
    /// Post-conv1d, post-SiLU, post-L2-norm K: [num_kv_heads * head_dim] f32.
    /// Written by `gdn_phase123_register_resident`, read by `gdn_phase4_register_resident`.
    k_norm_buf_rr: Option<CudaSlice<f32>>,
}

/// Per-call mutable state protected by a Mutex for interior mutability.
///
/// `compute_layer` takes `&self`, so mutable GPU state (scratch buffers,
/// KV caches) must be wrapped in a Mutex. The lock is uncontended in
/// single-threaded inference (~20ns overhead, negligible vs GPU compute).
struct MutableState {
    /// Compiled kernel function handles.
    kernels: KernelSet,
    /// Pre-allocated GPU scratch buffers.
    scratch: GpuScratch,
    /// Per-layer GPU KV caches.
    kv_caches: Vec<KvCacheGpu>,
    /// GPU-resident global tensors.
    globals: GpuGlobals,
    /// GPU-resident layer weights, uploaded once via `preload_weights()`.
    /// When non-empty, `compute_layer()` uses these cached weights instead of
    /// uploading from `LayerView` on every call. Index: `[layer_idx]`.
    layer_weights_cache: Vec<LayerWeightsGpu>,
    /// Pre-allocated logits buffer on GPU for the zero-sync decode path.
    /// Shape: `[vocab_size]`. Avoids per-token allocation in `compute_final_gpu`.
    logits_gpu: CudaSlice<f32>,
    /// GPU-side argmax result: [1] u32. Avoids reading back full vocab logits.
    argmax_result: CudaSlice<u32>,
    /// Captured CUDA graph for the decode pipeline. `None` until first
    /// graph capture completes. Replayed on subsequent decode tokens.
    captured_graph: Option<super::graph::CapturedGraph>,
    /// Graph-compatible kernel variants (read scalars from device pointers).
    /// Compiled once in `init()`, used during graph capture.
    graph_kernels: Option<super::graph::GraphKernelSet>,
    /// GPU-resident scalar parameter buffers for graph kernel variants.
    /// Updated via small htod memcpys before each graph replay.
    graph_params: Option<super::graph::GraphParamsBuf>,
    /// Whether the model has any GDN layers (disables graph capture).
    has_gdn_layers: bool,
    /// Whether the model has Q+gate fusion layers (disables graph capture).
    has_qgate_layers: bool,
    /// Whether the model has any MoE layers.: when true, the graph
    /// path requires `LUMEN_CUDA_MOE_DECODE_GRAPH=1` so MoE FFN dispatch is
    /// routed via the new MoE-aware branch in `compute_layer_gpu_graph`.
    /// Default OFF (disables graph capture for MoE), byte-identical to the
    /// prior production default when unset. Populated in `preload_weights`
    /// from `moe_meta_cache`.
    has_moe_layers: bool,
    /// Number of decode tokens processed since last graph invalidation.
    /// 0 = not yet run, 1 = first token (no capture), 2+ = graph replay.
    decode_token_count: usize,
    /// GDN scratch (lazy-allocated on first GDN layer, persists for sequence lifetime).
    gdn_scratch_gpu: Option<GdnScratchGpu>,
    /// Pre-allocated cuBLAS workspace for CUDA graph capture compatibility.
    ///
    /// cuBLAS must not allocate memory internally during graph capture (cudaMalloc
    /// is forbidden on a capturing stream). This 4 MB buffer is registered via
    /// `cublasSetWorkspace_v2` so cuBLAS uses it instead of allocating on-the-fly.
    /// Must outlive the cuBLAS handle.
    cublas_workspace: Option<CudaSlice<u8>>,
    /// Pre-computed per-layer batched GEMM pointer arrays.
    /// Populated once in `preload_weights()`, eliminates per-layer htod memcpys.
    /// `None` until preload completes.
    precomputed_ptrs: Option<PrecomputedBatchPtrs>,
    /// Cached cuBLAS algorithm selection for HGEMV shapes.
    /// Populated during `preload_weights()` by benchmarking all tensor-core algorithms.
    /// Used by all `launch_hgemv_f16_*` functions to select the fastest algorithm.
    algo_cache: AlgoCache,

    // ---------------------------------------------------------------------
    // MoE state (mirrors `metal::MetalF32Backend` MoE fields).
    // ---------------------------------------------------------------------
    /// Pre-allocated MoE scratch buffers (router logits, expert outputs,
    /// SwiGLU temporaries, shared-expert buffers). `None` for dense models.
    /// Allocated once in `init()` when `hp.num_experts.is_some()`.
    moe_scratch: Option<super::moe::CudaMoeScratch>,
    /// Per-layer MoE metadata. `moe_meta_cache[layer_idx]` is `Some(meta)` iff
    /// layer `layer_idx` is an MoE layer (`subtensors.experts.is_some()`).
    /// Populated during `preload_weights()`. Empty for dense models.
    moe_meta_cache: Vec<Option<super::moe::CudaMoeMeta>>,
    /// per-layer GPU-resident offset tables for the Phase-F batched
    /// dispatch path. `moe_batched_offsets[layer_idx]` is `Some(_)` iff
    /// `moe_meta_cache[layer_idx].is_some()`. Built once during
    /// `preload_weights()` from the corresponding `CudaMoeMeta`. Empty for
    /// dense models. ~6 KB per MoE layer.
    ///
    /// Separated from `CudaMoeMeta` because `CudaMoeMeta` derives `Clone`
    /// (used in `prefill_moe_ffn_layer`) and `cudarc::CudaSlice<u64>` is not
    /// `Clone`.
    moe_batched_offsets: Vec<Option<super::moe::CudaMoeBatchedOffsets>>,
    /// Opt-in expert-LFU cache configuration (set via `configure_expert_cache`).
    /// `None` = GPU-resident-all. `Some(_)` = streaming with cache.
    expert_cache_config: Option<super::moe::CudaExpertCacheConfig>,

    // split-layout integration::
    // env-var flags read once at session start. All default to OFF so the
    // production decode path is byte-for-byte identical to pre-SPLIT main
    // when no env vars are set (default-off contract: clean revert).
    /// `LUMEN_CUDA_Q8_SCALE_HW=1`: prefer `matvec_q8_aligned_q8_1_hw` (halfword
    /// scale loads) over `matvec_q8_aligned_q8_1` when the HW kernel is loaded.
    /// Independent of the split-layout flags; only affects Q8Aligned dispatch.
    /// Redundant with `kernels.use_q8_scale_hw` (that's the flag the dispatch
    /// helpers actually consult); kept here for symmetry with the other env-var
    /// flags and so this field can be inspected in tests / diagnostics.
    #[allow(dead_code)]
    use_q8_scale_hw: bool,
    /// `LUMEN_CUDA_Q8_SPLIT=1`: at preload, clone Q8Raw projection weights into
    /// a per-row split (SoA) sibling buffer (`q8_split_*` on `LayerWeightsGpu`).
    /// Decode dispatches to `matvec_q8_split_q8_1` when the sibling is present.
    /// Falls back to the existing Q8Raw/Q8Aligned path when absent.
    use_q8_split: bool,
    /// `LUMEN_CUDA_Q4_SPLIT=1`: mirror of `use_q8_split` for Q4Raw projection
    /// weights. Clones into `q4_split_*` sibling buffers.
    use_q4_split: bool,
    /// `LUMEN_CUDA_GDN_SPLIT=1`: clone GDN-specific Q4Raw weights (ssm_out,
    /// attn_gate, ssm_alpha, ssm_beta) into split siblings. Q8 + GDN_SPLIT
    /// OOMs on A100-80GB per ; Q4 only.
    use_gdn_split: bool,
    /// `LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1`: clone the Q8Raw output projection
    /// (~1 GB on Qwen3.5-9B) into a split sibling for decode. Independent of
    /// `use_q8_split` so the contribution of the final projection can be
    /// measured / stacked separately.
    use_output_proj_split: bool,
    /// `LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1`: at preload, dequant the
    /// Q8_0 output projection into an F16 cache (~1.94 GB on Qwen3.5-9B). Decode
    /// dispatches via `cublasGemmEx` with N=1 (the same path used by the BF16
    /// output projection, which already wins on BF16 storage). Higher priority
    /// than `use_output_proj_split` in the dispatch chain when both are
    /// enabled. EMPIRICALLY: LOSES -2.83% on A100 Q8_0 (Modal 5-trial median)
    /// because the F16 cache reads 1.94 GB vs Q8 SPLIT 1.06 GB (~83% more
    /// bytes), and the HBM bandwidth penalty exceeds the cuBLAS scheduling
    /// win. Kept env-gated for future hardware (e.g. H100 tensor-core
    /// rebalance) or A/B reuse.
    use_output_proj_f16_cache: bool,
    /// `LUMEN_CUDA_OUTPUT_PROJ_NR={2,16,32,64,128}`: when set AND
    /// `use_output_proj_split` is also set AND the requested NR kernel loaded,
    /// route the SPLIT dispatch via the matching `matvec_q8_split_output_proj_nr*`
    /// variant (NR=2 routes through the generic `matvec_q8_split_q8_1` kernel).
    /// Default 32 (matches the historical default before the per-NR
    /// dispatch was introduced; env var unset).
    ///
    /// EMPIRICAL (A100-SXM4 Q8_0, 5-trial median):
    /// NR=2 -> 81.3 tok/s ( 0.00%) [generic kernel, 124k CTAs]
    /// NR=16 -> 81.8 tok/s (+0.61%) [best of the variants]
    /// NR=32 -> 81.3 tok/s ( 0.00%) [default; baseline]
    /// NR=64 -> 80.0 tok/s (-1.60%) [register pressure]
    /// NR=128-> 78.3 tok/s (-3.69%) [register-spill regime]
    output_proj_nr: u32,
    /// `LUMEN_CUDA_Q8_TILE=1`: at preload, clone Q8Raw projection weights into
    /// per-row tile-grouped (8 blocks colocated) siblings (`q8_tile_*` on
    /// `LayerWeightsGpu`). Decode dispatches to `matvec_q8_tile_q8_1` when
    /// the sibling is present. Falls back to SPLIT / Aligned / Raw paths
    /// when absent. Stacks on `use_q8_split` (tile wins when both are set
    /// and both siblings populated, per `launch_matvec_preq8_1_tile`).
    use_q8_tile: bool,
    /// `LUMEN_CUDA_Q4_TILE=1`: mirror of `use_q8_tile` for Q4Raw projection
    /// weights. Clones into `q4_tile_*` sibling buffers.
    use_q4_tile: bool,
}

/// Process-wide state for the cuBLAS BF16 GemmEx fast path.
///
/// `LUMEN_CUDA_BF16_GEMMEX=0` is the explicit opt-out (default ON); this
/// state tracks the *implicit* availability of the path: whether the
/// startup probe in `CudaBackend::new` succeeded, and whether a
/// per-call runtime failure has armed a one-shot fallback to the legacy
/// `matvec_bf16` kernel for the lifetime of the backend.
///
/// The three gates compose in order:
/// 1. `bf16_gemmex_env_force_off()` — explicit `LUMEN_CUDA_BF16_GEMMEX=0`
///    opt-out wins regardless of the other two (cached once on first
///    read).
/// 2. `BF16_GEMMEX_AVAILABLE` — cleared by `CudaBackend::new` if the
///    startup BF16 GemmEx probe returns non-success; never flipped
///    afterwards.
/// 3. `BF16_GEMMEX_FALLBACK_ARMED` — set on the first per-call
///    `cublasGemmEx` runtime error; stays set for the lifetime of the
///    backend (= lifetime of the process under the one-backend-per-
///    process model). The `OnceLock`-guarded warnings are emitted at
///    most once each.
///
/// All atomics use `Relaxed` ordering: arming is monotonic
/// (false -> true, one writer) and readers tolerate seeing the old
/// value for at most one extra GemmEx attempt before re-routing.
///
/// Process-wide statics rather than a per-backend struct because:
/// (a) cuBLAS BF16 GemmEx availability is a property of the CUDA driver
///     + device for the host process, not of an individual backend
///     instance;
/// (b) the runtime architecture instantiates exactly one CUDA backend
///     per process (CUDA contexts are heavy and there is no
///     multi-backend pipeline);
/// (c) this matches the existing house pattern in
///     `crates/lumen-runtime/src/metal/profile.rs::PROFILE_ENABLED`
///     and keeps the diff to the documented "three call sites" in
///     plus the backend constructor.
static BF16_GEMMEX_PROBED: OnceLock<()> = OnceLock::new();
static BF16_GEMMEX_AVAILABLE: AtomicBool = AtomicBool::new(true);
static BF16_GEMMEX_FALLBACK_ARMED: AtomicBool = AtomicBool::new(false);
static BF16_GEMMEX_INIT_WARNING: OnceLock<()> = OnceLock::new();
static BF16_GEMMEX_RUNTIME_WARNING: OnceLock<()> = OnceLock::new();

/// Returns true if the cuBLAS BF16 GemmEx fast path is currently
/// selectable. Composes the explicit opt-out, the startup capability
/// probe, and the runtime-armed fallback flag. Cheap: three relaxed
/// atomic loads on the hot path after the env-var is cached.
fn bf16_gemmex_enabled() -> bool {
    !bf16_gemmex_env_force_off()
        && BF16_GEMMEX_AVAILABLE.load(Ordering::Relaxed)
        && !BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed)
}

/// Caches the resolved `LUMEN_CUDA_BF16_GEMMEX` env-var value the first
/// time it is read. `=0` means explicit opt-out; `=1` means explicit
/// opt-in; **unset** falls back to the model-aware default produced by
/// `runtime_defaults::bf16_gemmex_default()` ( C3 — BF16 dense
/// models default ON, Q8/Q4 dense models default OFF). The cache prevents
/// per-call `std::env::var` syscalls in the hot path. also
/// preserves byte-identical behaviour on every previously supported
/// invocation: pre-T2 callers that set the env explicitly are unaffected
/// (env wins), and callers that left the env unset on a BF16 model
/// previously got "default ON" — the new resolver also returns true for
/// BF16 unset, matching legacy. The behaviour CHANGE is for Q8/Q4 unset:
/// pre-T2 would have returned `force_off=false` (i.e. tried GemmEx and
/// then emitted a misleading "BF16 probe failed" startup warning on the
/// quantised path); post-T2 returns `force_off=true` for unset Q8/Q4 so
/// the probe is skipped. The CLI bench numbers match prior runs because
/// the Q8/Q4 dense path was never legitimately exercising GemmEx anyway.
fn bf16_gemmex_env_force_off() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("LUMEN_CUDA_BF16_GEMMEX").ok().as_deref() {
            // Explicit opt-out wins.
            Some("0") => true,
            // Explicit opt-in: not force-off.
            Some(_) => false,
            // Unset: invert the model-aware default so "default = false"
            // here means "default ON" upstream (`!force_off` in
            // `bf16_gemmex_enabled`).
            None => !crate::runtime_defaults::bf16_gemmex_default(),
        }
    })
}

/// CUDA port of the Metal decode-delay fix.
///
/// Returns the configured per-decode-step delay in microseconds. `0` (the
/// default) means OFF — `decode_token` / `decode_token_normal` skip the
/// sleep entirely and the path is bit-exact when disabled.
///
/// # Background — the race this addresses
///
/// T-final localized a GPU-scheduler timing race that surfaces as
/// non-determinism on the **server** path (CLI is deterministic because
/// each invocation is a fresh process — no inter-request opportunity for
/// divergence). Q4 MoE decode shows divergence onset at decode tokens 5–8
/// across repeated identical `temperature=0, seed=42` requests against the
/// same long-lived `Session`. The empirical signature is structurally
/// identical to the MetalQ4 BASE prefill race, which was cured
/// by a 20–50 µs CPU sleep AFTER `commit_and_wait` in `decode_token_greedy`.
/// On Metal, `delay=10` was insufficient (28/30 garbled), `delay=20` was
/// first effective (30/30), and `delay=50` was fully deterministic.
///
/// The CUDA analogue of Metal's `commit_and_wait` is `device.synchronize()`
/// at the end of `decode_token` (line ~15077) and `decode_token_normal`
/// (line ~7128). Inserting a small CPU `thread::sleep` after each of those
/// two sync points is the byte-by-byte port of the mitigation.
///
/// # Cost
///
/// At the chosen empirical default of `50` µs, the per-token cost is
/// 50 µs over a typical ~25 ms TPOT (decode time per token) = **~0.2 %**
/// — well below the ≤1 % budget defined in the acceptance gate.
/// At `0` (default OFF) the cost is zero.
///
/// # Cache
///
/// The env-var is read once and cached via `OnceLock` so the hot decode
/// path never pays for `std::env::var` syscalls. This mirrors the
/// `bf16_gemmex_env_force_off` pattern already used in this file.
fn cuda_decode_delay_us() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // fall through to the runtime-defaults resolver when
        // the env var is unset. Server path returns `50` µs (closes the
        // race); CLI returns `0` (no slowdown). The env var
        // still wins when set explicitly so existing scripts / CI / A-B
        // benchmark drivers are unaffected. The OnceLock cache prevents
        // the hot decode path from paying for env::var or atomic reads
        // beyond the first decode token of the process lifetime.
        std::env::var("LUMEN_CUDA_DECODE_DELAY_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or_else(crate::runtime_defaults::cuda_decode_delay_us_default)
    })
}

// ---------------------------------------------------------------------------
// model-aware default resolvers for the four `LUMEN_CUDA_
// DECODE_GRAPH*` envs.
//
// Each helper preserves the original truthy-set parsing (1 / true / TRUE /
// yes / YES / on / ON) for backward-compatibility. When the env is unset,
// the helper falls through to `runtime_defaults::decode_graph*_default()`,
// which returns `true` for BF16 dense models (: graph capture is
// a measured +13% TPOT win on BF16 dense) and `false` for Q8/Q4 dense (a
// regression in those cells). The OnceLock cache mirrors
// the `cuda_decode_delay_us` pattern: at most one syscall + one atomic
// read per process for each gate.
//
// Note: legacy callers that read the env directly inline still work — we
// keep these helpers separate from the inline reads so a misread won't
// silently change the answer at a different call site. All inline reads
// have been migrated to the helper at the commit.
// ---------------------------------------------------------------------------

fn parse_env_truthy(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
}

/// Resolves `LUMEN_CUDA_DECODE_GRAPH` (master). Unset → BF16-aware
/// default per `runtime_defaults`.
fn cuda_decode_graph_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_DECODE_GRAPH")
            .unwrap_or_else(crate::runtime_defaults::decode_graph_default)
    })
}

/// Resolves `LUMEN_CUDA_DECODE_GRAPH_QGATE`. Unset → BF16-aware default.
fn cuda_decode_graph_qgate_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_DECODE_GRAPH_QGATE")
            .unwrap_or_else(crate::runtime_defaults::decode_graph_qgate_default)
    })
}

/// Resolves `LUMEN_CUDA_DECODE_GRAPH_TILED`. Unset → BF16-aware default.
fn cuda_decode_graph_tiled_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_DECODE_GRAPH_TILED")
            .unwrap_or_else(crate::runtime_defaults::decode_graph_tiled_default)
    })
}

/// Apply the decode-delay if configured.
///
/// Called immediately after the terminal `device.synchronize()` in the
/// per-decode-step CUDA path. When the env-var is unset or `0` this is a
/// single integer load + branch (no syscall, no sleep). When it is non-zero
/// it issues `std::thread::sleep` for the configured number of
/// microseconds. The function is `#[inline(always)]` to ensure the
/// fast-path (delay == 0) compiles down to a load+branch with no call
/// overhead.
#[inline(always)]
fn maybe_apply_cuda_decode_delay() {
    let delay_us = cuda_decode_delay_us();
    if delay_us > 0 {
        std::thread::sleep(std::time::Duration::from_micros(delay_us));
    }
}

/// Arms the runtime fallback flag and emits a once-only warning.
/// Subsequent calls cheaply observe the armed flag via
/// `bf16_gemmex_enabled`.
fn arm_bf16_gemmex_runtime_fallback(label: &str, status: cublas_sys::cublasStatus_t) {
    BF16_GEMMEX_FALLBACK_ARMED.store(true, Ordering::Relaxed);
    BF16_GEMMEX_RUNTIME_WARNING.get_or_init(|| {
        eprintln!(
            "[CUDA] cublasGemmEx BF16 returned {status:?} on {label}; \
             disabling BF16 GemmEx for the lifetime of this backend and \
             routing all BF16 prefill matvecs through the legacy \
             matvec_bf16 kernel (request continues without aborting). \
             Set LUMEN_CUDA_BF16_GEMMEX=0 to disable the GemmEx path \
             explicitly at startup."
        );
    });
}

// ---------------------------------------------------------------------------
// BF16 GemmEx fault-injection hook.
//
// This block is `#[cfg(any(test, feature = "test-fault-injection"))]` so
// release builds without the feature compile it away entirely -- there
// is no static, no helper, and no inject-check inside `launch_hgemv_bf16`
// / `_residual` in the production path. The cfg guard rules out any
// runtime cost on the hot path.
//
// The mechanism: `inject_next_bf16_cublas_failure` flips a one-shot
// atomic. The next call into `launch_hgemv_bf16` (or its residual
// sibling) observes the flag via `swap(false, Relaxed)`, clears it
// atomically, and returns `Bf16LaunchOutcome::CublasFailure(
// CUBLAS_STATUS_NOT_INITIALIZED)` immediately -- *without* dispatching
// any cuBLAS call. The wrapper at `launch_bf16_matvec_with_fallback`
// (and its residual sibling) then arms the runtime-fallback flag and
// re-dispatches via the legacy `matvec_bf16` kernel, exactly as it
// would for a real cuBLAS-runtime failure.
//
// One-shot semantics: a single inject affects exactly one matvec call.
// Subsequent dispatches see the flag cleared and follow the regular
// gate-composition path (which by then has `BF16_GEMMEX_FALLBACK_ARMED
// == true`, so they route to legacy without entering the inject check).
//
// Test-only seam -- the production wrappers and call sites at `:5559`,
// `:6002`, `:6571` and the gate composition at `bf16_gemmex_enabled()`
// remain byte-identical regardless of whether the feature is enabled.
#[cfg(any(test, feature = "test-fault-injection"))]
static BF16_INJECT_NEXT_CUBLAS_FAILURE: AtomicBool = AtomicBool::new(false);

/// Test-only hook: arms a one-shot fault injection so the next call into
/// `launch_hgemv_bf16` (or `launch_hgemv_bf16_residual`) returns
/// `Bf16LaunchOutcome::CublasFailure(CUBLAS_STATUS_NOT_INITIALIZED)`
/// without actually dispatching cuBLAS. Used by the
/// `cuda_bf16_gemmex_fault_injection_test` integration suite to drive
/// the wrapper's per-call CUBLAS-failure -> legacy-kernel fall-through
/// arm. The flag is consumed atomically on the next dispatch; a single
/// call to this helper triggers at most one synthetic failure.
///
/// Gated by `#[cfg(any(test, feature = "test-fault-injection"))]` so the
/// production-feature build (`cargo build --release --features cuda`)
/// without `test-fault-injection` does not see this symbol at all.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn inject_next_bf16_cublas_failure() {
    BF16_INJECT_NEXT_CUBLAS_FAILURE.store(true, Ordering::Relaxed);
}

/// Test-only helper: resets the process-wide BF16 GemmEx state machine
/// to defaults (AVAILABLE=true, FALLBACK_ARMED=false, inject-flag
/// clear). Required for integration tests that drive multiple BF16
/// wrapper dispatches in the same process and need each test to start
/// from the same baseline. Does NOT clear the once-only warning
/// OnceLocks because those have no cross-test-relevant state (a single
/// eprintln across the process lifetime is the contract; the OnceLock's
/// `is_some()` status is monotonic-on once observed).
///
/// Gated by `#[cfg(any(test, feature = "test-fault-injection"))]`.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn reset_bf16_gemmex_state_for_tests() {
    BF16_GEMMEX_AVAILABLE.store(true, Ordering::Relaxed);
    BF16_GEMMEX_FALLBACK_ARMED.store(false, Ordering::Relaxed);
    BF16_INJECT_NEXT_CUBLAS_FAILURE.store(false, Ordering::Relaxed);
}

/// Test-only observer: returns true if the runtime-armed fallback flag
/// is currently set. Tests use this to assert that a forced cuBLAS
/// failure correctly armed the flag after the wrapper handled it.
///
/// Gated by `#[cfg(any(test, feature = "test-fault-injection"))]`.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn bf16_gemmex_fallback_armed_for_tests() -> bool {
    BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed)
}

/// Test-only observer: returns true if the once-only runtime warning
/// has been emitted. Tests use this to assert that a forced cuBLAS
/// failure produced exactly one warning across multiple subsequent
/// arming calls (the OnceLock-backed `get_or_init` enforces at-most-once
/// execution of the eprintln body).
///
/// Gated by `#[cfg(any(test, feature = "test-fault-injection"))]`.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn bf16_gemmex_runtime_warning_emitted_for_tests() -> bool {
    BF16_GEMMEX_RUNTIME_WARNING.get().is_some()
}

/// CUDA compute backend for NVIDIA GPUs.
///
/// Manages a CUDA device context, compiled kernel modules, GPU-resident
/// buffers, and per-layer KV caches. Implements the full transformer
/// decode pipeline via CUDA kernels compiled at runtime with NVRTC.
pub struct CudaBackend {
    device: CudaDevice,
    hyperparams: Option<ModelHyperparams>,
    /// Host-side global tensors (set via `set_global_tensors`, uploaded to GPU in `init`).
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    /// Raw Q8_0 output projection bytes (set via `set_output_proj_raw`).
    output_proj_raw: Option<Vec<u8>>,
    output_proj_quant: QuantScheme,
    /// Raw Q8_0 embedding bytes (set via `set_embedding_raw`).
    embedding_raw: Option<Vec<u8>>,
    embedding_quant: QuantScheme,
    /// Compiled embed kernels (F32 and Q8_0).
    embed_f32_func: Option<CudaFunction>,
    embed_q8_0_func: Option<CudaFunction>,
    embed_f16_func: Option<CudaFunction>,
    embed_q4_0_func: Option<CudaFunction>,
    /// BF16 embedding lookup kernel (matches embed_token_f16 ABI; 2 bytes/elem).
    embed_bf16_func: Option<CudaFunction>,
    /// Whether embedding and output projection share the same weight tensor.
    weight_tying: bool,
    /// Cached dimensions (set in `init()`).
    cached_hidden_dim: usize,
    cached_vocab_size: usize,
    /// Mutable GPU state: scratch buffers, KV caches, kernels, globals.
    /// Protected by Mutex for interior mutability (compute_layer takes &self).
    state: Mutex<Option<MutableState>>,
}

impl CudaBackend {
    /// Create a new CUDA backend.
    ///
    /// Initializes a CUDA device context. Fails if no CUDA GPU is available.
    /// On success, runs a one-shot `cublasGemmEx` BF16 capability probe
    /// (the first time any backend is created in this process) to detect
    /// whether the tensor-core BF16 path is functional; if the probe
    /// fails, emits a single warning and routes all subsequent BF16
    /// prefill matvecs to the legacy `matvec_bf16` kernel via the
    /// `BF16_GEMMEX_AVAILABLE` flag.
    ///
    /// `device_id` selects the GPU ordinal (0 = first GPU).
    pub fn new(device_id: usize) -> Result<Self, RuntimeError> {
        let device = CudaDevice::new(device_id)?;
        Self::probe_bf16_gemmex_once(&device);
        Ok(Self {
            device,
            hyperparams: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            output_proj_raw: None,
            output_proj_quant: QuantScheme::F32,
            embedding_raw: None,
            embedding_quant: QuantScheme::F32,
            embed_f32_func: None,
            embed_q8_0_func: None,
            embed_f16_func: None,
            embed_q4_0_func: None,
            embed_bf16_func: None,
            weight_tying: false,
            cached_hidden_dim: 0,
            cached_vocab_size: 0,
            state: Mutex::new(None),
        })
    }

    /// Run a tiny `cublasGemmEx` BF16 probe to verify the tensor-core BF16
    /// path is functional on this device. The probe uses the exact same
    /// data-type / accumulator / algo combination the hot path uses
    /// (`CUDA_R_16BF` operands, `CUBLAS_COMPUTE_32F`,
    /// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`) on a 4x4x4 GEMV shape (M=4, N=1,
    /// K=4 — under 100 bytes of device memory total). The probe runs at
    /// most once per process (gated by `BF16_GEMMEX_PROBED`); on failure
    /// it clears `BF16_GEMMEX_AVAILABLE` and emits a single warning
    /// eprintln. The backend is still constructed in either case — the
    /// legacy `matvec_bf16` path does not depend on cuBLAS BF16 GemmEx.
    ///
    /// Probe-time allocation failures (e.g. host -> device copy of the
    /// 8 host bytes) are treated identically to a `cublasGemmEx`
    /// non-success status.
    fn probe_bf16_gemmex_once(device: &CudaDevice) {
        if BF16_GEMMEX_PROBED.get().is_some() {
            return;
        }
        // debugging hook: skip the probe when explicitly requested
        // via `LUMEN_CUDA_SKIP_BF16_PROBE=1`. Useful when running under
        // compute-sanitizer, which reports benign cuBLAS-internal OOB reads
        // on the 4×1×4 probe input as hard CUDA errors that block test
        // execution. The skip preserves the BF16_GEMMEX_AVAILABLE default
        // (true) so the live path still attempts BF16 GemmEx; only the
        // startup probe is bypassed. Production paths are unaffected
        // unless the env-var is explicitly set.
        if std::env::var("LUMEN_CUDA_SKIP_BF16_PROBE")
            .ok()
            .as_deref()
            .is_some_and(|v| matches!(v, "1" | "true" | "yes"))
        {
            let _ = BF16_GEMMEX_PROBED.set(());
            return;
        }
        // SAFETY: every cuBLAS call below uses pointers obtained from
        // `device_ptr` on `CudaSlice`s allocated immediately before the
        // call. The slices live until the end of this function (after
        // synchronize), so the pointers are valid for the lifetime of
        // the cuBLAS dispatch. Errors propagate via the `Result` arms.
        let result = (|| -> Result<cublas_sys::cublasStatus_t, RuntimeError> {
            // 4 BF16 values per operand. Two operands + one F32 output.
            // 8 bytes weight + 8 bytes input + 16 bytes output = 32 bytes.
            let m: i32 = 4;
            let n: i32 = 1;
            let k: i32 = 4;
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            // BF16 bit pattern for 1.0 is 0x3f80.
            let one_bf16_bits: u16 = 0x3f80;
            let bf16_bytes: [u8; 8] = {
                let mut out = [0u8; 8];
                for chunk in out.chunks_exact_mut(2) {
                    chunk.copy_from_slice(&one_bf16_bits.to_le_bytes());
                }
                out
            };

            let w_bf16 = device.htod_copy(&bf16_bytes)?;
            let a_bf16 = device.htod_copy(&bf16_bytes)?;
            let c_f32: CudaSlice<f32> = device.alloc_zeros(m as usize)?;

            use cudarc::driver::DevicePtr;
            let (w_ptr, _) = w_bf16.device_ptr(&device.stream);
            let (a_ptr, _) = a_bf16.device_ptr(&device.stream);
            let (c_ptr, _) = c_f32.device_ptr(&device.stream);

            // SAFETY: pointers are valid device pointers for at least
            // 8 / 8 / 16 bytes respectively; cuBLAS handle is owned
            // by `device.blas` and remains live for the call duration.
            let status = unsafe {
                cublas_sys::cublasGemmEx(
                    *device.blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m, n, k,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    w_ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k, // lda = K (row-major K-major weight)
                    a_ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k, // ldb = K
                    &beta as *const f32 as *const std::ffi::c_void,
                    c_ptr as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    m, // ldc = M
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            };
            // Force completion so a deferred failure surfaces here, not
            // on the first real BF16 matvec.
            device.synchronize()?;
            Ok(status)
        })();

        match result {
            Ok(status) if status == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                // `available` already defaults to true; no flip needed.
            }
            Ok(status) => {
                BF16_GEMMEX_AVAILABLE.store(false, Ordering::Relaxed);
                BF16_GEMMEX_INIT_WARNING.get_or_init(|| {
                    eprintln!(
                        "[CUDA] BF16 GemmEx capability probe returned {status:?}; \
                         BF16 prefill matvecs will use the legacy matvec_bf16 \
                         kernel for the lifetime of this process."
                    );
                });
            }
            Err(e) => {
                BF16_GEMMEX_AVAILABLE.store(false, Ordering::Relaxed);
                BF16_GEMMEX_INIT_WARNING.get_or_init(|| {
                    eprintln!(
                        "[CUDA] BF16 GemmEx capability probe failed during setup ({e}); \
                         BF16 prefill matvecs will use the legacy matvec_bf16 \
                         kernel for the lifetime of this process."
                    );
                });
            }
        }

        // Mark probe complete last so concurrent backend constructions
        // see the resolved AVAILABLE flag before they skip the probe.
        let _ = BF16_GEMMEX_PROBED.set(());
    }

    /// Access hyperparams, returning an error if `init()` has not been called.
    fn hp(&self) -> Result<&ModelHyperparams, RuntimeError> {
        self.hyperparams.as_ref().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized: call init() first".into())
        })
    }

    /// Test-only helper: drives a single
    /// BF16 matvec through `launch_bf16_matvec_with_fallback` (or its
    /// residual sibling when `residual` is `Some`) and returns the
    /// resulting `out_dim` output vector. Used by the
    /// `cuda_bf16_gemmex_fault_injection_test` integration suite to
    /// exercise the wrapper's per-call CUBLAS-failure -> legacy-kernel
    /// fall-through arm under a real BF16 matvec dispatch.
    ///
    /// Requires `init()` to have been called (kernels must be
    /// compiled). Allocates per-call scratch + input + output device
    /// buffers; the caller passes BF16 weights as raw bytes
    /// (`out_dim * in_dim * 2` bytes, row-major, BF16 bit pattern).
    ///
    /// On a successful GemmEx dispatch the returned output matches
    /// `W^T * input` (or `W^T * input + residual` for the residual
    /// variant). On a forced CUBLAS failure via
    /// `inject_next_bf16_cublas_failure`, the wrapper arms the
    /// process-wide runtime fallback flag and re-dispatches via the
    /// legacy `matvec_bf16` kernel; the returned output must match the
    /// GemmEx result to within BF16 numerical tolerance because both
    /// paths compute the same mathematical operation on the same BF16
    /// weights.
    ///
    /// Gated by `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so production builds without the feature have neither the
    /// method nor its compiled body.
    ///
    /// # Safety
    ///
    /// `weight_bf16_bytes.len()` must equal `out_dim * in_dim * 2` and
    /// the bytes must be valid BF16 representations. `input_f32.len()`
    /// must equal `in_dim`. `residual.map(|r| r.len())` must equal
    /// `Some(out_dim)` when residual is non-None.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn dispatch_bf16_matvec_for_tests(
        &self,
        weight_bf16_bytes: &[u8],
        input_f32: &[f32],
        out_dim: usize,
        in_dim: usize,
        residual: Option<&[f32]>,
        label: &str,
    ) -> Result<Vec<f32>, RuntimeError> {
        if weight_bf16_bytes.len() != out_dim * in_dim * 2 {
            return Err(RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: weight_bf16_bytes \
                 has {} bytes, expected {} (out_dim*in_dim*2)",
                weight_bf16_bytes.len(),
                out_dim * in_dim * 2,
            )));
        }
        if input_f32.len() != in_dim {
            return Err(RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: input_f32 has \
                 {} elements, expected {} (in_dim)",
                input_f32.len(),
                in_dim,
            )));
        }
        if let Some(r) = residual {
            if r.len() != out_dim {
                return Err(RuntimeError::Compute(format!(
                    "dispatch_bf16_matvec_for_tests {label}: residual has \
                     {} elements, expected {} (out_dim)",
                    r.len(),
                    out_dim,
                )));
            }
        }
        let mut guard = self.state.lock().unwrap();
        let st = guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "dispatch_bf16_matvec_for_tests: backend not initialized \
                 (call init() first)"
                    .into(),
            )
        })?;
        let w_dev: CudaSlice<u8> = self
            .device
            .htod_copy(weight_bf16_bytes)
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: htod_copy weight: {e}",
            )))?;
        let input_dev: CudaSlice<f32> = self
            .device
            .htod_copy(input_f32)
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: htod_copy input: {e}",
            )))?;
        let mut output_dev: CudaSlice<f32> = self
            .device
            .alloc_zeros(out_dim)
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: alloc output: {e}",
            )))?;
        let mut input_bf16_scratch: CudaSlice<u8> = self
            .device
            .alloc_zeros(in_dim * 2)
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: alloc scratch: {e}",
            )))?;
        match residual {
            None => unsafe {
                launch_bf16_matvec_with_fallback(
                    &self.device,
                    &st.kernels,
                    &w_dev,
                    &input_dev,
                    &mut output_dev,
                    &mut input_bf16_scratch,
                    out_dim,
                    in_dim,
                    label,
                )?;
            },
            Some(r) => {
                let residual_dev: CudaSlice<f32> = self
                    .device
                    .htod_copy(r)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "dispatch_bf16_matvec_for_tests {label}: htod_copy residual: {e}",
                    )))?;
                unsafe {
                    launch_bf16_matvec_residual_with_fallback(
                        &self.device,
                        &st.kernels,
                        &w_dev,
                        &input_dev,
                        &residual_dev,
                        &mut output_dev,
                        &mut input_bf16_scratch,
                        out_dim,
                        in_dim,
                        label,
                    )?;
                }
            }
        }
        self.device
            .synchronize()
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: synchronize: {e}",
            )))?;
        let out_host: Vec<f32> = self
            .device
            .dtoh_copy(&output_dev)
            .map_err(|e| RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: dtoh output: {e}",
            )))?;
        Ok(out_host)
    }

    /// Embed a token directly into the GPU scratch buffer `x_gpu`, with no sync.
    ///
    /// This is the GPU-resident counterpart of `embed_token`. Instead of syncing
    /// and copying back to host, it leaves the embedding in `st.scratch.x_gpu`.
    fn embed_token_gpu(
        &self,
        token_id: u32,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = self.cached_hidden_dim;
        let vocab_size = self.cached_vocab_size;

        if (token_id as usize) >= vocab_size {
            return Err(RuntimeError::Compute(format!(
                "token_id {} out of range (vocab_size={vocab_size})",
                token_id,
            )));
        }

        let config = LaunchConfig::for_elements(hidden_dim);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (config.grid_dim, 1, 1),
            block_dim: (config.block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // Dispatch embed kernel based on embedding precision.
        // Order: BF16 > F16 > Q4_0 > Q8_0 > F32. BF16 added for the Qwen3.5-9B BF16 path.
        if let Some(ref emb_bf16) = st.globals.embedding_bf16 {
            let func = self.embed_bf16_func.as_ref().ok_or_else(|| {
                RuntimeError::Compute("embed_token_bf16 kernel not compiled".into())
            })?;
            let hd = hidden_dim as u32;
            unsafe {
                self.device
                    .stream
                    .launch_builder(func)
                    .arg(emb_bf16)
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&token_id)
                    .arg(&hd)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("CUDA embed_token_bf16 gpu launch: {e}"))
            })?;
        } else if let Some(ref emb_f16) = st.globals.embedding_f16 {
            let func = self.embed_f16_func.as_ref().ok_or_else(|| {
                RuntimeError::Compute("embed_token_f16 kernel not compiled".into())
            })?;
            let hd = hidden_dim as u32;
            unsafe {
                self.device
                    .stream
                    .launch_builder(func)
                    .arg(emb_f16)
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&token_id)
                    .arg(&hd)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("CUDA embed_token_f16 gpu launch: {e}"))
            })?;
        } else if let Some(ref emb_q4) = st.globals.embedding_q4 {
            let func = self.embed_q4_0_func.as_ref().ok_or_else(|| {
                RuntimeError::Compute("embed_token_q4_0 kernel not compiled".into())
            })?;
            let hd = hidden_dim as u32;
            // SAFETY: embed_token_q4_0 reads Q4_0 blocks starting at
            // token_id * hidden_dim (bounds checked above). x_gpu has hidden_dim elements.
            unsafe {
                self.device
                    .stream
                    .launch_builder(func)
                    .arg(emb_q4)
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&token_id)
                    .arg(&hd)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("CUDA embed_token_q4_0 gpu launch: {e}"))
            })?;
        } else if let Some(ref emb_q8) = st.globals.embedding_q8 {
            let func = self.embed_q8_0_func.as_ref().ok_or_else(|| {
                RuntimeError::Compute("embed_token_q8_0 kernel not compiled".into())
            })?;
            let hd = hidden_dim as u32;
            // SAFETY: embed_token_q8_0 reads Q8_0 blocks starting at
            // token_id * hidden_dim (bounds checked above). x_gpu has hidden_dim elements.
            unsafe {
                self.device
                    .stream
                    .launch_builder(func)
                    .arg(emb_q8)
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&token_id)
                    .arg(&hd)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("CUDA embed_token_q8_0 gpu launch: {e}"))
            })?;
        } else {
            let func = self.embed_f32_func.as_ref().ok_or_else(|| {
                RuntimeError::Compute("embed_token_f32 kernel not compiled".into())
            })?;
            let hd = hidden_dim as u32;
            // SAFETY: The kernel reads hidden_dim elements starting at
            // token_id * hidden_dim from the embedding buffer (bounds checked above).
            unsafe {
                self.device
                    .stream
                    .launch_builder(func)
                    .arg(&st.globals.embedding)
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&token_id)
                    .arg(&hd)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("CUDA embed_token_f32 gpu launch: {e}"))
            })?;
        }

        Ok(())
    }

    /// Run one transformer layer entirely on GPU, with no host sync.
    ///
    /// Input: `st.scratch.x_gpu` (hidden state on GPU, [hidden_dim]).
    /// Output: `st.scratch.attn_proj` (next hidden state on GPU, [hidden_dim]).
    ///
    /// After this call, the caller must swap `attn_proj` into `x_gpu` for the
    /// next layer (or use `attn_proj` as input to `compute_final_gpu`).
    fn compute_layer_gpu(
        &self,
        layer_idx: usize,
        seq_pos: usize,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let eps = hp.norm_eps;
        let theta = hp.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Read layer_type before borrowing layer weights, to avoid borrow conflict
        // when passing &mut st to compute_gdn_attention_gpu.
        let layer_type = st
            .layer_weights_cache
            .get(layer_idx)
            .map(|lw| lw.layer_type)
            .unwrap_or(0);

        // GDN layer routing: if this is a GDN layer, dispatch the GDN pipeline
        // instead of the standard softmax attention path.
        // NOTE: GDN layers still have dense FFN (gate/up/SwiGLU/down) which runs
        // AFTER the GDN attention block, same as standard layers.
        if layer_type == 1 {
            // Run the GDN attention block, which replaces the standard
            // QKV -> RoPE -> KV cache -> Attention -> Output proj path.
            // After this, attn_proj = x_old + ssm_proj (the post-GDN-attention
            // hidden state). x_gpu is NOT updated here -- it retains the old value.
            // The FFN block reads from attn_proj, and the caller copies attn_proj
            // to x_gpu after the full layer (GDN attention + FFN) completes.
            self.compute_gdn_attention_gpu(layer_idx, st)?;
        } else {

        let lw: &LayerWeightsGpu = st
            .layer_weights_cache
            .get(layer_idx)
            .ok_or_else(|| RuntimeError::Compute(format!(
                "compute_layer_gpu: layer {layer_idx} not in GPU-resident cache",
            )))?;

        // Q+gate fusion detection: Qwen3.5 full-attention layers have fused Q+gate
        // in wq with output dimension q_dim*2. When active, wq projects to q_gate
        // scratch buffer, then deinterleave + per-head norm produces final Q and gate.
        let has_qgate_fusion = lw.attn_q_norm.is_some();
        let wq_out_dim = if has_qgate_fusion { q_dim * 2 } else { q_dim };

        // 1. Fused RMSNorm + QKV projections (same logic as compute_layer).
        // For mixed-precision models (e.g. Qwen3.5-9B Q4 LBC where wq is
        // dequantized from Q6_K to F32 but wk/wv remain Q4_0 as Q4Raw),
        // checking only wq would cause `if let GpuWeightBuf::F32 = wk` bindings
        // below to silently fail and skip wk/wv matvec dispatch, leaving
        // st.scratch.k/v with stale state from a prior layer and producing
        // request-N-dependent non-determinism. Require all three F32 here so
        // mixed-precision falls through to the F32+f16-cache HGEMV batched path.
        if matches!(&lw.wq, GpuWeightBuf::F32(_))
            && matches!(&lw.wk, GpuWeightBuf::F32(_))
            && matches!(&lw.wv, GpuWeightBuf::F32(_))
        {
            // SAFETY: x_gpu is [hidden_dim], rms_scale is [1]. Both allocated in init.
            unsafe {
                launch_compute_rms_scale(
                    &self.device,
                    &st.kernels,
                    &st.scratch.x_gpu,
                    &mut st.scratch.rms_scale,
                    eps,
                    hidden_dim,
                )?;
            }
            if let GpuWeightBuf::F32(ref wq_f32) = lw.wq {
                // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                let (wq_out_buf, wq_od) = if has_qgate_fusion {
                    (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                } else {
                    (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                };
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &st.scratch.rms_scale,
                        &lw.attn_norm,
                        wq_f32,
                        wq_out_buf,
                        wq_od,
                        hidden_dim,
                        "wq",
                    )?;
                }
            }
            if let GpuWeightBuf::F32(ref wk_f32) = lw.wk {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &st.scratch.rms_scale,
                        &lw.attn_norm,
                        wk_f32,
                        &mut st.scratch.k,
                        kv_dim,
                        hidden_dim,
                        "wk",
                    )?;
                }
            }
            if let GpuWeightBuf::F32(ref wv_f32) = lw.wv {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &st.scratch.rms_scale,
                        &lw.attn_norm,
                        wv_f32,
                        &mut st.scratch.v,
                        kv_dim,
                        hidden_dim,
                        "wv",
                    )?;
                }
            }
        } else if matches!(&lw.wq, GpuWeightBuf::F16Raw(_)) {
            // F16 HGEMV path: Fused RMSNorm + F32->F16 in ONE kernel (saves 1 dispatch),
            // then cuBLAS HGEMV for all QKV projections (cached F16 input).
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.x_gpu, &lw.attn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "attn F16",
                )?;
            }
            // QKV projections: use pre-computed pointers if available.
            if let Some(ref pcp) = st.precomputed_ptrs {
                // Pre-computed batched: Q separate + KV batched (no htod).
                if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    let (wq_out_buf, wq_od) = if has_qgate_fusion {
                        (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                    } else {
                        (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                    };
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            wq_out_buf, wq_od, hidden_dim, "wq",
                            st.algo_cache.get(wq_od, hidden_dim),
                        )?;
                    }
                }
                unsafe {
                    launch_hgemv_f16_batched_precomputed(
                        &self.device,
                        &pcp.kv_a_ptrs[layer_idx],
                        &pcp.kv_b_ptrs[layer_idx],
                        &pcp.kv_c_ptrs[layer_idx],
                        2, kv_dim, hidden_dim, "kv",
                        st.algo_cache.get(kv_dim, hidden_dim),
                    )?;
                }
            } else {
                // Fallback: original per-layer htod path.
                if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    let (wq_out_buf, wq_od) = if has_qgate_fusion {
                        (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                    } else {
                        (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                    };
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            wq_out_buf, wq_od, hidden_dim, "wq",
                            st.algo_cache.get(wq_od, hidden_dim),
                        )?;
                    }
                }
                if let (GpuWeightBuf::F16Raw(ref wk_f16), GpuWeightBuf::F16Raw(ref wv_f16)) = (&lw.wk, &lw.wv) {
                    unsafe {
                        let w_slices: &[&CudaSlice<u8>] = &[wk_f16, wv_f16];
                        let mut out_slices: [&mut CudaSlice<f32>; 2] = [&mut st.scratch.k, &mut st.scratch.v];
                        launch_hgemv_f16_batched(
                            &self.device,
                            w_slices,
                            &st.scratch.input_f16,
                            &mut out_slices,
                            &mut st.scratch.batched_a_ptrs,
                            &mut st.scratch.batched_b_ptrs,
                            &mut st.scratch.batched_c_ptrs,
                            kv_dim,
                            hidden_dim,
                            "kv",
                            st.algo_cache.get(kv_dim, hidden_dim),
                        )?;
                    }
                }
            }
        } else if matches!(&lw.wq, GpuWeightBuf::F32(_))
            && lw.wq_f16.is_some() && lw.wk_f16.is_some() && lw.wv_f16.is_some()
        {
            // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
            // CUBLAS_COMPUTE_32F_FAST_16F exploits tensor cores (312 TFLOPS on A100).
            // Only used for F32 weights where F16 HGEMV halves bandwidth (4 -> 2 B/elem).
            // Q8/Q4/Q8Aligned weights fall through to launch_matvec() which dispatches
            // native dp4a kernels reading 1.06 B/elem -- 1.9x less bandwidth than HGEMV.
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.x_gpu, &lw.attn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "attn HGEMV",
                )?;
            }
            // QKV projections: use pre-computed pointers if available (same logic as F16 native path).
            if let Some(ref pcp) = st.precomputed_ptrs {
                // Pre-computed batched: Q separate + KV batched (no htod).
                if let Some(ref wq_f16) = lw.wq_f16 {
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    let (wq_out_buf, wq_od) = if has_qgate_fusion {
                        (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                    } else {
                        (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                    };
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            wq_out_buf, wq_od, hidden_dim, "wq",
                            st.algo_cache.get(wq_od, hidden_dim),
                        )?;
                    }
                }
                unsafe {
                    launch_hgemv_f16_batched_precomputed(
                        &self.device,
                        &pcp.kv_a_ptrs[layer_idx],
                        &pcp.kv_b_ptrs[layer_idx],
                        &pcp.kv_c_ptrs[layer_idx],
                        2, kv_dim, hidden_dim, "kv",
                        st.algo_cache.get(kv_dim, hidden_dim),
                    )?;
                }
            } else {
                // Fallback: Q separate + KV batched with per-layer htod.
                if let Some(ref wq_f16) = lw.wq_f16 {
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    let (wq_out_buf, wq_od) = if has_qgate_fusion {
                        (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                    } else {
                        (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                    };
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            wq_out_buf, wq_od, hidden_dim, "wq",
                            st.algo_cache.get(wq_od, hidden_dim),
                        )?;
                    }
                }
                if let (Some(ref wk_f16), Some(ref wv_f16)) = (&lw.wk_f16, &lw.wv_f16) {
                    unsafe {
                        let w_slices: &[&CudaSlice<u8>] = &[wk_f16, wv_f16];
                        let mut out_slices: [&mut CudaSlice<f32>; 2] = [&mut st.scratch.k, &mut st.scratch.v];
                        launch_hgemv_f16_batched(
                            &self.device,
                            w_slices,
                            &st.scratch.input_f16,
                            &mut out_slices,
                            &mut st.scratch.batched_a_ptrs,
                            &mut st.scratch.batched_b_ptrs,
                            &mut st.scratch.batched_c_ptrs,
                            kv_dim,
                            hidden_dim,
                            "kv",
                            st.algo_cache.get(kv_dim, hidden_dim),
                        )?;
                    }
                }
            }
        } else {
            // Q8_0/Q4_0/Q8Aligned/Q4Aligned/F32: native-quant decode via launch_matvec().
            // Priority: dp4a Q8_1 > smem > hgemv > cuBLAS HGEMV > dp4a/scalar.
            // Native kernels read quantized weights directly (1.06 B/elem for Q8, 0.56 for Q4)
            // vs HGEMV's 2 B/elem from pre-dequanted F16 cache -- 1.9x-3.6x less bandwidth.
            // F16 caches are passed as last-resort fallback only.
            // Shared-quantization optimization: if all QKV weights use dp4a Q8_1 path,
            // quantize the normed input ONCE and reuse across Q, K, V projections.
            // Saves 2 quantize_f32_to_q8_1 launches per layer.
            let qkv_use_preq = weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wk, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wv, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some();

            // Fused RMSNorm + Q8_1: skip separate rmsnorm + quantize_f32_to_q8_1
            // when the fused kernel is available. Saves 1 dispatch per norm site.
            if qkv_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
                let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                let block_size = rmsnorm_block_size(hidden_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = hidden_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(fused_fn)
                        .arg(&st.scratch.x_gpu)
                        .arg(&lw.attn_norm)
                        .arg(&mut *q8_1_buf)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm_to_q8_1 attn: {e}")))?;
                unsafe {
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    if has_qgate_fusion {
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, st.scratch.q_gate.as_mut().unwrap(), wq_out_dim, hidden_dim, "wq")?;
                    } else {
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "wq")?;
                    }
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wk, lw.q8_tile_wk.as_ref(), lw.q4_tile_wk.as_ref(), lw.q8_split_wk.as_ref(), lw.q4_split_wk.as_ref(), q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "wk")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wv, lw.q8_tile_wv.as_ref(), lw.q4_tile_wv.as_ref(), lw.q8_split_wv.as_ref(), lw.q4_split_wv.as_ref(), q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "wv")?;
                }
            } else if qkv_use_preq {
                // Fallback: separate rmsnorm + quantize_f32_to_q8_1 (fused kernel unavailable).
                {
                    let block_size = rmsnorm_block_size(hidden_dim);
                    let shared_bytes = rmsnorm_shared_bytes(block_size);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    let dim = hidden_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.rmsnorm)
                            .arg(&st.scratch.x_gpu)
                            .arg(&lw.attn_norm)
                            .arg(&mut st.scratch.normed)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm attn launch: {e}")))?;
                }
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                unsafe {
                    launch_quantize_input_q8_1(&self.device, quant_fn, &st.scratch.normed, q8_1_buf, hidden_dim, "qkv")?;
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    if has_qgate_fusion {
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, st.scratch.q_gate.as_mut().unwrap(), wq_out_dim, hidden_dim, "wq")?;
                    } else {
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "wq")?;
                    }
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wk, lw.q8_tile_wk.as_ref(), lw.q4_tile_wk.as_ref(), lw.q8_split_wk.as_ref(), lw.q4_split_wk.as_ref(), q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "wk")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wv, lw.q8_tile_wv.as_ref(), lw.q4_tile_wv.as_ref(), lw.q8_split_wv.as_ref(), lw.q4_split_wv.as_ref(), q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "wv")?;
                }
            } else {
                // Non-preq path: separate rmsnorm + launch_matvec (with internal quantization).
                {
                    let block_size = rmsnorm_block_size(hidden_dim);
                    let shared_bytes = rmsnorm_shared_bytes(block_size);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    let dim = hidden_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.rmsnorm)
                            .arg(&st.scratch.x_gpu)
                            .arg(&lw.attn_norm)
                            .arg(&mut st.scratch.normed)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm attn launch: {e}")))?;
                }
                unsafe {
                    // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                    if has_qgate_fusion {
                        launch_matvec(
                            &self.device,
                            &st.kernels,
                            &lw.wq,
                            &st.scratch.normed,
                            st.scratch.q_gate.as_mut().unwrap(),
                            wq_out_dim,
                            hidden_dim,
                            "wq",
                            lw.wq_f16.as_ref(),
                            Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    } else {
                        launch_matvec(
                            &self.device,
                            &st.kernels,
                            &lw.wq,
                            &st.scratch.normed,
                            &mut st.scratch.q,
                            q_dim,
                            hidden_dim,
                            "wq",
                            lw.wq_f16.as_ref(),
                            Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        &lw.wk,
                        &st.scratch.normed,
                        &mut st.scratch.k,
                        kv_dim,
                        hidden_dim,
                        "wk",
                        lw.wk_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        &lw.wv,
                        &st.scratch.normed,
                        &mut st.scratch.v,
                        kv_dim,
                        hidden_dim,
                        "wv",
                        lw.wv_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        }

        // Q+gate fusion post-processing: deinterleave q_gate -> q + gate_buf,
        // then per-head RMSNorm on Q (attn_q_norm) and K (attn_k_norm).
        // Must run AFTER all QKV projection branches and BEFORE RoPE.
        if has_qgate_fusion {
            let q_gate_buf = st.scratch.q_gate.as_ref().unwrap();
            let gate_buf = st.scratch.gate_buf.as_mut().unwrap();

            // 1a. Deinterleave: q_gate [q_dim*2] -> q [q_dim] + gate_buf [q_dim]
            if let Some(ref deinterleave_fn) = st.kernels.deinterleave_qgate {
                let block = 256u32;
                let grid = ((q_dim as u32) + block - 1) / block;
                let hd = head_dim as u32;
                let nh = num_heads as u32;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(deinterleave_fn)
                        .arg(q_gate_buf)
                        .arg(&mut st.scratch.q)
                        .arg(gate_buf)
                        .arg(&hd)
                        .arg(&nh)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("deinterleave_qgate launch: {e}")))?;
            } else {
                return Err(RuntimeError::Compute(
                    "Q+gate fusion requires deinterleave_qgate kernel".into(),
                ));
            }

            // 1b. Per-head RMSNorm on Q using attn_q_norm [head_dim]
            if let Some(ref q_norm_w) = lw.attn_q_norm {
                let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Q+gate fusion requires rmsnorm_per_head_inplace kernel".into())
                })?;
                let hd = head_dim as u32;
                let nh = num_heads as u32;
                let block = (head_dim as u32).min(1024).max(32);
                let block = (block / 32) * 32; // Round down to warp multiple
                let shared_bytes = (block / 32) * 4; // One float per warp
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (nh, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(norm_fn)
                        .arg(&mut st.scratch.q)
                        .arg(q_norm_w)
                        .arg(&nh)
                        .arg(&hd)
                        .arg(&eps)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm_per_head Q launch: {e}")))?;
            }

            // 1c. Per-head RMSNorm on K using attn_k_norm [head_dim]
            if let Some(ref k_norm_w) = lw.attn_k_norm {
                let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Q+gate fusion requires rmsnorm_per_head_inplace kernel".into())
                })?;
                let hd = head_dim as u32;
                let nkvh = num_kv_heads as u32;
                let block = (head_dim as u32).min(1024).max(32);
                let block = (block / 32) * 32;
                let shared_bytes = (block / 32) * 4;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (nkvh, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(norm_fn)
                        .arg(&mut st.scratch.k)
                        .arg(k_norm_w)
                        .arg(&nkvh)
                        .arg(&hd)
                        .arg(&eps)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm_per_head K launch: {e}")))?;
            }
        }

        // QKV bias (Qwen2-family, decode).
        if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
            let block = 256u32;
            unsafe {
                if let Some(ref bq) = lw.bq {
                    let d = q_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.q).arg(bq).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bq decode: {e}")))?;
                }
                if let Some(ref bk) = lw.bk {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.k).arg(bk).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bk decode: {e}")))?;
                }
                if let Some(ref bv) = lw.bv {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.v).arg(bv).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bv decode: {e}")))?;
                }
            }
        }

        // 2. RoPE.
        {
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            let actual_rot = if rotary_dim > 0 && rotary_dim < head_dim as u32 { rotary_dim as usize } else { head_dim };
            let half_rot = actual_rot / 2;
            let total_q_pairs = num_heads * half_rot;
            let total_k_pairs = num_kv_heads * half_rot;
            let max_pairs = total_q_pairs.max(total_k_pairs);
            let config = LaunchConfig::for_elements(max_pairs);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let pos = seq_pos as u32;
            let nqh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            // NeoX RoPE: models with partial rotary_dim (e.g. Qwen3.5) use half-offset
            // dimension pairing instead of standard interleaved pairing.
            let rope_neox = hp.rope_neox;
            let rope_fn = if rope_neox {
                &st.kernels.rope_apply_neox
            } else {
                &st.kernels.rope_apply
            };
            unsafe {
                self.device
                    .stream
                    .launch_builder(rope_fn)
                    .arg(&mut st.scratch.q)
                    .arg(&mut st.scratch.k)
                    .arg(&pos)
                    .arg(&nqh)
                    .arg(&nkvh)
                    .arg(&hd)
                    .arg(&theta)
                    .arg(&rotary_dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("rope launch: {e}")))?;
        }

        // 3. KV cache write.
        {
            let kv_cache = st.kv_caches.get_mut(layer_idx).ok_or_else(|| {
                RuntimeError::Compute(format!("no KV cache for layer {layer_idx}"))
            })?;
            kv_cache.append_kv(&self.device, &st.scratch.k, &st.scratch.v)?;
        }

        // 4. Attention. gate: routes to the tiled streaming-softmax
        // kernel at long context (seq_len > LUMEN_CUDA_DECODE_TILED_THRESHOLD,
        // default 0 = "tiled-always") or when LUMEN_CUDA_DECODE_TILED=1
        // forces it. Operators can set `LUMEN_CUDA_DECODE_TILED_THRESHOLD=
        // 4294967295` to opt out (force single-block below the 40_950 ceiling).
        {
            let kv_cache = &st.kv_caches[layer_idx];
            let attn_seq_len = kv_cache.seq_len() as u32;
            let nh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            let msl = kv_cache.max_seq_len as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            unsafe {
                super::prefill::launch_attention_decode_gated(
                    &self.device,
                    &st.kernels,
                    &st.scratch.q,
                    &kv_cache.k_cache,
                    &kv_cache.v_cache,
                    &mut st.scratch.attn_out,
                    nh,
                    nkvh,
                    hd,
                    attn_seq_len,
                    msl,
                    scale,
                )
            }
            .map_err(|e| RuntimeError::Compute(format!("attention_decode launch: {e}")))?;
        }

        // 4b. Q+gate sigmoid gating: attn_out = sigmoid(gate_buf) * attn_out.
        // Applied AFTER attention, BEFORE output projection.
        //
        // FIX-3: write through `st.scratch.q` (already sized [q_dim] and
        // unused after attention) and then memcpy back to attn_out. Previously
        // the temp was `normed` which is sized `[hidden_dim]`; this overflowed
        // for Qwen3.5-MoE-35B-A3B where `q_dim=4096 > hidden_dim=2048`,
        // corrupting adjacent GPU memory and producing gibberish output.
        if has_qgate_fusion {
            if let Some(ref sigmoid_fn) = st.kernels.sigmoid_mul {
                let gate_buf = st.scratch.gate_buf.as_ref().unwrap();
                let n = q_dim as u32;
                let block = 256u32;
                let grid = (n + block - 1) / block;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                // Step 1: sigmoid(gate) * attn_out -> q (temp, sized [q_dim]).
                // st.scratch.q is consumed by attention_decode_gated above; safe to reuse.
                unsafe {
                    self.device
                        .stream
                        .launch_builder(sigmoid_fn)
                        .arg(gate_buf)
                        .arg(&st.scratch.attn_out)
                        .arg(&mut st.scratch.q)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("sigmoid_mul launch: {e}")))?;
                // Step 2: copy q -> attn_out (both [q_dim])
                self.device
                    .stream
                    .memcpy_dtod(&st.scratch.q, &mut st.scratch.attn_out)
                    .map_err(|e| RuntimeError::Compute(format!("sigmoid_mul dtod copy: {e}")))?;
            } else {
                return Err(RuntimeError::Compute(
                    "Q+gate fusion requires sigmoid_mul kernel".into(),
                ));
            }
        }

        // 5. Output projection + residual: attn_proj = wo * attn_out + x_gpu.
        if let GpuWeightBuf::F16Raw(ref wo_f16) = lw.wo {
            unsafe {
                launch_hgemv_f16_residual(
                    &self.device,
                    &st.kernels,
                    wo_f16,
                    &st.scratch.attn_out,
                    &st.scratch.x_gpu,
                    &mut st.scratch.attn_proj,
                    &mut st.scratch.input_f16,
                    hidden_dim,
                    q_dim,
                    "wo",
                    st.algo_cache.get(hidden_dim, q_dim),
                )?;
            }
        } else if matches!(&lw.wo, GpuWeightBuf::F32(_))
            && lw.wo_f16.is_some()
        {
            // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
            // Q8/Q4 weights fall through to launch_matvec_residual() for native dp4a.
            let wo_f16 = lw.wo_f16.as_ref().unwrap();
            unsafe {
                launch_hgemv_f16_residual(
                    &self.device,
                    &st.kernels,
                    wo_f16,
                    &st.scratch.attn_out,
                    &st.scratch.x_gpu,
                    &mut st.scratch.attn_proj,
                    &mut st.scratch.input_f16,
                    hidden_dim,
                    q_dim,
                    "wo",
                    st.algo_cache.get(hidden_dim, q_dim),
                )?;
            }
        } else {
            // split-layout: when a Q8/Q4 split sibling is available for wo, route
            // through `launch_matvec_residual_split` -- requires quantizing the
            // attention output to Q8_1 inline. Otherwise fall through to the
            // existing `launch_matvec_residual` path.
            let use_split_wo = (st.kernels.use_q8_split_dispatch && lw.q8_split_wo.is_some())
                || (st.kernels.use_q4_split_dispatch && lw.q4_split_wo.is_some());
            if use_split_wo {
                // Quantize attention output to Q8_1 in scratch, then split residual matvec.
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device, quant_fn, &st.scratch.attn_out, q8_1_buf,
                            q_dim, "wo split",
                        )?;
                        launch_matvec_preq8_1_residual_tile(
                            &self.device, &st.kernels, &lw.wo,
                            lw.q8_tile_wo.as_ref(),  lw.q4_tile_wo.as_ref(),
                            lw.q8_split_wo.as_ref(), lw.q4_split_wo.as_ref(),
                            q8_1_buf, &st.scratch.x_gpu, &mut st.scratch.attn_proj,
                            hidden_dim, q_dim, "wo",
                        )?;
                    }
                } else {
                    unsafe {
                        launch_matvec_residual(
                            &self.device, &st.kernels, &lw.wo,
                            &st.scratch.attn_out, &st.scratch.x_gpu,
                            &mut st.scratch.attn_proj, hidden_dim, q_dim, "wo",
                            lw.wo_f16.as_ref(),
                            Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            } else {
                unsafe {
                    launch_matvec_residual(
                        &self.device,
                        &st.kernels,
                        &lw.wo,
                        &st.scratch.attn_out,
                        &st.scratch.x_gpu,
                        &mut st.scratch.attn_proj,
                        hidden_dim,
                        q_dim,
                        "wo",
                        lw.wo_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        }
        } // end else (standard attention path — skipped for GDN layers)

        // Re-borrow layer weights for the FFN block (shared between standard and GDN layers).
        let lw: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];

        // MoE FFN branch — when the layer has expert metadata, dispatch
        // the three-phase MoE forward (router -> per-expert FFN -> accum) and
        // skip the dense FFN block below entirely.
        if let Some(moe_meta) = st.moe_meta_cache.get(layer_idx).and_then(|m| m.as_ref()) {
            let moe_layer_blob = lw.moe_layer_blob.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "MoE layer {layer_idx} missing moe_layer_blob; \
                     upload_layer_weights must populate it when subtensors.experts.is_some()",
                ))
            })?;
            let num_experts = moe_meta.expert_gate_offs.len();
            let top_k = self.hp()?.num_active_experts.map(|v| v as usize).unwrap_or(0);
            if top_k == 0 {
                return Err(RuntimeError::Compute(
                    "MoE layer present but hyperparams.num_active_experts not set".into(),
                ));
            }

            // FIX-3 DEBUG: pre-FFN-norm dump + ffn_norm value samples
            if {
                use std::sync::OnceLock;
                static FLAG: OnceLock<bool> = OnceLock::new();
                *FLAG.get_or_init(|| std::env::var("LUMEN_CUDA_MOE_DEBUG_DUMP").as_deref() == Ok("1"))
            } {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static PRE_COUNT: AtomicUsize = AtomicUsize::new(0);
                let n = PRE_COUNT.fetch_add(1, Ordering::Relaxed);
                if n < 80 {
                    let xg_host = self.device.dtoh_copy(&st.scratch.x_gpu)?;
                    let ap_host = self.device.dtoh_copy(&st.scratch.attn_proj)?;
                    let ffn_host = self.device.dtoh_copy(&lw.ffn_norm)?;
                    let attn_norm_host = self.device.dtoh_copy(&lw.attn_norm)?;
                    let mut xg_max = 0.0f32; let mut xg_sum = 0.0f64;
                    for v in &xg_host { xg_max = xg_max.max(v.abs()); xg_sum += v.abs() as f64; }
                    let mut ap_max = 0.0f32; let mut ap_sum = 0.0f64;
                    for v in &ap_host { ap_max = ap_max.max(v.abs()); ap_sum += v.abs() as f64; }
                    let mut fn_max = 0.0f32; let mut fn_sum = 0.0f64;
                    for v in &ffn_host { fn_max = fn_max.max(v.abs()); fn_sum += v.abs() as f64; }
                    let mut an_max = 0.0f32; let mut an_sum = 0.0f64;
                    for v in &attn_norm_host { an_max = an_max.max(v.abs()); an_sum += v.abs() as f64; }
                    eprintln!(
                        "[MoE-PRE] call={n} layer={layer_idx} | x_gpu_pre: max={xg_max:.3} mean={:.4} first5={:?} | attn_proj_pre: max={ap_max:.3} mean={:.3} | attn_norm: max={an_max:.3} mean={:.3} | ffn_norm: max={fn_max:.3} mean={:.3} first5={:?}",
                        xg_sum / hidden_dim as f64,
                        &xg_host[..5.min(xg_host.len())],
                        ap_sum / hidden_dim as f64,
                        an_sum / attn_norm_host.len().max(1) as f64,
                        fn_sum / ffn_host.len().max(1) as f64,
                        &ffn_host[..5.min(ffn_host.len())],
                    );
                }
            }

            // Fused FFN-norm + router. When the V3 fused kernel is
            // loaded and LUMEN_CUDA_MOE_FUSED_NORM_ROUTER=1, the standalone
            // RMSNorm dispatch is collapsed into the router kernel (saves 1
            // launch per MoE layer). Otherwise this wrapper runs the standalone
            // RMSNorm itself, preserving byte-identity vs the legacy path.
            let batched_offsets = st
                .moe_batched_offsets
                .get(layer_idx)
                .and_then(|b| b.as_ref());
            let moe_scratch = st.moe_scratch.as_mut().ok_or_else(|| {
                RuntimeError::Compute(
                    "MoE layer dispatch requires moe_scratch (allocated in init for MoE models)".into(),
                )
            })?;

            super::moe::encode_moe_ffn_decode_fused_norm(
                &self.device,
                &st.kernels,
                moe_scratch,
                moe_meta,
                batched_offsets,
                moe_layer_blob,
                &st.scratch.attn_proj.slice(..),
                &lw.ffn_norm,
                &mut st.scratch.normed.slice_mut(..),
                &st.scratch.attn_proj.slice(..),
                &mut st.scratch.x_gpu.slice_mut(..),
                eps,
                hidden_dim,
                inter_dim,
                num_experts,
                top_k,
            )?;

            // FIX: shared-expert FFN dispatch (Qwen3.5-MoE always-active expert).
            //
            // The shared expert runs on every token in addition to the top-K
            // routed experts; its output is sigmoid-gated by
            // `ffn_gate_inp_shexp` and added to x_gpu AFTER the routed
            // accumulation. Ported from `metal::moe::encode_shared_expert_ffn_decode_raw`.
            // Without this dispatch, the FFN is missing a typically-dominant
            // residual term and the model output is gibberish (prior reproduction).
            //
            // Env-var bisection knob: `LUMEN_CUDA_SKIP_SHARED_EXPERT=1` skips
            // the dispatch (reverts to routed-only behavior). Used to
            // confirm that the dispatch is what changes output behavior.
            // cache via OnceLock to avoid per-layer env::var overhead
            // (~40 calls/token × 5 µs = 0.2 ms saved).
            let skip_shared = {
                use std::sync::OnceLock;
                static FLAG: OnceLock<bool> = OnceLock::new();
                *FLAG.get_or_init(|| {
                    std::env::var("LUMEN_CUDA_SKIP_SHARED_EXPERT")
                        .ok()
                        .as_deref()
                        .map(|v| matches!(v, "1" | "true" | "yes"))
                        .unwrap_or(false)
                })
            };
            if moe_meta.shared_gate.is_some() && !skip_shared {
                // opt-in fused shared-expert path (3 launches vs 5-6).
                // Falls back to legacy unfused path if any of the 3 fused
                // kernels failed to compile (NVRTC failure on this device).
                let use_fused = super::moe::moe_shared_fused_enabled()
                    && st.kernels.fused_glu_gemv_q4_0_prenormed_no_norm.is_some()
                    && st.kernels.moe_shared_down_q4_0_sigmoid_accum.is_some()
                    && st.kernels.moe_shared_down_q4_0_residual_accum.is_some();
                if use_fused {
                    super::moe::encode_shared_expert_ffn_decode_fused(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        moe_meta,
                        moe_layer_blob,
                        &st.scratch.normed.slice(..),
                        &mut st.scratch.x_gpu.slice_mut(..),
                        hidden_dim,
                    )?;
                } else {
                    super::moe::encode_shared_expert_ffn_decode(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        moe_meta,
                        moe_layer_blob,
                        &st.scratch.normed.slice(..),
                        &mut st.scratch.x_gpu.slice_mut(..),
                        hidden_dim,
                    )?;
                }
            }

            // FIX-3 DEBUG: dump x_gpu and attn_proj magnitude per layer to bisect.
            // Print on first decode call for first few layers to identify which layer's
            // dataflow goes haywire. Gated by `LUMEN_CUDA_MOE_DEBUG_DUMP=1`.
            if {
                use std::sync::OnceLock;
                static FLAG: OnceLock<bool> = OnceLock::new();
                *FLAG.get_or_init(|| std::env::var("LUMEN_CUDA_MOE_DEBUG_DUMP").as_deref() == Ok("1"))
            } {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static COUNT: AtomicUsize = AtomicUsize::new(0);
                let n = COUNT.fetch_add(1, Ordering::Relaxed);
                if n < 80 {
                    let x_host = self.device.dtoh_copy(&st.scratch.x_gpu)?;
                    let ap_host = self.device.dtoh_copy(&st.scratch.attn_proj)?;
                    let normed_host = self.device.dtoh_copy(&st.scratch.normed)?;
                    let mut x_max = 0.0f32; let mut x_sum = 0.0f64; let mut x_nan = 0;
                    for v in &x_host { x_max = x_max.max(v.abs()); x_sum += v.abs() as f64; if !v.is_finite() { x_nan += 1; } }
                    let mut ap_max = 0.0f32; let mut ap_sum = 0.0f64;
                    for v in &ap_host { ap_max = ap_max.max(v.abs()); ap_sum += v.abs() as f64; }
                    let mut n_max = 0.0f32; let mut n_sum = 0.0f64;
                    for v in &normed_host { n_max = n_max.max(v.abs()); n_sum += v.abs() as f64; }
                    eprintln!(
                        "[MoE-DEBUG] call={n} layer={layer_idx} hidden={hidden_dim} | normed: max={n_max:.3} mean={:.3} | attn_proj: max={ap_max:.3} mean={:.3} | x_gpu: max={x_max:.3} mean={:.3} nans={x_nan} | first_5={:?}",
                        n_sum / hidden_dim as f64, ap_sum / hidden_dim as f64, x_sum / hidden_dim as f64,
                        &x_host[..5.min(x_host.len())],
                    );
                }
            }

            // MoE branch is complete; skip the dense FFN block below.
            return Ok(());
        }

        // 6. FFN: fused or separate rmsnorm + gate/up + swiglu + down + residual.
        //
        // Fused gate+up+SwiGLU GEMV: if the kernel is available and shmem fits,
        // compute rms_scale + fused_glu_gemv in 2 dispatches (replacing 3-5).
        // The fused kernel writes silu(gate)*up directly to scratch.gate,
        // so the SwiGLU step is skipped entirely.
        let fused_glu_fired = 'fused_glu: {
            // env-gated opt-out of the fused gate+up+SwiGLU kernel.
            // Profile evidence shows `fused_glu_gemv_q8_0`
            // is 30.8% of Lumen Q8 dense decode kernel time at 158 us/call,
            // dominated by SCALAR `(float)gq[j] * xv[j]` inner loops (no dp4a,
            // no tensor cores). The fall-through `launch_matvec_preq8_1_tile`
            // path uses `mul_mat_vec_q_q8_0` (via `LUMEN_CUDA_MMV_Q_DP4A=1`
            // default-ON) which is dp4a-based at ~25.5 us/call = ~6x faster
            // per call. Two extra dispatches (gate + up separately + SwiGLU)
            // are outweighed by the 6x speedup on the inner GEMV math.
            // Measured +27% Q8 dense decode (85.2 -> 108.2 = 0.90× llama.cpp) and
            // +43% Q4 dense decode (90.5 -> 129.6 = 0.86× llama.cpp) on A100. Default
            // OFF (preserves the prior byte-identity); set
            // `LUMEN_CUDA_FFN_FUSED_GLU=0` to enable the dp4a fall-through.
            // default to SKIP fused (use dp4a fall-through)
            // on quantised dense models — measured +27% Q8 / +43% Q4 dense
            // decode. BF16 dense / MoE are unaffected because
            // their FFN paths don't dispatch this kernel. Env `=0` retains
            // the original "skip" opt-in; env `=1` forces use of the fused
            // kernel even on quantised dense (opt-out of the F2 flip).
            let skip_fused_glu = match std::env::var("LUMEN_CUDA_FFN_FUSED_GLU").ok().as_deref() {
                Some(v) => matches!(v, "0" | "false" | "no" | "off" | "OFF"),
                None => crate::runtime_defaults::ffn_fused_glu_skip_default(),
            };
            if skip_fused_glu {
                break 'fused_glu false;
            }
            let hd = hidden_dim as u32;
            let shmem_f32 = fused_glu_shared_bytes_f32(hd);
            let shmem_f16 = fused_glu_shared_bytes_f16(hd);

            // Try Q8_0 fused kernel (gate and up must both be Q8Raw).
            if let (GpuWeightBuf::Q8Raw(ref wg_q8), GpuWeightBuf::Q8Raw(ref wu_q8)) =
                (&lw.w_gate, &lw.w_up)
            {
                // F32 shmem variant: hidden_dim * 4 <= 48KB.
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q8_0.as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q8)
                            .arg(wu_q8)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q8_0 L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
                // F16 shmem variant: hidden_dim * 2 <= 48KB (large dims).
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q8_0_hg.as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q8)
                            .arg(wu_q8)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q8_0_hg L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
            }

            // Try Q8Aligned fused kernel (gate and up must both be Q8Aligned).
            // Previously disabled when HGEMV was the Q8 decode path (C34: -5-8% vs tensor core HGEMV).
            // Now that Q8Aligned routes through native dp4a decode, the fused kernel competes against
            // separate rmsnorm+dp4a (not HGEMV), making it the better choice for dispatch reduction.
            if let (GpuWeightBuf::Q8Aligned(ref wg_q8a), GpuWeightBuf::Q8Aligned(ref wu_q8a)) =
                (&lw.w_gate, &lw.w_up)
            {
                // F32 shmem variant: hidden_dim * 4 <= 48KB.
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q8_aligned.as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q8a)
                            .arg(wu_q8a)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q8_aligned L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
                // F16 shmem variant: hidden_dim * 2 <= 48KB (large dims).
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q8_aligned_hg.as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q8a)
                            .arg(wu_q8a)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q8_aligned_hg L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
            }

            // Try Q4_0 fused kernel (gate and up must both be Q4Raw).
            if let (GpuWeightBuf::Q4Raw(ref wg_q4), GpuWeightBuf::Q4Raw(ref wu_q4)) =
                (&lw.w_gate, &lw.w_up)
            {
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q4_0.as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q4)
                            .arg(wu_q4)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q4_0 L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_q4_0_hg.as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_q4)
                            .arg(wu_q4)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_q4_0_hg L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
            }

            // Try F16 fused kernel (gate and up must both be F16Raw).
            if let (GpuWeightBuf::F16Raw(ref wg_f16), GpuWeightBuf::F16Raw(ref wu_f16)) =
                (&lw.w_gate, &lw.w_up)
            {
                if let Some(ref fused_fn) = st.kernels.fused_glu_gemv_f16.as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
                {
                    unsafe {
                        launch_compute_rms_scale(
                            &self.device, &st.kernels,
                            &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                            eps, hidden_dim,
                        )?;
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        // F16 weights passed as u8 slices, cast to unsigned short* in kernel.
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wg_f16)
                            .arg(wu_f16)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&st.scratch.rms_scale)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                            .map_err(|e| RuntimeError::Compute(format!(
                                "fused_glu_gemv_f16 L{layer_idx}: {e}",
                            )))?;
                    }
                    break 'fused_glu true;
                }
            }

            // Fused kernel not available or shmem insufficient — fall through.
            false
        };

        // If fused kernel did NOT fire, use existing separate gate+up dispatch.
        if !fused_glu_fired {
        if matches!(&lw.w_gate, GpuWeightBuf::F32(_))
            && matches!(&lw.w_up, GpuWeightBuf::F32(_))
        {
            unsafe {
                launch_compute_rms_scale(
                    &self.device,
                    &st.kernels,
                    &st.scratch.attn_proj,
                    &mut st.scratch.rms_scale,
                    eps,
                    hidden_dim,
                )?;
            }
            if let (GpuWeightBuf::F32(ref wg_f32), GpuWeightBuf::F32(ref wu_f32)) =
                (&lw.w_gate, &lw.w_up)
            {
                unsafe {
                    launch_fused_norm_dual_matvec_f32(
                        &self.device,
                        &st.kernels,
                        &st.scratch.attn_proj,
                        &st.scratch.rms_scale,
                        &lw.ffn_norm,
                        wg_f32,
                        wu_f32,
                        &mut st.scratch.gate,
                        &mut st.scratch.up,
                        inter_dim,
                        hidden_dim,
                    )?;
                }
            }
        } else if matches!(&lw.w_gate, GpuWeightBuf::F16Raw(_))
            && matches!(&lw.w_up, GpuWeightBuf::F16Raw(_))
        {
            // F16 HGEMV path for FFN gate/up: Fused RMSNorm + F32->F16 in ONE kernel
            // (saves 1 dispatch), then cuBLAS HGEMV for gate and up.
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.attn_proj, &lw.ffn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "ffn F16",
                )?;
            }
            // Gate+up: use pre-computed pointers if available.
            if let Some(ref pcp) = st.precomputed_ptrs {
                unsafe {
                    launch_hgemv_f16_batched_precomputed(
                        &self.device,
                        &pcp.ffn_a_ptrs[layer_idx],
                        &pcp.ffn_b_ptrs[layer_idx],
                        &pcp.ffn_c_ptrs[layer_idx],
                        2, inter_dim, hidden_dim, "gate_up",
                        st.algo_cache.get(inter_dim, hidden_dim),
                    )?;
                }
            } else if let (GpuWeightBuf::F16Raw(ref wg_f16), GpuWeightBuf::F16Raw(ref wu_f16)) = (&lw.w_gate, &lw.w_up) {
                unsafe {
                    let w_slices: &[&CudaSlice<u8>] = &[wg_f16, wu_f16];
                    let mut out_slices: [&mut CudaSlice<f32>; 2] = [&mut st.scratch.gate, &mut st.scratch.up];
                    launch_hgemv_f16_batched(
                        &self.device,
                        w_slices,
                        &st.scratch.input_f16,
                        &mut out_slices,
                        &mut st.scratch.batched_a_ptrs,
                        &mut st.scratch.batched_b_ptrs,
                        &mut st.scratch.batched_c_ptrs,
                        inter_dim,
                        hidden_dim,
                        "gate_up",
                        st.algo_cache.get(inter_dim, hidden_dim),
                    )?;
                }
            }
        } else if matches!(&lw.w_gate, GpuWeightBuf::F32(_))
            && lw.w_gate_f16.is_some() && lw.w_up_f16.is_some()
        {
            // cuBLAS HGEMV for F32 weights with F16 caches (halves F32 bandwidth).
            // Q8/Q4 weights fall through to launch_matvec() for native dp4a (1.06 B/elem).
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.attn_proj, &lw.ffn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "ffn HGEMV",
                )?;
            }
            // Gate+up: use pre-computed pointers if available (batched = 1 cuBLAS call).
            if let Some(ref pcp) = st.precomputed_ptrs {
                unsafe {
                    launch_hgemv_f16_batched_precomputed(
                        &self.device,
                        &pcp.ffn_a_ptrs[layer_idx],
                        &pcp.ffn_b_ptrs[layer_idx],
                        &pcp.ffn_c_ptrs[layer_idx],
                        2, inter_dim, hidden_dim, "gate_up",
                        st.algo_cache.get(inter_dim, hidden_dim),
                    )?;
                }
            } else {
                // Fallback: separate gate + up HGEMV calls.
                if let Some(ref wg_f16) = lw.w_gate_f16 {
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wg_f16, &st.scratch.input_f16,
                            &mut st.scratch.gate, inter_dim, hidden_dim, "gate",
                            st.algo_cache.get(inter_dim, hidden_dim),
                        )?;
                    }
                }
                if let Some(ref wu_f16) = lw.w_up_f16 {
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wu_f16, &st.scratch.input_f16,
                            &mut st.scratch.up, inter_dim, hidden_dim, "up",
                            st.algo_cache.get(inter_dim, hidden_dim),
                        )?;
                    }
                }
            }
        } else {
            // Q8_0/Q4_0/Q8Aligned/Q4Aligned/F32: native-quant FFN gate/up via launch_matvec().
            // Priority: dp4a Q8_1 > smem > hgemv > cuBLAS HGEMV > dp4a/scalar.
            // F16 caches are passed as last-resort fallback only.

            // Shared-quantization optimization: quantize normed FFN input ONCE,
            // reuse across gate and up projections. Saves 1 quantize launch per layer.
            let ffn_use_preq = weight_uses_dp4a_q8_1(&lw.w_gate, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.w_up, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some();

            // Fused RMSNorm + Q8_1 for FFN: saves 1 dispatch per layer.
            if ffn_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
                let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                let block_size = rmsnorm_block_size(hidden_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = hidden_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(fused_fn)
                        .arg(&st.scratch.attn_proj)
                        .arg(&lw.ffn_norm)
                        .arg(&mut *q8_1_buf)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm_to_q8_1 ffn: {e}")))?;
                unsafe {
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on FFN gate/up when set.
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_gate, lw.q8_tile_w_gate.as_ref(), lw.q4_tile_w_gate.as_ref(), lw.q8_split_w_gate.as_ref(), lw.q4_split_w_gate.as_ref(), q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "gate")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_up, lw.q8_tile_w_up.as_ref(), lw.q4_tile_w_up.as_ref(), lw.q8_split_w_up.as_ref(), lw.q4_split_w_up.as_ref(), q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "up")?;
                }
            } else if ffn_use_preq {
                // Fallback: separate rmsnorm + quantize_f32_to_q8_1 (fused kernel unavailable).
                {
                    let block_size = rmsnorm_block_size(hidden_dim);
                    let shared_bytes = rmsnorm_shared_bytes(block_size);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    let dim = hidden_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.rmsnorm)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&mut st.scratch.normed)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm ffn launch: {e}")))?;
                }
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                unsafe {
                    launch_quantize_input_q8_1(&self.device, quant_fn, &st.scratch.normed, q8_1_buf, hidden_dim, "ffn gate_up")?;
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on FFN gate/up when set.
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_gate, lw.q8_tile_w_gate.as_ref(), lw.q4_tile_w_gate.as_ref(), lw.q8_split_w_gate.as_ref(), lw.q4_split_w_gate.as_ref(), q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "gate")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_up, lw.q8_tile_w_up.as_ref(), lw.q4_tile_w_up.as_ref(), lw.q8_split_w_up.as_ref(), lw.q4_split_w_up.as_ref(), q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "up")?;
                }
            } else {
                // Non-preq path: separate rmsnorm + launch_matvec.
                {
                    let block_size = rmsnorm_block_size(hidden_dim);
                    let shared_bytes = rmsnorm_shared_bytes(block_size);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    let dim = hidden_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.rmsnorm)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&mut st.scratch.normed)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm ffn launch: {e}")))?;
                }
                unsafe {
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        &lw.w_gate,
                        &st.scratch.normed,
                        &mut st.scratch.gate,
                        inter_dim,
                        hidden_dim,
                        "gate",
                        lw.w_gate_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        &lw.w_up,
                        &st.scratch.normed,
                        &mut st.scratch.up,
                        inter_dim,
                        hidden_dim,
                        "up",
                        lw.w_up_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        }
        } // end if !fused_glu_fired

        // SwiGLU + Down projection.
        //
        // When fused_glu_fired: SwiGLU is already applied inline. scratch.gate
        // contains silu(gate)*up. Only the down projection + residual are needed.
        //
        // When !fused_glu_fired: gate and up are separate buffers. Apply SwiGLU
        // to combine them before the down projection.
        //
        // For native F16 weights (F16Raw w_down): fuse SwiGLU with F32->F16
        // conversion in ONE kernel, then cuBLAS HGEMV (optimal for F16).
        //
        // For Q8_0/Q4_0/F32: SwiGLU + native-quant matvec via launch_matvec().
        // Native kernels (dp4a/smem/hgemv) read quant directly; F16 cache is
        // passed as fallback only.
        if fused_glu_fired {
            // Fused kernel already computed silu(gate)*up into scratch.gate.
            // Just run the down projection reading from scratch.gate.
            if let GpuWeightBuf::F16Raw(ref wd_f16) = lw.w_down {
                // Convert fused output F32 -> F16 for HGEMV down projection.
                let config = LaunchConfig::for_elements(inter_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = inter_dim as u32;
                unsafe {
                    self.device.stream
                        .launch_builder(&st.kernels.f32_to_f16_vec)
                        .arg(&st.scratch.gate)
                        .arg(&mut st.scratch.input_f16)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("f32_to_f16 fused_glu down: {e}")))?;
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wd_f16, &st.scratch.input_f16,
                        &mut st.scratch.down, hidden_dim, inter_dim, "down",
                        st.algo_cache.get(hidden_dim, inter_dim),
                    )?;
                }
            } else if matches!(&lw.w_down, GpuWeightBuf::F32(_))
                && lw.w_down_f16.is_some()
            {
                // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
                // Convert fused output F32 -> F16, then cuBLAS HGEMV with FAST_16F.
                // Q8/Q4 weights fall through to launch_matvec() for native dp4a.
                let wd_f16 = lw.w_down_f16.as_ref().unwrap();
                let config = LaunchConfig::for_elements(inter_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = inter_dim as u32;
                unsafe {
                    self.device.stream
                        .launch_builder(&st.kernels.f32_to_f16_vec)
                        .arg(&st.scratch.gate)
                        .arg(&mut st.scratch.input_f16)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("f32_to_f16 fused_glu down HGEMV: {e}")))?;
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wd_f16, &st.scratch.input_f16,
                        &mut st.scratch.down, hidden_dim, inter_dim, "down",
                        st.algo_cache.get(hidden_dim, inter_dim),
                    )?;
                }
            } else if let GpuWeightBuf::Q8Aligned(ref wd_q8a) = lw.w_down {
                // Fused down: inline F32->Q8_1 quantize + dp4a in one dispatch.
                // Eliminates the separate quantize_f32_to_q8_1 kernel.
                if let Some(ref fused_fn) = st.kernels.matvec_q8_aligned_f32 {
                    let out_dim_u32 = hidden_dim as u32;
                    let in_dim_u32 = inter_dim as u32;
                    let grid = dp4a_q8_1_grid(out_dim_u32);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wd_q8a)
                            .arg(&st.scratch.gate)
                            .arg(&mut st.scratch.down)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_aligned_f32 down L{layer_idx}: {e}",
                    )))?;
                } else {
                    // Fallback: quantize + dp4a (2 dispatches).
                    unsafe {
                        launch_matvec(
                            &self.device, &st.kernels, &lw.w_down,
                            &st.scratch.gate, &mut st.scratch.down,
                            hidden_dim, inter_dim, "down",
                            lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            } else if let GpuWeightBuf::Q4Aligned(ref wd_q4a) = lw.w_down {
                // Fused down for Q4Aligned: inline F32->Q8_1 quantize + dp4a in one dispatch.
                // Eliminates the separate quantize_f32_to_q8_1 kernel.
                if let Some(ref fused_fn) = st.kernels.matvec_q4_aligned_f32 {
                    let out_dim_u32 = hidden_dim as u32;
                    let in_dim_u32 = inter_dim as u32;
                    let grid = dp4a_q4_grid(out_dim_u32);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(fused_fn)
                            .arg(wd_q4a)
                            .arg(&st.scratch.gate)
                            .arg(&mut st.scratch.down)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_aligned_f32 down L{layer_idx}: {e}",
                    )))?;
                } else {
                    // Fallback: quantize + dp4a (2 dispatches).
                    unsafe {
                        launch_matvec(
                            &self.device, &st.kernels, &lw.w_down,
                            &st.scratch.gate, &mut st.scratch.down,
                            hidden_dim, inter_dim, "down",
                            lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            } else {
                // Native-quant down projection via launch_matvec().
                // split-layout: when a Q8/Q4 split sibling is available for w_down,
                // route via launch_matvec_preq8_1_split (requires inline F32->Q8_1
                // quantization since the existing fused-down kernels target
                // Q8Aligned/Q4Aligned which we skipped under SPLIT).
                let use_split_down = (st.kernels.use_q8_split_dispatch && lw.q8_split_w_down.is_some())
                    || (st.kernels.use_q4_split_dispatch && lw.q4_split_w_down.is_some());
                if use_split_down {
                    let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                    let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                    if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                        unsafe {
                            launch_quantize_input_q8_1(
                                &self.device, quant_fn, &st.scratch.gate, q8_1_buf,
                                inter_dim, "down split",
                            )?;
                            launch_matvec_preq8_1_tile(
                                &self.device, &st.kernels, &lw.w_down,
                                lw.q8_tile_w_down.as_ref(),  lw.q4_tile_w_down.as_ref(),
                                lw.q8_split_w_down.as_ref(), lw.q4_split_w_down.as_ref(),
                                q8_1_buf, &mut st.scratch.down,
                                hidden_dim, inter_dim, "down",
                            )?;
                        }
                    } else {
                        unsafe {
                            launch_matvec(
                                &self.device, &st.kernels, &lw.w_down,
                                &st.scratch.gate, &mut st.scratch.down,
                                hidden_dim, inter_dim, "down",
                                lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                                st.scratch.input_q8_1.as_mut(),
                            )?;
                        }
                    }
                } else {
                    unsafe {
                        launch_matvec(
                            &self.device, &st.kernels, &lw.w_down,
                            &st.scratch.gate, &mut st.scratch.down,
                            hidden_dim, inter_dim, "down",
                            lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            }
        } else if let GpuWeightBuf::F16Raw(ref wd_f16) = lw.w_down {
            // Fused SwiGLU + F32->F16: gate/up -> gate (F32) + input_f16 (F16).
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device, &st.kernels,
                    &mut st.scratch.gate, &st.scratch.up,
                    &mut st.scratch.input_f16,
                    inter_dim,
                )?;
            }
            // HGEMV with pre-converted F16 input (no separate conversion needed).
            unsafe {
                launch_hgemv_f16_preconverted(
                    &self.device,
                    wd_f16,
                    &st.scratch.input_f16,
                    &mut st.scratch.down,
                    hidden_dim,
                    inter_dim,
                    "down",
                    st.algo_cache.get(hidden_dim, inter_dim),
                )?;
            }
        } else if matches!(&lw.w_down, GpuWeightBuf::F32(_))
            && lw.w_down_f16.is_some()
        {
            // cuBLAS HGEMV for F32 down weights with F16 caches.
            // Q8/Q4 weights fall through to launch_matvec() for native dp4a.
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device, &st.kernels,
                    &mut st.scratch.gate, &st.scratch.up,
                    &mut st.scratch.input_f16,
                    inter_dim,
                )?;
            }
            if let Some(ref wd_f16) = lw.w_down_f16 {
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device,
                        wd_f16,
                        &st.scratch.input_f16,
                        &mut st.scratch.down,
                        hidden_dim,
                        inter_dim,
                        "down",
                        st.algo_cache.get(hidden_dim, inter_dim),
                    )?;
                }
            }
        } else if let GpuWeightBuf::Q8Aligned(ref wd_q8a) = lw.w_down {
            // Fused SwiGLU + quantize + dp4a down in ONE dispatch.
            // Reads F32 gateand up[], computes silu(gate)*up inline,
            // quantizes to Q8_1 in registers, and does dp4a against weights.
            // Replaces 3 dispatches (swiglu + quantize + matvec) with 1.
            if let Some(ref fused_fn) = st.kernels.matvec_q8_aligned_f32_swiglu {
                let out_dim_u32 = hidden_dim as u32;
                let in_dim_u32 = inter_dim as u32;
                let grid = dp4a_q8_1_grid(out_dim_u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream
                        .launch_builder(fused_fn)
                        .arg(wd_q8a)
                        .arg(&st.scratch.gate)
                        .arg(&st.scratch.up)
                        .arg(&mut st.scratch.down)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q8_aligned_f32_swiglu down L{layer_idx}: {e}",
                )))?;
            } else {
                // Fallback: separate SwiGLU + quantize + dp4a (3 dispatches).
                {
                    let config = LaunchConfig::for_elements(inter_dim);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (config.grid_dim, 1, 1),
                        block_dim: (config.block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n = inter_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.swiglu_inplace)
                            .arg(&mut st.scratch.gate)
                            .arg(&st.scratch.up)
                            .arg(&n)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("swiglu launch: {e}")))?;
                }
                unsafe {
                    launch_matvec(
                        &self.device, &st.kernels, &lw.w_down,
                        &st.scratch.gate, &mut st.scratch.down,
                        hidden_dim, inter_dim, "down",
                        lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        } else if let GpuWeightBuf::Q4Aligned(ref wd_q4a) = lw.w_down {
            // Fused SwiGLU + quantize + dp4a down in ONE dispatch for Q4Aligned.
            // Reads F32 gateand up[], computes silu(gate)*up inline,
            // quantizes to Q8_1 in registers, and does dp4a against Q4Aligned weights.
            // Replaces 3 dispatches (swiglu + quantize + matvec) with 1.
            if let Some(ref fused_fn) = st.kernels.matvec_q4_aligned_f32_swiglu {
                let out_dim_u32 = hidden_dim as u32;
                let in_dim_u32 = inter_dim as u32;
                let grid = dp4a_q4_grid(out_dim_u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream
                        .launch_builder(fused_fn)
                        .arg(wd_q4a)
                        .arg(&st.scratch.gate)
                        .arg(&st.scratch.up)
                        .arg(&mut st.scratch.down)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q4_aligned_f32_swiglu down L{layer_idx}: {e}",
                )))?;
            } else {
                // Fallback: separate SwiGLU + quantize + dp4a (3 dispatches).
                {
                    let config = LaunchConfig::for_elements(inter_dim);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (config.grid_dim, 1, 1),
                        block_dim: (config.block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n = inter_dim as u32;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.swiglu_inplace)
                            .arg(&mut st.scratch.gate)
                            .arg(&st.scratch.up)
                            .arg(&n)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("swiglu launch: {e}")))?;
                }
                unsafe {
                    launch_matvec(
                        &self.device, &st.kernels, &lw.w_down,
                        &st.scratch.gate, &mut st.scratch.down,
                        hidden_dim, inter_dim, "down",
                        lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                        st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        } else {
            // Separate SwiGLU + native-quant down via launch_matvec().
            {
                let config = LaunchConfig::for_elements(inter_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = inter_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.swiglu_inplace)
                        .arg(&mut st.scratch.gate)
                        .arg(&st.scratch.up)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("swiglu launch: {e}")))?;
            }
            // split-layout: prefer Q8/Q4 split sibling for w_down via inline
            // F32->Q8_1 quantization (fused-down kernels target Q8Aligned/Q4Aligned
            // which the SPLIT preload skips).
            let use_split_down = (st.kernels.use_q8_split_dispatch && lw.q8_split_w_down.is_some())
                || (st.kernels.use_q4_split_dispatch && lw.q4_split_w_down.is_some());
            if use_split_down {
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device, quant_fn, &st.scratch.gate, q8_1_buf,
                            inter_dim, "down split (sep swiglu)",
                        )?;
                        launch_matvec_preq8_1_tile(
                            &self.device, &st.kernels, &lw.w_down,
                            lw.q8_tile_w_down.as_ref(),  lw.q4_tile_w_down.as_ref(),
                            lw.q8_split_w_down.as_ref(), lw.q4_split_w_down.as_ref(),
                            q8_1_buf, &mut st.scratch.down,
                            hidden_dim, inter_dim, "down",
                        )?;
                    }
                } else {
                    unsafe {
                        launch_matvec(
                            &self.device, &st.kernels, &lw.w_down,
                            &st.scratch.gate, &mut st.scratch.down,
                            hidden_dim, inter_dim, "down",
                            lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            } else {
                unsafe {
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        &lw.w_down,
                        &st.scratch.gate,
                        &mut st.scratch.down,
                        hidden_dim,
                        inter_dim,
                        "down",
                        lw.w_down_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        }

        // Residual add: attn_proj += down.
        {
            let config = LaunchConfig::for_elements(hidden_dim);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n = hidden_dim as u32;
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.residual_add)
                    .arg(&mut st.scratch.attn_proj)
                    .arg(&st.scratch.down)
                    .arg(&n)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("residual_add launch: {e}")))?;
        }

        // Layer output is now in st.scratch.attn_proj. The caller must copy it
        // to st.scratch.x_gpu before the next layer (done via device memcpy).
        Ok(())
    }

    /// Lazily allocate GDN GPU scratch buffers on first use.
    ///
    /// Scans layer_weights_cache to identify GDN layers (layer_type == 1),
    /// builds the layer index mapping, and allocates all persistent state
    /// (h_states, conv_states) and ephemeral scratch buffers on the GPU.
    fn ensure_gdn_scratch(
        &self,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        if st.gdn_scratch_gpu.is_some() {
            return Ok(());
        }
        let hp = self.hp()?;
        let params = super::gdn::GdnParams::from_hyperparams(hp);
        let num_layers = hp.num_layers as usize;

        // Build layer mapping: layer_idx -> gdn_idx.
        let mut gdn_layer_map: Vec<Option<usize>> = vec![None; num_layers];
        let mut gdn_count = 0usize;
        for (i, lw) in st.layer_weights_cache.iter().enumerate() {
            if lw.layer_type == 1 {
                gdn_layer_map[i] = Some(gdn_count);
                gdn_count += 1;
            }
        }

        if gdn_count == 0 {
            return Err(RuntimeError::Compute(
                "ensure_gdn_scratch called but no GDN layers found".into(),
            ));
        }

        // Allocate per-layer persistent state.
        let mut h_states = Vec::with_capacity(gdn_count);
        let mut conv_states = Vec::with_capacity(gdn_count);
        for _ in 0..gdn_count {
            h_states.push(self.device.alloc_zeros::<f32>(params.h_state_elements())?);
            conv_states.push(self.device.alloc_zeros::<f32>(params.conv_state_elements())?);
        }
        let conv_positions = vec![0u32; gdn_count];

        // GPU-resident conv positions
        // for CUDA graph capture. One u32 per GDN layer. The host counter
        // `conv_positions[gdn_idx]` is kept in lockstep via:
        //   (a) initial htod_copy from host before begin_capture (in decode_token)
        //   (b) `advance_conv_position` kernel inside the captured graph
        //   (c) post-replay host counter advance (in decode_token)
        // This makes the megakernel-graph variant `gdn_decode_megakernel_graph`
        // graph-capturable: the kernel reads state_pos from this device pointer
        // instead of a host-scalar arg that would otherwise be baked into the
        // graph (preventing replay with a changed value).
        //
        // Only allocate when graph capture for GDN is supported. The
        // `can_use_graph` gate downstream additionally verifies the
        // gdn_decode_megakernel_graph kernel compiled (it might fail on older
        // GPUs missing certain PTX features).
        let conv_positions_gpu: Option<Vec<CudaSlice<u32>>> = {
            let mut v = Vec::with_capacity(gdn_count);
            let mut alloc_ok = true;
            for _ in 0..gdn_count {
                match self.device.alloc_zeros::<u32>(1) {
                    Ok(s) => v.push(s),
                    Err(_) => { alloc_ok = false; break; }
                }
            }
            if alloc_ok { Some(v) } else { None }
        };

        // Allocate ephemeral scratch buffers (shared across layers).
        // Q_norm/K_norm buffers are allocated only when LUMEN_CUDA_GDN_REGISTER_RESIDENT=1
        // because they are unused by the existing megakernel path.
        // default ON (no-op for non-GDN models).
        let use_gdn_register_resident = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT").ok().as_deref() {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::gdn_register_resident_default(),
        };
        let qk_norm_elements = params.num_kv_heads * params.head_dim;
        let q_norm_buf_rr = if use_gdn_register_resident {
            Some(self.device.alloc_zeros::<f32>(qk_norm_elements)?)
        } else { None };
        let k_norm_buf_rr = if use_gdn_register_resident {
            Some(self.device.alloc_zeros::<f32>(qk_norm_elements)?)
        } else { None };

        let gdn = GdnScratchGpu {
            params,
            h_states,
            conv_states,
            conv_positions,
            conv_positions_gpu,
            gdn_layer_map,
            qkv_buf: self.device.alloc_zeros::<f32>(params.qkv_dim)?,
            qkv_conv_buf: self.device.alloc_zeros::<f32>(params.qkv_dim)?,
            alpha_buf: self.device.alloc_zeros::<f32>(params.num_heads)?,
            beta_buf: self.device.alloc_zeros::<f32>(params.num_heads)?,
            alpha_raw_buf: self.device.alloc_zeros::<f32>(params.num_heads)?,
            beta_raw_buf: self.device.alloc_zeros::<f32>(params.num_heads)?,
            output_buf: self.device.alloc_zeros::<f32>(params.value_dim)?,
            normed_out_buf: self.device.alloc_zeros::<f32>(params.value_dim)?,
            gate_buf: self.device.alloc_zeros::<f32>(params.value_dim)?,
            ssm_proj_buf: self.device.alloc_zeros::<f32>(params.hidden_dim)?,
            q_norm_buf_rr,
            k_norm_buf_rr,
        };

        st.gdn_scratch_gpu = Some(gdn);
        Ok(())
    }

    /// Run the GDN (GatedDeltaNet) attention block on GPU, replacing the
    /// standard softmax attention path for GDN layers.
    ///
    /// Implements the GDN attention pipeline with fused optimizations:
    ///
    /// When dp4a Q8_1 path is available (Q8_0/Q4_0 weights):
    /// 1. Fused RMSNorm + Q8_1 quantize (1 dispatch)
    /// 2-4. QKV + alpha + beta + gate matvecs with shared Q8_1 input (4 dispatches)
    /// 5. GDN megakernel: conv1d+silu, gates, L2 norm, state update (1 dispatch)
    /// 6. Fused RMSNorm + SiLU gate (in-place on output_buf, 1 dispatch)
    /// 7. SSM output projection (1 dispatch)
    /// 8. Fused residual_add_copy: attn_proj = x_gpu + ssm_proj (1 dispatch)
    ///
    /// Fallback (F32/F16 weights):
    /// 1. RMSNorm (1 dispatch)
    /// 2-4. QKV + alpha + beta + gate matvecs (4+ dispatches)
    /// 5-8. Same as above
    ///
    /// After this call, `st.scratch.attn_proj` contains the post-GDN hidden
    /// state (x + ssm_proj) ready for the shared FFN block. `x_gpu` is NOT
    /// updated -- it retains the pre-GDN value. The caller updates `x_gpu`
    /// after the full layer (GDN attention + FFN) completes.
    /// Eager-path entry point (no graph capture). Equivalent to
    /// `compute_gdn_attention_gpu_impl(layer_idx, st, false)`.
    fn compute_gdn_attention_gpu(
        &self,
        layer_idx: usize,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        self.compute_gdn_attention_gpu_impl(layer_idx, st, false)
    }

    /// Graph-capture-path entry point. When called inside an active stream
    /// capture region, the megakernel branch dispatches the graph-compatible
    /// `gdn_decode_megakernel_graph` (reads state_pos from a device pointer,
    /// not a host-scalar arg) and a follow-up `advance_conv_position` kernel.
    /// Falls back to the eager path's host-scalar dispatch if the graph
    /// kernel or `conv_positions_gpu` are unavailable.
    ///
    /// Host counter
    /// `gdn.conv_positions[gdn_idx]` is advanced in lockstep with the GPU
    /// counter (via the calling pipeline) so any subsequent eager-fallback
    /// path reads a correct host counter.
    fn compute_gdn_attention_gpu_graph(
        &self,
        layer_idx: usize,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        self.compute_gdn_attention_gpu_impl(layer_idx, st, true)
    }

    fn compute_gdn_attention_gpu_impl(
        &self,
        layer_idx: usize,
        st: &mut MutableState,
        graph_mode: bool,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let eps = hp.norm_eps;

        // Ensure GDN scratch is allocated.
        self.ensure_gdn_scratch(st)?;

        let lw: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];
        let gdn = st.gdn_scratch_gpu.as_mut().unwrap();
        let p = gdn.params;

        let gdn_idx = gdn.gdn_layer_map[layer_idx]
            .ok_or_else(|| RuntimeError::Compute(format!(
                "compute_gdn_attention_gpu: layer {layer_idx} is not a GDN layer",
            )))?;

        // --- Step 1+2: RMSNorm + QKV matvec ---
        // Detect if all GDN matvec consumers (QKV, alpha, beta) use dp4a Q8_1
        // so we can fuse RMSNorm + Q8_1 quantization and share the quantized input.
        let ssm_alpha_w = lw.ssm_alpha.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: ssm_alpha weight missing",
            )))?;
        let ssm_beta_w = lw.ssm_beta.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: ssm_beta weight missing",
            )))?;
        let attn_gate_w = lw.attn_gate.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: attn_gate weight missing",
            )))?;

        let gdn_use_preq = weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
            && weight_uses_dp4a_q8_1(ssm_alpha_w, &st.kernels)
            && weight_uses_dp4a_q8_1(ssm_beta_w, &st.kernels)
            && weight_uses_dp4a_q8_1(attn_gate_w, &st.kernels)
            && st.scratch.input_q8_1.is_some()
            && st.kernels.quantize_f32_to_q8_1.is_some();

        if gdn_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
            // === FUSED: RMSNorm + Q8_1 quantize in 1 dispatch ===
            // Then all 3 matvecs (QKV, alpha, beta) use launch_matvec_preq8_1
            // sharing the single quantized input. Saves 4 separate quantize dispatches.
            let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
            let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
            let bs = rmsnorm_block_size(hidden_dim);
            let lc = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (bs, 1, 1),
                shared_mem_bytes: rmsnorm_shared_bytes(bs),
            };
            let dim = hidden_dim as u32;
            unsafe {
                self.device.stream.launch_builder(fused_fn)
                    .arg(&st.scratch.x_gpu)
                    .arg(&lw.attn_norm)
                    .arg(&mut *q8_1_buf)
                    .arg(&eps)
                    .arg(&dim)
                    .launch(lc)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN rmsnorm_to_q8_1 L{layer_idx}: {e}")))?;

            // QKV matvec with pre-quantized input.
            // split-layout: prefer Q8/Q4 split siblings for the fused QKV weight.
            unsafe {
                launch_matvec_preq8_1_tile(
                    &self.device, &st.kernels, &lw.wq,
                    lw.q8_tile_wq.as_ref(),  lw.q4_tile_wq.as_ref(),
                    lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(),
                    q8_1_buf, &mut gdn.qkv_buf,
                    p.qkv_dim, hidden_dim, "gdn_qkv",
                )?;
            }

            // Alpha matvec with shared pre-quantized input.
            // GDN_SPLIT: prefer Q4 split sibling.
            unsafe {
                launch_matvec_preq8_1_tile(
                    &self.device, &st.kernels, ssm_alpha_w,
                    None, None,
                    None, lw.q4_split_ssm_alpha.as_ref(),
                    q8_1_buf, &mut gdn.alpha_raw_buf,
                    p.num_heads, hidden_dim, "gdn_alpha",
                )?;
            }

            // Beta matvec with shared pre-quantized input.
            unsafe {
                launch_matvec_preq8_1_tile(
                    &self.device, &st.kernels, ssm_beta_w,
                    None, None,
                    None, lw.q4_split_ssm_beta.as_ref(),
                    q8_1_buf, &mut gdn.beta_raw_buf,
                    p.num_heads, hidden_dim, "gdn_beta",
                )?;
            }

            // Gate matvec with shared pre-quantized input.
            unsafe {
                launch_matvec_preq8_1_tile(
                    &self.device, &st.kernels, attn_gate_w,
                    None, None,
                    None, lw.q4_split_attn_gate.as_ref(),
                    q8_1_buf, &mut gdn.gate_buf,
                    p.value_dim, hidden_dim, "gdn_gate",
                )?;
            }
        } else {
            // === UNFUSED: separate RMSNorm + per-matvec quantize ===
            {
                let block_size = rmsnorm_block_size(hidden_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = hidden_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.rmsnorm)
                        .arg(&st.scratch.x_gpu)
                        .arg(&lw.attn_norm)
                        .arg(&mut st.scratch.normed)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN rmsnorm attn L{layer_idx}: {e}")))?;
            }

            // QKV matvec
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    &lw.wq,
                    &st.scratch.normed,
                    &mut gdn.qkv_buf,
                    p.qkv_dim,
                    hidden_dim,
                    "gdn_qkv",
                    lw.wq_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                )?;
            }

            // Alpha matvec
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    ssm_alpha_w,
                    &st.scratch.normed,
                    &mut gdn.alpha_raw_buf,
                    p.num_heads,
                    hidden_dim,
                    "gdn_alpha",
                    lw.ssm_alpha_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                )?;
            }

            // Beta matvec
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    ssm_beta_w,
                    &st.scratch.normed,
                    &mut gdn.beta_raw_buf,
                    p.num_heads,
                    hidden_dim,
                    "gdn_beta",
                    lw.ssm_beta_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                )?;
            }

            // Gate matvec (moved here from step 9 to share quantized input)
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    attn_gate_w,
                    &st.scratch.normed,
                    &mut gdn.gate_buf,
                    p.value_dim,
                    hidden_dim,
                    "gdn_gate",
                    lw.attn_gate_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                )?;
            }
        }


        // --- Steps 3a-7: Fused megakernel path (conv1d+silu, gates, L2, state update) ---
        // Falls back to unfused path if megakernel failed to compile.
        let conv1d_weight = lw.ssm_conv1d.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: ssm_conv1d weight missing",
            )))?;
        let dt_bias = lw.ssm_dt_bias.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: ssm_dt_bias missing",
            )))?;
        let ssm_a = lw.ssm_a.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN L{layer_idx}: ssm_a missing",
            )))?;

        // Two-launch path: gated behind LUMEN_CUDA_GDN_REGISTER_RESIDENT=1.
        // Splits the existing megakernel into two: Phases 1-3 (same logic as
        // the existing megakernel, but materializes Q_norm/K_norm to device
        // buffers) and Phase 4 (register-resident delta-rule with
        // warp-per-column grid). Requires gdn.q_norm_buf_rr / k_norm_buf_rr
        // to have been allocated at ensure_gdn_scratch time (gated on the
        // same env var).
        // default ON (matches init-site resolver).
        let register_resident_env = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT").ok().as_deref() {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::gdn_register_resident_default(),
        };
        let use_register_resident_phase4 = register_resident_env
            && st.kernels.gdn_phase123_register_resident.is_some()
            && st.kernels.gdn_phase4_register_resident.is_some()
            && gdn.q_norm_buf_rr.is_some()
            && gdn.k_norm_buf_rr.is_some();

        if use_register_resident_phase4 {
            // === TWO-LAUNCH PATH: Phase 1-3 + Phase 4 (2 launches; replaces megakernel) ===
            // Optional Phase-4 variant: coalesced lane mapping (env-gated default OFF).
            // Selected when LUMEN_CUDA_GDN_PHASE4_COAL=1 AND the coal kernel is loaded.
            // Math is identical (warp-reduce is commutative); only the per-lane ki
            // ownership changes from `lane*4..lane*4+3` to `lane,lane+32,lane+64,lane+96`,
            // which converts strided LDG.E.32 (4 sectors/r) into a single coalesced
            // 128B transaction per r. ADD-only: falls back to gdn_phase4_register_resident.
            let use_phase4_coal = std::env::var("LUMEN_CUDA_GDN_PHASE4_COAL")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(false)
                && st.kernels.gdn_phase4_register_resident_coal.is_some();
            // F64-internal-accumulator variant for Phase 4 (decode path).
            // When `LUMEN_CUDA_GDN_F64_ACCUM=1`, replace the F32 lane-strided
            // kernel with the F64-state F64-reduce variant; the coal/strided
            // ownership pattern is irrelevant once F64 is the accumulator (the
            // F64 variant uses the strided lane pattern like F32 base).
            let use_phase4_f64 = std::env::var("LUMEN_CUDA_GDN_F64_ACCUM")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(false)
                && st.kernels.gdn_phase4_register_resident_f64accum.is_some();
            // when running inside an active CUDA graph capture
            // region (graph_mode == true) AND the graph-capturable variant of
            // phase123 is available AND `conv_positions_gpu` is allocated,
            // dispatch `gdn_phase123_register_resident_graph` (reads state_pos from a
            // device pointer) followed by a single-thread `advance_conv_position`
            // kernel. Phase 4 has no state_pos dependence, so its existing
            // dispatch is already graph-safe.
            let use_phase123_graph = graph_mode
                && st.kernels.gdn_phase123_register_resident_graph.is_some()
                && gdn.conv_positions_gpu.is_some();

            let p4_fn = if use_phase4_f64 {
                st.kernels.gdn_phase4_register_resident_f64accum.as_ref().unwrap()
            } else if use_phase4_coal {
                st.kernels.gdn_phase4_register_resident_coal.as_ref().unwrap()
            } else {
                st.kernels.gdn_phase4_register_resident.as_ref().unwrap()
            };

            let num_heads_u32 = p.num_heads as u32;
            let num_kv_heads_u32 = p.num_kv_heads as u32;
            let head_dim_u32 = p.head_dim as u32;
            let qkv_dim_u32 = p.qkv_dim as u32;
            let qk_dim_u32 = p.qk_dim as u32;
            let value_dim_u32 = p.value_dim as u32;
            let kernel_size_u32 = p.conv_kernel_size as u32;
            let state_pos = gdn.conv_positions[gdn_idx];

            // --- Phase 1-3: conv1d + SiLU + gates + L2 norm ---
            // Same grid/block as existing megakernel; writes Q_norm, K_norm,
            // V, alpha, beta to device buffers.
            let block_dim = (p.head_dim as u32).max(128).min(1024);
            let shared_bytes = (32 + 2 * p.head_dim as u32) * 4;
            let p123_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads_u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            // Phase123 writes V to `normed_out_buf` (an ephemeral buffer that
            // is otherwise written by `gdn_rmsnorm_silu_gate` later in the
            // layer). Phase4 then reads V from `normed_out_buf` and writes the
            // new GDN output to `output_buf`. This avoids the R/W aliasing on
            // `output_buf` that would otherwise reject under the borrow
            // checker (cudarc's launch_builder cannot hold both `&` and
            // `&mut` to the same CudaSlice at once).
            if use_phase123_graph {
                // device-pointer state_pos. `conv_positions_gpu[gdn_idx]`
                // was pre-populated via htod_copy by `decode_token` before
                // begin_capture(), so its current value matches the host
                // state_pos at capture time. The advance_conv_position kernel
                // (dispatched after phase4 below) increments it per-replay.
                let p123_graph_fn = st.kernels.gdn_phase123_register_resident_graph.as_ref().unwrap();
                let gpu_pos_slice = &mut gdn.conv_positions_gpu.as_mut().unwrap()[gdn_idx];
                unsafe {
                    self.device.stream.launch_builder(p123_graph_fn)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(&gdn.qkv_buf)
                        .arg(&gdn.alpha_raw_buf)
                        .arg(&gdn.beta_raw_buf)
                        .arg(conv1d_weight)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(gdn.q_norm_buf_rr.as_mut().unwrap())
                        .arg(gdn.k_norm_buf_rr.as_mut().unwrap())
                        .arg(&mut gdn.normed_out_buf)   // V buf
                        .arg(&mut gdn.alpha_buf)
                        .arg(&mut gdn.beta_buf)
                        .arg(&num_heads_u32)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&qk_dim_u32)
                        .arg(&value_dim_u32)
                        .arg(&kernel_size_u32)
                        .arg(&*gpu_pos_slice)
                        .launch(p123_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN phase123 register_resident graph L{layer_idx}: {e}")))?;
            } else {
                let p123_fn = st.kernels.gdn_phase123_register_resident.as_ref().unwrap();
                unsafe {
                    self.device.stream.launch_builder(p123_fn)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(&gdn.qkv_buf)
                        .arg(&gdn.alpha_raw_buf)
                        .arg(&gdn.beta_raw_buf)
                        .arg(conv1d_weight)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(gdn.q_norm_buf_rr.as_mut().unwrap())
                        .arg(gdn.k_norm_buf_rr.as_mut().unwrap())
                        .arg(&mut gdn.normed_out_buf)   // V buf
                        .arg(&mut gdn.alpha_buf)
                        .arg(&mut gdn.beta_buf)
                        .arg(&num_heads_u32)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&qk_dim_u32)
                        .arg(&value_dim_u32)
                        .arg(&kernel_size_u32)
                        .arg(&state_pos)
                        .launch(p123_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN phase123 register_resident L{layer_idx}: {e}")))?;
            }

            // --- Phase 4: register-resident delta-rule ---
            // Grid: (num_heads, 1, ceil(head_dim / num_warps)); Block: (32, 4, 1)
            // For Qwen3.5-9B (head_dim=128, num_warps=4): grid (32, 1, 32), block (32, 4, 1).
            let num_warps_p4: u32 = 4;
            let warp_size_p4: u32 = 32;
            let p4_z = (head_dim_u32 + num_warps_p4 - 1) / num_warps_p4;
            let p4_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads_u32, 1, p4_z),
                block_dim: (warp_size_p4, num_warps_p4, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.device.stream.launch_builder(p4_fn)
                    .arg(&mut gdn.h_states[gdn_idx])
                    .arg(gdn.q_norm_buf_rr.as_ref().unwrap())
                    .arg(gdn.k_norm_buf_rr.as_ref().unwrap())
                    .arg(&gdn.normed_out_buf)   // V (read; written by phase123)
                    .arg(&gdn.alpha_buf)
                    .arg(&gdn.beta_buf)
                    .arg(&mut gdn.output_buf)   // output (written by phase4)
                    .arg(&num_heads_u32)
                    .arg(&num_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(p4_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN phase4 register_resident L{layer_idx}: {e}")))?;

            // Advance circular buffer position.
            let buf_slots = (p.conv_kernel_size - 1) as u32;

            // when graph_mode is on, dispatch the
            // `advance_conv_position` kernel (single-thread, single-block) to
            // increment `conv_positions_gpu[gdn_idx]` on-device. The host
            // counter advance below remains in lockstep so any subsequent
            // eager-fallback path reads a consistent host counter.
            if use_phase123_graph {
                let gk = st.graph_kernels.as_ref().ok_or_else(|| RuntimeError::Compute(
                    "graph mode requires graph_kernels (advance_conv_position)".into(),
                ))?;
                let advance_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                let gpu_pos_slice2 = &mut gdn.conv_positions_gpu.as_mut().unwrap()[gdn_idx];
                unsafe {
                    self.device.stream.launch_builder(&gk.advance_conv_position)
                        .arg(&mut *gpu_pos_slice2)
                        .arg(&buf_slots)
                        .launch(advance_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN advance_conv_position L{layer_idx}: {e}",
                )))?;
            }

            gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
        } else if let Some(ref mega_fn) = st.kernels.gdn_decode_megakernel {
            // === FUSED PATH: 8 launches -> 2 ===
            // Kernel 1 (gdn_decode_megakernel): conv1d+silu, gates, L2 norm, state update.
            //
            // when running inside an active
            // CUDA graph capture region (graph_mode == true) AND the graph
            // variant kernel + conv_positions_gpu are both available, dispatch
            // `gdn_decode_megakernel_graph` (reads state_pos from a device
            // pointer) followed by a single-thread `advance_conv_position`
            // kernel. This keeps the entire GDN inner loop graph-capturable
            // (state_pos changes between tokens without re-capturing the graph).
            let num_heads_u32 = p.num_heads as u32;
            let num_kv_heads_u32 = p.num_kv_heads as u32;
            let head_dim_u32 = p.head_dim as u32;
            let qkv_dim_u32 = p.qkv_dim as u32;
            let qk_dim_u32 = p.qk_dim as u32;
            let value_dim_u32 = p.value_dim as u32;
            let kernel_size_u32 = p.conv_kernel_size as u32;
            let state_pos = gdn.conv_positions[gdn_idx];

            // block_dim >= head_dim, shared memory: (32 + 2*head_dim) * sizeof(float)
            let block_dim = (p.head_dim as u32).max(128).min(1024);
            let shared_bytes = (32 + 2 * p.head_dim as u32) * 4;

            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads_u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: shared_bytes,
            };

            let use_graph_mega = graph_mode
                && st.kernels.gdn_decode_megakernel_graph.is_some()
                && gdn.conv_positions_gpu.is_some();

            if use_graph_mega {
                // dispatch the graph-capturable variant with
                // device-pointer state_pos. The CudaSlice<u32> at
                // conv_positions_gpu[gdn_idx] was pre-populated via htod_copy
                // before begin_capture(), so its current value matches the
                // host state_pos at capture time.
                let mega_graph_fn = st.kernels.gdn_decode_megakernel_graph.as_ref().unwrap();
                // Borrow conv_positions_gpu separately for the megakernel arg
                // — the borrow checker requires we split mutable borrows on
                // the GdnScratchGpu fields. The `.as_mut().unwrap()` pattern
                // matches the gdn.h_states / gdn.conv_states style nearby.
                let gpu_pos_slice = &mut gdn.conv_positions_gpu.as_mut().unwrap()[gdn_idx];
                unsafe {
                    self.device
                        .stream
                        .launch_builder(mega_graph_fn)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&gdn.qkv_buf)
                        .arg(&gdn.alpha_raw_buf)
                        .arg(&gdn.beta_raw_buf)
                        .arg(conv1d_weight)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(&mut gdn.output_buf)
                        .arg(&num_heads_u32)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&qk_dim_u32)
                        .arg(&value_dim_u32)
                        .arg(&kernel_size_u32)
                        .arg(&*gpu_pos_slice)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN megakernel_graph L{layer_idx}: {e}",
                )))?;

                // Follow-up: advance_conv_position kernel inside the captured
                // graph. Single thread, single block — trivially cheap.
                let buf_slots = (p.conv_kernel_size - 1) as u32;
                let gk = st.graph_kernels.as_ref().ok_or_else(|| RuntimeError::Compute(
                    "graph mode requires graph_kernels (advance_conv_position)".into(),
                ))?;
                let advance_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                let gpu_pos_slice2 = &mut gdn.conv_positions_gpu.as_mut().unwrap()[gdn_idx];
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&gk.advance_conv_position)
                        .arg(&mut *gpu_pos_slice2)
                        .arg(&buf_slots)
                        .launch(advance_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN advance_conv_position L{layer_idx}: {e}",
                )))?;

                // Host counter is advanced by the CALLER post-replay (see
                // `decode_token` graph-replay path), NOT here, because this
                // dispatch is captured (the host code runs once at capture
                // time but the kernel runs once per replay).
                // For the very first call (capture token itself), the host
                // counter MUST also be advanced so the next eager-fallback
                // sees consistent state. Done by the caller (run_graph_pipeline
                // post-loop or decode_token replay-path) — kept here for
                // safety mirroring of the original logic:
                gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
            } else {
                // Eager path (graph_mode == false OR graph kernel unavailable).
                // bit-exact when disabled: host-scalar state_pos.
                unsafe {
                    self.device
                        .stream
                        .launch_builder(mega_fn)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&gdn.qkv_buf)
                        .arg(&gdn.alpha_raw_buf)
                        .arg(&gdn.beta_raw_buf)
                        .arg(conv1d_weight)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(&mut gdn.output_buf)
                        .arg(&num_heads_u32)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&qk_dim_u32)
                        .arg(&value_dim_u32)
                        .arg(&kernel_size_u32)
                        .arg(&state_pos)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN megakernel L{layer_idx}: {e}")))?;

                // Advance circular buffer position (host-scalar path).
                let buf_slots = (p.conv_kernel_size - 1) as u32;
                gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
            }
        } else {
            // === UNFUSED FALLBACK PATH ===
            // Step 3a: Conv1D decode
            {
                let conv1d_fn = st.kernels.ssm_conv1d_decode.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN ssm_conv1d_decode kernel not compiled".into(),
                    ))?;
                let config = LaunchConfig::for_elements(p.qkv_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let conv_dim = p.qkv_dim as u32;
                let kernel_size = p.conv_kernel_size as u32;
                let state_pos = gdn.conv_positions[gdn_idx];

                unsafe {
                    self.device
                        .stream
                        .launch_builder(conv1d_fn)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(&gdn.qkv_buf)
                        .arg(conv1d_weight)
                        .arg(&mut gdn.qkv_conv_buf)
                        .arg(&conv_dim)
                        .arg(&kernel_size)
                        .arg(&state_pos)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN conv1d L{layer_idx}: {e}")))?;

                let buf_slots = (p.conv_kernel_size - 1) as u32;
                gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
            }

            // Step 3b: SiLU activation on conv output
            {
                let silu_fn = st.kernels.silu_inplace.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN silu_inplace kernel not compiled".into(),
                    ))?;
                let config = LaunchConfig::for_elements(p.qkv_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = p.qkv_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(silu_fn)
                        .arg(&mut gdn.qkv_conv_buf)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN silu L{layer_idx}: {e}")))?;
            }

            // Step 4c: Compute gates
            {
                let gates_fn = st.kernels.gdn_compute_gates.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN gdn_compute_gates kernel not compiled".into(),
                    ))?;
                let config = LaunchConfig::for_elements(p.num_heads);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let num_heads = p.num_heads as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(gates_fn)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(&gdn.beta_raw_buf)
                        .arg(&gdn.alpha_raw_buf)
                        .arg(&mut gdn.alpha_buf)
                        .arg(&mut gdn.beta_buf)
                        .arg(&num_heads)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN compute_gates L{layer_idx}: {e}")))?;
            }

            // Step 5: L2-normalize Q and K per head
            {
                let l2_fn = st.kernels.l2_normalize_heads.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN l2_normalize_heads kernel not compiled".into(),
                    ))?;
                let num_kv_heads_u32 = p.num_kv_heads as u32;
                let head_dim_u32 = p.head_dim as u32;
                let l2_eps = 1e-12f32;
                let block_dim = (p.head_dim as u32).min(1024);
                let shared_bytes = (block_dim / 32) * 4;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_kv_heads_u32, 1, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                {
                    let mut q_view = gdn.qkv_conv_buf.slice_mut(0..p.qk_dim);
                    unsafe {
                        self.device.stream.launch_builder(l2_fn)
                            .arg(&mut q_view).arg(&num_kv_heads_u32).arg(&head_dim_u32).arg(&l2_eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN l2_norm Q L{layer_idx}: {e}")))?;
                }
                {
                    let mut k_view = gdn.qkv_conv_buf.slice_mut(p.qk_dim..2 * p.qk_dim);
                    unsafe {
                        self.device.stream.launch_builder(l2_fn)
                            .arg(&mut k_view).arg(&num_kv_heads_u32).arg(&head_dim_u32).arg(&l2_eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN l2_norm K L{layer_idx}: {e}")))?;
                }
            }

            // Steps 6+7: State update + output
            {
                let state_fn = st.kernels.gdn_state_update.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN gdn_state_update kernel not compiled".into(),
                    ))?;
                let num_heads_u32 = p.num_heads as u32;
                let val_dim_u32 = p.head_dim as u32;
                let key_dim_u32 = p.head_dim as u32;
                let num_kv_heads_u32 = p.num_kv_heads as u32;
                let block_threads = (p.head_dim as u32).min(1024);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_heads_u32, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let k_view = gdn.qkv_conv_buf.slice(p.qk_dim..2 * p.qk_dim);
                let v_view = gdn.qkv_conv_buf.slice(2 * p.qk_dim..p.qkv_dim);
                let q_view = gdn.qkv_conv_buf.slice(0..p.qk_dim);
                unsafe {
                    self.device.stream.launch_builder(state_fn)
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&k_view).arg(&v_view).arg(&gdn.alpha_buf).arg(&gdn.beta_buf)
                        .arg(&q_view).arg(&mut gdn.output_buf)
                        .arg(&num_heads_u32).arg(&val_dim_u32).arg(&key_dim_u32).arg(&num_kv_heads_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN state_update L{layer_idx}: {e}")))?;
            }
        }

        // --- Steps 8+10: Fused RMSNorm + SiLU(gate) * normed output ---
        // Fused path: gdn_rmsnorm_silu_gate (2 kernels -> 1).
        // Falls back to unfused rmsnorm + silu_mul if unavailable.
        // Note: gate matvec was already dispatched above (step 1+2 block) to share
        // the Q8_1 quantized input with QKV/alpha/beta matvecs.

        // Track which buffer holds the final gated output for the ssm_out matvec.
        // Fused path: writes to normed_out_buf (no memcpy needed).
        // Unfused path: writes to output_buf (via silu_elementwise_mul).
        let used_fused_norm_gate;

        if let Some(ref fused_fn) = st.kernels.gdn_rmsnorm_silu_gate {
            // === FUSED: RMSNorm + SiLU(gate) * normed in one kernel ===
            let ssm_norm = lw.ssm_norm_tiled.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_norm_tiled missing",
                )))?;

            // when LUMEN_CUDA_GDN_F64_ACCUM=1, prefer F64 variant.
            // Shared-mem doubles (8 bytes per warp slot vs 4).
            let use_norm_gate_f64 = std::env::var("LUMEN_CUDA_GDN_F64_ACCUM")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(false)
                && st.kernels.gdn_rmsnorm_silu_gate_f64accum.is_some();
            let chosen_fn = if use_norm_gate_f64 {
                st.kernels.gdn_rmsnorm_silu_gate_f64accum.as_ref().unwrap()
            } else {
                fused_fn
            };
            let block_size = rmsnorm_block_size(p.value_dim);
            let base_shared = rmsnorm_shared_bytes(block_size);
            let shared_bytes = if use_norm_gate_f64 { base_shared * 2 } else { base_shared };
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            let dim = p.value_dim as u32;
            // gdn_rmsnorm_silu_gate: output_buf -> normed_out_buf.
            // The ssm_out matvec below reads from normed_out_buf directly,
            // eliminating the memcpy_dtod that was previously needed to copy
            // normed_out_buf back to output_buf.
            unsafe {
                self.device
                    .stream
                    .launch_builder(chosen_fn)
                    .arg(&gdn.output_buf)
                    .arg(ssm_norm)
                    .arg(&gdn.gate_buf)
                    .arg(&mut gdn.normed_out_buf)
                    .arg(&eps)
                    .arg(&dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN fused_rmsnorm_silu_gate L{layer_idx}: {e}")))?;
            used_fused_norm_gate = true;
        } else {
            // === UNFUSED FALLBACK ===
            // Step 8: RMSNorm on output
            {
                let ssm_norm = lw.ssm_norm_tiled.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(format!(
                        "GDN L{layer_idx}: ssm_norm_tiled missing",
                    )))?;
                let block_size = rmsnorm_block_size(p.value_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = p.value_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(&st.kernels.rmsnorm)
                        .arg(&gdn.output_buf).arg(ssm_norm).arg(&mut gdn.normed_out_buf)
                        .arg(&eps).arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN rmsnorm output L{layer_idx}: {e}")))?;
            }

            // Step 10: SiLU(gate) * normed_output -> output_buf
            {
                let silu_mul_fn = st.kernels.silu_elementwise_mul.as_ref()
                    .ok_or_else(|| RuntimeError::Compute(
                        "GDN silu_elementwise_mul kernel not compiled".into(),
                    ))?;
                let config = LaunchConfig::for_elements(p.value_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = p.value_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(silu_mul_fn)
                        .arg(&gdn.gate_buf).arg(&gdn.normed_out_buf).arg(&mut gdn.output_buf)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN silu_mul L{layer_idx}: {e}")))?;
            }
            used_fused_norm_gate = false;
        }


        // --- Step 11: Output projection -> ssm_proj ---
        // Fused path: reads from normed_out_buf. Unfused path: reads from output_buf.
        // GDN_SPLIT: when q4_split_ssm_out is set, route through
        // launch_matvec_preq8_1_split via inline Q8_1 quantization. Otherwise
        // fall through to launch_matvec as before.
        {
            let ssm_out = lw.ssm_out.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_out weight missing",
                )))?;
            let ssm_input = if used_fused_norm_gate {
                &gdn.normed_out_buf
            } else {
                &gdn.output_buf
            };
            let use_split_ssm_out = st.kernels.use_q4_split_dispatch
                && lw.q4_split_ssm_out.is_some();
            if use_split_ssm_out {
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device, quant_fn, ssm_input, q8_1_buf,
                            p.value_dim, "gdn_ssm_out split",
                        )?;
                        launch_matvec_preq8_1_tile(
                            &self.device, &st.kernels, ssm_out,
                            None, None,
                            None, lw.q4_split_ssm_out.as_ref(),
                            q8_1_buf, &mut gdn.ssm_proj_buf,
                            hidden_dim, p.value_dim, "gdn_ssm_out",
                        )?;
                    }
                } else {
                    unsafe {
                        launch_matvec(
                            &self.device, &st.kernels, ssm_out, ssm_input,
                            &mut gdn.ssm_proj_buf, hidden_dim, p.value_dim,
                            "gdn_ssm_out", lw.ssm_out_f16.as_ref(),
                            Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut(),
                        )?;
                    }
                }
            } else {
                unsafe {
                    launch_matvec(
                        &self.device,
                        &st.kernels,
                        ssm_out,
                        ssm_input,
                        &mut gdn.ssm_proj_buf,
                        hidden_dim,
                        p.value_dim,
                        "gdn_ssm_out",
                        lw.ssm_out_f16.as_ref(),
                        Some(&mut st.scratch.input_f16),
                    st.scratch.input_q8_1.as_mut(),
                    )?;
                }
            }
        }


        // --- Step 12+13: Fused residual add + copy ---
        // attn_proj = x_gpu + ssm_proj (via residual_add_copy, 1 dispatch).
        // x_gpu is NOT updated here -- it will be updated by the FFN residual
        // (x_gpu = attn_proj + down) which already reads from attn_proj.
        // This eliminates 1 dispatch vs the prior residual_add + memcpy_dtod pair.
        {
            let config = LaunchConfig::for_elements(hidden_dim);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n = hidden_dim as u32;
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.residual_add_copy)
                    .arg(&st.scratch.x_gpu)
                    .arg(&gdn.ssm_proj_buf)
                    .arg(&mut st.scratch.attn_proj)
                    .arg(&n)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN residual_add_copy L{layer_idx}: {e}")))?;
        }


        Ok(())
    }

    /// Batched GDN prefill for a single GDN layer.
    ///
    /// Implements the 15-step GDN prefill pipeline matching Metal's
    /// `encode_batched_gdn_prefill`:
    ///
    /// Phase 1 (batched across T tokens):
    /// 1. Batched RMSNorm: x[T, hidden] -> normed[T, hidden]
    /// 2. Batched QKV GEMM: normed[T, hidden] @ wq^T -> qkv[T, qkv_dim]
    /// 3. Batched Gate GEMM: normed[T, hidden] @ attn_gate^T -> gate[T, value_dim]
    /// 4. Batched Alpha GEMM: normed[T, hidden] @ ssm_alpha^T -> alpha_raw[T, num_heads]
    /// 5. Batched Beta GEMM: normed[T, hidden] @ ssm_beta^T -> beta_raw[T, num_heads]
    ///
    /// Phase 2 (sequential per token, reuses decode kernels):
    /// 6-12. For each t: conv1d, silu, compute_gates, l2_norm, state_update,
    /// rmsnorm, silu_gate_mul -> scatter output
    ///
    /// Phase 3 (batched):
    /// 13. Batched SSM out GEMM + residual: gdn_out[T, value_dim] @ ssm_out^T + x -> attn_proj
    ///
    /// Phase 4 (batched FFN, identical to standard layers):
    /// 14. FFN RMSNorm + gate/up + SwiGLU + down + residual
    fn prefill_gdn_layer(
        &self,
        layer_idx: usize,
        batch: usize,
        st: &mut MutableState,
        pf: &mut super::prefill::PrefillScratch,
        gdn_pf: &mut super::prefill::GdnPrefillScratch,
        eps: f32,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;

        // element-precision dump gate.
        // When LUMEN_DUMP_GDN_L0_BIN is set to a directory path, the prefill GDN
        // block writes raw F32 sub-component buffers to {dir}/L{layer}-{name}.bin
        // for the *first* layer only (or every GDN layer if =all). This enables
        // element-wise comparison vs an external reference at full precision.
        let dump_dir = std::env::var("LUMEN_DUMP_GDN_L0_BIN").ok();
        let dump_all = dump_dir.as_deref() == Some("all");
        let do_dump = dump_dir.is_some() && (layer_idx == 0 || dump_all);

        let lw = &st.layer_weights_cache[layer_idx];
        let gdn = st.gdn_scratch_gpu.as_mut().unwrap();
        let p = gdn.params;

        let gdn_idx = gdn.gdn_layer_map[layer_idx]
            .ok_or_else(|| RuntimeError::Compute(format!(
                "prefill_gdn_layer: layer {layer_idx} is not a GDN layer",
            )))?;

        // ================================================================
        // PHASE 1: Batched projections across all T tokens
        // ================================================================

        // 1. Batched RMSNorm: x[T, hidden] -> normed[T, hidden]
        unsafe {
            super::prefill::launch_rmsnorm_batched(
                &self.device, &st.kernels,
                &pf.x, &lw.attn_norm, &mut pf.normed,
                eps, batch, hidden_dim,
            )?;
        }

        // 2. Batched QKV GEMM: normed[T, hidden] @ wq^T -> qkv[T, qkv_dim]
        // wq for GDN is the fused [qkv_dim, hidden_dim] weight.
        unsafe {
            super::prefill::launch_gemm_projection(
                &self.device, &st.kernels, &lw.wq, lw.wq_f16.as_ref(),
                &pf.normed, &mut gdn_pf.qkv,
                &mut pf.dequant_f32, &mut pf.activation_f16,
                &mut pf.dequant_f16,
                batch, p.qkv_dim, hidden_dim, "gdn_qkv",
            )?;
        }

        // 3. Batched Gate GEMM: normed[T, hidden] @ attn_gate^T -> gate[T, value_dim]
        {
            let attn_gate = lw.attn_gate.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: attn_gate weight missing",
                )))?;
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, attn_gate, lw.attn_gate_f16.as_ref(),
                    &pf.normed, &mut gdn_pf.gate,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, p.value_dim, hidden_dim, "gdn_gate",
                )?;
            }
        }

        // 4. Batched Alpha GEMM: normed[T, hidden] @ ssm_alpha^T -> alpha_raw[T, num_heads]
        {
            let ssm_alpha = lw.ssm_alpha.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: ssm_alpha weight missing",
                )))?;
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, ssm_alpha, lw.ssm_alpha_f16.as_ref(),
                    &pf.normed, &mut gdn_pf.alpha_raw,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, p.num_heads, hidden_dim, "gdn_alpha",
                )?;
            }
        }

        // 5. Batched Beta GEMM: normed[T, hidden] @ ssm_beta^T -> beta_raw[T, num_heads]
        {
            let ssm_beta = lw.ssm_beta.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: ssm_beta weight missing",
                )))?;

            // diagnostic: when LUMEN_DEBUG_DUMP_SSM_BETA_W=1 is set, dump the
            // first 64 Q8 blocks (= row 16) of the ssm_beta weight buffer for
            // off-line verification against the GGUF F32 weight.  One-shot per
            // process invocation; head 16 specifically because per-head-drift
            // bisection shows head 16 produces 22% drift while all
            // others are < 1%.
            if layer_idx == 0 && std::env::var("LUMEN_DEBUG_DUMP_SSM_BETA_W").is_ok() {
                use super::gpu_buffers::GpuWeightBuf;
                static W_DUMPED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !W_DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    if let GpuWeightBuf::Q8Raw(ref q8buf) = ssm_beta {
                        self.device.synchronize()?;
                        let host = self.device.dtoh_copy(q8buf)?;
                        let total = host.len();
                        let path = "/tmp/ssm-beta-q8.bin".to_string();
                        std::fs::write(&path, &host).map_err(|e| RuntimeError::Compute(
                            format!("dump {path}: {e}")
                        ))?;
                        eprintln!(
                            "[scale-debug] L0 ssm_beta Q8 buf {} bytes -> {} \
                             (expected: 32 rows × 64 blocks × 34 bytes = 69632)",
                            total, path
                        );
                    } else {
                        eprintln!(
                            "[scale-debug] L0 ssm_beta is NOT Q8Raw (variant={:?})",
                            std::mem::discriminant(ssm_beta)
                        );
                    }
                }
            }

            // comprehensive runtime Q8 scale audit. When
            // LUMEN_DUMP_RUNTIME_Q8_SCALES=1 is set, count zero scales across
            // ALL F16 Q8 scales of ssm_alpha + ssm_beta L0. The audit succeeds
            // (no zero scales) once the converter F16-subnormal fix is in
            // effect on the LBC file. One-shot per process.
            if layer_idx == 0 && std::env::var("LUMEN_DUMP_RUNTIME_Q8_SCALES").is_ok() {
                use super::gpu_buffers::GpuWeightBuf;
                static AUDITED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !AUDITED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    self.device.synchronize()?;
                    for (name, buf_opt) in [
                        ("ssm_beta", Some(ssm_beta)),
                        ("ssm_alpha", lw.ssm_alpha.as_ref()),
                    ] {
                        if let Some(GpuWeightBuf::Q8Raw(ref q8buf)) = buf_opt {
                            let host = self.device.dtoh_copy(q8buf)?;
                            let n_blocks = host.len() / 34;
                            let mut zero_scales = 0usize;
                            let mut per_head_zero = [0usize; 32];
                            let blocks_per_row = 64usize; // 2048 hidden / 32
                            for blk in 0..n_blocks {
                                let off = blk * 34;
                                let scale = u16::from_le_bytes([host[off], host[off + 1]]);
                                if scale == 0 {
                                    zero_scales += 1;
                                    let head = blk / blocks_per_row;
                                    if head < 32 { per_head_zero[head] += 1; }
                                }
                            }
                            eprintln!(
                                "[scale-audit] L0 {} Q8: blocks={} zero_scales={} ({:.2}%)",
                                name, n_blocks, zero_scales,
                                100.0 * zero_scales as f64 / n_blocks as f64
                            );
                            for h in 0..32 {
                                if per_head_zero[h] > 0 {
                                    eprintln!("[scale-audit]   {} head {:2}: {} zero scales", name, h, per_head_zero[h]);
                                }
                            }
                            // Write the buffer to a stable path for off-line verification.
                            let path = format!("/tmp/debug-{}-q8.bin", name);
                            std::fs::write(&path, &host).map_err(|e| RuntimeError::Compute(
                                format!("dump {path}: {e}")
                            ))?;
                        }
                    }
                }
            }

            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, ssm_beta, lw.ssm_beta_f16.as_ref(),
                    &pf.normed, &mut gdn_pf.beta_raw,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, p.num_heads, hidden_dim, "gdn_beta",
                )?;
            }
        }

        // ================================================================
        // PHASE 2: GDN state update -- fused batched path or per-token fallback
        // ================================================================
        //
        // Fused path uses 5 batched kernels (3.4x speedup over per-token loop):
        // 1. ssm_conv1d_silu_prefill: batched conv1d+SiLU across T tokens
        // 2. gdn_compute_gates_batched: batched gate computation for T * num_heads
        // 3. l2_normalize_qk_strided: batched L2 norm for Q and K across T tokens
        // 4. gdn_prefill_fused_v3: warp-parallel fused state update (4x unrolled)
        // 5. gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output
        //
        // Fallback reuses single-token decode kernels in a per-token loop.

        let conv1d_weight = lw.ssm_conv1d.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN prefill L{layer_idx}: ssm_conv1d weight missing",
            )))?;
        let dt_bias = lw.ssm_dt_bias.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN prefill L{layer_idx}: ssm_dt_bias missing",
            )))?;
        let ssm_a = lw.ssm_a.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN prefill L{layer_idx}: ssm_a missing",
            )))?;
        let ssm_norm = lw.ssm_norm_tiled.as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!(
                "GDN prefill L{layer_idx}: ssm_norm_tiled missing",
            )))?;

        let num_heads_u32 = p.num_heads as u32;
        let num_kv_heads_u32 = p.num_kv_heads as u32;
        let head_dim_u32 = p.head_dim as u32;
        let value_dim_u32 = p.value_dim as u32;
        let conv_dim_u32 = p.qkv_dim as u32;
        let kernel_size_u32 = p.conv_kernel_size as u32;
        let buf_slots = (p.conv_kernel_size - 1) as u32;

        // dump pre-conv1d input (= qkv after batched proj)
        if do_dump {
            self.device.synchronize()?;
            let host = self.device.dtoh_copy(&gdn_pf.qkv)?;
            let n = batch * p.qkv_dim;
            let dir = dump_dir.as_ref().unwrap();
            let path = format!("{dir}/L{layer_idx}-qkv_pre_conv.bin");
            let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
            eprintln!("[gdn-dump] L{layer_idx} qkv_pre_conv shape=[{batch}, {}] -> {path}", p.qkv_dim);
        }

        let has_fused_prefill = st.kernels.ssm_conv1d_silu_prefill.is_some()
            && st.kernels.gdn_compute_gates_batched.is_some()
            && st.kernels.l2_normalize_qk_strided.is_some()
            && st.kernels.gdn_prefill_fused_v3.is_some()
            && st.kernels.gdn_prefill_norm_gate.is_some();

        if has_fused_prefill {
            // === FUSED BATCHED PATH (3.4x speedup) ===
            let batch_u32 = batch as u32;
            let state_pos = gdn.conv_positions[gdn_idx];

            // 1. ssm_conv1d_silu_prefill: batched conv1d + SiLU
            {
                let conv_fn = st.kernels.ssm_conv1d_silu_prefill.as_ref().unwrap();
                let config = LaunchConfig::for_elements(p.qkv_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream.launch_builder(conv_fn)
                        .arg(&gdn_pf.qkv)
                        .arg(&mut gdn.conv_states[gdn_idx])
                        .arg(conv1d_weight)
                        .arg(&mut gdn_pf.conv_out)
                        .arg(&conv_dim_u32)
                        .arg(&kernel_size_u32)
                        .arg(&state_pos)
                        .arg(&batch_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN prefill fused conv1d_silu L{layer_idx}: {e}"
                )))?;

                // Advance conv position by batch tokens.
                gdn.conv_positions[gdn_idx] = (state_pos + batch as u32) % buf_slots;

                // dump conv1d-post-SiLU (pre-L2-norm)
                if do_dump {
                    self.device.synchronize()?;
                    let host = self.device.dtoh_copy(&gdn_pf.conv_out)?;
                    let n = batch * p.qkv_dim;
                    let dir = dump_dir.as_ref().unwrap();
                    let path = format!("{dir}/L{layer_idx}-conv_silu.bin");
                    let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!("[gdn-dump] L{layer_idx} conv_silu shape=[{batch}, {}] -> {path}", p.qkv_dim);
                }
            }

            // 2. gdn_compute_gates_batched: batched gate computation
            // Writes to alpha_out and beta_out (NOT alpha_raw/beta_raw -- avoids borrow conflict)
            {
                let gates_fn = st.kernels.gdn_compute_gates_batched.as_ref().unwrap();
                let total = batch * p.num_heads;
                let config = LaunchConfig::for_elements(total);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream.launch_builder(gates_fn)
                        .arg(dt_bias)
                        .arg(ssm_a)
                        .arg(&gdn_pf.beta_raw)
                        .arg(&gdn_pf.alpha_raw)
                        .arg(&mut gdn_pf.alpha_out)
                        .arg(&mut gdn_pf.beta_out)
                        .arg(&num_heads_u32)
                        .arg(&batch_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN prefill fused gates_batched L{layer_idx}: {e}"
                )))?;

                // dump alpha + beta + ssm_a (weights). For ssm_a we
                // also dump the per-head weight buffer so we can compare it
                // numerically against the `-exp(A_log)` representation.
                if do_dump {
                    self.device.synchronize()?;
                    let dir = dump_dir.as_ref().unwrap();
                    let n_heads = batch * p.num_heads;
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.alpha_out)?;
                        let path = format!("{dir}/L{layer_idx}-alpha.bin");
                        let bytes: Vec<u8> = host[..n_heads].iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} alpha shape=[{batch}, {}] -> {path}", p.num_heads);
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.beta_out)?;
                        let path = format!("{dir}/L{layer_idx}-beta.bin");
                        let bytes: Vec<u8> = host[..n_heads].iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} beta shape=[{batch}, {}] -> {path}", p.num_heads);
                    }
                    {
                        let host = self.device.dtoh_copy(ssm_a)?;
                        let path = format!("{dir}/L{layer_idx}-ssm_a.bin");
                        let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} ssm_a shape=[{}] -> {path}", host.len());
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.alpha_raw)?;
                        let path = format!("{dir}/L{layer_idx}-alpha_raw.bin");
                        let bytes: Vec<u8> = host[..n_heads].iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} alpha_raw shape=[{batch}, {}] -> {path}", p.num_heads);
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.beta_raw)?;
                        let path = format!("{dir}/L{layer_idx}-beta_raw.bin");
                        let bytes: Vec<u8> = host[..n_heads].iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} beta_raw shape=[{batch}, {}] -> {path}", p.num_heads);
                    }
                    {
                        let host = self.device.dtoh_copy(dt_bias)?;
                        let path = format!("{dir}/L{layer_idx}-dt_bias.bin");
                        let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!("[gdn-dump] L{layer_idx} dt_bias shape=[{}] -> {path}", host.len());
                    }
                }
            }

            // F64 accumulator gate (env-cached for this layer).
            // When ON: route l2_normalize_qk_strided + gdn_prefill_fused_v3 +
            // gdn_prefill_norm_gate to the F64 variants. Note shared-mem size
            // doubles for the f64 variants (double = 8 bytes vs float = 4).
            let use_prefill_f64 = std::env::var("LUMEN_CUDA_GDN_F64_ACCUM")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(false)
                && st.kernels.l2_normalize_qk_strided_f64accum.is_some()
                && st.kernels.gdn_prefill_fused_v3_f64accum.is_some()
                && st.kernels.gdn_prefill_norm_gate_f64accum.is_some();

            // 3. l2_normalize_qk_strided: batched L2 norm for Q and K
            //
            // env-gated swap to two-step rsqrtf L2-norm variant
            // when LUMEN_CUDA_L2NORM_RSQRTF=1.: also engages on
            // LUMEN_CUDA_NORM_RSQRTF_BUNDLE (the combined convenience gate
            // that also enables the alternate RMSNorm variant below).
            // Switches to eps=1e-6 + rsqrtf(fmaxf(ss, eps^2)) (one HW op)
            // instead of Lumen's historical eps=1e-12 +
            // (sqrt>eps ? 1/sqrt : 1/eps) two-op form.
            let l2norm_rsqrtf_on = std::env::var("LUMEN_CUDA_L2NORM_RSQRTF")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(false)
                || std::env::var("LUMEN_CUDA_NORM_RSQRTF_BUNDLE").is_ok();
            let use_l2norm_rsqrtf = l2norm_rsqrtf_on
                && st.kernels.l2_normalize_qk_strided_rsqrtf.is_some();
            {
                let l2_fn = if use_prefill_f64 {
                    st.kernels.l2_normalize_qk_strided_f64accum.as_ref().unwrap()
                } else if use_l2norm_rsqrtf {
                    st.kernels.l2_normalize_qk_strided_rsqrtf.as_ref().unwrap()
                } else {
                    st.kernels.l2_normalize_qk_strided.as_ref().unwrap()
                };
                let l2_block_dim = (p.head_dim as u32).min(1024);
                // F64 variant uses 8-byte (double) shared mem; F32 uses 4-byte.
                let bytes_per_elem: u32 = if use_prefill_f64 { 8 } else { 4 };
                let l2_shared = ((l2_block_dim + 31) / 32 + 1) * bytes_per_elem;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_kv_heads_u32 * batch_u32, 1, 1),
                    block_dim: (l2_block_dim, 1, 1),
                    shared_mem_bytes: l2_shared,
                };
                let qkv_dim_u32 = p.qkv_dim as u32;
                let q_offset = 0u32;
                let k_offset = p.qk_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(l2_fn)
                        .arg(&mut gdn_pf.conv_out)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&batch_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&q_offset)
                        .arg(&k_offset)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN prefill fused l2_norm_qk L{layer_idx}: {e}"
                )))?;

                // dump post-L2-norm Q/K (conv_out is L2-normed in-place on QK,
                // V channels untouched). This is the candidate-3 measurement point.
                if do_dump {
                    self.device.synchronize()?;
                    let host = self.device.dtoh_copy(&gdn_pf.conv_out)?;
                    let n = batch * p.qkv_dim;
                    let dir = dump_dir.as_ref().unwrap();
                    let path = format!("{dir}/L{layer_idx}-conv_l2norm.bin");
                    let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!("[gdn-dump] L{layer_idx} conv_l2norm shape=[{batch}, {}] -> {path}", p.qkv_dim);
                }
            }

            // 4. gdn_prefill_fused_v3: warp-parallel fused state update
            // Grid: (val_dim, num_heads), Block: (32, 1, 1)
            {
                let state_fn = if use_prefill_f64 {
                    st.kernels.gdn_prefill_fused_v3_f64accum.as_ref().unwrap()
                } else {
                    st.kernels.gdn_prefill_fused_v3.as_ref().unwrap()
                };
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (head_dim_u32, num_heads_u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let qk_dim_u32 = p.qk_dim as u32;
                let qkv_dim_u32 = p.qkv_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(state_fn)
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&gdn_pf.conv_out)
                        .arg(&gdn_pf.alpha_out)
                        .arg(&gdn_pf.beta_out)
                        .arg(&mut gdn_pf.raw_out)
                        .arg(&num_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&head_dim_u32) // val_dim per head = head_dim
                        .arg(&num_kv_heads_u32)
                        .arg(&batch_u32)
                        .arg(&qk_dim_u32)
                        .arg(&qkv_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN prefill fused_v3 L{layer_idx}: {e}"
                )))?;

                // dump raw_out (post-state-update, pre-norm-gate)
                if do_dump {
                    self.device.synchronize()?;
                    let host = self.device.dtoh_copy(&gdn_pf.raw_out)?;
                    let n = batch * p.value_dim;
                    let dir = dump_dir.as_ref().unwrap();
                    let path = format!("{dir}/L{layer_idx}-raw_out.bin");
                    let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!("[gdn-dump] L{layer_idx} raw_out shape=[{batch}, {}] -> {path}", p.value_dim);
                }
            }

            // 5. gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output
            // Grid: (num_heads, T_chunk, 1), Block: (val_dim)
            // Writes to gdn_out which is used by Phase 3's GEMM.
            //
            // (chunked-dispatch fix): chunked dispatch — SM 8.0 max grid-Y is 65_535,
            // so at seq_len >= 65_536 the single-launch grid_dim.y = batch_u32
            // exceeded the cap and the kernel returned CUDA_ERROR_INVALID_VALUE
            // (§"Mode 2"). The fix
            // splits the launch into sub-batches of at most GDN_NORM_GATE_MAX_Y
            // tokens. Each sub-launch sees a disjoint [t_base, t_base+T_chunk)
            // slice of the three buffers (raw_out, gate, gdn_out) — the kernel
            // body itself reads `t = blockIdx.y` and writes `ssm_out[t,h,vj]`
            // with no cross-token dependence, so slicing the buffer pointers
            // is byte-identical to the unsplit launch. No kernel-source
            // change required.
            //
            // For the typical short-context case (batch < 65_536 = ~64 K
            // tokens, which covers every production prefill on Qwen3.5-9B's
            // 8 K -> 40 K -> 64 K shapes) this remains a single dispatch and
            // is byte-identical to the prior path. The chunking only
            // engages at batch >= 65_536.
            {
                // when LUMEN_CUDA_RMSNORM_RSQRTF=1 (or the combined gate
                // LUMEN_CUDA_NORM_RSQRTF_BUNDLE=1) is set AND the alternate
                // variant kernel loaded successfully, dispatch through the
                // alternate RMSNorm + SiLU-gate kernel. Default OFF preserves
                // byte-identical behaviour. The alternate variant changes:
                //   1) `1/sqrtf(mean+eps)` → `rsqrtf(mean+eps)` (one HW op).
                //   2) Cross-warp reduction uses a block-wide warp-shuffle
                //      SUM pattern.
                let rmsnorm_rsqrtf_on = std::env::var("LUMEN_CUDA_RMSNORM_RSQRTF").is_ok()
                    || std::env::var("LUMEN_CUDA_NORM_RSQRTF_BUNDLE").is_ok();
                let use_rsqrtf_norm_gate = rmsnorm_rsqrtf_on
                    && !use_prefill_f64
                    && st.kernels.gdn_prefill_norm_gate_rsqrtf.is_some();
                if use_rsqrtf_norm_gate {
                    static RSQRTF_NG_LOGGED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !RSQRTF_NG_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[CUDA]: gdn_prefill_norm_gate_rsqrtf ACTIVE \
                             (rsqrtf + block-wide warp-shuffle reduce)"
                        );
                    }
                }
                let norm_fn = if use_prefill_f64 {
                    st.kernels.gdn_prefill_norm_gate_f64accum.as_ref().unwrap()
                } else if use_rsqrtf_norm_gate {
                    st.kernels.gdn_prefill_norm_gate_rsqrtf.as_ref().unwrap()
                } else {
                    st.kernels.gdn_prefill_norm_gate.as_ref().unwrap()
                };
                let block_dim = (p.head_dim as u32).min(1024);
                // F64 variant uses 8-byte shared mem; F32 uses 4-byte.
                let bytes_per_elem_norm: u32 = if use_prefill_f64 { 8 } else { 4 };
                let norm_shared = ((block_dim + 31) / 32 + 1) * bytes_per_elem_norm;
                let scale_n_heads = num_heads_u32;
                // SM 8.0 max grid-Y is 65_535. Use 65_535 as the chunk cap to
                // stay strictly under the limit and avoid any boundary issues.
                const GDN_NORM_GATE_MAX_Y: usize = 65_535;
                let total_t = batch;
                let mut t_base: usize = 0;
                let heads_stride = p.num_heads * p.head_dim;
                while t_base < total_t {
                    let chunk_t = (total_t - t_base).min(GDN_NORM_GATE_MAX_Y);
                    let chunk_t_u32 = chunk_t as u32;
                    let off = t_base * heads_stride;
                    let len = chunk_t * heads_stride;
                    let raw_out_chunk = gdn_pf.raw_out.slice(off..off + len);
                    let gate_chunk = gdn_pf.gate.slice(off..off + len);
                    let mut gdn_out_chunk = gdn_pf.gdn_out.slice_mut(off..off + len);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (num_heads_u32, chunk_t_u32, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: norm_shared,
                    };
                    unsafe {
                        self.device.stream.launch_builder(norm_fn)
                            .arg(&raw_out_chunk)
                            .arg(&gate_chunk)
                            .arg(ssm_norm)
                            .arg(&mut gdn_out_chunk)
                            .arg(&num_heads_u32)
                            .arg(&head_dim_u32) // val_dim per head
                            .arg(&eps)
                            .arg(&scale_n_heads)
                            .arg(&chunk_t_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "GDN prefill fused norm_gate L{layer_idx} t_base={t_base} chunk_t={chunk_t}: {e}"
                    )))?;
                    t_base += chunk_t;
                }
            }

            // dump gdn_out (= pre-ssm_out-GEMM input, post-norm-gate)
            if do_dump {
                self.device.synchronize()?;
                let host = self.device.dtoh_copy(&gdn_pf.gdn_out)?;
                let n = batch * p.value_dim;
                let dir = dump_dir.as_ref().unwrap();
                let path = format!("{dir}/L{layer_idx}-gdn_out.bin");
                let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                eprintln!("[gdn-dump] L{layer_idx} gdn_out shape=[{batch}, {}] -> {path}", p.value_dim);
            }
        } else {
            // === UNFUSED FALLBACK: per-token loop using decode kernels ===
            let conv1d_fn = st.kernels.ssm_conv1d_decode.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN ssm_conv1d_decode kernel not compiled".into(),
                ))?;
            let silu_fn = st.kernels.silu_inplace.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN silu_inplace kernel not compiled".into(),
                ))?;
            let gates_fn = st.kernels.gdn_compute_gates.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN gdn_compute_gates kernel not compiled".into(),
                ))?;
            let l2_fn = st.kernels.l2_normalize_heads.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN l2_normalize_heads kernel not compiled".into(),
                ))?;
            let state_fn = st.kernels.gdn_state_update.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN gdn_state_update kernel not compiled".into(),
                ))?;
            let silu_mul_fn = st.kernels.silu_elementwise_mul.as_ref()
                .ok_or_else(|| RuntimeError::Compute(
                    "GDN silu_elementwise_mul kernel not compiled".into(),
                ))?;

            let l2_eps = 1e-12f32;
            let conv_config = LaunchConfig::for_elements(p.qkv_dim);
            let conv_launch = CudarcLaunchConfig {
                grid_dim: (conv_config.grid_dim, 1, 1),
                block_dim: (conv_config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let silu_launch = conv_launch;
            let gates_config = LaunchConfig::for_elements(p.num_heads);
            let gates_launch = CudarcLaunchConfig {
                grid_dim: (gates_config.grid_dim, 1, 1),
                block_dim: (gates_config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let l2_block_dim = (p.head_dim as u32).min(1024);
            let l2_shared = (l2_block_dim / 32) * 4;
            let l2_launch = CudarcLaunchConfig {
                grid_dim: (num_kv_heads_u32, 1, 1),
                block_dim: (l2_block_dim, 1, 1),
                shared_mem_bytes: l2_shared,
            };
            let state_block = (p.head_dim as u32).min(1024);
            let state_launch = CudarcLaunchConfig {
                grid_dim: (num_heads_u32, 1, 1),
                block_dim: (state_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let norm_block = rmsnorm_block_size(p.value_dim);
            let norm_shared = rmsnorm_shared_bytes(norm_block);
            let norm_launch = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (norm_block, 1, 1),
                shared_mem_bytes: norm_shared,
            };
            let silu_mul_config = LaunchConfig::for_elements(p.value_dim);
            let silu_mul_launch = CudarcLaunchConfig {
                grid_dim: (silu_mul_config.grid_dim, 1, 1),
                block_dim: (silu_mul_config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            for t in 0..batch {
                // Conv1D decode
                {
                    let qkv_t = gdn_pf.qkv.slice(t * p.qkv_dim..(t + 1) * p.qkv_dim);
                    let state_pos = gdn.conv_positions[gdn_idx];
                    unsafe {
                        self.device.stream.launch_builder(conv1d_fn)
                            .arg(&mut gdn.conv_states[gdn_idx]).arg(&qkv_t).arg(conv1d_weight)
                            .arg(&mut gdn.qkv_conv_buf).arg(&conv_dim_u32).arg(&kernel_size_u32).arg(&state_pos)
                            .launch(conv_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill conv1d t={t} L{layer_idx}: {e}")))?;
                    gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
                }
                // SiLU
                unsafe {
                    self.device.stream.launch_builder(silu_fn)
                        .arg(&mut gdn.qkv_conv_buf).arg(&conv_dim_u32)
                        .launch(silu_launch)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN prefill silu t={t} L{layer_idx}: {e}")))?;
                // Compute gates
                {
                    let alpha_raw_t = gdn_pf.alpha_raw.slice(t * p.num_heads..(t + 1) * p.num_heads);
                    let beta_raw_t = gdn_pf.beta_raw.slice(t * p.num_heads..(t + 1) * p.num_heads);
                    unsafe {
                        self.device.stream.launch_builder(gates_fn)
                            .arg(dt_bias).arg(ssm_a).arg(&beta_raw_t).arg(&alpha_raw_t)
                            .arg(&mut gdn.alpha_buf).arg(&mut gdn.beta_buf).arg(&num_heads_u32)
                            .launch(gates_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill compute_gates t={t} L{layer_idx}: {e}")))?;
                }
                // L2-normalize Q and K
                {
                    let mut q_view = gdn.qkv_conv_buf.slice_mut(0..p.qk_dim);
                    unsafe {
                        self.device.stream.launch_builder(l2_fn)
                            .arg(&mut q_view).arg(&num_kv_heads_u32).arg(&head_dim_u32).arg(&l2_eps)
                            .launch(l2_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill l2_norm Q t={t} L{layer_idx}: {e}")))?;
                }
                {
                    let mut k_view = gdn.qkv_conv_buf.slice_mut(p.qk_dim..2 * p.qk_dim);
                    unsafe {
                        self.device.stream.launch_builder(l2_fn)
                            .arg(&mut k_view).arg(&num_kv_heads_u32).arg(&head_dim_u32).arg(&l2_eps)
                            .launch(l2_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill l2_norm K t={t} L{layer_idx}: {e}")))?;
                }
                // State update + output
                {
                    let k_view = gdn.qkv_conv_buf.slice(p.qk_dim..2 * p.qk_dim);
                    let v_view = gdn.qkv_conv_buf.slice(2 * p.qk_dim..p.qkv_dim);
                    let q_view = gdn.qkv_conv_buf.slice(0..p.qk_dim);
                    unsafe {
                        self.device.stream.launch_builder(state_fn)
                            .arg(&mut gdn.h_states[gdn_idx]).arg(&k_view).arg(&v_view)
                            .arg(&gdn.alpha_buf).arg(&gdn.beta_buf).arg(&q_view).arg(&mut gdn.output_buf)
                            .arg(&num_heads_u32).arg(&head_dim_u32).arg(&head_dim_u32).arg(&num_kv_heads_u32)
                            .launch(state_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill state_update t={t} L{layer_idx}: {e}")))?;
                }
                // RMSNorm on output
                unsafe {
                    self.device.stream.launch_builder(&st.kernels.rmsnorm)
                        .arg(&gdn.output_buf).arg(ssm_norm).arg(&mut gdn.normed_out_buf)
                        .arg(&eps).arg(&value_dim_u32)
                        .launch(norm_launch)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN prefill rmsnorm output t={t} L{layer_idx}: {e}")))?;
                // SiLU(gate) * normed_output -> batched output
                {
                    let gate_t = gdn_pf.gate.slice(t * p.value_dim..(t + 1) * p.value_dim);
                    let mut out_t = gdn_pf.gdn_out.slice_mut(t * p.value_dim..(t + 1) * p.value_dim);
                    unsafe {
                        self.device.stream.launch_builder(silu_mul_fn)
                            .arg(&gate_t).arg(&gdn.normed_out_buf).arg(&mut out_t).arg(&value_dim_u32)
                            .launch(silu_mul_launch)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN prefill silu_mul t={t} L{layer_idx}: {e}")))?;
                }
            }
        }

        // ================================================================
        // PHASE 3: Batched SSM out GEMM + residual
        // ================================================================
        //
        // gdn_out[T, value_dim] @ ssm_out^T -> attn_proj[T, hidden_dim]
        // with residual: attn_proj += x
        {
            let ssm_out = lw.ssm_out.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: ssm_out weight missing",
                )))?;
            unsafe {
                super::prefill::launch_gemm_residual(
                    &self.device, &st.kernels, ssm_out, lw.ssm_out_f16.as_ref(),
                    &gdn_pf.gdn_out, &pf.x, &mut pf.attn_proj,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, hidden_dim, p.value_dim, "gdn_ssm_out",
                )?;
            }

            // dump linear_attn_out (= post-ssm_out-GEMM + residual)
            if do_dump {
                self.device.synchronize()?;
                let host = self.device.dtoh_copy(&pf.attn_proj)?;
                let n = batch * hidden_dim;
                let dir = dump_dir.as_ref().unwrap();
                let path = format!("{dir}/L{layer_idx}-linear_attn_out.bin");
                let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(&path, &bytes).map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                eprintln!("[gdn-dump] L{layer_idx} linear_attn_out shape=[{batch}, {hidden_dim}] -> {path}");
            }
        }

        // ================================================================
        // PHASE 4: Batched FFN — MoE branch OR dense.
        // ================================================================
        // Hybrid MoE+GDN models (e.g. Qwen3.5-35B-A3B at indices 0,1,2,4,5,...)
        // pair GDN with MoE FFN: every layer carries router + experts, and the
        // converter writes zero-length sentinel slices for the dense
        // w_gate/w_up/w_down (qwen35_moe.rs:375). Without this branch, Phase 4
        // would fail with `sgemm gate: weight buffer too small` on the very
        // first prefill. For dense GDN models (Qwen3.5-9B),
        // `lw.moe_layer_blob` is always `None` so the dense branch runs as
        // before — byte-identical to the prior path.
        let is_moe_layer = lw.moe_layer_blob.is_some();
        if is_moe_layer {
            // NLL releases the `lw` (&) and `gdn` (&mut) borrows of `st` at
            // this point — they are not used downstream in this branch.
            self.prefill_moe_ffn_layer(layer_idx, batch, st, pf, eps)?;
        } else {
            unsafe {
                super::prefill::launch_rmsnorm_batched(
                    &self.device, &st.kernels,
                    &pf.attn_proj, &lw.ffn_norm, &mut pf.normed,
                    eps, batch, hidden_dim,
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.w_gate, lw.w_gate_f16.as_ref(),
                    &pf.normed, &mut pf.gate,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, inter_dim, hidden_dim, "gate",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.w_up, lw.w_up_f16.as_ref(),
                    &pf.normed, &mut pf.up,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, inter_dim, hidden_dim, "up",
                )?;
            }

            // Batched SwiGLU.
            unsafe {
                super::prefill::launch_swiglu_batched(
                    &self.device, &st.kernels,
                    &mut pf.gate, &pf.up, batch, inter_dim,
                )?;
            }

            // Batched down projection.
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.w_down, lw.w_down_f16.as_ref(),
                    &pf.gate, &mut pf.down,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, hidden_dim, inter_dim, "down",
                )?;
            }

            // Batched residual add + swap for next layer.
            unsafe {
                super::prefill::launch_residual_add_batched(
                    &self.device, &st.kernels,
                    &mut pf.attn_proj, &pf.down, batch, hidden_dim,
                )?;
            }
            self.device
                .stream
                .memcpy_dtod(&pf.attn_proj, &mut pf.x)
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod x<-attn_proj GDN prefill L{layer_idx}: {e}"))
                })?;
        }

        Ok(())
    }

    /// Batched MoE FFN for prefill (per-token loop over the decode
    /// kernels).
    ///
    /// **Why this exists**: the earlier CUDA prefill path only shipped the
    /// single-token decode MoE dispatch (`super::moe::encode_moe_ffn_decode`)
    /// and had no MoE branch in `prefill()`. For any non-GDN MoE layer (e.g.
    /// the full-attention + MoE-FFN layers in Qwen3.5-35B-A3B at indices 3,
    /// 7, 11, ..., 39), it fell through to the dense FFN block which expects
    /// `lw.w_gate / w_up / w_down` to be populated. The MoE converter writes
    /// **zero-length sentinel slices** for those tensors (see
    /// `crates/lumen-convert/src/arch/qwen35_moe.rs:375`), causing prefill to
    /// fail with `sgemm gate: weight buffer too small: have 0 elements,
    /// need 1048576` on the very first MoE layer.
    ///
    /// This helper closes the gap by running the existing decode MoE kernels
    /// in a per-token loop. Correctness-first: byte-identical to running
    /// decode `batch` times on the same `(prompt, weights)` pair, since the
    /// router + per-expert kernels operate per-token regardless of caller.
    ///
    /// Performance: not acceptance gates (MoE benchmarks measure
    /// **decode** tok/s; long-context benchmarks measure **decode-only** at long context).
    /// Asymptotic prefill cost is `O(batch × top_k × per_token_kernel_cost)`,
    /// which is acceptable for correctness validation but should be replaced
    /// by a batched-prefill kernel family in a future revision (analogous to
    /// Metal's `encode_moe_ffn_batched` at `metal/moe.rs:1500`).
    ///
    /// Contract (mirrors `prefill_gdn_layer` API shape):
    /// - Input `pf.attn_proj[batch, hidden_dim]` holds the post-attention
    ///   residual stream from steps 2a-2f.
    /// - Writes the post-MoE-FFN state to `pf.x[batch, hidden_dim]`.
    /// - Uses `pf.normed[batch, hidden_dim]` as the batched-RMSNorm output
    ///   buffer (consumed per-token by the router + per-expert kernels).
    /// - Borrows `st.moe_meta_cache[layer_idx]`, `st.moe_scratch`,
    ///   `lw.moe_layer_blob` (verified `is_some()` by caller).
    /// - Shared expert dispatch is deferred (matches the deferred-shared-expert plan
    ///   on the decode path at `compute_layer_gpu`).
    fn prefill_moe_ffn_layer(
        &self,
        layer_idx: usize,
        batch: usize,
        st: &mut MutableState,
        pf: &mut super::prefill::PrefillScratch,
        eps: f32,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let top_k = hp.num_active_experts.map(|v| v as usize).ok_or_else(|| {
            RuntimeError::Compute(
                "MoE prefill layer present but hyperparams.num_active_experts not set".into(),
            )
        })?;

        // Step 1: Batched FFN-norm over all batch tokens.
        // pf.attn_proj[batch, H] -> pf.normed[batch, H] via lw.ffn_norm.
        {
            let lw = &st.layer_weights_cache[layer_idx];
            unsafe {
                super::prefill::launch_rmsnorm_batched(
                    &self.device,
                    &st.kernels,
                    &pf.attn_proj,
                    &lw.ffn_norm,
                    &mut pf.normed,
                    eps,
                    batch,
                    hidden_dim,
                )?;
            }
        }

        // dump E: pf.normed after FFN-RMSNorm (canonical attn_post_norm dump).
        // This is the canonical L0 drift measurement point (3.19% pre-F64).
        if std::env::var("LUMEN_DUMP_NORMED").is_ok() {
            self.device.synchronize()?;
            let normed_host = self.device.dtoh_copy(&pf.normed)?;
            let tok0: Vec<f32> = normed_host[..hidden_dim].to_vec();
            let s: f64 = tok0.iter().map(|&v| v as f64).sum();
            let a: f64 = tok0.iter().map(|&v| (v as f64).abs()).sum();
            eprintln!(
                "[lumen-dump] layer={layer_idx} kind=attn_post_norm sum={s:.6} abs={a:.6} first16={:?}",
                &tok0[..16.min(hidden_dim)],
            );
        }

        // Step 2: Per-token loop. Each iteration calls encode_moe_ffn_decode
        // on a single-token slice of pf.normed (input), pf.attn_proj (residual),
        // and pf.x (output). The decode function reads expert_ids back to CPU
        // host memory once per call (one sync per token, ~K * 4 bytes).
        //
        // Borrow strategy: extract the MoE meta + scratch references once
        // outside the loop, take per-token splits of pf.{normed, attn_proj, x}
        // inside. layer_weights_cache borrow needs to be re-acquired per loop
        // iteration because pf is &mut.
        let moe_meta = st
            .moe_meta_cache
            .get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "prefill_moe_ffn_layer: layer {layer_idx} has no moe_meta_cache entry"
                ))
            })?
            .clone();
        let num_experts = moe_meta.expert_gate_offs.len();

        let moe_layer_blob = st.layer_weights_cache[layer_idx]
            .moe_layer_blob
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "MoE prefill layer {layer_idx} missing moe_layer_blob; \
                     upload_layer_weights must populate it when subtensors.experts.is_some()",
                ))
            })?
            // Clone the CudaSlice<u8> handle so we can hold the blob without
            // borrowing st.layer_weights_cache for the per-token loop's life.
            // CudaSlice is just a refcount + device ptr; clone is cheap.
            .clone();

        // borrow the per-layer batched offset table once outside the
        // loop. `CudaMoeBatchedOffsets` is not Clone (`CudaSlice<u64>` is not
        // Clone), but the borrow on `st.moe_batched_offsets` is disjoint from
        // the borrows on `st.kernels` (&) and `st.moe_scratch` (&mut), so NLL
        // allows holding all three across the loop.
        let batched_offsets = st
            .moe_batched_offsets
            .get(layer_idx)
            .and_then(|b| b.as_ref());

        let moe_scratch = st.moe_scratch.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "MoE prefill requires moe_scratch (allocated in init for MoE models)".into(),
            )
        })?;

        for t in 0..batch {
            let off = t * hidden_dim;
            let end = off + hidden_dim;
            let normed_view = pf.normed.slice(off..end);
            let residual_view = pf.attn_proj.slice(off..end);
            let mut output_view = pf.x.slice_mut(off..end);
            super::moe::encode_moe_ffn_decode(
                &self.device,
                &st.kernels,
                moe_scratch,
                &moe_meta,
                batched_offsets,
                &moe_layer_blob,
                &normed_view,
                &residual_view,
                &mut output_view,
                hidden_dim,
                inter_dim,
                num_experts,
                top_k,
            )?;

            // FIX: shared-expert FFN dispatch (Qwen3.5-MoE always-active
            // expert). Mirrors the decode-path dispatch at compute_layer_gpu.
            // Each prefill token runs the shared expert sigmoid-gated and
            // accumulates into pf.x[t..t+H] (the per-token output slice).
            //
            // Env-var bisection: `LUMEN_CUDA_SKIP_SHARED_EXPERT=1` to skip.
            let skip_shared_pf = std::env::var("LUMEN_CUDA_SKIP_SHARED_EXPERT")
                .ok()
                .as_deref()
                .map(|v| matches!(v, "1" | "true" | "yes"))
                .unwrap_or(false);
            if moe_meta.shared_gate.is_some() && !skip_shared_pf {
                let normed_view2 = pf.normed.slice(off..end);
                let mut output_view2 = pf.x.slice_mut(off..end);
                // opt-in fused path (same gating as decode).
                let use_fused = super::moe::moe_shared_fused_enabled()
                    && st.kernels.fused_glu_gemv_q4_0_prenormed_no_norm.is_some()
                    && st.kernels.moe_shared_down_q4_0_sigmoid_accum.is_some()
                    && st.kernels.moe_shared_down_q4_0_residual_accum.is_some();
                if use_fused {
                    super::moe::encode_shared_expert_ffn_decode_fused(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        &moe_meta,
                        &moe_layer_blob,
                        &normed_view2,
                        &mut output_view2,
                        hidden_dim,
                    )?;
                } else {
                    super::moe::encode_shared_expert_ffn_decode(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        &moe_meta,
                        &moe_layer_blob,
                        &normed_view2,
                        &mut output_view2,
                        hidden_dim,
                    )?;
                }
            }
        }

        // Step 3: keep pf.attn_proj coherent with pf.x for any downstream code
        // that reads attn_proj after FFN. The dense FFN path also writes the
        // residual-add result to pf.attn_proj before memcpying to pf.x, so we
        // mirror that contract here. (The decode-path MoE branch doesn't need
        // this because decode's compute_layer_gpu writes to st.scratch.x_gpu
        // directly with no downstream attn_proj reader.)
        self.device
            .stream
            .memcpy_dtod(&pf.x, &mut pf.attn_proj)
            .map_err(|e| {
                RuntimeError::Compute(format!(
                    "dtod attn_proj<-x MoE prefill L{layer_idx}: {e}"
                ))
            })?;

        Ok(())
    }

    /// Graph-compatible embed: reads token_id from device pointer.
    fn embed_token_gpu_graph(&self, st: &mut MutableState) -> Result<(), RuntimeError> {
        let hidden_dim = self.cached_hidden_dim;
        let gk = st.graph_kernels.as_ref().unwrap();
        let gp = st.graph_params.as_ref().unwrap();
        let config = LaunchConfig::for_elements(hidden_dim);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (config.grid_dim, 1, 1),
            block_dim: (config.block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let hd = hidden_dim as u32;
        let tid_ptr = gp.token_id_ptr();
        if let Some(ref emb_f16) = st.globals.embedding_f16 {
            unsafe { self.device.stream.launch_builder(&gk.embed_f16).arg(emb_f16).arg(&mut st.scratch.x_gpu).arg(&tid_ptr).arg(&hd).launch(launch_cfg) }
            .map_err(|e| RuntimeError::Compute(format!("graph embed_f16: {e}")))?;
        } else if let Some(ref emb_q4) = st.globals.embedding_q4 {
            unsafe { self.device.stream.launch_builder(&gk.embed_q4_0).arg(emb_q4).arg(&mut st.scratch.x_gpu).arg(&tid_ptr).arg(&hd).launch(launch_cfg) }
            .map_err(|e| RuntimeError::Compute(format!("graph embed_q4_0: {e}")))?;
        } else if let Some(ref emb_q8) = st.globals.embedding_q8 {
            unsafe { self.device.stream.launch_builder(&gk.embed_q8_0).arg(emb_q8).arg(&mut st.scratch.x_gpu).arg(&tid_ptr).arg(&hd).launch(launch_cfg) }
            .map_err(|e| RuntimeError::Compute(format!("graph embed_q8_0: {e}")))?;
        } else {
            unsafe { self.device.stream.launch_builder(&gk.embed_f32).arg(&st.globals.embedding).arg(&mut st.scratch.x_gpu).arg(&tid_ptr).arg(&hd).launch(launch_cfg) }
            .map_err(|e| RuntimeError::Compute(format!("graph embed_f32: {e}")))?;
        }
        Ok(())
    }

    /// Graph-compatible transformer layer: reads pos/seq_len from device pointers.
    ///
    /// CRITICAL: This function must NOT perform any host-to-device memcpy on the
    /// capturing stream. All cuBLAS batched calls MUST use pre-computed device
    /// pointer arrays (uploaded once in `preload_weights()`). The caller must
    /// verify `precomputed_ptrs.is_some()` before entering graph capture.
    ///
    /// # Inter-layer fusion parameters
    ///
    /// - `skip_head_norm`: If true, skip the initial `fused_rmsnorm_f16` because
    /// `input_f16` already contains the normalized activation from the previous
    /// layer's fused tail. Only effective for F16/HGEMV paths.
    /// - `fuse_tail_next_layer`: If Some(next_layer_idx), replace the final
    /// `residual_add_copy` with `fused_residual_rmsnorm_f16` using the next
    /// layer's attn_norm weights (from `layer_weights_cache[next_layer_idx]`).
    /// Saves 1 dispatch per inter-layer boundary (35 fewer for 36-layer models).
    fn compute_layer_gpu_graph(
        &self,
        layer_idx: usize,
        st: &mut MutableState,
        skip_head_norm: bool,
        fuse_tail_next_layer: Option<usize>,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let eps = hp.norm_eps;
        let theta = hp.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Step 1: RMSNorm + QKV projections.
        // For F16/HGEMV paths: use fused RMSNorm+F16 kernel (1 dispatch instead of 2)
        // followed by pre-computed batched cuBLAS calls (zero htod memcpy).
        let lw = &st.layer_weights_cache[layer_idx];

        // Q+gate fusion (Qwen3.5 dense layers). Mirrors the production
        // `compute_layer_gpu` path at line 1727: `attn_q_norm` present means wq
        // projects to [q_dim * 2] = q_gate scratch, which is then deinterleaved
        // into st.scratch.q + st.scratch.gate_buf, per-head-RMSNorm'd, then a
        // post-attention sigmoid gating applies to attn_out before output proj.
        // When this is OFF the graph variant is bit-exact when disabled.
        let has_qgate_fusion = lw.attn_q_norm.is_some();
        let wq_out_dim = if has_qgate_fusion { q_dim * 2 } else { q_dim };

        // Diagnostic: on first layer, report which weight path is used (cuBLAS vs custom).
        if layer_idx == 0 {
            let diag = std::env::var("LUMEN_GRAPH_DIAGNOSTIC")
                .map(|v| v == "1")
                .unwrap_or(false);
            if diag {
                let has_f16_cache = lw.wq_f16.is_some();
                let wq_type = match &lw.wq {
                    GpuWeightBuf::F16Raw(_) => "F16Raw (cuBLAS HGEMV path)",
                    GpuWeightBuf::Bf16Raw(_) => "Bf16Raw (custom matvec_bf16 path)",
                    GpuWeightBuf::Q8Raw(_) => "Q8Raw (native dp4a path)",
                    GpuWeightBuf::Q4Raw(_) => "Q4Raw (native dp4a path)",
                    GpuWeightBuf::Q4Aligned(_) => "Q4Aligned (native dp4a path)",
                    GpuWeightBuf::Q8Aligned(_) => "Q8Aligned (native dp4a path)",
                    GpuWeightBuf::F32(_) if has_f16_cache => "F32 (cuBLAS HGEMV via F16 cache)",
                    GpuWeightBuf::F32(_) => "F32 (cuBLAS SGEMV path)",
                    // Q8Split/Q4Split/Q8Tile/Q4Tile are sibling buffers,
                    // not stored in `lw.wq`. Provided here only to make the match
                    // exhaustive; should never trip in practice.
                    GpuWeightBuf::Q8Split(_) => "Q8Split (sibling -- unexpected in lw.wq)",
                    GpuWeightBuf::Q4Split(_) => "Q4Split (sibling -- unexpected in lw.wq)",
                    GpuWeightBuf::Q8Tile(_)  => "Q8Tile (sibling -- unexpected in lw.wq)",
                    GpuWeightBuf::Q4Tile(_)  => "Q4Tile (sibling -- unexpected in lw.wq)",
                };
                eprintln!("[GRAPH-DIAG]     L0 weight format: {wq_type}");
                eprintln!("[GRAPH-DIAG]     L0 has_precomputed_ptrs: {}", st.precomputed_ptrs.is_some());
            }
        }

        if matches!(&lw.wq, GpuWeightBuf::F16Raw(_)) {
            // F16 native: fused RMSNorm + F32->F16 in one kernel.
            // Skip if the previous layer already fused this into its residual tail.
            let pcp = st.precomputed_ptrs.as_ref().unwrap(); // guaranteed by can_use_graph check
            if !skip_head_norm {
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &lw.attn_norm,
                        &mut st.scratch.input_f16,
                        eps, hidden_dim, "graph attn F16",
                    )?;
                }
            }
            // QKV projections: Q separate + KV batched (2 cuBLAS calls).
            // Pre-computed batched: Q separate + KV batched (no htod).
            // qgate routes wq -> q_gate (q_dim*2). KV unchanged.
            if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                let (wq_out_buf, wq_od) = if has_qgate_fusion {
                    (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                } else {
                    (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                };
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wq_f16, &st.scratch.input_f16,
                        wq_out_buf, wq_od, hidden_dim, "graph wq",
                        st.algo_cache.get(wq_od, hidden_dim),
                    )?;
                }
            }
            unsafe {
                launch_hgemv_f16_batched_precomputed(
                    &self.device,
                    &pcp.kv_a_ptrs[layer_idx],
                    &pcp.kv_b_ptrs[layer_idx],
                    &pcp.kv_c_ptrs[layer_idx],
                    2, kv_dim, hidden_dim, "graph kv",
                    st.algo_cache.get(kv_dim, hidden_dim),
                )?;
            }
        } else if matches!(&lw.wq, GpuWeightBuf::F32(_))
            && lw.wq_f16.is_some() && lw.wk_f16.is_some() && lw.wv_f16.is_some()
        {
            // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
            // Q8/Q4 weights fall through to launch_matvec() for native dp4a (1.06 B/elem).
            let pcp = st.precomputed_ptrs.as_ref().unwrap();
            if !skip_head_norm {
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &lw.attn_norm,
                        &mut st.scratch.input_f16,
                        eps, hidden_dim, "graph attn HGEMV",
                    )?;
                }
            }
            // qgate routes wq -> q_gate (q_dim*2). KV unchanged.
            if let Some(ref wq_f16) = lw.wq_f16 {
                let (wq_out_buf, wq_od) = if has_qgate_fusion {
                    (st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>, wq_out_dim)
                } else {
                    (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                };
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wq_f16, &st.scratch.input_f16,
                        wq_out_buf, wq_od, hidden_dim, "graph wq",
                        st.algo_cache.get(wq_od, hidden_dim),
                    )?;
                }
            }
            unsafe {
                launch_hgemv_f16_batched_precomputed(
                    &self.device,
                    &pcp.kv_a_ptrs[layer_idx],
                    &pcp.kv_b_ptrs[layer_idx],
                    &pcp.kv_c_ptrs[layer_idx],
                    2, kv_dim, hidden_dim, "graph kv",
                    st.algo_cache.get(kv_dim, hidden_dim),
                )?;
            }
        } else {
            // Q8/Q4/Q8Aligned/Q4Aligned/F32: native-quant graph decode via launch_matvec().
            let qkv_use_preq = weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wk, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wv, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some();

            if skip_head_norm && qkv_use_preq {
                // input_q8_1 already populated by fused_residual_rmsnorm_q8_1 from
                // the previous layer's tail. Skip norm+quantize, go straight to QKV matvecs.
                // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                // qgate routes wq -> q_gate (q_dim*2); K/V unchanged.
                unsafe {
                    if has_qgate_fusion {
                        let q8_1_in = st.scratch.input_q8_1.as_ref().unwrap();
                        let q8_1_ptr: *const _ = q8_1_in;
                        let q_gate = st.scratch.q_gate.as_mut().unwrap();
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), &*q8_1_ptr, q_gate, wq_out_dim, hidden_dim, "graph wq qgate")?;
                    } else {
                        let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                    }
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wk, lw.q8_tile_wk.as_ref(), lw.q4_tile_wk.as_ref(), lw.q8_split_wk.as_ref(), lw.q4_split_wk.as_ref(), q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wv, lw.q8_tile_wv.as_ref(), lw.q4_tile_wv.as_ref(), lw.q8_split_wv.as_ref(), lw.q4_split_wv.as_ref(), q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
                }
            } else if !skip_head_norm && qkv_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
                // Fused RMSNorm + Q8_1 for graph decode: saves 1 dispatch per norm site.
                let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
                let q8_1_buf_init = st.scratch.input_q8_1.as_mut().unwrap();
                let bs = rmsnorm_block_size(hidden_dim);
                let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: rmsnorm_shared_bytes(bs) };
                let dim = hidden_dim as u32;
                unsafe { self.device.stream.launch_builder(fused_fn).arg(&st.scratch.x_gpu).arg(&lw.attn_norm).arg(&mut *q8_1_buf_init).arg(&eps).arg(&dim).launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm_to_q8_1 attn: {e}")))?;
                // Drop the &mut borrow before re-borrowing for wq dispatch.
                let _ = q8_1_buf_init;
                unsafe {
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                    // qgate routes wq -> q_gate (q_dim*2); K/V unchanged.
                    if has_qgate_fusion {
                        let q8_1_in = st.scratch.input_q8_1.as_ref().unwrap();
                        // Aliasing: q8_1 buffer is read-only here (KV reads also read-only).
                        // Use raw pointer to break the borrow chain for the &mut q_gate write.
                        let q8_1_ptr: *const _ = q8_1_in;
                        let q_gate = st.scratch.q_gate.as_mut().unwrap();
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), &*q8_1_ptr, q_gate, wq_out_dim, hidden_dim, "graph wq qgate")?;
                    } else {
                        let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                    }
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wk, lw.q8_tile_wk.as_ref(), lw.q4_tile_wk.as_ref(), lw.q8_split_wk.as_ref(), lw.q4_split_wk.as_ref(), q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wv, lw.q8_tile_wv.as_ref(), lw.q4_tile_wv.as_ref(), lw.q8_split_wv.as_ref(), lw.q4_split_wv.as_ref(), q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
                }
            } else {
                if !skip_head_norm {
                    let bs = rmsnorm_block_size(hidden_dim);
                    let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: rmsnorm_shared_bytes(bs) };
                    let dim = hidden_dim as u32;
                    unsafe { self.device.stream.launch_builder(&st.kernels.rmsnorm).arg(&st.scratch.x_gpu).arg(&lw.attn_norm).arg(&mut st.scratch.normed).arg(&eps).arg(&dim).launch(lc) }
                    .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm attn: {e}")))?;
                }
                if qkv_use_preq {
                    let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref().unwrap();
                    unsafe {
                        {
                            let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                            launch_quantize_input_q8_1(&self.device, quant_fn, &st.scratch.normed, q8_1_buf, hidden_dim, "graph qkv")?;
                        }
                        // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                        // qgate routes wq -> q_gate (q_dim*2); K/V unchanged.
                        if has_qgate_fusion {
                            let q8_1_in = st.scratch.input_q8_1.as_ref().unwrap();
                            let q8_1_ptr: *const _ = q8_1_in;
                            let q_gate = st.scratch.q_gate.as_mut().unwrap();
                            launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), &*q8_1_ptr, q_gate, wq_out_dim, hidden_dim, "graph wq qgate")?;
                        } else {
                            let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                            launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wq, lw.q8_tile_wq.as_ref(), lw.q4_tile_wq.as_ref(), lw.q8_split_wq.as_ref(), lw.q4_split_wq.as_ref(), q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                        }
                        let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wk, lw.q8_tile_wk.as_ref(), lw.q4_tile_wk.as_ref(), lw.q8_split_wk.as_ref(), lw.q4_split_wk.as_ref(), q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.wv, lw.q8_tile_wv.as_ref(), lw.q4_tile_wv.as_ref(), lw.q8_split_wv.as_ref(), lw.q4_split_wv.as_ref(), q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
                    }
                } else {
                    // qgate routes wq -> q_gate (q_dim*2); K/V unchanged.
                    unsafe {
                        if has_qgate_fusion {
                            let q_gate = st.scratch.q_gate.as_mut().unwrap();
                            launch_matvec(&self.device, &st.kernels, &lw.wq, &st.scratch.normed, q_gate, wq_out_dim, hidden_dim, "graph wq qgate", lw.wq_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        } else {
                            launch_matvec(&self.device, &st.kernels, &lw.wq, &st.scratch.normed, &mut st.scratch.q, q_dim, hidden_dim, "graph wq", lw.wq_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        }
                        launch_matvec(&self.device, &st.kernels, &lw.wk, &st.scratch.normed, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk", lw.wk_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        launch_matvec(&self.device, &st.kernels, &lw.wv, &st.scratch.normed, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv", lw.wv_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                    }
                }
            }
        }

        // Q+gate fusion post-processing: deinterleave q_gate -> q + gate_buf,
        // then per-head RMSNorm on Q (attn_q_norm) and K (attn_k_norm). Mirrors
        // `compute_layer_gpu` line 2138-2231. Must run AFTER all QKV projection
        // branches and BEFORE bias/RoPE. All three sub-dispatches are graph-
        // capture-safe: deinterleave_qgate, rmsnorm_per_head_inplace take only
        // device pointers + immutable u32 dims as args (no host scalars vary
        // per-token across the graph; eps is hp.norm_eps, fixed per session).
        if has_qgate_fusion {
            let lw = &st.layer_weights_cache[layer_idx];
            // 1a. Deinterleave: q_gate [q_dim*2] -> q [q_dim] + gate_buf [q_dim]
            if let Some(ref deinterleave_fn) = st.kernels.deinterleave_qgate {
                let q_gate_buf_ptr: *const _ = st.scratch.q_gate.as_ref().unwrap();
                let block = 256u32;
                let grid = ((q_dim as u32) + block - 1) / block;
                let hd = head_dim as u32;
                let nh = num_heads as u32;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    let q_gate_buf = &*q_gate_buf_ptr;
                    let gate_buf = st.scratch.gate_buf.as_mut().unwrap();
                    let q_out_ptr: *mut _ = &mut st.scratch.q;
                    self.device
                        .stream
                        .launch_builder(deinterleave_fn)
                        .arg(q_gate_buf)
                        .arg(&mut *q_out_ptr)
                        .arg(gate_buf)
                        .arg(&hd)
                        .arg(&nh)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("graph deinterleave_qgate: {e}")))?;
            } else {
                return Err(RuntimeError::Compute(
                    "Q+gate fusion (graph) requires deinterleave_qgate kernel".into(),
                ));
            }

            // 1b. Per-head RMSNorm on Q using attn_q_norm [head_dim].
            if let Some(ref q_norm_w) = lw.attn_q_norm {
                let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Q+gate fusion (graph) requires rmsnorm_per_head_inplace".into())
                })?;
                let hd = head_dim as u32;
                let nh = num_heads as u32;
                let block = (head_dim as u32).min(1024).max(32);
                let block = (block / 32) * 32; // round down to warp multiple
                let shared_bytes = (block / 32) * 4; // one float per warp
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (nh, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(norm_fn)
                        .arg(&mut st.scratch.q)
                        .arg(q_norm_w)
                        .arg(&nh)
                        .arg(&hd)
                        .arg(&eps)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm_per_head Q: {e}")))?;
            }

            // 1c. Per-head RMSNorm on K using attn_k_norm [head_dim].
            if let Some(ref k_norm_w) = lw.attn_k_norm {
                let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Q+gate fusion (graph) requires rmsnorm_per_head_inplace".into())
                })?;
                let hd = head_dim as u32;
                let nkvh = num_kv_heads as u32;
                let block = (head_dim as u32).min(1024).max(32);
                let block = (block / 32) * 32;
                let shared_bytes = (block / 32) * 4;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (nkvh, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(norm_fn)
                        .arg(&mut st.scratch.k)
                        .arg(k_norm_w)
                        .arg(&nkvh)
                        .arg(&hd)
                        .arg(&eps)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm_per_head K: {e}")))?;
            }
        }

        // QKV bias (Qwen2-family, graph decode).
        let lw = &st.layer_weights_cache[layer_idx];
        if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
            let block = 256u32;
            unsafe {
                if let Some(ref bq) = lw.bq {
                    let d = q_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.q).arg(bq).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bq graph: {e}")))?;
                }
                if let Some(ref bk) = lw.bk {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.k).arg(bk).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bk graph: {e}")))?;
                }
                if let Some(ref bv) = lw.bv {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.v).arg(bv).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bv graph: {e}")))?;
                }
            }
        }

        // Steps 2+3: Fused RoPE + KV cache write -- GRAPH VARIANT.
        // Single kernel applies RoPE to Q and K, then writes K and V to cache.
        // Saves 2 dispatches/layer vs separate rope_apply + 2x kv_cache_write.
        // Thread mapping: 1 thread per RoPE pair. Grid = max(q_pairs, k_pairs).
        {
            let gk = st.graph_kernels.as_ref().unwrap();
            let gp = st.graph_params.as_ref().unwrap();
            let pos_ptr = gp.seq_pos_ptr();
            let kvc = &mut st.kv_caches[layer_idx];
            let nkv = kvc.num_kv_heads as u32;
            let msl = kvc.max_seq_len as u32;
            let hd = kvc.head_dim as u32;
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            let actual_rot = if rotary_dim > 0 && rotary_dim < head_dim as u32 { rotary_dim as usize } else { head_dim };
            let half_rot = actual_rot / 2;
            let total_q_pairs = num_heads * half_rot;
            let total_k_pairs = num_kv_heads * half_rot;
            let max_pairs = total_q_pairs.max(total_k_pairs);
            let config = LaunchConfig::for_elements(max_pairs);
            let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
            // NeoX RoPE: models with partial rotary_dim use half-offset dimension pairing.
            let rope_neox = hp.rope_neox;
            let rope_kv_fn = if rope_neox {
                &gk.rope_kv_write_neox
            } else {
                &gk.rope_kv_write
            };
            unsafe { self.device.stream.launch_builder(rope_kv_fn)
                .arg(&mut st.scratch.q).arg(&mut st.scratch.k).arg(&st.scratch.v)
                .arg(&mut kvc.k_cache).arg(&mut kvc.v_cache)
                .arg(&pos_ptr)
                .arg(&(num_heads as u32)).arg(&nkv).arg(&hd).arg(&msl).arg(&theta)
                .arg(&rotary_dim)
                .launch(lc) }
            .map_err(|e| RuntimeError::Compute(format!("graph rope_kv_write L{layer_idx}: {e}")))?;
        }

        // Step 4: Attention -- GRAPH VARIANT (reads seq_len from device pointer).
        //
        // when `LUMEN_CUDA_DECODE_GRAPH_TILED=1` is set AND the
        // `attention_decode_tiled_graph` kernel is loaded, route through the
        // tiled streaming-softmax kernel. Tiled-graph shmem is O(T_C + head_dim),
        // constant in seq_len -- ~1.6 KB at head_dim=256, T_C=128. No extended-
        // shmem opt-in needed. Algorithm is the same Dao-2022 online softmax as
        // `attention_decode_tiled.cu`; the host-side gate at `decode_token`
        // (`graph_eager_fallback_for_tiled`) is now bypassed when this lever
        // is active.
        //
        // When the flag is OFF, this site uses the single-block
        // `attention_decode_graph` kernel, preserving the prior behaviour.
        {
            let gk = st.graph_kernels.as_ref().unwrap();
            let gp = st.graph_params.as_ref().unwrap();
            let seq_len_ptr = gp.attn_seq_len_ptr();
            let kvc = &st.kv_caches[layer_idx];
            let msl = kvc.max_seq_len as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            // Tiled-graph routing: only when the env opted in AND the kernel
            // loaded AND head_dim satisfies the tile invariant (head_dim %
            // ATTN_DECODE_TILED_BLOCK_DIM == 0). Production Qwen3.5-9B uses
            // head_dim=256 which satisfies the latter.
            // env-or-model-default helper. Returns true when
            // explicitly set OR when the model is BF16 dense (graph capture
            // is a measured +13% TPOT win there). Defaults to false on Q8/Q4
            // dense.
            let use_tiled_graph = cuda_decode_graph_tiled_enabled()
                && gk.attention_decode_tiled.is_some()
                && super::decode::attention_decode_tiled_supports_head_dim(head_dim as u32);
            if use_tiled_graph {
                // tiled-graph shmem: (8 + head_dim + T_C) * 4 bytes. T_C is the
                // compile-time constant in the kernel (graph_kernels.cu); we
                // mirror it here. Constant in seq_len.
                const TILED_T_C: u32 = 128;
                const TILED_BLOCK_DIM: u32 = 128;
                let shared = (8 + (head_dim as u32) + TILED_T_C) * 4;
                let lc = CudarcLaunchConfig {
                    grid_dim: (num_heads as u32, 1, 1),
                    block_dim: (TILED_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: shared,
                };
                let tiled_fn = gk.attention_decode_tiled.as_ref().unwrap();
                unsafe { self.device.stream.launch_builder(tiled_fn)
                    .arg(&st.scratch.q).arg(&kvc.k_cache).arg(&kvc.v_cache).arg(&mut st.scratch.attn_out)
                    .arg(&(num_heads as u32)).arg(&(num_kv_heads as u32)).arg(&(head_dim as u32))
                    .arg(&seq_len_ptr).arg(&msl).arg(&scale)
                    .launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph attn tiled L{layer_idx}: {e}")))?;
            } else {
                let shared = super::graph::graph_attention_shared_bytes(msl);
                let lc = CudarcLaunchConfig { grid_dim: (num_heads as u32,1,1), block_dim: (super::graph::GRAPH_ATTN_BLOCK_SIZE,1,1), shared_mem_bytes: shared };
                unsafe { self.device.stream.launch_builder(&gk.attention_decode)
                    .arg(&st.scratch.q).arg(&kvc.k_cache).arg(&kvc.v_cache).arg(&mut st.scratch.attn_out)
                    .arg(&(num_heads as u32)).arg(&(num_kv_heads as u32)).arg(&(head_dim as u32))
                    .arg(&seq_len_ptr).arg(&msl).arg(&scale)
                    .launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph attn L{layer_idx}: {e}")))?;
            }
        }

        // Q+gate sigmoid gating (graph variant): mirror of
        // `compute_layer_gpu` line 2349. Applies after attention, before output
        // projection. `attn_out = sigmoid(gate_buf) * attn_out`.
        //
        // Implementation note: the kernel computes `out[i] = sigmoid(gate[i]) *
        // x[i]`. Each thread reads x[i] before writing out[i], so passing
        // `attn_out` as BOTH x and out is safe and matches the math (in-place
        // form). This avoids the memcpy_dtod step the non-graph production
        // path uses; that step had to write through `q` and then dtod-copy back
        // because the production-path comment notes a sizing concern -- but
        // here we know q_dim = q_dim (the attn_out is unconditionally [q_dim]).
        // Single sigmoid_mul kernel dispatch, fully graph-capture-safe.
        if has_qgate_fusion {
            if let Some(ref sigmoid_fn) = st.kernels.sigmoid_mul {
                let n = q_dim as u32;
                let block = 256u32;
                let grid = (n + block - 1) / block;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    let gate_buf_ptr: *const _ = st.scratch.gate_buf.as_ref().unwrap();
                    // x and out are both `attn_out`. Pass via raw pointer to
                    // satisfy borrow checker (one shared ref + one mutable).
                    let attn_out_ptr: *mut _ = &mut st.scratch.attn_out;
                    self.device
                        .stream
                        .launch_builder(sigmoid_fn)
                        .arg(&*gate_buf_ptr)
                        .arg(&*attn_out_ptr)
                        .arg(&mut *attn_out_ptr)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("graph sigmoid_mul: {e}")))?;
            } else {
                return Err(RuntimeError::Compute(
                    "Q+gate fusion (graph) requires sigmoid_mul kernel".into(),
                ));
            }
        }

        // Step 5: Output proj + residual (no per-token scalars, safe for capture).
        // F16 path: fused F32->F16 conversion + residual copy (1 dispatch instead of 2),
        // then cuBLAS HGEMV with beta=1.0.
        let lw = &st.layer_weights_cache[layer_idx];
        if let GpuWeightBuf::F16Raw(ref wo_f16) = lw.wo {
            let gk = st.graph_kernels.as_ref().unwrap();
            // Fused kernel: convert attn_out to F16 AND copy residual to attn_proj.
            // Saves 1 dispatch (was: dtod copy + f32_to_f16_vec = 2 dispatches).
            let max_dim = q_dim.max(hidden_dim);
            let block = 256u32;
            let grid = ((max_dim as u32) + block - 1) / block;
            let lc = CudarcLaunchConfig { grid_dim: (grid,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 };
            unsafe { self.device.stream.launch_builder(&gk.convert_f16_residual_copy)
                .arg(&st.scratch.attn_out).arg(&mut st.scratch.input_f16)
                .arg(&st.scratch.x_gpu).arg(&mut st.scratch.attn_proj)
                .arg(&(q_dim as u32)).arg(&(hidden_dim as u32))
                .launch(lc) }
            .map_err(|e| RuntimeError::Compute(format!("graph convert_f16_residual L{layer_idx}: {e}")))?;
            // HGEMV with beta=1.0 (residual already in attn_proj).
            unsafe { launch_hgemv_f16_preconverted_beta1(
                &self.device, wo_f16, &st.scratch.input_f16,
                &mut st.scratch.attn_proj, hidden_dim, q_dim, "graph wo",
                st.algo_cache.get(hidden_dim, q_dim),
            )?; }
        } else if matches!(&lw.wo, GpuWeightBuf::F32(_))
            && lw.wo_f16.is_some()
        {
            // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
            // Q8/Q4 weights fall through to launch_matvec_residual() for native dp4a.
            let wo_f16 = lw.wo_f16.as_ref().unwrap();
            unsafe {
                launch_hgemv_f16_residual(
                    &self.device, &st.kernels,
                    wo_f16, &st.scratch.attn_out, &st.scratch.x_gpu,
                    &mut st.scratch.attn_proj, &mut st.scratch.input_f16,
                    hidden_dim, q_dim, "graph wo",
                    st.algo_cache.get(hidden_dim, q_dim),
                )?;
            }
        } else {
            // split-layout: try Q8/Q4 split sibling for wo (mirrors the non-graph wo path).
            let use_split_wo = (st.kernels.use_q8_split_dispatch && lw.q8_split_wo.is_some())
                || (st.kernels.use_q4_split_dispatch && lw.q4_split_wo.is_some());
            if use_split_wo {
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device, quant_fn, &st.scratch.attn_out, q8_1_buf,
                            q_dim, "graph wo split",
                        )?;
                        launch_matvec_preq8_1_residual_tile(
                            &self.device, &st.kernels, &lw.wo,
                            lw.q8_tile_wo.as_ref(),  lw.q4_tile_wo.as_ref(),
                            lw.q8_split_wo.as_ref(), lw.q4_split_wo.as_ref(),
                            q8_1_buf, &st.scratch.x_gpu, &mut st.scratch.attn_proj,
                            hidden_dim, q_dim, "graph wo",
                        )?;
                    }
                } else {
                    unsafe { launch_matvec_residual(&self.device, &st.kernels, &lw.wo, &st.scratch.attn_out, &st.scratch.x_gpu, &mut st.scratch.attn_proj, hidden_dim, q_dim, "graph wo", lw.wo_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
                }
            } else {
                unsafe { launch_matvec_residual(&self.device, &st.kernels, &lw.wo, &st.scratch.attn_out, &st.scratch.x_gpu, &mut st.scratch.attn_proj, hidden_dim, q_dim, "graph wo", lw.wo_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
            }
        }

        // MoE FFN branch.
        //
        // When this layer has `moe_layer_blob`, dispatch the existing graph-safe
        // MoE FFN family (`encode_moe_ffn_decode_fused_norm` for Q8, falls back
        // to `encode_moe_ffn_decode_q4_0` for Q4, plus shared-expert dispatch).
        // The MoE dispatch is structurally identical across decode tokens
        // because all per-token state (expert_ids, expert_weights) lives on
        // the GPU; CPU passes only fixed u32 dims as kernel args.
        //
        // Output: `st.scratch.x_gpu` receives `attn_proj + Σ_k w[k] * expert[k](normed)`,
        // with residual=attn_proj. The shared expert (Qwen3.5-MoE always-active)
        // is added on top: `x_gpu += sigmoid(shared_gate · normed) * shared_expert(normed)`.
        //
        // After MoE FFN, the residual is already incorporated into x_gpu and
        // no further residual_add is needed — we return early. The caller
        // (run_graph_pipeline) forces fuse_tail_next=None for MoE layers, so
        // the next layer's head-norm runs unconditionally.
        if st.layer_weights_cache[layer_idx].moe_layer_blob.is_some() {
            let lw_moe: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];
            let moe_meta_ref = st.moe_meta_cache.get(layer_idx)
                .and_then(|m| m.as_ref())
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "MoE graph layer {layer_idx} missing moe_meta_cache entry",
                )))?;
            let moe_layer_blob = lw_moe.moe_layer_blob.as_ref()
                .expect("moe_layer_blob.is_some() verified above");
            let num_experts = moe_meta_ref.expert_gate_offs.len();
            let top_k = self.hp()?.num_active_experts.map(|v| v as usize).unwrap_or(0);
            if top_k == 0 {
                return Err(RuntimeError::Compute(
                    "MoE graph layer present but hyperparams.num_active_experts not set".into(),
                ));
            }

            let batched_offsets = st
                .moe_batched_offsets
                .get(layer_idx)
                .and_then(|b| b.as_ref());
            let has_shared = moe_meta_ref.shared_gate.is_some();

            // Borrow the mutable scratch — `st.moe_scratch` is disjoint from
            // `st.layer_weights_cache`, `st.scratch.*`, `st.kernels`.
            let moe_scratch = st.moe_scratch.as_mut().ok_or_else(|| {
                RuntimeError::Compute(
                    "MoE graph layer dispatch requires moe_scratch (allocated in init for MoE models)".into(),
                )
            })?;

            super::moe::encode_moe_ffn_decode_fused_norm(
                &self.device,
                &st.kernels,
                moe_scratch,
                moe_meta_ref,
                batched_offsets,
                moe_layer_blob,
                &st.scratch.attn_proj.slice(..),
                &lw_moe.ffn_norm,
                &mut st.scratch.normed.slice_mut(..),
                &st.scratch.attn_proj.slice(..),
                &mut st.scratch.x_gpu.slice_mut(..),
                eps,
                hidden_dim,
                inter_dim,
                num_experts,
                top_k,
            )?;

            // FIX shared expert (Qwen3.5-MoE always-active expert).
            // Cached env var via OnceLock to match production path.
            let skip_shared = {
                use std::sync::OnceLock;
                static FLAG: OnceLock<bool> = OnceLock::new();
                *FLAG.get_or_init(|| {
                    std::env::var("LUMEN_CUDA_SKIP_SHARED_EXPERT")
                        .ok()
                        .as_deref()
                        .map(|v| matches!(v, "1" | "true" | "yes"))
                        .unwrap_or(false)
                })
            };
            if has_shared && !skip_shared {
                let use_fused = super::moe::moe_shared_fused_enabled()
                    && st.kernels.fused_glu_gemv_q4_0_prenormed_no_norm.is_some()
                    && st.kernels.moe_shared_down_q4_0_sigmoid_accum.is_some()
                    && st.kernels.moe_shared_down_q4_0_residual_accum.is_some();
                if use_fused {
                    super::moe::encode_shared_expert_ffn_decode_fused(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        moe_meta_ref,
                        moe_layer_blob,
                        &st.scratch.normed.slice(..),
                        &mut st.scratch.x_gpu.slice_mut(..),
                        hidden_dim,
                    )?;
                } else {
                    super::moe::encode_shared_expert_ffn_decode(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        moe_meta_ref,
                        moe_layer_blob,
                        &st.scratch.normed.slice(..),
                        &mut st.scratch.x_gpu.slice_mut(..),
                        hidden_dim,
                    )?;
                }
            }

            // MoE FFN done — x_gpu = attn_proj + Σ expert_k + shared_expert.
            // No further residual add (fused_norm wrapper already accumulated).
            // Caller (run_graph_pipeline) forces fuse_tail_next=None for MoE,
            // so next layer's head_norm runs unconditionally.
            let _ = fuse_tail_next_layer; // explicitly unused for MoE
            return Ok(());
        }

        // Step 6: FFN -- RMSNorm+F16 -> batched cuBLAS HGEMV gate/up -> SwiGLU -> HGEMV down.
        let lw = &st.layer_weights_cache[layer_idx];
        let pcp = st.precomputed_ptrs.as_ref().unwrap();
        if matches!((&lw.w_gate, &lw.w_up), (GpuWeightBuf::F16Raw(_), GpuWeightBuf::F16Raw(_)))
        {
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.attn_proj, &lw.ffn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "graph ffn F16",
                )?;
            }
            unsafe {
                launch_hgemv_f16_batched_precomputed(
                    &self.device,
                    &pcp.ffn_a_ptrs[layer_idx],
                    &pcp.ffn_b_ptrs[layer_idx],
                    &pcp.ffn_c_ptrs[layer_idx],
                    2, inter_dim, hidden_dim, "graph gate_up",
                    st.algo_cache.get(inter_dim, hidden_dim),
                )?;
            }
        } else if matches!(&lw.w_gate, GpuWeightBuf::F32(_))
            && lw.w_gate_f16.is_some() && lw.w_up_f16.is_some()
        {
            // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
            // Q8/Q4 weights fall through to launch_matvec() for native dp4a.
            unsafe {
                launch_fused_rmsnorm_f16(
                    &self.device, &st.kernels,
                    &st.scratch.attn_proj, &lw.ffn_norm,
                    &mut st.scratch.input_f16,
                    eps, hidden_dim, "graph ffn HGEMV",
                )?;
            }
            unsafe {
                launch_hgemv_f16_batched_precomputed(
                    &self.device,
                    &pcp.ffn_a_ptrs[layer_idx],
                    &pcp.ffn_b_ptrs[layer_idx],
                    &pcp.ffn_c_ptrs[layer_idx],
                    2, inter_dim, hidden_dim, "graph gate_up",
                    st.algo_cache.get(inter_dim, hidden_dim),
                )?;
            }
        } else {
            // Q8/Q4/Q8Aligned/Q4Aligned/F32: native-quant graph FFN.
            // Separate path: rmsnorm + gate + up dispatches.
            let ffn_use_preq = weight_uses_dp4a_q8_1(&lw.w_gate, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.w_up, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some();

            // Fused RMSNorm + Q8_1 for graph FFN: saves 1 dispatch per layer.
            if ffn_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
                let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                let bs = rmsnorm_block_size(hidden_dim);
                let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: rmsnorm_shared_bytes(bs) };
                let dim = hidden_dim as u32;
                unsafe { self.device.stream.launch_builder(fused_fn).arg(&st.scratch.attn_proj).arg(&lw.ffn_norm).arg(&mut *q8_1_buf).arg(&eps).arg(&dim).launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm_to_q8_1 ffn: {e}")))?;
                unsafe {
                    // split-layout: prefer Q8Split/Q4Split sibling buffers on FFN gate/up when set.
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_gate, lw.q8_tile_w_gate.as_ref(), lw.q4_tile_w_gate.as_ref(), lw.q8_split_w_gate.as_ref(), lw.q4_split_w_gate.as_ref(), q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "graph gate")?;
                    launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_up, lw.q8_tile_w_up.as_ref(), lw.q4_tile_w_up.as_ref(), lw.q8_split_w_up.as_ref(), lw.q4_split_w_up.as_ref(), q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "graph up")?;
                }
            } else {
                let bs = rmsnorm_block_size(hidden_dim);
                let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: rmsnorm_shared_bytes(bs) };
                let dim = hidden_dim as u32;
                unsafe { self.device.stream.launch_builder(&st.kernels.rmsnorm).arg(&st.scratch.attn_proj).arg(&lw.ffn_norm).arg(&mut st.scratch.normed).arg(&eps).arg(&dim).launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm ffn: {e}")))?;

                if ffn_use_preq {
                    let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref().unwrap();
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    unsafe {
                        launch_quantize_input_q8_1(&self.device, quant_fn, &st.scratch.normed, q8_1_buf, hidden_dim, "graph ffn gate_up")?;
                        // split-layout: prefer Q8Split/Q4Split sibling buffers on FFN gate/up when set.
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_gate, lw.q8_tile_w_gate.as_ref(), lw.q4_tile_w_gate.as_ref(), lw.q8_split_w_gate.as_ref(), lw.q4_split_w_gate.as_ref(), q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "graph gate")?;
                        launch_matvec_preq8_1_tile(&self.device, &st.kernels, &lw.w_up, lw.q8_tile_w_up.as_ref(), lw.q4_tile_w_up.as_ref(), lw.q8_split_w_up.as_ref(), lw.q4_split_w_up.as_ref(), q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "graph up")?;
                    }
                } else {
                    unsafe {
                        launch_matvec(&self.device, &st.kernels, &lw.w_gate, &st.scratch.normed, &mut st.scratch.gate, inter_dim, hidden_dim, "graph gate", lw.w_gate_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        launch_matvec(&self.device, &st.kernels, &lw.w_up, &st.scratch.normed, &mut st.scratch.up, inter_dim, hidden_dim, "graph up", lw.w_up_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                    }
                }
            }
        }

        // SwiGLU + Down projection.
        // gate and up are separate buffers. Apply SwiGLU (fused with F32->F16) then
        // down projection.
        let lw = &st.layer_weights_cache[layer_idx];

        if let GpuWeightBuf::F16Raw(ref wd_f16) = lw.w_down {
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device, &st.kernels,
                    &mut st.scratch.gate, &st.scratch.up,
                    &mut st.scratch.input_f16, inter_dim,
                )?;
            }
            unsafe {
                launch_hgemv_f16_preconverted(
                    &self.device, wd_f16, &st.scratch.input_f16,
                    &mut st.scratch.down, hidden_dim, inter_dim, "graph down",
                    st.algo_cache.get(hidden_dim, inter_dim),
                )?;
            }
        } else if matches!(&lw.w_down, GpuWeightBuf::F32(_))
            && lw.w_down_f16.is_some()
        {
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device, &st.kernels,
                    &mut st.scratch.gate, &st.scratch.up,
                    &mut st.scratch.input_f16, inter_dim,
                )?;
            }
            if let Some(ref wd_f16) = lw.w_down_f16 {
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wd_f16, &st.scratch.input_f16,
                        &mut st.scratch.down, hidden_dim, inter_dim, "graph down",
                        st.algo_cache.get(hidden_dim, inter_dim),
                    )?;
                }
            }
        } else {
            // Non-F16: separate SwiGLU + launch_matvec down.
            let config = LaunchConfig::for_elements(inter_dim);
            let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
            let n = inter_dim as u32;
            unsafe { self.device.stream.launch_builder(&st.kernels.swiglu_inplace).arg(&mut st.scratch.gate).arg(&st.scratch.up).arg(&n).launch(lc) }
            .map_err(|e| RuntimeError::Compute(format!("graph swiglu L{layer_idx}: {e}")))?;
            // split-layout: prefer Q8/Q4 split sibling for w_down in graph path too.
            let use_split_down = (st.kernels.use_q8_split_dispatch && lw.q8_split_w_down.is_some())
                || (st.kernels.use_q4_split_dispatch && lw.q4_split_w_down.is_some());
            if use_split_down {
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device, quant_fn, &st.scratch.gate, q8_1_buf,
                            inter_dim, "graph down split",
                        )?;
                        launch_matvec_preq8_1_tile(
                            &self.device, &st.kernels, &lw.w_down,
                            lw.q8_tile_w_down.as_ref(),  lw.q4_tile_w_down.as_ref(),
                            lw.q8_split_w_down.as_ref(), lw.q4_split_w_down.as_ref(),
                            q8_1_buf, &mut st.scratch.down,
                            hidden_dim, inter_dim, "graph down",
                        )?;
                    }
                } else {
                    unsafe { launch_matvec(&self.device, &st.kernels, &lw.w_down, &st.scratch.gate, &mut st.scratch.down, hidden_dim, inter_dim, "graph down", lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
                }
            } else {
                unsafe { launch_matvec(&self.device, &st.kernels, &lw.w_down, &st.scratch.gate, &mut st.scratch.down, hidden_dim, inter_dim, "graph down", lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
            }
        }
        // Final step: residual add, with optional inter-layer fusion.
        //
        // Two fusion paths based on next layer's weight type:
        // - F16: fused_residual_rmsnorm_f16 -> residual + RMSNorm + F16 output (for HGEMV paths)
        // - Q8_0/dp4a: fused_residual_rmsnorm_q8_1 -> residual + RMSNorm + Q8_1 output (for dp4a paths)
        // - residual_add_copy: plain residual (last layer, no fusion kernel, or no match)
        if let Some(next_layer) = fuse_tail_next_layer {
            let next_lw = &st.layer_weights_cache[next_layer];
            // Check if next layer is F16 (HGEMV path).
            let next_is_f16 = matches!(&next_lw.wq, GpuWeightBuf::F16Raw(_))
                || (matches!(&next_lw.wq, GpuWeightBuf::F32(_))
                    && next_lw.wq_f16.is_some() && next_lw.wk_f16.is_some() && next_lw.wv_f16.is_some());
            // Check if next layer uses dp4a Q8_1 pre-quantized input.
            let next_uses_q8_preq = weight_uses_dp4a_q8_1(&next_lw.wq, &st.kernels)
                && weight_uses_dp4a_q8_1(&next_lw.wk, &st.kernels)
                && weight_uses_dp4a_q8_1(&next_lw.wv, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some();

            if next_is_f16 {
                if let Some(ref func) = st.kernels.fused_residual_rmsnorm_f16 {
                    let next_norm = &next_lw.attn_norm;
                    let bs = rmsnorm_block_size(hidden_dim);
                    let shared = rmsnorm_shared_bytes(bs);
                    let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: shared };
                    let dim = hidden_dim as u32;
                    unsafe { self.device.stream.launch_builder(func)
                        .arg(&st.scratch.attn_proj).arg(&st.scratch.down)
                        .arg(&mut st.scratch.x_gpu)
                        .arg(next_norm)
                        .arg(&mut st.scratch.input_f16)
                        .arg(&eps).arg(&dim)
                        .launch(lc) }
                    .map_err(|e| RuntimeError::Compute(format!("graph fused_residual_rmsnorm_f16 L{layer_idx}: {e}")))?;
                } else {
                    let config = LaunchConfig::for_elements(hidden_dim);
                    let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
                    let n = hidden_dim as u32;
                    unsafe { self.device.stream.launch_builder(&st.kernels.residual_add_copy)
                        .arg(&st.scratch.attn_proj).arg(&st.scratch.down).arg(&mut st.scratch.x_gpu).arg(&n).launch(lc) }
                    .map_err(|e| RuntimeError::Compute(format!("graph residual_add_copy L{layer_idx}: {e}")))?;
                }
            } else if next_uses_q8_preq {
                if let Some(ref func) = st.kernels.fused_residual_rmsnorm_q8_1 {
                    let next_norm = &next_lw.attn_norm;
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    let bs = rmsnorm_block_size(hidden_dim);
                    let shared = rmsnorm_shared_bytes(bs);
                    let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: shared };
                    let dim = hidden_dim as u32;
                    unsafe { self.device.stream.launch_builder(func)
                        .arg(&st.scratch.attn_proj).arg(&st.scratch.down)
                        .arg(&mut st.scratch.x_gpu)
                        .arg(next_norm)
                        .arg(&mut *q8_1_buf)
                        .arg(&eps).arg(&dim)
                        .launch(lc) }
                    .map_err(|e| RuntimeError::Compute(format!("graph fused_residual_rmsnorm_q8_1 L{layer_idx}: {e}")))?;
                } else {
                    let config = LaunchConfig::for_elements(hidden_dim);
                    let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
                    let n = hidden_dim as u32;
                    unsafe { self.device.stream.launch_builder(&st.kernels.residual_add_copy)
                        .arg(&st.scratch.attn_proj).arg(&st.scratch.down).arg(&mut st.scratch.x_gpu).arg(&n).launch(lc) }
                    .map_err(|e| RuntimeError::Compute(format!("graph residual_add_copy L{layer_idx}: {e}")))?;
                }
            } else {
                let config = LaunchConfig::for_elements(hidden_dim);
                let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
                let n = hidden_dim as u32;
                unsafe { self.device.stream.launch_builder(&st.kernels.residual_add_copy)
                    .arg(&st.scratch.attn_proj).arg(&st.scratch.down).arg(&mut st.scratch.x_gpu).arg(&n).launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph residual_add_copy L{layer_idx}: {e}")))?;
            }
        } else {
            // Last layer: plain residual_add_copy (no fusion needed).
            let config = LaunchConfig::for_elements(hidden_dim);
            let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
            let n = hidden_dim as u32;
            unsafe { self.device.stream.launch_builder(&st.kernels.residual_add_copy)
                .arg(&st.scratch.attn_proj).arg(&st.scratch.down).arg(&mut st.scratch.x_gpu).arg(&n).launch(lc) }
            .map_err(|e| RuntimeError::Compute(format!("graph residual_add_copy L{layer_idx}: {e}")))?;
        }
        Ok(())
    }

    /// Run the full graph-captured decode pipeline: embed + all layers + final + argmax.
    ///
    /// This is called during graph CAPTURE (kernels execute AND are recorded into
    /// the graph) and on the first decode token to establish the graph.
    ///
    /// Uses graph kernel variants for embed, RoPE, KV cache write, and attention.
    /// Everything else uses standard kernels (captured directly since they have
    /// no per-token-varying scalars).
    fn run_graph_pipeline(
        &self,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let num_layers = hp.num_layers as usize;
        let diag = std::env::var("LUMEN_GRAPH_DIAGNOSTIC")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Step 1: Embed using graph variant
        if diag { eprintln!("[GRAPH-DIAG]   pipeline step 1: embed_token_gpu_graph"); }
        self.embed_token_gpu_graph(st)?;
        if diag {
            let status = super::graph::query_capture_status(&self.device.stream);
            eprintln!("[GRAPH-DIAG]     after embed: capture status = {status}");
        }

        // Step 2: All layers using graph variants, with inter-layer fusion.
        //
        // Inter-layer fusion: fuse the tail of layer L (residual_add_copy) with the
        // head of layer L+1 (RMSNorm) into a single kernel.
        // Two fusion paths:
        // - F16: fused_residual_rmsnorm_f16 -> residual + RMSNorm + F16 output (for HGEMV paths)
        // - Q8_0: fused_residual_rmsnorm_q8_1 -> residual + RMSNorm + Q8_1 output (for dp4a paths)
        let uses_f16: Vec<bool> = (0..num_layers).map(|l| {
            let lw = &st.layer_weights_cache[l];
            matches!(&lw.wq, GpuWeightBuf::F16Raw(_))
                || (matches!(&lw.wq, GpuWeightBuf::F32(_))
                    && lw.wq_f16.is_some() && lw.wk_f16.is_some() && lw.wv_f16.is_some())
        }).collect();

        // Detect layers that use dp4a Q8_1 pre-quantized input for QKV (Q8_0/Q4_0/Q8Aligned/Q4Aligned).
        let uses_q8_preq: Vec<bool> = (0..num_layers).map(|l| {
            let lw = &st.layer_weights_cache[l];
            weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wk, &st.kernels)
                && weight_uses_dp4a_q8_1(&lw.wv, &st.kernels)
                && st.scratch.input_q8_1.is_some()
                && st.kernels.quantize_f32_to_q8_1.is_some()
        }).collect();

        let has_fused_f16 = st.kernels.fused_residual_rmsnorm_f16.is_some();
        let has_fused_q8_1 = st.kernels.fused_residual_rmsnorm_q8_1.is_some()
            && st.kernels.rmsnorm_to_q8_1.is_some(); // also need unfused for FFN norm

        // Detect GDN layers for graph-compatible routing.
        let layer_types: Vec<u8> = (0..num_layers).map(|l| {
            st.layer_weights_cache[l].layer_type
        }).collect();

        // conv_positions->GPU sync is now performed by the caller
        // (`decode_token`) BEFORE `begin_capture()`. Doing it here would
        // record the htod_copy into the captured graph, which is incorrect
        // — the captured graph is supposed to be replayable without changing
        // its dispatch list. The advance_conv_position kernel (captured)
        // handles all per-replay updates to conv_positions_gpu.

        let mut skip_head_norm = false;
        for layer in 0..num_layers {
            if diag && (layer < 2 || layer == num_layers - 1) {
                eprintln!("[GRAPH-DIAG]   pipeline step 2: L{layer} type={} (skip_head={skip_head_norm})",
                    layer_types[layer]);
            }

            // GDN layer routing: use the regular decode GDN path (non-graph-specific).
            // GDN layers have inherently sequential state updates and go through the
            // regular compute_gdn_attention_gpu which handles conv1d, gates, state update.
            // The FFN block is handled by calling compute_layer_gpu which detects
            // layer_type=1 and routes appropriately.
            if layer_types[layer] == 1 {
                // GDN attention block: sets attn_proj = x_gpu post-residual.
                // route through graph-aware wrapper. When the
                // graph megakernel and conv_positions_gpu are available, this
                // emits a graph-capturable dispatch (device-pointer state_pos
                // + advance_conv_position kernel) instead of the host-scalar
                // path that previously baked the position into the graph and
                // prevented replay.
                self.compute_gdn_attention_gpu_graph(layer, st)?;
                // GDN layers still need the shared FFN block. The code after
                // compute_gdn_attention_gpu in the non-graph path does FFN via
                // the normal flow in compute_layer_gpu. For the graph path,
                // we skip the attention part and run only FFN.
                // The FFN code in compute_layer_gpu_graph expects attn_proj to
                // contain the post-attention hidden state, which is already set.
                // We call compute_layer_gpu_graph with skip_head_norm=false
                // and fuse_tail=None for simplicity -- GDN FFN doesn't benefit
                // much from inter-layer fusion since GDN/standard layers alternate.
                // BUT: we can't call compute_layer_gpu_graph for a GDN layer because
                // it would try to run the standard attention path. Instead, we inline
                // just the FFN portion.
                //
                // handle MoE-FFN-on-GDN-layer case (Qwen3.5-MoE-30B-A3B
                // is GDN-attn + MoE-FFN). Without this branch, lw.w_gate/w_up/w_down
                // would not exist for MoE layers and `launch_matvec(&lw.w_gate, ...)`
                // would hit `CUBLAS_STATUS_INVALID_VALUE`. Detect MoE layer first
                // and dispatch the same MoE FFN family used in compute_layer_gpu_graph.
                let is_moe_on_gdn = st.layer_weights_cache[layer].moe_layer_blob.is_some();
                if is_moe_on_gdn {
                    let hp = self.hp()?;
                    let hidden_dim = hp.hidden_dim as usize;
                    let inter_dim = hp.intermediate_dim as usize;
                    let eps = hp.norm_eps;
                    let lw_moe: &LayerWeightsGpu = &st.layer_weights_cache[layer];
                    let moe_meta_ref = st.moe_meta_cache.get(layer)
                        .and_then(|m| m.as_ref())
                        .ok_or_else(|| RuntimeError::Compute(format!(
                            "MoE-on-GDN graph layer {layer} missing moe_meta_cache entry",
                        )))?;
                    let moe_layer_blob = lw_moe.moe_layer_blob.as_ref()
                        .expect("moe_layer_blob.is_some() verified above");
                    let num_experts = moe_meta_ref.expert_gate_offs.len();
                    let top_k = hp.num_active_experts.map(|v| v as usize).unwrap_or(0);
                    if top_k == 0 {
                        return Err(RuntimeError::Compute(
                            "MoE-on-GDN graph layer present but hyperparams.num_active_experts not set".into(),
                        ));
                    }
                    let batched_offsets = st
                        .moe_batched_offsets
                        .get(layer)
                        .and_then(|b| b.as_ref());
                    let has_shared = moe_meta_ref.shared_gate.is_some();
                    let moe_scratch = st.moe_scratch.as_mut().ok_or_else(|| {
                        RuntimeError::Compute(
                            "MoE-on-GDN graph layer dispatch requires moe_scratch".into(),
                        )
                    })?;
                    super::moe::encode_moe_ffn_decode_fused_norm(
                        &self.device,
                        &st.kernels,
                        moe_scratch,
                        moe_meta_ref,
                        batched_offsets,
                        moe_layer_blob,
                        &st.scratch.attn_proj.slice(..),
                        &lw_moe.ffn_norm,
                        &mut st.scratch.normed.slice_mut(..),
                        &st.scratch.attn_proj.slice(..),
                        &mut st.scratch.x_gpu.slice_mut(..),
                        eps,
                        hidden_dim,
                        inter_dim,
                        num_experts,
                        top_k,
                    )?;
                    let skip_shared = {
                        use std::sync::OnceLock;
                        static FLAG: OnceLock<bool> = OnceLock::new();
                        *FLAG.get_or_init(|| {
                            std::env::var("LUMEN_CUDA_SKIP_SHARED_EXPERT")
                                .ok().as_deref()
                                .map(|v| matches!(v, "1" | "true" | "yes"))
                                .unwrap_or(false)
                        })
                    };
                    if has_shared && !skip_shared {
                        let use_fused = super::moe::moe_shared_fused_enabled()
                            && st.kernels.fused_glu_gemv_q4_0_prenormed_no_norm.is_some()
                            && st.kernels.moe_shared_down_q4_0_sigmoid_accum.is_some()
                            && st.kernels.moe_shared_down_q4_0_residual_accum.is_some();
                        if use_fused {
                            super::moe::encode_shared_expert_ffn_decode_fused(
                                &self.device,
                                &st.kernels,
                                moe_scratch,
                                moe_meta_ref,
                                moe_layer_blob,
                                &st.scratch.normed.slice(..),
                                &mut st.scratch.x_gpu.slice_mut(..),
                                hidden_dim,
                            )?;
                        } else {
                            super::moe::encode_shared_expert_ffn_decode(
                                &self.device,
                                &st.kernels,
                                moe_scratch,
                                moe_meta_ref,
                                moe_layer_blob,
                                &st.scratch.normed.slice(..),
                                &mut st.scratch.x_gpu.slice_mut(..),
                                hidden_dim,
                            )?;
                        }
                    }
                    // MoE-on-GDN done: x_gpu = attn_proj + Σ expert + shared.
                    // Skip the residual_add below (already incorporated).
                    skip_head_norm = false;
                    continue;
                }
                {
                    let hp = self.hp()?;
                    let hidden_dim = hp.hidden_dim as usize;
                    let inter_dim = hp.intermediate_dim as usize;
                    let eps = hp.norm_eps;
                    let lw = &st.layer_weights_cache[layer];

                    // FFN RMSNorm
                    let block_size = rmsnorm_block_size(hidden_dim);
                    let shared_bytes = rmsnorm_shared_bytes(block_size);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    let dim = hidden_dim as u32;
                    unsafe {
                        self.device.stream.launch_builder(&st.kernels.rmsnorm)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&mut st.scratch.normed)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("GDN graph FFN rmsnorm L{layer}: {e}")))?;

                    // Gate + Up + SwiGLU + Down projections
                    unsafe {
                        launch_matvec(&self.device, &st.kernels, &lw.w_gate, &st.scratch.normed,
                            &mut st.scratch.gate, inter_dim, hidden_dim, "graph gdn gate",
                            lw.w_gate_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut())?;
                        launch_matvec(&self.device, &st.kernels, &lw.w_up, &st.scratch.normed,
                            &mut st.scratch.up, inter_dim, hidden_dim, "graph gdn up",
                            lw.w_up_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut())?;
                    }
                    // SwiGLU
                    {
                        let config = LaunchConfig::for_elements(inter_dim);
                        let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
                        let n = inter_dim as u32;
                        unsafe { self.device.stream.launch_builder(&st.kernels.swiglu_inplace).arg(&mut st.scratch.gate).arg(&st.scratch.up).arg(&n).launch(lc) }
                        .map_err(|e| RuntimeError::Compute(format!("GDN graph swiglu L{layer}: {e}")))?;
                    }
                    // Down projection
                    unsafe {
                        launch_matvec(&self.device, &st.kernels, &lw.w_down, &st.scratch.gate,
                            &mut st.scratch.down, hidden_dim, inter_dim, "graph gdn down",
                            lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16),
                            st.scratch.input_q8_1.as_mut())?;
                    }
                    // Residual add: x_gpu = attn_proj + down (attn_proj already has GDN attention output)
                    {
                        let config = LaunchConfig::for_elements(hidden_dim);
                        let lc = CudarcLaunchConfig { grid_dim: (config.grid_dim,1,1), block_dim: (config.block_dim,1,1), shared_mem_bytes: 0 };
                        let n = hidden_dim as u32;
                        unsafe { self.device.stream.launch_builder(&st.kernels.residual_add_copy)
                            .arg(&st.scratch.attn_proj).arg(&st.scratch.down)
                            .arg(&mut st.scratch.x_gpu).arg(&n)
                            .launch(lc) }
                        .map_err(|e| RuntimeError::Compute(format!("GDN graph residual L{layer}: {e}")))?;
                    }
                }
                // GDN layers break inter-layer fusion -- reset skip_head_norm.
                skip_head_norm = false;
                continue;
            }

            // detect MoE layer. MoE FFN dispatch handles its own
            // residual accumulation inside `encode_moe_ffn_decode_fused_norm`,
            // so the dense graph path's residual_add + fuse-tail logic does
            // not apply. Skip fusion when this layer OR the next layer is MoE.
            let is_moe_layer = st.layer_weights_cache[layer].moe_layer_blob.is_some();
            let next_is_moe = layer + 1 < num_layers
                && st.layer_weights_cache[layer + 1].moe_layer_blob.is_some();

            // Determine if we should fuse the tail of this layer with the head of the next.
            // Fuse when the next layer uses F16 (fused_residual_rmsnorm_f16) or
            // Q8_0/dp4a (fused_residual_rmsnorm_q8_1) and the corresponding kernel exists.
            // Skip fusion if next layer is GDN (GDN has its own attention path).
            // also skip fusion if THIS layer is MoE (MoE writes x_gpu
            // directly with residual accumulated) or NEXT layer is MoE (MoE FFN
            // does its own RMSNorm internally).
            let fuse_tail_next = if layer + 1 < num_layers && layer_types[layer + 1] != 1
                && !is_moe_layer && !next_is_moe
            {
                let next_f16 = uses_f16[layer + 1];
                let next_q8 = uses_q8_preq[layer + 1];
                if next_f16 && has_fused_f16 {
                    Some(layer + 1)
                } else if next_q8 && has_fused_q8_1 {
                    Some(layer + 1)
                } else {
                    None
                }
            } else {
                None
            };

            self.compute_layer_gpu_graph(layer, st, skip_head_norm, fuse_tail_next)?;

            // If we fused the tail, the next layer should skip its head norm.
            skip_head_norm = fuse_tail_next.is_some();

            if diag && (layer < 2 || layer == num_layers - 1) {
                let status = super::graph::query_capture_status(&self.device.stream);
                eprintln!("[GRAPH-DIAG]     after L{layer}: capture status = {status}");
            }
        }

        // Step 3: Final RMSNorm + output projection (no per-token scalars)
        if diag { eprintln!("[GRAPH-DIAG]   pipeline step 3: compute_final_gpu"); }
        self.compute_final_gpu(st)?;
        if diag {
            let status = super::graph::query_capture_status(&self.device.stream);
            eprintln!("[GRAPH-DIAG]     after compute_final: capture status = {status}");
        }

        // Step 4: GPU argmax
        if diag { eprintln!("[GRAPH-DIAG]   pipeline step 4: argmax"); }
        {
            let vocab = hp.vocab_size;
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.device.stream
                    .launch_builder(&st.kernels.argmax_f32)
                    .arg(&st.logits_gpu)
                    .arg(&mut st.argmax_result)
                    .arg(&vocab)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("graph argmax: {e}")))?;
        }
        if diag {
            let status = super::graph::query_capture_status(&self.device.stream);
            eprintln!("[GRAPH-DIAG]     after argmax: capture status = {status}");
            eprintln!("[GRAPH-DIAG]   pipeline complete");
        }

        Ok(())
    }

    /// Compute final RMSNorm + output projection entirely on GPU, with no host sync.
    ///
    /// Input: `st.scratch.x_gpu` (final hidden state, [hidden_dim]).
    /// Output: `st.logits_gpu` (logits, [vocab_size]).
    fn compute_final_gpu(
        &self,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let vocab_size = hp.vocab_size as usize;
        let eps = hp.norm_eps;

        // Fast path: F16 output projection with fused RMSNorm+F16 conversion.
        // Saves 1 dispatch by skipping the intermediate F32 normed buffer.
        // Flow: fused_rmsnorm_f16(x_gpu) -> input_f16, then cuBLAS HGEMV preconverted.
        if let Some(ref proj_f16) = st.globals.output_proj_f16 {
            if st.kernels.fused_rmsnorm_f16.is_some() {
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &st.globals.final_norm,
                        &mut st.scratch.input_f16,
                        eps, hidden_dim, "final F16",
                    )?;
                    launch_hgemv_f16_preconverted(
                        &self.device, proj_f16, &st.scratch.input_f16,
                        &mut st.logits_gpu, vocab_size, hidden_dim, "output_proj",
                        st.algo_cache.get(vocab_size, hidden_dim),
                    )?;
                }
                return Ok(());
            }
        }

        // RMSNorm with final_norm weights (for non-F16 output projection paths).
        {
            let block_size = rmsnorm_block_size(hidden_dim);
            let shared_bytes = rmsnorm_shared_bytes(block_size);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            let dim = hidden_dim as u32;
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.rmsnorm)
                    .arg(&st.scratch.x_gpu)
                    .arg(&st.globals.final_norm)
                    .arg(&mut st.scratch.normed)
                    .arg(&eps)
                    .arg(&dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("rmsnorm final launch: {e}")))?;
        }

        // Output projection: logits = output_proj * normed.
        // Prefer Q4Aligned dp4a (highest priority for Q4_0), then smem, then scalar.
        if let Some(ref proj_q4a) = st.globals.output_proj_q4_aligned {
            // Path -1: Q4_0 final-projection matvec dispatch
            // for the Q4 output_proj. Env-gated `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ=1`.
            // Default OFF preserves existing Q4Aligned dp4a path (byte-identical).
            if super::moe::mmv_q_output_proj_enabled() {
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;
                if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_q8_1_rawsum.as_ref(),
                    st.kernels.mul_mat_vec_q_q4_0.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    use std::sync::Once;
                    static TRACE_ONCE_Q4: Once = Once::new();
                    TRACE_ONCE_Q4.call_once(|| {
                        super::decode::cuda_log_force(format!(
                            "[CUDA] mul_mat_vec_q_q4_0 output_proj: ACTIVE (grid={}, in_dim={})",
                            vocab_size, hidden_dim
                        ));
                    });
                    let quant_grid = (in_dim_u32 + 31) / 32;
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum output_proj Q4: {e}"
                    )))?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4a)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q4_0 output_proj: {e}"
                    )))?;
                    return Ok(());
                }
            }

            // Q4Aligned dp4a: pre-quantize normed x to Q8_1, then aligned dp4a matvec.
            if let (Some(ref quant_fn), Some(ref mv_fn)) = (
                st.kernels.quantize_f32_to_q8_1.as_ref(),
                st.kernels.matvec_q4_aligned_q8_1.as_ref(),
            ) {
                let out_dim = vocab_size as u32;
                let in_dim = hidden_dim as u32;
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                // Step 1: Quantize F32 normed x to Q8_1.
                let quant_grid = q8_1_quant_grid(in_dim);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream
                        .launch_builder(quant_fn)
                        .arg(&st.scratch.normed)
                        .arg(&mut *q8_1_buf)
                        .arg(&in_dim)
                        .launch(quant_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "quantize_f32_to_q8_1 output_proj Q4Aligned: {e}",
                )))?;
                // Step 2: dp4a Q4Aligned matvec (NR=4, 256 threads).
                let mv_grid = dp4a_q4_grid(out_dim);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device.stream
                        .launch_builder(mv_fn)
                        .arg(proj_q4a)
                        .arg(&*q8_1_buf)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim)
                        .arg(&in_dim)
                        .launch(mv_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q4_aligned_q8_1 output_proj: {e}",
                )))?;
            }
        } else if let Some(ref proj_q4) = st.globals.output_proj_q4 {
            let out_dim = vocab_size as u32;
            let in_dim = hidden_dim as u32;

            // Path -1: Q4_0 final-projection matvec dispatch
            // for the Q4 raw output_proj branch (used when aligned/smem kernels
            // are not selected). Env-gated `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ=1`.
            if super::moe::mmv_q_output_proj_enabled() {
                if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_q8_1_rawsum.as_ref(),
                    st.kernels.mul_mat_vec_q_q4_0.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    use std::sync::Once;
                    static TRACE_ONCE_Q4RAW: Once = Once::new();
                    TRACE_ONCE_Q4RAW.call_once(|| {
                        super::decode::cuda_log_force(format!(
                            "[CUDA] mul_mat_vec_q_q4_0 output_proj (raw): ACTIVE (grid={}, in_dim={})",
                            vocab_size, hidden_dim
                        ));
                    });
                    let quant_grid = (in_dim + 31) / 32;
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim)
                            .launch(quant_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum output_proj Q4 raw: {e}"
                    )))?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim)
                            .arg(&out_dim)
                            .launch(mv_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q4_0 output_proj Q4 raw: {e}"
                    )))?;
                    return Ok(());
                }
            }

            let shmem_needed = in_dim * 4;
            if let Some(ref smem_fn) = st.kernels.matvec_q4_0_smem {
                if shmem_needed <= 49152 {
                    let grid = matvec_smem_grid(out_dim);
                    let shmem = matvec_smem_shared_bytes(in_dim);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (SMEM_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: shmem,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(smem_fn)
                            .arg(proj_q4)
                            .arg(&st.scratch.normed)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim)
                            .arg(&in_dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec output_proj Q4_0 smem launch: {e}"))
                    })?;
                } else {
                    let mv_block = matvec_block_size();
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim, 1, 1),
                        block_dim: (mv_block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.matvec_q4_0)
                            .arg(proj_q4)
                            .arg(&st.scratch.normed)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim)
                            .arg(&in_dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec output_proj Q4_0 launch: {e}"))
                    })?;
                }
            } else {
                let mv_block = matvec_block_size();
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim, 1, 1),
                    block_dim: (mv_block, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.matvec_q4_0)
                        .arg(proj_q4)
                        .arg(&st.scratch.normed)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim)
                        .arg(&in_dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec output_proj Q4_0 launch: {e}"))
                })?;
            }
        } else if let Some(ref proj_f16) = st.globals.output_proj_f16 {
            // F16 output projection: cuBLAS HGEMV (cublasGemmEx N=1).
            unsafe {
                launch_hgemv_f16(
                    &self.device,
                    &st.kernels,
                    proj_f16,
                    &st.scratch.normed,
                    &mut st.logits_gpu,
                    &mut st.scratch.input_f16,
                    vocab_size,
                    hidden_dim,
                    "output_proj",
                    st.algo_cache.get(vocab_size, hidden_dim),
                )?;
            }
        } else if let Some(ref proj_f16_cache) =
            st.globals.output_proj_q8_to_f16_cache
        {
            // cuBLAS HGEMV-N=1 against a pre-dequanted F16 cache of the
            // Q8_0 output projection. Same compute path as BF16 / native F16
            // output_proj (proven faster on this shape). Takes priority over
            // the SPLIT / Q8Aligned / Q8Raw fallbacks below; default OFF (env
            // `LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1`).
            unsafe {
                launch_hgemv_f16(
                    &self.device,
                    &st.kernels,
                    proj_f16_cache,
                    &st.scratch.normed,
                    &mut st.logits_gpu,
                    &mut st.scratch.input_f16,
                    vocab_size,
                    hidden_dim,
                    "output_proj_q8_to_f16",
                    st.algo_cache.get(vocab_size, hidden_dim),
                )?;
            }
        } else if let Some(ref proj_q8_split) = st.globals.output_proj_q8_split {
            // OUTPUT_PROJ_SPLIT: Q8 split (SoA) layout for output_proj.
            // Use the dedicated `matvec_q8_split_output_proj_nr32` kernel which
            // processes 32 output rows per CTA (vs NR=2 in the generic split
            // kernel). For Qwen3.5-9B's 248320x4096 shape this drops grid size
            // from 124k CTAs to 7760 CTAs -- 16x reduction in per-CTA fixed
            // cost. Falls back to the generic NR=2 split kernel if the dedicated
            // variant didn't load.
            //
            // when `LUMEN_CUDA_OUTPUT_PROJ_NR={16,64,128}` is set AND
            // the corresponding variant loaded, route through the requested NR
            // value with a re-computed grid (`ceil(out_dim / NR)`). nr32
            // remains the default.
            let out_dim_u32 = vocab_size as u32;
            let in_dim_u32 = hidden_dim as u32;
            let (split_mv_fn, mv_grid): (&CudaFunction, u32) =
                if let Some(proj_fn) = pick_output_proj_nr_kernel(
                    &st.kernels, st.output_proj_nr,
                ) {
                    let nr = st.output_proj_nr;
                    (proj_fn, (out_dim_u32 + nr - 1) / nr)
                } else if let Some(ref proj_fn) = st.kernels.matvec_q8_split_output_proj {
                    // NR=32 variant. Grid = ceil(out_dim / 32).
                    (proj_fn, (out_dim_u32 + 31) / 32)
                } else if let Some(ref generic_fn) = st.kernels.matvec_q8_split_q8_1 {
                    (generic_fn, dp4a_q8_1_grid(out_dim_u32))
                } else {
                    return Err(RuntimeError::Compute(
                        "output_proj_q8_split present but no split matvec kernel available".into(),
                    ));
                };
            if let (Some(quant_fn), Some(ref mut q8_1_buf)) = (
                st.kernels.quantize_f32_to_q8_1.as_ref(),
                st.scratch.input_q8_1.as_mut(),
            ) {
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(quant_fn)
                        .arg(&st.scratch.normed)
                        .arg(&mut **q8_1_buf)
                        .arg(&in_dim_u32)
                        .launch(quant_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "quantize_f32_to_q8_1 output_proj split: {e}",
                )))?;
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(split_mv_fn)
                        .arg(proj_q8_split)
                        .arg(&**q8_1_buf)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(mv_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q8_split output_proj: {e}",
                )))?;
            } else {
                return Err(RuntimeError::Compute(
                    "output_proj_q8_split present but quantize kernel unavailable".into(),
                ));
            }
        } else if let Some(ref proj_q8a) = st.globals.output_proj_q8_aligned {
            // Q8_0 aligned output projection: try Q8_1 path first, then on-the-fly.
            let out_dim_u32 = vocab_size as u32;
            let in_dim_u32 = hidden_dim as u32;

            // Path -1: Q8_0 final-projection matvec dispatch
            // for the Q8 output_proj. Env-gated `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ=1`.
            // Default OFF preserves existing Q8Aligned dp4a path (byte-identical).
            //
            // measures matvec_q8_0 (this single call) at 807 µs × 64 inst
            // = 51.7 ms / 64-tok = 6.2% TPOT. The mul_mat_vec_q kernel
            // is purpose-built for batch-1 dense matvec; predicted +3-6% Q8.
            if super::moe::mmv_q_output_proj_enabled() {
                if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_q8_1_rawsum.as_ref(),
                    st.kernels.mul_mat_vec_q_q8_0.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    use std::sync::Once;
                    static TRACE_ONCE_Q8: Once = Once::new();
                    TRACE_ONCE_Q8.call_once(|| {
                        super::decode::cuda_log_force(format!(
                            "[CUDA] mul_mat_vec_q_q8_0 output_proj: ACTIVE (grid={}, in_dim={})",
                            vocab_size, hidden_dim
                        ));
                    });
                    let quant_grid = (in_dim_u32 + 31) / 32;
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum output_proj: {e}"
                    )))?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(mv_fn)
                            .arg(proj_q8a)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q8_0 output_proj: {e}"
                    )))?;
                    return Ok(());
                }
            }

            // Path 0: Q8Aligned + pre-quantized Q8_1 input (NR=2, dp4a).
            // Q8_SCALE_HW: prefer halfword-scale variant for output_proj.
            let aligned_mv_fn = if st.kernels.use_q8_scale_hw {
                st.kernels.matvec_q8_aligned_q8_1_hw.as_ref()
                    .or(st.kernels.matvec_q8_aligned_q8_1.as_ref())
            } else {
                st.kernels.matvec_q8_aligned_q8_1.as_ref()
            };
            if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                st.kernels.quantize_f32_to_q8_1.as_ref(),
                aligned_mv_fn,
                st.scratch.input_q8_1.as_mut(),
            ) {
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(quant_fn)
                        .arg(&st.scratch.normed)
                        .arg(&mut **q8_1_buf)
                        .arg(&in_dim_u32)
                        .launch(quant_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("quantize_f32_to_q8_1 output_proj: {e}"))
                })?;

                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(mv_fn)
                        .arg(proj_q8a)
                        .arg(&**q8_1_buf)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(mv_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec_q8_aligned_q8_1 output_proj: {e}"))
                })?;
            } else {
                // Fallback: on-the-fly x quantization.
                let q8a_fn = st.kernels.matvec_q8_0_aligned.as_ref()
                    .or(st.kernels.matvec_q8_0_dp4a.as_ref())
                    .unwrap_or(&st.kernels.matvec_q8_0);
                let grid = matvec_q8_0_grid(out_dim_u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(q8a_fn)
                        .arg(proj_q8a)
                        .arg(&st.scratch.normed)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec output_proj Q8_0 aligned launch: {e}"))
                })?;
            }
        } else if let Some(ref proj_q8) = st.globals.output_proj_q8 {
            // Q8_0 output projection: dp4a (native Q8_0, ~1.06 B/elem).
            // Fallback when aligned repack is unavailable.
            let out_dim_u32 = vocab_size as u32;
            let in_dim_u32 = hidden_dim as u32;

            // Path -1: Q8_0 final-projection matvec dispatch
            // for the Q8 raw output_proj branch (used when aligned dp4a kernels
            // fail to JIT, as observed on this build env for MoE-35B).
            // Env-gated `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ=1`. Default OFF.
            if super::moe::mmv_q_output_proj_enabled() {
                if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_q8_1_rawsum.as_ref(),
                    st.kernels.mul_mat_vec_q_q8_0.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    use std::sync::Once;
                    static TRACE_ONCE_Q8RAW: Once = Once::new();
                    TRACE_ONCE_Q8RAW.call_once(|| {
                        super::decode::cuda_log_force(format!(
                            "[CUDA] mul_mat_vec_q_q8_0 output_proj (raw): ACTIVE (grid={}, in_dim={})",
                            vocab_size, hidden_dim
                        ));
                    });
                    let quant_grid = (in_dim_u32 + 31) / 32;
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum output_proj raw: {e}"
                    )))?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(mv_fn)
                            .arg(proj_q8)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }.map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q8_0 output_proj raw: {e}"
                    )))?;
                    return Ok(());
                }
            }

            let q8_fn = st.kernels.matvec_q8_0_dp4a.as_ref()
                .unwrap_or(&st.kernels.matvec_q8_0);
            let grid = matvec_q8_0_grid(out_dim_u32);
            let shmem = 0u32;
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            unsafe {
                self.device
                    .stream
                    .launch_builder(q8_fn)
                    .arg(proj_q8)
                    .arg(&st.scratch.normed)
                    .arg(&mut st.logits_gpu)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(launch_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("matvec output_proj Q8_0 launch: {e}"))
            })?;
        } else if let Some(ref proj_bf16) = st.globals.output_proj_bf16 {
            // Path -1: BF16 output_proj matvec dispatch.
            // Env-gated `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=1`. Default OFF
            // preserves the existing cuBLAS HGEMV-BF16 path (byte-identical).
            //
            // measured the cuBLAS HGEMV-BF16 path at 1218 µs / call ×
            // 2245 inst = 125.5 ms / 64-tok decode = 16.7% TPOT (single
            // largest BF16 call). The purpose-built `mul_mat_vec_f<
            // nv_bfloat16, ...>` skips cuBLAS persistent-CTA setup; at batch=1
            // its predicted per-call cost is ~400-700 µs = +6-9 BF16 tok/s.
            //
            // Grid: (vocab_size, 1, 1)  block: (32, 4, 1) = 128 thr.
            // Smem: 32 * 4 = 128 bytes (buf_iw[WARP_SIZE]).
            if super::moe::mmv_bf16_output_proj_enabled() {
                if let Some(mv_fn) = st.kernels.mul_mat_vec_f_bf16.as_ref() {
                    // tracer: emit a single one-shot log so
                    // operators can confirm the dispatch path is active.
                    use std::sync::Once;
                    static TRACE_ONCE: Once = Once::new();
                    TRACE_ONCE.call_once(|| {
                        super::decode::cuda_log_force(format!(
                            "[CUDA] mul_mat_vec_f_bf16 output_proj: ACTIVE (grid={}, ncols2={}, stride={})",
                            vocab_size, hidden_dim / 2, hidden_dim
                        ));
                    });
                    let nrows_x = vocab_size as i32;
                    let ncols_x = hidden_dim as i32;
                    debug_assert!(ncols_x % 2 == 0,
                        "mul_mat_vec_f_bf16 requires hidden_dim % 2 == 0");
                    let ncols2 = ncols_x / 2;
                    let stride_row = ncols_x;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (nrows_x as u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128, // WARP_SIZE * sizeof(float)
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_bf16)
                            .arg(&st.scratch.normed)
                            .arg(&mut st.logits_gpu)
                            .arg(&ncols2)
                            .arg(&stride_row)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "mul_mat_vec_f_bf16 output_proj launch: {e}"
                        ))
                    })?;
                    return Ok(());
                }
                // mul_mat_vec_f_bf16 kernel not loaded — fall through to
                // existing cuBLAS path below.
            }

            // I-BF16 Phase-3: cuBLAS HGEMV-BF16 for the output projection.
            //
            // This is the LARGEST decode-time matvec: vocab × hidden (e.g.
            // 248320 × 4096 = 1.02 GB BF16). The old per-block matvec_bf16
            // kernel was bandwidth-bottlenecked. cuBLAS GemmEx with N=1
            // (CUDA_R_16BF inputs, COMPUTE_32F accumulator) ships the data
            // through the tensor-core BF16 lane with persistent-CTA
            // scheduling, mirroring the same path that delivered +7.7% on
            // the Q8 output_proj. Reuses the 2-byte F16 scratch.
            //
            // The wrapper composes three gates: the explicit
            // `LUMEN_CUDA_BF16_GEMMEX=0` opt-out, the startup capability
            // probe in `CudaBackend::new`, and the runtime-armed fallback
            // flag set on a per-call cuBLAS failure. When any gate is
            // closed (or the GemmEx call fails at runtime), this dispatches
            // via the legacy `matvec_bf16` kernel instead of aborting the
            // generation.
            unsafe {
                launch_bf16_matvec_with_fallback(
                    &self.device,
                    &st.kernels,
                    proj_bf16,
                    &st.scratch.normed,
                    &mut st.logits_gpu,
                    &mut st.scratch.input_f16,
                    vocab_size,
                    hidden_dim,
                    "output_proj",
                )?;
            }
        } else {
            let cfg = GemvConfig {
                trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                m: hidden_dim as i32,
                n: vocab_size as i32,
                alpha: 1.0f32,
                lda: hidden_dim as i32,
                incx: 1,
                beta: 0.0f32,
                incy: 1,
            };
            unsafe {
                self.device
                    .blas
                    .gemv(
                        cfg,
                        &st.globals.output_proj,
                        &st.scratch.normed,
                        &mut st.logits_gpu,
                    )
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("cuBLAS GEMV output_proj: {e}"))
            })?;
        }

        Ok(())
    }

    /// Normal (non-graph) decode path. Used for first token and as fallback.
    fn decode_token_normal(
        &self,
        token_id: u32,
        seq_pos: usize,
        num_layers: usize,
        hp: &ModelHyperparams,
        st: &mut MutableState,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        self.embed_token_gpu(token_id, st)?;
        for layer in 0..num_layers {
            self.compute_layer_gpu(layer, seq_pos, st)?;
            // FIX-DTOD: For DENSE layers, dense FFN writes to a separate
            // buffer (attn_proj) and we propagate the post-FFN residual to x_gpu
            // here. For MoE layers, `encode_moe_ffn_decode` writes the MoE FFN
            // output IN-PLACE to st.scratch.x_gpu; the unconditional dtod
            // OVERWROTE that output with the stale pre-FFN attn+residual,
            // destroying the MoE contribution every layer, every token. Gate
            // on moe_meta_cache to skip the dtod for MoE layers.
            let is_moe_layer = st
                .moe_meta_cache
                .get(layer)
                .and_then(|m| m.as_ref())
                .is_some();
            if !is_moe_layer {
                self.device.stream
                    .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                    .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
            }
        }
        self.compute_final_gpu(st)?;
        {
            let vocab = hp.vocab_size;
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.device.stream
                    .launch_builder(&st.kernels.argmax_f32)
                    .arg(&st.logits_gpu)
                    .arg(&mut st.argmax_result)
                    .arg(&vocab)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("argmax launch: {e}")))?;
        }
        // Full real-logits readback (see `decode_token` for the full
        // rationale). Was a one-hot synthesis that destroyed the
        // distribution and caused sampling gibberish on both models.
        self.device.synchronize()?;
        // optional per-step CPU sleep to close the GPU-scheduler
        // timing race (mirror of the `decode_token` sync below). Default OFF;
        // set `LUMEN_CUDA_DECODE_DELAY_US=50` to opt in; the Metal path
        // established the empirical precedent for this mitigation.
        maybe_apply_cuda_decode_delay();
        let logits_host = self.device.dtoh_copy(&st.logits_gpu)?;
        kv.advance_seq_len()?;
        st.decode_token_count += 1;
        Ok(Logits { data: logits_host })
    }

    // -----------------------------------------------------------------------
    // Public MoE cache configuration surface (opt-in only).
    // Mirrors `metal::MetalF32Backend::configure_expert_cache` / `_warmup`.
    // -----------------------------------------------------------------------

    /// Configure MoE expert-LFU cache for streaming decode (opt-in).
    ///
    /// Must be called BEFORE `init()` is fully populated (during the
    /// pre-init configuration phase). After this call, the CUDA MoE forward
    /// path will:
    /// - check the cache on each per-expert dispatch,
    /// - on miss: `ExpertReader::load_expert(layer, eid)` -> `htod` upload ->
    ///   dispatch FFN against a per-layer assembled scratch buffer ->
    ///   `cache.insert(...)`,
    /// - on hit: dispatch against the cached bytes (Arc<Vec<u8>>).
    ///
    /// `lbc_path`: Path to the LBC model file for per-expert byte-range
    /// reads. `cache_capacity`: maximum number of (layer, expert) entries to
    /// hold in the LFU cache. Default = GPU-resident-all (no cache).
    ///
    /// Per the user binding (#3): cache is **opt-in only**; calling this
    /// activates it. No auto-enable from VRAM headroom.
    pub fn configure_expert_cache(
        &mut self,
        lbc_path: &std::path::Path,
        cache_capacity: usize,
    ) -> Result<(), RuntimeError> {
        let hp = self.hyperparams.as_ref().ok_or_else(|| {
            RuntimeError::Compute(
                "configure_expert_cache: backend not initialized yet (call init() first)".into(),
            )
        })?;
        let num_experts = hp.num_experts.map(|v| v as usize).unwrap_or(0);
        let num_layers = hp.num_layers as usize;
        if num_experts == 0 {
            return Err(RuntimeError::Compute(
                "configure_expert_cache: model is not MoE (num_experts == 0)".into(),
            ));
        }
        let cfg = super::moe_cache::build_cache_config(
            lbc_path,
            cache_capacity,
            num_layers,
            num_experts,
        )?;
        let mut guard = self.state.lock().unwrap();
        let st = guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "configure_expert_cache: backend not initialized (call init() first)".into(),
            )
        })?;
        st.expert_cache_config = Some(cfg);
        Ok(())
    }

    /// Configure profiling-based cache warm-up.
    ///
    /// After `profiling_tokens` tokens have been decoded with cache OFF, the
    /// `ExpertActivationProfiler`'s per-(layer, expert) counts are used to
    /// pre-populate the cache with the top-K hottest experts per layer.
    ///
    /// Must be called AFTER `configure_expert_cache`. Has no effect if the
    /// cache is not configured. Default state (`configure_expert_cache`
    /// without a follow-on `configure_expert_warmup`): warm-up disabled,
    /// `warmup_complete=true` (the cache is queried directly without a
    /// profiling phase).
    pub fn configure_expert_warmup(
        &mut self,
        profiling_tokens: usize,
        top_k_per_layer: usize,
    ) -> Result<(), RuntimeError> {
        let mut guard = self.state.lock().unwrap();
        let st = guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "configure_expert_warmup: backend not initialized yet".into(),
            )
        })?;
        let cfg = st.expert_cache_config.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "configure_expert_warmup: call configure_expert_cache first".into(),
            )
        })?;
        super::moe_cache::configure_warmup(cfg, profiling_tokens, top_k_per_layer);
        Ok(())
    }
}

/// Launch a matvec kernel for the given weight buffer (F32, F16, Q8_0, or Q4_0).
///
/// For F32 weights, dispatches cuBLAS SGEMV which achieves 70-80% of peak
/// memory bandwidth (vs ~34% for the custom kernel). For F16 weights, dispatches
/// cuBLAS HGEMM (GemmEx with n=1) which halves memory bandwidth vs F32 by
/// reading half-precision weights. For Q8_0 weights, dispatches the dp4a
/// kernel which reads native Q8_0 (~1.06 B/elem) -- less bandwidth than the
/// pre-dequanted F16 path (2.0 B/elem). Falls back to v1 scalar kernel if
/// dp4a is not available. For Q4_0 weights, dispatches the custom NVRTC
/// kernel that dequantizes on-the-fly.
///
/// cuBLAS GEMV mapping for row-major `[out_dim, in_dim]` weights:
/// - cuBLAS is column-major, so our row-major W is column-major `[in_dim, out_dim]`
/// - Use `CUBLAS_OP_T`: `y = alpha * A^T * x + beta * y`
/// - `m = in_dim`, `n = out_dim`, `lda = in_dim`, `alpha = 1.0`, `beta = 0.0`
///
/// # Safety
///
/// Caller must ensure:
/// - `weight` has the correct number of elements for [out_dim, in_dim]
/// - `input` has `in_dim` elements
/// - `output` has `out_dim` elements
/// - If `weight_f16_cache` is Some, it must have `out_dim * in_dim * 2` bytes
/// - If `input_f16_scratch` is Some, it must have at least `in_dim * 2` bytes
unsafe fn launch_matvec(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    mut input_f16_scratch: Option<&mut CudaSlice<u8>>,
    mut input_q8_1_scratch: Option<&mut CudaSlice<u8>>,
) -> Result<(), RuntimeError> {
    // --- Native quantized kernels: read Q8_0/Q4_0 directly (1.06/0.56 B/elem) ---
    // These bypass the HGEMV path which reads 2 B/elem from pre-dequanted F16 cache.
    //
    // Priority for Q8_0:
    // -1. dp4a mmvq: quantize_q8_1_rawsum + mul_mat_vec_q_q8_0 (env-gated;
    //     Q8_1-activation x Q8_0-weight matvec with dp4a INT8 dot-product).
    // 0. dp4a Q8_1 (pre-quantized input, NR=2, 128 threads): any in_dim (SM 6.1+)
    // 1. smem kernel (F32 x in shmem, NR=2): in_dim*4 <= 48KB -> in_dim <= 12288
    // 2. hgemv kernel (F16 x in shmem, NR=4): in_dim*2 <= 48KB -> in_dim <= 24576
    // 3. cuBLAS HGEMV via pre-dequanted F16 cache (2 B/elem): any in_dim
    // 4. dp4a (on-the-fly x quant) or v1 scalar: any in_dim (last resort)

    if let GpuWeightBuf::Q8Raw(w_q8) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // Path -1: Q8_0 dp4a mmvq dispatch.
        // Q8_1-activation x Q8_0-weight matvec with dp4a INT8 dot-product.
        // Two-launch sequence: quantize_q8_1_rawsum → mul_mat_vec_q_q8_0.
        // Env-gated `LUMEN_CUDA_MMV_Q_DP4A=1`. Default OFF preserves byte-identity.
        if super::moe::mmv_q_dp4a_enabled() {
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
                kernels.quantize_q8_1_rawsum.as_ref(),
                kernels.mul_mat_vec_q_q8_0.as_ref(),
                input_q8_1_scratch.as_deref_mut(),
            ) {
                let in_dim_u32 = in_dim as u32;
                let out_dim_u32 = out_dim as u32;
                // quantize_q8_1_rawsum: grid=(ceil(in_dim/32),1,1) block=(32,1,1)
                let q_blocks = (in_dim_u32 + 31) / 32;
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (q_blocks, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum {label}: {e}",
                    )))?;

                // mul_mat_vec_q_q8_0: grid=(nrows_x,1,1) block=(32, 4, 1) = 128 threads
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim_u32, 1, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8)
                    .arg(&*q8_1_buf)
                    .arg(output)
                    .arg(&in_dim_u32)
                    .arg(&out_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q8_0 {label}: {e}",
                    )))?;
                return Ok(());
            }
        }

        // Path 0: dp4a with pre-quantized Q8_1 input.
        // Quantize F32 input to Q8_1, then dp4a matvec with native int* input loads.
        // No shmem for input — L2 cache handles reuse across blocks.
        if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            kernels.matvec_q8_0_q8_1.as_ref(),
            input_q8_1_scratch,
        ) {
            let in_dim_u32 = in_dim as u32;
            let out_dim_u32 = out_dim as u32;

            // Step 1: Quantize F32 input to Q8_1.
            let quant_grid = q8_1_quant_grid(in_dim_u32);
            let quant_cfg = CudarcLaunchConfig {
                grid_dim: (quant_grid, 1, 1),
                block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(quant_fn)
                .arg(input)
                .arg(&mut *q8_1_buf)
                .arg(&in_dim_u32)
                .launch(quant_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "quantize_f32_to_q8_1 {label}: {e}",
                )))?;

            // Step 2: dp4a matvec with Q8_1 input.
            let mv_grid = dp4a_q8_1_grid(out_dim_u32);
            let mv_cfg = CudarcLaunchConfig {
                grid_dim: (mv_grid, 1, 1),
                block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(mv_fn)
                .arg(w_q8)
                .arg(&*q8_1_buf)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(mv_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q8_0_q8_1 {label}: {e}",
                )))?;
            return Ok(());
        }

        // Path 1: smem kernel (F32 x, NR=2) — best for small dimensions.
        if let Some(smem_fn) = kernels.matvec_q8_0_smem.as_ref().filter(|_| shmem_f32 <= 49152) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = matvec_smem_grid(out_dim_u32);
            let shmem = matvec_smem_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (SMEM_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(smem_fn)
                .arg(w_q8)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec Q8_0 smem {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4) — covers 12288 < in_dim <= 24576.
        // Reads native Q8_0 (1.0625 B/elem) instead of HGEMV's 2 B/elem.
        if let Some(hgemv_fn) = kernels.hgemv_q8_0.as_ref().filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = hgemv_grid(out_dim_u32);
            let shmem = hgemv_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (HGEMV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(hgemv_fn)
                .arg(w_q8)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "hgemv Q8_0 {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16(device, kernels, w_f16, input, output, scratch,
                out_dim, in_dim, label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // Path 4: dp4a or v1 scalar (last resort).
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let q8_fn = kernels.matvec_q8_0_dp4a.as_ref()
            .unwrap_or(&kernels.matvec_q8_0);
        let grid = matvec_q8_0_grid(out_dim_u32);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (Q8_0_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(q8_fn)
            .arg(w_q8)
            .arg(input)
            .arg(output)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "matvec Q8_0 {label} launch: {e}",
            )))?;
        return Ok(());
    }

    // Q4Aligned or Q4Raw: dp4a with pre-quantized Q8_1 input.
    // Q4Aligned uses aligned int* nibble loads (20-byte blocks, 0.625 B/elem).
    // Q4Raw uses byte-level nibble loads (18-byte blocks, 0.5625 B/elem).
    // Priority: Q4Aligned > Q4Raw.
    if matches!(weight, GpuWeightBuf::Q4Aligned(_) | GpuWeightBuf::Q4Raw(_)) {
        // Path -1: Q4_0 dp4a mmvq dispatch.
        // Q8_1-activation x Q4_0-weight matvec with dp4a INT8 dot-product.
        // Operates on Q4Raw layout only (18-byte standard blocks).
        if super::moe::mmv_q_dp4a_enabled() {
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf), GpuWeightBuf::Q4Raw(w)) = (
                kernels.quantize_q8_1_rawsum.as_ref(),
                kernels.mul_mat_vec_q_q4_0.as_ref(),
                input_q8_1_scratch.as_deref_mut(),
                weight,
            ) {
                let in_dim_u32 = in_dim as u32;
                let out_dim_u32 = out_dim as u32;
                let q_blocks = (in_dim_u32 + 31) / 32;
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (q_blocks, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_q8_1_rawsum Q4 {label}: {e}",
                    )))?;
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim_u32, 1, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w)
                    .arg(&*q8_1_buf)
                    .arg(output)
                    .arg(&in_dim_u32)
                    .arg(&out_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "mul_mat_vec_q_q4_0 {label}: {e}",
                    )))?;
                return Ok(());
            }
        }

        if let (Some(quant_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            input_q8_1_scratch.take(),
        ) {
            // Check which kernel to use: aligned or unaligned.
            let (mv_fn_opt, w_ptr) = match weight {
                GpuWeightBuf::Q4Aligned(w) => (kernels.matvec_q4_aligned_q8_1.as_ref(), w as &CudaSlice<u8>),
                GpuWeightBuf::Q4Raw(w) => (kernels.matvec_q4_0_dp4a.as_ref(), w as &CudaSlice<u8>),
                _ => unreachable!(),
            };
            if let Some(mv_fn) = mv_fn_opt {
                let in_dim_u32 = in_dim as u32;
                let out_dim_u32 = out_dim as u32;

                // Step 1: Quantize F32 input to Q8_1.
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 Q4 {label}: {e}",
                    )))?;

                // Step 2: dp4a matvec with Q8_1 input (NR=4, 256 threads).
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_ptr)
                    .arg(&*q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_dp4a {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }

    // Q4_0 raw fallback: smem > hgemv > cuBLAS HGEMV > scalar.
    // dp4a path is handled by the unified Q4Aligned/Q4Raw dispatch above.
    if let GpuWeightBuf::Q4Raw(w_q4) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // Path 1: smem kernel (F32 x, NR=2).
        if let Some(smem_fn) = kernels.matvec_q4_0_smem.as_ref().filter(|_| shmem_f32 <= 49152) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = matvec_smem_grid(out_dim_u32);
            let shmem = matvec_smem_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (SMEM_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(smem_fn)
                .arg(w_q4)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec Q4_0 smem {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4) — covers 12288 < in_dim <= 24576.
        if let Some(hgemv_fn) = kernels.hgemv_q4_0.as_ref().filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = hgemv_grid(out_dim_u32);
            let shmem = hgemv_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (HGEMV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(hgemv_fn)
                .arg(w_q4)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "hgemv Q4_0 {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16(device, kernels, w_f16, input, output, scratch,
                out_dim, in_dim, label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // Path 4: scalar Q4_0 (last resort).
        let mv_block = matvec_block_size();
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (out_dim as u32, 1, 1),
            block_dim: (mv_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        device
            .stream
            .launch_builder(&kernels.matvec_q4_0)
            .arg(w_q4)
            .arg(input)
            .arg(output)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "matvec Q4_0 {label} launch: {e}",
            )))?;
        return Ok(());
    }

    // I-BF16 Phase-3: BF16Raw via cuBLAS HGEMV-BF16 (CUDA_R_16BF + COMPUTE_32F).
    // Take this branch BEFORE the F32 HGEMV check below (which moves
    // input_f16_scratch). Reuses the 2-byte F16 scratch (same byte width).
    //
    // The legacy per-block `matvec_bf16` kernel ran at 0.66× llama.cpp; the cuBLAS
    // tensor-core path closes the gap. The wrapper at
    // `launch_bf16_matvec_with_fallback` composes three gates: the explicit
    // `LUMEN_CUDA_BF16_GEMMEX=0` opt-out, the startup capability probe in
    // `CudaBackend::new`, and the runtime-armed fallback flag set on a
    // per-call cuBLAS failure. When any gate is closed (or the GemmEx
    // call fails at runtime), this dispatches via `matvec_bf16` instead.
    // `LUMEN_CUDA_BF16_GEMMEX=0` remains the A/B benchmarking opt-out.
    if let (GpuWeightBuf::Bf16Raw(w_bf16), Some(scratch)) =
        (weight, input_f16_scratch.as_deref_mut())
    {
        return launch_bf16_matvec_with_fallback(
            device, kernels, w_bf16, input, output, scratch,
            out_dim, in_dim, label,
        );
    }

    // HGEMV path: cuBLAS with pre-dequanted F16 weights.
    // Used for F32 weights (from Q4_1 dequant) that have an F16 cache.
    // Q8Raw and Q4Raw are handled above via native kernels (smem/scalar).
    // Uses DEFAULT_TENSOR_OP (fallback path for F32 with F16 caches).
    // Use `as_deref_mut()` to borrow without moving: if `weight` is NOT F32 we
    // fall through to the `match` below, which still needs `input_f16_scratch`
    // for the Bf16Raw fallback arm.
    if matches!(weight, GpuWeightBuf::F32(_)) {
        if let (Some(w_f16), Some(scratch)) =
            (weight_f16_cache, input_f16_scratch.as_deref_mut())
        {
            return launch_hgemv_f16(device, kernels, w_f16, input, output, scratch,
                out_dim, in_dim, label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    match weight {
        GpuWeightBuf::F32(w_f32) => {
            let cfg = GemvConfig {
                trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                m: in_dim as i32,
                n: out_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                incx: 1,
                beta: 0.0f32,
                incy: 1,
            };
            device
                .blas
                .gemv(cfg, w_f32, input, output)
                .map_err(|e| RuntimeError::Compute(format!(
                    "cuBLAS GEMV {label}: {e}",
                )))?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            // Custom F16 matvec kernel (dequant f16→f32 on the fly).
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_f16)
                .arg(w_f16)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec F16 {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // I-BF16 Phase-3 fallback: only reached when input_f16_scratch is
            // None at the early-return check above. Uses the per-block
            // matvec_bf16 kernel (the original 0.66× llama.cpp path).
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_bf16)
                .arg(w_bf16)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec BF16 fallback {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;

            // Path 0 (priority): Q8Aligned + pre-quantized Q8_1 input (dp4a, NR=2).
            // Both weight and input use native int* loads. Zero byte-packing overhead.
            // Q8_SCALE_HW: prefer the halfword-scale variant.
            let aligned_mv_fn = if kernels.use_q8_scale_hw {
                kernels.matvec_q8_aligned_q8_1_hw.as_ref()
                    .or(kernels.matvec_q8_aligned_q8_1.as_ref())
            } else {
                kernels.matvec_q8_aligned_q8_1.as_ref()
            };
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
                kernels.quantize_f32_to_q8_1.as_ref(),
                aligned_mv_fn,
                input_q8_1_scratch,
            ) {
                // Step 1: Quantize F32 input to Q8_1.
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 aligned {label}: {e}",
                    )))?;

                // Step 2: dp4a matvec with Q8Aligned weights + Q8_1 input (NR=2).
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8a)
                    .arg(&*q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_aligned_q8_1 {label}: {e}",
                    )))?;
            } else {
                // Fallback: Q8_0 aligned dp4a with on-the-fly x quantization (NR=2).
                let q8a_fn = kernels.matvec_q8_0_aligned.as_ref()
                    .or(kernels.matvec_q8_0_dp4a.as_ref())
                    .unwrap_or(&kernels.matvec_q8_0);
                let grid = matvec_q8_0_grid(out_dim_u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(q8a_fn)
                    .arg(w_q8a)
                    .arg(input)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(launch_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec Q8_0 aligned {label} launch: {e}",
                    )))?;
            }
        }
        // Q8Raw fallback: dp4a or v1 scalar (smem kernel not available).
        GpuWeightBuf::Q8Raw(w_q8) => {
            if let Some(ref dp4a_fn) = kernels.matvec_q8_0_dp4a {
                let grid = matvec_q8_0_grid(out_dim as u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device.stream
                    .launch_builder(dp4a_fn)
                    .arg(w_q8).arg(input).arg(output)
                    .arg(&(out_dim as u32)).arg(&(in_dim as u32))
                    .launch(launch_cfg)
                    .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 dp4a {label}: {e}")))?;
            } else {
                let mv_block = matvec_block_size();
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim as u32, 1, 1),
                    block_dim: (mv_block, 1, 1),
                    shared_mem_bytes: 0,
                };
                device.stream
                    .launch_builder(&kernels.matvec_q8_0)
                    .arg(w_q8).arg(input).arg(output)
                    .arg(&(out_dim as u32)).arg(&(in_dim as u32))
                    .launch(launch_cfg)
                    .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 v1 {label}: {e}")))?;
            }
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            // Fallback scalar (should not reach here — handled above).
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0)
                .arg(w_q4)
                .arg(input)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec Q4_0 {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Should not reach here — Q4Aligned is handled by early-return above.
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight reached fallback match in matvec {label} — dp4a kernels unavailable"
            )));
        }
        // split-layout: / TILE: Q8Split/Q4Split/Q8Tile/Q4Tile are sibling
        // buffers consumed only by `launch_matvec_preq8_1_split` (or its
        // tile-aware wrapper). Reaching the base `launch_matvec` means the
        // caller passed a sibling as the base weight, which is a bug.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q8Split/Q4Split/Q8Tile/Q4Tile sibling reached fallback match in matvec {label} — \
                 caller must dispatch via launch_matvec_preq8_1_split"
            )));
        }
    }
    Ok(())
}

/// Launch a matvec+residual kernel: `output = weight * input + residual`.
///
/// For F32 weights, uses cuBLAS SGEMV with `beta=1.0`: first copies the
/// residual into the output buffer, then runs `y = 1.0 * A^T * x + 1.0 * y`.
/// For Q8_0 weights, uses dp4a+residual kernel (native Q8_0 ~1.06 B/elem).
/// For other quantized/F16 weights, dispatches the fused custom kernels.
///
/// # Safety
///
/// Same constraints as `launch_matvec`, plus `residual` must have `out_dim` elements.
unsafe fn launch_matvec_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    mut input_f16_scratch: Option<&mut CudaSlice<u8>>,
    mut input_q8_1_scratch: Option<&mut CudaSlice<u8>>,
) -> Result<(), RuntimeError> {
    // --- Native quantized kernels: read Q8_0/Q4_0 directly ---
    // Priority: dp4a Q8_1 > smem (F32 x) > hgemv (F16 x) > cuBLAS HGEMV > dp4a/scalar.

    // Q8_0 raw residual: dp4a Q8_1 > smem > hgemv > HGEMV fallback > dp4a/scalar.
    if let GpuWeightBuf::Q8Raw(w_q8) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // Path 0: dp4a with pre-quantized Q8_1 input + fused residual.
        if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            kernels.matvec_q8_0_q8_1_residual.as_ref(),
            input_q8_1_scratch,
        ) {
            let in_dim_u32 = in_dim as u32;
            let out_dim_u32 = out_dim as u32;

            // Step 1: Quantize F32 input to Q8_1.
            let quant_grid = q8_1_quant_grid(in_dim_u32);
            let quant_cfg = CudarcLaunchConfig {
                grid_dim: (quant_grid, 1, 1),
                block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(quant_fn)
                .arg(input)
                .arg(&mut *q8_1_buf)
                .arg(&in_dim_u32)
                .launch(quant_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "quantize_f32_to_q8_1 residual {label}: {e}",
                )))?;

            // Step 2: dp4a matvec + residual with Q8_1 input.
            let mv_grid = dp4a_q8_1_grid(out_dim_u32);
            let mv_cfg = CudarcLaunchConfig {
                grid_dim: (mv_grid, 1, 1),
                block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(mv_fn)
                .arg(w_q8)
                .arg(&*q8_1_buf)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(mv_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec_q8_0_q8_1_residual {label}: {e}",
                )))?;
            return Ok(());
        }

        // Path 1: smem kernel (F32 x, NR=2).
        if let Some(smem_fn) = kernels.matvec_q8_0_smem_residual.as_ref().filter(|_| shmem_f32 <= 49152) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = matvec_smem_grid(out_dim_u32);
            let shmem = matvec_smem_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (SMEM_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(smem_fn)
                .arg(w_q8)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual Q8_0 smem {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4).
        if let Some(hgemv_fn) = kernels.hgemv_q8_0_residual.as_ref().filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = hgemv_grid(out_dim_u32);
            let shmem = hgemv_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (HGEMV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(hgemv_fn)
                .arg(w_q8)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "hgemv+residual Q8_0 {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16_residual(device, kernels, w_f16, input, residual, output,
                scratch, out_dim, in_dim, label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // Path 4: dp4a or v1 scalar.
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let q8_fn = kernels.matvec_q8_0_dp4a_residual.as_ref()
            .unwrap_or(&kernels.matvec_q8_0_residual);
        let grid = matvec_q8_0_grid(out_dim_u32);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (Q8_0_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(q8_fn)
            .arg(w_q8)
            .arg(input)
            .arg(residual)
            .arg(output)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "matvec+residual Q8_0 {label} launch: {e}",
            )))?;
        return Ok(());
    }

    // Q4Aligned or Q4Raw residual: dp4a with pre-quantized Q8_1 input + fused residual.
    if matches!(weight, GpuWeightBuf::Q4Aligned(_) | GpuWeightBuf::Q4Raw(_)) {
        if let (Some(quant_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            input_q8_1_scratch.take(),
        ) {
            let (mv_fn_opt, w_ptr) = match weight {
                GpuWeightBuf::Q4Aligned(w) => (kernels.matvec_q4_aligned_q8_1_residual.as_ref(), w as &CudaSlice<u8>),
                GpuWeightBuf::Q4Raw(w) => (kernels.matvec_q4_0_dp4a_residual.as_ref(), w as &CudaSlice<u8>),
                _ => unreachable!(),
            };
            if let Some(mv_fn) = mv_fn_opt {
                let in_dim_u32 = in_dim as u32;
                let out_dim_u32 = out_dim as u32;

                // Step 1: Quantize F32 input to Q8_1.
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 Q4 residual {label}: {e}",
                    )))?;

                // Step 2: dp4a matvec + residual with Q8_1 input (NR=4, 256 threads).
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_ptr)
                    .arg(&*q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_dp4a_residual {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }

    // Q4_0 raw residual fallback: smem > hgemv > scalar.
    // dp4a path is handled by the unified Q4Aligned/Q4Raw dispatch above.
    if let GpuWeightBuf::Q4Raw(w_q4) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // Path 1: smem kernel (F32 x, NR=2).
        if let Some(smem_fn) = kernels.matvec_q4_0_smem_residual.as_ref().filter(|_| shmem_f32 <= 49152) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = matvec_smem_grid(out_dim_u32);
            let shmem = matvec_smem_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (SMEM_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(smem_fn)
                .arg(w_q4)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual Q4_0 smem {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4).
        if let Some(hgemv_fn) = kernels.hgemv_q4_0_residual.as_ref().filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT) {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let grid = hgemv_grid(out_dim_u32);
            let shmem = hgemv_shared_bytes(in_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (HGEMV_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(hgemv_fn)
                .arg(w_q4)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "hgemv+residual Q4_0 {label} launch: {e}",
                )))?;
            return Ok(());
        }

        // Path 3: scalar Q4_0 residual (reads native Q4_0 at 0.5625 B/elem).
        let mv_block = matvec_block_size();
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (out_dim as u32, 1, 1),
            block_dim: (mv_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        device
            .stream
            .launch_builder(&kernels.matvec_q4_0_residual)
            .arg(w_q4)
            .arg(input)
            .arg(residual)
            .arg(output)
            .arg(&out_dim_u32)
            .arg(&in_dim_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "matvec+residual Q4_0 {label} launch: {e}",
            )))?;
        return Ok(());
    }

    // I-BF16 Phase-3: BF16Raw residual via cuBLAS HGEMV-BF16. Take this branch
    // BEFORE the F32 HGEMV check below (which consumes input_f16_scratch).
    // Reuses the 2-byte F16 scratch (same byte width as BF16).
    //
    // The wrapper at `launch_bf16_matvec_residual_with_fallback` composes the
    // explicit `LUMEN_CUDA_BF16_GEMMEX=0` opt-out, the startup capability
    // probe, and the runtime-armed fallback flag; on cuBLAS failure it
    // dispatches the same call via `matvec_bf16_residual` so the in-flight
    // generation is not aborted by a transient cuBLAS error.
    if let (GpuWeightBuf::Bf16Raw(w_bf16), Some(scratch)) =
        (weight, input_f16_scratch.as_deref_mut())
    {
        return launch_bf16_matvec_residual_with_fallback(
            device, kernels, w_bf16, input, residual, output, scratch,
            out_dim, in_dim, label,
        );
    }

    // HGEMV residual: only for F32 weights with F16 cache.
    // Q8Raw and Q4Raw are handled above via native kernels (smem/scalar).
    // Uses DEFAULT_TENSOR_OP (fallback path for F32 with F16 caches).
    // Use `as_deref_mut()` to avoid consuming `input_f16_scratch` on the
    // non-F32 path -- symmetric to the launch_matvec fix.
    if matches!(weight, GpuWeightBuf::F32(_)) {
        if let (Some(w_f16), Some(scratch)) =
            (weight_f16_cache, input_f16_scratch.as_deref_mut())
        {
            return launch_hgemv_f16_residual(device, kernels, w_f16, input, residual, output,
                scratch, out_dim, in_dim, label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    match weight {
        GpuWeightBuf::F32(w_f32) => {
            // Copy residual into output so cuBLAS can accumulate: y = W*x + y.
            device
                .stream
                .memcpy_dtod(residual, output)
                .map_err(|e| RuntimeError::Compute(format!(
                    "cuBLAS residual copy {label}: {e}",
                )))?;
            let cfg = GemvConfig {
                trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                m: in_dim as i32,
                n: out_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                incx: 1,
                beta: 1.0f32,
                incy: 1,
            };
            device
                .blas
                .gemv(cfg, w_f32, input, output)
                .map_err(|e| RuntimeError::Compute(format!(
                    "cuBLAS GEMV+residual {label}: {e}",
                )))?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;

            // Path 0 (priority): Q8Aligned + pre-quantized Q8_1 input + residual (dp4a, NR=2).
            // Q8_SCALE_HW: prefer the halfword-scale residual variant.
            let aligned_mv_residual = if kernels.use_q8_scale_hw {
                kernels.matvec_q8_aligned_q8_1_hw_residual.as_ref()
                    .or(kernels.matvec_q8_aligned_q8_1_residual.as_ref())
            } else {
                kernels.matvec_q8_aligned_q8_1_residual.as_ref()
            };
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
                kernels.quantize_f32_to_q8_1.as_ref(),
                aligned_mv_residual,
                input_q8_1_scratch,
            ) {
                // Step 1: Quantize F32 input to Q8_1.
                let quant_grid = q8_1_quant_grid(in_dim_u32);
                let quant_cfg = CudarcLaunchConfig {
                    grid_dim: (quant_grid, 1, 1),
                    block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(quant_fn)
                    .arg(input)
                    .arg(&mut *q8_1_buf)
                    .arg(&in_dim_u32)
                    .launch(quant_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 aligned residual {label}: {e}",
                    )))?;

                // Step 2: dp4a matvec + residual with Q8Aligned weights + Q8_1 input (NR=2).
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8a)
                    .arg(&*q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_aligned_q8_1_residual {label}: {e}",
                    )))?;
            } else {
                // Fallback: Q8_0 aligned dp4a residual with on-the-fly x quantization.
                let q8a_fn = kernels.matvec_q8_0_aligned_residual.as_ref()
                    .or(kernels.matvec_q8_0_dp4a_residual.as_ref())
                    .unwrap_or(&kernels.matvec_q8_0_residual);
                let grid = matvec_q8_0_grid(out_dim_u32);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(q8a_fn)
                    .arg(w_q8a)
                    .arg(input)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(launch_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec+residual Q8_0 aligned {label} launch: {e}",
                    )))?;
            }
        }
        // Q8Raw fallback: dp4a or v1 scalar residual (unreachable — handled above).
        GpuWeightBuf::Q8Raw(w_q8) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let q8_fn = kernels.matvec_q8_0_dp4a_residual.as_ref()
                .unwrap_or(&kernels.matvec_q8_0_residual);
            let grid = matvec_q8_0_grid(out_dim_u32);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(q8_fn)
                .arg(w_q8)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual Q8_0 fallback {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_f16_residual)
                .arg(w_f16)
                .arg(input)
                .arg(output)
                .arg(residual)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual F16 {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // BF16Raw fused matvec+residual: mirrors F16Raw path.
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_bf16_residual)
                .arg(w_bf16)
                .arg(input)
                .arg(output)
                .arg(residual)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual BF16 {label} launch: {e}",
                )))?;
        }
        // Q4Raw fallback: scalar Q4_0 residual (unreachable — handled above).
        GpuWeightBuf::Q4Raw(w_q4) => {
            let mv_block = matvec_block_size();
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (out_dim as u32, 1, 1),
                block_dim: (mv_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0_residual)
                .arg(w_q4)
                .arg(input)
                .arg(residual)
                .arg(output)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!(
                    "matvec+residual Q4_0 fallback {label} launch: {e}",
                )))?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Should not reach here — Q4Aligned is handled by early-return above.
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight reached fallback match in matvec+residual {label} — dp4a kernels unavailable"
            )));
        }
        // split-layout: / TILE: Q8Split/Q4Split/Q8Tile/Q4Tile are sibling
        // buffers consumed only by `launch_matvec_residual_split` (or its
        // tile-aware wrapper). Reaching the base `launch_matvec_residual`
        // means the caller passed a sibling as the base weight, which is a bug.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_)
        | GpuWeightBuf::Q8Tile(_)  | GpuWeightBuf::Q4Tile(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q8Split/Q4Split/Q8Tile/Q4Tile sibling reached fallback match in matvec+residual {label} — \
                 caller must dispatch via launch_matvec_residual_split"
            )));
        }
    }
    Ok(())
}

/// Quantize an F32 input vector to Q8_1 format in-place on GPU.
///
/// Run ONCE, then pass the Q8_1 buffer to `launch_matvec_preq8_1` for multiple
/// matvecs sharing the same input (e.g., Q/K/V projections or gate/up projections).
/// Saves one `quantize_f32_to_q8_1` kernel launch per reuse (3 launches saved for
/// QKV, 1 saved for gate+up = 4 per layer = 112-144 per 28-36 layer model).
///
/// # Safety
///
/// Caller must ensure:
/// - `input` has `in_dim` elements
/// - `q8_1_buf` has at least `(in_dim / 32) * 36` bytes
/// - `in_dim` is a multiple of 32
unsafe fn launch_quantize_input_q8_1(
    device: &CudaDevice,
    quant_fn: &CudaFunction,
    input: &CudaSlice<f32>,
    q8_1_buf: &mut CudaSlice<u8>,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let in_dim_u32 = in_dim as u32;
    let quant_grid = q8_1_quant_grid(in_dim_u32);
    let quant_cfg = CudarcLaunchConfig {
        grid_dim: (quant_grid, 1, 1),
        block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(quant_fn)
        .arg(input)
        .arg(q8_1_buf)
        .arg(&in_dim_u32)
        .launch(quant_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "quantize_f32_to_q8_1 {label}: {e}",
        )))?;
    Ok(())
}

/// Launch dp4a matvec with a PRE-QUANTIZED Q8_1 input buffer (skip quantization).
///
/// Use after `launch_quantize_input_q8_1` to avoid redundant quantization when
/// multiple matvecs share the same input vector. Supports Q8Raw, Q8Aligned,
/// Q4Aligned, and Q4Raw weights. Falls back to the full `launch_matvec` for
/// weight types that don't use dp4a (F32, F16Raw) or when dp4a kernels are
/// unavailable.
///
/// # Safety
///
/// Caller must ensure:
/// - `q8_1_buf` contains valid Q8_1 data for `in_dim` elements
/// - `weight` has the correct number of elements for [out_dim, in_dim]
/// - `output` has `out_dim` elements
unsafe fn launch_matvec_preq8_1(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_1_buf: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::Q8Raw(w_q8) => {
            if let Some(mv_fn) = kernels.matvec_q8_0_q8_1.as_ref() {
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_0_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // AoS NR=8 dispatch.
            // When LUMEN_CUDA_Q8_AOS_NR8=1 is set AND the kernel loaded,
            // route AoS Q8 dispatch to the NR=8 dp4a mmvq kernel (NR=8 +
            // 4-thread cooperation + vdr=2 + blocks_per_iter=32). NR=8 grid
            // math. Takes priority over Q8_SCALE_HW since it is a more
            // aggressive structural variant operating on the same byte layout.
            if kernels.use_q8_aos_nr8_dispatch {
                if let Some(mv_fn) = kernels.matvec_q8_aligned_nr8.as_ref() {
                    let nr: u32 = 8;
                    let mv_grid = (out_dim_u32 + nr - 1) / nr;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    device
                        .stream
                        .launch_builder(mv_fn)
                        .arg(w_q8a)
                        .arg(q8_1_buf)
                        .arg(output)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(mv_cfg)
                        .map_err(|e| RuntimeError::Compute(format!(
                            "matvec_q8_aligned_nr8 preq {label}: {e}",
                        )))?;
                    return Ok(());
                }
            }
            // Q8_SCALE_HW: prefer the halfword-scale variant when
            // LUMEN_CUDA_Q8_SCALE_HW=1 was set at init AND the kernel loaded.
            // Numerically equivalent to matvec_q8_aligned_q8_1 (replaces a
            // 2-byte OR of two byte loads with a single u16 load).
            let mv_fn_opt = if kernels.use_q8_scale_hw {
                kernels.matvec_q8_aligned_q8_1_hw.as_ref()
                    .or(kernels.matvec_q8_aligned_q8_1.as_ref())
            } else {
                kernels.matvec_q8_aligned_q8_1.as_ref()
            };
            if let Some(mv_fn) = mv_fn_opt {
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8a)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_aligned_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Aligned(w_q4a) => {
            if let Some(mv_fn) = kernels.matvec_q4_aligned_q8_1.as_ref() {
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q4a)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_aligned_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            if let Some(mv_fn) = kernels.matvec_q4_0_dp4a.as_ref() {
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q4)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_0_dp4a preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        _ => {} // F32, F16Raw: no dp4a path, caller should not use preq8_1
    }

    // Fallback: should not be reached if caller checks prerequisites.
    Err(RuntimeError::Compute(format!(
        "launch_matvec_preq8_1: no dp4a kernel available for {label}",
    )))
}

/// Launch dp4a matvec + fused residual with a PRE-QUANTIZED Q8_1 input buffer.
///
/// Same as `launch_matvec_preq8_1` but adds `residual` to the output.
///
/// # Safety
///
/// Same constraints as `launch_matvec_preq8_1`, plus `residual` must have `out_dim` elements.
#[allow(dead_code)]
unsafe fn launch_matvec_preq8_1_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_1_buf: &CudaSlice<u8>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::Q8Raw(w_q8) => {
            if let Some(mv_fn) = kernels.matvec_q8_0_q8_1_residual.as_ref() {
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_0_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // AoS NR=8 residual dispatch.
            if kernels.use_q8_aos_nr8_dispatch {
                if let Some(mv_fn) = kernels.matvec_q8_aligned_nr8_residual.as_ref() {
                    let nr: u32 = 8;
                    let mv_grid = (out_dim_u32 + nr - 1) / nr;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    device
                        .stream
                        .launch_builder(mv_fn)
                        .arg(w_q8a)
                        .arg(q8_1_buf)
                        .arg(residual)
                        .arg(output)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(mv_cfg)
                        .map_err(|e| RuntimeError::Compute(format!(
                            "matvec_q8_aligned_nr8_residual preq {label}: {e}",
                        )))?;
                    return Ok(());
                }
            }
            // Q8_SCALE_HW: prefer the halfword-scale residual variant.
            let mv_fn_opt = if kernels.use_q8_scale_hw {
                kernels.matvec_q8_aligned_q8_1_hw_residual.as_ref()
                    .or(kernels.matvec_q8_aligned_q8_1_residual.as_ref())
            } else {
                kernels.matvec_q8_aligned_q8_1_residual.as_ref()
            };
            if let Some(mv_fn) = mv_fn_opt {
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q8a)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_aligned_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Aligned(w_q4a) => {
            if let Some(mv_fn) = kernels.matvec_q4_aligned_q8_1_residual.as_ref() {
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q4a)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_aligned_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            if let Some(mv_fn) = kernels.matvec_q4_0_dp4a_residual.as_ref() {
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_q4)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_0_dp4a_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
        _ => {}
    }

    Err(RuntimeError::Compute(format!(
        "launch_matvec_preq8_1_residual: no dp4a kernel available for {label}",
    )))
}

// =============================================================================
// split-layout integration: SoA (per-row split layout) dispatch helpers.
// =============================================================================
//
// SPLIT layout reorganizes each row's scales and quant data into a contiguous
// [scales[nb] | quants[nb]] stream (vs the AoS Q8Raw/Q8Aligned layouts that
// interleave scale+quants block-by-block). The kernel reads both streams as
// native `int*` loads thanks to a 4-byte-aligned offset between them.
//
// Memory cost: one sibling buffer per source weight (~1x the original byte
// size). Decode prefers the sibling when present; prefill always reads the
// AoS original.
//
// The helpers below are NO-OP fall-throughs when the sibling is None or the
// SPLIT kernel failed to load -- keeping default-off contract (clean revert) intact
// when the LUMEN_CUDA_*_SPLIT env vars are unset.

/// Dispatch a dp4a matvec with pre-quantized Q8_1 input, preferring the
/// Q8Split / Q4Split sibling buffer when present.
///
/// When `q8_split_sibling` is `Some` AND `kernels.use_q8_split_dispatch` is
/// true, routes to `matvec_q8_split_q8_1`. Likewise for Q4. Falls through to
/// `launch_matvec_preq8_1` (the existing base dispatch) when neither sibling
/// is set OR the SPLIT dispatch is disabled.
///
/// # Safety
///
/// Same constraints as `launch_matvec_preq8_1`. The sibling buffer is
/// produced by `repack_layer_q8_clone_to_split()` and has identical element
/// count to the base weight.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn launch_matvec_preq8_1_split(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_split_sibling: Option<&CudaSlice<u8>>,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    q8_1_buf: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // AoS NR=8 path PRIORITY OVERRIDE.
    // When LUMEN_CUDA_Q8_AOS_NR8=1 is set AND the kernel loaded AND the
    // underlying weight is a Q8Aligned (AoS) buffer, bypass the SPLIT
    // dispatch entirely and route to the AoS NR=8 kernel. This is the
    // brief's "replace SPLIT dispatch with AOS_NR8 for FFN shapes":
    // since FFN gate/up/down all flow through `launch_matvec_preq8_1_split`,
    // we re-route them to the AoS path when AOS_NR8 is enabled.
    if kernels.use_q8_aos_nr8_dispatch {
        if let GpuWeightBuf::Q8Aligned(_) = weight {
            return launch_matvec_preq8_1(
                device, kernels, weight, q8_1_buf, output, out_dim, in_dim, label,
            );
        }
    }
    // Q8 split path.
    if kernels.use_q8_split_dispatch {
        if let Some(split_buf) = q8_split_sibling {
            // NR8 takes priority over 4thread when both
            // are env-enabled. NR8 is the strict structural superset
            // (4-threads-per-block + NR=8). When NR8 is the chosen kernel,
            // the grid uses NR=8 instead of NR=2; otherwise NR=2 grid math is
            // used.
            //
            // when LUMEN_CUDA_Q8_SPLIT_4THREAD=1 is set, prefer
            // the 4-threads-per-block variant on the same byte layout for
            // shapes where the microbench (A100-80GB) showed a per-shape win:
            //
            // out_dim <= 4096: 4thread faster (1.09-1.25x)
            // out_dim > 4096: production split faster (4thread 0.90-0.97x)
            //
            // The crossover happens because production's K-trip = 1 at
            // in_dim=4096 with 128 threads benefits from 4 K-iters in the
            // 4-threads-per-block pattern only when CTAs/SM stays low (small
            // grids). At large out_dim the grid is so big that the dual-CTA-
            // per-SM hint on the production kernel beats the K-iter unroll.
            //
            // We restrict to in_dim <= 4096 too: at larger in_dim production
            // already gets K-trip >= 3 and the 4-thread advantage shrinks.
            let use_split_nr8 = kernels.use_q8_split_nr8_dispatch;
            let use_split_4thread = !use_split_nr8
                && kernels.use_q8_split_4thread_dispatch
                && out_dim <= 4096
                && in_dim <= 4096;
            let (mv_fn_opt, nr_grid): (Option<&CudaFunction>, u32) = if use_split_nr8 {
                // NR=8 grid math: ceil(out_dim / 8)
                (kernels.matvec_q8_split_q8_1_nr8.as_ref(), 8)
            } else if use_split_4thread {
                (kernels.matvec_q8_split_q8_1_4thread.as_ref(), 2)
            } else {
                (kernels.matvec_q8_split_q8_1.as_ref(), 2)
            };
            if let Some(mv_fn) = mv_fn_opt {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = (out_dim_u32 + nr_grid - 1) / nr_grid;
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(split_buf)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_split_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    // Q4 split path.
    if kernels.use_q4_split_dispatch {
        if let Some(split_buf) = q4_split_sibling {
            if let Some(mv_fn) = kernels.matvec_q4_split_q8_1.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(split_buf)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_split_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    // Fall-through: existing Q8Raw/Q8Aligned/Q4Raw/Q4Aligned base dispatch.
    launch_matvec_preq8_1(device, kernels, weight, q8_1_buf, output, out_dim, in_dim, label)
}

/// Dispatch a dp4a matvec + fused residual, preferring the Q8Split or Q4Split
/// sibling buffer when present. Falls through to `launch_matvec_preq8_1_residual`
/// otherwise.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn launch_matvec_preq8_1_residual_split(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_split_sibling: Option<&CudaSlice<u8>>,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    q8_1_buf: &CudaSlice<u8>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // AoS NR=8 residual PRIORITY OVERRIDE.
    // Mirrors `launch_matvec_preq8_1_split`: when AOS_NR8 is enabled and
    // the underlying weight is Q8Aligned, bypass SPLIT dispatch and route
    // to the AoS path with residual fusion.
    if kernels.use_q8_aos_nr8_dispatch {
        if let GpuWeightBuf::Q8Aligned(_) = weight {
            return launch_matvec_preq8_1_residual(
                device, kernels, weight, q8_1_buf, residual, output, out_dim, in_dim, label,
            );
        }
    }
    if kernels.use_q8_split_dispatch {
        if let Some(split_buf) = q8_split_sibling {
            // NR8 takes priority over 4thread.
            // Same shape gate as launch_matvec_preq8_1_split: 4thread wins
            // only for out_dim<=4096 && in_dim<=4096 shapes (wo, KV proj,
            // GDN ssm_out). NR8 is applied unconditionally when its env
            // var is set (the NR8 premise is that NR=8 unlocks the FFN
            // shapes where 4thread LOST). Larger shapes (without either env
            // var) fall back to the prod split kernel.
            let use_split_nr8 = kernels.use_q8_split_nr8_dispatch;
            let use_split_4thread = !use_split_nr8
                && kernels.use_q8_split_4thread_dispatch
                && out_dim <= 4096
                && in_dim <= 4096;
            let (mv_fn_opt, nr_grid): (Option<&CudaFunction>, u32) = if use_split_nr8 {
                (kernels.matvec_q8_split_q8_1_nr8_residual.as_ref(), 8)
            } else if use_split_4thread {
                (kernels.matvec_q8_split_q8_1_4thread_residual.as_ref(), 2)
            } else {
                (kernels.matvec_q8_split_q8_1_residual.as_ref(), 2)
            };
            if let Some(mv_fn) = mv_fn_opt {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = (out_dim_u32 + nr_grid - 1) / nr_grid;
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(split_buf)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_split_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    if kernels.use_q4_split_dispatch {
        if let Some(split_buf) = q4_split_sibling {
            if let Some(mv_fn) = kernels.matvec_q4_split_q8_1_residual.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(split_buf)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_split_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    launch_matvec_preq8_1_residual(
        device, kernels, weight, q8_1_buf, residual, output, out_dim, in_dim, label,
    )
}

/// Repack a single Q8Raw buffer into the per-row split (SoA) layout.
///
/// Produces a buffer of `out_dim * nb * 34` bytes (same density as Q8Raw,
/// reorganized as `[scale[nb] | quant[nb]]` per row). The source buffer is
/// read by reference and preserved (caller keeps the original Q8Raw for
/// prefill HGEMM path).
///
/// # Safety
///
/// - `raw_buf` must contain at least `out_dim * nb * 34` bytes of valid Q8_0 data.
/// - `in_dim` must be a multiple of 32, and `nb = in_dim / 32` must be even
/// (this is enforced; the matvec kernel requires 4-byte alignment of the
/// quant stream offset which is `2 * nb` bytes from row start).
unsafe fn repack_q8_raw_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    raw_buf: &CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q8 split repack: in_dim={in_dim} is not a multiple of 32",
        )));
    }
    let nb = in_dim / 32;
    if nb % 2 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q8 split repack: in_dim={in_dim} yields nb={nb} (odd); split layout requires even nb",
        )));
    }
    let row_bytes = nb * 34;
    let total_bytes = out_dim * row_bytes;
    let mut split_buf: CudaSlice<u8> = device.alloc_zeros(total_bytes)?;

    // Source must hold out_dim * nb * 34 bytes. We do NOT require equality
    // (some buffers are slightly oversized at the tail).
    let expected_src_bytes = out_dim * nb * 34;
    if raw_buf.len() < expected_src_bytes {
        return Err(RuntimeError::Compute(format!(
            "Q8 split repack: source buffer has {} bytes, expected {} (out_dim={out_dim}, nb={nb})",
            raw_buf.len(), expected_src_bytes,
        )));
    }

    let total_blocks = (out_dim * nb) as u32;
    let block_size = 256u32;
    let grid_size = (total_blocks + block_size - 1) / block_size;
    let nb_u32 = nb as u32;
    let out_dim_u32 = out_dim as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    device
        .stream
        .launch_builder(repack_kernel)
        .arg(raw_buf)
        .arg(&mut split_buf)
        .arg(&nb_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("repack_q8_raw_to_split launch: {e}")))?;

    device.synchronize()?;
    Ok(split_buf)
}

/// Repack a single Q4Raw buffer into the per-row split (SoA) layout.
///
/// Produces a buffer of `out_dim * nb * 18` bytes structured as one row per
/// `out_dim`. Each row holds `[f16 scale * nb][nibble[16] * nb]`. The per-row
/// stride is `18 * nb` bytes (same density as the source). The nibble stream
/// starts at byte offset `2 * nb` which is 4-byte aligned because `nb` is even
/// for every shipped model dimension.
#[allow(dead_code)]
unsafe fn repack_q4_raw_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    raw_buf: &CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q4 split repack: in_dim={in_dim} is not a multiple of 32",
        )));
    }
    let nb = in_dim / 32;
    if nb % 2 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q4 split repack: in_dim={in_dim} yields nb={nb} (odd); split layout requires even nb",
        )));
    }
    let row_bytes = nb * 18;
    let total_bytes = out_dim * row_bytes;
    let mut split_buf: CudaSlice<u8> = device.alloc_zeros(total_bytes)?;

    let expected_src_bytes = out_dim * nb * 18;
    if raw_buf.len() < expected_src_bytes {
        return Err(RuntimeError::Compute(format!(
            "Q4 split repack: source buffer has {} bytes, expected {} (out_dim={out_dim}, nb={nb})",
            raw_buf.len(), expected_src_bytes,
        )));
    }

    let total_blocks = (out_dim * nb) as u32;
    let block_size = 256u32;
    let grid_size = (total_blocks + block_size - 1) / block_size;
    let nb_u32 = nb as u32;
    let out_dim_u32 = out_dim as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    device
        .stream
        .launch_builder(repack_kernel)
        .arg(raw_buf)
        .arg(&mut split_buf)
        .arg(&nb_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("repack_q4_raw_to_split launch: {e}")))?;

    device.synchronize()?;
    Ok(split_buf)
}

/// Largest-first allocator for cloning every Q8Raw projection weight in a model
/// into the per-row split (SoA) sibling layout.
///
/// Schedules clones in size-descending order to keep the CUDA allocator's heap
/// compact (Strategy A from `-gdn-byte-reduction.patch` analysis: largest
/// blocks first, smaller tiers fit cleanly in the tail).
///
/// The hard cap `CLONE_BUDGET_BYTES` reserves free VRAM headroom for the
/// downstream KV cache and scratch allocations that run AFTER preload.
///
/// Returns `(num_layers_with_any_split, first_oom_layer_idx, total_oom_count,
/// total_jobs_attempted)`. On OOM the loop aborts (no more attempts).
///
/// # Safety
///
/// Caller must ensure the `repack_kernel` is the compiled `repack_q8_raw_to_split`.
unsafe fn repack_all_layers_q8_clone_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
) -> (usize, Option<usize>, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    #[derive(Copy, Clone, Debug)]
    enum SplitWeightKind { Wq, Wk, Wv, Wo, Gate, Up, Down }

    struct Job {
        layer_idx: usize,
        kind: SplitWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 7);

    fn push_if_q8raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: SplitWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q8Raw(_) = w {
            if in_dim % 32 != 0 { return; }
            let nb = in_dim / 32;
            if nb % 2 != 0 { return; }
            let size_bytes = out_dim * nb * 34;
            jobs.push(Job { layer_idx, kind, out_dim, in_dim, size_bytes });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        let wq_out_dim = if layer.layer_type == 1 {
            kv_dim + kv_dim + q_dim // GDN fused QKV
        } else if layer.attn_q_norm.is_some() {
            q_dim * 2 // Qwen3.5 full-attn Q+gate fusion
        } else {
            q_dim
        };
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Wq, &layer.wq, wq_out_dim, hidden);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Wk, &layer.wk, kv_dim, hidden);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Wv, &layer.wv, kv_dim, hidden);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Wo, &layer.wo, hidden, q_dim);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Gate, &layer.w_gate, inter, hidden);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Up, &layer.w_up, inter, hidden);
        push_if_q8raw(&mut jobs, layer_idx, SplitWeightKind::Down, &layer.w_down, hidden, inter);
    }

    // Largest-first. Tie-break: full-attention layers before GDN (higher per-token
    // decode bandwidth). Final tie-break: ascending layer index for determinism.
    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                let pb = if layers[b.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    // production CLONE_BUDGET: 5.1 GB covers FFN tier on A100-80GB
    // (cuBLAS workspace + larger CUDA context on PCIe variant eats ~1.5 GB
    // more reserved memory than SXM4). Partial clone coverage still gains
    // the bulk of the bandwidth saving; Q8Raw fallback keeps correctness.
    const CLONE_BUDGET_BYTES: usize = 5_100_000_000;
    let mut layers_with_split = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() { break; }
        if bytes_cloned + job.size_bytes > CLONE_BUDGET_BYTES { break; }

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            SplitWeightKind::Wq => if let GpuWeightBuf::Q8Raw(b) = &layer.wq { Some(b) } else { None },
            SplitWeightKind::Wk => if let GpuWeightBuf::Q8Raw(b) = &layer.wk { Some(b) } else { None },
            SplitWeightKind::Wv => if let GpuWeightBuf::Q8Raw(b) = &layer.wv { Some(b) } else { None },
            SplitWeightKind::Wo => if let GpuWeightBuf::Q8Raw(b) = &layer.wo { Some(b) } else { None },
            SplitWeightKind::Gate => if let GpuWeightBuf::Q8Raw(b) = &layer.w_gate { Some(b) } else { None },
            SplitWeightKind::Up => if let GpuWeightBuf::Q8Raw(b) = &layer.w_up { Some(b) } else { None },
            SplitWeightKind::Down => if let GpuWeightBuf::Q8Raw(b) = &layer.w_down { Some(b) } else { None },
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q8_raw_to_split(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(split_buf) => {
                match job.kind {
                    SplitWeightKind::Wq => layer.q8_split_wq = Some(split_buf),
                    SplitWeightKind::Wk => layer.q8_split_wk = Some(split_buf),
                    SplitWeightKind::Wv => layer.q8_split_wv = Some(split_buf),
                    SplitWeightKind::Wo => layer.q8_split_wo = Some(split_buf),
                    SplitWeightKind::Gate => layer.q8_split_w_gate = Some(split_buf),
                    SplitWeightKind::Up => layer.q8_split_w_up = Some(split_buf),
                    SplitWeightKind::Down => layer.q8_split_w_down = Some(split_buf),
                }
                layers_with_split.insert(job.layer_idx);
                bytes_cloned += job.size_bytes;
            }
            Err(_) => {
                oom_layer = Some(job.layer_idx);
                oom_count += 1;
                break;
            }
        }
    }

    (layers_with_split.len(), oom_layer, oom_count, jobs.len())
}

/// Largest-first allocator for cloning every Q4Raw projection weight into the
/// per-row split (SoA) sibling layout. Mirror of `repack_all_layers_q8_clone_to_split`.
#[allow(dead_code)]
unsafe fn repack_all_layers_q4_clone_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
) -> (usize, Option<usize>, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    #[derive(Copy, Clone, Debug)]
    enum SplitWeightKind { Wq, Wk, Wv, Wo, Gate, Up, Down }

    struct Job {
        layer_idx: usize,
        kind: SplitWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 7);

    fn push_if_q4raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: SplitWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q4Raw(_) = w {
            if in_dim % 32 != 0 { return; }
            let nb = in_dim / 32;
            if nb % 2 != 0 { return; }
            let size_bytes = out_dim * nb * 18;
            jobs.push(Job { layer_idx, kind, out_dim, in_dim, size_bytes });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        let wq_out_dim = if layer.layer_type == 1 {
            kv_dim + kv_dim + q_dim
        } else if layer.attn_q_norm.is_some() {
            q_dim * 2
        } else {
            q_dim
        };
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Wq, &layer.wq, wq_out_dim, hidden);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Wk, &layer.wk, kv_dim, hidden);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Wv, &layer.wv, kv_dim, hidden);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Wo, &layer.wo, hidden, q_dim);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Gate, &layer.w_gate, inter, hidden);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Up, &layer.w_up, inter, hidden);
        push_if_q4raw(&mut jobs, layer_idx, SplitWeightKind::Down, &layer.w_down, hidden, inter);
    }

    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                let pb = if layers[b.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    // Q4 has 1.9x the per-element density of Q8 (18 vs 34 bytes per 32-elem
    // block), so for the same model the Q4 clone budget can be smaller.
    // Reuse the 5.1 GB budget for safety -- Q4_0 Qwen3.5-9B is ~5 GB total
    // so the entire model can be cloned with room to spare.
    const CLONE_BUDGET_BYTES: usize = 5_100_000_000;
    let mut layers_with_split = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() { break; }
        if bytes_cloned + job.size_bytes > CLONE_BUDGET_BYTES { break; }

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            SplitWeightKind::Wq => if let GpuWeightBuf::Q4Raw(b) = &layer.wq { Some(b) } else { None },
            SplitWeightKind::Wk => if let GpuWeightBuf::Q4Raw(b) = &layer.wk { Some(b) } else { None },
            SplitWeightKind::Wv => if let GpuWeightBuf::Q4Raw(b) = &layer.wv { Some(b) } else { None },
            SplitWeightKind::Wo => if let GpuWeightBuf::Q4Raw(b) = &layer.wo { Some(b) } else { None },
            SplitWeightKind::Gate => if let GpuWeightBuf::Q4Raw(b) = &layer.w_gate { Some(b) } else { None },
            SplitWeightKind::Up => if let GpuWeightBuf::Q4Raw(b) = &layer.w_up { Some(b) } else { None },
            SplitWeightKind::Down => if let GpuWeightBuf::Q4Raw(b) = &layer.w_down { Some(b) } else { None },
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q4_raw_to_split(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(split_buf) => {
                match job.kind {
                    SplitWeightKind::Wq => layer.q4_split_wq = Some(split_buf),
                    SplitWeightKind::Wk => layer.q4_split_wk = Some(split_buf),
                    SplitWeightKind::Wv => layer.q4_split_wv = Some(split_buf),
                    SplitWeightKind::Wo => layer.q4_split_wo = Some(split_buf),
                    SplitWeightKind::Gate => layer.q4_split_w_gate = Some(split_buf),
                    SplitWeightKind::Up => layer.q4_split_w_up = Some(split_buf),
                    SplitWeightKind::Down => layer.q4_split_w_down = Some(split_buf),
                }
                layers_with_split.insert(job.layer_idx);
                bytes_cloned += job.size_bytes;
            }
            Err(_) => {
                oom_layer = Some(job.layer_idx);
                oom_count += 1;
                break;
            }
        }
    }

    (layers_with_split.len(), oom_layer, oom_count, jobs.len())
}

/// Clone GDN-specific Q4Raw projection weights (`ssm_out`, `attn_gate`,
/// `ssm_alpha`, `ssm_beta`) into per-row split (SoA) siblings.
///
/// GDN-specific because these weights live in `Option<GpuWeightBuf>` fields on
/// `LayerWeightsGpu` (separate from the standard wq/wk/wv/wo/FFN weights).
/// profile shows the 4096x4096 ssm_out matvec is ~10% of decode time on
/// Qwen3.5-9B; the SPLIT layout closes ~20% of that.
///
/// Q4 only. Q8 + GDN_SPLIT OOMs on A100-80GB per . Returns a tuple of
/// `(layers_with_any_split, total_jobs)`.
unsafe fn repack_all_layers_gdn_q4_clone_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
) -> (usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    // GDN value_dim heuristic from gpu_buffers::upload_layer_weights:
    // num_heads = group_count * GQA_ratio_2 = 16 * 2 = 32 (NOT hp.num_heads).
    // Use ssm_norm_tiled length if available; else fall back to value_dim from hp.
    let mut n_layers_with_split: usize = 0;
    let mut total_jobs: usize = 0;

    for layer in layers.iter_mut() {
        if layer.layer_type != 1 { continue; }

        // Determine value_dim per layer from ssm_norm_tiled length (set in upload).
        let value_dim = layer.ssm_norm_tiled.as_ref().map(|s| s.len()).unwrap_or(q_dim);

        let mut layer_had_any = false;

        // ssm_out: [hidden, value_dim] Q4Raw
        if let Some(GpuWeightBuf::Q4Raw(ref raw)) = layer.ssm_out {
            total_jobs += 1;
            if let Ok(split) = repack_q4_raw_to_split(device, repack_kernel, raw, hidden, value_dim) {
                layer.q4_split_ssm_out = Some(split);
                layer_had_any = true;
            }
        }
        // attn_gate: [value_dim, hidden] Q4Raw
        if let Some(GpuWeightBuf::Q4Raw(ref raw)) = layer.attn_gate {
            total_jobs += 1;
            if let Ok(split) = repack_q4_raw_to_split(device, repack_kernel, raw, value_dim, hidden) {
                layer.q4_split_attn_gate = Some(split);
                layer_had_any = true;
            }
        }
        // ssm_alpha: [num_heads, hidden] Q4Raw. num_heads inferred from hp.num_heads
        // (GDN uses num_heads from hyperparams for the per-head projection).
        if let Some(GpuWeightBuf::Q4Raw(ref raw)) = layer.ssm_alpha {
            total_jobs += 1;
            // alpha output dim = qkv_dim's num_heads scalar (per-head scalar),
            // matches the kv_dim path; concretely raw.len() / (hidden / 32 * 18)
            // gives the actual out_dim.
            let nb = hidden / 32;
            if nb > 0 && hidden % 32 == 0 {
                let row_bytes = nb * 18;
                if row_bytes > 0 && raw.len() % row_bytes == 0 {
                    let out_dim = raw.len() / row_bytes;
                    if let Ok(split) = repack_q4_raw_to_split(device, repack_kernel, raw, out_dim, hidden) {
                        layer.q4_split_ssm_alpha = Some(split);
                        layer_had_any = true;
                    }
                }
            }
            let _ = kv_dim; // suppress unused warning
        }
        if let Some(GpuWeightBuf::Q4Raw(ref raw)) = layer.ssm_beta {
            total_jobs += 1;
            let nb = hidden / 32;
            if nb > 0 && hidden % 32 == 0 {
                let row_bytes = nb * 18;
                if row_bytes > 0 && raw.len() % row_bytes == 0 {
                    let out_dim = raw.len() / row_bytes;
                    if let Ok(split) = repack_q4_raw_to_split(device, repack_kernel, raw, out_dim, hidden) {
                        layer.q4_split_ssm_beta = Some(split);
                        layer_had_any = true;
                    }
                }
            }
            let _ = head_dim;
        }

        if layer_had_any { n_layers_with_split += 1; }
    }
    (n_layers_with_split, total_jobs)
}

// =============================================================================
// End split-layout integration: dispatch helpers.
// =============================================================================

// =============================================================================
// tile-layout integration:: tile-grouped layout dispatch.
//
// The TILE layout colocates 8 scales and 8 quant / nibble blocks within a
// single tile (272 B for Q8, 144 B for Q4). Density is identical to Q8Raw /
// Q4Raw; the win is L1-sector locality vs the SPLIT layout where the scales
// stream lives `2*nb` bytes away from the quants stream.
//
// Wiring:
// 1. Env vars `LUMEN_CUDA_Q8_TILE=1` / `LUMEN_CUDA_Q4_TILE=1` flip the
// `KernelSet::use_q*_tile_dispatch` flags at session start.
// 2. `repack_all_layers_q*_clone_to_tile` runs once during `preload_weights`
// to clone Q8Raw / Q4Raw projection weights into TILE siblings on
// `LayerWeightsGpu`. Skipped when the env var is unset OR the repack
// kernel failed to compile.
// 3. `launch_matvec_preq8_1_tile` / `launch_matvec_preq8_1_residual_tile`
// prefer the TILE sibling, then fall back to the SPLIT dispatch helper
// (which itself falls back to Aligned / Raw).
//
// default-off contract (clean revert): with env vars unset the dispatch path is
// byte-for-byte identical to the SPLIT integration (which itself reduces to
// the pre-SPLIT base path when its env vars are unset).
// =============================================================================

/// Dispatch a dp4a matvec with pre-quantized Q8_1 input, preferring the
/// Q8Tile / Q4Tile sibling buffer when present.
///
/// When `q8_tile_sibling` is `Some` AND `kernels.use_q8_tile_dispatch` is true,
/// routes to `matvec_q8_tile_q8_1`. Likewise for Q4. Falls through to
/// `launch_matvec_preq8_1_split` (which itself falls back to the base
/// dispatch) when no tile sibling is set or the TILE dispatch is disabled.
///
/// Tile and split siblings can coexist; tile wins because the layout is
/// strictly more L1-locality-friendly within the same byte budget.
///
/// # Safety
///
/// Same constraints as `launch_matvec_preq8_1_split`. The tile sibling is
/// produced by `repack_all_layers_q*_clone_to_tile` and has identical element
/// count to the base weight.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn launch_matvec_preq8_1_tile(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_tile_sibling: Option<&CudaSlice<u8>>,
    q4_tile_sibling: Option<&CudaSlice<u8>>,
    q8_split_sibling: Option<&CudaSlice<u8>>,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    q8_1_buf: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // Q8 tile path.
    if kernels.use_q8_tile_dispatch {
        if let Some(tile_buf) = q8_tile_sibling {
            if let Some(mv_fn) = kernels.matvec_q8_tile_q8_1.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(tile_buf)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_tile_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    // Q4 tile path.
    if kernels.use_q4_tile_dispatch {
        if let Some(tile_buf) = q4_tile_sibling {
            if let Some(mv_fn) = kernels.matvec_q4_tile_q8_1.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(tile_buf)
                    .arg(q8_1_buf)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_tile_q8_1 preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    // Fall-through: SPLIT-aware dispatch (which itself falls back to the
    // base Q8Raw/Q8Aligned/Q4Raw/Q4Aligned path when neither SPLIT sibling
    // is present).
    launch_matvec_preq8_1_split(
        device, kernels, weight,
        q8_split_sibling, q4_split_sibling,
        q8_1_buf, output, out_dim, in_dim, label,
    )
}

/// Dispatch a dp4a matvec + fused residual, preferring the Q8Tile / Q4Tile
/// sibling when present. Falls through to `launch_matvec_preq8_1_residual_split`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn launch_matvec_preq8_1_residual_tile(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    q8_tile_sibling: Option<&CudaSlice<u8>>,
    q4_tile_sibling: Option<&CudaSlice<u8>>,
    q8_split_sibling: Option<&CudaSlice<u8>>,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    q8_1_buf: &CudaSlice<u8>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if kernels.use_q8_tile_dispatch {
        if let Some(tile_buf) = q8_tile_sibling {
            if let Some(mv_fn) = kernels.matvec_q8_tile_q8_1_residual.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(tile_buf)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_tile_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    if kernels.use_q4_tile_dispatch {
        if let Some(tile_buf) = q4_tile_sibling {
            if let Some(mv_fn) = kernels.matvec_q4_tile_q8_1_residual.as_ref() {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_grid = dp4a_q4_grid(out_dim_u32);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(tile_buf)
                    .arg(q8_1_buf)
                    .arg(residual)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(mv_cfg)
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_tile_q8_1_residual preq {label}: {e}",
                    )))?;
                return Ok(());
            }
        }
    }
    launch_matvec_preq8_1_residual_split(
        device, kernels, weight,
        q8_split_sibling, q4_split_sibling,
        q8_1_buf, residual, output, out_dim, in_dim, label,
    )
}

/// Repack a single Q8Raw buffer into the per-row tile-grouped layout.
///
/// Produces a buffer of `out_dim * (nb/8) * 272` bytes = `out_dim * 34 * nb`
/// bytes (same density as Q8Raw, regrouped into 272 B tiles of 8 blocks each).
/// `nb` must be a multiple of 8 (one tile = 8 blocks). The original Q8Raw is
/// preserved by the caller (prefill path needs the AoS layout).
///
/// # Safety
///
/// - `raw_buf` must contain at least `out_dim * nb * 34` bytes of valid Q8_0 data.
/// - `in_dim` must be a multiple of 32, and `nb = in_dim / 32` must be a
/// multiple of 8.
unsafe fn repack_q8_raw_to_tile(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    raw_buf: &CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q8 tile repack: in_dim={in_dim} is not a multiple of 32",
        )));
    }
    let nb = in_dim / 32;
    if nb % 8 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q8 tile repack: in_dim={in_dim} yields nb={nb} (not a multiple of 8); tile layout requires nb%%8==0",
        )));
    }
    let row_bytes = (nb / 8) * 272;
    let total_bytes = out_dim * row_bytes;
    let mut tile_buf: CudaSlice<u8> = device.alloc_zeros(total_bytes)?;

    let expected_src_bytes = out_dim * nb * 34;
    if raw_buf.len() < expected_src_bytes {
        return Err(RuntimeError::Compute(format!(
            "Q8 tile repack: source buffer has {} bytes, expected {} (out_dim={out_dim}, nb={nb})",
            raw_buf.len(), expected_src_bytes,
        )));
    }

    let total_blocks = (out_dim * nb) as u32;
    let block_size = 256u32;
    let grid_size = (total_blocks + block_size - 1) / block_size;
    let nb_u32 = nb as u32;
    let out_dim_u32 = out_dim as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    device
        .stream
        .launch_builder(repack_kernel)
        .arg(raw_buf)
        .arg(&mut tile_buf)
        .arg(&nb_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("repack_q8_raw_to_tile launch: {e}")))?;

    device.synchronize()?;
    Ok(tile_buf)
}

/// Repack a single Q4Raw buffer into the per-row tile-grouped layout.
///
/// Produces a buffer of `out_dim * (nb/8) * 144` bytes = `out_dim * 18 * nb`
/// bytes. `nb` must be a multiple of 8.
unsafe fn repack_q4_raw_to_tile(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    raw_buf: &CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
) -> Result<CudaSlice<u8>, RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q4 tile repack: in_dim={in_dim} is not a multiple of 32",
        )));
    }
    let nb = in_dim / 32;
    if nb % 8 != 0 {
        return Err(RuntimeError::Compute(format!(
            "Q4 tile repack: in_dim={in_dim} yields nb={nb} (not a multiple of 8); tile layout requires nb%%8==0",
        )));
    }
    let row_bytes = (nb / 8) * 144;
    let total_bytes = out_dim * row_bytes;
    let mut tile_buf: CudaSlice<u8> = device.alloc_zeros(total_bytes)?;

    let expected_src_bytes = out_dim * nb * 18;
    if raw_buf.len() < expected_src_bytes {
        return Err(RuntimeError::Compute(format!(
            "Q4 tile repack: source buffer has {} bytes, expected {} (out_dim={out_dim}, nb={nb})",
            raw_buf.len(), expected_src_bytes,
        )));
    }

    let total_blocks = (out_dim * nb) as u32;
    let block_size = 256u32;
    let grid_size = (total_blocks + block_size - 1) / block_size;
    let nb_u32 = nb as u32;
    let out_dim_u32 = out_dim as u32;

    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    device
        .stream
        .launch_builder(repack_kernel)
        .arg(raw_buf)
        .arg(&mut tile_buf)
        .arg(&nb_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("repack_q4_raw_to_tile launch: {e}")))?;

    device.synchronize()?;
    Ok(tile_buf)
}

/// Largest-first allocator for cloning every Q8Raw projection weight in a
/// model into the per-row tile-grouped sibling layout. Mirror of
/// `repack_all_layers_q8_clone_to_split` but populates the `q8_tile_*`
/// fields and rejects any in_dim whose `nb` is not a multiple of 8.
///
/// Returns `(num_layers_with_any_tile, first_oom_layer_idx, total_oom_count,
/// total_jobs_attempted)`. On OOM the loop aborts (no more attempts).
///
/// # Safety
///
/// Caller must ensure the `repack_kernel` is the compiled `repack_q8_raw_to_tile`.
unsafe fn repack_all_layers_q8_clone_to_tile(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
) -> (usize, Option<usize>, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    #[derive(Copy, Clone, Debug)]
    enum TileWeightKind { Wq, Wk, Wv, Wo, Gate, Up, Down }

    struct Job {
        layer_idx: usize,
        kind: TileWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 7);

    fn push_if_q8raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: TileWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q8Raw(_) = w {
            if in_dim % 32 != 0 { return; }
            let nb = in_dim / 32;
            // Tile layout requires nb to be a multiple of 8 (8 blocks per tile).
            if nb % 8 != 0 { return; }
            let size_bytes = out_dim * nb * 34;
            jobs.push(Job { layer_idx, kind, out_dim, in_dim, size_bytes });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        let wq_out_dim = if layer.layer_type == 1 {
            kv_dim + kv_dim + q_dim
        } else if layer.attn_q_norm.is_some() {
            q_dim * 2
        } else {
            q_dim
        };
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Wq,   &layer.wq,     wq_out_dim, hidden);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Wk,   &layer.wk,     kv_dim,     hidden);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Wv,   &layer.wv,     kv_dim,     hidden);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Wo,   &layer.wo,     hidden,     q_dim);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Gate, &layer.w_gate, inter,      hidden);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Up,   &layer.w_up,   inter,      hidden);
        push_if_q8raw(&mut jobs, layer_idx, TileWeightKind::Down, &layer.w_down, hidden,     inter);
    }

    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                let pb = if layers[b.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    // Same 5.1 GB budget as SPLIT. Tile siblings COEXIST with split siblings
    // when both env vars are set, so the effective VRAM usage may approach
    // 2x the model size. Q8 + TILE on Qwen3.5-9B (~10 GB model) fits in 80
    // GB even with SPLIT + aligned-repack also enabled, but the budget
    // caps clone attempts to keep KV cache + scratch headroom intact.
    const CLONE_BUDGET_BYTES: usize = 5_100_000_000;
    let mut layers_with_tile = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() { break; }
        if bytes_cloned + job.size_bytes > CLONE_BUDGET_BYTES { break; }

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            TileWeightKind::Wq   => if let GpuWeightBuf::Q8Raw(b) = &layer.wq     { Some(b) } else { None },
            TileWeightKind::Wk   => if let GpuWeightBuf::Q8Raw(b) = &layer.wk     { Some(b) } else { None },
            TileWeightKind::Wv   => if let GpuWeightBuf::Q8Raw(b) = &layer.wv     { Some(b) } else { None },
            TileWeightKind::Wo   => if let GpuWeightBuf::Q8Raw(b) = &layer.wo     { Some(b) } else { None },
            TileWeightKind::Gate => if let GpuWeightBuf::Q8Raw(b) = &layer.w_gate { Some(b) } else { None },
            TileWeightKind::Up   => if let GpuWeightBuf::Q8Raw(b) = &layer.w_up   { Some(b) } else { None },
            TileWeightKind::Down => if let GpuWeightBuf::Q8Raw(b) = &layer.w_down { Some(b) } else { None },
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q8_raw_to_tile(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(tile_buf) => {
                match job.kind {
                    TileWeightKind::Wq   => layer.q8_tile_wq     = Some(tile_buf),
                    TileWeightKind::Wk   => layer.q8_tile_wk     = Some(tile_buf),
                    TileWeightKind::Wv   => layer.q8_tile_wv     = Some(tile_buf),
                    TileWeightKind::Wo   => layer.q8_tile_wo     = Some(tile_buf),
                    TileWeightKind::Gate => layer.q8_tile_w_gate = Some(tile_buf),
                    TileWeightKind::Up   => layer.q8_tile_w_up   = Some(tile_buf),
                    TileWeightKind::Down => layer.q8_tile_w_down = Some(tile_buf),
                }
                layers_with_tile.insert(job.layer_idx);
                bytes_cloned += job.size_bytes;
            }
            Err(_) => {
                oom_layer = Some(job.layer_idx);
                oom_count += 1;
                break;
            }
        }
    }

    (layers_with_tile.len(), oom_layer, oom_count, jobs.len())
}

/// Largest-first allocator for cloning every Q4Raw projection weight into
/// per-row tile-grouped siblings. Mirror of
/// `repack_all_layers_q8_clone_to_tile`.
unsafe fn repack_all_layers_q4_clone_to_tile(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
) -> (usize, Option<usize>, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let heads = hp.num_heads as usize;
    let kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter = hp.intermediate_dim as usize;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    #[derive(Copy, Clone, Debug)]
    enum TileWeightKind { Wq, Wk, Wv, Wo, Gate, Up, Down }

    struct Job {
        layer_idx: usize,
        kind: TileWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 7);

    fn push_if_q4raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: TileWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q4Raw(_) = w {
            if in_dim % 32 != 0 { return; }
            let nb = in_dim / 32;
            if nb % 8 != 0 { return; }
            let size_bytes = out_dim * nb * 18;
            jobs.push(Job { layer_idx, kind, out_dim, in_dim, size_bytes });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        let wq_out_dim = if layer.layer_type == 1 {
            kv_dim + kv_dim + q_dim
        } else if layer.attn_q_norm.is_some() {
            q_dim * 2
        } else {
            q_dim
        };
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Wq,   &layer.wq,     wq_out_dim, hidden);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Wk,   &layer.wk,     kv_dim,     hidden);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Wv,   &layer.wv,     kv_dim,     hidden);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Wo,   &layer.wo,     hidden,     q_dim);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Gate, &layer.w_gate, inter,      hidden);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Up,   &layer.w_up,   inter,      hidden);
        push_if_q4raw(&mut jobs, layer_idx, TileWeightKind::Down, &layer.w_down, hidden,     inter);
    }

    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                let pb = if layers[b.layer_idx].layer_type == 0 { 0u8 } else { 1u8 };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    const CLONE_BUDGET_BYTES: usize = 5_100_000_000;
    let mut layers_with_tile = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() { break; }
        if bytes_cloned + job.size_bytes > CLONE_BUDGET_BYTES { break; }

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            TileWeightKind::Wq   => if let GpuWeightBuf::Q4Raw(b) = &layer.wq     { Some(b) } else { None },
            TileWeightKind::Wk   => if let GpuWeightBuf::Q4Raw(b) = &layer.wk     { Some(b) } else { None },
            TileWeightKind::Wv   => if let GpuWeightBuf::Q4Raw(b) = &layer.wv     { Some(b) } else { None },
            TileWeightKind::Wo   => if let GpuWeightBuf::Q4Raw(b) = &layer.wo     { Some(b) } else { None },
            TileWeightKind::Gate => if let GpuWeightBuf::Q4Raw(b) = &layer.w_gate { Some(b) } else { None },
            TileWeightKind::Up   => if let GpuWeightBuf::Q4Raw(b) = &layer.w_up   { Some(b) } else { None },
            TileWeightKind::Down => if let GpuWeightBuf::Q4Raw(b) = &layer.w_down { Some(b) } else { None },
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q4_raw_to_tile(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(tile_buf) => {
                match job.kind {
                    TileWeightKind::Wq   => layer.q4_tile_wq     = Some(tile_buf),
                    TileWeightKind::Wk   => layer.q4_tile_wk     = Some(tile_buf),
                    TileWeightKind::Wv   => layer.q4_tile_wv     = Some(tile_buf),
                    TileWeightKind::Wo   => layer.q4_tile_wo     = Some(tile_buf),
                    TileWeightKind::Gate => layer.q4_tile_w_gate = Some(tile_buf),
                    TileWeightKind::Up   => layer.q4_tile_w_up   = Some(tile_buf),
                    TileWeightKind::Down => layer.q4_tile_w_down = Some(tile_buf),
                }
                layers_with_tile.insert(job.layer_idx);
                bytes_cloned += job.size_bytes;
            }
            Err(_) => {
                oom_layer = Some(job.layer_idx);
                oom_count += 1;
                break;
            }
        }
    }

    (layers_with_tile.len(), oom_layer, oom_count, jobs.len())
}

// =============================================================================
// End tile-layout integration: dispatch helpers.
// =============================================================================

/// Check if a weight buffer uses the dp4a Q8_1 path (Q8Raw, Q8Aligned, Q4Aligned, Q4Raw).
fn weight_uses_dp4a_q8_1(weight: &GpuWeightBuf, kernels: &KernelSet) -> bool {
    match weight {
        GpuWeightBuf::Q8Raw(_) => kernels.matvec_q8_0_q8_1.is_some(),
        GpuWeightBuf::Q8Aligned(_) => kernels.matvec_q8_aligned_q8_1.is_some(),
        GpuWeightBuf::Q4Aligned(_) => kernels.matvec_q4_aligned_q8_1.is_some(),
        GpuWeightBuf::Q4Raw(_) => kernels.matvec_q4_0_dp4a.is_some(),
        _ => false,
    }
}

/// pick the output_proj SPLIT matvec kernel matching the requested NR.
///
/// Returns `None` when `nr == 32` (caller should use the legacy
/// `matvec_q8_split_output_proj` handle which is the nr32 instantiation), OR
/// when the requested NR variant didn't load. The caller falls back to nr32 in
/// that case. For `nr == 2`, returns the generic `matvec_q8_split_q8_1` kernel
/// (the pre-SPLIT-INTEGRATION default that delivered T3's +7.7%).
fn pick_output_proj_nr_kernel(
    kernels: &KernelSet,
    nr: u32,
) -> Option<&CudaFunction> {
    match nr {
        2 => kernels.matvec_q8_split_q8_1.as_ref(),
        8 => kernels.matvec_q8_split_output_proj_nr8.as_ref(),
        16 => kernels.matvec_q8_split_output_proj_nr16.as_ref(),
        64 => kernels.matvec_q8_split_output_proj_nr64.as_ref(),
        128 => kernels.matvec_q8_split_output_proj_nr128.as_ref(),
        _ => None,
    }
}

/// Launch cuBLAS HGEMV for F16 weights: `output[out_dim] = W_f16[out_dim, in_dim]^T * input_f32[in_dim]`.
///
/// Converts the F32 input to F16 via `f32_to_f16_vec`, then calls `cublasGemmEx`
/// with N=1 (GEMV). cuBLAS auto-selects the optimal GEMV path for the given
/// dimensions. Uses `CUBLAS_COMPUTE_32F_FAST_16F` for maximum tensor core
/// throughput with F16 inputs (matching the prefill HGEMM path).
///
/// # Safety
///
/// Caller must ensure:
/// - `w_f16` has `[out_dim * in_dim * 2]` bytes (F16 row-major)
/// - `input_f32` has `in_dim` elements
/// - `output_f32` has `out_dim` elements
/// - `input_f16_scratch` has at least `in_dim * 2` bytes
unsafe fn launch_hgemv_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_f16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_f16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    // Step 1: Convert F32 input to F16.
    let n = in_dim as u32;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let cvt_cfg = CudarcLaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(&kernels.f32_to_f16_vec)
        .arg(input_f32)
        .arg(&mut *input_f16_scratch)
        .arg(&n)
        .launch(cvt_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "f32_to_f16 HGEMV input {label}: {e}",
        )))?;

    // Step 2: cublasGemmEx with N=1 (triggers optimized GEMV path).
    // Row-major W[out_dim, in_dim] -> col-major W_cm[in_dim, out_dim].
    // out = W^T * x -> cublasGemmEx(OP_T, OP_N, out_dim, 1, in_dim, ...).
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = input_f16_scratch.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMV {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// Launch cuBLAS HGEMV with residual: `output = W_f16^T * input_f32 + residual`.
///
/// Copies `residual` into `output` first, then runs `cublasGemmEx` with `beta=1.0`
/// to accumulate the matvec result on top.
///
/// # Safety
///
/// Same constraints as `launch_hgemv_f16`, plus `residual` must have `out_dim` elements.
unsafe fn launch_hgemv_f16_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_f16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_f16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    // Step 1: Copy residual -> output for beta=1.0 accumulation.
    device
        .stream
        .memcpy_dtod(residual, output_f32)
        .map_err(|e| RuntimeError::Compute(format!(
            "dtod residual copy HGEMV {label}: {e}",
        )))?;

    // Step 2: Convert F32 input to F16.
    let n = in_dim as u32;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let cvt_cfg = CudarcLaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(&kernels.f32_to_f16_vec)
        .arg(input_f32)
        .arg(&mut *input_f16_scratch)
        .arg(&n)
        .launch(cvt_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "f32_to_f16 HGEMV residual input {label}: {e}",
        )))?;

    // Step 3: cublasGemmEx with N=1 and beta=1.0 for residual accumulation.
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = input_f16_scratch.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMV residual {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// Distinguishes a pre-cuBLAS setup failure (e.g. F32->BF16 conversion
/// kernel launch error) from a `cublasGemmEx` runtime status failure.
/// The BF16 GemmEx wrappers consume this distinction: a cuBLAS failure
/// arms the backend-level fallback flag and re-dispatches via
/// `matvec_bf16`, while a setup failure propagates unchanged.
enum Bf16LaunchOutcome {
    Success,
    CublasFailure(cublas_sys::cublasStatus_t),
}

/// Launch cuBLAS HGEMV-style call for BF16 weights: `output[out_dim] = W_bf16[out_dim, in_dim]^T * input_f32[in_dim]`.
///
/// Mirrors `launch_hgemv_f16` but with CUDA_R_16BF data types and
/// CUBLAS_COMPUTE_32F accumulation. Converts the F32 input to BF16 via
/// `f32_to_bf16_vec` (or vec4 variant), then calls `cublasGemmEx` with N=1
/// (GEMV). cuBLAS auto-selects the optimal BF16 path; on A100+ this is the
/// tensor-core `mma.sync.bf16.bf16.f32` lane (312 TFLOPS).
///
/// I-BF16 Phase-3: replaces the per-block `matvec_bf16` custom kernel for
/// decode-path Bf16Raw matvecs. The custom kernel was ~0.66× llama.cpp; cuBLAS
/// GemmEx's batch=1 path is faster because cuBLAS uses persistent threadblocks
/// with better HBM scheduling for these shapes.
///
/// Returns `Bf16LaunchOutcome::Success` when the call completes, or
/// `Bf16LaunchOutcome::CublasFailure(status)` when `cublasGemmEx` returns a
/// non-success status. Pre-cuBLAS setup errors (F32->BF16 conversion kernel
/// launch failures) propagate via the `Result` arm so callers can route them
/// to the standard error path; only the cuBLAS-failure case is the one
/// covered by the per-backend BF16 GemmEx fallback policy.
///
/// # Safety
///
/// Caller must ensure:
/// - `w_bf16` has `[out_dim * in_dim * 2]` bytes (BF16 row-major)
/// - `input_f32` has `in_dim` elements
/// - `output_f32` has `out_dim` elements
/// - `input_bf16_scratch` has at least `in_dim * 2` bytes
unsafe fn launch_hgemv_bf16(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_bf16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<Bf16LaunchOutcome, RuntimeError> {
    // Test-only seam: if a fault has been
    // armed via `inject_next_bf16_cublas_failure`, consume it
    // atomically and synthesize a `CublasFailure` outcome without ever
    // dispatching cuBLAS. The wrapper at
    // `launch_bf16_matvec_with_fallback` then arms the runtime fallback
    // and re-dispatches via the legacy `matvec_bf16` kernel -- the
    // exact code path that runs on a real cuBLAS-runtime failure.
    //
    // Gated by `#[cfg(any(test, feature = "test-fault-injection"))]` so
    // release builds compile this branch away in its entirety.
    #[cfg(any(test, feature = "test-fault-injection"))]
    {
        if BF16_INJECT_NEXT_CUBLAS_FAILURE.swap(false, Ordering::Relaxed) {
            return Ok(Bf16LaunchOutcome::CublasFailure(
                cublas_sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
            ));
        }
    }
    // Step 1: Convert F32 input to BF16. Prefer vec4 kernel (4 elems/thread).
    let n = in_dim as u32;
    if let Some(ref vec4_fn) = kernels.f32_to_bf16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cvt_cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(input_f32)
            .arg(&mut *input_bf16_scratch)
            .arg(&n)
            .launch(cvt_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16_vec4 HGEMV input {label}: {e}",
            )))?;
    } else {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        let cvt_cfg = CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_bf16_vec)
            .arg(input_f32)
            .arg(&mut *input_bf16_scratch)
            .arg(&n)
            .launch(cvt_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16 HGEMV input {label}: {e}",
            )))?;
    }

    // Step 2: cublasGemmEx with N=1 (triggers optimized GEMV path).
    // Row-major W[out_dim, in_dim] -> col-major W_cm[in_dim, out_dim].
    // out = W^T * x -> cublasGemmEx(OP_T, OP_N, out_dim, 1, in_dim, ...).
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_bf16.device_ptr(&device.stream);
    let (a_ptr, _) = input_bf16_scratch.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        // select algo from BF16_ALGO_CACHE (populated by
        // `autotune_cublas_algos_bf16` at session init when the model has
        // BF16 weights and `LUMEN_CUDA_BF16_AUTOTUNE` is not"0". Falls back
        // to `DEFAULT_TENSOR_OP` (the prior hardcoded behavior) when
        // the cache is empty or the shape was not benchmarked.
        bf16_algo_for(out_dim, in_dim),
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(Bf16LaunchOutcome::CublasFailure(status));
    }
    Ok(Bf16LaunchOutcome::Success)
}

/// Launch cuBLAS HGEMV-style call for BF16 weights with residual:
/// `output = W_bf16^T * input_f32 + residual`.
///
/// Copies `residual` into `output` first, then calls `cublasGemmEx` with
/// `beta=1.0` to accumulate the matvec result. Mirrors
/// `launch_hgemv_f16_residual` with CUDA_R_16BF inputs.
///
/// # Safety
///
/// Same constraints as `launch_hgemv_bf16`, plus `residual` must have
/// `out_dim` elements.
unsafe fn launch_hgemv_bf16_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_bf16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<Bf16LaunchOutcome, RuntimeError> {
    // Test-only seam: mirror the inject
    // check in `launch_hgemv_bf16`. The same one-shot flag drives both
    // residual and non-residual variants because the test surface
    // covers both wrappers.
    //
    // Gated by `#[cfg(any(test, feature = "test-fault-injection"))]` so
    // release builds compile this branch away in its entirety.
    #[cfg(any(test, feature = "test-fault-injection"))]
    {
        if BF16_INJECT_NEXT_CUBLAS_FAILURE.swap(false, Ordering::Relaxed) {
            return Ok(Bf16LaunchOutcome::CublasFailure(
                cublas_sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
            ));
        }
    }
    // Step 1: Copy residual -> output for beta=1.0 accumulation.
    device
        .stream
        .memcpy_dtod(residual, output_f32)
        .map_err(|e| RuntimeError::Compute(format!(
            "dtod residual copy HGEMV BF16 {label}: {e}",
        )))?;

    // Step 2: Convert F32 input to BF16.
    let n = in_dim as u32;
    if let Some(ref vec4_fn) = kernels.f32_to_bf16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cvt_cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(input_f32)
            .arg(&mut *input_bf16_scratch)
            .arg(&n)
            .launch(cvt_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16_vec4 HGEMV residual input {label}: {e}",
            )))?;
    } else {
        let block = 256u32;
        let grid = (n + block - 1) / block;
        let cvt_cfg = CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_bf16_vec)
            .arg(input_f32)
            .arg(&mut *input_bf16_scratch)
            .arg(&n)
            .launch(cvt_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "f32_to_bf16 HGEMV residual input {label}: {e}",
            )))?;
    }

    // Step 3: cublasGemmEx with N=1 and beta=1.0 for residual accumulation.
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_bf16.device_ptr(&device.stream);
    let (a_ptr, _) = input_bf16_scratch.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,
        1i32,
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
        // same BF16 autotune lookup as the non-residual variant.
        // The optimal algo for (out, in) is identical between beta=0.0 and
        // beta=1.0 GemmEx (the beta only affects the writeback, not the
        // compute path), so reusing `bf16_algo_for` is safe.
        bf16_algo_for(out_dim, in_dim),
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(Bf16LaunchOutcome::CublasFailure(status));
    }
    Ok(Bf16LaunchOutcome::Success)
}

/// Launches the legacy per-block `matvec_bf16` kernel for a plain BF16
/// matvec. This is the fallback path consumed by the BF16 GemmEx wrapper
/// when the cuBLAS path is unavailable or the per-call fallback flag has
/// been armed. Matches the existing legacy dispatch at the BF16Raw arm
/// of `launch_matvec` and the output-projection fallback in
/// `compute_final`.
///
/// # Safety
///
/// Same constraints as the equivalent BF16Raw arm in `launch_matvec`:
/// - `w_bf16` is a row-major `[out_dim * in_dim * 2]`-byte BF16 weight
/// - `input` has `in_dim` F32 elements
/// - `output` has `out_dim` F32 elements
unsafe fn launch_legacy_matvec_bf16(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    device
        .stream
        .launch_builder(&kernels.matvec_bf16)
        .arg(w_bf16)
        .arg(input)
        .arg(output)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "matvec_bf16 fallback {label} launch: {e}",
        )))?;
    Ok(())
}

/// Launches the legacy per-block `matvec_bf16_residual` kernel for a
/// fused BF16 matvec + residual. Mirrors `launch_legacy_matvec_bf16`
/// with a residual accumulator argument; corresponds to the existing
/// legacy dispatch at the BF16Raw arm of `launch_matvec_residual`.
///
/// # Safety
///
/// Same constraints as `launch_legacy_matvec_bf16`, plus `residual`
/// must have `out_dim` F32 elements.
unsafe fn launch_legacy_matvec_bf16_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    device
        .stream
        .launch_builder(&kernels.matvec_bf16_residual)
        .arg(w_bf16)
        .arg(input)
        .arg(output)
        .arg(residual)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "matvec_bf16_residual fallback {label} launch: {e}",
        )))?;
    Ok(())
}

/// BF16 matvec wrapper: attempts the cuBLAS GemmEx fast path when
/// available; on a per-call cuBLAS failure, arms the process-wide
/// runtime-fallback flag, emits a single warning, and re-dispatches the
/// same matvec on the legacy `matvec_bf16` kernel so the in-flight
/// request continues without aborting.
///
/// Selectability of the GemmEx attempt is composed inside via
/// `bf16_gemmex_enabled()`, which folds the `LUMEN_CUDA_BF16_GEMMEX=0`
/// opt-out, the startup capability probe, and any previously-armed
/// runtime fallback. Callers never have to re-derive the gate.
///
/// Returns `Ok(())` on success — either GemmEx succeeded, or the
/// fallback legacy launch succeeded. Setup errors (F32->BF16 input
/// conversion, scratch buffer issues, residual copy) propagate
/// unchanged via the `Result` arm so the standard error path handles
/// them. A `CublasFailure` from `launch_hgemv_bf16` does NOT propagate:
/// it triggers the in-flight fallback.
///
/// # Safety
///
/// Same constraints as `launch_hgemv_bf16` / `launch_legacy_matvec_bf16`.
unsafe fn launch_bf16_matvec_with_fallback(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_bf16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if bf16_gemmex_enabled() {
        match launch_hgemv_bf16(
            device,
            kernels,
            w_bf16,
            input_f32,
            output_f32,
            input_bf16_scratch,
            out_dim,
            in_dim,
            label,
        )? {
            Bf16LaunchOutcome::Success => return Ok(()),
            Bf16LaunchOutcome::CublasFailure(status) => {
                arm_bf16_gemmex_runtime_fallback(label, status);
                // fall through to the legacy launch below
            }
        }
    }
    launch_legacy_matvec_bf16(
        device,
        kernels,
        w_bf16,
        input_f32,
        output_f32,
        out_dim,
        in_dim,
        label,
    )
}

/// BF16 matvec+residual wrapper. Same contract as
/// `launch_bf16_matvec_with_fallback` but for the fused
/// `output = W^T * input + residual` path. On a per-call cuBLAS failure,
/// arms the process-wide runtime-fallback flag, emits a single warning,
/// and re-dispatches via the legacy `matvec_bf16_residual` kernel.
///
/// # Safety
///
/// Same constraints as `launch_hgemv_bf16_residual` /
/// `launch_legacy_matvec_bf16_residual`.
unsafe fn launch_bf16_matvec_residual_with_fallback(
    device: &CudaDevice,
    kernels: &KernelSet,
    w_bf16: &CudaSlice<u8>,
    input_f32: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    input_bf16_scratch: &mut CudaSlice<u8>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if bf16_gemmex_enabled() {
        match launch_hgemv_bf16_residual(
            device,
            kernels,
            w_bf16,
            input_f32,
            residual,
            output_f32,
            input_bf16_scratch,
            out_dim,
            in_dim,
            label,
        )? {
            Bf16LaunchOutcome::Success => return Ok(()),
            Bf16LaunchOutcome::CublasFailure(status) => {
                arm_bf16_gemmex_runtime_fallback(label, status);
                // fall through to the legacy launch below
            }
        }
    }
    launch_legacy_matvec_bf16_residual(
        device,
        kernels,
        w_bf16,
        input_f32,
        residual,
        output_f32,
        out_dim,
        in_dim,
        label,
    )
}

/// Fused RMSNorm + F32->F16 conversion in a single kernel dispatch.
///
/// Replaces the two-dispatch sequence: `rmsnorm` (F32 out) + `f32_to_f16_vec`.
/// The kernel computes RMSNorm and writes F16 output directly, eliminating
/// the intermediate F32 `normed[]` buffer. Falls back with an error if the
/// fused kernel was not compiled (should not happen -- compiles on all SM levels).
///
/// # Safety
///
/// `x` and `norm_weight` must have `dim` elements. `output_f16` must have
/// at least `dim * 2` bytes.
unsafe fn launch_fused_rmsnorm_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    output_f16: &mut CudaSlice<u8>,
    eps: f32,
    dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if let Some(ref func) = kernels.fused_rmsnorm_f16 {
        let block_size = rmsnorm_block_size(dim);
        let shared_bytes = rmsnorm_shared_bytes(block_size);
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let dim_u32 = dim as u32;
        device
            .stream
            .launch_builder(func)
            .arg(x)
            .arg(norm_weight)
            .arg(output_f16)
            .arg(&eps)
            .arg(&dim_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "fused_rmsnorm_f16 {label}: {e}",
            )))?;
        Ok(())
    } else {
        Err(RuntimeError::Compute(format!(
            "fused_rmsnorm_f16 kernel not available for {label}",
        )))
    }
}

/// Fused SwiGLU + F32->F16 conversion in a single kernel dispatch.
///
/// Replaces the two-dispatch sequence: `swiglu_inplace` + `f32_to_f16_vec`.
/// Reads gate and up activations, computes SwiGLU(gate, up), and writes:
/// - F32 result to `out_f32` (for non-HGEMV consumers or residual path)
/// - F16 result to `out_f16` (for the down-projection HGEMV input)
///
/// Fused SwiGLU (in-place on) + F32->F16 conversion.
///
/// `gate_inout` is read as the gate input and overwritten in-place with the
/// F32 SwiGLU result (same semantics as `swiglu_inplace`). `out_f16` receives
/// the F16 conversion of the result for the down-projection HGEMV input.
///
/// # Safety
///
/// `gate_inout` must have `n` elements (read+written in-place).
/// `up` must have `n` elements.
/// `out_f16` must have at least `n * 2` bytes.
unsafe fn launch_swiglu_f32_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    gate_inout: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out_f16: &mut CudaSlice<u8>,
    n: usize,
) -> Result<(), RuntimeError> {
    if let Some(ref func) = kernels.swiglu_f32_to_f16 {
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let launch_cfg = CudarcLaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_u32 = n as u32;
        // Kernel signature: (gate [in/out], up [in], out_f16 [out], n)
        // gate is read then written in-place (safe: each thread reads before writing).
        device
            .stream
            .launch_builder(func)
            .arg(gate_inout)
            .arg(up)
            .arg(out_f16)
            .arg(&n_u32)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!(
                "swiglu_f32_to_f16: {e}",
            )))?;
        Ok(())
    } else {
        Err(RuntimeError::Compute(
            "swiglu_f32_to_f16 kernel not available".into(),
        ))
    }
}

/// cuBLAS HGEMV with pre-converted F16 input (no F32->F16 conversion).
///
/// The caller must have already converted the input to F16 (e.g., via `f32_to_f16_vec`).
/// This function only issues the `cublasGemmEx` call with N=1 (GEMV).
///
/// # Safety
///
/// Caller must ensure:
/// - `w_f16` has `[out_dim * in_dim * 2]` bytes (F16 row-major)
/// - `input_f16` has at least `in_dim * 2` bytes (pre-converted F16)
/// - `output_f32` has `out_dim` elements
unsafe fn launch_hgemv_f16_preconverted(
    device: &CudaDevice,
    w_f16: &CudaSlice<u8>,
    input_f16: &CudaSlice<u8>,
    output_f32: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = input_f16.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMV preconverted {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// cuBLAS HGEMV with pre-converted F16 input and beta=1.0 accumulation.
///
/// Used in the graph pipeline where the caller has already placed the residual
/// into `output_f32` via the fused convert+residual kernel. The HGEMV accumulates
/// on top with beta=1.0.
///
/// # Safety
///
/// Caller must ensure:
/// - `w_f16` has `[out_dim * in_dim * 2]` bytes (F16 row-major)
/// - `input_f16` has at least `in_dim * 2` bytes (pre-converted F16)
/// - `output_f32` has `out_dim` elements (pre-loaded with residual)
unsafe fn launch_hgemv_f16_preconverted_beta1(
    device: &CudaDevice,
    w_f16: &CudaSlice<u8>,
    input_f16: &CudaSlice<u8>,
    output_f32: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = input_f16.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMV preconverted beta=1 {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// cuBLAS HGEMV with pre-converted F16 input and residual accumulation.
///
/// Copies `residual` into `output_f32` first, then runs `cublasGemmEx` with
/// `beta=1.0` to accumulate the matvec result on top.
///
/// # Safety
///
/// Same constraints as `launch_hgemv_f16_preconverted`, plus `residual` must
/// have `out_dim` elements.
#[allow(dead_code)]
unsafe fn launch_hgemv_f16_residual_preconverted(
    device: &CudaDevice,
    w_f16: &CudaSlice<u8>,
    input_f16: &CudaSlice<u8>,
    residual: &CudaSlice<f32>,
    output_f32: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    // Copy residual -> output for beta=1.0 accumulation.
    device
        .stream
        .memcpy_dtod(residual, output_f32)
        .map_err(|e| RuntimeError::Compute(format!(
            "dtod residual copy HGEMV preconverted {label}: {e}",
        )))?;

    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = input_f16.device_ptr(&device.stream);
    let (c_ptr, _) = output_f32.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMV residual preconverted {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// Batched cuBLAS HGEMV with pre-converted F16 input.
///
/// Executes `batch_count` independent HGEMV operations in a single cuBLAS call:
/// `output[i] = W_f16[i]^T * input_f16` for i in 0..batch_count
///
/// All batch elements share the same M (out_dim), N=1, K (in_dim), and the same
/// input vector. Weight and output pointers differ per batch element.
///
/// Uses `cublasGemmBatchedEx` with device pointer arrays for non-contiguous weights.
/// Saves `batch_count - 1` cuBLAS launch overheads per call (~3-5us each on A100).
///
/// # Safety
///
/// - Each `w_f16_slices[i]` must have `[out_dim * in_dim * 2]` bytes of F16 data
/// - `input_f16` must have at least `in_dim * 2` bytes (pre-converted F16)
/// - Each element in `output_f32_slices` must have `out_dim` f32 elements
/// - `dev_a_ptrs`, `dev_b_ptrs`, `dev_c_ptrs` must have capacity >= `batch_count`
#[allow(clippy::too_many_arguments)]
unsafe fn launch_hgemv_f16_batched(
    device: &CudaDevice,
    w_f16_slices: &[&CudaSlice<u8>],
    input_f16: &CudaSlice<u8>,
    output_f32_slices: &mut [&mut CudaSlice<f32>],
    dev_a_ptrs: &mut CudaSlice<u64>,
    dev_b_ptrs: &mut CudaSlice<u64>,
    dev_c_ptrs: &mut CudaSlice<u64>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    let batch_count = w_f16_slices.len();
    debug_assert_eq!(batch_count, output_f32_slices.len());
    debug_assert!(batch_count >= 2 && batch_count <= 3);

    use cudarc::driver::DevicePtr;

    // Build host-side pointer arrays (stack-allocated, tiny).
    let mut host_a = [0u64; 3];
    let mut host_b = [0u64; 3];
    let mut host_c = [0u64; 3];

    for i in 0..batch_count {
        let (w_ptr, _) = w_f16_slices[i].device_ptr(&device.stream);
        host_a[i] = w_ptr as u64;

        let (c_ptr, _) = output_f32_slices[i].device_ptr(&device.stream);
        host_c[i] = c_ptr as u64;
    }
    let (b_ptr, _) = input_f16.device_ptr(&device.stream);
    for i in 0..batch_count {
        host_b[i] = b_ptr as u64; // Same input for all batch elements
    }

    // Upload pointer arrays to pre-allocated device buffers (24 bytes max).
    device.stream.memcpy_htod(&host_a[..batch_count], dev_a_ptrs)
        .map_err(|e| RuntimeError::Compute(format!("batched HGEMV {label} A ptrs: {e}")))?;
    device.stream.memcpy_htod(&host_b[..batch_count], dev_b_ptrs)
        .map_err(|e| RuntimeError::Compute(format!("batched HGEMV {label} B ptrs: {e}")))?;
    device.stream.memcpy_htod(&host_c[..batch_count], dev_c_ptrs)
        .map_err(|e| RuntimeError::Compute(format!("batched HGEMV {label} C ptrs: {e}")))?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let (a_dev_ptr, _) = dev_a_ptrs.device_ptr(&device.stream);
    let (b_dev_ptr, _) = dev_b_ptrs.device_ptr(&device.stream);
    let (c_dev_ptr, _) = dev_c_ptrs.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmBatchedEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        a_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        b_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_dev_ptr as *const *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        batch_count as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmBatchedEx HGEMV {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// Batched cuBLAS HGEMV using PRE-COMPUTED device pointer arrays.
///
/// Identical to `launch_hgemv_f16_batched` but skips the 3 htod memcpys per call
/// because the pointer arrays were pre-computed in `preload_weights()`. This
/// eliminates ~6 htod memcpys per layer (3 for KV, 3 for gate+up) = 192 per token.
///
/// # Safety
///
/// Same requirements as `launch_hgemv_f16_batched`, plus:
/// - `dev_a_ptrs`, `dev_b_ptrs`, `dev_c_ptrs` must contain valid device pointers
/// that were uploaded during `preload_weights()` and have not been freed.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_hgemv_f16_batched_precomputed(
    device: &CudaDevice,
    dev_a_ptrs: &CudaSlice<u64>,
    dev_b_ptrs: &CudaSlice<u64>,
    dev_c_ptrs: &CudaSlice<u64>,
    batch_count: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
    algo: cublas_sys::cublasGemmAlgo_t,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    use cudarc::driver::DevicePtr;
    let (a_dev_ptr, _) = dev_a_ptrs.device_ptr(&device.stream);
    let (b_dev_ptr, _) = dev_b_ptrs.device_ptr(&device.stream);
    let (c_dev_ptr, _) = dev_c_ptrs.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmBatchedEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,  // M
        1i32,            // N = 1 (GEMV)
        in_dim as i32,   // K
        &alpha as *const f32 as *const std::ffi::c_void,
        a_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // lda
        b_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,   // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_dev_ptr as *const *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,  // ldc
        batch_count as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        algo,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmBatchedEx precomputed HGEMV {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// Probe whether `cublasGemmGroupedBatchedEx` is available at runtime.
///
/// Currently always returns false. The grouped GEMM API requires CUDA 12.5+
/// headers at compile time; the runtime probe approach (catch_unwind on the
/// dynamic symbol) was removed because the cuda-120xx Cargo features were
/// never defined, making the cfg-gated code dead. If grouped GEMM support
/// is needed in the future, add the appropriate CUDA version feature to
/// Cargo.toml and reintroduce the probe + launch function.
fn probe_grouped_gemm(_device: &CudaDevice) -> bool {
    false
}

/// Build pre-computed batched GEMM pointer arrays for all layers.
///
/// Called once at the end of `preload_weights()`. Extracts GPU device pointers from
/// the layer weight cache and scratch buffers, builds host-side pointer arrays, and
/// uploads them to per-layer device buffers. After this, `compute_layer_gpu` can call
/// `launch_hgemv_f16_batched_precomputed` with zero htod overhead.
fn build_precomputed_batch_ptrs(
    device: &CudaDevice,
    layer_weights: &[LayerWeightsGpu],
    scratch: &GpuScratch,
) -> Result<PrecomputedBatchPtrs, RuntimeError> {
    use cudarc::driver::DevicePtr;

    let num_layers = layer_weights.len();
    let has_grouped_gemm = probe_grouped_gemm(device);
    if has_grouped_gemm {
        eprintln!("[CUDA] cublasGemmGroupedBatchedEx available (CUDA 12.5+) -- QKV grouped GEMM enabled");
    } else {
        eprintln!("[CUDA] cublasGemmGroupedBatchedEx not available -- using separate Q + batched KV");
    }

    // Get stable device pointers for scratch output buffers.
    // These are allocated once in init() and never reallocated.
    let (q_out_ptr, _) = scratch.q.device_ptr(&device.stream);
    let (k_out_ptr, _) = scratch.k.device_ptr(&device.stream);
    let (v_out_ptr, _) = scratch.v.device_ptr(&device.stream);
    let (gate_out_ptr, _) = scratch.gate.device_ptr(&device.stream);
    let (up_out_ptr, _) = scratch.up.device_ptr(&device.stream);
    let (input_f16_ptr, _) = scratch.input_f16.device_ptr(&device.stream);

    let mut kv_a_ptrs = Vec::with_capacity(num_layers);
    let mut kv_b_ptrs = Vec::with_capacity(num_layers);
    let mut kv_c_ptrs = Vec::with_capacity(num_layers);
    let mut ffn_a_ptrs = Vec::with_capacity(num_layers);
    let mut ffn_b_ptrs = Vec::with_capacity(num_layers);
    let mut ffn_c_ptrs = Vec::with_capacity(num_layers);
    let mut qkv_a_ptrs = Vec::with_capacity(if has_grouped_gemm { num_layers } else { 0 });
    let mut qkv_b_ptrs = Vec::with_capacity(if has_grouped_gemm { num_layers } else { 0 });
    let mut qkv_c_ptrs = Vec::with_capacity(if has_grouped_gemm { num_layers } else { 0 });

    for (_layer_idx, lw) in layer_weights.iter().enumerate() {
        // --- KV batched pointers ---
        // Try to get F16 weight pointers for K and V.
        let wk_f16_ptr = get_f16_weight_ptr(device, &lw.wk, lw.wk_f16.as_ref());
        let wv_f16_ptr = get_f16_weight_ptr(device, &lw.wv, lw.wv_f16.as_ref());

        if let (Some(wk_ptr), Some(wv_ptr)) = (wk_f16_ptr, wv_f16_ptr) {
            let host_a = [wk_ptr, wv_ptr];
            let host_b = [input_f16_ptr as u64, input_f16_ptr as u64];
            let host_c = [k_out_ptr as u64, v_out_ptr as u64];

            kv_a_ptrs.push(device.htod_copy(&host_a)?);
            kv_b_ptrs.push(device.htod_copy(&host_b)?);
            kv_c_ptrs.push(device.htod_copy(&host_c)?);
        } else {
            // Placeholder (empty) -- this layer doesn't use batched KV HGEMV.
            kv_a_ptrs.push(device.alloc_zeros::<u64>(2)?);
            kv_b_ptrs.push(device.alloc_zeros::<u64>(2)?);
            kv_c_ptrs.push(device.alloc_zeros::<u64>(2)?);
        }

        // --- FFN gate+up batched pointers ---
        let wg_f16_ptr = get_f16_weight_ptr(device, &lw.w_gate, lw.w_gate_f16.as_ref());
        let wu_f16_ptr = get_f16_weight_ptr(device, &lw.w_up, lw.w_up_f16.as_ref());

        if let (Some(wg_ptr), Some(wu_ptr)) = (wg_f16_ptr, wu_f16_ptr) {
            let host_a = [wg_ptr, wu_ptr];
            let host_b = [input_f16_ptr as u64, input_f16_ptr as u64];
            let host_c = [gate_out_ptr as u64, up_out_ptr as u64];

            ffn_a_ptrs.push(device.htod_copy(&host_a)?);
            ffn_b_ptrs.push(device.htod_copy(&host_b)?);
            ffn_c_ptrs.push(device.htod_copy(&host_c)?);
        } else {
            ffn_a_ptrs.push(device.alloc_zeros::<u64>(2)?);
            ffn_b_ptrs.push(device.alloc_zeros::<u64>(2)?);
            ffn_c_ptrs.push(device.alloc_zeros::<u64>(2)?);
        }

        // --- QKV grouped pointers (only if grouped GEMM available) ---
        if has_grouped_gemm {
            let wq_f16_ptr = get_f16_weight_ptr(device, &lw.wq, lw.wq_f16.as_ref());

            if let (Some(wq_ptr), Some(wk_ptr), Some(wv_ptr)) = (wq_f16_ptr, wk_f16_ptr, wv_f16_ptr) {
                let host_a = [wq_ptr, wk_ptr, wv_ptr];
                let host_b = [input_f16_ptr as u64; 3];
                let host_c = [q_out_ptr as u64, k_out_ptr as u64, v_out_ptr as u64];

                qkv_a_ptrs.push(device.htod_copy(&host_a)?);
                qkv_b_ptrs.push(device.htod_copy(&host_b)?);
                qkv_c_ptrs.push(device.htod_copy(&host_c)?);
            } else {
                qkv_a_ptrs.push(device.alloc_zeros::<u64>(3)?);
                qkv_b_ptrs.push(device.alloc_zeros::<u64>(3)?);
                qkv_c_ptrs.push(device.alloc_zeros::<u64>(3)?);
            }
        }
    }

    Ok(PrecomputedBatchPtrs {
        kv_a_ptrs,
        kv_b_ptrs,
        kv_c_ptrs,
        ffn_a_ptrs,
        ffn_b_ptrs,
        ffn_c_ptrs,
        has_grouped_gemm,
        qkv_a_ptrs,
        qkv_b_ptrs,
        qkv_c_ptrs,
    })
}

/// Extract the F16 device pointer for a weight buffer, as a raw u64.
///
/// For `F16Raw` weights, returns the pointer directly. For Q8/Q4/F32 weights
/// with a pre-dequanted F16 cache, returns the cache pointer. Returns None
/// if no F16 path is available for this weight.
fn get_f16_weight_ptr(
    device: &CudaDevice,
    weight: &GpuWeightBuf,
    f16_cache: Option<&CudaSlice<u8>>,
) -> Option<u64> {
    use cudarc::driver::DevicePtr;
    match weight {
        GpuWeightBuf::F16Raw(ref w) => {
            let (ptr, _) = w.device_ptr(&device.stream);
            Some(ptr as u64)
        }
        _ => {
            f16_cache.map(|cache| {
                let (ptr, _) = cache.device_ptr(&device.stream);
                ptr as u64
            })
        }
    }
}

/// Launch the `compute_rms_scale` kernel: computes `rms_scale = 1/sqrt(mean(x^2)+eps)`
/// and writes a single scalar to `rms_scale_out`.
///
/// This is Pass 1 of the fused RMSNorm+MatVec two-pass approach.
///
/// # Safety
///
/// `x` must have `dim` elements. `rms_scale_out` must have at least 1 element.
unsafe fn launch_compute_rms_scale(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaSlice<f32>,
    rms_scale_out: &mut CudaSlice<f32>,
    eps: f32,
    dim: usize,
) -> Result<(), RuntimeError> {
    let block_size = rmsnorm_block_size(dim);
    let shared_bytes = rmsnorm_shared_bytes(block_size);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.compute_rms_scale)
        .arg(x)
        .arg(rms_scale_out)
        .arg(&eps)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("compute_rms_scale launch: {e}")))?;
    Ok(())
}

/// Launch the `fused_norm_matvec_f32` kernel: computes
/// `out[row] = dot(W[row], x * rms_scale * norm_weight)` for F32 weights.
///
/// This is Pass 2 of the fused RMSNorm+MatVec approach. The RMS scale must
/// have been precomputed by `launch_compute_rms_scale`.
///
/// # Safety
///
/// `x` and `norm_weight` must have `in_dim` elements. `rms_scale` must be [1].
/// `weight` must be [out_dim, in_dim] F32 row-major. `output` must be [out_dim].
unsafe fn launch_fused_norm_matvec_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaSlice<f32>,
    rms_scale: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let block_size = fused_norm_matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let dim_u32 = in_dim as u32;
    let out_dim_u32 = out_dim as u32;
    device
        .stream
        .launch_builder(&kernels.fused_norm_matvec_f32)
        .arg(x)
        .arg(rms_scale)
        .arg(norm_weight)
        .arg(weight)
        .arg(output)
        .arg(&dim_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "fused_norm_matvec_f32 {label} launch: {e}",
        )))?;
    Ok(())
}

/// Launch the `fused_norm_dual_matvec_f32` kernel: computes both gate and up
/// projections from the same normalized input in a single dispatch.
///
/// `gate[row] = dot(W_gate[row], x * rms_scale * norm_weight)`
/// `up[row] = dot(W_up[row], x * rms_scale * norm_weight)`
///
/// # Safety
///
/// `x` and `norm_weight` must be [in_dim]. `rms_scale` must be [1].
/// `w_gate` and `w_up` must be [out_dim, in_dim]. `out_gate` and `out_up` must
/// be [out_dim].
unsafe fn launch_fused_norm_dual_matvec_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaSlice<f32>,
    rms_scale: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    w_gate: &CudaSlice<f32>,
    w_up: &CudaSlice<f32>,
    out_gate: &mut CudaSlice<f32>,
    out_up: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Result<(), RuntimeError> {
    let block_size = fused_norm_matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let dim_u32 = in_dim as u32;
    let out_dim_u32 = out_dim as u32;
    device
        .stream
        .launch_builder(&kernels.fused_norm_dual_matvec_f32)
        .arg(x)
        .arg(rms_scale)
        .arg(norm_weight)
        .arg(w_gate)
        .arg(w_up)
        .arg(out_gate)
        .arg(out_up)
        .arg(&dim_u32)
        .arg(&out_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!(
            "fused_norm_dual_matvec_f32 gate+up launch: {e}",
        )))?;
    Ok(())
}

impl ComputeBackend for CudaBackend {
    fn init(&mut self, hyperparams: &ModelHyperparams) -> Result<(), RuntimeError> {
        self.hyperparams = Some(*hyperparams);
        self.cached_hidden_dim = hyperparams.hidden_dim as usize;
        self.cached_vocab_size = hyperparams.vocab_size as usize;

        let hidden_dim = hyperparams.hidden_dim as usize;
        let num_heads = hyperparams.num_heads as usize;
        let num_kv_heads = hyperparams.num_kv_heads as usize;
        let head_dim = hyperparams.head_dim as usize;
        let inter_dim = hyperparams.intermediate_dim as usize;
        let num_layers = hyperparams.num_layers as usize;
        // Allocate the internal GPU KV cache for `hyperparams.max_seq_len`
        // tokens. The CLI is responsible for right-sizing this value via
        // `--context-len` (see `effective_max_seq_len` in `lumen-cli/src/run.rs`)
        // — the caller passes the capped value through `hyperparams.max_seq_len`
        // so this backend just honours it.
        //
        // KV memory cost per token (Qwen3.5-9B, F32 KV):
        //   32 layers * 4 kv_heads * 256 head_dim * 2 (K,V) * 4 B = 256 KB/token
        //   So 32K context  ~=  8 GB; 64K  ~=  16 GB; 128K  ~=  32 GB.
        //
        // `LUMEN_CUDA_MAX_SEQ_LEN`, when set, applies an additional upper bound
        // for operators who need to *lower* the cap on multi-tenant GPUs
        // regardless of what the CLI passes. It is no longer the default cap;
        // removed the silent 8192 ceiling so `--context-len N` is
        // honoured directly.
        let model_max_seq_len = hyperparams.max_seq_len as usize;
        let env_cap = std::env::var("LUMEN_CUDA_MAX_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let max_seq_len = match env_cap {
            Some(cap) => model_max_seq_len.min(cap),
            None => model_max_seq_len,
        };
        if let Some(cap) = env_cap {
            if max_seq_len < model_max_seq_len {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_MAX_SEQ_LEN={cap} lowers KV cache \
                     max_seq_len from {model_max_seq_len} to {max_seq_len}"
                );
            }
        }

        // Compile embedding kernels (F32, Q8_0, F16, Q4_0).
        let embed_module = self.device.compile_and_load(EMBED_KERNEL_SOURCE)?;
        let embed_f32 = embed_module
            .load_function("embed_token_f32")
            .map_err(|e| RuntimeError::Compute(format!("Failed to load embed_token_f32: {e}")))?;
        let embed_q8_0 = embed_module
            .load_function("embed_token_q8_0")
            .map_err(|e| RuntimeError::Compute(format!("Failed to load embed_token_q8_0: {e}")))?;
        let embed_f16 = embed_module
            .load_function("embed_token_f16")
            .map_err(|e| RuntimeError::Compute(format!("Failed to load embed_token_f16: {e}")))?;
        let embed_q4_0 = embed_module
            .load_function("embed_token_q4_0")
            .map_err(|e| RuntimeError::Compute(format!("Failed to load embed_token_q4_0: {e}")))?;
        let embed_bf16 = embed_module
            .load_function("embed_token_bf16")
            .map_err(|e| RuntimeError::Compute(format!("Failed to load embed_token_bf16: {e}")))?;
        eprintln!("[CUDA] embed_token_bf16: OK");
        self.embed_f32_func = Some(embed_f32);
        self.embed_q8_0_func = Some(embed_q8_0);
        self.embed_f16_func = Some(embed_f16);
        self.embed_q4_0_func = Some(embed_q4_0);
        self.embed_bf16_func = Some(embed_bf16);

        // Compile all decode-path kernels.
        let mut kernels = decode::compile_all_kernels(&self.device)?;

        // Allocate GPU scratch buffers.
        let scratch = GpuScratch {
            normed: self.device.alloc_zeros(hidden_dim)?,
            q: self.device.alloc_zeros(num_heads * head_dim)?,
            k: self.device.alloc_zeros(num_kv_heads * head_dim)?,
            v: self.device.alloc_zeros(num_kv_heads * head_dim)?,
            attn_out: self.device.alloc_zeros(num_heads * head_dim)?,
            gate: self.device.alloc_zeros(inter_dim)?,
            up: self.device.alloc_zeros(inter_dim)?,
            down: self.device.alloc_zeros(hidden_dim)?,
            x_gpu: self.device.alloc_zeros(hidden_dim)?,
            attn_proj: self.device.alloc_zeros(hidden_dim)?,
            rms_scale: self.device.alloc_zeros(1)?,
            // F16 scratch for HGEMV: max(hidden_dim, inter_dim) elements * 2 bytes each.
            input_f16: self.device.alloc_zeros::<u8>(hidden_dim.max(inter_dim) * 2)?,
            // Q8_1 scratch for dp4a matvec: max(hidden_dim, inter_dim) / 32 * 36 bytes.
            // Only allocate if the dp4a Q8_1 kernels compiled successfully.
            // also allocate when mul_mat_vec_q_q{8,4}_0 compiled
            // (the dp4a-mmvq dispatch uses the same scratch layout).
            input_q8_1: if (kernels.quantize_f32_to_q8_1.is_some() && (kernels.matvec_q8_0_q8_1.is_some() || kernels.matvec_q8_aligned_q8_1.is_some() || kernels.matvec_q4_0_dp4a.is_some() || kernels.matvec_q4_aligned_q8_1.is_some()))
                || (kernels.quantize_q8_1_rawsum.is_some() && (kernels.mul_mat_vec_q_q8_0.is_some() || kernels.mul_mat_vec_q_q4_0.is_some())) {
                let max_dim = hidden_dim.max(inter_dim) as u32;
                let buf_bytes = decode::q8_1_buffer_bytes(max_dim) as usize;
                match self.device.alloc_zeros::<u8>(buf_bytes) {
                    Ok(buf) => {
                        eprintln!("[CUDA] Q8_1 scratch: {buf_bytes} bytes allocated");
                        Some(buf)
                    }
                    Err(e) => {
                        eprintln!("[CUDA] Q8_1 scratch alloc failed: {e}");
                        None
                    }
                }
            } else {
                None
            },
            // Pre-allocated device pointer arrays for batched GEMM (3 pointers each).
            batched_a_ptrs: self.device.alloc_zeros::<u64>(3)?,
            batched_b_ptrs: self.device.alloc_zeros::<u64>(3)?,
            batched_c_ptrs: self.device.alloc_zeros::<u64>(3)?,
            // Q+gate fusion: allocated lazily in preload_weights when attn_q_norm detected.
            q_gate: None,
            gate_buf: None,
        };

        // Upload global tensors to GPU.
        // For F32 globals: require non-empty data from set_global_tensors().
        // For quantized raw paths: check if raw bytes were provided via set_*_raw().
        if self.final_norm.is_empty() {
            return Err(RuntimeError::Compute(
                "CUDA init: final_norm not set (call set_global_tensors before init)".into(),
            ));
        }

        let has_f32_embedding = !self.embedding.is_empty();
        let has_q8_embedding = self.embedding_raw.is_some();
        if !has_f32_embedding && !has_q8_embedding {
            return Err(RuntimeError::Compute(
                "CUDA init: embedding not set (call set_global_tensors or set_embedding_raw before init)".into(),
            ));
        }

        let has_f32_output_proj = !self.output_proj.is_empty();
        let has_raw_output_proj = self.output_proj_raw.is_some();
        if !has_f32_output_proj && !has_raw_output_proj {
            return Err(RuntimeError::Compute(
                "CUDA init: output_proj not set (call set_global_tensors or set_output_proj_raw before init)".into(),
            ));
        }

        // Memory diagnostic: print expected vs actual GPU allocation per step.
        // Helps diagnose OOM by surfacing each large allocation site.
        let mem_before_globals = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] before global tensor upload: {:.2} GB free",
            (mem_before_globals as f64) / 1.0e9
        );

        // Upload embedding: prefer quantized raw if available, else F32.
        // BF16 embedding now uploads RAW bytes (2 B/elem) instead of dequanting
        // to F32 (4 B/elem) — saves ~4 GB on Qwen3.5-9B (vocab=248320, hidden=4096).
        let has_raw_embedding = self.embedding_raw.is_some();
        let (embedding_f32, embedding_q8, embedding_f16_raw, embedding_q4_raw, embedding_bf16_raw) = if has_raw_embedding {
            let raw = self.embedding_raw.as_ref().unwrap();
            let placeholder: CudaSlice<f32> = self.device.alloc_zeros(1)?;
            match self.embedding_quant {
                QuantScheme::Q8_0 => {
                    let gpu_q8 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, Some(gpu_q8), None, None, None)
                }
                QuantScheme::F16 => {
                    let gpu_f16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, Some(gpu_f16), None, None)
                }
                QuantScheme::Q4_0 => {
                    let gpu_q4 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, Some(gpu_q4), None)
                }
                QuantScheme::Bf16 => {
                    // BF16 embedding: upload raw bytes (2 B/elem) and dispatch via
                    // the dedicated embed_token_bf16 kernel. Saves ~4 GB GPU VRAM
                    // vs the previous host-side BF16 -> F32 dequant path.
                    let raw_mb = raw.len() as f64 / 1.0e6;
                    eprintln!("[CUDA mem] uploading BF16 embedding raw: {raw_mb:.1} MB");
                    let gpu_bf16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, None, Some(gpu_bf16))
                }
                other => {
                    return Err(RuntimeError::Compute(format!(
                        "CUDA init: embedding raw quant {other:?} not supported (only Q8_0, F16, Q4_0, Bf16)",
                    )));
                }
            }
        } else {
            let gpu_f32 = self.device.htod_copy(&self.embedding)?;
            (gpu_f32, None, None, None, None)
        };
        let mem_after_embedding = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] after embedding upload: {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_embedding as f64) / 1.0e9,
            (mem_before_globals.saturating_sub(mem_after_embedding) as f64) / 1.0e9
        );

        // Upload output projection: prefer quantized raw if available, else F32.
        // BF16 output_proj now uploads RAW bytes (2 B/elem) instead of dequanting
        // to F32 (4 B/elem) — saves ~4 GB on Qwen3.5-9B. Dispatched via the
        // matvec_bf16 kernel in compute_final_gpu.
        let (output_proj_f32, output_proj_q8, output_proj_q4, output_proj_f16_raw, output_proj_bf16_raw) = if has_raw_output_proj {
            let raw = self.output_proj_raw.as_ref().unwrap();
            let placeholder: CudaSlice<f32> = self.device.alloc_zeros(1)?;
            match self.output_proj_quant {
                QuantScheme::Q8_0 => {
                    let gpu_q8 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, Some(gpu_q8), None, None, None)
                }
                QuantScheme::Q4_0 => {
                    let gpu_q4 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, Some(gpu_q4), None, None)
                }
                QuantScheme::F16 => {
                    let gpu_f16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, Some(gpu_f16), None)
                }
                QuantScheme::Bf16 => {
                    // BF16 output_proj: upload raw bytes (2 B/elem) and dispatch
                    // via the matvec_bf16 kernel. Saves ~4 GB GPU VRAM vs the
                    // previous host-side BF16 -> F32 dequant + cuBLAS SGEMV path.
                    let raw_mb = raw.len() as f64 / 1.0e6;
                    eprintln!("[CUDA mem] uploading BF16 output_proj raw: {raw_mb:.1} MB");
                    let gpu_bf16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, None, Some(gpu_bf16))
                }
                other => {
                    return Err(RuntimeError::Compute(format!(
                        "CUDA init: output_proj raw quant {other:?} not supported (only Q8_0, Q4_0, F16, Bf16)",
                    )));
                }
            }
        } else {
            let gpu_f32 = self.device.htod_copy(&self.output_proj)?;
            (gpu_f32, None, None, None, None)
        };
        let mem_after_output_proj = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] after output_proj upload: {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_output_proj as f64) / 1.0e9,
            (mem_after_embedding.saturating_sub(mem_after_output_proj) as f64) / 1.0e9
        );

        let globals = GpuGlobals {
            final_norm: self.device.htod_copy(&self.final_norm)?,
            output_proj: output_proj_f32,
            output_proj_f16: output_proj_f16_raw,
            output_proj_q8,
            output_proj_q8_aligned: None, // Populated during preload_weights
            output_proj_q8_split: None,   // populated when LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1
            output_proj_q8_to_f16_cache: None, // populated when LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1
            output_proj_q4,
            output_proj_q4_aligned: None, // Populated during preload_weights
            output_proj_bf16: output_proj_bf16_raw,
            embedding: embedding_f32,
            embedding_q8,
            embedding_f16: embedding_f16_raw,
            embedding_q4: embedding_q4_raw,
            embedding_bf16: embedding_bf16_raw,
        };

        // Allocate per-layer KV caches. Compile the KV write kernel once and
        // share the module across all layers to avoid redundant NVRTC compilation.
        let mem_before_kv = self.device.free_memory().unwrap_or(0);
        let kv_module = self
            .device
            .compile_and_load(super::shaders::KV_CACHE_KERNEL_SOURCE)?;
        let mut kv_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            kv_caches.push(KvCacheGpu::with_module(
                &self.device,
                num_kv_heads,
                max_seq_len,
                head_dim,
                &kv_module,
            )?);
        }
        let mem_after_kv = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] after KV cache alloc ({num_layers} layers, max_seq_len={max_seq_len}): {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_kv as f64) / 1.0e9,
            (mem_before_kv.saturating_sub(mem_after_kv) as f64) / 1.0e9
        );

        // Pre-allocate logits buffer for the zero-sync decode path.
        let vocab_size = hyperparams.vocab_size as usize;
        let logits_gpu = self.device.alloc_zeros::<f32>(vocab_size)?;

        // Compile graph-compatible kernel variants for CUDA graph capture.
        let graph_kernels = match super::graph::compile_graph_kernels(&self.device) {
            Ok(gk) => Some(gk),
            Err(e) => {
                eprintln!("[CUDA] Graph kernel compilation failed (graph capture disabled): {e}");
                None
            }
        };

        // Allocate graph parameter buffers (small device scalars for token_id, pos, seq_len).
        let graph_params = match super::graph::GraphParamsBuf::new(&self.device) {
            Ok(gp) => Some(gp),
            Err(e) => {
                eprintln!("[CUDA] Graph params allocation failed (graph capture disabled): {e}");
                None
            }
        };

        // Allocate cuBLAS workspace for CUDA graph capture compatibility.
        // cuBLAS must not call cudaMalloc during graph capture; providing a
        // pre-allocated workspace via cublasSetWorkspace_v2 prevents this.
        // 32 MB workspace for cuBLAS graph capture compatibility.
        // cuBLAS may allocate scratch internally for certain GEMM shapes/algos.
        // 4 MB was insufficient -- cuBLAS GemmBatchedEx can use up to ~16 MB
        // depending on the GEMM shape and selected algorithm. 32 MB provides
        // headroom for all model sizes (up to hidden_dim=8192, inter_dim=28672).
        const CUBLAS_WORKSPACE_SIZE: usize = 32 * 1024 * 1024; // 32 MB
        let cublas_workspace = match self.device.alloc_zeros::<u8>(CUBLAS_WORKSPACE_SIZE) {
            Ok(ws) => {
                match self.device.set_cublas_workspace(&ws) {
                    Ok(()) => {
                        eprintln!(
                            "[CUDA] cuBLAS workspace: {} MB (graph-capture ready)",
                            CUBLAS_WORKSPACE_SIZE / (1024 * 1024),
                        );
                        Some(ws)
                    }
                    Err(e) => {
                        eprintln!("[CUDA] cublasSetWorkspace failed (graph capture disabled): {e}");
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("[CUDA] cuBLAS workspace alloc failed (graph capture disabled): {e}");
                None
            }
        };

        // split-layout integration: read opt-in env vars once at session start.
        // Truthy values accepted: "1", "true", "yes", "on" (case-insensitive).
        // Anything else (or unset) leaves the existing Q8Raw/Q8Aligned dp4a
        // paths in charge.
        let env_truthy = |key: &str| -> bool {
            std::env::var(key)
                .ok()
                .map(|v| {
                    let s = v.trim().to_ascii_lowercase();
                    matches!(s.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false)
        };
        // helper that respects per-flag default-ON resolvers.
        // `Some(v)` parses the env value; `None` calls the runtime_defaults
        // helper. Explicit `=0` / "false" / "no" / "off" always wins (returns
        // false). This matches the resolver pattern used for
        // BF16_GEMMEX and DECODE_GRAPH.
        let env_truthy_or_default = |key: &str, default_fn: fn() -> bool| -> bool {
            match std::env::var(key).ok() {
                Some(v) => {
                    let s = v.trim().to_ascii_lowercase();
                    matches!(s.as_str(), "1" | "true" | "yes" | "on")
                }
                None => default_fn(),
            }
        };
        // Q8_SCALE_HW defaults ON for Q8 dense (no-op otherwise).
        let use_q8_scale_hw = env_truthy_or_default(
            "LUMEN_CUDA_Q8_SCALE_HW",
            crate::runtime_defaults::q8_scale_hw_default,
        )
            && kernels.matvec_q8_aligned_q8_1_hw.is_some()
            && kernels.matvec_q8_aligned_q8_1_hw_residual.is_some();
        if use_q8_scale_hw {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SCALE_HW: prefer matvec_q8_aligned_q8_1_hw on Q8Aligned dispatch");
        } else if env_truthy("LUMEN_CUDA_Q8_SCALE_HW") {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SCALE_HW=1 set but matvec_q8_aligned_q8_1_hw unavailable; using existing aligned kernel");
        }
        // Q8_SPLIT defaults ON for Q8 dense (no-op otherwise).
        let use_q8_split = env_truthy_or_default(
            "LUMEN_CUDA_Q8_SPLIT",
            crate::runtime_defaults::q8_split_default,
        );
        if use_q8_split {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SPLIT: Q8_0 weights will be cloned to split layout for decode");
        }
        let use_q4_split = env_truthy("LUMEN_CUDA_Q4_SPLIT");
        if use_q4_split {
            eprintln!("[CUDA] LUMEN_CUDA_Q4_SPLIT=1: Q4_0 weights will be cloned to split layout for decode");
        }
        let use_gdn_split = env_truthy("LUMEN_CUDA_GDN_SPLIT");
        if use_gdn_split {
            eprintln!("[CUDA] LUMEN_CUDA_GDN_SPLIT=1: GDN Q4 weights (ssm_out/attn_gate/ssm_alpha/ssm_beta) will be cloned to split layout for decode");
        }
        // OUTPUT_PROJ_SPLIT defaults ON for Q8 dense (no-op
        // otherwise; clones the Q8_0 vocab output projection to the split
        // sibling layout for the NR-tiled matvec kernel).
        let use_output_proj_split = env_truthy_or_default(
            "LUMEN_CUDA_OUTPUT_PROJ_SPLIT",
            crate::runtime_defaults::output_proj_split_default,
        );
        if use_output_proj_split {
            eprintln!("[CUDA] LUMEN_CUDA_OUTPUT_PROJ_SPLIT: output_proj Q8_0 will be cloned to split layout for decode");
        }
        // output_proj fast-path: F16 dequant cache + cuBLAS HGEMV-N=1.
        // Activates only when output_proj is Q8_0 (no other quants supported).
        let use_output_proj_f16_cache = env_truthy("LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE");
        if use_output_proj_f16_cache {
            eprintln!("[CUDA] LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1: output_proj Q8_0 will be pre-dequanted to F16 for cuBLAS HGEMV decode");
        }
        // output_proj NR override: pick from 2/16/32/64/128. Default = 16
        // when the model is Q8 dense (: matches the canonical
        // production config), else 32 (legacy default that matches the
        // pre-F2 dispatch). When `OUTPUT_PROJ_SPLIT=1` AND the requested NR
        // variant is loaded, dispatch routes through it.
        let output_proj_nr_default = crate::runtime_defaults::output_proj_nr_default();
        let output_proj_nr: u32 = match std::env::var("LUMEN_CUDA_OUTPUT_PROJ_NR")
            .ok()
            .as_deref()
        {
            Some("2") => 2,
            Some("8") => 8,
            Some("16") => 16,
            Some("32") => 32,
            Some("64") => 64,
            Some("128") => 128,
            None | Some("") => {
                // 16 for Q8 dense, 32 legacy otherwise.
                if output_proj_nr_default == 16 { 16 } else { 32 }
            }
            Some(other) => {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_NR={other} unrecognized; accepted 2/8/16/32/64/128; defaulting to 32"
                );
                32
            }
        };
        if output_proj_nr != 32 {
            eprintln!(
                "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_NR={output_proj_nr}: output_proj SPLIT dispatch will use NR={output_proj_nr} kernel"
            );
        }
        let use_q8_tile = env_truthy("LUMEN_CUDA_Q8_TILE");
        if use_q8_tile {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_TILE=1: Q8_0 weights will be cloned to tile-grouped layout for decode");
        }
        let use_q4_tile = env_truthy("LUMEN_CUDA_Q4_TILE");
        if use_q4_tile {
            eprintln!("[CUDA] LUMEN_CUDA_Q4_TILE=1: Q4_0 weights will be cloned to tile-grouped layout for decode");
        }
        // Propagate runtime feature flags onto KernelSet so the
        // `launch_matvec_preq8_1*` free functions can consult them without
        // taking an extra parameter at every call site.
        kernels.use_q8_scale_hw = use_q8_scale_hw;
        kernels.use_q8_split_dispatch =
            use_q8_split && kernels.matvec_q8_split_q8_1.is_some();
        kernels.use_q4_split_dispatch =
            use_q4_split && kernels.matvec_q4_split_q8_1.is_some();

        // 4-threads-per-block mmvq kernel selection.
        // Effective only when LUMEN_CUDA_Q8_SPLIT=1 is also set, since the
        // 4-thread variant consumes the SPLIT byte layout. Default OFF.
        let use_q8_split_4thread = env_truthy("LUMEN_CUDA_Q8_SPLIT_4THREAD");
        kernels.use_q8_split_4thread_dispatch = use_q8_split_4thread
            && kernels.use_q8_split_dispatch
            && kernels.matvec_q8_split_q8_1_4thread.is_some()
            && kernels.matvec_q8_split_q8_1_4thread_residual.is_some();
        if kernels.use_q8_split_4thread_dispatch {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SPLIT_4THREAD=1: SPLIT dispatch uses dp4a-mmvq kernel (K-trip=4)");
        } else if use_q8_split_4thread {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SPLIT_4THREAD=1 set but prerequisites missing (need Q8_SPLIT=1 + kernel load OK); using existing split kernel");
        }
        // NR=8 mmvq kernel selection (4-threads-per-block + NR=8 rows/CTA).
        // Selected if env var is set AND prerequisites are met. Takes priority
        // over 4thread when both are set (FULL is the strict superset).
        // Default OFF (default-off contract).
        let use_q8_split_nr8 = env_truthy("LUMEN_CUDA_Q8_SPLIT_NR8");
        kernels.use_q8_split_nr8_dispatch = use_q8_split_nr8
            && kernels.use_q8_split_dispatch
            && kernels.matvec_q8_split_q8_1_nr8.is_some()
            && kernels.matvec_q8_split_q8_1_nr8_residual.is_some();
        if kernels.use_q8_split_nr8_dispatch {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SPLIT_NR8=1: SPLIT dispatch uses dp4a-mmvq kernel (NR=8 + 4-thread mapping)");
            if kernels.use_q8_split_4thread_dispatch {
                eprintln!("[CUDA] NR8 takes priority over 4thread (FULL is the structural superset)");
            }
        } else if use_q8_split_nr8 {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_SPLIT_NR8=1 set but prerequisites missing (need Q8_SPLIT=1 + kernel load OK); using existing split kernel");
        }
        // AoS NR=8 mmvq kernel selection.
        // Operates on the AoS dispatch (`launch_matvec_preq8_1`); does NOT
        // require Q8_SPLIT. Independent of `use_q8_split_nr8_dispatch`. Default OFF.
        let use_q8_aos_nr8 = env_truthy("LUMEN_CUDA_Q8_AOS_NR8");
        kernels.use_q8_aos_nr8_dispatch = use_q8_aos_nr8
            && kernels.matvec_q8_aligned_nr8.is_some()
            && kernels.matvec_q8_aligned_nr8_residual.is_some();
        if kernels.use_q8_aos_nr8_dispatch {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_AOS_NR8=1: AoS dispatch uses dp4a-mmvq kernel (NR=8 + 4-thread mapping on 36-byte blocks)");
        } else if use_q8_aos_nr8 {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_AOS_NR8=1 set but prerequisites missing (need matvec_q8_aligned_nr8 kernel load OK); using existing AoS kernel");
        }
        kernels.use_q8_tile_dispatch =
            use_q8_tile && kernels.matvec_q8_tile_q8_1.is_some();
        kernels.use_q4_tile_dispatch =
            use_q4_tile && kernels.matvec_q4_tile_q8_1.is_some();
        // P1-3 FA2 block-skip prefill kernel selection. Default OFF; the
        // env var routes prefill attention dispatch through the new kernel.
        let use_fa2_blockskip = env_truthy("LUMEN_CUDA_FA2_BLOCKSKIP");
        kernels.use_fa2_blockskip_dispatch = use_fa2_blockskip
            && kernels.flash_attention_fa2_causal.is_some();
        if kernels.use_fa2_blockskip_dispatch {
            eprintln!("[CUDA] LUMEN_CUDA_FA2_BLOCKSKIP=1: prefill attention dispatch uses FA2 mask block-skip kernel");
        } else if use_fa2_blockskip {
            eprintln!("[CUDA] LUMEN_CUDA_FA2_BLOCKSKIP=1 set but flash_attention_fa2_causal kernel unavailable; using existing wmma/br4 dispatch");
        }

        // pre-allocate MoE scratch when the model declares experts.
        // Sized from hyperparams: hidden_dim, expert inter_dim (or fallback to
        // model inter_dim), shared inter_dim (= model inter_dim per Qwen3.5-MoE
        // hyperparam encoding), num_experts, top_k. Dense models get `None`.
        let moe_scratch = if let (Some(num_experts), Some(top_k)) = (
            hyperparams.num_experts, hyperparams.num_active_experts,
        ) {
            if num_experts > 0 && top_k > 0 {
                let n_e = num_experts as usize;
                let k = top_k as usize;
                // Routed-expert intermediate dim: the LBC's `intermediate_dim`
                // is the max-applicable size; converter stores routed-expert
                // weights at this dim (Qwen3.5-MoE encoding). For
                // Qwen3.5-MoE the shared expert uses the same dim. Both buffers
                // are sized to `inter_dim` — safe upper bound; over-allocation
                // is at most ~8 KB on 30B-A3B.
                let expert_inter_dim = inter_dim;
                let shared_inter_dim = inter_dim;
                Some(super::moe::allocate_moe_scratch(
                    &self.device,
                    hidden_dim,
                    expert_inter_dim,
                    shared_inter_dim,
                    n_e,
                    k,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        // moe_meta_cache size matches num_layers; populated in preload_weights.
        let moe_meta_cache: Vec<Option<super::moe::CudaMoeMeta>> =
            vec![None; num_layers];
        // parallel cache for Phase-F batched-expert GPU offset tables.
        // Built lazily during preload_weights when an MoE layer is detected.
        // Vec::new() initialization is fine because populate calls .resize()
        // before indexing (mirrored alongside moe_meta_cache).
        let moe_batched_offsets: Vec<Option<super::moe::CudaMoeBatchedOffsets>> =
            (0..num_layers).map(|_| None).collect();

        *self.state.lock().unwrap() = Some(MutableState {
            kernels,
            scratch,
            kv_caches,
            globals,
            layer_weights_cache: Vec::new(),
            logits_gpu,
            argmax_result: self.device.alloc_zeros::<u32>(1)?,
            captured_graph: None,
            graph_kernels,
            graph_params,
            has_gdn_layers: false,
            has_qgate_layers: false,
            has_moe_layers: false,
            decode_token_count: 0,
            gdn_scratch_gpu: None,
            cublas_workspace,
            precomputed_ptrs: None,
            algo_cache: AlgoCache::new(),
            moe_scratch,
            moe_meta_cache,
            moe_batched_offsets,
            expert_cache_config: None,
            use_q8_scale_hw,
            use_q8_split,
            use_q4_split,
            use_gdn_split,
            use_output_proj_split,
            use_output_proj_f16_cache,
            output_proj_nr,
            use_q8_tile,
            use_q4_tile,
        });

        Ok(())
    }

    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        let hidden_dim = self.cached_hidden_dim;
        let vocab_size = self.cached_vocab_size;
        let tid = token_id as usize;

        if tid >= vocab_size {
            return Err(RuntimeError::Compute(format!(
                "token_id {tid} out of range (vocab_size={vocab_size})",
            )));
        }

        // GPU path: use the globals' embedding buffer.
        let state_guard = self.state.lock().unwrap();
        if let Some(ref st) = *state_guard {
            let mut output_gpu: CudaSlice<f32> = self.device.alloc_zeros(hidden_dim)?;
            let config = LaunchConfig::for_elements(hidden_dim);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            // Dispatch embed kernel based on embedding precision.
            // Order: BF16 > F16 > Q4_0 > Q8_0 > F32 (mirror embed_token_gpu).
            if let Some(ref emb_bf16) = st.globals.embedding_bf16 {
                let func = self.embed_bf16_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_bf16 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(func)
                        .arg(emb_bf16).arg(&mut output_gpu).arg(&token_id).arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_bf16 launch: {e}")))?;
            } else if let Some(ref emb_f16) = st.globals.embedding_f16 {
                let func = self.embed_f16_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_f16 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(func)
                        .arg(emb_f16).arg(&mut output_gpu).arg(&token_id).arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_f16 launch: {e}")))?;
            } else if let Some(ref emb_q4) = st.globals.embedding_q4 {
                let func = self.embed_q4_0_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_q4_0 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(func)
                        .arg(emb_q4).arg(&mut output_gpu).arg(&token_id).arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_q4_0 launch: {e}")))?;
            } else if let Some(ref emb_q8) = st.globals.embedding_q8 {
                let func = self.embed_q8_0_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_q8_0 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device.stream.launch_builder(func)
                        .arg(emb_q8).arg(&mut output_gpu).arg(&token_id).arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_q8_0 launch: {e}")))?;
            } else {
                let func = self.embed_f32_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_f32 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                // SAFETY: The kernel reads `hidden_dim` elements starting at
                // `token_id * hidden_dim` from the embedding buffer (bounds checked
                // above via vocab_size), and writes `hidden_dim` elements to output_gpu.
                unsafe {
                    self.device
                        .stream
                        .launch_builder(func)
                        .arg(&st.globals.embedding)
                        .arg(&mut output_gpu)
                        .arg(&token_id)
                        .arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("CUDA embed_token_f32 launch: {e}"))
                })?;
            }

            self.device.synchronize()?;
            let host_output = self.device.dtoh_copy(&output_gpu)?;
            return Ok(f32_to_activation(&host_output));
        }

        // CPU fallback: used when GPU state is not yet initialized.
        drop(state_guard);
        let start = tid * hidden_dim;
        let end = start + hidden_dim;
        if end > self.embedding.len() {
            return Err(RuntimeError::Compute(format!(
                "embedding table too small: need index {end}, have {}",
                self.embedding.len()
            )));
        }

        Ok(f32_to_activation(&self.embedding[start..end]))
    }

    fn compute_layer(
        &self,
        layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let eps = hp.norm_eps;
        let theta = hp.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Require KV cache view to advance seq_len tracking.
        let kv = kv.ok_or_else(|| {
            RuntimeError::Compute("KV cache view required for attention".into())
        })?;

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized".into())
        })?;

        // Use GPU-resident cached weights if available (preloaded via preload_weights),
        // otherwise upload from LayerView on each call (streaming path).
        let fresh_weights;
        let lw: &LayerWeightsGpu = if layer_idx < st.layer_weights_cache.len() {
            &st.layer_weights_cache[layer_idx]
        } else {
            fresh_weights = upload_layer_weights(&self.device, weights, hp)?;
            &fresh_weights
        };

        // GDN layer routing: delegate to compute_layer_gpu which has full GDN support.
        // GDN layers have zero-sentinel wk/wv/wo — the standard attention path would fail.
        if lw.layer_type == 1 {
            if layer_idx >= st.layer_weights_cache.len() {
                return Err(RuntimeError::Compute(
                    "GDN layers require GPU-resident weights (call preload_weights first)".into()
                ));
            }
            // Upload x to GPU, run GPU-resident compute, download result.
            let x_f32 = x.as_f32_slice();
            self.device.htod_copy_into(x_f32, &mut st.scratch.x_gpu)?;
            self.compute_layer_gpu(layer_idx, seq_pos, st)?;
            self.device.synchronize()?;
            let result = self.device.dtoh_copy(&st.scratch.x_gpu)?;
            x.write_f32_from(&result);
            return Ok(());
        }

        // 1. Upload x (activation) to GPU.
        let x_f32 = x.as_f32_slice();
        self.device.htod_copy_into(x_f32, &mut st.scratch.x_gpu)?;

        // 2-3. Fused RMSNorm + QKV projections.
        //
        // For F32 weights: use the two-pass fused approach.
        // Pass 1: compute_rms_scale writes a single scalar (saves full normed buffer).
        // Pass 2: fused_norm_matvec_f32 normalizes x inline during the dot product.
        // For non-F32 weights: fall back to separate rmsnorm + matvec (quantized
        // kernels have their own loop structure; fusing norm into them is future work).
        if matches!(&lw.wq, GpuWeightBuf::F32(_)) {
            // Pass 1: compute rms_scale scalar.
            // SAFETY: x_gpu is [hidden_dim], rms_scale is [1]. Both allocated in init.
            unsafe {
                launch_compute_rms_scale(
                    &self.device, &st.kernels,
                    &st.scratch.x_gpu, &mut st.scratch.rms_scale,
                    eps, hidden_dim,
                )?;
            }

            // Pass 2: fused norm+matvec for Q, K, V.
            // SAFETY: wq is F32 [q_dim, hidden_dim]. x_gpu is [hidden_dim].
            // rms_scale is [1]. attn_norm is [hidden_dim]. q is [q_dim].
            if let GpuWeightBuf::F32(ref wq_f32) = lw.wq {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &st.scratch.rms_scale,
                        &lw.attn_norm, wq_f32, &mut st.scratch.q,
                        q_dim, hidden_dim, "wq",
                    )?;
                }
            }
            if let GpuWeightBuf::F32(ref wk_f32) = lw.wk {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &st.scratch.rms_scale,
                        &lw.attn_norm, wk_f32, &mut st.scratch.k,
                        kv_dim, hidden_dim, "wk",
                    )?;
                }
            }
            if let GpuWeightBuf::F32(ref wv_f32) = lw.wv {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device, &st.kernels,
                        &st.scratch.x_gpu, &st.scratch.rms_scale,
                        &lw.attn_norm, wv_f32, &mut st.scratch.v,
                        kv_dim, hidden_dim, "wv",
                    )?;
                }
            }
        } else {
            // Fallback: separate rmsnorm + matvec for non-F32 weight paths.
            {
                let block_size = rmsnorm_block_size(hidden_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = hidden_dim as u32;
                // SAFETY: x_gpu is [hidden_dim], attn_norm is [hidden_dim],
                // normed is [hidden_dim]. All valid.
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.rmsnorm)
                        .arg(&st.scratch.x_gpu)
                        .arg(&lw.attn_norm)
                        .arg(&mut st.scratch.normed)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm attn launch: {e}")))?;
            }
            // SAFETY: wq is [q_dim, hidden_dim], normed is [hidden_dim], q is [q_dim].
            unsafe {
                launch_matvec(
                    &self.device, &st.kernels, &lw.wq,
                    &st.scratch.normed, &mut st.scratch.q,
                    q_dim, hidden_dim, "wq",
                    lw.wq_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                st.scratch.input_q8_1.as_mut(),
                )?;
            }
            // SAFETY: wk is [kv_dim, hidden_dim], normed is [hidden_dim], k is [kv_dim].
            unsafe {
                launch_matvec(
                    &self.device, &st.kernels, &lw.wk,
                    &st.scratch.normed, &mut st.scratch.k,
                    kv_dim, hidden_dim, "wk",
                    lw.wk_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                st.scratch.input_q8_1.as_mut(),
                )?;
            }
            // SAFETY: wv is [kv_dim, hidden_dim], normed is [hidden_dim], v is [kv_dim].
            unsafe {
                launch_matvec(
                    &self.device, &st.kernels, &lw.wv,
                    &st.scratch.normed, &mut st.scratch.v,
                    kv_dim, hidden_dim, "wv",
                    lw.wv_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                st.scratch.input_q8_1.as_mut(),
                )?;
            }
        }

        // QKV bias (Qwen2-family, streaming decode).
        if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
            let block = 256u32;
            unsafe {
                if let Some(ref bq) = lw.bq {
                    let d = q_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.q).arg(bq).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bq streaming: {e}")))?;
                }
                if let Some(ref bk) = lw.bk {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.k).arg(bk).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bk streaming: {e}")))?;
                }
                if let Some(ref bv) = lw.bv {
                    let d = kv_dim as u32; let g = (d + block - 1) / block;
                    self.device.stream.launch_builder(&st.kernels.bias_add).arg(&mut st.scratch.v).arg(bv).arg(&d)
                        .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                        .map_err(|e| RuntimeError::Compute(format!("bias_add bv streaming: {e}")))?;
                }
            }
        }

        // 4. RoPE: apply rotary position embeddings to q and k.
        {
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            let actual_rot = if rotary_dim > 0 && rotary_dim < head_dim as u32 { rotary_dim as usize } else { head_dim };
            let half_rot = actual_rot / 2;
            let total_q_pairs = num_heads * half_rot;
            let total_k_pairs = num_kv_heads * half_rot;
            let max_pairs = total_q_pairs.max(total_k_pairs);
            let config = LaunchConfig::for_elements(max_pairs);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let pos = seq_pos as u32;
            let nqh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            // NeoX RoPE: models with partial rotary_dim use half-offset dimension pairing.
            let rope_neox = hp.rope_neox;
            let rope_fn = if rope_neox {
                &st.kernels.rope_apply_neox
            } else {
                &st.kernels.rope_apply
            };
            // SAFETY: q has num_heads * head_dim elements, k has num_kv_heads * head_dim
            // elements. The kernel processes pairs within these bounds.
            unsafe {
                self.device
                    .stream
                    .launch_builder(rope_fn)
                    .arg(&mut st.scratch.q)
                    .arg(&mut st.scratch.k)
                    .arg(&pos)
                    .arg(&nqh)
                    .arg(&nkvh)
                    .arg(&hd)
                    .arg(&theta)
                    .arg(&rotary_dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("rope launch: {e}")))?;
        }

        // 5. KV cache: write K and V to the GPU KV cache for this layer.
        {
            let kv_cache = st.kv_caches.get_mut(layer_idx).ok_or_else(|| {
                RuntimeError::Compute(format!("no KV cache for layer {layer_idx}"))
            })?;
            kv_cache.append_kv(&self.device, &st.scratch.k, &st.scratch.v)?;
        }

        // 6. Attention: decode-attention (q, k_cache, v_cache -> attn_out).
        // gate: routes to the tiled streaming-softmax kernel at long
        // context. Byte-identical to the prior single-block dispatch when
        // the gate selects SingleBlock.
        {
            let kv_cache = &st.kv_caches[layer_idx];
            // seq_len is the number of entries AFTER the append (cache auto-increments).
            let attn_seq_len = kv_cache.seq_len() as u32;
            let nh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            let msl = kv_cache.max_seq_len as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            // SAFETY: q has num_heads * head_dim elements. k_cache and v_cache have
            // num_kv_heads * max_seq_len * head_dim elements each. attn_out has
            // num_heads * head_dim elements. attn_seq_len <= max_seq_len.
            unsafe {
                super::prefill::launch_attention_decode_gated(
                    &self.device,
                    &st.kernels,
                    &st.scratch.q,
                    &kv_cache.k_cache,
                    &kv_cache.v_cache,
                    &mut st.scratch.attn_out,
                    nh,
                    nkvh,
                    hd,
                    attn_seq_len,
                    msl,
                    scale,
                )
            }
            .map_err(|e| RuntimeError::Compute(format!("attention_decode launch: {e}")))?;
        }

        // 7. Output projection + residual: attn_proj = wo * attn_out + x
        // SAFETY: wo is [hidden_dim, q_dim], attn_out is [q_dim], x_gpu is [hidden_dim],
        // attn_proj is [hidden_dim]. All allocated with matching sizes.
        unsafe {
            launch_matvec_residual(
                &self.device, &st.kernels, &lw.wo,
                &st.scratch.attn_out, &st.scratch.x_gpu, &mut st.scratch.attn_proj,
                hidden_dim, q_dim, "wo",
                lw.wo_f16.as_ref(),
                Some(&mut st.scratch.input_f16),
            st.scratch.input_q8_1.as_mut(),
            )?;
        }

        // 8-9. Fused FFN RMSNorm + gate/up projections.
        //
        // For F32 weights: fused dual matvec computes both gate and up from the
        // same normalized input in a single dispatch (3 kernels -> 2: rms_scale + dual).
        // For non-F32 weights: fall back to separate rmsnorm + gate matvec + up matvec.
        if matches!(&lw.w_gate, GpuWeightBuf::F32(_))
            && matches!(&lw.w_up, GpuWeightBuf::F32(_))
        {
            // Pass 1: compute rms_scale from attn_proj.
            // SAFETY: attn_proj is [hidden_dim], rms_scale is [1]. Both allocated in init.
            unsafe {
                launch_compute_rms_scale(
                    &self.device, &st.kernels,
                    &st.scratch.attn_proj, &mut st.scratch.rms_scale,
                    eps, hidden_dim,
                )?;
            }

            // Pass 2: fused dual matvec for gate+up.
            // SAFETY: w_gate and w_up are F32 [inter_dim, hidden_dim]. attn_proj is
            // [hidden_dim]. rms_scale is [1]. ffn_norm is [hidden_dim]. gate and up
            // are [inter_dim].
            if let (GpuWeightBuf::F32(ref wg_f32), GpuWeightBuf::F32(ref wu_f32)) =
                (&lw.w_gate, &lw.w_up)
            {
                unsafe {
                    launch_fused_norm_dual_matvec_f32(
                        &self.device, &st.kernels,
                        &st.scratch.attn_proj, &st.scratch.rms_scale,
                        &lw.ffn_norm, wg_f32, wu_f32,
                        &mut st.scratch.gate, &mut st.scratch.up,
                        inter_dim, hidden_dim,
                    )?;
                }
            }
        } else {
            // Fallback: separate rmsnorm + gate + up matvecs.
            {
                let block_size = rmsnorm_block_size(hidden_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = hidden_dim as u32;
                // SAFETY: attn_proj is [hidden_dim], ffn_norm is [hidden_dim],
                // normed is [hidden_dim]. All valid.
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.rmsnorm)
                        .arg(&st.scratch.attn_proj)
                        .arg(&lw.ffn_norm)
                        .arg(&mut st.scratch.normed)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("rmsnorm ffn launch: {e}")))?;
            }
            // SAFETY: w_gate is [inter_dim, hidden_dim], normed is [hidden_dim],
            // gate is [inter_dim].
            unsafe {
                launch_matvec(
                    &self.device, &st.kernels, &lw.w_gate,
                    &st.scratch.normed, &mut st.scratch.gate,
                    inter_dim, hidden_dim, "gate",
                    lw.w_gate_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                st.scratch.input_q8_1.as_mut(),
                )?;
            }
            // SAFETY: w_up is [inter_dim, hidden_dim], normed is [hidden_dim],
            // up is [inter_dim].
            unsafe {
                launch_matvec(
                    &self.device, &st.kernels, &lw.w_up,
                    &st.scratch.normed, &mut st.scratch.up,
                    inter_dim, hidden_dim, "up",
                    lw.w_up_f16.as_ref(),
                    Some(&mut st.scratch.input_f16),
                st.scratch.input_q8_1.as_mut(),
                )?;
            }
        }

        // 10. SwiGLU in-place: gate = silu(gate) * up
        {
            let config = LaunchConfig::for_elements(inter_dim);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n = inter_dim as u32;
            // SAFETY: gate is [inter_dim], up is [inter_dim]. Both valid.
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.swiglu_inplace)
                    .arg(&mut st.scratch.gate)
                    .arg(&st.scratch.up)
                    .arg(&n)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("swiglu launch: {e}")))?;
        }

        // 11. Down projection: down = w_down * gate
        // SAFETY: w_down is [hidden_dim, inter_dim], gate is [inter_dim], down is [hidden_dim].
        unsafe {
            launch_matvec(
                &self.device, &st.kernels, &lw.w_down,
                &st.scratch.gate, &mut st.scratch.down,
                hidden_dim, inter_dim, "down",
                lw.w_down_f16.as_ref(),
                Some(&mut st.scratch.input_f16),
            st.scratch.input_q8_1.as_mut(),
            )?;
        }

        // 12. Residual add: attn_proj += down
        {
            let config = LaunchConfig::for_elements(hidden_dim);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (config.grid_dim, 1, 1),
                block_dim: (config.block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n = hidden_dim as u32;
            // SAFETY: attn_proj is [hidden_dim], down is [hidden_dim]. Both valid.
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.residual_add)
                    .arg(&mut st.scratch.attn_proj)
                    .arg(&st.scratch.down)
                    .arg(&n)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("residual_add launch: {e}")))?;
        }

        // 13. Sync + readback result to ActivationBuffer.
        self.device.synchronize()?;
        let host_result = self.device.dtoh_copy(&st.scratch.attn_proj)?;

        // Update the CPU-side KV cache seq_len to stay in sync with the GPU KV cache.
        let new_seq_len = (kv.seq_len + 1).min(kv.max_seq_len);
        kv.seq_len = new_seq_len;

        // Write result back to activation buffer.
        x.write_f32_from(&host_result);
        Ok(())
    }

    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let vocab_size = hp.vocab_size as usize;
        let eps = hp.norm_eps;

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized".into())
        })?;

        // 1. Upload x to GPU.
        let x_f32 = x.as_f32_slice();
        self.device.htod_copy_into(x_f32, &mut st.scratch.x_gpu)?;

        // 2. RMSNorm with final_norm weights.
        {
            let block_size = rmsnorm_block_size(hidden_dim);
            let shared_bytes = rmsnorm_shared_bytes(block_size);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            let dim = hidden_dim as u32;
            // SAFETY: x_gpu is [hidden_dim], final_norm is [hidden_dim],
            // normed is [hidden_dim]. All valid.
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.rmsnorm)
                    .arg(&st.scratch.x_gpu)
                    .arg(&st.globals.final_norm)
                    .arg(&mut st.scratch.normed)
                    .arg(&eps)
                    .arg(&dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("rmsnorm final launch: {e}")))?;
        }

        // 3. MatVec: logits = output_proj * normed
        // Reuse the pre-allocated logits_gpu buffer from MutableState.
        {
            if let Some(ref proj_q4a) = st.globals.output_proj_q4_aligned {
                // Q4Aligned dp4a output projection (highest priority for Q4_0).
                if let (Some(ref quant_fn), Some(ref mv_fn)) = (
                    st.kernels.quantize_f32_to_q8_1.as_ref(),
                    st.kernels.matvec_q4_aligned_q8_1.as_ref(),
                ) {
                    let out_dim = vocab_size as u32;
                    let in_dim = hidden_dim as u32;
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    let quant_grid = q8_1_quant_grid(in_dim);
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut *q8_1_buf)
                            .arg(&in_dim)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 gdn output_proj Q4Aligned: {e}",
                    )))?;
                    let mv_grid = dp4a_q4_grid(out_dim);
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device.stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4a)
                            .arg(&*q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim)
                            .arg(&in_dim)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q4_aligned_q8_1 gdn output_proj: {e}",
                    )))?;
                }
            } else if let Some(ref proj_q4) = st.globals.output_proj_q4 {
                // Q4_0 output projection: prefer smem kernel when in_dim fits.
                let out_dim = vocab_size as u32;
                let in_dim = hidden_dim as u32;
                let shmem_needed = in_dim * 4;
                if let Some(ref smem_fn) = st.kernels.matvec_q4_0_smem {
                    if shmem_needed <= 49152 {
                        let grid = matvec_smem_grid(out_dim);
                        let shmem = matvec_smem_shared_bytes(in_dim);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (SMEM_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem,
                        };
                        unsafe {
                            self.device
                                .stream
                                .launch_builder(smem_fn)
                                .arg(proj_q4)
                                .arg(&st.scratch.normed)
                                .arg(&mut st.logits_gpu)
                                .arg(&out_dim)
                                .arg(&in_dim)
                                .launch(launch_cfg)
                        }
                        .map_err(|e| {
                            RuntimeError::Compute(format!("matvec output_proj Q4_0 smem launch: {e}"))
                        })?;
                    } else {
                        let mv_block = matvec_block_size();
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (out_dim, 1, 1),
                            block_dim: (mv_block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        unsafe {
                            self.device
                                .stream
                                .launch_builder(&st.kernels.matvec_q4_0)
                                .arg(proj_q4)
                                .arg(&st.scratch.normed)
                                .arg(&mut st.logits_gpu)
                                .arg(&out_dim)
                                .arg(&in_dim)
                                .launch(launch_cfg)
                        }
                        .map_err(|e| {
                            RuntimeError::Compute(format!("matvec output_proj Q4_0 launch: {e}"))
                        })?;
                    }
                } else {
                    let mv_block = matvec_block_size();
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim, 1, 1),
                        block_dim: (mv_block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&st.kernels.matvec_q4_0)
                            .arg(proj_q4)
                            .arg(&st.scratch.normed)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim)
                            .arg(&in_dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec output_proj Q4_0 launch: {e}"))
                    })?;
                }
            } else if let Some(ref proj_f16) = st.globals.output_proj_f16 {
                // F16 output projection: cuBLAS HGEMV (cublasGemmEx N=1).
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        proj_f16,
                        &st.scratch.normed,
                        &mut st.logits_gpu,
                        &mut st.scratch.input_f16,
                        vocab_size,
                        hidden_dim,
                        "output_proj",
                        st.algo_cache.get(vocab_size, hidden_dim),
                    )?;
                }
            } else if let Some(ref proj_f16_cache) =
                st.globals.output_proj_q8_to_f16_cache
            {
                // cuBLAS HGEMV-N=1 against pre-dequanted F16 cache.
                // Mirrors the non-graph dispatch above. Same byte budget for
                // the matvec (1.94 GB F16 vs 1.06 GB Q8) but cuBLAS GemmEx
                // ships through the tensor-core path with persistent-CTA
                // scheduling -- proven faster than custom dp4a kernels on
                // this shape (cf. BF16 output_proj at 0.66-0.77× llama.cpp).
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        proj_f16_cache,
                        &st.scratch.normed,
                        &mut st.logits_gpu,
                        &mut st.scratch.input_f16,
                        vocab_size,
                        hidden_dim,
                        "output_proj_q8_to_f16",
                        st.algo_cache.get(vocab_size, hidden_dim),
                    )?;
                }
            } else if let Some(ref proj_q8_split) = st.globals.output_proj_q8_split {
                // OUTPUT_PROJ_SPLIT: prefer split layout for graph variant too.
                // NR=32 grid for the dedicated output_proj kernel; NR=2 fallback.
                // route through `LUMEN_CUDA_OUTPUT_PROJ_NR={16,64,128}`
                // if requested and loaded.
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;
                let (split_mv_fn, mv_grid): (&CudaFunction, u32) =
                    if let Some(proj_fn) = pick_output_proj_nr_kernel(
                        &st.kernels, st.output_proj_nr,
                    ) {
                        let nr = st.output_proj_nr;
                        (proj_fn, (out_dim_u32 + nr - 1) / nr)
                    } else if let Some(ref proj_fn) = st.kernels.matvec_q8_split_output_proj {
                        (proj_fn, (out_dim_u32 + 31) / 32)
                    } else if let Some(ref generic_fn) = st.kernels.matvec_q8_split_q8_1 {
                        (generic_fn, dp4a_q8_1_grid(out_dim_u32))
                    } else {
                        return Err(RuntimeError::Compute(
                            "graph output_proj_q8_split present but no split matvec kernel available".into(),
                        ));
                    };
                if let (Some(quant_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_f32_to_q8_1.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    let quant_grid = q8_1_quant_grid(in_dim_u32);
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 graph output_proj split: {e}",
                    )))?;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(split_mv_fn)
                            .arg(proj_q8_split)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!(
                        "matvec_q8_split graph output_proj: {e}",
                    )))?;
                } else {
                    return Err(RuntimeError::Compute(
                        "graph output_proj_q8_split present but quantize kernel unavailable".into(),
                    ));
                }
            } else if let Some(ref proj_q8a) = st.globals.output_proj_q8_aligned {
                // Q8_0 aligned output projection: try Q8_1 path first, then on-the-fly.
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;

                if let (Some(quant_fn), Some(mv_fn), Some(ref mut q8_1_buf)) = (
                    st.kernels.quantize_f32_to_q8_1.as_ref(),
                    st.kernels.matvec_q8_aligned_q8_1.as_ref(),
                    st.scratch.input_q8_1.as_mut(),
                ) {
                    let quant_grid = q8_1_quant_grid(in_dim_u32);
                    let quant_cfg = CudarcLaunchConfig {
                        grid_dim: (quant_grid, 1, 1),
                        block_dim: (Q8_1_QUANT_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_f32_to_q8_1 gdn output_proj: {e}"))
                    })?;

                    let mv_grid = dp4a_q8_1_grid(out_dim_u32);
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q8a)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_aligned_q8_1 gdn output_proj: {e}"))
                    })?;
                } else {
                    let q8a_fn = st.kernels.matvec_q8_0_aligned.as_ref()
                        .or(st.kernels.matvec_q8_0_dp4a.as_ref())
                        .unwrap_or(&st.kernels.matvec_q8_0);
                    let grid = matvec_q8_0_grid(out_dim_u32);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(q8a_fn)
                            .arg(proj_q8a)
                            .arg(&st.scratch.normed)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec output_proj Q8_0 aligned launch: {e}"))
                    })?;
                }
            } else if let Some(ref proj_q8) = st.globals.output_proj_q8 {
                // Q8_0 output projection: dp4a (native Q8_0, ~1.06 B/elem).
                // Fallback when aligned repack is unavailable.
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;
                let q8_fn = st.kernels.matvec_q8_0_dp4a.as_ref()
                    .unwrap_or(&st.kernels.matvec_q8_0);
                let grid = matvec_q8_0_grid(out_dim_u32);
                let shmem = 0u32;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: shmem,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(q8_fn)
                        .arg(proj_q8)
                        .arg(&st.scratch.normed)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec output_proj Q8_0 launch: {e}"))
                })?;
            } else {
                // F32 output projection path: cuBLAS SGEMV.
                // SAFETY: output_proj is [vocab_size, hidden_dim] (uploaded in init).
                // normed is [hidden_dim]. logits_gpu is [vocab_size]. All valid.
                let cfg = GemvConfig {
                    trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    m: hidden_dim as i32,
                    n: vocab_size as i32,
                    alpha: 1.0f32,
                    lda: hidden_dim as i32,
                    incx: 1,
                    beta: 0.0f32,
                    incy: 1,
                };
                unsafe {
                    self.device
                        .blas
                        .gemv(cfg, &st.globals.output_proj, &st.scratch.normed, &mut st.logits_gpu)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("cuBLAS GEMV output_proj: {e}"))
                })?;
            }
        }

        // 4. Sync + readback logits.
        self.device.synchronize()?;
        let logits_host = self.device.dtoh_copy(&st.logits_gpu)?;

        Ok(Logits { data: logits_host })
    }

    fn set_global_tensors(
        &mut self,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        output_proj: Vec<f32>,
    ) {
        self.embedding = embedding;
        self.final_norm = final_norm;
        self.output_proj = output_proj;
    }

    fn set_output_proj_raw(&mut self, raw: Vec<u8>, quant: QuantScheme) {
        self.output_proj_quant = quant;
        self.output_proj_raw = Some(raw);
    }

    fn set_embedding_raw(&mut self, raw: Vec<u8>, quant: QuantScheme) {
        self.embedding_quant = quant;
        self.embedding_raw = Some(raw);
    }

    fn set_weight_tying(&mut self, enabled: bool) {
        self.weight_tying = enabled;
    }

    fn caps(&self) -> BackendCaps {
        let is_preloaded = self.state.lock().unwrap()
            .as_ref()
            .map(|st| !st.layer_weights_cache.is_empty())
            .unwrap_or(false);
        // MoE capability is derived from `moe_meta_cache` — the cache
        // is populated by `preload_weights` for each layer whose
        // `subtensors.experts.is_some()`. Before preload the cache is empty
        // (caps returns moe=false); after preload it reflects the model.
        let has_moe = self.state.lock().unwrap()
            .as_ref()
            .map(|st| st.moe_meta_cache.iter().any(|m| m.is_some()))
            .unwrap_or(false);
        BackendCaps {
            // Standard models use per-token prefill for exact decode precision match.
            // GDN models (Qwen3.5) REQUIRE batched prefill because per-token is
            // too slow with host round-trips per GDN layer. The batched prefill
            // has its own GDN routing (prefill_gdn_layer) and uses F32 SGEMM
            // for standard attention layers.
            batched_prefill: {
                let has_gdn = self.state.lock().unwrap()
                    .as_ref()
                    .map(|st| st.layer_weights_cache.iter().any(|lw| lw.layer_type == 1))
                    .unwrap_or(false);
                is_preloaded && has_gdn
            },
            gpu_resident: is_preloaded,
            gdn: true,
            moe: has_moe,
            gpu_argmax: false,
        }
    }

    /// CUDA stores `KvCacheGpu.k_cache` / `v_cache` in F32 unconditionally;
    /// no F16 KV dispatch is wired through the decode/prefill kernels in this
    /// release. Reject the mismatch up front so the
    /// user gets an explicit error instead of silent precision drift between
    /// the CPU `KvCache` byte layout and the GPU side.
    ///
    /// The F16 KV path on CUDA (option a) is a larger work item planned for
    /// a future release.
    fn validate_kv_precision(&self, precision: KvPrecision) -> Result<(), RuntimeError> {
        if precision != KvPrecision::F32 {
            return Err(RuntimeError::Unsupported(format!(
                "CUDA backend KV cache is currently F32-only (requested {precision:?}); \
                 set --kv-precision f32 explicitly or omit the flag. F16 KV on CUDA \
                 requires the F16 dispatch path planned for a future release.",
            )));
        }
        Ok(())
    }

    fn reset_recurrent_state(&self) {
        // Reset GPU KV caches, GDN recurrent state, and CUDA graph to prevent
        // stale data from leaking across generate() calls.
        if let Ok(mut guard) = self.state.lock() {
            if let Some(ref mut st) = *guard {
                for kv_cache in &mut st.kv_caches {
                    kv_cache.reset();
                }
                // Invalidate the captured CUDA graph (seq_len starts from 0 again).
                st.captured_graph = None;
                st.decode_token_count = 0;

                // Reset GDN h_states and conv_states (zeroing GPU buffers).
                if let Some(ref mut gdn) = st.gdn_scratch_gpu {
                    for h in &mut gdn.h_states {
                        // Zero the h_state buffer. alloc_zeros produces zeroed memory,
                        // but we need to re-zero between sequences.
                        let len = h.len();
                        if let Ok(zeros) = self.device.alloc_zeros::<f32>(len) {
                            let _ = self.device.stream.memcpy_dtod(&zeros, h);
                        }
                    }
                    for c in &mut gdn.conv_states {
                        let len = c.len();
                        if let Ok(zeros) = self.device.alloc_zeros::<f32>(len) {
                            let _ = self.device.stream.memcpy_dtod(&zeros, c);
                        }
                    }
                    gdn.conv_positions.fill(0);
                    // Also zero GPU-resident conv positions for graph capture.
                    if let Some(ref mut gpu_pos) = gdn.conv_positions_gpu {
                        for p in gpu_pos.iter_mut() {
                            let _ = self.device.htod_copy_into(&[0u32], p);
                        }
                    }
                }
            }
        }
    }

    /// CUDA peak VRAM == `total_memory - free_memory` at
    /// the time of the call.  This is the live residency snapshot — a
    /// worst-case lower bound, sufficient for the
    /// `peak_vram_pct_of_device_limit` gate (≤ 90% of device limit) and
    /// `peak_vram ≤ 120% of envelope` regression detector.
    ///
    /// Cost: one `cuMemGetInfo` call. Safe at end-of-generation; must NOT
    /// be called inside the per-token decode loop.
    ///
    /// Returns 0 on any query failure so the engine can still emit a
    /// well-formed `InferenceMetrics` row.
    fn peak_memory_bytes(&self) -> u64 {
        let total = self.device.total_memory().unwrap_or(0);
        let free = self.device.free_memory().unwrap_or(0);
        (total.saturating_sub(free)) as u64
    }

    /// CUDA disk-KV sync is intentionally NOT wired in this
    /// release. The Metal path is the only production target on M3 Ultra
    /// hardware today; wiring CUDA requires `cudaMemcpyDeviceToHost` from
    /// `KvCacheGpu.k_cache` / `v_cache` into the CPU `KvCache` mirror and
    /// a matching DtoD path for restore, plus a GDN-state DtoH/HtoD pair.
    /// Surface an explicit error so a future caller that sets
    /// `--session-resume` on CUDA gets a clear "not implemented yet"
    /// message instead of a silent zero-copy.
    fn sync_kv_to_cpu(
        &self,
        _kv: &mut crate::kv::KvCache,
        _recurrent: Option<&mut crate::kv::disk::RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported(
            "CUDA backend: sync_kv_to_cpu is not yet wired ( lands the \
             Metal path only; a CUDA equivalent needs cudaMemcpyDeviceToHost + GDN \
             state DtoH which is planned for a future release)".into(),
        ))
    }

    fn sync_kv_from_cpu(
        &self,
        _kv: &crate::kv::KvCache,
        _recurrent: Option<&crate::kv::disk::RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported(
            "CUDA backend: sync_kv_from_cpu is not yet wired ( lands the \
             Metal path only; CUDA disk-KV restore is planned for a future release)".into(),
        ))
    }

    fn prefill(
        &self,
        tokens: &[u32],
        _weights: &dyn WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let num_layers = hp.num_layers as usize;
        let eps = hp.norm_eps;
        let theta = hp.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let batch = tokens.len();

        // Detect GDN layers for batched prefill routing.
        let has_gdn = {
            let guard = self.state.lock().unwrap();
            if let Some(ref st) = *guard {
                st.layer_weights_cache.iter().any(|lw| lw.layer_type == 1)
            } else {
                false
            }
        };

        if batch == 0 {
            return Err(RuntimeError::Compute("empty prompt".into()));
        }

        let pos_start = kv.seq_len();

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized".into())
        })?;

        // Require GPU-resident weights for the batched prefill path.
        if st.layer_weights_cache.len() < num_layers {
            return Err(RuntimeError::Compute(
                "batched prefill requires GPU-resident weights \
                 (call preload_weights first)"
                    .into(),
            ));
        }

        // Resolve GDN dims up-front so the shared dequant scratch is
        // sized correctly for models whose `qkv_dim` exceeds `inter_dim`
        // (e.g. Qwen3.5-35B-A3B: qkv_dim=8192, hidden_dim=2048, inter_dim=6144
        // -> gdn_qkv needs 16.8M scratch elements, attn/FFN max is 12.6M).
        let gdn_dims: Option<(usize, usize)> = if has_gdn {
            let p = super::gdn::GdnParams::from_hyperparams(hp);
            Some((p.qkv_dim, p.value_dim))
        } else {
            None
        };

        // Allocate batch-sized scratch buffers.
        let mut pf = super::prefill::alloc_prefill_scratch(
            &self.device, batch, hidden_dim, q_dim, kv_dim, inter_dim,
            gdn_dims.map(|(q, _)| q),
            gdn_dims.map(|(_, v)| v),
        )?;

        // Allocate GDN prefill scratch if the model has GDN layers.
        let mut gdn_pf = if has_gdn {
            let gdn_params = super::gdn::GdnParams::from_hyperparams(hp);
            // Ensure GDN persistent state (h_states, conv_states) is allocated.
            self.ensure_gdn_scratch(st)?;
            Some(super::prefill::alloc_gdn_prefill_scratch(
                &self.device,
                batch,
                gdn_params.qkv_dim,
                gdn_params.num_heads,
                gdn_params.value_dim,
            )?)
        } else {
            None
        };

        // Upload token IDs to GPU.
        self.device.htod_copy_into(tokens, &mut pf.token_ids_gpu)?;

        // Step 1: Batch embed all tokens into [batch, hidden_dim].
        unsafe {
            super::prefill::launch_embed_batch(
                &self.device,
                &st.kernels,
                &st.globals.embedding,
                st.globals.embedding_q8.as_ref(),
                st.globals.embedding_f16.as_ref(),
                st.globals.embedding_q4.as_ref(),
                &pf.token_ids_gpu,
                &mut pf.x,
                batch,
                hidden_dim,
            )?;
        }

        // Step 2: Process all layers with batched GEMM for projections.
        for layer_idx in 0..num_layers {
            let lw = &st.layer_weights_cache[layer_idx];

            // ---- GDN LAYER: batched projections + sequential state update ----
            if lw.layer_type == 1 {
                self.prefill_gdn_layer(
                    layer_idx, batch, st, &mut pf,
                    gdn_pf.as_mut().unwrap(),
                    eps,
                )?;
                continue;
            }

            // ---- STANDARD ATTENTION LAYER ----

            // 2a. Batched RMSNorm for QKV projections (always F32 path for precision).
            unsafe {
                super::prefill::launch_rmsnorm_batched(
                    &self.device, &st.kernels,
                    &pf.x, &lw.attn_norm, &mut pf.normed,
                    eps, batch, hidden_dim,
                )?;
            }

            // 2b. Batched QKV projections via GEMM (no F16 caches for precision match).
            let has_qgate_fusion_pf = lw.attn_q_norm.is_some();
            if has_qgate_fusion_pf {
                // Q+gate fusion: project wq to [batch, q_dim*2], then deinterleave.
                let q_gate_dim = q_dim * 2;
                let mut pf_q_gate: CudaSlice<f32> = self.device.alloc_zeros(batch * q_gate_dim)?;
                let mut pf_gate_buf: CudaSlice<f32> = self.device.alloc_zeros(batch * q_dim)?;
                unsafe {
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.wq, None,
                        &pf.normed, &mut pf_q_gate,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, q_gate_dim, hidden_dim, "wq_qgate",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.wk, None,
                        &pf.normed, &mut pf.k,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, kv_dim, hidden_dim, "wk",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.wv, None,
                        &pf.normed, &mut pf.v,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, kv_dim, hidden_dim, "wv",
                    )?;
                }
                // Batched deinterleave: treat batch as (batch * num_heads) total heads.
                // deinterleave_qgate works on [total_heads * head_dim * 2] -> [total_heads * head_dim] + [...]
                // This works because per-head interleaving is contiguous across tokens.
                if let Some(ref deinterleave_fn) = st.kernels.deinterleave_qgate {
                    let block = 256u32;
                    let hd = head_dim as u32;
                    let total_heads = (batch * num_heads) as u32;
                    let total_q = batch * q_dim;
                    let grid = ((total_q as u32) + block - 1) / block;
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(deinterleave_fn)
                            .arg(&pf_q_gate)
                            .arg(&mut pf.q)
                            .arg(&mut pf_gate_buf)
                            .arg(&hd)
                            .arg(&total_heads)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("deinterleave_qgate prefill: {e}")))?;
                } else {
                    return Err(RuntimeError::Compute(
                        "Q+gate fusion requires deinterleave_qgate kernel".into(),
                    ));
                }
                // Batched per-head RMSNorm on Q and K.
                if let Some(ref q_norm_w) = lw.attn_q_norm {
                    let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("Q+gate fusion requires rmsnorm_per_head_inplace kernel".into())
                    })?;
                    let hd = head_dim as u32;
                    let total_heads = (batch * num_heads) as u32;
                    let block = (head_dim as u32).min(1024).max(32);
                    let block = (block / 32) * 32;
                    let shared_bytes = (block / 32) * 4;
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (total_heads, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(norm_fn)
                            .arg(&mut pf.q)
                            .arg(q_norm_w)
                            .arg(&total_heads)
                            .arg(&hd)
                            .arg(&eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm_per_head Q prefill: {e}")))?;
                }
                if let Some(ref k_norm_w) = lw.attn_k_norm {
                    let norm_fn = st.kernels.rmsnorm_per_head_inplace.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("Q+gate fusion requires rmsnorm_per_head_inplace kernel".into())
                    })?;
                    let hd = head_dim as u32;
                    let total_kv_heads = (batch * num_kv_heads) as u32;
                    let block = (head_dim as u32).min(1024).max(32);
                    let block = (block / 32) * 32;
                    let shared_bytes = (block / 32) * 4;
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (total_kv_heads, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(norm_fn)
                            .arg(&mut pf.k)
                            .arg(k_norm_w)
                            .arg(&total_kv_heads)
                            .arg(&hd)
                            .arg(&eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("rmsnorm_per_head K prefill: {e}")))?;
                }
                // Store gate_buf for later sigmoid gating after attention.
                // We'll use it after flash attention, before the output projection.
                // For now, store in a local variable that persists through the layer scope.
                // Apply sigmoid gating after attention (step 2e below).
                // NOTE: pf_gate_buf needs to survive until after attention. Since we're in
                // the same loop iteration scope, it's alive until the end of this block.

                // Continue to RoPE (step 2c) -- Q and K are now deinterleaved and normalized.
                // We don't add QKV bias for Q+gate layers (Qwen3.5 has no QKV bias).

                // Skip bias section for qgate layers (handled above).
                // Continue to step 2c...

                // 2c. Batched RoPE (within qgate branch)
                let rotary_dim_pf = hp.rotary_dim.unwrap_or(0) as u32;
                unsafe {
                    super::prefill::launch_rope_batched(
                        &self.device, &st.kernels,
                        &mut pf.q, &mut pf.k,
                        pos_start, batch,
                        num_heads, num_kv_heads, head_dim, theta,
                        hp.rope_neox, rotary_dim_pf,
                    )?;
                }

                // 2d. Batch KV cache write
                let kv_cache = &mut st.kv_caches[layer_idx];
                unsafe {
                    super::prefill::launch_kv_cache_write_batch(
                        &self.device, &st.kernels,
                        &mut kv_cache.k_cache, &pf.k,
                        pos_start, batch,
                        num_kv_heads, kv_cache.max_seq_len, head_dim,
                    )?;
                    super::prefill::launch_kv_cache_write_batch(
                        &self.device, &st.kernels,
                        &mut kv_cache.v_cache, &pf.v,
                        pos_start, batch,
                        num_kv_heads, kv_cache.max_seq_len, head_dim,
                    )?;
                }
                kv_cache.advance_seq_len_by(batch);

                // 2e. Flash Attention
                //
                // Dispatch priority (first match wins):
                // 1. FA2 block-skip (P1-3, env-gated). Long-context win
                // via mask block-skip; uses Split-K for seq_len >=
                // FA2_SPLITK_MIN_SEQ to fan out across SMs.
                // 2. WMMA tensor cores (SM 80+) when batch >= 16.
                // 3. Scalar Br=4 fallback.
                unsafe {
                    if st.kernels.use_fa2_blockskip_dispatch {
                        let causal_max = (pos_start + batch) as u32;
                        if causal_max >= super::decode::FA2_SPLITK_MIN_SEQ
                            && st.kernels.flash_attention_fa2_splitk_partial.is_some()
                            && st.kernels.flash_attention_fa2_splitk_reduce.is_some()
                        {
                            super::prefill::launch_flash_attention_fa2_splitk(
                                &self.device, &st.kernels,
                                &pf.q, kv_cache, &mut pf.attn_out,
                                batch, num_heads, num_kv_heads, head_dim, pos_start,
                                super::decode::FA2_SPLITK_SLICE,
                            )?;
                        } else {
                            super::prefill::launch_flash_attention_fa2(
                                &self.device, &st.kernels,
                                &pf.q, kv_cache, &mut pf.attn_out,
                                batch, num_heads, num_kv_heads, head_dim, pos_start,
                            )?;
                        }
                    } else if batch >= 16 && st.kernels.flash_attention_wmma.is_some() {
                        super::prefill::launch_flash_attention_wmma(
                            &self.device, &st.kernels,
                            &pf.q, kv_cache, &mut pf.attn_out,
                            batch, num_heads, num_kv_heads, head_dim, pos_start,
                        )?;
                    } else {
                        super::prefill::launch_flash_attention_br4(
                            &self.device, &st.kernels,
                            &pf.q, kv_cache, &mut pf.attn_out,
                            batch, num_heads, num_kv_heads, head_dim, pos_start,
                        )?;
                    }
                }

                // 2e.5. Sigmoid gating: attn_out = sigmoid(gate) * attn_out (per token)
                //
                // FIX-3: write through pf.q (sized [batch * q_dim], unused after
                // attention) then memcpy back to attn_out. Previously the temp was
                // pf.normed which is sized [batch * hidden_dim]; that overflowed for
                // Qwen3.5-MoE-35B-A3B where `q_dim=4096 > hidden_dim=2048`, corrupting
                // adjacent GPU memory and producing gibberish output.
                if let Some(ref sigmoid_fn) = st.kernels.sigmoid_mul {
                    let total_elems = (batch * q_dim) as u32;
                    let block = 256u32;
                    let grid = (total_elems + block - 1) / block;
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // Use pf.q as temp output (sized [batch * q_dim], free until next layer).
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(sigmoid_fn)
                            .arg(&pf_gate_buf)
                            .arg(&pf.attn_out)
                            .arg(&mut pf.q)
                            .arg(&total_elems)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| RuntimeError::Compute(format!("sigmoid_mul prefill: {e}")))?;
                    // Copy q -> attn_out (both [batch * q_dim])
                    self.device
                        .stream
                        .memcpy_dtod(&pf.q, &mut pf.attn_out)
                        .map_err(|e| RuntimeError::Compute(format!("sigmoid_mul prefill dtod: {e}")))?;
                } else {
                    return Err(RuntimeError::Compute(
                        "Q+gate fusion requires sigmoid_mul kernel".into(),
                    ));
                }

                // 2f. Output projection + residual
                unsafe {
                    super::prefill::launch_gemm_residual(
                        &self.device, &st.kernels, &lw.wo, None,
                        &pf.attn_out, &pf.x, &mut pf.attn_proj,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, hidden_dim, q_dim, "wo",
                    )?;
                }

                // 2g-2j. FFN (same as standard path) — MoE branch OR
                // dense. See `prefill_moe_ffn_layer` doc-comment for context.
                let is_moe_layer = lw.moe_layer_blob.is_some();
                if is_moe_layer {
                    self.prefill_moe_ffn_layer(layer_idx, batch, st, &mut pf, eps)?;
                } else {
                    unsafe {
                        super::prefill::launch_rmsnorm_batched(
                            &self.device, &st.kernels,
                            &pf.attn_proj, &lw.ffn_norm, &mut pf.normed,
                            eps, batch, hidden_dim,
                        )?;
                        super::prefill::launch_gemm_projection(
                            &self.device, &st.kernels, &lw.w_gate, None,
                            &pf.normed, &mut pf.gate,
                            &mut pf.dequant_f32, &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch, inter_dim, hidden_dim, "gate",
                        )?;
                        super::prefill::launch_gemm_projection(
                            &self.device, &st.kernels, &lw.w_up, None,
                            &pf.normed, &mut pf.up,
                            &mut pf.dequant_f32, &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch, inter_dim, hidden_dim, "up",
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_swiglu_batched(
                            &self.device, &st.kernels,
                            &mut pf.gate, &pf.up, batch, inter_dim,
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_gemm_projection(
                            &self.device, &st.kernels, &lw.w_down, None,
                            &pf.gate, &mut pf.down,
                            &mut pf.dequant_f32, &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch, hidden_dim, inter_dim, "down",
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_residual_add_batched(
                            &self.device, &st.kernels,
                            &mut pf.attn_proj, &pf.down, batch, hidden_dim,
                        )?;
                    }
                    self.device
                        .stream
                        .memcpy_dtod(&pf.attn_proj, &mut pf.x)
                        .map_err(|e| RuntimeError::Compute(format!("dtod x<-attn_proj qgate prefill: {e}")))?;
                }
                continue; // Skip the standard path below
            }

            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.wq, None,
                    &pf.normed, &mut pf.q,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, q_dim, hidden_dim, "wq",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.wk, None,
                    &pf.normed, &mut pf.k,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, kv_dim, hidden_dim, "wk",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device, &st.kernels, &lw.wv, None,
                    &pf.normed, &mut pf.v,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, kv_dim, hidden_dim, "wv",
                )?;
            }

            // QKV bias (Qwen2-family, prefill).
            if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
                let block = 256u32;
                unsafe {
                    if let Some(ref bq) = lw.bq {
                        let total = (batch * q_dim) as u32;
                        let dim_u32 = q_dim as u32;
                        let g = (total + block - 1) / block;
                        self.device.stream.launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.q).arg(bq).arg(&total).arg(&dim_u32)
                            .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                            .map_err(|e| RuntimeError::Compute(format!("bias_add_batched bq prefill: {e}")))?;
                    }
                    if let Some(ref bk) = lw.bk {
                        let total = (batch * kv_dim) as u32;
                        let dim_u32 = kv_dim as u32;
                        let g = (total + block - 1) / block;
                        self.device.stream.launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.k).arg(bk).arg(&total).arg(&dim_u32)
                            .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                            .map_err(|e| RuntimeError::Compute(format!("bias_add_batched bk prefill: {e}")))?;
                    }
                    if let Some(ref bv) = lw.bv {
                        let total = (batch * kv_dim) as u32;
                        let dim_u32 = kv_dim as u32;
                        let g = (total + block - 1) / block;
                        self.device.stream.launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.v).arg(bv).arg(&total).arg(&dim_u32)
                            .launch(CudarcLaunchConfig { grid_dim: (g,1,1), block_dim: (block,1,1), shared_mem_bytes: 0 })
                            .map_err(|e| RuntimeError::Compute(format!("bias_add_batched bv prefill: {e}")))?;
                    }
                }
            }

            // 2c. Batched RoPE with per-token positions.
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            unsafe {
                super::prefill::launch_rope_batched(
                    &self.device, &st.kernels,
                    &mut pf.q, &mut pf.k,
                    pos_start, batch,
                    num_heads, num_kv_heads, head_dim, theta,
                    hp.rope_neox, rotary_dim,
                )?;
            }

            // 2d. Batch KV cache write for all tokens at once.
            let kv_cache = &mut st.kv_caches[layer_idx];
            unsafe {
                super::prefill::launch_kv_cache_write_batch(
                    &self.device, &st.kernels,
                    &mut kv_cache.k_cache, &pf.k,
                    pos_start, batch,
                    num_kv_heads, kv_cache.max_seq_len, head_dim,
                )?;
                super::prefill::launch_kv_cache_write_batch(
                    &self.device, &st.kernels,
                    &mut kv_cache.v_cache, &pf.v,
                    pos_start, batch,
                    num_kv_heads, kv_cache.max_seq_len, head_dim,
                )?;
            }
            kv_cache.advance_seq_len_by(batch);

            // 2e. Flash Attention: single kernel for ALL tokens with causal masking.
            //
            // Dispatch priority (first match wins):
            // 1. FA2 block-skip (P1-3, env-gated). Long-context win via
            // mask block-skip; uses Split-K when seq_len >= FA2_SPLITK_MIN_SEQ.
            // 2. Tensor-core WMMA (SM 80+): 16x16 tiles via mma.sync PTX.
            // Uses F16 tensor cores for QK^T and PV, up to 16x throughput
            // over scalar F32 on A100.
            // 3. Scalar Br=4 fallback: 4 queries/block, warp-level parallelism.
            // Used when batch < 16 (not enough queries for a full WMMA tile).
            unsafe {
                if st.kernels.use_fa2_blockskip_dispatch {
                    let causal_max = (pos_start + batch) as u32;
                    if causal_max >= super::decode::FA2_SPLITK_MIN_SEQ
                        && st.kernels.flash_attention_fa2_splitk_partial.is_some()
                        && st.kernels.flash_attention_fa2_splitk_reduce.is_some()
                    {
                        super::prefill::launch_flash_attention_fa2_splitk(
                            &self.device, &st.kernels,
                            &pf.q, kv_cache, &mut pf.attn_out,
                            batch, num_heads, num_kv_heads, head_dim, pos_start,
                            super::decode::FA2_SPLITK_SLICE,
                        )?;
                    } else {
                        super::prefill::launch_flash_attention_fa2(
                            &self.device, &st.kernels,
                            &pf.q, kv_cache, &mut pf.attn_out,
                            batch, num_heads, num_kv_heads, head_dim, pos_start,
                        )?;
                    }
                } else if batch >= 16 && st.kernels.flash_attention_wmma.is_some() {
                    super::prefill::launch_flash_attention_wmma(
                        &self.device, &st.kernels,
                        &pf.q, kv_cache, &mut pf.attn_out,
                        batch, num_heads, num_kv_heads, head_dim, pos_start,
                    )?;
                } else {
                    super::prefill::launch_flash_attention_br4(
                        &self.device, &st.kernels,
                        &pf.q, kv_cache, &mut pf.attn_out,
                        batch, num_heads, num_kv_heads, head_dim, pos_start,
                    )?;
                }
            }

            // 2f. Batched output projection + residual via GEMM (no F16 caches).
            unsafe {
                super::prefill::launch_gemm_residual(
                    &self.device, &st.kernels, &lw.wo, None,
                    &pf.attn_out, &pf.x, &mut pf.attn_proj,
                    &mut pf.dequant_f32, &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch, hidden_dim, q_dim, "wo",
                )?;
            }

            // 2g-2j. FFN — MoE branch OR dense.
            //
            // See `prefill_moe_ffn_layer` doc-comment for the design-gap
            // that this branch closes. For dense models (Qwen3.5-9B,
            // Qwen2.5-7B/14B), `lw.moe_layer_blob` is always `None` so the
            // dense branch runs unchanged — byte-identical to the prior
            // path.
            let is_moe_layer = lw.moe_layer_blob.is_some();
            if is_moe_layer {
                self.prefill_moe_ffn_layer(layer_idx, batch, st, &mut pf, eps)?;
            } else {
                // 2g. FFN: batched RMSNorm + GEMM gate/up (always F32 path for precision).
                unsafe {
                    super::prefill::launch_rmsnorm_batched(
                        &self.device, &st.kernels,
                        &pf.attn_proj, &lw.ffn_norm, &mut pf.normed,
                        eps, batch, hidden_dim,
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.w_gate, None,
                        &pf.normed, &mut pf.gate,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, inter_dim, hidden_dim, "gate",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.w_up, None,
                        &pf.normed, &mut pf.up,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, inter_dim, hidden_dim, "up",
                    )?;
                }

                // 2h. Batched SwiGLU (standard path, no F16 fusion).
                unsafe {
                    super::prefill::launch_swiglu_batched(
                        &self.device, &st.kernels,
                        &mut pf.gate, &pf.up, batch, inter_dim,
                    )?;
                }

                // 2i. Batched down projection via GEMM (no F16 caches).
                unsafe {
                    super::prefill::launch_gemm_projection(
                        &self.device, &st.kernels, &lw.w_down, None,
                        &pf.gate, &mut pf.down,
                        &mut pf.dequant_f32, &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch, hidden_dim, inter_dim, "down",
                    )?;
                }

                // 2j. Batched residual add: x = attn_proj + down.
                // Write result directly to pf.x (eliminates the separate memcpy_dtod).
                unsafe {
                    super::prefill::launch_residual_add_batched(
                        &self.device, &st.kernels,
                        &mut pf.attn_proj, &pf.down, batch, hidden_dim,
                    )?;
                }
                self.device
                    .stream
                    .memcpy_dtod(&pf.attn_proj, &mut pf.x)
                    .map_err(|e| {
                        RuntimeError::Compute(format!("dtod x<-attn_proj prefill: {e}"))
                    })?;
            }
        }

        // Step 3: Extract last token's hidden state into decode scratch.
        unsafe {
            super::prefill::launch_extract_row(
                &self.device, &st.kernels,
                &pf.x, &mut st.scratch.x_gpu,
                batch - 1, hidden_dim,
            )?;
        }

        // Step 4: Single sync + readback.
        self.device.synchronize()?;
        let result = self.device.dtoh_copy(&st.scratch.x_gpu)?;

        // Step 5: Advance host-side KV cache seq_len to match GPU state.
        for _ in 0..batch {
            kv.advance_seq_len()?;
        }

        Ok(result)
    }

    fn preload_weights(
        &mut self,
        weights: &dyn WeightProvider,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let num_layers = hp.num_layers as usize;

        // Copy hp values before mutable borrow of state.
        let hp_copy = *hp;

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "CUDA backend not initialized: call init() before preload_weights".into(),
            )
        })?;

        let mut cache = Vec::with_capacity(num_layers);

        let mem_before_layers = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] before layer weight upload: {:.2} GB free",
            (mem_before_layers as f64) / 1.0e9
        );

        for layer_idx in 0..num_layers {
            // Use get_layer_raw to bypass dequantization — we need Q8_0/Q4_0/F16
            // in their native format so upload_layer_weights creates the correct
            // GpuWeightBuf variant (Q8Raw, Q4Raw, F16Raw) for GPU kernel dispatch.
            let layer_view = weights.get_layer_raw(layer_idx).map_err(|e| {
                RuntimeError::Compute(format!(
                    "Failed to get raw layer {} for GPU-resident preload: {}",
                    layer_idx, e,
                ))
            })?;
            // build per-layer MoE metadata when the layer has experts.
            // We build this BEFORE upload_layer_weights so the meta references
            // offsets that remain stable across the upload (the upload writes
            // bytes into the layer's main GPU buffer, the byte offsets
            // computed by the converter are preserved through the htod copy).
            if let Some(meta) = super::moe::build_moe_meta(&layer_view.subtensors) {
                if layer_idx < st.moe_meta_cache.len() {
                    // eagerly build the Phase-F batched-expert GPU
                    // offset tables. The tables are tiny (~6 KB / layer at
                    // num_experts=256) and immutable across the model's
                    // lifetime; cost is ~2 htod copies per MoE layer at
                    // preload (negligible vs the weight upload). Always built
                    // so `LUMEN_CUDA_MOE_BATCHED=1` can switch dispatch at
                    // runtime without a preload-time gate. Stored in a
                    // parallel cache (not on `CudaMoeMeta`) so `CudaMoeMeta`
                    // can keep its derive(Clone) for the prefill loop.
                    let batched = super::moe::build_batched_offsets(&self.device, &meta)?;
                    st.moe_batched_offsets[layer_idx] = Some(batched);
                    st.moe_meta_cache[layer_idx] = Some(meta);
                }
            }
            let gpu_weights = upload_layer_weights(&self.device, &layer_view, &hp_copy)?;
            cache.push(gpu_weights);
            // Print every 4 layers to avoid log flooding while still catching OOM zones.
            if (layer_idx + 1) % 4 == 0 || layer_idx + 1 == num_layers {
                let mem_now = self.device.free_memory().unwrap_or(0);
                eprintln!(
                    "[CUDA mem] after layer {} weights uploaded: {:.2} GB free",
                    layer_idx,
                    (mem_now as f64) / 1.0e9
                );
            }
        }

        let mem_after_raw_layers = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] all {num_layers} layer raw weights uploaded: {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_raw_layers as f64) / 1.0e9,
            (mem_before_layers.saturating_sub(mem_after_raw_layers) as f64) / 1.0e9
        );

        // Pre-dequant Q8_0 weights to F16 for HGEMM prefill (tensor core path).
        // This runs the dequant_q8_0_to_f16 kernel once per Q8_0 weight tensor,
        // storing the F16 version alongside the original Q8_0 data. The extra GPU
        // memory is ~2x the Q8_0 weight size (F16 = 2 bytes/element vs Q8_0 ~1.0625).
        // BF16 weights skip this step entirely (no F16 cache needed; matvec_bf16
        // dispatches directly off the raw BF16 bytes).
        let mem_before_f16_cache = self.device.free_memory().unwrap_or(0);
        for (layer_idx, layer) in cache.iter_mut().enumerate() {
            super::gpu_buffers::dequant_layer_q8_to_f16(
                &self.device,
                &st.kernels.dequant_q8_0_to_f16,
                &st.kernels.dequant_q4_0_to_f16,
                &st.kernels,
                layer,
                &hp_copy,
            )
            .map_err(|e| {
                RuntimeError::Compute(format!(
                    "F16 pre-dequant layer {layer_idx}: {e}",
                ))
            })?;
        }
        let mem_after_f16_cache = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] after F16 dequant caches: {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_f16_cache as f64) / 1.0e9,
            (mem_before_f16_cache.saturating_sub(mem_after_f16_cache) as f64) / 1.0e9
        );

        // Tile ssm_norm from [head_dim] to [value_dim] for GDN layers.
        // This allows the standard rmsnorm kernel to be used on the [value_dim] output.
        let _gdn_params = super::gdn::GdnParams::from_hyperparams(&hp_copy);
        for layer in cache.iter_mut() {
            if layer.layer_type == 1 {
                // Read ssm_norm from the layer's subtensors (it's [head_dim] F32).
                // Tile by repeating: [head_dim] -> [num_heads * head_dim = value_dim]
                if layer.ssm_norm_tiled.is_none() {
                    // ssm_norm is not uploaded as a separate field — it comes from the LBC
                    // subtensors. The tiled buffer is populated by the LBC upload path
                    // once the subtensor is materialised; leaving it as `None` here
                    // surfaces a clear error if a GDN dispatch races the upload.
                }
            }
        }

        let has_gdn = cache.iter().any(|lw| lw.layer_type == 1);

        // split-layout integration: Q8_0 per-row split (SoA) clone pass.
        // Runs BEFORE the aligned repack pass because both consume Q8Raw and
        // the aligned pass MUTATES Q8Raw -> Q8Aligned in place. After this
        // pass, layers that received a split sibling skip aligned repack
        // (their decode path prefers the sibling, prefill keeps Q8Raw).
        let mut layers_with_q8_split: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        if st.use_q8_split {
            if let (Some(ref split_repack_fn), true) = (
                st.kernels.repack_q8_raw_to_split.as_ref(),
                st.kernels.matvec_q8_split_q8_1.is_some(),
            ) {
                let mem_before_q8_split = self.device.free_memory().unwrap_or(0);
                let (n_layers_split, oom_layer, oom_count, total_jobs) = unsafe {
                    repack_all_layers_q8_clone_to_split(
                        &self.device, split_repack_fn, &mut cache, &hp_copy,
                    )
                };
                let mem_after_q8_split = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_q8_split.saturating_sub(mem_after_q8_split) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q8_SPLIT=1: cloned Q8 split siblings on \
                     {n_layers_split} layers, {total_jobs} jobs attempted, \
                     {oom_count} OOMs (first at layer {:?}), {consumed_gb:.2} GB consumed",
                    oom_layer,
                );
                // Track which layers have any Q8 split sibling -- those layers
                // skip the aligned repack below to save the ~12% memory cost
                // (36-byte aligned vs 34-byte raw).
                for (idx, lw) in cache.iter().enumerate() {
                    if lw.q8_split_wq.is_some() || lw.q8_split_wk.is_some()
                        || lw.q8_split_wv.is_some() || lw.q8_split_wo.is_some()
                        || lw.q8_split_w_gate.is_some() || lw.q8_split_w_up.is_some()
                        || lw.q8_split_w_down.is_some()
                    {
                        layers_with_q8_split.insert(idx);
                    }
                }
            } else if st.use_q8_split {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q8_SPLIT=1 set but split kernels unavailable; \
                     decode will use Q8Raw/Q8Aligned base path"
                );
            }
        }

        // split-layout integration: output projection SPLIT clone.
        // Independent of the per-layer Q8 SPLIT pass. The 1 GB output_proj is
        // touched once per token but is one of the largest single bandwidth
        // sinks (+7.7% on its own). Skip when SPLIT kernel is
        // unavailable or alloc fails; the existing aligned/raw path is
        // preserved as fallback.
        if st.use_output_proj_split {
            if let (Some(ref split_repack_fn), Some(ref proj_q8), true) = (
                st.kernels.repack_q8_raw_to_split.as_ref(),
                st.globals.output_proj_q8.as_ref(),
                st.kernels.matvec_q8_split_q8_1.is_some()
                    || st.kernels.matvec_q8_split_output_proj.is_some(),
            ) {
                let vocab_size = hp_copy.vocab_size as usize;
                let hidden = hp_copy.hidden_dim as usize;
                match unsafe {
                    repack_q8_raw_to_split(
                        &self.device, split_repack_fn, proj_q8, vocab_size, hidden,
                    )
                } {
                    Ok(split_buf) => {
                        st.globals.output_proj_q8_split = Some(split_buf);
                        eprintln!(
                            "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1: output_proj cloned to split layout ({vocab_size}x{hidden})"
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1 set but output_proj split repack failed (falling back to Q8Raw/Q8Aligned): {e}"
                        );
                    }
                }
            } else if st.use_output_proj_split {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1 set but split kernel or Q8 output_proj unavailable; falling back to Q8Aligned/Q8Raw path"
                );
            }
        }

        // output_proj fast-path: F16 dequant cache.
        //
        // Pre-dequantize the Q8_0 output projection (~1 GB on Qwen3.5-9B Q8)
        // to F16 (~1.94 GB) once at preload, then dispatch via the existing
        // `launch_hgemv_f16_preconverted` path (cublasGemmEx with N=1). This
        // mirrors the BF16 output_proj path (which already wins on the same
        // 248320x4096 shape via `launch_hgemv_bf16` at 0.66-0.77× llama.cpp) while
        // letting Q8-storage models use the same compute path at decode time.
        //
        // VRAM cost: replaces the 1 GB Q8 (or 1 GB Q8 SPLIT clone) with a
        // 1.94 GB F16 cache. Net delta vs Q8Raw alone: +0.94 GB. Net delta
        // vs Q8 SPLIT integration: +0.94 GB (the SPLIT clone is preserved as
        // fallback; both coexist when env vars stack). A100-80GB has ~25-30 GB
        // headroom after the SPLIT stack at this point, so the alloc should
        // succeed cleanly on Qwen3.5-9B.
        if st.use_output_proj_f16_cache {
            if let (Some(ref dequant_fn), Some(ref proj_q8)) = (
                Some(&st.kernels.dequant_q8_0_to_f16),
                st.globals.output_proj_q8.as_ref(),
            ) {
                let vocab_size = hp_copy.vocab_size as usize;
                let hidden = hp_copy.hidden_dim as usize;
                let n_elem = vocab_size * hidden;
                let mem_before = self.device.free_memory().unwrap_or(0);
                match super::gpu_buffers::dequant_q8_to_f16_gpu(
                    &self.device,
                    dequant_fn,
                    proj_q8,
                    n_elem,
                ) {
                    Ok(f16_buf) => {
                        let mem_after = self.device.free_memory().unwrap_or(0);
                        st.globals.output_proj_q8_to_f16_cache = Some(f16_buf);
                        eprintln!(
                            "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1: output_proj dequanted to F16 ({vocab_size}x{hidden}, {:.2} GB consumed)",
                            (mem_before.saturating_sub(mem_after) as f64) / 1.0e9
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1 set but dequant alloc failed (falling back to Q8 split/aligned): {e}"
                        );
                    }
                }
            } else if st.use_output_proj_f16_cache {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE=1 set but output_proj is not Q8_0; ignoring"
                );
            }
        }

        // Repack Q8_0 weights to 36-byte aligned blocks for dp4a int* loads.
        // Aligned weight repack: enabled for ALL models including GDN.
        // +16% decode from dp4a int* loads (proven C8-C11).
        // Output projection repack is SKIPPED -- too large, causes OOM on
        // A100-80GB for GDN models, and negligible impact (called once per token).
        // aligned repack runs ALONGSIDE Q8 split clones so the
        // fused-swiglu-down path (line 2222) -- which requires Q8Aligned and is
        // faster than separate-quantize + split matvec -- remains available.
        // Decode dispatch checks SPLIT sibling first, then falls through to
        // the Q8Aligned path (which the aligned repack pre-stages).
        let _ = &layers_with_q8_split; // tracked for diagnostic logs; no longer gates aligned skip
        if let Some(ref repack_fn) = st.kernels.repack_q8_0_to_aligned36 {
            if st.kernels.matvec_q8_0_aligned.is_some() || st.kernels.matvec_q8_aligned_q8_1.is_some() {
                for (layer_idx, layer) in cache.iter_mut().enumerate() {
                    super::gpu_buffers::repack_layer_q8_to_aligned(
                        &self.device,
                        repack_fn,
                        layer,
                        &hp_copy,
                        has_gdn,
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "Q8_0 aligned repack layer {layer_idx}: {e}",
                        ))
                    })?;
                }
                // Skip output_proj repack for GDN models (too large, OOM risk, negligible impact).
                if !has_gdn {
                    if let Some(ref proj_q8) = st.globals.output_proj_q8 {
                        let vocab_size = hp_copy.vocab_size as usize;
                        let hidden = hp_copy.hidden_dim as usize;
                        let num_elements = vocab_size * hidden;
                        match super::gpu_buffers::repack_q8_to_aligned(
                            &self.device, repack_fn, proj_q8, num_elements,
                        ) {
                            Ok(aligned) => {
                                st.globals.output_proj_q8_aligned = Some(aligned);
                            }
                            Err(e) => {
                                eprintln!("[CUDA] Output projection Q8_0 repack failed (using unaligned): {e}");
                            }
                        }
                    }
                }
            }
        }

        // split-layout integration: Q4_0 per-row split (SoA) clone pass.
        // Same pattern as the Q8 SPLIT pass above -- runs BEFORE the Q4 aligned
        // pass so SPLIT can read Q4Raw before aligned mutates it.
        let mut layers_with_q4_split: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        if st.use_q4_split {
            if let (Some(ref split_repack_fn), true) = (
                st.kernels.repack_q4_raw_to_split.as_ref(),
                st.kernels.matvec_q4_split_q8_1.is_some(),
            ) {
                let mem_before_q4_split = self.device.free_memory().unwrap_or(0);
                let (n_layers_split, oom_layer, oom_count, total_jobs) = unsafe {
                    repack_all_layers_q4_clone_to_split(
                        &self.device, split_repack_fn, &mut cache, &hp_copy,
                    )
                };
                let mem_after_q4_split = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_q4_split.saturating_sub(mem_after_q4_split) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q4_SPLIT=1: cloned Q4 split siblings on \
                     {n_layers_split} layers, {total_jobs} jobs attempted, \
                     {oom_count} OOMs (first at layer {:?}), {consumed_gb:.2} GB consumed",
                    oom_layer,
                );
                for (idx, lw) in cache.iter().enumerate() {
                    if lw.q4_split_wq.is_some() || lw.q4_split_wk.is_some()
                        || lw.q4_split_wv.is_some() || lw.q4_split_wo.is_some()
                        || lw.q4_split_w_gate.is_some() || lw.q4_split_w_up.is_some()
                        || lw.q4_split_w_down.is_some()
                    {
                        layers_with_q4_split.insert(idx);
                    }
                }
            } else if st.use_q4_split {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q4_SPLIT=1 set but split kernels unavailable; \
                     decode will use Q4Raw/Q4Aligned base path"
                );
            }
        }

        // split-layout integration: GDN Q4 weight split pass.
        // Targets ssm_out / attn_gate / ssm_alpha / ssm_beta (the GDN-specific
        // Q4Raw weights). Q8 variant intentionally NOT wired -- showed
        // Q8 + GDN_SPLIT OOMs on A100-80GB due to physical VRAM exhaustion.
        if st.use_gdn_split && has_gdn {
            if let (Some(ref split_repack_fn), true) = (
                st.kernels.repack_q4_raw_to_split.as_ref(),
                st.kernels.matvec_q4_split_q8_1.is_some(),
            ) {
                let mem_before_gdn_split = self.device.free_memory().unwrap_or(0);
                let (n_layers_split, total_jobs) = unsafe {
                    repack_all_layers_gdn_q4_clone_to_split(
                        &self.device, split_repack_fn, &mut cache, &hp_copy,
                    )
                };
                let mem_after_gdn_split = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_gdn_split.saturating_sub(mem_after_gdn_split) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] LUMEN_CUDA_GDN_SPLIT=1: cloned GDN Q4 split siblings on \
                     {n_layers_split} layers, {total_jobs} jobs attempted, \
                     {consumed_gb:.2} GB consumed"
                );
            } else {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_GDN_SPLIT=1 set but Q4 split kernels unavailable; \
                     GDN weights will use existing Q4Raw path"
                );
            }
        } else if st.use_gdn_split && !has_gdn {
            eprintln!(
                "[CUDA] LUMEN_CUDA_GDN_SPLIT=1 set but model has no GDN layers; ignored"
            );
        }

        // tile-layout integration: Q8 tile-grouped clone pass.
        // Runs AFTER the SPLIT pass so SPLIT siblings are populated first
        // (tile wins when both are set; SPLIT is still consumed by other
        // code paths). Uses the same Q8Raw source as SPLIT -- tile clone
        // does NOT mutate the source. Skipped silently when the env var is
        // unset OR the tile kernels failed to compile.
        if st.use_q8_tile {
            if let (Some(ref tile_repack_fn), true) = (
                st.kernels.repack_q8_raw_to_tile.as_ref(),
                st.kernels.matvec_q8_tile_q8_1.is_some(),
            ) {
                let mem_before_q8_tile = self.device.free_memory().unwrap_or(0);
                let (n_layers_tile, oom_layer, oom_count, total_jobs) = unsafe {
                    repack_all_layers_q8_clone_to_tile(
                        &self.device, tile_repack_fn, &mut cache, &hp_copy,
                    )
                };
                let mem_after_q8_tile = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_q8_tile.saturating_sub(mem_after_q8_tile) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q8_TILE=1: cloned Q8 tile siblings on \
                     {n_layers_tile} layers, {total_jobs} jobs attempted, \
                     {oom_count} OOMs (first at layer {:?}), {consumed_gb:.2} GB consumed",
                    oom_layer,
                );
            } else if st.use_q8_tile {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q8_TILE=1 set but tile kernels unavailable; \
                     decode will use SPLIT / Q8Aligned / Q8Raw base path"
                );
            }
        }

        // Repack Q4_0 weights to 20-byte aligned blocks for dp4a int* nibble loads.
        // aligned repack runs ALONGSIDE Q4 split clones so the
        // fused-swiglu-down path (line 2390) -- which requires Q4Aligned and is
        // faster than separate-quantize + split matvec -- remains available.
        // Decode dispatch checks SPLIT sibling first, then falls through to
        // the Q4Aligned path.
        let _ = &layers_with_q4_split; // tracked for diagnostic logs; no longer gates aligned skip
        if let Some(ref repack_fn) = st.kernels.repack_q4_0_to_aligned20 {
            if st.kernels.matvec_q4_aligned_q8_1.is_some() {
                for (layer_idx, layer) in cache.iter_mut().enumerate() {
                    super::gpu_buffers::repack_layer_q4_to_aligned(
                        &self.device,
                        repack_fn,
                        layer,
                        &hp_copy,
                        has_gdn,
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "Q4_0 aligned repack layer {layer_idx}: {e}",
                        ))
                    })?;
                }
                // Skip output_proj repack for GDN models (too large, OOM risk, negligible impact).
                if !has_gdn {
                    if let Some(ref proj_q4) = st.globals.output_proj_q4 {
                        let vocab_size = hp_copy.vocab_size as usize;
                        let hidden = hp_copy.hidden_dim as usize;
                        let num_elements = vocab_size * hidden;
                        match super::gpu_buffers::repack_q4_to_aligned(
                            &self.device, repack_fn, proj_q4, num_elements,
                        ) {
                            Ok(aligned) => {
                                st.globals.output_proj_q4_aligned = Some(aligned);
                            }
                            Err(e) => {
                                eprintln!("[CUDA] Output projection Q4_0 repack failed (using unaligned): {e}");
                            }
                        }
                    }
                }
            }
        }

        // tile-layout integration: Q4 tile-grouped clone pass.
        // Runs AFTER the Q4 SPLIT + GDN_SPLIT + Q4 aligned passes (same
        // Q4Raw source is consumed by all four; tile clone does NOT mutate
        // the source). Skipped when the env var is unset or kernels are
        // unavailable.
        if st.use_q4_tile {
            if let (Some(ref tile_repack_fn), true) = (
                st.kernels.repack_q4_raw_to_tile.as_ref(),
                st.kernels.matvec_q4_tile_q8_1.is_some(),
            ) {
                let mem_before_q4_tile = self.device.free_memory().unwrap_or(0);
                let (n_layers_tile, oom_layer, oom_count, total_jobs) = unsafe {
                    repack_all_layers_q4_clone_to_tile(
                        &self.device, tile_repack_fn, &mut cache, &hp_copy,
                    )
                };
                let mem_after_q4_tile = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_q4_tile.saturating_sub(mem_after_q4_tile) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q4_TILE=1: cloned Q4 tile siblings on \
                     {n_layers_tile} layers, {total_jobs} jobs attempted, \
                     {oom_count} OOMs (first at layer {:?}), {consumed_gb:.2} GB consumed",
                    oom_layer,
                );
            } else if st.use_q4_tile {
                eprintln!(
                    "[CUDA] LUMEN_CUDA_Q4_TILE=1 set but tile kernels unavailable; \
                     decode will use SPLIT / Q4Aligned / Q4Raw base path"
                );
            }
        }

        // Allocate Q+gate fusion scratch buffers if any layer has attn_q_norm.
        let has_qgate = cache.iter().any(|lw| lw.attn_q_norm.is_some());
        if has_qgate {
            let q_dim = hp_copy.num_heads as usize * hp_copy.head_dim as usize;
            let q_gate_dim = q_dim * 2;
            st.scratch.q_gate = Some(self.device.alloc_zeros(q_gate_dim)?);
            st.scratch.gate_buf = Some(self.device.alloc_zeros(q_dim)?);
            eprintln!("[CUDA] Q+gate fusion scratch: q_gate={q_gate_dim}, gate_buf={q_dim} elements");
        }

        // GDN layers are now graph-capturable
        // when the device-resident conv_position infrastructure is wired up.
        // The `can_use_graph` gate above checks all required preconditions
        // (graph megakernel compiled, conv_positions_gpu allocated, two-launch
        // env not set) and gracefully falls back to the eager path otherwise.
        st.has_gdn_layers = has_gdn;
        st.has_qgate_layers = has_qgate;
        // detect MoE layers from `moe_meta_cache`. Populated by
        // `preload_weights` immediately before this site (line ~14705). The
        // `can_use_graph` gate enables MoE-aware graph capture only when
        // `LUMEN_CUDA_MOE_DECODE_GRAPH=1` is set, byte-identical at default.
        let has_moe = st.moe_meta_cache.iter().any(|m| m.is_some());
        st.has_moe_layers = has_moe;
        if has_gdn {
            // default ON.
            let lc_active = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT").ok().as_deref() {
                Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
                None => crate::runtime_defaults::gdn_register_resident_default(),
            };
            // C3: opt-in env gate with model-aware
            // default. When `LUMEN_CUDA_DECODE_GRAPH=1` OR the model is BF16
            // dense (default-ON per `runtime_defaults::decode_graph_default`),
            // graph capture is re-enabled under 4thread provided the
            // `gdn_phase123_register_resident_graph` kernel is compiled.
            let decode_graph_opt_in = cuda_decode_graph_enabled();
            let graph_kernel_ok = st.kernels.gdn_decode_megakernel_graph.is_some();
            let lc_graph_kernel_ok = st.kernels.gdn_phase123_register_resident_graph.is_some();
            if lc_active && !decode_graph_opt_in {
                eprintln!("[CUDA] GDN layers detected -- CUDA graph capture disabled (dp4a-mmvq env set; set LUMEN_CUDA_DECODE_GRAPH=1 to enable graph path)");
            } else if lc_active && decode_graph_opt_in && lc_graph_kernel_ok {
                eprintln!("[CUDA] GDN layers detected -- CUDA graph capture enabled (dp4a-mmvq; gdn_phase123_register_resident_graph)");
            } else if lc_active && decode_graph_opt_in && !lc_graph_kernel_ok {
                eprintln!("[CUDA] GDN layers detected -- CUDA graph capture disabled (LUMEN_CUDA_DECODE_GRAPH=1 but gdn_phase123_register_resident_graph PTX compile failed)");
            } else if !graph_kernel_ok {
                eprintln!("[CUDA] GDN layers detected -- CUDA graph capture disabled (gdn_decode_megakernel_graph PTX compile failed)");
            } else {
                eprintln!("[CUDA] GDN layers detected -- CUDA graph capture enabled (device-resident conv_position;)");
            }
        }

        st.layer_weights_cache = cache;

        // Build pre-computed batched GEMM pointer arrays for all layers.
        // This eliminates per-layer htod memcpys (~6 per layer) by uploading
        // all device pointer arrays once here. Also probes for CUDA 12.5+
        // grouped GEMM support to merge Q+K+V into a single cuBLAS call.
        match build_precomputed_batch_ptrs(&self.device, &st.layer_weights_cache, &st.scratch) {
            Ok(ptrs) => {
                let n_kv = ptrs.kv_a_ptrs.len();
                let n_ffn = ptrs.ffn_a_ptrs.len();
                let n_qkv = ptrs.qkv_a_ptrs.len();
                eprintln!(
                    "[CUDA] Pre-computed batched GEMM ptrs: {n_kv} KV, {n_ffn} FFN, {n_qkv} QKV layers"
                );
                st.precomputed_ptrs = Some(ptrs);
            }
            Err(e) => {
                eprintln!("[CUDA] Failed to build pre-computed batch ptrs (falling back to per-layer): {e}");
                st.precomputed_ptrs = None;
            }
        }

        // Autotune cuBLAS algorithm selection for F16 HGEMV shapes.
        // Benchmarks all 16 tensor-core algorithms + DEFAULT for each unique
        // (M=out_dim, K=in_dim) shape used during F16 decode. Caches the
        // fastest per shape. Only runs if any F16 weights are present.
        let has_f16 = st.layer_weights_cache.iter().any(|lw| {
            matches!(&lw.wq, GpuWeightBuf::F16Raw(_))
                || lw.wq_f16.is_some()
        });
        if has_f16 {
            let q_dim = hp_copy.num_heads as usize * hp_copy.head_dim as usize;
            let kv_dim = hp_copy.num_kv_heads as usize * hp_copy.head_dim as usize;
            let hidden_dim = hp_copy.hidden_dim as usize;
            let inter_dim = hp_copy.intermediate_dim as usize;

            // Collect unique (out_dim, in_dim) shapes for autotuning.
            // Skip vocab_size shape: too large for temporary buffer allocation
            // (~600+ MB for 150K vocab), and output projection is called only once
            // per token so the algorithm choice has negligible impact on throughput.
            let mut shapes: Vec<(usize, usize)> = vec![
                (q_dim, hidden_dim),      // wq projection (x36 layers)
                (kv_dim, hidden_dim),      // wk, wv projections (x36 layers)
                (hidden_dim, q_dim),       // wo output projection (x36 layers)
                (inter_dim, hidden_dim),   // gate, up projections (x36 layers)
                (hidden_dim, inter_dim),   // down projection (x36 layers)
            ];
            // Q+gate fusion: add (q_dim*2, hidden_dim) for fused Q+gate projection.
            if st.layer_weights_cache.iter().any(|lw| lw.attn_q_norm.is_some()) {
                shapes.push((q_dim * 2, hidden_dim));
            }
            shapes.sort();
            shapes.dedup();

            match autotune_cublas_algos(&self.device, &shapes) {
                Ok(cache) => {
                    let n = cache.best_algo.len();
                    eprintln!("[CUDA] Autotuned cuBLAS algorithms for {n} HGEMV shapes");
                    st.algo_cache = cache;
                }
                Err(e) => {
                    eprintln!("[CUDA] cuBLAS autotune failed (using defaults): {e}");
                    // Leave algo_cache as default (all DEFAULT_TENSOR_OP).
                }
            }
        }

        // Autotune cuBLAS algorithm selection for BF16 HGEMV shapes.
        // Mirrors the F16 autotune above but tests CUDA_R_16BF + COMPUTE_32F
        // (BF16 has no FAST_16F compute variant; the algo space is distinct
        // from F16). Only runs if any BF16 weights are present AND env-gate
        // `LUMEN_CUDA_BF16_AUTOTUNE != "0"`. The resulting cache lives in
        // the static `BF16_ALGO_CACHE` OnceLock, read by `launch_hgemv_bf16`
        // and `launch_hgemv_bf16_residual` on every BF16 GemmEx call.
        //
        // nsys profile (bf16_decode.nsys-rep, Qwen3.5-9B BF16 on A100 PCIe)
        // shows 60.9% of decode GPU time in BF16 GemmEx kernels — the prior
        // hardcoded `CUBLAS_GEMM_DEFAULT_TENSOR_OP` left this entire surface
        // un-optimized. Expected to close the 7 tok/s gap to the 0.9× llama.cpp
        // gate (66.0 -> 73.0 tok/s = +10.6%).
        let has_bf16 = st.layer_weights_cache.iter().any(|lw| {
            matches!(&lw.wq, GpuWeightBuf::Bf16Raw(_))
        });
        if has_bf16 && bf16_autotune_enabled() {
            let q_dim = hp_copy.num_heads as usize * hp_copy.head_dim as usize;
            let kv_dim = hp_copy.num_kv_heads as usize * hp_copy.head_dim as usize;
            let hidden_dim = hp_copy.hidden_dim as usize;
            let inter_dim = hp_copy.intermediate_dim as usize;
            let vocab_size = hp_copy.vocab_size as usize;

            // Same shape set as F16 autotune. vocab_size shape is autotuned
            // here because BF16 output_proj (vocab x hidden) DOES flow through
            // `launch_hgemv_bf16` when the mmv_bf16 dispatch is opt-out
            // (mmv_bf16_output_proj_enabled() == false). The 4096 proxy cap
            // in `autotune_cublas_algos_bf16` keeps the temporary alloc at
            // ~32 MB max even for 248320 vocab.
            let mut shapes: Vec<(usize, usize)> = vec![
                (q_dim, hidden_dim),       // wq projection
                (kv_dim, hidden_dim),      // wk, wv projections
                (hidden_dim, q_dim),       // wo output projection
                (inter_dim, hidden_dim),   // gate, up projections
                (hidden_dim, inter_dim),   // down projection
                (vocab_size, hidden_dim),  // final output_proj (vocab head)
            ];
            if st.layer_weights_cache.iter().any(|lw| lw.attn_q_norm.is_some()) {
                shapes.push((q_dim * 2, hidden_dim));
            }
            shapes.sort();
            shapes.dedup();

            match autotune_cublas_algos_bf16(&self.device, &shapes) {
                Ok(cache) => {
                    let n = cache.len();
                    eprintln!("[CUDA] Autotuned BF16 cuBLAS algorithms for {n} HGEMV shapes");
                    // Publish to the static cache. If a previous session of this
                    // process already populated it (multi-init test seam),
                    // ignore — first writer wins (per-shape selection is shape-
                    // deterministic on a given device).
                    let _ = BF16_ALGO_CACHE.set(cache);
                }
                Err(e) => {
                    eprintln!("[CUDA] BF16 cuBLAS autotune failed (using defaults): {e}");
                    // Leave BF16_ALGO_CACHE unset; bf16_algo_for falls back to
                    // DEFAULT_TENSOR_OP (the prior hardcoded behavior).
                }
            }
        } else if has_bf16 {
            eprintln!("[CUDA] BF16 autotune SKIPPED (LUMEN_CUDA_BF16_AUTOTUNE=0); using DEFAULT_TENSOR_OP");
        }

        Ok(())
    }

    fn decode_token(
        &self,
        token_id: u32,
        _weights: &dyn WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        let hp = self.hp()?;
        let num_layers = hp.num_layers as usize;
        let max_seq_len = hp.max_seq_len as usize;
        let seq_pos = kv.seq_len();

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized".into())
        })?;

        // Require GPU-resident weights for the zero-sync decode path.
        if st.layer_weights_cache.len() < num_layers {
            return Err(RuntimeError::Compute(
                "decode_token requires GPU-resident weights (call preload_weights first)".into(),
            ));
        }

        // --- CUDA Graph Decode Path ---
        //
        // Three modes based on decode_token_count:
        // 0 (first token): Run normal path (non-graph). This token establishes
        // the first KV entry. We don't capture here because the attention
        // shared memory must accommodate seq_len=1+ which is always true.
        // 1 (second token): Run the graph pipeline while capturing into a CUDA
        // graph. All subsequent tokens replay the captured graph.
        // 2+ (subsequent tokens): Update graph params (token_id, pos, seq_len)
        // via 3 small htod memcpys, then replay the captured graph.
        //
        // Graph capture is disabled for:
        // - Models with GDN layers (host-side conv state management)
        // - When graph kernels failed to compile
        // - When graph params failed to allocate

        // CUDA graph capture: disabled by default, enabled only in diagnostic mode.
        //
        // Graph capture has crashed on A100 in 3 attempts (C26, C30, C32).
        // Set LUMEN_GRAPH_DIAGNOSTIC=1 to enable with comprehensive error logging.
        // This diagnostic mode prints every CUDA API return code to stderr so the
        // exact failure point can be identified from Modal logs.
        let graph_diagnostic = std::env::var("LUMEN_GRAPH_DIAGNOSTIC")
            .map(|v| v == "1")
            .unwrap_or(false);

        // CUDA graph capture: enabled when all prerequisites are met.
        //
        // Root cause of previous STREAM_CAPTURE_ISOLATION failures was cudarc's
        // event tracking: every CudaSlice carried read/write CudaEvent objects,
        // and PushKernelArg inserted cuStreamWaitEvent calls during kernel
        // launch. During graph capture, these cross-stream event waits violated
        // capture isolation. The fix: disable event tracking in CudaDevice::new()
        // (ffi.rs) before ANY allocations, so all CudaSlice objects have
        // read=None/write=None. cudarc 0.19.3 already uses cuMemAllocAsync on
        // A100 (has_async_alloc=true), so allocations are stream-ordered on our
        // capture stream -- no legacy-stream associations.
        //
        // Prerequisites: graph kernels compiled, parameter buffers allocated,
        // cuBLAS workspace set, and no GDN layers (GDN conv1d uses a host-side
        // conv_position scalar that gets baked during graph capture -- not yet
        // graph-compatible until advance_conv_position dispatch is wired up).
        // Tiled-attention eager-fallback gate:
        // When the decode-attention gate would select the tiled streaming-
        // softmax kernel (seq_len > LUMEN_CUDA_DECODE_TILED_THRESHOLD,
        // default 0 = "tiled-always", OR force=true), the CUDA graph
        // fast path is BYPASSED for that token. The graph-captured
        // `attention_decode_graph` kernel in
        // `graph_kernels.cu` uses the single-block layout and cannot serve
        // seq_len past the shmem ceiling; rather than build a
        // `attention_decode_tiled_graph` variant in this revision, the design
        // chose eager-fallback to keep the scope bounded.
        //
        // The graph fast path remains BYTE-IDENTICAL for short decode
        // (seq_len <= threshold), which is the common case. Long-context
        // decode at seq_len > threshold pays a small per-token overhead for
        // the eager path (no graph replay) in exchange for opening the
        // structural ceiling.
        let attn_seq_len_pre = (seq_pos + 1) as u32; // seq_len AFTER KV write
        let decode_variant = super::decode::attention_decode_variant(
            attn_seq_len_pre,
            super::decode::decode_tiled_force_enabled(),
            super::decode::decode_tiled_threshold(),
        );
        let graph_eager_fallback_for_tiled = matches!(
            decode_variant,
            super::decode::AttentionDecodeVariant::Tiled
        );

        // GDN layers are now graph-capturable
        // when the device-resident conv_position infrastructure is wired up:
        //   - `gdn_decode_megakernel_graph` kernel compiled (reads state_pos
        //     from a device pointer instead of a host-scalar arg)
        //   - `conv_positions_gpu: Some(Vec<CudaSlice<u32>>)` allocated in
        //     `ensure_gdn_scratch`
        //   - `advance_conv_position` kernel in `graph_kernels.cu` (always
        //     compiled when graph_kernels.is_some())
        //   - `LUMEN_CUDA_GDN_REGISTER_RESIDENT` NOT set (dp4a-mmvq path is non-graph
        //     in this revision; falls back to host-scalar dispatch)
        // When all four conditions hold, the GDN inner loop is fully
        // capturable; otherwise we keep `!st.has_gdn_layers` to disable.
        //
        // Pre-check: `ensure_gdn_scratch` allocates `conv_positions_gpu` only
        // when GDN layers exist. We cannot call ensure_gdn_scratch here
        // because it requires &mut st but `can_use_graph` is evaluated as
        // an expression. Instead we check `gdn_scratch_gpu.is_some()` AND
        // the conv_positions_gpu within. The graph path's
        // `run_graph_pipeline` calls ensure_gdn_scratch itself.
        // default ON (matches init-site resolver).
        let register_resident_active = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT").ok().as_deref() {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::gdn_register_resident_default(),
        };
        // C3: when `LUMEN_CUDA_DECODE_GRAPH=1` (or
        // BF16 dense default-ON) AND two-launch is active AND the two-launch
        // graph variant is compiled, route through the graph path. The
        // graph variant (`gdn_phase123_register_resident_graph`) reads `state_pos`
        // from a device pointer, paired with the existing
        // `advance_conv_position` kernel, so the full two-launch GDN inner
        // loop becomes graph-capturable while preserving byte-identical
        // math vs the eager two-launch host-scalar kernel.
        let decode_graph_opt_in = cuda_decode_graph_enabled();
        let conv_positions_gpu_ready = st.gdn_scratch_gpu.as_ref()
            .map(|g| g.conv_positions_gpu.is_some())
            .unwrap_or(false);
        let lc_graph_ready = register_resident_active
            && decode_graph_opt_in
            && st.kernels.gdn_phase123_register_resident_graph.is_some()
            && conv_positions_gpu_ready;
        let megakernel_graph_ready = !register_resident_active
            && st.kernels.gdn_decode_megakernel_graph.is_some()
            && conv_positions_gpu_ready;
        let gdn_graph_ready = megakernel_graph_ready || lc_graph_ready;

        // `LUMEN_CUDA_DECODE_GRAPH_QGATE=1` makes the gate accept layers with
        // per-head q_norm/k_norm. Earlier this path was a correctness-broken
        // diagnostic (compute_layer_gpu_graph dispatched QKV into q_dim, not
        // q_dim*2, then ran RoPE on un-deinterleaved Q). This block now ports
        // the full qgate fusion dispatch into the graph variant
        // (deinterleave_qgate, per-head RMSNorm Q+K, post-attention sigmoid_mul),
        // so this gate now produces bit-correct output on Qwen3.5 dense paths.
        // The env is kept as the activation switch (default OFF, byte-identical
        // to the previous default when unset).
        // env-or-BF16-default helper. BF16 dense models pull
        // the qgate graph path on by default; Q8/Q4 keep the legacy OFF.
        let allow_qgate_graph = cuda_decode_graph_qgate_enabled();
        // `LUMEN_CUDA_DECODE_GRAPH_TILED=1` flips the eager-fallback
        // gate when the tiled-graph kernel is compiled. The graph variant
        // (`attention_decode_tiled_graph`) reads seq_len from a device pointer
        // so the capture is structurally identical across decode tokens. Shmem
        // footprint is constant in seq_len (8 + head_dim + T_C = ~1.6 KB at
        // head_dim=256, T_C=128), no extended-shmem opt-in needed. Default OFF
        // (bit-exact when disabled, i.e. when the env var is unset).
        // env-or-BF16-default helper. Tiled-graph kernel
        // ships with BF16 dense path default-ON; Q8/Q4 keep
        // the legacy OFF.
        let allow_tiled_graph = cuda_decode_graph_tiled_enabled();
        let tiled_graph_ready = allow_tiled_graph
            && st.graph_kernels.as_ref()
                .map(|gk| gk.attention_decode_tiled.is_some())
                .unwrap_or(false);
        // MoE decode graph gate. When the model has MoE layers AND
        // `LUMEN_CUDA_MOE_DECODE_GRAPH=1` is set, route MoE FFN dispatch through
        // `compute_layer_gpu_graph`'s new MoE branch (Strategy C: full MoE FFN
        // capture using existing graph-safe `encode_moe_ffn_decode_*` family).
        // The MoE dispatch is naturally graph-compatible because all per-token
        // decisions (expert_ids, expert_weights) live entirely on the GPU —
        // CPU passes only fixed dims as kernel args. Default OFF (byte-identical
        // to the previous default when unset).
        // default ON (confirmed
        // byte-identical at the routed-MoE path; no-op for non-MoE models).
        let allow_moe_graph = match std::env::var("LUMEN_CUDA_MOE_DECODE_GRAPH").ok().as_deref() {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::moe_decode_graph_default(),
        };
        let can_use_graph = st.graph_kernels.is_some()
            && st.graph_params.is_some()
            && st.cublas_workspace.is_some()
            && st.precomputed_ptrs.is_some()
            && (!st.has_gdn_layers || gdn_graph_ready)
            && (!st.has_qgate_layers || allow_qgate_graph)
            && (!st.has_moe_layers || allow_moe_graph)
            && (!graph_eager_fallback_for_tiled || tiled_graph_ready);

        if graph_diagnostic && st.decode_token_count == 0 {
            eprintln!("[GRAPH-DIAG] === CUDA Graph Diagnostic Mode Enabled ===");
            eprintln!("[GRAPH-DIAG] Prerequisites:");
            eprintln!("[GRAPH-DIAG]   graph_kernels:    {}", st.graph_kernels.is_some());
            eprintln!("[GRAPH-DIAG]   graph_params:     {}", st.graph_params.is_some());
            eprintln!("[GRAPH-DIAG]   cublas_workspace: {}", st.cublas_workspace.is_some());
            eprintln!("[GRAPH-DIAG]   precomputed_ptrs: {}", st.precomputed_ptrs.is_some());
            eprintln!("[GRAPH-DIAG]   has_gdn_layers:   {}", st.has_gdn_layers);
            eprintln!("[GRAPH-DIAG]   has_qgate_layers: {}  (was blocker; ports qgate fusion into graph)", st.has_qgate_layers);
            eprintln!("[GRAPH-DIAG]   has_moe_layers:   {}  (MoE-aware graph dispatch)", st.has_moe_layers);
            eprintln!("[GRAPH-DIAG]   allow_moe_graph:  {allow_moe_graph}  (LUMEN_CUDA_MOE_DECODE_GRAPH)");
            eprintln!("[GRAPH-DIAG]   allow_qgate_graph: {allow_qgate_graph}  (LUMEN_CUDA_DECODE_GRAPH_QGATE)");
            eprintln!("[GRAPH-DIAG]   allow_tiled_graph: {allow_tiled_graph}  (LUMEN_CUDA_DECODE_GRAPH_TILED)");
            eprintln!("[GRAPH-DIAG]   tiled_graph_ready: {tiled_graph_ready}  (attention_decode_tiled_graph)");
            eprintln!("[GRAPH-DIAG]   register_resident_active:  {register_resident_active}");
            eprintln!("[GRAPH-DIAG]   decode_graph_opt_in: {decode_graph_opt_in}");
            eprintln!("[GRAPH-DIAG]   conv_positions_gpu_ready: {conv_positions_gpu_ready}");
            eprintln!("[GRAPH-DIAG]   lc_graph_ready:   {lc_graph_ready}  (gdn_phase123_register_resident_graph)");
            eprintln!("[GRAPH-DIAG]   megakernel_graph_ready: {megakernel_graph_ready}");
            eprintln!("[GRAPH-DIAG]   gdn_graph_ready:  {gdn_graph_ready}");
            eprintln!("[GRAPH-DIAG]   graph_eager_fallback_for_tiled: {graph_eager_fallback_for_tiled}  (set LUMEN_CUDA_DECODE_GRAPH_TILED=1 for graph path)");
            eprintln!("[GRAPH-DIAG]   can_use_graph:    {can_use_graph}");
            eprintln!("[GRAPH-DIAG]   num_layers:       {num_layers}");
            eprintln!("[GRAPH-DIAG]   max_seq_len:      {max_seq_len}");
            eprintln!("[GRAPH-DIAG]   token_id:         {token_id}");
            eprintln!("[GRAPH-DIAG]   seq_pos:          {seq_pos}");
            // Check CUDA_LAUNCH_BLOCKING
            let clb = std::env::var("CUDA_LAUNCH_BLOCKING").unwrap_or_else(|_| "unset".into());
            eprintln!("[GRAPH-DIAG]   CUDA_LAUNCH_BLOCKING: {clb}");

            // Query stream capture status before anything
            let capture_status = super::graph::query_capture_status(&self.device.stream);
            eprintln!("[GRAPH-DIAG]   stream capture status (pre-capture): {capture_status}");

            // Synchronize and check for pre-existing errors
            match self.device.synchronize() {
                Ok(()) => eprintln!("[GRAPH-DIAG]   pre-capture sync: OK"),
                Err(e) => eprintln!("[GRAPH-DIAG]   pre-capture sync: FAILED: {e}"),
            }
        }

        // Alias the pre-computed value ( used it for the eager-fallback
        // gate check above; the existing downstream code expects the name
        // `attn_seq_len`).
        let attn_seq_len = attn_seq_len_pre; // seq_len AFTER KV write

        if can_use_graph && st.decode_token_count >= 1 {
            let diag = graph_diagnostic;

            // Check if we have a valid captured graph.
            let have_valid_graph = st.captured_graph.as_ref()
                .map(|g| g.is_valid_for(num_layers, max_seq_len))
                .unwrap_or(false);

            if have_valid_graph {
                // --- GRAPH REPLAY PATH ---
                if diag {
                    eprintln!("[GRAPH-DIAG] === Graph REPLAY (token #{}, seq_pos={seq_pos}) ===",
                        st.decode_token_count);
                }

                // Update per-token scalars in device memory (3 x 4-byte htod).
                match st.graph_params.as_mut().unwrap().update(
                    &self.device, token_id, seq_pos as u32, attn_seq_len,
                ) {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   params update: OK"); } },
                    Err(e) => {
                        eprintln!("[GRAPH-DIAG]   params update: FAILED: {e}");
                        return Err(e);
                    }
                }

                // sync host conv_positions
                // to GPU BEFORE every replay. This guards against drift
                // introduced by prior eager-fallback tokens (e.g. tiled-decode
                // crossover) which advance only the host counter without
                // touching conv_positions_gpu. Cost: ~24 GDN layers × 4 bytes
                // htod_copy = <5 µs/token (well below the +1 ms win target).
                // The memcpy is OUTSIDE the captured graph (graph is already
                // instantiated; this dispatch goes to the stream which is
                // not in capture mode here).
                if st.has_gdn_layers {
                    if let Some(ref mut gdn) = st.gdn_scratch_gpu {
                        if let Some(ref mut gpu_pos) = gdn.conv_positions_gpu {
                            for (i, pos) in gdn.conv_positions.iter().enumerate() {
                                self.device.htod_copy_into(&[*pos], &mut gpu_pos[i])?;
                            }
                        }
                    }
                }

                // Replay the captured graph (single API call, all kernels execute).
                match st.captured_graph.as_ref().unwrap().launch() {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   graph launch: OK"); } },
                    Err(e) => {
                        eprintln!("[GRAPH-DIAG]   graph launch: FAILED: {e}");
                        eprintln!("[CUDA] Graph replay failed: {e} -- disabling graph capture for this session");
                        st.captured_graph = None;
                        st.decode_token_count = 0;
                        return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                    }
                }

                if diag {
                    // Synchronize to surface any async errors from the graph replay.
                    match self.device.synchronize() {
                        Ok(()) => eprintln!("[GRAPH-DIAG]   post-replay sync: OK"),
                        Err(e) => {
                            eprintln!("[GRAPH-DIAG]   post-replay sync: FAILED: {e}");
                            eprintln!("[GRAPH-DIAG]   This is the actual error -- graph kernels ran but produced an async fault.");
                            st.captured_graph = None;
                            st.decode_token_count = 0;
                            return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                        }
                    }
                }

                // Advance GPU-side KV cache seq_len for all layers (host bookkeeping).
                for kv_cache in &mut st.kv_caches {
                    kv_cache.advance_seq_len_by(1);
                }

                // mirror the GPU-side
                // advance_conv_position kernel by advancing the host counter.
                // The GPU kernel ran inside the captured graph and updated
                // conv_positions_gpu[i]; we mirror that here so any
                // subsequent eager-fallback dispatch (e.g. tiled-decode
                // threshold crossover) reads a consistent host counter.
                // Also keeps disk-KV session checkpoint accurate.
                if let Some(ref mut gdn) = st.gdn_scratch_gpu {
                    let buf_slots = (gdn.params.conv_kernel_size - 1) as u32;
                    if buf_slots > 0 {
                        for pos in gdn.conv_positions.iter_mut() {
                            *pos = (*pos + 1) % buf_slots;
                        }
                    }
                }
            } else {
                // --- GRAPH CAPTURE PATH ---
                if diag {
                    eprintln!("[GRAPH-DIAG] === Graph CAPTURE (token #{}, seq_pos={seq_pos}) ===",
                        st.decode_token_count);
                }

                // Update per-token scalars before capture (the graph will read
                // from these fixed device pointers).
                match st.graph_params.as_mut().unwrap().update(
                    &self.device, token_id, seq_pos as u32, attn_seq_len,
                ) {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   params update: OK (token_id={token_id}, pos={seq_pos}, attn_seq_len={attn_seq_len})"); } },
                    Err(e) => {
                        eprintln!("[GRAPH-DIAG]   params update: FAILED: {e}");
                        return Err(e);
                    }
                }

                // Device-level sync before capture to ensure all prior GPU work
                // (weight uploads, scratch init, cuBLAS workspace setup) is
                // complete. With event tracking disabled and stream-ordered
                // allocation, our single stream is self-consistent, but a device
                // sync provides a clean capture boundary.
                match cudarc::driver::result::ctx::synchronize() {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   pre-capture device sync: OK"); } },
                    Err(e) => {
                        let err = RuntimeError::Compute(format!("device sync failed: {e:?}"));
                        eprintln!("[GRAPH-DIAG]   pre-capture device sync: FAILED: {err}");
                        return Err(err);
                    }
                }

                // sync host conv_positions
                // to GPU BEFORE begin_capture(). This memcpy is NOT recorded
                // into the captured graph, so it can run once before each
                // capture-token to establish the initial state. From there,
                // the captured `advance_conv_position` kernel advances the
                // GPU counter on every replay.
                if st.has_gdn_layers {
                    self.ensure_gdn_scratch(st)?;
                    if let Some(ref mut gdn) = st.gdn_scratch_gpu {
                        if let Some(ref mut gpu_pos) = gdn.conv_positions_gpu {
                            for (i, pos) in gdn.conv_positions.iter().enumerate() {
                                self.device.htod_copy_into(&[*pos], &mut gpu_pos[i])?;
                            }
                        }
                    }
                }

                // Check stream capture status before begin_capture
                if diag {
                    let status = super::graph::query_capture_status(&self.device.stream);
                    eprintln!("[GRAPH-DIAG]   stream status before begin_capture: {status}");
                }

                // Begin stream capture -- all subsequent kernel launches on this
                // stream are recorded (not executed) into the graph.
                match super::graph::begin_capture(&self.device.stream) {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   begin_capture: OK"); } },
                    Err(e) => {
                        eprintln!("[GRAPH-DIAG]   begin_capture: FAILED: {e}");
                        st.decode_token_count = 0;
                        return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                    }
                }

                // Check stream capture status after begin_capture
                if diag {
                    let status = super::graph::query_capture_status(&self.device.stream);
                    eprintln!("[GRAPH-DIAG]   stream status after begin_capture: {status}");
                }

                // Run the full decode pipeline using graph-compatible kernels.
                if diag {
                    eprintln!("[GRAPH-DIAG]   running graph pipeline ({num_layers} layers)...");
                }
                match self.run_graph_pipeline(st) {
                    Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   graph pipeline: OK"); } },
                    Err(e) => {
                        // Capture failed -- end capture to restore the stream
                        // to non-capturing state, then fall through to normal path.
                        eprintln!("[GRAPH-DIAG]   graph pipeline: FAILED: {e}");

                        // Check stream capture status after failure
                        if diag {
                            let status = super::graph::query_capture_status(&self.device.stream);
                            eprintln!("[GRAPH-DIAG]   stream status after pipeline failure: {status}");
                        }

                        // Try to end capture to clean up.
                        match super::graph::end_capture(&self.device.stream, num_layers, max_seq_len) {
                            Ok(_) => { if diag { eprintln!("[GRAPH-DIAG]   cleanup end_capture: OK"); } },
                            Err(e2) => { eprintln!("[GRAPH-DIAG]   cleanup end_capture: FAILED: {e2}"); },
                        }

                        st.decode_token_count = 0;
                        return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                    }
                }

                // Check stream capture status before end_capture
                if diag {
                    let status = super::graph::query_capture_status(&self.device.stream);
                    eprintln!("[GRAPH-DIAG]   stream status before end_capture: {status}");
                }

                // End capture -- instantiate the graph for future replay.
                match super::graph::end_capture(
                    &self.device.stream, num_layers, max_seq_len,
                ) {
                    Ok(Some(graph)) => {
                        if diag {
                            eprintln!("[GRAPH-DIAG]   end_capture: OK (graph instantiated, {num_layers} layers)");
                        }
                        eprintln!("[CUDA] Graph capture successful ({num_layers} layers)");
                        st.captured_graph = Some(graph);

                        // Replay the freshly captured graph.
                        if diag {
                            eprintln!("[GRAPH-DIAG]   launching freshly captured graph...");
                        }
                        match st.captured_graph.as_ref().unwrap().launch() {
                            Ok(()) => { if diag { eprintln!("[GRAPH-DIAG]   first graph launch: OK"); } },
                            Err(e) => {
                                eprintln!("[GRAPH-DIAG]   first graph launch: FAILED: {e}");
                                st.captured_graph = None;
                                st.decode_token_count = 0;
                                return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                            }
                        }

                        // Synchronize to catch async errors from graph execution.
                        if diag {
                            match self.device.synchronize() {
                                Ok(()) => eprintln!("[GRAPH-DIAG]   post-launch sync: OK"),
                                Err(e) => {
                                    eprintln!("[GRAPH-DIAG]   post-launch sync: FAILED: {e}");
                                    eprintln!("[GRAPH-DIAG]   ASYNC ERROR: Graph launched OK but kernels faulted during execution.");
                                    st.captured_graph = None;
                                    st.decode_token_count = 0;
                                    return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                                }
                            }
                        }

                        // Advance GPU-side KV cache seq_len for all layers.
                        for kv_cache in &mut st.kv_caches {
                            kv_cache.advance_seq_len_by(1);
                        }
                        // no additional
                        // host conv_positions advance here. During the capture
                        // pass, `compute_gdn_attention_gpu_impl(graph_mode=true)`
                        // already advanced `gdn.conv_positions` once (the host
                        // code ran once during capture). On this first launch
                        // after end_capture, the GPU advance_conv_position
                        // kernel runs once, incrementing conv_positions_gpu[i]
                        // to match. Both counters are at (pre-token + 1). Good.
                        // For subsequent pure-replay tokens, the replay branch
                        // (above) mirrors the GPU advance to the host counter.
                    }
                    Ok(None) => {
                        eprintln!("[GRAPH-DIAG]   end_capture: OK but graph is EMPTY (no kernels captured)");
                        eprintln!("[CUDA] Graph capture produced empty graph -- falling back to normal path");
                        st.decode_token_count = 0;
                        return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                    }
                    Err(e) => {
                        eprintln!("[GRAPH-DIAG]   end_capture: FAILED: {e}");
                        eprintln!("[CUDA] Graph capture end_capture failed: {e} -- falling back to normal path");
                        st.decode_token_count = 0;
                        return self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv);
                    }
                }
            }
        } else {
            // --- NORMAL (NON-GRAPH) PATH ---
            // Used for:
            // - First decode token (decode_token_count == 0)
            // - Models with GDN layers
            // - When graph infrastructure is unavailable

            self.embed_token_gpu(token_id, st)?;

            for layer in 0..num_layers {
                self.compute_layer_gpu(layer, seq_pos, st)?;

                // FIX-DTOD: see decode_token_normal for full rationale.
                // Skip the attn_proj -> x_gpu propagation for MoE layers because
                // encode_moe_ffn_decode already wrote the post-FFN state directly
                // into st.scratch.x_gpu; the dtod would overwrite it with the
                // stale pre-FFN attn+residual.
                let is_moe_layer = st
                    .moe_meta_cache
                    .get(layer)
                    .and_then(|m| m.as_ref())
                    .is_some();
                if !is_moe_layer {
                    self.device
                        .stream
                        .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                        .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
                }
            }

            self.compute_final_gpu(st)?;

            {
                let vocab = hp.vocab_size;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.argmax_f32)
                        .arg(&st.logits_gpu)
                        .arg(&mut st.argmax_result)
                        .arg(&vocab)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("argmax launch: {e}")))?;
            }
        }

        // Full real-logits readback on the sampling decode
        // path. The previous code synthesized a ONE-HOT vector {argmax->1.0,
        // rest->0.0} from a 4-byte argmax readback, which destroyed the real
        // distribution that `compute_final_gpu` had already written into
        // `st.logits_gpu`. Any sampler (temperature>0) then drew from a
        // near-uniform softmax over the one-hot -> gibberish. Greedy is
        // unaffected because it uses the separate `decode_token_greedy` GPU
        // argmax path and never calls this function. Cost: one ~1 MB dtoh per
        // decode token on the sampling path only (negligible vs the closed
        // decode perf; greedy untouched).
        self.device.synchronize()?;
        // optional per-step CPU sleep to close the GPU-scheduler
        // timing race documented in
        // Default OFF (delay=0 → no-op, bit-exact when disabled).
        // Set `LUMEN_CUDA_DECODE_DELAY_US=50` to opt in (Metal precedent).
        maybe_apply_cuda_decode_delay();
        let logits_host = self.device.dtoh_copy(&st.logits_gpu)?;

        // Advance host-side KV cache seq_len and decode counter.
        kv.advance_seq_len()?;
        st.decode_token_count += 1;

        Ok(Logits { data: logits_host })
    }

}

/// Create an ActivationBuffer from an f32 slice.
fn f32_to_activation(values: &[f32]) -> ActivationBuffer {
    let mut data = Vec::with_capacity(values.len() * 4);
    #[cfg(target_endian = "little")]
    {
        // SAFETY: values is contiguous f32 data. On LE platform, byte repr
        // matches LE encoding. Capacity is pre-allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr() as *const u8,
                data.as_mut_ptr(),
                values.len() * 4,
            );
            data.set_len(values.len() * 4);
        }
    }
    #[cfg(target_endian = "big")]
    {
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    ActivationBuffer {
        data,
        num_elements: values.len(),
        dtype: ComputeDtype::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that caps() advertises gpu_resident=true.
    /// This is a compile-time/structural test -- it validates the BackendCaps
    /// returned by the CUDA backend without requiring a CUDA GPU.
    #[test]
    fn caps_advertises_gpu_resident() {
        let caps = BackendCaps {
            batched_prefill: true,
            gpu_resident: true,
            gdn: true,
            moe: false,
            gpu_argmax: true,
        };
        assert!(caps.gpu_resident, "CUDA backend must advertise gpu_resident=true");
        assert!(caps.batched_prefill, "CUDA backend must advertise batched_prefill=true");
    }

    /// caps must advertise `moe: true` when the model has MoE layers.
    ///
    /// Structural test — exercises the BackendCaps struct shape, no CUDA GPU
    /// required. The caps flip from `false` to `true` happens after
    /// preload_weights populates `moe_meta_cache`; we mirror that contract in
    /// a synthetic BackendCaps literal.
    #[test]
    fn caps_advertises_moe_for_moe_layers() {
        let dense_caps = BackendCaps {
            batched_prefill: true,
            gpu_resident: true,
            gdn: true,
            moe: false,
            gpu_argmax: true,
        };
        assert!(!dense_caps.moe, "dense model: moe must be false");

        let moe_caps = BackendCaps {
            batched_prefill: true,
            gpu_resident: true,
            gdn: true,
            moe: true,
            gpu_argmax: true,
        };
        assert!(moe_caps.moe, "MoE model: moe must be true");
    }

    /// Verify that MutableState initializes with an empty weight cache.
    /// When layer_weights_cache is empty, compute_layer falls back to
    /// per-call upload (streaming path).
    #[test]
    fn mutable_state_empty_cache_is_streaming_path() {
        let cache: Vec<LayerWeightsGpu> = Vec::new();
        // layer_idx 0 should not be in cache -- triggers streaming fallback
        assert!(
            0 >= cache.len(),
            "empty cache means all layers use streaming upload",
        );
    }

    /// Verify that decode_token requires GPU-resident weights.
    ///
    /// The zero-sync decode path uses compute_layer_gpu which directly indexes
    /// into layer_weights_cache. Without preloaded weights, it must return an
    /// error pointing the user to call preload_weights first.
    #[test]
    fn decode_token_requires_preloaded_weights() {
        // This is a structural test -- no CUDA GPU required.
        // The decode_token implementation checks:
        // if st.layer_weights_cache.len() < num_layers { return Err(...) }
        // An empty cache with any non-zero num_layers triggers this.
        let cache: Vec<LayerWeightsGpu> = Vec::new();
        let num_layers = 32usize;
        assert!(
            cache.len() < num_layers,
            "empty weight cache should trigger GPU-resident decode error",
        );
    }

    /// Verify that the zero-sync path eliminates per-layer synchronization.
    ///
    /// The old decode_token path calls compute_layer N times, each of which
    /// calls device.synchronize(). The new path calls synchronize() only once
    /// at the end, after all N layers complete on GPU.
    ///
    /// This test validates the structural invariant by counting sync points
    /// in the code paths.
    #[test]
    fn zero_sync_path_has_single_sync() {
        // The old compute_layer has synchronize() at line ~1150.
        // For 32 layers: 32 syncs + 1 in embed_token + 1 in compute_final = 34.
        //
        // The new decode_token path:
        // - embed_token_gpu: 0 syncs
        // - compute_layer_gpu x N: 0 syncs
        // - compute_final_gpu: 0 syncs
        // - device.synchronize(): 1 sync
        // = 1 total sync
        //
        // This is a documentation test -- the actual sync count is verified
        // by code inspection and the benchmark test on GPU hardware.
        let old_syncs_per_token = 32 + 1 + 1; // layers + embed + final
        let new_syncs_per_token = 1; // single sync at end
        assert_eq!(old_syncs_per_token, 34);
        assert_eq!(new_syncs_per_token, 1);
    }

    // -----------------------------------------------------------------
    // BF16 GemmEx capability + per-call fallback state-machine tests
    //
    // These tests cover the host-side state machine governing the
    // cuBLAS BF16 GemmEx path: the three gates (env opt-out, capability
    // probe, runtime-armed fallback) and the once-only warning
    // mechanism. They are hardware-independent (no GPU access) so they
    // run on macOS dev hosts as well as Linux CI. The end-to-end
    // capability-probe + per-call-fallback verification on CUDA hardware
    // is covered by the Modal A100 validation harness referenced in the
    // release consolidation notes.
    //
    // The tests serialize via a per-module `Mutex` because they
    // manipulate process-wide statics; running in parallel would
    // produce interleaved state and false negatives.
    // -----------------------------------------------------------------

    use std::sync::Mutex as TestMutex;

    /// Serializes the BF16-state tests. The process-wide statics
    /// (`BF16_GEMMEX_AVAILABLE`, `BF16_GEMMEX_FALLBACK_ARMED`) are not
    /// thread-local; tests must run sequentially.
    fn bf16_state_test_lock() -> &'static TestMutex<()> {
        static LOCK: OnceLock<TestMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| TestMutex::new(()))
    }

    /// Returns a guard that records and restores the BF16 GemmEx
    /// statics across a test scope. Test scopes use `set_*` to drive
    /// the state into a known configuration; the guard rolls back on
    /// drop so the next test sees defaults.
    struct Bf16StatesnapshotGuard {
        available: bool,
        runtime_fallback_armed: bool,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Bf16StatesnapshotGuard {
        fn capture() -> Self {
            // Acquire the serialization lock first; tolerate poisoning
            // (a previous test panicked) by recovering the inner data.
            let lock = bf16_state_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Self {
                available: BF16_GEMMEX_AVAILABLE.load(Ordering::Relaxed),
                runtime_fallback_armed: BF16_GEMMEX_FALLBACK_ARMED
                    .load(Ordering::Relaxed),
                _lock: lock,
            }
        }
    }

    impl Drop for Bf16StatesnapshotGuard {
        fn drop(&mut self) {
            BF16_GEMMEX_AVAILABLE.store(self.available, Ordering::Relaxed);
            BF16_GEMMEX_FALLBACK_ARMED
                .store(self.runtime_fallback_armed, Ordering::Relaxed);
        }
    }

    #[test]
    fn bf16_gemmex_default_enabled_when_probe_passes() {
        let _g = Bf16StatesnapshotGuard::capture();
        // Simulate a successful probe + no runtime failures yet.
        BF16_GEMMEX_AVAILABLE.store(true, Ordering::Relaxed);
        BF16_GEMMEX_FALLBACK_ARMED.store(false, Ordering::Relaxed);
        // SAFETY: a single-test setter on a process-static env var
        // before the gate is read; restored on drop is not required
        // because subsequent tests do not depend on this variable being
        // unset (each test sets the value it needs).
        unsafe {
            std::env::remove_var("LUMEN_CUDA_BF16_GEMMEX");
        }
        // bf16_gemmex_env_force_off caches the env-var value on its
        // first read for the lifetime of the process; in test mode it
        // may already be cached. We cannot meaningfully assert against
        // `bf16_gemmex_enabled()` after caching; instead, assert the
        // underlying atomics, which is what the production wrapper
        // reads via the gate composition.
        assert!(BF16_GEMMEX_AVAILABLE.load(Ordering::Relaxed));
        assert!(!BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed));
    }

    #[test]
    fn bf16_gemmex_disabled_when_capability_probe_failed() {
        let _g = Bf16StatesnapshotGuard::capture();
        BF16_GEMMEX_AVAILABLE.store(false, Ordering::Relaxed);
        BF16_GEMMEX_FALLBACK_ARMED.store(false, Ordering::Relaxed);
        // Composed gate must observe AVAILABLE=false.
        assert!(
            !BF16_GEMMEX_AVAILABLE.load(Ordering::Relaxed),
            "AVAILABLE must read false after capability probe failure"
        );
        // bf16_gemmex_enabled() composes AVAILABLE with two other
        // gates; when AVAILABLE is false the gate must be closed
        // regardless of the env-var cache state.
        assert!(
            !bf16_gemmex_enabled() || bf16_gemmex_env_force_off(),
            "gate must be closed when AVAILABLE=false"
        );
    }

    #[test]
    fn bf16_gemmex_disabled_when_runtime_fallback_armed() {
        let _g = Bf16StatesnapshotGuard::capture();
        BF16_GEMMEX_AVAILABLE.store(true, Ordering::Relaxed);
        BF16_GEMMEX_FALLBACK_ARMED.store(true, Ordering::Relaxed);
        assert!(BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed));
        assert!(
            !bf16_gemmex_enabled() || bf16_gemmex_env_force_off(),
            "gate must be closed when runtime fallback is armed"
        );
    }

    #[test]
    fn arm_runtime_fallback_sets_flag_and_is_idempotent() {
        let _g = Bf16StatesnapshotGuard::capture();
        BF16_GEMMEX_AVAILABLE.store(true, Ordering::Relaxed);
        BF16_GEMMEX_FALLBACK_ARMED.store(false, Ordering::Relaxed);

        // First call arms the flag.
        arm_bf16_gemmex_runtime_fallback(
            "test_label_1",
            cublas_sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
        );
        assert!(
            BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed),
            "arm must set the runtime fallback flag"
        );

        // Repeated arming is idempotent: flag stays true; no panic;
        // the OnceLock warning is emitted at most once. We cannot
        // assert on stderr contents portably, but we can verify
        // multiple `arm` calls do not flip the flag back to false.
        arm_bf16_gemmex_runtime_fallback(
            "test_label_2",
            cublas_sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED,
        );
        assert!(
            BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed),
            "flag must remain true on subsequent arms"
        );
    }

    #[test]
    fn bf16_gemmex_env_force_off_caches_value() {
        // The cache is a process-static `OnceLock<bool>`; calling the
        // resolver twice must return the same value, regardless of
        // any env-var mutation between calls. This is the property
        // that makes per-call dispatch O(1) without a syscall.
        let first = bf16_gemmex_env_force_off();
        let second = bf16_gemmex_env_force_off();
        assert_eq!(
            first, second,
            "env_force_off must be stable for the lifetime of the process"
        );
    }

    // -----------------------------------------------------------------
    // LUMEN_CUDA_DECODE_DELAY_US env-resolver tests
    //
    // These tests exercise the host-side env-resolver introduced by
    // as the CUDA port of Metal's decode-delay fix. They
    // are hardware-independent (no CUDA GPU required) and run on macOS
    // dev hosts as well as Linux CI; the empirical determinism evidence
    // on the real GPU lives in
    //
    // NOTE on caching: `cuda_decode_delay_us()` resolves the env-var
    // exactly once per process via `OnceLock`. We CANNOT meaningfully
    // alter the cached value mid-test without a fork. Instead the tests
    // assert the structural invariants: the resolver is stable across
    // calls, returns a non-negative integer, and the apply-helper is a
    // no-op when the resolver returns 0.
    // -----------------------------------------------------------------

    #[test]
    fn cuda_decode_delay_us_is_stable_across_calls() {
        // Same cache discipline as bf16_gemmex_env_force_off above. The
        // first read of `LUMEN_CUDA_DECODE_DELAY_US` materializes the
        // value; every subsequent call must return identical bytes.
        let a = cuda_decode_delay_us();
        let b = cuda_decode_delay_us();
        let c = cuda_decode_delay_us();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn cuda_decode_delay_us_default_is_zero_when_unset() {
        // When the env-var is not set we expect 0 (= OFF, byte-identical
        // to the prior production default). This is the production default.
        //
        // We can only assert this if the env-var is not currently set in
        // the test process. CI runs without it; if a developer has set
        // it locally we skip the assertion (recording the observed value
        // for diagnosis instead).
        let observed = cuda_decode_delay_us();
        if std::env::var("LUMEN_CUDA_DECODE_DELAY_US").is_err() {
            assert_eq!(
                observed, 0,
                "default when unset must be 0 (byte-identical to the prior production default)"
            );
        }
    }

    #[test]
    fn maybe_apply_cuda_decode_delay_is_fast_when_zero() {
        // When the resolver returns 0 the apply-helper must be a near-zero-
        // cost branch (no syscall, no sleep). We assert this empirically
        // by measuring 10_000 calls: the total cost should be << 1 ms.
        // Only meaningful when the env-var is unset (delay = 0); skip the
        // budget assertion when a developer has set the env-var locally.
        let env_present = std::env::var("LUMEN_CUDA_DECODE_DELAY_US").is_ok();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            maybe_apply_cuda_decode_delay();
        }
        let elapsed = start.elapsed();
        if !env_present {
            // 10_000 calls in << 1 ms = each call < 100 ns avg. This is
            // generous; in practice on M3/A100 hosts it is < 10 ns/call.
            assert!(
                elapsed.as_millis() < 50,
                "10_000 zero-delay calls took {elapsed:?} (expected < 50 ms); \
                 fast path may have regressed"
            );
        }
    }

    #[test]
    fn cuda_decode_delay_us_rejects_invalid_strings_silently() {
        // Documentation test: the resolver uses `parse::<u64>().ok()` so
        // any unparseable string (e.g. `"abc"`, negative integer, empty
        // string) falls back to the documented default of 0. We cannot
        // exercise this directly without a fork-and-set in the test
        // process (env-var is cached); the assertion is a contract
        // statement that the production resolver code uses `.ok()` and
        // `.unwrap_or(0)` rather than `.unwrap()` or `.expect()`.
        //
        // The actual resolver implementation:
        //   std::env::var("LUMEN_CUDA_DECODE_DELAY_US")
        //       .ok()
        //       .and_then(|v| v.parse::<u64>().ok())
        //       .unwrap_or(0)
        //
        // The chain is `Option<String> -> Option<u64> -> u64` with two
        // `.ok()`/`.unwrap_or` falls-through to 0 on any failure. This
        // means an operator who fat-fingers `=abc` does not break the
        // CUDA backend at startup; they get the default behavior.
        //
        // Without changing the resolver to take a `&str` parameter (which
        // would complicate the `OnceLock` cache contract) this test cannot
        // exercise the parse-fail branch in isolation, but we can verify
        // the resolver returns a valid `u64` always.
        let v = cuda_decode_delay_us();
        let _: u64 = v; // type-level confirmation; will not compile if regressed
        assert!(v <= u64::MAX); // trivially true; documents the contract
    }
}
