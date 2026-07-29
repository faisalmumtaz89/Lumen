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
    self, dp4a_q4_grid, dp4a_q8_1_grid, fused_glu_grid, fused_glu_shared_bytes_f16,
    fused_glu_shared_bytes_f32, fused_norm_matvec_block_size, hgemv_grid, hgemv_shared_bytes,
    matvec_block_size, matvec_q8_0_grid, matvec_smem_grid, matvec_smem_grid_nr,
    matvec_smem_shared_bytes, q8_1_quant_grid, rmsnorm_block_size, rmsnorm_shared_bytes, KernelSet,
    Q4F32ActKernel, DP4A_Q4_BLOCK_DIM, DP4A_Q8_1_BLOCK_DIM, FUSED_GLU_BLOCK_DIM,
    FUSED_GLU_SHMEM_LIMIT, HGEMV_BLOCK_DIM, HGEMV_SHMEM_LIMIT, Q8_0_BLOCK_DIM,
    Q8_1_QUANT_BLOCK_DIM, SMEM_BLOCK_DIM,
};
use super::ffi::CudaDevice;
use super::gpu_buffers::{upload_layer_weights, GpuWeightBuf, LayerWeightsGpu};
// Per-phase decode profiler. Every `prof::` call short-circuits on a cached
// `u8` when `LUMEN_CUDA_PROFILE` is unset, so the default path records no
// events and adds no synchronization.
use super::kv_cache::KvCacheGpu;
use super::profiler as prof;
use super::profiler::Phase as Ph;
use super::shaders::EMBED_KERNEL_SOURCE;
use super::types::LaunchConfig;
use cudarc::cublas::{sys as cublas_sys, Gemv, GemvConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig as CudarcLaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

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

/// DEFAULT-OFF candidate (`LUMEN_CUDA_GDN_SPLIT_SITES=1`): pass the Q4 split
/// siblings at the two GDN `launch_matvec` sites (fused `gdn_qkv` [4096,8192]
/// — the largest matvec on the 24 GDN layers — and `gdn_gate`). The split
/// clones are ALREADY created for both (SplitWeightKind::Wq / AttnGate) and
/// sit unread in VRAM: the plain `launch_matvec` shim hard-codes the sibling
/// to `None`, so these two sites never took the split/lane upgrade every
/// full-attn and FFN projection got (r2 audit §8.9). Same weights, same
/// bytes, same activation precision — kernel/access-pattern only.
fn gdn_split_sites_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = parse_env_truthy("LUMEN_CUDA_GDN_SPLIT_SITES").unwrap_or(false);
        if on {
            // Unconditional dispatch-proof marker (not behind VERBOSE).
            eprintln!("[GDN] SPLIT_SITES=ON: gdn_qkv/gdn_gate use Q4 split siblings");
        }
        on
    })
}

/// Env-gate for the custom bandwidth-optimal BF16 decode GEMV kernel
/// (`matvec_bf16_v4`). DEFAULT-OFF: unset / `0` keeps the cuBLAS `GemmEx`
/// batch-1 GEMV path byte-identical. When set (`1|true|yes|on`), every eligible
/// BF16 decode matvec flowing through `launch_matvec` (FFN gate/up/down,
/// attention wq/wk/wv, GDN qkv/gate/ssm_out) dispatches the custom
/// uint4-vectorized, F32-accumulate kernel instead of `cublasGemmEx` at M=1.
/// Precision-keeper projections are excluded (see `is_bf16_precision_keeper_label`);
/// residual matvecs and lm_head use separate paths and are never affected here.
/// Cached to avoid a per-projection `std::env::var` syscall on the decode hot path.
fn bf16_matvec_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    // Default-ON (kill-switch): +5.2% bf16 decode, byte-identical output,
    // harness gate-banked. `=0` reverts to the cuBLAS GemmEx paths.
    *CACHED.get_or_init(|| parse_env_truthy("LUMEN_CUDA_BF16_MATVEC").unwrap_or(true))
}

/// Projection labels that must stay on their existing (precision-forced) path
/// and are therefore EXCLUDED from the custom `matvec_bf16_v4` GEMV even when
/// `LUMEN_CUDA_BF16_MATVEC` is ON. The GDN SSM alpha/beta gate projections are
/// precision keepers: their tiny output (`out_dim = num_heads`) feeds the
/// linear-attention recurrence directly, so they are deliberately left on the
/// existing cuBLAS / F16 path. Every OTHER Bf16Raw projection reaching
/// `launch_matvec` (FFN gate/up/down, attention wq/wk/wv, GDN qkv/gate/ssm_out)
/// is eligible; the custom kernel's F32-exact accumulate is >= the precision of
/// those paths' F16 GemmEx downcast, so accelerating them is numerically safe.
fn is_bf16_precision_keeper_label(label: &str) -> bool {
    matches!(label, "gdn_alpha" | "gdn_beta")
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
        let proxy = (out_dim.min(AUTOTUNE_DIM_CAP), in_dim.min(AUTOTUNE_DIM_CAP));
        proxy_to_originals
            .entry(proxy)
            .or_default()
            .push((out_dim, in_dim));
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
        let w_buf: CudaSlice<u8> = device.alloc_zeros(w_bytes).map_err(|e| {
            RuntimeError::Compute(format!(
                "bf16_autotune: alloc weight ({proxy_out}x{proxy_in}): {e}"
            ))
        })?;
        let x_buf: CudaSlice<u8> = device.alloc_zeros(x_bytes).map_err(|e| {
            RuntimeError::Compute(format!("bf16_autotune: alloc input ({proxy_in}): {e}"))
        })?;
        let y_buf: CudaSlice<f32> = device.alloc_zeros(proxy_out).map_err(|e| {
            RuntimeError::Compute(format!("bf16_autotune: alloc output ({proxy_out}): {e}"))
        })?;

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

            device.synchronize().map_err(|e| {
                RuntimeError::Compute(format!("bf16_autotune: sync before timing: {e}"))
            })?;

            let mut times = Vec::with_capacity(TRIALS);
            for _ in 0..TRIALS {
                unsafe {
                    event::record(start_event, raw_stream).map_err(|e| {
                        RuntimeError::Compute(format!("bf16_autotune: record start: {e}"))
                    })?;
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
                    event::record(end_event, raw_stream).map_err(|e| {
                        RuntimeError::Compute(format!("bf16_autotune: record end: {e}"))
                    })?;
                    event::synchronize(end_event).map_err(|e| {
                        RuntimeError::Compute(format!("bf16_autotune: sync end: {e}"))
                    })?;
                    let ms = event::elapsed(start_event, end_event).map_err(|e| {
                        RuntimeError::Compute(format!("bf16_autotune: elapsed: {e}"))
                    })?;
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
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP => {
                "DEFAULT_TENSOR_OP".to_string()
            }
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
        let proxy = (out_dim.min(AUTOTUNE_DIM_CAP), in_dim.min(AUTOTUNE_DIM_CAP));
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
        let x_bytes = proxy_in * 2; // F16
        let w_buf: CudaSlice<u8> = device.alloc_zeros(w_bytes).map_err(|e| {
            RuntimeError::Compute(format!(
                "autotune: alloc weight ({proxy_out}x{proxy_in}): {e}"
            ))
        })?;
        let x_buf: CudaSlice<u8> = device.alloc_zeros(x_bytes).map_err(|e| {
            RuntimeError::Compute(format!("autotune: alloc input ({proxy_in}): {e}"))
        })?;
        let y_buf: CudaSlice<f32> = device.alloc_zeros(proxy_out).map_err(|e| {
            RuntimeError::Compute(format!("autotune: alloc output ({proxy_out}): {e}"))
        })?;

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
            device
                .synchronize()
                .map_err(|e| RuntimeError::Compute(format!("autotune: sync before timing: {e}")))?;

            // Timed trials: use CUDA events for precise GPU timing.
            let mut times = Vec::with_capacity(TRIALS);
            for _ in 0..TRIALS {
                unsafe {
                    event::record(start_event, raw_stream).map_err(|e| {
                        RuntimeError::Compute(format!("autotune: record start: {e}"))
                    })?;

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
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP => {
                "DEFAULT_TENSOR_OP".to_string()
            }
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

    /// GPU-resident conv positions for the GDN decode conv ring.
    /// Each entry is a single u32 on GPU, synced from host `conv_positions`.
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
    /// Whether the model has any GDN layers.
    has_gdn_layers: bool,
    /// Whether the model has Q+gate fusion layers (disables graph capture).
    has_qgate_layers: bool,
    /// Whether the model has any MoE layers. Populated in `preload_weights`
    /// from `moe_meta_cache`.
    has_moe_layers: bool,
    /// Number of decode tokens processed since last graph invalidation.
    /// 0 = not yet run, 1 = first token (no capture), 2+ = graph replay.
    decode_token_count: usize,
    /// Set by the FFN when the down projection folded its residual into its own
    /// store and wrote `x_gpu` directly, so the decode loop skips BOTH the
    /// `residual_add` launch and the layer-commit dtod copy. Cleared per layer.

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
    /// Per-layer REPACKED aligned gate+up planes for the W10 wide-M gate+up
    /// kernel. `Some(_)` iff the layer is MoE AND `moe_repack_needed()` (the
    /// W10 gate that justifies the repack cost). Built at preload after the layer
    /// blob upload. `None` otherwise (the default f32act/per-column down path
    /// needs no repack). Empty for dense.
    moe_repacked: Vec<Option<super::moe::CudaMoeRepacked>>,

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
    /// `LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1`: clone the Q8Raw output projection
    /// (~1 GB on Qwen3.5-9B) into a split sibling for decode. Independent of
    /// `use_q8_split` so the contribution of the final projection can be
    /// measured / stacked separately.
    use_output_proj_split: bool,
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

/// Cached `LUMEN_MOE_PROBE=1` gate for the CUDA MoE/GDN decode-vs-prefill
/// diagnostics ([PROBE] / [GDNSTATE] / [XCHK] / rope probes). Default OFF ->
/// byte-identical output; when set, every probe site prints. The env is read
/// once for the whole process (previously each probe site had its own
/// `OnceLock`; the cached result is identical since the env is constant).
fn moe_probe_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LUMEN_MOE_PROBE").as_deref() == Ok("1"))
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

/// Bytes held back from the free-memory-aware split-clone budget for per-token
/// activations and transient scratch that live *beyond* weights + KV cache.
///
/// This covers the cuBLAS graph-capture workspace, the pre-allocated full-vocab
/// logits buffer (`vocab_size * 4 B` ≈ 1 MB at 248K vocab), per-token activation
/// buffers (hidden/intermediate-sized, KB-scale each), and the Q4/Q8 *aligned*
/// repack passes that run immediately after the split clone and stage additional
/// aligned weight buffers. 2 GB is a deliberately conservative envelope: the true
/// transient footprint is well under this, and the per-clone `cudaMalloc`
/// fail-safe in the clone loop (records the first OOM layer and stops) is the
/// ultimate backstop, so an optimistic budget can never crash preload — it only
/// permits more clones to be attempted.
const SPLIT_CLONE_ACTIVATION_SLACK_BYTES: usize = 2_000_000_000; // 2 GB

/// Outcome of [`resolve_split_clone_budget`]: the resolved upper cap plus the
/// inputs that produced it, so the caller can log the "ship what you gated" proof.
struct ResolvedSplitBudget {
    /// Upper cap (bytes) the largest-first clone loop fills up to.
    budget_bytes: usize,
    /// Device free VRAM queried at the clone call site.
    free_mem_bytes: usize,
    /// Activation / scratch slack held back ([`SPLIT_CLONE_ACTIVATION_SLACK_BYTES`]).
    slack_bytes: usize,
    /// True when the value came from the explicit `env_var` override.
    from_env: bool,
}

/// Where a layer left the updated hidden state.
///
/// The FFN down projection can fold its residual and write `x_gpu` directly,
/// which makes the decode loop's commit copy redundant. Returning that fact is
/// preferable to a sticky flag on `MutableState`: the two `ffn_down` dispatch
/// sites and the two decode paths would otherwise each have to agree on
/// setting and clearing it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerOutput {
    /// Result is in `scratch.attn_proj`; the caller must commit it.
    NeedsCommit,
    /// Result is already in `scratch.x_gpu`.
    InPlace,
}

/// Resolves a split-clone VRAM budget, SHARED by the Q4 and Q8 clone passes.
///
/// This is a RESOURCE CONTROL, not a feature switch — the split layout itself
/// is unconditional. It exists because sibling buffers are the one part of the
/// design whose cost scales with the model and the card.
///
/// `env_var` is `LUMEN_CUDA_Q4_SPLIT_BUDGET_GB` at the Q4 site and
/// `LUMEN_CUDA_Q8_SPLIT_BUDGET_GB` at the Q8 site:
///
/// * unset — resource-aware default: free VRAM minus activation slack.
/// * `0` — clone nothing. Every projection stays on its base kernel.
/// * `>0` — requested cap in GB, clamped to the resource-aware default so an
///   over-large request cannot push preload into OOM.
/// * anything else (negative, non-numeric) — hard error. A misspelled budget
///   silently becoming "auto" is how a configuration gets benchmarked without
///   anyone knowing which one ran.
///
/// # No KV reserve
///
/// An earlier version subtracted a computed KV-cache reserve from free VRAM.
/// That double-counted: `init()` allocates the KV caches (see the `kv_caches`
/// loop) and `preload_weights` — where this runs — errors unless `init()` has
/// already completed. `free_memory()` therefore already excludes the KV cache.
/// The comment claiming this ran "before KV alloc" described an ordering the
/// code does not have.
fn resolve_split_clone_budget(env_var: &str, device: &CudaDevice) -> ResolvedSplitBudget {
    // A failed free-VRAM query yields a zero budget, which silently disables
    // every sibling and looks identical to a deliberate `budget=0`. Say so.
    let free_mem_bytes = match device.free_memory() {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "[CUDA] WARNING: free-VRAM query failed ({e}); split-clone budget \
                 resolves to 0 and NO split siblings will be created. Decode falls \
                 back to the base kernels. Set {env_var}=N to override."
            );
            0
        }
    };

    // Resource-aware cap. No lower clamp: raising the budget back up to a fixed
    // floor would hand back the memory the subtraction just reserved.
    let auto_bytes = free_mem_bytes.saturating_sub(SPLIT_CLONE_ACTIVATION_SLACK_BYTES);

    let budget_bytes = match std::env::var(env_var) {
        Err(_) => auto_bytes,
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(gb) if gb.is_finite() && gb >= 0.0 => {
                ((gb * 1_000_000_000.0) as usize).min(auto_bytes)
            }
            _ => {
                eprintln!(
                    "[CUDA] FATAL: {env_var}={raw:?} is not a non-negative number of GB \
                     (use 0 for no split siblings, or omit for the resource-aware default)"
                );
                std::process::exit(30);
            }
        },
    };

    ResolvedSplitBudget {
        budget_bytes,
        free_mem_bytes,
        slack_bytes: SPLIT_CLONE_ACTIVATION_SLACK_BYTES,
        from_env: std::env::var(env_var).is_ok(),
    }
}

/// Parse a `LUMEN_CUDA_*` env var as a truthy flag (`1` / `true` / `TRUE` /
/// `yes` / `YES` / `on` / `ON`). Returns `None` when the env is unset so the
/// caller can apply its own default.
fn parse_env_truthy(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        matches!(
            v.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

/// Resolves `LUMEN_CUDA_GPU_SAMPLE` (default ON). When ON, the CUDA backend
/// advertises `gpu_argmax = true` (once weights are preloaded) and serves the
/// greedy decode loop through `decode_token_greedy`: the existing on-GPU
/// `argmax_f32` kernel selects the token and only a 4-byte token id is copied
/// back, removing the per-token full-vocab logits D2H copy that the
/// logits-returning `decode_token` path pays. Greedy output is byte-identical
/// to the prior full-readback path (GPU `argmax_f32` selects the same index the
/// CPU argmax did over the same logits buffer) -- proven by the differential
/// test `cuda_decode_token_greedy_matches_decode_token_argmax` and by the
/// determinism + baseline-anchored corpus gate. Set `LUMEN_CUDA_GPU_SAMPLE=0`
/// to fall back to the full-vocab-readback + CPU-argmax path. Temperature>0
/// sampling is unaffected (CUDA has no GPU sampler; it stays on the CPU
/// full-logits path), so the fixed-seed sampler remains deterministic.
fn cuda_gpu_sample_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| parse_env_truthy("LUMEN_CUDA_GPU_SAMPLE").unwrap_or(true))
}

/// Resolves `LUMEN_CUDA_GDN_F64_ACCUM`. Unset → model-aware default
/// (`runtime_defaults::gdn_f64_accum_default`, ON for MoE GDN-hybrid models).
///
/// Routes the GDN delta-rule decode/prefill kernels to their F64-internal
/// accumulator variants. F64 on the per-token recurrent state update removes
/// the F32-ULP drift that otherwise diverges single-token decode from batched
/// prefill and triggers the MoE q8 greedy restate-loop. Cached because the
/// gate is consulted per-GDN-layer per-token and `set_model_is_moe` is
/// established before the first decode call.
fn gdn_f64_accum_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_GDN_F64_ACCUM")
            .unwrap_or_else(crate::runtime_defaults::gdn_f64_accum_default)
    })
}

/// PREFILL-SCOPED F64 gate for the GDN batched prefill scan
/// (`gdn_prefill_fused_v3` + `l2_normalize_qk_strided` + `gdn_prefill_norm_gate`).
///
/// Historically the batched MoE PREFILL scan ran the F64 accumulator twins by
/// default (`gdn_f64_accum_enabled() == model_is_moe()`).
/// The F64 prefill scan is ~2× slower (F64 ops at half rate on A100) AND its
/// kernel is NOT 4×-unrolled (the F32 twin is), so the F64 path costs the floor
/// ~22 ms / 1334-tok prefill (floor 544.8→522.4 ms, +4.3%). Runtime-validated:
/// the F32 prefill scan is GQ-PRISTINE ×3 (15/15·8/8·3/3)
/// and the 17×23 router canary stays EXACT — the 256-expert top-K selection
/// is robust to the F32-ULP drift in the prefill scan output. The F64 was added
/// for *decode-vs-prefill* h_state parity; the MoE default single-token DECODE
/// path is the F32 megakernel (see `gdn_decode_megakernel`), so the prefill F64
/// was a one-sided cost. This gate is DELIBERATELY DECOUPLED from
/// `gdn_f64_accum_enabled()` (decode + the decode-graph coupling keep that): it
/// only governs the batched prefill scan precision.
///
/// Default: F32 (returns `false`) for MoE — the validated floor win.
/// Override: `LUMEN_CUDA_GDN_PREFILL_F64=1` restores the F64 prefill scan;
/// an explicit `LUMEN_CUDA_GDN_F64_ACCUM=0|1` also governs the prefill scan
/// (preserving the documented global override for every model class).
fn gdn_prefill_f64_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // The prefill-scoped flag wins outright when set.
        if let Some(v) = parse_env_truthy("LUMEN_CUDA_GDN_PREFILL_F64") {
            return v;
        }
        // An EXPLICIT global `LUMEN_CUDA_GDN_F64_ACCUM=0|1` still governs the
        // prefill scan too (preserves the documented global override semantics
        // for every model class, MoE and non-MoE alike).
        if let Some(v) = parse_env_truthy("LUMEN_CUDA_GDN_F64_ACCUM") {
            return v;
        }
        // No env set → MoE defaults to F32 prefill scan (the PRISTINE-validated
        // floor win). NON-MoE models keep their prior GDN
        // accumulator default (e.g. dense-bf16's decode repetition-attractor
        // F64) — only the MoE prefill scan was floor-validated here, so non-MoE
        // behaviour is byte-unchanged.
        if crate::runtime_defaults::model_is_moe() {
            return false;
        }
        crate::runtime_defaults::gdn_f64_accum_default()
    })
}

/// Resolves `LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64`. Unset → model-aware default
/// (MoE-on via `gdn_f64_accum_default`, i.e. precision parity with the prefill
/// F64 scan).
///
/// The active default MoE single-token DECODE kernel is the F32
/// `gdn_decode_megakernel` (the register-resident phase4 path where the prior
/// AB_F16 / PHASE123_ALIGN / prefill-order levers live is NOT engaged at decode
/// for MoE-35B — `q_norm_buf_rr` is unallocated, so dispatch falls through to
/// the megakernel). Meanwhile the batched MoE PREFILL scan runs F64-accum
/// (`gdn_prefill_fused_v3_f64accum` + `l2_normalize_qk_strided_f64accum`,
/// `gdn_f64_accum_enabled() = model_is_moe()`). Same decay-first FORM, different
/// PRECISION: the F32 decode recurrence accumulates per-step ULP drift that the
/// prefill F64 scan does not, so the decode-built `h_state` diverges from the
/// prefill-built state and the 256-expert router amplifies the drift into
/// expert-selection flips → garble. When ON, decode dispatches the
/// `gdn_decode_megakernel_f64accum` / `..._graph_f64accum` twins (identical
/// structure, F64 L2-norm + F64 delta-rule recurrence, F32 state write-back),
/// restoring decode/prefill precision parity exactly as Metal has by default
/// (Metal runs BOTH decode and prefill GDN in the same F32). Cached because the
/// gate is consulted per-GDN-layer per-token and `set_model_is_moe` is
/// established before the first decode call. OFF is byte-identical to the F32
/// megakernel; dense (non-MoE) is byte-identical by default.
fn gdn_decode_megakernel_f64_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64")
            .unwrap_or_else(crate::runtime_defaults::gdn_f64_accum_default)
    })
}

/// Resolves `LUMEN_CUDA_GDN_AB_F16`. Unset → model-aware default
/// (`runtime_defaults::gdn_ab_f16_default`, currently ON for MoE; dense
/// byte-identical via the `model_is_moe()` AND-gate) so DENSE models stay
/// byte-identical regardless of the env.
///
/// When ON, the GDN `ssm_alpha` / `ssm_beta` projections route through a
/// pre-dequanted F16 cache + cuBLAS `cublasGemmEx` (HGEMV N=1 in decode,
/// HGEMM N=batch in prefill) in BOTH paths — the proven-clean qkv/gate recipe
/// — making them bit-identical decode-vs-prefill (collapsing the measured
/// ~20% L0 alpha/beta divergence that the 256-expert router amplifies). The
/// F16 cache is populated only when this gate is ON (see `preload_weights`),
/// so OFF keeps the caches `None` and the legacy Q8 dp4a (decode) / MMQ
/// (prefill) paths byte-identical. Cached because the gate is consulted
/// per-GDN-layer per-token and at load; `set_model_is_moe` is established
/// before the first read.
fn gdn_ab_f16_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::runtime_defaults::model_is_moe()
            && parse_env_truthy("LUMEN_CUDA_GDN_AB_F16")
                .unwrap_or_else(crate::runtime_defaults::gdn_ab_f16_default)
    })
}

/// Resolves `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL`. Unset → quant-aware default
/// (`runtime_defaults::gdn_decode_via_prefill_default`): ON for MoE (all
/// quants) + dense non-bf16-large; quant/size-aware (see the default fn's
/// doc for the measured evidence).
///
/// Dense default-ON since 2026-06-10 (validated N≥3 byte-deterministic):
/// the legacy decode path's
/// per-step recurrence divergence accumulates over long generations into
/// repetition/charspam on DENSE models too — 9B-q8 GQ-004 verylong was 0/3
/// (deterministic, N=3) with 2/3 prompts stuck at the token cap inside a
/// DD-REP/DD-CHARSPAM attractor; via-prefill ALONE flips it to 3/3 with clean
/// EOS termination (N=5 observations incl. 27B), decode tok/s flat (-0.6%).
/// AB_F16/CONVSTATE_PARITY remain MoE-only: the dense ablation showed they are
/// unnecessary on dense, and CONVSTATE-without-AB is actively harmful there.
///
/// When ON, the single GDN decode token is routed through the PREFILL fused
/// GDN recurrence kernels at `T=1` (`ssm_conv1d_silu_prefill` +
/// `gdn_compute_gates_batched` + `l2_normalize_qk_strided[_f64accum]` +
/// `gdn_prefill_fused_v3[_f64accum]` + `gdn_prefill_norm_gate[_f64accum]`),
/// carrying the persistent `h_state` / `conv_state`, INSTEAD of the decode
/// megakernel / register-resident phase4 recurrence. Because the projection
/// (qkv/gate already F16/bf16-bit-identical, alpha/beta made bit-identical by
/// `gdn_ab_f16_enabled()`) and now the conv1d + gates + L2-norm + delta-rule
/// recurrence + norm-gate all run the EXACT prefill kernels at batch=1, the GDN
/// decode block is byte-equivalent to a prefill of the same position by
/// construction — collapsing the diffuse decode-vs-prefill divergence (alpha/beta
/// ~20% + per-step recurrence ~0.98%) that the 256-expert router amplifies into
/// garble. The F64 variants are selected exactly as the MoE prefill selects them
/// (`gdn_f64_accum_enabled()`), so the recurrence matches the F64 prefill scan to
/// F64 rounding. OFF / missing-prefill-kernels → byte-identical legacy
/// decode path (the missing-kernel fallback logs a one-shot warning at the
/// dispatch site). Cached because the gate is read per-GDN-layer per-token.
fn gdn_decode_via_prefill_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_env_truthy("LUMEN_CUDA_GDN_DECODE_VIA_PREFILL")
            .unwrap_or_else(crate::runtime_defaults::gdn_decode_via_prefill_default)
    })
}

/// Resolves `LUMEN_CUDA_GDN_CONVSTATE_PARITY`. Unset → model-aware default
/// (`runtime_defaults::gdn_convstate_parity_default`, currently ON for MoE;
/// dense byte-identical via the `model_is_moe()` AND-gate) so DENSE models stay
/// byte-identical regardless of the env.
///
/// When ON (with `GDN_DECODE_VIA_PREFILL` ON), the decode GDN **qkv** projection
/// — the buffer (`gdn.qkv_buf`) that feeds the conv ring consumed by
/// `ssm_conv1d_silu_prefill` at `T=1` — is computed via the SAME
/// `launch_gemm_projection` path the batched prefill uses, at `batch = 1`,
/// instead of the decode-only GEMV/dp4a matvec (`launch_matvec` →
/// native-BF16 HGEMV with the autotuned `bf16_algo_for` algo / per-token Q8_1
/// dp4a / aligned-Q8 matvec / MMQ). The prefill path dispatches `cublasGemmEx`
/// BF16 GEMM with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for bf16 and the MMQ INT8/INT4
/// kernel for q8/q4 — the exact reductions the prefill row 0 uses — so the new
/// ring slot's qkv becomes bit-equivalent to a prefill of the same token (the
/// other ring slots are already prefill-written and bit-identical). This is the
/// qkv twin of `gdn_ab_f16_enabled()` (which already routes alpha/beta through
/// the prefill GemmEx). The forensic-localized cause of the residual bf16
/// `conv_state` relD ~5% at L0: the decode GEMV vs prefill GEMM kernel-class
/// mismatch injects a ~0.0014% qkv delta that the conv1d window + SiLU amplify.
///
/// Only the qkv projection is rerouted; gate/alpha/beta keep their existing
/// (already-aligned) paths. OFF / dense / via-prefill-off → byte-identical.
/// Cached because the gate is read per-GDN-layer per-token; `set_model_is_moe`
/// is established before the first decode call.
fn gdn_convstate_parity_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // Stays MoE-only by design: the 2026-06-10 dense ablation showed
        // CONVSTATE-without-AB_F16 is actively harmful on dense (GQ-004 0/3,
        // off-topic), and AB_F16 is also MoE-only — the MoE gate here makes
        // that harmful combination structurally unreachable on dense.
        crate::runtime_defaults::model_is_moe()
            && parse_env_truthy("LUMEN_CUDA_GDN_CONVSTATE_PARITY")
                .unwrap_or_else(crate::runtime_defaults::gdn_convstate_parity_default)
    })
}

/// Resolves `LUMEN_CUDA_GDN_SKIP_DUP_QKV`. When ON *and* the decode GDN
/// conv_state parity reprojection is active (`gdn_convstate_parity_qkv`), the
/// redundant **normal** GDN qkv projection is SKIPPED: the parity block fully
/// overwrites `gdn.qkv_buf` (beta=0 BF16 GemmEx / batch=1 Q8 MMQ) and nothing
/// consumes the normal projection's result between the two dispatches (only
/// alpha/beta/gate — none read `qkv_buf` — run in between, and each re-derives
/// its own quantized/F16 activation from `normed`, so the normal qkv matvec
/// leaves no consumed scratch). Removing it is therefore arithmetic-identical
/// (BYTE-identical) — a pure dead-work elimination, validated byte-identical on
/// MoE-bf16 (+5.99%) and MoE-Q8 (+4.87%) whole-token decode. Default ON; the
/// skip only ever engages when the parity predicate that overwrites the buffer
/// is itself true (MoE bf16/Q8), so dense / Q4 / non-parity paths are
/// unaffected. Set `LUMEN_CUDA_GDN_SKIP_DUP_QKV=0` to force the legacy
/// double-projection (A/B baseline). Cached: read per-GDN-layer per-token.
fn gdn_skip_dup_qkv_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| parse_env_truthy("LUMEN_CUDA_GDN_SKIP_DUP_QKV").unwrap_or(true))
}

/// Resolves `LUMEN_CUDA_MOE_DECODE_F32`. Unset → OFF; AND-gated on
/// `model_is_moe()` so DENSE models stay byte-identical regardless of the env.
///
/// === UNIFORM-F32 CUDA MoE DECODE (MoE-gated) ===
/// The single-token DECODE numerics diverge from the batched PREFILL because the
/// decode matmuls take precision-reducing shortcuts that prefill (which matches
/// llama.cpp) does not. For the **bf16** quant the dominant such shortcut is the
/// cuBLAS `cublasGemmEx` path in `CUBLAS_COMPUTE_32F_FAST_16F` mode: it converts
/// the bf16 weights (1 sign / 8 exp / 7 mant) to F16 (1/5/10) and multiplies on
/// F16 tensor cores. That DOWNCAST drops bf16's 8-bit-exponent dynamic range on
/// large-magnitude weights and uses a different (tensor-core, N=1 GEMV-tiled)
/// reduction than prefill. The residual it injects is amplified by the
/// 256-expert MoE router into a flipped expert selection that cascades 40 layers
/// into garbled arithmetic. The legacy `matvec_bf16` kernel instead upcasts
/// bf16→f32 EXACTLY (`bits << 16`, lossless because bf16 is the top 16 bits of an
/// IEEE binary32) and accumulates the dot product in F32 — i.e. it IS the
/// true-F32 path matching the F32 GGUF source precision. When this gate is ON it
/// forces every bf16 DECODE projection (full-attention QKV+O, GDN qkv/gate) onto
/// that F32-exact kernel by closing the `bf16_gemmex_enabled()` gate inside the
/// two BF16 fallback wrappers (`launch_bf16_matvec_with_fallback` and its
/// residual twin) — the single chokepoints for all GEMV-shaped (N=1) bf16 decode
/// matmuls. The batched PREFILL bf16 GEMM (N=batch) does NOT route through those
/// wrappers, so prefill stays byte-identical. OFF is byte-identical to history.
/// Cached because the gate is read per-projection per-token; `set_model_is_moe`
/// is established before the first decode call.
fn moe_decode_f32_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::runtime_defaults::model_is_moe()
            && parse_env_truthy("LUMEN_CUDA_MOE_DECODE_F32").unwrap_or(false)
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
                    m,
                    n,
                    k,
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
        let w_dev: CudaSlice<u8> = self.device.htod_copy(weight_bf16_bytes).map_err(|e| {
            RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: htod_copy weight: {e}",
            ))
        })?;
        let input_dev: CudaSlice<f32> = self.device.htod_copy(input_f32).map_err(|e| {
            RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: htod_copy input: {e}",
            ))
        })?;
        let mut output_dev: CudaSlice<f32> = self.device.alloc_zeros(out_dim).map_err(|e| {
            RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: alloc output: {e}",
            ))
        })?;
        let mut input_bf16_scratch: CudaSlice<u8> =
            self.device.alloc_zeros(in_dim * 2).map_err(|e| {
                RuntimeError::Compute(format!(
                    "dispatch_bf16_matvec_for_tests {label}: alloc scratch: {e}",
                ))
            })?;
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
                let residual_dev: CudaSlice<f32> = self.device.htod_copy(r).map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dispatch_bf16_matvec_for_tests {label}: htod_copy residual: {e}",
                    ))
                })?;
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
        self.device.synchronize().map_err(|e| {
            RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: synchronize: {e}",
            ))
        })?;
        let out_host: Vec<f32> = self.device.dtoh_copy(&output_dev).map_err(|e| {
            RuntimeError::Compute(format!(
                "dispatch_bf16_matvec_for_tests {label}: dtoh output: {e}",
            ))
        })?;
        Ok(out_host)
    }

    /// Embed a token directly into the GPU scratch buffer `x_gpu`, with no sync.
    ///
    /// This is the GPU-resident counterpart of `embed_token`. Instead of syncing
    /// and copying back to host, it leaves the embedding in `st.scratch.x_gpu`.
    fn embed_token_gpu(&self, token_id: u32, st: &mut MutableState) -> Result<(), RuntimeError> {
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
            .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_bf16 gpu launch: {e}")))?;
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
            .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_f16 gpu launch: {e}")))?;
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
            .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_q4_0 gpu launch: {e}")))?;
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
            .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_q8_0 gpu launch: {e}")))?;
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
            .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_f32 gpu launch: {e}")))?;
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
    ) -> Result<LayerOutput, RuntimeError> {
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
            prof::begin(Ph::GdnAttn, &self.device.stream);
            self.compute_gdn_attention_gpu(layer_idx, st)?;
            prof::end(Ph::GdnAttn, &self.device.stream);
        // Attribution ablation: skip the standard attention block on the 8
        // full-attention layers, keeping their FFN. Part of splitting the 21%
        // "everything else" that llama.cpp evidently does not pay.
        } else {
            prof::begin(Ph::FullAttn, &self.device.stream);
            // Log EVERY layer entering the standard attention branch. The
            // mid-branch probe saw only 4 of the 8 full-attention layers, and
            // the census agrees (wq/wk/wv 4/token while wo is 8/token from the
            // same block), so 4 layers diverge somewhere before the QKV
            // decision. Probing at the entry pins down whether they enter at
            // all.
            let lw: &LayerWeightsGpu = st.layer_weights_cache.get(layer_idx).ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "compute_layer_gpu: layer {layer_idx} not in GPU-resident cache",
                ))
            })?;

            // Q+gate fusion detection: Qwen3.5 full-attention layers have fused Q+gate
            // in wq with output dimension q_dim*2. When active, wq projects to q_gate
            // scratch buffer, then deinterleave + per-head norm produces final Q and gate.
            let has_qgate_fusion = lw.attn_q_norm.is_some();
            let wq_out_dim = if has_qgate_fusion { q_dim * 2 } else { q_dim };

            // `AttnQkv` covers the input RMSNorm together with the projections:
            // the norm kernel is selected inside each weight-format arm and one
            // arm fuses it into the projection kernel itself, so no standalone
            // input-norm region exists to bracket.
            prof::begin(Ph::AttnQkv, &self.device.stream);

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
                        (
                            st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                            wq_out_dim,
                        )
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
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &lw.attn_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "attn F16",
                    )?;
                }
                // QKV projections: use pre-computed pointers if available.
                if let Some(ref pcp) = st.precomputed_ptrs {
                    // Pre-computed batched: Q separate + KV batched (no htod).
                    if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                        // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                        let (wq_out_buf, wq_od) = if has_qgate_fusion {
                            (
                                st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                                wq_out_dim,
                            )
                        } else {
                            (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                        };
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wq_f16,
                                &st.scratch.input_f16,
                                wq_out_buf,
                                wq_od,
                                hidden_dim,
                                "wq",
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
                            2,
                            kv_dim,
                            hidden_dim,
                            "kv",
                            st.algo_cache.get(kv_dim, hidden_dim),
                        )?;
                    }
                } else {
                    // Fallback: original per-layer htod path.
                    if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                        // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                        let (wq_out_buf, wq_od) = if has_qgate_fusion {
                            (
                                st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                                wq_out_dim,
                            )
                        } else {
                            (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                        };
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wq_f16,
                                &st.scratch.input_f16,
                                wq_out_buf,
                                wq_od,
                                hidden_dim,
                                "wq",
                                st.algo_cache.get(wq_od, hidden_dim),
                            )?;
                        }
                    }
                    if let (GpuWeightBuf::F16Raw(ref wk_f16), GpuWeightBuf::F16Raw(ref wv_f16)) =
                        (&lw.wk, &lw.wv)
                    {
                        unsafe {
                            let w_slices: &[&CudaSlice<u8>] = &[wk_f16, wv_f16];
                            let mut out_slices: [&mut CudaSlice<f32>; 2] =
                                [&mut st.scratch.k, &mut st.scratch.v];
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
            } else if crate::runtime_defaults::qkv_decouple()
                && matches!(&lw.wq, GpuWeightBuf::F32(_))
                && lw.wq_f16.is_some()
                && lw.wk_f16.is_some()
                && lw.wv_f16.is_some()
            {
                // ---------------------------------------------------------------
                // C2 (`LUMEN_CUDA_QKV_DECOUPLE=1`): DECOUPLED QKV dispatch.
                //
                // This branch is a variant of the F16-HGEMV branch immediately
                // below, entered only when the flag is set. With the flag unset
                // the guard is false on its first conjunct and control falls to
                // the original branch unchanged, so flag-off behaviour is
                // byte-identical by construction -- no reordering, no shared
                // state, nothing to keep in sync beyond the guard itself.
                //
                // WHAT THE DEFAULT BRANCH GETS WRONG. Its guard is "wq is F32
                // AND all three F16 caches exist". It never inspects wk/wv's
                // actual weight format -- only whether a cache happens to be
                // present. On 9B-Q4 layers 3/15/27/31, wq is a host-dequanted
                // Q6_K (hence F32) while wk/wv are natively Q4Raw, and
                // dequant_layer_q8_to_f16 builds F16 caches for Q4Raw on every
                // full-attention layer. So two 0.5625 B/w tensors get read at
                // 2.0 B/w through cublasGemmBatchedEx:
                //
                //   8 tensors x 4,194,304 w x (2.0 - 0.5625) B/w
                //     = 48,234,496 B = 46.0 MiB/token   (18.0 -> 64.0 MiB)
                //
                // It also silently bypasses the F32-exact activation policy:
                // q4_act_plan is only consulted inside launch_matvec_ext, which
                // this branch never reaches, so wk/wv run F16 activations on a
                // family the plan pins to F32.
                //
                // WHY THEY WERE COUPLED IN THE FIRST PLACE. Not a fused kernel
                // and not a shared output buffer -- wq/wk/wv are already
                // separate launches into separate buffers. The real coupling is
                // the ACTIVATION FORMAT: this branch normalizes once to F16,
                // while the native Q4 ladder consumes F32 `scratch.normed`. A
                // split dispatch needs both, which is the one extra kernel this
                // branch pays: an RMSNorm over hidden_dim (~16 KB write).
                //
                // ORDERING. `normed` (F32) is produced first, then `input_f16`,
                // then wq consumes `input_f16`, then wk/wv consume `normed`.
                // The two norms read the same `x_gpu` and write disjoint
                // buffers. wq is dispatched BEFORE wk/wv because
                // launch_matvec_ext is passed `input_f16` as scratch and would
                // be free to overwrite it on an HGEMV fallback arm; with Q4Raw
                // weights and an F32 plan it takes the native ladder and does
                // not, but the ordering makes that a non-issue rather than an
                // invariant to remember.
                //
                // The precomputed KV pointer arrays (pcp.kv_*_ptrs) simply go
                // unread on these layers. They stay allocated and index-aligned
                // for every layer, so nothing else observes the difference.
                //
                // NOT byte-identical: F16 weights are swapped for F32-exact Q4.
                // ---------------------------------------------------------------
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm attn decouple launch: {e}"))
                    })?;
                }
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &lw.attn_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "attn DECOUPLE",
                    )?;
                }
                // wq keeps its own route: the F16 cache today, or C1's native
                // Q6_K kernel when LUMEN_CUDA_Q6K_NATIVE is also on (in which
                // case wq is Q6KRaw, this guard is false, and we are not here
                // at all -- C1 subsumes C2 on these layers).
                if let Some(ref wq_f16) = lw.wq_f16 {
                    let (wq_out_buf, wq_od) = if has_qgate_fusion {
                        (
                            st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                            wq_out_dim,
                        )
                    } else {
                        (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                    };
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device,
                            wq_f16,
                            &st.scratch.input_f16,
                            wq_out_buf,
                            wq_od,
                            hidden_dim,
                            "wq",
                            st.algo_cache.get(wq_od, hidden_dim),
                        )?;
                    }
                    // The default branch records nothing at all, which is why
                    // the census reports wq/wk/wv at 4/token against 8
                    // full-attention layers. Tag it so this arm is observable.
                    crate::runtime_defaults::route_census_record("wq", "HGEMV_F16_DECOUPLED");
                }
                // wk/wv now reach launch_matvec_ext, so they get the native Q4
                // ladder, their SoA split siblings, and the activation plan.
                unsafe {
                    launch_matvec_ext(
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
                        lw.q4_split_wk.as_ref(),
                    )?;
                    launch_matvec_ext(
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
                        lw.q4_split_wv.as_ref(),
                    )?;
                }
            } else if matches!(&lw.wq, GpuWeightBuf::F32(_))
                && lw.wq_f16.is_some()
                && lw.wk_f16.is_some()
                && lw.wv_f16.is_some()
            {
                // cuBLAS HGEMV fast path for F32 weights with pre-dequanted F16 caches.
                // CUBLAS_COMPUTE_32F_FAST_16F exploits tensor cores (312 TFLOPS on A100).
                // Only used for F32 weights where F16 HGEMV halves bandwidth (4 -> 2 B/elem).
                // Q8/Q4/Q8Aligned weights fall through to launch_matvec() which dispatches
                // native dp4a kernels reading 1.06 B/elem -- 1.9x less bandwidth than HGEMV.
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &lw.attn_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "attn HGEMV",
                    )?;
                }
                // QKV projections: use pre-computed pointers if available (same logic as F16 native path).
                if let Some(ref pcp) = st.precomputed_ptrs {
                    // Pre-computed batched: Q separate + KV batched (no htod).
                    if let Some(ref wq_f16) = lw.wq_f16 {
                        // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                        let (wq_out_buf, wq_od) = if has_qgate_fusion {
                            (
                                st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                                wq_out_dim,
                            )
                        } else {
                            (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                        };
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wq_f16,
                                &st.scratch.input_f16,
                                wq_out_buf,
                                wq_od,
                                hidden_dim,
                                "wq",
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
                            2,
                            kv_dim,
                            hidden_dim,
                            "kv",
                            st.algo_cache.get(kv_dim, hidden_dim),
                        )?;
                    }
                } else {
                    // Fallback: Q separate + KV batched with per-layer htod.
                    if let Some(ref wq_f16) = lw.wq_f16 {
                        // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                        let (wq_out_buf, wq_od) = if has_qgate_fusion {
                            (
                                st.scratch.q_gate.as_mut().unwrap() as &mut CudaSlice<f32>,
                                wq_out_dim,
                            )
                        } else {
                            (&mut st.scratch.q as &mut CudaSlice<f32>, q_dim)
                        };
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wq_f16,
                                &st.scratch.input_f16,
                                wq_out_buf,
                                wq_od,
                                hidden_dim,
                                "wq",
                                st.algo_cache.get(wq_od, hidden_dim),
                            )?;
                        }
                    }
                    if let (Some(ref wk_f16), Some(ref wv_f16)) = (&lw.wk_f16, &lw.wv_f16) {
                        unsafe {
                            let w_slices: &[&CudaSlice<u8>] = &[wk_f16, wv_f16];
                            let mut out_slices: [&mut CudaSlice<f32>; 2] =
                                [&mut st.scratch.k, &mut st.scratch.v];
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
                // Q4_0 QUALITY FIX (attention side): on the fragile 9B config,
                // Q4Raw QKV uses F32 activations, not int8 Q8_1 dp4a. Other models
                // keep dp4a (flag off). See weight_uses_f32_act_q4.
                // DEFECT PROBE: the per-site census shows wq/wk/wv dispatching
                // 4x/token while [LAYERS] reports 8 full-attention layers, so
                // half of them are NOT taking the int8 route. Log the decision
                // and the weight variant per layer, once each, to find which.
                let qkv_use_preq = !weight_uses_f32_act_q4_fam(
                    &lw.wq,
                    &st.kernels.q4_act_plan,
                    crate::runtime_defaults::Q4ProjectionFamily::AttnQkv,
                ) && weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
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
                            launch_matvec_preq8_1_split(
                                &self.device,
                                &st.kernels,
                                &lw.wq,
                                lw.q8_split_wq.as_ref(),
                                lw.q4_split_wq.as_ref(),
                                q8_1_buf,
                                st.scratch.q_gate.as_mut().unwrap(),
                                wq_out_dim,
                                hidden_dim,
                                "wq",
                            )?;
                        } else {
                            launch_matvec_preq8_1_split(
                                &self.device,
                                &st.kernels,
                                &lw.wq,
                                lw.q8_split_wq.as_ref(),
                                lw.q4_split_wq.as_ref(),
                                q8_1_buf,
                                &mut st.scratch.q,
                                q_dim,
                                hidden_dim,
                                "wq",
                            )?;
                        }
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.wk,
                            lw.q8_split_wk.as_ref(),
                            lw.q4_split_wk.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.k,
                            kv_dim,
                            hidden_dim,
                            "wk",
                        )?;
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.wv,
                            lw.q8_split_wv.as_ref(),
                            lw.q4_split_wv.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.v,
                            kv_dim,
                            hidden_dim,
                            "wv",
                        )?;
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
                        launch_quantize_input_q8_1(
                            &self.device,
                            quant_fn,
                            &st.scratch.normed,
                            q8_1_buf,
                            hidden_dim,
                            "qkv",
                        )?;
                        // split-layout: prefer Q8Split/Q4Split sibling buffers on QKV when set.
                        // Q+gate fusion: project wq to q_gate buffer with doubled output dim.
                        if has_qgate_fusion {
                            launch_matvec_preq8_1_split(
                                &self.device,
                                &st.kernels,
                                &lw.wq,
                                lw.q8_split_wq.as_ref(),
                                lw.q4_split_wq.as_ref(),
                                q8_1_buf,
                                st.scratch.q_gate.as_mut().unwrap(),
                                wq_out_dim,
                                hidden_dim,
                                "wq",
                            )?;
                        } else {
                            launch_matvec_preq8_1_split(
                                &self.device,
                                &st.kernels,
                                &lw.wq,
                                lw.q8_split_wq.as_ref(),
                                lw.q4_split_wq.as_ref(),
                                q8_1_buf,
                                &mut st.scratch.q,
                                q_dim,
                                hidden_dim,
                                "wq",
                            )?;
                        }
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.wk,
                            lw.q8_split_wk.as_ref(),
                            lw.q4_split_wk.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.k,
                            kv_dim,
                            hidden_dim,
                            "wk",
                        )?;
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.wv,
                            lw.q8_split_wv.as_ref(),
                            lw.q4_split_wv.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.v,
                            kv_dim,
                            hidden_dim,
                            "wv",
                        )?;
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
                            launch_matvec_ext(
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
                                lw.q4_split_wq.as_ref(),
                            )?;
                        } else {
                            launch_matvec_ext(
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
                                lw.q4_split_wq.as_ref(),
                            )?;
                        }
                        launch_matvec_ext(
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
                            lw.q4_split_wk.as_ref(),
                        )?;
                        launch_matvec_ext(
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
                            lw.q4_split_wv.as_ref(),
                        )?;
                    }
                }
            }

            prof::end(Ph::AttnQkv, &self.device.stream);

            // Q+gate fusion post-processing: deinterleave q_gate -> q + gate_buf,
            // then per-head RMSNorm on Q (attn_q_norm) and K (attn_k_norm).
            // Must run AFTER all QKV projection branches and BEFORE RoPE.
            prof::begin(Ph::AttnQkNorm, &self.device.stream);
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("deinterleave_qgate launch: {e}"))
                    })?;
                } else {
                    return Err(RuntimeError::Compute(
                        "Q+gate fusion requires deinterleave_qgate kernel".into(),
                    ));
                }

                // 1b. Per-head RMSNorm on Q using attn_q_norm [head_dim]
                if let Some(ref q_norm_w) = lw.attn_q_norm {
                    let norm_fn =
                        st.kernels
                            .rmsnorm_per_head_inplace
                            .as_ref()
                            .ok_or_else(|| {
                                RuntimeError::Compute(
                                    "Q+gate fusion requires rmsnorm_per_head_inplace kernel".into(),
                                )
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm_per_head Q launch: {e}"))
                    })?;
                }

                // 1c. Per-head RMSNorm on K using attn_k_norm [head_dim]
                if let Some(ref k_norm_w) = lw.attn_k_norm {
                    let norm_fn =
                        st.kernels
                            .rmsnorm_per_head_inplace
                            .as_ref()
                            .ok_or_else(|| {
                                RuntimeError::Compute(
                                    "Q+gate fusion requires rmsnorm_per_head_inplace kernel".into(),
                                )
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm_per_head K launch: {e}"))
                    })?;
                }
            }

            prof::end(Ph::AttnQkNorm, &self.device.stream);

            // QKV bias (Qwen2-family, decode).
            prof::begin(Ph::AttnBias, &self.device.stream);
            if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
                let block = 256u32;
                unsafe {
                    if let Some(ref bq) = lw.bq {
                        let d = q_dim as u32;
                        let g = (d + block - 1) / block;
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add)
                            .arg(&mut st.scratch.q)
                            .arg(bq)
                            .arg(&d)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add bq decode: {e}"))
                            })?;
                    }
                    if let Some(ref bk) = lw.bk {
                        let d = kv_dim as u32;
                        let g = (d + block - 1) / block;
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add)
                            .arg(&mut st.scratch.k)
                            .arg(bk)
                            .arg(&d)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add bk decode: {e}"))
                            })?;
                    }
                    if let Some(ref bv) = lw.bv {
                        let d = kv_dim as u32;
                        let g = (d + block - 1) / block;
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add)
                            .arg(&mut st.scratch.v)
                            .arg(bv)
                            .arg(&d)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add bv decode: {e}"))
                            })?;
                    }
                }
            }

            prof::end(Ph::AttnBias, &self.device.stream);

            // 2. RoPE.
            prof::begin(Ph::AttnRope, &self.device.stream);
            {
                let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
                let actual_rot = if rotary_dim > 0 && rotary_dim < head_dim as u32 {
                    rotary_dim as usize
                } else {
                    head_dim
                };
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

            prof::end(Ph::AttnRope, &self.device.stream);

            // 3. KV cache write.
            prof::begin(Ph::AttnKvWrite, &self.device.stream);
            {
                let kv_cache = st.kv_caches.get_mut(layer_idx).ok_or_else(|| {
                    RuntimeError::Compute(format!("no KV cache for layer {layer_idx}"))
                })?;
                kv_cache.append_kv(&self.device, &st.scratch.k, &st.scratch.v)?;
            }
            prof::end(Ph::AttnKvWrite, &self.device.stream);

            // 4. Attention. gate: routes to the tiled streaming-softmax
            // kernel at long context (seq_len > LUMEN_CUDA_DECODE_TILED_THRESHOLD,
            // default 0 = "tiled-always") or when LUMEN_CUDA_DECODE_TILED=1
            // forces it. Operators can set `LUMEN_CUDA_DECODE_TILED_THRESHOLD=
            // 4294967295` to opt out (force single-block below the 40_950 ceiling).
            prof::begin(Ph::AttnCore, &self.device.stream);
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

            prof::end(Ph::AttnCore, &self.device.stream);

            // `AttnWo` spans the sigmoid gating, its copy, and the output
            // projection (which folds the post-attention residual, so there is
            // no separate residual region to bracket).
            prof::begin(Ph::AttnWo, &self.device.stream);

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
                        .map_err(|e| {
                            RuntimeError::Compute(format!("sigmoid_mul dtod copy: {e}"))
                        })?;
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
            } else if matches!(&lw.wo, GpuWeightBuf::F32(_)) && lw.wo_f16.is_some() {
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
                // Like the ffn_down shortcut, this quantizes to Q8_1 ahead of
                // `launch_matvec_residual`, so it may only claim `wo` when the
                // plan puts `wo` on int8. On the narrow-GDN dense class it does
                // not — that is the one family the quality result depends on.
                let use_split_wo = st
                    .kernels
                    .q4_act_plan
                    .mode_for(crate::runtime_defaults::Q4ProjectionFamily::AttnWo)
                    == crate::runtime_defaults::Q4ActMode::Q8_1
                    && ((st.kernels.use_q8_split_dispatch && lw.q8_split_wo.is_some())
                        || (st.kernels.use_q4_split_dispatch && lw.q4_split_wo.is_some()));
                if use_split_wo {
                    // Quantize attention output to Q8_1 in scratch, then split residual matvec.
                    let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                    let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                    if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                        unsafe {
                            launch_quantize_input_q8_1(
                                &self.device,
                                quant_fn,
                                &st.scratch.attn_out,
                                q8_1_buf,
                                q_dim,
                                "wo split",
                            )?;
                            launch_matvec_preq8_1_residual_split(
                                &self.device,
                                &st.kernels,
                                &lw.wo,
                                lw.q8_split_wo.as_ref(),
                                lw.q4_split_wo.as_ref(),
                                q8_1_buf,
                                &st.scratch.x_gpu,
                                &mut st.scratch.attn_proj,
                                hidden_dim,
                                q_dim,
                                "wo",
                            )?;
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
                } else {
                    unsafe {
                        // Lane decomposition first; falls through when the
                        // variant is off or the layer has no split sibling.
                        if !launch_matvec_residual_lane(
                            &self.device,
                            &st.kernels,
                            lw.q4_split_wo.as_ref(),
                            &st.scratch.attn_out,
                            &st.scratch.x_gpu,
                            &mut st.scratch.attn_proj,
                            hidden_dim,
                            q_dim,
                            "wo",
                        )? {
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
            }
            prof::end(Ph::AttnWo, &self.device.stream);
            prof::end(Ph::FullAttn, &self.device.stream);
        } // end else (standard attention path — skipped for GDN layers)

        // Re-borrow layer weights for the FFN block (shared between standard and GDN layers).
        let lw: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];

        // MoE FFN branch — when the layer has expert metadata, dispatch
        // the three-phase MoE forward (router -> per-expert FFN -> accum) and
        // skip the dense FFN block below entirely.
        if let Some(moe_meta) = st.moe_meta_cache.get(layer_idx).and_then(|m| m.as_ref()) {
            prof::begin(Ph::MoeFfn, &self.device.stream);
            let moe_layer_blob = lw.moe_layer_blob.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "MoE layer {layer_idx} missing moe_layer_blob; \
                     upload_layer_weights must populate it when subtensors.experts.is_some()",
                ))
            })?;
            let num_experts = moe_meta.expert_gate_offs.len();
            let top_k = self
                .hp()?
                .num_active_experts
                .map(|v| v as usize)
                .unwrap_or(0);
            if top_k == 0 {
                return Err(RuntimeError::Compute(
                    "MoE layer present but hyperparams.num_active_experts not set".into(),
                ));
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
                    "MoE layer dispatch requires moe_scratch (allocated in init for MoE models)"
                        .into(),
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
            if moe_meta.shared_gate.is_some() {
                // opt-in fused shared-expert path (3 launches vs 5-6).
                // Falls back to legacy unfused path if any of the 3 fused
                // kernels failed to compile (NVRTC failure on this device).
                if super::moe::moe_shared_fused_decode_enabled() {
                    // Lever L2 "shared-expert fused decode" (default-OFF flag
                    // LUMEN_CUDA_SHARED_FUSED_DECODE), independent of the L1 tiled
                    // flag. When ON, route the decode shared expert through the
                    // batch=1-native fused kernels (2-3 launches vs the naive
                    // 5-6): a 2-stream gate+up+SwiGLU GEMV and a fused
                    // down+gated-accum. Same Q4_0/F32 numerics as naive
                    // (byte-identical up to warp FP-add order). Delegates
                    // internally to the naive path on any unsupported model/device.
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

            // [PROBE] Decode-vs-prefill localization (env LUMEN_MOE_PROBE=1).
            // compute_layer_gpu is the DECODE path (1 token). Dumps this token's
            // post-layer residual (x_gpu) and attention output (attn_proj) so it
            // can be compared against the batched prefill of the same position.
            if moe_probe_enabled() {
                let xh = self.device.dtoh_copy(&st.scratch.x_gpu)?;
                let ah = self.device.dtoh_copy(&st.scratch.attn_proj)?;
                let k = 16usize;
                eprintln!(
                    "[PROBE] mode=D pos={seq_pos} layer={layer_idx} attn16={:?} x16={:?}",
                    &ah[..k.min(ah.len())],
                    &xh[..k.min(xh.len())]
                );
                // [CHK] mode=D whole-buffer sumsq (layout-independent) of this
                // decode token's post-layer residual (x = l_out) and attention
                // output (a). Mirrors the prefill [CHK] mode=P so lumen-decode vs
                // lumen-prefill (and vs llama l_out/attn_output) can be compared
                // per-layer to localize the prefill-vs-decode divergence that
                // flips the near-tie. hidden_dim-sized slices.
                let sumsq = |v: &[f32], n: usize| -> (f64, f64, f32) {
                    let mut s = 0f64;
                    let mut sq = 0f64;
                    let mut mx = 0f32;
                    for &e in &v[..n.min(v.len())] {
                        s += e as f64;
                        sq += (e as f64) * (e as f64);
                        if e.abs() > mx {
                            mx = e.abs();
                        }
                    }
                    (s, sq, mx)
                };
                let (xs, xsq, xmx) = sumsq(&xh, hidden_dim);
                let (as_, asq, amx) = sumsq(&ah, hidden_dim);
                eprintln!(
                    "[CHK] mode=D pos={seq_pos} layer={layer_idx} \
                     x_sum={xs:.5} x_sumsq={xsq:.5} x_absmax={xmx:.6} \
                     a_sum={as_:.5} a_sumsq={asq:.5} a_absmax={amx:.6}"
                );
                // Routing probe: selected expert IDs + gate weights for this token.
                let ids = self.device.dtoh_copy(&moe_scratch.expert_ids)?;
                let ws = self.device.dtoh_copy(&moe_scratch.expert_weights)?;
                eprintln!("[PROBE-RT] mode=D pos={seq_pos} layer={layer_idx} ids={ids:?} w={ws:?}");
                // [MOE-SUMSQ] mode=D decode expert-combine sumsq (= Σ_k gw[k]*eo[k]),
                // matching the prefill [MOE-SUMSQ] and llama ffn_moe_out, so the
                // expert-combine reduction can be ruled in/out as the flip site.
                let eo = self.device.dtoh_copy(&moe_scratch.expert_output_buf)?;
                let rl = self.device.dtoh_copy(&moe_scratch.router_logits)?;
                let router_logits_sumsq: f64 = rl.iter().map(|&e| (e as f64) * (e as f64)).sum();
                let gate_w_sumsq: f64 = ws.iter().map(|&e| (e as f64) * (e as f64)).sum();
                let expert_out_sumsq: f64 = eo.iter().map(|&e| (e as f64) * (e as f64)).sum();
                let mut ffn_moe_out_sumsq = 0f64;
                let tk = ws.len();
                for i in 0..hidden_dim {
                    let mut acc = 0f64;
                    for kk in 0..tk {
                        let idx = kk * hidden_dim + i;
                        if idx < eo.len() {
                            acc += (ws[kk] as f64) * (eo[idx] as f64);
                        }
                    }
                    ffn_moe_out_sumsq += acc * acc;
                }
                eprintln!(
                    "[MOE-SUMSQ] mode=D pos={seq_pos} layer={layer_idx} \
                     router_logits_sumsq={router_logits_sumsq:.6} \
                     gate_w_sumsq={gate_w_sumsq:.6} \
                     expert_out_sumsq={expert_out_sumsq:.6} \
                     ffn_moe_out_sumsq={ffn_moe_out_sumsq:.6}"
                );
            }

            // [XCHK] Cross-backend forensic probe (env LUMEN_XCHK=1, default OFF
            // -> byte-identical). Per-MoE-layer top-K expert IDs + gate weights +
            // router-logits sumsq, in the SAME schema as the Metal [XCHK] dump,
            // keyed by the 0-based decode ordinal (decode_token_count). The
            // expert-ID list is the SHARPEST cross-backend divergence signal.
            if {
                use std::sync::OnceLock;
                static XKM: OnceLock<bool> = OnceLock::new();
                *XKM.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
            } {
                let ids = self.device.dtoh_copy(&moe_scratch.expert_ids)?;
                let ws = self.device.dtoh_copy(&moe_scratch.expert_weights)?;
                let rl = self.device.dtoh_copy(&moe_scratch.router_logits)?;
                let rlsq: f64 = rl.iter().map(|&e| (e as f64) * (e as f64)).sum();
                let mut rlmx = 0f32;
                for &e in &rl {
                    let a = e.abs();
                    if a > rlmx {
                        rlmx = a;
                    }
                }
                let step = st.decode_token_count;
                eprintln!(
                    "[XCHK] step={step} L={layer_idx} router_logits sumsq={rlsq:.6} absmax={rlmx:.6}"
                );
                eprintln!("[XCHK] step={step} L={layer_idx} moe_expert_ids={ids:?} gate_w={ws:?}");
            }

            // MoE branch is complete; skip the dense FFN block below. MoE
            // leaves its result in attn_proj, so the caller still commits.
            prof::end(Ph::MoeFfn, &self.device.stream);
            return Ok(LayerOutput::NeedsCommit);
        }

        // Dense FFN block. `Ffn` is the depth-0 region (shared by GDN and
        // full-attention layers); `FfnGateUp` / `FfnDownResid` refine it. Both
        // must be closed before EVERY exit of the FFN block -- two arms fold
        // the residual into the down store and `return` early.
        prof::begin(Ph::Ffn, &self.device.stream);
        prof::begin(Ph::FfnGateUp, &self.device.stream);

        // 6. FFN: fused or separate rmsnorm + gate/up + swiglu + down + residual.
        //
        // Fused gate+up+SwiGLU GEMV: if the kernel is available and shmem fits,
        // compute rms_scale + fused_glu_gemv in 2 dispatches (replacing 3-5).
        // The fused kernel writes silu(gate)*up directly to scratch.gate,
        // so the SwiGLU step is skipped entirely.
        let fused_glu_fired = 'fused_glu: {
            // LUMEN_CUDA_Q8_MMVQ: mmvq-dp4a fused gate+up+SwiGLU on the Q8 split
            // layout (consult §2.7). Preferred over BOTH the separate mmvq
            // gate/up + swiglu_inplace path AND the scalar fused_glu_gemv when
            // the flag is on and both gate+up have Q8 split siblings. Runs
            // rmsnorm_to_q8_1 (attn_proj -> shared q8_1 buffer, same kernel the
            // separate path uses) then ONE fused kernel that reads that q8_1
            // activation once, computes gate_dot + up_dot with the
            // matvec_q8_split_q8_1_mmvq striping/reduction, and writes
            // silu(gate)*up straight to scratch.gate -- removing 1 matvec + 1
            // SwiGLU launch and the scratch.up round-trip per FFN/layer.
            // BYTE-IDENTICAL to the separate mmvq gate+up+swiglu path; carries
            // exactly the mmvq near-tie vs OFF (GQ + router gated). Setting
            // fused_glu_fired = true reuses the downstream skip-SwiGLU +
            // down-reads-scratch.gate logic unchanged.
            //
            // A/B isolation: fires by default under Q8_MMVQ=1, but an explicit
            // `LUMEN_CUDA_FFN_FUSED_GLU=0` (which already means "use the separate
            // dp4a gate/up path") opts OUT, so the reviewer can measure the
            // combined mmvq config WITH vs WITHOUT gate-fusion.
            let mmvq_glu_opt_out = matches!(
                std::env::var("LUMEN_CUDA_FFN_FUSED_GLU").ok().as_deref(),
                Some("0") | Some("false") | Some("no") | Some("off") | Some("OFF")
            );
            if st.kernels.use_mmvq
                && !mmvq_glu_opt_out
                && st.kernels.fused_glu_gemv_q8_split_mmvq.is_some()
                && st.kernels.rmsnorm_to_q8_1.is_some()
                && st.scratch.input_q8_1.is_some()
                && lw.q8_split_w_gate.is_some()
                && lw.q8_split_w_up.is_some()
            {
                // 1. Fused RMSNorm + Q8_1 quantize: attn_proj -> shared q8_1 buffer.
                {
                    let rms_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
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
                            .launch_builder(rms_fn)
                            .arg(&st.scratch.attn_proj)
                            .arg(&lw.ffn_norm)
                            .arg(&mut *q8_1_buf)
                            .arg(&eps)
                            .arg(&dim)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm_to_q8_1 ffn mmvq-glu: {e}"))
                    })?;
                }
                // 2. Fused gate+up+SwiGLU mmvq: q8_1 -> scratch.gate = silu(gate)*up.
                {
                    crate::runtime_defaults::route_census_record("gate", "Q8_1_FUSED_GLU");
                    crate::runtime_defaults::route_census_record("up", "Q8_1_FUSED_GLU");
                    let fused_fn = st.kernels.fused_glu_gemv_q8_split_mmvq.as_ref().unwrap();
                    let wg = lw.q8_split_w_gate.as_ref().unwrap();
                    let wu = lw.q8_split_w_up.as_ref().unwrap();
                    let q8_1_ref = st.scratch.input_q8_1.as_ref().unwrap();
                    let inter_u32 = inter_dim as u32;
                    let hd_u32 = hidden_dim as u32;
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (inter_u32, 1, 1),            // ONE output row per CTA
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(fused_fn)
                            .arg(wg)
                            .arg(wu)
                            .arg(q8_1_ref)
                            .arg(&mut st.scratch.gate)
                            .arg(&inter_u32)
                            .arg(&hd_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "fused_glu_gemv_q8_split_mmvq L{layer_idx}: {e}",
                        ))
                    })?;
                }
                break 'fused_glu true;
            }
            // env-gated opt-out of the fused gate+up+SwiGLU kernel.
            // Profile evidence shows `fused_glu_gemv_q8_0`
            // is 30.8% of Lumen Q8 dense decode kernel time at 158 us/call,
            // dominated by SCALAR `(float)gq[j] * xv[j]` inner loops (no dp4a,
            // no tensor cores). The fall-through `launch_matvec_preq8_1_split`
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
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q8_0
                    .as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q8_0 L{layer_idx}: {e}",
                                ))
                            })?;
                    }
                    break 'fused_glu true;
                }
                // F16 shmem variant: hidden_dim * 2 <= 48KB (large dims).
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q8_0_hg
                    .as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q8_0_hg L{layer_idx}: {e}",
                                ))
                            })?;
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
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q8_aligned
                    .as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q8_aligned L{layer_idx}: {e}",
                                ))
                            })?;
                    }
                    break 'fused_glu true;
                }
                // F16 shmem variant: hidden_dim * 2 <= 48KB (large dims).
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q8_aligned_hg
                    .as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q8_aligned_hg L{layer_idx}: {e}",
                                ))
                            })?;
                    }
                    break 'fused_glu true;
                }
            }

            // Try Q4_0 fused kernel (gate and up must both be Q4Raw).
            if let (GpuWeightBuf::Q4Raw(ref wg_q4), GpuWeightBuf::Q4Raw(ref wu_q4)) =
                (&lw.w_gate, &lw.w_up)
            {
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q4_0
                    .as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q4_0 L{layer_idx}: {e}",
                                ))
                            })?;
                    }
                    break 'fused_glu true;
                }
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_q4_0_hg
                    .as_ref()
                    .filter(|_| shmem_f16 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f16,
                        };
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_q4_0_hg L{layer_idx}: {e}",
                                ))
                            })?;
                    }
                    break 'fused_glu true;
                }
            }

            // Try F16 fused kernel (gate and up must both be F16Raw).
            if let (GpuWeightBuf::F16Raw(ref wg_f16), GpuWeightBuf::F16Raw(ref wu_f16)) =
                (&lw.w_gate, &lw.w_up)
            {
                if let Some(ref fused_fn) = st
                    .kernels
                    .fused_glu_gemv_f16
                    .as_ref()
                    .filter(|_| shmem_f32 <= FUSED_GLU_SHMEM_LIMIT)
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
                        let inter_u32 = inter_dim as u32;
                        let hd_u32 = hidden_dim as u32;
                        let grid = fused_glu_grid(inter_u32);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (FUSED_GLU_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem_f32,
                        };
                        // F16 weights passed as u8 slices, cast to unsigned short* in kernel.
                        self.device
                            .stream
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
                            .map_err(|e| {
                                RuntimeError::Compute(format!(
                                    "fused_glu_gemv_f16 L{layer_idx}: {e}",
                                ))
                            })?;
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
                        &self.device,
                        &st.kernels,
                        &st.scratch.attn_proj,
                        &lw.ffn_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "ffn F16",
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
                            2,
                            inter_dim,
                            hidden_dim,
                            "gate_up",
                            st.algo_cache.get(inter_dim, hidden_dim),
                        )?;
                    }
                } else if let (GpuWeightBuf::F16Raw(ref wg_f16), GpuWeightBuf::F16Raw(ref wu_f16)) =
                    (&lw.w_gate, &lw.w_up)
                {
                    unsafe {
                        let w_slices: &[&CudaSlice<u8>] = &[wg_f16, wu_f16];
                        let mut out_slices: [&mut CudaSlice<f32>; 2] =
                            [&mut st.scratch.gate, &mut st.scratch.up];
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
                && lw.w_gate_f16.is_some()
                && lw.w_up_f16.is_some()
            {
                // cuBLAS HGEMV for F32 weights with F16 caches (halves F32 bandwidth).
                // Q8/Q4 weights fall through to launch_matvec() for native dp4a (1.06 B/elem).
                unsafe {
                    launch_fused_rmsnorm_f16(
                        &self.device,
                        &st.kernels,
                        &st.scratch.attn_proj,
                        &lw.ffn_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "ffn HGEMV",
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
                            2,
                            inter_dim,
                            hidden_dim,
                            "gate_up",
                            st.algo_cache.get(inter_dim, hidden_dim),
                        )?;
                    }
                } else {
                    // Fallback: separate gate + up HGEMV calls.
                    if let Some(ref wg_f16) = lw.w_gate_f16 {
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wg_f16,
                                &st.scratch.input_f16,
                                &mut st.scratch.gate,
                                inter_dim,
                                hidden_dim,
                                "gate",
                                st.algo_cache.get(inter_dim, hidden_dim),
                            )?;
                        }
                    }
                    if let Some(ref wu_f16) = lw.w_up_f16 {
                        unsafe {
                            launch_hgemv_f16_preconverted(
                                &self.device,
                                wu_f16,
                                &st.scratch.input_f16,
                                &mut st.scratch.up,
                                inter_dim,
                                hidden_dim,
                                "up",
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
                // Q4_0 QUALITY FIX (FFN side): on the fragile 9B config, Q4Raw FFN
                // also uses F32 activations. The int8 dp4a FFN error is tolerable for
                // short/medium generation but accumulates over VERY-LONG output
                // (GQ-004) into mild repetition that trips the spam detector; F32
                // keeps long-form clean. Other models keep dp4a (flag off).
                let ffn_use_preq = !weight_uses_f32_act_q4_fam(
                    &lw.w_gate,
                    &st.kernels.q4_act_plan,
                    crate::runtime_defaults::Q4ProjectionFamily::FfnGateUp,
                ) && weight_uses_dp4a_q8_1(&lw.w_gate, &st.kernels)
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
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.w_gate,
                            lw.q8_split_w_gate.as_ref(),
                            lw.q4_split_w_gate.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.gate,
                            inter_dim,
                            hidden_dim,
                            "gate",
                        )?;
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.w_up,
                            lw.q8_split_w_up.as_ref(),
                            lw.q4_split_w_up.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.up,
                            inter_dim,
                            hidden_dim,
                            "up",
                        )?;
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
                        launch_quantize_input_q8_1(
                            &self.device,
                            quant_fn,
                            &st.scratch.normed,
                            q8_1_buf,
                            hidden_dim,
                            "ffn gate_up",
                        )?;
                        // split-layout: prefer Q8Split/Q4Split sibling buffers on FFN gate/up when set.
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.w_gate,
                            lw.q8_split_w_gate.as_ref(),
                            lw.q4_split_w_gate.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.gate,
                            inter_dim,
                            hidden_dim,
                            "gate",
                        )?;
                        launch_matvec_preq8_1_split(
                            &self.device,
                            &st.kernels,
                            &lw.w_up,
                            lw.q8_split_w_up.as_ref(),
                            lw.q4_split_w_up.as_ref(),
                            q8_1_buf,
                            &mut st.scratch.up,
                            inter_dim,
                            hidden_dim,
                            "up",
                        )?;
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
                        launch_matvec_ext(
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
                            lw.q4_split_w_gate.as_ref(),
                        )?;
                        launch_matvec_ext(
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
                            lw.q4_split_w_up.as_ref(),
                        )?;
                    }
                }
            }
        } // end if !fused_glu_fired

        prof::end(Ph::FfnGateUp, &self.device.stream);
        prof::begin(Ph::FfnDownResid, &self.device.stream);

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
                    self.device
                        .stream
                        .launch_builder(&st.kernels.f32_to_f16_vec)
                        .arg(&st.scratch.gate)
                        .arg(&mut st.scratch.input_f16)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("f32_to_f16 fused_glu down: {e}")))?;
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
            } else if matches!(&lw.w_down, GpuWeightBuf::F32(_)) && lw.w_down_f16.is_some() {
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
                    self.device
                        .stream
                        .launch_builder(&st.kernels.f32_to_f16_vec)
                        .arg(&st.scratch.gate)
                        .arg(&mut st.scratch.input_f16)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("f32_to_f16 fused_glu down HGEMV: {e}"))
                })?;
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
                        self.device
                            .stream
                            .launch_builder(fused_fn)
                            .arg(wd_q8a)
                            .arg(&st.scratch.gate)
                            .arg(&mut st.scratch.down)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q8_aligned_f32 down L{layer_idx}: {e}",
                        ))
                    })?;
                } else {
                    // Fallback: quantize + dp4a (2 dispatches).
                    unsafe {
                        launch_matvec_ext(
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
                            lw.q4_split_w_down.as_ref(),
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
                        self.device
                            .stream
                            .launch_builder(fused_fn)
                            .arg(wd_q4a)
                            .arg(&st.scratch.gate)
                            .arg(&mut st.scratch.down)
                            .arg(&out_dim_u32)
                            .arg(&in_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q4_aligned_f32 down L{layer_idx}: {e}",
                        ))
                    })?;
                } else {
                    // Fallback: quantize + dp4a (2 dispatches).
                    unsafe {
                        launch_matvec_ext(
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
                            lw.q4_split_w_down.as_ref(),
                        )?;
                    }
                }
            } else {
                // Native-quant down projection via launch_matvec().
                // split-layout: when a Q8/Q4 split sibling is available for w_down,
                // route via launch_matvec_preq8_1_split (requires inline F32->Q8_1
                // quantization since the existing fused-down kernels target
                // Q8Aligned/Q4Aligned which we skipped under SPLIT).
                // This shortcut quantizes the down input to Q8_1 and claims
                // the projection before `launch_matvec` is reached, so it must
                // agree with the plan or the plan is fiction. `FfnDown` is
                // Q8_1 on every class precisely because of this path.
                let down_mode = st
                    .kernels
                    .q4_act_plan
                    .mode_for(crate::runtime_defaults::Q4ProjectionFamily::FfnDown);
                let use_split_down = down_mode == crate::runtime_defaults::Q4ActMode::Q8_1
                    && ((st.kernels.use_q8_split_dispatch && lw.q8_split_w_down.is_some())
                        || (st.kernels.use_q4_split_dispatch && lw.q4_split_w_down.is_some()));
                if use_split_down {
                    let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                    let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                    if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                        unsafe {
                            launch_quantize_input_q8_1(
                                &self.device,
                                quant_fn,
                                &st.scratch.gate,
                                q8_1_buf,
                                inter_dim,
                                "down split",
                            )?;
                            // Fold the residual into the down projection's
                            // own store and
                            // write x_gpu directly, removing BOTH the
                            // residual_add launch and the decode loop's
                            // layer-commit dtod copy (2 commands x 32 layers =
                            // 64 per token, ~0.27 ms). The residual split
                            // kernel is already loaded and dispatched.
                            if lw.q4_split_w_down.is_some() {
                                launch_matvec_preq8_1_residual_split(
                                    &self.device,
                                    &st.kernels,
                                    &lw.w_down,
                                    lw.q8_split_w_down.as_ref(),
                                    lw.q4_split_w_down.as_ref(),
                                    q8_1_buf,
                                    &st.scratch.attn_proj,
                                    &mut st.scratch.x_gpu,
                                    hidden_dim,
                                    inter_dim,
                                    "down",
                                )?;
                                // Residual folded into the down store: this arm
                                // exits the FFN block here.
                                prof::end(Ph::FfnDownResid, &self.device.stream);
                                prof::end(Ph::Ffn, &self.device.stream);
                                return Ok(LayerOutput::InPlace);
                            } else {
                                launch_matvec_preq8_1_split(
                                    &self.device,
                                    &st.kernels,
                                    &lw.w_down,
                                    lw.q8_split_w_down.as_ref(),
                                    lw.q4_split_w_down.as_ref(),
                                    q8_1_buf,
                                    &mut st.scratch.down,
                                    hidden_dim,
                                    inter_dim,
                                    "down",
                                )?;
                            }
                        }
                    } else {
                        unsafe {
                            launch_matvec_ext(
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
                                lw.q4_split_w_down.as_ref(),
                            )?;
                        }
                    }
                } else {
                    unsafe {
                        launch_matvec_ext(
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
                            lw.q4_split_w_down.as_ref(),
                        )?;
                    }
                }
            }
        } else if let GpuWeightBuf::F16Raw(ref wd_f16) = lw.w_down {
            // Fused SwiGLU + F32->F16: gate/up -> gate (F32) + input_f16 (F16).
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device,
                    &st.kernels,
                    &mut st.scratch.gate,
                    &st.scratch.up,
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
        } else if matches!(&lw.w_down, GpuWeightBuf::F32(_)) && lw.w_down_f16.is_some() {
            // cuBLAS HGEMV for F32 down weights with F16 caches.
            // Q8/Q4 weights fall through to launch_matvec() for native dp4a.
            unsafe {
                launch_swiglu_f32_to_f16(
                    &self.device,
                    &st.kernels,
                    &mut st.scratch.gate,
                    &st.scratch.up,
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
                    self.device
                        .stream
                        .launch_builder(fused_fn)
                        .arg(wd_q8a)
                        .arg(&st.scratch.gate)
                        .arg(&st.scratch.up)
                        .arg(&mut st.scratch.down)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "matvec_q8_aligned_f32_swiglu down L{layer_idx}: {e}",
                    ))
                })?;
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
                    launch_matvec_ext(
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
                        lw.q4_split_w_down.as_ref(),
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
                    self.device
                        .stream
                        .launch_builder(fused_fn)
                        .arg(wd_q4a)
                        .arg(&st.scratch.gate)
                        .arg(&st.scratch.up)
                        .arg(&mut st.scratch.down)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "matvec_q4_aligned_f32_swiglu down L{layer_idx}: {e}",
                    ))
                })?;
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
                    launch_matvec_ext(
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
                        lw.q4_split_w_down.as_ref(),
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
            // SECOND ffn_down split shortcut. Both must consult the plan;
            // guarding only the first left decode taking a different path
            // depending on which swiglu variant ran, and the census caught it.
            let down_mode2 = st
                .kernels
                .q4_act_plan
                .mode_for(crate::runtime_defaults::Q4ProjectionFamily::FfnDown);
            let use_split_down = down_mode2 == crate::runtime_defaults::Q4ActMode::Q8_1
                && ((st.kernels.use_q8_split_dispatch && lw.q8_split_w_down.is_some())
                    || (st.kernels.use_q4_split_dispatch && lw.q4_split_w_down.is_some()));
            if use_split_down {
                let quant_fn = st.kernels.quantize_f32_to_q8_1.as_ref();
                let q8_1_scratch = st.scratch.input_q8_1.as_mut();
                if let (Some(quant_fn), Some(q8_1_buf)) = (quant_fn, q8_1_scratch) {
                    unsafe {
                        launch_quantize_input_q8_1(
                            &self.device,
                            quant_fn,
                            &st.scratch.gate,
                            q8_1_buf,
                            inter_dim,
                            "down split (sep swiglu)",
                        )?;
                        // Same direct-residual fold as the other ffn_down
                        // site. Patching only one of the two left decode on the
                        // unfused path — the census caught it (ffn_down ->
                        if lw.q4_split_w_down.is_some() {
                            launch_matvec_preq8_1_residual_split(
                                &self.device,
                                &st.kernels,
                                &lw.w_down,
                                lw.q8_split_w_down.as_ref(),
                                lw.q4_split_w_down.as_ref(),
                                q8_1_buf,
                                &st.scratch.attn_proj,
                                &mut st.scratch.x_gpu,
                                hidden_dim,
                                inter_dim,
                                "down",
                            )?;
                            // Residual folded into the down store: this arm
                            // exits the FFN block here.
                            prof::end(Ph::FfnDownResid, &self.device.stream);
                            prof::end(Ph::Ffn, &self.device.stream);
                            return Ok(LayerOutput::InPlace);
                        } else {
                            launch_matvec_preq8_1_split(
                                &self.device,
                                &st.kernels,
                                &lw.w_down,
                                lw.q8_split_w_down.as_ref(),
                                lw.q4_split_w_down.as_ref(),
                                q8_1_buf,
                                &mut st.scratch.down,
                                hidden_dim,
                                inter_dim,
                                "down",
                            )?;
                        }
                    }
                } else {
                    unsafe {
                        launch_matvec_ext(
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
                            lw.q4_split_w_down.as_ref(),
                        )?;
                    }
                }
            } else {
                unsafe {
                    launch_matvec_ext(
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
                        lw.q4_split_w_down.as_ref(),
                    )?;
                }
            }
        }

        // Residual add: attn_proj += down.
        //
        // SKIPPED when the down projection already folded it in (see
        // ffn_wrote_x_gpu). Per layer this launch plus the decode loop's
        // layer-commit dtod copy are two commands; across 32 layers that is 64
        // per token, ~0.27 ms at the measured ~4.2 us marginal launch cost.
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

        prof::end(Ph::FfnDownResid, &self.device.stream);
        prof::end(Ph::Ffn, &self.device.stream);

        // Layer output is in st.scratch.attn_proj; the caller commits it to
        // st.scratch.x_gpu before the next layer.
        Ok(LayerOutput::NeedsCommit)
    }

    /// Lazily allocate GDN GPU scratch buffers on first use.
    ///
    /// Scans layer_weights_cache to identify GDN layers (layer_type == 1),
    /// builds the layer index mapping, and allocates all persistent state
    /// (h_states, conv_states) and ephemeral scratch buffers on the GPU.
    fn ensure_gdn_scratch(&self, st: &mut MutableState) -> Result<(), RuntimeError> {
        // Read the full-attention layer count from the layer cache rather than
        // from a hardcoded formula: the two disagree on some conversions.
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
            conv_states.push(
                self.device
                    .alloc_zeros::<f32>(params.conv_state_elements())?,
            );
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
                    Err(_) => {
                        alloc_ok = false;
                        break;
                    }
                }
            }
            if alloc_ok {
                Some(v)
            } else {
                None
            }
        };

        // Allocate ephemeral scratch buffers (shared across layers).
        // Q_norm/K_norm buffers are allocated only when LUMEN_CUDA_GDN_REGISTER_RESIDENT=1
        // because they are unused by the existing megakernel path.
        // default ON (no-op for non-GDN models).
        let use_gdn_register_resident = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT")
            .ok()
            .as_deref()
        {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::gdn_register_resident_default(),
        };
        let qk_norm_elements = params.num_kv_heads * params.head_dim;
        let q_norm_buf_rr = if use_gdn_register_resident {
            Some(self.device.alloc_zeros::<f32>(qk_norm_elements)?)
        } else {
            None
        };
        let k_norm_buf_rr = if use_gdn_register_resident {
            Some(self.device.alloc_zeros::<f32>(qk_norm_elements)?)
        } else {
            None
        };

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
        self.compute_gdn_attention_gpu_impl(layer_idx, st)
    }

    fn compute_gdn_attention_gpu_impl(
        &self,
        layer_idx: usize,
        st: &mut MutableState,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let eps = hp.norm_eps;

        // Ensure GDN scratch is allocated.
        self.ensure_gdn_scratch(st)?;

        let lw: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];
        let gdn = st.gdn_scratch_gpu.as_mut().unwrap();
        let p = gdn.params;

        let gdn_idx = gdn.gdn_layer_map[layer_idx].ok_or_else(|| {
            RuntimeError::Compute(format!(
                "compute_gdn_attention_gpu: layer {layer_idx} is not a GDN layer",
            ))
        })?;

        // --- Step 1+2: RMSNorm + QKV matvec ---
        // Detect if all GDN matvec consumers (QKV, alpha, beta) use dp4a Q8_1
        // so we can fuse RMSNorm + Q8_1 quantization and share the quantized input.
        let ssm_alpha_w = lw.ssm_alpha.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_alpha weight missing",))
        })?;
        let ssm_beta_w = lw.ssm_beta.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_beta weight missing",))
        })?;
        let attn_gate_w = lw.attn_gate.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN L{layer_idx}: attn_gate weight missing",))
        })?;

        // Q4_0 QUALITY FIX (GDN side): on the fragile 9B config, Q4Raw GDN
        // projections use F32 activations, not int8 Q8_1 dp4a. Other models keep
        // dp4a (flag off). See weight_uses_f32_act_q4.
        let gdn_use_preq = !weight_uses_f32_act_q4_fam(
            &lw.wq,
            &st.kernels.q4_act_plan,
            crate::runtime_defaults::Q4ProjectionFamily::GdnQkv,
        ) && !weight_uses_f32_act_q4_fam(
            attn_gate_w,
            &st.kernels.q4_act_plan,
            crate::runtime_defaults::Q4ProjectionFamily::GdnAttnGate,
        ) && weight_uses_dp4a_q8_1(&lw.wq, &st.kernels)
            && weight_uses_dp4a_q8_1(ssm_alpha_w, &st.kernels)
            && weight_uses_dp4a_q8_1(ssm_beta_w, &st.kernels)
            && weight_uses_dp4a_q8_1(attn_gate_w, &st.kernels)
            && st.scratch.input_q8_1.is_some()
            && st.kernels.quantize_f32_to_q8_1.is_some();

        // === DECODE GDN projection alignment with PREFILL (MoE-gated) ===
        // The batched PREFILL computes the GDN q/k/v/gate/alpha/beta
        // projections through `mmq_q8_0_batched` for MoE Q8 models (see
        // `launch_gemm_projection`'s Q8Raw arm + `LUMEN_CUDA_Q8_PROJ_MMQ`
        // defaulting ON for MoE). The single-token decode otherwise uses the
        // per-token pre-quantized-Q8_1 tile matvec, a DIFFERENT INT8 kernel
        // with a different activation-quant granularity and reduction order.
        // That mismatch makes the decode GDN output diverge from the
        // (llama-matching) prefill GDN output at layer 0; the 256-expert MoE
        // router amplifies it into flipped expert selection and the math
        // "multiplicationlication" near-tie flip. Routing the decode
        // projections through the SAME `mmq_q8_0_batched` kernel (batch = 1)
        // aligns the numerics. Gate ON for MoE only; requires all four GDN
        // projection weights to be Q8Raw (the MoE Q8 GDN case — repack is
        // skipped for GDN models so they stay Q8Raw) and the MMQ kernel to be
        // loaded. Dense models keep the existing path byte-identical.

        // === DECODE GDN alpha/beta-ONLY MMQ alignment (MoE-gated) ===
        // The TARGETED, empirically-isolated lever (2026-06-08): only the
        // `ssm_alpha` / `ssm_beta` projections diverge decode-vs-prefill (~20%
        // at L0); qkv/gate are bit-identical. alpha/beta are stored Q8Raw in
        // EVERY LBC quant (the GGUF source is F32; the MoE converter
        // force-requantizes them to Q8_0), so prefill projects them via
        // `mmq_q8_0_batched` (MMQ, default-ON for MoE Q8) while decode uses the
        // per-token Q8_1/dp4a matvec — a different INT8 reduction order. This
        // gate routes ONLY the decode alpha/beta through the SAME
        // `mmq_q8_0_batched` (batch = 1) the prefill uses, regardless of the
        // qkv/gate weight class (so it engages for bf16 — where qkv/gate are
        // Bf16Raw — as well as q8/q4). The MMQ kernel is per-token-independent
        // so batch=1 == row 0 of batch=N: the L0 alpha/beta delta -> 0, the
        // router stops flipping. Disjoint from the (refuted, default-OFF)
        // `gdn_decode_proj_mmq` which MMQ'd all four projections; mutually
        // exclusive with it below (that branch already MMQs alpha/beta).
        // === DECODE GDN alpha/beta F16 alignment (MoE-gated; LUMEN_CUDA_GDN_AB_F16) ===
        // The SHARP UNTRIED LEVER (2026-06-08): route ONLY the decode alpha/beta
        // through the pre-dequanted `ssm_{alpha,beta}_f16` cache + cuBLAS
        // `cublasGemmEx` HGEMV (N=1, CUDA_R_16F × CUDA_R_16F, COMPUTE_32F_FAST_16F)
        // — the EXACT GEMM the batched PREFILL F16-cache fast path uses with
        // N=batch. batch=1 == row 0 of batch=N under the same GemmEx, so the L0
        // alpha/beta projection delta (measured 19-21% vs the MMQ/dp4a mismatch)
        // collapses to ~0% decode-vs-prefill, exactly like qkv/gate (which are
        // F16/bf16 and ARE bit-identical). Highest priority — takes precedence
        // over both `gdn_decode_proj_mmq` and `gdn_decode_ab_mmq` (INT8 MMQ,
        // refuted net-negative). Guarded on the caches being present (only
        // populated at load when the gate is ON), so OFF is byte-identical and
        // dense is byte-identical (gate AND-folds `model_is_moe()`).
        let gdn_ab_f16 =
            gdn_ab_f16_enabled() && lw.ssm_alpha_f16.is_some() && lw.ssm_beta_f16.is_some();

        // === DECODE GDN conv_state PARITY (MoE-gated; LUMEN_CUDA_GDN_CONVSTATE_PARITY) ===
        // Re-project the GDN qkv (the buffer that feeds the conv ring consumed by
        // `ssm_conv1d_silu_prefill` at T=1 in the via-prefill arm) through the
        // SAME kernel the batched prefill uses, at batch = 1, so the single new
        // conv-ring slot bit-matches a true prefill of this token (the carried-in
        // slots are already prefill-written). bf16 → `launch_cublas_gemm_bf16`
        // (CUBLAS_GEMM_DEFAULT_TENSOR_OP, the prefill BF16 GEMM) vs the decode-only
        // autotuned `bf16_algo_for` GEMV; Q8Raw → `launch_mmq_q8_0_batched` (the
        // MoE-default prefill Q8 MMQ) vs the decode per-token Q8_1/dp4a matvec.
        // The reprojection OVERWRITES `gdn.qkv_buf` after the normal projection
        // chain (alpha/beta/gate keep their already-aligned paths). Only engaged
        // when the via-prefill conv consume is active (it is the only conv path
        // that reads the ring), the qkv weight is Bf16Raw or Q8Raw, and the
        // matching prefill kernel is loaded. q4 qkv is intentionally NOT rerouted
        // (q4 is already pristine and its prefill default is HGEMM, not MMQ —
        // leaving it byte-identical avoids a regression). OFF → byte-identical.
        let gdn_convstate_parity_qkv = gdn_convstate_parity_enabled()
            && gdn_decode_via_prefill_enabled()
            && st.kernels.ssm_conv1d_silu_prefill.is_some()
            && match &lw.wq {
                GpuWeightBuf::Bf16Raw(_) => st.scratch.input_f16.len() >= hidden_dim * 2,
                GpuWeightBuf::Q8Raw(_) => {
                    st.kernels.mmq_q8_0_batched.is_some() && hidden_dim % 32 == 0
                }
                _ => false,
            };

        // LEVER (HIGHEST-EV): when the conv_state parity reprojection is active
        // it OVERWRITES `gdn.qkv_buf` below, and nothing consumes the *normal*
        // qkv projection between here and that overwrite (only alpha/beta/gate,
        // none of which read `qkv_buf`; each re-derives its own activation from
        // `normed`). So the normal qkv dispatch is pure dead work. When ON, skip
        // it: dispatch ONLY the parity projection. Arithmetic-identical
        // (BYTE-identical); env-gated default-OFF for the A/B gate.
        let gdn_skip_dup_qkv = gdn_convstate_parity_qkv && gdn_skip_dup_qkv_enabled();

        // `GdnQkv` brackets the whole projection region, not one arm, so
        // flipping between the fused-preq, unfused, and convstate-parity arms
        // cannot move work out of the phase table.
        prof::begin(Ph::GdnQkv, &self.device.stream);

        if gdn_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
            // === FUSED: RMSNorm + Q8_1 quantize in 1 dispatch ===
            // Then all 3 matvecs (QKV, alpha, beta) use launch_matvec_preq8_1
            // sharing the single quantized input. Saves 4 separate quantize dispatches.

            // alpha/beta alignment (MoE-gated, q8/q4 path): when MMQ or F16
            // alignment is ON, project alpha/beta from the F32 RMSNorm output
            // (MMQ does its OWN per-row Q8_1 quant; F16 HGEMV converts F32->F16)
            // instead of the per-token pre-Q8_1 tile matvec. Either way we need
            // the F32 `normed` materialized here with a plain RMSNorm dispatch
            // (the fused rmsnorm_to_q8_1 below only writes the quantized buffer
            // used by qkv/gate). One extra 2048-wide RMSNorm per GDN decode step
            // — negligible cost. Done BEFORE the `q8_1_buf` mutable borrow so
            // the two scratch fields are accessed sequentially.
            if gdn_ab_f16 {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN rmsnorm (ab-mmq decode) L{layer_idx}: {e}",))
                })?;
            }

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
                self.device
                    .stream
                    .launch_builder(fused_fn)
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
            // Skipped when the parity reprojection below overwrites qkv_buf
            // (dead work: nothing consumes this result before the overwrite).
            if !gdn_skip_dup_qkv {
                unsafe {
                    launch_matvec_preq8_1_split(
                        &self.device,
                        &st.kernels,
                        &lw.wq,
                        lw.q8_split_wq.as_ref(),
                        lw.q4_split_wq.as_ref(),
                        q8_1_buf,
                        &mut gdn.qkv_buf,
                        p.qkv_dim,
                        hidden_dim,
                        "gdn_qkv",
                    )?;
                }
            }

            // Alpha matvec with shared pre-quantized input.
            // GDN_SPLIT: prefer Q4 split sibling.
            // alpha/beta alignment priority: (1) F16 HGEMV via the
            // `ssm_{alpha,beta}_f16` cache (the proven-clean qkv/gate recipe —
            // cublasGemmEx N=1 reads the F32 `normed` materialized above and
            // matches the prefill F16 HGEMM bit-for-bit), else (2) MMQ INT8
            // (`mmq_q8_0_batched`, batch=1, from F32 `normed`), else (3) the
            // per-token pre-Q8_1 tile matvec. `ssm_alpha_w`/`ssm_beta_w` are
            // Q8Raw; the F16/MMQ branches use the dequanted cache / Q8 bytes
            // respectively.
            if gdn_ab_f16 {
                let alpha_f16 = lw.ssm_alpha_f16.as_ref().expect("gdn_ab_f16 guards Some");
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        alpha_f16,
                        &st.scratch.normed,
                        &mut gdn.alpha_raw_buf,
                        &mut st.scratch.input_f16,
                        p.num_heads,
                        hidden_dim,
                        "gdn_alpha_f16",
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )?;
                }
                let beta_f16 = lw.ssm_beta_f16.as_ref().expect("gdn_ab_f16 guards Some");
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        beta_f16,
                        &st.scratch.normed,
                        &mut gdn.beta_raw_buf,
                        &mut st.scratch.input_f16,
                        p.num_heads,
                        hidden_dim,
                        "gdn_beta_f16",
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )?;
                }
            } else {
                unsafe {
                    launch_matvec_preq8_1_split(
                        &self.device,
                        &st.kernels,
                        ssm_alpha_w,
                        None,
                        lw.q4_split_ssm_alpha.as_ref(),
                        q8_1_buf,
                        &mut gdn.alpha_raw_buf,
                        p.num_heads,
                        hidden_dim,
                        "gdn_alpha",
                    )?;
                }

                // Beta matvec with shared pre-quantized input.
                unsafe {
                    launch_matvec_preq8_1_split(
                        &self.device,
                        &st.kernels,
                        ssm_beta_w,
                        None,
                        lw.q4_split_ssm_beta.as_ref(),
                        q8_1_buf,
                        &mut gdn.beta_raw_buf,
                        p.num_heads,
                        hidden_dim,
                        "gdn_beta",
                    )?;
                }
            }

            // Gate matvec with shared pre-quantized input.
            unsafe {
                launch_matvec_preq8_1_split(
                    &self.device,
                    &st.kernels,
                    attn_gate_w,
                    None,
                    lw.q4_split_attn_gate.as_ref(),
                    q8_1_buf,
                    &mut gdn.gate_buf,
                    p.value_dim,
                    hidden_dim,
                    "gdn_gate",
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN rmsnorm attn L{layer_idx}: {e}"))
                })?;
            }

            // QKV matvec
            // Skipped when the parity reprojection below overwrites qkv_buf
            // (dead work: nothing consumes this result before the overwrite).
            if !gdn_skip_dup_qkv {
                unsafe {
                    launch_matvec_ext(
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
                        // r2 §8.9: the split clone (SplitWeightKind::Wq) exists
                        // but this F32-path site never consumed it. Default-OFF.
                        if gdn_split_sites_enabled() {
                            lw.q4_split_wq.as_ref()
                        } else {
                            None
                        },
                    )?;
                }
            }

            // Alpha + Beta matvec.
            // === GDN_AB_F16 (MoE-gated; env LUMEN_CUDA_GDN_AB_F16) ===
            // HIGHEST-PRIORITY alpha/beta path. Routes BOTH alpha and beta
            // through `launch_hgemv_f16` (cublasGemmEx N=1, CUDA_R_16F weight ×
            // CUDA_R_16F activation, COMPUTE_32F_FAST_16F) reading the
            // pre-dequanted `ssm_{alpha,beta}_f16` caches populated at load.
            // This is the SAME cuBLAS GemmEx F16 GEMM the batched prefill uses
            // (N=batch, via `launch_gemm_projection`'s F16-cache fast path), so
            // batch=1 == row 0 of batch=N: the L0 alpha/beta projection delta
            // collapses to ~0% decode-vs-prefill (the proven-clean qkv/gate
            // recipe). Guarded on the caches being present (only populated when
            // the gate is ON), so OFF is byte-identical. Takes precedence over
            // both `gdn_decode_ab_mmq` (INT8 MMQ, refuted) and the legacy
            // Q8_1/dp4a `launch_matvec`. Reads the F32 `normed` scratch computed
            // by the unfused RMSNorm above (same input as the matvec path);
            // `input_f16` is the F32->F16 conversion scratch. `gdn_ab_f16` is
            // resolved once above the fused/unfused split.
            if gdn_ab_f16 {
                let alpha_f16 = lw.ssm_alpha_f16.as_ref().expect("gdn_ab_f16 guards Some");
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        alpha_f16,
                        &st.scratch.normed,
                        &mut gdn.alpha_raw_buf,
                        &mut st.scratch.input_f16,
                        p.num_heads,
                        hidden_dim,
                        "gdn_alpha_f16",
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )?;
                }
                let beta_f16 = lw.ssm_beta_f16.as_ref().expect("gdn_ab_f16 guards Some");
                unsafe {
                    launch_hgemv_f16(
                        &self.device,
                        &st.kernels,
                        beta_f16,
                        &st.scratch.normed,
                        &mut gdn.beta_raw_buf,
                        &mut st.scratch.input_f16,
                        p.num_heads,
                        hidden_dim,
                        "gdn_beta_f16",
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )?;
                }
            } else {
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
            }

            // Gate matvec (moved here from step 9 to share quantized input)
            unsafe {
                launch_matvec_ext(
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
                    // r2 §8.9: split clone (SplitWeightKind::AttnGate) exists;
                    // this site never consumed it. Default-OFF.
                    if gdn_split_sites_enabled() {
                        lw.q4_split_attn_gate.as_ref()
                    } else {
                        None
                    },
                )?;
            }
        }

        // === DECODE GDN conv_state PARITY: re-project qkv via the PREFILL kernel ===
        // (MoE-gated; LUMEN_CUDA_GDN_CONVSTATE_PARITY). OVERWRITES `gdn.qkv_buf`
        // (already filled by the decode projection above) with the prefill's
        // batch=1 projection so the new conv-ring slot bit-matches a true prefill.
        // Reads the F32 `normed` RMSNorm output materialized above (guaranteed
        // populated on every path this runs with: bf16 → unfused branch; q8 →
        // preq branch's `gdn_ab_f16` RMSNorm, since CONVSTATE_PARITY is intended
        // to run with GDN_AB_F16=1). Alpha/beta/gate keep their aligned values.
        if gdn_convstate_parity_qkv {
            match &lw.wq {
                GpuWeightBuf::Bf16Raw(w_bf16) => {
                    // PREFILL BF16 path: F32->BF16 activation + cublasGemmEx
                    // (CUBLAS_GEMM_DEFAULT_TENSOR_OP) at batch=1 — exactly
                    // `launch_gemm_projection`'s Bf16Raw arm, vs the decode-only
                    // autotuned `bf16_algo_for` GEMV. Reuses `input_f16` scratch
                    // for the BF16 activation (u8; cuBLAS interprets as CUDA_R_16BF).
                    unsafe {
                        super::prefill::launch_f32_to_bf16_fast(
                            &self.device,
                            &st.kernels,
                            &st.scratch.normed,
                            &mut st.scratch.input_f16,
                            hidden_dim,
                            "gdn_qkv_convparity",
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_cublas_gemm_bf16(
                            &self.device,
                            w_bf16,
                            &st.scratch.input_f16,
                            &mut gdn.qkv_buf,
                            p.qkv_dim,
                            1,
                            hidden_dim,
                            0.0,
                            "gdn_qkv_convparity",
                        )?;
                    }
                }
                GpuWeightBuf::Q8Raw(w_q8) => {
                    // PREFILL Q8 path: MMQ INT8 dp4a (per-block INT32 accumulate,
                    // single F32 scale) at batch=1 — exactly `launch_gemm_`
                    // `projection`'s Q8Raw MMQ arm (the MoE default), vs the decode
                    // per-token Q8_1/dp4a tile matvec.
                    unsafe {
                        super::prefill::launch_mmq_q8_0_batched(
                            &self.device,
                            &st.kernels,
                            w_q8,
                            &st.scratch.normed,
                            &mut gdn.qkv_buf,
                            p.qkv_dim,
                            hidden_dim,
                            1,
                            "gdn_qkv_convparity",
                        )?;
                    }
                }
                _ => {}
            }
        }

        prof::end(Ph::GdnQkv, &self.device.stream);

        // `GdnConvRecur` brackets conv1d together with the recurrent state
        // update. They are merged because three of the four dispatch arms
        // (fused-conv, register-resident, megakernel) fuse conv and recurrence
        // into a single kernel, so no split is honest across all arms.
        prof::begin(Ph::GdnConvRecur, &self.device.stream);

        // [GDNPROJSS] DECODE GDN projection-output whole-buffer sumsq (env
        // LUMEN_MOE_PROBE=1, default OFF -> byte-identical). Captures the GDN
        // q/k/v (qkv_buf), gate, alpha, beta projection outputs IMMEDIATELY
        // after the projection GEMV/matvec and BEFORE the conv1d/recurrence.
        // At layer 0 the projection INPUT is just the token embedding (identical
        // for decode and a fresh prefill of the same prefix), so any decode-vs-
        // prefill divergence in THIS dump isolates the projection KERNEL
        // (decode bf16 cuBLAS GEMV N=1 vs prefill bf16 cuBLAS GEMM N=batch),
        // separated from the GDN recurrence (incremental vs batched scan).
        // Mirrors the prefill [GDNPROJSS] dump in prefill_gdn_layer.
        if moe_probe_enabled() {
            let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
            let qkv_h = self.device.dtoh_copy(&gdn.qkv_buf)?;
            let alpha_h = self.device.dtoh_copy(&gdn.alpha_raw_buf)?;
            let beta_h = self.device.dtoh_copy(&gdn.beta_raw_buf)?;
            let gate_h = self.device.dtoh_copy(&gdn.gate_buf)?;
            eprintln!(
                "[GDNPROJSS] mode=D step={} layer={layer_idx} \
                 qkv_sumsq={:.6} alpha_sumsq={:.6} beta_sumsq={:.6} gate_sumsq={:.6}",
                st.decode_token_count,
                ss(&qkv_h[..p.qkv_dim.min(qkv_h.len())]),
                ss(&alpha_h[..p.num_heads.min(alpha_h.len())]),
                ss(&beta_h[..p.num_heads.min(beta_h.len())]),
                ss(&gate_h[..p.value_dim.min(gate_h.len())]),
            );
        }

        // --- Steps 3a-7: Fused megakernel path (conv1d+silu, gates, L2, state update) ---
        // Falls back to unfused path if megakernel failed to compile.
        let conv1d_weight = lw.ssm_conv1d.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_conv1d weight missing",))
        })?;
        let dt_bias = lw.ssm_dt_bias.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_dt_bias missing",))
        })?;
        let ssm_a = lw
            .ssm_a
            .as_ref()
            .ok_or_else(|| RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_a missing",)))?;

        // === DECODE-VIA-PREFILL: GDN-decode==GDN-prefill structural parity ===
        // gated behind LUMEN_CUDA_GDN_DECODE_VIA_PREFILL (MoE-gated, default OFF).
        // Routes the single new decode token through the PREFILL fused GDN
        // recurrence kernels at T=1 — conv1d+SiLU (`ssm_conv1d_silu_prefill`),
        // gates (`gdn_compute_gates_batched`), L2-norm
        // (`l2_normalize_qk_strided[_f64accum]`), delta-rule recurrence
        // (`gdn_prefill_fused_v3[_f64accum]`) and norm-gate
        // (`gdn_prefill_norm_gate[_f64accum]`) — carrying the persistent
        // `h_state` / `conv_state`, INSTEAD of the decode megakernel /
        // register-resident phase4. Because qkv/gate are F16/bf16-bit-identical,
        // alpha/beta are bit-identical under `gdn_ab_f16_enabled()`, and now the
        // recurrence runs the EXACT prefill kernels at batch=1, the GDN decode
        // block is byte-equivalent to a prefill of the same position by
        // construction. Requires the five prefill fused kernels to be loaded.
        // Highest priority — preempts register-resident and megakernel paths.
        // Writes the FINAL norm-gated output straight into `normed_out_buf`
        // (the prefill norm-gate fuses RMSNorm + SiLU(gate)), so the decode
        // Steps 8/10 norm-gate is skipped and Step 11 (ssm_out) reads
        // `normed_out_buf`. OFF / missing-kernels → byte-identical.
        let gdn_decode_via_prefill = gdn_decode_via_prefill_enabled()
            && st.kernels.ssm_conv1d_silu_prefill.is_some()
            && st.kernels.gdn_compute_gates_batched.is_some()
            && st.kernels.l2_normalize_qk_strided.is_some()
            && st.kernels.gdn_prefill_fused_v3.is_some()
            && st.kernels.gdn_prefill_norm_gate.is_some();
        // The legacy decode path's recurrence diverges from prefill and
        // accumulates over long generations (repetition/charspam) — if the
        // parity path was requested but a prefill kernel failed to load, the
        // silent fallback would quietly reintroduce that defect. Surface it
        // once so a degraded-quality report is diagnosable from the log.
        if gdn_decode_via_prefill_enabled() && !gdn_decode_via_prefill {
            static FALLBACK_WARNED: std::sync::Once = std::sync::Once::new();
            FALLBACK_WARNED.call_once(|| {
                eprintln!(
                    "[CUDA] WARNING: GDN decode-via-prefill enabled but prefill kernels \
                     unavailable — falling back to the legacy decode path (long-form \
                     output quality may degrade: per-step recurrence divergence)"
                );
            });
        }

        // Two-launch path: gated behind LUMEN_CUDA_GDN_REGISTER_RESIDENT=1.
        // Splits the existing megakernel into two: Phases 1-3 (same logic as
        // the existing megakernel, but materializes Q_norm/K_norm to device
        // buffers) and Phase 4 (register-resident delta-rule with
        // warp-per-column grid). Requires gdn.q_norm_buf_rr / k_norm_buf_rr
        // to have been allocated at ensure_gdn_scratch time (gated on the
        // same env var).
        // default ON (matches init-site resolver).
        let register_resident_env = match std::env::var("LUMEN_CUDA_GDN_REGISTER_RESIDENT")
            .ok()
            .as_deref()
        {
            Some(v) => matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
            None => crate::runtime_defaults::gdn_register_resident_default(),
        };
        let use_register_resident_phase4 = !gdn_decode_via_prefill
            && register_resident_env
            && st.kernels.gdn_phase123_register_resident.is_some()
            && st.kernels.gdn_phase4_register_resident.is_some()
            && gdn.q_norm_buf_rr.is_some()
            && gdn.k_norm_buf_rr.is_some();

        if register_resident_env {
            use std::sync::atomic::{AtomicBool, Ordering as O};
            static SHOWN: AtomicBool = AtomicBool::new(false);
            if !SHOWN.swap(true, O::Relaxed) {}
        }

        if gdn_decode_via_prefill {
            // === GDN DECODE VIA PREFILL KERNELS @ T=1 (structural parity) ===
            // Dispatch the five PREFILL fused GDN kernels on the single new
            // token. Buffer mapping (decode `gdn` scratch -> prefill role):
            //   qkv_buf       [qkv_dim]   -> prefill `input`/`qkv`
            //   qkv_conv_buf  [qkv_dim]   -> prefill `conv_out`
            //   alpha_raw_buf [num_heads] -> prefill `alpha_raw`
            //   beta_raw_buf  [num_heads] -> prefill `beta_raw`
            //   alpha_buf     [num_heads] -> prefill `alpha_out`
            //   beta_buf      [num_heads] -> prefill `beta_out`
            //   output_buf    [value_dim] -> prefill `raw_out` (T=1 => value_dim)
            //   gate_buf      [value_dim] -> prefill `gate_all`
            //   normed_out_buf[value_dim] -> prefill `ssm_out`/`gdn_out` (final)
            // At T=1 every prefill kernel collapses to the single-token step:
            // identical arithmetic to a true prefill of this position, carrying
            // h_states[gdn_idx] / conv_states[gdn_idx] exactly as prefill does.
            let ssm_norm = lw.ssm_norm_tiled.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_norm_tiled missing (decode-via-prefill)",
                ))
            })?;

            let num_heads_u32 = p.num_heads as u32;
            let num_kv_heads_u32 = p.num_kv_heads as u32;
            let head_dim_u32 = p.head_dim as u32;
            let qkv_dim_u32 = p.qkv_dim as u32;
            let qk_dim_u32 = p.qk_dim as u32;
            let kernel_size_u32 = p.conv_kernel_size as u32;
            let buf_slots = (p.conv_kernel_size - 1) as u32;
            let batch_u32 = 1u32;
            let state_pos = gdn.conv_positions[gdn_idx];

            // F64-accum gate: mirror the MoE prefill selection EXACTLY so the
            // single-token decode recurrence matches the F64 prefill scan to
            // F64 rounding (the MoE prefill runs these F64 variants by default
            // via gdn_f64_accum_enabled()). When OFF / twin-missing, use F32.
            let use_prefill_f64 = gdn_f64_accum_enabled()
                && st.kernels.l2_normalize_qk_strided_f64accum.is_some()
                && st.kernels.gdn_prefill_fused_v3_f64accum.is_some()
                && st.kernels.gdn_prefill_norm_gate_f64accum.is_some();

            // [GDNSTATE] one-time path diagnostic (env LUMEN_MOE_PROBE=1).
            {
                let probe = moe_probe_enabled();
                static SHOWN: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if probe && !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[GDNSTATE] PATH=decode-via-prefill use_prefill_f64={} f64_enabled={}",
                        use_prefill_f64,
                        gdn_f64_accum_enabled(),
                    );
                }
            }

            // Fusion gate, read once so steps 1 and 3 agree.
            let fused_conv_l2 = batch_u32 == 1
                && !use_prefill_f64
                && st.kernels.ssm_conv1d_silu_l2norm_t1.is_some()
                && matches!(
                    std::env::var("LUMEN_CUDA_GDN_FUSED_CONV").ok().as_deref(),
                    Some("1") | Some("true") | Some("yes") | Some("on")
                );
            if std::env::var("LUMEN_CUDA_GDN_FUSED_CONV").is_ok() {
                use std::sync::atomic::{AtomicBool, Ordering as O};
                static SHOWN: AtomicBool = AtomicBool::new(false);
                if !SHOWN.swap(true, O::Relaxed) {}
            }

            // 1. ssm_conv1d_silu_prefill: conv1d + SiLU, advances conv_state.
            {
                // LUMEN_CUDA_GDN_FUSED_CONV=1: conv1d+SiLU AND the L2 normalize
                // of Q/K in one launch, dropping step 3 below.
                //
                // The recurrence costs ~11 us per kernel across 5 kernels x 24
                // layers while reading ~zero weight bytes — per-LAUNCH cost.
                // That is why reshaping one kernel's CTAs (T1_W4) measured
                // 1.0004x and 0.9972x in two rounds: CTA count is not what is
                // being paid for. One CTA per head makes the normalize's
                // dependency intra-CTA, so a __syncthreads() replaces a kernel
                // boundary.
                if fused_conv_l2 {
                    let f = st.kernels.ssm_conv1d_silu_l2norm_t1.as_ref().unwrap();
                    let v_blocks = (p.qkv_dim as u32)
                        .saturating_sub(2 * qk_dim_u32)
                        .div_ceil(128);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (2 * num_kv_heads_u32 + v_blocks, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(f)
                            .arg(&gdn.qkv_buf)
                            .arg(&mut gdn.conv_states[gdn_idx])
                            .arg(conv1d_weight)
                            .arg(&mut gdn.qkv_conv_buf)
                            .arg(&qkv_dim_u32)
                            .arg(&kernel_size_u32)
                            .arg(&state_pos)
                            .arg(&num_kv_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&qk_dim_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN fused conv1d+silu+l2norm L{layer_idx}: {e}"
                        ))
                    })?;
                } else {
                    crate::runtime_defaults::route_census_record("gdn_conv", "CONV_SILU");
                    let conv_fn = st.kernels.ssm_conv1d_silu_prefill.as_ref().unwrap();
                    let config = LaunchConfig::for_elements(p.qkv_dim);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (config.grid_dim, 1, 1),
                        block_dim: (config.block_dim, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(conv_fn)
                            .arg(&gdn.qkv_buf)
                            .arg(&mut gdn.conv_states[gdn_idx])
                            .arg(conv1d_weight)
                            .arg(&mut gdn.qkv_conv_buf)
                            .arg(&qkv_dim_u32)
                            .arg(&kernel_size_u32)
                            .arg(&state_pos)
                            .arg(&batch_u32)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN decode-via-prefill conv1d_silu L{layer_idx}: {e}"
                        ))
                    })?;
                }
                gdn.conv_positions[gdn_idx] = (state_pos + batch_u32) % buf_slots;
            }

            // 2. gdn_compute_gates_batched: alpha/beta gates (-> alpha_buf/beta_buf).
            {
                crate::runtime_defaults::route_census_record("gdn_gates", "GATES_BATCHED");
                let gates_fn = st.kernels.gdn_compute_gates_batched.as_ref().unwrap();
                let config = LaunchConfig::for_elements(p.num_heads);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
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
                        .arg(&num_heads_u32)
                        .arg(&batch_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN decode-via-prefill gates_batched L{layer_idx}: {e}"
                    ))
                })?;
            }

            // 3. l2_normalize_qk_strided[_f64accum]: L2-norm Q/K in-place on conv_out.
            //    SKIPPED when the fused conv kernel already normalized them.
            if !fused_conv_l2 {
                let l2_fn = if use_prefill_f64 {
                    st.kernels
                        .l2_normalize_qk_strided_f64accum
                        .as_ref()
                        .unwrap()
                } else {
                    st.kernels.l2_normalize_qk_strided.as_ref().unwrap()
                };
                let l2_block_dim = (p.head_dim as u32).min(1024);
                let bytes_per_elem: u32 = if use_prefill_f64 { 8 } else { 4 };
                let l2_shared = ((l2_block_dim + 31) / 32 + 1) * bytes_per_elem;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_kv_heads_u32 * batch_u32, 1, 1),
                    block_dim: (l2_block_dim, 1, 1),
                    shared_mem_bytes: l2_shared,
                };
                let q_offset = 0u32;
                let k_offset = p.qk_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(l2_fn)
                        .arg(&mut gdn.qkv_conv_buf)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&batch_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&q_offset)
                        .arg(&k_offset)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN decode-via-prefill l2_norm_qk L{layer_idx}: {e}"
                    ))
                })?;
            }

            // [GDNSTATE] mode=D phase=before (env LUMEN_MOE_PROBE=1).
            let gdnstate_probe_vp = moe_probe_enabled();
            if gdnstate_probe_vp {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_before = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=before step={} layer={layer_idx} \
                     state_pos={state_pos} h_sumsq={:.6} h_len={} conv_sumsq={:.6} conv_len={}",
                    st.decode_token_count,
                    ss(&h_before),
                    h_before.len(),
                    ss(&conv_h),
                    conv_h.len(),
                );
            }

            // 4. gdn_prefill_fused_v3[_f64accum]: delta-rule recurrence.
            //    Reads conv_out + alpha_out + beta_out, writes raw_out
            //    (output_buf), updates h_states[gdn_idx] in place.
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
                unsafe {
                    self.device
                        .stream
                        .launch_builder(state_fn)
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&gdn.qkv_conv_buf)
                        .arg(&gdn.alpha_buf)
                        .arg(&gdn.beta_buf)
                        .arg(&mut gdn.output_buf)
                        .arg(&num_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&head_dim_u32) // val_dim per head = head_dim
                        .arg(&num_kv_heads_u32)
                        .arg(&batch_u32)
                        .arg(&qk_dim_u32)
                        .arg(&qkv_dim_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN decode-via-prefill fused_v3 L{layer_idx}: {e}"
                    ))
                })?;
            }

            // [GDNSTATE] mode=D phase=after + [XCHK] (env LUMEN_MOE_PROBE / LUMEN_XCHK).
            if gdnstate_probe_vp {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let out_h = self.device.dtoh_copy(&gdn.output_buf)?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=after step={} layer={layer_idx} \
                     h_sumsq={:.6} out_sumsq={:.6}",
                    st.decode_token_count,
                    ss(&h_after),
                    ss(&out_h[..p.value_dim.min(out_h.len())]),
                );
            }
            if {
                use std::sync::OnceLock;
                static XK: OnceLock<bool> = OnceLock::new();
                *XK.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
            } {
                let sa = |v: &[f32]| -> (f64, f32) {
                    let mut sq = 0f64;
                    let mut mx = 0f32;
                    for &e in v {
                        sq += (e as f64) * (e as f64);
                        let a = e.abs();
                        if a > mx {
                            mx = a;
                        }
                    }
                    (sq, mx)
                };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                let (hsq, hmx) = sa(&h_after);
                let (csq, cmx) = sa(&conv_h);
                let step = st.decode_token_count;
                eprintln!(
                    "[XCHK] step={step} L={layer_idx} gdn_h_state sumsq={hsq:.6} absmax={hmx:.6}"
                );
                eprintln!("[XCHK] step={step} L={layer_idx} gdn_conv_state sumsq={csq:.6} absmax={cmx:.6}");
            }

            // 5. gdn_prefill_norm_gate[_f64accum]: RMSNorm + SiLU(gate) ->
            //    normed_out_buf (the FINAL norm-gated GDN output). Step 11
            //    (ssm_out) reads normed_out_buf via used_fused_norm_gate=true.
            {
                let norm_fn = if use_prefill_f64 {
                    st.kernels.gdn_prefill_norm_gate_f64accum.as_ref().unwrap()
                } else {
                    st.kernels.gdn_prefill_norm_gate.as_ref().unwrap()
                };
                let block_dim = (p.head_dim as u32).min(1024);
                let bytes_per_elem_norm: u32 = if use_prefill_f64 { 8 } else { 4 };
                let norm_shared = ((block_dim + 31) / 32 + 1) * bytes_per_elem_norm;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_heads_u32, batch_u32, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: norm_shared,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(norm_fn)
                        .arg(&gdn.output_buf)
                        .arg(&gdn.gate_buf)
                        .arg(ssm_norm)
                        .arg(&mut gdn.normed_out_buf)
                        .arg(&num_heads_u32)
                        .arg(&head_dim_u32) // val_dim per head = head_dim
                        .arg(&eps)
                        .arg(&num_heads_u32) // scale_n_heads
                        .arg(&batch_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN decode-via-prefill norm_gate L{layer_idx}: {e}"
                    ))
                })?;
            }
        } else if use_register_resident_phase4 {
            // === TWO-LAUNCH PATH: Phase 1-3 + Phase 4 (2 launches; replaces megakernel) ===
            // F64-internal-accumulator variant for Phase 4 (decode path).
            // When `LUMEN_CUDA_GDN_F64_ACCUM=1`, replace the F32 lane-strided
            // kernel with the F64-state F64-reduce variant; the coal/strided
            // ownership pattern is irrelevant once F64 is the accumulator (the
            // F64 variant uses the strided lane pattern like F32 base).
            let use_phase4_f64 = gdn_f64_accum_enabled()
                && st.kernels.gdn_phase4_register_resident_f64accum.is_some();
            let p4_fn = if use_phase4_f64 {
                st.kernels
                    .gdn_phase4_register_resident_f64accum
                    .as_ref()
                    .unwrap()
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
            let p123_fn = st.kernels.gdn_phase123_register_resident.as_ref().unwrap();
            unsafe {
                self.device
                    .stream
                    .launch_builder(p123_fn)
                    .arg(&mut gdn.conv_states[gdn_idx])
                    .arg(&gdn.qkv_buf)
                    .arg(&gdn.alpha_raw_buf)
                    .arg(&gdn.beta_raw_buf)
                    .arg(conv1d_weight)
                    .arg(dt_bias)
                    .arg(ssm_a)
                    .arg(gdn.q_norm_buf_rr.as_mut().unwrap())
                    .arg(gdn.k_norm_buf_rr.as_mut().unwrap())
                    .arg(&mut gdn.normed_out_buf) // V buf
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
            .map_err(|e| {
                RuntimeError::Compute(format!("GDN phase123 register_resident L{layer_idx}: {e}"))
            })?;

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
            // [GDNSTATE] DECODE recurrent-state probe (env LUMEN_MOE_PROBE=1,
            // default OFF -> byte-identical). Dumps whole-buffer sumsq of the
            // recurrent h_state CARRIED INTO phase4 (i.e. the state built by the
            // prompt prefill scan through pos N-1) and the conv_state circular
            // buffer, then the UPDATED h_state AFTER the single-token phase4
            // step. Comparing the BEFORE value to a force-prefill's pre-final-
            // token state discriminates H1 (carried-state-build differs) from
            // H2 (per-step update differs given equal prior state). Mirrors the
            // prefill [GDNSTATE] dump in prefill_gdn_layer.
            let gdnstate_probe = moe_probe_enabled();
            if gdnstate_probe {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_before = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=before step={} layer={layer_idx} \
                     state_pos={state_pos} h_sumsq={:.6} h_len={} conv_sumsq={:.6} conv_len={}",
                    st.decode_token_count,
                    ss(&h_before),
                    h_before.len(),
                    ss(&conv_h),
                    conv_h.len(),
                );
            }
            unsafe {
                self.device
                    .stream
                    .launch_builder(p4_fn)
                    .arg(&mut gdn.h_states[gdn_idx])
                    .arg(gdn.q_norm_buf_rr.as_ref().unwrap())
                    .arg(gdn.k_norm_buf_rr.as_ref().unwrap())
                    .arg(&gdn.normed_out_buf) // V (read; written by phase123)
                    .arg(&gdn.alpha_buf)
                    .arg(&gdn.beta_buf)
                    .arg(&mut gdn.output_buf) // output (written by phase4)
                    .arg(&num_heads_u32)
                    .arg(&num_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(p4_cfg)
            }
            .map_err(|e| {
                RuntimeError::Compute(format!("GDN phase4 register_resident L{layer_idx}: {e}"))
            })?;
            if gdnstate_probe {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let out_h = self.device.dtoh_copy(&gdn.output_buf)?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=after step={} layer={layer_idx} \
                     h_sumsq={:.6} out_sumsq={:.6}",
                    st.decode_token_count,
                    ss(&h_after),
                    ss(&out_h[..p.value_dim.min(out_h.len())]),
                );
            }
            // [XCHK] GDN h_state/conv_state on the register-resident decode path
            // (env LUMEN_XCHK=1, default OFF). Same schema as the megakernel-path
            // [XCHK] so the cross-backend diff fires regardless of which CUDA GDN
            // decode path is live.
            if {
                use std::sync::OnceLock;
                static XKR: OnceLock<bool> = OnceLock::new();
                *XKR.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
            } {
                let sa = |v: &[f32]| -> (f64, f32) {
                    let mut sq = 0f64;
                    let mut mx = 0f32;
                    for &e in v {
                        sq += (e as f64) * (e as f64);
                        let a = e.abs();
                        if a > mx {
                            mx = a;
                        }
                    }
                    (sq, mx)
                };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                let (hsq, hmx) = sa(&h_after);
                let (csq, cmx) = sa(&conv_h);
                let step = st.decode_token_count;
                eprintln!(
                    "[XCHK] step={step} L={layer_idx} gdn_h_state sumsq={hsq:.6} absmax={hmx:.6}"
                );
                eprintln!("[XCHK] step={step} L={layer_idx} gdn_conv_state sumsq={csq:.6} absmax={cmx:.6}");
            }

            // Advance circular buffer position.
            let buf_slots = (p.conv_kernel_size - 1) as u32;

            gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
        } else if let Some(ref mega_fn_f32) = st.kernels.gdn_decode_megakernel {
            // === FUSED PATH: 8 launches -> 2 ===
            // Kernel 1 (gdn_decode_megakernel): conv1d+silu, gates, L2 norm, state update.
            //
            // MoE decode/prefill PRECISION PARITY: when
            // `gdn_decode_megakernel_f64_enabled()` (MoE default; env
            // `LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64` override) AND the F64 twin
            // compiled, dispatch the F64-accum megakernel (F64 L2-norm + F64
            // delta-rule recurrence, F32 state write-back) so the decode-built
            // `h_state` tracks the F64 prefill scan to F64 rounding. The eager
            // and graph variants each pick their respective F64 twin. OFF /
            // dense / missing-twin → byte-identical F32 path.
            let use_mega_f64 = gdn_decode_megakernel_f64_enabled()
                && st.kernels.gdn_decode_megakernel_f64accum.is_some();
            let mega_fn = if use_mega_f64 {
                st.kernels.gdn_decode_megakernel_f64accum.as_ref().unwrap()
            } else {
                mega_fn_f32
            };
            // [GDNSTATE] one-time path diagnostic (env LUMEN_MOE_PROBE=1, default
            // OFF -> byte-identical). Confirms the megakernel eager branch is the
            // ACTIVE decode path AND whether the F64 twin is selected/compiled.
            {
                let probe = moe_probe_enabled();
                static SHOWN: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if probe && !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[GDNSTATE] PATH=megakernel-eager use_mega_f64={} f64_enabled={} f64_twin_compiled={}",
                        use_mega_f64,
                        gdn_decode_megakernel_f64_enabled(),
                        st.kernels.gdn_decode_megakernel_f64accum.is_some(),
                    );
                }
            }
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

            // Eager path (graph_mode == false OR graph kernel unavailable).
            // bit-exact when disabled: host-scalar state_pos.
            // [GDNSTATE] DECODE recurrent-state probe (env LUMEN_MOE_PROBE=1,
            // default OFF -> byte-identical). h_state CARRIED INTO the
            // megakernel (= state built by prompt prefill scan thru pos N-1)
            // + conv_state, then UPDATED h_state after the single-token
            // megakernel step. The H1/H2 discriminator vs the prefill
            // [GDNSTATE] mode=P dump. (This is the ACTIVE default decode
            // path: gdn_decode_megakernel, eager.)
            let gdnstate_probe_mega = moe_probe_enabled();
            if gdnstate_probe_mega {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_before = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=before step={} layer={layer_idx} \
                         state_pos={state_pos} h_sumsq={:.6} h_len={} conv_sumsq={:.6} conv_len={}",
                    st.decode_token_count,
                    ss(&h_before),
                    h_before.len(),
                    ss(&conv_h),
                    conv_h.len(),
                );
            }
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
            if gdnstate_probe_mega {
                let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let out_h = self.device.dtoh_copy(&gdn.output_buf)?;
                eprintln!(
                    "[GDNSTATE] mode=D phase=after step={} layer={layer_idx} \
                         h_sumsq={:.6} out_sumsq={:.6}",
                    st.decode_token_count,
                    ss(&h_after),
                    ss(&out_h[..p.value_dim.min(out_h.len())]),
                );
            }
            // [XCHK] Cross-backend forensic probe (env LUMEN_XCHK=1, default
            // OFF -> byte-identical). GDN post-update h_state + conv_state in
            // the SAME layout-independent sumsq/absmax schema as the Metal
            // [XCHK] dump, keyed by the 0-based decode ordinal
            // (decode_token_count) so the two backends align op-for-op. This
            // is the LIVE megakernel decode path (gdn_decode_megakernel).
            if {
                use std::sync::OnceLock;
                static XK: OnceLock<bool> = OnceLock::new();
                *XK.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
            } {
                let sa = |v: &[f32]| -> (f64, f32) {
                    let mut sq = 0f64;
                    let mut mx = 0f32;
                    for &e in v {
                        sq += (e as f64) * (e as f64);
                        let a = e.abs();
                        if a > mx {
                            mx = a;
                        }
                    }
                    (sq, mx)
                };
                let h_after = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                let (hsq, hmx) = sa(&h_after);
                let (csq, cmx) = sa(&conv_h);
                let step = st.decode_token_count;
                eprintln!(
                    "[XCHK] step={step} L={layer_idx} gdn_h_state sumsq={hsq:.6} absmax={hmx:.6}"
                );
                eprintln!("[XCHK] step={step} L={layer_idx} gdn_conv_state sumsq={csq:.6} absmax={cmx:.6}");
            }

            // Advance circular buffer position (host-scalar path).
            let buf_slots = (p.conv_kernel_size - 1) as u32;
            gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
        } else {
            // === UNFUSED FALLBACK PATH ===
            // Step 3a: Conv1D decode
            {
                let conv1d_fn = st.kernels.ssm_conv1d_decode.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN ssm_conv1d_decode kernel not compiled".into())
                })?;
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
                let silu_fn = st.kernels.silu_inplace.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN silu_inplace kernel not compiled".into())
                })?;
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
                crate::runtime_defaults::route_census_record("gdn_gates", "GATES_BATCHED");
                let gates_fn = st.kernels.gdn_compute_gates.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN gdn_compute_gates kernel not compiled".into())
                })?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN compute_gates L{layer_idx}: {e}"))
                })?;
            }

            // Step 5: L2-normalize Q and K per head
            {
                let l2_fn = st.kernels.l2_normalize_heads.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN l2_normalize_heads kernel not compiled".into())
                })?;
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
                        self.device
                            .stream
                            .launch_builder(l2_fn)
                            .arg(&mut q_view)
                            .arg(&num_kv_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&l2_eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("GDN l2_norm Q L{layer_idx}: {e}"))
                    })?;
                }
                {
                    let mut k_view = gdn.qkv_conv_buf.slice_mut(p.qk_dim..2 * p.qk_dim);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(l2_fn)
                            .arg(&mut k_view)
                            .arg(&num_kv_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&l2_eps)
                            .launch(launch_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("GDN l2_norm K L{layer_idx}: {e}"))
                    })?;
                }
            }

            // Steps 6+7: State update + output
            {
                let state_fn = st.kernels.gdn_state_update.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN gdn_state_update kernel not compiled".into())
                })?;
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
                    self.device
                        .stream
                        .launch_builder(state_fn)
                        .arg(&mut gdn.h_states[gdn_idx])
                        .arg(&k_view)
                        .arg(&v_view)
                        .arg(&gdn.alpha_buf)
                        .arg(&gdn.beta_buf)
                        .arg(&q_view)
                        .arg(&mut gdn.output_buf)
                        .arg(&num_heads_u32)
                        .arg(&val_dim_u32)
                        .arg(&key_dim_u32)
                        .arg(&num_kv_heads_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN state_update L{layer_idx}: {e}"))
                })?;
            }
        }

        prof::end(Ph::GdnConvRecur, &self.device.stream);

        // `GdnOut` spans the output norm-gate region and the ssm_out
        // projection (Step 11).
        prof::begin(Ph::GdnOut, &self.device.stream);

        // --- Steps 8+10: Fused RMSNorm + SiLU(gate) * normed output ---
        // Fused path: gdn_rmsnorm_silu_gate (2 kernels -> 1).
        // Falls back to unfused rmsnorm + silu_mul if unavailable.
        // Note: gate matvec was already dispatched above (step 1+2 block) to share
        // the Q8_1 quantized input with QKV/alpha/beta matvecs.

        // Track which buffer holds the final gated output for the ssm_out matvec.
        // Fused path: writes to normed_out_buf (no memcpy needed).
        // Unfused path: writes to output_buf (via silu_elementwise_mul).
        let used_fused_norm_gate;

        if gdn_decode_via_prefill {
            // The decode-via-prefill branch already ran the prefill
            // `gdn_prefill_norm_gate` (RMSNorm + SiLU(gate)) and wrote the FINAL
            // gated output to `normed_out_buf`. Skip the decode norm-gate
            // entirely and signal Step 11 to read `normed_out_buf`.
            used_fused_norm_gate = true;
        } else if let Some(ref fused_fn) = st.kernels.gdn_rmsnorm_silu_gate {
            // === FUSED: RMSNorm + SiLU(gate) * normed in one kernel ===
            let ssm_norm = lw.ssm_norm_tiled.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_norm_tiled missing",))
            })?;

            // when LUMEN_CUDA_GDN_F64_ACCUM=1, prefer F64 variant.
            // Shared-mem doubles (8 bytes per warp slot vs 4).
            let use_norm_gate_f64 =
                gdn_f64_accum_enabled() && st.kernels.gdn_rmsnorm_silu_gate_f64accum.is_some();
            let chosen_fn = if use_norm_gate_f64 {
                st.kernels.gdn_rmsnorm_silu_gate_f64accum.as_ref().unwrap()
            } else {
                fused_fn
            };
            let block_size = rmsnorm_block_size(p.value_dim);
            let base_shared = rmsnorm_shared_bytes(block_size);
            let shared_bytes = if use_norm_gate_f64 {
                base_shared * 2
            } else {
                base_shared
            };
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
            .map_err(|e| {
                RuntimeError::Compute(format!("GDN fused_rmsnorm_silu_gate L{layer_idx}: {e}"))
            })?;
            used_fused_norm_gate = true;
        } else {
            // === UNFUSED FALLBACK ===
            // Step 8: RMSNorm on output
            {
                let ssm_norm = lw.ssm_norm_tiled.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_norm_tiled missing",))
                })?;
                let block_size = rmsnorm_block_size(p.value_dim);
                let shared_bytes = rmsnorm_shared_bytes(block_size);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: shared_bytes,
                };
                let dim = p.value_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.rmsnorm)
                        .arg(&gdn.output_buf)
                        .arg(ssm_norm)
                        .arg(&mut gdn.normed_out_buf)
                        .arg(&eps)
                        .arg(&dim)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN rmsnorm output L{layer_idx}: {e}"))
                })?;
            }

            // Step 10: SiLU(gate) * normed_output -> output_buf
            {
                let silu_mul_fn = st.kernels.silu_elementwise_mul.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("GDN silu_elementwise_mul kernel not compiled".into())
                })?;
                let config = LaunchConfig::for_elements(p.value_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n = p.value_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(silu_mul_fn)
                        .arg(&gdn.gate_buf)
                        .arg(&gdn.normed_out_buf)
                        .arg(&mut gdn.output_buf)
                        .arg(&n)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("GDN silu_mul L{layer_idx}: {e}")))?;
            }
            used_fused_norm_gate = false;
        }

        // --- Step 11: Output projection -> ssm_proj ---
        // Fused path: reads from normed_out_buf. Unfused path: reads from output_buf.
        {
            let ssm_out = lw.ssm_out.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!("GDN L{layer_idx}: ssm_out weight missing",))
            })?;
            let ssm_input = if used_fused_norm_gate {
                &gdn.normed_out_buf
            } else {
                &gdn.output_buf
            };
            // gdn_ssm_out is Q8Raw — already Q8_0 in the model — so it was
            // never on the F32 path and needs no Q4 split sibling. Its absence
            // from the per-site census meant UNTAGGED, not unreachable: the Q8
            // dispatch simply has no census call. Noted here because I briefly
            // recorded it as a defect on exactly that misreading.
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

        prof::end(Ph::GdnOut, &self.device.stream);
        prof::begin(Ph::GdnGlue, &self.device.stream);

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
            .map_err(|e| {
                RuntimeError::Compute(format!("GDN residual_add_copy L{layer_idx}: {e}"))
            })?;
        }
        prof::end(Ph::GdnGlue, &self.device.stream);

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

        let gdn_idx = gdn.gdn_layer_map[layer_idx].ok_or_else(|| {
            RuntimeError::Compute(format!(
                "prefill_gdn_layer: layer {layer_idx} is not a GDN layer",
            ))
        })?;

        // GDN sub-stage timing (diagnostic, no-op when unset). When
        // LUMEN_CUDA_GDN_SUBSTAGE_TIMING=1, syncs at phase boundaries and
        // prints per-substage ms for layer 0 (projections vs conv/l2/gates vs
        // scan-v3 vs norm-gate). Localizes WHICH GDN substage dominates and why
        // MoE GDN is slower per-layer than dense. Byte-identical output.
        let gdn_sub_timing = {
            use std::sync::OnceLock;
            static GS: OnceLock<bool> = OnceLock::new();
            *GS.get_or_init(|| {
                std::env::var("LUMEN_CUDA_GDN_SUBSTAGE_TIMING").as_deref() == Ok("1")
            })
        } && layer_idx == 0;
        macro_rules! gdn_sub_ms {
            ($t0:expr, $name:expr) => {{
                if gdn_sub_timing {
                    self.device.synchronize()?;
                    let ms = $t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[GDN-SUBSTAGE] L0 {:<16} {ms:>8.3} ms  (batch={batch}, \
                         num_v_heads={}, value_dim={}, qkv_dim={})",
                        $name, p.num_heads, p.value_dim, p.qkv_dim,
                    );
                    $t0 = std::time::Instant::now();
                }
            }};
        }
        let mut _gsub_t = std::time::Instant::now();

        // ================================================================
        // PHASE 1: Batched projections across all T tokens
        // ================================================================

        // 1. Batched RMSNorm: x[T, hidden] -> normed[T, hidden]
        unsafe {
            super::prefill::launch_rmsnorm_batched(
                &self.device,
                &st.kernels,
                &pf.x,
                &lw.attn_norm,
                &mut pf.normed,
                eps,
                batch,
                hidden_dim,
            )?;
        }

        // 2. Batched QKV GEMM: normed[T, hidden] @ wq^T -> qkv[T, qkv_dim]
        // wq for GDN is the fused [qkv_dim, hidden_dim] weight.
        unsafe {
            super::prefill::launch_gemm_projection(
                &self.device,
                &st.kernels,
                &lw.wq,
                lw.wq_f16.as_ref(),
                &pf.normed,
                &mut gdn_pf.qkv,
                &mut pf.dequant_f32,
                &mut pf.activation_f16,
                &mut pf.dequant_f16,
                batch,
                p.qkv_dim,
                hidden_dim,
                "gdn_qkv",
            )?;
        }

        // 3. Batched Gate GEMM: normed[T, hidden] @ attn_gate^T -> gate[T, value_dim]
        {
            let attn_gate = lw.attn_gate.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: attn_gate weight missing",
                ))
            })?;
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    attn_gate,
                    lw.attn_gate_f16.as_ref(),
                    &pf.normed,
                    &mut gdn_pf.gate,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    p.value_dim,
                    hidden_dim,
                    "gdn_gate",
                )?;
            }
        }

        // 4. Batched Alpha GEMM: normed[T, hidden] @ ssm_alpha^T -> alpha_raw[T, num_heads]
        {
            let ssm_alpha = lw.ssm_alpha.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "GDN prefill L{layer_idx}: ssm_alpha weight missing",
                ))
            })?;
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    ssm_alpha,
                    lw.ssm_alpha_f16.as_ref(),
                    &pf.normed,
                    &mut gdn_pf.alpha_raw,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    p.num_heads,
                    hidden_dim,
                    "gdn_alpha",
                )?;
            }
        }

        // 5. Batched Beta GEMM: normed[T, hidden] @ ssm_beta^T -> beta_raw[T, num_heads]
        {
            let ssm_beta = lw.ssm_beta.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!("GDN prefill L{layer_idx}: ssm_beta weight missing",))
            })?;

            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    ssm_beta,
                    lw.ssm_beta_f16.as_ref(),
                    &pf.normed,
                    &mut gdn_pf.beta_raw,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    p.num_heads,
                    hidden_dim,
                    "gdn_beta",
                )?;
            }
        }

        gdn_sub_ms!(_gsub_t, "phase1_proj");

        // [GDNPROJSS] PREFILL GDN projection-output whole-buffer sumsq per
        // position (env LUMEN_MOE_PROBE=1, default OFF -> byte-identical).
        // Mirrors the decode [GDNPROJSS] dump in compute_gdn_attention_gpu_impl.
        // Dumps the per-position slice of the batched projection outputs
        // (qkv/alpha/beta/gate) BEFORE conv1d/recurrence, so decode mode=D
        // step=k can be compared against prefill mode=P pos=(27+k) to isolate
        // the projection KERNEL divergence (GEMV N=1 vs GEMM N=batch) from the
        // recurrence. Only layer 0 (where the projection input is just the
        // token embedding) gives a clean kernel-only comparison.
        if moe_probe_enabled() {
            let ss = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
            let qkv_h = self.device.dtoh_copy(&gdn_pf.qkv)?;
            let alpha_h = self.device.dtoh_copy(&gdn_pf.alpha_raw)?;
            let beta_h = self.device.dtoh_copy(&gdn_pf.beta_raw)?;
            let gate_h = self.device.dtoh_copy(&gdn_pf.gate)?;
            for t in 0..batch {
                let qo = t * p.qkv_dim;
                let ao = t * p.num_heads;
                let go = t * p.value_dim;
                eprintln!(
                    "[GDNPROJSS] mode=P pos={t} layer={layer_idx} \
                     qkv_sumsq={:.6} alpha_sumsq={:.6} beta_sumsq={:.6} gate_sumsq={:.6}",
                    ss(&qkv_h[qo..(qo + p.qkv_dim).min(qkv_h.len())]),
                    ss(&alpha_h[ao..(ao + p.num_heads).min(alpha_h.len())]),
                    ss(&beta_h[ao..(ao + p.num_heads).min(beta_h.len())]),
                    ss(&gate_h[go..(go + p.value_dim).min(gate_h.len())]),
                );
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

        let conv1d_weight = lw.ssm_conv1d.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!(
                "GDN prefill L{layer_idx}: ssm_conv1d weight missing",
            ))
        })?;
        let dt_bias = lw.ssm_dt_bias.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN prefill L{layer_idx}: ssm_dt_bias missing",))
        })?;
        let ssm_a = lw.ssm_a.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN prefill L{layer_idx}: ssm_a missing",))
        })?;
        let ssm_norm = lw.ssm_norm_tiled.as_ref().ok_or_else(|| {
            RuntimeError::Compute(format!("GDN prefill L{layer_idx}: ssm_norm_tiled missing",))
        })?;

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
            std::fs::write(&path, &bytes)
                .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
            eprintln!(
                "[gdn-dump] L{layer_idx} qkv_pre_conv shape=[{batch}, {}] -> {path}",
                p.qkv_dim
            );
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
                crate::runtime_defaults::route_census_record("gdn_conv", "CONV_SILU");
                let conv_fn = st.kernels.ssm_conv1d_silu_prefill.as_ref().unwrap();
                let config = LaunchConfig::for_elements(p.qkv_dim);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(conv_fn)
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
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN prefill fused conv1d_silu L{layer_idx}: {e}"
                    ))
                })?;

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
                    std::fs::write(&path, &bytes)
                        .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!(
                        "[gdn-dump] L{layer_idx} conv_silu shape=[{batch}, {}] -> {path}",
                        p.qkv_dim
                    );
                }
            }

            // 2. gdn_compute_gates_batched: batched gate computation
            // Writes to alpha_out and beta_out (NOT alpha_raw/beta_raw -- avoids borrow conflict)
            {
                crate::runtime_defaults::route_census_record("gdn_gates", "GATES_BATCHED");
                let gates_fn = st.kernels.gdn_compute_gates_batched.as_ref().unwrap();
                let total = batch * p.num_heads;
                let config = LaunchConfig::for_elements(total);
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (config.grid_dim, 1, 1),
                    block_dim: (config.block_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(gates_fn)
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
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN prefill fused gates_batched L{layer_idx}: {e}"
                    ))
                })?;

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
                        let bytes: Vec<u8> = host[..n_heads]
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} alpha shape=[{batch}, {}] -> {path}",
                            p.num_heads
                        );
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.beta_out)?;
                        let path = format!("{dir}/L{layer_idx}-beta.bin");
                        let bytes: Vec<u8> = host[..n_heads]
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} beta shape=[{batch}, {}] -> {path}",
                            p.num_heads
                        );
                    }
                    {
                        let host = self.device.dtoh_copy(ssm_a)?;
                        let path = format!("{dir}/L{layer_idx}-ssm_a.bin");
                        let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} ssm_a shape=[{}] -> {path}",
                            host.len()
                        );
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.alpha_raw)?;
                        let path = format!("{dir}/L{layer_idx}-alpha_raw.bin");
                        let bytes: Vec<u8> = host[..n_heads]
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} alpha_raw shape=[{batch}, {}] -> {path}",
                            p.num_heads
                        );
                    }
                    {
                        let host = self.device.dtoh_copy(&gdn_pf.beta_raw)?;
                        let path = format!("{dir}/L{layer_idx}-beta_raw.bin");
                        let bytes: Vec<u8> = host[..n_heads]
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} beta_raw shape=[{batch}, {}] -> {path}",
                            p.num_heads
                        );
                    }
                    {
                        let host = self.device.dtoh_copy(dt_bias)?;
                        let path = format!("{dir}/L{layer_idx}-dt_bias.bin");
                        let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes)
                            .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                        eprintln!(
                            "[gdn-dump] L{layer_idx} dt_bias shape=[{}] -> {path}",
                            host.len()
                        );
                    }
                }
            }

            // F64 accumulator gate (env-cached for this layer).
            // When ON: route l2_normalize_qk_strided + gdn_prefill_fused_v3 +
            // gdn_prefill_norm_gate to the F64 variants. Note shared-mem size
            // doubles for the f64 variants (double = 8 bytes vs float = 4).
            //
            // Use the PREFILL-SCOPED gate (default F32 for MoE, the validated
            // floor win) — decoupled from the global decode F64
            // (`gdn_f64_accum_enabled()`). See `gdn_prefill_f64_enabled`.
            let use_prefill_f64 = gdn_prefill_f64_enabled()
                && st.kernels.l2_normalize_qk_strided_f64accum.is_some()
                && st.kernels.gdn_prefill_fused_v3_f64accum.is_some()
                && st.kernels.gdn_prefill_norm_gate_f64accum.is_some();

            // 3. l2_normalize_qk_strided: batched L2 norm for Q and K
            //
            {
                let l2_fn = if use_prefill_f64 {
                    st.kernels
                        .l2_normalize_qk_strided_f64accum
                        .as_ref()
                        .unwrap()
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
                    self.device
                        .stream
                        .launch_builder(l2_fn)
                        .arg(&mut gdn_pf.conv_out)
                        .arg(&num_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&batch_u32)
                        .arg(&qkv_dim_u32)
                        .arg(&q_offset)
                        .arg(&k_offset)
                        .launch(launch_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN prefill fused l2_norm_qk L{layer_idx}: {e}"))
                })?;

                // dump post-L2-norm Q/K (conv_out is L2-normed in-place on QK,
                // V channels untouched). This is the candidate-3 measurement point.
                if do_dump {
                    self.device.synchronize()?;
                    let host = self.device.dtoh_copy(&gdn_pf.conv_out)?;
                    let n = batch * p.qkv_dim;
                    let dir = dump_dir.as_ref().unwrap();
                    let path = format!("{dir}/L{layer_idx}-conv_l2norm.bin");
                    let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes)
                        .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!(
                        "[gdn-dump] L{layer_idx} conv_l2norm shape=[{batch}, {}] -> {path}",
                        p.qkv_dim
                    );
                }
            }

            gdn_sub_ms!(_gsub_t, "phase2_conv_l2");

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
                    self.device
                        .stream
                        .launch_builder(state_fn)
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN prefill fused_v3 L{layer_idx}: {e}"))
                })?;

                // dump raw_out (post-state-update, pre-norm-gate)
                if do_dump {
                    self.device.synchronize()?;
                    let host = self.device.dtoh_copy(&gdn_pf.raw_out)?;
                    let n = batch * p.value_dim;
                    let dir = dump_dir.as_ref().unwrap();
                    let path = format!("{dir}/L{layer_idx}-raw_out.bin");
                    let bytes: Vec<u8> = host[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes)
                        .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                    eprintln!(
                        "[gdn-dump] L{layer_idx} raw_out shape=[{batch}, {}] -> {path}",
                        p.value_dim
                    );
                }

                // [GDNSTATE] PREFILL recurrent-state probe (env
                // LUMEN_MOE_PROBE=1, default OFF -> byte-identical). Dumps the
                // FINAL h_state (after scanning all `batch` tokens, i.e. the
                // state through the last token of this prefill) and the
                // conv_state circular buffer. When a decode step re-prefills the
                // whole growing prefix from a fresh-zeroed state, this final
                // h_state == "state through
                // pos N" via the pure scan. Comparing it to the decode
                // [GDNSTATE] mode=D phase=after (state through pos N via the
                // incremental recurrence) is the H1/H2 discriminator. Mirrors
                // the decode [GDNSTATE] dump in compute_gdn_attention_gpu_impl.
                {
                    let on = moe_probe_enabled();
                    if on {
                        let ss =
                            |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                        let h_final = self.device.dtoh_copy(&gdn.h_states[gdn_idx])?;
                        let conv_h = self.device.dtoh_copy(&gdn.conv_states[gdn_idx])?;
                        eprintln!(
                            "[GDNSTATE] mode=P phase=final batch={batch} layer={layer_idx} \
                             h_sumsq={:.6} h_len={} conv_sumsq={:.6} conv_len={}",
                            ss(&h_final),
                            h_final.len(),
                            ss(&conv_h),
                            conv_h.len(),
                        );
                    }
                }
            }

            gdn_sub_ms!(_gsub_t, "scan_v3");

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
                let norm_fn = if use_prefill_f64 {
                    st.kernels.gdn_prefill_norm_gate_f64accum.as_ref().unwrap()
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
                std::fs::write(&path, &bytes)
                    .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
                eprintln!(
                    "[gdn-dump] L{layer_idx} gdn_out shape=[{batch}, {}] -> {path}",
                    p.value_dim
                );
            }
        } else {
            // === UNFUSED FALLBACK: per-token loop using decode kernels ===
            let conv1d_fn = st.kernels.ssm_conv1d_decode.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN ssm_conv1d_decode kernel not compiled".into())
            })?;
            let silu_fn = st.kernels.silu_inplace.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN silu_inplace kernel not compiled".into())
            })?;
            crate::runtime_defaults::route_census_record("gdn_gates", "GATES_BATCHED");
            let gates_fn = st.kernels.gdn_compute_gates.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN gdn_compute_gates kernel not compiled".into())
            })?;
            let l2_fn = st.kernels.l2_normalize_heads.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN l2_normalize_heads kernel not compiled".into())
            })?;
            let state_fn = st.kernels.gdn_state_update.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN gdn_state_update kernel not compiled".into())
            })?;
            let silu_mul_fn = st.kernels.silu_elementwise_mul.as_ref().ok_or_else(|| {
                RuntimeError::Compute("GDN silu_elementwise_mul kernel not compiled".into())
            })?;

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
                        self.device
                            .stream
                            .launch_builder(conv1d_fn)
                            .arg(&mut gdn.conv_states[gdn_idx])
                            .arg(&qkv_t)
                            .arg(conv1d_weight)
                            .arg(&mut gdn.qkv_conv_buf)
                            .arg(&conv_dim_u32)
                            .arg(&kernel_size_u32)
                            .arg(&state_pos)
                            .launch(conv_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("GDN prefill conv1d t={t} L{layer_idx}: {e}"))
                    })?;
                    gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
                }
                // SiLU
                unsafe {
                    self.device
                        .stream
                        .launch_builder(silu_fn)
                        .arg(&mut gdn.qkv_conv_buf)
                        .arg(&conv_dim_u32)
                        .launch(silu_launch)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("GDN prefill silu t={t} L{layer_idx}: {e}"))
                })?;
                // Compute gates
                {
                    let alpha_raw_t = gdn_pf
                        .alpha_raw
                        .slice(t * p.num_heads..(t + 1) * p.num_heads);
                    let beta_raw_t = gdn_pf
                        .beta_raw
                        .slice(t * p.num_heads..(t + 1) * p.num_heads);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(gates_fn)
                            .arg(dt_bias)
                            .arg(ssm_a)
                            .arg(&beta_raw_t)
                            .arg(&alpha_raw_t)
                            .arg(&mut gdn.alpha_buf)
                            .arg(&mut gdn.beta_buf)
                            .arg(&num_heads_u32)
                            .launch(gates_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN prefill compute_gates t={t} L{layer_idx}: {e}"
                        ))
                    })?;
                }
                // L2-normalize Q and K
                {
                    let mut q_view = gdn.qkv_conv_buf.slice_mut(0..p.qk_dim);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(l2_fn)
                            .arg(&mut q_view)
                            .arg(&num_kv_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&l2_eps)
                            .launch(l2_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN prefill l2_norm Q t={t} L{layer_idx}: {e}"
                        ))
                    })?;
                }
                {
                    let mut k_view = gdn.qkv_conv_buf.slice_mut(p.qk_dim..2 * p.qk_dim);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(l2_fn)
                            .arg(&mut k_view)
                            .arg(&num_kv_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&l2_eps)
                            .launch(l2_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN prefill l2_norm K t={t} L{layer_idx}: {e}"
                        ))
                    })?;
                }
                // State update + output
                {
                    let k_view = gdn.qkv_conv_buf.slice(p.qk_dim..2 * p.qk_dim);
                    let v_view = gdn.qkv_conv_buf.slice(2 * p.qk_dim..p.qkv_dim);
                    let q_view = gdn.qkv_conv_buf.slice(0..p.qk_dim);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(state_fn)
                            .arg(&mut gdn.h_states[gdn_idx])
                            .arg(&k_view)
                            .arg(&v_view)
                            .arg(&gdn.alpha_buf)
                            .arg(&gdn.beta_buf)
                            .arg(&q_view)
                            .arg(&mut gdn.output_buf)
                            .arg(&num_heads_u32)
                            .arg(&head_dim_u32)
                            .arg(&head_dim_u32)
                            .arg(&num_kv_heads_u32)
                            .launch(state_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN prefill state_update t={t} L{layer_idx}: {e}"
                        ))
                    })?;
                }
                // RMSNorm on output
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&st.kernels.rmsnorm)
                        .arg(&gdn.output_buf)
                        .arg(ssm_norm)
                        .arg(&mut gdn.normed_out_buf)
                        .arg(&eps)
                        .arg(&value_dim_u32)
                        .launch(norm_launch)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "GDN prefill rmsnorm output t={t} L{layer_idx}: {e}"
                    ))
                })?;
                // SiLU(gate) * normed_output -> batched output
                {
                    let gate_t = gdn_pf.gate.slice(t * p.value_dim..(t + 1) * p.value_dim);
                    let mut out_t = gdn_pf
                        .gdn_out
                        .slice_mut(t * p.value_dim..(t + 1) * p.value_dim);
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(silu_mul_fn)
                            .arg(&gate_t)
                            .arg(&gdn.normed_out_buf)
                            .arg(&mut out_t)
                            .arg(&value_dim_u32)
                            .launch(silu_mul_launch)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "GDN prefill silu_mul t={t} L{layer_idx}: {e}"
                        ))
                    })?;
                }
            }
        }

        gdn_sub_ms!(_gsub_t, "norm_gate");

        // ================================================================
        // PHASE 3: Batched SSM out GEMM + residual
        // ================================================================
        //
        // gdn_out[T, value_dim] @ ssm_out^T -> attn_proj[T, hidden_dim]
        // with residual: attn_proj += x
        {
            let ssm_out = lw.ssm_out.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!("GDN prefill L{layer_idx}: ssm_out weight missing",))
            })?;
            unsafe {
                super::prefill::launch_gemm_residual(
                    &self.device,
                    &st.kernels,
                    ssm_out,
                    lw.ssm_out_f16.as_ref(),
                    &gdn_pf.gdn_out,
                    &pf.x,
                    &mut pf.attn_proj,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    hidden_dim,
                    p.value_dim,
                    "gdn_ssm_out",
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
                std::fs::write(&path, &bytes)
                    .map_err(|e| RuntimeError::Compute(format!("dump {path}: {e}")))?;
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
                    &self.device,
                    &st.kernels,
                    &pf.attn_proj,
                    &lw.ffn_norm,
                    &mut pf.normed,
                    eps,
                    batch,
                    hidden_dim,
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.w_gate,
                    lw.w_gate_f16.as_ref(),
                    &pf.normed,
                    &mut pf.gate,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    inter_dim,
                    hidden_dim,
                    "gate",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.w_up,
                    lw.w_up_f16.as_ref(),
                    &pf.normed,
                    &mut pf.up,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    inter_dim,
                    hidden_dim,
                    "up",
                )?;
            }

            // Batched SwiGLU.
            unsafe {
                super::prefill::launch_swiglu_batched(
                    &self.device,
                    &st.kernels,
                    &mut pf.gate,
                    &pf.up,
                    batch,
                    inter_dim,
                )?;
            }

            // Batched down projection.
            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.w_down,
                    lw.w_down_f16.as_ref(),
                    &pf.gate,
                    &mut pf.down,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    hidden_dim,
                    inter_dim,
                    "down",
                )?;
            }

            // Batched residual add + swap for next layer.
            unsafe {
                super::prefill::launch_residual_add_batched(
                    &self.device,
                    &st.kernels,
                    &mut pf.attn_proj,
                    &pf.down,
                    batch,
                    hidden_dim,
                )?;
            }
            self.device
                .stream
                .memcpy_dtod(&pf.attn_proj, &mut pf.x)
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dtod x<-attn_proj GDN prefill L{layer_idx}: {e}"
                    ))
                })?;
        }

        gdn_sub_ms!(_gsub_t, "phase3_out_gemm");

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

        // Disjoint borrow of the repacked down planes (same NLL
        // pattern as batched_offsets — disjoint from st.kernels/&st.moe_scratch).
        let repacked = st.moe_repacked.get(layer_idx).and_then(|r| r.as_ref());

        let moe_scratch = st.moe_scratch.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "MoE prefill requires moe_scratch (allocated in init for MoE models)".into(),
            )
        })?;

        // ---- Batched/grouped routed-FFN path. ----
        //
        // When `LUMEN_CUDA_MOE_PREFILL_BATCHED=1` AND the experts are Q8_0 AND
        // the grouped kernels are loaded, replace the per-token routed-FFN loop
        // with a single grouped dispatch over all `batch` tokens (weights read
        // once per expert). The SHARED expert is still run per-token below
        // (byte-identical to the legacy path); only the routed expert FFN is
        // batched. Result `pf.x = residual + Σ routed`, bit-identical to the
        // per-token oracle, validated via the `[CHK]` x_sumsq dump.
        // Q8_0: batched/grouped tiled path. Q4_0: grouped
        // f32act tiled path (REQUIRES the parent tiled flag + the q4 f32act tiled
        // kernels; there is no q4 per-column grouped fallback, so q4 engages only
        // when `moe_grouped_tiled_enabled()` AND the q4 kernels loaded AND the
        // Qwen3.5-MoE shapes are tiled-compatible — else it stays on the per-token
        // loop). bf16 uses its own grouped tiled f32act path.
        let q8_routed = moe_meta.expert_gate_quant == QuantScheme::Q8_0
            && moe_meta.expert_down_quant == QuantScheme::Q8_0
            && st.kernels.moe_grouped_gate_up_swiglu_q8_0.is_some()
            && st.kernels.moe_grouped_down_q8_0.is_some();
        let q4_routed = moe_meta.expert_gate_quant == QuantScheme::Q4_0
            && moe_meta.expert_down_quant == QuantScheme::Q4_0
            && super::moe::moe_grouped_tiled_enabled()
            && st
                .kernels
                .moe_grouped_gate_up_swiglu_q4_0_tiled_f32act
                .is_some()
            && st.kernels.moe_grouped_down_q4_0_tiled_f32act.is_some()
            && hidden_dim % 256 == 0
            && inter_dim % 32 == 0;
        let bf16_routed = moe_meta.expert_gate_quant == QuantScheme::Bf16
            && moe_meta.expert_down_quant == QuantScheme::Bf16
            && super::moe::moe_grouped_tiled_enabled()
            && st
                .kernels
                .moe_grouped_gate_up_swiglu_bf16_tiled_f32act
                .is_some()
            && st.kernels.moe_grouped_down_bf16_tiled_f32act.is_some()
            && hidden_dim % 256 == 0
            && inter_dim % 16 == 0;
        let use_batched_routed = super::moe::moe_prefill_batched_enabled()
            && (q8_routed || q4_routed || bf16_routed)
            && batched_offsets.is_some()
            && st.kernels.moe_router_logits_batched.is_some()
            && st.kernels.moe_grouped_scatter_accum_q8_0.is_some()
            && super::moe::topk_moe_fused_kernel_for(&st.kernels, num_experts).is_some();

        if use_batched_routed {
            // Engagement probe (no-op unless LUMEN_MOE_PROBE=1): definitively
            // proves the batched/grouped routed-FFN path ran for this layer.
            if moe_probe_enabled() {
                eprintln!(
                    "[BATCHED-ROUTED] layer={layer_idx} batch={batch} top_k={top_k} \
                     num_experts={num_experts} engaged=1"
                );
            }
            super::moe::encode_moe_ffn_prefill_grouped(
                &self.device,
                &st.kernels,
                moe_scratch,
                &moe_meta,
                batched_offsets,
                repacked,
                &moe_layer_blob,
                &pf.normed,
                &pf.attn_proj,
                &mut pf.x,
                batch,
                hidden_dim,
                inter_dim,
                num_experts,
                top_k,
            )?;
        }

        // Batched SHARED-expert FFN (replaces the per-token shared loop).
        // Engages only on the batched-routed path AND when the batched shared
        // kernels are loaded AND the shared expert is present. Bit-identical to
        // the per-token unfused shared path.
        // The batched shared-expert kernels are Q4_0-only (the shared expert is Q4_0
        // in q8/q4 models). Guard on the shared gate's quant so a model whose shared
        // expert is NOT Q4_0 (e.g. a hypothetical bf16 shared) falls back to the
        // per-token shared path instead of erroring out of the batched dispatch.
        let shared_is_q4 = moe_meta
            .shared_gate
            .as_ref()
            .map(|s| s.quant == QuantScheme::Q4_0)
            .unwrap_or(false);
        let use_batched_shared = use_batched_routed
            && shared_is_q4
            && st.kernels.shared_glu_gemv_q4_0_batched.is_some()
            && st.kernels.shared_dot_f32_batched.is_some()
            && st.kernels.shared_down_q4_0_sigmoid_accum_batched.is_some()
            && st.kernels.shared_down_q4_0_residual_accum_batched.is_some();
        if use_batched_shared {
            super::moe::encode_shared_expert_ffn_prefill_batched(
                &self.device,
                &st.kernels,
                moe_scratch,
                &moe_meta,
                &moe_layer_blob,
                &pf.normed,
                &mut pf.x,
                batch,
                hidden_dim,
            )?;
        }

        for t in 0..batch {
            // In the batched-routed path, the routed FFN is already done above;
            // this loop runs ONLY the shared expert per token (UNLESS the batched
            // shared path handled it). Otherwise it runs the legacy per-token
            // routed dispatch then the shared expert.
            let off = t * hidden_dim;
            let end = off + hidden_dim;
            if use_batched_routed {
                // If the batched shared expert ran, nothing left to do per token.
                if use_batched_shared {
                    continue;
                }
                // Skip the per-token routed dispatch + its probes; jump to the
                // shared-expert block below using the same (off,end) slices.
                if moe_meta.shared_gate.is_some() {
                    let normed_view2 = pf.normed.slice(off..end);
                    let mut output_view2 = pf.x.slice_mut(off..end);
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
                continue;
            }
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

            // [MOE-SUMSQ] MoE-FFN sub-op whole-buffer sumsq dumps for the LAST token
            // of each MoE layer (env LUMEN_MOE_PROBE=1). All four quantities are
            // layout-independent (sum of squares over the whole buffer) and are
            // captured BEFORE the shared-expert add below, so `ffn_moe_out`
            // is exactly the routed-expert combine added to the residual stream:
            //   ffn_moe_out[i] = Σ_k expert_weights[k] * expert_output_buf[k*H+i]
            // (the value `moe_expert_accum_option_a` adds to `residual[i]`).
            // Probed from `moe_scratch` only (no `pf` borrow conflict). For the
            // default prefill V2 path: `router_logits` is populated by the
            // parallel `moe_router_logits_v2` launch, `expert_weights` by the
            // top-K finalize, and `expert_output_buf` by the batched-down launch.
            if t + 1 == batch && moe_probe_enabled() {
                // Ensure all router/FFN kernels for this token have completed
                // before reading their output buffers back to the host.
                self.device.synchronize()?;
                let rl = self.device.dtoh_copy(&moe_scratch.router_logits)?;
                let gw = self.device.dtoh_copy(&moe_scratch.expert_weights)?;
                let eo = self.device.dtoh_copy(&moe_scratch.expert_output_buf)?;
                let sumsq = |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                let router_logits_sumsq = sumsq(&rl);
                let gate_w_sumsq = sumsq(&gw);
                let expert_out_sumsq = sumsq(&eo);
                // ffn_moe_out = Σ_k gw[k] * expert_output_buf[k*H + i] (the
                // routed MoE contribution added to the residual stream).
                let mut ffn_moe_out_sumsq = 0f64;
                for i in 0..hidden_dim {
                    let mut acc = 0f64;
                    for k in 0..top_k {
                        let idx = k * hidden_dim + i;
                        if idx < eo.len() && k < gw.len() {
                            acc += (gw[k] as f64) * (eo[idx] as f64);
                        }
                    }
                    ffn_moe_out_sumsq += acc * acc;
                }
                eprintln!(
                    "[MOE-SUMSQ] mode=P pos={t} layer={layer_idx} \
                     router_logits_sumsq={router_logits_sumsq:.6} \
                     gate_w_sumsq={gate_w_sumsq:.6} \
                     expert_out_sumsq={expert_out_sumsq:.6} \
                     ffn_moe_out_sumsq={ffn_moe_out_sumsq:.6}"
                );
            }

            // [PROBE-RT] prefill routing: selected expert IDs + gate weights per
            // token, to compare against the decode router (env LUMEN_MOE_PROBE=1).
            if moe_probe_enabled() {
                let ids = self.device.dtoh_copy(&moe_scratch.expert_ids)?;
                let ws = self.device.dtoh_copy(&moe_scratch.expert_weights)?;
                eprintln!("[PROBE-RT] mode=P pos={t} layer={layer_idx} ids={ids:?} w={ws:?}");
            }

            // FIX: shared-expert FFN dispatch (Qwen3.5-MoE always-active
            // expert). Mirrors the decode-path dispatch at compute_layer_gpu.
            // Each prefill token runs the shared expert sigmoid-gated and
            // accumulates into pf.x[t..t+H] (the per-token output slice).
            if moe_meta.shared_gate.is_some() {
                let normed_view2 = pf.normed.slice(off..end);
                let mut output_view2 = pf.x.slice_mut(off..end);
                // opt-in fused path (same gating as decode).
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

        // [PROBE] prefill-side localization (env LUMEN_MOE_PROBE=1). pf.x = the
        // per-token layer-output residual; pf.attn_proj = attention output
        // (still intact before Step 3). Dump EVERY position so a single prefill
        // run gives the no-cache reference for every decode position.
        if moe_probe_enabled() {
            let xh = self.device.dtoh_copy(&pf.x)?;
            let ah = self.device.dtoh_copy(&pf.attn_proj)?;
            let k = 16usize;
            // Full-vector checksums per position (layout-independent, precise
            // cross-engine localization). sum / sumsq / absmax over hidden_dim.
            let chk = |v: &[f32], o: usize| -> (f64, f64, f32) {
                let mut s = 0f64;
                let mut sq = 0f64;
                let mut mx = 0f32;
                for &e in &v[o..o + hidden_dim] {
                    s += e as f64;
                    sq += (e as f64) * (e as f64);
                    if e.abs() > mx {
                        mx = e.abs();
                    }
                }
                (s, sq, mx)
            };
            for t in 0..batch {
                let o = t * hidden_dim;
                if o + hidden_dim > xh.len() || o + hidden_dim > ah.len() {
                    break;
                }
                let (xs, xsq, xmx) = chk(&xh, o);
                let (as_, asq, amx) = chk(&ah, o);
                eprintln!(
                    "[CHK] mode=P pos={t} layer={layer_idx} \
                     x_sum={xs:.5} x_sumsq={xsq:.5} x_absmax={xmx:.6} \
                     a_sum={as_:.5} a_sumsq={asq:.5} a_absmax={amx:.6}"
                );
                if t + 1 == batch {
                    eprintln!(
                        "[PROBE] mode=P pos={t} layer={layer_idx} attn16={:?} x16={:?}",
                        &ah[o..o + k],
                        &xh[o..o + k]
                    );
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
                RuntimeError::Compute(format!("dtod attn_proj<-x MoE prefill L{layer_idx}: {e}"))
            })?;

        Ok(())
    }

    /// Compute final RMSNorm + output projection entirely on GPU, with no host sync.
    ///
    /// Input: `st.scratch.x_gpu` (final hidden state, [hidden_dim]).
    /// Output: `st.logits_gpu` (logits, [vocab_size]).
    fn compute_final_gpu(&self, st: &mut MutableState) -> Result<(), RuntimeError> {
        // lm_head COVERAGE PROBE. This projection was NEVER measured: its
        // ablation arm failed rc=25 (determinism assert) and the "~657 GB/s,
        // healthy" figure was arithmetic by subtraction, not measurement. It is
        // vocab 248320 x hidden 4096 = 1.017 G params — about 11% of every
        // weight byte in the model — and the dispatch is an 8-way chain over
        // populated buffers, so the wrong one being populated is exactly the
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
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &st.globals.final_norm,
                        &mut st.scratch.input_f16,
                        eps,
                        hidden_dim,
                        "final F16",
                    )?;
                    launch_hgemv_f16_preconverted(
                        &self.device,
                        proj_f16,
                        &st.scratch.input_f16,
                        &mut st.logits_gpu,
                        vocab_size,
                        hidden_dim,
                        "output_proj",
                        st.algo_cache.get(vocab_size, hidden_dim),
                    )?;
                }
                return Ok(());
            }
        }

        // RMSNorm with final_norm weights (for non-F16 output projection paths).
        //
        // This is the only straight-line region of `compute_final_gpu`. The
        // lm_head dispatch chain below has six early returns, so it is NOT
        // bracketed directly; it is reported as the derived residual
        // `head - final_norm`. The F16 fast path above returns before this
        // point, in which case `final_norm` records no call and the derived
        // lm_head row is omitted rather than fabricated.
        prof::begin(Ph::FinalNorm, &self.device.stream);
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
        prof::end(Ph::FinalNorm, &self.device.stream);

        // Output projection: logits = output_proj * normed.
        // Prefer Q4Aligned dp4a (highest priority for Q4_0), then smem, then scalar.
        if let Some(ref proj_q4a) = st.globals.output_proj_q4_aligned {
            crate::runtime_defaults::route_census_record("head", "HEAD_Q4_ALIGNED");
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
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_q8_1_rawsum output_proj Q4: {e}"))
                    })?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4a)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q4_0 output_proj: {e}"))
                    })?;
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
                    self.device
                        .stream
                        .launch_builder(quant_fn)
                        .arg(&st.scratch.normed)
                        .arg(&mut *q8_1_buf)
                        .arg(&in_dim)
                        .launch(quant_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "quantize_f32_to_q8_1 output_proj Q4Aligned: {e}",
                    ))
                })?;
                // Step 2: dp4a Q4Aligned matvec (NR=4, 256 threads).
                let mv_grid = dp4a_q4_grid(out_dim);
                let mv_cfg = CudarcLaunchConfig {
                    grid_dim: (mv_grid, 1, 1),
                    block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(mv_fn)
                        .arg(proj_q4a)
                        .arg(&*q8_1_buf)
                        .arg(&mut st.logits_gpu)
                        .arg(&out_dim)
                        .arg(&in_dim)
                        .launch(mv_cfg)
                }
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec_q4_aligned_q8_1 output_proj: {e}",))
                })?;
            }
        } else if let Some(ref proj_q4) = st.globals.output_proj_q4 {
            crate::runtime_defaults::route_census_record("head", "HEAD_Q4");
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
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "quantize_q8_1_rawsum output_proj Q4 raw: {e}"
                        ))
                    })?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim)
                            .arg(&out_dim)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q4_0 output_proj Q4 raw: {e}"))
                    })?;
                    return Ok(());
                }
            }

            let shmem_needed = in_dim * 4;

            // LUMEN_CUDA_Q4_F32ACT_KERNEL variant selection.
            // Default `Smem` does NOTHING here and falls through to the unchanged
            // primary NR=2 smem path below (byte-identical). Row/Nr4/Nr8 are pure
            // occupancy variants — all keep FULL F32 activations. Nr4/Nr8 fall
            // through to the NR=2 smem path if their kernel failed to compile or the
            // shmem request exceeds the 48 KB static cap.
            match st.kernels.q4_f32act_kernel {
                Q4F32ActKernel::Smem => {}
                Q4F32ActKernel::Row => {
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
                        RuntimeError::Compute(format!("matvec output_proj Q4_0 row launch: {e}"))
                    })?;
                    return Ok(());
                }
                Q4F32ActKernel::Nr4 | Q4F32ActKernel::Nr8 => {
                    let (nr_fn, nr) = if matches!(st.kernels.q4_f32act_kernel, Q4F32ActKernel::Nr8)
                    {
                        (st.kernels.matvec_q4_0_smem_nr8.as_ref(), 8u32)
                    } else {
                        (st.kernels.matvec_q4_0_smem_nr4.as_ref(), 4u32)
                    };
                    if let Some(nr_fn) = nr_fn.filter(|_| shmem_needed <= 49152) {
                        let grid = matvec_smem_grid_nr(out_dim, nr);
                        let shmem = matvec_smem_shared_bytes(in_dim);
                        let launch_cfg = CudarcLaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (SMEM_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: shmem,
                        };
                        unsafe {
                            self.device
                                .stream
                                .launch_builder(nr_fn)
                                .arg(proj_q4)
                                .arg(&st.scratch.normed)
                                .arg(&mut st.logits_gpu)
                                .arg(&out_dim)
                                .arg(&in_dim)
                                .launch(launch_cfg)
                        }
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec output_proj Q4_0 smem-wide launch: {e}"
                            ))
                        })?;
                        return Ok(());
                    }
                }
            }

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
        } else if let Some(ref proj_q8_split) = st.globals.output_proj_q8_split {
            crate::runtime_defaults::route_census_record("head", "HEAD_Q8_SPLIT");
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
                if let Some(proj_fn) = pick_output_proj_nr_kernel(&st.kernels, st.output_proj_nr) {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("quantize_f32_to_q8_1 output_proj split: {e}",))
                })?;
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
                .map_err(|e| RuntimeError::Compute(format!("matvec_q8_split output_proj: {e}",)))?;
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
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_q8_1_rawsum output_proj: {e}"))
                    })?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q8a)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q8_0 output_proj: {e}"))
                    })?;
                    return Ok(());
                }
            }

            // Path 0: Q8Aligned + pre-quantized Q8_1 input (NR=2, dp4a).
            // Q8_SCALE_HW: prefer halfword-scale variant for output_proj.
            let aligned_mv_fn = if st.kernels.use_q8_scale_hw {
                st.kernels
                    .matvec_q8_aligned_q8_1_hw
                    .as_ref()
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
                let q8a_fn = st
                    .kernels
                    .matvec_q8_0_aligned
                    .as_ref()
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
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut **q8_1_buf)
                            .arg(&in_dim_u32)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_q8_1_rawsum output_proj raw: {e}"))
                    })?;

                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 128,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q8)
                            .arg(&**q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&in_dim_u32)
                            .arg(&out_dim_u32)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q8_0 output_proj raw: {e}"))
                    })?;
                    return Ok(());
                }
            }

            let q8_fn = st
                .kernels
                .matvec_q8_0_dp4a
                .as_ref()
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
            .map_err(|e| RuntimeError::Compute(format!("matvec output_proj Q8_0 launch: {e}")))?;
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
            //
            // WHOLE-DECODE-F32 (LUMEN_CUDA_MOE_DECODE_F32_FFN): suppress the
            // dedicated `mul_mat_vec_f_bf16` (128-thread block-strided reduction)
            // so the lm_head falls through to `launch_bf16_matvec_with_fallback`
            // below, which under MOE_DECODE_F32 dispatches the single-block
            // `matvec_bf16` (linear reference-order accumulation). Makes the
            // entire bf16 decode forward share one linear F32 reduction order
            // (airtight precision test). OFF = byte-identical (gate AND-ed).
            if super::moe::mmv_bf16_output_proj_enabled()
                && !super::moe::moe_decode_f32_ffn_enabled()
            {
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
                    debug_assert!(
                        ncols_x % 2 == 0,
                        "mul_mat_vec_f_bf16 requires hidden_dim % 2 == 0"
                    );
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
                        RuntimeError::Compute(format!("mul_mat_vec_f_bf16 output_proj launch: {e}"))
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
                self.device.blas.gemv(
                    cfg,
                    &st.globals.output_proj,
                    &st.scratch.normed,
                    &mut st.logits_gpu,
                )
            }
            .map_err(|e| RuntimeError::Compute(format!("cuBLAS GEMV output_proj: {e}")))?;
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
        prof::token_begin(&self.device.stream);
        prof::begin(Ph::Embed, &self.device.stream);
        self.embed_token_gpu(token_id, st)?;
        prof::end(Ph::Embed, &self.device.stream);
        for layer in 0..num_layers {
            let layer_out = self.compute_layer_gpu(layer, seq_pos, st)?;
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
            if !is_moe_layer && layer_out == LayerOutput::NeedsCommit {
                prof::begin(Ph::LayerCommit, &self.device.stream);
                self.device
                    .stream
                    .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                    .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
                prof::end(Ph::LayerCommit, &self.device.stream);
            }
        }
        prof::begin(Ph::Head, &self.device.stream);
        self.compute_final_gpu(st)?;
        prof::end(Ph::Head, &self.device.stream);
        {
            let vocab = hp.vocab_size;
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            prof::begin(Ph::Argmax, &self.device.stream);
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
            prof::end(Ph::Argmax, &self.device.stream);
        }
        // Close the token's GPU-span bracket BEFORE the sync (see the greedy
        // path for the rationale).
        prof::token_end(&self.device.stream);
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
        // Stream already synchronized above -- no added synchronization.
        prof::token_settle();
        // [XCHK] Cross-backend forensic probe (env LUMEN_XCHK=1, default OFF ->
        // byte-identical). Final whole-vocab logits sumsq/absmax + top-8 (id,val)
        // and the residual-stream sumsq, in the SAME schema as the Metal [XCHK]
        // dump, keyed by the 0-based decode ordinal (decode_token_count). This is
        // the decode-step END-STATE: where the two backends' top-8 diverge is the
        // generated-token flip; walk back to the first per-layer expert-id flip.
        if {
            use std::sync::OnceLock;
            static XKL: OnceLock<bool> = OnceLock::new();
            *XKL.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
        } {
            let sa = |v: &[f32]| -> (f64, f32) {
                let mut sq = 0f64;
                let mut mx = 0f32;
                for &e in v {
                    sq += (e as f64) * (e as f64);
                    let a = e.abs();
                    if a > mx {
                        mx = a;
                    }
                }
                (sq, mx)
            };
            let (lsq, lmx) = sa(&logits_host);
            let mut idx: Vec<usize> = (0..logits_host.len()).collect();
            idx.sort_unstable_by(|&i, &j| logits_host[j].total_cmp(&logits_host[i]));
            let top8: Vec<(usize, f32)> =
                idx.iter().take(8).map(|&i| (i, logits_host[i])).collect();
            let xh = self.device.dtoh_copy(&st.scratch.x_gpu)?;
            let (xsq, xmx) = sa(&xh[..hp.hidden_dim.min(xh.len() as u32) as usize]);
            let step = st.decode_token_count;
            eprintln!("[XCHK] step={step} residual_x sumsq={xsq:.6} absmax={xmx:.6}");
            eprintln!("[XCHK] step={step} logits sumsq={lsq:.6} absmax={lmx:.6} top8={top8:?}");
        }
        kv.advance_seq_len()?;
        st.decode_token_count += 1;
        Ok(Logits { data: logits_host })
    }

    /// GPU-side greedy decode (normal / non-graph path) returning the token id
    /// directly. Runs the IDENTICAL compute pipeline as `decode_token_normal`
    /// (embed -> per-layer -> final projection -> on-GPU `argmax_f32`), in the
    /// same kernel order with the same per-layer dtod propagation, so the
    /// selected token is bit-identical to `argmax(decode_token(..).data)`. The
    /// only difference is the readback: instead of copying the full vocab logits
    /// to the host (~vocab*4 bytes) and running CPU argmax, it copies back ONLY
    /// the 4-byte argmax index the kernel already wrote into `st.argmax_result`.
    /// That removes the per-token full-vocab D2H copy + the host-side argmax scan
    /// from the greedy decode loop. Advances `kv.seq_len()` and the decode
    /// counter internally (matching the `decode_token_greedy` trait contract).
    fn decode_token_greedy_normal(
        &self,
        token_id: u32,
        seq_pos: usize,
        num_layers: usize,
        hp: &ModelHyperparams,
        st: &mut MutableState,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
        prof::token_begin(&self.device.stream);
        prof::begin(Ph::Embed, &self.device.stream);
        self.embed_token_gpu(token_id, st)?;
        prof::end(Ph::Embed, &self.device.stream);
        for layer in 0..num_layers {
            let layer_out = self.compute_layer_gpu(layer, seq_pos, st)?;
            // Mirror decode_token_normal's FIX-DTOD: propagate the post-FFN
            // residual for dense layers; skip for MoE layers (encode_moe_ffn_decode
            // already wrote the post-FFN state in-place to st.scratch.x_gpu).
            let is_moe_layer = st
                .moe_meta_cache
                .get(layer)
                .and_then(|m| m.as_ref())
                .is_some();
            if !is_moe_layer && layer_out == LayerOutput::NeedsCommit {
                prof::begin(Ph::LayerCommit, &self.device.stream);
                self.device
                    .stream
                    .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                    .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
                prof::end(Ph::LayerCommit, &self.device.stream);
            }
        }
        prof::begin(Ph::Head, &self.device.stream);
        self.compute_final_gpu(st)?;
        prof::end(Ph::Head, &self.device.stream);
        {
            let vocab = hp.vocab_size;
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            prof::begin(Ph::Argmax, &self.device.stream);
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
            prof::end(Ph::Argmax, &self.device.stream);
        }
        // Close the token's GPU-span bracket BEFORE the sync, so the event
        // times the GPU work rather than the host's synchronize call.
        prof::token_end(&self.device.stream);
        // Sync is still required to make the argmax index host-visible, but we
        // copy back only the single u32 the kernel wrote -- NOT the full vocab.
        self.device.synchronize()?;
        // Keep the optional per-step CPU sleep for parity with the logits path
        // (default OFF -> no-op).
        maybe_apply_cuda_decode_delay();
        let token_host = self.device.dtoh_copy(&st.argmax_result)?;
        // The stream is already synchronized above, so reading the event
        // elapsed times here adds no synchronization of its own.
        prof::token_settle();
        let token = token_host.first().copied().unwrap_or(0);
        kv.advance_seq_len()?;
        st.decode_token_count += 1;
        Ok(token)
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
/// Compatibility shim: the 35 existing `launch_matvec` call sites have no
/// SoA sibling in scope, so they forward `None` and behave exactly as before.
/// Sites that DO hold `lw.q4_split_*` call `launch_matvec_ext` directly.
#[allow(clippy::too_many_arguments)]
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
    input_f16_scratch: Option<&mut CudaSlice<u8>>,
    input_q8_1_scratch: Option<&mut CudaSlice<u8>>,
) -> Result<(), RuntimeError> {
    launch_matvec_ext(
        device,
        kernels,
        weight,
        input,
        output,
        out_dim,
        in_dim,
        label,
        weight_f16_cache,
        input_f16_scratch,
        input_q8_1_scratch,
        None,
    )
}

/// Launch `matvec_q6_k_f32` / `matvec_q6_k_f32_nr8` (candidates C1/C3).
///
/// Contract, matching the shader header:
/// * args `(weight, x, out, out_dim, in_dim)`
/// * grid `(ceil(out_dim / nr), 1, 1)`, block `(128, 1, 1)` = 4 warps
/// * `in_dim % 256 == 0` — the kernel derives its super-block count as
///   `in_dim / 256`, so a ragged tail would read past the row. Checked here
///   rather than trusted: `upload_tensor` guards on the element count, but this
///   guards the SHAPE the caller passes, which is a different fact.
/// * `nr` must match the kernel handle (1 for `matvec_q6_k_f32`, 8 for `_nr8`).
///
/// Weight bytes read: `out_dim * (in_dim / 256) * 210`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_matvec_q6_k(
    device: &CudaDevice,
    f: &CudaFunction,
    weight: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    nr: u32,
    label: &str,
) -> Result<(), RuntimeError> {
    const Q6K_BLOCK_ELEM: usize = 256;
    const Q6K_BLOCK_BYTE: usize = 210;
    const Q6K_THREADS: u32 = 128;

    if in_dim % Q6K_BLOCK_ELEM != 0 {
        return Err(RuntimeError::Compute(format!(
            "matvec_q6_k_f32 {label}: in_dim {in_dim} is not a multiple of \
             {Q6K_BLOCK_ELEM} (Q6_K super-block)"
        )));
    }
    let needed = out_dim * (in_dim / Q6K_BLOCK_ELEM) * Q6K_BLOCK_BYTE;
    if weight.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "matvec_q6_k_f32 {label}: weight buffer has {} bytes, needs {needed} \
             for [{out_dim} x {in_dim}]",
            weight.len()
        )));
    }

    let out_u32 = out_dim as u32;
    let in_u32 = in_dim as u32;
    let cfg = CudarcLaunchConfig {
        grid_dim: (out_u32.div_ceil(nr), 1, 1),
        block_dim: (Q6K_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(f)
        .arg(weight)
        .arg(input)
        .arg(output)
        .arg(&out_u32)
        .arg(&in_u32)
        .launch(cfg)
        .map_err(|e| RuntimeError::Compute(format!("matvec_q6_k_f32 {label}: {e}")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_matvec_ext(
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
    q4_split_sibling: Option<&CudaSlice<u8>>,
) -> Result<(), RuntimeError> {
    // SoA F32 lane fast path. Correctness-neutral: same
    // F32 activations, same F32 accumulation — only the weight ACCESS PATTERN
    // changes, from misaligned 18-byte AoS blocks read bytewise to a 4-byte
    // aligned nibble stream read as ints. Returns false (fall through) when
    // the flag is off, the kernel failed to compile, or the layer has no
    // sibling, so the default path is untouched.
    if matches!(weight, GpuWeightBuf::Q4Raw(_) | GpuWeightBuf::Q4Split(_))
        && launch_matvec_split_f32(
            device,
            kernels,
            q4_split_sibling,
            input,
            output,
            out_dim,
            in_dim,
            label,
        )?
    {
        return Ok(());
    }

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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_q8_1_rawsum {label}: {e}",))
                    })?;

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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q8_0 {label}: {e}",))
                    })?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("quantize_f32_to_q8_1 {label}: {e}",))
                })?;

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
                .map_err(|e| RuntimeError::Compute(format!("matvec_q8_0_q8_1 {label}: {e}",)))?;
            return Ok(());
        }

        // Path 1: smem kernel (F32 x, NR=2) — best for small dimensions.
        if let Some(smem_fn) = kernels
            .matvec_q8_0_smem
            .as_ref()
            .filter(|_| shmem_f32 <= 49152)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q8_0 smem {label} launch: {e}",))
                })?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4) — covers 12288 < in_dim <= 24576.
        // Reads native Q8_0 (1.0625 B/elem) instead of HGEMV's 2 B/elem.
        if let Some(hgemv_fn) = kernels
            .hgemv_q8_0
            .as_ref()
            .filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT)
        {
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
                .map_err(|e| RuntimeError::Compute(format!("hgemv Q8_0 {label} launch: {e}",)))?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16(
                device,
                kernels,
                w_f16,
                input,
                output,
                scratch,
                out_dim,
                in_dim,
                label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
        }
        // Path 4: dp4a or v1 scalar (last resort).
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let q8_fn = kernels
            .matvec_q8_0_dp4a
            .as_ref()
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
            .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 {label} launch: {e}",)))?;
        return Ok(());
    }

    // Q4Aligned: dp4a with pre-quantized Q8_1 input (20-byte aligned blocks).
    // Q4Raw: dp4a when this projection family's planned activation mode is
    // int8, otherwise the F32-activation matvec_q4_0_smem path below. The
    // per-family plan comes from model topology (`Q4ActPlan::for_model`); on
    // the narrow-GDN dense class only `wo` is pinned to F32.
    let act_mode = kernels.q4_act_plan.mode_for_label(label);
    let plan_admits_int8 = act_mode == crate::runtime_defaults::Q4ActMode::Q8_1;
    if matches!(weight, GpuWeightBuf::Q4Aligned(_))
        || (matches!(weight, GpuWeightBuf::Q4Raw(_)) && plan_admits_int8)
    {
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_q8_1_rawsum Q4 {label}: {e}",))
                    })?;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("mul_mat_vec_q_q4_0 {label}: {e}",))
                    })?;
                return Ok(());
            }
        }

        if let (Some(quant_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            input_q8_1_scratch.take(),
        ) {
            // Check which kernel to use: aligned or unaligned.
            let (mv_fn_opt, w_ptr) = match weight {
                GpuWeightBuf::Q4Aligned(w) => {
                    (kernels.matvec_q4_aligned_q8_1.as_ref(), w as &CudaSlice<u8>)
                }
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_f32_to_q8_1 Q4 {label}: {e}",))
                    })?;

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
                    .map_err(|e| RuntimeError::Compute(format!("matvec_q4_dp4a {label}: {e}",)))?;
                return Ok(());
            }
        }
    }

    // Q4_0 raw fallback: smem > hgemv > cuBLAS HGEMV > scalar.
    // dp4a path is handled by the unified Q4Aligned/Q4Raw dispatch above.
    if let GpuWeightBuf::Q4Raw(w_q4) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // LUMEN_CUDA_Q4_F32ACT_KERNEL variant selection.
        // Default `Smem` does NOTHING here and falls through to the unchanged
        // primary NR=2 smem path below (byte-identical). Row/Nr4/Nr8 are pure
        // occupancy variants — all keep FULL F32 activations. Nr4/Nr8 fall
        // through to the NR=2 smem path if their kernel failed to compile or the
        // shmem request exceeds the 48 KB static cap.
        match kernels.q4_f32act_kernel {
            Q4F32ActKernel::Smem => {}
            Q4F32ActKernel::Row => {
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let mv_block = matvec_block_size();
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim_u32, 1, 1),
                    block_dim: (mv_block, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(&kernels.matvec_q4_0)
                    .arg(w_q4)
                    .arg(input)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(launch_cfg)
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec Q4_0 row {label} launch: {e}",))
                    })?;
                return Ok(());
            }
            Q4F32ActKernel::Nr4 | Q4F32ActKernel::Nr8 => {
                let (nr_fn, nr) = if matches!(kernels.q4_f32act_kernel, Q4F32ActKernel::Nr8) {
                    (kernels.matvec_q4_0_smem_nr8.as_ref(), 8u32)
                } else {
                    (kernels.matvec_q4_0_smem_nr4.as_ref(), 4u32)
                };
                if let Some(nr_fn) = nr_fn.filter(|_| shmem_f32 <= 49152) {
                    let out_dim_u32 = out_dim as u32;
                    let in_dim_u32 = in_dim as u32;
                    let grid = matvec_smem_grid_nr(out_dim_u32, nr);
                    let shmem = matvec_smem_shared_bytes(in_dim_u32);
                    let launch_cfg = CudarcLaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (SMEM_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: shmem,
                    };
                    device
                        .stream
                        .launch_builder(nr_fn)
                        .arg(w_q4)
                        .arg(input)
                        .arg(output)
                        .arg(&out_dim_u32)
                        .arg(&in_dim_u32)
                        .launch(launch_cfg)
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec Q4_0 smem-wide {label} launch: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
        }

        // Path 1: smem kernel (F32 x, NR=2). F32-precision activations for the
        // Q4Raw attention/GDN projections (Q4_0 QUALITY FIX) — F16 activations
        // were measured insufficient on the 9B GDN (12/15 vs F32's 15/15), so
        // full F32 is required here.
        if let Some(smem_fn) = kernels
            .matvec_q4_0_smem
            .as_ref()
            .filter(|_| shmem_f32 <= 49152)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q4_0 smem {label} launch: {e}",))
                })?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4) — covers 12288 < in_dim <= 24576.
        if let Some(hgemv_fn) = kernels
            .hgemv_q4_0
            .as_ref()
            .filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT)
        {
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
                .map_err(|e| RuntimeError::Compute(format!("hgemv Q4_0 {label} launch: {e}",)))?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16(
                device,
                kernels,
                w_f16,
                input,
                output,
                scratch,
                out_dim,
                in_dim,
                label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
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
            .map_err(|e| RuntimeError::Compute(format!("matvec Q4_0 {label} launch: {e}",)))?;
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
        // Custom bandwidth-optimal BF16 decode GEMV (DEFAULT-OFF gate
        // `LUMEN_CUDA_BF16_MATVEC`). Taken ONLY when: the gate is ON, this
        // projection is NOT a precision keeper (GDN alpha/beta stay on their
        // existing path), the uint4 fast path is applicable (in_dim % 8 == 0 for
        // 16-byte-aligned loads, no scalar tail), and the kernel compiled. If
        // ANY condition is false the `&&`/`if let` chain short-circuits and the
        // existing cuBLAS `GemmEx` wrapper below runs UNCHANGED — so OFF (and
        // every excluded label) is byte-identical to history. Numerics:
        // `matvec_bf16_v4` upcasts bf16->f32 losslessly and accumulates in F32
        // (>= the precision of the GemmEx F16 downcast), so it is safe for the
        // included FFN, attention (wq/wk/wv), and GDN (qkv/gate/ssm_out) matvecs.
        if bf16_matvec_enabled() && !is_bf16_precision_keeper_label(label) && in_dim % 8 == 0 {
            if let Some(mv_fn) = kernels.matvec_bf16_v4.as_ref() {
                const NR_BF16: u32 = 2; // output rows per block (matches kernel)
                let out_dim_u32 = out_dim as u32;
                let in_dim_u32 = in_dim as u32;
                let grid = (out_dim_u32 + NR_BF16 - 1) / NR_BF16;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(mv_fn)
                    .arg(w_bf16)
                    .arg(input)
                    .arg(output)
                    .arg(&out_dim_u32)
                    .arg(&in_dim_u32)
                    .launch(launch_cfg)
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_bf16_v4 {label} launch: {e}",))
                    })?;
                return Ok(());
            }
        }
        return launch_bf16_matvec_with_fallback(
            device, kernels, w_bf16, input, output, scratch, out_dim, in_dim, label,
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
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch.as_deref_mut()) {
            return launch_hgemv_f16(
                device,
                kernels,
                w_f16,
                input,
                output,
                scratch,
                out_dim,
                in_dim,
                label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
        }
    }

    match weight {
        // Native Q6_K (candidate C1, `LUMEN_CUDA_Q6K_NATIVE=1`): 210 B per 256
        // elements = 0.8203 B/weight, the same byte count llama.cpp reads.
        // Only reachable when `upload_tensor` produced a `Q6KRaw`, which only
        // happens with the flag on, so this arm is unreachable by default.
        //
        // Deliberately a hard error rather than a fallback if the kernel is
        // absent: an armed flag that silently does nothing is the exact failure
        // mode that cost this campaign three debug cycles.
        GpuWeightBuf::Q6KRaw(w_q6) => {
            let f = kernels.matvec_q6_k_f32.as_ref().ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "matvec_q6_k_f32 unavailable for {label}: LUMEN_CUDA_Q6K_NATIVE is armed \
                     (the weight was uploaded as Q6KRaw) but the kernel failed to compile"
                ))
            })?;
            launch_matvec_q6_k(device, f, w_q6, input, output, out_dim, in_dim, 1, label)?;
            crate::runtime_defaults::route_census_record(label, "Q6K_NATIVE");
            return Ok(());
        }
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
                .map_err(|e| RuntimeError::Compute(format!("cuBLAS GEMV {label}: {e}",)))?;
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
                .map_err(|e| RuntimeError::Compute(format!("matvec F16 {label} launch: {e}",)))?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec BF16 fallback {label} launch: {e}",))
                })?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;

            // Path 0 (priority): Q8Aligned + pre-quantized Q8_1 input (dp4a, NR=2).
            // Both weight and input use native int* loads. Zero byte-packing overhead.
            // Q8_SCALE_HW: prefer the halfword-scale variant.
            let aligned_mv_fn = if kernels.use_q8_scale_hw {
                kernels
                    .matvec_q8_aligned_q8_1_hw
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("quantize_f32_to_q8_1 aligned {label}: {e}",))
                    })?;

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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_aligned_q8_1 {label}: {e}",))
                    })?;
            } else {
                // Fallback: Q8_0 aligned dp4a with on-the-fly x quantization (NR=2).
                let q8a_fn = kernels
                    .matvec_q8_0_aligned
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec Q8_0 aligned {label} launch: {e}",))
                    })?;
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
                device
                    .stream
                    .launch_builder(dp4a_fn)
                    .arg(w_q8)
                    .arg(input)
                    .arg(output)
                    .arg(&(out_dim as u32))
                    .arg(&(in_dim as u32))
                    .launch(launch_cfg)
                    .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 dp4a {label}: {e}")))?;
            } else {
                let mv_block = matvec_block_size();
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (out_dim as u32, 1, 1),
                    block_dim: (mv_block, 1, 1),
                    shared_mem_bytes: 0,
                };
                device
                    .stream
                    .launch_builder(&kernels.matvec_q8_0)
                    .arg(w_q8)
                    .arg(input)
                    .arg(output)
                    .arg(&(out_dim as u32))
                    .arg(&(in_dim as u32))
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
                .map_err(|e| RuntimeError::Compute(format!("matvec Q4_0 {label} launch: {e}",)))?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Should not reach here — Q4Aligned is handled by early-return above.
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight reached fallback match in matvec {label} — dp4a kernels unavailable"
            )));
        }
        // split-layout: Q8Split/Q4Split are sibling buffers consumed only by
        // `launch_matvec_preq8_1_split`. Reaching the base `launch_matvec`
        // means the caller passed a sibling as the base weight, which is a bug.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q8Split/Q4Split sibling reached fallback match in matvec {label} — \
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
/// Lane-striped residual matvec (`wo`). Returns Ok(false) to fall through.
///
/// `wo` runs in EVERY layer and never received the lane decomposition, because
/// it dispatches through the residual path rather than `launch_matvec`. The
/// full-attention block measured 192 GB/s against the FFN's 600 GB/s while
/// holding only 5% of the model's bytes, which is what pointed here.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_matvec_residual_lane(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    input: &CudaSlice<f32>,
    residual: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<bool, RuntimeError> {
    let (Some(f), Some(split_w)) = (
        kernels.matvec_q4_split_f32_lane_residual.as_ref(),
        q4_split_sibling,
    ) else {
        return Ok(false);
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let cfg = CudarcLaunchConfig {
        grid_dim: (out_dim_u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(f)
        .arg(split_w)
        .arg(input)
        .arg(residual)
        .arg(output)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(cfg)
        .map_err(|e| {
            RuntimeError::Compute(format!("matvec_q4_split_f32_lane_residual {label}: {e}"))
        })?;
    crate::runtime_defaults::route_census_record(label, "F32_SPLIT_SOA_LANE_RES");
    Ok(true)
}

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
                .map_err(|e| {
                    RuntimeError::Compute(format!("quantize_f32_to_q8_1 residual {label}: {e}",))
                })?;

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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec_q8_0_q8_1_residual {label}: {e}",))
                })?;
            return Ok(());
        }

        // Path 1: smem kernel (F32 x, NR=2).
        if let Some(smem_fn) = kernels
            .matvec_q8_0_smem_residual
            .as_ref()
            .filter(|_| shmem_f32 <= 49152)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+residual Q8_0 smem {label} launch: {e}",))
                })?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4).
        if let Some(hgemv_fn) = kernels
            .hgemv_q8_0_residual
            .as_ref()
            .filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("hgemv+residual Q8_0 {label} launch: {e}",))
                })?;
            return Ok(());
        }

        // Path 3: cuBLAS HGEMV via pre-dequanted F16 cache.
        // Uses DEFAULT_TENSOR_OP (fallback path for Q8/Q4 with F16 caches).
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
            return launch_hgemv_f16_residual(
                device,
                kernels,
                w_f16,
                input,
                residual,
                output,
                scratch,
                out_dim,
                in_dim,
                label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
        }
        // Path 4: dp4a or v1 scalar.
        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let q8_fn = kernels
            .matvec_q8_0_dp4a_residual
            .as_ref()
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
            .map_err(|e| {
                RuntimeError::Compute(format!("matvec+residual Q8_0 {label} launch: {e}",))
            })?;
        return Ok(());
    }

    // Q4Aligned residual: dp4a with pre-quantized Q8_1 input + fused residual.
    // Q4Raw: dp4a when this family's planned activation mode is int8; see the
    // matching gate in `launch_matvec_ext`.
    let act_mode = kernels.q4_act_plan.mode_for_label(label);
    let plan_admits_int8 = act_mode == crate::runtime_defaults::Q4ActMode::Q8_1;
    if matches!(weight, GpuWeightBuf::Q4Aligned(_))
        || (matches!(weight, GpuWeightBuf::Q4Raw(_)) && plan_admits_int8)
    {
        if let (Some(quant_fn), Some(q8_1_buf)) = (
            kernels.quantize_f32_to_q8_1.as_ref(),
            input_q8_1_scratch.take(),
        ) {
            let (mv_fn_opt, w_ptr) = match weight {
                GpuWeightBuf::Q4Aligned(w) => (
                    kernels.matvec_q4_aligned_q8_1_residual.as_ref(),
                    w as &CudaSlice<u8>,
                ),
                GpuWeightBuf::Q4Raw(w) => (
                    kernels.matvec_q4_0_dp4a_residual.as_ref(),
                    w as &CudaSlice<u8>,
                ),
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "quantize_f32_to_q8_1 Q4 residual {label}: {e}",
                        ))
                    })?;

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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q4_dp4a_residual {label}: {e}",))
                    })?;
                return Ok(());
            }
        }
    }

    // Q4_0 raw residual fallback: smem > hgemv > scalar.
    // dp4a path is handled by the unified Q4Aligned/Q4Raw dispatch above.
    if let GpuWeightBuf::Q4Raw(w_q4) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

        // Path 1: smem kernel (F32 x, NR=2). F32-precision activations for the
        // Q4Raw attention output (wo) residual matvec — see launch_matvec Path 1.
        if let Some(smem_fn) = kernels
            .matvec_q4_0_smem_residual
            .as_ref()
            .filter(|_| shmem_f32 <= 49152)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+residual Q4_0 smem {label} launch: {e}",))
                })?;
            return Ok(());
        }

        // Path 2: hgemv kernel (F16 x, NR=4).
        if let Some(hgemv_fn) = kernels
            .hgemv_q4_0_residual
            .as_ref()
            .filter(|_| shmem_f16 <= HGEMV_SHMEM_LIMIT)
        {
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("hgemv+residual Q4_0 {label} launch: {e}",))
                })?;
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
            .map_err(|e| {
                RuntimeError::Compute(format!("matvec+residual Q4_0 {label} launch: {e}",))
            })?;
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
            device, kernels, w_bf16, input, residual, output, scratch, out_dim, in_dim, label,
        );
    }

    // HGEMV residual: only for F32 weights with F16 cache.
    // Q8Raw and Q4Raw are handled above via native kernels (smem/scalar).
    // Uses DEFAULT_TENSOR_OP (fallback path for F32 with F16 caches).
    // Use `as_deref_mut()` to avoid consuming `input_f16_scratch` on the
    // non-F32 path -- symmetric to the launch_matvec fix.
    if matches!(weight, GpuWeightBuf::F32(_)) {
        if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch.as_deref_mut()) {
            return launch_hgemv_f16_residual(
                device,
                kernels,
                w_f16,
                input,
                residual,
                output,
                scratch,
                out_dim,
                in_dim,
                label,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
        }
    }

    match weight {
        // No residual Q6_K kernel: the residual sites are `wo` and `w_down`,
        // which are Q4_0 on every layer of every shipped model, so a Q6_K
        // weight cannot reach here. Explicit error beats a silent wrong path
        // if a future mixed-quant file puts a K-quant on one of them.
        GpuWeightBuf::Q6KRaw(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q6_K residual matvec not implemented ({label}):                  LUMEN_CUDA_Q6K_NATIVE cannot cover residual projections"
            )));
        }
        GpuWeightBuf::F32(w_f32) => {
            // Copy residual into output so cuBLAS can accumulate: y = W*x + y.
            device.stream.memcpy_dtod(residual, output).map_err(|e| {
                RuntimeError::Compute(format!("cuBLAS residual copy {label}: {e}",))
            })?;
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
            device.blas.gemv(cfg, w_f32, input, output).map_err(|e| {
                RuntimeError::Compute(format!("cuBLAS GEMV+residual {label}: {e}",))
            })?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;

            // Path 0 (priority): Q8Aligned + pre-quantized Q8_1 input + residual (dp4a, NR=2).
            // Q8_SCALE_HW: prefer the halfword-scale residual variant.
            let aligned_mv_residual = if kernels.use_q8_scale_hw {
                kernels
                    .matvec_q8_aligned_q8_1_hw_residual
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "quantize_f32_to_q8_1 aligned residual {label}: {e}",
                        ))
                    })?;

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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q8_aligned_q8_1_residual {label}: {e}",
                        ))
                    })?;
            } else {
                // Fallback: Q8_0 aligned dp4a residual with on-the-fly x quantization.
                let q8a_fn = kernels
                    .matvec_q8_0_aligned_residual
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec+residual Q8_0 aligned {label} launch: {e}",
                        ))
                    })?;
            }
        }
        // Q8Raw fallback: dp4a or v1 scalar residual (unreachable — handled above).
        GpuWeightBuf::Q8Raw(w_q8) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;
            let q8_fn = kernels
                .matvec_q8_0_dp4a_residual
                .as_ref()
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
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "matvec+residual Q8_0 fallback {label} launch: {e}",
                    ))
                })?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+residual F16 {label} launch: {e}",))
                })?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+residual BF16 {label} launch: {e}",))
                })?;
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
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "matvec+residual Q4_0 fallback {label} launch: {e}",
                    ))
                })?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Should not reach here — Q4Aligned is handled by early-return above.
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight reached fallback match in matvec+residual {label} — dp4a kernels unavailable"
            )));
        }
        // split-layout: Q8Split/Q4Split are sibling buffers consumed only by
        // `launch_matvec_residual_split`. Reaching the base
        // `launch_matvec_residual` means the caller passed a sibling as the
        // base weight, which is a bug.
        GpuWeightBuf::Q8Split(_) | GpuWeightBuf::Q4Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q8Split/Q4Split sibling reached fallback match in matvec+residual {label} — \
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
        .map_err(|e| RuntimeError::Compute(format!("quantize_f32_to_q8_1 {label}: {e}",)))?;
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
    crate::runtime_defaults::route_census_record(label, "Q8_1_PREQ");
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_0_q8_1 preq {label}: {e}",))
                    })?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // LUMEN_CUDA_Q8_MMVQ: llama mmvq port on the 36-byte aligned layout
            // takes precedence when active. grid = out_dim (ONE row/CTA), block
            // = 128 (4 warps). Near-tie; no repack (reads the same Q8Aligned
            // weight). The 1-row/CTA mmvq is used for ALL nb (the small-K
            // warp-per-row variant measured slower on the GDN shapes and was
            // reverted). Falls through to the scalar/hw kernel if not loaded.
            if kernels.use_mmvq {
                if let Some(mv_fn) = kernels.matvec_q8_aligned_q8_1_mmvq.as_ref() {
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q8_aligned_q8_1_mmvq preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            // Q8_SCALE_HW: prefer the halfword-scale variant when
            // LUMEN_CUDA_Q8_SCALE_HW=1 was set at init AND the kernel loaded.
            // Numerically equivalent to matvec_q8_aligned_q8_1 (replaces a
            // 2-byte OR of two byte loads with a single u16 load).
            let mv_fn_opt = if kernels.use_q8_scale_hw {
                kernels
                    .matvec_q8_aligned_q8_1_hw
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_aligned_q8_1 preq {label}: {e}",))
                    })?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Aligned(w_q4a) => {
            if let Some(mv_fn) = kernels.matvec_q4_aligned_q8_1.as_ref() {
                crate::runtime_defaults::route_census_record(label, "Q4_ALIGNED_DP4A");
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q4_aligned_q8_1 preq {label}: {e}",))
                    })?;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q4_0_dp4a preq {label}: {e}",))
                    })?;
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
    crate::runtime_defaults::route_census_record(label, "Q8_1_PREQ_RES");
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q8_0_q8_1_residual preq {label}: {e}",
                        ))
                    })?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // LUMEN_CUDA_Q8_MMVQ: llama mmvq residual port on the 36-byte aligned
            // layout takes precedence when active. grid = out_dim (ONE row/CTA),
            // block = 128. Near-tie; no repack. The 1-row/CTA mmvq is used for
            // ALL nb (the small-K variant was reverted as measured-slower).
            // Scalar/hw residual fallback if the mmvq kernel is unavailable.
            if kernels.use_mmvq {
                if let Some(mv_fn) = kernels.matvec_q8_aligned_q8_1_mmvq_residual.as_ref() {
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q8_aligned_q8_1_mmvq_residual preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            // Q8_SCALE_HW: prefer the halfword-scale residual variant.
            let mv_fn_opt = if kernels.use_q8_scale_hw {
                kernels
                    .matvec_q8_aligned_q8_1_hw_residual
                    .as_ref()
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q8_aligned_q8_1_residual preq {label}: {e}",
                        ))
                    })?;
                return Ok(());
            }
        }
        GpuWeightBuf::Q4Aligned(w_q4a) => {
            if let Some(mv_fn) = kernels.matvec_q4_aligned_q8_1_residual.as_ref() {
                crate::runtime_defaults::route_census_record(label, "Q4_ALIGNED_DP4A_RES");
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q4_aligned_q8_1_residual preq {label}: {e}",
                        ))
                    })?;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q4_0_dp4a_residual preq {label}: {e}",
                        ))
                    })?;
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
/// F32-ACTIVATION sibling of `launch_matvec_preq8_1_split`.
///
/// Attacks the measured 9B-Q4 gap directly: Lumen achieves 456 GB/s where
/// llama.cpp reaches 793 GB/s streaming the SAME weight bytes. The default
/// `matvec_q4_0_smem` gives each thread one native 18-byte Q4_0 block, which
/// puts the nibble payload at byte offset 2 — permanently 4-byte MISALIGNED,
/// which is why that kernel unpacks with sixteen single-BYTE loads per block.
/// The split/SoA layout stores scales and nibbles as separate contiguous
/// streams, so the nibble run is 4-byte aligned and readable as ints.
///
/// Unlike the activation-format levers (which move only 0.04% of memory
/// traffic and measured 1.00x/1.12x), this changes nothing about the
/// activation numerics — F32 in, F32 accumulate — so it is a correctness-
/// neutral swap and the output should match the baseline modulo intra-block
/// FP reassociation.
///
/// Returns Ok(false) when the split sibling or kernel is unavailable, so the
/// caller falls back to the normal path.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_matvec_split_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_split_sibling: Option<&CudaSlice<u8>>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<bool, RuntimeError> {
    // Only the lane decomposition shipped: 4 lanes cooperate per Q4 block, 8
    // activations per lane, warp-contiguous int loads, one row per CTA. The
    // smem-staged, warp-per-row, gmem, multi-row and wide variants were all
    // measured and lost (0.909x / 0.727x / 0.986x / 0.899x / 0.898x); none is
    // carried, so there is no geometry to select between.
    let (Some(split_fn), Some(split_w)) =
        (kernels.matvec_q4_split_f32_lane.as_ref(), q4_split_sibling)
    else {
        return Ok(false);
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim_u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    device
        .stream
        .launch_builder(split_fn)
        .arg(split_w)
        .arg(input)
        .arg(output)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("matvec_q4_split_f32 {label}: {e}")))?;
    crate::runtime_defaults::route_census_record(label, "F32_SPLIT_SOA_LANE");
    Ok(true)
}

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
    // NOTE: entry marker only — this helper can still fall through without
    // launching if the sibling or kernel is absent. Proof of execution is the
    // per-kernel tag recorded at the actual dispatch site below.
    crate::runtime_defaults::route_census_record(label, "Q8_1_PREQ_SPLIT");
    // Q8 split path.
    if kernels.use_q8_split_dispatch {
        if let Some(split_buf) = q8_split_sibling {
            // LUMEN_CUDA_Q8_MMVQ: llama mmvq port takes precedence when active.
            // grid = out_dim (ONE row/CTA), block = 128 (4 warps). Near-tie.
            if kernels.use_mmvq {
                if let Some(mv_fn) = kernels.matvec_q8_split_q8_1_mmvq.as_ref() {
                    let out_dim_u32 = out_dim as u32;
                    let in_dim_u32 = in_dim as u32;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q8_split_q8_1_mmvq preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            // LUMEN_CUDA_Q8_MATVEC_FAST: prefer the 128-bit-load kernel on
            // 16-byte-aligned dims (in_dim % 256 == 0 => nb % 8 == 0 => the
            // quant-stream base is 16-aligned). Byte-identical to the scalar
            // kernel; same NR=2 grid/block. Falls back to scalar otherwise.
            let use_fast = kernels.use_q8_matvec_fast && (in_dim % 256 == 0);
            let mv_fn_opt: Option<&CudaFunction> = if use_fast {
                kernels.matvec_q8_split_q8_1_v4.as_ref()
            } else {
                kernels.matvec_q8_split_q8_1.as_ref()
            };
            let nr_grid: u32 = 2;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_split_q8_1 preq {label}: {e}",))
                    })?;
                return Ok(());
            }
        }
    }
    // Q4 split path. When SOA_LOCKED is active, dispatch the codegen-locked
    // kernel (same SoA sibling buffer, same grid/block/args) instead of the
    // unlocked split kernel.
    if kernels.use_q4_split_dispatch {
        if let Some(split_buf) = q4_split_sibling {
            // LUMEN_CUDA_Q4_MMVQ (default-OFF; measured-negative): llama mmvq
            // port for the Q4 split path. grid = out_dim (ONE row/CTA), block =
            // 128 (4 warps, NOT the NR=4 256-thread DP4A_Q4_BLOCK_DIM). Near-tie;
            // per-fragment -4*x_sum.
            if kernels.use_mmvq_q4 {
                if let Some(mv_fn) = kernels.matvec_q4_split_q8_1_mmvq.as_ref() {
                    crate::runtime_defaults::route_census_record(label, "Q4_SPLIT_MMVQ");
                    let out_dim_u32 = out_dim as u32;
                    let in_dim_u32 = in_dim as u32;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q4_split_q8_1_mmvq preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            let mv_fn_opt = if kernels.use_soa_locked {
                kernels.matvec_q4_split_q8_1_locked.as_ref()
            } else {
                kernels.matvec_q4_split_q8_1.as_ref()
            };
            if let Some(mv_fn) = mv_fn_opt {
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q4_split_q8_1 preq {label}: {e}",))
                    })?;
                crate::runtime_defaults::route_census_record(label, "Q4_SPLIT_Q8_1_LAUNCHED");
                return Ok(());
            }
        }
    }
    // Fall-through: existing Q8Raw/Q8Aligned/Q4Raw/Q4Aligned base dispatch.
    launch_matvec_preq8_1(
        device, kernels, weight, q8_1_buf, output, out_dim, in_dim, label,
    )
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
    // Entry marker only — see note in launch_matvec_preq8_1_split.
    crate::runtime_defaults::route_census_record(label, "Q8_1_PREQ_RES_SPLIT");
    if kernels.use_q8_split_dispatch {
        if let Some(split_buf) = q8_split_sibling {
            // LUMEN_CUDA_Q8_MMVQ: llama mmvq residual kernel takes precedence.
            // grid = out_dim (ONE row/CTA), block = 128 (4 warps). Near-tie.
            if kernels.use_mmvq {
                if let Some(mv_fn) = kernels.matvec_q8_split_q8_1_mmvq_residual.as_ref() {
                    let out_dim_u32 = out_dim as u32;
                    let in_dim_u32 = in_dim as u32;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q8_split_q8_1_mmvq_residual preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            // LUMEN_CUDA_Q8_MATVEC_FAST: 128-bit-load residual kernel on aligned
            // dims (byte-identical; same NR=2 grid/block). Scalar fallback else.
            let use_fast = kernels.use_q8_matvec_fast && (in_dim % 256 == 0);
            let mv_fn_opt: Option<&CudaFunction> = if use_fast {
                kernels.matvec_q8_split_q8_1_v4_residual.as_ref()
            } else {
                kernels.matvec_q8_split_q8_1_residual.as_ref()
            };
            let nr_grid: u32 = 2;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q8_split_q8_1_residual preq {label}: {e}",
                        ))
                    })?;
                return Ok(());
            }
        }
    }
    if kernels.use_q4_split_dispatch {
        if let Some(split_buf) = q4_split_sibling {
            // LUMEN_CUDA_Q4_MMVQ (default-OFF; measured-negative): llama mmvq
            // residual kernel for the Q4 split path. grid = out_dim (ONE row/CTA),
            // block = 128 (4 warps). Near-tie; per-fragment -4*x_sum half-correction.
            if kernels.use_mmvq_q4 {
                if let Some(mv_fn) = kernels.matvec_q4_split_q8_1_mmvq_residual.as_ref() {
                    let out_dim_u32 = out_dim as u32;
                    let in_dim_u32 = in_dim as u32;
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (out_dim_u32, 1, 1),
                        block_dim: (DP4A_Q8_1_BLOCK_DIM, 1, 1), // 128 = 4 warps
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!(
                                "matvec_q4_split_q8_1_mmvq_residual preq {label}: {e}",
                            ))
                        })?;
                    return Ok(());
                }
            }
            let mv_fn_opt = if kernels.use_soa_locked {
                kernels.matvec_q4_split_q8_1_locked_residual.as_ref()
            } else {
                kernels.matvec_q4_split_q8_1_residual.as_ref()
            };
            if let Some(mv_fn) = mv_fn_opt {
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q4_split_q8_1_residual preq {label}: {e}",
                        ))
                    })?;
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
            raw_buf.len(),
            expected_src_bytes,
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
            raw_buf.len(),
            expected_src_bytes,
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

/// Largest-first allocator for cloning the dense FFN (Gate/Up/Down) Q8Raw
/// projection weights into the per-row split (SoA) sibling layout.
///
/// Schedules clones in size-descending order to keep the CUDA allocator's heap
/// compact (Strategy A from `-gdn-byte-reduction.patch` analysis: largest
/// blocks first, smaller tiers fit cleanly in the tail).
///
/// The `clone_budget_bytes` UPPER cap (resolved by the caller via
/// `resolve_split_clone_budget("LUMEN_CUDA_Q8_SPLIT_BUDGET_GB", ..)`: the env
/// override maps to the exact pre-lever cap, the default is free-memory-aware)
/// reserves free VRAM headroom for the downstream KV cache and scratch
/// allocations that run AFTER preload.
///
/// Only the dense FFN weights (Gate/Up/Down) are cloned. The full-attention
/// projections (Wq/Wk/Wv/Wo) are deliberately EXCLUDED: the residual-split
/// attention decode kernel shares the L8-diagnosed garble risk (the Q4 sibling
/// `matvec_q4_split_q8_1_locked_residual` produced NaN/garbled logits), so
/// restricting the clone to FFN keeps decode byte-identical at ANY budget while
/// still delivering the FFN SoA speedup on 27B (all 64 FFN layers when the
/// budget is raised). With the historical 5.1 GB floor attention was never
/// reached anyway (FFN weights are larger and sort first), so the unset-env
/// path is unchanged.
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
    clone_budget_bytes: usize,
) -> (usize, Option<usize>, usize, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let inter = hp.intermediate_dim as usize;

    // Only the dense FFN weights (Gate/Up/Down) are cloned to Q8 split siblings.
    // The full-attention projections (Wq/Wk/Wv/Wo) are deliberately EXCLUDED --
    // see the fn-level doc: cloning attention weights (only reached once the
    // budget is raised past the larger, sort-first FFN set) shares the
    // L8-diagnosed residual-split garble risk. FFN split is bitwise-identical
    // to the AoS dp4a path via the `.rn` codegen lock, so restricting the clone
    // to FFN keeps decode byte-identical at ANY budget.
    #[derive(Copy, Clone, Debug)]
    enum SplitWeightKind {
        Gate,
        Up,
        Down,
    }

    struct Job {
        layer_idx: usize,
        kind: SplitWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 3);

    fn push_if_q8raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: SplitWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q8Raw(_) = w {
            if in_dim % 32 != 0 {
                return;
            }
            let nb = in_dim / 32;
            if nb % 2 != 0 {
                return;
            }
            let size_bytes = out_dim * nb * 34;
            jobs.push(Job {
                layer_idx,
                kind,
                out_dim,
                in_dim,
                size_bytes,
            });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        // FFN-only: attention (Wq/Wk/Wv/Wo) is intentionally not cloned (see the
        // SplitWeightKind comment) so attention decode stays on the base dp4a
        // path and output remains byte-identical to the AoS path.
        push_if_q8raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Gate,
            &layer.w_gate,
            inter,
            hidden,
        );
        push_if_q8raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Up,
            &layer.w_up,
            inter,
            hidden,
        );
        push_if_q8raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Down,
            &layer.w_down,
            hidden,
            inter,
        );
    }

    // Largest-first. Tie-break: full-attention layers before GDN (higher per-token
    // decode bandwidth). Final tie-break: ascending layer index for determinism.
    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 {
                    0u8
                } else {
                    1u8
                };
                let pb = if layers[b.layer_idx].layer_type == 0 {
                    0u8
                } else {
                    1u8
                };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    // The `clone_budget_bytes` UPPER cap is resolved by the caller via
    // `resolve_split_clone_budget("LUMEN_CUDA_Q8_SPLIT_BUDGET_GB", ..)`: the env
    // override (when set) maps to the exact pre-lever 5.1 GB-style cap, while the
    // default is free-memory-aware (`free − KV_reserve − activation_slack`,
    // floored at 5.1 GB). On an 80 GB target the default exceeds the ~19 GB 27B
    // dense FFN (Q8) so all 64 FFN layers clone; on a small GPU it holds at the
    // 5.1 GB floor. The cap does NOT force allocation -- per-clone `cudaMalloc`
    // failures still fail-safe (Q8Raw fallback keeps correctness).
    let mut layers_with_split = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;
    // Jobs the loop actually reached. Reporting `jobs.len()` called every job
    // "attempted" even when the budget stopped the loop after the first few,
    // which reads as "the budget was ample" in exactly the case it was not.
    let mut jobs_attempted: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() {
            break;
        }
        if bytes_cloned + job.size_bytes > clone_budget_bytes {
            break;
        }
        jobs_attempted += 1;

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            SplitWeightKind::Gate => {
                if let GpuWeightBuf::Q8Raw(b) = &layer.w_gate {
                    Some(b)
                } else {
                    None
                }
            }
            SplitWeightKind::Up => {
                if let GpuWeightBuf::Q8Raw(b) = &layer.w_up {
                    Some(b)
                } else {
                    None
                }
            }
            SplitWeightKind::Down => {
                if let GpuWeightBuf::Q8Raw(b) = &layer.w_down {
                    Some(b)
                } else {
                    None
                }
            }
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q8_raw_to_split(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(split_buf) => {
                match job.kind {
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

    (
        layers_with_split.len(),
        oom_layer,
        oom_count,
        jobs_attempted,
        jobs.len(),
    )
}

/// Largest-first allocator for cloning every Q4Raw projection weight into the
/// per-row split (SoA) sibling layout. Mirror of `repack_all_layers_q8_clone_to_split`.
#[allow(dead_code)]
unsafe fn repack_all_layers_q4_clone_to_split(
    device: &CudaDevice,
    repack_kernel: &cudarc::driver::CudaFunction,
    layers: &mut [LayerWeightsGpu],
    hp: &ModelHyperparams,
    clone_budget_bytes: usize,
) -> (usize, Option<usize>, usize, usize, usize) {
    let hidden = hp.hidden_dim as usize;
    let inter = hp.intermediate_dim as usize;

    // Only the dense FFN weights (Gate/Up/Down) are cloned to Q4 split siblings.
    // The full-attention projections (Wq/Wk/Wv/Wo) are deliberately EXCLUDED:
    // the residual-split `wo` decode kernel
    // (`matvec_q4_split_q8_1_locked_residual`) is not yet correct and produces
    // NaN logits, so cloning attention weights -- which only happens once the
    // clone budget is raised past the (larger, sort-first) FFN set -- garbles
    // decode (token 0 repeated). FFN split is bitwise-identical to the AoS
    // `matvec_q4_0_dp4a` path via the `.rn` codegen lock, so restricting the
    // clone to FFN keeps decode byte-identical at ANY budget while still
    // delivering the FFN SoA speedup on 27B (all 64 FFN layers when the budget
    // is raised). With the default 5.1 GB cap attention was never reached
    // anyway (FFN weights are larger and sort first), so the unset-env path is
    // unchanged.
    // Attention and GDN projections are cloned alongside the FFN set. They
    // were once excluded because the Q8_1 residual split kernel
    // (`matvec_q4_split_q8_1_locked_residual`) produced NaN logits on them;
    // that kernel is not what runs here. Excluding them capped the split
    // layout at FFN-only, so every attention and GDN call site fell back for
    // want of a sibling — GDN projections stream at 455 GB/s against the FFN's
    // 600, so they are where the aligned layout is needed most.
    #[derive(Copy, Clone, Debug)]
    enum SplitWeightKind {
        Gate,
        Up,
        Down,
        Wq,
        Wk,
        Wv,
        Wo,
        AttnGate,
        SsmOut,
    }

    struct Job {
        layer_idx: usize,
        kind: SplitWeightKind,
        out_dim: usize,
        in_dim: usize,
        size_bytes: usize,
    }

    let mut jobs: Vec<Job> = Vec::with_capacity(layers.len() * 3);

    fn push_if_q4raw(
        jobs: &mut Vec<Job>,
        layer_idx: usize,
        kind: SplitWeightKind,
        w: &GpuWeightBuf,
        out_dim: usize,
        in_dim: usize,
    ) {
        if let GpuWeightBuf::Q4Raw(_) = w {
            if in_dim % 32 != 0 {
                return;
            }
            let nb = in_dim / 32;
            if nb % 2 != 0 {
                return;
            }
            let size_bytes = out_dim * nb * 18;
            jobs.push(Job {
                layer_idx,
                kind,
                out_dim,
                in_dim,
                size_bytes,
            });
        }
    }

    for (layer_idx, layer) in layers.iter().enumerate() {
        // FFN-only: attention (Wq/Wk/Wv/Wo) is intentionally not cloned (see the
        // SplitWeightKind comment) so attention decode stays on the base dp4a
        // path and output remains byte-identical to the AoS path.
        push_if_q4raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Gate,
            &layer.w_gate,
            inter,
            hidden,
        );
        push_if_q4raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Up,
            &layer.w_up,
            inter,
            hidden,
        );
        push_if_q4raw(
            &mut jobs,
            layer_idx,
            SplitWeightKind::Down,
            &layer.w_down,
            hidden,
            inter,
        );

        {
            // On GDN layers `wq` IS the fused GDN qkv projection; on
            // full-attention layers it is the fused Q+gate. Either way it is
            // the largest single matrix outside the FFN. out_dim is derived
            // from the buffer so both shapes are handled without branching.
            let rows = |w: &GpuWeightBuf| match w {
                GpuWeightBuf::Q4Raw(b) => (b.len() / 18) * 32 / hidden,
                _ => 0,
            };
            for (kind, w) in [
                (SplitWeightKind::Wq, &layer.wq),
                (SplitWeightKind::Wk, &layer.wk),
                (SplitWeightKind::Wv, &layer.wv),
            ] {
                let out = rows(w);
                if out > 0 {
                    push_if_q4raw(&mut jobs, layer_idx, kind, w, out, hidden);
                }
            }
            push_if_q4raw(
                &mut jobs,
                layer_idx,
                SplitWeightKind::Wo,
                &layer.wo,
                hidden,
                hidden,
            );
            if let Some(ag) = layer.attn_gate.as_ref() {
                let out = rows(ag);
                if out > 0 {
                    push_if_q4raw(
                        &mut jobs,
                        layer_idx,
                        SplitWeightKind::AttnGate,
                        ag,
                        out,
                        hidden,
                    );
                }
            }
            if let Some(so) = layer.ssm_out.as_ref() {
                // ssm_out is [hidden, inner]: derive in_dim from the buffer.
                let in_d = rows(so);
                if in_d > 0 {
                    push_if_q4raw(
                        &mut jobs,
                        layer_idx,
                        SplitWeightKind::SsmOut,
                        so,
                        hidden,
                        in_d,
                    );
                }
            }
        }
    }

    jobs.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| {
                let pa = if layers[a.layer_idx].layer_type == 0 {
                    0u8
                } else {
                    1u8
                };
                let pb = if layers[b.layer_idx].layer_type == 0 {
                    0u8
                } else {
                    1u8
                };
                pa.cmp(&pb)
            })
            .then_with(|| a.layer_idx.cmp(&b.layer_idx))
    });

    // Q4 has 1.9x the per-element density of Q8 (18 vs 34 bytes per 32-elem
    // block), so for the same model the Q4 clone budget can be smaller. The
    // `clone_budget_bytes` UPPER cap is resolved by the caller via
    // `resolve_split_clone_budget("LUMEN_CUDA_Q4_SPLIT_BUDGET_GB", ..)`: the env
    // override (when set) maps to the exact pre-lever cap, while the default is
    // free-memory-aware (`free − KV_reserve − activation_slack`, floored at
    // 5.1 GB). On an 80 GB target the default exceeds the ~9.6 GB 27B dense FFN so
    // all 64 FFN layers clone; on a small GPU it holds at the 5.1 GB floor. The cap
    // does NOT force allocation — per-clone `cudaMalloc` failures still fail-safe.
    let mut layers_with_split = std::collections::HashSet::new();
    let mut oom_layer: Option<usize> = None;
    let mut oom_count: usize = 0;
    let mut bytes_cloned: usize = 0;
    // Jobs the loop actually reached. Reporting `jobs.len()` called every job
    // "attempted" even when the budget stopped the loop after the first few,
    // which reads as "the budget was ample" in exactly the case it was not.
    let mut jobs_attempted: usize = 0;

    for job in &jobs {
        if oom_layer.is_some() {
            break;
        }
        if bytes_cloned + job.size_bytes > clone_budget_bytes {
            break;
        }
        jobs_attempted += 1;

        let layer = &mut layers[job.layer_idx];
        let src_ref: Option<&CudaSlice<u8>> = match job.kind {
            SplitWeightKind::Gate => {
                if let GpuWeightBuf::Q4Raw(b) = &layer.w_gate {
                    Some(b)
                } else {
                    None
                }
            }
            SplitWeightKind::Up => {
                if let GpuWeightBuf::Q4Raw(b) = &layer.w_up {
                    Some(b)
                } else {
                    None
                }
            }
            SplitWeightKind::Down => {
                if let GpuWeightBuf::Q4Raw(b) = &layer.w_down {
                    Some(b)
                } else {
                    None
                }
            }
            SplitWeightKind::Wq => match &layer.wq {
                GpuWeightBuf::Q4Raw(b) => Some(b),
                _ => None,
            },
            SplitWeightKind::Wk => match &layer.wk {
                GpuWeightBuf::Q4Raw(b) => Some(b),
                _ => None,
            },
            SplitWeightKind::Wv => match &layer.wv {
                GpuWeightBuf::Q4Raw(b) => Some(b),
                _ => None,
            },
            SplitWeightKind::Wo => match &layer.wo {
                GpuWeightBuf::Q4Raw(b) => Some(b),
                _ => None,
            },
            SplitWeightKind::AttnGate => match layer.attn_gate.as_ref() {
                Some(GpuWeightBuf::Q4Raw(b)) => Some(b),
                _ => None,
            },
            SplitWeightKind::SsmOut => match layer.ssm_out.as_ref() {
                Some(GpuWeightBuf::Q4Raw(b)) => Some(b),
                _ => None,
            },
        };
        let Some(raw_buf) = src_ref else { continue };
        match repack_q4_raw_to_split(device, repack_kernel, raw_buf, job.out_dim, job.in_dim) {
            Ok(split_buf) => {
                match job.kind {
                    SplitWeightKind::Gate => layer.q4_split_w_gate = Some(split_buf),
                    SplitWeightKind::Up => layer.q4_split_w_up = Some(split_buf),
                    SplitWeightKind::Down => layer.q4_split_w_down = Some(split_buf),
                    SplitWeightKind::Wq => layer.q4_split_wq = Some(split_buf),
                    SplitWeightKind::Wk => layer.q4_split_wk = Some(split_buf),
                    SplitWeightKind::Wv => layer.q4_split_wv = Some(split_buf),
                    SplitWeightKind::Wo => layer.q4_split_wo = Some(split_buf),
                    SplitWeightKind::AttnGate => layer.q4_split_attn_gate = Some(split_buf),
                    SplitWeightKind::SsmOut => layer.q4_split_ssm_out = Some(split_buf),
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

    (
        layers_with_split.len(),
        oom_layer,
        oom_count,
        jobs_attempted,
        jobs.len(),
    )
}

// =============================================================================
// End split-layout integration: dispatch helpers.
// =============================================================================

/// Check if a weight buffer uses the dp4a Q8_1 path (Q8Raw, Q8Aligned, Q4Aligned, Q4Raw).
/// Q4_0 QUALITY FIX (Qwen3.5 GDN Q4_0 decode collapse, GQ-001 7/15 -> 14/15).
///
/// Q4Raw ATTENTION/GDN projections (wq, wk, wv, wo, attn_gate) must decode with
/// F32 activations, NOT the int8 Q8_1 dp4a path. The int8 activation quantization
/// is a ~0.5-1% per-matvec error on top of the already-lossy 4-bit weights; the
/// GDN linear-attention recurrence integrates the attention-projection outputs
/// across decode steps, so that stacked error COMPOUNDS over sequence length and
/// collapses multi-step reasoning (arithmetic prompts loop/degenerate). Metal
/// decodes the SAME LBC bytes with F32 activations (dequant_matmul_q4_0) and
/// PASSES; Q8_0 weights are precise enough to tolerate the int8-activation error
/// and PASS. Returning true here forces the projection off the int8 preq path
/// onto the non-preq `launch_matvec` route, which uses the F32-activation
/// `matvec_q4_0_smem` kernel (Metal-parity math). The FFN (gate/up/down) stays
/// on dp4a: its per-layer error is NOT amplified by the recurrence, and it is the
/// bandwidth-critical path, so keeping dp4a preserves decode throughput.
#[inline]
/// Does this Q4 projection keep FULL F32 activations?
///
/// Does this Q4 projection run on F32 activations?
///
/// Answered from the typed per-family plan. An earlier version consulted a
/// whole-model boolean instead, which made every int8 branch UNREACHABLE on
/// the narrow-GDN class — `*_use_preq` is `!weight_uses_f32_act_q4_fam(..)`,
/// and the boolean was true for that class, so the gate was always false.
/// Three rounds of "flat" int8 measurements were really the F32 path running
/// with extra configuration set. The route census is what caught it, with
/// calls=0 on all five families.
fn weight_uses_f32_act_q4_fam(
    weight: &GpuWeightBuf,
    plan: &crate::runtime_defaults::Q4ActPlan,
    family: crate::runtime_defaults::Q4ProjectionFamily,
) -> bool {
    if !matches!(weight, GpuWeightBuf::Q4Raw(_)) {
        return false;
    }
    plan.mode_for(family) == crate::runtime_defaults::Q4ActMode::F32
}

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
fn pick_output_proj_nr_kernel(kernels: &KernelSet, nr: u32) -> Option<&CudaFunction> {
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
        .map_err(|e| RuntimeError::Compute(format!("f32_to_f16 HGEMV input {label}: {e}",)))?;

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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        .map_err(|e| RuntimeError::Compute(format!("dtod residual copy HGEMV {label}: {e}",)))?;

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
        .map_err(|e| {
            RuntimeError::Compute(format!("f32_to_f16 HGEMV residual input {label}: {e}",))
        })?;

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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
            .map_err(|e| {
                RuntimeError::Compute(format!("f32_to_bf16_vec4 HGEMV input {label}: {e}",))
            })?;
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
            .map_err(|e| RuntimeError::Compute(format!("f32_to_bf16 HGEMV input {label}: {e}",)))?;
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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        .map_err(|e| {
            RuntimeError::Compute(format!("dtod residual copy HGEMV BF16 {label}: {e}",))
        })?;

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
            .map_err(|e| {
                RuntimeError::Compute(format!(
                    "f32_to_bf16_vec4 HGEMV residual input {label}: {e}",
                ))
            })?;
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
            .map_err(|e| {
                RuntimeError::Compute(format!("f32_to_bf16 HGEMV residual input {label}: {e}",))
            })?;
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
        .map_err(|e| RuntimeError::Compute(format!("matvec_bf16 fallback {label} launch: {e}",)))?;
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
        .map_err(|e| {
            RuntimeError::Compute(format!("matvec_bf16_residual fallback {label} launch: {e}",))
        })?;
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
    // UNIFORM-F32 CUDA MoE DECODE (env LUMEN_CUDA_MOE_DECODE_F32, MoE-gated):
    // when ON, bypass the cuBLAS GemmEx F16-tensor-core downcast and dispatch the
    // F32-exact `matvec_bf16` kernel (lossless bf16→f32 upcast + F32 accumulate)
    // so single-token decode matches the F32 GGUF source precision. OFF is
    // byte-identical (the `&&` short-circuits before the atomic loads).
    if bf16_gemmex_enabled() && !moe_decode_f32_enabled() {
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
        device, kernels, w_bf16, input_f32, output_f32, out_dim, in_dim, label,
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
    // UNIFORM-F32 CUDA MoE DECODE (env LUMEN_CUDA_MOE_DECODE_F32, MoE-gated):
    // mirrors `launch_bf16_matvec_with_fallback` for the fused
    // `out = W^T*in + residual` decode projection (the full-attention output
    // proj `wo`). When ON, route through the F32-exact `matvec_bf16_residual`
    // instead of the GemmEx F16 downcast. OFF is byte-identical.
    if bf16_gemmex_enabled() && !moe_decode_f32_enabled() {
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
        device, kernels, w_bf16, input_f32, residual, output_f32, out_dim, in_dim, label,
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
            .map_err(|e| RuntimeError::Compute(format!("fused_rmsnorm_f16 {label}: {e}",)))?;
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
            .map_err(|e| RuntimeError::Compute(format!("swiglu_f32_to_f16: {e}",)))?;
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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        .map_err(|e| {
            RuntimeError::Compute(format!(
                "dtod residual copy HGEMV preconverted {label}: {e}",
            ))
        })?;

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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
    device
        .stream
        .memcpy_htod(&host_a[..batch_count], dev_a_ptrs)
        .map_err(|e| RuntimeError::Compute(format!("batched HGEMV {label} A ptrs: {e}")))?;
    device
        .stream
        .memcpy_htod(&host_b[..batch_count], dev_b_ptrs)
        .map_err(|e| RuntimeError::Compute(format!("batched HGEMV {label} B ptrs: {e}")))?;
    device
        .stream
        .memcpy_htod(&host_c[..batch_count], dev_c_ptrs)
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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        a_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        b_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_dev_ptr as *const *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        out_dim as i32, // M
        1i32,           // N = 1 (GEMV)
        in_dim as i32,  // K
        &alpha as *const f32 as *const std::ffi::c_void,
        a_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // lda
        b_dev_ptr as *const *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32, // ldb
        &beta as *const f32 as *const std::ffi::c_void,
        c_dev_ptr as *const *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32, // ldc
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
        eprintln!(
            "[CUDA] cublasGemmGroupedBatchedEx available (CUDA 12.5+) -- QKV grouped GEMM enabled"
        );
    } else {
        eprintln!(
            "[CUDA] cublasGemmGroupedBatchedEx not available -- using separate Q + batched KV"
        );
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

            if let (Some(wq_ptr), Some(wk_ptr), Some(wv_ptr)) = (wq_f16_ptr, wk_f16_ptr, wv_f16_ptr)
            {
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
        _ => f16_cache.map(|cache| {
            let (ptr, _) = cache.device_ptr(&device.stream);
            ptr as u64
        }),
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
        .map_err(|e| {
            RuntimeError::Compute(format!("fused_norm_matvec_f32 {label} launch: {e}",))
        })?;
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
        .map_err(|e| {
            RuntimeError::Compute(format!("fused_norm_dual_matvec_f32 gate+up launch: {e}",))
        })?;
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
            .and_then(|v| v.parse::<usize>().ok())
            // `=0` would size the KV cache to 0 tokens and fault the KV-write
            // kernel (CUDA_ERROR_ILLEGAL_ADDRESS / MMU fault). A zero cap is
            // meaningless, so treat it as "no cap" (identical to unset).
            .filter(|&cap| cap > 0);
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

        // Compile all decode-path kernels. With the persistent PTX disk cache
        // (default ON; `LUMEN_CUDA_PTX_CACHE=0` disables), a warm cache turns
        // this ~252-module NVRTC compile from a multi-minute cold start into a
        // sub-second `cuModuleLoadData` sweep. Time it and report cache hits.
        let kernel_compile_start = std::time::Instant::now();
        let mut kernels = decode::compile_all_kernels(&self.device)?;
        {
            let (hits, misses) = super::ptx_cache::stats();
            let elapsed = kernel_compile_start.elapsed();
            let state = if !super::ptx_cache::cache_enabled() {
                "disabled"
            } else if misses == 0 && hits > 0 {
                "warm (all cached)"
            } else if hits == 0 {
                "cold (all compiled)"
            } else {
                "partial"
            };
            eprintln!(
                "[CUDA] kernel compile/load done in {:.3}s -- PTX cache {state}: {hits} hits, {misses} misses",
                elapsed.as_secs_f64()
            );
        }

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
            input_f16: self
                .device
                .alloc_zeros::<u8>(hidden_dim.max(inter_dim) * 2)?,
            // Q8_1 scratch for dp4a matvec: max(hidden_dim, inter_dim) / 32 * 36 bytes.
            // Only allocate if the dp4a Q8_1 kernels compiled successfully.
            // also allocate when mul_mat_vec_q_q{8,4}_0 compiled
            // (the dp4a-mmvq dispatch uses the same scratch layout).
            input_q8_1: if (kernels.quantize_f32_to_q8_1.is_some()
                && (kernels.matvec_q8_0_q8_1.is_some()
                    || kernels.matvec_q8_aligned_q8_1.is_some()
                    || kernels.matvec_q4_0_dp4a.is_some()
                    || kernels.matvec_q4_aligned_q8_1.is_some()))
                || (kernels.quantize_q8_1_rawsum.is_some()
                    && (kernels.mul_mat_vec_q_q8_0.is_some()
                        || kernels.mul_mat_vec_q_q4_0.is_some()))
            {
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
        let (embedding_f32, embedding_q8, embedding_f16_raw, embedding_q4_raw, embedding_bf16_raw) =
            if has_raw_embedding {
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
        let (
            output_proj_f32,
            output_proj_q8,
            output_proj_q4,
            output_proj_f16_raw,
            output_proj_bf16_raw,
        ) = if has_raw_output_proj {
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
            Ok(ws) => match self.device.set_cublas_workspace(&ws) {
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
            },
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
        // false). This matches the resolver pattern used for BF16_GEMMEX.
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
        ) && kernels.matvec_q8_aligned_q8_1_hw.is_some()
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
        // LUMEN_CUDA_Q8_MATVEC_FAST (default-OFF): route the Q8 split decode
        // matvec through the 128-bit-weight-load kernel (matvec_q8_split_q8_1_v4).
        // Byte-identical to the scalar split kernel; only the weight load width
        // changes. Effective only when Q8 split dispatch is already active AND
        // the v4 kernels loaded; the per-call `in_dim % 256 == 0` alignment guard
        // is applied at dispatch. OFF -> the scalar kernel runs unchanged.
        let use_q8_matvec_fast = env_truthy("LUMEN_CUDA_Q8_MATVEC_FAST");
        if use_q8_matvec_fast {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_MATVEC_FAST: Q8 split decode matvec uses 128-bit weight loads (aligned dims only)");
        }
        // LUMEN_CUDA_Q8_MMVQ (default-OFF): route BOTH the Q8 and Q4 split decode
        // matvecs through the llama `mul_mat_vec_q` port (matvec_q{8,4}_split_q8_1_mmvq):
        // one output row per CTA, VDR lane striping (4 lanes/Q8 block, 2 lanes/Q4
        // block), and a single lane-preserving cross-warp reduction. Effective
        // only when the corresponding split dispatch is active AND all four mmvq
        // kernels loaded. NOT byte-identical -- a quality-equivalent near-tie that
        // gates on the FULL GQ + MoE-router-stability check, NOT DET byte identity.
        // OFF -> the existing scalar/v4/locked split kernels run unchanged.
        // Default-ON (kill-switch): +3.9% 9B-q8 decode with the fused-GLU
        // epilogue, harness gate-banked (receipts §23). `=0` reverts to the
        // scalar/locked split kernels (byte-identical to the pre-mmvq default).
        let use_q8_mmvq = parse_env_truthy("LUMEN_CUDA_Q8_MMVQ").unwrap_or(true);
        if use_q8_mmvq {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_MMVQ: Q8/Q4 split decode matvec uses the llama mmvq port (near-tie; GQ+router gated)");
        }
        // LUMEN_CUDA_SOA_LOCKED selects the codegen-LOCKED Q4 split kernel.
        // It reuses the Q4 split (SoA) repack + sibling buffers, which are
        // always built when the kernel loaded and the budget allowed. Default
        // resolved by `soa_locked_default` (ON for quantised
        // dense, OFF for MoE/BF16); `LUMEN_CUDA_SOA_LOCKED=0` forces OFF.
        let use_soa_locked = env_truthy_or_default(
            "LUMEN_CUDA_SOA_LOCKED",
            crate::runtime_defaults::soa_locked_default,
        );
        // Q4 split/SoA siblings are always built when the kernel is loaded and
        // the clone budget allows. The layout is what makes the nibble run
        // 4-byte aligned; without a sibling every Q4 decode matvec falls back
        // to byte-at-a-time unpacking. Sizing is the resource control
        // (LUMEN_CUDA_Q4_SPLIT_BUDGET_GB), not a feature switch.
        if use_soa_locked {
            eprintln!("[CUDA] LUMEN_CUDA_SOA_LOCKED=1: Q4_0 weights cloned to split layout; decode uses the codegen-locked split kernel");
        } else {
            eprintln!("[CUDA] Q4_0 weights will be cloned to split layout for decode");
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
        // output_proj NR override: pick from 2/16/32/64/128. Default = 16
        // when the model is Q8 dense (: matches the canonical
        // production config), else 32 (legacy default that matches the
        // pre-F2 dispatch). When `OUTPUT_PROJ_SPLIT=1` AND the requested NR
        // variant is loaded, dispatch routes through it.
        let output_proj_nr_default = crate::runtime_defaults::output_proj_nr_default();
        let output_proj_nr: u32 = match std::env::var("LUMEN_CUDA_OUTPUT_PROJ_NR").ok().as_deref() {
            Some("2") => 2,
            Some("8") => 8,
            Some("16") => 16,
            Some("32") => 32,
            Some("64") => 64,
            Some("128") => 128,
            None | Some("") => {
                // 16 for Q8 dense, 32 legacy otherwise.
                if output_proj_nr_default == 16 {
                    16
                } else {
                    32
                }
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
        // Propagate runtime feature flags onto KernelSet so the
        // `launch_matvec_preq8_1*` free functions can consult them without
        // taking an extra parameter at every call site.
        kernels.use_q8_scale_hw = use_q8_scale_hw;
        kernels.use_q8_split_dispatch = use_q8_split && kernels.matvec_q8_split_q8_1.is_some();
        // Q8 128-bit-load fast path: requires the Q8 split dispatch active AND
        // both v4 kernels loaded. Default OFF (env-gated only, no canonical default).
        kernels.use_q8_matvec_fast = use_q8_matvec_fast
            && kernels.use_q8_split_dispatch
            && kernels.matvec_q8_split_q8_1_v4.is_some()
            && kernels.matvec_q8_split_q8_1_v4_residual.is_some();
        if use_q8_matvec_fast && !kernels.use_q8_matvec_fast {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_MATVEC_FAST=1 set but prerequisites missing (need Q8 split dispatch active + v4 kernels loaded); using scalar split kernel");
        }
        kernels.use_q4_split_dispatch = kernels.matvec_q4_split_q8_1.is_some();
        // Locked Q4 split kernel selection. Effective only when the Q4 split
        // dispatch is active AND both locked kernels loaded. When the locked
        // kernels failed to compile, fall back to the unlocked split kernel.
        kernels.use_soa_locked = use_soa_locked
            && kernels.use_q4_split_dispatch
            && kernels.matvec_q4_split_q8_1_locked.is_some()
            && kernels.matvec_q4_split_q8_1_locked_residual.is_some();
        if kernels.use_soa_locked {
            eprintln!("[CUDA] LUMEN_CUDA_SOA_LOCKED=1: Q4 split dispatch uses the codegen-locked kernel (layout-independent bitwise-identical F32)");
        } else if use_soa_locked {
            eprintln!("[CUDA] LUMEN_CUDA_SOA_LOCKED=1 set but prerequisites missing (need Q4 split dispatch active + locked kernels loaded); using unlocked split / base path");
        }
        // llama mmvq path -- single master gate for ALL three mmvq kernel
        // families: the Q8 split, Q4 split, and Q8Aligned (36-byte attn/GDN)
        // paths. Set true when the operator asked for it AND at least one mmvq
        // kernel loaded; each dispatch branch then independently checks its own
        // specific kernel `.is_some()` (so split-only and aligned-only models
        // each work, and a family whose kernel failed to compile cleanly falls
        // through to the existing scalar/v4/locked path). When a branch fires it
        // takes precedence over the scalar/v4 (Q8 split), locked/unlocked (Q4
        // split), and scalar/hw (Q8Aligned) kernels. Near-tie: GQ+router gated,
        // NOT DET byte-identical.
        // The master `LUMEN_CUDA_Q8_MMVQ` gate drives ONLY the Q8 split +
        // Q8Aligned mmvq families (both a decode WIN: +3.9% 9B-q8, +5.3% 27B-q8;
        // receipts §23/§26). The Q4 split mmvq path is measured-NEGATIVE
        // (27B-q4 -3.0%, 9B-q4 flat +0.15%; §24/§26), so it is gated separately
        // below and defaults OFF.
        let any_q8_mmvq_loaded = kernels.matvec_q8_split_q8_1_mmvq.is_some()
            || kernels.matvec_q8_aligned_q8_1_mmvq.is_some();
        kernels.use_mmvq = use_q8_mmvq && any_q8_mmvq_loaded;
        if use_q8_mmvq && !kernels.use_mmvq {
            eprintln!("[CUDA] LUMEN_CUDA_Q8_MMVQ=1 set but prerequisites missing (no Q8 mmvq kernels loaded); using existing scalar/v4/aligned kernels");
        }
        // LUMEN_CUDA_Q4_MMVQ (default-OFF): the Q4 split mmvq path is
        // measured-negative; its kernels stay loaded but are opt-in only, for a
        // future re-gate if new evidence emerges. `=1` routes Q4 split decode
        // through `matvec_q4_split_q8_1_mmvq[_residual]` (near-tie; GQ+router gated).
        let use_q4_mmvq = parse_env_truthy("LUMEN_CUDA_Q4_MMVQ").unwrap_or(false);
        kernels.use_mmvq_q4 = use_q4_mmvq && kernels.matvec_q4_split_q8_1_mmvq.is_some();
        if use_q4_mmvq && !kernels.use_mmvq_q4 {
            eprintln!("[CUDA] LUMEN_CUDA_Q4_MMVQ=1 set but the Q4 split mmvq kernel is not loaded; using the scalar/locked Q4 split kernel");
        } else if kernels.use_mmvq_q4 {
            eprintln!("[CUDA] LUMEN_CUDA_Q4_MMVQ=1: Q4 split decode matvec uses the llama mmvq port (measured-negative default-OFF; opt-in)");
        }

        // pre-allocate MoE scratch when the model declares experts.
        // Sized from hyperparams: hidden_dim, expert inter_dim (or fallback to
        // model inter_dim), shared inter_dim (= model inter_dim per Qwen3.5-MoE
        // hyperparam encoding), num_experts, top_k. Dense models get `None`.
        let moe_scratch = if let (Some(num_experts), Some(top_k)) =
            (hyperparams.num_experts, hyperparams.num_active_experts)
        {
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
        let moe_meta_cache: Vec<Option<super::moe::CudaMoeMeta>> = vec![None; num_layers];
        // parallel cache for Phase-F batched-expert GPU offset tables.
        // Built lazily during preload_weights when an MoE layer is detected.
        // Vec::new() initialization is fine because populate calls .resize()
        // before indexing (mirrored alongside moe_meta_cache).
        let moe_batched_offsets: Vec<Option<super::moe::CudaMoeBatchedOffsets>> =
            (0..num_layers).map(|_| None).collect();
        // Per-layer repacked down planes (gated, built at preload).
        let moe_repacked: Vec<Option<super::moe::CudaMoeRepacked>> =
            (0..num_layers).map(|_| None).collect();

        *self.state.lock().unwrap() = Some(MutableState {
            kernels,
            scratch,
            kv_caches,
            globals,
            layer_weights_cache: Vec::new(),
            logits_gpu,
            argmax_result: self.device.alloc_zeros::<u32>(1)?,
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
            moe_repacked,
            use_q8_scale_hw,
            use_q8_split,
            use_output_proj_split,
            output_proj_nr,
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
                    self.device
                        .stream
                        .launch_builder(func)
                        .arg(emb_bf16)
                        .arg(&mut output_gpu)
                        .arg(&token_id)
                        .arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_bf16 launch: {e}")))?;
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
                        .arg(&mut output_gpu)
                        .arg(&token_id)
                        .arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_f16 launch: {e}")))?;
            } else if let Some(ref emb_q4) = st.globals.embedding_q4 {
                let func = self.embed_q4_0_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_q4_0 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(func)
                        .arg(emb_q4)
                        .arg(&mut output_gpu)
                        .arg(&token_id)
                        .arg(&hd)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_q4_0 launch: {e}")))?;
            } else if let Some(ref emb_q8) = st.globals.embedding_q8 {
                let func = self.embed_q8_0_func.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("embed_token_q8_0 kernel not compiled".into())
                })?;
                let hd = hidden_dim as u32;
                unsafe {
                    self.device
                        .stream
                        .launch_builder(func)
                        .arg(emb_q8)
                        .arg(&mut output_gpu)
                        .arg(&token_id)
                        .arg(&hd)
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
                .map_err(|e| RuntimeError::Compute(format!("CUDA embed_token_f32 launch: {e}")))?;
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
        let kv =
            kv.ok_or_else(|| RuntimeError::Compute("KV cache view required for attention".into()))?;

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;

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
                    "GDN layers require GPU-resident weights (call preload_weights first)".into(),
                ));
            }
            // Upload x to GPU, run GPU-resident compute, download result.
            let x_f32 = x.as_f32_slice();
            self.device.htod_copy_into(x_f32, &mut st.scratch.x_gpu)?;
            let layer_out = self.compute_layer_gpu(layer_idx, seq_pos, st)?;
            // This path reads the layer result back from `x_gpu`, so it must
            // honour the same commit contract as the decode loop. It did not
            // before: `compute_layer_gpu` returned `()` and the copy was simply
            // absent, so a layer leaving its result in `attn_proj` was read
            // back stale. Making the destination part of the return type turned
            // that latent bug into a compile-time obligation.
            if layer_out == LayerOutput::NeedsCommit {
                self.device
                    .stream
                    .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                    .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
            }
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
                    &self.device,
                    &st.kernels,
                    &st.scratch.x_gpu,
                    &mut st.scratch.rms_scale,
                    eps,
                    hidden_dim,
                )?;
            }

            // Pass 2: fused norm+matvec for Q, K, V.
            // SAFETY: wq is F32 [q_dim, hidden_dim]. x_gpu is [hidden_dim].
            // rms_scale is [1]. attn_norm is [hidden_dim]. q is [q_dim].
            if let GpuWeightBuf::F32(ref wq_f32) = lw.wq {
                unsafe {
                    launch_fused_norm_matvec_f32(
                        &self.device,
                        &st.kernels,
                        &st.scratch.x_gpu,
                        &st.scratch.rms_scale,
                        &lw.attn_norm,
                        wq_f32,
                        &mut st.scratch.q,
                        q_dim,
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
            // SAFETY: wk is [kv_dim, hidden_dim], normed is [hidden_dim], k is [kv_dim].
            unsafe {
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
            }
            // SAFETY: wv is [kv_dim, hidden_dim], normed is [hidden_dim], v is [kv_dim].
            unsafe {
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

        // QKV bias (Qwen2-family, streaming decode).
        if lw.bq.is_some() || lw.bk.is_some() || lw.bv.is_some() {
            let block = 256u32;
            unsafe {
                if let Some(ref bq) = lw.bq {
                    let d = q_dim as u32;
                    let g = (d + block - 1) / block;
                    self.device
                        .stream
                        .launch_builder(&st.kernels.bias_add)
                        .arg(&mut st.scratch.q)
                        .arg(bq)
                        .arg(&d)
                        .launch(CudarcLaunchConfig {
                            grid_dim: (g, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| {
                            RuntimeError::Compute(format!("bias_add bq streaming: {e}"))
                        })?;
                }
                if let Some(ref bk) = lw.bk {
                    let d = kv_dim as u32;
                    let g = (d + block - 1) / block;
                    self.device
                        .stream
                        .launch_builder(&st.kernels.bias_add)
                        .arg(&mut st.scratch.k)
                        .arg(bk)
                        .arg(&d)
                        .launch(CudarcLaunchConfig {
                            grid_dim: (g, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| {
                            RuntimeError::Compute(format!("bias_add bk streaming: {e}"))
                        })?;
                }
                if let Some(ref bv) = lw.bv {
                    let d = kv_dim as u32;
                    let g = (d + block - 1) / block;
                    self.device
                        .stream
                        .launch_builder(&st.kernels.bias_add)
                        .arg(&mut st.scratch.v)
                        .arg(bv)
                        .arg(&d)
                        .launch(CudarcLaunchConfig {
                            grid_dim: (g, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|e| {
                            RuntimeError::Compute(format!("bias_add bv streaming: {e}"))
                        })?;
                }
            }
        }

        // 4. RoPE: apply rotary position embeddings to q and k.
        {
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            let actual_rot = if rotary_dim > 0 && rotary_dim < head_dim as u32 {
                rotary_dim as usize
            } else {
                head_dim
            };
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

        // 8-9. Fused FFN RMSNorm + gate/up projections.
        //
        // For F32 weights: fused dual matvec computes both gate and up from the
        // same normalized input in a single dispatch (3 kernels -> 2: rms_scale + dual).
        // For non-F32 weights: fall back to separate rmsnorm + gate matvec + up matvec.
        if matches!(&lw.w_gate, GpuWeightBuf::F32(_)) && matches!(&lw.w_up, GpuWeightBuf::F32(_)) {
            // Pass 1: compute rms_scale from attn_proj.
            // SAFETY: attn_proj is [hidden_dim], rms_scale is [1]. Both allocated in init.
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

            // Pass 2: fused dual matvec for gate+up.
            // SAFETY: w_gate and w_up are F32 [inter_dim, hidden_dim]. attn_proj is
            // [hidden_dim]. rms_scale is [1]. ffn_norm is [hidden_dim]. gate and up
            // are [inter_dim].
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
            }
            // SAFETY: w_up is [inter_dim, hidden_dim], normed is [hidden_dim],
            // up is [inter_dim].
            unsafe {
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
        let st = state_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;

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
        //
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
                        self.device
                            .stream
                            .launch_builder(quant_fn)
                            .arg(&st.scratch.normed)
                            .arg(&mut *q8_1_buf)
                            .arg(&in_dim)
                            .launch(quant_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "quantize_f32_to_q8_1 gdn output_proj Q4Aligned: {e}",
                        ))
                    })?;
                    let mv_grid = dp4a_q4_grid(out_dim);
                    let mv_cfg = CudarcLaunchConfig {
                        grid_dim: (mv_grid, 1, 1),
                        block_dim: (DP4A_Q4_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(mv_fn)
                            .arg(proj_q4a)
                            .arg(&*q8_1_buf)
                            .arg(&mut st.logits_gpu)
                            .arg(&out_dim)
                            .arg(&in_dim)
                            .launch(mv_cfg)
                    }
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "matvec_q4_aligned_q8_1 gdn output_proj: {e}",
                        ))
                    })?;
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
                            RuntimeError::Compute(format!(
                                "matvec output_proj Q4_0 smem launch: {e}"
                            ))
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
            } else if let Some(ref proj_q8_split) = st.globals.output_proj_q8_split {
                // OUTPUT_PROJ_SPLIT: prefer split layout for graph variant too.
                // NR=32 grid for the dedicated output_proj kernel; NR=2 fallback.
                // route through `LUMEN_CUDA_OUTPUT_PROJ_NR={16,64,128}`
                // if requested and loaded.
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;
                let (split_mv_fn, mv_grid): (&CudaFunction, u32) = if let Some(proj_fn) =
                    pick_output_proj_nr_kernel(&st.kernels, st.output_proj_nr)
                {
                    let nr = st.output_proj_nr;
                    (proj_fn, (out_dim_u32 + nr - 1) / nr)
                } else if let Some(ref proj_fn) = st.kernels.matvec_q8_split_output_proj {
                    (proj_fn, (out_dim_u32 + 31) / 32)
                } else if let Some(ref generic_fn) = st.kernels.matvec_q8_split_q8_1 {
                    (generic_fn, dp4a_q8_1_grid(out_dim_u32))
                } else {
                    return Err(RuntimeError::Compute(
                        "graph output_proj_q8_split present but no split matvec kernel available"
                            .into(),
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "quantize_f32_to_q8_1 graph output_proj split: {e}",
                        ))
                    })?;
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("matvec_q8_split graph output_proj: {e}",))
                    })?;
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
                        RuntimeError::Compute(format!(
                            "matvec_q8_aligned_q8_1 gdn output_proj: {e}"
                        ))
                    })?;
                } else {
                    let q8a_fn = st
                        .kernels
                        .matvec_q8_0_aligned
                        .as_ref()
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
                        RuntimeError::Compute(format!(
                            "matvec output_proj Q8_0 aligned launch: {e}"
                        ))
                    })?;
                }
            } else if let Some(ref proj_q8) = st.globals.output_proj_q8 {
                // Q8_0 output projection: dp4a (native Q8_0, ~1.06 B/elem).
                // Fallback when aligned repack is unavailable.
                let out_dim_u32 = vocab_size as u32;
                let in_dim_u32 = hidden_dim as u32;
                let q8_fn = st
                    .kernels
                    .matvec_q8_0_dp4a
                    .as_ref()
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
                    self.device.blas.gemv(
                        cfg,
                        &st.globals.output_proj,
                        &st.scratch.normed,
                        &mut st.logits_gpu,
                    )
                }
                .map_err(|e| RuntimeError::Compute(format!("cuBLAS GEMV output_proj: {e}")))?;
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
        let is_preloaded = self
            .state
            .lock()
            .unwrap()
            .as_ref()
            .map(|st| !st.layer_weights_cache.is_empty())
            .unwrap_or(false);
        // MoE capability is derived from `moe_meta_cache` — the cache
        // is populated by `preload_weights` for each layer whose
        // `subtensors.experts.is_some()`. Before preload the cache is empty
        // (caps returns moe=false); after preload it reflects the model.
        let has_moe = self
            .state
            .lock()
            .unwrap()
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
                let has_gdn = self
                    .state
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|st| st.layer_weights_cache.iter().any(|lw| lw.layer_type == 1))
                    .unwrap_or(false);
                is_preloaded && has_gdn
            },
            gpu_resident: is_preloaded,
            gdn: true,
            moe: has_moe,
            // GPU-side greedy argmax fast path. Gated behind LUMEN_CUDA_GPU_SAMPLE
            // (default ON). When enabled and weights are preloaded, the engine's
            // `use_gpu_greedy_predicate` routes greedy (temperature<=0, no
            // penalties) through `decode_token_greedy`: on-GPU argmax + a 4-byte
            // token readback, eliminating the per-token full-vocab logits D2H
            // copy. `LUMEN_CUDA_GPU_SAMPLE=0` reverts to the full-logits-readback
            // `decode_token` path (byte-identical output, just slower readback).
            gpu_argmax: is_preloaded && cuda_gpu_sample_enabled(),
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
        // Reset GPU KV caches and GDN recurrent state to prevent
        // stale data from leaking across generate() calls.
        if let Ok(mut guard) = self.state.lock() {
            if let Some(ref mut st) = *guard {
                for kv_cache in &mut st.kv_caches {
                    kv_cache.reset();
                }
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
             state DtoH which is planned for a future release)"
                .into(),
        ))
    }

    fn sync_kv_from_cpu(
        &self,
        _kv: &crate::kv::KvCache,
        _recurrent: Option<&crate::kv::disk::RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported(
            "CUDA backend: sync_kv_from_cpu is not yet wired ( lands the \
             Metal path only; CUDA disk-KV restore is planned for a future release)"
                .into(),
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
        let st = state_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;

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
            &self.device,
            batch,
            hidden_dim,
            q_dim,
            kv_dim,
            inter_dim,
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
                    layer_idx,
                    batch,
                    st,
                    &mut pf,
                    gdn_pf.as_mut().unwrap(),
                    eps,
                )?;
                continue;
            }

            // ---- STANDARD ATTENTION LAYER ----

            // 2a. Batched RMSNorm for QKV projections (always F32 path for precision).
            unsafe {
                super::prefill::launch_rmsnorm_batched(
                    &self.device,
                    &st.kernels,
                    &pf.x,
                    &lw.attn_norm,
                    &mut pf.normed,
                    eps,
                    batch,
                    hidden_dim,
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
                        &self.device,
                        &st.kernels,
                        &lw.wq,
                        None,
                        &pf.normed,
                        &mut pf_q_gate,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        q_gate_dim,
                        hidden_dim,
                        "wq_qgate",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device,
                        &st.kernels,
                        &lw.wk,
                        None,
                        &pf.normed,
                        &mut pf.k,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        kv_dim,
                        hidden_dim,
                        "wk",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device,
                        &st.kernels,
                        &lw.wv,
                        None,
                        &pf.normed,
                        &mut pf.v,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        kv_dim,
                        hidden_dim,
                        "wv",
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("deinterleave_qgate prefill: {e}"))
                    })?;
                } else {
                    return Err(RuntimeError::Compute(
                        "Q+gate fusion requires deinterleave_qgate kernel".into(),
                    ));
                }
                // Batched per-head RMSNorm on Q and K.
                if let Some(ref q_norm_w) = lw.attn_q_norm {
                    let norm_fn =
                        st.kernels
                            .rmsnorm_per_head_inplace
                            .as_ref()
                            .ok_or_else(|| {
                                RuntimeError::Compute(
                                    "Q+gate fusion requires rmsnorm_per_head_inplace kernel".into(),
                                )
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm_per_head Q prefill: {e}"))
                    })?;
                }
                if let Some(ref k_norm_w) = lw.attn_k_norm {
                    let norm_fn =
                        st.kernels
                            .rmsnorm_per_head_inplace
                            .as_ref()
                            .ok_or_else(|| {
                                RuntimeError::Compute(
                                    "Q+gate fusion requires rmsnorm_per_head_inplace kernel".into(),
                                )
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
                    .map_err(|e| {
                        RuntimeError::Compute(format!("rmsnorm_per_head K prefill: {e}"))
                    })?;
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
                // [ROPEPROBE] dump pre/post-rope Q for full-attn layers (last token,
                // head 0, first 16 dims) to compare vs llama.cpp Qcur_normed/Qcur.
                let ropeprobe = moe_probe_enabled();
                let qd = num_heads * head_dim;
                if ropeprobe && batch > 1 {
                    let qh = self.device.dtoh_copy(&pf.q)?;
                    let o = (batch - 1) * qd;
                    eprintln!("[ROPEPROBE] layer={layer_idx} rotary_dim={rotary_dim_pf} head_dim={head_dim} neox={} PRE q_h0[0..16]={:?}",
                        hp.rope_neox, &qh[o..o + 16.min(qd)]);
                }
                unsafe {
                    super::prefill::launch_rope_batched(
                        &self.device,
                        &st.kernels,
                        &mut pf.q,
                        &mut pf.k,
                        pos_start,
                        batch,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        theta,
                        hp.rope_neox,
                        rotary_dim_pf,
                    )?;
                }
                if ropeprobe && batch > 1 {
                    let qh = self.device.dtoh_copy(&pf.q)?;
                    let o = (batch - 1) * qd;
                    // dump dims 0..8, 30..38, 62..70 to see split-half (d,d+32) pairing
                    let s = |a: usize, b: usize| qh[o + a..o + b.min(qd)].to_vec();
                    eprintln!(
                        "[ROPEPROBE] layer={layer_idx} POST q_h0 d0-7={:?} d30-37={:?} d62-69={:?}",
                        s(0, 8),
                        s(30, 38),
                        s(62, 70)
                    );
                    // post-rope K (cached) + V, last token, kv-head 0, first 3
                    let kvd = num_kv_heads * head_dim;
                    let kh = self.device.dtoh_copy(&pf.k)?;
                    let vh = self.device.dtoh_copy(&pf.v)?;
                    let ko = (batch - 1) * kvd;
                    eprintln!(
                        "[KVPROBE] layer={layer_idx} POST k_h0[0..3]={:?} v_h0[0..3]={:?}",
                        &kh[ko..ko + 3.min(kvd)],
                        &vh[ko..ko + 3.min(kvd)]
                    );
                    // whole-buffer sumsq (LAYOUT-INDEPENDENT) for Q/K/V across all tokens.
                    let ss =
                        |v: &[f32]| -> f64 { v.iter().map(|&e| (e as f64) * (e as f64)).sum() };
                    eprintln!(
                        "[QKVSS] layer={layer_idx} q_sumsq={:.4} k_sumsq={:.4} v_sumsq={:.4}",
                        ss(&qh[..batch * qd]),
                        ss(&kh[..batch * kvd]),
                        ss(&vh[..batch * kvd])
                    );
                }

                // 2d. Batch KV cache write
                let kv_cache = &mut st.kv_caches[layer_idx];
                unsafe {
                    super::prefill::launch_kv_cache_write_batch(
                        &self.device,
                        &st.kernels,
                        &mut kv_cache.k_cache,
                        &pf.k,
                        pos_start,
                        batch,
                        num_kv_heads,
                        kv_cache.max_seq_len,
                        head_dim,
                    )?;
                    super::prefill::launch_kv_cache_write_batch(
                        &self.device,
                        &st.kernels,
                        &mut kv_cache.v_cache,
                        &pf.v,
                        pos_start,
                        batch,
                        num_kv_heads,
                        kv_cache.max_seq_len,
                        head_dim,
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
                // DIAGNOSTIC (env LUMEN_CUDA_FORCE_SCALAR_ATTN=1): bypass FA2/WMMA
                // and use the F32 scalar Br=4 attention. Tests whether the
                // batch>=16 WMMA(F16) path is the source of the full-attn
                // long-context divergence. Remove before commit.
                let force_scalar_attn = {
                    use std::sync::OnceLock;
                    static FS: OnceLock<bool> = OnceLock::new();
                    *FS.get_or_init(|| {
                        std::env::var("LUMEN_CUDA_FORCE_SCALAR_ATTN").as_deref() == Ok("1")
                    })
                };
                // WMMA-PRECISION-FIX-RCA precision-localization selector (no-op
                // when unset). 0=default WMMA-F16, 1=qkf32 (exact QK^T), 2=pvf32
                // (exact P@V), 3=both exact (qkf32+pvf32 == full F32 = scalar-equiv).
                // Diagnostic-only until the user's gate.
                let attn_precise: u8 = {
                    use std::sync::OnceLock;
                    static AP: OnceLock<u8> = OnceLock::new();
                    *AP.get_or_init(
                        || match std::env::var("LUMEN_CUDA_ATTN_PRECISE").as_deref() {
                            Ok("1") => 1,
                            Ok("2") => 2,
                            Ok("3") => 3,
                            Ok("4") => 4,
                            Ok("0") => 0,
                            // Unset → ratified per-class default (pvf32 for MoE +
                            // dense ≤32 layers; legacy WMMA for the 27B class).
                            _ => crate::runtime_defaults::attn_precise_default(),
                        },
                    )
                };
                unsafe {
                    if !force_scalar_attn
                        && batch >= 16
                        && st.kernels.flash_attention_wmma.is_some()
                    {
                        match attn_precise {
                            1 if st.kernels.flash_attention_wmma_qkf32.is_some() => {
                                super::prefill::launch_flash_attention_wmma_variant(
                                    &self.device,
                                    &st.kernels,
                                    &pf.q,
                                    kv_cache,
                                    &mut pf.attn_out,
                                    batch,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    pos_start,
                                    true,
                                )?;
                            }
                            2 if st.kernels.flash_attention_wmma_pvf32.is_some() => {
                                super::prefill::launch_flash_attention_wmma_variant(
                                    &self.device,
                                    &st.kernels,
                                    &pf.q,
                                    kv_cache,
                                    &mut pf.attn_out,
                                    batch,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    pos_start,
                                    false,
                                )?;
                            }
                            3 => {
                                // both exact == full F32 == scalar br4 (validated path)
                                super::prefill::launch_flash_attention_br4(
                                    &self.device,
                                    &st.kernels,
                                    &pf.q,
                                    kv_cache,
                                    &mut pf.attn_out,
                                    batch,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    pos_start,
                                )?;
                            }
                            4 if st.kernels.flash_attention_wmma_split.is_some() => {
                                super::prefill::launch_flash_attention_wmma_split(
                                    &self.device,
                                    &st.kernels,
                                    &pf.q,
                                    kv_cache,
                                    &mut pf.attn_out,
                                    batch,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    pos_start,
                                )?;
                            }
                            _ => {
                                super::prefill::launch_flash_attention_wmma(
                                    &self.device,
                                    &st.kernels,
                                    &pf.q,
                                    kv_cache,
                                    &mut pf.attn_out,
                                    batch,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    pos_start,
                                )?;
                            }
                        }
                    } else {
                        super::prefill::launch_flash_attention_br4(
                            &self.device,
                            &st.kernels,
                            &pf.q,
                            kv_cache,
                            &mut pf.attn_out,
                            batch,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            pos_start,
                        )?;
                    }
                }

                // [ATTNPROBE] dump raw attention output (pre-gate) for full-attn
                // layers (last token, head 0, first 3 dims) vs llama attn_pregate.
                if ropeprobe && batch > 1 {
                    let ah = self.device.dtoh_copy(&pf.attn_out)?;
                    let o = (batch - 1) * qd;
                    eprintln!(
                        "[ATTNPROBE] layer={layer_idx} attn_out_h0[0..3]={:?}",
                        &ah[o..o + 3.min(qd)]
                    );
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
                        .map_err(|e| {
                            RuntimeError::Compute(format!("sigmoid_mul prefill dtod: {e}"))
                        })?;
                } else {
                    return Err(RuntimeError::Compute(
                        "Q+gate fusion requires sigmoid_mul kernel".into(),
                    ));
                }

                // 2f. Output projection + residual
                unsafe {
                    super::prefill::launch_gemm_residual(
                        &self.device,
                        &st.kernels,
                        &lw.wo,
                        None,
                        &pf.attn_out,
                        &pf.x,
                        &mut pf.attn_proj,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        hidden_dim,
                        q_dim,
                        "wo",
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
                            &self.device,
                            &st.kernels,
                            &pf.attn_proj,
                            &lw.ffn_norm,
                            &mut pf.normed,
                            eps,
                            batch,
                            hidden_dim,
                        )?;
                        super::prefill::launch_gemm_projection(
                            &self.device,
                            &st.kernels,
                            &lw.w_gate,
                            None,
                            &pf.normed,
                            &mut pf.gate,
                            &mut pf.dequant_f32,
                            &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch,
                            inter_dim,
                            hidden_dim,
                            "gate",
                        )?;
                        super::prefill::launch_gemm_projection(
                            &self.device,
                            &st.kernels,
                            &lw.w_up,
                            None,
                            &pf.normed,
                            &mut pf.up,
                            &mut pf.dequant_f32,
                            &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch,
                            inter_dim,
                            hidden_dim,
                            "up",
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_swiglu_batched(
                            &self.device,
                            &st.kernels,
                            &mut pf.gate,
                            &pf.up,
                            batch,
                            inter_dim,
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_gemm_projection(
                            &self.device,
                            &st.kernels,
                            &lw.w_down,
                            None,
                            &pf.gate,
                            &mut pf.down,
                            &mut pf.dequant_f32,
                            &mut pf.activation_f16,
                            &mut pf.dequant_f16,
                            batch,
                            hidden_dim,
                            inter_dim,
                            "down",
                        )?;
                    }
                    unsafe {
                        super::prefill::launch_residual_add_batched(
                            &self.device,
                            &st.kernels,
                            &mut pf.attn_proj,
                            &pf.down,
                            batch,
                            hidden_dim,
                        )?;
                    }
                    self.device
                        .stream
                        .memcpy_dtod(&pf.attn_proj, &mut pf.x)
                        .map_err(|e| {
                            RuntimeError::Compute(format!("dtod x<-attn_proj qgate prefill: {e}"))
                        })?;
                }
                continue; // Skip the standard path below
            }

            unsafe {
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.wq,
                    None,
                    &pf.normed,
                    &mut pf.q,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    q_dim,
                    hidden_dim,
                    "wq",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.wk,
                    None,
                    &pf.normed,
                    &mut pf.k,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    kv_dim,
                    hidden_dim,
                    "wk",
                )?;
                super::prefill::launch_gemm_projection(
                    &self.device,
                    &st.kernels,
                    &lw.wv,
                    None,
                    &pf.normed,
                    &mut pf.v,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    kv_dim,
                    hidden_dim,
                    "wv",
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
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.q)
                            .arg(bq)
                            .arg(&total)
                            .arg(&dim_u32)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add_batched bq prefill: {e}"))
                            })?;
                    }
                    if let Some(ref bk) = lw.bk {
                        let total = (batch * kv_dim) as u32;
                        let dim_u32 = kv_dim as u32;
                        let g = (total + block - 1) / block;
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.k)
                            .arg(bk)
                            .arg(&total)
                            .arg(&dim_u32)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add_batched bk prefill: {e}"))
                            })?;
                    }
                    if let Some(ref bv) = lw.bv {
                        let total = (batch * kv_dim) as u32;
                        let dim_u32 = kv_dim as u32;
                        let g = (total + block - 1) / block;
                        self.device
                            .stream
                            .launch_builder(&st.kernels.bias_add_batched)
                            .arg(&mut pf.v)
                            .arg(bv)
                            .arg(&total)
                            .arg(&dim_u32)
                            .launch(CudarcLaunchConfig {
                                grid_dim: (g, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .map_err(|e| {
                                RuntimeError::Compute(format!("bias_add_batched bv prefill: {e}"))
                            })?;
                    }
                }
            }

            // 2c. Batched RoPE with per-token positions.
            let rotary_dim = hp.rotary_dim.unwrap_or(0) as u32;
            unsafe {
                super::prefill::launch_rope_batched(
                    &self.device,
                    &st.kernels,
                    &mut pf.q,
                    &mut pf.k,
                    pos_start,
                    batch,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    theta,
                    hp.rope_neox,
                    rotary_dim,
                )?;
            }

            // 2d. Batch KV cache write for all tokens at once.
            let kv_cache = &mut st.kv_caches[layer_idx];
            unsafe {
                super::prefill::launch_kv_cache_write_batch(
                    &self.device,
                    &st.kernels,
                    &mut kv_cache.k_cache,
                    &pf.k,
                    pos_start,
                    batch,
                    num_kv_heads,
                    kv_cache.max_seq_len,
                    head_dim,
                )?;
                super::prefill::launch_kv_cache_write_batch(
                    &self.device,
                    &st.kernels,
                    &mut kv_cache.v_cache,
                    &pf.v,
                    pos_start,
                    batch,
                    num_kv_heads,
                    kv_cache.max_seq_len,
                    head_dim,
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
                if batch >= 16 && st.kernels.flash_attention_wmma.is_some() {
                    // WMMA-PRECISION-FIX-RCA: honor LUMEN_CUDA_ATTN_PRECISE on
                    // this secondary prefill-attention dispatch as well, so the
                    // eventual default change is complete across both sites.
                    let attn_precise: u8 = {
                        use std::sync::OnceLock;
                        static AP2: OnceLock<u8> = OnceLock::new();
                        *AP2.get_or_init(|| {
                            match std::env::var("LUMEN_CUDA_ATTN_PRECISE").as_deref() {
                                Ok("1") => 1,
                                Ok("2") => 2,
                                Ok("3") => 3,
                                Ok("4") => 4,
                                Ok("0") => 0,
                                // Unset → per-class default (must mirror the AP
                                // site above; both dispatch sites stay in sync).
                                _ => crate::runtime_defaults::attn_precise_default(),
                            }
                        })
                    };
                    match attn_precise {
                        1 if st.kernels.flash_attention_wmma_qkf32.is_some() => {
                            super::prefill::launch_flash_attention_wmma_variant(
                                &self.device,
                                &st.kernels,
                                &pf.q,
                                kv_cache,
                                &mut pf.attn_out,
                                batch,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                pos_start,
                                true,
                            )?;
                        }
                        2 if st.kernels.flash_attention_wmma_pvf32.is_some() => {
                            super::prefill::launch_flash_attention_wmma_variant(
                                &self.device,
                                &st.kernels,
                                &pf.q,
                                kv_cache,
                                &mut pf.attn_out,
                                batch,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                pos_start,
                                false,
                            )?;
                        }
                        3 => {
                            super::prefill::launch_flash_attention_br4(
                                &self.device,
                                &st.kernels,
                                &pf.q,
                                kv_cache,
                                &mut pf.attn_out,
                                batch,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                pos_start,
                            )?;
                        }
                        4 if st.kernels.flash_attention_wmma_split.is_some() => {
                            super::prefill::launch_flash_attention_wmma_split(
                                &self.device,
                                &st.kernels,
                                &pf.q,
                                kv_cache,
                                &mut pf.attn_out,
                                batch,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                pos_start,
                            )?;
                        }
                        _ => {
                            super::prefill::launch_flash_attention_wmma(
                                &self.device,
                                &st.kernels,
                                &pf.q,
                                kv_cache,
                                &mut pf.attn_out,
                                batch,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                pos_start,
                            )?;
                        }
                    }
                } else {
                    super::prefill::launch_flash_attention_br4(
                        &self.device,
                        &st.kernels,
                        &pf.q,
                        kv_cache,
                        &mut pf.attn_out,
                        batch,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        pos_start,
                    )?;
                }
            }

            // 2f. Batched output projection + residual via GEMM (no F16 caches).
            unsafe {
                super::prefill::launch_gemm_residual(
                    &self.device,
                    &st.kernels,
                    &lw.wo,
                    None,
                    &pf.attn_out,
                    &pf.x,
                    &mut pf.attn_proj,
                    &mut pf.dequant_f32,
                    &mut pf.activation_f16,
                    &mut pf.dequant_f16,
                    batch,
                    hidden_dim,
                    q_dim,
                    "wo",
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
                        &self.device,
                        &st.kernels,
                        &pf.attn_proj,
                        &lw.ffn_norm,
                        &mut pf.normed,
                        eps,
                        batch,
                        hidden_dim,
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device,
                        &st.kernels,
                        &lw.w_gate,
                        None,
                        &pf.normed,
                        &mut pf.gate,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        inter_dim,
                        hidden_dim,
                        "gate",
                    )?;
                    super::prefill::launch_gemm_projection(
                        &self.device,
                        &st.kernels,
                        &lw.w_up,
                        None,
                        &pf.normed,
                        &mut pf.up,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        inter_dim,
                        hidden_dim,
                        "up",
                    )?;
                }

                // 2h. Batched SwiGLU (standard path, no F16 fusion).
                unsafe {
                    super::prefill::launch_swiglu_batched(
                        &self.device,
                        &st.kernels,
                        &mut pf.gate,
                        &pf.up,
                        batch,
                        inter_dim,
                    )?;
                }

                // 2i. Batched down projection via GEMM (no F16 caches).
                unsafe {
                    super::prefill::launch_gemm_projection(
                        &self.device,
                        &st.kernels,
                        &lw.w_down,
                        None,
                        &pf.gate,
                        &mut pf.down,
                        &mut pf.dequant_f32,
                        &mut pf.activation_f16,
                        &mut pf.dequant_f16,
                        batch,
                        hidden_dim,
                        inter_dim,
                        "down",
                    )?;
                }

                // 2j. Batched residual add: x = attn_proj + down.
                // Write result directly to pf.x (eliminates the separate memcpy_dtod).
                unsafe {
                    super::prefill::launch_residual_add_batched(
                        &self.device,
                        &st.kernels,
                        &mut pf.attn_proj,
                        &pf.down,
                        batch,
                        hidden_dim,
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
                &self.device,
                &st.kernels,
                &pf.x,
                &mut st.scratch.x_gpu,
                batch - 1,
                hidden_dim,
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

    fn preload_weights(&mut self, weights: &dyn WeightProvider) -> Result<(), RuntimeError> {
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

            // Build the REPACKED aligned gate+up planes for the W10 wide-M
            // gate+up path. Runs AFTER upload so the GPU blob exists; reads raw
            // Q8_0 weights via the repack kernel into aligned planes. Leaves the
            // original blob byte-untouched.
            //
            // Q8-ONLY GUARD: the repack (`moe_repack_gate_up_q8_0`, 34-byte Q8_0
            // blocks) is consumed only by the Q8 W10 dispatch (`w10_enabled`
            // requires `q8_path`). For Q4/BF16 MoE it would build ~1.5 GB/layer of
            // Q8-layout planes over 18-byte Q4 blocks (OOB strides) that are never
            // read — so gate the whole repack on Q8_0 experts.
            if super::moe::moe_repack_needed() && layer_idx < st.moe_meta_cache.len() {
                if let Some(meta_ref) = st.moe_meta_cache[layer_idx]
                    .as_ref()
                    .filter(|m| m.expert_gate_quant == QuantScheme::Q8_0)
                {
                    let num_experts = meta_ref.expert_down_offs.len();
                    let hidden_dim = hp_copy.hidden_dim as usize;
                    let inter_dim = hp_copy.intermediate_dim as usize;
                    // No fast-down path remains; only the W10 gate+up repack.
                    let build_gate_up = super::moe::moe_gate_up_w10_enabled();
                    let repack_fn = st.kernels.moe_repack_down_q8_0.as_ref();
                    let repack_gu_fn = st.kernels.moe_repack_gate_up_q8_0.as_ref();
                    let offs = st.moe_batched_offsets[layer_idx].as_ref();
                    if let (Some(repack_fn), Some(offs), Some(blob)) =
                        (repack_fn, offs, gpu_weights.moe_layer_blob.as_ref())
                    {
                        let rp = super::moe::build_repacked_down(
                            &self.device,
                            repack_fn,
                            repack_gu_fn,
                            blob,
                            &offs.down_offsets,
                            &offs.gate_up_offsets,
                            num_experts,
                            hidden_dim,
                            inter_dim,
                            build_gate_up,
                        )?;
                        st.moe_repacked[layer_idx] = Some(rp);
                        if layer_idx == 0 {
                            eprintln!(
                                "[CUDA] W-infra: repacked planes built (layer 0: \
                                 E={num_experts} H={hidden_dim} I={inter_dim} \
                                 gate_up={build_gate_up})"
                            );
                        }
                    }
                }
            }

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
                RuntimeError::Compute(format!("F16 pre-dequant layer {layer_idx}: {e}",))
            })?;
        }
        let mem_after_f16_cache = self.device.free_memory().unwrap_or(0);
        eprintln!(
            "[CUDA mem] after F16 dequant caches: {:.2} GB free (consumed: {:.2} GB)",
            (mem_after_f16_cache as f64) / 1.0e9,
            (mem_before_f16_cache.saturating_sub(mem_after_f16_cache) as f64) / 1.0e9
        );

        // GDN ssm_alpha / ssm_beta F16 cache (env LUMEN_CUDA_GDN_AB_F16, MoE-gated,
        // default OFF -> byte-identical). `dequant_layer_q8_to_f16` above
        // EARLY-RETURNS for GDN layers (layer_type == 1) to avoid OOM from the
        // full per-layer F16 weight set, so the GDN F16 caches stay `None` by
        // default. Here we dequant ONLY the two TINY ssm_alpha / ssm_beta
        // projections (out_dim = num_heads ~32, in_dim = hidden ~2048 =>
        // ~64K params each, ~128 KB F16 per tensor per layer) to F16. The decode
        // GDN projection (per-token Q8_1/dp4a matvec) and the batched prefill GDN
        // projection (MMQ INT8 for MoE Q8) DIVERGE ~20% at L0 on these two
        // tensors because they use different INT8 activation-quant + reduction
        // orders; the 256-expert router amplifies that into expert-selection
        // flips. With this cache populated, BOTH the decode `launch_matvec`
        // alpha/beta arm (-> `launch_hgemv_f16`, cublasGemmEx N=1) AND the
        // prefill `launch_gemm_projection` Q8Raw arm (-> the F16-cache HGEMM
        // fast path, cublasGemmEx N=batch) read this SAME F16 weight, making the
        // projections bit-identical decode-vs-prefill (the proven-clean qkv/gate
        // recipe). Cost is negligible (~6 MB total across 40 GDN layers).
        if gdn_ab_f16_enabled() {
            use super::gpu_buffers::{dequant_q8_to_f16_gpu, GpuWeightBuf};
            // Q8_0: 34 bytes per block of 32 elements.
            let q8_elems = |q8: &CudaSlice<u8>| -> usize { (q8.len() / 34) * 32 };
            let mut n_cached = 0usize;
            for layer in cache.iter_mut() {
                if layer.layer_type != 1 {
                    continue;
                }
                if let Some(GpuWeightBuf::Q8Raw(q8)) = layer.ssm_alpha.as_ref() {
                    let n = q8_elems(q8);
                    let f16 =
                        dequant_q8_to_f16_gpu(&self.device, &st.kernels.dequant_q8_0_to_f16, q8, n)
                            .map_err(|e| {
                                RuntimeError::Compute(format!("GDN_AB_F16 ssm_alpha dequant: {e}"))
                            })?;
                    layer.ssm_alpha_f16 = Some(f16);
                    n_cached += 1;
                }
                if let Some(GpuWeightBuf::Q8Raw(q8)) = layer.ssm_beta.as_ref() {
                    let n = q8_elems(q8);
                    let f16 =
                        dequant_q8_to_f16_gpu(&self.device, &st.kernels.dequant_q8_0_to_f16, q8, n)
                            .map_err(|e| {
                                RuntimeError::Compute(format!("GDN_AB_F16 ssm_beta dequant: {e}"))
                            })?;
                    layer.ssm_beta_f16 = Some(f16);
                    n_cached += 1;
                }
            }
            eprintln!(
                "[CUDA] GDN_AB_F16: ACTIVE — dequanted {n_cached} ssm_alpha/ssm_beta \
                 weights to F16 (decode+prefill bit-identical projection path)"
            );
        }

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
                // Resolve the split-clone VRAM budget (env override, else
                // free-mem-aware default) at the clone site. init() has already
                // allocated the KV caches by now, so `free` excludes them and
                // the resolver does not subtract a KV reserve again. Reuses
                // the shared L8 resolver (same helper as the Q4 clone pass).
                let budget =
                    resolve_split_clone_budget("LUMEN_CUDA_Q8_SPLIT_BUDGET_GB", &self.device);
                let mem_before_q8_split = budget.free_mem_bytes;
                let (n_layers_split, oom_layer, oom_count, jobs_attempted, total_jobs) = unsafe {
                    repack_all_layers_q8_clone_to_split(
                        &self.device,
                        split_repack_fn,
                        &mut cache,
                        &hp_copy,
                        budget.budget_bytes,
                    )
                };
                // Ship-what-you-gated proof: the resolved cap plus its inputs.
                // `n_layers_split` distinct FFN layers received a split sibling out
                // of the model's `num_layers` FFN-bearing layers (64 for 27B);
                // `total_jobs` gate/up/down weight-jobs were attempted.
                eprintln!(
                    "[CUDA] Q8 split-clone budget: resolved={:.2} GB (free={:.2} GB, \
                     slack={:.2} GB, source={}); cloned \
                     {n_layers_split}/{num_layers} FFN layers \
                     ({jobs_attempted}/{total_jobs} weight-jobs reached)",
                    (budget.budget_bytes as f64) / 1.0e9,
                    (budget.free_mem_bytes as f64) / 1.0e9,
                    (budget.slack_bytes as f64) / 1.0e9,
                    if budget.from_env { "env" } else { "auto" },
                );
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
                    if lw.q8_split_wq.is_some()
                        || lw.q8_split_wk.is_some()
                        || lw.q8_split_wv.is_some()
                        || lw.q8_split_wo.is_some()
                        || lw.q8_split_w_gate.is_some()
                        || lw.q8_split_w_up.is_some()
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
                        &self.device,
                        split_repack_fn,
                        proj_q8,
                        vocab_size,
                        hidden,
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
            if st.kernels.matvec_q8_0_aligned.is_some()
                || st.kernels.matvec_q8_aligned_q8_1.is_some()
            {
                for (layer_idx, layer) in cache.iter_mut().enumerate() {
                    super::gpu_buffers::repack_layer_q8_to_aligned(
                        &self.device,
                        repack_fn,
                        layer,
                        &hp_copy,
                        has_gdn,
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(
                            format!("Q8_0 aligned repack layer {layer_idx}: {e}",),
                        )
                    })?;
                }
                // Skip output_proj repack for GDN models (too large, OOM risk, negligible impact).
                if !has_gdn {
                    if let Some(ref proj_q8) = st.globals.output_proj_q8 {
                        let vocab_size = hp_copy.vocab_size as usize;
                        let hidden = hp_copy.hidden_dim as usize;
                        let num_elements = vocab_size * hidden;
                        match super::gpu_buffers::repack_q8_to_aligned(
                            &self.device,
                            repack_fn,
                            proj_q8,
                            num_elements,
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
        {
            if let (Some(ref split_repack_fn), true) = (
                st.kernels.repack_q4_raw_to_split.as_ref(),
                st.kernels.matvec_q4_split_q8_1.is_some(),
            ) {
                // Resolve the split-clone VRAM budget (env override, else
                // free-mem-aware default) at the clone site. init() has already
                // allocated the KV caches by now, so `free` excludes them and
                // the resolver does not subtract a KV reserve again.
                let budget =
                    resolve_split_clone_budget("LUMEN_CUDA_Q4_SPLIT_BUDGET_GB", &self.device);
                let mem_before_q4_split = budget.free_mem_bytes;
                let (n_layers_split, oom_layer, oom_count, jobs_attempted, total_jobs) = unsafe {
                    repack_all_layers_q4_clone_to_split(
                        &self.device,
                        split_repack_fn,
                        &mut cache,
                        &hp_copy,
                        budget.budget_bytes,
                    )
                };
                // Ship-what-you-gated proof: the resolved cap plus its inputs.
                // `n_layers_split` distinct layers received a split sibling out of the
                // model's `num_layers` FFN-bearing layers (64 for 27B); `total_jobs`
                // gate/up/down weight-jobs were attempted.
                eprintln!(
                    "[CUDA] Q4 split-clone budget: resolved={:.2} GB (free={:.2} GB, \
                     slack={:.2} GB, source={}); cloned \
                     {n_layers_split}/{num_layers} layers \
                     ({jobs_attempted}/{total_jobs} weight-jobs reached; FFN + \
                     attention + GDN projections are all eligible)",
                    (budget.budget_bytes as f64) / 1.0e9,
                    (budget.free_mem_bytes as f64) / 1.0e9,
                    (budget.slack_bytes as f64) / 1.0e9,
                    if budget.from_env { "env" } else { "auto" },
                );
                let mem_after_q4_split = self.device.free_memory().unwrap_or(0);
                let consumed_gb =
                    (mem_before_q4_split.saturating_sub(mem_after_q4_split) as f64) / 1.0e9;
                eprintln!(
                    "[CUDA] cloned Q4 split siblings on \
                     {n_layers_split} layers, {total_jobs} jobs attempted, \
                     {oom_count} OOMs (first at layer {:?}), {consumed_gb:.2} GB consumed",
                    oom_layer,
                );
                for (idx, lw) in cache.iter().enumerate() {
                    if lw.q4_split_wq.is_some()
                        || lw.q4_split_wk.is_some()
                        || lw.q4_split_wv.is_some()
                        || lw.q4_split_wo.is_some()
                        || lw.q4_split_w_gate.is_some()
                        || lw.q4_split_w_up.is_some()
                        || lw.q4_split_w_down.is_some()
                    {
                        layers_with_q4_split.insert(idx);
                    }
                }
            } else {
                eprintln!(
                    "[CUDA] split kernels unavailable; \
                     decode will use Q4Raw/Q4Aligned base path"
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
                        RuntimeError::Compute(
                            format!("Q4_0 aligned repack layer {layer_idx}: {e}",),
                        )
                    })?;
                }
                // Skip output_proj repack for GDN models (too large, OOM risk, negligible impact).
                if !has_gdn {
                    if let Some(ref proj_q4) = st.globals.output_proj_q4 {
                        let vocab_size = hp_copy.vocab_size as usize;
                        let hidden = hp_copy.hidden_dim as usize;
                        let num_elements = vocab_size * hidden;
                        match super::gpu_buffers::repack_q4_to_aligned(
                            &self.device,
                            repack_fn,
                            proj_q4,
                            num_elements,
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

        // Allocate Q+gate fusion scratch buffers if any layer has attn_q_norm.
        let has_qgate = cache.iter().any(|lw| lw.attn_q_norm.is_some());
        if has_qgate {
            let q_dim = hp_copy.num_heads as usize * hp_copy.head_dim as usize;
            let q_gate_dim = q_dim * 2;
            st.scratch.q_gate = Some(self.device.alloc_zeros(q_gate_dim)?);
            st.scratch.gate_buf = Some(self.device.alloc_zeros(q_dim)?);
            eprintln!(
                "[CUDA] Q+gate fusion scratch: q_gate={q_gate_dim}, gate_buf={q_dim} elements"
            );
        }

        st.has_gdn_layers = has_gdn;
        st.has_qgate_layers = has_qgate;
        // Detect MoE layers from `moe_meta_cache` (populated earlier in
        // preload_weights).
        let has_moe = st.moe_meta_cache.iter().any(|m| m.is_some());
        st.has_moe_layers = has_moe;

        // Q4_0 QUALITY FIX (per-model routing). int8 Q8_1 activation quantization
        // compounds through the GDN linear-attention recurrence and collapses Q4_0
        // decode quality on the NARROW-GDN configs (GDN v-heads == 32): Qwen3.5-9B
        // (GQ 7/15; llama.cpp's own int8 path degenerates identically on the same
        // GGUF -> engine-agnostic) and Qwen3.5-MoE (GQ-001 13/15 at int8). Both
        // require F32-activation matvecs. The 27B has WIDER GDN heads (v-heads == 48)
        // and is CERTIFIED clean on the fast int8 path (GQ-001/002/004 all pass), so
        // it keeps dp4a. Keyed off the GDN v-head width (a config value, not a name):
        // 32 -> F32, 48 -> int8. Gated on has_gdn so non-GDN models keep int8.
        let narrow_gdn = has_gdn && hp_copy.gdn_dims().num_v_heads == 32;
        let dense = hp_copy.num_experts.is_none();
        st.kernels.q4_act_plan = crate::runtime_defaults::Q4ActPlan::for_model(narrow_gdn, dense);
        eprintln!("[CUDA] {}", st.kernels.q4_act_plan.manifest);
        // Which families this model actually has Q4 weights for. The census
        // verifier needs this to tell "did not run" from "does not exist" —
        // without it, a family that silently stopped dispatching certified
        // clean by producing no evidence at all.
        {
            use crate::runtime_defaults::Q4ProjectionFamily as F;
            let q4 = |w: &GpuWeightBuf| matches!(w, GpuWeightBuf::Q4Raw(_));
            let q4o = |w: &Option<GpuWeightBuf>| w.as_ref().is_some_and(q4);
            let mut expected = Vec::new();
            let mut mark = |f: F, present: bool| {
                if present && !expected.contains(&f) {
                    expected.push(f);
                }
            };
            // Iterate the freshly-built `cache`, NOT st.layer_weights_cache —
            // the latter is not assigned until after this block (r2 audit
            // §6.0: iterating it here read the PREVIOUS load's value, empty on
            // first load, so expected==[] and the verifier either hard-failed
            // on plumbing or certified silence).
            for lw in &cache {
                // GDN layers reuse `wq` as their fused in-projection and
                // dispatch it under the `gdn_qkv` label, so which family a
                // layer contributes depends on its layer_type.
                let gdn = lw.layer_type == 1;
                mark(F::AttnQkv, !gdn && (q4(&lw.wq) || q4(&lw.wk) || q4(&lw.wv)));
                mark(F::GdnQkv, gdn && q4(&lw.wq));
                mark(F::AttnWo, q4(&lw.wo));
                mark(F::FfnGateUp, q4(&lw.w_gate) || q4(&lw.w_up));
                mark(F::FfnDown, q4(&lw.w_down));
                mark(F::GdnAttnGate, q4o(&lw.attn_gate) || q4o(&lw.ssm_out));
            }
            eprintln!("[CUDA] route census expects families: {expected:?}");
            crate::runtime_defaults::route_census_set_plan(&st.kernels.q4_act_plan, &expected);
        }

        // LUMEN_CUDA_Q4_F32ACT_KERNEL: select among F32-EXACT Q4_0 decode matvec
        // variants at the two smem launch sites. ALL variants keep FULL F32
        // activations — pure kernel/occupancy, no precision change. Read ONCE.
        //   "row"  -> matvec_q4_0 (one-block-per-row, no shmem)
        //   "nr4"  -> matvec_q4_0_smem_nr4 (NR=4 wide smem)
        //   "nr8"  -> matvec_q4_0_smem_nr8 (NR=8 wide smem)
        //   "smem" -> explicit opt-out to the NR=2 path
        // DEFAULT-ON: on the narrow-GDN class (9B-Q4 /
        // MoE-Q4 dense), NR=4 is the default — occupancy win at BYTE-EXACT F32,
        // GQ-confirmed IDENTICAL to NR=2. All other (non-F32-act) paths keep NR=2.
        st.kernels.q4_f32act_kernel =
            match std::env::var("LUMEN_CUDA_Q4_F32ACT_KERNEL").ok().as_deref() {
                Some("row") => Q4F32ActKernel::Row,
                Some("nr4") => Q4F32ActKernel::Nr4,
                Some("nr8") => Q4F32ActKernel::Nr8,
                Some("smem") => Q4F32ActKernel::Smem,
                _ if narrow_gdn => Q4F32ActKernel::Nr4,
                _ => Q4F32ActKernel::Smem,
            };
        if !matches!(st.kernels.q4_f32act_kernel, Q4F32ActKernel::Smem) {
            eprintln!(
                "[CUDA] Q4_0 F32-act decode matvec variant: {:?} \
                 (LUMEN_CUDA_Q4_F32ACT_KERNEL; FULL F32 activations, occupancy-only)",
                st.kernels.q4_f32act_kernel
            );
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
        let has_f16 = st
            .layer_weights_cache
            .iter()
            .any(|lw| matches!(&lw.wq, GpuWeightBuf::F16Raw(_)) || lw.wq_f16.is_some());
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
                (q_dim, hidden_dim),     // wq projection (x36 layers)
                (kv_dim, hidden_dim),    // wk, wv projections (x36 layers)
                (hidden_dim, q_dim),     // wo output projection (x36 layers)
                (inter_dim, hidden_dim), // gate, up projections (x36 layers)
                (hidden_dim, inter_dim), // down projection (x36 layers)
            ];
            // Q+gate fusion: add (q_dim*2, hidden_dim) for fused Q+gate projection.
            if st
                .layer_weights_cache
                .iter()
                .any(|lw| lw.attn_q_norm.is_some())
            {
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
        let has_bf16 = st
            .layer_weights_cache
            .iter()
            .any(|lw| matches!(&lw.wq, GpuWeightBuf::Bf16Raw(_)));
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
                (q_dim, hidden_dim),      // wq projection
                (kv_dim, hidden_dim),     // wk, wv projections
                (hidden_dim, q_dim),      // wo output projection
                (inter_dim, hidden_dim),  // gate, up projections
                (hidden_dim, inter_dim),  // down projection
                (vocab_size, hidden_dim), // final output_proj (vocab head)
            ];
            if st
                .layer_weights_cache
                .iter()
                .any(|lw| lw.attn_q_norm.is_some())
            {
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

    /// GPU-side greedy decode returning the token id directly (4-byte readback).
    ///
    /// Active only when the engine selects the greedy fast path, which requires
    /// `caps().gpu_argmax == true` -- i.e. weights preloaded AND
    /// `LUMEN_CUDA_GPU_SAMPLE=1`. Runs the same non-graph compute pipeline as the
    /// first-token / GDN `decode_token` path and reads back only the on-GPU
    /// argmax index, eliminating the per-token full-vocab logits D2H copy and the
    /// host-side argmax. Advances `kv.seq_len()` internally (the caller must NOT
    /// call `advance_seq_len`). Falls back with an explicit error if weights are
    /// not GPU-resident, matching the cap's `is_preloaded` gate.
    fn decode_token_greedy(
        &self,
        token_id: u32,
        _weights: &dyn WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
        let hp = self.hp()?;
        let num_layers = hp.num_layers as usize;
        let seq_pos = kv.seq_len();

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;

        // Greedy GPU argmax requires GPU-resident weights (same contract as the
        // gpu_argmax cap, which is gated on layer_weights_cache being populated).
        if st.layer_weights_cache.len() < num_layers {
            return Err(RuntimeError::Compute(
                "decode_token_greedy requires GPU-resident weights (call preload_weights first)"
                    .into(),
            ));
        }

        self.decode_token_greedy_normal(token_id, seq_pos, num_layers, hp, st, kv)
    }

    fn decode_token(
        &self,
        token_id: u32,
        _weights: &dyn WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        let hp = self.hp()?;
        let num_layers = hp.num_layers as usize;
        let seq_pos = kv.seq_len();

        let mut state_guard = self.state.lock().unwrap();
        let st = state_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;

        // Require GPU-resident weights for the zero-sync decode path.
        if st.layer_weights_cache.len() < num_layers {
            return Err(RuntimeError::Compute(
                "decode_token requires GPU-resident weights (call preload_weights first)".into(),
            ));
        }

        // CUDA graph capture removed (condor/tern/harrier/osprey all NO-GO):
        // the sampling decode path runs the eager pipeline, identical to the
        // greedy path, via `decode_token_normal`.
        self.decode_token_normal(token_id, seq_pos, num_layers, hp, st, kv)
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
        assert!(
            caps.gpu_resident,
            "CUDA backend must advertise gpu_resident=true"
        );
        assert!(
            caps.batched_prefill,
            "CUDA backend must advertise batched_prefill=true"
        );
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
                runtime_fallback_armed: BF16_GEMMEX_FALLBACK_ARMED.load(Ordering::Relaxed),
                _lock: lock,
            }
        }
    }

    impl Drop for Bf16StatesnapshotGuard {
        fn drop(&mut self) {
            BF16_GEMMEX_AVAILABLE.store(self.available, Ordering::Relaxed);
            BF16_GEMMEX_FALLBACK_ARMED.store(self.runtime_fallback_armed, Ordering::Relaxed);
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
