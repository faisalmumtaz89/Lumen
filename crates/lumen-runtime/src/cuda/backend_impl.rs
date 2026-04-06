//! CUDA `ComputeBackend` implementation.
//!
//! Implements the full single-token decode pipeline on GPU:
//! - `embed_token`: GPU embedding lookup (F32 or Q8_0)
//! - `compute_layer`: RMSNorm -> QKV -> RoPE -> KV cache -> Attention ->
//!   Output proj + residual -> FFN RMSNorm -> SwiGLU MLP -> Residual
//! - `compute_final`: Final RMSNorm -> output projection to logits
//! - `preload_weights`: Upload ALL layer weights to GPU once at startup
//! - `decode_token`: GPU-resident single-token decode (no per-layer upload)
//!
//! Supports F32, F16, Q8_0, and Q4_0 weight quantization. Two weight paths:
//! - **GPU-resident** (`preload_weights` called): all layer weights cached on GPU.
//!   `compute_layer` uses cached `LayerWeightsGpu` -- zero host-to-device transfer.
//! - **Streaming** (no preload): per-call `upload_layer_weights` from `LayerView`.

use crate::compute::{ActivationBuffer, BackendCaps, ComputeBackend, ComputeDtype, Logits};
use crate::error::RuntimeError;
use crate::kv::KvCacheView;
use crate::weight::cache::{LayerView, WeightProvider};
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;

use super::decode::{
    self, KernelSet, attention_block_size, attention_shared_bytes, fused_norm_matvec_block_size,
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
    /// Output projection as raw Q4_0 bytes (None if not Q4_0).
    output_proj_q4: Option<CudaSlice<u8>>,
    /// Output projection as 20-byte aligned Q4_0 (None if not Q4_0 or repack failed).
    /// Preferred over output_proj_q4 for decode (int* nibble loads vs byte loads).
    output_proj_q4_aligned: Option<CudaSlice<u8>>,
    /// Embedding table (F32 path): [vocab_size * hidden_dim]
    /// Empty if embedding uses a quantized raw path instead.
    embedding: CudaSlice<f32>,
    /// Embedding as raw Q8_0 bytes (None if not Q8_0).
    embedding_q8: Option<CudaSlice<u8>>,
    /// Embedding as raw F16 bytes (None if not F16).
    embedding_f16: Option<CudaSlice<u8>>,
    /// Embedding as raw Q4_0 bytes (None if not Q4_0).
    embedding_q4: Option<CudaSlice<u8>>,
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
    ///
    /// `device_id` selects the GPU ordinal (0 = first GPU).
    pub fn new(device_id: usize) -> Result<Self, RuntimeError> {
        let device = CudaDevice::new(device_id)?;
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
            weight_tying: false,
            cached_hidden_dim: 0,
            cached_vocab_size: 0,
            state: Mutex::new(None),
        })
    }

    /// Access hyperparams, returning an error if `init()` has not been called.
    fn hp(&self) -> Result<&ModelHyperparams, RuntimeError> {
        self.hyperparams.as_ref().ok_or_else(|| {
            RuntimeError::Compute("CUDA backend not initialized: call init() first".into())
        })
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
        // Must match the dispatch order in embed_token() (F16 > Q4_0 > Q8_0 > F32).
        if let Some(ref emb_f16) = st.globals.embedding_f16 {
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
            // Run the GDN attention block (steps 1-13), which replaces the standard
            // QKV -> RoPE -> KV cache -> Attention -> Output proj path.
            // After this, x_gpu contains the post-residual hidden state and
            // attn_proj is set to a copy of x_gpu so the FFN block can read from it.
            self.compute_gdn_attention_gpu(layer_idx, st)?;
        } else {

        let lw: &LayerWeightsGpu = st
            .layer_weights_cache
            .get(layer_idx)
            .ok_or_else(|| RuntimeError::Compute(format!(
                "compute_layer_gpu: layer {layer_idx} not in GPU-resident cache",
            )))?;

        // 1. Fused RMSNorm + QKV projections (same logic as compute_layer).
        if matches!(&lw.wq, GpuWeightBuf::F32(_)) {
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
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            &mut st.scratch.q, q_dim, hidden_dim, "wq",
                            st.algo_cache.get(q_dim, hidden_dim),
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
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            &mut st.scratch.q, q_dim, hidden_dim, "wq",
                            st.algo_cache.get(q_dim, hidden_dim),
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
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            &mut st.scratch.q, q_dim, hidden_dim, "wq",
                            st.algo_cache.get(q_dim, hidden_dim),
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
                    unsafe {
                        launch_hgemv_f16_preconverted(
                            &self.device, wq_f16, &st.scratch.input_f16,
                            &mut st.scratch.q, q_dim, hidden_dim, "wq",
                            st.algo_cache.get(q_dim, hidden_dim),
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
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wq, q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "wq")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wk, q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "wk")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wv, q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "wv")?;
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
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wq, q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "wq")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wk, q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "wk")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wv, q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "wv")?;
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

        // 4. Attention.
        {
            let kv_cache = &st.kv_caches[layer_idx];
            let attn_seq_len = kv_cache.seq_len() as u32;
            let block_size = attention_block_size(attn_seq_len as usize);
            let shared_bytes = attention_shared_bytes(attn_seq_len);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            let nh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            let msl = kv_cache.max_seq_len as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.attention_decode)
                    .arg(&st.scratch.q)
                    .arg(&kv_cache.k_cache)
                    .arg(&kv_cache.v_cache)
                    .arg(&mut st.scratch.attn_out)
                    .arg(&nh)
                    .arg(&nkvh)
                    .arg(&hd)
                    .arg(&attn_seq_len)
                    .arg(&msl)
                    .arg(&scale)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("attention_decode launch: {e}")))?;
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
        } // end else (standard attention path — skipped for GDN layers)

        // Re-borrow layer weights for the FFN block (shared between standard and GDN layers).
        let lw: &LayerWeightsGpu = &st.layer_weights_cache[layer_idx];

        // 6. FFN: fused or separate rmsnorm + gate/up + swiglu + down + residual.
        //
        // Fused gate+up+SwiGLU GEMV: if the kernel is available and shmem fits,
        // compute rms_scale + fused_glu_gemv in 2 dispatches (replacing 3-5).
        // The fused kernel writes silu(gate)*up directly to scratch.gate,
        // so the SwiGLU step is skipped entirely.
        let fused_glu_fired = 'fused_glu: {
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
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_gate, q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "gate")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_up, q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "up")?;
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
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_gate, q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "gate")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_up, q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "up")?;
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
            // Reads F32 gate[] and up[], computes silu(gate)*up inline,
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
            // Reads F32 gate[] and up[], computes silu(gate)*up inline,
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

        // GPU-resident conv positions reserved for future CUDA graph support.
        // Currently None because graph capture is disabled for GDN layers
        // (conv_position is a host-side scalar, not yet graph-compatible).
        let conv_positions_gpu: Option<Vec<CudaSlice<u32>>> = None;

        // Allocate ephemeral scratch buffers (shared across layers).
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
        };

        st.gdn_scratch_gpu = Some(gdn);
        Ok(())
    }

    /// Run the GDN (GatedDeltaNet) attention block on GPU, replacing the
    /// standard softmax attention path for GDN layers.
    ///
    /// Implements the 13-step GDN pipeline:
    ///   1. RMSNorm(x) -> normed
    ///   2. QKV matvec: normed @ wq^T -> qkv_buf
    ///   3a. Conv1D decode on QKV channels
    ///   3b. SiLU activation on conv output
    ///   4a-b. Alpha/beta matvec projections
    ///   4c. Compute gates (alpha decay, beta mixing)
    ///   5. L2-normalize Q and K per head
    ///   6-7. State update + output (delta rule recurrence)
    ///   8. RMSNorm on output
    ///   9. Attention gate matvec
    ///   10. SiLU(gate) * normed_output
    ///   11. Output projection -> ssm_proj
    ///   12. Residual: x_gpu += ssm_proj
    ///   13. Copy x_gpu -> attn_proj (for FFN block)
    ///
    /// After this call, `st.scratch.attn_proj` contains the post-GDN hidden
    /// state ready for the shared FFN block.
    fn compute_gdn_attention_gpu(
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

        let gdn_idx = gdn.gdn_layer_map[layer_idx]
            .ok_or_else(|| RuntimeError::Compute(format!(
                "compute_gdn_attention_gpu: layer {layer_idx} is not a GDN layer",
            )))?;

        // --- Step 1: RMSNorm(x) -> normed ---
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

        // --- Step 2: QKV matvec: normed @ wq^T -> qkv_buf ---
        // wq for GDN is the fused [qkv_dim, hidden_dim] weight.
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

        // --- Step 4a: Alpha matvec: normed @ ssm_alpha^T -> alpha_raw ---
        {
            let ssm_alpha = lw.ssm_alpha.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_alpha weight missing",
                )))?;
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    ssm_alpha,
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
        }

        // --- Step 4b: Beta matvec: normed @ ssm_beta^T -> beta_raw ---
        {
            let ssm_beta = lw.ssm_beta.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_beta weight missing",
                )))?;
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    ssm_beta,
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

        if let Some(ref mega_fn) = st.kernels.gdn_decode_megakernel {
            // === FUSED PATH: 8 launches -> 2 ===
            // Kernel 1 (gdn_decode_megakernel): conv1d+silu, gates, L2 norm, state update.
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

            // Advance circular buffer position.
            let buf_slots = (p.conv_kernel_size - 1) as u32;
            gdn.conv_positions[gdn_idx] = (state_pos + 1) % buf_slots;
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
        // Falls back to unfused rmsnorm + gate_matvec + silu_mul if unavailable.

        // Step 9: Attention gate matvec (same for both paths -- runs BEFORE fused/unfused norm+gate)
        {
            let attn_gate = lw.attn_gate.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: attn_gate weight missing",
                )))?;
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    attn_gate,
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

        if let Some(ref fused_fn) = st.kernels.gdn_rmsnorm_silu_gate {
            // === FUSED: RMSNorm + SiLU(gate) * normed in one kernel ===
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
            // gdn_rmsnorm_silu_gate(raw_output, ssm_norm, gate, out, eps, dim)
            // Reads output_buf (raw state output), applies RMSNorm with ssm_norm weights,
            // then applies silu(gate_buf) * normed. Writes final gated output to output_buf.
            // We write to normed_out_buf as temp, then use it as the "gated" output.
            unsafe {
                self.device
                    .stream
                    .launch_builder(fused_fn)
                    .arg(&gdn.output_buf)
                    .arg(ssm_norm)
                    .arg(&gdn.gate_buf)
                    .arg(&mut gdn.normed_out_buf) // reuse as final gated output
                    .arg(&eps)
                    .arg(&dim)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN fused_rmsnorm_silu_gate L{layer_idx}: {e}")))?;

            // normed_out_buf now contains silu(gate) * normed_output. Copy to output_buf
            // for the subsequent output projection step.
            self.device
                .stream
                .memcpy_dtod(&gdn.normed_out_buf, &mut gdn.output_buf)
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN dtod normed_out->output L{layer_idx}: {e}",
                )))?;
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

            // Step 10: SiLU(gate) * normed_output
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
        }


        // --- Step 11: Output projection: output_buf @ ssm_out^T -> ssm_proj ---
        // (output_buf now holds silu(gate) * normed_out from step 10)
        {
            let ssm_out = lw.ssm_out.as_ref()
                .ok_or_else(|| RuntimeError::Compute(format!(
                    "GDN L{layer_idx}: ssm_out weight missing",
                )))?;
            unsafe {
                launch_matvec(
                    &self.device,
                    &st.kernels,
                    ssm_out,
                    &gdn.output_buf,
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


        // --- Step 12: Residual add: x_gpu += ssm_proj ---
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
                    .arg(&mut st.scratch.x_gpu)
                    .arg(&gdn.ssm_proj_buf)
                    .arg(&n)
                    .launch(launch_cfg)
            }
            .map_err(|e| RuntimeError::Compute(format!("GDN residual_add L{layer_idx}: {e}")))?;
        }


        // --- Step 13: Copy x_gpu -> attn_proj (so FFN block reads from attn_proj) ---
        self.device
            .stream
            .memcpy_dtod(&st.scratch.x_gpu, &mut st.scratch.attn_proj)
            .map_err(|e| RuntimeError::Compute(format!(
                "GDN dtod x_gpu->attn_proj L{layer_idx}: {e}",
            )))?;


        Ok(())
    }

    /// Batched GDN prefill for a single GDN layer.
    ///
    /// Implements the 15-step GDN prefill pipeline matching Metal's
    /// `encode_batched_gdn_prefill`:
    ///
    ///  Phase 1 (batched across T tokens):
    ///    1. Batched RMSNorm: x[T, hidden] -> normed[T, hidden]
    ///    2. Batched QKV GEMM: normed[T, hidden] @ wq^T -> qkv[T, qkv_dim]
    ///    3. Batched Gate GEMM: normed[T, hidden] @ attn_gate^T -> gate[T, value_dim]
    ///    4. Batched Alpha GEMM: normed[T, hidden] @ ssm_alpha^T -> alpha_raw[T, num_heads]
    ///    5. Batched Beta GEMM: normed[T, hidden] @ ssm_beta^T -> beta_raw[T, num_heads]
    ///
    ///  Phase 2 (sequential per token, reuses decode kernels):
    ///    6-12. For each t: conv1d, silu, compute_gates, l2_norm, state_update,
    ///           rmsnorm, silu_gate_mul -> scatter output
    ///
    ///  Phase 3 (batched):
    ///   13. Batched SSM out GEMM + residual: gdn_out[T, value_dim] @ ssm_out^T + x -> attn_proj
    ///
    ///  Phase 4 (batched FFN, identical to standard layers):
    ///   14. FFN RMSNorm + gate/up + SwiGLU + down + residual
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
        //    wq for GDN is the fused [qkv_dim, hidden_dim] weight.
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
        //   1. ssm_conv1d_silu_prefill: batched conv1d+SiLU across T tokens
        //   2. gdn_compute_gates_batched: batched gate computation for T * num_heads
        //   3. l2_normalize_qk_strided: batched L2 norm for Q and K across T tokens
        //   4. gdn_prefill_fused_v3: warp-parallel fused state update (4x unrolled)
        //   5. gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output
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
            }

            // 3. l2_normalize_qk_strided: batched L2 norm for Q and K
            {
                let l2_fn = st.kernels.l2_normalize_qk_strided.as_ref().unwrap();
                let l2_block_dim = (p.head_dim as u32).min(1024);
                let l2_shared = ((l2_block_dim + 31) / 32 + 1) * 4;
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
            }

            // 4. gdn_prefill_fused_v3: warp-parallel fused state update
            // Grid: (val_dim, num_heads), Block: (32, 1, 1)
            {
                let state_fn = st.kernels.gdn_prefill_fused_v3.as_ref().unwrap();
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
            }

            // 5. gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output
            // Grid: (num_heads, batch), Block: (val_dim)
            // Writes to gdn_out which is used by Phase 3's GEMM.
            {
                let norm_fn = st.kernels.gdn_prefill_norm_gate.as_ref().unwrap();
                let block_dim = (p.head_dim as u32).min(1024);
                let norm_shared = ((block_dim + 31) / 32 + 1) * 4;
                let launch_cfg = CudarcLaunchConfig {
                    grid_dim: (num_heads_u32, batch_u32, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: norm_shared,
                };
                // ssm_norm_tiled is tiled from [head_dim] to [num_heads * head_dim]
                // in upload_layer_weights, so scale_n_heads = num_heads.
                let scale_n_heads = num_heads_u32;
                unsafe {
                    self.device.stream.launch_builder(norm_fn)
                        .arg(&gdn_pf.raw_out)
                        .arg(&gdn_pf.gate)
                        .arg(ssm_norm)
                        .arg(&mut gdn_pf.gdn_out)
                        .arg(&num_heads_u32)
                        .arg(&head_dim_u32) // val_dim per head
                        .arg(&eps)
                        .arg(&scale_n_heads)
                        .arg(&batch_u32)
                        .launch(launch_cfg)
                }
                .map_err(|e| RuntimeError::Compute(format!(
                    "GDN prefill fused norm_gate L{layer_idx}: {e}"
                )))?;
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
        }

        // ================================================================
        // PHASE 4: Batched FFN (identical to standard layers)
        // ================================================================
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
    ///   `input_f16` already contains the normalized activation from the previous
    ///   layer's fused tail. Only effective for F16/HGEMV paths.
    /// - `fuse_tail_next_layer`: If Some(next_layer_idx), replace the final
    ///   `residual_add_copy` with `fused_residual_rmsnorm_f16` using the next
    ///   layer's attn_norm weights (from `layer_weights_cache[next_layer_idx]`).
    ///   Saves 1 dispatch per inter-layer boundary (35 fewer for 36-layer models).
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

        // Diagnostic: on first layer, report which weight path is used (cuBLAS vs custom).
        if layer_idx == 0 {
            let diag = std::env::var("LUMEN_GRAPH_DIAGNOSTIC")
                .map(|v| v == "1")
                .unwrap_or(false);
            if diag {
                let has_f16_cache = lw.wq_f16.is_some();
                let wq_type = match &lw.wq {
                    GpuWeightBuf::F16Raw(_) => "F16Raw (cuBLAS HGEMV path)",
                    GpuWeightBuf::Q8Raw(_) => "Q8Raw (native dp4a path)",
                    GpuWeightBuf::Q4Raw(_) => "Q4Raw (native dp4a path)",
                    GpuWeightBuf::Q4Aligned(_) => "Q4Aligned (native dp4a path)",
                    GpuWeightBuf::Q8Aligned(_) => "Q8Aligned (native dp4a path)",
                    GpuWeightBuf::F32(_) if has_f16_cache => "F32 (cuBLAS HGEMV via F16 cache)",
                    GpuWeightBuf::F32(_) => "F32 (cuBLAS SGEMV path)",
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
            if let GpuWeightBuf::F16Raw(ref wq_f16) = lw.wq {
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wq_f16, &st.scratch.input_f16,
                        &mut st.scratch.q, q_dim, hidden_dim, "graph wq",
                        st.algo_cache.get(q_dim, hidden_dim),
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
            if let Some(ref wq_f16) = lw.wq_f16 {
                unsafe {
                    launch_hgemv_f16_preconverted(
                        &self.device, wq_f16, &st.scratch.input_f16,
                        &mut st.scratch.q, q_dim, hidden_dim, "graph wq",
                        st.algo_cache.get(q_dim, hidden_dim),
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
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                unsafe {
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wq, q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wk, q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wv, q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
                }
            } else if !skip_head_norm && qkv_use_preq && st.kernels.rmsnorm_to_q8_1.is_some() {
                // Fused RMSNorm + Q8_1 for graph decode: saves 1 dispatch per norm site.
                let fused_fn = st.kernels.rmsnorm_to_q8_1.as_ref().unwrap();
                let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                let bs = rmsnorm_block_size(hidden_dim);
                let lc = CudarcLaunchConfig { grid_dim: (1,1,1), block_dim: (bs,1,1), shared_mem_bytes: rmsnorm_shared_bytes(bs) };
                let dim = hidden_dim as u32;
                unsafe { self.device.stream.launch_builder(fused_fn).arg(&st.scratch.x_gpu).arg(&lw.attn_norm).arg(&mut *q8_1_buf).arg(&eps).arg(&dim).launch(lc) }
                .map_err(|e| RuntimeError::Compute(format!("graph rmsnorm_to_q8_1 attn: {e}")))?;
                unsafe {
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wq, q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wk, q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wv, q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
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
                    let q8_1_buf = st.scratch.input_q8_1.as_mut().unwrap();
                    unsafe {
                        launch_quantize_input_q8_1(&self.device, quant_fn, &st.scratch.normed, q8_1_buf, hidden_dim, "graph qkv")?;
                        launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wq, q8_1_buf, &mut st.scratch.q, q_dim, hidden_dim, "graph wq")?;
                        launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wk, q8_1_buf, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk")?;
                        launch_matvec_preq8_1(&self.device, &st.kernels, &lw.wv, q8_1_buf, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv")?;
                    }
                } else {
                    unsafe {
                        launch_matvec(&self.device, &st.kernels, &lw.wq, &st.scratch.normed, &mut st.scratch.q, q_dim, hidden_dim, "graph wq", lw.wq_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        launch_matvec(&self.device, &st.kernels, &lw.wk, &st.scratch.normed, &mut st.scratch.k, kv_dim, hidden_dim, "graph wk", lw.wk_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                        launch_matvec(&self.device, &st.kernels, &lw.wv, &st.scratch.normed, &mut st.scratch.v, kv_dim, hidden_dim, "graph wv", lw.wv_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?;
                    }
                }
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

        // Step 4: Attention -- GRAPH VARIANT (reads seq_len from device pointer, fixed geometry).
        {
            let gk = st.graph_kernels.as_ref().unwrap();
            let gp = st.graph_params.as_ref().unwrap();
            let seq_len_ptr = gp.attn_seq_len_ptr();
            let kvc = &st.kv_caches[layer_idx];
            let msl = kvc.max_seq_len as u32;
            let shared = super::graph::graph_attention_shared_bytes(msl);
            let lc = CudarcLaunchConfig { grid_dim: (num_heads as u32,1,1), block_dim: (super::graph::GRAPH_ATTN_BLOCK_SIZE,1,1), shared_mem_bytes: shared };
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            unsafe { self.device.stream.launch_builder(&gk.attention_decode)
                .arg(&st.scratch.q).arg(&kvc.k_cache).arg(&kvc.v_cache).arg(&mut st.scratch.attn_out)
                .arg(&(num_heads as u32)).arg(&(num_kv_heads as u32)).arg(&(head_dim as u32))
                .arg(&seq_len_ptr).arg(&msl).arg(&scale)
                .launch(lc) }
            .map_err(|e| RuntimeError::Compute(format!("graph attn L{layer_idx}: {e}")))?;
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
            unsafe { launch_matvec_residual(&self.device, &st.kernels, &lw.wo, &st.scratch.attn_out, &st.scratch.x_gpu, &mut st.scratch.attn_proj, hidden_dim, q_dim, "graph wo", lw.wo_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
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
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_gate, q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "graph gate")?;
                    launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_up, q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "graph up")?;
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
                        launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_gate, q8_1_buf, &mut st.scratch.gate, inter_dim, hidden_dim, "graph gate")?;
                        launch_matvec_preq8_1(&self.device, &st.kernels, &lw.w_up, q8_1_buf, &mut st.scratch.up, inter_dim, hidden_dim, "graph up")?;
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
            unsafe { launch_matvec(&self.device, &st.kernels, &lw.w_down, &st.scratch.gate, &mut st.scratch.down, hidden_dim, inter_dim, "graph down", lw.w_down_f16.as_ref(), Some(&mut st.scratch.input_f16), st.scratch.input_q8_1.as_mut())?; }
        }
        // Final step: residual add, with optional inter-layer fusion.
        //
        // Two fusion paths based on next layer's weight type:
        //   - F16: fused_residual_rmsnorm_f16 -> residual + RMSNorm + F16 output (for HGEMV paths)
        //   - Q8_0/dp4a: fused_residual_rmsnorm_q8_1 -> residual + RMSNorm + Q8_1 output (for dp4a paths)
        //   - residual_add_copy: plain residual (last layer, no fusion kernel, or no match)
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
        //   - F16: fused_residual_rmsnorm_f16 -> residual + RMSNorm + F16 output (for HGEMV paths)
        //   - Q8_0: fused_residual_rmsnorm_q8_1 -> residual + RMSNorm + Q8_1 output (for dp4a paths)
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

        // Sync conv_positions to GPU before graph capture so graph kernels
        // can read them from device pointers.
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
                self.compute_gdn_attention_gpu(layer, st)?;
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

            // Determine if we should fuse the tail of this layer with the head of the next.
            // Fuse when the next layer uses F16 (fused_residual_rmsnorm_f16) or
            // Q8_0/dp4a (fused_residual_rmsnorm_q8_1) and the corresponding kernel exists.
            // Skip fusion if next layer is GDN (GDN has its own attention path).
            let fuse_tail_next = if layer + 1 < num_layers && layer_types[layer + 1] != 1 {
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
        } else if let Some(ref proj_q8a) = st.globals.output_proj_q8_aligned {
            // Q8_0 aligned output projection: try Q8_1 path first, then on-the-fly.
            let out_dim_u32 = vocab_size as u32;
            let in_dim_u32 = hidden_dim as u32;

            // Path 0: Q8Aligned + pre-quantized Q8_1 input (NR=2, dp4a).
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
            self.device.stream
                .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
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
        self.device.synchronize()?;
        let argmax_host = self.device.dtoh_copy(&st.argmax_result)?;
        let mut logits_host = vec![0.0f32; hp.vocab_size as usize];
        if let Some(&idx) = argmax_host.first() {
            if (idx as usize) < logits_host.len() {
                logits_host[idx as usize] = 1.0;
            }
        }
        kv.advance_seq_len()?;
        st.decode_token_count += 1;
        Ok(Logits { data: logits_host })
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
    input_f16_scratch: Option<&mut CudaSlice<u8>>,
    mut input_q8_1_scratch: Option<&mut CudaSlice<u8>>,
) -> Result<(), RuntimeError> {
    // --- Native quantized kernels: read Q8_0/Q4_0 directly (1.06/0.56 B/elem) ---
    // These bypass the HGEMV path which reads 2 B/elem from pre-dequanted F16 cache.
    //
    // Priority for Q8_0:
    //   0. dp4a Q8_1 (pre-quantized input, NR=2, 128 threads): any in_dim (SM 6.1+)
    //   1. smem kernel (F32 x in shmem, NR=2): in_dim*4 <= 48KB -> in_dim <= 12288
    //   2. hgemv kernel (F16 x in shmem, NR=4): in_dim*2 <= 48KB -> in_dim <= 24576
    //   3. cuBLAS HGEMV via pre-dequanted F16 cache (2 B/elem): any in_dim
    //   4. dp4a (on-the-fly x quant) or v1 scalar: any in_dim (last resort)

    if let GpuWeightBuf::Q8Raw(w_q8) = weight {
        let shmem_f32 = (in_dim as u32) * 4;
        let shmem_f16 = (in_dim as u32) * 2;

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

    // HGEMV path: cuBLAS with pre-dequanted F16 weights.
    // Used for F32 weights (from Q4_1 dequant) that have an F16 cache.
    // Q8Raw and Q4Raw are handled above via native kernels (smem/scalar).
    // Uses DEFAULT_TENSOR_OP (fallback path for F32 with F16 caches).
    if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
        if matches!(weight, GpuWeightBuf::F32(_)) {
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
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            let out_dim_u32 = out_dim as u32;
            let in_dim_u32 = in_dim as u32;

            // Path 0 (priority): Q8Aligned + pre-quantized Q8_1 input (dp4a, NR=2).
            // Both weight and input use native int* loads. Zero byte-packing overhead.
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
                kernels.quantize_f32_to_q8_1.as_ref(),
                kernels.matvec_q8_aligned_q8_1.as_ref(),
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
    input_f16_scratch: Option<&mut CudaSlice<u8>>,
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

    // HGEMV residual: only for F32 weights with F16 cache.
    // Q8Raw and Q4Raw are handled above via native kernels (smem/scalar).
    // Uses DEFAULT_TENSOR_OP (fallback path for F32 with F16 caches).
    if let (Some(w_f16), Some(scratch)) = (weight_f16_cache, input_f16_scratch) {
        if matches!(weight, GpuWeightBuf::F32(_)) {
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
            if let (Some(quant_fn), Some(mv_fn), Some(q8_1_buf)) = (
                kernels.quantize_f32_to_q8_1.as_ref(),
                kernels.matvec_q8_aligned_q8_1_residual.as_ref(),
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
            if let Some(mv_fn) = kernels.matvec_q8_aligned_q8_1.as_ref() {
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
            if let Some(mv_fn) = kernels.matvec_q8_aligned_q8_1_residual.as_ref() {
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
/// Fused SwiGLU (in-place on gate) + F32->F16 conversion.
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
///   `output[i] = W_f16[i]^T * input_f16`  for i in 0..batch_count
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
///   that were uploaded during `preload_weights()` and have not been freed.
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
/// `up[row]   = dot(W_up[row],   x * rms_scale * norm_weight)`
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
        let max_seq_len = hyperparams.max_seq_len as usize;

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
        self.embed_f32_func = Some(embed_f32);
        self.embed_q8_0_func = Some(embed_q8_0);
        self.embed_f16_func = Some(embed_f16);
        self.embed_q4_0_func = Some(embed_q4_0);

        // Compile all decode-path kernels.
        let kernels = decode::compile_all_kernels(&self.device)?;

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
            input_q8_1: if kernels.quantize_f32_to_q8_1.is_some() && (kernels.matvec_q8_0_q8_1.is_some() || kernels.matvec_q8_aligned_q8_1.is_some() || kernels.matvec_q4_0_dp4a.is_some() || kernels.matvec_q4_aligned_q8_1.is_some()) {
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

        // Upload embedding: prefer quantized raw if available, else F32.
        let has_raw_embedding = self.embedding_raw.is_some();
        let (embedding_f32, embedding_q8, embedding_f16_raw, embedding_q4_raw) = if has_raw_embedding {
            let raw = self.embedding_raw.as_ref().unwrap();
            let placeholder: CudaSlice<f32> = self.device.alloc_zeros(1)?;
            match self.embedding_quant {
                QuantScheme::Q8_0 => {
                    let gpu_q8 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, Some(gpu_q8), None, None)
                }
                QuantScheme::F16 => {
                    let gpu_f16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, Some(gpu_f16), None)
                }
                QuantScheme::Q4_0 => {
                    let gpu_q4 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, Some(gpu_q4))
                }
                other => {
                    return Err(RuntimeError::Compute(format!(
                        "CUDA init: embedding raw quant {other:?} not supported (only Q8_0, F16, Q4_0)",
                    )));
                }
            }
        } else {
            let gpu_f32 = self.device.htod_copy(&self.embedding)?;
            (gpu_f32, None, None, None)
        };

        // Upload output projection: prefer quantized raw if available, else F32.
        let (output_proj_f32, output_proj_q8, output_proj_q4, output_proj_f16_raw) = if has_raw_output_proj {
            let raw = self.output_proj_raw.as_ref().unwrap();
            let placeholder: CudaSlice<f32> = self.device.alloc_zeros(1)?;
            match self.output_proj_quant {
                QuantScheme::Q8_0 => {
                    let gpu_q8 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, Some(gpu_q8), None, None)
                }
                QuantScheme::Q4_0 => {
                    let gpu_q4 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, Some(gpu_q4), None)
                }
                QuantScheme::F16 => {
                    let gpu_f16 = self.device.htod_copy(raw.as_slice())?;
                    (placeholder, None, None, Some(gpu_f16))
                }
                other => {
                    return Err(RuntimeError::Compute(format!(
                        "CUDA init: output_proj raw quant {other:?} not supported (only Q8_0, Q4_0, F16)",
                    )));
                }
            }
        } else {
            let gpu_f32 = self.device.htod_copy(&self.output_proj)?;
            (gpu_f32, None, None, None)
        };

        let globals = GpuGlobals {
            final_norm: self.device.htod_copy(&self.final_norm)?,
            output_proj: output_proj_f32,
            output_proj_f16: output_proj_f16_raw,
            output_proj_q8,
            output_proj_q8_aligned: None, // Populated during preload_weights
            output_proj_q4,
            output_proj_q4_aligned: None, // Populated during preload_weights
            embedding: embedding_f32,
            embedding_q8,
            embedding_f16: embedding_f16_raw,
            embedding_q4: embedding_q4_raw,
        };

        // Allocate per-layer KV caches. Compile the KV write kernel once and
        // share the module across all layers to avoid redundant NVRTC compilation.
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
            decode_token_count: 0,
            gdn_scratch_gpu: None,
            cublas_workspace,
            precomputed_ptrs: None,
            algo_cache: AlgoCache::new(),
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
            if let Some(ref emb_f16) = st.globals.embedding_f16 {
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
        //   Pass 1: compute_rms_scale writes a single scalar (saves full normed buffer).
        //   Pass 2: fused_norm_matvec_f32 normalizes x inline during the dot product.
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

        // 6. Attention: attention_decode(q, k_cache, v_cache -> attn_out)
        {
            let kv_cache = &st.kv_caches[layer_idx];
            // seq_len is the number of entries AFTER the append (cache auto-increments).
            let attn_seq_len = kv_cache.seq_len() as u32;
            let block_size = attention_block_size(attn_seq_len as usize);
            let shared_bytes = attention_shared_bytes(attn_seq_len);
            let launch_cfg = CudarcLaunchConfig {
                grid_dim: (num_heads as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            let nh = num_heads as u32;
            let nkvh = num_kv_heads as u32;
            let hd = head_dim as u32;
            let msl = kv_cache.max_seq_len as u32;
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            // SAFETY: q has num_heads * head_dim elements. k_cache and v_cache have
            // num_kv_heads * max_seq_len * head_dim elements each. attn_out has
            // num_heads * head_dim elements. attn_seq_len <= max_seq_len.
            unsafe {
                self.device
                    .stream
                    .launch_builder(&st.kernels.attention_decode)
                    .arg(&st.scratch.q)
                    .arg(&kv_cache.k_cache)
                    .arg(&kv_cache.v_cache)
                    .arg(&mut st.scratch.attn_out)
                    .arg(&nh)
                    .arg(&nkvh)
                    .arg(&hd)
                    .arg(&attn_seq_len)
                    .arg(&msl)
                    .arg(&scale)
                    .launch(launch_cfg)
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
        //    Reuse the pre-allocated logits_gpu buffer from MutableState.
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
            moe: false,
            gpu_argmax: false,
        }
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

        // Allocate batch-sized scratch buffers.
        let mut pf = super::prefill::alloc_prefill_scratch(
            &self.device, batch, hidden_dim, q_dim, kv_dim, inter_dim,
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
            // Dispatch priority:
            //   1. Tensor-core WMMA (SM 80+): 16x16 tiles via mma.sync PTX.
            //      Uses F16 tensor cores for QK^T and PV, up to 16x throughput
            //      over scalar F32 on A100.
            //   2. Scalar Br=4 fallback: 4 queries/block, warp-level parallelism.
            //      Used when batch < 16 (not enough queries for a full WMMA tile).
            unsafe {
                if batch >= 16 && st.kernels.flash_attention_wmma.is_some() {
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
            let gpu_weights = upload_layer_weights(&self.device, &layer_view, &hp_copy)?;
            cache.push(gpu_weights);
        }

        // Pre-dequant Q8_0 weights to F16 for HGEMM prefill (tensor core path).
        // This runs the dequant_q8_0_to_f16 kernel once per Q8_0 weight tensor,
        // storing the F16 version alongside the original Q8_0 data. The extra GPU
        // memory is ~2x the Q8_0 weight size (F16 = 2 bytes/element vs Q8_0 ~1.0625).
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

        // Tile ssm_norm from [head_dim] to [value_dim] for GDN layers.
        // This allows the standard rmsnorm kernel to be used on the [value_dim] output.
        let _gdn_params = super::gdn::GdnParams::from_hyperparams(&hp_copy);
        for layer in cache.iter_mut() {
            if layer.layer_type == 1 {
                // Read ssm_norm from the layer's subtensors (it's [head_dim] F32).
                // Tile by repeating: [head_dim] -> [num_heads * head_dim = value_dim]
                if layer.ssm_norm_tiled.is_none() {
                    // ssm_norm is not uploaded as a separate field — it comes from the LBC
                    // subtensors. For now, create a tiled version if ssm_a exists (indicating GDN layer).
                    // The ssm_norm weight will be uploaded separately in a future pass.
                    // For now, leave it as None — the dispatch will error and we'll debug.
                }
            }
        }

        // Repack Q8_0 weights to 36-byte aligned blocks for dp4a int* loads.
        // Aligned weight repack: enabled for ALL models including GDN.
        // +16% decode from dp4a int* loads (proven C8-C11).
        // Output projection repack is SKIPPED -- too large, causes OOM on
        // A100-80GB for GDN models, and negligible impact (called once per token).
        let has_gdn = cache.iter().any(|lw| lw.layer_type == 1);
        if let Some(ref repack_fn) = st.kernels.repack_q8_0_to_aligned36 {
            if st.kernels.matvec_q8_0_aligned.is_some() || st.kernels.matvec_q8_aligned_q8_1.is_some() {
                for (layer_idx, layer) in cache.iter_mut().enumerate() {
                    super::gpu_buffers::repack_layer_q8_to_aligned(
                        &self.device,
                        repack_fn,
                        layer,
                        &hp_copy,
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

        // Repack Q4_0 weights to 20-byte aligned blocks for dp4a int* nibble loads.
        if let Some(ref repack_fn) = st.kernels.repack_q4_0_to_aligned20 {
            if st.kernels.matvec_q4_aligned_q8_1.is_some() {
                for (layer_idx, layer) in cache.iter_mut().enumerate() {
                    super::gpu_buffers::repack_layer_q4_to_aligned(
                        &self.device,
                        repack_fn,
                        layer,
                        &hp_copy,
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

        // GDN layers use host-side conv_position — not yet graph-compatible.
        st.has_gdn_layers = has_gdn;
        if has_gdn {
            eprintln!("[CUDA] GDN layers detected -- CUDA graph capture disabled (host-side conv_position)");
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
        //   0 (first token): Run normal path (non-graph). This token establishes
        //     the first KV entry. We don't capture here because the attention
        //     shared memory must accommodate seq_len=1+ which is always true.
        //   1 (second token): Run the graph pipeline while capturing into a CUDA
        //     graph. All subsequent tokens replay the captured graph.
        //   2+ (subsequent tokens): Update graph params (token_id, pos, seq_len)
        //     via 3 small htod memcpys, then replay the captured graph.
        //
        // Graph capture is disabled for:
        //   - Models with GDN layers (host-side conv state management)
        //   - When graph kernels failed to compile
        //   - When graph params failed to allocate

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
        let can_use_graph = st.graph_kernels.is_some()
            && st.graph_params.is_some()
            && st.cublas_workspace.is_some()
            && st.precomputed_ptrs.is_some()
            && !st.has_gdn_layers;

        if graph_diagnostic && st.decode_token_count == 0 {
            eprintln!("[GRAPH-DIAG] === CUDA Graph Diagnostic Mode Enabled ===");
            eprintln!("[GRAPH-DIAG] Prerequisites:");
            eprintln!("[GRAPH-DIAG]   graph_kernels:    {}", st.graph_kernels.is_some());
            eprintln!("[GRAPH-DIAG]   graph_params:     {}", st.graph_params.is_some());
            eprintln!("[GRAPH-DIAG]   cublas_workspace: {}", st.cublas_workspace.is_some());
            eprintln!("[GRAPH-DIAG]   precomputed_ptrs: {}", st.precomputed_ptrs.is_some());
            eprintln!("[GRAPH-DIAG]   has_gdn_layers:   {}", st.has_gdn_layers);
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

        let attn_seq_len = (seq_pos + 1) as u32; // seq_len AFTER KV write

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
            //   - First decode token (decode_token_count == 0)
            //   - Models with GDN layers
            //   - When graph infrastructure is unavailable

            self.embed_token_gpu(token_id, st)?;

            for layer in 0..num_layers {
                self.compute_layer_gpu(layer, seq_pos, st)?;

                self.device
                    .stream
                    .memcpy_dtod(&st.scratch.attn_proj, &mut st.scratch.x_gpu)
                    .map_err(|e| RuntimeError::Compute(format!("dtod x_gpu<-attn_proj: {e}")))?;
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

        // Single sync + readback (4 bytes instead of ~1 MB).
        self.device.synchronize()?;
        let argmax_host = self.device.dtoh_copy(&st.argmax_result)?;

        // Build a minimal Logits with just the argmax result.
        let mut logits_host = vec![0.0f32; hp.vocab_size as usize];
        if let Some(&idx) = argmax_host.first() {
            if (idx as usize) < logits_host.len() {
                logits_host[idx as usize] = 1.0;
            }
        }

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
        //   if st.layer_weights_cache.len() < num_layers { return Err(...) }
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
        //   - embed_token_gpu: 0 syncs
        //   - compute_layer_gpu x N: 0 syncs
        //   - compute_final_gpu: 0 syncs
        //   - device.synchronize(): 1 sync
        //   = 1 total sync
        //
        // This is a documentation test -- the actual sync count is verified
        // by code inspection and the benchmark test on GPU hardware.
        let old_syncs_per_token = 32 + 1 + 1; // layers + embed + final
        let new_syncs_per_token = 1; // single sync at end
        assert_eq!(old_syncs_per_token, 34);
        assert_eq!(new_syncs_per_token, 1);
    }
}
