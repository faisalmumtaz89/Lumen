//! Metal GPU F32 compute backend for Apple Silicon.
//!
//! Implements `ComputeBackend` using Metal GPU compute shaders. On Apple Silicon
//! unified memory, weight data from mmap is already in GPU-accessible memory,
//! enabling zero-copy weight access via `MTLBuffer(bytesNoCopy:)`.
//!
//! Decode path: each `compute_layer` call encodes and executes GPU commands per
//! layer (async commit). Prefill path: ALL layers are encoded into a SINGLE
//! Metal command buffer with one commit_and_wait() at the end, eliminating
//! N-1 GPU-CPU sync barriers.
//!
//! # Performance characteristics
//!
//! - Matrix-vector multiply: GPU-parallelized across output rows
//! - RMSNorm: SIMD group reductions for fast sum-of-squares
//! - Attention: Scores computed in parallel, softmax on GPU, value accumulation parallel
//! - Activation buffers: Metal shared-mode buffers (CPU/GPU zero-copy)

pub(crate) mod ffi;
pub(crate) mod shaders;
pub(crate) mod io;
mod decode_greedy;
mod decode_single_cb;
mod gdn;
mod gpu_resident;
mod moe;
mod pipelines;
mod prefill;
mod prefill_encode;
pub(crate) mod types;
mod backend_impl;
pub use types::*;

use crate::weight::cache::LayerView;
use crate::error::RuntimeError;
use crate::expert::cache::ExpertLfuCache;
use crate::expert::profiler::ExpertActivationProfiler;
use crate::expert::reader::ExpertReader;
use self::ffi::{
    MetalBuffer, MetalCommandQueue,
    MetalDevice, MTLSize,
};
use self::io::MetalIOQueue;
use lumen_format::quantization::QuantScheme;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Page size for alignment checks (4 KiB on all Apple Silicon).
const PAGE_SIZE: usize = 4096;

/// Async expert prefetch state for one-layer lookahead.
///
/// After layer N's router produces expert_ids, a background thread pre-reads
/// those same experts for layer N+1. If layer N+1's router selects the same
/// experts (common due to routing locality), the prefetch result is used
/// directly, avoiding synchronous disk I/O.
struct PrefetchState {
    /// The target layer index for which experts were prefetched.
    target_layer: usize,
    /// The expert IDs that were prefetched (retained for diagnostics).
    #[allow(dead_code)]
    expert_ids: Vec<u32>,
    /// Join handle resolving to prefetched expert data.
    /// Each result corresponds to the expert_ids in order.
    handle: std::thread::JoinHandle<Vec<(u32, Result<(Vec<u8>, lumen_format::index::ExpertSlice), crate::expert::reader::ExpertReaderError>)>>,
}


// ============================================================================
// MetalF32Backend
// ============================================================================

/// Metal GPU F32 compute backend.
///
/// Identical API to NaiveF32Backend and SimdF32Backend. The engine interacts
/// with it through `Box<dyn ComputeBackend>` with no awareness of the GPU.
pub struct MetalF32Backend {
    device: MetalDevice,
    queue: MetalCommandQueue,
    pipelines: Option<MetalPipelines>,

    // Global tensors (GPU buffers)
    embedding_buf: Option<MetalBuffer>,
    final_norm_buf: Option<MetalBuffer>,
    output_proj_buf: Option<MetalBuffer>,

    // Keep CPU copies for set_global_tensors / embed_token fallback
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    /// Raw output_proj bytes for Q8_0 GPU dispatch (avoids CPU dequant).
    output_proj_raw: Option<Vec<u8>>,
    /// Quantization scheme of the output_proj tensor.
    output_proj_quant: QuantScheme,
    /// Raw embedding bytes for Q8_0/Q4_0 GPU dequant kernels.
    embedding_raw: Option<Vec<u8>>,
    /// Quantization scheme of the embedding tensor.
    embedding_quant: QuantScheme,
    /// Whether output_proj shares embedding storage (weight tying).
    weight_tying: bool,

    scratch: Mutex<Option<MetalScratch>>,
    cached_hidden_dim: usize,
    cached_vocab_size: usize,

    // ====================================================================
    // MoE expert caching infrastructure
    // ====================================================================
    // Only active for MoE models in streaming mode (non-GPU-resident).
    // Records expert activation patterns and caches hot experts to avoid
    // redundant SSD reads on subsequent tokens.

    /// Expert activation profiler: tracks per-(layer, expert) activation counts.
    /// Initialized when the model has num_experts > 0.
    expert_profiler: Option<Mutex<ExpertActivationProfiler>>,

    /// LFU cache for expert weights: keeps hot experts in RAM.
    /// Checked before loading from disk in the streaming MoE decode path.
    expert_cache: Option<Mutex<ExpertLfuCache>>,

    /// Direct byte-range reader for individual expert weights from LBC file.
    /// Used on cache misses to load only the needed expert (not the full layer blob).
    expert_reader: Option<Mutex<ExpertReader>>,

    /// Path to the LBC model file (stored for ExpertReader initialization).
    lbc_path: Option<PathBuf>,

    /// Number of profiling tokens remaining before triggering cache warm-up.
    /// When this reaches 0, `warm_from_profile()` is called to pre-populate the
    /// expert cache with the hottest experts observed during the profiling phase.
    /// Uses AtomicUsize for interior mutability (called from &self methods).
    profiling_tokens_remaining: AtomicUsize,
    /// Number of top-K experts per layer to cache during warmup.
    profiling_top_k: usize,
    /// Whether cache warmup has been completed.
    /// Uses AtomicBool for interior mutability (called from &self methods).
    warmup_complete: AtomicBool,

    // Cache-conditional routing bias
    // ====================================================================

    /// Bias magnitude for cache-conditional routing. When > 0.0, cached experts
    /// receive a logit boost of `cache_bias_lambda` before softmax in the MoE
    /// router, nudging borderline selections toward already-cached experts.
    /// Default 0.0 (disabled). Set via `configure_routing_bias()`.
    cache_bias_lambda: f32,

    // ====================================================================
    // MoE I/O instrumentation
    // ====================================================================

    /// Bytes of expert data loaded from disk via ExpertReader (Tier 2 misses).
    expert_bytes_from_disk: AtomicU64,
    /// Bytes of expert data served from ExpertLfuCache (Tier 1 + Tier 2 hits).
    expert_bytes_from_cache: AtomicU64,
    /// Bytes of expert data accessed via full layer blob fallback (Tier 3).
    expert_bytes_from_blob: AtomicU64,

    // ====================================================================
    // Option A dispatch
    // ====================================================================

    /// When true, MoE decode dispatches only the top-K selected expert FFNs
    /// instead of all num_experts (Option B). In streaming mode, expert_ids are
    /// available CPU-side after synchronous router readback. In
    /// GPU-resident mode, a two-CB split per MoE layer achieves the same
    /// selective dispatch. Default false (opt-in via
    /// `configure_option_a(true)`).
    use_option_a: bool,

    // ====================================================================
    // Async expert prefetching
    // ====================================================================

    /// One-layer lookahead prefetch handle. After layer N's router produces
    /// expert_ids, a background thread pre-reads the same experts for layer N+1
    /// from disk. At layer N+1, the prefetch result is checked before falling
    /// back to synchronous load. Only active when use_option_a is true.
    ///
    /// The handle contains: (target_layer, expert_ids, join_handle).
    /// The join_handle resolves to Vec<(expert_id, Result<(Vec<u8>, ExpertSlice)>)>.
    prefetch_handle: Mutex<Option<PrefetchState>>,

    // ====================================================================
    // Router diagnostics
    // ====================================================================

    /// When true, router debug readback is active: after each decode token,
    /// expert_ids and expert_weights are read back for all MoE layers and
    /// stored in `router_debug_log`.
    router_debug_enabled: bool,

    /// Accumulated per-layer routing stats from decode tokens.
    /// Only populated when `router_debug_enabled` is true.
    router_debug_log: Mutex<Vec<RouterLayerStats>>,

    // ====================================================================
    // Metal IO command queue for direct NVMe-to-GPU DMA
    // ====================================================================

    /// Metal IO command queue for direct file-to-GPU DMA transfers.
    /// Available on Metal 3 (M2+) with macOS 13+. When present, streaming
    /// expert loading bypasses CPU memory and loads directly from NVMe SSD
    /// into the Metal buffer. Falls back to pread + blit when None.
    metal_io_queue: Option<MetalIOQueue>,
}

impl MetalF32Backend {
    /// Create a new Metal compute backend.
    ///
    /// Returns an error if Metal is not available.
    pub fn new() -> Result<Self, RuntimeError> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            RuntimeError::Compute("Metal GPU not available on this system".into())
        })?;

        let queue = device.new_command_queue().ok_or_else(|| {
            RuntimeError::Compute("Failed to create Metal command queue".into())
        })?;

        // Attempt to create a Metal IO command queue (Metal 3 / macOS 13+).
        // This enables direct NVMe-to-GPU DMA for streaming expert loading.
        let metal_io_queue = MetalIOQueue::new(&device);
        // MTLIOCommandQueue availability is observable via MetalF32Backend API if needed.

        Ok(Self {
            device,
            queue,
            pipelines: None,
            embedding_buf: None,
            final_norm_buf: None,
            output_proj_buf: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            output_proj_raw: None,
            output_proj_quant: QuantScheme::F32,
            embedding_raw: None,
            embedding_quant: QuantScheme::F32,
            weight_tying: false,
            scratch: Mutex::new(None),
            cached_hidden_dim: 0,
            cached_vocab_size: 0,
            expert_profiler: None,
            expert_cache: None,
            expert_reader: None,
            lbc_path: None,
            profiling_tokens_remaining: AtomicUsize::new(0),
            profiling_top_k: 0,
            warmup_complete: AtomicBool::new(false),
            cache_bias_lambda: 0.0,
            expert_bytes_from_disk: AtomicU64::new(0),
            expert_bytes_from_cache: AtomicU64::new(0),
            expert_bytes_from_blob: AtomicU64::new(0),
            use_option_a: false,
            prefetch_handle: Mutex::new(None),
            router_debug_enabled: false,
            router_debug_log: Mutex::new(Vec::new()),
            metal_io_queue,
        })
    }




    /// Returns whether expert cache warmup has been completed.
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete.load(Ordering::Relaxed)
    }

    /// Returns a snapshot of expert activation profiler statistics.
    /// Returns None if the model is not MoE or profiler is not initialized.
    pub fn expert_profiler_summary(&self) -> Option<crate::expert::profiler::ProfilerSummary> {
        self.expert_profiler.as_ref().map(|p| p.lock().unwrap().summary())
    }

    /// Returns a snapshot of expert cache statistics.
    /// Returns None if expert caching is not configured.
    pub fn expert_cache_stats(&self) -> Option<crate::expert::cache::CacheStats> {
        self.expert_cache.as_ref().map(|c| c.lock().unwrap().stats())
    }

    /// Returns cumulative MoE expert I/O byte counters.
    ///
    /// Returns `(bytes_from_disk, bytes_from_cache, bytes_from_blob)`:
    /// - `bytes_from_disk`: loaded via ExpertReader on cache miss (Tier 2)
    /// - `bytes_from_cache`: served from ExpertLfuCache (Tier 1 + Tier 2 hits)
    /// - `bytes_from_blob`: accessed via full layer blob fallback (Tier 3)
    pub fn expert_io_stats(&self) -> (u64, u64, u64) {
        (
            self.expert_bytes_from_disk.load(Ordering::Relaxed),
            self.expert_bytes_from_cache.load(Ordering::Relaxed),
            self.expert_bytes_from_blob.load(Ordering::Relaxed),
        )
    }

    /// Returns whether Metal IO DMA (MTLIOCommandQueue) is available.
    ///
    /// When true, streaming expert cache misses use direct NVMe-to-GPU DMA
    /// instead of pread + CPU copy.
    pub fn has_metal_io_queue(&self) -> bool {
        self.metal_io_queue.is_some()
    }

    /// Returns the accumulated router debug log and clears it.
    ///
    /// Each entry is a `RouterLayerStats` captured from one MoE layer during
    /// one decode token. The log contains entries for ALL MoE layers across
    /// ALL tokens decoded since the last call to this method (or since init).
    pub fn get_router_debug_log(&self) -> Vec<RouterLayerStats> {
        let mut log = self.router_debug_log.lock().unwrap();
        std::mem::take(&mut *log)
    }


    /// Set raw Q8_0 output projection bytes for GPU-native dequant-matmul.
    ///
    /// When called, compute_final() will use the fused dequant_matmul_q8_0
    /// kernel instead of matmul_f32, reducing bandwidth 3.76x.
    /// The F32 `output_proj` from set_global_tensors is still needed for
    /// CPU-side embed_token (if output_proj is tied to embedding).
    pub fn set_output_proj_q8(&mut self, raw_bytes: Vec<u8>, quant: QuantScheme) {
        self.output_proj_quant = quant;
        self.output_proj_raw = Some(raw_bytes);
    }

    /// Get the device name (for diagnostics).
    pub fn device_name(&self) -> String {
        self.device.name()
    }

    /// Upload f32 data to a GPU buffer.
    fn upload_f32(&self, data: &[f32]) -> Result<MetalBuffer, RuntimeError> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.device.new_buffer_with_bytes(bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create Metal buffer".into())
        })
    }

    /// Create a zero-copy MetalBuffer wrapping the entire layer blob.
    ///
    /// On Apple Silicon, mmap'd data is in unified memory shared between CPU and GPU.
    /// `MTLBuffer(bytesNoCopy:)` wraps it without copying -- the GPU accesses the same
    /// physical pages. Subtensors within the blob are accessed via buffer offsets in
    /// `set_buffer(&buf, offset, index)`.
    ///
    /// # Page alignment
    ///
    /// `bytesNoCopy` requires page-aligned pointers (4096 bytes on Apple Silicon).
    /// mmap'd data is always page-aligned. If the pointer is NOT page-aligned
    /// (heap-allocated LayerView from async provider), we fall back to
    /// `new_buffer_with_bytes` which copies.
    fn create_layer_buffer(&self, weights: &LayerView) -> Result<MetalBuffer, RuntimeError> {
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = blob.len();

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Page-aligned: use bytesNoCopy (zero-copy).
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

            // SAFETY: The LayerView's backing memory (mmap) outlives this buffer.
            // The engine holds a borrow on &dyn WeightProvider during generate(),
            // which keeps the mmap alive. The buffer is used only within this
            // compute_layer call and dropped before returning.
            let buf = unsafe {
                self.device.new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
            // Fall through to copy path if bytesNoCopy fails (shouldn't happen
            // with page-aligned mmap, but defensive).
        }

        // Not page-aligned (heap data from async provider): copy.
        self.device.new_buffer_with_bytes(blob).ok_or_else(|| {
            RuntimeError::Compute("Failed to create layer buffer (copy fallback)".into())
        })
    }

    /// Create a Metal buffer covering only the non-expert portion of a
    /// MoE layer blob. This avoids page-faulting the expert byte range from mmap,
    /// since expert data will be served from the LFU cache instead.
    ///
    /// `non_expert_end` is the byte offset in the blob where expert data begins.
    /// The returned buffer covers `blob[0..non_expert_end]` (rounded up to page size).
    fn create_partial_layer_buffer(
        &self,
        weights: &LayerView,
        non_expert_end: usize,
    ) -> Result<MetalBuffer, RuntimeError> {
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = non_expert_end.min(blob.len());

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty partial layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            // Ensure we don't exceed the blob's total length (page-rounded).
            let max_aligned = (blob.len() + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let aligned_len = aligned_len.min(max_aligned);

            let buf = unsafe {
                self.device.new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
        }

        // Not page-aligned: copy only the non-expert portion.
        self.device.new_buffer_with_bytes(&blob[..len]).ok_or_else(|| {
            RuntimeError::Compute("Failed to create partial layer buffer (copy fallback)".into())
        })
    }

    /// Compute the byte offset where expert data begins in a MoE layer blob.
    ///
    /// Returns the end offset of the last non-expert tensor (attention weights,
    /// norms, router, biases). Everything before this offset is non-expert data;
    /// everything at or after it is expert data. If the layer has no experts,
    /// returns the full blob length.
    fn non_expert_byte_end(st: &lumen_format::index::SubtensorOffsets) -> usize {
        let mut end: u64 = 0;

        // Attention weights
        let slices = [&st.wq, &st.wk, &st.wv, &st.wo, &st.attn_norm, &st.ffn_norm];
        for s in &slices {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Dense FFN weights (zero-length sentinels for MoE, but check anyway)
        for s in &[&st.w_gate, &st.w_up, &st.w_down] {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Optional biases
        for opt in &[&st.bq, &st.bk, &st.bv] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Router weight (non-expert, always loaded)
        if let Some(ref s) = st.router_weight {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Shared expert weights (always loaded, non-expert).
        // Qwen3.5-MoE has an always-active shared expert whose gate/up/down
        // weights live in the layer blob alongside attention/norm/router data.
        for opt in &[&st.shared_expert_gate, &st.shared_expert_up, &st.shared_expert_down] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Extended attention fields (always loaded, non-expert).
        // attn_gate, attn_post_norm are per-layer tensors used by hybrid models.
        for opt in &[&st.attn_gate, &st.attn_post_norm] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // SSM / linear attention fields (always loaded, non-expert).
        // These are per-layer tensors for GatedDeltaNet hybrid layers.
        for opt in &[&st.ssm_a, &st.ssm_conv1d, &st.ssm_dt, &st.ssm_beta, &st.ssm_alpha, &st.ssm_norm, &st.ssm_out] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Per-head Q/K RMSNorm weights and shared expert gate input weight.
        for opt in &[&st.attn_q_norm, &st.attn_k_norm, &st.ffn_gate_inp_shexp] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        end as usize
    }

    /// Dispatch a matmul_bytes_f32 kernel: out = W_bytes * x
    ///
    /// Note: Not used by the optimized compute_layer (which inlines encoding into
    /// batched command buffers with zero-copy offsets), but retained for testing
    /// and for potential use by future code paths.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_bytes(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        // Create a buffer wrapping the weight bytes (copy for safety)
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch a dequant_matmul_q8_0 kernel: out = dequant(W_q8) * x
    ///
    /// The kernel performs fused Q8_0 dequantization and matrix-vector multiply.
    /// `in_dim` is the element count (not byte stride). The kernel computes the
    /// Q8_0 row byte stride internally from `in_dim`.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_q8_0(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for Q8_0 matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for Q8_0 matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for Q8_0 matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch the appropriate matmul kernel based on quantization scheme.
    ///
    /// For Q8_0 weights, uses the fused `dequant_matmul_q8_0` kernel.
    /// For F32/unquantized weights, uses `matmul_bytes_f32` (cast uchar* to float*).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_for_quant(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        quant: QuantScheme,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        match quant {
            QuantScheme::Q8_0 => {
                self.dispatch_matmul_q8_0(
                    pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch,
                )
            }
            _ => {
                self.dispatch_matmul_bytes(
                    pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch,
                )
            }
        }
    }

    /// Dispatch rmsnorm_bytes kernel.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rmsnorm_bytes(
        &self,
        pipelines: &MetalPipelines,
        x_buf: &MetalBuffer,
        w_bytes: &[u8],
        out_buf: &MetalBuffer,
        dim: usize,
        eps: f32,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for rmsnorm".into())
        })?;
        let dim_u32 = dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for rmsnorm".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for rmsnorm".into())
        })?;

        enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
        enc.set_buffer(x_buf, 0, 0);
        enc.set_buffer(&w_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&dim_u32.to_le_bytes(), 3);
        enc.set_bytes(&eps.to_le_bytes(), 4);
        enc.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(scratch.norm_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }
}



#[cfg(test)]
mod tests;
