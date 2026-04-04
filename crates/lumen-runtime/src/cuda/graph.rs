//! CUDA graph capture and replay for the decode pipeline.
//!
//! Captures the entire multi-layer decode kernel sequence into a CUDA graph
//! on the first token, then replays it on subsequent tokens. This eliminates
//! per-kernel launch overhead (~5 us x 15 kernels x 32 layers = 2.4 ms/token).
//!
//! # Design: Scalar parameter indirection
//!
//! CUDA graph capture records the exact kernel launch configuration (grid_dim,
//! block_dim, shared_mem_bytes) and all kernel arguments (device pointers and
//! scalar values). On replay, these must be identical -- but decode needs
//! token_id, seq_pos, and attn_seq_len to change every token.
//!
//! The solution: graph-compatible kernel variants (`*_graph` in graph_kernels.cu)
//! that read these per-token scalars from **device pointers** instead of scalar
//! arguments. The device pointers are fixed (same GPU buffer every time), but
//! their contents are updated via a small `memcpy_htod` BEFORE graph replay.
//! Since the memcpy happens outside the graph, the graph itself is structurally
//! identical across tokens.
//!
//! # Fixed-geometry attention
//!
//! The standard `attention_decode` kernel uses `block_dim = min(seq_len, 256)`
//! and `shared_mem = (8 + seq_len) * 4`, both varying per token. The graph
//! variant `attention_decode_graph` uses fixed `block_dim = 256` and
//! `shared_mem = (8 + max_seq_len) * 4`. Extra threads participate in
//! reductions with identity values but do no real work. The seq_len is read
//! from a device pointer.
//!
//! # When graphs help
//!
//! For quantized models (Q4_0/Q8_0) where kernel execution time is short,
//! launch overhead is a significant fraction of total time:
//! - 8B Q8_0: 15 kernels * 32 layers * 5 us = 2.4 ms / 19 ms = ~13%
//! - 1B Q8_0: 15 kernels * 22 layers * 5 us = 1.65 ms / 3.8 ms = ~43%
//! For F32 models the savings are marginal (~3%).
//!
//! # Limitations
//!
//! - Graph capture cannot include host-device synchronization
//! - Memory allocations during capture are not supported
//! - All scratch buffers must be pre-allocated
//! - The attention kernel's fixed max geometry wastes some shared memory
//!   (at most max_seq_len * 4 bytes per block, well within the 48 KB limit)
//! - Graph must be re-captured if max_seq_len changes (session reset)

// CUDA graph infrastructure -- actively used by decode_token for graph capture/replay.

use crate::error::RuntimeError;
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaFunction, CudaSlice};
use std::sync::Arc;

use super::ffi::CudaDevice;
use super::shaders;

// ---------------------------------------------------------------------------
// Graph-compatible kernel set
// ---------------------------------------------------------------------------

/// Compiled graph-compatible kernel functions.
///
/// These read per-token-varying scalars from device pointers instead of scalar
/// args, enabling CUDA graph capture. All other kernels (rmsnorm, matvec,
/// swiglu, residual_add) have no per-token-varying scalars and can be captured
/// directly from the standard KernelSet.
pub(crate) struct GraphKernelSet {
    /// Embedding lookup variants (read token_id from device pointer).
    pub(crate) embed_f32: CudaFunction,
    pub(crate) embed_q8_0: CudaFunction,
    pub(crate) embed_f16: CudaFunction,
    pub(crate) embed_q4_0: CudaFunction,
    /// RoPE variant (reads pos from device pointer).
    /// Superseded by `rope_kv_write` fused kernel; kept for unfused fallback.
    #[allow(dead_code)]
    pub(crate) rope_apply: CudaFunction,
    /// KV cache write variant (reads pos from device pointer).
    /// Superseded by `rope_kv_write` fused kernel; kept for unfused fallback.
    #[allow(dead_code)]
    pub(crate) kv_cache_write: CudaFunction,
    /// Attention decode variant (reads seq_len from device pointer, fixed geometry).
    pub(crate) attention_decode: CudaFunction,
    /// Fused RoPE + KV cache write (reads pos from device pointer).
    /// Combines rope_apply_graph + 2x kv_cache_write_graph into 1 kernel.
    pub(crate) rope_kv_write: CudaFunction,
    /// Fused F32->F16 conversion + residual copy (for HGEMV output projection).
    /// Combines f32_to_f16_vec + memcpy_dtod into 1 kernel.
    pub(crate) convert_f16_residual_copy: CudaFunction,
    /// GDN Conv1D decode variant (reads state_pos from device pointer).
    pub(crate) ssm_conv1d_decode: CudaFunction,
    /// Advance conv position on GPU (single-thread kernel).
    pub(crate) advance_conv_position: CudaFunction,
}

/// Compile all graph-compatible kernel variants.
pub(crate) fn compile_graph_kernels(device: &CudaDevice) -> Result<GraphKernelSet, RuntimeError> {
    let module = device.compile_and_load(shaders::GRAPH_KERNEL_SOURCE)?;

    let load = |name: &str| -> Result<CudaFunction, RuntimeError> {
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load graph kernel '{name}': {e}"))
        })
    };

    Ok(GraphKernelSet {
        embed_f32: load("embed_token_f32_graph")?,
        embed_q8_0: load("embed_token_q8_0_graph")?,
        embed_f16: load("embed_token_f16_graph")?,
        embed_q4_0: load("embed_token_q4_0_graph")?,
        rope_apply: load("rope_apply_graph")?,
        kv_cache_write: load("kv_cache_write_graph")?,
        attention_decode: load("attention_decode_graph")?,
        rope_kv_write: load("rope_kv_write_graph")?,
        convert_f16_residual_copy: load("convert_f32_to_f16_and_residual_copy")?,
        ssm_conv1d_decode: load("ssm_conv1d_decode_graph")?,
        advance_conv_position: load("advance_conv_position")?,
    })
}

// ---------------------------------------------------------------------------
// Graph parameter buffer
// ---------------------------------------------------------------------------

/// GPU-resident buffers holding per-token scalar parameters.
///
/// All three scalars are packed into a single contiguous `CudaSlice<u32>` of
/// 3 elements: `[token_id, seq_pos, attn_seq_len]`. Updated via ONE 12-byte
/// `memcpy_htod` call (~1.5 us) instead of three separate 4-byte calls (~4.5 us).
///
/// Graph-compatible kernels receive device pointers to individual elements
/// within this packed buffer (offsets 0, 1, 2). The pointers are baked into
/// the captured graph at capture time; only the VALUES change per token.
pub(crate) struct GraphParamsBuf {
    /// Packed GPU buffer: [token_id, seq_pos, attn_seq_len] (3 x u32 = 12 bytes).
    packed: CudaSlice<u32>,
}

impl GraphParamsBuf {
    /// Allocate the packed 3-element buffer on GPU (zeroed).
    pub(crate) fn new(device: &CudaDevice) -> Result<Self, RuntimeError> {
        Ok(Self {
            packed: device.alloc_zeros::<u32>(3)?,
        })
    }

    /// Update all three scalar parameters with a single 12-byte memcpy.
    ///
    /// One host-to-device memcpy (~1.5 us) instead of three (~4.5 us), executed
    /// BEFORE graph replay so the graph kernels see the updated values.
    pub(crate) fn update(
        &mut self,
        device: &CudaDevice,
        token_id: u32,
        seq_pos: u32,
        attn_seq_len: u32,
    ) -> Result<(), RuntimeError> {
        device.htod_copy_into(&[token_id, seq_pos, attn_seq_len], &mut self.packed)?;
        Ok(())
    }

    /// Device pointer to token_id scalar (element 0 of packed buffer).
    pub(crate) fn token_id_ptr(&self) -> cudarc::driver::CudaView<'_, u32> {
        self.packed.slice(0..1)
    }

    /// Device pointer to seq_pos scalar (element 1 of packed buffer).
    pub(crate) fn seq_pos_ptr(&self) -> cudarc::driver::CudaView<'_, u32> {
        self.packed.slice(1..2)
    }

    /// Device pointer to attn_seq_len scalar (element 2 of packed buffer).
    pub(crate) fn attn_seq_len_ptr(&self) -> cudarc::driver::CudaView<'_, u32> {
        self.packed.slice(2..3)
    }
}

// ---------------------------------------------------------------------------
// Captured graph
// ---------------------------------------------------------------------------

/// A captured CUDA graph that can be replayed to execute the full decode pipeline.
///
/// With graph-compatible kernels, the graph is valid for ALL sequence positions
/// (not just the seq_len at capture time). The only structural constraint is
/// num_layers and max_seq_len matching the model configuration at capture time.
pub(crate) struct CapturedGraph {
    /// The cudarc graph handle (owns CUgraph + CUgraphExec).
    graph: cudarc::driver::CudaGraph,
    /// Number of layers captured. Must match current model.
    captured_num_layers: usize,
    /// Max sequence length used for attention shared memory sizing.
    captured_max_seq_len: usize,
}

// SAFETY: CapturedGraph is only accessed through MutableState which is behind a
// Mutex. The Mutex guarantees single-threaded access, satisfying the CUDA graph's
// requirement that it not be accessed concurrently from multiple threads.
// CudaGraph internally holds an Arc<CudaStream> which is already Send.
unsafe impl Send for CapturedGraph {}
unsafe impl Sync for CapturedGraph {}

impl CapturedGraph {
    /// Check if this captured graph can be replayed for the given model config.
    ///
    /// With graph-compatible kernels, the graph is valid for ANY seq_pos as long
    /// as the structural parameters (num_layers, max_seq_len) match.
    pub(crate) fn is_valid_for(&self, num_layers: usize, max_seq_len: usize) -> bool {
        self.captured_num_layers == num_layers && self.captured_max_seq_len == max_seq_len
    }

    /// Replay the captured graph. All kernels execute with a single API call.
    ///
    /// The caller MUST update GraphParamsBuf with the correct token_id, seq_pos,
    /// and attn_seq_len BEFORE calling this.
    pub(crate) fn launch(&self) -> Result<(), RuntimeError> {
        self.graph.launch().map_err(|e| {
            RuntimeError::Compute(format!("CUDA graph launch failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// Graph capture helpers
// ---------------------------------------------------------------------------

/// Query the current stream capture status for diagnostic purposes.
///
/// Returns a human-readable string describing the capture state. This is
/// useful for debugging graph capture failures -- it confirms whether the
/// stream is in capturing mode, and if so, whether the capture is active
/// or has been invalidated.
pub(crate) fn query_capture_status(
    stream: &Arc<cudarc::driver::CudaStream>,
) -> String {
    match stream.capture_status() {
        Ok(status) => {
            match status {
                cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE =>
                    "NONE (not capturing)".to_string(),
                cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE =>
                    "ACTIVE (capturing)".to_string(),
                cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED =>
                    "INVALIDATED (capture broken by illegal operation)".to_string(),
            }
        }
        Err(e) => format!("capture_status() FAILED: {e}"),
    }
}

/// Begin capturing kernel launches on the given stream into a CUDA graph.
///
/// All kernel launches, cuBLAS calls, and device memcpys on this stream
/// between `begin_capture` and `end_capture` are recorded (not executed).
///
/// Uses `CU_STREAM_CAPTURE_MODE_RELAXED` for maximum compatibility.
/// With event tracking disabled in `CudaDevice::new()`, our single-stream
/// allocations and kernel launches produce no cross-stream dependencies.
/// cuBLAS internal stream usage may still benefit from relaxed mode.
pub(crate) fn begin_capture(
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<(), RuntimeError> {
    stream
        .begin_capture(cuda_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| RuntimeError::Compute(format!("CUDA graph begin_capture failed: {e}")))
}

/// End capturing and instantiate the graph for replay.
///
/// Returns `None` if the capture produced an empty graph (no kernels launched).
/// Returns the `CapturedGraph` with metadata for validity checking.
pub(crate) fn end_capture(
    stream: &Arc<cudarc::driver::CudaStream>,
    num_layers: usize,
    max_seq_len: usize,
) -> Result<Option<CapturedGraph>, RuntimeError> {
    let flags =
        cuda_sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(flags)
        .map_err(|e| RuntimeError::Compute(format!("CUDA graph end_capture failed: {e}")))?;

    Ok(graph.map(|g| CapturedGraph {
        graph: g,
        captured_num_layers: num_layers,
        captured_max_seq_len: max_seq_len,
    }))
}

// ---------------------------------------------------------------------------
// Fixed-geometry attention launch config
// ---------------------------------------------------------------------------

/// Block size for graph-compatible attention: always 256 (8 warps).
///
/// Unlike the standard `attention_block_size(seq_len)` which varies,
/// this is constant for graph capture compatibility.
pub(crate) const GRAPH_ATTN_BLOCK_SIZE: u32 = 256;

/// Shared memory bytes for graph-compatible attention with fixed max geometry.
///
/// Layout: 8 floats for warp reduction + max_seq_len floats for scores.
/// The kernel only accesses [0..seq_len-1] of the scores array, but the
/// allocation is fixed at max_seq_len for graph capture compatibility.
pub(crate) fn graph_attention_shared_bytes(max_seq_len: u32) -> u32 {
    // Cap at 48KB (A100 default shared memory limit = 49152 bytes).
    // With 8 floats for warp reduction, max effective seq_len = (49152/4 - 8) = 12280.
    let raw = (8 + max_seq_len) * 4;
    raw.min(49152)
}
