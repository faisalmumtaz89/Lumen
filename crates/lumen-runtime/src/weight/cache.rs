//! Unified cache manager API.
//!
//! The [`WeightProvider`] trait is the core abstraction through which the
//! execution engine accesses model weights. It supports prefetching,
//! non-blocking access, and explicit release hints -- all needed to
//! implement windowed prefetch and avoid the prefetch-release conflict
//! described in PRIMA.CPP.

use crate::error::RuntimeError;
use crate::storage::IoSnapshot;
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use std::fmt;
use std::sync::Arc;

/// Backing storage for layer data -- either owned or zero-copy mmap reference.
enum LayerBacking {
    /// Heap-allocated owned data (used by sync/async providers).
    Owned(Arc<[u8]>),
    /// Zero-copy reference into mmap. Uses raw pointer because the WeightProvider
    /// trait returns LayerView by value (no lifetime parameter).
    ///
    /// # Safety
    ///
    /// The mmap region MUST outlive all LayerViews. This is guaranteed by the
    /// engine's borrow of `&dyn WeightProvider` during `generate()`.
    ZeroCopy {
        ptr: *const u8,
        len: usize,
    },
}

// SAFETY: The raw pointer in ZeroCopy points to read-only mmap memory.
// The mmap outlives all LayerViews (enforced by engine borrowing WeightProvider).
unsafe impl Send for LayerBacking {}
unsafe impl Sync for LayerBacking {}

impl Clone for LayerBacking {
    fn clone(&self) -> Self {
        match self {
            LayerBacking::Owned(arc) => LayerBacking::Owned(Arc::clone(arc)),
            LayerBacking::ZeroCopy { ptr, len } => LayerBacking::ZeroCopy { ptr: *ptr, len: *len },
        }
    }
}

/// A read-only view into a layer's weight data currently resident in memory.
///
/// `LayerView` references contiguous bytes plus metadata describing how to
/// interpret the sub-tensors within. Cheaply cloneable (reference-counted
/// for owned data, pointer copy for zero-copy mmap).
impl LayerBacking {
    fn as_bytes(&self) -> &[u8] {
        match self {
            LayerBacking::Owned(arc) => arc,
            // SAFETY: Caller guarantees mmap outlives all LayerViews.
            LayerBacking::ZeroCopy { ptr, len } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            LayerBacking::Owned(arc) => arc.len(),
            LayerBacking::ZeroCopy { len, .. } => *len,
        }
    }
}

pub struct LayerView {
    data: LayerBacking,

    /// Layer index (0-based).
    pub layer_idx: usize,

    /// Sub-tensor layout within this blob.
    pub subtensors: SubtensorOffsets,
}

impl Clone for LayerView {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            layer_idx: self.layer_idx,
            subtensors: self.subtensors.clone(),
        }
    }
}

impl LayerView {
    /// Create a `LayerView` from an owned byte buffer.
    pub fn from_owned(layer_idx: usize, data: Vec<u8>, subtensors: SubtensorOffsets) -> Self {
        Self {
            data: LayerBacking::Owned(Arc::from(data)),
            layer_idx,
            subtensors,
        }
    }

    /// Create a zero-copy `LayerView` from a raw pointer into mmap memory.
    ///
    /// # Safety
    ///
    /// The caller MUST guarantee that the pointed-to memory remains valid
    /// for the lifetime of this `LayerView` and all its clones.
    pub unsafe fn from_mmap_ptr(
        layer_idx: usize,
        ptr: *const u8,
        len: usize,
        subtensors: SubtensorOffsets,
    ) -> Self {
        Self {
            data: LayerBacking::ZeroCopy { ptr, len },
            layer_idx,
            subtensors,
        }
    }

    /// Returns the raw bytes of the entire layer blob.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Returns the bytes for a specific sub-tensor within this layer.
    ///
    /// Returns an error if the slice offset+length overflows or exceeds blob bounds.
    pub fn subtensor_bytes(&self, slice: &TensorSlice) -> Result<&[u8], RuntimeError> {
        let bytes = self.data.as_bytes();
        let start = slice.offset as usize;
        let end = start.checked_add(slice.length as usize).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "subtensor offset {} + length {} overflows",
                slice.offset, slice.length,
            ))
        })?;
        if end > bytes.len() {
            return Err(RuntimeError::Compute(format!(
                "subtensor out of bounds: offset={}, length={}, blob_size={}",
                slice.offset, slice.length, bytes.len(),
            )));
        }
        Ok(&bytes[start..end])
    }

    /// Total byte size of this layer blob.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for LayerView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerView")
            .field("layer_idx", &self.layer_idx)
            .field("byte_len", &self.byte_len())
            .finish()
    }
}

/// Priority level for prefetch requests.
///
/// The cache manager uses priority to decide scheduling order when
/// multiple prefetch requests are outstanding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefetchPriority {
    Normal,
    High,
}

/// A handle to an in-flight prefetch operation.
///
/// Dropping the handle does NOT cancel the prefetch;
/// use [`PrefetchHandle::cancel`] explicitly.
pub struct PrefetchHandle {
    pub layer: usize,
    pub priority: PrefetchPriority,
    completed: bool,
    cancelled: bool,
}

impl PrefetchHandle {
    pub fn new(layer: usize, priority: PrefetchPriority) -> Self {
        Self {
            layer,
            priority,
            completed: false,
            cancelled: false,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn mark_complete(&mut self) {
        self.completed = true;
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}

impl fmt::Debug for PrefetchHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefetchHandle")
            .field("layer", &self.layer)
            .field("priority", &self.priority)
            .field("completed", &self.completed)
            .field("cancelled", &self.cancelled)
            .finish()
    }
}

/// Statistics from the weight cache, exposed for telemetry and auto-tuning.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub layers_cached: usize,
    pub bytes_cached: u64,
    pub capacity_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub inflight_prefetches: usize,
}

impl CacheStats {
    /// Cache hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Prefetch effectiveness rate as a fraction in [0.0, 1.0].
    pub fn prefetch_hit_rate(&self) -> f64 {
        let total = self.prefetch_hits + self.prefetch_misses;
        if total == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total as f64
        }
    }

    /// Cache utilization as a fraction in [0.0, 1.0].
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            0.0
        } else {
            self.bytes_cached as f64 / self.capacity_bytes as f64
        }
    }
}

/// The unified cache manager API (Section 6 of the design spec).
///
/// Abstracts over both mmap-based and async-read-based storage backends.
/// The execution engine interacts exclusively through this interface.
///
/// # Prefetch protocol
///
/// The pipeline scheduler calls [`prefetch_layer`] for upcoming layers.
/// The compute loop calls [`try_get_layer`] (non-blocking) or
/// [`get_layer_blocking`] (fallback). After a layer is no longer needed,
/// [`release_layer_hint`] signals that it can be evicted.
pub trait WeightProvider: Send + Sync {
    /// Begin prefetching the given layer into cache.
    ///
    /// Returns a handle that can be polled for completion. Multiple
    /// prefetch calls for the same layer are idempotent.
    fn prefetch_layer(
        &self,
        layer: usize,
        priority: PrefetchPriority,
    ) -> Result<PrefetchHandle, RuntimeError>;

    /// Block until the given layer is available, then return a view.
    fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError>;

    /// Get raw layer data without dequantization.
    ///
    /// Returns the layer's weight tensors in their original quantization format
    /// (Q8_0, Q4_0, F16, etc.) without converting to F32. Used by GPU backends
    /// that dispatch quantized kernels directly on the GPU.
    ///
    /// Default implementation falls back to `get_layer_blocking` (which may
    /// dequantize for CPU backends). Override in providers that can return raw data.
    fn get_layer_raw(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        self.get_layer_blocking(layer)
    }

    /// Non-blocking attempt to get a layer view.
    ///
    /// Returns `Some(view)` if the layer is currently cached.
    fn try_get_layer(&self, layer: usize) -> Option<LayerView>;

    /// Hint that the given layer may be evicted.
    fn release_layer_hint(&self, layer: usize);

    /// Signal the start of a new forward pass (one token).
    ///
    /// Providers that maintain per-token state (e.g. a compute cursor for
    /// windowed prefetch) should reset it here. Default is a no-op.
    fn begin_pass(&self) {}

    fn stats(&self) -> CacheStats;

    fn num_layers(&self) -> usize;

    /// Returns an I/O snapshot from the underlying storage backend.
    fn io_snapshot(&self) -> Option<IoSnapshot> { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_format::index::TensorSlice;
    use lumen_format::QuantScheme;

    fn dummy_subtensors() -> SubtensorOffsets {
        let s = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };
        SubtensorOffsets {
            wq: s, wk: s, wv: s, wo: s,
            bq: None, bk: None, bv: None,
            w_gate: s, w_up: s, w_down: s,
            attn_norm: s, ffn_norm: s,
            router_weight: None,
            experts: None,
            shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
            attn_gate: None, attn_post_norm: None,
            ssm_a: None, ssm_conv1d: None, ssm_dt: None,
            ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
            attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
            layer_type: None,
        }
    }

    #[test]
    fn layer_view_from_owned_and_byte_len() {
        let data = vec![1u8, 2, 3, 4];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        assert_eq!(view.byte_len(), 4);
        assert_eq!(view.layer_idx, 0);
    }

    #[test]
    fn layer_view_as_bytes() {
        let data = vec![10u8, 20, 30];
        let view = LayerView::from_owned(1, data, dummy_subtensors());
        assert_eq!(view.as_bytes(), &[10, 20, 30]);
    }

    #[test]
    fn layer_view_subtensor_bytes_valid() {
        let data = vec![0u8; 100];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        let slice = TensorSlice { offset: 10, length: 20, quant: QuantScheme::F32 };
        let bytes = view.subtensor_bytes(&slice).unwrap();
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn layer_view_subtensor_bytes_out_of_bounds() {
        let data = vec![0u8; 50];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        let slice = TensorSlice { offset: 40, length: 20, quant: QuantScheme::F32 };
        assert!(view.subtensor_bytes(&slice).is_err());
    }

    #[test]
    fn layer_view_subtensor_bytes_overflow() {
        let data = vec![0u8; 100];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        let slice = TensorSlice { offset: u64::MAX, length: 1, quant: QuantScheme::F32 };
        assert!(view.subtensor_bytes(&slice).is_err());
    }

    #[test]
    fn layer_view_subtensor_bytes_exact_boundary() {
        let data = vec![0u8; 100];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        let slice = TensorSlice { offset: 0, length: 100, quant: QuantScheme::F32 };
        let bytes = view.subtensor_bytes(&slice).unwrap();
        assert_eq!(bytes.len(), 100);
    }

    #[test]
    fn layer_view_subtensor_bytes_zero_length() {
        let data = vec![0u8; 10];
        let view = LayerView::from_owned(0, data, dummy_subtensors());
        let slice = TensorSlice { offset: 5, length: 0, quant: QuantScheme::F32 };
        let bytes = view.subtensor_bytes(&slice).unwrap();
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn layer_view_clone_shares_arc() {
        let data = vec![1u8; 16];
        let view1 = LayerView::from_owned(0, data, dummy_subtensors());
        let view2 = view1.clone();
        // Both views point to the same underlying data
        assert_eq!(view1.as_bytes().as_ptr(), view2.as_bytes().as_ptr());
    }

    #[test]
    fn prefetch_handle_state_transitions() {
        let mut handle = PrefetchHandle::new(5, PrefetchPriority::High);
        assert!(!handle.is_complete());
        assert!(!handle.is_cancelled());
        assert_eq!(handle.layer, 5);

        handle.mark_complete();
        assert!(handle.is_complete());
        assert!(!handle.is_cancelled());

        handle.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn cache_stats_hit_rate() {
        // 0 total → 0.0
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        // All hits → 1.0
        let stats = CacheStats { hits: 100, misses: 0, ..Default::default() };
        assert_eq!(stats.hit_rate(), 1.0);

        // 50/50 → 0.5
        let stats = CacheStats { hits: 50, misses: 50, ..Default::default() };
        assert!((stats.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cache_stats_utilization() {
        // Zero capacity → 0.0
        let stats = CacheStats { capacity_bytes: 0, bytes_cached: 100, ..Default::default() };
        assert_eq!(stats.utilization(), 0.0);

        // Half full
        let stats = CacheStats { capacity_bytes: 200, bytes_cached: 100, ..Default::default() };
        assert!((stats.utilization() - 0.5).abs() < 1e-10);
    }
}
