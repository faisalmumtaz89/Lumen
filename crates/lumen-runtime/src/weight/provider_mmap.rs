//! Mmap-based weight provider with zero-copy layer access and PRIMA.CPP-aware
//! windowed prefetch.
//!
//! Layer views are pre-built at init time and returned by cheap clone (pointer
//! copy). No syscalls, no atomic I/O tracking, and no heap allocation on the
//! hot path. The OS page cache is the only cache; `madvise(WILLNEED)` hints
//! drive prefetch and `madvise(DONTNEED)` hints release pages after use,
//! enabling windowed streaming for models larger than available RAM.

use crate::weight::cache::{
    CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider,
};
use crate::error::RuntimeError;
use crate::storage::{IoSnapshot, MmapConfig, MmapPageCacheBackend, StorageBackend};
use crate::storage::mmap::MmapStorageBackend;
use crate::weight::provider_sync::{bytes_to_f32, read_embedding_global, read_output_proj_global};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Atomic cache statistics -- lock-free telemetry for the mmap hot path.
struct AtomicCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl AtomicCacheStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            // Zero-copy: no heap cache, so these are always 0.
            layers_cached: 0,
            bytes_cached: 0,
            capacity_bytes: 0,
            prefetch_hits: 0,
            prefetch_misses: 0,
            inflight_prefetches: 0,
        }
    }
}

/// Mmap weight provider with zero-copy layer access and windowed prefetch.
///
/// Every `get_layer_blocking` / `try_get_layer` returns a `LayerView` backed
/// by a raw pointer into the mmap region -- zero allocation, zero memcpy.
/// The mmap region outlives all `LayerView`s because the engine borrows
/// `&dyn WeightProvider` for the duration of `generate()`.
///
/// # Hot-path optimizations
///
/// 1. **Pre-cached LayerViews**: All views are built once at `open()` time.
///    `get_layer_blocking` / `try_get_layer` return a clone (pointer copy,
///    no `slice_ref` call, no bounds check, no `IoTracker::record_read`).
///
/// 2. **Windowed prefetch**: `prefetch_layer` issues `madvise(WILLNEED)`
///    within a sliding window of the compute cursor. `release_layer_hint`
///    issues `madvise(DONTNEED)` to allow the OS to reclaim pages. This
///    enables streaming models larger than available RAM.
///
/// 3. **No IoTracker recording on hot path**: The `slice_ref` path through
///    the storage backend (which does 2 atomic fetch_add ops per call) is
///    bypassed entirely by using pre-cached views.
pub struct MmapWeightProvider {
    lbc: LbcFile,
    backend: MmapStorageBackend,
    /// Tracks the highest layer accessed by compute (get_layer_blocking).
    /// Prefetch is only allowed within prefetch_window of this cursor.
    compute_cursor: AtomicUsize,
    /// Configuration for prefetch window size.
    prefetch_window: usize,
    /// Global tensors.
    pub embedding: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub output_proj: Vec<f32>,
    /// Raw output_proj bytes (Q8_0 or F32). Used by Metal backend.
    pub output_proj_raw: Vec<u8>,
    /// Quantization scheme of the output_proj tensor.
    pub output_proj_quant: QuantScheme,
    /// Raw embedding bytes (Q8_0/Q4_0 or empty for F32). Used by Metal backend.
    pub embedding_raw: Vec<u8>,
    /// Quantization scheme of the embedding tensor.
    pub embedding_quant: QuantScheme,
    /// Whether output_proj shares embedding storage (weight tying).
    pub weight_tying: bool,
    stats: AtomicCacheStats,
    /// Pre-built LayerViews for every layer. Built once at init, returned
    /// by clone on every access. Clone of a ZeroCopy LayerView is just a
    /// pointer + length copy (no Arc, no atomic refcount).
    cached_views: Vec<LayerView>,
}

impl MmapWeightProvider {
    pub fn open(path: &Path, mmap_config: MmapConfig) -> Result<Self, RuntimeError> {
        let lbc = LbcFile::open(path)?;
        let mut backend = MmapStorageBackend::new();
        backend.configure(mmap_config.clone());
        backend.open(path)?;

        // Read global tensors from mmap (these are small and copied once at init)
        let vocab_size = lbc.header.hyperparams.vocab_size as usize;
        let hidden_dim = lbc.header.hyperparams.hidden_dim as usize;

        let embedding_bytes = backend.read_range(
            lbc.header.embedding.offset, lbc.header.embedding.length,
        )?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim);
        let final_norm = bytes_to_f32(
            &backend.read_range(lbc.header.final_norm.offset, lbc.header.final_norm.length)?,
        );
        let output_proj_bytes = backend.read_range(
            lbc.header.output_proj.offset, lbc.header.output_proj.length,
        )?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim);

        let prefetch_window = mmap_config.prefetch_window;

        // Pre-build all LayerViews at init time. This moves the cost of
        // slice_ref (bounds check + IoTracker::record_read) out of the hot
        // path entirely. Each view is ~40 bytes (ptr, len, subtensors).
        let num_layers = lbc.layer_indices.len();
        let mut cached_views = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let idx = &lbc.layer_indices[layer];
            let (ptr, len) = backend.slice_ref(idx.layer_offset_bytes, idx.layer_length_bytes)?;
            // SAFETY: The mmap region (backend) is owned by this struct and
            // outlives all LayerViews. The engine borrows &dyn WeightProvider
            // for the duration of generate().
            let view = unsafe { LayerView::from_mmap_ptr(layer, ptr, len, idx.subtensors.clone()) };
            cached_views.push(view);
        }

        let weight_tying = lbc.header.weight_tying;
        Ok(Self {
            lbc,
            backend,
            compute_cursor: AtomicUsize::new(0),
            prefetch_window,
            embedding,
            final_norm,
            output_proj,
            output_proj_raw,
            output_proj_quant,
            embedding_raw,
            embedding_quant,
            weight_tying,
            stats: AtomicCacheStats::new(),
            cached_views,
        })
    }

    pub fn lbc(&self) -> &LbcFile {
        &self.lbc
    }

}

impl WeightProvider for MmapWeightProvider {
    fn prefetch_layer(
        &self,
        layer: usize,
        priority: PrefetchPriority,
    ) -> Result<PrefetchHandle, RuntimeError> {
        let cursor = self.compute_cursor.load(Ordering::Relaxed);

        // Only prefetch within the window of the compute cursor (not the prefetch cursor)
        if layer > cursor + self.prefetch_window {
            let mut handle = PrefetchHandle::new(layer, priority);
            handle.mark_complete();
            return Ok(handle);
        }

        // Issue madvise(WILLNEED) for this layer's data
        if layer < self.lbc.layer_indices.len() {
            let idx = &self.lbc.layer_indices[layer];
            self.backend
                .advise_willneed(idx.layer_offset_bytes, idx.layer_length_bytes)?;
        }

        let mut handle = PrefetchHandle::new(layer, priority);
        handle.mark_complete();
        Ok(handle)
    }

    fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        // Advance compute cursor so prefetch window tracks actual compute position.
        // fetch_max is atomic and lock-free.
        self.compute_cursor.fetch_max(layer, Ordering::Relaxed);

        if layer >= self.cached_views.len() {
            return Err(RuntimeError::LayerUnavailable {
                layer,
                reason: format!(
                    "layer index out of range (num_layers={})",
                    self.cached_views.len()
                ),
            });
        }

        // Zero-copy: every access is instant, no cache miss concept.
        self.stats.hits.fetch_add(1, Ordering::Relaxed);

        // Return a clone of the pre-cached view. For ZeroCopy backing this
        // is a pointer + length copy -- no slice_ref, no bounds check, no
        // IoTracker::record_read (0 atomic ops, 0 syscalls).
        Ok(self.cached_views[layer].clone())
    }

    fn try_get_layer(&self, layer: usize) -> Option<LayerView> {
        // Advance compute cursor so prefetch window tracks correctly even
        // when the engine uses try_get_layer (which always succeeds for mmap).
        self.compute_cursor.fetch_max(layer, Ordering::Relaxed);
        // Return the pre-cached view directly. No slice_ref, no io tracking.
        self.cached_views.get(layer).cloned()
    }

    fn release_layer_hint(&self, layer: usize) {
        self.stats.evictions.fetch_add(1, Ordering::Relaxed);

        // Issue DONTNEED to release pages back to OS, allowing the kernel
        // to reclaim memory for models that exceed available RAM.
        if layer < self.lbc.layer_indices.len() {
            let idx = &self.lbc.layer_indices[layer];
            let _ = self.backend.advise_dontneed(idx.layer_offset_bytes, idx.layer_length_bytes);
        }
    }

    fn begin_pass(&self) {
        // Reset the compute cursor so the prefetch window tracks correctly
        // from layer 0 on each new token. Without this, fetch_max keeps
        // the cursor at num_layers-1 after the first token, making the
        // window check a permanent no-op.
        self.compute_cursor.store(0, Ordering::Relaxed);
    }

    fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    fn num_layers(&self) -> usize {
        self.lbc.header.num_layers as usize
    }

    fn io_snapshot(&self) -> Option<IoSnapshot> {
        self.backend.io_tracker().map(|t| t.snapshot())
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use lumen_format::test_model::{generate_test_model, TestModelConfig};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MMAP_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_test_lbc() -> std::path::PathBuf {
        let config = TestModelConfig::default();
        let data = generate_test_model(&config);
        let id = MMAP_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_test_mmap_wp_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.lbc");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        path
    }

    #[test]
    fn mmap_weight_provider_loads_layers() {
        let path = create_test_lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        assert_eq!(provider.num_layers(), 2);

        let view = provider.get_layer_blocking(0).unwrap();
        assert_eq!(view.layer_idx, 0);
        assert!(view.byte_len() > 0);

        // Second access is also a zero-copy hit (no cache miss concept)
        let view2 = provider.get_layer_blocking(0).unwrap();
        assert_eq!(view.byte_len(), view2.byte_len());

        let stats = provider.stats();
        assert_eq!(stats.hits, 2); // Both are instant zero-copy hits
    }

    #[test]
    fn mmap_release_layer() {
        let path = create_test_lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        let _view = provider.get_layer_blocking(0).unwrap();
        provider.release_layer_hint(0);

        // With zero-copy, try_get_layer always succeeds for valid layers
        // (the mmap is still mapped; release only issues DONTNEED advice).
        assert!(provider.try_get_layer(0).is_some());

        let stats = provider.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn mmap_zero_copy_data_matches_read_range() {
        let path = create_test_lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        // Get the layer data via zero-copy.
        let view = provider.get_layer_blocking(0).unwrap();
        let zero_copy_bytes = view.as_bytes().to_vec();

        // Get the same layer data via read_range (the old copying path).
        let idx = &provider.lbc().layer_indices[0];
        let mut backend = MmapStorageBackend::new();
        backend.open(&path).unwrap();
        let copied_bytes = backend.read_range(idx.layer_offset_bytes, idx.layer_length_bytes).unwrap();

        assert_eq!(zero_copy_bytes, copied_bytes, "zero-copy and copied data must be identical");
    }

    #[test]
    fn mmap_subtensor_bytes_via_zero_copy() {
        let path = create_test_lbc();
        let provider = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        let view = provider.get_layer_blocking(0).unwrap();
        let st = &view.subtensors;

        // All subtensor slices should be readable without error.
        let _ = view.subtensor_bytes(&st.wq).unwrap();
        let _ = view.subtensor_bytes(&st.wk).unwrap();
        let _ = view.subtensor_bytes(&st.wv).unwrap();
        let _ = view.subtensor_bytes(&st.wo).unwrap();
        let _ = view.subtensor_bytes(&st.w_gate).unwrap();
        let _ = view.subtensor_bytes(&st.w_up).unwrap();
        let _ = view.subtensor_bytes(&st.w_down).unwrap();
        let _ = view.subtensor_bytes(&st.attn_norm).unwrap();
        let _ = view.subtensor_bytes(&st.ffn_norm).unwrap();
    }
}
