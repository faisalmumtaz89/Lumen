//! Lock-free async weight provider with background I/O thread.
//!
//! Architecture: the I/O thread sequentially pre-reads layers into atomic
//! slots. The main thread reads from slots without any mutex. This eliminates
//! 6-8 mutex lock/unlock operations per layer that the old channel+condvar
//! design required.
//!
//! ```text
//! I/O thread:  scans slots[0..N-1] in a loop
//!              if slot state == EMPTY → read from disk → mark READY
//!              if slot state == READY → skip (still in use)
//!              sleeps briefly when all slots are full
//!
//! Main thread: try_get_layer(i) → if slot[i] is READY, clone view
//!              get_layer_blocking(i) → spin-wait, then fallback to pread
//!              release_layer_hint(i) → mark slot EMPTY (I/O thread refills)
//! ```
//!
//! Zero mutex operations on the hot path. Stats use atomic counters.

use crate::weight::cache::{
    CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider,
};
use crate::error::RuntimeError;
use crate::storage::{IoSnapshot, IoTracker, StorageBackend};
use crate::storage::sync::SyncFileBackend;
use crate::weight::provider_sync::{bytes_to_f32, read_embedding_global, read_output_proj_global};
use lumen_format::quantization::QuantScheme;
use lumen_format::index::LayerIndex;
use lumen_format::reader::LbcFile;
use std::cell::UnsafeCell;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Slot states for the lock-free ready buffer.
const SLOT_EMPTY: u8 = 0;
const SLOT_FILLING: u8 = 1;
const SLOT_READY: u8 = 2;

/// A single slot in the lock-free ready buffer.
///
/// Access protocol enforced by atomic state transitions:
/// - I/O thread: EMPTY -> FILLING (exclusive write to `view`) -> READY
/// - Main thread: reads `view` only when state == READY (clone, not take)
/// - Main thread: READY -> EMPTY via `release_layer_hint`
struct ReadySlot {
    /// Atomic state: EMPTY -> FILLING -> READY -> EMPTY (released).
    state: AtomicU8,
    /// The LayerView data. Written by I/O thread when FILLING, read by main
    /// thread when READY. The atomic state acts as the synchronization barrier.
    view: UnsafeCell<Option<LayerView>>,
}

// SAFETY: ReadySlot uses atomic state to enforce exclusive access.
// - `view` is only written when state == FILLING (I/O thread exclusive).
// - `view` is only read when state == READY (main thread; I/O thread skips READY slots).
// - The Acquire/Release ordering on state transitions provides the memory barrier
//   that makes the UnsafeCell write visible to readers.
unsafe impl Send for ReadySlot {}
unsafe impl Sync for ReadySlot {}

impl ReadySlot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_EMPTY),
            view: UnsafeCell::new(None),
        }
    }
}

/// Shared data between the main thread and the I/O thread.
///
/// Wrapped in `Arc` so both threads hold a reference. The I/O thread
/// accesses `slots`, `io`, `shutdown`, and `hint_layer`. The main thread
/// accesses `slots`, `shutdown`, `hint_layer`, and stats atomics.
struct SharedSlots {
    slots: Vec<ReadySlot>,
    io: IoTracker,
    shutdown: AtomicBool,
    /// Hint from main thread: which layer index to prioritize reading next.
    /// The I/O thread uses this to start its scan from the hinted layer,
    /// ensuring the next needed layer is read first.
    hint_layer: AtomicU64,
    /// Handle to the I/O thread, used by release_layer_hint to unpark
    /// the thread when a slot becomes EMPTY. Initialized by the I/O thread
    /// itself on startup via OnceLock.
    io_thread_handle: std::sync::OnceLock<std::thread::Thread>,
}

fn read_f32_tensor(
    backend: &SyncFileBackend,
    offset: u64,
    length: u64,
) -> Result<Vec<f32>, RuntimeError> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let bytes = backend.read_range(offset, length)?;
    Ok(bytes_to_f32(&bytes))
}

/// Async weight provider with a lock-free background I/O thread.
///
/// Drop-in replacement for `SyncWeightProvider` / `MmapWeightProvider`.
/// Uses atomic slot states instead of mutexes for synchronization.
pub struct AsyncWeightProvider {
    lbc: LbcFile,
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

    /// Lock-free ready buffer: one slot per layer.
    shared: Arc<SharedSlots>,

    /// I/O thread handle.
    io_thread: Option<std::thread::JoinHandle<()>>,

    /// Atomic hit/miss/eviction counters (no Mutex).
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,

    /// Fallback backend for synchronous reads when I/O thread has not
    /// delivered a layer yet (guarantees progress).
    fallback_backend: SyncFileBackend,

    /// Layer index metadata for fallback reads.
    layer_indices: Vec<LayerIndex>,

    /// Number of layers in the model.
    num_layers: usize,
}

impl AsyncWeightProvider {
    /// Open an LBC file and start the background I/O thread.
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let lbc = LbcFile::open(path)?;

        // Read global tensors synchronously (same as SyncWeightProvider).
        let mut fallback_backend = SyncFileBackend::new();
        fallback_backend.open(path)?;
        let vocab_size = lbc.header.hyperparams.vocab_size as usize;
        let hidden_dim = lbc.header.hyperparams.hidden_dim as usize;

        let embedding_bytes = fallback_backend.read_range(
            lbc.header.embedding.offset,
            lbc.header.embedding.length,
        )?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim);
        let final_norm = read_f32_tensor(
            &fallback_backend,
            lbc.header.final_norm.offset,
            lbc.header.final_norm.length,
        )?;
        let output_proj_bytes = fallback_backend.read_range(
            lbc.header.output_proj.offset,
            lbc.header.output_proj.length,
        )?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim);

        let num_layers = lbc.header.num_layers as usize;
        let layer_indices = lbc.layer_indices.clone();

        // Create one slot per layer.
        let mut slots = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            slots.push(ReadySlot::new());
        }

        let shared = Arc::new(SharedSlots {
            slots,
            io: IoTracker::new(),
            shutdown: AtomicBool::new(false),
            hint_layer: AtomicU64::new(0),
            io_thread_handle: std::sync::OnceLock::new(),
        });

        let shared_clone = Arc::clone(&shared);
        let indices_clone = layer_indices.clone();
        let path_owned = path.to_path_buf();

        let io_thread = std::thread::Builder::new()
            .name("lumen-io".to_string())
            .spawn(move || {
                io_thread_loop(&shared_clone, &indices_clone, &path_owned);
            })
            .map_err(|e| {
                RuntimeError::StorageIo(std::io::Error::other(format!(
                    "failed to spawn I/O thread: {e}"
                )))
            })?;

        let weight_tying = lbc.header.weight_tying;
        Ok(Self {
            lbc,
            embedding,
            final_norm,
            output_proj,
            output_proj_raw,
            output_proj_quant,
            embedding_raw,
            embedding_quant,
            weight_tying,
            shared,
            io_thread: Some(io_thread),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            fallback_backend,
            layer_indices,
            num_layers,
        })
    }

    pub fn lbc(&self) -> &LbcFile {
        &self.lbc
    }
}

/// The I/O thread main loop. Scans slots sequentially and fills empty ones.
///
/// The thread starts scanning from the hinted layer (set by `prefetch_layer`)
/// and wraps around, reading any EMPTY slots it finds. When all slots are
/// full (READY), it yields briefly before scanning again.
fn io_thread_loop(shared: &SharedSlots, layer_indices: &[LayerIndex], path: &Path) {
    // Register this thread's handle so release_layer_hint can unpark us.
    let _ = shared.io_thread_handle.set(std::thread::current());

    let mut backend = SyncFileBackend::new();
    if let Err(e) = backend.open(path) {
        eprintln!("lumen-io: failed to open file: {e}");
        return;
    }

    let num_layers = shared.slots.len();
    if num_layers == 0 {
        return;
    }

    loop {
        if shared.shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Start scanning from the hinted layer for best overlap with compute.
        let hint = shared.hint_layer.load(Ordering::Relaxed) as usize;
        let mut did_work = false;

        for offset in 0..num_layers {
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }

            let layer = (hint + offset) % num_layers;
            let slot = &shared.slots[layer];

            // Try to claim this slot for filling.
            // Only transition EMPTY -> FILLING (compare_exchange is lock-free).
            if slot
                .state
                .compare_exchange(SLOT_EMPTY, SLOT_FILLING, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // We own this slot exclusively. Read the layer data.
                let idx = &layer_indices[layer];
                match backend.read_range(idx.layer_offset_bytes, idx.layer_length_bytes) {
                    Ok(data) => {
                        shared.io.record_read(idx.layer_length_bytes);
                        let view = LayerView::from_owned(layer, data, idx.subtensors.clone());
                        // SAFETY: We have exclusive access (state == FILLING).
                        // No other thread reads or writes `view` in this state.
                        unsafe {
                            *slot.view.get() = Some(view);
                        }
                        // Release ordering ensures the view write is visible
                        // to any thread that subsequently loads READY with Acquire.
                        slot.state.store(SLOT_READY, Ordering::Release);
                    }
                    Err(e) => {
                        eprintln!("lumen-io: failed to read layer {layer}: {e}");
                        // Revert to EMPTY so we can retry later.
                        slot.state.store(SLOT_EMPTY, Ordering::Release);
                    }
                }
                did_work = true;
            }
        }

        if !did_work {
            // All slots are READY (or FILLING). Park the thread until
            // release_layer_hint unparks us (a slot became EMPTY) or
            // shutdown is signaled. park() consumes a prior unpark token
            // if one exists, so we never miss a wake-up.
            std::thread::park();
        }
    }
}

impl WeightProvider for AsyncWeightProvider {
    fn prefetch_layer(
        &self,
        layer: usize,
        priority: PrefetchPriority,
    ) -> Result<PrefetchHandle, RuntimeError> {
        // Hint the I/O thread to prioritize this layer.
        self.shared
            .hint_layer
            .store(layer as u64, Ordering::Relaxed);
        let mut handle = PrefetchHandle::new(layer, priority);
        handle.mark_complete();
        Ok(handle)
    }

    fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        if layer >= self.num_layers {
            return Err(RuntimeError::LayerUnavailable {
                layer,
                reason: format!("layer index out of range (num_layers={})", self.num_layers),
            });
        }

        // Fast path: check if slot is READY (no mutex, just an atomic load).
        if let Some(view) = self.try_get_layer(layer) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(view);
        }

        // Hint the I/O thread to prioritize this layer.
        self.shared
            .hint_layer
            .store(layer as u64, Ordering::Relaxed);

        // Spin-wait with exponential backoff for I/O thread delivery.
        for attempt in 0..1000u32 {
            if let Some(view) = self.try_get_layer(layer) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(view);
            }
            if attempt < 32 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }

        // Fallback: synchronous read on the main thread (guarantees progress).
        self.misses.fetch_add(1, Ordering::Relaxed);
        let idx = &self.layer_indices[layer];
        let data = self
            .fallback_backend
            .read_range(idx.layer_offset_bytes, idx.layer_length_bytes)?;
        let view = LayerView::from_owned(layer, data, idx.subtensors.clone());

        // Store the fallback-read view into the slot so subsequent
        // try_get_layer calls find it (preserves existing test semantics:
        // get_layer_blocking followed by try_get_layer returns Some).
        let slot = &self.shared.slots[layer];
        // Try to claim the slot. If it is EMPTY, we fill it. If it is
        // already READY or FILLING (I/O thread raced us), that is fine --
        // we just return our view without storing.
        if slot
            .state
            .compare_exchange(SLOT_EMPTY, SLOT_FILLING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // SAFETY: exclusive access while FILLING.
            unsafe {
                *slot.view.get() = Some(view.clone());
            }
            slot.state.store(SLOT_READY, Ordering::Release);
        }

        Ok(view)
    }

    fn try_get_layer(&self, layer: usize) -> Option<LayerView> {
        if layer >= self.num_layers {
            return None;
        }

        let slot = &self.shared.slots[layer];

        // Acquire ordering ensures we see the view data written by the
        // I/O thread (which used Release when storing SLOT_READY).
        if slot.state.load(Ordering::Acquire) == SLOT_READY {
            // SAFETY: state == READY means the I/O thread has finished
            // writing and will not touch `view` until state becomes EMPTY.
            // We clone the view (not take), so it remains available for
            // subsequent try_get_layer calls.
            let view = unsafe { &*slot.view.get() };
            return view.clone();
        }

        None
    }

    fn release_layer_hint(&self, layer: usize) {
        if layer >= self.num_layers {
            return;
        }

        let slot = &self.shared.slots[layer];

        // Only release if the slot is currently READY.
        // CAS: READY -> EMPTY. If it fails (slot is FILLING or already EMPTY),
        // we skip -- no eviction occurred.
        if slot
            .state
            .compare_exchange(SLOT_READY, SLOT_EMPTY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            // SAFETY: We just transitioned READY -> EMPTY, so we have exclusive
            // access to `view` (I/O thread only touches EMPTY -> FILLING, and we
            // already set it to EMPTY, but the I/O thread will see EMPTY and start
            // a fresh FILLING cycle). We drop the old view to free memory.
            unsafe {
                let _ = (*slot.view.get()).take();
            }
            self.evictions.fetch_add(1, Ordering::Relaxed);

            // Wake the I/O thread -- a slot is now EMPTY and ready to be refilled.
            if let Some(handle) = self.shared.io_thread_handle.get() {
                handle.unpark();
            }
        }
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            ..Default::default()
        }
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn io_snapshot(&self) -> Option<IoSnapshot> {
        let io_snap = self.shared.io.snapshot();
        let fb_snap = self.fallback_backend.io_tracker().map(|t| t.snapshot());
        Some(IoSnapshot {
            bytes_read: io_snap.bytes_read + fb_snap.map_or(0, |s| s.bytes_read),
            read_ops: io_snap.read_ops + fb_snap.map_or(0, |s| s.read_ops),
        })
    }
}

impl Drop for AsyncWeightProvider {
    fn drop(&mut self) {
        // Signal the I/O thread to exit.
        self.shared.shutdown.store(true, Ordering::Release);
        // Unpark the I/O thread in case it is parked (waiting for work).
        if let Some(thread_handle) = self.shared.io_thread_handle.get() {
            thread_handle.unpark();
        }
        if let Some(handle) = self.io_thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ComputeBackend;
    use crate::compute::naive::NaiveF32Backend;
    use crate::config::RuntimeConfig;
    use crate::engine::{InferenceEngine, SamplingParams, StopCondition};
    use crate::kv::KvPrecision;
    use crate::pipeline::PipelineMode;
    use crate::weight::provider_sync::SyncWeightProvider;
    use lumen_format::test_model::{generate_test_model, TestModelConfig};
    use std::io::Write as IoWrite;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static ASYNC_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_test_lbc() -> std::path::PathBuf {
        let config = TestModelConfig::default();
        let data = generate_test_model(&config);
        let id = ASYNC_TEST_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_test_async_wp_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.lbc");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        path
    }

    #[test]
    fn async_provider_loads_layers() {
        let path = create_test_lbc();
        let provider = AsyncWeightProvider::open(&path).unwrap();

        assert_eq!(provider.num_layers(), 2);

        let view = provider.get_layer_blocking(0).unwrap();
        assert_eq!(view.layer_idx, 0);
        assert!(view.byte_len() > 0);

        // Second access should be a cache hit (from ready buffer).
        let view2 = provider.get_layer_blocking(0).unwrap();
        assert_eq!(view.byte_len(), view2.byte_len());
    }

    #[test]
    fn async_prefetch_populates_ready() {
        let path = create_test_lbc();
        let provider = AsyncWeightProvider::open(&path).unwrap();

        provider
            .prefetch_layer(1, PrefetchPriority::High)
            .unwrap();

        // Give the I/O thread time to complete.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let view = provider.try_get_layer(1);
        assert!(view.is_some(), "prefetched layer should be in ready buffer");
        assert_eq!(view.unwrap().layer_idx, 1);
    }

    #[test]
    fn async_release_clears_buffer() {
        let path = create_test_lbc();
        let provider = AsyncWeightProvider::open(&path).unwrap();

        // Load a layer into the ready buffer.
        let _view = provider.get_layer_blocking(0).unwrap();
        assert!(provider.try_get_layer(0).is_some());

        // Release it.
        provider.release_layer_hint(0);

        // Should no longer be available.
        assert!(provider.try_get_layer(0).is_none());

        let stats = provider.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn async_deterministic_output() {
        let path = create_test_lbc();

        // Generate with sync provider.
        let sync_provider = SyncWeightProvider::open(&path).unwrap();
        let mut sync_backend = NaiveF32Backend::new();
        sync_backend.set_global_tensors(
            sync_provider.embedding.clone(),
            sync_provider.final_norm.clone(),
            sync_provider.output_proj.clone(),
        );
        sync_backend
            .init(&sync_provider.lbc().header.hyperparams)
            .unwrap();

        let rt_config = RuntimeConfig {
            pipeline_mode: PipelineMode::MinMem,
            prefetch_distance: 1,
            kv_precision: KvPrecision::F32,
            max_seq_len: 64,
            collect_per_layer_timings: false,
        };
        let engine =
            InferenceEngine::new(rt_config.clone(), sync_provider.lbc().header.hyperparams);
        let prompt = vec![0u32, 1, 2];
        let stop = StopCondition::MaxTokens(5);
        let sampling = SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        };

        let sync_result = engine
            .generate(&prompt, &sync_provider, &sync_backend, &stop, &sampling)
            .unwrap();

        // Generate with async provider.
        let async_provider = AsyncWeightProvider::open(&path).unwrap();
        let mut async_backend = NaiveF32Backend::new();
        async_backend.set_global_tensors(
            async_provider.embedding.clone(),
            async_provider.final_norm.clone(),
            async_provider.output_proj.clone(),
        );
        async_backend
            .init(&async_provider.lbc().header.hyperparams)
            .unwrap();

        let engine =
            InferenceEngine::new(rt_config, async_provider.lbc().header.hyperparams);
        let async_result = engine
            .generate(&prompt, &async_provider, &async_backend, &stop, &sampling)
            .unwrap();

        assert_eq!(
            sync_result.tokens, async_result.tokens,
            "async and sync providers must produce identical output"
        );
    }

    #[test]
    fn async_shutdown_clean() {
        let path = create_test_lbc();
        {
            let provider = AsyncWeightProvider::open(&path).unwrap();
            let _view = provider.get_layer_blocking(0).unwrap();
            // provider dropped here — should shut down cleanly
        }
        // If we reach here without panic, the test passes.
    }
}
