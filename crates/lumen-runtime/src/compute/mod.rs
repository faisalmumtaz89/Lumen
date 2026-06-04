//! Compute backend trait.
//!
//! The compute backend is the pluggable kernel execution layer. The runtime
//! owns memory and scheduling; the backend provides matrix operations
//! (matmul, attention, MLP, normalization, sampling).

pub mod cpu_naive;
pub mod cpu_simd;
pub mod simd_kernels;

use crate::weight::cache::{LayerView, WeightProvider};
use crate::error::RuntimeError;
use crate::kv::disk::RecurrentState;
use crate::kv::{KvCache, KvCacheView, KvPrecision};
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;

/// What a backend can do. Queried once at init, cached by the engine.
#[derive(Debug, Clone)]
pub struct BackendCaps {
    pub batched_prefill: bool,
    pub gpu_resident: bool,
    pub gdn: bool,
    pub moe: bool,
    pub gpu_argmax: bool,
}

/// A contiguous buffer of activations flowing through the model.
///
/// Shape: `[batch_size, seq_len, hidden_dim]` flattened to bytes.
#[derive(Debug)]
pub struct ActivationBuffer {
    pub data: Vec<u8>,
    pub num_elements: usize,
    pub dtype: ComputeDtype,
}

/// Data types used in compute operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDtype {
    F32,
    F16,
    Bf16,
}

impl ComputeDtype {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

impl ActivationBuffer {
    pub fn zeros(num_elements: usize, dtype: ComputeDtype) -> Self {
        let byte_len = num_elements * dtype.byte_size();
        Self {
            data: vec![0u8; byte_len],
            num_elements,
            dtype,
        }
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Zero-copy f32 view of the byte data (LE platforms only).
    ///
    /// Returns a `&[f32]` slice backed by the same memory as `self.data`,
    /// eliminating the memcpy that `read_f32_into` performs.
    ///
    /// # Safety contract
    ///
    /// - Only valid when `dtype == F32`.
    /// - Requires the `data` buffer to be 4-byte aligned (guaranteed by global
    ///   allocators on all supported platforms: system/jemalloc align to >= 8).
    /// - The returned slice borrows `self`; caller must not mutate `self.data`
    ///   while the slice is live.
    ///
    /// # Panics
    ///
    /// Panics on big-endian platforms (LE byte order != native f32 repr).
    /// Debug-asserts dtype and alignment.
    #[inline(always)]
    pub fn as_f32_slice(&self) -> &[f32] {
        debug_assert_eq!(self.dtype, ComputeDtype::F32);
        #[cfg(target_endian = "little")]
        {
            let ptr = self.data.as_ptr();
            // Global allocators align to >= max_align_t (8 or 16 bytes).
            // f32 needs 4-byte alignment. Debug-assert to catch exotic allocators.
            debug_assert_eq!(
                ptr.align_offset(std::mem::align_of::<f32>()),
                0,
                "ActivationBuffer::as_f32_slice: data not 4-byte aligned"
            );
            // SAFETY: data is Vec<u8> containing LE f32 bytes. On LE platform,
            // byte repr = f32 repr. Alignment verified above. Length is
            // num_elements * 4 bytes = num_elements f32s.
            unsafe { std::slice::from_raw_parts(ptr as *const f32, self.num_elements) }
        }
        #[cfg(target_endian = "big")]
        {
            panic!("as_f32_slice requires little-endian platform");
        }
    }

    /// Read f32 data from this buffer into the provided slice (no allocation).
    /// Uses unsafe memcpy on little-endian platforms to skip per-element conversion.
    #[inline]
    pub fn read_f32_into(&self, out: &mut [f32]) {
        assert_eq!(self.dtype, ComputeDtype::F32);
        assert_eq!(out.len(), self.num_elements);
        #[cfg(target_endian = "little")]
        {
            // SAFETY: F32 data stored as LE bytes; on LE platform byte repr = f32 repr.
            // Asserts above guarantee out.len() == num_elements and data.len() >= num_elements * 4.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    self.num_elements * 4,
                );
            }
        }
        #[cfg(target_endian = "big")]
        {
            for (i, chunk) in self.data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
    }

    /// Write f32 data into this buffer, reusing existing allocation when possible.
    /// Uses unsafe memcpy on little-endian platforms to skip per-element conversion.
    ///
    /// Hot-path optimized: when the buffer already has sufficient capacity (which
    /// it does on every call after the first in compute_layer's layer loop), this
    /// skips the clear+reserve overhead and directly overwrites the bytes.
    #[inline]
    pub fn write_f32_from(&mut self, values: &[f32]) {
        debug_assert_eq!(self.dtype, ComputeDtype::F32);
        self.num_elements = values.len();
        let needed = values.len() * 4;
        #[cfg(target_endian = "little")]
        {
            // Fast path: if capacity is already sufficient, skip clear+reserve.
            // This is the common case in the layer loop (buffer size is constant).
            if self.data.capacity() < needed {
                self.data.reserve(needed - self.data.len());
            }
            // SAFETY: capacity >= needed (either pre-existing or just reserved).
            // f32 values are contiguous and LE byte repr = f32 repr on this platform.
            // We overwrite all `needed` bytes, so prior contents are irrelevant.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr() as *const u8,
                    self.data.as_mut_ptr(),
                    needed,
                );
                self.data.set_len(needed);
            }
        }
        #[cfg(target_endian = "big")]
        {
            self.data.clear();
            self.data.reserve(needed);
            for &v in values {
                self.data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// Logits output from the final layer, ready for sampling.
#[derive(Debug)]
pub struct Logits {
    /// Shape: `[vocab_size]` for single-token decode.
    pub data: Vec<f32>,
}

impl Logits {
    pub fn vocab_size(&self) -> usize {
        self.data.len()
    }

    /// Returns the index of the maximum logit (greedy sampling).
    /// Uses `f32::total_cmp` for deterministic NaN handling.
    /// Returns 0 if logits are empty.
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// The pluggable compute backend interface.
///
/// # Contract
///
/// - `compute_layer` performs: attention norm -> attention -> residual ->
///   ffn norm -> MLP -> residual.
/// - `compute_final` projects the last hidden state to logits.
/// - The backend must NOT hold references to `LayerView` data after
///   `compute_layer` returns.
pub trait ComputeBackend: Send + Sync {
    /// Initialize with model hyperparameters. Called once before inference.
    fn init(&mut self, hyperparams: &ModelHyperparams) -> Result<(), RuntimeError>;

    /// Execute a single transformer layer.
    ///
    /// # KV seq_len contract
    ///
    /// This method does NOT advance `kv.seq_len()`. The caller (typically
    /// `forward_pass()` in the engine) MUST call `kv.advance_seq_len()`
    /// after each token's full forward pass completes.
    fn compute_layer(
        &self,
        layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError>;

    /// Project the final hidden state to vocabulary logits.
    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError>;

    /// Embed a token id into the initial hidden state.
    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError>;

    /// Print a per-operation profiling summary (if profiling was enabled).
    /// Default implementation is a no-op. Backends that support profiling
    /// override this to print accumulated timing data.
    fn print_profile(&self) {}

    /// Enable or disable per-operation profiling.
    /// Default implementation is a no-op. Only backends with profiling support
    /// need to override this.
    fn set_profile(&mut self, _enabled: bool) {}

    /// Validate that this backend supports the requested KV cache precision.
    ///
    /// Each backend has a single hardcoded storage layout for the GPU- or
    /// CPU-resident KV cache; if the runtime config asks for a different
    /// precision the backend silently ignored it before, producing memory-
    /// layout mismatches that would later corrupt KV writes (Metal: F16-only
    /// `gpu_k_cache`/`gpu_v_cache`; CUDA: F32-only `KvCacheGpu`). Reject the
    /// mismatch up front so the caller sees an explicit error instead of
    /// downstream silent data corruption.
    ///
    /// Default impl (CPU naive / SIMD) accepts both `F32` and `F16` because
    /// the CPU `KvCache` byte buffers are sized by `config.precision` and the
    /// `KvCacheView` append/read helpers dispatch on precision at runtime.
    ///
    /// Backends with a hardcoded precision MUST override this and return
    /// `RuntimeError::Unsupported` for mismatched configs.
    fn validate_kv_precision(&self, precision: KvPrecision) -> Result<(), RuntimeError> {
        // CPU backends store KV in `Vec<u8>` sized by precision and dispatch
        // append/read at runtime, so any implemented precision works.
        if !precision.is_implemented() {
            return Err(RuntimeError::Unsupported(format!(
                "KV cache precision {precision:?} is not yet implemented"
            )));
        }
        Ok(())
    }

    // ====================================================================
    // Phase 3: enriched trait methods for GPU fast paths.
    // All have default implementations so CPU backends compile unchanged.
    // ====================================================================

    /// Query backend capabilities.
    fn caps(&self) -> BackendCaps {
        BackendCaps {
            batched_prefill: false,
            gpu_resident: false,
            gdn: false,
            moe: false,
            gpu_argmax: false,
        }
    }

    /// Set global tensors (embedding, final_norm, output_proj).
    fn set_global_tensors(
        &mut self,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        output_proj: Vec<f32>,
    );

    /// Set raw quantized output projection bytes.
    fn set_output_proj_raw(&mut self, _raw: Vec<u8>, _quant: QuantScheme) {}

    /// Set raw quantized embedding bytes.
    fn set_embedding_raw(&mut self, _raw: Vec<u8>, _quant: QuantScheme) {}

    /// Enable weight tying (output_proj shares embedding storage).
    fn set_weight_tying(&mut self, _enabled: bool) {}

    /// Batched GPU prefill. Returns the final hidden state of the last token.
    ///
    /// # KV seq_len contract
    ///
    /// This method advances `kv.seq_len()` internally (once per prompt token).
    /// The caller must NOT call `kv.advance_seq_len()` after prefill returns.
    fn prefill(
        &self,
        _tokens: &[u32],
        _weights: &dyn WeightProvider,
        _kv: &mut KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        Err(RuntimeError::Compute("batched prefill not supported".into()))
    }

    /// Batched prefill resuming from `start_pos`, used by suffix prefill /
    /// prompt-cache paths.
    ///
    /// # Contract
    ///
    /// - `start_pos` MUST equal the current `kv.seq_len()`. The default
    ///   implementation enforces this and then delegates to [`Self::prefill`],
    ///   which already resumes from `kv.seq_len()` on every supporting backend
    ///   (CUDA reads `pos_start = kv.seq_len()` at the top of `prefill`; Metal
    ///   reads `seq_pos_start = kv.seq_len` in its batched layer encoder).
    /// - `suffix_tokens` is just the new tail that extends the prior KV;
    ///   callers pass the slice _after_ truncating `kv` to the common prefix.
    /// - `kv.seq_len()` is advanced internally exactly like [`Self::prefill`].
    ///
    /// Backends that need to deviate from the "delegate to `prefill`" contract
    /// (e.g. to recompute a partial RoPE table from `start_pos` lazily) can
    /// override this directly.
    fn prefill_from(
        &self,
        start_pos: usize,
        suffix_tokens: &[u32],
        weights: &dyn WeightProvider,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        if start_pos != kv.seq_len() {
            return Err(RuntimeError::Compute(format!(
                "prefill_from: start_pos {start_pos} != kv.seq_len() {} \
                 (callers must truncate KV to the resume point first)",
                kv.seq_len()
            )));
        }
        self.prefill(suffix_tokens, weights, kv)
    }

    /// Preload weights to GPU memory.
    fn preload_weights(
        &mut self,
        _weights: &dyn WeightProvider,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// GPU-resident decode returning logits for a single token.
    ///
    /// # KV seq_len contract
    ///
    /// This method advances `kv.seq_len()` internally (once per call).
    /// The caller must NOT call `kv.advance_seq_len()` after decode_token returns.
    fn decode_token(
        &self,
        _token_id: u32,
        _weights: &dyn WeightProvider,
        _kv: &mut KvCache,
    ) -> Result<Logits, RuntimeError> {
        Err(RuntimeError::Compute("GPU-resident decode not supported".into()))
    }

    /// GPU-side greedy decode returning token ID directly.
    ///
    /// # KV seq_len contract
    ///
    /// This method advances `kv.seq_len()` internally (once per call).
    /// The caller must NOT call `kv.advance_seq_len()` after decode_token_greedy returns.
    fn decode_token_greedy(
        &self,
        _token_id: u32,
        _weights: &dyn WeightProvider,
        _kv: &mut KvCache,
    ) -> Result<u32, RuntimeError> {
        Err(RuntimeError::Compute("GPU-side argmax not supported".into()))
    }

    /// Reset recurrent state (GDN h_state, conv_state).
    fn reset_recurrent_state(&self) {}

    // ====================================================================
    // disk-KV save/load sync hooks.
    // ====================================================================

    /// Mirror the live GPU KV state (and optional GDN recurrent state) back
    /// into CPU-visible buffers, blocking until the GPU work has settled.
    ///
    /// The disk-save path (`kv::disk::save_atomic`) writes byte-for-byte
    /// from `KvCache.keys[layer]` / `KvCache.values[layer]` and from the
    /// caller-supplied [`RecurrentState`]; on GPU backends those CPU
    /// mirrors only contain useful data after this method runs. The default
    /// implementation is a no-op so the CPU backends (which already own the
    /// authoritative buffers in `KvCache`) compile unchanged.
    ///
    /// Backends overriding this MUST:
    /// - synchronously wait for any in-flight command buffers (Metal:
    ///   submit an empty CB and `commit_and_wait`; CUDA: `device.synchronize`);
    /// - populate `KvCache.set_layer_raw_bytes` for each layer in the
    ///   storage layout declared by `validate_kv_precision`;
    /// - populate the supplied `RecurrentState` when `Some` and the backend
    ///   has GDN state (Metal Qwen3.5-9B); on backends without GDN, leave
    ///   the recurrent argument untouched and return `Ok(())`.
    ///
    /// Errors are surfaced verbatim — a partial sync MUST leave the CPU
    /// buffers in a known state (either fully synced or unmodified).
    fn sync_kv_to_cpu(
        &self,
        _kv: &mut KvCache,
        _recurrent: Option<&mut RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Inverse of [`Self::sync_kv_to_cpu`]: copy CPU KV bytes (and optional
    /// recurrent state) into the backend's GPU-resident buffers so the
    /// next forward pass uses the restored state.
    ///
    /// Same contract as [`Self::sync_kv_to_cpu`] in reverse: backends MUST
    /// block until the upload completes, MUST validate that the supplied
    /// `RecurrentState` layout matches what the backend has allocated, and
    /// MUST leave their buffers in a known state on error.
    ///
    /// The default impl is a no-op so the CPU backends compile unchanged.
    fn sync_kv_from_cpu(
        &self,
        _kv: &KvCache,
        _recurrent: Option<&RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Describe the GDN layout this backend uses, if any. Returns `None`
    /// when the backend either has no GDN state or has not yet allocated
    /// it (e.g. a non-GDN model on the Metal backend). The disk-save path
    /// uses this to size and tag the optional recurrent section.
    fn gdn_layout(&self) -> Option<crate::kv::disk::GdnLayout> {
        None
    }

    /// report this backend's current GPU residency in bytes, if
    /// the backend can introspect its driver.
    ///
    /// On Metal this returns `MTLDevice.currentAllocatedSize` (the byte
    /// count for all MTLBuffer / MTLTexture / MTLHeap objects the process
    /// holds against the backend's device).  CUDA can return
    /// `cuMemGetInfo`'s allocated portion when implemented.  CPU backends
    /// return 0 by default (no separate device memory residency to count).
    ///
    /// Used by the `/debug/memory_breakdown` HTTP endpoint
    /// (env-gated, default OFF) to attribute long-session RSS growth
    /// between "Rust heap state" and "Metal/CUDA driver state".
    ///
    /// Cost: at most one syscall / driver query.  Cheap enough for the
    /// 30-second sampling cadence used in the soak harness; must NOT be
    /// called inside the per-token decode loop.
    fn current_allocated_bytes(&self) -> u64 {
        0
    }

    /// Report peak memory residency observed over the
    /// lifetime of this backend instance, in bytes.
    ///
    /// Semantics by backend:
    /// - **CUDA**: `device.total_memory() - min(device.free_memory())` snapshot
    ///   at call site, i.e. the worst-case VRAM consumption observed at the
    ///   moment of the call.  The CUDA backend does NOT continuously track a
    ///   running min(free) across the whole generation; instead the engine
    ///   samples this at end-of-gen and the value is "VRAM in use right now".
    ///   This is a worst-case lower bound and is sufficient for the
    ///   `peak_vram_pct_of_device_limit` gate (≤ 90% of device limit).
    /// - **Metal**: `MTLDevice.currentAllocatedSize()`, i.e. the byte count
    ///   for all outstanding MTLBuffer / MTLTexture / MTLHeap objects.  On
    ///   Apple unified memory this is the authoritative GPU residency
    ///   measure.
    /// - **CPU naive / SIMD**: process RSS via `getrusage(RUSAGE_SELF)`.
    ///   On macOS, `ru_maxrss` is in **bytes**; on Linux, it is in
    ///   **kilobytes** (1024-byte units). Implementations multiply by 1024
    ///   on Linux.
    ///
    /// Default impl returns 0 for backwards compatibility with any future
    /// backend; callers MUST treat a return of 0 as "unknown / not reported"
    /// rather than "no memory used".
    ///
    /// Cost: at most one syscall / driver query.  Safe to call at
    /// end-of-generation in the engine; must NOT be called inside the
    /// per-token decode loop.
    fn peak_memory_bytes(&self) -> u64 {
        0
    }
}

/// Read process Resident Set Size (RSS) in bytes via
/// `getrusage(RUSAGE_SELF).ru_maxrss`.
///
/// Unit handling: macOS reports `ru_maxrss` in **bytes**, Linux reports it
/// in **kilobytes** (1024-byte units). Returns 0 on any syscall failure
/// so callers can use this in `InferenceMetrics::peak_memory_bytes`
/// without an extra Option wrap.
///
/// Used by CPU backends (naive + SIMD) to populate the
/// `ComputeBackend::peak_memory_bytes()` trait method.  Cheap (one
/// syscall) so safe to call at end-of-generation but NOT in the
/// per-token decode loop.
#[cfg(unix)]
pub fn process_rss_bytes() -> u64 {
    // SAFETY: getrusage with a stack-allocated `rusage` struct is a
    // straightforward libc call; we initialise to zeros so a partial
    // write from a faulty libc still leaves a well-defined value.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage as *mut libc::rusage) };
    if rc != 0 {
        return 0;
    }
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        // Darwin reports bytes directly.
        raw
    } else {
        // Linux reports kilobytes; convert to bytes.
        raw.saturating_mul(1024)
    }
}

/// Windows / non-unix fallback returns 0. This fallback does not
/// target Windows for prod-readiness.
#[cfg(not(unix))]
pub fn process_rss_bytes() -> u64 {
    0
}
