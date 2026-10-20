//! Compute backend trait.
//!
//! The compute backend is the pluggable kernel execution layer. The runtime
//! owns memory and scheduling; the backend provides matrix operations
//! (matmul, attention, MLP, normalization, sampling).

pub mod cpu_naive;
mod cpu_qwen35;
pub mod cpu_simd;
pub mod simd_kernels;

use crate::error::RuntimeError;
use crate::kv::disk::RecurrentState;
use crate::kv::{KvCache, KvCacheView, KvPrecision};
use crate::weight::cache::{LayerView, WeightProvider};
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

/// Run-level configuration for the GPU temperature sampler (Option A, Metal
/// `LUMEN_METAL_GPU_SAMPLER=1`). The engine resolves this ONCE per generation
/// from `SamplingParams` + the prompt and passes it to every
/// `decode_token_sampled` call; the backend reads the prompt-history fields only
/// on the fresh-run initialisation (when no pipelined CB is in flight) to seed
/// the GPU RNG-state ring and history frequency array, exactly mirroring the CPU
/// `Xorshift64::new(seed)` + `SamplerState` prompt seeding.
#[derive(Clone)]
pub struct GpuSamplerRunCfg {
    /// 1.0 / temperature (temperature > 0 guaranteed by the engine route gate).
    pub inv_temp: f32,
    /// True iff any penalty (rep != 1, presence != 0, freq != 0) is active.
    pub pen_active: bool,
    /// repetition / presence / frequency penalty values (SamplingParams).
    pub rep: f32,
    pub presence: f32,
    pub freq: f32,
    /// The post-finalizer `Xorshift64::new(seed).state()` u64 — the EXACT RNG
    /// state the CPU sampler would start from, used to seed the GPU RNG `ring[0]`.
    pub rng_seed_state: u64,
    /// The full prompt-history token ids (what the engine recorded into
    /// SamplerState before decode). Used once at fresh-run to seed the GPU
    /// history frequency array so the first sampled token's penalties match the
    /// CPU. Cheap to clone (Arc).
    pub prompt_history: std::sync::Arc<Vec<u32>>,
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
    /// `gpu_k_cache`/`gpu_v_cache`; CUDA: the F32, F16 or BF16 store it was built for). Reject the
    /// mismatch up front so the caller sees an explicit error instead of
    /// downstream silent data corruption.
    ///
    /// Default impl (CPU naive / SIMD) accepts `F32` and `F16`: the CPU
    /// `KvCache` byte buffers are sized by `config.precision`, and the
    /// `KvCacheView` append/read helpers and the CPU attention read those two.
    ///
    /// Backends with a hardcoded precision MUST override this and return
    /// `RuntimeError::Unsupported` for mismatched configs.
    fn validate_kv_precision(&self, precision: KvPrecision) -> Result<(), RuntimeError> {
        crate::kv::kv_precision_supported(
            lumen_format::serving_rules::ServingBackend::Cpu,
            precision,
        )
        .map_err(RuntimeError::Unsupported)
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
        Err(RuntimeError::Compute(
            "batched prefill not supported".into(),
        ))
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

    /// Prepare the backend for the model's weights; called once after `init`.
    ///
    /// The GPU backends upload the layers to device memory. The CPU backends
    /// read every layer, check it against what their compute reads and record
    /// each layer's kind; they refuse to compute, save or restore before this
    /// has succeeded.
    fn preload_weights(&mut self, _weights: &dyn WeightProvider) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Whether the forward loop hands this backend layers exactly as stored
    /// (`WeightProvider::get_layer_raw`) instead of the provider's cached
    /// views. The CPU backends decode stored formats themselves and return
    /// `true`; the default keeps every other backend on the cached views.
    fn reads_raw_layers(&self) -> bool {
        false
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
        Err(RuntimeError::Compute(
            "GPU-resident decode not supported".into(),
        ))
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
        Err(RuntimeError::Compute(
            "GPU-side argmax not supported".into(),
        ))
    }

    /// [`Self::decode_token_greedy`]'s computation for `token_id`, returning the
    /// full logits instead of their argmax and leaving no decode step in flight,
    /// so a session can save the state its greedy continuation reaches. Advances
    /// `kv.seq_len()` once, like `decode_token_greedy`.
    fn decode_token_greedy_logits(
        &self,
        _token_id: u32,
        _weights: &dyn WeightProvider,
        _kv: &mut KvCache,
    ) -> Result<Logits, RuntimeError> {
        Err(RuntimeError::Unsupported(
            "this backend cannot save a session".into(),
        ))
    }

    /// [Option A] GPU temperature-sampled decode returning the token id directly
    /// (Metal only; `LUMEN_METAL_GPU_SAMPLER=1`). The token is SAMPLED on the GPU
    /// (parity-matched to the CPU `sample_logits`) and chained through the lean
    /// pipeline exactly like `decode_token_greedy`, so temp>0 decode pipelines
    /// without the serial CPU sampler + logit readback. Advances `kv.seq_len()`
    /// internally (once per call); the caller must NOT call `advance_seq_len`.
    /// Backends that do not support it return an error and `supports_gpu_sampler`
    /// reports false so the engine never calls it.
    fn decode_token_sampled(
        &self,
        _prev_token: u32,
        _weights: &dyn WeightProvider,
        _kv: &mut KvCache,
        _cfg: &GpuSamplerRunCfg,
    ) -> Result<u32, RuntimeError> {
        Err(RuntimeError::Compute(
            "GPU temperature sampler not supported".into(),
        ))
    }

    /// Whether this backend's `decode_token_sampled` (Option A GPU sampler) path
    /// is active for the given sampling config. The engine consults this to opt
    /// into the GPU-sampled lean driver; default false. Implementations check the
    /// `LUMEN_METAL_GPU_SAMPLER` flag, kernel availability, and that the config is
    /// the subset the kernel reproduces (no top_k/top_p/min_p, full-history
    /// window). `params` is the resolved `SamplingParams`.
    fn supports_gpu_sampler(&self, _params: &crate::sampling::SamplingParams) -> bool {
        false
    }

    /// Reset recurrent state (GDN h_state, conv_state).
    ///
    /// Every backend with GDN layers (Metal, CUDA and the CPU backends) owns
    /// that state itself: a backend serves one live hybrid sequence at a time.
    fn reset_recurrent_state(&self) {}

    /// Reconcile speculative decode-ahead with the host KV cursor before a
    /// warm append repairs the un-fed tail token.
    ///
    /// A pipelined decode path (Metal's lean greedy pipeline) keeps one
    /// committed command buffer in flight beyond the last returned token; that
    /// CB has already processed the tail token device-side (KV slot written,
    /// recurrent state advanced) while the host cursor still lags by one.
    /// Re-feeding the tail in that state advances GDN recurrent state TWICE
    /// and corrupts the continuation.
    ///
    /// Returns `true` when the backend had such a speculative step in flight
    /// (now synchronized): the caller must only advance the host KV cursor
    /// instead of re-feeding the tail. The default (no speculation) returns
    /// `false`.
    fn reconcile_speculative_tail(
        &self,
        _kv: &mut crate::kv::KvCache,
    ) -> Result<bool, RuntimeError> {
        Ok(false)
    }

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
    /// implementation is a no-op for a backend whose KV already lives in
    /// `KvCache` and which has no recurrent state. The CPU backends own their
    /// KV in `KvCache` and override this only to copy their GDN state and to
    /// refuse before `preload_weights` has prepared them.
    ///
    /// Backends overriding this MUST:
    /// - synchronously wait for any in-flight command buffers (Metal:
    ///   submit an empty CB and `commit_and_wait`; CUDA: `device.synchronize`);
    /// - populate `KvCache.set_layer_raw_bytes` for each layer in the
    ///   storage layout declared by `validate_kv_precision`;
    /// - populate the supplied `RecurrentState` when `Some` and the backend
    ///   has GDN state (Metal, the CPU backends); on backends without GDN, leave
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

    /// Whether this backend reads or writes the host-side `KvCache` key/value
    /// mirror.
    ///
    /// CPU and SIMD backends compute attention directly from the host buffers
    /// (via `KvCache::view_mut` in the forward pass), and Metal's batched
    /// prefill also writes them (then [`Self::sync_kv_to_cpu`] serialises them
    /// for disk-KV), so all of those return `true` — the default.
    ///
    /// A fully device-resident backend whose live KV lives in its own VRAM
    /// cache and which never syncs KV back to the host (CUDA: `KvCacheGpu`,
    /// with `sync_kv_to_cpu` unimplemented) returns `false`. The session then
    /// builds its [`KvCache`] via `KvCache::new_without_host_mirror`, skipping
    /// the per-layer host buffers that would otherwise be allocated — and
    /// zeroed — on every session reset for no purpose.
    fn uses_host_kv_mirror(&self) -> bool {
        true
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
    /// The default impl is a no-op; the CPU backends restore their GDN state
    /// here and refuse before `preload_weights` has prepared them.
    fn sync_kv_from_cpu(
        &self,
        _kv: &KvCache,
        _recurrent: Option<&RecurrentState>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Describe the GDN layout this backend uses, if any. Returns `None`
    /// when the backend either has no GDN state or has not yet allocated
    /// it (e.g. a non-GDN model on the Metal backend, or a CPU backend before
    /// `preload_weights`). The disk-save path
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

/// The host memory a CPU backend needs to run `lbc` with a `max_seq_len`-token
/// KV cache stored as `kv`: the layer weights as stored and the raw global
/// planes the provider keeps; the KV cache, allocated for every layer (twice
/// while a session restores, with `restoring`); the F32 embedding, final norm
/// and head, one copy in the provider and one in the backend; with `simd`, the
/// SIMD backend's Q8_0 head on aarch64; the GDN recurrent state; RoPE tables and
/// scratch; and headroom for the allocator, thread stacks and the tokenizer.
pub fn cpu_memory_required(
    lbc: &lumen_format::reader::LbcFile,
    max_seq_len: usize,
    kv: KvPrecision,
    simd: bool,
    restoring: bool,
) -> u64 {
    let hp = &lbc.header.hyperparams;
    let (vocab, hidden) = (hp.vocab_size as u64, hp.hidden_dim as u64);
    let (head_dim, seq) = (hp.head_dim as u64, max_seq_len as u64);
    let layers: u64 = lbc.layer_indices.iter().map(|l| l.layer_length_bytes).sum();
    let globals =
        lbc.header.embedding.length + lbc.header.output_proj.length + lbc.header.final_norm.length;
    let kv_cache = hp.num_layers as u64
        * 2
        * hp.num_kv_heads as u64
        * seq
        * head_dim
        * kv.bytes_per_element() as u64
        * if restoring { 2 } else { 1 };
    let f32_globals = 2 * (2 * vocab * hidden + hidden) * 4;
    let q8_head = if simd && cfg!(target_arch = "aarch64") && hidden % 32 == 0 {
        vocab * hidden / 32 * 34
    } else {
        0
    };
    let g = hp.gdn_dims();
    let gdn_layers = lbc
        .layer_indices
        .iter()
        .filter(|l| l.subtensors.layer_type == Some(1))
        .count() as u64;
    let recurrent = gdn_layers
        * (g.num_v_heads as u64 * g.head_dim as u64 * g.head_dim as u64
            + (g.conv_kernel as u64).saturating_sub(1) * g.qkv_dim() as u64)
        * 4;
    let scratch = (seq * head_dim + hp.num_heads as u64 * seq + vocab) * 4;
    let need = layers + globals + kv_cache + f32_globals + q8_head + recurrent + scratch;
    need + need / 16 + (256 << 20)
}

/// The refusal when `need` bytes at a `max_seq_len`-token context exceed the
/// `available` ones; `None` when they fit.
fn cpu_memory_refusal(need: u64, max_seq_len: usize, available: u64) -> Option<String> {
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64);
    (need > available).then(|| {
        format!(
            "this model needs about {} of memory on the CPU backend at a {max_seq_len}-token \
             context, but {} is available; free memory, choose a smaller quantization or a \
             shorter --context-len",
            gib(need),
            gib(available)
        )
    })
}

/// Refuse to run `lbc` on a CPU backend when [`cpu_memory_required`] exceeds
/// [`host_available_bytes`]; nothing is refused where the platform reports no
/// available memory.
pub fn check_cpu_memory(
    lbc: &lumen_format::reader::LbcFile,
    max_seq_len: usize,
    kv: KvPrecision,
    simd: bool,
    restoring: bool,
) -> Result<(), RuntimeError> {
    let Some(available) = host_available_bytes() else {
        return Ok(());
    };
    let need = cpu_memory_required(lbc, max_seq_len, kv, simd, restoring);
    match cpu_memory_refusal(need, max_seq_len, available) {
        Some(message) => Err(RuntimeError::Unsupported(message)),
        None => Ok(()),
    }
}

/// Bytes of memory this process can still claim, or `None` where the platform
/// reports nothing.
///
/// - Linux: the smaller of `MemAvailable` and the room under every cgroup
///   memory limit from the process's cgroup up to the root (see
///   [`cgroup_room`]).
/// - macOS: physical memory times the kernel's memorystatus level, the
///   percentage of memory it can make available without terminating
///   processes (what `memory_pressure` reports as the free percentage).
#[cfg(target_os = "linux")]
pub fn host_available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    let host = kib.saturating_mul(1024);
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let root = std::path::Path::new("/sys/fs/cgroup");
    Some(cgroup_room(&cgroups, root).map_or(host, |room| room.min(host)))
}

#[cfg(target_os = "macos")]
pub fn host_available_bytes() -> Option<u64> {
    fn sysctl<T: Copy + Default>(name: &std::ffi::CStr) -> Option<T> {
        let mut value = T::default();
        let mut len = std::mem::size_of::<T>();
        // SAFETY: `value` is a writable `T` and `len` its size; sysctlbyname
        // writes at most `len` bytes and reports the size it wrote.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut T as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len == std::mem::size_of::<T>()).then_some(value)
    }
    let total: u64 = sysctl(c"hw.memsize")?;
    let percent: u32 = sysctl(c"kern.memorystatus_level")?;
    Some(total / 100 * percent as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn host_available_bytes() -> Option<u64> {
    None
}

/// The least room under any cgroup memory limit on the path from the process's
/// cgroup up to the root of the hierarchy mounted at `root`, given
/// `proc_self_cgroup` (the content of `/proc/self/cgroup`); `None` when no
/// level sets a limit. At each level the room is the limit less the usage that
/// is not reclaimable file cache: cgroup v2 `memory.max - (memory.current -
/// inactive_file - active_file)`, v1 the same from `memory.limit_in_bytes`,
/// `memory.usage_in_bytes` and `total_inactive_file` / `total_active_file`.
#[cfg(any(target_os = "linux", test))]
fn cgroup_room(proc_self_cgroup: &str, root: &std::path::Path) -> Option<u64> {
    let number = |p: std::path::PathBuf| -> Option<u64> {
        std::fs::read_to_string(p).ok()?.trim().parse().ok()
    };
    let mut room: Option<u64> = None;
    for line in proc_self_cgroup.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (base, limit, usage, cache) = if controllers.is_empty() {
            (
                root.to_path_buf(),
                "memory.max",
                "memory.current",
                ["inactive_file", "active_file"],
            )
        } else if controllers.split(',').any(|c| c == "memory") {
            (
                root.join("memory"),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                ["total_inactive_file", "total_active_file"],
            )
        } else {
            continue;
        };
        let mut level = std::path::PathBuf::from(path.trim_start_matches('/'));
        loop {
            let dir = base.join(&level);
            if let (Some(limit), Some(usage)) = (number(dir.join(limit)), number(dir.join(usage))) {
                let reclaimable: u64 = std::fs::read_to_string(dir.join("memory.stat"))
                    .unwrap_or_default()
                    .lines()
                    .filter_map(|l| l.split_once(' '))
                    .filter(|(k, _)| cache.contains(k))
                    .filter_map(|(_, v)| v.trim().parse::<u64>().ok())
                    .sum();
                let left = limit.saturating_sub(usage.saturating_sub(reclaimable));
                room = Some(room.map_or(left, |r| r.min(left)));
            }
            if !level.pop() {
                break;
            }
        }
    }
    room
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    use std::path::Path;

    const GIB: u64 = 1 << 30;

    fn write(dir: &Path, rel: &str, files: &[(&str, String)]) {
        let d = dir.join(rel);
        std::fs::create_dir_all(&d).unwrap();
        for (name, body) in files {
            std::fs::write(d.join(name), body).unwrap();
        }
    }

    /// cgroup v2: the room is the tightest level's limit less its usage that
    /// is not file cache; an unlimited level and the root count for nothing.
    #[test]
    fn cgroup_v2_room_is_the_tightest_level_less_unreclaimable_usage() {
        let root = tempfile::tempdir().unwrap();
        let stat = |inactive: u64, active: u64| {
            format!("anon 1\ninactive_file {inactive}\nactive_file {active}\nshmem 7\n")
        };
        // 32 GiB limit with 24 GiB in use, 2 GiB of it file cache: 10 GiB left.
        write(
            root.path(),
            "pod",
            &[
                ("memory.max", format!("{}\n", 32 * GIB)),
                ("memory.current", format!("{}\n", 24 * GIB)),
                ("memory.stat", stat(GIB, GIB)),
            ],
        );
        // no limit of its own
        write(
            root.path(),
            "pod/app",
            &[
                ("memory.max", "max\n".into()),
                ("memory.current", format!("{}\n", 20 * GIB)),
                ("memory.stat", stat(0, 0)),
            ],
        );
        assert_eq!(cgroup_room("0::/pod/app\n", root.path()), Some(10 * GIB));
        // a tighter inner limit wins: 12 GiB less (11 - 3) GiB in use
        write(
            root.path(),
            "pod/app",
            &[
                ("memory.max", format!("{}\n", 12 * GIB)),
                ("memory.current", format!("{}\n", 11 * GIB)),
                ("memory.stat", stat(2 * GIB, GIB)),
            ],
        );
        assert_eq!(cgroup_room("0::/pod/app\n", root.path()), Some(4 * GIB));
        // usage past the limit leaves no room rather than wrapping
        write(
            root.path(),
            "pod/app",
            &[
                ("memory.max", format!("{}\n", GIB)),
                ("memory.current", format!("{}\n", 3 * GIB)),
                ("memory.stat", stat(0, 0)),
            ],
        );
        assert_eq!(cgroup_room("0::/pod/app\n", root.path()), Some(0));
        // nothing limited anywhere
        assert_eq!(cgroup_room("0::/elsewhere\n", root.path()), None);
    }

    #[test]
    fn cgroup_v1_room_reads_the_memory_controller() {
        let root = tempfile::tempdir().unwrap();
        write(
            root.path(),
            "memory/docker/abc",
            &[
                ("memory.limit_in_bytes", format!("{}\n", 16 * GIB)),
                ("memory.usage_in_bytes", format!("{}\n", 9 * GIB)),
                (
                    "memory.stat",
                    format!("cache 5\ntotal_inactive_file {GIB}\ntotal_active_file {GIB}\n"),
                ),
            ],
        );
        write(
            root.path(),
            "memory",
            &[
                ("memory.limit_in_bytes", "9223372036854771712\n".into()),
                ("memory.usage_in_bytes", format!("{}\n", 40 * GIB)),
            ],
        );
        let proc = "12:cpu,cpuacct:/docker/abc\n11:memory:/docker/abc\n";
        assert_eq!(cgroup_room(proc, root.path()), Some(9 * GIB));
    }

    /// The need counts the weights, the F32 globals twice and a KV cache that a
    /// restore holds twice.
    #[test]
    fn cpu_memory_counts_the_weights_globals_and_kv_cache() {
        use lumen_format::test_model::{generate_test_model, TestModelConfig};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.lbc");
        std::fs::write(&path, generate_test_model(&TestModelConfig::default())).unwrap();
        let lbc = lumen_format::reader::LbcFile::open(&path).unwrap();
        let hp = lbc.header.hyperparams;
        let h = &lbc.header;
        let layers: u64 = lbc.layer_indices.iter().map(|l| l.layer_length_bytes).sum();
        let weights = layers + h.embedding.length + h.output_proj.length + h.final_norm.length;
        let f32_globals = 2 * (2 * hp.vocab_size as u64 + 1) * hp.hidden_dim as u64 * 4;
        let kv = hp.num_layers as u64 * 2 * hp.num_kv_heads as u64 * 64 * hp.head_dim as u64 * 4;
        let need = cpu_memory_required(&lbc, 64, KvPrecision::F32, false, false);
        assert!(need >= weights + f32_globals + kv + (256 << 20), "{need}");
        let restoring = cpu_memory_required(&lbc, 64, KvPrecision::F32, false, true);
        assert!(restoring - need >= kv, "{restoring} vs {need}");
        let f16 = cpu_memory_required(&lbc, 64, KvPrecision::F16, false, false);
        assert!(need - f16 >= kv / 2, "{need} vs {f16}");
    }

    #[test]
    fn the_refusal_names_the_need_and_what_is_available() {
        assert_eq!(cpu_memory_refusal(27 * GIB, 4096, 27 * GIB), None);
        let message = cpu_memory_refusal(27 * GIB, 4096, 27 * GIB - GIB / 2).unwrap();
        assert!(
            message.starts_with(
                "this model needs about 27.0 GiB of memory on the CPU backend at a 4096-token \
                 context, but 26.5 GiB is available"
            ),
            "{message}"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn this_host_reports_its_available_memory() {
        let available = host_available_bytes().expect("the platform reports memory");
        assert!(available > 0);
    }
}
