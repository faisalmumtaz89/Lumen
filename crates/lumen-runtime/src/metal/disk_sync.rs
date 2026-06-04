//! GPU↔CPU synchronisation for the Metal backend.
//!
//! The disk-KV save path streams bytes verbatim out of the CPU `KvCache`
//! mirror and the caller-supplied `RecurrentState`. Those CPU buffers are
//! only valid while inference is "between" forward passes AND after the
//! Metal command queue has drained — the K/V on disk is exactly what the
//! GPU computed. This module provides the two extraction helpers
//! (`sync_kv_to_cpu_impl`, `sync_kv_from_cpu_impl`) that the
//! `ComputeBackend` overrides in `backend_impl.rs` delegate to.
//!
//! # Layout
//!
//! The CPU `KvCache` is laid out head-first `[head][pos][dim]` (see
//! `crates/lumen-runtime/src/kv/mod.rs`); the GPU buffers use
//! Metal-native shapes:
//!
//! - `gpu_k_cache[layer]` -- row-major `[pos][kv_head*head_dim + d]`,
//!   F16, total `max_seq_len * kv_dim * 2` bytes.
//! - `gpu_v_cache[layer]` -- transposed `[kv_head*head_dim + d][pos]`,
//!   F16, same byte count.
//! - `gdn_h_states[gdn_idx]` -- contiguous F32, identical CPU/GPU layout.
//! - `gdn_conv_states[gdn_idx]` -- contiguous F32, identical layout.
//!
//! K and V must therefore be element-wise transposed to/from the CPU
//! mirror — a byte blit would corrupt every read. Recurrent buffers are
//! transferred verbatim.
//!
//! # Synchronisation
//!
//! Each direction submits an empty command buffer and `commit_and_wait()`s
//! before touching `contents()` on any GPU buffer. Apple Silicon's
//! `StorageModeShared` buffers are CPU-visible at any time, but Metal's
//! hazard tracker only guarantees that completed-CB writes are observable
//! after the CB completes. The drain is therefore a correctness fence,
//! not a memory transfer.

use super::MetalF32Backend;
use crate::error::RuntimeError;
use crate::kv::disk::{GdnLayout, RecurrentState, RECURRENT_DTYPE_F32};
use crate::kv::{KvCache, KvPrecision};

impl MetalF32Backend {
    /// Submit an empty command buffer and block until completion. Acts as
    /// a hazard fence so any subsequent CPU read of a shared-mode buffer
    /// observes the result of all previously submitted GPU work.
    fn drain_gpu_work(&self) -> Result<(), RuntimeError> {
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute(
                "Metal disk-KV sync: failed to allocate command buffer for drain".into(),
            )
        })?;
        cmd.commit_and_wait();
        Ok(())
    }

    /// Transpose the GPU-native K layout (`[pos][head*head_dim + d]`) into
    /// the CPU head-first layout (`[head][pos][dim]`), F16 in both cases.
    ///
    /// Source bytes come straight from `gpu_k_cache[layer].contents()`; we
    /// reinterpret as `&[u16]` so the inner copy is a single 2-byte word
    /// per element. The transposition copies `seq_len` positions per head,
    /// leaving the trailing `(max_seq_len - seq_len)` positions zeroed
    /// (the disk format writes the full pre-allocated buffer either way,
    /// and unused positions stay at the zero-init values).
    fn transpose_gpu_k_to_cpu_f16(
        src_words: &[u16],
        dst_bytes: &mut [u8],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        debug_assert_eq!(src_words.len(), max_seq_len * num_kv_heads * head_dim);
        debug_assert_eq!(dst_bytes.len(), max_seq_len * num_kv_heads * head_dim * 2);
        // Ensure trailing positions are zeroed (defensive — the CPU
        // buffer starts at zero, but in case of repeated syncs we
        // overwrite the active range only and leave the rest alone).
        for h in 0..num_kv_heads {
            for pos in 0..seq_len {
                // Source word index for `(pos, head=h, dim=0..head_dim)`.
                let src_base = pos * (num_kv_heads * head_dim) + h * head_dim;
                // Destination word index for `(head=h, pos, dim=0..head_dim)`.
                let dst_base = (h * max_seq_len * head_dim + pos * head_dim) * 2;
                for d in 0..head_dim {
                    let w = src_words[src_base + d].to_le_bytes();
                    let off = dst_base + d * 2;
                    dst_bytes[off] = w[0];
                    dst_bytes[off + 1] = w[1];
                }
            }
        }
    }

    /// Transpose the GPU-native V layout (`[head*head_dim + d][pos]`) into
    /// the CPU head-first layout (`[head][pos][dim]`). Both F16.
    fn transpose_gpu_v_to_cpu_f16(
        src_words: &[u16],
        dst_bytes: &mut [u8],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        debug_assert_eq!(src_words.len(), max_seq_len * num_kv_heads * head_dim);
        debug_assert_eq!(dst_bytes.len(), max_seq_len * num_kv_heads * head_dim * 2);
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                // V GPU index for `(head=h, d, pos)`: (h*head_dim + d) * max_seq_len + pos.
                let src_base = (h * head_dim + d) * max_seq_len;
                for pos in 0..seq_len {
                    let w = src_words[src_base + pos].to_le_bytes();
                    let off = (h * max_seq_len * head_dim + pos * head_dim + d) * 2;
                    dst_bytes[off] = w[0];
                    dst_bytes[off + 1] = w[1];
                }
            }
        }
    }

    /// Inverse transposition for K, CPU → GPU.
    fn transpose_cpu_k_to_gpu_f16(
        src_bytes: &[u8],
        dst_words: &mut [u16],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        debug_assert_eq!(src_bytes.len(), max_seq_len * num_kv_heads * head_dim * 2);
        debug_assert_eq!(dst_words.len(), max_seq_len * num_kv_heads * head_dim);
        // Zero the destination first so positions ≥ seq_len match the
        // GPU's zero-init state (otherwise stale GPU bytes from a prior
        // session would persist past the active window).
        for w in dst_words.iter_mut() {
            *w = 0;
        }
        for h in 0..num_kv_heads {
            for pos in 0..seq_len {
                let dst_base = pos * (num_kv_heads * head_dim) + h * head_dim;
                let src_base = (h * max_seq_len * head_dim + pos * head_dim) * 2;
                for d in 0..head_dim {
                    let off = src_base + d * 2;
                    let w = u16::from_le_bytes([src_bytes[off], src_bytes[off + 1]]);
                    dst_words[dst_base + d] = w;
                }
            }
        }
    }

    /// Inverse transposition for V, CPU → GPU.
    fn transpose_cpu_v_to_gpu_f16(
        src_bytes: &[u8],
        dst_words: &mut [u16],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) {
        debug_assert_eq!(src_bytes.len(), max_seq_len * num_kv_heads * head_dim * 2);
        debug_assert_eq!(dst_words.len(), max_seq_len * num_kv_heads * head_dim);
        for w in dst_words.iter_mut() {
            *w = 0;
        }
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let dst_base = (h * head_dim + d) * max_seq_len;
                for pos in 0..seq_len {
                    let off = (h * max_seq_len * head_dim + pos * head_dim + d) * 2;
                    let w = u16::from_le_bytes([src_bytes[off], src_bytes[off + 1]]);
                    dst_words[dst_base + pos] = w;
                }
            }
        }
    }

    /// `ComputeBackend::sync_kv_to_cpu` implementation.
    pub(crate) fn sync_kv_to_cpu_impl(
        &self,
        kv: &mut KvCache,
        recurrent: Option<&mut RecurrentState>,
    ) -> Result<(), RuntimeError> {
        //  invariant: Metal is F16-only.
        let cfg = kv.config().clone();
        if cfg.precision != KvPrecision::F16 {
            return Err(RuntimeError::Unsupported(format!(
                "Metal sync_kv_to_cpu: KV precision must be F16 (got {:?}); \
                 see ComputeBackend::validate_kv_precision",
                cfg.precision
            )));
        }
        self.drain_gpu_work()?;

        let num_layers = cfg.num_layers;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let max_seq_len = cfg.max_seq_len;
        let seq_len = kv.seq_len();

        // The lock is held only for the per-layer GPU reads + transpose so
        // we never block the engine on a multi-second I/O step.
        for layer in 0..num_layers {
            // Snapshot the GPU bytes for this layer into a local Vec, then
            // release the scratch lock before touching `kv` (which is the
            // mutable borrow that prevents nested locks).
            let (k_src, v_src) = {
                let scratch_guard = self.scratch.lock().map_err(|_| {
                    RuntimeError::Compute("Metal sync_kv_to_cpu: scratch lock poisoned".into())
                })?;
                let s = scratch_guard.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "Metal sync_kv_to_cpu: scratch not initialised (call init() first)".into(),
                    )
                })?;
                let expected_words = max_seq_len * num_kv_heads * head_dim;
                let k_buf = &s.gpu_k_cache[layer];
                let v_buf = &s.gpu_v_cache[layer];
                if (k_buf.length() as usize) < expected_words * 2
                    || (v_buf.length() as usize) < expected_words * 2
                {
                    return Err(RuntimeError::Compute(format!(
                        "Metal sync_kv_to_cpu: layer {layer} GPU buffer too small \
                         (k={}, v={}, expected={} bytes)",
                        k_buf.length(),
                        v_buf.length(),
                        expected_words * 2,
                    )));
                }
                let mut k_src = vec![0u16; expected_words];
                let mut v_src = vec![0u16; expected_words];
                k_buf.read_u16(&mut k_src);
                v_buf.read_u16(&mut v_src);
                (k_src, v_src)
            };

            let (k_dst, v_dst) = kv.layer_raw_bytes_mut(layer)?;
            // Zero only the active range so we never carry stale CPU bytes
            // (`positions >= seq_len` stay at their existing zero state).
            for h in 0..num_kv_heads {
                for pos in 0..seq_len {
                    let off = (h * max_seq_len * head_dim + pos * head_dim) * 2;
                    for d in 0..head_dim {
                        let i = off + d * 2;
                        k_dst[i] = 0;
                        k_dst[i + 1] = 0;
                        v_dst[i] = 0;
                        v_dst[i + 1] = 0;
                    }
                }
            }
            Self::transpose_gpu_k_to_cpu_f16(
                &k_src, k_dst, seq_len, num_kv_heads, head_dim, max_seq_len,
            );
            Self::transpose_gpu_v_to_cpu_f16(
                &v_src, v_dst, seq_len, num_kv_heads, head_dim, max_seq_len,
            );
        }

        // Recurrent state — copy GPU F32 → CPU F32 verbatim (same layout).
        if let Some(rec) = recurrent {
            let layout = rec.layout;
            let scratch_guard = self.scratch.lock().map_err(|_| {
                RuntimeError::Compute(
                    "Metal sync_kv_to_cpu: scratch lock poisoned during recurrent read".into(),
                )
            })?;
            let s = scratch_guard.as_ref().ok_or_else(|| {
                RuntimeError::Compute(
                    "Metal sync_kv_to_cpu: scratch missing during recurrent read".into(),
                )
            })?;
            if (s.gdn_num_layers as u32) != layout.num_gdn_layers {
                return Err(RuntimeError::Compute(format!(
                    "Metal sync_kv_to_cpu: recurrent layout num_gdn_layers={} but \
                     backend has {} GDN layers allocated; reset_recurrent_state was \
                     not called or the layout was synthesized from the wrong model",
                    layout.num_gdn_layers, s.gdn_num_layers,
                )));
            }
            if layout.gdn_dtype_tag != RECURRENT_DTYPE_F32 {
                return Err(RuntimeError::Unsupported(format!(
                    "Metal sync_kv_to_cpu: only F32 recurrent dtype supported (got tag {})",
                    layout.gdn_dtype_tag,
                )));
            }
            let h_floats = layout.h_state_bytes_per_layer() / 4;
            let c_floats = layout.conv_state_bytes_per_layer() / 4;
            for gdn_idx in 0..(layout.num_gdn_layers as usize) {
                let h_buf = &s.gdn_h_states[gdn_idx];
                let c_buf = &s.gdn_conv_states[gdn_idx];
                if (h_buf.length() as usize) < h_floats * 4 {
                    return Err(RuntimeError::Compute(format!(
                        "Metal sync_kv_to_cpu: GDN layer {gdn_idx} h_state buffer too \
                         small ({} bytes, expected {})",
                        h_buf.length(),
                        h_floats * 4,
                    )));
                }
                if (c_buf.length() as usize) < c_floats * 4 {
                    return Err(RuntimeError::Compute(format!(
                        "Metal sync_kv_to_cpu: GDN layer {gdn_idx} conv_state buffer too \
                         small ({} bytes, expected {})",
                        c_buf.length(),
                        c_floats * 4,
                    )));
                }
                let mut h_floats_vec = vec![0.0f32; h_floats];
                h_buf.read_f32(&mut h_floats_vec);
                // Pack F32 → LE bytes into the CPU mirror.
                let h_bytes = &mut rec.h_states[gdn_idx];
                for (i, f) in h_floats_vec.iter().enumerate() {
                    let b = f.to_le_bytes();
                    h_bytes[i * 4] = b[0];
                    h_bytes[i * 4 + 1] = b[1];
                    h_bytes[i * 4 + 2] = b[2];
                    h_bytes[i * 4 + 3] = b[3];
                }
                let mut c_floats_vec = vec![0.0f32; c_floats];
                c_buf.read_f32(&mut c_floats_vec);
                let c_bytes = &mut rec.conv_states[gdn_idx];
                for (i, f) in c_floats_vec.iter().enumerate() {
                    let b = f.to_le_bytes();
                    c_bytes[i * 4] = b[0];
                    c_bytes[i * 4 + 1] = b[1];
                    c_bytes[i * 4 + 2] = b[2];
                    c_bytes[i * 4 + 3] = b[3];
                }
                rec.conv_positions[gdn_idx] = s.gdn_conv_positions[gdn_idx];
            }
        }
        Ok(())
    }

    /// `ComputeBackend::sync_kv_from_cpu` implementation.
    pub(crate) fn sync_kv_from_cpu_impl(
        &self,
        kv: &KvCache,
        recurrent: Option<&RecurrentState>,
    ) -> Result<(), RuntimeError> {
        let cfg = kv.config().clone();
        if cfg.precision != KvPrecision::F16 {
            return Err(RuntimeError::Unsupported(format!(
                "Metal sync_kv_from_cpu: KV precision must be F16 (got {:?})",
                cfg.precision
            )));
        }
        self.drain_gpu_work()?;

        let num_layers = cfg.num_layers;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let max_seq_len = cfg.max_seq_len;
        let seq_len = kv.seq_len();

        for layer in 0..num_layers {
            let (k_src, v_src) = kv.layer_raw_bytes(layer)?;
            let expected_words = max_seq_len * num_kv_heads * head_dim;
            // Stage GPU words on the heap, transpose, then upload in one go.
            let mut k_gpu = vec![0u16; expected_words];
            let mut v_gpu = vec![0u16; expected_words];
            Self::transpose_cpu_k_to_gpu_f16(
                k_src, &mut k_gpu, seq_len, num_kv_heads, head_dim, max_seq_len,
            );
            Self::transpose_cpu_v_to_gpu_f16(
                v_src, &mut v_gpu, seq_len, num_kv_heads, head_dim, max_seq_len,
            );
            let scratch_guard = self.scratch.lock().map_err(|_| {
                RuntimeError::Compute(
                    "Metal sync_kv_from_cpu: scratch lock poisoned".into(),
                )
            })?;
            let s = scratch_guard.as_ref().ok_or_else(|| {
                RuntimeError::Compute(
                    "Metal sync_kv_from_cpu: scratch not initialised".into(),
                )
            })?;
            let k_buf = &s.gpu_k_cache[layer];
            let v_buf = &s.gpu_v_cache[layer];
            if (k_buf.length() as usize) < expected_words * 2
                || (v_buf.length() as usize) < expected_words * 2
            {
                return Err(RuntimeError::Compute(format!(
                    "Metal sync_kv_from_cpu: layer {layer} GPU buffer too small"
                )));
            }
            k_buf.write_u16(&k_gpu);
            v_buf.write_u16(&v_gpu);
        }

        if let Some(rec) = recurrent {
            let layout = rec.layout;
            if layout.gdn_dtype_tag != RECURRENT_DTYPE_F32 {
                return Err(RuntimeError::Unsupported(format!(
                    "Metal sync_kv_from_cpu: only F32 recurrent dtype supported (got tag {})",
                    layout.gdn_dtype_tag,
                )));
            }
            // Allocate target buffers if the backend hasn't seen any GDN
            // dispatch yet (post-reset cold-load). We mirror the per-layer
            // allocation pattern from `backend_impl.rs::compute_layer`.
            self.ensure_gdn_storage_for_layout(&layout)?;

            let mut scratch_guard = self.scratch.lock().map_err(|_| {
                RuntimeError::Compute(
                    "Metal sync_kv_from_cpu: scratch lock poisoned during recurrent restore".into(),
                )
            })?;
            let s = scratch_guard.as_mut().ok_or_else(|| {
                RuntimeError::Compute(
                    "Metal sync_kv_from_cpu: scratch missing during recurrent restore".into(),
                )
            })?;
            if (s.gdn_num_layers as u32) != layout.num_gdn_layers {
                return Err(RuntimeError::Compute(format!(
                    "Metal sync_kv_from_cpu: backend has {} GDN layers but recurrent \
                     state declares {}; layout mismatch",
                    s.gdn_num_layers, layout.num_gdn_layers,
                )));
            }
            let h_floats = layout.h_state_bytes_per_layer() / 4;
            let c_floats = layout.conv_state_bytes_per_layer() / 4;
            for gdn_idx in 0..(layout.num_gdn_layers as usize) {
                let h_bytes = &rec.h_states[gdn_idx];
                let c_bytes = &rec.conv_states[gdn_idx];
                let mut h_floats_vec = vec![0.0f32; h_floats];
                for i in 0..h_floats {
                    h_floats_vec[i] = f32::from_le_bytes([
                        h_bytes[i * 4],
                        h_bytes[i * 4 + 1],
                        h_bytes[i * 4 + 2],
                        h_bytes[i * 4 + 3],
                    ]);
                }
                let mut c_floats_vec = vec![0.0f32; c_floats];
                for i in 0..c_floats {
                    c_floats_vec[i] = f32::from_le_bytes([
                        c_bytes[i * 4],
                        c_bytes[i * 4 + 1],
                        c_bytes[i * 4 + 2],
                        c_bytes[i * 4 + 3],
                    ]);
                }
                s.gdn_h_states[gdn_idx].write_f32(&h_floats_vec);
                s.gdn_conv_states[gdn_idx].write_f32(&c_floats_vec);
                s.gdn_conv_positions[gdn_idx] = rec.conv_positions[gdn_idx];
            }
        }
        Ok(())
    }

    /// Allocate `gdn_h_states` / `gdn_conv_states` for the requested
    /// layout on a cold backend (post-reset, no GDN dispatch yet).
    ///
    /// No-op when the backend already has GDN buffers; rejects when the
    /// existing allocations disagree with the requested layout (the
    /// caller MUST `reset_recurrent_state` and `ensure_gdn_storage_for_layout`
    /// again to switch layouts — but in practice the engine never does
    /// because GDN layout is model-fixed).
    fn ensure_gdn_storage_for_layout(
        &self,
        layout: &GdnLayout,
    ) -> Result<(), RuntimeError> {
        let mut scratch_guard = self.scratch.lock().map_err(|_| {
            RuntimeError::Compute("ensure_gdn_storage_for_layout: scratch lock poisoned".into())
        })?;
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute(
                "ensure_gdn_storage_for_layout: scratch not initialised".into(),
            )
        })?;
        let want_layers = layout.num_gdn_layers as usize;
        let h_state_size = layout.h_state_bytes_per_layer() / 4;
        let conv_state_size = layout.conv_state_bytes_per_layer() / 4;

        if s.gdn_num_layers == 0 {
            // First touch — allocate from scratch in the layout the caller
            // declares (matches the per-layer allocator in compute_layer).
            for _ in 0..want_layers {
                let h_buf = self
                    .device
                    .new_buffer(h_state_size * 4)
                    .ok_or_else(|| {
                        RuntimeError::Compute("disk-restore: failed to allocate GDN h_state".into())
                    })?;
                h_buf.write_f32(&vec![0.0f32; h_state_size]);
                let c_buf = self
                    .device
                    .new_buffer(conv_state_size * 4)
                    .ok_or_else(|| {
                        RuntimeError::Compute(
                            "disk-restore: failed to allocate GDN conv_state".into(),
                        )
                    })?;
                c_buf.write_f32(&vec![0.0f32; conv_state_size]);
                s.gdn_h_states.push(h_buf);
                s.gdn_conv_states.push(c_buf);
                s.gdn_conv_positions.push(0);
            }
            s.gdn_conv_kernel_size = layout.gdn_conv_kernel_size as usize;
            s.gdn_num_layers = want_layers;
            return Ok(());
        }

        if s.gdn_num_layers != want_layers {
            return Err(RuntimeError::Compute(format!(
                "disk-restore: backend has {} GDN layers but caller expects {}; \
                 layout mismatch",
                s.gdn_num_layers, want_layers,
            )));
        }
        // Already-allocated layers — verify the byte sizes match before
        // accepting the restore. If a future model changes GDN dimensions
        // mid-flight we would see a mismatch here instead of corruption.
        for gdn_idx in 0..want_layers {
            let want_h_bytes = h_state_size * 4;
            let want_c_bytes = conv_state_size * 4;
            if (s.gdn_h_states[gdn_idx].length() as usize) != want_h_bytes {
                return Err(RuntimeError::Compute(format!(
                    "disk-restore: GDN layer {gdn_idx} h_state size {} != expected {}",
                    s.gdn_h_states[gdn_idx].length(),
                    want_h_bytes,
                )));
            }
            if (s.gdn_conv_states[gdn_idx].length() as usize) != want_c_bytes {
                return Err(RuntimeError::Compute(format!(
                    "disk-restore: GDN layer {gdn_idx} conv_state size {} != expected {}",
                    s.gdn_conv_states[gdn_idx].length(),
                    want_c_bytes,
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Element-wise sanity: K transpose is its own inverse (GPU → CPU →
    /// GPU yields identical bytes for every position < seq_len).
    #[test]
    fn k_transpose_round_trip() {
        let max_seq_len = 8;
        let num_kv_heads = 3;
        let head_dim = 4;
        let seq_len = 5;
        let total = max_seq_len * num_kv_heads * head_dim;

        // Seed the GPU-native buffer with a distinct value per element so
        // we can detect any indexing bug.
        let mut gpu_src = vec![0u16; total];
        for pos in 0..max_seq_len {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let v = ((pos * 0x100 + h * 0x10 + d) & 0xFFFF) as u16;
                    gpu_src[pos * (num_kv_heads * head_dim) + h * head_dim + d] = v;
                }
            }
        }
        let mut cpu_bytes = vec![0u8; total * 2];
        MetalF32Backend::transpose_gpu_k_to_cpu_f16(
            &gpu_src,
            &mut cpu_bytes,
            seq_len,
            num_kv_heads,
            head_dim,
            max_seq_len,
        );
        let mut gpu_round = vec![0u16; total];
        MetalF32Backend::transpose_cpu_k_to_gpu_f16(
            &cpu_bytes,
            &mut gpu_round,
            seq_len,
            num_kv_heads,
            head_dim,
            max_seq_len,
        );
        // The first `seq_len` positions in every head must round-trip;
        // positions ≥ seq_len are zeroed by the CPU→GPU helper.
        for pos in 0..seq_len {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let i = pos * (num_kv_heads * head_dim) + h * head_dim + d;
                    assert_eq!(
                        gpu_src[i], gpu_round[i],
                        "K round-trip differs at pos={pos} h={h} d={d}"
                    );
                }
            }
        }
        for pos in seq_len..max_seq_len {
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let i = pos * (num_kv_heads * head_dim) + h * head_dim + d;
                    assert_eq!(gpu_round[i], 0, "K tail must zero past seq_len");
                }
            }
        }
    }

    /// V transpose round-trip — same contract as K but for the transposed
    /// layout `[head*head_dim+d][pos]`.
    #[test]
    fn v_transpose_round_trip() {
        let max_seq_len = 8;
        let num_kv_heads = 3;
        let head_dim = 4;
        let seq_len = 5;
        let total = max_seq_len * num_kv_heads * head_dim;

        let mut gpu_src = vec![0u16; total];
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                for pos in 0..max_seq_len {
                    let v = ((h * 0x1000 + d * 0x100 + pos) & 0xFFFF) as u16;
                    gpu_src[(h * head_dim + d) * max_seq_len + pos] = v;
                }
            }
        }
        let mut cpu_bytes = vec![0u8; total * 2];
        MetalF32Backend::transpose_gpu_v_to_cpu_f16(
            &gpu_src,
            &mut cpu_bytes,
            seq_len,
            num_kv_heads,
            head_dim,
            max_seq_len,
        );
        let mut gpu_round = vec![0u16; total];
        MetalF32Backend::transpose_cpu_v_to_gpu_f16(
            &cpu_bytes,
            &mut gpu_round,
            seq_len,
            num_kv_heads,
            head_dim,
            max_seq_len,
        );
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                for pos in 0..seq_len {
                    let i = (h * head_dim + d) * max_seq_len + pos;
                    assert_eq!(gpu_src[i], gpu_round[i],
                        "V round-trip differs at h={h} d={d} pos={pos}");
                }
                for pos in seq_len..max_seq_len {
                    let i = (h * head_dim + d) * max_seq_len + pos;
                    assert_eq!(gpu_round[i], 0, "V tail must zero past seq_len");
                }
            }
        }
    }

    /// The CPU mirror produced by K transpose matches the head-first
    /// `[head][pos][dim]` layout the disk format expects. We verify a
    /// single known byte position to catch off-by-one head-dim arithmetic.
    #[test]
    fn k_transpose_writes_head_first_cpu_layout() {
        let max_seq_len = 4;
        let num_kv_heads = 2;
        let head_dim = 3;
        let seq_len = 2;
        let total = max_seq_len * num_kv_heads * head_dim;
        let mut gpu_src = vec![0u16; total];
        // Mark element (pos=1, head=1, d=2) so it has a unique value.
        let marker: u16 = 0xBEEF;
        gpu_src[1 * (num_kv_heads * head_dim) + 1 * head_dim + 2] = marker;
        let mut cpu_bytes = vec![0u8; total * 2];
        MetalF32Backend::transpose_gpu_k_to_cpu_f16(
            &gpu_src, &mut cpu_bytes, seq_len, num_kv_heads, head_dim, max_seq_len,
        );
        // CPU expects `[head=1, pos=1, d=2]` at byte offset
        // `(1*max_seq_len*head_dim + 1*head_dim + 2) * 2`.
        let off = (1 * max_seq_len * head_dim + 1 * head_dim + 2) * 2;
        let got = u16::from_le_bytes([cpu_bytes[off], cpu_bytes[off + 1]]);
        assert_eq!(got, marker, "K transpose dest byte mismatch");
    }

    // ---- GDN extract/restore round-trip on a live Metal backend ----
    //
    // These tests require the Metal device, so they are macOS-only and need
    // a working M-series GPU. They cover the full extract path: allocate GDN
    // buffers via `sync_kv_from_cpu`, write known F32 patterns, then read
    // them back via `sync_kv_to_cpu` and assert byte-equality on the
    // raw recurrent state. They DO NOT touch the K/V transposition (covered
    // by the unit tests above) — the test cfg gives both paths independent
    // failure surfaces.

    use crate::compute::ComputeBackend;
    use crate::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_format::ModelHyperparams;

    fn make_metal_backend_for_tests() -> Option<MetalF32Backend> {
        // Use the smallest viable Qwen3.5-9B-shaped hyperparams so allocation
        // is cheap. The backend's init() honours these dimensions for its
        // GPU buffers (KV cache, rope tables, etc.).
        let hp = ModelHyperparams {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_dim: 64,
            vocab_size: 32,
            max_seq_len: 16,
            norm_eps: 1e-5,
            rope_params: None,
            num_experts: None,
            num_active_experts: None,
            rotary_dim: None,
            rope_neox: false,
        };
        match MetalF32Backend::new() {
            Ok(mut b) => match ComputeBackend::init(&mut b, &hp) {
                Ok(()) => Some(b),
                Err(_) => None,
            },
            Err(_) => None,
        }
    }

    /// Bit-identical GDN extract/restore round-trip.
    ///
    /// Pattern:
    /// 1. Fresh backend (no GDN buffers yet — gdn_num_layers == 0).
    /// 2. Construct a small but realistic GDN layout (2 layers).
    /// 3. Build a deterministic RecurrentState; `sync_kv_from_cpu` triggers
    ///    `ensure_gdn_storage_for_layout` which allocates GPU buffers and
    ///    writes our F32 bytes into them.
    /// 4. Build a zeroed RecurrentState of the same layout, then
    ///    `sync_kv_to_cpu` reads it back.
    /// 5. Assert every byte matches the original.
    #[test]
    fn gdn_extract_restore_round_trip() {
        let Some(backend) = make_metal_backend_for_tests() else {
            eprintln!("[skip] gdn_extract_restore_round_trip: no Metal device");
            return;
        };
        // Pre-populate a small KV cache so the K/V side of sync_kv_from_cpu
        // has something to read (we ignore those bytes — this test only
        // asserts on GDN state).
        let cfg = KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            precision: KvPrecision::F16,
        };
        let kv = KvCache::new(cfg).unwrap();

        let layout = GdnLayout {
            num_gdn_layers: 2,
            gdn_num_heads: 2,
            gdn_head_dim: 4,
            gdn_conv_kernel_size: 4,
            gdn_conv_qkv_dim: 16,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        };

        // Build deterministic input.
        let h_bytes = layout.h_state_bytes_per_layer();
        let c_bytes = layout.conv_state_bytes_per_layer();
        let mut rec = RecurrentState::zeroed(layout);
        for layer in 0..(layout.num_gdn_layers as usize) {
            for i in 0..h_bytes {
                rec.h_states[layer][i] = ((layer * 37 + i) % 251) as u8;
            }
            for i in 0..c_bytes {
                rec.conv_states[layer][i] = ((layer * 23 + i * 5) % 241) as u8;
            }
            rec.conv_positions[layer] = (layer as u32) % (layout.gdn_conv_kernel_size - 1);
        }

        // Upload to GPU (also allocates GDN storage for the fresh backend).
        ComputeBackend::sync_kv_from_cpu(&backend, &kv, Some(&rec)).unwrap();

        // Read back into a zeroed mirror.
        let mut rec_back = RecurrentState::zeroed(layout);
        let mut kv_back = KvCache::new(KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            precision: KvPrecision::F16,
        })
        .unwrap();
        ComputeBackend::sync_kv_to_cpu(&backend, &mut kv_back, Some(&mut rec_back)).unwrap();

        assert_eq!(rec_back.layout, layout);
        for layer in 0..(layout.num_gdn_layers as usize) {
            assert_eq!(
                rec_back.h_states[layer], rec.h_states[layer],
                "GDN h_state layer {layer} must round-trip bit-identically"
            );
            assert_eq!(
                rec_back.conv_states[layer], rec.conv_states[layer],
                "GDN conv_state layer {layer} must round-trip bit-identically"
            );
            assert_eq!(
                rec_back.conv_positions[layer], rec.conv_positions[layer],
                "GDN conv_position layer {layer} must round-trip"
            );
        }
    }

    /// Restore rejects a recurrent state with the wrong num_gdn_layers.
    #[test]
    fn gdn_restore_rejects_num_gdn_layers_mismatch() {
        let Some(backend) = make_metal_backend_for_tests() else {
            eprintln!("[skip] gdn_restore_rejects_num_gdn_layers_mismatch: no Metal device");
            return;
        };
        let layout = GdnLayout {
            num_gdn_layers: 2,
            gdn_num_heads: 2,
            gdn_head_dim: 4,
            gdn_conv_kernel_size: 4,
            gdn_conv_qkv_dim: 16,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        };
        let rec = RecurrentState::zeroed(layout);
        let kv = KvCache::new(KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            precision: KvPrecision::F16,
        })
        .unwrap();
        // First call allocates GDN storage with 2 layers.
        ComputeBackend::sync_kv_from_cpu(&backend, &kv, Some(&rec)).unwrap();

        // Now restore with 3 GDN layers should reject.
        let mismatched_layout = GdnLayout { num_gdn_layers: 3, ..layout };
        let bad = RecurrentState::zeroed(mismatched_layout);
        let err = ComputeBackend::sync_kv_from_cpu(&backend, &kv, Some(&bad)).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("GDN layers") || msg.contains("layout mismatch"),
            "expected GDN layer count mismatch error, got: {msg}"
        );
    }

    /// Restore rejects a recurrent state with the wrong head_dim
    /// (changes the per-layer byte count even at constant layer count).
    #[test]
    fn gdn_restore_rejects_head_dim_mismatch() {
        let Some(backend) = make_metal_backend_for_tests() else {
            eprintln!("[skip] gdn_restore_rejects_head_dim_mismatch: no Metal device");
            return;
        };
        let layout = GdnLayout {
            num_gdn_layers: 2,
            gdn_num_heads: 2,
            gdn_head_dim: 4,
            gdn_conv_kernel_size: 4,
            gdn_conv_qkv_dim: 16,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        };
        let rec = RecurrentState::zeroed(layout);
        let kv = KvCache::new(KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            precision: KvPrecision::F16,
        })
        .unwrap();
        ComputeBackend::sync_kv_from_cpu(&backend, &kv, Some(&rec)).unwrap();

        let mismatched_layout = GdnLayout { gdn_head_dim: 8, ..layout };
        let bad = RecurrentState::zeroed(mismatched_layout);
        let err = ComputeBackend::sync_kv_from_cpu(&backend, &kv, Some(&bad)).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("size") || msg.contains("mismatch"),
            "expected head_dim-driven byte-size mismatch error, got: {msg}"
        );
    }
}
