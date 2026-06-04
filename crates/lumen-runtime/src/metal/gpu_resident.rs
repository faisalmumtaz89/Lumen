//! GPU-resident weight preloading for Metal backend.
//!
//! Packs all layer weight data and global tensors into a single contiguous
//! `StorageModePrivate` Metal buffer, eliminating TLB misses, reducing virtual
//! address ranges, and enabling GPU memory controller optimizations.

use super::ffi::{MetalBuffer, MTLSize};
use super::repack_q8;
use super::repack_q4;
use super::types::{CachedLayerMeta, CachedMoeMeta};
use super::{MetalF32Backend, PAGE_SIZE};
use crate::error::RuntimeError;
use lumen_format::quantization::QuantScheme;

impl MetalF32Backend {
    /// Pre-load ALL layer weights into a single private (GPU-only) Metal buffer.
    ///
    /// Packs all layer weight data and global tensors into one contiguous private
    /// buffer using page-aligned offsets. Data is staged in a shared buffer then
    /// blit-copied to the private buffer. This:
    /// - Eliminates TLB misses from first-touch page faults on mmap'd memory
    /// - Reduces virtual address ranges from 22+ to 1 (lower TLB pressure)
    /// - Enables GPU memory controller optimizations via StorageModePrivate
    /// - Eliminates buffer object creation overhead per layer per token
    ///
    /// Memory cost: ~model_size bytes of GPU memory (e.g. 1.4 GB for TinyLlama Q8_0).
    pub fn preload_weights_gpu_resident(
        &self,
        weights: &dyn crate::weight::cache::WeightProvider,
    ) -> Result<(), RuntimeError> {
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized: call init() first".into())
        })?;

        let num_layers = s.num_layers;

        // Quiet by default — CLI controls verbosity.

        // ====================================================================
        // `LUMEN_METAL_MMAP_ONLY=1` eligibility probe.
        // ====================================================================
        //
        // When set AND layer 0's mmap pointer is page-aligned, Pass 1 emits
        // mmap-relative `base` offsets and records each layer's mmap pointer.
        // Pass 2/3 (staging + blit to private buffer) is replaced by a single
        // `newBufferWithBytesNoCopy:` wrapping the union span of all layer
        // mmap pages — zero CPU heap dup, zero staging dup, zero private dup.
        // Post-Pass-3 setup (MoE detection, GDN state, repack,
        // paired repack, etc.) runs unchanged.
        //
        // shipped this as `LUMEN_METAL_BF16_MMAP_ONLY` gated to BF16.
        // generalized to `LUMEN_METAL_MMAP_ONLY` covering BF16, Q8, Q4
        // for MoE 30B-A3B Q8/Q4 LBCs where the legacy Pass 1/2/3 dup pushes
        // peak RSS above the 5 GB free-RAM BAIL threshold even on 96 GB hosts
        // BF16 alias `LUMEN_METAL_BF16_MMAP_ONLY=1` is preserved for backward
        // compat — either env enables the same path.
        //
        // Why safe across quant schemes: the no-copy MTLBuffer wraps raw
        // mmap pages. BF16/Q8/Q4 weights are NOT mutated at residency time;
        // on-disk bytes are exactly what the MSL kernels read. mmap regions
        // are page-aligned on Unix. The MTLBuffer's lifetime is bounded by
        // MetalScratch, which is bounded by the engine holding the
        // WeightProvider (mmap owner). Globals (embedding/norm/output_proj)
        // remain on their existing buffers via the `gpu_global_offsets =
        // None` fallback already supported by decode/prefill paths.
        //
        // Q8 repack (FFN-down + gate+up SoA, env-default-ON via)
        // and Q4 repack (env-default-OFF) operate by reading raw mmap
        // bytes via `lv.subtensor_bytes(&st.<w>)` and writing into NEW Metal
        // buffers — they do NOT touch the unified buffer's bytes, so the
        // no-copy path is fully compatible with both repack passes.
        //
        // Fallback: if probe fails (non-aligned mmap ptr, or no layers),
        // legacy Pass 1/2/3 runs unchanged. When env unset, the entire
        // branch is skipped — binary-identical to the legacy path.
        let mmap_only_env = {
            let v_master = std::env::var("LUMEN_METAL_MMAP_ONLY")
                .ok()
                .as_deref()
                .map(|s| !s.is_empty() && s != "0")
                .unwrap_or(false);
            let v_bf16_alias = std::env::var("LUMEN_METAL_BF16_MMAP_ONLY")
                .ok()
                .as_deref()
                .map(|s| !s.is_empty() && s != "0")
                .unwrap_or(false);
            v_master || v_bf16_alias
        };

        // mmap-only path scratch:
        // - mmap_only: gate decision after probe of layer 0.
        // - mmap_min_ptr / mmap_max_end: union span of all layer mmap pages.
        // - layer_ptrs[i] = (raw mmap ptr usize, len) for layer i — only
        //   populated when mmap_only is true.
        let mut mmap_min_ptr: usize = usize::MAX;
        let mut mmap_max_end: usize = 0;
        let mut layer_ptrs: Vec<(usize, usize)> = Vec::new();

        let mmap_only = if mmap_only_env && num_layers > 0 {
            // Probe-pass: walk all layers, record ptr/len, check first layer
            // is page-aligned. get_layer_blocking() is O(1) for the mmap
            // provider (cached LayerView clone — pointer copy only).
            // removed BF16 quant gate; the no-copy path is correct
            // for any quant scheme because the unified buffer holds raw bytes.
            // get_layer_raw keeps the native blob layout (see the main upload
            // loop below for why get_layer_blocking would corrupt sync weights).
            let lv0 = weights.get_layer_raw(0).map_err(|e| {
                RuntimeError::Compute(format!(
                    " MMAP_ONLY: probe layer 0 failed: {}", e
                ))
            })?;
            let probe_ptr = lv0.as_bytes().as_ptr() as usize;
            let probe_aligned = probe_ptr != 0 && (probe_ptr % PAGE_SIZE == 0);
            if probe_aligned {
                layer_ptrs.reserve(num_layers);
                for layer in 0..num_layers {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " MMAP_ONLY: probe layer {} failed: {}", layer, e
                        ))
                    })?;
                    let bytes = lv.as_bytes();
                    let p = bytes.as_ptr() as usize;
                    let l = bytes.len();
                    layer_ptrs.push((p, l));
                    if p < mmap_min_ptr { mmap_min_ptr = p; }
                    let end = p.saturating_add(l);
                    if end > mmap_max_end { mmap_max_end = end; }
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        // === Pass 1: Collect layer blobs and compute page-aligned offsets ===
        let align = |size: usize| -> usize { (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1) };

        let mut layer_blobs: Vec<Vec<u8>> = Vec::with_capacity(num_layers);
        let mut layer_offsets: Vec<usize> = Vec::with_capacity(num_layers);
        let mut layer_metas: Vec<CachedLayerMeta> = Vec::with_capacity(num_layers);
        let mut cursor: usize = 0;
        let mut gdn_layer_counter: usize = 0;

        for layer in 0..num_layers {
            // get_layer_raw (NOT get_layer_blocking): GPU-resident upload needs
            // the weights in their native quant scheme (Q8_0/Q4_0/F16/BF16) with
            // the original blob layout. SyncWeightProvider::get_layer_blocking
            // dequantizes to F32 AND rebuilds the blob, leaving the GDN ssm_*
            // subtensor offsets pointing into the wrong blob -> corrupt weights
            // (pad-token garbage). MmapWeightProvider returns raw bytes for both
            // methods, so this is a no-op on the mmap path. Mirrors the CUDA
            // backend, which uses get_layer_raw for the same reason.
            let layer_view = weights.get_layer_raw(layer).map_err(|e| {
                RuntimeError::Compute(format!(
                    "Failed to get layer {} for GPU-resident loading: {}", layer, e
                ))
            })?;
            let blob = layer_view.as_bytes();
            // In mmap-only mode, `base` is the mmap-relative
            // byte offset of this layer's blob within the union span; subtensor
            // offsets `base + st.<sub>.offset` index into the no-copy MTLBuffer
            // that wraps `[mmap_min_ptr, mmap_max_end)`. Default-mode `base`
            // is `cursor` (page-packed offset into staging/private buffer).
            let base: u64 = if mmap_only {
                (layer_ptrs[layer].0 - mmap_min_ptr) as u64
            } else {
                cursor as u64
            };
            let st = &layer_view.subtensors;
            layer_metas.push(CachedLayerMeta {
                attn_norm_off: base + st.attn_norm.offset,
                wq_off: base + st.wq.offset,
                wo_off: base + st.wo.offset,
                // Prefer attn_post_norm when ffn_norm sentinel is absent (length=0).
                // Qwen3.5-35B-A3B uses post_attention_norm.weight as the FFN pre-norm;
                // ffn_norm is left as a zero-sentinel (offset=0, length=0) in the LBC.
                // Using offset=0 would read attn_qkv/attn_q Q4_0 data as F32 → NaN.
                ffn_norm_off: if st.ffn_norm.length == 0 {
                    st.attn_post_norm.map_or(0, |s| base + s.offset)
                } else {
                    base + st.ffn_norm.offset
                },
                w_gate_off: base + st.w_gate.offset,
                w_up_off: base + st.w_up.offset,
                w_down_off: base + st.w_down.offset,
                wq_quant: st.wq.quant,
                wo_quant: st.wo.quant,
                w_gate_quant: st.w_gate.quant,
                w_up_quant: st.w_up.quant,
                w_down_quant: st.w_down.quant,
                bq_off: st.bq.map(|b| base + b.offset),
                bk_off: st.bk.map(|b| base + b.offset),
                bv_off: st.bv.map(|b| base + b.offset),
                // MoE metadata: populated from SubtensorOffsets when this layer
                // has router_weight and experts (MoE model). None for dense layers.
                moe_meta: match (&st.router_weight, &st.experts) {
                    (Some(router), Some(experts)) if !experts.is_empty() => {
                        // Use the first expert's quant schemes as representative
                        // (all experts in a layer share the same quantization).
                        let first = &experts[0];
                        Some(CachedMoeMeta {
                            router_weight_off: base + router.offset,
                            expert_gate_offs: experts.iter().map(|e| base + e.gate.offset).collect(),
                            expert_up_offs: experts.iter().map(|e| base + e.up.offset).collect(),
                            expert_down_offs: experts.iter().map(|e| base + e.down.offset).collect(),
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        })
                    }
                    _ => None,
                },
                // Shared expert offsets (Qwen3.5-MoE).
                shared_expert_gate_off: st.shared_expert_gate.map(|s| base + s.offset),
                shared_expert_up_off: st.shared_expert_up.map(|s| base + s.offset),
                shared_expert_down_off: st.shared_expert_down.map(|s| base + s.offset),
                shared_expert_gate_quant: st.shared_expert_gate.map(|s| s.quant),
                shared_expert_down_quant: st.shared_expert_down.map(|s| s.quant),
                // Extended attention fields.
                attn_gate_off: st.attn_gate.map(|s| base + s.offset),
                attn_gate_quant: st.attn_gate.map(|s| s.quant),
                attn_post_norm_off: st.attn_post_norm.map(|s| base + s.offset),

                // Q+gate fusion: active for Qwen3.5 full-attention layers where
                // attn_q.weight contains interleaved Q+gate (8192 output rows).
                // Detected by presence of attn_q_norm (per-head Q RMSNorm), which
                // only exists on full-attention layers with separate Q/K/V projections.
                // When true, the decode path deinterleaves Q+gate, projects K/V
                // separately from wk/wv, and applies SiLU-gated output.
                has_qgate_fusion: st.attn_q_norm.is_some(),
                wk_off: if st.wk.length > 0 { Some(base + st.wk.offset) } else { None },
                wv_off: if st.wv.length > 0 { Some(base + st.wv.offset) } else { None },
                wk_quant: if st.wk.length > 0 { Some(st.wk.quant) } else { None },
                wv_quant: if st.wv.length > 0 { Some(st.wv.quant) } else { None },
                // Per-head Q/K RMSNorm weights.
                attn_q_norm_off: st.attn_q_norm.map(|s| base + s.offset),
                attn_k_norm_off: st.attn_k_norm.map(|s| base + s.offset),
                // Shared expert gate input weight.
                ffn_gate_inp_shexp_off: st.ffn_gate_inp_shexp.map(|s| base + s.offset),

                // Layer type discriminator.
                layer_type: st.layer_type,

                // GatedDeltaNet offsets.
                ssm_a_off: st.ssm_a.map(|s| base + s.offset),
                ssm_conv1d_off: st.ssm_conv1d.map(|s| base + s.offset),
                ssm_dt_off: st.ssm_dt.map(|s| base + s.offset),
                ssm_beta_off: st.ssm_beta.map(|s| base + s.offset),
                ssm_alpha_off: st.ssm_alpha.map(|s| base + s.offset),
                ssm_norm_off: st.ssm_norm.map(|s| base + s.offset),
                ssm_out_off: st.ssm_out.map(|s| base + s.offset),
                ssm_out_quant: st.ssm_out.map(|s| s.quant),
                gdn_layer_idx: if st.layer_type == Some(1) {
                    let idx = gdn_layer_counter;
                    gdn_layer_counter += 1;
                    Some(idx)
                } else {
                    None
                },
            });
            if mmap_only {
                // Defer offset/blob accumulation; the mmap-only
                // branch resolves layer_offsets from layer_ptrs and skips
                // Pass 2/3. Push placeholder zero so
                // layer_offsets.len() == num_layers.
                layer_offsets.push(0);
            } else {
                layer_offsets.push(cursor);
                layer_blobs.push(blob.to_vec());
                cursor = align(cursor + blob.len());
            }
        }

        // ====================================================================
        // MMAP_ONLY: replace Pass 2/3 with a single no-copy MTLBuffer.
        // ====================================================================
        //
        // In mmap-only mode:
        //   - All layer blobs live in `[mmap_min_ptr, mmap_max_end)`.
        //   - One `newBufferWithBytesNoCopy:` wraps that union span; the
        //     GPU reads weights directly from mmap'd OS pages (unified memory
        //     on Apple Silicon — no DMA copy, no private allocation).
        //   - `layer_offsets[i]` = mmap-relative byte offset of layer i.
        //   - Globals (embedding/norm/output_proj) live in their existing
        //     per-tensor buffers (initialized by backend_impl.rs); the
        //     `gpu_global_offsets = None` branch in decode/prefill paths
        //     binds those buffers via fallback.
        //   - VRAM ledger: peak transient = 0 above the steady-state mmap
        //     resident set. Steady state = LBC file size (mmap'd pages,
        //     OS-managed) + per-tensor globals (~1.5 GB for Qwen3.5-9B BF16)
        //     + scratch/KV/RoPE/MoE-meta buffers.
        let layer_bytes_total: usize;
        let total_size: usize;
        let include_globals: bool;
        let global_bytes: usize;
        let (embed_offset, norm_offset, proj_offset): (usize, usize, usize);

        if mmap_only {
            // Sanity: union span > 0 and the first layer's pointer is page-aligned
            // (probe pass guaranteed this; double-check defensively).
            if mmap_min_ptr == usize::MAX || mmap_max_end <= mmap_min_ptr {
                return Err(RuntimeError::Compute(
                    " MMAP_ONLY: invalid mmap span (no layers recorded)".into(),
                ));
            }
            if mmap_min_ptr % PAGE_SIZE != 0 {
                return Err(RuntimeError::Compute(format!(
                    " MMAP_ONLY: mmap_min_ptr {:#x} not page-aligned",
                    mmap_min_ptr
                )));
            }
            // Fill layer_offsets with mmap-relative byte offsets.
            // `base` in layer_metas was already computed in Pass 1 using these
            // same offsets, so the two are consistent (no double accounting).
            for (i, (ptr, _len)) in layer_ptrs.iter().enumerate() {
                layer_offsets[i] = *ptr - mmap_min_ptr;
            }

            let span_raw = mmap_max_end - mmap_min_ptr;
            // Round span up to page boundary as required by
            // newBufferWithBytesNoCopy on Apple Silicon.
            let span = align(span_raw);

            // Sanity: don't wrap absurd sizes (defensive — Qwen3.5-9B BF16
            // mmap span is ~16.3 GB; MoE-30B BF16 ~60 GB if we ever extend).
            const MAX_MMAP_SPAN_BYTES: usize = 96 * 1024 * 1024 * 1024; // 96 GB
            if span > MAX_MMAP_SPAN_BYTES {
                return Err(RuntimeError::Compute(format!(
                    " MMAP_ONLY: union span {} bytes exceeds ceiling {}",
                    span, MAX_MMAP_SPAN_BYTES
                )));
            }

            layer_bytes_total = layer_ptrs.iter().map(|(_, l)| *l).sum::<usize>();
            total_size = span;
            include_globals = false;
            global_bytes = 0;
            embed_offset = 0;
            norm_offset = 0;
            proj_offset = 0;

            // Wrap mmap pages in a single MTLBuffer (zero-copy on unified memory).
            //
            // SAFETY: The mmap region is owned by the WeightProvider that the
            // engine borrows for the duration of `generate()`. The MetalScratch
            // (which holds the MTLBuffer in `gpu_unified_weight_buf`) drops
            // before the engine drops the provider, so the MTLBuffer's
            // dereferences always see live mmap pages. The deallocator block is
            // nil (we do not own the memory — the kernel mmap does).
            let unified_buf = unsafe {
                self.device.new_buffer_no_copy(mmap_min_ptr as *mut std::ffi::c_void, span)
            }.ok_or_else(|| {
                RuntimeError::Compute(format!(
                    " MMAP_ONLY: newBufferWithBytesNoCopy failed (ptr={:#x}, len={})",
                    mmap_min_ptr, span
                ))
            })?;

            // Drop layer_blobs (empty in mmap-only mode but allocator may
            // have reserved capacity from with_capacity).
            drop(layer_blobs);

            s.gpu_unified_weight_buf = Some(unified_buf);
            s.gpu_layer_offsets = layer_offsets;
            s.gpu_global_offsets = None; // Use legacy per-tensor global buffers.
            s.cached_layer_meta = layer_metas;

            // instrumentation: surface span size via the resident summary path.
            let layer_mb = layer_bytes_total as f64 / (1024.0 * 1024.0);
            let total_mb = total_size as f64 / (1024.0 * 1024.0);
            let _ = (num_layers, layer_mb, total_mb);
        } else {
            // Legacy path: Append global tensors at page-aligned offsets.
            // For large-vocab models (>64K vocab), the embedding + output_proj tables
            // can exceed 2 GB, causing a 3 GB private buffer that degrades GPU cache
            // performance. Only pack globals into the unified buffer when they're small.
            let embed_buf_ref = self.embedding_buf.as_ref().ok_or_else(|| {
                RuntimeError::Compute("Embedding buffer not initialized for unified preload".into())
            })?;
            let embed_len = embed_buf_ref.length() as usize;

            let norm_buf_ref = self.final_norm_buf.as_ref().ok_or_else(|| {
                RuntimeError::Compute("Final norm buffer not initialized for unified preload".into())
            })?;
            let norm_len = norm_buf_ref.length() as usize;

            let proj_buf_ref = self.output_proj_buf.as_ref().ok_or_else(|| {
                RuntimeError::Compute("Output proj buffer not initialized for unified preload".into())
            })?;
            let proj_len = proj_buf_ref.length() as usize;

            // Weight tying: output_proj shares embedding storage (no separate allocation)
            let effective_proj_len = if self.weight_tying { 0 } else { proj_len };
            global_bytes = embed_len + norm_len + effective_proj_len;
            // Include globals in the unified private buffer.
            include_globals = true;

            let (eo, no_, po) = if include_globals {
                let eo = cursor;
                cursor = align(cursor + embed_len);
                let no_ = cursor;
                cursor = align(cursor + norm_len);
                if self.weight_tying {
                    // output_proj reuses embedding offset
                    (eo, no_, eo)
                } else {
                    let po = cursor;
                    cursor = align(cursor + proj_len);
                    (eo, no_, po)
                }
            } else {
                (0, 0, 0)
            };
            embed_offset = eo;
            norm_offset = no_;
            proj_offset = po;

            total_size = cursor;

            // === Pass 2: Allocate shared staging buffer and copy all data via CPU ===
            let staging_buf = self.device.new_buffer(total_size).ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "Failed to allocate staging buffer ({} bytes, {:.1} MB)",
                    total_size, total_size as f64 / (1024.0 * 1024.0)
                ))
            })?;

            let dst_base = staging_buf.contents() as *mut u8;
            let mut layer_bytes_total_local: usize = 0;

            for (layer, blob) in layer_blobs.iter().enumerate() {
                let off = layer_offsets[layer];
                unsafe {
                    std::ptr::copy_nonoverlapping(blob.as_ptr(), dst_base.add(off), blob.len());
                }
                layer_bytes_total_local += blob.len();
            }

            if include_globals {
                // Copy global tensors from their existing Metal buffers
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        embed_buf_ref.contents() as *const u8, dst_base.add(embed_offset), embed_len,
                    );
                    std::ptr::copy_nonoverlapping(
                        norm_buf_ref.contents() as *const u8, dst_base.add(norm_offset), norm_len,
                    );
                    if !self.weight_tying {
                        std::ptr::copy_nonoverlapping(
                            proj_buf_ref.contents() as *const u8, dst_base.add(proj_offset), proj_len,
                        );
                    }
                }
            }

            layer_bytes_total = layer_bytes_total_local;

            // Free temporary layer blobs before allocating private buffer
            drop(layer_blobs);

            // === Pass 3: Blit copy from shared staging to private GPU-only buffer ===
            let private_buf = self.device.new_buffer_private(total_size).ok_or_else(|| {
                RuntimeError::Compute(format!(
                    "Failed to allocate private GPU buffer ({} bytes, {:.1} MB)",
                    total_size, total_size as f64 / (1024.0 * 1024.0)
                ))
            })?;

            let blit_cmd = self.queue.new_command_buffer().ok_or_else(|| {
                RuntimeError::Compute("Failed to create command buffer for weight blit".into())
            })?;
            let blit_enc = blit_cmd.new_blit_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create blit encoder for weight copy".into())
            })?;
            blit_enc.copy_from_buffer(&staging_buf, 0, &private_buf, 0, total_size as u64);
            blit_enc.end_encoding();
            blit_cmd.commit_and_wait();

            // Staging buffer dropped here, freeing shared memory
            drop(staging_buf);

            let layer_mb = layer_bytes_total as f64 / (1024.0 * 1024.0);
            let global_mb = if include_globals { global_bytes as f64 / (1024.0 * 1024.0) } else { 0.0 };
            let total_mb = total_size as f64 / (1024.0 * 1024.0);
            // GPU-resident buffer info available via MetalF32Backend::gpu_resident_summary().
            let _ = (num_layers, layer_mb, global_mb, total_mb, include_globals);

            s.gpu_unified_weight_buf = Some(private_buf);
            s.gpu_layer_offsets = layer_offsets;
            if include_globals {
                s.gpu_global_offsets = Some((embed_offset, norm_offset, proj_offset));
            } else {
                s.gpu_global_offsets = None;  // Forces fallback to separate shared buffers
            }
            s.cached_layer_meta = layer_metas;
        }
        // ====================================================================
        // End of/ split: both paths have populated
        // `s.gpu_unified_weight_buf`, `s.gpu_layer_offsets`,
        // `s.gpu_global_offsets`, `s.cached_layer_meta`. The remaining
        // setup (Qwen3.5-MoE detection, GDN state, MoE offsets,/
        // repack, warmup, etc.) runs unchanged for both paths.
        // ====================================================================
        // Suppress unused-variable warnings when only the legacy path uses
        // the global offsets (mmap-only path zeros them as `_unused`).
        let _ = (global_bytes, total_size, layer_bytes_total, include_globals,
                 embed_offset, norm_offset, proj_offset);

        // ====================================================================
        // Qwen3.5-MoE detection
        // ====================================================================
        // Detect hybrid architecture from format-level metadata:
        //   1. Has shared expert weights (shared_expert_gate on at least one layer)
        //   2. Has layer_type discriminators (some layers have layer_type = Some(0) or Some(1))
        //   3. Has MoE routing (at least one layer with moe_meta)
        {
            let has_layer_types = s.cached_layer_meta.iter().any(|m| m.layer_type.is_some());
            let has_moe = s.cached_layer_meta.iter().any(|m| m.moe_meta.is_some());

            // Detection: Qwen3.5 family has hybrid layer types (GDN + full attention).
            // MoE variant (Qwen3.5-35B-A3B) also has MoE routing.
            // Dense variant (Qwen3.5-9B) has layer_types but no MoE.
            if has_layer_types {
                // All Qwen3.5 variants (MoE and dense) use NeoX-style RoPE.
                // rope_neox is already set from hyperparams in init(); this is defensive.
                s.rope_neox = true;
                if has_moe {
                    // Shared expert intermediate dimension
                    s.shared_expert_inter_dim = s.inter_dim;
                    let se_inter = s.shared_expert_inter_dim;
                    let hidden = s.hidden_dim;
                    s.shared_expert_gate_buf = Some(self.device.new_buffer(se_inter * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate shared_expert_gate_buf".into())
                    })?);
                    s.shared_expert_down_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate shared_expert_down_buf".into())
                    })?);
                }

                // Partial RoPE: Qwen3.5 uses partial_rotary_factor=0.25,
                // meaning only the first head_dim/4 dimensions of each head are rotated.
                let head_dim = s.head_dim;
                if head_dim >= 128 {
                    s.rotary_dim = head_dim / 4;  // 128/4 = 32 for Qwen3.5-9B, 256/4 = 64 for -35B
                }

                // Allocate attention gate scratch buffer (for full attention layers with attn_gate)
                let has_attn_gate = s.cached_layer_meta.iter().any(|m| m.attn_gate_off.is_some());
                if has_attn_gate {
                    let hidden = s.hidden_dim;
                    s.attn_gate_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate attn_gate_buf".into())
                    })?);
                }

                // Recompute RoPE cos/sin tables for partial rotation.
                // theta is sourced from hyperparams (stored in MetalScratch during init).
                let rotary_half_dim = s.rotary_dim / 2;
                let theta: f64 = s.rope_theta;
                let max_seq = s.max_seq_len;
                let mut cos_table = vec![0.0f32; max_seq * rotary_half_dim];
                let mut sin_table = vec![0.0f32; max_seq * rotary_half_dim];
                for pos in 0..max_seq {
                    for i in 0..rotary_half_dim {
                        let freq = 1.0 / theta.powf((2 * i) as f64 / s.rotary_dim as f64);
                        let angle = pos as f64 * freq;
                        cos_table[pos * rotary_half_dim + i] = angle.cos() as f32;
                        sin_table[pos * rotary_half_dim + i] = angle.sin() as f32;
                    }
                }
                // Upload new tables to existing RoPE buffers (resize if needed).
                let new_rope_bytes = cos_table.len() * 4;
                s.rope_cos_buf = self.device.new_buffer(new_rope_bytes).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate partial RoPE cos buffer".into())
                })?;
                s.rope_sin_buf = self.device.new_buffer(new_rope_bytes).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate partial RoPE sin buffer".into())
                })?;
                s.rope_cos_buf.write_f32(&cos_table);
                s.rope_sin_buf.write_f32(&sin_table);

                // Count layer types for diagnostics
                let n_linear = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(1)).count();
                let n_full = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(0)).count();
                let n_moe = s.cached_layer_meta.iter().filter(|m| m.moe_meta.is_some()).count();
                let n_shared = s.cached_layer_meta.iter().filter(|m| m.shared_expert_gate_off.is_some()).count();
                let se_inter_display = s.shared_expert_inter_dim;
                let _ = (n_linear, n_full, n_moe, n_shared, se_inter_display);

            }
        }

        // ====================================================================
        // GatedDeltaNet state allocation
        // ====================================================================
        // Allocate persistent h_state and conv_state buffers for all GDN layers.
        // This runs for ANY model with layer_type=1 layers (both MoE and dense).
        {
            let n_linear = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(1)).count();
            if n_linear > 0 {
                const GDN_NUM_HEADS: usize = 32;    // ssm.time_step_rank
                const GDN_HEAD_DIM: usize = 128;    // ssm.state_size
                const GDN_QKV_DIM: usize = 8192;    // Q(2048) + K(2048) + V(4096)
                let conv_kernel_size: usize = 4;    // Qwen3.5 uses kernel_size=4
                let hidden = s.hidden_dim;

                // h_state: [GDN_NUM_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM] per GDN layer
                let h_state_size = GDN_NUM_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
                // conv_state: [(kernel_size - 1) * GDN_QKV_DIM] per GDN layer
                let conv_state_size = (conv_kernel_size - 1) * GDN_QKV_DIM;

                let mut h_states = Vec::with_capacity(n_linear);
                let mut conv_states = Vec::with_capacity(n_linear);
                for _ in 0..n_linear {
                    let h_buf = self.device.new_buffer(h_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN h_state buffer".into())
                    })?;
                    // Zero-initialize h_state (new sequence starts with zero state)
                    h_buf.write_f32(&vec![0.0f32; h_state_size]);
                    h_states.push(h_buf);

                    let c_buf = self.device.new_buffer(conv_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN conv_state buffer".into())
                    })?;
                    c_buf.write_f32(&vec![0.0f32; conv_state_size]);
                    conv_states.push(c_buf);
                }

                s.gdn_h_states = h_states;
                s.gdn_conv_states = conv_states;
                s.gdn_conv_positions = vec![0u32; n_linear];
                s.gdn_conv_kernel_size = conv_kernel_size;
                s.gdn_num_layers = n_linear;

                // Allocate GDN scratch buffers using GDN-specific dimensions
                let gdn_q_dim = GDN_NUM_HEADS * GDN_HEAD_DIM; // 4096
                s.gdn_alpha_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN alpha buffer".into())
                })?);
                s.gdn_beta_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN beta buffer".into())
                })?);
                // Output of state query: [GDN_NUM_HEADS * GDN_HEAD_DIM] = 4096
                s.gdn_output_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN output buffer".into())
                })?);
                // SSM output projection result: [hidden_dim]
                s.gdn_ssm_proj_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN ssm_proj buffer".into())
                })?);
                // Attention gate sigmoid output: [q_dim=4096] (gate applied BEFORE ssm_out_proj)
                s.gdn_gate_sigmoid_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN gate_sigmoid buffer".into())
                })?);
                // L2-norm scaled output: [GDN_NUM_HEADS * GDN_HEAD_DIM] = 4096
                s.gdn_normed_out_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN normed_out buffer".into())
                })?);
                // Q8_0 matvec outputs for alpha/beta gate projections [GDN_NUM_HEADS] f32
                s.gdn_alpha_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN alpha_raw buffer".into())
                })?);
                s.gdn_beta_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN beta_raw buffer".into())
                })?);
                // Conv1d output for all QKV channels [GDN_QKV_DIM=8192] f32
                s.gdn_qkv_conv_buf = Some(self.device.new_buffer(GDN_QKV_DIM * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN qkv_conv buffer".into())
                })?);

                let h_state_mb = (n_linear * h_state_size * 4) as f64 / (1024.0 * 1024.0);
                let conv_mb = (n_linear * conv_state_size * 4) as f64 / (1024.0 * 1024.0);
                let _ = (h_state_mb, conv_mb, conv_kernel_size);
            }
        }

        // Clear legacy per-layer buffers (unified replaces them)
        s.gpu_resident_layers = None;

        // ====================================================================
        // Build MoE expert offset tables for batched GPU-side dispatch.
        // Upload per-layer offset arrays to GPU buffers so the batched kernels
        // can look up expert weight positions without CPU readback.
        // ====================================================================
        {
            let n_experts = s.moe_num_experts;
            if n_experts > 0 {
                let mut gate_up_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                let mut down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                for meta in &s.cached_layer_meta {
                    if let Some(ref moe_meta) = meta.moe_meta {
                        // Build gate+up offset table: [n_experts * 2] u64
                        let mut gu_offsets = vec![0u64; n_experts * 2];
                        for e in 0..n_experts.min(moe_meta.expert_gate_offs.len()) {
                            gu_offsets[e * 2]     = moe_meta.expert_gate_offs[e];
                            gu_offsets[e * 2 + 1] = moe_meta.expert_up_offs[e];
                        }
                        let gu_bytes: Vec<u8> = gu_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let gu_buf = self.device.new_buffer_with_bytes(&gu_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE gate_up offset table".into())
                        })?;
                        gate_up_vecs.push(Some(gu_buf));

                        // Build down offset table: [n_experts] u64
                        let mut d_offsets = vec![0u64; n_experts];
                        for e in 0..n_experts.min(moe_meta.expert_down_offs.len()) {
                            d_offsets[e] = moe_meta.expert_down_offs[e];
                        }
                        let d_bytes: Vec<u8> = d_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let d_buf = self.device.new_buffer_with_bytes(&d_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE down offset table".into())
                        })?;
                        down_vecs.push(Some(d_buf));
                    } else {
                        gate_up_vecs.push(None);
                        down_vecs.push(None);
                    }
                }
                s.moe_gate_up_offsets = gate_up_vecs;
                s.moe_down_offsets = down_vecs;

                // Build shared expert down offset tables for fused kernels.
                // Each MoE layer with a shared expert gets a single u64 GPU buffer
                // containing the byte offset of the shared expert down weight matrix.
                let mut se_down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                for meta in &s.cached_layer_meta {
                    if let Some(se_down_off) = meta.shared_expert_down_off {
                        let off_bytes: Vec<u8> = se_down_off.to_le_bytes().to_vec();
                        let buf = self.device.new_buffer_with_bytes(&off_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE shared expert down offset".into())
                        })?;
                        se_down_vecs.push(Some(buf));
                    } else {
                        se_down_vecs.push(None);
                    }
                }
                s.moe_shared_down_offsets = se_down_vecs;
            }
        }

        // ====================================================================
        // Runtime Q8_0 hot-weight repack (env-gated, default OFF).
        // ====================================================================
        //
        // When `LUMEN_METAL_Q8_REPACKED=1`, allocate extra Metal buffers
        // containing the FFN-down weights and the gate+up pair in a stripe
        // SoA layout (see `metal/repack_q8.rs`). The packed kernels in
        // `shaders/gemm_q8_0.msl` (`*_packed`) consume these. The original
        // buffers + AoS kernels are preserved unchanged as a fallback path.
        //
        // VRAM cost (per layer, Qwen3.5-9B Q8):
        //   FFN-down:   ~50 MB (same as raw Q8, byte count preserved)
        //   Gate+Up:    ~100 MB (2 × 50 MB, paired interleaved)
        //
        // Across 32 layers: ~1.6 GB FFN-down + ~3.2 GB gate+up =  ~4.8 GB
        // additional VRAM. M3 Ultra 96 GB headroom comfortably accomodates
        // this; for smaller machines, the env gate keeps it off by default.
        {
            use super::graph_reorder as gr;
            let want_repack = gr::q8_repacked_enabled();
            let want_ffn_down = gr::q8_repacked_ffn_down_enabled();
            let want_gate_up = gr::q8_repacked_gate_up_enabled();
            if want_repack && (want_ffn_down || want_gate_up) {
                let hidden_dim_u = s.hidden_dim;
                let inter_dim_u = s.inter_dim;

                let mut ffn_down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                let mut gate_up_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);

                let mut down_ok_count: usize = 0;
                let mut gate_up_ok_count: usize = 0;

                for layer in 0..num_layers {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " repack: failed to get layer {}: {}", layer, e
                        ))
                    })?;
                    let st = &lv.subtensors;

                    // FFN-down: target shape [hidden_dim, inter_dim] Q8_0
                    //   N = hidden_dim (output rows), K = inter_dim
                    //   Qwen3.5-9B: N=4096, K=12288 — both multiples of 32.
                    let ffn_down_buf: Option<MetalBuffer> = if want_ffn_down
                        && st.w_down.quant == QuantScheme::Q8_0
                        && hidden_dim_u % 32 == 0
                        && inter_dim_u % 32 == 0
                        && st.w_down.length > 0
                    {
                        let src = lv.subtensor_bytes(&st.w_down).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " repack: failed to read w_down at layer {}: {}", layer, e
                            ))
                        })?;
                        match repack_q8::build_repacked_buffer_single(
                            &self.device, src, hidden_dim_u, inter_dim_u,
                        ) {
                            Ok(buf) => { down_ok_count += 1; Some(buf) }
                            Err(_) => None,
                        }
                    } else {
                        None
                    };
                    ffn_down_vecs.push(ffn_down_buf);

                    // Gate+Up pair: target shape [inter_dim, hidden_dim] Q8_0 each.
                    //   N = inter_dim, K = hidden_dim. Both gate AND up must be Q8.
                    //   Qwen3.5-9B: N=12288, K=4096 — both multiples of 32.
                    let gate_up_buf: Option<MetalBuffer> = if want_gate_up
                        && st.w_gate.quant == QuantScheme::Q8_0
                        && st.w_up.quant == QuantScheme::Q8_0
                        && inter_dim_u % 32 == 0
                        && hidden_dim_u % 32 == 0
                        && st.w_gate.length > 0
                        && st.w_up.length > 0
                        && st.w_gate.length == st.w_up.length
                    {
                        let src_g = lv.subtensor_bytes(&st.w_gate).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " repack: failed to read w_gate at layer {}: {}", layer, e
                            ))
                        })?;
                        let src_u = lv.subtensor_bytes(&st.w_up).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " repack: failed to read w_up at layer {}: {}", layer, e
                            ))
                        })?;
                        match repack_q8::build_repacked_buffer_pair(
                            &self.device, src_g, src_u, inter_dim_u, hidden_dim_u,
                        ) {
                            Ok(buf) => { gate_up_ok_count += 1; Some(buf) }
                            Err(_) => None,
                        }
                    } else {
                        None
                    };
                    gate_up_vecs.push(gate_up_buf);
                }

                s.repacked_ffn_down = ffn_down_vecs;
                s.repacked_ffn_gate_up = gate_up_vecs;

                // Diagnostic counters (silenced by default; use env LUMEN_METAL_LOG to enable).
                let _ = (down_ok_count, gate_up_ok_count);
            }
        }

        // Q4_0 hot-weight repack pass.
        //
        // Same pattern as the Q8 block above. Allocates `MTLBuffer`s
        // holding hot Q4_0 FFN tensors in a Metal-friendly stripe SoA layout
        // (see `metal/repack_q4.rs`). The packed kernels in
        // `shaders/gemm_q4.msl` (`*_packed`) consume these. The original
        // buffers + AoS kernels are preserved unchanged as a fallback path.
        //
        // VRAM cost (per layer, Qwen3.5-9B Q4):
        //   FFN-down:   ~25 MB (same as raw Q4, byte count preserved)
        //   Gate+Up:    ~50 MB (2 × 25 MB, paired interleaved)
        //
        // Across 32 layers: ~0.8 GB FFN-down + ~1.6 GB gate+up = ~2.4 GB
        // additional VRAM. M3 Ultra 96 GB headroom comfortably accomodates
        // this; for smaller machines, the env gate keeps it off by default.
        {
            use super::graph_reorder as gr;
            let want_repack = gr::q4_repacked_enabled();
            let want_ffn_down = gr::q4_repacked_ffn_down_enabled();
            let want_gate_up = gr::q4_repacked_gate_up_enabled();
            if want_repack && (want_ffn_down || want_gate_up) {
                let hidden_dim_u = s.hidden_dim;
                let inter_dim_u = s.inter_dim;

                let mut ffn_down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                let mut gate_up_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);

                let mut down_ok_count: usize = 0;
                let mut gate_up_ok_count: usize = 0;

                for layer in 0..num_layers {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " Q4 repack: failed to get layer {}: {}", layer, e
                        ))
                    })?;
                    let st = &lv.subtensors;

                    // FFN-down: target shape [hidden_dim, inter_dim] Q4_0
                    //   N = hidden_dim (output rows), K = inter_dim
                    //   Qwen3.5-9B: N=4096, K=12288 — both multiples of 32.
                    let ffn_down_buf: Option<MetalBuffer> = if want_ffn_down
                        && st.w_down.quant == QuantScheme::Q4_0
                        && hidden_dim_u % 32 == 0
                        && inter_dim_u % 32 == 0
                        && st.w_down.length > 0
                    {
                        let src = lv.subtensor_bytes(&st.w_down).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " Q4 repack: failed to read w_down at layer {}: {}", layer, e
                            ))
                        })?;
                        match repack_q4::build_repacked_buffer_single(
                            &self.device, src, hidden_dim_u, inter_dim_u,
                        ) {
                            Ok(buf) => { down_ok_count += 1; Some(buf) }
                            Err(_) => None,
                        }
                    } else {
                        None
                    };
                    ffn_down_vecs.push(ffn_down_buf);

                    // Gate+Up pair: target shape [inter_dim, hidden_dim] Q4_0 each.
                    //   N = inter_dim, K = hidden_dim. Both gate AND up must be Q4.
                    //   Qwen3.5-9B: N=12288, K=4096 — both multiples of 32.
                    let gate_up_buf: Option<MetalBuffer> = if want_gate_up
                        && st.w_gate.quant == QuantScheme::Q4_0
                        && st.w_up.quant == QuantScheme::Q4_0
                        && inter_dim_u % 32 == 0
                        && hidden_dim_u % 32 == 0
                        && st.w_gate.length > 0
                        && st.w_up.length > 0
                        && st.w_gate.length == st.w_up.length
                    {
                        let src_g = lv.subtensor_bytes(&st.w_gate).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " Q4 repack: failed to read w_gate at layer {}: {}", layer, e
                            ))
                        })?;
                        let src_u = lv.subtensor_bytes(&st.w_up).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " Q4 repack: failed to read w_up at layer {}: {}", layer, e
                            ))
                        })?;
                        match repack_q4::build_repacked_buffer_pair(
                            &self.device, src_g, src_u, inter_dim_u, hidden_dim_u,
                        ) {
                            Ok(buf) => { gate_up_ok_count += 1; Some(buf) }
                            Err(_) => None,
                        }
                    } else {
                        None
                    };
                    gate_up_vecs.push(gate_up_buf);
                }

                s.repacked_ffn_down_q4 = ffn_down_vecs;
                s.repacked_ffn_gate_up_q4 = gate_up_vecs;

                // Diagnostic counters (silenced by default; use env LUMEN_METAL_LOG to enable).
                let _ = (down_ok_count, gate_up_ok_count);
            }
        }

        // ====================================================================
        // BF16 GDN qkv-proj + attn-gate-proj concat-then-stripe repack.
        // ====================================================================
        //
        // When `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED=1`, allocate one Metal
        // buffer per GDN layer (24 layers on Qwen3.5-9B) holding the
        // qkv and attn_gate BF16 weights concatenated along the output (N)
        // axis and byte-permuted into the stripe layout (see
        // `metal/repack_bf16.rs`). The packed kernel in
        // `shaders/gemm_residual_bf16.msl` (`tiled_matmul_bf16_k64_qkv_gate_paired`)
        // consumes these. The original sequential two-dispatch path is
        // preserved as a fallback when this is OFF or when a layer doesn't
        // qualify (non-BF16, wrong shape, alignment mismatch, etc).
        //
        // VRAM cost (per GDN layer, Qwen3.5-9B BF16):
        //   (qkv_n + gate_n) * hidden_dim * 2 = (8192 + 4096) * 4096 * 2 = 96 MB
        // 24 GDN layers x 96 MB = 2.30 GB extra resident. Well under the
        // 4.8 GB Apple AGX TLB threshold established (Q8 repack at
        // 4.8 GB = +6.89%) vs (BF16 repack at 6.1 GB = -54.74%).
        {
            use super::graph_reorder as gr;
            if gr::bf16_gdn_qkv_gate_paired_enabled() {
                let hidden_dim_u = s.hidden_dim;
                // The packed buffer Vec is indexed by `gdn_idx` (sequential GDN
                // layer counter 0..n_gdn_layers-1), matching the convention used
                // for `gdn_h_states` and `gdn_conv_states`. Non-GDN (full-attn)
                // layers do not enter the Vec at all.
                let n_gdn_layers = s.cached_layer_meta.iter()
                    .filter(|m| m.layer_type == Some(1))
                    .count();
                let mut qkv_gate_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
                // parallel record of per-layer `(qkv_n, gate_n)` so that
                // the load-time warmup dispatch (below) can issue a correctly
                // shaped touch-dispatch against each populated buffer.
                let mut qkv_gate_shapes: Vec<Option<(u32, u32)>> = Vec::with_capacity(n_gdn_layers);

                for layer in 0..num_layers {
                    // Skip layers that aren't GDN. We rely on `layer_type == Some(1)`
                    // as the canonical GDN marker (matches `gdn_h_states` ordering).
                    let meta = &s.cached_layer_meta[layer];
                    if meta.layer_type != Some(1) {
                        continue;
                    }
                    let attn_gate_off = match meta.attn_gate_off {
                        Some(_) => {},
                        None => {
                            qkv_gate_vecs.push(None);
                            qkv_gate_shapes.push(None);
                            continue;
                        }
                    };
                    let _ = attn_gate_off;

                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " BF16 GDN paired repack: failed to get layer {}: {}", layer, e
                        ))
                    })?;
                    let st = &lv.subtensors;

                    let attn_gate_st = match &st.attn_gate {
                        Some(a) => a,
                        None => {
                            qkv_gate_vecs.push(None);
                            qkv_gate_shapes.push(None);
                            continue;
                        }
                    };

                    // BOTH tensors must be BF16. Otherwise skip — Q8/Q4 GDN layers
                    // use the existing Q8/Q4 dispatch paths.
                    if st.wq.quant != QuantScheme::Bf16 || attn_gate_st.quant != QuantScheme::Bf16 {
                        qkv_gate_vecs.push(None);
                        qkv_gate_shapes.push(None);
                        continue;
                    }

                    // Derive the projection N dimensions from the BF16 tensor lengths
                    // (each tensor is `N * K * 2` bytes).
                    let row_bytes = hidden_dim_u
                        .checked_mul(2)
                        .ok_or_else(|| RuntimeError::Compute(
                            " BF16 repack: hidden_dim * 2 overflow".into()
                        ))?;
                    if row_bytes == 0 {
                        qkv_gate_vecs.push(None);
                        qkv_gate_shapes.push(None);
                        continue;
                    }
                    let qkv_n = (st.wq.length as usize) / row_bytes;
                    let gate_n = (attn_gate_st.length as usize) / row_bytes;

                    // Alignment guards: TILE_N=32 on N, TILE_K_64=64 on K.
                    // For Qwen3.5-9B GDN both are exact multiples.
                    if qkv_n == 0
                        || gate_n == 0
                        || qkv_n % super::repack_bf16::TILE_N != 0
                        || gate_n % super::repack_bf16::TILE_N != 0
                        || hidden_dim_u % super::repack_bf16::TILE_K_64 != 0
                    {
                        qkv_gate_vecs.push(None);
                        qkv_gate_shapes.push(None);
                        continue;
                    }

                    // Sanity-check the byte counts match the inferred shape.
                    if st.wq.length as usize != qkv_n * row_bytes
                        || attn_gate_st.length as usize != gate_n * row_bytes
                    {
                        qkv_gate_vecs.push(None);
                        qkv_gate_shapes.push(None);
                        continue;
                    }

                    let src_qkv = lv.subtensor_bytes(&st.wq).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " BF16 repack: failed to read wq at layer {}: {}", layer, e
                        ))
                    })?;
                    let src_gate = lv.subtensor_bytes(attn_gate_st).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " BF16 repack: failed to read attn_gate at layer {}: {}", layer, e
                        ))
                    })?;

                    let buf = super::repack_bf16::build_repacked_buffer_qkv_gate(
                        &self.device, src_qkv, src_gate, qkv_n, gate_n, hidden_dim_u,
                    );
                    match buf {
                        Ok(b) => {
                            qkv_gate_vecs.push(Some(b));
                            qkv_gate_shapes.push(Some((qkv_n as u32, gate_n as u32)));
                        }
                        Err(_) => {
                            qkv_gate_vecs.push(None);
                            qkv_gate_shapes.push(None);
                        }
                    }
                }

                let _ok_count = qkv_gate_vecs.iter().filter(|o| o.is_some()).count();
                s.repacked_gdn_qkv_gate_bf16 = qkv_gate_vecs;

                // Diagnostic counter silenced. Re-enable if needed by inserting
                // an `eprintln!` here using `_ok_count` and `n_gdn_layers`.

                // ================================================================
                // Load-time warmup dispatch for the BF16 GDN repack buffer.
                // ================================================================
                //
                // The 2.30 GB BF16 GDN repack buffer (24 layers × 96 MB) is
                // allocated as a `StorageModeShared` Metal buffer via
                // `device.new_buffer_with_bytes(..)`. The buffer pages are not
                // committed to the GPU's translation table until the kernel
                // first dispatches against it. On a fresh process, that first
                // dispatch occurs DURING the first prefill and incurs roughly
                // 280 ms of one-shot overhead versus subsequent dispatches.
                //
                // The fix: issue a tiny `M=1` dispatch against every populated
                // packed buffer right here, at the tail of preload. The grid
                // walks every (row_group, k_block) of every layer's repack
                // buffer, forcing Apple's driver to commit the page-table
                // mapping at preload time (where it's a one-time UX cost equal
                // for all users), rather than at first-inference time (where
                // it pessimizes single-shot CLI users specifically).
                //
                // Cost target <10 ms (ideally <5 ms): 24 layers ×
                // (12288/32) = 9216 TGs × 128 threads × ~64 MMA iterations.
                // Apple M3 Ultra dispatches this in roughly 3-5 ms total; the
                // observed first-prefill saving is ~280 ms.
                //
                // Correctness: the warmup dispatch writes garbage into
                // `s.qkv_buf` and `s.gate_buf`. Both are persistent scratch
                // buffers that are rewritten at the start of every layer's
                // GEMM dispatch in production; clobbering them at preload
                // time has zero observable effect on inference output.
                //
                // The warmup is naturally scoped to `bf16_gdn_qkv_gate_paired_enabled()`
                // — this entire block runs only when the resolver returns
                // true, so non-BF16-paired runs pay zero warmup cost.
                // The dispatch is also gated behind
                // `LUMEN_METAL_BF16_GDN_WARMUP` (default OFF). When explicitly
                // enabled, it attempts to commit the GPU page-table mapping
                // for the 2.30 GB packed buffer at preload time.
                //
                // The minimal warmup mitigates but does not reliably eliminate
                // the cold-start pessimization on the first one or two
                // inferences of a freshly started process; the steady-state
                // throughput improvement is real and reproducible, but the
                // cold-start cost appears to depend on macOS / Apple AGX
                // driver state that the warmup cannot address. The warmup is
                // retained env-OFF for downstream investigation. The
                // `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED` parent gate remains
                // OFF by default per `graph_reorder::bf16_gdn_qkv_gate_paired_enabled`.
                //
                // Implementation strategy is selected by `LUMEN_METAL_BF16_GDN_WARMUP_MODE`:
                //   - `minimal`: a tiny single-thread kernel reads
                //     the first BF16 element of each layer's packed buffer.
                //     Cost: <1µs per dispatch × 24 layers = <50µs total.
                //   - `full`: dispatches the actual production paired GEMM
                //     kernel at M=32 (TILE_M, fully aligned) against every
                //     packed buffer. Cost: <5 ms total. Discards Y outputs
                //     into scratch buffers production will overwrite.
                let warmup_enabled = std::env::var("LUMEN_METAL_BF16_GDN_WARMUP")
                    .ok()
                    .as_deref()
                    .map(|s| s != "0" && !s.is_empty())
                    .unwrap_or(false);
                let warmup_mode = std::env::var("LUMEN_METAL_BF16_GDN_WARMUP_MODE")
                    .ok()
                    .unwrap_or_else(|| "minimal".to_string());
                if warmup_enabled {
                    if let Some(pipelines) = self.pipelines.as_ref() {
                    let any_populated = s.repacked_gdn_qkv_gate_bf16.iter().any(|o| o.is_some());
                    if any_populated {
                        if let Some(cmd) = self.queue.new_command_buffer() {
                            if let Some(enc) = cmd.new_compute_encoder() {
                                if warmup_mode == "full" {
                                    // Production-shape warmup using the actual
                                    // paired GEMM kernel at M=32 (TILE_M).
                                    // Commits page-table for every byte that
                                    // the production dispatch will touch.
                                    let k_u32 = hidden_dim_u as u32;
                                    enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_qkv_gate_paired);
                                    enc.set_threadgroup_memory_length(8192, 0);
                                    for (slot, buf_opt) in s.repacked_gdn_qkv_gate_bf16.iter().enumerate() {
                                        let Some(packed_buf) = buf_opt.as_ref() else { continue };
                                        let Some(shape_opt) = qkv_gate_shapes.get(slot) else { continue };
                                        let Some((qkv_n_u32, gate_n_u32)) = *shape_opt else { continue };
                                        let n_total = qkv_n_u32 as u64 + gate_n_u32 as u64;
                                        if n_total == 0 { continue; }
                                        enc.set_buffer(packed_buf, 0, 0);
                                        enc.set_buffer(&s.normed_buf, 0, 1);
                                        enc.set_buffer(&s.qkv_buf, 0, 2);
                                        enc.set_buffer(&s.gate_buf, 0, 3);
                                        enc.set_bytes(&32u32.to_le_bytes(), 4);
                                        enc.set_bytes(&qkv_n_u32.to_le_bytes(), 5);
                                        enc.set_bytes(&gate_n_u32.to_le_bytes(), 6);
                                        enc.set_bytes(&k_u32.to_le_bytes(), 7);
                                        enc.dispatch_threadgroups(
                                            MTLSize::new(n_total.div_ceil(32), 1, 1),
                                            MTLSize::new(128, 1, 1),
                                        );
                                    }
                                } else {
                                    // Minimal warmup: 1-thread no-op per layer.
                                    enc.set_pipeline_state(&pipelines.bf16_paired_warmup);
                                    for buf_opt in s.repacked_gdn_qkv_gate_bf16.iter() {
                                        let Some(packed_buf) = buf_opt.as_ref() else { continue };
                                        enc.set_buffer(packed_buf, 0, 0);
                                        enc.set_buffer(&s.qkv_buf, 0, 1);
                                        enc.dispatch_threadgroups(
                                            MTLSize::new(1, 1, 1),
                                            MTLSize::new(1, 1, 1),
                                        );
                                    }
                                }
                                enc.end_encoding();
                            }
                            cmd.commit_and_wait();
                        }
                    }
                    }
                }
            }
        }

        // ====================================================================
        // Full-prefill warmup at preload time.
        // ====================================================================
        //
        // Brief: "After the GDN paired repack buffer is allocated, run a
        // complete dummy prefill at M=131 with throwaway input. This exercises
        // EVERY paired-dispatch code path including the cold-cache penalty.
        // After this dummy prefill, all page-table mappings are committed."
        //
        // Why a full prefill is stronger than the minimal touch
        // dispatch: the 1-thread no-op kernel committed only the FIRST
        // byte of each packed buffer (24 pages out of 6,144 × 24 = 147,456
        // pages of the 2.30 GB packed buffer). The remaining 147,432 pages
        // were still being faulted in lazily during the first production
        // prefill. the full prefill walks every page of the packed
        // buffer (via the production GEMM grid), commits page-table mappings
        // for every scratch buffer the production prefill will touch
        // (`qkv_buf`, `gate_buf`, `normed_buf`, `q_buf`, `k_buf`, `v_buf`,
        // `scores_buf`, KV slots, GDN h_states, GDN conv_states), and runs
        // every Metal pipeline state transition that the production
        // prefill will use. Cost: ~180 ms one-shot at preload time;
        // saving: the +50% to +54% cold-pair regression that the
        // minimal touch could not eliminate.
        //
        // Correctness scope:
        //   1. The dummy prefill mutates production scratch buffers
        //      (`qkv_buf`, etc.) but those are rewritten at the start of
        //      every layer's GEMM dispatch in production. Clobbering them
        //      at preload time has zero observable effect on inference
        //      output.
        //   2. The dummy prefill ALSO mutates `s.gdn_h_states` and
        //      `s.gdn_conv_states` (the recurrent SSM state). We must
        //      `reset_gdn_state()` after the dummy prefill or the FIRST
        //      production sequence would inherit the garbage SSM state.
        //   3. The dummy prefill mutates a throwaway `KvCache` we allocate
        //      with `max_seq_len = 131` (matching the dummy token count).
        //      The throwaway KV cache is dropped after the prefill returns;
        //      production uses its own KV cache from the caller.
        //
        // Skip conditions:
        //   - `bf16_paired_full_prefill_warmup_enabled()` returns false:
        //     either user opted out (`LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP=0`)
        //     or the parent BF16 paired gate is OFF.
        //   - No populated entries in `repacked_gdn_qkv_gate_bf16`: this is
        //     a non-BF16 model (Q8 / Q4), so no paired dispatch will fire
        //     in production and there's nothing to warm up.
        //
        // The block uses an explicit `drop(scratch_guard)` to release the
        // scratch mutex before calling `self.prefill(..)`, because
        // `prefill` re-acquires the same mutex internally. We re-fetch the
        // necessary `KvCacheConfig` parameters from scratch under the
        // current guard, then drop it.
        {
            use super::graph_reorder as gr;
            if gr::bf16_paired_full_prefill_warmup_enabled() {
                // Re-fetch scratch for the warmup metadata. The scratch
                // guard is the same `scratch_guard` opened at the top of
                // this function (still held here).
                let s_ref = scratch_guard.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "scratch unexpectedly None at warmup time".into(),
                    )
                })?;
                let any_populated = s_ref.repacked_gdn_qkv_gate_bf16.iter().any(|o| o.is_some());
                let num_kv_heads_u = s_ref.num_kv_heads;
                let head_dim_u = s_ref.head_dim;
                let num_layers_u = s_ref.num_layers;

                if any_populated && num_layers_u > 0 && num_kv_heads_u > 0 && head_dim_u > 0 {
                    // Release the scratch lock before calling `prefill`,
                    // which re-acquires it internally.
                    drop(scratch_guard);

                    // Throwaway KvCache: F32 KV at exactly 131 positions
                    // (matching the dummy token count = production
                    // paired-bench M). KV memory cost:
                    //   131 tokens × num_kv_heads × head_dim × 4 (F32) × 2 (K+V)
                    //   × num_layers
                    // For Qwen3.5-9B (num_kv_heads=2, head_dim=128,
                    // num_layers=32): 131 × 2 × 128 × 4 × 2 × 32 = 8.6 MB.
                    // Dropped at the end of this scope.
                    const DUMMY_M: usize = 131;
                    let kv_config = crate::kv::KvCacheConfig {
                        max_seq_len: DUMMY_M,
                        num_layers: num_layers_u,
                        num_kv_heads: num_kv_heads_u,
                        head_dim: head_dim_u,
                        precision: crate::kv::KvPrecision::F32,
                    };

                    if let Ok(mut throwaway_kv) = crate::kv::KvCache::new(kv_config) {
                        // Synthesize a `DUMMY_M`-token zero prompt. Token 0
                        // is a valid row in the embed table for every
                        // model we ship (vocab size >= 1). The embed
                        // kernel will read row 0 from the embed table.
                        let dummy_tokens: Vec<u32> = vec![0u32; DUMMY_M];

                        // Drive a full prefill. We intentionally `let _ =`
                        // the result — the hidden state is discarded; the
                        // only side-effect we care about is the
                        // GPU page-table commit + per-process residency
                        // state that the production prefill will reuse.
                        let _ = self.prefill(&dummy_tokens, weights, &mut throwaway_kv);

                        // Reset GDN recurrent state — the dummy prefill
                        // wrote garbage SSM state into `gdn_h_states` and
                        // `gdn_conv_states`. Without this reset, the
                        // first production sequence would inherit this
                        // garbage and produce divergent output.
                        self.reset_gdn_state();

                        // `throwaway_kv` drops here (KV memory returned).
                    }
                }
            }
        }

        Ok(())
    }
}
