//! GPU-resident weight preloading for Metal backend.
//!
//! Packs all layer weight data and global tensors into a single contiguous
//! `StorageModePrivate` Metal buffer, eliminating TLB misses, reducing virtual
//! address ranges, and enabling GPU memory controller optimizations.

use super::ffi::MetalBuffer;
use super::repack_q4;
use super::repack_q8;
use super::types::{CachedLayerMeta, CachedMoeMeta};
use super::{MetalF32Backend, PAGE_SIZE};
use crate::error::RuntimeError;
use lumen_format::index::TensorSlice;
use lumen_format::quantization::QuantScheme;

/// Quant schemes the Metal DENSE DECODE path can serve for named layer
/// slices. Q4_1 is deliberately absent: Metal's Q4_1 kernels cover only MoE
/// experts and batched prefill; the dense decode dispatch falls through to
/// an F32-reading pipeline that would silently misread Q4_1 blocks.
fn dense_slice_quant_supported(q: QuantScheme) -> bool {
    matches!(
        q,
        QuantScheme::F32
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q8_0
            | QuantScheme::Q4_0
    )
}

/// Reject layer tensors the Metal dispatch paths would misparse, before any
/// Metal path can execute over the blob. Called once per layer at GPU-resident
/// preload, and at zero-copy layer-buffer creation for the streaming /
/// non-resident paths (which never run preload).
pub(super) fn validate_layer_quants(
    layer: usize,
    st: &lumen_format::index::SubtensorOffsets,
) -> Result<(), RuntimeError> {
    // Reject quant schemes the Metal DENSE decode path has no dispatch
    // kernels for. Without this, a `--target generic` LBC (K-quant or
    // source-fidelity Q4_1 tensors intact, fine on CUDA) runs "successfully"
    // and generates garbage — the dispatch catch-alls feed those bytes to an
    // F32-reading pipeline. Fail loudly with the remedy instead.
    for (name, slice) in st.named_slices() {
        if slice.length == 0 {
            continue;
        }
        if !dense_slice_quant_supported(slice.quant) {
            return Err(RuntimeError::Compute(format!(
                "layer {layer} tensor '{name}' is {:?}: the Metal \
                 backend has no dense DECODE dispatch kernels for this \
                 quant scheme. This LBC was converted for a different \
                 backend (`--target generic`); re-convert with \
                 `lumen convert --target metal` (K-quant and legacy \
                 Q5_0 layer tensors are upcast to Q8_0, Q4_1 is \
                 re-quantized to Q4_0).",
                slice.quant
            )));
        }
    }
    // The dense-FFN gate+up dispatch arms for Q4_0/F16/Bf16/F32
    // select on the gate's quant alone and bind w_up_off regardless,
    // and the fused shaders stride both pointers with one row_bytes
    // derived from the gate's scheme — a gate/up quant mismatch
    // (producible: the converter upcasts K-quant tensors per tensor)
    // would compute silently wrong output.
    if st.w_gate.length > 0 && st.w_up.length > 0 && st.w_gate.quant != st.w_up.quant {
        return Err(RuntimeError::Compute(format!(
            "layer {layer}: ffn_gate is {:?} but ffn_up is {:?}: the \
             Metal fused FFN kernels require the pair to share one \
             quant scheme. Re-convert with a uniform quantization \
             (e.g. `lumen convert --target metal --requant q8_0`).",
            st.w_gate.quant, st.w_up.quant
        )));
    }
    // On GDN layers the decode path pairs attn_gate with the qkv
    // route: the Q8_0 qkv+gate 2-stream kernel (default-on) decodes
    // the gate as Q8_0 without consulting its quant, and the non-Q8
    // routes' gate fallbacks read schemes they lack an arm for as
    // F32 — so a Q8_0/non-Q8_0 split between attn_qkv and
    // attn_gate in either direction computes silently wrong output.
    // Full-attention layers dispatch attn_gate on its own quant and
    // are exempt.
    if let (Some(1), Some(gate)) = (st.layer_type, st.attn_gate.as_ref()) {
        if gate.length > 0
            && ((st.wq.quant == QuantScheme::Q8_0) != (gate.quant == QuantScheme::Q8_0))
        {
            return Err(RuntimeError::Compute(format!(
                "layer {layer}: attn_qkv is {:?} but attn_gate is {:?}: \
                 the Metal GDN decode path requires the pair to agree \
                 on whether it is Q8_0. Re-convert with \
                 `lumen convert --target metal` (it writes the pair \
                 uniformly).",
                st.wq.quant, gate.quant
            )));
        }
    }
    // The GDN gate pipelines (decode and prefill) decode ssm_alpha/ssm_beta
    // as Q8_0 unconditionally (offsets are bound without consulting their
    // quant). The converter stores them as Q8_0 on every Metal-target path,
    // but source-fidelity conversions for other targets keep F32 gates, and
    // `--dequantize` writes F32 when the source stores them as Q8_0 —
    // these pipelines would parse those bytes as Q8_0 blocks, computing
    // silently wrong output.
    for (name, slice) in [("ssm_alpha", &st.ssm_alpha), ("ssm_beta", &st.ssm_beta)] {
        if let Some(s) = slice {
            if s.length > 0 && s.quant != QuantScheme::Q8_0 {
                return Err(RuntimeError::Compute(format!(
                    "layer {layer}: {name} is {:?}: the Metal GDN gate \
                     pipelines read ssm_alpha/ssm_beta as Q8_0. Re-convert \
                     with `lumen convert --target metal` (it stores these \
                     tensors as Q8_0).",
                    s.quant
                )));
            }
        }
    }
    // The MoE expert dispatch has the same shape: it selects the
    // fused gate+up pipeline on the expert gate's quant alone
    // (CachedMoeMeta carries no up quant), so a per-expert gate/up
    // mismatch would also compute silently wrong output.
    if let Some(experts) = st.experts.as_ref() {
        for (i, e) in experts.iter().enumerate() {
            if e.gate.length > 0 && e.up.length > 0 && e.gate.quant != e.up.quant {
                return Err(RuntimeError::Compute(format!(
                    "layer {layer} expert {i}: gate is {:?} but up is {:?}: \
                     the Metal fused expert FFN kernels require the pair \
                     to share one quant scheme. Re-convert with \
                     `lumen convert --target metal` from a source GGUF \
                     whose expert tensors share one quantization.",
                    e.gate.quant, e.up.quant
                )));
            }
        }
        // Expert dispatch takes its quant schemes from expert 0 alone
        // (CachedMoeMeta carries one scheme set for the whole bank), so a
        // layer whose experts differ from expert 0 would decode the others
        // at the wrong stride. The format stores per-expert schemes and does
        // not enforce uniformity; reject the divergence here.
        if let Some(first) = experts.first() {
            for (i, e) in experts.iter().enumerate().skip(1) {
                let pairs = [
                    ("gate", e.gate.quant, first.gate.quant),
                    ("up", e.up.quant, first.up.quant),
                    ("down", e.down.quant, first.down.quant),
                ];
                if let Some((name, got, want)) = pairs.iter().find(|(_, a, b)| a != b).copied() {
                    return Err(RuntimeError::Compute(format!(
                        "layer {layer} expert {i}: {name} is {got:?} but \
                         expert 0's is {want:?}: the Metal expert dispatch \
                         applies expert 0's quant schemes to every expert. \
                         Re-convert from a source GGUF whose experts share \
                         one quantization.",
                    )));
                }
            }
        }
    }
    // The shared-expert FFN dispatch selects on the gate's quant alone and
    // binds up_off regardless (CachedMoeMeta carries no shared-expert up
    // quant), and only the Q8_0/Q4_0/Q4_1 arms select genuinely fused
    // gate+up+SwiGLU shaders — the F16/Bf16/F32 arms select plain matmuls
    // whose parameter lists end before the up-projection binding, silently
    // dropping up and SwiGLU.
    let shexp_present = [
        st.shared_expert_gate.as_ref(),
        st.shared_expert_up.as_ref(),
        st.shared_expert_down.as_ref(),
    ]
    .map(|t| t.is_some_and(|s| s.length > 0));
    if shexp_present.iter().any(|&p| p) && !shexp_present.iter().all(|&p| p) {
        // The runtime uses the gate's presence as the shared-expert feature
        // flag and unwraps the other two tensors during dispatch.
        return Err(RuntimeError::Compute(format!(
            "layer {layer}: incomplete shared-expert tensors (gate/up/down \
             present: {shexp_present:?}): a shared expert requires all \
             three. Re-convert with `lumen convert --target metal`."
        )));
    }
    if let (Some(gate), Some(up)) = (st.shared_expert_gate.as_ref(), st.shared_expert_up.as_ref()) {
        if gate.length > 0 && up.length > 0 {
            if gate.quant != up.quant {
                return Err(RuntimeError::Compute(format!(
                    "layer {layer}: shared-expert gate is {:?} but up is {:?}: \
                     the Metal fused shared-expert kernels require the pair to \
                     share one quant scheme. Re-convert with \
                     `lumen convert --target metal`.",
                    gate.quant, up.quant
                )));
            }
            if !matches!(gate.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0) {
                return Err(RuntimeError::Compute(format!(
                    "layer {layer}: shared-expert gate/up is {:?}: the Metal \
                     shared-expert FFN has fused kernels only for Q8_0/Q4_0 \
                     (its Q4_1 arm is unreachable — the layer-tensor \
                     allowlist rejects Q4_1 upstream). Re-convert with \
                     `lumen convert --target metal` (it quantizes the \
                     shared-expert tensors).",
                    gate.quant
                )));
            }
        }
    }
    // Every Metal shader reads norm tensors, the MoE routers, and the SSM
    // scalar tensors as F32 without consulting their quant (CUDA rejects
    // non-F32 norms at load; Metal must too). The converter writes them F32
    // on its forced paths, but a source GGUF storing e.g. F16 norms passes
    // the allowlist above and would be misread.
    let f32_only: [(&str, Option<&TensorSlice>); 11] = [
        ("attn_norm", Some(&st.attn_norm)),
        ("ffn_norm", Some(&st.ffn_norm)),
        ("attn_post_norm", st.attn_post_norm.as_ref()),
        ("attn_q_norm", st.attn_q_norm.as_ref()),
        ("attn_k_norm", st.attn_k_norm.as_ref()),
        ("ssm_norm", st.ssm_norm.as_ref()),
        ("ssm_a", st.ssm_a.as_ref()),
        ("ssm_conv1d", st.ssm_conv1d.as_ref()),
        ("ssm_dt", st.ssm_dt.as_ref()),
        ("router_weight", st.router_weight.as_ref()),
        ("ffn_gate_inp_shexp", st.ffn_gate_inp_shexp.as_ref()),
    ];
    for (name, slice) in f32_only {
        if let Some(s) = slice {
            if s.length > 0 && s.quant != QuantScheme::F32 {
                return Err(RuntimeError::Compute(format!(
                    "layer {layer}: {name} is {:?}: the Metal kernels read \
                     this tensor as F32. Re-convert from a source GGUF whose \
                     norm/router/SSM-scalar tensors are F32.",
                    s.quant
                )));
            }
        }
    }
    Ok(())
}

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
        // for MoE 35B-A3B Q8/Q4 LBCs where the legacy Pass 1/2/3 dup pushes
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
                RuntimeError::Compute(format!(" MMAP_ONLY: probe layer 0 failed: {}", e))
            })?;
            let probe_ptr = lv0.as_bytes().as_ptr() as usize;
            let probe_aligned = probe_ptr != 0 && (probe_ptr % PAGE_SIZE == 0);
            if probe_aligned {
                layer_ptrs.reserve(num_layers);
                for layer in 0..num_layers {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " MMAP_ONLY: probe layer {} failed: {}",
                            layer, e
                        ))
                    })?;
                    let bytes = lv.as_bytes();
                    let p = bytes.as_ptr() as usize;
                    let l = bytes.len();
                    layer_ptrs.push((p, l));
                    if p < mmap_min_ptr {
                        mmap_min_ptr = p;
                    }
                    let end = p.saturating_add(l);
                    if end > mmap_max_end {
                        mmap_max_end = end;
                    }
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
                    "Failed to get layer {} for GPU-resident loading: {}",
                    layer, e
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
            validate_layer_quants(layer, st)?;
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
                            expert_gate_offs: experts
                                .iter()
                                .map(|e| base + e.gate.offset)
                                .collect(),
                            expert_up_offs: experts.iter().map(|e| base + e.up.offset).collect(),
                            expert_down_offs: experts
                                .iter()
                                .map(|e| base + e.down.offset)
                                .collect(),
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
                wk_off: if st.wk.length > 0 {
                    Some(base + st.wk.offset)
                } else {
                    None
                },
                wv_off: if st.wv.length > 0 {
                    Some(base + st.wv.offset)
                } else {
                    None
                },
                wk_quant: if st.wk.length > 0 {
                    Some(st.wk.quant)
                } else {
                    None
                },
                wv_quant: if st.wv.length > 0 {
                    Some(st.wv.quant)
                } else {
                    None
                },
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
            // mmap span is ~16.3 GB; MoE-35B BF16 ~60 GB if we ever extend).
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
                self.device
                    .new_buffer_no_copy(mmap_min_ptr as *mut std::ffi::c_void, span)
            }
            .ok_or_else(|| {
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
                RuntimeError::Compute(
                    "Final norm buffer not initialized for unified preload".into(),
                )
            })?;
            let norm_len = norm_buf_ref.length() as usize;

            let proj_buf_ref = self.output_proj_buf.as_ref().ok_or_else(|| {
                RuntimeError::Compute(
                    "Output proj buffer not initialized for unified preload".into(),
                )
            })?;
            let proj_len = proj_buf_ref.length() as usize;

            // Weight tying: output_proj shares embedding storage (no separate
            // allocation) — but only when both buffers hold the same
            // representation. The frontend can admit a raw non-F32 head while
            // the embedding stays F32 (BF16 models on Metal admit the raw
            // head but not the raw embedding); the final-projection shader is
            // selected by output_proj_quant, so aliasing it to differently-
            // represented embedding bytes would compute wrong logits.
            let tie_alias = self.weight_tying && self.output_proj_quant == self.embedding_quant;
            let effective_proj_len = if tie_alias { 0 } else { proj_len };
            global_bytes = embed_len + norm_len + effective_proj_len;
            // Include globals in the unified private buffer.
            include_globals = true;

            let (eo, no_, po) = if include_globals {
                let eo = cursor;
                cursor = align(cursor + embed_len);
                let no_ = cursor;
                cursor = align(cursor + norm_len);
                if tie_alias {
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
                    total_size,
                    total_size as f64 / (1024.0 * 1024.0)
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
                        embed_buf_ref.contents() as *const u8,
                        dst_base.add(embed_offset),
                        embed_len,
                    );
                    std::ptr::copy_nonoverlapping(
                        norm_buf_ref.contents() as *const u8,
                        dst_base.add(norm_offset),
                        norm_len,
                    );
                    if !tie_alias {
                        std::ptr::copy_nonoverlapping(
                            proj_buf_ref.contents() as *const u8,
                            dst_base.add(proj_offset),
                            proj_len,
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
                    total_size,
                    total_size as f64 / (1024.0 * 1024.0)
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
            let global_mb = if include_globals {
                global_bytes as f64 / (1024.0 * 1024.0)
            } else {
                0.0
            };
            let total_mb = total_size as f64 / (1024.0 * 1024.0);
            // GPU-resident buffer info available via MetalF32Backend::gpu_resident_summary().
            let _ = (num_layers, layer_mb, global_mb, total_mb, include_globals);

            s.gpu_unified_weight_buf = Some(private_buf);
            s.gpu_layer_offsets = layer_offsets;
            if include_globals {
                s.gpu_global_offsets = Some((embed_offset, norm_offset, proj_offset));
            } else {
                s.gpu_global_offsets = None; // Forces fallback to separate shared buffers
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
        let _ = (
            global_bytes,
            total_size,
            layer_bytes_total,
            include_globals,
            embed_offset,
            norm_offset,
            proj_offset,
        );

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
                    s.shared_expert_gate_buf =
                        Some(self.device.new_buffer(se_inter * 4).ok_or_else(|| {
                            RuntimeError::Compute(
                                "Failed to allocate shared_expert_gate_buf".into(),
                            )
                        })?);
                    s.shared_expert_down_buf =
                        Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                            RuntimeError::Compute(
                                "Failed to allocate shared_expert_down_buf".into(),
                            )
                        })?);
                }

                // Partial RoPE: Qwen3.5 uses partial_rotary_factor=0.25,
                // meaning only the first head_dim/4 dimensions of each head are rotated.
                let head_dim = s.head_dim;
                if head_dim >= 128 {
                    s.rotary_dim = head_dim / 4; // 128/4 = 32 for Qwen3.5-9B, 256/4 = 64 for -35B
                }

                // Allocate attention gate scratch buffer (for full attention layers with attn_gate)
                let has_attn_gate = s
                    .cached_layer_meta
                    .iter()
                    .any(|m| m.attn_gate_off.is_some());
                if has_attn_gate {
                    let hidden = s.hidden_dim;
                    s.attn_gate_buf =
                        Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
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
                let n_linear = s
                    .cached_layer_meta
                    .iter()
                    .filter(|m| m.layer_type == Some(1))
                    .count();
                let n_full = s
                    .cached_layer_meta
                    .iter()
                    .filter(|m| m.layer_type == Some(0))
                    .count();
                let n_moe = s
                    .cached_layer_meta
                    .iter()
                    .filter(|m| m.moe_meta.is_some())
                    .count();
                let n_shared = s
                    .cached_layer_meta
                    .iter()
                    .filter(|m| m.shared_expert_gate_off.is_some())
                    .count();
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
            let n_linear = s
                .cached_layer_meta
                .iter()
                .filter(|m| m.layer_type == Some(1))
                .count();
            if n_linear > 0 {
                // GDN dims from the resolved SSM dims (9B {32,16,128,4} default,
                // 27B {48,16,128,4}), populated in init() from hyperparams.gdn_dims().
                let gdn_num_v_heads = s.gdn_num_v_heads; // ssm.time_step_rank
                let gdn_num_k_heads = s.gdn_num_k_heads; // ssm.group_count
                let gdn_head_dim = s.gdn_head_dim; // ssm.state_size
                let conv_kernel_size = s.gdn_conv_kernel_size; // ssm.conv_kernel
                                                               // Fused QKV channels: 2*qk_dim + v_dim (9B=8192, 27B=10240).
                let gdn_qkv_dim =
                    2 * gdn_num_k_heads * gdn_head_dim + gdn_num_v_heads * gdn_head_dim;
                // V / gate / output-projection width = num_v_heads*head_dim (9B=4096, 27B=6144).
                let gdn_q_dim = gdn_num_v_heads * gdn_head_dim;
                let hidden = s.hidden_dim;

                // h_state: [num_v_heads, head_dim, head_dim] per GDN layer
                let h_state_size = gdn_num_v_heads * gdn_head_dim * gdn_head_dim;
                // conv_state: [(kernel_size - 1) * qkv_dim] per GDN layer
                let conv_state_size = (conv_kernel_size - 1) * gdn_qkv_dim;

                // Persistent h_state: F32 (4 B/elem). Zero-initialized (new
                // sequence starts with zero state).
                let mut h_states = Vec::with_capacity(n_linear);
                let mut conv_states = Vec::with_capacity(n_linear);
                for _ in 0..n_linear {
                    let h_buf = self.device.new_buffer(h_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN h_state buffer".into())
                    })?;
                    h_buf.write_f32(&vec![0.0f32; h_state_size]);
                    h_states.push(h_buf);

                    let c_buf = self.device.new_buffer(conv_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN conv_state buffer".into())
                    })?;
                    c_buf.write_f32(&vec![0.0f32; conv_state_size]);
                    conv_states.push(c_buf);
                }

                s.gdn_h_states = h_states;
                // F16 h_state mirrors: allocated lazily on the first decode touch (the
                // default F16 decode recurrence; kept length-synced with gdn_h_states).
                s.gdn_h_states_f16 = (0..n_linear)
                    .map(|_| std::cell::RefCell::new(None))
                    .collect();
                s.gdn_conv_states = conv_states;
                s.gdn_conv_positions = vec![0u32; n_linear];
                s.gdn_conv_kernel_size = conv_kernel_size;
                s.gdn_num_layers = n_linear;

                // Allocate GDN scratch buffers using GDN-specific dimensions
                // (gdn_q_dim = num_v_heads*head_dim, computed above).
                s.gdn_alpha_buf =
                    Some(self.device.new_buffer(gdn_num_v_heads * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha buffer".into())
                    })?);
                s.gdn_beta_buf =
                    Some(self.device.new_buffer(gdn_num_v_heads * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta buffer".into())
                    })?);
                // Output of state query: [num_v_heads * head_dim] (9B=4096, 27B=6144)
                s.gdn_output_buf =
                    Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN output buffer".into())
                    })?);
                // SSM output projection result: [hidden_dim]
                s.gdn_ssm_proj_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN ssm_proj buffer".into())
                })?);
                // Attention gate sigmoid output: [v_dim] (gate applied BEFORE ssm_out_proj)
                s.gdn_gate_sigmoid_buf =
                    Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN gate_sigmoid buffer".into())
                    })?);
                // L2-norm scaled output: [num_v_heads * head_dim] (9B=4096, 27B=6144)
                s.gdn_normed_out_buf =
                    Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN normed_out buffer".into())
                    })?);
                // Q8_0 matvec outputs for alpha/beta gate projections [num_v_heads] f32
                s.gdn_alpha_raw_buf =
                    Some(self.device.new_buffer(gdn_num_v_heads * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha_raw buffer".into())
                    })?);
                s.gdn_beta_raw_buf =
                    Some(self.device.new_buffer(gdn_num_v_heads * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta_raw buffer".into())
                    })?);
                // Conv1d output for all QKV channels [qkv_dim] f32 (9B=8192, 27B=10240)
                s.gdn_qkv_conv_buf =
                    Some(self.device.new_buffer(gdn_qkv_dim * 4).ok_or_else(|| {
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
                            gu_offsets[e * 2] = moe_meta.expert_gate_offs[e];
                            gu_offsets[e * 2 + 1] = moe_meta.expert_up_offs[e];
                        }
                        let gu_bytes: Vec<u8> =
                            gu_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let gu_buf =
                            self.device
                                .new_buffer_with_bytes(&gu_bytes)
                                .ok_or_else(|| {
                                    RuntimeError::Compute(
                                        "Failed to allocate MoE gate_up offset table".into(),
                                    )
                                })?;
                        gate_up_vecs.push(Some(gu_buf));

                        // Build down offset table: [n_experts] u64
                        let mut d_offsets = vec![0u64; n_experts];
                        for e in 0..n_experts.min(moe_meta.expert_down_offs.len()) {
                            d_offsets[e] = moe_meta.expert_down_offs[e];
                        }
                        let d_bytes: Vec<u8> =
                            d_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let d_buf =
                            self.device.new_buffer_with_bytes(&d_bytes).ok_or_else(|| {
                                RuntimeError::Compute(
                                    "Failed to allocate MoE down offset table".into(),
                                )
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
                        let buf =
                            self.device
                                .new_buffer_with_bytes(&off_bytes)
                                .ok_or_else(|| {
                                    RuntimeError::Compute(
                                        "Failed to allocate MoE shared expert down offset".into(),
                                    )
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
                            " repack: failed to get layer {}: {}",
                            layer, e
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
                                " repack: failed to read w_down at layer {}: {}",
                                layer, e
                            ))
                        })?;
                        match repack_q8::build_repacked_buffer_single(
                            &self.device,
                            src,
                            hidden_dim_u,
                            inter_dim_u,
                        ) {
                            Ok(buf) => {
                                down_ok_count += 1;
                                Some(buf)
                            }
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
                                " repack: failed to read w_gate at layer {}: {}",
                                layer, e
                            ))
                        })?;
                        let src_u = lv.subtensor_bytes(&st.w_up).map_err(|e| {
                            RuntimeError::Compute(format!(
                                " repack: failed to read w_up at layer {}: {}",
                                layer, e
                            ))
                        })?;
                        match repack_q8::build_repacked_buffer_pair(
                            &self.device,
                            src_g,
                            src_u,
                            inter_dim_u,
                            hidden_dim_u,
                        ) {
                            Ok(buf) => {
                                gate_up_ok_count += 1;
                                Some(buf)
                            }
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

        // MLX-style Q4_0 FFN-down decode-qmv repack (default). Builds per-layer
        // sequential-nibble qweights + f32 scales for
        // the qmv_q4_0_residual decode kernel; absent => NR2 fallback. Requires
        // inter_dim % 512 == 0 and hidden_dim % 8 == 0 (Qwen3.5-9B: 12288, 4096 OK).
        if crate::metal::q4_fast_decode_enabled() {
            let hidden_dim_u = s.hidden_dim;
            let inter_dim_u = s.inter_dim;
            let mut qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            // F16-scales FFN-down (default): build the
            // scale buffer as f16 (2 B/block) instead of f32 (4 B) for the f16sc
            // kernel. Same sequential-nibble qweights either way. Only valid when
            // the f16sc pipeline compiled; otherwise stay on the f32 builder so the
            // dispatch falls back cleanly to the f32-scale qmv_q4_0_residual.
            let down_f16sc = crate::metal::q4_qmv_down_f16sc_enabled()
                && self
                    .pipelines
                    .as_ref()
                    .map(|p| p.qmv_q4_0_residual_f16sc.is_some())
                    .unwrap_or(false);
            if inter_dim_u % 512 == 0 && hidden_dim_u % 8 == 0 {
                for layer in 0..num_layers {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("qmv-down: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;
                    if st.w_down.quant == QuantScheme::Q4_0 && st.w_down.length > 0 {
                        let src = lv.subtensor_bytes(&st.w_down).map_err(|e| {
                            RuntimeError::Compute(format!("qmv-down: read w_down {}: {}", layer, e))
                        })?;
                        let built = if down_f16sc {
                            repack_q4::build_qmv_decode_buffers_f16sc(
                                &self.device,
                                src,
                                hidden_dim_u,
                                inter_dim_u,
                            )
                        } else {
                            repack_q4::build_qmv_decode_buffers(
                                &self.device,
                                src,
                                hidden_dim_u,
                                inter_dim_u,
                            )
                        };
                        match built {
                            Ok((qw, sc)) => {
                                qw_vecs.push(Some(qw));
                                sc_vecs.push(Some(sc));
                            }
                            Err(_) => {
                                qw_vecs.push(None);
                                sc_vecs.push(None);
                            }
                        }
                    } else {
                        qw_vecs.push(None);
                        sc_vecs.push(None);
                    }
                }
            }
            let nbuilt = qw_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[qmv-down] flag ON: built {}/{} qmv buffers (hidden={}, inter={})",
                nbuilt, num_layers, s.hidden_dim, s.inter_dim
            );
            s.qmv_down_qw = qw_vecs;
            s.qmv_down_scales = sc_vecs;
        }

        // MLX-style Q4_0 GDN qkv-projection decode-qmv repack (default).
        // Builds per-GDN-layer
        // sequential-nibble qweights + f32 scales for the `qmv_q4_0_rmsnorm`
        // kernel (fused RMSNorm + matvec); absent => the existing NR2 fused path.
        //
        // The qkv weight `st.wq` is Q4_0 [qkv_dim, hidden_dim]. qkv_dim (out rows)
        // is derived from the tensor byte length: Q4_0 row = (hidden/32)*18 bytes.
        // Requires hidden_dim % 512 == 0 (in_dim) and qkv_dim % 8 == 0 (out_dim).
        // Qwen3.5-9B GDN: in=hidden=4096, out=qkv_dim=8192 -> OK.
        //
        // The Vec is indexed by `gdn_idx` (sequential GDN-layer counter 0..n_gdn-1),
        // matching `gdn_h_states`/`gdn_conv_states`: non-GDN layers do NOT push an
        // entry (same convention as the BF16 paired GDN repack below).
        if crate::metal::q4_fast_decode_enabled() {
            let hidden_dim_u = s.hidden_dim;
            let n_gdn_layers = s
                .cached_layer_meta
                .iter()
                .filter(|m| m.layer_type == Some(1))
                .count();
            let mut qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
            let mut sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
            // GDN attn_gate decode-qmv buffers (matrix #1). Pushed in LOCKSTEP with the
            // qkv vecs inside the SAME GDN loop so both stay indexed by `gdn_idx`.
            let mut gate_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
            let mut gate_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
            // Q4_0 bytes per output row for in_dim = hidden_dim (2B scale + 16B nibbles
            // per 32-element block). hidden % 32 is implied by hidden % 512 == 0.
            let q4_row_bytes = (hidden_dim_u / 32) * 18;
            // F16-scales GDN QKV-in-proj + attn_gate (default):
            // build both decode-qmv scale buffers as f16 (2 B/block) instead of f32
            // (4 B). Only when the f16sc kernel compiled; otherwise stay on the f32
            // builder so the GDN dispatch falls back cleanly to the f32-scale
            // qmv_q4_0_rmsnorm. Same sequential-nibble qweights either way; f16 = the
            // on-disk Q4_0 native scale precision -> byte-identical. Mirrors lm_head f16sc.
            let proj_f16sc = crate::metal::q4_proj_f16sc_enabled()
                && self
                    .pipelines
                    .as_ref()
                    .map(|p| p.qmv_q4_0_rmsnorm_f16sc.is_some())
                    .unwrap_or(false);
            if hidden_dim_u % 512 == 0 && q4_row_bytes > 0 {
                for layer in 0..num_layers {
                    // Only GDN layers (layer_type == Some(1)) participate; skip others
                    // entirely (no push) so the index aligns with gdn_idx.
                    if s.cached_layer_meta[layer].layer_type != Some(1) {
                        continue;
                    }
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("qmv-proj: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;
                    // qkv_dim = out rows derived from tensor length / Q4_0 row bytes.
                    let qkv_dim = (st.wq.length as usize) / q4_row_bytes;
                    if st.wq.quant == QuantScheme::Q4_0
                        && st.wq.length > 0
                        && st.wq.length as usize == qkv_dim * q4_row_bytes
                        && qkv_dim % 8 == 0
                    {
                        let src = lv.subtensor_bytes(&st.wq).map_err(|e| {
                            RuntimeError::Compute(format!("qmv-proj: read wq {}: {}", layer, e))
                        })?;
                        let qkv_build = if proj_f16sc {
                            repack_q4::build_qmv_decode_buffers_f16sc(
                                &self.device,
                                src,
                                qkv_dim,
                                hidden_dim_u,
                            )
                        } else {
                            repack_q4::build_qmv_decode_buffers(
                                &self.device,
                                src,
                                qkv_dim,
                                hidden_dim_u,
                            )
                        };
                        match qkv_build {
                            Ok((qw, sc)) => {
                                qw_vecs.push(Some(qw));
                                sc_vecs.push(Some(sc));
                            }
                            Err(_) => {
                                qw_vecs.push(None);
                                sc_vecs.push(None);
                            }
                        }
                    } else {
                        qw_vecs.push(None);
                        sc_vecs.push(None);
                    }

                    // --- Matrix #1: GDN attn_gate (st.attn_gate, [q_dim, hidden_dim]) ---
                    // rmsnorm-fused -> qmv_q4_0_rmsnorm. Same in_dim (hidden) and norm
                    // (attn_norm) as qkv; out rows derived from the tensor byte length.
                    // Push EXACTLY once per GDN layer to keep gate vecs aligned to gdn_idx.
                    let gate_built = match &st.attn_gate {
                        Some(ag)
                            if ag.quant == QuantScheme::Q4_0
                                && ag.length > 0
                                && (ag.length as usize) % q4_row_bytes == 0
                                && ((ag.length as usize) / q4_row_bytes) % 8 == 0 =>
                        {
                            let gate_dim = (ag.length as usize) / q4_row_bytes;
                            match lv.subtensor_bytes(ag) {
                                Ok(gsrc) => {
                                    let gate_build = if proj_f16sc {
                                        repack_q4::build_qmv_decode_buffers_f16sc(
                                            &self.device,
                                            gsrc,
                                            gate_dim,
                                            hidden_dim_u,
                                        )
                                    } else {
                                        repack_q4::build_qmv_decode_buffers(
                                            &self.device,
                                            gsrc,
                                            gate_dim,
                                            hidden_dim_u,
                                        )
                                    };
                                    match gate_build {
                                        Ok((qw, sc)) => Some((qw, sc)),
                                        Err(_) => None,
                                    }
                                }
                                Err(_) => None,
                            }
                        }
                        _ => None,
                    };
                    match gate_built {
                        Some((qw, sc)) => {
                            gate_qw_vecs.push(Some(qw));
                            gate_sc_vecs.push(Some(sc));
                        }
                        None => {
                            gate_qw_vecs.push(None);
                            gate_sc_vecs.push(None);
                        }
                    }
                }
            }
            let nbuilt = qw_vecs.iter().filter(|b| b.is_some()).count();
            let ngate = gate_qw_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[qmv-proj] flag ON: built {}/{} GDN qkv + {}/{} GDN attn_gate qmv buffers (hidden={})",
                nbuilt, n_gdn_layers, ngate, n_gdn_layers, hidden_dim_u
            );
            s.qmv_gdn_qkv_qw = qw_vecs;
            s.qmv_gdn_qkv_scales = sc_vecs;
            s.qmv_gdn_attn_gate_qw = gate_qw_vecs;
            s.qmv_gdn_attn_gate_scales = gate_sc_vecs;

            // --- Matrices #2 (full-attn Q+gate, st.wq) and #3 (full-attn Wo, st.wo) ---
            // Indexed by `layer_idx` (0..num_layers; None for GDN layers, matching the
            // qmv_down convention). Full-attn = layer_type != Some(1).
            //   #2 st.wq  [qgate_dim, hidden_dim]: in_dim = hidden (%512), out = qgate_dim (%8).
            //   #3 st.wo  [hidden_dim, q_dim]:     in_dim = q_dim (%512),  out = hidden_dim (%8).
            // Wo's in_dim is q_dim, so it uses a SEPARATE Q4 row-byte stride.
            let q_dim_u = s.q_dim;
            let q4_row_bytes_qdim = (q_dim_u / 32) * 18; // for Wo in_dim = q_dim
            let mut wq_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wq_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wo_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wo_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                // GDN layers do not participate (None placeholder keeps layer_idx alignment).
                if s.cached_layer_meta[layer].layer_type == Some(1) {
                    wq_qw_vecs.push(None);
                    wq_sc_vecs.push(None);
                    wo_qw_vecs.push(None);
                    wo_sc_vecs.push(None);
                    continue;
                }
                let lv = weights.get_layer_raw(layer).map_err(|e| {
                    RuntimeError::Compute(format!("qmv-proj attn: layer {}: {}", layer, e))
                })?;
                let st = &lv.subtensors;

                // #2: Q+gate projection (st.wq). out = qgate_dim from byte length.
                // ONLY built for Q+gate-fusion layers: those are the layers whose decode
                // dispatch (the has_qgate_fusion + use_fused_attn_norm Q+gate path)
                // consumes this buffer. A non-fusion full-attn layer would size wq to
                // qkv_dim and use a different (un-qmv'd) dispatch, so skip it here.
                let wq_built = if s.cached_layer_meta[layer].has_qgate_fusion
                    && hidden_dim_u % 512 == 0
                    && q4_row_bytes > 0
                {
                    let qgate_dim = (st.wq.length as usize) / q4_row_bytes;
                    // Invariant guard: under Q+gate fusion the documented wq output dim
                    // is exactly 2*q_dim (Q + gate interleaved). Catches silent mis-sizing
                    // (debug only; runtime falls back via the strict byte-length check).
                    debug_assert!(
                        qgate_dim == 2 * q_dim_u,
                        "qmv-proj wq layer {}: derived qgate_dim {} != 2*q_dim {}",
                        layer,
                        qgate_dim,
                        2 * q_dim_u
                    );
                    if st.wq.quant == QuantScheme::Q4_0
                        && st.wq.length > 0
                        && st.wq.length as usize == qgate_dim * q4_row_bytes
                        && qgate_dim % 8 == 0
                    {
                        match lv.subtensor_bytes(&st.wq) {
                            Ok(src) => repack_q4::build_qmv_decode_buffers(
                                &self.device,
                                src,
                                qgate_dim,
                                hidden_dim_u,
                            )
                            .ok(),
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                match wq_built {
                    Some((qw, sc)) => {
                        wq_qw_vecs.push(Some(qw));
                        wq_sc_vecs.push(Some(sc));
                    }
                    None => {
                        wq_qw_vecs.push(None);
                        wq_sc_vecs.push(None);
                    }
                }

                // #3: Output projection (st.wo). in_dim = q_dim, out = hidden_dim.
                let wo_built = if q_dim_u % 512 == 0 && q4_row_bytes_qdim > 0 {
                    let wo_out = (st.wo.length as usize) / q4_row_bytes_qdim;
                    if st.wo.quant == QuantScheme::Q4_0
                        && st.wo.length > 0
                        && st.wo.length as usize == wo_out * q4_row_bytes_qdim
                        && wo_out % 8 == 0
                    {
                        match lv.subtensor_bytes(&st.wo) {
                            Ok(src) => repack_q4::build_qmv_decode_buffers(
                                &self.device,
                                src,
                                wo_out,
                                q_dim_u,
                            )
                            .ok(),
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                match wo_built {
                    Some((qw, sc)) => {
                        wo_qw_vecs.push(Some(qw));
                        wo_sc_vecs.push(Some(sc));
                    }
                    None => {
                        wo_qw_vecs.push(None);
                        wo_sc_vecs.push(None);
                    }
                }
            }
            let n_wq = wq_qw_vecs.iter().filter(|b| b.is_some()).count();
            let n_wo = wo_qw_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[qmv-proj] flag ON: built {} full-attn Q+gate + {} full-attn Wo qmv buffers (hidden={}, q_dim={})",
                n_wq, n_wo, hidden_dim_u, q_dim_u
            );
            s.qmv_attn_wq_qw = wq_qw_vecs;
            s.qmv_attn_wq_scales = wq_sc_vecs;
            s.qmv_attn_wo_qw = wo_qw_vecs;
            s.qmv_attn_wo_scales = wo_sc_vecs;

            // Persistent zero residual buffer for the Q+gate-fusion Wo path (out =
            // hidden_dim). Allocated from a zeroed Vec so its contents are guaranteed
            // 0.0 regardless of Metal's default-init behavior. Only when Wo qmv built.
            if n_wo > 0 {
                let zeros = vec![0u8; hidden_dim_u * 4]; // hidden_dim f32 zeros
                s.qmv_zero_residual_buf = self.device.new_buffer_with_bytes(&zeros);
                if s.qmv_zero_residual_buf.is_none() {
                    // Allocation failed -> drop Wo qmv so dispatch falls back to NR2.
                    eprintln!("[qmv-proj] WARN: zero residual alloc failed; disabling Wo qmv");
                    s.qmv_attn_wo_qw = Vec::new();
                    s.qmv_attn_wo_scales = Vec::new();
                }
            }
        }

        // MLX-style Q4 full-attn K/V decode-qmv repack (default). K (`st.wk`) and V
        // (`st.wv`) are Q4_0 [kv_dim, hidden_dim].
        // Both read the SAME pre-norm hidden as Q (rmsnorm-fused) -> the
        // `qmv_q4_0_rmsnorm` kernel (same as the Q+gate fast path). in_dim = hidden
        // (%512), out = kv_dim from byte length (%8). Indexed by `layer_idx`
        // (0..num_layers; None for GDN layers, matching the wq/wo convention).
        if crate::metal::q4_fast_decode_enabled() {
            let hidden_dim_u = s.hidden_dim;
            let num_layers = s.cached_layer_meta.len();
            let q4_row_bytes = (hidden_dim_u / 32) * 18; // Q4_0 row for in_dim = hidden
            let mut wk_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wk_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wv_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut wv_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            if hidden_dim_u % 512 == 0 && q4_row_bytes > 0 {
                for layer in 0..num_layers {
                    // GDN layers do not participate (None keeps layer_idx alignment).
                    if s.cached_layer_meta[layer].layer_type == Some(1) {
                        wk_qw_vecs.push(None);
                        wk_sc_vecs.push(None);
                        wv_qw_vecs.push(None);
                        wv_sc_vecs.push(None);
                        continue;
                    }
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("qmv-kv: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;

                    // K projection (st.wk): out = kv_dim from byte length, in = hidden.
                    let wk_built = if st.wk.quant == QuantScheme::Q4_0 && st.wk.length > 0 {
                        let out_dim = (st.wk.length as usize) / q4_row_bytes;
                        if st.wk.length as usize == out_dim * q4_row_bytes && out_dim % 8 == 0 {
                            match lv.subtensor_bytes(&st.wk) {
                                Ok(src) => repack_q4::build_qmv_decode_buffers(
                                    &self.device,
                                    src,
                                    out_dim,
                                    hidden_dim_u,
                                )
                                .ok(),
                                Err(_) => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    match wk_built {
                        Some((qw, sc)) => {
                            wk_qw_vecs.push(Some(qw));
                            wk_sc_vecs.push(Some(sc));
                        }
                        None => {
                            wk_qw_vecs.push(None);
                            wk_sc_vecs.push(None);
                        }
                    }

                    // V projection (st.wv): out = kv_dim from byte length, in = hidden.
                    let wv_built = if st.wv.quant == QuantScheme::Q4_0 && st.wv.length > 0 {
                        let out_dim = (st.wv.length as usize) / q4_row_bytes;
                        if st.wv.length as usize == out_dim * q4_row_bytes && out_dim % 8 == 0 {
                            match lv.subtensor_bytes(&st.wv) {
                                Ok(src) => repack_q4::build_qmv_decode_buffers(
                                    &self.device,
                                    src,
                                    out_dim,
                                    hidden_dim_u,
                                )
                                .ok(),
                                Err(_) => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    match wv_built {
                        Some((qw, sc)) => {
                            wv_qw_vecs.push(Some(qw));
                            wv_sc_vecs.push(Some(sc));
                        }
                        None => {
                            wv_qw_vecs.push(None);
                            wv_sc_vecs.push(None);
                        }
                    }
                }
            }
            let n_wk = wk_qw_vecs.iter().filter(|b| b.is_some()).count();
            let n_wv = wv_qw_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[qmv-kv] flag ON: built {} full-attn K + {} full-attn V qmv buffers (hidden={})",
                n_wk, n_wv, hidden_dim_u
            );
            s.qmv_attn_wk_qw = wk_qw_vecs;
            s.qmv_attn_wk_scales = wk_sc_vecs;
            s.qmv_attn_wv_qw = wv_qw_vecs;
            s.qmv_attn_wv_scales = wv_sc_vecs;
        }

        // MLX-style Q4 lm_head (output projection) decode-qmv repack (default).
        // MLX's 4-bit model quantizes
        // the lm_head to 4-bit; Lumen ships it as Q8_0 (~1080 MB, ~13% of decode).
        // This re-quantizes the Q8_0 output_proj -> Q4_0 at load time and builds
        // the GLOBAL (single, non-per-layer) sequential-nibble qweights + f32
        // scales for the `qmv_q4_0_rmsnorm` kernel (fused final-RMSNorm + matvec).
        // The decode lm_head dispatch picks these up when both buffers are Some.
        //
        // The Q8->Q4 re-quant is a deliberate precision tradeoff (matches MLX's
        // 4-bit lm_head; NOT byte-identical to the Q8 path). Guarded so it only
        // engages for a genuine, separate (non-weight-tied) Q8_0 output_proj whose
        // buffer length is consistent with [vocab, hidden] (in=hidden % 512 == 0,
        // out=vocab % 8 == 0). Any mismatch -> skip (existing Q8 lm_head path).
        if crate::metal::q4_fast_decode_enabled() {
            let hidden_dim_u = s.hidden_dim;
            let vocab = self.cached_vocab_size;
            // Q8_0 row bytes for in_dim = hidden (2B f16 scale + 32 i8 per 32-block).
            let q8_row_bytes = if hidden_dim_u % 32 == 0 {
                (hidden_dim_u / 32) * 34
            } else {
                0
            };
            let mut built = false;
            // F16-scales lm_head (default): re-quant the Q8
            // output_proj to Q4 but emit the per-block scale as f16 (2 B/block) for the
            // f16sc kernel. Only when the f16sc pipeline compiled; otherwise the f32
            // builder so the lm_head dispatch falls back cleanly to qmv_q4_0_rmsnorm.
            // The f16 scale produced by requant is byte-identical to the f32-widened one.
            let lmhead_f16sc = crate::metal::q4_lmhead_f16sc_enabled()
                && self
                    .pipelines
                    .as_ref()
                    .map(|p| p.qmv_q4_0_rmsnorm_f16sc.is_some())
                    .unwrap_or(false);
            // Require an untied Q8_0 output_proj: the lever is validated on
            // standalone Q8_0 lm_heads only, so tied models conservatively
            // skip. F16/Bf16/Q4 lm_heads also skip (Q8-only lever).
            if !self.weight_tying
                && self.output_proj_quant == QuantScheme::Q8_0
                && hidden_dim_u % 512 == 0
                && vocab % 8 == 0
                && q8_row_bytes > 0
            {
                if let Some(proj_buf) = self.output_proj_buf.as_ref() {
                    let buf_len = proj_buf.length() as usize;
                    let expected = vocab * q8_row_bytes;
                    // Strict consistency: the standalone shared output_proj buffer
                    // must be exactly [vocab, hidden] Q8_0. (Mirrors the per-layer
                    // qmv byte-length guards.) Mismatch -> fall back.
                    if buf_len == expected {
                        // SAFETY: `output_proj_buf` is StorageModeShared (built via
                        // new_buffer_with_bytes), so `contents()` is a valid CPU
                        // pointer to `buf_len` readable bytes for the lifetime of the
                        // buffer. We only READ here; the slice does not outlive this
                        // block. Same access pattern as the unified-blit above.
                        let q8_src: &[u8] = unsafe {
                            std::slice::from_raw_parts(proj_buf.contents() as *const u8, buf_len)
                        };
                        let lmhead_build = if lmhead_f16sc {
                            repack_q4::build_qmv_lmhead_buffers_from_q8_f16sc(
                                &self.device,
                                q8_src,
                                vocab,
                                hidden_dim_u,
                            )
                        } else {
                            repack_q4::build_qmv_lmhead_buffers_from_q8(
                                &self.device,
                                q8_src,
                                vocab,
                                hidden_dim_u,
                            )
                        };
                        match lmhead_build {
                            Ok((qw, sc)) => {
                                s.qmv_lmhead_qw = Some(qw);
                                s.qmv_lmhead_scales = Some(sc);
                                built = true;
                            }
                            Err(e) => {
                                eprintln!(
                                    "[qmv-lmhead] re-quant/build failed ({}); using Q8 lm_head",
                                    e
                                );
                            }
                        }
                    } else {
                        eprintln!(
                            "[qmv-lmhead] output_proj buf {} != expected {} (vocab={}, hidden={}); using Q8 lm_head",
                            buf_len, expected, vocab, hidden_dim_u
                        );
                    }
                }
            }
            eprintln!(
                "[qmv-lmhead] flag ON: Q4 lm_head {} (quant={:?}, tied={}, vocab={}, hidden={})",
                if built {
                    "built"
                } else {
                    "SKIPPED (fallback to Q8)"
                },
                self.output_proj_quant,
                self.weight_tying,
                vocab,
                hidden_dim_u
            );
        }

        // GDN ssm_out Q8_0 -> Q4_0 NATIVE-NR2 requant (env LUMEN_METAL_Q4_SSMOUT_NR2=1,
        // default OFF). On Qwen3.5-9B ssm_out ships Q8_0; this re-quantizes each GDN
        // layer's Q8_0 ssm_out weight to Q4_0 in the on-disk GGUF block layout and
        // stores it in a standalone per-GDN-layer buffer (`s.q4nr2_ssm_out`, indexed
        // by gdn_idx). The fused decode ssm_out dispatch (gdn.rs) then binds this Q4
        // buffer to buffer(0) and runs the EXISTING
        // `dequant_matmul_q4_0_silu_deferred_residual_copy_nr2` kernel (one fused
        // dispatch) instead of the Q8 fused kernel -> halves the ssm_out weight
        // stream (~0.21 -> ~0.11 GB/token over 24 layers) with NO extra dispatch.
        // Distinct from / supersedes the earlier qmv-layout requant (which built the
        // qmv layout + 3-dispatch path, measured FLAT). Precision tradeoff (Q8->Q4
        // requant), gated by the correctness harness.
        if crate::metal::q4_ssmout_nr2_enabled() {
            let hidden_dim_u = s.hidden_dim;
            // GDN value_dim = in_dim for ssm_out (= num_v_heads * head_dim).
            let gdn_value_dim = s.gdn_num_v_heads * s.gdn_head_dim;
            let n_gdn_layers = s
                .cached_layer_meta
                .iter()
                .filter(|m| m.layer_type == Some(1))
                .count();
            // Q8_0 bytes per output row for in_dim = value_dim (2 B f16 scale + 32 i8).
            let q8_row_bytes = (gdn_value_dim / 32) * 34;
            let mut nr2_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(n_gdn_layers);
            if gdn_value_dim % 32 == 0 && hidden_dim_u > 0 {
                for layer in 0..num_layers {
                    if s.cached_layer_meta[layer].layer_type != Some(1) {
                        continue;
                    }
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("ssmout-nr2: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;
                    // Requant only a genuine Q8_0 ssm_out of the expected byte length
                    // [hidden_dim, value_dim]; otherwise leave None (native-Q4 / other
                    // quants keep their existing fused path).
                    let built = match &st.ssm_out {
                        Some(so)
                            if so.quant == QuantScheme::Q8_0
                                && so.length > 0
                                && so.length as usize == hidden_dim_u * q8_row_bytes =>
                        {
                            match lv.subtensor_bytes(so) {
                                Ok(src) => repack_q4::build_nr2_q4_buffer_from_q8(
                                    &self.device,
                                    src,
                                    hidden_dim_u,
                                    gdn_value_dim,
                                )
                                .ok(),
                                Err(_) => None,
                            }
                        }
                        _ => None,
                    };
                    nr2_vecs.push(built);
                }
            }
            let nbuilt = nr2_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[ssmout-nr2] flag ON: built {}/{} GDN ssm_out Q8->Q4 NR2 buffers (in=value_dim={}, out=hidden={})",
                nbuilt, n_gdn_layers, gdn_value_dim, hidden_dim_u
            );
            s.q4nr2_ssm_out = nr2_vecs;
        }

        // MLX-style Q4_0 DENSE FFN gate/up dual-matrix decode-qmv repack (default).
        // Builds per-layer SEPARATE
        // sequential-nibble qweights + f32 scales for BOTH gate (`st.w_gate`)
        // and up (`st.w_up`), consumed together by the `qmv_q4_0_gate_up_swiglu`
        // dual-matrix kernel (fused RMSNorm + SwiGLU). Absent / partial => the
        // existing `rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row` fused path.
        //
        // gate/up are Q4_0 [inter_dim, hidden_dim]: out rows = inter_dim, in_dim =
        // hidden_dim. Requires hidden_dim % 512 == 0 (in_dim) and inter_dim % 8 == 0
        // (out_dim). Qwen3.5-9B dense FFN: in=hidden=4096, out=inter=12288 -> OK.
        //
        // Indexed by `layer_idx` (0..num_layers). MoE layers (whose w_gate/w_up are
        // zero-length sentinels) push None placeholders to keep the index aligned;
        // only genuine dense-FFN Q4_0 layers build buffers.
        // Build the separate gate/up qmv buffers consumed by the decode gate/up path.
        if crate::metal::q4_fast_decode_enabled() {
            let hidden_dim_u = s.hidden_dim;
            let inter_dim_u = s.inter_dim;
            let mut gate_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut gate_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut up_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut up_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            // Q4_0 bytes per output row for in_dim = hidden_dim.
            let q4_row_bytes = (hidden_dim_u / 32) * 18;
            let expected_len = inter_dim_u * q4_row_bytes;
            // F16-scales dense FFN gate/up (default): build
            // BOTH gate and up scale buffers as f16 (2 B/block) instead of f32 (4 B)
            // for the f16sc kernel. Only when the f16sc pipeline compiled; otherwise
            // stay on the f32 builder so the dispatch falls back cleanly to the
            // f32-scale qmv_q4_0_gate_up_swiglu. Byte-identical (f16 = on-disk native).
            let gateup_f16sc = crate::metal::q4_gateup_f16sc_enabled()
                && self
                    .pipelines
                    .as_ref()
                    .map(|p| p.qmv_q4_0_gate_up_swiglu_f16sc.is_some())
                    .unwrap_or(false);
            for layer in 0..num_layers {
                // Build only when BOTH gate and up are genuine Q4_0 [inter_dim, hidden]
                // tensors AND the shape constraints hold; any miss -> None pair so the
                // dispatch falls back to the 8row path for this layer.
                // NOTE: include BOTH full-attn (type 0) AND GDN (type 1) layers —
                // the dense FFN gate/up is identical Q4_0 [inter_dim, hidden_dim]
                // for every layer and flows through the SAME decode FFN dispatch
                // (the layer's attention type is irrelevant to its FFN). The
                // quant + length checks below reject MoE zero-length sentinels.
                let pair = if hidden_dim_u % 512 == 0 && inter_dim_u % 8 == 0 && q4_row_bytes > 0 {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("qmv-gateup: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;
                    if st.w_gate.quant == QuantScheme::Q4_0
                        && st.w_up.quant == QuantScheme::Q4_0
                        && st.w_gate.length as usize == expected_len
                        && st.w_up.length as usize == expected_len
                    {
                        let g = lv.subtensor_bytes(&st.w_gate).ok().and_then(|src| {
                            if gateup_f16sc {
                                repack_q4::build_qmv_decode_buffers_f16sc(
                                    &self.device,
                                    src,
                                    inter_dim_u,
                                    hidden_dim_u,
                                )
                                .ok()
                            } else {
                                repack_q4::build_qmv_decode_buffers(
                                    &self.device,
                                    src,
                                    inter_dim_u,
                                    hidden_dim_u,
                                )
                                .ok()
                            }
                        });
                        let u = lv.subtensor_bytes(&st.w_up).ok().and_then(|src| {
                            if gateup_f16sc {
                                repack_q4::build_qmv_decode_buffers_f16sc(
                                    &self.device,
                                    src,
                                    inter_dim_u,
                                    hidden_dim_u,
                                )
                                .ok()
                            } else {
                                repack_q4::build_qmv_decode_buffers(
                                    &self.device,
                                    src,
                                    inter_dim_u,
                                    hidden_dim_u,
                                )
                                .ok()
                            }
                        });
                        match (g, u) {
                            (Some(g), Some(u)) => Some((g, u)),
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                match pair {
                    Some(((gqw, gsc), (uqw, usc))) => {
                        gate_qw_vecs.push(Some(gqw));
                        gate_sc_vecs.push(Some(gsc));
                        up_qw_vecs.push(Some(uqw));
                        up_sc_vecs.push(Some(usc));
                    }
                    None => {
                        gate_qw_vecs.push(None);
                        gate_sc_vecs.push(None);
                        up_qw_vecs.push(None);
                        up_sc_vecs.push(None);
                    }
                }
            }
            let nbuilt = gate_qw_vecs.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[qmv-gateup] flag ON: built {}/{} dense FFN gate/up qmv pairs (in=hidden={}, out=inter={})",
                nbuilt, num_layers, hidden_dim_u, inter_dim_u
            );
            s.qmv_ffn_gate_qw = gate_qw_vecs;
            s.qmv_ffn_gate_scales = gate_sc_vecs;
            s.qmv_ffn_up_qw = up_qw_vecs;
            s.qmv_ffn_up_scales = up_sc_vecs;
        }

        // LM-head-structure (LS) dense FFN gate/up: build ONE ROW-INTERLEAVED
        // packed nibble buffer + ONE row-interleaved packed f16-scale buffer per
        // layer (physical row 2d = whole gate row d, 2d+1 = whole up row d) for the
        // single-stream kernel `qmv_q4_0_gate_up_swiglu_ls_h2math` (2*inter_dim/8
        // TGs). Same correctness gate as the f16sc/IL builds above; requires
        // in_dim(hidden) % 512 == 0 and inter_dim % 4 == 0. Only when the LS
        // pipeline compiled; any miss -> None pair so the dispatch falls back to
        // the h2math/default path. Byte-identical.
        if self
            .pipelines
            .as_ref()
            .map(|p| p.qmv_q4_0_gate_up_swiglu_ls_h2math.is_some())
            .unwrap_or(false)
        {
            let hidden_dim_u = s.hidden_dim;
            let inter_dim_u = s.inter_dim;
            let q4_row_bytes = (hidden_dim_u / 32) * 18;
            let expected_len = inter_dim_u * q4_row_bytes;
            let mut ls_qw_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            let mut ls_sc_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let built = if hidden_dim_u % 512 == 0 && inter_dim_u % 4 == 0 && q4_row_bytes > 0 {
                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!("qmv-gateup-ls: layer {}: {}", layer, e))
                    })?;
                    let st = &lv.subtensors;
                    if st.w_gate.quant == QuantScheme::Q4_0
                        && st.w_up.quant == QuantScheme::Q4_0
                        && st.w_gate.length as usize == expected_len
                        && st.w_up.length as usize == expected_len
                    {
                        match (lv.subtensor_bytes(&st.w_gate), lv.subtensor_bytes(&st.w_up)) {
                            (Ok(gsrc), Ok(usrc)) => repack_q4::build_qmv_gate_up_ls_buffers(
                                &self.device,
                                gsrc,
                                usrc,
                                inter_dim_u,
                                hidden_dim_u,
                            )
                            .ok(),
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                match built {
                    Some((qw, sc)) => {
                        ls_qw_vecs.push(Some(qw));
                        ls_sc_vecs.push(Some(sc));
                    }
                    None => {
                        ls_qw_vecs.push(None);
                        ls_sc_vecs.push(None);
                    }
                }
            }
            s.qmv_ffn_gate_up_ls_qw = ls_qw_vecs;
            s.qmv_ffn_gate_up_ls_scales = ls_sc_vecs;
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
                let n_gdn_layers = s
                    .cached_layer_meta
                    .iter()
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
                        Some(_) => {}
                        None => {
                            qkv_gate_vecs.push(None);
                            qkv_gate_shapes.push(None);
                            continue;
                        }
                    };
                    let _ = attn_gate_off;

                    let lv = weights.get_layer_raw(layer).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " BF16 GDN paired repack: failed to get layer {}: {}",
                            layer, e
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
                    let row_bytes = hidden_dim_u.checked_mul(2).ok_or_else(|| {
                        RuntimeError::Compute(" BF16 repack: hidden_dim * 2 overflow".into())
                    })?;
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
                            " BF16 repack: failed to read wq at layer {}: {}",
                            layer, e
                        ))
                    })?;
                    let src_gate = lv.subtensor_bytes(attn_gate_st).map_err(|e| {
                        RuntimeError::Compute(format!(
                            " BF16 repack: failed to read attn_gate at layer {}: {}",
                            layer, e
                        ))
                    })?;

                    let buf = super::repack_bf16::build_repacked_buffer_qkv_gate(
                        &self.device,
                        src_qkv,
                        src_gate,
                        qkv_n,
                        gate_n,
                        hidden_dim_u,
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
                    RuntimeError::Compute("scratch unexpectedly None at warmup time".into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_format::index::{SubtensorOffsets, TensorSlice};

    fn slice(quant: QuantScheme) -> TensorSlice {
        TensorSlice {
            offset: 0,
            length: 1024,
            quant,
        }
    }

    fn gdn_layer(gates: QuantScheme) -> SubtensorOffsets {
        SubtensorOffsets {
            wq: slice(QuantScheme::Q8_0),
            wk: slice(QuantScheme::Q8_0),
            wv: slice(QuantScheme::Q8_0),
            wo: slice(QuantScheme::Q8_0),
            bq: None,
            bk: None,
            bv: None,
            w_gate: slice(QuantScheme::Q8_0),
            w_up: slice(QuantScheme::Q8_0),
            w_down: slice(QuantScheme::Q8_0),
            attn_norm: slice(QuantScheme::F32),
            ffn_norm: slice(QuantScheme::F32),
            router_weight: None,
            experts: None,
            shared_expert_gate: None,
            shared_expert_up: None,
            shared_expert_down: None,
            attn_gate: Some(slice(QuantScheme::Q8_0)),
            attn_post_norm: None,
            ssm_a: Some(slice(QuantScheme::F32)),
            ssm_conv1d: Some(slice(QuantScheme::F32)),
            ssm_dt: Some(slice(QuantScheme::F32)),
            ssm_beta: Some(slice(gates)),
            ssm_alpha: Some(slice(gates)),
            ssm_norm: Some(slice(QuantScheme::F32)),
            ssm_out: Some(slice(QuantScheme::Q8_0)),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn_gate_inp_shexp: None,
            layer_type: Some(1),
        }
    }

    #[test]
    fn q8_gates_accepted() {
        assert!(validate_layer_quants(0, &gdn_layer(QuantScheme::Q8_0)).is_ok());
    }

    #[test]
    fn f32_gates_rejected() {
        let err = validate_layer_quants(0, &gdn_layer(QuantScheme::F32)).unwrap_err();
        assert!(err.to_string().contains("ssm_alpha is F32"), "{err}");
    }

    #[test]
    fn gate_up_mismatch_rejected() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.w_up = slice(QuantScheme::Q4_0);
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(err.to_string().contains("ffn_gate is Q8_0"), "{err}");
    }

    #[test]
    fn gdn_qkv_gate_split_rejected_both_directions() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.attn_gate = Some(slice(QuantScheme::Q4_0));
        assert!(validate_layer_quants(0, &st).is_err());
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.wq = slice(QuantScheme::Q4_0);
        st.w_gate = slice(QuantScheme::Q4_0);
        st.w_up = slice(QuantScheme::Q4_0);
        st.w_down = slice(QuantScheme::Q4_0);
        assert!(validate_layer_quants(0, &st).is_err());
    }

    #[test]
    fn full_attention_gate_split_exempt() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = Some(0);
        st.attn_gate = Some(slice(QuantScheme::Q4_0));
        st.ssm_alpha = None;
        st.ssm_beta = None;
        assert!(validate_layer_quants(0, &st).is_ok());
    }

    #[test]
    fn expert_bank_nonuniform_rejected() {
        use lumen_format::index::ExpertSlice;
        let e = |q: QuantScheme| ExpertSlice {
            gate: slice(q),
            up: slice(q),
            down: slice(q),
        };
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = None;
        st.attn_gate = None;
        st.ssm_alpha = None;
        st.ssm_beta = None;
        st.experts = Some(vec![e(QuantScheme::Q4_0), e(QuantScheme::Q4_0)]);
        assert!(validate_layer_quants(0, &st).is_ok());
        st.experts = Some(vec![e(QuantScheme::Q4_0), e(QuantScheme::Q8_0)]);
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(err.to_string().contains("expert 0's"), "{err}");
    }

    #[test]
    fn shared_expert_pair_and_quant_rejected() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = None;
        st.attn_gate = None;
        st.ssm_alpha = None;
        st.ssm_beta = None;
        st.shared_expert_gate = Some(slice(QuantScheme::Q4_0));
        st.shared_expert_up = Some(slice(QuantScheme::Q4_0));
        st.shared_expert_down = Some(slice(QuantScheme::Q4_0));
        assert!(validate_layer_quants(0, &st).is_ok());
        st.shared_expert_up = Some(slice(QuantScheme::Q8_0));
        assert!(validate_layer_quants(0, &st).is_err());
        st.shared_expert_gate = Some(slice(QuantScheme::F16));
        st.shared_expert_up = Some(slice(QuantScheme::F16));
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(err.to_string().contains("fused kernels only"), "{err}");
    }

    #[test]
    fn f32_expert_bank_accepted() {
        // F16/Bf16/F32 expert banks are legitimately served: the batched
        // dispatch's quant gate routes them to the legacy per-expert path,
        // which has real float arms. Only pair/uniformity divergence and the
        // schemes the dense allowlist rejects are refused.
        use lumen_format::index::ExpertSlice;
        let e = |q: QuantScheme| ExpertSlice {
            gate: slice(q),
            up: slice(q),
            down: slice(q),
        };
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = None;
        st.attn_gate = None;
        st.ssm_alpha = None;
        st.ssm_beta = None;
        st.experts = Some(vec![e(QuantScheme::F32), e(QuantScheme::F32)]);
        assert!(validate_layer_quants(0, &st).is_ok());
        st.experts = Some(vec![e(QuantScheme::Bf16), e(QuantScheme::Bf16)]);
        assert!(validate_layer_quants(0, &st).is_ok());
    }

    #[test]
    fn expert_gate_up_pair_mismatch_rejected() {
        use lumen_format::index::ExpertSlice;
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = None;
        st.attn_gate = None;
        st.ssm_alpha = None;
        st.ssm_beta = None;
        st.experts = Some(vec![ExpertSlice {
            gate: slice(QuantScheme::Q4_0),
            up: slice(QuantScheme::Q8_0),
            down: slice(QuantScheme::Q4_0),
        }]);
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(err.to_string().contains("expert 0: gate is Q4_0"), "{err}");
    }

    #[test]
    fn incomplete_shared_expert_rejected() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.layer_type = None;
        st.attn_gate = None;
        st.ssm_alpha = None;
        st.ssm_beta = None;
        st.shared_expert_gate = Some(slice(QuantScheme::Q4_0));
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(
            err.to_string().contains("incomplete shared-expert"),
            "{err}"
        );
    }

    #[test]
    fn non_f32_norm_rejected() {
        let setters: [(&str, fn(&mut SubtensorOffsets, TensorSlice)); 11] = [
            ("attn_norm", |st, t| st.attn_norm = t),
            ("ffn_norm", |st, t| st.ffn_norm = t),
            ("attn_post_norm", |st, t| st.attn_post_norm = Some(t)),
            ("attn_q_norm", |st, t| st.attn_q_norm = Some(t)),
            ("attn_k_norm", |st, t| st.attn_k_norm = Some(t)),
            ("ssm_norm", |st, t| st.ssm_norm = Some(t)),
            ("ssm_a", |st, t| st.ssm_a = Some(t)),
            ("ssm_conv1d", |st, t| st.ssm_conv1d = Some(t)),
            ("ssm_dt", |st, t| st.ssm_dt = Some(t)),
            ("router_weight", |st, t| st.router_weight = Some(t)),
            ("ffn_gate_inp_shexp", |st, t| {
                st.ffn_gate_inp_shexp = Some(t)
            }),
        ];
        for (name, set) in setters {
            let mut st = gdn_layer(QuantScheme::Q8_0);
            set(&mut st, slice(QuantScheme::F16));
            let err = validate_layer_quants(0, &st).expect_err(&format!("{name} must require F32"));
            assert!(
                err.to_string().contains(&format!("{name} is F16")),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn unsupported_dense_quant_rejected() {
        let mut st = gdn_layer(QuantScheme::Q8_0);
        st.w_down = slice(QuantScheme::Q6_K);
        let err = validate_layer_quants(0, &st).unwrap_err();
        assert!(err.to_string().contains("w_down"), "{err}");
    }

    #[test]
    fn dense_slice_quant_allowlist() {
        for q in [
            QuantScheme::F32,
            QuantScheme::F16,
            QuantScheme::Bf16,
            QuantScheme::Q8_0,
            QuantScheme::Q4_0,
        ] {
            assert!(dense_slice_quant_supported(q), "{q:?} must be servable");
        }
        // Q4_1 has MoE-expert kernels only; dense dispatch would misread it.
        for q in [
            QuantScheme::Q4_1,
            QuantScheme::Q5_0,
            QuantScheme::Q4_K,
            QuantScheme::Q5_K,
            QuantScheme::Q6_K,
            QuantScheme::Q2_K,
            QuantScheme::Q3_K,
            QuantScheme::CtInt4G32,
        ] {
            assert!(!dense_slice_quant_supported(q), "{q:?} must be rejected");
        }
    }
}
