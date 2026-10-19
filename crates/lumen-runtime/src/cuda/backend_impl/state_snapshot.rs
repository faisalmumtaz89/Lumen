//! Copy the state a decode continues from out of the backend, and back in
//! (feature `test-state-snapshot`).
//!
//! Two builds can then decode from the same state, so a change to how a
//! prompt is prefilled can be kept apart from its effect on decode. The state
//! is everything a decode step reads that an earlier step or prefill wrote:
//! each attention layer's KV rows `[0, seq_len)`, each GDN layer's recurrent
//! state, conv ring and ring position, the last hidden row, the decode step
//! counter, and the KV length the host cache records.

use super::{CudaBackend, KvStore};
use crate::error::RuntimeError;
use crate::kv::KvCache;

/// The optional parts of the prefill workspace a poisoning filled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoisonedParts {
    /// The tiled attention's score block.
    pub score_block: bool,
    /// The fused Q+gate pair.
    pub q_gate: bool,
    /// The grouped MoE prefill scratch.
    pub grouped_moe: bool,
}

/// Decode state copied to the host. Vectors are in layer order.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSnapshot {
    /// Tokens held by every attention layer's cache and by the host cache.
    pub seq_len: usize,
    /// K and V of each attention layer, `[kv_heads][seq_len][head_dim]`.
    pub kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Recurrent state of each GDN layer.
    pub h_states: Vec<Vec<f32>>,
    /// Conv ring of each GDN layer.
    pub conv_states: Vec<Vec<f32>>,
    /// Ring write position of each GDN layer.
    pub conv_positions: Vec<u32>,
    /// The last hidden row a prefill or decode step left.
    pub x: Vec<f32>,
    /// Decode steps taken since the last reset.
    pub decode_token_count: usize,
}

impl CudaBackend {
    /// Copy the decode state out. `kv` is the host cache the backend was
    /// driven with; its length must equal every device cache's.
    pub fn snapshot_state(&self, kv: &KvCache) -> Result<StateSnapshot, RuntimeError> {
        let mut guard = self.state.lock().unwrap();
        let st = guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;
        if st.has_gdn_layers {
            self.ensure_gdn_scratch(st)?;
        }
        self.device.synchronize()?;
        let seq_len = kv.seq_len();
        let mut kv_rows = Vec::new();
        for cache in st.kv_caches.iter().flatten() {
            if cache.seq_len() != seq_len {
                return Err(RuntimeError::KvCache(format!(
                    "device KV holds {} tokens, host KV {seq_len}",
                    cache.seq_len()
                )));
            }
            let KvStore::F32 { k, v } = &cache.store else {
                return Err(RuntimeError::Unsupported(
                    "state snapshots cover an F32 KV store only".into(),
                ));
            };
            let row = cache.head_dim;
            let head_stride = cache.max_seq_len * row;
            let live = |buf: &[f32]| -> Vec<f32> {
                (0..cache.num_kv_heads)
                    .flat_map(|h| &buf[h * head_stride..h * head_stride + seq_len * row])
                    .copied()
                    .collect()
            };
            kv_rows.push((
                live(&self.device.dtoh_copy(k)?),
                live(&self.device.dtoh_copy(v)?),
            ));
        }
        let (h_states, conv_states, conv_positions) = match &st.gdn_scratch_gpu {
            Some(gdn) => (
                gdn.h_states
                    .iter()
                    .map(|h| self.device.dtoh_copy(h))
                    .collect::<Result<_, _>>()?,
                gdn.conv_states
                    .iter()
                    .map(|c| self.device.dtoh_copy(c))
                    .collect::<Result<_, _>>()?,
                gdn.conv_positions.clone(),
            ),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };
        Ok(StateSnapshot {
            seq_len,
            kv: kv_rows,
            h_states,
            conv_states,
            conv_positions,
            x: self.device.dtoh_copy(&st.scratch.x_gpu)?,
            decode_token_count: st.decode_token_count,
        })
    }

    /// Write a snapshot taken from a backend of the same model and context
    /// back in, and bring the empty host cache `kv` to its length. Every
    /// buffer's size and every ring position is checked before anything is
    /// written.
    pub fn restore_state(
        &self,
        snap: &StateSnapshot,
        kv: &mut KvCache,
    ) -> Result<(), RuntimeError> {
        if kv.seq_len() != 0 || kv.max_seq_len() < snap.seq_len {
            return Err(RuntimeError::KvCache(format!(
                "restore needs an empty host KV cache of at least {} tokens, it holds {} of {}",
                snap.seq_len,
                kv.seq_len(),
                kv.max_seq_len()
            )));
        }
        let mut guard = self.state.lock().unwrap();
        let st = guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;
        if st.has_gdn_layers {
            self.ensure_gdn_scratch(st)?;
        }
        let mismatch = |what: &str, want: usize, got: usize| {
            Err(RuntimeError::Compute(format!(
                "snapshot {what}: backend has {want}, snapshot {got}"
            )))
        };
        let caches: Vec<_> = st.kv_caches.iter_mut().flatten().collect();
        if caches.len() != snap.kv.len() {
            return mismatch("attention layers", caches.len(), snap.kv.len());
        }
        for (cache, (k, v)) in caches.iter().zip(&snap.kv) {
            if snap.seq_len > cache.max_seq_len {
                return mismatch("KV capacity", cache.max_seq_len, snap.seq_len);
            }
            let rows = cache.num_kv_heads * snap.seq_len * cache.head_dim;
            if k.len() != rows || v.len() != rows {
                return mismatch("KV rows", rows, k.len().max(v.len()));
            }
            if !matches!(cache.store, KvStore::F32 { .. }) {
                return Err(RuntimeError::Unsupported(
                    "state snapshots cover an F32 KV store only".into(),
                ));
            }
        }
        match &st.gdn_scratch_gpu {
            Some(gdn) => {
                if gdn.h_states.len() != snap.h_states.len()
                    || gdn.conv_states.len() != snap.conv_states.len()
                    || gdn.conv_positions.len() != snap.conv_positions.len()
                {
                    return mismatch("GDN layers", gdn.h_states.len(), snap.h_states.len());
                }
                for (dev, host) in gdn.h_states.iter().zip(&snap.h_states) {
                    if dev.len() != host.len() {
                        return mismatch("GDN state", dev.len(), host.len());
                    }
                }
                for (dev, host) in gdn.conv_states.iter().zip(&snap.conv_states) {
                    if dev.len() != host.len() {
                        return mismatch("conv ring", dev.len(), host.len());
                    }
                }
                // The conv kernel indexes the ring with the position unreduced.
                let slots = gdn.params.conv_kernel_size - 1;
                if let Some(&p) = snap.conv_positions.iter().find(|&&p| p as usize >= slots) {
                    return mismatch("conv ring slots", slots, p as usize + 1);
                }
            }
            None if !snap.h_states.is_empty() => {
                return mismatch("GDN layers", 0, snap.h_states.len());
            }
            None => {}
        }
        if st.scratch.x_gpu.len() != snap.x.len() {
            return mismatch("hidden row", st.scratch.x_gpu.len(), snap.x.len());
        }

        for (cache, (k, v)) in caches.into_iter().zip(&snap.kv) {
            let row = snap.seq_len * cache.head_dim;
            let head_stride = cache.max_seq_len * cache.head_dim;
            let KvStore::F32 { k: dk, v: dv } = &mut cache.store else {
                unreachable!("checked above");
            };
            for h in 0..cache.num_kv_heads {
                let at = h * head_stride;
                self.device
                    .stream
                    .memcpy_htod(&k[h * row..(h + 1) * row], &mut dk.slice_mut(at..at + row))
                    .map_err(|e| RuntimeError::Compute(format!("restore K: {e}")))?;
                self.device
                    .stream
                    .memcpy_htod(&v[h * row..(h + 1) * row], &mut dv.slice_mut(at..at + row))
                    .map_err(|e| RuntimeError::Compute(format!("restore V: {e}")))?;
            }
            cache.reset();
            cache.advance_seq_len_by(snap.seq_len);
        }
        if let Some(gdn) = st.gdn_scratch_gpu.as_mut() {
            for (dev, host) in gdn.h_states.iter_mut().zip(&snap.h_states) {
                self.device.htod_copy_into(host, dev)?;
            }
            for (dev, host) in gdn.conv_states.iter_mut().zip(&snap.conv_states) {
                self.device.htod_copy_into(host, dev)?;
            }
            gdn.conv_positions.clone_from(&snap.conv_positions);
        }
        self.device.htod_copy_into(&snap.x, &mut st.scratch.x_gpu)?;
        st.decode_token_count = snap.decode_token_count;
        self.device.synchronize()?;
        for _ in 0..snap.seq_len {
            kv.advance_seq_len()?;
        }
        Ok(())
    }

    /// Fill every region of the retained F32 prefill workspace and every buffer
    /// of the grouped MoE prefill scratch with 0xFF bytes (NaN as F32), so a
    /// test can show no request reads a row of them that the request did not
    /// write first. Returns which of the parts a model may not have were there
    /// to fill.
    pub fn poison_prefill_workspace(&self) -> Result<PoisonedParts, RuntimeError> {
        use super::super::workspace::{Elem, Role};
        use cudarc::driver::{DevicePtr, DeviceSlice};
        let mut guard = self.state.lock().unwrap();
        let st = guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("CUDA backend not initialized".into()))?;
        let ws = st
            .prefill_workspace
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("no prefill workspace".into()))?;
        let stream = &self.device.stream;
        let fill = |ptr: u64, bytes: usize| -> Result<(), RuntimeError> {
            // SAFETY: `ptr` addresses a live workspace region or buffer of `bytes` bytes.
            unsafe { cudarc::driver::result::memset_d8_async(ptr, 0xFF, bytes, stream.cu_stream()) }
                .map_err(|e| RuntimeError::Compute(format!("poison: {e}")))
        };
        macro_rules! poison {
            ($($buf:expr),* $(,)?) => {
                $(fill($buf.device_ptr(stream).0, $buf.num_bytes())?;)*
            };
        }
        for role in Role::ALL {
            if !ws.has(role) {
                continue;
            }
            let [mut region] = ws.regions([role]);
            match role.elem() {
                Elem::F32 => {
                    let view = region.f32();
                    poison!(view);
                }
                Elem::U32 => {
                    let view = region.u32();
                    poison!(view);
                }
                Elem::U8 => {
                    let view = region.u8();
                    poison!(view);
                }
            }
        }
        let mut parts = PoisonedParts {
            q_gate: ws.has(Role::QGateFused),
            score_block: ws.has(Role::Scores),
            ..PoisonedParts::default()
        };
        if let Some(m) = st
            .moe_scratch
            .as_ref()
            .and_then(|m| m.prefill_grouped.as_ref())
        {
            poison!(
                m.swiglu_compact,
                m.down_compact,
                m.col_expert,
                m.col_src_tok,
                m.dst_to_col,
                m.expert_bounds,
                m.gate_up_tiles16,
                m.router_logits_batched,
                m.expert_ids_batched,
                m.expert_weights_batched,
                m.shared_swiglu_batched,
                m.shared_logit_batched,
            );
            for q in m.w10_xq_q.iter() {
                poison!(q);
            }
            for d in m.w10_xq_d.iter() {
                poison!(d);
            }
            for t in m.w10_tiles.iter() {
                poison!(t);
            }
            parts.grouped_moe = true;
        }
        self.device.synchronize()?;
        Ok(parts)
    }
}
