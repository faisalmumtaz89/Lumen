//! GPU-resident weight preloading for Metal backend.
//!
//! Packs all layer weight data and global tensors into a single contiguous
//! `StorageModePrivate` Metal buffer, eliminating TLB misses, reducing virtual
//! address ranges, and enabling GPU memory controller optimizations.

use super::ffi::MetalBuffer;
use super::types::{CachedLayerMeta, CachedMoeMeta};
use super::{MetalF32Backend, PAGE_SIZE};
use crate::error::RuntimeError;

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

        println!("Pre-loading {} layers into single private GPU buffer...", num_layers);

        // === Pass 1: Collect layer blobs and compute page-aligned offsets ===
        let align = |size: usize| -> usize { (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1) };

        let mut layer_blobs: Vec<Vec<u8>> = Vec::with_capacity(num_layers);
        let mut layer_offsets: Vec<usize> = Vec::with_capacity(num_layers);
        let mut layer_metas: Vec<CachedLayerMeta> = Vec::with_capacity(num_layers);
        let mut cursor: usize = 0;
        let mut gdn_layer_counter: usize = 0;

        for layer in 0..num_layers {
            let layer_view = weights.get_layer_blocking(layer).map_err(|e| {
                RuntimeError::Compute(format!(
                    "Failed to get layer {} for GPU-resident loading: {}", layer, e
                ))
            })?;
            let blob = layer_view.as_bytes();
            let base = cursor as u64;
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
            layer_offsets.push(cursor);
            layer_blobs.push(blob.to_vec());
            cursor = align(cursor + blob.len());
        }

        // Append global tensors at page-aligned offsets.
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
        let global_bytes = embed_len + norm_len + effective_proj_len;
        // Include globals in the unified private buffer.
        let include_globals = true;

        let (embed_offset, norm_offset, proj_offset) = if include_globals {
            let eo = cursor;
            cursor = align(cursor + embed_len);
            let no = cursor;
            cursor = align(cursor + norm_len);
            if self.weight_tying {
                // output_proj reuses embedding offset
                (eo, no, eo)
            } else {
                let po = cursor;
                cursor = align(cursor + proj_len);
                (eo, no, po)
            }
        } else {
            (0, 0, 0)
        };

        let total_size = cursor;

        // === Pass 2: Allocate shared staging buffer and copy all data via CPU ===
        let staging_buf = self.device.new_buffer(total_size).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "Failed to allocate staging buffer ({} bytes, {:.1} MB)",
                total_size, total_size as f64 / (1024.0 * 1024.0)
            ))
        })?;

        let dst_base = staging_buf.contents() as *mut u8;
        let mut layer_bytes_total: usize = 0;

        for (layer, blob) in layer_blobs.iter().enumerate() {
            let off = layer_offsets[layer];
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), dst_base.add(off), blob.len());
            }
            layer_bytes_total += blob.len();
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
        if include_globals {
            println!(
                "GPU-resident private buffer: {} layers ({:.1} MB) + globals ({:.1} MB) = {:.1} MB (StorageModePrivate)",
                num_layers, layer_mb, global_mb, total_mb,
            );
        } else {
            println!(
                "GPU-resident private buffer: {} layers ({:.1} MB) = {:.1} MB (StorageModePrivate); globals ({:.1} MB) in shared buffers",
                num_layers, layer_mb, total_mb,
                global_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        s.gpu_unified_weight_buf = Some(private_buf);
        s.gpu_layer_offsets = layer_offsets;
        if include_globals {
            s.gpu_global_offsets = Some((embed_offset, norm_offset, proj_offset));
        } else {
            s.gpu_global_offsets = None;  // Forces fallback to separate shared buffers
        }
        s.cached_layer_meta = layer_metas;

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
                println!(
                    "Qwen3.5 hybrid detected: {} layers ({} linear_attn, {} full_attn), {} MoE, {} shared_expert, rotary_dim={}, se_inter={}",
                    num_layers, n_linear, n_full, n_moe, n_shared, s.rotary_dim, se_inter_display,
                );

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
                println!(
                    "GDN state: {} layers, h_state={:.1} MB ({} heads x {}x{} per layer), conv={:.1} MB (kernel_size={})",
                    n_linear, h_state_mb, GDN_NUM_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM, conv_mb, conv_kernel_size,
                );
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

        Ok(())
    }
}
