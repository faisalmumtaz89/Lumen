//! `ComputeBackend` trait implementation for `MetalF32Backend`.
//!
//! Extracted from mod.rs to reduce file size. Contains `init`, `compute_layer`,
//! `compute_final`, and `embed_token` — the four required trait methods.

use crate::compute::{ActivationBuffer, BackendCaps, ComputeBackend, ComputeDtype, Logits};
use crate::error::RuntimeError;
use crate::expert::profiler::ExpertActivationProfiler;
use crate::expert::reader::ExpertReader;
use crate::kv::{KvCache, KvCacheView};
use crate::weight::cache::{LayerView, WeightProvider};
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

use super::ffi::{MetalBuffer, MTLSize};
use super::types::{CachedLayerMeta, CachedMoeMeta, MetalScratch};
use super::{MetalF32Backend, PrefetchState, PAGE_SIZE};
impl ComputeBackend for MetalF32Backend {
    fn init(&mut self, hyperparams: &ModelHyperparams) -> Result<(), RuntimeError> {
        self.cached_hidden_dim = hyperparams.hidden_dim as usize;
        self.cached_vocab_size = hyperparams.vocab_size as usize;

        // Compile shader pipelines
        let pipelines = self.compile_pipelines()?;

        // Determine threadgroup sizes based on pipeline capabilities
        let matmul_tg_size = pipelines.matmul_bytes_f32
            .max_total_threads_per_threadgroup()
            .min(256); // cap at 256 for matmul (good balance)
        let norm_tg_size = pipelines.rmsnorm_bytes
            .max_total_threads_per_threadgroup()
            .min(256);
        let mha_tg_size = pipelines.multi_head_attention
            .max_total_threads_per_threadgroup()
            .min(256); // threads per head in multi_head_attention

        self.pipelines = Some(pipelines);

        // Upload global tensors to GPU
        // Upload embedding: use raw quantized bytes if available, else F32
        if let Some(ref raw) = self.embedding_raw {
            if matches!(self.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                self.embedding_buf = Some(
                    self.device.new_buffer_with_bytes(raw).ok_or_else(|| {
                        RuntimeError::Compute("Failed to create quantized embedding buffer".into())
                    })?
                );
            } else {
                self.embedding_buf = Some(self.upload_f32(&self.embedding)?);
            }
        } else {
            self.embedding_buf = Some(self.upload_f32(&self.embedding)?);
        }
        self.final_norm_buf = Some(self.upload_f32(&self.final_norm)?);
        // Upload output_proj: use raw Q8_0 bytes if available, else F32
        if let Some(ref raw) = self.output_proj_raw {
            if matches!(self.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                self.output_proj_buf = Some(
                    self.device.new_buffer_with_bytes(raw).ok_or_else(|| {
                        RuntimeError::Compute("Failed to create Q8_0 output_proj buffer".into())
                    })?
                );
            } else {
                self.output_proj_buf = Some(self.upload_f32(&self.output_proj)?);
            }
        } else {
            self.output_proj_buf = Some(self.upload_f32(&self.output_proj)?);
        }

        // Compute dimensions
        let hidden_dim = hyperparams.hidden_dim as usize;
        let num_heads = hyperparams.num_heads as usize;
        let num_kv_heads = hyperparams.num_kv_heads as usize;
        let num_layers = hyperparams.num_layers as usize;
        let head_dim = hyperparams.head_dim as usize;
        let inter_dim = hyperparams.intermediate_dim as usize;
        let vocab_size = hyperparams.vocab_size as usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Use the max_seq_len from hyperparams. The caller (engine / CLI) is
        // responsible for capping this to a reasonable value via --context-len.
        // Models with very large context windows (e.g. 128K) allocate massive
        // KV caches that can devastate GPU performance (8+ GB KV = 5-7x slower
        // due to GPU memory pressure / TLB thrashing).
        let max_seq_len = hyperparams.max_seq_len as usize;
        let qkv_dim = q_dim + 2 * kv_dim;
        let half_dim = head_dim / 2;
        let gqa_ratio = num_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        // MoE dimensions (0 for dense models)
        let moe_num_experts = hyperparams.num_experts.unwrap_or(0) as usize;
        let moe_num_active_experts = hyperparams.num_active_experts.unwrap_or(0) as usize;
        // For MoE models, inter_dim is the per-expert intermediate dimension
        // (same as dense inter_dim for uniform-expert architectures like Mixtral).
        let moe_expert_inter_dim = if moe_num_experts > 0 { inter_dim } else { 0 };

        // Pre-compute RoPE tables and upload to GPU (f64 precision for intermediate math,
        // stored as f32. Matches the Qwen3.5 path in gpu_resident.rs and avoids accumulated
        // f32 powf rounding error that causes degeneration on high-theta models like Llama 3.1.)
        let theta: f64 = hyperparams.rope_params.as_ref().map(|r| r.theta as f64).unwrap_or(10000.0);
        let mut rope_cos = vec![0.0f32; max_seq_len * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                rope_cos[pos * half_dim + i] = angle.cos() as f32;
                rope_sin[pos * half_dim + i] = angle.sin() as f32;
            }
        }

        let rope_cos_buf = self.upload_f32(&rope_cos)?;
        let rope_sin_buf = self.upload_f32(&rope_sin)?;

        // Create scratch GPU buffers
        let make_buf = |n: usize| -> Result<MetalBuffer, RuntimeError> {
            let len = n.max(1) * 4; // at least 4 bytes
            self.device.new_buffer(len).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate Metal buffer of {len} bytes"))
            })
        };

        // Allocate GPU-resident KV cache buffers: one pair per layer,
        // each sized for [max_seq_len * kv_dim] halfs (f16 = 2 bytes per element).
        let kv_cache_size = max_seq_len * kv_dim;
        let make_buf_f16 = |n: usize| -> Result<MetalBuffer, RuntimeError> {
            let len = n.max(1) * 2; // f16: 2 bytes per element
            self.device.new_buffer(len).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate Metal buffer of {len} bytes"))
            })
        };
        let mut gpu_k_cache = Vec::with_capacity(num_layers);
        let mut gpu_v_cache = Vec::with_capacity(num_layers);
        for _layer in 0..num_layers {
            gpu_k_cache.push(make_buf_f16(kv_cache_size)?);
            gpu_v_cache.push(make_buf_f16(kv_cache_size)?);
        }

        // Allocate MoE decode scratch buffers when the model has experts.
        // Option B dispatches ALL num_experts expert FFNs; non-selected experts
        // produce zero output via zero routing weights.
        let (moe_router_logits, moe_expert_ids, moe_expert_weights, moe_expert_output) =
            if moe_num_experts > 0 {
                println!(
                    "MoE model detected: {} experts, top-{} active. Allocating MoE scratch buffers.",
                    moe_num_experts, moe_num_active_experts,
                );
                (
                    Some(make_buf(moe_num_experts)?),                           // [num_experts] f32
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate MoE expert_ids buffer".into())
                    })?),                                                       // [top_k] u32
                    Some(make_buf(moe_num_active_experts)?),                    // [top_k] f32
                    Some(make_buf(moe_num_experts * hidden_dim)?),              // [num_experts * hidden_dim] f32
                )
            } else {
                (None, None, None, None)
            };

        let scratch = MetalScratch {
            // Persistent activation buffer: allocated once, reused every layer.
            x_buf: make_buf(hidden_dim)?,
            normed_buf: make_buf(hidden_dim)?,
            qkv_buf: make_buf(qkv_dim.max(8192))?,   // GDN needs 8192 (Q4096+K2048+V2048)
            q_buf: make_buf(q_dim)?,
            k_buf: make_buf(kv_dim.max(2048))?,    // GDN needs 2048 (16 KV heads * 128 dim)
            v_buf: make_buf(kv_dim.max(2048))?,    // GDN needs 2048 (16 KV heads * 128 dim)
            attn_out_buf: make_buf(q_dim)?,
            scores_buf: make_buf(max_seq_len)?,
            attn_proj_buf: make_buf(hidden_dim)?,
            gate_buf: make_buf(inter_dim.max(q_dim))?,  // max of FFN inter_dim and attn q_dim (Q+gate deinterleave)
            up_buf: make_buf(inter_dim)?,
            down_buf: make_buf(hidden_dim)?,
            logits_buf: make_buf(vocab_size)?,
            argmax_result_buf: self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to allocate argmax result buffer (4 bytes)".into())
            })?,
            rope_cos_buf,
            rope_sin_buf,
            gpu_k_cache,
            gpu_v_cache,
            mha_scores_buf: make_buf(num_heads * max_seq_len)?,
            // Flash decode: tile_size=256, max_tiles = ceil(max_seq/256)
            // Each tile: head_dim + 2 floats (weighted_v + max + sum)
            flash_decode_partial_buf: make_buf(
                num_heads * ((max_seq_len + 255) / 256) * (head_dim + 2)
            )?,
            hidden_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            head_dim,
            inter_dim,
            eps: hyperparams.norm_eps,
            q_dim,
            kv_dim,
            qkv_dim,
            gqa_ratio,
            vocab_size,
            half_dim,
            max_seq_len,
            attn_scale,
            matmul_tg_size,
            norm_tg_size,
            mha_tg_size,

            gpu_x_valid: false,
            last_async_cmd: None,
            layer_buf_cache: (0..num_layers).map(|_| None).collect(),
            moe_partial_buf_cache: (0..num_layers).map(|_| None).collect(),
            gpu_resident_layers: None,
            gpu_unified_weight_buf: None,
            gpu_layer_offsets: Vec::new(),
            gpu_global_offsets: None,

            // Batched buffers: lazily allocated on first prefill call
            batch_x_buf: None,
            batch_normed_buf: None,
            batch_qkv_buf: None,
            batch_q_buf: None,
            batch_k_buf: None,
            batch_v_buf: None,
            batch_attn_out_buf: None,
            batch_attn_proj_buf: None,
            batch_gate_buf: None,
            batch_up_buf: None,
            batch_down_buf: None,
            batch_scores_buf: None,
            splitk_partial_buf: None,
            splitk_alloc_elems: 0,
            current_max_batch: 0,

            logits_readback: vec![0.0f32; vocab_size],
            cached_layer_meta: Vec::new(),

            // MoE parameters and scratch buffers
            moe_num_experts,
            moe_num_active_experts,
            moe_expert_inter_dim,
            moe_router_logits,
            moe_expert_ids,
            moe_expert_weights,
            moe_expert_output,
            // Batched MoE buffers: lazily allocated in ensure_batch_buffers
            moe_batch_router_logits: None,
            moe_batch_expert_ids: None,
            moe_batch_expert_weights: None,
            moe_batch_expert_output: None,

            // Per-layer expert IDs buffers for GPU-resident profiling.
            // Allocated for MoE models so each layer's expert selections are preserved.
            moe_per_layer_expert_ids: if moe_num_experts > 0 {
                (0..num_layers).map(|_| {
                    // Each layer gets its own [top_k] u32 buffer.
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4)
                        .expect("Failed to allocate per-layer MoE expert_ids buffer"))
                }).collect()
            } else {
                Vec::new()
            },

            // Per-layer expert weights buffers for router diagnostics.
            // Allocated for MoE models when router_debug is enabled.
            moe_per_layer_expert_weights: if moe_num_experts > 0 && self.router_debug_enabled {
                (0..num_layers).map(|_| {
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4)
                        .expect("Failed to allocate per-layer MoE expert_weights buffer"))
                }).collect()
            } else {
                Vec::new()
            },

            // Batched MoE offset tables — populated in preload_weights_gpu_resident.
            moe_gate_up_offsets: Vec::new(),
            moe_down_offsets: Vec::new(),
            moe_batched_swiglu_buf: if moe_num_experts > 0 && moe_num_active_experts > 0 {
                Some(self.device.new_buffer((moe_num_active_experts * moe_expert_inter_dim * 4).max(4))
                    .expect("Failed to allocate batched swiglu buffer"))
            } else {
                None
            },
            // Populated in preload_weights_gpu_resident alongside moe_gate_up_offsets
            moe_shared_down_offsets: Vec::new(),
            moe_shared_gate_scalar_buf: if moe_num_experts > 0 {
                // Single f32 for shared expert gating scalar
                Some(self.device.new_buffer(4)
                    .expect("Failed to allocate shared expert gate scalar buffer"))
            } else {
                None
            },

            // Qwen3.5-MoE scratch: defaults for non-hybrid models.
            // Overridden in preload_weights_gpu_resident when hybrid model is detected.
            rope_theta: hyperparams.rope_params.as_ref().map(|r| r.theta as f64).unwrap_or(10000.0),
            rope_neox: hyperparams.rope_neox,
            rotary_dim: head_dim,
            shared_expert_inter_dim: 0,
            shared_expert_gate_buf: None,
            shared_expert_down_buf: None,
            attn_gate_buf: None,

            // GDN state: defaults for non-hybrid models. Overridden in
            // preload_weights_gpu_resident when Qwen3.5-MoE is detected.
            gdn_h_states: Vec::new(),
            gdn_conv_states: Vec::new(),
            gdn_conv_positions: Vec::new(),
            gdn_alpha_buf: None,
            gdn_beta_buf: None,
            gdn_output_buf: None,
            gdn_ssm_proj_buf: None,
            gdn_gate_sigmoid_buf: None,
            gdn_normed_out_buf: None,
            gdn_alpha_raw_buf: None,
            gdn_beta_raw_buf: None,
            gdn_qkv_conv_buf: None,
            gdn_conv_kernel_size: 4,
            gdn_num_layers: 0,
            gdn_layer_idx_map: Vec::new(),
        };

        *self.scratch.lock().unwrap() = Some(scratch);

        // Pre-allocate batch buffers for common prefill sizes (up to 512 tokens).
        // This moves Metal buffer allocation from the first prefill call into
        // init(), so the first prefill is as fast as subsequent ones.
        {
            let default_batch = max_seq_len.min(512);
            let mut scratch_guard = self.scratch.lock().unwrap();
            let s = scratch_guard.as_mut().unwrap();
            self.ensure_batch_buffers(s, default_batch)?;
        }

        // Initialize MoE expert caching infrastructure.
        // Only activated for MoE models (num_experts > 0).
        if moe_num_experts > 0 {
            // Always initialize the profiler for MoE models.
            self.expert_profiler = Some(Mutex::new(
                ExpertActivationProfiler::new(num_layers, moe_num_experts),
            ));

            // Initialize ExpertReader if LBC path was provided via configure_expert_cache.
            if let Some(ref path) = self.lbc_path {
                match ExpertReader::open(path) {
                    Ok(reader) => {
                        self.expert_reader = Some(Mutex::new(reader));
                        println!(
                            "MoE expert cache: reader initialized for {}",
                            path.display(),
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to open ExpertReader at {}: {e}. \
                             Expert caching disabled, falling back to full-layer loading.",
                            path.display(),
                        );
                    }
                }
            }

            if let Some(ref cache) = self.expert_cache {
                let stats = cache.lock().unwrap().stats();
                println!(
                    "MoE expert cache: capacity={} experts, reader={}",
                    stats.capacity,
                    if self.expert_reader.is_some() { "active" } else { "inactive" },
                );
            }
        }

        Ok(())
    }

    fn compute_layer(
        &self,
        layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized: call init() first".into())
        })?;

        let hidden_dim = s.hidden_dim;
        let num_heads = s.num_heads;
        let num_kv_heads = s.num_kv_heads;
        let head_dim = s.head_dim;
        let inter_dim = s.inter_dim;
        let eps = s.eps;
        let q_dim = s.q_dim;
        let kv_dim = s.kv_dim;
        let qkv_dim = s.qkv_dim;
        let half_dim = s.half_dim;
        let attn_scale = s.attn_scale;
        let matmul_tg_size = s.matmul_tg_size;
        let norm_tg_size = s.norm_tg_size;

        // GPU activation persistence: skip CPU->GPU upload if GPU already
        // has valid activation data. When embed_token wrote directly to x_buf
        // on GPU (gpu_x_valid=true), we skip the upload even at layer_idx==0.
        // compute_final always resets gpu_x_valid=false, ensuring fresh data
        // is loaded at the start of each forward pass.
        if !s.gpu_x_valid {
            // Drain any pending async GPU work before CPU writes to shared buffers.
            // This prevents races where a previous forward pass's async copy_buffer
            // hasn't completed yet.
            if let Some(prev_cmd) = s.last_async_cmd.take() {
                prev_cmd.wait_until_completed();
            }
            let x_f32 = x.as_f32_slice();
            s.x_buf.write_f32(x_f32);
        }

        // GPU-resident path: prefer unified private buffer, then per-layer buffers,
        // then fall back to cached zero-copy layer buffer.
        let layer_buf: &MetalBuffer;
        let base_off: u64;
        if let Some(ref ubuf) = s.gpu_unified_weight_buf {
            layer_buf = ubuf;
            base_off = s.gpu_layer_offsets[layer_idx] as u64;
        } else if let Some(ref layers) = s.gpu_resident_layers {
            if let Some(buf) = layers.get(layer_idx) {
                layer_buf = buf;
                base_off = 0;
            } else {
                return Err(RuntimeError::Compute(format!(
                    "gpu_resident_layers missing layer {}", layer_idx
                )));
            }
        } else {
            // Zero-copy layer buffer: cached per-layer to avoid re-creating Metal buffers.
            let blob = weights.as_bytes();
            let blob_ptr = blob.as_ptr() as usize;

            // For MoE layers with expert cache, use a partial buffer
            // covering only non-expert data (attention, norms, router). This avoids
            // page-faulting the expert byte range from mmap. Expert weights are served
            // from the LFU cache or loaded individually via ExpertReader.
            let is_moe_layer = s.moe_num_experts > 0
                && weights.subtensors.experts.is_some()
                && self.expert_cache.is_some();
            let use_partial = is_moe_layer && {
                let cache = self.expert_cache.as_ref().unwrap().lock().unwrap();
                let num_exp = s.moe_num_experts;
                (0..num_exp).all(|e| cache.contains(&(layer_idx, e as u32)))
            };

            if use_partial {
                // Partial buffer: only non-expert bytes.
                let non_exp_end = Self::non_expert_byte_end(&weights.subtensors);
                let cached = s.moe_partial_buf_cache.get(layer_idx).and_then(|c| c.as_ref());
                let need_create = match cached {
                    Some((ptr, end, _)) => *ptr != blob_ptr || *end != non_exp_end,
                    None => true,
                };
                if need_create {
                    let buf = self.create_partial_layer_buffer(weights, non_exp_end)?;
                    s.moe_partial_buf_cache[layer_idx] = Some((blob_ptr, non_exp_end, buf));
                }
                layer_buf = &s.moe_partial_buf_cache[layer_idx].as_ref().unwrap().2;

                // Hint to OS that expert byte pages can be reclaimed.
                // madvise(MADV_DONTNEED) on expert range [non_exp_end..blob_len].
                // Best-effort: no error handling if the call fails.
                if non_exp_end < blob.len() {
                    let expert_page_start = (non_exp_end + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
                    if expert_page_start < blob.len() {
                        unsafe {
                            libc::madvise(
                                (blob_ptr + expert_page_start) as *mut libc::c_void,
                                blob.len() - expert_page_start,
                                libc::MADV_DONTNEED,
                            );
                        }
                    }
                }
            } else {
                // Full buffer: all bytes (dense layers or MoE without full cache).
                let cached = s.layer_buf_cache.get(layer_idx).and_then(|c| c.as_ref());
                let need_create = match cached {
                    Some((ptr, _)) => *ptr != blob_ptr,
                    None => true,
                };
                if need_create {
                    let buf = self.create_layer_buffer(weights)?;
                    s.layer_buf_cache[layer_idx] = Some((blob_ptr, buf));
                }
                layer_buf = &s.layer_buf_cache[layer_idx].as_ref().unwrap().1;
            }
            base_off = 0;
        }
        let st = &weights.subtensors;

        let attn_norm_off = base_off + st.attn_norm.offset;
        let wq_off = base_off + st.wq.offset;  // Wq/Wk/Wv are contiguous; wq_off is the fused QKV start
        let wo_off = base_off + st.wo.offset;
        // Prefer attn_post_norm when ffn_norm is the zero-sentinel (Qwen3.5-35B-A3B).
        let ffn_norm_off = if st.ffn_norm.length == 0 {
            st.attn_post_norm.map_or(0, |s| base_off + s.offset)
        } else {
            base_off + st.ffn_norm.offset
        };
        let w_gate_off = base_off + st.w_gate.offset;
        let w_up_off = base_off + st.w_up.offset;
        let w_down_off = base_off + st.w_down.offset;

        let is_gdn_layer = st.layer_type == Some(1);

        let kv = kv.ok_or_else(|| {
            RuntimeError::Compute("KV cache view required for attention".into())
        })?;
        let new_seq_len = kv.seq_len + 1;

        // ================================================================
        // Lazy GDN state allocation for streaming path.
        // GPU-resident path allocates in preload_weights_gpu_resident;
        // streaming path needs lazy init on first encounter of each GDN layer.
        // ================================================================
        if is_gdn_layer {
            // Extend gdn_layer_idx_map to cover this layer_idx if needed.
            if s.gdn_layer_idx_map.len() <= layer_idx {
                s.gdn_layer_idx_map.resize(layer_idx + 1, None);
            }
            if s.gdn_layer_idx_map[layer_idx].is_none() {
                let gdn_idx = s.gdn_h_states.len();
                s.gdn_layer_idx_map[layer_idx] = Some(gdn_idx);

                // GDN dimensions differ from full-attention hyperparams.
                const GDN_NUM_HEADS: usize = 32;    // ssm.time_step_rank
                const GDN_HEAD_DIM: usize = 128;    // ssm.state_size
                const GDN_QKV_DIM: usize = 8192;    // Q(2048) + K(2048) + V(4096)
                let gdn_q_dim = GDN_NUM_HEADS * GDN_HEAD_DIM; // 4096
                let conv_kernel_size = 4usize;
                let h_state_size = GDN_NUM_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM; // 32*128*128
                let conv_state_size = (conv_kernel_size - 1) * GDN_QKV_DIM;     // 3*8192

                let h_buf = self.device.new_buffer(h_state_size * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN h_state".into())
                })?;
                h_buf.write_f32(&vec![0.0f32; h_state_size]);

                let c_buf = self.device.new_buffer(conv_state_size * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN conv_state".into())
                })?;
                c_buf.write_f32(&vec![0.0f32; conv_state_size]);

                s.gdn_h_states.push(h_buf);
                s.gdn_conv_states.push(c_buf);
                s.gdn_conv_positions.push(0);
                s.gdn_conv_kernel_size = conv_kernel_size;
                s.gdn_num_layers = s.gdn_h_states.len();

                // Allocate GDN scratch buffers if not yet done (first GDN layer).
                if s.gdn_alpha_buf.is_none() {
                    s.gdn_alpha_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha buf".into())
                    })?);
                    s.gdn_beta_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta buf".into())
                    })?);
                    s.gdn_output_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN output buf".into())
                    })?);
                    s.gdn_ssm_proj_buf = Some(self.device.new_buffer(hidden_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN ssm_proj buf".into())
                    })?);
                    s.gdn_gate_sigmoid_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN gate sigmoid buf".into())
                    })?);
                    s.gdn_normed_out_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN normed_out buf".into())
                    })?);
                    s.gdn_alpha_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha_raw buf".into())
                    })?);
                    s.gdn_beta_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta_raw buf".into())
                    })?);
                    s.gdn_qkv_conv_buf = Some(self.device.new_buffer(GDN_QKV_DIM * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN qkv_conv buf".into())
                    })?);
                }
            }
        }

        // ================================================================
        // SINGLE command buffer for the ENTIRE layer (16 encoders, 1 sync).
        // Previous: 3 command buffers × 22 layers = 66 sync barriers/token.
        // Now: 1 command buffer × 22 layers = 22 sync barriers/token.
        // ================================================================
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer".into())
        })?;

        // ================================================================
        // GDN vs softmax attention routing.
        // GDN layers use encode_gdn_layer_decode (linear attention with
        // recurrent state). Softmax layers use the standard RoPE + KV cache
        // + multi-head attention path. Both produce valid attn_proj_buf for
        // the downstream FFN norm.
        // ================================================================
        if is_gdn_layer {
            // Build CachedLayerMeta from SubtensorOffsets for GDN dispatch.
            let gdn_idx = s.gdn_layer_idx_map[layer_idx].unwrap();
            let gdn_meta = CachedLayerMeta {
                attn_norm_off,
                wq_off,
                wo_off,
                ffn_norm_off,
                w_gate_off,
                w_up_off,
                w_down_off,
                wq_quant: st.wq.quant,
                wo_quant: st.wo.quant,
                w_gate_quant: st.w_gate.quant,
                w_up_quant: st.w_up.quant,
                w_down_quant: st.w_down.quant,
                bq_off: st.bq.map(|b| base_off + b.offset),
                bk_off: st.bk.map(|b| base_off + b.offset),
                bv_off: st.bv.map(|b| base_off + b.offset),
                moe_meta: None,
                shared_expert_gate_off: None,
                shared_expert_up_off: None,
                shared_expert_down_off: None,
                shared_expert_gate_quant: None,
                shared_expert_down_quant: None,
                attn_gate_off: st.attn_gate.map(|g| base_off + g.offset),
                attn_gate_quant: st.attn_gate.map(|g| g.quant),
                attn_post_norm_off: st.attn_post_norm.map(|n| base_off + n.offset),
                layer_type: st.layer_type,
                ssm_a_off: st.ssm_a.map(|t| base_off + t.offset),
                ssm_conv1d_off: st.ssm_conv1d.map(|t| base_off + t.offset),
                ssm_dt_off: st.ssm_dt.map(|t| base_off + t.offset),
                ssm_beta_off: st.ssm_beta.map(|t| base_off + t.offset),
                ssm_alpha_off: st.ssm_alpha.map(|t| base_off + t.offset),
                ssm_norm_off: st.ssm_norm.map(|t| base_off + t.offset),
                ssm_out_off: st.ssm_out.map(|t| base_off + t.offset),
                ssm_out_quant: st.ssm_out.map(|t| t.quant),
                gdn_layer_idx: Some(gdn_idx),
                // GDN layers never have Q+gate fusion or separate K/V weights.
                has_qgate_fusion: false,
                wk_off: None,
                wv_off: None,
                wk_quant: None,
                wv_quant: None,
                attn_q_norm_off: None,
                attn_k_norm_off: None,
                ffn_gate_inp_shexp_off: st.ffn_gate_inp_shexp.map(|t| base_off + t.offset),
            };

            let new_conv_pos = Self::encode_gdn_layer_decode(
                &cmd, pipelines, s, layer_buf, &gdn_meta, gdn_idx,
            )?;
            s.gdn_conv_positions[gdn_idx] = new_conv_pos;
        } else {
        // --- Encoder 1: Attention RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 2: QKV projection ---
        // Two paths: Q+gate fusion (Qwen3.5 full-attention) vs fused QKV (standard).
        let has_qgate_fusion = st.attn_q_norm.is_some();
        let q_byte_off: u64 = 0;
        let k_byte_off: u64 = (q_dim * 4) as u64;
        let v_byte_off: u64 = ((q_dim + kv_dim) * 4) as u64;
        // Use partial rotary dimension when set (e.g. Qwen3.5: rotary_dim=64, not head_dim=256).
        let rope_half_dim = if s.rotary_dim > 0 && s.rotary_dim < head_dim { s.rotary_dim / 2 } else { half_dim };
        let use_partial_rope = rope_half_dim != half_dim;

        if has_qgate_fusion {
            // Q+gate fusion path (Qwen3.5 full-attention layers).
            // Q weight [hidden, q_dim*2] produces interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...].
            // K and V come from separate wk, wv weights.
            let qgate_dim = q_dim * 2;
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for Q+gate matmul".into())
                })?;
                let tg = match st.wq.quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.qkv_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match st.wq.quant {
                    QuantScheme::Q8_0 => ((qgate_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((qgate_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((qgate_dim as u64) + 1) / 2,
                    _ => qgate_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // De-interleave Q+gate -> q_buf + gate_buf
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for deinterleave".into())
                })?;
                let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("deinterleave_qgate pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso);
                enc.set_buffer(&s.qkv_buf, 0, 0);
                enc.set_buffer(&s.q_buf, 0, 1);
                enc.set_buffer(&s.gate_buf, 0, 2);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
                let di_tg = 256u64.min(q_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(di_tg), 1, 1),
                    MTLSize::new(di_tg, 1, 1),
                );
                enc.end_encoding();
            }
            // Project K from wk
            {
                let wk_off_val = base_off + st.wk.offset;
                let wk_quant = st.wk.quant;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for K matmul".into())
                })?;
                let tg = match wk_quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wk_off_val, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.k_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(wk_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match wk_quant {
                    QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                    _ => kv_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // Project V from wv
            {
                let wv_off_val = base_off + st.wv.offset;
                let wv_quant = st.wv.quant;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for V matmul".into())
                })?;
                let tg = match wv_quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wv_off_val, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.v_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(wv_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match wv_quant {
                    QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                    _ => kv_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // Per-head Q/K RMSNorm
            if let (Some(ref q_norm), Some(ref k_norm)) = (&st.attn_q_norm, &st.attn_k_norm) {
                let q_norm_off = base_off + q_norm.offset;
                let k_norm_off = base_off + k_norm.offset;
                let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                })?;
                let head_dim_u32 = head_dim as u32;
                let tg_rms = 256u64.min(head_dim as u64).max(32);
                // Q RMSNorm
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for Q RMSNorm".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(&s.q_buf, 0, 0);
                    enc.set_buffer(layer_buf, q_norm_off, 1);
                    enc.set_buffer(&s.q_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    enc.end_encoding();
                }
                // K RMSNorm
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for K RMSNorm".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(&s.k_buf, 0, 0);
                    enc.set_buffer(layer_buf, k_norm_off, 1);
                    enc.set_buffer(&s.k_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_kv_heads as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
            // Assemble Q|K|V into qkv_buf for RoPE/attention
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for QKV assembly".into())
                })?;
                enc.set_pipeline_state(&pipelines.copy_buffer);
                // Copy Q
                enc.set_buffer(&s.q_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, 0, 1);
                let tg_q = 256u64.min(q_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(tg_q), 1, 1),
                    MTLSize::new(tg_q, 1, 1),
                );
                // Copy K
                enc.set_buffer(&s.k_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                let tg_k = 256u64.min(kv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((kv_dim as u64).div_ceil(tg_k), 1, 1),
                    MTLSize::new(tg_k, 1, 1),
                );
                // Copy V
                enc.set_buffer(&s.v_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                let tg_v = 256u64.min(kv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((kv_dim as u64).div_ceil(tg_v), 1, 1),
                    MTLSize::new(tg_v, 1, 1),
                );
                enc.end_encoding();
            }
        } else {
        // Fused QKV projection (standard path: Wq|Wk|Wv contiguous).
        {
            let has_bias = st.bq.is_some() && st.bk.is_some() && st.bv.is_some();
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let in_dim_u32 = hidden_dim as u32;

            let tg = if has_bias && matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                match st.wq.quant {
                    QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_bias_nr2),
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_bias_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.matmul_f16_deferred_bias_nr2),
                    _ => unreachable!(),
                };
                128u64
            } else {
                match st.wq.quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                }
            };
            // Weight pointer starts at Wq; contiguous through Wk, Wv
            enc.set_buffer(layer_buf, wq_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.qkv_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            let qkv_out_dim_u32 = qkv_dim as u32;
            if matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                enc.set_bytes(&qkv_out_dim_u32.to_le_bytes(), 4);
            }
            if has_bias && matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let bq_off_abs = base_off + st.bq.as_ref().unwrap().offset;
                let bk_off_abs = base_off + st.bk.as_ref().unwrap().offset;
                let bv_off_abs = base_off + st.bv.as_ref().unwrap().offset;
                enc.set_buffer(layer_buf, bq_off_abs, 5);
                enc.set_buffer(layer_buf, bk_off_abs, 6);
                enc.set_buffer(layer_buf, bv_off_abs, 7);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 8);
                let qk_dim = (q_dim + kv_dim) as u32;
                enc.set_bytes(&qk_dim.to_le_bytes(), 9);
            }
            let n_tg_qkv = if tg == 64 {
                ((qkv_dim as u64) + 7) / 8  // (dead path: Q8_0 now uses deferred with tg=128)
            } else {
                match st.wq.quant {
                    QuantScheme::Q8_0 => ((qkv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2,
                    _ => qkv_dim as u64,
                }
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_qkv, 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            enc.end_encoding();
        }

        // --- QKV bias addition fallback (F32 weights with bias only) ---
        if !matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && (st.bq.is_some() || st.bk.is_some() || st.bv.is_some())
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for QKV bias".into())
            })?;
            enc.set_pipeline_state(&pipelines.bias_add);
            if let Some(ref bq) = st.bq {
                let bq_off = base_off + bq.offset;
                enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                enc.set_buffer(layer_buf, bq_off, 1);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                let n_tg_bq = (q_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bq, 1, 1), MTLSize::new(256, 1, 1));
            }
            if let Some(ref bk) = st.bk {
                let bk_off = base_off + bk.offset;
                enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                enc.set_buffer(layer_buf, bk_off, 1);
                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                let n_tg_bk = (kv_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bk, 1, 1), MTLSize::new(256, 1, 1));
            }
            if let Some(ref bv) = st.bv {
                let bv_off = base_off + bv.offset;
                enc.set_buffer(&s.qkv_buf, v_byte_off, 0);
                enc.set_buffer(layer_buf, bv_off, 1);
                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                let n_tg_bv = (kv_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bv, 1, 1), MTLSize::new(256, 1, 1));
            }
            enc.end_encoding();
        }
        } // end if has_qgate_fusion else

        // --- RoPE + KV cache write ---
        // For partial RoPE (Qwen3.5: rotary_dim=64 < head_dim=256), use separate dispatches.
        // For full RoPE, use fused dispatch.
        if use_partial_rope {
            // Separate RoPE Q + RoPE K + KV cache write (partial rotary dim).
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for partial RoPE".into())
            })?;
            let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
            let rope_pipe = if s.rope_neox {
                pipelines.rope_neox.as_ref().unwrap_or(&pipelines.rope)
            } else {
                &pipelines.rope
            };
            // RoPE Q
            enc.set_pipeline_state(rope_pipe);
            enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
            enc.set_buffer(&s.rope_cos_buf, 0, 1);
            enc.set_buffer(&s.rope_sin_buf, 0, 2);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&pos_offset_u32.to_le_bytes(), 6);
            let q_total_half = (num_heads * rope_half_dim) as u64;
            let tg_q = 64u64.min(q_total_half.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(q_total_half.div_ceil(tg_q), 1, 1),
                MTLSize::new(tg_q, 1, 1),
            );
            // RoPE K
            enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
            let k_total_half = (num_kv_heads * rope_half_dim) as u64;
            let tg_k = 64u64.min(k_total_half.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(k_total_half.div_ceil(tg_k), 1, 1),
                MTLSize::new(tg_k, 1, 1),
            );
            // KV cache write
            enc.set_pipeline_state(&pipelines.write_kv_cache);
            enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
            enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
            enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 2);
            enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 3);
            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 5);
            enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 6);
            let tg_kv = 64u64.min(kv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((kv_dim as u64).div_ceil(tg_kv), 1, 1),
                MTLSize::new(tg_kv, 1, 1),
            );
            enc.end_encoding();
        } else {
            // Fused RoPE Q + RoPE K + KV cache write (full rotary dim).
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let pos_offset_u32 = (seq_pos * half_dim) as u32;
            let fused_pipe = if s.rope_neox {
                pipelines.fused_rope_neox_kv_write.as_ref().unwrap_or(&pipelines.fused_rope_kv_write)
            } else {
                &pipelines.fused_rope_kv_write
            };
            enc.set_pipeline_state(fused_pipe);
            enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
            enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
            enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
            enc.set_buffer(&s.rope_cos_buf, 0, 3);
            enc.set_buffer(&s.rope_sin_buf, 0, 4);
            enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
            enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
            enc.set_bytes(&(half_dim as u32).to_le_bytes(), 10);
            enc.set_bytes(&pos_offset_u32.to_le_bytes(), 11);
            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 12);
            enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 13);
            enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 14);
            let total_threads = (num_heads * half_dim + num_kv_heads * half_dim + kv_dim) as u64;
            let tg = 64u64.min(total_threads.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(total_threads.div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 8: Attention (flash decode for long KV, original for short) ---
        //
        // Flash Decoding splits the KV sequence into tiles processed by separate
        // threadgroups, then reduces. This provides parallelism across the KV
        // dimension which is critical when seq_len is large (e.g., 512+).
        // For short sequences (<=128), the original single-threadgroup kernel
        // is used since the overhead of the two-phase approach is not justified.
        {
            let num_heads_u32 = num_heads as u32;
            let num_kv_heads_u32 = num_kv_heads as u32;
            let head_dim_u32 = head_dim as u32;
            let kv_dim_u32 = kv_dim as u32;
            let seq_len_u32 = new_seq_len as u32;
            let max_seq_len_u32 = s.max_seq_len as u32;

            const FLASH_DECODE_TILE_SIZE: u32 = 256;
            const FLASH_DECODE_THRESHOLD: usize = FLASH_DECODE_TILE_SIZE as usize + 1; // 257: single-tile is a no-op reduce

            if new_seq_len >= FLASH_DECODE_THRESHOLD {
                // --- Flash Decode Phase 1: tiled attention ---
                let tile_size_u32 = FLASH_DECODE_TILE_SIZE;
                let num_tiles = ((new_seq_len as u32) + tile_size_u32 - 1) / tile_size_u32;
                let num_tiles_u32 = num_tiles;

                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
                    enc.set_pipeline_state(&pipelines.flash_decode_attention);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                    enc.set_buffer(&s.flash_decode_partial_buf, 0, 3);
                    enc.set_bytes(&num_heads_u32.to_le_bytes(), 4);
                    enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 5);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 6);
                    enc.set_bytes(&kv_dim_u32.to_le_bytes(), 7);
                    enc.set_bytes(&seq_len_u32.to_le_bytes(), 8);
                    enc.set_bytes(&attn_scale.to_le_bytes(), 9);
                    enc.set_bytes(&tile_size_u32.to_le_bytes(), 10);
                    enc.set_bytes(&num_tiles_u32.to_le_bytes(), 11);
                    enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                    // Each threadgroup gets 128 threads (tile_size=256, each thread handles multiple scores)
                    // Use flattened 1D grid: threadgroup i = head * num_tiles + tile_idx
                    let tg_threads = 128u64;
                    let total_tgs = (num_heads as u64) * (num_tiles as u64);
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_tgs, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                    enc.end_encoding();
                }

                // --- Flash Decode Phase 2: reduce across tiles ---
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
                    enc.set_pipeline_state(&pipelines.flash_decode_reduce);
                    enc.set_buffer(&s.flash_decode_partial_buf, 0, 0);
                    enc.set_buffer(&s.attn_out_buf, 0, 1);
                    enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&num_tiles_u32.to_le_bytes(), 4);
                    // One threadgroup per head, enough threads for head_dim
                    let tg_threads = (head_dim as u64).max(1).min(256);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                    enc.end_encoding();
                }
            } else {
                // --- Original single-threadgroup MHA for short sequences ---
                let mha_tg_size = s.mha_tg_size;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                enc.set_pipeline_state(&pipelines.multi_head_attention);
                enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                enc.set_buffer(&s.attn_out_buf, 0, 3);
                enc.set_buffer(&s.mha_scores_buf, 0, 4);
                enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
                enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 7);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 8);
                enc.set_bytes(&seq_len_u32.to_le_bytes(), 9);
                enc.set_bytes(&attn_scale.to_le_bytes(), 10);
                enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 11);
                let tg_threads = mha_tg_size.min(
                    (head_dim.max(new_seq_len) as u64).max(1)
                );
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
                enc.end_encoding();
            }
        }

        // --- Sigmoid gate for Q+gate fusion (Qwen3.5 full-attention) ---
        // sigmoid(gate_buf) * attn_out_buf -> attn_out_buf (in-place)
        if has_qgate_fusion {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for sigmoid gate".into())
            })?;
            let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso);
            enc.set_buffer(&s.gate_buf, 0, 0);       // gate [q_dim]
            enc.set_buffer(&s.attn_out_buf, 0, 1);   // attn output [q_dim]
            enc.set_buffer(&s.attn_out_buf, 0, 2);   // output (in-place)
            let total_gate_elems = q_dim as u32;
            enc.set_bytes(&total_gate_elems.to_le_bytes(), 3);
            let tg = 256u64.min(total_gate_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_gate_elems as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 6: Wo projection + Residual 1 (fused) ---
        // attn_proj_buf = Wo * attn_out + x_buf  (eliminates separate add_residual)
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let in_dim_u32 = q_dim as u32;
            let tg_wo = match st.wo.quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, wo_off, 0);
            enc.set_buffer(&s.attn_out_buf, 0, 1);
            enc.set_buffer(&s.attn_proj_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            enc.set_buffer(&s.x_buf, 0, 4);  // residual input
            if matches!(st.wo.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let wo_out_dim_u32 = hidden_dim as u32;
                enc.set_bytes(&wo_out_dim_u32.to_le_bytes(), 5);
            }
            let n_tg_wo = match st.wo.quant {
                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                _ => hidden_dim as u64,
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_wo, 1, 1),
                MTLSize::new(tg_wo, 1, 1),
            );
            enc.end_encoding();
        }
        } // end if !is_gdn_layer (softmax attention block)

        // --- Encoder 11: FFN RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- FFN block: MoE vs dense path for streaming decode ---
        // If this layer has MoE experts, dispatch via encode_moe_ffn_decode and skip
        // the dense FFN path below.
        //
        // Option A: When use_option_a is true and we're in streaming mode,
        // split into two CBs: CB1 (attention+norm+router) -> readback -> CB2 (top-K FFNs).
        // This eliminates (num_experts - top_k) / num_experts of expert FFN compute.
        //
        // Partial layer loading: Assemble expert weights from
        // LFU cache (hits) + ExpertReader (misses), avoiding the need to
        // create a Metal buffer for the full layer blob's expert region.
        // When all experts are cached, the layer_buf is a partial buffer
        // covering only non-expert bytes -- zero expert I/O from disk.
        // When some experts are missing, they are loaded individually via
        // ExpertReader, which reads only the needed gate+up+down byte
        // ranges from the LBC file.

        // Check if Option A is active for this layer.
        // Option A requires: (1) use_option_a flag, (2) streaming mode (not GPU-resident),
        // (3) MoE model with expert cache or reader available.
        let is_streaming_mode = s.gpu_unified_weight_buf.is_none()
            && s.gpu_resident_layers.is_none();
        let option_a_active = self.use_option_a
            && is_streaming_mode
            && s.moe_num_experts > 0
            && self.expert_profiler.is_some();

        let moe_handled = if s.moe_num_experts > 0 {
            if let (Some(ref router), Some(ref experts)) = (&st.router_weight, &st.experts) {
                let first = &experts[0];
                let num_experts = s.moe_num_experts;
                let top_k = s.moe_num_active_experts;

                // ==============================================================
                // Option A: Two-CB split for selective expert dispatch
                // ==============================================================
                if option_a_active {
                    // Phase 1: Encode router into the current CB, then commit+wait.
                    // The router writes expert_ids and expert_weights to GPU buffers.
                    {
                        // Build is_cached buffer for biased routing.
                        let (is_cached_buf, bias_lambda) = if self.cache_bias_lambda > 0.0
                            && self.warmup_complete.load(Ordering::Relaxed)
                            && self.expert_cache.is_some()
                        {
                            let cache = self.expert_cache.as_ref().unwrap().lock().unwrap();
                            let is_cached_data: Vec<u8> = (0..num_experts)
                                .map(|e| if cache.contains(&(layer_idx, e as u32)) { 1u8 } else { 0u8 })
                                .collect();
                            drop(cache);
                            (self.device.new_buffer_with_bytes(&is_cached_data), self.cache_bias_lambda)
                        } else {
                            (None, 0.0)
                        };

                        let use_biased = bias_lambda > 0.0 && is_cached_buf.is_some();
                        let router_softmax = if use_biased {
                            pipelines.moe_router_softmax_biased.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("MoE router_softmax_biased pipeline not compiled.".into())
                            })?
                        } else {
                            pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("MoE router_softmax pipeline not compiled.".into())
                            })?
                        };

                        let expert_ids_buf = s.moe_expert_ids.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE expert_ids buffer not allocated".into())
                        })?;
                        let expert_weights_buf = s.moe_expert_weights.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE expert_weights buffer not allocated".into())
                        })?;

                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for MoE router".into())
                        })?;
                        enc.set_pipeline_state(router_softmax);
                        enc.set_buffer(&s.normed_buf, 0, 0);
                        enc.set_buffer(layer_buf, base_off + router.offset, 1);
                        enc.set_buffer(expert_ids_buf, 0, 2);
                        enc.set_buffer(expert_weights_buf, 0, 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
                        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
                        if use_biased {
                            enc.set_buffer(is_cached_buf.as_ref().unwrap(), 0, 7);
                            enc.set_bytes(&bias_lambda.to_le_bytes(), 8);
                        }
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
                        enc.end_encoding();
                    }

                    // Commit CB1 (attention + FFN norm + router) and wait.
                    cmd.commit();
                    cmd.wait_until_completed();

                    // Read back expert_ids from GPU.
                    let expert_ids_buf = s.moe_expert_ids.as_ref().unwrap();
                    let mut cpu_ids = vec![0u32; top_k];
                    expert_ids_buf.read_u32(&mut cpu_ids);

                    // Record expert activation in profiler.
                    if let Some(ref profiler) = self.expert_profiler {
                        profiler.lock().unwrap().record(layer_idx, &cpu_ids);
                    }

                    // Populate expert cache with selected experts from blob.
                    if let Some(ref cache_mutex) = self.expert_cache {
                        let blob = weights.as_bytes();
                        let mut cache = cache_mutex.lock().unwrap();
                        for &eid in &cpu_ids {
                            let key = (layer_idx, eid);
                            if !cache.contains(&key) {
                                let eid_usize = eid as usize;
                                if eid_usize < experts.len() {
                                    let expert_slice = &experts[eid_usize];
                                    let gate_start = expert_slice.gate.offset as usize;
                                    let gate_end = gate_start + expert_slice.gate.length as usize;
                                    let up_start = expert_slice.up.offset as usize;
                                    let up_end = up_start + expert_slice.up.length as usize;
                                    let down_start = expert_slice.down.offset as usize;
                                    let down_end = down_start + expert_slice.down.length as usize;

                                    if gate_end <= blob.len() && up_end <= blob.len() && down_end <= blob.len() {
                                        let mut data = Vec::with_capacity(
                                            expert_slice.gate.length as usize
                                            + expert_slice.up.length as usize
                                            + expert_slice.down.length as usize
                                        );
                                        data.extend_from_slice(&blob[gate_start..gate_end]);
                                        data.extend_from_slice(&blob[up_start..up_end]);
                                        data.extend_from_slice(&blob[down_start..down_end]);

                                        let local_slice = lumen_format::index::ExpertSlice {
                                            gate: lumen_format::index::TensorSlice {
                                                offset: 0,
                                                length: expert_slice.gate.length,
                                                quant: expert_slice.gate.quant,
                                            },
                                            up: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length,
                                                length: expert_slice.up.length,
                                                quant: expert_slice.up.quant,
                                            },
                                            down: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length + expert_slice.up.length,
                                                length: expert_slice.down.length,
                                                quant: expert_slice.down.quant,
                                            },
                                        };
                                        cache.insert(key, data, local_slice);
                                    }
                                }
                            }
                        }
                    }

                    // Phase 2: Assemble only top-K expert weights and dispatch.
                    // Try prefetch first, then cache, then ExpertReader, then full blob.

                    // Check if async prefetch from the previous layer produced
                    // results for THIS layer. Prefetch hit means zero-wait I/O.
                    let mut prefetched: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                        (0..top_k).map(|_| None).collect();
                    {
                        let mut ph_guard = self.prefetch_handle.lock().unwrap();
                        if let Some(ps) = ph_guard.take() {
                            if ps.target_layer == layer_idx {
                                // Join the prefetch thread (should be fast -- I/O already done).
                                match ps.handle.join() {
                                    Ok(results) => {
                                        for (eid, result) in results {
                                            if let Ok((data, slices)) = result {
                                                // Find which slot k this expert corresponds to.
                                                for (k, &cpu_eid) in cpu_ids.iter().enumerate() {
                                                    if cpu_eid == eid && prefetched[k].is_none() {
                                                        self.expert_bytes_from_disk.fetch_add(data.len() as u64, Ordering::Relaxed);
                                                        // Also insert into cache.
                                                        if let Some(ref cm) = self.expert_cache {
                                                            let mut c = cm.lock().unwrap();
                                                            c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                        }
                                                        prefetched[k] = Some((data, slices));
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        // Prefetch thread panicked -- ignore and fall back.
                                    }
                                }
                            }
                            // If target_layer didn't match, discard the stale prefetch.
                        }
                    }

                    let mut assembled = Vec::new();
                    let mut expert_gate_offs = Vec::with_capacity(top_k);
                    let mut expert_up_offs = Vec::with_capacity(top_k);
                    let mut expert_down_offs = Vec::with_capacity(top_k);
                    let mut assembly_ok = true;

                    // Two-pass assembly: prefetch hits + cache hits first, then disk for misses.
                    if let Some(ref cache_mutex) = self.expert_cache {
                        // Pass 1: collect from prefetch and cache, identify misses.
                        let mut per_expert: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                            (0..top_k).map(|_| None).collect();
                        let mut miss_indices: Vec<usize> = Vec::new();

                        // Use prefetch hits first.
                        for k in 0..top_k {
                            if let Some(data_slices) = prefetched[k].take() {
                                per_expert[k] = Some(data_slices);
                            }
                        }

                        {
                            let mut cache = cache_mutex.lock().unwrap();
                            for (k, &eid) in cpu_ids.iter().enumerate() {
                                if per_expert[k].is_some() {
                                    continue; // Already from prefetch
                                }
                                let key = (layer_idx, eid);
                                if let Some((data, slices)) = cache.get_with_slices(&key) {
                                    self.expert_bytes_from_cache.fetch_add(data.len() as u64, Ordering::Relaxed);
                                    per_expert[k] = Some((data.to_vec(), slices));
                                } else {
                                    miss_indices.push(k);
                                }
                            }
                        }

                        // Pass 2: load misses from disk.
                        //
                        // When MetalIOQueue is available, use Metal IO DMA to
                        // load expert weight byte ranges directly from NVMe SSD into a
                        // Metal shared buffer, bypassing CPU memory allocation and page
                        // faults. Falls back to load_experts_parallel (pread).
                        if !miss_indices.is_empty() {
                            let loaded_via_metal_io = if let (Some(ref io_queue), Some(ref lbc_path)) =
                                (&self.metal_io_queue, &self.lbc_path)
                            {
                                // Metal IO DMA path for cache misses.
                                // Build read plans using ExpertReader's validation,
                                // then load directly via MTLIOCommandQueue.
                                if let Some(ref reader_mutex) = self.expert_reader {
                                    let reader = reader_mutex.lock().unwrap();
                                    let layer_indices = reader.layer_indices();

                                    // Validate layer and build file offsets for each miss.
                                    let mut dma_plans: Vec<(usize, u32, u64, u64, u64, u64, u64, u64,
                                        lumen_format::quantization::QuantScheme,
                                        lumen_format::quantization::QuantScheme,
                                        lumen_format::quantization::QuantScheme)> = Vec::new();
                                    let mut plans_ok = true;

                                    if let Some(layer_idx_entry) = layer_indices.get(layer_idx) {
                                        let blob_off = layer_idx_entry.layer_offset_bytes;
                                        if let Some(ref expert_entries) = layer_idx_entry.subtensors.experts {
                                            for &k in &miss_indices {
                                                let eid = cpu_ids[k] as usize;
                                                if eid < expert_entries.len() {
                                                    let es = &expert_entries[eid];
                                                    dma_plans.push((
                                                        k, cpu_ids[k],
                                                        blob_off + es.gate.offset, es.gate.length,
                                                        blob_off + es.up.offset, es.up.length,
                                                        blob_off + es.down.offset, es.down.length,
                                                        es.gate.quant, es.up.quant, es.down.quant,
                                                    ));
                                                } else {
                                                    plans_ok = false;
                                                    break;
                                                }
                                            }
                                        } else {
                                            plans_ok = false;
                                        }
                                    } else {
                                        plans_ok = false;
                                    }
                                    drop(reader);

                                    if plans_ok && !dma_plans.is_empty() {
                                        // Compute total size and build DMA ranges.
                                        let mut total_dma_bytes: u64 = 0;
                                        let mut range_info: Vec<(usize, u32, u64, u64, u64, u64,
                                            lumen_format::quantization::QuantScheme,
                                            lumen_format::quantization::QuantScheme,
                                            lumen_format::quantization::QuantScheme)> = Vec::new();
                                        let mut io_ranges: Vec<(u64, u64, u64)> = Vec::new();

                                        for &(k, eid, gate_off, gate_len, up_off, up_len, down_off, down_len,
                                              gq, uq, dq) in &dma_plans
                                        {
                                            let base = total_dma_bytes;
                                            // gate
                                            io_ranges.push((base, gate_off, gate_len));
                                            // up
                                            io_ranges.push((base + gate_len, up_off, up_len));
                                            // down
                                            io_ranges.push((base + gate_len + up_len, down_off, down_len));
                                            let expert_total = gate_len + up_len + down_len;
                                            range_info.push((k, eid, base, gate_len, up_len, down_len, gq, uq, dq));
                                            total_dma_bytes += expert_total;
                                        }

                                        // Allocate a shared Metal buffer and DMA load all ranges.
                                        if total_dma_bytes > 0 {
                                            if let Some(dma_buf) = self.device.new_buffer(total_dma_bytes as usize) {
                                                match io_queue.load_ranges_sync(
                                                    &self.device, &dma_buf, lbc_path, &io_ranges,
                                                ) {
                                                    Ok(()) => {
                                                        // DMA succeeded. Read back into per_expert
                                                        // and populate cache.
                                                        let ptr = dma_buf.contents() as *const u8;
                                                        for &(k, eid, base, gate_len, up_len, down_len, gq, uq, dq) in &range_info {
                                                            let expert_total = (gate_len + up_len + down_len) as usize;
                                                            let mut data = vec![0u8; expert_total];
                                                            unsafe {
                                                                std::ptr::copy_nonoverlapping(
                                                                    ptr.add(base as usize),
                                                                    data.as_mut_ptr(),
                                                                    expert_total,
                                                                );
                                                            }
                                                            let slices = lumen_format::index::ExpertSlice {
                                                                gate: lumen_format::index::TensorSlice {
                                                                    offset: 0,
                                                                    length: gate_len,
                                                                    quant: gq,
                                                                },
                                                                up: lumen_format::index::TensorSlice {
                                                                    offset: gate_len,
                                                                    length: up_len,
                                                                    quant: uq,
                                                                },
                                                                down: lumen_format::index::TensorSlice {
                                                                    offset: gate_len + up_len,
                                                                    length: down_len,
                                                                    quant: dq,
                                                                },
                                                            };
                                                            self.expert_bytes_from_disk.fetch_add(expert_total as u64, Ordering::Relaxed);
                                                            if let Some(ref cm) = self.expert_cache {
                                                                let mut c = cm.lock().unwrap();
                                                                c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                            }
                                                            per_expert[k] = Some((data, slices));
                                                        }
                                                        true // Successfully loaded via Metal IO
                                                    }
                                                    Err(_) => false, // Fall back to pread
                                                }
                                            } else {
                                                false // Buffer allocation failed
                                            }
                                        } else {
                                            true // No bytes to load
                                        }
                                    } else {
                                        false // Plan validation failed
                                    }
                                } else {
                                    false // No expert reader
                                }
                            } else {
                                false // No Metal IO queue or no LBC path
                            };

                            // Fallback: load misses via ExpertReader pread.
                            if !loaded_via_metal_io {
                                if let Some(ref reader_mutex) = self.expert_reader {
                                    let reader = reader_mutex.lock().unwrap();
                                    let requests: Vec<(usize, u32)> = miss_indices
                                        .iter()
                                        .map(|&k| (layer_idx, cpu_ids[k]))
                                        .collect();
                                    let results = reader.load_experts_parallel(&requests);
                                    drop(reader);
                                    for (ri, &k) in miss_indices.iter().enumerate() {
                                        let eid = cpu_ids[k];
                                        match &results[ri] {
                                            Ok((data, slices)) => {
                                                self.expert_bytes_from_disk.fetch_add(data.len() as u64, Ordering::Relaxed);
                                                if let Some(ref cm) = self.expert_cache {
                                                    let mut c = cm.lock().unwrap();
                                                    c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                }
                                                per_expert[k] = Some((data.clone(), slices.clone()));
                                            }
                                            Err(_) => {
                                                assembly_ok = false;
                                                break;
                                            }
                                        }
                                    }
                                } else {
                                    assembly_ok = false;
                                }
                            }
                        }

                        // Assemble buffer from collected per-expert data.
                        if assembly_ok {
                            for k in 0..top_k {
                                if let Some((ref data, ref slices)) = per_expert[k] {
                                    let base = assembled.len() as u64;
                                    expert_gate_offs.push(base + slices.gate.offset);
                                    expert_up_offs.push(base + slices.up.offset);
                                    expert_down_offs.push(base + slices.down.offset);
                                    assembled.extend_from_slice(data);
                                } else {
                                    assembly_ok = false;
                                    break;
                                }
                            }
                        }
                    } else {
                        assembly_ok = false;
                    }

                    // If assembly from cache/reader succeeded for ALL top-K experts:
                    if assembly_ok && expert_gate_offs.len() == top_k {
                        let moe_meta = CachedMoeMeta {
                            router_weight_off: base_off + router.offset,
                            expert_gate_offs,
                            expert_up_offs,
                            expert_down_offs,
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        };

                        let cmd2 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create CB2 for Option A expert FFNs".into())
                        })?;
                        match self.device.new_buffer_with_bytes(&assembled) {
                            Some(ewb) => {
                                Self::encode_moe_ffn_decode(
                                    &cmd2, pipelines, s, layer_buf, &moe_meta,
                                    None,
                                    Some(&ewb),
                                    None, 0.0,
                                    Some(&cpu_ids),
                                    true,  // Skip router (already ran in CB1)
                                    None,  // No per-layer routing weights in streaming
                                )?;
                                cmd2.commit();
                                cmd2.wait_until_completed();
                            }
                            None => {
                                // Buffer creation failed -- fall back to full blob in CB2.
                                let moe_meta_blob = CachedMoeMeta {
                                    router_weight_off: base_off + router.offset,
                                            expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                                    expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                                    expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                                    expert_gate_quant: first.gate.quant,
                                            expert_down_quant: first.down.quant,
                                };
                                Self::encode_moe_ffn_decode(
                                    &cmd2, pipelines, s, layer_buf, &moe_meta_blob,
                                    None, None,
                                    None, 0.0,
                                    Some(&cpu_ids),
                                    true,
                                    None,  // No per-layer routing weights in streaming
                                )?;
                                cmd2.commit();
                                cmd2.wait_until_completed();
                            }
                        }
                    } else {
                        // Fallback: assemble from full blob for top-K only.
                        let moe_meta_blob = CachedMoeMeta {
                            router_weight_off: base_off + router.offset,
                            expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                            expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                            expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        };
                        let expert_blob_bytes: u64 = cpu_ids.iter()
                            .filter_map(|&eid| experts.get(eid as usize))
                            .map(|e| (e.gate.length + e.up.length + e.down.length) as u64)
                            .sum();
                        self.expert_bytes_from_blob.fetch_add(expert_blob_bytes, Ordering::Relaxed);

                        let cmd2 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create CB2 for Option A fallback".into())
                        })?;
                        Self::encode_moe_ffn_decode(
                            &cmd2, pipelines, s, layer_buf, &moe_meta_blob,
                            None, None,
                            None, 0.0,
                            Some(&cpu_ids),
                            true,  // Skip router
                            None,  // No per-layer routing weights in streaming
                        )?;
                        cmd2.commit();
                        cmd2.wait_until_completed();
                    }

                    // Dispatch shared expert (always-active) for Qwen3.5-MoE.
                    // The shared expert runs on every token in addition to the top-K
                    // routed experts. Its output is added to x_buf after the routed
                    // expert accumulation. Uses CB3 since CB2 is already committed.
                    if let (Some(se_gate), Some(se_up), Some(se_down)) = (
                        &st.shared_expert_gate, &st.shared_expert_up, &st.shared_expert_down,
                    ) {
                        let cmd3 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute(
                                "Failed to create CB3 for Option A shared expert".into(),
                            )
                        })?;
                        Self::encode_shared_expert_ffn_decode_raw(
                            &cmd3, pipelines, s, layer_buf,
                            base_off + se_gate.offset,
                            base_off + se_up.offset,
                            base_off + se_down.offset,
                            se_gate.quant,
                            se_down.quant,
                            st.ffn_gate_inp_shexp.map(|fgis| base_off + fgis.offset),
                        )?;
                        cmd3.commit();
                        cmd3.wait_until_completed();
                    }

                    // Check if profiling phase is complete and trigger warmup.
                    self.maybe_trigger_warmup();

                    // Spawn async prefetch for layer N+1's experts.
                    // Speculative: assume layer N+1 will select the same experts.
                    // If not, the prefetch result is discarded and fallback to sync load.
                    // Uses load_experts_parallel for concurrent pread with
                    // F_NOCACHE file descriptors, increasing NVMe queue depth.
                    let next_layer = layer_idx + 1;
                    if next_layer < s.num_layers && self.lbc_path.is_some() {
                        let lbc_path = self.lbc_path.as_ref().unwrap().clone();
                        let prefetch_ids = cpu_ids.clone();
                        let prefetch_layer = next_layer;

                        let handle = std::thread::spawn(move || {
                            let mut results = Vec::new();
                            match ExpertReader::open(&lbc_path) {
                                Ok(reader) => {
                                    let requests: Vec<(usize, u32)> = prefetch_ids
                                        .iter()
                                        .map(|&eid| (prefetch_layer, eid))
                                        .collect();
                                    let par_results = reader.load_experts_parallel(&requests);
                                    for (i, result) in par_results.into_iter().enumerate() {
                                        results.push((prefetch_ids[i], result));
                                    }
                                }
                                Err(_) => {
                                    // Reader open failed -- return empty results.
                                }
                            }
                            results
                        });

                        let ps = PrefetchState {
                            target_layer: next_layer,
                            expert_ids: cpu_ids,
                            handle,
                        };
                        *self.prefetch_handle.lock().unwrap() = Some(ps);
                    }

                    // Mark that we already committed synchronously -- clear last_async_cmd.
                    s.last_async_cmd = None;
                    s.gpu_x_valid = true;
                    kv.seq_len = new_seq_len;
                    return Ok(());
                }

                // ==============================================================
                // Option B: Original full-dispatch path (all experts)
                // ==============================================================
                let cache_assembled_buf: Option<MetalBuffer> = if let Some(ref cache_mutex) = self.expert_cache {
                    let mut cache = cache_mutex.lock().unwrap();

                    // Classify each expert as cached or not.
                    let mut cached_mask = vec![false; num_experts];
                    let mut num_cached = 0usize;
                    for e in 0..num_experts {
                        if cache.contains(&(layer_idx, e as u32)) {
                            cached_mask[e] = true;
                            num_cached += 1;
                        }
                    }

                    // Build is_cached GPU buffer for biased routing.
                    // Only created when cache_bias_lambda > 0 and warmup is complete.
                    let is_cached_buf: Option<MetalBuffer> = if self.cache_bias_lambda > 0.0
                        && self.warmup_complete.load(Ordering::Relaxed)
                    {
                        let is_cached_data: Vec<u8> = cached_mask.iter()
                            .map(|&c| if c { 1u8 } else { 0u8 })
                            .collect();
                        self.device.new_buffer_with_bytes(&is_cached_data)
                    } else {
                        None
                    };
                    let bias_lambda = self.cache_bias_lambda;

                    if num_cached == num_experts {
                        // Tier 1: All experts cached -- assemble from cache only.
                        let mut assembled = Vec::new();
                        let mut expert_gate_offs = Vec::with_capacity(num_experts);
                        let mut expert_up_offs = Vec::with_capacity(num_experts);
                        let mut expert_down_offs = Vec::with_capacity(num_experts);
                        let mut ok = true;

                        for eid in 0..num_experts {
                            let key = (layer_idx, eid as u32);
                            if let Some((data, slices)) = cache.get_with_slices(&key) {
                                let base = assembled.len() as u64;
                                expert_gate_offs.push(base + slices.gate.offset);
                                expert_up_offs.push(base + slices.up.offset);
                                expert_down_offs.push(base + slices.down.offset);
                                assembled.extend_from_slice(&data);
                            } else {
                                ok = false;
                                break;
                            }
                        }

                        if ok {
                            // Tier 1 -- all bytes from cache.
                            self.expert_bytes_from_cache.fetch_add(
                                assembled.len() as u64, Ordering::Relaxed,
                            );
                            let moe_meta = CachedMoeMeta {
                                router_weight_off: base_off + router.offset,
                                    expert_gate_offs,
                                expert_up_offs,
                                expert_down_offs,
                                expert_gate_quant: first.gate.quant,
                                    expert_down_quant: first.down.quant,
                            };
                            drop(cache);
                            match self.device.new_buffer_with_bytes(&assembled) {
                                Some(buf) => {
                                    Self::encode_moe_ffn_decode(
                                        &cmd, pipelines, s, layer_buf, &moe_meta,
                                        None,
                                        Some(&buf),
                                        is_cached_buf.as_ref(), bias_lambda,
                                        None,  // Option A handled below after readback
                                        false, // Do not skip router
                                        None,  // No per-layer routing weights in streaming
                                    )?;
                                    Some(buf)
                                }
                                None => None,
                            }
                        } else {
                            None
                        }
                    } else if num_cached > 0 && self.expert_reader.is_some() {
                        // Tier 2: Some experts cached, load misses via ExpertReader.
                        // This avoids reading the full layer blob for a partial hit.
                        let mut assembled = Vec::new();
                        let mut expert_gate_offs = Vec::with_capacity(num_experts);
                        let mut expert_up_offs = Vec::with_capacity(num_experts);
                        let mut expert_down_offs = Vec::with_capacity(num_experts);
                        let mut ok = true;
                        let mut tier2_cache_bytes = 0u64;
                        let mut tier2_disk_bytes = 0u64;

                        // Collect cached experts first (holding cache lock).
                        let mut per_expert: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                            (0..num_experts).map(|_| None).collect();
                        for eid in 0..num_experts {
                            if cached_mask[eid] {
                                let key = (layer_idx, eid as u32);
                                if let Some((data, slices)) = cache.get_with_slices(&key) {
                                    tier2_cache_bytes += data.len() as u64;
                                    per_expert[eid] = Some((data.to_vec(), slices));
                                }
                            }
                        }
                        drop(cache); // Release cache lock before disk I/O.

                        // Load missing experts via ExpertReader.
                        {
                            let mut reader = self.expert_reader.as_ref().unwrap().lock().unwrap();
                            for eid in 0..num_experts {
                                if per_expert[eid].is_none() {
                                    match reader.load_expert(layer_idx, eid as u32) {
                                        Ok((data, slices)) => {
                                            tier2_disk_bytes += data.len() as u64;
                                            // Insert into cache for future tokens.
                                            if let Some(ref cm) = self.expert_cache {
                                                let mut c = cm.lock().unwrap();
                                                c.insert((layer_idx, eid as u32), data.clone(), slices.clone());
                                            }
                                            per_expert[eid] = Some((data, slices));
                                        }
                                        Err(_e) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // Tier 2 I/O counters.
                        self.expert_bytes_from_cache.fetch_add(tier2_cache_bytes, Ordering::Relaxed);
                        self.expert_bytes_from_disk.fetch_add(tier2_disk_bytes, Ordering::Relaxed);

                        if ok {
                            for eid in 0..num_experts {
                                let (ref data, ref slices) = per_expert[eid].as_ref().unwrap();
                                let base = assembled.len() as u64;
                                expert_gate_offs.push(base + slices.gate.offset);
                                expert_up_offs.push(base + slices.up.offset);
                                expert_down_offs.push(base + slices.down.offset);
                                assembled.extend_from_slice(data);
                            }

                            let moe_meta = CachedMoeMeta {
                                router_weight_off: base_off + router.offset,
                                    expert_gate_offs,
                                expert_up_offs,
                                expert_down_offs,
                                expert_gate_quant: first.gate.quant,
                                    expert_down_quant: first.down.quant,
                            };
                            match self.device.new_buffer_with_bytes(&assembled) {
                                Some(buf) => {
                                    Self::encode_moe_ffn_decode(
                                        &cmd, pipelines, s, layer_buf, &moe_meta,
                                        None,
                                        Some(&buf),
                                        is_cached_buf.as_ref(), bias_lambda,
                                        None,  // Option A handled below after readback
                                        false, // Do not skip router
                                        None,  // No per-layer routing weights in streaming
                                    )?;
                                    Some(buf)
                                }
                                None => None,
                            }
                        } else {
                            None // ExpertReader failed, fall back to full blob
                        }
                    } else {
                        None // No cached experts or no reader
                    }
                } else {
                    None // No expert cache configured
                };

                // Tier 3 fallback: expert weights from full layer blob.
                if cache_assembled_buf.is_none() {
                    // Tier 3 -- all expert bytes from full blob.
                    let expert_blob_bytes: u64 = experts.iter()
                        .map(|e| (e.gate.length + e.up.length + e.down.length) as u64)
                        .sum();
                    self.expert_bytes_from_blob.fetch_add(expert_blob_bytes, Ordering::Relaxed);

                    let moe_meta = CachedMoeMeta {
                        router_weight_off: base_off + router.offset,
                        expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                        expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                        expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                        expert_gate_quant: first.gate.quant,
                        expert_down_quant: first.down.quant,
                    };
                    Self::encode_moe_ffn_decode(
                        &cmd, pipelines, s, layer_buf, &moe_meta,
                        None,
                        None,
                        None, 0.0,  // No cache bias in Tier 3 fallback
                        None,  // No Option A in Tier 3 fallback
                        false, // Do not skip router
                        None,  // No per-layer routing weights in streaming
                    )?;
                }

                true
            } else {
                return Err(RuntimeError::Compute(
                    "Model has num_experts > 0 but layer is missing router_weight/experts \
                     in SubtensorOffsets. The LBC file may need re-conversion.".into()
                ));
            }
        } else {
            false
        };

        // --- Dense FFN path: only executed for non-MoE layers ---
        if !moe_handled {

        // --- Fused Gate + Up + SwiGLU (single dispatch for Q8_0, deferred reduction) ---
        // Prefill uses original fused kernel (batched GEMM style, 4 rows/TG)
        if st.w_gate.quant == QuantScheme::Q8_0 && st.w_up.quant == QuantScheme::Q8_0 {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0);
            enc.set_buffer(layer_buf, w_gate_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.gate_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, w_up_off, 5);
            enc.dispatch_threadgroups(
                MTLSize::new(((inter_dim as u64) + 3) / 4, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        } else if st.w_gate.quant == QuantScheme::Q4_0 && st.w_up.quant == QuantScheme::Q4_0 {
            // Q4_0: fused Gate + Up + SwiGLU with deferred reduction (1 row/TG)
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
            enc.set_buffer(layer_buf, w_gate_off, 0);    // gate weights
            enc.set_buffer(&s.normed_buf, 0, 1);         // normed input x
            enc.set_buffer(&s.gate_buf, 0, 2);           // output (SwiGLU result)
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, w_up_off, 5);      // up weights
            enc.dispatch_threadgroups(
                MTLSize::new(inter_dim as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        } else {
            // Fallback: separate Gate + Up + SwiGLU for F32
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                let in_dim_u32 = hidden_dim as u32;
                enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                enc.set_buffer(layer_buf, w_gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.gate_buf, 0, 2);
                enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
                enc.dispatch_threadgroups(
                    MTLSize::new(inter_dim as u64, 1, 1),
                    MTLSize::new(matmul_tg_size, 1, 1),
                );
                enc.set_buffer(layer_buf, w_up_off, 0);
                enc.set_buffer(&s.up_buf, 0, 2);
                enc.dispatch_threadgroups(
                    MTLSize::new(inter_dim as u64, 1, 1),
                    MTLSize::new(matmul_tg_size, 1, 1),
                );
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                enc.set_pipeline_state(&pipelines.swiglu);
                enc.set_buffer(&s.gate_buf, 0, 0);
                enc.set_buffer(&s.up_buf, 0, 1);
                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                let tg = 256u64.min(inter_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                enc.end_encoding();
            }
        }

                // --- Encoder 11: Down projection + Residual 2 (fused) ---
        // x_buf = W_down * gate + attn_proj_buf  (eliminates separate add_write)
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let tg_down = match st.w_down.quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, w_down_off, 0);
            enc.set_buffer(&s.gate_buf, 0, 1);
            enc.set_buffer(&s.x_buf, 0, 2);  // write directly to x_buf
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
            enc.set_buffer(&s.attn_proj_buf, 0, 4);  // residual input
            if matches!(st.w_down.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let down_out_dim_u32 = hidden_dim as u32;
                enc.set_bytes(&down_out_dim_u32.to_le_bytes(), 5);
            }
            let n_tg_down = match st.w_down.quant {
                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                _ => hidden_dim as u64,
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_down, 1, 1),
                MTLSize::new(tg_down, 1, 1),
            );
            enc.end_encoding();
        }

        } // end if !moe_handled (dense FFN path)

        // For MoE layers in streaming mode: commit synchronously so we can
        // read back expert_ids from the GPU and record activation patterns.
        // In streaming mode, I/O dominates latency, so the sync wait cost
        // is negligible. For non-MoE layers, continue with async commit.
        let moe_sync_needed = moe_handled
            && !s.gpu_unified_weight_buf.is_some()
            && !s.gpu_resident_layers.is_some()
            && self.expert_profiler.is_some();

        if moe_sync_needed {
            cmd.commit();
            cmd.wait_until_completed();

            // Read back expert_ids from GPU buffer and record in profiler.
            let top_k = s.moe_num_active_experts;
            if let Some(ref expert_ids_buf) = s.moe_expert_ids {
                let mut ids = vec![0u32; top_k];
                expert_ids_buf.read_u32(&mut ids);

                // Drop scratch guard before locking profiler to avoid
                // holding two locks simultaneously (even though they can't
                // deadlock, it's cleaner).
                // Actually, we still need `s` for the rest of this function,
                // so just lock the profiler while holding scratch. The profiler
                // mutex is always locked after scratch, so ordering is consistent.
                if let Some(ref profiler) = self.expert_profiler {
                    profiler.lock().unwrap().record(layer_idx, &ids);
                }

                // Populate the expert LFU cache with expert weight data from
                // the current layer blob. On future tokens, these cached weights
                // will enable selective expert loading.
                if let Some(ref cache_mutex) = self.expert_cache {
                    let blob = weights.as_bytes();
                    if let Some(ref experts) = weights.subtensors.experts {
                        let mut cache = cache_mutex.lock().unwrap();
                        for &eid in &ids {
                            let key = (layer_idx, eid);
                            if !cache.contains(&key) {
                                let eid_usize = eid as usize;
                                if eid_usize < experts.len() {
                                    let expert_slice = &experts[eid_usize];
                                    // Extract gate+up+down bytes from the layer blob.
                                    let gate_start = expert_slice.gate.offset as usize;
                                    let gate_end = gate_start + expert_slice.gate.length as usize;
                                    let up_start = expert_slice.up.offset as usize;
                                    let up_end = up_start + expert_slice.up.length as usize;
                                    let down_start = expert_slice.down.offset as usize;
                                    let down_end = down_start + expert_slice.down.length as usize;

                                    if gate_end <= blob.len() && up_end <= blob.len() && down_end <= blob.len() {
                                        let mut data = Vec::with_capacity(
                                            expert_slice.gate.length as usize
                                            + expert_slice.up.length as usize
                                            + expert_slice.down.length as usize
                                        );
                                        data.extend_from_slice(&blob[gate_start..gate_end]);
                                        data.extend_from_slice(&blob[up_start..up_end]);
                                        data.extend_from_slice(&blob[down_start..down_end]);

                                        // Build local ExpertSlice with offsets relative to
                                        // the concatenated data buffer.
                                        let local_slice = lumen_format::index::ExpertSlice {
                                            gate: lumen_format::index::TensorSlice {
                                                offset: 0,
                                                length: expert_slice.gate.length,
                                                quant: expert_slice.gate.quant,
                                            },
                                            up: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length,
                                                length: expert_slice.up.length,
                                                quant: expert_slice.up.quant,
                                            },
                                            down: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length + expert_slice.up.length,
                                                length: expert_slice.down.length,
                                                quant: expert_slice.down.quant,
                                            },
                                        };
                                        cache.insert(key, data, local_slice);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check if profiling phase is complete and trigger cache warmup.
            self.maybe_trigger_warmup();

            // Clear last_async_cmd since we already waited.
            s.last_async_cmd = None;
        } else {
            // Async commit: GPU processes this layer while CPU prepares the next.
            // Metal queue is FIFO — layers execute in order, each completing before
            // the next starts. No CPU wait needed between layers.
            cmd.commit();

            // Store command buffer reference so we can wait on it at the start of the
            // next forward pass (when layer_idx == 0 needs to write fresh embeddings).
            // Drop the previous one first — if it hasn't completed, Metal still tracks it.
            s.last_async_cmd = Some(cmd);
        }

        // Mark GPU activation as valid. compute_final will skip CPU→GPU upload.
        s.gpu_x_valid = true;

        // Update KV cache seq_len (CPU tracking only — GPU KV cache is authoritative).
        // Skip reading K,V back to CPU since the GPU KV cache has the data and
        // the CPU KV cache data is never read during Metal inference.
        kv.seq_len = new_seq_len;

        Ok(())
    }

    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized".into())
        })?;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized".into())
        })?;

        let hidden_dim = s.hidden_dim;
        let vocab_size = s.vocab_size;
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;

        // Drain any pending async GPU work before CPU reads/writes.
        if let Some(prev_cmd) = s.last_async_cmd.take() {
            prev_cmd.wait_until_completed();
        }

        // GPU activation persistence: skip CPU→GPU upload if GPU already has valid data
        if !s.gpu_x_valid {
            let x_f32 = x.as_f32_slice();
            s.x_buf.write_f32(x_f32);
        }
        // Reset for next token (embed_token creates fresh CPU buffer)
        s.gpu_x_valid = false;

        // Resolve global tensor buffers: prefer unified buffer, fall back to separate
        let (norm_buf_ref, norm_off): (&MetalBuffer, u64) =
            if let Some((_, norm_o, _)) = s.gpu_global_offsets {
                let ubuf = s.gpu_unified_weight_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Unified buffer missing but global offsets set".into())
                })?;
                (ubuf, norm_o as u64)
            } else {
                let fnb = self.final_norm_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Final norm buffer not initialized".into())
                })?;
                (fnb, 0u64)
            };
        let (proj_buf_ref, proj_off): (&MetalBuffer, u64) =
            if let Some((_, _, proj_o)) = s.gpu_global_offsets {
                let ubuf = s.gpu_unified_weight_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Unified buffer missing but global offsets set".into())
                })?;
                (ubuf, proj_o as u64)
            } else {
                let opb = self.output_proj_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Output proj buffer not initialized".into())
                })?;
                (opb, 0u64)
            };

        // Single command buffer for final RMSNorm + logits projection (2 encoders, 1 sync)

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for compute_final".into())
        })?;

        // --- Encoder 1: Final RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for final rmsnorm".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(norm_buf_ref, norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 2: Logits projection (deferred NR0=2 for Q8_0, NR0=4 for others, 128 threads) ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for logits projection".into())
            })?;
            let in_dim_u32 = hidden_dim as u32;
            let (proj_tg, proj_rows_per_tg) = match self.output_proj_quant {
                QuantScheme::Q8_0 => {
                    enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::Q4_0 => {
                    enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::F16 => {
                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                    (128u64, 2u64)
                }
                _ => {
                    enc.set_pipeline_state(&pipelines.matmul_f32_deferred);
                    (128u64, 4u64)
                }
            };
            enc.set_buffer(proj_buf_ref, proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(proj_tg, 1, 1),
            );
            enc.end_encoding();
        }

        cmd.commit_and_wait();

        // Read logits from GPU
        let mut logits_data = vec![0.0f32; vocab_size];
        s.logits_buf.read_f32(&mut logits_data);

        Ok(Logits { data: logits_data })
    }

    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        let hidden_dim = self.cached_hidden_dim;
        let vocab_size = self.cached_vocab_size;
        let tid = token_id as usize;

        if tid >= vocab_size {
            return Err(RuntimeError::Compute(format!(
                "token_id {tid} out of range (vocab_size={vocab_size})",
            )));
        }

        // GPU path: dispatch embed_token kernel to write directly to x_buf.
        // This eliminates the CPU lookup + CPU->GPU upload roundtrip per decode token.
        // The returned ActivationBuffer is a placeholder -- compute_layer will use
        // the GPU-resident x_buf directly since gpu_x_valid is set to true.
        if let (Some(_embedding_buf), Some(pipelines)) = (&self.embedding_buf, &self.pipelines) {
            let mut scratch_guard = self.scratch.lock().unwrap();
            if let Some(s) = scratch_guard.as_mut() {
                // Drain any pending async GPU work before writing to x_buf.
                if let Some(prev_cmd) = s.last_async_cmd.take() {
                    prev_cmd.wait_until_completed();
                }

                // Resolve embedding buffer: prefer unified, fall back to separate
                let (embed_buf_ref, embed_off): (&MetalBuffer, u64) =
                    if let Some((emb_o, _, _)) = s.gpu_global_offsets {
                        (s.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
                    } else {
                        (self.embedding_buf.as_ref().unwrap(), 0u64)
                    };

                let token_id_u32 = token_id;
                let hidden_dim_u32 = hidden_dim as u32;

                let cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create command buffer for embed_token".into())
                })?;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for embed_token".into())
                })?;

                // Select the appropriate kernel based on embedding quantization
                match self.embedding_quant {
                    QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0),
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0),
                    _ => enc.set_pipeline_state(&pipelines.embed_token),
                }
                enc.set_buffer(embed_buf_ref, embed_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_bytes(&token_id_u32.to_le_bytes(), 2);
                enc.set_bytes(&hidden_dim_u32.to_le_bytes(), 3);
                let tg = 256u64.min(hidden_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                enc.end_encoding();
                // Async commit: Metal queue FIFO guarantees this embed kernel
                // completes before compute_layer's command buffer executes.
                cmd.commit();

                // Store as last_async_cmd so it gets drained when needed.
                s.last_async_cmd = Some(cmd);

                // Mark GPU x_buf as valid -- compute_layer will skip CPU->GPU upload.
                s.gpu_x_valid = true;

                // Return a placeholder ActivationBuffer. The engine passes this to
                // compute_layer, but with gpu_x_valid=true the data is ignored.
                return Ok(ActivationBuffer {
                    data: vec![0u8; hidden_dim * 4],
                    num_elements: hidden_dim,
                    dtype: ComputeDtype::F32,
                });
            }
        }

        // CPU fallback: used when Metal buffers are not yet initialized.
        let start = tid * hidden_dim;
        let end = start + hidden_dim;
        let embed = &self.embedding[start..end];

        let mut data = Vec::with_capacity(embed.len() * 4);
        data.reserve(embed.len() * 4);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    embed.as_ptr() as *const u8,
                    data.as_mut_ptr(),
                    embed.len() * 4,
                );
                data.set_len(embed.len() * 4);
            }
        }
        #[cfg(target_endian = "big")]
        {
            for &v in embed {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }

        Ok(ActivationBuffer {
            data,
            num_elements: embed.len(),
            dtype: ComputeDtype::F32,
        })
    }

    // ====================================================================
    // Phase 3: enriched trait methods for GPU fast paths.
    // ====================================================================

    fn caps(&self) -> BackendCaps {
        BackendCaps {
            batched_prefill: true,
            gpu_resident: true,
            gdn: true,
            moe: true,
            gpu_argmax: true,
        }
    }

    fn set_global_tensors(
        &mut self,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        output_proj: Vec<f32>,
    ) {
        self.embedding = embedding;
        self.final_norm = final_norm;
        self.output_proj = output_proj;
    }

    fn set_output_proj_raw(&mut self, raw: Vec<u8>, quant: QuantScheme) {
        self.output_proj_quant = quant;
        self.output_proj_raw = Some(raw);
    }

    fn set_embedding_raw(&mut self, raw: Vec<u8>, quant: QuantScheme) {
        self.embedding_quant = quant;
        self.embedding_raw = Some(raw);
    }

    fn set_weight_tying(&mut self, enabled: bool) {
        self.weight_tying = enabled;
    }

    fn prefill(
        &self,
        tokens: &[u32],
        weights: &dyn WeightProvider,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        // Delegate to the existing inherent method.
        MetalF32Backend::prefill(self, tokens, weights, kv)
    }

    fn preload_weights(
        &mut self,
        weights: &dyn WeightProvider,
    ) -> Result<(), RuntimeError> {
        self.preload_weights_gpu_resident(weights)
    }

    fn decode_token(
        &self,
        token_id: u32,
        weights: &dyn WeightProvider,
        kv: &mut KvCache,
    ) -> Result<Logits, RuntimeError> {
        self.decode_token_single_cb(token_id, weights, kv)
    }

    fn decode_token_greedy(
        &self,
        token_id: u32,
        weights: &dyn WeightProvider,
        kv: &mut KvCache,
    ) -> Result<u32, RuntimeError> {
        MetalF32Backend::decode_token_greedy(self, token_id, weights, kv)
    }

    fn reset_recurrent_state(&self) {
        self.reset_gdn_state();
    }
}
