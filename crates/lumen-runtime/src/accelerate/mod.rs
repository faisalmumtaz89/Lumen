//! Accelerate-based batched prefill backend for macOS.
//!
//! Exploits Apple's AMX coprocessor via `cblas_sgemm` for matrix-matrix
//! multiplies during prompt processing. Weights are dequantized from Q8_0
//! to F32 per-layer into a reusable buffer, then the entire token batch
//! is projected at once.
//!
//! This is used ONLY for prefill (prompt processing). Decode (token-at-a-time
//! generation) continues to use the SIMD Q8_0 mat-vec backend which is
//! memory-bandwidth-optimal for single-token inference.

pub(crate) mod ffi;

use self::ffi::{cblas_sgemm, CblasOrder, CblasTranspose};
use crate::weight::cache::{LayerView, WeightProvider};
use crate::error::RuntimeError;
use crate::kv::{KvCache, KvCacheView};
use crate::compute::simd_kernels;
use crate::thread_pool::ThreadPool;
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;

/// Q8_0 quantization constants (must match simd_kernels).
const Q8_0_GROUP_SIZE: usize = 32;
const Q8_0_BLOCK_SIZE: usize = 34; // 2 bytes f16 scale + 32 bytes int8

/// Batched prefill backend using Apple Accelerate for AMX-accelerated GEMM.
///
/// Holds pre-allocated scratch buffers sized for the maximum batch (prompt length).
/// Created once before inference, used for prefill, then discarded.
pub struct AccelerateBatchBackend {
    // Global tensors (shared with SIMD backend)
    embedding: Vec<f32>,
    #[allow(dead_code)]
    final_norm: Vec<f32>,

    // Reusable dequantization buffer — sized for the largest weight matrix per layer.
    // For LLaMA: w_gate/w_up are [inter_dim x hidden_dim], which is the largest.
    dequant_buf: Vec<f32>,

    // Batch scratch buffers — all sized for [max_batch * dim]
    x_batch: Vec<f32>,      // [batch, hidden_dim] — input activations
    normed_batch: Vec<f32>,  // [batch, hidden_dim] — after RMSNorm
    q_batch: Vec<f32>,       // [batch, q_dim]
    k_batch: Vec<f32>,       // [batch, kv_dim]
    v_batch: Vec<f32>,       // [batch, kv_dim]
    attn_out_batch: Vec<f32>,// [batch, q_dim] — attention output per token
    attn_proj_batch: Vec<f32>,// [batch, hidden_dim] — after Wo projection
    gate_batch: Vec<f32>,    // [batch, inter_dim]
    up_batch: Vec<f32>,      // [batch, inter_dim]
    down_batch: Vec<f32>,    // [batch, hidden_dim]
    scores: Vec<f32>,        // [num_heads, max_seq_len] — for per-token attention

    // RoPE tables
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,

    // Cached dimensions
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    half_dim: usize,
    inter_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    gqa_ratio: usize,
    eps: f32,
    attn_scale: f32,
    max_seq_len: usize,

    // Thread pool for parallel dequantization and attention
    pool: ThreadPool,
}

impl AccelerateBatchBackend {
    /// Create a new batched prefill backend.
    ///
    /// `max_batch` is the maximum number of tokens in a single prefill batch
    /// (typically the prompt length).
    pub fn new(
        hp: &ModelHyperparams,
        max_batch: usize,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
    ) -> Self {
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let max_seq_len = hp.max_seq_len as usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let gqa_ratio = num_heads / num_kv_heads;
        let half_dim = head_dim / 2;
        let eps = hp.norm_eps;

        // RoPE tables
        let theta = hp.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let mut rope_cos = vec![0.0f32; max_seq_len * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + i] = angle.cos();
                rope_sin[pos * half_dim + i] = angle.sin();
            }
        }

        // Dequant buffer: sized for the largest weight matrix.
        // w_gate and w_up are [inter_dim x hidden_dim] — the biggest per-layer matrices.
        let max_weight_elems = inter_dim * hidden_dim;

        AccelerateBatchBackend {
            embedding,
            final_norm,
            dequant_buf: vec![0.0f32; max_weight_elems],
            x_batch: vec![0.0f32; max_batch * hidden_dim],
            normed_batch: vec![0.0f32; max_batch * hidden_dim],
            q_batch: vec![0.0f32; max_batch * q_dim],
            k_batch: vec![0.0f32; max_batch * kv_dim],
            v_batch: vec![0.0f32; max_batch * kv_dim],
            attn_out_batch: vec![0.0f32; max_batch * q_dim],
            attn_proj_batch: vec![0.0f32; max_batch * hidden_dim],
            gate_batch: vec![0.0f32; max_batch * inter_dim],
            up_batch: vec![0.0f32; max_batch * inter_dim],
            down_batch: vec![0.0f32; max_batch * hidden_dim],
            scores: vec![0.0f32; num_heads * max_seq_len],
            rope_cos,
            rope_sin,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            half_dim,
            inter_dim,
            q_dim,
            kv_dim,
            gqa_ratio,
            eps,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            max_seq_len,
            pool: ThreadPool::with_default_threads(),
        }
    }

    /// Run batched prefill: process all prompt tokens through all layers.
    ///
    /// Returns the final hidden state of the LAST token (used to produce
    /// the first generation logit).
    pub fn prefill(
        &mut self,
        prompt_tokens: &[u32],
        weights: &dyn WeightProvider,
        kv: &mut KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        let batch_size = prompt_tokens.len();
        if batch_size == 0 {
            return Err(RuntimeError::Compute("empty prompt".into()));
        }
        let num_layers = kv.config().num_layers;

        // 1. Embed all tokens into x_batch [batch_size, hidden_dim]
        for (t, &token_id) in prompt_tokens.iter().enumerate() {
            let tok = token_id as usize;
            let emb_start = tok * self.hidden_dim;
            let emb_end = emb_start + self.hidden_dim;
            if emb_end > self.embedding.len() {
                return Err(RuntimeError::Compute(format!(
                    "token id {token_id} out of embedding range"
                )));
            }
            let dst_start = t * self.hidden_dim;
            self.x_batch[dst_start..dst_start + self.hidden_dim]
                .copy_from_slice(&self.embedding[emb_start..emb_end]);
        }

        // 2. Process each layer
        for layer in 0..num_layers {
            weights.begin_pass();
            let layer_view = match weights.try_get_layer(layer) {
                Some(view) => view,
                None => weights.get_layer_blocking(layer)?,
            };
            let mut kv_view = kv.view_mut(layer)?;

            self.compute_layer_batched(
                batch_size,
                &layer_view,
                &mut kv_view,
            )?;

            kv.commit_view(kv_view)?;
        }

        // 3. Advance KV cache by batch_size tokens
        for _ in 0..batch_size {
            kv.advance_seq_len()?;
        }

        // 4. Extract the last token's hidden state
        let last_start = (batch_size - 1) * self.hidden_dim;
        let last_hidden = self.x_batch[last_start..last_start + self.hidden_dim].to_vec();
        Ok(last_hidden)
    }

    /// Process one transformer layer for a batch of tokens.
    fn compute_layer_batched(
        &mut self,
        batch_size: usize,
        weights: &LayerView,
        kv: &mut KvCacheView,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = self.hidden_dim;
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;
        let inter_dim = self.inter_dim;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let gqa_ratio = self.gqa_ratio;
        let eps = self.eps;
        let seq_pos_start = kv.seq_len; // position of first token in this batch

        let st = &weights.subtensors;
        let attn_norm_bytes = weights.subtensor_bytes(&st.attn_norm)?;
        let wq_bytes = weights.subtensor_bytes(&st.wq)?;
        let wk_bytes = weights.subtensor_bytes(&st.wk)?;
        let wv_bytes = weights.subtensor_bytes(&st.wv)?;
        let wo_bytes = weights.subtensor_bytes(&st.wo)?;
        let ffn_norm_bytes = weights.subtensor_bytes(&st.ffn_norm)?;
        let w_gate_bytes = weights.subtensor_bytes(&st.w_gate)?;
        let w_up_bytes = weights.subtensor_bytes(&st.w_up)?;
        let w_down_bytes = weights.subtensor_bytes(&st.w_down)?;

        // ---- 1. RMSNorm all tokens ----
        for t in 0..batch_size {
            let x_start = t * hidden_dim;
            let x_slice = &self.x_batch[x_start..x_start + hidden_dim];
            let out_slice = &mut self.normed_batch[t * hidden_dim..(t + 1) * hidden_dim];
            match st.attn_norm.quant {
                QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(out_slice, x_slice, attn_norm_bytes, eps),
                _ => simd_kernels::rmsnorm_bytes_simd(out_slice, x_slice, attn_norm_bytes, eps),
            }
        }

        // ---- 2. Q projection: [batch, hidden_dim] x W_q^T -> [batch, q_dim] ----
        dequant_and_gemm(
            &mut self.q_batch, &self.normed_batch, &mut self.dequant_buf,
            wq_bytes, st.wq.quant,
            batch_size, q_dim, hidden_dim, &self.pool,
        );

        // ---- 3. K, V projections ----
        dequant_and_gemm(
            &mut self.k_batch, &self.normed_batch, &mut self.dequant_buf,
            wk_bytes, st.wk.quant,
            batch_size, kv_dim, hidden_dim, &self.pool,
        );
        dequant_and_gemm(
            &mut self.v_batch, &self.normed_batch, &mut self.dequant_buf,
            wv_bytes, st.wv.quant,
            batch_size, kv_dim, hidden_dim, &self.pool,
        );

        // ---- 4. RoPE per-token + write KV cache ----
        for t in 0..batch_size {
            let pos = seq_pos_start + t;
            let q_start = t * q_dim;
            let k_start = t * kv_dim;
            let q_slice = &mut self.q_batch[q_start..q_start + q_dim];
            let k_slice = &mut self.k_batch[k_start..k_start + kv_dim];

            apply_rope_single(
                q_slice, k_slice,
                num_heads, num_kv_heads, head_dim, pos,
                &self.rope_cos, &self.rope_sin, self.half_dim,
            );

            // Pre-scale Q by attention scale
            for v in q_slice.iter_mut() {
                *v *= self.attn_scale;
            }

            // Write K, V to KV cache
            kv.append_keys_f32(k_slice);
            kv.append_values_f32(&self.v_batch[k_start..k_start + kv_dim]);
        }

        // ---- 5. Causal attention per-token ----
        // Parallelize across attention heads using the thread pool.
        // Each head reads from disjoint Q regions and writes to disjoint
        // attn_out and scores regions. KV cache is read-only at this point.
        let new_total_seq_len = seq_pos_start + batch_size;
        let max_seq_len = self.max_seq_len;
        let total_threads = self.pool.total_threads();
        let use_parallel_attn = num_heads >= total_threads && total_threads > 1;

        for t in 0..batch_size {
            let current_pos = seq_pos_start + t;
            let attend_len = current_pos + 1; // causal: attend to positions 0..=current_pos

            let attn_out_start = t * q_dim;
            // Zero the output for this token
            self.attn_out_batch[attn_out_start..attn_out_start + q_dim].fill(0.0);

            if use_parallel_attn && attend_len >= 64 {
                // Parallel path: transmit pointers as usize (Copy) to avoid borrowing self.
                // SAFETY: Each head writes to disjoint regions of scores and attn_out.
                // KV cache is read-only after the write phase in step 4.
                let q_ptr = self.q_batch.as_ptr() as usize;
                let scores_ptr = self.scores.as_mut_ptr() as usize;
                let attn_out_ptr = self.attn_out_batch.as_mut_ptr() as usize;
                let kv_keys_ptr = kv.keys.as_ptr() as usize;
                let kv_keys_len = kv.keys.len();
                let kv_values_ptr = kv.values.as_ptr() as usize;
                let kv_values_len = kv.values.len();

                self.pool.parallel_for_heads(num_heads, |h_start, h_end| {
                    for h in h_start..h_end {
                        let kv_h = h / gqa_ratio;
                        let kv_head_offset = kv_h * head_dim;

                        let q_head = unsafe {
                            std::slice::from_raw_parts(
                                (q_ptr as *const f32).add(t * q_dim + h * head_dim),
                                head_dim,
                            )
                        };

                        let head_scores = unsafe {
                            std::slice::from_raw_parts_mut(
                                (scores_ptr as *mut f32).add(h * max_seq_len),
                                attend_len,
                            )
                        };

                        for (p, score_slot) in head_scores.iter_mut().enumerate() {
                            let k_byte_start = (p * kv_dim + kv_head_offset) * 4;
                            debug_assert!(k_byte_start + head_dim * 4 <= kv_keys_len);
                            let k_slice = unsafe {
                                std::slice::from_raw_parts(
                                    (kv_keys_ptr as *const u8).add(k_byte_start) as *const f32,
                                    head_dim,
                                )
                            };
                            *score_slot = simd_kernels::dot_product_simd(q_head, k_slice);
                        }

                        simd_kernels::softmax_inplace_simd(head_scores);

                        let out_head = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_out_ptr as *mut f32).add(attn_out_start + h * head_dim),
                                head_dim,
                            )
                        };
                        out_head.fill(0.0);
                        for (p, &score) in head_scores.iter().enumerate() {
                            let v_byte_start = (p * kv_dim + kv_head_offset) * 4;
                            debug_assert!(v_byte_start + head_dim * 4 <= kv_values_len);
                            let v_slice = unsafe {
                                std::slice::from_raw_parts(
                                    (kv_values_ptr as *const u8).add(v_byte_start) as *const f32,
                                    head_dim,
                                )
                            };
                            simd_kernels::vscale_add_inplace_simd(out_head, v_slice, score);
                        }
                    }
                });
            } else {
                // Serial fallback for short sequences or small head counts
                for h in 0..num_heads {
                    let kv_h = h / gqa_ratio;
                    let kv_head_offset = kv_h * head_dim;

                    let q_head = &self.q_batch[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];

                    let scores_start = h * max_seq_len;
                    for p in 0..attend_len {
                        let k_elem_start = p * kv_dim + kv_head_offset;
                        let k_slice = kv.keys_f32_slice(k_elem_start, head_dim);
                        self.scores[scores_start + p] = simd_kernels::dot_product_simd(q_head, k_slice);
                    }

                    simd_kernels::softmax_inplace_simd(
                        &mut self.scores[scores_start..scores_start + attend_len],
                    );

                    let out_head = &mut self.attn_out_batch[attn_out_start + h * head_dim..attn_out_start + (h + 1) * head_dim];
                    out_head.fill(0.0);
                    for p in 0..attend_len {
                        let score = self.scores[scores_start + p];
                        let v_elem_start = p * kv_dim + kv_head_offset;
                        let v_slice = kv.values_f32_slice(v_elem_start, head_dim);
                        simd_kernels::vscale_add_inplace_simd(out_head, v_slice, score);
                    }
                }
            }
        }

        // ---- 6. Wo projection: [batch, q_dim] x W_o^T -> [batch, hidden_dim] ----
        dequant_and_gemm(
            &mut self.attn_proj_batch, &self.attn_out_batch, &mut self.dequant_buf,
            wo_bytes, st.wo.quant,
            batch_size, hidden_dim, q_dim, &self.pool,
        );

        // ---- 7. Residual add: x_batch += attn_proj_batch ----
        let n = batch_size * hidden_dim;
        simd_kernels::vadd_inplace_simd(
            &mut self.x_batch[..n],
            &self.attn_proj_batch[..n],
        );

        // ---- 8. FFN RMSNorm ----
        for t in 0..batch_size {
            let x_start = t * hidden_dim;
            let x_slice = &self.x_batch[x_start..x_start + hidden_dim];
            let out_slice = &mut self.normed_batch[t * hidden_dim..(t + 1) * hidden_dim];
            match st.ffn_norm.quant {
                QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(out_slice, x_slice, ffn_norm_bytes, eps),
                _ => simd_kernels::rmsnorm_bytes_simd(out_slice, x_slice, ffn_norm_bytes, eps),
            }
        }

        // ---- 9. Gate + Up projections ----
        dequant_and_gemm(
            &mut self.gate_batch, &self.normed_batch, &mut self.dequant_buf,
            w_gate_bytes, st.w_gate.quant,
            batch_size, inter_dim, hidden_dim, &self.pool,
        );
        dequant_and_gemm(
            &mut self.up_batch, &self.normed_batch, &mut self.dequant_buf,
            w_up_bytes, st.w_up.quant,
            batch_size, inter_dim, hidden_dim, &self.pool,
        );

        // ---- 10. SwiGLU: gate = silu(gate) * up ----
        simd_kernels::swiglu_inplace_simd(
            &mut self.gate_batch[..batch_size * inter_dim],
            &self.up_batch[..batch_size * inter_dim],
        );

        // ---- 11. Down projection: [batch, inter_dim] x W_down^T -> [batch, hidden_dim] ----
        dequant_and_gemm(
            &mut self.down_batch, &self.gate_batch, &mut self.dequant_buf,
            w_down_bytes, st.w_down.quant,
            batch_size, hidden_dim, inter_dim, &self.pool,
        );

        // ---- 12. Residual add: x_batch += down_batch ----
        let n = batch_size * hidden_dim;
        simd_kernels::vadd_inplace_simd(
            &mut self.x_batch[..n],
            &self.down_batch[..n],
        );

        // Update KV cache seq_len
        kv.seq_len = new_total_seq_len;

        Ok(())
    }

}

/// Dequantize a weight matrix (if Q8_0) and perform batched GEMM via Accelerate.
///
/// Computes: out[batch, out_dim] = input[batch, in_dim] x W^T[in_dim, out_dim]
///
/// Where W is stored row-major as [out_dim, in_dim] (each row is one output neuron).
/// `dequant_buf` is a reusable scratch buffer for dequantized weights.
/// `pool` is used to parallelize Q8_0 dequantization across cores.
#[allow(clippy::too_many_arguments)]
fn dequant_and_gemm(
    out: &mut [f32],
    input: &[f32],
    dequant_buf: &mut [f32],
    w_bytes: &[u8],
    quant: QuantScheme,
    batch_size: usize,
    out_dim: usize,
    in_dim: usize,
    pool: &ThreadPool,
) {
    match quant {
        QuantScheme::Q8_0 => {
            dequantize_q8_0_to_f32_parallel(pool, dequant_buf, w_bytes, out_dim, in_dim);
            gemm_batch(out, input, dequant_buf, batch_size, out_dim, in_dim);
        }
        _ => {
            let w_f32 = unsafe {
                std::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const f32,
                    out_dim * in_dim,
                )
            };
            gemm_batch(out, input, w_f32, batch_size, out_dim, in_dim);
        }
    }
}

/// Parallel wrapper around `dequantize_q8_0_to_f32`.
///
/// Splits the row range across the thread pool so each worker dequantizes a
/// contiguous slice of rows. Falls back to single-threaded for small matrices
/// (below PARALLEL_THRESHOLD or too few rows per thread).
fn dequantize_q8_0_to_f32_parallel(
    pool: &ThreadPool,
    dst: &mut [f32],
    q8_bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
) {
    if !pool.should_parallelize(out_dim) {
        dequantize_q8_0_to_f32(dst, q8_bytes, out_dim, in_dim);
        return;
    }
    let num_blocks_per_row = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks_per_row * Q8_0_BLOCK_SIZE;
    let dst_ptr = dst.as_mut_ptr() as usize;
    let src_ptr = q8_bytes.as_ptr() as usize;

    pool.parallel_for(out_dim, |row_start, row_end| {
        let num_rows = row_end - row_start;
        // SAFETY: Each thread writes to a disjoint row range of dst and reads
        // from the corresponding disjoint byte range of q8_bytes.
        unsafe {
            let sub_dst = std::slice::from_raw_parts_mut(
                (dst_ptr as *mut f32).add(row_start * in_dim),
                num_rows * in_dim,
            );
            let sub_src = std::slice::from_raw_parts(
                (src_ptr as *const u8).add(row_start * row_bytes),
                num_rows * row_bytes,
            );
            dequantize_q8_0_to_f32(sub_dst, sub_src, num_rows, in_dim);
        }
    });
}

/// Dequantize Q8_0 weight matrix to F32.
///
/// Q8_0 format: groups of 32 elements, each group stored as:
///   [f16 scale (2 bytes)] [32 x i8 quantized values (32 bytes)] = 34 bytes per block
///
/// Output: `dst[row * in_dim + col]` for row-major [out_dim x in_dim].
///
/// On aarch64, uses NEON intrinsics to process 16 elements per iteration
/// (2 iterations per Q8_0 block). The NEON path vectorizes i8→f32 widening
/// and scale multiplication for ~4-8x throughput over the scalar fallback.
#[cfg(target_arch = "aarch64")]
fn dequantize_q8_0_to_f32(
    dst: &mut [f32],
    q8_bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
) {
    use std::arch::aarch64::*;

    let num_blocks_per_row = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks_per_row * Q8_0_BLOCK_SIZE;

    for row in 0..out_dim {
        let row_start = row * row_bytes;
        let dst_row_start = row * in_dim;

        for b in 0..num_blocks_per_row {
            let block_start = row_start + b * Q8_0_BLOCK_SIZE;
            let scale_bits = u16::from_le_bytes([
                q8_bytes[block_start],
                q8_bytes[block_start + 1],
            ]);
            let scale = f16_to_f32(scale_bits);

            let elem_base = dst_row_start + b * Q8_0_GROUP_SIZE;
            let remaining = (in_dim - b * Q8_0_GROUP_SIZE).min(Q8_0_GROUP_SIZE);
            let q8_ptr = q8_bytes[block_start + 2..].as_ptr() as *const i8;
            let dst_ptr = dst[elem_base..].as_mut_ptr();

            unsafe {
                let scale_v = vdupq_n_f32(scale);

                // Process full 16-element chunks with NEON
                let mut j = 0usize;
                while j + 16 <= remaining {
                    // Load 16 x i8
                    let q8x16 = vld1q_s8(q8_ptr.add(j));
                    // Widen lower 8: i8 -> i16
                    let lo_i16 = vmovl_s8(vget_low_s8(q8x16));
                    // Widen upper 8: i8 -> i16
                    let hi_i16 = vmovl_s8(vget_high_s8(q8x16));
                    // Widen i16 -> i32 -> f32, multiply by scale
                    let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16))), scale_v);
                    let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16))), scale_v);
                    let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16))), scale_v);
                    let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16))), scale_v);
                    // Store 16 x f32
                    vst1q_f32(dst_ptr.add(j), f0);
                    vst1q_f32(dst_ptr.add(j + 4), f1);
                    vst1q_f32(dst_ptr.add(j + 8), f2);
                    vst1q_f32(dst_ptr.add(j + 12), f3);
                    j += 16;
                }

                // Scalar tail for remaining elements
                while j < remaining {
                    let q = *q8_ptr.add(j);
                    *dst_ptr.add(j) = scale * q as f32;
                    j += 1;
                }
            }
        }
    }
}

/// Scalar fallback for non-aarch64 platforms.
#[cfg(not(target_arch = "aarch64"))]
fn dequantize_q8_0_to_f32(
    dst: &mut [f32],
    q8_bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_row = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks_per_row * Q8_0_BLOCK_SIZE;

    for row in 0..out_dim {
        let row_start = row * row_bytes;
        let dst_row_start = row * in_dim;

        for b in 0..num_blocks_per_row {
            let block_start = row_start + b * Q8_0_BLOCK_SIZE;
            let scale_bits = u16::from_le_bytes([
                q8_bytes[block_start],
                q8_bytes[block_start + 1],
            ]);
            let scale = f16_to_f32(scale_bits);

            let elem_base = dst_row_start + b * Q8_0_GROUP_SIZE;
            let remaining = (in_dim - b * Q8_0_GROUP_SIZE).min(Q8_0_GROUP_SIZE);

            for j in 0..remaining {
                let q = q8_bytes[block_start + 2 + j] as i8;
                dst[elem_base + j] = scale * q as f32;
            }
        }
    }
}

/// Safe wrapper around `cblas_sgemm` for batched projection.
///
/// Computes: out[M x N] = A[M x K] * B^T[K x N]
///
/// where:
///   - A = input batch [batch_size x in_dim], row-major
///   - B = weight matrix [out_dim x in_dim], row-major (transposed in GEMM call)
///   - out = result [batch_size x out_dim], row-major
fn gemm_batch(
    out: &mut [f32],
    a: &[f32],       // [M x K]
    b: &[f32],       // [N x K] (will be transposed)
    m: usize,        // batch_size
    n: usize,        // out_dim
    k: usize,        // in_dim
) {
    debug_assert!(a.len() >= m * k, "a too small: {} < {}", a.len(), m * k);
    debug_assert!(b.len() >= n * k, "b too small: {} < {}", b.len(), n * k);
    debug_assert!(out.len() >= m * n, "out too small: {} < {}", out.len(), m * n);

    unsafe {
        cblas_sgemm(
            CblasOrder::RowMajor,
            CblasTranspose::NoTrans,  // A is already [M x K]
            CblasTranspose::Trans,    // B stored as [N x K], transposed to [K x N]
            m as i32,                 // M = batch_size
            n as i32,                 // N = out_dim
            k as i32,                 // K = in_dim
            1.0,                      // alpha
            a.as_ptr(),
            k as i32,                 // lda = K (row stride of A)
            b.as_ptr(),
            k as i32,                 // ldb = K (row stride of B before transpose)
            0.0,                      // beta
            out.as_mut_ptr(),
            n as i32,                 // ldc = N (row stride of C)
        );
    }
}

/// Hardware f16-to-f32 conversion using ARM FCVT instruction (single instruction).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let result: f32;
    unsafe {
        std::arch::asm!(
            "fmov {tmp:h}, {bits:w}",
            "fcvt {out:s}, {tmp:h}",
            bits = in(reg) bits as u32,
            tmp = out(vreg) _,
            out = lateout(vreg) result,
        );
    }
    result
}

/// Software f16-to-f32 conversion for non-aarch64 platforms.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;

    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let f = frac as f32 / 1024.0;
        let v = f * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    let v = 2.0f32.powi(exp as i32 - 15) * (1.0 + frac as f32 / 1024.0);
    if sign == 1 { -v } else { v }
}

/// Apply RoPE to Q and K for a single token at a given position.
///
/// On aarch64, uses NEON vld2q/vst2q deinterleave intrinsics with FMA to process
/// 8 elements per iteration (4 rotation pairs). Falls back to scalar on other platforms.
#[allow(clippy::too_many_arguments)]
fn apply_rope_single(
    q: &mut [f32],
    k: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    half_dim: usize,
) {
    let cos_base = pos * half_dim;
    let sin_base = pos * half_dim;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let pairs4 = half_dim / 4;
        let tail_start = pairs4 * 4;

        unsafe {
            for h in 0..num_q_heads {
                let head_start = h * head_dim;
                for p in 0..pairs4 {
                    let i = p * 4;
                    let idx = head_start + 2 * i;
                    let interleaved = vld2q_f32(q.as_ptr().add(idx));
                    let cos_vec = vld1q_f32(rope_cos.as_ptr().add(cos_base + i));
                    let sin_vec = vld1q_f32(rope_sin.as_ptr().add(sin_base + i));
                    let neg_sin = vnegq_f32(sin_vec);
                    let new_evens = vfmaq_f32(vmulq_f32(interleaved.0, cos_vec), interleaved.1, neg_sin);
                    let new_odds = vfmaq_f32(vmulq_f32(interleaved.1, cos_vec), interleaved.0, sin_vec);
                    vst2q_f32(q.as_mut_ptr().add(idx), float32x4x2_t(new_evens, new_odds));
                }
                for i in tail_start..half_dim {
                    let idx0 = head_start + 2 * i;
                    let idx1 = idx0 + 1;
                    let v0 = *q.get_unchecked(idx0);
                    let v1 = *q.get_unchecked(idx1);
                    *q.get_unchecked_mut(idx0) =
                        v0 * *rope_cos.get_unchecked(cos_base + i) - v1 * *rope_sin.get_unchecked(sin_base + i);
                    *q.get_unchecked_mut(idx1) =
                        v0 * *rope_sin.get_unchecked(sin_base + i) + v1 * *rope_cos.get_unchecked(cos_base + i);
                }
            }
            for h in 0..num_kv_heads {
                let head_start = h * head_dim;
                for p in 0..pairs4 {
                    let i = p * 4;
                    let idx = head_start + 2 * i;
                    let interleaved = vld2q_f32(k.as_ptr().add(idx));
                    let cos_vec = vld1q_f32(rope_cos.as_ptr().add(cos_base + i));
                    let sin_vec = vld1q_f32(rope_sin.as_ptr().add(sin_base + i));
                    let neg_sin = vnegq_f32(sin_vec);
                    let new_evens = vfmaq_f32(vmulq_f32(interleaved.0, cos_vec), interleaved.1, neg_sin);
                    let new_odds = vfmaq_f32(vmulq_f32(interleaved.1, cos_vec), interleaved.0, sin_vec);
                    vst2q_f32(k.as_mut_ptr().add(idx), float32x4x2_t(new_evens, new_odds));
                }
                for i in tail_start..half_dim {
                    let idx0 = head_start + 2 * i;
                    let idx1 = idx0 + 1;
                    let v0 = *k.get_unchecked(idx0);
                    let v1 = *k.get_unchecked(idx1);
                    *k.get_unchecked_mut(idx0) =
                        v0 * *rope_cos.get_unchecked(cos_base + i) - v1 * *rope_sin.get_unchecked(sin_base + i);
                    *k.get_unchecked_mut(idx1) =
                        v0 * *rope_sin.get_unchecked(sin_base + i) + v1 * *rope_cos.get_unchecked(cos_base + i);
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for h in 0..num_q_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let idx0 = head_start + 2 * i;
                let idx1 = head_start + 2 * i + 1;
                let v0 = q[idx0];
                let v1 = q[idx1];
                q[idx0] = v0 * rope_cos[cos_base + i] - v1 * rope_sin[sin_base + i];
                q[idx1] = v0 * rope_sin[sin_base + i] + v1 * rope_cos[cos_base + i];
            }
        }
        for h in 0..num_kv_heads {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let idx0 = head_start + 2 * i;
                let idx1 = head_start + 2 * i + 1;
                let v0 = k[idx0];
                let v1 = k[idx1];
                k[idx0] = v0 * rope_cos[cos_base + i] - v1 * rope_sin[sin_base + i];
                k[idx1] = v0 * rope_sin[sin_base + i] + v1 * rope_cos[cos_base + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_roundtrip() {
        // Create a known Q8_0 block: scale=1.0 in f16, values=[0,1,2,...,31]
        let mut block = [0u8; Q8_0_BLOCK_SIZE];
        // f16 for 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        for j in 0..Q8_0_GROUP_SIZE {
            block[2 + j] = j as u8;
        }

        let mut dst = vec![0.0f32; Q8_0_GROUP_SIZE];
        dequantize_q8_0_to_f32(&mut dst, &block, 1, Q8_0_GROUP_SIZE);

        for j in 0..Q8_0_GROUP_SIZE {
            let expected = j as f32; // scale=1.0 * q=j
            assert!(
                (dst[j] - expected).abs() < 1e-3,
                "dst[{j}] = {}, expected {expected}", dst[j]
            );
        }
    }

    #[test]
    fn test_gemm_batch_identity() {
        // out = [1,2,3; 4,5,6] * I_3^T = [1,2,3; 4,5,6]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity
        let mut out = vec![0.0f32; 6]; // 2x3

        gemm_batch(&mut out, &a, &b, 2, 3, 3);

        for i in 0..6 {
            assert!(
                (out[i] - a[i]).abs() < 1e-5,
                "out[{i}] = {}, expected {}", out[i], a[i]
            );
        }
    }

    #[test]
    fn test_gemm_batch_transpose() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[1, 2], [3, 4]]  (2x2, will be transposed)
        // C = A * B^T = [[1*1+2*2, 1*3+2*4], [3*1+4*2, 3*3+4*4]] = [[5, 11], [11, 25]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];

        gemm_batch(&mut out, &a, &b, 2, 2, 2);

        let expected = [5.0, 11.0, 11.0, 25.0];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-4,
                "out[{i}] = {}, expected {}", out[i], expected[i]
            );
        }
    }

    #[test]
    fn test_f16_to_f32_values() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }
}
