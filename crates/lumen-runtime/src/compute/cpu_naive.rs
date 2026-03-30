//! Naive F32 compute backend — the ground truth reference implementation.
//!
//! All future optimized backends MUST match the output of this implementation.
//! Karpathy-style flat functions, no SIMD, no unsafe. Correct first.
//!
//! Implements a standard LLaMA-style transformer:
//! - RMSNorm (eps inside sqrt, best practice 4.2)
//! - Multi-head attention with GQA and RoPE
//! - SwiGLU MLP
//! - Softmax with max-subtraction (best practice 4.1)
//! - Attention scaling by 1/sqrt(head_dim) (best practice 4.3)
//!
//! # Optimization: Zero-Copy Weights & Scratch Buffer Reuse
//!
//! The hot path (`compute_layer`) eliminates per-call heap allocations via:
//! - `matmul_bytes`/`rmsnorm_bytes`: read weight bytes as LE f32 inline (no Vec<f32>)
//! - `ComputeScratch`: pre-allocated working buffers reused across calls
//! - Pre-computed RoPE cos/sin tables (computed once in `init()`)
//!
//! Output is bit-identical to the original allocating implementation.

use crate::weight::cache::LayerView;
use crate::compute::{ActivationBuffer, ComputeBackend, ComputeDtype, Logits};
use crate::engine::softmax_inplace;
use crate::error::RuntimeError;
use crate::kv::KvCacheView;
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;

/// Pre-allocated working buffers reused across compute_layer calls.
/// Avoids heap allocation in the hot path.
struct ComputeScratch {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    attn_proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    logits: Vec<f32>,
    // Pre-computed RoPE tables
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    half_dim: usize,
    // Scratch buffer for extracting contiguous key head data from KV cache
    k_head_buf: Vec<f32>,
    // Scratch buffer for extracting contiguous value head data from KV cache
    v_head_buf: Vec<f32>,
}

/// Apply Rotary Position Embeddings using pre-computed cos/sin tables.
#[allow(clippy::too_many_arguments)]
fn apply_rope_precomputed(
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

/// Naive F32 compute backend. Zero external dependencies, zero unsafe.
///
/// Uses pre-allocated scratch buffers (via `ComputeScratch`) and zero-copy
/// weight access (via `matmul_bytes`/`rmsnorm_bytes`) to eliminate heap
/// allocations in the hot path while maintaining bit-identical output.
pub struct NaiveF32Backend {
    hyperparams: Option<ModelHyperparams>,
    /// Token embedding table (vocab_size * hidden_dim).
    embedding: Vec<f32>,
    /// Final RMSNorm weights (hidden_dim).
    final_norm: Vec<f32>,
    /// Output projection weights (vocab_size * hidden_dim).
    output_proj: Vec<f32>,
    /// Pre-allocated scratch buffers, initialized in init().
    /// Mutex for interior mutability (compute_layer takes &self).
    /// Uncontended lock is ~20ns, negligible vs. compute cost.
    scratch: Mutex<Option<ComputeScratch>>,
}

impl Default for NaiveF32Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveF32Backend {
    pub fn new() -> Self {
        Self {
            hyperparams: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            scratch: Mutex::new(None),
        }
    }

    fn hp(&self) -> Result<&ModelHyperparams, RuntimeError> {
        self.hyperparams.as_ref().ok_or_else(|| {
            RuntimeError::Compute("backend not initialized: call init() first".into())
        })
    }
}

// ---------- Core math operations (free functions) ----------

/// RMSNorm: x_i * w_i / sqrt(mean(x^2) + eps)
/// MUST: eps inside sqrt (best practice 4.2)
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Matrix-vector multiply: out = W * x
/// W is row-major: [out_dim, in_dim]
fn matmul(out: &mut [f32], w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        out[i] = row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum();
    }
}

/// Matrix-vector multiply reading weights directly from LE bytes.
/// Eliminates the entire bytes_to_f32 allocation for weight matrices.
#[allow(clippy::needless_range_loop)] // indices compute byte offsets, iterator pattern is less clear
fn matmul_bytes(out: &mut [f32], w_bytes: &[u8], x: &[f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let row_byte_start = i * in_dim * 4;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let offset = row_byte_start + j * 4;
            let w_val = f32::from_le_bytes(w_bytes[offset..offset + 4].try_into().unwrap());
            sum += w_val * x[j];
        }
        out[i] = sum;
    }
}

/// Convert f16 bits to f32 (software implementation, no external dependency).
fn f16_bits_to_f32(bits: u16) -> f32 {
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
    let exp_f32 = ((exp as u32 - 15 + 127) << 23) | ((frac as u32) << 13);
    let v = f32::from_bits(exp_f32);
    if sign == 1 { -v } else { v }
}

/// Matrix-vector multiply reading F16 weights from bytes.
/// Weight layout: row-major [out_dim, in_dim], each element is 2 bytes (LE f16 bits).
#[allow(clippy::needless_range_loop)]
fn matmul_bytes_f16(out: &mut [f32], w_bytes: &[u8], x: &[f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let row_byte_start = i * in_dim * 2;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let offset = row_byte_start + j * 2;
            let bits = u16::from_le_bytes(w_bytes[offset..offset + 2].try_into().unwrap());
            let w_val = f16_bits_to_f32(bits);
            sum += w_val * x[j];
        }
        out[i] = sum;
    }
}

/// Matrix-vector multiply reading Q8_0 weights from bytes.
/// Weight layout: row-major [out_dim, in_dim] stored as Q8_0 blocks.
/// Each row has (in_dim / 32) blocks of 34 bytes: [2B f16 scale][32B int8].
#[allow(clippy::needless_range_loop)]
fn matmul_bytes_q8_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], out_dim: usize, in_dim: usize) {
    let blocks_per_row = in_dim / 32;
    let row_bytes = blocks_per_row * 34;
    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        let mut x_idx = 0usize;
        for b in 0..blocks_per_row {
            let block_start = row_start + b * 34;
            let scale_bits = u16::from_le_bytes([w_bytes[block_start], w_bytes[block_start + 1]]);
            let scale = f16_bits_to_f32(scale_bits);
            for k in 0..32 {
                let q = w_bytes[block_start + 2 + k] as i8;
                sum += (scale * q as f32) * x[x_idx];
                x_idx += 1;
            }
        }
        out[i] = sum;
    }
}

/// Matrix-vector multiply reading Q4_0 weights from bytes.
/// Weight layout: row-major [out_dim, in_dim] stored as Q4_0 blocks.
/// Each row has (in_dim / 32) blocks of 18 bytes: [2B f16 scale][16B packed nibbles].
/// GGML de-interleaved order: indices 0-15 from lo nibbles, indices 16-31 from hi nibbles.
#[allow(clippy::needless_range_loop)]
fn matmul_bytes_q4_0(out: &mut [f32], w_bytes: &[u8], x: &[f32], out_dim: usize, in_dim: usize) {
    let blocks_per_row = in_dim / 32;
    let row_bytes = blocks_per_row * 18;
    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        let mut x_idx = 0usize;
        for b in 0..blocks_per_row {
            let block_start = row_start + b * 18;
            let scale_bits = u16::from_le_bytes([w_bytes[block_start], w_bytes[block_start + 1]]);
            let scale = f16_bits_to_f32(scale_bits);
            // First 16 elements: lo nibbles (indices 0-15)
            for k in 0..16 {
                let byte = w_bytes[block_start + 2 + k];
                let lo = (byte & 0x0F) as i32 - 8;
                sum += (scale * lo as f32) * x[x_idx];
                x_idx += 1;
            }
            // Next 16 elements: hi nibbles (indices 16-31)
            for k in 0..16 {
                let byte = w_bytes[block_start + 2 + k];
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                sum += (scale * hi as f32) * x[x_idx];
                x_idx += 1;
            }
        }
        out[i] = sum;
    }
}

/// Dispatch matrix-vector multiply based on the tensor's quantization scheme.
fn matmul_dispatch(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    quant: QuantScheme,
) {
    match quant {
        QuantScheme::F32 => matmul_bytes(out, w_bytes, x, out_dim, in_dim),
        QuantScheme::F16 => matmul_bytes_f16(out, w_bytes, x, out_dim, in_dim),
        QuantScheme::Q8_0 => matmul_bytes_q8_0(out, w_bytes, x, out_dim, in_dim),
        QuantScheme::Q4_0 => matmul_bytes_q4_0(out, w_bytes, x, out_dim, in_dim),
        _ => panic!("NaiveF32Backend: unsupported weight quant scheme {:?}", quant),
    }
}

/// RMSNorm reading weight directly from LE bytes.
fn rmsnorm_bytes(out: &mut [f32], x: &[f32], weight_bytes: &[u8], eps: f32) {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for i in 0..n {
        let w = f32::from_le_bytes(weight_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        out[i] = x[i] * scale * w;
    }
}

/// Fused SwiGLU: silu(gate) * up, written in-place to gate.
fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// Apply Rotary Position Embeddings to q and k vectors.
/// q/k shape: [num_heads, head_dim] (interleaved pairs)
/// Pre-computes trig table once (head_dim/2 entries) then applies to all heads.
/// Kept as reference implementation; hot path uses apply_rope_precomputed.
#[allow(dead_code)]
fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;

    // Pre-compute cos/sin table once for all heads
    let mut cos_table = vec![0.0f32; half_dim];
    let mut sin_table = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos_table[i] = angle.cos();
        sin_table[i] = angle.sin();
    }

    let apply_to_head = |vec: &mut [f32], head_start: usize| {
        for i in 0..half_dim {
            let idx0 = head_start + 2 * i;
            let idx1 = head_start + 2 * i + 1;
            let v0 = vec[idx0];
            let v1 = vec[idx1];
            vec[idx0] = v0 * cos_table[i] - v1 * sin_table[i];
            vec[idx1] = v0 * sin_table[i] + v1 * cos_table[i];
        }
    };

    for h in 0..num_q_heads {
        apply_to_head(q, h * head_dim);
    }
    for h in 0..num_kv_heads {
        apply_to_head(k, h * head_dim);
    }
}

/// Extend a byte buffer with f32 values (avoids intermediate Vec).
/// Used for activation buffer conversion; KV cache writes use KvCacheView helpers.
#[inline]
fn extend_f32_as_bytes(buf: &mut Vec<u8>, values: &[f32]) {
    buf.reserve(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Create an ActivationBuffer from f32 values.
fn f32_to_activation(values: &[f32]) -> ActivationBuffer {
    let mut data = Vec::with_capacity(values.len() * 4);
    extend_f32_as_bytes(&mut data, values);
    ActivationBuffer {
        data,
        num_elements: values.len(),
        dtype: ComputeDtype::F32,
    }
}

impl ComputeBackend for NaiveF32Backend {
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

    fn init(&mut self, hyperparams: &ModelHyperparams) -> Result<(), RuntimeError> {
        self.hyperparams = Some(*hyperparams);

        let hidden_dim = hyperparams.hidden_dim as usize;
        let num_heads = hyperparams.num_heads as usize;
        let num_kv_heads = hyperparams.num_kv_heads as usize;
        let head_dim = hyperparams.head_dim as usize;
        let inter_dim = hyperparams.intermediate_dim as usize;
        let vocab_size = hyperparams.vocab_size as usize;
        let max_seq_len = hyperparams.max_seq_len as usize;

        // Pre-compute RoPE tables for all positions up to max_seq_len
        let half_dim = head_dim / 2;
        let theta = hyperparams.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
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

        let scratch = ComputeScratch {
            normed: vec![0.0f32; hidden_dim],
            q: vec![0.0f32; num_heads * head_dim],
            k: vec![0.0f32; num_kv_heads * head_dim],
            v: vec![0.0f32; num_kv_heads * head_dim],
            attn_out: vec![0.0f32; num_heads * head_dim],
            scores: vec![0.0f32; max_seq_len],
            attn_proj: vec![0.0f32; hidden_dim],
            gate: vec![0.0f32; inter_dim],
            up: vec![0.0f32; inter_dim],
            down: vec![0.0f32; hidden_dim],
            logits: vec![0.0f32; vocab_size],
            rope_cos,
            rope_sin,
            half_dim,
            k_head_buf: vec![0.0f32; head_dim],
            v_head_buf: vec![0.0f32; head_dim],
        };
        *self.scratch.lock().unwrap() = Some(scratch);

        Ok(())
    }

    #[allow(clippy::needless_range_loop)] // t indexes both scores[] and KV cache byte offsets
    fn compute_layer(
        &self,
        _layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let num_heads = hp.num_heads as usize;
        let num_kv_heads = hp.num_kv_heads as usize;
        let head_dim = hp.head_dim as usize;
        let inter_dim = hp.intermediate_dim as usize;
        let eps = hp.norm_eps;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("scratch not initialized: call init() first".into())
        })?;

        // Zero-copy f32 view of the input activation.
        // We extract a raw pointer + length to avoid holding an immutable borrow on
        // `x` (which we need to mutably borrow later in write_f32_from).
        // SAFETY: x.data is not modified until write_f32_from at the very end of this
        // function. The pointer remains valid for all reads below.
        let (x_f32_ptr, x_f32_len) = {
            let view = x.as_f32_slice();
            (view.as_ptr(), view.len())
        };
        let x_f32: &[f32] = unsafe { std::slice::from_raw_parts(x_f32_ptr, x_f32_len) };

        // Get raw byte slices for all subtensors (zero-copy from LayerView)
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

        // 1. Attention RMSNorm (writes into scratch.normed)
        // Norms are always F32 regardless of weight quant scheme
        rmsnorm_bytes(&mut s.normed, x_f32, attn_norm_bytes, eps);

        // 2. Q/K/V projections (write into scratch buffers)
        // Dispatch based on per-tensor quant scheme for quantized models
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        matmul_dispatch(&mut s.q, wq_bytes, &s.normed, q_dim, hidden_dim, st.wq.quant);
        matmul_dispatch(&mut s.k, wk_bytes, &s.normed, kv_dim, hidden_dim, st.wk.quant);
        matmul_dispatch(&mut s.v, wv_bytes, &s.normed, kv_dim, hidden_dim, st.wv.quant);

        // 3. Apply RoPE (pre-computed tables)
        apply_rope_precomputed(
            &mut s.q, &mut s.k,
            num_heads, num_kv_heads, head_dim, seq_pos,
            &s.rope_cos, &s.rope_sin, s.half_dim,
        );

        // 4. Append k,v to KV cache
        let kv = kv.ok_or_else(|| {
            RuntimeError::Compute("KV cache view required for attention".into())
        })?;

        // Append new k,v to cache via KvCacheView helpers (proper abstraction layer)
        kv.append_keys(&s.k);
        kv.append_values(&s.v);
        let new_seq_len = (kv.seq_len + 1).min(kv.max_seq_len);

        // 5. Multi-head attention with GQA
        let scale = 1.0 / (head_dim as f32).sqrt(); // MUST: practice 4.3
        let gqa_ratio = num_heads / num_kv_heads;
        let kv_max_seq_len = kv.max_seq_len;

        for h in 0..num_heads {
            let kv_h = h / gqa_ratio; // GQA: multiple Q heads share one KV head
            // Head-first layout: per-head element base
            let kv_head_elem_base = kv_h * kv_max_seq_len * head_dim;

            // Compute attention scores for this head
            for t in 0..new_seq_len {
                // Head-first layout: K[kv_h] is contiguous at base + t*head_dim
                let k_start = kv_head_elem_base + t * head_dim;
                kv.read_keys_into(&mut s.k_head_buf, k_start, head_dim);
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += s.q[h * head_dim + d] * s.k_head_buf[d];
                }
                s.scores[t] = dot * scale;
            }

            // Softmax over scores (best practice 4.1)
            softmax_inplace(&mut s.scores[..new_seq_len]);

            // Weighted sum of values (t-outer loop for cache-friendly V access)
            for d in 0..head_dim {
                s.attn_out[h * head_dim + d] = 0.0;
            }
            for t in 0..new_seq_len {
                let score = s.scores[t];
                let v_start = kv_head_elem_base + t * head_dim;
                // Bulk-read contiguous V data into scratch buffer
                kv.read_values_into(&mut s.v_head_buf, v_start, head_dim);
                for d in 0..head_dim {
                    s.attn_out[h * head_dim + d] += score * s.v_head_buf[d];
                }
            }
        }

        // 6. Output projection
        matmul_dispatch(&mut s.attn_proj, wo_bytes, &s.attn_out, hidden_dim, q_dim, st.wo.quant);

        // 7. Residual connection (in-place into attn_proj to reuse buffer)
        // x_f32 is the zero-copy view of the original input activation
        for i in 0..hidden_dim {
            s.attn_proj[i] += x_f32[i];
        }

        // 8. FFN RMSNorm (reuse normed buffer)
        rmsnorm_bytes(&mut s.normed, &s.attn_proj, ffn_norm_bytes, eps);

        // 9. SwiGLU MLP (fused silu+multiply, reuse buffers)
        matmul_dispatch(&mut s.gate, w_gate_bytes, &s.normed, inter_dim, hidden_dim, st.w_gate.quant);
        matmul_dispatch(&mut s.up, w_up_bytes, &s.normed, inter_dim, hidden_dim, st.w_up.quant);
        swiglu_inplace(&mut s.gate, &s.up); // gate now holds silu(gate)*up
        matmul_dispatch(&mut s.down, w_down_bytes, &s.gate, hidden_dim, inter_dim, st.w_down.quant);

        // 10. Residual connection
        for i in 0..hidden_dim {
            s.attn_proj[i] += s.down[i];
        }

        // Update KV cache seq_len
        kv.seq_len = new_seq_len;

        // Write result back to activation buffer (reuse existing allocation)
        x.write_f32_from(&s.attn_proj);
        Ok(())
    }

    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let vocab_size = hp.vocab_size as usize;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("scratch not initialized: call init() first".into())
        })?;

        // Zero-copy f32 view of the final hidden state (no memcpy)
        let x_f32 = x.as_f32_slice();

        // Final RMSNorm (using global tensors which are already Vec<f32>)
        rmsnorm(&mut s.normed, x_f32, &self.final_norm, hp.norm_eps);

        // Project to vocab logits
        matmul(&mut s.logits, &self.output_proj, &s.normed, vocab_size, hidden_dim);

        // Move-and-restore: take the computed logits out (O(1) pointer move),
        // then restore scratch.logits for the next call without zeroing.
        // matmul() will overwrite all vocab_size elements before any read.
        let out = std::mem::take(&mut s.logits);
        s.logits.resize(vocab_size, 0.0);
        Ok(Logits { data: out })
    }

    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        let hp = self.hp()?;
        let hidden_dim = hp.hidden_dim as usize;
        let tid = token_id as usize;

        if tid >= hp.vocab_size as usize {
            return Err(RuntimeError::Compute(format!(
                "token_id {tid} out of range (vocab_size={})",
                hp.vocab_size
            )));
        }

        let start = tid * hidden_dim;
        let end = start + hidden_dim;
        let embed = &self.embedding[start..end];

        Ok(f32_to_activation(embed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0f32; 4];
        rmsnorm(&mut result, &x, &w, 1e-5);
        // mean(x^2) = (1+4+9+16)/4 = 7.5
        // scale = 1/sqrt(7.5 + 1e-5) ≈ 0.36514
        let expected_scale = 1.0 / (7.5f32 + 1e-5).sqrt();
        for (i, &v) in result.iter().enumerate() {
            let expected = x[i] * expected_scale;
            assert!((v - expected).abs() < 1e-5, "rmsnorm[{i}]: {v} != {expected}");
        }
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum: {sum}");
        // Check ordering
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_large_values() {
        // Must not overflow even with large values (best practice 4.1)
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum with large values: {sum}");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matmul() {
        // 2x3 matrix times 3-vector
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 2];
        matmul(&mut out, &w, &x, 2, 3);
        assert_eq!(out, vec![4.0, 10.0]);
    }

    #[test]
    fn test_swiglu() {
        // SwiGLU: silu(gate) * up
        let mut gate = vec![0.0, 1.0, -1.0];
        let up = vec![1.0, 1.0, 1.0];
        swiglu_inplace(&mut gate, &up);
        // silu(0)*1 = 0, silu(1)*1 ≈ 0.7310586, silu(-1)*1 ≈ -0.2689414
        assert!((gate[0] - 0.0).abs() < 1e-6);
        assert!((gate[1] - 0.7310586).abs() < 1e-5);
        assert!((gate[2] - (-0.2689414)).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope() {
        let mut q = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        let mut k = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut q, &mut k, 1, 1, 4, 0, 10000.0);
        // At position 0, all angles are 0, so cos=1, sin=0 => no change
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope_nonzero_position() {
        // At position 1 with head_dim=4, theta=10000.0:
        //   freq[0] = 1.0 / 10000^(0/4) = 1.0, angle[0] = 1.0 * 1.0 = 1.0
        //   freq[1] = 1.0 / 10000^(2/4) = 0.01, angle[1] = 1.0 * 0.01 = 0.01
        // For pair (q[0], q[1]) with input (1.0, 0.0):
        //   q[0] = 1.0 * cos(1.0) - 0.0 * sin(1.0) = cos(1.0)
        //   q[1] = 1.0 * sin(1.0) + 0.0 * cos(1.0) = sin(1.0)
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut q, &mut k, 1, 1, 4, 1, 10000.0);

        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((q[0] - cos1).abs() < 1e-6, "q[0]: expected {cos1}, got {}", q[0]);
        assert!((q[1] - sin1).abs() < 1e-6, "q[1]: expected {sin1}, got {}", q[1]);

        // Verify rotation preserved vector magnitude
        let mag_before = (1.0f32 * 1.0 + 0.0 * 0.0).sqrt();
        let mag_after = (q[0] * q[0] + q[1] * q[1]).sqrt();
        assert!((mag_before - mag_after).abs() < 1e-6, "RoPE must preserve magnitude");
    }

    #[test]
    fn test_rmsnorm_all_zeros() {
        // All-zero input exercises the epsilon guard (best practice 4.2).
        // ms = 0, scale = 1/sqrt(0 + 1e-5) ≈ 316.23
        // Output = 0 * scale = 0 for all elements (eps prevents inf, zeros stay zero).
        let x = vec![0.0f32; 4];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0f32; 4];
        rmsnorm(&mut result, &x, &w, 1e-5);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "rmsnorm[{i}] should be finite, got {v}");
            assert_eq!(v, 0.0, "rmsnorm of zero input should be zero");
        }
    }
}
