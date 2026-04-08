//! SIMD-accelerated F32 compute backend.
//!
//! Mirrors `cpu_naive::NaiveF32Backend` exactly, substituting SIMD kernel
//! calls for the scalar math operations. Must produce identical output to the
//! naive backend (verified by e2e tests in Task #3).
//!
//! Substitutions:
//! - `rmsnorm_bytes` -> `simd_kernels::rmsnorm_bytes_simd`
//! - `matmul_bytes`  -> `simd_kernels::matmul_bytes_simd`
//! - `matmul`        -> `simd_kernels::matmul_simd`
//! - `swiglu_inplace` -> `simd_kernels::swiglu_inplace_simd`
//! - `softmax_inplace` -> `simd_kernels::softmax_inplace_simd`
//! - attention dot product -> `simd_kernels::dot_product_simd`
//! - attention value accum -> `vscale_add_inplace` (NEON FMA)
//! - residual additions    -> `vadd_inplace` (NEON)
//! - attention scale       -> precomputed once in `init()` (practice 4.3)
//!
//! RoPE uses NEON vld2q/vst2q for paired rotation on aarch64.

use crate::weight::cache::LayerView;
use crate::compute::{ActivationBuffer, ComputeBackend, ComputeDtype, Logits};
use crate::error::RuntimeError;
use crate::kv::{KvCacheView, KvPrecision};
use crate::compute::simd_kernels;
use crate::thread_pool::ThreadPool;
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

/// Accumulated per-operation timing across all layers and tokens.
/// Each field accumulates the total time spent in that operation.
#[derive(Debug, Clone, Default)]
pub struct ComputeProfile {
    /// Number of compute_layer calls profiled.
    pub calls: u64,
    /// 1. RMSNorm (pre-attention)
    pub attn_rmsnorm: Duration,
    /// 2. Quantize normed (for Q/K/V projections, aarch64 Q8_0 path)
    pub quantize_normed_attn: Duration,
    /// 3. Q+K+V matmul (fused or separate dispatch)
    pub qkv_matmul: Duration,
    /// 4. RoPE
    pub rope: Duration,
    /// 5. KV cache write
    pub kv_cache_write: Duration,
    /// 6. Attention loop (all heads: dot products + softmax + value accumulation)
    pub attention: Duration,
    /// 7. Quantize attn_out + Wo matmul
    pub wo_matmul: Duration,
    /// 8. Residual add (post-attention)
    pub residual_attn: Duration,
    /// 9. FFN RMSNorm
    pub ffn_rmsnorm: Duration,
    /// 10. Quantize normed (for gate+up, aarch64 Q8_0 path)
    pub quantize_normed_ffn: Duration,
    /// 11. Gate+Up matmul (fused dispatch)
    pub gate_up_matmul: Duration,
    /// 12. SwiGLU activation
    pub swiglu: Duration,
    /// 13. Quantize gate + Down matmul
    pub down_matmul: Duration,
    /// 14. Residual add (post-FFN)
    pub residual_ffn: Duration,
}

impl ComputeProfile {
    /// Print a formatted summary of the per-operation profile.
    pub fn print_summary(&self, num_layers: usize) {
        if self.calls == 0 {
            eprintln!("[profile] No compute_layer calls recorded.");
            return;
        }
        let total = self.attn_rmsnorm
            + self.quantize_normed_attn
            + self.qkv_matmul
            + self.rope
            + self.kv_cache_write
            + self.attention
            + self.wo_matmul
            + self.residual_attn
            + self.ffn_rmsnorm
            + self.quantize_normed_ffn
            + self.gate_up_matmul
            + self.swiglu
            + self.down_matmul
            + self.residual_ffn;

        let total_us = total.as_secs_f64() * 1_000_000.0;
        let calls = self.calls as f64;
        let num_layers = if num_layers > 0 { num_layers } else { 1 };
        let tokens = (self.calls as usize) / num_layers;

        eprintln!("\n========== COMPUTE PROFILE ({} layer calls, {} tokens x {} layers) ==========",
            self.calls, tokens, num_layers);
        eprintln!("{:<30} {:>10} {:>10} {:>10} {:>7}",
            "Operation", "Total(us)", "Per-call", "Per-tok", "Pct(%)");
        eprintln!("{:-<72}", "");

        let ops: &[(&str, Duration)] = &[
            ("1. Attn RMSNorm",          self.attn_rmsnorm),
            ("2. Quant normed(attn)",     self.quantize_normed_attn),
            ("3. Q+K+V matmul",          self.qkv_matmul),
            ("4. RoPE",                  self.rope),
            ("5. KV cache write",        self.kv_cache_write),
            ("6. Attention loop",        self.attention),
            ("7. Wo matmul",             self.wo_matmul),
            ("8. Residual add(attn)",    self.residual_attn),
            ("9. FFN RMSNorm",           self.ffn_rmsnorm),
            ("10. Quant normed(ffn)",    self.quantize_normed_ffn),
            ("11. Gate+Up matmul",       self.gate_up_matmul),
            ("12. SwiGLU",               self.swiglu),
            ("13. Down matmul",          self.down_matmul),
            ("14. Residual add(ffn)",    self.residual_ffn),
        ];

        for (name, dur) in ops {
            let us = dur.as_secs_f64() * 1_000_000.0;
            let avg = us / calls;
            let per_tok = if tokens > 0 { us / tokens as f64 } else { 0.0 };
            let pct = if total_us > 0.0 { us / total_us * 100.0 } else { 0.0 };
            eprintln!("{name:<30} {us:>10.0} {avg:>10.1} {per_tok:>10.1} {pct:>6.1}%");
        }

        eprintln!("{:-<72}", "");
        let per_tok_total = if tokens > 0 { total_us / tokens as f64 } else { 0.0 };
        eprintln!("{:<30} {:>10.0} {:>10.1} {:>10.1} {:>6.1}%",
            "TOTAL", total_us, total_us / calls, per_tok_total, 100.0);
        eprintln!("\nPer-token compute: {:.1} us = {:.3} ms", per_tok_total, per_tok_total / 1000.0);
        eprintln!("=======================================================================\n");
    }
}

/// Pre-allocated working buffers reused across compute_layer calls.
/// Mirrors NaiveF32Backend's ComputeScratch with additional buffers
/// for extracting contiguous key/value slices for SIMD dot product and FMA.
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
    // Scratch buffers for extracting contiguous key/value head data from KV cache.
    // Only used on big-endian platforms (LE uses zero-copy keys_f32_slice/values_f32_slice).
    #[allow(dead_code)]
    k_head_buf: Vec<f32>,
    #[allow(dead_code)]
    v_head_buf: Vec<f32>,
    // Pre-computed attention scale: 1.0 / sqrt(head_dim) (practice 4.3)
    attn_scale: f32,
    // Pre-quantized Q8_0 scratch buffers: quantize normed/gate ONCE per layer,
    // then reuse for all matmuls that share the same input vector.
    // Eliminates 3 redundant quantizations + 3 heap allocations for Q/K/V projections
    // and 2 redundant quantizations for fused gate+up, per layer.
    normed_q8: Vec<u8>,  // Pre-quantized normed for Q/K/V and gate+up projections
    gate_q8: Vec<u8>,    // Pre-quantized gate for w_down matmul
    /// Pre-quantized attn_out for wo projection (aarch64 Q8_0 path)
    attn_out_q8: Vec<u8>,
    /// Pre-quantized normed for output projection (aarch64 Q8_0 path)
    final_normed_q8: Vec<u8>,
    // Cached model dimensions (computed once in init, never changes).
    // Eliminates per-call hp() Option check, as-usize casts, and derived multiplications.
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter_dim: usize,
    eps: f32,
    q_dim: usize,   // num_heads * head_dim
    kv_dim: usize,   // num_kv_heads * head_dim
    gqa_ratio: usize, // num_heads / num_kv_heads
    vocab_size: usize,
    max_seq_len: usize,
}

/// Apply Rotary Position Embeddings using pre-computed cos/sin tables.
/// Uses NEON vld2q/vst2q deinterleave intrinsics on aarch64 to process
/// 4 rotation pairs (8 elements) per iteration.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
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

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let pairs4 = half_dim / 4; // 4 pairs = 8 elements per NEON iteration
        let tail_start = pairs4 * 4;

        unsafe {
            // Q heads
            for h in 0..num_q_heads {
                let head_start = h * head_dim;
                for p in 0..pairs4 {
                    let i = p * 4;
                    let idx = head_start + 2 * i;

                    // vld2q deinterleaves: .0 = evens (v0), .1 = odds (v1)
                    let interleaved = vld2q_f32(q.as_ptr().add(idx));
                    let evens = interleaved.0;
                    let odds = interleaved.1;

                    let cos_vec = vld1q_f32(rope_cos.as_ptr().add(cos_base + i));
                    let sin_vec = vld1q_f32(rope_sin.as_ptr().add(sin_base + i));

                    // new_even = even * cos - odd * sin
                    let neg_sin = vnegq_f32(sin_vec);
                    let new_evens = vfmaq_f32(vmulq_f32(evens, cos_vec), odds, neg_sin);
                    // new_odd = odd * cos + even * sin  (= even * sin + odd * cos)
                    let new_odds = vfmaq_f32(vmulq_f32(odds, cos_vec), evens, sin_vec);

                    // vst2q interleaves back: pairs new_evens/new_odds
                    let result = float32x4x2_t(new_evens, new_odds);
                    vst2q_f32(q.as_mut_ptr().add(idx), result);
                }
                // Scalar tail for remaining pairs
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
            // K heads
            for h in 0..num_kv_heads {
                let head_start = h * head_dim;
                for p in 0..pairs4 {
                    let i = p * 4;
                    let idx = head_start + 2 * i;

                    let interleaved = vld2q_f32(k.as_ptr().add(idx));
                    let evens = interleaved.0;
                    let odds = interleaved.1;

                    let cos_vec = vld1q_f32(rope_cos.as_ptr().add(cos_base + i));
                    let sin_vec = vld1q_f32(rope_sin.as_ptr().add(sin_base + i));

                    let neg_sin = vnegq_f32(sin_vec);
                    let new_evens = vfmaq_f32(vmulq_f32(evens, cos_vec), odds, neg_sin);
                    let new_odds = vfmaq_f32(vmulq_f32(odds, cos_vec), evens, sin_vec);

                    let result = float32x4x2_t(new_evens, new_odds);
                    vst2q_f32(k.as_mut_ptr().add(idx), result);
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

/// RMSNorm using f32 weight slice (for global tensors, not byte-encoded).
/// Uses SIMD dot product for the mean-of-squares accumulation, then NEON multiply.
#[inline(always)]
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let ms = simd_kernels::dot_product_simd(x, x) / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let remainder = n % 4;
        unsafe {
            let scale_vec = vdupq_n_f32(scale);
            for c in 0..chunks {
                let offset = c * 4;
                let x_vec = vld1q_f32(x.as_ptr().add(offset));
                let w_vec = vld1q_f32(weight.as_ptr().add(offset));
                let scaled = vmulq_f32(x_vec, scale_vec);
                let result = vmulq_f32(scaled, w_vec);
                vst1q_f32(out.as_mut_ptr().add(offset), result);
            }
            let tail = chunks * 4;
            for j in 0..remainder {
                let idx = tail + j;
                *out.get_unchecked_mut(idx) =
                    *x.get_unchecked(idx) * scale * *weight.get_unchecked(idx);
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            out[i] = x[i] * scale * weight[i];
        }
    }
}

/// NEON-accelerated vector addition: dst[i] += src[i].
/// 4x unrolled (16 floats/iter) for better throughput on wide pipelines.
/// Used for residual connections (practice 4.3 commentary: SIMD all hotpath ops).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vadd_inplace(dst: &mut [f32], src: &[f32]) {
    use std::arch::aarch64::*;
    let n = dst.len().min(src.len());
    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;
    unsafe {
        for c in 0..chunks16 {
            let base = c * 16;
            let d0 = vld1q_f32(dst.as_ptr().add(base));
            let s0 = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), vaddq_f32(d0, s0));
            let d1 = vld1q_f32(dst.as_ptr().add(base + 4));
            let s1 = vld1q_f32(src.as_ptr().add(base + 4));
            vst1q_f32(dst.as_mut_ptr().add(base + 4), vaddq_f32(d1, s1));
            let d2 = vld1q_f32(dst.as_ptr().add(base + 8));
            let s2 = vld1q_f32(src.as_ptr().add(base + 8));
            vst1q_f32(dst.as_mut_ptr().add(base + 8), vaddq_f32(d2, s2));
            let d3 = vld1q_f32(dst.as_ptr().add(base + 12));
            let s3 = vld1q_f32(src.as_ptr().add(base + 12));
            vst1q_f32(dst.as_mut_ptr().add(base + 12), vaddq_f32(d3, s3));
        }
        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let d = vld1q_f32(dst.as_ptr().add(idx));
            let s = vld1q_f32(src.as_ptr().add(idx));
            vst1q_f32(dst.as_mut_ptr().add(idx), vaddq_f32(d, s));
        }
        for j in 0..remainder {
            let idx = tail_start + j;
            *dst.get_unchecked_mut(idx) += *src.get_unchecked(idx);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn vadd_inplace(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += src[i];
    }
}

/// NEON-accelerated scaled addition: dst[i] += scale * src[i].
/// 4x unrolled (16 floats/iter) for better throughput on wide pipelines.
/// Used for attention value accumulation (the inner d-loop over head_dim).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vscale_add_inplace(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let n = dst.len().min(src.len());
    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;
    unsafe {
        let sv = vdupq_n_f32(scale);
        for c in 0..chunks16 {
            let base = c * 16;
            let d0 = vld1q_f32(dst.as_ptr().add(base));
            let s0 = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), vfmaq_f32(d0, s0, sv));
            let d1 = vld1q_f32(dst.as_ptr().add(base + 4));
            let s1 = vld1q_f32(src.as_ptr().add(base + 4));
            vst1q_f32(dst.as_mut_ptr().add(base + 4), vfmaq_f32(d1, s1, sv));
            let d2 = vld1q_f32(dst.as_ptr().add(base + 8));
            let s2 = vld1q_f32(src.as_ptr().add(base + 8));
            vst1q_f32(dst.as_mut_ptr().add(base + 8), vfmaq_f32(d2, s2, sv));
            let d3 = vld1q_f32(dst.as_ptr().add(base + 12));
            let s3 = vld1q_f32(src.as_ptr().add(base + 12));
            vst1q_f32(dst.as_mut_ptr().add(base + 12), vfmaq_f32(d3, s3, sv));
        }
        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let d = vld1q_f32(dst.as_ptr().add(idx));
            let s = vld1q_f32(src.as_ptr().add(idx));
            vst1q_f32(dst.as_mut_ptr().add(idx), vfmaq_f32(d, s, sv));
        }
        for j in 0..remainder {
            let idx = tail_start + j;
            *dst.get_unchecked_mut(idx) += scale * *src.get_unchecked(idx);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn vscale_add_inplace(dst: &mut [f32], src: &[f32], scale: f32) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += scale * src[i];
    }
}

/// NEON-accelerated in-place scalar multiply: dst[i] *= scale.
/// 4x unrolled (16 floats/iter) for better throughput on wide pipelines.
/// Used to pre-scale the Q vector by 1/sqrt(head_dim) after RoPE, eliminating
/// a per-position multiply from the attention score inner loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vscale_inplace(dst: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;
    unsafe {
        let sv = vdupq_n_f32(scale);
        for c in 0..chunks16 {
            let base = c * 16;
            let d0 = vld1q_f32(dst.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), vmulq_f32(d0, sv));
            let d1 = vld1q_f32(dst.as_ptr().add(base + 4));
            vst1q_f32(dst.as_mut_ptr().add(base + 4), vmulq_f32(d1, sv));
            let d2 = vld1q_f32(dst.as_ptr().add(base + 8));
            vst1q_f32(dst.as_mut_ptr().add(base + 8), vmulq_f32(d2, sv));
            let d3 = vld1q_f32(dst.as_ptr().add(base + 12));
            vst1q_f32(dst.as_mut_ptr().add(base + 12), vmulq_f32(d3, sv));
        }
        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let d = vld1q_f32(dst.as_ptr().add(idx));
            vst1q_f32(dst.as_mut_ptr().add(idx), vmulq_f32(d, sv));
        }
        for j in 0..remainder {
            let idx = tail_start + j;
            *dst.get_unchecked_mut(idx) *= scale;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn vscale_inplace(dst: &mut [f32], scale: f32) {
    for v in dst.iter_mut() {
        *v *= scale;
    }
}

/// SIMD-accelerated F32 compute backend.
///
/// Identical structure to NaiveF32Backend. All SIMD acceleration happens inside
/// kernel calls; the control flow and data layout are unchanged.
///
/// Multi-threading: uses a `ThreadPool` to parallelize large matmuls (Q/K/V
/// projections, FFN, output projection). Small ops (attention dot products)
/// remain single-threaded to avoid spawn overhead.
pub struct SimdF32Backend {
    hyperparams: Option<ModelHyperparams>,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    /// Output projection weights pre-quantized to Q8_0 for faster logits computation.
    /// Quantized once during init() from the F32 output_proj when hidden_dim is a
    /// multiple of 32. Reduces memory bandwidth by ~4x in compute_final().
    /// Empty when hidden_dim is not Q8_0-compatible (e.g., test models with dim=8).
    output_proj_q8: Vec<u8>,
    // SAFETY: SimdF32Backend is only accessed from the inference engine's
    // generate() loop, which is single-threaded. The &self in ComputeBackend
    // trait does not imply concurrent access -- it's for shared ownership
    // (multiple layers share the same backend). Only one compute_layer or
    // compute_final call is active at any time. UnsafeCell removes the ~50ns
    // Mutex lock/unlock overhead per call (x22 layers = ~1.1us/token saved).
    scratch: UnsafeCell<Option<ComputeScratch>>,
    // Cached dimensions set once in init(), used by embed_token without
    // scratch lock or Option<HP> unwrap on every call.
    cached_hidden_dim: usize,
    cached_vocab_size: usize,
    /// Thread pool for parallel matmul dispatch.
    /// Created once at construction, reused for all compute calls.
    pool: ThreadPool,
    /// Per-operation profiling: when true, `compute_layer` records timing for
    /// each operation. Zero-cost when false (no `Instant::now()` calls).
    /// Wrapped in UnsafeCell for the same single-threaded safety reason as scratch.
    profile_enabled: bool,
    profile: UnsafeCell<ComputeProfile>,
}

// SAFETY: SimdF32Backend is accessed single-threaded from the engine's generate()
// loop. The UnsafeCell<Option<ComputeScratch>> is never accessed concurrently --
// compute_layer and compute_final are called sequentially, one at a time.
// The ThreadPool and Vec fields are inherently Send+Sync.
unsafe impl Send for SimdF32Backend {}
unsafe impl Sync for SimdF32Backend {}

impl Default for SimdF32Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdF32Backend {
    /// Create with default thread count (available parallelism - 1 workers).
    pub fn new() -> Self {
        Self::with_threads(0) // 0 = auto-detect
    }

    /// Create with explicit thread count.
    /// `num_threads` = 0 means auto-detect (available parallelism - 1).
    /// `num_threads` = 1 means single-threaded (no worker threads, just caller).
    pub fn with_threads(num_threads: usize) -> Self {
        let pool = if num_threads == 0 {
            ThreadPool::with_default_threads()
        } else {
            // num_threads here is total desired parallelism.
            // ThreadPool::new takes *worker* count (excluding caller).
            // So for num_threads=1, we want 0 workers (but ThreadPool clamps to 1).
            // For num_threads=4, we want 3 workers.
            if num_threads <= 1 {
                // Single-threaded: create pool with 1 worker but should_parallelize
                // will return false for small dims anyway. The parallel_for with
                // total_threads=2 still works correctly for large dims.
                ThreadPool::new(1)
            } else {
                ThreadPool::new(num_threads - 1)
            }
        };

        Self {
            hyperparams: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            output_proj_q8: Vec::new(),
            scratch: UnsafeCell::new(None),
            cached_hidden_dim: 0,
            cached_vocab_size: 0,
            pool,
            profile_enabled: false,
            profile: UnsafeCell::new(ComputeProfile::default()),
        }
    }

    /// Returns the total thread count (workers + caller thread).
    pub fn thread_count(&self) -> usize {
        self.pool.total_threads()
    }
}

impl ComputeBackend for SimdF32Backend {
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
        self.cached_hidden_dim = hyperparams.hidden_dim as usize;
        self.cached_vocab_size = hyperparams.vocab_size as usize;

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

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let gqa_ratio = num_heads / num_kv_heads;

        let scratch = ComputeScratch {
            normed: vec![0.0f32; hidden_dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; q_dim],
            scores: vec![0.0f32; num_heads * max_seq_len],
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
            attn_scale: 1.0 / (head_dim as f32).sqrt(), // MUST: practice 4.3
            // Pre-quantized Q8_0 scratch buffers (sized for hidden_dim, inter_dim, q_dim)
            normed_q8: vec![0u8; hidden_dim.div_ceil(32) * 34],
            gate_q8: vec![0u8; inter_dim.div_ceil(32) * 34],
            attn_out_q8: vec![0u8; q_dim.div_ceil(32) * 34],
            final_normed_q8: vec![0u8; hidden_dim.div_ceil(32) * 34],
            // Cached dimensions — eliminates per-call hp() + as-usize + derived muls
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            inter_dim,
            eps: hyperparams.norm_eps,
            q_dim,
            kv_dim,
            gqa_ratio,
            vocab_size,
            max_seq_len,
        };
        *self.scratch.get_mut() = Some(scratch);

        // Pre-quantize output projection to Q8_0 for faster compute_final.
        // Each row (hidden_dim f32 elements) becomes num_blocks * 34 bytes in Q8_0.
        // This trades ~0.5-1% precision for ~4x less memory bandwidth in the
        // output projection matmul (vocab_size x hidden_dim).
        // Only possible when hidden_dim is a multiple of Q8_0 group size (32).
        #[cfg(target_arch = "aarch64")]
        {
            if hidden_dim % 32 == 0 {
                let num_blocks_out = hidden_dim.div_ceil(32);
                let row_bytes = num_blocks_out * 34;
                let total_bytes = vocab_size * row_bytes;
                let mut q8_buf = vec![0u8; total_bytes];

                for row in 0..vocab_size {
                    let f32_start = row * hidden_dim;
                    let f32_slice = &self.output_proj[f32_start..f32_start + hidden_dim];
                    let q8_start = row * row_bytes;
                    let q8_slice = &mut q8_buf[q8_start..q8_start + row_bytes];
                    simd_kernels::quantize_f32_to_q8_0_pub(f32_slice, q8_slice, hidden_dim);
                }
                self.output_proj_q8 = q8_buf;
            }
        }

        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_layer(
        &self,
        _layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        // SAFETY: compute_layer is only called from the engine's single-threaded
        // generate() loop. No concurrent access to scratch or profile is possible.
        let s = unsafe { &mut *self.scratch.get() }.as_mut().ok_or_else(|| {
            RuntimeError::Compute("scratch not initialized: call init() first".into())
        })?;
        let profile = self.profile_enabled;
        let prof = unsafe { &mut *self.profile.get() };

        // Profile helper macros: zero-cost when profile == false (no Instant::now() calls).
        macro_rules! tick {
            () => { if profile { Some(Instant::now()) } else { None } };
        }
        macro_rules! tock {
            ($start:expr, $field:ident) => {
                if let Some(t) = $start { prof.$field += t.elapsed(); }
            };
        }

        // Read cached dimensions from scratch (no hp() Option check, no as-usize casts)
        let hidden_dim = s.hidden_dim;
        let num_heads = s.num_heads;
        let num_kv_heads = s.num_kv_heads;
        let head_dim = s.head_dim;
        let inter_dim = s.inter_dim;
        let eps = s.eps;
        let q_dim = s.q_dim;
        let kv_dim = s.kv_dim;
        let gqa_ratio = s.gqa_ratio;

        // Zero-copy f32 view of the input activation.
        // We extract a raw pointer + length to avoid holding an immutable borrow on
        // `x` (which we need to mutably borrow later in write_f32_from).
        // SAFETY: x.data is not modified until write_f32_from at the very end of this
        // function. The pointer remains valid for all reads below. Alignment is
        // verified inside as_f32_slice (debug_assert).
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

        // 1+2. Attention RMSNorm + Q/K/V projections (quant-aware, multi-threaded)
        //
        // RMSNorm is computed at the start of each branch below, so it runs
        // exactly once per code path. On the all-Q8_0 path, normed is then
        // quantized once and shared across all three Q/K/V projections.
        //
        // Fused dispatch: When all three projections are Q8_0, we issue a SINGLE
        // parallel_for partitioned by q_dim (the largest output dimension). Each
        // thread computes its chunk of Q, and conditionally its chunk of K and V
        // (threads with start >= kv_dim skip K/V). This eliminates 2 condvar
        // wake+join cycles per layer (~10us each) = ~44 fewer wakes per token
        // across 22 layers.
        #[cfg(target_arch = "aarch64")]
        {
            let all_q8 = st.wq.quant == QuantScheme::Q8_0
                && st.wk.quant == QuantScheme::Q8_0
                && st.wv.quant == QuantScheme::Q8_0;

            if all_q8 {
                // 1. Attention RMSNorm
                let t0 = tick!();
                match st.attn_norm.quant {
                    QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
                    _ => simd_kernels::rmsnorm_bytes_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
                }
                tock!(t0, attn_rmsnorm);

                // Pre-quantize normed ONCE, use for all three Q8_0 projections
                let t0 = tick!();
                simd_kernels::quantize_f32_to_q8_0_pub(&s.normed, &mut s.normed_q8, hidden_dim);
                tock!(t0, quantize_normed_attn);

                // Fused Q+K+V: single dispatch, each thread processes its chunk
                // of all 3 matrices. Partitioned by q_dim (largest dimension).
                // K and V have output dimension kv_dim <= q_dim, so threads whose
                // chunk starts at >= kv_dim skip K/V work.
                let t0 = tick!();
                let num_blocks_hidden = hidden_dim.div_ceil(32);
                let row_bytes_hidden = num_blocks_hidden * 34;

                let q_addr = s.q.as_mut_ptr() as usize;
                let k_addr = s.k.as_mut_ptr() as usize;
                let v_addr = s.v.as_mut_ptr() as usize;
                let normed_q8_ref: &[u8] = &s.normed_q8;

                if self.pool.should_parallelize(q_dim) {
                    self.pool.parallel_for(q_dim, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        // Q projection (always: start..end within q_dim)
                        let q_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (q_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let wq_offset = start * row_bytes_hidden;
                        let wq_sub = &wq_bytes[wq_offset..wq_offset + chunk_len * row_bytes_hidden];
                        simd_kernels::matmul_q8_0_preq(
                            q_slice, wq_sub, normed_q8_ref, chunk_len, hidden_dim,
                        );

                        // K projection (only if this chunk overlaps kv_dim)
                        if start < kv_dim {
                            let k_end = end.min(kv_dim);
                            let k_chunk = k_end - start;
                            let k_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (k_addr as *mut f32).add(start), k_chunk,
                                )
                            };
                            let wk_offset = start * row_bytes_hidden;
                            let wk_sub =
                                &wk_bytes[wk_offset..wk_offset + k_chunk * row_bytes_hidden];
                            simd_kernels::matmul_q8_0_preq(
                                k_slice, wk_sub, normed_q8_ref, k_chunk, hidden_dim,
                            );
                        }

                        // V projection (only if this chunk overlaps kv_dim)
                        if start < kv_dim {
                            let v_end = end.min(kv_dim);
                            let v_chunk = v_end - start;
                            let v_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (v_addr as *mut f32).add(start), v_chunk,
                                )
                            };
                            let wv_offset = start * row_bytes_hidden;
                            let wv_sub =
                                &wv_bytes[wv_offset..wv_offset + v_chunk * row_bytes_hidden];
                            simd_kernels::matmul_q8_0_preq(
                                v_slice, wv_sub, normed_q8_ref, v_chunk, hidden_dim,
                            );
                        }
                    });
                } else {
                    // Single-threaded fallback for small dims
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.q, wq_bytes, normed_q8_ref, q_dim, hidden_dim,
                    );
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.k, wk_bytes, normed_q8_ref, kv_dim, hidden_dim,
                    );
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.v, wv_bytes, normed_q8_ref, kv_dim, hidden_dim,
                    );
                }
                tock!(t0, qkv_matmul);
            } else {
                // Non-all-Q8_0 path: separate RMSNorm (cannot fuse since quantize
                // may not be needed)
                let t0 = tick!();
                match st.attn_norm.quant {
                    QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
                    _ => simd_kernels::rmsnorm_bytes_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
                }
                tock!(t0, attn_rmsnorm);

                // Mixed or F32 quant: separate dispatches (cannot fuse different kernels)
                let t0 = tick!();
                if st.wq.quant == QuantScheme::Q8_0 {
                    simd_kernels::quantize_f32_to_q8_0_pub(&s.normed, &mut s.normed_q8, hidden_dim);
                }
                tock!(t0, quantize_normed_attn);
                let t0 = tick!();
                if st.wq.quant == QuantScheme::Q8_0 {
                    simd_kernels::matmul_q8_0_preq_parallel(&mut s.q, wq_bytes, &s.normed_q8, q_dim, hidden_dim, &self.pool);
                } else {
                    simd_kernels::matmul_bytes_simd_parallel(&mut s.q, wq_bytes, &s.normed, q_dim, hidden_dim, &self.pool);
                }
                if st.wk.quant == QuantScheme::Q8_0 {
                    simd_kernels::matmul_q8_0_preq_parallel(&mut s.k, wk_bytes, &s.normed_q8, kv_dim, hidden_dim, &self.pool);
                } else {
                    simd_kernels::matmul_bytes_simd_parallel(&mut s.k, wk_bytes, &s.normed, kv_dim, hidden_dim, &self.pool);
                }
                if st.wv.quant == QuantScheme::Q8_0 {
                    simd_kernels::matmul_q8_0_preq_parallel(&mut s.v, wv_bytes, &s.normed_q8, kv_dim, hidden_dim, &self.pool);
                } else {
                    simd_kernels::matmul_bytes_simd_parallel(&mut s.v, wv_bytes, &s.normed, kv_dim, hidden_dim, &self.pool);
                }
                tock!(t0, qkv_matmul);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Non-aarch64: separate RMSNorm (no fused path available)
            let t0 = tick!();
            match st.attn_norm.quant {
                QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
                _ => simd_kernels::rmsnorm_bytes_simd(&mut s.normed, x_f32, attn_norm_bytes, eps),
            }
            tock!(t0, attn_rmsnorm);

            let t0 = tick!();
            match st.wq.quant {
                QuantScheme::Q8_0 => simd_kernels::matmul_q8_0_simd_parallel(&mut s.q, wq_bytes, &s.normed, q_dim, hidden_dim, &self.pool),
                _ => simd_kernels::matmul_bytes_simd_parallel(&mut s.q, wq_bytes, &s.normed, q_dim, hidden_dim, &self.pool),
            }
            match st.wk.quant {
                QuantScheme::Q8_0 => simd_kernels::matmul_q8_0_simd_parallel(&mut s.k, wk_bytes, &s.normed, kv_dim, hidden_dim, &self.pool),
                _ => simd_kernels::matmul_bytes_simd_parallel(&mut s.k, wk_bytes, &s.normed, kv_dim, hidden_dim, &self.pool),
            }
            match st.wv.quant {
                QuantScheme::Q8_0 => simd_kernels::matmul_q8_0_simd_parallel(&mut s.v, wv_bytes, &s.normed, kv_dim, hidden_dim, &self.pool),
                _ => simd_kernels::matmul_bytes_simd_parallel(&mut s.v, wv_bytes, &s.normed, kv_dim, hidden_dim, &self.pool),
            }
            tock!(t0, qkv_matmul);
        }

        // 3. Apply RoPE (pre-computed tables, NEON vectorized on aarch64)
        let t0 = tick!();
        apply_rope_precomputed(
            &mut s.q, &mut s.k,
            num_heads, num_kv_heads, head_dim, seq_pos,
            &s.rope_cos, &s.rope_sin, s.half_dim,
        );
        // 3b. Pre-scale Q by attention scale factor (1/sqrt(head_dim), practice 4.3).
        vscale_inplace(&mut s.q, s.attn_scale);
        tock!(t0, rope);

        // 4. Append k,v to KV cache
        let t0 = tick!();
        let kv = kv.ok_or_else(|| {
            RuntimeError::Compute("KV cache view required for attention".into())
        })?;
        kv.append_keys(&s.k);
        kv.append_values(&s.v);
        let new_seq_len = (kv.seq_len + 1).min(kv.max_seq_len);
        tock!(t0, kv_cache_write);

        // 5. Multi-head attention with GQA (parallel across heads)
        //
        // Each head's computation is independent: it reads from a disjoint region of Q
        // and writes to a disjoint region of attn_out. The scores buffer is partitioned
        // so head h uses scores[h * max_seq_len .. h * max_seq_len + new_seq_len].
        // KV cache access (keys_f32_slice, values_f32_slice) is read-only and thread-safe.
        // dot_product_simd, softmax_inplace_simd, vscale_add_inplace are all stateless.
        //
        // Q is pre-scaled by attn_scale in step 3b, so no per-position scale multiply needed.
        let t0 = tick!();
        let max_seq_len = s.max_seq_len;
        let kv_is_f16 = kv.precision == KvPrecision::F16;
        let bytes_per_elem = kv.precision.bytes_per_element();
        let head_dim_bytes = head_dim * bytes_per_elem;
        let kv_max_seq_len = kv.max_seq_len;

        if num_heads >= self.pool.total_threads() && self.pool.total_threads() > 1 && new_seq_len >= 64 {
            // Parallel path: each thread processes a range of heads.
            // Uses parallel_for_heads which skips the row-count threshold check
            // (num_heads=16 is below PARALLEL_THRESHOLD=256, but each head does
            // O(new_seq_len * head_dim) work which is expensive enough to justify
            // the thread wake cost).
            //
            // SAFETY: Each head reads from disjoint Q[h*head_dim..(h+1)*head_dim] and
            // writes to disjoint attn_out[h*head_dim..(h+1)*head_dim] and
            // scores[h*max_seq_len..(h+1)*max_seq_len]. KV cache is read-only.
            let q_ptr = s.q.as_ptr() as usize;
            let attn_out_ptr = s.attn_out.as_mut_ptr() as usize;
            let scores_ptr = s.scores.as_mut_ptr() as usize;
            // KV cache pointers for zero-copy slice access
            let kv_keys_ptr = kv.keys.as_ptr() as usize;
            let kv_keys_len = kv.keys.len();
            let kv_values_ptr = kv.values.as_ptr() as usize;
            let kv_values_len = kv.values.len();

            self.pool.parallel_for_heads(num_heads, |start_head, end_head| {
                for h in start_head..end_head {
                    let kv_h = h / gqa_ratio;
                    // Head-first layout: per-head base byte offset
                    let kv_head_byte_base = kv_h * kv_max_seq_len * head_dim_bytes;

                    // SAFETY: q_ptr + h*head_dim is within the q allocation (size = num_heads * head_dim).
                    // Each head reads a disjoint slice.
                    let q_head = unsafe {
                        std::slice::from_raw_parts(
                            (q_ptr as *const f32).add(h * head_dim),
                            head_dim,
                        )
                    };

                    // SAFETY: scores_ptr + h*max_seq_len is within the scores allocation
                    // (size = num_heads * max_seq_len). Each head writes a disjoint region.
                    let head_scores = unsafe {
                        std::slice::from_raw_parts_mut(
                            (scores_ptr as *mut f32).add(h * max_seq_len),
                            new_seq_len,
                        )
                    };

                    // Compute attention scores using SIMD dot product.
                    // Q is already scaled by 1/sqrt(head_dim), so scores = Q . K directly.
                    // Head-first layout: K[kv_h] is contiguous at base + t*head_dim_bytes.
                    for t in 0..new_seq_len {
                        let k_byte_start = kv_head_byte_base + t * head_dim_bytes;
                        let dot = if kv_is_f16 {
                            debug_assert!(k_byte_start + head_dim * 2 <= kv_keys_len);
                            unsafe {
                                let src = (kv_keys_ptr as *const u8).add(k_byte_start);
                                simd_kernels::dot_product_f16_f32_simd(q_head, src, head_dim)
                            }
                        } else {
                            debug_assert!(k_byte_start + head_dim * 4 <= kv_keys_len);
                            let k_slice = unsafe {
                                std::slice::from_raw_parts(
                                    (kv_keys_ptr as *const u8).add(k_byte_start) as *const f32,
                                    head_dim,
                                )
                            };
                            simd_kernels::dot_product_simd(q_head, k_slice)
                        };
                        unsafe { *head_scores.get_unchecked_mut(t) = dot; }
                    }

                    // Softmax over scores (SIMD, best practice 4.1)
                    simd_kernels::softmax_inplace_simd(head_scores);

                    // Weighted sum of values (SIMD FMA: out += score * v_slice)
                    // SAFETY: attn_out_ptr + h*head_dim is within the attn_out allocation.
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (attn_out_ptr as *mut f32).add(h * head_dim),
                            head_dim,
                        )
                    };
                    out_slice.fill(0.0);

                    for t in 0..new_seq_len {
                        let score = unsafe { *head_scores.get_unchecked(t) };
                        let v_byte_start = kv_head_byte_base + t * head_dim_bytes;
                        if kv_is_f16 {
                            debug_assert!(v_byte_start + head_dim * 2 <= kv_values_len);
                            unsafe {
                                let src = (kv_values_ptr as *const u8).add(v_byte_start);
                                simd_kernels::vscale_add_f16_f32_inplace(out_slice, src, score, head_dim);
                            }
                        } else {
                            debug_assert!(v_byte_start + head_dim * 4 <= kv_values_len);
                            let v_slice = unsafe {
                                std::slice::from_raw_parts(
                                    (kv_values_ptr as *const u8).add(v_byte_start) as *const f32,
                                    head_dim,
                                )
                            };
                            vscale_add_inplace(out_slice, v_slice, score);
                        }
                    }
                }
            });
        } else {
            // Serial fallback for small head counts or when parallelism isn't beneficial.
            for h in 0..num_heads {
                let kv_h = h / gqa_ratio;
                // Head-first layout: compute per-head base element offset
                let kv_head_elem_base = kv_h * kv_max_seq_len * head_dim;

                let q_head_ptr = unsafe { s.q.as_ptr().add(h * head_dim) };
                let q_head = unsafe { std::slice::from_raw_parts(q_head_ptr, head_dim) };

                // Each head uses its own region of the scores buffer.
                let scores_start = h * max_seq_len;

                // Compute attention scores. Q is pre-scaled, so no per-position multiply.
                // Head-first layout: K[kv_h] is contiguous at base + t*head_dim.
                for t in 0..new_seq_len {
                    let k_start = kv_head_elem_base + t * head_dim;
                    let dot = if kv_is_f16 {
                        let k_byte_start = k_start * 2;
                        unsafe {
                            let src = kv.keys.as_ptr().add(k_byte_start);
                            simd_kernels::dot_product_f16_f32_simd(q_head, src, head_dim)
                        }
                    } else {
                        #[cfg(target_endian = "little")]
                        let k_slice = kv.keys_f32_slice(k_start, head_dim);
                        #[cfg(not(target_endian = "little"))]
                        let k_slice = {
                            kv.read_keys_f32_into(&mut s.k_head_buf, k_start, head_dim);
                            &s.k_head_buf[..head_dim]
                        };
                        simd_kernels::dot_product_simd(q_head, k_slice)
                    };
                    unsafe { *s.scores.get_unchecked_mut(scores_start + t) = dot; }
                }

                // Softmax over scores (SIMD, best practice 4.1)
                simd_kernels::softmax_inplace_simd(
                    &mut s.scores[scores_start..scores_start + new_seq_len],
                );

                // Weighted sum of values (SIMD FMA: out += score * v_slice)
                let out_ptr = unsafe { s.attn_out.as_mut_ptr().add(h * head_dim) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, head_dim) };
                out_slice.fill(0.0);

                for t in 0..new_seq_len {
                    let score = unsafe { *s.scores.get_unchecked(scores_start + t) };
                    let v_start = kv_head_elem_base + t * head_dim;
                    if kv_is_f16 {
                        let v_byte_start = v_start * 2;
                        unsafe {
                            let src = kv.values.as_ptr().add(v_byte_start);
                            simd_kernels::vscale_add_f16_f32_inplace(out_slice, src, score, head_dim);
                        }
                    } else {
                        #[cfg(target_endian = "little")]
                        let v_slice = kv.values_f32_slice(v_start, head_dim);
                        #[cfg(not(target_endian = "little"))]
                        let v_slice = {
                            kv.read_values_f32_into(&mut s.v_head_buf, v_start, head_dim);
                            &s.v_head_buf[..head_dim]
                        };
                        vscale_add_inplace(out_slice, v_slice, score);
                    }
                }
            }
        }

        tock!(t0, attention);

        // 6+7. Fused Wo projection + residual add (single parallel dispatch).
        //
        // Instead of: matmul_parallel(attn_proj, wo, attn_out) THEN vadd(attn_proj, x_f32),
        // each thread computes its matmul chunk AND adds the residual in-place.
        // Eliminates one serial pass over hidden_dim (saves ~1-3us per layer).
        let t0 = tick!();
        #[cfg(target_arch = "aarch64")]
        {
            if st.wo.quant == QuantScheme::Q8_0 {
                let num_blocks_q = q_dim.div_ceil(32);
                let row_bytes_q = num_blocks_q * 34;
                let attn_proj_addr = s.attn_proj.as_mut_ptr() as usize;

                if self.pool.should_parallelize(hidden_dim) {
                    // Fused quantize + matmul + residual: the first thread to enter
                    // quantizes attn_out while other threads are still waking up from
                    // the spin-wait loop (~1.5us quantize overlaps with thread wake).
                    // All threads then spin briefly until quantization completes,
                    // then proceed with their matmul chunk.
                    //
                    // Uses a 3-state protocol (0=unclaimed, 1=in-progress, 2=done)
                    // to separate "claimed the work" from "work is finished".
                    let quant_state = AtomicU8::new(0);
                    let attn_out_ref: &[f32] = &s.attn_out;
                    let attn_out_q8_ptr = s.attn_out_q8.as_mut_ptr() as usize;
                    let attn_out_q8_size = s.attn_out_q8.len();

                    self.pool.parallel_for(hidden_dim, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        // First thread claims the quantization work (CAS 0 -> 1).
                        if quant_state.compare_exchange(
                            0, 1, AtomicOrdering::AcqRel, AtomicOrdering::Acquire,
                        ).is_ok() {
                            // SAFETY: attn_out_q8 is not read until after the Release
                            // store below, and only one thread enters this block.
                            unsafe {
                                let q8_buf = std::slice::from_raw_parts_mut(
                                    attn_out_q8_ptr as *mut u8, attn_out_q8_size,
                                );
                                simd_kernels::quantize_f32_to_q8_0_pub(
                                    attn_out_ref, q8_buf, q_dim,
                                );
                            }
                            // Signal done (1 -> 2). Release ensures all q8 writes
                            // are visible to threads that Acquire-load state == 2.
                            quant_state.store(2, AtomicOrdering::Release);
                        } else {
                            // Another thread claimed it. Spin until done (state == 2).
                            while quant_state.load(AtomicOrdering::Acquire) != 2 {
                                std::hint::spin_loop();
                            }
                        }

                        // Now attn_out_q8 is fully written. All threads proceed.
                        let attn_out_q8_ref = unsafe {
                            std::slice::from_raw_parts(
                                attn_out_q8_ptr as *const u8, attn_out_q8_size,
                            )
                        };

                        // SAFETY: Each thread writes to a disjoint range of attn_proj[].
                        // wo_bytes, attn_out_q8_ref, x_f32 are read-only shared references.
                        let proj_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_proj_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let wo_offset = start * row_bytes_q;
                        let wo_sub = &wo_bytes[wo_offset..wo_offset + chunk_len * row_bytes_q];
                        simd_kernels::matmul_q8_0_preq(
                            proj_slice, wo_sub, attn_out_q8_ref, chunk_len, q_dim,
                        );
                        // Fused residual: attn_proj[chunk] += x_f32[chunk]
                        let x_chunk = &x_f32[start..end];
                        vadd_inplace(proj_slice, x_chunk);
                    });
                } else {
                    // Single-threaded fallback: quantize then matmul sequentially.
                    simd_kernels::quantize_f32_to_q8_0_pub(&s.attn_out, &mut s.attn_out_q8, q_dim);
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.attn_proj, wo_bytes, &s.attn_out_q8, hidden_dim, q_dim,
                    );
                    vadd_inplace(&mut s.attn_proj, x_f32);
                }
            } else {
                let attn_proj_addr = s.attn_proj.as_mut_ptr() as usize;
                let attn_out_ref: &[f32] = &s.attn_out;

                if self.pool.should_parallelize(hidden_dim) {
                    self.pool.parallel_for(hidden_dim, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        let proj_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_proj_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let w_byte_offset = start * q_dim * 4;
                        let wo_sub = &wo_bytes[w_byte_offset..w_byte_offset + chunk_len * q_dim * 4];
                        simd_kernels::matmul_bytes_simd_2row(
                            proj_slice, wo_sub, attn_out_ref, chunk_len, q_dim,
                        );
                        let x_chunk = &x_f32[start..end];
                        vadd_inplace(proj_slice, x_chunk);
                    });
                } else {
                    simd_kernels::matmul_bytes_simd_2row(
                        &mut s.attn_proj, wo_bytes, attn_out_ref, hidden_dim, q_dim,
                    );
                    vadd_inplace(&mut s.attn_proj, x_f32);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            match st.wo.quant {
                QuantScheme::Q8_0 => simd_kernels::matmul_q8_0_simd_parallel(&mut s.attn_proj, wo_bytes, &s.attn_out, hidden_dim, q_dim, &self.pool),
                _ => simd_kernels::matmul_bytes_simd_parallel(&mut s.attn_proj, wo_bytes, &s.attn_out, hidden_dim, q_dim, &self.pool),
            }
            vadd_inplace(&mut s.attn_proj, x_f32);
        }
        tock!(t0, wo_matmul);
        // Note: residual_attn timing is now included in wo_matmul (fused).
        // The profile field residual_attn will read zero for fused paths.

        // 8. FFN RMSNorm (SIMD, quant-aware)
        //
        // RMSNorm followed by separate Q8_0 quantize when gate weights are Q8_0.
        // Quantize is timed under quantize_normed_ffn on aarch64.

        // 9. SwiGLU MLP: fused gate+up+SwiGLU+quantize parallel dispatch
        //
        // Three-level fusion eliminates serial bottlenecks between matmul and w_down:
        //   (a) gate+up matmuls share the same input vector `normed` -> single dispatch
        //   (b) SwiGLU is applied per-chunk (no serial pass over inter_dim)
        //   (c) gate quantization (aarch64 Q8_0) is fused per-chunk when 32-aligned
        //
        // This eliminates:
        //   - 1 serial SwiGLU pass over inter_dim (~5632 elements)
        //   - 1 serial quantize_f32_to_q8_0 pass over inter_dim (when 32-aligned)
        // Savings: ~3-7us per layer = ~66-154us per 22-layer token.
        //
        // Pre-quantize optimization (aarch64, Q8_0 only): quantize normed ONCE before
        // dispatch, then each thread uses matmul_q8_0_preq (no-alloc, no-quantize).

        // Determine if we can fuse gate quantization into the parallel dispatch.
        // Requirements: aarch64, Q8_0 w_down, and inter_dim is a multiple of 32.
        // When inter_dim % 32 == 0, chunk boundaries from parallel_for are verified
        // per-chunk to ensure 32-alignment before fusing quantization.
        #[cfg(target_arch = "aarch64")]
        let fuse_gate_quant = st.w_down.quant == QuantScheme::Q8_0 && inter_dim % 32 == 0;
        #[cfg(not(target_arch = "aarch64"))]
        let _fuse_gate_quant = false;

        {
            let gate_addr = s.gate.as_mut_ptr() as usize;
            let up_addr = s.up.as_mut_ptr() as usize;
            let is_q8 = st.w_gate.quant == QuantScheme::Q8_0;

            // Fused RMSNorm + quantize on aarch64 when gate is Q8_0 and norm is F32.
            // Produces both normed f32 AND normed_q8 in a single pass.
            // On non-aarch64 or when conditions are not met, separate calls.
            #[cfg(target_arch = "aarch64")]
            {
                let t0 = tick!();
                match st.ffn_norm.quant {
                    QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(&mut s.normed, &s.attn_proj, ffn_norm_bytes, eps),
                    _ => simd_kernels::rmsnorm_bytes_simd(&mut s.normed, &s.attn_proj, ffn_norm_bytes, eps),
                }
                tock!(t0, ffn_rmsnorm);
                if is_q8 {
                    let t0 = tick!();
                    simd_kernels::quantize_f32_to_q8_0_pub(&s.normed, &mut s.normed_q8, hidden_dim);
                    tock!(t0, quantize_normed_ffn);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let t0 = tick!();
                match st.ffn_norm.quant {
                    QuantScheme::Q8_0 => simd_kernels::rmsnorm_q8_0_simd(&mut s.normed, &s.attn_proj, ffn_norm_bytes, eps),
                    _ => simd_kernels::rmsnorm_bytes_simd(&mut s.normed, &s.attn_proj, ffn_norm_bytes, eps),
                }
                tock!(t0, ffn_rmsnorm);
            }

            let normed_ref: &[f32] = &s.normed;

            #[cfg(target_arch = "aarch64")]
            let normed_q8_ref: &[u8] = &s.normed_q8;

            // gate_q8 pointer for fused quantization (aarch64 only).
            // Each thread writes to a disjoint 34-byte-aligned region of gate_q8.
            #[cfg(target_arch = "aarch64")]
            let gate_q8_addr = s.gate_q8.as_mut_ptr() as usize;

            let t0 = tick!();
            if self.pool.should_parallelize(inter_dim) {
                self.pool.parallel_for(inter_dim, |start, end| {
                    let chunk_len = end - start;
                    if chunk_len == 0 {
                        return;
                    }

                    // SAFETY: Each thread writes to a disjoint range of gate[] and up[].
                    // normed, normed_q8, w_gate_bytes, w_up_bytes are read-only shared references.
                    let gate_slice = unsafe {
                        std::slice::from_raw_parts_mut((gate_addr as *mut f32).add(start), chunk_len)
                    };
                    let up_slice = unsafe {
                        std::slice::from_raw_parts_mut((up_addr as *mut f32).add(start), chunk_len)
                    };

                    if is_q8 {
                        let num_blocks = hidden_dim.div_ceil(32); // Q8_0_GROUP_SIZE = 32
                        let row_bytes = num_blocks * 34; // Q8_0_BLOCK_SIZE = 34
                        let w_byte_offset = start * row_bytes;
                        let w_gate_sub = &w_gate_bytes[w_byte_offset..w_byte_offset + chunk_len * row_bytes];
                        let w_up_sub = &w_up_bytes[w_byte_offset..w_byte_offset + chunk_len * row_bytes];
                        // Use pre-quantized path: no allocation, no re-quantization per chunk.
                        #[cfg(target_arch = "aarch64")]
                        {
                            simd_kernels::matmul_q8_0_preq(gate_slice, w_gate_sub, normed_q8_ref, chunk_len, hidden_dim);
                            simd_kernels::matmul_q8_0_preq(up_slice, w_up_sub, normed_q8_ref, chunk_len, hidden_dim);
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            simd_kernels::matmul_q8_0_simd_2row(gate_slice, w_gate_sub, normed_ref, chunk_len, hidden_dim);
                            simd_kernels::matmul_q8_0_simd_2row(up_slice, w_up_sub, normed_ref, chunk_len, hidden_dim);
                        }
                    } else {
                        let w_byte_offset = start * hidden_dim * 4;
                        let w_gate_sub = &w_gate_bytes[w_byte_offset..w_byte_offset + chunk_len * hidden_dim * 4];
                        let w_up_sub = &w_up_bytes[w_byte_offset..w_byte_offset + chunk_len * hidden_dim * 4];
                        simd_kernels::matmul_bytes_simd_2row(gate_slice, w_gate_sub, normed_ref, chunk_len, hidden_dim);
                        simd_kernels::matmul_bytes_simd_2row(up_slice, w_up_sub, normed_ref, chunk_len, hidden_dim);
                    }

                    // Fused SwiGLU: apply silu(gate) * up per-chunk, eliminating
                    // the serial pass over inter_dim after the parallel dispatch.
                    simd_kernels::swiglu_inplace_simd(gate_slice, up_slice);

                    // Fused gate quantization (aarch64, Q8_0 w_down, 32-aligned chunks):
                    // quantize this chunk of gate directly into gate_q8, eliminating
                    // the serial quantize_f32_to_q8_0_pub call after the dispatch.
                    #[cfg(target_arch = "aarch64")]
                    if fuse_gate_quant && start % 32 == 0 && chunk_len % 32 == 0 {
                        let block_start = start / 32;
                        let q8_offset = block_start * 34; // 34 bytes per Q8_0 block
                        let q8_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (gate_q8_addr as *mut u8).add(q8_offset),
                                chunk_len.div_ceil(32) * 34,
                            )
                        };
                        simd_kernels::quantize_f32_to_q8_0_pub(gate_slice, q8_slice, chunk_len);
                    }
                });
            } else {
                // Below parallel threshold: single-threaded path
                if is_q8 {
                    #[cfg(target_arch = "aarch64")]
                    {
                        simd_kernels::matmul_q8_0_preq(&mut s.gate, w_gate_bytes, normed_q8_ref, inter_dim, hidden_dim);
                        simd_kernels::matmul_q8_0_preq(&mut s.up, w_up_bytes, normed_q8_ref, inter_dim, hidden_dim);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        simd_kernels::matmul_q8_0_simd_2row(&mut s.gate, w_gate_bytes, normed_ref, inter_dim, hidden_dim);
                        simd_kernels::matmul_q8_0_simd_2row(&mut s.up, w_up_bytes, normed_ref, inter_dim, hidden_dim);
                    }
                } else {
                    simd_kernels::matmul_bytes_simd_2row(&mut s.gate, w_gate_bytes, normed_ref, inter_dim, hidden_dim);
                    simd_kernels::matmul_bytes_simd_2row(&mut s.up, w_up_bytes, normed_ref, inter_dim, hidden_dim);
                }
                // Single-threaded: SwiGLU applied after matmuls (cannot fuse into dispatch)
                simd_kernels::swiglu_inplace_simd(&mut s.gate, &s.up);
                // Single-threaded gate quantization (fused here to avoid separate step)
                #[cfg(target_arch = "aarch64")]
                if fuse_gate_quant {
                    simd_kernels::quantize_f32_to_q8_0_pub(&s.gate, &mut s.gate_q8, inter_dim);
                }
            }
            tock!(t0, gate_up_matmul);
        }

        // 9b+10. Fused Down projection + FFN residual add (single parallel dispatch).
        //
        // Instead of: matmul_parallel(down, w_down, gate_q8) THEN vadd(attn_proj, down),
        // each thread computes its down chunk AND adds it to attn_proj in-place.
        // Eliminates one serial pass over hidden_dim (saves ~1-3us per layer).
        let t0 = tick!();
        #[cfg(target_arch = "aarch64")]
        {
            if st.w_down.quant == QuantScheme::Q8_0 {
                if !fuse_gate_quant {
                    simd_kernels::quantize_f32_to_q8_0_pub(&s.gate, &mut s.gate_q8, inter_dim);
                }
                let gate_q8_ref: &[u8] = &s.gate_q8;
                let num_blocks_inter = inter_dim.div_ceil(32);
                let row_bytes_inter = num_blocks_inter * 34;
                let down_addr = s.down.as_mut_ptr() as usize;
                let attn_proj_addr = s.attn_proj.as_mut_ptr() as usize;

                if self.pool.should_parallelize(hidden_dim) {
                    self.pool.parallel_for(hidden_dim, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        // SAFETY: Each thread writes to disjoint ranges of down[] and attn_proj[].
                        // w_down_bytes and gate_q8_ref are read-only shared references.
                        let down_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (down_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let wd_offset = start * row_bytes_inter;
                        let wd_sub = &w_down_bytes[wd_offset..wd_offset + chunk_len * row_bytes_inter];
                        simd_kernels::matmul_q8_0_preq(
                            down_slice, wd_sub, gate_q8_ref, chunk_len, inter_dim,
                        );
                        // Fused residual: attn_proj[chunk] += down[chunk]
                        let proj_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_proj_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        vadd_inplace(proj_slice, down_slice);
                    });
                } else {
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.down, w_down_bytes, gate_q8_ref, hidden_dim, inter_dim,
                    );
                    vadd_inplace(&mut s.attn_proj, &s.down);
                }
            } else {
                let down_addr = s.down.as_mut_ptr() as usize;
                let attn_proj_addr = s.attn_proj.as_mut_ptr() as usize;
                let gate_ref: &[f32] = &s.gate;

                if self.pool.should_parallelize(hidden_dim) {
                    self.pool.parallel_for(hidden_dim, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        let down_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (down_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let w_byte_offset = start * inter_dim * 4;
                        let wd_sub = &w_down_bytes[w_byte_offset..w_byte_offset + chunk_len * inter_dim * 4];
                        simd_kernels::matmul_bytes_simd_2row(
                            down_slice, wd_sub, gate_ref, chunk_len, inter_dim,
                        );
                        let proj_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_proj_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        vadd_inplace(proj_slice, down_slice);
                    });
                } else {
                    simd_kernels::matmul_bytes_simd_2row(
                        &mut s.down, w_down_bytes, gate_ref, hidden_dim, inter_dim,
                    );
                    vadd_inplace(&mut s.attn_proj, &s.down);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            match st.w_down.quant {
                QuantScheme::Q8_0 => simd_kernels::matmul_q8_0_simd_parallel(&mut s.down, w_down_bytes, &s.gate, hidden_dim, inter_dim, &self.pool),
                _ => simd_kernels::matmul_bytes_simd_parallel(&mut s.down, w_down_bytes, &s.gate, hidden_dim, inter_dim, &self.pool),
            }
            vadd_inplace(&mut s.attn_proj, &s.down);
        }
        tock!(t0, down_matmul);
        // Note: residual_ffn timing is now included in down_matmul (fused).
        // The profile field residual_ffn will read zero for fused paths.

        // Update KV cache seq_len
        kv.seq_len = new_seq_len;

        // Increment profiled call counter
        if profile { prof.calls += 1; }

        // Write result back to activation buffer (reuse existing allocation)
        x.write_f32_from(&s.attn_proj);
        Ok(())
    }

    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        // SAFETY: compute_final is only called from the engine's single-threaded
        // generate() loop. No concurrent access to scratch is possible.
        let s = unsafe { &mut *self.scratch.get() }.as_mut().ok_or_else(|| {
            RuntimeError::Compute("scratch not initialized: call init() first".into())
        })?;

        // Read cached dimensions from scratch (no hp() Option check)
        let hidden_dim = s.hidden_dim;
        let vocab_size = s.vocab_size;
        let eps = s.eps;

        // Zero-copy f32 view of the final hidden state (no memcpy)
        let x_f32 = x.as_f32_slice();

        // Final RMSNorm (using global tensors which are already Vec<f32>)
        // Uses SIMD dot_product for the mean-of-squares
        rmsnorm(&mut s.normed, x_f32, &self.final_norm, eps);

        // Project to vocab logits.
        // Use Q8_0 output projection when available (~4x less memory bandwidth).
        // Pre-quantize normed into scratch buffer to eliminate the last heap allocation
        // in the hot path (matmul_q8_0_simd_parallel internally allocates x_q8).
        // Falls back to F32 matmul when hidden_dim is not Q8_0-compatible or on non-aarch64.
        #[cfg(target_arch = "aarch64")]
        {
            if !self.output_proj_q8.is_empty() {
                // Fused quantize + matmul: instead of serial quantize then parallel
                // matmul, the first thread to enter the dispatch quantizes normed
                // while other threads are still waking up (~1.5us overlap).
                let num_blocks = hidden_dim.div_ceil(32);
                let row_bytes = num_blocks * 34;
                let normed_ref: &[f32] = &s.normed;
                let final_normed_q8_ptr = s.final_normed_q8.as_mut_ptr() as usize;
                let final_normed_q8_size = s.final_normed_q8.len();
                let logits_addr = s.logits.as_mut_ptr() as usize;
                let output_proj_q8_ref: &[u8] = &self.output_proj_q8;

                if self.pool.should_parallelize(vocab_size) {
                    // Fused quantize+matmul: the first thread to arrive quantizes
                    // normed while other threads spin-wait. Uses a 3-state protocol:
                    //   0 = unclaimed, 1 = in-progress, 2 = done.
                    // The previous AtomicBool approach had a race: swap(true) made
                    // quantized==true immediately, so other threads' spin condition
                    // `while !quantized.load()` exited before quantization completed.
                    // The 3-state protocol separates "claimed" (1) from "done" (2).
                    let quant_state = AtomicU8::new(0);

                    self.pool.parallel_for(vocab_size, |start, end| {
                        let chunk_len = end - start;
                        if chunk_len == 0 { return; }

                        // First thread claims the quantization work (CAS 0 -> 1).
                        if quant_state.compare_exchange(
                            0, 1, AtomicOrdering::AcqRel, AtomicOrdering::Acquire,
                        ).is_ok() {
                            unsafe {
                                let q8_buf = std::slice::from_raw_parts_mut(
                                    final_normed_q8_ptr as *mut u8, final_normed_q8_size,
                                );
                                simd_kernels::quantize_f32_to_q8_0_pub(
                                    normed_ref, q8_buf, hidden_dim,
                                );
                            }
                            // Signal done (1 -> 2). Release ensures all q8 writes
                            // are visible to threads that Acquire-load state == 2.
                            quant_state.store(2, AtomicOrdering::Release);
                        } else {
                            // Another thread claimed it. Spin until done (state == 2).
                            while quant_state.load(AtomicOrdering::Acquire) != 2 {
                                std::hint::spin_loop();
                            }
                        }

                        let q8_ref = unsafe {
                            std::slice::from_raw_parts(
                                final_normed_q8_ptr as *const u8, final_normed_q8_size,
                            )
                        };
                        let out_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                (logits_addr as *mut f32).add(start), chunk_len,
                            )
                        };
                        let w_offset = start * row_bytes;
                        let w_sub = &output_proj_q8_ref[w_offset..w_offset + chunk_len * row_bytes];
                        simd_kernels::matmul_q8_0_preq(
                            out_slice, w_sub, q8_ref, chunk_len, hidden_dim,
                        );
                    });
                } else {
                    // Single-threaded: quantize then matmul sequentially.
                    simd_kernels::quantize_f32_to_q8_0_pub(normed_ref, &mut s.final_normed_q8, hidden_dim);
                    simd_kernels::matmul_q8_0_preq(
                        &mut s.logits, output_proj_q8_ref, &s.final_normed_q8,
                        vocab_size, hidden_dim,
                    );
                }
            } else {
                simd_kernels::matmul_simd_parallel(
                    &mut s.logits,
                    &self.output_proj,
                    &s.normed,
                    vocab_size,
                    hidden_dim,
                    &self.pool,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            simd_kernels::matmul_simd_parallel(
                &mut s.logits,
                &self.output_proj,
                &s.normed,
                vocab_size,
                hidden_dim,
                &self.pool,
            );
        }

        // Move the computed logits out of scratch (O(1) pointer move, no 128KB memcpy).
        // Then re-provision scratch.logits for the next call. The engine drops the
        // previously returned Logits on reassignment, so one alloc per token is
        // unavoidable with the current trait interface. But we eliminate the 128KB
        // data copy that clone() performed (~10us saved for vocab_size=32000).
        let out = std::mem::take(&mut s.logits);
        s.logits = vec![0.0f32; vocab_size];
        Ok(Logits { data: out })
    }

    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        // Use cached dimensions (set once in init) -- avoids Option unwrap
        // and as-usize casts on every call.
        let hidden_dim = self.cached_hidden_dim;
        let vocab_size = self.cached_vocab_size;
        let tid = token_id as usize;

        if tid >= vocab_size {
            return Err(RuntimeError::Compute(format!(
                "token_id {tid} out of range (vocab_size={vocab_size})",
            )));
        }

        let start = tid * hidden_dim;
        let end = start + hidden_dim;
        let embed = &self.embedding[start..end];

        // Build the byte buffer directly (one allocation, one memcpy).
        // Avoids the previous pattern of writing into scratch embed_buf then cloning
        // (which did two memcpys: embed->buf, buf->clone).
        let byte_len = hidden_dim * 4;
        let mut data = Vec::with_capacity(byte_len);
        #[cfg(target_endian = "little")]
        {
            // SAFETY: embed is a contiguous &[f32] slice. On LE platforms, the
            // in-memory representation IS the LE byte sequence. Vec::with_capacity
            // guarantees the allocation exists. We copy byte_len bytes and set_len.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    embed.as_ptr() as *const u8,
                    data.as_mut_ptr(),
                    byte_len,
                );
                data.set_len(byte_len);
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
            num_elements: hidden_dim,
            dtype: ComputeDtype::F32,
        })
    }

    fn set_profile(&mut self, enabled: bool) {
        self.profile_enabled = enabled;
        if enabled {
            // Reset accumulated profile data when enabling
            *self.profile.get_mut() = ComputeProfile::default();
        }
    }

    fn print_profile(&self) {
        if !self.profile_enabled {
            return;
        }
        let prof = unsafe { &*self.profile.get() };
        let num_layers = self.hyperparams.map(|hp| hp.num_layers as usize).unwrap_or(0);
        prof.print_summary(num_layers);
    }
}
