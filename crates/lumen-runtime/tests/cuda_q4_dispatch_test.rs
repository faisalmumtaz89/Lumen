//! CUDA Q4_0 dispatch integration tests.
//!
//! Tests end-to-end Q4_0 weight support: loading Q4_0 blocks to GPU,
//! dispatching matvec_q4_0 kernels via the GpuWeightBuf::Q4Raw path,
//! and verifying output against a CPU dequantized F32 reference.
//!
//! The test constructs synthetic Q4_0 layer weights, dequantizes them to F32
//! on the CPU for a reference run, then runs the same weights through the CUDA
//! backend with native Q4_0 dispatch. Outputs are compared within quantization
//! tolerance (Q4_0 has ~5% quantization noise due to 4-bit precision).
//!
//! These tests require a CUDA-capable GPU. They are gated behind
//! `--features cuda` and will fail on macOS (no NVIDIA GPU).
//!
//! Run on Modal:
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_q4_dispatch_test

#![cfg(feature = "cuda")]

use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::cache::LayerView;

// ---------------------------------------------------------------------------
// Test model configuration
// ---------------------------------------------------------------------------

/// Tiny model dimensions for fast testing. All dims are multiples of 32
/// to satisfy Q4_0 block alignment (32 elements per block).
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const HIDDEN_DIM: usize = 32; // NUM_HEADS * HEAD_DIM
const INTER_DIM: usize = 64;
const VOCAB_SIZE: usize = 32;
const MAX_SEQ_LEN: usize = 64;

/// Q4_0 block layout constants (GGML standard).
const Q4_0_BLOCK_BYTES: usize = 18;
const Q4_0_GROUP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64)
// ---------------------------------------------------------------------------

struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        let frac = (u >> 40) as f32 / (1u64 << 24) as f32;
        (frac - 0.5) * 0.2 // [-0.1, 0.1)
    }

    fn gen_f32_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }

    /// Generate norm weights centered around 1.0 (small perturbation).
    fn gen_norm_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + self.next_f32() * 0.01).collect()
    }
}

// ---------------------------------------------------------------------------
// f16 conversion helpers
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        if frac != 0 { return sign | 0x7E00; }
        return sign | 0x7C00;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return sign | 0x7C00; }
    if new_exp <= 0 {
        if new_exp < -10 { return sign; }
        let full_frac = frac | 0x800000;
        let shift = (1 - new_exp) as u32;
        let f16_frac = (full_frac >> (13 + shift)) as u16;
        return sign | f16_frac;
    }
    let f16_exp = (new_exp as u16) << 10;
    let f16_frac = (frac >> 13) as u16;
    sign | f16_exp | f16_frac
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let v = (frac as f32) * 6.103515625e-05 / 1024.0;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        let f32_bits = (sign << 31) | 0x7f800000 | if frac != 0 { 0x400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }

    let f32_exp = exp + 127 - 15;
    let f32_frac = frac << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
}

// ---------------------------------------------------------------------------
// Q4_0 quantization/dequantization
// ---------------------------------------------------------------------------

/// Quantize an f32 weight matrix [out_dim, in_dim] to Q4_0 format.
///
/// `in_dim` must be a multiple of 32. Returns raw Q4_0 byte stream.
fn quantize_f32_to_q4_0(f32_data: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % Q4_0_GROUP_SIZE, 0);
    assert_eq!(f32_data.len(), out_dim * in_dim);

    let blocks_per_row = in_dim / Q4_0_GROUP_SIZE;
    let mut q4_bytes = Vec::with_capacity(out_dim * blocks_per_row * Q4_0_BLOCK_BYTES);

    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let base = row * in_dim + block * Q4_0_GROUP_SIZE;
            let block_vals = &f32_data[base..base + Q4_0_GROUP_SIZE];

            let amax = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if amax == 0.0 { 0.0 } else { amax / 8.0 };

            // Write f16 scale (little-endian).
            let scale_bits = f32_to_f16_bits(scale);
            q4_bytes.push((scale_bits & 0xFF) as u8);
            q4_bytes.push((scale_bits >> 8) as u8);

            // Quantize 32 values into 16 nibble-pair bytes.
            for i in 0..16 {
                let v0 = block_vals[2 * i];
                let v1 = block_vals[2 * i + 1];

                let q0 = if scale == 0.0 {
                    8u8
                } else {
                    ((v0 / scale + 8.0).round() as i32).clamp(0, 15) as u8
                };
                let q1 = if scale == 0.0 {
                    8u8
                } else {
                    ((v1 / scale + 8.0).round() as i32).clamp(0, 15) as u8
                };

                q4_bytes.push(q0 | (q1 << 4));
            }
        }
    }

    q4_bytes
}

/// Dequantize a Q4_0 weight matrix back to f32 (CPU reference).
fn dequant_q4_0_to_f32(q4_bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(in_dim % Q4_0_GROUP_SIZE, 0);
    let blocks_per_row = in_dim / Q4_0_GROUP_SIZE;

    let mut result = Vec::with_capacity(out_dim * in_dim);
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
            let bp = &q4_bytes[block_offset..];

            let scale_bits = bp[0] as u16 | ((bp[1] as u16) << 8);
            let scale = f16_bits_to_f32(scale_bits);

            for i in 0..16 {
                let byte_val = bp[2 + i];
                let nibble_lo = byte_val & 0x0F;
                let nibble_hi = (byte_val >> 4) & 0x0F;

                result.push(scale * (nibble_lo as f32 - 8.0));
                result.push(scale * (nibble_hi as f32 - 8.0));
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Compare two f32 slices element-wise with a tolerance.
fn assert_f32_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(), expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(), expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance {
            if mismatches < 5 {
                eprintln!("  {label}[{i}]: CUDA={a:.6}, CPU={e:.6}, diff={diff:.2e}");
            }
            mismatches += 1;
        }
        max_diff = max_diff.max(diff);
    }
    assert!(
        mismatches == 0,
        "{label}: {mismatches}/{} elements exceed tolerance {tolerance:.1e} (max_diff={max_diff:.2e})",
        actual.len()
    );
    eprintln!("  {label}: max_diff={max_diff:.2e} (within tolerance {tolerance:.1e})");
}

/// Build model hyperparams for the test model.
fn test_hyperparams() -> lumen_format::hyperparams::ModelHyperparams {
    lumen_format::hyperparams::ModelHyperparams {
        num_layers: NUM_LAYERS as u32,
        num_heads: NUM_HEADS as u32,
        num_kv_heads: NUM_KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        hidden_dim: HIDDEN_DIM as u32,
        intermediate_dim: INTER_DIM as u32,
        vocab_size: VOCAB_SIZE as u32,
        max_seq_len: MAX_SEQ_LEN as u32,
        rope_params: Some(lumen_format::hyperparams::RopeParams::default()),
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-5,
        rotary_dim: None, rope_neox: false,
    }
}

/// Build a Q4_0 LayerView from random weights. Returns the LayerView with Q4_0
/// tensor slices and the dequantized F32 weights for CPU reference.
fn build_q4_layer(
    rng: &mut TestRng,
    q_dim: usize,
    kv_dim: usize,
) -> (LayerView, Vec<f32>) {
    let mut blob = Vec::new();
    let mut offset = 0u64;
    let mut deq_all = Vec::new();

    // Build one Q4_0 tensor: generate f32, quantize to Q4_0, dequantize for reference.
    let add_q4_tensor = |rng_ref: &mut TestRng,
                              blob: &mut Vec<u8>,
                              off: &mut u64,
                              deq: &mut Vec<f32>,
                              out_d: usize,
                              in_d: usize|
     -> TensorSlice {
        let f32_vals = rng_ref.gen_f32_vec(out_d * in_d);
        let q4_bytes = quantize_f32_to_q4_0(&f32_vals, out_d, in_d);
        let deq_vals = dequant_q4_0_to_f32(&q4_bytes, out_d, in_d);
        let len = q4_bytes.len() as u64;
        let ts = TensorSlice { offset: *off, length: len, quant: QuantScheme::Q4_0 };
        blob.extend_from_slice(&q4_bytes);
        *off += len;
        deq.extend_from_slice(&deq_vals);
        ts
    };

    // Build one F32 tensor (for norms).
    let add_f32_tensor = |data: &[f32],
                               blob: &mut Vec<u8>,
                               off: &mut u64|
     -> TensorSlice {
        let bytes = f32_to_le_bytes(data);
        let len = bytes.len() as u64;
        let ts = TensorSlice { offset: *off, length: len, quant: QuantScheme::F32 };
        blob.extend_from_slice(&bytes);
        *off += len;
        ts
    };

    let wq = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, q_dim, HIDDEN_DIM);
    let wk = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, kv_dim, HIDDEN_DIM);
    let wv = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, kv_dim, HIDDEN_DIM);
    let wo = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, HIDDEN_DIM, q_dim);
    let w_gate = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, INTER_DIM, HIDDEN_DIM);
    let w_up = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, INTER_DIM, HIDDEN_DIM);
    let w_down = add_q4_tensor(rng, &mut blob, &mut offset, &mut deq_all, HIDDEN_DIM, INTER_DIM);

    let attn_norm_vals = rng.gen_norm_vec(HIDDEN_DIM);
    let ffn_norm_vals = rng.gen_norm_vec(HIDDEN_DIM);
    let attn_norm = add_f32_tensor(&attn_norm_vals, &mut blob, &mut offset);
    let ffn_norm = add_f32_tensor(&ffn_norm_vals, &mut blob, &mut offset);

    let subtensors = SubtensorOffsets {
        wq, wk, wv, wo,
        bq: None, bk: None, bv: None,
        w_gate, w_up, w_down,
        attn_norm, ffn_norm,
        router_weight: None,
        experts: None,
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: None,
    };

    let view = LayerView::from_owned(0, blob, subtensors);
    (view, deq_all)
}

/// Build a dequantized F32 LayerView from a Q4_0 LayerView.
///
/// Reads each Q4_0 subtensor from the Q4_0 view, dequantizes to F32,
/// and builds a new LayerView with F32 tensor slices. Norm tensors (already F32)
/// are copied verbatim.
fn dequant_layer_to_f32(
    q4_view: &LayerView,
    q_dim: usize,
    kv_dim: usize,
) -> Result<LayerView, RuntimeError> {
    let st = &q4_view.subtensors;
    let mut blob = Vec::new();
    let mut offset = 0u64;

    let dequant_append = |view: &LayerView,
                           blob: &mut Vec<u8>,
                           off: &mut u64,
                           slice: &TensorSlice,
                           out_d: usize,
                           in_d: usize|
     -> Result<TensorSlice, RuntimeError> {
        let raw = view.subtensor_bytes(slice)?;
        let f32_vals = dequant_q4_0_to_f32(raw, out_d, in_d);
        let f32_bytes = f32_to_le_bytes(&f32_vals);
        let len = f32_bytes.len() as u64;
        let ts = TensorSlice { offset: *off, length: len, quant: QuantScheme::F32 };
        blob.extend_from_slice(&f32_bytes);
        *off += len;
        Ok(ts)
    };

    let copy_f32 = |view: &LayerView,
                     blob: &mut Vec<u8>,
                     off: &mut u64,
                     slice: &TensorSlice|
     -> Result<TensorSlice, RuntimeError> {
        let raw = view.subtensor_bytes(slice)?;
        let len = raw.len() as u64;
        let ts = TensorSlice { offset: *off, length: len, quant: QuantScheme::F32 };
        blob.extend_from_slice(raw);
        *off += len;
        Ok(ts)
    };

    let wq = dequant_append(q4_view, &mut blob, &mut offset, &st.wq, q_dim, HIDDEN_DIM)?;
    let wk = dequant_append(q4_view, &mut blob, &mut offset, &st.wk, kv_dim, HIDDEN_DIM)?;
    let wv = dequant_append(q4_view, &mut blob, &mut offset, &st.wv, kv_dim, HIDDEN_DIM)?;
    let wo = dequant_append(q4_view, &mut blob, &mut offset, &st.wo, HIDDEN_DIM, q_dim)?;
    let w_gate = dequant_append(q4_view, &mut blob, &mut offset, &st.w_gate, INTER_DIM, HIDDEN_DIM)?;
    let w_up = dequant_append(q4_view, &mut blob, &mut offset, &st.w_up, INTER_DIM, HIDDEN_DIM)?;
    let w_down = dequant_append(q4_view, &mut blob, &mut offset, &st.w_down, HIDDEN_DIM, INTER_DIM)?;
    let attn_norm = copy_f32(q4_view, &mut blob, &mut offset, &st.attn_norm)?;
    let ffn_norm = copy_f32(q4_view, &mut blob, &mut offset, &st.ffn_norm)?;

    let subtensors = SubtensorOffsets {
        wq, wk, wv, wo,
        bq: None, bk: None, bv: None,
        w_gate, w_up, w_down,
        attn_norm, ffn_norm,
        router_weight: None,
        experts: None,
        shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
        attn_gate: None, attn_post_norm: None,
        ssm_a: None, ssm_conv1d: None, ssm_dt: None,
        ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
        attn_q_norm: None, attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: None,
    };

    Ok(LayerView::from_owned(0, blob, subtensors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_cuda_q4_0_compute_layer_matches_cpu_dequant() {
    let hp = test_hyperparams();
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    let mut rng = TestRng::new(42);

    // Generate global tensors (always F32).
    let embedding = rng.gen_f32_vec(VOCAB_SIZE * HIDDEN_DIM);
    let final_norm = rng.gen_norm_vec(HIDDEN_DIM);
    let output_proj = rng.gen_f32_vec(VOCAB_SIZE * HIDDEN_DIM);

    // Generate Q4_0 layer weights.
    let mut q4_layers = Vec::new();
    for _ in 0..NUM_LAYERS {
        let (view, _deq) = build_q4_layer(&mut rng, q_dim, kv_dim);
        q4_layers.push(view);
    }

    // CPU backend uses dequantized F32 weights.
    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(embedding.clone(), final_norm.clone(), output_proj.clone());
    cpu.init(&hp).unwrap();

    // CUDA backend uses native Q4_0 weights.
    let mut cuda = match CudaBackend::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping CUDA Q4_0 test (no GPU?): {e}");
            return;
        }
    };
    cuda.set_global_tensors(embedding, final_norm, output_proj);
    cuda.init(&hp).unwrap();

    let kv_config = KvCacheConfig {
        max_seq_len: MAX_SEQ_LEN,
        num_layers: NUM_LAYERS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        precision: KvPrecision::F32,
    };
    let mut cpu_kv = KvCache::new(kv_config.clone()).unwrap();
    let mut cuda_kv = KvCache::new(kv_config).unwrap();

    // Embed token 0 (F32 embedding, same for both).
    let mut cpu_x = cpu.embed_token(0).unwrap();
    let mut cuda_x = cuda.embed_token(0).unwrap();

    eprintln!("=== Q4_0 compute_layer comparison (token 0) ===");
    eprintln!("  model: {NUM_LAYERS} layers, hidden_dim={HIDDEN_DIM}, Q4_0 weights");

    for layer_idx in 0..NUM_LAYERS {
        // CUDA backend: use Q4_0 layer view directly.
        let q4_layer_view = &q4_layers[layer_idx];

        // CPU backend: dequantize Q4_0 to F32.
        let f32_layer_view = dequant_layer_to_f32(q4_layer_view, q_dim, kv_dim).unwrap();

        let seq_pos = cpu_kv.seq_len();

        // CPU compute_layer with dequantized F32 weights.
        {
            let mut kv_view = cpu_kv.view_mut(layer_idx).unwrap();
            cpu.compute_layer(layer_idx, &mut cpu_x, &f32_layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cpu_kv.commit_view(kv_view).unwrap();
        }

        // CUDA compute_layer with native Q4_0 weights.
        {
            let mut kv_view = cuda_kv.view_mut(layer_idx).unwrap();
            cuda.compute_layer(layer_idx, &mut cuda_x, q4_layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cuda_kv.commit_view(kv_view).unwrap();
        }

        // Compare layer outputs. Q4_0 quantization introduces more noise than Q8_0
        // (4 bits vs 8 bits), so we use a wider tolerance. The dequantized F32
        // reference has the SAME quantization noise as the GPU path (both start
        // from the same Q4_0 blocks), so the only difference is float reduction order.
        let cpu_vals = cpu_x.as_f32_slice();
        let cuda_vals = cuda_x.as_f32_slice();
        assert_f32_close(
            &format!("q4_layer_{layer_idx}"),
            cuda_vals,
            cpu_vals,
            1e-2, // Wider tolerance for Q4_0 reduction order differences
        );
    }

    // Advance KV caches.
    cpu_kv.advance_seq_len().unwrap();
    cuda_kv.advance_seq_len().unwrap();

    // Run compute_final (F32 output_proj for both).
    let cpu_logits = cpu.compute_final(&cpu_x).unwrap();
    let cuda_logits = cuda.compute_final(&cuda_x).unwrap();

    eprintln!("=== Q4_0 compute_final ===");
    assert_f32_close("q4_final_logits", &cuda_logits.data, &cpu_logits.data, 1e-2);

    let cpu_argmax = cpu_logits.argmax();
    let cuda_argmax = cuda_logits.argmax();
    eprintln!("  argmax: CPU={cpu_argmax}, CUDA={cuda_argmax}");
}

#[test]
fn test_cuda_q4_0_compute_final_with_q4_output_proj() {
    // Verify that set_output_proj_raw with Q4_0 data dispatches matvec_q4_0
    // correctly in compute_final.
    let hp = test_hyperparams();
    let mut rng = TestRng::new(99);

    let embedding = rng.gen_f32_vec(VOCAB_SIZE * HIDDEN_DIM);
    let final_norm = rng.gen_norm_vec(HIDDEN_DIM);
    let output_proj_f32 = rng.gen_f32_vec(VOCAB_SIZE * HIDDEN_DIM);

    // Quantize output_proj to Q4_0.
    let q4_output_proj = quantize_f32_to_q4_0(&output_proj_f32, VOCAB_SIZE, HIDDEN_DIM);
    let deq_output_proj = dequant_q4_0_to_f32(&q4_output_proj, VOCAB_SIZE, HIDDEN_DIM);

    // CPU backend with dequantized output_proj.
    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(embedding.clone(), final_norm.clone(), deq_output_proj);
    cpu.init(&hp).unwrap();

    // CUDA backend with Q4_0 output_proj.
    let mut cuda = match CudaBackend::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping CUDA Q4_0 test (no GPU?): {e}");
            return;
        }
    };
    cuda.set_global_tensors(
        embedding,
        final_norm,
        Vec::new(), // F32 output_proj not used when raw Q4_0 is set
    );
    cuda.set_output_proj_raw(q4_output_proj, QuantScheme::Q4_0);
    cuda.init(&hp).unwrap();

    // Embed token and compute_final.
    let x = cpu.embed_token(0).unwrap();
    let cpu_logits = cpu.compute_final(&x).unwrap();
    let cuda_logits = cuda.compute_final(&x).unwrap();

    eprintln!("=== Q4_0 output_proj compute_final ===");
    assert_f32_close(
        "q4_output_proj_logits",
        &cuda_logits.data,
        &cpu_logits.data,
        1e-2,
    );
}
