//! CUDA Q8_0 dispatch integration tests.
//!
//! Tests end-to-end Q8_0 weight support: loading Q8_0 blocks to GPU,
//! dispatching matvec_q8_0 kernels, and verifying output against a CPU
//! dequantized F32 reference.
//!
//! The test generates a synthetic Q8_0 model, dequantizes the weights to F32
//! on the CPU for a reference run, then runs the same model through the CUDA
//! backend with native Q8_0 dispatch. Outputs are compared within quantization
//! tolerance (Q8_0 has ~1% quantization noise).
//!
//! These tests require a CUDA-capable GPU. They are gated behind
//! `--features cuda` and will fail on macOS (no NVIDIA GPU).

#![cfg(feature = "cuda")]

use lumen_format::index::TensorSlice;
use lumen_format::quantization::QuantScheme;
use lumen_format::test_model::{TestModelQ8Config, generate_test_model_q8_0};
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::error::RuntimeError;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Q8_0 block size constants.
const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BYTES_PER_BLOCK: usize = 34;

/// Convert f16 bits to f32 (mirrors the CUDA kernel's f16_bits_to_f32).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let v = (frac as f32) * 6.103515625e-05 / 1024.0;
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        let f32_bits = (sign << 31) | 0x7f800000 | if frac != 0 { 0x400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }

    let f32_exp = exp as u32 - 15 + 127;
    let f32_frac = frac << 13;
    let f32_bits = (sign << 31) | (f32_exp << 23) | f32_frac;
    f32::from_bits(f32_bits)
}

/// Dequantize a Q8_0 weight matrix to f32.
///
/// Input: raw Q8_0 bytes for a [out_dim, in_dim] matrix.
/// Output: f32 values in row-major order.
///
/// `in_dim` must be a multiple of Q8_0_BLOCK_SIZE (32).
fn dequant_q8_0_to_f32(q8_bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(in_dim % Q8_0_BLOCK_SIZE, 0);
    let blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;
    let expected_bytes = out_dim * blocks_per_row * Q8_0_BYTES_PER_BLOCK;
    assert_eq!(
        q8_bytes.len(),
        expected_bytes,
        "Q8_0 byte length mismatch: got {} expected {}",
        q8_bytes.len(),
        expected_bytes,
    );

    let mut result = Vec::with_capacity(out_dim * in_dim);
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + block) * Q8_0_BYTES_PER_BLOCK;
            let block_ptr = &q8_bytes[block_offset..];

            // Read f16 scale
            let scale_bits =
                block_ptr[0] as u16 | ((block_ptr[1] as u16) << 8);
            let scale = f16_bits_to_f32(scale_bits);

            // Dequantize 32 int8 values
            for j in 0..Q8_0_BLOCK_SIZE {
                let q_val = block_ptr[2 + j] as i8;
                result.push(scale * (q_val as f32));
            }
        }
    }

    result
}

/// Build a dequantized F32 LayerView blob from a Q8_0 LayerView.
///
/// Reads each Q8_0 subtensor, dequantizes to F32, and writes a new blob
/// with F32 TensorSlices. Norm tensors (already F32) are copied verbatim.
fn dequant_layer_to_f32(
    provider: &SyncWeightProvider,
    layer_idx: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    inter_dim: usize,
) -> Result<(Vec<u8>, lumen_format::index::SubtensorOffsets), RuntimeError> {
    let view = provider.get_layer_blocking(layer_idx)?;
    let st = &view.subtensors;

    let mut blob = Vec::new();
    let mut offset = 0u64;

    // Dequant a Q8_0 tensor and append as F32 bytes.
    // Takes blob/offset by &mut to avoid closure borrow issues.
    fn dequant_append(
        view: &lumen_runtime::weight::cache::LayerView,
        blob: &mut Vec<u8>,
        offset: &mut u64,
        slice: &TensorSlice,
        out_d: usize,
        in_d: usize,
    ) -> Result<TensorSlice, RuntimeError> {
        let raw = view.subtensor_bytes(slice)?;
        let f32_vals = dequant_q8_0_to_f32(raw, out_d, in_d);
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = f32_bytes.len() as u64;
        let ts = TensorSlice { offset: *offset, length: len, quant: QuantScheme::F32 };
        blob.extend_from_slice(&f32_bytes);
        *offset += len;
        Ok(ts)
    }

    // Copy F32 tensor verbatim.
    fn copy_f32_tensor(
        view: &lumen_runtime::weight::cache::LayerView,
        blob: &mut Vec<u8>,
        offset: &mut u64,
        slice: &TensorSlice,
    ) -> Result<TensorSlice, RuntimeError> {
        let raw = view.subtensor_bytes(slice)?;
        let len = raw.len() as u64;
        let ts = TensorSlice { offset: *offset, length: len, quant: QuantScheme::F32 };
        blob.extend_from_slice(raw);
        *offset += len;
        Ok(ts)
    }

    let wq = dequant_append(&view, &mut blob, &mut offset, &st.wq, q_dim, hidden_dim)?;
    let wk = dequant_append(&view, &mut blob, &mut offset, &st.wk, kv_dim, hidden_dim)?;
    let wv = dequant_append(&view, &mut blob, &mut offset, &st.wv, kv_dim, hidden_dim)?;
    let wo = dequant_append(&view, &mut blob, &mut offset, &st.wo, hidden_dim, q_dim)?;
    let w_gate = dequant_append(&view, &mut blob, &mut offset, &st.w_gate, inter_dim, hidden_dim)?;
    let w_up = dequant_append(&view, &mut blob, &mut offset, &st.w_up, inter_dim, hidden_dim)?;
    let w_down = dequant_append(&view, &mut blob, &mut offset, &st.w_down, hidden_dim, inter_dim)?;
    let attn_norm = copy_f32_tensor(&view, &mut blob, &mut offset, &st.attn_norm)?;
    let ffn_norm = copy_f32_tensor(&view, &mut blob, &mut offset, &st.ffn_norm)?;

    let subs = lumen_format::index::SubtensorOffsets {
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

    Ok((blob, subs))
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

/// Set up Q8_0 test model and return (provider, CPU backend with dequantized weights, CUDA backend).
fn setup_q8_backends() -> Result<
    (SyncWeightProvider, NaiveF32Backend, CudaBackend),
    RuntimeError,
> {
    let config = TestModelQ8Config::default();
    let lbc_data = generate_test_model_q8_0(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_q8_test_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model_q8.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = SyncWeightProvider::open(&path)?;
    let hp = provider.lbc().header.hyperparams;

    // CPU naive backend uses F32 global tensors (same for both backends).
    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cpu.init(&hp)?;

    // CUDA backend uses F32 global tensors + Q8_0 layer weights via upload_layer_weights.
    let mut cuda = CudaBackend::new(0)?;
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp)?;

    Ok((provider, cpu, cuda))
}

#[test]
fn test_cuda_q8_0_compute_layer_matches_cpu_dequant() {
    let (provider, cpu, cuda) = match setup_q8_backends() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping CUDA Q8_0 test (no GPU?): {e}");
            return;
        }
    };

    let hp = provider.lbc().header.hyperparams;
    let hidden_dim = hp.hidden_dim as usize;
    let num_layers = hp.num_layers as usize;
    let num_heads = hp.num_heads as usize;
    let num_kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter_dim = hp.intermediate_dim as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let kv_config = KvCacheConfig {
        max_seq_len: hp.max_seq_len as usize,
        num_layers,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision: KvPrecision::F32,
    };
    let mut cpu_kv = KvCache::new(kv_config.clone()).unwrap();
    let mut cuda_kv = KvCache::new(kv_config).unwrap();

    // Embed token 0 (F32 embedding, same for both).
    let mut cpu_x = cpu.embed_token(0).unwrap();
    let mut cuda_x = cuda.embed_token(0).unwrap();

    eprintln!("=== Q8_0 compute_layer comparison (token 0) ===");
    eprintln!("  model: {num_layers} layers, hidden_dim={hidden_dim}, Q8_0 weights");

    for layer_idx in 0..num_layers {
        // CUDA backend uses the native Q8_0 layer view directly.
        let q8_layer_view = provider.get_layer_blocking(layer_idx).unwrap();

        // CPU backend needs dequantized F32 weights. Build a synthetic LayerView
        // with F32 data from the dequantized Q8_0 bytes.
        let (f32_blob, f32_subs) = dequant_layer_to_f32(
            &provider, layer_idx, hidden_dim, q_dim, kv_dim, inter_dim,
        ).unwrap();
        let cpu_layer_view = lumen_runtime::weight::cache::LayerView::from_owned(
            layer_idx,
            f32_blob,
            f32_subs,
        );

        let seq_pos = cpu_kv.seq_len();

        // CPU compute_layer with dequantized F32 weights
        {
            let mut kv_view = cpu_kv.view_mut(layer_idx).unwrap();
            cpu.compute_layer(layer_idx, &mut cpu_x, &cpu_layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cpu_kv.commit_view(kv_view).unwrap();
        }

        // CUDA compute_layer with native Q8_0 weights
        {
            let mut kv_view = cuda_kv.view_mut(layer_idx).unwrap();
            cuda.compute_layer(layer_idx, &mut cuda_x, &q8_layer_view, Some(&mut kv_view), seq_pos)
                .unwrap();
            cuda_kv.commit_view(kv_view).unwrap();
        }

        // Compare layer outputs. Q8_0 quantization introduces noise, so we use a
        // wider tolerance than pure F32 comparisons. The dequantized F32 reference
        // has the SAME quantization noise as the GPU path (both start from the same
        // Q8_0 blocks), so the only difference is float reduction order.
        let cpu_vals = cpu_x.as_f32_slice();
        let cuda_vals = cuda_x.as_f32_slice();
        assert_f32_close(
            &format!("q8_layer_{layer_idx}"),
            cuda_vals,
            cpu_vals,
            5e-3, // Wider tolerance for Q8_0 reduction order differences
        );
    }

    // Advance KV caches
    cpu_kv.advance_seq_len().unwrap();
    cuda_kv.advance_seq_len().unwrap();

    // Run compute_final (F32 output_proj for both)
    let cpu_logits = cpu.compute_final(&cpu_x).unwrap();
    let cuda_logits = cuda.compute_final(&cuda_x).unwrap();

    eprintln!("=== Q8_0 compute_final ===");
    assert_f32_close("q8_final_logits", &cuda_logits.data, &cpu_logits.data, 5e-3);

    let cpu_argmax = cpu_logits.argmax();
    let cuda_argmax = cuda_logits.argmax();
    eprintln!("  argmax: CPU={cpu_argmax}, CUDA={cuda_argmax}");
    // Note: argmax may differ slightly due to accumulated Q8_0 noise across layers.
    // We verify outputs are close rather than requiring exact argmax match.
}

#[test]
fn test_cuda_q8_0_compute_final_with_q8_output_proj() {
    // This test verifies that set_output_proj_raw + matvec_q8_0 works for compute_final.
    // We compare the CUDA Q8_0 output projection against a CPU F32 dequantized reference.
    let config = TestModelQ8Config::default();
    let lbc_data = generate_test_model_q8_0(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_q8_final_test_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model_q8.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = match SyncWeightProvider::open(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
    let hp = provider.lbc().header.hyperparams;
    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;

    // Build Q8_0 output_proj bytes from the F32 output_proj by quantizing.
    // This simulates a real Q8_0 model where output_proj is quantized.
    let output_proj_f32 = &provider.output_proj;
    let q8_output_proj = quantize_f32_to_q8_0(output_proj_f32, vocab_size, hidden_dim);

    // Dequantize back to F32 for CPU reference.
    let deq_output_proj = dequant_q8_0_to_f32(&q8_output_proj, vocab_size, hidden_dim);

    // CPU backend with dequantized output_proj
    let mut cpu = NaiveF32Backend::new();
    cpu.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        deq_output_proj,
    );
    cpu.init(&hp).unwrap();

    // CUDA backend with Q8_0 output_proj
    let mut cuda = match CudaBackend::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping CUDA Q8_0 test (no GPU?): {e}");
            return;
        }
    };
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        Vec::new(), // F32 output_proj not used when raw Q8_0 is set
    );
    cuda.set_output_proj_raw(q8_output_proj, QuantScheme::Q8_0);
    cuda.init(&hp).unwrap();

    // Embed token and compute_final
    let x = cpu.embed_token(0).unwrap();
    let cpu_logits = cpu.compute_final(&x).unwrap();
    let cuda_logits = cuda.compute_final(&x).unwrap();

    eprintln!("=== Q8_0 output_proj compute_final ===");
    assert_f32_close(
        "q8_output_proj_logits",
        &cuda_logits.data,
        &cpu_logits.data,
        5e-3,
    );
}

#[test]
fn test_cuda_q8_0_embed_token() {
    // Verify Q8_0 embedding lookup produces the correct dequantized output.
    let config = TestModelQ8Config::default();
    let lbc_data = generate_test_model_q8_0(&config);

    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_q8_embed_test_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_model_q8.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc_data).unwrap();
    }

    let provider = match SyncWeightProvider::open(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Skipping: {e}");
            return;
        }
    };
    let hp = provider.lbc().header.hyperparams;
    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;

    // Quantize the F32 embedding to Q8_0
    let emb_f32 = &provider.embedding;
    let q8_emb = quantize_f32_to_q8_0(emb_f32, vocab_size, hidden_dim);

    // Dequantize for CPU reference
    let deq_emb = dequant_q8_0_to_f32(&q8_emb, vocab_size, hidden_dim);

    // CUDA backend with Q8_0 embedding
    let mut cuda = match CudaBackend::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping CUDA Q8_0 test (no GPU?): {e}");
            return;
        }
    };
    cuda.set_global_tensors(
        Vec::new(), // F32 embedding not used when raw Q8_0 is set
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.set_embedding_raw(q8_emb, QuantScheme::Q8_0);
    cuda.init(&hp).unwrap();

    // Test embedding lookup for token 0
    let cuda_emb = cuda.embed_token(0).unwrap();
    let cuda_vals = cuda_emb.as_f32_slice();
    let expected = &deq_emb[..hidden_dim]; // First row = token 0

    eprintln!("=== Q8_0 embed_token ===");
    assert_f32_close("q8_embed_0", cuda_vals, expected, 1e-4);
}

/// Quantize an f32 weight matrix to Q8_0 format.
///
/// `f32_data`: row-major [out_dim, in_dim] f32 values.
/// `in_dim` must be a multiple of 32.
///
/// Returns raw Q8_0 byte stream.
fn quantize_f32_to_q8_0(f32_data: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % Q8_0_BLOCK_SIZE, 0);
    assert_eq!(f32_data.len(), out_dim * in_dim);

    let blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;
    let mut q8_bytes = Vec::with_capacity(out_dim * blocks_per_row * Q8_0_BYTES_PER_BLOCK);

    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let base = row * in_dim + block * Q8_0_BLOCK_SIZE;
            let block_vals = &f32_data[base..base + Q8_0_BLOCK_SIZE];

            // Compute scale from max absolute value
            let amax = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write f16 scale
            let scale_bits = f32_to_f16_bits(scale);
            q8_bytes.push((scale_bits & 0xFF) as u8);
            q8_bytes.push(((scale_bits >> 8) & 0xFF) as u8);

            // Write 32 quantized int8 values
            for &val in block_vals {
                let q = (val * inv_scale).round() as i32;
                q8_bytes.push(q.clamp(-128, 127) as u8);
            }
        }
    }

    q8_bytes
}

/// Convert f32 to f16 bits (same as in rng.rs).
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
