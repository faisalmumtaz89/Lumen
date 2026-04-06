//! Integration tests for GDN layer routing in the CUDA backend.
//!
//! Tests that `compute_layer` correctly detects GDN layers (via `ssm_conv1d`
//! presence) and routes them through the GDN path rather than the standard
//! attention path. Verifies end-to-end correctness with a synthetic hybrid
//! model (mixed GDN + attention layers).
//!
//! Requires a CUDA-capable GPU (run on Modal).
//!
//!   cargo test --release -p lumen-runtime --features cuda --test cuda_gdn_integration_test

#![cfg(feature = "cuda")]

use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
use lumen_runtime::weight::cache::LayerView;

/// Build a synthetic hyperparams struct for a small test model.
///
/// Uses minimal dimensions to keep tests fast:
///   hidden_dim=64, num_heads=4, num_kv_heads=2, head_dim=16,
///   inter_dim=128, vocab_size=256, max_seq_len=64, 4 layers.
fn test_hyperparams() -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        max_seq_len: 64,
        rope_params: None,
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-6,
        rotary_dim: None, rope_neox: false,
    }
}

/// GDN test dimensions derived from hyperparams (num_kv_heads=2).
///
/// GdnParams::from_hyperparams produces:
///   num_heads = 2 * num_kv_heads = 4
///   head_dim = 128 (SSM state_size, fixed for Qwen3.5)
///   qk_dim = num_kv_heads * head_dim = 256
///   value_dim = num_heads * head_dim = 512
///   qkv_dim = qk_dim + qk_dim + value_dim = 1024
///   conv_kernel_size = 4
///
/// These are the correct GDN dimensions for the test hyperparams.
/// Using realistic dimensions ensures the CPU reference kernels exercise
/// the same code paths as production models.
#[allow(dead_code)]
struct TestGdnDims {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    qk_dim: usize,
    value_dim: usize,
    qkv_dim: usize,
    conv_kernel_size: usize,
}

fn test_gdn_dims() -> TestGdnDims {
    let num_kv_heads = 2usize;
    let head_dim = 128usize;
    let num_heads = num_kv_heads * 2;
    let qk_dim = num_kv_heads * head_dim;
    let value_dim = num_heads * head_dim;
    let qkv_dim = qk_dim + qk_dim + value_dim;
    TestGdnDims {
        num_heads,
        num_kv_heads,
        head_dim,
        qk_dim,
        value_dim,
        qkv_dim,
        conv_kernel_size: 4,
    }
}

/// Write f32 values to a byte buffer at a given offset.
fn write_f32_at(buf: &mut Vec<u8>, offset: usize, values: &[f32]) {
    let needed = offset + values.len() * 4;
    if buf.len() < needed {
        buf.resize(needed, 0);
    }
    for (i, &v) in values.iter().enumerate() {
        let bytes = v.to_le_bytes();
        let pos = offset + i * 4;
        buf[pos..pos + 4].copy_from_slice(&bytes);
    }
}

/// Build a synthetic LayerView for a standard attention layer.
///
/// Allocates a byte blob with F32 weights for all required subtensors
/// (attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down).
/// All SSM fields are None, marking this as a standard attention layer.
fn build_attention_layer(layer_idx: usize, hp: &ModelHyperparams) -> LayerView {
    let hidden_dim = hp.hidden_dim as usize;
    let num_heads = hp.num_heads as usize;
    let num_kv_heads = hp.num_kv_heads as usize;
    let head_dim = hp.head_dim as usize;
    let inter_dim = hp.intermediate_dim as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let mut data = Vec::new();
    let mut offset: u64 = 0;

    // Helper to append a tensor and return its TensorSlice.
    let mut append_tensor = |data: &mut Vec<u8>, num_elements: usize| -> TensorSlice {
        let start = offset;
        let len_bytes = (num_elements * 4) as u64;
        // Fill with small non-zero values for numerical stability.
        let values: Vec<f32> = (0..num_elements)
            .map(|i| ((i as f32) * 0.001 + 0.01).sin() * 0.1)
            .collect();
        write_f32_at(data, start as usize, &values);
        offset += len_bytes;
        TensorSlice {
            offset: start,
            length: len_bytes,
            quant: QuantScheme::F32,
        }
    };

    let attn_norm = append_tensor(&mut data, hidden_dim);
    let wq = append_tensor(&mut data, q_dim * hidden_dim);
    let wk = append_tensor(&mut data, kv_dim * hidden_dim);
    let wv = append_tensor(&mut data, kv_dim * hidden_dim);
    let wo = append_tensor(&mut data, hidden_dim * q_dim);
    let ffn_norm = append_tensor(&mut data, hidden_dim);
    let w_gate = append_tensor(&mut data, inter_dim * hidden_dim);
    let w_up = append_tensor(&mut data, inter_dim * hidden_dim);
    let w_down = append_tensor(&mut data, hidden_dim * inter_dim);

    let subtensors = SubtensorOffsets {
        wq,
        wk,
        wv,
        wo,
        w_gate,
        w_up,
        w_down,
        attn_norm,
        ffn_norm,
        bq: None,
        bk: None,
        bv: None,
        router_weight: None,
        experts: None,
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate: None,
        attn_post_norm: None,
        ssm_a: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: Some(0),
    };

    LayerView::from_owned(layer_idx, data, subtensors)
}

/// Build a synthetic LayerView for a GDN layer.
///
/// Allocates a byte blob with F32 weights for all standard subtensors
/// plus the GDN-specific SSM fields (ssm_conv1d, ssm_dt, ssm_a, ssm_alpha,
/// ssm_beta, ssm_norm, ssm_out, attn_gate).
fn build_gdn_layer(layer_idx: usize, hp: &ModelHyperparams) -> LayerView {
    let hidden_dim = hp.hidden_dim as usize;
    let inter_dim = hp.intermediate_dim as usize;
    let gdn = test_gdn_dims();

    let mut data = Vec::new();
    let mut offset: u64 = 0;

    let mut append_tensor = |data: &mut Vec<u8>, num_elements: usize| -> TensorSlice {
        let start = offset;
        let len_bytes = (num_elements * 4) as u64;
        let values: Vec<f32> = (0..num_elements)
            .map(|i| ((i as f32) * 0.001 + 0.01).sin() * 0.1)
            .collect();
        write_f32_at(data, start as usize, &values);
        offset += len_bytes;
        TensorSlice {
            offset: start,
            length: len_bytes,
            quant: QuantScheme::F32,
        }
    };

    // Standard tensors (same as attention layer).
    let attn_norm = append_tensor(&mut data, hidden_dim);
    // wq for GDN is the fused QKV projection: [qkv_dim, hidden_dim]
    let wq = append_tensor(&mut data, gdn.qkv_dim * hidden_dim);
    // wk/wv not used for GDN but must be present in SubtensorOffsets
    let wk = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };
    let wv = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };
    // wo not used for GDN (ssm_out replaces it) but must be present
    let wo = TensorSlice { offset: 0, length: 0, quant: QuantScheme::F32 };
    let ffn_norm = append_tensor(&mut data, hidden_dim);
    let w_gate = append_tensor(&mut data, inter_dim * hidden_dim);
    let w_up = append_tensor(&mut data, inter_dim * hidden_dim);
    let w_down = append_tensor(&mut data, hidden_dim * inter_dim);

    // GDN-specific tensors.
    let ssm_conv1d = Some(append_tensor(&mut data, gdn.conv_kernel_size * gdn.qkv_dim));
    let ssm_dt = Some(append_tensor(&mut data, gdn.num_heads));
    // ssm_a: use append_tensor then overwrite with negative decay values.
    let ssm_a_slice = append_tensor(&mut data, gdn.num_heads);
    let ssm_a_values: Vec<f32> = (0..gdn.num_heads).map(|_| -0.5f32).collect();
    write_f32_at(&mut data, ssm_a_slice.offset as usize, &ssm_a_values);
    let ssm_a = Some(ssm_a_slice);

    let ssm_alpha = Some(append_tensor(&mut data, gdn.num_heads * hidden_dim));
    let ssm_beta = Some(append_tensor(&mut data, gdn.num_heads * hidden_dim));
    let ssm_norm = Some(append_tensor(&mut data, gdn.head_dim));
    let ssm_out = Some(append_tensor(&mut data, hidden_dim * gdn.value_dim));
    let attn_gate = Some(append_tensor(&mut data, gdn.value_dim * hidden_dim));

    let subtensors = SubtensorOffsets {
        wq,
        wk,
        wv,
        wo,
        w_gate,
        w_up,
        w_down,
        attn_norm,
        ffn_norm,
        bq: None,
        bk: None,
        bv: None,
        router_weight: None,
        experts: None,
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate,
        attn_post_norm: None,
        ssm_a,
        ssm_conv1d,
        ssm_dt,
        ssm_beta,
        ssm_alpha,
        ssm_norm,
        ssm_out,
        attn_q_norm: None,
        attn_k_norm: None,
        ffn_gate_inp_shexp: None,
        layer_type: Some(1),
    };

    LayerView::from_owned(layer_idx, data, subtensors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that GDN layer detection works correctly:
/// layers with ssm_conv1d.is_some() are GDN, others are standard attention.
#[test]
fn test_gdn_layer_detection() {
    let hp = test_hyperparams();
    let attn_layer = build_attention_layer(0, &hp);
    let gdn_layer = build_gdn_layer(1, &hp);

    assert!(
        attn_layer.subtensors.ssm_conv1d.is_none(),
        "Attention layer should not have ssm_conv1d"
    );
    assert!(
        gdn_layer.subtensors.ssm_conv1d.is_some(),
        "GDN layer should have ssm_conv1d"
    );
}

/// Verify that CudaBackend reports gdn: true in capabilities.
#[test]
fn test_cuda_backend_caps_gdn() {
    let backend = lumen_runtime::CudaBackend::new(0)
        .expect("CudaBackend::new(0) should succeed (CUDA not required for check)");
    let caps = backend.caps();
    assert!(caps.gdn, "CUDA backend should report gdn=true");
}

/// Verify that compute_layer routes GDN layers through the GDN path.
///
/// This test creates a CudaBackend, initializes it, and calls compute_layer
/// with a synthetic GDN layer. The test verifies:
/// 1. compute_layer does not return an error (GDN path was found)
/// 2. The output is finite (no NaN/Inf from the CPU reference kernels)
/// 3. The output has the correct shape (hidden_dim elements)
#[test]
fn test_compute_layer_gdn_routing() {
    let hp = test_hyperparams();
    let mut backend = lumen_runtime::CudaBackend::new(0)
        .expect("CudaBackend::new(0) should succeed");

    // Set up minimal global tensors (required for init).
    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;
    backend.set_global_tensors(
        vec![0.01f32; vocab_size * hidden_dim], // embedding
        vec![1.0f32; hidden_dim],                // final_norm
        vec![0.01f32; vocab_size * hidden_dim], // output_proj
    );

    backend.init(&hp).expect("init should succeed");

    // Build a GDN layer and compute it.
    let gdn_layer = build_gdn_layer(0, &hp);
    let mut x = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    // Fill x with non-zero values.
    let x_values: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
    x.write_f32_from(&x_values);

    // compute_layer with kv=None should work for GDN layers (they don't use KV cache).
    let result = backend.compute_layer(0, &mut x, &gdn_layer, None, 0);
    assert!(
        result.is_ok(),
        "compute_layer should succeed for GDN layer: {:?}",
        result.err()
    );

    // Verify output is finite and has correct shape.
    let output_f32 = x.as_f32_slice();
    assert_eq!(output_f32.len(), hidden_dim);
    for (i, &val) in output_f32.iter().enumerate() {
        assert!(
            val.is_finite(),
            "GDN output[{i}] should be finite, got {val}"
        );
    }
}

/// Verify that reset_recurrent_state clears GDN state.
///
/// After computing a GDN layer (which populates h_states and conv_states),
/// reset_recurrent_state() should zero everything so the next sequence
/// starts fresh.
#[test]
fn test_reset_recurrent_state() {
    let hp = test_hyperparams();
    let mut backend = lumen_runtime::CudaBackend::new(0)
        .expect("CudaBackend::new(0) should succeed");

    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;
    backend.set_global_tensors(
        vec![0.01f32; vocab_size * hidden_dim],
        vec![1.0f32; hidden_dim],
        vec![0.01f32; vocab_size * hidden_dim],
    );
    backend.init(&hp).expect("init should succeed");

    // Compute a GDN layer to populate state.
    let gdn_layer = build_gdn_layer(0, &hp);
    let mut x = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    let x_values: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
    x.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x, &gdn_layer, None, 0)
        .expect("first GDN compute should succeed");

    // Compute the same layer again (state should be non-zero from first call).
    let mut x2 = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x2.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x2, &gdn_layer, None, 1)
        .expect("second GDN compute should succeed");

    // After reset, computing again should give the same result as the first call
    // (since state is zeroed, the initial conditions match).
    backend.reset_recurrent_state();

    let mut x3 = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x3.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x3, &gdn_layer, None, 0)
        .expect("post-reset GDN compute should succeed");

    // x3 (post-reset) should match x (first call) since both start from zero state.
    let out1 = x.as_f32_slice();
    let out3 = x3.as_f32_slice();
    for i in 0..hidden_dim {
        assert!(
            (out1[i] - out3[i]).abs() < 1e-5,
            "post-reset output[{i}] = {} should match first call {} (tol=1e-5)",
            out3[i],
            out1[i],
        );
    }

    // x2 (second call, with accumulated state) should differ from x (first call).
    let out2 = x2.as_f32_slice();
    let any_differ = (0..hidden_dim).any(|i| (out1[i] - out2[i]).abs() > 1e-6);
    assert!(
        any_differ,
        "Second call (with state) should differ from first call (fresh state)"
    );
}

/// Test hybrid model: alternating GDN and attention layers.
///
/// Verifies that compute_layer correctly routes each layer type and that
/// the GDN layer map builds up correctly across multiple layer indices.
#[test]
fn test_hybrid_model_routing() {
    let hp = test_hyperparams();
    let mut backend = lumen_runtime::CudaBackend::new(0)
        .expect("CudaBackend::new(0) should succeed");

    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;
    backend.set_global_tensors(
        vec![0.01f32; vocab_size * hidden_dim],
        vec![1.0f32; hidden_dim],
        vec![0.01f32; vocab_size * hidden_dim],
    );
    backend.init(&hp).expect("init should succeed");

    // 4-layer hybrid model: GDN, Attn, GDN, Attn
    let layers: Vec<LayerView> = vec![
        build_gdn_layer(0, &hp),       // layer 0: GDN
        build_attention_layer(1, &hp),  // layer 1: Attention
        build_gdn_layer(2, &hp),        // layer 2: GDN
        build_attention_layer(3, &hp),  // layer 3: Attention
    ];

    let x_values: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let mut x = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x.write_f32_from(&x_values);

    // Layer 0: GDN (no KV cache needed)
    let result = backend.compute_layer(0, &mut x, &layers[0], None, 0);
    assert!(result.is_ok(), "layer 0 (GDN) failed: {:?}", result.err());

    // Layer 1: Attention (needs KV cache)
    let mut kv = lumen_runtime::KvCache::new(lumen_runtime::KvCacheConfig {
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        max_seq_len: hp.max_seq_len as usize,
        head_dim: hp.head_dim as usize,
        precision: lumen_runtime::KvPrecision::F32,
    })
    .expect("KV cache allocation");
    let mut kv_view = kv.view_mut(1).expect("kv view");
    let result = backend.compute_layer(1, &mut x, &layers[1], Some(&mut kv_view), 0);
    assert!(result.is_ok(), "layer 1 (Attn) failed: {:?}", result.err());

    // Layer 2: GDN
    let result = backend.compute_layer(2, &mut x, &layers[2], None, 0);
    assert!(result.is_ok(), "layer 2 (GDN) failed: {:?}", result.err());

    // Verify final output is finite.
    for (i, &val) in x.as_f32_slice().iter().enumerate() {
        assert!(
            val.is_finite(),
            "hybrid output[{i}] should be finite, got {val}"
        );
    }
}

/// Verify that GDN layer state persists across token positions.
///
/// h_states should accumulate across multiple compute_layer calls
/// (same layer, sequential tokens), producing different outputs each time.
#[test]
fn test_gdn_state_persistence() {
    let hp = test_hyperparams();
    let mut backend = lumen_runtime::CudaBackend::new(0)
        .expect("CudaBackend::new(0) should succeed");

    let hidden_dim = hp.hidden_dim as usize;
    let vocab_size = hp.vocab_size as usize;
    backend.set_global_tensors(
        vec![0.01f32; vocab_size * hidden_dim],
        vec![1.0f32; hidden_dim],
        vec![0.01f32; vocab_size * hidden_dim],
    );
    backend.init(&hp).expect("init should succeed");

    let gdn_layer = build_gdn_layer(0, &hp);
    let x_values: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();

    // Token 0: compute and capture output.
    let mut x0 = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x0.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x0, &gdn_layer, None, 0)
        .expect("token 0 GDN");
    let out0: Vec<f32> = x0.as_f32_slice().to_vec();

    // Token 1: same input but state has been updated by token 0.
    let mut x1 = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x1.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x1, &gdn_layer, None, 1)
        .expect("token 1 GDN");
    let out1: Vec<f32> = x1.as_f32_slice().to_vec();

    // Token 2: state updated by tokens 0+1.
    let mut x2 = ActivationBuffer::zeros(hidden_dim, ComputeDtype::F32);
    x2.write_f32_from(&x_values);
    backend
        .compute_layer(0, &mut x2, &gdn_layer, None, 2)
        .expect("token 2 GDN");
    let out2: Vec<f32> = x2.as_f32_slice().to_vec();

    // All outputs should be different due to state accumulation.
    let diff_01: f32 = out0
        .iter()
        .zip(&out1)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_12: f32 = out1
        .iter()
        .zip(&out2)
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_01 > 1e-6,
        "token 0 and 1 outputs should differ (diff={})",
        diff_01
    );
    assert!(
        diff_12 > 1e-6,
        "token 1 and 2 outputs should differ (diff={})",
        diff_12
    );

    // All outputs should be finite.
    for (tok, out) in [(0, &out0), (1, &out1), (2, &out2)] {
        for (i, &val) in out.iter().enumerate() {
            assert!(
                val.is_finite(),
                "token {tok} output[{i}] should be finite, got {val}"
            );
        }
    }
}
