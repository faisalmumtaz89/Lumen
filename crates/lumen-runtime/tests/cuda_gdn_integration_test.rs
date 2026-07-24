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

use lumen_format::header::LbcHeader;
use lumen_format::hyperparams::{GdnDims, ModelHyperparams, RopeParams};
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
use lumen_format::writer::{write_lbc, GlobalTensors};
use lumen_runtime::compute::{ActivationBuffer, ComputeBackend, ComputeDtype};
use lumen_runtime::weight::cache::LayerView;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

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
        rotary_dim: None,
        rope_neox: false,
        gdn: None,
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
    let wk = TensorSlice {
        offset: 0,
        length: 0,
        quant: QuantScheme::F32,
    };
    let wv = TensorSlice {
        offset: 0,
        length: 0,
        quant: QuantScheme::F32,
    };
    // wo not used for GDN (ssm_out replaces it) but must be present
    let wo = TensorSlice {
        offset: 0,
        length: 0,
        quant: QuantScheme::F32,
    };
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
// Runnable hybrid GDN model (for the `compute_layer` GDN-path tests)
// ---------------------------------------------------------------------------
//
// The synthetic `build_gdn_layer` above yields a `LayerView` for the *detection*
// test, but the shipped `compute_layer` GDN contract (2026) requires GPU-resident
// (preloaded) weights that are (a) dimensionally consistent with
// `GdnParams::from_hyperparams` and (b) include `attn_gate` — the decode
// megakernel errors "attn_gate weight missing" otherwise. The format-guard
// fixture `generate_test_model_q8_0_gdn` satisfies NEITHER (no `attn_gate`; its
// ssm_* tensors are sized from hidden/head_dim, not the SSM dims; and `hp.gdn`
// is `None` so `GdnParams` defaults to the Qwen3.5-9B dims). So we build a small,
// self-consistent hybrid LBC here: `hp.gdn = Some({V=4,K=2,D=128,C=4})` with
// layer 0 a full GDN layer (all ssm_* + attn_gate present) and layer 1 a standard
// full-attention layer. All weights are F32 (`launch_matvec` has an F32 arm), so
// preload uploads them natively and the decode GDN pipeline runs end-to-end.

const GDN_V_HEADS: usize = 4; // ssm.time_step_rank (state/V heads)
const GDN_K_HEADS: usize = 2; // ssm.group_count (Q/K heads pre-GQA)
const GDN_STATE_DIM: usize = 128; // ssm.state_size (per-head dim)
const GDN_CONV_KERNEL: usize = 4;

fn gdn_model_hyperparams() -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 64,
        max_seq_len: 64,
        rope_params: Some(RopeParams::default()),
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-5,
        rotary_dim: None,
        rope_neox: false,
        // GDN SSM dims resolved by `GdnParams::from_hyperparams`. MUST match the
        // ssm_* tensor sizes below (qkv_dim=1024, value_dim=512, etc.).
        gdn: Some(GdnDims {
            num_v_heads: GDN_V_HEADS as u32,
            num_k_heads: GDN_K_HEADS as u32,
            head_dim: GDN_STATE_DIM as u32,
            conv_kernel: GDN_CONV_KERNEL as u32,
        }),
    }
}

fn gen_weight_vals(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.001 + phase).sin() * 0.1)
        .collect()
}

fn gen_norm_vals(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + ((i as f32) * 0.01).sin() * 0.01)
        .collect()
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build a 2-layer hybrid GDN LBC (bytes): layer 0 = GDN (all ssm_* + attn_gate),
/// layer 1 = full attention. All tensors F32. Dims match `gdn_model_hyperparams()`.
fn build_gdn_hybrid_lbc() -> Vec<u8> {
    let hp = gdn_model_hyperparams();
    let hidden = hp.hidden_dim as usize;
    let inter = hp.intermediate_dim as usize;
    let q_dim = hp.num_heads as usize * hp.head_dim as usize;
    let kv_dim = hp.num_kv_heads as usize * hp.head_dim as usize;
    let vocab = hp.vocab_size as usize;

    // GDN SSM dims (must mirror GdnParams::from_hyperparams for hp.gdn above).
    let v_heads = GDN_V_HEADS;
    let k_heads = GDN_K_HEADS;
    let d = GDN_STATE_DIM;
    let qk_dim = k_heads * d; // 256
    let value_dim = v_heads * d; // 512
    let qkv_dim = 2 * qk_dim + value_dim; // 1024

    let embedding = f32_vec_to_bytes(&gen_weight_vals(vocab * hidden, 0.1));
    let final_norm = f32_vec_to_bytes(&gen_norm_vals(hidden));
    let output_proj = f32_vec_to_bytes(&gen_weight_vals(vocab * hidden, 0.2));

    let mut layer_blobs: Vec<Vec<u8>> = Vec::new();
    let mut layer_indices: Vec<LayerIndex> = Vec::new();

    for layer in 0..hp.num_layers as usize {
        let mut blob: Vec<u8> = Vec::new();
        let mut off: u64 = 0;
        let push = |blob: &mut Vec<u8>, off: &mut u64, values: &[f32]| -> TensorSlice {
            let bytes = f32_vec_to_bytes(values);
            let len = bytes.len() as u64;
            let ts = TensorSlice {
                offset: *off,
                length: len,
                quant: QuantScheme::F32,
            };
            blob.extend_from_slice(&bytes);
            *off += len;
            ts
        };

        let attn_norm = push(&mut blob, &mut off, &gen_norm_vals(hidden));

        let is_gdn = layer == 0;
        // GDN: wq is the FUSED QKV projection [qkv_dim, hidden]; full-attn: [q_dim, hidden].
        let wq_out = if is_gdn { qkv_dim } else { q_dim };
        let wq = push(&mut blob, &mut off, &gen_weight_vals(wq_out * hidden, 0.3));
        let wk = push(&mut blob, &mut off, &gen_weight_vals(kv_dim * hidden, 0.4));
        let wv = push(&mut blob, &mut off, &gen_weight_vals(kv_dim * hidden, 0.5));
        let wo = push(&mut blob, &mut off, &gen_weight_vals(hidden * q_dim, 0.6));
        let w_gate = push(&mut blob, &mut off, &gen_weight_vals(inter * hidden, 0.7));
        let w_up = push(&mut blob, &mut off, &gen_weight_vals(inter * hidden, 0.8));
        let w_down = push(&mut blob, &mut off, &gen_weight_vals(hidden * inter, 0.9));
        let ffn_norm = push(&mut blob, &mut off, &gen_norm_vals(hidden));

        #[allow(clippy::type_complexity)]
        let (
            ssm_a,
            ssm_conv1d,
            ssm_dt,
            ssm_beta,
            ssm_alpha,
            ssm_norm,
            ssm_out,
            attn_gate,
            layer_type,
        ): (
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<TensorSlice>,
            Option<u8>,
        ) = if is_gdn {
            // Conv weight: [conv_dim=qkv_dim, kernel]. Order irrelevant (finiteness test).
            let conv = push(
                &mut blob,
                &mut off,
                &gen_weight_vals(qkv_dim * GDN_CONV_KERNEL, 1.1),
            );
            // ssm_a MUST be negative: alpha = exp(ssm_a * softplus(..)) then lands in (0,1].
            let a = push(&mut blob, &mut off, &vec![-0.5f32; v_heads]);
            let dt = push(&mut blob, &mut off, &gen_weight_vals(v_heads, 1.2));
            // ssm_norm is [head_dim]; upload tiles it to [value_dim].
            let norm = push(&mut blob, &mut off, &gen_norm_vals(d));
            // alpha/beta project normed[hidden] -> [num_v_heads].
            let alpha = push(&mut blob, &mut off, &gen_weight_vals(v_heads * hidden, 1.3));
            let beta = push(&mut blob, &mut off, &gen_weight_vals(v_heads * hidden, 1.4));
            // ssm_out projects [value_dim] -> [hidden].
            let out = push(
                &mut blob,
                &mut off,
                &gen_weight_vals(hidden * value_dim, 1.5),
            );
            // attn_gate projects normed[hidden] -> [value_dim].
            let gate = push(
                &mut blob,
                &mut off,
                &gen_weight_vals(value_dim * hidden, 1.6),
            );
            (
                Some(a),
                Some(conv),
                Some(dt),
                Some(beta),
                Some(alpha),
                Some(norm),
                Some(out),
                Some(gate),
                Some(1u8),
            )
        } else {
            (None, None, None, None, None, None, None, None, Some(0u8))
        };

        let subtensors = SubtensorOffsets {
            wq,
            wk,
            wv,
            wo,
            bq: None,
            bk: None,
            bv: None,
            w_gate,
            w_up,
            w_down,
            attn_norm,
            ffn_norm,
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
            layer_type,
        };

        layer_indices.push(LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors,
        });
        layer_blobs.push(blob);
    }

    let qd = QuantizationDescriptor {
        scheme: QuantScheme::F32,
        group_size: QuantGroupSize::PerTensor,
        block_byte_size: 4,
        scale_offset_in_block: None,
    };
    let header = LbcHeader::new(hp, qd);
    let globals = GlobalTensors {
        embedding,
        final_norm,
        output_proj,
    };
    let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();

    let mut out = Vec::new();
    write_lbc(
        &mut out,
        &header,
        &layer_indices,
        &globals,
        &blob_refs,
        None,
    )
    .expect("failed to write GDN hybrid test model");
    out
}

static GDN_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

struct GdnSetup {
    provider: SyncWeightProvider,
    backend: lumen_runtime::CudaBackend,
    hp: ModelHyperparams,
}

/// Build the hybrid GDN LBC, open a `SyncWeightProvider`, create a `CudaBackend`,
/// `init`, and `preload_weights` (GPU-resident) — the shipped GDN-compute contract.
fn setup_gdn_backend() -> GdnSetup {
    let lbc = build_gdn_hybrid_lbc();
    let id = GDN_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lumen_cuda_gdn_integ_{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gdn_hybrid.lbc");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&lbc).unwrap();
    }
    let provider = SyncWeightProvider::open(&path).expect("open GDN provider");
    let hp = provider.lbc().header.hyperparams;
    let mut backend = lumen_runtime::CudaBackend::new(0).expect("CudaBackend::new(0)");
    backend.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    backend.init(&hp).expect("init");
    // Shipped contract: GDN layers require GPU-resident (preloaded) weights.
    backend
        .preload_weights(&provider)
        .expect("preload_weights (GDN requires GPU-resident weights)");
    GdnSetup {
        provider,
        backend,
        hp,
    }
}

fn new_gdn_kv(hp: &ModelHyperparams) -> lumen_runtime::KvCache {
    lumen_runtime::KvCache::new(lumen_runtime::KvCacheConfig {
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        max_seq_len: hp.max_seq_len as usize,
        head_dim: hp.head_dim as usize,
        precision: lumen_runtime::KvPrecision::F32,
    })
    .expect("kv cache allocation")
}

fn gdn_input(hidden: usize) -> ActivationBuffer {
    let vals: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let mut x = ActivationBuffer::zeros(hidden, ComputeDtype::F32);
    x.write_f32_from(&vals);
    x
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

/// Verify that `compute_layer` routes GDN layers (layer_type==1) through the GDN
/// path under the shipped contract (GPU-resident preloaded weights + a KV view).
///
/// Verifies:
/// 1. `compute_layer` succeeds (the GDN pipeline ran; `attn_gate`/ssm_* resolved)
/// 2. The output is finite (no NaN/Inf from the GDN megakernel)
/// 3. The output has the correct shape (hidden_dim elements)
#[test]
fn test_compute_layer_gdn_routing() {
    let s = setup_gdn_backend();
    let hidden = s.hp.hidden_dim as usize;

    // Layer 0 of the hybrid model is the GDN layer.
    let layer0 = s.provider.get_layer_raw(0).expect("layer 0 (GDN) view");
    assert_eq!(
        layer0.subtensors.layer_type,
        Some(1),
        "layer 0 must be a GDN layer"
    );

    let mut kv = new_gdn_kv(&s.hp);
    let mut x = gdn_input(hidden);

    // Shipped contract: kv view is REQUIRED even for GDN (advances seq tracking).
    let mut kv_view = kv.view_mut(0).expect("kv view");
    let result = s
        .backend
        .compute_layer(0, &mut x, &layer0, Some(&mut kv_view), 0);
    kv.commit_view(kv_view).ok();
    assert!(
        result.is_ok(),
        "compute_layer should route GDN layer through the GDN path: {:?}",
        result.err()
    );

    let output_f32 = x.as_f32_slice();
    assert_eq!(output_f32.len(), hidden);
    for (i, &val) in output_f32.iter().enumerate() {
        assert!(
            val.is_finite(),
            "GDN output[{i}] should be finite, got {val}"
        );
    }
}

/// Verify that `reset_recurrent_state` clears GDN state.
///
/// Exercised through the shipped GDN decode entry `decode_token` (the engine's
/// per-token path for GDN models — `prefill` + `decode_token`). NOTE: a single
/// `compute_layer` call on the GDN layer does advance h_states in-place (the
/// register-resident phase4 carries them), but that contribution to one layer's
/// output rounds away against the residual; `decode_token` amplifies it through
/// `compute_final`/output_proj, so state effects are observable.
///
/// After a fresh-state decode, `reset_recurrent_state()` (which zeros GDN
/// h_states/conv_states and the GPU KV caches) plus a fresh KV cache reproduces
/// the very first (position-0) decode, while a second decode (advanced state +
/// grown KV) differs.
#[test]
fn test_reset_recurrent_state() {
    let s = setup_gdn_backend();
    let tok = 1u32;

    // Fresh state: decode the same token twice.
    s.backend.reset_recurrent_state();
    let mut kv1 = new_gdn_kv(&s.hp);
    let a0: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv1)
        .expect("decode a0 (pos 0)")
        .data;
    let a1: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv1)
        .expect("decode a1 (pos 1)")
        .data;

    // Reset + fresh KV -> the next position-0 decode reproduces the first one.
    s.backend.reset_recurrent_state();
    let mut kv2 = new_gdn_kv(&s.hp);
    let b0: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv2)
        .expect("decode b0 (post-reset, pos 0)")
        .data;

    assert_eq!(a0.len(), b0.len(), "logits length mismatch");
    for i in 0..a0.len() {
        assert!(
            (a0[i] - b0[i]).abs() < 1e-4,
            "post-reset logit[{i}] = {} should match first decode {} (tol=1e-4)",
            b0[i],
            a0[i],
        );
    }

    let any_differ = a0.iter().zip(&a1).any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(
        any_differ,
        "second decode (advanced state) should differ from first (fresh state)"
    );

    for (i, &val) in a0.iter().enumerate() {
        assert!(val.is_finite(), "a0 logit[{i}] should be finite, got {val}");
    }
}

/// Test hybrid model routing: layer 0 (GDN) and layer 1 (full attention) both
/// compute through `compute_layer`, each routed to its correct path.
#[test]
fn test_hybrid_model_routing() {
    let s = setup_gdn_backend();
    let hidden = s.hp.hidden_dim as usize;

    let layer0 = s.provider.get_layer_raw(0).expect("layer 0 view");
    let layer1 = s.provider.get_layer_raw(1).expect("layer 1 view");
    assert_eq!(layer0.subtensors.layer_type, Some(1), "layer 0 must be GDN");
    assert_eq!(
        layer1.subtensors.layer_type,
        Some(0),
        "layer 1 must be full attention"
    );

    let mut kv = new_gdn_kv(&s.hp);
    let mut x = gdn_input(hidden);

    // Layer 0: GDN path.
    {
        let mut v = kv.view_mut(0).expect("kv view 0");
        s.backend
            .compute_layer(0, &mut x, &layer0, Some(&mut v), 0)
            .expect("layer 0 (GDN) compute");
        kv.commit_view(v).unwrap();
    }
    // Layer 1: standard attention path.
    {
        let mut v = kv.view_mut(1).expect("kv view 1");
        s.backend
            .compute_layer(1, &mut x, &layer1, Some(&mut v), 0)
            .expect("layer 1 (attention) compute");
        kv.commit_view(v).unwrap();
    }

    for (i, &val) in x.as_f32_slice().iter().enumerate() {
        assert!(
            val.is_finite(),
            "hybrid output[{i}] should be finite, got {val}"
        );
    }
}

/// Verify that GDN recurrent state persists across token positions.
///
/// Exercised through the shipped GDN decode entry `decode_token` (the engine's
/// per-token path for GDN models). The recurrent h_states/conv_states (and the
/// attention KV) evolve as the sequence advances, so decoding the SAME token
/// repeatedly yields DIFFERENT logits each step. (A single-layer `compute_layer`
/// call advances h_states in-place too, but that per-layer contribution rounds
/// away against the residual; `decode_token` amplifies it through the full model
/// + output projection.)
#[test]
fn test_gdn_state_persistence() {
    let s = setup_gdn_backend();
    let mut kv = new_gdn_kv(&s.hp);
    s.backend.reset_recurrent_state();

    let tok = 1u32;
    let out0: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv)
        .expect("decode token 0")
        .data;
    let out1: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv)
        .expect("decode token 1")
        .data;
    let out2: Vec<f32> = s
        .backend
        .decode_token(tok, &s.provider, &mut kv)
        .expect("decode token 2")
        .data;

    let diff_01: f32 = out0.iter().zip(&out1).map(|(a, b)| (a - b).abs()).sum();
    let diff_12: f32 = out1.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();

    assert!(
        diff_01 > 1e-6,
        "token 0 and 1 logits should differ (GDN state + KV advanced), diff={diff_01}"
    );
    assert!(
        diff_12 > 1e-6,
        "token 1 and 2 logits should differ (GDN state + KV advanced), diff={diff_12}"
    );

    for (step, out) in [(0, &out0), (1, &out1), (2, &out2)] {
        for (i, &val) in out.iter().enumerate() {
            assert!(
                val.is_finite(),
                "step {step} logit[{i}] should be finite, got {val}"
            );
        }
    }
}
