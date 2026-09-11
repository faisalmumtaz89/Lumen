//! A small, self-consistent GDN/full-attention hybrid LBC: layer 0 a full GDN
//! layer (every `ssm_*` tensor and `attn_gate` present), layer 1 a standard
//! full-attention layer, all tensors F32, dims consistent with
//! `GdnParams::from_hyperparams`. The format crate's `generate_test_model_q8_0_gdn`
//! fixture cannot run the GDN compute (no `attn_gate`, ssm dims sized from
//! hidden/head_dim), so the tests that execute GDN layers build this one.

use lumen_format::header::LbcHeader;
use lumen_format::hyperparams::{GdnDims, ModelHyperparams, RopeParams};
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
use lumen_format::writer::{write_lbc, GlobalTensors};
use std::sync::atomic::AtomicU64;

pub const GDN_V_HEADS: usize = 4; // ssm.time_step_rank (state/V heads)
pub const GDN_K_HEADS: usize = 2; // ssm.group_count (Q/K heads pre-GQA)
pub const GDN_STATE_DIM: usize = 128; // ssm.state_size (per-head dim)
pub const GDN_CONV_KERNEL: usize = 4;

pub fn gdn_model_hyperparams() -> ModelHyperparams {
    gdn_model_hyperparams_with(64)
}

/// `gdn_model_hyperparams` with the given context length.
pub fn gdn_model_hyperparams_with(max_seq_len: u32) -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 128,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 64,
        max_seq_len,
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

pub fn gen_weight_vals(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.001 + phase).sin() * 0.1)
        .collect()
}

pub fn gen_norm_vals(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + ((i as f32) * 0.01).sin() * 0.01)
        .collect()
}

pub fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build a 2-layer hybrid GDN LBC (bytes): layer 0 = GDN (all ssm_* + attn_gate),
/// layer 1 = full attention. All tensors F32. Dims match `gdn_model_hyperparams()`.
pub fn build_gdn_hybrid_lbc() -> Vec<u8> {
    build_gdn_hybrid_lbc_with(gdn_model_hyperparams())
}

/// `build_gdn_hybrid_lbc` for the given hyperparams (the GDN dims are the constants above).
pub fn build_gdn_hybrid_lbc_with(hp: ModelHyperparams) -> Vec<u8> {
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

pub static GDN_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
