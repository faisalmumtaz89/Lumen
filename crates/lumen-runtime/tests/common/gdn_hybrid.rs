//! A small, self-consistent GDN/full-attention hybrid LBC: layer 0 a full GDN
//! layer (every `ssm_*` tensor and `attn_gate` present), layer 1 a standard
//! full-attention layer, all tensors F32, dims consistent with
//! `GdnParams::from_hyperparams`. The format crate's `generate_test_model_q8_0_gdn`
//! fixture cannot run the GDN compute (no `attn_gate`, ssm dims sized from
//! hidden/head_dim), so the tests that execute GDN layers build this one.

use lumen_format::header::LbcHeader;
use lumen_format::hyperparams::{GdnDims, ModelHyperparams, RopeParams};
use lumen_format::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
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

// ---------------------------------------------------------------------------
// The Qwen3.5-shaped fixture the CPU backends run.

/// What [`build_qwen35_cpu_lbc`] builds.
#[derive(Clone, Copy, Default)]
pub struct Qwen35Fixture {
    /// Prepend a Llama-style dense layer (F32, no Q/K norm).
    pub plain_first: bool,
    /// Store the matrices in every scheme the CPU decodes (Q8_0, Q4_0, Q4_1,
    /// Q5_0, F16, BF16, Q4_K, Q6_K) instead of F32 alone.
    pub mixed_quant: bool,
    /// Give the last layer's `attn_q_norm` one element too few.
    pub bad_last_layer: bool,
}

const CPU_HEAD_DIM: u32 = 256;
const CPU_ROTARY_DIM: u32 = 64;
const CPU_EXPERTS: usize = 4;

/// Hyperparams of [`build_qwen35_cpu_lbc`]: GDN 4 V heads / 2 K heads x 128,
/// gated attention 4 heads / 2 KV heads x 256 with 64 rotary dims, 4 experts
/// with 2 active.
fn qwen35_cpu_hyperparams(fx: &Qwen35Fixture, max_seq_len: u32) -> ModelHyperparams {
    ModelHyperparams {
        num_layers: 4 + fx.plain_first as u32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: CPU_HEAD_DIM,
        hidden_dim: 64,
        intermediate_dim: 64,
        vocab_size: 64,
        max_seq_len,
        rope_params: Some(RopeParams {
            theta: 10_000_000.0,
            ..RopeParams::default()
        }),
        num_experts: Some(CPU_EXPERTS as u32),
        num_active_experts: Some(2),
        norm_eps: 1e-6,
        rotary_dim: Some(CPU_ROTARY_DIM),
        rope_neox: true,
        gdn: Some(GdnDims {
            num_v_heads: GDN_V_HEADS as u32,
            num_k_heads: GDN_K_HEADS as u32,
            head_dim: GDN_STATE_DIM as u32,
            conv_kernel: GDN_CONV_KERNEL as u32,
        }),
    }
}

/// Deterministic values in `[-scale, scale)`, distinct per `seed`.
fn rand_vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// f32 -> IEEE f16 bits for the normal range (truncating), enough for scales.
fn f16_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let exp = exp.clamp(1, 30) as u16;
    sign | (exp << 10) | ((b >> 13) & 0x3ff) as u16
}

/// `values` (a row-major `rows x cols` matrix) stored as `scheme`. The K-quant,
/// Q4_1 and Q5_0 blocks are seeded bytes with small scales: what both CPU
/// backends decode from them is what matters, not that they round `values`.
fn encode(values: &[f32], cols: usize, scheme: QuantScheme, seed: u64) -> Vec<u8> {
    let mut out = Vec::new();
    match scheme {
        QuantScheme::F32 => out.extend(values.iter().flat_map(|v| v.to_le_bytes())),
        QuantScheme::F16 => out.extend(values.iter().flat_map(|v| f16_bits(*v).to_le_bytes())),
        QuantScheme::Bf16 => out.extend(
            values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()),
        ),
        QuantScheme::Q8_0 => {
            for b in values.chunks_exact(32) {
                let amax = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                out.extend_from_slice(&f16_bits(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend(b.iter().map(|v| (v * inv).round() as i8 as u8));
            }
        }
        QuantScheme::Q4_0 => {
            for b in values.chunks_exact(32) {
                let max = b
                    .iter()
                    .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
                let d = max / -8.0;
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                let q = |v: f32| ((v * inv + 8.5) as i32).clamp(0, 15) as u8;
                out.extend_from_slice(&f16_bits(d).to_le_bytes());
                out.extend((0..16).map(|i| q(b[i]) | (q(b[i + 16]) << 4)));
            }
        }
        QuantScheme::Q4_1 | QuantScheme::Q5_0 | QuantScheme::Q4_K | QuantScheme::Q6_K => {
            let (elems, block, scales): (usize, usize, &[usize]) = match scheme {
                QuantScheme::Q4_1 => (32, 20, &[0, 2]),
                QuantScheme::Q5_0 => (32, 22, &[0]),
                QuantScheme::Q4_K => (256, 144, &[0, 2]),
                _ => (256, 210, &[208]),
            };
            assert_eq!(cols % elems, 0, "{scheme:?} rows of {cols}");
            let raw = rand_vals(values.len() / elems * block, seed, 1.0);
            out = raw.iter().map(|v| (v.to_bits() >> 7) as u8).collect();
            let d = match scheme {
                QuantScheme::Q4_1 => 0.01,
                QuantScheme::Q5_0 => 0.006,
                QuantScheme::Q4_K => 0.0002,
                _ => 0.0003,
            };
            for blk in out.chunks_exact_mut(block) {
                for &o in scales {
                    let v = if o == 2 && scheme == QuantScheme::Q4_1 {
                        -0.07
                    } else {
                        d
                    };
                    blk[o..o + 2].copy_from_slice(&f16_bits(v).to_le_bytes());
                }
            }
        }
        other => panic!("no fixture encoding for {other:?}"),
    }
    out
}

/// A Qwen3.5-shaped hybrid the CPU backends run end to end: [an optional
/// plain dense layer,] GDN + dense FFN, GDN + MoE, gated attention + dense FFN
/// (its FFN norm stored as `attn_post_norm` behind a zero-length `ffn_norm`),
/// gated attention + MoE with a gated shared expert. Every layer's weights are
/// distinct.
pub fn build_qwen35_cpu_lbc(fx: &Qwen35Fixture, max_seq_len: u32) -> Vec<u8> {
    let hp = qwen35_cpu_hyperparams(fx, max_seq_len);
    let hidden = hp.hidden_dim as usize;
    let inter = hp.intermediate_dim as usize;
    let hd = CPU_HEAD_DIM as usize;
    let q_dim = hp.num_heads as usize * hd;
    let kv_dim = hp.num_kv_heads as usize * hd;
    let vocab = hp.vocab_size as usize;
    let d = GDN_STATE_DIM;
    let value_dim = GDN_V_HEADS * d;
    let qkv_dim = 2 * GDN_K_HEADS * d + value_dim;
    // (scheme for hidden-wide inputs, scheme for 256-multiple inputs)
    let schemes: [(QuantScheme, QuantScheme); 4] = if fx.mixed_quant {
        [
            (QuantScheme::Q8_0, QuantScheme::Q4_K),
            (QuantScheme::Q4_0, QuantScheme::Q6_K),
            (QuantScheme::F16, QuantScheme::Q5_0),
            (QuantScheme::Bf16, QuantScheme::Q4_1),
        ]
    } else {
        [(QuantScheme::F32, QuantScheme::F32); 4]
    };

    let embedding = f32_vec_to_bytes(&rand_vals(vocab * hidden, 1, 1.0));
    let final_norm = f32_vec_to_bytes(&gen_norm_vals(hidden));
    let output_proj = f32_vec_to_bytes(&rand_vals(vocab * hidden, 2, 0.3));

    let mut layer_blobs: Vec<Vec<u8>> = Vec::new();
    let mut layer_indices: Vec<LayerIndex> = Vec::new();
    let kinds: Vec<Option<usize>> = (fx.plain_first.then_some(None).into_iter())
        .chain((0..4).map(Some))
        .collect();
    let last = kinds.len() - 1;
    for (layer, kind) in kinds.into_iter().enumerate() {
        let mut blob: Vec<u8> = Vec::new();
        let seed = 100 * (layer as u64 + 1);
        let mut n = 0u64;
        let mut push_as = |blob: &mut Vec<u8>, v: &[f32], cols: usize, q: QuantScheme| {
            n += 1;
            let bytes = encode(v, cols, q, seed + n);
            let ts = TensorSlice {
                offset: blob.len() as u64,
                length: bytes.len() as u64,
                quant: q,
            };
            blob.extend_from_slice(&bytes);
            ts
        };
        let zero = TensorSlice {
            offset: 0,
            length: 0,
            quant: QuantScheme::F32,
        };
        let mut st = SubtensorOffsets {
            wq: zero,
            wk: zero,
            wv: zero,
            wo: zero,
            bq: None,
            bk: None,
            bv: None,
            w_gate: zero,
            w_up: zero,
            w_down: zero,
            attn_norm: zero,
            ffn_norm: zero,
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
            layer_type: None,
        };
        let f32q = QuantScheme::F32;
        let norm = |i: u64| -> Vec<f32> {
            rand_vals(hidden, seed + 50 + i, 0.2)
                .iter()
                .map(|v| 1.0 + v)
                .collect()
        };
        st.attn_norm = push_as(&mut blob, &norm(0), hidden, f32q);
        let (narrow, wide) = kind.map_or((f32q, f32q), |k| schemes[k]);
        let mat = |rows: usize, cols: usize, i: u64, scale: f32| {
            rand_vals(rows * cols, seed + i, scale / (cols as f32).sqrt())
        };
        match kind {
            None => {
                st.wq = push_as(&mut blob, &mat(q_dim, hidden, 1, 1.0), hidden, f32q);
                st.wk = push_as(&mut blob, &mat(kv_dim, hidden, 2, 1.0), hidden, f32q);
                st.wv = push_as(&mut blob, &mat(kv_dim, hidden, 3, 1.0), hidden, f32q);
                st.wo = push_as(&mut blob, &mat(hidden, q_dim, 4, 1.0), q_dim, f32q);
                st.layer_type = Some(0);
            }
            Some(k) if k < 2 => {
                st.wq = push_as(&mut blob, &mat(qkv_dim, hidden, 1, 1.0), hidden, narrow);
                st.attn_gate = Some(push_as(
                    &mut blob,
                    &mat(value_dim, hidden, 2, 1.0),
                    hidden,
                    narrow,
                ));
                st.ssm_alpha = Some(push_as(
                    &mut blob,
                    &mat(GDN_V_HEADS, hidden, 3, 1.0),
                    hidden,
                    f32q,
                ));
                st.ssm_beta = Some(push_as(
                    &mut blob,
                    &mat(GDN_V_HEADS, hidden, 4, 1.0),
                    hidden,
                    f32q,
                ));
                let a: Vec<f32> = (0..GDN_V_HEADS).map(|h| -0.1 - 0.7 * h as f32).collect();
                st.ssm_a = Some(push_as(&mut blob, &a, GDN_V_HEADS, f32q));
                st.ssm_dt = Some(push_as(
                    &mut blob,
                    &rand_vals(GDN_V_HEADS, seed + 5, 1.0),
                    GDN_V_HEADS,
                    f32q,
                ));
                st.ssm_conv1d = Some(push_as(
                    &mut blob,
                    &rand_vals(qkv_dim * GDN_CONV_KERNEL, seed + 6, 0.6),
                    qkv_dim * GDN_CONV_KERNEL,
                    f32q,
                ));
                let sn: Vec<f32> = rand_vals(d, seed + 7, 0.3)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect();
                st.ssm_norm = Some(push_as(&mut blob, &sn, d, f32q));
                st.ssm_out = Some(push_as(
                    &mut blob,
                    &mat(hidden, value_dim, 8, 1.0),
                    value_dim,
                    wide,
                ));
                st.layer_type = Some(1);
            }
            Some(_) => {
                st.wq = push_as(&mut blob, &mat(2 * q_dim, hidden, 1, 1.0), hidden, narrow);
                st.wk = push_as(&mut blob, &mat(kv_dim, hidden, 2, 1.0), hidden, narrow);
                st.wv = push_as(&mut blob, &mat(kv_dim, hidden, 3, 1.0), hidden, narrow);
                st.wo = push_as(&mut blob, &mat(hidden, q_dim, 4, 1.0), q_dim, wide);
                let qn_len = if fx.bad_last_layer && layer == last {
                    hd - 1
                } else {
                    hd
                };
                let qn: Vec<f32> = rand_vals(qn_len, seed + 5, 0.3)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect();
                let kn: Vec<f32> = rand_vals(hd, seed + 6, 0.3)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect();
                st.attn_q_norm = Some(push_as(&mut blob, &qn, qn_len, f32q));
                st.attn_k_norm = Some(push_as(&mut blob, &kn, hd, f32q));
                st.layer_type = Some(0);
            }
        }
        if kind == Some(2) {
            st.attn_post_norm = Some(push_as(&mut blob, &norm(1), hidden, f32q));
        } else {
            st.ffn_norm = push_as(&mut blob, &norm(1), hidden, f32q);
        }
        if matches!(kind, Some(1) | Some(3)) {
            st.router_weight = Some(push_as(
                &mut blob,
                &mat(CPU_EXPERTS, hidden, 20, 3.0),
                hidden,
                f32q,
            ));
            let mut experts = Vec::new();
            for e in 0..=CPU_EXPERTS as u64 {
                let gate = push_as(
                    &mut blob,
                    &mat(inter, hidden, 21 + 3 * e, 1.0),
                    hidden,
                    narrow,
                );
                let up = push_as(
                    &mut blob,
                    &mat(inter, hidden, 22 + 3 * e, 1.0),
                    hidden,
                    narrow,
                );
                let down = push_as(&mut blob, &mat(hidden, inter, 23 + 3 * e, 1.0), inter, f32q);
                experts.push(ExpertSlice { gate, up, down });
            }
            let shared = experts.pop().unwrap();
            st.experts = Some(experts);
            st.shared_expert_gate = Some(shared.gate);
            st.shared_expert_up = Some(shared.up);
            st.shared_expert_down = Some(shared.down);
            st.ffn_gate_inp_shexp =
                Some(push_as(&mut blob, &mat(1, hidden, 40, 1.0), hidden, f32q));
        } else {
            let q = if kind.is_none() { f32q } else { narrow };
            st.w_gate = push_as(&mut blob, &mat(inter, hidden, 20, 1.0), hidden, q);
            st.w_up = push_as(&mut blob, &mat(inter, hidden, 21, 1.0), hidden, q);
            st.w_down = push_as(&mut blob, &mat(hidden, inter, 22, 1.0), inter, f32q);
        }
        layer_indices.push(LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors: st,
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
    .expect("failed to write the Qwen3.5 CPU fixture");
    out
}
