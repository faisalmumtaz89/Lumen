//! End-to-end conversions of synthetic Q5_0 models, asserting the slice
//! planner and the blob writer stay in lockstep for the Metal upcast gates.
//!
//! The StreamingLbcWriter hard-errors when a layer blob's length differs from
//! the planner's `layer_length_bytes`, so a successful conversion IS the
//! planner/writer agreement check; the extent assertion below additionally
//! pins every subtensor slice inside the recorded layer.

use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const HID: u64 = 64;
const INTER: u64 = 128;
const VOCAB: u64 = 256;
const HEADS: u32 = 8;
const KVH: u32 = 4;
const LAYERS: u32 = 2;
const NEXP: u64 = 4;

fn q5_bytes(n: u64) -> Vec<u8> {
    let mut v = vec![0u8; (n / 32) as usize * 22];
    for blk in v.chunks_exact_mut(22) {
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for (i, b) in blk[6..22].iter_mut().enumerate() {
            *b = (i as u8) | (((15 - i) as u8) << 4);
        }
    }
    v
}

fn build_q5_model(moe: bool) -> Vec<u8> {
    let arch = if moe { "qwen35moe" } else { "qwen35" };
    let k = |s: &str| format!("{arch}.{s}");
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", arch);
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), HID as u32 / HEADS);
    b.add_u32(&k("embedding_length"), HID as u32);
    b.add_u32(&k("feed_forward_length"), INTER as u32);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    if moe {
        b.add_u32(&k("expert_count"), NEXP as u32);
        b.add_u32(&k("expert_used_count"), 2);
    }
    b.add_f32_tensor(
        "token_embd.weight",
        &[VOCAB, HID],
        &vec![0.0; (VOCAB * HID) as usize],
    );
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    let kvd = (HID / HEADS as u64) * KVH as u64;
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let mut q = |nm: &str, dims: &[u64]| {
            let n: u64 = dims.iter().product();
            b.add_tensor(&format!("{p}.{nm}"), GgmlType::Q5_0, dims, q5_bytes(n));
        };
        q("attn_qkv.weight", &[HID, HID]);
        q("attn_q.weight", &[HID, HID]);
        q("attn_k.weight", &[HID, kvd]);
        q("attn_v.weight", &[HID, kvd]);
        q("attn_output.weight", &[HID, HID]);
        if moe {
            q("ffn_gate_exps.weight", &[HID, INTER, NEXP]);
            q("ffn_up_exps.weight", &[HID, INTER, NEXP]);
            q("ffn_down_exps.weight", &[INTER, HID, NEXP]);
            b.add_f32_tensor(
                &format!("{p}.ffn_gate_inp.weight"),
                &[HID, NEXP],
                &vec![0.0; (HID * NEXP) as usize],
            );
        } else {
            q("ffn_gate.weight", &[HID, INTER]);
            q("ffn_up.weight", &[HID, INTER]);
            q("ffn_down.weight", &[INTER, HID]);
        }
        b.add_f32_tensor(
            &format!("{p}.attn_norm.weight"),
            &[HID],
            &vec![1.0; HID as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ffn_norm.weight"),
            &[HID],
            &vec![1.0; HID as usize],
        );
    }
    b.build()
}

/// Converts and returns the gate tensor's (quant, length) after asserting
/// every layer's subtensor extents sit inside the recorded layer length.
fn convert_and_probe(label: &str, moe: bool, opts: &ConvertOptions) -> (QuantScheme, u64) {
    let gguf = build_q5_model(moe);
    let out = std::env::temp_dir().join(format!(
        "lumen_lockstep_{}_{}.lbc",
        label,
        std::process::id()
    ));
    convert_gguf_bytes_to_lbc(&gguf, &out, opts)
        .unwrap_or_else(|e| panic!("{label}: conversion failed (planner/writer desync?): {e:?}"));
    let f = LbcFile::open(&out).unwrap();
    for (li, idx) in f.layer_indices.iter().enumerate() {
        let mut extent = 0u64;
        for (_n, s) in idx.subtensors.named_slices() {
            extent = extent.max(s.offset + s.length);
        }
        if let Some(ex) = idx.subtensors.experts.as_ref() {
            for e in ex.iter() {
                for s in [&e.gate, &e.up, &e.down] {
                    extent = extent.max(s.offset + s.length);
                }
            }
        }
        assert_eq!(
            extent, idx.layer_length_bytes,
            "{label}: layer {li} planner extent != blob length"
        );
    }
    let probe = f.layer_indices[0]
        .subtensors
        .experts
        .as_ref()
        .and_then(|e| e.first().map(|x| (x.gate.quant, x.gate.length)))
        .unwrap_or((
            f.layer_indices[0].subtensors.w_gate.quant,
            f.layer_indices[0].subtensors.w_gate.length,
        ));
    std::fs::remove_file(&out).ok();
    probe
}

#[test]
fn metal_target_upcasts_q5_0_dense_and_moe() {
    let metal = ConvertOptions {
        target: ConvertTarget::Metal,
        ..Default::default()
    };
    // Dense gate: 64x128 = 8192 elems -> 256 Q8_0 blocks x 34 B.
    assert_eq!(
        convert_and_probe("dense_metal", false, &metal),
        (QuantScheme::Q8_0, 256 * 34)
    );
    // Per-expert gate: same 8192 elems per expert.
    assert_eq!(
        convert_and_probe("moe_metal", true, &metal),
        (QuantScheme::Q8_0, 256 * 34)
    );
}

#[test]
fn generic_target_preserves_q5_0() {
    let generic = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    // CUDA dequantizes Q5_0 at load; the plane stays 256 blocks x 22 B.
    assert_eq!(
        convert_and_probe("dense_generic", false, &generic),
        (QuantScheme::Q5_0, 256 * 22)
    );
    assert_eq!(
        convert_and_probe("moe_generic", true, &generic),
        (QuantScheme::Q5_0, 256 * 22)
    );
}

#[test]
fn moe_requant_is_refused() {
    let gguf = build_q5_model(true);
    let out = std::env::temp_dir().join(format!("lockstep_moe_req_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        requant_to: Some(QuantScheme::Q4_0),
        ..Default::default()
    };
    std::fs::remove_file(&out).ok();
    let err =
        convert_gguf_bytes_to_lbc(&gguf, &out, &opts).expect_err("MoE --requant must be refused");
    assert!(
        matches!(
            err,
            lumen_convert::convert::ConvertError::UnsupportedOption(_)
        ),
        "{err}"
    );
    assert!(!out.exists(), "no partial output on refusal");
}

#[test]
fn moe_dequantize_writes_f32_experts_on_both_targets() {
    // F16/Bf16/F32 expert banks are served by the Metal legacy per-expert
    // path and by CUDA, so `--dequantize` MoE output is loadable on both
    // targets; the gates stay Q8_0 on Metal via the shared gdn_gates logic.
    for target in [ConvertTarget::Metal, ConvertTarget::Generic] {
        let opts = ConvertOptions {
            target,
            dequantize_to_f32: true,
            ..Default::default()
        };
        assert_eq!(
            convert_and_probe(&format!("moe_deq_{target:?}"), true, &opts),
            (QuantScheme::F32, 8192 * 4)
        );
    }
}

#[test]
fn metal_target_q5_0_respects_flag_precedence() {
    // --dequantize wins over the upcast on both sides of the planner/writer.
    let deq = ConvertOptions {
        target: ConvertTarget::Metal,
        dequantize_to_f32: true,
        ..Default::default()
    };
    assert_eq!(
        convert_and_probe("dense_metal_deq", false, &deq),
        (QuantScheme::F32, 8192 * 4)
    );
    // --requant wins over the upcast for dense layer tensors.
    let req = ConvertOptions {
        target: ConvertTarget::Metal,
        requant_to: Some(QuantScheme::Q4_0),
        ..Default::default()
    };
    assert_eq!(
        convert_and_probe("dense_metal_req", false, &req),
        (QuantScheme::Q4_0, 256 * 18)
    );
}

/// Build a dense qwen35 model whose layer 0 is a complete GDN layer with
/// `ssm_alpha`/`ssm_beta` stored as `gates`.
fn build_gdn_model(gates: GgmlType) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    let arch = "qwen35";
    let k = |s: &str| format!("{arch}.{s}");
    b.add_string("general.architecture", arch);
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), HID as u32 / HEADS);
    b.add_u32(&k("embedding_length"), HID as u32);
    b.add_u32(&k("feed_forward_length"), INTER as u32);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    b.add_u32(&k("ssm.time_step_rank"), 8);
    b.add_u32(&k("ssm.group_count"), 4);
    b.add_u32(&k("ssm.state_size"), 8);
    b.add_u32(&k("ssm.conv_kernel"), 4);
    b.add_f32_tensor(
        "token_embd.weight",
        &[VOCAB, HID],
        &vec![0.0; (VOCAB * HID) as usize],
    );
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    let kvd = (HID / HEADS as u64) * KVH as u64;
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let mut q = |nm: &str, dims: &[u64]| {
            let n: u64 = dims.iter().product();
            b.add_tensor(&format!("{p}.{nm}"), GgmlType::Q5_0, dims, q5_bytes(n));
        };
        q("attn_qkv.weight", &[HID, HID]);
        q("attn_q.weight", &[HID, HID]);
        q("attn_k.weight", &[HID, kvd]);
        q("attn_v.weight", &[HID, kvd]);
        q("attn_output.weight", &[HID, HID]);
        q("ffn_gate.weight", &[HID, INTER]);
        q("ffn_up.weight", &[HID, INTER]);
        q("ffn_down.weight", &[INTER, HID]);
        b.add_f32_tensor(
            &format!("{p}.attn_norm.weight"),
            &[HID],
            &vec![1.0; HID as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ffn_norm.weight"),
            &[HID],
            &vec![1.0; HID as usize],
        );
        let nh = 8u64;
        b.add_f32_tensor(&format!("{p}.ssm_a"), &[nh], &vec![-0.5; nh as usize]);
        b.add_f32_tensor(
            &format!("{p}.ssm_conv1d.weight"),
            &[4, HID],
            &vec![0.1; (4 * HID) as usize],
        );
        b.add_f32_tensor(&format!("{p}.ssm_dt.bias"), &[nh], &vec![0.0; nh as usize]);
        b.add_f32_tensor(
            &format!("{p}.ssm_norm.weight"),
            &[HID / nh],
            &vec![1.0; (HID / nh) as usize],
        );
        b.add_tensor(
            &format!("{p}.ssm_out.weight"),
            GgmlType::Q5_0,
            &[HID, HID],
            q5_bytes(HID * HID),
        );
        let ne = HID * nh;
        match gates {
            GgmlType::F32 => {
                b.add_f32_tensor(
                    &format!("{p}.ssm_alpha.weight"),
                    &[HID, nh],
                    &vec![0.02; ne as usize],
                );
                b.add_f32_tensor(
                    &format!("{p}.ssm_beta.weight"),
                    &[HID, nh],
                    &vec![0.02; ne as usize],
                );
            }
            GgmlType::Q8_0 => {
                let bytes = |n: u64| {
                    let mut v = vec![0u8; (n / 32) as usize * 34];
                    for c in v.chunks_exact_mut(34) {
                        c[0] = 0x00;
                        c[1] = 0x3C;
                    }
                    v
                };
                b.add_tensor(
                    &format!("{p}.ssm_alpha.weight"),
                    GgmlType::Q8_0,
                    &[HID, nh],
                    bytes(ne),
                );
                b.add_tensor(
                    &format!("{p}.ssm_beta.weight"),
                    GgmlType::Q8_0,
                    &[HID, nh],
                    bytes(ne),
                );
            }
            other => panic!("unsupported test gate type {other:?}"),
        }
    }
    b.build()
}

/// Converts a GDN model and returns layer 0's `(ssm_alpha.quant, length)`
/// after the same planner-extent lockstep assertion as `convert_and_probe`.
fn convert_and_probe_gates(
    label: &str,
    gates: GgmlType,
    opts: &ConvertOptions,
) -> (QuantScheme, u64) {
    let gguf = build_gdn_model(gates);
    let out = std::env::temp_dir().join(format!(
        "lumen_lockstep_gates_{}_{}.lbc",
        label,
        std::process::id()
    ));
    convert_gguf_bytes_to_lbc(&gguf, &out, opts)
        .unwrap_or_else(|e| panic!("{label}: conversion failed (planner/writer desync?): {e:?}"));
    let f = LbcFile::open(&out).unwrap();
    for (li, idx) in f.layer_indices.iter().enumerate() {
        let mut extent = 0u64;
        for (_n, s) in idx.subtensors.named_slices() {
            extent = extent.max(s.offset + s.length);
        }
        assert_eq!(
            extent, idx.layer_length_bytes,
            "{label}: layer {li} planner extent != blob length"
        );
    }
    let alpha = f.layer_indices[0].subtensors.ssm_alpha.unwrap();
    let beta = f.layer_indices[0].subtensors.ssm_beta.unwrap();
    assert_eq!(
        (alpha.quant, alpha.length),
        (beta.quant, beta.length),
        "{label}: ssm_alpha and ssm_beta must be handled identically"
    );
    std::fs::remove_file(&out).ok();
    (alpha.quant, alpha.length)
}

#[test]
fn dequantize_gdn_gates_lockstep() {
    // Own-binary test, but shield the expectations from an inherited
    // LUMEN_CONVERT_SOURCE_FIDELITY in the invoking shell.
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");
    // 64x8 = 512 elems: Q8_0 = 16 blocks x 34 B; F32 = 512 x 4 B.
    let metal_deq = ConvertOptions {
        target: ConvertTarget::Metal,
        dequantize_to_f32: true,
        ..Default::default()
    };
    // Metal keeps the Q8_0 force under --dequantize (loadable output);
    // F32-source gates previously aborted here on a plan/writer mismatch.
    assert_eq!(
        convert_and_probe_gates("metal_deq_f32src", GgmlType::F32, &metal_deq),
        (QuantScheme::Q8_0, 16 * 34)
    );
    assert_eq!(
        convert_and_probe_gates("metal_deq_q8src", GgmlType::Q8_0, &metal_deq),
        (QuantScheme::Q8_0, 16 * 34)
    );
    let generic_deq = ConvertOptions {
        target: ConvertTarget::Generic,
        dequantize_to_f32: true,
        ..Default::default()
    };
    // Non-Metal targets honor --dequantize for the gates (F32 is servable).
    assert_eq!(
        convert_and_probe_gates("generic_deq_f32src", GgmlType::F32, &generic_deq),
        (QuantScheme::F32, 512 * 4)
    );
    assert_eq!(
        convert_and_probe_gates("generic_deq_q8src", GgmlType::Q8_0, &generic_deq),
        (QuantScheme::F32, 512 * 4)
    );
    // Default (no --dequantize) is unchanged: gates force to Q8_0 everywhere.
    let generic = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    assert_eq!(
        convert_and_probe_gates("generic_default_f32src", GgmlType::F32, &generic),
        (QuantScheme::Q8_0, 16 * 34)
    );
}
