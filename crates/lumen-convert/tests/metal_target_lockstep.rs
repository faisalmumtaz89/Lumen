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
