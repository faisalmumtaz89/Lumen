//! End-to-end coverage of the Q6_K output-head fidelity gate.
//!
//! `LUMEN_CONVERT_SOURCE_FIDELITY` / `LUMEN_CONVERT_KEEP_Q6K_OUTPUT` are
//! process-global env vars read during conversion, so these combinations run
//! sequentially inside ONE test function in this dedicated integration
//! binary — its own process — where no parallel test can observe the
//! mutation.

use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const HID: u64 = 64;
const INTER: u64 = 128;
const VOCAB: u64 = 256;

fn q8(n: u64) -> Vec<u8> {
    let mut v = vec![0u8; (n / 32) as usize * 34];
    for b in v.chunks_exact_mut(34) {
        b[0] = 0x00;
        b[1] = 0x3C;
    }
    v
}

fn build_q6k_headed_model() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "qwen35");
    for (k, v) in [
        ("block_count", 1u32),
        ("attention.head_count", 8),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 8),
        ("embedding_length", 64),
        ("feed_forward_length", 128),
        ("context_length", 64),
    ] {
        b.add_u32(&format!("qwen35.{k}"), v);
    }
    b.add_f32("qwen35.rope.freq_base", 10000.0);
    b.add_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-5);
    b.add_f32_tensor(
        "token_embd.weight",
        &[VOCAB, HID],
        &vec![0.0; (VOCAB * HID) as usize],
    );
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    let n = (VOCAB * HID) as usize;
    b.add_tensor(
        "output.weight",
        GgmlType::Q6_K,
        &[HID, VOCAB],
        vec![0u8; n / 256 * 210],
    );
    let kvd = 32u64;
    for (nm, dims) in [
        ("attn_qkv.weight", [HID, HID]),
        ("attn_q.weight", [HID, HID]),
        ("attn_k.weight", [HID, kvd]),
        ("attn_v.weight", [HID, kvd]),
        ("attn_output.weight", [HID, HID]),
        ("ffn_gate.weight", [HID, INTER]),
        ("ffn_up.weight", [HID, INTER]),
        ("ffn_down.weight", [INTER, HID]),
    ] {
        let ne: u64 = dims.iter().product();
        b.add_tensor(&format!("blk.0.{nm}"), GgmlType::Q8_0, &dims, q8(ne));
    }
    b.add_f32_tensor("blk.0.attn_norm.weight", &[HID], &vec![1.0; HID as usize]);
    b.add_f32_tensor("blk.0.ffn_norm.weight", &[HID], &vec![1.0; HID as usize]);
    b.build()
}

fn head_quant(target: ConvertTarget) -> (QuantScheme, u64) {
    let gguf = build_q6k_headed_model();
    let out = std::env::temp_dir().join(format!("q6k_gate_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(&gguf, &out, &opts).unwrap();
    let f = LbcFile::open(&out).unwrap();
    let q = (f.header.output_proj.quant, f.header.output_proj.length);
    std::fs::remove_file(&out).ok();
    q
}

#[test]
fn q6k_head_fidelity_gate_matrix() {
    // A developer shell may carry these; the matrix requires a clean start.
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");
    std::env::remove_var("LUMEN_CONVERT_KEEP_Q6K_OUTPUT");

    // Default: requantized on every target.
    assert_eq!(
        head_quant(ConvertTarget::Generic),
        (QuantScheme::Q8_0, 17408)
    );
    assert_eq!(head_quant(ConvertTarget::Metal), (QuantScheme::Q8_0, 17408));

    // Fidelity keep: Generic preserves Q6_K; Metal still requantizes (it has
    // no Q6_K head kernel and would otherwise fall to the slow F32 path).
    std::env::set_var("LUMEN_CONVERT_SOURCE_FIDELITY", "1");
    assert_eq!(
        head_quant(ConvertTarget::Generic),
        (QuantScheme::Q6_K, 13440)
    );
    assert_eq!(head_quant(ConvertTarget::Metal), (QuantScheme::Q8_0, 17408));
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");

    // The narrower opt-in behaves identically.
    std::env::set_var("LUMEN_CONVERT_KEEP_Q6K_OUTPUT", "1");
    assert_eq!(
        head_quant(ConvertTarget::Generic),
        (QuantScheme::Q6_K, 13440)
    );
    assert_eq!(head_quant(ConvertTarget::Metal), (QuantScheme::Q8_0, 17408));
    std::env::remove_var("LUMEN_CONVERT_KEEP_Q6K_OUTPUT");
}
