//! The K-quant source policy belongs to the dense converter, whose planner reads
//! the `ffn_gate` / `ffn_up` / `ffn_down` names the policy reads a scheme off.
//! The MoE planner reads `ffn_*_exps` and zeroes `w_gate` / `w_up` / `w_down`
//! (`arch/qwen35_moe.rs`), so a `qwen35moe` file that carries one of those dense
//! names is carrying a plane no planner reads: its artifact must be the one
//! 0.31.0 wrote, on both targets, exactly as the file without it.
//!
//! The pin was derived by building this same fixture against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and hashing each artifact.
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};

// A K-quant plane needs a 256-multiple input dimension (the converter's contract gate).
const HID: u64 = 256;
const INTER: u64 = 512;
const VOCAB: u64 = 256;
const QKV_ROWS: u64 = 384; // (2 * group_count + time_step_rank) * state_size = (4 + 8) * 32
const HEADS: u32 = 8;
const KVH: u32 = 4;
const NEXP: u64 = 2;
// Four layers: the converter's layer kinds are positional (full attention at 3, 7, …).
const LAYERS: u32 = 4;

/// `pure_q8_0`-style bytes for the two source types this file builds.
fn bytes_for(t: GgmlType, n: u64) -> Vec<u8> {
    let n = n as usize;
    match t {
        GgmlType::Q8_0 => {
            let mut v = vec![0u8; n / 32 * 34];
            for b in v.chunks_exact_mut(34) {
                b[1] = 0x3C;
            }
            v
        }
        GgmlType::Q4_K => {
            let mut v = vec![0u8; n / 256 * 144];
            for (b, blk) in v.chunks_exact_mut(144).enumerate() {
                blk[0..2].copy_from_slice(&(0x3C00u16 + (b as u16 & 0xFF)).to_le_bytes());
                blk[2..4].copy_from_slice(&(0x3800u16 + (b as u16 & 0x7F)).to_le_bytes());
                for (i, q) in blk[16..].iter_mut().enumerate() {
                    *q = ((i * 31 + b * 7) & 0xFF) as u8;
                }
            }
            v
        }
        other => panic!("fixture: unsupported type {other:?}"),
    }
}

/// A four-layer `qwen35moe` model — stacked experts, a router, the GDN set on the
/// linear-attention layers — with every quantised plane Q8_0, plus `extra`.
fn build_moe(extra: &[(&str, GgmlType, [u64; 2])]) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    let k = |s: &str| format!("qwen35moe.{s}");
    b.add_string("general.architecture", "qwen35moe");
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), HID as u32 / HEADS);
    b.add_u32(&k("embedding_length"), HID as u32);
    b.add_u32(&k("feed_forward_length"), INTER as u32);
    b.add_u32(&k("expert_feed_forward_length"), INTER as u32);
    b.add_u32(&k("expert_count"), NEXP as u32);
    b.add_u32(&k("expert_used_count"), 2);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    b.add_u32(&k("ssm.time_step_rank"), 8);
    b.add_u32(&k("ssm.group_count"), 2);
    b.add_u32(&k("ssm.state_size"), 32);
    b.add_u32(&k("ssm.conv_kernel"), 4);
    let ne = VOCAB * HID;
    b.add_tensor(
        "token_embd.weight",
        GgmlType::Q8_0,
        &[VOCAB, HID],
        bytes_for(GgmlType::Q8_0, ne),
    );
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    b.add_tensor(
        "output.weight",
        GgmlType::Q8_0,
        &[HID, VOCAB],
        bytes_for(GgmlType::Q8_0, ne),
    );
    let kvd = (HID / HEADS as u64) * KVH as u64;
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let full = l == 3;
        let mut planes: Vec<(&str, Vec<u64>)> = vec![
            ("ffn_gate_exps.weight", vec![HID, INTER, NEXP]),
            ("ffn_up_exps.weight", vec![HID, INTER, NEXP]),
            ("ffn_down_exps.weight", vec![INTER, HID, NEXP]),
        ];
        if full {
            planes.extend([
                ("attn_q.weight", vec![HID, HID]),
                ("attn_k.weight", vec![HID, kvd]),
                ("attn_v.weight", vec![HID, kvd]),
                ("attn_output.weight", vec![HID, HID]),
            ]);
        } else {
            planes.extend([
                ("attn_qkv.weight", vec![HID, QKV_ROWS]),
                ("attn_gate.weight", vec![HID, HID]),
                ("ssm_out.weight", vec![HID, HID]),
            ]);
        }
        for (nm, dims) in planes {
            let n: u64 = dims.iter().product();
            b.add_tensor(
                &format!("{p}.{nm}"),
                GgmlType::Q8_0,
                &dims,
                bytes_for(GgmlType::Q8_0, n),
            );
        }
        b.add_f32_tensor(
            &format!("{p}.ffn_gate_inp.weight"),
            &[HID, NEXP],
            &vec![0.0; (HID * NEXP) as usize],
        );
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
        if full {
            continue;
        }
        let nh = 8u64;
        b.add_f32_tensor(&format!("{p}.ssm_a"), &[nh], &vec![-0.5; nh as usize]);
        b.add_f32_tensor(
            &format!("{p}.ssm_conv1d.weight"),
            &[4, QKV_ROWS],
            &vec![0.1; (4 * QKV_ROWS) as usize],
        );
        b.add_f32_tensor(&format!("{p}.ssm_dt.bias"), &[nh], &vec![0.0; nh as usize]);
        b.add_f32_tensor(
            &format!("{p}.ssm_norm.weight"),
            &[HID / nh],
            &vec![1.0; (HID / nh) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_alpha.weight"),
            &[HID, nh],
            &vec![0.02; (HID * nh) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_beta.weight"),
            &[HID, nh],
            &vec![0.02; (HID * nh) as usize],
        );
    }
    for (nm, t, dims) in extra {
        let n: u64 = dims.iter().product();
        b.add_tensor(nm, *t, dims, bytes_for(*t, n));
    }
    b.build()
}

/// The artifact `gguf` converts to for `target`.
fn convert(label: &str, gguf: &[u8], target: ConvertTarget) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!("kquant_arch_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(gguf, &out, &opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    std::fs::remove_file(&out).ok();
    bytes
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 artifact digest of the MoE fixture, the same on both targets and with
/// or without the dense `ffn_gate` no MoE plan reads.
const MOE_Q8_0_0_31_0: &str = "107f18a478e39c17d41bd27be643ca3d8d4212b7715b9e07dd467f3130c3c17b";

#[test]
fn moe_source_is_0_31_0_with_an_unread_dense_ffn_tensor() {
    let plain = build_moe(&[]);
    // The same file plus a Q4_K `blk.1.ffn_gate.weight`: a canonical name the MoE
    // planner never looks up, so no plane of the artifact may change — not the
    // header's scheme, and not the `ssm_alpha` / `ssm_beta` gates the
    // source-fidelity policy would keep as F32.
    let unread = build_moe(&[("blk.1.ffn_gate.weight", GgmlType::Q4_K, [HID, INTER])]);
    for (target, label) in [
        (ConvertTarget::Generic, "generic"),
        (ConvertTarget::Metal, "metal"),
    ] {
        let a = convert(&format!("moe_{label}"), &plain, target);
        let b = convert(&format!("moe_unread_{label}"), &unread, target);
        assert_eq!(
            sha256_hex(&a),
            MOE_Q8_0_0_31_0,
            "the MoE source on the {label} target moved from 0.31.0"
        );
        assert_eq!(
            sha256_hex(&b),
            MOE_Q8_0_0_31_0,
            "an unread dense ffn_gate moved the MoE source's {label} artifact from 0.31.0"
        );
    }
}
