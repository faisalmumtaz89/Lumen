//! The explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch keeps a Q5_K `ssm_out`, on a
//! source that is not a K-quant source, exactly as 0.31.0 kept it.
//!
//! `ssm_out_keeps_source` has two arms: the K-quant source default, whose two
//! conditions `kquant_ssm_out_geometry.rs` and `kquant_ssm_out_header.rs` cover, and
//! this switch, which answers for a source the default policy never touches. Without
//! the switch such a plane is requantised to the Q8_0 floor; with it the source bytes
//! are carried. Both halves run here, so the switch is what the difference is
//! attributed to.
//!
//! The pin was derived by building this same fixture against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and hashing the `ssm_out` plane of its artifact; the fixture GGUF hashes
//! the same on both trees, which is the cross-check that the transcription did not
//! drift.
//!
//! `LUMEN_CONVERT_SOURCE_FIDELITY` is process-global, so this binary has one test
//! function: no parallel test can observe the variable while it is set, whatever
//! `--test-threads` is, and the run without it comes first.
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const VOCAB: u64 = 256;
const HID: u64 = 256;
const INTER: u64 = 512;
const HEADS: u32 = 8;
const KVH: u32 = 4;
const STATE: u64 = 32;
const V_HEADS: u64 = 8;
const GROUPS: u64 = 2;
/// `ssm_out`'s row width: whole superblocks, so the plane is servable as stored and
/// the geometry rule is not what decides this case.
const V_DIM: u64 = V_HEADS * STATE;
// Four layers: the converter's layer kinds are positional (full attention at 3, 7, …).
const LAYERS: u32 = 4;

/// The source bytes of one plane, distinct per type and per superblock.
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
        GgmlType::Q5_K => {
            let mut v = vec![0u8; n / 256 * 176];
            for (b, blk) in v.chunks_exact_mut(176).enumerate() {
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

/// A four-layer qwen35 source that is NOT a K-quant source: every dense FFN projection
/// is Q8_0, so the K-quant source policy does not apply and the only thing that can
/// keep the Q5_K `ssm_out` is the explicit switch.
fn build() -> Vec<u8> {
    let qkv_rows: u64 = (2 * GROUPS + V_HEADS) * STATE;
    let kvd = (HID / HEADS as u64) * KVH as u64;
    let mut b = GgufBuilder::new();
    let k = |s: &str| format!("qwen35.{s}");
    b.add_string("general.architecture", "qwen35");
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), HID as u32 / HEADS);
    b.add_u32(&k("embedding_length"), HID as u32);
    b.add_u32(&k("feed_forward_length"), INTER as u32);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    b.add_u32(&k("ssm.time_step_rank"), V_HEADS as u32);
    b.add_u32(&k("ssm.group_count"), GROUPS as u32);
    b.add_u32(&k("ssm.state_size"), STATE as u32);
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
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let full = l == 3;
        let mut planes: Vec<(&str, [u64; 2], GgmlType)> = vec![
            ("attn_q.weight", [HID, HID], GgmlType::Q8_0),
            ("attn_k.weight", [HID, kvd], GgmlType::Q8_0),
            ("attn_v.weight", [HID, kvd], GgmlType::Q8_0),
            ("attn_output.weight", [HID, HID], GgmlType::Q8_0),
            ("ffn_gate.weight", [HID, INTER], GgmlType::Q8_0),
            ("ffn_up.weight", [HID, INTER], GgmlType::Q8_0),
            ("ffn_down.weight", [INTER, HID], GgmlType::Q8_0),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [HID, qkv_rows], GgmlType::Q8_0),
                ("attn_gate.weight", [HID, V_DIM], GgmlType::Q8_0),
                ("ssm_out.weight", [V_DIM, HID], GgmlType::Q5_K),
            ]);
        }
        for (nm, dims, t) in planes {
            let n: u64 = dims.iter().product();
            b.add_tensor(&format!("{p}.{nm}"), t, &dims, bytes_for(t, n));
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
        if full {
            continue;
        }
        b.add_f32_tensor(
            &format!("{p}.ssm_a"),
            &[V_HEADS],
            &vec![-0.5; V_HEADS as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_conv1d.weight"),
            &[4, qkv_rows],
            &vec![0.1; (4 * qkv_rows) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_dt.bias"),
            &[V_HEADS],
            &vec![0.0; V_HEADS as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_norm.weight"),
            &[STATE],
            &vec![1.0; STATE as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_alpha.weight"),
            &[HID, V_HEADS],
            &vec![0.02; (HID * V_HEADS) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_beta.weight"),
            &[HID, V_HEADS],
            &vec![0.02; (HID * V_HEADS) as usize],
        );
    }
    b.build()
}

/// (primary scheme, `ssm_out` scheme, `ssm_out` plane bytes) of layer 0 of the generic
/// artifact the fixture converts to under default options.
fn convert_ssm_out(label: &str) -> (QuantScheme, QuantScheme, Vec<u8>) {
    let out = std::env::temp_dir().join(format!("ssm_out_sf_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(&build(), &out, &opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    let f = LbcFile::open(&out).unwrap();
    let layer = &f.layer_indices[0];
    let slice = *layer
        .subtensors
        .ssm_out
        .as_ref()
        .expect("fixture layer 0 has an ssm_out");
    let start = (layer.layer_offset_bytes + slice.offset) as usize;
    let plane = bytes[start..start + slice.length as usize].to_vec();
    let primary = f.header.quantization.scheme;
    drop(f);
    std::fs::remove_file(&out).ok();
    (primary, slice.quant, plane)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 `ssm_out` plane of this fixture under the switch: the Q5_K source plane
/// carried verbatim, 45 056 bytes.
const KEPT_SSM_OUT_0_31_0: &str =
    "9bc4ca919ceaddd70091fb205d3edeb510542c7ad034b058915a1ac7955a481c";

#[test]
fn the_source_fidelity_switch_keeps_a_q5_k_ssm_out_off_a_non_kquant_source() {
    // Without the switch: the Q8_0 floor, as every non-K-quant source has always had.
    let (primary, quant, plane) = convert_ssm_out("default");
    assert_eq!(
        primary,
        QuantScheme::Q8_0,
        "fixture must not be a K-quant source, or the default arm answers instead"
    );
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a Q5_K ssm_out was kept without the switch"
    );
    assert_eq!(plane.len(), 69_632, "the requantised plane is 69 632 bytes");

    // With it: the source plane, byte for byte, as in 0.31.0.
    std::env::set_var("LUMEN_CONVERT_SOURCE_FIDELITY", "1");
    let kept = convert_ssm_out("fidelity");
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");
    let (primary, quant, plane) = kept;
    assert_eq!(primary, QuantScheme::Q8_0, "the header scheme is unchanged");
    assert_eq!(
        quant,
        QuantScheme::Q5_K,
        "the switch did not keep the Q5_K ssm_out"
    );
    assert_eq!(plane.len(), 45_056, "the kept plane is the source's length");
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q5_K, V_DIM * HID),
        "the kept ssm_out is not the source bytes"
    );
    assert_eq!(
        sha256_hex(&plane),
        KEPT_SSM_OUT_0_31_0,
        "the ssm_out this switch keeps moved from 0.31.0"
    );
}
