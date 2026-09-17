//! What a K-quant source preserves by default has to be a plane the runtime serves.
//! The GDN output projection is read at `gdn_v_dim` — `hidden` rows of
//! `gdn_v_dim / block_elems` blocks — so an `ssm_out` whose row width is not whole
//! blocks for its scheme is rejected by
//! `lumen_format::serving_rules::validate_projection_geometry`, which the converter
//! runs over the planned layers before it writes a byte. The K-quant source policy
//! plans such a plane as 0.31.0 planned it — requantised to Q8_0, or a stored Q8_0
//! unchanged, which that gate then refuses either way at a width that is not whole
//! 32-element blocks — instead of carrying it. The explicit
//! `LUMEN_CONVERT_SOURCE_FIDELITY` switch is outside this rule and unchanged, so these
//! fixtures set no environment.
//!
//! GGUF sizes a tensor from its flattened element count, so a K-quant plane whose row
//! is narrower than a superblock is a file the converter reads without complaint —
//! the source of the first fixture. `ssm_out`'s width is the GDN V dimension (4096 on
//! the 9B), and no K-quant tensor of a standard export has a row length that is not a
//! multiple of the 256-element superblock, so no shipped file converts differently;
//! the pin is the guard against the default widening to one that does.
//!
//! The pin was derived by building this same fixture against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and hashing the `ssm_out` plane of its artifact; the fixture GGUF hashes
//! the same on both trees, which is the cross-check that the transcription did not
//! drift. The pin is on the plane, not on the whole artifact: a K-quant source's
//! header carries its own primary scheme, so its artifact is not byte-identical to
//! 0.31.0's by design.
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const VOCAB: u64 = 256;
const HEADS: u32 = 8;
const KVH: u32 = 4;
const STATE: u64 = 32;
const GROUPS: u64 = 2;
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

/// A four-layer qwen35 K-quant source: Q8_0 everywhere except the Q4_K `ffn_down`
/// that makes it one (its in_dim is `2 * hid`, so the layer contract gate passes at
/// either width) and the Q4_K `ssm_out` under test. `v_heads` sets the GDN V
/// dimension `v_heads * STATE`, which is the width `ssm_out` is read at and the
/// property under test; `hid` is the row count.
fn build(hid: u64, v_heads: u64) -> Vec<u8> {
    let inter: u64 = 2 * hid;
    let v_dim: u64 = v_heads * STATE;
    let qkv_rows: u64 = (2 * GROUPS + v_heads) * STATE;
    let kvd = (hid / HEADS as u64) * KVH as u64;
    let mut b = GgufBuilder::new();
    let k = |s: &str| format!("qwen35.{s}");
    b.add_string("general.architecture", "qwen35");
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), hid as u32 / HEADS);
    b.add_u32(&k("embedding_length"), hid as u32);
    b.add_u32(&k("feed_forward_length"), inter as u32);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    b.add_u32(&k("ssm.time_step_rank"), v_heads as u32);
    b.add_u32(&k("ssm.group_count"), GROUPS as u32);
    b.add_u32(&k("ssm.state_size"), STATE as u32);
    b.add_u32(&k("ssm.conv_kernel"), 4);
    let ne = VOCAB * hid;
    b.add_tensor(
        "token_embd.weight",
        GgmlType::Q8_0,
        &[VOCAB, hid],
        bytes_for(GgmlType::Q8_0, ne),
    );
    b.add_f32_tensor("output_norm.weight", &[hid], &vec![1.0; hid as usize]);
    b.add_tensor(
        "output.weight",
        GgmlType::Q8_0,
        &[hid, VOCAB],
        bytes_for(GgmlType::Q8_0, ne),
    );
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let full = l == 3;
        let mut planes: Vec<(&str, [u64; 2], GgmlType)> = vec![
            ("attn_q.weight", [hid, hid], GgmlType::Q8_0),
            ("attn_k.weight", [hid, kvd], GgmlType::Q8_0),
            ("attn_v.weight", [hid, kvd], GgmlType::Q8_0),
            ("attn_output.weight", [hid, hid], GgmlType::Q8_0),
            ("ffn_gate.weight", [hid, inter], GgmlType::Q8_0),
            ("ffn_up.weight", [hid, inter], GgmlType::Q8_0),
            ("ffn_down.weight", [inter, hid], GgmlType::Q4_K),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [hid, qkv_rows], GgmlType::Q8_0),
                ("attn_gate.weight", [hid, v_dim], GgmlType::Q8_0),
                ("ssm_out.weight", [v_dim, hid], GgmlType::Q4_K),
            ]);
        }
        for (nm, dims, t) in planes {
            let n: u64 = dims.iter().product();
            b.add_tensor(&format!("{p}.{nm}"), t, &dims, bytes_for(t, n));
        }
        b.add_f32_tensor(
            &format!("{p}.attn_norm.weight"),
            &[hid],
            &vec![1.0; hid as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ffn_norm.weight"),
            &[hid],
            &vec![1.0; hid as usize],
        );
        if full {
            continue;
        }
        b.add_f32_tensor(
            &format!("{p}.ssm_a"),
            &[v_heads],
            &vec![-0.5; v_heads as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_conv1d.weight"),
            &[4, qkv_rows],
            &vec![0.1; (4 * qkv_rows) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_dt.bias"),
            &[v_heads],
            &vec![0.0; v_heads as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_norm.weight"),
            &[STATE],
            &vec![1.0; STATE as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_alpha.weight"),
            &[hid, v_heads],
            &vec![0.02; (hid * v_heads) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_beta.weight"),
            &[hid, v_heads],
            &vec![0.02; (hid * v_heads) as usize],
        );
    }
    b.build()
}

/// (version, primary scheme, `ssm_out` scheme, `ssm_out` plane bytes) of layer 0 of
/// the generic artifact `gguf` converts to under default options.
fn convert_ssm_out(label: &str, gguf: &[u8]) -> (u32, QuantScheme, QuantScheme, Vec<u8>) {
    let out = std::env::temp_dir().join(format!("kquant_ssm_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(gguf, &out, &opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
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
    let version = f.header.version;
    let primary = f.header.quantization.scheme;
    drop(f);
    std::fs::remove_file(&out).ok();
    (version, primary, slice.quant, plane)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 `ssm_out` plane of the narrow fixture: the Q4_K source plane
/// requantised to Q8_0, 17 408 bytes.
const NARROW_SSM_OUT_0_31_0: &str =
    "3f1f6f76c52276c865bae097486a0ce164cd509c98c6410b677f516084ad7c3c";

#[test]
fn a_kquant_ssm_out_narrower_than_a_superblock_keeps_the_0_31_0_plane() {
    let (version, primary, quant, plane) = convert_ssm_out("narrow", &build(128, 128 / STATE));
    // The fixture is a K-quant source, so the policy really is the one under test.
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(version, 4, "the fixture's Q8_0 embedding is not a v5 plane");
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "an ssm_out whose rows are narrower than a superblock was carried verbatim"
    );
    assert_eq!(plane.len(), 17_408, "the 0.31.0 plane is 17 408 bytes");
    assert_eq!(
        sha256_hex(&plane),
        NARROW_SSM_OUT_0_31_0,
        "the ssm_out plane moved from 0.31.0"
    );
    // And what the converter wrote is what the loader accepts.
    lumen_format::serving_rules::validate_projection_row_width("ssm_out", quant, 128)
        .expect("the written ssm_out must pass the loader's own row-width rule");
}

#[test]
fn a_kquant_ssm_out_of_whole_superblock_rows_is_still_carried_verbatim() {
    let (_, primary, quant, plane) = convert_ssm_out("wide", &build(256, 256 / STATE));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::Q4_K,
        "the Q4_K ssm_out was not preserved"
    );
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q4_K, 256 * 256),
        "the preserved ssm_out is not the source bytes"
    );
    lumen_format::serving_rules::validate_projection_row_width("ssm_out", quant, 256)
        .expect("the written ssm_out must pass the loader's own row-width rule");
}

/// The width the rule is asked about is the GDN V dimension, not `hidden`: this
/// fixture is 128 wide with a 256-wide `ssm_out` (8 V heads of 32), the geometry the
/// loader reads the plane at, and the plane is servable there.
#[test]
fn the_kept_width_is_the_gdn_v_dimension_not_the_hidden_dimension() {
    let (_, primary, quant, plane) = convert_ssm_out("v_dim", &build(128, 8));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::Q4_K,
        "an ssm_out servable at its own width was requantised"
    );
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q4_K, 256 * 128),
        "the preserved ssm_out is not the source bytes"
    );
}
