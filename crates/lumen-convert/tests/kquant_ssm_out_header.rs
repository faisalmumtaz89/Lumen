//! What a K-quant source preserves by default has to be servable on the artifact the
//! conversion actually writes — at its geometry (`kquant_ssm_out_geometry.rs`) and
//! under its header. `--requant q8_0`, `--requant q4_0` and `--dequantize` stamp a
//! Q8_0 / Q4_0 / F32 primary scheme, and CUDA's Q4_K / Q5_K / Q6_K layer arms are
//! scoped on a K-quant header (`runtime_defaults::kquant_artifact`), so a K-quant
//! `ssm_out` kept under one of those headers is dequantised to F32 at load — 4 bytes
//! per weight, against the 34 bytes per 32 weights of the Q8_0 plane 0.31.0 wrote
//! there, on every GDN layer. The default therefore keeps nothing on those
//! three routes: each writes the `ssm_out` 0.31.0 wrote for it. The other
//! preserved planes go by arms that read a plane's stored scheme whatever
//! the header is: a kept embedding survives both `--requant` schemes, while
//! `--dequantize` dequantises it; a preserved Q6_K head survives `--requant q8_0`
//! and `--dequantize`, while `--requant q4_0` requantises it as 0.31.0 did.
//!
//! The pins were derived by building this same fixture against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and hashing the `ssm_out` plane of its artifact; the fixture GGUF hashes
//! the same on both trees, which is the cross-check that the transcription did not
//! drift. The pins are on the plane, not on the whole artifact: 0.31.0 requantised
//! this source's F32 GDN gates to Q8_0 unless `--dequantize` was given, and the
//! branch keeps them F32, so its two `--requant` artifacts differ from 0.31.0's in
//! `ssm_alpha` / `ssm_beta` alone, its default artifact differs in its header scheme
//! and `ssm_out` as well, and only its `--dequantize` artifact is byte-identical.
//! The embedding and head are Q8_0 here and match 0.31.0's on every route.
//!
//! The explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch is outside this rule and
//! unchanged — it keeps only the Q5_K and Q8_0 `ssm_out` 0.31.0 kept, both of which
//! CUDA serves whatever the header is — so these fixtures set no environment.
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
/// `ssm_out`'s row width: whole superblocks, so the plane is servable at its geometry
/// and the header is what decides this case.
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
/// that makes it one and the Q4_K `ssm_out` under test, whose row width is the whole
/// superblocks the loader reads.
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
            ("ffn_down.weight", [INTER, HID], GgmlType::Q4_K),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [HID, qkv_rows], GgmlType::Q8_0),
                ("attn_gate.weight", [HID, V_DIM], GgmlType::Q8_0),
                ("ssm_out.weight", [V_DIM, HID], GgmlType::Q4_K),
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
/// artifact the fixture converts to under `opts`.
fn convert_ssm_out(label: &str, opts: &ConvertOptions) -> (QuantScheme, QuantScheme, Vec<u8>) {
    let out =
        std::env::temp_dir().join(format!("kquant_ssm_hdr_{label}_{}.lbc", std::process::id()));
    convert_gguf_bytes_to_lbc(&build(), &out, opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
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

fn generic(requant_to: Option<QuantScheme>, dequantize_to_f32: bool) -> ConvertOptions {
    ConvertOptions {
        target: ConvertTarget::Generic,
        requant_to,
        dequantize_to_f32,
        ..Default::default()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 `ssm_out` plane of this fixture: the Q4_K source plane requantised to
/// the Q8_0 floor, 69 632 bytes. One digest for all three routes — 0.31.0 wrote the
/// same floored plane under `--requant q8_0`, `--requant q4_0` (the floor pre-empts
/// the 4-bit target) and `--dequantize` (`ssm_out` is never dequantised).
const SSM_OUT_0_31_0: &str = "b5822048397cfb7e72443fbe3f37fb07bbd0b397c11177cfd71cf768d44e924a";

#[test]
fn the_default_keep_is_off_under_requant_q8_0() {
    let (primary, quant, plane) =
        convert_ssm_out("requant_q8_0", &generic(Some(QuantScheme::Q8_0), false));
    assert_eq!(
        primary,
        QuantScheme::Q8_0,
        "the header is the requant target"
    );
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a K-quant ssm_out was kept under a header CUDA's K-quant arms are closed on"
    );
    assert_eq!(plane.len(), 69_632, "the 0.31.0 plane is 69 632 bytes");
    assert_eq!(
        sha256_hex(&plane),
        SSM_OUT_0_31_0,
        "the ssm_out plane moved from 0.31.0"
    );
}

#[test]
fn the_default_keep_is_off_under_requant_q4_0() {
    let (primary, quant, plane) =
        convert_ssm_out("requant_q4_0", &generic(Some(QuantScheme::Q4_0), false));
    assert_eq!(
        primary,
        QuantScheme::Q4_0,
        "the header is the requant target"
    );
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a K-quant ssm_out was kept under a header CUDA's K-quant arms are closed on"
    );
    assert_eq!(
        sha256_hex(&plane),
        SSM_OUT_0_31_0,
        "the ssm_out plane moved from 0.31.0"
    );
}

#[test]
fn the_default_keep_is_off_under_dequantize() {
    let (primary, quant, plane) = convert_ssm_out("dequantize", &generic(None, true));
    assert_eq!(primary, QuantScheme::F32, "the header is F32");
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a K-quant ssm_out was kept under a header CUDA's K-quant arms are closed on"
    );
    assert_eq!(
        sha256_hex(&plane),
        SSM_OUT_0_31_0,
        "the ssm_out plane moved from 0.31.0"
    );
}

/// The control: the same fixture on the route that does stamp a K-quant header keeps
/// the plane as stored, so the three assertions above are the header condition and not
/// the policy switched off.
#[test]
fn the_default_keep_is_on_without_either_flag() {
    let (primary, quant, plane) = convert_ssm_out("default", &generic(None, false));
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
        bytes_for(GgmlType::Q4_K, V_DIM * HID),
        "the preserved ssm_out is not the source bytes"
    );
}
