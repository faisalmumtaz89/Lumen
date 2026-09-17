//! The K-quant source policy: a file whose planned dense FFN projections are
//! Q4_K/Q5_K/Q6_K is converted under the source-fidelity policy by default on the
//! generic target, carries every Q4_K / Q5_K / Q6_K plane as stored (layers,
//! `ssm_out`, embedding, and a Q6_K head — the one head scheme the converter
//! preserves; a Q4_K/Q5_K head is requantised as before), takes its K-quant scheme in
//! the header, and is written at `LBC_VERSION_KQUANT_EMBEDDING` when its embedding is
//! an as-stored K-quant plane (fixtures F and H keep version 4: their embedding is
//! Q8_0, and so does J: it ties its head to the embedding, so the embedding is
//! dequantised and the tied head is not a K-quant one); a file that is not a K-quant
//! source — the shipping Q4_0 shape with its Q5_K `ssm_out`, Q6_K head and Q6_K
//! full-attention `attn_q`, a GDN pair stored as K-quant, a K-quant embedding
//! alone, or a K-quant `blk.` tensor no planner lookup resolves to —
//! converts exactly as before.
//!
//! The Metal target is untouched by the policy: it has no K-quant kernel, so it upcasts
//! or re-quantises every K-quant plane exactly as in 0.31.0. `metal_target_is_0_31_0`
//! pins that for every fixture in this file by artifact digest.
//!
//! `LUMEN_CONVERT_SOURCE_FIDELITY` is process-global, so this binary has one test
//! function; the scenario that sets the variable runs last in it.
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;
use std::collections::HashMap;

// K-quant planes need a 256-multiple input dimension (the converter's contract gate).
const HID: u64 = 256;
const INTER: u64 = 512;
const VOCAB: u64 = 256;
const QKV_ROWS: u64 = 384; // (2 * group_count + time_step_rank) * state_size = (4 + 8) * 32
const HEADS: u32 = 8;
const KVH: u32 = 4;
// Four layers: the converter's layer kinds are positional (full attention at 3, 7, …),
// so layer 3 is the full-attention layer whose `attn_q` is planned.
const LAYERS: u32 = 4;

fn bytes_for(t: GgmlType, n: u64) -> Vec<u8> {
    let n = n as usize;
    match t {
        GgmlType::Q4_0 => vec![0u8; n / 32 * 18],
        GgmlType::Q4_1 => vec![0u8; n / 32 * 20],
        GgmlType::Q8_0 => {
            let mut v = vec![0u8; n / 32 * 34];
            for b in v.chunks_exact_mut(34) {
                b[1] = 0x3C;
            }
            v
        }
        // K-quant superblocks keyed on their index: a pattern over the quant bytes,
        // the packed 6-bit scales and mins left at zero, the f16 scales per superblock
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let bb = if t == GgmlType::Q4_K { 144 } else { 176 };
            let mut v = vec![0u8; n / 256 * bb];
            for (b, blk) in v.chunks_exact_mut(bb).enumerate() {
                blk[0..2].copy_from_slice(&(0x3C00u16 + (b as u16 & 0xFF)).to_le_bytes());
                blk[2..4].copy_from_slice(&(0x3800u16 + (b as u16 & 0x7F)).to_le_bytes());
                for (i, q) in blk[16..].iter_mut().enumerate() {
                    *q = ((i * 31 + b * 7) & 0xFF) as u8;
                }
            }
            v
        }
        GgmlType::Q6_K => {
            let mut v = vec![0u8; n / 256 * 210];
            for (b, blk) in v.chunks_exact_mut(210).enumerate() {
                for (i, q) in blk[..208].iter_mut().enumerate() {
                    *q = ((i * 13 + b * 5) & 0xFF) as u8;
                }
                blk[208..210].copy_from_slice(&(0x3C00u16 + (b as u16 & 0xFF)).to_le_bytes());
            }
            v
        }
        GgmlType::Q3_K => vec![0u8; n / 256 * 110],
        GgmlType::F32 => vec![0u8; n * 4],
        other => panic!("fixture: unsupported type {other:?}"),
    }
}

/// A four-layer qwen35 model whose weight types come from `ty(layer, suffix)`;
/// the embedding and the head take `embd` and `head`; the GDN gates are F32.
fn build(embd: GgmlType, head: GgmlType, ty: impl Fn(u32, &str) -> GgmlType) -> Vec<u8> {
    build_with_head(embd, Some(head), ty, &[])
}

/// [`build`] with no `output.weight`: the source ties its head to the embedding, and
/// the converter's weight-tying path gives the head the embedding's storage and scheme.
fn build_tied(embd: GgmlType, ty: impl Fn(u32, &str) -> GgmlType) -> Vec<u8> {
    build_with_head(embd, None, ty, &[])
}

/// [`build`] plus `extra` tensors appended after the layers: tensors no planner lookup
/// resolves to — a non-canonical layer spelling, or a duplicate of a canonical name
/// after the tensor `find_tensor` returns — so the artifact must not move.
fn build_with_extra(
    embd: GgmlType,
    head: GgmlType,
    ty: impl Fn(u32, &str) -> GgmlType,
    extra: &[(&str, GgmlType, [u64; 2])],
) -> Vec<u8> {
    build_with_head(embd, Some(head), ty, extra)
}

fn build_with_head(
    embd: GgmlType,
    head: Option<GgmlType>,
    ty: impl Fn(u32, &str) -> GgmlType,
    extra: &[(&str, GgmlType, [u64; 2])],
) -> Vec<u8> {
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
    b.add_u32(&k("ssm.time_step_rank"), 8);
    b.add_u32(&k("ssm.group_count"), 2);
    b.add_u32(&k("ssm.state_size"), 32);
    b.add_u32(&k("ssm.conv_kernel"), 4);
    let ne = VOCAB * HID;
    if embd == GgmlType::F32 {
        b.add_f32_tensor("token_embd.weight", &[VOCAB, HID], &vec![0.0; ne as usize]);
    } else {
        b.add_tensor(
            "token_embd.weight",
            embd,
            &[VOCAB, HID],
            bytes_for(embd, ne),
        );
    }
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    if let Some(head) = head {
        b.add_tensor("output.weight", head, &[HID, VOCAB], bytes_for(head, ne));
    }
    let kvd = (HID / HEADS as u64) * KVH as u64;
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        // Layer 3 is the full-attention layer: attention projections and no GDN tensors,
        // as in the real files; the GDN layers carry the fused QKV, the gate and the SSM set.
        let full = l == 3;
        let mut planes: Vec<(&str, [u64; 2])> = vec![
            ("attn_q.weight", [HID, HID]),
            ("attn_k.weight", [HID, kvd]),
            ("attn_v.weight", [HID, kvd]),
            ("attn_output.weight", [HID, HID]),
            ("ffn_gate.weight", [HID, INTER]),
            ("ffn_up.weight", [HID, INTER]),
            ("ffn_down.weight", [INTER, HID]),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [HID, QKV_ROWS]),
                ("attn_gate.weight", [HID, HID]),
                ("ssm_out.weight", [HID, HID]),
            ]);
        }
        for (nm, dims) in planes {
            let t = ty(l, nm);
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

/// Bytes an LBC plane of `quant` holds for `n` elements.
fn lbc_len(quant: QuantScheme, n: u64) -> u64 {
    match quant {
        QuantScheme::F32 => n * 4,
        QuantScheme::Q4_0 => n / 32 * 18,
        QuantScheme::Q4_1 => n / 32 * 20,
        QuantScheme::Q8_0 => n / 32 * 34,
        QuantScheme::Q4_K => n / 256 * 144,
        QuantScheme::Q5_K => n / 256 * 176,
        QuantScheme::Q6_K => n / 256 * 210,
        other => panic!("unexpected LBC scheme {other:?}"),
    }
}

/// The stored plane at `plane` (scheme, offset, length) carries the source's bytes.
fn assert_source(what: &str, p: &Probe, plane: (QuantScheme, u64, u64), src: GgmlType, n: u64) {
    let stored = &p.bytes[plane.1 as usize..(plane.1 + plane.2) as usize];
    let expected = bytes_for(src, n);
    assert!(
        stored == &expected[..],
        "{what}: the stored bytes are not the source bytes"
    );
}

struct Probe {
    bytes: Vec<u8>,
    version: u32,
    primary: QuantScheme,
    embedding: (QuantScheme, u64),
    head: (QuantScheme, u64),
    layers: Vec<HashMap<String, (QuantScheme, u64)>>,
    /// (scheme, absolute file offset, length) of every plane, by layer and slot
    planes: Vec<HashMap<String, (QuantScheme, u64, u64)>>,
    embedding_plane: (QuantScheme, u64, u64),
    head_plane: (QuantScheme, u64, u64),
}

/// The artifact bytes `gguf` converts to for `target`, or the converter's refusal.
fn try_convert(
    label: &str,
    gguf: &[u8],
    target: ConvertTarget,
) -> Result<Vec<u8>, lumen_convert::convert::ConvertError> {
    let out =
        std::env::temp_dir().join(format!("kquant_policy_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target,
        ..Default::default()
    };
    let r = convert_gguf_bytes_to_lbc(gguf, &out, &opts).map(|_| std::fs::read(&out).unwrap());
    std::fs::remove_file(&out).ok();
    r
}

/// The artifact `gguf` converts to under `opts`.
fn convert_with(label: &str, gguf: &[u8], opts: &ConvertOptions) -> Probe {
    let out =
        std::env::temp_dir().join(format!("kquant_policy_{label}_{}.lbc", std::process::id()));
    convert_gguf_bytes_to_lbc(gguf, &out, opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    std::fs::remove_file(&out).ok();
    probe(label, &bytes)
}

fn convert(label: &str, gguf: &[u8], target: ConvertTarget) -> Probe {
    let bytes = try_convert(label, gguf, target).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    probe(label, &bytes)
}

/// Every scheme, length, plane offset and the bytes of the artifact `bytes`.
fn probe(label: &str, bytes: &[u8]) -> Probe {
    let out = std::env::temp_dir().join(format!(
        "kquant_policy_probe_{label}_{}.lbc",
        std::process::id()
    ));
    std::fs::write(&out, bytes).unwrap();
    let f = LbcFile::open(&out).unwrap();
    std::fs::remove_file(&out).ok();
    let planes: Vec<HashMap<String, (QuantScheme, u64, u64)>> = f
        .layer_indices
        .iter()
        .map(|idx| {
            idx.subtensors
                .named_slices()
                .into_iter()
                .map(|(n, s)| {
                    (
                        n.to_string(),
                        (s.quant, idx.layer_offset_bytes + s.offset, s.length),
                    )
                })
                .collect::<HashMap<String, (QuantScheme, u64, u64)>>()
        })
        .collect();
    let layers = planes
        .iter()
        .map(|m| m.iter().map(|(n, p)| (n.clone(), (p.0, p.2))).collect())
        .collect();
    Probe {
        version: f.header.version,
        bytes: bytes.to_vec(),
        primary: f.header.quantization.scheme,
        embedding: (f.header.embedding.quant, f.header.embedding.length),
        head: (f.header.output_proj.quant, f.header.output_proj.length),
        layers,
        planes,
        embedding_plane: (
            f.header.embedding.quant,
            f.header.embedding.offset,
            f.header.embedding.length,
        ),
        head_plane: (
            f.header.output_proj.quant,
            f.header.output_proj.offset,
            f.header.output_proj.length,
        ),
    }
}

fn slice(p: &Probe, layer: usize, name: &str) -> (QuantScheme, u64) {
    *p.layers[layer]
        .get(name)
        .unwrap_or_else(|| panic!("layer {layer} has no {name}: {:?}", p.layers[layer].keys()))
}

/// Every source fixture this file converts, by name. One table, so the policy
/// assertions below and the two digest gates cover the same set of sources.
fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let mut v: Vec<(&'static str, Vec<u8>)> = vec![
        // A. The shipping Q4_0 shape: Q5_K ssm_out, Q6_K head, a Q6_K attn_q in the
        // full-attention layer, one Q4_1 ffn_down, every FFN projection Q4_0.
        (
            "shipping_q4_0",
            build(GgmlType::Q4_0, GgmlType::Q6_K, |l, nm| match (l, nm) {
                (_, "ssm_out.weight") => GgmlType::Q5_K,
                (3, "attn_q.weight") => GgmlType::Q6_K,
                (3, "ffn_down.weight") => GgmlType::Q4_1,
                _ => GgmlType::Q4_0,
            }),
        ),
        // B. Only the GDN pair is K-quant.
        (
            "kquant_pair",
            build(GgmlType::Q8_0, GgmlType::Q8_0, |_, nm| match nm {
                "attn_qkv.weight" | "attn_gate.weight" => GgmlType::Q4_K,
                _ => GgmlType::Q8_0,
            }),
        ),
        // D. A K-quant source: a Q4_K FFN pair and ssm_out on layer 0, a Q6_K ssm_out
        // on layer 1, a Q5_K head, a Q6_K embedding, one Q4_1 ffn_down.
        (
            "kquant_src",
            build(GgmlType::Q6_K, GgmlType::Q5_K, |l, nm| match (l, nm) {
                (0, "ffn_up.weight") | (0, "ffn_gate.weight") => GgmlType::Q4_K,
                (0, "ssm_out.weight") => GgmlType::Q4_K,
                (1, "ssm_out.weight") => GgmlType::Q6_K,
                (3, "ffn_down.weight") => GgmlType::Q4_1,
                _ => GgmlType::Q8_0,
            }),
        ),
        // E. A Q5_K-dominant K-quant source with a Q6_K head and ssm_out.
        (
            "q5_dominant",
            build(GgmlType::Q5_K, GgmlType::Q6_K, |_, nm| match nm {
                "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => GgmlType::Q5_K,
                "ssm_out.weight" => GgmlType::Q6_K,
                _ => GgmlType::Q8_0,
            }),
        ),
        // F. A K-quant source whose GDN pair mixes a Q4_K attn_qkv with a Q8_0 gate
        // (layer 0) and carries a K-quant pair (layer 1).
        (
            "kquant_mixed_gdn_pair",
            build(GgmlType::Q8_0, GgmlType::Q8_0, |l, nm| match (l, nm) {
                (_, "ffn_up.weight") | (_, "ffn_gate.weight") => GgmlType::Q4_K,
                (0, "attn_qkv.weight") => GgmlType::Q4_K,
                (1, "attn_qkv.weight") | (1, "attn_gate.weight") => GgmlType::Q4_K,
                _ => GgmlType::Q8_0,
            }),
        ),
        // G. The Q3_K_M shape: Q3_K gate and up, Q4_K down.
        (
            "q3_k_gate_up",
            build(GgmlType::Q8_0, GgmlType::Q8_0, |_, nm| match nm {
                "ffn_gate.weight" | "ffn_up.weight" => GgmlType::Q3_K,
                "ffn_down.weight" => GgmlType::Q4_K,
                _ => GgmlType::Q8_0,
            }),
        ),
        // H. A K-quant source whose GDN pair mixes a Q4_0 attn_qkv with a K-quant gate.
        (
            "q4_0_qkv_kq_gate",
            build(GgmlType::Q8_0, GgmlType::Q8_0, |_, nm| match nm {
                "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => GgmlType::Q4_K,
                "attn_qkv.weight" => GgmlType::Q4_0,
                "attn_gate.weight" => GgmlType::Q4_K,
                _ => GgmlType::Q8_0,
            }),
        ),
        // Non-K-quant sources whose 0.31.0 digests are pinned on both targets.
        (
            "pure_q4_0",
            build(GgmlType::Q4_0, GgmlType::Q4_0, |_, _| GgmlType::Q4_0),
        ),
        (
            "pure_q8_0",
            build(GgmlType::Q8_0, GgmlType::Q8_0, |_, _| GgmlType::Q8_0),
        ),
        (
            "q4_0_q6k_embd",
            build(GgmlType::Q6_K, GgmlType::Q6_K, |_, _| GgmlType::Q4_0),
        ),
        // C. Only the embedding is K-quant (also a both-target 0.31.0 pin).
        (
            "q8_0_q5k_embd",
            build(GgmlType::Q5_K, GgmlType::Q8_0, |_, _| GgmlType::Q8_0),
        ),
        // K. `pure_q8_0` carrying a Q4_K tensor no planner lookup resolves to: a
        // layer index spelled `blk.00`, and a second `blk.0.ffn_gate.weight` after
        // the one `find_tensor` returns. Every projection the planner reads is still
        // Q8_0, so both convert to `pure_q8_0`'s 0.31.0 bytes on both targets.
        (
            "pure_q8_0_noncanonical_layer",
            build_with_extra(
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                |_, _| GgmlType::Q8_0,
                &[("blk.00.ffn_gate.weight", GgmlType::Q4_K, [HID, INTER])],
            ),
        ),
        (
            "pure_q8_0_shadowed_ffn_gate",
            build_with_extra(
                GgmlType::Q8_0,
                GgmlType::Q8_0,
                |_, _| GgmlType::Q8_0,
                &[("blk.0.ffn_gate.weight", GgmlType::Q4_K, [HID, INTER])],
            ),
        ),
    ];
    // I. A K-quant source whose GDN pair mixes a K-quant attn_qkv with an F32 gate.
    for (name, qkv) in [
        ("q4_k_qkv_f32_gate", GgmlType::Q4_K),
        ("q5_k_qkv_f32_gate", GgmlType::Q5_K),
        ("q6_k_qkv_f32_gate", GgmlType::Q6_K),
    ] {
        v.push((
            name,
            build(GgmlType::Q8_0, GgmlType::Q8_0, |_, nm| match nm {
                "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => GgmlType::Q4_K,
                "attn_qkv.weight" => qkv,
                "attn_gate.weight" => GgmlType::F32,
                _ => GgmlType::Q8_0,
            }),
        ));
    }
    // J. A K-quant source with a Q4_K embedding and no `output.weight`: the head is
    // tied to the embedding and takes its scheme, so the embedding is dequantised.
    v.push((
        "tied_kquant_src",
        build_tied(GgmlType::Q4_K, |_, nm| match nm {
            "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => GgmlType::Q4_K,
            _ => GgmlType::Q8_0,
        }),
    ));
    v
}

fn fixture(name: &str) -> Vec<u8> {
    fixtures()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
        .1
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn kquant_source_policy_matrix() {
    assert_non_kquant_conversions_pinned();
    let hid2 = HID * HID;
    let ffn = HID * INTER;

    // A. The shipping Q4_0 shape is not a K-quant source; the conversion is exactly
    // the pre-existing one.
    let shipping = fixture("shipping_q4_0");
    let a = convert("shipping", &shipping, ConvertTarget::Generic);
    assert_eq!(a.primary, QuantScheme::Q4_0, "A: header scheme");
    assert_eq!(
        a.head,
        (QuantScheme::Q8_0, lbc_len(QuantScheme::Q8_0, VOCAB * HID)),
        "A: head requantised"
    );
    assert_eq!(
        a.embedding,
        (QuantScheme::Q4_0, lbc_len(QuantScheme::Q4_0, VOCAB * HID)),
        "A: embedding"
    );
    for l in 0..2 {
        assert_eq!(
            slice(&a, l, "ssm_out"),
            (QuantScheme::Q8_0, lbc_len(QuantScheme::Q8_0, hid2)),
            "A: ssm_out layer {l}"
        );
    }
    assert_eq!(
        slice(&a, 3, "w_down"),
        (QuantScheme::Q4_0, lbc_len(QuantScheme::Q4_0, ffn)),
        "A: Q4_1 requantised"
    );
    assert_eq!(
        slice(&a, 3, "wq"),
        (QuantScheme::Q6_K, lbc_len(QuantScheme::Q6_K, hid2)),
        "A: attn_q Q6_K verbatim as before"
    );
    assert_eq!(
        slice(&a, 0, "ssm_alpha").0,
        QuantScheme::Q8_0,
        "A: F32 gates requantised as before"
    );

    // B. Only the GDN pair is K-quant: not a K-quant source. Generic keeps the
    // pair verbatim (as before); the Metal target's pair force still applies.
    let pair = fixture("kquant_pair");
    let b = convert("pair", &pair, ConvertTarget::Generic);
    assert_eq!(b.primary, QuantScheme::Q8_0, "B: header scheme");
    assert_eq!(
        slice(&b, 0, "wq"),
        (
            QuantScheme::Q4_K,
            lbc_len(QuantScheme::Q4_K, HID * QKV_ROWS)
        ),
        "B: attn_qkv verbatim"
    );
    assert_eq!(
        slice(&b, 0, "attn_gate").0,
        QuantScheme::Q4_K,
        "B: attn_gate verbatim"
    );
    let bm = convert("pair_metal", &pair, ConvertTarget::Metal);
    assert_eq!(
        slice(&bm, 0, "wq").0,
        QuantScheme::Q8_0,
        "B: Metal pair force on attn_qkv"
    );
    assert_eq!(
        slice(&bm, 0, "attn_gate").0,
        QuantScheme::Q8_0,
        "B: Metal pair force on attn_gate"
    );

    // C. Only the embedding is K-quant: not a K-quant source, so the embedding
    // converts exactly as before on both targets (a Q5_K embedding to F32); the
    // K-quant policy never reaches a non-K-quant source.
    let embd_only = fixture("q8_0_q5k_embd");
    let c = convert("embd", &embd_only, ConvertTarget::Generic);
    assert_eq!(c.primary, QuantScheme::Q8_0, "C: header scheme");
    assert_eq!(
        c.embedding,
        (QuantScheme::F32, lbc_len(QuantScheme::F32, VOCAB * HID)),
        "C: embedding as before"
    );
    let cm = convert("embd_metal", &embd_only, ConvertTarget::Metal);
    assert_eq!(
        cm.embedding,
        (QuantScheme::F32, lbc_len(QuantScheme::F32, VOCAB * HID)),
        "C: Metal embedding F32"
    );

    // D. A K-quant source (one Q4_K ffn_up): every K-quant plane verbatim — a
    // Q4_K and a Q6_K ssm_out, a Q6_K embedding — while its Q5_K head is
    // re-quantised (only a Q6_K head is preserved); the Q4_1 kept, the F32 gates
    // kept, the header at the dominant K-quant scheme; and the default conversion
    // is byte-identical to the fidelity conversion.
    let kquant_src = fixture("kquant_src");
    let d = convert("kquant", &kquant_src, ConvertTarget::Generic);
    assert_eq!(
        d.version,
        lumen_format::LBC_VERSION_KQUANT_EMBEDDING,
        "D: an as-stored K-quant embedding stamps the newer version"
    );
    assert_eq!(
        a.version,
        lumen_format::LBC_VERSION,
        "A: a non-K-quant source keeps version 4"
    );
    assert_eq!(
        d.primary,
        QuantScheme::Q4_K,
        "D: header scheme from the predicate"
    );
    assert_eq!(
        d.head,
        (QuantScheme::Q8_0, lbc_len(QuantScheme::Q8_0, VOCAB * HID)),
        "D: Q5_K head requantised (the converter preserves only a Q6_K head)"
    );
    assert_eq!(
        d.embedding,
        (QuantScheme::Q6_K, lbc_len(QuantScheme::Q6_K, VOCAB * HID)),
        "D: Q6_K embedding kept"
    );
    assert_eq!(
        slice(&d, 0, "w_up"),
        (QuantScheme::Q4_K, lbc_len(QuantScheme::Q4_K, ffn)),
        "D: ffn_up verbatim"
    );
    assert_eq!(
        slice(&d, 0, "ssm_out"),
        (QuantScheme::Q4_K, lbc_len(QuantScheme::Q4_K, hid2)),
        "D: Q4_K ssm_out kept"
    );
    assert_eq!(
        slice(&d, 1, "ssm_out"),
        (QuantScheme::Q6_K, lbc_len(QuantScheme::Q6_K, hid2)),
        "D: Q6_K ssm_out kept"
    );
    assert_eq!(
        slice(&d, 3, "w_down"),
        (QuantScheme::Q4_1, lbc_len(QuantScheme::Q4_1, ffn)),
        "D: Q4_1 kept"
    );
    assert_eq!(
        slice(&d, 0, "ssm_alpha").0,
        QuantScheme::F32,
        "D: F32 gates kept"
    );
    // and the kept planes carry the source's bytes
    assert_source(
        "D: ffn_up",
        &d,
        d.planes[0]["w_up"],
        GgmlType::Q4_K,
        HID * INTER,
    );
    assert_source(
        "D: embedding",
        &d,
        d.embedding_plane,
        GgmlType::Q6_K,
        VOCAB * HID,
    );

    // D under `--requant q8_0`: the embedding keeps its stored K-quant scheme (the flag
    // requantises layer tensors, `ssm_out` included — the header it stamps is one the
    // K-quant layer kernels are closed on, so the default keep is off there,
    // `kquant_ssm_out_header.rs`), so the artifact is still version 5; under
    // `--dequantize` the embedding becomes F32 and the artifact is version 4.
    let d_requant = convert_with(
        "kquant_requant",
        &kquant_src,
        &ConvertOptions {
            requant_to: Some(QuantScheme::Q8_0),
            ..Default::default()
        },
    );
    assert_eq!(
        d_requant.embedding.0,
        QuantScheme::Q6_K,
        "D: --requant q8_0 keeps the stored embedding"
    );
    assert_eq!(
        d_requant.primary,
        QuantScheme::Q8_0,
        "D: header is the requant target"
    );
    assert_eq!(
        d_requant.version,
        lumen_format::LBC_VERSION_KQUANT_EMBEDDING,
        "D: --requant q8_0 still stamps the newer version"
    );
    assert_eq!(
        slice(&d_requant, 0, "ssm_out"),
        (QuantScheme::Q8_0, lbc_len(QuantScheme::Q8_0, hid2)),
        "D: --requant q8_0 requantises the ssm_out the header's kernels cannot read"
    );
    let d_deq = convert_with(
        "kquant_dequantize",
        &kquant_src,
        &ConvertOptions {
            dequantize_to_f32: true,
            ..Default::default()
        },
    );
    assert_eq!(
        d_deq.embedding.0,
        QuantScheme::F32,
        "D: --dequantize dequantises the embedding"
    );
    assert_eq!(
        d_deq.version,
        lumen_format::LBC_VERSION,
        "D: with no K-quant embedding the artifact keeps version 4"
    );

    // E. A Q5_K-dominant source takes Q5_K in the header and keeps its Q6_K head.
    let q5 = fixture("q5_dominant");
    let e = convert("q5", &q5, ConvertTarget::Generic);
    assert_eq!(e.primary, QuantScheme::Q5_K, "E: header scheme");
    assert_eq!(e.head.0, QuantScheme::Q6_K, "E: Q6_K head kept");
    assert_source("E: head", &e, e.head_plane, GgmlType::Q6_K, VOCAB * HID);
    assert_eq!(
        e.version,
        lumen_format::LBC_VERSION_KQUANT_EMBEDDING,
        "E: a Q5_K embedding is kept as stored, so the artifact takes the newer version"
    );

    // F. A K-quant source whose GDN pair is mixed — a Q4_K attn_qkv beside a Q8_0
    // attn_gate on layer 0, a K-quant pair on layer 1 — carries both as stored, and
    // with a Q8_0 embedding and head the artifact keeps version 4.
    let mixed = fixture("kquant_mixed_gdn_pair");
    let f = convert("mixed_pair", &mixed, ConvertTarget::Generic);
    assert_eq!(f.primary, QuantScheme::Q4_K, "F: header scheme");
    assert_eq!(
        f.version,
        lumen_format::LBC_VERSION,
        "F: a K-quant source with a Q8_0 embedding keeps version 4"
    );
    assert_eq!(
        slice(&f, 0, "wq"),
        (
            QuantScheme::Q4_K,
            lbc_len(QuantScheme::Q4_K, HID * QKV_ROWS)
        ),
        "F: Q4_K attn_qkv as stored"
    );
    assert_eq!(
        slice(&f, 0, "attn_gate"),
        (QuantScheme::Q8_0, lbc_len(QuantScheme::Q8_0, HID * HID)),
        "F: Q8_0 attn_gate as stored"
    );
    assert_eq!(
        slice(&f, 1, "attn_gate").0,
        QuantScheme::Q4_K,
        "F: a K-quant pair as stored"
    );

    // G. A K-quant source with Q3_K planes (the Q3_K_M shape): the Q3_K gate and up
    // ride through as before, the Q4_K down is carried, and the header takes whichever
    // of Q4_K / Q5_K / Q6_K has the most planned layer planes — Q3_K is not one of
    // them, so Q4_K here, even though Q3_K has twice as many planes.
    let q3 = fixture("q3_k_gate_up");
    let g = convert("q3", &q3, ConvertTarget::Generic);
    assert_eq!(g.primary, QuantScheme::Q4_K, "G: header scheme");
    assert_eq!(
        slice(&g, 0, "w_gate").0,
        QuantScheme::Q3_K,
        "G: Q3_K gate as stored"
    );
    assert_eq!(
        slice(&g, 0, "w_down"),
        (QuantScheme::Q4_K, lbc_len(QuantScheme::Q4_K, ffn)),
        "G: Q4_K down as stored"
    );

    // H. A K-quant source whose GDN pair mixes a Q4_0 attn_qkv with a K-quant
    // attn_gate: the generic target carries both as stored.
    let mixed_pair = fixture("q4_0_qkv_kq_gate");
    let h = convert("q4_0_qkv_kq_gate", &mixed_pair, ConvertTarget::Generic);
    assert_eq!(
        slice(&h, 0, "wq").0,
        QuantScheme::Q4_0,
        "H: generic qkv as stored"
    );
    assert_eq!(
        slice(&h, 0, "attn_gate").0,
        QuantScheme::Q4_K,
        "H: generic gate as stored"
    );
    assert_eq!(
        h.version,
        lumen_format::LBC_VERSION,
        "H: a Q8_0 embedding keeps version 4"
    );

    // I. A K-quant source whose GDN pair mixes a K-quant attn_qkv with an F32
    // attn_gate: the generic target carries both as stored.
    for (name, qkv) in [
        ("q4_k_qkv_f32_gate", GgmlType::Q4_K),
        ("q5_k_qkv_f32_gate", GgmlType::Q5_K),
        ("q6_k_qkv_f32_gate", GgmlType::Q6_K),
    ] {
        let i = convert(name, &fixture(name), ConvertTarget::Generic);
        assert_eq!(
            slice(&i, 0, "wq").0,
            qkv.to_lbc_quant().expect("a K-quant scheme"),
            "I/{qkv:?}: generic qkv as stored"
        );
        assert_eq!(
            slice(&i, 0, "attn_gate").0,
            QuantScheme::F32,
            "I/{qkv:?}: generic gate as stored"
        );
    }

    // J. A K-quant source whose head is tied to the embedding: the head shares the
    // embedding's plane and scheme, so a K-quant embedding would make a Q4_K head no
    // target serves. The embedding is dequantised instead, exactly as at 0.31.0 — the
    // layer planes still follow the policy.
    let tied = fixture("tied_kquant_src");
    let j = convert("tied_kquant_src", &tied, ConvertTarget::Generic);
    assert_eq!(
        j.embedding,
        (QuantScheme::F32, lbc_len(QuantScheme::F32, VOCAB * HID)),
        "J: a tied head dequantises the K-quant embedding"
    );
    assert_eq!(
        j.head,
        (QuantScheme::F32, lbc_len(QuantScheme::F32, VOCAB * HID)),
        "J: the tied head is F32, not Q4_K"
    );
    assert_eq!(
        j.version,
        lumen_format::LBC_VERSION,
        "J: no as-stored K-quant embedding, so version 4"
    );
    assert_eq!(
        slice(&j, 0, "w_up"),
        (QuantScheme::Q4_K, lbc_len(QuantScheme::Q4_K, ffn)),
        "J: the layer planes still follow the policy"
    );
    let jm = convert("tied_kquant_src_metal", &tied, ConvertTarget::Metal);
    assert_eq!(
        jm.embedding,
        (QuantScheme::F32, lbc_len(QuantScheme::F32, VOCAB * HID)),
        "J: Metal embedding F32"
    );

    // The Metal target is untouched by the policy: every fixture's `--target metal`
    // artifact is the one 0.31.0 wrote, byte for byte.
    metal_target_is_0_31_0();

    // D under the fidelity flag: the same bytes.
    std::env::set_var("LUMEN_CONVERT_SOURCE_FIDELITY", "1");
    let d_fid = convert("kquant_fidelity", &kquant_src, ConvertTarget::Generic);
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");
    assert_eq!(
        d.bytes, d_fid.bytes,
        "D: default conversion == fidelity conversion, byte for byte"
    );
}

/// The digests of the artifacts these non-K-quant sources convert to, pinned at
/// 0.31.0 (the same bytes from that release's converter): a change here moves a
/// shipped conversion. A failed assertion prints both digests.
fn assert_non_kquant_conversions_pinned() {
    const BOTH: &[(ConvertTarget, &str)] = &[
        (ConvertTarget::Generic, "generic"),
        (ConvertTarget::Metal, "metal"),
    ];
    // the shipping shape's Q6_K attn_q beside Q4_0 k/v is a generic-only artifact (the
    // Metal fused QKV launch needs one scheme, as before)
    const GENERIC: &[(ConvertTarget, &str)] = &[(ConvertTarget::Generic, "generic")];
    let names: &[(&str, &[(ConvertTarget, &str)])] = &[
        ("shipping_q4_0", GENERIC),
        ("pure_q4_0", BOTH),
        ("pure_q8_0", BOTH),
        ("q4_0_q6k_embd", BOTH),
        ("q8_0_q5k_embd", BOTH),
        ("pure_q8_0_noncanonical_layer", BOTH),
        ("pure_q8_0_shadowed_ffn_gate", BOTH),
    ];
    for (name, targets) in names {
        let gguf = fixture(name);
        for &(target, label) in targets.iter() {
            let p = convert(&format!("pin_{name}_{label}"), &gguf, target);
            let want = PINNED
                .iter()
                .find(|(n, l, _)| *n == *name && *l == label)
                .map(|(_, _, d)| *d)
                .unwrap_or_else(|| panic!("no pin for {name} {label}"));
            assert_eq!(
                sha256_hex(&p.bytes),
                want,
                "{name} on the {label} target moved from 0.31.0"
            );
        }
    }
}

/// Every fixture's `--target metal` artifact is byte for byte the one 0.31.0's
/// converter wrote — the K-quant source policy is generic-target-only, and the Metal
/// backend has no K-quant kernel, so a Metal conversion must not move at all.
///
/// The pins were derived by building these same fixtures against the 0.31.0 converter
/// (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
/// worktree and hashing each artifact; a source the 0.31.0 Metal target refuses is
/// pinned as [`METAL_REFUSED`] and must still be refused.
fn metal_target_is_0_31_0() {
    for (name, gguf) in fixtures() {
        let want = METAL_PINNED
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("no Metal pin for {name}"));
        match try_convert(&format!("metal_pin_{name}"), &gguf, ConvertTarget::Metal) {
            Ok(bytes) => assert_eq!(
                sha256_hex(&bytes),
                want,
                "{name}: the --target metal artifact moved from 0.31.0"
            ),
            Err(e) => assert_eq!(
                want, METAL_REFUSED,
                "{name}: the Metal target refuses the source ({e:?}) but a digest is pinned"
            ),
        }
    }
    // every non-K-quant fixture pinned on both targets must agree with `PINNED`
    for (name, label, digest) in PINNED {
        if *label != "metal" {
            continue;
        }
        let m = METAL_PINNED
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("no Metal pin for {name}"));
        assert_eq!(m, *digest, "{name}: the two Metal pins disagree");
    }
}

/// (fixture, target, sha256 of the artifact) — see `assert_non_kquant_conversions_pinned`.
const PINNED: &[(&str, &str, &str)] = &[
    (
        "shipping_q4_0",
        "generic",
        "4bf2ca97a7ce748ebfe401410b7ea61a5f58455033cf798e5dc7c2f809f954f7",
    ),
    (
        "pure_q4_0",
        "generic",
        "051102a4865657eb7e8e15880d9fe2fde6a772ec9003c5181195370bbab64200",
    ),
    (
        "pure_q4_0",
        "metal",
        "051102a4865657eb7e8e15880d9fe2fde6a772ec9003c5181195370bbab64200",
    ),
    (
        "pure_q8_0",
        "generic",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "pure_q8_0",
        "metal",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "q4_0_q6k_embd",
        "generic",
        "9c04d7f7893c2aed6e35a67734dc096fed42b981c0a819d4b5885805ba6aff80",
    ),
    (
        "q4_0_q6k_embd",
        "metal",
        "9c04d7f7893c2aed6e35a67734dc096fed42b981c0a819d4b5885805ba6aff80",
    ),
    (
        "q8_0_q5k_embd",
        "generic",
        "4b295d9954af68941706adc72ddebf95b4802d4097ddb1d74e51fc84537d4407",
    ),
    (
        "q8_0_q5k_embd",
        "metal",
        "4b295d9954af68941706adc72ddebf95b4802d4097ddb1d74e51fc84537d4407",
    ),
    (
        "pure_q8_0_noncanonical_layer",
        "generic",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "pure_q8_0_noncanonical_layer",
        "metal",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "pure_q8_0_shadowed_ffn_gate",
        "generic",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "pure_q8_0_shadowed_ffn_gate",
        "metal",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
];

/// A fixture the 0.31.0 Metal target refuses to convert; see [`metal_target_is_0_31_0`].
const METAL_REFUSED: &str = "refused";

/// (fixture, sha256 of its `--target metal` artifact at 0.31.0) — see
/// [`metal_target_is_0_31_0`].
const METAL_PINNED: &[(&str, &str)] = &[
    ("shipping_q4_0", METAL_REFUSED),
    (
        "kquant_pair",
        "f31705423abe1e828a1b20f2d8dd4fa01fe4696e693283b47d6dd3f7d5462895",
    ),
    (
        "kquant_src",
        "3fee47f6ff3beed88f7de9401d8cdc0a9d740d561f5ef3459c735c3a75336c2a",
    ),
    (
        "q5_dominant",
        "837c2ca034f8851d217e1d821246c8fde751e0c54471740bae85bb6bda63904e",
    ),
    (
        "kquant_mixed_gdn_pair",
        "0bf0c22de1c81b317bcbd7c23d6cac96ea0d004fb85c56b34cd665e44c679fd6",
    ),
    (
        "q3_k_gate_up",
        "4ac5019c504712207893fc994f7b975734f3b47a254e4bd7948c36f9be79c64c",
    ),
    (
        "q4_0_qkv_kq_gate",
        "4daf35dff78d4d208e0e2d4b6c7ec2e0402ec4a8ee7755f7c1f31892a8c5c610",
    ),
    (
        "pure_q4_0",
        "051102a4865657eb7e8e15880d9fe2fde6a772ec9003c5181195370bbab64200",
    ),
    (
        "pure_q8_0",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "q4_0_q6k_embd",
        "9c04d7f7893c2aed6e35a67734dc096fed42b981c0a819d4b5885805ba6aff80",
    ),
    (
        "q8_0_q5k_embd",
        "4b295d9954af68941706adc72ddebf95b4802d4097ddb1d74e51fc84537d4407",
    ),
    (
        "q4_k_qkv_f32_gate",
        "4daf35dff78d4d208e0e2d4b6c7ec2e0402ec4a8ee7755f7c1f31892a8c5c610",
    ),
    (
        "q5_k_qkv_f32_gate",
        "4daf35dff78d4d208e0e2d4b6c7ec2e0402ec4a8ee7755f7c1f31892a8c5c610",
    ),
    (
        "q6_k_qkv_f32_gate",
        "c5aad6f32b5a337e639e315b81960503dc4a6344fcc1a7a85e5dab6ed306aa71",
    ),
    (
        "tied_kquant_src",
        "f7d4ff8f8633dd84a6785e208872e0aea5650b39157d19f9f56b7c095a0f1762",
    ),
    (
        "pure_q8_0_noncanonical_layer",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
    (
        "pure_q8_0_shadowed_ffn_gate",
        "0238150d01202e87b8adf526846ffc138ce23a7ce831744edc7342175c8f81bb",
    ),
];
