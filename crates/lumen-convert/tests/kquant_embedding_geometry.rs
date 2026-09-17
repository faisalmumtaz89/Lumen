//! What a K-quant source preserves by default has to be a plane the runtime serves.
//! The embedding gather reads the table as whole 256-element superblocks, and the loader
//! sizes the plane from the header's `vocab_size * hidden_dim`
//! (`lumen_format::serving_rules::kquant_global_plane_len`) and requires the stored plane
//! to be exactly that many bytes, so a K-quant embedding whose header `vocab x hidden` is
//! not whole superblocks, and one whose stored plane is some other length, are both
//! refused at load; the K-quant source policy dequantises such an embedding instead of
//! carrying it, which is the embedding 0.31.0 wrote for the same file. The two questions
//! are about the plane, not the row count, and the fourth fixture is the boundary where
//! that separates: a `token_embd` short of the header's vocab by fewer elements than a
//! superblock holds stores exactly the plane the header needs, and is carried.
//!
//! These are files the converter reads without complaint. GGUF sizes a tensor from its
//! flattened element count at `div_ceil`, so a K-quant plane can end in a partial
//! superblock — the source of the first fixture (257 x 128 = 32,896 elements: 128 whole
//! superblocks plus a 128-element tail, which GGUF stores in `div_ceil` = 129
//! superblocks and the loader refuses). And the header's vocab is the
//! tokenizer's token count when the source carries one (`hyperparams.rs`), not the
//! tensor's row count, so a padded `token_embd` is whole superblocks over a vocab that
//! needs fewer of them — the source of the third fixture (258 rows of 128 over a
//! 256-token tokenizer).
//!
//! The quantiser's own rule is per row, and `token_embd`'s row length is the hidden width,
//! so every standard export whose embedding is K-quant has a hidden width that is whole
//! superblocks and a flattened count that is too; the registry's two K-quant cells are both
//! `qwen3.8-27b`, at hidden 5120 with 248,320 tokens and 248,320 embedding rows, so no
//! shipped file converts differently. The pins are the guard against the default widening
//! to one that does.
//!
//! Each pin was derived by building the same fixture against the 0.31.0 converter (release
//! commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway worktree and hashing
//! the embedding plane of its artifact; both fixture GGUFs hash the same on both trees,
//! which is the cross-check that the transcription did not drift.
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const HID: u64 = 128;
const HEADS: u32 = 8;
const KVH: u32 = 4;
const STATE: u64 = 32;
const GROUPS: u64 = 2;
// Four layers: the converter's layer kinds are positional (full attention at 3, 7, …).
const LAYERS: u32 = 4;

/// The source bytes of one plane, varying with the type and the superblock index.
/// K-quant planes are sized the way GGUF sizes them — `div_ceil`, so a partial final
/// superblock is a whole stored block.
fn bytes_for(t: GgmlType, n: u64) -> Vec<u8> {
    let n = n as usize;
    match t {
        GgmlType::Q8_0 => {
            let mut v = vec![0u8; n.div_ceil(32) * 34];
            for b in v.chunks_exact_mut(34) {
                b[1] = 0x3C;
            }
            v
        }
        GgmlType::Q4_K => {
            let mut v = vec![0u8; n.div_ceil(256) * 144];
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

/// A four-layer qwen35 K-quant source whose `token_embd` has `rows` rows: Q8_0 everywhere
/// except the Q4_K `ffn_down` that makes it one (its in_dim is `2 * HID`, whole superblocks
/// at either size) and the Q4_K embedding. `tokens` is the tokenizer's token count, which
/// is the header's vocab when the source carries one and which the head is sized to; `None`
/// leaves the source without a token list, so the header's vocab is `rows` itself. The
/// embedding's geometry against that vocab is the property under test; the head is Q8_0, so
/// the head rule is not what decides here.
fn build(rows: u64, tokens: Option<u64>) -> Vec<u8> {
    let vocab = tokens.unwrap_or(rows);
    let inter: u64 = 2 * HID;
    let v_heads: u64 = HID / STATE;
    let qkv_rows: u64 = (2 * GROUPS + v_heads) * STATE;
    let kvd = (HID / HEADS as u64) * KVH as u64;
    let mut b = GgufBuilder::new();
    let k = |s: &str| format!("qwen35.{s}");
    b.add_string("general.architecture", "qwen35");
    b.add_u32(&k("block_count"), LAYERS);
    b.add_u32(&k("attention.head_count"), HEADS);
    b.add_u32(&k("attention.head_count_kv"), KVH);
    b.add_u32(&k("attention.key_length"), HID as u32 / HEADS);
    b.add_u32(&k("embedding_length"), HID as u32);
    b.add_u32(&k("feed_forward_length"), inter as u32);
    b.add_u32(&k("context_length"), 64);
    b.add_f32(&k("rope.freq_base"), 10000.0);
    b.add_f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    b.add_u32(&k("ssm.time_step_rank"), v_heads as u32);
    b.add_u32(&k("ssm.group_count"), GROUPS as u32);
    b.add_u32(&k("ssm.state_size"), STATE as u32);
    b.add_u32(&k("ssm.conv_kernel"), 4);
    if let Some(t) = tokens {
        let toks: Vec<String> = (0..t).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
        b.add_string("tokenizer.ggml.model", "gpt2");
        b.add_string_array("tokenizer.ggml.tokens", &refs);
    }
    b.add_tensor(
        "token_embd.weight",
        GgmlType::Q4_K,
        &[rows, HID],
        bytes_for(GgmlType::Q4_K, rows * HID),
    );
    b.add_f32_tensor("output_norm.weight", &[HID], &vec![1.0; HID as usize]);
    b.add_tensor(
        "output.weight",
        GgmlType::Q8_0,
        &[HID, vocab],
        bytes_for(GgmlType::Q8_0, vocab * HID),
    );
    for l in 0..LAYERS {
        let p = format!("blk.{l}");
        let full = l == 3;
        let mut planes: Vec<(&str, [u64; 2], GgmlType)> = vec![
            ("attn_q.weight", [HID, HID], GgmlType::Q8_0),
            ("attn_k.weight", [HID, kvd], GgmlType::Q8_0),
            ("attn_v.weight", [HID, kvd], GgmlType::Q8_0),
            ("attn_output.weight", [HID, HID], GgmlType::Q8_0),
            ("ffn_gate.weight", [HID, inter], GgmlType::Q8_0),
            ("ffn_up.weight", [HID, inter], GgmlType::Q8_0),
            ("ffn_down.weight", [inter, HID], GgmlType::Q4_K),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [HID, qkv_rows], GgmlType::Q8_0),
                ("attn_gate.weight", [HID, HID], GgmlType::Q8_0),
                ("ssm_out.weight", [HID, HID], GgmlType::Q8_0),
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
            &[HID, v_heads],
            &vec![0.02; (HID * v_heads) as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_beta.weight"),
            &[HID, v_heads],
            &vec![0.02; (HID * v_heads) as usize],
        );
    }
    b.build()
}

/// (primary scheme, LBC version, embedding scheme, embedding plane bytes) of the generic
/// artifact `gguf` converts to under default options.
fn convert_embedding(label: &str, gguf: &[u8]) -> (QuantScheme, u32, QuantScheme, Vec<u8>) {
    let out = std::env::temp_dir().join(format!("kquant_embd_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(gguf, &out, &opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    let f = LbcFile::open(&out).unwrap();
    let embd = f.header.embedding;
    let plane = bytes[embd.offset as usize..(embd.offset + embd.length) as usize].to_vec();
    let primary = f.header.quantization.scheme;
    let version = f.header.version;
    drop(f);
    std::fs::remove_file(&out).ok();
    (primary, version, embd.quant, plane)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 embedding plane of the 257-row fixture: the Q4_K source embedding
/// dequantised to F32.
const PARTIAL_EMBEDDING_0_31_0: &str =
    "7bfa89ee41f18871269c5b9b6970fc8411661211c4f11631ef51eab4afd8f839";

/// The 0.31.0 embedding plane of the 258-row / 256-token fixture: the same
/// dequantisation, over the rows the source stores.
const PADDED_EMBEDDING_0_31_0: &str =
    "8675651898d77ef1cbe478e28b4ef7e367fabcdd4e667f536d1a1a0ab6b7c2f2";

#[test]
fn a_kquant_embedding_of_a_partial_superblock_keeps_the_0_31_0_embedding() {
    // 257 x 128 = 32,896 elements: 128 whole superblocks plus a 128-element tail.
    let (primary, version, quant, plane) = convert_embedding("partial", &build(257, None));
    // The fixture is a K-quant source, so the policy really is the one under test.
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::F32,
        "a K-quant embedding whose element count is not whole superblocks was carried verbatim"
    );
    assert_eq!(plane.len(), 257 * 128 * 4);
    assert_eq!(
        sha256_hex(&plane),
        PARTIAL_EMBEDDING_0_31_0,
        "the embedding plane moved from 0.31.0"
    );
    assert_eq!(version, 4, "no as-stored K-quant embedding, so version 4");
    // The source geometry is one the loader's rule refuses, so the F32 fallback was
    // the only servable choice.
    assert!(
        lumen_format::serving_rules::kquant_global_plane_len(QuantScheme::Q4_K, 257 * 128).is_err(),
        "the rule the default has to respect"
    );
}

#[test]
fn a_kquant_embedding_of_whole_superblocks_is_still_carried_verbatim() {
    let (primary, version, quant, plane) = convert_embedding("whole", &build(256, None));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::Q4_K,
        "the K-quant embedding was not preserved"
    );
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q4_K, 256 * HID),
        "the preserved embedding is not the source bytes"
    );
    assert_eq!(
        version, 5,
        "an as-stored K-quant embedding stamps version 5"
    );
}

#[test]
fn a_kquant_embedding_padded_past_the_header_vocab_keeps_the_0_31_0_embedding() {
    // 258 rows of 128 is 33,024 elements — whole superblocks, so the count alone
    // admits the plane. The header's vocab is the tokenizer's 256 tokens, and the
    // loader sizes the embedding at 256 x 128: 128 superblocks, 18,432 bytes, not the
    // 129 superblocks the source stores.
    let (primary, version, quant, plane) = convert_embedding("padded", &build(258, Some(256)));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::F32,
        "a K-quant embedding with rows the header's vocab does not have was carried verbatim"
    );
    assert_eq!(plane.len(), 258 * (HID as usize) * 4);
    assert_eq!(
        sha256_hex(&plane),
        PADDED_EMBEDDING_0_31_0,
        "the embedding plane moved from 0.31.0"
    );
    assert_eq!(version, 4, "no as-stored K-quant embedding, so version 4");
    // The source count passes the superblock rule on its own; it is the header's
    // geometry that refuses the plane, which is the rule the default has to respect.
    assert!(
        lumen_format::serving_rules::kquant_global_plane_len(QuantScheme::Q4_K, 258 * 128).is_ok(),
        "the source count is whole superblocks"
    );
    assert_eq!(
        lumen_format::serving_rules::kquant_global_plane_len(QuantScheme::Q4_K, 256 * 128),
        Ok(18432),
        "the header's geometry needs a shorter plane than the source stores"
    );
}

/// The accepted boundary: a `token_embd` one row short of the header's vocab, at a
/// hidden width small enough that the missing row fits inside the final superblock's
/// padding. 255 x 128 = 32,640 elements, which GGUF stores at `div_ceil` in 128
/// superblocks = 18,432 bytes — exactly the plane the header's 256 x 128 needs, so both
/// of the loader's questions pass, the plane is carried and the artifact is version 5.
/// The gather for the 256th token reads that padding: inside the plane the loader sized,
/// not past it. (0.31.0 wrote F32 here — 130,560 bytes under a header geometry that
/// needs 131,072 — and its F32 gather read past the plane; nothing is pinned on that.)
#[test]
fn a_kquant_embedding_one_row_short_of_the_header_vocab_is_carried() {
    let (primary, version, quant, plane) = convert_embedding("short", &build(255, Some(256)));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::Q4_K,
        "the K-quant embedding the header's geometry accepts was not preserved"
    );
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q4_K, 255 * HID),
        "the preserved embedding is not the source bytes"
    );
    assert_eq!(
        version, 5,
        "an as-stored K-quant embedding stamps version 5"
    );
    // The loader's rule on the loader's numbers: the header's product is whole
    // superblocks and needs exactly the bytes the source stores.
    assert_eq!(
        lumen_format::serving_rules::kquant_global_plane_len(QuantScheme::Q4_K, 256 * 128),
        Ok(plane.len()),
        "the plane the loader sizes is the plane the source stores"
    );
    // The source's own element count is not whole superblocks — it is not the operand
    // either side measures.
    assert!(
        lumen_format::serving_rules::kquant_global_plane_len(QuantScheme::Q4_K, 255 * 128).is_err(),
        "the source count is not the number the rule is run on"
    );
}
