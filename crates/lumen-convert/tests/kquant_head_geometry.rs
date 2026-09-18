//! What a K-quant source preserves by default has to be a plane the runtime serves. The
//! head matvec reads `hidden_dim / 256` whole superblocks per row, so a Q6_K head whose
//! row width is not whole superblocks is refused at load
//! (`lumen_format::serving_rules::validate_output_head_row_alignment`), and the loader
//! sizes the head plane from the header's `vocab_size * hidden_dim`, so a head stored
//! as a plane shorter than that product is refused there as well (`weight/kquant.rs` on
//! the host read, `lumen_format::serving_rules::kquant_global_plane_len` on the CUDA
//! upload); the K-quant source policy requantises a head failing either rule instead of
//! carrying it, which is the head 0.31.0 wrote for the same file. The explicit
//! `LUMEN_CONVERT_SOURCE_FIDELITY` / `LUMEN_CONVERT_KEEP_Q6K_OUTPUT` switches are
//! outside both rules and unchanged — `q6k_head_gate.rs` pins them — so these fixtures
//! set no environment.
//!
//! GGUF sizes a tensor from its flattened element count, so a K-quant plane whose row
//! is narrower than a superblock is a file the converter reads without complaint —
//! the source of the first fixture. The registry's two K-quant cells are both
//! `qwen3.8-27b`, at hidden 5120, and no K-quant tensor of a standard export has a row
//! length that is not a multiple of the 256-element superblock, so no shipped file
//! converts differently on that rule; the pin is the guard against the default widening
//! to one that does.
//!
//! The header's vocab is the tokenizer's token count when the source carries one, else
//! `token_embd`'s row count (`hyperparams.rs`), and nothing compares `output.weight`'s
//! rows to it, so a head of fewer rows than the vocab is a file the converter reads
//! without complaint as well — the source of the third fixture, a shape no other
//! fixture in this crate's tests builds. The length rule is an equality, as the
//! embedding keep's is: a head plane longer than the header's `vocab_size * hidden_dim`
//! is requantised by it too — the host read would accept such a plane, the CUDA upload
//! refuses any length but the one that product needs (`cuda/backend_impl.rs`) — so the
//! head of a default artifact is either exactly the plane that product needs or the one
//! 0.31.0 wrote.
//!
//! The pins were derived by building these same fixtures against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and reading the head plane of each artifact — hashed for the first, and
//! for the third its length, since that head plane is the source tensor requantised
//! and 0.31.0 sized it from the tensor, not from the header (so it is shorter than
//! `vocab_size * hidden_dim` at both commits, which is why the third pin is on the
//! bytes the converter writes and not on a load). The fixture GGUFs hash the same on
//! both trees, which is the cross-check that the transcription did not drift.
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

/// The source bytes of one plane, varying with the type and the superblock index.
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
        other => panic!("fixture: unsupported type {other:?}"),
    }
}

/// A four-layer qwen35 K-quant source of hidden width `hid`: the quantized tensors Q8_0
/// except the Q4_K `ffn_down` that makes it one (its in_dim is `2 * hid`, so the layer
/// contract gate passes at either width) and the Q6_K head. `hid` sets the head's
/// row length and `head_rows` its row count against the `VOCAB` the header takes from
/// `token_embd`, which are the two properties under test.
fn build(hid: u64, head_rows: u64) -> Vec<u8> {
    let inter: u64 = 2 * hid;
    let v_heads: u64 = hid / STATE;
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
        GgmlType::Q6_K,
        &[hid, head_rows],
        bytes_for(GgmlType::Q6_K, head_rows * hid),
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
                ("attn_gate.weight", [hid, hid], GgmlType::Q8_0),
                ("ssm_out.weight", [hid, hid], GgmlType::Q8_0),
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

/// (primary scheme, head scheme, head plane bytes) of the generic artifact `gguf`
/// converts to under default options.
fn convert_head(label: &str, gguf: &[u8]) -> (QuantScheme, QuantScheme, Vec<u8>) {
    let out = std::env::temp_dir().join(format!("kquant_head_{label}_{}.lbc", std::process::id()));
    let opts = ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    };
    convert_gguf_bytes_to_lbc(gguf, &out, &opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    let f = LbcFile::open(&out).unwrap();
    let head = f.header.output_proj;
    let plane = bytes[head.offset as usize..(head.offset + head.length) as usize].to_vec();
    let primary = f.header.quantization.scheme;
    drop(f);
    std::fs::remove_file(&out).ok();
    (primary, head.quant, plane)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 head plane of the hidden-128 fixture: the Q6_K source head
/// requantised to Q8_0.
const NARROW_HEAD_0_31_0: &str = "72efe354aa57b13c6f888d37d19ebf1abcc0121e3445a85c65957bcff891590e";

/// The 0.31.0 head plane of the short-head fixture: its 65,280-element Q6_K source head
/// requantised to Q8_0, 34 bytes per 32 elements. The length is the pin — the bytes are
/// the same requantisation the first fixture's digest already covers.
const SHORT_HEAD_0_31_0_LEN: usize = 69_360;

#[test]
fn a_q6k_head_narrower_than_a_superblock_keeps_the_0_31_0_head() {
    let (primary, quant, plane) = convert_head("narrow", &build(128, VOCAB));
    // The fixture is a K-quant source, so the policy really is the one under test.
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a Q6_K head whose rows are narrower than a superblock was carried verbatim"
    );
    assert_eq!(
        sha256_hex(&plane),
        NARROW_HEAD_0_31_0,
        "the head plane moved from 0.31.0"
    );
    // And what the converter wrote is what the loader accepts.
    lumen_format::serving_rules::validate_output_head_row_alignment(quant, 128)
        .expect("the written head must pass the loader's own head rule");
}

#[test]
fn a_q6k_head_of_whole_superblock_rows_is_still_carried_verbatim() {
    let (primary, quant, plane) = convert_head("wide", &build(256, VOCAB));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(quant, QuantScheme::Q6_K, "the Q6_K head was not preserved");
    assert_eq!(
        plane,
        bytes_for(GgmlType::Q6_K, VOCAB * 256),
        "the preserved head is not the source bytes"
    );
    lumen_format::serving_rules::validate_output_head_row_alignment(quant, 256)
        .expect("the written head must pass the loader's own head rule");
}

/// The head's row width is whole superblocks here, so the length rule is the one that
/// decides. 0.31.0 sized a requantised head from the source tensor rather than from the
/// header, so the plane pinned below is itself shorter than `vocab_size * hidden_dim` at
/// both commits; the pin is on the bytes the converter writes.
#[test]
fn a_q6k_head_short_of_the_headers_vocab_keeps_the_0_31_0_head() {
    // 255 rows of 256 elements: whole superblocks, one row short of the header's vocab.
    let (primary, quant, plane) = convert_head("short", &build(256, VOCAB - 1));
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    lumen_format::serving_rules::validate_output_head_row_alignment(QuantScheme::Q6_K, 256).expect(
        "the head's row width must be servable, so that the length rule is the one under test",
    );
    assert_ne!(
        lumen_format::serving_rules::kquant_global_plane_len(
            QuantScheme::Q6_K,
            (VOCAB * 256) as usize
        )
        .unwrap(),
        bytes_for(GgmlType::Q6_K, (VOCAB - 1) * 256).len(),
        "the fixture's head plane is the length the header's vocab x hidden needs"
    );
    assert_eq!(
        quant,
        QuantScheme::Q8_0,
        "a Q6_K head short of the header's vocab x hidden was carried verbatim"
    );
    assert_eq!(
        plane.len(),
        SHORT_HEAD_0_31_0_LEN,
        "the head plane moved from 0.31.0"
    );
}
