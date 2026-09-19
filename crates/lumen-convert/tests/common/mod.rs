//! The K-quant source fixture the `kquant_gate_extent*` and interrupted-conversion
//! binaries share: a dense qwen35 GGUF whose two F32 GDN gates are built at a
//! caller-chosen element count. Each binary uses the helpers it needs.
//! The default policy and the explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch need
//! separate binaries — the switch is a process-wide environment variable — and both
//! read the same fixture from here.
#![allow(dead_code)]
use lumen_convert::convert::{convert_gguf_bytes_to_lbc, ConvertOptions, ConvertTarget};
use lumen_convert::gguf::{GgmlType, GgufBuilder};
use lumen_format::quantization::QuantScheme;
use lumen_format::reader::LbcFile;

const VOCAB: u64 = 256;
const HEADS: u32 = 8;
const KVH: u32 = 4;
const STATE: u64 = 32;
const GROUPS: u64 = 2;
const HID: u64 = 128;
const V_HEADS: u64 = 8;
/// The extent the projections read a gate at: `num_v_heads x hidden`.
pub const GATE: u64 = HID * V_HEADS;
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
        other => panic!("fixture: unsupported type {other:?}"),
    }
}

/// A four-layer qwen35 K-quant source: the quantized tensors Q8_0 except the Q4_K `ffn_down`
/// that makes it one and the Q4_K `ssm_out`, which is servable at this width. `alpha_n` and
/// `beta_n` set the stored element count of the two F32 gates, the property under test.
pub fn build(alpha_n: u64, beta_n: u64) -> Vec<u8> {
    let inter: u64 = 2 * HID;
    let v_dim: u64 = V_HEADS * STATE;
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
    b.add_u32(&k("feed_forward_length"), inter as u32);
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
            ("ffn_gate.weight", [HID, inter], GgmlType::Q8_0),
            ("ffn_up.weight", [HID, inter], GgmlType::Q8_0),
            ("ffn_down.weight", [inter, HID], GgmlType::Q4_K),
        ];
        if !full {
            planes.extend([
                ("attn_qkv.weight", [HID, qkv_rows], GgmlType::Q8_0),
                ("attn_gate.weight", [HID, v_dim], GgmlType::Q8_0),
                ("ssm_out.weight", [v_dim, HID], GgmlType::Q4_K),
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
            &[alpha_n],
            &vec![0.02; alpha_n as usize],
        );
        b.add_f32_tensor(
            &format!("{p}.ssm_beta.weight"),
            &[beta_n],
            &vec![0.02; beta_n as usize],
        );
    }
    b.build()
}

/// The primary scheme and the (scheme, bytes) of layer 0's two gates in the generic
/// artifact `gguf` converts to under the caller's options.
pub fn convert_gates(
    label: &str,
    gguf: &[u8],
    opts: &ConvertOptions,
) -> (QuantScheme, (QuantScheme, Vec<u8>), (QuantScheme, Vec<u8>)) {
    let out = std::env::temp_dir().join(format!("kquant_gate_{label}_{}.lbc", std::process::id()));
    convert_gguf_bytes_to_lbc(gguf, &out, opts).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let bytes = std::fs::read(&out).unwrap();
    let f = LbcFile::open(&out).unwrap();
    let layer = &f.layer_indices[0];
    let plane = |slice: &lumen_format::index::TensorSlice| {
        let start = (layer.layer_offset_bytes + slice.offset) as usize;
        (
            slice.quant,
            bytes[start..start + slice.length as usize].to_vec(),
        )
    };
    let alpha = plane(
        layer
            .subtensors
            .ssm_alpha
            .as_ref()
            .expect("fixture has alpha"),
    );
    let beta = plane(
        layer
            .subtensors
            .ssm_beta
            .as_ref()
            .expect("fixture has beta"),
    );
    let primary = f.header.quantization.scheme;
    drop(f);
    std::fs::remove_file(&out).ok();
    (primary, alpha, beta)
}

pub fn generic() -> ConvertOptions {
    ConvertOptions {
        target: ConvertTarget::Generic,
        ..Default::default()
    }
}
