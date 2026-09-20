//! Synthetic HF checkpoint generator for end-to-end testing.
//!
//! Writes the smallest checkpoint this importer accepts — one GDN layer's
//! projections, the MLP and the head, quantized as [`Modules`] says — plus
//! the donor GGUF it takes its hyperparameters from, and converts the pair
//! to an `.lbc`. The counterpart of [`lumen_format::test_model`] for the HF
//! import path, so the binaries' admission tests share one source.

use std::path::{Path, PathBuf};

use lumen_format::QuantScheme;

use crate::convert::ConvertError;

const HID: usize = 64;
const INTER: usize = 64;
const VOCAB: usize = 32;
const VH: usize = 2;
const KH: usize = 1;
const GHD: usize = 32;
const QKV_ROWS: usize = 2 * KH * GHD + VH * GHD;

/// Every module the converter lowers as a matrix, with its logical `[n, k]`.
const MODULES: [(&str, usize, usize); 7] = [
    ("model.layers.0.linear_attn.in_proj_qkv", QKV_ROWS, HID),
    ("model.layers.0.linear_attn.in_proj_z", VH * GHD, HID),
    ("model.layers.0.linear_attn.out_proj", HID, VH * GHD),
    ("model.layers.0.mlp.gate_proj", INTER, HID),
    ("model.layers.0.mlp.up_proj", INTER, HID),
    ("model.layers.0.mlp.down_proj", HID, INTER),
    ("lm_head", VOCAB, HID),
];

/// How one module's weight is stored in the checkpoint.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Weight {
    Nvfp4,
    Fp8,
    Int4G32,
    Bf16,
}

/// What a synthetic checkpoint's modules carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modules {
    /// NVFP4 MLP and head, FP8 GDN projections: the ModelOpt
    /// mixed-precision export.
    Nvfp4AndFp8,
    /// Every projection FP8, the head left in BF16.
    Fp8Only,
    /// INT4 group-32 projections except the GDN output projection, which is
    /// FP8, and the head, which stays BF16: one planar projection in the
    /// body is enough to make that planar scheme the primary.
    Int4WithOneFp8,
    /// INT4 group-32 projections under an NVFP4 head: the head alone is
    /// planar, and a planar module anywhere makes the primary planar.
    Int4WithNvfp4Head,
}

impl Modules {
    /// The primary scheme an artifact built from these modules carries.
    pub fn primary_scheme(self) -> QuantScheme {
        match self {
            Self::Nvfp4AndFp8 | Self::Int4WithNvfp4Head => QuantScheme::Nvfp4,
            Self::Fp8Only | Self::Int4WithOneFp8 => QuantScheme::Fp8E4M3,
        }
    }

    /// INT4 group-32 is the compressed-tensors dialect, which declares one
    /// config group for every Linear; the planar schemes are the ModelOpt
    /// dialect, which declares each module by name.
    fn is_modelopt(self) -> bool {
        !matches!(self, Self::Int4WithOneFp8 | Self::Int4WithNvfp4Head)
    }

    fn weight_of(self, base: &str) -> Weight {
        match self {
            Self::Nvfp4AndFp8 if base.contains("linear_attn") => Weight::Fp8,
            Self::Nvfp4AndFp8 => Weight::Nvfp4,
            Self::Int4WithNvfp4Head if base == "lm_head" => Weight::Nvfp4,
            Self::Int4WithNvfp4Head => Weight::Int4G32,
            Self::Fp8Only | Self::Int4WithOneFp8 if base == "lm_head" => Weight::Bf16,
            Self::Fp8Only => Weight::Fp8,
            Self::Int4WithOneFp8 if base.ends_with("out_proj") => Weight::Fp8,
            Self::Int4WithOneFp8 => Weight::Int4G32,
        }
    }
}

fn rand_bytes(len: usize, seed: &mut u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*seed >> 33) as u8
        })
        .collect()
}

fn bf16_bytes(len: usize, seed: &mut u64) -> Vec<u8> {
    // Small finite values: 0x3F80 is bf16 1.0, so every weight is in [1, 2).
    (0..len)
        .flat_map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mantissa = ((*seed >> 40) & 0x7F) as u16;
            (0x3F80u16 | mantissa).to_le_bytes()
        })
        .collect()
}

/// One safetensors shard: `u64` header length, the JSON header, the data.
fn shard_bytes(entries: &[(String, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let begin = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

/// Write the donor GGUF the converter takes its hyperparameters from.
fn write_donor(path: &Path) {
    use crate::gguf::GgufBuilder;
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "qwen35");
    b.add_u32("qwen35.block_count", 1);
    b.add_u32("qwen35.attention.head_count", 4);
    b.add_u32("qwen35.attention.head_count_kv", 2);
    b.add_u32("qwen35.attention.key_length", 8);
    b.add_u32("qwen35.embedding_length", HID as u32);
    b.add_u32("qwen35.feed_forward_length", INTER as u32);
    b.add_u32("qwen35.context_length", 64);
    b.add_f32("qwen35.rope.freq_base", 10000.0);
    b.add_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-5);
    b.add_u32("qwen35.ssm.time_step_rank", VH as u32);
    b.add_u32("qwen35.ssm.group_count", KH as u32);
    b.add_u32("qwen35.ssm.state_size", GHD as u32);
    b.add_u32("qwen35.ssm.conv_kernel", 4);
    let names: Vec<String> = (0..VOCAB).map(|i| format!("t{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    b.add_string_array("tokenizer.ggml.tokens", &refs);
    b.add_f32_tensor(
        "token_embd.weight",
        &[VOCAB as u64, HID as u64],
        &vec![0.0; VOCAB * HID],
    );
    std::fs::write(path, b.build()).unwrap();
}

/// A written checkpoint and the donor GGUF it converts against.
pub struct Checkpoint {
    pub dir: PathBuf,
    pub donor: PathBuf,
}

/// Write a one-layer checkpoint of `modules`, and its donor GGUF, into `dir`.
pub fn write_checkpoint(dir: &Path, modules: Modules) -> Checkpoint {
    let mut seed = 42u64;
    let mut entries: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut quantized_layers = serde_json::Map::new();

    for (base, n, k) in MODULES {
        let (n64, k64) = (n as u64, k as u64);
        match modules.weight_of(base) {
            Weight::Nvfp4 => {
                entries.push((
                    format!("{base}.weight"),
                    "U8",
                    vec![n64, k64 / 2],
                    rand_bytes(n * k / 2, &mut seed),
                ));
                entries.push((
                    format!("{base}.weight_scale"),
                    "F8_E4M3",
                    vec![n64, k64 / 16],
                    // 0x38 is E4M3 1.0: every block scale is finite and non-zero.
                    vec![0x38u8; n * (k / 16)],
                ));
                entries.push((
                    format!("{base}.weight_scale_2"),
                    "F32",
                    vec![],
                    1.5e-4f32.to_le_bytes().to_vec(),
                ));
                quantized_layers.insert(
                    base.to_owned(),
                    serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16 }),
                );
            }
            Weight::Fp8 => {
                entries.push((
                    format!("{base}.weight"),
                    "F8_E4M3",
                    vec![n64, k64],
                    rand_bytes(n * k, &mut seed),
                ));
                entries.push((
                    format!("{base}.weight_scale"),
                    "F32",
                    vec![],
                    9.7e-4f32.to_le_bytes().to_vec(),
                ));
                quantized_layers
                    .insert(base.to_owned(), serde_json::json!({ "quant_algo": "FP8" }));
            }
            Weight::Int4G32 => {
                let groups = k / 32;
                let zero_rows = n.div_ceil(8);
                entries.push((
                    format!("{base}.weight_packed"),
                    "I32",
                    vec![n64, k64 / 8],
                    rand_bytes(n * (k / 8) * 4, &mut seed),
                ));
                entries.push((
                    format!("{base}.weight_scale"),
                    "BF16",
                    vec![n64, groups as u64],
                    bf16_bytes(n * groups, &mut seed),
                ));
                entries.push((
                    format!("{base}.weight_zero_point"),
                    "I32",
                    vec![zero_rows as u64, groups as u64],
                    rand_bytes(zero_rows * groups * 4, &mut seed),
                ));
                entries.push((
                    format!("{base}.weight_shape"),
                    "I64",
                    vec![2],
                    [n as i64, k as i64]
                        .iter()
                        .flat_map(|d| d.to_le_bytes())
                        .collect(),
                ));
            }
            Weight::Bf16 => {
                entries.push((
                    format!("{base}.weight"),
                    "BF16",
                    vec![n64, k64],
                    bf16_bytes(n * k, &mut seed),
                ));
            }
        }
    }

    for (name, shape) in [
        ("model.embed_tokens.weight", vec![VOCAB as u64, HID as u64]),
        ("model.norm.weight", vec![HID as u64]),
        ("model.layers.0.input_layernorm.weight", vec![HID as u64]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![HID as u64],
        ),
        ("model.layers.0.linear_attn.A_log", vec![VH as u64]),
        (
            "model.layers.0.linear_attn.conv1d.weight",
            vec![QKV_ROWS as u64, 1, 4],
        ),
        ("model.layers.0.linear_attn.dt_bias", vec![VH as u64]),
        (
            "model.layers.0.linear_attn.in_proj_b.weight",
            vec![VH as u64, HID as u64],
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight",
            vec![VH as u64, HID as u64],
        ),
        ("model.layers.0.linear_attn.norm.weight", vec![GHD as u64]),
    ] {
        let elements: u64 = shape.iter().product();
        entries.push((
            name.to_owned(),
            "BF16",
            shape,
            bf16_bytes(elements as usize, &mut seed),
        ));
    }

    let ckpt = dir.join("checkpoint");
    std::fs::create_dir_all(&ckpt).unwrap();
    std::fs::write(ckpt.join("model-00001.safetensors"), shard_bytes(&entries)).unwrap();
    let weight_map: serde_json::Map<String, serde_json::Value> = entries
        .iter()
        .map(|(n, ..)| (n.clone(), serde_json::json!("model-00001.safetensors")))
        .collect();
    write_json(
        &ckpt.join("model.safetensors.index.json"),
        &serde_json::json!({ "weight_map": weight_map }),
    );

    let quantization_config = if modules.is_modelopt() {
        serde_json::json!({ "quant_method": "modelopt" })
    } else {
        serde_json::json!({
            "format": "pack-quantized",
            "ignore": ["lm_head"],
            "config_groups": { "group_0": {
                "targets": ["Linear"],
                "input_activations": null,
                "output_activations": null,
                "weights": {
                    "type": "int", "num_bits": 4, "group_size": 32,
                    "symmetric": false, "strategy": "group", "dynamic": false
                }
            }}
        })
    };
    write_json(
        &ckpt.join("config.json"),
        &serde_json::json!({
            "quantization_config": quantization_config,
            "text_config": {
                "hidden_size": HID, "num_hidden_layers": 1,
                "intermediate_size": INTER, "vocab_size": VOCAB,
            },
        }),
    );
    if modules.is_modelopt() {
        write_json(
            &ckpt.join("hf_quant_config.json"),
            &serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": quantized_layers,
            }}),
        );
    }

    let donor = dir.join("donor.gguf");
    write_donor(&donor);
    Checkpoint { dir: ckpt, donor }
}

fn write_json(path: &Path, value: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

/// Convert a checkpoint of `modules` written in `dir`. Returns the
/// artifact's path.
pub fn write_artifact(dir: &Path, modules: Modules) -> Result<PathBuf, ConvertError> {
    let ckpt = write_checkpoint(dir, modules);
    let artifact = dir.join("model.lbc");
    let stats = crate::convert_hf::convert_hf_ct_to_lbc(&ckpt.dir, &ckpt.donor, &artifact)?;
    if stats.quant_scheme != modules.primary_scheme() {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "the synthetic checkpoint converted as {:?}, not {:?}",
            stats.quant_scheme,
            modules.primary_scheme()
        )));
    }
    Ok(artifact)
}

/// A Q4_0 artifact: the control for "an existing scheme is unaffected".
pub fn write_q4_0_artifact(dir: &Path) -> PathBuf {
    let bytes = lumen_format::test_model::generate_test_model_q4_0(
        &lumen_format::test_model::TestModelQ4Config {
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            hidden_dim: HID as u32,
            intermediate_dim: INTER as u32,
            vocab_size: VOCAB as u32,
            max_seq_len: 64,
            seed: 7,
        },
    );
    let path = dir.join("q4_0.lbc");
    std::fs::write(&path, bytes).unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_ct::HfCtCheckpoint;
    use lumen_format::serving_rules::{scheme_has_no_serving_kernels, unservable_scheme};
    use lumen_format::LbcFile;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-testckpt-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// One layer slice's bytes, read out of the artifact.
    fn slice_bytes(
        artifact: &Path,
        pick: fn(&lumen_format::SubtensorOffsets) -> &lumen_format::TensorSlice,
    ) -> (Vec<u8>, QuantScheme) {
        let lbc = LbcFile::open(artifact).unwrap();
        let file = std::fs::read(artifact).unwrap();
        let layer = &lbc.layer_indices[0];
        let slice = pick(&layer.subtensors);
        let begin = layer.layer_offset_bytes + slice.offset;
        (
            file[begin as usize..(begin + slice.length) as usize].to_vec(),
            slice.quant,
        )
    }

    #[test]
    fn the_synthetic_floating_point_weights_are_all_in_one_binade() {
        let mut seed = 1u64;
        for pair in bf16_bytes(256, &mut seed).chunks_exact(2) {
            let bits = u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16;
            let value = f32::from_bits(bits);
            assert!((1.0..2.0).contains(&value), "{value}");
        }
    }

    #[test]
    fn an_fp8_only_checkpoint_converts_with_fp8_as_its_primary_scheme() {
        let dir = temp_dir("fp8-only");
        let ckpt = write_checkpoint(&dir, Modules::Fp8Only);
        let artifact = dir.join("model.lbc");
        let stats =
            crate::convert_hf::convert_hf_ct_to_lbc(&ckpt.dir, &ckpt.donor, &artifact).unwrap();
        assert_eq!(stats.quant_scheme, QuantScheme::Fp8E4M3);

        // The MLP gate carries its planes verbatim — the E4M3 weight bytes
        // then the per-tensor scale; the role has no row permutation.
        let source = HfCtCheckpoint::open(&ckpt.dir).unwrap();
        let mut expected = source
            .tensor_bytes("model.layers.0.mlp.gate_proj.weight")
            .unwrap();
        expected.extend_from_slice(
            &source
                .tensor_bytes("model.layers.0.mlp.gate_proj.weight_scale")
                .unwrap(),
        );
        let (bytes, quant) = slice_bytes(&artifact, |st| &st.w_gate);
        assert_eq!(quant, QuantScheme::Fp8E4M3);
        assert_eq!(bytes, expected);

        let lbc = LbcFile::open(&artifact).unwrap();
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
        assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));
    }

    #[test]
    fn one_fp8_projection_makes_fp8_an_int4_checkpoint_s_primary_scheme() {
        let dir = temp_dir("int4-one-fp8");
        let artifact = write_artifact(&dir, Modules::Int4WithOneFp8).unwrap();
        let lbc = LbcFile::open(&artifact).unwrap();
        // Every other projection is INT4, and the one FP8 slice still names
        // the header — so the header alone refuses the artifact.
        assert_eq!(
            slice_bytes(&artifact, |st| &st.w_gate).1,
            QuantScheme::CtInt4G32
        );
        assert_eq!(
            slice_bytes(&artifact, |st| st.ssm_out.as_ref().unwrap()).1,
            QuantScheme::Fp8E4M3
        );
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
        assert!(scheme_has_no_serving_kernels(QuantScheme::Fp8E4M3));
        assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));
    }

    #[test]
    fn an_nvfp4_head_makes_nvfp4_an_int4_checkpoint_s_primary_scheme() {
        let dir = temp_dir("int4-nvfp4-head");
        let artifact = write_artifact(&dir, Modules::Int4WithNvfp4Head).unwrap();
        let lbc = LbcFile::open(&artifact).unwrap();
        // The body is INT4 throughout and the head alone is planar, but the
        // primary is what a reader checks before it reads anything: it names
        // the planar scheme, so a reader that does not know that tag refuses
        // the file at its header instead of reading the head as a float.
        assert_eq!(
            slice_bytes(&artifact, |st| &st.w_gate).1,
            QuantScheme::CtInt4G32
        );
        assert_eq!(lbc.header.output_proj.quant, QuantScheme::Nvfp4);
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::Nvfp4);
        assert!(scheme_has_no_serving_kernels(QuantScheme::Nvfp4));
        assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Nvfp4));
    }
}
