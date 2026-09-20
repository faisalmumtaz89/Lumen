//! Synthetic ModelOpt checkpoint generator for end-to-end testing.
//!
//! Writes the smallest checkpoint this importer accepts — one GDN layer with
//! NVFP4 MLP modules, FP8 GDN projections and an NVFP4 head — plus the donor
//! GGUF it takes its hyperparameters from, and converts the pair to an
//! `.lbc`. The counterpart of [`lumen_format::test_model`] for the ModelOpt
//! import path, so the binaries' admission tests share one source.

use std::path::{Path, PathBuf};

use crate::convert::ConvertError;

const HID: usize = 64;
const INTER: usize = 64;
const VOCAB: usize = 32;
const VH: usize = 2;
const KH: usize = 1;
const GHD: usize = 32;
const QKV_ROWS: usize = 2 * KH * GHD + VH * GHD;

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
    // Small finite values: 0x3C00-ish exponents keep every weight around 1.
    (0..len)
        .flat_map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mantissa = ((*seed >> 40) & 0x7F) as u16;
            (0x3F80u16 | mantissa).to_le_bytes()
        })
        .collect()
}

/// One safetensors shard: `u64` header length, the JSON header, the data.
fn shard_bytes(entries: &[(String, &str, Vec<u64>, Vec<u8>)]) -> (Vec<u8>, String) {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        let begin = data.len();
        data.extend_from_slice(bytes);
        let shape = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{begin},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    (out, header)
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

/// A one-layer ModelOpt checkpoint — NVFP4 MLP and head, FP8 GDN
/// projections — converted to an `.lbc`. Returns the artifact's path.
pub fn write_nvfp4_artifact(dir: &Path) -> Result<PathBuf, ConvertError> {
    let mut seed = 42u64;
    let mut entries: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut quantized: Vec<String> = Vec::new();

    let mut nvfp4 = |base: &str, n: usize, k: usize, seed: &mut u64| {
        entries.push((
            format!("{base}.weight"),
            "U8",
            vec![n as u64, k as u64 / 2],
            rand_bytes(n * k / 2, seed),
        ));
        entries.push((
            format!("{base}.weight_scale"),
            "F8_E4M3",
            vec![n as u64, k as u64 / 16],
            // 0x38 is E4M3 1.0: every block scale is finite and non-zero.
            vec![0x38u8; n * (k / 16)],
        ));
        entries.push((
            format!("{base}.weight_scale_2"),
            "F32",
            vec![],
            1.5e-4f32.to_le_bytes().to_vec(),
        ));
        quantized.push(format!(
            "\"{base}\":{{\"quant_algo\":\"NVFP4\",\"group_size\":16}}"
        ));
    };
    for (base, n, k) in [
        ("model.layers.0.mlp.gate_proj", INTER, HID),
        ("model.layers.0.mlp.up_proj", INTER, HID),
        ("model.layers.0.mlp.down_proj", HID, INTER),
        ("lm_head", VOCAB, HID),
    ] {
        nvfp4(base, n, k, &mut seed);
    }

    let mut fp8 = |base: &str, n: usize, k: usize, seed: &mut u64| {
        entries.push((
            format!("{base}.weight"),
            "F8_E4M3",
            vec![n as u64, k as u64],
            rand_bytes(n * k, seed),
        ));
        entries.push((
            format!("{base}.weight_scale"),
            "F32",
            vec![],
            9.7e-4f32.to_le_bytes().to_vec(),
        ));
        quantized.push(format!("\"{base}\":{{\"quant_algo\":\"FP8\"}}"));
    };
    for (base, n, k) in [
        ("model.layers.0.linear_attn.in_proj_qkv", QKV_ROWS, HID),
        ("model.layers.0.linear_attn.in_proj_z", VH * GHD, HID),
        ("model.layers.0.linear_attn.out_proj", HID, VH * GHD),
    ] {
        fp8(base, n, k, &mut seed);
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
    let (shard, _) = shard_bytes(&entries);
    std::fs::write(ckpt.join("model-00001.safetensors"), shard).unwrap();
    let weight_map = entries
        .iter()
        .map(|(n, ..)| format!("\"{n}\":\"model-00001.safetensors\""))
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        ckpt.join("model.safetensors.index.json"),
        format!("{{\"weight_map\":{{{weight_map}}}}}"),
    )
    .unwrap();
    std::fs::write(
        ckpt.join("config.json"),
        format!(
            "{{\"quantization_config\":{{\"quant_method\":\"modelopt\"}},\
              \"text_config\":{{\"hidden_size\":{HID},\"num_hidden_layers\":1,\
              \"intermediate_size\":{INTER},\"vocab_size\":{VOCAB}}}}}"
        ),
    )
    .unwrap();
    std::fs::write(
        ckpt.join("hf_quant_config.json"),
        format!(
            "{{\"quantization\":{{\"quant_algo\":\"MIXED_PRECISION\",\
              \"quantized_layers\":{{{}}}}}}}",
            quantized.join(",")
        ),
    )
    .unwrap();

    let donor = dir.join("donor.gguf");
    write_donor(&donor);
    let artifact = dir.join("nvfp4.lbc");
    let stats = crate::convert_hf::convert_hf_ct_to_lbc(&ckpt, &donor, &artifact)?;
    if stats.quant_scheme != lumen_format::QuantScheme::Nvfp4 {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "the synthetic checkpoint converted as {:?}, not Nvfp4",
            stats.quant_scheme
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
