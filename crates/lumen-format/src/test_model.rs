//! Synthetic test model generator for E2E testing.
//!
//! Generates a tiny LBC file with deterministic random weights.
//! Configuration: 2 layers, 2 heads, hidden_dim=8, vocab_size=32.

use crate::header::LbcHeader;
use crate::hyperparams::{ModelHyperparams, RopeParams};
use crate::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use crate::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
use crate::rng::WeightRng;
use crate::writer::{write_lbc, GlobalTensors};

/// Default test model configuration.
pub struct TestModelConfig {
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub seed: u64,
}

impl Default for TestModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_size: 32,
            max_seq_len: 64,
            seed: 42,
        }
    }
}

/// Generate a complete synthetic LBC file as bytes.
pub fn generate_test_model(config: &TestModelConfig) -> Vec<u8> {
    let mut rng = WeightRng::new(config.seed);

    let hidden = config.hidden_dim as usize;
    let inter = config.intermediate_dim as usize;
    let heads = config.num_heads as usize;
    let kv_heads = config.num_kv_heads as usize;
    let head_dim = config.head_dim as usize;
    let vocab = config.vocab_size as usize;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    // Global tensors
    let embedding = rng.gen_f32_bytes(vocab * hidden);
    let final_norm = rng.gen_norm_bytes(hidden);
    let output_proj = rng.gen_f32_bytes(vocab * hidden);

    // Layer tensors
    let mut layer_blobs = Vec::new();
    let mut layer_indices = Vec::new();

    for _ in 0..config.num_layers {
        let mut blob = Vec::new();
        let mut offset = 0u64;

        // Helper to append tensor data and record slice
        let mut add_tensor = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice {
                offset,
                length: len,
                quant: QuantScheme::F32,
            };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let wq = add_tensor(rng.gen_f32_bytes(q_dim * hidden));
        let wk = add_tensor(rng.gen_f32_bytes(kv_dim * hidden));
        let wv = add_tensor(rng.gen_f32_bytes(kv_dim * hidden));
        let wo = add_tensor(rng.gen_f32_bytes(hidden * q_dim));
        let w_gate = add_tensor(rng.gen_f32_bytes(inter * hidden));
        let w_up = add_tensor(rng.gen_f32_bytes(inter * hidden));
        let w_down = add_tensor(rng.gen_f32_bytes(hidden * inter));
        let attn_norm = add_tensor(rng.gen_norm_bytes(hidden));
        let ffn_norm = add_tensor(rng.gen_norm_bytes(hidden));

        let subtensors = SubtensorOffsets {
            wq, wk, wv, wo,
            bq: None, bk: None, bv: None,
            w_gate, w_up, w_down,
            attn_norm, ffn_norm,
            router_weight: None,
            experts: None,
            shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
            attn_gate: None, attn_post_norm: None,
            ssm_a: None, ssm_conv1d: None, ssm_dt: None,
            ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
            attn_q_norm: None, attn_k_norm: None,
            ffn_gate_inp_shexp: None,
            layer_type: None,
        };

        layer_indices.push(LayerIndex {
            layer_offset_bytes: 0, // writer will fix
            layer_length_bytes: blob.len() as u64,
            subtensors,
        });
        layer_blobs.push(blob);
    }

    // Build header
    let hp = ModelHyperparams {
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        hidden_dim: config.hidden_dim,
        intermediate_dim: config.intermediate_dim,
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        rope_params: Some(RopeParams::default()),
        num_experts: None,
        num_active_experts: None,
        norm_eps: 1e-5,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::F32,
        group_size: QuantGroupSize::PerTensor,
        block_byte_size: 4,
        scale_offset_in_block: None,
    };
    let header = LbcHeader::new(hp, qd);

    let globals = GlobalTensors {
        embedding,
        final_norm,
        output_proj,
    };

    let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();

    let mut out = Vec::new();
    write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs)
        .expect("failed to write test model");

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::LbcFile;
    use std::path::PathBuf;

    #[test]
    fn generates_valid_lbc() {
        let config = TestModelConfig::default();
        let data = generate_test_model(&config);

        let lbc = LbcFile::from_bytes(&data, PathBuf::from("test.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 8);
        assert_eq!(lbc.header.hyperparams.vocab_size, 32);
        assert_eq!(lbc.layer_indices.len(), 2);

        // Verify layer indices are valid
        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }
    }

    #[test]
    fn deterministic() {
        let config = TestModelConfig::default();
        let data1 = generate_test_model(&config);
        let data2 = generate_test_model(&config);
        assert_eq!(data1, data2, "same seed should produce identical output");
    }
}
