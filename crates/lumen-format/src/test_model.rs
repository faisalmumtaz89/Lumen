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
        rotary_dim: None, rope_neox: false,
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
    write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs, None)
        .expect("failed to write test model");

    out
}

// ---------------------------------------------------------------------------
// Q8_0 test model
// ---------------------------------------------------------------------------

/// Configuration for a Q8_0-quantized test model.
/// All dimensions MUST be multiples of 32 (Q8_0 block size).
pub struct TestModelQ8Config {
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

impl Default for TestModelQ8Config {
    fn default() -> Self {
        Self {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            max_seq_len: 512,
            seed: 42,
        }
    }
}

/// Generate a Q8_0-quantized synthetic LBC file.
/// Projection weights are Q8_0 (34 bytes per 32 elements).
/// Norm weights are F32 (4 bytes per element).
/// Global tensors: embedding Q8_0, output_proj Q8_0, final_norm F32.
pub fn generate_test_model_q8_0(config: &TestModelQ8Config) -> Vec<u8> {
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
    let embedding = rng.gen_q8_0_bytes(vocab * hidden);
    let final_norm = rng.gen_norm_bytes(hidden);
    let output_proj = rng.gen_q8_0_bytes(vocab * hidden);

    // Layer tensors
    let mut layer_blobs = Vec::new();
    let mut layer_indices = Vec::new();

    for _ in 0..config.num_layers {
        let mut blob = Vec::new();
        let mut offset = 0u64;

        let mut add_q8 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::Q8_0 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let wq = add_q8(rng.gen_q8_0_bytes(q_dim * hidden));
        let wk = add_q8(rng.gen_q8_0_bytes(kv_dim * hidden));
        let wv = add_q8(rng.gen_q8_0_bytes(kv_dim * hidden));
        let wo = add_q8(rng.gen_q8_0_bytes(hidden * q_dim));
        let w_gate = add_q8(rng.gen_q8_0_bytes(inter * hidden));
        let w_up = add_q8(rng.gen_q8_0_bytes(inter * hidden));
        let w_down = add_q8(rng.gen_q8_0_bytes(hidden * inter));

        // Norms are always F32
        let mut add_f32 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::F32 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let attn_norm = add_f32(rng.gen_norm_bytes(hidden));
        let ffn_norm = add_f32(rng.gen_norm_bytes(hidden));

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
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors,
        });
        layer_blobs.push(blob);
    }

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
        rotary_dim: None, rope_neox: false,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::Q8_0,
        group_size: QuantGroupSize::Group(32),
        block_byte_size: 34,
        scale_offset_in_block: Some(0),
    };
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::Q8_0;
    header.output_proj.quant = QuantScheme::Q8_0;
    header.final_norm.quant = QuantScheme::F32;

    let globals = GlobalTensors { embedding, final_norm, output_proj };
    let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();

    let mut out = Vec::new();
    write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs, None)
        .expect("failed to write Q8_0 test model");
    out
}

// ---------------------------------------------------------------------------
// F16 test model
// ---------------------------------------------------------------------------

/// Configuration for an F16 (half-precision) test model.
pub struct TestModelF16Config {
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

impl Default for TestModelF16Config {
    fn default() -> Self {
        Self {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 32,
            intermediate_dim: 64,
            vocab_size: 256,
            max_seq_len: 512,
            seed: 42,
        }
    }
}

/// Generate an F16-quantized synthetic LBC file.
/// Projection weights are F16 (2 bytes per element).
/// Norm weights are F32 (4 bytes per element).
/// Global tensors: embedding F16, output_proj F16, final_norm F32.
pub fn generate_test_model_f16(config: &TestModelF16Config) -> Vec<u8> {
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
    let embedding = rng.gen_f16_bytes(vocab * hidden);
    let final_norm = rng.gen_norm_bytes(hidden);
    let output_proj = rng.gen_f16_bytes(vocab * hidden);

    // Layer tensors
    let mut layer_blobs = Vec::new();
    let mut layer_indices = Vec::new();

    for _ in 0..config.num_layers {
        let mut blob = Vec::new();
        let mut offset = 0u64;

        let mut add_f16 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::F16 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let wq = add_f16(rng.gen_f16_bytes(q_dim * hidden));
        let wk = add_f16(rng.gen_f16_bytes(kv_dim * hidden));
        let wv = add_f16(rng.gen_f16_bytes(kv_dim * hidden));
        let wo = add_f16(rng.gen_f16_bytes(hidden * q_dim));
        let w_gate = add_f16(rng.gen_f16_bytes(inter * hidden));
        let w_up = add_f16(rng.gen_f16_bytes(inter * hidden));
        let w_down = add_f16(rng.gen_f16_bytes(hidden * inter));

        // Norms are always F32
        let mut add_f32 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::F32 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let attn_norm = add_f32(rng.gen_norm_bytes(hidden));
        let ffn_norm = add_f32(rng.gen_norm_bytes(hidden));

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
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors,
        });
        layer_blobs.push(blob);
    }

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
        rotary_dim: None, rope_neox: false,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::F16,
        group_size: QuantGroupSize::PerTensor,
        block_byte_size: 2,
        scale_offset_in_block: None,
    };
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::F16;
    header.output_proj.quant = QuantScheme::F16;
    header.final_norm.quant = QuantScheme::F32;

    let globals = GlobalTensors { embedding, final_norm, output_proj };
    let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();

    let mut out = Vec::new();
    write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs, None)
        .expect("failed to write F16 test model");
    out
}

// ---------------------------------------------------------------------------
// Q4_0 test model
// ---------------------------------------------------------------------------

/// Configuration for a Q4_0-quantized test model.
/// All dimensions MUST be multiples of 32 (Q4_0 block size).
pub struct TestModelQ4Config {
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

impl Default for TestModelQ4Config {
    fn default() -> Self {
        Self {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 32,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            max_seq_len: 512,
            seed: 42,
        }
    }
}

/// Generate a Q4_0-quantized synthetic LBC file.
/// Projection weights are Q4_0 (18 bytes per 32 elements).
/// Norm weights are F32 (4 bytes per element).
/// Global tensors: embedding Q4_0, output_proj Q4_0, final_norm F32.
pub fn generate_test_model_q4_0(config: &TestModelQ4Config) -> Vec<u8> {
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
    let embedding = rng.gen_q4_0_bytes(vocab * hidden);
    let final_norm = rng.gen_norm_bytes(hidden);
    let output_proj = rng.gen_q4_0_bytes(vocab * hidden);

    // Layer tensors
    let mut layer_blobs = Vec::new();
    let mut layer_indices = Vec::new();

    for _ in 0..config.num_layers {
        let mut blob = Vec::new();
        let mut offset = 0u64;

        let mut add_q4 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::Q4_0 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let wq = add_q4(rng.gen_q4_0_bytes(q_dim * hidden));
        let wk = add_q4(rng.gen_q4_0_bytes(kv_dim * hidden));
        let wv = add_q4(rng.gen_q4_0_bytes(kv_dim * hidden));
        let wo = add_q4(rng.gen_q4_0_bytes(hidden * q_dim));
        let w_gate = add_q4(rng.gen_q4_0_bytes(inter * hidden));
        let w_up = add_q4(rng.gen_q4_0_bytes(inter * hidden));
        let w_down = add_q4(rng.gen_q4_0_bytes(hidden * inter));

        // Norms are always F32
        let mut add_f32 = |data: Vec<u8>| -> TensorSlice {
            let len = data.len() as u64;
            let slice = TensorSlice { offset, length: len, quant: QuantScheme::F32 };
            blob.extend_from_slice(&data);
            offset += len;
            slice
        };

        let attn_norm = add_f32(rng.gen_norm_bytes(hidden));
        let ffn_norm = add_f32(rng.gen_norm_bytes(hidden));

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
            layer_offset_bytes: 0,
            layer_length_bytes: blob.len() as u64,
            subtensors,
        });
        layer_blobs.push(blob);
    }

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
        rotary_dim: None, rope_neox: false,
    };
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::Q4_0,
        group_size: QuantGroupSize::Group(32),
        block_byte_size: 18,
        scale_offset_in_block: Some(0),
    };
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::Q4_0;
    header.output_proj.quant = QuantScheme::Q4_0;
    header.final_norm.quant = QuantScheme::F32;

    let globals = GlobalTensors { embedding, final_norm, output_proj };
    let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();

    let mut out = Vec::new();
    write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs, None)
        .expect("failed to write Q4_0 test model");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::LbcFile;
    use std::path::PathBuf;

    // ---- F32 tests (original) ----

    #[test]
    fn generates_valid_lbc() {
        let config = TestModelConfig::default();
        let data = generate_test_model(&config);

        let lbc = LbcFile::from_bytes(&data, PathBuf::from("test.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 8);
        assert_eq!(lbc.header.hyperparams.vocab_size, 32);
        assert_eq!(lbc.layer_indices.len(), 2);

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

    // ---- Q8_0 tests ----

    #[test]
    fn q8_0_generates_valid_lbc() {
        let config = TestModelQ8Config::default();
        let data = generate_test_model_q8_0(&config);

        let lbc = LbcFile::from_bytes(&data, PathBuf::from("test_q8.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 64);
        assert_eq!(lbc.header.hyperparams.vocab_size, 256);
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::Q8_0);
        assert_eq!(lbc.layer_indices.len(), 2);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
            // Verify projection weights are Q8_0
            assert_eq!(idx.subtensors.wq.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.wk.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.wv.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.wo.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.w_gate.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.w_up.quant, QuantScheme::Q8_0);
            assert_eq!(idx.subtensors.w_down.quant, QuantScheme::Q8_0);
            // Verify norms are F32
            assert_eq!(idx.subtensors.attn_norm.quant, QuantScheme::F32);
            assert_eq!(idx.subtensors.ffn_norm.quant, QuantScheme::F32);
        }
    }

    #[test]
    fn q8_0_deterministic() {
        let config = TestModelQ8Config::default();
        let data1 = generate_test_model_q8_0(&config);
        let data2 = generate_test_model_q8_0(&config);
        assert_eq!(data1, data2, "Q8_0: same seed should produce identical output");
    }

    // ---- F16 tests ----

    #[test]
    fn f16_generates_valid_lbc() {
        let config = TestModelF16Config::default();
        let data = generate_test_model_f16(&config);

        let lbc = LbcFile::from_bytes(&data, PathBuf::from("test_f16.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 32);
        assert_eq!(lbc.header.hyperparams.vocab_size, 256);
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::F16);
        assert_eq!(lbc.layer_indices.len(), 2);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
            // Verify projection weights are F16
            assert_eq!(idx.subtensors.wq.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.wk.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.wv.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.wo.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.w_gate.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.w_up.quant, QuantScheme::F16);
            assert_eq!(idx.subtensors.w_down.quant, QuantScheme::F16);
            // Verify norms are F32
            assert_eq!(idx.subtensors.attn_norm.quant, QuantScheme::F32);
            assert_eq!(idx.subtensors.ffn_norm.quant, QuantScheme::F32);
        }

        // Verify F16 byte sizes: each F16 element = 2 bytes
        let hidden = config.hidden_dim as usize;
        let heads = config.num_heads as usize;
        let head_dim = config.head_dim as usize;
        let q_dim = heads * head_dim;
        let expected_wq_bytes = q_dim * hidden * 2;
        assert_eq!(
            lbc.layer_indices[0].subtensors.wq.length,
            expected_wq_bytes as u64,
            "wq F16 byte size mismatch"
        );
    }

    #[test]
    fn f16_deterministic() {
        let config = TestModelF16Config::default();
        let data1 = generate_test_model_f16(&config);
        let data2 = generate_test_model_f16(&config);
        assert_eq!(data1, data2, "F16: same seed should produce identical output");
    }

    // ---- Q4_0 tests ----

    #[test]
    fn q4_0_generates_valid_lbc() {
        let config = TestModelQ4Config::default();
        let data = generate_test_model_q4_0(&config);

        let lbc = LbcFile::from_bytes(&data, PathBuf::from("test_q4.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 64);
        assert_eq!(lbc.header.hyperparams.vocab_size, 256);
        assert_eq!(lbc.header.quantization.scheme, QuantScheme::Q4_0);
        assert_eq!(lbc.layer_indices.len(), 2);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
            // Verify projection weights are Q4_0
            assert_eq!(idx.subtensors.wq.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.wk.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.wv.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.wo.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.w_gate.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.w_up.quant, QuantScheme::Q4_0);
            assert_eq!(idx.subtensors.w_down.quant, QuantScheme::Q4_0);
            // Verify norms are F32
            assert_eq!(idx.subtensors.attn_norm.quant, QuantScheme::F32);
            assert_eq!(idx.subtensors.ffn_norm.quant, QuantScheme::F32);
        }

        // Verify Q4_0 byte sizes: 18 bytes per 32 elements
        let hidden = config.hidden_dim as usize;
        let heads = config.num_heads as usize;
        let head_dim = config.head_dim as usize;
        let q_dim = heads * head_dim;
        let n_elements = q_dim * hidden;
        let expected_wq_bytes = (n_elements / 32) * 18;
        assert_eq!(
            lbc.layer_indices[0].subtensors.wq.length,
            expected_wq_bytes as u64,
            "wq Q4_0 byte size mismatch"
        );
    }

    #[test]
    fn q4_0_deterministic() {
        let config = TestModelQ4Config::default();
        let data1 = generate_test_model_q4_0(&config);
        let data2 = generate_test_model_q4_0(&config);
        assert_eq!(data1, data2, "Q4_0: same seed should produce identical output");
    }
}
