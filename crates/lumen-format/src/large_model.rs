//! Large synthetic model generation for benchmarks.
//!
//! Uses [`StreamingLbcWriter`] for O(1 layer + globals) peak memory.

use crate::header::LbcHeader;
use crate::hyperparams::{ModelHyperparams, RopeParams};
use crate::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use crate::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
use crate::rng::WeightRng;
use crate::streaming_writer::{LayerShape, StreamingLbcWriter};
use crate::writer::GlobalTensors;
use std::io::{self, Write};

/// Configuration for generating large synthetic models.
#[derive(Debug, Clone)]
pub struct LargeModelConfig {
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

impl LargeModelConfig {
    /// ~256 MB model: 20 layers, ~12 MB/layer + ~16 MB globals.
    pub fn bench_256mb() -> Self {
        Self {
            num_layers: 20,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            hidden_dim: 512,
            intermediate_dim: 1408,
            vocab_size: 4096,
            max_seq_len: 512,
            seed: 42,
        }
    }

    /// ~1 GB model: 20 layers, ~45 MB/layer + ~63 MB globals.
    pub fn bench_1gb() -> Self {
        Self {
            num_layers: 20,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            hidden_dim: 1024,
            intermediate_dim: 2816,
            vocab_size: 8000,
            max_seq_len: 1024,
            seed: 42,
        }
    }

    /// ~4 GB model: 24 layers, ~150 MB/layer + ~500 MB globals.
    pub fn bench_4gb() -> Self {
        Self {
            num_layers: 24,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            hidden_dim: 2048,
            intermediate_dim: 5504,
            vocab_size: 32000,
            max_seq_len: 2048,
            seed: 42,
        }
    }

    /// Llama-7B-equivalent: 32 layers, ~676 MB/layer, ~22 GB total.
    pub fn llama_7b() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            hidden_dim: 4096,
            intermediate_dim: 11008,
            vocab_size: 32000,
            max_seq_len: 4096,
            seed: 42,
        }
    }

    /// Size of a single layer blob in bytes (all F32).
    pub fn layer_blob_size(&self) -> u64 {
        let hidden = self.hidden_dim as u64;
        let inter = self.intermediate_dim as u64;
        let q_dim = (self.num_heads * self.head_dim) as u64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;

        let wq = q_dim * hidden;
        let wk = kv_dim * hidden;
        let wv = kv_dim * hidden;
        let wo = hidden * q_dim;
        let w_gate = inter * hidden;
        let w_up = inter * hidden;
        let w_down = hidden * inter;
        let attn_norm = hidden;
        let ffn_norm = hidden;

        (wq + wk + wv + wo + w_gate + w_up + w_down + attn_norm + ffn_norm) * 4 // F32 = 4 bytes
    }

    /// Estimated total file size including header, index, globals, and alignment.
    pub fn estimated_file_size(&self) -> u64 {
        let hidden = self.hidden_dim as u64;
        let vocab = self.vocab_size as u64;

        let globals = (vocab * hidden + hidden + vocab * hidden) * 4; // embedding + norm + output_proj
        let layers = self.layer_blob_size() * self.num_layers as u64;

        // Approximate: header+index ~4KB, alignment padding ~128KB per layer
        let overhead = 4096 + self.num_layers as u64 * 128 * 1024;

        globals + layers + overhead
    }

    /// Build model hyperparams from this config.
    pub fn to_hyperparams(&self) -> ModelHyperparams {
        ModelHyperparams {
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            hidden_dim: self.hidden_dim,
            intermediate_dim: self.intermediate_dim,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            rope_params: Some(RopeParams::default()),
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-5,
        }
    }
}

/// Pre-compute a `LayerShape` from config without generating data.
pub fn compute_layer_shape(config: &LargeModelConfig) -> LayerShape {
    let hidden = config.hidden_dim as usize;
    let inter = config.intermediate_dim as usize;
    let q_dim = (config.num_heads * config.head_dim) as usize;
    let kv_dim = (config.num_kv_heads * config.head_dim) as usize;

    let sizes = [
        q_dim * hidden,   // wq
        kv_dim * hidden,  // wk
        kv_dim * hidden,  // wv
        hidden * q_dim,   // wo
        inter * hidden,   // w_gate
        inter * hidden,   // w_up
        hidden * inter,   // w_down
        hidden,           // attn_norm
        hidden,           // ffn_norm
    ];

    let mut offset = 0u64;
    let mut slices = Vec::with_capacity(9);
    for &n in &sizes {
        let len = (n * 4) as u64;
        slices.push(TensorSlice {
            offset,
            length: len,
            quant: QuantScheme::F32,
        });
        offset += len;
    }

    let blob_size = offset;
    let subtensors = SubtensorOffsets {
        wq: slices[0], wk: slices[1], wv: slices[2], wo: slices[3],
        bq: None, bk: None, bv: None,
        w_gate: slices[4], w_up: slices[5], w_down: slices[6],
        attn_norm: slices[7], ffn_norm: slices[8],
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

    LayerShape {
        blob_size,
        index: LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors,
        },
    }
}

/// Generate a layer blob using deterministic random weights.
fn generate_layer_blob(config: &LargeModelConfig, rng: &mut WeightRng) -> Vec<u8> {
    let hidden = config.hidden_dim as usize;
    let inter = config.intermediate_dim as usize;
    let q_dim = (config.num_heads * config.head_dim) as usize;
    let kv_dim = (config.num_kv_heads * config.head_dim) as usize;

    let blob_size = config.layer_blob_size() as usize;
    let mut blob = Vec::with_capacity(blob_size);

    blob.extend_from_slice(&rng.gen_f32_bytes(q_dim * hidden));
    blob.extend_from_slice(&rng.gen_f32_bytes(kv_dim * hidden));
    blob.extend_from_slice(&rng.gen_f32_bytes(kv_dim * hidden));
    blob.extend_from_slice(&rng.gen_f32_bytes(hidden * q_dim));
    blob.extend_from_slice(&rng.gen_f32_bytes(inter * hidden));
    blob.extend_from_slice(&rng.gen_f32_bytes(inter * hidden));
    blob.extend_from_slice(&rng.gen_f32_bytes(hidden * inter));
    blob.extend_from_slice(&rng.gen_norm_bytes(hidden));
    blob.extend_from_slice(&rng.gen_norm_bytes(hidden));

    blob
}

/// Streaming generation: O(1 layer + globals) peak memory.
pub fn generate_large_model<W: Write>(w: W, config: &LargeModelConfig) -> io::Result<()> {
    let mut rng = WeightRng::new(config.seed);
    let hidden = config.hidden_dim as usize;
    let vocab = config.vocab_size as usize;

    // Generate globals (these stay in memory)
    let globals = GlobalTensors {
        embedding: rng.gen_f32_bytes(vocab * hidden),
        final_norm: rng.gen_norm_bytes(hidden),
        output_proj: rng.gen_f32_bytes(vocab * hidden),
    };

    let shape = compute_layer_shape(config);
    let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

    let hp = config.to_hyperparams();
    let qd = QuantizationDescriptor {
        scheme: QuantScheme::F32,
        group_size: QuantGroupSize::PerTensor,
        block_byte_size: 4,
        scale_offset_in_block: None,
    };
    let header = LbcHeader::new(hp, qd);

    let mut sw = StreamingLbcWriter::begin(w, &header, &layer_shapes, &globals)?;
    for _ in 0..config.num_layers {
        let blob = generate_layer_blob(config, &mut rng);
        sw.write_layer(&blob)?;
    }
    sw.finish()?;

    Ok(())
}

/// Format a byte count as a human-readable string.
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::LbcFile;
    use std::path::PathBuf;

    #[test]
    fn layer_blob_size_matches_compute_layer_shape() {
        for config in [
            LargeModelConfig::bench_256mb(),
            LargeModelConfig::bench_1gb(),
            LargeModelConfig::bench_4gb(),
            LargeModelConfig::llama_7b(),
        ] {
            let shape = compute_layer_shape(&config);
            assert_eq!(
                shape.blob_size,
                config.layer_blob_size(),
                "mismatch for config with {} layers",
                config.num_layers
            );
        }
    }

    #[test]
    fn llama_7b_layer_size() {
        let config = LargeModelConfig::llama_7b();
        // Manual calculation for Llama-7B (F32):
        // hidden=4096, inter=11008, q_dim=4096, kv_dim=4096
        // wq: 4096*4096 = 16M elements
        // wk: 4096*4096 = 16M
        // wv: 4096*4096 = 16M
        // wo: 4096*4096 = 16M
        // w_gate: 11008*4096 = 45M
        // w_up: 11008*4096 = 45M
        // w_down: 4096*11008 = 45M
        // attn_norm: 4096
        // ffn_norm: 4096
        // Total elements = 16M*4 + 45M*3 + 8192 = 64M + 135M + 8192 ≈ 199M
        // Total bytes = elements * 4
        let expected_elements: u64 = 4 * 4096 * 4096 + 3 * 11008 * 4096 + 2 * 4096;
        let expected_bytes = expected_elements * 4;
        assert_eq!(config.layer_blob_size(), expected_bytes);
    }

    #[test]
    fn generate_small_roundtrip() {
        // Use a tiny config for fast test
        let config = LargeModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_size: 32,
            max_seq_len: 64,
            seed: 42,
        };

        let mut out = Vec::new();
        generate_large_model(&mut out, &config).unwrap();

        let lbc = LbcFile::from_bytes(&out, PathBuf::from("test.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.layer_indices.len(), 2);
        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }
    }

    #[test]
    fn estimated_file_size_reasonable() {
        // 256MB config
        let config = LargeModelConfig::bench_256mb();
        let est = config.estimated_file_size();
        assert!(est > 100_000_000, "256mb config: estimated {est} too small");
        assert!(est < 500_000_000, "256mb config: estimated {est} too large");

        // 1GB config
        let config = LargeModelConfig::bench_1gb();
        let est = config.estimated_file_size();
        assert!(est > 500_000_000, "1gb config: estimated {est} too small");
        assert!(est < 2_000_000_000, "1gb config: estimated {est} too large");

        // 4GB config
        let config = LargeModelConfig::bench_4gb();
        let est = config.estimated_file_size();
        assert!(est > 2_000_000_000, "4gb config: estimated {est} too small");
        assert!(est < 8_000_000_000, "4gb config: estimated {est} too large");
    }

    #[test]
    fn format_size_works() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KiB");
        assert_eq!(format_size(1024 * 1024), "1.0 MiB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GiB");
    }
}
