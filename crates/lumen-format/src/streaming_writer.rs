//! Streaming LBC writer for generating large models without holding all data in memory.
//!
//! Key insight: layer sizes are fully deterministic from config, so all offsets
//! can be pre-computed without generating data. No `Seek` needed — plain `W: Write`.

use crate::crc::crc32;
use crate::header::{GlobalTensorRange, LbcHeader};
use crate::index::LayerIndex;
use crate::writer::{align_up, serialize_header, serialize_layer_indices, write_zeros, GlobalTensors};
use std::io::{self, Write};

/// Pre-computed shape for a single layer blob.
#[derive(Debug, Clone)]
pub struct LayerShape {
    /// Size of the layer blob in bytes.
    pub blob_size: u64,
    /// Layer index with subtensor layout (offsets relative to blob start).
    pub index: LayerIndex,
}

/// Streaming LBC writer that writes header + index + globals first,
/// then accepts layer blobs one at a time. Peak memory: O(1 layer + globals).
pub struct StreamingLbcWriter<W: Write> {
    writer: W,
    fixed_indices: Vec<LayerIndex>,
    layers_written: usize,
    total_layers: usize,
}

impl<W: Write> StreamingLbcWriter<W> {
    /// Compute layout, write header + index + globals. O(globals) memory.
    pub fn begin(
        mut w: W,
        header: &LbcHeader,
        layer_shapes: &[LayerShape],
        global_tensors: &GlobalTensors,
    ) -> io::Result<Self> {
        let num_layers = header.num_layers as usize;
        if layer_shapes.len() != num_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "layer_shapes.len()={} != header.num_layers={}",
                    layer_shapes.len(), num_layers
                ),
            ));
        }

        let alignment: usize = header.alignment.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("alignment {} exceeds platform pointer size", header.alignment),
            )
        })?;

        // --- Phase 1: compute layout (same as writer.rs) ---
        let header_bytes = serialize_header(header);
        let index_bytes = serialize_layer_indices(
            &layer_shapes.iter().map(|s| s.index.clone()).collect::<Vec<_>>(),
        );

        let raw_prefix = header_bytes.len() + index_bytes.len();
        let padding = align_up(raw_prefix, alignment) - raw_prefix;

        let globals_start = raw_prefix + padding;
        let globals_total = global_tensors.embedding.len()
            + global_tensors.final_norm.len()
            + global_tensors.output_proj.len();

        let layers_start = align_up(globals_start + globals_total, alignment);

        // --- Phase 2: fix up header offsets ---
        let mut fixed_header = header.clone();
        fixed_header.layer_index_offset = header_bytes.len() as u64;
        fixed_header.payload_offset = globals_start as u64;

        let mut cursor = globals_start as u64;
        fixed_header.embedding = GlobalTensorRange {
            offset: cursor,
            length: global_tensors.embedding.len() as u64,
            quant: header.embedding.quant,
        };
        cursor += global_tensors.embedding.len() as u64;
        fixed_header.final_norm = GlobalTensorRange {
            offset: cursor,
            length: global_tensors.final_norm.len() as u64,
            quant: header.final_norm.quant,
        };
        cursor += global_tensors.final_norm.len() as u64;
        if fixed_header.weight_tying {
            fixed_header.output_proj = GlobalTensorRange {
                offset: fixed_header.embedding.offset,
                length: fixed_header.embedding.length,
                quant: fixed_header.embedding.quant,
            };
        } else {
            fixed_header.output_proj = GlobalTensorRange {
                offset: cursor,
                length: global_tensors.output_proj.len() as u64,
                quant: header.output_proj.quant,
            };
        }

        // Re-serialize header with correct offsets, compute checksum
        let mut header_bytes = serialize_header(&fixed_header);
        let checksum = crc32(&header_bytes);
        header_bytes[12..16].copy_from_slice(&checksum.to_le_bytes());

        // Fix up layer indices
        let mut fixed_indices: Vec<LayerIndex> = layer_shapes.iter().map(|s| s.index.clone()).collect();
        let mut layer_cursor = layers_start as u64;
        for (i, idx) in fixed_indices.iter_mut().enumerate() {
            idx.layer_offset_bytes = layer_cursor;
            idx.layer_length_bytes = layer_shapes[i].blob_size;
            layer_cursor += layer_shapes[i].blob_size;
            layer_cursor = align_up(layer_cursor as usize, alignment) as u64;
        }
        let index_bytes = serialize_layer_indices(&fixed_indices);

        // --- Phase 3: write header + index + globals ---
        w.write_all(&header_bytes)?;
        w.write_all(&index_bytes)?;
        write_zeros(&mut w, padding)?;

        w.write_all(&global_tensors.embedding)?;
        w.write_all(&global_tensors.final_norm)?;
        w.write_all(&global_tensors.output_proj)?;

        let globals_end = globals_start + globals_total;
        let layers_padding = layers_start - globals_end;
        write_zeros(&mut w, layers_padding)?;

        Ok(Self {
            writer: w,
            fixed_indices,
            layers_written: 0,
            total_layers: num_layers,
        })
    }

    /// Write one layer blob. Called N times in order. O(1 layer) memory.
    pub fn write_layer(&mut self, blob: &[u8]) -> io::Result<()> {
        if self.layers_written >= self.total_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "too many layers: already wrote {}, total is {}",
                    self.layers_written, self.total_layers
                ),
            ));
        }

        let expected_size = self.fixed_indices[self.layers_written].layer_length_bytes;
        if blob.len() as u64 != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "layer {} blob size mismatch: got {}, expected {}",
                    self.layers_written,
                    blob.len(),
                    expected_size
                ),
            ));
        }

        self.writer.write_all(blob)?;

        // Write alignment padding between layers (not after the last one)
        if self.layers_written + 1 < self.total_layers {
            let cur_end = self.fixed_indices[self.layers_written].layer_offset_bytes
                + self.fixed_indices[self.layers_written].layer_length_bytes;
            let next_start = self.fixed_indices[self.layers_written + 1].layer_offset_bytes;
            let padding = (next_start - cur_end) as usize;
            write_zeros(&mut self.writer, padding)?;
        }

        self.layers_written += 1;
        Ok(())
    }

    /// Verify all layers written, return inner writer.
    pub fn finish(self) -> io::Result<W> {
        if self.layers_written != self.total_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "incomplete write: wrote {} layers, expected {}",
                    self.layers_written, self.total_layers
                ),
            ));
        }
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::LbcFile;
    use crate::test_model::{generate_test_model, TestModelConfig};
    use crate::hyperparams::{ModelHyperparams, RopeParams};
    use crate::index::{SubtensorOffsets, TensorSlice};
    use crate::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
    use crate::rng::WeightRng;
    use std::path::PathBuf;

    fn make_test_layer_shape(config: &TestModelConfig) -> (LayerShape, impl Fn(&mut WeightRng) -> Vec<u8>) {
        let hidden = config.hidden_dim as usize;
        let inter = config.intermediate_dim as usize;
        let heads = config.num_heads as usize;
        let kv_heads = config.num_kv_heads as usize;
        let head_dim = config.head_dim as usize;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        // Compute sizes (same order as test_model.rs)
        let wq_n = q_dim * hidden;
        let wk_n = kv_dim * hidden;
        let wv_n = kv_dim * hidden;
        let wo_n = hidden * q_dim;
        let w_gate_n = inter * hidden;
        let w_up_n = inter * hidden;
        let w_down_n = hidden * inter;
        let attn_norm_n = hidden;
        let ffn_norm_n = hidden;

        let sizes = [wq_n, wk_n, wv_n, wo_n, w_gate_n, w_up_n, w_down_n, attn_norm_n, ffn_norm_n];
        let mut offset = 0u64;
        let mut slices = Vec::new();
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
            attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
            layer_type: None,
        };

        let shape = LayerShape {
            blob_size,
            index: LayerIndex {
                layer_offset_bytes: 0,
                layer_length_bytes: blob_size,
                subtensors,
            },
        };

        let gen_blob = move |rng: &mut WeightRng| -> Vec<u8> {
            let mut blob = Vec::with_capacity(blob_size as usize);
            // Same order as test_model.rs: wq, wk, wv, wo, w_gate, w_up, w_down, attn_norm, ffn_norm
            blob.extend_from_slice(&rng.gen_f32_bytes(wq_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(wk_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(wv_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(wo_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(w_gate_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(w_up_n));
            blob.extend_from_slice(&rng.gen_f32_bytes(w_down_n));
            blob.extend_from_slice(&rng.gen_norm_bytes(attn_norm_n));
            blob.extend_from_slice(&rng.gen_norm_bytes(ffn_norm_n));
            blob
        };

        (shape, gen_blob)
    }

    #[test]
    fn byte_identical_with_generate_test_model() {
        let config = TestModelConfig::default();
        let expected = generate_test_model(&config);

        // Reproduce via streaming writer
        let mut rng = WeightRng::new(config.seed);

        let hidden = config.hidden_dim as usize;
        let vocab = config.vocab_size as usize;

        let embedding = rng.gen_f32_bytes(vocab * hidden);
        let final_norm = rng.gen_norm_bytes(hidden);
        let output_proj = rng.gen_f32_bytes(vocab * hidden);

        let globals = GlobalTensors { embedding, final_norm, output_proj };

        let (shape, gen_blob) = make_test_layer_shape(&config);
        let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

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

        let mut out = Vec::new();
        let mut sw = StreamingLbcWriter::begin(&mut out, &header, &layer_shapes, &globals).unwrap();
        for _ in 0..config.num_layers {
            let blob = gen_blob(&mut rng);
            sw.write_layer(&blob).unwrap();
        }
        sw.finish().unwrap();

        assert_eq!(out.len(), expected.len(), "size mismatch");
        assert_eq!(out, expected, "streaming writer must produce identical bytes");
    }

    #[test]
    fn roundtrip_valid_lbc() {
        let config = TestModelConfig {
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            hidden_dim: 32,
            intermediate_dim: 64,
            vocab_size: 64,
            max_seq_len: 128,
            seed: 99,
        };

        let mut rng = WeightRng::new(config.seed);
        let hidden = config.hidden_dim as usize;
        let vocab = config.vocab_size as usize;

        let globals = GlobalTensors {
            embedding: rng.gen_f32_bytes(vocab * hidden),
            final_norm: rng.gen_norm_bytes(hidden),
            output_proj: rng.gen_f32_bytes(vocab * hidden),
        };

        let (shape, gen_blob) = make_test_layer_shape(&config);
        let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

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

        let mut out = Vec::new();
        let mut sw = StreamingLbcWriter::begin(&mut out, &header, &layer_shapes, &globals).unwrap();
        for _ in 0..config.num_layers {
            sw.write_layer(&gen_blob(&mut rng)).unwrap();
        }
        sw.finish().unwrap();

        // Parse back
        let lbc = LbcFile::from_bytes(&out, PathBuf::from("test.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, config.num_layers);
        assert_eq!(lbc.layer_indices.len(), config.num_layers as usize);
        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }
    }

    #[test]
    fn too_many_layers_error() {
        let config = TestModelConfig::default();
        let mut rng = WeightRng::new(config.seed);
        let hidden = config.hidden_dim as usize;
        let vocab = config.vocab_size as usize;

        let globals = GlobalTensors {
            embedding: rng.gen_f32_bytes(vocab * hidden),
            final_norm: rng.gen_norm_bytes(hidden),
            output_proj: rng.gen_f32_bytes(vocab * hidden),
        };

        let (shape, gen_blob) = make_test_layer_shape(&config);
        let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

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

        let mut out = Vec::new();
        let mut sw = StreamingLbcWriter::begin(&mut out, &header, &layer_shapes, &globals).unwrap();
        for _ in 0..config.num_layers {
            sw.write_layer(&gen_blob(&mut rng)).unwrap();
        }
        // Extra layer should error
        let extra = gen_blob(&mut rng);
        let err = sw.write_layer(&extra);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("too many layers"));
    }

    #[test]
    fn too_few_layers_error() {
        let config = TestModelConfig::default();
        let mut rng = WeightRng::new(config.seed);
        let hidden = config.hidden_dim as usize;
        let vocab = config.vocab_size as usize;

        let globals = GlobalTensors {
            embedding: rng.gen_f32_bytes(vocab * hidden),
            final_norm: rng.gen_norm_bytes(hidden),
            output_proj: rng.gen_f32_bytes(vocab * hidden),
        };

        let (shape, gen_blob) = make_test_layer_shape(&config);
        let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

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

        let mut out = Vec::new();
        let mut sw = StreamingLbcWriter::begin(&mut out, &header, &layer_shapes, &globals).unwrap();
        // Write only 1 of 2 layers
        sw.write_layer(&gen_blob(&mut rng)).unwrap();
        let err = sw.finish();
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("incomplete write"));
    }

    #[test]
    fn wrong_blob_size_error() {
        let config = TestModelConfig::default();
        let mut rng = WeightRng::new(config.seed);
        let hidden = config.hidden_dim as usize;
        let vocab = config.vocab_size as usize;

        let globals = GlobalTensors {
            embedding: rng.gen_f32_bytes(vocab * hidden),
            final_norm: rng.gen_norm_bytes(hidden),
            output_proj: rng.gen_f32_bytes(vocab * hidden),
        };

        let (shape, _) = make_test_layer_shape(&config);
        let layer_shapes: Vec<LayerShape> = (0..config.num_layers).map(|_| shape.clone()).collect();

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

        let mut out = Vec::new();
        let mut sw = StreamingLbcWriter::begin(&mut out, &header, &layer_shapes, &globals).unwrap();
        let err = sw.write_layer(&[0u8; 17]); // Wrong size
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("blob size mismatch"));
    }
}
