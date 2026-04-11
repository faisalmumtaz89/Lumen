//! GGUF-to-LBC converter.
//!
//! Reads a GGUF file, extracts hyperparameters and tensor
//! data, and writes an LBC file using the streaming writer. Memory-efficient:
//! only one layer blob is held in memory at a time.

use crate::arch;
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufError, GgufFile};
use crate::hyperparams::{detect_quant_scheme, extract_hyperparams, quant_descriptor_for};
use crate::tensor_names::*;
use crate::tensor_io::read_tensor_data;
use lumen_format::header::LbcHeader;
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::StreamingLbcWriter;
use lumen_format::tokenizer::TokenizerSection;
use lumen_format::writer::GlobalTensors;
use std::fmt;
use std::io::{BufReader, BufWriter, Read, Seek};
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Options controlling the GGUF-to-LBC conversion.
pub struct ConvertOptions {
    /// LBC alignment in bytes (default: 128 KiB = 131072).
    pub alignment: u64,
    /// If true, dequantize all quantized tensors to F32.
    /// Produces larger files but compatible with the naive F32 backend.
    pub dequantize_to_f32: bool,
    /// If set, requantize weight tensors to this scheme during conversion.
    /// Only Q4_0 is currently supported as a target. The source weights are
    /// first dequantized to F32, then requantized to the target scheme.
    /// Norm tensors remain F32 regardless.
    pub requant_to: Option<QuantScheme>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            alignment: 128 * 1024,
            dequantize_to_f32: false,
            requant_to: None,
        }
    }
}

/// Statistics about a completed conversion.
#[derive(Debug)]
pub struct ConvertStats {
    pub input_size: u64,
    pub output_size: u64,
    pub num_layers: u32,
    pub architecture: String,
    pub tensor_count: usize,
    pub quant_scheme: QuantScheme,
}

impl fmt::Display for ConvertStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Converted {} tensors ({} layers, arch={}, quant={:?})\n  Input:  {:.2} MB\n  Output: {:.2} MB",
            self.tensor_count,
            self.num_layers,
            self.architecture,
            self.quant_scheme,
            self.input_size as f64 / 1_048_576.0,
            self.output_size as f64 / 1_048_576.0,
        )
    }
}

/// Errors that can occur during conversion.
#[derive(Debug)]
pub enum ConvertError {
    Io(std::io::Error),
    Gguf(GgufError),
    Format(lumen_format::FormatError),
    UnsupportedArchitecture(String),
    MissingMetadata(String),
    MissingTensor(String),
    TensorShapeMismatch {
        tensor: String,
        expected: String,
        got: String,
    },
    UnsupportedTensorType {
        tensor: String,
        ggml_type: String,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Gguf(e) => write!(f, "GGUF parse error: {e}"),
            Self::Format(e) => write!(f, "LBC format error: {e}"),
            Self::UnsupportedArchitecture(a) => {
                write!(f, "unsupported architecture: {a}")
            }
            Self::MissingMetadata(k) => write!(f, "missing GGUF metadata: {k}"),
            Self::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Self::TensorShapeMismatch {
                tensor,
                expected,
                got,
            } => write!(f, "tensor {tensor}: expected shape {expected}, got {got}"),
            Self::UnsupportedTensorType { tensor, ggml_type } => {
                write!(f, "tensor {tensor}: unsupported GGML type {ggml_type}")
            }
        }
    }
}

impl std::error::Error for ConvertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Gguf(e) => Some(e),
            Self::Format(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<GgufError> for ConvertError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}

impl From<lumen_format::FormatError> for ConvertError {
    fn from(e: lumen_format::FormatError) -> Self {
        Self::Format(e)
    }
}

/// Convert a GGUF file to LBC format.
///
/// Reads the GGUF header and metadata to extract model hyperparameters, then
/// streams tensor data layer-by-layer to produce the LBC file. Peak memory
/// usage is O(1 layer blob + global tensors).
pub fn convert_gguf_to_lbc(
    gguf_path: &Path,
    lbc_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertStats, ConvertError> {
    let input_size = std::fs::metadata(gguf_path)?.len();

    // Parse GGUF header (does not read tensor data)
    let mut file = BufReader::new(std::fs::File::open(gguf_path)?);
    let gguf = GgufFile::parse(&mut file)?;

    let stats = do_convert(&gguf, gguf_path, lbc_path, options, input_size)?;
    Ok(stats)
}

/// Convert from an in-memory GGUF buffer (useful for testing).
pub fn convert_gguf_bytes_to_lbc(
    gguf_data: &[u8],
    lbc_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertStats, ConvertError> {
    let mut cursor = std::io::Cursor::new(gguf_data);
    let gguf = GgufFile::parse(&mut cursor)?;

    let stats = do_convert_from_reader(
        &gguf,
        &mut std::io::Cursor::new(gguf_data),
        lbc_path,
        options,
        gguf_data.len() as u64,
    )?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Core conversion logic
// ---------------------------------------------------------------------------

fn do_convert(
    gguf: &GgufFile,
    gguf_path: &Path,
    lbc_path: &Path,
    opts: &ConvertOptions,
    input_size: u64,
) -> Result<ConvertStats, ConvertError> {
    let mut reader = BufReader::new(std::fs::File::open(gguf_path)?);
    do_convert_from_reader(gguf, &mut reader, lbc_path, opts, input_size)
}

fn do_convert_from_reader<R: Read + Seek>(
    gguf: &GgufFile,
    reader: &mut R,
    lbc_path: &Path,
    opts: &ConvertOptions,
    input_size: u64,
) -> Result<ConvertStats, ConvertError> {
    let (hp, arch) = extract_hyperparams(gguf)?;
    let num_layers = hp.num_layers as usize;

    // Detect primary quantization scheme.
    let quant_scheme = if let Some(target) = opts.requant_to {
        target
    } else if opts.dequantize_to_f32 {
        QuantScheme::F32
    } else {
        detect_quant_scheme(gguf, hp.num_layers)
    };
    let qd = quant_descriptor_for(quant_scheme);

    // Read embedding tensor, keeping quantized format when possible (GPU dequant kernels handle it)
    let embedding_tensor = gguf
        .find_tensor(EMBEDDING_NAME)
        .ok_or_else(|| ConvertError::MissingTensor(EMBEDDING_NAME.into()))?;
    let embedding_bytes = read_tensor_data(reader, gguf, embedding_tensor)?;
    let embedding_ggml_type = embedding_tensor.ggml_type;
    let embedding_n_elements = embedding_tensor.n_elements();

    // For Q8_0, Q4_0, and F16 embeddings, keep raw bytes in the LBC file.
    // The runtime will use GPU dequant kernels for embedding lookup.
    let (embedding, embedding_quant) = match embedding_ggml_type {
        GgmlType::F16 => {
            eprintln!("  Keeping embedding as F16 ({} bytes, {} elements)",
                embedding_bytes.len(), embedding_n_elements);
            (embedding_bytes, QuantScheme::F16)
        }
        GgmlType::Q8_0 => {
            eprintln!("  Keeping embedding as Q8_0 ({} bytes, {} elements)",
                embedding_bytes.len(), embedding_n_elements);
            (embedding_bytes, QuantScheme::Q8_0)
        }
        GgmlType::Q4_0 => {
            eprintln!("  Keeping embedding as Q4_0 ({} bytes, {} elements)",
                embedding_bytes.len(), embedding_n_elements);
            (embedding_bytes, QuantScheme::Q4_0)
        }
        _ => {
            let f32_data = ensure_f32_global(
                embedding_bytes,
                embedding_ggml_type,
                EMBEDDING_NAME,
                embedding_n_elements,
            )?;
            (f32_data, QuantScheme::F32)
        }
    };

    let final_norm_tensor = gguf
        .find_tensor(FINAL_NORM_NAME)
        .ok_or_else(|| ConvertError::MissingTensor(FINAL_NORM_NAME.into()))?;
    let final_norm_bytes = read_tensor_data(reader, gguf, final_norm_tensor)?;
    let final_norm = ensure_f32_global(
        final_norm_bytes,
        final_norm_tensor.ggml_type,
        FINAL_NORM_NAME,
        final_norm_tensor.n_elements(),
    )?;

    // Weight tying: if output.weight is absent, use dedup (store once, share at runtime).
    let requant_target = opts.requant_to;
    let (output_proj, weight_tying, output_proj_quant) = if let Some(output_proj_tensor) = gguf.find_tensor(OUTPUT_PROJ_NAME) {
        let output_proj_bytes = read_tensor_data(reader, gguf, output_proj_tensor)?;
        // Handle requantization of output_proj
        if let Some(QuantScheme::Q4_0) = requant_target {
            let f32_data = ensure_f32_global(
                output_proj_bytes.clone(),
                output_proj_tensor.ggml_type,
                OUTPUT_PROJ_NAME,
                output_proj_tensor.n_elements(),
            )?;
            let n_elems = output_proj_tensor.n_elements() as usize;
            let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
            eprintln!("  Requantized output.weight to Q4_0 ({} bytes, {} elements)",
                q4_data.len(), n_elems);
            (q4_data, false, QuantScheme::Q4_0)
        } else if output_proj_tensor.ggml_type == GgmlType::Q8_0 {
            eprintln!("  Keeping output.weight as Q8_0 ({} bytes, {} elements)",
                output_proj_bytes.len(), output_proj_tensor.n_elements());
            (output_proj_bytes, false, QuantScheme::Q8_0)
        } else if output_proj_tensor.ggml_type == GgmlType::Q4_0 {
            eprintln!("  Keeping output.weight as Q4_0 ({} bytes, {} elements)",
                output_proj_bytes.len(), output_proj_tensor.n_elements());
            (output_proj_bytes, false, QuantScheme::Q4_0)
        } else if output_proj_tensor.ggml_type == GgmlType::F16 {
            eprintln!("  Keeping output.weight as F16 ({} bytes, {} elements)",
                output_proj_bytes.len(), output_proj_tensor.n_elements());
            (output_proj_bytes, false, QuantScheme::F16)
        } else if matches!(output_proj_tensor.ggml_type,
            GgmlType::Q6_K | GgmlType::Q5_K | GgmlType::Q4_K |
            GgmlType::Q3_K | GgmlType::Q2_K | GgmlType::Q8_K)
        {
            // K-quant output.weight: dequantize and requantize to a supported format.
            // llama-quantize often keeps output.weight as Q6_K even in Q4_0 GGUFs.
            // The runtime only has fast dispatch kernels for Q8_0/Q4_0/F16/F32,
            // so storing K-quant as-is would hit the slow F32 fallback path.
            let f32_data = ensure_f32_global(
                output_proj_bytes,
                output_proj_tensor.ggml_type,
                OUTPUT_PROJ_NAME,
                output_proj_tensor.n_elements(),
            )?;
            let n_elems = output_proj_tensor.n_elements() as usize;
            if requant_target == Some(QuantScheme::Q4_0) {
                let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
                eprintln!("  K-quant output.weight ({:?}): requantized to Q4_0 ({} bytes)",
                    output_proj_tensor.ggml_type, q4_data.len());
                (q4_data, false, QuantScheme::Q4_0)
            } else {
                let q8_data = quantize_f32_to_q8_0(&f32_data, n_elems);
                eprintln!("  K-quant output.weight ({:?}): requantized to Q8_0 ({} bytes)",
                    output_proj_tensor.ggml_type, q8_data.len());
                (q8_data, false, QuantScheme::Q8_0)
            }
        } else {
            let data = ensure_f32_global(
                output_proj_bytes,
                output_proj_tensor.ggml_type,
                OUTPUT_PROJ_NAME,
                output_proj_tensor.n_elements(),
            )?;
            (data, false, QuantScheme::F32)
        }
    } else {
        // Weight tying: output_proj shares embedding storage (zero-copy dedup).
        // Runtime uses the embedding buffer for both lookup and logits projection.
        let embed_size_mb = embedding.len() as f64 / 1_048_576.0;
        eprintln!("  Weight tying: output_proj shares embedding storage ({:.1} MB saved)", embed_size_mb);
        (Vec::new(), true, embedding_quant)
    };

    // Select architecture-specific converter
    let converter = arch::select_converter(&arch, hp.num_experts);

    // Compute layer shapes
    let mut layer_shapes = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        layer_shapes.push(converter.compute_layer_shape(
            gguf, layer, opts.dequantize_to_f32, opts.requant_to,
        )?);
    }

    // Build LBC header with quant metadata
    let mut header = LbcHeader::new(hp, qd);
    header.alignment = opts.alignment;
    header.embedding.quant = embedding_quant;
    header.final_norm.quant = QuantScheme::F32; // norms always F32
    header.output_proj.quant = output_proj_quant;
    header.weight_tying = weight_tying;

    // Create LBC file with streaming writer
    let output_file = std::fs::File::create(lbc_path)?;
    let writer = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    let global_tensors = GlobalTensors {
        embedding,
        final_norm,
        output_proj,
    };

    // Extract tokenizer data from GGUF and embed in LBC v3.
    let tokenizer_section = crate::tokenizer_data::extract_tokenizer(gguf).map(|td| {
        eprintln!("  Tokenizer: model={} pre={} vocab={} merges={} scores={}",
            td.model_type, td.pre_tokenizer, td.tokens.len(), td.merges.len(), td.scores.len());
        TokenizerSection {
            model_type: td.model_type,
            pre_tokenizer: td.pre_tokenizer,
            tokens: td.tokens,
            token_types: td.token_types,
            scores: td.scores,
            merges: td.merges,
            bos_token_id: td.bos_token_id,
            eos_token_id: td.eos_token_id,
            pad_token_id: td.pad_token_id,
            add_bos_token: td.add_bos_token,
            add_eos_token: td.add_eos_token,
            add_space_prefix: td.add_space_prefix,
            chat_template: td.chat_template,
        }
    });

    let mut streaming =
        StreamingLbcWriter::begin(writer, &header, &layer_shapes, &global_tensors, tokenizer_section.as_ref())?;

    // Write each layer blob (stream from GGUF file)
    for (layer, shape) in layer_shapes.iter().enumerate().take(num_layers) {
        let mut layer_blob =
            Vec::with_capacity(shape.blob_size as usize);

        converter.write_layer_blob(
            &mut layer_blob, reader, gguf, layer, opts.dequantize_to_f32, opts.requant_to,
        )?;

        streaming.write_layer(&layer_blob)?;

        let kind_label = converter.layer_kind_label(layer);
        if kind_label.is_empty() {
            eprintln!(
                "  Layer {}/{} ({:.1} MB)",
                layer + 1,
                num_layers,
                layer_blob.len() as f64 / 1_048_576.0,
            );
        } else {
            eprintln!(
                "  Layer {}/{} ({:.1} MB, {})",
                layer + 1,
                num_layers,
                layer_blob.len() as f64 / 1_048_576.0,
                kind_label,
            );
        }
    }

    streaming.finish()?;

    let output_size = std::fs::metadata(lbc_path)?.len();

    Ok(ConvertStats {
        input_size,
        output_size,
        num_layers: hp.num_layers,
        architecture: arch,
        tensor_count: gguf.tensors.len(),
        quant_scheme,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufBuilder, GgmlType};
    use crate::tensor_io::{layer_tensor_name, expert_tensor_name};
    use lumen_format::reader::LbcFile;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_convert_test_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a minimal GGUF file representing a 2-layer LLaMA-like model
    /// with F32 tensors for testing.
    fn build_test_gguf(
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let hidden = hidden_dim as usize;
        let inter = intermediate_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();

        // Metadata
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", num_layers);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden_dim);
        builder.add_u32("llama.feed_forward_length", intermediate_dim);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        // Global tensors
        let embedding_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.001) % 1.0)
            .collect();
        builder.add_f32_tensor(
            EMBEDDING_NAME,
            &[vocab as u64, hidden as u64],
            &embedding_data,
        );

        let norm_data: Vec<f32> = vec![1.0; hidden];
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &norm_data);

        let output_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.002) % 1.0)
            .collect();
        builder.add_f32_tensor(
            OUTPUT_PROJ_NAME,
            &[vocab as u64, hidden as u64],
            &output_data,
        );

        // Per-layer tensors
        for layer in 0..num_layers as usize {
            let layer_tensors: Vec<(&str, usize, usize)> = vec![
                (ATTN_Q, q_dim, hidden),
                (ATTN_K, kv_dim, hidden),
                (ATTN_V, kv_dim, hidden),
                (ATTN_OUTPUT, hidden, q_dim),
                (FFN_GATE, inter, hidden),
                (FFN_UP, inter, hidden),
                (FFN_DOWN, hidden, inter),
                (ATTN_NORM, hidden, 1),
                (FFN_NORM, hidden, 1),
            ];

            for (suffix, rows, cols) in layer_tensors {
                let name = layer_tensor_name(layer, suffix);
                let n = rows * cols;
                let data: Vec<f32> = (0..n)
                    .map(|i| ((i + layer * 1000) as f32 * 0.001) % 1.0)
                    .collect();
                let dims = if cols == 1 {
                    vec![rows as u64]
                } else {
                    vec![rows as u64, cols as u64]
                };
                builder.add_f32_tensor(&name, &dims, &data);
            }
        }

        builder.build()
    }

    #[test]
    fn round_trip_f32() {
        let dir = temp_dir();
        let lbc_path = dir.join("test.lbc");

        let gguf_data = build_test_gguf(2, 2, 2, 8, 16, 32);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };

        let stats =
            convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.architecture, "llama");
        assert_eq!(stats.quant_scheme, QuantScheme::F32);

        // Verify the LBC file is valid
        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 8);
        assert_eq!(lbc.header.hyperparams.vocab_size, 32);
        assert_eq!(lbc.header.hyperparams.num_heads, 2);
        assert_eq!(lbc.header.hyperparams.num_kv_heads, 2);
        assert_eq!(lbc.header.hyperparams.head_dim, 4);
        assert_eq!(lbc.header.hyperparams.intermediate_dim, 16);
        assert_eq!(lbc.header.hyperparams.max_seq_len, 64);

        // Verify layer indices are valid
        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }

        // Verify global tensor sizes
        // embedding: vocab * hidden * 4 bytes = 32 * 8 * 4 = 1024
        assert_eq!(lbc.header.embedding.length, 32 * 8 * 4);
        // final_norm: hidden * 4 = 8 * 4 = 32
        assert_eq!(lbc.header.final_norm.length, 8 * 4);
        // output_proj: vocab * hidden * 4 = 32 * 8 * 4 = 1024
        assert_eq!(lbc.header.output_proj.length, 32 * 8 * 4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extract_hyperparams_llama() {
        let gguf_data = build_test_gguf(4, 8, 4, 64, 128, 256);
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        let (hp, arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(arch, "llama");
        assert_eq!(hp.num_layers, 4);
        assert_eq!(hp.num_heads, 8);
        assert_eq!(hp.num_kv_heads, 4);
        assert_eq!(hp.head_dim, 8); // 64 / 8
        assert_eq!(hp.hidden_dim, 64);
        assert_eq!(hp.intermediate_dim, 128);
        assert_eq!(hp.vocab_size, 256);
        assert_eq!(hp.max_seq_len, 64);
        assert!((hp.norm_eps - 1e-5).abs() < 1e-10);

        let rope = hp.rope_params.unwrap();
        assert!((rope.theta - 10000.0).abs() < 0.01);
        assert!((rope.scaling_factor - 1.0).abs() < 0.01);
        assert_eq!(rope.scaling_type, lumen_format::hyperparams::RopeScalingType::None);
    }

    #[test]
    fn tensor_mapping_completeness() {
        use crate::tensor_names::LAYER_TENSOR_SUFFIXES;
        let gguf_data = build_test_gguf(2, 2, 2, 8, 16, 32);
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        // Check all 3 global tensors exist
        assert!(gguf.find_tensor(EMBEDDING_NAME).is_some());
        assert!(gguf.find_tensor(FINAL_NORM_NAME).is_some());
        assert!(gguf.find_tensor(OUTPUT_PROJ_NAME).is_some());

        // Check all 9 per-layer tensors exist for each layer
        for layer in 0..2 {
            for suffix in &LAYER_TENSOR_SUFFIXES {
                let name = layer_tensor_name(layer, suffix);
                assert!(
                    gguf.find_tensor(&name).is_some(),
                    "missing tensor: {name}"
                );
            }
        }

        // Total: 3 global + 2 * 9 layer = 21 tensors
        assert_eq!(gguf.tensors.len(), 21);
    }

    #[test]
    fn missing_tensor_error() {
        // Build a GGUF file missing one tensor
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 1);
        builder.add_u32("llama.attention.head_count", 2);
        builder.add_u32("llama.attention.head_count_kv", 2);
        builder.add_u32("llama.embedding_length", 8);
        builder.add_u32("llama.feed_forward_length", 16);
        // Only add embedding, no other tensors
        let data = vec![0.0f32; 32 * 8];
        builder.add_f32_tensor(EMBEDDING_NAME, &[32, 8], &data);

        let gguf_data = builder.build();
        let dir = temp_dir();
        let lbc_path = dir.join("missing.lbc");

        let result = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &ConvertOptions::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            ConvertError::MissingTensor(name) => {
                assert_eq!(name, FINAL_NORM_NAME);
            }
            other => panic!("expected MissingTensor, got: {other}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unsupported_architecture() {
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "gpt2");
        let gguf_data = builder.build();

        let dir = temp_dir();
        let lbc_path = dir.join("unsupported.lbc");

        let result = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &ConvertOptions::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            ConvertError::UnsupportedArchitecture(a) => assert_eq!(a, "gpt2"),
            other => panic!("expected UnsupportedArchitecture, got: {other}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Build a minimal GGUF WITHOUT output.weight (weight-tying scenario).
    /// The token_embd.weight should be used as the output projection.
    fn build_test_gguf_tied_weights(
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let hidden = hidden_dim as usize;
        let inter = intermediate_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();

        // Metadata
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", num_layers);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden_dim);
        builder.add_u32("llama.feed_forward_length", intermediate_dim);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        // Token list for vocab size
        let tokens: Vec<String> = (0..vocab).map(|i| format!("tok_{i}")).collect();
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        builder.add_string_array("tokenizer.ggml.tokens", &token_refs);

        // Global tensors -- NO output.weight (weight tying)
        let embedding_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.001) % 1.0)
            .collect();
        builder.add_f32_tensor(
            EMBEDDING_NAME,
            &[vocab as u64, hidden as u64],
            &embedding_data,
        );

        let norm_data: Vec<f32> = vec![1.0; hidden];
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &norm_data);

        // NOTE: output.weight is deliberately omitted to test weight tying

        // Per-layer tensors
        for layer in 0..num_layers as usize {
            let layer_tensors: Vec<(&str, usize, usize)> = vec![
                (ATTN_Q, q_dim, hidden),
                (ATTN_K, kv_dim, hidden),
                (ATTN_V, kv_dim, hidden),
                (ATTN_OUTPUT, hidden, q_dim),
                (FFN_GATE, inter, hidden),
                (FFN_UP, inter, hidden),
                (FFN_DOWN, hidden, inter),
                (ATTN_NORM, hidden, 1),
                (FFN_NORM, hidden, 1),
            ];

            for (suffix, rows, cols) in layer_tensors {
                let name = layer_tensor_name(layer, suffix);
                let n = rows * cols;
                let data: Vec<f32> = (0..n)
                    .map(|i| ((i + layer * 1000) as f32 * 0.001) % 1.0)
                    .collect();
                let dims = if cols == 1 {
                    vec![rows as u64]
                } else {
                    vec![rows as u64, cols as u64]
                };
                builder.add_f32_tensor(&name, &dims, &data);
            }
        }

        builder.build()
    }

    #[test]
    fn weight_tying_fallback() {
        let dir = temp_dir();
        let lbc_path = dir.join("tied.lbc");

        let gguf_data = build_test_gguf_tied_weights(2, 2, 2, 8, 16, 32);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };

        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.architecture, "llama");

        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 8);
        assert_eq!(lbc.header.hyperparams.vocab_size, 32);

        assert_eq!(lbc.header.output_proj.length, 32 * 8 * 4);
        assert_eq!(lbc.header.embedding.length, lbc.header.output_proj.length);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn weight_tying_data_matches_embedding() {
        let dir = temp_dir();
        let lbc_path = dir.join("tied_data.lbc");

        let gguf_data = build_test_gguf_tied_weights(1, 2, 2, 8, 16, 32);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };

        convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();

        let lbc_bytes = std::fs::read(&lbc_path).unwrap();
        let lbc = LbcFile::from_bytes(&lbc_bytes, lbc_path.clone()).unwrap();

        let emb_start = lbc.header.embedding.offset as usize;
        let emb_end = emb_start + lbc.header.embedding.length as usize;
        let emb_data = &lbc_bytes[emb_start..emb_end];

        let out_start = lbc.header.output_proj.offset as usize;
        let out_end = out_start + lbc.header.output_proj.length as usize;
        let out_data = &lbc_bytes[out_start..out_end];

        assert_eq!(emb_data, out_data, "weight-tied output_proj data should match embedding data");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Build a minimal GGUF WITHOUT output.weight where token_embd.weight is Q8_0.
    fn build_test_gguf_tied_weights_q8(
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let hidden = hidden_dim as usize;
        let inter = intermediate_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();

        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", num_layers);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden_dim);
        builder.add_u32("llama.feed_forward_length", intermediate_dim);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        let tokens: Vec<String> = (0..vocab).map(|i| format!("tok_{i}")).collect();
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        builder.add_string_array("tokenizer.ggml.tokens", &token_refs);

        let embedding_f32: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32 * 0.001) % 1.0) - 0.5)
            .collect();
        let embedding_q8 = build_q8_0_blocks(&embedding_f32);
        builder.add_tensor(
            EMBEDDING_NAME,
            GgmlType::Q8_0,
            &[vocab as u64, hidden as u64],
            embedding_q8,
        );

        let norm_data: Vec<f32> = vec![1.0; hidden];
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &norm_data);

        for layer in 0..num_layers as usize {
            let layer_tensors: Vec<(&str, usize, usize)> = vec![
                (ATTN_Q, q_dim, hidden),
                (ATTN_K, kv_dim, hidden),
                (ATTN_V, kv_dim, hidden),
                (ATTN_OUTPUT, hidden, q_dim),
                (FFN_GATE, inter, hidden),
                (FFN_UP, inter, hidden),
                (FFN_DOWN, hidden, inter),
            ];

            for (suffix, rows, cols) in layer_tensors {
                let name = layer_tensor_name(layer, suffix);
                let n = rows * cols;
                let f32_values: Vec<f32> = (0..n)
                    .map(|i| ((i + layer * 1000) as f32 * 0.001 - 0.5).clamp(-1.0, 1.0))
                    .collect();
                let q8_data = build_q8_0_blocks(&f32_values);
                let dims = vec![rows as u64, cols as u64];
                builder.add_tensor(&name, GgmlType::Q8_0, &dims, q8_data);
            }

            let norm: Vec<f32> = vec![1.0; hidden];
            let attn_norm_name = layer_tensor_name(layer, ATTN_NORM);
            builder.add_f32_tensor(&attn_norm_name, &[hidden as u64], &norm);
            let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
            builder.add_f32_tensor(&ffn_norm_name, &[hidden as u64], &norm);
        }

        builder.build()
    }

    fn build_q8_0_blocks(values: &[f32]) -> Vec<u8> {
        let mut data = Vec::new();
        for chunk in values.chunks(32) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let scale_bits = f32_to_f16_bits(scale);
            data.extend_from_slice(&scale_bits.to_le_bytes());
            for &v in chunk {
                let q = if scale > 0.0 {
                    (v / scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0i8
                };
                data.push(q as u8);
            }
        }
        data
    }

    #[test]
    fn weight_tying_q8_0_preserves_quantization() {
        let dir = temp_dir();
        let lbc_path = dir.join("tied_q8.lbc");

        let gguf_data = build_test_gguf_tied_weights_q8(1, 2, 2, 32, 64, 64);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };

        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.quant_scheme, QuantScheme::Q8_0);

        let lbc = LbcFile::open(&lbc_path).unwrap();
        let vocab = 64usize;
        let hidden = 32usize;
        let n_elements = vocab * hidden;
        let expected_q8_bytes = (n_elements / 32) * 34;
        let f32_bytes = n_elements * 4;

        assert_eq!(
            lbc.header.output_proj.length, expected_q8_bytes as u64,
            "output_proj should be Q8_0 sized ({} bytes), not F32 ({} bytes)",
            expected_q8_bytes, f32_bytes
        );

        assert_eq!(lbc.header.embedding.length, expected_q8_bytes as u64,
            "embedding should be Q8_0 sized ({} bytes), not F32 ({} bytes)",
            expected_q8_bytes, f32_bytes);

        assert_eq!(lbc.header.output_proj.length, lbc.header.embedding.length,
            "Q8_0 output_proj ({}) should be same size as Q8_0 embedding ({})",
            lbc.header.output_proj.length, lbc.header.embedding.length);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn weight_tying_q8_0_data_matches_source() {
        let dir = temp_dir();
        let lbc_path = dir.join("tied_q8_data.lbc");

        let vocab = 64usize;
        let hidden = 32usize;

        let gguf_data = build_test_gguf_tied_weights_q8(1, 2, 2, hidden as u32, 64, vocab as u32);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };

        convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();

        let lbc_bytes = std::fs::read(&lbc_path).unwrap();
        let lbc = LbcFile::from_bytes(&lbc_bytes, lbc_path.clone()).unwrap();

        let out_start = lbc.header.output_proj.offset as usize;
        let out_end = out_start + lbc.header.output_proj.length as usize;
        let out_data = &lbc_bytes[out_start..out_end];

        let n_elements = vocab * hidden;
        let expected_q8_bytes = (n_elements / 32) * 34;
        assert_eq!(out_data.len(), expected_q8_bytes,
            "output_proj should contain Q8_0 blocks");

        let embedding_f32: Vec<f32> = (0..n_elements)
            .map(|i| ((i as f32 * 0.001) % 1.0) - 0.5)
            .collect();
        let expected_q8 = build_q8_0_blocks(&embedding_f32);
        assert_eq!(out_data, &expected_q8[..],
            "output_proj Q8_0 data should match the original embedding Q8_0 data");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn f16_to_f32_conversion() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-7);
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-7);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_sign_negative());
        assert!(f16_to_f32(0x7C01).is_nan());
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-7);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn bf16_to_f32_conversion() {
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-7);
        assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-7);
        assert!((bf16_to_f32(0x4000) - 2.0).abs() < 1e-7);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    /// Encode an f32 to f16 bits (for constructing test blocks).
    fn f32_to_f16_bits(val: f32) -> u16 {
        if val == 0.0 {
            return if val.is_sign_negative() { 0x8000 } else { 0 };
        }
        if val.is_nan() {
            return 0x7C01;
        }
        if val.is_infinite() {
            return if val > 0.0 { 0x7C00 } else { 0xFC00 };
        }
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = bits & 0x7FFFFF;
        if exp > 15 {
            return if sign == 1 { 0xFC00 } else { 0x7C00 };
        }
        if exp < -14 {
            return if sign == 1 { 0x8000 } else { 0 };
        }
        let f16_exp = (exp + 15) as u16;
        let f16_frac = (frac >> 13) as u16;
        ((sign as u16) << 15) | (f16_exp << 10) | f16_frac
    }

    #[test]
    fn f32_to_f16_bits_roundtrip() {
        for val in [0.0f32, 1.0, -1.0, 0.5, 2.0, 0.25, -0.125] {
            let bits = f32_to_f16_bits(val);
            let recovered = f16_to_f32(bits);
            assert!(
                (recovered - val).abs() < 1e-3,
                "roundtrip failed: {val} -> 0x{bits:04X} -> {recovered}"
            );
        }
    }

    #[test]
    fn dequantize_q8_0_known_block() {
        let scale_bits = f32_to_f16_bits(2.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }
        assert_eq!(block.len(), 34);

        let result = dequantize_q8_0(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            let expected = 2.0 * i as f32;
            assert!(
                (v - expected).abs() < 1e-2,
                "Q8_0 mismatch at {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn dequantize_q8_0_negative_values() {
        let scale_bits = f32_to_f16_bits(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        for i in 1..=32 {
            block.push((-i as i8) as u8);
        }

        let result = dequantize_q8_0(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        for (i, &v) in values.iter().enumerate() {
            let expected = -(i as f32 + 1.0);
            assert!(
                (v - expected).abs() < 1e-2,
                "Q8_0 neg mismatch at {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn dequantize_q4_0_known_block() {
        let scale_bits = f32_to_f16_bits(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        for i in 0..16u8 {
            let lo = i;
            let hi = 15 - i;
            block.push(lo | (hi << 4));
        }
        assert_eq!(block.len(), 18);

        let result = dequantize_q4_0(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        for i in 0..16 {
            let expected_lo = 0.5 * (i as f32 - 8.0);
            assert!(
                (values[i] - expected_lo).abs() < 1e-2,
                "Q4_0 lo mismatch at index {i}: got {}, expected {expected_lo}",
                values[i]
            );
        }
        for i in 0..16 {
            let expected_hi = 0.5 * ((15 - i) as f32 - 8.0);
            assert!(
                (values[16 + i] - expected_hi).abs() < 1e-2,
                "Q4_0 hi mismatch at index {}: got {}, expected {expected_hi}",
                16 + i, values[16 + i]
            );
        }
    }

    #[test]
    fn dequantize_q4_1_known_block() {
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        block.resize(block.len() + 16, 0x33);
        assert_eq!(block.len(), 20);

        let result = dequantize_q4_1(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 3.5).abs() < 1e-2,
                "Q4_1 mismatch at {i}: got {v}, expected 3.5"
            );
        }
    }

    #[test]
    fn dequantize_q5_0_known_block() {
        let scale_bits = f32_to_f16_bits(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        let qh: u32 = 0xAAAA_AAAA;
        block.extend_from_slice(&qh.to_le_bytes());
        block.resize(block.len() + 16, 0x00);
        assert_eq!(block.len(), 22);

        let result = dequantize_q5_0(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            let high_bit = ((qh >> i) & 1) as u8;
            let expected = 1.0 * ((high_bit << 4) as f32 - 16.0);
            assert!(
                (v - expected).abs() < 1e-2,
                "Q5_0 mismatch at {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn dequantize_q4_k_known_block() {
        let d_bits = f32_to_f16_bits(1.0);
        let dmin_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(&dmin_bits.to_le_bytes());
        let scales = [1u8; 12];
        block.extend_from_slice(&scales);
        block.resize(block.len() + 128, 0x55);
        assert_eq!(block.len(), 144);

        let result = dequantize_q4_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-2,
                "Q4_K mismatch at {i}: got {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn dequantize_q4_k_with_min() {
        let d_bits = f32_to_f16_bits(1.0);
        let dmin_bits = f32_to_f16_bits(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(&dmin_bits.to_le_bytes());

        let mut scales = [0u8; 12];
        for s in &mut scales[..4] { *s = 2; }
        for s in &mut scales[4..8] { *s = 3; }
        for s in &mut scales[8..12] { *s = 0x32; }
        block.extend_from_slice(&scales);
        block.resize(block.len() + 128, 0x44);
        assert_eq!(block.len(), 144);

        let result = dequantize_q4_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 6.5).abs() < 1e-2,
                "Q4_K+min mismatch at {i}: got {v}, expected 6.5"
            );
        }
    }

    #[test]
    fn dequantize_q5_k_known_block() {
        let d_bits = f32_to_f16_bits(1.0);
        let dmin_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(&dmin_bits.to_le_bytes());
        block.extend_from_slice(&[1u8; 12]);
        block.extend_from_slice(&[0u8; 32]);
        block.resize(block.len() + 128, 0x33);
        assert_eq!(block.len(), 176);

        let result = dequantize_q5_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-2,
                "Q5_K mismatch at {i}: got {v}, expected 3.0"
            );
        }
    }

    #[test]
    fn dequantize_q5_k_with_high_bits() {
        let d_bits = f32_to_f16_bits(1.0);
        let dmin_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&d_bits.to_le_bytes());
        block.extend_from_slice(&dmin_bits.to_le_bytes());
        block.extend_from_slice(&[1u8; 12]);
        block.extend_from_slice(&[0xFFu8; 32]);
        block.resize(block.len() + 128, 0x00);
        assert_eq!(block.len(), 176);

        let result = dequantize_q5_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 16.0).abs() < 1e-2,
                "Q5_K high-bit mismatch at {i}: got {v}, expected 16.0"
            );
        }
    }

    #[test]
    fn dequantize_q6_k_known_block() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0u8; 128]);
        block.extend_from_slice(&[0u8; 64]);
        block.extend_from_slice(&[1u8; 16]);
        let d_bits = f32_to_f16_bits(1.0);
        block.extend_from_slice(&d_bits.to_le_bytes());
        assert_eq!(block.len(), 210);

        let result = dequantize_q6_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - (-32.0)).abs() < 1e-2,
                "Q6_K mismatch at {i}: got {v}, expected -32.0"
            );
        }
    }

    #[test]
    fn dequantize_q6_k_nonzero_values() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0x11u8; 128]);
        block.extend_from_slice(&[0u8; 64]);
        block.extend_from_slice(&[2u8; 16]);
        let d_bits = f32_to_f16_bits(0.5);
        block.extend_from_slice(&d_bits.to_le_bytes());
        assert_eq!(block.len(), 210);

        let result = dequantize_q6_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - (-31.0)).abs() < 1e-2,
                "Q6_K nonzero mismatch at {i}: got {v}, expected -31.0"
            );
        }
    }

    #[test]
    fn dequantize_q2_k_known_block() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0x12u8; 16]);
        block.extend_from_slice(&[0x55u8; 64]);
        let d_bits = f32_to_f16_bits(1.0);
        block.extend_from_slice(&d_bits.to_le_bytes());
        let dmin_bits = f32_to_f16_bits(0.5);
        block.extend_from_slice(&dmin_bits.to_le_bytes());
        assert_eq!(block.len(), 84);

        let result = dequantize_q2_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 1.5).abs() < 1e-2,
                "Q2_K mismatch at {i}: got {v}, expected 1.5"
            );
        }
    }

    #[test]
    fn dequantize_q2_k_zero_scale() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0x30u8; 16]);
        block.extend_from_slice(&[0x00u8; 64]);
        let d_bits = f32_to_f16_bits(1.0);
        block.extend_from_slice(&d_bits.to_le_bytes());
        let dmin_bits = f32_to_f16_bits(2.0);
        block.extend_from_slice(&dmin_bits.to_le_bytes());
        assert_eq!(block.len(), 84);

        let result = dequantize_q2_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - (-6.0)).abs() < 1e-2,
                "Q2_K zero-scale mismatch at {i}: got {v}, expected -6.0"
            );
        }
    }

    #[test]
    fn dequantize_q3_k_known_block() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0u8; 32]);
        block.extend_from_slice(&[0u8; 64]);
        block.extend_from_slice(&[0u8; 12]);
        let d_bits = f32_to_f16_bits(1.0);
        block.extend_from_slice(&d_bits.to_le_bytes());
        assert_eq!(block.len(), 110);

        let result = dequantize_q3_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 128.0).abs() < 1e-1,
                "Q3_K mismatch at {i}: got {v}, expected 128.0"
            );
        }
    }

    #[test]
    fn dequantize_q3_k_with_hmask() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0xFFu8; 32]);
        block.extend_from_slice(&[0u8; 64]);
        block.extend_from_slice(&[0u8; 12]);
        let d_bits = f32_to_f16_bits(1.0);
        block.extend_from_slice(&d_bits.to_le_bytes());
        assert_eq!(block.len(), 110);

        let result = dequantize_q3_k(&block, 256);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 256);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v.abs() < 1e-2,
                "Q3_K hmask mismatch at {i}: got {v}, expected 0.0"
            );
        }
    }

    #[test]
    fn dequantize_f32_passthrough() {
        let input: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result =
            dequantize_to_f32_bytes(&input, GgmlType::F32, 3, "test").unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn dequantize_f16_via_dispatcher() {
        let input: Vec<u8> = [0x3C00u16, 0x4000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result =
            dequantize_to_f32_bytes(&input, GgmlType::F16, 2, "test").unwrap();
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn dequantize_unsupported_type_errors() {
        // Q8_K has known block geometry but no dequant path
        let result = dequantize_to_f32_bytes(&[], GgmlType::Q8_K, 0, "test");
        assert!(result.is_err());
        match result.unwrap_err() {
            ConvertError::UnsupportedTensorType { tensor, .. } => {
                assert_eq!(tensor, "test");
            }
            other => panic!("expected UnsupportedTensorType, got: {other}"),
        }
    }

    #[test]
    fn dequantize_q8_1_known_block() {
        // Q8_1: [f16 scale][f16 min][32 x i8] = 36 bytes
        // val[i] = scale * qs[i] + min
        let scale_bits = f32_to_f16_bits(2.0);
        let min_bits = f32_to_f16_bits(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }
        assert_eq!(block.len(), 36);

        let result = dequantize_q8_1(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            let expected = 2.0 * i as f32 + 0.5;
            assert!(
                (v - expected).abs() < 1e-2,
                "Q8_1 mismatch at {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn dequantize_q8_1_negative_values() {
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(-3.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        for i in 1..=32 {
            block.push((-i as i8) as u8);
        }

        let result = dequantize_q8_1(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        for (i, &v) in values.iter().enumerate() {
            let expected = -(i as f32 + 1.0) - 3.0;
            assert!(
                (v - expected).abs() < 1e-2,
                "Q8_1 neg mismatch at {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn dequantize_q8_1_via_dispatcher() {
        // Verify Q8_1 works through dequantize_to_f32_bytes
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }

        let result = dequantize_to_f32_bytes(&block, GgmlType::Q8_1, 32, "test");
        assert!(result.is_ok(), "Q8_1 should be supported in dispatcher");
        let values: Vec<f32> = result.unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-2,
                "Q8_1 dispatcher mismatch at {i}: got {v}, expected {i}",
            );
        }
    }

    #[test]
    fn dequantize_q8_1_to_q8_0_round_trip() {
        // Test the full Q8_1 -> F32 -> Q8_0 pipeline
        let scale_bits = f32_to_f16_bits(0.1);
        let min_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }

        // Step 1: Dequant Q8_1 -> F32
        let f32_bytes = dequantize_q8_1(&block, 32);
        let f32_values: Vec<f32> = f32_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // Step 2: Quantize F32 -> Q8_0
        let q8_0_bytes = quantize_f32_to_q8_0(&f32_bytes, 32);
        assert_eq!(q8_0_bytes.len(), 34, "Q8_0 block should be 34 bytes");

        // Step 3: Dequant Q8_0 -> F32 and compare
        let f32_roundtrip = dequantize_q8_0(&q8_0_bytes, 32);
        let rt_values: Vec<f32> = f32_roundtrip
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        for (i, (&orig, &rt)) in f32_values.iter().zip(rt_values.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < 0.02,
                "Q8_1->Q8_0 round-trip mismatch at {i}: orig={orig}, rt={rt}",
            );
        }
    }

    #[test]
    fn dequantize_q5_1_known_block() {
        // Q5_1: [f16 scale][f16 min][4 bytes qh][16 bytes qs] = 24 bytes
        // val[i] = scale * (nibble | (high_bit << 4)) + min
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        // All high bits = 0
        block.extend_from_slice(&0u32.to_le_bytes());
        // All nibbles = 0x33 -> lo=3, hi=3
        block.resize(block.len() + 16, 0x33);
        assert_eq!(block.len(), 24);

        let result = dequantize_q5_1(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 32);
        // All values should be 3.0 (lo nibble 3 or hi nibble 3, no high bits)
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-2,
                "Q5_1 mismatch at {i}: got {v}, expected 3.0"
            );
        }
    }

    #[test]
    fn dequantize_q5_1_with_min_and_high_bits() {
        // Q5_1 with non-zero min and some high bits set
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(10.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        // Set high bit for element 0 only
        let qh: u32 = 1;
        block.extend_from_slice(&qh.to_le_bytes());
        // All nibbles = 0x00
        block.resize(block.len() + 16, 0x00);

        let result = dequantize_q5_1(&block, 32);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // Element 0: scale * (0 | (1 << 4)) + min = 1.0 * 16 + 10.0 = 26.0
        assert!(
            (values[0] - 26.0).abs() < 1e-2,
            "Q5_1 high-bit mismatch: got {}, expected 26.0", values[0]
        );
        // Element 1: scale * (0 | (0 << 4)) + min = 0 + 10.0 = 10.0
        assert!(
            (values[1] - 10.0).abs() < 1e-2,
            "Q5_1 no-high-bit mismatch: got {}, expected 10.0", values[1]
        );
    }

    #[test]
    fn dequantize_q5_1_via_dispatcher() {
        let scale_bits = f32_to_f16_bits(1.0);
        let min_bits = f32_to_f16_bits(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bits.to_le_bytes());
        block.extend_from_slice(&min_bits.to_le_bytes());
        block.extend_from_slice(&0u32.to_le_bytes());
        block.resize(block.len() + 16, 0x00);

        let result = dequantize_to_f32_bytes(&block, GgmlType::Q5_1, 32, "test");
        assert!(result.is_ok(), "Q5_1 should be supported in dispatcher");
    }

    #[test]
    fn has_dequant_path_covers_all_supported_types() {
        // Every type handled by dequantize_to_f32_bytes must return true
        let supported = [
            GgmlType::F32, GgmlType::F16, GgmlType::BF16,
            GgmlType::Q8_0, GgmlType::Q8_1,
            GgmlType::Q4_0, GgmlType::Q4_1,
            GgmlType::Q5_0, GgmlType::Q5_1,
            GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K,
            GgmlType::Q2_K, GgmlType::Q3_K,
            GgmlType::MXFP4,
        ];
        for t in supported {
            assert!(t.has_dequant_path(), "{t:?} should have a dequant path");
        }

        // Types without dequant path
        let unsupported = [
            GgmlType::Q8_K, GgmlType::F64,
            GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64,
            GgmlType::IQ2_XXS, GgmlType::IQ2_XS, GgmlType::IQ2_S,
            GgmlType::IQ3_XXS, GgmlType::IQ3_S,
            GgmlType::IQ1_S, GgmlType::IQ1_M,
            GgmlType::IQ4_NL, GgmlType::IQ4_XS,
            GgmlType::Unknown(9999),
        ];
        for t in unsupported {
            assert!(!t.has_dequant_path(), "{t:?} should NOT have a dequant path");
        }
    }

    #[test]
    fn round_trip_q8_0_dequantize() {
        let dir = temp_dir();
        let lbc_path = dir.join("q8_deq.lbc");

        let hidden = 32usize;
        let vocab = 32usize;
        let num_heads = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = hidden / num_heads as usize;
        let q_dim = (num_heads as usize) * head_dim;
        let kv_dim = (num_kv_heads as usize) * head_dim;
        let inter = 64usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 1);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden as u32);
        builder.add_u32("llama.feed_forward_length", inter as u32);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        let embedding_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.001) % 1.0)
            .collect();
        builder.add_f32_tensor(
            EMBEDDING_NAME,
            &[vocab as u64, hidden as u64],
            &embedding_data,
        );
        builder.add_f32_tensor(
            FINAL_NORM_NAME,
            &[hidden as u64],
            &vec![1.0; hidden],
        );
        builder.add_f32_tensor(
            OUTPUT_PROJ_NAME,
            &[vocab as u64, hidden as u64],
            &vec![0.1; vocab * hidden],
        );

        fn build_q8_0_blocks_inner(values: &[f32]) -> Vec<u8> {
            let mut data = Vec::new();
            for chunk in values.chunks(32) {
                let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 127.0;
                let bits = f32::to_bits(scale);
                let sign = (bits >> 31) & 1;
                let exp = ((bits >> 23) & 0xFF) as i32 - 127;
                let frac = bits & 0x7FFFFF;
                let scale_bits = if scale == 0.0 {
                    if scale.is_sign_negative() { 0x8000u16 } else { 0u16 }
                } else if exp > 15 {
                    if sign == 1 { 0xFC00 } else { 0x7C00 }
                } else if exp < -14 {
                    if sign == 1 { 0x8000 } else { 0 }
                } else {
                    let f16_exp = (exp + 15) as u16;
                    let f16_frac = (frac >> 13) as u16;
                    ((sign as u16) << 15) | (f16_exp << 10) | f16_frac
                };
                data.extend_from_slice(&scale_bits.to_le_bytes());
                for &v in chunk {
                    let q = if scale > 0.0 {
                        (v / scale).round().clamp(-128.0, 127.0) as i8
                    } else {
                        0i8
                    };
                    data.push(q as u8);
                }
            }
            data
        }

        let layer_tensor_specs: Vec<(&str, usize, usize)> = vec![
            (ATTN_Q, q_dim, hidden),
            (ATTN_K, kv_dim, hidden),
            (ATTN_V, kv_dim, hidden),
            (ATTN_OUTPUT, hidden, q_dim),
            (FFN_GATE, inter, hidden),
            (FFN_UP, inter, hidden),
            (FFN_DOWN, hidden, inter),
            (ATTN_NORM, hidden, 1),
            (FFN_NORM, hidden, 1),
        ];

        for (suffix, rows, cols) in &layer_tensor_specs {
            let name = layer_tensor_name(0, suffix);
            let n = rows * cols;
            let f32_values: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.01 - 0.5).clamp(-1.0, 1.0))
                .collect();

            if *suffix == ATTN_NORM || *suffix == FFN_NORM {
                builder.add_f32_tensor(&name, &[n as u64], &f32_values);
            } else {
                let q8_data = build_q8_0_blocks_inner(&f32_values);
                let dims = if *cols == 1 {
                    vec![*rows as u64]
                } else {
                    vec![*rows as u64, *cols as u64]
                };
                builder.add_tensor(&name, GgmlType::Q8_0, &dims, q8_data);
            }
        }

        let gguf_data = builder.build();

        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: true,
            requant_to: None,
        };
        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.quant_scheme, QuantScheme::F32);

        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(
            lbc.header.quantization.scheme,
            QuantScheme::F32,
            "LBC should report F32 quant scheme when dequantized"
        );

        let idx = &lbc.layer_indices[0];
        assert_eq!(
            idx.subtensors.wq.length,
            (q_dim * hidden * 4) as u64,
            "wq should be F32 size"
        );
        assert_eq!(
            idx.subtensors.wq.quant,
            QuantScheme::F32,
            "wq quant should be F32"
        );
        assert_eq!(
            idx.subtensors.attn_norm.length,
            (hidden * 4) as u64,
            "attn_norm should be F32 size"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dequantize_q8_0_two_blocks() {
        let scale_bits = f32_to_f16_bits(1.0);
        let mut data = Vec::new();
        data.extend_from_slice(&scale_bits.to_le_bytes());
        for i in 0..32i8 { data.push(i as u8); }
        let scale2_bits = f32_to_f16_bits(0.5);
        data.extend_from_slice(&scale2_bits.to_le_bytes());
        for i in 0..32i8 { data.push((i * 2) as u8); }

        let result = dequantize_q8_0(&data, 64);
        let values: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(values.len(), 64);
        for (i, &v) in values[..32].iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-2, "block1[{i}]: got {v}, expected {i}");
        }
        for (i, &v) in values[32..64].iter().enumerate() {
            let expected = 0.5 * (i * 2) as f32;
            assert!((v - expected).abs() < 1e-2, "block2[{i}]: got {v}, expected {expected}");
        }
    }

    #[test]
    fn round_trip_dequantize_f32_noop() {
        let dir = temp_dir();
        let lbc_path = dir.join("f32_deq.lbc");

        let gguf_data = build_test_gguf(1, 2, 2, 8, 16, 32);
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: true,
            requant_to: None,
        };

        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.quant_scheme, QuantScheme::F32);

        let lbc_path2 = dir.join("f32_normal.lbc");
        let opts2 = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };
        let stats2 =
            convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path2, &opts2).unwrap();
        assert_eq!(stats2.quant_scheme, QuantScheme::F32);

        let lbc1 = LbcFile::open(&lbc_path).unwrap();
        let lbc2 = LbcFile::open(&lbc_path2).unwrap();
        assert_eq!(
            lbc1.layer_indices[0].layer_length_bytes,
            lbc2.layer_indices[0].layer_length_bytes,
            "F32 dequantize should produce same layer sizes as pass-through"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gguf_v2_converter_roundtrip() {
        let dir = temp_dir();
        let lbc_path = dir.join("v2_test.lbc");

        let hidden = 8usize;
        let vocab = 32usize;
        let num_heads = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = hidden / num_heads as usize;
        let q_dim = num_heads as usize * head_dim;
        let kv_dim = num_kv_heads as usize * head_dim;
        let inter = 16usize;

        let mut builder = GgufBuilder::new();
        builder.version(2);

        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 1);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden as u32);
        builder.add_u32("llama.feed_forward_length", inter as u32);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        let tokens: Vec<String> = (0..vocab).map(|i| format!("t{i}")).collect();
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        builder.add_string_array("tokenizer.ggml.tokens", &token_refs);

        let embedding_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.001) % 1.0)
            .collect();
        builder.add_f32_tensor(EMBEDDING_NAME, &[vocab as u64, hidden as u64], &embedding_data);
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &vec![1.0; hidden]);
        builder.add_f32_tensor(OUTPUT_PROJ_NAME, &[vocab as u64, hidden as u64], &vec![0.2; vocab * hidden]);

        let layer_tensors: Vec<(&str, usize, usize)> = vec![
            (ATTN_Q, q_dim, hidden), (ATTN_K, kv_dim, hidden), (ATTN_V, kv_dim, hidden),
            (ATTN_OUTPUT, hidden, q_dim), (FFN_GATE, inter, hidden), (FFN_UP, inter, hidden),
            (FFN_DOWN, hidden, inter), (ATTN_NORM, hidden, 1), (FFN_NORM, hidden, 1),
        ];

        for (suffix, rows, cols) in layer_tensors {
            let name = layer_tensor_name(0, suffix);
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) % 1.0).collect();
            let dims = if cols == 1 { vec![rows as u64] } else { vec![rows as u64, cols as u64] };
            builder.add_f32_tensor(&name, &dims, &data);
        }

        let gguf_data = builder.build();
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();
        assert_eq!(gguf.version, 2);

        let opts = ConvertOptions { alignment: 128, dequantize_to_f32: false, requant_to: None };
        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.architecture, "llama");
        assert_eq!(stats.quant_scheme, QuantScheme::F32);

        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(lbc.header.num_layers, 1);
        assert_eq!(lbc.header.hyperparams.hidden_dim, hidden as u32);
        assert_eq!(lbc.header.hyperparams.vocab_size, vocab as u32);
        assert_eq!(lbc.header.embedding.length, (vocab * hidden * 4) as u64);
        assert_eq!(lbc.header.final_norm.length, (hidden * 4) as u64);
        assert_eq!(lbc.header.output_proj.length, (vocab * hidden * 4) as u64);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn convert_stats_display() {
        let stats = ConvertStats {
            input_size: 10 * 1024 * 1024,
            output_size: 8 * 1024 * 1024,
            num_layers: 32,
            architecture: "llama".into(),
            tensor_count: 291,
            quant_scheme: QuantScheme::Q4_K,
        };
        let s = format!("{stats}");
        assert!(s.contains("291 tensors"));
        assert!(s.contains("32 layers"));
        assert!(s.contains("llama"));
    }

    fn build_test_gguf_moe(
        num_layers: u32, num_heads: u32, num_kv_heads: u32, hidden_dim: u32,
        intermediate_dim: u32, vocab_size: u32, num_experts: u32, num_active_experts: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let hidden = hidden_dim as usize;
        let inter = intermediate_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", num_layers);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden_dim);
        builder.add_u32("llama.feed_forward_length", intermediate_dim);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
        builder.add_u32("llama.expert_count", num_experts);
        builder.add_u32("llama.expert_used_count", num_active_experts);

        let embedding_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.001) % 1.0).collect();
        builder.add_f32_tensor(EMBEDDING_NAME, &[vocab as u64, hidden as u64], &embedding_data);
        let norm_data: Vec<f32> = vec![1.0; hidden];
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &norm_data);
        let output_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.002) % 1.0).collect();
        builder.add_f32_tensor(OUTPUT_PROJ_NAME, &[vocab as u64, hidden as u64], &output_data);

        for layer in 0..num_layers as usize {
            let attn_tensors: Vec<(&str, usize, usize)> = vec![
                (ATTN_Q, q_dim, hidden), (ATTN_K, kv_dim, hidden),
                (ATTN_V, kv_dim, hidden), (ATTN_OUTPUT, hidden, q_dim),
            ];
            for (suffix, rows, cols) in attn_tensors {
                let name = layer_tensor_name(layer, suffix);
                let n = rows * cols;
                let data: Vec<f32> = (0..n).map(|i| ((i + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&name, &[rows as u64, cols as u64], &data);
            }

            let norm: Vec<f32> = vec![1.0; hidden];
            builder.add_f32_tensor(&layer_tensor_name(layer, ATTN_NORM), &[hidden as u64], &norm);
            builder.add_f32_tensor(&layer_tensor_name(layer, FFN_NORM), &[hidden as u64], &norm);

            let router_data: Vec<f32> = (0..(num_experts as usize * hidden)).map(|i| (i as f32 * 0.003) % 1.0).collect();
            builder.add_f32_tensor(&layer_tensor_name(layer, FFN_GATE_INP), &[num_experts as u64, hidden as u64], &router_data);

            for e in 0..num_experts as usize {
                let gate_data: Vec<f32> = (0..inter * hidden).map(|i| ((i + e * 100 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "gate", e), &[inter as u64, hidden as u64], &gate_data);
                let up_data: Vec<f32> = (0..inter * hidden).map(|i| ((i + e * 200 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "up", e), &[inter as u64, hidden as u64], &up_data);
                let down_data: Vec<f32> = (0..hidden * inter).map(|i| ((i + e * 300 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "down", e), &[hidden as u64, inter as u64], &down_data);
            }
        }

        builder.build()
    }

    #[test]
    fn moe_round_trip_f32() {
        let dir = temp_dir();
        let lbc_path = dir.join("moe_test.lbc");
        let num_experts = 4u32;
        let num_active = 2u32;
        let gguf_data = build_test_gguf_moe(2, 2, 2, 8, 16, 32, num_experts, num_active);
        let opts = ConvertOptions { alignment: 128, dequantize_to_f32: false, requant_to: None };

        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.architecture, "llama");
        assert_eq!(stats.quant_scheme, QuantScheme::F32);

        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.num_experts, Some(num_experts));
        assert_eq!(lbc.header.hyperparams.num_active_experts, Some(num_active));
        assert!(lbc.header.has_expert_index);

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
            let st = &idx.subtensors;
            assert!(st.router_weight.is_some(), "layer {i}: missing router_weight");
            let experts = st.experts.as_ref().expect(&format!("layer {i}: missing experts"));
            assert_eq!(experts.len(), num_experts as usize);
            assert_eq!(st.w_gate.length, 0);
            assert_eq!(st.w_up.length, 0);
            assert_eq!(st.w_down.length, 0);
            for (e, expert) in experts.iter().enumerate() {
                assert!(expert.gate.length > 0, "layer {i} expert {e}: gate length=0");
                assert!(expert.up.length > 0, "layer {i} expert {e}: up length=0");
                assert!(expert.down.length > 0, "layer {i} expert {e}: down length=0");
            }
            let router = st.router_weight.as_ref().unwrap();
            let expected_router_size = (num_experts as u64) * 8 * 4;
            assert_eq!(router.length, expected_router_size);
        }

        assert_eq!(lbc.header.embedding.length, 32 * 8 * 4);
        assert_eq!(lbc.header.final_norm.length, 8 * 4);
        assert_eq!(lbc.header.output_proj.length, 32 * 8 * 4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn moe_hyperparams_extraction() {
        let gguf_data = build_test_gguf_moe(2, 2, 2, 8, 16, 32, 8, 2);
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        let (hp, arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(arch, "llama");
        assert_eq!(hp.num_experts, Some(8));
        assert_eq!(hp.num_active_experts, Some(2));
        assert!(hp.is_moe());
    }

    #[test]
    fn moe_tensor_naming() {
        assert_eq!(layer_tensor_name(0, FFN_GATE_INP), "blk.0.ffn_gate_inp.weight");
        assert_eq!(expert_tensor_name(0, "gate", 0), "blk.0.ffn_gate.0.weight");
        assert_eq!(expert_tensor_name(0, "up", 3), "blk.0.ffn_up.3.weight");
        assert_eq!(expert_tensor_name(1, "down", 7), "blk.1.ffn_down.7.weight");
    }

    fn build_test_gguf_moe_f16_router(
        num_layers: u32, num_heads: u32, num_kv_heads: u32, hidden_dim: u32,
        intermediate_dim: u32, vocab_size: u32, num_experts: u32, num_active_experts: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let hidden = hidden_dim as usize;
        let inter = intermediate_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", num_layers);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden_dim);
        builder.add_u32("llama.feed_forward_length", intermediate_dim);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
        builder.add_u32("llama.expert_count", num_experts);
        builder.add_u32("llama.expert_used_count", num_active_experts);

        let embedding_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.001) % 1.0).collect();
        builder.add_f32_tensor(EMBEDDING_NAME, &[vocab as u64, hidden as u64], &embedding_data);
        let norm_data: Vec<f32> = vec![1.0; hidden];
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &norm_data);
        let output_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.002) % 1.0).collect();
        builder.add_f32_tensor(OUTPUT_PROJ_NAME, &[vocab as u64, hidden as u64], &output_data);

        for layer in 0..num_layers as usize {
            let attn_tensors: Vec<(&str, usize, usize)> = vec![
                (ATTN_Q, q_dim, hidden), (ATTN_K, kv_dim, hidden),
                (ATTN_V, kv_dim, hidden), (ATTN_OUTPUT, hidden, q_dim),
            ];
            for (suffix, rows, cols) in attn_tensors {
                let name = layer_tensor_name(layer, suffix);
                let n = rows * cols;
                let data: Vec<f32> = (0..n).map(|i| ((i + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&name, &[rows as u64, cols as u64], &data);
            }

            let norm: Vec<f32> = vec![1.0; hidden];
            builder.add_f32_tensor(&layer_tensor_name(layer, ATTN_NORM), &[hidden as u64], &norm);
            builder.add_f32_tensor(&layer_tensor_name(layer, FFN_NORM), &[hidden as u64], &norm);

            let router_n = num_experts as usize * hidden;
            let router_f32: Vec<f32> = (0..router_n).map(|i| (i as f32 * 0.003) % 1.0).collect();
            let router_f16_bytes: Vec<u8> = router_f32.iter().flat_map(|&v| {
                f32_to_f16_bits_convert(v).to_le_bytes()
            }).collect();
            builder.add_tensor(
                &layer_tensor_name(layer, FFN_GATE_INP),
                GgmlType::F16,
                &[num_experts as u64, hidden as u64],
                router_f16_bytes,
            );

            for e in 0..num_experts as usize {
                let gate_data: Vec<f32> = (0..inter * hidden).map(|i| ((i + e * 100 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "gate", e), &[inter as u64, hidden as u64], &gate_data);
                let up_data: Vec<f32> = (0..inter * hidden).map(|i| ((i + e * 200 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "up", e), &[inter as u64, hidden as u64], &up_data);
                let down_data: Vec<f32> = (0..hidden * inter).map(|i| ((i + e * 300 + layer * 1000) as f32 * 0.001) % 1.0).collect();
                builder.add_f32_tensor(&expert_tensor_name(layer, "down", e), &[hidden as u64, inter as u64], &down_data);
            }
        }

        builder.build()
    }

    #[test]
    fn moe_f16_router_forced_to_f32() {
        let dir = temp_dir();
        let lbc_path = dir.join("moe_f16_router.lbc");
        let num_experts = 4u32;
        let num_active = 2u32;
        let hidden_dim = 8u32;
        let gguf_data = build_test_gguf_moe_f16_router(2, 2, 2, hidden_dim, 16, 32, num_experts, num_active);

        let opts = ConvertOptions { alignment: 128, dequantize_to_f32: false, requant_to: None };
        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 2);

        let lbc = LbcFile::open(&lbc_path).unwrap();
        assert_eq!(lbc.header.hyperparams.num_experts, Some(num_experts));

        for (i, idx) in lbc.layer_indices.iter().enumerate() {
            idx.validate(i).unwrap();
            let st = &idx.subtensors;
            let router = st.router_weight.as_ref().expect(&format!("layer {i}: missing router_weight"));
            assert_eq!(router.quant, QuantScheme::F32);
            let expected_size = (num_experts as u64) * (hidden_dim as u64) * 4;
            assert_eq!(router.length, expected_size);
            let experts = st.experts.as_ref().expect(&format!("layer {i}: missing experts"));
            assert_eq!(experts.len(), num_experts as usize);
            for (e, expert) in experts.iter().enumerate() {
                assert!(expert.gate.length > 0, "layer {i} expert {e}: gate length=0");
                assert!(expert.up.length > 0, "layer {i} expert {e}: up length=0");
                assert!(expert.down.length > 0, "layer {i} expert {e}: down length=0");
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn moe_f16_router_values_correct() {
        let dir = temp_dir();
        let lbc_path = dir.join("moe_f16_router_vals.lbc");
        let num_experts = 4u32;
        let hidden_dim = 8u32;
        let gguf_data = build_test_gguf_moe_f16_router(1, 2, 2, hidden_dim, 16, 32, num_experts, 2);

        let opts = ConvertOptions { alignment: 128, dequantize_to_f32: false, requant_to: None };
        convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();

        let lbc = LbcFile::open(&lbc_path).unwrap();
        let idx = &lbc.layer_indices[0];
        let router = idx.subtensors.router_weight.as_ref().unwrap();

        let lbc_bytes = std::fs::read(&lbc_path).unwrap();
        let layer_blob_start = idx.layer_offset_bytes as usize;
        let router_start = layer_blob_start + router.offset as usize;
        let router_end = router_start + router.length as usize;
        let router_bytes = &lbc_bytes[router_start..router_end];

        let router_n = num_experts as usize * hidden_dim as usize;
        assert_eq!(router_bytes.len(), router_n * 4);
        let mut recovered_f32 = Vec::with_capacity(router_n);
        for chunk in router_bytes.chunks_exact(4) {
            recovered_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let expected_f32: Vec<f32> = (0..router_n).map(|i| (i as f32 * 0.003) % 1.0).collect();
        for (i, (&got, &expected)) in recovered_f32.iter().zip(expected_f32.iter()).enumerate() {
            let expected_via_f16 = f16_to_f32(f32_to_f16_bits_convert(expected));
            assert!(
                (got - expected_via_f16).abs() < 1e-6,
                "router value [{i}]: got {got}, expected {expected_via_f16} (original {expected})"
            );
        }

        for (i, &v) in recovered_f32.iter().enumerate() {
            assert!(v.is_finite(), "router value [{i}] is not finite: {v}");
            assert!(v >= -0.01 && v <= 1.01, "router value [{i}] out of expected range: {v}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn e8m0_to_f32_half_known_values() {
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(126), 0.25);
        assert!(e8m0_to_f32_half(0) > 0.0);
        assert!(e8m0_to_f32_half(0) < 1e-37);
        assert_eq!(e8m0_to_f32_half(1), e8m0_to_f32_half(0) * 2.0);
    }

    #[test]
    fn dequantize_mxfp4_single_block() {
        let mut block = [0u8; 17];
        block[0] = 128;
        block[1] = 0x10;
        block[2] = 0x32;
        for i in 3..17 { block[i] = 0x00; }

        let out = dequantize_mxfp4(&block, 32);
        let f32s: Vec<f32> = out.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(f32s.len(), 32);
        assert_eq!(f32s[0], 0.0);
        assert_eq!(f32s[1], 2.0);
        assert_eq!(f32s[16], 1.0);
        assert_eq!(f32s[17], 3.0);
        for i in 2..16 {
            assert_eq!(f32s[i], 0.0, "position {i} should be 0.0");
        }
    }

    #[test]
    fn dequantize_mxfp4_negative_values() {
        let mut block = [0u8; 17];
        block[0] = 128;
        block[1] = 0x9A;
        block[2] = 0xFE;
        for i in 3..17 { block[i] = 0x00; }

        let out = dequantize_mxfp4(&block, 32);
        let f32s: Vec<f32> = out.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(f32s[0], -2.0);
        assert_eq!(f32s[1], -8.0);
        assert_eq!(f32s[16], -1.0);
        assert_eq!(f32s[17], -12.0);
    }

    #[test]
    fn dequantize_mxfp4_with_scaling() {
        let mut block = [0u8; 17];
        block[0] = 126;
        block[1] = 0x47;
        for i in 2..17 { block[i] = 0x00; }

        let out = dequantize_mxfp4(&block, 32);
        let f32s: Vec<f32> = out.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(f32s[0], 12.0 * 0.25);
        assert_eq!(f32s[16], 4.0 * 0.25);
    }

    #[test]
    fn dequantize_mxfp4_multiple_blocks() {
        let mut data = vec![0u8; 34];
        data[0] = 128;
        data[1] = 0x01;
        data[17] = 129;
        data[18] = 0x05;

        let out = dequantize_mxfp4(&data, 64);
        let f32s: Vec<f32> = out.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(f32s.len(), 64);
        assert_eq!(f32s[0], 1.0);
        assert_eq!(f32s[32], 12.0);
    }

    #[test]
    fn dequantize_mxfp4_in_dequantize_to_f32_bytes() {
        let mut block = [0u8; 17];
        block[0] = 128;
        block[1] = 0x31;

        let result = dequantize_to_f32_bytes(&block, GgmlType::MXFP4, 32, "test");
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let f32s: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(f32s[0], 1.0);
        assert_eq!(f32s[16], 3.0);
    }

    /// Build synthetic Q6_K blocks from F32 values.
    /// Q6_K: 256 elements per block, 210 bytes per block.
    /// Layout: [128 bytes ql][64 bytes qh][16 bytes scales][2 bytes f16 d]
    /// We build trivial blocks (all zeros with zero scale) for testing
    /// that the converter handles Q6_K output_proj correctly.
    fn build_q6_k_zero_blocks(n_elements: usize) -> Vec<u8> {
        assert!(n_elements % 256 == 0, "Q6_K requires elements divisible by 256");
        let num_blocks = n_elements / 256;
        // Each block: 210 bytes of zeros = all-zero dequantized output
        vec![0u8; num_blocks * 210]
    }

    #[test]
    fn output_proj_q6_k_requantized_to_q8_0() {
        // When output.weight is Q6_K (common from llama-quantize), the converter
        // should requantize it to Q8_0 instead of falling back to F32.
        let dir = temp_dir();
        let lbc_path = dir.join("q6k_output.lbc");

        let hidden = 32usize;
        let vocab = 256usize; // 256*32 = 8192 elements, divisible by 256 (Q6_K block)
        let num_heads = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = hidden / num_heads as usize;
        let q_dim = num_heads as usize * head_dim;
        let kv_dim = num_kv_heads as usize * head_dim;
        let inter = 64usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 1);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden as u32);
        builder.add_u32("llama.feed_forward_length", inter as u32);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        // Global tensors: embedding and final_norm as F32, output_proj as Q6_K
        let embedding_data: Vec<f32> = vec![0.1; vocab * hidden];
        builder.add_f32_tensor(EMBEDDING_NAME, &[vocab as u64, hidden as u64], &embedding_data);
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &vec![1.0; hidden]);

        // Q6_K output_proj: 8192 elements = 32 blocks * 210 bytes = 6720 bytes
        let q6k_data = build_q6_k_zero_blocks(vocab * hidden);
        builder.add_tensor(OUTPUT_PROJ_NAME, GgmlType::Q6_K, &[vocab as u64, hidden as u64], q6k_data);

        // Per-layer tensors as F32
        for suffix in &LAYER_TENSOR_SUFFIXES {
            let name = layer_tensor_name(0, suffix);
            let (rows, cols) = match *suffix {
                ATTN_Q => (q_dim, hidden),
                ATTN_K => (kv_dim, hidden),
                ATTN_V => (kv_dim, hidden),
                ATTN_OUTPUT => (hidden, q_dim),
                FFN_GATE => (inter, hidden),
                FFN_UP => (inter, hidden),
                FFN_DOWN => (hidden, inter),
                ATTN_NORM => (hidden, 1),
                FFN_NORM => (hidden, 1),
                _ => unreachable!(),
            };
            let n = rows * cols;
            let data: Vec<f32> = vec![0.01; n];
            let dims = if cols == 1 { vec![rows as u64] } else { vec![rows as u64, cols as u64] };
            builder.add_f32_tensor(&name, &dims, &data);
        }

        let gguf_data = builder.build();

        // Convert without explicit requant target -- K-quant should be requantized to Q8_0
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: None,
        };
        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 1);

        let lbc = LbcFile::open(&lbc_path).unwrap();

        // Output proj should be Q8_0 (not F32)
        assert_eq!(
            lbc.header.output_proj.quant,
            QuantScheme::Q8_0,
            "Q6_K output.weight should be requantized to Q8_0, not stored as F32"
        );

        // Q8_0 size: 8192 elements / 32 * 34 bytes = 8704 bytes
        let expected_q8_size = ((vocab * hidden) / 32) * 34;
        assert_eq!(
            lbc.header.output_proj.length,
            expected_q8_size as u64,
            "output_proj length should match Q8_0 encoding"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn output_proj_q6_k_with_requant_q4_0() {
        // When requant_to=Q4_0 and output.weight is Q6_K, it should be
        // handled by the existing requant_target=Q4_0 branch (line 269).
        let dir = temp_dir();
        let lbc_path = dir.join("q6k_requant_q4.lbc");

        let hidden = 32usize;
        let vocab = 256usize;
        let num_heads = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = hidden / num_heads as usize;
        let q_dim = num_heads as usize * head_dim;
        let kv_dim = num_kv_heads as usize * head_dim;
        let inter = 64usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 1);
        builder.add_u32("llama.attention.head_count", num_heads);
        builder.add_u32("llama.attention.head_count_kv", num_kv_heads);
        builder.add_u32("llama.embedding_length", hidden as u32);
        builder.add_u32("llama.feed_forward_length", inter as u32);
        builder.add_u32("llama.context_length", 64);
        builder.add_f32("llama.rope.freq_base", 10000.0);
        builder.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

        let embedding_data: Vec<f32> = vec![0.1; vocab * hidden];
        builder.add_f32_tensor(EMBEDDING_NAME, &[vocab as u64, hidden as u64], &embedding_data);
        builder.add_f32_tensor(FINAL_NORM_NAME, &[hidden as u64], &vec![1.0; hidden]);

        let q6k_data = build_q6_k_zero_blocks(vocab * hidden);
        builder.add_tensor(OUTPUT_PROJ_NAME, GgmlType::Q6_K, &[vocab as u64, hidden as u64], q6k_data);

        for suffix in &LAYER_TENSOR_SUFFIXES {
            let name = layer_tensor_name(0, suffix);
            let (rows, cols) = match *suffix {
                ATTN_Q => (q_dim, hidden),
                ATTN_K => (kv_dim, hidden),
                ATTN_V => (kv_dim, hidden),
                ATTN_OUTPUT => (hidden, q_dim),
                FFN_GATE => (inter, hidden),
                FFN_UP => (inter, hidden),
                FFN_DOWN => (hidden, inter),
                ATTN_NORM => (hidden, 1),
                FFN_NORM => (hidden, 1),
                _ => unreachable!(),
            };
            let n = rows * cols;
            let data: Vec<f32> = vec![0.01; n];
            let dims = if cols == 1 { vec![rows as u64] } else { vec![rows as u64, cols as u64] };
            builder.add_f32_tensor(&name, &dims, &data);
        }

        let gguf_data = builder.build();

        // Convert with requant_to=Q4_0
        let opts = ConvertOptions {
            alignment: 128,
            dequantize_to_f32: false,
            requant_to: Some(QuantScheme::Q4_0),
        };
        let stats = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &opts).unwrap();
        assert_eq!(stats.num_layers, 1);

        let lbc = LbcFile::open(&lbc_path).unwrap();

        // Output proj should be Q4_0
        assert_eq!(
            lbc.header.output_proj.quant,
            QuantScheme::Q4_0,
            "Q6_K output.weight with requant_to=Q4_0 should be Q4_0"
        );

        // Q4_0 size: 8192 elements / 32 * 18 bytes = 4608 bytes
        let expected_q4_size = ((vocab * hidden) / 32) * 18;
        assert_eq!(
            lbc.header.output_proj.length,
            expected_q4_size as u64,
            "output_proj length should match Q4_0 encoding"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
