//! GGUF-to-LBC converter.
//!
//! Reads a GGUF file, extracts hyperparameters and tensor
//! data, and writes an LBC file using the streaming writer. Memory-efficient:
//! only one layer blob is held in memory at a time.

use crate::arch;
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufError, GgufFile};
use crate::hyperparams::{detect_quant_scheme, extract_hyperparams, quant_descriptor_for};
use crate::sharded::{MultiShardReader, ShardError, ShardedGguf};
use crate::tensor_names::*;
use crate::tensor_io::read_tensor_data;
use lumen_format::header::LbcHeader;
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::StreamingLbcWriter;
use lumen_format::tokenizer::TokenizerSection;
use lumen_format::writer::GlobalTensors;
use std::fmt;
use std::io::{BufWriter, Read, Seek};
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Runtime backend the LBC is being prepared for.
///
/// Different GPU backends support different sets of quantization kernels.
/// CUDA ships dedicated K-quant (Q2/Q3/Q4/Q5/Q6_K) dequant kernels, so
/// K-quant layer tensors can ride through unchanged. Metal currently has
/// **no** K-quant kernels, so layer tensors stored as K-quant in the source
/// GGUF (e.g. `attn_q` in the Q4 MoE-30B GGUF) must be upcast to a
/// scheme Metal does support (Q8_0).
///
/// `Generic` leaves K-quant layer tensors as-is (CUDA path).
/// `Metal` upcasts any K-quant layer-projection tensor (attn_q/k/v, ffn_*)
/// to Q8_0 at convert time -- numerically lossless within Q8_0 precision
/// (Q8_0 is higher precision per element than Q6_K in practice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvertTarget {
    /// CUDA-style: keep K-quant layer tensors as-is.
    #[default]
    Generic,
    /// Metal-style: upcast K-quant layer tensors to Q8_0 (Metal has no K-quant
    /// dispatch kernels, would hit the slow F32 fallback or NaN out).
    Metal,
}

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
    /// Runtime backend the LBC is being prepared for. See [`ConvertTarget`].
    pub target: ConvertTarget,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            alignment: 128 * 1024,
            dequantize_to_f32: false,
            requant_to: None,
            target: ConvertTarget::default(),
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

/// Convert a GGUF file (single-shard or multi-shard) to LBC format.
///
/// Auto-detects multi-shard GGUFs by parsing the filename for the
/// `*-NNNNN-of-MMMMM.gguf` pattern. For a single-file GGUF the conversion is
/// byte-identical to the legacy path. For a multi-shard GGUF, all sibling
/// shards in the same directory are discovered, validated (consistent
/// `split.count`, contiguous `split.no`, no tensor-name collisions, matching
/// GGUF version + alignment), and presented as a single merged view to the
/// downstream pipeline.
///
/// Reads the GGUF header and metadata to extract model hyperparameters, then
/// streams tensor data layer-by-layer to produce the LBC file. Peak memory
/// usage is O(1 layer blob + global tensors) -- the 68 GB of BF16 weights in a
/// large model are never loaded simultaneously, regardless of shard count.
pub fn convert_gguf_to_lbc(
    gguf_path: &Path,
    lbc_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertStats, ConvertError> {
    let sharded = ShardedGguf::open(gguf_path).map_err(convert_error_from_shard)?;
    let input_size = sharded.total_disk_size()?;
    if sharded.shard_count() > 1 {
        eprintln!(
            "  Multi-shard GGUF: {} shards, {:.2} GB total",
            sharded.shard_count(),
            input_size as f64 / 1_073_741_824.0,
        );
    }
    let gguf = sharded.merged();
    let mut reader = MultiShardReader::open(&sharded)?;
    do_convert_from_reader(gguf, &mut reader, lbc_path, options, input_size)
}

/// Convert a sharded GGUF directly. Useful when the caller has already
/// constructed a [`ShardedGguf`] (e.g. to inspect metadata before deciding
/// whether to convert). Equivalent to [`convert_gguf_to_lbc`] with the path
/// resolved.
pub fn convert_sharded_gguf_to_lbc(
    sharded: &ShardedGguf,
    lbc_path: &Path,
    options: &ConvertOptions,
) -> Result<ConvertStats, ConvertError> {
    let input_size = sharded.total_disk_size()?;
    let gguf = sharded.merged();
    let mut reader = MultiShardReader::open(sharded)?;
    do_convert_from_reader(gguf, &mut reader, lbc_path, options, input_size)
}

/// Adapter: ShardError -> ConvertError. ShardError already implements
/// `Into<ConvertError>`, but the explicit function keeps `?` in the public
/// API tidy.
fn convert_error_from_shard(e: ShardError) -> ConvertError {
    e.into()
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
        GgmlType::BF16 => {
            // BF16 embedding: 2 bytes/elem, same on-disk footprint as F16.
            // Storing as F32 instead (the prior default fall-through) would
            // double the embedding size and trip A100-80GB OOM on Qwen3.5-9B
            // (2 GB BF16 -> 4 GB F32 just for embedding + same for output_proj).
            eprintln!("  Keeping embedding as Bf16 ({} bytes, {} elements)",
                embedding_bytes.len(), embedding_n_elements);
            (embedding_bytes, QuantScheme::Bf16)
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
        } else if output_proj_tensor.ggml_type == GgmlType::BF16 {
            // BF16 output_proj: 2 bytes/elem, fits in the runtime's Bf16Raw
            // upload path. Keeping the source bytes avoids the F32 inflation
            // that previously caused A100-80GB OOM during preload_weights on
            // Qwen3.5-9B BF16 (embedding 2 GB + output_proj 2 GB + 30+ GB of
            // F32-inflated layer weights blew past free VRAM).
            eprintln!("  Keeping output.weight as Bf16 ({} bytes, {} elements)",
                output_proj_bytes.len(), output_proj_tensor.n_elements());
            (output_proj_bytes, false, QuantScheme::Bf16)
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
            gguf, layer, opts.dequantize_to_f32, opts.requant_to, opts.target,
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
            &mut layer_blob, reader, gguf, layer, opts.dequantize_to_f32, opts.requant_to, opts.target,
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
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_convert_test_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a minimal synthetic Qwen3.5-arch GGUF with no real layer
    /// tensors -- enough for `extract_hyperparams` to read the metadata
    /// header but not enough to drive a full pipeline conversion. Used by
    /// the metadata-extraction unit tests below.
    ///
    /// The full converter pipeline is exercised by integration tests against
    /// real `bartowski/Qwen_Qwen3.5-9B-GGUF` files on the Modal benchmark
    /// harness (`modal/bench_real_models.py`), not by synthetic GGUFs --
    /// Qwen3.5 has 24 GDN layers interleaved with 8 full-attention layers
    /// and per-head Q/K RMSNorm, which a synthetic mini-GGUF cannot
    /// reproduce faithfully.
    fn build_minimal_qwen35_metadata_gguf(
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Vec<u8> {
        let head_dim = hidden_dim / num_heads;
        let hidden = hidden_dim as usize;
        let vocab = vocab_size as usize;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "qwen35");
        builder.add_u32("qwen35.block_count", num_layers);
        builder.add_u32("qwen35.attention.head_count", num_heads);
        builder.add_u32("qwen35.attention.head_count_kv", num_kv_heads);
        builder.add_u32("qwen35.attention.key_length", head_dim);
        builder.add_u32("qwen35.embedding_length", hidden_dim);
        builder.add_u32("qwen35.feed_forward_length", intermediate_dim);
        builder.add_u32("qwen35.context_length", 64);
        builder.add_f32("qwen35.rope.freq_base", 10000.0);
        builder.add_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-5);

        // Token embedding tensor satisfies the vocab_size discovery
        // fallback when `tokenizer.ggml.tokens` is absent.
        let embedding_data: Vec<f32> = vec![0.0; vocab * hidden];
        builder.add_f32_tensor(
            EMBEDDING_NAME,
            &[vocab as u64, hidden as u64],
            &embedding_data,
        );

        builder.build()
    }

    #[test]
    fn extract_hyperparams_qwen35() {
        let gguf_data = build_minimal_qwen35_metadata_gguf(4, 8, 4, 64, 128, 256);
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        let (hp, arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(arch, "qwen35");
        assert_eq!(hp.num_layers, 4);
        assert_eq!(hp.num_heads, 8);
        assert_eq!(hp.num_kv_heads, 4);
        assert_eq!(hp.head_dim, 8); // 64 / 8
        assert_eq!(hp.hidden_dim, 64);
        assert_eq!(hp.intermediate_dim, 128);
        assert_eq!(hp.vocab_size, 256);
        assert_eq!(hp.max_seq_len, 64);
        assert!((hp.norm_eps - 1e-5).abs() < 1e-10);
        // Qwen3.5 uses NeoX-style partial rotary embeddings.
        assert!(hp.rope_neox);

        let rope = hp.rope_params.unwrap();
        assert!((rope.theta - 10000.0).abs() < 0.01);
        assert!((rope.scaling_factor - 1.0).abs() < 0.01);
        assert_eq!(rope.scaling_type, lumen_format::hyperparams::RopeScalingType::None);
    }

    #[test]
    fn unsupported_architecture() {
        // `gpt2` and `llama` are both outside the Qwen3.5 family and must
        // be rejected with a clear UnsupportedArchitecture error.
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

    /// `convert_gguf_to_lbc` must route a single-file GGUF (no shard suffix)
    /// through the sharded path AND surface the legacy
    /// `UnsupportedArchitecture` error -- proving the new path is fully
    /// backward-compatible with the legacy single-file flow.
    #[test]
    fn convert_gguf_to_lbc_single_file_path_routing() {
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "gpt2");
        let gguf_data = builder.build();

        let dir = temp_dir();
        let in_path = dir.join("solo-routing.gguf");
        std::fs::write(&in_path, &gguf_data).unwrap();
        let lbc_path = dir.join("solo-routing.lbc");

        let result = convert_gguf_to_lbc(&in_path, &lbc_path, &ConvertOptions::default());
        match result {
            Err(ConvertError::UnsupportedArchitecture(a)) => assert_eq!(a, "gpt2"),
            other => panic!("expected UnsupportedArchitecture, got: {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `convert_gguf_to_lbc` must auto-discover sibling shards from a
    /// `*-NNNNN-of-MMMMM.gguf` filename, merge them, and surface a downstream
    /// converter error consistently. We exercise the routing with a
    /// `general.architecture = gpt2` payload split across 2 shards so the
    /// merge succeeds and downstream rejects with the expected error.
    #[test]
    fn convert_gguf_to_lbc_multi_shard_path_routing() {
        let dir = temp_dir();
        let stem = "multishard-routing";

        // Shard 1 carries the (unsupported) architecture string in metadata.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "gpt2");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_u64("split.tensors.count", 2);
        b1.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
        let p1 = dir.join(format!("{stem}-00001-of-00002.gguf"));
        std::fs::write(&p1, b1.build()).unwrap();

        // Shard 2 declares split metadata only and one tensor.
        let mut b2 = GgufBuilder::new();
        b2.add_string("general.architecture", "gpt2");
        b2.add_u16("split.no", 1);
        b2.add_u16("split.count", 2);
        b2.add_u64("split.tensors.count", 2);
        b2.add_f32_tensor("blk.0.attn_q.weight", &[4], &[0.0; 4]);
        let p2 = dir.join(format!("{stem}-00002-of-00002.gguf"));
        std::fs::write(&p2, b2.build()).unwrap();

        let lbc_path = dir.join(format!("{stem}.lbc"));
        let result = convert_gguf_to_lbc(&p1, &lbc_path, &ConvertOptions::default());
        match result {
            Err(ConvertError::UnsupportedArchitecture(a)) => assert_eq!(a, "gpt2"),
            other => panic!(
                "expected UnsupportedArchitecture after multi-shard merge, got: {other:?}"
            ),
        }

        // Sanity: the test really exercised the multi-shard path -- the same
        // routing succeeds when entering from shard 2's filename instead of
        // shard 1, proving sibling discovery + merge.
        let result_from_p2 = convert_gguf_to_lbc(&p2, &lbc_path, &ConvertOptions::default());
        match result_from_p2 {
            Err(ConvertError::UnsupportedArchitecture(a)) => assert_eq!(a, "gpt2"),
            other => panic!(
                "expected UnsupportedArchitecture from shard 2 entry, got: {other:?}"
            ),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `convert_gguf_to_lbc` must surface the multi-shard validation errors
    /// (e.g. missing sibling shards) as I/O errors rather than panicking or
    /// silently treating the file as single-shard.
    #[test]
    fn convert_gguf_to_lbc_missing_shard_error() {
        let dir = temp_dir();
        let stem = "missing-shard";

        // Write only shard 1 but claim total=2.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "gpt2");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
        let p1 = dir.join(format!("{stem}-00001-of-00002.gguf"));
        std::fs::write(&p1, b1.build()).unwrap();

        let lbc_path = dir.join(format!("{stem}.lbc"));
        let result = convert_gguf_to_lbc(&p1, &lbc_path, &ConvertOptions::default());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("expected 2 shard") || err_msg.contains("only found 1"),
            "expected missing-shard error to mention shard counts, got: {err_msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_llama_arch_rejected() {
        // Regression test for the Qwen3.5-only converter: GGUFs whose
        // `general.architecture` reports `llama` or any other non-Qwen3.5
        // family member must be rejected at extract_hyperparams.
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        let gguf_data = builder.build();

        let dir = temp_dir();
        let lbc_path = dir.join("legacy_llama.lbc");

        let result = convert_gguf_bytes_to_lbc(&gguf_data, &lbc_path, &ConvertOptions::default());
        assert!(matches!(
            result,
            Err(ConvertError::UnsupportedArchitecture(ref a)) if a == "llama"
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for the Qwen3.5-Next BF16 GGUF pattern produced by
    /// `convert_hf_to_gguf.py --outtype bf16`: the file declares
    /// `block_count = 33` for a 32-layer model because it counts the trailing
    /// MTP (Next-N) head at blk.32 as a regular block. The MTP head ships
    /// `nextn.eh_proj.weight`, `nextn.enorm.weight`, `nextn.hnorm.weight`, and
    /// `nextn.shared_head_norm.weight` alongside attn_q/k/v/output -- Lumen
    /// must NOT try to convert it as a backbone layer (it would crash on the
    /// wrong attention layout otherwise).
    ///
    /// We assert that `extract_hyperparams` reports the OBSERVED
    /// `num_layers = 32`, not the metadata value 33.
    #[test]
    fn mtp_head_excluded_from_layer_count() {
        use crate::hyperparams::extract_hyperparams;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "qwen35");
        // Lie: 32 real layers but metadata says 33 (counting the MTP head).
        builder.add_u32("qwen35.block_count", 33);
        builder.add_u32("qwen35.attention.head_count", 2);
        builder.add_u32("qwen35.attention.head_count_kv", 2);
        builder.add_u32("qwen35.embedding_length", 8);
        builder.add_u32("qwen35.feed_forward_length", 16);

        // Global tensors -- minimal, just so the metadata reader doesn't trip.
        builder.add_f32_tensor(EMBEDDING_NAME, &[8, 4], &vec![0.0f32; 32]);

        // 32 real layers: each carries `blk.N.attn_q.weight`.
        for layer in 0..32 {
            let name = layer_tensor_name(layer as usize, ATTN_Q);
            builder.add_f32_tensor(&name, &[8, 8], &vec![0.0f32; 64]);
        }

        // 1 MTP head at blk.32: has `attn_q.weight` AND `nextn.eh_proj.weight`.
        // Without the `nextn.*` exclusion this would be counted as a real
        // layer and num_layers would be 33.
        builder.add_f32_tensor("blk.32.attn_q.weight", &[8, 8], &vec![0.0f32; 64]);
        builder.add_f32_tensor("blk.32.nextn.eh_proj.weight", &[8, 8], &vec![0.0f32; 64]);

        let gguf_data = builder.build();
        let mut cur = std::io::Cursor::new(&gguf_data);
        let gguf = crate::gguf::GgufFile::parse(&mut cur).unwrap();

        // Sanity: metadata really says 33.
        assert_eq!(gguf.get_u32("qwen35.block_count"), Some(33));

        let (hp, _arch) = extract_hyperparams(&gguf)
            .expect("extract_hyperparams should succeed");
        assert_eq!(
            hp.num_layers, 32,
            "num_layers must reflect the 32 real layers, ignoring the MTP head at blk.32 \
             (metadata block_count=33). Got {}", hp.num_layers,
        );
    }

    /// `block_count` overshoot WITHOUT an MTP head (pure off-by-one bug in a
    /// GGUF producer): if blk.32 simply doesn't have an attention weight,
    /// `real_main_layer_count` must still stop at the first incomplete layer.
    /// This covers e.g. a producer that wrote `block_count=N+1` but only
    /// emitted N layers of attention tensors.
    #[test]
    fn block_count_off_by_one_truncates_at_missing_attn() {
        use crate::hyperparams::extract_hyperparams;

        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "qwen35");
        builder.add_u32("qwen35.block_count", 3); // overshoot by 1
        builder.add_u32("qwen35.attention.head_count", 2);
        builder.add_u32("qwen35.attention.head_count_kv", 2);
        builder.add_u32("qwen35.embedding_length", 8);
        builder.add_u32("qwen35.feed_forward_length", 16);
        builder.add_f32_tensor(EMBEDDING_NAME, &[8, 4], &vec![0.0f32; 32]);

        // Real layers 0 and 1 each have attn_q.weight.
        for layer in 0..2 {
            let name = layer_tensor_name(layer as usize, ATTN_Q);
            builder.add_f32_tensor(&name, &[8, 8], &vec![0.0f32; 64]);
        }
        // blk.2 exists in the tensor list but lacks attn_q.weight -- this
        // happens when a producer stopped emitting layers part-way through.
        // We emulate it with just a norm tensor so the index exists but
        // doesn't carry a required attention weight.
        builder.add_f32_tensor("blk.2.attn_norm.weight", &[8], &vec![1.0f32; 8]);

        let gguf_data = builder.build();
        let mut cur = std::io::Cursor::new(&gguf_data);
        let gguf = crate::gguf::GgufFile::parse(&mut cur).unwrap();

        let (hp, _arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(hp.num_layers, 2,
                   "expected truncation to 2 real layers, got {}", hp.num_layers);
    }

    /// `block_count` matches the actual count: no warning, no surprise.
    #[test]
    fn block_count_matches_reality_no_op() {
        use crate::hyperparams::extract_hyperparams;
        let gguf_data = build_minimal_qwen35_metadata_gguf(4, 2, 2, 8, 16, 32);
        // Add 4 real layers (each with attn_q.weight) so the observed
        // count matches metadata block_count=4.
        let mut cur = std::io::Cursor::new(&gguf_data);
        // Validate the minimal metadata parses, then rebuild below with real layers.
        crate::gguf::GgufFile::parse(&mut cur).unwrap();
        // Re-construct with attention tensors for each layer.
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "qwen35");
        builder.add_u32("qwen35.block_count", 4);
        builder.add_u32("qwen35.attention.head_count", 2);
        builder.add_u32("qwen35.attention.head_count_kv", 2);
        builder.add_u32("qwen35.embedding_length", 8);
        builder.add_u32("qwen35.feed_forward_length", 16);
        builder.add_f32_tensor(EMBEDDING_NAME, &[32, 8], &vec![0.0f32; 256]);
        for layer in 0..4 {
            let name = layer_tensor_name(layer as usize, ATTN_Q);
            builder.add_f32_tensor(&name, &[8, 8], &vec![0.0f32; 64]);
        }
        let bytes = builder.build();
        let mut cur2 = std::io::Cursor::new(&bytes);
        let gguf = crate::gguf::GgufFile::parse(&mut cur2).unwrap();

        let (hp, _arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(hp.num_layers, 4);
    }

    // NOTE: The synthetic weight-tying conversion tests (F32 + Q8_0
    // variants) were removed when the generic Dense converter was retired
    // in favour of the Qwen3.5-specific converter. Weight tying is still
    // exercised end-to-end by integration tests against the real
    // Qwen3.5-9B GGUF in `modal/test_qwen35.py`.

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

    /// Non-degenerate Q2_K block: exercises the GGML qs traversal ORDER, which
    /// the uniform-block tests above cannot (any traversal agrees when every
    /// quant + scale is identical). Reference values come from GGML's
    /// `dequantize_row_q2_K` (two 128-value groups; four shift passes 0,2,4,6
    /// over the same 32 qs bytes per group; `q[l]` then `q[l+16]`). Regression
    /// guard for the byte-traversal bug that corrupted every Q2_K tensor in the
    /// Qwen3.6-27B LBC and produced token salad.
    #[test]
    fn dequantize_q2_k_non_degenerate_traversal() {
        // scales[j]: low nibble = scale, high nibble = min, both varied.
        let mut block = Vec::with_capacity(84);
        for j in 0..16u8 {
            block.push((j & 0x0F) | (((15 - j) & 0x0F) << 4));
        }
        // qs: 64 varied bytes.
        for i in 0..64u8 {
            block.push(i.wrapping_mul(37).wrapping_add(11));
        }
        block.extend_from_slice(&f32_to_f16_bits(0.05).to_le_bytes()); // d
        block.extend_from_slice(&f32_to_f16_bits(0.02).to_le_bytes()); // dmin
        assert_eq!(block.len(), 84);

        let values: Vec<f32> = dequantize_q2_k(&block, 256)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 256);

        // Independent re-implementation of the canonical GGML traversal.
        let d = f16(0.05);
        let dmin = f16(0.02);
        let scales: Vec<u8> = (0..16u8).map(|j| (j & 0x0F) | (((15 - j) & 0x0F) << 4)).collect();
        let qs: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
        let mut expected = [0f32; 256];
        let mut yi = 0usize;
        let mut q_off = 0usize;
        let mut is = 0usize;
        for _g in 0..2 {
            let mut shift = 0u8;
            for _j in 0..4 {
                let sc = scales[is]; is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                for l in 0..16 { expected[yi] = dl * (((qs[q_off + l] >> shift) & 3) as f32) - ml; yi += 1; }
                let sc = scales[is]; is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                for l in 0..16 { expected[yi] = dl * (((qs[q_off + l + 16] >> shift) & 3) as f32) - ml; yi += 1; }
                shift += 2;
            }
            q_off += 32;
        }
        for i in 0..256 {
            assert!(
                (values[i] - expected[i]).abs() < 1e-5,
                "Q2_K traversal mismatch at {i}: got {}, expected {}",
                values[i], expected[i]
            );
        }
        // Sanity: this block is genuinely non-uniform, so a naive linear scan
        // would have diverged (guards against the test degenerating).
        assert!(
            values.iter().cloned().fold(f32::MIN, f32::max)
                - values.iter().cloned().fold(f32::MAX, f32::min)
                > 1.0,
            "test block must be non-degenerate"
        );
    }

    // f16 decode helper mirroring dequant.rs::f16_to_f32, for the test's
    // independent reference computation.
    fn f16(v: f32) -> f32 {
        let h = f32_to_f16_bits(v);
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 { sign << 31 } else {
                let mut e = exp as i32; let mut m = mant;
                while (m & 0x400) == 0 { m <<= 1; e -= 1; }
                e += 1; m &= 0x3ff;
                (sign << 31) | (((e + 112) as u32) << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | (0xff << 23) | (mant << 13)
        } else {
            (sign << 31) | ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    /// Non-degenerate Q3_K block: exercises the GGML qs/hmask traversal order.
    /// Regression guard for the linear-scan bug that corrupted the Qwen3.6-27B
    /// ffn_down (Q3_K) tensors. Reference = GGML `dequantize_row_q3_K`.
    #[test]
    fn dequantize_q3_k_non_degenerate_traversal() {
        let mut block = Vec::with_capacity(110);
        for i in 0..32u8 { block.push(i.wrapping_mul(53).wrapping_add(7)); }   // hmask
        for i in 0..64u8 { block.push(i.wrapping_mul(29).wrapping_add(3)); }   // qs
        for i in 0..12u8 { block.push(i.wrapping_mul(17).wrapping_add(5)); }   // scales
        block.extend_from_slice(&f32_to_f16_bits(0.1).to_le_bytes());          // d
        assert_eq!(block.len(), 110);

        let values: Vec<f32> = dequantize_q3_k(&block, 256)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 256);

        // Independent GGML reference.
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scale_bytes = &block[96..108];
        let d = f16(0.1);
        // GGML aux uint32 shuffle for the 16 6-bit Q3_K scales (matches the
        // production decode in dequant.rs and llama.cpp byte-for-byte).
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let a0 = u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
        let a1 = u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]);
        let tmp = u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]);
        let out0 = (a0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        let out1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let out2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        let out3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        let mut sc = [0u8; 16];
        sc[0..4].copy_from_slice(&out0.to_le_bytes());
        sc[4..8].copy_from_slice(&out1.to_le_bytes());
        sc[8..12].copy_from_slice(&out2.to_le_bytes());
        sc[12..16].copy_from_slice(&out3.to_le_bytes());
        let mut expected = [0f32; 256];
        let mut yi = 0usize;
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut m = 1u8;
        for _g in 0..2 {
            let mut shift = 0u8;
            for _j in 0..4 {
                let dl = d * (sc[is] as i8 as f32 - 32.0); is += 1;
                for l in 0..16 {
                    let qb = ((qs[q_off + l] >> shift) & 3) as i32;
                    let sub = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    expected[yi] = dl * (qb - sub) as f32; yi += 1;
                }
                let dl = d * (sc[is] as i8 as f32 - 32.0); is += 1;
                for l in 0..16 {
                    let qb = ((qs[q_off + l + 16] >> shift) & 3) as i32;
                    let sub = if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                    expected[yi] = dl * (qb - sub) as f32; yi += 1;
                }
                shift += 2; m <<= 1;
            }
            q_off += 32;
        }
        for i in 0..256 {
            assert!(
                (values[i] - expected[i]).abs() < 1e-4,
                "Q3_K traversal mismatch at {i}: got {}, expected {}",
                values[i], expected[i]
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

    // round_trip_q8_0_dequantize was removed when the generic Dense
    // converter was retired: it built a synthetic llama-arch Q8_0 GGUF
    // and round-tripped it through the Dense converter. Real Q8_0
    // dequantize round-trip is exercised by `dequantize_q8_0_known_block`,
    // `dequantize_q8_0_two_blocks`, and by integration tests against the
    // real Qwen3.5-9B Q8_0 GGUF.

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

    // round_trip_dequantize_f32_noop and gguf_v2_converter_roundtrip were
    // removed when the generic Dense converter was retired: both
    // depended on the synthetic-llama Dense converter pipeline. GGUF v2
    // and v3 are still parsed by the converter (see `gguf.rs` tests) and
    // round-tripped end-to-end by integration tests against real
    // Qwen3.5-9B GGUFs.

    #[test]
    fn convert_stats_display() {
        let stats = ConvertStats {
            input_size: 10 * 1024 * 1024,
            output_size: 8 * 1024 * 1024,
            num_layers: 32,
            architecture: "qwen35".into(),
            tensor_count: 291,
            quant_scheme: QuantScheme::Q4_K,
        };
        let s = format!("{stats}");
        assert!(s.contains("291 tensors"));
        assert!(s.contains("32 layers"));
        assert!(s.contains("qwen35"));
    }

    // The MoE pipeline tests (`moe_round_trip_f32`, `moe_f16_router_*`)
    // were removed when the generic MoE converter was retired: they
    // built synthetic llama-arch MoE GGUFs and exercised the generic
    // `MoeConverter` which has been deleted. The Qwen3.5-MoE converter
    // (`arch::qwen35_moe`) is exercised by integration tests against
    // real Qwen3.5-MoE GGUFs.

    /// Hyperparameter extraction for the Qwen3.5 MoE variant: verifies
    /// that `num_experts` and `num_active_experts` are read out of GGUF
    /// metadata correctly for the MoE family.
    #[test]
    fn moe_hyperparams_extraction_qwen35moe() {
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "qwen35moe");
        builder.add_u32("qwen35moe.block_count", 2);
        builder.add_u32("qwen35moe.attention.head_count", 2);
        builder.add_u32("qwen35moe.attention.head_count_kv", 2);
        builder.add_u32("qwen35moe.attention.key_length", 4);
        builder.add_u32("qwen35moe.embedding_length", 8);
        builder.add_u32("qwen35moe.feed_forward_length", 16);
        builder.add_u32("qwen35moe.expert_count", 8);
        builder.add_u32("qwen35moe.expert_used_count", 2);
        builder.add_f32_tensor(EMBEDDING_NAME, &[32, 8], &vec![0.0f32; 256]);
        for layer in 0..2 {
            let name = layer_tensor_name(layer as usize, ATTN_Q);
            builder.add_f32_tensor(&name, &[8, 8], &vec![0.0f32; 64]);
        }
        let gguf_data = builder.build();
        let mut cursor = std::io::Cursor::new(&gguf_data);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        let (hp, arch) = extract_hyperparams(&gguf).unwrap();
        assert_eq!(arch, "qwen35moe");
        assert_eq!(hp.num_experts, Some(8));
        assert_eq!(hp.num_active_experts, Some(2));
        assert!(hp.is_moe());
    }

    /// Tensor-naming helpers do not depend on architecture or converter
    /// implementation; they are pure string-builder utilities.
    #[test]
    fn moe_tensor_naming() {
        assert_eq!(layer_tensor_name(0, FFN_GATE_INP), "blk.0.ffn_gate_inp.weight");
        assert_eq!(expert_tensor_name(0, "gate", 0), "blk.0.ffn_gate.0.weight");
        assert_eq!(expert_tensor_name(0, "up", 3), "blk.0.ffn_up.3.weight");
        assert_eq!(expert_tensor_name(1, "down", 7), "blk.1.ffn_down.7.weight");
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

    // The Q6_K-output-projection requantisation tests (and their
    // `build_q6_k_zero_blocks` helper) were removed when the generic
    // Dense converter was retired: they exercised the synthetic-llama
    // Dense converter pipeline. The K-quant -> Q8_0/Q4_0 requantisation
    // logic itself is unchanged (see `convert.rs` lines that match
    // `requant_target == Some`). It is exercised end-to-end by
    // integration tests against real Qwen3.5-9B GGUFs whose
    // output.weight is stored as Q6_K.
}
