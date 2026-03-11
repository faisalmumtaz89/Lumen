//! Synchronous weight provider backed by `SyncFileBackend`.
//!
//! No prefetching — `prefetch_layer` is a no-op. `get_layer_blocking` reads
//! from disk and caches in memory. Simple and correct — the baseline for testing.

use crate::weight::cache::{
    CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider,
};
use crate::error::RuntimeError;
use crate::storage::{IoSnapshot, StorageBackend};
use crate::storage::sync::SyncFileBackend;
use lumen_format::reader::LbcFile;
use lumen_format::quantization::QuantScheme;
use std::path::Path;
use std::sync::Mutex;

/// Synchronous, blocking weight provider for testing and correctness validation.
pub struct SyncWeightProvider {
    lbc: LbcFile,
    backend: SyncFileBackend,
    /// Cached layer views, protected by Mutex for sound Send+Sync.
    cache: Mutex<Vec<Option<LayerView>>>,
    /// Global tensors read at open time.
    pub embedding: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub output_proj: Vec<f32>,
    /// Raw output_proj bytes (Q8_0 or F32). Used by Metal backend to skip
    /// CPU-side dequantization and upload native quantized data to GPU.
    pub output_proj_raw: Vec<u8>,
    /// Quantization scheme of the output_proj tensor.
    pub output_proj_quant: QuantScheme,
    /// Raw embedding bytes (Q8_0/Q4_0 or empty for F32). Used by Metal backend
    /// to upload native quantized embedding data and use dequant kernels on GPU.
    pub embedding_raw: Vec<u8>,
    /// Quantization scheme of the embedding tensor.
    pub embedding_quant: QuantScheme,
    /// Whether output_proj shares embedding storage (weight tying).
    pub weight_tying: bool,
    stats: Mutex<CacheStats>,
}

impl SyncWeightProvider {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let lbc = LbcFile::open(path)?;
        let mut backend = SyncFileBackend::new();
        backend.open(path)?;

        let num_layers = lbc.header.num_layers as usize;
        let vocab_size = lbc.header.hyperparams.vocab_size as usize;
        let hidden_dim = lbc.header.hyperparams.hidden_dim as usize;

        // Read global tensors
        let embedding_bytes = backend.read_range(lbc.header.embedding.offset, lbc.header.embedding.length)?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim);
        let final_norm = read_f32_tensor(&backend, lbc.header.final_norm.offset, lbc.header.final_norm.length)?;
        let output_proj_bytes = backend.read_range(lbc.header.output_proj.offset, lbc.header.output_proj.length)?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim);

        let weight_tying = lbc.header.weight_tying;
        Ok(Self {
            lbc,
            backend,
            cache: Mutex::new(vec![None; num_layers]),
            embedding,
            final_norm,
            output_proj,
            output_proj_raw,
            output_proj_quant,
            embedding_raw,
            embedding_quant,
            weight_tying,
            stats: Mutex::new(CacheStats::default()),
        })
    }

    /// Open from an in-memory LBC buffer (for tests).
    pub fn from_lbc_bytes(data: &[u8], path: &Path) -> Result<Self, RuntimeError> {
        let lbc = LbcFile::from_bytes(data, path.to_path_buf())?;
        let mut backend = SyncFileBackend::new();
        backend.open(path)?;

        let num_layers = lbc.header.num_layers as usize;
        let vocab_size = lbc.header.hyperparams.vocab_size as usize;
        let hidden_dim = lbc.header.hyperparams.hidden_dim as usize;

        let embedding_bytes = backend.read_range(lbc.header.embedding.offset, lbc.header.embedding.length)?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim);
        let final_norm = read_f32_tensor(&backend, lbc.header.final_norm.offset, lbc.header.final_norm.length)?;
        let output_proj_bytes = backend.read_range(lbc.header.output_proj.offset, lbc.header.output_proj.length)?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim);

        let weight_tying = lbc.header.weight_tying;
        Ok(Self {
            lbc,
            backend,
            cache: Mutex::new(vec![None; num_layers]),
            embedding,
            final_norm,
            output_proj,
            output_proj_raw,
            output_proj_quant,
            embedding_raw,
            embedding_quant,
            weight_tying,
            stats: Mutex::new(CacheStats::default()),
        })
    }

    pub fn lbc(&self) -> &LbcFile {
        &self.lbc
    }
}

fn read_f32_tensor(
    backend: &SyncFileBackend,
    offset: u64,
    length: u64,
) -> Result<Vec<f32>, RuntimeError> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let bytes = backend.read_range(offset, length)?;
    Ok(bytes_to_f32(&bytes))
}

pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 4 == 0,
        "bytes_to_f32: input length {} is not a multiple of 4 (trailing {} bytes would be silently dropped)",
        bytes.len(),
        bytes.len() % 4,
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Convert f16 bits to f32 (software implementation, no external dependency).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let f = frac as f32 / 1024.0;
        let v = f * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    let exp_f32 = ((exp as u32 - 15 + 127) << 23) | ((frac as u32) << 13);
    let v = f32::from_bits(exp_f32);
    if sign == 1 { -v } else { v }
}

/// Dequantize Q8_0 bytes to Vec<f32>.
/// Q8_0 block layout: [2 bytes f16 scale] [32 bytes int8 values], total 34 bytes per 32 elements.
pub fn dequantize_q8_0_to_f32(src: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    let block_size = 34;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n_elements && offset + block_size <= src.len() {
        let scale_bits = u16::from_le_bytes([src[offset], src[offset + 1]]);
        let scale = f16_bits_to_f32(scale_bits);
        for i in 0..32 {
            if written >= n_elements {
                break;
            }
            let q = src[offset + 2 + i] as i8;
            out.push(scale * q as f32);
            written += 1;
        }
        offset += block_size;
    }
    out
}


/// Dequantize Q4_0 bytes to Vec<f32>.
/// Q4_0 block layout: [2 bytes f16 scale] [16 bytes packed nibbles], total 18 bytes per 32 elements.
/// GGML de-interleaved order: indices 0-15 use lo nibbles, indices 16-31 use hi nibbles.
pub fn dequantize_q4_0_to_f32(src: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    let block_size = 18;
    let mut written = 0usize;
    let mut offset = 0usize;
    while written < n_elements && offset + block_size <= src.len() {
        let scale_bits = u16::from_le_bytes([src[offset], src[offset + 1]]);
        let scale = f16_bits_to_f32(scale_bits);
        // De-interleaved: first 16 elements from lo nibbles (indices 0-15)
        for i in 0..16 {
            if written >= n_elements {
                break;
            }
            let byte = src[offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            out.push(scale * lo as f32);
            written += 1;
        }
        // Then 16 elements from hi nibbles (indices 16-31)
        for i in 0..16 {
            if written >= n_elements {
                break;
            }
            let byte = src[offset + 2 + i];
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(scale * hi as f32);
            written += 1;
        }
        offset += block_size;
    }
    out
}
/// Dequantize F16 (IEEE 754 half-precision) bytes to Vec<f32>.
/// Each element is 2 bytes (little-endian u16 f16 bits).
fn dequantize_f16_to_f32(src: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let offset = i * 2;
        if offset + 2 > src.len() {
            break;
        }
        let bits = u16::from_le_bytes([src[offset], src[offset + 1]]);
        out.push(f16_bits_to_f32(bits));
    }
    out
}

/// Detect embedding quantization from byte length and model dimensions.
/// Returns (f32_data, raw_bytes, quant_scheme).
/// Same heuristic as read_output_proj_global: compare byte length against
/// expected sizes for F32, Q8_0, Q4_0, and F16.
pub fn read_embedding_global(
    raw_bytes: Vec<u8>,
    vocab_size: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<u8>, QuantScheme) {
    let n_elements = vocab_size * hidden_dim;
    let expected_f32_bytes = n_elements * 4;
    let expected_q8_bytes = (n_elements / 32) * 34;
    let expected_q4_bytes = (n_elements / 32) * 18;
    let expected_f16_bytes = n_elements * 2;

    if raw_bytes.len() == expected_f32_bytes {
        let f32_data = bytes_to_f32(&raw_bytes);
        (f32_data, Vec::new(), QuantScheme::F32)
    } else if raw_bytes.len() == expected_f16_bytes {
        let f32_data = dequantize_f16_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::F16)
    } else if raw_bytes.len() == expected_q8_bytes {
        let f32_data = dequantize_q8_0_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::Q8_0)
    } else if raw_bytes.len() == expected_q4_bytes {
        let f32_data = dequantize_q4_0_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::Q4_0)
    } else {
        // Unknown format -- try F32 interpretation (backward compat)
        let f32_data = bytes_to_f32(&raw_bytes);
        (f32_data, Vec::new(), QuantScheme::F32)
    }
}

/// Detect output_proj quantization from byte length and model dimensions.
/// Returns (f32_data, raw_bytes, quant_scheme).
pub fn read_output_proj_global(
    raw_bytes: Vec<u8>,
    vocab_size: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<u8>, QuantScheme) {
    let n_elements = vocab_size * hidden_dim;
    let expected_f32_bytes = n_elements * 4;
    let expected_q8_bytes = (n_elements / 32) * 34;
    let expected_q4_bytes = (n_elements / 32) * 18;
    let expected_f16_bytes = n_elements * 2;

    if raw_bytes.len() == expected_f32_bytes {
        let f32_data = bytes_to_f32(&raw_bytes);
        (f32_data, raw_bytes, QuantScheme::F32)
    } else if raw_bytes.len() == expected_f16_bytes {
        let f32_data = dequantize_f16_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::F16)
    } else if raw_bytes.len() == expected_q8_bytes {
        let f32_data = dequantize_q8_0_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::Q8_0)
    } else if raw_bytes.len() == expected_q4_bytes {
        let f32_data = dequantize_q4_0_to_f32(&raw_bytes, n_elements);
        (f32_data, raw_bytes, QuantScheme::Q4_0)
    } else {
        // Unknown format -- try F32 interpretation (backward compat)
        let f32_data = bytes_to_f32(&raw_bytes);
        (f32_data, Vec::new(), QuantScheme::F32)
    }
}

impl WeightProvider for SyncWeightProvider {
    fn prefetch_layer(
        &self,
        layer: usize,
        priority: PrefetchPriority,
    ) -> Result<PrefetchHandle, RuntimeError> {
        // No-op for sync provider — return completed handle
        let mut handle = PrefetchHandle::new(layer, priority);
        handle.mark_complete();
        Ok(handle)
    }

    fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        if layer >= cache.len() {
            return Err(RuntimeError::LayerUnavailable {
                layer,
                reason: format!("layer index out of range (num_layers={})", cache.len()),
            });
        }

        if let Some(ref view) = cache[layer] {
            stats.hits += 1;
            return Ok(view.clone());
        }

        stats.misses += 1;

        let idx = &self.lbc.layer_indices[layer];
        let data = self.backend.read_range(idx.layer_offset_bytes, idx.layer_length_bytes)?;
        let view = LayerView::from_owned(layer, data, idx.subtensors.clone());

        cache[layer] = Some(view.clone());
        stats.layers_cached += 1;
        stats.bytes_cached += idx.layer_length_bytes;

        Ok(view)
    }

    fn try_get_layer(&self, layer: usize) -> Option<LayerView> {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if layer >= cache.len() {
            return None;
        }
        cache[layer].clone()
    }

    fn release_layer_hint(&self, layer: usize) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        if layer >= cache.len() {
            return;
        }
        if let Some(view) = cache[layer].take() {
            stats.bytes_cached = stats.bytes_cached.saturating_sub(view.byte_len() as u64);
            stats.layers_cached = stats.layers_cached.saturating_sub(1);
            stats.evictions += 1;
        }
    }

    fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn num_layers(&self) -> usize {
        self.lbc.header.num_layers as usize
    }

    fn io_snapshot(&self) -> Option<IoSnapshot> {
        self.backend.io_tracker().map(|t| t.snapshot())
    }
}
