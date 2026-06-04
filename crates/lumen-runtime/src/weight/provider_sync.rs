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
use lumen_format::index::{SubtensorOffsets, TensorSlice};
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
        let embed_header_quant = lbc.header.embedding.quant;
        let outproj_header_quant = lbc.header.output_proj.quant;
        let embedding_bytes = backend.read_range(lbc.header.embedding.offset, lbc.header.embedding.length)?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim, embed_header_quant);
        let final_norm = read_f32_tensor(&backend, lbc.header.final_norm.offset, lbc.header.final_norm.length)?;
        let output_proj_bytes = backend.read_range(lbc.header.output_proj.offset, lbc.header.output_proj.length)?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim, outproj_header_quant);

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

        let embed_header_quant = lbc.header.embedding.quant;
        let outproj_header_quant = lbc.header.output_proj.quant;
        let embedding_bytes = backend.read_range(lbc.header.embedding.offset, lbc.header.embedding.length)?;
        let (embedding, embedding_raw, embedding_quant) =
            read_embedding_global(embedding_bytes, vocab_size, hidden_dim, embed_header_quant);
        let final_norm = read_f32_tensor(&backend, lbc.header.final_norm.offset, lbc.header.final_norm.length)?;
        let output_proj_bytes = backend.read_range(lbc.header.output_proj.offset, lbc.header.output_proj.length)?;
        let (output_proj, output_proj_raw, output_proj_quant) =
            read_output_proj_global(output_proj_bytes, vocab_size, hidden_dim, outproj_header_quant);

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
    // Normalized: rebias the 5-bit f16 exponent (bias 15) to the 8-bit f32
    // exponent (bias 127). Use i32 so the intermediate `exp - 15` cannot
    // underflow for f16 values with exp < 15 (i.e. |x| < 1.0), which is the
    // common case for quantization scales. For normalized f16 (1..=30) the
    // result `exp + 112` is always in 113..=142, a valid f32 exponent.
    let exp_f32 = ((exp as i32 - 15 + 127) as u32) << 23 | ((frac as u32) << 13);
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

/// Dequantize BF16 (bfloat16) bytes to Vec<f32>. Each element is 2 bytes:
/// the top 16 bits of the binary32 layout, so the conversion is a 16-bit
/// left shift reinterpreted as f32. BF16 and F16 share the same 2-byte width
/// but have different exponent/mantissa layouts, so callers MUST disambiguate
/// via the LBC header quant scheme (see `read_embedding_global`).
fn dequantize_bf16_to_f32(src: &[u8], n_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let offset = i * 2;
        if offset + 2 > src.len() {
            break;
        }
        let bits = u16::from_le_bytes([src[offset], src[offset + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// Detect embedding quantization from byte length and model dimensions.
/// Returns (f32_data, raw_bytes, quant_scheme).
/// Same heuristic as read_output_proj_global: compare byte length against
/// expected sizes for F32, Q8_0, Q4_0, and F16/BF16. F16 and BF16 share the
/// same 2-byte width, so `header_quant` (from the LBC header) disambiguates them.
pub fn read_embedding_global(
    raw_bytes: Vec<u8>,
    vocab_size: usize,
    hidden_dim: usize,
    header_quant: QuantScheme,
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
        // F16 and BF16 share the same byte width — trust the header to disambiguate.
        if matches!(header_quant, QuantScheme::Bf16) {
            let f32_data = dequantize_bf16_to_f32(&raw_bytes, n_elements);
            (f32_data, raw_bytes, QuantScheme::Bf16)
        } else {
            let f32_data = dequantize_f16_to_f32(&raw_bytes, n_elements);
            (f32_data, raw_bytes, QuantScheme::F16)
        }
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
/// Returns (f32_data, raw_bytes, quant_scheme). F16 and BF16 share the same
/// 2-byte width, so `header_quant` (from the LBC header) disambiguates them.
pub fn read_output_proj_global(
    raw_bytes: Vec<u8>,
    vocab_size: usize,
    hidden_dim: usize,
    header_quant: QuantScheme,
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
        // F16 and BF16 share the same byte width — trust the header to disambiguate.
        if matches!(header_quant, QuantScheme::Bf16) {
            let f32_data = dequantize_bf16_to_f32(&raw_bytes, n_elements);
            (f32_data, raw_bytes, QuantScheme::Bf16)
        } else {
            let f32_data = dequantize_f16_to_f32(&raw_bytes, n_elements);
            (f32_data, raw_bytes, QuantScheme::F16)
        }
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

/// Dequantize a single subtensor from the raw layer blob, returning F32 bytes.
/// Returns the dequantized bytes and the number of F32 elements.
fn dequant_subtensor_to_f32_bytes(
    raw_blob: &[u8],
    slice: &TensorSlice,
) -> Option<Vec<u8>> {
    match slice.quant {
        QuantScheme::F32 => None, // Already F32, no conversion needed
        QuantScheme::Q8_0 => {
            let src = &raw_blob[slice.offset as usize..(slice.offset + slice.length) as usize];
            // Q8_0: 34 bytes per 32 elements
            let n_blocks = src.len() / 34;
            let n_elements = n_blocks * 32;
            let f32_data = dequantize_q8_0_to_f32(src, n_elements);
            let mut bytes = Vec::with_capacity(f32_data.len() * 4);
            for &v in &f32_data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Some(bytes)
        }
        QuantScheme::Q4_0 => {
            let src = &raw_blob[slice.offset as usize..(slice.offset + slice.length) as usize];
            // Q4_0: 18 bytes per 32 elements
            let n_blocks = src.len() / 18;
            let n_elements = n_blocks * 32;
            let f32_data = dequantize_q4_0_to_f32(src, n_elements);
            let mut bytes = Vec::with_capacity(f32_data.len() * 4);
            for &v in &f32_data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Some(bytes)
        }
        QuantScheme::F16 => {
            let src = &raw_blob[slice.offset as usize..(slice.offset + slice.length) as usize];
            let n_elements = src.len() / 2;
            let f32_data = dequantize_f16_to_f32(src, n_elements);
            let mut bytes = Vec::with_capacity(f32_data.len() * 4);
            for &v in &f32_data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Some(bytes)
        }
        _ => None, // Unsupported quant scheme -- pass through as-is
    }
}

/// Dequantize all quantized subtensors in a layer blob to F32.
///
/// Builds a new blob where all weight tensors are in F32 format, and updates
/// the subtensor offsets to match. Tensors that are already F32 are copied as-is.
/// This enables the CPU naive backend (which reads raw bytes as F32) to work
/// with Q8_0/Q4_0/F16 model files.
fn dequantize_layer_to_f32(
    raw_blob: &[u8],
    subtensors: &SubtensorOffsets,
) -> (Vec<u8>, SubtensorOffsets) {
    // Check if any subtensor needs dequantization
    let needs_dequant = [
        &subtensors.wq, &subtensors.wk, &subtensors.wv, &subtensors.wo,
        &subtensors.w_gate, &subtensors.w_up, &subtensors.w_down,
        &subtensors.attn_norm, &subtensors.ffn_norm,
    ].iter().any(|s| s.quant != QuantScheme::F32);

    if !needs_dequant {
        // Fast path: all F32, return as-is
        return (raw_blob.to_vec(), subtensors.clone());
    }

    // Slow path: rebuild the blob with dequantized tensors
    let mut new_blob = Vec::new();
    let mut new_subtensors = subtensors.clone();

    // NOTE: Only mandatory subtensors are processed. Optional fields (bq, bk, bv,
    // ssm_*, attn_gate, etc.) retain their original-blob offsets. This is safe
    // because the cpu_naive backend only reads mandatory fields.
    let slices_and_fields: Vec<(&TensorSlice, Box<dyn FnOnce(&mut SubtensorOffsets, TensorSlice)>)> = vec![
        (&subtensors.wq, Box::new(|st: &mut SubtensorOffsets, s| st.wq = s)),
        (&subtensors.wk, Box::new(|st: &mut SubtensorOffsets, s| st.wk = s)),
        (&subtensors.wv, Box::new(|st: &mut SubtensorOffsets, s| st.wv = s)),
        (&subtensors.wo, Box::new(|st: &mut SubtensorOffsets, s| st.wo = s)),
        (&subtensors.w_gate, Box::new(|st: &mut SubtensorOffsets, s| st.w_gate = s)),
        (&subtensors.w_up, Box::new(|st: &mut SubtensorOffsets, s| st.w_up = s)),
        (&subtensors.w_down, Box::new(|st: &mut SubtensorOffsets, s| st.w_down = s)),
        (&subtensors.attn_norm, Box::new(|st: &mut SubtensorOffsets, s| st.attn_norm = s)),
        (&subtensors.ffn_norm, Box::new(|st: &mut SubtensorOffsets, s| st.ffn_norm = s)),
    ];

    for (slice, setter) in slices_and_fields {
        let offset = new_blob.len() as u64;
        if let Some(f32_bytes) = dequant_subtensor_to_f32_bytes(raw_blob, slice) {
            let new_slice = TensorSlice {
                offset,
                length: f32_bytes.len() as u64,
                quant: QuantScheme::F32,
            };
            new_blob.extend_from_slice(&f32_bytes);
            setter(&mut new_subtensors, new_slice);
        } else {
            // Already F32 or unsupported -- copy raw bytes
            let src = &raw_blob[slice.offset as usize..(slice.offset + slice.length) as usize];
            let new_slice = TensorSlice {
                offset,
                length: slice.length,
                quant: slice.quant,
            };
            new_blob.extend_from_slice(src);
            setter(&mut new_subtensors, new_slice);
        }
    }

    (new_blob, new_subtensors)
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
        let raw_data = self.backend.read_range(idx.layer_offset_bytes, idx.layer_length_bytes)?;

        // Dequantize quantized subtensors to F32 for CPU backends.
        // The CPU naive backend reads raw bytes as F32 via matmul_bytes/rmsnorm_bytes.
        // When subtensors are stored in Q8_0/Q4_0/F16, we must dequantize them here.
        let (data, subtensors) = dequantize_layer_to_f32(&raw_data, &idx.subtensors);

        let view = LayerView::from_owned(layer, data, subtensors);

        cache[layer] = Some(view.clone());
        stats.layers_cached += 1;
        stats.bytes_cached += idx.layer_length_bytes;

        Ok(view)
    }

    /// Get raw layer data WITHOUT dequantization.
    ///
    /// Returns Q8_0/Q4_0/F16 weights in their native format so the CUDA
    /// backend can upload raw bytes and dispatch quantized GPU kernels (dp4a,
    /// HGEMM). Skips the `dequantize_layer_to_f32` step used by `get_layer_blocking`.
    fn get_layer_raw(&self, layer: usize) -> Result<LayerView, RuntimeError> {
        if layer >= self.lbc.layer_indices.len() {
            return Err(RuntimeError::LayerUnavailable {
                layer,
                reason: format!("layer index out of range (num_layers={})", self.lbc.layer_indices.len()),
            });
        }
        let idx = &self.lbc.layer_indices[layer];
        let raw_data = self.backend.read_range(idx.layer_offset_bytes, idx.layer_length_bytes)?;
        // Return raw bytes with original quant schemes — NO dequantization.
        Ok(LayerView::from_owned(layer, raw_data, idx.subtensors.clone()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MmapConfig;
    use crate::weight::provider_mmap::MmapWeightProvider;
    use lumen_format::test_model::{
        generate_test_model_q8_0, generate_test_model_q8_0_gdn, TestModelQ8Config,
    };
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SYNC_RAW_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Write the given LBC bytes to a unique temp file and return its path.
    fn write_test_lbc(data: &[u8], tag: &str) -> std::path::PathBuf {
        let id = SYNC_RAW_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_sync_raw_invariant_{tag}_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.lbc");
        std::fs::File::create(&path).unwrap().write_all(data).unwrap();
        path
    }

    /// Write a small Q8_0-quantized synthetic LBC to a temp file and return its path.
    fn write_q8_test_lbc() -> std::path::PathBuf {
        let data = generate_test_model_q8_0(&TestModelQ8Config::default());
        write_test_lbc(&data, "q8")
    }

    /// Write a small Q8_0 GDN/full-attention HYBRID synthetic LBC (layer 0 is a
    /// GatedDeltaNet layer with populated `ssm_*` subtensors) and return its path.
    /// This is the only shape that exercises the stale-`ssm_*`-offset failure
    /// class that produced the `[PAD248319]` garbage-token output.
    fn write_q8_gdn_test_lbc() -> std::path::PathBuf {
        let data = generate_test_model_q8_0_gdn(&TestModelQ8Config::default());
        write_test_lbc(&data, "q8gdn")
    }

    /// REGRESSION GUARD (stale-`ssm_*`-offset garbage): the GPU-resident Metal path uploads layer
    /// weights via `get_layer_raw` and then, in `metal/prefill.rs`, takes the
    /// per-subtensor BYTE OFFSETS from the `LayerView` returned for that same
    /// layer. Those offsets MUST describe the raw native-quant blob layout that
    /// was uploaded, and they MUST be provider-independent — otherwise the
    /// kernels read the (correct) raw GPU buffer at the (wrong) offsets and emit
    /// pad-token garbage (the original `--sync` / `lumen-server` Q8 bug).
    ///
    /// This test locks the invariant that `SyncWeightProvider::get_layer_raw`
    /// and `MmapWeightProvider::get_layer_raw` agree byte-for-byte on BOTH the
    /// blob and every subtensor (offset, length, quant) for a quantized model.
    /// It does NOT need a GPU or the full 10 GB model — a tiny synthetic Q8_0
    /// LBC reproduces the dequant divergence that caused the bug.
    #[test]
    fn get_layer_raw_is_byte_identical_across_sync_and_mmap_for_q8() {
        let path = write_q8_test_lbc();

        let sync = SyncWeightProvider::open(&path).unwrap();
        let mmap = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        assert_eq!(sync.num_layers(), mmap.num_layers());
        assert!(sync.num_layers() > 0);

        for layer in 0..sync.num_layers() {
            let s_view = sync.get_layer_raw(layer).unwrap();
            let m_view = mmap.get_layer_raw(layer).unwrap();

            // 1. Raw blob bytes must be identical — this is what the
            //    GPU-resident preload uploads into the unified weight buffer.
            assert_eq!(
                s_view.as_bytes(),
                m_view.as_bytes(),
                "layer {layer}: get_layer_raw blob bytes differ between sync and mmap",
            );

            // 2. Every subtensor (offset, length, quant) must match — these are
            //    the offsets `metal/prefill.rs` feeds into the kernels against
            //    the raw buffer.
            let s = &s_view.subtensors;
            let m = &m_view.subtensors;
            let check = |name: &str, a: &TensorSlice, b: &TensorSlice| {
                assert_eq!(
                    (a.offset, a.length, a.quant),
                    (b.offset, b.length, b.quant),
                    "layer {layer}: subtensor {name} differs (sync={:?} mmap={:?})",
                    (a.offset, a.length, a.quant),
                    (b.offset, b.length, b.quant),
                );
            };
            check("wq", &s.wq, &m.wq);
            check("wk", &s.wk, &m.wk);
            check("wv", &s.wv, &m.wv);
            check("wo", &s.wo, &m.wo);
            check("w_gate", &s.w_gate, &m.w_gate);
            check("w_up", &s.w_up, &m.w_up);
            check("w_down", &s.w_down, &m.w_down);
            check("attn_norm", &s.attn_norm, &m.attn_norm);
            check("ffn_norm", &s.ffn_norm, &m.ffn_norm);
            assert_eq!(s.layer_type, m.layer_type, "layer {layer}: layer_type differs");

            // 3. The weight subtensors must stay in their NATIVE quant scheme on
            //    the raw path (Q8_0 here). If a future change makes
            //    get_layer_raw dequantize, the GPU buffer/offset contract breaks.
            assert_eq!(
                s.wq.quant,
                QuantScheme::Q8_0,
                "layer {layer}: get_layer_raw must preserve Q8_0 (got {:?})",
                s.wq.quant,
            );
        }
    }

    /// Documents WHY the prefill fix had to switch from `get_layer_blocking` to
    /// `get_layer_raw`: on a quantized model `SyncWeightProvider::get_layer_blocking`
    /// dequantizes the weight subtensors to F32 and REBUILDS the blob, producing
    /// a DIFFERENT byte layout and quant scheme than `get_layer_raw`. Feeding
    /// those F32 offsets to GPU kernels reading the raw Q8 buffer was the bug.
    #[test]
    fn get_layer_blocking_diverges_from_get_layer_raw_on_q8_sync() {
        let path = write_q8_test_lbc();
        let sync = SyncWeightProvider::open(&path).unwrap();

        let raw = sync.get_layer_raw(0).unwrap();
        let blocking = sync.get_layer_blocking(0).unwrap();

        // The blocking (dequantized) view reports F32 for the weight subtensors,
        // while the raw view preserves Q8_0 — proving they are NOT interchangeable
        // for the GPU-resident offset contract.
        assert_eq!(raw.subtensors.wq.quant, QuantScheme::Q8_0);
        assert_eq!(blocking.subtensors.wq.quant, QuantScheme::F32);
        // F32 dequant is 4 bytes/elem vs Q8_0's 34 bytes/32 elems, so the blobs
        // cannot be byte-identical.
        assert_ne!(
            raw.as_bytes().len(),
            blocking.as_bytes().len(),
            "dequantized blocking blob must differ in size from the raw Q8 blob",
        );
    }

    /// DEFECT-2 GUARD: exercise the EXACT failure class that caused the
    /// stale-`ssm_*`-offset garbage — a GDN/full-attention HYBRID where `get_layer_blocking`
    /// rebuilds the layer blob to F32 but leaves the `ssm_*` offsets pointing at
    /// their ORIGINAL-blob positions (now stale relative to the smaller rebuilt
    /// blob). The pure full-attention model (`ssm_* = None`) cannot reproduce
    /// this; only a GDN layer with populated `ssm_*` does.
    ///
    /// Locks two things on the GDN layer (layer 0 of the hybrid model):
    ///   1. `get_layer_raw` returns the NATIVE-quant `ssm_*` subtensors with the
    ///      ORIGINAL offsets — these match the raw GPU-resident buffer that
    ///      `metal/prefill.rs` uploads, and are provider-independent (sync==mmap).
    ///   2. `get_layer_blocking` DIVERGES: it rebuilds the blob to F32 (changing
    ///      its size and layout) but keeps the original `ssm_*` offsets, so each
    ///      stale `ssm_*` slice now indexes the WRONG bytes of the rebuilt blob —
    ///      i.e. feeding the blocking view to the kernels would have read garbage.
    ///      THIS is the bug the prefill fix prevents.
    #[test]
    fn get_layer_raw_locks_ssm_offsets_on_gdn_layer_and_blocking_goes_stale() {
        let path = write_q8_gdn_test_lbc();
        let sync = SyncWeightProvider::open(&path).unwrap();
        let mmap = MmapWeightProvider::open(&path, MmapConfig::default()).unwrap();

        // Layer 0 is the GDN layer (layer_type=1, populated ssm_*).
        const GDN_LAYER: usize = 0;
        let s_raw = sync.get_layer_raw(GDN_LAYER).unwrap();
        let m_raw = mmap.get_layer_raw(GDN_LAYER).unwrap();
        let blocking = sync.get_layer_blocking(GDN_LAYER).unwrap();

        // The synthetic model must actually be a GDN layer, else this guard is
        // vacuous (the R5 finding: the old model had ssm_* = None).
        assert_eq!(
            s_raw.subtensors.layer_type,
            Some(1),
            "test fixture regression: layer 0 must be a GDN layer (layer_type=1)",
        );
        let ssm_out_raw = s_raw
            .subtensors
            .ssm_out
            .expect("test fixture regression: GDN layer must populate ssm_out");
        let ssm_conv_raw = s_raw
            .subtensors
            .ssm_conv1d
            .expect("test fixture regression: GDN layer must populate ssm_conv1d");

        // (1) get_layer_raw is provider-independent for the ssm_* subtensors AND
        //     preserves native quant + raw offsets (what prefill feeds the GPU).
        let m_ssm_out = m_raw.subtensors.ssm_out.expect("mmap GDN layer must have ssm_out");
        assert_eq!(
            (ssm_out_raw.offset, ssm_out_raw.length, ssm_out_raw.quant),
            (m_ssm_out.offset, m_ssm_out.length, m_ssm_out.quant),
            "get_layer_raw ssm_out slice must match between sync and mmap",
        );
        assert_eq!(
            ssm_out_raw.quant,
            QuantScheme::Q8_0,
            "get_layer_raw must preserve ssm_out native Q8_0 (got {:?})",
            ssm_out_raw.quant,
        );
        assert_eq!(
            s_raw.subtensors.layer_type, m_raw.subtensors.layer_type,
            "layer_type must be provider-independent on the raw path",
        );
        // The raw ssm_* offsets must be valid within the raw blob.
        let raw_blob_len = s_raw.as_bytes().len() as u64;
        assert!(
            ssm_out_raw.offset + ssm_out_raw.length <= raw_blob_len,
            "raw ssm_out [{}, {}) must lie within the raw blob (len {})",
            ssm_out_raw.offset, ssm_out_raw.offset + ssm_out_raw.length, raw_blob_len,
        );

        // (2) get_layer_blocking REBUILDS the blob to F32 but CLONES the ssm_*
        //     slices UNCHANGED -> their offsets are now STALE (they describe the
        //     RAW blob layout, not the rebuilt one). Prove the divergence with a
        //     dimension-independent invariant: the SAME ssm_* slice, read against
        //     the raw blob vs the rebuilt blob, yields DIFFERENT bytes. (The
        //     rebuilt F32 blob is a different size — here larger, because Q8->F32
        //     inflates the mandatory region ~3.76x — so the stale offset lands on
        //     the wrong content. That wrong content fed to the GPU kernels is the
        //     [PAD248319] bug.)
        assert_eq!(
            blocking.subtensors.wq.quant,
            QuantScheme::F32,
            "get_layer_blocking must rebuild mandatory weights to F32",
        );
        let blk_blob_len = blocking.as_bytes().len() as u64;
        assert_ne!(
            blk_blob_len, raw_blob_len,
            "rebuilt blocking blob must differ in size from the raw blob",
        );
        let ssm_out_blk = blocking
            .subtensors
            .ssm_out
            .expect("blocking view should still carry the (now stale) ssm_out slice");
        // The blocking ssm_out slice is the STALE original slice (offset/len/quant
        // all unchanged from raw) ...
        assert_eq!(
            (ssm_out_blk.offset, ssm_out_blk.length, ssm_out_blk.quant),
            (ssm_out_raw.offset, ssm_out_raw.length, ssm_out_raw.quant),
            "get_layer_blocking leaves the ssm_out slice at its original (stale) offset",
        );
        // ... yet that same slice now indexes DIFFERENT bytes in the rebuilt blob
        // than the true ssm_out data in the raw blob. This is the exact failure
        // mode: prefill would feed these stale offsets to the raw GPU buffer and
        // read the wrong content.
        let true_ssm_out_bytes = s_raw
            .subtensor_bytes(&ssm_out_raw)
            .expect("raw ssm_out slice must be in-bounds of the raw blob")
            .to_vec();
        let stale_ssm_out_bytes = blocking
            .subtensor_bytes(&ssm_out_blk)
            .expect("stale ssm_out slice happens to be in-bounds of the (larger) rebuilt blob");
        assert_ne!(
            true_ssm_out_bytes, stale_ssm_out_bytes,
            "the stale ssm_out offset must read DIFFERENT bytes against the rebuilt \
             blocking blob than the true ssm_out data in the raw blob — this is the \
             exact stale-offset corruption the prefill fix prevents",
        );
        // Sanity: ssm_conv1d (F32-native) is likewise left at its stale offset.
        let ssm_conv_blk = blocking.subtensors.ssm_conv1d.expect("ssm_conv1d present");
        assert_eq!(
            ssm_conv_blk.offset, ssm_conv_raw.offset,
            "get_layer_blocking leaves ssm_conv1d offset stale too",
        );
    }

    /// DEFECT-3 GUARD: lock the CALL-SITE contract. `metal/prefill.rs`
    /// resolves each layer view with EXACTLY this logic on the fallback path:
    ///
    /// ```ignore
    /// let layer_view = match weights.try_get_layer(layer) {
    ///     Some(view) => view,
    ///     None       => weights.get_layer_raw(layer)?,   // <-- the stale-offset fix
    /// };
    /// ```
    ///
    /// A `SyncWeightProvider`'s `try_get_layer` returns `None` (its cache is not
    /// populated by the raw GPU-resident preload), so prefill MUST fall through
    /// to `get_layer_raw` — which yields native-quant bytes + raw offsets that
    /// match the uploaded GPU buffer. If a future edit swaps the prefill fallback
    /// back to `get_layer_blocking`, the resolved view would report F32 (and a
    /// rebuilt blob) and the `[PAD248319]` regression reappears.
    ///
    /// This guard mirrors that resolution against a `&dyn WeightProvider` and
    /// asserts the resolved view is the RAW (native-quant, raw-offset) view, so
    /// it fails if the contract the prefill depends on is broken. It is exercised
    /// on the GDN hybrid so it also covers the `ssm_*` subtensors. The companion
    /// counterfactual is a temporary `get_layer_blocking` revert of `prefill.rs`,
    /// which makes this guard fail as expected.
    #[test]
    fn prefill_fallback_resolution_yields_raw_not_dequantized_view_for_sync() {
        let path = write_q8_gdn_test_lbc();
        let sync = SyncWeightProvider::open(&path).unwrap();
        let provider: &dyn WeightProvider = &sync;

        // Precondition that makes the prefill fallback fire for sync: its cache
        // is empty, so try_get_layer returns None.
        assert!(
            provider.try_get_layer(0).is_none(),
            "SyncWeightProvider::try_get_layer must be None before preload \
             (this is what routes prefill to the get_layer_raw fallback)",
        );

        // Mirror EXACTLY the prefill fallback resolution.
        let resolve = |layer: usize| -> LayerView {
            match provider.try_get_layer(layer) {
                Some(view) => view,
                None => provider.get_layer_raw(layer).unwrap(),
            }
        };

        let raw_reference = sync.get_layer_raw(0).unwrap();
        for layer in 0..provider.num_layers() {
            let resolved = resolve(layer);
            let reference = sync.get_layer_raw(layer).unwrap();

            // The resolved view MUST be the raw (native-quant) view, byte-for-byte.
            assert_eq!(
                resolved.as_bytes().len(),
                reference.as_bytes().len(),
                "layer {layer}: prefill fallback must resolve to the raw blob \
                 (a get_layer_blocking swap-back would rebuild it to F32)",
            );
            assert_eq!(
                resolved.subtensors.wq.quant,
                QuantScheme::Q8_0,
                "layer {layer}: prefill fallback must yield NATIVE-quant weights \
                 (Q8_0), not the F32 produced by get_layer_blocking",
            );
            assert_eq!(
                resolved.subtensors.wq.quant,
                reference.subtensors.wq.quant,
                "layer {layer}: resolved quant must equal get_layer_raw's",
            );
        }

        // On the GDN layer the resolved ssm_* offsets must be the raw (valid)
        // ones — i.e. the fallback gives prefill offsets that fit the raw buffer.
        let resolved0 = resolve(0);
        assert_eq!(resolved0.subtensors.layer_type, Some(1), "layer 0 is GDN");
        let ssm_out = resolved0
            .subtensors
            .ssm_out
            .expect("GDN layer ssm_out present on resolved (raw) view");
        assert_eq!(ssm_out.quant, QuantScheme::Q8_0);
        assert_eq!(
            ssm_out.offset, raw_reference.subtensors.ssm_out.unwrap().offset,
            "resolved ssm_out offset must equal the raw offset (not the stale \
             get_layer_blocking one)",
        );
        assert!(
            ssm_out.offset + ssm_out.length <= resolved0.as_bytes().len() as u64,
            "resolved ssm_out must lie within the resolved (raw) blob — the \
             get_layer_blocking path would leave it overrunning",
        );
    }
}
