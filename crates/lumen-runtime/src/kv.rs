//! KV cache management types.
//!
//! The KV cache stores key and value projections from attention layers so
//! they can be reused across token generation steps. In v1, the KV cache
//! is RAM-resident. Future versions may support disk-backed KV (KVSwap-style).
//!
//! The runtime manages KV memory explicitly and provides knobs for precision,
//! eviction policy, and chunking (paged KV).

use crate::error::RuntimeError;
use crate::compute::simd_kernels;

/// Precision for KV cache storage.
///
/// Lower precision reduces memory footprint at the cost of potential
/// quality degradation. FlexGen reports compressing attention cache
/// to 4 bits with negligible accuracy loss in some scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPrecision {
    F32,
    F16,
    /// 8-bit integer quantized.
    Int8,
    /// 4-bit integer quantized (most compressed).
    Int4,
}

impl KvPrecision {
    /// Bytes per element for this precision.
    ///
    /// Note: Int4 actually uses 0.5 bytes per element (two elements packed per
    /// byte), but `usize` cannot represent fractional bytes. This method
    /// returns 1 for Int4 as the minimum addressable unit, which makes memory
    /// estimates conservative (2x overestimate for Int4).
    ///
    /// Int8 and Int4 are reserved for future quantized KV support and are not
    /// yet implemented.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // Reserved for future use -- see `is_implemented()`.
            Self::Int8 => 1,
            // Int4 stores two elements per byte; returns 1 as the minimum
            // addressable unit. Reserved for future use -- see `is_implemented()`.
            Self::Int4 => 1,
        }
    }

    /// Returns `true` if this precision is fully implemented in the KV cache
    /// read/write paths.
    ///
    /// Currently F32 and F16 are supported. Int8 and Int4 are defined
    /// for forward-compatible configuration but will be rejected at
    /// [`KvCache::new`] time until the corresponding KV storage paths are
    /// implemented.
    pub fn is_implemented(&self) -> bool {
        match self {
            Self::F32 | Self::F16 => true,
            Self::Int8 | Self::Int4 => false,
        }
    }
}

/// Configuration for the KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Maximum sequence length to support.
    pub max_seq_len: usize,

    /// Number of transformer layers.
    pub num_layers: usize,

    /// Number of KV heads per layer.
    pub num_kv_heads: usize,

    /// Dimension per head.
    pub head_dim: usize,

    /// Storage precision for cached KV values.
    pub precision: KvPrecision,
}

impl KvCacheConfig {
    /// Compute the memory required for the full KV cache (all layers,
    /// max sequence length), in bytes.
    pub fn full_memory_bytes(&self) -> u64 {
        self.bytes_per_token() * self.max_seq_len as u64
    }

    /// Compute the memory required per token across all layers, in bytes.
    pub fn bytes_per_token(&self) -> u64 {
        let per_token_per_head = self.head_dim * self.precision.bytes_per_element();
        let per_token_per_layer = per_token_per_head * self.num_kv_heads * 2; // K + V
        (per_token_per_layer * self.num_layers) as u64
    }
}

/// A mutable view into the KV cache for a single layer.
///
/// The compute backend receives this during `compute_layer` to read
/// previous KV entries and write new ones.
///
/// ## Memory layout: head-first
///
/// Buffers are laid out as `[head][position][dim]`. For a given KV head,
/// all positions form a contiguous block, which means a single-head scan
/// over all positions is a simple sequential read with zero wasted cache
/// lines. The per-head stride in bytes is `max_seq_len * head_dim * bpe`.
///
/// The full buffer is pre-allocated at KV cache creation time; `seq_len`
/// tracks how many positions have been written.
#[derive(Debug)]
pub struct KvCacheView {
    /// Layer index this view is for.
    pub layer_idx: usize,

    /// Key cache data. Layout: `[num_kv_heads][max_seq_len][head_dim]`.
    /// Pre-allocated to full size; only positions `0..seq_len` are valid.
    pub keys: Vec<u8>,

    /// Value cache data. Layout: `[num_kv_heads][max_seq_len][head_dim]`.
    /// Pre-allocated to full size; only positions `0..seq_len` are valid.
    pub values: Vec<u8>,

    /// Current sequence length (number of tokens with cached KV).
    pub seq_len: usize,

    /// Number of KV heads.
    pub num_kv_heads: usize,

    /// Dimension per head.
    pub head_dim: usize,

    /// Maximum sequence length (needed for head-first offset computation).
    pub max_seq_len: usize,

    /// Precision of the stored data.
    pub precision: KvPrecision,
}

// ---- Scatter-write helpers (free functions to avoid borrow conflicts) --------

/// Scatter-write f32 data into a byte buffer in head-first `[head][pos][dim]` layout.
///
/// Copies `num_kv_heads` slices of `head_dim` f32 elements from contiguous `data`
/// into the corresponding head/pos/dim positions in `buf`.
#[inline]
fn scatter_write_f32(
    buf: &mut [u8],
    data: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq_len: usize,
) {
    assert_eq!(data.len(), num_kv_heads * head_dim, "scatter_write_f32: data length mismatch");
    debug_assert!(pos < max_seq_len, "scatter_write_f32: pos={pos} >= max_seq_len={max_seq_len}");
    let head_bytes = head_dim * 4;
    #[cfg(target_endian = "little")]
    {
        for h in 0..num_kv_heads {
            let src_byte = h * head_bytes;
            let dst_byte = (h * max_seq_len * head_dim + pos * head_dim) * 4;
            debug_assert!(dst_byte + head_bytes <= buf.len());
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (data.as_ptr() as *const u8).add(src_byte),
                    buf.as_mut_ptr().add(dst_byte),
                    head_bytes,
                );
            }
        }
    }
    #[cfg(target_endian = "big")]
    {
        for h in 0..num_kv_heads {
            let dst_byte = (h * max_seq_len * head_dim + pos * head_dim) * 4;
            for e in 0..head_dim {
                let val = data[h * head_dim + e];
                let bytes = val.to_le_bytes();
                let off = dst_byte + e * 4;
                buf[off..off + 4].copy_from_slice(&bytes);
            }
        }
    }
}

/// Scatter-write f32 data as f16 into a byte buffer in head-first `[head][pos][dim]` layout.
///
/// Converts each f32 element to f16 and writes to the corresponding position.
/// Uses NEON FCVTN batch conversion on aarch64 for high throughput.
#[inline]
fn scatter_write_f16_from_f32(
    buf: &mut [u8],
    data: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq_len: usize,
) {
    assert_eq!(data.len(), num_kv_heads * head_dim, "scatter_write_f16: data length mismatch");
    debug_assert!(pos < max_seq_len, "scatter_write_f16: pos={pos} >= max_seq_len={max_seq_len}");
    for h in 0..num_kv_heads {
        let src_offset = h * head_dim;
        let dst_byte = (h * max_seq_len * head_dim + pos * head_dim) * 2;
        debug_assert!(dst_byte + head_dim * 2 <= buf.len());
        unsafe {
            let src = data.as_ptr().add(src_offset);
            let dst = buf.as_mut_ptr().add(dst_byte);
            #[cfg(target_arch = "aarch64")]
            simd_kernels::f32_to_f16_batch(src, dst, head_dim);
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..head_dim {
                    let bits = simd_kernels::f32_to_f16_inline(*src.add(i));
                    std::ptr::copy_nonoverlapping(
                        &bits as *const u16 as *const u8,
                        dst.add(i * 2),
                        2,
                    );
                }
            }
        }
    }
}

impl KvCacheView {
    /// Byte size of a single token's KV entry (K + V) for this layer.
    pub fn bytes_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim * self.precision.bytes_per_element() * 2
    }

    /// Total bytes currently used by this layer's KV cache.
    pub fn current_bytes(&self) -> usize {
        self.seq_len * self.bytes_per_token()
    }

    /// Byte offset for head-first layout: `[head][pos][dim]`.
    /// Returns the byte offset into the keys or values buffer for the start
    /// of `head`'s data at position `pos`.
    #[inline(always)]
    pub fn head_pos_byte_offset(&self, head: usize, pos: usize) -> usize {
        let bpe = self.precision.bytes_per_element();
        (head * self.max_seq_len * self.head_dim + pos * self.head_dim) * bpe
    }

    /// Read an f32 value from the key cache at the given element index.
    /// Only valid when precision is F32.
    /// Uses unsafe read_unaligned on little-endian platforms to skip bounds checks.
    #[inline(always)]
    pub fn read_key_f32(&self, element_idx: usize) -> f32 {
        let offset = element_idx * 4;
        debug_assert!(offset + 4 <= self.keys.len());
        #[cfg(target_endian = "little")]
        unsafe {
            // SAFETY: offset+4 within bounds (debug_assert above),
            // read_unaligned does not require alignment.
            std::ptr::read_unaligned(self.keys.as_ptr().add(offset) as *const f32)
        }
        #[cfg(target_endian = "big")]
        {
            f32::from_le_bytes(self.keys[offset..offset + 4].try_into().unwrap())
        }
    }

    /// Read an f32 value from the value cache at the given element index.
    /// Only valid when precision is F32.
    /// Uses unsafe read_unaligned on little-endian platforms to skip bounds checks.
    #[inline(always)]
    pub fn read_value_f32(&self, element_idx: usize) -> f32 {
        let offset = element_idx * 4;
        debug_assert!(offset + 4 <= self.values.len());
        #[cfg(target_endian = "little")]
        unsafe {
            // SAFETY: offset+4 within bounds (debug_assert above),
            // read_unaligned does not require alignment.
            std::ptr::read_unaligned(self.values.as_ptr().add(offset) as *const f32)
        }
        #[cfg(target_endian = "big")]
        {
            f32::from_le_bytes(self.values[offset..offset + 4].try_into().unwrap())
        }
    }

    /// Get a direct f32 slice into the key cache. Zero-copy on little-endian.
    ///
    /// Returns a `&[f32]` view directly into the underlying `Vec<u8>` storage,
    /// avoiding the memcpy that `read_keys_f32_into` performs. This is the
    /// preferred access method in the SIMD attention inner loop.
    ///
    /// # Safety
    /// Only valid when precision is F32 and on little-endian platforms (where
    /// the in-memory byte representation of f32 IS the LE encoding). Both
    /// conditions hold for all our targets (aarch64/x86_64 with KvPrecision::F32).
    /// `Vec<u8>` from `Vec::reserve` is always at least byte-aligned, and f32
    /// `read_unaligned` is used implicitly via `from_raw_parts` on naturally-
    /// aligned heap allocations.
    #[cfg(target_endian = "little")]
    #[inline(always)]
    pub fn keys_f32_slice(&self, start_element_idx: usize, count: usize) -> &[f32] {
        let byte_start = start_element_idx * 4;
        debug_assert!(byte_start + count * 4 <= self.keys.len());
        unsafe {
            std::slice::from_raw_parts(
                self.keys.as_ptr().add(byte_start) as *const f32,
                count,
            )
        }
    }

    /// Get a direct f32 slice into the value cache. Zero-copy on little-endian.
    ///
    /// Same safety rationale as `keys_f32_slice`.
    #[cfg(target_endian = "little")]
    #[inline(always)]
    pub fn values_f32_slice(&self, start_element_idx: usize, count: usize) -> &[f32] {
        let byte_start = start_element_idx * 4;
        debug_assert!(byte_start + count * 4 <= self.values.len());
        unsafe {
            std::slice::from_raw_parts(
                self.values.as_ptr().add(byte_start) as *const f32,
                count,
            )
        }
    }

    /// Bulk-read f32 keys into a pre-allocated buffer.
    /// Reads `count` contiguous f32 values starting at `start_element_idx`.
    /// Uses unsafe memcpy on little-endian platforms for zero per-element overhead.
    #[inline(always)]
    pub fn read_keys_f32_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        debug_assert!(buf.len() >= count);
        let byte_start = start_element_idx * 4;
        debug_assert!(byte_start + count * 4 <= self.keys.len());
        #[cfg(target_endian = "little")]
        unsafe {
            // SAFETY: f32 is 4 bytes LE, buf has sufficient capacity,
            // byte_start + count*4 within bounds (debug_asserts above).
            std::ptr::copy_nonoverlapping(
                self.keys.as_ptr().add(byte_start),
                buf.as_mut_ptr() as *mut u8,
                count * 4,
            );
        }
        #[cfg(target_endian = "big")]
        {
            for i in 0..count {
                let offset = byte_start + i * 4;
                buf[i] = f32::from_le_bytes(self.keys[offset..offset + 4].try_into().unwrap());
            }
        }
    }

    /// Bulk-read f32 values into a pre-allocated buffer.
    /// Reads `count` contiguous f32 values starting at `start_element_idx`.
    /// Uses unsafe memcpy on little-endian platforms for zero per-element overhead.
    #[inline(always)]
    pub fn read_values_f32_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        debug_assert!(buf.len() >= count);
        let byte_start = start_element_idx * 4;
        debug_assert!(byte_start + count * 4 <= self.values.len());
        #[cfg(target_endian = "little")]
        unsafe {
            // SAFETY: f32 is 4 bytes LE, buf has sufficient capacity,
            // byte_start + count*4 within bounds (debug_asserts above).
            std::ptr::copy_nonoverlapping(
                self.values.as_ptr().add(byte_start),
                buf.as_mut_ptr() as *mut u8,
                count * 4,
            );
        }
        #[cfg(target_endian = "big")]
        {
            for i in 0..count {
                let offset = byte_start + i * 4;
                buf[i] = f32::from_le_bytes(self.values[offset..offset + 4].try_into().unwrap());
            }
        }
    }

    /// Append f32 key data in head-first layout.
    ///
    /// Input `data` has shape `[num_kv_heads][head_dim]` (contiguous, one
    /// token's worth). Each head's slice is scatter-written to the correct
    /// position in `[head][pos][dim]` layout.
    #[inline(always)]
    pub fn append_keys_f32(&mut self, data: &[f32]) {
        if self.seq_len >= self.max_seq_len { return; }
        scatter_write_f32(&mut self.keys, data, self.num_kv_heads, self.head_dim, self.seq_len, self.max_seq_len);
    }

    /// Append f32 value data in head-first layout.
    ///
    /// Same scatter-write logic as `append_keys_f32`.
    #[inline(always)]
    pub fn append_values_f32(&mut self, data: &[f32]) {
        if self.seq_len >= self.max_seq_len { return; }
        scatter_write_f32(&mut self.values, data, self.num_kv_heads, self.head_dim, self.seq_len, self.max_seq_len);
    }

    // ---- F16 KV cache methods ------------------------------------------------

    /// Convert f32 input to f16 and scatter-write to key cache in head-first layout.
    /// Uses NEON FCVTN batch conversion on aarch64 for high throughput.
    #[inline(always)]
    pub fn append_keys_f16_from_f32(&mut self, data: &[f32]) {
        if self.seq_len >= self.max_seq_len { return; }
        scatter_write_f16_from_f32(&mut self.keys, data, self.num_kv_heads, self.head_dim, self.seq_len, self.max_seq_len);
    }

    /// Convert f32 input to f16 and scatter-write to value cache in head-first layout.
    /// Uses NEON FCVTN batch conversion on aarch64 for high throughput.
    #[inline(always)]
    pub fn append_values_f16_from_f32(&mut self, data: &[f32]) {
        if self.seq_len >= self.max_seq_len { return; }
        scatter_write_f16_from_f32(&mut self.values, data, self.num_kv_heads, self.head_dim, self.seq_len, self.max_seq_len);
    }

    /// Precision-aware key append: dispatches to F32 or F16 path based on `self.precision`.
    #[inline(always)]
    pub fn append_keys(&mut self, data: &[f32]) {
        match self.precision {
            KvPrecision::F32 => self.append_keys_f32(data),
            KvPrecision::F16 => self.append_keys_f16_from_f32(data),
            _ => unreachable!("unimplemented KV precision {:?}", self.precision),
        }
    }

    /// Precision-aware value append: dispatches to F32 or F16 path based on `self.precision`.
    #[inline(always)]
    pub fn append_values(&mut self, data: &[f32]) {
        match self.precision {
            KvPrecision::F32 => self.append_values_f32(data),
            KvPrecision::F16 => self.append_values_f16_from_f32(data),
            _ => unreachable!("unimplemented KV precision {:?}", self.precision),
        }
    }

    /// Read F16 keys and dequantize to f32 into a pre-allocated buffer.
    /// Uses NEON FCVTL batch conversion on aarch64 for high throughput.
    #[inline(always)]
    pub fn read_keys_f16_to_f32_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        debug_assert!(buf.len() >= count);
        let byte_start = start_element_idx * 2;
        debug_assert!(byte_start + count * 2 <= self.keys.len());
        unsafe {
            let src = self.keys.as_ptr().add(byte_start);
            #[cfg(target_arch = "aarch64")]
            simd_kernels::f16_to_f32_batch(src, buf.as_mut_ptr(), count);
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..count {
                    let bits = std::ptr::read_unaligned(src.add(i * 2) as *const u16);
                    *buf.get_unchecked_mut(i) = simd_kernels::f16_to_f32_inline(bits);
                }
            }
        }
    }

    /// Read F16 values and dequantize to f32 into a pre-allocated buffer.
    /// Uses NEON FCVTL batch conversion on aarch64 for high throughput.
    #[inline(always)]
    pub fn read_values_f16_to_f32_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        debug_assert!(buf.len() >= count);
        let byte_start = start_element_idx * 2;
        debug_assert!(byte_start + count * 2 <= self.values.len());
        unsafe {
            let src = self.values.as_ptr().add(byte_start);
            #[cfg(target_arch = "aarch64")]
            simd_kernels::f16_to_f32_batch(src, buf.as_mut_ptr(), count);
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..count {
                    let bits = std::ptr::read_unaligned(src.add(i * 2) as *const u16);
                    *buf.get_unchecked_mut(i) = simd_kernels::f16_to_f32_inline(bits);
                }
            }
        }
    }

    /// Precision-aware bulk key read into a pre-allocated f32 buffer.
    /// Dispatches to F32 memcpy or F16 dequantize based on `self.precision`.
    #[inline(always)]
    pub fn read_keys_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        match self.precision {
            KvPrecision::F32 => self.read_keys_f32_into(buf, start_element_idx, count),
            KvPrecision::F16 => self.read_keys_f16_to_f32_into(buf, start_element_idx, count),
            _ => unreachable!("unimplemented KV precision {:?}", self.precision),
        }
    }

    /// Precision-aware bulk value read into a pre-allocated f32 buffer.
    /// Dispatches to F32 memcpy or F16 dequantize based on `self.precision`.
    #[inline(always)]
    pub fn read_values_into(&self, buf: &mut [f32], start_element_idx: usize, count: usize) {
        match self.precision {
            KvPrecision::F32 => self.read_values_f32_into(buf, start_element_idx, count),
            KvPrecision::F16 => self.read_values_f16_to_f32_into(buf, start_element_idx, count),
            _ => unreachable!("unimplemented KV precision {:?}", self.precision),
        }
    }
}

/// The in-memory KV cache holding entries for all layers.
///
/// This is the v1 RAM-resident implementation. Each layer has a
/// contiguous buffer for keys and values that grows as tokens are
/// generated.
pub struct KvCache {
    config: KvCacheConfig,

    /// Per-layer key caches. Outer index = layer.
    keys: Vec<Vec<u8>>,

    /// Per-layer value caches. Outer index = layer.
    values: Vec<Vec<u8>>,

    /// Current sequence length.
    seq_len: usize,
}

impl KvCache {
    /// Allocate a new KV cache.
    ///
    /// Returns [`RuntimeError::Unsupported`] if `config.precision` specifies a
    /// quantized format (Int8 or Int4) that is not yet implemented. Accepting
    /// an unimplemented precision would silently corrupt data.
    pub fn new(config: KvCacheConfig) -> Result<Self, RuntimeError> {
        if !config.precision.is_implemented() {
            return Err(RuntimeError::Unsupported(format!(
                "KV cache precision {:?} is not yet implemented",
                config.precision
            )));
        }
        let num_layers = config.num_layers;
        // Head-first layout: [head][pos][dim] — pre-fill entire buffer.
        // This allows scatter-writes during append and contiguous reads per head.
        let per_layer_bytes = config.num_kv_heads
            * config.head_dim
            * config.precision.bytes_per_element()
            * config.max_seq_len;
        let keys = (0..num_layers)
            .map(|_| vec![0u8; per_layer_bytes])
            .collect();
        let values = (0..num_layers)
            .map(|_| vec![0u8; per_layer_bytes])
            .collect();
        Ok(Self {
            config,
            keys,
            values,
            seq_len: 0,
        })
    }

    /// Get a mutable view into a specific layer's KV cache.
    ///
    /// Uses `std::mem::take` to move data out without cloning.
    /// The caller MUST call `commit_view` after compute to move data back.
    pub fn view_mut(&mut self, layer_idx: usize) -> Result<KvCacheView, RuntimeError> {
        if layer_idx >= self.config.num_layers {
            return Err(RuntimeError::KvCache(format!(
                "layer index {layer_idx} out of range (num_layers={})",
                self.config.num_layers
            )));
        }

        Ok(KvCacheView {
            layer_idx,
            keys: std::mem::take(&mut self.keys[layer_idx]),
            values: std::mem::take(&mut self.values[layer_idx]),
            seq_len: self.seq_len,
            num_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            max_seq_len: self.config.max_seq_len,
            precision: self.config.precision,
        })
    }

    /// Commit updated KV data back from a view after compute.
    ///
    /// Moves the data back from the view into the cache (no copy).
    pub fn commit_view(&mut self, view: KvCacheView) -> Result<(), RuntimeError> {
        let idx = view.layer_idx;
        if idx >= self.config.num_layers {
            return Err(RuntimeError::KvCache(format!(
                "layer index {idx} out of range"
            )));
        }
        self.keys[idx] = view.keys;
        self.values[idx] = view.values;
        Ok(())
    }

    /// Advance the sequence length by one token.
    ///
    /// Called after all layers have processed a new token.
    /// Returns an error if advancing would exceed max_seq_len.
    pub fn advance_seq_len(&mut self) -> Result<(), RuntimeError> {
        if self.seq_len + 1 > self.config.max_seq_len {
            return Err(RuntimeError::KvCache(format!(
                "sequence length {} would exceed max_seq_len {}",
                self.seq_len + 1,
                self.config.max_seq_len
            )));
        }
        self.seq_len += 1;
        Ok(())
    }

    /// Returns the current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Returns the configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Total bytes currently used across all layers (actual data, not pre-allocated).
    pub fn total_bytes(&self) -> u64 {
        let bpe = self.config.precision.bytes_per_element() as u64;
        let per_token = self.config.num_kv_heads as u64
            * self.config.head_dim as u64
            * bpe
            * 2; // K + V
        self.seq_len as u64 * per_token * self.config.num_layers as u64
    }

    /// Reset the cache (e.g., for a new generation session).
    ///
    /// With head-first pre-filled layout, buffers are kept; only the
    /// seq_len cursor is zeroed. Stale data is harmless because reads
    /// only access positions < seq_len.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCache")
            .field("num_layers", &self.config.num_layers)
            .field("seq_len", &self.seq_len)
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            precision: KvPrecision::F32,
        }
    }

    #[test]
    fn kv_precision_bytes_per_element() {
        assert_eq!(KvPrecision::F32.bytes_per_element(), 4);
        assert_eq!(KvPrecision::F16.bytes_per_element(), 2);
        assert_eq!(KvPrecision::Int8.bytes_per_element(), 1);
        assert_eq!(KvPrecision::Int4.bytes_per_element(), 1);
    }

    #[test]
    fn kv_cache_config_bytes_per_token() {
        let cfg = test_config();
        // per_token_per_head = 4 * 4 = 16
        // per_token_per_layer = 16 * 2 * 2 = 64
        // bytes_per_token = 64 * 2 = 128
        assert_eq!(cfg.bytes_per_token(), 128);
    }

    #[test]
    fn kv_cache_config_full_memory_bytes() {
        let cfg = test_config();
        // bytes_per_token * max_seq_len = 128 * 16 = 2048
        assert_eq!(cfg.full_memory_bytes(), 2048);
    }

    #[test]
    fn kv_cache_new_correct_layers_and_seq_len() {
        let kv = KvCache::new(test_config()).unwrap();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.config().num_layers, 2);
        assert_eq!(kv.total_bytes(), 0);
    }

    #[test]
    fn kv_cache_prefilled_buffer_size() {
        let cfg = test_config();
        let kv = KvCache::new(cfg.clone()).unwrap();
        // Pre-filled: num_kv_heads * head_dim * bpe * max_seq_len
        // = 2 * 4 * 4 * 16 = 512 bytes per layer
        assert_eq!(kv.keys[0].len(), 512);
        assert_eq!(kv.values[0].len(), 512);
    }

    #[test]
    fn kv_cache_view_mut_and_commit() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();
        assert_eq!(view.layer_idx, 0);
        assert_eq!(view.seq_len, 0);
        assert_eq!(view.max_seq_len, 16);

        // Write one token (2 heads × 4 dim = 8 floats)
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        view.append_keys_f32(&data);
        view.append_values_f32(&data);

        kv.commit_view(view).unwrap();
        kv.advance_seq_len().unwrap();

        // total_bytes = 1 token × (2 heads × 4 dim × 4 bpe × 2 K+V) × 2 layers
        // But only layer 0 was written, and total_bytes uses seq_len * per_token * num_layers
        // seq_len=1, per_token=64, num_layers=2 → 128
        assert_eq!(kv.total_bytes(), 128);
    }

    #[test]
    fn kv_cache_view_mut_out_of_range() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let result = kv.view_mut(99);
        assert!(result.is_err());
    }

    #[test]
    fn kv_cache_advance_seq_len() {
        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: 3,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            precision: KvPrecision::F32,
        }).unwrap();
        assert_eq!(kv.seq_len(), 0);
        kv.advance_seq_len().unwrap();
        assert_eq!(kv.seq_len(), 1);
        kv.advance_seq_len().unwrap();
        kv.advance_seq_len().unwrap();
        assert_eq!(kv.seq_len(), 3);
        // Next should exceed max_seq_len
        assert!(kv.advance_seq_len().is_err());
    }

    #[test]
    fn kv_cache_reset() {
        let mut kv = KvCache::new(test_config()).unwrap();
        // Write one token
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let mut view = kv.view_mut(0).unwrap();
        view.append_keys_f32(&data);
        kv.commit_view(view).unwrap();
        kv.advance_seq_len().unwrap();

        assert!(kv.seq_len() > 0);
        assert!(kv.total_bytes() > 0);

        kv.reset();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.total_bytes(), 0);
        // Buffers are still pre-filled (not cleared)
        assert_eq!(kv.keys[0].len(), 512);
    }

    #[test]
    fn kv_cache_total_bytes_reflects_data() {
        let mut kv = KvCache::new(test_config()).unwrap();
        // Write one token to both layers
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        for layer in 0..2 {
            let mut view = kv.view_mut(layer).unwrap();
            view.append_keys_f32(&data);
            view.append_values_f32(&data);
            kv.commit_view(view).unwrap();
        }
        kv.advance_seq_len().unwrap();
        // 1 token × (2 heads × 4 dim × 4 bpe × 2 K+V) × 2 layers = 128
        assert_eq!(kv.total_bytes(), 128);
    }

    #[test]
    fn kv_cache_view_bytes_per_token() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let view = kv.view_mut(0).unwrap();
        // num_kv_heads=2, head_dim=4, F32=4 bytes, * 2 (K+V)
        // = 2 * 4 * 4 * 2 = 64
        assert_eq!(view.bytes_per_token(), 64);
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_view_append_and_read_f32_keys() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        // One token: 2 heads × 4 dim = 8 floats
        let data = [1.5f32, -2.0, 3.25, 0.0, 7.0, -0.5, 100.0, 42.0];
        view.append_keys_f32(&data);

        // Head-first layout: head 0 at element_idx 0..4, head 1 at element_idx 64..68
        // (head 1 base = 1 * 16 * 4 = 64 elements)
        for e in 0..4 {
            let got = view.read_key_f32(e);
            assert_eq!(got, data[e], "head0 key[{e}]: expected {}, got {got}", data[e]);
        }
        let head1_base = 1 * 16 * 4; // 64
        for e in 0..4 {
            let got = view.read_key_f32(head1_base + e);
            assert_eq!(got, data[4 + e], "head1 key[{e}]: expected {}, got {got}", data[4 + e]);
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_view_append_and_read_f32_values() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [0.1f32, 100.0, -0.001, f32::MAX, 0.5, -0.5, 1.0, 2.0];
        view.append_values_f32(&data);

        for e in 0..4 {
            let got = view.read_value_f32(e);
            assert_eq!(got, data[e], "head0 value[{e}]: expected {}, got {got}", data[e]);
        }
        let head1_base = 1 * 16 * 4;
        for e in 0..4 {
            let got = view.read_value_f32(head1_base + e);
            assert_eq!(got, data[4 + e], "head1 value[{e}]: expected {}, got {got}", data[4 + e]);
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_view_append_f32_multiple_positions() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        // Position 0
        let data0: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        view.append_keys_f32(&data0);
        view.seq_len = 1; // advance manually for second append

        // Position 1
        let data1: Vec<f32> = (11..=18).map(|i| i as f32).collect();
        view.append_keys_f32(&data1);

        // Head 0: pos 0 at element 0..4, pos 1 at element 4..8
        for e in 0..4 { assert_eq!(view.read_key_f32(e), data0[e]); }
        for e in 0..4 { assert_eq!(view.read_key_f32(4 + e), data1[e]); }

        // Head 1: base = 64. pos 0 at 64..68, pos 1 at 68..72
        let h1 = 16 * 4; // 64
        for e in 0..4 { assert_eq!(view.read_key_f32(h1 + e), data0[4 + e]); }
        for e in 0..4 { assert_eq!(view.read_key_f32(h1 + 4 + e), data1[4 + e]); }
        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_keys_f32_slice_head_first() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [1.5f32, -2.0, 3.25, 0.0, 7.0, -0.5, 100.0, 42.0];
        view.append_keys_f32(&data);

        // Head 0, pos 0: elements 0..4
        let slice = view.keys_f32_slice(0, 4);
        assert_eq!(slice, &[1.5, -2.0, 3.25, 0.0]);

        // Head 1, pos 0: elements 64..68
        let h1 = 16 * 4;
        let slice = view.keys_f32_slice(h1, 4);
        assert_eq!(slice, &[7.0, -0.5, 100.0, 42.0]);

        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_values_f32_slice_head_first() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [0.1f32, 100.0, -0.001, f32::MAX, 1.0, 2.0, 3.0, 4.0];
        view.append_values_f32(&data);

        let slice = view.values_f32_slice(0, 4);
        assert_eq!(slice, &[0.1, 100.0, -0.001, f32::MAX]);

        let h1 = 16 * 4;
        let slice = view.values_f32_slice(h1, 4);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);

        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_f32_slice_multiple_positions() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data0: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        view.append_keys_f32(&data0);
        view.seq_len = 1;

        let data1: Vec<f32> = (11..=18).map(|i| i as f32).collect();
        view.append_keys_f32(&data1);

        // Head 0: pos 0,1 are contiguous at elements 0..8
        let slice = view.keys_f32_slice(0, 8);
        assert_eq!(&slice[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&slice[4..8], &[11.0, 12.0, 13.0, 14.0]);

        // Verify slice matches read_keys_f32_into
        let mut buf = [0.0f32; 4];
        view.read_keys_f32_into(&mut buf, 4, 4);
        assert_eq!(&buf, &[11.0, 12.0, 13.0, 14.0]);

        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_precision_is_implemented() {
        assert!(KvPrecision::F32.is_implemented());
        assert!(KvPrecision::F16.is_implemented());
        assert!(!KvPrecision::Int8.is_implemented());
        assert!(!KvPrecision::Int4.is_implemented());
    }

    #[test]
    fn kv_cache_rejects_int8_precision() {
        let result = KvCache::new(KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            precision: KvPrecision::Int8,
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, RuntimeError::Unsupported(_)),
            "expected Unsupported error, got: {err:?}"
        );
    }

    #[test]
    fn kv_cache_rejects_int4_precision() {
        let result = KvCache::new(KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            precision: KvPrecision::Int4,
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, RuntimeError::Unsupported(_)),
            "expected Unsupported error, got: {err:?}"
        );
    }

    // ---- F16 KV cache tests --------------------------------------------------

    fn test_config_f16() -> KvCacheConfig {
        KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            precision: KvPrecision::F16,
        }
    }

    #[test]
    fn kv_cache_f16_creates_successfully() {
        let kv = KvCache::new(test_config_f16()).unwrap();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.config().num_layers, 2);
    }

    #[test]
    fn kv_cache_f16_append_and_read_keys() {
        let mut kv = KvCache::new(test_config_f16()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        // 2 heads × 4 dim = 8 floats
        let data = [1.5f32, -2.0, 3.25, 0.0, 7.0, -0.5, 100.0, 42.0];
        view.append_keys_f16_from_f32(&data);

        // Head 0 at byte 0, head 1 at byte head1_base
        let mut buf = [0.0f32; 4];
        view.read_keys_f16_to_f32_into(&mut buf, 0, 4);
        for (i, &expected) in data[0..4].iter().enumerate() {
            assert!(
                (buf[i] - expected).abs() < 0.01,
                "head0 key[{i}]: expected {expected}, got {}", buf[i]
            );
        }

        // Head 1: element offset = 1 * 16 * 4 = 64
        let h1 = 16 * 4;
        view.read_keys_f16_to_f32_into(&mut buf, h1, 4);
        for (i, &expected) in data[4..8].iter().enumerate() {
            assert!(
                (buf[i] - expected).abs() < 0.01,
                "head1 key[{i}]: expected {expected}, got {}", buf[i]
            );
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_f16_append_and_read_values() {
        let mut kv = KvCache::new(test_config_f16()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [0.1f32, 100.0, -0.001, 0.5, 1.0, 2.0, 3.0, 4.0];
        view.append_values_f16_from_f32(&data);

        let mut buf = [0.0f32; 4];
        view.read_values_f16_to_f32_into(&mut buf, 0, 4);
        for (i, &expected) in data[0..4].iter().enumerate() {
            let tol = expected.abs() * 0.01 + 0.001;
            assert!(
                (buf[i] - expected).abs() < tol,
                "head0 value[{i}]: expected {expected}, got {}", buf[i]
            );
        }

        let h1 = 16 * 4;
        view.read_values_f16_to_f32_into(&mut buf, h1, 4);
        for (i, &expected) in data[4..8].iter().enumerate() {
            let tol = expected.abs() * 0.01 + 0.001;
            assert!(
                (buf[i] - expected).abs() < tol,
                "head1 value[{i}]: expected {expected}, got {}", buf[i]
            );
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_f16_auto_dispatch_append() {
        let mut kv = KvCache::new(test_config_f16()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        view.append_keys(&data);
        view.append_values(&data);

        let mut kbuf = [0.0f32; 4];
        let mut vbuf = [0.0f32; 4];
        view.read_keys_into(&mut kbuf, 0, 4);
        view.read_values_into(&mut vbuf, 0, 4);
        for i in 0..4 {
            assert!(
                (kbuf[i] - data[i]).abs() < 0.01,
                "key[{i}] mismatch: got {}, expected {}",
                kbuf[i], data[i]
            );
            assert!(
                (vbuf[i] - data[i]).abs() < 0.01,
                "value[{i}] mismatch: got {}, expected {}",
                vbuf[i], data[i]
            );
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_f16_half_memory_of_f32() {
        // F16 pre-fill size = 2 * 4 * 2 * 16 = 256 bytes/layer
        let kv_f16 = KvCache::new(test_config_f16()).unwrap();
        assert_eq!(kv_f16.keys[0].len(), 256);

        // F32 pre-fill size = 2 * 4 * 4 * 16 = 512 bytes/layer
        let kv_f32 = KvCache::new(test_config()).unwrap();
        assert_eq!(kv_f32.keys[0].len(), 512);

        assert_eq!(kv_f32.keys[0].len(), kv_f16.keys[0].len() * 2);
    }

    #[test]
    fn kv_cache_f16_multiple_positions() {
        let mut kv = KvCache::new(test_config_f16()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        // Position 0
        let data0 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        view.append_keys_f16_from_f32(&data0);
        view.seq_len = 1;

        // Position 1
        let data1 = [11.0f32, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        view.append_keys_f16_from_f32(&data1);

        // Head 0: pos 0 at element 0..4, pos 1 at element 4..8
        let mut buf = [0.0f32; 4];
        view.read_keys_f16_to_f32_into(&mut buf, 0, 4);
        for i in 0..4 { assert!((buf[i] - data0[i]).abs() < 0.01); }
        view.read_keys_f16_to_f32_into(&mut buf, 4, 4);
        for i in 0..4 { assert!((buf[i] - data1[i]).abs() < 0.01); }

        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_f16_preallocation() {
        let cfg = test_config_f16();
        let kv = KvCache::new(cfg.clone()).unwrap();
        // Pre-filled: num_kv_heads * head_dim * bytes_per_element * max_seq_len
        // = 2 * 4 * 2 * 16 = 256 bytes per layer
        assert_eq!(kv.keys[0].len(), 256);
    }

    // ---- Head-first layout verification tests --------------------------------

    #[test]
    fn kv_cache_head_first_layout_contiguous_per_head() {
        // Verify that positions for the same head are contiguous in memory
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        // Write 3 positions
        for pos in 0..3u32 {
            let base = (pos + 1) as f32 * 10.0;
            let data: Vec<f32> = (0..8).map(|i| base + i as f32).collect();
            view.append_keys_f32(&data);
            view.seq_len = pos as usize + 1;
        }

        // Head 0 should be contiguous: pos0[0..4], pos1[0..4], pos2[0..4]
        let slice = view.keys_f32_slice(0, 12); // 3 positions × 4 elements
        assert_eq!(slice[0], 10.0);  // pos0, head0, e0
        assert_eq!(slice[4], 20.0);  // pos1, head0, e0
        assert_eq!(slice[8], 30.0);  // pos2, head0, e0

        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_head_pos_byte_offset() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let view = kv.view_mut(0).unwrap();
        // head=0, pos=0 → 0
        assert_eq!(view.head_pos_byte_offset(0, 0), 0);
        // head=0, pos=1 → head_dim * bpe = 4 * 4 = 16
        assert_eq!(view.head_pos_byte_offset(0, 1), 16);
        // head=1, pos=0 → max_seq_len * head_dim * bpe = 16 * 4 * 4 = 256
        assert_eq!(view.head_pos_byte_offset(1, 0), 256);
        // head=1, pos=3 → 256 + 3 * 16 = 304
        assert_eq!(view.head_pos_byte_offset(1, 3), 304);
        kv.commit_view(view).unwrap();
    }
}
