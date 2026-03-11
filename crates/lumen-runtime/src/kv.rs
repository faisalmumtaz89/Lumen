//! KV cache management types.
//!
//! The KV cache stores key and value projections from attention layers so
//! they can be reused across token generation steps. In v1, the KV cache
//! is RAM-resident. Future versions may support disk-backed KV (KVSwap-style).
//!
//! The runtime manages KV memory explicitly and provides knobs for precision,
//! eviction policy, and chunking (paged KV).

use crate::error::RuntimeError;

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
    /// Currently only F32 is supported. F16, Int8, and Int4 are defined
    /// for forward-compatible configuration but will be rejected at
    /// [`KvCache::new`] time until the corresponding KV storage paths are
    /// implemented.
    pub fn is_implemented(&self) -> bool {
        match self {
            Self::F32 => true,
            Self::F16 | Self::Int8 | Self::Int4 => false,
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
#[derive(Debug)]
pub struct KvCacheView {
    /// Layer index this view is for.
    pub layer_idx: usize,

    /// Key cache data. Shape: `[current_seq_len, num_kv_heads, head_dim]`.
    /// Stored as raw bytes in the configured precision.
    pub keys: Vec<u8>,

    /// Value cache data. Shape: `[current_seq_len, num_kv_heads, head_dim]`.
    pub values: Vec<u8>,

    /// Current sequence length (number of tokens with cached KV).
    pub seq_len: usize,

    /// Number of KV heads.
    pub num_kv_heads: usize,

    /// Dimension per head.
    pub head_dim: usize,

    /// Precision of the stored data.
    pub precision: KvPrecision,
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

    /// Append f32 key data as bytes.
    /// Uses unsafe memcpy on little-endian platforms for bulk copy.
    #[inline(always)]
    pub fn append_keys_f32(&mut self, data: &[f32]) {
        let byte_len = data.len() * 4;
        self.keys.reserve(byte_len);
        #[cfg(target_endian = "little")]
        {
            let old_len = self.keys.len();
            // SAFETY: reserve guarantees capacity >= old_len + byte_len,
            // data is contiguous f32 with LE byte repr on this platform.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.keys.as_mut_ptr().add(old_len),
                    byte_len,
                );
                self.keys.set_len(old_len + byte_len);
            }
        }
        #[cfg(target_endian = "big")]
        {
            for &v in data {
                self.keys.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// Append f32 value data as bytes.
    /// Uses unsafe memcpy on little-endian platforms for bulk copy.
    #[inline(always)]
    pub fn append_values_f32(&mut self, data: &[f32]) {
        let byte_len = data.len() * 4;
        self.values.reserve(byte_len);
        #[cfg(target_endian = "little")]
        {
            let old_len = self.values.len();
            // SAFETY: reserve guarantees capacity >= old_len + byte_len,
            // data is contiguous f32 with LE byte repr on this platform.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    self.values.as_mut_ptr().add(old_len),
                    byte_len,
                );
                self.values.set_len(old_len + byte_len);
            }
        }
        #[cfg(target_endian = "big")]
        {
            for &v in data {
                self.values.extend_from_slice(&v.to_le_bytes());
            }
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
    /// quantized format (Int8 or Int4) that is not yet implemented. All KV
    /// read/write methods currently assume F32 byte layout, so accepting an
    /// unimplemented precision would silently corrupt data.
    pub fn new(config: KvCacheConfig) -> Result<Self, RuntimeError> {
        if !config.precision.is_implemented() {
            return Err(RuntimeError::Unsupported(format!(
                "KV cache precision {:?} is not yet implemented",
                config.precision
            )));
        }
        let num_layers = config.num_layers;
        Ok(Self {
            config,
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
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

    /// Total bytes currently used across all layers.
    pub fn total_bytes(&self) -> u64 {
        self.keys
            .iter()
            .chain(self.values.iter())
            .map(|v| v.len() as u64)
            .sum()
    }

    /// Reset the cache (e.g., for a new generation session).
    pub fn reset(&mut self) {
        for k in &mut self.keys {
            k.clear();
        }
        for v in &mut self.values {
            v.clear();
        }
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
    fn kv_cache_view_mut_and_commit() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();
        assert_eq!(view.layer_idx, 0);
        assert_eq!(view.seq_len, 0);

        // Simulate writing some data
        view.keys.extend_from_slice(&[1u8; 32]);
        view.values.extend_from_slice(&[2u8; 32]);

        kv.commit_view(view).unwrap();

        // Data should be back in cache
        assert_eq!(kv.total_bytes(), 64);
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
        // Write some data
        let mut view = kv.view_mut(0).unwrap();
        view.keys.extend_from_slice(&[1u8; 16]);
        kv.commit_view(view).unwrap();
        kv.advance_seq_len().unwrap();

        assert!(kv.seq_len() > 0);
        assert!(kv.total_bytes() > 0);

        kv.reset();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.total_bytes(), 0);
    }

    #[test]
    fn kv_cache_total_bytes_reflects_data() {
        let mut kv = KvCache::new(test_config()).unwrap();
        // Add data to both layers
        for layer in 0..2 {
            let mut view = kv.view_mut(layer).unwrap();
            view.keys.extend_from_slice(&[0u8; 10]);
            view.values.extend_from_slice(&[0u8; 10]);
            kv.commit_view(view).unwrap();
        }
        // 2 layers * (10 keys + 10 values) = 40
        assert_eq!(kv.total_bytes(), 40);
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

        let data = [1.5f32, -2.0, 3.25, 0.0];
        view.append_keys_f32(&data);
        assert_eq!(view.keys.len(), 16); // 4 floats * 4 bytes

        for (i, &expected) in data.iter().enumerate() {
            let got = view.read_key_f32(i);
            assert_eq!(got, expected, "key[{i}]: expected {expected}, got {got}");
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_view_append_and_read_f32_values() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [0.1f32, 100.0, -0.001, f32::MAX];
        view.append_values_f32(&data);
        assert_eq!(view.values.len(), 16);

        for (i, &expected) in data.iter().enumerate() {
            let got = view.read_value_f32(i);
            assert_eq!(got, expected, "value[{i}]: expected {expected}, got {got}");
        }
        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_cache_view_append_f32_multiple_batches() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        view.append_keys_f32(&[1.0, 2.0]);
        view.append_keys_f32(&[3.0, 4.0]);
        assert_eq!(view.keys.len(), 16); // 4 floats * 4 bytes

        assert_eq!(view.read_key_f32(0), 1.0);
        assert_eq!(view.read_key_f32(1), 2.0);
        assert_eq!(view.read_key_f32(2), 3.0);
        assert_eq!(view.read_key_f32(3), 4.0);
        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_keys_f32_slice_matches_read() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [1.5f32, -2.0, 3.25, 0.0, 7.0, -0.5, 100.0, 42.0];
        view.append_keys_f32(&data);

        // Full slice from start
        let slice = view.keys_f32_slice(0, 8);
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(slice[i], expected, "keys_f32_slice[{i}]: expected {expected}, got {}", slice[i]);
        }

        // Partial slice from offset (simulating head_dim=4, kv_h=1)
        let partial = view.keys_f32_slice(4, 4);
        assert_eq!(partial, &[7.0, -0.5, 100.0, 42.0]);

        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_values_f32_slice_matches_read() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        let data = [0.1f32, 100.0, -0.001, f32::MAX];
        view.append_values_f32(&data);

        let slice = view.values_f32_slice(0, 4);
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(slice[i], expected, "values_f32_slice[{i}]: expected {expected}, got {}", slice[i]);
        }

        // Partial slice
        let partial = view.values_f32_slice(2, 2);
        assert_eq!(partial, &[-0.001, f32::MAX]);

        kv.commit_view(view).unwrap();
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn kv_cache_view_f32_slice_multiple_appends() {
        let mut kv = KvCache::new(test_config()).unwrap();
        let mut view = kv.view_mut(0).unwrap();

        view.append_keys_f32(&[1.0, 2.0]);
        view.append_keys_f32(&[3.0, 4.0]);

        let slice = view.keys_f32_slice(0, 4);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);

        // Verify slice at offset matches read_keys_f32_into
        let slice_offset = view.keys_f32_slice(2, 2);
        let mut buf = [0.0f32; 2];
        view.read_keys_f32_into(&mut buf, 2, 2);
        assert_eq!(slice_offset, &buf, "f32_slice must match read_f32_into");

        kv.commit_view(view).unwrap();
    }

    #[test]
    fn kv_precision_is_implemented() {
        assert!(KvPrecision::F32.is_implemented());
        assert!(!KvPrecision::F16.is_implemented());
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
}
