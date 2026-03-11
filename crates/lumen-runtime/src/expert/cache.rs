//! LFU (Least Frequently Used) expert cache for MoE streaming.
//!
//! Holds the hottest experts permanently resident in RAM to avoid SSD round-trips.
//! Research shows LFU achieves 84.6% speedup over LRU for expert caching (FlashMoE, 2025).
//!
//! Implementation uses the classic O(1) LFU algorithm:
//! - `HashMap<ExpertKey, CacheEntry>` for O(1) key lookup
//! - `BTreeMap<u64, HashSet<ExpertKey>>` for O(1) eviction of least-frequent entries
//! - `min_freq` tracked explicitly so eviction never scans

use lumen_format::ExpertSlice;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// Expert identifier: (layer_index, expert_index).
pub type ExpertKey = (usize, u32);

/// An LFU (Least Frequently Used) cache for MoE expert weights.
/// Holds the hot experts permanently resident in RAM.
pub struct ExpertLfuCache {
    /// Maximum number of experts to cache.
    capacity: usize,
    /// Key -> cache entry mapping for O(1) lookup.
    entries: HashMap<ExpertKey, CacheEntry>,
    /// Frequency -> set of keys at that frequency, for O(1) eviction.
    freq_map: BTreeMap<u64, HashSet<ExpertKey>>,
    /// Current minimum frequency in the cache (for O(1) eviction).
    min_freq: u64,
    /// Running cache statistics.
    total_hits: u64,
    total_misses: u64,
}

/// Internal cache entry holding expert weight data and metadata.
struct CacheEntry {
    /// Raw weight bytes (gate, up, down concatenated as stored in LBC blob).
    data: Arc<Vec<u8>>,
    /// Offsets within `data` for gate/up/down projection weights.
    slices: ExpertSlice,
    /// Current access frequency (number of get() calls).
    freq: u64,
    /// Size of the weight data in bytes.
    size_bytes: usize,
}

/// Cache statistics snapshot.
pub struct CacheStats {
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
    pub cached_experts: usize,
    pub cached_bytes: usize,
    pub capacity: usize,
}

impl ExpertLfuCache {
    /// Create a new LFU cache with the given maximum expert capacity.
    ///
    /// `capacity` is the maximum number of experts (not bytes) to hold.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            freq_map: BTreeMap::new(),
            min_freq: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Insert expert weights into the cache. Evicts the LFU entry if at capacity.
    ///
    /// Returns the evicted key if eviction occurred.
    /// If the key already exists, the existing entry is replaced.
    pub fn insert(
        &mut self,
        key: ExpertKey,
        data: Vec<u8>,
        slices: ExpertSlice,
    ) -> Option<ExpertKey> {
        if self.capacity == 0 {
            return None;
        }

        // If key already exists, update it in place.
        if self.entries.contains_key(&key) {
            self.remove_from_freq_map(&key);
            let size_bytes = data.len();
            let entry = CacheEntry {
                data: Arc::new(data),
                slices,
                freq: 1,
                size_bytes,
            };
            self.entries.insert(key, entry);
            self.add_to_freq_map(1, key);
            self.min_freq = 1;
            return None;
        }

        // Evict if at capacity.
        let evicted = if self.entries.len() >= self.capacity {
            self.evict_lfu()
        } else {
            None
        };

        // Insert new entry with freq=1.
        let size_bytes = data.len();
        let entry = CacheEntry {
            data: Arc::new(data),
            slices,
            freq: 1,
            size_bytes,
        };
        self.entries.insert(key, entry);
        self.add_to_freq_map(1, key);
        self.min_freq = 1;

        evicted
    }

    /// Look up expert weights. Increments frequency on hit.
    /// Returns None on miss (caller must load from SSD and call insert).
    pub fn get(&mut self, key: &ExpertKey) -> Option<Arc<Vec<u8>>> {
        if !self.entries.contains_key(key) {
            self.total_misses += 1;
            return None;
        }

        self.total_hits += 1;

        // Move from old frequency bucket to new.
        let old_freq = self.entries[key].freq;
        let new_freq = old_freq + 1;

        self.remove_from_freq_map_at(old_freq, key);
        self.add_to_freq_map(new_freq, *key);

        // Update min_freq if we just emptied the min bucket.
        if old_freq == self.min_freq && self.freq_bucket_empty(old_freq) {
            self.min_freq = new_freq;
        }

        let entry = self.entries.get_mut(key).unwrap();
        entry.freq = new_freq;

        Some(Arc::clone(&entry.data))
    }

    /// Check if expert is cached without updating frequency.
    pub fn contains(&self, key: &ExpertKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Pre-populate cache with known hot experts from profiler output.
    ///
    /// `hot_experts[layer] = Vec<expert_id>` -- sorted by frequency descending.
    /// The `loader` function is called to fetch the raw bytes for each expert.
    /// Experts are inserted in frequency order (most frequent first), so if
    /// capacity is limited, the hottest experts are retained.
    pub fn warm_from_profile(
        &mut self,
        hot_experts: &[Vec<u32>],
        loader: impl Fn(ExpertKey) -> (Vec<u8>, ExpertSlice),
    ) {
        for (layer, experts) in hot_experts.iter().enumerate() {
            for &expert_id in experts {
                if self.entries.len() >= self.capacity {
                    return;
                }
                let key = (layer, expert_id);
                if !self.entries.contains_key(&key) {
                    let (data, slices) = loader(key);
                    self.insert(key, data, slices);
                }
            }
        }
    }

    /// Cache hit rate since creation (or since last manual reset of stats).
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.0;
        }
        self.total_hits as f64 / total as f64
    }

    /// Current number of cached experts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes held in cache.
    pub fn cached_bytes(&self) -> usize {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_rate: self.hit_rate(),
            cached_experts: self.entries.len(),
            cached_bytes: self.cached_bytes(),
            capacity: self.capacity,
        }
    }

    /// Get expert data AND slices together, updating frequency (like `get`).
    /// Returns `(data_bytes, expert_slice)` for assembling cache-based Metal buffers.
    pub fn get_with_slices(&mut self, key: &ExpertKey) -> Option<(Arc<Vec<u8>>, ExpertSlice)> {
        if !self.entries.contains_key(key) {
            self.total_misses += 1;
            return None;
        }

        self.total_hits += 1;

        let old_freq = self.entries[key].freq;
        let new_freq = old_freq + 1;

        self.remove_from_freq_map_at(old_freq, key);
        self.add_to_freq_map(new_freq, *key);

        if old_freq == self.min_freq && self.freq_bucket_empty(old_freq) {
            self.min_freq = new_freq;
        }

        let entry = self.entries.get_mut(key).unwrap();
        entry.freq = new_freq;

        Some((Arc::clone(&entry.data), entry.slices.clone()))
    }

    /// Get the ExpertSlice for a cached expert (for Metal buffer setup).
    pub fn get_slices(&self, key: &ExpertKey) -> Option<&ExpertSlice> {
        self.entries.get(key).map(|e| &e.slices)
    }

    // ---- Internal helpers ----

    /// Evict the least-frequently-used entry. Returns the evicted key.
    fn evict_lfu(&mut self) -> Option<ExpertKey> {
        if self.entries.is_empty() {
            return None;
        }

        // Get any key from the min_freq bucket.
        let evict_key = {
            let bucket = self.freq_map.get(&self.min_freq)?;
            // Pick an arbitrary element from the set.
            let key = *bucket.iter().next()?;
            key
        };

        self.remove_from_freq_map_at(self.min_freq, &evict_key);
        self.entries.remove(&evict_key);

        Some(evict_key)
    }

    /// Add a key to the frequency bucket.
    fn add_to_freq_map(&mut self, freq: u64, key: ExpertKey) {
        self.freq_map.entry(freq).or_default().insert(key);
    }

    /// Remove a key from its current frequency bucket (looks up freq from entry).
    fn remove_from_freq_map(&mut self, key: &ExpertKey) {
        if let Some(entry) = self.entries.get(key) {
            let freq = entry.freq;
            self.remove_from_freq_map_at(freq, key);
        }
    }

    /// Remove a key from a specific frequency bucket.
    fn remove_from_freq_map_at(&mut self, freq: u64, key: &ExpertKey) {
        if let Some(bucket) = self.freq_map.get_mut(&freq) {
            bucket.remove(key);
            if bucket.is_empty() {
                self.freq_map.remove(&freq);
            }
        }
    }

    /// Check if a frequency bucket is empty (or absent).
    fn freq_bucket_empty(&self, freq: u64) -> bool {
        self.freq_map
            .get(&freq)
            .map_or(true, |bucket| bucket.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_format::index::TensorSlice;
    use lumen_format::quantization::QuantScheme;

    /// Helper to create a dummy ExpertSlice with given offsets.
    fn dummy_slices(gate_len: u64, up_len: u64, down_len: u64) -> ExpertSlice {
        let make = |off: u64, len: u64| TensorSlice {
            offset: off,
            length: len,
            quant: QuantScheme::F32,
        };
        ExpertSlice {
            gate: make(0, gate_len),
            up: make(gate_len, up_len),
            down: make(gate_len + up_len, down_len),
        }
    }

    /// Helper to create dummy expert data of given size.
    fn dummy_data(size: usize) -> Vec<u8> {
        vec![42u8; size]
    }

    #[test]
    fn test_lfu_basic_insert_get() {
        let mut cache = ExpertLfuCache::new(10);
        let slices = dummy_slices(100, 100, 100);

        // Insert 3 experts.
        cache.insert((0, 0), dummy_data(300), slices.clone());
        cache.insert((0, 1), dummy_data(300), slices.clone());
        cache.insert((1, 0), dummy_data(300), slices.clone());

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.cached_bytes(), 900);

        // Get all 3 -> all hits.
        assert!(cache.get(&(0, 0)).is_some());
        assert!(cache.get(&(0, 1)).is_some());
        assert!(cache.get(&(1, 0)).is_some());

        assert!((cache.hit_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = ExpertLfuCache::new(3);
        let slices = dummy_slices(100, 100, 100);

        // Fill to capacity.
        cache.insert((0, 0), dummy_data(300), slices.clone());
        cache.insert((0, 1), dummy_data(300), slices.clone());
        cache.insert((0, 2), dummy_data(300), slices.clone());

        assert_eq!(cache.len(), 3);

        // Insert a 4th expert -> one must be evicted.
        let evicted = cache.insert((0, 3), dummy_data(300), slices.clone());
        assert!(evicted.is_some());
        assert_eq!(cache.len(), 3);

        // The evicted key should be one of the original three (all at freq=1).
        let evicted_key = evicted.unwrap();
        assert!(!cache.contains(&evicted_key));
    }

    #[test]
    fn test_lfu_frequency_ordering() {
        let mut cache = ExpertLfuCache::new(2);
        let slices = dummy_slices(100, 100, 100);

        // Insert A and B.
        cache.insert((0, 0), dummy_data(300), slices.clone()); // A
        cache.insert((0, 1), dummy_data(300), slices.clone()); // B

        // Access A 10 times to raise its frequency.
        for _ in 0..10 {
            cache.get(&(0, 0));
        }

        // Access B only once (already at freq=1 from insert, now freq=2).
        cache.get(&(0, 1));

        // Insert C -> B should be evicted (freq=2) not A (freq=11).
        let evicted = cache.insert((0, 2), dummy_data(300), slices.clone());
        assert_eq!(evicted, Some((0, 1)));

        // A should still be present.
        assert!(cache.contains(&(0, 0)));
        // C should be present.
        assert!(cache.contains(&(0, 2)));
        // B should be gone.
        assert!(!cache.contains(&(0, 1)));
    }

    #[test]
    fn test_lfu_hit_rate() {
        let mut cache = ExpertLfuCache::new(10);
        let slices = dummy_slices(100, 100, 100);

        cache.insert((0, 0), dummy_data(300), slices.clone());

        // 9 hits.
        for _ in 0..9 {
            cache.get(&(0, 0));
        }
        // 1 miss.
        cache.get(&(0, 1));

        let rate = cache.hit_rate();
        assert!(
            (rate - 0.9).abs() < 1e-10,
            "Expected hit rate 0.9, got {rate}"
        );
    }

    #[test]
    fn test_lfu_warm_from_profile() {
        let mut cache = ExpertLfuCache::new(10);

        // Simulate profiler output: layer 0 has experts [0, 3, 5], layer 1 has [1, 2].
        let hot_experts = vec![vec![0, 3, 5], vec![1, 2]];

        cache.warm_from_profile(&hot_experts, |(layer, expert_id)| {
            let data = vec![(layer as u8) * 10 + (expert_id as u8); 300];
            let slices = dummy_slices(100, 100, 100);
            (data, slices)
        });

        // All 5 experts should be cached.
        assert_eq!(cache.len(), 5);
        assert!(cache.contains(&(0, 0)));
        assert!(cache.contains(&(0, 3)));
        assert!(cache.contains(&(0, 5)));
        assert!(cache.contains(&(1, 1)));
        assert!(cache.contains(&(1, 2)));
    }

    #[test]
    fn test_lfu_warm_from_profile_capacity_limit() {
        let mut cache = ExpertLfuCache::new(3);

        // Try to warm with 5 experts but capacity is only 3.
        let hot_experts = vec![vec![0, 1, 2], vec![3, 4]];

        cache.warm_from_profile(&hot_experts, |(_layer, _eid)| {
            (dummy_data(100), dummy_slices(30, 30, 40))
        });

        // Only first 3 should fit (layer 0 experts 0,1,2).
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&(0, 0)));
        assert!(cache.contains(&(0, 1)));
        assert!(cache.contains(&(0, 2)));
    }

    #[test]
    fn test_lfu_empty_cache() {
        let cache = ExpertLfuCache::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.cached_bytes(), 0);
        assert!((cache.hit_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lfu_zero_capacity() {
        let mut cache = ExpertLfuCache::new(0);
        let slices = dummy_slices(100, 100, 100);

        // Insert should not panic, but entry is not stored.
        let evicted = cache.insert((0, 0), dummy_data(300), slices);
        assert!(evicted.is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lfu_stats() {
        let mut cache = ExpertLfuCache::new(5);
        let slices = dummy_slices(100, 100, 100);

        cache.insert((0, 0), dummy_data(300), slices.clone());
        cache.insert((0, 1), dummy_data(200), slices.clone());

        // 2 hits, 1 miss.
        cache.get(&(0, 0));
        cache.get(&(0, 1));
        cache.get(&(0, 2)); // miss

        let stats = cache.stats();
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.total_misses, 1);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 1e-10);
        assert_eq!(stats.cached_experts, 2);
        assert_eq!(stats.cached_bytes, 500);
        assert_eq!(stats.capacity, 5);
    }

    #[test]
    fn test_lfu_replace_existing() {
        let mut cache = ExpertLfuCache::new(5);
        let slices = dummy_slices(100, 100, 100);

        cache.insert((0, 0), dummy_data(300), slices.clone());
        assert_eq!(cache.cached_bytes(), 300);

        // Replace with different size data.
        let evicted = cache.insert((0, 0), dummy_data(500), slices.clone());
        assert!(evicted.is_none()); // No eviction, just replacement.
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.cached_bytes(), 500);
    }

    #[test]
    fn test_lfu_get_returns_correct_data() {
        let mut cache = ExpertLfuCache::new(5);
        let slices = dummy_slices(100, 100, 100);

        let data = vec![1, 2, 3, 4, 5];
        cache.insert((0, 0), data.clone(), slices);

        let retrieved = cache.get(&(0, 0)).unwrap();
        assert_eq!(&*retrieved, &data);
    }

    #[test]
    fn test_lfu_eviction_cascade() {
        // Test that min_freq is correctly maintained through multiple evictions.
        let mut cache = ExpertLfuCache::new(2);
        let slices = dummy_slices(10, 10, 10);

        // Insert A and B.
        cache.insert((0, 0), dummy_data(30), slices.clone()); // A: freq=1
        cache.insert((0, 1), dummy_data(30), slices.clone()); // B: freq=1

        // Access A to freq=2.
        cache.get(&(0, 0));

        // Insert C -> evicts B (freq=1).
        let evicted1 = cache.insert((0, 2), dummy_data(30), slices.clone());
        assert_eq!(evicted1, Some((0, 1)));

        // Insert D -> evicts C (freq=1) because A is at freq=2.
        let evicted2 = cache.insert((0, 3), dummy_data(30), slices.clone());
        assert_eq!(evicted2, Some((0, 2)));

        // A should still be present.
        assert!(cache.contains(&(0, 0)));
        assert!(cache.contains(&(0, 3)));
    }
}
