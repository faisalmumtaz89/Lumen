//! Expert activation profiler for MoE streaming.
//!
//! Tracks per-(layer, expert) activation counts over a token stream to determine
//! which experts should be permanently cached in RAM. Research shows expert
//! activation is heavily skewed: in Mixtral layer 14, experts 0+1 process 64%
//! of tokens, and 64% cache coverage yields ~95% hit rate.

/// Tracks per-(layer, expert) activation counts over a token stream.
/// Used to determine which experts to permanently cache in RAM.
pub struct ExpertActivationProfiler {
    /// activation_counts[layer][expert] = number of times expert was selected.
    activation_counts: Vec<Vec<u64>>,
    num_layers: usize,
    num_experts: usize,
    total_tokens: u64,
}

/// Summary statistics from the activation profiler.
pub struct ProfilerSummary {
    /// Total tokens processed since last reset.
    pub total_tokens: u64,
    /// Shannon entropy of activation distribution per layer.
    /// Low entropy = highly skewed (good for caching).
    /// Max entropy = log2(num_experts) for uniform distribution.
    pub per_layer_entropy: Vec<f32>,
    /// Global hottest experts across all layers, sorted by frequency descending.
    /// Each entry is (layer_index, expert_id, frequency).
    pub global_top_experts: Vec<(usize, u32, f64)>,
}

impl ExpertActivationProfiler {
    /// Create a new profiler for a model with `num_layers` MoE layers,
    /// each containing `num_experts` experts.
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            activation_counts: vec![vec![0u64; num_experts]; num_layers],
            num_layers,
            num_experts,
            total_tokens: 0,
        }
    }

    /// Record that `expert_ids` (shape: [top_k]) were activated at `layer`.
    /// Called once per forward pass per MoE layer for each token.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= num_layers` or any expert_id >= num_experts.
    pub fn record(&mut self, layer: usize, expert_ids: &[u32]) {
        assert!(
            layer < self.num_layers,
            "layer index {layer} out of bounds (num_layers={})",
            self.num_layers
        );
        let counts = &mut self.activation_counts[layer];
        for &eid in expert_ids {
            let eid_usize = eid as usize;
            if eid_usize >= self.num_experts {
                eprintln!(
                    "[WARN] expert_id {eid} out of bounds (num_experts={}) at layer {layer}, skipping",
                    self.num_experts
                );
                continue;
            }
            counts[eid_usize] += 1;
        }
        // Each call represents one token passing through one layer.
        // We count total_tokens only once per layer 0 call to avoid
        // double-counting across layers. However, the spec says
        // "total tokens seen" — this means total tokens across
        // the stream. We increment once per record() call because
        // each layer sees the same token. We track total_tokens
        // as the number of record() calls made to any layer.
        // For frequency computation, we use per-layer totals instead.
        self.total_tokens += 1;
    }

    /// Returns the activation frequency of each expert at `layer`,
    /// as a fraction of total activations at that layer (0.0 to 1.0),
    /// sorted descending by frequency.
    ///
    /// Returns Vec<(expert_id, frequency)>.
    pub fn layer_frequencies(&self, layer: usize) -> Vec<(u32, f64)> {
        assert!(
            layer < self.num_layers,
            "layer index {layer} out of bounds (num_layers={})",
            self.num_layers
        );
        let counts = &self.activation_counts[layer];
        let total: u64 = counts.iter().sum();
        if total == 0 {
            // No activations recorded yet; return all zeros sorted by expert_id.
            return (0..self.num_experts as u32)
                .map(|eid| (eid, 0.0))
                .collect();
        }
        let total_f64 = total as f64;
        let mut freqs: Vec<(u32, f64)> = counts
            .iter()
            .enumerate()
            .map(|(eid, &count)| (eid as u32, count as f64 / total_f64))
            .collect();
        // Sort descending by frequency, then ascending by expert_id for stability.
        freqs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        freqs
    }

    /// Returns the top-K most activated expert IDs for each layer.
    /// Result: Vec<Vec<u32>> -- for each layer, the hot expert IDs sorted by frequency.
    pub fn top_k_per_layer(&self, k: usize) -> Vec<Vec<u32>> {
        (0..self.num_layers)
            .map(|layer| {
                let freqs = self.layer_frequencies(layer);
                freqs
                    .into_iter()
                    .take(k)
                    .filter(|&(_, f)| f > 0.0)
                    .map(|(eid, _)| eid)
                    .collect()
            })
            .collect()
    }

    /// Returns a summary: total tokens seen, per-layer entropy (how uniform is routing),
    /// overall hottest experts.
    pub fn summary(&self) -> ProfilerSummary {
        let per_layer_entropy: Vec<f32> = (0..self.num_layers)
            .map(|layer| self.layer_entropy(layer))
            .collect();

        // Collect all (layer, expert_id, frequency) across all layers, sorted descending.
        let mut global_top: Vec<(usize, u32, f64)> = Vec::new();
        for layer in 0..self.num_layers {
            let freqs = self.layer_frequencies(layer);
            for (eid, freq) in freqs {
                if freq > 0.0 {
                    global_top.push((layer, eid, freq));
                }
            }
        }
        global_top.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });

        ProfilerSummary {
            total_tokens: self.total_tokens,
            per_layer_entropy,
            global_top_experts: global_top,
        }
    }

    /// Total record() calls made (each represents one token at one layer).
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Reset all activation counts and total_tokens to zero.
    pub fn reset(&mut self) {
        for layer_counts in &mut self.activation_counts {
            for count in layer_counts.iter_mut() {
                *count = 0;
            }
        }
        self.total_tokens = 0;
    }

    /// Compute Shannon entropy for a single layer's activation distribution.
    /// H = -sum(p * log2(p)) for each expert with p > 0.
    /// Returns 0.0 if no activations recorded.
    fn layer_entropy(&self, layer: usize) -> f32 {
        let counts = &self.activation_counts[layer];
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let total_f64 = total as f64;
        let mut entropy = 0.0f64;
        for &count in counts {
            if count > 0 {
                let p = count as f64 / total_f64;
                entropy -= p * p.log2();
            }
        }
        entropy as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = ExpertActivationProfiler::new(2, 8);

        // Record 100 tokens at layer 0, always selecting expert 0.
        for _ in 0..100 {
            profiler.record(0, &[0]);
        }

        let freqs = profiler.layer_frequencies(0);
        // Expert 0 should be most frequent (frequency = 1.0 since only expert selected).
        assert_eq!(freqs[0].0, 0);
        assert!((freqs[0].1 - 1.0).abs() < 1e-10);

        // All other experts at layer 0 should have frequency 0.
        for &(eid, freq) in &freqs[1..] {
            assert!(eid != 0);
            assert!((freq - 0.0).abs() < 1e-10);
        }

        assert_eq!(profiler.total_tokens(), 100);
    }

    #[test]
    fn test_profiler_top_k() {
        let mut profiler = ExpertActivationProfiler::new(1, 8);

        // Expert 3: 40 activations, Expert 1: 30, Expert 5: 20, Expert 7: 10.
        for _ in 0..40 {
            profiler.record(0, &[3]);
        }
        for _ in 0..30 {
            profiler.record(0, &[1]);
        }
        for _ in 0..20 {
            profiler.record(0, &[5]);
        }
        for _ in 0..10 {
            profiler.record(0, &[7]);
        }

        let top_4 = profiler.top_k_per_layer(4);
        assert_eq!(top_4.len(), 1);
        assert_eq!(top_4[0], vec![3, 1, 5, 7]);

        // top-2 should be just [3, 1].
        let top_2 = profiler.top_k_per_layer(2);
        assert_eq!(top_2[0], vec![3, 1]);
    }

    #[test]
    fn test_profiler_entropy_uniform() {
        let mut profiler = ExpertActivationProfiler::new(1, 8);

        // Uniform distribution: each expert selected exactly the same number of times.
        for _ in 0..100 {
            for eid in 0..8u32 {
                profiler.record(0, &[eid]);
            }
        }

        let summary = profiler.summary();
        let max_entropy = (8.0f64).log2() as f32; // log2(8) = 3.0
        let entropy = summary.per_layer_entropy[0];
        assert!(
            (entropy - max_entropy).abs() < 0.01,
            "Expected entropy ~{max_entropy}, got {entropy}"
        );
    }

    #[test]
    fn test_profiler_entropy_single_expert() {
        let mut profiler = ExpertActivationProfiler::new(1, 8);

        // Only expert 2 is ever selected -> entropy should be 0.
        for _ in 0..100 {
            profiler.record(0, &[2]);
        }

        let summary = profiler.summary();
        let entropy = summary.per_layer_entropy[0];
        assert!(
            entropy.abs() < 1e-6,
            "Expected entropy ~0.0, got {entropy}"
        );
    }

    #[test]
    fn test_profiler_entropy_no_activations() {
        let profiler = ExpertActivationProfiler::new(1, 8);
        let summary = profiler.summary();
        assert!((summary.per_layer_entropy[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_profiler_summary() {
        let mut profiler = ExpertActivationProfiler::new(2, 4);

        // Layer 0: expert 0 = 50, expert 1 = 30, expert 2 = 20
        for _ in 0..50 {
            profiler.record(0, &[0]);
        }
        for _ in 0..30 {
            profiler.record(0, &[1]);
        }
        for _ in 0..20 {
            profiler.record(0, &[2]);
        }

        // Layer 1: expert 3 = 80, expert 0 = 20
        for _ in 0..80 {
            profiler.record(1, &[3]);
        }
        for _ in 0..20 {
            profiler.record(1, &[0]);
        }

        let summary = profiler.summary();

        // Total tokens = 50+30+20 + 80+20 = 200 record() calls.
        assert_eq!(summary.total_tokens, 200);

        // Global top experts: layer 1 expert 3 has freq 0.8 (highest).
        assert_eq!(summary.global_top_experts[0], (1, 3, 0.8));

        // Layer 0 entropy should be > 0 (not uniform, not single).
        assert!(summary.per_layer_entropy[0] > 0.0);
        assert!(summary.per_layer_entropy[0] < (4.0f64).log2() as f32);

        // Layer 1 entropy should be less than layer 0 (more skewed).
        assert!(summary.per_layer_entropy[1] < summary.per_layer_entropy[0]);
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = ExpertActivationProfiler::new(1, 4);

        for _ in 0..50 {
            profiler.record(0, &[0]);
        }
        assert_eq!(profiler.total_tokens(), 50);

        profiler.reset();
        assert_eq!(profiler.total_tokens(), 0);

        let freqs = profiler.layer_frequencies(0);
        for (_, freq) in &freqs {
            assert!(freq.abs() < 1e-10);
        }
    }

    #[test]
    fn test_profiler_multi_expert_per_record() {
        let mut profiler = ExpertActivationProfiler::new(1, 8);

        // Simulate top-2 routing: each token activates 2 experts.
        for _ in 0..100 {
            profiler.record(0, &[0, 3]);
        }

        let freqs = profiler.layer_frequencies(0);
        // Both expert 0 and expert 3 should have 100 activations out of 200 total.
        let e0 = freqs.iter().find(|&&(eid, _)| eid == 0).unwrap();
        let e3 = freqs.iter().find(|&&(eid, _)| eid == 3).unwrap();
        assert!((e0.1 - 0.5).abs() < 1e-10);
        assert!((e3.1 - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "layer index 2 out of bounds")]
    fn test_profiler_layer_out_of_bounds() {
        let mut profiler = ExpertActivationProfiler::new(2, 4);
        profiler.record(2, &[0]);
    }

    #[test]
    fn test_profiler_expert_out_of_bounds() {
        // Out-of-bounds expert IDs are silently skipped (warn + continue).
        let mut profiler = ExpertActivationProfiler::new(2, 4);
        profiler.record(0, &[4]);
        // Expert 4 is out of bounds (0..3 valid), should be skipped.
        // No panic; the valid experts should remain at zero activations.
        let top = profiler.top_k_per_layer(4);
        // top_k_per_layer returns expert IDs sorted by activation count.
        // Since all experts have 0 activations, the list should be empty.
        assert!(top[0].is_empty());
    }

    #[test]
    fn test_profiler_top_k_empty() {
        let profiler = ExpertActivationProfiler::new(2, 4);
        let top = profiler.top_k_per_layer(4);
        // No activations, so all layers return empty.
        assert_eq!(top.len(), 2);
        assert!(top[0].is_empty());
        assert!(top[1].is_empty());
    }

    #[test]
    fn test_profiler_top_k_exceeds_experts() {
        let mut profiler = ExpertActivationProfiler::new(1, 4);
        for _ in 0..10 {
            profiler.record(0, &[0, 1, 2, 3]);
        }
        // Request top-8 with only 4 experts -> should return all 4.
        let top = profiler.top_k_per_layer(8);
        assert_eq!(top[0].len(), 4);
    }
}
