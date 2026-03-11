//! Deterministic pseudo-random generator for reproducible weight generation.

/// Deterministic pseudo-random f32 generator for reproducible test weights.
///
/// Uses xorshift64 — simple, fast, and zero deps.
pub(crate) struct WeightRng {
    state: u64,
}

impl WeightRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a small f32 in [-0.1, 0.1] — small enough to not blow up.
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        let frac = (u >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        (frac - 0.5) * 0.2 // [-0.1, 0.1)
    }

    /// Generate n f32 values as bytes.
    pub fn gen_f32_bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| self.next_f32().to_le_bytes())
            .collect()
    }

    /// Generate n f32 values centered around 1.0 (for norm weights).
    pub fn gen_norm_bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (1.0 + self.next_f32() * 0.01).to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut rng1 = WeightRng::new(42);
        let mut rng2 = WeightRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn zero_seed_uses_fallback() {
        let mut rng = WeightRng::new(0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn f32_range() {
        let mut rng = WeightRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= -0.1 && v < 0.1, "got {v}");
        }
    }
}
