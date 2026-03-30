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

    /// Generate n elements as F16 (IEEE 754 half-precision) bytes.
    /// Each element is 2 bytes (little-endian u16 f16 bits).
    /// Returns raw bytes only (no f32 reference needed at format level).
    pub fn gen_f16_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let v = self.next_f32();
            let bits = f32_to_f16_bits(v);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    /// Generate n elements as Q4_0 bytes.
    /// Q4_0 block layout: [2 bytes f16 scale] [16 bytes packed nibbles] = 18 bytes per 32 elements.
    /// n MUST be a multiple of 32.
    pub fn gen_q4_0_bytes(&mut self, n: usize) -> Vec<u8> {
        assert!(n % 32 == 0, "gen_q4_0_bytes: n={n} must be a multiple of 32");
        let num_blocks = n / 32;
        let mut out = Vec::with_capacity(num_blocks * 18);
        for _ in 0..num_blocks {
            // Generate 32 f32 values for this block
            let mut vals = [0.0f32; 32];
            for v in &mut vals {
                *v = self.next_f32();
            }
            // Find max absolute value for scale
            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0; // Q4_0 range is [-8, 7], we use 7 as max positive
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write f16 scale
            let scale_f16 = f32_to_f16_bits(scale);
            out.extend_from_slice(&scale_f16.to_le_bytes());

            // Quantize and pack nibbles (GGML de-interleaved order):
            // byte[i] = lo_nibble(vals[i]) | hi_nibble(vals[i+16]) << 4
            let mut packed = [0u8; 16];
            for i in 0..16 {
                let lo_val = (vals[i] * inv_scale + 8.5).clamp(0.0, 15.0) as u8;
                let hi_val = (vals[i + 16] * inv_scale + 8.5).clamp(0.0, 15.0) as u8;
                packed[i] = (lo_val & 0x0F) | ((hi_val & 0x0F) << 4);
            }
            out.extend_from_slice(&packed);
        }
        out
    }

    /// Generate n elements as Q8_0 bytes.
    /// Q8_0 block layout: [2 bytes f16 scale] [32 bytes int8 values] = 34 bytes per 32 elements.
    /// n MUST be a multiple of 32.
    pub fn gen_q8_0_bytes(&mut self, n: usize) -> Vec<u8> {
        assert!(n % 32 == 0, "gen_q8_0_bytes: n={n} must be a multiple of 32");
        let num_blocks = n / 32;
        let mut out = Vec::with_capacity(num_blocks * 34);
        for _ in 0..num_blocks {
            let mut vals = [0.0f32; 32];
            for v in &mut vals {
                *v = self.next_f32();
            }
            let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            let scale_f16 = f32_to_f16_bits(scale);
            out.extend_from_slice(&scale_f16.to_le_bytes());

            for &v in &vals {
                let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
        out
    }
}

/// Convert f32 to f16 bits (truncation, no rounding -- matches common quantization practice).
pub(crate) fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        let f16_frac = if frac != 0 { 0x200 } else { 0 }; // preserve NaN vs Inf
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow -> Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        // Underflow -> zero (no denormal handling for simplicity)
        return (sign << 15) as u16;
    }

    let f16_frac = frac >> 13; // truncate
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
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
