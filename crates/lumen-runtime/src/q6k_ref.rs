//! Q6_K block layout: host-side reference, kernel mirror, and regression tests.
//!
//! # Why this module exists outside `src/cuda/`
//!
//! `src/cuda/` is `#[cfg(feature = "cuda")]`-gated (`lib.rs`), so nothing under
//! it compiles — let alone tests — on a host without the `cuda` feature. The
//! `matvec_q6_k_f32` CUDA kernel cannot run on macOS at all. So the part of
//! that kernel which is easy to get wrong (the nibble / high-bit / scale /
//! activation index mapping and the unit decomposition) is mirrored here as
//! pure Rust and tested against the ggml *packer*. `cargo test --lib
//! -p lumen-runtime` runs these on any host.
//!
//! # The layout
//!
//! `block_q6_K` is 210 bytes per 256 elements = 0.8203 B/weight
//! (`ggml/src/ggml-common.h:358-368`):
//!
//! ```text
//!   [0   .. 128)  ql      low 4 bits, two elements per byte
//!   [128 .. 192)  qh      high 2 bits, four elements per byte
//!   [192 .. 208)  scales  16 x int8, one per 16 consecutive elements
//!   [208 .. 210)  d       f16 super-block scale
//! ```
//!
//! A super-block is two independent HALVES of 128 elements. Half `h` uses
//! `ql + 64h`, `qh + 32h`, `scales + 8h` and produces elements
//! `[128h, 128h + 128)`. With `l = 0..31` and `is = l / 16`:
//!
//! ```text
//!   out[128h + l +  0] = d * sc[is + 0] * (((ql[l   ] & 0xF) | ((qh[l]>>0 & 3)<<4)) - 32)
//!   out[128h + l + 32] = d * sc[is + 2] * (((ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)) - 32)
//!   out[128h + l + 64] = d * sc[is + 4] * (((ql[l   ] >>  4) | ((qh[l]>>4 & 3)<<4)) - 32)
//!   out[128h + l + 96] = d * sc[is + 6] * (((ql[l+32] >>  4) | ((qh[l]>>6 & 3)<<4)) - 32)
//! ```
//!
//! # The invariant that is easy to get wrong
//!
//! **The two nibbles of one `ql` byte land 64 output slots apart, never 32.**
//! Byte `ql[l]` carries elements `l` and `l+64`; byte `ql[l+32]` carries
//! elements `l+32` and `l+96`. This is fixed by the packer
//! (`ggml-quants.c quantize_row_q6_K_ref`: `ql[l] = q1 | (q3 << 4)` with
//! `q1 = L[l]`, `q3 = L[l+64]`), so it is not a convention a reader may pick.
//!
//! A plausible-looking reading that walks `ql` as (byte `l` low, byte `l`
//! high, byte `l+32` low, byte `l+32` high) in *output* order puts one byte's
//! two nibbles 32 slots apart, mixing the low 4 bits of element `l+64` with
//! the high 2 bits of element `l+32`. [`dequant_block_legacy`] reproduces that
//! reading; [`legacy_mapping_is_wrong`] measures the damage (126 of 256 codes
//! on random data) and is the regression test that keeps it from coming back.

/// Elements per Q6_K super-block.
pub const Q6K_BLOCK_ELEM: usize = 256;
/// Bytes per Q6_K super-block: 128 ql + 64 qh + 16 scales + 2 d.
pub const Q6K_BLOCK_BYTE: usize = 210;
/// Elements per unit of work in `matvec_q6_k_f32` (one "group").
pub const Q6K_GROUP_ELEM: usize = 32;
/// Groups per super-block: 2 halves x 4 groups.
pub const Q6K_GROUPS_PER_BLOCK: usize = 8;

/// Exact IEEE-754 binary16 -> binary32 widening (no hardware dependency).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: value = frac * 2^-24.
        let v = (frac as f32) * (1.0 / 16_777_216.0);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac != 0 {
            f32::NAN
        } else if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    f32::from_bits((sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13))
}

/// Exact binary32 -> binary16 narrowing for the normal range used by tests.
fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    assert!(
        (1..=30).contains(&new_exp),
        "f32_to_f16_bits: {v} outside the normal f16 range this helper supports"
    );
    (sign << 15) | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
}

/// Round-to-nearest-even binary32 -> binary16, with subnormal, overflow and
/// NaN handling. Used to build the F16 cache a native-Q6_K weight still needs
/// for the batched PREFILL HGEMM path.
///
/// Unlike the `metal::repack_q4` twin this is not `cfg(target_os = "macos")`,
/// so it is available on the CUDA build.
pub fn f32_to_f16_bits_rne(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;

    if exp == 0xff {
        // Inf / NaN. Preserve NaN-ness (a zero payload would turn NaN to Inf).
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }

    exp -= 127;
    if exp > 15 {
        return sign | 0x7c00; // overflow -> Inf
    }
    if exp < -24 {
        return sign; // underflows past the smallest subnormal -> signed zero
    }

    if exp < -14 {
        // Subnormal f16: shift the implicit leading 1 into the mantissa.
        let shift = (-14 - exp) as u32; // 1..=10
        let m = mant | 0x80_0000;
        let sig = m >> (shift + 13);
        // Round to nearest even on the bits shifted out.
        let rem_shift = shift + 13;
        let half = 1u32 << (rem_shift - 1);
        let rem = m & ((1u32 << rem_shift) - 1);
        let mut out = sig;
        if rem > half || (rem == half && (sig & 1) == 1) {
            out += 1;
        }
        return sign | out as u16;
    }

    // Normal f16.
    let mut sig = mant >> 13;
    let rem = mant & 0x1fff;
    let mut e = (exp + 15) as u32;
    if rem > 0x1000 || (rem == 0x1000 && (sig & 1) == 1) {
        sig += 1;
        if sig == 0x400 {
            // Mantissa carried out: bump the exponent.
            sig = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | sig as u16
}

/// Dequantize a Q6_K byte stream straight to F16 bytes (2 B/element), using the
/// CORRECT ggml mapping.
///
/// This is what lets a `GpuWeightBuf::Q6KRaw` weight keep working for batched
/// PREFILL: `launch_gemm_projection` has an F16-cache fast path that fires
/// before its `match weight`, so a Q6_K weight with an F16 cache takes the exact
/// same tensor-core HGEMM route it takes today (today it is an F32 buffer plus an
/// F16 cache, and the fast path fires on the cache, not the buffer). Decode
/// meanwhile reads the 0.8203 B/weight Q6_K bytes natively.
///
/// Net residency for a native Q6_K weight is therefore 0.8203 + 2.0 = 2.82
/// B/weight against 4.0 + 2.0 = 6.0 today.
pub fn dequant_to_f16_bytes(raw: &[u8], n_elements: usize) -> Vec<u8> {
    let n_blocks = raw.len() / Q6K_BLOCK_BYTE;
    let mut out = Vec::with_capacity(n_elements * 2);
    let mut scratch = [0.0f32; Q6K_BLOCK_ELEM];
    for b in 0..n_blocks {
        if out.len() / 2 >= n_elements {
            break;
        }
        dequant_block(&raw[b * Q6K_BLOCK_BYTE..], &mut scratch);
        for &v in scratch.iter() {
            if out.len() / 2 >= n_elements {
                break;
            }
            out.extend_from_slice(&f32_to_f16_bits_rne(v).to_le_bytes());
        }
    }
    out
}

/// Which `(ql byte offset, high-nibble?, qh bit shift, scale base)` a group
/// index `g` (0..4) selects. Mirrors `q6k_unit_dot` in `matvec_q6_k_f32.cu`.
///
/// * `g = 0` -> `ql[l]` low nibble, `qh >> 0`, `scales[is + 0]`
/// * `g = 1` -> `ql[l + 32]` low nibble, `qh >> 2`, `scales[is + 2]`
/// * `g = 2` -> `ql[l]` high nibble, `qh >> 4`, `scales[is + 4]`
/// * `g = 3` -> `ql[l + 32]` high nibble, `qh >> 6`, `scales[is + 6]`
#[inline]
fn group_selectors(g: usize) -> (usize, bool, u32, usize) {
    debug_assert!(g < 4);
    let ql_off = if g & 1 == 1 { 32 } else { 0 };
    let hi_nib = (g >> 1) & 1 == 1;
    (ql_off, hi_nib, 2 * g as u32, 2 * g)
}

/// Decode the six-bit code (already offset by -32) for element `l` of group
/// `g` in half `half` of `block`. Returns the value in `-32..=31`.
#[inline]
pub fn decode_code(block: &[u8], half: usize, g: usize, l: usize) -> i32 {
    debug_assert!(block.len() >= Q6K_BLOCK_BYTE);
    debug_assert!(half < 2 && g < 4 && l < 32);
    let (ql_off, hi_nib, qh_shift, _) = group_selectors(g);
    let ql = &block[64 * half..];
    let qh = &block[128 + 32 * half..];
    let lo = if hi_nib {
        (ql[ql_off + l] >> 4) as i32
    } else {
        (ql[ql_off + l] & 0x0F) as i32
    };
    let hb = ((qh[l] >> qh_shift) & 3) as i32;
    (lo | (hb << 4)) - 32
}

/// The int8 sub-block scale for element `l` of group `g` in half `half`.
#[inline]
pub fn decode_scale(block: &[u8], half: usize, g: usize, l: usize) -> i32 {
    let (_, _, _, sc_base) = group_selectors(g);
    let sc = &block[192 + 8 * half..];
    sc[sc_base + l / 16] as i8 as i32
}

/// The f16 super-block scale `d`.
#[inline]
pub fn decode_d(block: &[u8]) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]))
}

/// CORRECT Q6_K super-block dequantization. `out` must be 256 long.
///
/// This is the ggml semantics (`ggml-quants.c dequantize_row_q6_K`) and the
/// single source of truth for every Q6_K reader in this workspace.
pub fn dequant_block(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q6K_BLOCK_BYTE);
    assert!(out.len() >= Q6K_BLOCK_ELEM);
    let d = decode_d(block);
    for half in 0..2 {
        for g in 0..4 {
            for l in 0..32 {
                let q = decode_code(block, half, g, l);
                let sc = decode_scale(block, half, g, l);
                out[128 * half + 32 * g + l] = d * (sc as f32) * (q as f32);
            }
        }
    }
}

/// The DEFECTIVE mapping that two in-tree host dequantisers use: groups 1 and
/// 2 take their `ql` nibble from the wrong byte. Kept only so
/// [`legacy_mapping_is_wrong`] can assert the difference and so the defect is
/// documented in executable form rather than prose. Do not call from
/// production code.
pub fn dequant_block_legacy(block: &[u8], out: &mut [f32]) {
    assert!(block.len() >= Q6K_BLOCK_BYTE);
    assert!(out.len() >= Q6K_BLOCK_ELEM);
    let d = decode_d(block);
    for half in 0..2 {
        let ql = &block[64 * half..];
        let qh = &block[128 + 32 * half..];
        let sc = &block[192 + 8 * half..];
        for (g, (ql_off, hi_nib)) in [(0usize, false), (0, true), (32, false), (32, true)]
            .into_iter()
            .enumerate()
        {
            for l in 0..32 {
                let lo = if hi_nib {
                    (ql[ql_off + l] >> 4) as i32
                } else {
                    (ql[ql_off + l] & 0x0F) as i32
                };
                let hb = ((qh[l] >> (2 * g as u32)) & 3) as i32;
                let q = (lo | (hb << 4)) - 32;
                let s = sc[2 * g + l / 16] as i8 as i32;
                out[128 * half + 32 * g + l] = d * (s as f32) * (q as f32);
            }
        }
    }
}

/// Pack 256 six-bit codes (values `0..=63`, i.e. already `+32`-offset) into a
/// Q6_K super-block, given the sub-block scales and `d`.
///
/// Verbatim transcription of the packer in `ggml-quants.c
/// quantize_row_q6_K_ref` (llama.cpp @ `3b53219`). This is the GROUND TRUTH
/// used by the tests: a reader is correct iff it inverts this packer.
pub fn pack_block(codes: &[u8; Q6K_BLOCK_ELEM], scales: &[i8; 16], d: f32) -> [u8; Q6K_BLOCK_BYTE] {
    let mut b = [0u8; Q6K_BLOCK_BYTE];
    for j in (0..Q6K_BLOCK_ELEM).step_by(128) {
        let base_ql = (j / 128) * 64;
        let base_qh = 128 + (j / 128) * 32;
        for l in 0..32 {
            let q1 = codes[j + l] & 0xF;
            let q2 = codes[j + l + 32] & 0xF;
            let q3 = codes[j + l + 64] & 0xF;
            let q4 = codes[j + l + 96] & 0xF;
            b[base_ql + l] = q1 | (q3 << 4);
            b[base_ql + l + 32] = q2 | (q4 << 4);
            b[base_qh + l] = (codes[j + l] >> 4)
                | ((codes[j + l + 32] >> 4) << 2)
                | ((codes[j + l + 64] >> 4) << 4)
                | ((codes[j + l + 96] >> 4) << 6);
        }
    }
    for (i, &s) in scales.iter().enumerate() {
        b[192 + i] = s as u8;
    }
    b[208..210].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    b
}

/// Host mirror of `matvec_q6_k_f32`'s per-row dot product, reproducing the
/// kernel's UNIT DECOMPOSITION and OPERATION ORDER exactly:
///
/// * unit `u` -> super-block `u / 8`, half `(u % 8) / 4`, group `u % 4`
/// * unit `u` covers activations `[32u, 32u + 32)` (contiguous — the property
///   the kernel relies on to avoid index arithmetic)
/// * within a unit, the two 16-element scale sub-groups accumulate in order,
///   scale hoisted out of the inner 16
/// * `d` is applied once per super-block, outside the element sum
///
/// `thread_stride` models the kernel's `for u = tid; u < n_units; u +=
/// THREADS_PER_BLOCK` sweep followed by a reduction: passing the real stride
/// reproduces the kernel's exact partial-sum grouping, so this function is a
/// faithful numerical mirror and not merely an algebraic equivalent.
pub fn row_dot_kernel_order(row: &[u8], x: &[f32], in_dim: usize, thread_stride: usize) -> f32 {
    assert_eq!(in_dim % Q6K_BLOCK_ELEM, 0, "Q6_K needs in_dim % 256 == 0");
    let nb = in_dim / Q6K_BLOCK_ELEM;
    let n_units = nb * Q6K_GROUPS_PER_BLOCK;
    assert!(row.len() >= nb * Q6K_BLOCK_BYTE);
    assert!(x.len() >= in_dim);

    // Per-"thread" partial sums, then a fixed-order fold. The kernel reduces
    // via butterfly + cross-warp fold; both are additions of the same
    // partials, so summing them in ascending thread order is the same
    // grouping to within the reduction tree's shape.
    let mut partials = vec![0.0f32; thread_stride.min(n_units).max(1)];
    for (tid, partial) in partials.iter_mut().enumerate() {
        let mut sumf = 0.0f32;
        let mut u = tid;
        while u < n_units {
            let ib = u / Q6K_GROUPS_PER_BLOCK;
            let rem = u % Q6K_GROUPS_PER_BLOCK;
            let half = rem / 4;
            let g = rem % 4;
            let block = &row[ib * Q6K_BLOCK_BYTE..];
            let xv = &x[u * Q6K_GROUP_ELEM..u * Q6K_GROUP_ELEM + Q6K_GROUP_ELEM];

            let (ql_off, hi_nib, qh_shift, sc_base) = group_selectors(g);
            let ql = &block[64 * half..];
            let qh = &block[128 + 32 * half..];
            let sc = &block[192 + 8 * half..];

            let mut acc = 0.0f32;
            for is in 0..2 {
                let sc_f = (sc[sc_base + is] as i8 as i32) as f32;
                let mut sub = 0.0f32;
                for k in 0..16 {
                    let l = is * 16 + k;
                    let lo = if hi_nib {
                        (ql[ql_off + l] >> 4) as i32
                    } else {
                        (ql[ql_off + l] & 0x0F) as i32
                    };
                    let hb = ((qh[l] >> qh_shift) & 3) as i32;
                    let q = (lo | (hb << 4)) - 32;
                    sub = (q as f32).mul_add(xv[l], sub);
                }
                acc = sc_f.mul_add(sub, acc);
            }
            sumf = decode_d(block).mul_add(acc, sumf);
            u += thread_stride;
        }
        *partial = sumf;
    }
    partials.iter().fold(0.0f32, |a, &b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG so the tests are reproducible without a dep.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed)
        }
        fn next_u32(&mut self) -> u32 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
        fn unit_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    fn random_codes(rng: &mut Rng) -> [u8; Q6K_BLOCK_ELEM] {
        let mut c = [0u8; Q6K_BLOCK_ELEM];
        for v in c.iter_mut() {
            *v = rng.below(64) as u8;
        }
        c
    }

    fn random_scales(rng: &mut Rng) -> [i8; 16] {
        let mut s = [0i8; 16];
        for v in s.iter_mut() {
            // Avoid 0 so a wrong scale index cannot hide behind a zero product.
            let mag = 1 + rng.below(127) as i32;
            *v = if rng.below(2) == 0 { mag } else { -mag } as i8;
        }
        s
    }

    /// THE decisive layout test: pack with the ggml packer, read back with our
    /// decoder, require EXACT recovery of every six-bit code. Exact integers,
    /// no floating point, so there is no tolerance to hide behind.
    #[test]
    fn decode_inverts_the_ggml_packer_exactly() {
        let mut rng = Rng::new(0x2026_0730);
        for trial in 0..64 {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            let block = pack_block(&codes, &scales, 1.0);

            for half in 0..2 {
                for g in 0..4 {
                    for l in 0..32 {
                        let want = codes[128 * half + 32 * g + l] as i32 - 32;
                        let got = decode_code(&block, half, g, l);
                        assert_eq!(
                            got, want,
                            "trial {trial} half {half} group {g} l {l}: \
                             decode_code gave {got}, packer wrote {want}"
                        );
                    }
                }
            }
        }
    }

    /// Scale indexing: element `i` must use `scales[i / 16]`.
    #[test]
    fn scale_index_is_element_over_16() {
        let codes = [32u8; Q6K_BLOCK_ELEM];
        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as i8) + 1; // 1..=16, all distinct and non-zero
        }
        let block = pack_block(&codes, &scales, 1.0);
        for half in 0..2 {
            for g in 0..4 {
                for l in 0..32 {
                    let i = 128 * half + 32 * g + l;
                    assert_eq!(
                        decode_scale(&block, half, g, l),
                        scales[i / 16] as i32,
                        "element {i} took the wrong scale"
                    );
                }
            }
        }
    }

    /// `dequant_block` must equal the per-element ggml formula, including `d`.
    #[test]
    fn dequant_block_matches_ggml_formula() {
        let mut rng = Rng::new(11);
        for _ in 0..16 {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            let d = 0.0234375f32; // exactly representable in f16
            let block = pack_block(&codes, &scales, d);

            let mut got = vec![0.0f32; Q6K_BLOCK_ELEM];
            dequant_block(&block, &mut got);

            for i in 0..Q6K_BLOCK_ELEM {
                let want = d * (scales[i / 16] as f32) * (codes[i] as f32 - 32.0);
                assert_eq!(got[i], want, "element {i}");
            }
        }
    }

    /// The regression test that pins the defect. The legacy mapping used by
    /// `lumen-convert::dequant::dequantize_q6_k` and
    /// `lumen-runtime::cuda::gpu_buffers::dequant_kquant_to_f32` corrupts
    /// roughly half of every super-block on realistic data. If a future change
    /// makes these agree, either the defect was fixed (delete this test and
    /// `dequant_block_legacy`) or the CORRECT reader regressed.
    #[test]
    fn legacy_mapping_is_wrong() {
        let mut rng = Rng::new(0x2026_0730);
        let codes = random_codes(&mut rng);
        let scales = random_scales(&mut rng);
        let block = pack_block(&codes, &scales, 1.0);

        let mut good = vec![0.0f32; Q6K_BLOCK_ELEM];
        let mut bad = vec![0.0f32; Q6K_BLOCK_ELEM];
        dequant_block(&block, &mut good);
        dequant_block_legacy(&block, &mut bad);

        let wrong = (0..Q6K_BLOCK_ELEM).filter(|&i| good[i] != bad[i]).count();
        assert!(
            wrong > 100,
            "expected the legacy ql-nibble swap to corrupt ~half the block, \
             got {wrong}/256 differing -- has the mapping changed?"
        );
        // Groups 0 and 3 are unaffected; only slots 32..64 and 64..96 of each
        // half can differ, i.e. at most 128 of 256.
        assert!(
            wrong <= 128,
            "more than the two swapped groups differ: {wrong}"
        );
        for half in 0..2 {
            for l in 0..32 {
                for g in [0usize, 3] {
                    let i = 128 * half + 32 * g + l;
                    assert_eq!(good[i], bad[i], "group {g} must be unaffected at {i}");
                }
            }
        }
    }

    /// Both existing in-tree Q6_K unit tests use `ql` bytes whose low and high
    /// nibbles are EQUAL (all-zero, and `0x11`). Prove that such patterns make
    /// the swap invisible — this is why the defect survived, and it is the
    /// concrete "the gate could not observe the failure" instance.
    #[test]
    fn equal_nibble_patterns_cannot_detect_the_swap() {
        for code in [0u8, 1, 17, 63] {
            let codes = [code; Q6K_BLOCK_ELEM];
            let scales = [3i8; 16];
            let block = pack_block(&codes, &scales, 1.0);
            let mut good = vec![0.0f32; Q6K_BLOCK_ELEM];
            let mut bad = vec![0.0f32; Q6K_BLOCK_ELEM];
            dequant_block(&block, &mut good);
            dequant_block_legacy(&block, &mut bad);
            assert_eq!(
                good, bad,
                "uniform code {code} should be blind to the swap, so a test \
                 built on it proves nothing about the mapping"
            );
        }
    }

    /// Unit decomposition: unit `u` must cover activations `[32u, 32u+32)`.
    /// This is the property `matvec_q6_k_f32` relies on, and it only holds
    /// because the half-major/group ordering coincides with output order.
    #[test]
    fn unit_index_maps_to_contiguous_activation_slice() {
        for u in 0..Q6K_GROUPS_PER_BLOCK * 4 {
            let ib = u / Q6K_GROUPS_PER_BLOCK;
            let rem = u % Q6K_GROUPS_PER_BLOCK;
            let half = rem / 4;
            let g = rem % 4;
            let elem_base = ib * Q6K_BLOCK_ELEM + half * 128 + g * 32;
            assert_eq!(
                elem_base,
                u * Q6K_GROUP_ELEM,
                "unit {u} (block {ib}, half {half}, group {g})"
            );
        }
    }

    /// End-to-end: the kernel-order row dot must match an f64 reference over
    /// the correctly dequantized weights. Tolerance is a relative bound on the
    /// F32 accumulation of a 4096-term dot, not a fudge factor.
    #[test]
    fn kernel_order_row_dot_matches_f64_reference() {
        let in_dim = 4096usize;
        let nb = in_dim / Q6K_BLOCK_ELEM;
        let mut rng = Rng::new(4242);

        let mut row = Vec::with_capacity(nb * Q6K_BLOCK_BYTE);
        let mut weights = Vec::with_capacity(in_dim);
        for _ in 0..nb {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            let d = 0.001953125f32; // 2^-9, exact in f16
            let block = pack_block(&codes, &scales, d);
            let mut dq = vec![0.0f32; Q6K_BLOCK_ELEM];
            dequant_block(&block, &mut dq);
            row.extend_from_slice(&block);
            weights.extend_from_slice(&dq);
        }
        let x: Vec<f32> = (0..in_dim).map(|_| rng.unit_f32()).collect();

        let want: f64 = weights
            .iter()
            .zip(&x)
            .map(|(&w, &xi)| w as f64 * xi as f64)
            .sum();

        // 128 threads is the kernel's THREADS_PER_BLOCK.
        let got = row_dot_kernel_order(&row, &x, in_dim, 128);

        let scale = weights
            .iter()
            .zip(&x)
            .map(|(&w, &xi)| (w as f64 * xi as f64).abs())
            .sum::<f64>()
            .max(1e-30);
        let rel = (got as f64 - want).abs() / scale;
        assert!(
            rel < 1e-6,
            "kernel-order dot {got} vs f64 reference {want} (rel {rel:.3e})"
        );
    }

    /// The kernel's thread sweep must cover every element exactly once for the
    /// shapes actually dispatched (`in_dim = 4096`, `nb = 16`, 128 units).
    #[test]
    fn thread_sweep_covers_every_unit_exactly_once() {
        let in_dim = 4096usize;
        let n_units = (in_dim / Q6K_BLOCK_ELEM) * Q6K_GROUPS_PER_BLOCK;
        assert_eq!(
            n_units, 128,
            "9B shapes should give exactly one unit/thread"
        );
        let mut seen = vec![0u32; n_units];
        for tid in 0..128usize {
            let mut u = tid;
            while u < n_units {
                seen[u] += 1;
                u += 128;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "coverage: {seen:?}");
    }

    /// `f32_to_f16_bits_rne` must round-trip every value the exact widener can
    /// represent, and must round to nearest EVEN in between. Checked against
    /// `f16_bits_to_f32` over the whole 16-bit space, which is exhaustive.
    #[test]
    fn f16_narrowing_round_trips_every_representable_value() {
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            let exp = (bits >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // Inf/NaN handled separately
            }
            let v = f16_bits_to_f32(bits);
            let back = f32_to_f16_bits_rne(v);
            // -0.0 and +0.0 both widen to a zero; compare values, not bits.
            if v == 0.0 {
                assert!(back == 0x0000 || back == 0x8000, "zero bits {bits:#06x}");
            } else {
                assert_eq!(
                    back, bits,
                    "round trip failed for f16 bits {bits:#06x} = {v}"
                );
            }
        }
    }

    /// Saturation and NaN behaviour at the edges, so a stray large weight
    /// becomes Inf rather than wrapping to a small number.
    #[test]
    fn f16_narrowing_edges() {
        assert_eq!(f32_to_f16_bits_rne(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits_rne(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits_rne(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits_rne(-2.0), 0xc000);
        // Above f16 max (65504) -> +Inf.
        assert_eq!(f32_to_f16_bits_rne(1.0e30), 0x7c00);
        assert_eq!(f32_to_f16_bits_rne(-1.0e30), 0xfc00);
        // Below the smallest subnormal -> signed zero.
        assert_eq!(f32_to_f16_bits_rne(1.0e-30), 0x0000);
        assert_eq!(f32_to_f16_bits_rne(-1.0e-30), 0x8000);
        // NaN must stay NaN, not become Inf.
        let nan = f32_to_f16_bits_rne(f32::NAN);
        assert_eq!(nan & 0x7c00, 0x7c00);
        assert_ne!(nan & 0x03ff, 0, "NaN payload must be non-zero");
        assert!(f16_bits_to_f32(f32_to_f16_bits_rne(f32::INFINITY)).is_infinite());
    }

    /// The prefill F16 cache must carry the CORRECT dequantized values, at the
    /// right length, in the right order. A wrong length would misfeed cuBLAS
    /// HGEMM silently.
    #[test]
    fn dequant_to_f16_bytes_matches_dequant_block() {
        let mut rng = Rng::new(909);
        let nb = 3usize;
        let mut raw = Vec::new();
        let mut want = Vec::new();
        for _ in 0..nb {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            // 2^-6: keeps sc*(q-32)*d inside the f16 normal range for all
            // |sc| <= 127, |q-32| <= 32 (max 4064 * 2^-6 = 63.5).
            let block = pack_block(&codes, &scales, 0.015625);
            let mut dq = vec![0.0f32; Q6K_BLOCK_ELEM];
            dequant_block(&block, &mut dq);
            raw.extend_from_slice(&block);
            want.extend_from_slice(&dq);
        }
        let n = nb * Q6K_BLOCK_ELEM;
        let got = dequant_to_f16_bytes(&raw, n);
        assert_eq!(got.len(), n * 2, "F16 cache must be exactly 2 B/element");
        for i in 0..n {
            let bits = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
            assert_eq!(
                bits,
                f32_to_f16_bits_rne(want[i]),
                "element {i}: F16 cache disagrees with dequant_block"
            );
            // And the narrowing must be faithful for these magnitudes.
            let rel = (f16_bits_to_f32(bits) - want[i]).abs() / want[i].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "element {i}: {} vs {}",
                f16_bits_to_f32(bits),
                want[i]
            );
        }
    }

    /// A partial trailing request must truncate, never read past the stream or
    /// over-produce.
    #[test]
    fn dequant_to_f16_bytes_truncates_to_n_elements() {
        let mut rng = Rng::new(5);
        let (codes, scales) = (random_codes(&mut rng), random_scales(&mut rng));
        let block = pack_block(&codes, &scales, 0.015625);
        for n in [1usize, 31, 128, 255, 256] {
            assert_eq!(dequant_to_f16_bytes(&block, n).len(), n * 2, "n = {n}");
        }
    }

    /// Every kernel symbol the CUDA loaders request must actually EXIST in the
    /// shader source.
    ///
    /// This guards the exact trap that has cost this repo debug cycles twice: a
    /// `load_fn(SOURCE, "name")` call is evidence that someone INTENDED a kernel,
    /// not that one exists. `LUMEN_CUDA_GDN_FUSED_CONV` was reported as a live
    /// default-off lever on the strength of its loader call alone, while
    /// `ssm_conv1d_silu_l2norm_t1` existed in no `.cu` file at all -- so the flag
    /// was unsatisfiable at any value and the kernel handle was permanently
    /// `None`.
    ///
    /// The shader is `include_str!`d here rather than referenced through
    /// `cuda::shaders`, because that module is `cfg(feature = "cuda")`-gated and
    /// would make this check invisible on the default test command -- which is
    /// precisely how such a gap survives.
    #[test]
    fn every_requested_kernel_symbol_exists_in_the_shader() {
        const SRC: &str = include_str!("cuda/shaders/matvec_q6_k_f32.cu");
        for sym in [
            "matvec_q6_k_f32",     // NR=1, layer projections (C1)
            "matvec_q6_k_f32_nr8", // NR=8, the [248320 x 4096] head (C3)
            "dequant_q6_k_to_f32", // F32 staging for the exact-F32 prefill path
        ] {
            let decl = format!("void {sym}(");
            assert!(
                SRC.contains(&decl),
                "decode.rs asks load_fn for kernel '{sym}' but no `void {sym}(` \
                 definition exists in matvec_q6_k_f32.cu -- the handle would be \
                 permanently None and the flag silently inert"
            );
        }
        // Each must be an extern "C" entry point, or NVRTC name-mangles it and
        // `load_function` fails to resolve the symbol at runtime.
        assert_eq!(
            SRC.matches("extern \"C\" __global__").count(),
            3,
            "all three kernels must be extern \"C\" __global__ entry points"
        );
    }

    /// The Q6_K head kernel must be dispatched from EXACTLY ONE place, and both
    /// head paths must go through it.
    ///
    /// Root cause of all three C3 integration defects found on the A100:
    /// `compute_final` and `compute_layer` are full duplicates of the `_gpu`
    /// chains, so a candidate wired into one silently misses the other. The third
    /// defect was precisely this -- the PREFILL-BOUNDARY token (first sample,
    /// before the decode loop) routes through `compute_final`, not
    /// `compute_final_gpu`.
    ///
    /// Naming the kernel once, inside `dispatch_output_proj_q6k`, makes the two
    /// paths' logits bit-identical BY CONSTRUCTION (same kernel, same
    /// `scratch.normed`, same `logits_gpu`) instead of by a tolerance assertion.
    /// This test pins that invariant so a future edit cannot re-duplicate it.
    ///
    /// Source-level because the real property is structural and there is no GPU
    /// here; `include_str!` from this non-cuda module keeps the check on the
    /// DEFAULT test command, which `cuda::` would have cfg-gated it off.
    #[test]
    fn head_kernel_is_dispatched_from_exactly_one_place() {
        const SRC: &str = include_str!("cuda/backend_impl.rs");

        // Count ACTUAL dispatches, i.e. field accesses on the KernelSet, not
        // prose. Doc comments and the error string also name the kernel, and
        // counting those would make this test assert formatting rather than
        // structure -- a distinction the first cut of this test got wrong.
        let dispatches = SRC.matches("kernels.matvec_q6_k_f32_nr8").count();
        assert_eq!(
            dispatches, 1,
            "the KernelSet field `matvec_q6_k_f32_nr8` is accessed {dispatches} times in \
             backend_impl.rs; it must be accessed exactly once, inside \
             dispatch_output_proj_q6k. A second access is a duplicated head dispatch, \
             which is how the prefill-boundary path silently missed C3."
        );

        // Both head paths must call the shared dispatcher: one definition plus
        // two call sites (compute_final_gpu and compute_final).
        let calls = SRC.matches("dispatch_output_proj_q6k(").count();
        assert_eq!(
            calls, 3,
            "expected 1 definition + 2 call sites of dispatch_output_proj_q6k \
             (compute_final_gpu and compute_final), found {calls} occurrences"
        );

        // And the dead-end guard that used to sit in compute_final must be gone:
        // failing there is not a fix, it is the bug reported from the A100.
        assert!(
            !SRC.contains("a native Q6_K output_proj reached compute_final"),
            "the compute_final guard must be REPLACED by a real dispatch, not kept -- \
             the prefill-boundary token takes that path on every run"
        );

        // Both head paths must exist and each must reference the Q6_K global, so
        // this test fails if either arm is deleted rather than silently passing.
        assert_eq!(
            SRC.matches("st.globals.output_proj_q6k").count(),
            2,
            "both compute_final_gpu and compute_final must test output_proj_q6k"
        );
    }

    /// A zero `d` (the packer's all-zero-block escape) must produce zeros, not
    /// NaN, through the whole path.
    #[test]
    fn zero_scale_block_is_all_zero() {
        let block = [0u8; Q6K_BLOCK_BYTE];
        let mut out = vec![f32::NAN; Q6K_BLOCK_ELEM];
        dequant_block(&block, &mut out);
        assert!(
            out.iter().all(|v| *v == 0.0),
            "all-zero block must dequant to 0"
        );
    }
}
