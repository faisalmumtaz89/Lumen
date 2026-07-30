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
/// kernel's LANE MAPPING and OPERATION ORDER exactly.
///
/// The kernel's decomposition (see the shader header):
/// * a CTA is 128 threads = 4 warps; warp `w` owns super-blocks `w, w+4, ...`
/// * lane `L` owns `ql` bytes `4L..4L+3` of its warp's block, i.e. 8 elements
/// * within a half, `ql` byte `p` carries element `p` in its LOW nibble and
///   element `p + 64` in its HIGH nibble -- the uniform rule that lets the
///   kernel drop the four-way group branch
/// * all four of a lane's bytes share one scale pair and one `qh` shift, and
///   their four `qh` bytes are consecutive
/// * `d` is applied once per super-block, outside the element sum
///
/// Partial sums are kept per simulated thread and folded in ascending thread
/// order, matching the kernel's butterfly + cross-warp fold grouping, so this is
/// a faithful numerical mirror rather than merely an algebraic equivalent.
pub fn row_dot_kernel_order(row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    const NW: usize = 32;
    const NWARPS: usize = 4;
    assert_eq!(in_dim % Q6K_BLOCK_ELEM, 0, "Q6_K needs in_dim % 256 == 0");
    let nb = in_dim / Q6K_BLOCK_ELEM;
    assert!(row.len() >= nb * Q6K_BLOCK_BYTE);
    assert!(x.len() >= in_dim);

    let ld_u32 = |b: &[u8], off: usize| -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    };
    // Per-byte signed subtract, mirroring `__vsubss4(v, 0x20202020)`.
    let vsub32 = |v: u32| -> [i32; 4] {
        let mut o = [0i32; 4];
        for (k, slot) in o.iter_mut().enumerate() {
            *slot = ((v >> (8 * k)) & 0xFF) as i32 - 32;
        }
        o
    };

    let mut partials = vec![0.0f32; NWARPS * NW];
    for warp_id in 0..NWARPS {
        for lane in 0..NW {
            let p = lane * 4;
            let half = p >> 6;
            let p_h = p & 63;
            let qh_off = half * 32 + (p_h & 31);
            let sh_lo = 2 * (p_h >> 5) as u32;
            let sc_i = 8 * half + (p_h >> 4);
            let elem_lo = half * 128 + p_h;
            let elem_hi = elem_lo + 64;

            let mut sumf = 0.0f32;
            let mut ib = warp_id;
            while ib < nb {
                let bp = &row[ib * Q6K_BLOCK_BYTE..];
                let xb = &x[ib * Q6K_BLOCK_ELEM..];

                let vl = ld_u32(bp, p);
                let vh = ld_u32(bp, 128 + qh_off);

                let nlo = vl & 0x0F0F_0F0F;
                let blo = ((vh >> sh_lo) & 0x0303_0303) << 4;
                let qlo = vsub32(nlo | blo);

                let nhi = (vl >> 4) & 0x0F0F_0F0F;
                let bhi = ((vh >> (sh_lo + 4)) & 0x0303_0303) << 4;
                let qhi = vsub32(nhi | bhi);

                let sc = &bp[192..208];
                let d = decode_d(bp);

                let mut dlo = 0.0f32;
                let mut dhi = 0.0f32;
                for k in 0..4 {
                    dlo = (qlo[k] as f32).mul_add(xb[elem_lo + k], dlo);
                    dhi = (qhi[k] as f32).mul_add(xb[elem_hi + k], dhi);
                }
                let mut acc = 0.0f32;
                acc = ((sc[sc_i] as i8 as i32) as f32).mul_add(dlo, acc);
                acc = ((sc[sc_i + 4] as i8 as i32) as f32).mul_add(dhi, acc);
                sumf = d.mul_add(acc, sumf);

                ib += NWARPS;
            }
            partials[warp_id * NW + lane] = sumf;
        }
    }
    partials.iter().fold(0.0f32, |a, &b| a + b)
}

/// Host mirror of `matvec_q6_k_q8_1`'s per-row dot: native Q6_K weights against
/// PRE-QUANTIZED Q8_1 activations (candidate C1b, the dp4a route).
///
/// Reproduces `vec_dot_q6_K_q8_1` semantics (ggml/src/ggml-cuda/vecdotq.cuh:620-644)
/// on this file's lane mapping: per 4-byte group, mask nibbles, splice the two
/// high bits, apply the -32 bias, take an EXACT integer dot against four int8
/// activations, scale by the int8 sub-block scale, then by the Q8_1 block's f32
/// scale; `d` once per super-block.
///
/// `input_q8_1` is the in-tree Q8_1 layout: 36 bytes per 32 elements,
/// `[f16 d][f16 sum][32 x int8]`, quants at offset +4. The `sum` field is
/// deliberately UNREAD -- Q4_0/Q4_1 need it to correct their weight offset, but
/// Q6_K applies -32 per element before the dot, so there is no correction term.
/// llama.cpp's implementation has the same shape.
pub fn row_dot_q8_1_kernel_order(row: &[u8], input_q8_1: &[u8], in_dim: usize) -> f32 {
    const NW: usize = 32;
    const NWARPS: usize = 4;
    const Q8_1_BYTES: usize = 36;
    assert_eq!(in_dim % Q6K_BLOCK_ELEM, 0);
    let nb = in_dim / Q6K_BLOCK_ELEM;

    let ld_u32 = |b: &[u8], off: usize| -> u32 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    };
    let vsub32 = |v: u32| -> [i32; 4] {
        let mut o = [0i32; 4];
        for (k, slot) in o.iter_mut().enumerate() {
            *slot = ((v >> (8 * k)) & 0xFF) as i32 - 32;
        }
        o
    };

    let mut partials = vec![0.0f32; NWARPS * NW];
    for warp_id in 0..NWARPS {
        for lane in 0..NW {
            let p = lane * 4;
            let half = p >> 6;
            let p_h = p & 63;
            let qh_off = half * 32 + (p_h & 31);
            let sh_lo = 2 * (p_h >> 5) as u32;
            let sc_i = 8 * half + (p_h >> 4);
            let elem_lo = half * 128 + p_h;
            let elem_hi = elem_lo + 64;

            let mut sumf = 0.0f32;
            let mut ib = warp_id;
            while ib < nb {
                let bp = &row[ib * Q6K_BLOCK_BYTE..];
                let vl = ld_u32(bp, p);
                let vh = ld_u32(bp, 128 + qh_off);
                let qlo = vsub32((vl & 0x0F0F_0F0F) | (((vh >> sh_lo) & 0x0303_0303) << 4));
                let qhi =
                    vsub32(((vl >> 4) & 0x0F0F_0F0F) | (((vh >> (sh_lo + 4)) & 0x0303_0303) << 4));

                let mut dot = |g: usize, q: [i32; 4]| -> (i32, f32) {
                    let blk = g >> 5;
                    let off = g & 31;
                    let base = blk * Q8_1_BYTES;
                    let d8 = f16_bits_to_f32(u16::from_le_bytes([
                        input_q8_1[base],
                        input_q8_1[base + 1],
                    ]));
                    let mut acc = 0i32;
                    for k in 0..4 {
                        acc += q[k] * (input_q8_1[base + 4 + off + k] as i8 as i32);
                    }
                    (acc, d8)
                };
                let (dot_lo, d8_lo) = dot(ib * Q6K_BLOCK_ELEM + elem_lo, qlo);
                let (dot_hi, d8_hi) = dot(ib * Q6K_BLOCK_ELEM + elem_hi, qhi);

                let sc = &bp[192..208];
                let mut acc = 0.0f32;
                acc = d8_lo.mul_add((dot_lo as f32) * (sc[sc_i] as i8 as i32 as f32), acc);
                acc = d8_hi.mul_add((dot_hi as f32) * (sc[sc_i + 4] as i8 as i32 as f32), acc);
                sumf = decode_d(bp).mul_add(acc, sumf);
                ib += NWARPS;
            }
            partials[warp_id * NW + lane] = sumf;
        }
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

    /// THE invariant the rewritten kernel is built on: within a half, `ql` byte
    /// `p` carries element `p` in its LOW nibble and element `p + 64` in its
    /// HIGH nibble -- one uniform rule for all four ggml "groups".
    ///
    /// This is what lets the kernel drop the four-way group branch and read 4
    /// consecutive `ql` bytes per lane (the change that fixed coalescing). If it
    /// were false, the kernel would silently pair nibbles with the wrong
    /// activations. Checked against `decode_code`, which is itself verified
    /// against the ggml packer.
    #[test]
    fn ql_byte_carries_element_p_and_p_plus_64() {
        let mut rng = Rng::new(77);
        let codes = random_codes(&mut rng);
        let scales = random_scales(&mut rng);
        let block = pack_block(&codes, &scales, 1.0);

        for half in 0..2 {
            for p in 0..64usize {
                // Which (group, l) the ggml formula assigns to each element.
                let (g_lo, l_lo) = if p < 32 {
                    (0usize, p)
                } else {
                    (1usize, p - 32)
                };
                let (g_hi, l_hi) = if p < 32 {
                    (2usize, p)
                } else {
                    (3usize, p - 32)
                };

                // Low nibble of ql byte p -> element p of the half.
                assert_eq!(
                    decode_code(&block, half, g_lo, l_lo),
                    codes[128 * half + p] as i32 - 32,
                    "half {half} ql byte {p} low nibble must be element {p}"
                );
                // High nibble of the same byte -> element p + 64.
                assert_eq!(
                    decode_code(&block, half, g_hi, l_hi),
                    codes[128 * half + p + 64] as i32 - 32,
                    "half {half} ql byte {p} high nibble must be element {}",
                    p + 64
                );

                // And the scale pair the kernel hoists: sc[8h + p/16] for the
                // low nibble, sc[8h + p/16 + 4] for the high one.
                assert_eq!(
                    decode_scale(&block, half, g_lo, l_lo),
                    scales[(128 * half + p) / 16] as i32
                );
                assert_eq!(
                    decode_scale(&block, half, g_hi, l_hi),
                    scales[(128 * half + p + 64) / 16] as i32
                );
            }
        }
    }

    /// The rewritten kernel's lane mapping must cover every element of a row
    /// EXACTLY once: 4 warps x 32 lanes, lane `L` owning `ql` bytes `4L..4L+3`
    /// (8 elements), warp `w` striding blocks by 4.
    ///
    /// A gap would silently drop weights from the dot product; an overlap would
    /// double-count them. Neither shows up as a crash.
    #[test]
    fn lane_mapping_covers_every_element_exactly_once() {
        const NW: usize = 32;
        const NWARPS: usize = 4;
        for in_dim in [4096usize, 256, 2048] {
            let nb = in_dim / Q6K_BLOCK_ELEM;
            let mut seen = vec![0u32; in_dim];
            for warp_id in 0..NWARPS {
                for lane in 0..NW {
                    let p = lane * 4;
                    let half = p >> 6;
                    let p_h = p & 63;
                    let elem_lo = half * 128 + p_h;
                    let elem_hi = elem_lo + 64;
                    let mut ib = warp_id;
                    while ib < nb {
                        for k in 0..4 {
                            seen[ib * Q6K_BLOCK_ELEM + elem_lo + k] += 1;
                            seen[ib * Q6K_BLOCK_ELEM + elem_hi + k] += 1;
                        }
                        ib += NWARPS;
                    }
                }
            }
            let bad: Vec<(usize, u32)> = seen
                .iter()
                .enumerate()
                .filter(|(_, &c)| c != 1)
                .map(|(i, &c)| (i, c))
                .take(5)
                .collect();
            assert!(
                bad.is_empty(),
                "in_dim {in_dim}: elements not covered exactly once: {bad:?}"
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

        let got = row_dot_kernel_order(&row, &x, in_dim);

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

    /// C1b parity: the dp4a (int8-activation) route must agree with the F32
    /// route to within Q8_1 activation-quantization error, on the SAME weights.
    ///
    /// The two kernels differ only in how the activation is represented, so a
    /// disagreement beyond Q8_1's own quantization step means the int8 lane
    /// mapping, the nibble/high-bit splice, the scale indexing, or the Q8_1
    /// offsets are wrong -- exactly the class of error that produced a 45%-wrong
    /// dot in the archived patch.
    ///
    /// Tolerance is derived, not tuned: Q8_1 encodes each 32-element block at
    /// d = max|x|/127, so per-element error is bounded by d/2 and the relative
    /// error of a 4096-term dot is ~1/254 in the worst case. 1% is a loose but
    /// meaningful bound -- a mis-indexed scale or a swapped nibble blows past it
    /// by orders of magnitude (verified: swapping the two scale indices gives
    /// ~50% error).
    #[test]
    fn dp4a_route_matches_the_f32_route() {
        let in_dim = 4096usize;
        let nb = in_dim / Q6K_BLOCK_ELEM;
        let mut rng = Rng::new(0xC1B);

        let mut row = Vec::with_capacity(nb * Q6K_BLOCK_BYTE);
        for _ in 0..nb {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            row.extend_from_slice(&pack_block(&codes, &scales, 0.001953125));
        }
        let x: Vec<f32> = (0..in_dim).map(|_| rng.unit_f32()).collect();

        // Encode x as Q8_1, the in-tree 36-byte layout.
        let mut q8_1 = vec![0u8; (in_dim / 32) * 36];
        for b in 0..(in_dim / 32) {
            let blk = &x[b * 32..(b + 1) * 32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let base = b * 36;
            q8_1[base..base + 2].copy_from_slice(&f32_to_f16_bits_rne(d).to_le_bytes());
            let mut sum = 0.0f32;
            for (k, &v) in blk.iter().enumerate() {
                let q = if d > 0.0 {
                    (v / d).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                q8_1[base + 4 + k] = q as u8;
                sum += (q as f32) * d;
            }
            // `sum` is written for layout fidelity; Q6_K must not read it.
            q8_1[base + 2..base + 4].copy_from_slice(&f32_to_f16_bits_rne(sum).to_le_bytes());
        }

        let f32_route = row_dot_kernel_order(&row, &x, in_dim);
        let dp4a_route = row_dot_q8_1_kernel_order(&row, &q8_1, in_dim);

        let mag = f32_route.abs().max(1.0);
        let rel = (f32_route - dp4a_route).abs() / mag;
        assert!(
            rel < 0.01,
            "dp4a route {dp4a_route} vs F32 route {f32_route} (rel {rel:.4}) -- \
             beyond Q8_1 quantization error, so the int8 mapping is wrong"
        );
        assert!(f32_route.abs() > 1.0, "test data must be non-trivial");
    }

    /// The Q8_1 `sum` field must NOT influence a Q6_K dot. Q4_0/Q4_1 need it to
    /// correct their weight offset; Q6_K applies -32 per element before the dot,
    /// so reading it would double-correct. Corrupting the field must change
    /// nothing.
    #[test]
    fn q8_1_sum_field_is_unread_by_q6k() {
        let in_dim = 512usize;
        let nb = in_dim / Q6K_BLOCK_ELEM;
        let mut rng = Rng::new(5150);
        let mut row = Vec::new();
        for _ in 0..nb {
            let codes = random_codes(&mut rng);
            let scales = random_scales(&mut rng);
            row.extend_from_slice(&pack_block(&codes, &scales, 0.015625));
        }
        let mut q8_1 = vec![0u8; (in_dim / 32) * 36];
        for b in 0..(in_dim / 32) {
            let base = b * 36;
            q8_1[base..base + 2].copy_from_slice(&f32_to_f16_bits_rne(0.01).to_le_bytes());
            for k in 0..32 {
                q8_1[base + 4 + k] = (rng.below(255) as i32 - 127) as i8 as u8;
            }
        }
        let clean = row_dot_q8_1_kernel_order(&row, &q8_1, in_dim);
        for b in 0..(in_dim / 32) {
            q8_1[b * 36 + 2] = 0x7B;
            q8_1[b * 36 + 3] = 0x5C;
        }
        let poisoned = row_dot_q8_1_kernel_order(&row, &q8_1, in_dim);
        assert_eq!(
            clean, poisoned,
            "the Q8_1 sum field must not affect a Q6_K dot"
        );
    }

    /// G1 (`LUMEN_CUDA_GDN_AB_FUSED`): the fused alpha+beta kernel must be
    /// BIT-IDENTICAL to the single-matrix kernel it replaces, not merely close.
    ///
    /// The claim in the flag's docs is that the fusion changes only the host's
    /// launch count -- `blockIdx.y` selects the (weight, destination) pair and
    /// every other line is verbatim `mul_mat_vec_q_q8_0`. That is a claim about
    /// SOURCE, and it is checkable without a GPU: extract both kernel bodies and
    /// require the fused body to reduce to the single body once the stream
    /// selection is removed. If someone "optimizes" the fused kernel by
    /// interleaving the two accumulations, the per-row op order changes, the last
    /// bit moves, and the golden anchors are no longer guaranteed -- and this test
    /// fails, forcing the reassociation to be argued explicitly.
    ///
    /// Negative-verified: perturbing one arithmetic line in the fused body (e.g.
    /// hoisting `d8_0 * d8_1`) breaks the equality.
    #[test]
    fn gdn_ab_fused_kernel_is_verbatim_the_single_matrix_kernel() {
        const SRC: &str = include_str!("cuda/shaders/mmv_q.cu");

        fn body_of(src: &str, sym: &str) -> String {
            let at = src
                .find(&format!("void {sym}("))
                .unwrap_or_else(|| panic!("kernel {sym} not found in mmv_q.cu"));
            // From the end of the signature to the closing brace at column 0.
            let open = src[at..].find('{').expect("kernel body") + at;
            let end = src[open..]
                .find("\n}\n")
                .expect("kernel closing brace at column 0")
                + open;
            src[open..end].to_string()
        }

        // Compare only the ARITHMETIC lines, so comments and the stream-selection
        // preamble cannot mask or fake a match.
        fn arith(body: &str) -> Vec<String> {
            body.lines()
                // Strip TRAILING comments too, not just comment-only lines: the
                // single-matrix kernel annotates several lines (`// rpb=1`,
                // `// [0..128)`) and the fused copy does not. A first cut of this
                // test compared raw lines and failed on exactly that -- the test
                // being over-strict about formatting, not the kernel differing.
                .map(|l| match l.find("//") {
                    Some(i) => l[..i].trim(),
                    None => l.trim(),
                })
                .filter(|l| {
                    !l.is_empty()
                        && !l.contains("blockIdx.y")
                        && !l.contains("__restrict__ vx =")
                        && !l.contains("__restrict__ dst =")
                })
                .map(|l| l.to_string())
                .collect()
        }

        let single = arith(&body_of(SRC, "mul_mat_vec_q_q8_0"));
        let fused = arith(&body_of(SRC, "mul_mat_vec_q_q8_0_ab_fused"));

        assert!(
            !single.is_empty() && !fused.is_empty(),
            "failed to extract kernel bodies"
        );
        assert_eq!(
            fused, single,
            "the fused alpha+beta kernel is no longer verbatim mul_mat_vec_q_q8_0 \
             (ignoring the blockIdx.y stream selection). Any arithmetic difference \
             REASSOCIATES the per-row reduction, so the bit-identity claim in \
             `runtime_defaults::gdn_ab_fused` no longer holds and DET / the golden \
             anchors must be re-argued rather than assumed."
        );

        // The fused kernel must actually take TWO weight and TWO dst pointers, or
        // it is not fusing anything.
        let sig_at = SRC.find("void mul_mat_vec_q_q8_0_ab_fused(").unwrap();
        // Slice to the opening BRACE, not to the first ')': the parameter comments
        // contain parentheses ("stream A (alpha)"), and stopping at the first ')'
        // truncated the signature before `vx_b` -- another instance of the test
        // being wrong about text rather than the kernel being wrong.
        let sig_end = sig_at + SRC[sig_at..].find('{').unwrap();
        let sig = &SRC[sig_at..sig_end];
        for param in ["vx_a", "vx_b", "dst_a", "dst_b", "vy"] {
            assert!(sig.contains(param), "fused signature is missing `{param}`");
        }
    }

    /// ARCH-CLASS INVARIANT: every kernel's module CONTENT must be legal at its
    /// loader's minimum architecture.
    ///
    /// THIS IS THE TEST THAT WOULD HAVE CAUGHT THE A100 FAILURE. The dp4a kernel
    /// was appended to matvec_q6_k_f32.cu to share its verified lane mapping and
    /// `.rn` primitives. But NVRTC compiles a TRANSLATION UNIT, not a kernel, and
    /// the F32 kernels load through the plain `load_fn`, which passes no
    /// `--gpu-architecture` and so targets a default below sm_61. One
    /// `dp4a.s32.s32` anywhere in that file made the WHOLE module fail with
    /// CUDA_ERROR_INVALID_PTX, so `matvec_q6_k_f32`, `matvec_q6_k_f32_nr4` AND
    /// `dequant_q6_k_to_f32` all became `None` -- C1 and C3 silently disabled.
    ///
    /// The sibling symbol-existence test cannot see this: every symbol was present
    /// and correctly named. The file simply could not be compiled by its loader.
    /// Arch requirement is a property of the FILE, so that is what this checks.
    ///
    /// It also caught its own first draft: matching the bare string "dp4a" tripped
    /// on a comment that said "No dp4a, no mma", so the patterns below are PTX
    /// mnemonics with their separator, not prose.
    #[test]
    fn kernel_module_content_matches_its_loader_min_arch() {
        // (file, source, loader, min-arch class). Only the shaders this module
        // owns; the rest of the directory has its own loader assignments.
        struct Shader {
            file: &'static str,
            src: &'static str,
            loader: &'static str,
            allows_raised: bool,
        }
        let shaders = [
            Shader {
                file: "matvec_q6_k_f32.cu",
                src: include_str!("cuda/shaders/matvec_q6_k_f32.cu"),
                loader: "load_fn (plain, no --gpu-architecture)",
                allows_raised: false,
            },
            Shader {
                file: "matvec_q6_k_q8_1.cu",
                src: include_str!("cuda/shaders/matvec_q6_k_q8_1.cu"),
                loader: "load_fn_sm80_fast_math (compute_80)",
                allows_raised: true,
            },
        ];
        // PTX mnemonics that require a raised arch, with the loader that provides it.
        let raised: [(&str, &str); 4] = [
            ("dp4a.", "sm_61+ -> load_fn_sm61 / load_fn_sm80*"),
            ("mma.sync", "sm_80+ -> load_fn_sm80*"),
            (
                "cvt.rn.bf16.f32",
                "sm_80+ -> load_fn_sm80*, or a __CUDA_ARCH__ guard",
            ),
            ("__dp4a(", "intrinsic NVRTC cannot resolve without headers"),
        ];

        for sh in &shaders {
            for (opcode, why) in &raised {
                let present = sh.src.contains(opcode);
                if !sh.allows_raised {
                    assert!(
                        !present,
                        "{} is loaded via {} but contains `{}`, which needs {}. NVRTC \
                         compiles the whole translation unit, so this makes EVERY kernel \
                         in the file fail to load -- not just the one using it. Move the \
                         offending kernel into its own .cu file.",
                        sh.file, sh.loader, opcode, why
                    );
                }
            }
        }

        // Positive half: the dp4a file must actually still contain its opcode, or
        // the split silently degraded into a plain kernel that only looks correct.
        let dp4a_src = shaders[1].src;
        assert!(
            dp4a_src.contains("dp4a."),
            "matvec_q6_k_q8_1.cu must contain the dp4a opcode it exists for"
        );
        // And the two files must not have drifted apart on the decode constants
        // they both hard-code.
        for c in ["Q6K_BLOCK_ELEM   256", "Q6K_BLOCK_BYTE   210"] {
            assert!(
                shaders[0].src.contains(c) && dp4a_src.contains(c),
                "both Q6_K shaders must agree on `{c}`"
            );
        }
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
