//! Lossless repack of Q4_0 weights into a tensor-core tile-major layout.
//!
//! The packed form feeds the SM80 W4A16 decode kernels: two separate planes —
//! a nibble plane in fragment-native K16×N64 tiles and an FP16 scale plane —
//! carrying the SOURCE BITS VERBATIM. No dequantization or requantization
//! happens here: every 4-bit quant and every 16-bit scale pattern (including
//! signed zero, subnormals and non-finite bits) round-trips exactly, which is
//! what makes the layout eligible as a pure-layout lever with the quantization
//! unchanged.
//!
//! Source contract (matches `dequant_q4_0_f16.cu` and the split-repack path):
//! row-major `[N, K]`, one 18-byte block per 32 input columns:
//!   bytes 0..2  = little-endian FP16 scale bits
//!   bytes 2..18 = 16 packed bytes; for r in 0..16:
//!     q(n, 32g + r)      = byte[2+r] & 0x0F
//!     q(n, 32g + 16 + r) = byte[2+r] >> 4
//!   value = fp16(scale * (q - 8))
//!
//! Packed layout (kernel-facing, `Q[k][n] = q(n, k)`):
//! * Nibble plane: K16×N64 tiles, tile_id = (k0/16)*(N/64) + (n0/64); each
//!   tile is 128 u32 words. Word `warp*32 + lane` (warp 0..4, lane 0..32) holds
//!   the eight quants of one `mma.sync.m16n8k16` A-operand fragment slot, with
//!   c = lane/4, r = 2*(lane%4), n = n0 + 16*warp + c:
//!     v = [Q[k0+r][n],   Q[k0+r+1][n],   Q[k0+r+8][n],   Q[k0+r+9][n],
//!          Q[k0+r][n+8], Q[k0+r+1][n+8], Q[k0+r+8][n+8], Q[k0+r+9][n+8]]
//!   packed in nibble order [0,2,4,6,1,3,5,7] (low nibble first), so the
//!   in-kernel LOP3 expansion yields both halves of the fragment directly.
//!   Warp-major word order makes the consuming warp's 32 loads hit 32
//!   distinct shared-memory banks (lane-major order serialized 4-way).
//!   This permutation is derived from the Marlin kernel's packing
//!   (IST-DASLab/marlin and the vLLM repacker, both Apache-2.0).
//! * Scale plane: logical `D[g][n]` (g = k/32, bits verbatim), stored g-major;
//!   within every 64-column chunk the 8×8 block is transposed
//!   (`phys[8i + j] = logical[i + 8j]`) to match the ldmatrix lane mapping.
//!
//! Group-32 scales are kept as-is — the layout does NOT assume Marlin's
//! group-128 grouping.

/// One Q4_0 tensor repacked into the two-plane tile-major layout.
pub struct MarlinQ4Packed {
    /// Output dimension (source rows). Must be a multiple of 64.
    pub n: usize,
    /// Input dimension (source columns). Must be a multiple of 32.
    pub k: usize,
    /// Nibble plane: (K/16)*(N/64) tiles × 128 words. Length = N*K/8.
    pub q_words: Vec<u32>,
    /// Scale plane: raw FP16 bit patterns, length = (K/32)*N.
    pub scale_bits: Vec<u16>,
}

const BLOCK_BYTES: usize = 18;

/// Read quant q(n, k) from the source Q4_0 bytes.
#[inline]
fn src_quant(src: &[u8], nb: usize, n: usize, k: usize) -> u32 {
    let g = k / 32;
    let r = k % 32;
    let block = &src[(n * nb + g) * BLOCK_BYTES..];
    if r < 16 {
        (block[2 + r] & 0x0F) as u32
    } else {
        (block[2 + (r - 16)] >> 4) as u32
    }
}

/// Fragment slot offsets within one K16 column pair: (dk, dn) per v index.
const FRAG_OFFSETS: [(usize, usize); 8] = [
    (0, 0),
    (1, 0),
    (8, 0),
    (9, 0),
    (0, 8),
    (1, 8),
    (8, 8),
    (9, 8),
];

/// Nibble order within the packed word: v[ORDER[b]] lives at bits 4b..4b+4.
const NIBBLE_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Repack a row-major `[N, K]` Q4_0 tensor. Pure byte shuffle — lossless by
/// construction; `unpack` is the exact inverse.
pub fn pack_q4_marlin(src: &[u8], n: usize, k: usize) -> Result<MarlinQ4Packed, String> {
    if n == 0 || k == 0 || n % 64 != 0 || k % 32 != 0 {
        return Err(format!(
            "marlin repack needs N%64==0 && K%32==0, got N={n} K={k}"
        ));
    }
    let nb = k / 32;
    let expect = n * nb * BLOCK_BYTES;
    if src.len() != expect {
        return Err(format!(
            "marlin repack: source is {} bytes, N={n} K={k} needs {expect}",
            src.len()
        ));
    }

    let mut q_words = vec![0u32; n * k / 8];
    for k0 in (0..k).step_by(16) {
        for n0 in (0..n).step_by(64) {
            let tile_id = (k0 / 16) * (n / 64) + (n0 / 64);
            let tile = &mut q_words[tile_id * 128..(tile_id + 1) * 128];
            for warp in 0..4 {
                for lane in 0..32 {
                    let c = lane / 4;
                    let r = 2 * (lane % 4);
                    let col = n0 + 16 * warp + c;
                    let mut word = 0u32;
                    for (b, &vi) in NIBBLE_ORDER.iter().enumerate() {
                        let (dk, dn) = FRAG_OFFSETS[vi];
                        let q = src_quant(src, nb, col + dn, k0 + r + dk);
                        word |= q << (4 * b);
                    }
                    tile[warp * 32 + lane] = word;
                }
            }
        }
    }

    let mut scale_bits = vec![0u16; nb * n];
    for g in 0..nb {
        for chunk in 0..n / 64 {
            for p in 0..64 {
                let (i, j) = (p % 8, p / 8);
                let col = chunk * 64 + p;
                let block = &src[(col * nb + g) * BLOCK_BYTES..];
                let bits = u16::from_le_bytes([block[0], block[1]]);
                scale_bits[g * n + chunk * 64 + 8 * i + j] = bits;
            }
        }
    }

    Ok(MarlinQ4Packed {
        n,
        k,
        q_words,
        scale_bits,
    })
}

/// Exact inverse of `pack_q4_marlin`: reconstructs the original row-major
/// Q4_0 bytes bit-for-bit. This is the P1 oracle — every repacked tensor can
/// be verified against its mmap'd source before the layout is trusted.
pub fn unpack_q4_marlin(p: &MarlinQ4Packed) -> Vec<u8> {
    let (n, k) = (p.n, p.k);
    let nb = k / 32;
    let mut out = vec![0u8; n * nb * BLOCK_BYTES];

    for g in 0..nb {
        for chunk in 0..n / 64 {
            for p_log in 0..64 {
                let (i, j) = (p_log % 8, p_log / 8);
                let col = chunk * 64 + p_log;
                let bits = p.scale_bits[g * n + chunk * 64 + 8 * i + j];
                let block = &mut out[(col * nb + g) * BLOCK_BYTES..];
                block[..2].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }

    for k0 in (0..k).step_by(16) {
        for n0 in (0..n).step_by(64) {
            let tile_id = (k0 / 16) * (n / 64) + (n0 / 64);
            let tile = &p.q_words[tile_id * 128..(tile_id + 1) * 128];
            for warp in 0..4 {
                for lane in 0..32 {
                    let c = lane / 4;
                    let r = 2 * (lane % 4);
                    let col = n0 + 16 * warp + c;
                    let word = tile[warp * 32 + lane];
                    for (b, &vi) in NIBBLE_ORDER.iter().enumerate() {
                        let (dk, dn) = FRAG_OFFSETS[vi];
                        let q = ((word >> (4 * b)) & 0xF) as u8;
                        let kk = k0 + r + dk;
                        let (row, g2, r2) = (col + dn, kk / 32, kk % 32);
                        let block = &mut out[(row * nb + g2) * BLOCK_BYTES..];
                        if r2 < 16 {
                            block[2 + r2] |= q;
                        } else {
                            block[2 + (r2 - 16)] |= q << 4;
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic byte stream (no RNG dependency in the test suite).
    fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn src_tensor(seed: u64, n: usize, k: usize) -> Vec<u8> {
        lcg_bytes(seed, n * (k / 32) * BLOCK_BYTES)
    }

    #[test]
    fn round_trip_exact_various_shapes() {
        for (seed, n, k) in [
            (1u64, 64, 32),
            (2, 64, 128),
            (3, 128, 128),
            (4, 192, 160),
            (5, 256, 544), // K%128 != 0 — repack itself only needs K%32
        ] {
            let src = src_tensor(seed, n, k);
            let packed = pack_q4_marlin(&src, n, k).unwrap();
            assert_eq!(packed.q_words.len(), n * k / 8);
            assert_eq!(packed.scale_bits.len(), (k / 32) * n);
            assert_eq!(
                unpack_q4_marlin(&packed),
                src,
                "round-trip mismatch N={n} K={k}"
            );
        }
    }

    #[test]
    fn round_trip_preserves_all_nibbles_and_hostile_scale_bits() {
        // Every block cycles through all 16 quant values, and the scale plane
        // walks the hostile FP16 patterns: ±0, subnormal, max finite, inf, NaN.
        let (n, k) = (64, 64);
        let nb = k / 32;
        let hostile: [u16; 8] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x7C00, 0x7E00,
        ];
        let mut src = vec![0u8; n * nb * BLOCK_BYTES];
        for row in 0..n {
            for g in 0..nb {
                let block = &mut src[(row * nb + g) * BLOCK_BYTES..][..BLOCK_BYTES];
                block[..2].copy_from_slice(&hostile[(row * nb + g) % hostile.len()].to_le_bytes());
                for r in 0..16 {
                    let lo = (row + g + r) % 16;
                    let hi = (row + g + r + 7) % 16;
                    block[2 + r] = (lo as u8) | ((hi as u8) << 4);
                }
            }
        }
        let packed = pack_q4_marlin(&src, n, k).unwrap();
        assert_eq!(unpack_q4_marlin(&packed), src);
        // Hostile bit patterns must appear verbatim in the packed scale plane.
        for h in hostile {
            assert!(
                packed.scale_bits.contains(&h),
                "scale bits {h:#06x} not preserved"
            );
        }
    }

    #[test]
    fn packed_words_cover_every_quant_exactly_once() {
        // Structural check independent of the inverse: with q(n,k) chosen as a
        // (k,n)-dependent pattern, decoding every packed word through the
        // documented fragment mapping must visit each (k,n) slot exactly once
        // with the right value.
        let (n, k) = (128, 64);
        let nb = k / 32;
        let q_of = |row: usize, kk: usize| ((row * 17 + kk * 31) % 16) as u32;
        let mut src = vec![0u8; n * nb * BLOCK_BYTES];
        for row in 0..n {
            for g in 0..nb {
                let block = &mut src[(row * nb + g) * BLOCK_BYTES..][..BLOCK_BYTES];
                for r in 0..16 {
                    let lo = q_of(row, 32 * g + r) as u8;
                    let hi = q_of(row, 32 * g + 16 + r) as u8;
                    block[2 + r] = lo | (hi << 4);
                }
            }
        }
        let packed = pack_q4_marlin(&src, n, k).unwrap();

        let mut seen = vec![false; n * k];
        for k0 in (0..k).step_by(16) {
            for n0 in (0..n).step_by(64) {
                let tile_id = (k0 / 16) * (n / 64) + (n0 / 64);
                for warp in 0..4 {
                    for lane in 0..32 {
                        let c = lane / 4;
                        let r = 2 * (lane % 4);
                        let word = packed.q_words[tile_id * 128 + warp * 32 + lane];
                        for (b, &vi) in NIBBLE_ORDER.iter().enumerate() {
                            let (dk, dn) = FRAG_OFFSETS[vi];
                            let (row, kk) = (n0 + 16 * warp + c + dn, k0 + r + dk);
                            assert_eq!(
                                (word >> (4 * b)) & 0xF,
                                q_of(row, kk),
                                "wrong quant at n={row} k={kk}"
                            );
                            assert!(!seen[row * k + kk], "slot n={row} k={kk} packed twice");
                            seen[row * k + kk] = true;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "some (n,k) slot never packed");
    }

    #[test]
    fn single_nibble_lands_at_hand_computed_position() {
        // Delta probe cross-checked against a by-hand evaluation of the layout
        // formula (not the code): q(n=9, k=25)=0xF with all else zero.
        // k=25 -> k0=16, r+dk must give 9: lane%4=0 => r=0? need r+dk=9 =>
        // (r=0,dk=9) v-index 3 -> nibble order position b where ORDER[b]=3 -> b=5.
        // n=9 -> n0=0, 16*warp+c+dn=9 => warp=0, c=1, dn=8 ... but v=3 has dn=0,
        // so instead c=9? c max 7 -> warp=0 impossible; hence (r=0,dk=9,dn=0)
        // needs c=9 -> invalid; the valid decomposition is dn=8, v-index 7
        // (dk=9,dn=8) -> ORDER position b=7, c=1, lane=4*(1? ) ... lane/4=c=1,
        // lane%4=0 -> lane=4, warp=0. tile_id=(16/16)*(64/64)+0=1.
        // word index = tile*128 + warp*32 + lane = 128 + 4. bits 28..32.
        let (n, k) = (64, 32);
        let nb = k / 32;
        let mut src = vec![0u8; n * nb * BLOCK_BYTES];
        // k=25: r'=25 -> high nibble of byte[2+9] in row 9's only block.
        src[(9 * nb) * BLOCK_BYTES + 2 + 9] = 0xF0;
        let packed = pack_q4_marlin(&src, n, k).unwrap();
        for (idx, &w) in packed.q_words.iter().enumerate() {
            if idx == 128 + 4 {
                assert_eq!(w, 0xF000_0000, "delta nibble misplaced within word");
            } else {
                assert_eq!(w, 0, "unexpected nonzero word at {idx}");
            }
        }
    }

    #[test]
    fn scale_plane_transpose_matches_hand_positions() {
        // D[g][n] with distinguishable bits: g*1000 + n. For chunk 0,
        // logical p = i + 8j stores at phys 8i + j: n=1 (i=1,j=0) -> phys 8;
        // n=8 (i=0,j=1) -> phys 1; n=63 (i=7,j=7) -> phys 63.
        let (n, k) = (64, 64);
        let nb = k / 32;
        let mut src = vec![0u8; n * nb * BLOCK_BYTES];
        for row in 0..n {
            for g in 0..nb {
                let bits = (g * 1000 + row) as u16;
                src[(row * nb + g) * BLOCK_BYTES..][..2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let packed = pack_q4_marlin(&src, n, k).unwrap();
        assert_eq!(packed.scale_bits[8], 1);
        assert_eq!(packed.scale_bits[1], 8);
        assert_eq!(packed.scale_bits[63], 63);
        assert_eq!(packed.scale_bits[n + 8], 1001); // g=1 plane offset
    }

    #[test]
    fn rejects_bad_shapes_and_sizes() {
        assert!(pack_q4_marlin(&[], 0, 0).is_err());
        assert!(pack_q4_marlin(&[0; 18 * 32], 32, 32).is_err()); // N%64
        assert!(pack_q4_marlin(&[0; 100], 64, 48).is_err()); // K%32
        assert!(pack_q4_marlin(&[0; 100], 64, 32).is_err()); // size mismatch
    }
}
