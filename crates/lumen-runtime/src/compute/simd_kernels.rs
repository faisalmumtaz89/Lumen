//! SIMD-optimized kernel primitives using aarch64 NEON intrinsics.
//!
//! Each function provides a safe wrapper around unsafe NEON intrinsics.
//! On non-aarch64 targets, all functions fall back to the naive implementations.
//!
//! Design: 4-accumulator pattern for ILP on Apple M4 (4 FMA units in flight).
//! Primary loop processes 16 f32s per iteration (4 NEON registers x 4 lanes).
//! Secondary loop processes 4 f32s. Scalar tail handles remainder.

// ---- aarch64 NEON implementation ----

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Matrix-vector multiply reading weights from LE bytes.
/// out = W * x, where W is row-major [out_dim, in_dim] stored as LE f32 bytes.
///
/// Hot path (~90% of compute). Uses 4 independent NEON accumulators for:
/// - 4x instruction-level parallelism (M4 can issue 4 FMA in flight)
/// - Better precision via independent partial sums (reduces rounding error)
/// - ~1.5-2x speedup for large dimensions vs single-accumulator
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn matmul_bytes_simd(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(w_bytes.len() >= out_dim * in_dim * 4);
    assert!(x.len() >= in_dim);

    let chunks16 = in_dim / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (in_dim - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = in_dim - tail_start;

    for i in 0..out_dim {
        let row_byte_start = i * in_dim * 4;

        // SAFETY: Bounds verified by assertions above. NEON always available on aarch64.
        // 4-accumulator pattern: acc0..acc3 process 16 f32s (64 bytes) per iteration.
        // vld1q_u8 loads 16 bytes (4 LE f32s), vreinterpretq reinterprets as f32x4.
        // vfmaq_f32 is fused multiply-add. All loads within w_bytes/x bounds.
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            // Primary loop: 16 floats (64 bytes of weights) per iteration
            for c in 0..chunks16 {
                let base_byte = row_byte_start + c * 64;
                let base_x = c * 16;

                // Prefetch next chunk of weight data (64 bytes ahead)
                if c + 1 < chunks16 {
                    let next_byte = row_byte_start + (c + 1) * 64;
                    std::arch::asm!(
                        "prfm pldl1strm, [{addr}]",
                        addr = in(reg) w_bytes.as_ptr().add(next_byte),
                        options(nostack, readonly, preserves_flags),
                    );
                }

                let w0 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte)));
                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                acc0 = vfmaq_f32(acc0, w0, x0);

                let w1 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 16)));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                acc1 = vfmaq_f32(acc1, w1, x1);

                let w2 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 32)));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                acc2 = vfmaq_f32(acc2, w2, x2);

                let w3 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 48)));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));
                acc3 = vfmaq_f32(acc3, w3, x3);
            }

            // Secondary loop: 4 floats per iteration for remainder
            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let byte_offset = row_byte_start + idx * 4;
                let w_vec = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(byte_offset)));
                let x_vec = vld1q_f32(x.as_ptr().add(idx));
                acc0 = vfmaq_f32(acc0, w_vec, x_vec);
            }

            // Reduce 4 accumulators to scalar
            // Pairwise add for better precision: (acc0+acc1) + (acc2+acc3)
            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut sum = vaddvq_f32(sum_all);

            // Scalar tail
            for j in 0..remainder {
                let idx = tail_start + j;
                let byte_offset = row_byte_start + idx * 4;
                let w_val = f32::from_le_bytes([
                    *w_bytes.get_unchecked(byte_offset),
                    *w_bytes.get_unchecked(byte_offset + 1),
                    *w_bytes.get_unchecked(byte_offset + 2),
                    *w_bytes.get_unchecked(byte_offset + 3),
                ]);
                sum += w_val * x.get_unchecked(idx);
            }

            *out.get_unchecked_mut(i) = sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn matmul_bytes_simd(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_bytes_fallback(out, w_bytes, x, out_dim, in_dim);
}

/// Two-row matmul variant: processes 2 output rows simultaneously.
/// Amortizes the cost of loading x from memory -- both rows read the same x vector.
///
/// Falls back to single-row `matmul_bytes_simd` for the odd last row when out_dim is odd.
/// Significant speedup for Q/K/V projections where out_dim >> 1.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn matmul_bytes_simd_2row(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(w_bytes.len() >= out_dim * in_dim * 4);
    assert!(x.len() >= in_dim);

    let row_pairs = out_dim / 2;
    let chunks16 = in_dim / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (in_dim - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = in_dim - tail_start;

    for p in 0..row_pairs {
        let row0 = p * 2;
        let row1 = row0 + 1;
        let row0_byte_start = row0 * in_dim * 4;
        let row1_byte_start = row1 * in_dim * 4;

        // SAFETY: Bounds verified by assertions. Both rows share the same x loads.
        unsafe {
            // Row 0 accumulators
            let mut r0_acc0 = vdupq_n_f32(0.0);
            let mut r0_acc1 = vdupq_n_f32(0.0);
            // Row 1 accumulators
            let mut r1_acc0 = vdupq_n_f32(0.0);
            let mut r1_acc1 = vdupq_n_f32(0.0);

            // Primary loop: load x once, use for both rows
            for c in 0..chunks16 {
                let base_x = c * 16;

                // Prefetch next chunk of weight data for both rows (64 bytes ahead)
                if c + 1 < chunks16 {
                    let next0 = row0_byte_start + (c + 1) * 64;
                    let next1 = row1_byte_start + (c + 1) * 64;
                    std::arch::asm!(
                        "prfm pldl1strm, [{r0}]",
                        "prfm pldl1strm, [{r1}]",
                        r0 = in(reg) w_bytes.as_ptr().add(next0),
                        r1 = in(reg) w_bytes.as_ptr().add(next1),
                        options(nostack, readonly, preserves_flags),
                    );
                }

                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));

                let base0 = row0_byte_start + c * 64;
                let w0_0 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base0)));
                let w0_1 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base0 + 16)));
                let w0_2 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base0 + 32)));
                let w0_3 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base0 + 48)));
                r0_acc0 = vfmaq_f32(r0_acc0, w0_0, x0);
                r0_acc0 = vfmaq_f32(r0_acc0, w0_1, x1);
                r0_acc1 = vfmaq_f32(r0_acc1, w0_2, x2);
                r0_acc1 = vfmaq_f32(r0_acc1, w0_3, x3);

                let base1 = row1_byte_start + c * 64;
                let w1_0 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base1)));
                let w1_1 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base1 + 16)));
                let w1_2 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base1 + 32)));
                let w1_3 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base1 + 48)));
                r1_acc0 = vfmaq_f32(r1_acc0, w1_0, x0);
                r1_acc0 = vfmaq_f32(r1_acc0, w1_1, x1);
                r1_acc1 = vfmaq_f32(r1_acc1, w1_2, x2);
                r1_acc1 = vfmaq_f32(r1_acc1, w1_3, x3);
            }

            // Secondary 4-at-a-time loop
            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let x_vec = vld1q_f32(x.as_ptr().add(idx));

                let b0 = row0_byte_start + idx * 4;
                let w0 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(b0)));
                r0_acc0 = vfmaq_f32(r0_acc0, w0, x_vec);

                let b1 = row1_byte_start + idx * 4;
                let w1 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(b1)));
                r1_acc0 = vfmaq_f32(r1_acc0, w1, x_vec);
            }

            // Reduce accumulators
            let r0_sum_vec = vaddq_f32(r0_acc0, r0_acc1);
            let mut sum0 = vaddvq_f32(r0_sum_vec);
            let r1_sum_vec = vaddq_f32(r1_acc0, r1_acc1);
            let mut sum1 = vaddvq_f32(r1_sum_vec);

            // Scalar tail
            for j in 0..remainder {
                let idx = tail_start + j;
                let x_val = *x.get_unchecked(idx);

                let b0 = row0_byte_start + idx * 4;
                let w0_val = f32::from_le_bytes([
                    *w_bytes.get_unchecked(b0),
                    *w_bytes.get_unchecked(b0 + 1),
                    *w_bytes.get_unchecked(b0 + 2),
                    *w_bytes.get_unchecked(b0 + 3),
                ]);
                sum0 += w0_val * x_val;

                let b1 = row1_byte_start + idx * 4;
                let w1_val = f32::from_le_bytes([
                    *w_bytes.get_unchecked(b1),
                    *w_bytes.get_unchecked(b1 + 1),
                    *w_bytes.get_unchecked(b1 + 2),
                    *w_bytes.get_unchecked(b1 + 3),
                ]);
                sum1 += w1_val * x_val;
            }

            *out.get_unchecked_mut(row0) = sum0;
            *out.get_unchecked_mut(row1) = sum1;
        }
    }

    // Handle odd last row with single-row path
    if out_dim % 2 != 0 {
        let last = out_dim - 1;
        let row_byte_start = last * in_dim * 4;

        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            for c in 0..chunks16 {
                let base_byte = row_byte_start + c * 64;
                let base_x = c * 16;

                let w0 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte)));
                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                acc0 = vfmaq_f32(acc0, w0, x0);

                let w1 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 16)));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                acc1 = vfmaq_f32(acc1, w1, x1);

                let w2 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 32)));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                acc2 = vfmaq_f32(acc2, w2, x2);

                let w3 = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(base_byte + 48)));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));
                acc3 = vfmaq_f32(acc3, w3, x3);
            }

            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let byte_offset = row_byte_start + idx * 4;
                let w_vec = vreinterpretq_f32_u8(vld1q_u8(w_bytes.as_ptr().add(byte_offset)));
                let x_vec = vld1q_f32(x.as_ptr().add(idx));
                acc0 = vfmaq_f32(acc0, w_vec, x_vec);
            }

            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut sum = vaddvq_f32(sum_all);

            for j in 0..remainder {
                let idx = tail_start + j;
                let byte_offset = row_byte_start + idx * 4;
                let w_val = f32::from_le_bytes([
                    *w_bytes.get_unchecked(byte_offset),
                    *w_bytes.get_unchecked(byte_offset + 1),
                    *w_bytes.get_unchecked(byte_offset + 2),
                    *w_bytes.get_unchecked(byte_offset + 3),
                ]);
                sum += w_val * x.get_unchecked(idx);
            }

            *out.get_unchecked_mut(last) = sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn matmul_bytes_simd_2row(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_bytes_fallback(out, w_bytes, x, out_dim, in_dim);
}

/// Matrix-vector multiply with weights already as f32 slice.
/// out = W * x, where W is row-major [out_dim, in_dim].
///
/// 4-accumulator pattern identical to matmul_bytes_simd.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn matmul_simd(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(w.len() >= out_dim * in_dim);
    assert!(x.len() >= in_dim);

    let chunks16 = in_dim / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (in_dim - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = in_dim - tail_start;

    for i in 0..out_dim {
        let row_start = i * in_dim;

        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            for c in 0..chunks16 {
                let base_w = row_start + c * 16;
                let base_x = c * 16;

                let w0 = vld1q_f32(w.as_ptr().add(base_w));
                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                acc0 = vfmaq_f32(acc0, w0, x0);

                let w1 = vld1q_f32(w.as_ptr().add(base_w + 4));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                acc1 = vfmaq_f32(acc1, w1, x1);

                let w2 = vld1q_f32(w.as_ptr().add(base_w + 8));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                acc2 = vfmaq_f32(acc2, w2, x2);

                let w3 = vld1q_f32(w.as_ptr().add(base_w + 12));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));
                acc3 = vfmaq_f32(acc3, w3, x3);
            }

            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let w_vec = vld1q_f32(w.as_ptr().add(row_start + idx));
                let x_vec = vld1q_f32(x.as_ptr().add(idx));
                acc0 = vfmaq_f32(acc0, w_vec, x_vec);
            }

            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut sum = vaddvq_f32(sum_all);

            for j in 0..remainder {
                let idx = tail_start + j;
                sum += w.get_unchecked(row_start + idx) * x.get_unchecked(idx);
            }

            *out.get_unchecked_mut(i) = sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn matmul_simd(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_fallback(out, w, x, out_dim, in_dim);
}

/// Two-row matmul variant for f32 weights: processes 2 output rows simultaneously.
/// Amortizes the cost of loading x from memory -- both rows read the same x vector.
///
/// Falls back to single-row pattern for the odd last row when out_dim is odd.
/// Same 4-accumulator ILP pattern as matmul_simd, extended to 2 rows sharing x loads.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn matmul_simd_2row(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(w.len() >= out_dim * in_dim);
    assert!(x.len() >= in_dim);

    let row_pairs = out_dim / 2;
    let chunks16 = in_dim / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (in_dim - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = in_dim - tail_start;

    for p in 0..row_pairs {
        let row0 = p * 2;
        let row1 = row0 + 1;
        let row0_start = row0 * in_dim;
        let row1_start = row1 * in_dim;

        // SAFETY: Bounds verified by assertions. Both rows share the same x loads.
        unsafe {
            // Row 0 accumulators (2 for paired reduction, matching 2row bytes pattern)
            let mut r0_acc0 = vdupq_n_f32(0.0);
            let mut r0_acc1 = vdupq_n_f32(0.0);
            // Row 1 accumulators
            let mut r1_acc0 = vdupq_n_f32(0.0);
            let mut r1_acc1 = vdupq_n_f32(0.0);

            // Primary loop: load x once, use for both rows
            for c in 0..chunks16 {
                let base_x = c * 16;

                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));

                let base0 = row0_start + c * 16;
                r0_acc0 = vfmaq_f32(r0_acc0, vld1q_f32(w.as_ptr().add(base0)), x0);
                r0_acc0 = vfmaq_f32(r0_acc0, vld1q_f32(w.as_ptr().add(base0 + 4)), x1);
                r0_acc1 = vfmaq_f32(r0_acc1, vld1q_f32(w.as_ptr().add(base0 + 8)), x2);
                r0_acc1 = vfmaq_f32(r0_acc1, vld1q_f32(w.as_ptr().add(base0 + 12)), x3);

                let base1 = row1_start + c * 16;
                r1_acc0 = vfmaq_f32(r1_acc0, vld1q_f32(w.as_ptr().add(base1)), x0);
                r1_acc0 = vfmaq_f32(r1_acc0, vld1q_f32(w.as_ptr().add(base1 + 4)), x1);
                r1_acc1 = vfmaq_f32(r1_acc1, vld1q_f32(w.as_ptr().add(base1 + 8)), x2);
                r1_acc1 = vfmaq_f32(r1_acc1, vld1q_f32(w.as_ptr().add(base1 + 12)), x3);
            }

            // Secondary 4-at-a-time loop
            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let x_vec = vld1q_f32(x.as_ptr().add(idx));

                r0_acc0 = vfmaq_f32(r0_acc0, vld1q_f32(w.as_ptr().add(row0_start + idx)), x_vec);
                r1_acc0 = vfmaq_f32(r1_acc0, vld1q_f32(w.as_ptr().add(row1_start + idx)), x_vec);
            }

            // Reduce accumulators
            let r0_sum_vec = vaddq_f32(r0_acc0, r0_acc1);
            let mut sum0 = vaddvq_f32(r0_sum_vec);
            let r1_sum_vec = vaddq_f32(r1_acc0, r1_acc1);
            let mut sum1 = vaddvq_f32(r1_sum_vec);

            // Scalar tail
            for j in 0..remainder {
                let idx = tail_start + j;
                let x_val = *x.get_unchecked(idx);
                sum0 += *w.get_unchecked(row0_start + idx) * x_val;
                sum1 += *w.get_unchecked(row1_start + idx) * x_val;
            }

            *out.get_unchecked_mut(row0) = sum0;
            *out.get_unchecked_mut(row1) = sum1;
        }
    }

    // Handle odd last row with single-row path
    if out_dim % 2 != 0 {
        let last = out_dim - 1;
        let row_start = last * in_dim;

        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            for c in 0..chunks16 {
                let base_w = row_start + c * 16;
                let base_x = c * 16;

                let w0 = vld1q_f32(w.as_ptr().add(base_w));
                let x0 = vld1q_f32(x.as_ptr().add(base_x));
                acc0 = vfmaq_f32(acc0, w0, x0);

                let w1 = vld1q_f32(w.as_ptr().add(base_w + 4));
                let x1 = vld1q_f32(x.as_ptr().add(base_x + 4));
                acc1 = vfmaq_f32(acc1, w1, x1);

                let w2 = vld1q_f32(w.as_ptr().add(base_w + 8));
                let x2 = vld1q_f32(x.as_ptr().add(base_x + 8));
                acc2 = vfmaq_f32(acc2, w2, x2);

                let w3 = vld1q_f32(w.as_ptr().add(base_w + 12));
                let x3 = vld1q_f32(x.as_ptr().add(base_x + 12));
                acc3 = vfmaq_f32(acc3, w3, x3);
            }

            for c in 0..chunks4 {
                let idx = mid_start + c * 4;
                let w_vec = vld1q_f32(w.as_ptr().add(row_start + idx));
                let x_vec = vld1q_f32(x.as_ptr().add(idx));
                acc0 = vfmaq_f32(acc0, w_vec, x_vec);
            }

            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut sum = vaddvq_f32(sum_all);

            for j in 0..remainder {
                let idx = tail_start + j;
                sum += w.get_unchecked(row_start + idx) * x.get_unchecked(idx);
            }

            *out.get_unchecked_mut(last) = sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn matmul_simd_2row(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_fallback(out, w, x, out_dim, in_dim);
}

/// Parallel matrix-vector multiply reading weights from LE bytes.
/// out = W * x, where W is row-major [out_dim, in_dim] stored as LE f32 bytes.
///
/// Splits the outer loop (rows) across threads using `ThreadPool::parallel_for`.
/// Each thread computes a contiguous chunk of output rows independently.
/// The inner dot product per row uses the existing SIMD kernel.
///
/// Bit-identical to `matmul_bytes_simd_2row`: each row's computation is independent,
/// so partitioning rows across threads does not affect the result.
///
/// Falls back to single-threaded `matmul_bytes_simd_2row` when:
/// - out_dim < 256 (avoid overhead on small ops)
/// - rows per thread < 64 (not enough work to amortize spawn cost)
pub fn matmul_bytes_simd_parallel(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    pool: &crate::thread_pool::ThreadPool,
) {
    if !pool.should_parallelize(out_dim) {
        matmul_bytes_simd_2row(out, w_bytes, x, out_dim, in_dim);
        return;
    }

    assert!(out.len() >= out_dim);
    assert!(w_bytes.len() >= out_dim * in_dim * 4);
    assert!(x.len() >= in_dim);

    // Cast to usize to satisfy Sync bound on the Fn closure.
    // *mut f32 is not Sync, but usize is. We reconstruct the pointer inside the closure.
    // SAFETY: parallel_for uses std::thread::scope -- all threads complete before return.
    let out_addr = out.as_mut_ptr() as usize;

    pool.parallel_for(out_dim, |start, end| {
        let chunk_len = end - start;
        if chunk_len == 0 {
            return;
        }

        // SAFETY: Each thread writes to a disjoint contiguous range of `out`.
        // The ranges [start..end) are non-overlapping by construction of parallel_for.
        // w_bytes and x are read-only shared references (safe to read concurrently).
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), chunk_len)
        };

        // Compute the sub-range using the byte offset into w_bytes
        let w_byte_offset = start * in_dim * 4;
        let w_sub = &w_bytes[w_byte_offset..w_byte_offset + chunk_len * in_dim * 4];

        matmul_bytes_simd_2row(out_slice, w_sub, x, chunk_len, in_dim);
    });
}

/// Parallel matrix-vector multiply with weights as f32 slice.
/// out = W * x, where W is row-major [out_dim, in_dim].
///
/// Same parallelization strategy as `matmul_bytes_simd_parallel`.
pub fn matmul_simd_parallel(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    pool: &crate::thread_pool::ThreadPool,
) {
    if !pool.should_parallelize(out_dim) {
        matmul_simd_2row(out, w, x, out_dim, in_dim);
        return;
    }

    assert!(out.len() >= out_dim);
    assert!(w.len() >= out_dim * in_dim);
    assert!(x.len() >= in_dim);

    let out_addr = out.as_mut_ptr() as usize;

    pool.parallel_for(out_dim, |start, end| {
        let chunk_len = end - start;
        if chunk_len == 0 {
            return;
        }

        // SAFETY: Disjoint output ranges. w and x are read-only shared references.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), chunk_len)
        };

        let w_offset = start * in_dim;
        let w_sub = &w[w_offset..w_offset + chunk_len * in_dim];

        matmul_simd_2row(out_slice, w_sub, x, chunk_len, in_dim);
    });
}

// ---------------------------------------------------------------------------
// Q8_0 quantized matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q8_0 block size: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes per 32 elements.
const Q8_0_BLOCK_SIZE: usize = 34;
const Q8_0_GROUP_SIZE: usize = 32;

/// Hardware f16-to-f32 conversion using ARM FCVT instruction (single instruction).
/// ~10x fewer instructions than the software `f16_to_f32_inline`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn f16_to_f32_hw(bits: u16) -> f32 {
    let result: f32;
    unsafe {
        std::arch::asm!(
            "fmov {tmp:h}, {bits:w}",
            "fcvt {out:s}, {tmp:h}",
            bits = in(reg) bits as u32,
            tmp = out(vreg) _,
            out = lateout(vreg) result,
        );
    }
    result
}

/// Hardware f32-to-f16 conversion using ARM FCVT instruction (single instruction).
/// Used in the quantization path instead of the software `f32_to_f16_inline`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn f32_to_f16_hw(val: f32) -> u16 {
    let result: u16;
    unsafe {
        std::arch::asm!(
            "fcvt {tmp:h}, {val:s}",
            "fmov {out:w}, {tmp:h}",
            val = in(vreg) val,
            tmp = out(vreg) _,
            out = lateout(reg) result,
        );
    }
    result
}

/// Batch-convert f16 bytes to f32, using NEON FCVTL for 4-wide vectorized conversion.
/// `src` points to `count` packed f16 values (2 bytes each).
/// `dst` must have capacity for `count` f32 values.
///
/// Uses FCVTL (float convert to longer) which widens 4 f16 → 4 f32 in one instruction.
/// Falls back to scalar FCVT for the tail elements.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn f16_to_f32_batch(src: *const u8, dst: *mut f32, count: usize) {
    unsafe {
        let chunks = count / 4;
        let remainder = count % 4;
        let mut s = src as *const u16;
        let mut d = dst;

        for _ in 0..chunks {
            // Load 4 f16 values into a 64-bit NEON register, then widen to 4 f32
            std::arch::asm!(
                "ld1 {{v0.4h}}, [{src}]",
                "fcvtl v1.4s, v0.4h",
                "st1 {{v1.4s}}, [{dst}]",
                src = in(reg) s,
                dst = in(reg) d,
                out("v0") _,
                out("v1") _,
            );
            s = s.add(4);
            d = d.add(4);
        }

        // Scalar tail
        for i in 0..remainder {
            *d.add(i) = f16_to_f32_hw(*s.add(i));
        }
    }
}

/// Batch-convert f32 to f16 bytes, using NEON FCVTN for 4-wide vectorized conversion.
/// `src` points to `count` f32 values.
/// `dst` must have capacity for `count` packed f16 values (2 bytes each).
///
/// Uses FCVTN (float convert to narrower) which narrows 4 f32 → 4 f16 in one instruction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn f32_to_f16_batch(src: *const f32, dst: *mut u8, count: usize) {
    unsafe {
        let chunks = count / 4;
        let remainder = count % 4;
        let mut s = src;
        let mut d = dst as *mut u16;

        for _ in 0..chunks {
            // Load 4 f32 values, narrow to 4 f16, store
            std::arch::asm!(
                "ld1 {{v0.4s}}, [{src}]",
                "fcvtn v1.4h, v0.4s",
                "st1 {{v1.4h}}, [{dst}]",
                src = in(reg) s,
                dst = in(reg) d,
                out("v0") _,
                out("v1") _,
            );
            s = s.add(4);
            d = d.add(4);
        }

        // Scalar tail
        for i in 0..remainder {
            let bits = f32_to_f16_hw(*s.add(i));
            *d.add(i) = bits;
        }
    }
}

/// Convert f16 (IEEE 754 binary16) bits to f32.
/// Self-contained to avoid cross-crate dependency on the converter's f16_to_f32.
#[inline(always)]
pub(crate) fn f16_to_f32_inline(bits: u16) -> f32 {
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

    let e = (exp as i32) - 15;
    let f = 1.0 + frac as f32 / 1024.0;
    let v = f * 2.0f32.powi(e);
    if sign == 1 { -v } else { v }
}

/// Q8_0 matrix-vector multiply using widening-chain approach (i8->i16->i32->f32).
///
/// `out[i] = dot(dequant(W_q8[i, :]), x[:])`
///
/// This is the original implementation, kept as reference and fallback.
/// The primary `matmul_q8_0_simd` now uses integer dot-product (`sdot`) for ~2-3x throughput.
///
/// Layout per Q8_0 block (GGML standard):
///   [2 bytes f16 scale] [32 bytes int8 quants]
///   Total: 34 bytes per 32 elements
///
/// Dot product for each block: `scale * sum(quant[j] * x[j])`
///
/// Optimizations:
/// - Scale factor applied once per block (not per sub-group) -> 8x fewer vmulq_f32
/// - 2 independent block accumulators for ILP on M4's FMA units
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_simd_widen(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim}, num_blocks={num_blocks})",
        w_q8.len(), out_dim * row_bytes,
    );

    for i in 0..out_dim {
        let row_start = i * row_bytes;

        unsafe {
            let mut row_acc = vdupq_n_f32(0.0);

            for b in 0..num_blocks {
                let block_start = row_start + b * Q8_0_BLOCK_SIZE;
                let x_base = b * Q8_0_GROUP_SIZE;

                // Read f16 scale (2 bytes LE) and convert to f32
                let scale_bits = u16::from_le_bytes([
                    *w_q8.get_unchecked(block_start),
                    *w_q8.get_unchecked(block_start + 1),
                ]);
                let scale = f16_to_f32_hw(scale_bits);
                let scale_v = vdupq_n_f32(scale);

                let q_ptr = w_q8.as_ptr().add(block_start + 2);

                // 2 independent block accumulators for ILP within each block
                let mut block_acc0 = vdupq_n_f32(0.0);
                let mut block_acc1 = vdupq_n_f32(0.0);

                // Group 0: quants[0..16]
                let q8_0 = vld1q_s8(q_ptr as *const i8);
                let q16_lo_0 = vmovl_s8(vget_low_s8(q8_0));
                let q16_hi_0 = vmovl_s8(vget_high_s8(q8_0));

                let qf_a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo_0)));
                let x_a = vld1q_f32(x.as_ptr().add(x_base));
                block_acc0 = vfmaq_f32(block_acc0, qf_a, x_a);

                let qf_b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_lo_0)));
                let x_b = vld1q_f32(x.as_ptr().add(x_base + 4));
                block_acc1 = vfmaq_f32(block_acc1, qf_b, x_b);

                let qf_c = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi_0)));
                let x_c = vld1q_f32(x.as_ptr().add(x_base + 8));
                block_acc0 = vfmaq_f32(block_acc0, qf_c, x_c);

                let qf_d = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_hi_0)));
                let x_d = vld1q_f32(x.as_ptr().add(x_base + 12));
                block_acc1 = vfmaq_f32(block_acc1, qf_d, x_d);

                // Group 1: quants[16..32]
                let q8_1 = vld1q_s8(q_ptr.add(16) as *const i8);
                let q16_lo_1 = vmovl_s8(vget_low_s8(q8_1));
                let q16_hi_1 = vmovl_s8(vget_high_s8(q8_1));

                let qf_e = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo_1)));
                let x_e = vld1q_f32(x.as_ptr().add(x_base + 16));
                block_acc0 = vfmaq_f32(block_acc0, qf_e, x_e);

                let qf_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_lo_1)));
                let x_f = vld1q_f32(x.as_ptr().add(x_base + 20));
                block_acc1 = vfmaq_f32(block_acc1, qf_f, x_f);

                let qf_g = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi_1)));
                let x_g = vld1q_f32(x.as_ptr().add(x_base + 24));
                block_acc0 = vfmaq_f32(block_acc0, qf_g, x_g);

                let qf_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_hi_1)));
                let x_h = vld1q_f32(x.as_ptr().add(x_base + 28));
                block_acc1 = vfmaq_f32(block_acc1, qf_h, x_h);

                // Combine block accumulators and apply scale ONCE per block
                let block_sum = vaddq_f32(block_acc0, block_acc1);
                row_acc = vfmaq_f32(row_acc, block_sum, scale_v);
            }

            *out.get_unchecked_mut(i) = vaddvq_f32(row_acc);
        }
    }
}

/// Q8_0 2-row matmul using widening-chain approach (reference/fallback).
/// Loads x once and reuses for both rows, halving memory bandwidth for x.
///
/// Same optimizations as single-row: factored-out scale, dual accumulators.
/// Falls back to single-row for odd last row.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_simd_2row_widen(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim}, num_blocks={num_blocks})",
        w_q8.len(), out_dim * row_bytes,
    );

    let row_pairs = out_dim / 2;

    for p in 0..row_pairs {
        let row0 = p * 2;
        let row1 = row0 + 1;
        let row0_start = row0 * row_bytes;
        let row1_start = row1 * row_bytes;

        unsafe {
            let mut r0_acc = vdupq_n_f32(0.0);
            let mut r1_acc = vdupq_n_f32(0.0);

            for b in 0..num_blocks {
                let x_base = b * Q8_0_GROUP_SIZE;
                let b0_start = row0_start + b * Q8_0_BLOCK_SIZE;
                let b1_start = row1_start + b * Q8_0_BLOCK_SIZE;

                // Read scales for both rows
                let scale0_bits = u16::from_le_bytes([
                    *w_q8.get_unchecked(b0_start),
                    *w_q8.get_unchecked(b0_start + 1),
                ]);
                let scale0 = f16_to_f32_hw(scale0_bits);
                let scale0_v = vdupq_n_f32(scale0);

                let scale1_bits = u16::from_le_bytes([
                    *w_q8.get_unchecked(b1_start),
                    *w_q8.get_unchecked(b1_start + 1),
                ]);
                let scale1 = f16_to_f32_hw(scale1_bits);
                let scale1_v = vdupq_n_f32(scale1);

                let q_ptr0 = w_q8.as_ptr().add(b0_start + 2);
                let q_ptr1 = w_q8.as_ptr().add(b1_start + 2);

                // 2 block accumulators per row for ILP
                let mut blk0_acc0 = vdupq_n_f32(0.0);
                let mut blk0_acc1 = vdupq_n_f32(0.0);
                let mut blk1_acc0 = vdupq_n_f32(0.0);
                let mut blk1_acc1 = vdupq_n_f32(0.0);

                // Group 0: quants[0..16] -- load x once, use for both rows
                let q8_r0_0 = vld1q_s8(q_ptr0 as *const i8);
                let q8_r1_0 = vld1q_s8(q_ptr1 as *const i8);

                let q16_r0_lo_0 = vmovl_s8(vget_low_s8(q8_r0_0));
                let q16_r0_hi_0 = vmovl_s8(vget_high_s8(q8_r0_0));
                let q16_r1_lo_0 = vmovl_s8(vget_low_s8(q8_r1_0));
                let q16_r1_hi_0 = vmovl_s8(vget_high_s8(q8_r1_0));

                // Sub-group a: elements [0..4]
                let x_a = vld1q_f32(x.as_ptr().add(x_base));
                let qf_r0_a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r0_lo_0)));
                blk0_acc0 = vfmaq_f32(blk0_acc0, qf_r0_a, x_a);
                let qf_r1_a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r1_lo_0)));
                blk1_acc0 = vfmaq_f32(blk1_acc0, qf_r1_a, x_a);

                // Sub-group b: elements [4..8]
                let x_b = vld1q_f32(x.as_ptr().add(x_base + 4));
                let qf_r0_b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r0_lo_0)));
                blk0_acc1 = vfmaq_f32(blk0_acc1, qf_r0_b, x_b);
                let qf_r1_b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r1_lo_0)));
                blk1_acc1 = vfmaq_f32(blk1_acc1, qf_r1_b, x_b);

                // Sub-group c: elements [8..12]
                let x_c = vld1q_f32(x.as_ptr().add(x_base + 8));
                let qf_r0_c = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r0_hi_0)));
                blk0_acc0 = vfmaq_f32(blk0_acc0, qf_r0_c, x_c);
                let qf_r1_c = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r1_hi_0)));
                blk1_acc0 = vfmaq_f32(blk1_acc0, qf_r1_c, x_c);

                // Sub-group d: elements [12..16]
                let x_d = vld1q_f32(x.as_ptr().add(x_base + 12));
                let qf_r0_d = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r0_hi_0)));
                blk0_acc1 = vfmaq_f32(blk0_acc1, qf_r0_d, x_d);
                let qf_r1_d = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r1_hi_0)));
                blk1_acc1 = vfmaq_f32(blk1_acc1, qf_r1_d, x_d);

                // Group 1: quants[16..32]
                let q8_r0_1 = vld1q_s8(q_ptr0.add(16) as *const i8);
                let q8_r1_1 = vld1q_s8(q_ptr1.add(16) as *const i8);

                let q16_r0_lo_1 = vmovl_s8(vget_low_s8(q8_r0_1));
                let q16_r0_hi_1 = vmovl_s8(vget_high_s8(q8_r0_1));
                let q16_r1_lo_1 = vmovl_s8(vget_low_s8(q8_r1_1));
                let q16_r1_hi_1 = vmovl_s8(vget_high_s8(q8_r1_1));

                // Sub-group e: elements [16..20]
                let x_e = vld1q_f32(x.as_ptr().add(x_base + 16));
                let qf_r0_e = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r0_lo_1)));
                blk0_acc0 = vfmaq_f32(blk0_acc0, qf_r0_e, x_e);
                let qf_r1_e = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r1_lo_1)));
                blk1_acc0 = vfmaq_f32(blk1_acc0, qf_r1_e, x_e);

                // Sub-group f: elements [20..24]
                let x_f = vld1q_f32(x.as_ptr().add(x_base + 20));
                let qf_r0_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r0_lo_1)));
                blk0_acc1 = vfmaq_f32(blk0_acc1, qf_r0_f, x_f);
                let qf_r1_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r1_lo_1)));
                blk1_acc1 = vfmaq_f32(blk1_acc1, qf_r1_f, x_f);

                // Sub-group g: elements [24..28]
                let x_g = vld1q_f32(x.as_ptr().add(x_base + 24));
                let qf_r0_g = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r0_hi_1)));
                blk0_acc0 = vfmaq_f32(blk0_acc0, qf_r0_g, x_g);
                let qf_r1_g = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_r1_hi_1)));
                blk1_acc0 = vfmaq_f32(blk1_acc0, qf_r1_g, x_g);

                // Sub-group h: elements [28..32]
                let x_h = vld1q_f32(x.as_ptr().add(x_base + 28));
                let qf_r0_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r0_hi_1)));
                blk0_acc1 = vfmaq_f32(blk0_acc1, qf_r0_h, x_h);
                let qf_r1_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_r1_hi_1)));
                blk1_acc1 = vfmaq_f32(blk1_acc1, qf_r1_h, x_h);

                // Apply scale once per block per row
                let blk0_sum = vaddq_f32(blk0_acc0, blk0_acc1);
                r0_acc = vfmaq_f32(r0_acc, blk0_sum, scale0_v);

                let blk1_sum = vaddq_f32(blk1_acc0, blk1_acc1);
                r1_acc = vfmaq_f32(r1_acc, blk1_sum, scale1_v);
            }

            *out.get_unchecked_mut(row0) = vaddvq_f32(r0_acc);
            *out.get_unchecked_mut(row1) = vaddvq_f32(r1_acc);
        }
    }

    // Handle odd last row with single-row path
    if out_dim % 2 != 0 {
        let last = out_dim - 1;
        let row_start = last * row_bytes;

        unsafe {
            let mut row_acc = vdupq_n_f32(0.0);

            for b in 0..num_blocks {
                let block_start = row_start + b * Q8_0_BLOCK_SIZE;
                let x_base = b * Q8_0_GROUP_SIZE;

                let scale_bits = u16::from_le_bytes([
                    *w_q8.get_unchecked(block_start),
                    *w_q8.get_unchecked(block_start + 1),
                ]);
                let scale = f16_to_f32_hw(scale_bits);
                let scale_v = vdupq_n_f32(scale);

                let q_ptr = w_q8.as_ptr().add(block_start + 2);

                let mut block_acc0 = vdupq_n_f32(0.0);
                let mut block_acc1 = vdupq_n_f32(0.0);

                let q8_0 = vld1q_s8(q_ptr as *const i8);
                let q16_lo_0 = vmovl_s8(vget_low_s8(q8_0));
                let q16_hi_0 = vmovl_s8(vget_high_s8(q8_0));

                block_acc0 = vfmaq_f32(block_acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo_0))), vld1q_f32(x.as_ptr().add(x_base)));
                block_acc1 = vfmaq_f32(block_acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_lo_0))), vld1q_f32(x.as_ptr().add(x_base + 4)));
                block_acc0 = vfmaq_f32(block_acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi_0))), vld1q_f32(x.as_ptr().add(x_base + 8)));
                block_acc1 = vfmaq_f32(block_acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_hi_0))), vld1q_f32(x.as_ptr().add(x_base + 12)));

                let q8_1 = vld1q_s8(q_ptr.add(16) as *const i8);
                let q16_lo_1 = vmovl_s8(vget_low_s8(q8_1));
                let q16_hi_1 = vmovl_s8(vget_high_s8(q8_1));

                block_acc0 = vfmaq_f32(block_acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo_1))), vld1q_f32(x.as_ptr().add(x_base + 16)));
                block_acc1 = vfmaq_f32(block_acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_lo_1))), vld1q_f32(x.as_ptr().add(x_base + 20)));
                block_acc0 = vfmaq_f32(block_acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi_1))), vld1q_f32(x.as_ptr().add(x_base + 24)));
                block_acc1 = vfmaq_f32(block_acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_hi_1))), vld1q_f32(x.as_ptr().add(x_base + 28)));

                let block_sum = vaddq_f32(block_acc0, block_acc1);
                row_acc = vfmaq_f32(row_acc, block_sum, scale_v);
            }

            *out.get_unchecked_mut(last) = vaddvq_f32(row_acc);
        }
    }
}

// ==================== sdot-based Q8_0 kernels (primary path) ====================
//
// Strategy: quantize the f32 input vector `x` to Q8_0 format ONCE, then use ARM's
// `sdot` (signed integer dot product) instruction for the inner loop. This replaces
// the widening chain (vmovl_s8 -> vmovl_s16 -> vcvtq_f32 -> vfmaq_f32) with just
// 2 sdot instructions per 32 elements, achieving ~2-3x throughput improvement.
//
// The quantization of x introduces ~0.5-1% relative error vs. the widening approach
// (which operates on the exact f32 x values). This matches llama.cpp's approach.

/// Convert f32 to f16 bits (software fallback, kept as reference).
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn f32_to_f16_inline(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp == 0xff {
        let f16_frac = if frac == 0 { 0 } else { 0x200u32 };
        return ((sign << 15) | (0x1f << 10) | f16_frac) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | (0x1f << 10)) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let shift = 1 - new_exp;
        let f16_frac = (0x800000u32 | frac) >> (shift + 13);
        return ((sign << 15) | f16_frac) as u16;
    }

    let f16_frac = frac >> 13;
    ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
}

/// Quantize an f32 vector into Q8_0 block format using NEON intrinsics.
///
/// Output layout per block: [2 bytes f16 scale][32 bytes int8 quants]
/// `in_dim` must be a multiple of 32.
///
/// Uses NEON vabsq/vmaxq for finding amax, vcvtnq_s32_f32 for rounding to nearest.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn quantize_f32_to_q8_0(x: &[f32], x_q8: &mut [u8], num_blocks: usize) {
    debug_assert!(x.len() >= num_blocks * Q8_0_GROUP_SIZE);
    debug_assert!(x_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);

    unsafe {
        for b in 0..num_blocks {
            let x_base = b * Q8_0_GROUP_SIZE;
            let out_base = b * Q8_0_BLOCK_SIZE;
            let x_ptr = x.as_ptr().add(x_base);

            // Find amax using NEON: process 32 floats in 8 groups of 4
            let mut amax_v = vdupq_n_f32(0.0);
            for g in 0..8 {
                let v = vld1q_f32(x_ptr.add(g * 4));
                amax_v = vmaxq_f32(amax_v, vabsq_f32(v));
            }
            let amax = vmaxvq_f32(amax_v);

            // Compute scale and inverse scale
            let scale: f32;
            let inv_scale: f32;
            if amax == 0.0 {
                scale = 0.0;
                inv_scale = 0.0;
            } else {
                scale = amax / 127.0;
                inv_scale = 127.0 / amax;
            }

            // Store f16 scale (hw conversion)
            let scale_bits = f32_to_f16_hw(scale);
            let scale_bytes = scale_bits.to_le_bytes();
            *x_q8.get_unchecked_mut(out_base) = scale_bytes[0];
            *x_q8.get_unchecked_mut(out_base + 1) = scale_bytes[1];

            // Quantize 32 floats to int8 using NEON
            let inv_scale_v = vdupq_n_f32(inv_scale);
            let q_ptr = x_q8.as_mut_ptr().add(out_base + 2);

            // Process 8 floats at a time (2 NEON registers -> 8 i8s via narrowing)
            for g in 0..4 {
                let f0 = vld1q_f32(x_ptr.add(g * 8));
                let f1 = vld1q_f32(x_ptr.add(g * 8 + 4));

                // Multiply by inv_scale and round to nearest integer
                let scaled0 = vmulq_f32(f0, inv_scale_v);
                let scaled1 = vmulq_f32(f1, inv_scale_v);

                // Convert to int32 with round-to-nearest (vcvtnq)
                let i32_0 = vcvtnq_s32_f32(scaled0);
                let i32_1 = vcvtnq_s32_f32(scaled1);

                // Narrow: i32 -> i16 -> i8 (saturating)
                let i16_0 = vmovn_s32(i32_0); // int16x4
                let i16_1 = vmovn_s32(i32_1); // int16x4
                let i16_combined = vcombine_s16(i16_0, i16_1); // int16x8
                let i8_result = vmovn_s16(i16_combined); // int8x8

                vst1_s8(q_ptr.add(g * 8) as *mut i8, i8_result);
            }
        }
    }
}

/// Integer dot product of two Q8_0 block sequences using ARM `sdot` instruction.
///
/// Each block: [2 bytes f16 scale][32 bytes int8 quants].
/// Result: sum over blocks of (w_scale * x_scale * dot(w_quants, x_quants)).
///
/// Uses 4 independent accumulators for instruction-level parallelism and processes
/// 2 blocks (64 elements) per iteration for better register utilization.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vec_dot_q8_0_q8_0(w_q8: &[u8], x_q8: &[u8], num_blocks: usize) -> f32 {
    unsafe {
        let mut sumf = 0.0f32;

        // Process 2 blocks at a time for better ILP
        let pairs = num_blocks / 2;
        let w_ptr = w_q8.as_ptr();
        let x_ptr = x_q8.as_ptr();

        for p in 0..pairs {
            let b0 = p * 2;
            let b1 = b0 + 1;

            let w_off0 = b0 * Q8_0_BLOCK_SIZE;
            let w_off1 = b1 * Q8_0_BLOCK_SIZE;
            let x_off0 = b0 * Q8_0_BLOCK_SIZE;
            let x_off1 = b1 * Q8_0_BLOCK_SIZE;

            // Prefetch next pair of blocks (both w and x data)
            if p + 1 < pairs {
                let next_b0 = (p + 1) * 2;
                let next_w = next_b0 * Q8_0_BLOCK_SIZE;
                let next_x = next_b0 * Q8_0_BLOCK_SIZE;
                std::arch::asm!(
                    "prfm pldl1strm, [{w}]",
                    "prfm pldl1strm, [{x}]",
                    w = in(reg) w_ptr.add(next_w + 2),
                    x = in(reg) x_ptr.add(next_x + 2),
                    options(nostack, readonly, preserves_flags),
                );
            }

            // Read scales
            let ws0 = f16_to_f32_hw(u16::from_le_bytes([
                *w_ptr.add(w_off0),
                *w_ptr.add(w_off0 + 1),
            ]));
            let xs0 = f16_to_f32_hw(u16::from_le_bytes([
                *x_ptr.add(x_off0),
                *x_ptr.add(x_off0 + 1),
            ]));
            let ws1 = f16_to_f32_hw(u16::from_le_bytes([
                *w_ptr.add(w_off1),
                *w_ptr.add(w_off1 + 1),
            ]));
            let xs1 = f16_to_f32_hw(u16::from_le_bytes([
                *x_ptr.add(x_off1),
                *x_ptr.add(x_off1 + 1),
            ]));

            // Pointers to quant data (32 bytes each)
            let wq0 = w_ptr.add(w_off0 + 2) as *const i8;
            let xq0 = x_ptr.add(x_off0 + 2) as *const i8;
            let wq1 = w_ptr.add(w_off1 + 2) as *const i8;
            let xq1 = x_ptr.add(x_off1 + 2) as *const i8;

            // 4 independent accumulators: 2 per block for ILP
            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            let mut acc3: int32x4_t = vdupq_n_s32(0);

            // Block 0: 32 elements = 2 x 16 bytes
            let w0_lo: int8x16_t = vld1q_s8(wq0);
            let x0_lo: int8x16_t = vld1q_s8(xq0);
            let w0_hi: int8x16_t = vld1q_s8(wq0.add(16));
            let x0_hi: int8x16_t = vld1q_s8(xq0.add(16));

            // Block 1: 32 elements = 2 x 16 bytes
            let w1_lo: int8x16_t = vld1q_s8(wq1);
            let x1_lo: int8x16_t = vld1q_s8(xq1);
            let w1_hi: int8x16_t = vld1q_s8(wq1.add(16));
            let x1_hi: int8x16_t = vld1q_s8(xq1.add(16));

            // sdot: each instruction computes 4 dot products of 4 int8 pairs
            // and accumulates into int32x4. So 1 sdot handles 16 int8 elements.
            std::arch::asm!(
                "sdot {acc0:v}.4s, {w0lo:v}.16b, {x0lo:v}.16b",
                "sdot {acc1:v}.4s, {w0hi:v}.16b, {x0hi:v}.16b",
                "sdot {acc2:v}.4s, {w1lo:v}.16b, {x1lo:v}.16b",
                "sdot {acc3:v}.4s, {w1hi:v}.16b, {x1hi:v}.16b",
                acc0 = inout(vreg) acc0,
                acc1 = inout(vreg) acc1,
                acc2 = inout(vreg) acc2,
                acc3 = inout(vreg) acc3,
                w0lo = in(vreg) w0_lo,
                x0lo = in(vreg) x0_lo,
                w0hi = in(vreg) w0_hi,
                x0hi = in(vreg) x0_hi,
                w1lo = in(vreg) w1_lo,
                x1lo = in(vreg) x1_lo,
                w1hi = in(vreg) w1_hi,
                x1hi = in(vreg) x1_hi,
                options(nostack, pure, nomem),
            );

            // Horizontal sum for block 0
            let sum0 = vaddq_s32(acc0, acc1);
            let dot0 = vaddvq_s32(sum0);
            sumf += ws0 * xs0 * (dot0 as f32);

            // Horizontal sum for block 1
            let sum1 = vaddq_s32(acc2, acc3);
            let dot1 = vaddvq_s32(sum1);
            sumf += ws1 * xs1 * (dot1 as f32);
        }

        // Handle odd last block
        if num_blocks % 2 != 0 {
            let b = num_blocks - 1;
            let w_off = b * Q8_0_BLOCK_SIZE;
            let x_off = b * Q8_0_BLOCK_SIZE;

            let ws = f16_to_f32_hw(u16::from_le_bytes([
                *w_ptr.add(w_off),
                *w_ptr.add(w_off + 1),
            ]));
            let xs = f16_to_f32_hw(u16::from_le_bytes([
                *x_ptr.add(x_off),
                *x_ptr.add(x_off + 1),
            ]));

            let wq = w_ptr.add(w_off + 2) as *const i8;
            let xq = x_ptr.add(x_off + 2) as *const i8;

            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);

            let w_lo: int8x16_t = vld1q_s8(wq);
            let x_lo: int8x16_t = vld1q_s8(xq);
            let w_hi: int8x16_t = vld1q_s8(wq.add(16));
            let x_hi: int8x16_t = vld1q_s8(xq.add(16));

            std::arch::asm!(
                "sdot {acc0:v}.4s, {wlo:v}.16b, {xlo:v}.16b",
                "sdot {acc1:v}.4s, {whi:v}.16b, {xhi:v}.16b",
                acc0 = inout(vreg) acc0,
                acc1 = inout(vreg) acc1,
                wlo = in(vreg) w_lo,
                xlo = in(vreg) x_lo,
                whi = in(vreg) w_hi,
                xhi = in(vreg) x_hi,
                options(nostack, pure, nomem),
            );

            let sum = vaddq_s32(acc0, acc1);
            let dot = vaddvq_s32(sum);
            sumf += ws * xs * (dot as f32);
        }

        sumf
    }
}

/// Maximum number of Q8_0 blocks for stack-based x scales array.
/// Supports up to 8192-dim (256 blocks * 32 elements = 8192).
/// 256 * 4 bytes = 1024 bytes on stack.
#[cfg(target_arch = "aarch64")]
const MAX_X_SCALE_BLOCKS: usize = 256;

/// Precompute x_q8 scales as f32 into a stack buffer.
/// Avoids repeated f16 decode in the inner row loop.
/// num_blocks must be <= MAX_X_SCALE_BLOCKS.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn precompute_x_scales(x_q8: &[u8], num_blocks: usize, scales: &mut [f32]) {
    debug_assert!(num_blocks <= scales.len());
    unsafe {
        for b in 0..num_blocks {
            let off = b * Q8_0_BLOCK_SIZE;
            let bits = u16::from_le_bytes([
                *x_q8.get_unchecked(off),
                *x_q8.get_unchecked(off + 1),
            ]);
            *scales.get_unchecked_mut(b) = f16_to_f32_hw(bits);
        }
    }
}

/// Inner function: compute output rows from pre-quantized x_q8.
/// Called by matmul_q8_0_simd, matmul_q8_0_simd_2row, and the parallel variant.
/// No allocation, no quantization -- pure dot product work.
///
/// Processes 2 rows at a time, loading x_q8 blocks once and reusing for both rows.
/// This halves x_q8 memory traffic and increases ILP with 4 independent accumulators.
/// x scales are precomputed to avoid repeated f16 decode.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn matmul_q8_0_rows_preq(
    out: &mut [f32],
    w_q8: &[u8],
    x_q8: &[u8],
    out_dim: usize,
    num_blocks: usize,
    row_bytes: usize,
) {
    let mut x_scales_buf = [0.0f32; MAX_X_SCALE_BLOCKS];
    precompute_x_scales(x_q8, num_blocks, &mut x_scales_buf);
    let x_scales = &x_scales_buf[..num_blocks];
    let row_pairs = out_dim / 2;
    let x_ptr = x_q8.as_ptr();
    let w_ptr = w_q8.as_ptr();

    unsafe {
        for p in 0..row_pairs {
            let row0 = p * 2;
            let row1 = row0 + 1;
            let w0_base = w_ptr.add(row0 * row_bytes);
            let w1_base = w_ptr.add(row1 * row_bytes);

            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;

            for b in 0..num_blocks {
                let x_off = b * Q8_0_BLOCK_SIZE;
                let w_off = b * Q8_0_BLOCK_SIZE;

                // Prefetch next block's weight data (2 rows)
                if b + 1 < num_blocks {
                    let next_off = (b + 1) * Q8_0_BLOCK_SIZE;
                    std::arch::asm!(
                        "prfm pldl1strm, [{w0}]",
                        "prfm pldl1strm, [{w1}]",
                        w0 = in(reg) w0_base.add(next_off + 2),
                        w1 = in(reg) w1_base.add(next_off + 2),
                        options(nostack, readonly, preserves_flags),
                    );
                }

                // x scale from precomputed array (no f16 decode)
                let xs = *x_scales.get_unchecked(b);

                // Read weight scales for both rows (hw f16 decode)
                let ws0 = f16_to_f32_hw(u16::from_le_bytes([
                    *w0_base.add(w_off),
                    *w0_base.add(w_off + 1),
                ]));
                let ws1 = f16_to_f32_hw(u16::from_le_bytes([
                    *w1_base.add(w_off),
                    *w1_base.add(w_off + 1),
                ]));

                // Load x_q8 quants once (shared)
                let xq_ptr = x_ptr.add(x_off + 2) as *const i8;
                let x_lo: int8x16_t = vld1q_s8(xq_ptr);
                let x_hi: int8x16_t = vld1q_s8(xq_ptr.add(16));

                // Load weight quants for row 0
                let wq0_ptr = w0_base.add(w_off + 2) as *const i8;
                let w0_lo: int8x16_t = vld1q_s8(wq0_ptr);
                let w0_hi: int8x16_t = vld1q_s8(wq0_ptr.add(16));

                // Load weight quants for row 1
                let wq1_ptr = w1_base.add(w_off + 2) as *const i8;
                let w1_lo: int8x16_t = vld1q_s8(wq1_ptr);
                let w1_hi: int8x16_t = vld1q_s8(wq1_ptr.add(16));

                // 4 independent sdot: 2 per row for ILP
                let mut acc0: int32x4_t = vdupq_n_s32(0);
                let mut acc1: int32x4_t = vdupq_n_s32(0);
                let mut acc2: int32x4_t = vdupq_n_s32(0);
                let mut acc3: int32x4_t = vdupq_n_s32(0);

                std::arch::asm!(
                    "sdot {acc0:v}.4s, {w0lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc1:v}.4s, {w0hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc2:v}.4s, {w1lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc3:v}.4s, {w1hi:v}.16b, {xhi:v}.16b",
                    acc0 = inout(vreg) acc0,
                    acc1 = inout(vreg) acc1,
                    acc2 = inout(vreg) acc2,
                    acc3 = inout(vreg) acc3,
                    w0lo = in(vreg) w0_lo,
                    w0hi = in(vreg) w0_hi,
                    w1lo = in(vreg) w1_lo,
                    w1hi = in(vreg) w1_hi,
                    xlo = in(vreg) x_lo,
                    xhi = in(vreg) x_hi,
                    options(nostack, pure, nomem),
                );

                // Horizontal sums and scale application
                let dot0 = vaddvq_s32(vaddq_s32(acc0, acc1));
                sum0 += ws0 * xs * (dot0 as f32);

                let dot1 = vaddvq_s32(vaddq_s32(acc2, acc3));
                sum1 += ws1 * xs * (dot1 as f32);
            }

            *out.get_unchecked_mut(row0) = sum0;
            *out.get_unchecked_mut(row1) = sum1;
        }

        // Handle odd last row
        if out_dim % 2 != 0 {
            let last = out_dim - 1;
            *out.get_unchecked_mut(last) = vec_dot_q8_0_q8_0(
                std::slice::from_raw_parts(w_ptr.add(last * row_bytes), row_bytes),
                x_q8,
                num_blocks,
            );
        }
    }
}

/// Inner function: compute output rows from pre-quantized x_q8.
/// Processes 4 rows at a time, loading x_q8 blocks once and reusing across all 4 rows.
/// This reduces x_q8 memory traffic by ~17% compared to the 2-row variant
/// (80 bytes/row vs 96 bytes/row per block iteration).
///
/// Uses 8 independent sdot accumulators (2 per row) in a single asm block
/// for maximum ILP on Apple M4 (18 registers total: 8 acc + 2 x + 8 weight).
///
/// Remainder handling:
/// - 2-3 leftover rows: falls back to the 2-row path
/// - 1 leftover row: uses vec_dot_q8_0_q8_0
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn matmul_q8_0_rows_preq_4row(
    out: &mut [f32],
    w_q8: &[u8],
    x_q8: &[u8],
    out_dim: usize,
    num_blocks: usize,
    row_bytes: usize,
) {
    let mut x_scales_buf = [0.0f32; MAX_X_SCALE_BLOCKS];
    precompute_x_scales(x_q8, num_blocks, &mut x_scales_buf);
    let x_scales = &x_scales_buf[..num_blocks];
    let row_quads = out_dim / 4;
    let remainder = out_dim % 4;
    let x_ptr = x_q8.as_ptr();
    let w_ptr = w_q8.as_ptr();

    unsafe {
        for q in 0..row_quads {
            let row0 = q * 4;
            let row1 = row0 + 1;
            let row2 = row0 + 2;
            let row3 = row0 + 3;
            let w0_base = w_ptr.add(row0 * row_bytes);
            let w1_base = w_ptr.add(row1 * row_bytes);
            let w2_base = w_ptr.add(row2 * row_bytes);
            let w3_base = w_ptr.add(row3 * row_bytes);

            // Persistent f32x4 accumulators across all blocks.
            // Avoids per-block vaddvq_s32 (costly cross-lane reduction, ~3-4 cycles each).
            // Instead: vcvtq_f32_s32 (~2 cycles) + vfmaq_f32 (~1 cycle) per row per block.
            // Single horizontal sum per row after all blocks.
            let mut sum0_v: float32x4_t = vdupq_n_f32(0.0);
            let mut sum1_v: float32x4_t = vdupq_n_f32(0.0);
            let mut sum2_v: float32x4_t = vdupq_n_f32(0.0);
            let mut sum3_v: float32x4_t = vdupq_n_f32(0.0);

            // 2-block unrolled inner loop: halves loop overhead (branch,
            // counter, prefetch check) and improves instruction scheduling.
            let pairs = num_blocks / 2;
            let odd = num_blocks & 1;

            for p in 0..pairs {
                let b0 = p * 2;
                let b1 = b0 + 1;
                let x_off0 = b0 * Q8_0_BLOCK_SIZE;
                let w_off0 = x_off0;
                let x_off1 = b1 * Q8_0_BLOCK_SIZE;
                let w_off1 = x_off1;

                // Prefetch 2 blocks ahead (covers next pair)
                if p + 1 < pairs {
                    let pf_off = (b0 + 2) * Q8_0_BLOCK_SIZE;
                    std::arch::asm!(
                        "prfm pldl1strm, [{w0}]",
                        "prfm pldl1strm, [{w1}]",
                        "prfm pldl1strm, [{w2}]",
                        "prfm pldl1strm, [{w3}]",
                        w0 = in(reg) w0_base.add(pf_off + 2),
                        w1 = in(reg) w1_base.add(pf_off + 2),
                        w2 = in(reg) w2_base.add(pf_off + 2),
                        w3 = in(reg) w3_base.add(pf_off + 2),
                        options(nostack, readonly, preserves_flags),
                    );
                }

                // === Block b0 ===
                let xs0 = *x_scales.get_unchecked(b0);

                // Batch f16→f32: gather 4 weight scales, convert in one fcvtl
                let s0_0 = u16::from_le_bytes([*w0_base.add(w_off0), *w0_base.add(w_off0 + 1)]);
                let s1_0 = u16::from_le_bytes([*w1_base.add(w_off0), *w1_base.add(w_off0 + 1)]);
                let s2_0 = u16::from_le_bytes([*w2_base.add(w_off0), *w2_base.add(w_off0 + 1)]);
                let s3_0 = u16::from_le_bytes([*w3_base.add(w_off0), *w3_base.add(w_off0 + 1)]);
                let packed0: u64 = (s0_0 as u64)
                    | ((s1_0 as u64) << 16)
                    | ((s2_0 as u64) << 32)
                    | ((s3_0 as u64) << 48);
                let wscales0: float32x4_t;
                std::arch::asm!(
                    "fmov {tmp:d}, {packed:x}",
                    "fcvtl {out:v}.4s, {tmp:v}.4h",
                    packed = in(reg) packed0,
                    tmp = out(vreg) _,
                    out = lateout(vreg) wscales0,
                    options(nostack, pure, nomem),
                );
                let scale_xs0 = vmulq_n_f32(wscales0, xs0);

                let xq_ptr0 = x_ptr.add(x_off0 + 2) as *const i8;
                let x_lo0: int8x16_t = vld1q_s8(xq_ptr0);
                let x_hi0: int8x16_t = vld1q_s8(xq_ptr0.add(16));

                let wq0_ptr0 = w0_base.add(w_off0 + 2) as *const i8;
                let w0_lo0: int8x16_t = vld1q_s8(wq0_ptr0);
                let w0_hi0: int8x16_t = vld1q_s8(wq0_ptr0.add(16));

                let wq1_ptr0 = w1_base.add(w_off0 + 2) as *const i8;
                let w1_lo0: int8x16_t = vld1q_s8(wq1_ptr0);
                let w1_hi0: int8x16_t = vld1q_s8(wq1_ptr0.add(16));

                let wq2_ptr0 = w2_base.add(w_off0 + 2) as *const i8;
                let w2_lo0: int8x16_t = vld1q_s8(wq2_ptr0);
                let w2_hi0: int8x16_t = vld1q_s8(wq2_ptr0.add(16));

                let wq3_ptr0 = w3_base.add(w_off0 + 2) as *const i8;
                let w3_lo0: int8x16_t = vld1q_s8(wq3_ptr0);
                let w3_hi0: int8x16_t = vld1q_s8(wq3_ptr0.add(16));

                let mut acc0: int32x4_t = vdupq_n_s32(0);
                let mut acc1: int32x4_t = vdupq_n_s32(0);
                let mut acc2: int32x4_t = vdupq_n_s32(0);
                let mut acc3: int32x4_t = vdupq_n_s32(0);
                let mut acc4: int32x4_t = vdupq_n_s32(0);
                let mut acc5: int32x4_t = vdupq_n_s32(0);
                let mut acc6: int32x4_t = vdupq_n_s32(0);
                let mut acc7: int32x4_t = vdupq_n_s32(0);

                std::arch::asm!(
                    "sdot {acc0:v}.4s, {w0lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc1:v}.4s, {w0hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc2:v}.4s, {w1lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc3:v}.4s, {w1hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc4:v}.4s, {w2lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc5:v}.4s, {w2hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc6:v}.4s, {w3lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc7:v}.4s, {w3hi:v}.16b, {xhi:v}.16b",
                    acc0 = inout(vreg) acc0,
                    acc1 = inout(vreg) acc1,
                    acc2 = inout(vreg) acc2,
                    acc3 = inout(vreg) acc3,
                    acc4 = inout(vreg) acc4,
                    acc5 = inout(vreg) acc5,
                    acc6 = inout(vreg) acc6,
                    acc7 = inout(vreg) acc7,
                    w0lo = in(vreg) w0_lo0,
                    w0hi = in(vreg) w0_hi0,
                    w1lo = in(vreg) w1_lo0,
                    w1hi = in(vreg) w1_hi0,
                    w2lo = in(vreg) w2_lo0,
                    w2hi = in(vreg) w2_hi0,
                    w3lo = in(vreg) w3_lo0,
                    w3hi = in(vreg) w3_hi0,
                    xlo = in(vreg) x_lo0,
                    xhi = in(vreg) x_hi0,
                    options(nostack, pure, nomem),
                );

                let combined0 = vaddq_s32(acc0, acc1);
                let f0 = vcvtq_f32_s32(combined0);
                sum0_v = vfmaq_laneq_f32::<0>(sum0_v, f0, scale_xs0);

                let combined1 = vaddq_s32(acc2, acc3);
                let f1 = vcvtq_f32_s32(combined1);
                sum1_v = vfmaq_laneq_f32::<1>(sum1_v, f1, scale_xs0);

                let combined2 = vaddq_s32(acc4, acc5);
                let f2 = vcvtq_f32_s32(combined2);
                sum2_v = vfmaq_laneq_f32::<2>(sum2_v, f2, scale_xs0);

                let combined3 = vaddq_s32(acc6, acc7);
                let f3 = vcvtq_f32_s32(combined3);
                sum3_v = vfmaq_laneq_f32::<3>(sum3_v, f3, scale_xs0);

                // === Block b1 ===
                let xs1 = *x_scales.get_unchecked(b1);

                // Batch f16→f32: gather 4 weight scales, convert in one fcvtl
                let s0_1 = u16::from_le_bytes([*w0_base.add(w_off1), *w0_base.add(w_off1 + 1)]);
                let s1_1 = u16::from_le_bytes([*w1_base.add(w_off1), *w1_base.add(w_off1 + 1)]);
                let s2_1 = u16::from_le_bytes([*w2_base.add(w_off1), *w2_base.add(w_off1 + 1)]);
                let s3_1 = u16::from_le_bytes([*w3_base.add(w_off1), *w3_base.add(w_off1 + 1)]);
                let packed1: u64 = (s0_1 as u64)
                    | ((s1_1 as u64) << 16)
                    | ((s2_1 as u64) << 32)
                    | ((s3_1 as u64) << 48);
                let wscales1: float32x4_t;
                std::arch::asm!(
                    "fmov {tmp:d}, {packed:x}",
                    "fcvtl {out:v}.4s, {tmp:v}.4h",
                    packed = in(reg) packed1,
                    tmp = out(vreg) _,
                    out = lateout(vreg) wscales1,
                    options(nostack, pure, nomem),
                );
                let scale_xs1 = vmulq_n_f32(wscales1, xs1);

                let xq_ptr1 = x_ptr.add(x_off1 + 2) as *const i8;
                let x_lo1: int8x16_t = vld1q_s8(xq_ptr1);
                let x_hi1: int8x16_t = vld1q_s8(xq_ptr1.add(16));

                let wq0_ptr1 = w0_base.add(w_off1 + 2) as *const i8;
                let w0_lo1: int8x16_t = vld1q_s8(wq0_ptr1);
                let w0_hi1: int8x16_t = vld1q_s8(wq0_ptr1.add(16));

                let wq1_ptr1 = w1_base.add(w_off1 + 2) as *const i8;
                let w1_lo1: int8x16_t = vld1q_s8(wq1_ptr1);
                let w1_hi1: int8x16_t = vld1q_s8(wq1_ptr1.add(16));

                let wq2_ptr1 = w2_base.add(w_off1 + 2) as *const i8;
                let w2_lo1: int8x16_t = vld1q_s8(wq2_ptr1);
                let w2_hi1: int8x16_t = vld1q_s8(wq2_ptr1.add(16));

                let wq3_ptr1 = w3_base.add(w_off1 + 2) as *const i8;
                let w3_lo1: int8x16_t = vld1q_s8(wq3_ptr1);
                let w3_hi1: int8x16_t = vld1q_s8(wq3_ptr1.add(16));

                let mut acc0b: int32x4_t = vdupq_n_s32(0);
                let mut acc1b: int32x4_t = vdupq_n_s32(0);
                let mut acc2b: int32x4_t = vdupq_n_s32(0);
                let mut acc3b: int32x4_t = vdupq_n_s32(0);
                let mut acc4b: int32x4_t = vdupq_n_s32(0);
                let mut acc5b: int32x4_t = vdupq_n_s32(0);
                let mut acc6b: int32x4_t = vdupq_n_s32(0);
                let mut acc7b: int32x4_t = vdupq_n_s32(0);

                std::arch::asm!(
                    "sdot {acc0:v}.4s, {w0lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc1:v}.4s, {w0hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc2:v}.4s, {w1lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc3:v}.4s, {w1hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc4:v}.4s, {w2lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc5:v}.4s, {w2hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc6:v}.4s, {w3lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc7:v}.4s, {w3hi:v}.16b, {xhi:v}.16b",
                    acc0 = inout(vreg) acc0b,
                    acc1 = inout(vreg) acc1b,
                    acc2 = inout(vreg) acc2b,
                    acc3 = inout(vreg) acc3b,
                    acc4 = inout(vreg) acc4b,
                    acc5 = inout(vreg) acc5b,
                    acc6 = inout(vreg) acc6b,
                    acc7 = inout(vreg) acc7b,
                    w0lo = in(vreg) w0_lo1,
                    w0hi = in(vreg) w0_hi1,
                    w1lo = in(vreg) w1_lo1,
                    w1hi = in(vreg) w1_hi1,
                    w2lo = in(vreg) w2_lo1,
                    w2hi = in(vreg) w2_hi1,
                    w3lo = in(vreg) w3_lo1,
                    w3hi = in(vreg) w3_hi1,
                    xlo = in(vreg) x_lo1,
                    xhi = in(vreg) x_hi1,
                    options(nostack, pure, nomem),
                );

                let combined0b = vaddq_s32(acc0b, acc1b);
                let f0b = vcvtq_f32_s32(combined0b);
                sum0_v = vfmaq_laneq_f32::<0>(sum0_v, f0b, scale_xs1);

                let combined1b = vaddq_s32(acc2b, acc3b);
                let f1b = vcvtq_f32_s32(combined1b);
                sum1_v = vfmaq_laneq_f32::<1>(sum1_v, f1b, scale_xs1);

                let combined2b = vaddq_s32(acc4b, acc5b);
                let f2b = vcvtq_f32_s32(combined2b);
                sum2_v = vfmaq_laneq_f32::<2>(sum2_v, f2b, scale_xs1);

                let combined3b = vaddq_s32(acc6b, acc7b);
                let f3b = vcvtq_f32_s32(combined3b);
                sum3_v = vfmaq_laneq_f32::<3>(sum3_v, f3b, scale_xs1);
            }

            // Handle odd remainder block (when num_blocks is odd)
            if odd == 1 {
                let b = num_blocks - 1;
                let x_off = b * Q8_0_BLOCK_SIZE;
                let w_off = x_off;

                let xs = *x_scales.get_unchecked(b);

                // Batch f16→f32: gather 4 weight scales, convert in one fcvtl
                let sr0 = u16::from_le_bytes([*w0_base.add(w_off), *w0_base.add(w_off + 1)]);
                let sr1 = u16::from_le_bytes([*w1_base.add(w_off), *w1_base.add(w_off + 1)]);
                let sr2 = u16::from_le_bytes([*w2_base.add(w_off), *w2_base.add(w_off + 1)]);
                let sr3 = u16::from_le_bytes([*w3_base.add(w_off), *w3_base.add(w_off + 1)]);
                let packed_r: u64 = (sr0 as u64)
                    | ((sr1 as u64) << 16)
                    | ((sr2 as u64) << 32)
                    | ((sr3 as u64) << 48);
                let wscales_r: float32x4_t;
                std::arch::asm!(
                    "fmov {tmp:d}, {packed:x}",
                    "fcvtl {out:v}.4s, {tmp:v}.4h",
                    packed = in(reg) packed_r,
                    tmp = out(vreg) _,
                    out = lateout(vreg) wscales_r,
                    options(nostack, pure, nomem),
                );
                let scale_xs_r = vmulq_n_f32(wscales_r, xs);

                let xq_ptr = x_ptr.add(x_off + 2) as *const i8;
                let x_lo: int8x16_t = vld1q_s8(xq_ptr);
                let x_hi: int8x16_t = vld1q_s8(xq_ptr.add(16));

                let wq0_ptr = w0_base.add(w_off + 2) as *const i8;
                let w0_lo: int8x16_t = vld1q_s8(wq0_ptr);
                let w0_hi: int8x16_t = vld1q_s8(wq0_ptr.add(16));

                let wq1_ptr = w1_base.add(w_off + 2) as *const i8;
                let w1_lo: int8x16_t = vld1q_s8(wq1_ptr);
                let w1_hi: int8x16_t = vld1q_s8(wq1_ptr.add(16));

                let wq2_ptr = w2_base.add(w_off + 2) as *const i8;
                let w2_lo: int8x16_t = vld1q_s8(wq2_ptr);
                let w2_hi: int8x16_t = vld1q_s8(wq2_ptr.add(16));

                let wq3_ptr = w3_base.add(w_off + 2) as *const i8;
                let w3_lo: int8x16_t = vld1q_s8(wq3_ptr);
                let w3_hi: int8x16_t = vld1q_s8(wq3_ptr.add(16));

                let mut acc0: int32x4_t = vdupq_n_s32(0);
                let mut acc1: int32x4_t = vdupq_n_s32(0);
                let mut acc2: int32x4_t = vdupq_n_s32(0);
                let mut acc3: int32x4_t = vdupq_n_s32(0);
                let mut acc4: int32x4_t = vdupq_n_s32(0);
                let mut acc5: int32x4_t = vdupq_n_s32(0);
                let mut acc6: int32x4_t = vdupq_n_s32(0);
                let mut acc7: int32x4_t = vdupq_n_s32(0);

                std::arch::asm!(
                    "sdot {acc0:v}.4s, {w0lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc1:v}.4s, {w0hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc2:v}.4s, {w1lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc3:v}.4s, {w1hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc4:v}.4s, {w2lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc5:v}.4s, {w2hi:v}.16b, {xhi:v}.16b",
                    "sdot {acc6:v}.4s, {w3lo:v}.16b, {xlo:v}.16b",
                    "sdot {acc7:v}.4s, {w3hi:v}.16b, {xhi:v}.16b",
                    acc0 = inout(vreg) acc0,
                    acc1 = inout(vreg) acc1,
                    acc2 = inout(vreg) acc2,
                    acc3 = inout(vreg) acc3,
                    acc4 = inout(vreg) acc4,
                    acc5 = inout(vreg) acc5,
                    acc6 = inout(vreg) acc6,
                    acc7 = inout(vreg) acc7,
                    w0lo = in(vreg) w0_lo,
                    w0hi = in(vreg) w0_hi,
                    w1lo = in(vreg) w1_lo,
                    w1hi = in(vreg) w1_hi,
                    w2lo = in(vreg) w2_lo,
                    w2hi = in(vreg) w2_hi,
                    w3lo = in(vreg) w3_lo,
                    w3hi = in(vreg) w3_hi,
                    xlo = in(vreg) x_lo,
                    xhi = in(vreg) x_hi,
                    options(nostack, pure, nomem),
                );

                let combined0 = vaddq_s32(acc0, acc1);
                let f0 = vcvtq_f32_s32(combined0);
                sum0_v = vfmaq_laneq_f32::<0>(sum0_v, f0, scale_xs_r);

                let combined1 = vaddq_s32(acc2, acc3);
                let f1 = vcvtq_f32_s32(combined1);
                sum1_v = vfmaq_laneq_f32::<1>(sum1_v, f1, scale_xs_r);

                let combined2 = vaddq_s32(acc4, acc5);
                let f2 = vcvtq_f32_s32(combined2);
                sum2_v = vfmaq_laneq_f32::<2>(sum2_v, f2, scale_xs_r);

                let combined3 = vaddq_s32(acc6, acc7);
                let f3 = vcvtq_f32_s32(combined3);
                sum3_v = vfmaq_laneq_f32::<3>(sum3_v, f3, scale_xs_r);
            }

            // Single horizontal sum per row after all blocks
            *out.get_unchecked_mut(row0) = vaddvq_f32(sum0_v);
            *out.get_unchecked_mut(row1) = vaddvq_f32(sum1_v);
            *out.get_unchecked_mut(row2) = vaddvq_f32(sum2_v);
            *out.get_unchecked_mut(row3) = vaddvq_f32(sum3_v);
        }

        // Handle remainder rows
        let done = row_quads * 4;
        if remainder >= 2 {
            // Process remaining pair(s) with 2-row path
            let rem_out = &mut out[done..];
            let rem_w = &w_q8[done * row_bytes..];
            matmul_q8_0_rows_preq(rem_out, rem_w, x_q8, remainder, num_blocks, row_bytes);
        } else if remainder == 1 {
            // Single leftover row
            let last = out_dim - 1;
            *out.get_unchecked_mut(last) = vec_dot_q8_0_q8_0(
                std::slice::from_raw_parts(w_ptr.add(last * row_bytes), row_bytes),
                x_q8,
                num_blocks,
            );
        }
    }
}

/// Maximum Q8_0 buffer size for stack allocation.
/// Supports up to 8192-dim vectors (256 blocks * 34 bytes = 8704 bytes).
/// Larger dimensions fall back to heap allocation.
#[cfg(target_arch = "aarch64")]
const Q8_0_STACK_BUF_SIZE: usize = 256 * Q8_0_BLOCK_SIZE; // 8704 bytes

/// Q8_0 matrix-vector multiply using ARM `sdot` integer dot product.
///
/// `out[i] = dot(dequant(W_q8[i, :]), x[:])`
///
/// Strategy: quantize x to Q8_0 ONCE (amortized across all rows), then use
/// `vec_dot_q8_0_q8_0` for each output row. This replaces the widening chain
/// (i8->i16->i32->f32->FMA) with integer dot product (2 sdot per 32 elements).
///
/// Uses stack buffer for x_q8 when dim <= 8192 to avoid heap allocation.
///
/// Precision: quantizing x introduces ~0.5-1% relative error. This matches
/// llama.cpp's approach and is acceptable for inference.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_simd(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim}, num_blocks={num_blocks})",
        w_q8.len(), out_dim * row_bytes,
    );

    let x_q8_len = num_blocks * Q8_0_BLOCK_SIZE;

    if x_q8_len <= Q8_0_STACK_BUF_SIZE {
        // Stack-based path: no heap allocation
        let mut x_q8_buf = [0u8; Q8_0_STACK_BUF_SIZE];
        let x_q8 = &mut x_q8_buf[..x_q8_len];
        quantize_f32_to_q8_0(x, x_q8, num_blocks);
        matmul_q8_0_rows_preq_4row(out, w_q8, x_q8, out_dim, num_blocks, row_bytes);
    } else {
        // Heap fallback for very large dimensions
        let mut x_q8 = vec![0u8; x_q8_len];
        quantize_f32_to_q8_0(x, &mut x_q8, num_blocks);
        matmul_q8_0_rows_preq_4row(out, w_q8, &x_q8, out_dim, num_blocks, row_bytes);
    }
}

/// Q8_0 2-row matmul using ARM `sdot` integer dot product.
///
/// Quantizes x once and computes all rows via `vec_dot_q8_0_q8_0`.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_simd_2row(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim}, num_blocks={num_blocks})",
        w_q8.len(), out_dim * row_bytes,
    );

    let x_q8_len = num_blocks * Q8_0_BLOCK_SIZE;

    if x_q8_len <= Q8_0_STACK_BUF_SIZE {
        let mut x_q8_buf = [0u8; Q8_0_STACK_BUF_SIZE];
        let x_q8 = &mut x_q8_buf[..x_q8_len];
        quantize_f32_to_q8_0(x, x_q8, num_blocks);
        matmul_q8_0_rows_preq_4row(out, w_q8, x_q8, out_dim, num_blocks, row_bytes);
    } else {
        let mut x_q8 = vec![0u8; x_q8_len];
        quantize_f32_to_q8_0(x, &mut x_q8, num_blocks);
        matmul_q8_0_rows_preq_4row(out, w_q8, &x_q8, out_dim, num_blocks, row_bytes);
    }
}

/// Scalar fallback for Q8_0 matmul on non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_q8_0_simd(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    assert!(w_q8.len() >= out_dim * row_bytes);

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;

        for b in 0..num_blocks {
            let block_start = row_start + b * Q8_0_BLOCK_SIZE;
            let x_base = b * Q8_0_GROUP_SIZE;

            let scale_bits = u16::from_le_bytes([w_q8[block_start], w_q8[block_start + 1]]);
            let scale = f16_to_f32_inline(scale_bits);

            for j in 0..Q8_0_GROUP_SIZE {
                let x_idx = x_base + j;
                if x_idx < in_dim {
                    let q = w_q8[block_start + 2 + j] as i8;
                    sum += scale * q as f32 * x[x_idx];
                }
            }
        }

        out[i] = sum;
    }
}

/// Scalar fallback for Q8_0 2-row matmul on non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_q8_0_simd_2row(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_q8_0_simd(out, w_q8, x, out_dim, in_dim);
}

/// Scalar fallback for Q8_0 widen matmul on non-aarch64 targets (used by tests).
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_q8_0_simd_widen(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_q8_0_simd(out, w_q8, x, out_dim, in_dim);
}

/// Scalar fallback for Q8_0 2-row widen matmul on non-aarch64 targets (used by tests).
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_q8_0_simd_2row_widen(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    matmul_q8_0_simd(out, w_q8, x, out_dim, in_dim);
}

/// RMSNorm of Q8_0 weight bytes: dequantize weight in-flight, then apply RMSNorm.
///
/// For norm tensors stored as Q8_0. Dequantizes each block and applies the
/// standard RMSNorm formula: `out[i] = x[i] * dequant(w[i]) * inv_rms`
/// where `inv_rms = 1/sqrt(mean(x^2) + eps)`.
///
/// In practice, norm tensors are almost always F32. This exists for completeness.
#[inline(always)]
pub fn rmsnorm_q8_0_simd(out: &mut [f32], x: &[f32], weight_q8: &[u8], eps: f32) {
    let n = x.len();
    assert!(out.len() >= n);
    let num_blocks = n.div_ceil(Q8_0_GROUP_SIZE);
    assert!(weight_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);

    // Phase 1: mean of squares (reuse existing SIMD dot_product)
    let ms = dot_product_simd(x, x) / n as f32;
    let inv_rms = 1.0 / (ms + eps).sqrt();

    // Phase 2: dequantize weight and multiply
    for b in 0..num_blocks {
        let block_start = b * Q8_0_BLOCK_SIZE;
        let x_base = b * Q8_0_GROUP_SIZE;
        let w_scale_bits = u16::from_le_bytes([weight_q8[block_start], weight_q8[block_start + 1]]);
        let w_scale = f16_to_f32_inline(w_scale_bits);

        for j in 0..Q8_0_GROUP_SIZE {
            let idx = x_base + j;
            if idx >= n {
                break;
            }
            let q = weight_q8[block_start + 2 + j] as i8;
            let w_val = w_scale * q as f32;
            out[idx] = x[idx] * inv_rms * w_val;
        }
    }
}

/// Parallel Q8_0 matrix-vector multiply. Distributes output rows across threads.
///
/// Quantizes x to Q8_0 format ONCE on the calling thread, then distributes
/// pre-quantized data to worker threads for the row dot products.
/// Falls back to single-threaded path for small out_dim.
pub fn matmul_q8_0_simd_parallel(
    out: &mut [f32],
    w_q8: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    pool: &crate::thread_pool::ThreadPool,
) {
    if !pool.should_parallelize(out_dim) {
        matmul_q8_0_simd_2row(out, w_q8, x, out_dim, in_dim);
        return;
    }

    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim})",
        w_q8.len(), out_dim * row_bytes,
    );

    // Quantize x ONCE on the calling thread (amortized across all rows + threads).
    // This is the one place we need heap allocation since the buffer is shared across
    // threads (stack buffer would go out of scope). But it's only ONE allocation
    // regardless of thread count.
    #[cfg(target_arch = "aarch64")]
    let x_q8 = {
        let x_q8_len = num_blocks * Q8_0_BLOCK_SIZE;
        let mut buf = vec![0u8; x_q8_len];
        quantize_f32_to_q8_0(x, &mut buf, num_blocks);
        buf
    };

    let out_addr = out.as_mut_ptr() as usize;

    pool.parallel_for(out_dim, |start, end| {
        let chunk_len = end - start;
        if chunk_len == 0 {
            return;
        }

        // SAFETY: Each thread writes to a disjoint contiguous range of `out`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), chunk_len)
        };

        let w_byte_offset = start * row_bytes;
        let w_sub = &w_q8[w_byte_offset..w_byte_offset + chunk_len * row_bytes];

        #[cfg(target_arch = "aarch64")]
        matmul_q8_0_rows_preq_4row(out_slice, w_sub, &x_q8, chunk_len, num_blocks, row_bytes);

        #[cfg(not(target_arch = "aarch64"))]
        matmul_q8_0_simd_2row(out_slice, w_sub, x, chunk_len, in_dim);
    });
}

/// RMSNorm reading weights from LE bytes.
/// out_i = x_i * w_i / sqrt(mean(x^2) + eps)
///
/// Phase 1: 4-accumulator NEON pattern for mean-of-squares (16 floats/iter ILP).
/// Phase 2: 4-wide NEON element-wise x * scale * weight (16 floats/iter).
/// MUST: eps inside sqrt (best practice 4.2).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn rmsnorm_bytes_simd(out: &mut [f32], x: &[f32], weight_bytes: &[u8], eps: f32) {
    let n = x.len();
    assert!(out.len() >= n);
    assert!(weight_bytes.len() >= n * 4);

    if n == 0 {
        return;
    }

    // Phase 1 loop structure: 16-at-a-time, then 4-at-a-time, then scalar tail
    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;

    unsafe {
        // Phase 1: compute mean of squares with 4 independent accumulators
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for c in 0..chunks16 {
            let base = c * 16;
            let x0 = vld1q_f32(x.as_ptr().add(base));
            acc0 = vfmaq_f32(acc0, x0, x0);
            let x1 = vld1q_f32(x.as_ptr().add(base + 4));
            acc1 = vfmaq_f32(acc1, x1, x1);
            let x2 = vld1q_f32(x.as_ptr().add(base + 8));
            acc2 = vfmaq_f32(acc2, x2, x2);
            let x3 = vld1q_f32(x.as_ptr().add(base + 12));
            acc3 = vfmaq_f32(acc3, x3, x3);
        }

        // Secondary loop: 4 floats per iteration
        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let x_vec = vld1q_f32(x.as_ptr().add(idx));
            acc0 = vfmaq_f32(acc0, x_vec, x_vec);
        }

        // Pairwise reduction: (acc0+acc1) + (acc2+acc3)
        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let sum_all = vaddq_f32(sum01, sum23);
        let mut ms_sum = vaddvq_f32(sum_all);

        // Scalar tail
        for j in 0..remainder {
            let v = *x.get_unchecked(tail_start + j);
            ms_sum += v * v;
        }

        let ms = ms_sum / n as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        let scale_vec = vdupq_n_f32(scale);

        // Phase 2: element-wise x * scale * weight (16 floats per iteration)
        let p2_chunks16 = n / 16;
        let p2_mid_start = p2_chunks16 * 16;
        let p2_chunks4 = (n - p2_mid_start) / 4;
        let p2_tail_start = p2_mid_start + p2_chunks4 * 4;
        let p2_remainder = n - p2_tail_start;

        for c in 0..p2_chunks16 {
            let base = c * 16;
            let byte_base = base * 4;

            let x0 = vld1q_f32(x.as_ptr().add(base));
            let w0 = vreinterpretq_f32_u8(vld1q_u8(weight_bytes.as_ptr().add(byte_base)));
            let s0 = vmulq_f32(vmulq_f32(x0, scale_vec), w0);
            vst1q_f32(out.as_mut_ptr().add(base), s0);

            let x1 = vld1q_f32(x.as_ptr().add(base + 4));
            let w1 = vreinterpretq_f32_u8(vld1q_u8(weight_bytes.as_ptr().add(byte_base + 16)));
            let s1 = vmulq_f32(vmulq_f32(x1, scale_vec), w1);
            vst1q_f32(out.as_mut_ptr().add(base + 4), s1);

            let x2 = vld1q_f32(x.as_ptr().add(base + 8));
            let w2 = vreinterpretq_f32_u8(vld1q_u8(weight_bytes.as_ptr().add(byte_base + 32)));
            let s2 = vmulq_f32(vmulq_f32(x2, scale_vec), w2);
            vst1q_f32(out.as_mut_ptr().add(base + 8), s2);

            let x3 = vld1q_f32(x.as_ptr().add(base + 12));
            let w3 = vreinterpretq_f32_u8(vld1q_u8(weight_bytes.as_ptr().add(byte_base + 48)));
            let s3 = vmulq_f32(vmulq_f32(x3, scale_vec), w3);
            vst1q_f32(out.as_mut_ptr().add(base + 12), s3);
        }

        // Secondary phase 2: 4 floats per iteration
        for c in 0..p2_chunks4 {
            let idx = p2_mid_start + c * 4;
            let byte_offset = idx * 4;
            let x_vec = vld1q_f32(x.as_ptr().add(idx));
            let w_vec = vreinterpretq_f32_u8(vld1q_u8(weight_bytes.as_ptr().add(byte_offset)));
            let result = vmulq_f32(vmulq_f32(x_vec, scale_vec), w_vec);
            vst1q_f32(out.as_mut_ptr().add(idx), result);
        }

        // Scalar tail
        for j in 0..p2_remainder {
            let idx = p2_tail_start + j;
            let byte_offset = idx * 4;
            let w = f32::from_le_bytes([
                *weight_bytes.get_unchecked(byte_offset),
                *weight_bytes.get_unchecked(byte_offset + 1),
                *weight_bytes.get_unchecked(byte_offset + 2),
                *weight_bytes.get_unchecked(byte_offset + 3),
            ]);
            *out.get_unchecked_mut(idx) = *x.get_unchecked(idx) * scale * w;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn rmsnorm_bytes_simd(out: &mut [f32], x: &[f32], weight_bytes: &[u8], eps: f32) {
    rmsnorm_bytes_fallback(out, x, weight_bytes, eps);
}

/// NEON dot product of two f32 slices.
/// Used in attention score computation.
///
/// 4-accumulator pattern for ILP and precision.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for c in 0..chunks16 {
            let base = c * 16;

            let a0 = vld1q_f32(a.as_ptr().add(base));
            let b0 = vld1q_f32(b.as_ptr().add(base));
            acc0 = vfmaq_f32(acc0, a0, b0);

            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            let b1 = vld1q_f32(b.as_ptr().add(base + 4));
            acc1 = vfmaq_f32(acc1, a1, b1);

            let a2 = vld1q_f32(a.as_ptr().add(base + 8));
            let b2 = vld1q_f32(b.as_ptr().add(base + 8));
            acc2 = vfmaq_f32(acc2, a2, b2);

            let a3 = vld1q_f32(a.as_ptr().add(base + 12));
            let b3 = vld1q_f32(b.as_ptr().add(base + 12));
            acc3 = vfmaq_f32(acc3, a3, b3);
        }

        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            let b_vec = vld1q_f32(b.as_ptr().add(idx));
            acc0 = vfmaq_f32(acc0, a_vec, b_vec);
        }

        // Pairwise reduction for better precision
        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let sum_all = vaddq_f32(sum01, sum23);
        let mut sum = vaddvq_f32(sum_all);

        for j in 0..remainder {
            sum += a.get_unchecked(tail_start + j) * b.get_unchecked(tail_start + j);
        }

        sum
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    dot_product_fallback(a, b)
}

/// Fused f16→f32 dot product: computes dot(a_f32, b_f16) without intermediate buffer.
///
/// `a` is f32 query data. `b_f16` points to `n` packed f16 values (2 bytes each).
/// Loads f16 data, widens to f32 via FCVTL (inline asm), and accumulates with NEON FMA —
/// all within registers, eliminating the store-to-load latency of a separate dequant buffer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn dot_product_f16_f32_simd(a: &[f32], b_f16: *const u8, n: usize) -> f32 {
    debug_assert!(a.len() >= n);

    /// Load 4 packed f16 values and widen to float32x4_t via FCVTL.
    #[inline(always)]
    unsafe fn fcvtl_4h(ptr: *const u16) -> float32x4_t {
        let result: float32x4_t;
        std::arch::asm!(
            "ld1 {{v0.4h}}, [{src}]",
            "fcvtl {out:v}.4s, v0.4h",
            src = in(reg) ptr,
            out = out(vreg) result,
            out("v0") _,
            options(nostack, readonly),
        );
        result
    }

    let chunks8 = n / 8;
    let mid_start = chunks8 * 8;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        // Primary loop: 8 elements per iteration (two FCVTL + two FMA)
        for c in 0..chunks8 {
            let base = c * 8;
            let b_ptr = b_f16.add(base * 2) as *const u16;

            let f0 = fcvtl_4h(b_ptr);
            let a0 = vld1q_f32(a.as_ptr().add(base));
            acc0 = vfmaq_f32(acc0, a0, f0);

            let f1 = fcvtl_4h(b_ptr.add(4));
            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            acc1 = vfmaq_f32(acc1, a1, f1);
        }

        // Secondary: 4 elements
        for c in 0..chunks4 {
            let idx = mid_start + c * 4;
            let b_ptr = b_f16.add(idx * 2) as *const u16;
            let f = fcvtl_4h(b_ptr);
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            acc0 = vfmaq_f32(acc0, a_vec, f);
        }

        let sum_all = vaddq_f32(acc0, acc1);
        let mut sum = vaddvq_f32(sum_all);

        // Scalar tail
        for j in 0..remainder {
            let idx = tail_start + j;
            let bits = std::ptr::read_unaligned(b_f16.add(idx * 2) as *const u16);
            sum += *a.get_unchecked(idx) * f16_to_f32_hw(bits);
        }

        sum
    }
}

/// Fused f16 value weighted accumulation: out[i] += scale * dequant(v_f16[i]).
///
/// Same register-fusion as `dot_product_f16_f32_simd`: loads f16 values, widens
/// to f32 via FCVTL, scales, and accumulates into `out` — all without intermediate buffer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn vscale_add_f16_f32_inplace(out: &mut [f32], v_f16: *const u8, scale: f32, n: usize) {
    debug_assert!(out.len() >= n);

    /// Load 4 packed f16 values and widen to float32x4_t via FCVTL.
    #[inline(always)]
    unsafe fn fcvtl_4h(ptr: *const u16) -> float32x4_t {
        let result: float32x4_t;
        std::arch::asm!(
            "ld1 {{v0.4h}}, [{src}]",
            "fcvtl {out:v}.4s, v0.4h",
            src = in(reg) ptr,
            out = out(vreg) result,
            out("v0") _,
            options(nostack, readonly),
        );
        result
    }

    let scale_vec = unsafe { vdupq_n_f32(scale) };
    let chunks4 = n / 4;
    let tail_start = chunks4 * 4;
    let remainder = n - tail_start;

    unsafe {
        for c in 0..chunks4 {
            let idx = c * 4;
            let b_ptr = v_f16.add(idx * 2) as *const u16;
            let f = fcvtl_4h(b_ptr);
            let out_vec = vld1q_f32(out.as_ptr().add(idx));
            let result = vfmaq_f32(out_vec, f, scale_vec);
            vst1q_f32(out.as_mut_ptr().add(idx), result);
        }

        // Scalar tail
        for j in 0..remainder {
            let idx = tail_start + j;
            let bits = std::ptr::read_unaligned(v_f16.add(idx * 2) as *const u16);
            *out.get_unchecked_mut(idx) += scale * f16_to_f32_hw(bits);
        }
    }
}

/// Fused SwiGLU: gate_i = silu(gate_i) * up_i, written in-place.
/// SiLU(x) = x / (1 + exp(-x)).
/// Uses scalar exp with NEON multiply.
///
/// 16-wide primary loop: computes 16 scalar silu values, then 4 NEON multiplies
/// with the up vector. This improves instruction scheduling by amortizing NEON
/// load/store overhead and keeping the FMA pipeline occupied between exp() calls.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn swiglu_inplace_simd(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    let n = gate.len();

    let chunks16 = n / 16;
    let mid_start = chunks16 * 16;
    let chunks4 = (n - mid_start) / 4;
    let tail_start = mid_start + chunks4 * 4;
    let remainder = n - tail_start;

    // SiLU requires exp(), which has no NEON intrinsic.
    // Strategy: compute silu(gate) scalar into temp arrays, then NEON multiply by up.

    // Primary loop: 16 elements per iteration (4 groups of 4)
    for c in 0..chunks16 {
        let base = c * 16;
        unsafe {
            // Group 0: elements [base..base+4]
            let g0 = *gate.get_unchecked(base);
            let g1 = *gate.get_unchecked(base + 1);
            let g2 = *gate.get_unchecked(base + 2);
            let g3 = *gate.get_unchecked(base + 3);
            let s0 = [
                g0 / (1.0 + (-g0).exp()),
                g1 / (1.0 + (-g1).exp()),
                g2 / (1.0 + (-g2).exp()),
                g3 / (1.0 + (-g3).exp()),
            ];

            // Group 1: elements [base+4..base+8]
            let g4 = *gate.get_unchecked(base + 4);
            let g5 = *gate.get_unchecked(base + 5);
            let g6 = *gate.get_unchecked(base + 6);
            let g7 = *gate.get_unchecked(base + 7);
            let s1 = [
                g4 / (1.0 + (-g4).exp()),
                g5 / (1.0 + (-g5).exp()),
                g6 / (1.0 + (-g6).exp()),
                g7 / (1.0 + (-g7).exp()),
            ];

            // Group 2: elements [base+8..base+12]
            let g8 = *gate.get_unchecked(base + 8);
            let g9 = *gate.get_unchecked(base + 9);
            let g10 = *gate.get_unchecked(base + 10);
            let g11 = *gate.get_unchecked(base + 11);
            let s2 = [
                g8 / (1.0 + (-g8).exp()),
                g9 / (1.0 + (-g9).exp()),
                g10 / (1.0 + (-g10).exp()),
                g11 / (1.0 + (-g11).exp()),
            ];

            // Group 3: elements [base+12..base+16]
            let g12 = *gate.get_unchecked(base + 12);
            let g13 = *gate.get_unchecked(base + 13);
            let g14 = *gate.get_unchecked(base + 14);
            let g15 = *gate.get_unchecked(base + 15);
            let s3 = [
                g12 / (1.0 + (-g12).exp()),
                g13 / (1.0 + (-g13).exp()),
                g14 / (1.0 + (-g14).exp()),
                g15 / (1.0 + (-g15).exp()),
            ];

            // 4 NEON multiplies: silu * up
            let result0 = vmulq_f32(vld1q_f32(s0.as_ptr()), vld1q_f32(up.as_ptr().add(base)));
            vst1q_f32(gate.as_mut_ptr().add(base), result0);

            let result1 = vmulq_f32(vld1q_f32(s1.as_ptr()), vld1q_f32(up.as_ptr().add(base + 4)));
            vst1q_f32(gate.as_mut_ptr().add(base + 4), result1);

            let result2 = vmulq_f32(vld1q_f32(s2.as_ptr()), vld1q_f32(up.as_ptr().add(base + 8)));
            vst1q_f32(gate.as_mut_ptr().add(base + 8), result2);

            let result3 = vmulq_f32(vld1q_f32(s3.as_ptr()), vld1q_f32(up.as_ptr().add(base + 12)));
            vst1q_f32(gate.as_mut_ptr().add(base + 12), result3);
        }
    }

    // Secondary loop: 4 elements per iteration
    for c in 0..chunks4 {
        let base = mid_start + c * 4;
        unsafe {
            let g0 = *gate.get_unchecked(base);
            let g1 = *gate.get_unchecked(base + 1);
            let g2 = *gate.get_unchecked(base + 2);
            let g3 = *gate.get_unchecked(base + 3);

            let silu_arr = [
                g0 / (1.0 + (-g0).exp()),
                g1 / (1.0 + (-g1).exp()),
                g2 / (1.0 + (-g2).exp()),
                g3 / (1.0 + (-g3).exp()),
            ];

            let silu_vec = vld1q_f32(silu_arr.as_ptr());
            let up_vec = vld1q_f32(up.as_ptr().add(base));
            let result = vmulq_f32(silu_vec, up_vec);
            vst1q_f32(gate.as_mut_ptr().add(base), result);
        }
    }

    // Scalar tail
    for j in 0..remainder {
        let idx = tail_start + j;
        let g = gate[idx];
        gate[idx] = (g / (1.0 + (-g).exp())) * up[idx];
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn swiglu_inplace_simd(gate: &mut [f32], up: &[f32]) {
    swiglu_inplace_fallback(gate, up);
}

/// Softmax with NEON max reduction, scalar exp, NEON sum and divide.
/// MUST subtract max before exp (best practice 4.1).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn softmax_inplace_simd(logits: &mut [f32]) {
    let n = logits.len();
    if n == 0 {
        return;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        // Phase 1: NEON max reduction
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let v = vld1q_f32(logits.as_ptr().add(c * 4));
            max_vec = vmaxq_f32(max_vec, v);
        }
        let mut max_val = vmaxvq_f32(max_vec);

        let tail_start = chunks * 4;
        for j in 0..remainder {
            let v = *logits.get_unchecked(tail_start + j);
            if v > max_val {
                max_val = v;
            }
        }

        // Handle non-finite max (all NaN or all -inf)
        if !max_val.is_finite() {
            let uniform = 1.0 / n as f32;
            logits.fill(uniform);
            return;
        }

        // Phase 2: subtract max and exp (scalar -- no good NEON exp)
        let mut sum = 0.0f32;
        for i in 0..n {
            let v = (*logits.get_unchecked(i) - max_val).exp();
            *logits.get_unchecked_mut(i) = v;
            sum += v;
        }

        if sum <= f32::EPSILON {
            let uniform = 1.0 / n as f32;
            logits.fill(uniform);
            return;
        }

        // Phase 3: NEON divide by sum
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = vdupq_n_f32(inv_sum);
        for c in 0..chunks {
            let v = vld1q_f32(logits.as_ptr().add(c * 4));
            let result = vmulq_f32(v, inv_sum_vec);
            vst1q_f32(logits.as_mut_ptr().add(c * 4), result);
        }

        for j in 0..remainder {
            let idx = tail_start + j;
            *logits.get_unchecked_mut(idx) *= inv_sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn softmax_inplace_simd(logits: &mut [f32]) {
    softmax_inplace_fallback(logits);
}

/// SIMD-accelerated element-wise vector addition: dst[i] += src[i].
/// Used for residual connections in the transformer compute loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn vadd_inplace_simd(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    let n = dst.len();

    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for c in 0..chunks {
            let offset = c * 4;
            let d_vec = vld1q_f32(dst.as_ptr().add(offset));
            let s_vec = vld1q_f32(src.as_ptr().add(offset));
            let result = vaddq_f32(d_vec, s_vec);
            vst1q_f32(dst.as_mut_ptr().add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        dst[idx] += src[idx];
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn vadd_inplace_simd(dst: &mut [f32], src: &[f32]) {
    vadd_inplace_fallback(dst, src);
}

/// SIMD-accelerated scaled vector addition: dst[i] += src[i] * scale.
/// Used for attention value accumulation (weighted sum of value vectors).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn vscale_add_inplace_simd(dst: &mut [f32], src: &[f32], scale: f32) {
    assert_eq!(dst.len(), src.len());
    let n = dst.len();

    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let scale_vec = vdupq_n_f32(scale);
        for c in 0..chunks {
            let offset = c * 4;
            let d_vec = vld1q_f32(dst.as_ptr().add(offset));
            let s_vec = vld1q_f32(src.as_ptr().add(offset));
            // dst = dst + src * scale (FMA)
            let result = vfmaq_f32(d_vec, s_vec, scale_vec);
            vst1q_f32(dst.as_mut_ptr().add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        dst[idx] += src[idx] * scale;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub fn vscale_add_inplace_simd(dst: &mut [f32], src: &[f32], scale: f32) {
    vscale_add_inplace_fallback(dst, src, scale);
}

// ---- Fallback implementations (used on non-aarch64 and in tests) ----

/// Scalar matmul_bytes -- matches cpu_naive::matmul_bytes exactly.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn matmul_bytes_fallback(
    out: &mut [f32],
    w_bytes: &[u8],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    for i in 0..out_dim {
        let row_byte_start = i * in_dim * 4;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            let offset = row_byte_start + j * 4;
            let w_val = f32::from_le_bytes(w_bytes[offset..offset + 4].try_into().unwrap());
            sum += w_val * x[j];
        }
        out[i] = sum;
    }
}

/// Scalar matmul -- matches cpu_naive::matmul exactly.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn matmul_fallback(
    out: &mut [f32],
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        out[i] = row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum();
    }
}

/// Scalar rmsnorm_bytes -- matches cpu_naive::rmsnorm_bytes exactly.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn rmsnorm_bytes_fallback(out: &mut [f32], x: &[f32], weight_bytes: &[u8], eps: f32) {
    let n = x.len();
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for i in 0..n {
        let w = f32::from_le_bytes(weight_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        out[i] = x[i] * scale * w;
    }
}

/// Scalar dot product.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn dot_product_fallback(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum()
}

/// Scalar swiglu -- matches cpu_naive::swiglu_inplace exactly.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn swiglu_inplace_fallback(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// Scalar softmax -- matches engine::softmax_inplace exactly.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn softmax_inplace_fallback(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
        return;
    }
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum <= f32::EPSILON {
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
        return;
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
}

/// Scalar vadd_inplace -- element-wise dst += src.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn vadd_inplace_fallback(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

/// Scalar vscale_add_inplace -- element-wise dst += src * scale.
#[cfg(any(not(target_arch = "aarch64"), test))]
fn vscale_add_inplace_fallback(dst: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dst.len() {
        dst[i] += src[i] * scale;
    }
}

// ---------------------------------------------------------------------------
// Pre-quantized Q8_0 public API (eliminates redundant quantization in compute_simd)
// ---------------------------------------------------------------------------

/// Make `quantize_f32_to_q8_0` available to `compute_simd.rs`.
/// Callers must provide a properly sized buffer: `in_dim.div_ceil(32) * 34` bytes.
#[cfg(target_arch = "aarch64")]
pub fn quantize_f32_to_q8_0_pub(x: &[f32], x_q8: &mut [u8], in_dim: usize) {
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    debug_assert!(x.len() >= in_dim);
    debug_assert!(x_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);
    quantize_f32_to_q8_0(x, x_q8, num_blocks);
}

/// Q8_0 matmul taking **pre-quantized** `x_q8` (no internal quantization or allocation).
/// Parallel dispatch: distributes rows across threads, each using the shared `x_q8`.
///
/// This is the fast path for cases where the same input vector is multiplied by
/// multiple weight matrices (Q/K/V projections): quantize once, matmul many times.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_preq_parallel(
    out: &mut [f32],
    w_q8: &[u8],
    x_q8: &[u8],
    out_dim: usize,
    in_dim: usize,
    pool: &crate::thread_pool::ThreadPool,
) {
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;

    if !pool.should_parallelize(out_dim) {
        // Single-threaded: use 4-row kernel for better register reuse.
        matmul_q8_0_rows_preq_4row(out, w_q8, x_q8, out_dim, num_blocks, row_bytes);
        return;
    }

    assert!(out.len() >= out_dim);
    assert!(x_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim})",
        w_q8.len(), out_dim * row_bytes,
    );

    let out_addr = out.as_mut_ptr() as usize;

    pool.parallel_for(out_dim, |start, end| {
        let chunk_len = end - start;
        if chunk_len == 0 {
            return;
        }

        // SAFETY: Each thread writes to a disjoint contiguous range of `out`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), chunk_len)
        };

        let w_byte_offset = start * row_bytes;
        let w_sub = &w_q8[w_byte_offset..w_byte_offset + chunk_len * row_bytes];

        matmul_q8_0_rows_preq_4row(out_slice, w_sub, x_q8, chunk_len, num_blocks, row_bytes);
    });
}

/// Q8_0 matmul taking **pre-quantized** `x_q8`, single-threaded.
/// Used for small output dimensions or inside parallel closures where threads
/// already have their own chunk.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_preq(
    out: &mut [f32],
    w_q8: &[u8],
    x_q8: &[u8],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;
    assert!(out.len() >= out_dim);
    assert!(x_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim})",
        w_q8.len(), out_dim * row_bytes,
    );
    matmul_q8_0_rows_preq_4row(out, w_q8, x_q8, out_dim, num_blocks, row_bytes);
}

/// Q8_0 matmul with **fused parallel quantization** of the input vector.
///
/// Eliminates serial quantization from the critical path by splitting the work
/// into two phases within a single `parallel_for` dispatch:
///
///   Phase 1: Each thread quantizes its share of input Q8_0 blocks in parallel.
///            Blocks are distributed evenly across threads based on thread index
///            (derived from the start/end row range).
///   Barrier: An atomic spin-barrier ensures all threads complete quantization
///            before any thread begins the matmul phase.
///   Phase 2: Each thread computes its assigned output rows using the now-complete
///            `x_q8` buffer, identical to `matmul_q8_0_preq_parallel`.
///
/// Saves ~1-2 us of serial quantization per call. Over 22 layers x 3 calls
/// per layer = 66 calls, this removes ~100 us/token from the critical path.
///
/// Falls back to serial quantize + serial matmul when `out_dim` is below the
/// parallel threshold.
///
/// # Safety contract
///
/// `x_q8` must be pre-allocated to `in_dim.div_ceil(32) * 34` bytes.
/// Each thread writes to a disjoint region of `x_q8` in Phase 1 and reads
/// the full `x_q8` in Phase 2 (safe after the barrier).
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_fused_quant_parallel(
    out: &mut [f32],
    w_q8: &[u8],
    x_f32: &[f32],
    x_q8: &mut [u8],
    out_dim: usize,
    in_dim: usize,
    pool: &crate::thread_pool::ThreadPool,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let num_blocks = in_dim.div_ceil(Q8_0_GROUP_SIZE);
    let row_bytes = num_blocks * Q8_0_BLOCK_SIZE;

    if !pool.should_parallelize(out_dim) {
        // Single-threaded: quantize then matmul sequentially.
        quantize_f32_to_q8_0(x_f32, x_q8, num_blocks);
        matmul_q8_0_rows_preq_4row(out, w_q8, x_q8, out_dim, num_blocks, row_bytes);
        return;
    }

    assert!(out.len() >= out_dim);
    assert!(x_f32.len() >= in_dim);
    assert!(x_q8.len() >= num_blocks * Q8_0_BLOCK_SIZE);
    assert!(
        w_q8.len() >= out_dim * row_bytes,
        "w_q8 too short: {} < {} (out_dim={out_dim}, in_dim={in_dim})",
        w_q8.len(), out_dim * row_bytes,
    );

    let out_addr = out.as_mut_ptr() as usize;
    let x_f32_ptr = x_f32.as_ptr() as usize;
    let x_q8_ptr = x_q8.as_mut_ptr() as usize;

    // Barrier: total_threads will decrement this. When it hits 0, all quant is done.
    let total_threads = pool.total_threads();
    let barrier = AtomicUsize::new(total_threads);

    pool.parallel_for(out_dim, |start, end| {
        let chunk_len = end - start;
        if chunk_len == 0 {
            // Still must participate in barrier to avoid deadlock.
            barrier.fetch_sub(1, Ordering::AcqRel);
            while barrier.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
            return;
        }

        // ---- Phase 1: Parallel quantization ----
        // Derive a thread index from our position in the row space.
        // parallel_for distributes rows as: chunks of (out_dim / total_threads),
        // with the first (out_dim % total_threads) chunks getting +1 row.
        // We use start to compute a proportional share of quantization blocks.
        //
        // Simple proportional distribution: thread with start position `s` in
        // range [0, out_dim) gets blocks proportional to s/out_dim.
        let my_block_start = (start as u64 * num_blocks as u64 / out_dim as u64) as usize;
        let my_block_end = (end as u64 * num_blocks as u64 / out_dim as u64) as usize;

        if my_block_start < my_block_end {
            let src = unsafe {
                std::slice::from_raw_parts(
                    (x_f32_ptr as *const f32).add(my_block_start * Q8_0_GROUP_SIZE),
                    (my_block_end - my_block_start) * Q8_0_GROUP_SIZE,
                )
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (x_q8_ptr as *mut u8).add(my_block_start * Q8_0_BLOCK_SIZE),
                    (my_block_end - my_block_start) * Q8_0_BLOCK_SIZE,
                )
            };
            quantize_f32_to_q8_0(src, dst, my_block_end - my_block_start);
        }

        // ---- Barrier: spin until all threads finish quantization ----
        barrier.fetch_sub(1, Ordering::AcqRel);
        while barrier.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }

        // ---- Phase 2: Matmul using the now-complete x_q8 ----
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(start), chunk_len)
        };
        let x_q8_shared = unsafe {
            std::slice::from_raw_parts(x_q8_ptr as *const u8, num_blocks * Q8_0_BLOCK_SIZE)
        };
        let w_byte_offset = start * row_bytes;
        let w_sub = &w_q8[w_byte_offset..w_byte_offset + chunk_len * row_bytes];

        matmul_q8_0_rows_preq_4row(out_slice, w_sub, x_q8_shared, chunk_len, num_blocks, row_bytes);
    });
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert f32 slice to LE bytes (simulates weight storage).
    fn f32_to_le_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Relative error helper -- uses absolute error when values are near zero.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        let diff = (a - b).abs();
        if a.abs().max(b.abs()) < 1e-6 {
            diff < tol
        } else {
            diff / a.abs().max(b.abs()) < tol
        }
    }

    fn assert_slices_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(a, e, tol),
                "{label}[{i}]: got {a}, expected {e}, diff={}",
                (a - e).abs()
            );
        }
    }

    // ==================== matmul_bytes_simd tests ====================

    #[test]
    fn test_matmul_bytes_simd_1x1() {
        let w = f32_to_le_bytes(&[3.0]);
        let x = [2.0f32];
        let mut out = [0.0f32];
        matmul_bytes_simd(&mut out, &w, &x, 1, 1);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    fn test_matmul_bytes_simd_4x4() {
        let w_f32: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..4).map(|i| (i + 1) as f32).collect();

        let mut out_simd = vec![0.0f32; 4];
        let mut out_naive = vec![0.0f32; 4];

        matmul_bytes_simd(&mut out_simd, &w, &x, 4, 4);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 4, 4);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "matmul_bytes 4x4");
    }

    #[test]
    fn test_matmul_bytes_simd_7x3() {
        // Non-multiple-of-4 dimensions
        let w_f32: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x = [1.0f32, 2.0, 3.0];

        let mut out_simd = vec![0.0f32; 7];
        let mut out_naive = vec![0.0f32; 7];

        matmul_bytes_simd(&mut out_simd, &w, &x, 7, 3);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 7, 3);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "matmul_bytes 7x3");
    }

    #[test]
    fn test_matmul_bytes_simd_128x64() {
        let in_dim = 64;
        let out_dim = 128;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-5, "matmul_bytes 128x64");
    }

    #[test]
    fn test_matmul_bytes_simd_1024x512() {
        let in_dim = 512;
        let out_dim = 1024;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-4, "matmul_bytes 1024x512");
    }

    // ==================== matmul_simd tests ====================

    #[test]
    fn test_matmul_simd_1x1() {
        let w = [3.0f32];
        let x = [2.0f32];
        let mut out = [0.0f32];
        matmul_simd(&mut out, &w, &x, 1, 1);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    fn test_matmul_simd_4x4() {
        let w: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let x: Vec<f32> = (0..4).map(|i| (i + 1) as f32).collect();

        let mut out_simd = vec![0.0f32; 4];
        let mut out_naive = vec![0.0f32; 4];

        matmul_simd(&mut out_simd, &w, &x, 4, 4);
        matmul_fallback(&mut out_naive, &w, &x, 4, 4);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "matmul f32 4x4");
    }

    #[test]
    fn test_matmul_simd_7x3() {
        let w: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let x = [1.0f32, 2.0, 3.0];

        let mut out_simd = vec![0.0f32; 7];
        let mut out_naive = vec![0.0f32; 7];

        matmul_simd(&mut out_simd, &w, &x, 7, 3);
        matmul_fallback(&mut out_naive, &w, &x, 7, 3);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "matmul f32 7x3");
    }

    #[test]
    fn test_matmul_simd_128x64() {
        let in_dim = 64;
        let out_dim = 128;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-5, "matmul f32 128x64");
    }

    // ==================== rmsnorm_bytes_simd tests ====================

    #[test]
    fn test_rmsnorm_bytes_simd_dim1() {
        let x = [3.0f32];
        let w = f32_to_le_bytes(&[2.0]);
        let mut out_simd = [0.0f32];
        let mut out_naive = [0.0f32];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "rmsnorm dim=1");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_dim4() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let w = f32_to_le_bytes(&weights);
        let mut out_simd = [0.0f32; 4];
        let mut out_naive = [0.0f32; 4];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "rmsnorm dim=4");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_dim7() {
        let x: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let weights: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let w = f32_to_le_bytes(&weights);
        let mut out_simd = vec![0.0f32; 7];
        let mut out_naive = vec![0.0f32; 7];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "rmsnorm dim=7");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_dim128() {
        let x: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let weights: Vec<f32> = (0..128).map(|i| ((i % 11) as f32 + 1.0) * 0.1).collect();
        let w = f32_to_le_bytes(&weights);
        let mut out_simd = vec![0.0f32; 128];
        let mut out_naive = vec![0.0f32; 128];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-5, "rmsnorm dim=128");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_all_zeros() {
        let x = [0.0f32; 4];
        let weights = [1.0f32; 4];
        let w = f32_to_le_bytes(&weights);
        let mut out = [0.0f32; 4];

        rmsnorm_bytes_simd(&mut out, &x, &w, 1e-5);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "rmsnorm_simd[{i}] should be finite, got {v}");
            assert_eq!(v, 0.0, "rmsnorm of zero input should be zero");
        }
    }

    // ==================== dot_product_simd tests ====================

    #[test]
    fn test_dot_product_simd_len1() {
        assert_eq!(dot_product_simd(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn test_dot_product_simd_len4() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-6),
            "dot len=4: simd={simd_result}, naive={naive_result}"
        );
    }

    #[test]
    fn test_dot_product_simd_len7() {
        let a: Vec<f32> = (0..7).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..7).map(|i| (i + 1) as f32 * 0.5).collect();
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-6),
            "dot len=7: simd={simd_result}, naive={naive_result}"
        );
    }

    #[test]
    fn test_dot_product_simd_len128() {
        let a: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-5),
            "dot len=128: simd={simd_result}, naive={naive_result}"
        );
    }

    #[test]
    fn test_dot_product_simd_all_zeros() {
        let a = [0.0f32; 8];
        let b = [1.0f32; 8];
        assert_eq!(dot_product_simd(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_simd_large_values() {
        let a = [1e6f32; 4];
        let b = [1e6f32; 4];
        let result = dot_product_simd(&a, &b);
        let expected = 4.0e12;
        assert!(
            approx_eq(result, expected, 1e-6),
            "dot large: simd={result}, expected={expected}"
        );
    }

    // ==================== swiglu_inplace_simd tests ====================

    #[test]
    fn test_swiglu_simd_len1() {
        let mut gate_simd = [1.0f32];
        let mut gate_naive = [1.0f32];
        let up = [2.0f32];

        swiglu_inplace_simd(&mut gate_simd, &up);
        swiglu_inplace_fallback(&mut gate_naive, &up);

        assert_slices_close(&gate_simd, &gate_naive, 1e-6, "swiglu len=1");
    }

    #[test]
    fn test_swiglu_simd_len4() {
        let mut gate_simd = [0.0f32, 1.0, -1.0, 2.0];
        let mut gate_naive = [0.0f32, 1.0, -1.0, 2.0];
        let up = [1.0f32, 1.0, 1.0, 1.0];

        swiglu_inplace_simd(&mut gate_simd, &up);
        swiglu_inplace_fallback(&mut gate_naive, &up);

        assert_slices_close(&gate_simd, &gate_naive, 1e-6, "swiglu len=4");
    }

    #[test]
    fn test_swiglu_simd_len7() {
        let mut gate_simd: Vec<f32> = (0..7).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let mut gate_naive = gate_simd.clone();
        let up: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.3).collect();

        swiglu_inplace_simd(&mut gate_simd, &up);
        swiglu_inplace_fallback(&mut gate_naive, &up);

        assert_slices_close(&gate_simd, &gate_naive, 1e-6, "swiglu len=7");
    }

    #[test]
    fn test_swiglu_simd_len128() {
        let mut gate_simd: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let mut gate_naive = gate_simd.clone();
        let up: Vec<f32> = (0..128).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        swiglu_inplace_simd(&mut gate_simd, &up);
        swiglu_inplace_fallback(&mut gate_naive, &up);

        assert_slices_close(&gate_simd, &gate_naive, 1e-6, "swiglu len=128");
    }

    #[test]
    fn test_swiglu_simd_zero_gate() {
        let mut gate = [0.0f32; 4];
        let up = [1.0f32; 4];
        swiglu_inplace_simd(&mut gate, &up);
        // silu(0) = 0
        for &v in &gate {
            assert_eq!(v, 0.0);
        }
    }

    // ==================== softmax_inplace_simd tests ====================

    #[test]
    fn test_softmax_simd_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace_simd(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn test_softmax_simd_single() {
        let mut v = [5.0f32];
        softmax_inplace_simd(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_simd_len3() {
        let mut simd = [1.0f32, 2.0, 3.0];
        let mut naive = [1.0f32, 2.0, 3.0];

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-6, "softmax len=3");

        // Verify sum to 1
        let sum: f32 = simd.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    #[test]
    fn test_softmax_simd_len4() {
        let mut simd = [1.0f32, 2.0, 3.0, 4.0];
        let mut naive = [1.0f32, 2.0, 3.0, 4.0];

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-6, "softmax len=4");
    }

    #[test]
    fn test_softmax_simd_len7() {
        let mut simd: Vec<f32> = (0..7).map(|i| (i as f32) * 0.5).collect();
        let mut naive = simd.clone();

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-6, "softmax len=7");
    }

    #[test]
    fn test_softmax_simd_len128() {
        let mut simd: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let mut naive = simd.clone();

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-6, "softmax len=128");
    }

    #[test]
    fn test_softmax_simd_large_values() {
        // Must not overflow (best practice 4.1)
        let mut simd = [1000.0f32, 1001.0, 1002.0];
        let mut naive = [1000.0f32, 1001.0, 1002.0];

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-5, "softmax large values");
        assert!(simd.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_simd_all_same() {
        let mut v = [3.0f32; 4];
        softmax_inplace_simd(&mut v);
        for &p in &v {
            assert!((p - 0.25).abs() < 1e-6, "expected uniform 0.25, got {p}");
        }
    }

    #[test]
    fn test_softmax_simd_all_nan() {
        let mut v = [f32::NAN; 3];
        softmax_inplace_simd(&mut v);
        for &p in &v {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-6,
                "expected uniform, got {p}"
            );
        }
    }

    #[test]
    fn test_softmax_simd_all_neg_inf() {
        let mut v = [f32::NEG_INFINITY; 3];
        softmax_inplace_simd(&mut v);
        for &p in &v {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-6,
                "expected uniform, got {p}"
            );
        }
    }

    #[test]
    fn test_softmax_simd_tiny_values() {
        let mut simd = [1e-38f32, 2e-38, 3e-38];
        let mut naive = [1e-38f32, 2e-38, 3e-38];

        softmax_inplace_simd(&mut simd);
        softmax_inplace_fallback(&mut naive);

        assert_slices_close(&simd, &naive, 1e-5, "softmax tiny values");
    }

    // ==================== Edge case: zero-length inputs ====================

    #[test]
    fn test_matmul_bytes_simd_zero_dim() {
        let mut out = vec![0.0f32; 0];
        matmul_bytes_simd(&mut out, &[], &[], 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_matmul_simd_zero_dim() {
        let mut out = vec![0.0f32; 0];
        matmul_simd(&mut out, &[], &[], 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_dot_product_simd_zero_len() {
        assert_eq!(dot_product_simd(&[], &[]), 0.0);
    }

    #[test]
    fn test_swiglu_simd_zero_len() {
        let mut gate: Vec<f32> = vec![];
        swiglu_inplace_simd(&mut gate, &[]);
        assert!(gate.is_empty());
    }

    #[test]
    fn test_rmsnorm_bytes_simd_zero_len() {
        let mut out: Vec<f32> = vec![];
        rmsnorm_bytes_simd(&mut out, &[], &[], 1e-5);
        assert!(out.is_empty());
    }

    // ==================== matmul_bytes_simd: additional edge cases ====================

    #[test]
    fn test_matmul_bytes_simd_all_zeros() {
        let w = f32_to_le_bytes(&[0.0f32; 16]);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        matmul_bytes_simd(&mut out, &w, &x, 4, 4);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_matmul_bytes_simd_identity_like() {
        // Diagonal-like: 1 on diagonal, 0 elsewhere for 4x4
        let mut w_f32 = vec![0.0f32; 16];
        w_f32[0] = 1.0;
        w_f32[5] = 1.0;
        w_f32[10] = 1.0;
        w_f32[15] = 1.0;
        let w = f32_to_le_bytes(&w_f32);
        let x = [2.0f32, 3.0, 5.0, 7.0];
        let mut out = [0.0f32; 4];
        matmul_bytes_simd(&mut out, &w, &x, 4, 4);
        assert_slices_close(&out, &x, 1e-7, "matmul identity");
    }

    // ==================== Cross-validation: SIMD vs naive from compute_naive ====================
    // These tests ensure the SIMD implementations match the fallback (which mirrors compute_naive)
    // for various non-trivial input patterns.

    #[test]
    fn test_matmul_bytes_simd_negative_values() {
        let w_f32: Vec<f32> = (0..20).map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 }).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..5).map(|i| (i as f32) - 2.0).collect();

        let mut out_simd = vec![0.0f32; 4];
        let mut out_naive = vec![0.0f32; 4];

        matmul_bytes_simd(&mut out_simd, &w, &x, 4, 5);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 4, 5);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "matmul_bytes negative values");
    }

    #[test]
    fn test_swiglu_simd_large_negative() {
        // Large negative gate values should not produce NaN
        let mut gate = [-100.0f32; 4];
        let up = [1.0f32; 4];
        swiglu_inplace_simd(&mut gate, &up);
        for &v in &gate {
            assert!(v.is_finite(), "swiglu with large negative should be finite, got {v}");
        }
    }

    #[test]
    fn test_swiglu_simd_large_positive() {
        let mut gate_simd = [100.0f32; 4];
        let mut gate_naive = [100.0f32; 4];
        let up = [1.0f32; 4];

        swiglu_inplace_simd(&mut gate_simd, &up);
        swiglu_inplace_fallback(&mut gate_naive, &up);

        assert_slices_close(&gate_simd, &gate_naive, 1e-5, "swiglu large positive");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_all_same() {
        let x = [3.0f32; 8];
        let weights = [1.0f32; 8];
        let w = f32_to_le_bytes(&weights);
        let mut out_simd = vec![0.0f32; 8];
        let mut out_naive = vec![0.0f32; 8];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-6, "rmsnorm all same");
    }

    #[test]
    fn test_rmsnorm_bytes_simd_tiny_values() {
        let x = [1e-38f32; 4];
        let weights = [1.0f32; 4];
        let w = f32_to_le_bytes(&weights);
        let mut out_simd = vec![0.0f32; 4];
        let mut out_naive = vec![0.0f32; 4];

        rmsnorm_bytes_simd(&mut out_simd, &x, &w, 1e-5);
        rmsnorm_bytes_fallback(&mut out_naive, &x, &w, 1e-5);

        assert_slices_close(&out_simd, &out_naive, 1e-5, "rmsnorm tiny values");
    }

    #[test]
    fn test_dot_product_simd_len1024() {
        let a: Vec<f32> = (0..1024).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
        let b: Vec<f32> = (0..1024).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-4),
            "dot len=1024: simd={simd_result}, naive={naive_result}"
        );
    }

    // ==================== vadd_inplace_simd tests ====================

    #[test]
    fn test_vadd_inplace_simd_len1() {
        let mut dst = [1.0f32];
        let src = [2.0f32];
        vadd_inplace_simd(&mut dst, &src);
        assert_eq!(dst[0], 3.0);
    }

    #[test]
    fn test_vadd_inplace_simd_len4() {
        let mut dst_simd = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst_naive = [1.0f32, 2.0, 3.0, 4.0];
        let src = [5.0f32, 6.0, 7.0, 8.0];

        vadd_inplace_simd(&mut dst_simd, &src);
        vadd_inplace_fallback(&mut dst_naive, &src);

        assert_slices_close(&dst_simd, &dst_naive, 1e-7, "vadd len=4");
    }

    #[test]
    fn test_vadd_inplace_simd_len7() {
        let mut dst_simd: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut dst_naive = dst_simd.clone();
        let src: Vec<f32> = (0..7).map(|i| (i as f32) * 0.5).collect();

        vadd_inplace_simd(&mut dst_simd, &src);
        vadd_inplace_fallback(&mut dst_naive, &src);

        assert_slices_close(&dst_simd, &dst_naive, 1e-7, "vadd len=7");
    }

    #[test]
    fn test_vadd_inplace_simd_len128() {
        let mut dst_simd: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let mut dst_naive = dst_simd.clone();
        let src: Vec<f32> = (0..128).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        vadd_inplace_simd(&mut dst_simd, &src);
        vadd_inplace_fallback(&mut dst_naive, &src);

        assert_slices_close(&dst_simd, &dst_naive, 1e-6, "vadd len=128");
    }

    #[test]
    fn test_vadd_inplace_simd_zero_len() {
        let mut dst: Vec<f32> = vec![];
        vadd_inplace_simd(&mut dst, &[]);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_vadd_inplace_simd_zeros() {
        let mut dst = [1.0f32; 8];
        let src = [0.0f32; 8];
        vadd_inplace_simd(&mut dst, &src);
        for &v in &dst {
            assert_eq!(v, 1.0);
        }
    }

    // ==================== vscale_add_inplace_simd tests ====================

    #[test]
    fn test_vscale_add_inplace_simd_len1() {
        let mut dst = [1.0f32];
        let src = [2.0f32];
        vscale_add_inplace_simd(&mut dst, &src, 3.0);
        assert_eq!(dst[0], 7.0); // 1 + 2*3
    }

    #[test]
    fn test_vscale_add_inplace_simd_len4() {
        let mut dst_simd = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst_naive = [1.0f32, 2.0, 3.0, 4.0];
        let src = [5.0f32, 6.0, 7.0, 8.0];
        let scale = 0.5;

        vscale_add_inplace_simd(&mut dst_simd, &src, scale);
        vscale_add_inplace_fallback(&mut dst_naive, &src, scale);

        assert_slices_close(&dst_simd, &dst_naive, 1e-6, "vscale_add len=4");
    }

    #[test]
    fn test_vscale_add_inplace_simd_len7() {
        let mut dst_simd: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut dst_naive = dst_simd.clone();
        let src: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let scale = -0.7;

        vscale_add_inplace_simd(&mut dst_simd, &src, scale);
        vscale_add_inplace_fallback(&mut dst_naive, &src, scale);

        assert_slices_close(&dst_simd, &dst_naive, 1e-6, "vscale_add len=7");
    }

    #[test]
    fn test_vscale_add_inplace_simd_len128() {
        let mut dst_simd: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let mut dst_naive = dst_simd.clone();
        let src: Vec<f32> = (0..128).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let scale = 2.5;

        vscale_add_inplace_simd(&mut dst_simd, &src, scale);
        vscale_add_inplace_fallback(&mut dst_naive, &src, scale);

        assert_slices_close(&dst_simd, &dst_naive, 1e-5, "vscale_add len=128");
    }

    #[test]
    fn test_vscale_add_inplace_simd_zero_scale() {
        let mut dst = [1.0f32; 4];
        let src = [100.0f32; 4];
        vscale_add_inplace_simd(&mut dst, &src, 0.0);
        for &v in &dst {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_vscale_add_inplace_simd_zero_len() {
        let mut dst: Vec<f32> = vec![];
        vscale_add_inplace_simd(&mut dst, &[], 1.0);
        assert!(dst.is_empty());
    }

    // ==================== matmul_bytes_simd_2row tests ====================

    #[test]
    fn test_matmul_bytes_simd_2row_1x1() {
        let w = f32_to_le_bytes(&[3.0]);
        let x = [2.0f32];
        let mut out = [0.0f32];
        matmul_bytes_simd_2row(&mut out, &w, &x, 1, 1);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    fn test_matmul_bytes_simd_2row_2x4() {
        let w_f32: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x = [1.0f32, 2.0, 3.0, 4.0];

        let mut out_2row = vec![0.0f32; 2];
        let mut out_naive = vec![0.0f32; 2];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, 2, 4);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 2, 4);

        assert_slices_close(&out_2row, &out_naive, 1e-6, "matmul_2row 2x4");
    }

    #[test]
    fn test_matmul_bytes_simd_2row_3x4() {
        // Odd out_dim: 2 rows paired + 1 odd row
        let w_f32: Vec<f32> = (0..12).map(|i| (i + 1) as f32 * 0.1).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x = [1.0f32, 2.0, 3.0, 4.0];

        let mut out_2row = vec![0.0f32; 3];
        let mut out_naive = vec![0.0f32; 3];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, 3, 4);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 3, 4);

        assert_slices_close(&out_2row, &out_naive, 1e-6, "matmul_2row 3x4 (odd)");
    }

    #[test]
    fn test_matmul_bytes_simd_2row_7x3() {
        let w_f32: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let w = f32_to_le_bytes(&w_f32);
        let x = [1.0f32, 2.0, 3.0];

        let mut out_2row = vec![0.0f32; 7];
        let mut out_naive = vec![0.0f32; 7];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, 7, 3);
        matmul_bytes_fallback(&mut out_naive, &w, &x, 7, 3);

        assert_slices_close(&out_2row, &out_naive, 1e-6, "matmul_2row 7x3");
    }

    #[test]
    fn test_matmul_bytes_simd_2row_128x64() {
        let in_dim = 64;
        let out_dim = 128;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 2e-5, "matmul_2row 128x64");
    }

    #[test]
    fn test_matmul_bytes_simd_2row_vs_1row() {
        // Verify 2-row variant matches single-row variant exactly
        let in_dim = 64;
        let out_dim = 33; // odd
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_1row = vec![0.0f32; out_dim];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_bytes_simd(&mut out_1row, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-4, "matmul_2row vs 1row");
    }

    #[test]
    fn test_matmul_bytes_simd_2row_zero_dim() {
        let mut out = vec![0.0f32; 0];
        matmul_bytes_simd_2row(&mut out, &[], &[], 0, 0);
        assert!(out.is_empty());
    }

    // ==================== Precision tests at large dimensions ====================

    #[test]
    fn test_matmul_bytes_simd_precision_dim1024() {
        let in_dim = 1024;
        let out_dim = 4;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-4, "precision matmul_bytes dim=1024");
    }

    #[test]
    fn test_matmul_bytes_simd_precision_dim2048() {
        let in_dim = 2048;
        let out_dim = 4;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-4, "precision matmul_bytes dim=2048");
    }

    #[test]
    fn test_matmul_bytes_simd_precision_dim4096() {
        let in_dim = 4096;
        let out_dim = 4;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 2e-4, "precision matmul_bytes dim=4096");
    }

    #[test]
    fn test_matmul_simd_precision_dim1024() {
        let in_dim = 1024;
        let out_dim = 4;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-4, "precision matmul f32 dim=1024");
    }

    #[test]
    fn test_matmul_simd_precision_dim2048() {
        let in_dim = 2048;
        let out_dim = 4;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 1e-4, "precision matmul f32 dim=2048");
    }

    #[test]
    fn test_matmul_simd_precision_dim4096() {
        let in_dim = 4096;
        let out_dim = 4;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_simd = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd(&mut out_simd, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_simd, &out_naive, 2e-4, "precision matmul f32 dim=4096");
    }

    #[test]
    fn test_dot_product_simd_precision_dim2048() {
        let a: Vec<f32> = (0..2048).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
        let b: Vec<f32> = (0..2048).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-4),
            "dot len=2048: simd={simd_result}, naive={naive_result}"
        );
    }

    #[test]
    fn test_dot_product_simd_precision_dim4096() {
        let a: Vec<f32> = (0..4096).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
        let b: Vec<f32> = (0..4096).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
        let simd_result = dot_product_simd(&a, &b);
        let naive_result = dot_product_fallback(&a, &b);
        assert!(
            approx_eq(simd_result, naive_result, 1e-4),
            "dot len=4096: simd={simd_result}, naive={naive_result}"
        );
    }

    #[test]
    fn test_matmul_bytes_2row_precision_dim1024() {
        let in_dim = 1024;
        let out_dim = 4;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 1e-4, "precision matmul_2row dim=1024");
    }

    #[test]
    fn test_matmul_bytes_2row_precision_dim4096() {
        let in_dim = 4096;
        let out_dim = 5; // odd out_dim to test fallback path too
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 2e-4, "precision matmul_2row dim=4096 odd");
    }

    // ==================== Fuzz tests: random inputs, SIMD vs naive ====================

    /// Simple deterministic PRNG (xorshift32) for reproducible fuzz tests.
    /// Avoids any external dependency.
    struct Rng32 {
        state: u32,
    }

    impl Rng32 {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.state = x;
            x
        }

        /// Returns a random f32 in [-range, +range].
        fn next_f32(&mut self, range: f32) -> f32 {
            let u = self.next_u32();
            // Map u32 to [0, 1), then to [-range, +range)
            let f = (u as f32) / (u32::MAX as f32); // [0, 1)
            (f * 2.0 - 1.0) * range
        }

        /// Returns a random usize in [lo, hi) (exclusive upper bound).
        fn next_range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u32() as usize % (hi - lo))
        }
    }

    #[test]
    fn fuzz_matmul_bytes_simd_random() {
        let mut rng = Rng32::new(42);

        for trial in 0..100 {
            let out_dim = rng.next_range(1, 129);
            let in_dim = rng.next_range(1, 129);

            let w_f32: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_f32(1.0))
                .collect();
            let w_bytes = f32_to_le_bytes(&w_f32);
            let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32(1.0)).collect();

            let mut out_simd = vec![0.0f32; out_dim];
            let mut out_naive = vec![0.0f32; out_dim];

            matmul_bytes_simd(&mut out_simd, &w_bytes, &x, out_dim, in_dim);
            matmul_bytes_fallback(&mut out_naive, &w_bytes, &x, out_dim, in_dim);

            // Tolerance: relative 1e-5, absolute 1e-6.
            // For matmul, FMA vs scalar accumulation across in_dim elements
            // causes O(in_dim * eps) absolute error, so scale absolute threshold.
            let abs_tol = 1e-6 * (in_dim as f32).sqrt().max(1.0);
            for i in 0..out_dim {
                let diff = (out_simd[i] - out_naive[i]).abs();
                let mag = out_simd[i].abs().max(out_naive[i].abs()).max(1e-6);
                let rel = diff / mag;
                assert!(
                    rel < 1e-5 || diff < abs_tol,
                    "fuzz_matmul_bytes trial {trial} ({out_dim}x{in_dim}) elem {i}: \
                     simd={}, naive={}, diff={diff}, rel={rel}",
                    out_simd[i], out_naive[i]
                );
            }
        }
    }

    #[test]
    fn fuzz_rmsnorm_bytes_simd_random() {
        let mut rng = Rng32::new(123);

        for trial in 0..100 {
            let dim = rng.next_range(1, 129);

            let x: Vec<f32> = (0..dim).map(|_| rng.next_f32(2.0)).collect();
            let weights: Vec<f32> = (0..dim).map(|_| rng.next_f32(1.0)).collect();
            let w_bytes = f32_to_le_bytes(&weights);

            let mut out_simd = vec![0.0f32; dim];
            let mut out_naive = vec![0.0f32; dim];

            rmsnorm_bytes_simd(&mut out_simd, &x, &w_bytes, 1e-5);
            rmsnorm_bytes_fallback(&mut out_naive, &x, &w_bytes, 1e-5);

            for i in 0..dim {
                let diff = (out_simd[i] - out_naive[i]).abs();
                let mag = out_simd[i].abs().max(out_naive[i].abs()).max(1e-6);
                let rel = diff / mag;
                assert!(
                    rel < 1e-5 || diff < 1e-6,
                    "fuzz_rmsnorm trial {trial} (dim={dim}) elem {i}: \
                     simd={}, naive={}, diff={diff}, rel={rel}",
                    out_simd[i], out_naive[i]
                );
            }
        }
    }

    #[test]
    fn fuzz_dot_product_simd_random() {
        let mut rng = Rng32::new(456);

        for trial in 0..100 {
            let len = rng.next_range(1, 257);

            let a: Vec<f32> = (0..len).map(|_| rng.next_f32(2.0)).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32(2.0)).collect();

            let simd_result = dot_product_simd(&a, &b);
            let naive_result = dot_product_fallback(&a, &b);

            let diff = (simd_result - naive_result).abs();
            let mag = simd_result.abs().max(naive_result.abs()).max(1e-6);
            let rel = diff / mag;
            assert!(
                rel < 1e-5 || diff < 1e-6,
                "fuzz_dot trial {trial} (len={len}): \
                 simd={simd_result}, naive={naive_result}, diff={diff}, rel={rel}"
            );
        }
    }

    #[test]
    fn fuzz_swiglu_inplace_simd_random() {
        let mut rng = Rng32::new(789);

        for trial in 0..100 {
            let len = rng.next_range(1, 129);

            let gate_orig: Vec<f32> = (0..len).map(|_| rng.next_f32(3.0)).collect();
            let up: Vec<f32> = (0..len).map(|_| rng.next_f32(3.0)).collect();

            let mut gate_simd = gate_orig.clone();
            let mut gate_naive = gate_orig.clone();

            swiglu_inplace_simd(&mut gate_simd, &up);
            swiglu_inplace_fallback(&mut gate_naive, &up);

            for i in 0..len {
                let diff = (gate_simd[i] - gate_naive[i]).abs();
                let mag = gate_simd[i].abs().max(gate_naive[i].abs()).max(1e-6);
                let rel = diff / mag;
                assert!(
                    rel < 1e-5 || diff < 1e-6,
                    "fuzz_swiglu trial {trial} (len={len}) elem {i}: \
                     simd={}, naive={}, diff={diff}, rel={rel}",
                    gate_simd[i], gate_naive[i]
                );
            }
        }
    }

    #[test]
    fn fuzz_matmul_bytes_simd_2row_random() {
        let mut rng = Rng32::new(2024);

        for trial in 0..50 {
            let out_dim = rng.next_range(1, 65);
            let in_dim = rng.next_range(1, 129);

            let w_f32: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_f32(1.0))
                .collect();
            let w_bytes = f32_to_le_bytes(&w_f32);
            let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32(1.0)).collect();

            let mut out_2row = vec![0.0f32; out_dim];
            let mut out_naive = vec![0.0f32; out_dim];

            matmul_bytes_simd_2row(&mut out_2row, &w_bytes, &x, out_dim, in_dim);
            matmul_bytes_fallback(&mut out_naive, &w_bytes, &x, out_dim, in_dim);

            let abs_tol = 1e-6 * (in_dim as f32).sqrt().max(1.0);
            for i in 0..out_dim {
                let diff = (out_2row[i] - out_naive[i]).abs();
                let mag = out_2row[i].abs().max(out_naive[i].abs()).max(1e-6);
                let rel = diff / mag;
                assert!(
                    rel < 1e-5 || diff < abs_tol,
                    "fuzz_matmul_2row trial {trial} ({out_dim}x{in_dim}) elem {i}: \
                     2row={}, naive={}, diff={diff}, rel={rel}",
                    out_2row[i], out_naive[i]
                );
            }
        }
    }

    // ==================== matmul_simd_2row tests ====================

    #[test]
    fn test_matmul_simd_2row_matches_single_row_2x4() {
        let w: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let x = [1.0f32, 2.0, 3.0, 4.0];

        let mut out_2row = vec![0.0f32; 2];
        let mut out_1row = vec![0.0f32; 2];

        matmul_simd_2row(&mut out_2row, &w, &x, 2, 4);
        matmul_simd(&mut out_1row, &w, &x, 2, 4);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "matmul_f32_2row 2x4");
    }

    #[test]
    fn test_matmul_simd_2row_matches_single_row_4x4() {
        let w: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let x: Vec<f32> = (0..4).map(|i| (i + 1) as f32).collect();

        let mut out_2row = vec![0.0f32; 4];
        let mut out_1row = vec![0.0f32; 4];

        matmul_simd_2row(&mut out_2row, &w, &x, 4, 4);
        matmul_simd(&mut out_1row, &w, &x, 4, 4);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "matmul_f32_2row 4x4");
    }

    #[test]
    fn test_matmul_simd_2row_odd_dim() {
        // Odd out_dim: 2 rows paired + 1 odd row
        let w: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1).collect();
        let x = [1.0f32, 2.0, 3.0];

        let mut out_2row = vec![0.0f32; 5];
        let mut out_naive = vec![0.0f32; 5];

        matmul_simd_2row(&mut out_2row, &w, &x, 5, 3);
        matmul_fallback(&mut out_naive, &w, &x, 5, 3);

        assert_slices_close(&out_2row, &out_naive, 1e-6, "matmul_f32_2row 5x3 (odd)");
    }

    #[test]
    fn test_matmul_simd_2row_7x3() {
        let w: Vec<f32> = (0..21).map(|i| (i as f32) * 0.1).collect();
        let x = [1.0f32, 2.0, 3.0];

        let mut out_2row = vec![0.0f32; 7];
        let mut out_naive = vec![0.0f32; 7];

        matmul_simd_2row(&mut out_2row, &w, &x, 7, 3);
        matmul_fallback(&mut out_naive, &w, &x, 7, 3);

        assert_slices_close(&out_2row, &out_naive, 1e-6, "matmul_f32_2row 7x3");
    }

    #[test]
    fn test_matmul_simd_2row_128x64() {
        let in_dim = 64;
        let out_dim = 128;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 2e-5, "matmul_f32_2row 128x64");
    }

    #[test]
    fn test_matmul_simd_2row_large() {
        let in_dim = 512;
        let out_dim = 1024;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 1e-4, "matmul_f32_2row 1024x512");
    }

    #[test]
    fn test_matmul_simd_2row_vs_single_row() {
        // Verify 2-row variant matches single-row variant for various dimensions
        let in_dim = 64;
        let out_dim = 33; // odd
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_1row = vec![0.0f32; out_dim];

        matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_simd(&mut out_1row, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-4, "matmul_f32_2row vs 1row");
    }

    #[test]
    fn test_matmul_simd_2row_zero_dim() {
        let mut out = vec![0.0f32; 0];
        matmul_simd_2row(&mut out, &[], &[], 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_matmul_simd_2row_1x1() {
        let w = [3.0f32];
        let x = [2.0f32];
        let mut out = [0.0f32];
        matmul_simd_2row(&mut out, &w, &x, 1, 1);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    fn test_matmul_simd_2row_precision_dim1024() {
        let in_dim = 1024;
        let out_dim = 4;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 1e-4, "precision matmul_f32_2row dim=1024");
    }

    #[test]
    fn test_matmul_simd_2row_precision_dim4096() {
        let in_dim = 4096;
        let out_dim = 5; // odd out_dim to test fallback path too
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();

        let mut out_2row = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
        matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_naive, 2e-4, "precision matmul_f32_2row dim=4096 odd");
    }

    #[test]
    fn fuzz_matmul_simd_2row_random() {
        let mut rng = Rng32::new(3141);

        for trial in 0..50 {
            let out_dim = rng.next_range(1, 65);
            let in_dim = rng.next_range(1, 129);

            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_f32(1.0))
                .collect();
            let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32(1.0)).collect();

            let mut out_2row = vec![0.0f32; out_dim];
            let mut out_naive = vec![0.0f32; out_dim];

            matmul_simd_2row(&mut out_2row, &w, &x, out_dim, in_dim);
            matmul_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

            let abs_tol = 1e-6 * (in_dim as f32).sqrt().max(1.0);
            for i in 0..out_dim {
                let diff = (out_2row[i] - out_naive[i]).abs();
                let mag = out_2row[i].abs().max(out_naive[i].abs()).max(1e-6);
                let rel = diff / mag;
                assert!(
                    rel < 1e-5 || diff < abs_tol,
                    "fuzz_matmul_f32_2row trial {trial} ({out_dim}x{in_dim}) elem {i}: \
                     2row={}, naive={}, diff={diff}, rel={rel}",
                    out_2row[i], out_naive[i]
                );
            }
        }
    }

    // ==================== Parallel matmul tests ====================

    #[test]
    fn test_parallel_matmul_bytes_small_falls_back() {
        // out_dim=64 < PARALLEL_THRESHOLD(256): should use single-threaded path
        // and produce identical results to the 2-row version.
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 64;
        let in_dim = 32;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        // Must be bit-identical since it falls back to the same code
        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes small fallback elem {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_bytes_large_bit_identical() {
        // out_dim=512 > PARALLEL_THRESHOLD: will actually parallelize.
        // Each row's computation is independent, so output must be bit-identical.
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 512;
        let in_dim = 256;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes large elem {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_bytes_1024x512_bit_identical() {
        // Real-world-ish dimensions
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 1024;
        let in_dim = 512;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes 1024x512 elem {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_f32_large_bit_identical() {
        // Test the f32 weight variant
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 512;
        let in_dim = 256;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_f32 large elem {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_bytes_odd_dim() {
        // Non-power-of-2, odd dimensions to test edge cases in row splitting
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 511;
        let in_dim = 127;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes odd {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_bytes_single_thread() {
        // With only 1 worker thread, should still produce identical results
        let pool = crate::thread_pool::ThreadPool::new(1);
        let out_dim = 512;
        let in_dim = 256;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes single_thread {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_bytes_many_threads() {
        // More threads than typical (8), should still work correctly
        let pool = crate::thread_pool::ThreadPool::new(7);
        let out_dim = 1024;
        let in_dim = 512;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_serial = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_simd_2row(&mut out_serial, &w, &x, out_dim, in_dim);

        for i in 0..out_dim {
            assert_eq!(
                out_parallel[i].to_bits(),
                out_serial[i].to_bits(),
                "parallel_bytes many_threads {i}: parallel={}, serial={}",
                out_parallel[i],
                out_serial[i]
            );
        }
    }

    #[test]
    fn test_parallel_matmul_correctness_vs_naive() {
        // Verify parallel result matches scalar naive (not just SIMD 2-row)
        let pool = crate::thread_pool::ThreadPool::new(3);
        let out_dim = 512;
        let in_dim = 256;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = f32_to_le_bytes(&w_f32);
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let mut out_parallel = vec![0.0f32; out_dim];
        let mut out_naive = vec![0.0f32; out_dim];

        matmul_bytes_simd_parallel(&mut out_parallel, &w, &x, out_dim, in_dim, &pool);
        matmul_bytes_fallback(&mut out_naive, &w, &x, out_dim, in_dim);

        assert_slices_close(&out_parallel, &out_naive, 1e-4, "parallel vs naive 512x256");
    }

    // ==================== Q8_0 kernel tests ====================

    /// Encode f32 values into Q8_0 block format for testing.
    /// Each block: [2 bytes f16 scale][32 bytes int8 quants]
    /// Quantization: scale = max(abs(vals)) / 127, quant = round(val / scale)
    fn encode_q8_0_blocks(vals: &[f32]) -> Vec<u8> {
        let group_size = 32;
        let mut bytes = Vec::new();
        for chunk in vals.chunks(group_size) {
            let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            // Convert scale to f16 bits
            let scale_bits = f32_to_f16_bits(scale);
            bytes.extend_from_slice(&scale_bits.to_le_bytes());

            // Quantize and write 32 int8 values
            for i in 0..group_size {
                if i < chunk.len() {
                    let q = (chunk[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    bytes.push(q as u8);
                } else {
                    bytes.push(0u8);
                }
            }
        }
        bytes
    }

    /// Convert f32 to f16 bits (IEEE 754 binary16). Test helper only.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7fffff;

        if exp == 0xff {
            // Inf/NaN
            let f16_frac = if frac == 0 { 0 } else { 0x200 }; // NaN
            return ((sign << 15) | (0x1f << 10) | f16_frac) as u16;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow -> Inf
            return ((sign << 15) | (0x1f << 10)) as u16;
        }
        if new_exp <= 0 {
            // Subnormal or zero
            if new_exp < -10 {
                return (sign << 15) as u16; // too small, zero
            }
            let shift = 1 - new_exp;
            let f16_frac = ((0x800000 | frac) >> (shift + 13)) as u32;
            return ((sign << 15) | f16_frac) as u16;
        }

        let f16_frac = (frac >> 13) as u32;
        ((sign << 15) | ((new_exp as u32) << 10) | f16_frac) as u16
    }

    /// Dequantize Q8_0 blocks back to f32 (reference implementation for testing).
    fn dequant_q8_0_reference(q8_bytes: &[u8], n_elements: usize) -> Vec<f32> {
        let group_size = 32;
        let block_size = 34;
        let num_blocks = (n_elements + group_size - 1) / group_size;
        let mut out = Vec::with_capacity(n_elements);

        for b in 0..num_blocks {
            let block_start = b * block_size;
            let scale_bits = u16::from_le_bytes([q8_bytes[block_start], q8_bytes[block_start + 1]]);
            let scale = f16_to_f32_inline(scale_bits);

            for j in 0..group_size {
                let idx = b * group_size + j;
                if idx >= n_elements {
                    break;
                }
                let q = q8_bytes[block_start + 2 + j] as i8;
                out.push(scale * q as f32);
            }
        }
        out
    }

    // ---- sdot-based matmul_q8_0_simd tests ----
    // The sdot path quantizes x to int8 before computing, introducing additional
    // quantization error (~0.5-1% relative). Tests compare against the widen
    // (exact f32 x) path and against the dequant+F32 reference with relaxed tolerance.

    #[test]
    fn test_matmul_q8_0_simd_simple_32() {
        // Minimum case: 1 output row, 32 input elements (one Q8_0 block)
        let in_dim = 32;
        let out_dim = 1;

        let w_f32: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        // sdot path (quantizes x internally)
        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        // Reference: dequantize Q8_0 to f32, then naive matmul (exact x)
        let w_dequant = dequant_q8_0_reference(&w_q8, in_dim);
        let w_dequant_bytes: Vec<u8> = w_dequant.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out_ref = vec![0.0f32; out_dim];
        matmul_bytes_fallback(&mut out_ref, &w_dequant_bytes, &x, out_dim, in_dim);

        // Relaxed tolerance: double quantization (w + x) adds ~1% error
        assert_slices_close(&out_sdot, &out_ref, 1e-2, "Q8_0 sdot matmul 1x32 vs dequant+F32");
    }

    #[test]
    fn test_matmul_q8_0_simd_4x64() {
        let in_dim = 64;
        let out_dim = 4;

        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        let w_dequant = dequant_q8_0_reference(&w_q8, out_dim * in_dim);
        let w_dequant_bytes: Vec<u8> = w_dequant.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out_ref = vec![0.0f32; out_dim];
        matmul_bytes_fallback(&mut out_ref, &w_dequant_bytes, &x, out_dim, in_dim);

        assert_slices_close(&out_sdot, &out_ref, 1e-2, "Q8_0 sdot matmul 4x64 vs dequant+F32");
    }

    #[test]
    fn test_matmul_q8_0_simd_128x256() {
        // Realistic dimensions, compare sdot vs widen path
        let in_dim = 256;
        let out_dim = 128;

        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        let mut out_widen = vec![0.0f32; out_dim];
        matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

        // Compare sdot vs widen. Small weight magnitudes (0.001 scale) produce tiny
        // dot products where x quantization error is large relative to result.
        // Use absolute error bound: max |error| should be bounded by a fraction of
        // the quantization step size * dim.
        for (i, (&s, &w)) in out_sdot.iter().zip(out_widen.iter()).enumerate() {
            let diff = (s - w).abs();
            // Accept if relative error < 10% OR absolute diff < 1e-3
            let ok = diff < 1e-3 || diff / s.abs().max(w.abs()).max(1e-8) < 0.1;
            assert!(
                ok,
                "Q8_0 sdot vs widen 128x256[{i}]: sdot={s}, widen={w}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_matmul_q8_0_simd_zero_weights() {
        let in_dim = 32;
        let out_dim = 2;
        let w_f32 = vec![0.0f32; out_dim * in_dim];
        let x: Vec<f32> = (0..in_dim).map(|i| i as f32 + 1.0).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);
        let mut out = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out, &w_q8, &x, out_dim, in_dim);

        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "Q8_0 sdot zero weights: out[{i}] should be 0, got {v}");
        }
    }

    #[test]
    fn test_matmul_q8_0_simd_zero_input() {
        let in_dim = 64;
        let out_dim = 4;
        let w_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| i as f32 * 0.01).collect();
        let x = vec![0.0f32; in_dim];

        let w_q8 = encode_q8_0_blocks(&w_f32);
        let mut out = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out, &w_q8, &x, out_dim, in_dim);

        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "Q8_0 sdot zero input: out[{i}] should be 0, got {v}");
        }
    }

    #[test]
    fn test_matmul_q8_0_vs_dequant_f32_relative_error() {
        // Key quality test: sdot path vs dequant+F32.
        // Double quantization (w + x) means ~1% error is expected.
        let in_dim = 256;
        let out_dim = 64;

        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let phase = (i as f32 * 0.037).sin();
                phase * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        let w_dequant = dequant_q8_0_reference(&w_q8, out_dim * in_dim);
        let w_dequant_bytes: Vec<u8> = w_dequant.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out_ref = vec![0.0f32; out_dim];
        matmul_bytes_fallback(&mut out_ref, &w_dequant_bytes, &x, out_dim, in_dim);

        // sdot quantizes x to int8, so higher tolerance than widen path
        assert_slices_close(&out_sdot, &out_ref, 1e-2, "Q8_0 sdot vs dequant+F32");
    }

    // ---- sdot vs widen comparison tests ----
    // These verify the sdot path matches the widen path within quantization tolerance.

    #[test]
    fn test_matmul_q8_0_sdot_vs_widen_1x32() {
        let in_dim = 32;
        let out_dim = 1;

        let w_f32: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        let mut out_widen = vec![0.0f32; out_dim];
        matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_sdot, &out_widen, 1e-2, "sdot vs widen 1x32");
    }

    #[test]
    fn test_matmul_q8_0_sdot_vs_widen_128x256() {
        let in_dim = 256;
        let out_dim = 128;

        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let phase = (i as f32 * 0.037).sin();
                phase * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        let mut out_widen = vec![0.0f32; out_dim];
        matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_sdot, &out_widen, 1e-2, "sdot vs widen 128x256");
    }

    // ---- widen path tests (preserved for regression) ----
    // These verify the widen path still matches the dequant+F32 reference exactly.

    #[test]
    fn test_matmul_q8_0_widen_vs_dequant_f32() {
        let in_dim = 256;
        let out_dim = 64;

        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let phase = (i as f32 * 0.037).sin();
                phase * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_widen = vec![0.0f32; out_dim];
        matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

        let w_dequant = dequant_q8_0_reference(&w_q8, out_dim * in_dim);
        let w_dequant_bytes: Vec<u8> = w_dequant.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out_ref = vec![0.0f32; out_dim];
        matmul_bytes_fallback(&mut out_ref, &w_dequant_bytes, &x, out_dim, in_dim);

        // Widen path uses exact f32 x, so tight tolerance
        assert_slices_close(&out_widen, &out_ref, 1e-5, "widen vs dequant+F32");
    }

    // ==================== rmsnorm_q8_0_simd tests ====================

    #[test]
    fn test_rmsnorm_q8_0_simple() {
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_f32: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_q8 = vec![0.0f32; n];
        rmsnorm_q8_0_simd(&mut out_q8, &x, &w_q8, 1e-5);

        // Reference: dequantize weights, then compute RMSNorm manually
        let w_dequant = dequant_q8_0_reference(&w_q8, n);
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (ms + 1e-5f32).sqrt();
        let out_ref: Vec<f32> = x.iter()
            .zip(w_dequant.iter())
            .map(|(&xi, &wi)| xi * inv_rms * wi)
            .collect();

        assert_slices_close(&out_q8, &out_ref, 1e-4, "rmsnorm Q8_0 dim=32");
    }

    #[test]
    fn test_rmsnorm_q8_0_dim128() {
        let n = 128;
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let w_f32: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 + 1.0) * 0.1).collect();

        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_q8 = vec![0.0f32; n];
        rmsnorm_q8_0_simd(&mut out_q8, &x, &w_q8, 1e-5);

        let w_dequant = dequant_q8_0_reference(&w_q8, n);
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (ms + 1e-5f32).sqrt();
        let out_ref: Vec<f32> = x.iter()
            .zip(w_dequant.iter())
            .map(|(&xi, &wi)| xi * inv_rms * wi)
            .collect();

        assert_slices_close(&out_q8, &out_ref, 1e-4, "rmsnorm Q8_0 dim=128");
    }

    #[test]
    fn test_rmsnorm_q8_0_zero_input() {
        let n = 32;
        let x = vec![0.0f32; n];
        let w_f32: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out = vec![0.0f32; n];
        rmsnorm_q8_0_simd(&mut out, &x, &w_q8, 1e-5);

        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "rmsnorm Q8_0 zero input [{i}] should be finite, got {v}");
            assert_eq!(v, 0.0, "rmsnorm Q8_0 zero input [{i}] should be 0");
        }
    }

    // ==================== matmul_q8_0_simd_2row tests ====================

    #[test]
    fn test_matmul_q8_0_2row_2x32() {
        let in_dim = 32;
        let out_dim = 2;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_2row = vec![0.0f32; out_dim];
        matmul_q8_0_simd_2row(&mut out_2row, &w_q8, &x, out_dim, in_dim);

        let mut out_1row = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_1row, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "Q8_0 2row vs 1row 2x32");
    }

    #[test]
    fn test_matmul_q8_0_2row_3x64() {
        // Odd out_dim to test fallback path
        let in_dim = 64;
        let out_dim = 3;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_2row = vec![0.0f32; out_dim];
        matmul_q8_0_simd_2row(&mut out_2row, &w_q8, &x, out_dim, in_dim);

        let mut out_1row = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_1row, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "Q8_0 2row vs 1row 3x64");
    }

    #[test]
    fn test_matmul_q8_0_2row_128x256() {
        let in_dim = 256;
        let out_dim = 128;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_2row = vec![0.0f32; out_dim];
        matmul_q8_0_simd_2row(&mut out_2row, &w_q8, &x, out_dim, in_dim);

        let mut out_1row = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_1row, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "Q8_0 2row vs 1row 128x256");
    }

    #[test]
    fn test_matmul_q8_0_2row_1x32() {
        // Single row (out_dim=1, all handled by odd fallback)
        let in_dim = 32;
        let out_dim = 1;
        let w_f32: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_2row = vec![0.0f32; out_dim];
        matmul_q8_0_simd_2row(&mut out_2row, &w_q8, &x, out_dim, in_dim);

        let mut out_1row = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_1row, &w_q8, &x, out_dim, in_dim);

        assert_slices_close(&out_2row, &out_1row, 1e-6, "Q8_0 2row vs 1row 1x32");
    }

    #[test]
    fn test_matmul_q8_0_2row_vs_dequant_f32() {
        // Verify 2-row sdot matches dequant+F32 reference
        let in_dim = 256;
        let out_dim = 64;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let phase = (i as f32 * 0.037).sin();
                phase * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        let mut out_q8 = vec![0.0f32; out_dim];
        matmul_q8_0_simd_2row(&mut out_q8, &w_q8, &x, out_dim, in_dim);

        let w_dequant = dequant_q8_0_reference(&w_q8, out_dim * in_dim);
        let w_dequant_bytes: Vec<u8> = w_dequant.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out_ref = vec![0.0f32; out_dim];
        matmul_bytes_fallback(&mut out_ref, &w_dequant_bytes, &x, out_dim, in_dim);

        // sdot path quantizes x, so relaxed tolerance
        assert_slices_close(&out_q8, &out_ref, 1e-2, "Q8_0 sdot 2row vs dequant+F32");
    }

    // ==================== 4-row matmul_q8_0 tests ====================

    #[test]
    fn test_matmul_q8_0_4row_vs_widen() {
        // Compare 4-row output against widen reference for a realistic dimension
        let in_dim = 256;
        let out_dim = 128;
        let w_f32: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let phase = (i as f32 * 0.037).sin();
                phase * 0.5
            })
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let w_q8 = encode_q8_0_blocks(&w_f32);

        // sdot path (now uses 4-row internally)
        let mut out_sdot = vec![0.0f32; out_dim];
        matmul_q8_0_simd(&mut out_sdot, &w_q8, &x, out_dim, in_dim);

        // widen path (uses exact f32 x, no x quantization)
        let mut out_widen = vec![0.0f32; out_dim];
        matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

        // Tolerance: sdot quantizes x to int8, widen uses exact f32 x
        assert_slices_close(&out_sdot, &out_widen, 1e-2, "Q8_0 4row sdot vs widen 128x256");
    }

    #[test]
    fn test_matmul_q8_0_4row_odd_dims() {
        // Test with out_dim not divisible by 4 to exercise remainder handling.
        // Compare sdot (4-row) vs widen path. Both see quantized weights,
        // only difference is x quantization in sdot path.
        for &out_dim in &[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17] {
            let in_dim = 64;
            let w_f32: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let w_q8 = encode_q8_0_blocks(&w_f32);

            // 4-row path via matmul_q8_0_simd
            let mut out_4row = vec![0.0f32; out_dim];
            matmul_q8_0_simd(&mut out_4row, &w_q8, &x, out_dim, in_dim);

            // Widen reference (exact f32 x, no x quantization)
            let mut out_widen = vec![0.0f32; out_dim];
            matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

            // sdot quantizes x to int8, so accept relative < 10% OR absolute < 1e-3
            for (i, (&s, &w)) in out_4row.iter().zip(out_widen.iter()).enumerate() {
                let diff = (s - w).abs();
                let ok = diff < 1e-3 || diff / s.abs().max(w.abs()).max(1e-8) < 0.1;
                assert!(
                    ok,
                    "Q8_0 4row odd dims {out_dim}x{in_dim}[{i}]: sdot={s}, widen={w}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_matmul_q8_0_4row_small() {
        // Small dimensions: 4x32 (exactly 1 quad), 8x64 (2 quads)
        // Compare sdot (4-row) vs widen (exact f32 x path).
        for &(out_dim, in_dim) in &[(4, 32), (8, 64), (4, 64), (8, 32)] {
            let w_f32: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i % 53) as f32 - 26.0) * 0.02)
                .collect();
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
            let w_q8 = encode_q8_0_blocks(&w_f32);

            let mut out_4row = vec![0.0f32; out_dim];
            matmul_q8_0_simd(&mut out_4row, &w_q8, &x, out_dim, in_dim);

            // Widen reference (exact f32 x path)
            let mut out_widen = vec![0.0f32; out_dim];
            matmul_q8_0_simd_widen(&mut out_widen, &w_q8, &x, out_dim, in_dim);

            // sdot quantizes x to int8, so accept relative < 10% OR absolute < 1e-3
            for (i, (&s, &w)) in out_4row.iter().zip(out_widen.iter()).enumerate() {
                let diff = (s - w).abs();
                let ok = diff < 1e-3 || diff / s.abs().max(w.abs()).max(1e-8) < 0.1;
                assert!(
                    ok,
                    "Q8_0 4row small {out_dim}x{in_dim}[{i}]: sdot={s}, widen={w}, diff={diff}"
                );
            }
        }
    }

    // ==================== f16_to_f32_inline tests ====================

    #[test]
    fn test_f16_roundtrip() {
        // Test that our f16 encode/decode roundtrips for common values
        let test_values = [0.0f32, 1.0, -1.0, 0.5, 0.001, 100.0, -0.25];
        for &v in &test_values {
            let bits = f32_to_f16_bits(v);
            let recovered = f16_to_f32_inline(bits);
            let tol = v.abs() * 0.01 + 1e-4; // f16 has limited precision
            assert!(
                (recovered - v).abs() < tol,
                "f16 roundtrip: input={v}, bits=0x{bits:04x}, recovered={recovered}, diff={}",
                (recovered - v).abs()
            );
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert_eq!(f16_to_f32_inline(0x0000), 0.0);
        // Negative zero
        assert_eq!(f16_to_f32_inline(0x8000), -0.0);
        // Infinity
        assert_eq!(f16_to_f32_inline(0x7c00), f32::INFINITY);
        // Negative infinity
        assert_eq!(f16_to_f32_inline(0xfc00), f32::NEG_INFINITY);
        // NaN
        assert!(f16_to_f32_inline(0x7e00).is_nan());
    }
}
