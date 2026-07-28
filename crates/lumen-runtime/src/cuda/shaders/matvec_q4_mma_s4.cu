// ==========================================================================
// matvec_q4_mma_s4: Q4_0 x Q8_1 decode matvec on the INT4 TENSOR CORES.
//
// WHY THIS INSTRUCTION FAMILY
//
// The Q4 FFN is INSTRUCTION-bound, not memory-bound, and the evidence is on
// the same GPU in the same token:
//
//   lm_head (Q8_0)  1.08 GB / 0.567 ms = 1905 GB/s   ~98% of A100 PCIe peak
//   FFN     (Q4_0)  2.72 GB / 2.449 ms = 1135 GB/s   ~59% of the same peak
//
// Q4 carries half the bytes per weight, so to be bandwidth-bound it would have
// to process ~2x lm_head's weights/ms. It manages 1.13x (2.02 vs 1.79 G
// weights/ms). The shortfall is the per-byte work: nibble extraction, sign
// correction and dp4a issue. If the FFN became bandwidth-bound it would stream
// 2.716 GB at ~1905 GB/s = 1.426 ms against 2.449 now — about 1.0 ms.
//
// A100 rates dense INT4 at 1248 TOPS against 624 for INT8, and llama.cpp's Q4
// path is still dp4a rather than this instruction family, so this is the one
// lever in the campaign aimed at BEATING the reference rather than matching
// its byte count.
//
// EXACTNESS — this is not a precision shortcut
//
// The Q8_1 activation keeps its full 8-bit range. Decompose each signed int8
// exactly into two 4-bit halves plus a constant:
//
//     lo = (q8 & 15) - 8      in [-8, 7]
//     hi =  q8 >> 4           in [-8, 7]   (arithmetic shift)
//     q8 = lo + 16*hi + 8                  (exact, all q8 in [-128,127])
//
// and centre the Q4 nibble into signed s4:
//
//     w4 = nibble - 8         in [-8, 7]
//
// Feeding three of the MMA's eight N columns with (lo, hi, ones):
//
//     dot(w4, q8) = C[:,0] + 16*C[:,1] + 8*C[:,2]
//
// all in INT32. Five N columns are wasted, but INT4 tensor-core capacity far
// exceeds what streaming Q4 weights requires, and the arithmetic is bit-exact
// against the dp4a path — so the comparison with llama.cpp stays honest.
//
// FRAGMENT LAYOUT — mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
//   A: 16x32 s4, 2 x u32 per thread (16 nibbles)
//   B: 32x8  s4, 1 x u32 per thread (8 nibbles)
//   C/D: 16x8 s32, 4 x s32 per thread
//   group_id = lane / 4 (0..7), tig = lane % 4 (0..3)
//   A: a[0] holds rows group_id, k = tig*8 .. tig*8+7
//      a[1] holds rows group_id+8, same k range
//   B: b[0] holds col group_id, k = tig*8 .. tig*8+7
//   D: d[0..1] = rows group_id, cols tig*2, tig*2+1
//      d[2..3] = rows group_id+8, same cols
//
// Grid:  (ceil(out_dim / 16), 1, 1)   one warp per 16 output rows
// Block: (32, WARPS_PER_CTA, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage. Requires sm_80.

#define MMA_WARPS   4
#define MMA_M       16
#define MMA_K       32
#define Q4_BLK      32            // Q4_0 block: 32 weights
#define Q8_1_BLK    32            // Q8_1 block: 32 activations + scale/sum

__device__ __forceinline__ void mma_m16n8k32_s4(
    int (&d)[4], const unsigned int (&a)[2], unsigned int b, const int (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// Pack 8 signed 4-bit values (each already in [-8,7]) into one u32, low nibble
// first — the order the s4 fragment expects.
__device__ __forceinline__ unsigned int pack_s4x8(const int* v) {
    unsigned int r = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r |= ((unsigned int)(v[i] & 0xF)) << (4 * i);
    }
    return r;
}

extern "C" __global__ void matvec_q4_mma_s4(
    const unsigned char* __restrict__ weight_split,  // [out_dim][f16 scale x nb][nibbles]
    const unsigned char* __restrict__ q8_1,          // [nb][32 int8 + f16 d + f16 s]
    float* __restrict__ out,                         // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp = threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int row0 = (blockIdx.x * MMA_WARPS + warp) * MMA_M;
    if (row0 >= out_dim) return;

    const unsigned int gid = lane >> 2;        // 0..7
    const unsigned int tig = lane & 3;         // 0..3
    const unsigned int nb = in_dim / Q4_BLK;

    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + 16ULL);

    // Accumulate per-block, because each Q4_0 block has its own f16 scale and
    // each Q8_1 block its own d/sum — the MMA is exact within a block only.
    float acc_r0 = 0.0f, acc_r8 = 0.0f;

    for (unsigned int ib = 0; ib < nb; ib++) {
        // --- activation block: 32 int8 + f16 d + f16 sum ---
        const unsigned char* ab = q8_1 + (unsigned long long)ib * 36ULL;
        const signed char* q8 = (const signed char*)ab;
        const unsigned short d_bits = (unsigned short)(ab[32] | ((unsigned short)ab[33] << 8));
        float d_act;
        asm("cvt.f32.f16 %0, %1;" : "=f"(d_act) : "h"(d_bits));

        // This lane's 8 K-slots for the B fragment.
        int lo[8], hi[8], one[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int k = tig * 8 + j;
            const int v = (int)q8[k];
            lo[j] = (v & 15) - 8;      // in [-8,7]
            hi[j] = v >> 4;            // arithmetic shift, in [-8,7]
            one[j] = 1;
        }

        // TWO MMAs, ZERO SHUFFLES.
        //
        // v1 put lo/hi/ones in columns 0/1/2 and measured 0.196x — a 5x
        // regression — because recombining C0 + 16*C1 + 8*C2 needs terms from
        // TWO lanes (tig=0 holds cols 0,1; tig=1 holds col 2), forcing a
        // cross-lane reduction on EVERY 32-element block. At nb=128-384 that
        // is hundreds of reductions per row and it swamped the arithmetic win.
        //
        // Splitting into two MMAs puts every term a row needs on the SAME lane
        // (tig=0 holds D[gid][0] and D[gid][1] for both its rows), so the
        // per-block reduction disappears entirely. Two tensor-core ops are far
        // cheaper than ~36 shuffle ops.
        //   MMA1 B = [lo, hi] -> d0 = C_lo(gid), d1 = C_hi(gid),
        //                        d2 = C_lo(gid+8), d3 = C_hi(gid+8)
        //   MMA2 B = [ones]   -> d0 = C_ones(gid), d2 = C_ones(gid+8)
        unsigned int b_lohi = 0;
        if (gid == 0)      b_lohi = pack_s4x8(lo);
        else if (gid == 1) b_lohi = pack_s4x8(hi);
        unsigned int b_ones = (gid == 0) ? pack_s4x8(one) : 0u;

        // --- weight fragment: rows row0+gid and row0+gid+8 ---
        int wv[8];
        unsigned int afrag[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned int r = row0 + gid + (unsigned int)(half * 8);
            if (r < out_dim) {
                const unsigned char* rp = weight_split + (unsigned long long)r * row_bytes;
                const unsigned char* nib = rp + 2ULL * nb + (unsigned long long)ib * 16ULL;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int k = tig * 8 + j;
                    const int byte = (k < 16) ? (int)nib[k] : (int)nib[k - 16];
                    const int q = (k < 16) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                    wv[j] = q - 8;
                }
                afrag[half] = pack_s4x8(wv);
            } else {
                afrag[half] = 0;
            }
        }

        int zero4[4] = {0, 0, 0, 0};
        int d1f[4], d2f[4];
        mma_m16n8k32_s4(d1f, afrag, b_lohi, zero4);
        mma_m16n8k32_s4(d2f, afrag, b_ones, zero4);

        // tig==0 holds columns 0 and 1 for both its rows, so all three terms
        // land on one lane and no cross-lane step is needed.
        if (tig == 0) {
            const int dot_r0 = d1f[0] + 16 * d1f[1] + 8 * d2f[0];
            const int dot_r8 = d1f[2] + 16 * d1f[3] + 8 * d2f[2];
            const unsigned int rA = row0 + gid;
            const unsigned int rB = row0 + gid + 8;
            float sA = 0.0f, sB = 0.0f;
            if (rA < out_dim) {
                const unsigned short* sc =
                    (const unsigned short*)(weight_split + (unsigned long long)rA * row_bytes);
                asm("cvt.f32.f16 %0, %1;" : "=f"(sA) : "h"(sc[ib]));
            }
            if (rB < out_dim) {
                const unsigned short* sc =
                    (const unsigned short*)(weight_split + (unsigned long long)rB * row_bytes);
                asm("cvt.f32.f16 %0, %1;" : "=f"(sB) : "h"(sc[ib]));
            }
            acc_r0 += sA * d_act * (float)dot_r0;
            acc_r8 += sB * d_act * (float)dot_r8;
        }
    }

    if (tig == 0) {
        const unsigned int rA = row0 + gid;
        const unsigned int rB = row0 + gid + 8;
        if (rA < out_dim) out[rA] = acc_r0;
        if (rB < out_dim) out[rB] = acc_r8;
    }
}
