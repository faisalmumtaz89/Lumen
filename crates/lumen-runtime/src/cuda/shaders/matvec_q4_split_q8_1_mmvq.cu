// ==========================================================================
// Q4 split-layout matvec against pre-quantized Q8_1 input -- llama mmvq port.
//
// A faithful port of llama.cpp's `mul_mat_vec_q` work-decomposition onto the
// Lumen split (SoA) Q4 layout produced by repack_q4_raw_to_split:
//
//   Per row (in `nb`-block units):
//     [f16 scale * nb][nibble[16] * nb]
//   Row stride: 18*nb bytes.  Scale stream @ offset 0, nibble stream @ 2*nb.
//
// DELTA vs matvec_q4_split_q8_1 (the scalar NR=4 kernel)
// ------------------------------------------------------
// The scalar kernel assigns ONE complete 32-element Q4 block to a single lane
// (an 8-deep dependent dp4a chain over the 4 nibble words), processes NR=4
// output rows per thread with 256 threads / 8 warps, and performs a FULL warp
// all-reduce in EVERY warp for EVERY row (8 shuffle trees / output). This mmvq
// kernel instead:
//
//   * 128 threads = 4 warps, ONE output row per CTA (grid = out_dim).
//   * VDR striping: TWO lanes cooperate on one Q4 block. Each lane owns two of
//     the four nibble words (frag = tid&1 -> words 2*frag, 2*frag+1) and
//     executes FOUR dp4a's (lo+hi per word), halving the dependency chain and
//     the per-lane live activation state.
//   * llama lane-preserving cross-warp reduction: warps 1-3 write their 32 lane
//     partials to shared memory and return; warp 0 folds the three lane-aligned
//     partials into its own lane and executes ONE final warp reduction. That is
//     ONE shuffle tree / output instead of eight.
//
// CRITICAL: per-fragment -4*xsum zero-point correction
// ----------------------------------------------------
// The Q4_0 dequant identity is
//     dot(w,x) = w_scale * (x_scale * Sum(nibble_i * q_i) - 8 * x_sum)
// where x_sum = x_scale * Sum(q_i) is precomputed in the Q8_1 input block.
// In the scalar kernel one lane owns the WHOLE block and applies `-8*x_sum`
// once. Here TWO lanes each own HALF the block, so each lane must apply only
// `-4*x_sum`. The two fragment partials sum to
//     w_scale*(x_scale*(si0+si1) - 8*x_sum) = w_scale*(x_scale*si_full - 8*x_sum),
// exactly the full-block correction. Applying -8*x_sum in BOTH lanes would
// DOUBLE the zero-point correction and corrupt the result.
//
// The SoA split layout is KEPT (not llama's AoS). The de-interleaved nibble
// mapping is unchanged from the scalar kernel: for word k, low nibbles pair
// with xv[k] and high nibbles with xv[k+4]; each fragment covers its two words.
//
// PRECISION: NOT byte-identical. Integer dp4a products stay EXACT, but the
// per-fragment F32 scaling/correction and the changed reduction order alter the
// F32 rounding tree -- a quality-equivalent NEAR-TIE gated on the FULL GQ +
// MoE-router-stability check, NOT DET byte identity. The F32 op sequence is
// pinned with `.rn` inline-PTX (mirroring matvec_q4_split_q8_1_locked.cu) so
// it is stable across NVRTC/driver JITs and free of contraction drift.
//
// Two kernels:
//   1. matvec_q4_split_q8_1_mmvq:          W*x -> out
//   2. matvec_q4_split_q8_1_mmvq_residual: W*x + residual -> out
//
// Alignment contract (inherited from the split repack, which enforces even nb):
//   * nb even => row stride 18*nb, scale stream (2*nb) and nibble stream base
//     are all 4-byte aligned. Per-lane nibble fragment @ ib*16 + 8*frag is
//     4-byte aligned. No 16-byte (int4) requirement -- scalar uint loads.
//
// Requires compute capability >= 6.1 for dp4a (Pascal+).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NW       32
#define THREADS_PER_BLOCK 128  // 4 warps, ONE output row per CTA
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// THE LOCK -- pinned add.rn (mirrors matvec_q4_split_q8_1_locked.cu).
__device__ __forceinline__ float fadd_rn_locked(float a, float b) {
    float o;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

// LOCKED per-fragment Q4 epilogue:
//   acc += w_scale * (x_scale * float(si) + (-4) * x_sum)
// where the -4 (NOT -8) is the HALF-block zero-point correction owned by this
// fragment. Pinned cvt/mul/fma.rn sequence -- mirrors the locked kernel's
// q4_epilogue_locked with km4 in place of km8.
__device__ __forceinline__ float q4_frag_epilogue_locked(
    float acc, int si, float w_scale, float x_scale, float x_sum)
{
    const float km4 = -4.0f;
    asm volatile(
        "{\n\t"
        "  .reg .f32 fsi, inner;\n\t"
        "  cvt.rn.f32.s32 fsi, %1;\n\t"
        "  mul.rn.f32     fsi, %3, fsi;\n\t"       // x_scale * si
        "  fma.rn.f32     inner, %5, %4, fsi;\n\t" // (-4)*x_sum + x_scale*si
        "  fma.rn.f32     %0, %2, inner, %0;\n\t"  // acc += w_scale * inner
        "}\n\t"
        : "+f"(acc)
        : "r"(si), "f"(w_scale), "f"(x_scale), "f"(x_sum), "f"(km4));
    return acc;
}

__device__ __forceinline__ float shfl_xor_f32_locked(float v, int m) {
    unsigned in = __float_as_uint(v), out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(out) : "r"(in), "r"(m), "r"(0xffffffffu));
    return __uint_as_float(out);
}

__device__ __forceinline__ float warp_allreduce_sum_locked(float v) {
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 16));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 8));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 4));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 2));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 1));
    return v;
}

// ==========================================================================
// mmvq body. grid = out_dim (1 row/CTA), block = 128 (4 warps).
// ==========================================================================
__device__ __forceinline__ void matvec_q4_split_mmvq_body(
    const char* __restrict__ weight_q4_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    const float* __restrict__ residual)
{
    unsigned int row = blockIdx.x;            // ONE row per CTA
    if (row >= out_dim) return;               // uniform (grid == out_dim), safe

    unsigned int tid     = threadIdx.x;
    unsigned int warp_id = tid / NW;
    unsigned int lane    = tid % NW;
    int frag = (int)(tid & 1u);               // 0..1: which half-block this lane owns
    unsigned int ib0 = tid >> 1;              // 0..63: first block for this lane

    unsigned int nb = in_dim >> 5;            // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 18ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    const char* row_base    = weight_q4_split + (unsigned long long)row * row_bytes;
    const char* w_nibbles   = row_base + scales_bytes_per_row;  // nibble stream @ 2*nb

    float tmp = 0.0f;

    // Stride 64: the 64 distinct ib0 values (one per 2-lane pair) advance by 64
    // each iteration, so the whole CTA covers 64 blocks per pass.
    for (unsigned int ib = ib0; ib < nb; ib += 64) {
        // Lane's two nibble words: bytes ib*16 + 8*frag (4-byte aligned).
        const unsigned int* wq =
            (const unsigned int*)(w_nibbles + (unsigned long long)ib * 16ULL) + 2 * frag;
        // Full block activation base (NOT fragment-offset): the de-interleaved
        // mapping indexes xv[2*frag+i] (lo) and xv[2*frag+i+4] (hi).
        const int* xq = (const int*)(input_q8_1 + (unsigned long long)ib * 36ULL + 4);

        int si = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            unsigned int p = wq[i];                       // weight word 2*frag+i
            int lo = (int)(p & 0x0F0F0F0Fu);
            int hi = (int)((p >> 4) & 0x0F0F0F0Fu);
            si = dp4a_s32(lo, xq[2 * frag + i],     si);  // low nibbles -> xv[k]
            si = dp4a_s32(hi, xq[2 * frag + i + 4], si);  // high nibbles -> xv[k+4]
        }

        float w_scale = f16_bits_to_f32(
            *(const unsigned short*)(row_base + (unsigned long long)ib * 2ULL));
        float x_scale = f16_bits_to_f32(
            *(const unsigned short*)(input_q8_1 + (unsigned long long)ib * 36ULL));
        float x_sum = f16_bits_to_f32(
            *(const unsigned short*)(input_q8_1 + (unsigned long long)ib * 36ULL + 2));

        // HALF-block zero-point correction: -4*x_sum per fragment (two fragments
        // sum to the full -8*x_sum).
        tmp = q4_frag_epilogue_locked(tmp, si, w_scale, x_scale, x_sum);
    }

    // ---- llama lane-preserving cross-warp reduction ----
    __shared__ float partial[NWARPS - 1][NW];  // [3][32] = 384 bytes

    if (warp_id > 0) {
        partial[warp_id - 1][lane] = tmp;
    }
    __syncthreads();

    if (warp_id > 0) {
        return;
    }

    tmp = fadd_rn_locked(tmp, partial[0][lane]);
    tmp = fadd_rn_locked(tmp, partial[1][lane]);
    tmp = fadd_rn_locked(tmp, partial[2][lane]);

    tmp = warp_allreduce_sum_locked(tmp);

    if (lane == 0) {
        if (residual != 0) {
            tmp = fadd_rn_locked(tmp, residual[row]);
        }
        out[row] = tmp;
    }
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q4_split_q8_1_mmvq(
    const char* __restrict__ weight_q4_split,    // [out_dim * nb * 18] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_split_mmvq_body(weight_q4_split, input_q8_1, out, out_dim, in_dim, 0);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q4_split_q8_1_mmvq_residual(
    const char* __restrict__ weight_q4_split,    // [out_dim * nb * 18] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,          // [out_dim] F32 residual
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_split_mmvq_body(weight_q4_split, input_q8_1, out, out_dim, in_dim, residual);
}
