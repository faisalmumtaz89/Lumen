// ==========================================================================
// Q8 split-layout matvec against pre-quantized Q8_1 input -- llama mmvq port.
//
// A faithful port of llama.cpp's `mul_mat_vec_q` work-decomposition onto the
// Lumen split (SoA) Q8 layout produced by repack_q8_raw_to_split:
//
//   Per row (in `nb`-block units):
//     [f16 scale * nb][int8 quant[32] * nb]
//   Row stride: 34*nb bytes.  Scale stream @ offset 0, quant stream @ 2*nb.
//
// DELTA vs matvec_q8_split_q8_1 (the scalar NR=2 kernel)
// ------------------------------------------------------
// The scalar kernel assigns ONE complete 32-element quant block to a single
// lane (an 8-deep dependent dp4a chain), processes NR=2 output rows per thread,
// and performs a FULL warp all-reduce in EVERY warp for EVERY row (4 shuffle
// trees / output). This mmvq kernel instead:
//
//   * 128 threads = 4 warps, ONE output row per CTA (grid = out_dim).
//   * VDR striping: FOUR lanes cooperate on one Q8 block. Each lane owns two
//     int32 weight words (frag = tid&3 -> words 2*frag, 2*frag+1) and executes
//     only TWO dp4a's, cutting the dependency chain 4x and the per-lane live
//     activation state from 8 int32 to 2.
//   * llama lane-preserving cross-warp reduction: warps 1-3 write their 32 lane
//     partials to shared memory and return; warp 0 folds the three lane-aligned
//     partials into its own lane and executes ONE final warp reduction. That is
//     ONE shuffle tree / output instead of four (the highest-confidence delta).
//
// The SoA split layout is KEPT (not llama's AoS) -- the quant stream is at least
// as dense and better-aligned than the 36-byte aligned AoS. The lane->byte
// mapping is what changes: within a warp the eight 8-byte fragments cover a
// dense 256-byte quant-stream interval.
//
// PRECISION: this is NOT byte-identical to the scalar kernel. Integer dp4a
// products stay EXACT, but llama scales/reduces four F32 lane fragments per
// block and the cross-warp/warp reduction order changes, so the F32 rounding
// tree differs. It is a quality-equivalent NEAR-TIE gated on the FULL GQ +
// MoE-router-stability check (this kernel feeds MoE experts), NOT DET byte
// identity. The F32 op sequence is pinned with `.rn` inline-PTX (mirroring
// matvec_q4_split_q8_1_locked.cu) so the result is stable across NVRTC/driver
// JITs and free of fast-math contraction drift.
//
// Two kernels:
//   1. matvec_q8_split_q8_1_mmvq:          W*x -> out
//   2. matvec_q8_split_q8_1_mmvq_residual: W*x + residual -> out
//
// Alignment contract (inherited from the split repack, which enforces even nb):
//   * nb even => row stride 34*nb, scale stream (2*nb) and quant stream base
//     are all 4-byte aligned. Per-lane quant fragment @ ib*32 + 8*frag and
//     input fragment @ ib*36 + 4 + 8*frag are 4-byte aligned. No 16-byte
//     alignment (int4) requirement -- two scalar int loads per lane.
//
// Requires compute capability >= 6.1 for dp4a (Pascal+).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NW       32
#define THREADS_PER_BLOCK 128  // 4 warps, ONE output row per CTA
#define NWARPS   (THREADS_PER_BLOCK / NW)  // 4
#define Q8_1_BYTES        36   // 2B f16 scale + 2B f16 sum + 32B int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper (matches matvec_q8_split_q8_1.cu: the `__dp4a`
// intrinsic NVRTC-fails PTX-load on this sm_80 build env).
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

// LOCKED per-fragment Q8 epilogue: acc += w_scale * (x_scale * float(si)).
// Pinned cvt/mul/fma.rn op sequence -- no contraction, JIT-stable.
__device__ __forceinline__ float q8_frag_epilogue_locked(
    float acc, int si, float w_scale, float x_scale)
{
    asm volatile(
        "{\n\t"
        "  .reg .f32 fsi, prod;\n\t"
        "  cvt.rn.f32.s32 fsi, %1;\n\t"
        "  mul.rn.f32     prod, %3, fsi;\n\t"   // x_scale * si
        "  fma.rn.f32     %0, %2, prod, %0;\n\t" // acc += w_scale * prod
        "}\n\t"
        : "+f"(acc)
        : "r"(si), "f"(w_scale), "f"(x_scale));
    return acc;
}

// LOCKED shuffle (round-neutral bit move).
__device__ __forceinline__ float shfl_xor_f32_locked(float v, int m) {
    unsigned in = __float_as_uint(v), out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(out) : "r"(in), "r"(m), "r"(0xffffffffu));
    return __uint_as_float(out);
}

// LOCKED warp all-reduce (single 5-shuffle tree, pinned adds).
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
__device__ __forceinline__ void matvec_q8_split_mmvq_body(
    const char* __restrict__ weight_q8_split,
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
    int frag = (int)(tid & 3u);               // 0..3: which lane owns the block
    unsigned int ib0 = tid >> 2;              // 0..31: first block for this lane

    unsigned int nb = in_dim >> 5;            // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    const char* row_base   = weight_q8_split + (unsigned long long)row * row_bytes;
    const char* w_quants   = row_base + scales_bytes_per_row;  // quant stream @ 2*nb

    float tmp = 0.0f;

    // Stride 32: the 32 distinct ib0 values (one per 4-lane group) advance by 32
    // each iteration, so the whole CTA covers 32 blocks per pass.
    for (unsigned int ib = ib0; ib < nb; ib += 32) {
        // Lane's two weight words: bytes ib*32 + 8*frag (4-byte aligned).
        const int* wq = (const int*)(w_quants + (unsigned long long)ib * 32ULL) + 2 * frag;
        // Lane's two activation words: bytes ib*36 + 4 + 8*frag (4-byte aligned).
        const int* xq = (const int*)(input_q8_1 + (unsigned long long)ib * 36ULL + 4) + 2 * frag;

        int si = dp4a_s32(wq[0], xq[0], 0);
        si     = dp4a_s32(wq[1], xq[1], si);

        // Per-block scales (all 4 lanes of the block load the same scalars --
        // llama does the same; the fragment scaling is what makes this a tie).
        float w_scale = f16_bits_to_f32(
            *(const unsigned short*)(row_base + (unsigned long long)ib * 2ULL));
        float x_scale = f16_bits_to_f32(
            *(const unsigned short*)(input_q8_1 + (unsigned long long)ib * 36ULL));

        tmp = q8_frag_epilogue_locked(tmp, si, w_scale, x_scale);
    }

    // ---- llama lane-preserving cross-warp reduction ----
    // Warps 1-3 stash their 32 lane partials and return; warp 0 folds the three
    // lane-aligned partials into its own lane then runs ONE warp reduction.
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

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1_mmvq(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_mmvq_body(weight_q8_split, input_q8_1, out, out_dim, in_dim, 0);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2) void matvec_q8_split_q8_1_mmvq_residual(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34] split bytes
    const char* __restrict__ input_q8_1,         // [nb * 36] Q8_1 pre-quantized input
    const float* __restrict__ residual,          // [out_dim] F32 residual
    float* __restrict__ out,                     // [out_dim] F32 output
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_mmvq_body(weight_q8_split, input_q8_1, out, out_dim, in_dim, residual);
}
