// Tiled shared-memory-staged MMQ Q8_0 GEMM for CUDA MoE prefill.
//
// Replaces the slow `mmq_q8_0_batched` matvec (shaders/mmq_q8_0.cu) for the
// large prefill GDN/FFN projections. Same Q8_0 weight format, same on-the-fly
// Q8_1 activation quant, and — critically — the SAME per-32-block
// int32-dot-then-f32-scale accumulation order required for MoE-router fidelity.
//
// THE FIX (vs the matvec): the matvec re-quantized each token's activation
// row ONCE PER OUTPUT-ROW-TILE (out_dim/NR times redundantly) and re-streamed
// weights per token. This kernel stages a tile of [BM tokens x BK k-blocks] of
// quantized activation into shared memory ONCE (cooperatively, via 8-thread
// subgroups) and a [BN rows x BK k-blocks] weight tile into shared, then every
// thread reuses them across its (token,row) outputs with NO cross-thread
// reduction.
//
// Tile: CTA = BM=32 tokens x BN=128 rows, BK=8 k-blocks (256 scalar K) per
// stage, 512 threads (16 warps). Each thread owns 1 token x 8 rows (8 F32
// accumulators, 8 int32 block-sums). No reductions. ~47.4 KB shared (< 48 KB,
// so no dynamic-shmem opt-in). grid=(ceil(out_dim/BN), ceil(batch/BM)).
//
// NUMERICS (load-bearing): for each 32-wide K-block: 8 dp4a -> exact int32 sum,
// then acc += (w_scale * x_scale) * (float)sum, F32-accumulate across blocks.
// Activation quant matches the matvec EXACTLY: x_scale=amax/127,
// x_scale_inv=(amax>0)?127/amax:0, q=__float2int_rn(x*x_scale_inv) (no clamp;
// |x|<=amax guarantees |product|<=127). The ONLY numerical difference vs the
// matvec is the F32 cross-block accumulation GROUPING (per-thread sequential
// here vs warp-tree there) — within the per-op 1e-7 / e2e 1e-5 tolerance;
// validated by the x_sumsq oracle + MoE router-selection (generated tokens).
//
// NVRTC-compatible: no system includes, extern "C" linkage, inline PTX helpers.

#define TMMQ_BM        32     // tokens per CTA
#define TMMQ_BN        128    // output rows per CTA
#define TMMQ_BK        8      // K-blocks (of 32) staged per iteration
#define TMMQ_PACKS     8      // int32 packs per 32-elem block (32 int8 / 4)
#define TMMQ_THREADS   512    // 16 warps
#define TMMQ_BLOCK     32     // Q8_0 block size (scalar K per block)
#define TMMQ_Q8_BYTES  34     // 2-byte f16 scale + 32 int8

// +1 int32 pad per token/row to avoid shared-memory bank conflicts.
#define TMMQ_XQ_STRIDE (TMMQ_BK * TMMQ_PACKS + 1)  // 65 int32 per token
#define TMMQ_XS_STRIDE (TMMQ_BK + 1)               // 9  float per token
#define TMMQ_WQ_STRIDE (TMMQ_BK * TMMQ_PACKS + 1)  // 65 int32 per row
#define TMMQ_WS_STRIDE (TMMQ_BK + 1)               // 9  float per row

// f16 bits -> f32 (single PTX instruction on SM 53+).
__device__ __forceinline__ float tmmq_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Pack 4 signed bytes into one int32 for dp4a.
__device__ __forceinline__ int tmmq_pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// dp4a.s32.s32: 4-way signed int8 dot product + int32 accumulator. SM 6.1+.
__device__ __forceinline__ int tmmq_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float tmmq_absmax4(float a, float b, float c, float d) {
    float m = fabsf(a);
    float t = fabsf(b); if (t > m) m = t;
    t = fabsf(c); if (t > m) m = t;
    t = fabsf(d); if (t > m) m = t;
    return m;
}

// Unaligned 32-bit load from a byte pointer (Q8_0 weight q-bytes are 16-bit
// aligned, NOT 32-bit aligned, so we read two 16-bit halves and combine —
// matches the matvec's w16 access pattern, NVRTC-safe).
__device__ __forceinline__ int tmmq_load_u32_from_u16pair(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;
    return (int)h[0] | ((int)h[1] << 16);
}

// ============================================================================
// Core tiled GEMM body. `residual` is null for the plain variant; non-null for
// the residual-add variant (out = residual + W @ x). Templated on a compile
// constant via a parameter so we can share the body without runtime branch in
// the hot loop (the residual read is in the epilogue only).
// ============================================================================
__device__ __forceinline__ void tmmq_q8_0_tiled_body(
    const unsigned char* __restrict__ weight_q8, // [out_dim, nb*34] row-major
    const float* __restrict__ x,                 // [batch, in_dim]
    const float* __restrict__ residual,          // [batch, out_dim] or null
    float* __restrict__ out,                     // [batch, out_dim]
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    // Shared tiles (packed int32 + f32 scales), padded to avoid bank conflicts.
    __shared__ int   s_xq[TMMQ_BM * TMMQ_XQ_STRIDE];
    __shared__ float s_xs[TMMQ_BM * TMMQ_XS_STRIDE];
    __shared__ int   s_wq[TMMQ_BN * TMMQ_WQ_STRIDE];
    __shared__ float s_ws[TMMQ_BN * TMMQ_WS_STRIDE];

    const unsigned int nb = in_dim >> 5;                 // K-blocks per row
    const unsigned long long row_bytes =
        (unsigned long long)nb * TMMQ_Q8_BYTES;

    const unsigned int tid = threadIdx.x;

    // 8-thread subgroup mapping for cooperative staging.
    const unsigned int sg    = tid >> 3;   // subgroup id
    const unsigned int lane8 = tid & 7;    // 0..7 within subgroup
    const unsigned int nsg   = TMMQ_THREADS >> 3;  // 64 subgroups

    // Warp/lane mapping for compute (each thread owns 1 token x 8 rows).
    const unsigned int warp = tid >> 5;    // 0..15
    const unsigned int lane = tid & 31;
    const unsigned int warp_m = warp & 3;  // 0..3 token group
    const unsigned int warp_n = warp >> 2; // 0..3 row group
    const unsigned int token    = warp_m * 8 + (lane & 7);   // 0..31 (local)
    const unsigned int row_chunk = lane >> 3;                // 0..3
    const unsigned int row_base  = warp_n * 32 + row_chunk * 8; // 8 rows

    const unsigned int gm = blockIdx.y * TMMQ_BM + token;       // global token
    const unsigned int gn_base = blockIdx.x * TMMQ_BN + row_base; // global row

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    // March over the K dimension in BK-block stages.
    for (unsigned int kb0 = 0; kb0 < nb; kb0 += TMMQ_BK) {

        // ---- Stage activation tile: [BM tokens x BK blocks], quantized once.
        for (unsigned int xb = sg; xb < TMMQ_BM * TMMQ_BK; xb += nsg) {
            unsigned int m  = xb / TMMQ_BK;
            unsigned int kk = xb % TMMQ_BK;
            unsigned int kblk = kb0 + kk;
            unsigned int gmm = blockIdx.y * TMMQ_BM + m;

            float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
            if (gmm < batch && kblk < nb) {
                const float* xp =
                    x + (unsigned long long)gmm * in_dim + kblk * TMMQ_BLOCK + lane8 * 4;
                v0 = xp[0]; v1 = xp[1]; v2 = xp[2]; v3 = xp[3];
            }

            float a = tmmq_absmax4(v0, v1, v2, v3);
            // width-8 max reduction within the subgroup.
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4, 8));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2, 8));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1, 8));

            // EXACT match to the matvec: scale = amax/127, inv = 127/amax.
            float x_scale = a * (1.0f / 127.0f);
            float x_scale_inv = (a > 0.0f) ? (127.0f / a) : 0.0f;

            int q0 = (int)__float2int_rn(v0 * x_scale_inv);
            int q1 = (int)__float2int_rn(v1 * x_scale_inv);
            int q2 = (int)__float2int_rn(v2 * x_scale_inv);
            int q3 = (int)__float2int_rn(v3 * x_scale_inv);

            s_xq[m * TMMQ_XQ_STRIDE + kk * TMMQ_PACKS + lane8] =
                tmmq_pack_i8x4(q0, q1, q2, q3);
            if (lane8 == 0) s_xs[m * TMMQ_XS_STRIDE + kk] = x_scale;
        }

        // ---- Stage weight tile: [BN rows x BK blocks].
        for (unsigned int wb = sg; wb < TMMQ_BN * TMMQ_BK; wb += nsg) {
            unsigned int n  = wb / TMMQ_BK;
            unsigned int kk = wb % TMMQ_BK;
            unsigned int kblk = kb0 + kk;
            unsigned int gn = blockIdx.x * TMMQ_BN + n;

            int packed = 0;
            float w_scale = 0.f;
            if (gn < out_dim && kblk < nb) {
                const unsigned char* bp = weight_q8
                    + (unsigned long long)gn * row_bytes
                    + (unsigned long long)kblk * TMMQ_Q8_BYTES;
                // q bytes start at offset 2; each subgroup lane reads one int32.
                packed = tmmq_load_u32_from_u16pair(bp + 2 + 4 * lane8);
                if (lane8 == 0) {
                    unsigned short sb = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
                    w_scale = tmmq_f16_to_f32(sb);
                }
            }
            s_wq[n * TMMQ_WQ_STRIDE + kk * TMMQ_PACKS + lane8] = packed;
            if (lane8 == 0) s_ws[n * TMMQ_WS_STRIDE + kk] = w_scale;
        }

        __syncthreads();

        // ---- Compute: each thread does 1 token x 8 rows over the BK blocks.
        if (gm < batch) {
            #pragma unroll
            for (int kk = 0; kk < TMMQ_BK; kk++) {
                if (kb0 + (unsigned)kk >= nb) break;

                const int* xq = &s_xq[token * TMMQ_XQ_STRIDE + kk * TMMQ_PACKS];
                int xp0 = xq[0], xp1 = xq[1], xp2 = xq[2], xp3 = xq[3];
                int xp4 = xq[4], xp5 = xq[5], xp6 = xq[6], xp7 = xq[7];
                float xs = s_xs[token * TMMQ_XS_STRIDE + kk];

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    unsigned int row = row_base + i;
                    const int* wp = &s_wq[row * TMMQ_WQ_STRIDE + kk * TMMQ_PACKS];
                    int sum = 0;
                    sum = tmmq_dp4a(wp[0], xp0, sum);
                    sum = tmmq_dp4a(wp[1], xp1, sum);
                    sum = tmmq_dp4a(wp[2], xp2, sum);
                    sum = tmmq_dp4a(wp[3], xp3, sum);
                    sum = tmmq_dp4a(wp[4], xp4, sum);
                    sum = tmmq_dp4a(wp[5], xp5, sum);
                    sum = tmmq_dp4a(wp[6], xp6, sum);
                    sum = tmmq_dp4a(wp[7], xp7, sum);
                    float ws = s_ws[row * TMMQ_WS_STRIDE + kk];
                    acc[i] += (ws * xs) * (float)sum;  // exact required order
                }
            }
        }

        __syncthreads();
    }

    // ---- Epilogue: write 8 outputs per thread (with optional residual add).
    if (gm < batch) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int gn = gn_base + i;
            if (gn < out_dim) {
                float y = acc[i];
                if (residual != nullptr) {
                    y += residual[(unsigned long long)gm * out_dim + gn];
                }
                out[(unsigned long long)gm * out_dim + gn] = y;
            }
        }
    }
}

// `__launch_bounds__(TMMQ_THREADS, 1)` caps ptxas register allocation to
// 65536/512 = 128 regs/thread so the 512-thread launch fits the A100 64K
// reg/block budget (without it, ptxas over-allocated -> CUDA_ERROR_LAUNCH_
// OUT_OF_RESOURCES). The `1` min-blocks hint keeps the per-thread reg ceiling
// generous (1 CTA/SM) since this kernel is compute/shmem bound, not occupancy
// bound at these large prefill shapes.

// Plain variant: out[t,r] = sum_k dequant(W[r,k]) * x[t,k].
extern "C" __global__ void __launch_bounds__(TMMQ_THREADS, 1) mmq_q8_0_tiled(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    tmmq_q8_0_tiled_body(
        (const unsigned char*)weight_q8, x, nullptr, out,
        out_dim, in_dim, batch);
}

// Residual-add variant: out[t,r] = residual[t,r] + sum_k dequant(W[r,k]) * x[t,k].
extern "C" __global__ void __launch_bounds__(TMMQ_THREADS, 1) mmq_q8_0_tiled_residual(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    tmmq_q8_0_tiled_body(
        (const unsigned char*)weight_q8, x, residual, out,
        out_dim, in_dim, batch);
}

// ============================================================================
// LEVER C: int8 TENSOR-CORE (mma.sync m16n8k32 s8.s8.s32) dense Q8_0 GEMM.
//
// Drop-in numerics-equivalent replacement for `mmq_q8_0_tiled` (the dp4a path)
// on the dense GDN/full-attn projections (qkv / alpha / beta / gate). Same Q8_0
// row-major weight format, same on-the-fly Q8_1 activation quant, and — binding —
// the SAME per-32-block int32-dot-then-f32-scale accumulation order required for
// 256-expert MoE-router fidelity. The ONLY change vs the dp4a kernel is the inner
// product engine: 8×dp4a per k32 block  ->  one mma.sync.m16n8k32.s8.s8.s32.
//
// NUMERICS (load-bearing, identical contract to the dp4a tiled kernel):
//   per k32 block: s8·s8 -> exact s32 (via mma), then acc += (w_scale*x_scale)*
//   (float)sum, F32-accumulate across blocks in per-thread register C. NO int32
//   cross-block reduction, NO split-K fixup. Activation quant byte-identical to
//   the dp4a kernel (amax/127, 127/amax, __float2int_rn). The F32 cross-block
//   accumulation GROUPING differs only in per-lane MMA fragment order vs the
//   dp4a per-thread order — within per-op 1e-7 / e2e 1e-5, exactly the same near-tie
//   class as the grouped gate+up kernel (whose bit-validated m16n8k32 maps this reuses).
//
// TILE: CTA = M16 tokens × N64 output rows. 8 warps (256 threads); warp w owns
//   the n8 sub-tile rows [8w .. 8w+7]. A (16×32 s8) is shared across all 8 warps.
//   K marched in IMC_BK=4 k32-blocks (=128 scalar K) per stage, double-buffered.
//   Register-C: each lane holds 4 s32 (mma frag) -> scaled into 4 f32 per warp.
//   grid = (ceil(out_dim/64), ceil(batch/16)). ~ small shared (< 20 KB).
//
// Fragment maps: the SAME bit-validated m16n8k32 maps as moe_grouped.cu's grouped gate+up:
//   A (16×32 .row): row=lane/4 + 8*(l&1); k_int=(lane&3)+4*(l>>1)
//   B (8×32  .col): n=lane/4;            k_int=(lane&3)+4*l
//   C (16×8  s32):  m=(i>>1)*8 + lane/4; n=(lane&3)*2 + (i&1)
// ============================================================================
#define IMC_BM   16    // tokens per CTA (one m16 mma tile)
#define IMC_BN   64    // output rows per CTA (8 warps × n8)
#define IMC_BK   4     // k32-blocks staged per K-iter (=128 scalar K)

// mma.sync m16n8k32 s8.s8.s32. A=4 b32 (16×32 s8), B=2 b32 (8×32 s8), D=4 s32.
__device__ __forceinline__ void imc_mma_m16n8k32(
    int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)a; (void)b; d[0]=d[1]=d[2]=d[3]=0;
#endif
}
// Load A (16×32 s8 row-major [m][k]) -> 4 b32/lane. Bit-validated fragment map.
__device__ __forceinline__ void imc_load_A(unsigned (&a)[4], const signed char* t, unsigned lane) {
    #pragma unroll
    for (int l=0;l<4;++l) { const int i=(int)(lane>>2)+8*(l&1); const int j=(int)(lane&3)+4*(l>>1);
        a[l]=*reinterpret_cast<const unsigned*>(t+(size_t)i*32+(size_t)j*4); }
}
// Load B (8×32 s8 row-major [n][k]) -> 2 b32/lane. Bit-validated fragment map.
__device__ __forceinline__ void imc_load_B(unsigned (&b)[2], const signed char* t, unsigned lane) {
    #pragma unroll
    for (int l=0;l<2;++l) { const int n=(int)(lane>>2); const int j=(int)(lane&3)+4*l;
        b[l]=*reinterpret_cast<const unsigned*>(t+(size_t)n*32+(size_t)j*4); }
}

extern "C" __global__ void __launch_bounds__(256, 2) mmq_q8_0_imma(
    const char* __restrict__ weight_q8,   // [out_dim, nb*34] row-major Q8_0
    const float* __restrict__ x,          // [batch, in_dim]
    float* __restrict__ out,              // [batch, out_dim]
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch)
{
    const unsigned int tid  = threadIdx.x;
    const unsigned int warp = tid >> 5;   // 0..7  -> owns n8 sub-tile rows [8w..8w+7]
    const unsigned int lane = tid & 31;
    const unsigned int nb   = in_dim >> 5;                 // K-blocks (of 32)
    const unsigned long long row_bytes = (unsigned long long)nb * TMMQ_Q8_BYTES;

    const unsigned int gm0 = blockIdx.y * IMC_BM;          // global token base
    const unsigned int gn0 = blockIdx.x * IMC_BN;          // global row base

    // Shared: A [2][IMC_BK][16][32] s8 + Ad [2][IMC_BK][16] f32;
    //         B [2][IMC_BK][64][32] s8 + Bd [2][IMC_BK][64] f32.
    __shared__ signed char sA[2][IMC_BK][16][32];
    __shared__ float       sAd[2][IMC_BK][16];
    __shared__ signed char sB[2][IMC_BK][64][32];
    __shared__ float       sBd[2][IMC_BK][64];

    // Per-warp register C: 4 s32 -> 4 f32, mapped C (16×8): element i ->
    //   m = (i>>1)*8 + lane/4 (token 0..15), n = (lane&3)*2 + (i&1) (row 0..7 in n8).
    float acc[4]; acc[0]=acc[1]=acc[2]=acc[3]=0.f;

    // 8-thread subgroup mapping (cooperative staging), mirrors dp4a kernel.
    const unsigned int sg    = tid >> 3;   // 0..31 subgroups
    const unsigned int lane8 = tid & 7;    // 0..7
    const unsigned int nsg   = 256 >> 3;   // 32 subgroups

    // ---- Staging macro: load IMC_BK k-blocks at k base kb0 into buffer `buf`. ----
    #define IMC_STAGE(buf, kb0)                                                       \
    {                                                                                 \
        /* activation: [16 tokens × IMC_BK kblks], quant once. 8-lane subgroup. */    \
        for (unsigned int xb = sg; xb < IMC_BM * IMC_BK; xb += nsg) {                  \
            const unsigned int m  = xb / IMC_BK;                                       \
            const unsigned int kk = xb % IMC_BK;                                       \
            const unsigned int kblk = (kb0) + kk;                                      \
            const unsigned int gmm  = gm0 + m;                                         \
            float v0=0.f,v1=0.f,v2=0.f,v3=0.f;                                         \
            if (gmm < batch && kblk < nb) {                                            \
                const float* xp = x + (unsigned long long)gmm*in_dim                   \
                    + kblk*TMMQ_BLOCK + lane8*4;                                       \
                v0=xp[0];v1=xp[1];v2=xp[2];v3=xp[3];                                   \
            }                                                                          \
            float a = tmmq_absmax4(v0,v1,v2,v3);                                       \
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 4, 8));                       \
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2, 8));                       \
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1, 8));                       \
            const float xs  = a * (1.0f/127.0f);                                       \
            const float inv = (a>0.f) ? (127.0f/a) : 0.f;                             \
            signed char* dst = &sA[buf][kk][m][lane8*4];                               \
            dst[0]=(signed char)__float2int_rn(v0*inv);                                \
            dst[1]=(signed char)__float2int_rn(v1*inv);                                \
            dst[2]=(signed char)__float2int_rn(v2*inv);                                \
            dst[3]=(signed char)__float2int_rn(v3*inv);                                \
            if (lane8==0) sAd[buf][kk][m]=xs;                                          \
        }                                                                             \
        /* weight: [64 rows × IMC_BK kblks] from raw Q8_0. 8-lane subgroup reads one  \
           int32 (4 s8) of the 32-wide q-block; lane8==0 reads the f16 scale. */      \
        for (unsigned int wb = sg; wb < IMC_BN * IMC_BK; wb += nsg) {                  \
            const unsigned int n  = wb / IMC_BK;                                       \
            const unsigned int kk = wb % IMC_BK;                                       \
            const unsigned int kblk = (kb0) + kk;                                      \
            const unsigned int gn = gn0 + n;                                           \
            int packed = 0; float ws = 0.f;                                            \
            if (gn < out_dim && kblk < nb) {                                           \
                const unsigned char* bp = (const unsigned char*)weight_q8              \
                    + (unsigned long long)gn*row_bytes                                 \
                    + (unsigned long long)kblk*TMMQ_Q8_BYTES;                          \
                packed = tmmq_load_u32_from_u16pair(bp + 2 + 4*lane8);                 \
                if (lane8==0) {                                                        \
                    unsigned short sbv = (unsigned short)(unsigned char)bp[0]          \
                        | ((unsigned short)(unsigned char)bp[1] << 8);                 \
                    ws = tmmq_f16_to_f32(sbv);                                         \
                }                                                                      \
            }                                                                          \
            *reinterpret_cast<int*>(&sB[buf][kk][n][lane8*4]) = packed;                \
            if (lane8==0) sBd[buf][kk][n]=ws;                                          \
        }                                                                             \
    }

    const int n_kiter = (int)((nb + IMC_BK - 1) / IMC_BK);
    IMC_STAGE(0, 0u);
    __syncthreads();

    for (int it=0; it<n_kiter; ++it) {
        const int cur = it & 1;
        const int nxt = (it+1) & 1;
        if (it+1 < n_kiter) { IMC_STAGE(nxt, (unsigned int)((it+1)*IMC_BK)); }

        #pragma unroll
        for (int kk=0; kk<IMC_BK; ++kk) {
            const unsigned int kblk = (unsigned int)(it*IMC_BK + kk);
            if (kblk >= nb) continue;
            unsigned a[4]; imc_load_A(a, &sA[cur][kk][0][0], lane);
            unsigned b[2]; imc_load_B(b, &sB[cur][kk][warp*8][0], lane);
            int d[4]={0,0,0,0};
            imc_mma_m16n8k32(d, a, b);
            // C map: i -> m=(i>>1)*8+lane/4, n=(lane&3)*2+(i&1) (row in n8).
            #pragma unroll
            for (int i=0;i<4;++i) {
                const int m  = (i>>1)*8 + (int)(lane>>2);
                const int nn = (int)((lane&3)*2) + (i&1);
                const float xs = sAd[cur][kk][m];
                const float ws = sBd[cur][kk][warp*8 + nn];
                acc[i] = __fmaf_rn(xs*ws, (float)d[i], acc[i]);
            }
        }
        __syncthreads();
    }

    // Epilogue: each lane writes its 4 C elements (distinct (m,n)).
    #pragma unroll
    for (int i=0;i<4;++i) {
        const int m  = (i>>1)*8 + (int)(lane>>2);
        const int nn = (int)((lane&3)*2) + (i&1);
        const unsigned int gm = gm0 + (unsigned int)m;
        const unsigned int gn = gn0 + warp*8 + (unsigned int)nn;
        if (gm < batch && gn < out_dim) {
            out[(unsigned long long)gm*out_dim + gn] = acc[i];
        }
    }
    #undef IMC_STAGE
}
