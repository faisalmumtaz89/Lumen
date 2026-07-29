// ==========================================================================
// matvec_q6_k_f32: native Q6_K matvec, F32 activations.
//
// WHY
//
// A "Q4_0" GGUF is MIXED: llama-quantize deliberately keeps sensitive tensors
// at higher precision. On Qwen3.5-9B-Q4_0 that is `output.weight`
// (vocab 248320 x hidden 4096 = 1.017 G params, Q6_K in the file) plus `wq`
// (the fused Q+gate [4096, 8192]) on 4 of the 8 full-attention layers.
//
// Lumen has no K-quant matvec, so those tensors are dequantised to F32 at
// upload (4 B/weight) and decode then streams a 2 B/weight F16 cache, while
// llama.cpp reads Q6_K NATIVELY at 210 bytes per 256 elements =
// 0.8203 B/weight. That is 2.44x more bytes than the competitor moves on the
// identical tensor. Closing it removes a handicap; it is not a shortcut. The
// alternative (requantising to Q4_0 at 0.5625 B/weight) would buy speed by
// taking LOWER precision than the competitor and is deliberately NOT done.
//
// BLOCK LAYOUT (256 elements / 210 bytes), matching ggml `block_q6_K`
// (ggml/src/ggml-common.h:358-368 @ 3b53219):
//   [0   .. 128)  ql      low 4 bits, two elements per byte
//   [128 .. 192)  qh      high 2 bits, four elements per byte
//   [192 .. 208)  scales  16 x int8 sub-block scales (one per 16 elements)
//   [208 .. 210)  d       f16 super-block scale
//
// ELEMENT MAPPING — read this before touching the inner loop.
//
// A super-block is two independent HALVES of 128 elements. Half `h`
// (h = 0,1) uses ql + 64*h, qh + 32*h, scales + 8*h, and produces output
// elements [128*h, 128*h + 128). Within a half, l = 0..31 and is = l/16:
//
//   out[128h + l +  0] = d * sc[is + 0] * ((ql[l   ] & 0xF) | (((qh[l]>>0)&3)<<4) - 32)
//   out[128h + l + 32] = d * sc[is + 2] * ((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4) - 32)
//   out[128h + l + 64] = d * sc[is + 4] * ((ql[l   ] >>  4) | (((qh[l]>>4)&3)<<4) - 32)
//   out[128h + l + 96] = d * sc[is + 6] * ((ql[l+32] >>  4) | (((qh[l]>>6)&3)<<4) - 32)
//
// THE INVARIANT THAT IS EASY TO GET WRONG: the two nibbles of ONE ql byte
// land 64 output slots apart, never 32. Byte ql[l] carries elements l and
// l+64; byte ql[l+32] carries elements l+32 and l+96. This is fixed by the
// PACKER (ggml-quants.c quantize_row_q6_K_ref: `ql[l] = q1 | (q3 << 4)` where
// q1 = L[l] and q3 = L[l+64]), so it is not a convention we may choose.
//
// A "natural" reading that walks ql as (byte l low, byte l high, byte l+32
// low, byte l+32 high) in output order is WRONG: it puts one byte's two
// nibbles 32 slots apart, mixing the low 4 bits of element l+64 with the high
// 2 bits of element l+32. Measured on random codes that corrupts 126 of 256
// elements per super-block. Two in-tree host dequantisers currently have
// exactly that defect (see `q6k_layout_fix()` in runtime_defaults.rs); this
// kernel does NOT reproduce it, and its layout logic is regression-tested
// against the ggml packer by `lumen_runtime::q6k_ref` unit tests, which run
// on hosts with no GPU.
//
// DECOMPOSITION
//
// Unit of work = one GROUP of 32 consecutive OUTPUT elements. A super-block
// holds 8 groups (2 halves x 4 groups) and, because the half-major/group
// ordering above coincides with output order, group index `u` covers exactly
// elements [32*u, 32*u + 32). So the activation slice for unit u is a
// contiguous 32-float run at x + 32*u — no index arithmetic needed.
//
// A CTA is 128 threads (4 warps) and owns NR output rows. Thread t sweeps
// units t, t+128, ...; for each unit it loads the 32 activations ONCE into
// registers and reuses them across all NR rows, which is what amortizes the
// activation traffic on the extreme-aspect-ratio head shape
// [out=248320, in=4096] (nb=16 -> exactly 128 units, one per thread).
//
// NUMERICS: F32 accumulation with a FIXED operation order, pinned with
// inline-PTX `.rn` (round-to-nearest-even, no contraction) so the result is
// independent of compiler reassociation. Order: units ascending per thread,
// within a unit the two 16-element scale groups in order, then a fixed
// butterfly warp reduction and a fixed cross-warp fold. The per-element
// weight term `sc * (q - 32)` is computed in INT32 (exact: |sc| <= 127,
// |q-32| <= 32, so |product| <= 4064) and converted once, which is both
// cheaper and more accurate than two float multiplies. `d` is applied once
// per super-block outside the element sum — mathematically exact by
// distributivity, and what llama.cpp's own mmvq does (vecdotq.cuh:643).
//
// This is NOT bit-identical to the F32/F16-HGEMV path it replaces (different
// math on differently-rounded weights); it is a numerics-changing candidate
// and is quality-gated externally.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1) -- 4 warps x 32 threads
// Requires: in_dim % 256 == 0 (checked by the host before dispatch).
//
// NVRTC-compatible: no system includes, extern "C" linkage. Only baseline
// PTX (`cvt.f32.f16`, `add.rn.f32`, `fma.rn.f32`, `cvt.rn.f32.s32`,
// `shfl.sync.bfly.b32`), so it loads through the plain `load_fn` loader —
// the same one `matvec_q4_split_f32_lane.cu` uses. No dp4a, no mma.
// ==========================================================================

#define NW                32   // warp size
#define THREADS_PER_BLOCK 128  // 4 warps
#define NWARPS            (THREADS_PER_BLOCK / NW)

#define Q6K_BLOCK_ELEM   256   // elements per super-block
#define Q6K_BLOCK_BYTE   210   // bytes per super-block
#define Q6K_GROUP_ELEM   32    // elements per unit of work
#define Q6K_GROUPS_PER_B 8     // groups per super-block (2 halves x 4)

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float q6k_f16_to_f32(unsigned short bits) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(bits));
    return r;
}

// THE LOCK -- pinned add.rn (no contraction, no reassociation).
__device__ __forceinline__ float q6k_add_rn(float a, float b) {
    float o;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

// THE LOCK -- pinned fma.rn: acc + a*b.
__device__ __forceinline__ float q6k_fma_rn(float a, float b, float acc) {
    float o;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(o) : "f"(a), "f"(b), "f"(acc));
    return o;
}

// Exact int32 -> f32 with pinned rounding mode.
__device__ __forceinline__ float q6k_i2f_rn(int v) {
    float o;
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(o) : "r"(v));
    return o;
}

__device__ __forceinline__ float q6k_shfl_xor(float v, int m) {
    unsigned in = __float_as_uint(v), out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(out) : "r"(in), "r"(m), "r"(0xffffffffu));
    return __uint_as_float(out);
}

// LOCKED warp all-reduce (fixed butterfly order).
__device__ __forceinline__ float q6k_warp_allreduce(float v) {
    v = q6k_add_rn(v, q6k_shfl_xor(v, 16));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 8));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 4));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 2));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 1));
    return v;
}

// --------------------------------------------------------------------------
// Decode the 32 six-bit codes of unit `u` for one row and dot them with the
// 32 pre-loaded activations. Returns the group's contribution BEFORE the
// per-super-block `d` scale.
//
// unit u -> super-block ib = u / 8, half = (u % 8) / 4, group g = u % 4.
// --------------------------------------------------------------------------
__device__ __forceinline__ float q6k_unit_dot(
    const unsigned char* __restrict__ block_base,  // start of super-block ib
    unsigned int half,
    unsigned int g,
    const float* __restrict__ xv)                  // 32 activations for this unit
{
    const unsigned char* ql = block_base + 64u * half;
    const unsigned char* qh = block_base + 128u + 32u * half;
    const signed char*   sc = (const signed char*)(block_base + 192u + 8u * half);

    // Group g selects: which ql byte (l vs l+32), which nibble (low vs high),
    // the qh bit pair (2*g), and the scale base (2*g).
    //   g=0 -> ql[l]    low   g=1 -> ql[l+32] low
    //   g=2 -> ql[l]    high  g=3 -> ql[l+32] high
    const unsigned int ql_off   = (g & 1u) ? 32u : 0u;   // g=1,3 use the +32 byte
    const unsigned int hi_nib   = (g >> 1) & 1u;         // g=2,3 use the high nibble
    const unsigned int qh_shift = 2u * g;
    const signed char* scg      = sc + 2u * g;

    float acc = 0.0f;

    // Two scale sub-groups of 16 elements each (is = l/16). Hoisting the
    // scale out of the inner 16 keeps the op count down and the order fixed.
    #pragma unroll
    for (int is = 0; is < 2; is++) {
        const float sc_f = q6k_i2f_rn((int)scg[is]);
        float sub = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            const int l = is * 16 + k;
            const int lo = hi_nib ? (int)(ql[ql_off + l] >> 4)
                                  : (int)(ql[ql_off + l] & 0x0F);
            const int hb = (int)((qh[l] >> qh_shift) & 3u);
            const int q  = (lo | (hb << 4)) - 32;        // 6-bit code, offset
            sub = q6k_fma_rn(q6k_i2f_rn(q), xv[l], sub);
        }
        acc = q6k_fma_rn(sc_f, sub, acc);
    }
    return acc;
}

// --------------------------------------------------------------------------
// Generic kernel template: <NR rows per CTA>.
// --------------------------------------------------------------------------
template<int NR>
__device__ __forceinline__ void matvec_q6_k_f32_impl(
    const unsigned char* __restrict__ weight,  // [out_dim][nb * 210]
    const float* __restrict__ x,               // [in_dim]
    float* __restrict__ out,                   // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int r0      = blockIdx.x * NR;
    const unsigned int warp_id = threadIdx.x / NW;
    const unsigned int lane    = threadIdx.x % NW;

    const unsigned int nb        = in_dim / Q6K_BLOCK_ELEM;
    const unsigned int n_units   = nb * Q6K_GROUPS_PER_B;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (unsigned long long)Q6K_BLOCK_BYTE;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Each thread sweeps units with a fixed stride; the activation slice is
    // loaded once per unit and reused across all NR rows.
    for (unsigned int u = threadIdx.x; u < n_units; u += THREADS_PER_BLOCK) {
        const unsigned int ib   = u / Q6K_GROUPS_PER_B;
        const unsigned int rem  = u % Q6K_GROUPS_PER_B;
        const unsigned int half = rem / 4u;
        const unsigned int g    = rem % 4u;

        // Unit u covers output elements [32*u, 32*u + 32) -- contiguous.
        const float* xsrc = x + (unsigned long long)u * Q6K_GROUP_ELEM;
        float xv[Q6K_GROUP_ELEM];
        #pragma unroll
        for (int k = 0; k < Q6K_GROUP_ELEM; k++) xv[k] = xsrc[k];

        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const unsigned int row = r0 + (unsigned int)r;
            if (NR > 1 && row >= out_dim) break;

            const unsigned char* bp = weight
                + (unsigned long long)row * row_bytes
                + (unsigned long long)ib * Q6K_BLOCK_BYTE;

            const float d = q6k_f16_to_f32(
                (unsigned short)(bp[208] | ((unsigned short)bp[209] << 8)));

            const float acc = q6k_unit_dot(bp, half, g, xv);
            sumf[r] = q6k_fma_rn(d, acc, sumf[r]);
        }
    }

    // LOCKED reduction: butterfly within each warp, then a fixed fold over
    // the NWARPS-1 partner warps, then a single store.
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = q6k_warp_allreduce(sumf[r]);

    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) shmem[(warp_id - 1) * NR + r] = sumf[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++) {
                total = q6k_add_rn(total, shmem[w * NR + r]);
            }
            const unsigned int row = r0 + (unsigned int)r;
            if (row < out_dim) out[row] = total;
        }
    }
}

// NR=1: one output row per CTA. Used for layer projections (e.g. the fused
// Q+gate `wq` [out=8192, in=4096] -> 8192 CTAs).
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void matvec_q6_k_f32(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q6_k_f32_impl<1>(weight, x, out, out_dim, in_dim);
}

// NR=8: eight rows per CTA. Used for the extreme-aspect-ratio head shape
// [out=248320, in=4096] -> 31040 CTAs, ~287x oversubscription on 108 SMs,
// with the activation slice amortized 8x.
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void matvec_q6_k_f32_nr8(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q6_k_f32_impl<8>(weight, x, out, out_dim, in_dim);
}
