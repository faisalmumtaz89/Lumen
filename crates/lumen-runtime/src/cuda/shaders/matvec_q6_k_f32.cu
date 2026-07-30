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


// ---------------------------------------------------------------------------
// THE MATVEC. Rewritten after the first version measured ~90 GB/s on an A100
// (~4% MBU, ~9x off the achievable rate) and turned C1 into a REGRESSION:
// full_attn 1165 -> 2072us where the byte math predicted about -200us.
//
// WHAT WAS WRONG. Both faults are named in the header of the in-tree kernel that
// already reaches the target rate class, matvec_q4_split_f32_lane.cu:
//
//  1. OCCUPANCY. The old kernel staged `float xv[32]` -- 32 registers of
//     activations per thread -- which forced `__launch_bounds__(128, 2)`: two
//     CTAs per SM, 256 of 2048 resident threads, 12.5% occupancy. A
//     bandwidth-bound kernel at 12.5% occupancy has almost no memory-level
//     parallelism with which to hide latency. The Q4 lane kernel's header calls
//     out deleting exactly that array as "what lifts occupancy", and NR=8
//     multiplied the same pressure. Activations are now 8 floats (two float4).
//  2. COALESCING. Each thread read a 32-byte RUN of `ql`, one byte at a time, so
//     at any given instruction the warp's 32 addresses were scattered over ~840
//     bytes -- 32 useful bytes per 4+ sectors fetched, repeated 32 times per
//     unit. Lanes now read 4 CONSECUTIVE ql bytes each, so a warp covers one
//     block's 128-byte ql array contiguously, and activations are float4.
//
// ALIGNMENT -- the reason this cannot simply copy the Q4 kernel. Q6_K's block is
// 210 bytes, not a multiple of 4, so `ql` is only 2-byte aligned on odd blocks
// and a 4-byte `int` load would be illegal. 210 IS even, so 16-bit loads are
// always legal: each lane issues two `unsigned short` loads and assembles them
// with a shift+or. That recovers whole-word nibble arithmetic (the 0x0F0F0F0F /
// 0x30303030 masks, and one `__vsubss4` applying the -32 bias to four bytes at
// once) without ever misaligning an access.
//
// Shared-memory staging of the 210-byte block was considered and REJECTED: a
// block is consumed by exactly one warp, so each lane would stage ~7 bytes in
// order to read ~6.5 back -- more instructions than the short loads it saves.
//
// DECOMPOSITION. One warp owns one super-block; lane L owns `ql` bytes 4L..4L+3,
// i.e. 8 elements. Within a half, `ql` byte p carries element p in its LOW
// nibble and element p+64 in its HIGH nibble -- one uniform rule covering all
// four ggml "groups", which is what collapses the old four-way branch:
//
//   half = p / 64,  p_h = p % 64
//   low  nibble -> element half*128 + p_h        qh bit shift 2*(p_h/32)
//   high nibble -> element half*128 + p_h + 64   qh bit shift 2*(p_h/32) + 4
//   scales: sc[8*half + p_h/16] (low), sc[8*half + p_h/16 + 4] (high)
//
// Because p is a multiple of 4, all four of a lane's bytes share one scale pair
// and one qh shift, and the four qh bytes they need are consecutive -- so qh is
// two short loads too. Both element runs are contiguous, hence float4.
//
// A CTA is 128 threads = 4 warps owning NR output rows, striding blocks by 4.
// The activation float4 pair is loaded ONCE per block-step and reused across all
// NR rows, which is what amortizes the activation re-read at head scale.
//
// NUMERICS UNCHANGED from the first version: F32 accumulation in a fixed order,
// every FP op pinned with inline-PTX `.rn`, `d` applied once per super-block, and
// per-element weight terms exact in int32 before conversion.
// `q6k_ref::row_dot_kernel_order` mirrors this decomposition and is checked
// against an f64 reference; the layout tests are decomposition-independent.
// ---------------------------------------------------------------------------

// Assemble a 4-byte little-endian word from two 2-byte-aligned halves.
// `base + off` must be 2-byte aligned, which every Q6_K block offset is.
__device__ __forceinline__ unsigned int q6k_load_u32_align2(
    const unsigned char* __restrict__ base, unsigned int off)
{
    const unsigned short* p = (const unsigned short*)(base + off);
    return (unsigned int)p[0] | ((unsigned int)p[1] << 16);
}

// Dot four already -32-biased int8 codes packed in `q4` with four activations,
// in a fixed order, pinned .rn.
__device__ __forceinline__ float q6k_dot4(unsigned int q4, const float4 x, float acc)
{
    acc = q6k_fma_rn(q6k_i2f_rn((int)(signed char)(q4      )), x.x, acc);
    acc = q6k_fma_rn(q6k_i2f_rn((int)(signed char)(q4 >>  8)), x.y, acc);
    acc = q6k_fma_rn(q6k_i2f_rn((int)(signed char)(q4 >> 16)), x.z, acc);
    acc = q6k_fma_rn(q6k_i2f_rn((int)(signed char)(q4 >> 24)), x.w, acc);
    return acc;
}

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

    const unsigned int nb = in_dim / Q6K_BLOCK_ELEM;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (unsigned long long)Q6K_BLOCK_BYTE;

    // Lane -> ql bytes 4L..4L+3, and everything that follows from it.
    const unsigned int p       = lane * 4u;                 // 0..124
    const unsigned int half    = p >> 6;                    // 0 or 1
    const unsigned int p_h     = p & 63u;                   // 0..60 in the half
    const unsigned int qh_off  = half * 32u + (p_h & 31u);  // 4-aligned
    const unsigned int sh_lo   = 2u * (p_h >> 5);           // 0 or 2
    const unsigned int sc_i    = 8u * half + (p_h >> 4);    // low-nibble scale
    const unsigned int elem_lo = half * 128u + p_h;
    const unsigned int elem_hi = elem_lo + 64u;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Warp `warp_id` takes blocks warp_id, warp_id+4, ...: the CTA advances 4
    // blocks per step and each block is owned by exactly one warp.
    for (unsigned int ib = warp_id; ib < nb; ib += NWARPS) {
        // Activations: two contiguous runs of 4, loaded once for all NR rows.
        const float* xb = x + (unsigned long long)ib * Q6K_BLOCK_ELEM;
        const float4 xl = *(const float4*)(xb + elem_lo);
        const float4 xh = *(const float4*)(xb + elem_hi);

        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const unsigned int row = r0 + (unsigned int)r;
            if (NR > 1 && row >= out_dim) break;

            const unsigned char* bp = weight
                + (unsigned long long)row * row_bytes
                + (unsigned long long)ib * Q6K_BLOCK_BYTE;

            const unsigned int vl = q6k_load_u32_align2(bp, p);
            const unsigned int vh = q6k_load_u32_align2(bp, 128u + qh_off);

            const unsigned int nlo = vl & 0x0F0F0F0Fu;
            const unsigned int blo = ((vh >> sh_lo) & 0x03030303u) << 4;
            const unsigned int qlo = __vsubss4(nlo | blo, 0x20202020u);

            const unsigned int nhi = (vl >> 4) & 0x0F0F0F0Fu;
            const unsigned int bhi = ((vh >> (sh_lo + 4u)) & 0x03030303u) << 4;
            const unsigned int qhi = __vsubss4(nhi | bhi, 0x20202020u);

            const signed char* sc = (const signed char*)(bp + 192u);
            const float d = q6k_f16_to_f32(
                (unsigned short)(bp[208] | ((unsigned short)bp[209] << 8)));

            float acc = 0.0f;
            acc = q6k_fma_rn(q6k_i2f_rn((int)sc[sc_i]), q6k_dot4(qlo, xl, 0.0f), acc);
            acc = q6k_fma_rn(q6k_i2f_rn((int)sc[sc_i + 4u]), q6k_dot4(qhi, xh, 0.0f), acc);
            sumf[r] = q6k_fma_rn(d, acc, sumf[r]);
        }
    }

    // LOCKED reduction: butterfly within each warp, fixed cross-warp fold.
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

// NR=1: one output row per CTA -- the shape the fast in-tree Q4 lane kernel uses.
// For `wq` [out=8192, in=4096] that is 8192 CTAs (~76 waves over 108 SMs) with no
// activation staging, so occupancy is bounded only by the small register
// footprint. Deliberately NO `__launch_bounds__` cap: pinning CTAs/SM to 2 is the
// fault this rewrite exists to remove.
extern "C" __global__ void matvec_q6_k_f32(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q6_k_f32_impl<1>(weight, x, out, out_dim, in_dim);
}

// NR=4: four rows per CTA, for the extreme-aspect-ratio head [248320 x 4096].
// At NR=1 the head would be 248320 CTAs each re-reading the whole 16 KB
// activation vector (~4 GB of L1/L2 traffic against 834 MB of weight DRAM); NR=4
// quarters that for 62080 CTAs, still ~574x oversubscribed on 108 SMs. The reuse
// costs 8 activation floats plus 4 accumulators -- unlike the old NR=8, which
// paid 32 staged floats for it.
extern "C" __global__ void matvec_q6_k_f32_nr4(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q6_k_f32_impl<4>(weight, x, out, out_dim, in_dim);
}

// ==========================================================================
// dequant_q6_k_to_f32: Q6_K -> F32 staging for the EXACT-F32 PREFILL path.
//
// WHY THIS EXISTS
//
// `launch_gemm_projection` / `launch_gemm_residual` have an F16-cache fast path
// that normally serves a native-Q6_K weight during prefill. `LUMEN_CUDA_
// PREFILL_F32` disables that fast path (it is presence-parsed, so even `=0`
// enables the bypass) and forces every projection through cuBLAS SGEMM-F32
// against a dequantized F32 staging buffer. Q8_0 and Q4_0 already have
// `launch_dequant_q8_0_to_f32` / `launch_dequant_q4_0_to_f32` for exactly this;
// Q6_K did not, so a native-Q6_K `attn_q` hit the "not implemented" arm and
// aborted prefill. This is the missing sibling.
//
// It does NOT weaken the exact-F32 contract: the output is true F32, dequantized
// at full precision from the stored 6-bit codes, which is strictly MORE precise
// than the F16 cache the fast path would have used. The SGEMM that consumes it
// is unchanged.
//
// MAPPING: term-for-term ggml `dequantize_block_q6_K`
// (ggml/src/ggml-cuda/convert.cu:270-294 @ 3b53219), i.e. the CORRECT pairing --
// ql[l] low -> element l, ql[l+32] low -> l+32, ql[l] high -> l+64,
// ql[l+32] high -> l+96. See the header of this file and `q6k_ref` for why the
// natural reading is wrong.
//
// Grid:  (n_blocks, 1, 1)   one CTA per 256-element super-block
// Block: (64, 1, 1)         ip = tid/32 selects the half, il = tid%32
// ==========================================================================
extern "C" __global__ void dequant_q6_k_to_f32(
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int n_elements)
{
    const unsigned long long ib = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int ip  = tid >> 5;        // 0 or 1: which 128-element half
    const unsigned int il  = tid & 31u;       // 0..31 within the half

    const unsigned char* bp = w + ib * (unsigned long long)Q6K_BLOCK_BYTE;
    const unsigned long long base =
        ib * (unsigned long long)Q6K_BLOCK_ELEM + 128ull * ip + il;

    const float d = q6k_f16_to_f32(
        (unsigned short)(bp[208] | ((unsigned short)bp[209] << 8)));

    const unsigned char* ql = bp + 64u * ip + il;
    const unsigned char  qh = bp[128u + 32u * ip + il];
    const signed char*   sc = (const signed char*)(bp + 192u + 8u * ip + (il >> 4));

    if (base +  0ull < n_elements)
        y[base +  0ull] = d * (float)sc[0]
            * (float)((int)((ql[ 0] & 0x0F) | (((qh >> 0) & 3) << 4)) - 32);
    if (base + 32ull < n_elements)
        y[base + 32ull] = d * (float)sc[2]
            * (float)((int)((ql[32] & 0x0F) | (((qh >> 2) & 3) << 4)) - 32);
    if (base + 64ull < n_elements)
        y[base + 64ull] = d * (float)sc[4]
            * (float)((int)((ql[ 0] >>   4) | (((qh >> 4) & 3) << 4)) - 32);
    if (base + 96ull < n_elements)
        y[base + 96ull] = d * (float)sc[6]
            * (float)((int)((ql[32] >>   4) | (((qh >> 6) & 3) << 4)) - 32);
}
