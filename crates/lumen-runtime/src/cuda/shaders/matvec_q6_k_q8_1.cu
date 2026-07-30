// ==========================================================================
// matvec_q6_k_q8_1: native Q6_K weights x PRE-QUANTIZED Q8_1 activations (dp4a).
//
// WHY THIS IS ITS OWN FILE -- read before merging it back.
//
// It was originally appended to matvec_q6_k_f32.cu to share that file's verified
// lane mapping and `.rn` primitives. That broke BOTH Q6_K candidates on the A100:
// `matvec_q6_k_f32`, `matvec_q6_k_f32_nr4` and `dequant_q6_k_to_f32` are loaded
// through the PLAIN `load_fn`, whose NVRTC invocation passes no
// `--gpu-architecture` and therefore targets a default below sm_61. Adding
// `dp4a.s32.s32` to that translation unit made the WHOLE MODULE fail with
// CUDA_ERROR_INVALID_PTX, so all three handles became `None` -- C1 and C3 both
// silently disabled, and the head microbench panicked on the compile error.
//
// NVRTC compiles a translation unit, not a kernel. Arch requirements are a
// property of the FILE, so a shader may only contain opcodes its loader's arch
// supports. Any sm_61+ opcode (`dp4a`), sm_80+ opcode (`mma.sync`), or
// unguarded `cvt.rn.bf16.f32` must live in a file loaded through
// `load_fn_sm61` / `load_fn_sm80` / `load_fn_sm80_fast_math`.
// `q6k_ref::plain_load_fn_shaders_contain_no_raised_arch_opcodes` enforces this.
//
// The helpers below are duplicated from matvec_q6_k_f32.cu rather than shared:
// NVRTC has no include path, and every sibling dp4a shader in this directory
// carries its own `f16_bits_to_f32` / `dp4a_s32` copies for the same reason.
// The DECODE LOGIC is identical by construction and is checked against the F32
// route by `q6k_ref::dp4a_route_matches_the_f32_route`.
//
// WHAT IT COMPUTES
//
// The F32-activation sibling is COMPUTE-bound: F32 activations allow one `fma`
// per element with no way to fold four multiply-accumulates into one
// instruction. `dp4a` does exactly that fold, which is the only way a Q6_K
// matvec competes with an int8 route.
//
// Scoped to the FOUR coupled `wq` surfaces (Qwen3.5-9B-Q4_0 full-attention
// layers 3/15/27/31): those are the sites where an int8 activation plan has
// ALREADY pre-quantized the activation into `scratch.input_q8_1`, so the buffer
// this consumes exists for free -- and under such a plan they currently divert
// to F16 HGEMV at 2.0 B/weight, so this replaces the worst read in the attention
// block with 0.8203 B/weight.
//
// SEMANTICS: `vec_dot_q6_K_q8_1` (ggml/src/ggml-cuda/vecdotq.cuh:620-644 @
// 3b53219) on this repo's lane mapping rather than llama's `iqs` striding. Per
// 4-byte group: mask the nibbles, splice the two high bits, apply the -32 bias to
// all four bytes with one `__vsubss4`, `dp4a` against four int8 activations,
// scale by the int8 sub-block scale, then by the Q8_1 block's f32 scale. `d` once
// per super-block.
//
// LANE MAPPING (identical to the F32 kernel; see that file's header for the
// derivation): one warp owns one super-block, lane L owns `ql` bytes 4L..4L+3,
// and within a half `ql` byte p carries element p in its LOW nibble and element
// p+64 in its HIGH nibble.
//
// Q6_K's 210-byte block is only 2-byte aligned on odd blocks, so `ql`/`qh` are
// read as two `unsigned short` loads assembled with a shift+or -- never a 4-byte
// `int` load, which would be illegal.
//
// Q8_1 layout: 36 bytes per 32 elements, `[f16 d][f16 sum][32 x int8]`, quants at
// +4. A lane's four elements are 4-byte aligned inside it (`elem_lo % 4 == 0`,
// 36 % 4 == 0), so the activation read is one aligned `int`. The `sum` half-word
// is deliberately UNREAD: Q4_0/Q4_1 need it to correct their weight offset, but
// Q6_K applies -32 per element before the dot, so reading it would
// double-correct. llama.cpp's impl has the same shape.
// `q6k_ref::q8_1_sum_field_is_unread_by_q6k` pins that.
//
// NUMERICS: the integer dot is exact and order-independent; every FP op is pinned
// with inline-PTX `.rn`, so the F32 result is independent of compiler
// reassociation and immune to the `--use_fast_math` this loads under.
//
// Grid:  (out_dim, 1, 1)   one row per CTA
// Block: (128, 1, 1)       4 warps
// Requires: SM 6.1+ (dp4a). MUST load via load_fn_sm80_fast_math.
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NW               32
#define NWARPS           4
#define Q6K_BLOCK_ELEM   256
#define Q6K_BLOCK_BYTE   210
#define Q8_1_BYTES       36

__device__ __forceinline__ float q6k_f16_to_f32(unsigned short bits) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(bits));
    return r;
}

__device__ __forceinline__ float q6k_add_rn(float a, float b) {
    float o;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

__device__ __forceinline__ float q6k_mul_rn(float a, float b) {
    float o;
    asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

__device__ __forceinline__ float q6k_fma_rn(float a, float b, float acc) {
    float o;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(o) : "f"(a), "f"(b), "f"(acc));
    return o;
}

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

__device__ __forceinline__ float q6k_warp_allreduce(float v) {
    v = q6k_add_rn(v, q6k_shfl_xor(v, 16));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 8));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 4));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 2));
    v = q6k_add_rn(v, q6k_shfl_xor(v, 1));
    return v;
}

// Assemble a 4-byte little-endian word from two 2-byte-aligned halves.
__device__ __forceinline__ unsigned int q6k_load_u32_align2(
    const unsigned char* __restrict__ base, unsigned int off)
{
    const unsigned short* p = (const unsigned short*)(base + off);
    return (unsigned int)p[0] | ((unsigned int)p[1] << 16);
}

// inline-PTX dp4a. The `__dp4a` intrinsic NVRTC-fails in this build env; the
// raw opcode loads cleanly on compute_80. Same wrapper as the sibling dp4a
// shaders in this directory.
__device__ __forceinline__ int q6k_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

extern "C" __global__ void matvec_q6_k_q8_1(
    const unsigned char* __restrict__ weight,     // [out_dim][nb * 210]
    const char* __restrict__ input_q8_1,          // [in_dim/32][36]
    float* __restrict__ out,                      // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int warp_id = threadIdx.x / NW;
    const unsigned int lane    = threadIdx.x % NW;

    const unsigned int nb = in_dim / Q6K_BLOCK_ELEM;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (unsigned long long)Q6K_BLOCK_BYTE;
    const unsigned char* rp = weight + (unsigned long long)row * row_bytes;

    // Identical lane mapping to the F32 kernel -- see this file's header.
    const unsigned int p       = lane * 4u;
    const unsigned int half    = p >> 6;
    const unsigned int p_h     = p & 63u;
    const unsigned int qh_off  = half * 32u + (p_h & 31u);
    const unsigned int sh_lo   = 2u * (p_h >> 5);
    const unsigned int sc_i    = 8u * half + (p_h >> 4);
    const unsigned int elem_lo = half * 128u + p_h;
    const unsigned int elem_hi = elem_lo + 64u;

    float sumf = 0.0f;

    for (unsigned int ib = warp_id; ib < nb; ib += NWARPS) {
        const unsigned char* bp = rp + (unsigned long long)ib * Q6K_BLOCK_BYTE;

        const unsigned int vl = q6k_load_u32_align2(bp, p);
        const unsigned int vh = q6k_load_u32_align2(bp, 128u + qh_off);

        const unsigned int nlo = vl & 0x0F0F0F0Fu;
        const unsigned int blo = ((vh >> sh_lo) & 0x03030303u) << 4;
        const unsigned int qlo = __vsubss4(nlo | blo, 0x20202020u);

        const unsigned int nhi = (vl >> 4) & 0x0F0F0F0Fu;
        const unsigned int bhi = ((vh >> (sh_lo + 4u)) & 0x03030303u) << 4;
        const unsigned int qhi = __vsubss4(nhi | bhi, 0x20202020u);

        // Q8_1 activations: one aligned int per group, plus the block's f32
        // scale. The `sum` half-word at +2 is unused by Q6_K (see header).
        const unsigned int g_lo = ib * Q6K_BLOCK_ELEM + elem_lo;
        const unsigned int g_hi = ib * Q6K_BLOCK_ELEM + elem_hi;
        const char* xb_lo = input_q8_1 + (unsigned long long)(g_lo >> 5) * Q8_1_BYTES;
        const char* xb_hi = input_q8_1 + (unsigned long long)(g_hi >> 5) * Q8_1_BYTES;

        const int u_lo = *(const int*)(xb_lo + 4 + (g_lo & 31u));
        const int u_hi = *(const int*)(xb_hi + 4 + (g_hi & 31u));

        const float d8_lo = q6k_f16_to_f32((unsigned short)((unsigned char)xb_lo[0]
                            | ((unsigned short)(unsigned char)xb_lo[1] << 8)));
        const float d8_hi = q6k_f16_to_f32((unsigned short)((unsigned char)xb_hi[0]
                            | ((unsigned short)(unsigned char)xb_hi[1] << 8)));

        const signed char* sc = (const signed char*)(bp + 192u);
        const float d = q6k_f16_to_f32(
            (unsigned short)(bp[208] | ((unsigned short)bp[209] << 8)));

        // dot -> scale by the int8 sub-block scale -> scale by the Q8_1 block
        // scale, exactly the order vec_dot_q6_K_q8_1_impl_mmvq uses.
        const int dot_lo = q6k_dp4a((int)qlo, u_lo, 0);
        const int dot_hi = q6k_dp4a((int)qhi, u_hi, 0);

        float acc = 0.0f;
        acc = q6k_fma_rn(d8_lo,
                         q6k_mul_rn(q6k_i2f_rn(dot_lo), q6k_i2f_rn((int)sc[sc_i])),
                         acc);
        acc = q6k_fma_rn(d8_hi,
                         q6k_mul_rn(q6k_i2f_rn(dot_hi), q6k_i2f_rn((int)sc[sc_i + 4u])),
                         acc);
        sumf = q6k_fma_rn(d, acc, sumf);
    }

    sumf = q6k_warp_allreduce(sumf);

    __shared__ float shmem[NWARPS - 1];
    if (warp_id > 0 && lane == 0) shmem[warp_id - 1] = sumf;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = sumf;
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++) total = q6k_add_rn(total, shmem[w]);
        out[row] = total;
    }
}
