// ==========================================================================
// Q6_K kernels: decode matvec against pre-quantized Q8_1 input, the F16
// dequant tile the prefill HGEMM consumes, and the embedding row-gather.
//
// The weight is the GGML Q6_K superblock stream exactly as the artifact
// stores it (210 bytes per 256 elements, no repack):
//   bytes [0..128)   ql   low 4 bits of the 6-bit quants
//   bytes [128..192) qh   high 2 bits, four values per byte
//   bytes [192..208) sc   16 x int8 sub-block scales (one per 16 elements)
//   bytes [208..210) d    f16 super-block scale
// Element idx = 128 n + 32 g + j (half n, group g = 0..4, j = 0..32):
//   q  = (ql[64 n + 32 (g & 1) + j] nibble (g >> 1)) | (((qh[32 n + j] >> 2 g) & 3) << 4)
//   y  = (d * sc[8 n + 2 g + j / 16]) * (q - 32)
// which is the order the host reference (`dequant_kquant_to_f32`) applies;
// the dequant and gather kernels use the `_rn` intrinsics so no FMA
// contraction can change a bit of it.
//
// Matvec: a 32-element group (n, g) is one Q8_1 block and carries two
// sub-block scales (j < 16, j >= 16). With the activation `x = x_scale * xq`
// and q in 0..64 the group contribution is
//   x_scale * d * (sc0 * (dot0 - 32 sum0) + sc1 * (dot1 - 32 sum1))
// with the dots and sums from dp4a over each 16-element half.
//
// Structure: one warp per row, Q6K_NR = 4 rows per Q6K_THREADS = 128-thread
// CTA. Within a superblock the 32 lanes read the 32 consecutive words of the
// 128-byte ql field — four 32-byte sectors, five when the 210-byte stride misaligns them — plus the qh
// word and the two scale bytes their elements need. Lane word `lane` holds the
// low nibbles of group (n, g_lo) and the high nibbles of group (n, g_lo + 2)
// for elements j0 .. j0 + 3, so a lane does two weight dp4a per superblock, and
// two more for the activation sums the `- 32` term needs — the lane's own 4-element sums,
// folded by the final warp reduction. Q6_K is the one scheme where the
// thread-per-group body of the Q4_K/Q5_K kernels loses: its 210-byte
// superblock is only 2-byte aligned, so that body's per-thread 16-bit loads
// leave the kernel issue-bound; the warp-per-row body streams the weight
// faster than the Q4_0 kernel. Alignment: alternate superblocks start 2 mod 4,
// so the word loads go through `q6k_load_u32` (a warp-uniform branch: one
// 32-bit load when the address is 4-byte aligned, two 16-bit loads otherwise).
//
// Numerics: the matvec's f32 accumulation order follows this body's geometry, so its
// output is not bit-identical to another implementation's; the gate is the f64 host reference
// tolerance (max_abs / rel_l2, `tests/cuda_kquant_test.rs`), not exactness. The F16 tile
// and the gathers are exact (`_rn` intrinsics).
//
// Requires compute capability >= 6.1 for dp4a. NVRTC-compatible: no system
// includes, extern "C" linkage.
// ==========================================================================

#define Q6K_THREADS 128
#define Q6K_NR      4   // one warp per row: Q6K_THREADS / 32
#define Q6K_BLOCK_BYTES 210
static_assert(Q6K_THREADS % 32 == 0 && Q6K_THREADS <= 1024, "Q6K_THREADS is a whole number of warps");
static_assert(Q6K_THREADS == 32 * Q6K_NR, "one warp per row: Q6K_THREADS must be 32 * Q6K_NR");
#define Q8_1_BYTES  36

__device__ __forceinline__ float q6k_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short q6k_f32_to_f16(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

__device__ __forceinline__ int q6k_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float q6k_warp_reduce(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Element `within` (0..256) of one superblock, in the host reference's
// arithmetic order: (d * sc) * (q - 32), every product rounded.
__device__ __forceinline__ float q6k_dequant_one(const unsigned char* blk, unsigned int within) {
    const unsigned int n = within >> 7;
    const unsigned int g = (within >> 5) & 3u;
    const unsigned int j = within & 31u;
    const unsigned char ql = blk[64u * n + 32u * (g & 1u) + j];
    const unsigned char qh = blk[128u + 32u * n + j];
    const unsigned int lo = (g >> 1) ? ((ql >> 4) & 0x0Fu) : (ql & 0x0Fu);
    const unsigned int hi = ((qh >> (2u * g)) & 3u) << 4;
    const int q = (int)(lo | hi) - 32;
    const float sc = (float)(signed char)blk[192u + 8u * n + 2u * g + (j >> 4)];
    const float d = q6k_f16_to_f32((unsigned short)blk[208] | ((unsigned short)blk[209] << 8));
    return __fmul_rn(__fmul_rn(d, sc), (float)q);
}

// 32-bit word at a byte offset that is only known to be even (a Q6_K superblock is
// 210 bytes, so alternate superblocks start 2 mod 4). Warp-uniform branch.
__device__ __forceinline__ unsigned int q6k_load_u32(const unsigned char* p) {
    if (((unsigned long long)p & 3ULL) == 0ULL) {
        return *(const unsigned int*)p;
    }
    const unsigned int lo = *(const unsigned short*)p;
    const unsigned int hi = *(const unsigned short*)(p + 2);
    return lo | (hi << 16);
}

// Shared body of the two matvec kernels; `residual == 0` means none.
__device__ __forceinline__ void q6k_matvec_body(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row = blockIdx.x * Q6K_NR + warp;
    if (row >= out_dim) return;                       // uniform across the warp

    const unsigned int nsb = in_dim >> 8;
    const unsigned char* row_base = weight + (unsigned long long)row * nsb * (unsigned long long)Q6K_BLOCK_BYTES;
    // lane -> word `lane` of the 128-byte ql field = half n, ql byte 32 (g & 1) + j0 .. +3:
    // low nibbles are group g_lo = (g & 1), high nibbles group g_hi = g_lo + 2, elements
    // j0 .. j0 + 3 of each; the qh word covers the same four elements for both groups
    const unsigned int n = lane >> 4;
    const unsigned int g_lo = (lane >> 3) & 1u;
    const unsigned int g_hi = g_lo + 2u;
    const unsigned int j0 = 4u * (lane & 7u);
    const unsigned int k = lane & 7u;                 // word within the 32-element group

    float acc = 0.0f;
    #pragma unroll 4
    for (unsigned int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = row_base + (unsigned long long)sb * (unsigned long long)Q6K_BLOCK_BYTES;
        const unsigned int ql = q6k_load_u32(blk + 4 * lane);
        const unsigned int qh = q6k_load_u32(blk + 128 + 32 * n + j0);
        const float sc_lo = (float)(signed char)blk[192 + 8 * n + 2 * g_lo + (j0 >> 4)];
        const float sc_hi = (float)(signed char)blk[192 + 8 * n + 2 * g_hi + (j0 >> 4)];
        const float d = q6k_f16_to_f32(*(const unsigned short*)(blk + 208));

        const char* xb_lo = input_q8_1 + (unsigned long long)(8u * sb + 4u * n + g_lo) * (unsigned long long)Q8_1_BYTES;
        const char* xb_hi = input_q8_1 + (unsigned long long)(8u * sb + 4u * n + g_hi) * (unsigned long long)Q8_1_BYTES;
        const float xs_lo = q6k_f16_to_f32(*(const unsigned short*)xb_lo);
        const float xs_hi = q6k_f16_to_f32(*(const unsigned short*)xb_hi);
        const int x_lo = ((const int*)(xb_lo + 4))[k];
        const int x_hi = ((const int*)(xb_hi + 4))[k];

        const unsigned int q_lo = (ql & 0x0F0F0F0Fu) | (((qh >> (2u * g_lo)) & 0x03030303u) << 4);
        const unsigned int q_hi = ((ql >> 4) & 0x0F0F0F0Fu) | (((qh >> (2u * g_hi)) & 0x03030303u) << 4);
        const int dot_lo = q6k_dp4a((int)q_lo, x_lo, 0);
        const int dot_hi = q6k_dp4a((int)q_hi, x_hi, 0);
        const int xsum_lo = q6k_dp4a(0x01010101, x_lo, 0);
        const int xsum_hi = q6k_dp4a(0x01010101, x_hi, 0);

        acc += xs_lo * (d * sc_lo) * (float)(dot_lo - 32 * xsum_lo)
             + xs_hi * (d * sc_hi) * (float)(dot_hi - 32 * xsum_hi);
    }

    acc = q6k_warp_reduce(acc);
    if (lane == 0) {
        out[row] = residual ? acc + residual[row] : acc;
    }
}

// Grid: (ceil(out_dim / Q6K_NR), 1, 1)   Block: (Q6K_THREADS, 1, 1) — one warp per row; the launcher reads both defines
extern "C" __global__ __launch_bounds__(Q6K_THREADS, 1) void matvec_q6_k_q8_1(
    const unsigned char* __restrict__ weight,     // [out_dim * (in_dim/256) * 210]
    const char* __restrict__ input_q8_1,          // [(in_dim/32) * 36]
    float* __restrict__ out,                      // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    q6k_matvec_body(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(Q6K_THREADS, 1) void matvec_q6_k_q8_1_residual(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,           // [out_dim]
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    q6k_matvec_body(weight, input_q8_1, residual, out, out_dim, in_dim);
}

// Whole-tensor dequant to F16 for the prefill HGEMM. One thread per element.
// Grid: (ceil(num_elements / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void dequant_q6_k_to_f16(
    const unsigned char* __restrict__ weight,
    unsigned short* __restrict__ out_f16,
    unsigned int num_elements)
{
    const unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;
    const unsigned char* blk = weight + (unsigned long long)(e >> 8) * (unsigned long long)Q6K_BLOCK_BYTES;
    out_f16[e] = q6k_f32_to_f16(q6k_dequant_one(blk, e & 255u));
}

// Embedding row-gather, one token: output[i] = table[token_id * hidden_dim + i].
// Grid: (ceil(hidden_dim / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void embed_token_q6_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;
    const unsigned long long e = (unsigned long long)token_id * hidden_dim + i;
    const unsigned char* blk = table + (e >> 8) * (unsigned long long)Q6K_BLOCK_BYTES;
    output[i] = q6k_dequant_one(blk, (unsigned int)(e & 255u));
}

// Embedding row-gather, a batch of tokens: output[t * hidden_dim + i].
// Grid: (ceil(batch * hidden_dim / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void embed_batch_q6_k(
    const unsigned char* __restrict__ table,
    const unsigned int* __restrict__ token_ids,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int hidden_dim)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * hidden_dim) return;
    const unsigned int tok = idx / hidden_dim;
    const unsigned int i = idx - tok * hidden_dim;
    const unsigned long long e = (unsigned long long)token_ids[tok] * hidden_dim + i;
    const unsigned char* blk = table + (e >> 8) * (unsigned long long)Q6K_BLOCK_BYTES;
    output[idx] = q6k_dequant_one(blk, (unsigned int)(e & 255u));
}
