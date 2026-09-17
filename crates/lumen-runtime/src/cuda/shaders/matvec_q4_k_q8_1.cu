// ==========================================================================
// Q4_K kernels: decode matvec against pre-quantized Q8_1 input, the F16
// dequant tile the prefill HGEMM consumes, and the embedding row-gather.
//
// The weight is the GGML Q4_K superblock stream exactly as the artifact
// stores it (144 bytes per 256 elements, no repack):
//   bytes [0..2)    f16 d        super-block scale of the scales
//   bytes [2..4)    f16 dmin     super-block scale of the mins
//   bytes [4..16)   12 bytes     8 x 6-bit scales and 8 x 6-bit mins
//   bytes [16..144) 128 bytes    nibbles; sub-block pair p (p = 0..4) owns
//                                bytes [16 + 32p, 16 + 32p + 32): the low
//                                nibbles are sub-block 2p, the high nibbles
//                                sub-block 2p + 1
// Sub-block j (32 elements, j = 0..8) dequantizes as
//   y = (d * sc_j) * q - (dmin * m_j),   q in 0..16
// which is the order the host reference (`dequant_kquant_to_f32`) applies;
// the dequant and gather kernels use the `_rn` intrinsics so no FMA
// contraction can change a bit of it.
//
// Matvec: one Q4_K sub-block is one Q8_1 block (32 elements), so with the
// activation quantized as `x = x_scale * xq` the sub-block contribution is
//   x_scale * ((d * sc_j) * dot(q, xq) - (dmin * m_j) * sum(xq))
// with both integer sums from dp4a; sum(xq) is computed once per block and
// shared across the NR rows (the Q8_1 header's f16 sum field is not used,
// so the min term carries no extra rounding).
//
// Structure: Q4K_NR = 2 rows per Q4K_THREADS = 64-thread CTA (two warps);
// each thread owns the sub-blocks `threadIdx.x, +64, ...` of the row group,
// two in flight (`#pragma unroll 2`), warp butterfly + shared-memory
// reduction. The small CTA: at in_dim 5120 a row has 160 sub-blocks, so a
// 256-thread CTA idles 96 threads and the 4-row group re-reads the activation
// for a quarter of the work; 64 threads keep every lane busy at every in_dim
// the 27B has (5120, 6144, 17408). Alignment: the superblock is 144 bytes, so
// with a 256-byte cudaMalloc base every superblock, its 16-byte header and
// every 32-byte nibble pair are 16-byte aligned.
//
// Numerics: the matvec's f32 accumulation order follows this body's geometry, so its
// output is not bit-identical to another implementation's; the gate is the f64 host reference
// tolerance (max_abs / rel_l2, `tests/cuda_kquant_test.rs`), not exactness. The F16 tile
// and the gathers are exact (`_rn` intrinsics).
//
// Requires compute capability >= 6.1 for dp4a. NVRTC-compatible: no system
// includes, extern "C" linkage.
// ==========================================================================

#define Q4K_NR      2
#define Q4K_THREADS 64
#define Q4K_NWARPS  (Q4K_THREADS / 32)
#define Q4K_BLOCK_BYTES 144
static_assert(Q4K_THREADS % 32 == 0 && Q4K_THREADS <= 1024, "Q4K_THREADS is a whole number of warps");
#define Q8_1_BYTES  36

__device__ __forceinline__ float q4k_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short q4k_f32_to_f16(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

__device__ __forceinline__ int q4k_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float q4k_warp_reduce(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// GGML get_scale_min_k4 on the 12 packed scale bytes (byte loads).
__device__ __forceinline__ void q4k_scale_min(
    unsigned int j, const unsigned char* s, unsigned int* sc, unsigned int* mn)
{
    if (j < 4u) {
        *sc = s[j] & 63u;
        *mn = s[j + 4] & 63u;
    } else {
        *sc = (s[j + 4] & 0x0Fu) | ((s[j - 4] >> 6) << 4);
        *mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// The same on a header already in registers: byte i (0..12) of the 12
// scale bytes, which sit in words y, z, w of the 16-byte superblock header.
__device__ __forceinline__ unsigned int q4k_sbyte(const uint4 hdr, unsigned int i) {
    const unsigned int word = (i < 4u) ? hdr.y : ((i < 8u) ? hdr.z : hdr.w);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

__device__ __forceinline__ void q4k_scale_min_hdr(
    unsigned int j, const uint4 hdr, unsigned int* sc, unsigned int* mn)
{
    if (j < 4u) {
        *sc = q4k_sbyte(hdr, j) & 63u;
        *mn = q4k_sbyte(hdr, j + 4) & 63u;
    } else {
        *sc = (q4k_sbyte(hdr, j + 4) & 0x0Fu) | ((q4k_sbyte(hdr, j - 4) >> 6) << 4);
        *mn = (q4k_sbyte(hdr, j + 4) >> 4) | ((q4k_sbyte(hdr, j) >> 6) << 4);
    }
}

// Element `within` (0..256) of one superblock, in the host reference's
// arithmetic order: (d * sc) * q - (dmin * m), every product rounded.
__device__ __forceinline__ float q4k_dequant_one(const unsigned char* blk, unsigned int within) {
    const unsigned int j = within >> 5;
    const unsigned int l = within & 31u;
    const float d = q4k_f16_to_f32((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    const float dmin = q4k_f16_to_f32((unsigned short)blk[2] | ((unsigned short)blk[3] << 8));
    unsigned int sc, mn;
    q4k_scale_min(j, blk + 4, &sc, &mn);
    const unsigned char byte = blk[16 + (j >> 1) * 32 + l];
    const unsigned int q = (j & 1u) ? ((byte >> 4) & 0x0Fu) : (byte & 0x0Fu);
    const float d1 = __fmul_rn(d, (float)sc);
    const float m1 = __fmul_rn(dmin, (float)mn);
    return __fsub_rn(__fmul_rn(d1, (float)q), m1);
}

// Shared body of the two matvec kernels; `residual == 0` means none.
__device__ __forceinline__ void q4k_matvec_body(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int r0 = blockIdx.x * Q4K_NR;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    const unsigned int nsub = in_dim >> 5;   // 32-element sub-blocks per row
    const unsigned int nsb = in_dim >> 8;    // superblocks per row
    const unsigned long long row_bytes = (unsigned long long)nsb * (unsigned long long)Q4K_BLOCK_BYTES;

    float sumf[Q4K_NR];
    #pragma unroll
    for (int r = 0; r < Q4K_NR; r++) sumf[r] = 0.0f;

    #pragma unroll 2
    for (unsigned int sub = threadIdx.x; sub < nsub; sub += Q4K_THREADS) {
        const unsigned int sbk = sub >> 3;
        const unsigned int j = sub & 7u;
        const unsigned int shift = (j & 1u) * 4u;

        // Activation block, shared by the NR rows.
        const char* xb = input_q8_1 + (unsigned long long)sub * (unsigned long long)Q8_1_BYTES;
        const float x_scale = q4k_f16_to_f32(*(const unsigned short*)xb);
        const int* xq = (const int*)(xb + 4);
        int xv[8];
        int xsum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            xv[k] = xq[k];
            xsum = q4k_dp4a(0x01010101, xv[k], xsum);
        }

        #pragma unroll
        for (int row = 0; row < Q4K_NR; row++) {
            if (r0 + row >= out_dim) break;
            const unsigned char* blk = weight
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)sbk * (unsigned long long)Q4K_BLOCK_BYTES;

            const uint4 hdr = *(const uint4*)blk;           // d | dmin<<16, 12 scale bytes
            const float d = q4k_f16_to_f32((unsigned short)(hdr.x & 0xffffu));
            const float dmin = q4k_f16_to_f32((unsigned short)(hdr.x >> 16));
            unsigned int sc, mn;
            q4k_scale_min_hdr(j, hdr, &sc, &mn);

            const uint4* qs = (const uint4*)(blk + 16 + (j >> 1) * 32);
            const uint4 q0 = qs[0];
            const uint4 q1 = qs[1];
            const unsigned int w[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int nib = (int)((w[k] >> shift) & 0x0F0F0F0Fu);
                acc = q4k_dp4a(nib, xv[k], acc);
            }
            sumf[row] += x_scale * ((d * (float)sc) * (float)acc - (dmin * (float)mn) * (float)xsum);
        }
    }

    #pragma unroll
    for (int r = 0; r < Q4K_NR; r++) sumf[r] = q4k_warp_reduce(sumf[r]);

    __shared__ float shmem[(Q4K_NWARPS - 1) * Q4K_NR];
    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < Q4K_NR; r++) shmem[(warp_id - 1) * Q4K_NR + r] = sumf[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < Q4K_NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < Q4K_NWARPS - 1; w++) total += shmem[w * Q4K_NR + r];
            if (r0 + r < out_dim) {
                out[r0 + r] = residual ? total + residual[r0 + r] : total;
            }
        }
    }
}

// Grid: (ceil(out_dim / Q4K_NR), 1, 1)   Block: (Q4K_THREADS, 1, 1) — the launcher reads both defines
extern "C" __global__ __launch_bounds__(Q4K_THREADS, 1) void matvec_q4_k_q8_1(
    const unsigned char* __restrict__ weight,     // [out_dim * (in_dim/256) * 144]
    const char* __restrict__ input_q8_1,          // [(in_dim/32) * 36]
    float* __restrict__ out,                      // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    q4k_matvec_body(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(Q4K_THREADS, 1) void matvec_q4_k_q8_1_residual(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,           // [out_dim]
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    q4k_matvec_body(weight, input_q8_1, residual, out, out_dim, in_dim);
}

// Whole-tensor dequant to F16 for the prefill HGEMM. One thread per element.
// Grid: (ceil(num_elements / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void dequant_q4_k_to_f16(
    const unsigned char* __restrict__ weight,
    unsigned short* __restrict__ out_f16,
    unsigned int num_elements)
{
    const unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;
    const unsigned char* blk = weight + (unsigned long long)(e >> 8) * (unsigned long long)Q4K_BLOCK_BYTES;
    out_f16[e] = q4k_f32_to_f16(q4k_dequant_one(blk, e & 255u));
}

// Embedding row-gather, one token: output[i] = table[token_id * hidden_dim + i].
// Grid: (ceil(hidden_dim / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void embed_token_q4_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;
    const unsigned long long e = (unsigned long long)token_id * hidden_dim + i;
    const unsigned char* blk = table + (e >> 8) * (unsigned long long)Q4K_BLOCK_BYTES;
    output[i] = q4k_dequant_one(blk, (unsigned int)(e & 255u));
}

// Embedding row-gather, a batch of tokens: output[t * hidden_dim + i].
// Grid: (ceil(batch * hidden_dim / 256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void embed_batch_q4_k(
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
    const unsigned char* blk = table + (e >> 8) * (unsigned long long)Q4K_BLOCK_BYTES;
    output[idx] = q4k_dequant_one(blk, (unsigned int)(e & 255u));
}
