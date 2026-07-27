// ==========================================================================
// matvec_q6_k_f32: native Q6_K matvec, F32 activations.
//
// WHY
//
// A "Q4_0" GGUF is MIXED: llama-quantize deliberately keeps sensitive tensors
// at higher precision. On 9B-Q4 that is `output.weight` (vocab 248320 x hidden
// 4096 = 1.017 G params) plus `wq` on 4 of the 8 full-attention layers.
//
// Lumen has no K-quant matvec, so those tensors are dequantised at upload:
// originally to F32 (4 B/weight) and, after the requant fix, to Q8_0
// (1.0625 B/weight). llama.cpp reads Q6_K NATIVELY at 210 bytes per 256
// elements = 0.8203 B/weight. So on lm_head we move 1.08 GB per token where
// llama.cpp moves 0.834 GB — 30% more on the single largest tensor in the
// model. That is a handicap we carry, so closing it is apples-to-apples, not
// a shortcut: the alternative (requantising to Q4_0 at 0.5625 B/weight) would
// buy speed by taking LOWER precision than the competitor, which would make
// the benchmark dishonest.
//
// Measured context: lm_head costs 0.567 ms/token = 9.0% of the token, running
// at ~1905 GB/s, i.e. already at HBM peak. There is no kernel inefficiency to
// recover here — only bytes. 0.246 GB fewer at that rate is ~0.13 ms.
//
// BLOCK LAYOUT (256 elements / 210 bytes), matching the reference dequant:
//   [0..128)   ql      low 4 bits, two elements per byte
//   [128..192) qh      high 2 bits, four elements per byte
//   [192..208) scales  16 x int8 sub-block scales
//   [208..210) d       f16 super-block scale
//
// Element order, per half (half = 0,1) with ql+64*half, qh+32*half, sc+8*half:
//   group 0: q = (ql[j]      & 0xF) | ((qh[j]      & 3) << 4), scale sc[j/16]
//   group 1: q = (ql[j] >> 4        ) | (((qh[j]>>2) & 3) << 4), scale sc[2 + j/16]
//   group 2: q = (ql[32+j]   & 0xF) | (((qh[j]>>4) & 3) << 4), scale sc[4 + j/16]
//   group 3: q = (ql[32+j] >> 4    ) | (((qh[j]>>6) & 3) << 4), scale sc[6 + j/16]
// with the value being d * sc * (q - 32).
//
// DECOMPOSITION: one warp per output row, lanes striding super-blocks. Each
// lane owns whole super-blocks so the 210-byte stride never splits a block
// across lanes, and the four groups are consumed in the reference order so the
// F32 accumulation matches the dequant-then-multiply result up to
// reassociation.
//
// Grid:  (ceil(out_dim / WARPS), 1, 1)
// Block: (32, WARPS, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define Q6K_WARPS      4
#define Q6K_BLOCK_ELEM 256
#define Q6K_BLOCK_BYTE 210

__device__ __forceinline__ float f16_bits_to_f32_q6k(unsigned short bits) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(bits));
    return r;
}

__device__ __forceinline__ float warp_reduce_sum_q6k(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

extern "C" __global__ void matvec_q6_k_f32(
    const unsigned char* __restrict__ weight,  // [out_dim * nb * 210]
    const float* __restrict__ x,               // [in_dim]
    float* __restrict__ out,                   // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp = threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x * Q6K_WARPS + warp;
    if (row >= out_dim) return;

    const unsigned int nb = in_dim / Q6K_BLOCK_ELEM;
    const unsigned char* rp =
        weight + (unsigned long long)row * (unsigned long long)nb * Q6K_BLOCK_BYTE;

    float sumf = 0.0f;

    // Each lane takes whole super-blocks: the 210-byte stride is not a multiple
    // of 4, so splitting a block across lanes would misalign every access.
    for (unsigned int ib = lane; ib < nb; ib += 32) {
        const unsigned char* bp = rp + (unsigned long long)ib * Q6K_BLOCK_BYTE;
        const unsigned char* ql = bp;
        const unsigned char* qh = bp + 128;
        const signed char*   sc = (const signed char*)(bp + 192);
        const float d = f16_bits_to_f32_q6k(
            (unsigned short)(bp[208] | ((unsigned short)bp[209] << 8)));

        const float* xb = x + (unsigned long long)ib * Q6K_BLOCK_ELEM;

        float acc = 0.0f;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned char* qlh = ql + 64 * half;
            const unsigned char* qhh = qh + 32 * half;
            const signed char*   sch = sc + 8 * half;
            const float* xh = xb + 128 * half;

            #pragma unroll
            for (int j = 0; j < 32; j++) {
                const int h = qhh[j];
                // group 0
                {
                    const int q = (int)(qlh[j] & 0x0F) | (((h) & 3) << 4);
                    acc += (float)sch[j >> 4] * (float)(q - 32) * xh[j];
                }
                // group 1
                {
                    const int q = (int)(qlh[j] >> 4) | (((h >> 2) & 3) << 4);
                    acc += (float)sch[2 + (j >> 4)] * (float)(q - 32) * xh[32 + j];
                }
                // group 2
                {
                    const int q = (int)(qlh[32 + j] & 0x0F) | (((h >> 4) & 3) << 4);
                    acc += (float)sch[4 + (j >> 4)] * (float)(q - 32) * xh[64 + j];
                }
                // group 3
                {
                    const int q = (int)(qlh[32 + j] >> 4) | (((h >> 6) & 3) << 4);
                    acc += (float)sch[6 + (j >> 4)] * (float)(q - 32) * xh[96 + j];
                }
            }
        }
        sumf += d * acc;
    }

    sumf = warp_reduce_sum_q6k(sumf);
    if (lane == 0) out[row] = sumf;
}
