// Q5_K GEMV against pre-quantized Q8_1 input (GDN ssm_out projections).
//
// Serves ssm_out in its source K-quant form (5.5 bpw) instead of a
// requantized Q8_0 copy (8.5 bpw): ~0.4 GB/token less projection traffic on
// the Q4_0-preset 27B artifact (48 GDN layers).
//
// Consumes a lossless SPLIT of GGML Q5_K superblocks (256 elements, 176
// bytes: d f16 + dmin f16 + scales[12] 6-bit packed + qh[32] high bits +
// qs[128] low nibbles) into four aligned planes:
//   qs plane: 128 B/superblock   qh plane: 32 B/superblock
//   sc plane:  12 B/superblock   dm plane:  4 B/superblock (d, dmin)
// Dequant identity (GGML): sub-block j of 8 (32 elements each):
//   value = d*sc_j*q - dmin*m_j,  q = (qs nibble) | (qh bit j ? 16 : 0)
// Dot vs a Q8_1 block (f16 scale s, f16 sum field S over the same 32 lanes):
//   contrib_j = d*sc_j * s_b * dp4a(q, xq)  -  dmin*m_j * S_b
// The sum field supplies the min term with no extra dp4a. This route's
// feeders (quantize_q8_1_rawsum and the norm-gate Q8 epilogue) write S as
// the raw f32 sum of the block, making the min term exact.

#ifndef Q5_NR
#define Q5_NR 2
#endif
#define Q5_THREADS (Q5_NR * 32)

__device__ __forceinline__ float q5_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int q5_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float q5_warp_reduce(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// GGML get_scale_min_k4.
__device__ __forceinline__ void q5_scale_min(
    unsigned int j, const unsigned char* s, float* sc, float* mn)
{
    if (j < 4u) {
        *sc = (float)(s[j] & 63);
        *mn = (float)(s[j + 4] & 63);
    } else {
        *sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        *mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

extern "C" __global__ __launch_bounds__(Q5_THREADS, 2)
void matvec_q5k_split_q8_1(
    const unsigned char* __restrict__ qs_plane,
    const unsigned char* __restrict__ qh_plane,
    const unsigned char* __restrict__ sc_plane,
    const unsigned short* __restrict__ dm_plane, // [rows][SB*2] (d, dmin)
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * Q5_NR + warp;
    if (row >= out_dim) return;

    const unsigned int nsb = in_dim >> 8;
    const unsigned char* qs_row = qs_plane + (unsigned long long)row * nsb * 128u;
    const unsigned char* qh_row = qh_plane + (unsigned long long)row * nsb * 32u;
    const unsigned char* sc_row = sc_plane + (unsigned long long)row * nsb * 12u;
    const unsigned short* dm_row = dm_plane + (unsigned long long)row * nsb * 2u;

    float acc = 0.0f;

    // 8 sub-blocks of 32 per superblock; lane owns sub-blocks strided by 32.
    const unsigned int total_sub = nsb * 8u;
    for (unsigned int sub = lane; sub < total_sub; sub += 32u) {
        const unsigned int sb = sub >> 3;
        const unsigned int j = sub & 7u;

        const float d = q5_f16_to_f32(dm_row[sb * 2u]);
        const float dmin = q5_f16_to_f32(dm_row[sb * 2u + 1u]);
        float sc, mn;
        q5_scale_min(j, sc_row + sb * 12u, &sc, &mn);

        const unsigned char* qs = qs_row + sb * 128u + (j >> 1) * 32u;
        const unsigned char* qh = qh_row + sb * 32u;
        const int hi_nib = (int)(j & 1u);

        const unsigned int pos = sb * 256u + j * 32u;
        const char* xb = input_q8_1 + (unsigned long long)(pos >> 5) * 36u;
        const float x_scale = q5_f16_to_f32(*(const unsigned short*)xb);
        const float x_sum = q5_f16_to_f32(*(const unsigned short*)(xb + 2));
        const int* xq = (const int*)(xb + 4);

        int dot = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const unsigned int qsw = *(const unsigned int*)(qs + 4 * k);
            const unsigned int qhw = *(const unsigned int*)(qh + 4 * k);
            const unsigned int nib = hi_nib ? ((qsw >> 4) & 0x0F0F0F0Fu)
                                            : (qsw & 0x0F0F0F0Fu);
            const unsigned int hbit = ((qhw >> j) & 0x01010101u) << 4;
            dot = q5_dp4a((int)(nib | hbit), xq[k], dot);
        }
        acc += d * sc * x_scale * (float)dot - dmin * mn * x_sum;
    }

    acc = q5_warp_reduce(acc);
    if (lane == 0) {
        out[row] = acc;
    }
}


extern "C" __global__ __launch_bounds__(Q5_THREADS, 2)
void matvec_q5k_split_q8_1_residual(
    const unsigned char* __restrict__ qs_plane,
    const unsigned char* __restrict__ qh_plane,
    const unsigned char* __restrict__ sc_plane,
    const unsigned short* __restrict__ dm_plane,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row = blockIdx.x * Q5_NR + warp;
    if (row >= out_dim) return;

    const unsigned int nsb = in_dim >> 8;
    const unsigned char* qs_row = qs_plane + (unsigned long long)row * nsb * 128u;
    const unsigned char* qh_row = qh_plane + (unsigned long long)row * nsb * 32u;
    const unsigned char* sc_row = sc_plane + (unsigned long long)row * nsb * 12u;
    const unsigned short* dm_row = dm_plane + (unsigned long long)row * nsb * 2u;

    float acc = 0.0f;
    const unsigned int total_sub = nsb * 8u;
    for (unsigned int sub = lane; sub < total_sub; sub += 32u) {
        const unsigned int sb = sub >> 3;
        const unsigned int j = sub & 7u;
        const float d = q5_f16_to_f32(dm_row[sb * 2u]);
        const float dmin = q5_f16_to_f32(dm_row[sb * 2u + 1u]);
        float sc, mn;
        q5_scale_min(j, sc_row + sb * 12u, &sc, &mn);
        const unsigned char* qs = qs_row + sb * 128u + (j >> 1) * 32u;
        const unsigned char* qh = qh_row + sb * 32u;
        const int hi_nib = (int)(j & 1u);
        const unsigned int pos = sb * 256u + j * 32u;
        const char* xb = input_q8_1 + (unsigned long long)(pos >> 5) * 36u;
        const float x_scale = q5_f16_to_f32(*(const unsigned short*)xb);
        const float x_sum = q5_f16_to_f32(*(const unsigned short*)(xb + 2));
        const int* xq = (const int*)(xb + 4);
        int dot = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const unsigned int qsw = *(const unsigned int*)(qs + 4 * k);
            const unsigned int qhw = *(const unsigned int*)(qh + 4 * k);
            const unsigned int nib = hi_nib ? ((qsw >> 4) & 0x0F0F0F0Fu)
                                            : (qsw & 0x0F0F0F0Fu);
            const unsigned int hbit = ((qhw >> j) & 0x01010101u) << 4;
            dot = q5_dp4a((int)(nib | hbit), xq[k], dot);
        }
        acc += d * sc * x_scale * (float)dot - dmin * mn * x_sum;
    }

    acc = q5_warp_reduce(acc);
    if (lane == 0) {
        out[row] = acc + residual[row];
    }
}
