// Q4_1 GEMV against pre-quantized Q8_1 input.
//
// Q4_1 block (20 bytes / 32 elements): d f16, m f16, 16 nibble-pair bytes.
// value = d*q + m (q in 0..15). Dot vs a Q8_1 block (f16 scale s, f16 sum
// S = s*sum(xq)):   contrib = d * s * dp4a(q, xq) + m * S
// — the Q8_1 sum supplies the min term exactly. Blocks are 4-byte aligned
// (20 % 4 == 0), so u32 loads need no repacking. Serves the 8/64 ffn_down
// tensors that Q4_0-preset GGUFs store as Q4_1 (source fidelity: the min
// term is part of the source quantization).

#ifndef Q41_NR
#define Q41_NR 1
#endif
#define Q41_THREADS 256

__device__ __forceinline__ float q41_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int q41_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float q41_block_reduce(float v) {
    __shared__ float shmem[Q41_THREADS / 32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (lane == 0) shmem[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int w = 0; w < Q41_THREADS / 32; w++) total += shmem[w];
    }
    return total;
}

__device__ __forceinline__ void matvec_q4_1_body(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;
    const unsigned int nb = in_dim >> 5;
    const unsigned char* wrow = weight + (unsigned long long)row * nb * 20u;

    float acc = 0.0f;
    for (unsigned int b = threadIdx.x; b < nb; b += Q41_THREADS) {
        const unsigned char* blk = wrow + b * 20u;
        const float d = q41_f16_to_f32(*(const unsigned short*)blk);
        const float m = q41_f16_to_f32(*(const unsigned short*)(blk + 2));
        const char* xb = input_q8_1 + (unsigned long long)b * 36u;
        const float x_scale = q41_f16_to_f32(*(const unsigned short*)xb);
        const float x_sum = q41_f16_to_f32(*(const unsigned short*)(xb + 2));
        const int* xq = (const int*)(xb + 4);
        int dot = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned int w = *(const unsigned int*)(blk + 4 + 4 * k);
            const int lo = (int)(w & 0x0F0F0F0Fu);
            const int hi = (int)((w >> 4) & 0x0F0F0F0Fu);
            dot = q41_dp4a(lo, xq[k], dot);
            dot = q41_dp4a(hi, xq[k + 4], dot);
        }
        acc += d * x_scale * (float)dot + m * x_sum;
    }

    const float total = q41_block_reduce(acc);
    if (threadIdx.x == 0) {
        out[row] = (residual != 0) ? total + residual[row] : total;
    }
}

extern "C" __global__ __launch_bounds__(Q41_THREADS, 1)
void matvec_q4_1_q8_1(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_1_body(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(Q41_THREADS, 1)
void matvec_q4_1_q8_1_residual(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_1_body(weight, input_q8_1, residual, out, out_dim, in_dim);
}
