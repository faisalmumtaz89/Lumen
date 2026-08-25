// CtInt4G32 GEMV + dequant (compressed-tensors pack-quantized INT4 g32,
// imported HF checkpoints).
//
// Decode-image block (20 bytes / 32 elements, built by
// `repack_ct4_blocks` at preload): d bf16 (2B), zp u8 (1B), pad (1B),
// 16 nibble-pair bytes — byte i = element i (low) | element i+16 (high),
// the same pairing as GGML Q4 blocks. value = d * (q - zp), q in 0..15.
//
// Dot vs a Q8_1 activation block (f16 scale s, int8 quants xq):
//   contrib = d * s * (dp4a(q, xq) - zp * sum(xq))
// with sum(xq) computed exactly in the integer domain (dp4a against
// 0x01010101). The block's f16 sum field is NOT used: the zp term is large
// (zp up to 15) and partially cancels the dot term, so the f16 rounding of
// that field would inject noise on the same scale as the quantization
// itself. Blocks are 4-byte aligned (20 % 4 == 0), nibbles at offset +4.

#define CT4_THREADS 256

__device__ __forceinline__ float ct4_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float ct4_bf16_to_f32(unsigned short bits) {
    unsigned int u = ((unsigned int)bits) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ int ct4_dp4a(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float ct4_block_reduce(float v) {
    __shared__ float shmem[CT4_THREADS / 32];
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
        for (int w = 0; w < CT4_THREADS / 32; w++) total += shmem[w];
    }
    return total;
}

__device__ __forceinline__ void matvec_ct4_body(
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
    for (unsigned int b = threadIdx.x; b < nb; b += CT4_THREADS) {
        const unsigned char* blk = wrow + b * 20u;
        const float d = ct4_bf16_to_f32(*(const unsigned short*)blk);
        const char* xb = input_q8_1 + (unsigned long long)b * 36u;
        const float x_scale = ct4_f16_to_f32(*(const unsigned short*)xb);
        const int* xq = (const int*)(xb + 4);
        int dot = 0;
        int sum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned int w = *(const unsigned int*)(blk + 4 + 4 * k);
            const int lo = (int)(w & 0x0F0F0F0Fu);
            const int hi = (int)((w >> 4) & 0x0F0F0F0Fu);
            dot = ct4_dp4a(lo, xq[k], dot);
            dot = ct4_dp4a(hi, xq[k + 4], dot);
            sum = ct4_dp4a(0x01010101, xq[k], sum);
            sum = ct4_dp4a(0x01010101, xq[k + 4], sum);
        }
        acc += d * x_scale * (float)(dot - (int)blk[2] * sum);
    }

    const float total = ct4_block_reduce(acc);
    if (threadIdx.x == 0) {
        out[row] = (residual != 0) ? total + residual[row] : total;
    }
}

extern "C" __global__ __launch_bounds__(CT4_THREADS, 1)
void matvec_ct4_q8_1(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(CT4_THREADS, 1)
void matvec_ct4_q8_1_residual(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body(weight, input_q8_1, residual, out, out_dim, in_dim);
}

// ── exact-K variants (LUMEN_CUDA_CT4_EXACTK) ────────────────────────────────
// K=5120 / K=6144 rows have only 160 / 192 g32 blocks, so the 256-thread
// kernel idles 37.5% / 25% of its warps. These entries run one thread per
// block (TPB == nb for those shapes). Bit-identity with the 256-thread
// kernel: each thread still owns the same block index (b = threadIdx.x,
// single iteration), and the reduction folds a zero-padded 8-slot array in
// the same order — missing warps contribute +0.0f, which is exact.
template<int TPB>
__device__ __forceinline__ float ct4_block_reduce_pad8(float v) {
    __shared__ float shmem[8];
    if (threadIdx.x < 8) shmem[threadIdx.x] = 0.0f;
    __syncthreads();
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
        for (int w = 0; w < 8; w++) total += shmem[w];
    }
    return total;
}

template<int TPB>
__device__ __forceinline__ void matvec_ct4_body_exactk(
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
    for (unsigned int b = threadIdx.x; b < nb; b += TPB) {
        const unsigned char* blk = wrow + b * 20u;
        const float d = ct4_bf16_to_f32(*(const unsigned short*)blk);
        const char* xb = input_q8_1 + (unsigned long long)b * 36u;
        const float x_scale = ct4_f16_to_f32(*(const unsigned short*)xb);
        const int* xq = (const int*)(xb + 4);
        int dot = 0;
        int sum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned int w = *(const unsigned int*)(blk + 4 + 4 * k);
            const int lo = (int)(w & 0x0F0F0F0Fu);
            const int hi = (int)((w >> 4) & 0x0F0F0F0Fu);
            dot = ct4_dp4a(lo, xq[k], dot);
            dot = ct4_dp4a(hi, xq[k + 4], dot);
            sum = ct4_dp4a(0x01010101, xq[k], sum);
            sum = ct4_dp4a(0x01010101, xq[k + 4], sum);
        }
        acc += d * x_scale * (float)(dot - (int)blk[2] * sum);
    }

    const float total = ct4_block_reduce_pad8<TPB>(acc);
    if (threadIdx.x == 0) {
        out[row] = (residual != 0) ? total + residual[row] : total;
    }
}

extern "C" __global__ __launch_bounds__(160, 1)
void matvec_ct4_q8_1_t160(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body_exactk<160>(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(192, 1)
void matvec_ct4_q8_1_t192(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body_exactk<192>(weight, input_q8_1, 0, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(160, 1)
void matvec_ct4_q8_1_residual_t160(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body_exactk<160>(weight, input_q8_1, residual, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(192, 1)
void matvec_ct4_q8_1_residual_t192(
    const unsigned char* __restrict__ weight,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_ct4_body_exactk<192>(weight, input_q8_1, residual, out, out_dim, in_dim);
}

// Dequantize CtInt4G32 decode-image blocks to F16 (prefill HGEMM path).
// One thread per element; element e lives in block e/32, nibble pairing
// byte (e%32)%16, high nibble when (e%32) >= 16.
extern "C" __global__ void dequant_ct4_to_f16(
    const unsigned char* __restrict__ weight,
    unsigned short* __restrict__ out_f16,
    unsigned int num_elements)
{
    const unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;
    const unsigned int blk_idx = e >> 5;
    const unsigned int within = e & 31u;
    const unsigned char* blk = weight + (unsigned long long)blk_idx * 20u;
    const float d = ct4_bf16_to_f32(*(const unsigned short*)blk);
    const float zp = (float)blk[2];
    const unsigned char byte = blk[4 + (within & 15u)];
    const int q = (within < 16u) ? (byte & 0xF) : (byte >> 4);
    const float v = d * ((float)q - zp);
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(v));
    out_f16[e] = bits;
}
