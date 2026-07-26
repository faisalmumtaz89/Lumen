// ==========================================================================
// Q4 split-layout matvec, F32 acts, x read DIRECT FROM GLOBAL (no staging).
//
// SINGLE-VARIABLE CONTROL against `matvec_q4_split_f32` (variant 1).
//
// Variant 1 measured 0.909x on 9B-Q4. Two explanations were live:
//   (i)  the shared-memory staging of x, and
//   (ii) the NR=4 / 256-thread decomposition (half the block idles when
//        nb = 128 < blockDim, and `xv[32]` keeps 32 registers live).
//
// This kernel is a byte-for-byte clone of variant 1 with exactly ONE change:
// the cooperative x->shared copy and its __syncthreads() are removed, and the
// activation float4s are read straight from global. NR, block dim, grid,
// `xv[32]`, the dot code, row order and the reduction are all IDENTICAL. So
// the A/B against variant 1 attributes the regression to staging or clears it,
// with nothing else moving.
//
// Why staging is suspect: variant 1 indexes `x_smem + ib*32` with
// `ib = threadIdx.x`, i.e. a lane stride of 32 floats — EXACTLY the 32-bank
// period on SM80. Every lane of a warp therefore addresses the same bank, and
// the eight float4 loads serialize. Reading from global instead puts the
// traffic through L1/L2, where x (16-48 KB, read identically by every block)
// is hot after first touch. That is what llama.cpp's mmvq does: it stages
// nothing.
//
// The warp partials use a small STATIC shared array, so the launch requests
// ZERO dynamic shared memory — which also lifts the 48 KB ceiling that
// ffn_down (in_dim 12288) sat exactly on.
//
// Numerics unchanged: F32 in, F32 accumulate, dequant scale*(nibble-8), same
// accumulation order as variant 1.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: none dynamic; NR*(BLOCK_DIM/WARP_SIZE) floats static.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR              4
#define WARP_SIZE       32
#define BLOCK_DIM       256
#define Q4_BLOCK_ELEMS  32
#define Q4_NIBBLE_BYTES 16

__device__ __forceinline__ float f16_bits_to_f32_gmem(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum_gmem(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Identical to variant 1's q4_block_dot_f32 — kept verbatim so the control
// isolates staging alone.
__device__ __forceinline__ float q4_block_dot_gmem(
    const int* __restrict__ nib4,
    const float* __restrict__ xv)
{
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        const int packed = nib4[w];
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int qlo = (lo >> (k * 8)) & 0xFF;
            const int qhi = (hi >> (k * 8)) & 0xFF;
            acc += (float)(qlo - 8) * xv[w * 4 + k];
            acc += (float)(qhi - 8) * xv[16 + w * 4 + k];
        }
    }
    return acc;
}

extern "C" __global__ void matvec_q4_split_f32_gmem(
    const unsigned char* __restrict__ weight_split,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int nb = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + (unsigned long long)Q4_NIBBLE_BYTES);

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += BLOCK_DIM) {
        // THE ONE CHANGE vs variant 1: source is global, not shared.
        float xv[Q4_BLOCK_ELEMS];
        const float4* x4 = (const float4*)(x + (unsigned long long)ib * Q4_BLOCK_ELEMS);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float4 v = x4[k];
            xv[k * 4 + 0] = v.x;
            xv[k * 4 + 1] = v.y;
            xv[k * 4 + 2] = v.z;
            xv[k * 4 + 3] = v.w;
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;
            const unsigned char* rp =
                weight_split + (unsigned long long)(r0 + row) * row_bytes;
            const unsigned short* row_scales = (const unsigned short*)rp;
            const float scale = f16_bits_to_f32_gmem(row_scales[ib]);
            const int* nib4 = (const int*)(rp + 2ULL * nb
                                           + (unsigned long long)ib * Q4_NIBBLE_BYTES);
            sumf[row] += scale * q4_block_dot_gmem(nib4, xv);
        }
    }

    __shared__ float warp_partials[NR][BLOCK_DIM / WARP_SIZE];
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    #pragma unroll
    for (int row = 0; row < NR; row++) {
        float v = warp_reduce_sum_gmem(sumf[row]);
        if (lane == 0) warp_partials[row][warp_id] = v;
    }
    __syncthreads();

    if (threadIdx.x < NR) {
        const unsigned int row = threadIdx.x;
        if (r0 + row < out_dim) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < BLOCK_DIM / WARP_SIZE; w++) total += warp_partials[row][w];
            out[r0 + row] = total;
        }
    }
}
