// ==========================================================================
// Q4 split-layout matvec against FULL F32 activations (NR=4).
//
// WHY THIS KERNEL EXISTS
//
// 9B-Q4 decode runs at 88 tok/s while llama.cpp reaches 153 on the identical
// GGUF. Both stream the same 5.181 GB of weights per token, so the gap is not
// bytes — it is 456 GB/s vs 793 GB/s of achieved bandwidth on the same bytes.
// Activation precision cannot explain it either: the whole activation vector
// is ~2 MB/token, 0.04% of weight traffic, and measured F16/Q8 activation
// zoning moved decode by only 1.01-1.12x.
//
// The cost is in how `matvec_q4_0_smem` (today's default) touches memory:
//
//   * ONE THREAD PER 18-BYTE BLOCK. Consecutive threads read addresses 18
//     bytes apart — never 32B-aligned, so each memory transaction is mostly
//     wasted and no two lanes coalesce.
//   * BYTEWISE NIBBLE UNPACKING. The inner loop does 16 single-byte loads
//     (`qs[i]`) per block. Byte loads cannot coalesce into wide transactions.
//   * 32 ACTIVATION FLOATS IN REGISTERS (`float xv[32]`) plus NR accumulators,
//     which caps occupancy and therefore memory-level parallelism.
//
// This kernel keeps EXACT F32 activation numerics (so it is a correctness-
// neutral swap, unlike the activation-format levers) and fixes the access
// pattern by consuming the SPLIT/SoA weight layout produced by
// `repack_q4_raw_to_split`:
//
//     per row: [f16 scale x nb][16 nibble bytes x nb]
//
// The nibble stream is contiguous and 4-byte aligned (nb is even on every
// shipped dim), so a block's 16 nibble bytes are exactly FOUR aligned 32-bit
// words. Each thread loads them as `int`s and masks both halves with
// `& 0x0F0F0F0F`, which is free in registers — the same trick llama.cpp's
// mmvq uses to reach ~55% of HBM.
//
// Numerics: dequant is `scale * (nibble - 8)`, accumulated in F32 in the same
// order as the smem kernel (block-local sum, then a single warp reduction), so
// results are bit-comparable to the existing F32 path modulo FP reassociation
// within a block, which the correctness gate checks against the baseline.
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: in_dim * sizeof(float) for the x-vector.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR              4       // rows per thread block (matches the split dp4a kernel)
#define WARP_SIZE       32
#define BLOCK_DIM       256     // 8 warps
#define Q4_BLOCK_ELEMS  32
#define Q4_NIBBLE_BYTES 16

__device__ __forceinline__ float f16_bits_to_f32_splitf32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum_splitf32(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Dequantize-and-dot one Q4_0 block against 32 F32 activations.
//
// The 16 nibble bytes arrive as four aligned ints. The GGML de-interleaved
// layout puts elements 0-15 in the LOW nibbles of bytes 0-15 and elements
// 16-31 in the HIGH nibbles, so each int contributes 4 low-half elements at
// [w*4 + k] and 4 high-half elements at [16 + w*4 + k].
__device__ __forceinline__ float q4_block_dot_f32(
    const int* __restrict__ nib4,   // 4 aligned ints = 16 nibble bytes
    const float* __restrict__ xv)   // 32 activations
{
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        const int packed = nib4[w];
        // Both halves extracted with register-only ops — no byte loads.
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

extern "C" __global__ void matvec_q4_split_f32(
    const unsigned char* __restrict__ weight_split,  // [out_dim * 18 * nb]
    const float* __restrict__ x,                     // [in_dim]
    float* __restrict__ out,                         // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    extern __shared__ float x_smem_sf32[];

    const unsigned int r0 = blockIdx.x * NR;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int nb = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + (unsigned long long)Q4_NIBBLE_BYTES);

    for (unsigned int i = threadIdx.x; i < in_dim; i += BLOCK_DIM) {
        x_smem_sf32[i] = x[i];
    }
    __syncthreads();

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += BLOCK_DIM) {
        float xv[Q4_BLOCK_ELEMS];
        const float4* x4 = (const float4*)(x_smem_sf32 + ib * Q4_BLOCK_ELEMS);
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

            // scales stream: nb halfwords, then the nibble stream
            const unsigned short* row_scales = (const unsigned short*)rp;
            const float scale = f16_bits_to_f32_splitf32(row_scales[ib]);

            // 4-byte aligned because 2*nb is a multiple of 4 for even nb
            const int* nib4 = (const int*)(rp + 2ULL * nb
                                           + (unsigned long long)ib * Q4_NIBBLE_BYTES);

            sumf[row] += scale * q4_block_dot_f32(nib4, xv);
        }
    }

    // Warp reduction, then one shared-memory pass across warps.
    __shared__ float warp_partials[NR][BLOCK_DIM / WARP_SIZE];
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    #pragma unroll
    for (int row = 0; row < NR; row++) {
        float v = warp_reduce_sum_splitf32(sumf[row]);
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
