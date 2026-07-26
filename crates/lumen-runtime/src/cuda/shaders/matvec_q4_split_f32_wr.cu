// ==========================================================================
// Q4 split-layout matvec, WARP-PER-ROW, no shared-memory staging (F32 acts).
//
// WHY A SECOND SoA KERNEL
//
// `matvec_q4_split_f32` (the NR=4 / 256-thread / smem-staged variant) measured
// 80.04 tok/s vs an 88.03 baseline = 0.909x on 9B-Q4. That is a REGRESSION, and
// reading it against the actual 9B shapes says the fault is the work
// decomposition, not the SoA idea:
//
//   * gate/up have in_dim 4096 -> nb = 128 blocks, against BLOCK_DIM = 256
//     threads. HALF THE BLOCK HAS NO WORK. down (nb = 384) is unbalanced
//     2-vs-1 across threads.
//   * `float xv[32]` per thread is 32 registers spent purely on staging
//     activations, which caps occupancy on top of the above.
//   * staging all of x in shared memory costs in_dim*4 bytes = 48 KB on
//     ffn_down, i.e. exactly the SM80 cap, so only ONE BLOCK PER SM can be
//     resident there — the single largest projection in the model runs at
//     minimum memory-level parallelism.
//
// This variant fixes all three:
//
//   * ONE WARP PER OUTPUT ROW. Lanes stride the block axis by 32, so every
//     lane has work whenever nb >= 32 (nb is 128/384 on 9B — 4 and 12 blocks
//     per lane, evenly divided). Utilization is independent of nb vs blockDim.
//   * NO SHARED MEMORY AT ALL. x is read straight from global. It is 16-48 KB
//     and every warp in every block reads the same vector, so it is L2- and
//     L1-hot after the first touch — this is what llama.cpp's mmvq does, and
//     it means occupancy is bounded by registers alone rather than by 48 KB of
//     smem per block.
//   * NO xv[32] ARRAY. Activations are consumed as two `float4` loads per
//     packed int, live only across the four FMA pairs that use them, so the
//     activation register footprint drops from 32 floats to 8.
//
// Numerics are IDENTICAL in kind to the F32 baseline: F32 activations, F32
// accumulate, dequant `scale * (nibble - 8)`. Only the summation ORDER differs
// (lane-strided partial sums + warp reduction), so this is a correctness-
// neutral swap modulo FP reassociation, which the correctness gate checks.
//
// Grid:  (ceil(out_dim / WARPS_PER_BLOCK), 1, 1)
// Block: (WARPS_PER_BLOCK * 32, 1, 1)
// Shared memory: NONE.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define WARP_SIZE        32
#define WARPS_PER_BLOCK  4      // 128 threads; 4 output rows per block
#define Q4_BLOCK_ELEMS   32
#define Q4_NIBBLE_BYTES  16

__device__ __forceinline__ float f16_bits_to_f32_wr(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum_wr(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Dequantize-and-dot one Q4_0 block against 32 F32 activations read from
// global memory as float4s. The GGML de-interleaved layout puts elements 0-15
// in the LOW nibbles of bytes 0-15 and elements 16-31 in the HIGH nibbles, so
// packed int `w` pairs with x[w*4 .. w*4+3] (low) and x[16+w*4 .. 16+w*4+3]
// (high) — both 16-byte aligned, hence both single float4 loads.
__device__ __forceinline__ float q4_block_dot_global(
    const int* __restrict__ nib4,   // 4 aligned ints = 16 nibble bytes
    const float4* __restrict__ x4)  // 8 float4s = 32 activations
{
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; w++) {
        const int packed = nib4[w];
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;
        const float4 xl = x4[w];       // elements w*4 .. w*4+3
        const float4 xh = x4[4 + w];   // elements 16+w*4 .. 16+w*4+3

        // Zero-point applied as an INTEGER subtract before the convert, so
        // each term is one sub + one FMA — cheaper than accumulating q*x and
        // correcting with 8*sum(x) afterwards, which would add a 7-add
        // reduction per packed int.
        acc += (float)(((lo      ) & 0xFF) - 8) * xl.x;
        acc += (float)(((lo >>  8) & 0xFF) - 8) * xl.y;
        acc += (float)(((lo >> 16) & 0xFF) - 8) * xl.z;
        acc += (float)(((lo >> 24) & 0xFF) - 8) * xl.w;
        acc += (float)(((hi      ) & 0xFF) - 8) * xh.x;
        acc += (float)(((hi >>  8) & 0xFF) - 8) * xh.y;
        acc += (float)(((hi >> 16) & 0xFF) - 8) * xh.z;
        acc += (float)(((hi >> 24) & 0xFF) - 8) * xh.w;
    }
    return acc;
}

extern "C" __global__ void matvec_q4_split_f32_wr(
    const unsigned char* __restrict__ weight_split,  // [out_dim * 18 * nb]
    const float* __restrict__ x,                     // [in_dim]
    float* __restrict__ out,                         // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const unsigned int nb = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + (unsigned long long)Q4_NIBBLE_BYTES);
    const unsigned char* rp = weight_split + (unsigned long long)row * row_bytes;

    // Scales stream first (nb halfwords), then the nibble stream. 2*nb is a
    // multiple of 4 for even nb, so the nibble stream is 4-byte aligned and
    // readable as ints.
    const unsigned short* row_scales = (const unsigned short*)rp;
    const unsigned char*  row_nibs   = rp + 2ULL * nb;

    float sumf = 0.0f;
    for (unsigned int ib = lane; ib < nb; ib += WARP_SIZE) {
        const float scale = f16_bits_to_f32_wr(row_scales[ib]);
        const int* nib4 =
            (const int*)(row_nibs + (unsigned long long)ib * Q4_NIBBLE_BYTES);
        const float4* x4 = (const float4*)(x + (unsigned long long)ib * Q4_BLOCK_ELEMS);
        sumf += scale * q4_block_dot_global(nib4, x4);
    }

    sumf = warp_reduce_sum_wr(sumf);
    if (lane == 0) out[row] = sumf;
}
