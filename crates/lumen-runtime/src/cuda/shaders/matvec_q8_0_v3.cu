// Q8_0 matrix-vector multiply (GEMV) v3: vectorized loads.
//
// Two optimizations over v1:
//
//   1. Quant data: loads 8 int8 quants via two 32-bit int loads (2 x 4 bytes)
//      instead of 8 individual byte loads. The int values are split into
//      individual bytes via bit shifts. Reduces quant load instructions 4x.
//
//   2. x-vector: loads 8 floats via two float4 loads (2 x 16 bytes) instead
//      of 8 individual float loads. Reduces x load instructions 4x.
//
// Architecture: identical to v1 -- NR=2 multi-row deferred-reduction,
// 128 threads (4 warps). Same grid/block launch config as v1.
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//   Dequant: float_val = scale * (float)(int8_t)quant[j]
//
// Operation: out[i] = sum_j(dequant(weight_q8[i, j]) * x[j])
// Weight matrix: [out_dim, in_dim] stored as Q8_0 blocks, row-major
// Input vector:  [in_dim] f32
// Output vector: [out_dim] f32
//
// Thread mapping (identical to v1):
//   4 threads collectively process one 32-element Q8_0 block (4 x 8 = 32).
//   Each thread handles NQ=8 elements from its sub-chunk (il selects which
//   quarter of the 32 quants). Stride = NSG * NQ = 32 blocks per outer
//   iteration.
//
// Alignment notes:
//   - x-vector float4 loads: each thread accesses x at offset
//     (ib0 * 32 + il * 8), always a multiple of 8 floats = 32 bytes,
//     satisfying float4's 16-byte alignment requirement.
//   - Quant int loads: Q8_0 blocks are 34 bytes, so quant pointers may not
//     be 4-byte aligned. CUDA handles misaligned global memory loads
//     transparently (may split into two cache transactions). Within a single
//     cache line (128 bytes), the penalty is negligible.
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR       2     // rows per thread block
#define NW       32    // warp size
#define NSG      4     // warps per block
#define NQ       8     // elements per thread per iteration (sub-chunk of Q8_0 block)
#define THREADS_PER_BLOCK (NSG * NW)  // 128
#define Q8_0_BLOCK_SIZE   32
#define Q8_0_BYTES        34   // 2 bytes f16 scale + 32 bytes int8 data

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Q8_0 matrix-vector multiply v3: vectorized loads.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
//
// in_dim must be a multiple of Q8_0_BLOCK_SIZE (32).
extern "C" __global__ void matvec_q8_0_v3(
    const char* __restrict__ weight_q8,  // [out_dim * num_blocks_per_row * 34] raw Q8_0 bytes
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;  // first output row for this block

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;
    unsigned int ix      = lane / (NW / NQ);   // lane / 4 -> 0..7
    unsigned int il      = lane % (NW / NQ);   // lane % 4 -> 0..3

    unsigned int nb = in_dim >> 5;  // number of Q8_0 blocks per row
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;  // starting block for this thread

    // Per-row accumulators
    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Register-cached x-values (loaded once, reused for both rows).
    // Use float2 vector loads where possible (8 floats = 2 x float4).
    float yl[NQ];

    // Pointer into x for this thread's starting position
    const float* yb = x + (unsigned long long)ib0 * Q8_0_BLOCK_SIZE + il * NQ;

    // Main loop: 4 warps cooperatively process all blocks with stride NSG*NQ=32
    for (unsigned int ib = ib0; ib < nb; ib += NSG * NQ) {
        // Load NQ=8 x-values into registers using float4 vector loads.
        // Each thread loads 8 floats = 32 bytes = 2 x float4 (16 bytes each).
        {
            const float4* yb4 = (const float4*)yb;
            float4 v0 = yb4[0];
            float4 v1 = yb4[1];
            yl[0] = v0.x;
            yl[1] = v0.y;
            yl[2] = v0.z;
            yl[3] = v0.w;
            yl[4] = v1.x;
            yl[5] = v1.y;
            yl[6] = v1.z;
            yl[7] = v1.w;
        }

        // Process both rows with the same cached x-values
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            // Read f16 scale
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

            // Vectorized load: read 8 quant bytes via a single 64-bit load.
            // Each thread processes il-th quarter of the 32-byte quant block:
            //   il=0: bytes[2..9], il=1: bytes[10..17],
            //   il=2: bytes[18..25], il=3: bytes[26..33]
            // We load 8 bytes as two ints (2 x 4 bytes = 8 bytes).
            const int* qs_int = (const int*)(bp + 2 + il * NQ);
            int q_lo = qs_int[0];  // bytes [0..3] of this sub-chunk
            int q_hi = qs_int[1];  // bytes [4..7] of this sub-chunk

            // Extract individual int8 values and compute dot product.
            // Using bit shifts to extract each byte avoids char-by-char loads.
            float sumq = 0.0f;

            // First 4 quants from q_lo
            sumq += (float)(signed char)(q_lo       ) * yl[0];
            sumq += (float)(signed char)(q_lo >>  8 ) * yl[1];
            sumq += (float)(signed char)(q_lo >> 16 ) * yl[2];
            sumq += (float)(signed char)(q_lo >> 24 ) * yl[3];

            // Second 4 quants from q_hi
            sumq += (float)(signed char)(q_hi       ) * yl[4];
            sumq += (float)(signed char)(q_hi >>  8 ) * yl[5];
            sumq += (float)(signed char)(q_hi >> 16 ) * yl[6];
            sumq += (float)(signed char)(q_hi >> 24 ) * yl[7];

            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * Q8_0_BLOCK_SIZE;
    }

    // Cross-warp reduction via shared memory.
    // Layout: NR rows x NW slots.
    __shared__ float shmem[NR * NW];

    // Initialize shmem (only warp 0 needs to zero it)
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + lane] = 0.0f;
        }
    }

    // Intra-warp reduction
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    // Lane 0 of each warp writes its partial sum
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    // Warp 0 does the final reduction across warps
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < NSG) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val;
                }
            }
        }
    }
}

// Q8_0 matrix-vector multiply v3 with fused residual addition:
// out = dequant(weight_q8) * x + residual
//
// Same vectorized load pattern as matvec_q8_0_v3,
// with fused residual addition at the final write.
//
// Grid:  (ceil(out_dim / NR), 1, 1)  -- one block per NR rows
// Block: (128, 1, 1)                 -- 4 warps x 32 threads
extern "C" __global__ void matvec_q8_0_v3_residual(
    const char* __restrict__ weight_q8,    // [out_dim * num_blocks_per_row * 34] raw Q8_0 bytes
    const float* __restrict__ x,           // [in_dim]
    const float* __restrict__ residual,    // [out_dim], added to output
    float* __restrict__ out,               // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;
    unsigned int ix      = lane / (NW / NQ);
    unsigned int il      = lane % (NW / NQ);

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * Q8_0_BYTES;

    unsigned int ib0 = warp_id * NQ + ix;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    float yl[NQ];
    const float* yb = x + (unsigned long long)ib0 * Q8_0_BLOCK_SIZE + il * NQ;

    for (unsigned int ib = ib0; ib < nb; ib += NSG * NQ) {
        // Vectorized x load (2 x float4 = 8 floats)
        {
            const float4* yb4 = (const float4*)yb;
            float4 v0 = yb4[0];
            float4 v1 = yb4[1];
            yl[0] = v0.x;
            yl[1] = v0.y;
            yl[2] = v0.z;
            yl[3] = v0.w;
            yl[4] = v1.x;
            yl[5] = v1.y;
            yl[6] = v1.z;
            yl[7] = v1.w;
        }

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* bp = weight_q8
                + (unsigned long long)(r0 + row) * row_bytes
                + (unsigned long long)ib * Q8_0_BYTES;

            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            float scale = f16_bits_to_f32(scale_bits);

            // Vectorized 64-bit quant load (2 x int = 8 bytes)
            const int* qs_int = (const int*)(bp + 2 + il * NQ);
            int q_lo = qs_int[0];
            int q_hi = qs_int[1];

            float sumq = 0.0f;
            sumq += (float)(signed char)(q_lo       ) * yl[0];
            sumq += (float)(signed char)(q_lo >>  8 ) * yl[1];
            sumq += (float)(signed char)(q_lo >> 16 ) * yl[2];
            sumq += (float)(signed char)(q_lo >> 24 ) * yl[3];
            sumq += (float)(signed char)(q_hi       ) * yl[4];
            sumq += (float)(signed char)(q_hi >>  8 ) * yl[5];
            sumq += (float)(signed char)(q_hi >> 16 ) * yl[6];
            sumq += (float)(signed char)(q_hi >> 24 ) * yl[7];

            sumf[row] += sumq * scale;
        }

        yb += NSG * NQ * Q8_0_BLOCK_SIZE;
    }

    __shared__ float shmem[NR * NW];

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + lane] = 0.0f;
        }
    }

    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[r * NW + warp_id] = sumf[r];
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            if (r0 + r < out_dim) {
                float val = (lane < NSG) ? shmem[r * NW + lane] : 0.0f;
                val = warp_reduce_sum(val);
                if (lane == 0) {
                    out[r0 + r] = val + residual[r0 + r];
                }
            }
        }
    }
}
