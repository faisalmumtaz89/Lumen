// STATUS: DISABLED — regressive on A100. 75% lane waste (8 of 32 lanes active per
// K-tile iteration). Decode 30% slower than v1. Needs redesign: wider tiles or
// different thread mapping. See tracker.md Wave C8.
//
// matvec_q8_0_v4: High-throughput Q8_0 matrix-vector multiply for CUDA (SM 6.1+)
//
// Strategy: Cooperative x quantization into shared memory + dp4a INT8 dot products.
//
// w_q8:    Q8_0 weight data for [out_dim, in_dim] matrix
//          Each row: ceil(in_dim/32) blocks, each block = 2 bytes f16 scale + 32 bytes int8
// x:       [in_dim] input vector (f32)
// out:     [out_dim] output vector (f32)
// out_dim: total number of output rows
// in_dim:  number of elements per row (must be multiple of 32)
//
// Q8_0 block layout (34 bytes):
//   [f16 scale (2 bytes)] [32 x int8 quantized values (32 bytes)]
//   dequantized value = scale * (float)int8_val
//
// Dispatch: grid = (ceil(out_dim / NR),), block = (BLOCK_DIM,)
//
// Performance target: ~65-70% DRAM bandwidth utilization on A100 (2 TB/s)
// via dp4a INT8 compute + cooperative x quantization + coalesced weight loads.
// ============================================================================

#define NR              4     // Rows per block (1 warp per row)
#define NW              4     // Warps per block
#define BLOCK_DIM       128   // Threads per block (NR * 32)
#define WARP_SIZE       32
#define Q8_BLOCK_BYTES  34    // 2-byte f16 scale + 32 int8 values
#define Q8_BLOCK_ELEMS  32    // Elements per Q8_0 block
#define K_TILE_BLOCKS   8     // Q8_0 blocks per K-tile = 256 elements

// Hardware f16 -> f32 conversion via PTX (avoids software emulation)
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Quantize a single float to int8 given inverse scale, clamp to [-127, 127]
__device__ __forceinline__ int quantize_f32_to_i8(float val, float inv_scale) {
    int q = __float2int_rn(val * inv_scale);
    // Clamp to [-127, 127] -- we avoid -128 for symmetric quantization
    q = max(-127, min(127, q));
    return q;
}

// ============================================================================
// matvec_q8_0_v4: dp4a-accelerated Q8_0 matvec with cooperative x quantization
// ============================================================================
//
// Architecture:
//   - 128 threads = 4 warps, each warp handles 1 output row
//   - K-dimension tiled in chunks of 256 elements (8 Q8_0 blocks)
//   - Per tile: all 128 threads cooperatively quantize x to int8 in shared mem
//   - Per tile: each warp independently dp4a-accumulates against its row's weights
//   - Final warp-shuffle reduction produces 1 result per row
//
// Memory layout in shared memory:
//   x_packed[K_TILE_BLOCKS * 8]:  int32, holding packed int8 quants (4 per word)
//   x_scales[K_TILE_BLOCKS]:      float, per-block x quantization scale
//
extern "C" __global__ void matvec_q8_0_v4(
    const char* __restrict__ weight_q8,  // Q8_0 weight matrix [out_dim, num_blocks * 34]
    const float* __restrict__ x,         // input vector [in_dim]
    float* __restrict__ out,             // output vector [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;      // 0..3
    const unsigned int lane    = threadIdx.x % WARP_SIZE;      // 0..31
    const unsigned int row     = blockIdx.x * NR + warp_id;

    const unsigned int num_blocks = in_dim / Q8_BLOCK_ELEMS;

    // Shared memory for cooperatively quantized x tile
    // 8 blocks * 8 int32 per block = 64 int32 = 256 packed int8 values
    __shared__ int    x_packed[K_TILE_BLOCKS * 8];
    __shared__ float  x_scales[K_TILE_BLOCKS];

    // Per-thread accumulator (float to avoid int32 overflow for very large K)
    float acc = 0.0f;

    // Process K in tiles of K_TILE_BLOCKS Q8_0 blocks (256 elements)
    for (unsigned int tile_start = 0; tile_start < num_blocks; tile_start += K_TILE_BLOCKS) {
        const unsigned int tile_blocks = min((unsigned int)K_TILE_BLOCKS, num_blocks - tile_start);
        const unsigned int tile_elems  = tile_blocks * Q8_BLOCK_ELEMS;
        const unsigned int x_base      = tile_start * Q8_BLOCK_ELEMS;

        // ================================================================
        // Step 1: Cooperative x quantization into shared memory
        //
        // All 128 threads collaborate to quantize 256 floats (8 Q8_0 blocks).
        // Each thread handles 2 floats (128 threads * 2 = 256 elements).
        // We quantize per-Q8_0-block (32 elements) to match weight block scales.
        //
        // Sub-step 1a: Find per-block amax (32 threads per block)
        // Sub-step 1b: Compute per-block scale and quantize
        // ================================================================

        // Each thread is assigned to a specific Q8_0 block within the tile:
        // thread 0-31 -> block 0, thread 32-63 -> block 1, etc.
        // Within each block, the lane (thread % 32) indexes the element.
        const unsigned int my_block = threadIdx.x / WARP_SIZE;  // Which block in tile (0..3)
        const unsigned int my_lane  = threadIdx.x % WARP_SIZE;  // Element within block

        // First pass: blocks 0-3 (handled by warps 0-3)
        if (my_block < tile_blocks) {
            unsigned int elem_idx = x_base + my_block * Q8_BLOCK_ELEMS + my_lane;
            float val = (elem_idx < in_dim) ? x[elem_idx] : 0.0f;
            float abs_val = fabsf(val);

            // Warp-level max reduction to find amax for this block
            float amax = abs_val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            }
            // Now amax is uniform across the warp (all 32 threads have the same value)

            float scale = amax / 127.0f;
            float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

            // Lane 0 writes the scale for this block
            if (my_lane == 0) {
                x_scales[my_block] = scale;
            }

            // Quantize this element
            int q = quantize_f32_to_i8(val, inv_scale);

            // Pack 4 int8 values into one int32 using warp shuffles
            // Layout within int32: [lane%4==0 in bits 0:7, lane%4==1 in bits 8:15, ...]
            // This matches the byte order expected by dp4a
            unsigned int pack_group = my_lane / 4;   // Which int32 word (0..7)
            unsigned int pack_pos   = my_lane % 4;   // Position within word (0..3)

            // Gather 4 values into lane that is pack_pos==0
            int q0 = __shfl_sync(0xffffffff, q, pack_group * 4 + 0);
            int q1 = __shfl_sync(0xffffffff, q, pack_group * 4 + 1);
            int q2 = __shfl_sync(0xffffffff, q, pack_group * 4 + 2);
            int q3 = __shfl_sync(0xffffffff, q, pack_group * 4 + 3);

            int packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

            // Only one thread per group of 4 writes (avoid bank conflicts)
            if (pack_pos == 0) {
                x_packed[my_block * 8 + pack_group] = packed;
            }
        }

        // Second pass: blocks 4-7 (same warps handle the next 4 blocks)
        if (my_block + NW < tile_blocks) {
            unsigned int blk2 = my_block + NW;
            unsigned int elem_idx = x_base + blk2 * Q8_BLOCK_ELEMS + my_lane;
            float val = (elem_idx < in_dim) ? x[elem_idx] : 0.0f;
            float abs_val = fabsf(val);

            float amax = abs_val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            }

            float scale = amax / 127.0f;
            float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

            if (my_lane == 0) {
                x_scales[blk2] = scale;
            }

            int q = quantize_f32_to_i8(val, inv_scale);

            unsigned int pack_group = my_lane / 4;
            unsigned int pack_pos   = my_lane % 4;

            int q0 = __shfl_sync(0xffffffff, q, pack_group * 4 + 0);
            int q1 = __shfl_sync(0xffffffff, q, pack_group * 4 + 1);
            int q2 = __shfl_sync(0xffffffff, q, pack_group * 4 + 2);
            int q3 = __shfl_sync(0xffffffff, q, pack_group * 4 + 3);

            int packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

            if (pack_pos == 0) {
                x_packed[blk2 * 8 + pack_group] = packed;
            }
        }

        __syncthreads();

        // ================================================================
        // Step 2: Each warp processes its row's weight blocks against
        //         the quantized x tile using dp4a
        //
        // Each lane handles one Q8_0 block at a time (32 lanes -> 32 blocks
        // max per iteration, but tile has only 8 blocks, so lane < 8 is active).
        // ================================================================

        if (row < out_dim) {
            // Compute row byte offset (use 64-bit to avoid overflow for large models)
            const unsigned long long row_offset = (unsigned long long)row * (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

            for (unsigned int b = lane; b < tile_blocks; b += WARP_SIZE) {
                unsigned int block_idx = tile_start + b;
                const char* bp = weight_q8 + row_offset + (unsigned long long)block_idx * Q8_BLOCK_BYTES;

                // Read weight scale (f16 stored as 2 LE bytes)
                unsigned short ws_bits = (unsigned short)(unsigned char)bp[0]
                                       | ((unsigned short)(unsigned char)bp[1] << 8);
                float w_scale = f16_bits_to_f32(ws_bits);

                // Read weight quants as 8 x int32 (32 int8 values packed)
                // Q8_0 blocks are 34-byte aligned (offset +2 from block start),
                // so we load byte-by-byte and pack to handle misalignment safely.
                const unsigned char* wq = (const unsigned char*)(bp + 2);

                // dp4a accumulation: 8 int32 words (32 int8 values total)
                int dp_sum = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    // Load 4 weight bytes and pack into int32
                    int w_word = (int)(signed char)wq[i * 4 + 0]
                              | ((int)(signed char)wq[i * 4 + 1] << 8)
                              | ((int)(signed char)wq[i * 4 + 2] << 16)
                              | ((int)(signed char)wq[i * 4 + 3] << 24);

                    dp_sum = __dp4a(w_word, x_packed[b * 8 + i], dp_sum);
                }

                // Scale correction: result = w_scale * x_scale * dp4a_sum
                // dp4a computes sum of (w_int8 * x_int8) products (integer),
                // and the real dot product = w_scale * x_scale * dp4a_sum
                float x_scale = x_scales[b];
                acc += w_scale * x_scale * (float)dp_sum;
            }
        }

        __syncthreads();  // Ensure all warps done before next tile overwrites shmem
    }

    // ================================================================
    // Step 3: Warp-level reduction (lane partial sums -> single result)
    // ================================================================
    if (row < out_dim) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            out[row] = acc;
        }
    }
}

// ============================================================================
// matvec_q8_0_v4_residual: Same as v4 but adds to existing output buffer
// Used for residual connections (out[row] += result)
// ============================================================================
extern "C" __global__ void matvec_q8_0_v4_residual(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int row     = blockIdx.x * NR + warp_id;

    const unsigned int num_blocks = in_dim / Q8_BLOCK_ELEMS;

    __shared__ int    x_packed[K_TILE_BLOCKS * 8];
    __shared__ float  x_scales[K_TILE_BLOCKS];

    float acc = 0.0f;

    for (unsigned int tile_start = 0; tile_start < num_blocks; tile_start += K_TILE_BLOCKS) {
        const unsigned int tile_blocks = min((unsigned int)K_TILE_BLOCKS, num_blocks - tile_start);
        const unsigned int x_base      = tile_start * Q8_BLOCK_ELEMS;

        const unsigned int my_block = threadIdx.x / WARP_SIZE;
        const unsigned int my_lane  = threadIdx.x % WARP_SIZE;

        // First pass: blocks 0-3
        if (my_block < tile_blocks) {
            unsigned int elem_idx = x_base + my_block * Q8_BLOCK_ELEMS + my_lane;
            float val = (elem_idx < in_dim) ? x[elem_idx] : 0.0f;
            float abs_val = fabsf(val);

            float amax = abs_val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            }

            float scale = amax / 127.0f;
            float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

            if (my_lane == 0) {
                x_scales[my_block] = scale;
            }

            int q = quantize_f32_to_i8(val, inv_scale);

            unsigned int pack_group = my_lane / 4;
            unsigned int pack_pos   = my_lane % 4;

            int q0 = __shfl_sync(0xffffffff, q, pack_group * 4 + 0);
            int q1 = __shfl_sync(0xffffffff, q, pack_group * 4 + 1);
            int q2 = __shfl_sync(0xffffffff, q, pack_group * 4 + 2);
            int q3 = __shfl_sync(0xffffffff, q, pack_group * 4 + 3);

            int packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

            if (pack_pos == 0) {
                x_packed[my_block * 8 + pack_group] = packed;
            }
        }

        // Second pass: blocks 4-7
        if (my_block + NW < tile_blocks) {
            unsigned int blk2 = my_block + NW;
            unsigned int elem_idx = x_base + blk2 * Q8_BLOCK_ELEMS + my_lane;
            float val = (elem_idx < in_dim) ? x[elem_idx] : 0.0f;
            float abs_val = fabsf(val);

            float amax = abs_val;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            }

            float scale = amax / 127.0f;
            float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

            if (my_lane == 0) {
                x_scales[blk2] = scale;
            }

            int q = quantize_f32_to_i8(val, inv_scale);

            unsigned int pack_group = my_lane / 4;
            unsigned int pack_pos   = my_lane % 4;

            int q0 = __shfl_sync(0xffffffff, q, pack_group * 4 + 0);
            int q1 = __shfl_sync(0xffffffff, q, pack_group * 4 + 1);
            int q2 = __shfl_sync(0xffffffff, q, pack_group * 4 + 2);
            int q3 = __shfl_sync(0xffffffff, q, pack_group * 4 + 3);

            int packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);

            if (pack_pos == 0) {
                x_packed[blk2 * 8 + pack_group] = packed;
            }
        }

        __syncthreads();

        if (row < out_dim) {
            const unsigned long long row_offset = (unsigned long long)row * (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

            for (unsigned int b = lane; b < tile_blocks; b += WARP_SIZE) {
                unsigned int block_idx = tile_start + b;
                const char* bp = weight_q8 + row_offset + (unsigned long long)block_idx * Q8_BLOCK_BYTES;

                unsigned short ws_bits = (unsigned short)(unsigned char)bp[0]
                                       | ((unsigned short)(unsigned char)bp[1] << 8);
                float w_scale = f16_bits_to_f32(ws_bits);

                const unsigned char* wq = (const unsigned char*)(bp + 2);

                int dp_sum = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int w_word = (int)(signed char)wq[i * 4 + 0]
                              | ((int)(signed char)wq[i * 4 + 1] << 8)
                              | ((int)(signed char)wq[i * 4 + 2] << 16)
                              | ((int)(signed char)wq[i * 4 + 3] << 24);

                    dp_sum = __dp4a(w_word, x_packed[b * 8 + i], dp_sum);
                }

                float x_scale = x_scales[b];
                acc += w_scale * x_scale * (float)dp_sum;
            }
        }

        __syncthreads();
    }

    if (row < out_dim) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            out[row] += acc;  // Residual add
        }
    }
}
