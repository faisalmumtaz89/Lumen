// Q4_0 quantized matrix-vector multiply (GEMV) kernels for decode.
//
// Operation: out[i] = sum_j(dequant(weight_q4[i, j]) * x[j])
// Weight matrix: [out_dim, in_dim] stored as Q4_0 blocks (row-major)
// Input vector:  [in_dim] f32
// Output vector: [out_dim] f32
//
// Q4_0 block layout (GGML standard, 18 bytes per 32 elements):
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..17]: 16 bytes = 32 x 4-bit unsigned values packed as nibble pairs
//     De-interleaved layout: elements 0-15 from lo nibbles of bytes 0-15,
//     elements 16-31 from hi nibbles of bytes 0-15
//   Dequantize: float_value = scale * ((float)(nibble) - 8.0f)
//
// Strategy: one thread block per output row. 256 threads cooperatively iterate
// over Q4_0 blocks, dequantize nibbles on the fly, accumulate partial dot
// products, and reduce via warp shuffle + shared memory.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define Q4_0_BLOCK_SIZE 18   // bytes per Q4_0 block
#define Q4_0_GROUP_SIZE 32   // elements per Q4_0 block

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
// Duplicated per .cu file because each is compiled as a separate NVRTC module.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Warp-level reduction: sum all lanes via butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Block-level reduction: warp shuffle within warps, shared memory across warps.
// Returns the final sum in thread 0 of the block.
__device__ float block_reduce_sum(float val) {
    val = warp_reduce_sum(val);

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    unsigned int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Q4_0 matrix-vector multiply: out = dequant(weight_q4) * x
//
// Grid:  (out_dim, 1, 1)     -- one block per output row
// Block: (BLOCK_SIZE, 1, 1)  -- 256 threads per block
//
// Each thread iterates over a strided subset of Q4_0 blocks in its row,
// dequantizing 32 elements per block and accumulating the dot product with x.
extern "C" __global__ void matvec_q4_0(
    const char* __restrict__ weight_q4,  // [out_dim * row_bytes] Q4_0 packed
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    // Number of Q4_0 blocks per row. in_dim must be a multiple of 32.
    unsigned int num_blocks = in_dim / Q4_0_GROUP_SIZE;
    unsigned int row_bytes = num_blocks * Q4_0_BLOCK_SIZE;
    const char* row_ptr = weight_q4 + (unsigned long long)row * row_bytes;

    float sum = 0.0f;

    // Each thread processes blocks at stride BLOCK_SIZE.
    for (unsigned int b = threadIdx.x; b < num_blocks; b += BLOCK_SIZE) {
        const char* block_ptr = row_ptr + b * Q4_0_BLOCK_SIZE;

        // Read f16 scale (2 bytes, little-endian) and convert to f32.
        unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                                  | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = f16_bits_to_f32(scale_bits);

        // Base index into x for this block's 32 elements.
        unsigned int x_base = b * Q4_0_GROUP_SIZE;

        // Unpack 16 nibble bytes into 32 dequantized values, dot with x.
        // GGML de-interleaved layout: lo nibble of byte i = element i,
        // hi nibble of byte i = element i+16.
        float block_sum = 0.0f;
        const char* qs = block_ptr + 2;

        for (unsigned int i = 0; i < 16; i++) {
            unsigned char byte_val = (unsigned char)qs[i];
            unsigned int nibble_lo = byte_val & 0x0Fu;
            unsigned int nibble_hi = (byte_val >> 4) & 0x0Fu;

            float dq_lo = (float)nibble_lo - 8.0f;
            float dq_hi = (float)nibble_hi - 8.0f;

            block_sum += dq_lo * x[x_base + i]
                       + dq_hi * x[x_base + i + 16];
        }

        sum += scale * block_sum;
    }

    // Reduce partial sums across the block.
    sum = block_reduce_sum(sum);

    // Thread 0 writes the final result.
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// Q4_0 matrix-vector multiply with fused residual addition:
// out = dequant(weight_q4) * x + residual
//
// Fuses the output projection and residual connection into a single kernel,
// eliminating one global memory read+write pass over out_dim elements.
//
// Grid:  (out_dim, 1, 1)     -- one block per output row
// Block: (BLOCK_SIZE, 1, 1)  -- 256 threads per block
extern "C" __global__ void matvec_q4_0_residual(
    const char* __restrict__ weight_q4,    // [out_dim * row_bytes] Q4_0 packed
    const float* __restrict__ x,           // [in_dim]
    const float* __restrict__ residual,    // [out_dim], added to output
    float* __restrict__ out,               // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    unsigned int num_blocks = in_dim / Q4_0_GROUP_SIZE;
    unsigned int row_bytes = num_blocks * Q4_0_BLOCK_SIZE;
    const char* row_ptr = weight_q4 + (unsigned long long)row * row_bytes;

    float sum = 0.0f;

    for (unsigned int b = threadIdx.x; b < num_blocks; b += BLOCK_SIZE) {
        const char* block_ptr = row_ptr + b * Q4_0_BLOCK_SIZE;

        unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                                  | ((unsigned short)(unsigned char)block_ptr[1] << 8);
        float scale = f16_bits_to_f32(scale_bits);

        unsigned int x_base = b * Q4_0_GROUP_SIZE;
        float block_sum = 0.0f;
        const char* qs = block_ptr + 2;

        for (unsigned int i = 0; i < 16; i++) {
            unsigned char byte_val = (unsigned char)qs[i];
            unsigned int nibble_lo = byte_val & 0x0Fu;
            unsigned int nibble_hi = (byte_val >> 4) & 0x0Fu;

            float dq_lo = (float)nibble_lo - 8.0f;
            float dq_hi = (float)nibble_hi - 8.0f;

            block_sum += dq_lo * x[x_base + i]
                       + dq_hi * x[x_base + i + 16];
        }

        sum += scale * block_sum;
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        out[row] = sum + residual[row];
    }
}
