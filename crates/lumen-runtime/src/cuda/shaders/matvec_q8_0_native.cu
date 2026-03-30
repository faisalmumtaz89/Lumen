// Q8_0 native warp-cooperative matvec: scalar dequant + FMA, NO x-quantization.
//
// Key insight: dp4a requires quantizing x to int8 per block (50+ instructions of
// amax/scale/quantize/pack overhead). This kernel skips dp4a entirely and uses
// simple scalar FMA: scale * (float)(signed char)q[lane] * x[lane]. The FMA approach
// is bandwidth-bound (not compute-bound), so eliminating x-quantization overhead
// lets the memory pipeline stay fully fed.
//
// Architecture:
//   - 4 warps (128 threads), NR=2 rows per block (same grid as dp4a/v1)
//   - Each warp processes ALL Q8_0 blocks for its assigned row
//   - Per block: lane 0 reads f16 scale, broadcasts via __shfl_sync
//   - Each lane reads 1 int8 quant value + 1 float x value → 1 FMA
//   - Accumulate across all blocks in registers (no per-block reduction!)
//   - Single warp_reduce_sum at the END (5 shuffles total, not per-block)
//
// Bytes per element: 34/32 = 1.0625 (same as llama.cpp's native Q8_0)
// Expected: ~130 tok/s on 8B (vs 79.2 with aligned dp4a, vs 131 llama.cpp)
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//
// Grid:  (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1) -- 4 warps x 32 threads
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define NR              2     // Rows per block
#define WARP_SIZE       32
#define BLOCK_DIM       128   // 4 warps
#define Q8_BLOCK_SIZE   32    // Elements per Q8_0 block
#define Q8_BLOCK_BYTES  34    // Bytes per Q8_0 block

// Hardware f16->f32 conversion via PTX
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

extern "C" __global__ void matvec_q8_0_native(
    const char* __restrict__ weight_q8,  // [out_dim, num_blocks * 34]
    const float* __restrict__ x,         // [in_dim]
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int r0      = blockIdx.x * NR;
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    // Each warp handles one row. With NR=2 and 4 warps: warps 0,1 handle row r0,
    // warps 2,3 handle row r0+1. Two warps per row provides 2x parallelism on
    // the K-dimension (each warp processes alternating blocks).
    const unsigned int row = r0 + (warp_id >> 1);  // warp 0,1 -> row r0; warp 2,3 -> row r0+1
    const unsigned int warp_pair = warp_id & 1;    // 0 or 1 within the pair

    if (row >= out_dim) return;

    const char* row_ptr = weight_q8 + (unsigned long long)row * row_bytes;

    float acc = 0.0f;

    // Each warp processes blocks at stride 2 (two warps share a row).
    // Warp 0 (of pair): blocks 0, 2, 4, ...
    // Warp 1 (of pair): blocks 1, 3, 5, ...
    for (unsigned int b = warp_pair; b < num_blocks; b += 2) {
        const char* bp = row_ptr + (unsigned long long)b * Q8_BLOCK_BYTES;

        // Lane 0 reads f16 scale, broadcasts to all 32 lanes (1 shuffle)
        float scale;
        if (lane == 0u) {
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            scale = f16_bits_to_f32(scale_bits);
        }
        scale = __shfl_sync(0xffffffff, scale, 0);

        // Each lane reads its int8 quant value and the corresponding x value
        // Lane `lane` handles element `b * 32 + lane` of the input
        float q_val = (float)(signed char)bp[2 + lane];
        float x_val = x[b * Q8_BLOCK_SIZE + lane];

        // Simple FMA: accumulate scale * q * x
        acc += scale * q_val * x_val;
    }

    // Warp-level reduction: sum all 32 lanes (5 shuffles)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // Two warps per row: need to combine their results via shared memory
    __shared__ float shmem[NR * 2];  // 2 partial sums per row

    if (lane == 0) {
        shmem[(warp_id >> 1) * 2 + warp_pair] = acc;
    }
    __syncthreads();

    // Warp 0 and 2 (warp_pair == 0) write the final result
    if (lane == 0 && warp_pair == 0) {
        float total = shmem[(warp_id >> 1) * 2] + shmem[(warp_id >> 1) * 2 + 1];
        out[row] = total;
    }
}

// Residual variant: out[row] += result
extern "C" __global__ void matvec_q8_0_native_residual(
    const char* __restrict__ weight_q8,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane    = threadIdx.x % WARP_SIZE;
    const unsigned int r0      = blockIdx.x * NR;
    const unsigned int num_blocks = in_dim / Q8_BLOCK_SIZE;
    const unsigned long long row_bytes = (unsigned long long)num_blocks * Q8_BLOCK_BYTES;

    const unsigned int row = r0 + (warp_id >> 1);
    const unsigned int warp_pair = warp_id & 1;

    if (row >= out_dim) return;

    const char* row_ptr = weight_q8 + (unsigned long long)row * row_bytes;
    float acc = 0.0f;

    for (unsigned int b = warp_pair; b < num_blocks; b += 2) {
        const char* bp = row_ptr + (unsigned long long)b * Q8_BLOCK_BYTES;

        float scale;
        if (lane == 0u) {
            unsigned short scale_bits = (unsigned short)(unsigned char)bp[0]
                                      | ((unsigned short)(unsigned char)bp[1] << 8);
            scale = f16_bits_to_f32(scale_bits);
        }
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_val = (float)(signed char)bp[2 + lane];
        float x_val = x[b * Q8_BLOCK_SIZE + lane];
        acc += scale * q_val * x_val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    __shared__ float shmem[NR * 2];
    if (lane == 0) {
        shmem[(warp_id >> 1) * 2 + warp_pair] = acc;
    }
    __syncthreads();

    if (lane == 0 && warp_pair == 0) {
        float total = shmem[(warp_id >> 1) * 2] + shmem[(warp_id >> 1) * 2 + 1];
        out[row] += total;  // Residual add
    }
}
