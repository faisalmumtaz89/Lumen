// F32 matrix-vector multiply (GEMV) kernels for decode.
//
// Operation: out[i] = sum_j(weight[i * in_dim + j] * x[j])
// Weight matrix: [out_dim, in_dim] row-major
// Input vector:  [in_dim]
// Output vector: [out_dim]
//
// Strategy: one thread block per output row. 256 threads cooperatively compute
// the dot product using vectorized float4 loads (when alignment permits),
// shared memory partial sums, and warp shuffle for the final reduction.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Block-level reduction: warp shuffle within warps, then shared memory across warps.
// Returns the final sum in thread 0 of the block.
__device__ float block_reduce_sum(float val) {
    // Phase 1: reduce within each warp
    val = warp_reduce_sum(val);

    // Phase 2: first thread of each warp writes to shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE]; // 8 warps for BLOCK_SIZE=256
    unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    unsigned int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Phase 3: first warp reduces the warp sums
    if (warp_id == 0) {
        val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// F32 matrix-vector multiply: out = weight * x
//
// Grid:  (out_dim, 1, 1)     -- one block per output row
// Block: (BLOCK_SIZE, 1, 1)  -- 256 threads per block
//
// Uses float4 vectorized loads when in_dim is a multiple of 4 (guaranteed
// 16-byte alignment for every row since cudaMalloc base is 256-byte aligned
// and row stride is in_dim*4 bytes = a multiple of 16). Falls back to scalar
// loads for non-aligned dimensions.
extern "C" __global__ void matvec_f32(
    const float* __restrict__ weight,  // [out_dim, in_dim] row-major
    const float* __restrict__ x,       // [in_dim]
    float* __restrict__ out,           // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const float* row_ptr = weight + (unsigned long long)row * in_dim;
    float sum = 0.0f;

    // float4 path: only when in_dim is a multiple of 4. This guarantees every
    // row start is 16-byte aligned (base is 256B aligned, stride is in_dim*4
    // which is a multiple of 16 when in_dim % 4 == 0).
    if ((in_dim & 3u) == 0u) {
        unsigned int vec4_count = in_dim >> 2;
        const float4* row_vec4 = (const float4*)row_ptr;
        const float4* x_vec4 = (const float4*)x;

        for (unsigned int i = threadIdx.x; i < vec4_count; i += BLOCK_SIZE) {
            float4 w = row_vec4[i];
            float4 v = x_vec4[i];
            sum += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
    } else {
        // Scalar fallback for non-aligned in_dim.
        for (unsigned int j = threadIdx.x; j < in_dim; j += BLOCK_SIZE) {
            sum += row_ptr[j] * x[j];
        }
    }

    // Reduce partial sums across the block.
    sum = block_reduce_sum(sum);

    // Thread 0 writes the final result.
    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// F32 matrix-vector multiply with fused residual addition:
// out = weight * x + residual
//
// Fuses the output projection and residual connection into a single kernel,
// eliminating one global memory read+write pass over out_dim elements.
//
// Grid:  (out_dim, 1, 1)     -- one block per output row
// Block: (BLOCK_SIZE, 1, 1)  -- 256 threads per block
extern "C" __global__ void matvec_f32_residual(
    const float* __restrict__ weight,    // [out_dim, in_dim] row-major
    const float* __restrict__ x,         // [in_dim]
    const float* __restrict__ residual,  // [out_dim], added to output
    float* __restrict__ out,             // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const float* row_ptr = weight + (unsigned long long)row * in_dim;
    float sum = 0.0f;

    // float4 path when in_dim is 4-aligned (see matvec_f32 for alignment rationale).
    if ((in_dim & 3u) == 0u) {
        unsigned int vec4_count = in_dim >> 2;
        const float4* row_vec4 = (const float4*)row_ptr;
        const float4* x_vec4 = (const float4*)x;

        for (unsigned int i = threadIdx.x; i < vec4_count; i += BLOCK_SIZE) {
            float4 w = row_vec4[i];
            float4 v = x_vec4[i];
            sum += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
    } else {
        // Scalar fallback for non-aligned in_dim.
        for (unsigned int j = threadIdx.x; j < in_dim; j += BLOCK_SIZE) {
            sum += row_ptr[j] * x[j];
        }
    }

    // Block-level reduction.
    sum = block_reduce_sum(sum);

    // Thread 0 writes matvec result + residual.
    if (threadIdx.x == 0) {
        out[row] = sum + residual[row];
    }
}
