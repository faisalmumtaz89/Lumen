// ============================================================================
// F16 (half-precision) matrix-vector multiply for CUDA.
//
// IEEE 754 half-precision weights: each weight is 2 bytes (unsigned short),
// no block structure, no scale factors. Plain contiguous f16 values.
// Weight matrix [out_dim, in_dim] stored row-major as out_dim * in_dim * 2 bytes.
//
// Strategy: one block per output row, 256 threads per block, warp shuffle
// reduction. Each thread strides across the input dimension, accumulating
// in f32. Uses half2 loads (4 bytes) for 2 elements at once when aligned.
//
// NVRTC-compatible: no cuda_fp16.h dependency, inline f16->f32 conversion.
// ============================================================================

// ---------------------------------------------------------------------------
// f16 bit conversion (NVRTC-safe, no cuda_fp16.h required)
// ---------------------------------------------------------------------------

/// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
/// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
/// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short h) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// ---------------------------------------------------------------------------
// Warp-level reduction (sum)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction (sum) using shared memory + warp shuffle
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[8]; // max 256 threads = 8 warps

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < (unsigned int)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// ============================================================================
// matvec_f16: out[row] = dot(W_f16[row, :], x[:])
//
// Dispatch: grid = (out_dim,), block = (256,)
// One block per output row.
// ============================================================================
extern "C" __global__ void matvec_f16(
    const unsigned short* __restrict__ weight_f16,  // [out_dim * in_dim] f16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr = weight_f16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    // Stride across in_dim, accumulating in f32.
    // Use half2 loads (2 f16 values = 4 bytes) when possible for bandwidth.
    unsigned int aligned_in = in_dim & ~1u; // floor to even
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        // Load 2 consecutive f16 values as a single 32-bit load
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = f16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = f16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    // Handle odd trailing element
    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += f16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    // Block-level reduction
    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// matvec_f16_residual: out[row] = dot(W_f16[row, :], x[:]) + residual[row]
//
// Fused residual add: saves one kernel launch and one global memory pass.
// Dispatch: grid = (out_dim,), block = (256,)
// ============================================================================
extern "C" __global__ void matvec_f16_residual(
    const unsigned short* __restrict__ weight_f16,  // [out_dim * in_dim] f16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    const float*          __restrict__ residual,     // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr = weight_f16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    unsigned int aligned_in = in_dim & ~1u;
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = f16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = f16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += f16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        out[row] = sum + residual[row];
    }
}
