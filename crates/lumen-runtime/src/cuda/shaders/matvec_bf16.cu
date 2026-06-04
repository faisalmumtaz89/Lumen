// ============================================================================
// BF16 (brain floating-point) matrix-vector multiply for CUDA.
//
// IEEE-style bfloat16 weights: each weight is 2 bytes (unsigned short),
// no block structure, no scale factors. Plain contiguous bf16 values.
// Weight matrix [out_dim, in_dim] stored row-major as out_dim * in_dim * 2 bytes.
//
// BF16 layout: 1 sign | 8 exponent | 7 mantissa. Same dynamic range as F32
// (both have 8-bit exponent); precision is 7 fractional bits (~2.4 decimal).
//
// Strategy: one block per output row, 256 threads per block, warp shuffle
// reduction. Each thread strides across the input dimension, accumulating
// in f32. Uses packed u32 loads for 2 BF16 values at once.
//
// NVRTC-compatible: hardware cvt.f32.bf16 PTX instruction available on
// SM_80+ (PTX ISA 7.0). The runtime loads this file with compute_80; older
// GPUs do not support the conversion intrinsic and will not select this
// dispatch path.
// ============================================================================

// ---------------------------------------------------------------------------
// BF16 -> F32 conversion (NVRTC-safe, no cuda_bf16.h required)
//
// BF16 is the top 16 bits of an IEEE 754 binary32 value, so conversion to
// F32 is a left-shift by 16. Use this fallback when cvt.f32.bf16 is not
// available; on SM_80+ the compiler typically promotes the shift to the
// dedicated CVT instruction anyway, but the explicit bit-cast is provably
// equivalent and avoids any compute-capability dependency at compile time.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short b) {
    unsigned int x = ((unsigned int)b) << 16;
    return __int_as_float((int)x);
}

// ---------------------------------------------------------------------------
// Warp-level reduction (sum)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_bf16(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction (sum) using shared memory + warp shuffle
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum_bf16(float val) {
    __shared__ float shared[8]; // max 256 threads = 8 warps

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum_bf16(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < (unsigned int)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum_bf16(val);
    }
    return val;
}

// ============================================================================
// matvec_bf16: out[row] = dot(W_bf16[row, :], x[:])
//
// Dispatch: grid = (out_dim,), block = (256,)
// One block per output row.
// ============================================================================
extern "C" __global__ void matvec_bf16(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr =
        weight_bf16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    // Stride across in_dim, accumulating in f32.
    // Load 2 BF16 values at once via a single 32-bit load when aligned.
    unsigned int aligned_in = in_dim & ~1u; // floor to even
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = bf16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    // Handle odd trailing element
    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += bf16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    // Block-level reduction
    sum = block_reduce_sum_bf16(sum);

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// matvec_bf16_residual: out[row] = dot(W_bf16[row, :], x[:]) + residual[row]
//
// Fused residual add: saves one kernel launch and one global memory pass.
// Dispatch: grid = (out_dim,), block = (256,)
// ============================================================================
extern "C" __global__ void matvec_bf16_residual(
    const unsigned short* __restrict__ weight_bf16, // [out_dim * in_dim] bf16 bits
    const float*          __restrict__ x,           // [in_dim] f32
    float*                __restrict__ out,          // [out_dim] f32
    const float*          __restrict__ residual,     // [out_dim] f32
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned short* row_ptr =
        weight_bf16 + (unsigned long long)row * in_dim;

    float sum = 0.0f;

    unsigned int aligned_in = in_dim & ~1u;
    for (unsigned int j = threadIdx.x * 2u; j < aligned_in; j += blockDim.x * 2u) {
        unsigned int packed = *(const unsigned int*)(row_ptr + j);
        float w0 = bf16_bits_to_f32((unsigned short)(packed & 0xffffu));
        float w1 = bf16_bits_to_f32((unsigned short)(packed >> 16));
        sum += w0 * x[j] + w1 * x[j + 1u];
    }

    if (in_dim & 1u) {
        unsigned int j = aligned_in + threadIdx.x;
        if (j < in_dim) {
            sum += bf16_bits_to_f32(row_ptr[j]) * x[j];
        }
    }

    sum = block_reduce_sum_bf16(sum);

    if (threadIdx.x == 0) {
        out[row] = sum + residual[row];
    }
}
