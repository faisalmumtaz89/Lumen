// ============================================================================
// F32 <-> F16 bulk conversion kernels for CUDA.
//
// Used to convert activations between F32 (compute precision) and F16
// (cuBLAS HGEMM precision).
//
// Two variants:
//   f32_to_f16_vec    -- scalar, 1 element/thread.  Dispatch: ceil(n/256), 256
//   f32_to_f16_vec4   -- vectorized, 4 elems/thread. Dispatch: ceil(n/(256*4)), 256
//                        Falls back to scalar for the tail.
//
// All conversions use hardware PTX cvt.rn.f16.f32 (single instruction on SM 53+).
// ============================================================================

/// Hardware f32->f16 conversion via PTX (single instruction on SM 53+).
/// Replaces ~30 ALU software bit-manipulation with the native CVT instruction.
/// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

/// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
/// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short h) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// ============================================================================
// f32_to_f16_vec: Convert N f32 values to f16 (stored as unsigned short).
// Scalar path: 1 element per thread.
//
// Dispatch: grid = ceil(n / 256), block = 256
// ============================================================================
extern "C" __global__ void f32_to_f16_vec(
    const float*          __restrict__ src,  // [n] f32
    unsigned short*       __restrict__ dst,  // [n] f16 bits
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[gid] = f32_to_f16_bits(src[gid]);
}

// ============================================================================
// f32_to_f16_vec4: Vectorized F32->F16 conversion, 4 elements per thread.
//
// Uses float4 loads to coalesce memory reads (128 bytes per warp vs 32).
// Writes are packed into uint2 (2x unsigned int = 4x unsigned short) for
// coalesced 64-bit stores.
//
// Dispatch: grid = ceil(n / (256 * 4)), block = 256
// Handles n not divisible by 4 via a tail loop.
// ============================================================================
extern "C" __global__ void f32_to_f16_vec4(
    const float*          __restrict__ src,  // [n] f32
    unsigned short*       __restrict__ dst,  // [n] f16 bits
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = gid * 4u;

    if (base + 3u < n) {
        // Fast path: process 4 elements at once with vectorized load
        float4 v = reinterpret_cast<const float4*>(src)[gid];
        unsigned short h0 = f32_to_f16_bits(v.x);
        unsigned short h1 = f32_to_f16_bits(v.y);
        unsigned short h2 = f32_to_f16_bits(v.z);
        unsigned short h3 = f32_to_f16_bits(v.w);

        // Pack 4 half values into 2 unsigned ints for coalesced 64-bit store
        unsigned int pack0 = ((unsigned int)h1 << 16) | (unsigned int)h0;
        unsigned int pack1 = ((unsigned int)h3 << 16) | (unsigned int)h2;
        reinterpret_cast<uint2*>(dst)[gid] = make_uint2(pack0, pack1);
    } else {
        // Tail: handle remaining elements
        for (unsigned int i = base; i < n && i < base + 4u; i++) {
            dst[i] = f32_to_f16_bits(src[i]);
        }
    }
}

// ============================================================================
// f16_to_f32_vec: Convert N f16 values (stored as unsigned short) to f32.
//
// Dispatch: grid = ceil(n / 256), block = 256
// ============================================================================
extern "C" __global__ void f16_to_f32_vec(
    const unsigned short* __restrict__ src,  // [n] f16 bits
    float*                __restrict__ dst,  // [n] f32
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[gid] = f16_bits_to_f32(src[gid]);
}
