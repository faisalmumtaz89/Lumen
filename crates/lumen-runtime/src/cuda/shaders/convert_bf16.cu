// ============================================================================
// F32 <-> BF16 bulk conversion kernels for CUDA.
//
// Used to convert activations between F32 (compute precision) and BF16
// (cuBLAS GemmEx precision for BF16-weight prefill).
//
// BF16 is the top 16 bits of an IEEE 754 binary32 value, so conversion is
// a simple right-shift by 16 (with RNE rounding for f32->bf16). The hardware
// `cvt.rn.bf16.f32` PTX instruction is available on SM_80+ (Ampere). On
// older GPUs we fall back to a software RNE round.
//
// Two variants:
//   f32_to_bf16_vec   -- scalar, 1 element/thread.   Dispatch: ceil(n/256), 256
//   f32_to_bf16_vec4  -- vectorized, 4 elems/thread. Dispatch: ceil(n/(256*4)), 256
//
// Output layout (same as `f32_to_f16_vec`): 16-bit `unsigned short` per element.
// cuBLAS reads these as `CUDA_R_16BF`.
// ============================================================================

/// Hardware f32->bf16 conversion via PTX (single instruction on SM 80+).
///
/// Falls back to a software round-to-nearest-even when SM < 80. The
/// fallback matches the IEEE 754 RNE semantics used by PyTorch / cuBLAS.
__device__ __forceinline__ unsigned short f32_to_bf16_bits(float val) {
#if __CUDA_ARCH__ >= 800
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
#else
    // Software RNE: take the top 16 bits of the F32, with round-to-nearest-even
    // applied to the discarded low 16 bits. Standard rounding rule:
    //   if low_16 > 0x8000 -> round up
    //   if low_16 < 0x8000 -> round down
    //   if low_16 == 0x8000 -> round to even (LSB of result must be 0)
    unsigned int bits = __float_as_uint(val);
    // Propagate NaNs (set high mantissa bit so RNE round doesn't accidentally
    // produce Inf from a NaN with no mantissa bits in the top 7).
    if (((bits >> 23) & 0xff) == 0xff && (bits & 0x7fffff) != 0) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7fffu + lsb;
    bits += rounding_bias;
    return (unsigned short)(bits >> 16);
#endif
}

// ============================================================================
// f32_to_bf16_vec: Convert N f32 values to bf16 (stored as unsigned short).
// Scalar path: 1 element per thread.
//
// Dispatch: grid = ceil(n / 256), block = 256
// ============================================================================
extern "C" __global__ void f32_to_bf16_vec(
    const float*          __restrict__ src,  // [n] f32
    unsigned short*       __restrict__ dst,  // [n] bf16 bits
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[gid] = f32_to_bf16_bits(src[gid]);
}

// ============================================================================
// f32_to_bf16_vec4: Vectorized F32->BF16 conversion, 4 elements per thread.
//
// Uses float4 loads to coalesce memory reads (128 bytes per warp vs 32).
// Writes are packed into uint2 (2x unsigned int = 4x unsigned short) for
// coalesced 64-bit stores.
//
// Dispatch: grid = ceil(n / (256 * 4)), block = 256
// Handles n not divisible by 4 via a tail loop.
// ============================================================================
extern "C" __global__ void f32_to_bf16_vec4(
    const float*          __restrict__ src,  // [n] f32
    unsigned short*       __restrict__ dst,  // [n] bf16 bits
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = gid * 4u;

    if (base + 3u < n) {
        // Fast path: process 4 elements at once with vectorized load
        float4 v = reinterpret_cast<const float4*>(src)[gid];
        unsigned short b0 = f32_to_bf16_bits(v.x);
        unsigned short b1 = f32_to_bf16_bits(v.y);
        unsigned short b2 = f32_to_bf16_bits(v.z);
        unsigned short b3 = f32_to_bf16_bits(v.w);

        // Pack 4 bfloat values into 2 unsigned ints for coalesced 64-bit store
        unsigned int pack0 = ((unsigned int)b1 << 16) | (unsigned int)b0;
        unsigned int pack1 = ((unsigned int)b3 << 16) | (unsigned int)b2;
        reinterpret_cast<uint2*>(dst)[gid] = make_uint2(pack0, pack1);
    } else {
        // Tail: handle remaining elements
        for (unsigned int i = base; i < n && i < base + 4u; i++) {
            dst[i] = f32_to_bf16_bits(src[i]);
        }
    }
}
