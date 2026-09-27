// Native prefill producer kernels: shared helpers.
//
// This file and the kernel files after it are compiled as one NVRTC module for compute_120a, with no
// include path and no header: BF16 is a 16-bit pattern, FP8 E4M3 and FP4 E2M1 are produced by the
// PTX conversions below. In the producer kernels every arithmetic step whose rounding the outputs
// depend on is written with an explicit rounding (the _rn intrinsics) or as the PTX instruction it
// must be, so neither the compiler's contraction nor the driver's assembler can change a result;
// the GDN and attention kernels also use the math library, and their load-time self-checks catch a
// build that changes their outputs.
//
// Activation layouts: BF16 row-major. FP8 codes are one byte per element. FP4 codes pack two elements
// per byte (element 2i in the low nibble of byte i); their E4M3 block scales, one per 16 elements,
// use the 128x4 tiled layout of `native_sf_offset`, and the scale bytes of rows [M, pad128(M)) are
// written as zero.

#define NATIVE_HIDDEN 5120
#define NATIVE_INTER 17408

__device__ __forceinline__ float native_bf16_to_f32(unsigned short b)
{
    return __uint_as_float((unsigned int)b << 16);
}

// Two F32 values rounded to nearest even as one BF16 pair: lo in bits 15:0, hi in bits 31:16.
__device__ __forceinline__ unsigned int native_bf16x2_rn(float lo, float hi)
{
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float native_lo(unsigned int pair)
{
    return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float native_hi(unsigned int pair)
{
    return __uint_as_float(pair & 0xFFFF0000u);
}

__device__ __forceinline__ float native_mul_ftz(float a, float b)
{
    float d;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float native_rcp_approx_ftz(float a)
{
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float native_rsqrt_approx_ftz(float a)
{
    float d;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float native_div_full(float a, float b)
{
    float d;
    asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

// 1 / (1 + 2^((0 - g) * log2 e)) with ex2.approx and div.full: the approximate sigmoid of the gates.
__device__ __forceinline__ float native_sigmoid(float g)
{
    float t = __fmul_rn(__fsub_rn(0.0f, g), 1.44269502162933349609375f);
    float e;
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    return native_div_full(1.0f, __fadd_rn(e, 1.0f));
}

// Static FP8 E4M3 code of x with inv = div.full(1, input_scale): clamp(x * inv, +-448), round to nearest
// even. max/min drop a NaN operand, so a NaN input becomes -448.
__device__ __forceinline__ unsigned int native_fp8x2(float lo, float hi, float inv)
{
    float a = fminf(fmaxf(__fmul_rn(lo, inv), -448.0f), 448.0f);
    float b = fminf(fmaxf(__fmul_rn(hi, inv), -448.0f), 448.0f);
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
    return r;
}

// Eight BF16 values (four pairs) -> eight FP8 codes.
__device__ __forceinline__ uint2 native_fp8x8(const unsigned int* v, float inv)
{
    uint2 r;
    r.x = native_fp8x2(native_lo(v[0]), native_hi(v[0]), inv)
        | (native_fp8x2(native_lo(v[1]), native_hi(v[1]), inv) << 16);
    r.y = native_fp8x2(native_lo(v[2]), native_hi(v[2]), inv)
        | (native_fp8x2(native_lo(v[3]), native_hi(v[3]), inv) << 16);
    return r;
}

// Byte offset of block scale `blk` of row `row` among `n_blk` scale columns: tiles of 128 rows x 4 scale
// columns (512 bytes), tiles row-major, and inside a tile (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4.
__device__ __forceinline__ unsigned long long native_sf_offset(unsigned int row, unsigned int blk, unsigned int n_blk)
{
    unsigned int tiles_c = (n_blk + 3) / 4;
    return ((unsigned long long)(row / 128) * tiles_c + blk / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4
        + blk % 4;
}

__device__ __forceinline__ unsigned int native_pad128(unsigned int m)
{
    return (m + 127) / 128 * 128;
}

// Lane-wise maximum of two BF16 pairs; a NaN lane yields the other operand's lane.
__device__ __forceinline__ unsigned int native_max_bf16x2(unsigned int a, unsigned int b)
{
    unsigned int r;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Lane-wise magnitude maximum over n BF16 pairs: |v0|, then max with each further |vi|.
__device__ __forceinline__ unsigned int native_absmax_pairs(const unsigned int* v, int n)
{
    unsigned int m;
    asm("abs.bf16x2 %0, %1;" : "=r"(m) : "r"(v[0]));
    for (int i = 1; i < n; i++) {
        unsigned int a;
        asm("abs.bf16x2 %0, %1;" : "=r"(a) : "r"(v[i]));
        m = native_max_bf16x2(m, a);
    }
    return m;
}

// E4M3 block scale byte and element multiplier for a block whose largest magnitude is `vec_max`, with the
// global scale S = 1 / input_scale:
//   SF    = e4m3(S * (vec_max * rcp(6)))
//   scale = vec_max != 0 ? rcp(f32(SF) * rcp(S)) : 0
// (flush-to-zero multiplies, approximate reciprocals).
__device__ __forceinline__ float native_fp4_block_scale(float vec_max, float s, unsigned int* sf)
{
    float sfv = native_mul_ftz(s, native_mul_ftz(vec_max, native_rcp_approx_ftz(6.0f)));
    unsigned short code;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(code) : "f"(0.0f), "f"(sfv));
    *sf = code & 0xFFu;
    unsigned int half2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half2) : "h"(code));
    float sfq;
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l; }" : "=f"(sfq) : "r"(half2));
    unsigned int nonzero;
    asm("{ .reg .pred p; setp.neu.ftz.f32 p, %1, 0f00000000; selp.u32 %0, 1, 0, p; }" : "=r"(nonzero) : "f"(vec_max));
    return nonzero ? native_rcp_approx_ftz(native_mul_ftz(sfq, native_rcp_approx_ftz(s))) : 0.0f;
}

// The block maximum from the lane-wise maxima of its pairs: (x > y) ? x : y.
__device__ __forceinline__ float native_fp4_vec_max(unsigned int lane_max)
{
    float x = native_lo(lane_max);
    float y = native_hi(lane_max);
    return x > y ? x : y;
}

// Four BF16 pairs -> eight E2M1 codes (x * scale, flush-to-zero multiply, round to nearest even,
// saturating at 6).
__device__ __forceinline__ unsigned int native_fp4x8(const unsigned int* v, float scale)
{
    unsigned int r = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float lo = native_mul_ftz(native_lo(v[i]), scale);
        float hi = native_mul_ftz(native_hi(v[i]), scale);
        unsigned int b;
        asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; cvt.u32.u8 %0, b; }" : "=r"(b) : "f"(hi), "f"(lo));
        r |= b << (8 * i);
    }
    return r;
}
