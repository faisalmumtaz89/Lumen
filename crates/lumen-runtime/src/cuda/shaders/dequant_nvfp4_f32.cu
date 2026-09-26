// NVFP4 (E2M1 with per-16-element E4M3 block scales) software decode.
//
// Decodes an NVFP4 plane into the f32 prefill dequant scratch, each weight multiplied by its folded scale.
// Decode uses `matvec_nvfp4_wide_f32` instead, which decodes the same codes and scales but multiplies each
// 16-weight group's partial sum by the folded scale once, so the two paths do not round identically. For
// finite scale codes, every f32 this produces equals `lumen_format::planar_dequant::dequantize_nvfp4`'s,
// with no tolerance; a NaN scale code decodes to a NaN whose payload need not match the host's.
//
// E2M1 decode, one code (the low 4 bits of a byte; the high nibble is the next weight in the block):
//   1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit. Both zero codes decode to +0.0 (the format has
//   ONE zero: a zero magnitude drops the sign), the subnormal 0x1 is 0.5, and the largest is 6.0.
//
// The NVFP4 block is 16 weights (one E4M3 block scale) under a per-tensor global scale:
//   value = e2m1(code) * e4m3(block_scale) * global_scale
// All three are multiplied in f32 in the order the host decoder uses: the two scales are folded first,
// `e2m1(code) * (e4m3(block_scale) * global_scale)` — the host decoder's test
// `nvfp4_folds_the_scales_before_the_nibble_multiply` exists because a different association gives
// different bits.
//
// NVRTC-compatible: no system includes, extern "C" linkage, no __constant__, no #if __CUDA_ARCH__.
//
// Packing: 16 codes per block, two per byte (low nibble first), and one E4M3 block scale per block.

extern "C" __device__ __forceinline__ float lumen_e2m1_to_f32(unsigned int code)
{
    // 1 sign, 2 exp (bias 1), 1 mantissa. Mirrors the host decoder's bit construction exactly.
    unsigned int c = code & 0xFu;
    unsigned int sign = (c & 8u) << 28;
    unsigned int exp = (c >> 1) & 3u;
    unsigned int mantissa = c & 1u;
    unsigned int magnitude;
    if (exp == 0u) {
        // Subnormal: 0.5 for code 0x1, zero otherwise. 126 << 23 encodes the exponent -1.
        magnitude = (mantissa != 0u) ? (126u << 23) : 0u;
    } else {
        magnitude = ((exp + 126u) << 23) | (mantissa << 22);
    }
    // A zero magnitude drops the sign: this format has one zero, not two.
    return __uint_as_float((magnitude == 0u) ? 0u : (sign | magnitude));
}

extern "C" __device__ __forceinline__ float lumen_e4m3_to_f32(unsigned int code)
{
    // 1 sign, 4 exp (bias 7), 3 mantissa. 0x7F/0xFF are the only NaNs; there are no infinities.
    unsigned int c = code & 0xFFu;
    unsigned int sign = (c & 0x80u) << 24;
    unsigned int exp = (c >> 3) & 0xFu;
    unsigned int mantissa = c & 7u;
    if (exp == 15u && mantissa == 7u) {
        return __uint_as_float(0x7FC00000u); // quiet NaN, the host decoder's f32::NAN
    }
    // The subnormal is `mantissa / 512` in real float arithmetic, as the host decoder computes it, NOT the
    // normal-number bit construction with exponent 0, which assumes an implicit leading one a subnormal
    // does not have.
    if (exp == 0u) {
        float sub = ((float)mantissa) / 512.0f;          // exact in f32 for mantissa 0..=7
        return __uint_as_float(__float_as_uint(sub) | sign);  // ±0.0 keeps its sign
    }
    return __uint_as_float(sign | ((exp + 120u) << 23) | (mantissa << 20));
}

// One thread per FOUR output elements (one `float4`), not per block.
//
// Thread `t` writes elements [4t, 4t+4) as a single `float4` at byte offset 16t. Consecutive threads
// therefore write consecutive 16-byte runs, so a warp's store instruction covers ONE contiguous 512-byte
// span. A thread per 16-weight block would put consecutive threads 64 bytes apart and scatter every store
// across 2 KB.
//
// The work per thread is correspondingly smaller: read TWO weight bytes (elements 4t..4t+3 are bytes 2t and
// 2t+1), decode four codes, apply the block's folded scale, store one `float4`. Consecutive threads read
// consecutive 2-byte runs, so the loads coalesce too.
//
// The block a thread belongs to is `t / 4` (16 elements = 4 float4s), and its scale is that block's E4M3.
// Every value is `e2m1(code) * (e4m3(bs) * global)`, the host decoder's order.
//
// Grid:  (ceil(n / (256 * 4)), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void dequant_nvfp4_to_f32(
    const unsigned char* __restrict__ plane,  // weight | block_scale | global_scale(F32 LE)
    float* __restrict__ out,                  // [n] f32
    unsigned int n)                           // element count
{
    const unsigned int n_quads = n / 4u;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_quads) {
        return;
    }

    // The plane's layout is the converter's (`convert_hf.rs::lower_nvfp4`), so both sub-planes and the
    // global scale after them are derived from n — the same rule the matvec kernels follow.
    const unsigned int weight_bytes = n / 2u;
    const unsigned char* block_scales = plane + weight_bytes;
    const unsigned char* gs = plane + weight_bytes + n / 16u;
    const float global_scale = __uint_as_float(
        (unsigned int)gs[0] | ((unsigned int)gs[1] << 8) | ((unsigned int)gs[2] << 16) |
        ((unsigned int)gs[3] << 24));

    // Elements 4t..4t+3 live in weight bytes 2t and 2t+1, inside block `t / 4`.
    const unsigned int blk = t / 4u;
    const unsigned char* p = plane + (size_t)t * 2u;
    const unsigned int b0 = (unsigned int)p[0];
    const unsigned int b1 = (unsigned int)p[1];

    const float folded = lumen_e4m3_to_f32((unsigned int)block_scales[blk]) * global_scale;

    float4 v;
    v.x = lumen_e2m1_to_f32(b0 & 0x0Fu) * folded;
    v.y = lumen_e2m1_to_f32((b0 >> 4) & 0x0Fu) * folded;
    v.z = lumen_e2m1_to_f32(b1 & 0x0Fu) * folded;
    v.w = lumen_e2m1_to_f32((b1 >> 4) & 0x0Fu) * folded;

    reinterpret_cast<float4*>(out)[t] = v;
}
