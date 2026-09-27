// FP8 E4M3 software decode with a per-tensor F32 scale.
//
// Decodes an FP8 plane into the f32 prefill dequant scratch. For every finite code, the f32 it produces
// equals `lumen_format::planar_dequant::dequantize_fp8`'s bit for bit; a NaN code decodes to a NaN whose
// payload need not match the host's. Decode uses `matvec_fp8_f32` instead.
//
// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. Unlike E5M2 there are NO infinities, and 0x7F/0xFF are
// the format's only two NaNs (both return `is_nan()`). The largest finite magnitude is 448.0. `-0.0` is
// preserved — the host decoder pins that, so this must too.
//
// value = e4m3(code) * scale
//
// NVRTC-compatible: no system includes, extern "C" linkage, no __constant__, no #if __CUDA_ARCH__.
//
// Grid:  (ceil(n / 256), 1, 1)
// Block: (256, 1, 1)

extern "C" __global__ void dequant_fp8_to_f32(
    const unsigned char* __restrict__ plane, // weight[n] | global_scale(F32 LE)
    float* __restrict__ out,                 // [n] f32
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const unsigned char* gs = plane + n;
    const float scale = __uint_as_float(
        (unsigned int)gs[0] | ((unsigned int)gs[1] << 8) | ((unsigned int)gs[2] << 16) |
        ((unsigned int)gs[3] << 24));
    unsigned int c = (unsigned int)plane[i];
    unsigned int sign = (c & 0x80u) << 24;
    unsigned int exp = (c >> 3) & 0xFu;
    unsigned int mantissa = c & 7u;

    // The format's only two NaNs: exponent all-ones with mantissa all-ones.
    if (exp == 15u && mantissa == 7u) {
        out[i] = __uint_as_float(0x7FC00000u);
        return;
    }

    // The subnormal is `mantissa / 512` in real float arithmetic, as the host decoder computes it, NOT the
    // normal-number bit construction with exponent 0, which assumes an implicit leading one a subnormal
    // does not have.
    float v;
    if (exp == 0u) {
        v = ((float)mantissa) / 512.0f;             // exact in f32 for mantissa 0..=7
        v = __uint_as_float(__float_as_uint(v) | sign);   // ±0.0 keeps its sign
    } else {
        v = __uint_as_float(sign | ((exp + 120u) << 23) | (mantissa << 20));
    }
    out[i] = v * scale;
}
