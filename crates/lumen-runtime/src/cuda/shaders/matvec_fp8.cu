// FP8 E4M3 matrix-vector multiply (GEMV), plain, residual, residual-rounded and three-matrix, against
// an f32 activation.
//
// Layout served: ONE plane, exactly as the converter writes it (`convert_hf.rs::lower_fp8` concatenates
// weight | global_scale, then the activation scale when the checkpoint has one):
//   plane = weight F8 [out_dim, in_dim] | weight_scale F32 LE (4 bytes) [| input_scale F32 LE, not read here]
//   value = e4m3(code) * weight_scale
//
// The scale is a PER-TENSOR scalar, unlike NVFP4's per-16 block scale, so it factors out of the whole dot
// product: `out[i] = weight_scale * sum_j e4m3(w[i,j]) * x[j]`. It is applied once per row after the
// reduction, one multiply per row rather than one per weight. Reading it from the plane, right after the
// weights, keeps FP8 on the same single-argument contract NVFP4 uses, so one launch helper serves both.
//
// The residual twin exists for the attention `wo` sites: the caller needs `y = W*x + residual`
// accumulated, so the second entry point takes a residual vector and adds it on the lane-0 write. It is a
// separate kernel rather than a branch because the K-quant set does the same, and a branch would put a
// load of a possibly-null pointer on the hot path.
//
// NVRTC-compatible: no system includes, extern "C" linkage, no __constant__, no #if __CUDA_ARCH__.
//
// Grid: (ceil(out_dim / (FP8_THREADS/32)), 1, 1)   Block: (FP8_THREADS, 1, 1), 1 KB static shared memory.

#ifndef FP8_THREADS
#define FP8_THREADS 128
#endif

// E4M3 decode: 1 sign, 4 exp (bias 7), 3 mantissa. No infinities; 0x7F/0xFF are the only NaNs; the largest
// finite magnitude is 448.0; -0.0 is preserved. Identical arithmetic to `dequant_fp8_to_f32`, which is
// bit-identical to the host decoder in `lumen_format::planar_dequant` on all 256 codes.
extern "C" __device__ __forceinline__ float fp8_e4m3_to_f32(unsigned int code)
{
    unsigned int c = code & 0xFFu;
    unsigned int sign = (c & 0x80u) << 24;
    unsigned int exp = (c >> 3) & 0xFu;
    unsigned int mantissa = c & 7u;
    if (exp == 15u && mantissa == 7u) {
        return __uint_as_float(0x7FC00000u);
    }
    // The subnormal is a DIVISION, not a bit pattern: a subnormal has no implicit leading one, so the
    // normal-number construction with exponent 0 gives the wrong value.
    if (exp == 0u) {
        float sub = ((float)mantissa) / 512.0f;
        return __uint_as_float(__float_as_uint(sub) | sign);
    }
    return __uint_as_float(sign | ((exp + 120u) << 23) | (mantissa << 20));
}

// The 256 decoded E4M3 values, built by the block from `fp8_e4m3_to_f32` itself, so a lookup returns exactly
// the bits the decoder would. Reading a table replaces the decoder's shifts, masks and branches per weight
// with one shared-memory load, with bit-identical output.
extern "C" __device__ __forceinline__ void fp8_fill_lut(float* lut)
{
    for (unsigned int c = threadIdx.x; c < 256u; c += blockDim.x) {
        lut[c] = fp8_e4m3_to_f32(c);
    }
    __syncthreads();
}

// One row's dot product, returned per lane; the caller reduces.
//
// Lane `l` owns columns [l*4, l*4+4) of every 128-column stripe, stripes in ascending order, and accumulates
// `acc += value * x` in that sequence. The four stripes of an iteration are all LOADED before any is consumed
// (their weight bytes as one `uchar4` each, their activations as one `float4` each), which keeps more loads
// in flight than consuming each stripe as it arrives — but the ADDS happen in the same order as a
// stripe-at-a-time walk, so the result is bit-identical to it.
//
// in_dim is a multiple of 4, enforced at admission, so every row is covered by whole 4-byte groups.
extern "C" __device__ __forceinline__ float fp8_row_dot(
    const unsigned char* __restrict__ wrow,   // [in_dim]
    const float* __restrict__ x,
    const float* __restrict__ lut,            // [256], shared memory
    unsigned int in_dim)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int stripe = 32u * 4u;
    float acc = 0.0f;
    unsigned int col = lane * 4u;
    for (; col + 3u * stripe + 3u < in_dim; col += 4u * stripe) {
        const uchar4 w0 = *reinterpret_cast<const uchar4*>(wrow + col);
        const uchar4 w1 = *reinterpret_cast<const uchar4*>(wrow + col + stripe);
        const uchar4 w2 = *reinterpret_cast<const uchar4*>(wrow + col + 2u * stripe);
        const uchar4 w3 = *reinterpret_cast<const uchar4*>(wrow + col + 3u * stripe);
        const float4 x0 = *reinterpret_cast<const float4*>(x + col);
        const float4 x1 = *reinterpret_cast<const float4*>(x + col + stripe);
        const float4 x2 = *reinterpret_cast<const float4*>(x + col + 2u * stripe);
        const float4 x3 = *reinterpret_cast<const float4*>(x + col + 3u * stripe);
        acc += lut[w0.x] * x0.x;
        acc += lut[w0.y] * x0.y;
        acc += lut[w0.z] * x0.z;
        acc += lut[w0.w] * x0.w;
        acc += lut[w1.x] * x1.x;
        acc += lut[w1.y] * x1.y;
        acc += lut[w1.z] * x1.z;
        acc += lut[w1.w] * x1.w;
        acc += lut[w2.x] * x2.x;
        acc += lut[w2.y] * x2.y;
        acc += lut[w2.z] * x2.z;
        acc += lut[w2.w] * x2.w;
        acc += lut[w3.x] * x3.x;
        acc += lut[w3.y] * x3.y;
        acc += lut[w3.z] * x3.z;
        acc += lut[w3.w] * x3.w;
    }
    // Remaining whole stripes, one at a time, continuing the same order.
    for (; col + 3u < in_dim; col += stripe) {
        const uchar4 w = *reinterpret_cast<const uchar4*>(wrow + col);
        const float4 xv = *reinterpret_cast<const float4*>(x + col);
        acc += lut[w.x] * xv.x;
        acc += lut[w.y] * xv.y;
        acc += lut[w.z] * xv.z;
        acc += lut[w.w] * xv.w;
    }
    return acc;
}

extern "C" __device__ __forceinline__ float fp8_warp_reduce(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// The per-tensor scale: the 4 bytes right after the weights, found from the dimensions, so it needs no
// separate buffer.
extern "C" __device__ __forceinline__ float fp8_plane_scale(
    const unsigned char* plane, unsigned int out_dim, unsigned int in_dim)
{
    const unsigned char* gs = plane + (size_t)out_dim * in_dim;
    return __uint_as_float((unsigned int)gs[0] | ((unsigned int)gs[1] << 8) |
                           ((unsigned int)gs[2] << 16) | ((unsigned int)gs[3] << 24));
}

extern "C" __global__ __launch_bounds__(FP8_THREADS, 1) void matvec_fp8_f32(
    const unsigned char* __restrict__ plane,    // weight[out_dim*in_dim] | weight_scale(F32 LE)
    const float* __restrict__ x,                // [in_dim] f32
    float* __restrict__ out,                    // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    __shared__ float lut[256];
    fp8_fill_lut(lut);

    const unsigned int NWARP = FP8_THREADS / 32u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int stride = gridDim.x * NWARP;

    const float weight_scale = fp8_plane_scale(plane, out_dim, in_dim);

    for (unsigned int i = blockIdx.x * NWARP + warp; i < out_dim; i += stride) {
        const float acc = fp8_row_dot(plane + (size_t)i * in_dim, x, lut, in_dim);
        const float v = fp8_warp_reduce(acc);
        if (lane == 0u) {
            // The per-tensor scale is applied AFTER the reduction.
            out[i] = weight_scale * v;
        }
    }
}

// Row loop of the residual matvecs. `kRoundProduct` rounds `weight_scale * v` to f32 before the add, which is
// exactly `matvec_fp8_f32` followed by `residual_add_copy`; without it the store contracts to one FMA.
template <bool kRoundProduct>
__device__ __forceinline__ void fp8_matvec_residual_rows(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    __shared__ float lut[256];
    fp8_fill_lut(lut);

    const unsigned int NWARP = FP8_THREADS / 32u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int stride = gridDim.x * NWARP;

    const float weight_scale = fp8_plane_scale(plane, out_dim, in_dim);

    for (unsigned int i = blockIdx.x * NWARP + warp; i < out_dim; i += stride) {
        const float acc = fp8_row_dot(plane + (size_t)i * in_dim, x, lut, in_dim);
        const float v = fp8_warp_reduce(acc);
        if (lane == 0u) {
            out[i] = kRoundProduct ? __fadd_rn(residual[i], __fmul_rn(weight_scale, v))
                                   : weight_scale * v + residual[i];
        }
    }
}

extern "C" __global__ __launch_bounds__(FP8_THREADS, 1) void matvec_fp8_f32_residual(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,
    const float* __restrict__ residual,         // [out_dim], added to the result
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    fp8_matvec_residual_rows<false>(plane, x, residual, out, out_dim, in_dim);
}

// `residual + round(W*x)`: the GDN output projection with its residual add folded into the store, bit-identical
// to the unfused projection + `residual_add_copy`.
extern "C" __global__ __launch_bounds__(FP8_THREADS, 1) void matvec_fp8_f32_residual_rounded(
    const unsigned char* __restrict__ plane,
    const float* __restrict__ x,
    const float* __restrict__ residual,         // [out_dim], added to the rounded product
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    fp8_matvec_residual_rows<true>(plane, x, residual, out, out_dim, in_dim);
}

// Three FP8 matvecs over the SAME activation in one launch — the attention q (or q+gate), k and v projections.
//
// Served separately, the narrow k and v projections are launches with too few rows to fill the device, so
// their cost is mostly launch and ramp. Here block ranges select the matrix
// (ceil(d0/4) blocks for the first, then the second, then the third) and every row runs exactly
// `matvec_fp8_f32`'s row code, so each output is bit-identical to its own launch.
//
// Grid: (ceil(d0/4) + ceil(d1/4) + ceil(d2/4), 1, 1)   Block: (FP8_THREADS = 128, 1, 1), 1 KB static shared.
extern "C" __global__ __launch_bounds__(FP8_THREADS, 1) void matvec_fp8_three_f32(
    const unsigned char* __restrict__ w0,
    const unsigned char* __restrict__ w1,
    const unsigned char* __restrict__ w2,
    const float* __restrict__ x,               // [in_dim], shared by all three
    float* __restrict__ out0,
    float* __restrict__ out1,
    float* __restrict__ out2,
    unsigned int d0,
    unsigned int d1,
    unsigned int d2,
    unsigned int in_dim)
{
    __shared__ float lut[256];
    fp8_fill_lut(lut);

    const unsigned int NWARP = FP8_THREADS / 32u;
    const unsigned int b0 = (d0 + NWARP - 1u) / NWARP;
    const unsigned int b1 = (d1 + NWARP - 1u) / NWARP;
    unsigned int b = blockIdx.x;
    const unsigned char* plane = w0;
    float* out = out0;
    unsigned int out_dim = d0;
    if (b >= b0 + b1) {
        b -= b0 + b1;
        plane = w2;
        out = out2;
        out_dim = d2;
    } else if (b >= b0) {
        b -= b0;
        plane = w1;
        out = out1;
        out_dim = d1;
    }
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int i = b * NWARP + warp;
    if (i >= out_dim) {
        return;
    }
    const float weight_scale = fp8_plane_scale(plane, out_dim, in_dim);
    const float acc = fp8_row_dot(plane + (size_t)i * in_dim, x, lut, in_dim);
    const float v = fp8_warp_reduce(acc);
    if (lane == 0u) {
        out[i] = weight_scale * v;
    }
}
