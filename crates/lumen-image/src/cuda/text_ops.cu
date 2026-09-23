// Kernels for the text tower's GPU forward, in the reference's arithmetic.
//
// The reference runs the tower in bf16: every tensor between two operations is
// bf16, and each operation computes in f32 and rounds its result to bf16 once
// (nearest even). These kernels take and return bf16 (as `unsigned short`
// bits) and round exactly where the reference does, so the tower's output
// follows the reference's instead of drifting from it layer by layer.
//
// Reused from sources this crate already compiles:
//   - the projections: `blas::gemm_bf16` (cuBLAS, f32 accumulate) rounded to
//     bf16 once by `f32_to_bf16_bits`
//   - the attention: `flash_attn_bf16` with the whole prompt as its causal
//     text prefix, after `repeat_kv` gives every query head its key/value head
//
// Written here:
//   - rms_norm_bf16      RMSNorm: f32 in the norm, rounded, times the weight, rounded
//   - rope_bf16          the tower's half-split rotary, each product and the sum rounded
//   - repeat_kv          key/value heads expanded to the query heads they serve
//   - silu_mul_bf16      silu(gate) rounded, times up, rounded
//   - add_bf16           the residual adds
//
// *Rotation.* `image_ops.cu`'s `mrope_interleaved` rotates adjacent complex
// pairs *within* a head, pairing channel 2p with channel 2p+1. This tower pairs
// channel j with channel j + head_dim/2 (`rotate_half`) against one shared
// angle, so the two are not interchangeable.
//
// *Grouped heads.* 32 query heads over 8 key/value heads: `groups = 4`
// consecutive query heads share one key/value head (`repeat_kv`, which is
// `repeat_interleave` over the head axis).
//
// NVRTC-compatible: no includes, extern "C" linkage.

#define TEXT_NORM_THREADS 256

__device__ __forceinline__ float bf16_to_f32(unsigned short h)
{
    return __uint_as_float(((unsigned int)h) << 16);
}

// f32 -> bf16, nearest even. A second copy of `image_ops.cu`'s helper, because
// NVRTC compiles each source into its own module.
__device__ __forceinline__ unsigned short text_bf16_rne(float val)
{
    unsigned int bits = __float_as_uint(val);
    // Keep a NaN a NaN: force a mantissa bit so the round cannot carry to inf.
    if (((bits >> 23) & 0xffu) == 0xffu && (bits & 0x7fffffu) != 0u) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}

// ---------------------------------------------------------------------------
// RMSNorm over each `dim`-wide row, as the reference computes it:
//
//     y   = bf16(x * rsqrt(mean(x^2) + eps))     in f32, then rounded
//     out = bf16(weight * y)                     the bf16 weight times y, rounded
//
// The weight arrives widened to f32, which is exact for a bf16 weight. One
// block of TEXT_NORM_THREADS per row.
// ---------------------------------------------------------------------------
extern "C" __global__ void rms_norm_bf16(
    const unsigned short* __restrict__ x,  // [rows, dim]
    const float* __restrict__ weight,      // [dim]
    unsigned short* __restrict__ out,      // [rows, dim]
    unsigned int dim,
    float eps)
{
    __shared__ float partial[TEXT_NORM_THREADS / 32];
    const unsigned short* xr = x + (unsigned long long)blockIdx.x * dim;
    unsigned short* orow = out + (unsigned long long)blockIdx.x * dim;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf16_to_f32(xr[i]);
        sum += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (blockDim.x >> 5) ? partial[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
    float scale = rsqrtf(partial[0] / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float y = bf16_to_f32(text_bf16_rne(bf16_to_f32(xr[i]) * scale));
        orow[i] = text_bf16_rne(weight[i] * y);
    }
}

// ---------------------------------------------------------------------------
// The half-split rotary, in place on `[seq, heads, head_dim]`:
//
//     out = bf16( bf16(x * cos) + bf16(rotate_half(x) * sin) )
//
// with `rotate_half(x) = [-x[half:], x[:half]]` and `cos`/`sin` the bf16
// `[seq, head_dim]` tables (their second half repeats the first). One block per
// position, threads over the rotary slots, each looping over the heads.
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_bf16(
    unsigned short* __restrict__ x,             // [seq, heads * head_dim], in place
    const unsigned short* __restrict__ cos_tab, // [seq, head_dim]
    const unsigned short* __restrict__ sin_tab, // [seq, head_dim]
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim)
{
    unsigned int s = blockIdx.x;
    unsigned int half = head_dim >> 1;
    if (s >= seq) return;
    unsigned long long trig = (unsigned long long)s * head_dim;
    for (unsigned int j = threadIdx.x; j < half; j += blockDim.x) {
        float cl = bf16_to_f32(cos_tab[trig + j]), sl = bf16_to_f32(sin_tab[trig + j]);
        float ch = bf16_to_f32(cos_tab[trig + j + half]), shi = bf16_to_f32(sin_tab[trig + j + half]);
        for (unsigned int h = 0; h < heads; h++) {
            unsigned long long base = ((unsigned long long)s * heads + h) * head_dim;
            float lo = bf16_to_f32(x[base + j]);
            float hi = bf16_to_f32(x[base + j + half]);
            float lo_c = bf16_to_f32(text_bf16_rne(lo * cl));
            float hi_s = bf16_to_f32(text_bf16_rne(-hi * sl));
            float hi_c = bf16_to_f32(text_bf16_rne(hi * ch));
            float lo_s = bf16_to_f32(text_bf16_rne(lo * shi));
            x[base + j] = text_bf16_rne(lo_c + hi_s);
            x[base + j + half] = text_bf16_rne(hi_c + lo_s);
        }
    }
}

// ---------------------------------------------------------------------------
// `[seq, nkv, head_dim]` -> `[seq, nkv * groups, head_dim]`: query head h reads
// key/value head h / groups. One thread per output element.
// ---------------------------------------------------------------------------
extern "C" __global__ void repeat_kv(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned int seq,
    unsigned int nkv,
    unsigned int groups,
    unsigned int head_dim)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nq = (unsigned long long)nkv * groups;
    if (i >= (unsigned long long)seq * nq * head_dim) return;
    unsigned long long d = i % head_dim;
    unsigned long long h = (i / head_dim) % nq;
    unsigned long long s = i / (head_dim * nq);
    dst[i] = src[(s * nkv + h / groups) * head_dim + d];
}

// ---------------------------------------------------------------------------
// out = bf16( bf16(silu(gate)) * up ), silu computed in f32 as x / (1 + e^-x).
// ---------------------------------------------------------------------------
extern "C" __global__ void silu_mul_bf16(
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = bf16_to_f32(gate[i]);
    float a = bf16_to_f32(text_bf16_rne(g / (1.0f + expf(-g))));
    out[i] = text_bf16_rne(a * bf16_to_f32(up[i]));
}

// ---------------------------------------------------------------------------
// out = bf16(a + b), the residual adds.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_bf16(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = text_bf16_rne(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}
