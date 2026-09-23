// Kernels for the DiT's GPU forward, in the reference's arithmetic.
//
// The reference runs the transformer in bf16: every tensor between two
// operations is bf16, and each operation computes in f32 and rounds its result
// to bf16 once, nearest even. These kernels take and return bf16 (as
// `unsigned short` bits) and round exactly where the reference does; the
// projections are bf16 GEMMs whose output cuBLAS rounds the same way.
//
//   - zero_center_rmsnorm the text projection's `scale - 1` RMSNorm
//   - gelu_tanh           nn.GELU(approximate="tanh")
//   - pack_rows           the joint sequence, gathered from the text and
//                         image projections
//   - layernorm_scale     LayerNorm, then times the AdaLN `1 + scale`
//   - add_gated           x + tanh(gate) * y, the residual update
//   - swiglu              silu(gate) * up, the MLP's inner activation
//   - head_norm_rope      the per-head RMSNorm and the rotation of Q or K
//
// The modulation reaches the AdaLN kernels already in the form they multiply
// by: `bf16(1 + scale)` and `bf16(tanh(gate))`, rounded once per forward on the
// host, as the reference rounds each of those expressions before it
// multiplies. It is gathered per token, not broadcast per sequence: with
// `causal_condition` the target-image tokens read the `t = 0` row of a two-row
// modulation while every other token reads the sampled-timestep row.
//
// NVRTC-compatible: no includes, extern "C" linkage.

#define ROW_THREADS 256

__device__ __forceinline__ float bf16_to_f32(unsigned short h)
{
    return __uint_as_float(((unsigned int)h) << 16);
}

// f32 -> bf16, round to nearest even, the same encoding as image_ops.cu's
// `f32_to_bf16_rne`; each NVRTC module carries its own copy.
__device__ __forceinline__ unsigned short bf16_rne(float val)
{
    unsigned int bits = __float_as_uint(val);
    if (((bits >> 23) & 0xffu) == 0xffu && (bits & 0x7fffffu) != 0u) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}

// The sum of `v` over a block of ROW_THREADS, in a fixed order; every thread
// gets the result.
__device__ __forceinline__ float block_sum(float v, float* s_part)
{
    s_part[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int stride = ROW_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float total = s_part[0];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------
// Zero-centred RMSNorm, all in f32 and rounded once, as the reference's
// `QwenImage21ZeroCenterRMSNorm` computes it:
//
//     out = bf16(x * rsqrt(mean(x^2) + eps) * (w + 1))
//
// The checkpoint stores `scale - 1`; the weight arrives widened to f32.
// One block of ROW_THREADS per row.
// ---------------------------------------------------------------------------
extern "C" __global__ void zero_center_rmsnorm(
    const unsigned short* __restrict__ x,  // [rows, dim]
    const float* __restrict__ weight,      // [dim], stored as (scale - 1)
    unsigned short* __restrict__ out,      // [rows, dim]
    unsigned int dim,
    float eps)
{
    __shared__ float s_part[ROW_THREADS];
    const unsigned short* r = x + (unsigned long long)blockIdx.x * dim;
    unsigned short* o = out + (unsigned long long)blockIdx.x * dim;

    float sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        float v = bf16_to_f32(r[i]);
        sq += v * v;
    }
    float inv = rsqrtf(block_sum(sq, s_part) / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        o[i] = bf16_rne(bf16_to_f32(r[i]) * inv * (weight[i] + 1.0f));
    }
}

// ---------------------------------------------------------------------------
// GELU, tanh approximation, in f32 and rounded:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))). One thread per element.
// ---------------------------------------------------------------------------
extern "C" __global__ void gelu_tanh(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out[i] = bf16_rne(0.5f * v * (1.0f + tanhf(inner)));
}

// ---------------------------------------------------------------------------
// out[t, :] = source[t] >= 0 ? txt[source[t], :] : img[-source[t] - 1, :]
//
// The joint sequence the blocks run over is the text projection's rows and
// the image projection's rows in slot order. Both projections are already on
// the device, so the sequence is gathered here rather than assembled on the
// host and uploaded.
// ---------------------------------------------------------------------------
extern "C" __global__ void pack_rows(
    const unsigned short* __restrict__ txt,  // [text_rows, cols]
    const unsigned short* __restrict__ img,  // [image_rows, cols]
    const int* __restrict__ source,          // [rows]
    unsigned short* __restrict__ out,        // [rows, cols]
    unsigned int rows,
    unsigned int cols)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    int s = source[r];
    const unsigned short* src = (s >= 0)
        ? txt + (unsigned long long)s * cols
        : img + (unsigned long long)(-(long long)s - 1) * cols;
    out[i] = src[c];
}

// ---------------------------------------------------------------------------
// out[t, c] = bf16(bf16(layernorm(x[t])[c]) * one_plus[mod_row[t], col_off + c])
//
// `one_plus` holds `bf16(1 + scale)`. The LayerNorm has no affine parameters
// and computes in f32: the mean first, then the squared deviations, each
// reduced in a fixed order. One block of ROW_THREADS per row.
// ---------------------------------------------------------------------------
extern "C" __global__ void layernorm_scale(
    const unsigned short* __restrict__ x,        // [rows, dim]
    const unsigned short* __restrict__ one_plus, // [nmod, mod_stride]
    const int* __restrict__ mod_row,             // [rows]
    unsigned short* __restrict__ out,            // [rows, dim]
    unsigned int dim,
    unsigned int col_off,
    unsigned int mod_stride,
    float eps)
{
    __shared__ float s_part[ROW_THREADS];
    const unsigned short* r = x + (unsigned long long)blockIdx.x * dim;
    unsigned short* o = out + (unsigned long long)blockIdx.x * dim;
    const unsigned short* m =
        one_plus + (unsigned long long)mod_row[blockIdx.x] * mod_stride + col_off;
    float n = (float)dim;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        sum += bf16_to_f32(r[i]);
    }
    float mean = block_sum(sum, s_part) / n;

    float sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        float d = bf16_to_f32(r[i]) - mean;
        sq += d * d;
    }
    float inv = rsqrtf(block_sum(sq, s_part) / n + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        float y = bf16_to_f32(bf16_rne((bf16_to_f32(r[i]) - mean) * inv));
        o[i] = bf16_rne(y * bf16_to_f32(m[i]));
    }
}

// ---------------------------------------------------------------------------
// x[t, c] = bf16(x[t, c] + bf16(tanh_gate[mod_row[t], col_off + c] * y[t, c]))
//
// In place on the residual stream. `tanh_gate` holds `bf16(tanh(gate))`.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_gated(
    unsigned short* __restrict__ x,               // [rows, cols], updated
    const unsigned short* __restrict__ y,         // [rows, cols]
    const unsigned short* __restrict__ tanh_gate, // [nmod, mod_stride]
    const int* __restrict__ mod_row,              // [rows]
    unsigned int rows,
    unsigned int cols,
    unsigned int col_off,
    unsigned int mod_stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    float g = bf16_to_f32(tanh_gate[(unsigned long long)mod_row[r] * mod_stride + col_off + c]);
    float gy = bf16_to_f32(bf16_rne(g * bf16_to_f32(y[i])));
    x[i] = bf16_rne(bf16_to_f32(x[i]) + gy);
}

// ---------------------------------------------------------------------------
// out[i] = bf16(bf16(silu(gate[i])) * up[i]), silu being `g / (1 + exp(-g))`.
// ---------------------------------------------------------------------------
extern "C" __global__ void swiglu(
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = bf16_to_f32(gate[i]);
    float a = bf16_to_f32(bf16_rne(g / (1.0f + expf(-g))));
    out[i] = bf16_rne(a * bf16_to_f32(up[i]));
}

// ---------------------------------------------------------------------------
// One head row of Q or K, as the reference's `RMSNorm` and complex rotation
// compute it:
//
//     y = bf16(x * rsqrt(mean(x^2) + eps))
//     n = bf16(y * w)
//     out = bf16(n_pair * (cos + i sin))     the rotation in f32
//
// One warp per (token, head) row of 128: each lane owns four consecutive
// elements, two complex pairs, so the row's sum of squares is a warp
// reduction and the rotation never crosses lanes. `head_dim` is fixed at 128
// by that split; the launcher refuses anything else.
// ---------------------------------------------------------------------------
extern "C" __global__ void head_norm_rope(
    const unsigned short* __restrict__ x,  // [seq * heads, 128]
    const float* __restrict__ weight,      // [128]
    const float* __restrict__ freqs,       // [seq, 128] as (cos, sin) pairs
    unsigned short* __restrict__ out,      // [seq * heads, 128]
    unsigned int seq,
    unsigned int heads,
    float eps)
{
    unsigned int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int lane = threadIdx.x & 31;
    if (warp_global >= seq * heads) return;
    unsigned int s = warp_global / heads;

    unsigned long long base = (unsigned long long)warp_global * 128 + lane * 4;
    uint2 packed_in = *(const uint2*)(x + base);
    float v0 = __uint_as_float(packed_in.x << 16);
    float v1 = __uint_as_float(packed_in.x & 0xffff0000u);
    float v2 = __uint_as_float(packed_in.y << 16);
    float v3 = __uint_as_float(packed_in.y & 0xffff0000u);
    float4 w = *(const float4*)(weight + lane * 4);
    float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    for (unsigned int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    float rms = rsqrtf(sq / 128.0f + eps);
    float n0 = bf16_to_f32(bf16_rne(bf16_to_f32(bf16_rne(v0 * rms)) * w.x));
    float n1 = bf16_to_f32(bf16_rne(bf16_to_f32(bf16_rne(v1 * rms)) * w.y));
    float n2 = bf16_to_f32(bf16_rne(bf16_to_f32(bf16_rne(v2 * rms)) * w.z));
    float n3 = bf16_to_f32(bf16_rne(bf16_to_f32(bf16_rne(v3 * rms)) * w.w));

    float4 f = *(const float4*)(freqs + (unsigned long long)s * 128 + lane * 4);
    uint2 packed;
    packed.x = (unsigned int)bf16_rne(n0 * f.x - n1 * f.y)
        | ((unsigned int)bf16_rne(n0 * f.y + n1 * f.x) << 16);
    packed.y = (unsigned int)bf16_rne(n2 * f.z - n3 * f.w)
        | ((unsigned int)bf16_rne(n2 * f.w + n3 * f.z) << 16);
    *(uint2*)(out + base) = packed;
}
