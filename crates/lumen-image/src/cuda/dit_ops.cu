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
//   - add_gated_layernorm_scale
//                         x + tanh(gate) * y, the residual update, then the
//                         next layernorm_scale of the updated row
//   - swiglu              silu(gate) * up, the MLP's inner activation
//   - head_norm_rope      the per-head RMSNorm and the rotation of Q or K
//
// The modulation reaches the AdaLN kernels already in the form they multiply
// by: `bf16(1 + scale)` and `bf16(tanh(gate))`, rounded once per forward on the
// host, as the reference rounds each of those expressions before it
// multiplies. It is gathered per token, not broadcast per sequence: with
// `causal_condition` the target-image tokens read the sampled-timestep row of a
// two-row modulation while every other token reads the `t = 0` row.
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
// The residual update followed by the next norm, one block per row:
//
//     x[t, c]   = bf16(x[t, c] + bf16(tanh_gate[mod_row[t], gate_off + c] * y[t, c]))
//     out[t, c] = layernorm_scale of the updated row, reading
//                 one_plus[mod_row[t], scale_off + c]
//
// The same arithmetic as the residual add and `layernorm_scale` run one after
// the other. The row is read and written eight elements (16 bytes) at a time
// and the updated values are kept in shared memory, where the norm's sums
// take them in `layernorm_scale`'s order: thread i adds elements i,
// i + ROW_THREADS, ... before the block reduction. `dim`, both offsets and
// both strides are multiples of eight and `dim` is at most ADD_NORM_MAX_DIM;
// the launcher refuses anything else.
// ---------------------------------------------------------------------------
#define ADD_NORM_MAX_DIM 4096

__device__ __forceinline__ void unpack8(uint4 v, float* f)
{
    unsigned int w[4] = {v.x, v.y, v.z, v.w};
    for (unsigned int k = 0; k < 4; k++) {
        f[2 * k] = __uint_as_float(w[k] << 16);
        f[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
    }
}

__device__ __forceinline__ uint4 pack8(const unsigned short* h)
{
    return make_uint4(
        (unsigned int)h[0] | ((unsigned int)h[1] << 16),
        (unsigned int)h[2] | ((unsigned int)h[3] << 16),
        (unsigned int)h[4] | ((unsigned int)h[5] << 16),
        (unsigned int)h[6] | ((unsigned int)h[7] << 16));
}

extern "C" __global__ void add_gated_layernorm_scale(
    unsigned short* __restrict__ x,               // [rows, dim], updated
    const unsigned short* __restrict__ y,         // [rows, dim]
    const unsigned short* __restrict__ tanh_gate, // [nmod, gate_stride]
    const unsigned short* __restrict__ one_plus,  // [nmod, scale_stride]
    const int* __restrict__ mod_row,              // [rows]
    unsigned short* __restrict__ out,             // [rows, dim]
    unsigned int dim,
    unsigned int gate_off,
    unsigned int gate_stride,
    unsigned int scale_off,
    unsigned int scale_stride,
    float eps)
{
    __shared__ float s_part[ROW_THREADS];
    __shared__ __align__(16) float s_row[ADD_NORM_MAX_DIM];
    const unsigned long long row = (unsigned long long)blockIdx.x * dim;
    const unsigned long long m = (unsigned long long)mod_row[blockIdx.x];
    const unsigned short* g = tanh_gate + m * gate_stride + gate_off;
    const unsigned short* s = one_plus + m * scale_stride + scale_off;
    float n = (float)dim;

    for (unsigned int c = threadIdx.x * 8; c < dim; c += ROW_THREADS * 8) {
        float xv[8], yv[8], gv[8];
        unpack8(*(const uint4*)(x + row + c), xv);
        unpack8(*(const uint4*)(y + row + c), yv);
        unpack8(*(const uint4*)(g + c), gv);
        unsigned short h[8];
        for (unsigned int e = 0; e < 8; e++) {
            float gy = bf16_to_f32(bf16_rne(gv[e] * yv[e]));
            h[e] = bf16_rne(xv[e] + gy);
        }
        *(uint4*)(x + row + c) = pack8(h);
        *(float4*)(s_row + c) = make_float4(bf16_to_f32(h[0]), bf16_to_f32(h[1]), bf16_to_f32(h[2]), bf16_to_f32(h[3]));
        *(float4*)(s_row + c + 4) = make_float4(bf16_to_f32(h[4]), bf16_to_f32(h[5]), bf16_to_f32(h[6]), bf16_to_f32(h[7]));
    }
    __syncthreads();

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        sum += s_row[i];
    }
    float mean = block_sum(sum, s_part) / n;

    float sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += ROW_THREADS) {
        float d = s_row[i] - mean;
        sq += d * d;
    }
    float inv = rsqrtf(block_sum(sq, s_part) / n + eps);

    for (unsigned int c = threadIdx.x * 8; c < dim; c += ROW_THREADS * 8) {
        float sv[8];
        unpack8(*(const uint4*)(s + c), sv);
        float4 lo = *(const float4*)(s_row + c);
        float4 hi = *(const float4*)(s_row + c + 4);
        float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        unsigned short h[8];
        for (unsigned int e = 0; e < 8; e++) {
            float normed = bf16_to_f32(bf16_rne((v[e] - mean) * inv));
            h[e] = bf16_rne(normed * sv[e]);
        }
        *(uint4*)(out + row + c) = pack8(h);
    }
}

// ---------------------------------------------------------------------------
// out[i] = bf16(bf16(silu(gate[i])) * up[i]), silu being `g / (1 + exp(-g))`.
// Each thread takes eight consecutive elements, loaded and stored 16 bytes at
// a time; a tail shorter than eight goes element by element.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned short swiglu_one(unsigned short gate, unsigned short up)
{
    float g = bf16_to_f32(gate);
    float a = bf16_to_f32(bf16_rne(g / (1.0f + expf(-g))));
    return bf16_rne(a * bf16_to_f32(up));
}

extern "C" __global__ void swiglu(
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned long long base = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base >= n) return;
    if (base + 8 <= n) {
        uint4 gv = *(const uint4*)(gate + base);
        uint4 uv = *(const uint4*)(up + base);
        unsigned int* gw = (unsigned int*)&gv;
        unsigned int* uw = (unsigned int*)&uv;
        uint4 ov;
        unsigned int* ow = (unsigned int*)&ov;
        for (unsigned int w = 0; w < 4; w++) {
            unsigned short lo = swiglu_one((unsigned short)(gw[w] & 0xffffu), (unsigned short)(uw[w] & 0xffffu));
            unsigned short hi = swiglu_one((unsigned short)(gw[w] >> 16), (unsigned short)(uw[w] >> 16));
            ow[w] = (unsigned int)lo | ((unsigned int)hi << 16);
        }
        *(uint4*)(out + base) = ov;
    } else {
        for (unsigned long long i = base; i < n; i++) {
            out[i] = swiglu_one(gate[i], up[i]);
        }
    }
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
