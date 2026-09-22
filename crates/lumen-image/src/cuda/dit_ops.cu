// Kernels the DiT's GPU forward needs that no existing source provides.
//
// `gemm_f32_bias` in image_ops.cu serves the f32 weights.
//
// Written here:
//   - gemm_16bit          the same GEMM for BF16/F16 weights
//   - zero_center_rmsnorm the text projection's `scale - 1` RMSNorm
//   - gelu_tanh_inplace   nn.GELU(approximate="tanh")
//   - add_gated_gather    x + tanh(modulation[row_of(token)]) * y
//   - pack_rows           the joint sequence, gathered from the text and
//                         image projections
//   - layernorm_scale_bf16 LayerNorm, the AdaLN scale and the bf16 rounding a
//                         projection's input needs, in one pass
//   - swiglu_bf16         silu(gate) * up, rounded to bf16 for the down
//                         projection
//   - head_norm_rope_bf16 the per-head norm, the rotation and the attention
//                         operand's bf16 truncation for Q and K
//
// The `_bf16` kernels write their result in the format their consumer reads
// (a bf16 GEMM, or the attention kernel's operands), so the activation is
// passed over once instead of written in f32 and converted afterwards.
//
// The modulation is gathered per token (`add_gated_gather`,
// `layernorm_scale_bf16`), not broadcast per sequence: with
// `causal_condition` the target-image tokens read the `t = 0` row of a two-row
// modulation while every other token reads the sampled-timestep row.
// `scale_one_plus` broadcasts one row over all tokens and cannot express that,
// so the row selection lives in the kernels.
//
// NVRTC-compatible: no includes, extern "C" linkage.

#define BM 32
#define BN 32
#define BK 32

// ---------------------------------------------------------------------------
// GEMM against 16-bit weights: C[M,N] = A[M,K] * W^T[N,K], W stored as f16 or
// bf16 bits.
//
// Same tiling, same shared-memory staging and same per-element accumulation
// order as gemm_f32_bias: each weight element is widened to f32 on the way into
// the tile and the inner product then runs on the identical f32 values. So a
// 16-bit weight gives exactly the result its f32 widening would, which is what
// keeps this comparable to the CPU reference rather than merely close to it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float widen16(unsigned short h, unsigned int is_bf16)
{
    if (is_bf16 != 0) {
        // BF16 is the upper half of an IEEE f32.
        union { unsigned int u; float f; } cv;
        cv.u = ((unsigned int)h) << 16;
        return cv.f;
    }
    // f16 -> f32 is a single hardware convert (SM 53+).
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

extern "C" __global__ void gemm_16bit(
    const float* __restrict__ A,            // [M, K]
    const unsigned short* __restrict__ W,   // [N, K] f16 or bf16 bits
    float* __restrict__ C,                  // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int is_bf16)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * BM + ty;
    unsigned int col = blockIdx.x * BN + tx;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];

    float sum = 0.0f;
    unsigned int k_tiles = (K + BK - 1) / BK;
    for (unsigned int t = 0; t < k_tiles; t++) {
        unsigned int a_col = t * BK + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[(unsigned long long)row * K + a_col] : 0.0f;
        unsigned int b_col = t * BK + ty;
        Bs[tx][ty] = (col < N && b_col < K)
            ? widen16(W[(unsigned long long)col * K + b_col], is_bf16)
            : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < BK; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[(unsigned long long)row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Zero-centred RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * (w[i] + 1)
//
// QwenImage21ZeroCenterRMSNorm stores `scale - 1`, so the effective scale is
// formed here rather than in a pre-pass over the weights: adding one on the fly
// is the same f32 expression the CPU reference evaluates, and it leaves the
// checkpoint bytes on the device untouched.
// One block per row; blockDim is 256, matching the shared array.
// ---------------------------------------------------------------------------
extern "C" __global__ void zero_center_rmsnorm(
    const float* __restrict__ x,       // [rows, dim]
    const float* __restrict__ weight,  // [dim], stored as (scale - 1)
    float* __restrict__ out,           // [rows, dim]
    unsigned int dim,
    float eps)
{
    unsigned int row = blockIdx.x;
    const float* r = x + (unsigned long long)row * dim;
    float* o = out + (unsigned long long)row * dim;

    // Strided accumulation then a tree reduction, in a fixed order.
    __shared__ float s_sq[256];
    float sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = r[i];
        sq += v * v;
    }
    s_sq[threadIdx.x] = sq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(s_sq[0] / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        o[i] = r[i] * inv * (weight[i] + 1.0f);
    }
}

// ---------------------------------------------------------------------------
// GELU, tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
// In place, one thread per element.
// ---------------------------------------------------------------------------
extern "C" __global__ void gelu_tanh_inplace(
    float* __restrict__ x,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978846f * (v + 0.044715f * v * v * v)));
}

// ---------------------------------------------------------------------------
// out[t, c] = x[t, c] + tanh(modu[mod_row[t], col_off + c]) * y[t, c]
//
// `mod_stride` is the modulation row width in floats: `4 * cols` for the
// shared block modulation.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_gated_gather(
    const float* __restrict__ x,       // [rows, cols]
    const float* __restrict__ y,       // [rows, cols]
    const float* __restrict__ modu,    // [nmod, mod_stride]
    const int* __restrict__ mod_row,   // [rows]
    float* __restrict__ out,           // [rows, cols]
    unsigned int rows,
    unsigned int cols,
    unsigned int col_off,
    unsigned int mod_stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    const float* m = modu + (unsigned long long)mod_row[r] * mod_stride;
    out[i] = x[i] + tanhf(m[col_off + c]) * y[i];
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
    const float* __restrict__ txt,     // [text_rows, cols]
    const float* __restrict__ img,     // [image_rows, cols]
    const int* __restrict__ source,    // [rows]
    float* __restrict__ out,           // [rows, cols]
    unsigned int rows,
    unsigned int cols)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    int s = source[r];
    const float* src = (s >= 0)
        ? txt + (unsigned long long)s * cols
        : img + (unsigned long long)(-(long long)s - 1) * cols;
    out[i] = src[c];
}

// ---------------------------------------------------------------------------
// f32 -> bf16, round to nearest even, the same encoding as image_ops.cu's
// `f32_to_bf16_rne`; each NVRTC module carries its own copy.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// out[t, c] = bf16(layernorm(x[t])[c] * (1 + modu[mod_row[t], col_off + c]))
//
// One block per row. The LayerNorm is image_ops.cu's `layernorm_noaffine`
// (mean first, then the squared deviations, each reduced in a fixed order).
// ---------------------------------------------------------------------------
extern "C" __global__ void layernorm_scale_bf16(
    const float* __restrict__ x,           // [rows, dim]
    const float* __restrict__ modu,        // [nmod, mod_stride]
    const int* __restrict__ mod_row,       // [rows]
    unsigned short* __restrict__ out,      // [rows, dim]
    unsigned int dim,
    unsigned int col_off,
    unsigned int mod_stride,
    float eps)
{
    unsigned int row = blockIdx.x;
    const float* r = x + (unsigned long long)row * dim;
    unsigned short* o = out + (unsigned long long)row * dim;
    const float* m = modu + (unsigned long long)mod_row[row] * mod_stride + col_off;

    __shared__ float s_part[256];
    float n = (float)dim;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += r[i];
    }
    s_part[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = s_part[0] / n;
    __syncthreads();

    float sq = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = r[i] - mean;
        sq += d * d;
    }
    s_part[threadIdx.x] = sq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(s_part[0] / n + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        o[i] = bf16_rne(((r[i] - mean) * inv) * (1.0f + m[i]));
    }
}

// ---------------------------------------------------------------------------
// out[i] = bf16(silu(gate[i]) * up[i]), silu being `g / (1 + exp(-g))` as in
// lumen-runtime's `swiglu_inplace`.
// ---------------------------------------------------------------------------
extern "C" __global__ void swiglu_bf16(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu_g = g / (1.0f + expf(-g));
    out[i] = bf16_rne(silu_g * up[i]);
}

// ---------------------------------------------------------------------------
// One head row of Q or K, from the projection to the attention operand:
// per-head RMSNorm (`rmsnorm_per_head` with the shared `[head_dim]` weight),
// the complex rotation (`rope_complex`), then the truncating bf16 conversion
// the attention kernel's operands use.
//
// One warp per (token, head) row of 128: each lane owns four consecutive
// elements, two complex pairs, so the row's sum of squares is a warp
// reduction and the rotation never crosses lanes. `head_dim` is fixed at 128
// by that split; the launcher refuses anything else.
// ---------------------------------------------------------------------------
extern "C" __global__ void head_norm_rope_bf16(
    const float* __restrict__ x,           // [seq * heads, 128]
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
    float4 v = *(const float4*)(x + base);
    float4 w = *(const float4*)(weight + lane * 4);
    float sq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    for (unsigned int o = 16; o > 0; o >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    float rms = rsqrtf(sq / 128.0f + eps);
    float n0 = v.x * rms * w.x;
    float n1 = v.y * rms * w.y;
    float n2 = v.z * rms * w.z;
    float n3 = v.w * rms * w.w;

    float4 f = *(const float4*)(freqs + (unsigned long long)s * 128 + lane * 4);
    float r0 = n0 * f.x - n1 * f.y;
    float i0 = n0 * f.y + n1 * f.x;
    float r1 = n2 * f.z - n3 * f.w;
    float i1 = n2 * f.w + n3 * f.z;

    uint2 packed;
    packed.x = (__float_as_uint(r0) >> 16) | (__float_as_uint(i0) & 0xffff0000u);
    packed.y = (__float_as_uint(r1) >> 16) | (__float_as_uint(i1) & 0xffff0000u);
    *(uint2*)(out + base) = packed;
}
