// Kernels the image pipeline needs.
//
// Written here:
//   - gemm_f32_bias        linear layers with a bias
//   - f32_to_bf16_trunc    f32 -> bf16 by truncation, for the attention operands
//   - f32_to_bf16_bits     f32 -> bf16 rounded to nearest even, for the linears
//   - mask_softmax_rows    the block-causal mask and row softmax of the scores,
//                          for the unfused attention `cuda-ops-check` compares
//                          the fused kernel against
//   - layernorm_noaffine   LayerNorm with no affine params
//   - scale_one_plus       x * (1 + scale)
//   - rope_complex         the DiT's 3-axis RoPE, applied as a complex multiply
//   - mrope_interleaved    the text encoder's interleaved mRoPE
//   - attn_block_causal    text causal + a bidirectional target-image block
// The last four are the per-op references `cuda-ops-check` runs; the DiT
// forward itself runs the fused kernels in `dit_ops.cu` and `flash_attn.cu`.
// `mrope_interleaved` is launched by nothing: the text tower rotates with
// `rope_half_interleaved`.
//
// NVRTC-compatible: no includes, extern "C" linkage.

// ---------------------------------------------------------------------------
// Linear with bias: C[M,N] = A[M,K] * W^T[N,K] + bias[N]
// Same tiling as lumen-runtime's gemm_f32; the bias is added at the write.
// ---------------------------------------------------------------------------
#define BM 32
#define BN 32
#define BK 32

extern "C" __global__ void gemm_f32_bias(
    const float* __restrict__ A,      // [M, K]
    const float* __restrict__ W,      // [N, K]
    const float* __restrict__ bias,   // [N], read only when has_bias is 1
    float* __restrict__ C,            // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int has_bias)
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
        Bs[tx][ty] = (col < N && b_col < K) ? W[(unsigned long long)col * K + b_col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < BK; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        if (has_bias != 0) {
            sum += bias[col];
        }
        C[(unsigned long long)row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm with no affine parameters, one block per row.
//
// The DiT's `img_norm1`/`img_norm2` are `nn.LayerNorm(dim, elementwise_affine=
// False)`, so mean and variance are over the row and nothing is scaled after.
// ---------------------------------------------------------------------------
extern "C" __global__ void layernorm_noaffine(
    const float* __restrict__ x,   // [rows, dim]
    float* __restrict__ out,       // [rows, dim]
    unsigned int rows,
    unsigned int dim,
    float eps)
{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float* r = x + (unsigned long long)row * dim;
    float* o = out + (unsigned long long)row * dim;

    // The mean first, then the squared deviations from it, as the CPU
    // reference (`layer_norm_rows`) does. `E[x²] - mean²` cancels for a row
    // whose values sit far from zero relative to their spread and can even go
    // negative; the two-pass form cannot. Each reduction runs in a fixed order
    // so the result is reproducible.
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
    float var = s_part[0] / n;
    float inv = rsqrtf(var + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        o[i] = (r[i] - mean) * inv;
    }
}

// ---------------------------------------------------------------------------
// x * (1 + scale), broadcasting `scale` over rows.
// The DiT's modulation is scale-only everywhere it is applied.
// ---------------------------------------------------------------------------
extern "C" __global__ void scale_one_plus(
    const float* __restrict__ x,      // [rows, dim]
    const float* __restrict__ scale,  // [dim]
    float* __restrict__ out,          // [rows, dim]
    unsigned int total,
    unsigned int dim)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int col = i % dim;
    out[i] = x[i] * (1.0f + scale[col]);
}

// ---------------------------------------------------------------------------
// The DiT's 3-axis RoPE, applied as a complex multiply.
//
// `freqs` is [seq, head_dim] holding the interleaved (cos, sin) pairs for all
// three axes concatenated along the head dimension, exactly as the reference
// builds it. Each adjacent pair of head channels is one complex number.
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_complex(
    float* __restrict__ q,                    // [seq, heads * head_dim]
    const float* __restrict__ freqs,          // [seq, head_dim] as (cos, sin) pairs
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim)
{
    // One thread per complex pair per head.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pairs = head_dim / 2;
    unsigned int per_seq = heads * pairs;
    if (idx >= seq * per_seq) return;
    unsigned int s = idx / per_seq;
    unsigned int rem = idx % per_seq;
    unsigned int h = rem / pairs;
    unsigned int p = rem % pairs;

    unsigned long long base = (unsigned long long)s * heads * head_dim + (unsigned long long)h * head_dim;
    float xr = q[base + 2 * p];
    float xi = q[base + 2 * p + 1];
    // The frequency table is shared across heads within a position.
    float cr = freqs[(unsigned long long)s * head_dim + 2 * p];
    float ci = freqs[(unsigned long long)s * head_dim + 2 * p + 1];
    q[base + 2 * p] = xr * cr - xi * ci;
    q[base + 2 * p + 1] = xr * ci + xi * cr;
}

// ---------------------------------------------------------------------------
// The text encoder's interleaved mRoPE.
//
// Three position rows (T, H, W) are combined into one (cos, sin) table per
// position: the table is built from row T and, for the frequency slots that
// belong to H and W, overwritten by those rows' values at stride 3. This
// mirrors `recomposition_frequencies` in the reference; it is not the same
// layout as the DiT's RoPE and the two are not interchangeable.
// ---------------------------------------------------------------------------
extern "C" __global__ void mrope_interleaved(
    float* __restrict__ q,                  // [seq, heads * head_dim]
    const float* __restrict__ freqs_thw,    // [3, seq, half] real frequencies
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int section_h,
    unsigned int section_w)
{
    unsigned int half = head_dim / 2;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq * half) return;
    unsigned int s = idx / half;
    unsigned int p = idx % half;

    // Which row this slot takes: everything from T, except the interleaved
    // slots belonging to H (offset 1) and W (offset 2).
    unsigned int row = 0;
    unsigned int oh = section_h * 3;
    unsigned int ow = section_w * 3;
    if (p >= 1 && p < oh && ((p - 1) % 3) == 0) row = 1;
    if (p >= 2 && p < ow && ((p - 2) % 3) == 0) row = 2;

    float ang = freqs_thw[(unsigned long long)row * seq * half + (unsigned long long)s * half + p];
    float cr = cosf(ang);
    float ci = sinf(ang);
    for (unsigned int h = 0; h < heads; h++) {
        unsigned long long base =
            (unsigned long long)s * heads * head_dim + (unsigned long long)h * head_dim;
        float xr = q[base + 2 * p];
        float xi = q[base + 2 * p + 1];
        q[base + 2 * p] = xr * cr - xi * ci;
        q[base + 2 * p + 1] = xr * ci + xi * cr;
    }
}

// ---------------------------------------------------------------------------
// Attention with the DiT's block-causal mask.
//
// `image_id[s]` is -1 at text positions and the index of the image block a
// token belongs to otherwise. A query may attend to a key when
//     q_index >= kv_index            (the sequence is causal), or
//     same image block               (each image block is internally
//                                     bidirectional).
// `key_valid[s]` is 0 at right-padded positions, which are never attended to.
//
// One block per (query position, head). Head dims up to 256 are staged in
// shared memory; the score row is computed inline.
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_block_causal(
    const float* __restrict__ q,            // [seq, heads, head_dim]
    const float* __restrict__ k,            // [seq, heads, head_dim]
    const float* __restrict__ v,            // [seq, heads, head_dim]
    float* __restrict__ out,                // [seq, heads, head_dim]
    const int* __restrict__ image_id,       // [seq]
    const int* __restrict__ key_valid,      // [seq] or null
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim)
{
    unsigned int qi = blockIdx.x;
    unsigned int h = blockIdx.y;
    unsigned int tid = threadIdx.x;
    if (qi >= seq || h >= heads) return;

    extern __shared__ float sh[];      // q row then the running output
    float* qrow = sh;
    float* orow = sh + head_dim;

    unsigned long long qbase = ((unsigned long long)qi * heads + h) * head_dim;
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        qrow[i] = q[qbase + i];
        orow[i] = 0.0f;
    }
    __syncthreads();

    // Two passes: the max for stability, then the weighted sum. Both walk keys
    // in ascending order, so the accumulation order is fixed.
    float scale = rsqrtf((float)head_dim);
    float max_score = -1e30f;
    for (unsigned int kj = 0; kj < seq; kj++) {
        if (key_valid != 0 && key_valid[kj] == 0) continue;
        bool allowed = (qi >= kj) || (image_id[qi] >= 0 && image_id[qi] == image_id[kj]);
        if (!allowed) continue;
        float dot = 0.0f;
        unsigned long long kbase = ((unsigned long long)kj * heads + h) * head_dim;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += qrow[d] * k[kbase + d];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }
    if (max_score < -1e29f) max_score = 0.0f;

    float denom = 0.0f;
    for (unsigned int kj = 0; kj < seq; kj++) {
        if (key_valid != 0 && key_valid[kj] == 0) continue;
        bool allowed = (qi >= kj) || (image_id[qi] >= 0 && image_id[qi] == image_id[kj]);
        if (!allowed) continue;
        float dot = 0.0f;
        unsigned long long kbase = ((unsigned long long)kj * heads + h) * head_dim;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += qrow[d] * k[kbase + d];
        }
        float w = expf(dot * scale - max_score);
        denom += w;
        unsigned long long vbase = kbase;
        for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
            orow[d] += w * v[vbase + d];
        }
        __syncthreads();
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out[qbase + d] = orow[d] * inv;
    }
}

// ---------------------------------------------------------------------------
// F32 -> BF16 conversion, for feeding cuBLAS GemmEx from our f32 activations.
//
// Round to nearest-even, not truncation: truncation biases every activation
// toward zero by up to one bf16 ulp, and across 32 layers and 40 steps that
// bias compounds into a visible loss. The reference's own bf16 casts round to
// nearest, so this is also what matches it.
//
// NVRTC compiles this module with no headers on a `default` target, so the
// software rounding is used rather than `cvt.rn.bf16.f32`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float val)
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
// Truncating f32 -> bf16, for the attention operands.
//
// The attention operands reproduce the reference more closely when truncated
// than when rounded to nearest, the opposite of the projections' inputs, so
// the two paths keep their own conversion.
// ---------------------------------------------------------------------------
extern "C" __global__ void f32_to_bf16_trunc(
    const float* __restrict__ x,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (unsigned short)(__float_as_uint(x[i]) >> 16);
}

extern "C" __global__ void f32_to_bf16_bits(
    const float* __restrict__ x,
    unsigned short* __restrict__ out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16_rne(x[i]);
}

// ---------------------------------------------------------------------------
// Block-causal softmax over the last axis: f32 attention scores in, bf16
// probabilities out.
//
// One block per (head, query) row of `[heads * seq, seq]`, so the whole row
// fits in shared memory and a single pass suffices: read once, block-reduce
// the max and the sum, write once. The probabilities are written as bf16: the
// PV product that consumes them runs on the tensor cores with bf16 operands,
// which is also the precision the reference computes attention in.
//
// The mask is computed from `text_count` rather than read from a buffer. The
// rule `(q >= kv) or same_image_block` reduces, for one image block that
// follows a text prefix, to: an image query sees every key; a text query at
// position q sees keys [0, q]. That is a comparison per element; a
// materialised per-head mask would be `heads * seq * seq` bytes, half a
// gigabyte per layer at 1024x1024.
//
// The row is staged in dynamic shared memory sized by the launcher, which
// refuses `seq > SOFTMAX_MAX_SEQ` so a bigger resolution fails loudly instead of asking
// for more shared memory than a block can have.
// ---------------------------------------------------------------------------
extern "C" __global__ void mask_softmax_rows(
    const float* __restrict__ scores,
    unsigned short* __restrict__ probs,
    unsigned int seq,
    unsigned int text_count)
{
    extern __shared__ float row[];
    unsigned int r = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float* src = scores + (unsigned long long)r * seq;
    // Only a text query masks anything, and then only keys after itself.
    unsigned int q = r % seq;
    unsigned int limit = (q < text_count) ? (q + 1u) : seq;

    for (unsigned int i = tid; i < seq; i += blockDim.x) {
        row[i] = (i < limit) ? src[i] : -1e30f;
    }
    __syncthreads();

    // Row max: a full-warp reduction, then one slot per warp, then the first
    // warp reduces those. Every shuffle below runs with all 32 lanes active: a
    // reduction guarded down to fewer lanes that still passes mask 0xffffffff
    // is undefined. The max and the sum have their own slots: a warp that
    // finishes its exponentials early must not overwrite a slot another warp
    // is still reading as the max.
    __shared__ float part_max[8];
    __shared__ float part_sum[8];
    float mx = -1e30f;
    for (unsigned int i = tid; i < seq; i += blockDim.x) {
        mx = fmaxf(mx, row[i]);
    }
    for (unsigned int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if ((tid & 31u) == 0u) part_max[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32u) {
        float v = (tid < 8u) ? part_max[tid] : -1e30f;
        for (unsigned int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        }
        if (tid == 0u) part_max[0] = v;
    }
    __syncthreads();
    mx = part_max[0];
    if (mx < -1e29f) mx = 0.0f;

    float sum = 0.0f;
    for (unsigned int i = tid; i < seq; i += blockDim.x) {
        float e = expf(row[i] - mx);
        row[i] = e;
        sum += e;
    }
    for (unsigned int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if ((tid & 31u) == 0u) part_sum[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32u) {
        float v = (tid < 8u) ? part_sum[tid] : 0.0f;
        for (unsigned int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, o);
        }
        if (tid == 0u) part_sum[0] = v;
    }
    __syncthreads();
    float inv = (part_sum[0] > 0.0f) ? (1.0f / part_sum[0]) : 0.0f;

    unsigned short* dst = probs + (unsigned long long)r * seq;
    for (unsigned int i = tid; i < seq; i += blockDim.x) {
        dst[i] = f32_to_bf16_rne(row[i] * inv);
    }
}
