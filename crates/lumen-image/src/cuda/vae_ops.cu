// Kernels the VAE decoder's GPU forward needs.
//
// The decoder runs in bf16, as the reference does when loaded in bf16: every
// activation is stored as bf16, and each op computes in f32 and rounds its
// result to bf16 exactly where the reference's bf16 op does (B below is round
// to nearest even). The four surprises in `crate::vae` are restated in
// `vae_gpu.rs`; the kernels need to know that the convolutions are 2-D (they
// run in `vae_conv.cu`), that the norm is `F.normalize` rather than an epsilon
// RMS, and that the DupUp3D shortcut is parameter-free.
//
// Written here:
//   - copy_rows              a run of rows between two planes (banding)
//   - nearest_2x             nearest-exact 2x upsampling
//   - dup_up_first_chunk     the parameter-free DupUp3D shortcut, added
//   - rms_norm_channels      F.normalize over the channel axis, times sqrt(C)
//   - silu_inplace           SiLU
//   - add_inplace            dst = B(dst + src)
//   - widen_bf16, round_bf16 the attention's f32 operands and its output
//   - softmax_rows_f32       row softmax of the attention scores, in place
//
// NVRTC-compatible: no includes, extern "C" linkage.
//
// Every shape here is an `NCHW` activation: the channel axis is *not* innermost,
// and a kernel that reads it as though it were would still produce plausible
// numbers. The column of each spatial position is the innermost axis, which is
// what the per-element index splits below assume.

typedef unsigned short bf16_bits;

__device__ __forceinline__ float bf16_f32(bf16_bits b)
{
    return __uint_as_float((unsigned int)b << 16);
}

// Round to the nearest bf16, ties to even, as torch's float -> bfloat16 does.
__device__ __forceinline__ bf16_bits f32_bf16(float f)
{
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_bits)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_bits)(u >> 16);
}

// ---------------------------------------------------------------------------
// A run of rows copied between two planes of the same width:
//
//     dst[o, dst_row0 + r, :] = src[o, src_row0 + r, :]    for r < rows
//
// for every plane `o` of `NC`. A banded decode uses it to cut a band of rows
// out of the decoder's input and to place a band's finished rows into the
// image. Each element is a copy, so no arithmetic is involved.
// ---------------------------------------------------------------------------
extern "C" __global__ void copy_rows(
    const bf16_bits* __restrict__ src,   // [NC, src_h, W]
    bf16_bits* __restrict__ dst,         // [NC, dst_h, W]
    unsigned int NC,
    unsigned int W,
    unsigned int src_h,
    unsigned int src_row0,
    unsigned int dst_h,
    unsigned int dst_row0,
    unsigned int rows)
{
    unsigned long long total = (unsigned long long)NC * rows * W;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int col = (unsigned int)(i % W);
    unsigned long long rem = i / W;
    unsigned int r = (unsigned int)(rem % rows);
    unsigned long long o = rem / rows;
    dst[(o * dst_h + dst_row0 + r) * W + col] = src[(o * src_h + src_row0 + r) * W + col];
}

// ---------------------------------------------------------------------------
// `QwenImage21Upsample(scale_factor=(2, 2), mode="nearest-exact")`.
//
// For an exact integer scale of 2, nearest-exact maps output index `i` to
// `floor((i + 0.5) / 2)`, which equals `i / 2` for every non-negative `i`, so
// this agrees with plain nearest. Each output element is an independent copy of
// one input element, so no accumulation order arises.
//
// `NC` is `N * C`: batch and channel are both just "plane" indices here, so one
// counter covers them and the batch is not a separate argument.
// ---------------------------------------------------------------------------
extern "C" __global__ void nearest_2x(
    const bf16_bits* __restrict__ x,     // [N, C, H, W]
    bf16_bits* __restrict__ out,         // [N, C, 2H, 2W]
    unsigned int NC,
    unsigned int H,
    unsigned int W)
{
    unsigned int IW = 2 * W;
    unsigned long long total = (unsigned long long)NC * (2 * H) * IW;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int col = (unsigned int)(i % IW);
    unsigned int rem = (unsigned int)(i / IW);
    unsigned int y = rem % (2 * H);
    unsigned int o = rem / (2 * H);
    unsigned int src = (y / 2) * W + col / 2;

    out[i] = x[(unsigned long long)o * H * W + src];
}

// ---------------------------------------------------------------------------
// `QwenImage21DupUp3D.forward(x, first_chunk=True)` for a single input frame.
//
// The reference repeat-interleaves the channel axis by `repeats`, views the
// result as `[B, out_c, factor_t, factor_s, factor_s, T, H, W]`, permutes to
// `[B, out_c, T, factor_t, H, factor_s, W, factor_s]` and merges each pair:
//
//     out[b, o, t*factor_t + ft, h*factor_s + fs1, w*factor_s + fs2]
//         == x[b, (o*factor + ft*factor_s^2 + fs1*factor_s + fs2) / repeats, t, h, w]
//
// `first_chunk=True` then keeps `x[:, :, factor_t - 1:]`. With one frame the
// temporal axis has exactly `factor_t` entries, so the slice keeps exactly one
// of them — index `factor_t - 1` — which is the `ft = factor_t - 1` below.
//
// `repeats` is passed rather than derived, matching the reference: `factor` is
// `factor_t * factor_s^2`, and a transposed source channel index would be a
// silent wrong answer, so the quantity the integer division needs is explicit.
// So is the kernel's own `factor`, which is built from `factor_t` and
// `factor_s` rather than from `IC` and `OC`, because only those two determine
// the permute the reference performs.
//
// The result is *added* into `out`, `out = B(out + shortcut)`, because the
// shortcut is added to the block's output, and a separate output buffer would
// be a second copy of the largest activation in the decoder for no gain. One
// thread per output element, each reading exactly one input element.
// ---------------------------------------------------------------------------
extern "C" __global__ void dup_up_first_chunk(
    const bf16_bits* __restrict__ x,     // [N, IC, H, W]
    bf16_bits* __restrict__ out,         // [N, OC, H*factor_s, W*factor_s], added to
    unsigned int N,
    unsigned int IC,
    unsigned int H,
    unsigned int W,
    unsigned int OC,
    unsigned int factor_t,
    unsigned int factor_s,
    unsigned int repeats)
{
    unsigned int fs = factor_s;
    unsigned int ft = factor_t - 1;
    unsigned int factor = factor_t * fs * fs;
    unsigned int OW = W * fs;
    unsigned int OH = H * fs;
    unsigned long long total = (unsigned long long)N * OC * OH * OW;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int col = (unsigned int)(i % OW);
    unsigned int rem = (unsigned int)(i / OW);
    unsigned int oy = rem % OH;
    unsigned int rem2 = rem / OH;
    unsigned int o = rem2 % OC;
    unsigned int b = rem2 / OC;

    unsigned int fs1 = oy % fs;
    unsigned int y = oy / fs;
    unsigned int fs2 = col % fs;
    unsigned int xi = col / fs;

    unsigned int j = o * factor + ft * fs * fs + fs1 * fs + fs2;
    unsigned int src_c = j / repeats;
    out[i] = f32_bf16(bf16_f32(out[i]) + bf16_f32(x[((unsigned long long)(b * IC + src_c)) * H * W + y * W + xi]));
}

// ---------------------------------------------------------------------------
// `QwenImage21RMS_norm` in bf16, with the reference's three roundings:
//
//     n   = B(x / max(||x||_2, 1e-12))     F.normalize on x widened to f32
//     s   = B(n * scale)                   scale = sqrt(C) as an f32 scalar
//     out = B(B(s * gamma) + 0.0)          `+ self.bias`, the float 0.0
//
// where the norm runs over the channel axis at a fixed spatial position. The
// `+ 0.0` changes nothing but the sign of a zero, which it makes positive.
//
// This is `F.normalize`, *not* `x / sqrt(mean(x^2) + eps)`: the guard is a clamp
// on the norm and there is no epsilon inside the square root, so the two
// disagree sharply for a small-magnitude channel vector.
//
// One thread per spatial position of each batch item. The thread reduces the
// channel column once, walking `cc` in ascending order (the order the
// reference's `sum_sq` loop uses), then writes every channel of that position.
// Adjacent threads hold adjacent positions, so each channel read and write is a
// coalesced row segment.
//
// `rows` is `N * C`: a thread index splits into a spatial position `p` and a
// batch item `n`, and the batch's channel-0 row is `n * C`, which is where the
// reduction starts.
// ---------------------------------------------------------------------------
extern "C" __global__ void rms_norm_channels(
    const bf16_bits* __restrict__ x,       // [rows, HW]
    const bf16_bits* __restrict__ gamma,   // [C]
    bf16_bits* __restrict__ out,           // [rows, HW]
    unsigned int C,
    unsigned int HW,
    unsigned int rows,
    float eps,
    float scale)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long positions = (unsigned long long)(rows / C) * HW;
    if (i >= positions) return;

    unsigned int p = (unsigned int)(i % HW);
    unsigned int n = (unsigned int)(i / HW);

    const bf16_bits* xp = x + (unsigned long long)n * C * HW + p;
    bf16_bits* op = out + (unsigned long long)n * C * HW + p;
    float sum_sq = 0.0f;
    for (unsigned int cc = 0; cc < C; cc++) {
        float vv = bf16_f32(xp[(unsigned long long)cc * HW]);
        sum_sq += vv * vv;
    }
    float denom = sqrtf(sum_sq);
    if (denom < eps) denom = eps;

    for (unsigned int cc = 0; cc < C; cc++) {
        unsigned long long off = (unsigned long long)cc * HW;
        float nv = bf16_f32(f32_bf16(bf16_f32(xp[off]) / denom));
        float sv = bf16_f32(f32_bf16(nv * scale));
        float gv = bf16_f32(f32_bf16(sv * bf16_f32(gamma[cc])));
        op[off] = f32_bf16(gv + 0.0f);
    }
}

// ---------------------------------------------------------------------------
// SiLU, `B(x / (1 + exp(-x)))` in f32, the expression torch's kernel uses.
// ---------------------------------------------------------------------------
extern "C" __global__ void silu_inplace(
    bf16_bits* __restrict__ x,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf16_f32(x[i]);
    x[i] = f32_bf16(v / (1.0f + expf(-v)));
}

// ---------------------------------------------------------------------------
// dst = B(dst + src), elementwise.
// ---------------------------------------------------------------------------
extern "C" __global__ void add_inplace(
    bf16_bits* __restrict__ dst,
    const bf16_bits* __restrict__ src,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = f32_bf16(bf16_f32(dst[i]) + bf16_f32(src[i]));
}

// ---------------------------------------------------------------------------
// The attention's operands widened to f32, and its f32 context rounded back.
// ---------------------------------------------------------------------------
extern "C" __global__ void widen_bf16(
    const bf16_bits* __restrict__ x,
    float* __restrict__ out,
    unsigned long long n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf16_f32(x[i]);
}

extern "C" __global__ void round_bf16(
    const float* __restrict__ x,
    bf16_bits* __restrict__ out,
    unsigned long long n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_bf16(x[i]);
}

// ---------------------------------------------------------------------------
// Softmax over each row of a `[rows, cols]` f32 matrix, in place.
//
// This is the reference's per-position softmax of the attention block: the
// max over the row, `exp(score - max)`, then division by the sum. The two
// matrix products around it run on cuBLAS (`vae_gpu.rs`), so the scores arrive
// already scaled by `1 / sqrt(C)`.
//
// One block per row. The max and the sum are block reductions: every thread
// folds its own stride of the row, the partials meet in shared memory, and
// every thread reads the same final value. All 32 lanes of each warp take part
// in every shuffle, and the block never returns early, so every barrier is
// reached by the whole block.
// ---------------------------------------------------------------------------
extern "C" __global__ void softmax_rows_f32(
    float* __restrict__ x,   // [rows, cols]
    unsigned int cols)
{
    __shared__ float warp_part[32];
    __shared__ float shared_value;
    float* row = x + (unsigned long long)blockIdx.x * cols;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    unsigned int warps = (blockDim.x + 31) >> 5;

    float max = __int_as_float(0xff800000);
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = row[j];
        if (v > max) max = v;
    }
    for (unsigned int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, max, o);
        if (other > max) max = other;
    }
    if (lane == 0) warp_part[warp] = max;
    __syncthreads();
    if (tid == 0) {
        float m = warp_part[0];
        for (unsigned int w = 1; w < warps; w++) {
            if (warp_part[w] > m) m = warp_part[w];
        }
        shared_value = m;
    }
    __syncthreads();
    max = shared_value;

    float sum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float e = expf(row[j] - max);
        row[j] = e;
        sum += e;
    }
    for (unsigned int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    __syncthreads();
    if (lane == 0) warp_part[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (unsigned int w = 0; w < warps; w++) t += warp_part[w];
        shared_value = t;
    }
    __syncthreads();
    float inv = 1.0f / shared_value;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        row[j] *= inv;
    }
}
