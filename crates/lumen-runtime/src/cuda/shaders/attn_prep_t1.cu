// ==========================================================================
// q35_attn_prep_t1: deinterleave Q/gate + per-head RMSNorm(Q,K) + NeoX RoPE
// in ONE launch, for T=1 decode.
//
// WHY
//
// The full-attention block costs 1.377 ms/token across only 8 layers = 172 us
// per layer, while carrying just 0.26 GB of weights (5% of the model). After
// the QKV projections each layer issues NINE separate commands: deinterleave,
// Q RMSNorm, K RMSNorm, RoPE, K-cache write, V-cache write, attention,
// sigmoid gate, and a device-to-device copy. At the measured ~4.2 us marginal
// launch cost that is ~38 us/layer of pure command overhead before any work.
//
// This collapses the first four into one kernel. The three ops that previously
// needed separate launches only did so because each depends on the whole head
// vector: the RMSNorm reduces over it, and NeoX RoPE pairs element d with
// element d + rotary_dim/2. Giving one CTA one head makes both dependencies
// intra-CTA, so __syncthreads() replaces two kernel boundaries.
//
// Work assignment:
//   CTA [0, n_q)              -> Q head: deinterleave from the fused Q+gate
//                                projection, emit gate, RMSNorm, RoPE
//   CTA [n_q, n_q + n_kv)     -> K head: RMSNorm in place, RoPE
//
// V needs neither norm nor rotation and is left to the cache write.
//
// Numerics: the sum-of-squares uses the same warp-then-block reduction shape
// as `rmsnorm_per_head_inplace`, the same `sqrtf(ss/head_dim + eps)` form, and
// RoPE uses the same `powf(theta, 2d/rot)` frequency and the same pairing as
// `rope_apply_neox`. Values are therefore equivalent to running the four
// kernels in sequence, up to F32 reassociation inside the reduction.
//
// Grid:  (n_q_heads + n_kv_heads, 1, 1)
// Block: (BLOCK, 1, 1)
// Shared: head_dim floats for the head vector + BLOCK/32 for the reduction.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define PREP_BLOCK 256
#define PREP_WARPS (PREP_BLOCK / 32)

__device__ __forceinline__ float warp_reduce_sum_prep(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

extern "C" __global__ void q35_attn_prep_t1(
    const float* __restrict__ qgate,   // [n_q * 2 * head_dim] fused Q+gate
    float* __restrict__ q,             // [n_q * head_dim]  out
    float* __restrict__ gate,          // [n_q * head_dim]  out
    float* __restrict__ k,             // [n_kv * head_dim] in-place
    const float* __restrict__ q_norm_w,// [head_dim]
    const float* __restrict__ k_norm_w,// [head_dim]
    float eps,
    unsigned int n_q_heads,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int pos,
    float theta_base,
    unsigned int rotary_dim)
{
    extern __shared__ float smem[];          // head_dim floats
    __shared__ float red[PREP_WARPS];
    __shared__ float inv_rms_s;

    const unsigned int blk = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const bool is_q = (blk < n_q_heads);
    const unsigned int head = is_q ? blk : (blk - n_q_heads);
    if (!is_q && head >= n_kv_heads) return;

    const float* nw = is_q ? q_norm_w : k_norm_w;

    // --- load the head vector (deinterleaving Q and emitting gate for Q) ---
    float ss = 0.0f;
    if (is_q) {
        const unsigned int qbase = head * 2u * head_dim;
        const unsigned int obase = head * head_dim;
        for (unsigned int i = tid; i < head_dim; i += PREP_BLOCK) {
            const float v = qgate[qbase + i];
            smem[i] = v;
            ss += v * v;
            gate[obase + i] = qgate[qbase + head_dim + i];
        }
    } else {
        const unsigned int kbase = head * head_dim;
        for (unsigned int i = tid; i < head_dim; i += PREP_BLOCK) {
            const float v = k[kbase + i];
            smem[i] = v;
            ss += v * v;
        }
    }

    // --- per-head RMSNorm (same reduction shape as rmsnorm_per_head_inplace) ---
    ss = warp_reduce_sum_prep(ss);
    if ((tid & 31u) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < PREP_WARPS; i++) total += red[i];
        inv_rms_s = 1.0f / sqrtf(total / (float)head_dim + eps);
    }
    __syncthreads();

    const float inv_rms = inv_rms_s;
    for (unsigned int i = tid; i < head_dim; i += PREP_BLOCK) {
        smem[i] = smem[i] * inv_rms * nw[i];
    }
    __syncthreads();   // RoPE below pairs i with i + half_rot

    // --- partial NeoX RoPE over the first `rotary_dim` elements ---
    const unsigned int rot =
        (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    const unsigned int half_rot = rot >> 1;
    for (unsigned int d = tid; d < half_rot; d += PREP_BLOCK) {
        const float freq = 1.0f / powf(theta_base, (float)(2u * d) / (float)rot);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float x0 = smem[d];
        const float x1 = smem[d + half_rot];
        smem[d]            = x0 * cos_a - x1 * sin_a;
        smem[d + half_rot] = x0 * sin_a + x1 * cos_a;
    }
    __syncthreads();

    // --- store ---
    float* dst = is_q ? (q + head * head_dim) : (k + head * head_dim);
    for (unsigned int i = tid; i < head_dim; i += PREP_BLOCK) {
        dst[i] = smem[i];
    }
}
