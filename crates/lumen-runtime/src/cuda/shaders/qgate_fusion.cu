// Q+gate fusion kernels for Qwen3.5 full-attention layers.
//
// Qwen3.5 full-attention layers have fused Q+gate in attn_q.weight:
//   wq output: [Q_h0(head_dim), gate_h0(head_dim), Q_h1(head_dim), gate_h1(head_dim), ...]
//   Total output: q_dim * 2 = num_heads * head_dim * 2
//
// After deinterleaving, the gate is applied after attention:
//   attn_out_gated = sigmoid(gate) * attn_out
//
// Kernels:
//   deinterleave_qgate: Split interleaved Q+gate into separate Q and gate buffers
//   sigmoid_mul: sigmoid(gate) * x in-place
//   rmsnorm_per_head_inplace: Per-head RMSNorm (shared weight across heads)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ============================================================================
// deinterleave_qgate: Split interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
// into separate Q [Q_h0, Q_h1, ...] and gate [gate_h0, gate_h1, ...] buffers.
//
// Input:  qgate[num_heads * head_dim * 2] -- interleaved Q+gate
// Output: q[num_heads * head_dim]         -- Q vectors only
//         gate[num_heads * head_dim]      -- gate vectors only
//
// Grid: ceil(q_dim / 256) blocks of 256 threads, where q_dim = num_heads * head_dim.
// ============================================================================
extern "C" __global__ void deinterleave_qgate(
    const float* __restrict__ qgate,   // [num_heads * head_dim * 2]
    float* __restrict__ q,             // [num_heads * head_dim]
    float* __restrict__ gate,          // [num_heads * head_dim]
    unsigned int head_dim,
    unsigned int num_heads)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q_dim = num_heads * head_dim;
    if (idx >= q_dim) return;

    // Element idx belongs to head h, position p within head
    unsigned int h = idx / head_dim;
    unsigned int p = idx % head_dim;

    // In the interleaved layout, head h's Q is at offset h * 2 * head_dim
    // and gate is at h * 2 * head_dim + head_dim
    unsigned int qgate_q_offset = h * 2 * head_dim + p;
    unsigned int qgate_g_offset = h * 2 * head_dim + head_dim + p;

    q[idx] = qgate[qgate_q_offset];
    gate[idx] = qgate[qgate_g_offset];
}

// ============================================================================
// sigmoid_mul: Compute sigmoid(gate) * x, writing result to out.
//
// For each element i:
//   out[i] = sigmoid(gate[i]) * x[i]
//         = x[i] / (1 + exp(-gate[i]))
//
// Grid: ceil(n / 256) blocks of 256 threads.
// ============================================================================
extern "C" __global__ void sigmoid_mul(
    const float* __restrict__ gate,    // [n]
    const float* __restrict__ x,       // [n]
    float* __restrict__ out,           // [n]
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = sig * x[idx];
}

// sigmoid_mul_inplace: same arithmetic, x updated in place (element-wise
// same-index read/write — no cross-thread aliasing, so no __restrict__ claim
// on the in/out pointer). Removes the temp-buffer round-trip + DtoD copy.
extern "C" __global__ void sigmoid_mul_inplace(
    const float* __restrict__ gate,    // [n]
    float* x,                          // [n] in/out
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    x[idx] = sig * x[idx];
}

// ============================================================================
// rmsnorm_per_head_inplace: Per-head RMSNorm with shared weight across heads.
//
// For each head h in [0, num_heads):
//   rms = sqrt(mean(x[h*head_dim .. (h+1)*head_dim]^2) + eps)
//   x[h*head_dim + i] = x[h*head_dim + i] / rms * weight[i]
//
// weight is [head_dim], shared across all heads (not [num_heads * head_dim]).
//
// Grid: (num_heads, 1, 1) -- one block per head
// Block: (block_dim, 1, 1) -- threads cooperate within a head
// Shared memory: (block_size / 32) * sizeof(float)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_qgate(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

extern "C" __global__ void rmsnorm_per_head_inplace(
    float* __restrict__ x,             // [num_heads * head_dim] modified in-place
    const float* __restrict__ weight,  // [head_dim] shared across heads
    unsigned int num_heads,
    unsigned int head_dim,
    float eps)
{
    extern __shared__ float shared[];

    unsigned int h = blockIdx.x;
    if (h >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    float* head = x + h * head_dim;

    // Phase 1: Sum of squares
    float ss = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        float v = head[i];
        ss += v * v;
    }

    ss = warp_reduce_sum_qgate(ss);
    if (lane_id == 0) shared[warp_id] = ss;
    __syncthreads();

    float total_ss = 0.0f;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total_ss = warp_reduce_sum_qgate(total_ss);
    }
    if (tid == 0) shared[0] = total_ss;
    __syncthreads();
    total_ss = shared[0];

    // RMSNorm: x[i] = x[i] / rms * weight[i]
    float rms = sqrtf(total_ss / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    // Phase 2: Normalize in-place with shared weight
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        head[i] = head[i] * inv_rms * weight[i];
    }
}


// ============================================================================
// attn_prep_fused: ONE launch replacing the six-launch full-attention prep
// chain (deinterleave_qgate, per-head Q RMSNorm, per-head K RMSNorm, NeoX
// RoPE, K append, V append). This region is CPU-launch-shadow bound at
// decode, so the lever is the launch count, not kernel microseconds.
//
// Grid: (num_q_heads + 2*num_kv_heads, 1, 1) CTAs — one per Q head, then one
// per K head, then one per V head. Block: (head_dim, 1, 1); host guarantees
// head_dim == blockDim.x, head_dim % 32 == 0, head_dim <= 1024,
// half_rot <= 128, and an F32 KV cache.
//
// DETERMINISM: each per-value op sequence is cloned from its source kernel —
// the deinterleave indexing, the rmsnorm_per_head_inplace reduction (block
// size == head_dim makes its strided loops one-element-per-thread, exactly
// this shape), the tabled-RoPE cos/sin (identical expression per pair index,
// computed once per CTA), the NeoX pairing, and the head-first cache write.
// Value flow (deinterleave -> norm -> rope -> store) matches the chain
// order; intermediate register/shared staging replaces exact F32 global
// round-trips.
// ============================================================================
extern "C" __global__ void attn_prep_fused(
    const float* __restrict__ qgate,     // [num_q_heads * head_dim * 2]
    float* __restrict__ q,               // [num_q_heads * head_dim] OUT
    float* __restrict__ gate,            // [num_q_heads * head_dim] OUT
    float* __restrict__ k,               // [num_kv_heads * head_dim] IN/OUT
    const float* __restrict__ v,         // [num_kv_heads * head_dim]
    const float* __restrict__ q_norm_w,  // [head_dim]
    const float* __restrict__ k_norm_w,  // [head_dim]
    float* __restrict__ k_cache,         // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ v_cache,         // [num_kv_heads, max_seq_len, head_dim]
    unsigned int pos,
    unsigned int max_seq_len,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float eps,
    float theta_base,
    unsigned int rotary_dim)
{
    __shared__ float red[33];
    __shared__ float val[1024];
    __shared__ float tcos[128];
    __shared__ float tsin[128];

    unsigned int b = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = blockDim.x >> 5;

    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;

    // ---- V heads: pure cache copy (kv_cache_write clone) ----
    if (b >= num_q_heads + num_kv_heads) {
        unsigned int vh = b - num_q_heads - num_kv_heads;
        if (tid < head_dim) {
            unsigned int cache_idx = vh * max_seq_len * head_dim + pos * head_dim + tid;
            v_cache[cache_idx] = v[vh * head_dim + tid];
        }
        return;
    }

    // ---- rope table: identical expression per pair index, once per CTA ----
    if (tid < half_rot) {
        unsigned int d = tid;
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        tcos[d] = cosf(angle);
        tsin[d] = sinf(angle);
    }

    int is_q = b < num_q_heads;
    unsigned int h = is_q ? b : (b - num_q_heads);
    const float* norm_w = is_q ? q_norm_w : k_norm_w;

    // ---- load + (Q only) deinterleave clone ----
    float x = 0.0f;
    if (tid < head_dim) {
        if (is_q) {
            x = qgate[h * 2 * head_dim + tid];
            gate[h * head_dim + tid] = qgate[h * 2 * head_dim + head_dim + tid];
        } else {
            x = k[h * head_dim + tid];
        }
    }

    // ---- per-head RMSNorm (rmsnorm_per_head_inplace clone at
    // block_size == head_dim: one element per thread) ----
    float ss = x * x;
    ss = warp_reduce_sum_qgate(ss);
    if (lane_id == 0) red[warp_id] = ss;
    __syncthreads();
    float total_ss = 0.0f;
    if (warp_id == 0) {
        total_ss = (lane_id < num_warps) ? red[lane_id] : 0.0f;
        total_ss = warp_reduce_sum_qgate(total_ss);
    }
    if (tid == 0) red[0] = total_ss;
    __syncthreads();
    total_ss = red[0];
    float rms = sqrtf(total_ss / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;
    float normed = x * inv_rms * norm_w[tid < head_dim ? tid : 0];

    // ---- NeoX rope from the table (pairs (d, d + half_rot)) ----
    val[tid] = normed;
    __syncthreads();
    float out_v = normed;
    if (tid < 2u * half_rot) {
        if (tid < half_rot) {
            out_v = val[tid] * tcos[tid] - val[tid + half_rot] * tsin[tid];
        } else {
            unsigned int d = tid - half_rot;
            out_v = val[d] * tsin[d] + val[tid] * tcos[d];
        }
    }

    // ---- store: Q to q; K to k AND its cache slot ----
    if (tid < head_dim) {
        if (is_q) {
            q[h * head_dim + tid] = out_v;
        } else {
            k[h * head_dim + tid] = out_v;
            unsigned int cache_idx = h * max_seq_len * head_dim + pos * head_dim + tid;
            k_cache[cache_idx] = out_v;
        }
    }
}
