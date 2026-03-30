// Multi-head attention decode kernel with GQA support for CUDA.
//
// Computes attention output for a single query token against the full KV cache.
// One thread block per query head. Supports Grouped Query Attention (GQA) where
// multiple Q heads share the same KV head.
//
// KV cache layout (head-first):
//   K cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//   V cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//
// Three phases per head:
//   Phase 1: QK dot products -- each thread handles a subset of seq positions
//   Phase 2: Numerically stable softmax (max-subtract + exp + normalize)
//   Phase 3: Weighted V accumulation with per-dimension parallel reduction

// Warp-level max reduction using butterfly shuffle.
// FULL_MASK = 0xffffffff activates all 32 lanes.
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Cross-warp max reduction via shared memory.
// partial[] must have at least num_warps elements.
// Returns the global max in all threads (broadcast via shared memory).
__device__ __forceinline__ float block_reduce_max(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = warp_reduce_max(val);

    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    // First warp reads all partial values and reduces
    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : -3.402823466e+38f;
        v = warp_reduce_max(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

// Cross-warp sum reduction via shared memory.
__device__ __forceinline__ float block_reduce_sum(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

extern "C" __global__ void attention_decode(
    const float* __restrict__ q,           // [num_heads * head_dim]
    const float* __restrict__ k_cache,     // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ v_cache,     // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ attn_out,          // [num_heads * head_dim]
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int seq_len,        // current sequence length (positions 0..seq_len-1)
    unsigned int max_seq_len,    // allocated cache dimension
    float scale                  // 1/sqrt(head_dim)
)
{
    unsigned int head = blockIdx.x;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // GQA mapping: multiple Q heads share the same KV head
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Base pointers for this head
    const float* q_head = q + head * head_dim;
    float* out_head = attn_out + head * head_dim;

    // KV cache base for this KV head (head-first layout)
    unsigned long long kv_base = (unsigned long long)kv_h * max_seq_len * head_dim;

    // Shared memory layout:
    //   [0..7]:   partial reduction workspace (up to 8 warps for 256 threads)
    //   [8..8+seq_len-1]: attention scores (softmax probabilities)
    //
    // Dynamic shared memory is used for the scores array since seq_len varies.
    // The host must set shared_mem_bytes = (8 + seq_len) * sizeof(float).
    extern __shared__ float smem[];
    volatile float* partial = smem;        // 8 floats for warp reduction
    float* scores = smem + 8;              // seq_len floats for attention scores

    // ---- Phase 1: QK dot products with float4 vectorized loads ----
    // head_dim is always a multiple of 4 (64, 128, 256 in standard models).
    // float4 reduces instruction count 4x and uses 128-bit memory transactions.
    unsigned int head_dim_vec = head_dim >> 2;
    const float4* q_head_v = reinterpret_cast<const float4*>(q_head);

    for (unsigned int t = tid; t < seq_len; t += block_size) {
        const float4* k_vec_v = reinterpret_cast<const float4*>(k_cache + kv_base + (unsigned long long)t * head_dim);
        float dot = 0.0f;
        for (unsigned int d4 = 0; d4 < head_dim_vec; d4++) {
            float4 qv = q_head_v[d4];
            float4 kv = k_vec_v[d4];
            dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ---- Phase 2: Numerically stable softmax ----

    // 2a: Find max score across all positions
    float local_max = -3.402823466e+38f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        local_max = fmaxf(local_max, scores[t]);
    }
    float global_max = block_reduce_max(local_max, partial, tid, block_size);

    // 2b: Compute exp(score - max) and accumulate sum
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        float e = expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    float global_sum = block_reduce_sum(local_sum, partial, tid, block_size);

    // 2c: Normalize to get softmax probabilities
    float inv_sum = 1.0f / global_sum;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation with float4 vectorized loads ----
    // Each thread handles 4 contiguous dimensions per iteration,
    // reducing instruction count 4x and using 128-bit V cache reads.
    for (unsigned int d4 = tid; d4 < head_dim_vec; d4 += block_size) {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int t = 0; t < seq_len; t++) {
            float s = scores[t];
            float4 vv = reinterpret_cast<const float4*>(v_cache + kv_base + (unsigned long long)t * head_dim)[d4];
            acc.x += s * vv.x;
            acc.y += s * vv.y;
            acc.z += s * vv.z;
            acc.w += s * vv.w;
        }
        reinterpret_cast<float4*>(out_head)[d4] = acc;
    }
}
