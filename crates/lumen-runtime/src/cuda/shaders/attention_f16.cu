// Multi-head attention decode kernel reading an F16 KV cache.
//
// Mirrors `attention_decode` from attention.cu but loads K and V as
// half-precision (unsigned short) and converts to f32 in registers before
// the dot product / weighted accumulation. The query Q remains f32 (the
// activation pipeline is f32 end-to-end).
//
// KV cache layout (head-first, identical to F32 path):
//   K cache: [num_kv_heads, max_seq_len, head_dim] -- f16 bits
//   V cache: [num_kv_heads, max_seq_len, head_dim] -- f16 bits
//
// Three phases per head (unchanged from F32 variant):
//   Phase 1: QK dot products
//   Phase 2: Numerically stable softmax (max-subtract + exp + normalize)
//   Phase 3: Weighted V accumulation with per-dimension parallel reduction
//
// NVRTC-compatible: inline PTX for f16<->f32, no cuda_fp16.h.

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float af16_to_f32(unsigned short h) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(h));
    return result;
}

// Warp-level max reduction using butterfly shuffle.
__device__ __forceinline__ float af16_warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float af16_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Cross-warp max reduction via shared memory.
__device__ __forceinline__ float af16_block_reduce_max(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = af16_warp_reduce_max(val);

    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : -3.402823466e+38f;
        v = af16_warp_reduce_max(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

// Cross-warp sum reduction via shared memory.
__device__ __forceinline__ float af16_block_reduce_sum(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = af16_warp_reduce_sum(val);

    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = af16_warp_reduce_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

extern "C" __global__ void attention_decode_f16(
    const float*          __restrict__ q,          // [num_heads * head_dim] f32
    const unsigned short* __restrict__ k_cache,    // [num_kv_heads, max_seq_len, head_dim] f16
    const unsigned short* __restrict__ v_cache,    // [num_kv_heads, max_seq_len, head_dim] f16
    float*                __restrict__ attn_out,   // [num_heads * head_dim] f32
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

    // Shared memory layout (same as F32 path):
    //   [0..7]: partial reduction workspace (up to 8 warps for 256 threads)
    //   [8..8+seq_len-1]: attention scores (softmax probabilities)
    // The host sets shared_mem_bytes = (8 + seq_len) * sizeof(float).
    extern __shared__ float smem[];
    volatile float* partial = smem;
    float* scores = smem + 8;

    // ---- Phase 1: QK dot products -- scalar load+convert per element ----
    // head_dim is always a multiple of 4 in standard models, but vectorized
    // unsigned-short loads would need short4; we keep the scalar loop here for
    // simplicity. The dominant cost remains memory bandwidth (1 byte/elem each
    // for K and V, halved vs the F32 path).
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        const unsigned short* k_vec = k_cache + kv_base + (unsigned long long)t * head_dim;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += q_head[d] * af16_to_f32(k_vec[d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ---- Phase 2: Numerically stable softmax (identical to F32 path) ----

    // 2a: find max
    float local_max = -3.402823466e+38f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        local_max = fmaxf(local_max, scores[t]);
    }
    float global_max = af16_block_reduce_max(local_max, partial, tid, block_size);

    // 2b: exp(score - max), accumulate sum
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        float e = expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    float global_sum = af16_block_reduce_sum(local_sum, partial, tid, block_size);

    // 2c: normalize
    float inv_sum = 1.0f / global_sum;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation -- scalar load+convert per element.
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < seq_len; t++) {
            float s = scores[t];
            float v = af16_to_f32(v_cache[kv_base + (unsigned long long)t * head_dim + d]);
            acc += s * v;
        }
        out_head[d] = acc;
    }
}
