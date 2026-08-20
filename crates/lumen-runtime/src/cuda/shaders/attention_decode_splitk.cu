// ==========================================================================
// Split-K decode attention: sequence-parallel variant of
// attention_decode_tiled.cu for the 1-token decode step.
//
// The tiled kernel launches ONE CTA per Q head. On models with few Q heads
// (Qwen3.8-27B full attention: 24 heads) that is 24 CTAs on a 108-SM A100 —
// ~1.4% of maximum resident-warp capacity — and each CTA walks its KV tiles
// serially,
// so per-call latency grows linearly with seq_len. This variant splits the
// sequence into ATTN_SPLITK_S chunks per head:
//
//   pass 1 (attention_decode_splitk_partial, grid = num_heads * S):
//     chunk c of head h runs the SAME tile walk as the tiled kernel over
//     positions [c*ceil(seq/S), min(seq, (c+1)*ceil(seq/S))), emitting the
//     chunk-local softmax triple (m_c, l_c, o_c[head_dim]) UNNORMALIZED.
//   pass 2 (attention_decode_splitk_merge, grid = num_heads):
//     M = max_c m_c;  L = sum_c l_c * exp(m_c - M);
//     out[d] = sum_c o_c[d] * exp(m_c - M) / L
//     — the standard online-softmax merge identity.
//
// PRECISION: per-chunk phases reuse the tiled kernel's exact per-tile
// arithmetic (same qk dot, same block reductions, same online update), but
// the cross-chunk merge sums in a different order than the tiled kernel's
// progressive rescale, so outputs are a quality-equivalent NEAR-TIE (same
// class as the shipping mmvq family), not guaranteed byte-identical to the
// tiled route. Deterministic for a fixed S. One asymmetry beyond
// reassociation: the per-chunk o/l accumulators are chunk-locally scaled,
// so a chunk whose local max sits far below the global max can overflow to
// inf (then NaN in the merge) where the tiled route's globally-rescaled
// accumulator would not — this needs |V·softmax| sums near F32_MAX (~1e38),
// orders of magnitude beyond model activations.
//
// Scratch layout (device buffers, sized by the host):
//   m_part [num_heads * S]              F32
//   l_part [num_heads * S]              F32
//   o_part [num_heads * S * head_dim]   F32
//
// Requires (enforced by the host gate
// `attention_decode_splitk_supports_head_dim`): head_dim % 128 == 0 and
// head_dim <= 1024 (MAX_SLOTS = 8 slots of one 128-thread block).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define T_C        128u
#define NEG_INF (-3.402823466e+38f)

__device__ __forceinline__ float splitk_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ float splitk_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ float splitk_block_reduce_max(
    float val, volatile float* partial, unsigned int tid, unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;
    val = splitk_warp_max(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();
    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : NEG_INF;
        v = splitk_warp_max(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float splitk_block_reduce_sum(
    float val, volatile float* partial, unsigned int tid, unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;
    val = splitk_warp_sum(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();
    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = splitk_warp_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float splitk_qk_dot(
    const float* __restrict__ q_row, const float* __restrict__ k_vec,
    unsigned int head_dim, float scale)
{
    float dot = 0.0f;
    if ((head_dim & 3u) == 0u) {
        unsigned int hd4 = head_dim >> 2;
        const float4* q4 = reinterpret_cast<const float4*>(q_row);
        const float4* k4 = reinterpret_cast<const float4*>(k_vec);
        for (unsigned int d4 = 0; d4 < hd4; d4++) {
            float4 q = q4[d4];
            float4 k = k4[d4];
            dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
        }
    } else {
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_vec[d];
        }
    }
    return dot * scale;
}

extern "C" __global__ void attention_decode_splitk_partial(
    const float* __restrict__ q,           // [num_heads * head_dim]
    const float* __restrict__ k_cache,     // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ v_cache,     // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ m_part,            // [num_heads * S]
    float* __restrict__ l_part,            // [num_heads * S]
    float* __restrict__ o_part,            // [num_heads * S * head_dim]
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int seq_len,
    unsigned int max_seq_len,
    float scale,
    unsigned int num_chunks)               // S
{
    unsigned int blk = blockIdx.x;
    unsigned int head = blk / num_chunks;
    unsigned int chunk = blk % num_chunks;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int chunk_span = (seq_len + num_chunks - 1u) / num_chunks;
    unsigned int p0 = chunk * chunk_span;
    unsigned int p1 = p0 + chunk_span;
    if (p1 > seq_len) p1 = seq_len;

    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;
    const float* q_head = q + head * head_dim;
    unsigned long long kv_base =
        (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)head_dim;

    extern __shared__ float smem[];
    volatile float* partial = smem;
    float* q_row = smem + 8;
    float* s_tile = smem + 8 + head_dim;

    for (unsigned int d = tid; d < head_dim; d += block_size) {
        q_row[d] = q_head[d];
    }
    __syncthreads();

    float m_prev = NEG_INF;
    float l_prev = 0.0f;
    constexpr unsigned int MAX_SLOTS = 8u;
    float o_acc[MAX_SLOTS];
#pragma unroll
    for (unsigned int s = 0; s < MAX_SLOTS; s++) {
        o_acc[s] = 0.0f;
    }
    unsigned int num_slots = (head_dim + block_size - 1u) / block_size;

    if (p0 < p1) {
        unsigned int num_tiles = (p1 - p0 + T_C - 1u) / T_C;
        for (unsigned int tile = 0; tile < num_tiles; tile++) {
            unsigned int tile_start = p0 + tile * T_C;
            unsigned int tile_end_raw = tile_start + T_C;
            unsigned int tile_end = (tile_end_raw < p1) ? tile_end_raw : p1;
            unsigned int tile_len = tile_end - tile_start;

            for (unsigned int j = tid; j < T_C; j += block_size) {
                if (j < tile_len) {
                    unsigned int pos = tile_start + j;
                    const float* k_vec =
                        k_cache + kv_base + (unsigned long long)pos * (unsigned long long)head_dim;
                    s_tile[j] = splitk_qk_dot(q_row, k_vec, head_dim, scale);
                } else {
                    s_tile[j] = NEG_INF;
                }
            }
            __syncthreads();

            float local_max = NEG_INF;
            for (unsigned int j = tid; j < T_C; j += block_size) {
                local_max = fmaxf(local_max, s_tile[j]);
            }
            float tile_max = splitk_block_reduce_max(local_max, partial, tid, block_size);
            float m_new = fmaxf(m_prev, tile_max);
            float rescale = expf(m_prev - m_new);

            for (unsigned int j = tid; j < T_C; j += block_size) {
                if (j < tile_len) {
                    s_tile[j] = expf(s_tile[j] - m_new);
                } else {
                    s_tile[j] = 0.0f;
                }
            }
            __syncthreads();

            float local_sum = 0.0f;
            for (unsigned int j = tid; j < T_C; j += block_size) {
                local_sum += s_tile[j];
            }
            float tile_sum = splitk_block_reduce_sum(local_sum, partial, tid, block_size);
            float l_new = rescale * l_prev + tile_sum;

#pragma unroll
            for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
                if (slot >= num_slots) break;
                unsigned int d = tid + slot * block_size;
                if (d < head_dim) {
                    float pv = 0.0f;
                    for (unsigned int j = 0; j < tile_len; j++) {
                        unsigned int pos = tile_start + j;
                        float v_dj = v_cache[kv_base
                            + (unsigned long long)pos * (unsigned long long)head_dim
                            + (unsigned long long)d];
                        pv += s_tile[j] * v_dj;
                    }
                    o_acc[slot] = rescale * o_acc[slot] + pv;
                }
            }

            m_prev = m_new;
            l_prev = l_new;
            __syncthreads();
        }
    }

    if (tid == 0u) {
        m_part[head * num_chunks + chunk] = m_prev;
        l_part[head * num_chunks + chunk] = l_prev;
    }
#pragma unroll
    for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
        if (slot >= num_slots) break;
        unsigned int d = tid + slot * block_size;
        if (d < head_dim) {
            o_part[((unsigned long long)head * num_chunks + chunk)
                   * (unsigned long long)head_dim + d] = o_acc[slot];
        }
    }
}

extern "C" __global__ void attention_decode_splitk_merge(
    const float* __restrict__ m_part,      // [num_heads * S]
    const float* __restrict__ l_part,      // [num_heads * S]
    const float* __restrict__ o_part,      // [num_heads * S * head_dim]
    float* __restrict__ attn_out,          // [num_heads * head_dim]
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int num_chunks)               // S
{
    unsigned int head = blockIdx.x;
    if (head >= num_heads) return;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    float m_max = NEG_INF;
    for (unsigned int c = 0; c < num_chunks; c++) {
        m_max = fmaxf(m_max, m_part[head * num_chunks + c]);
    }
    // Empty chunks carry (m = NEG_INF, l = 0, o = 0) and are skipped. The
    // `!= 0` form (not `> 0`) keeps a NaN `l` chunk IN the sum so NaN
    // propagates like the tiled kernel instead of being silently dropped.
    float l_total = 0.0f;
    for (unsigned int c = 0; c < num_chunks; c++) {
        float lc = l_part[head * num_chunks + c];
        if (lc != 0.0f) {
            l_total += lc * expf(m_part[head * num_chunks + c] - m_max);
        }
    }
    float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;

    for (unsigned int d = tid; d < head_dim; d += block_size) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < num_chunks; c++) {
            float lc = l_part[head * num_chunks + c];
            if (lc != 0.0f) {
                acc += o_part[((unsigned long long)head * num_chunks + c)
                              * (unsigned long long)head_dim + d]
                    * expf(m_part[head * num_chunks + c] - m_max);
            }
        }
        attn_out[head * head_dim + d] = acc * inv_l;
    }
}
