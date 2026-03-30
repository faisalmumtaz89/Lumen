// Flash Attention v2 for causal prefill (CUDA).
//
// Replaces the sequential per-token attention_decode loop in the prefill path
// with a single batched kernel that processes all query tokens in parallel.
// Uses online softmax (Milakov & Gimelshein, 2018; Dao et al., 2022) to avoid
// materializing the full [batch, seq_len] attention score matrix.
//
// KV cache layout (head-first):
//   K cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//   V cache: [num_kv_heads, max_seq_len, head_dim] -- F32
//
// Q layout: [batch, num_heads * head_dim] -- F32 (from batched QKV projection)
// O layout: [batch, num_heads * head_dim] -- F32 (output, same shape as Q)
//
// Two kernel variants:
//   flash_attention_causal_v2  -- Br=1, one query per block (128 threads)
//   flash_attention_causal_br4 -- Br=4, four queries per block (128 threads, 1 warp each)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ------------------------------------------------------------------
// Warp-level reductions (namespaced to avoid linker conflicts with attention.cu)
// ------------------------------------------------------------------

__device__ __forceinline__ float fa_warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float fa_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ float fa_block_reduce_max(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = fa_warp_reduce_max(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : -3.402823466e+38f;
        v = fa_warp_reduce_max(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float fa_block_reduce_sum(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = fa_warp_reduce_sum(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = fa_warp_reduce_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

// ------------------------------------------------------------------
// KV tile size. Each iteration processes FA_BC key-value positions.
// ------------------------------------------------------------------
#define FA_BC 32

// ------------------------------------------------------------------
// Flash Attention v2 -- Br=1 (one query per thread block)
//
// Grid:  (num_heads, batch, 1)
// Block: (BLOCK_SIZE, 1, 1) -- 128 threads
//
// Dynamic shared memory layout:
//   float partial[8]           -- warp reduction scratch
//   float q_row[head_dim]      -- query vector for this token
//   float s_tile[FA_BC]        -- attention scores for current KV tile
//
// Total shmem: (8 + head_dim + FA_BC) * sizeof(float)
//
// Algorithm per block:
//   1. Load Q row into shared memory.
//   2. For each KV tile:
//      a. Threads cooperatively compute dot(Q, K[j]) * scale -> s_tile[j]
//      b. Block-reduce to find tile_max, compute exp, block-reduce for sum
//      c. Rescale running O accumulator and add P * V contribution
//   3. Normalize O by 1/l.
//
// The output accumulator O is stored in global memory (o_head[d]) since
// head_dim can be up to 256 and each thread handles a fraction of it.
// ------------------------------------------------------------------

extern "C" __global__ void flash_attention_causal_v2(
    const float* __restrict__ Q,         // [batch, num_heads * head_dim]
    const float* __restrict__ K,         // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ V,         // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ O,               // [batch, num_heads * head_dim]
    unsigned int batch,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int pos_start,              // position of first query token
    unsigned int max_seq_len,
    float scale)                         // 1/sqrt(head_dim)
{
    unsigned int head = blockIdx.x;
    unsigned int q_idx = blockIdx.y;     // which token in batch
    if (head >= num_heads || q_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // GQA mapping
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Causal: attend to positions 0..(pos_start + q_idx) inclusive
    unsigned int seq_len = pos_start + q_idx + 1;

    // Pointers
    unsigned int q_dim = num_heads * head_dim;
    const float* q_head = Q + (unsigned long long)q_idx * q_dim + head * head_dim;
    float* o_head = O + (unsigned long long)q_idx * q_dim + head * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Shared memory
    extern __shared__ float smem[];
    volatile float* partial = smem;             // 8 floats
    float* q_shmem = smem + 8;                  // head_dim floats
    float* s_tile = smem + 8 + head_dim;        // FA_BC floats

    // Load Q vector into shared memory
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        q_shmem[d] = q_head[d];
    }
    __syncthreads();

    // Initialize output to zero
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        o_head[d] = 0.0f;
    }

    // Online softmax state
    float m_prev = -3.402823466e+38f;
    float l_prev = 0.0f;

    // Process KV in tiles of FA_BC
    unsigned int num_kv_tiles = (seq_len + FA_BC - 1) / FA_BC;

    for (unsigned int tile = 0; tile < num_kv_tiles; tile++) {
        unsigned int tile_start = tile * FA_BC;
        unsigned int tile_end = tile_start + FA_BC;
        if (tile_end > seq_len) tile_end = seq_len;
        unsigned int tile_len = tile_end - tile_start;

        // Phase A: Compute attention scores S[j] = dot(Q, K[j]) * scale
        // Each thread handles a subset of tile positions, writes to shared memory.
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            unsigned int kv_pos = tile_start + j;
            const float* k_vec = k_base + kv_pos * head_dim;
            float dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; d++) {
                dot += q_shmem[d] * k_vec[d];
            }
            s_tile[j] = dot * scale;
        }
        // Zero unused slots
        for (unsigned int j = tile_len + tid; j < FA_BC; j += block_size) {
            s_tile[j] = -3.402823466e+38f;
        }
        __syncthreads();

        // Phase B1: Tile max
        float local_max = -3.402823466e+38f;
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        float tile_max = fa_block_reduce_max(local_max, partial, tid, block_size);
        float m_new = fmaxf(m_prev, tile_max);

        // Phase B2: Compute P[j] = exp(S[j] - m_new) and write to shmem
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            s_tile[j] = expf(s_tile[j] - m_new);
        }
        for (unsigned int j = tile_len + tid; j < FA_BC; j += block_size) {
            s_tile[j] = 0.0f;
        }
        __syncthreads();

        // Phase B3: Sum of P
        float local_psum = 0.0f;
        for (unsigned int j = tid; j < tile_len; j += block_size) {
            local_psum += s_tile[j];
        }
        float tile_psum = fa_block_reduce_sum(local_psum, partial, tid, block_size);

        // Phase B4: Rescale running O and accumulate P * V
        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + tile_psum;

        for (unsigned int d = tid; d < head_dim; d += block_size) {
            float o_val = o_head[d] * rescale;
            float pv_sum = 0.0f;
            for (unsigned int j = 0; j < tile_len; j++) {
                pv_sum += s_tile[j] * v_base[(tile_start + j) * head_dim + d];
            }
            o_head[d] = o_val + pv_sum;
        }
        __syncthreads();

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization: O = O / l
    if (l_prev > 0.0f) {
        float inv_l = 1.0f / l_prev;
        for (unsigned int d = tid; d < head_dim; d += block_size) {
            o_head[d] *= inv_l;
        }
    }
}


// ------------------------------------------------------------------
// Flash Attention v2 -- Br=4 (four query rows per thread block)
//
// Each warp independently handles one query row using warp-level reductions.
// This avoids block-level syncs between queries and increases parallelism.
//
// Grid:  (num_heads, ceil(batch / 4), 1)
// Block: (128, 1, 1) -- 4 warps of 32 threads
//
// Dynamic shared memory layout:
//   float q_rows[4][head_dim]      -- query vectors
//   float s_tiles[4][FA_BC]        -- score tiles
//
// Total shmem: 4 * (head_dim + FA_BC) * sizeof(float)
//
// Each warp's Q row is at smem[warp_id * head_dim].
// Each warp's score tile is at smem[4 * head_dim + warp_id * FA_BC].
// ------------------------------------------------------------------

#define FA_BR 4
#define FA_WARP_SIZE 32

extern "C" __global__ void flash_attention_causal_br4(
    const float* __restrict__ Q,         // [batch, num_heads * head_dim]
    const float* __restrict__ K,         // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ V,         // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ O,               // [batch, num_heads * head_dim]
    unsigned int batch,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int pos_start,
    unsigned int max_seq_len,
    float scale)
{
    unsigned int head = blockIdx.x;
    unsigned int q_tile = blockIdx.y;    // which Q tile (groups of 4 tokens)
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid >> 5;     // 0..3
    unsigned int lane = tid & 31u;

    // Which query token does this warp handle?
    unsigned int q_idx = q_tile * FA_BR + warp_id;
    if (q_idx >= batch) return;

    // GQA mapping
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Causal boundary
    unsigned int seq_len = pos_start + q_idx + 1;

    // Pointers
    unsigned int q_dim = num_heads * head_dim;
    const float* q_head = Q + (unsigned long long)q_idx * q_dim + head * head_dim;
    float* o_head = O + (unsigned long long)q_idx * q_dim + head * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Shared memory: Q rows followed by score tiles
    extern __shared__ float smem[];
    float* q_shmem = smem + warp_id * head_dim;
    float* s_tile = smem + FA_BR * head_dim + warp_id * FA_BC;

    // Load Q vector (warp-cooperative)
    for (unsigned int d = lane; d < head_dim; d += FA_WARP_SIZE) {
        q_shmem[d] = q_head[d];
    }
    // Need block sync since different warps write to different shmem regions
    // and we read from our own region after this point.
    __syncthreads();

    // Initialize output
    for (unsigned int d = lane; d < head_dim; d += FA_WARP_SIZE) {
        o_head[d] = 0.0f;
    }

    // Online softmax state
    float m_prev = -3.402823466e+38f;
    float l_prev = 0.0f;

    unsigned int num_kv_tiles = (seq_len + FA_BC - 1) / FA_BC;

    for (unsigned int tile = 0; tile < num_kv_tiles; tile++) {
        unsigned int tile_start = tile * FA_BC;
        unsigned int tile_end = tile_start + FA_BC;
        if (tile_end > seq_len) tile_end = seq_len;
        unsigned int tile_len = tile_end - tile_start;

        // Phase A: Compute scores
        // With 32 lanes and FA_BC=32, each lane handles exactly 1 position
        for (unsigned int j = lane; j < tile_len; j += FA_WARP_SIZE) {
            unsigned int kv_pos = tile_start + j;
            const float* k_vec = k_base + kv_pos * head_dim;
            float dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; d++) {
                dot += q_shmem[d] * k_vec[d];
            }
            s_tile[j] = dot * scale;
        }
        for (unsigned int j = tile_len + lane; j < FA_BC; j += FA_WARP_SIZE) {
            s_tile[j] = -3.402823466e+38f;
        }
        __syncwarp(0xffffffff);

        // Phase B1: Tile max (warp reduction)
        float local_max = -3.402823466e+38f;
        for (unsigned int j = lane; j < tile_len; j += FA_WARP_SIZE) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        float tile_max = fa_warp_reduce_max(local_max);
        float m_new = fmaxf(m_prev, tile_max);

        // Phase B2: Compute P[j] = exp(S[j] - m_new)
        for (unsigned int j = lane; j < tile_len; j += FA_WARP_SIZE) {
            s_tile[j] = expf(s_tile[j] - m_new);
        }
        for (unsigned int j = tile_len + lane; j < FA_BC; j += FA_WARP_SIZE) {
            s_tile[j] = 0.0f;
        }
        __syncwarp(0xffffffff);

        // Phase B3: Sum of P
        float local_psum = 0.0f;
        for (unsigned int j = lane; j < tile_len; j += FA_WARP_SIZE) {
            local_psum += s_tile[j];
        }
        float tile_psum = fa_warp_reduce_sum(local_psum);

        // Phase B4: Rescale + accumulate P * V
        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + tile_psum;

        for (unsigned int d = lane; d < head_dim; d += FA_WARP_SIZE) {
            float o_val = o_head[d] * rescale;
            float pv_sum = 0.0f;
            for (unsigned int j = 0; j < tile_len; j++) {
                pv_sum += s_tile[j] * v_base[(tile_start + j) * head_dim + d];
            }
            o_head[d] = o_val + pv_sum;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization
    if (l_prev > 0.0f) {
        float inv_l = 1.0f / l_prev;
        for (unsigned int d = lane; d < head_dim; d += FA_WARP_SIZE) {
            o_head[d] *= inv_l;
        }
    }
}
