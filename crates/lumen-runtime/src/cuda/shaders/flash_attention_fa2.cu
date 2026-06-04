// Flash Attention 2 with mask block-skip and Split-K reduce (CUDA, SM 70+).
//
// Two-kernel decomposition tuned for long-context causal prefill (pp4096+):
//
//   flash_attention_fa2_causal:
//     Streaming-softmax FA2 (Dao et al., 2022) with two structural wins over
//     the existing `flash_attention_causal_v2` / `_br4` kernels:
//
//       (1) Mask block-skip. The existing kernels iterate every KV tile up to
//           the largest query position in the block, then apply a per-element
//           causal mask. This wastes O(seq_len^2 / 2) FLOPs on the lower-
//           triangular zeros. The new kernel computes the per-Q-tile causal
//           boundary `tile_max_kv` = pos_start + q_row_end and stops the KV
//           loop there -- the entire upper-triangular block of tiles is
//           never visited. Within the diagonal tile, per-element masking
//           still applies.
//
//       (2) Multiple Q rows per block (Br=4 default, configurable). Each
//           warp owns one query row and uses warp-level reductions for
//           tile-max and tile-sum, avoiding block-wide syncs between rows.
//           Identical to the existing `flash_attention_causal_br4` per-row
//           plan but extended with the block-skip and an explicit Split-K
//           entry point.
//
//   flash_attention_fa2_splitk_reduce:
//     When the launcher uses Split-K mode (sequence chunked across multiple
//     CTAs along the KV axis), this reduce kernel merges the per-CTA
//     (output, max, sum) tuples using the FA2 online-softmax rescale rule:
//
//       m_new = max(m_a, m_b)
//       l_new = exp(m_a - m_new) * l_a + exp(m_b - m_new) * l_b
//       O_new = (exp(m_a - m_new) * l_a * O_a + exp(m_b - m_new) * l_b * O_b)
//               / l_new
//
//     The split kernel writes O_partial, m_partial, l_partial per (Q-row,
//     head, split). The reduce kernel collapses across splits into the final
//     [batch, num_heads * head_dim] output buffer.
//
// KV cache layout (head-first, F32):
//   K cache: [num_kv_heads, max_seq_len, head_dim]
//   V cache: [num_kv_heads, max_seq_len, head_dim]
//
// Q layout: [batch, num_heads * head_dim] -- F32
// O layout: [batch, num_heads * head_dim] -- F32
//
// NVRTC-compatible: no system includes, extern "C" linkage. The launcher
// (host side) is responsible for picking Split-K vs single-kernel mode based
// on seq_len -- typically Split-K above ~4096 KV positions.

// ---------------------------------------------------------------------------
// Constants and helpers
// ---------------------------------------------------------------------------

#define FA2_BC        64   // KV tile size along sequence axis
#define FA2_BR        4    // query rows per block (one warp per row)
#define FA2_WARP_SIZE 32

// Warp-level reductions, namespaced to avoid linker conflict with other
// flash_attention*.cu units (NVRTC links all kernels into one module).
__device__ __forceinline__ float fa2_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return v;
}

__device__ __forceinline__ float fa2_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// Helper: causal upper bound for query position q_pos.
// In a prefill batch starting at pos_start, the query at q_pos attends to
// KV positions [0, q_pos] inclusive. Used both for the block-skip cutoff
// and the per-element mask in the diagonal tile.
__device__ __forceinline__ unsigned int fa2_kv_upper_for_q(
    unsigned int pos_start,
    unsigned int q_local_idx)
{
    return pos_start + q_local_idx;
}

// ---------------------------------------------------------------------------
// Single-kernel FA2 with mask block-skip
//
// Grid:  (num_heads, ceil(batch / FA2_BR), 1)
// Block: (128, 1, 1)   --  4 warps of 32 threads, one warp per query row
//
// Shared memory:
//   float q_rows[FA2_BR][head_dim]
//   float s_tiles[FA2_BR][FA2_BC]
// Total: FA2_BR * (head_dim + FA2_BC) * sizeof(float)
//   = 4 * (head_dim + 64) * 4
//   ~ 1.5 KB at head_dim=128, 4.25 KB at head_dim=256.
//
// Online state (per-warp registers, not shared):
//   m_prev, l_prev  -- running max / normaliser
//   o[head_dim/32]  -- per-thread chunk of the output accumulator;
//                       held in global memory (o_head) for head_dim > 32 to
//                       avoid spilling registers.
// ---------------------------------------------------------------------------

extern "C" __global__ void flash_attention_fa2_causal(
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
    float scale)                          // 1/sqrt(head_dim)
{
    unsigned int head = blockIdx.x;
    unsigned int q_tile = blockIdx.y;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid >> 5;     // 0..3
    unsigned int lane = tid & 31u;

    // Each warp handles one query row in this Br=4 block.
    unsigned int q_idx_in_batch = q_tile * FA2_BR + warp_id;
    bool active = (q_idx_in_batch < batch);

    // GQA mapping
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Per-row causal boundary -- the entire KV loop stops here.
    // We still walk one diagonal tile that contains positions <= q_pos plus
    // some positions > q_pos that get masked to -inf inside the tile.
    unsigned int q_pos = fa2_kv_upper_for_q(pos_start, q_idx_in_batch);
    unsigned int kv_upper_excl = active ? q_pos + 1u : 0u;
    unsigned int num_kv_tiles = (kv_upper_excl + FA2_BC - 1) / FA2_BC;

    // Determine block-skip: the union over all 4 warps in the block.
    // We share the LAST active warp's bound so all warps loop the same
    // number of tiles (avoids warp-divergent loop iterations). Inactive
    // warps must still participate in `__syncthreads()` calls, so their
    // bound is 0 and their per-element loads short-circuit.
    //
    // The block-skip is the max across the warps' kv_upper_excl values:
    // tiles above this index are guaranteed to be fully masked for every
    // active warp in this block.
    //
    // We can compute this with a warp-wide max reduction over lane 0 of
    // each warp, but the easier path is: each warp's kv_upper_excl is
    // monotonically increasing in warp_id (since q_idx_in_batch =
    // q_tile * FA2_BR + warp_id). So the LAST active warp owns the
    // maximum.
    unsigned int last_active_warp = 3u;
    while (last_active_warp > 0u &&
           q_tile * FA2_BR + last_active_warp >= batch) {
        last_active_warp -= 1u;
    }
    unsigned int block_q_pos =
        fa2_kv_upper_for_q(pos_start, q_tile * FA2_BR + last_active_warp);
    unsigned int block_kv_upper_excl = block_q_pos + 1u;
    unsigned int block_num_tiles = (block_kv_upper_excl + FA2_BC - 1) / FA2_BC;

    // Each warp still iterates its own causal bound -- but the block-level
    // sync requires the SAME tile count for all warps. We loop the block
    // max, and inside the warp body, treat tiles past this warp's own
    // boundary as no-ops.
    unsigned int q_dim = num_heads * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* q_head_ptr = Q + (unsigned long long)q_idx_in_batch * q_dim + head * head_dim;
    float* o_head_ptr = O + (unsigned long long)q_idx_in_batch * q_dim + head * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Shared memory: Q row per warp, then S tile per warp.
    extern __shared__ float fa2_smem[];
    float* q_shmem = fa2_smem + warp_id * head_dim;
    float* s_tile  = fa2_smem + FA2_BR * head_dim + warp_id * FA2_BC;

    // Load this warp's Q row.
    if (active) {
        for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
            q_shmem[d] = q_head_ptr[d];
        }
    }
    // Sync at the block level because we may rely on shared memory below
    // and other warps' Q rows are populated separately.
    __syncthreads();

    // Initialize output to zero -- this is the FA2 running accumulator.
    if (active) {
        for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
            o_head_ptr[d] = 0.0f;
        }
    }

    float m_prev = -3.402823466e+38f;
    float l_prev = 0.0f;

    // Walk tiles. Block-skip threshold: block_num_tiles caps the loop, but
    // each warp's individual cutoff (num_kv_tiles) further short-circuits.
    for (unsigned int tile = 0; tile < block_num_tiles; tile++) {
        unsigned int tile_start = tile * FA2_BC;

        // Per-warp early exit: this warp has already passed its causal bound.
        bool warp_in_range = active && (tile < num_kv_tiles);
        unsigned int tile_end = tile_start + FA2_BC;
        unsigned int my_upper = active ? kv_upper_excl : 0u;
        if (tile_end > my_upper) tile_end = my_upper;
        unsigned int tile_len = warp_in_range ? (tile_end - tile_start) : 0u;

        // ---- Phase A: scores ----
        // FA2_BC = 64, warp size = 32 -> each lane handles 2 positions per
        // tile. Inactive lanes/warps write -inf so they do not affect the
        // tile-max reduction.
        for (unsigned int j = lane; j < FA2_BC; j += FA2_WARP_SIZE) {
            float dot = 0.0f;
            if (j < tile_len) {
                unsigned int kv_pos = tile_start + j;
                const float* k_vec = k_base + kv_pos * head_dim;
                // float4 path when head_dim % 4 == 0 (true for all production
                // models: 64/128/256). Fall back to scalar otherwise.
                if ((head_dim & 3u) == 0u) {
                    unsigned int hd4 = head_dim >> 2;
                    const float4* q4 = reinterpret_cast<const float4*>(q_shmem);
                    const float4* k4 = reinterpret_cast<const float4*>(k_vec);
                    for (unsigned int d4 = 0; d4 < hd4; d4++) {
                        float4 q = q4[d4];
                        float4 k = k4[d4];
                        dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
                    }
                } else {
                    for (unsigned int d = 0; d < head_dim; d++) {
                        dot += q_shmem[d] * k_vec[d];
                    }
                }
                dot *= scale;
            } else {
                dot = -3.402823466e+38f;
            }
            s_tile[j] = dot;
        }
        __syncwarp(0xffffffff);

        // ---- Phase B: tile max + exp + sum (warp reduction) ----
        float local_max = -3.402823466e+38f;
        if (warp_in_range) {
            for (unsigned int j = lane; j < tile_len; j += FA2_WARP_SIZE) {
                local_max = fmaxf(local_max, s_tile[j]);
            }
        }
        float tile_max = fa2_warp_max(local_max);
        // Broadcast guard: if every score is -inf (e.g., the warp is past its
        // causal bound and tile_len==0), tile_max is still -inf and we must
        // skip the exp/rescale to avoid producing NaN.
        bool tile_has_content = warp_in_range && tile_len > 0u
            && tile_max > -3.402823466e+38f;
        float m_new = tile_has_content ? fmaxf(m_prev, tile_max) : m_prev;

        // Phase B2: exp(s - m_new) into s_tile.
        if (warp_in_range) {
            for (unsigned int j = lane; j < FA2_BC; j += FA2_WARP_SIZE) {
                float p = 0.0f;
                if (j < tile_len) {
                    p = expf(s_tile[j] - m_new);
                }
                s_tile[j] = p;
            }
        }
        __syncwarp(0xffffffff);

        // Phase B3: sum exp
        float local_sum = 0.0f;
        if (warp_in_range) {
            for (unsigned int j = lane; j < tile_len; j += FA2_WARP_SIZE) {
                local_sum += s_tile[j];
            }
        }
        float tile_sum = fa2_warp_sum(local_sum);

        // Phase B4: rescale running O and accumulate P @ V into o_head_ptr.
        float rescale = tile_has_content ? expf(m_prev - m_new) : 1.0f;
        float l_new = tile_has_content
            ? (rescale * l_prev + tile_sum)
            : l_prev;

        if (warp_in_range) {
            for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
                float o_val = o_head_ptr[d] * rescale;
                float pv = 0.0f;
                for (unsigned int j = 0; j < tile_len; j++) {
                    pv += s_tile[j] * v_base[(tile_start + j) * head_dim + d];
                }
                o_head_ptr[d] = o_val + pv;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // ---- Final normalisation: O = O / l ----
    if (active && l_prev > 0.0f) {
        float inv_l = 1.0f / l_prev;
        for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
            o_head_ptr[d] *= inv_l;
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K partial kernel
//
// Each block processes ONE Q row, ONE head, and ONE [kv_start, kv_end) slice.
// Output is the partial (O, m, l) tuple for that slice. The reduce kernel
// merges across splits.
//
// Grid:  (num_heads, batch, num_splits)
// Block: (FA2_WARP_SIZE, 1, 1)   --  single warp per block
//
// The kernel uses the warp-private streaming softmax exactly like the
// single-kernel path, but its KV range is the intersection of the global
// causal bound with the (kv_start, kv_end) slice assigned by the launcher.
// ---------------------------------------------------------------------------

extern "C" __global__ void flash_attention_fa2_splitk_partial(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_partial,       // [num_splits, batch, num_heads, head_dim]
    float* __restrict__ m_partial,       // [num_splits, batch, num_heads]
    float* __restrict__ l_partial,       // [num_splits, batch, num_heads]
    unsigned int batch,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int pos_start,
    unsigned int max_seq_len,
    float scale,
    unsigned int split_size,             // KV positions per split
    unsigned int num_splits)
{
    unsigned int head = blockIdx.x;
    unsigned int q_idx = blockIdx.y;
    unsigned int split = blockIdx.z;
    if (head >= num_heads || q_idx >= batch || split >= num_splits) return;

    unsigned int lane = threadIdx.x;

    // GQA mapping
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Causal upper bound for this query.
    unsigned int q_pos = fa2_kv_upper_for_q(pos_start, q_idx);
    unsigned int causal_excl = q_pos + 1u;

    // KV slice assigned to this split.
    unsigned int kv_start = split * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > causal_excl) kv_end = causal_excl;

    unsigned int q_dim = num_heads * head_dim;
    unsigned int kv_stride = max_seq_len * head_dim;
    const float* q_head_ptr = Q + (unsigned long long)q_idx * q_dim + head * head_dim;
    const float* k_base = K + (unsigned long long)kv_h * kv_stride;
    const float* v_base = V + (unsigned long long)kv_h * kv_stride;

    // Per-block output is [num_splits, batch, num_heads, head_dim] in
    // (split, q_idx, head, d) major order, so the partial pointer is:
    unsigned long long partial_o_offset =
        (((unsigned long long)split * batch + q_idx) * num_heads + head)
        * head_dim;
    float* o_part = O_partial + partial_o_offset;
    unsigned long long partial_ml_offset =
        ((unsigned long long)split * batch + q_idx) * num_heads + head;

    // Empty slice (this split lies entirely past the causal bound or it
    // starts past kv_end): write the identity (zero output, -inf max, zero
    // sum) so the reduce kernel handles it cleanly.
    if (kv_end <= kv_start) {
        if (lane == 0u) {
            m_partial[partial_ml_offset] = -3.402823466e+38f;
            l_partial[partial_ml_offset] = 0.0f;
        }
        for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
            o_part[d] = 0.0f;
        }
        return;
    }

    // Shared memory: q_shmem [head_dim] then s_tile [FA2_BC]
    extern __shared__ float fa2_split_smem[];
    float* q_shmem = fa2_split_smem;
    float* s_tile  = fa2_split_smem + head_dim;

    for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
        q_shmem[d] = q_head_ptr[d];
        o_part[d] = 0.0f;
    }
    __syncwarp(0xffffffff);

    float m_prev = -3.402823466e+38f;
    float l_prev = 0.0f;

    unsigned int slice_len = kv_end - kv_start;
    unsigned int num_tiles = (slice_len + FA2_BC - 1) / FA2_BC;

    for (unsigned int tile = 0; tile < num_tiles; tile++) {
        unsigned int tile_start_local = tile * FA2_BC;
        unsigned int tile_end_local = tile_start_local + FA2_BC;
        if (tile_end_local > slice_len) tile_end_local = slice_len;
        unsigned int tile_len = tile_end_local - tile_start_local;
        unsigned int tile_start_global = kv_start + tile_start_local;

        // Compute scores
        for (unsigned int j = lane; j < FA2_BC; j += FA2_WARP_SIZE) {
            float dot;
            if (j < tile_len) {
                unsigned int kv_pos = tile_start_global + j;
                const float* k_vec = k_base + kv_pos * head_dim;
                dot = 0.0f;
                if ((head_dim & 3u) == 0u) {
                    unsigned int hd4 = head_dim >> 2;
                    const float4* q4 = reinterpret_cast<const float4*>(q_shmem);
                    const float4* k4 = reinterpret_cast<const float4*>(k_vec);
                    for (unsigned int d4 = 0; d4 < hd4; d4++) {
                        float4 q = q4[d4];
                        float4 k = k4[d4];
                        dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
                    }
                } else {
                    for (unsigned int d = 0; d < head_dim; d++) {
                        dot += q_shmem[d] * k_vec[d];
                    }
                }
                dot *= scale;
            } else {
                dot = -3.402823466e+38f;
            }
            s_tile[j] = dot;
        }
        __syncwarp(0xffffffff);

        // Tile-max
        float local_max = -3.402823466e+38f;
        for (unsigned int j = lane; j < tile_len; j += FA2_WARP_SIZE) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        float tile_max = fa2_warp_max(local_max);
        float m_new = fmaxf(m_prev, tile_max);

        // exp into s_tile
        for (unsigned int j = lane; j < FA2_BC; j += FA2_WARP_SIZE) {
            float p = 0.0f;
            if (j < tile_len) {
                p = expf(s_tile[j] - m_new);
            }
            s_tile[j] = p;
        }
        __syncwarp(0xffffffff);

        float local_sum = 0.0f;
        for (unsigned int j = lane; j < tile_len; j += FA2_WARP_SIZE) {
            local_sum += s_tile[j];
        }
        float tile_sum = fa2_warp_sum(local_sum);

        float rescale = expf(m_prev - m_new);
        float l_new = rescale * l_prev + tile_sum;

        for (unsigned int d = lane; d < head_dim; d += FA2_WARP_SIZE) {
            float o_val = o_part[d] * rescale;
            float pv = 0.0f;
            for (unsigned int j = 0; j < tile_len; j++) {
                pv += s_tile[j] * v_base[(tile_start_global + j) * head_dim + d];
            }
            o_part[d] = o_val + pv;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    if (lane == 0u) {
        m_partial[partial_ml_offset] = m_prev;
        l_partial[partial_ml_offset] = l_prev;
    }
}

// ---------------------------------------------------------------------------
// Split-K reduce kernel
//
// Merges per-split (O, m, l) tuples into the final output using the FA2
// online-softmax combine rule. One block per (q_idx, head); each thread
// handles a subset of head_dim.
//
// Contract with the partial kernel:
//
//   Per slice s, the partial kernel writes:
//     O_partial[s, q, h, :]  = sum_j exp(s_{s,j} - m_s) * V_j     (numerator)
//     m_partial[s, q, h]     = m_s = max_j s_{s,j}
//     l_partial[s, q, h]     = sum_j exp(s_{s,j} - m_s)            (denominator)
//
//   Note that the streaming softmax inside the partial kernel rescales O on
//   every tile, so the FINAL state of `o_part[d]` is the un-normalised
//   numerator for the slice. We do NOT divide by l_s inside the partial
//   kernel.
//
//   The merge under FA2 then gives, with m* = max_s m_s:
//     O_global = sum_s exp(m_s - m*) * O_s / sum_s exp(m_s - m*) * l_s
//
//   Both numerator and denominator are computed across splits in this
//   reduce kernel; the final output is `numerator / denominator` element-
//   wise per head_dim coordinate.
//
// Grid:  (num_heads, batch, 1)
// Block: (256, 1, 1)
// ---------------------------------------------------------------------------

extern "C" __global__ void flash_attention_fa2_splitk_reduce(
    const float* __restrict__ O_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    float* __restrict__ O,
    unsigned int batch,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int num_splits)
{
    unsigned int head = blockIdx.x;
    unsigned int q_idx = blockIdx.y;
    if (head >= num_heads || q_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int q_dim = num_heads * head_dim;
    float* o_out = O + (unsigned long long)q_idx * q_dim + head * head_dim;

    // Shared memory layout:
    //   m_arr      [num_splits]   per-split max (loaded from m_partial)
    //   l_arr      [num_splits]   per-split sum (loaded from l_partial)
    //   rescale    [num_splits]   exp(m_s - m_global) precomputed
    //   scratch    [1]            stashes global_sum so all threads see it
    //
    // Host must pass shared_mem_bytes = (3 * num_splits + 1) * 4.
    extern __shared__ float reduce_smem[];
    float* m_arr = reduce_smem;
    float* l_arr = reduce_smem + num_splits;
    float* rescale_arr = reduce_smem + 2u * num_splits;
    float* scratch = reduce_smem + 3u * num_splits;   // scratch[0] = global_sum

    unsigned long long ml_base = (unsigned long long)q_idx * num_heads + head;
    unsigned long long ml_stride = (unsigned long long)batch * num_heads;

    for (unsigned int s = tid; s < num_splits; s += block_size) {
        m_arr[s] = m_partial[s * ml_stride + ml_base];
        l_arr[s] = l_partial[s * ml_stride + ml_base];
    }
    __syncthreads();

    // Phase 1: find global max + compute rescale factors + global sum.
    // num_splits is small (typically <= 64); tid 0 scans the array.
    if (tid == 0u) {
        float global_max = -3.402823466e+38f;
        for (unsigned int s = 0; s < num_splits; s++) {
            if (l_arr[s] > 0.0f && m_arr[s] > global_max) {
                global_max = m_arr[s];
            }
        }
        float global_sum = 0.0f;
        for (unsigned int s = 0; s < num_splits; s++) {
            if (l_arr[s] > 0.0f) {
                float r = expf(m_arr[s] - global_max);
                rescale_arr[s] = r;
                global_sum += r * l_arr[s];
            } else {
                rescale_arr[s] = 0.0f;
            }
        }
        scratch[0] = global_sum;
    }
    __syncthreads();
    float global_sum = scratch[0];
    float inv_global = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Phase 2: weighted sum of partial numerators / global_sum.
    // O_partial layout: [num_splits, batch, num_heads, head_dim] in
    // (s, q, h, d)-major.
    unsigned long long partial_base =
        ((unsigned long long)q_idx * num_heads + head) * head_dim;
    unsigned long long partial_stride =
        (unsigned long long)batch * num_heads * head_dim;

    for (unsigned int d = tid; d < head_dim; d += block_size) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < num_splits; s++) {
            float r = rescale_arr[s];
            if (r > 0.0f) {
                acc += r * O_partial[s * partial_stride + partial_base + d];
            }
        }
        o_out[d] = acc * inv_global;
    }
}
