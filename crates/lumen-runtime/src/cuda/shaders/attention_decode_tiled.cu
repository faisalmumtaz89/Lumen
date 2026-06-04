// Tiled streaming-softmax decode attention kernel (CUDA, SM 70+).
//
// Closes: the single-block `attention_decode`
// kernel (`attention.cu`) materialises the full per-token score array in
// dynamic shared memory and is therefore capped at seq_len <= 40_950 on
// SM 8.0 (163 KB extended shmem ceiling per `decode.rs::attention_shared_bytes`).
//
// This kernel removes that ceiling by streaming the softmax over fixed-size
// KV tiles using Dao 2022 (FlashAttention-2) online-softmax mechanics. Per-CTA
// shared memory is constant in `seq_len` -- it scales with `T_C` (tile width)
// and `head_dim` only:
//
//   shmem = (8 + T_C + head_dim) * 4 bytes  ~= 1.6 KB at T_C=128, head_dim=256
//
// well under the 48 KB default shmem cap on any SM 6.0+ device. No
// `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)` opt-in is
// required.
//
// ---------------------------------------------------------------------------
// Algorithm (Dao 2022 online softmax, decode shape M=1):
//
//   per_thread state (held in registers across tiles):
//     m_prev = -INF                    // running max of pre-softmax scores
//     l_prev = 0                       // running normaliser
//     o_acc[head_dim / block_size]     // un-normalised output numerator,
//                                         each lane owns head_dim/block_size
//                                         output dimensions
//
//   shmem state:
//     q_row[head_dim]                  // Q for this head, cooperatively loaded
//     s_tile[T_C]                      // tile scores (post-exp during phase C)
//     partial[8]                       // warp-reduction scratch
//
//   for tile = 0..num_tiles:
//     tile_start = tile * T_C
//     tile_len   = min(T_C, seq_len - tile_start)
//
//     // Phase A: scores for this tile (one lane per position when T_C ==
//     // block_size, else strided).
//     for j in tile (strided across lanes):
//       if j < tile_len:
//         k_vec = K_cache[kv_h, tile_start + j, :]   // float4 vectorised
//         dot   = sum(q_row * k_vec) * scale
//         s_tile[j] = dot
//       else:
//         s_tile[j] = -INF
//
//     // Phase B: tile max (block-reduce; partial[8] is warp scratch)
//     tile_max = block_reduce_max(...)
//     m_new    = max(m_prev, tile_max)
//
//     // Phase C: streaming softmax update
//     rescale  = exp(m_prev - m_new)
//     for j < tile_len:
//       s_tile[j] = exp(s_tile[j] - m_new)
//     tile_sum = block_reduce_sum(s_tile[0..tile_len])
//     l_new    = rescale * l_prev + tile_sum
//
//     // Phase D: rescale O_prev and accumulate P @ V_tile (float4 V loads)
//     for d_owned (head_dim/block_size dims per lane):
//       pv = 0
//       for j < tile_len:
//         pv += s_tile[j] * V_cache[kv_h, tile_start + j, d_owned]
//       o_acc[slot] = rescale * o_acc[slot] + pv
//
//     m_prev = m_new
//     l_prev = l_new
//
//   // Final: O = O / l
//   for d_owned:
//     out[head, d_owned] = o_acc[slot] / l_prev
//
// ---------------------------------------------------------------------------
// Grid + block:
//   grid_dim  = (num_heads, 1, 1)   // one CTA per query head
//   block_dim = (BLOCK_DIM, 1, 1)   // BLOCK_DIM = 128 = 4 warps
//
// Per-thread register footprint:
//   m_prev, l_prev                    : 2 floats
//   o_acc[head_dim / BLOCK_DIM]        : 2 floats at head_dim=256, BLOCK_DIM=128
//   (loop induction, partial scratch addressing)
//
// KV cache layout (head-first, F32):
//   K cache: [num_kv_heads, max_seq_len, head_dim]
//   V cache: [num_kv_heads, max_seq_len, head_dim]
//
// GQA: kv_h = head / (num_heads / num_kv_heads).
//
// ---------------------------------------------------------------------------
// Numerical correctness:
//
// The streaming-softmax algorithm is mathematically equivalent to the
// single-pass softmax up to floating-point reassociation. The FA2 prefill
// path uses the identical mechanics and is byte-exact within `< 1e-4`
// per element vs the single-pass kernel on the production Qwen3.5-9B
// shapes.
//
// Numerical-stability invariants honoured:
//   1. Subtract max before exp (line: `expf(s - m_new)` after `m_new` is
//      computed across the tile + carry).
//   2. Sentinel `-FLT_MAX` for out-of-range positions in the partial last
//      tile (so they cannot influence the tile_max reduction).
//   3. Guard against `l_prev == 0` at final normalisation (degenerate
//      empty input, defensive only -- decode always has seq_len >= 1).
//
// NVRTC-compatible: no system includes, extern "C" linkage. Compiled at
// runtime via the existing `compile_and_load` pipeline in `ffi.rs`.

#define T_C        128u    // KV positions per tile (compile-time constant; recompile to tune)
#define BLOCK_DIM  128u    // threads per CTA (4 warps)

// Invariant: head_dim must be divisible by BLOCK_DIM for the o_accslot
// addressing to be exact (each lane owns head_dim/BLOCK_DIM output dims).
// Qwen3.5-9B uses head_dim = 256; 256 % 128 == 0.
// Pass 3 refinement #1.)
//
// We cannot use `static_assert` here because `head_dim` is a runtime kernel
// argument, not a template parameter. The invariant is enforced at the
// HOST side in `decode.rs::attention_decode_variant` / kernel launch path:
// a head_dim violation surfaces as a buffer-size mismatch at launch time.
// (NVRTC's C++14 mode would let us static_assert on `BLOCK_DIM` alone, which
// is a tautology -- skipped.) The corresponding host-side guard lives in
// the launcher in `prefill.rs::launch_attention_decode_tiled`.

#define NEG_INF (-3.402823466e+38f)

// Warp-level max reduction (butterfly shuffle, full lane mask).
__device__ __forceinline__ float tiled_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

// Warp-level sum reduction (butterfly shuffle, full lane mask).
__device__ __forceinline__ float tiled_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Block-wide max reduction via shared scratch. `partial[]` MUST have at
// least `num_warps = ceil(block_size / 32)` floats. At BLOCK_DIM=128 we
// have num_warps=4, but we conservatively allocate 8 to share the layout
// with the single-block kernel and to allow BLOCK_DIM=256 retuning later.
__device__ __forceinline__ float tiled_block_reduce_max(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = tiled_warp_max(val);
    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : NEG_INF;
        v = tiled_warp_max(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

// Block-wide sum reduction via shared scratch.
__device__ __forceinline__ float tiled_block_reduce_sum(
    float val,
    volatile float* partial,
    unsigned int tid,
    unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = tiled_warp_sum(val);
    if (lane == 0u) {
        partial[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = tiled_warp_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    __syncthreads();

    return partial[0];
}

// Cooperative dot product: `dot(q_row, K[kv_h, pos, :])` * `scale`.
// Float4-vectorised path when head_dim is a multiple of 4 (always true for
// production shapes 64/128/256). Within the kernel, this is called from one
// lane per KV position, so the inner loop walks head_dim on the SINGLE lane
// (vs the single-block kernel which strides head_dim across threads). The
// trade-off is acceptable because the outer loop's parallelism is already
// `T_C = block_size` lanes per tile.
__device__ __forceinline__ float tiled_qk_dot(
    const float* __restrict__ q_row,
    const float* __restrict__ k_vec,
    unsigned int head_dim,
    float scale)
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

// ---------------------------------------------------------------------------
// Kernel: attention_decode_tiled
//
// Single CTA per query head. M=1 (one decode query token). Streams the
// softmax over `ceil(seq_len / T_C)` tiles using Dao 2022 online softmax.
//
// Arguments mirror `attention_decode` in `attention.cu` exactly so the gate
// dispatch at the host site can swap kernels without changing call signature.
// ---------------------------------------------------------------------------

extern "C" __global__ void attention_decode_tiled(
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
    unsigned int block_size = blockDim.x;   // = BLOCK_DIM at launch

    // Degenerate seq_len = 0: no work to do; zero the output and exit. (Defensive
    // -- production decode always has seq_len >= 1 because the KV cache is
    // appended BEFORE the attention call. audit Subject (B) Pass 1
    // covers the `seq_len = 0` gate case.)
    if (seq_len == 0u) {
        for (unsigned int d = tid; d < head_dim; d += block_size) {
            attn_out[head * head_dim + d] = 0.0f;
        }
        return;
    }

    // GQA mapping: multiple Q heads share the same KV head.
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    // Base pointers for this head.
    const float* q_head = q + head * head_dim;
    float* out_head = attn_out + head * head_dim;

    // KV cache base for this KV head (head-first layout, u64 to avoid overflow
    // at max_seq_len * head_dim products that exceed u32 in long-context).
    unsigned long long kv_base = (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)head_dim;

    // Shared memory layout (constant in seq_len):
    //   [0..7]:           partial[8]   -- warp-reduction scratch
    //   [8..8+head_dim):  q_row[head_dim]
    //   [8+head_dim..]:   s_tile[T_C]
    //
    // Host must set shared_mem_bytes = (8 + head_dim + T_C) * sizeof(float).
    extern __shared__ float smem[];
    volatile float* partial = smem;
    float* q_row = smem + 8;
    float* s_tile = smem + 8 + head_dim;

    // ---- Phase 0: cooperatively load Q row into shmem ----
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        q_row[d] = q_head[d];
    }
    __syncthreads();

    // Per-thread running softmax state.
    float m_prev = NEG_INF;
    float l_prev = 0.0f;

    // Per-thread output accumulator: each lane owns `head_dim / block_size`
    // output dimensions (lane `tid` owns d = tid, tid + block_size, ...).
    //
    // We cannot declare a runtime-sized register array, so we allow up to
    // ATTN_DECODE_TILED_MAX_SLOTS = 8 slots per lane. At BLOCK_DIM=128 this
    // covers head_dim up to 128 * 8 = 1024 (well past any plausible head_dim
    // a transformer would use). Slot use is bounded by `num_slots` and reads
    // past num_slots are skipped.
    constexpr unsigned int MAX_SLOTS = 8u;
    float o_acc[MAX_SLOTS];
#pragma unroll
    for (unsigned int s = 0; s < MAX_SLOTS; s++) {
        o_acc[s] = 0.0f;
    }
    unsigned int num_slots = (head_dim + block_size - 1u) / block_size;
    // (At head_dim=256, block_size=128 -> num_slots = 2.
    //  At head_dim=128, block_size=128 -> num_slots = 1.
    //  At head_dim=64,  block_size=128 -> num_slots = 1 (slot 0 used by lanes
    //                                                     [0..head_dim), others
    //                                                     contribute 0).)

    // Number of tiles to walk. ceil(seq_len / T_C).
    unsigned int num_tiles = (seq_len + T_C - 1u) / T_C;

    // ----------------------------------------------------------------------
    // Outer loop: stream over KV tiles
    // ----------------------------------------------------------------------
    for (unsigned int tile = 0; tile < num_tiles; tile++) {
        unsigned int tile_start = tile * T_C;
        unsigned int tile_end_raw = tile_start + T_C;
        unsigned int tile_end = (tile_end_raw < seq_len) ? tile_end_raw : seq_len;
        unsigned int tile_len = tile_end - tile_start;

        // ---- Phase A: scores for this tile ----
        // At BLOCK_DIM=128 and T_C=128, each lane handles exactly one position.
        // If a future T_C != BLOCK_DIM is chosen, this stride covers both
        // cases correctly.
        for (unsigned int j = tid; j < T_C; j += block_size) {
            if (j < tile_len) {
                unsigned int pos = tile_start + j;
                const float* k_vec = k_cache + kv_base + (unsigned long long)pos * (unsigned long long)head_dim;
                s_tile[j] = tiled_qk_dot(q_row, k_vec, head_dim, scale);
            } else {
                // Out-of-range positions in the partial last tile: sentinel
                // -INF so they do not influence tile_max.
                s_tile[j] = NEG_INF;
            }
        }
        __syncthreads();

        // ---- Phase B: tile max (block reduction) ----
        float local_max = NEG_INF;
        for (unsigned int j = tid; j < T_C; j += block_size) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        float tile_max = tiled_block_reduce_max(local_max, partial, tid, block_size);

        // Online softmax update (per-thread; identical across all lanes
        // because tile_max is broadcast from the block reduction).
        float m_new = fmaxf(m_prev, tile_max);
        float rescale = expf(m_prev - m_new);

        // ---- Phase C: exp(s - m_new) into s_tile + tile_sum reduction ----
        for (unsigned int j = tid; j < T_C; j += block_size) {
            if (j < tile_len) {
                s_tile[j] = expf(s_tile[j] - m_new);
            } else {
                s_tile[j] = 0.0f;  // out-of-range: no contribution to tile_sum
            }
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (unsigned int j = tid; j < T_C; j += block_size) {
            local_sum += s_tile[j];
        }
        float tile_sum = tiled_block_reduce_sum(local_sum, partial, tid, block_size);

        float l_new = rescale * l_prev + tile_sum;

        // ---- Phase D: rescale O_prev and accumulate P @ V_tile ----
        // Each lane owns dimensions d = tid + slot * block_size for slot in
        // [0, num_slots). Float4 path is awkward here because the lane stride
        // (block_size = 128) is not a multiple of 4 in a vec-friendly way; we
        // use the scalar V path (matches the V-side of FA2 prefill at
        // `flash_attention_fa2.cu:294-302`).
#pragma unroll
        for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
            if (slot >= num_slots) break;
            unsigned int d = tid + slot * block_size;
            if (d < head_dim) {
                float pv = 0.0f;
                for (unsigned int j = 0; j < tile_len; j++) {
                    unsigned int pos = tile_start + j;
                    float v_dj = v_cache[kv_base + (unsigned long long)pos * (unsigned long long)head_dim + (unsigned long long)d];
                    pv += s_tile[j] * v_dj;
                }
                o_acc[slot] = rescale * o_acc[slot] + pv;
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        // s_tile will be overwritten by the next tile's Phase A; sync to
        // ensure all lanes have finished reading s_tile in their Phase D
        // inner loop before any lane starts writing again.
        __syncthreads();
    }

    // ---- Final: normalise and write output ----
    // Defensive guard: if l_prev == 0 (degenerate, e.g. all -inf scores in a
    // seq_len = 0 path that the early-return above already handles), we write
    // zeros to avoid producing NaN. In the normal path l_prev > 0 always.
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

#pragma unroll
    for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
        if (slot >= num_slots) break;
        unsigned int d = tid + slot * block_size;
        if (d < head_dim) {
            out_head[d] = o_acc[slot] * inv_l;
        }
    }
}
