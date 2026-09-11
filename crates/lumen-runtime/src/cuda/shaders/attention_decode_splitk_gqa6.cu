// ==========================================================================
// GQA-shared split-K decode attention, specialised for 6 query heads per KV
// head and head_dim 256 (Qwen3.8-27B full attention: 24 Q heads, 4 KV heads).
//
// Same (m, l, o) partial contract as attention_decode_splitk.cu, and the same
// two-pass shape, but a different work decomposition:
//
//   attention_decode_splitk.cu       one CTA per (QUERY head, chunk).
//                                    Every K row and every V row of the chunk
//                                    is fetched once per query head — six
//                                    times over for a 6:1 GQA group. The PV
//                                    phase reads V one scalar at a time
//                                    (v_cache[pos * head_dim + d]).
//
//   this file                        one CTA per (KV head, chunk). Each K row
//                                    and V row is fetched ONCE and serves all
//                                    six query heads of the group, and every
//                                    Q, K and V read is a 16-byte load. (The
//                                    partial's scratch stores and the merge's
//                                    reads of them stay scalar; they are a
//                                    fifth of the traffic.)
//
// Decomposition of one CTA (128 threads = 4 warps), per 16-key tile:
//
//   stage    Q for the six heads (6 KiB) and the first V tile (16 KiB F32,
//            8 KiB half) are copied into shared with 16-byte loads, both
//            issued before the first barrier so the two streams overlap;
//            each later tile's V is staged before that tile's dot products.
//   QK       a warp owns one key at a time; lanes own dimensions. Lane l
//            holds K[pos][4l .. 4l+4) and K[pos][128+4l .. 128+4l+4) — two
//            float4 (widened from halves on a half store) — in registers while
//            all six head scores are formed from them, each by a warp
//            shuffle-xor tree. Q is reloaded from shared per tile (2 float4
//            per head per lane) so its registers live only through the phase.
//   softmax  scores are [6][16] in shared; one warp per query head, four warps
//            covering six heads in two rounds; the tile's max joins the
//            running max and the tile's sum the running sum (rescaled).
//   PV       thread t owns dims t and t+128 and keeps 12 accumulators
//            (2 dims x 6 heads), rescaled once per tile and walking the staged
//            V tile in ascending key order, reusing each V value across the
//            group.
//
// A CTA walks one tile below the one-tile bound (the form every context took
// before the loop, bit-identical) and a contiguous run of whole tiles above
// it, so the split count — and with it the scratch — is bounded by the host's
// target at any context.
//
// Shared memory: 6*256 (Q) + 16*256 (V; halves on a half store) + 6*16
// (scores) + 6 m + 6 l + 6 rescale floats: 22'984 B (F32) / 14'792 B (half),
// under the 48 KiB default dynamic-shared cap, so no opt-in is needed.
//
// PRECISION: F32 accumulation throughout (K and V widened exactly from halves
// on a half store), the same expf, no atomics and no fast-math. The QK
// reduction tree, the partition and the PV order all reassociate the sum
// differently from attention_decode_splitk.cu, so the two routes are a
// quality-equivalent NEAR-TIE rather than bit-identical — the same relationship
// the split-K pair has with the tiled kernel. Deterministic for a fixed
// (S, partition): every reduction is a fixed tree and every accumulation walks
// ascending indices. The integration suite holds both partitions within
// 2e-6 of an F64 reference at every context from 1 to 32,768 keys and prints
// the observed maximum per length, so the headroom is a recorded number
// rather than a figure in this comment.
//
// The empty-range arm writes (m = -inf, l = 0, o = 0) and the merge drops it,
// exactly as in attention_decode_splitk.cu. It is DEFENSIVE: neither partition
// the host derives leaves a CTA empty. The arm keeps a caller that oversplits
// — a test, or a future policy — exact instead of merely lucky.
//
// Scratch layout (device buffers, sized by the host), identical to the
// sibling pair's but with the larger split count this decomposition uses:
//   m_part [num_heads * S]              F32
//   l_part [num_heads * S]              F32
//   o_part [num_heads * S * head_dim]   F32
//
// Requires (enforced by the host gate
// `attention_decode_splitk_gqa6_supports`): num_heads / num_kv_heads == 6,
// head_dim == 256, and the store the entry point is for (F32 words for _f32,
// half bit patterns for _f16).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define GQA6_NEG_INF (-3.402823466e+38f)
#define GQA6_HD       256u
#define GQA6_MERGE_LANES 8u       // independent numerator accumulators in the merge (the tree below sums exactly eight)
#define GQA6_G        6u
#define GQA6_BLOCK    128u
#define GQA6_WARPS    (GQA6_BLOCK / 32u)

__device__ __forceinline__ float gqa6_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ float gqa6_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Four-warp block reduction over a 4-float shared scratch. Used by the merge.
__device__ __forceinline__ float gqa6_block_max(float v, volatile float* scr, unsigned int tid) {
    v = gqa6_warp_max(v);
    if ((tid & 31u) == 0u) scr[tid >> 5] = v;
    __syncthreads();
    float r = fmaxf(fmaxf(scr[0], scr[1]), fmaxf(scr[2], scr[3]));
    __syncthreads();
    return r;
}

__device__ __forceinline__ float gqa6_block_sum(float v, volatile float* scr, unsigned int tid) {
    v = gqa6_warp_sum(v);
    if ((tid & 31u) == 0u) scr[tid >> 5] = v;
    __syncthreads();
    float r = (scr[0] + scr[1]) + (scr[2] + scr[3]);
    __syncthreads();
    return r;
}

#define GQA6_TILE     16u

__device__ __forceinline__ float gqa6_h2f(unsigned int h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)(h & 0xffffu)));
    return f;
}

// Four packed halves (two 32-bit words, low half first) -> float4.
__device__ __forceinline__ float4 gqa6_h4_to_f4(unsigned int lo, unsigned int hi) {
    float4 o;
    o.x = gqa6_h2f(lo);
    o.y = gqa6_h2f(lo >> 16);
    o.z = gqa6_h2f(hi);
    o.w = gqa6_h2f(hi >> 16);
    return o;
}

// --------------------------------------------------------------------------
// The partial pass: a TILE LOOP with a bounded split count, so one kernel
// serves any context the cache holds with fixed scratch and no route switch.
//
// Geometry: grid (S, num_kv_heads).
//   partition = 0  the one-tile form, S = ceil(keys / 16): CTA `chunk` owns keys
//                  [chunk * span, min((chunk + 1) * span, keys)), span = ceil(keys / S)
//                  (at most 16 keys; the form every context took before the loop,
//                  kept bit-identical below the one-tile bound).
//   partition = 1  whole-tile balanced, S < N = ceil(keys / 16): CTA `chunk` owns
//                  tiles [floor(chunk * N / S), floor((chunk + 1) * N / S)) of 16 keys
//                  (the context's last tile may be shorter); no empty CTAs, at most
//                  one tile of difference between CTAs.
// Per tile: Q for the six heads in registers (reloaded from shared per tile so
// the registers live only through the dot products), a warp per key, the K row
// in registers, six dots per key in the fixed tree, scores [6][16] in shared,
// the V tile staged once, twelve accumulators per thread over the tile's keys
// in ascending order.
// Across tiles: the tiled kernel's recurrence: m' = max(m, m_t), rescale =
// exp(m - m'), p_j = exp(s_j - m'), acc = rescale * acc + sum p_j v_j,
// l = rescale * l + sum p_j; the running (m, l, rescale) per head live in
// shared memory (a register array indexed at run time demotes to local memory).
// On the first tile m = -inf, the rescale is 0 and no exp is spent: the tile's
// numbers are the one-tile form's exactly.
// Output: (m, l, o[256]) per head per CTA, the partial format the eight-lane
// merge below consumes.
// Occupancy: __launch_bounds__(128, 4) on both loops (a floor on resident
// CTAs, not a cap). On the engine's compile path — NVRTC at its default PTX
// target, then the driver's ptxas for the device — ptxas reports 96 registers
// for both loops, no stack frame, no spills (an offline nvcc -arch=sm_120 build
// reports 116 / 118; the numbers that count are the engine's). At 96 registers
// an SM holds five CTAs by registers: the F32 loop is held to four by its
// 22,984 B of shared, the half loop runs five (14,792 B). The retained one-tile
// partials compile to 72 registers on the same path and the half one runs six
// CTAs per SM (shared-limited at 14,768 B); bounding the half loop to six
// forces 80 registers with spills to local memory on this path and was
// measured slower at every context, so the loop keeps five.
// A tile whose span would exceed 16 keys calls __trap(), which aborts the
// launch (the driver reports the error at the next synchronisation; no
// partial reaches the merge). It is unreachable by construction: on the span
// partition the host keeps S = ceil(keys / 16), so a CTA's whole range is at
// most 16 keys, and on the whole-tile partition every iteration clips its
// tile to 16 keys whatever the CTA's range.
// --------------------------------------------------------------------------

// The CTA's key range.
__device__ __forceinline__ void gqa6_loop_range(
    unsigned int chunk, unsigned int seq_len, unsigned int num_chunks, unsigned int partition,
    unsigned int* k0, unsigned int* k1)
{
    if (partition == 0u) {
        const unsigned int span = (seq_len + num_chunks - 1u) / num_chunks;
        const unsigned int p0 = chunk * span;
        unsigned int p1 = p0 + span;
        if (p0 >= seq_len) { *k0 = 0u; *k1 = 0u; return; }
        if (p1 > seq_len) p1 = seq_len;
        *k0 = p0; *k1 = p1;
    } else {
        const unsigned long long n = ((unsigned long long)seq_len + GQA6_TILE - 1ull) / GQA6_TILE;
        const unsigned int t0 = (unsigned int)(((unsigned long long)chunk * n) / num_chunks);
        const unsigned int t1 = (unsigned int)(((unsigned long long)(chunk + 1u) * n) / num_chunks);
        unsigned int p0 = t0 * GQA6_TILE;
        unsigned int p1 = t1 * GQA6_TILE;
        if (p1 > seq_len) p1 = seq_len;
        if (p0 > p1) p0 = p1;
        *k0 = p0; *k1 = p1;
    }
}

// Stage `span` keys of V from position `p0` into the shared buffer `dst`
// (F32 floats, or half bit patterns when V_HALF): 16 bytes per thread per step.
#define GQA6_STAGE_V(V_HALF, dst, p0, span)                                                         \
    {                                                                                               \
        if (V_HALF) {                                                                               \
            const uint4* v8src = reinterpret_cast<const uint4*>(                                    \
                v_cache + kv_base + (unsigned long long)(p0) * (unsigned long long)GQA6_HD);        \
            uint4* v8dst = reinterpret_cast<uint4*>(dst);                                           \
            const unsigned int nv8 = (span) * (GQA6_HD / 8u);                                       \
            for (unsigned int i = tid; i < nv8; i += GQA6_BLOCK) v8dst[i] = v8src[i];               \
        } else {                                                                                    \
            const float4* v4src = reinterpret_cast<const float4*>(                                  \
                v_cache + kv_base + (unsigned long long)(p0) * (unsigned long long)GQA6_HD);        \
            float4* v4dst = reinterpret_cast<float4*>(dst);                                         \
            const unsigned int nv4 = (span) * (GQA6_HD / 4u);                                       \
            for (unsigned int i = tid; i < nv4; i += GQA6_BLOCK) v4dst[i] = v4src[i];               \
        }                                                                                           \
    }

// The body shared by the F32 and half variants. V_HALF selects the V staging.
#define GQA6_LOOP_BODY(V_HALF, KLOAD)                                                               \
    const unsigned int chunk = blockIdx.x;                                                          \
    const unsigned int kv_h  = blockIdx.y;                                                          \
    const unsigned int tid   = threadIdx.x;                                                         \
    const unsigned int lane  = tid & 31u;                                                           \
    const unsigned int warp  = tid >> 5;                                                            \
                                                                                                    \
    unsigned int k0, k1;                                                                            \
    gqa6_loop_range(chunk, seq_len, num_chunks, partition, &k0, &k1);                               \
                                                                                                    \
    /* Empty range: block-uniform, the whole CTA returns before any barrier. */                    \
    if (k1 <= k0) {                                                                                 \
        _Pragma("unroll")                                                                           \
        for (unsigned int g = 0; g < GQA6_G; g++) {                                                 \
            unsigned int head = kv_h * GQA6_G + g;                                                  \
            unsigned long long base =                                                               \
                ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;                          \
            o_part[base + tid] = 0.0f;                                                              \
            o_part[base + tid + GQA6_BLOCK] = 0.0f;                                                 \
        }                                                                                           \
        if (tid < GQA6_G) {                                                                         \
            unsigned int head = kv_h * GQA6_G + tid;                                                \
            m_part[head * num_chunks + chunk] = GQA6_NEG_INF;                                       \
            l_part[head * num_chunks + chunk] = 0.0f;                                               \
        }                                                                                           \
        return;                                                                                     \
    }                                                                                               \
                                                                                                    \
    extern __shared__ __align__(16) float smem[];                                                   \
    float* s_q = smem;                                                        /* [6][256] */        \
    const unsigned int v_floats = (V_HALF) ? (GQA6_TILE * GQA6_HD) / 2u : GQA6_TILE * GQA6_HD;      \
    float* s_v = s_q + GQA6_G * GQA6_HD;                                      /* [16][256] */       \
    const unsigned short* s_vh = reinterpret_cast<const unsigned short*>(s_v);                      \
    float* s_score = s_v + v_floats;                                          /* [6][16] */         \
    float* s_m = s_score + GQA6_G * GQA6_TILE;                                /* [6] running max */ \
    float* s_l = s_m + GQA6_G;                                                /* [6] running sum */ \
    float* s_resc = s_l + GQA6_G;                                             /* [6] rescale */     \
                                                                                                    \
    const unsigned long long kv_base =                                                              \
        (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)GQA6_HD;   \
                                                                                                    \
    /* Q and tile 0's V are issued before the first barrier so their loads overlap. */              \
    {                                                                                               \
        const float4* q4src =                                                                       \
            reinterpret_cast<const float4*>(q + (unsigned long long)kv_h * GQA6_G * GQA6_HD);       \
        float4* q4dst = reinterpret_cast<float4*>(s_q);                                             \
        for (unsigned int i = tid; i < (GQA6_G * GQA6_HD) / 4u; i += GQA6_BLOCK) {                  \
            q4dst[i] = q4src[i];                                                                    \
        }                                                                                           \
        unsigned int e0 = k0 + GQA6_TILE;                                                           \
        if (e0 > k1) e0 = k1;                                                                       \
        GQA6_STAGE_V(V_HALF, s_v, k0, e0 - k0);                                                     \
    }                                                                                               \
    __syncthreads();                                                                                \
                                                                                                    \
    /* Each thread holds its two dimensions of the numerator for each head;   */                   \
    /* the running max and sum per head are block-uniform and live in shared. */                   \
    float acc0[GQA6_G];                                                                             \
    float acc1[GQA6_G];                                                                             \
    _Pragma("unroll")                                                                               \
    for (unsigned int g = 0; g < GQA6_G; g++) {                                                     \
        acc0[g] = 0.0f;                                                                             \
        acc1[g] = 0.0f;                                                                             \
    }                                                                                               \
                                                                                                    \
    for (unsigned int p0 = k0; p0 < k1; p0 += GQA6_TILE) {                                          \
        unsigned int p1 = p0 + GQA6_TILE;                                                           \
        if (p1 > k1) p1 = k1;                                                                       \
        const unsigned int span = p1 - p0;                                                          \
        if (span > GQA6_TILE) { __trap(); }                                                         \
        const bool first = (p0 == k0);                                                              \
                                                                                                    \
        /* Later tiles stage their V here, before the dot products, so the     */                   \
        /* load overlaps them (tile 0's V came in with Q).                      */                  \
        if (!first) { GQA6_STAGE_V(V_HALF, s_v, p0, span); }                                        \
                                                                                                    \
        /* QK: a warp per key, the K row in registers across all six heads.  */                     \
        /* Q is reloaded from shared for each tile (12 shared float4 per lane) */                    \
        /* so its 48 registers live only through the dot products, not across */                    \
        /* the PV pass and the tile loop.                                       */                   \
        float4 qa[GQA6_G];                                                                          \
        float4 qb[GQA6_G];                                                                          \
        _Pragma("unroll")                                                                           \
        for (unsigned int g = 0; g < GQA6_G; g++) {                                                 \
            const float4* q4 = reinterpret_cast<const float4*>(s_q + g * GQA6_HD);                  \
            qa[g] = q4[lane];                                                                       \
            qb[g] = q4[32u + lane];                                                                 \
        }                                                                                           \
        for (unsigned int j = warp; j < span; j += GQA6_WARPS) {                                    \
            float4 ka, kb;                                                                          \
            KLOAD(p0 + j, ka, kb);                                                                  \
            _Pragma("unroll")                                                                       \
            for (unsigned int g = 0; g < GQA6_G; g++) {                                             \
                float dot = qa[g].x * ka.x + qa[g].y * ka.y + qa[g].z * ka.z + qa[g].w * ka.w;       \
                dot += qb[g].x * kb.x + qb[g].y * kb.y + qb[g].z * kb.z + qb[g].w * kb.w;           \
                dot = gqa6_warp_sum(dot) * scale;                                                   \
                if (lane == 0u) s_score[g * GQA6_TILE + j] = dot;                                   \
            }                                                                                       \
        }                                                                                           \
        __syncthreads();                                                                            \
                                                                                                    \
        /* Softmax over the tile with the running max: one warp per head.    */                     \
        /* m' = max(m, tile max); p = exp(s - m'); the warp's lane 0 updates   */                    \
        /* the running (m, l) and the rescale the PV pass applies.             */                   \
        for (unsigned int g = warp; g < GQA6_G; g += GQA6_WARPS) {                                  \
            float s = (lane < span) ? s_score[g * GQA6_TILE + lane] : GQA6_NEG_INF;                 \
            float mt = gqa6_warp_max(s);                                                            \
            float mn = first ? mt : fmaxf(s_m[g], mt);                                              \
            float p = (lane < span) ? expf(s - mn) : 0.0f;                                          \
            float lt = gqa6_warp_sum(p);                                                            \
            if (lane < span) s_score[g * GQA6_TILE + lane] = p;                                     \
            if (lane == 0u) {                                                                       \
                if (first) {                                                                        \
                    /* m = -inf, rescale = 0: the tile's numbers are the shipped */                 \
                    /* partial's exactly, with no exp spent.                     */                 \
                    s_resc[g] = 0.0f;                                                               \
                    s_l[g] = lt;                                                                    \
                } else {                                                                            \
                    const float r = expf(s_m[g] - mn);                                              \
                    s_resc[g] = r;                                                                  \
                    s_l[g] = r * s_l[g] + lt;                                                       \
                }                                                                                   \
                s_m[g] = mn;                                                                        \
            }                                                                                       \
        }                                                                                           \
        __syncthreads();                                                                            \
                                                                                                    \
        /* PV with the rescale: acc = rescale * acc + sum_j p_j v_j, ascending j. */                \
        if (!first) {                                                                               \
            _Pragma("unroll")                                                                       \
            for (unsigned int g = 0; g < GQA6_G; g++) {                                             \
                const float r = s_resc[g];                                                          \
                acc0[g] *= r;                                                                       \
                acc1[g] *= r;                                                                       \
            }                                                                                       \
        }                                                                                           \
        for (unsigned int j = 0; j < span; j++) {                                                   \
            float v0, v1;                                                                           \
            if (V_HALF) {                                                                           \
                v0 = gqa6_h2f(s_vh[j * GQA6_HD + tid]);                                             \
                v1 = gqa6_h2f(s_vh[j * GQA6_HD + tid + GQA6_BLOCK]);                                \
            } else {                                                                                \
                v0 = s_v[j * GQA6_HD + tid];                                                        \
                v1 = s_v[j * GQA6_HD + tid + GQA6_BLOCK];                                           \
            }                                                                                       \
            _Pragma("unroll")                                                                       \
            for (unsigned int g = 0; g < GQA6_G; g++) {                                             \
                const float p = s_score[g * GQA6_TILE + j];                                         \
                acc0[g] += p * v0;                                                                  \
                acc1[g] += p * v1;                                                                  \
            }                                                                                       \
        }                                                                                           \
        /* The next tile overwrites shared V and the scores; everyone is done. */                   \
        __syncthreads();                                                                            \
    }                                                                                               \
                                                                                                    \
    _Pragma("unroll")                                                                               \
    for (unsigned int g = 0; g < GQA6_G; g++) {                                                     \
        unsigned int head = kv_h * GQA6_G + g;                                                      \
        unsigned long long base = ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;        \
        o_part[base + tid] = acc0[g];                                                               \
        o_part[base + tid + GQA6_BLOCK] = acc1[g];                                                  \
    }                                                                                               \
    if (tid < GQA6_G) {                                                                             \
        unsigned int head = kv_h * GQA6_G + tid;                                                    \
        m_part[head * num_chunks + chunk] = s_m[tid];                                               \
        l_part[head * num_chunks + chunk] = s_l[tid];                                               \
    }

#define GQA6_KLOAD_F32(pos, ka, kb)                                                                 \
    {                                                                                               \
        const float4* k4 = reinterpret_cast<const float4*>(                                         \
            k_cache + kv_base + (unsigned long long)(pos) * (unsigned long long)GQA6_HD);           \
        ka = k4[lane];                                                                              \
        kb = k4[32u + lane];                                                                        \
    }

#define GQA6_KLOAD_F16(pos, ka, kb)                                                                 \
    {                                                                                               \
        const uint2* k2 = reinterpret_cast<const uint2*>(                                           \
            k_cache + kv_base + (unsigned long long)(pos) * (unsigned long long)GQA6_HD);           \
        const uint2 ra = k2[lane];                                                                  \
        const uint2 rb = k2[32u + lane];                                                            \
        ka = gqa6_h4_to_f4(ra.x, ra.y);                                                             \
        kb = gqa6_h4_to_f4(rb.x, rb.y);                                                             \
    }

extern "C" __global__ void __launch_bounds__(128, 4) attention_decode_splitk_partial_gqa6_loop_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ o_part,
    unsigned int seq_len,
    unsigned int max_seq_len,
    float scale,
    unsigned int num_chunks,
    unsigned int partition)
{
    GQA6_LOOP_BODY(false, GQA6_KLOAD_F32)
}

extern "C" __global__ void __launch_bounds__(128, 4) attention_decode_splitk_partial_gqa6_loop_f16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ o_part,
    unsigned int seq_len,
    unsigned int max_seq_len,
    float scale,
    unsigned int num_chunks,
    unsigned int partition)
{
    GQA6_LOOP_BODY(true, GQA6_KLOAD_F16)
}

// --------------------------------------------------------------------------
// The previous one-tile partials, retained under their names as the A/B
// control (`LUMEN_CUDA_ATTN_SPLITK_GQA6_MAX_CHUNKS`) and as the tests'
// reference for the loop's one-tile form: grid (S, num_kv_heads) with
// S = ceil(keys / C), one chunk of at most C <= 32 keys per CTA, the same
// per-tile arithmetic the loop runs. Byte-for-byte the kernels of the
// previous release (before this release's bound raise); the loop's first tile reproduces
// them bit for bit. Shared: 6*256 Q + C*256 V (halves for _f16) + 6*C scores
// + 12 (m, l) floats.
// --------------------------------------------------------------------------

extern "C" __global__ void attention_decode_splitk_partial_gqa6_f32(
    const float* __restrict__ q,           // [num_heads * 256]
    const float* __restrict__ k_cache,     // [num_kv_heads, max_seq_len, 256]
    const float* __restrict__ v_cache,     // [num_kv_heads, max_seq_len, 256]
    float* __restrict__ m_part,            // [num_heads * S]
    float* __restrict__ l_part,            // [num_heads * S]
    float* __restrict__ o_part,            // [num_heads * S * 256]
    unsigned int seq_len,
    unsigned int max_seq_len,
    float scale,
    unsigned int num_chunks,               // S
    unsigned int chunk_cap)                // C
{
    const unsigned int chunk = blockIdx.x;
    const unsigned int kv_h  = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int lane  = tid & 31u;
    const unsigned int warp  = tid >> 5;

    const unsigned int chunk_span = (seq_len + num_chunks - 1u) / num_chunks;
    const unsigned int p0 = chunk * chunk_span;
    unsigned int span = 0u;
    if (p0 < seq_len) {
        unsigned int p1 = p0 + chunk_span;
        if (p1 > seq_len) p1 = seq_len;
        span = p1 - p0;
    }

    // Empty chunk. `span` is block-uniform, so the whole CTA returns here and
    // no barrier is skipped by a subset of the threads.
    if (span == 0u) {
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            unsigned int head = kv_h * GQA6_G + g;
            unsigned long long base =
                ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;
            o_part[base + tid] = 0.0f;
            o_part[base + tid + GQA6_BLOCK] = 0.0f;
        }
        if (tid < GQA6_G) {
            unsigned int head = kv_h * GQA6_G + tid;
            m_part[head * num_chunks + chunk] = GQA6_NEG_INF;
            l_part[head * num_chunks + chunk] = 0.0f;
        }
        return;
    }

    // 16-byte aligned: s_q and s_v are cast to float4 below, and both start
    // at multiples of 16 bytes from the base (s_v at 6 * 256 floats).
    extern __shared__ __align__(16) float smem[];
    float* s_q     = smem;                          // [6][256]
    float* s_v     = s_q + GQA6_G * GQA6_HD;        // [C][256]
    float* s_score = s_v + chunk_cap * GQA6_HD;     // [6][C]
    float* s_m     = s_score + GQA6_G * chunk_cap;  // [6]
    float* s_l     = s_m + GQA6_G;                  // [6]

    const unsigned long long kv_base =
        (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)GQA6_HD;

    // Both staging streams are issued before the barrier so their loads overlap.
    {
        const float4* q4src =
            reinterpret_cast<const float4*>(q + (unsigned long long)kv_h * GQA6_G * GQA6_HD);
        float4* q4dst = reinterpret_cast<float4*>(s_q);
        for (unsigned int i = tid; i < (GQA6_G * GQA6_HD) / 4u; i += GQA6_BLOCK) {
            q4dst[i] = q4src[i];
        }
        const float4* v4src = reinterpret_cast<const float4*>(
            v_cache + kv_base + (unsigned long long)p0 * (unsigned long long)GQA6_HD);
        float4* v4dst = reinterpret_cast<float4*>(s_v);
        const unsigned int nv4 = span * (GQA6_HD / 4u);
        for (unsigned int i = tid; i < nv4; i += GQA6_BLOCK) {
            v4dst[i] = v4src[i];
        }
    }
    __syncthreads();

    // Q stays in registers for the whole QK phase: 2 float4 per head per lane.
    float4 qa[GQA6_G];
    float4 qb[GQA6_G];
#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        const float4* q4 = reinterpret_cast<const float4*>(s_q + g * GQA6_HD);
        qa[g] = q4[lane];
        qb[g] = q4[32u + lane];
    }

    // QK: a warp per position, the K row held in registers across all six heads.
    for (unsigned int j = warp; j < span; j += GQA6_WARPS) {
        const float4* k4 = reinterpret_cast<const float4*>(
            k_cache + kv_base + (unsigned long long)(p0 + j) * (unsigned long long)GQA6_HD);
        const float4 ka = k4[lane];
        const float4 kb = k4[32u + lane];
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            float dot = qa[g].x * ka.x + qa[g].y * ka.y + qa[g].z * ka.z + qa[g].w * ka.w;
            dot += qb[g].x * kb.x + qb[g].y * kb.y + qb[g].z * kb.z + qb[g].w * kb.w;
            dot = gqa6_warp_sum(dot) * scale;
            if (lane == 0u) s_score[g * chunk_cap + j] = dot;
        }
    }
    __syncthreads();

    // Softmax: one warp per query head, four warps over six heads.
    for (unsigned int g = warp; g < GQA6_G; g += GQA6_WARPS) {
        float s = (lane < span) ? s_score[g * chunk_cap + lane] : GQA6_NEG_INF;
        float m = gqa6_warp_max(s);
        float p = (lane < span) ? expf(s - m) : 0.0f;
        float l = gqa6_warp_sum(p);
        if (lane < span) s_score[g * chunk_cap + lane] = p;
        if (lane == 0u) {
            s_m[g] = m;
            s_l[g] = l;
        }
    }
    __syncthreads();

    // PV: thread t owns dims t and t+128; ascending positions; each staged V
    // value feeds all six heads.
    float acc0[GQA6_G];
    float acc1[GQA6_G];
#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        acc0[g] = 0.0f;
        acc1[g] = 0.0f;
    }
    for (unsigned int j = 0; j < span; j++) {
        const float v0 = s_v[j * GQA6_HD + tid];
        const float v1 = s_v[j * GQA6_HD + tid + GQA6_BLOCK];
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            const float p = s_score[g * chunk_cap + j];
            acc0[g] += p * v0;
            acc1[g] += p * v1;
        }
    }

#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        unsigned int head = kv_h * GQA6_G + g;
        unsigned long long base = ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;
        o_part[base + tid] = acc0[g];
        o_part[base + tid + GQA6_BLOCK] = acc1[g];
    }
    if (tid < GQA6_G) {
        unsigned int head = kv_h * GQA6_G + tid;
        m_part[head * num_chunks + chunk] = s_m[tid];
        l_part[head * num_chunks + chunk] = s_l[tid];
    }
}

extern "C" __global__ void attention_decode_splitk_partial_gqa6_f16(
    const float* __restrict__ q,                    // [num_heads * 256]
    const unsigned short* __restrict__ k_cache,     // [num_kv_heads, max_seq_len, 256] half bits
    const unsigned short* __restrict__ v_cache,     // [num_kv_heads, max_seq_len, 256] half bits
    float* __restrict__ m_part,                     // [num_heads * S]
    float* __restrict__ l_part,                     // [num_heads * S]
    float* __restrict__ o_part,                     // [num_heads * S * 256]
    unsigned int seq_len,
    unsigned int max_seq_len,
    float scale,
    unsigned int num_chunks,                        // S
    unsigned int chunk_cap)                         // C
{
    const unsigned int chunk = blockIdx.x;
    const unsigned int kv_h  = blockIdx.y;
    const unsigned int tid   = threadIdx.x;
    const unsigned int lane  = tid & 31u;
    const unsigned int warp  = tid >> 5;

    const unsigned int chunk_span = (seq_len + num_chunks - 1u) / num_chunks;
    const unsigned int p0 = chunk * chunk_span;
    unsigned int span = 0u;
    if (p0 < seq_len) {
        unsigned int p1 = p0 + chunk_span;
        if (p1 > seq_len) p1 = seq_len;
        span = p1 - p0;
    }

    // Empty chunk: block-uniform, the whole CTA returns before any barrier.
    if (span == 0u) {
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            unsigned int head = kv_h * GQA6_G + g;
            unsigned long long base =
                ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;
            o_part[base + tid] = 0.0f;
            o_part[base + tid + GQA6_BLOCK] = 0.0f;
        }
        if (tid < GQA6_G) {
            unsigned int head = kv_h * GQA6_G + tid;
            m_part[head * num_chunks + chunk] = GQA6_NEG_INF;
            l_part[head * num_chunks + chunk] = 0.0f;
        }
        return;
    }

    // Shared layout: s_q [6][256] F32, s_vh [C][256] halves (C * 128 floats of
    // space), then the score block and the (m, l) slots as in the F32 kernel.
    // 16-byte aligned: s_q starts at 0, s_vh at 6144 B, s_score at
    // 6144 + C * 512 B — a multiple of 16 for every C.
    extern __shared__ __align__(16) float smem[];
    float* s_q = smem;                                               // [6][256]
    unsigned short* s_vh = reinterpret_cast<unsigned short*>(s_q + GQA6_G * GQA6_HD); // [C][256]
    float* s_score = s_q + GQA6_G * GQA6_HD + (chunk_cap * GQA6_HD) / 2u;  // [6][C]
    float* s_m = s_score + GQA6_G * chunk_cap;                       // [6]
    float* s_l = s_m + GQA6_G;                                       // [6]

    const unsigned long long kv_base =
        (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)GQA6_HD;

    // Both staging streams are issued before the barrier so their loads overlap.
    {
        const float4* q4src =
            reinterpret_cast<const float4*>(q + (unsigned long long)kv_h * GQA6_G * GQA6_HD);
        float4* q4dst = reinterpret_cast<float4*>(s_q);
        for (unsigned int i = tid; i < (GQA6_G * GQA6_HD) / 4u; i += GQA6_BLOCK) {
            q4dst[i] = q4src[i];
        }
        // V tile as halves: eight halves (16 bytes) per thread per step.
        const uint4* v8src = reinterpret_cast<const uint4*>(
            v_cache + kv_base + (unsigned long long)p0 * (unsigned long long)GQA6_HD);
        uint4* v8dst = reinterpret_cast<uint4*>(s_vh);
        const unsigned int nv8 = span * (GQA6_HD / 8u);
        for (unsigned int i = tid; i < nv8; i += GQA6_BLOCK) {
            v8dst[i] = v8src[i];
        }
    }
    __syncthreads();

    // Q stays in registers for the whole QK phase: 2 float4 per head per lane.
    float4 qa[GQA6_G];
    float4 qb[GQA6_G];
#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        const float4* q4 = reinterpret_cast<const float4*>(s_q + g * GQA6_HD);
        qa[g] = q4[lane];
        qb[g] = q4[32u + lane];
    }

    // QK: a warp per position; two 8-byte loads per lane per K row, widened
    // into the same float4 pairs the F32 kernel reads.
    for (unsigned int j = warp; j < span; j += GQA6_WARPS) {
        const uint2* k2 = reinterpret_cast<const uint2*>(
            k_cache + kv_base + (unsigned long long)(p0 + j) * (unsigned long long)GQA6_HD);
        const uint2 ra = k2[lane];
        const uint2 rb = k2[32u + lane];
        const float4 ka = gqa6_h4_to_f4(ra.x, ra.y);
        const float4 kb = gqa6_h4_to_f4(rb.x, rb.y);
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            float dot = qa[g].x * ka.x + qa[g].y * ka.y + qa[g].z * ka.z + qa[g].w * ka.w;
            dot += qb[g].x * kb.x + qb[g].y * kb.y + qb[g].z * kb.z + qb[g].w * kb.w;
            dot = gqa6_warp_sum(dot) * scale;
            if (lane == 0u) s_score[g * chunk_cap + j] = dot;
        }
    }
    __syncthreads();

    // Softmax: one warp per query head, four warps over six heads.
    for (unsigned int g = warp; g < GQA6_G; g += GQA6_WARPS) {
        float s = (lane < span) ? s_score[g * chunk_cap + lane] : GQA6_NEG_INF;
        float m = gqa6_warp_max(s);
        float p = (lane < span) ? expf(s - m) : 0.0f;
        float l = gqa6_warp_sum(p);
        if (lane < span) s_score[g * chunk_cap + lane] = p;
        if (lane == 0u) {
            s_m[g] = m;
            s_l[g] = l;
        }
    }
    __syncthreads();

    // PV: thread t owns dims t and t+128; ascending positions; each staged V
    // half is widened once and feeds all six heads.
    float acc0[GQA6_G];
    float acc1[GQA6_G];
#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        acc0[g] = 0.0f;
        acc1[g] = 0.0f;
    }
    for (unsigned int j = 0; j < span; j++) {
        const float v0 = gqa6_h2f(s_vh[j * GQA6_HD + tid]);
        const float v1 = gqa6_h2f(s_vh[j * GQA6_HD + tid + GQA6_BLOCK]);
#pragma unroll
        for (unsigned int g = 0; g < GQA6_G; g++) {
            const float p = s_score[g * chunk_cap + j];
            acc0[g] += p * v0;
            acc1[g] += p * v1;
        }
    }

#pragma unroll
    for (unsigned int g = 0; g < GQA6_G; g++) {
        unsigned int head = kv_h * GQA6_G + g;
        unsigned long long base = ((unsigned long long)head * num_chunks + chunk) * GQA6_HD;
        o_part[base + tid] = acc0[g];
        o_part[base + tid + GQA6_BLOCK] = acc1[g];
    }
    if (tid < GQA6_G) {
        unsigned int head = kv_h * GQA6_G + tid;
        m_part[head * num_chunks + chunk] = s_m[tid];
        l_part[head * num_chunks + chunk] = s_l[tid];
    }
}

// --------------------------------------------------------------------------
// Merge. grid = (num_heads, 256 / 128), block = 128 threads: two CTAs per
// query head, each owning 128 of the head's dimensions, one per thread.
//
// attention_decode_splitk_merge re-evaluates expf(m[c] - M) once per (chunk,
// dimension) — 256 times per chunk. Here the per-chunk rescale is computed
// once into shared, L is a fixed-tree reduction, and the output accumulation
// costs one multiply per (chunk, dimension). This decomposition multiplies
// the split count, so that redundancy would otherwise grow with it.
//
// Splitting the dimensions across two CTAs costs a second evaluation of the
// S rescale factors and buys grid parallelism: one CTA per head is 24 CTAs
// on a 170-SM card.
//
// Shared: num_chunks + 4 floats.
// --------------------------------------------------------------------------
extern "C" __global__ void attention_decode_splitk_merge_gqa6_f32(
    const float* __restrict__ m_part,      // [num_heads * S]
    const float* __restrict__ l_part,      // [num_heads * S]
    const float* __restrict__ o_part,      // [num_heads * S * 256]
    float* __restrict__ attn_out,          // [num_heads * 256]
    unsigned int num_chunks)               // S
{
    const unsigned int head = blockIdx.x;
    const unsigned int dt   = blockIdx.y;
    const unsigned int tid  = threadIdx.x;

    extern __shared__ __align__(16) float smem[];
    float* s_alpha = smem;                    // [num_chunks]
    volatile float* scr = smem + num_chunks;  // [4]

    const float* mp = m_part + (unsigned long long)head * num_chunks;
    const float* lp = l_part + (unsigned long long)head * num_chunks;

    float lm = GQA6_NEG_INF;
    for (unsigned int c = tid; c < num_chunks; c += GQA6_BLOCK) {
        lm = fmaxf(lm, mp[c]);
    }
    const float m_max = gqa6_block_max(lm, scr, tid);

    // Empty chunks carry (m = -inf, l = 0, o = 0) and contribute nothing. The
    // `!= 0` form (not `> 0`) keeps a NaN `l` chunk IN the sum so NaN
    // propagates, matching attention_decode_splitk_merge.
    float ls = 0.0f;
    for (unsigned int c = tid; c < num_chunks; c += GQA6_BLOCK) {
        const float lc = lp[c];
        const float a = (lc != 0.0f) ? expf(mp[c] - m_max) : 0.0f;
        s_alpha[c] = a;
        ls += lc * a;
    }
    const float l_total = gqa6_block_sum(ls, scr, tid);
    const float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;

    // The numerator runs over every chunk. Eight independent lanes, chunk c
    // into lane c % 8, then a fixed tree: the rounding-error growth of the
    // serial sum drops with the chain length (S/8 + 3 terms deep instead of
    // S), and the order stays fixed, so the result is deterministic. Real
    // activations at 5k and 11k keys put the serial form's worst coordinate
    // error at 8.7e-5 against 2.0e-5 for the per-query-head pair.
    const float* op = o_part + (unsigned long long)head * num_chunks * GQA6_HD;
    const unsigned int d = dt * GQA6_BLOCK + tid;
    float acc[GQA6_MERGE_LANES];
#pragma unroll
    for (unsigned int i = 0; i < GQA6_MERGE_LANES; i++) acc[i] = 0.0f;
    unsigned int c = 0;
    for (; c + GQA6_MERGE_LANES <= num_chunks; c += GQA6_MERGE_LANES) {
#pragma unroll
        for (unsigned int i = 0; i < GQA6_MERGE_LANES; i++) {
            acc[i] += op[(unsigned long long)(c + i) * GQA6_HD + d] * s_alpha[c + i];
        }
    }
    // Tail: the same lanes, compile-time indices only, so `acc` stays in
    // registers (a runtime index into a local array demotes it to local
    // memory).
#pragma unroll
    for (unsigned int i = 0; i < GQA6_MERGE_LANES; i++) {
        if (c + i < num_chunks) {
            acc[i] += op[(unsigned long long)(c + i) * GQA6_HD + d] * s_alpha[c + i];
        }
    }
    static_assert(GQA6_MERGE_LANES == 8u, "the merge's lane tree sums exactly eight lanes");
    const float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    attn_out[head * GQA6_HD + d] = sum * inv_l;
}
