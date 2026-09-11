// ==========================================================================
// Decode attention: the one kernel pair every model and every context takes.
// Specialised at compile time for the model's group size G (query heads per
// KV head) and head_dim HD: the host prepends `#define DECODE_G <g>u` and
// `#define DECODE_HD <hd>u` to this source before NVRTC sees it (G in 1..=8,
// HD in {128, 256}). Every shipped model has HD 256: Qwen3.5-9B G = 4
// (16 Q / 4 KV), Qwen3.6-27B and Qwen3.8-27B G = 6 (24 / 4),
// Qwen3.5-MoE-35B-A3B G = 8 (16 / 2).
//
// Two passes over split-K partials: `attention_decode_partial_{f32,f16}`
// writes (m, l, o) per (query head, chunk), `attention_decode_merge` combines
// them. One CTA per (KV head, chunk): each K row and V row is fetched ONCE
// and serves all G query heads of the group, and every Q, K and V read is a
// 16-byte load on the F32 store (on the half store Q and V stay 16-byte; K
// rows are read as 8-byte loads). (The partial's scratch stores and the
// merge's reads of them stay scalar; they are a fifth of the traffic.)
//
// Decomposition of one CTA (128 threads = 4 warps), per 16-key tile:
//
//   stage    Q for the G heads (G*HD floats) and the first V tile (16*HD
//            floats, halves on a half store) are copied into shared with
//            16-byte loads, both issued before the first barrier so the two
//            streams overlap; each later tile's V is staged before that
//            tile's dot products.
//   QK       a warp owns one key at a time; lanes own dimensions. Lane l
//            holds the HD/128 float4 K[pos][128c+4l .. 128c+4l+4) (at 256:
//            dims 4l..4l+4 and 128+4l..128+4l+4, widened from halves on a
//            half store) in registers while all G head scores are formed
//            from them, each by a warp shuffle-xor tree. Q is reloaded from
//            shared per tile so its registers live only through the phase.
//   softmax  scores are [G][16] in shared; one warp per query head, four warps
//            covering G heads in ceil(G/4) rounds; the tile's max joins the
//            running max and the tile's sum the running sum (rescaled).
//   PV       thread t owns dims t + 128i for i < HD/128 and keeps
//            (HD/128) x G accumulators, rescaled once per tile and walking
//            the staged V tile in ascending key order, reusing each V value
//            across the group.
//
// A CTA walks one tile up to the host's one-tile bound and a contiguous run
// of whole tiles above it, so the split count — and with it the scratch — is
// bounded by the host's target at any context the cache holds. The two
// partitions are the same loop; a one-tile CTA runs it once.
//
// Shared memory: G*HD (Q) + 16*HD (V; half that in floats on a half store)
// + G*16 (scores) + G m + G l + G rescale floats: at (6, 256) 22'984 B (F32)
// / 14'792 B (half), at (8, 256) 25'184 / 16'992 B, under the 48 KiB default
// dynamic-shared cap, so no opt-in is needed.
//
// PRECISION: F32 accumulation throughout (K and V widened exactly from halves
// on a half store), the same expf, no atomics and no fast-math. Deterministic
// for a fixed (S, partition): every reduction is a fixed tree and every
// accumulation walks ascending indices, so the output depends on the launch
// geometry and on nothing else. The integration suites hold both partitions
// within 2e-6 of an F64 reference at every (G, HD) in the domain (recorded
// maximum 2.45e-7) and at every context from 1 to 32,768 keys at (6, 256), and print
// the observed maximum per length, so the headroom is a recorded number
// rather than a figure in this comment.
//
// The empty-range arm writes (m = -inf, l = 0, o = 0) and the merge drops it.
// It is DEFENSIVE: neither partition the host derives leaves a CTA empty. The
// arm keeps a caller that oversplits — a test, or a future policy — exact
// instead of merely lucky.
//
// Scratch layout (device buffers, sized by the host once per model for the
// largest split count the policy can launch):
//   m_part [num_heads * S]              F32
//   l_part [num_heads * S]              F32
//   o_part [num_heads * S * head_dim]   F32
//
// Requires (enforced by the host, which compiled this module for the model's
// shape): num_heads / num_kv_heads == DECODE_G, head_dim == DECODE_HD, and the
// store the entry point is for (F32 words for _f32, half bit patterns for
// _f16).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#ifndef DECODE_G
#error "DECODE_G (query heads per KV head) must be defined by the host before this source"
#endif
#ifndef DECODE_HD
#error "DECODE_HD (head_dim) must be defined by the host before this source"
#endif
static_assert(DECODE_G >= 1u && DECODE_G <= 8u, "the group size G must be in 1..=8");
static_assert(DECODE_HD == 128u || DECODE_HD == 256u, "head_dim must be 128 or 256");

#define DECODE_NEG_INF (-3.402823466e+38f)
#define DECODE_MERGE_LANES 8u       // independent numerator accumulators in the merge (the tree below sums exactly eight)
#define DECODE_BLOCK    128u
#define DECODE_WARPS    (DECODE_BLOCK / 32u)
// QK: lane l holds DECODE_LC float4 of a K row, chunk c at dims [128c + 4l, +4).
// PV: thread t owns dims t + 128i, i < DECODE_DPT.
#define DECODE_LC       (DECODE_HD / 128u)
#define DECODE_DPT      (DECODE_HD / 128u)

__device__ __forceinline__ float decode_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ float decode_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Four-warp block reduction over a 4-float shared scratch. Used by the merge.
__device__ __forceinline__ float decode_block_max(float v, volatile float* scr, unsigned int tid) {
    v = decode_warp_max(v);
    if ((tid & 31u) == 0u) scr[tid >> 5] = v;
    __syncthreads();
    float r = fmaxf(fmaxf(scr[0], scr[1]), fmaxf(scr[2], scr[3]));
    __syncthreads();
    return r;
}

__device__ __forceinline__ float decode_block_sum(float v, volatile float* scr, unsigned int tid) {
    v = decode_warp_sum(v);
    if ((tid & 31u) == 0u) scr[tid >> 5] = v;
    __syncthreads();
    float r = (scr[0] + scr[1]) + (scr[2] + scr[3]);
    __syncthreads();
    return r;
}

#define DECODE_TILE     16u

__device__ __forceinline__ float decode_h2f(unsigned int h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)(h & 0xffffu)));
    return f;
}

// Four packed halves (two 32-bit words, low half first) -> float4.
__device__ __forceinline__ float4 decode_h4_to_f4(unsigned int lo, unsigned int hi) {
    float4 o;
    o.x = decode_h2f(lo);
    o.y = decode_h2f(lo >> 16);
    o.z = decode_h2f(hi);
    o.w = decode_h2f(hi >> 16);
    return o;
}

// --------------------------------------------------------------------------
// The partial pass: a TILE LOOP with a bounded split count, so one kernel
// serves any context the cache holds with fixed scratch and no route switch.
//
// Geometry: grid (S, num_kv_heads).
//   partition = 0  one tile per CTA, S = ceil(keys / 16): CTA `chunk` owns keys
//                  [chunk * span, min((chunk + 1) * span, keys)), span = ceil(keys / S)
//                  (at most 16 keys).
//   partition = 1  whole-tile balanced, S < N = ceil(keys / 16): CTA `chunk` owns
//                  tiles [floor(chunk * N / S), floor((chunk + 1) * N / S)) of 16 keys
//                  (the context's last tile may be shorter); no empty CTAs, at most
//                  one tile of difference between CTAs.
// Per tile: Q for the G heads in registers (reloaded from shared per tile so
// the registers live only through the dot products), a warp per key, the K row
// in registers, G dots per key in the fixed tree, scores [G][16] in shared,
// the V tile staged once, (HD/128) x G accumulators per thread over the tile's
// keys in ascending order.
// Across tiles: the running-max recurrence: m' = max(m, m_t), rescale =
// exp(m - m'), p_j = exp(s_j - m'), acc = rescale * acc + sum p_j v_j,
// l = rescale * l + sum p_j; the running (m, l, rescale) per head live in
// shared memory (a register array indexed at run time demotes to local memory).
// On the first tile m = -inf, the rescale is 0 and no exp is spent, so a
// one-tile CTA pays nothing for the loop.
// Output: (m, l, o[HD]) per head per CTA, the partial format the eight-lane
// merge below consumes.
// Occupancy: __launch_bounds__(128, 4) on both entry points (a floor on
// resident CTAs, not a cap). On the engine's compile path — NVRTC at its
// default PTX target, then the driver's ptxas for the device — ptxas reports
// 72 registers at G = 4, 96 at G = 6 and 128 at G = 8 (HD 256), no stack
// frame, no spills. At (6, 256) an SM holds five CTAs by registers: the F32
// entry is held to four by its 22,984 B of shared, the half entry runs five
// (14,792 B). Bounding the half entry to six forces 80 registers with spills
// to local memory on this path and was measured slower at every context, so
// the bound stays at four.
// A tile whose span would exceed 16 keys calls __trap(), which aborts the
// launch (the driver reports the error at the next synchronisation; no
// partial reaches the merge). It is unreachable by construction: on the span
// partition the host keeps S = ceil(keys / 16), so a CTA's whole range is at
// most 16 keys, and on the whole-tile partition every iteration clips its
// tile to 16 keys whatever the CTA's range.
// --------------------------------------------------------------------------

// The CTA's key range.
__device__ __forceinline__ void decode_loop_range(
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
        const unsigned long long n = ((unsigned long long)seq_len + DECODE_TILE - 1ull) / DECODE_TILE;
        const unsigned int t0 = (unsigned int)(((unsigned long long)chunk * n) / num_chunks);
        const unsigned int t1 = (unsigned int)(((unsigned long long)(chunk + 1u) * n) / num_chunks);
        unsigned int p0 = t0 * DECODE_TILE;
        unsigned int p1 = t1 * DECODE_TILE;
        if (p1 > seq_len) p1 = seq_len;
        if (p0 > p1) p0 = p1;
        *k0 = p0; *k1 = p1;
    }
}

// Stage `span` keys of V from position `p0` into the shared buffer `dst`
// (F32 floats, or half bit patterns when V_HALF): 16 bytes per thread per step.
#define DECODE_STAGE_V(V_HALF, dst, p0, span)                                                         \
    {                                                                                               \
        if (V_HALF) {                                                                               \
            const uint4* v8src = reinterpret_cast<const uint4*>(                                    \
                v_cache + kv_base + (unsigned long long)(p0) * (unsigned long long)DECODE_HD);        \
            uint4* v8dst = reinterpret_cast<uint4*>(dst);                                           \
            const unsigned int nv8 = (span) * (DECODE_HD / 8u);                                       \
            for (unsigned int i = tid; i < nv8; i += DECODE_BLOCK) v8dst[i] = v8src[i];               \
        } else {                                                                                    \
            const float4* v4src = reinterpret_cast<const float4*>(                                  \
                v_cache + kv_base + (unsigned long long)(p0) * (unsigned long long)DECODE_HD);        \
            float4* v4dst = reinterpret_cast<float4*>(dst);                                         \
            const unsigned int nv4 = (span) * (DECODE_HD / 4u);                                       \
            for (unsigned int i = tid; i < nv4; i += DECODE_BLOCK) v4dst[i] = v4src[i];               \
        }                                                                                           \
    }

// The body shared by the F32 and half variants. V_HALF selects the V staging.
#define DECODE_LOOP_BODY(V_HALF, KLOAD)                                                               \
    const unsigned int chunk = blockIdx.x;                                                          \
    const unsigned int kv_h  = blockIdx.y;                                                          \
    const unsigned int tid   = threadIdx.x;                                                         \
    const unsigned int lane  = tid & 31u;                                                           \
    const unsigned int warp  = tid >> 5;                                                            \
                                                                                                    \
    unsigned int k0, k1;                                                                            \
    decode_loop_range(chunk, seq_len, num_chunks, partition, &k0, &k1);                               \
                                                                                                    \
    /* Empty range: block-uniform, the whole CTA returns before any barrier. */                    \
    if (k1 <= k0) {                                                                                 \
        _Pragma("unroll")                                                                           \
        for (unsigned int g = 0; g < DECODE_G; g++) {                                                 \
            unsigned int head = kv_h * DECODE_G + g;                                                  \
            unsigned long long base =                                                               \
                ((unsigned long long)head * num_chunks + chunk) * DECODE_HD;                          \
            DECODE_ZERO_O(base);                                                                      \
        }                                                                                           \
        if (tid < DECODE_G) {                                                                         \
            unsigned int head = kv_h * DECODE_G + tid;                                                \
            m_part[head * num_chunks + chunk] = DECODE_NEG_INF;                                       \
            l_part[head * num_chunks + chunk] = 0.0f;                                               \
        }                                                                                           \
        return;                                                                                     \
    }                                                                                               \
                                                                                                    \
    extern __shared__ __align__(16) float smem[];                                                   \
    float* s_q = smem;                                                        /* [G][HD] */         \
    const unsigned int v_floats = (V_HALF) ? (DECODE_TILE * DECODE_HD) / 2u : DECODE_TILE * DECODE_HD;      \
    float* s_v = s_q + DECODE_G * DECODE_HD;                                      /* [16][HD] */        \
    const unsigned short* s_vh = reinterpret_cast<const unsigned short*>(s_v);                      \
    float* s_score = s_v + v_floats;                                          /* [G][16] */         \
    float* s_m = s_score + DECODE_G * DECODE_TILE;                                /* [G] running max */ \
    float* s_l = s_m + DECODE_G;                                                /* [G] running sum */ \
    float* s_resc = s_l + DECODE_G;                                             /* [G] rescale */     \
                                                                                                    \
    const unsigned long long kv_base =                                                              \
        (unsigned long long)kv_h * (unsigned long long)max_seq_len * (unsigned long long)DECODE_HD;   \
                                                                                                    \
    /* Q and tile 0's V are issued before the first barrier so their loads overlap. */              \
    {                                                                                               \
        const float4* q4src =                                                                       \
            reinterpret_cast<const float4*>(q + (unsigned long long)kv_h * DECODE_G * DECODE_HD);       \
        float4* q4dst = reinterpret_cast<float4*>(s_q);                                             \
        for (unsigned int i = tid; i < (DECODE_G * DECODE_HD) / 4u; i += DECODE_BLOCK) {                  \
            q4dst[i] = q4src[i];                                                                    \
        }                                                                                           \
        unsigned int e0 = k0 + DECODE_TILE;                                                           \
        if (e0 > k1) e0 = k1;                                                                       \
        DECODE_STAGE_V(V_HALF, s_v, k0, e0 - k0);                                                     \
    }                                                                                               \
    __syncthreads();                                                                                \
                                                                                                    \
    /* Each thread holds its dimensions of the numerator for each head; the   */                   \
    /* running max and sum per head are block-uniform and live in shared.     */                   \
    float acc[DECODE_DPT][DECODE_G];                                                                    \
    _Pragma("unroll")                                                                               \
    for (unsigned int g = 0; g < DECODE_G; g++) {                                                     \
        _Pragma("unroll")                                                                           \
        for (unsigned int i = 0; i < DECODE_DPT; i++) acc[i][g] = 0.0f;                               \
    }                                                                                               \
                                                                                                    \
    for (unsigned int p0 = k0; p0 < k1; p0 += DECODE_TILE) {                                          \
        unsigned int p1 = p0 + DECODE_TILE;                                                           \
        if (p1 > k1) p1 = k1;                                                                       \
        const unsigned int span = p1 - p0;                                                          \
        if (span > DECODE_TILE) { __trap(); }                                                         \
        const bool first = (p0 == k0);                                                              \
                                                                                                    \
        /* Later tiles stage their V here, before the dot products, so the     */                   \
        /* load overlaps them (tile 0's V came in with Q).                      */                  \
        if (!first) { DECODE_STAGE_V(V_HALF, s_v, p0, span); }                                        \
                                                                                                    \
        /* QK: a warp per key, the K row in registers across all G heads.    */                     \
        /* Q is reloaded from shared for each tile so its registers live only  */                    \
        /* through the dot products, not across the PV pass and the tile loop. */                   \
        DECODE_Q_DECL;                                                                                \
        DECODE_Q_LOAD;                                                                                \
        for (unsigned int j = warp; j < span; j += DECODE_WARPS) {                                    \
            DECODE_K_DECL;                                                                            \
            KLOAD(p0 + j);                                                                          \
            _Pragma("unroll")                                                                       \
            for (unsigned int g = 0; g < DECODE_G; g++) {                                             \
                float dot;                                                                          \
                DECODE_QK_DOT(g, dot);                                                                \
                dot = decode_warp_sum(dot) * scale;                                                   \
                if (lane == 0u) s_score[g * DECODE_TILE + j] = dot;                                   \
            }                                                                                       \
        }                                                                                           \
        __syncthreads();                                                                            \
                                                                                                    \
        /* Softmax over the tile with the running max: one warp per head.    */                     \
        /* m' = max(m, tile max); p = exp(s - m'); the warp's lane 0 updates   */                    \
        /* the running (m, l) and the rescale the PV pass applies.             */                   \
        for (unsigned int g = warp; g < DECODE_G; g += DECODE_WARPS) {                                  \
            float s = (lane < span) ? s_score[g * DECODE_TILE + lane] : DECODE_NEG_INF;                 \
            float mt = decode_warp_max(s);                                                            \
            float mn = first ? mt : fmaxf(s_m[g], mt);                                              \
            float p = (lane < span) ? expf(s - mn) : 0.0f;                                          \
            float lt = decode_warp_sum(p);                                                            \
            if (lane < span) s_score[g * DECODE_TILE + lane] = p;                                     \
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
            for (unsigned int g = 0; g < DECODE_G; g++) {                                             \
                const float r = s_resc[g];                                                          \
                _Pragma("unroll")                                                                   \
                for (unsigned int i = 0; i < DECODE_DPT; i++) acc[i][g] *= r;                         \
            }                                                                                       \
        }                                                                                           \
        for (unsigned int j = 0; j < span; j++) {                                                   \
            float vv[DECODE_DPT];                                                                     \
            DECODE_V_LOAD(V_HALF, j, vv);                                                             \
            _Pragma("unroll")                                                                       \
            for (unsigned int g = 0; g < DECODE_G; g++) {                                             \
                const float p = s_score[g * DECODE_TILE + j];                                         \
                _Pragma("unroll")                                                                   \
                for (unsigned int i = 0; i < DECODE_DPT; i++) acc[i][g] += p * vv[i];                 \
            }                                                                                       \
        }                                                                                           \
        /* The next tile overwrites shared V and the scores; everyone is done. */                   \
        __syncthreads();                                                                            \
    }                                                                                               \
                                                                                                    \
    _Pragma("unroll")                                                                               \
    for (unsigned int g = 0; g < DECODE_G; g++) {                                                     \
        unsigned int head = kv_h * DECODE_G + g;                                                      \
        unsigned long long base = ((unsigned long long)head * num_chunks + chunk) * DECODE_HD;        \
        DECODE_STORE_O(base, g);                                                                      \
    }                                                                                               \
    if (tid < DECODE_G) {                                                                             \
        unsigned int head = kv_h * DECODE_G + tid;                                                    \
        m_part[head * num_chunks + chunk] = s_m[tid];                                               \
        l_part[head * num_chunks + chunk] = s_l[tid];                                               \
    }

// The per-shape pieces of the loop body: what a lane holds of Q and K, the
// dot product over them, what a thread holds of V, and the o_part stores.
// Chunk c of a lane is float4 c*32 + lane of the row (at 256: the two float4
// the (6, 256) kernel holds as qa/ka and qb/kb, summed in that order).
#define DECODE_Q_DECL float4 qr[DECODE_LC][DECODE_G]
#define DECODE_Q_LOAD                                                                                 \
    _Pragma("unroll")                                                                               \
    for (unsigned int g = 0; g < DECODE_G; g++) {                                                     \
        const float4* q4 = reinterpret_cast<const float4*>(s_q + g * DECODE_HD);                      \
        _Pragma("unroll")                                                                           \
        for (unsigned int c = 0; c < DECODE_LC; c++) qr[c][g] = q4[32u * c + lane];                   \
    }
#define DECODE_K_DECL float4 kr[DECODE_LC]
#define DECODE_KLOAD_F32(pos)                                                                         \
    {                                                                                               \
        const float4* k4 = reinterpret_cast<const float4*>(                                         \
            k_cache + kv_base + (unsigned long long)(pos) * (unsigned long long)DECODE_HD);           \
        _Pragma("unroll")                                                                           \
        for (unsigned int c = 0; c < DECODE_LC; c++) kr[c] = k4[32u * c + lane];                      \
    }
#define DECODE_KLOAD_F16(pos)                                                                         \
    {                                                                                               \
        const uint2* k2 = reinterpret_cast<const uint2*>(                                           \
            k_cache + kv_base + (unsigned long long)(pos) * (unsigned long long)DECODE_HD);           \
        _Pragma("unroll")                                                                           \
        for (unsigned int c = 0; c < DECODE_LC; c++) {                                                \
            const uint2 r = k2[32u * c + lane];                                                     \
            kr[c] = decode_h4_to_f4(r.x, r.y);                                                        \
        }                                                                                           \
    }
/* chunk 0 assigns, later chunks add: at HD 256 this is `dot = qa.ka; dot += qb.kb`. */
#define DECODE_QK_DOT(g, dot)                                                                         \
    {                                                                                               \
        dot = qr[0][g].x * kr[0].x + qr[0][g].y * kr[0].y + qr[0][g].z * kr[0].z                    \
            + qr[0][g].w * kr[0].w;                                                                 \
        _Pragma("unroll")                                                                           \
        for (unsigned int c = 1; c < DECODE_LC; c++) {                                                \
            dot += qr[c][g].x * kr[c].x + qr[c][g].y * kr[c].y + qr[c][g].z * kr[c].z               \
                + qr[c][g].w * kr[c].w;                                                             \
        }                                                                                           \
    }
#define DECODE_V_LOAD(V_HALF, j, vv)                                                                  \
    {                                                                                               \
        _Pragma("unroll")                                                                           \
        for (unsigned int i = 0; i < DECODE_DPT; i++) {                                               \
            if (V_HALF) vv[i] = decode_h2f(s_vh[(j) * DECODE_HD + tid + DECODE_BLOCK * i]);               \
            else        vv[i] = s_v[(j) * DECODE_HD + tid + DECODE_BLOCK * i];                          \
        }                                                                                           \
    }
#define DECODE_ZERO_O(base)                                                                           \
    {                                                                                               \
        _Pragma("unroll")                                                                           \
        for (unsigned int i = 0; i < DECODE_DPT; i++) o_part[(base) + tid + DECODE_BLOCK * i] = 0.0f;   \
    }
#define DECODE_STORE_O(base, g)                                                                       \
    {                                                                                               \
        _Pragma("unroll")                                                                           \
        for (unsigned int i = 0; i < DECODE_DPT; i++) o_part[(base) + tid + DECODE_BLOCK * i] = acc[i][g]; \
    }

extern "C" __global__ void __launch_bounds__(128, 4) attention_decode_partial_f32(
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
    DECODE_LOOP_BODY(false, DECODE_KLOAD_F32)
}

extern "C" __global__ void __launch_bounds__(128, 4) attention_decode_partial_f16(
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
    DECODE_LOOP_BODY(true, DECODE_KLOAD_F16)
}


// --------------------------------------------------------------------------
// Merge. grid = (num_heads, HD / 128), block = 128 threads: HD / 128 CTAs per
// query head, each owning 128 of the head's dimensions, one per thread.
//
// The per-chunk rescale expf(m[c] - M) is computed once into shared, L is a
// fixed-tree reduction, and the output accumulation costs one multiply per
// (chunk, dimension), so the merge's cost grows with S only linearly.
//
// Splitting the dimensions across CTAs at HD 256 costs a second evaluation of
// the S rescale factors and buys grid parallelism: one CTA per head would be
// 24 CTAs on a 170-SM card for a 24-head model.
//
// Shared: num_chunks + 4 floats.
// --------------------------------------------------------------------------
extern "C" __global__ void attention_decode_merge(
    const float* __restrict__ m_part,      // [num_heads * S]
    const float* __restrict__ l_part,      // [num_heads * S]
    const float* __restrict__ o_part,      // [num_heads * S * HD]
    float* __restrict__ attn_out,          // [num_heads * HD]
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

    float lm = DECODE_NEG_INF;
    for (unsigned int c = tid; c < num_chunks; c += DECODE_BLOCK) {
        lm = fmaxf(lm, mp[c]);
    }
    const float m_max = decode_block_max(lm, scr, tid);

    // Empty chunks carry (m = -inf, l = 0, o = 0) and contribute nothing. The
    // `!= 0` form (not `> 0`) keeps a NaN `l` chunk IN the sum so NaN
    // propagates instead of being dropped.
    float ls = 0.0f;
    for (unsigned int c = tid; c < num_chunks; c += DECODE_BLOCK) {
        const float lc = lp[c];
        const float a = (lc != 0.0f) ? expf(mp[c] - m_max) : 0.0f;
        s_alpha[c] = a;
        ls += lc * a;
    }
    const float l_total = decode_block_sum(ls, scr, tid);
    const float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;

    // The numerator runs over every chunk. Eight independent lanes, chunk c
    // into lane c % 8, then a fixed tree: the rounding-error growth of the
    // serial sum drops with the chain length (S/8 + 3 terms deep instead of
    // S), and the order stays fixed, so the result is deterministic. Real
    // activations at 5k and 11k keys put the serial form's worst coordinate
    // error at 8.7e-5, where a merge over a handful of chunks stays near
    // 2.0e-5; the tree takes the chain length out of the error.
    const float* op = o_part + (unsigned long long)head * num_chunks * DECODE_HD;
    const unsigned int d = dt * DECODE_BLOCK + tid;
    // HD is a multiple of the block, so every thread of a host-issued grid
    // owns a dimension; the guard only protects a wider grid than HD / 128.
    // It sits after the block reductions, which every thread must join.
    if (d >= DECODE_HD) return;
    float acc[DECODE_MERGE_LANES];
#pragma unroll
    for (unsigned int i = 0; i < DECODE_MERGE_LANES; i++) acc[i] = 0.0f;
    unsigned int c = 0;
    for (; c + DECODE_MERGE_LANES <= num_chunks; c += DECODE_MERGE_LANES) {
#pragma unroll
        for (unsigned int i = 0; i < DECODE_MERGE_LANES; i++) {
            acc[i] += op[(unsigned long long)(c + i) * DECODE_HD + d] * s_alpha[c + i];
        }
    }
    // Tail: the same lanes, compile-time indices only, so `acc` stays in
    // registers (a runtime index into a local array demotes it to local
    // memory).
#pragma unroll
    for (unsigned int i = 0; i < DECODE_MERGE_LANES; i++) {
        if (c + i < num_chunks) {
            acc[i] += op[(unsigned long long)(c + i) * DECODE_HD + d] * s_alpha[c + i];
        }
    }
    static_assert(DECODE_MERGE_LANES == 8u, "the merge's lane tree sums exactly eight lanes");
    const float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    attn_out[head * DECODE_HD + d] = sum * inv_l;
}
