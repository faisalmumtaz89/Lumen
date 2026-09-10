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
// Decomposition of one CTA (128 threads = 4 warps, chunk of C <= 32 positions):
//
//   stage    Q for the six heads (6 KiB) and the whole V tile (C KiB) are
//            copied into shared with float4 loads, both issued before the
//            first barrier so the two streams overlap.
//   QK       a warp owns one position at a time; lanes own dimensions. Lane l
//            holds K[pos][4l .. 4l+4) and K[pos][128+4l .. 128+4l+4) — two
//            float4 — in registers while all six head scores are formed from
//            them, each by a warp shuffle-xor tree. Q stays in registers for
//            the whole phase (2 float4 per head per lane).
//   softmax  scores are [6][C] in shared; one warp per query head, four warps
//            covering six heads in two rounds. C <= 32 so a head's whole chunk
//            fits one warp's lanes and the reduction is two shuffle trees.
//   PV       thread t owns dims t and t+128 and keeps 12 accumulators
//            (2 dims x 6 heads), walking the staged V tile in ascending
//            position order and reusing each V value across the group.
//
// Shared memory: 6*256 (Q) + C*256 (V) + 6*C (scores) + 12 (m, l) floats.
// At C = 16 that is 22'960 B, under the 48 KiB default dynamic-shared cap, so
// no cudaFuncSetAttribute opt-in is needed.
//
// PRECISION: F32 storage and accumulation throughout, the same expf, no
// atomics and no fast-math. The QK reduction tree, the chunk partition and
// the PV order all reassociate the sum differently from
// attention_decode_splitk.cu, so the two routes are a quality-equivalent
// NEAR-TIE rather than bit-identical — the same relationship the split-K pair
// has with the tiled kernel. Deterministic for a fixed (S, C): every
// reduction is a fixed tree and every accumulation walks ascending indices.
// Against an F64 reference over 24 heads x head_dim 256 the largest absolute
// error observed at contexts from 1 to 4096 is 4.3e-7, below the 8.2e-7 of
// attention_decode_splitk.cu on the same inputs (smaller chunks accumulate
// less drift).
//
// The empty-chunk arm (p0 >= seq_len) writes (m = -inf, l = 0, o = 0) and the
// merge drops it, exactly as in attention_decode_splitk.cu. It is DEFENSIVE:
// the host derives S = ceil(seq_len / C), so (S - 1) * ceil(seq_len / S) <
// seq_len for every context it dispatches and no chunk is ever empty. The arm
// keeps a caller that oversplits — a test, or a future policy — exact instead
// of merely lucky.
//
// Scratch layout (device buffers, sized by the host), identical to the
// sibling pair's but with the larger split count this decomposition uses:
//   m_part [num_heads * S]              F32
//   l_part [num_heads * S]              F32
//   o_part [num_heads * S * head_dim]   F32
//
// Requires (enforced by the host gate
// `attention_decode_splitk_gqa6_supports`): num_heads / num_kv_heads == 6,
// head_dim == 256, F32 KV, and a chunk span of at most C.
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

// --------------------------------------------------------------------------
// Partial pass. grid = (num_chunks, num_kv_heads), block = 128 threads.
// `chunk_cap` is the score and V-tile stride C; the host guarantees the chunk
// span is at most C and C is at most 32.
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
