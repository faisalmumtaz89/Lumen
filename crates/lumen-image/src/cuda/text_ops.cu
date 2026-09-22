// Kernels the text tower's GPU forward needs that no other source provides.
//
// Reused unchanged, from sources this crate already compiles:
//   - gemm_16bit        (dit_ops.cu)   the projections, against BF16/F16 weights
//   - rmsnorm_per_head  (lumen-runtime's norm.cu), shared-weight mode
//   - swiglu_inplace    (lumen-runtime's activations.cu)  silu(gate) * up
//   - residual_add      (lumen-runtime's activations.cu)  the residual adds
//
// Written here:
//   - embed_gather           gather the prompt's rows out of the embedding table
//   - rope_half_interleaved  the text tower's interleaved mRoPE
//   - attn_causal_gqa        causal grouped-query attention
//
// Two kernels that look reusable are not, and the reasons are worth recording
// because both failures would produce a correctly-shaped, finite, wrong tensor.
//
// *Rotation.* `image_ops.cu`'s `mrope_interleaved` rotates adjacent complex
// pairs *within* a head, pairing channel 2p with channel 2p+1, and takes a
// single real-frequency plane. `text_encoder.rs`'s `apply_rope` pairs channel j
// with channel j + head_dim/2 against one shared angle, over a `cos`/`sin` table
// that is itself interleaved across the rotary slots. The two are different
// layouts and not interchangeable, so `rope_half_interleaved` mirrors
// `apply_rope` channel for channel.
//
// *Attention.* `image_ops.cu`'s `attn_block_causal` indexes its keys and values
// with the *query* head index (`kbase = ((kj * heads) + h) * head_dim`), so it
// is one query head per key/value head by construction. This tower is grouped:
// 32 query heads over 8 key/value heads, with `groups = 4` consecutive query
// heads sharing one key/value head. Calling it here would read the wrong key
// heads and still return a finite tensor of the right shape. `attn_causal_gqa`
// carries the GQA mapping instead.
//
// NVRTC-compatible: no includes, extern "C" linkage.

#define TEXT_ATTN_THREADS 1024

// ---------------------------------------------------------------------------
// Widen one stored 16-bit float to f32.
//
// A second copy of `dit_ops.cu`'s `widen16` because NVRTC compiles each source
// into its own module, so a `__device__` helper in one is not visible to the
// other. Named apart from it so the two can never collide.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float embed_widen16(unsigned short h, unsigned int is_bf16)
{
    if (is_bf16 != 0) {
        // BF16 is the upper half of an IEEE f32.
        union { unsigned int u; float f; } cv;
        cv.u = ((unsigned int)h) << 16;
        return cv.f;
    }
    // f16 -> f32 is a single hardware convert (SM 53+).
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

// Little-endian scalar decoders, matching `text_encoder.rs::decode_row`.
__device__ __forceinline__ unsigned short embed_u16le(const unsigned char* p)
{
    return (unsigned short)((unsigned int)p[0] | ((unsigned int)p[1] << 8));
}

__device__ __forceinline__ float embed_f32le(const unsigned char* p)
{
    unsigned int u = (unsigned int)p[0] | ((unsigned int)p[1] << 8)
                   | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
    union { unsigned int u; float f; } cv;
    cv.u = u;
    return cv.f;
}

// ---------------------------------------------------------------------------
// Gather a prompt's embedding rows straight out of the stored table.
//
// `text_encoder.rs::embed` reads only the rows the prompt addresses, never the
// whole table, and this does the same on the device: the table stays resident in
// its stored dtype (BF16, ~1.16 GiB for the shipped 151936x4096 vocabulary) and
// the gathered rows are widened to f32 here, rather than expanding the whole
// table to f32 and adding ~1.3 GiB to a forward whose weights are 14.1 GiB.
//
// `dtype` is the `.lbi` storage scheme: 0 = F16, 1 = BF16, 2 = F32. Every id is
// range-checked against the vocabulary on the host before the upload, so
// `ids[row] < vocab` and each gathered row lies inside the buffer.
//
// One block per prompt position; `blockDim.x` threads stride over the hidden
// width.
// ---------------------------------------------------------------------------
extern "C" __global__ void embed_gather(
    const unsigned char* __restrict__ table,  // [vocab, hidden] at the stored width
    const unsigned int* __restrict__ ids,     // [seq]
    float* __restrict__ out,                  // [seq, hidden]
    unsigned int seq,
    unsigned int hidden,
    unsigned int dtype)
{
    unsigned int row = blockIdx.x;
    if (row >= seq) return;

    unsigned int width = (dtype == 2) ? 4u : 2u;
    const unsigned char* src = table + (unsigned long long)ids[row] * hidden * width;
    float* dst = out + (unsigned long long)row * hidden;

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const unsigned char* p = src + (unsigned long long)i * width;
        float v;
        if (dtype == 0) {
            v = embed_widen16(embed_u16le(p), 0u);
        } else if (dtype == 1) {
            v = embed_widen16(embed_u16le(p), 1u);
        } else {
            v = embed_f32le(p);
        }
        dst[i] = v;
    }
}

// ---------------------------------------------------------------------------
// The text tower's mRoPE, in place on one `[seq, heads, head_dim]` tensor.
//
// `text_encoder.rs::apply_rope` is, for every head:
//
//     out[j]          = x[j]          * cos[j]          - x[j + half] * sin[j]
//     out[j + half]   = x[j + half]   * cos[j + half]   + x[j]        * sin[j + half]
//
// with `half = head_dim / 2` and `cos`/`sin` the `[seq, head_dim]` table
// `rope_tables` builds (whose second half repeats its first). `v` carries no
// norm and no rotation; `q` and `k` are rotated by their own callers.
//
// One block per position, `blockDim.x` threads over the rotary slots, striding
// over the heads; the four trig values are hoisted out of the head loop because
// every head of a position shares them.
// ---------------------------------------------------------------------------
extern "C" __global__ void rope_half_interleaved(
    float* __restrict__ x,                // [seq, heads * head_dim], in place
    const float* __restrict__ cos_tab,    // [seq, head_dim]
    const float* __restrict__ sin_tab,    // [seq, head_dim]
    unsigned int seq,
    unsigned int heads,
    unsigned int head_dim)
{
    unsigned int s = blockIdx.x;
    unsigned int slots = head_dim >> 1;
    if (s >= seq) return;

    unsigned long long trig = (unsigned long long)s * head_dim;
    for (unsigned int j = threadIdx.x; j < slots; j += blockDim.x) {
        float cl = cos_tab[trig + j];
        float sl = sin_tab[trig + j];
        float ch = cos_tab[trig + j + slots];
        float sh = sin_tab[trig + j + slots];
        for (unsigned int h = 0; h < heads; h++) {
            unsigned long long base = ((unsigned long long)s * heads + h) * head_dim;
            float lo = x[base + j];
            float hi = x[base + j + slots];
            x[base + j] = lo * cl - hi * sl;
            x[base + j + slots] = hi * ch + lo * sh;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal grouped-query attention.
//
// Mirrors `text_encoder.rs::attention` one score at a time: the same `dot *
// scale`, the same max-subtraction softmax, the same `p * v` accumulation with
// keys ascending, and the same GQA mapping — `kv = h / groups`, so a block of
// `groups` consecutive query heads shares one key/value head, which is
// `repeat_interleave` over the head axis.
//
// Causality is by construction, as in the reference: query `qi` forms scores
// only for keys `0..=qi`. The reference adds -inf to the future scores and
// softmaxes them to zero, which is the same thing as never forming them.
//
// One block per (query position, query head); `blockDim.x` threads stride over
// the head dimension. Each thread walks the full head when forming a dot — the
// same redundancy `attn_block_causal` uses — so no block-wide reduction is
// needed and every thread computes the identical score. Only the owning thread
// writes each output channel, so a channel accumulates its keys in ascending
// order, exactly as the reference's inner loop does.
// ---------------------------------------------------------------------------
extern "C" __global__ void attn_causal_gqa(
    const float* __restrict__ q,     // [seq, nq * head_dim]
    const float* __restrict__ k,     // [seq, nkv * head_dim]
    const float* __restrict__ v,     // [seq, nkv * head_dim]
    float* __restrict__ out,         // [seq, nq * head_dim]
    unsigned int seq,
    unsigned int nq,
    unsigned int nkv,
    unsigned int head_dim,
    float scale)
{
    unsigned int qi = blockIdx.x;
    unsigned int h = blockIdx.y;
    if (qi >= seq || h >= nq) return;

    unsigned int groups = nq / nkv;
    unsigned int kvh = h / groups;
    unsigned int tid = threadIdx.x;

    extern __shared__ float sh[];
    float* qrow = sh;
    float* orow = sh + head_dim;

    unsigned long long qbase = ((unsigned long long)qi * nq + h) * head_dim;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        qrow[d] = q[qbase + d];
        orow[d] = 0.0f;
    }
    __syncthreads();

    unsigned long long kvoff = (unsigned long long)kvh * head_dim;

    // Pass 1: the peak score, keys ascending. -1e30 matches the sentinel
    // `attn_block_causal` uses; a query always has at least key 0, so the peak
    // is a real score by the end of the loop and the sentinel never survives.
    float peak = -1e30f;
    for (unsigned int kj = 0; kj <= qi; kj++) {
        const float* krow = k + (unsigned long long)kj * nkv * head_dim + kvoff;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += qrow[d] * krow[d];
        }
        float score = dot * scale;
        if (score > peak) peak = score;
    }

    // Pass 2: the softmax denominator.
    float total = 0.0f;
    for (unsigned int kj = 0; kj <= qi; kj++) {
        const float* krow = k + (unsigned long long)kj * nkv * head_dim + kvoff;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += qrow[d] * krow[d];
        }
        total += expf(dot * scale - peak);
    }
    float inv = 1.0f / total;

    // Pass 3: the weighted sum, one owning thread per output channel.
    for (unsigned int kj = 0; kj <= qi; kj++) {
        unsigned long long kbase = (unsigned long long)kj * nkv * head_dim + kvoff;
        const float* krow = k + kbase;
        const float* vrow = v + kbase;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += qrow[d] * krow[d];
        }
        float p = expf(dot * scale - peak) * inv;
        for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
            orow[d] += p * vrow[d];
        }
    }

    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        out[qbase + d] = orow[d];
    }
}
