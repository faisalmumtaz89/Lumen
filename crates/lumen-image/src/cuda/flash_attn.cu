// Fused block-causal attention for the DiT and the text tower, bf16 operands on the tensor cores.
//
// `out[q] = softmax(scale · Q[q] · Kᵀ) · V` over one head, without ever writing
// the `[seq, seq]` scores: each block owns 64 query rows of one head and walks
// the keys in tiles of 64, keeping a running row maximum and row sum
// (the online softmax) and a f32 output accumulator in registers. The
// probabilities are rounded to bf16 for the P·V product, the way the
// upstream model's fused attention does; everything else stays f32.
//
// Mask: a text query (position < text_count) attends to keys [0, q]; an image
// query attends to every key. That is the reference's
// `(q >= kv) or same_image_block` rule for a text prefix followed by at most
// one image block; a text prefix of the whole sequence is plain causal
// attention.
//
// Layout: Q, K and V are `[seq, heads * 128]` bf16, one head's row being 128
// contiguous elements at column `head * 128`; the output is the same shape,
// rounded to bf16 (nearest even) because its consumer is the output
// projection, a bf16 GEMM.
//
// Warp work split: four warps, each owning 16 query rows
// and their accumulators, so no reduction ever crosses warps. Tensor-core
// fragments follow `mma.sync.m16n8k16` (row.col, bf16 in, f32 accumulate):
//   A (16x16): lane holds rows {lane/4, lane/4+8}, columns 2·(lane%4)+{0,1}
//              and 8 more;
//   B (16x8):  lane holds k = 2·(lane%4)+{0,1} (+8), column lane/4;
//   C (16x8):  lane holds rows {lane/4, lane/4+8}, columns 2·(lane%4)+{0,1}.
// The C layout of two adjacent 8-column tiles is the A layout of one 16-wide
// k-slab, which is what lets P feed the P·V product straight from registers.
//
// Shared rows are padded to 136 elements (272 bytes) so the eight row reads of
// an `ldmatrix` land in distinct bank groups.
//
// Compiled for compute_80 or later: bf16 `mma.sync` and `ldmatrix` need it.
// NVRTC-compatible: no includes, extern "C" linkage.

#define FA_BQ 64
#define FA_BK 64
#define FA_D 128
#define FA_PAD 136
#define FA_THREADS 128
// A score below this is masked; the running maxima start here.
#define FA_NEG_INF (-1e30f)
#define FA_MASKED (-1e29f)

__device__ __forceinline__ unsigned int fa_pack_bf16(float lo, float hi)
{
    // Round to nearest even, as the upstream model's bf16 cast does.
    unsigned int a = __float_as_uint(lo);
    unsigned int b = __float_as_uint(hi);
    a += 0x7fffu + ((a >> 16) & 1u);
    b += 0x7fffu + ((b >> 16) & 1u);
    return (a >> 16) | (b & 0xffff0000u);
}

__device__ __forceinline__ unsigned int fa_smem_addr(const void* p)
{
    return (unsigned int)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void fa_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fa_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    unsigned int addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fa_mma(
    float* c,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Copy a 64x128 bf16 tile (rows `row0..row0+64` of one head) into padded
// shared memory; rows past `seq` are zero-filled.
__device__ __forceinline__ void fa_load_tile(
    unsigned short* dst,
    const unsigned short* __restrict__ src,
    unsigned int row0,
    unsigned int seq,
    unsigned int row_stride,
    unsigned int tid)
{
    // 64 rows x 16 chunks of 8 elements; 128 threads take 8 chunks each.
    for (unsigned int i = tid; i < FA_BK * (FA_D / 8); i += FA_THREADS) {
        unsigned int r = i / (FA_D / 8);
        unsigned int c = (i % (FA_D / 8)) * 8;
        unsigned int row = row0 + r;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (row < seq) {
            v = *(const uint4*)(src + (unsigned long long)row * row_stride + c);
        }
        *(uint4*)(dst + r * FA_PAD + c) = v;
    }
}

extern "C" __global__ void __launch_bounds__(FA_THREADS)
flash_attn_bf16(
    const unsigned short* __restrict__ q,   // [seq, heads * 128] bf16
    const unsigned short* __restrict__ k,   // [seq, heads * 128] bf16
    const unsigned short* __restrict__ v,   // [seq, heads * 128] bf16
    unsigned short* __restrict__ out,       // [seq, heads * 128] bf16
    unsigned int seq,
    unsigned int heads,
    unsigned int text_count,
    float scale_log2)                       // softmax scale times log2(e)
{
    __shared__ __align__(16) unsigned short s_k[FA_BK * FA_PAD];
    __shared__ __align__(16) unsigned short s_v[FA_BK * FA_PAD];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int head = blockIdx.y;
    const unsigned int q0 = blockIdx.x * FA_BQ;
    const unsigned int row_stride = heads * FA_D;
    const unsigned int head_off = head * FA_D;

    // The two query rows this lane's fragments hold.
    const unsigned int qr0 = q0 + warp * 16 + (lane >> 2);
    const unsigned int qr1 = qr0 + 8;

    // Q fragments for this warp's 16 rows: 8 k-slabs of 16, staged through
    // the K buffer before the first key tile arrives.
    unsigned int qa[8][4];
    fa_load_tile(s_k, q + head_off, q0, seq, row_stride, tid);
    __syncthreads();
    {
        // ldmatrix.x4: lanes 0-7 address matrix 0 (rows 0-7, k 0-7), 8-15
        // matrix 1 (rows 8-15, k 0-7), 16-23 matrix 2 (rows 0-7, k 8-15),
        // 24-31 matrix 3 (rows 8-15, k 8-15): the a0..a3 order.
        unsigned int r = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        unsigned int c = (lane >> 4) * 8;
        for (unsigned int ks = 0; ks < 8; ks++) {
            fa_ldmatrix_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3],
                           fa_smem_addr(s_k + r * FA_PAD + ks * 16 + c));
        }
    }
    __syncthreads();

    float o[16][4];
    for (unsigned int n = 0; n < 16; n++) {
        o[n][0] = 0.0f; o[n][1] = 0.0f; o[n][2] = 0.0f; o[n][3] = 0.0f;
    }
    float m0 = FA_NEG_INF, m1 = FA_NEG_INF;   // running row maxima (scaled log2 units)
    float l0 = 0.0f, l1 = 0.0f;       // running row sums

    // A text row sees only keys up to itself; this block has such rows only if
    // it starts inside the text prefix.
    const bool block_has_text = q0 < text_count;
    const unsigned int tiles = (seq + FA_BK - 1) / FA_BK;

    for (unsigned int t = 0; t < tiles; t++) {
        const unsigned int k0 = t * FA_BK;
        fa_load_tile(s_k, k + head_off, k0, seq, row_stride, tid);
        fa_load_tile(s_v, v + head_off, k0, seq, row_stride, tid);
        __syncthreads();

        // S = Q Kᵀ for this warp's 16 rows x 64 keys: 8 n-tiles of 8 keys.
        float s[8][4];
        for (unsigned int n = 0; n < 8; n++) {
            s[n][0] = 0.0f; s[n][1] = 0.0f; s[n][2] = 0.0f; s[n][3] = 0.0f;
        }
        for (unsigned int ks = 0; ks < 8; ks++) {
            // Two n-tiles per ldmatrix.x4: matrices 0,1 are keys n..n+7 at
            // k-halves 0 and 8 (b0, b1 of tile n); 2,3 the same for tile n+1.
            for (unsigned int np = 0; np < 4; np++) {
                unsigned int key = np * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                unsigned int kc = ks * 16 + ((lane >> 3) & 1) * 8;
                unsigned int b0, b1, b2, b3;
                fa_ldmatrix_x4(b0, b1, b2, b3, fa_smem_addr(s_k + key * FA_PAD + kc));
                fa_mma(s[np * 2], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
                fa_mma(s[np * 2 + 1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
            }
        }

        // Scale into log2 units and mask: keys past the sequence, and keys a
        // text query may not see.
        const bool partial = k0 + FA_BK > seq;
        float tmax0 = FA_NEG_INF, tmax1 = FA_NEG_INF;
        for (unsigned int n = 0; n < 8; n++) {
            unsigned int key = k0 + n * 8 + (lane & 3) * 2;
            for (unsigned int e = 0; e < 4; e++) {
                unsigned int kk = key + (e & 1);
                unsigned int qr = (e < 2) ? qr0 : qr1;
                float val = s[n][e] * scale_log2;
                bool masked = (partial && kk >= seq)
                    || (block_has_text && qr < text_count && kk > qr);
                if (masked) val = FA_NEG_INF;
                s[n][e] = val;
                if (e < 2) tmax0 = fmaxf(tmax0, val); else tmax1 = fmaxf(tmax1, val);
            }
        }
        // The four lanes sharing a row combine their maxima.
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));

        float mnew0 = fmaxf(m0, tmax0);
        float mnew1 = fmaxf(m1, tmax1);
        // A row with nothing visible yet keeps its accumulators at zero.
        float alpha0 = (mnew0 <= FA_MASKED) ? 1.0f : exp2f(m0 - mnew0);
        float alpha1 = (mnew1 <= FA_MASKED) ? 1.0f : exp2f(m1 - mnew1);
        float rs0 = 0.0f, rs1 = 0.0f;
        unsigned int p[4][4];   // P as A fragments: 4 k-slabs of 16 keys
        for (unsigned int n = 0; n < 8; n++) {
            float p0 = (s[n][0] <= FA_MASKED) ? 0.0f : exp2f(s[n][0] - mnew0);
            float p1 = (s[n][1] <= FA_MASKED) ? 0.0f : exp2f(s[n][1] - mnew0);
            float p2 = (s[n][2] <= FA_MASKED) ? 0.0f : exp2f(s[n][2] - mnew1);
            float p3 = (s[n][3] <= FA_MASKED) ? 0.0f : exp2f(s[n][3] - mnew1);
            rs0 += p0 + p1;
            rs1 += p2 + p3;
            // C(tile n) -> A(slab n/2): even n gives a0/a1, odd n a2/a3.
            unsigned int slab = n >> 1;
            if ((n & 1) == 0) {
                p[slab][0] = fa_pack_bf16(p0, p1);
                p[slab][1] = fa_pack_bf16(p2, p3);
            } else {
                p[slab][2] = fa_pack_bf16(p0, p1);
                p[slab][3] = fa_pack_bf16(p2, p3);
            }
        }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * alpha0 + rs0;
        l1 = l1 * alpha1 + rs1;
        m0 = mnew0;
        m1 = mnew1;
        for (unsigned int n = 0; n < 16; n++) {
            o[n][0] *= alpha0; o[n][1] *= alpha0;
            o[n][2] *= alpha1; o[n][3] *= alpha1;
        }

        // O += P V: 4 k-slabs of 16 keys x 16 n-tiles of 8 output columns.
        // V is [key][d] in shared memory, so the B fragments (k = key) come
        // from a transposed ldmatrix over eight key rows.
        for (unsigned int slab = 0; slab < 4; slab++) {
            for (unsigned int np = 0; np < 8; np++) {
                // Matrices 0,1: d-tile np*2, keys 0-7 and 8-15 of the slab;
                // matrices 2,3: d-tile np*2+1.
                unsigned int key = slab * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                unsigned int dc = np * 16 + ((lane >> 4) & 1) * 8;
                unsigned int b0, b1, b2, b3;
                fa_ldmatrix_x4_trans(b0, b1, b2, b3, fa_smem_addr(s_v + key * FA_PAD + dc));
                fa_mma(o[np * 2], p[slab][0], p[slab][1], p[slab][2], p[slab][3], b0, b1);
                fa_mma(o[np * 2 + 1], p[slab][0], p[slab][1], p[slab][2], p[slab][3], b2, b3);
            }
        }
        __syncthreads();
    }

    // Normalise and store this lane's two rows, two columns at a time.
    float inv0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    float inv1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    for (unsigned int n = 0; n < 16; n++) {
        unsigned int col = head_off + n * 8 + (lane & 3) * 2;
        if (qr0 < seq) {
            *(unsigned int*)(out + (unsigned long long)qr0 * row_stride + col) =
                fa_pack_bf16(o[n][0] * inv0, o[n][1] * inv0);
        }
        if (qr1 < seq) {
            *(unsigned int*)(out + (unsigned long long)qr1 * row_stride + col) =
                fa_pack_bf16(o[n][2] * inv1, o[n][3] * inv1);
        }
    }
}
