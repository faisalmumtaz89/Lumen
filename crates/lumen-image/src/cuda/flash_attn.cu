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
// Shared tiles are 128 elements (256 bytes) a row with their 16-byte chunks
// XOR-swizzled by the row (`fa_swz`): the eight row reads of an `ldmatrix`
// land on eight distinct chunks, and eight threads copying eight adjacent
// chunks (half a row) fill one aligned 128-byte span, so neither the fragment
// loads nor the tile copies conflict. The swizzle only places values; the
// arithmetic is the same as with plain rows. Without padding the two tiles take
// 32 KiB, which with the kernel's registers leaves room for three blocks per
// multiprocessor on an RTX 5090.
//
// The key and value tiles arrive by `cp.async`, one buffer each: V(t) is copied
// while Q Kᵀ(t) and the softmax run, and K(t+1) while P V(t) runs, so the
// loads overlap the tensor-core work instead of preceding it. Only a tile past
// the sequence end or a block holding text rows can mask a score; every other
// tile takes a path without the mask arithmetic, and a row whose running
// maximum did not move skips the rescale of its accumulators (a factor of
// exactly 1). For scores above the masked sentinel (-1e29 in log2 units, far
// beyond any bf16 activations this model produces) each row's arithmetic is
// the same on every path; a row whose scores all fall below it is treated as
// masked only on the masking path.
//
// Compiled for compute_80 or later: bf16 `mma.sync`, `ldmatrix` and `cp.async`
// need it.
// NVRTC-compatible: no includes, extern "C" linkage.

#define FA_BQ 64
#define FA_BK 64
#define FA_D 128
#define FA_THREADS 128
// A score below this is masked; the running maxima start here.
#define FA_NEG_INF (-1e30f)
#define FA_MASKED (-1e29f)

// Element offset of (row, col) in a tile of 128-element rows whose 16-byte
// chunks are XOR-swizzled by the row; `col` is a multiple of 8.
__device__ __forceinline__ unsigned int fa_swz(unsigned int row, unsigned int col)
{
    return row * FA_D + (((col >> 3) ^ (row & 7u)) << 3);
}

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

// `cp.async`: a 16-byte global -> shared copy that bypasses the registers and
// completes in the background; `src_bytes` 0 writes zeros instead of reading.
__device__ __forceinline__ void fa_cp_async16(unsigned int dst, const void* src, unsigned int src_bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void fa_cp_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` of this thread's committed copy groups are pending.
#define FA_CP_WAIT(n) asm volatile("cp.async.wait_group " #n ";\n" ::)

// Start copying a 64x128 bf16 tile (rows `row0..row0+64` of one head) into
// swizzled shared memory; rows past `seq` are zero-filled.
__device__ __forceinline__ void fa_load_tile_async(
    unsigned short* dst,
    const unsigned short* __restrict__ src,
    unsigned int row0,
    unsigned int seq,
    unsigned int row_stride,
    unsigned int tid)
{
    for (unsigned int i = tid; i < FA_BK * (FA_D / 8); i += FA_THREADS) {
        unsigned int r = i / (FA_D / 8);
        unsigned int c = (i % (FA_D / 8)) * 8;
        unsigned int row = row0 + r;
        bool inside = row < seq;
        const unsigned short* from = src + (unsigned long long)(inside ? row : 0u) * row_stride + c;
        fa_cp_async16(fa_smem_addr(dst + fa_swz(r, c)), from, inside ? 16u : 0u);
    }
}

// Copy a 64x128 bf16 tile (rows `row0..row0+64` of one head) into swizzled
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
        *(uint4*)(dst + fa_swz(r, c)) = v;
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
    __shared__ __align__(128) unsigned short s_k[FA_BK * FA_D];
    __shared__ __align__(128) unsigned short s_v[FA_BK * FA_D];

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
                           fa_smem_addr(s_k + fa_swz(r, ks * 16 + c)));
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

    // The copies run in the background: V(t) arrives while Q Kᵀ(t) is
    // computed, and K(t+1) while P V(t) is.
    fa_load_tile_async(s_k, k + head_off, 0, seq, row_stride, tid);
    fa_cp_commit();
    for (unsigned int t = 0; t < tiles; t++) {
        const unsigned int k0 = t * FA_BK;
        fa_load_tile_async(s_v, v + head_off, k0, seq, row_stride, tid);
        fa_cp_commit();
        FA_CP_WAIT(1);   // K(t) has landed; V(t) may still be in flight
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
                fa_ldmatrix_x4(b0, b1, b2, b3, fa_smem_addr(s_k + fa_swz(key, kc)));
                fa_mma(s[np * 2], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
                fa_mma(s[np * 2 + 1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
            }
        }

        // Scale into log2 units and mask keys past the sequence and keys a
        // text query may not see. Only a tile past the sequence end or a block
        // holding text rows can mask anything; every other tile takes the
        // unmasked path.
        const bool masking = block_has_text || k0 + FA_BK > seq;
        float tmax0 = FA_NEG_INF, tmax1 = FA_NEG_INF;
        if (masking) {
            for (unsigned int n = 0; n < 8; n++) {
                unsigned int key = k0 + n * 8 + (lane & 3) * 2;
                for (unsigned int e = 0; e < 4; e++) {
                    unsigned int kk = key + (e & 1);
                    unsigned int qr = (e < 2) ? qr0 : qr1;
                    float val = s[n][e] * scale_log2;
                    bool masked = kk >= seq || (qr < text_count && kk > qr);
                    if (masked) val = FA_NEG_INF;
                    s[n][e] = val;
                    if (e < 2) tmax0 = fmaxf(tmax0, val); else tmax1 = fmaxf(tmax1, val);
                }
            }
        } else {
            for (unsigned int n = 0; n < 8; n++) {
                for (unsigned int e = 0; e < 4; e++) {
                    float val = s[n][e] * scale_log2;
                    s[n][e] = val;
                    if (e < 2) tmax0 = fmaxf(tmax0, val); else tmax1 = fmaxf(tmax1, val);
                }
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
        // Off the masking path no score is masked, so the masked cases below
        // arise there only for scores at or below the sentinel.
        float alpha0 = (masking && mnew0 <= FA_MASKED) ? 1.0f : exp2f(m0 - mnew0);
        float alpha1 = (masking && mnew1 <= FA_MASKED) ? 1.0f : exp2f(m1 - mnew1);
        float rs0 = 0.0f, rs1 = 0.0f;
        unsigned int p[4][4];   // P as A fragments: 4 k-slabs of 16 keys
        for (unsigned int n = 0; n < 8; n++) {
            float p0 = (masking && s[n][0] <= FA_MASKED) ? 0.0f : exp2f(s[n][0] - mnew0);
            float p1 = (masking && s[n][1] <= FA_MASKED) ? 0.0f : exp2f(s[n][1] - mnew0);
            float p2 = (masking && s[n][2] <= FA_MASKED) ? 0.0f : exp2f(s[n][2] - mnew1);
            float p3 = (masking && s[n][3] <= FA_MASKED) ? 0.0f : exp2f(s[n][3] - mnew1);
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
        // A row whose maximum did not move has alpha exactly 1, and scaling
        // by 1 is exact, so the rescale is skipped then.
        if (alpha0 != 1.0f || alpha1 != 1.0f) {
            for (unsigned int n = 0; n < 16; n++) {
                o[n][0] *= alpha0; o[n][1] *= alpha0;
                o[n][2] *= alpha1; o[n][3] *= alpha1;
            }
        }

        // Every warp is done with K(t) and V(t) has landed; start K(t+1).
        FA_CP_WAIT(0);
        __syncthreads();
        if (t + 1 < tiles) {
            fa_load_tile_async(s_k, k + head_off, k0 + FA_BK, seq, row_stride, tid);
            fa_cp_commit();
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
                fa_ldmatrix_x4_trans(b0, b1, b2, b3, fa_smem_addr(s_v + fa_swz(key, dc)));
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
