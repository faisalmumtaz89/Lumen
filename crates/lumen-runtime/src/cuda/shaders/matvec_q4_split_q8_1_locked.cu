// ==========================================================================
// Codegen-LOCKED Q4_0 split-layout matvec against pre-quantized Q8_1 input.
//
// Bit-for-bit deterministic sibling of matvec_q4_split_q8_1. Consumes the SAME
// per-row split (SoA) layout produced by repack_q4_raw_to_split:
//
//   Per row (in `nb`-block units):
//     [f16 scale * nb][nibble[16] * nb]
//   Row stride: 2*nb + 16*nb = 18*nb bytes
//   Scale stream at row offset 0; nibble stream at row offset 2*nb.
//
// DETERMINISM
// -----------
// The SoA weight load order (AoS vs SoA) can perturb the last bit of a plain
// `float` F32 accumulator under --use_fast_math (the NVRTC->driver JIT is free to
// contract/reassociate), which flips a small number of argmax near-ties at decode
// time and fails the byte-identical-greedy correctness gate. This kernel pins the
// per-block epilogue, the warp all-reduce, and the cross-warp fold with inline-PTX
// `.rn` (round-to-nearest-even, no contraction) so the F32 output op SEQUENCE is
// fixed and independent of the weight load order. The integer dp4a dot is exact
// and order-independent, and block visitation is fixed (`ib = threadIdx.x; ib +=
// 256`) and identical to the unlocked kernel.
//
// The nibble stream is read as native 4-byte `unsigned int` loads (4 per block)
// with bitmask de-interleave -- coalesced 32-bit transactions. The resulting
// integer dp4a accumulator is identical regardless of how the nibble bytes are
// fetched, so this load width is a pure throughput choice, not a numeric one.
//
// Dequant identity (Q4_0 zero-point -8):
//   dot(w, x) = w_scale * (x_scale * Sum(nibble_i * q_i) - 8 * x_sum)
// where Sum(nibble_i * q_i) is the EXACT int dp4a accumulator and x_sum =
// x_scale * Sum(q_i) is precomputed in the Q8_1 input block.
//
// Two kernels:
//   1. matvec_q4_split_q8_1_locked:          W*x -> out
//   2. matvec_q4_split_q8_1_locked_residual: W*x + residual -> out
//
// Requires compute capability >= 6.1 for dp4a (Pascal+).
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR       4
#define NW       32
#define THREADS_PER_BLOCK 256
#define NWARPS   (THREADS_PER_BLOCK / NW)
#define Q8_1_BYTES        36

__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// THE LOCK -- pinned add.rn.
__device__ __forceinline__ float fadd_rn_locked(float a, float b) {
    float o;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(a), "f"(b));
    return o;
}

// LOCKED epilogue (pinned fma.rn op sequence).
__device__ __forceinline__ float q4_epilogue_locked(
    float acc, int dot_i32, float w_scale, float x_scale, float x_sum)
{
    const float km8 = -8.0f;
    asm volatile(
        "{\n\t"
        "  .reg .f32 fdot, inner;\n\t"
        "  cvt.rn.f32.s32 fdot, %1;\n\t"
        "  mul.rn.f32     fdot, %3, fdot;\n\t"
        "  fma.rn.f32     inner, %5, %4, fdot;\n\t"
        "  fma.rn.f32     %0, %2, inner, %0;\n\t"
        "}\n\t"
        : "+f"(acc)
        : "r"(dot_i32), "f"(w_scale), "f"(x_scale), "f"(x_sum), "f"(km8));
    return acc;
}

__device__ __forceinline__ float shfl_xor_f32_locked(float v, int m) {
    unsigned in = __float_as_uint(v), out;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=r"(out) : "r"(in), "r"(m), "r"(0xffffffffu));
    return __uint_as_float(out);
}

// LOCKED warp all-reduce.
__device__ __forceinline__ float warp_allreduce_sum_locked(float v) {
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 16));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 8));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 4));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 2));
    v = fadd_rn_locked(v, shfl_xor_f32_locked(v, 1));
    return v;
}

// Original de-interleaved unpack: lo/hi nibbles via bitmask (NOT byte loads).
__device__ __forceinline__ void unpack_nibbles_4bytes_deinterleaved(
    unsigned int packed, int &out_lo, int &out_hi)
{
    unsigned int lo = packed & 0x0F0F0F0Fu;
    unsigned int hi = (packed >> 4) & 0x0F0F0F0Fu;
    out_lo = (int)lo;
    out_hi = (int)hi;
}

// LOCKED reduction + store.
__device__ __forceinline__ void reduce_and_store_locked(
    float* __restrict__ out, float sumf[NR], unsigned int r0, unsigned int out_dim,
    unsigned int warp_id, unsigned int lane, const float* __restrict__ residual)
{
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_allreduce_sum_locked(sumf[r]);
    }

    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[(warp_id - 1) * NR + r] = sumf[r];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++) {
                total = fadd_rn_locked(total, shmem[w * NR + r]);
            }
            if (r0 + r < out_dim) {
                if (residual != 0) {
                    total = fadd_rn_locked(total, residual[r0 + r]);
                }
                out[r0 + r] = total;
            }
        }
    }
}

__device__ __forceinline__ void matvec_q4_split_body(
    const char* __restrict__ weight_q4_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim,
    const float* __restrict__ residual,
    unsigned int vblock)
{
    unsigned int r0 = vblock * NR;
    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * 18ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // In-bounds row count (last CTA may be partial; out_dim%NR is 0 for the
    // shipped shapes, so active_rows==NR on the hot path). Hoisted out of the
    // K-loop so Phase A below issues a clean, branch-light burst of loads.
    unsigned int active_rows = out_dim - r0;
    if (active_rows > (unsigned int)NR) active_rows = NR;

    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);
        unsigned short x_sum_bits = *(const unsigned short*)(x_block + 2);
        float x_sum = f16_bits_to_f32(x_sum_bits);
        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // ---- PHASE A: issue ALL row loads up-front to raise memory-level
        // parallelism (the load-hoist "unlock"). The base interleaved load->use
        // per row, leaving little distance between a global load and its
        // consumption (long-scoreboard stalls). Hoisting all NR rows' weights
        // into registers first makes the NR*4 nibble loads mutually independent,
        // so many are in flight at once to hide DRAM latency. The per-row sumf[]
        // accumulate ORDER in Phase B is UNCHANGED, so the F32 reduction sequence
        // is byte-identical -> gate-clean. Loads stay 4-byte `unsigned int` (the
        // proven word-load width): 2*nb is always a multiple of 4 since nb is
        // even, so no alignment constraint beyond the base's natural alignment.
        unsigned int wpk[NR][4];
        unsigned short wsb[NR];
        const unsigned long long nibble_off = scales_bytes_per_row
            + (unsigned long long)ib * 16ULL;
        const unsigned long long scale_off = (unsigned long long)ib * 2ULL;
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if ((unsigned int)row >= active_rows) break;
            const char* row_base = weight_q4_split
                + (unsigned long long)(r0 + row) * row_bytes;
            wsb[row] = *(const unsigned short*)(row_base + scale_off);
            const unsigned int* w_nibbles =
                (const unsigned int*)(row_base + nibble_off);
#ifdef LUMEN_Q4_V4LOAD
            // 128-bit load variant (compile-time define, absent by default):
            // one uint4 transaction replaces four u32 loads. Host guarantees
            // 16-byte alignment (2*nb % 16 == 0 for the dispatched shapes;
            // base is cudaMalloc-256-aligned). Integer loads are exact — the
            // unpacked words and everything downstream are bit-identical.
            {
                const uint4 wv4 = *(const uint4*)w_nibbles;
                wpk[row][0] = wv4.x;
                wpk[row][1] = wv4.y;
                wpk[row][2] = wv4.z;
                wpk[row][3] = wv4.w;
            }
#else
            #pragma unroll
            for (int k = 0; k < 4; k++) wpk[row][k] = w_nibbles[k];
#endif
        }

        // ---- PHASE B: locked accumulate, rows 0..NR-1 in the SAME order, with
        // the SAME per-k (lo->xv[k], hi->xv[k+4]) sequence and locked epilogue.
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if ((unsigned int)row >= active_rows) break;
            float w_scale = f16_bits_to_f32(wsb[row]);
            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int w_lo, w_hi;
                unpack_nibbles_4bytes_deinterleaved(wpk[row][k], w_lo, w_hi);
                acc = dp4a_s32(w_lo, xv[k],     acc);   // lo nibbles -> xv[k]
                acc = dp4a_s32(w_hi, xv[k + 4], acc);   // hi nibbles -> xv[k+4]
            }
            // LOCKED epilogue (integer acc is bit-identical to the byte-load path).
            sumf[row] = q4_epilogue_locked(sumf[row], acc, w_scale, x_scale, x_sum);
        }
    }

    reduce_and_store_locked(out, sumf, r0, out_dim, warp_id, lane, residual);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1_locked(
    const char* __restrict__ weight_q4_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_split_body(weight_q4_split, input_q8_1, out, out_dim, in_dim, 0, blockIdx.x);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1_locked_residual(
    const char* __restrict__ weight_q4_split,
    const char* __restrict__ input_q8_1,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q4_split_body(weight_q4_split, input_q8_1, out, out_dim, in_dim, residual, blockIdx.x);
}

// Banked two-weight variant: ONE launch covers weight A's rows then weight B's
// rows, both against the SAME pre-quantized Q8_1 input. Removes a launch and
// lets B's CTAs fill the tail of A's final wave. The block-level pointer/index
// select is uniform per CTA and the per-row body (locked epilogue, locked
// reductions, fixed block visitation) is untouched, so each output element is
// bit-identical to the two-launch route.
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1_locked_banked(
    const char* __restrict__ weight_a,
    const char* __restrict__ weight_b,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out_a_ptr,
    float* __restrict__ out_b_ptr,
    unsigned int out_a,
    unsigned int out_b,
    unsigned int in_dim)
{
    unsigned int grid_a = (out_a + (unsigned int)NR - 1u) / (unsigned int)NR;
    if (blockIdx.x < grid_a) {
        matvec_q4_split_body(weight_a, input_q8_1, out_a_ptr, out_a, in_dim, 0, blockIdx.x);
    } else {
        matvec_q4_split_body(weight_b, input_q8_1, out_b_ptr, out_b, in_dim, 0,
                             blockIdx.x - grid_a);
    }
}


// Four-way banked variant: one launch covers qkv, gate, alpha, beta rows —
// all against the SAME pre-quantized Q8_1 input. Same virtual-row-concat
// construction as the two-way banked kernel (block-uniform select, untouched
// per-row body) => bit-identical outputs; removes three launch boundaries
// and lets the tiny alpha/beta grids (12 CTAs each at out_dim 48) ride the
// tail of the large grids instead of paying two standalone ramp/drains.
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1) void matvec_q4_split_q8_1_locked_bank4(
    const char* __restrict__ weight_a,
    const char* __restrict__ weight_b,
    const char* __restrict__ weight_c,
    const char* __restrict__ weight_d,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out_a_ptr,
    float* __restrict__ out_b_ptr,
    float* __restrict__ out_c_ptr,
    float* __restrict__ out_d_ptr,
    unsigned int out_a,
    unsigned int out_b,
    unsigned int out_c,
    unsigned int out_d,
    unsigned int in_dim)
{
    const unsigned int nr = (unsigned int)NR;
    unsigned int grid_a = (out_a + nr - 1u) / nr;
    unsigned int grid_b = (out_b + nr - 1u) / nr;
    unsigned int grid_c = (out_c + nr - 1u) / nr;
    unsigned int vb = blockIdx.x;
    if (vb < grid_a) {
        matvec_q4_split_body(weight_a, input_q8_1, out_a_ptr, out_a, in_dim, 0, vb);
        return;
    }
    vb -= grid_a;
    if (vb < grid_b) {
        matvec_q4_split_body(weight_b, input_q8_1, out_b_ptr, out_b, in_dim, 0, vb);
        return;
    }
    vb -= grid_b;
    if (vb < grid_c) {
        matvec_q4_split_body(weight_c, input_q8_1, out_c_ptr, out_c, in_dim, 0, vb);
        return;
    }
    vb -= grid_c;
    matvec_q4_split_body(weight_d, input_q8_1, out_d_ptr, out_d, in_dim, 0, vb);
}
