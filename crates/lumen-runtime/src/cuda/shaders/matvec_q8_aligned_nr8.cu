// ==========================================================================
// Q8 mmvq variant: AoS (36B aligned) layout, 4-threads-per-block + NR=8.
//
// Combines the 4-threads-per-block dp4a-mmvq pattern (vdr=2,
// blocks_per_iter=32, NR=8) with the AoS 36-byte block layout.
//
// Sibling kernels:
// - `matvec_q8_split_q8_1_nr8.cu`     : SPLIT (SoA, 34B) + 4-thread + NR=8
// -> REGRESSED at FFN shapes (-37.5% / -21.6%) and e2e (-4.85%)
// - `matvec_q8_split_q8_1_4thread.cu` : SPLIT (SoA, 34B) + 4-thread + NR=2
// -> 1.25x at 4096x4096, -7-10% at FFN shapes
// - Production `matvec_q8_aligned_q8_1.cu` : AoS (36B) + Lumen vdr=8 + NR=2
// -> baseline (this kernel's target to beat)
// - Production `matvec_q8_split_q8_1.cu` : SPLIT (SoA, 34B) + Lumen vdr=8 + NR=2
// -> secondary baseline (active when env-gated SPLIT dispatch is set)
//
// The combination tested here: AoS + 4-thread mapping + NR=8.
//
// AoS-specific motivation:
// The SPLIT layout has the scale stream `2*nb` bytes away from the matching
// quant byte (scales packed at row_base, quants at row_base + 2*nb). At
// in_dim=12288, that's `2*384 = 768 B` between an f16 scale read and the
// matching 8 B of quants -- never coalesced into the same L1 sector, never
// sharing the same L2 line. The AoS layout colocates them within 36 bytes
// of each other; an L1 sector covers a full block including its scale.
//
// If the FFN-shape regression of the SPLIT NR8 variant is driven by
// scale-stream cache pressure (separate from the quant stream), AoS should
// show a different verdict. If AoS+NR=8 also loses, the regression is
// compute/scheduling on the dp4a chain itself — independent of layout —
// and the structural ceiling is locked decisively.
//
// Layout consumed (UNCHANGED, same Q8Aligned bytes as production AoS kernel):
// Block stride: 36 bytes per Q8_0 block
// Per block: [f16 scale (2B)][pad (2B)][int8 quants (32B)]
// Row stride: nb * 36 bytes
// Quant data lives at offset +4 of each block, 4-byte aligned. Native
// int* loads are valid (PROD AoS kernel already relies on this).
//
// Constants (dp4a-mmvq + NR=8):
// QK = 32 elements / Q8_0 block
// QI = QK / 4 = 8 int32s / Q8_0 block
// VDR = 2 int32s consumed / dp4a inner call
// NWARPS = 4 4 warps / CTA
// WARP_SIZE = 32
// THREADS = NWARPS * 32 = 128
// NR = 8 rows / CTA (the UNTRIED parameter on AoS)
// KQS_PER_BLOCK = QI / VDR = 4 threads cooperating on a Q8 block
// BLOCKS_PER_ITER = VDR * THREADS / QI = 32 blocks advanced per K-iter
//
// Thread mapping (4-threads-per-block):
// tid = threadIdx.x 0..127
// kbx_start = tid / 4 0..31, 4 threads share a block
// kqs = vdr * (tid % 4) = 2 * (tid%4) ∈ {0,2,4,6} int32 offset
// for (kbx = kbx_start; kbx < nb; kbx += 32) {
// load f16 input scale from x_block(kbx) + 0
// load 2 int32 input quants from x_block(kbx) + 4 + kqs*4
// for (row = 0; row < 8; row++) {
// load f16 weight scale from w_block(r0+row, kbx) + 0
// load 2 int32 weight quants from w_block(r0+row, kbx) + 4 + kqs*4
// tmp[row] += w_scale * x_scale * dp4a_chain(w_int[0..1], x_int[0..1])
// }
// }
//
// Cross-warp reduction (same as the SPLIT NR=8 variant):
// __shared__ float tmp_shared[NWARPS - 1][NR][NW] // 3*8*32 = 768 floats = 3 KB
// warps 1..3 store per-row partials at (warp-1, row, lane)
// warp 0 reads them back, sums lane-wise, warp_reduce_sum, then lane r
// writes dst[r0 + r] for r < NR=8.
//
// K-loop trip count:
// in_dim=4096 -> nb=128 -> trip=4
// in_dim=12288 -> nb=384 -> trip=12
// in_dim=8192 -> nb=256 -> trip=8
// in_dim=1024 -> nb=32 -> trip=1 (small-K shape; the K-trip≥4 advantage degenerates)
//
// All ≥ 4 except in_dim=1024 (K, V proj). The 1024 path falls back to
// production AoS automatically via the shape gate in the dispatcher.
//
// Two kernels (mirror existing AoS pair `matvec_q8_aligned_q8_1`):
// 1. matvec_q8_aligned_nr8: W*x -> out
// 2. matvec_q8_aligned_nr8_residual: W*x + residual -> out
//
// Env gate: `LUMEN_CUDA_Q8_AOS_NR8=1` selects these in place of the
// production AoS kernels at decode dispatch. Default OFF (default-off contract).
// Mutually exclusive with `LUMEN_CUDA_Q8_SPLIT_NR8` (which is the SPLIT-layout
// variant); if both are set the AoS version wins (the rationale here is that
// AoS may unlock what SPLIT couldn't).
//
// Requires compute capability >= 6.1 for dp4a_s32().
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR              8     // rows per thread block  ★ AoS UNTRIED PARAMETER ★
#define NW              32    // warp size
#define NWARPS          4     // 4 warps per CTA (calc_nwarps for ncols=1)
#define THREADS_PER_BLOCK (NWARPS * NW)   // 128

#define QK              32    // elements per Q8_0 block
#define QI              8     // int32s per Q8_0 block (= QK / 4)
#define VDR             2     // int32s consumed per dp4a inner call (VDR_Q8_0_Q8_1_MMVQ)
#define KQS_PER_BLOCK   (QI / VDR)        // 4 (= threads cooperating on one Q8_0 block)
#define BLOCKS_PER_ITER ((VDR * THREADS_PER_BLOCK) / QI)   // 32

#define Q8_ALIGNED_BYTES  36    // 2B f16 scale + 2B pad + 32B int8 quants
#define Q8_1_BYTES        36    // 2B f16 scale + 2B f16 sum + 32B int8 quants

// Hardware f16->f32 conversion (single PTX instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. The `__dp4a` intrinsic NVRTC-fails in
// this build env (driver 580.126.20 / CUDA 12.2 / sm_80). The inline
// `dp4a.s32.s32` opcode loads cleanly on compute_80.
__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Warp-level reduction over all 32 lanes (butterfly shuffle).
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ==========================================================================
// Kernel 1: NR=8 Q8 AoS x Q8_1 -> F32 (no residual).
//
// Grid: (ceil(out_dim / NR), 1, 1)
// Block: (THREADS_PER_BLOCK, 1, 1) -- 128 threads, 4 warps
//
// __launch_bounds__(128, 1): minimum CTAs/SM = 1, no maximum; ptxas picks
// the register-bounded occupancy. NR=8 brings 8 floats of `tmp[]` plus 2
// int x+ 2 int win registers (~32-38 regs/thread expected).
//
// At NR=8: 5 CTAs/SM × 4 warps/CTA = 20 warps/SM.
//
// Per-CTA byte traffic per K-iter at in_dim=12288 (FFN down, trip=12):
// x: 8 B quant (2 int32 shared by 4 threads cooperating per Q8 block)
// x: 2 B scale (1 f16 shared by 4 threads)
// w: 8 rows × (8 B quant + 2 B scale) = 80 B (1 row block = 36 B, 8 rows)
// total per K-iter: 90 B/CTA × trip 12 = 1080 B/CTA aggregate.
//
// IDENTICAL byte counts to the SPLIT NR8 port (same number of LDGs).
// The DIFFERENCE is layout: scales and quants live in the same 36-byte
// block (interleaved with pad), so an L1 sector (32 B) covers all 8 quant
// bytes consumed by 4 cooperating threads, AND the f16 scale comes from
// the same 32 B sector (offset 0-1 vs offset 8 + kqs*4 within the block).
// The L2 line (128 B) covers ~3.5 blocks of one row = ~28 elements ≈ 1
// dp4a chain step's worth of weight data.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_aligned_nr8(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36]
    const char* __restrict__ input_q8_1,         // [nb * 36]
    float* __restrict__ out,                     // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / NW;
    unsigned int lane    = tid % NW;

    unsigned int nb = in_dim >> 5;       // in_dim / 32 = number of Q8 blocks
    unsigned long long row_bytes = (unsigned long long)nb * (unsigned long long)Q8_ALIGNED_BYTES;

    // 4-threads-per-block: 4 threads cooperate per Q8_0 block, each handles vdr=2 int32s.
    unsigned int kbx_start = tid / KQS_PER_BLOCK;        // 0..31 — initial K-block id
    unsigned int kqs_int   = VDR * (tid % KQS_PER_BLOCK); // 0, 2, 4, 6 — int32 offset

    // Per-row accumulators in registers. NR=8 floats per thread.
    float tmp[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) tmp[r] = 0.0f;

    // Main K-loop. blocks_per_iter=32 → trip ≥ 4 at all FFN shapes.
    for (unsigned int kbx = kbx_start; kbx < nb; kbx += BLOCKS_PER_ITER) {

        // --- Load Q8_1 input block scale + 2 int32 quants (SHARED across NR rows) ---
        // Q8_1 block: bytes 0-1 = f16 scale, 2-3 = f16 sum (unused for Q8_0 zero_point=0),
        // 4-35 = int8 quants. Native int* load at byte offset (4 + kqs_int*4).
        const char* x_block = input_q8_1 + (unsigned long long)kbx * (unsigned long long)Q8_1_BYTES;
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);
        const int* x_packed = (const int*)(x_block + 4) + kqs_int;
        int x0 = x_packed[0];
        int x1 = x_packed[1];

        // --- Process NR=8 output rows for this (kbx, kqs) ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            unsigned int target_row = r0 + (unsigned int)row;
            if (target_row >= out_dim) break;

            // AoS block address: row * row_bytes + kbx * 36
            const char* w_block = weight_q8_aligned
                + (unsigned long long)target_row * row_bytes
                + (unsigned long long)kbx * (unsigned long long)Q8_ALIGNED_BYTES;

            // f16 weight scale at offset 0-1 of the block.
            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // 2 int32s of weight quants at offset +4 + kqs_int*4 (4-byte aligned).
            const int* w_packed = (const int*)(w_block + 4) + kqs_int;
            int w0 = w_packed[0];
            int w1 = w_packed[1];

            // dp4a chain: 2 int32 pairs = 8 INT8 multiply-accumulates.
            int acc = dp4a_s32(w0, x0, 0);
            acc     = dp4a_s32(w1, x1, acc);

            tmp[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction (shmem reduction) ---
    // tmp_shared[nwarps-1][NR][warp_size] = [3][8][32] floats = 768 floats = 3 KB
    __shared__ float tmp_shared[NWARPS - 1][NR][NW];

    if (warp_id > 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            tmp_shared[warp_id - 1][r][lane] = tmp[r];
        }
    }
    __syncthreads();

    if (warp_id > 0) return;

    // Warp 0: accumulate partials from warps 1..3 (lane-wise), warp_reduce.
    // After warp_reduce_sum, every lane holds the same total for row r.
    // Lane `r` (r < NR=8) writes dst[r0 + r] for its corresponding row.
    float total[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        float t = tmp[r];
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++) {
            t += tmp_shared[w][r][lane];
        }
        total[r] = warp_reduce_sum(t);
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        if (lane == r && (r0 + (unsigned int)r) < out_dim) {
            out[r0 + (unsigned int)r] = total[r];
        }
    }
}

// ==========================================================================
// Kernel 2: NR=8 Q8 AoS x Q8_1 + residual -> F32.
//
// Identical to kernel 1 with `out[r] = total + residual[r]` at the final write.
// Used for Wo (attention output projection) and FFN down projection.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_aligned_nr8_residual(
    const char* __restrict__ weight_q8_aligned,  // [out_dim * nb * 36]
    const char* __restrict__ input_q8_1,         // [nb * 36]
    const float* __restrict__ residual,          // [out_dim]
    float* __restrict__ out,                     // [out_dim]
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / NW;
    unsigned int lane    = tid % NW;

    unsigned int nb = in_dim >> 5;
    unsigned long long row_bytes = (unsigned long long)nb * (unsigned long long)Q8_ALIGNED_BYTES;

    unsigned int kbx_start = tid / KQS_PER_BLOCK;
    unsigned int kqs_int   = VDR * (tid % KQS_PER_BLOCK);

    float tmp[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) tmp[r] = 0.0f;

    for (unsigned int kbx = kbx_start; kbx < nb; kbx += BLOCKS_PER_ITER) {
        const char* x_block = input_q8_1 + (unsigned long long)kbx * (unsigned long long)Q8_1_BYTES;
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);
        const int* x_packed = (const int*)(x_block + 4) + kqs_int;
        int x0 = x_packed[0];
        int x1 = x_packed[1];

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            unsigned int target_row = r0 + (unsigned int)row;
            if (target_row >= out_dim) break;

            const char* w_block = weight_q8_aligned
                + (unsigned long long)target_row * row_bytes
                + (unsigned long long)kbx * (unsigned long long)Q8_ALIGNED_BYTES;

            unsigned short w_scale_bits = *(const unsigned short*)w_block;
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(w_block + 4) + kqs_int;
            int w0 = w_packed[0];
            int w1 = w_packed[1];

            int acc = dp4a_s32(w0, x0, 0);
            acc     = dp4a_s32(w1, x1, acc);

            tmp[row] += w_scale * x_scale * (float)acc;
        }
    }

    __shared__ float tmp_shared[NWARPS - 1][NR][NW];

    if (warp_id > 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            tmp_shared[warp_id - 1][r][lane] = tmp[r];
        }
    }
    __syncthreads();

    if (warp_id > 0) return;

    float total[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        float t = tmp[r];
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; w++) {
            t += tmp_shared[w][r][lane];
        }
        total[r] = warp_reduce_sum(t);
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        if (lane == r && (r0 + (unsigned int)r) < out_dim) {
            out[r0 + (unsigned int)r] = total[r] + residual[r0 + (unsigned int)r];
        }
    }
}
