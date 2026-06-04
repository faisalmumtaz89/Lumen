// ==========================================================================
// Q8 mmvq variant: SPLIT (SoA, 34B) layout, NR=8 rows per CTA. Sibling to
// matvec_q8_split_q8_1.cu (NR=2). NR=8 trades shmem occupancy for K-loop
// trip count.
//
// The sibling 4-thread kernel (`matvec_q8_split_q8_1_4thread.cu`) keeps
// NR=2 (rows per CTA). This kernel uses NR=8 -- the single parameter that
// the 4-thread variant left untested.
//
// The motivation: at FFN shapes (12288x4096 and 4096x12288), the 4-thread
// kernel regressed vs production (0.90-0.93x). One diagnosis was that "at
// HBM saturation, lower occupancy beats higher occupancy". However that
// interpretation was conditioned on NR=2 -- at NR=8 each CTA amortizes
// one x-vector load across 8 row dp4a chains, dropping the per-row byte
// rate by 4x vs NR=2 while preserving 5 CTAs/SM x 4 warps = 20 warps/SM
// (vs Lumen's 8 warps/SM). If NR=8 wins at FFN shapes the kernel-ceiling
// claim is refuted.
//
// Notes on parameter choice:
// * The standard mmvq Q-quant kernel for ncols_dst=1 uses NR=1
// (`calc_rows_per_block(1)` = 1 in the generic path); it falls back to
// NR=`nwarps=4` only in the small-k branch. NR=8 is neither default;
// it is an extrapolation chosen on the rationale that the SPLIT layout's
// per-CTA x-load amortization grows linearly in NR while register pressure
// remains bounded (8 floats / thread of `tmp[NR]` vs the 46 reg/thread
// Lumen prod baseline + 2 sumffloats -- leaves register room for the
// extra 6).
//
// Numerical contract: identical to `matvec_q8_split_q8_1` (NR=2) and
// `matvec_q8_split_q8_1_4thread` (NR=2). The differential test in
// `crates/lumen-runtime/tests/cuda_q8_split_nr8_test.rs` verifies bit
// agreement within the same 1e-3 absolute tolerance used by the
// existing split kernels.
//
// Layout consumed (UNCHANGED, same SPLIT bytes as both NR=2 kernels):
// Row stride: 34 * nb
// Per row: [f16 scale * nb][int8 quants[32] * nb]
// scales: row_base
// quants: row_base + 2*nb
//
// Constants (dp4a-mmvq + NR=8):
// QK = 32 elements / Q8_0 block
// QI = QK / 4 = 8 int32s / Q8_0 block
// VDR = 2 int32s consumed / dp4a inner call
// NWARPS = 4 4 warps / CTA
// WARP_SIZE = 32
// THREADS = NWARPS * 32 = 128
// NR = 8 rows / CTA (the UNTRIED parameter)
// KQS_PER_BLOCK = QI / VDR = 4 threads cooperating on a Q8 block
// BLOCKS_PER_ITER = VDR * THREADS / QI = 32 blocks advanced per K-iter
//
// Thread mapping (4-threads-per-block pattern):
// tid = warp_id * 32 + lane
// kbx_start = tid / 4 0..31, 4 threads share a block
// kqs = vdr * (tid % 4) = 2 * (tid%4) ∈ {0,2,4,6} int32 offset
// for (kbx = kbx_start; kbx < nb; kbx += 32) {
// for (row = 0; row < 8; row++) {
// load 2 int32 quants of weight row `row` block kbx at kqs
// load 2 int32 quants of x at kqs (SHARED across 8 rows)
// load f16 scales (w_scale, x_scale)
// tmp[row] += w_scale * x_scale * dp4a_chain(w_int[0..1], x_int[0..1])
// }
// }
//
// Cross-warp reduction:
// __shared__ float tmp_shared[3][8][32] 768 floats = 3 KB
// warps 1..3 store their per-row partials indexed by (warp-1, row, lane)
// warp 0 reads them back, sums lane-wise into tmp[row], warp_reduce_sum
// lanes 0..7 (NR=8) of warp 0 each write one final output
//
// K-loop trip count:
// in_dim=4096 -> nb=128 -> trip=4 (FFN gate/up — separated, attn_q half)
// in_dim=12288 -> nb=384 -> trip=12 (FFN down)
// in_dim=4096 -> nb=128 -> trip=4 (12288×4096 FFN gate/up too — same trip)
//
// All ≥ 4 — enough for the ptxas compiler to interleave 4+ K-iters into one
// body, hiding LDG latency (this is the WHOLE point of the K-trip≥4
// structural change; Lumen production has K-trip=1 at the 4096 hidden dim).
//
// Two kernels (mirroring the existing pair):
// 1. matvec_q8_split_q8_1_nr8: W*x -> out
// 2. matvec_q8_split_q8_1_nr8_residual: W*x + residual -> out
//
// Env gate: `LUMEN_CUDA_Q8_SPLIT_NR8=1` selects these in place of the
// production split kernels at decode dispatch. Default OFF (default-off contract).
//
// Requires compute capability >= 6.1 for dp4a_s32().
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR              8     // rows per thread block  ★ THE UNTRIED PARAMETER ★
#define NW              32    // warp size
#define NWARPS          4     // 4 warps per CTA (calc_nwarps for ncols=1)
#define THREADS_PER_BLOCK (NWARPS * NW)   // 128

#define QK              32    // elements per Q8_0 block
#define QI              8     // int32s per Q8_0 block (= QK / 4)
#define VDR             2     // int32s consumed per dp4a inner call (VDR_Q8_0_Q8_1_MMVQ)
#define KQS_PER_BLOCK   (QI / VDR)        // 4 (= threads cooperating on one Q8_0 block)
#define BLOCKS_PER_ITER ((VDR * THREADS_PER_BLOCK) / QI)   // 32

#define Q8_1_BYTES      36    // 2B f16 scale + 2B f16 sum + 32B int8 data

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
// Kernel 1: NR=8 Q8 SPLIT x Q8_1 -> F32 (no residual).
//
// Grid: (ceil(out_dim / NR), 1, 1)
// Block: (THREADS_PER_BLOCK, 1, 1) -- 128 threads, 4 warps
//
// __launch_bounds__(128, 1): minimum CTAs/SM = 1, no maximum. ptxas
// computes the register-bounded
// resident occupancy on this constraint. With 8 sumffloats + 2 int x[]
// + 2 int w+ scratch, register pressure is expected ~36-42 (vs Lumen
// prod 46). At ≤40 reg/thread the A100 SM (65536 regs / 128 threads-per-
// CTA = 512 reg/thread × 128 = 65536 cap) admits 5 CTAs/SM = 20 warps/SM.
//
// At the FFN gate/up shape (12288×4096), per-CTA byte traffic:
// per K-iter: 8 (x quant) + 2 (x scale) + 8 rows × (8 w quant + 2 w scale) = 90 bytes
// total: trip 4 × 90 = 360 bytes/CTA-K-iter, amortized across 32 dp4a inst
//
// Compare to NR=2 4-thread variant at same shape: 8 + 2 + 2 rows × 10 = 30 B/iter ×
// trip 4 = 120 B/CTA. The NR=8 kernel reads 3× more bytes per CTA but
// drives 4× more SM-resident CTAs at 1/3 the grid size — net HBM byte
// rate should be balanced and the per-row x-load amortization is the win.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_q8_1_nr8(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34]
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
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    // 4-threads-per-block: 4 threads cooperate per Q8_0 block, each handles vdr=2 int32s.
    unsigned int kbx_start = tid / KQS_PER_BLOCK;        // 0..31 — initial K-block id
    unsigned int kqs_int   = VDR * (tid % KQS_PER_BLOCK); // 0, 2, 4, 6 — int32 offset

    // Per-row accumulators in registers. NR=8 floats per thread.
    float tmp[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) tmp[r] = 0.0f;

    // Main K-loop. blocks_per_iter=32 → trip ≥ 4 at all FFN shapes.
    for (unsigned int kbx = kbx_start; kbx < nb; kbx += BLOCKS_PER_ITER) {

        // --- Load Q8_1 input block scale and 2 int32 quants (SHARED across all NR rows) ---
        // Q8_1 block layout: bytes 0-1 = f16 scale, 2-3 = f16 sum (unused),
        // 4-35 = int8 quants. Native int* load at byte offset (4 + kqs_int*4).
        const char* x_block = input_q8_1 + (unsigned long long)kbx * Q8_1_BYTES;
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

            const char* row_base = weight_q8_split
                + (unsigned long long)target_row * row_bytes;

            // f16 weight scale at offset kbx*2 in the scales stream.
            unsigned short w_scale_bits =
                *(const unsigned short*)(row_base + (unsigned long long)kbx * 2ULL);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // 2 int32s of weight quants. (scales_bytes_per_row + kbx*32 + kqs_int*4).
            // Always 4-byte aligned (nb even => row_bytes%4==0, scales_bytes_per_row%4==0;
            // kbx*32 is 4-byte-aligned).
            const int* w_packed = (const int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)kbx * 32ULL) + kqs_int;
            int w0 = w_packed[0];
            int w1 = w_packed[1];

            // dp4a chain: 2 int32 pairs = 8 INT8 multiply-accumulates.
            int acc = dp4a_s32(w0, x0, 0);
            acc     = dp4a_s32(w1, x1, acc);

            tmp[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction (shmem reduction) ---
    // tmp_shared[nwarps-1][NR][warp_size]
    // = [3][8][32] floats = 768 floats = 3 KB per CTA
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
// Kernel 2: NR=8 Q8 SPLIT x Q8_1 + residual -> F32.
//
// Identical to kernel 1 with `out[r] = total + residual[r]` at the final write.
// Used for Wo (attention output projection) and FFN down projection.
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_q8_1_nr8_residual(
    const char* __restrict__ weight_q8_split,    // [out_dim * nb * 34]
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
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    unsigned int kbx_start = tid / KQS_PER_BLOCK;
    unsigned int kqs_int   = VDR * (tid % KQS_PER_BLOCK);

    float tmp[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) tmp[r] = 0.0f;

    for (unsigned int kbx = kbx_start; kbx < nb; kbx += BLOCKS_PER_ITER) {
        const char* x_block = input_q8_1 + (unsigned long long)kbx * Q8_1_BYTES;
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);
        const int* x_packed = (const int*)(x_block + 4) + kqs_int;
        int x0 = x_packed[0];
        int x1 = x_packed[1];

        #pragma unroll
        for (int row = 0; row < NR; row++) {
            unsigned int target_row = r0 + (unsigned int)row;
            if (target_row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)target_row * row_bytes;

            unsigned short w_scale_bits =
                *(const unsigned short*)(row_base + (unsigned long long)kbx * 2ULL);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            const int* w_packed = (const int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)kbx * 32ULL) + kqs_int;
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
