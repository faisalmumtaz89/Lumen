// ==========================================================================
// Q8 mmvq variant: SPLIT layout, 4-threads-per-block + NR=2 dp4a-mmvq.
//
// 4-threads-per-block thread→block mapping on the Lumen SPLIT layout. The
// sibling `matvec_q8_split_q8_1.cu` consumes the same SoA per-row layout
// (34-byte rows) with a Lumen-native mapping where each thread owns one
// full Q8_0 block per K-iteration (vdr=8 int32s, K-loop trip = 1 at
// in_dim=4096). PTX/SASS diff identified this as the structural gap: with
// vdr=2 and `blocks_per_iter=32`, K-loop trip = 4 because 4 threads
// cooperate on each block (kqs ∈ {0,2,4,6}) and the warp reduction
// collapses the kbx/kqs distributions to per-row sums.
//
// This kernel uses that 4-thread-per-block pattern on the SPLIT layout:
//
// Layout (UNCHANGED from matvec_q8_split_q8_1.cu):
// Row stride: 34 * nb bytes
// Per row: [f16 scale * nb][int8 quants[32] * nb]
// row_scales = base
// row_quants = base + 2*nb
//
// Constants (dp4a-mmvq):
// qk = 32 (elements per Q8_0 block)
// qi = qk / sizeof(int) = 8 (int32s per Q8_0 block)
// vdr = 2 (int32s consumed per dp4a inner call)
// nwarps = 4 (4 warps per CTA, Ampere mmvq default)
// warp_size = 32
// threads_per_block = 128
// rows_per_block = NR = 2 (calc_rows_per_block for ncols=1)
// blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 4 * 32 / 8 = 32
//
// Thread mapping (4-threads-per-block):
// tid = warp_id * 32 + lane (0..127)
// kbx_start = tid / (qi / vdr) = tid / 4 (0..31, 4 threads share a block)
// kqs_int = vdr * (tid % (qi / vdr)) = 2 * (tid % 4) (∈ {0, 2, 4, 6})
// for (kbx = kbx_start; kbx < nb; kbx += 32) {
// for (i in 0..NR) {
// load 2 int32 quants of weight row i block kbx at kqs_int
// load 2 int32 quants of input block kbx at kqs_int (same 8 bytes
// for all NR rows)
// load row[i] f16 scale at block kbx (once per kbx; could hoist)
// load input f16 scale at block kbx
// tmp[i] += w_scale * x_scale * dp4a_chain(w_int[0..1], x_int[0..1])
// }
// }
//
// Cross-warp reduction:
// __shared__ float tmp_shared[nwarps-1][NR][warp_size]
// warps 1..nwarps-1 write their per-row partial sums (indexed by threadIdx.x)
// warp 0 reads them back, accumulates, then warp_reduce_sum
// warp 0 lane < NR writes dst[r0 + lane]
//
// K-loop trip count at FFN hidden=4096: nb=128, blocks_per_iter=32, trip=4.
// At FFN inter_dim=12288: nb=384, trip=12. Both are ≥4 — enough for the NVCC
// compiler to interleave 4+ K-iters into a single body, hiding LDG latency.
//
// Two kernels (one without residual, one with):
// 1. matvec_q8_split_q8_1_4thread: W*x -> out
// 2. matvec_q8_split_q8_1_4thread_residual: W*x + residual -> out
//
// Both consume EXACTLY THE SAME byte layout as matvec_q8_split_q8_1{,_residual}.
// They are env-gated drop-in replacements: when `LUMEN_CUDA_Q8_SPLIT_4THREAD=1`,
// the kernel function pointer in KernelSet is swapped to these
// 4-thread-per-block variants. The grid/block dispatch math is identical.
//
// Requires compute capability >= 6.1 for dp4a_s32().
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

#define NR              2     // rows per thread block (rows_per_cuda_block)
#define NW              32    // warp size
#define NWARPS          4     // 4 warps per CTA (calc_nwarps for ncols=1)
#define THREADS_PER_BLOCK (NWARPS * NW)   // 128

#define QK              32    // elements per Q8_0 block
#define QI              8     // int32s per Q8_0 block (= QK / 4)
#define VDR             2     // int32s consumed per dp4a inner call (VDR_Q8_0_Q8_1_MMVQ)
#define KQS_PER_BLOCK   (QI / VDR)        // 4 (= threads cooperating on one Q8_0 block)
#define BLOCKS_PER_ITER ((VDR * THREADS_PER_BLOCK) / QI)   // 32

#define Q8_1_BYTES      36    // 2B f16 scale + 2B f16 sum + 32B int8 data
#define Q8_SPLIT_QUANT_BYTES 32   // int8 quants per block in SPLIT layout
#define Q8_SPLIT_SCALE_BYTES 2    // f16 scale per block in SPLIT layout

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
// Kernel 1: 4-thread-per-block Q8 SPLIT x Q8_1 -> F32 (NR=2, no residual).
//
// Grid: (ceil(out_dim / NR), 1, 1) -- same as matvec_q8_split_q8_1
// Block: (THREADS_PER_BLOCK, 1, 1) -- 128 threads, 4 warps
//
// launch_bounds(128, 1) -- 1 minimum CTA per SM, no maximum.
// (Uses (nwarps*warp_size, 1). The Lumen prod kernel uses (128, 2) which
// hard-caps occupancy at 2 CTAs/SM = 12.5% (root cause). an earlier variant dropped
// launch_bounds and got -2.4%; but an earlier variant kept the old vdr=8 K-trip=1
// mapping. With the K-trip=4 mapping the higher occupancy should not
// collide with HBM contention because per-iter byte throughput is 1/4
// as much per thread.)
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_q8_1_4thread(
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

    // 4-threads-per-block: 4 threads cooperate per Q8_0 block, each handling vdr=2 int32s.
    unsigned int kbx_start = tid / KQS_PER_BLOCK;        // 0..31 — initial K-block id
    unsigned int kqs_int   = VDR * (tid % KQS_PER_BLOCK); // 0, 2, 4, 6 — int32 offset within block

    // Per-row accumulators in registers.
    float tmp[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) tmp[r] = 0.0f;

    // Main K-loop: each thread processes one (kbx, kqs) tuple per iteration.
    // blocks_per_iter = 32 → at nb=128 (in_dim=4096), trip = 4. NVCC can
    // interleave 4 K-iters of LDG into one body to hide latency (target).
    for (unsigned int kbx = kbx_start; kbx < nb; kbx += BLOCKS_PER_ITER) {

        // --- Load input block scale (f16) and 2 int32s of input quants ---
        // Q8_1 block layout: bytes 0-1 = f16 scale, 2-3 = f16 sum (unused),
        // 4-35 = int8 quants. Native int* load at byte offset (4 + kqs_int*4).
        const char* x_block = input_q8_1 + (unsigned long long)kbx * Q8_1_BYTES;
        unsigned short x_scale_bits = *(const unsigned short*)x_block;
        float x_scale = f16_bits_to_f32(x_scale_bits);
        const int* x_packed = (const int*)(x_block + 4) + kqs_int;
        int x0 = x_packed[0];
        int x1 = x_packed[1];

        // --- Process NR output rows for this (kbx, kqs) ---
        #pragma unroll
        for (int row = 0; row < NR; row++) {
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

            // f16 weight scale at offset kbx*2 in the scales stream.
            unsigned short w_scale_bits =
                *(const unsigned short*)(row_base + (unsigned long long)kbx * 2ULL);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // 2 int32s of weight quants at offset (scales_bytes_per_row + kbx*32 + kqs_int*4)
            // in the quants stream. Always 4-byte aligned (nb even => row_bytes%4==0,
            // scales_bytes_per_row%4==0; kbx*32 is 4-byte-aligned).
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
    // Each thread holds its own partial sum; warps 1..3 store to shmem indexed
    // by (warp_id-1, row, lane); warp 0 reads them back, sums lane-wise into its
    // tmp[r], then warp_reduce_sum collapses across lanes.
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
    // Lane `r` (r < NR) writes dst[r0 + r] for its corresponding row.
    //
    // We split the per-row reduction into static per-row scalars to avoid any
    // register-indirect array access (`tmp[lane]` where `lane` is runtime)
    // which would force a local-memory spill on some NVCC versions.
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
        if (lane == r && (r0 + r) < out_dim) {
            out[r0 + r] = total[r];
        }
    }
}

// ==========================================================================
// Kernel 2: 4-thread-per-block Q8 SPLIT x Q8_1 + residual -> F32 (NR=2).
//
// Identical to kernel 1 with `out[r] = total + residual[r]` at the final write.
// Used for Wo (attention output projection) and down projection (FFN).
// ==========================================================================
extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_q8_1_4thread_residual(
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
            if (r0 + row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)(r0 + row) * row_bytes;

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
        if (lane == r && (r0 + r) < out_dim) {
            out[r0 + r] = total[r] + residual[r0 + r];
        }
    }
}
