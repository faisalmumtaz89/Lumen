// ==========================================================================
// Output-proj specialized matvec: Q8 split layout x Q8_1 input (dp4a, large NR).
//
// Sibling of `matvec_q8_split_q8_1` (NR=2) tuned for extreme-aspect-ratio
// matvecs where `out_dim` is huge relative to `in_dim`. Qwen3.5-9B's
// output_proj is the motivating case: [out=248320, in=4096].
//
// Default NR=2 grid for that shape: ceil(248320/2) = 124160 CTAs. A100 has
// 108 SMs, so each SM gets ~1149 CTAs to drain serially. Per-CTA fixed cost
// (launch dispatch, register init, shmem reduce, output write) dominates the
// 32-block dp4a body; only ~2 KB of weight traffic is amortized per CTA.
//
// This kernel raises NR (rows per CTA) so each CTA does proportionally more
// dp4a work for the same per-CTA fixed cost. Sweet spot is dictated by:
// 1. Register pressure: NR floats per thread in `sumf[]`.
// 2. SHMEM reduction footprint: `(NWARPS-1) * NR` floats.
// 3. SM occupancy: A100 has 108 SMs * 2-4 CTAs-resident slots = 216-432
// CTAs in flight. We want >=~1000 CTAs total so all SMs stay fed even
// with imperfect load balance.
//
// Grid sizes for out_dim=248320:
// NR=16 -> 15520 CTAs (oversub ~36x — plenty of work)
// NR=32 -> 7760 CTAs (oversub ~18x — fine)
// NR=64 -> 3880 CTAs (oversub ~9x — fine)
// NR=128 -> 1940 CTAs (oversub ~4.5x — still feeds all SMs)
//
// We compile four specialised kernels (NR=16/32/64/128) and let the host
// pick one via env-var `LUMEN_CUDA_OUTPUT_PROJ_NR=<value>`.
//
// Reuses the Q8 split per-row layout (same as `matvec_q8_split_q8_1`):
// Per row: [f16 scale * nb][int8 quant[32] * nb], stride 34*nb.
// Quant stream offset 2*nb is 4-byte aligned because every shipped nb is
// even (nb = in_dim / 32; for in_dim=4096 -> nb=128, divisible by 2).
//
// Reuses the Q8_1 input layout (same as the NR=2 kernel):
// nb x [f16 scale][f16 sum][int8 quants[32]], quants at offset +4.
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// Requires compute capability >= 6.1 for __dp4a() (Pascal+).
// ==========================================================================

#define NW                32   // warp size
#define THREADS_PER_BLOCK 128  // 4 warps — same as NR=2 split kernel
#define NWARPS            (THREADS_PER_BLOCK / NW)
#define Q8_BLOCK_SIZE     32
#define Q8_1_BYTES        36

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// inline-PTX dp4a wrapper. See matvec_q8_split_q8_1.cu for the
// rationale (the `__dp4a` intrinsic NVRTC-fails in this build env).
__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Warp-level reduction: sum all lanes in a warp using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ==========================================================================
// Generic kernel template: <NR rows per CTA>.
// Each thread streams over `nb` Q8 blocks (1 input block per iter, shared
// across NR rows), accumulates a partial F32 sum per row, then cross-warp
// reduces and writes the final NR outputs.
// ==========================================================================
template<int NR>
__device__ __forceinline__ void matvec_q8_split_output_proj_impl(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    unsigned int r0 = blockIdx.x * NR;

    unsigned int warp_id = threadIdx.x / NW;
    unsigned int lane    = threadIdx.x % NW;

    unsigned int nb = in_dim >> 5;  // in_dim / 32
    unsigned long long row_bytes = (unsigned long long)nb * 34ULL;
    unsigned long long scales_bytes_per_row = (unsigned long long)nb * 2ULL;

    float sumf[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) sumf[r] = 0.0f;

    // Main loop: each thread handles one Q8 block at a time. The shared input
    // x-vector block is loaded once and reused across all NR rows.
    for (unsigned int ib = threadIdx.x; ib < nb; ib += THREADS_PER_BLOCK) {

        // --- Load Q8_1 input block (shared across NR rows) ---
        const char* x_block = input_q8_1 + (unsigned long long)ib * Q8_1_BYTES;

        unsigned short x_scale_bits = (unsigned short)(unsigned char)x_block[0]
                                    | ((unsigned short)(unsigned char)x_block[1] << 8);
        float x_scale = f16_bits_to_f32(x_scale_bits);

        const int* x_packed = (const int*)(x_block + 4);
        int xv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) xv[k] = x_packed[k];

        // --- Process NR output rows ---
        // We do an explicit unroll only for small NR; for larger NR (>=32) the
        // compiler unrolls based on its own heuristics to avoid massive ICache
        // blowup at NR=128.
        for (int row = 0; row < NR; row++) {
            unsigned int target_row = r0 + (unsigned int)row;
            if (target_row >= out_dim) break;

            const char* row_base = weight_q8_split
                + (unsigned long long)target_row * row_bytes;

            // Scale stream: 2-byte halfword per block at offset `ib*2`.
            const char* scale_byte = row_base + (unsigned long long)ib * 2ULL;
            unsigned short w_scale_bits = (unsigned short)(unsigned char)scale_byte[0]
                                        | ((unsigned short)(unsigned char)scale_byte[1] << 8);
            float w_scale = f16_bits_to_f32(w_scale_bits);

            // Quant stream: 32 bytes per block at offset `2*nb + ib*32`.
            // 4-byte aligned (`row_bytes % 4 == 0` when nb is even).
            const int* w_packed = (const int*)(row_base
                + scales_bytes_per_row
                + (unsigned long long)ib * 32ULL);

            int acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                acc = dp4a_s32(w_packed[k], xv[k], acc);
            }

            sumf[row] += w_scale * x_scale * (float)acc;
        }
    }

    // --- Cross-warp reduction via simple shmem ---
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        sumf[r] = warp_reduce_sum(sumf[r]);
    }

    // shmem footprint: (NWARPS-1) * NR floats.
    // NR=16 -> 192 B
    // NR=32 -> 384 B
    // NR=64 -> 768 B
    // NR=128 -> 1536 B
    // All within the 48 KB / 100 KB shmem budget. The hard ceiling is register
    // pressure (sumf[NR]) rather than shmem.
    __shared__ float shmem[(NWARPS - 1) * NR];

    if (warp_id > 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            shmem[(warp_id - 1) * NR + r] = sumf[r];
        }
    }
    __syncthreads();

    // Final write: only warp 0 emits outputs. Each lane in warp 0 handles
    // one or more row indices `r` via a stride loop. Because `sumf[r]` is a
    // per-thread register, and only warp 0's `sumf[r]` is the right partial
    // for row r in the cross-warp sum, we restrict writes to warp 0 lanes.
    //
    // For NR <= 32: lanes 0..NR-1 each write one output, lane stride loop
    // performs at most 1 iter. For NR up to 128: each lane writes NR/32 rows
    // (stride 32). For NR=128, each lane writes 4 rows.
    if (warp_id == 0) {
        for (int r = (int)lane; r < NR; r += NW) {
            unsigned int target_row = r0 + (unsigned int)r;
            if (target_row >= out_dim) continue;
            float total = sumf[r];
            #pragma unroll
            for (int w = 0; w < NWARPS - 1; w++) {
                total += shmem[w * NR + r];
            }
            out[target_row] = total;
        }
    }
}

// ==========================================================================
// Explicit kernel instantiations: NR = 16, 32, 64, 128.
//
// Grid: (ceil(out_dim / NR), 1, 1)
// Block: (128, 1, 1) -- 4 warps x 32 threads
// ==========================================================================

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void matvec_q8_split_output_proj_nr8(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    // tighter NR than the nr16 variant. With 8 floats/thread of
    // sumfplus the per-iter x-block load (8 ints) the register footprint
    // is small enough to allow 4 resident CTAs/SM (vs 2 for nr16).
    // Grid: ceil(out_dim/8) = 31040 CTAs on Qwen3.5-9B output_proj.
    matvec_q8_split_output_proj_impl<8>(
        weight_q8_split, input_q8_1, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void matvec_q8_split_output_proj_nr16(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_output_proj_impl<16>(
        weight_q8_split, input_q8_1, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void matvec_q8_split_output_proj_nr32(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_output_proj_impl<32>(
        weight_q8_split, input_q8_1, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_output_proj_nr64(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_output_proj_impl<64>(
        weight_q8_split, input_q8_1, out, out_dim, in_dim);
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void matvec_q8_split_output_proj_nr128(
    const char* __restrict__ weight_q8_split,
    const char* __restrict__ input_q8_1,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    matvec_q8_split_output_proj_impl<128>(
        weight_q8_split, input_q8_1, out, out_dim, in_dim);
}
