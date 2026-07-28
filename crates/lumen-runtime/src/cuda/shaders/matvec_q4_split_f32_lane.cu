// ==========================================================================
// Q4 split-layout matvec, F32 acts, LANE-STRIPED inner product (F32-exact
// analogue of llama.cpp's mmvq decomposition).
//
// WHY LANES COOPERATE PER BLOCK
//
// The straightforward decomposition gives ONE THREAD A WHOLE Q4 BLOCK: 32
// activations live per thread, four packed ints read per thread, and a 16-byte
// lane stride on the weight loads, so a warp's accesses are never contiguous.
// Having several lanes cooperate on one block collapses per-lane live state and
// makes the weight loads warp-contiguous instead.
//
// Here: FOUR LANES PER Q4 BLOCK.
//
//   * Each lane reads exactly ONE packed int (4 nibble bytes = 8 weights).
//     Lanes 4g..4g+3 read four CONSECUTIVE ints, so a 32-lane warp covers 8
//     blocks as 128 CONTIGUOUS BYTES — genuine warp coalescing, which none of
//     the earlier variants achieved (they had a 16-byte lane stride).
//   * Each lane holds EIGHT activations (two float4s) instead of 32. The
//     `xv[32]` array is gone entirely; activation register footprint drops 4x,
//     which is what lifts occupancy.
//   * x is read straight from global — no shared staging, so no 32-bank-period
//     lane stride and no dynamic shared request. Within a 4-lane group the
//     float4 addresses are consecutive, so those loads coalesce too.
//   * The block's four lane-partials are combined with two __shfl_xor steps
//     (masks 1 and 2), then the group leader applies the block scale ONCE.
//     No atomics, no second kernel, and no split-K across CTAs: at these
//     grids (1024-3072 CTAs, i.e. 9-28 waves over 108 SMs) residency is not
//     the constraint, so splitting K would add reduction traffic for nothing.
//
// ONE ROW PER CTA, 128 threads. All four warps work the same row, striding the
// block axis by 32 blocks (8 blocks per warp per step). x is re-read by every
// CTA but is only 16-48 KB against a 40 MB L2, so it stays resident.
//
// Numerics: F32 activations, F32 accumulation, dequant scale*(nibble-8) — the
// same values the baseline computes, in a different summation order. The
// correctness gate checks the resulting token ids against baseline.
//
// Grid:  (out_dim, 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Shared memory: NR_WARPS floats, static.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

#define WARP_SIZE       32
#define BLOCK_DIM       128
#define NR_WARPS        (BLOCK_DIM / WARP_SIZE)   // 4
#define LANES_PER_BLK   4                          // lanes cooperating per Q4 block
#define BLKS_PER_WARP   (WARP_SIZE / LANES_PER_BLK) // 8
#define Q4_BLOCK_ELEMS  32
#define Q4_NIBBLE_BYTES 16

__device__ __forceinline__ float f16_bits_to_f32_lane(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float warp_reduce_sum_lane(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

extern "C" __global__ void matvec_q4_split_f32_lane(
    const unsigned char* __restrict__ weight_split,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane    = tid % WARP_SIZE;
    const unsigned int grp     = lane / LANES_PER_BLK;   // which block within the warp
    const unsigned int w       = lane % LANES_PER_BLK;   // which packed int within the block

    const unsigned int nb = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + (unsigned long long)Q4_NIBBLE_BYTES);
    const unsigned char* rp = weight_split + (unsigned long long)row * row_bytes;
    const unsigned short* row_scales = (const unsigned short*)rp;
    const unsigned char*  row_nibs   = rp + 2ULL * nb;

    float sumf = 0.0f;

    // Each warp step covers BLKS_PER_WARP blocks; warps stride by NR_WARPS
    // groups so the whole CTA advances BLOCK_DIM/LANES_PER_BLK = 32 blocks.
    for (unsigned int ib = warp_id * BLKS_PER_WARP + grp;
         ib < nb;
         ib += NR_WARPS * BLKS_PER_WARP) {

        // ONE int of nibbles per lane. Lanes 4g..4g+3 hit consecutive ints.
        const int packed = *(const int*)(row_nibs
            + (unsigned long long)ib * Q4_NIBBLE_BYTES + (unsigned long long)w * 4ULL);
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;

        // EIGHT activations per lane, two float4s, straight from global.
        // GGML de-interleaving: low nibbles are elements w*4..w*4+3, high
        // nibbles are elements 16+w*4..16+w*4+3.
        const float* xb = x + (unsigned long long)ib * Q4_BLOCK_ELEMS;
        const float4 xl = *(const float4*)(xb + w * 4);
        const float4 xh = *(const float4*)(xb + 16 + w * 4);

        float part = 0.0f;
        part += (float)(((lo      ) & 0xFF) - 8) * xl.x;
        part += (float)(((lo >>  8) & 0xFF) - 8) * xl.y;
        part += (float)(((lo >> 16) & 0xFF) - 8) * xl.z;
        part += (float)(((lo >> 24) & 0xFF) - 8) * xl.w;
        part += (float)(((hi      ) & 0xFF) - 8) * xh.x;
        part += (float)(((hi >>  8) & 0xFF) - 8) * xh.y;
        part += (float)(((hi >> 16) & 0xFF) - 8) * xh.z;
        part += (float)(((hi >> 24) & 0xFF) - 8) * xh.w;

        // Combine the four lanes of this block. After both steps every lane in
        // the group holds the full block sum; only the leader applies the
        // scale and accumulates, so the later warp reduction counts it once.
        part += __shfl_xor_sync(0xffffffff, part, 1);
        part += __shfl_xor_sync(0xffffffff, part, 2);
        if (w == 0) {
            sumf += f16_bits_to_f32_lane(row_scales[ib]) * part;
        }
    }

    sumf = warp_reduce_sum_lane(sumf);

    __shared__ float warp_partials[NR_WARPS];
    if (lane == 0) warp_partials[warp_id] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < NR_WARPS; i++) total += warp_partials[i];
        out[row] = total;
    }
}

// --------------------------------------------------------------------------
// RESIDUAL variant: out[row] = dot + residual[row].
//
// `wo` (the attention output projection) dispatches through
// `launch_matvec_residual`, a path the lane kernel never covered — so one
// matvec in EVERY one of the 32 layers was still running the old
// one-thread-per-block kernel. The full-attention block measured 192 GB/s
// against the FFN's 600 GB/s while carrying only 5% of the model's bytes,
// which is what pointed here.
//
// Identical decomposition and numerics to `matvec_q4_split_f32_lane`; the
// only change is folding the residual add into the single writing thread,
// which also saves a separate elementwise-add launch.
// --------------------------------------------------------------------------
extern "C" __global__ void matvec_q4_split_f32_lane_residual(
    const unsigned char* __restrict__ weight_split,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    unsigned int out_dim,
    unsigned int in_dim)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane    = tid % WARP_SIZE;
    const unsigned int grp     = lane / LANES_PER_BLK;
    const unsigned int w       = lane % LANES_PER_BLK;

    const unsigned int nb = in_dim / Q4_BLOCK_ELEMS;
    const unsigned long long row_bytes =
        (unsigned long long)nb * (2ULL + (unsigned long long)Q4_NIBBLE_BYTES);
    const unsigned char* rp = weight_split + (unsigned long long)row * row_bytes;
    const unsigned short* row_scales = (const unsigned short*)rp;
    const unsigned char*  row_nibs   = rp + 2ULL * nb;

    float sumf = 0.0f;
    for (unsigned int ib = warp_id * BLKS_PER_WARP + grp;
         ib < nb;
         ib += NR_WARPS * BLKS_PER_WARP) {

        const int packed = *(const int*)(row_nibs
            + (unsigned long long)ib * Q4_NIBBLE_BYTES + (unsigned long long)w * 4ULL);
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;

        const float* xb = x + (unsigned long long)ib * Q4_BLOCK_ELEMS;
        const float4 xl = *(const float4*)(xb + w * 4);
        const float4 xh = *(const float4*)(xb + 16 + w * 4);

        float part = 0.0f;
        part += (float)(((lo      ) & 0xFF) - 8) * xl.x;
        part += (float)(((lo >>  8) & 0xFF) - 8) * xl.y;
        part += (float)(((lo >> 16) & 0xFF) - 8) * xl.z;
        part += (float)(((lo >> 24) & 0xFF) - 8) * xl.w;
        part += (float)(((hi      ) & 0xFF) - 8) * xh.x;
        part += (float)(((hi >>  8) & 0xFF) - 8) * xh.y;
        part += (float)(((hi >> 16) & 0xFF) - 8) * xh.z;
        part += (float)(((hi >> 24) & 0xFF) - 8) * xh.w;

        part += __shfl_xor_sync(0xffffffff, part, 1);
        part += __shfl_xor_sync(0xffffffff, part, 2);
        if (w == 0) {
            sumf += f16_bits_to_f32_lane(row_scales[ib]) * part;
        }
    }

    sumf = warp_reduce_sum_lane(sumf);

    __shared__ float warp_partials_res[NR_WARPS];
    if (lane == 0) warp_partials_res[warp_id] = sumf;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < NR_WARPS; i++) total += warp_partials_res[i];
        out[row] = total + residual[row];
    }
}

// --------------------------------------------------------------------------
// FUSED gate + up + SiLU, lane-striped.
//
// The dense FFN issues THREE dispatches per layer: gate matvec, up matvec, and
// a SwiGLU elementwise pass over inter_dim. Across 32 layers that is 96
// launches per token plus a full read+write of a 12288-float buffer per layer.
// llama.cpp fuses exactly this pattern (PR #16715), and its fusion stack is
// documented as the single biggest batch-1 decode win (+27-42% on
// memory-bound models).
//
// One CTA owns one output row and computes BOTH projections for it, then
// applies the activation in-register:
//
//     out[row] = silu(gate[row]) * up[row]
//
// so the SwiGLU buffer round-trip disappears entirely and three launches
// become one. Both passes reuse the same lane decomposition as
// matvec_q4_split_f32_lane, and x is read from global where it is L2-hot
// across every CTA.
//
// Numerics: each dot product is accumulated exactly as the standalone lane
// kernel accumulates it, and silu(g)*u is the same expression the standalone
// SwiGLU kernel evaluates in F32. So the only difference from the unfused path
// is that the intermediate never round-trips through device memory.
//
// Grid:  (inter_dim, 1, 1)
// Block: (128, 1, 1)

// --------------------------------------------------------------------------
// MULTI-ROW lane kernel: ROWS_PER_CTA output rows share one pass over x.
//
// WHY: the single-row lane kernel re-reads the WHOLE activation vector in
// every CTA. For an FFN gate projection that is 12288 CTAs x 16 KB = 196 MB of
// L2 traffic for ONE matvec, ~590 MB per layer, ~19 GB per token across 32
// layers. At A100's ~4 TB/s L2 that is ~4.7 ms — the same order as the entire
// measured FFN time (4.50 ms at an implied 604 GB/s of HBM). So the FFN may not
// be HBM-limited at all; it may be limited by L2 bandwidth spent re-reading x.
//
// Holding R rows per CTA divides that traffic by R: the two float4s a lane
// loads for a Q4 block feed R independent dot products instead of one. Weight
// traffic is unchanged (each row's nibbles are still read exactly once), so
// this is a pure reduction in redundant activation reads.
//
// Cost is R accumulators and R weight pointers in registers, which is cheap
// because the lane decomposition already dropped per-lane activation state
// from 32 floats to 8.
//
// Numerics: each row's dot product accumulates in exactly the same order as
// the single-row kernel, so results are bit-identical to variant 4.
//
// Grid:  (ceil(out_dim / ROWS_PER_CTA), 1, 1)
// Block: (128, 1, 1)


// --------------------------------------------------------------------------
// WIDE-LANE variant: TWO lanes per Q4 block, int2 weight loads.
//
// One row per CTA. Widening to multiple rows per CTA, or to fewer/wider loads,
// both measured slower here: the grid must stay large enough to keep memory-level
// parallelism up, which is the binding resource at batch 1.

// This keeps ONE row per CTA (grid unchanged, parallelism unchanged) and
// instead widens each load. With two lanes per block each lane reads an int2 —
// 8 nibble bytes = 16 weights — so a 32-lane warp covers 16 blocks as 256
// CONTIGUOUS bytes per instruction instead of 128, halving the number of load
// instructions for the same bytes and doubling bytes-in-flight per warp.
//
// Per-lane activation state rises from 8 floats to 16 (four float4s), still
// far below the 32 the pre-lane kernels held.
//
// Numerics: same dequant, same F32 accumulation, and the intra-block reduction
// is one __shfl_xor step instead of two because only two lanes cooperate.
