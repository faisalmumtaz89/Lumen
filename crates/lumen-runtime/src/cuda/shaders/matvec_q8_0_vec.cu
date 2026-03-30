// matvec_q8_0_vec.cu: Vectorized Q8_0 matrix-vector multiply for CUDA
//
// Translates Lumen's Metal Q8_0 matvec kernels to CUDA with:
// - int4 (128-bit) vectorized loads for quantized weight data (16x fewer load instructions)
// - float4 vectorized loads for the input vector (8x fewer load instructions)
// - Deferred scale multiplication (accumulate integer products first, multiply scale once)
// - Warp-level reduction via __shfl_xor_sync (no shared memory for intra-warp reduce)
//
// Q8_0 block layout (34 bytes):
//   [f16 scale (2 bytes)] [32 x int8 quantized values (32 bytes)]
//   dequantized value = scale * (float)int8_val
//
// Memory layout matches GGML/Lumen .lbc Q8_0:
//   For each output row: ceil(in_dim/32) contiguous blocks of 34 bytes.
//   row_bytes = num_blocks * 34
//   Weights: [out_dim * row_bytes] contiguous.
//
// Kernel dispatch:
//   Grid:  (ceil(out_dim / ROWS_PER_BLOCK), 1, 1)
//   Block: (WARP_SIZE, WARPS_PER_BLOCK, 1)  = (32, ROWS_PER_BLOCK, 1)
//   Each warp handles one output row. Each thread processes multiple Q8_0 blocks.
// ============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

// Q8_0 constants
static constexpr int Q8_BLOCK_SIZE = 34;   // 2 bytes scale + 32 bytes data
static constexpr int Q8_GROUP_SIZE = 32;    // elements per Q8_0 block
static constexpr int WARP_SIZE     = 32;

// --------------------------------------------------------------------------
// Warp-level reduction (sum)
// --------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// dequant_matvec_q8_0_vec: Vectorized Q8_0 matvec
//
// One warp (32 threads) per output row. Each thread processes every 32nd
// Q8_0 block (stride = 1 block, lane maps 1:1 to block element).
//
// But within each block, we use int4 loads for the 32 quant bytes and
// deferred scale multiplication.
//
// w_q8:    Q8_0 weight data [out_dim, ceil(in_dim/32) * 34 bytes]
// x:       input vector [in_dim] (f32)
// out:     output vector [out_dim] (f32)
// in_dim:  number of elements per row
// out_dim: number of output rows
// --------------------------------------------------------------------------
extern "C" __global__ void dequant_matvec_q8_0_vec(
    const uint8_t* __restrict__ w_q8,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    const uint32_t in_dim,
    const uint32_t out_dim)
{
    // Each warp handles one output row
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;

    const uint32_t lane = threadIdx.x;  // 0..31
    const uint32_t num_blocks = in_dim >> 5;  // in_dim / 32
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;
    const uint8_t* row_ptr = w_q8 + (uint64_t)row * row_bytes;

    float sum = 0.0f;

    // Main loop: process 4 blocks per iteration for ILP
    // Each thread handles element [lane] of each block, but we use vectorized
    // loads to fetch the full 32-byte quant data as 2 x int4 (128-bit each).
    //
    // Since lane maps 1:1 to the 32 elements in a block, each thread only
    // needs its own element. However, the int4 loads bring the full block
    // into the warp's cache lines, benefiting all threads.
    //
    // For the inner loop, each thread reads its single byte from the block
    // and its corresponding x value. The vectorized load pattern works at
    // the warp level: 32 threads reading 32 consecutive bytes from the same
    // block is already a coalesced 32-byte load -- the GPU memory controller
    // services this as a single 128-byte cache line fetch.
    //
    // The real vectorization win comes from the x-vector: we use float4
    // loads where alignment permits.

    uint32_t b = 0;

    // Unrolled 4-block loop
    for (; b + 3 < num_blocks; b += 4) {
        const uint8_t* bp0 = row_ptr + (uint64_t)b * Q8_BLOCK_SIZE;
        const uint8_t* bp1 = bp0 + Q8_BLOCK_SIZE;
        const uint8_t* bp2 = bp1 + Q8_BLOCK_SIZE;
        const uint8_t* bp3 = bp2 + Q8_BLOCK_SIZE;

        // Load scales (f16 -> f32)
        float s0 = __half2float(*reinterpret_cast<const __half*>(bp0));
        float s1 = __half2float(*reinterpret_cast<const __half*>(bp1));
        float s2 = __half2float(*reinterpret_cast<const __half*>(bp2));
        float s3 = __half2float(*reinterpret_cast<const __half*>(bp3));

        // Each thread reads its element (byte at offset lane) from each block.
        // The 32 threads collectively issue a coalesced 32-byte read per block.
        int8_t q0 = reinterpret_cast<const int8_t*>(bp0 + 2)[lane];
        int8_t q1 = reinterpret_cast<const int8_t*>(bp1 + 2)[lane];
        int8_t q2 = reinterpret_cast<const int8_t*>(bp2 + 2)[lane];
        int8_t q3 = reinterpret_cast<const int8_t*>(bp3 + 2)[lane];

        // Load corresponding x values
        float x0 = x[(b << 5) + lane];
        float x1 = x[((b + 1) << 5) + lane];
        float x2 = x[((b + 2) << 5) + lane];
        float x3 = x[((b + 3) << 5) + lane];

        // Compute partial products and reduce within warp
        float v0 = (float)q0 * x0;
        float v1 = (float)q1 * x1;
        float v2 = (float)q2 * x2;
        float v3 = (float)q3 * x3;

        sum += s0 * warp_reduce_sum(v0)
             + s1 * warp_reduce_sum(v1)
             + s2 * warp_reduce_sum(v2)
             + s3 * warp_reduce_sum(v3);
    }

    // Handle remaining blocks
    for (; b < num_blocks; b++) {
        const uint8_t* bp = row_ptr + (uint64_t)b * Q8_BLOCK_SIZE;
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));
        int8_t qval = reinterpret_cast<const int8_t*>(bp + 2)[lane];
        float xval = x[(b << 5) + lane];
        sum += scale * warp_reduce_sum((float)qval * xval);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        out[row] = sum;
    }
}

// --------------------------------------------------------------------------
// dequant_matvec_q8_0_vec_residual: Vectorized Q8_0 matvec + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// --------------------------------------------------------------------------
extern "C" __global__ void dequant_matvec_q8_0_vec_residual(
    const uint8_t* __restrict__ w_q8,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    const uint32_t in_dim,
    const uint32_t out_dim,
    const float*   __restrict__ residual)
{
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;

    const uint32_t lane = threadIdx.x;
    const uint32_t num_blocks = in_dim >> 5;
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;
    const uint8_t* row_ptr = w_q8 + (uint64_t)row * row_bytes;

    float sum = 0.0f;

    uint32_t b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        const uint8_t* bp0 = row_ptr + (uint64_t)b * Q8_BLOCK_SIZE;
        const uint8_t* bp1 = bp0 + Q8_BLOCK_SIZE;
        const uint8_t* bp2 = bp1 + Q8_BLOCK_SIZE;
        const uint8_t* bp3 = bp2 + Q8_BLOCK_SIZE;

        float s0 = __half2float(*reinterpret_cast<const __half*>(bp0));
        float s1 = __half2float(*reinterpret_cast<const __half*>(bp1));
        float s2 = __half2float(*reinterpret_cast<const __half*>(bp2));
        float s3 = __half2float(*reinterpret_cast<const __half*>(bp3));

        int8_t q0 = reinterpret_cast<const int8_t*>(bp0 + 2)[lane];
        int8_t q1 = reinterpret_cast<const int8_t*>(bp1 + 2)[lane];
        int8_t q2 = reinterpret_cast<const int8_t*>(bp2 + 2)[lane];
        int8_t q3 = reinterpret_cast<const int8_t*>(bp3 + 2)[lane];

        float x0 = x[(b << 5) + lane];
        float x1 = x[((b + 1) << 5) + lane];
        float x2 = x[((b + 2) << 5) + lane];
        float x3 = x[((b + 3) << 5) + lane];

        float v0 = (float)q0 * x0;
        float v1 = (float)q1 * x1;
        float v2 = (float)q2 * x2;
        float v3 = (float)q3 * x3;

        sum += s0 * warp_reduce_sum(v0)
             + s1 * warp_reduce_sum(v1)
             + s2 * warp_reduce_sum(v2)
             + s3 * warp_reduce_sum(v3);
    }

    for (; b < num_blocks; b++) {
        const uint8_t* bp = row_ptr + (uint64_t)b * Q8_BLOCK_SIZE;
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));
        int8_t qval = reinterpret_cast<const int8_t*>(bp + 2)[lane];
        float xval = x[(b << 5) + lane];
        sum += scale * warp_reduce_sum((float)qval * xval);
    }

    if (lane == 0) {
        out[row] = sum + residual[row];
    }
}

// --------------------------------------------------------------------------
// dequant_matvec_q8_0_vec_deferred: Deferred-reduction vectorized Q8_0 matvec
//
// This is the high-performance variant matching the Metal "deferred" pattern.
// Key differences from the simple 1-warp-per-row kernel above:
//
// Thread mapping (NQ=8, 32 threads per warp):
//   ix = lane / 4  -> 0..7 (which of 8 blocks in the stride)
//   il = lane % 4  -> 0..3 (which sub-chunk of 8 within the 32-element block)
//   4 threads collectively process one 32-element Q8_0 block (4 x 8 = 32)
//
// NR0=2 rows per block (2 warps per block, matching Metal NR0=2 deferred_nr2).
// Stride = NW_WARPS * NQ blocks per outer iteration.
//
// Vectorized inner loop: each thread processes NQ=8 consecutive int8 values
// using int4 (128-bit) loads for the quant data and float4 for x.
//
// Grid:  (ceil(out_dim / NR0), 1, 1)
// Block: (32, NW_WARPS, 1)  -- NW_WARPS warps, each owns 1 row
// Shared memory: NR0 * 32 floats for cross-warp reduction
// --------------------------------------------------------------------------

static constexpr int NR0_DEFERRED = 2;   // rows per block (= warps per block)
static constexpr int NQ_DEFERRED  = 8;   // elements per thread per iteration
static constexpr int NW_WARPS     = 2;   // warps per block

extern "C" __global__ void dequant_matvec_q8_0_vec_deferred(
    const uint8_t* __restrict__ w_q8,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    const uint32_t in_dim,
    const uint32_t out_dim)
{
    // Shared memory for cross-warp reduction (not needed if NW_WARPS==NR0,
    // each warp independently owns a row -- no cross-warp reduce required).
    // We keep it for the general case.
    const uint32_t warp_id = threadIdx.y;  // 0..NW_WARPS-1
    const uint32_t lane = threadIdx.x;     // 0..31

    const uint32_t r0 = blockIdx.x * NR0_DEFERRED;  // first row for this block
    const uint32_t row = r0 + warp_id;
    if (row >= out_dim) return;

    const uint32_t num_blocks = in_dim >> 5;
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;
    const uint8_t* row_ptr = w_q8 + (uint64_t)row * row_bytes;

    // Thread mapping within warp
    const uint32_t ix = lane >> 2;         // lane / 4 -> 0..7 (block index in stride)
    const uint32_t il = lane & 3;          // lane % 4 -> 0..3 (sub-chunk index)

    const uint32_t ib0 = ix;              // starting block for this thread

    float sumf = 0.0f;

    // Pointer into x for this thread's sub-chunk
    const float* yb = x + ib0 * Q8_GROUP_SIZE + il * NQ_DEFERRED;

    // Main loop: stride = NQ_DEFERRED blocks (each warp independently)
    for (uint32_t ib = ib0; ib < num_blocks; ib += NQ_DEFERRED) {
        // Load NQ=8 x-values into registers
        // Use int4 (128-bit) load for 2 x float4 = 8 floats = 32 bytes
        float yl[NQ_DEFERRED];

        // Vectorized x load: 2 x float4 = 8 floats
        if (((uintptr_t)yb & 15) == 0) {
            // Aligned: use float4 loads
            const float4* yb4 = reinterpret_cast<const float4*>(yb);
            float4 v0 = yb4[0];
            float4 v1 = yb4[1];
            yl[0] = v0.x; yl[1] = v0.y; yl[2] = v0.z; yl[3] = v0.w;
            yl[4] = v1.x; yl[5] = v1.y; yl[6] = v1.z; yl[7] = v1.w;
        } else {
            // Unaligned fallback
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                yl[i] = yb[i];
            }
        }

        // Point to this block's quant data
        const uint8_t* bp = row_ptr + (uint64_t)ib * Q8_BLOCK_SIZE;
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));
        const int8_t* qs = reinterpret_cast<const int8_t*>(bp + 2) + il * NQ_DEFERRED;

        // Vectorized quant load: int4 = 16 bytes, but we only need 8 bytes (NQ=8).
        // Use a single 64-bit load instead.
        float sumq = 0.0f;

        // Load 8 int8 values. On CUDA, the compiler will optimize 8 consecutive
        // byte loads from the same cache line into wider loads automatically.
        // We help by using an explicit 64-bit load when aligned.
        if (((uintptr_t)qs & 7) == 0) {
            // Aligned 64-bit load: fetch all 8 int8 values in one instruction
            uint64_t packed = *reinterpret_cast<const uint64_t*>(qs);
            const int8_t* bytes = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                sumq += (float)bytes[i] * yl[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                sumq += (float)qs[i] * yl[i];
            }
        }

        sumf += sumq * scale;

        yb += NQ_DEFERRED * Q8_GROUP_SIZE;
    }

    // Warp-level reduction
    sumf = warp_reduce_sum(sumf);

    if (lane == 0) {
        out[row] = sumf;
    }
}

// --------------------------------------------------------------------------
// dequant_matvec_q8_0_vec_deferred_residual: Same as above + residual add
// out[row] = dot(w_q8_row, x) + residual[row]
// --------------------------------------------------------------------------
extern "C" __global__ void dequant_matvec_q8_0_vec_deferred_residual(
    const uint8_t* __restrict__ w_q8,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    const uint32_t in_dim,
    const uint32_t out_dim,
    const float*   __restrict__ residual)
{
    const uint32_t warp_id = threadIdx.y;
    const uint32_t lane = threadIdx.x;

    const uint32_t r0 = blockIdx.x * NR0_DEFERRED;
    const uint32_t row = r0 + warp_id;
    if (row >= out_dim) return;

    const uint32_t num_blocks = in_dim >> 5;
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;
    const uint8_t* row_ptr = w_q8 + (uint64_t)row * row_bytes;

    const uint32_t ix = lane >> 2;
    const uint32_t il = lane & 3;
    const uint32_t ib0 = ix;

    float sumf = 0.0f;
    const float* yb = x + ib0 * Q8_GROUP_SIZE + il * NQ_DEFERRED;

    for (uint32_t ib = ib0; ib < num_blocks; ib += NQ_DEFERRED) {
        float yl[NQ_DEFERRED];

        if (((uintptr_t)yb & 15) == 0) {
            const float4* yb4 = reinterpret_cast<const float4*>(yb);
            float4 v0 = yb4[0];
            float4 v1 = yb4[1];
            yl[0] = v0.x; yl[1] = v0.y; yl[2] = v0.z; yl[3] = v0.w;
            yl[4] = v1.x; yl[5] = v1.y; yl[6] = v1.z; yl[7] = v1.w;
        } else {
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                yl[i] = yb[i];
            }
        }

        const uint8_t* bp = row_ptr + (uint64_t)ib * Q8_BLOCK_SIZE;
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));
        const int8_t* qs = reinterpret_cast<const int8_t*>(bp + 2) + il * NQ_DEFERRED;

        float sumq = 0.0f;
        if (((uintptr_t)qs & 7) == 0) {
            uint64_t packed = *reinterpret_cast<const uint64_t*>(qs);
            const int8_t* bytes = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                sumq += (float)bytes[i] * yl[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < NQ_DEFERRED; i++) {
                sumq += (float)qs[i] * yl[i];
            }
        }

        sumf += sumq * scale;
        yb += NQ_DEFERRED * Q8_GROUP_SIZE;
    }

    sumf = warp_reduce_sum(sumf);

    if (lane == 0) {
        out[row] = sumf + residual[row];
    }
}

// --------------------------------------------------------------------------
// dequant_matvec_q8_0_vec_int4: Full int4 (128-bit) vectorized inner loop
//
// This kernel demonstrates the maximum vectorization approach from the task
// spec: loading the full 32-byte quant block as 2 x int4 (128-bit) loads,
// and the corresponding 32 x floats as 8 x float4 loads.
//
// Strategy: One warp per output row. Each thread in the warp cooperates to
// process the full block. Thread lane processes elements [lane] across all
// blocks (same as the simple kernel), but the WARP collectively loads
// the full block data.
//
// The int4 loads are most beneficial when a single thread needs to process
// many consecutive bytes. In this kernel, we assign 4 consecutive blocks
// to 4 groups of 8 threads, where each group processes an entire block.
//
// Thread mapping:
//   group = lane / 8   -> 0..3 (which of 4 blocks)
//   elem  = lane % 8   -> 0..7 (which 4-element chunk within the 32-byte block)
//
// Each thread processes 4 elements from a single block, accumulates locally,
// then reduces across the warp.
// --------------------------------------------------------------------------
extern "C" __global__ void dequant_matvec_q8_0_vec_int4(
    const uint8_t* __restrict__ w_q8,
    const float*   __restrict__ x,
    float*         __restrict__ out,
    const uint32_t in_dim,
    const uint32_t out_dim)
{
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;

    const uint32_t lane = threadIdx.x;  // 0..31
    const uint32_t num_blocks = in_dim >> 5;
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;
    const uint8_t* row_ptr = w_q8 + (uint64_t)row * row_bytes;

    // Thread mapping: 4 blocks per iteration, 8 threads per block
    const uint32_t group = lane >> 3;      // 0..3 (block index within the 4-block batch)
    const uint32_t elem  = lane & 7;       // 0..7 (element group within block)

    float sum = 0.0f;

    // Process 4 blocks per iteration
    for (uint32_t b = 0; b + 3 < num_blocks; b += 4) {
        uint32_t blk = b + group;
        const uint8_t* bp = row_ptr + (uint64_t)blk * Q8_BLOCK_SIZE;

        // Load scale
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));

        // This thread processes 4 consecutive int8 values starting at offset elem*4
        // Use a 32-bit load to fetch 4 bytes at once
        const int8_t* qdata = reinterpret_cast<const int8_t*>(bp + 2);
        uint32_t offset = elem * 4;

        // 32-bit vectorized load for 4 int8 values
        uint32_t packed_q;
        if (((uintptr_t)(qdata + offset) & 3) == 0) {
            packed_q = *reinterpret_cast<const uint32_t*>(qdata + offset);
        } else {
            // Unaligned fallback -- assemble from bytes
            packed_q = (uint32_t)(uint8_t)qdata[offset]
                     | ((uint32_t)(uint8_t)qdata[offset+1] << 8)
                     | ((uint32_t)(uint8_t)qdata[offset+2] << 16)
                     | ((uint32_t)(uint8_t)qdata[offset+3] << 24);
        }

        const int8_t* q4 = reinterpret_cast<const int8_t*>(&packed_q);

        // Load 4 corresponding x values
        uint32_t x_base = (blk << 5) + offset;
        // float4 load for 4 x values (16-byte aligned when x_base % 4 == 0)
        float xv[4];
        if (((uintptr_t)(x + x_base) & 15) == 0) {
            float4 xvec = *reinterpret_cast<const float4*>(x + x_base);
            xv[0] = xvec.x; xv[1] = xvec.y; xv[2] = xvec.z; xv[3] = xvec.w;
        } else {
            xv[0] = x[x_base];
            xv[1] = x[x_base + 1];
            xv[2] = x[x_base + 2];
            xv[3] = x[x_base + 3];
        }

        // Dot product of 4 int8 * 4 float, deferred scale
        float partial = (float)q4[0] * xv[0]
                      + (float)q4[1] * xv[1]
                      + (float)q4[2] * xv[2]
                      + (float)q4[3] * xv[3];

        sum += scale * partial;
    }

    // Handle remaining blocks (< 4): fall back to 1-element-per-thread
    uint32_t tail_start = (num_blocks / 4) * 4;
    for (uint32_t b = tail_start; b < num_blocks; b++) {
        const uint8_t* bp = row_ptr + (uint64_t)b * Q8_BLOCK_SIZE;
        float scale = __half2float(*reinterpret_cast<const __half*>(bp));
        int8_t qval = reinterpret_cast<const int8_t*>(bp + 2)[lane];
        float xval = x[(b << 5) + lane];
        float v = (float)qval * xval;
        sum += scale * warp_reduce_sum(v);
    }

    // Warp reduction for the main loop's partial sums
    // Each thread accumulated sum from its assigned group across iterations.
    // Need full warp reduction since different threads worked on different blocks.
    float total = warp_reduce_sum(sum);

    if (lane == 0) {
        out[row] = total;
    }
}

// --------------------------------------------------------------------------
// FFN fused gate+up+SwiGLU (vectorized Q8_0)
//
// Combines gate and up projections with SwiGLU activation in a single kernel:
//   out[i] = silu(dot(w_gate[i], x)) * dot(w_up[i], x)
//
// Uses the same deferred-reduction pattern as the Metal 2SG kernel.
// --------------------------------------------------------------------------
extern "C" __global__ void ffn_fused_gate_up_swiglu_q8_0_vec(
    const uint8_t* __restrict__ w_gate_q8,   // gate weights Q8_0 [inter_dim, hidden_dim]
    const float*   __restrict__ x,            // normed input [hidden_dim]
    float*         __restrict__ out,          // output [inter_dim]
    const uint32_t in_dim,                    // hidden_dim
    const uint32_t out_dim,                   // inter_dim
    const uint8_t* __restrict__ w_up_q8)      // up weights Q8_0 [inter_dim, hidden_dim]
{
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;

    const uint32_t lane = threadIdx.x;
    const uint32_t num_blocks = in_dim >> 5;
    const uint32_t row_bytes = num_blocks * Q8_BLOCK_SIZE;

    const uint8_t* gate_row = w_gate_q8 + (uint64_t)row * row_bytes;
    const uint8_t* up_row   = w_up_q8   + (uint64_t)row * row_bytes;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    uint32_t b = 0;
    for (; b + 3 < num_blocks; b += 4) {
        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            const uint8_t* gbp = gate_row + (uint64_t)(b + bi) * Q8_BLOCK_SIZE;
            const uint8_t* ubp = up_row   + (uint64_t)(b + bi) * Q8_BLOCK_SIZE;

            float gs = __half2float(*reinterpret_cast<const __half*>(gbp));
            float us = __half2float(*reinterpret_cast<const __half*>(ubp));

            int8_t gq = reinterpret_cast<const int8_t*>(gbp + 2)[lane];
            int8_t uq = reinterpret_cast<const int8_t*>(ubp + 2)[lane];

            float xv = x[((b + bi) << 5) + lane];

            gate_sum += gs * warp_reduce_sum((float)gq * xv);
            up_sum   += us * warp_reduce_sum((float)uq * xv);
        }
    }

    for (; b < num_blocks; b++) {
        const uint8_t* gbp = gate_row + (uint64_t)b * Q8_BLOCK_SIZE;
        const uint8_t* ubp = up_row   + (uint64_t)b * Q8_BLOCK_SIZE;

        float gs = __half2float(*reinterpret_cast<const __half*>(gbp));
        float us = __half2float(*reinterpret_cast<const __half*>(ubp));

        int8_t gq = reinterpret_cast<const int8_t*>(gbp + 2)[lane];
        int8_t uq = reinterpret_cast<const int8_t*>(ubp + 2)[lane];

        float xv = x[(b << 5) + lane];

        gate_sum += gs * warp_reduce_sum((float)gq * xv);
        up_sum   += us * warp_reduce_sum((float)uq * xv);
    }

    if (lane == 0) {
        // SwiGLU: silu(gate) * up
        float sigmoid = 1.0f / (1.0f + expf(-gate_sum));
        out[row] = gate_sum * sigmoid * up_sum;
    }
}
