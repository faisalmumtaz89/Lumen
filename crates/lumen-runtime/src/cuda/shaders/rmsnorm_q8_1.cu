// ============================================================================
// Fused RMSNorm + Q8_1 Quantization Kernel
//
// Combines two separate dispatches into one:
//   1. RMSNorm: rms_scale = 1/sqrt(mean(x^2) + eps), normed[i] = x[i] * rms_scale * weight[i]
//   2. Q8_1 quantize: group normed values into 32-element blocks, find absmax,
//      quantize to int8, write [f16 scale, f16 sum, 32 x int8] blocks
//
// Saves 1 kernel dispatch per fusion site. For Q8_0 decode with dp4a Q8_1
// pre-quantization: 2 sites/layer (attn_norm + ffn_norm) = 2 fewer dispatches
// per layer = 72 fewer for 36-layer models.
//
// Architecture:
//   - Single block of up to 1024 threads (same as rmsnorm)
//   - Phase 1: Cooperative sum-of-squares reduction (warp shuffle + shmem)
//   - Phase 2: Broadcast rms_scale to all threads
//   - Phase 3: Each warp handles Q8_1 blocks in strided fashion:
//     - Apply normalization inline: val = x[i] * rms_scale * weight[i]
//     - Warp-wide absmax reduction for scale computation
//     - Quantize to int8 and write Q8_1 block header + quants
//
// Q8_1 block layout (36 bytes per 32 elements):
//   bytes [0..1]: f16 scale (d = max(|normed|) / 127)
//   bytes [2..3]: f16 weighted sum (s = d * sum(quants))
//   bytes [4..35]: 32 x int8 quantized values
//
// Dispatch: grid = (1), block = (block_size), block_size = min(dim, 1024)
// Shared memory: (block_size / 32) * sizeof(float)
//
// dim MUST be a multiple of 32 (Q8_1 block size).
//
// NVRTC-compatible: no system includes, extern "C" linkage.
// ============================================================================

#define WARP_SIZE 32
#define Q8_1_BYTES 36

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Hardware f32->f16 conversion via PTX.
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

extern "C" __global__ void rmsnorm_to_q8_1(
    const float* __restrict__ x,         // [dim] input activation
    const float* __restrict__ weight,    // [dim] RMSNorm weight
    char* __restrict__ output_q8_1,      // [dim/32 * 36] Q8_1 output
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int num_warps = block_size >> 5;

    // ---- Phase 1: Compute sum-of-squares for RMSNorm ----
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warp partial sums.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // ---- Phase 2: Broadcast rms_scale ----
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // ---- Phase 3: Normalize + Quantize to Q8_1 blocks ----
    // Each warp handles one Q8_1 block (32 elements) per iteration.
    // Warps stride by num_warps blocks.
    const unsigned int num_blocks = dim >> 5;  // dim / 32

    for (unsigned int blk = warp_id; blk < num_blocks; blk += num_warps) {
        unsigned int base = blk * WARP_SIZE;
        unsigned int idx = base + lane_id;

        // Apply RMSNorm inline: normed = x[i] * rms * weight[i].
        float val = x[idx] * rms * weight[idx];

        // Warp-wide absolute max reduction for Q8_1 scale.
        float amax = val < 0.0f ? -val : val;
        float tmp;
        tmp = __shfl_xor_sync(0xffffffff, amax, 16);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 8);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 4);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 2);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 1);
        amax = tmp > amax ? tmp : amax;

        // Compute Q8_1 scale and inverse.
        float scale = amax / 127.0f;
        float scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        // Quantize.
        int qi = __float2int_rn(val * scale_inv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);

        // Compute sum(quants) for Q8_1 sum field.
        float qi_f = (float)qi;
        float qsum = qi_f;
        qsum += __shfl_xor_sync(0xffffffff, qsum, 16);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 8);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 4);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 2);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 1);

        float weighted_sum = scale * qsum;

        // Write Q8_1 block.
        char* block_out = output_q8_1 + (unsigned long long)blk * Q8_1_BYTES;

        if (lane_id == 0) {
            // Write f16 scale (bytes 0-1, little-endian).
            unsigned short d_f16 = f32_to_f16_bits(scale);
            block_out[0] = (char)(d_f16 & 0xFF);
            block_out[1] = (char)((d_f16 >> 8) & 0xFF);

            // Write f16 weighted sum (bytes 2-3).
            unsigned short s_f16 = f32_to_f16_bits(weighted_sum);
            block_out[2] = (char)(s_f16 & 0xFF);
            block_out[3] = (char)((s_f16 >> 8) & 0xFF);
        }

        // All lanes write their quantized byte.
        block_out[4 + lane_id] = (char)(qi & 0xFF);
    }
}

// ============================================================================
// Fused RMSNorm + Q8_1 Quantization Kernel — RAW-SUM convention.
//
// IDENTICAL to `rmsnorm_to_q8_1` above EXCEPT the Q8_1 `s` (sum) field:
//   - `rmsnorm_to_q8_1` writes  s = d * sum(quantized_int8)   (the dp4a-aligned
//     "weighted sum" convention consumed by matvec_q4_aligned_q8_1 etc.)
//   - THIS kernel writes        s = raw F32 sum(normed values)  — exactly
//     matching the standalone `quantize_q8_1_rawsum` kernel (mmv_q.cu), whose
//     `s` is consumed by the `mul_mat_vec_q_q4_0` bias-correction term
//     `return d4 * (sumi*ds8.x - 4*ds8.y)`.
//
// This kernel exists so the Q4 lm_head ("Path -1": quantize_q8_1_rawsum +
// mul_mat_vec_q_q4_0) can FUSE its standalone `rmsnorm` + `quantize_q8_1_rawsum`
// into ONE launch, dropping 1 dispatch + the F32 `normed` HBM round-trip, while
// remaining BIT-IDENTICAL to the unfused path.
//
// BIT-IDENTITY ARGUMENT (vs `rmsnorm` then `quantize_q8_1_rawsum`):
//   * Phase1/2 reduction tree is byte-for-byte the standalone `rmsnorm`'s tree
//     (warp_reduce_sum butterfly -> shared[warp_id] -> warp0 re-reduce ->
//     rms = 1/sqrtf(total/dim + eps)), so `total` and `rms` are identical when
//     launched with the SAME block_size (num_warps).
//   * Phase3 computes `val = x[idx] * rms * weight[idx]` in the SAME left-to-
//     right operand order as the standalone rmsnorm's `out[i]=x[i]*rms*weight[i]`.
//     It is a pure mul-mul chain (no add) so NO FMA contraction can occur; the
//     in-register `val` is bit-identical to the F32 value the standalone path
//     stores to `normed[]` and reloads (an F32 store/reload is a bitwise no-op).
//   * amax = warp_reduce_max over the same 32 lane values; d = amax/127;
//     q = (amax==0)?0:__float2int_rn(val/d); raw sum = warp_reduce_sum(val) —
//     all identical to quantize_q8_1_rawsum operating on the reloaded `normed`.
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4.
// `dim` MUST be a multiple of 32. block_size MUST equal the standalone
// `rmsnorm`'s block_size (rmsnorm_block_size(dim)) for the rms reduction tree
// to match bit-for-bit.
// ============================================================================
extern "C" __global__ void rmsnorm_to_q8_1_rawsum(
    const float* __restrict__ x,         // [dim] input activation
    const float* __restrict__ weight,    // [dim] RMSNorm weight
    char* __restrict__ output_q8_1,      // [dim/32 * 36] Q8_1 output (raw-sum s)
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int num_warps = block_size >> 5;

    // ---- Phase 1: Compute sum-of-squares for RMSNorm (identical to rmsnorm) ----
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = x[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // ---- Phase 2: Broadcast rms_scale ----
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // ---- Phase 3: Normalize + Quantize to Q8_1 blocks (RAW-SUM convention) ----
    const unsigned int num_blocks = dim >> 5;  // dim / 32

    for (unsigned int blk = warp_id; blk < num_blocks; blk += num_warps) {
        unsigned int base = blk * WARP_SIZE;
        unsigned int idx = base + lane_id;

        // Apply RMSNorm inline: normed = x[i] * rms * weight[i].
        // SAME left-to-right operand order as the standalone rmsnorm Phase6
        // (out[i] = x[i] * rms * weight[i]); pure mul-mul, no FMA contraction.
        float val = x[idx] * rms * weight[idx];

        // Warp-wide absolute max reduction for the Q8_1 scale.
        float amax = fabsf(val);
        float tmp;
        tmp = __shfl_xor_sync(0xffffffff, amax, 16);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 8);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 4);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 2);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 1);
        amax = tmp > amax ? tmp : amax;

        // Q8_1 scale and quantize — EXACTLY matching quantize_q8_1_rawsum:
        //   d = amax / 127; q = (amax==0)?0:(int8_t)__float2int_rn(val / d).
        // Same int8_t cast as the standalone (val/d is in [-127,127] by
        // construction since d = amax/127, so __float2int_rn ∈ [-127,127]).
        const float d = amax / 127.0f;
        const signed char qi =
            (amax == 0.0f) ? (signed char)0 : (signed char)__float2int_rn(val / d);

        // RAW F32 sum of the 32 normed values (NOT d*sum(quants)).
        float raw_sum = val;
        raw_sum += __shfl_xor_sync(0xffffffff, raw_sum, 16);
        raw_sum += __shfl_xor_sync(0xffffffff, raw_sum, 8);
        raw_sum += __shfl_xor_sync(0xffffffff, raw_sum, 4);
        raw_sum += __shfl_xor_sync(0xffffffff, raw_sum, 2);
        raw_sum += __shfl_xor_sync(0xffffffff, raw_sum, 1);

        // Write Q8_1 block.
        char* block_out = output_q8_1 + (unsigned long long)blk * Q8_1_BYTES;

        if (lane_id == 0) {
            // f16 d (bytes 0-1, little-endian).
            unsigned short d_f16 = f32_to_f16_bits(d);
            block_out[0] = (char)(d_f16 & 0xFF);
            block_out[1] = (char)((d_f16 >> 8) & 0xFF);

            // f16 raw sum (bytes 2-3).
            unsigned short s_f16 = f32_to_f16_bits(raw_sum);
            block_out[2] = (char)(s_f16 & 0xFF);
            block_out[3] = (char)((s_f16 >> 8) & 0xFF);
        }

        // All lanes write their quantized byte (same byte as
        // quantize_q8_1_rawsum's `yb[4+iqs] = (unsigned char)q`).
        block_out[4 + lane_id] = (char)(unsigned char)qi;
    }
}

// ============================================================================
// Fused Residual Add + RMSNorm + Q8_1 Quantization Kernel
//
// Combines the inter-layer boundary fusion with Q8_1 quantization:
//   1. Residual add: x_out[i] = a[i] + b[i]
//   2. RMSNorm: rms_scale = 1/sqrt(mean(x_out^2) + eps)
//   3. Q8_1 quantize the normed output
//
// Eliminates 2 dispatches per inter-layer boundary vs separate
// residual_add_copy + rmsnorm + quantize_f32_to_q8_1 (3 -> 1).
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_residual_rmsnorm_q8_1(
    const float* __restrict__ a,         // [dim] first residual input
    const float* __restrict__ b,         // [dim] second residual input
    float* __restrict__ x_out,           // [dim] summed output (for later use as residual)
    const float* __restrict__ weight,    // [dim] RMSNorm weights
    char* __restrict__ output_q8_1,      // [dim/32 * 36] Q8_1 output
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int num_warps = block_size >> 5;

    // ---- Phase 1: Residual add + sum-of-squares ----
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = a[i] + b[i];
        x_out[i] = val;
        sum_sq += val * val;
    }

    // Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // ---- Phase 2: Broadcast rms_scale ----
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // ---- Phase 3: Normalize + Quantize to Q8_1 blocks ----
    const unsigned int num_blocks = dim >> 5;

    for (unsigned int blk = warp_id; blk < num_blocks; blk += num_warps) {
        unsigned int base = blk * WARP_SIZE;
        unsigned int idx = base + lane_id;

        // Read the summed value back (already in x_out from Phase 1).
        float val = x_out[idx] * rms * weight[idx];

        float amax = val < 0.0f ? -val : val;
        float tmp;
        tmp = __shfl_xor_sync(0xffffffff, amax, 16);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 8);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 4);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 2);
        amax = tmp > amax ? tmp : amax;
        tmp = __shfl_xor_sync(0xffffffff, amax, 1);
        amax = tmp > amax ? tmp : amax;

        float scale = amax / 127.0f;
        float scale_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int qi = __float2int_rn(val * scale_inv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);

        float qi_f = (float)qi;
        float qsum = qi_f;
        qsum += __shfl_xor_sync(0xffffffff, qsum, 16);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 8);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 4);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 2);
        qsum += __shfl_xor_sync(0xffffffff, qsum, 1);

        float weighted_sum = scale * qsum;

        char* block_out = output_q8_1 + (unsigned long long)blk * Q8_1_BYTES;

        if (lane_id == 0) {
            unsigned short d_f16 = f32_to_f16_bits(scale);
            block_out[0] = (char)(d_f16 & 0xFF);
            block_out[1] = (char)((d_f16 >> 8) & 0xFF);

            unsigned short s_f16 = f32_to_f16_bits(weighted_sum);
            block_out[2] = (char)(s_f16 & 0xFF);
            block_out[3] = (char)((s_f16 >> 8) & 0xFF);
        }

        block_out[4 + lane_id] = (char)(qi & 0xFF);
    }
}
