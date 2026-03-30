// ============================================================================
// Fused kernels for F16 HGEMV decode path: dispatch count reduction.
//
// These kernels eliminate intermediate dispatches by fusing operations that
// would otherwise require separate kernel launches:
//
//   1. fused_rmsnorm_f16: RMSNorm + F32->F16 conversion in a single kernel.
//      Replaces the two-dispatch sequence: rmsnorm -> f32_to_f16_vec.
//      The normalized F16 output feeds cuBLAS HGEMV directly.
//
//   2. swiglu_f32_to_f16: SwiGLU activation + F32->F16 conversion.
//      Replaces the two-dispatch sequence: swiglu_inplace -> f32_to_f16_vec.
//      The activated F16 output feeds the down-projection HGEMV directly.
//
// Savings: 3 fewer dispatches per layer (2 from fused_rmsnorm_f16 at
// attn_norm and ffn_norm sites, 1 from swiglu_f32_to_f16).
// For 32 layers: 96 fewer dispatches per token.
// ============================================================================

// Hardware f32->f16 conversion via PTX (single instruction on SM 53+).
// Duplicated from convert_f16.cu to avoid symbol conflicts since each .cu file
// compiles to a separate PTX module.
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// Fused RMSNorm + F32->F16 conversion.
//
// Computes RMSNorm(x, weight, eps) and writes the result as F16 half bits.
// This eliminates the intermediate F32 normed[] buffer entirely.
//
// RMSNorm(x, weight, eps) = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_rmsnorm_f16(
    const float* __restrict__ x,         // [dim] input activation (F32)
    const float* __restrict__ weight,    // [dim] RMSNorm weights (F32)
    unsigned short* __restrict__ out_f16, // [dim] output (F16 bits)
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    // Phase 1: Sum-of-squares reduction.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Phase 4: Cross-warp reduction.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 5: Broadcast RMS scale.
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 6: Normalize and convert to F16 in one pass.
    // No intermediate F32 buffer -- write directly as F16 bits.
    for (unsigned int i = tid; i < dim; i += block_size) {
        float normed = x[i] * rms * weight[i];
        out_f16[i] = f32_to_f16_bits(normed);
    }
}

// ============================================================================
// Fused Batched RMSNorm + F32->F16 conversion (for prefill).
//
// Batched version of fused_rmsnorm_f16: processes [batch, dim] input.
// One threadblock per row (batch element), same as rmsnorm_batched.
// Output is [batch, dim] in F16 (unsigned short), eliminating the separate
// f32_to_f16_vec dispatch before HGEMM projections.
//
// Saves 1 kernel launch per norm site per layer during prefill.
// For a 32-layer model with 2 norm sites: 64 fewer launches per prefill.
//
// Dispatch: grid = (batch), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_rmsnorm_f16_batched(
    const float* __restrict__ x,          // [batch, dim] input activation (F32)
    const float* __restrict__ weight,     // [dim] RMSNorm weights (F32)
    unsigned short* __restrict__ out_f16, // [batch, dim] output (F16 bits)
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    unsigned int batch_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    // Address the correct row using 64-bit arithmetic.
    const float* row_in = x + (unsigned long long)batch_idx * dim;
    unsigned short* row_out = out_f16 + (unsigned long long)batch_idx * dim;

    // Phase 1: Sum-of-squares reduction.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = row_in[i];
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Phase 4: Cross-warp reduction.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 5: Broadcast RMS scale.
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 6: Normalize and convert to F16 in one pass.
    for (unsigned int i = tid; i < dim; i += block_size) {
        float normed = row_in[i] * rms * weight[i];
        row_out[i] = f32_to_f16_bits(normed);
    }
}

// ============================================================================
// Fused Batched SwiGLU + F32->F16 conversion (for prefill).
//
// Batched version: processes [batch * inter_dim] elements.
// Computes SwiGLU(gate, up) and writes F16 output for down-projection HGEMM.
// The F32 result is also written back to gate for potential later use.
//
// Saves 1 kernel launch per layer during prefill (replaces swiglu_batched +
// f32_to_f16_vec).
//
// Dispatch: grid = ceil(total / 256), block = 256
// ============================================================================
extern "C" __global__ void swiglu_f32_to_f16_batched(
    float*                    gate,      // [total] gate (F32, in-place write)
    const float* __restrict__ up,        // [total] up (F32)
    unsigned short* __restrict__ out_f16, // [total] output (F16 bits)
    unsigned int total)                  // batch * inter_dim
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    float result = silu_g * up[idx];

    gate[idx] = result;
    out_f16[idx] = f32_to_f16_bits(result);
}

// ============================================================================
// Fused SwiGLU (in-place on gate) + F32->F16 conversion.
//
// Computes SwiGLU(gate, up) = SiLU(gate[i]) * up[i] and writes:
//   1. The F32 result back to gate[i] (in-place, same as swiglu_inplace)
//   2. The F16 result to out_f16[i] (for the down-projection HGEMV input)
//
// This replaces the two-dispatch sequence:
//   swiglu_inplace (writes F32 gate) + f32_to_f16_vec (converts gate to F16)
//
// Safe for in-place: each thread reads gate[idx] before writing gate[idx].
// No __restrict__ on gate since it is both read and written.
//
// SiLU(x) = x / (1 + exp(-x))
// SwiGLU(gate, up) = SiLU(gate[i]) * up[i]
//
// Dispatch: grid = ceil(n / 256), block = 256
// ============================================================================
extern "C" __global__ void swiglu_f32_to_f16(
    float*                    gate,     // [n] gate activation (F32, in-place write)
    const float* __restrict__ up,       // [n] up activation (F32)
    unsigned short* __restrict__ out_f16, // [n] output (F16 bits)
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    // SiLU: g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    float result = silu_g * up[idx];

    // Write F32 in-place to gate (for down-projection reads) and F16 to scratch.
    gate[idx] = result;
    out_f16[idx] = f32_to_f16_bits(result);
}

// ============================================================================
// Fused Residual Add + RMSNorm + F32->F16 conversion.
//
// Combines the end of one transformer layer with the start of the next:
//   1. Residual add: x[i] = a[i] + b[i]  (e.g., attn_proj + ffn_down)
//   2. RMSNorm: rms_scale = 1/sqrt(mean(x^2) + eps)
//   3. F16 output: out_f16[i] = f16(x[i] * rms_scale * weight[i])
//
// Eliminates 2 separate kernel dispatches (residual_add_copy + fused_rmsnorm_f16)
// per inter-layer boundary. For 36-layer models: 35 fewer dispatches per token.
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_residual_rmsnorm_f16(
    const float* __restrict__ a,          // [dim] first residual input
    const float* __restrict__ b,          // [dim] second residual input
    float* __restrict__ x_out,            // [dim] summed output (for later use as residual)
    const float* __restrict__ weight,     // [dim] RMSNorm weights (F32)
    unsigned short* __restrict__ out_f16, // [dim] normalized F16 output
    float eps,
    unsigned int dim)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    // Phase 1: Residual add + sum-of-squares reduction.
    // Compute x = a + b and accumulate x^2 simultaneously.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = a[i] + b[i];
        x_out[i] = val;  // Store summed result for later residual use
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Phase 4: Cross-warp reduction.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 5: Broadcast RMS scale.
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 6: Normalize using stored x values and convert to F16.
    for (unsigned int i = tid; i < dim; i += block_size) {
        float normed = x_out[i] * rms * weight[i];
        out_f16[i] = f32_to_f16_bits(normed);
    }
}
