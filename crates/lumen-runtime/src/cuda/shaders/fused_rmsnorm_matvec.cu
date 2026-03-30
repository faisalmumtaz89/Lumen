// ============================================================================
// Fused RMSNorm + Matrix-Vector Multiply (Two-Pass Approach)
//
// Eliminates the intermediate `normed[hidden_dim]` buffer by splitting the
// operation into two kernels:
//
//   Pass 1 (compute_rms_scale): Single block computes
//           rms_scale = 1 / sqrt(mean(x^2) + eps)
//           and writes ONE scalar float to device memory.
//
//   Pass 2 (fused_norm_matvec_f32): Each block computes one output row:
//           out[row] = dot(W[row], x * rms_scale * norm_weight)
//           The normalization is applied inline during the dot product --
//           the normalized vector is never materialized in global memory.
//
// Savings vs separate rmsnorm + matvec:
//   - 1 fewer kernel launch (~5 us on typical GPUs)
//   - Eliminates 2 * dim * sizeof(float) global memory traffic
//     (the normed[] write from rmsnorm + the normed[] read from matvec)
//   - Only adds 1 scalar write + out_dim scalar reads (4 bytes each, negligible)
//
// The standalone rmsnorm and matvec kernels are kept in norm.cu / matvec_f32.cu
// for paths where fusion is not applicable (e.g., normed output feeds multiple
// consumers, or for correctness testing as the reference implementation).
// ============================================================================

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
// Pass 1: Compute RMS scale factor
//
// rms_scale = 1 / sqrt(mean(x^2) + eps)
//
// Dispatch: grid = (1), block = (block_size)
// Shared memory: (block_size / 32) * sizeof(float)
//
// Writes exactly 1 float to `out_scale`. This is the only global write.
// ============================================================================
extern "C" __global__ void compute_rms_scale(
    const float* __restrict__ x,     // [dim] input activation
    float* __restrict__ out_scale,   // [1] output: rms_scale scalar
    const float eps,
    const unsigned int dim
)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane_id = tid & 31u;
    const unsigned int num_warps = block_size >> 5;

    extern __shared__ float shared[];

    // Accumulate sum-of-squares with strided access.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float v = x[i];
        sum_sq += v * v;
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

    // Thread 0 writes the scalar result.
    if (tid == 0) {
        out_scale[0] = rsqrtf(total / (float)dim + eps);
    }
}

// ============================================================================
// Pass 2: Fused Norm + Matrix-Vector Multiply (F32)
//
// out[row] = dot(W[row], x * rms_scale * norm_weight)
//
// The normalized value `x[j] * rms_scale * norm_weight[j]` is computed inline
// during the dot product. The normed vector is never written to global memory.
//
// Uses float4 vectorized loads when dim is a multiple of 4 for bandwidth
// efficiency (4x fewer load transactions). Falls back to scalar for unaligned.
//
// Dispatch: grid = (out_dim), block = (256)
// Shared memory: 0 (block_reduce_sum uses statically-allocated shared memory)
// ============================================================================

#define FUSED_BLOCK_SIZE 256
#define FUSED_WARP_SIZE 32

// Block-level reduction: warp shuffle + shared memory across warps.
__device__ float fused_block_reduce_sum(float val) {
    val = warp_reduce_sum(val);

    __shared__ float warp_sums[FUSED_BLOCK_SIZE / FUSED_WARP_SIZE];
    unsigned int lane = threadIdx.x & (FUSED_WARP_SIZE - 1);
    unsigned int warp_id = threadIdx.x / FUSED_WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < (FUSED_BLOCK_SIZE / FUSED_WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

extern "C" __global__ void fused_norm_matvec_f32(
    const float* __restrict__ x,            // [dim] input activation
    const float* __restrict__ rms_scale,    // [1] precomputed rms_scale scalar
    const float* __restrict__ norm_weight,  // [dim] RMSNorm weights
    const float* __restrict__ weight,       // [out_dim, dim] row-major
    float* __restrict__ out,                // [out_dim]
    const unsigned int dim,
    const unsigned int out_dim
)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const float scale = rms_scale[0];
    const float* __restrict__ w_row = weight + (unsigned long long)row * dim;
    float sum = 0.0f;

    // float4 path: 4x fewer load transactions when dim is 4-aligned.
    if ((dim & 3u) == 0u) {
        unsigned int vec4_count = dim >> 2;
        const float4* w_vec4 = (const float4*)w_row;
        const float4* x_vec4 = (const float4*)x;
        const float4* nw_vec4 = (const float4*)norm_weight;

        for (unsigned int i = threadIdx.x; i < vec4_count; i += FUSED_BLOCK_SIZE) {
            float4 w = w_vec4[i];
            float4 xv = x_vec4[i];
            float4 nw = nw_vec4[i];

            // Inline normalization: normed_j = x[j] * scale * norm_weight[j]
            sum += w.x * (xv.x * scale * nw.x);
            sum += w.y * (xv.y * scale * nw.y);
            sum += w.z * (xv.z * scale * nw.z);
            sum += w.w * (xv.w * scale * nw.w);
        }
    } else {
        // Scalar fallback for non-aligned dim.
        for (unsigned int j = threadIdx.x; j < dim; j += FUSED_BLOCK_SIZE) {
            float normed_j = x[j] * scale * norm_weight[j];
            sum += w_row[j] * normed_j;
        }
    }

    sum = fused_block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Pass 2 variant: Fused Norm + Dual Matrix-Vector Multiply (F32)
//
// Computes two independent matvecs from the same normalized input:
//   gate[row] = dot(W_gate[row], x * rms_scale * norm_weight)
//   up[row]   = dot(W_up[row],   x * rms_scale * norm_weight)
//
// Used for the FFN gate+up pattern. Saves 1 additional kernel launch vs
// separate fused_norm_matvec calls (the rms_scale read is shared).
//
// Dispatch: grid = (out_dim), block = (256)
// ============================================================================

extern "C" __global__ void fused_norm_dual_matvec_f32(
    const float* __restrict__ x,            // [dim] input activation
    const float* __restrict__ rms_scale,    // [1] precomputed rms_scale scalar
    const float* __restrict__ norm_weight,  // [dim] RMSNorm weights
    const float* __restrict__ w_gate,       // [out_dim, dim] gate projection
    const float* __restrict__ w_up,         // [out_dim, dim] up projection
    float* __restrict__ out_gate,           // [out_dim]
    float* __restrict__ out_up,             // [out_dim]
    const unsigned int dim,
    const unsigned int out_dim
)
{
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const float scale = rms_scale[0];
    const float* __restrict__ gate_row = w_gate + (unsigned long long)row * dim;
    const float* __restrict__ up_row   = w_up   + (unsigned long long)row * dim;
    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // float4 path for bandwidth efficiency.
    if ((dim & 3u) == 0u) {
        unsigned int vec4_count = dim >> 2;
        const float4* x_vec4 = (const float4*)x;
        const float4* nw_vec4 = (const float4*)norm_weight;
        const float4* g_vec4 = (const float4*)gate_row;
        const float4* u_vec4 = (const float4*)up_row;

        for (unsigned int i = threadIdx.x; i < vec4_count; i += FUSED_BLOCK_SIZE) {
            float4 xv = x_vec4[i];
            float4 nw = nw_vec4[i];
            float4 gw = g_vec4[i];
            float4 uw = u_vec4[i];

            float n0 = xv.x * scale * nw.x;
            float n1 = xv.y * scale * nw.y;
            float n2 = xv.z * scale * nw.z;
            float n3 = xv.w * scale * nw.w;

            gate_sum += gw.x * n0 + gw.y * n1 + gw.z * n2 + gw.w * n3;
            up_sum   += uw.x * n0 + uw.y * n1 + uw.z * n2 + uw.w * n3;
        }
    } else {
        for (unsigned int j = threadIdx.x; j < dim; j += FUSED_BLOCK_SIZE) {
            float normed_j = x[j] * scale * norm_weight[j];
            gate_sum += gate_row[j] * normed_j;
            up_sum   += up_row[j]   * normed_j;
        }
    }

    // Reduce gate dot product.
    gate_sum = fused_block_reduce_sum(gate_sum);
    float final_gate = gate_sum;
    __syncthreads();

    // Reduce up dot product.
    up_sum = fused_block_reduce_sum(up_sum);

    if (threadIdx.x == 0) {
        out_gate[row] = final_gate;
        out_up[row]   = up_sum;
    }
}
