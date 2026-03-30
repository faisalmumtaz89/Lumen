// RMSNorm kernel for CUDA.
//
// RMSNorm(x, weight, eps) = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// Uses a single-pass shared memory reduction for the sum-of-squares, with warp
// shuffle intrinsics (__shfl_xor_sync) for the final warp-level reduction.
// The kernel launches exactly 1 block of up to 1024 threads for the given dim.

// Warp-level sum reduction using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Single-block RMSNorm kernel.
//
// Each thread accumulates sum-of-squares for its strided elements, then a
// shared-memory reduction across warps computes the global sum. After the
// reduction, each thread applies the normalization to its elements.
//
// Block size should be a power of 2, up to 1024. Shared memory usage:
// (blockDim.x / 32) * sizeof(float) bytes (max 32 floats = 128 bytes).
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float eps,
    unsigned int dim)
{
    // Shared memory for per-warp partial sums (max 32 warps per block).
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;       // tid / 32
    unsigned int lane_id = tid & 31u;      // tid % 32
    unsigned int num_warps = block_size >> 5;

    // Phase 1: Each thread accumulates sum-of-squares for its strided elements.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Phase 2: Warp-level reduction.
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Store per-warp results to shared memory.
    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Phase 4: First warp reduces across all warp partial sums.
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Phase 5: Broadcast the RMS scale to all threads via shared memory.
    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 6: Apply normalization: out[i] = x[i] * rms * weight[i].
    for (unsigned int i = tid; i < dim; i += block_size) {
        out[i] = x[i] * rms * weight[i];
    }
}

// Per-head RMSNorm: applies RMSNorm independently to each head's [head_dim] vector.
//
// Used by Qwen3.5 full-attention layers for per-head Q and K normalization.
// Each block handles one head. weight can be:
//   - [num_heads * head_dim] (per-head weights), or
//   - [head_dim] (shared weights, periodic via modulo indexing — set weight_stride=0)
//
// Grid: (num_heads, 1, 1)
// Block: (min(head_dim, 1024), 1, 1)
// Shared memory: (blockDim.x / 32) * 4 bytes
extern "C" __global__ void rmsnorm_per_head(
    const float* __restrict__ x,       // [num_heads * head_dim]
    const float* __restrict__ weight,  // [num_heads * head_dim] or [head_dim]
    float* __restrict__ out,           // [num_heads * head_dim]
    float eps,
    unsigned int head_dim,
    unsigned int weight_stride)        // head_dim for per-head weights, 0 for shared
{
    extern __shared__ float shared[];

    unsigned int head = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int warp_id = tid >> 5;
    unsigned int lane_id = tid & 31u;
    unsigned int num_warps = block_size >> 5;

    const float* hx = x + head * head_dim;
    float* hout = out + head * head_dim;
    // weight_stride==0 means shared weights (modular indexing via head_dim)
    const float* hw = (weight_stride > 0) ? (weight + head * weight_stride) : weight;

    // Sum of squares for this head
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        float v = hx[i];
        sum_sq += v * v;
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    // Cross-warp reduction
    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum(total);
    }

    // Broadcast RMS scale
    float rms;
    if (tid == 0) {
        rms = rsqrtf(total / (float)head_dim + eps);
        shared[0] = rms;
    }
    __syncthreads();
    rms = shared[0];

    // Normalize with per-head (or shared) weight
    for (unsigned int i = tid; i < head_dim; i += block_size) {
        unsigned int wi = (weight_stride > 0) ? i : (i % head_dim);
        hout[i] = hx[i] * rms * hw[wi];
    }
}

// ============================================================================
// Fused Residual Add + RMSNorm (F32 output).
//
// Combines the end of one transformer layer with the start of the next for
// quantized (Q8_0/Q4_0) decode paths that consume F32 normed activations:
//   1. Residual add: x_out[i] = a[i] + b[i]  (e.g., attn_proj + ffn_down)
//   2. RMSNorm: rms_scale = 1/sqrt(mean(x^2) + eps)
//   3. F32 output: normed[i] = x_out[i] * rms_scale * weight[i]
//
// Eliminates 1 dispatch per inter-layer boundary by merging residual_add_copy
// and rmsnorm into a single kernel.  For 48-layer models: 47 fewer dispatches.
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_residual_rmsnorm_f32(
    const float* __restrict__ a,       // [dim] first residual input
    const float* __restrict__ b,       // [dim] second residual input
    float* __restrict__ x_out,         // [dim] summed output (for later use as residual)
    const float* __restrict__ weight,  // [dim] RMSNorm weights (F32)
    float* __restrict__ normed,        // [dim] normalized F32 output
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
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = a[i] + b[i];
        x_out[i] = val;
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

    // Phase 6: Normalize using stored x values (F32 output).
    for (unsigned int i = tid; i < dim; i += block_size) {
        normed[i] = x_out[i] * rms * weight[i];
    }
}

// ============================================================================
// Fused Residual Add + compute_rms_scale (scalar output).
//
// Combines residual_add_copy with compute_rms_scale for the fused_glu_gemv
// path.  Writes x_out = a + b, then computes and stores the scalar
// rms_scale = 1/sqrt(mean(x_out^2) + eps).
//
// This replaces 2 dispatches (residual_add_copy + compute_rms_scale) with 1,
// saving 1 dispatch per inter-layer boundary for quantized decode paths that
// use the fused GLU GEMV kernel.
//
// Dispatch: grid = (1), block = (block_size), shmem = (block_size / 32) * 4
// ============================================================================
extern "C" __global__ void fused_residual_rms_scale(
    const float* __restrict__ a,       // [dim] first residual input
    const float* __restrict__ b,       // [dim] second residual input
    float* __restrict__ x_out,         // [dim] summed output (for later use)
    float* __restrict__ out_scale,     // [1]   scalar rms_scale output
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
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = a[i] + b[i];
        x_out[i] = val;
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

    // Phase 5: Write scalar rms_scale to global memory.
    if (tid == 0) {
        out_scale[0] = 1.0f / sqrtf(total / (float)dim + eps);
    }
}
