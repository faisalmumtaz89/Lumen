// Batched prefill kernels for CUDA.
//
// These kernels operate on [batch, dim] activation matrices instead of single
// [dim] vectors. Used by the batched prefill path where projections are GEMM
// instead of matvec, giving ~100x fewer kernel launches.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// ---------- f16 bit conversion (shared with embed.cu) ----------

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// ---------- Batch embedding: gather multiple token rows ----------
//
// Grid: 1D, one thread per element in [batch * hidden_dim].
// Each thread copies one element from the embedding table to the output.

extern "C" __global__ void embed_batch_f32(
    const float* __restrict__ embedding_table,
    const unsigned int* __restrict__ token_ids,  // [batch]
    float* __restrict__ output,                  // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int row = idx / hidden_dim;
    unsigned int col = idx % hidden_dim;
    unsigned int token_id = token_ids[row];
    output[idx] = embedding_table[(unsigned long long)token_id * hidden_dim + col];
}

extern "C" __global__ void embed_batch_q8_0(
    const char* __restrict__ embedding_q8,
    const unsigned int* __restrict__ token_ids,  // [batch]
    float* __restrict__ output,                  // [batch, hidden_dim]
    unsigned int batch,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * hidden_dim;
    if (idx >= total) return;

    unsigned int row = idx / hidden_dim;
    unsigned int col = idx % hidden_dim;
    unsigned int token_id = token_ids[row];

    unsigned int global_elem = token_id * hidden_dim + col;
    unsigned int block_idx = global_elem >> 5;
    unsigned int elem_in_block = global_elem & 31u;

    const char* block_ptr = embedding_q8 + block_idx * 34;

    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    float val = (float)(signed char)block_ptr[2 + elem_in_block];
    output[idx] = val * scale;
}

// ---------- Batched RMSNorm ----------
//
// Grid: (batch, 1, 1) -- one block per row.
// Block: (block_size, 1, 1) -- up to 1024 threads.
// Shared memory: (block_size / 32) floats for warp partial sums.
//
// Each block normalizes one row of the [batch, dim] input matrix.
// out[b][i] = x[b][i] * weight[i] / sqrt(mean(x[b]^2) + eps)

__device__ __forceinline__ float warp_reduce_sum_pf(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

extern "C" __global__ void rmsnorm_batched(
    const float* __restrict__ x,       // [batch, dim]
    const float* __restrict__ weight,  // [dim]
    float* __restrict__ out,           // [batch, dim]
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

    const float* row_in = x + (unsigned long long)batch_idx * dim;
    float* row_out = out + (unsigned long long)batch_idx * dim;

    // Phase 1: sum of squares.
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += block_size) {
        float val = row_in[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum_pf(sum_sq);
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        total = warp_reduce_sum_pf(total);
    }

    if (tid == 0) {
        float rms = 1.0f / sqrtf(total / (float)dim + eps);
        shared[0] = rms;
    }
    __syncthreads();

    float rms = shared[0];

    // Phase 2: normalize.
    for (unsigned int i = tid; i < dim; i += block_size) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

// ---------- Batched RoPE ----------
//
// Grid: 1D, one thread per (token, pair) in the Q space.
// Processes Q and K heads for all tokens in the batch.
// Each token has its own position: pos_start + token_index.

extern "C" __global__ void rope_apply_batched(
    float* __restrict__ q,         // [batch, q_dim]
    float* __restrict__ k,         // [batch, kv_dim]
    unsigned int pos_start,        // position of first token in batch
    unsigned int batch,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float theta_base)
{
    unsigned int half_dim = head_dim >> 1;
    unsigned int total_q_pairs = num_q_heads * half_dim;
    unsigned int total_k_pairs = num_kv_heads * half_dim;
    unsigned int q_dim = num_q_heads * head_dim;
    unsigned int kv_dim = num_kv_heads * head_dim;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_work = batch * total_q_pairs;
    if (idx >= total_work) return;

    unsigned int token = idx / total_q_pairs;
    unsigned int pair_idx = idx % total_q_pairs;
    unsigned int pos = pos_start + token;

    // Process Q.
    {
        unsigned int d = pair_idx % half_dim;
        unsigned int head_offset = (pair_idx / half_dim) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int base = token * q_dim + head_offset + 2 * d;
        float x0 = q[base];
        float x1 = q[base + 1];
        q[base]     = x0 * cos_a - x1 * sin_a;
        q[base + 1] = x0 * sin_a + x1 * cos_a;
    }

    // Process K (only if this thread is also within K range).
    if (pair_idx < total_k_pairs) {
        unsigned int d = pair_idx % half_dim;
        unsigned int head_offset = (pair_idx / half_dim) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int base = token * kv_dim + head_offset + 2 * d;
        float x0 = k[base];
        float x1 = k[base + 1];
        k[base]     = x0 * cos_a - x1 * sin_a;
        k[base + 1] = x0 * sin_a + x1 * cos_a;
    }
}

// ---------- Batched KV cache write ----------
//
// Writes batch tokens' K or V data to the cache at positions pos_start..pos_start+batch-1.
// Grid: 1D, one thread per element in [batch * num_kv_heads * head_dim].
//
// Input data: [batch, num_kv_heads * head_dim] (row-major)
// Cache layout: [num_kv_heads, max_seq_len, head_dim] (head-first)

extern "C" __global__ void kv_cache_write_batch(
    float* __restrict__ cache,         // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ data,    // [batch, num_kv_heads * head_dim]
    unsigned int pos_start,
    unsigned int batch,
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kv_dim = num_kv_heads * head_dim;
    unsigned int total = batch * kv_dim;
    if (idx >= total) return;

    unsigned int token = idx / kv_dim;
    unsigned int within_token = idx % kv_dim;
    unsigned int head = within_token / head_dim;
    unsigned int dim_offset = within_token % head_dim;

    unsigned int pos = pos_start + token;
    unsigned int cache_idx = head * max_seq_len * head_dim + pos * head_dim + dim_offset;
    cache[cache_idx] = data[idx];
}

// ---------- Batched SwiGLU ----------
//
// SwiGLU(gate, up) = silu(gate[i]) * up[i], in-place on gate.
// Grid: 1D, one thread per element in [batch * inter_dim].

extern "C" __global__ void swiglu_batched(
    float* __restrict__ gate,      // [batch, inter_dim]
    const float* __restrict__ up,  // [batch, inter_dim]
    unsigned int total)            // batch * inter_dim
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    gate[idx] = silu_g * up[idx];
}

// ---------- Batched residual add ----------
//
// x[i] += residual[i] for all batch * dim elements.
// Grid: 1D, one thread per element.

extern "C" __global__ void residual_add_batched(
    float* __restrict__ x,             // [batch, dim]
    const float* __restrict__ residual, // [batch, dim]
    unsigned int total)                // batch * dim
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    x[idx] += residual[idx];
}

// ---------- Extract single row from batch matrix ----------
//
// Copies row `row_idx` from a [batch, dim] matrix to a [dim] vector.
// Grid: 1D, one thread per element.

extern "C" __global__ void extract_row(
    const float* __restrict__ matrix,  // [batch, dim]
    float* __restrict__ output,        // [dim]
    unsigned int row_idx,
    unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    output[idx] = matrix[(unsigned long long)row_idx * dim + idx];
}

// ---------- Scatter single row into batch matrix ----------
//
// Copies a [dim] vector into row `row_idx` of a [batch, dim] matrix.
// Grid: 1D, one thread per element.

extern "C" __global__ void scatter_row(
    float* __restrict__ matrix,        // [batch, dim]
    const float* __restrict__ input,   // [dim]
    unsigned int row_idx,
    unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    matrix[(unsigned long long)row_idx * dim + idx] = input[idx];
}
