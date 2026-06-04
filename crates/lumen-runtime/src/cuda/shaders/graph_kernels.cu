// Graph-compatible kernel variants for CUDA graph capture/replay.
//
// These kernels read per-token-varying scalar parameters from GPU memory
// (device pointers) instead of kernel scalar arguments. This makes ALL kernel
// launch parameters (grid_dim, block_dim, shared_mem_bytes, pointer args)
// identical across tokens, enabling CUDA graph capture once and replay many times.
//
// The per-token parameters are stored in a small GPU buffer (GraphParamsBuf)
// that is updated via a single htod memcpy before graph replay. The memcpy
// itself happens OUTSIDE the graph (before launch), so the graph sees fixed
// device pointers that point to updated values.
//
// Cost: 1 extra global memory load per thread per kernel that needs a varying
// scalar. At ~100 bytes/cycle L1 cache bandwidth, this is negligible (<1 ns).

// ============================================================================
// Embedding lookup -- reads token_id from device pointer
// ============================================================================

extern "C" __global__ void embed_token_f32_graph(
    const float* __restrict__ embedding_table,
    float* __restrict__ output,
    const unsigned int* __restrict__ p_token_id,  // device pointer to token_id
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        unsigned int token_id = *p_token_id;
        output[idx] = embedding_table[token_id * hidden_dim + idx];
    }
}

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

extern "C" __global__ void embed_token_q8_0_graph(
    const char* __restrict__ embedding_q8,
    float* __restrict__ output,
    const unsigned int* __restrict__ p_token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    unsigned int token_id = *p_token_id;
    unsigned int global_elem = token_id * hidden_dim + idx;
    unsigned int block_idx = global_elem >> 5;
    unsigned int elem_in_block = global_elem & 31u;

    const char* block_ptr = embedding_q8 + block_idx * 34;
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_to_f32(scale_bits);
    float val = (float)(signed char)block_ptr[2 + elem_in_block];
    output[idx] = val * scale;
}

extern "C" __global__ void embed_token_f16_graph(
    const unsigned short* __restrict__ embedding_f16,
    float* __restrict__ output,
    const unsigned int* __restrict__ p_token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        unsigned int token_id = *p_token_id;
        output[idx] = f16_to_f32(embedding_f16[(unsigned long long)token_id * hidden_dim + idx]);
    }
}

extern "C" __global__ void embed_token_q4_0_graph(
    const char* __restrict__ embedding_q4,
    float* __restrict__ output,
    const unsigned int* __restrict__ p_token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    unsigned int token_id = *p_token_id;
    unsigned int global_elem = (unsigned long long)token_id * hidden_dim + idx;
    unsigned int block_idx = global_elem >> 5;
    unsigned int elem_in_block = global_elem & 31u;

    const char* block_ptr = embedding_q4 + block_idx * 18;
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_to_f32(scale_bits);

    // De-interleaved Q4_0: elements 0-15 = lo nibbles, 16-31 = hi nibbles.
    unsigned int byte_idx = (elem_in_block < 16u) ? elem_in_block : (elem_in_block - 16u);
    unsigned char byte_val = (unsigned char)block_ptr[2 + byte_idx];
    unsigned int nibble = (elem_in_block < 16u) ? (byte_val & 0x0Fu) : ((byte_val >> 4) & 0x0Fu);
    output[idx] = scale * ((float)nibble - 8.0f);
}

// ============================================================================
// RoPE -- reads pos from device pointer
// ============================================================================

extern "C" __global__ void rope_apply_graph(
    float* __restrict__ q,
    float* __restrict__ k,
    const unsigned int* __restrict__ p_pos,  // device pointer to position
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float theta_base,
    unsigned int rotary_dim)                 // 0 = full head_dim (backward compatible)
{
    // Actual rotary dimension: 0 means full head_dim (backward compatible)
    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;
    unsigned int total_q_pairs = num_q_heads * half_rot;
    unsigned int total_k_pairs = num_kv_heads * half_rot;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int pos = *p_pos;

    if (idx < total_q_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    if (idx < total_k_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float x0 = k[i0];
        float x1 = k[i1];
        k[i0] = x0 * cos_a - x1 * sin_a;
        k[i1] = x0 * sin_a + x1 * cos_a;
    }
}

// ============================================================================
// KV cache write -- reads pos from device pointer
// ============================================================================

extern "C" __global__ void kv_cache_write_graph(
    float* __restrict__ cache,
    const float* __restrict__ data,
    const unsigned int* __restrict__ p_pos,  // device pointer to position
    unsigned int num_kv_heads,
    unsigned int max_seq_len,
    unsigned int head_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    unsigned int pos = *p_pos;
    unsigned int head = idx / head_dim;
    unsigned int dim_offset = idx % head_dim;

    unsigned int cache_idx = head * max_seq_len * head_dim + pos * head_dim + dim_offset;
    cache[cache_idx] = data[idx];
}

// ============================================================================
// Fused RoPE + KV cache write -- reads pos from device pointer
// ============================================================================
//
// Combines rope_apply_graph and kv_cache_write_graph (x2) into a single kernel.
// Eliminates 2 kernel launches per layer in the graph decode pipeline.
//
// Thread mapping: 1 thread per RoPE pair (= half_dim elements per head).
//   total_threads = max(num_q_heads, num_kv_heads) * (head_dim / 2)
//
// Each thread handles:
//   - Q RoPE: rotate pair (q[2d], q[2d+1]) for its Q head (if idx < q_pairs)
//   - K RoPE + K cache write: rotate pair (k[2d], k[2d+1]) and write both
//     elements to k_cache (if idx < k_pairs). No cross-thread data dependency
//     because the SAME thread writes and reads the pair.
//   - V cache write: write v[2d] and v[2d+1] to v_cache (if idx < k_pairs).
//
// Correctness: each thread writes RoPE'd K values to the kbuffer in registers,
// then immediately writes them to k_cache. No other thread reads these K elements,
// so no grid-wide synchronization is needed.

extern "C" __global__ void rope_kv_write_graph(
    float* __restrict__ q,                       // [num_q_heads * head_dim] (in-place RoPE)
    float* __restrict__ k,                       // [num_kv_heads * head_dim] (in-place RoPE + cache write)
    const float* __restrict__ v,                 // [num_kv_heads * head_dim] (cache write only)
    float* __restrict__ k_cache,                 // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ v_cache,                 // [num_kv_heads, max_seq_len, head_dim]
    const unsigned int* __restrict__ p_pos,      // device pointer to position
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int max_seq_len,
    float theta_base,
    unsigned int rotary_dim)                     // 0 = full head_dim (backward compatible)
{
    // Actual rotary dimension: 0 means full head_dim (backward compatible)
    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;
    unsigned int half_dim = head_dim >> 1;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pos = *p_pos;

    // --- Q RoPE ---
    // Thread mapping: half_rot pairs per head for RoPE, half_dim pairs per head for V cache.
    // Use max(half_rot, half_dim) pairs in the grid for K+V cache writes.
    unsigned int total_q_rot_pairs = num_q_heads * half_rot;
    if (idx < total_q_rot_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;

        // Freq base uses actual_rot for correct frequency spacing
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // --- K RoPE + KV cache write ---
    // Each thread handles one pair of elements. For d < half_rot, apply RoPE to K
    // and write both rotated K and V to cache. For d >= half_rot (partial RoPE),
    // K is unrotated -- just copy K and V to cache.
    unsigned int total_kv_pairs = num_kv_heads * half_dim;
    if (idx < total_kv_pairs) {
        unsigned int d = idx % half_dim;
        unsigned int kv_head = idx / half_dim;
        unsigned int head_offset = kv_head * head_dim;

        unsigned int i0 = head_offset + 2 * d;
        unsigned int i1 = i0 + 1;
        float k0 = k[i0];
        float k1 = k[i1];

        // Apply RoPE rotation only to the first half_rot pairs
        if (d < half_rot) {
            float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0_rot = k0 * cos_a - k1 * sin_a;
            float k1_rot = k0 * sin_a + k1 * cos_a;
            k[i0] = k0_rot;
            k[i1] = k1_rot;
            k0 = k0_rot;
            k1 = k1_rot;
        }

        // Write K (rotated or pass-through) and V to cache
        unsigned int cache_base = kv_head * max_seq_len * head_dim + pos * head_dim;
        k_cache[cache_base + 2 * d]     = k0;
        k_cache[cache_base + 2 * d + 1] = k1;

        // Write V to cache (no RoPE, just scatter to head-first layout)
        v_cache[cache_base + 2 * d]     = v[i0];
        v_cache[cache_base + 2 * d + 1] = v[i1];
    }
}

// NeoX-style fused RoPE + KV cache write for graph decode (Qwen2/Qwen3.5).
// Pairs at (d, d+half_rot) instead of (2d, 2d+1).
extern "C" __global__ void rope_kv_write_neox_graph(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const unsigned int* __restrict__ p_pos,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int max_seq_len,
    float theta_base,
    unsigned int rotary_dim)
{
    unsigned int actual_rot = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;
    unsigned int half_rot = actual_rot >> 1;
    unsigned int half_dim = head_dim >> 1;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pos = *p_pos;

    // Q RoPE (NeoX half-split)
    unsigned int total_q_rot_pairs = num_q_heads * half_rot;
    if (idx < total_q_rot_pairs) {
        unsigned int d = idx % half_rot;
        unsigned int head_offset = (idx / half_rot) * head_dim;
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        unsigned int i0 = head_offset + d;
        unsigned int i1 = head_offset + d + half_rot;
        float x0 = q[i0];
        float x1 = q[i1];
        q[i0] = x0 * cos_a - x1 * sin_a;
        q[i1] = x0 * sin_a + x1 * cos_a;
    }

    // K RoPE + KV cache write (NeoX)
    unsigned int total_kv_pairs = num_kv_heads * half_dim;
    if (idx < total_kv_pairs) {
        unsigned int d = idx % half_dim;
        unsigned int kv_head = idx / half_dim;
        unsigned int head_offset = kv_head * head_dim;
        unsigned int i0 = head_offset + d;
        unsigned int i1 = head_offset + d + half_dim;
        float k0 = k[i0];
        float k1 = k[i1];

        if (d < half_rot) {
            float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)actual_rot);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float k0_rot = k0 * cos_a - k1 * sin_a;
            float k1_rot = k0 * sin_a + k1 * cos_a;
            k[i0] = k0_rot;
            k[i1] = k1_rot;
            k0 = k0_rot;
            k1 = k1_rot;
        }

        unsigned int cache_base = kv_head * max_seq_len * head_dim + pos * head_dim;
        k_cache[cache_base + d] = k0;
        k_cache[cache_base + d + half_dim] = k1;
        v_cache[cache_base + d] = v[i0];
        v_cache[cache_base + d + half_dim] = v[i1];
    }
}

// ============================================================================
// Fused F32->F16 conversion + residual copy for HGEMV output projection.
// ============================================================================
//
// Combines the two operations needed before HGEMV with beta=1.0:
//   1. Convert attn_out (F32) to F16 for the HGEMV input
//   2. Copy residual to output buffer for beta=1.0 accumulation
//
// Without this kernel: memcpy_dtod + f32_to_f16_vec = 2 dispatches.
// With this kernel: 1 dispatch.
//
// Thread mapping: 1 thread per element (max of out_dim and in_dim).
// Both operations are independent and embarrassingly parallel.

// Hardware f32->f16 conversion via PTX.
__device__ __forceinline__ unsigned short graph_f32_to_f16(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

extern "C" __global__ void convert_f32_to_f16_and_residual_copy(
    const float* __restrict__ input_f32,     // [in_dim] source for F16 conversion
    unsigned short* __restrict__ output_f16,  // [in_dim] F16 output
    const float* __restrict__ residual,       // [out_dim] source for residual copy
    float* __restrict__ output_f32,           // [out_dim] destination for residual
    unsigned int in_dim,
    unsigned int out_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert input to F16 (for HGEMV input).
    if (idx < in_dim) {
        output_f16[idx] = graph_f32_to_f16(input_f32[idx]);
    }

    // Copy residual to output buffer (for HGEMV beta=1.0 accumulation).
    if (idx < out_dim) {
        output_f32[idx] = residual[idx];
    }
}

// ============================================================================
// Attention decode -- reads seq_len from device pointer, uses FIXED geometry
// ============================================================================
//
// Key difference from attention.cu: the block_dim is always MAX_ATTN_BLOCK (256)
// and shared_mem_bytes is always (8 + max_seq_len) * 4. Extra threads (tid >= seq_len)
// participate in reductions with identity values (max=-inf, sum=0) but do no
// real work. This allows the kernel's launch configuration to be completely
// static, enabling CUDA graph capture.

// Warp-level reductions (same as attention.cu)
__device__ __forceinline__ float graph_warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float graph_warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ float graph_block_reduce_max(
    float val, volatile float* partial, unsigned int tid, unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = graph_warp_reduce_max(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : -3.402823466e+38f;
        v = graph_warp_reduce_max(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float graph_block_reduce_sum(
    float val, volatile float* partial, unsigned int tid, unsigned int block_size)
{
    unsigned int lane = tid & 31u;
    unsigned int warp_id = tid >> 5;
    unsigned int num_warps = (block_size + 31u) >> 5;

    val = graph_warp_reduce_sum(val);
    if (lane == 0u) partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0u) {
        float v = (lane < num_warps) ? partial[lane] : 0.0f;
        v = graph_warp_reduce_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

extern "C" __global__ void attention_decode_graph(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ attn_out,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    const unsigned int* __restrict__ p_seq_len,  // device pointer to current seq_len
    unsigned int max_seq_len,
    float scale)
{
    unsigned int head = blockIdx.x;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // Read dynamic seq_len from device memory.
    unsigned int seq_len = *p_seq_len;

    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    const float* q_head = q + head * head_dim;
    float* out_head = attn_out + head * head_dim;
    unsigned long long kv_base = (unsigned long long)kv_h * max_seq_len * head_dim;

    // Shared memory: [0..7] partial reduction, [8..8+max_seq_len-1] scores.
    // Allocated at MAX size for graph compatibility. Only [8..8+seq_len-1] used.
    extern __shared__ float smem[];
    volatile float* partial = smem;
    float* scores = smem + 8;

    // Phase 1: QK dot products with float4 vectorized loads.
    // head_dim is always a multiple of 4 (64, 128, 256 in all standard models).
    // float4 reduces instruction count 4x and uses 128-bit memory transactions.
    unsigned int head_dim_vec = head_dim >> 2;
    const float4* q_head_v = reinterpret_cast<const float4*>(q_head);

    for (unsigned int t = tid; t < seq_len; t += block_size) {
        const float4* k_vec_v = reinterpret_cast<const float4*>(k_cache + kv_base + (unsigned long long)t * head_dim);
        float dot = 0.0f;
        for (unsigned int d4 = 0; d4 < head_dim_vec; d4++) {
            float4 qv = q_head_v[d4];
            float4 kv = k_vec_v[d4];
            dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Numerically stable softmax
    float local_max = -3.402823466e+38f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        local_max = fmaxf(local_max, scores[t]);
    }
    float global_max = graph_block_reduce_max(local_max, partial, tid, block_size);

    float local_sum = 0.0f;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        float e = expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    float global_sum = graph_block_reduce_sum(local_sum, partial, tid, block_size);

    float inv_sum = 1.0f / global_sum;
    for (unsigned int t = tid; t < seq_len; t += block_size) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted V accumulation with float4 vectorized loads.
    // Each thread accumulates 4 contiguous dimensions per iteration,
    // reducing instruction count 4x and using 128-bit V cache reads.
    for (unsigned int d4 = tid; d4 < head_dim_vec; d4 += block_size) {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int t = 0; t < seq_len; t++) {
            float s = scores[t];
            float4 vv = reinterpret_cast<const float4*>(v_cache + kv_base + (unsigned long long)t * head_dim)[d4];
            acc.x += s * vv.x;
            acc.y += s * vv.y;
            acc.z += s * vv.z;
            acc.w += s * vv.w;
        }
        reinterpret_cast<float4*>(out_head)[d4] = acc;
    }
}

// ============================================================================
// GDN Conv1D decode -- reads state_pos from device pointer (graph-compatible)
// ============================================================================
//
// Identical to ssm_conv1d_decode in gdn.cu, except state_pos is read from
// a device pointer instead of a scalar argument. This allows CUDA graph
// capture since the pointer is baked in but its value can change.
//
// Grid: 1D, ceil(conv_dim / 256) blocks of 256 threads

extern "C" __global__ void ssm_conv1d_decode_graph(
    float* __restrict__ conv_state,   // [buf_slots, conv_dim] circular buffer R/W
    const float* __restrict__ input,  // [conv_dim] new token values
    const float* __restrict__ weight, // [conv_dim, kernel_size] convolution weights
    float* __restrict__ output,       // [conv_dim] convolved output
    unsigned int conv_dim,
    unsigned int kernel_size,
    const unsigned int* __restrict__ p_state_pos)  // device pointer to state_pos
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= conv_dim) return;

    unsigned int state_pos = *p_state_pos;
    float sum = 0.0f;
    unsigned int buf_slots = kernel_size - 1;

    // Taps 0..kernel_size-2: read from circular buffer (oldest to newest)
    for (unsigned int tap = 0; tap < buf_slots; tap++) {
        unsigned int slot = (state_pos + tap) % buf_slots;
        sum += weight[gid * kernel_size + tap] * conv_state[slot * conv_dim + gid];
    }

    // Tap kernel_size-1: current input (newest)
    sum += weight[gid * kernel_size + buf_slots] * input[gid];

    output[gid] = sum;

    // Update circular buffer: overwrite oldest entry (at state_pos) with current input
    conv_state[state_pos * conv_dim + gid] = input[gid];
}

// ============================================================================
// Advance conv position -- increments GPU-resident conv_pos for graph capture
// ============================================================================
//
// Single-thread kernel that advances the circular buffer write position.
// Dispatched AFTER ssm_conv1d_decode_graph for each GDN layer inside the
// captured graph. The position update happens on-GPU, so no host<->device
// sync is needed during graph replay.
//
// Grid: (1, 1, 1), Block: (1, 1, 1)

extern "C" __global__ void advance_conv_position(
    unsigned int* __restrict__ conv_pos,
    unsigned int buf_slots)
{
    *conv_pos = (*conv_pos + 1) % buf_slots;
}

// ============================================================================
// Tiled streaming-softmax decode attention -- graph variant
// ============================================================================
//
// Identical mathematics to `attention_decode_tiled` in `attention_decode_tiled.cu`,
// except `seq_len` is read from a device pointer (`p_seq_len`) instead of a host
// scalar. This enables CUDA graph capture for long-context decode: the captured
// graph bakes in the device pointer, while the value pointed to changes per
// token (updated via `memcpy_htod` BEFORE graph replay -- same mechanism as
// `attention_decode_graph` above).
//
// Shared memory layout (constant in seq_len -- the whole point of the tiled
// kernel): partial[8] + q_row[head_dim] + s_tile[T_C] = (8 + head_dim + T_C) *
// 4 bytes. At head_dim=256 + T_C=128: ~1.6 KB. Static across all decode tokens.
// Unlike `attention_decode_graph`'s O(max_seq_len) shmem, this one needs no
// special opt-in for extended dynamic shmem.
//
// Grid: (num_heads, 1, 1), Block: (TILED_BLOCK_DIM=128, 1, 1).
// Invariant: head_dim must be a multiple of TILED_BLOCK_DIM (128). Production
// Qwen3.5-9B has head_dim=256 (256 % 128 == 0). Smaller test models that fail
// this guard fall back to the host-scalar variant in eager mode (the wrapper
// at backend_impl.rs picks the variant before capture).
//
// Algorithm reference: Dao 2022 online softmax (FlashAttention-2 decode shape).
// Tile width T_C = 128, fixed at compile time (same as host kernel).

#define TILED_T_C        128u
#define TILED_BLOCK_DIM  128u
#define TILED_NEG_INF    (-3.402823466e+38f)

// Inline Q@K dot-product helper (float4 vectorised when head_dim % 4 == 0).
// Same algorithm as `tiled_qk_dot` in attention_decode_tiled.cu, inlined here
// to avoid the shared-translation-unit prefix-name collision with the host
// kernel's helpers.
__device__ __forceinline__ float tiled_graph_qk_dot(
    const float* __restrict__ q_row,
    const float* __restrict__ k_vec,
    unsigned int head_dim,
    float scale)
{
    float dot = 0.0f;
    if ((head_dim & 3u) == 0u) {
        unsigned int hd4 = head_dim >> 2;
        const float4* q4 = reinterpret_cast<const float4*>(q_row);
        const float4* k4 = reinterpret_cast<const float4*>(k_vec);
        for (unsigned int d4 = 0; d4 < hd4; d4++) {
            float4 q = q4[d4];
            float4 k = k4[d4];
            dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
        }
    } else {
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_vec[d];
        }
    }
    return dot * scale;
}

extern "C" __global__ void attention_decode_tiled_graph(
    const float* __restrict__ q,           // [num_heads * head_dim]
    const float* __restrict__ k_cache,     // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ v_cache,     // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ attn_out,          // [num_heads * head_dim]
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    const unsigned int* __restrict__ p_seq_len, // device pointer to current seq_len
    unsigned int max_seq_len,
    float scale)
{
    unsigned int head = blockIdx.x;
    if (head >= num_heads) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;   // = TILED_BLOCK_DIM at launch

    // Read dynamic seq_len from device memory (the only per-token-varying input).
    unsigned int seq_len = *p_seq_len;

    // Degenerate seq_len = 0: zero output and exit. Defensive only -- production
    // decode always has seq_len >= 1 because KV is appended BEFORE attn call.
    if (seq_len == 0u) {
        for (unsigned int d = tid; d < head_dim; d += block_size) {
            attn_out[head * head_dim + d] = 0.0f;
        }
        return;
    }

    // GQA mapping: multiple Q heads share the same KV head.
    unsigned int gqa_ratio = num_heads / num_kv_heads;
    unsigned int kv_h = head / gqa_ratio;

    const float* q_head = q + head * head_dim;
    float* out_head = attn_out + head * head_dim;

    unsigned long long kv_base = (unsigned long long)kv_h
                                 * (unsigned long long)max_seq_len
                                 * (unsigned long long)head_dim;

    // Shared memory layout (constant in seq_len):
    //   [0..7]:           partial[8] -- warp-reduction scratch
    //   [8..8+head_dim):  q_row[head_dim]
    //   [8+head_dim..]:   s_tile[T_C]
    extern __shared__ float smem[];
    volatile float* partial = smem;
    float* q_row = smem + 8;
    float* s_tile = smem + 8 + head_dim;

    // Phase 0: cooperative Q row load.
    for (unsigned int d = tid; d < head_dim; d += block_size) {
        q_row[d] = q_head[d];
    }
    __syncthreads();

    // Per-thread running softmax state (registers across tiles).
    float m_prev = TILED_NEG_INF;
    float l_prev = 0.0f;

    constexpr unsigned int MAX_SLOTS = 8u;
    float o_acc[MAX_SLOTS];
#pragma unroll
    for (unsigned int s = 0; s < MAX_SLOTS; s++) o_acc[s] = 0.0f;
    unsigned int num_slots = (head_dim + block_size - 1u) / block_size;

    unsigned int num_tiles = (seq_len + TILED_T_C - 1u) / TILED_T_C;

    for (unsigned int tile = 0; tile < num_tiles; tile++) {
        unsigned int tile_start = tile * TILED_T_C;
        unsigned int tile_end_raw = tile_start + TILED_T_C;
        unsigned int tile_end = (tile_end_raw < seq_len) ? tile_end_raw : seq_len;
        unsigned int tile_len = tile_end - tile_start;

        // Phase A: scores for this tile (one lane per position when
        // T_C == block_size; strided otherwise).
        for (unsigned int j = tid; j < TILED_T_C; j += block_size) {
            if (j < tile_len) {
                unsigned int pos = tile_start + j;
                const float* k_vec = k_cache + kv_base
                                     + (unsigned long long)pos
                                     * (unsigned long long)head_dim;
                s_tile[j] = tiled_graph_qk_dot(q_row, k_vec, head_dim, scale);
            } else {
                s_tile[j] = TILED_NEG_INF;
            }
        }
        __syncthreads();

        // Phase B: tile max + online-softmax update.
        float local_max = TILED_NEG_INF;
        for (unsigned int j = tid; j < TILED_T_C; j += block_size) {
            local_max = fmaxf(local_max, s_tile[j]);
        }
        float tile_max = graph_block_reduce_max(local_max, partial, tid, block_size);

        float m_new = fmaxf(m_prev, tile_max);
        float rescale = expf(m_prev - m_new);

        // Phase C: exp(s - m_new) into s_tile + tile_sum reduction.
        for (unsigned int j = tid; j < TILED_T_C; j += block_size) {
            if (j < tile_len) {
                s_tile[j] = expf(s_tile[j] - m_new);
            } else {
                s_tile[j] = 0.0f;
            }
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (unsigned int j = tid; j < TILED_T_C; j += block_size) {
            local_sum += s_tile[j];
        }
        float tile_sum = graph_block_reduce_sum(local_sum, partial, tid, block_size);

        float l_new = rescale * l_prev + tile_sum;

        // Phase D: rescale O_prev and accumulate P @ V_tile (scalar V loads).
#pragma unroll
        for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
            if (slot >= num_slots) break;
            unsigned int d = tid + slot * block_size;
            if (d < head_dim) {
                float pv = 0.0f;
                for (unsigned int j = 0; j < tile_len; j++) {
                    unsigned int pos = tile_start + j;
                    float v_dj = v_cache[kv_base
                                         + (unsigned long long)pos
                                         * (unsigned long long)head_dim
                                         + (unsigned long long)d];
                    pv += s_tile[j] * v_dj;
                }
                o_acc[slot] = rescale * o_acc[slot] + pv;
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        __syncthreads(); // next tile overwrites s_tile in Phase A.
    }

    // Final: normalise and write output.
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < MAX_SLOTS; slot++) {
        if (slot >= num_slots) break;
        unsigned int d = tid + slot * block_size;
        if (d < head_dim) {
            out_head[d] = o_acc[slot] * inv_l;
        }
    }
}
