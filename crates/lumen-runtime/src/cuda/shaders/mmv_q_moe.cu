// dp4a-mmvq Q-quant MoE-batched decode matvec (Q8_0 weights × Q8_1
// activation × ids → F32 dst). c_rows_per_block=2 per warp tile.
//
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each WARP handles ONE token (one expert slot).
// No shared-memory reduction needed: each warp works alone with `warp_reduce_sum`.
//
// For Qwen3.5 MoE (top_k=8 experts active per token), in Lumen's decode (single
// token), the mapping is:
//   - ncols_dst = top_k = 8                    (8 experts processed in parallel)
//   - nchannels_dst = 1                        (single token)
//   - ids= expert_ids(top_k expert indices)
//   - nrows_x = inter_dim or hidden_dim        (rows of weight matrix)
//   - stride_channel_x = bytes per expert weight matrix
//
// Lumen's existing path uses `(gate_offsets, up_offsets, down_offsets)` arrays
// to locate per-expert weight slices inside `layer_buf`. The linear-stride
// kernel form uses `ids` + `stride_channel_x`. We adapt to use Lumen's offset
// arrays, since each expert's weight is at an arbitrary byte offset (not
// necessarily linear stride).
//
// Variants implemented:
//   1. mmv_q_moe_gate_up_swiglu_q8_0 - fused gate + up matvec + SwiGLU (replaces v3)
//   2. mmv_q_moe_down_q8_0           - down matvec (replaces v3 down)
//   3. mmv_q_moe_gate_up_swiglu_q4_0 - Q4_0 variant
//   4. mmv_q_moe_down_q4_0           - Q4_0 variant
//
// Fused gate+up+SwiGLU: each warp does BOTH gate and up dot products for the
// same row-tile in a single pass (re-using the per-block Q8_1 activation read).
// This is the fused-dense-matvec pattern adapted to the MoE-batched topology.
//
// NVRTC-compatible: no system includes, inline PTX for dp4a.
// All kernels use sm_61+ inline PTX `dp4a.s32.s32` (works on A100/sm_80 at
// full HW throughput).

typedef signed char         int8_t;
typedef unsigned char       uint8_t;
typedef int                 int32_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

#define WARP_SIZE 32
#define QK8_0 32
#define QK8_1 32
#define QK4_0 32
#define QI8_0 8
#define QI8_1 8
#define QI4_0 4
#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMVQ 2

// Per-MoE-warp parameter: c_rows_per_block=2 (best perf via tuning).
#define C_ROWS_PER_BLOCK 2

// HW fp16<->f32 conversion (PTX cvt single-instruction on sm_53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short f32_to_f16_bits(float v) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(v));
    return bits;
}

// SIMD dp4a (4-element signed int8 dot product + accumulate).
__device__ __forceinline__ int dp4a_s(int a, int b, int c) {
    int r;
    asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Warp-level sum-reduction (5 butterfly shuffles).
__device__ __forceinline__ float warp_reduce_sum_moe(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_moe(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;
}

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================================
// quantize_q8_1_moe — per-block Q8_1 quantizer.
//
// Same as's `quantize_q8_1_rawsum` (used for the dense matvec path);
// duplicated here to keep mmv_q_moe.cu standalone-compilable for NVRTC.
//
// Input:  normed_x [hidden_dim]    F32 activation (RMSNorm-output)
// Output: vy        [num_blocks*36] block_q8_1 quantized blocks
//
// Each block (32 floats) reduced to: {f16 d, f16 sum_F32, int8 qs[32]}.
//
// CRITICAL: ds.y = raw F32 sum (NOT d * sum_qs as Lumen's legacy
// `quantize_f32_to_q8_1` produces). The Q4_0 vec_dot bias correction
// expects this raw-sum convention.
//
// Grid: (ceil(in_dim / QK8_1), 1, 1) -- one CTA per 32-elem block
// Block: (QK8_1, 1, 1) -- 32 threads = 1 warp
// ============================================================================
extern "C" __global__ void quantize_q8_1_moe(
    const float* __restrict__ x,
    unsigned char* __restrict__ vy,
    unsigned int in_dim)
{
    const unsigned int i0  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int ib  = i0 / QK8_1;
    const unsigned int iqs = i0 % QK8_1;

    const float xi = (i0 < in_dim) ? x[i0] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max_moe(amax);
    sum  = warp_reduce_sum_moe(sum);

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)__float2int_rn(xi / d);

    unsigned char* yb = vy + ib * 36;
    yb[4 + iqs] = (unsigned char)q;

    if (iqs == 0) {
        const unsigned short d_bits = f32_to_f16_bits(d);
        const unsigned short s_bits = f32_to_f16_bits(sum);
        yb[0] = (unsigned char)(d_bits & 0xFF);
        yb[1] = (unsigned char)((d_bits >> 8) & 0xFF);
        yb[2] = (unsigned char)(s_bits & 0xFF);
        yb[3] = (unsigned char)((s_bits >> 8) & 0xFF);
    }
}

// ============================================================================
// Inline vec_dot implementations (byte-identical to).
// ============================================================================

__device__ __forceinline__ float vec_dot_q8_0_q8_1_impl_moe(
    const int8_t* __restrict__ bq8_0_qs,
    const int8_t* __restrict__ bq8_1_qs,
    int iqs,
    float d8_0, float d8_1)
{
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        const int8_t* vp = bq8_0_qs + (iqs + i) * 4;
        const int8_t* up = bq8_1_qs + (iqs + i) * 4;
        int v = ((int)(unsigned char)vp[0])
              | ((int)(unsigned char)vp[1] << 8)
              | ((int)(unsigned char)vp[2] << 16)
              | ((int)(unsigned char)vp[3] << 24);
        int u = ((int)(unsigned char)up[0])
              | ((int)(unsigned char)up[1] << 8)
              | ((int)(unsigned char)up[2] << 16)
              | ((int)(unsigned char)up[3] << 24);
        sumi = dp4a_s(v, u, sumi);
    }
    return d8_0 * d8_1 * (float)sumi;
}

__device__ __forceinline__ float vec_dot_q4_0_q8_1_impl_moe(
    const uint8_t* __restrict__ bq4_0_qs,
    const int8_t* __restrict__ bq8_1_qs,
    int iqs,
    float d4, float ds8_x, float ds8_y)
{
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        const uint8_t* vp = bq4_0_qs + (iqs + i) * 4;
        int v = ((int)vp[0])
              | ((int)vp[1] << 8)
              | ((int)vp[2] << 16)
              | ((int)vp[3] << 24);
        const int vi0 = (v >> 0) & 0x0F0F0F0F;
        const int vi1 = (v >> 4) & 0x0F0F0F0F;

        const int8_t* up0 = bq8_1_qs + (iqs + i) * 4;
        const int8_t* up1 = bq8_1_qs + (iqs + i + QI4_0) * 4;
        int u0 = ((int)(unsigned char)up0[0])
               | ((int)(unsigned char)up0[1] << 8)
               | ((int)(unsigned char)up0[2] << 16)
               | ((int)(unsigned char)up0[3] << 24);
        int u1 = ((int)(unsigned char)up1[0])
               | ((int)(unsigned char)up1[1] << 8)
               | ((int)(unsigned char)up1[2] << 16)
               | ((int)(unsigned char)up1[3] << 24);

        sumi = dp4a_s(vi0, u0, sumi);
        sumi = dp4a_s(vi1, u1, sumi);
    }
    return d4 * ((float)sumi * ds8_x - 4.0f * ds8_y);
}

// ============================================================================
// mmv_q_moe_gate_up_swiglu_q8_0 — FUSED gate matvec + up matvec + SwiGLU.
//
// Decomposition (pre-fusion):
//   - 1 launch of mul_mat_vec_q_moe<Q8_0, 2> for gate_proj
//   - 1 launch of mul_mat_vec_q_moe<Q8_0, 2> for up_proj
//   - 1 element-wise SiLU(gate) * up step
// We fuse all 3 into ONE kernel for lower launch overhead and shared per-block
// Q8_1 activation reads. This is the dense `has_fusion=true` pattern applied
// to the MoE-batched topology.
//
// Grid: (ceil(inter_dim / C_ROWS_PER_BLOCK), 1, 1)     [nchannels_dst=1 for decode]
// Block: (WARP_SIZE, top_k, 1)                        [each WARP = 1 expert slot]
//
// Each warp:
//   - reads expert_ids[k] -> resolves gate_off, up_off in layer_buf
//   - computes gate dot products for c_rows rows
//   - computes up dot products for c_rows rows
//   - writes swiglu_buf[k*inter_dim + r] = silu(gate[r]) * up[r] for c_rows
//
// Total dispatch ops: (inter_dim/2) * top_k * 32 = (inter_dim/2) * 256 threads.
// vs Lumen v3: (inter_dim/4) * top_k * 256 = (inter_dim/4) * 2048 threads.
// dp4a path: 8x fewer threads but each does the FULL matvec (dp4a), no
// cross-warp reduction needed.
// ============================================================================
extern "C" __global__ void mmv_q_moe_gate_up_swiglu_q8_0(
    const unsigned char* __restrict__ vy,           // [num_blocks*36] Q8_1 normed_x
    const unsigned char* __restrict__ layer_buf,    // raw byte blob
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ gate_up_offsets, // [num_experts*2]
    float* __restrict__ swiglu_buf,                 // [top_k*inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = threadIdx.y;             // expert slot index
    if (k >= top_k) return;

    const unsigned int row0 = C_ROWS_PER_BLOCK * blockIdx.x;
    if (row0 >= inter_dim) return;

    const unsigned int lane = threadIdx.x;

    // Resolve per-expert weight offsets.
    const unsigned int expert_id = expert_ids[k];
    const uint64_t gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const uint64_t up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const int blocks_per_row_x = hidden_dim / QK8_0;
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * WARP_SIZE / QI8_0;  // 2*32/8 = 8

    // Per-warp lane partition:
    //   kbx = lane / (qi/vdr) = lane / 4 = [0..8)
    //   kqs = vdr * (lane % (qi/vdr)) = 2 * (lane % 4) = {0,2,4,6}
    const int kbx0 = lane / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
    const int kqs  = VDR_Q8_0_Q8_1_MMVQ * (lane % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));

    // Row bytes for one weight matrix row.
    const uint64_t row_bytes = (uint64_t)blocks_per_row_x * 34;

    // c_rows partial sums.
    float gate_sum[C_ROWS_PER_BLOCK] = {0.0f};
    float up_sum[C_ROWS_PER_BLOCK]   = {0.0f};

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        // Load Q8_1 activation block (shared across all rows).
        const unsigned char* bq8_1 = vy + (uint64_t)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const float d8_1 = f16_bits_to_f32(d8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r >= inter_dim) continue;

            // Gate row at (row0+r): byte offset
            const unsigned char* bq8_0_gate = layer_buf + gate_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 34;
            const unsigned short d_gate_bits =
                (unsigned short)bq8_0_gate[0] | ((unsigned short)bq8_0_gate[1] << 8);
            const float d_gate = f16_bits_to_f32(d_gate_bits);
            const int8_t* qs_gate = (const int8_t*)(bq8_0_gate + 2);

            gate_sum[r] += vec_dot_q8_0_q8_1_impl_moe(qs_gate, bq8_1_qs, kqs, d_gate, d8_1);

            // Up row at (row0+r)
            const unsigned char* bq8_0_up = layer_buf + up_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 34;
            const unsigned short d_up_bits =
                (unsigned short)bq8_0_up[0] | ((unsigned short)bq8_0_up[1] << 8);
            const float d_up = f16_bits_to_f32(d_up_bits);
            const int8_t* qs_up = (const int8_t*)(bq8_0_up + 2);

            up_sum[r] += vec_dot_q8_0_q8_1_impl_moe(qs_up, bq8_1_qs, kqs, d_up, d8_1);
        }
    }

    // Warp reduction (per row).
    #pragma unroll
    for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
        gate_sum[r] = warp_reduce_sum_moe(gate_sum[r]);
        up_sum[r]   = warp_reduce_sum_moe(up_sum[r]);
    }

    // Lane 0 of each warp writes SiLU(gate) * up.
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r < inter_dim) {
                const float g = gate_sum[r];
                const float u = up_sum[r];
                swiglu_buf[(size_t)k * (size_t)inter_dim + (row0 + r)] = silu(g) * u;
            }
        }
    }
}

// ============================================================================
// mmv_q_moe_down_q8_0 — down matvec (Q8_0 weights × Q8_1 swiglu × ids → F32 dst).
//
// Input:  swiglu_buf [top_k*inter_dim]      F32 (must be quantized to Q8_1 first
//                                              via quantize_q8_1_moe_swiglu)
// Output: down_out   [top_k*hidden_dim]     F32
//
// Grid: (ceil(hidden_dim / 2), 1, 1)
// Block: (WARP_SIZE, top_k, 1)
//
// Each warp computes 2 rows of the down matrix for one expert slot.
// ============================================================================
extern "C" __global__ void mmv_q_moe_down_q8_0(
    const unsigned char* __restrict__ vy,           // [top_k * num_blocks * 36] Q8_1 swiglu
    const unsigned char* __restrict__ layer_buf,
    const unsigned int* __restrict__ expert_ids,    // [top_k]
    const unsigned long long* __restrict__ down_offsets, // [num_experts]
    float* __restrict__ down_out,                   // [top_k * hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = threadIdx.y;
    if (k >= top_k) return;

    const unsigned int row0 = C_ROWS_PER_BLOCK * blockIdx.x;
    if (row0 >= hidden_dim) return;

    const unsigned int lane = threadIdx.x;
    const unsigned int expert_id = expert_ids[k];
    const uint64_t down_off = down_offsets[expert_id];

    const int blocks_per_row_x = inter_dim / QK8_0;
    constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * WARP_SIZE / QI8_0;

    const int kbx0 = lane / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
    const int kqs  = VDR_Q8_0_Q8_1_MMVQ * (lane % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));

    const uint64_t row_bytes = (uint64_t)blocks_per_row_x * 34;

    // Pointer to per-expert Q8_1 activation slice.
    const unsigned char* vy_k = vy + (uint64_t)k * (uint64_t)blocks_per_row_x * 36;

    float tmp[C_ROWS_PER_BLOCK] = {0.0f};

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const unsigned char* bq8_1 = vy_k + (uint64_t)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const float d8_1 = f16_bits_to_f32(d8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r >= hidden_dim) continue;

            const unsigned char* bq8_0 = layer_buf + down_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 34;
            const unsigned short d_bits =
                (unsigned short)bq8_0[0] | ((unsigned short)bq8_0[1] << 8);
            const float d8_0 = f16_bits_to_f32(d_bits);
            const int8_t* bq8_0_qs = (const int8_t*)(bq8_0 + 2);

            tmp[r] += vec_dot_q8_0_q8_1_impl_moe(bq8_0_qs, bq8_1_qs, kqs, d8_0, d8_1);
        }
    }

    #pragma unroll
    for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
        tmp[r] = warp_reduce_sum_moe(tmp[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r < hidden_dim) {
                down_out[(size_t)k * (size_t)hidden_dim + (row0 + r)] = tmp[r];
            }
        }
    }
}

// ============================================================================
// quantize_q8_1_moe_swiglu — quantize per-expert swiglu_buf to Q8_1.
//
// Input:  swiglu_buf [top_k * inter_dim]        F32
// Output: vy         [top_k * num_blocks * 36]  block_q8_1
//
// Grid:  (ceil(inter_dim / QK8_1), top_k, 1)
// Block: (QK8_1, 1, 1)  = 32 threads = 1 warp
// ============================================================================
extern "C" __global__ void quantize_q8_1_moe_swiglu(
    const float* __restrict__ swiglu_buf,
    unsigned char* __restrict__ vy,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = blockIdx.y;
    if (k >= top_k) return;

    const unsigned int i0  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int ib_layer = i0 / QK8_1;
    const unsigned int iqs = i0 % QK8_1;

    const unsigned int blocks_per_layer = (inter_dim + QK8_1 - 1) / QK8_1;

    const float xi = (i0 < inter_dim) ? swiglu_buf[(size_t)k * (size_t)inter_dim + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max_moe(amax);
    sum  = warp_reduce_sum_moe(sum);

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)__float2int_rn(xi / d);

    unsigned char* yb = vy + (size_t)k * (size_t)blocks_per_layer * 36 + (size_t)ib_layer * 36;
    yb[4 + iqs] = (unsigned char)q;

    if (iqs == 0) {
        const unsigned short d_bits = f32_to_f16_bits(d);
        const unsigned short s_bits = f32_to_f16_bits(sum);
        yb[0] = (unsigned char)(d_bits & 0xFF);
        yb[1] = (unsigned char)((d_bits >> 8) & 0xFF);
        yb[2] = (unsigned char)(s_bits & 0xFF);
        yb[3] = (unsigned char)((s_bits >> 8) & 0xFF);
    }
}

// ============================================================================
// mmv_q_moe_gate_up_swiglu_q4_0 — Q4_0 variant.
// ============================================================================
extern "C" __global__ void mmv_q_moe_gate_up_swiglu_q4_0(
    const unsigned char* __restrict__ vy,           // [num_blocks*36] Q8_1 normed_x
    const unsigned char* __restrict__ layer_buf,
    const unsigned int* __restrict__ expert_ids,
    const unsigned long long* __restrict__ gate_up_offsets,
    float* __restrict__ swiglu_buf,
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = threadIdx.y;
    if (k >= top_k) return;

    const unsigned int row0 = C_ROWS_PER_BLOCK * blockIdx.x;
    if (row0 >= inter_dim) return;

    const unsigned int lane = threadIdx.x;
    const unsigned int expert_id = expert_ids[k];
    const uint64_t gate_off = gate_up_offsets[(size_t)expert_id * 2 + 0];
    const uint64_t up_off   = gate_up_offsets[(size_t)expert_id * 2 + 1];

    const int blocks_per_row_x = hidden_dim / QK4_0;
    // For Q4_0: vdr*WARP_SIZE/qi = 2*32/4 = 16 blocks per warp iter.
    constexpr int blocks_per_iter = VDR_Q4_0_Q8_1_MMVQ * WARP_SIZE / QI4_0;  // 16

    // For Q4_0: kbx = lane / (qi/vdr) = lane / 2 = [0..16); kqs = 2*(lane % 2) = {0,2}.
    const int kbx0 = lane / (QI4_0 / VDR_Q4_0_Q8_1_MMVQ);
    const int kqs  = VDR_Q4_0_Q8_1_MMVQ * (lane % (QI4_0 / VDR_Q4_0_Q8_1_MMVQ));

    // Q4_0 row bytes = num_blocks * 18.
    const uint64_t row_bytes = (uint64_t)blocks_per_row_x * 18;

    float gate_sum[C_ROWS_PER_BLOCK] = {0.0f};
    float up_sum[C_ROWS_PER_BLOCK]   = {0.0f};

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const unsigned char* bq8_1 = vy + (uint64_t)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const unsigned short s8_1_bits =
            (unsigned short)bq8_1[2] | ((unsigned short)bq8_1[3] << 8);
        const float ds8_x = f16_bits_to_f32(d8_1_bits);
        const float ds8_y = f16_bits_to_f32(s8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r >= inter_dim) continue;

            // Gate row
            const unsigned char* bq4_0_gate = layer_buf + gate_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 18;
            const unsigned short d_gate_bits =
                (unsigned short)bq4_0_gate[0] | ((unsigned short)bq4_0_gate[1] << 8);
            const float d_gate = f16_bits_to_f32(d_gate_bits);
            const uint8_t* qs_gate = bq4_0_gate + 2;

            gate_sum[r] += vec_dot_q4_0_q8_1_impl_moe(qs_gate, bq8_1_qs, kqs, d_gate, ds8_x, ds8_y);

            // Up row
            const unsigned char* bq4_0_up = layer_buf + up_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 18;
            const unsigned short d_up_bits =
                (unsigned short)bq4_0_up[0] | ((unsigned short)bq4_0_up[1] << 8);
            const float d_up = f16_bits_to_f32(d_up_bits);
            const uint8_t* qs_up = bq4_0_up + 2;

            up_sum[r] += vec_dot_q4_0_q8_1_impl_moe(qs_up, bq8_1_qs, kqs, d_up, ds8_x, ds8_y);
        }
    }

    #pragma unroll
    for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
        gate_sum[r] = warp_reduce_sum_moe(gate_sum[r]);
        up_sum[r]   = warp_reduce_sum_moe(up_sum[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r < inter_dim) {
                const float g = gate_sum[r];
                const float u = up_sum[r];
                swiglu_buf[(size_t)k * (size_t)inter_dim + (row0 + r)] = silu(g) * u;
            }
        }
    }
}

// ============================================================================
// mmv_q_moe_down_q4_0 — Q4_0 down variant.
// ============================================================================
extern "C" __global__ void mmv_q_moe_down_q4_0(
    const unsigned char* __restrict__ vy,
    const unsigned char* __restrict__ layer_buf,
    const unsigned int* __restrict__ expert_ids,
    const unsigned long long* __restrict__ down_offsets,
    float* __restrict__ down_out,
    unsigned int hidden_dim,
    unsigned int inter_dim,
    unsigned int top_k)
{
    const unsigned int k = threadIdx.y;
    if (k >= top_k) return;

    const unsigned int row0 = C_ROWS_PER_BLOCK * blockIdx.x;
    if (row0 >= hidden_dim) return;

    const unsigned int lane = threadIdx.x;
    const unsigned int expert_id = expert_ids[k];
    const uint64_t down_off = down_offsets[expert_id];

    const int blocks_per_row_x = inter_dim / QK4_0;
    constexpr int blocks_per_iter = VDR_Q4_0_Q8_1_MMVQ * WARP_SIZE / QI4_0;

    const int kbx0 = lane / (QI4_0 / VDR_Q4_0_Q8_1_MMVQ);
    const int kqs  = VDR_Q4_0_Q8_1_MMVQ * (lane % (QI4_0 / VDR_Q4_0_Q8_1_MMVQ));

    const uint64_t row_bytes = (uint64_t)blocks_per_row_x * 18;
    const unsigned char* vy_k = vy + (uint64_t)k * (uint64_t)blocks_per_row_x * 36;

    float tmp[C_ROWS_PER_BLOCK] = {0.0f};

    for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const unsigned char* bq8_1 = vy_k + (uint64_t)kbx * 36;
        const unsigned short d8_1_bits =
            (unsigned short)bq8_1[0] | ((unsigned short)bq8_1[1] << 8);
        const unsigned short s8_1_bits =
            (unsigned short)bq8_1[2] | ((unsigned short)bq8_1[3] << 8);
        const float ds8_x = f16_bits_to_f32(d8_1_bits);
        const float ds8_y = f16_bits_to_f32(s8_1_bits);
        const int8_t* bq8_1_qs = (const int8_t*)(bq8_1 + 4);

        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r >= hidden_dim) continue;

            const unsigned char* bq4_0 = layer_buf + down_off
                + (uint64_t)(row0 + r) * row_bytes + (uint64_t)kbx * 18;
            const unsigned short d_bits =
                (unsigned short)bq4_0[0] | ((unsigned short)bq4_0[1] << 8);
            const float d4 = f16_bits_to_f32(d_bits);
            const uint8_t* qs = bq4_0 + 2;

            tmp[r] += vec_dot_q4_0_q8_1_impl_moe(qs, bq8_1_qs, kqs, d4, ds8_x, ds8_y);
        }
    }

    #pragma unroll
    for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
        tmp[r] = warp_reduce_sum_moe(tmp[r]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < C_ROWS_PER_BLOCK; ++r) {
            if (row0 + r < hidden_dim) {
                down_out[(size_t)k * (size_t)hidden_dim + (row0 + r)] = tmp[r];
            }
        }
    }
}
