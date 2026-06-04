// MoE per-expert FFN kernels — Q4_0 variant.
//
// Sibling of moe_expert.cu (Q8_0). Same dispatch contract (one launch per
// (expert, token) pair), same algebra (gate · x, up · x, SwiGLU, down · swig),
// only the weight-format unpacking differs.
//
// Q4_0 block layout (GGML standard, 18 bytes per 32 elements):
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..17]: 16 bytes = 32 × 4-bit unsigned values packed as nibble pairs
//     GGML de-interleaved layout: lo nibble of byte b = element b
//                                 hi nibble of byte b = element b+16
//   Dequantize: float_value = scale * ((float)(nibble) - 8.0f)
//
// Bandwidth-floor on A100 is ~0.5625 B/elem (vs 1.0625 for Q8_0) — 1.89× faster
// in the limit; in practice the gain is smaller due to compute pressure from
// the nibble unpack.
//
// NVRTC-compatible: inline PTX for f16->f32, no cuda_fp16.h.
// Mirrors the helper functions used in moe_expert.cu and matvec_q4_0.cu.

#define BLOCK_DIM_Q4 128
#define Q4_0_BLOCK_ELEMS 32
#define Q4_0_BLOCK_BYTES 18   // 2 bytes F16 scale + 16 bytes nibble-packed

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Duplicated per .cu file (NVRTC compiles each as a separate module).
__device__ __forceinline__ float meq4_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Load Q4_0 block scale (2-byte F16 at start of block).
__device__ __forceinline__ float meq4_load_scale(const unsigned char* block_ptr) {
    // Read two bytes little-endian and convert F16 → F32.
    unsigned short f16_bits = (unsigned short)block_ptr[0]
                            | ((unsigned short)block_ptr[1] << 8);
    return meq4_f16_to_f32(f16_bits);
}

__device__ __forceinline__ float meq4_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// Per-expert gate + up + SwiGLU (Q4_0 variant).
//
// Inputs:
//   normed_x       [hidden_dim] F32
//   layer_buf      raw byte blob (Q4_0 weights at gate_off and up_off)
//   gate_off       byte offset of this expert's gate weight within layer_buf
//   up_off         byte offset of this expert's up weight within layer_buf
//
// Output:
//   swiglu_out     [inter_dim] F32 = silu(gate · x) * (up · x), one row per thread
//
// Grid: gridDim.x = ceil(inter_dim / BLOCK_DIM_Q4). Each thread handles one row.
extern "C" __global__ void moe_expert_gate_up_swiglu_q4_0(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long gate_off,
    unsigned long long up_off,
    float* __restrict__ swiglu_out,                 // [inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * BLOCK_DIM_Q4 + threadIdx.x;
    if (row >= inter_dim) return;

    const unsigned int blocks_per_row = hidden_dim / Q4_0_BLOCK_ELEMS;
    const size_t row_stride = (size_t)blocks_per_row * Q4_0_BLOCK_BYTES;

    const unsigned char* gate_row = layer_buf + gate_off + (size_t)row * row_stride;
    const unsigned char* up_row   = layer_buf + up_off   + (size_t)row * row_stride;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gblk = gate_row + (size_t)b * Q4_0_BLOCK_BYTES;
        const unsigned char* ublk = up_row   + (size_t)b * Q4_0_BLOCK_BYTES;
        float gscale = meq4_load_scale(gblk);
        float uscale = meq4_load_scale(ublk);
        const unsigned char* gq = gblk + 2;
        const unsigned char* uq = ublk + 2;

        // GGML de-interleaved nibbles: lo of byte i = elem i, hi = elem i+16.
        // Mirrors matvec_q4_0.cu:108-118 exactly.
        const unsigned int x_base = b * Q4_0_BLOCK_ELEMS;
        float g_block_sum = 0.0f;
        float u_block_sum = 0.0f;
        for (unsigned int i = 0; i < 16; ++i) {
            unsigned char gb = gq[i];
            unsigned char ub = uq[i];
            float gq_lo = (float)(gb & 0x0F) - 8.0f;
            float gq_hi = (float)(gb >> 4)   - 8.0f;
            float uq_lo = (float)(ub & 0x0F) - 8.0f;
            float uq_hi = (float)(ub >> 4)   - 8.0f;
            float xlo = normed_x[x_base + i];
            float xhi = normed_x[x_base + i + 16];
            g_block_sum += gq_lo * xlo + gq_hi * xhi;
            u_block_sum += uq_lo * xlo + uq_hi * xhi;
        }
        gate_acc += gscale * g_block_sum;
        up_acc   += uscale * u_block_sum;
    }
    swiglu_out[row] = meq4_swiglu(gate_acc, up_acc);
}

// Per-expert down projection (Q4_0 variant).
//
// Inputs:
//   swiglu_in      [inter_dim] F32
//   layer_buf      raw byte blob
//   down_off       byte offset of this expert's down weight within layer_buf
//
// Output:
//   expert_out     [hidden_dim] F32 = down · swiglu_in
extern "C" __global__ void moe_expert_down_q4_0(
    const float* __restrict__ swiglu_in,            // [inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long down_off,
    float* __restrict__ expert_out,                 // [hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * BLOCK_DIM_Q4 + threadIdx.x;
    if (row >= hidden_dim) return;

    const unsigned int blocks_per_row = inter_dim / Q4_0_BLOCK_ELEMS;
    const size_t row_stride = (size_t)blocks_per_row * Q4_0_BLOCK_BYTES;

    const unsigned char* down_row = layer_buf + down_off + (size_t)row * row_stride;

    float dot = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = down_row + (size_t)b * Q4_0_BLOCK_BYTES;
        float scale = meq4_load_scale(blk);
        const unsigned char* qs = blk + 2;
        const unsigned int x_base = b * Q4_0_BLOCK_ELEMS;
        float block_sum = 0.0f;
        for (unsigned int i = 0; i < 16; ++i) {
            unsigned char by = qs[i];
            float q_lo = (float)(by & 0x0F) - 8.0f;
            float q_hi = (float)(by >> 4)   - 8.0f;
            block_sum += q_lo * swiglu_in[x_base + i]
                       + q_hi * swiglu_in[x_base + i + 16];
        }
        dot += scale * block_sum;
    }
    expert_out[row] = dot;
}
