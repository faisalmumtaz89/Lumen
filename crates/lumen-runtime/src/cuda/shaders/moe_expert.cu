// MoE per-expert FFN kernel.
//
// Single-expert variant that processes one expert per launch — the default
// dispatch path. Mirrors the existing CUDA `fused_glu_gemv_q8_0` but
// reads weights from a per-layer raw byte blob at runtime-computed offsets:
//
//   gate_weight = layer_buf + gate_off    (row-major [inter_dim, hidden_dim] Q8_0)
//   up_weight   = layer_buf + up_off      (row-major [inter_dim, hidden_dim] Q8_0)
//   down_weight = layer_buf + down_off    (row-major [hidden_dim, inter_dim] Q8_0)
//
// Per-expert dispatch contract (Phase B ):
//   - K kernel launches per MoE FFN per token (K = top_k, typically 8).
//   - Each launch processes one (expert, token) pair.
//   - Output slot is `output_buf + k * hidden_dim * 4` (dense top-K layout).
//
// Two kernels:
//   - moe_expert_gate_up_swiglu_q8_0: gate · x, up · x, SwiGLU → swiglu_out
//   - moe_expert_down_q8_0:           down · swiglu_out → expert_out_slot
//
// Both are bandwidth-bound on A100 (Q8_0 ≈ 1.06 B/elem); compute is
// dominated by HBM reads of `layer_buf` from the expert's weight rows.
//
// Numerical correctness: matches the existing per-tensor `fused_glu_gemv_q8_0`
// kernel mathematically (same accumulator order, same SwiGLU formulation).

// NVRTC-compatible: inline PTX for f16->f32, no cuda_fp16.h. Matches the
// pattern used in attention_f16.cu / dequant_q8_0_f16.cu / hgemv_q8_0.cu.
// (cudarc::nvrtc::compile_ptx is invoked without --include-path, so system
// headers like <cuda_fp16.h> are unreachable; the fix was wrong.)

#define BLOCK_DIM 128
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34   // 2 bytes F16 scale + 32 bytes int8 quants

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float me_f16_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ float me_load_q8_scale(const unsigned char* block_ptr) {
    unsigned short f16_bits = *reinterpret_cast<const unsigned short*>(block_ptr);
    return me_f16_to_f32(f16_bits);
}

__device__ __forceinline__ float me_swiglu(float g, float u) {
    float silu_g = g / (1.0f + expf(-g));
    return silu_g * u;
}

// Per-expert gate + up + SwiGLU.
//
// Inputs:
//   normed_x       [hidden_dim] F32
//   layer_buf      raw byte blob (Q8_0 weights at gate_off and up_off)
//   gate_off       byte offset of this expert's gate weight within layer_buf
//   up_off         byte offset of this expert's up weight within layer_buf
//
// Output:
//   swiglu_out     [inter_dim] F32 = silu(gate · x) * (up · x), one row per thread
extern "C" __global__ void moe_expert_gate_up_swiglu_q8_0(
    const float* __restrict__ normed_x,             // [hidden_dim]
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long gate_off,
    unsigned long long up_off,
    float* __restrict__ swiglu_out,                 // [inter_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (row >= inter_dim) return;

    const unsigned int blocks_per_row = hidden_dim / Q8_0_BLOCK_SIZE;
    const size_t row_stride = (size_t)blocks_per_row * Q8_0_BLOCK_BYTES;

    const unsigned char* gate_row = layer_buf + gate_off + (size_t)row * row_stride;
    const unsigned char* up_row   = layer_buf + up_off   + (size_t)row * row_stride;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* gblk = gate_row + (size_t)b * Q8_0_BLOCK_BYTES;
        const unsigned char* ublk = up_row   + (size_t)b * Q8_0_BLOCK_BYTES;
        float gscale = me_load_q8_scale(gblk);
        float uscale = me_load_q8_scale(ublk);
        const signed char* gquants = reinterpret_cast<const signed char*>(gblk + 2);
        const signed char* uquants = reinterpret_cast<const signed char*>(ublk + 2);
        for (unsigned int e = 0; e < Q8_0_BLOCK_SIZE; ++e) {
            float xv = normed_x[(size_t)b * Q8_0_BLOCK_SIZE + e];
            gate_acc += ((float)gquants[e] * gscale) * xv;
            up_acc   += ((float)uquants[e] * uscale) * xv;
        }
    }
    swiglu_out[row] = me_swiglu(gate_acc, up_acc);
}

// Per-expert down projection.
//
// Inputs:
//   swiglu_in      [inter_dim] F32 (output of gate+up+SwiGLU)
//   layer_buf      raw byte blob
//   down_off       byte offset of this expert's down weight within layer_buf
//
// Output:
//   expert_out     [hidden_dim] F32 = down · swiglu_in
//
// Caller writes the output to a dense slot (slot k = `k * hidden_dim` floats
// inside the expert_output_buf scratch).
extern "C" __global__ void moe_expert_down_q8_0(
    const float* __restrict__ swiglu_in,            // [inter_dim]
    const unsigned char* __restrict__ layer_buf,    // raw bytes
    unsigned long long down_off,
    float* __restrict__ expert_out,                 // [hidden_dim]
    unsigned int hidden_dim,
    unsigned int inter_dim)
{
    const unsigned int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (row >= hidden_dim) return;

    const unsigned int blocks_per_row = inter_dim / Q8_0_BLOCK_SIZE;
    const size_t row_stride = (size_t)blocks_per_row * Q8_0_BLOCK_BYTES;

    const unsigned char* down_row = layer_buf + down_off + (size_t)row * row_stride;

    float dot = 0.0f;
    for (unsigned int b = 0; b < blocks_per_row; ++b) {
        const unsigned char* blk = down_row + (size_t)b * Q8_0_BLOCK_BYTES;
        float scale = me_load_q8_scale(blk);
        const signed char* quants = reinterpret_cast<const signed char*>(blk + 2);
        for (unsigned int e = 0; e < Q8_0_BLOCK_SIZE; ++e) {
            dot += ((float)quants[e] * scale) * swiglu_in[(size_t)b * Q8_0_BLOCK_SIZE + e];
        }
    }
    expert_out[row] = dot;
}
