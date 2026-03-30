// Token embedding lookup kernels for CUDA.
//
// F32: direct row copy from embedding table.
// Q8_0: dequantize Q8_0 block on the fly (34 bytes = f16 scale + 32 int8).

extern "C" __global__ void embed_token_f32(
    const float* __restrict__ embedding_table,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = embedding_table[token_id * hidden_dim + idx];
    }
}

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
// Replaces ~15 ALU software bit-manipulation with the native CVT instruction.
// NVRTC-compatible: inline PTX requires no headers or include paths.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
// Dequantized value = scale * (float)quant[elem_in_block]
extern "C" __global__ void embed_token_q8_0(
    const char* __restrict__ embedding_q8,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    unsigned int global_elem = token_id * hidden_dim + idx;
    unsigned int block_idx = global_elem >> 5;       // / 32
    unsigned int elem_in_block = global_elem & 31u;  // % 32

    const char* block_ptr = embedding_q8 + block_idx * 34;

    // Read f16 scale (little-endian) and convert to f32.
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    float val = (float)(signed char)block_ptr[2 + elem_in_block];
    output[idx] = val * scale;
}

// F16 embedding: each element is 2-byte IEEE 754 half-precision.
// Dequantizes to f32 on the fly via f16_bits_to_f32.
extern "C" __global__ void embed_token_f16(
    const unsigned short* __restrict__ embedding_f16,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = f16_bits_to_f32(embedding_f16[(unsigned long long)token_id * hidden_dim + idx]);
    }
}

// Q4_0 embedding: 18 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (LE)
//   bytes [2..17]: 16 bytes of nibble pairs (2 x 4-bit values per byte)
//   Dequant: val = scale * ((float)nibble - 8.0f)
extern "C" __global__ void embed_token_q4_0(
    const char* __restrict__ embedding_q4,
    float* __restrict__ output,
    unsigned int token_id,
    unsigned int hidden_dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    unsigned int global_elem = (unsigned long long)token_id * hidden_dim + idx;
    unsigned int block_idx = global_elem >> 5;
    unsigned int elem_in_block = global_elem & 31u;

    const char* block_ptr = embedding_q4 + block_idx * 18;

    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    unsigned int byte_idx = elem_in_block >> 1;
    unsigned char byte_val = (unsigned char)block_ptr[2 + byte_idx];
    unsigned int nibble = (elem_in_block & 1u) ? (byte_val >> 4) : (byte_val & 0x0Fu);
    output[idx] = scale * ((float)nibble - 8.0f);
}
