// Q4_0 -> F16 dequantization kernel for prefill HGEMM.
//
// Dequantizes a Q4_0 weight matrix into contiguous F16 buffer
// for cuBLAS HGEMM tensor core dispatch.
//
// Q4_0 block layout (GGML): 18 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..17]: 16 bytes = 32 x 4-bit unsigned nibbles packed as pairs
//   Dequant: float_val = scale * ((float)nibble - 8.0f)
//
// Grid: (ceil(num_elements / 256), 1, 1)
// Block: (256, 1, 1)

// Hardware f16<->f32 conversion via PTX.
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

extern "C" __global__ void dequant_q4_0_to_f16(
    const char* __restrict__ q4_data,
    unsigned short* __restrict__ f16_out,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    unsigned int block_idx = idx >> 5;           // idx / 32
    unsigned int elem_in_block = idx & 31u;      // idx % 32

    const char* block_ptr = q4_data + (unsigned long long)block_idx * 18;

    // Read f16 scale (little-endian).
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    // Extract nibble.
    unsigned int byte_idx = elem_in_block >> 1;
    unsigned char byte_val = (unsigned char)block_ptr[2 + byte_idx];
    unsigned int nibble = (elem_in_block & 1u) ? (byte_val >> 4) : (byte_val & 0x0Fu);

    float val = scale * ((float)nibble - 8.0f);
    f16_out[idx] = f32_to_f16_bits(val);
}

// Q4_0 -> F32 dequantization kernel for prefill SGEMM.
//
// Dequantizes a Q4_0 weight matrix into contiguous F32 buffer
// for cuBLAS SGEMM dispatch. Simpler alternative to F16 path that
// avoids F32<->F16 activation conversion overhead.
//
// Grid: (ceil(num_elements / 256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void dequant_q4_0_to_f32(
    const char* __restrict__ q4_data,
    float* __restrict__ f32_out,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    unsigned int block_idx = idx >> 5;           // idx / 32
    unsigned int elem_in_block = idx & 31u;      // idx % 32

    const char* block_ptr = q4_data + (unsigned long long)block_idx * 18;

    // Read f16 scale (little-endian).
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    // Extract nibble.
    unsigned int byte_idx = elem_in_block >> 1;
    unsigned char byte_val = (unsigned char)block_ptr[2 + byte_idx];
    unsigned int nibble = (elem_in_block & 1u) ? (byte_val >> 4) : (byte_val & 0x0Fu);

    f32_out[idx] = scale * ((float)nibble - 8.0f);
}
