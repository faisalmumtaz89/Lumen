// Q8_0 -> F16 dequantization kernel for batched prefill GEMM.
//
// Dequantizes a Q8_0 weight matrix [out_dim, in_dim] into a contiguous F16
// buffer [out_dim, in_dim] that can be fed to cuBLAS HGEMM. Each thread
// dequantizes one element: reads the block's F16 scale and the element's
// int8 quantized value, computes `scale * (float)quant`, then converts
// the result to F16.
//
// Q8_0 block layout (GGML): 34 bytes per block of 32 elements.
//   bytes [0..1]: f16 scale (IEEE 754 half-precision, little-endian)
//   bytes [2..33]: 32 x int8 quantized values
//   Dequant: float_val = scale * (float)(int8_t)quant[j]
//
// Grid:  (ceil(num_elements / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

// Hardware f16->f32 conversion via PTX (single instruction on SM 53+).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

// Hardware f32->f16 conversion via PTX (single instruction, round-to-nearest-even).
__device__ __forceinline__ unsigned short f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// Dequantize Q8_0 weight data to F16. Each thread handles one element.
//
// Parameters:
//   q8_data:      Raw Q8_0 bytes [num_blocks * 34]
//   f16_out:      Output F16 buffer [num_elements] (as unsigned short)
//   num_elements: Total number of dequantized elements (out_dim * in_dim)
extern "C" __global__ void dequant_q8_0_to_f16(
    const char* __restrict__ q8_data,
    unsigned short* __restrict__ f16_out,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Each Q8_0 block has 32 elements.
    unsigned int block_idx = idx >> 5;           // idx / 32
    unsigned int elem_in_block = idx & 31u;      // idx % 32

    // Block pointer: 34 bytes per block (2 byte scale + 32 byte data).
    const char* block_ptr = q8_data + (unsigned long long)block_idx * 34;

    // Read f16 scale from first 2 bytes (little-endian).
    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    // Read int8 quantized value and dequantize.
    float val = scale * (float)(signed char)block_ptr[2 + elem_in_block];

    // Convert to f16 and store.
    f16_out[idx] = f32_to_f16_bits(val);
}

// Dequantize Q8_0 weight data to F32. Each thread handles one element.
//
// Simpler alternative that dequantizes to F32 for the cuBLAS SGEMM path.
// Uses 2x the scratch memory compared to F16 but avoids F32<->F16 activation
// conversion overhead.
extern "C" __global__ void dequant_q8_0_to_f32(
    const char* __restrict__ q8_data,
    float* __restrict__ f32_out,
    unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    unsigned int block_idx = idx >> 5;
    unsigned int elem_in_block = idx & 31u;

    const char* block_ptr = q8_data + (unsigned long long)block_idx * 34;

    unsigned short scale_bits = (unsigned short)(unsigned char)block_ptr[0]
                              | ((unsigned short)(unsigned char)block_ptr[1] << 8);
    float scale = f16_bits_to_f32(scale_bits);

    f32_out[idx] = scale * (float)(signed char)block_ptr[2 + elem_in_block];
}
