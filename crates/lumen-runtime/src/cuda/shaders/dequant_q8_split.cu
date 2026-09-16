// ==========================================================================
// Q8_0 split-layout (SoA) dequant for the prefill: the F16 tile the HGEMM
// consumes and the F32 tile of the SGEMM fallback, from the per-row split
// layout `repack_q8_raw_to_split` produces. With it a Q8_0 plane needs no
// AoS copy once its split sibling exists: decode reads the split layout with
// `matvec_q8_split_q8_1`, prefill dequantizes it here, and the raw plane is
// released.
//
// Split row layout (nb = in_dim / 32 blocks per row, 34 * nb bytes per row):
//   bytes [0 .. 2 nb)      nb x f16 block scales
//   bytes [2 nb .. 34 nb)  nb x 32 int8 quants
//
// Element idx = row * in_dim + col dequantizes as
//   scale[row][col / 32] * (float) q[row][col]
// in exactly the arithmetic of `dequant_q8_0_to_f16` / `dequant_q8_0_to_f32`
// on the AoS plane (one f16 -> f32 convert, one f32 multiply, `cvt.rn` to f16),
// so the two tiles are bit-identical: the layout changes where a byte lives,
// never the value it dequantizes to.
//
// Grid: (ceil(num_elements / 256), 1, 1)   Block: (256, 1, 1)
// NVRTC-compatible: no system includes, extern "C" linkage.
// ==========================================================================

__device__ __forceinline__ float q8s_f16_bits_to_f32(unsigned short bits) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
    return result;
}

__device__ __forceinline__ unsigned short q8s_f32_to_f16_bits(float val) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(val));
    return result;
}

// The dequantized value of element `idx` of a [rows x in_dim] split-layout plane.
__device__ __forceinline__ float q8s_dequant_one(
    const char* __restrict__ split, unsigned int idx, unsigned int in_dim)
{
    const unsigned int row = idx / in_dim;
    const unsigned int col = idx - row * in_dim;
    const unsigned int nb = in_dim >> 5;
    const char* row_base = split + (unsigned long long)row * (unsigned long long)nb * 34ULL;
    const unsigned int block = col >> 5;
    const unsigned short scale_bits = (unsigned short)(unsigned char)row_base[2 * block]
                                    | ((unsigned short)(unsigned char)row_base[2 * block + 1] << 8);
    const float scale = q8s_f16_bits_to_f32(scale_bits);
    const signed char q = (signed char)row_base[2 * nb + col];
    return scale * (float)q;
}

extern "C" __global__ void dequant_q8_split_to_f16(
    const char* __restrict__ split,           // [rows * (in_dim / 32) * 34]
    unsigned short* __restrict__ f16_out,     // [rows * in_dim]
    unsigned int num_elements,                // rows * in_dim
    unsigned int in_dim)                      // multiple of 32
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    f16_out[idx] = q8s_f32_to_f16_bits(q8s_dequant_one(split, idx, in_dim));
}

extern "C" __global__ void dequant_q8_split_to_f32(
    const char* __restrict__ split,
    float* __restrict__ f32_out,
    unsigned int num_elements,
    unsigned int in_dim)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    f32_out[idx] = q8s_dequant_one(split, idx, in_dim);
}
