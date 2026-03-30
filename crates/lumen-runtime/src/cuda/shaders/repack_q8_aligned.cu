// Repack Q8_0 blocks from 34 bytes to 36 bytes (aligned layout).
//
// Standard Q8_0 block (34 bytes):
//   bytes [0..1]:  f16 scale
//   bytes [2..33]: 32 x int8 quantized values
//
// Aligned Q8_0 block (36 bytes):
//   bytes [0..1]:  f16 scale (copied)
//   bytes [2..3]:  padding (zeroed)
//   bytes [4..35]: 32 x int8 quantized values (copied from offset +2)
//
// The padding ensures quant data at offset +4 is 4-byte aligned when the
// buffer start is 4-byte aligned (guaranteed by CUDA allocator). This allows
// the aligned dp4a kernel to use native int* loads instead of byte-level
// manual packing.
//
// Memory overhead: 36/34 = 1.059x (+5.9%)
// ALU savings: 8 int* loads vs 32 byte loads + 24 shift-or ops per block
//
// Grid:  (ceil(num_blocks / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void repack_q8_0_to_aligned36(
    const char* __restrict__ src,  // [num_blocks * 34] standard Q8_0 layout
    char* __restrict__ dst,        // [num_blocks * 36] aligned layout
    unsigned int num_blocks)
{
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const char* s = src + (unsigned long long)bid * 34;
    char* d = dst + (unsigned long long)bid * 36;

    // Copy scale (2 bytes, offset 0-1)
    d[0] = s[0];
    d[1] = s[1];

    // Padding (2 bytes, offset 2-3)
    d[2] = 0;
    d[3] = 0;

    // Copy 32 quant bytes from src offset +2 to dst offset +4.
    // Source at s+2 is NOT 4-byte aligned (34-byte blocks), so we use
    // byte-level copy. This runs once during preload, not on the hot path.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        d[4 + i] = s[2 + i];
    }
}
