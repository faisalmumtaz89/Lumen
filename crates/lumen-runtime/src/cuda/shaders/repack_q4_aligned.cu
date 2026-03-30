// Repack Q4_0 blocks from 18 bytes to 20 bytes (aligned layout).
//
// Standard Q4_0 block (18 bytes):
//   bytes [0..1]:   f16 scale
//   bytes [2..17]:  16 bytes of packed nibble pairs (32 elements)
//
// Aligned Q4_0 block (20 bytes):
//   bytes [0..1]:   f16 scale (copied)
//   bytes [2..3]:   padding (zeroed)
//   bytes [4..19]:  16 bytes of packed nibble pairs (copied from offset +2)
//
// The padding ensures nibble data at offset +4 is 4-byte aligned when the
// buffer start is 4-byte aligned (guaranteed by CUDA allocator). This allows
// the aligned dp4a kernel to use native int* loads (4 instructions loading
// 4 nibble bytes each) instead of 16 individual byte loads per block.
//
// Memory overhead: 20/18 = 1.111x (+11.1%)
// Load savings: 4 int* loads vs 16 byte loads per block
//
// Grid:  (ceil(num_blocks / 256), 1, 1)
// Block: (256, 1, 1)
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void repack_q4_0_to_aligned20(
    const char* __restrict__ src,  // [num_blocks * 18] standard Q4_0 layout
    char* __restrict__ dst,        // [num_blocks * 20] aligned layout
    unsigned int num_blocks)
{
    unsigned int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;

    const char* s = src + (unsigned long long)bid * 18;
    char* d = dst + (unsigned long long)bid * 20;

    // Copy scale (2 bytes, offset 0-1)
    d[0] = s[0];
    d[1] = s[1];

    // Padding (2 bytes, offset 2-3)
    d[2] = 0;
    d[3] = 0;

    // Copy 16 nibble bytes from src offset +2 to dst offset +4.
    // Source at s+2 is NOT 4-byte aligned (18-byte blocks), so we use
    // byte-level copy. This runs once during preload, not on the hot path.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        d[4 + i] = s[2 + i];
    }
}
