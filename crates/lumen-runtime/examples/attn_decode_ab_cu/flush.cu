// Cache-eviction helper for the harness's "evicted" measurement mode: a
// trivial pass over a buffer sized above the device L2, so the ~9 MB K/V
// working set is not resident when the next attention launch starts.
//
// Write-only: a read-modify-write would double the flush's own cost without
// evicting anything extra; the stores allocate in L2 and displace K/V.
extern "C" __global__ void gqa6_l2_flush(float* __restrict__ buf, unsigned int n, float val) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        buf[i] = val;
    }
}
