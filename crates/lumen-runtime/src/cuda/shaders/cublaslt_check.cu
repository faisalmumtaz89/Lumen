// Verification and timing helpers for the native prefill's cuBLASLt plans (cublaslt_algo_cache.rs).
//
// A selected algorithm's BF16 output D is checked, element by element, against an F32 reference R and
// a magnitude M computed by cuBLAS SGEMMs of the decoded operands (M uses their absolute values):
//
//   |D - R| <= 2^-8 * max(|R|, |D|) + 2^-17 * M
//
// The first term is BF16's rounding of the F32 result: half an ulp of an 8-bit significand is at most
// 2^-8 relative (at the bottom of a binade), measured against the larger of the two values; the second
// covers the difference between two F32 summation orders over the same products, 128 units of 2^-24
// relative to the sum of their magnitudes. A NaN or infinite D is a violation, so an element the GEMM
// never wrote (the buffer is filled with NaN first) or an overflow is caught.
//
// NVRTC-compatible: no system includes, extern "C" linkage.

extern "C" __global__ void lumen_lt_abs(float* __restrict__ x, unsigned long long n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fabsf(x[i]);
    }
}

// D is BF16 [rows][ldd]; R and M are F32 [rows][n]. counts[0] += violations; worst holds the largest
// finite |D - R| / bound as float bits (non-negative floats order like their bits).
extern "C" __global__ void lumen_lt_check(
    const unsigned short* __restrict__ d,
    unsigned int ldd,
    const float* __restrict__ r,
    const float* __restrict__ m,
    unsigned int rows,
    unsigned int n,
    unsigned int* __restrict__ counts,
    unsigned int* __restrict__ worst)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)rows * n) {
        return;
    }
    unsigned int row = (unsigned int)(i / n);
    unsigned int col = (unsigned int)(i % n);
    float dev = __uint_as_float(((unsigned int)d[(unsigned long long)row * ldd + col]) << 16);
    float ref = r[i];
    float bound = ldexpf(fmaxf(fabsf(ref), fabsf(dev)), -8) + ldexpf(m[i], -17);
    float err = fabsf(dev - ref);
    if (!(err <= bound) || !(fabsf(dev) <= 3.40282347e38f)) {
        atomicAdd(&counts[0], 1u);
    } else if (bound > 0.0f) {
        atomicMax(worst, __float_as_uint(err / bound));
    }
}

// Occupies the stream for `ns` nanoseconds of the global timer, so work queued behind it is enqueued
// by the host before the GPU reaches it.
extern "C" __global__ void lumen_lt_spin(unsigned long long ns)
{
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= ns) {
            break;
        }
    }
}
