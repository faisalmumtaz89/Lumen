// Validation-only: a launch that does nothing measurable.
//
// Measures the MARGINAL cost of a kernel launch on THIS build in THIS decode
// context, by injecting N of them per token and reading the slope. The 4.2 us
// figure the campaign has been quoting came from removing 24 launches in the
// conv fusion (+0.10 ms) — a single point, extrapolated across ~395 launches
// to claim ~1.66 ms of launch overhead. Marginal is not average, and launches
// overlap with real work, so that estimate probably OVERSTATES the recoverable
// gap. A slope across several N settles it.
//
// Writes one byte to a scratch buffer so the launch cannot be elided.
extern "C" __global__ void noop_probe(unsigned char* __restrict__ sink) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sink[0] = (unsigned char)(sink[0] + 1u);
    }
}
