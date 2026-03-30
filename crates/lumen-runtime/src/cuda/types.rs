//! Shared type definitions for the CUDA backend.

/// CUDA launch configuration for a 1D kernel dispatch.
pub(crate) struct LaunchConfig {
    /// Number of thread blocks (grid dimension).
    pub(crate) grid_dim: u32,
    /// Number of threads per block.
    pub(crate) block_dim: u32,
}

impl LaunchConfig {
    /// Compute a 1D launch configuration for `num_elements` work items.
    ///
    /// Uses 256 threads per block (standard for memory-bound kernels on NVIDIA GPUs).
    /// Grid dimension is ceil(num_elements / block_dim).
    pub(crate) fn for_elements(num_elements: usize) -> Self {
        let block_dim = 256u32;
        let grid_dim = (num_elements as u32).div_ceil(block_dim);
        Self { grid_dim, block_dim }
    }
}
