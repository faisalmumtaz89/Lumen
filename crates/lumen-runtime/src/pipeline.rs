//! Pipeline scheduler types (PIPO-inspired task graph).
//!
//! The pipeline breaks inference into fine-grained tasks that can overlap:
//! weight loading, KV cache loading, compute, KV cache saving, and eviction.

/// Pipeline execution mode controlling the tradeoff between memory usage
/// and compute/I/O overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineMode {
    /// Prefetch aggressively, keep multiple layers in cache for
    /// maximum compute/I/O overlap. Higher peak RAM usage.
    Perf,

    /// Only one layer's weights resident at a time. Minimizes RAM
    /// footprint at the cost of potential pipeline bubbles.
    MinMem,

    /// Auto-select based on measured I/O and compute throughput.
    Auto,
}

impl Default for PipelineMode {
    fn default() -> Self {
        Self::Auto
    }
}

