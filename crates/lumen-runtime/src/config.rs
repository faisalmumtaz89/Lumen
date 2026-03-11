//! Runtime configuration types.
//!
//! Defines the parameters that control the inference engine's behavior:
//! pipeline mode, prefetch distance, KV precision, and telemetry toggles.

use crate::kv::KvPrecision;
use crate::pipeline::PipelineMode;

/// Complete runtime configuration.
///
/// All fields have defaults. The auto-tuner (when implemented) or user
/// can override any.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Pipeline execution mode.
    pub pipeline_mode: PipelineMode,

    /// Number of layers to prefetch ahead of the compute cursor.
    pub prefetch_distance: usize,

    /// KV cache precision.
    pub kv_precision: KvPrecision,

    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// Whether to collect per-layer timing telemetry.
    pub collect_per_layer_timings: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pipeline_mode: PipelineMode::Auto,
            prefetch_distance: 2,
            kv_precision: KvPrecision::F32,
            max_seq_len: 4096,
            // Off by default: each layer timing requires 3 Instant::now()
            // calls (clock_gettime syscalls), totalling ~66 syscalls/token
            // for a 22-layer model. Enable explicitly for profiling.
            collect_per_layer_timings: false,
        }
    }
}
