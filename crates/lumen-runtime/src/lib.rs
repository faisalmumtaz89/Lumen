//! Core inference runtime for Lumen.
//!
//! ```text
//!   Execution Core (token loop)
//!          |
//!   Pipeline Scheduler
//!          |
//!   +------+------+
//!   |             |
//! Storage     Compute
//! + Cache     Backend
//! ```

/// Crate-wide lock serializing tests that mutate process environment
/// variables. Process env is global state: a per-module lock cannot exclude
/// an env-mutating test in another module, and a concurrent `set_var` while
/// another thread walks `env::vars()` is undefined behavior.
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_os = "macos")]
pub mod accelerate;
pub mod chat_template;
pub mod compute;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod engine;
pub mod error;
pub mod eval;
pub mod expert;
pub mod kv;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod pipeline;
pub mod runtime_defaults;
pub mod sampling;
pub mod session;
pub mod storage;
pub mod telemetry;
pub mod thread_pool;
pub mod tooling;
pub mod weight;

#[cfg(target_os = "macos")]
pub use accelerate::AccelerateBatchBackend;
pub use chat_template::{render_chat_prompt, ChatTemplateError};
pub use compute::cpu_naive::NaiveF32Backend;
pub use compute::cpu_simd::SimdF32Backend;
pub use compute::ComputeBackend;
pub use config::RuntimeConfig;
pub use engine::{InferenceEngine, SamplingParams, StopCondition};
pub use error::RuntimeError;
pub use eval::{coherence_score, CoherenceVerdict};
pub use kv::{KvCache, KvCacheConfig, KvPrecision};
#[cfg(target_os = "macos")]
pub use metal::MetalF32Backend;
#[cfg(target_os = "macos")]
pub use metal::RouterLayerStats;
pub use pipeline::PipelineMode;
pub use session::{PrefillResult, Session, SuffixPrefillResult, TokenStream};
pub use storage::mmap::MmapStorageBackend;
#[cfg(unix)]
pub use storage::purge_file_cache;
pub use storage::sync::SyncFileBackend;
pub use storage::{IoSnapshot, IoTracker, MmapPageCacheBackend, StorageBackend};
pub use telemetry::{
    InferenceMetrics, IoMetrics, KvCacheStats, PerLayerTiming, ServerMemoryBreakdown,
};
pub use tooling::{
    parse_final, parse_final_with_schemas, ParsedAssistant, ParsedToolCall, Qwen35Renderer,
    ReasoningDelta, ReasoningExtractor, StreamingDelta, StreamingFinish, StreamingParser,
    ToolResult, ToolSchema, ToolSchemas, THINK_CLOSE, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
};
pub use weight::cache::{CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider};
pub use weight::provider_async::AsyncWeightProvider;
pub use weight::provider_mmap::MmapWeightProvider;
pub use weight::provider_sync::SyncWeightProvider;

// MoE expert caching re-exports
pub use expert::cache::{ExpertKey, ExpertLfuCache};
pub use expert::profiler::{ExpertActivationProfiler, ProfilerSummary};
pub use expert::reader::{ExpertReader, ExpertReaderError};

// Metal IO re-exports (Metal 3, macOS 13+)
#[cfg(target_os = "macos")]
pub use metal::io::MetalIOQueue;

// CUDA backend re-exports (NVIDIA GPUs, requires --features cuda)
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
