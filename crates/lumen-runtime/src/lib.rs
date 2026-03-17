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

pub mod compute;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod accelerate;
pub mod config;
pub mod engine;
pub mod error;
pub mod expert;
pub mod kv;
pub mod pipeline;
pub mod storage;
pub mod telemetry;
pub mod thread_pool;
pub mod weight;

pub use weight::cache::{CacheStats, LayerView, PrefetchHandle, PrefetchPriority, WeightProvider};
pub use compute::ComputeBackend;
pub use compute::cpu_naive::NaiveF32Backend;
pub use compute::cpu_simd::SimdF32Backend;
#[cfg(target_os = "macos")]
pub use metal::MetalF32Backend;
#[cfg(target_os = "macos")]
pub use metal::RouterLayerStats;
#[cfg(target_os = "macos")]
pub use accelerate::AccelerateBatchBackend;
pub use config::RuntimeConfig;
pub use engine::InferenceEngine;
pub use error::RuntimeError;
pub use kv::{KvCache, KvCacheConfig, KvPrecision};
pub use pipeline::PipelineMode;
pub use storage::{IoSnapshot, IoTracker, MmapPageCacheBackend, StorageBackend};
#[cfg(unix)]
pub use storage::purge_file_cache;
pub use storage::mmap::MmapStorageBackend;
pub use storage::sync::SyncFileBackend;
pub use telemetry::{InferenceMetrics, IoMetrics, PerLayerTiming};
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
