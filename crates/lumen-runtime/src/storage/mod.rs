//! Storage layer traits and types.
//!
//! Two I/O backends are supported:
//! - `MmapPageCacheBackend`: mmap-based, using OS page cache
//! - `AsyncReadBackend`: explicit async reads with a user-space buffer pool
//!
//! Both implement [`StorageBackend`] and can back the
//! [`WeightProvider`](crate::weight::cache::WeightProvider) trait.

pub mod mmap;
pub mod sync;

use crate::error::RuntimeError;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic I/O counter for tracking storage read operations.
///
/// Uses relaxed ordering — counters are advisory telemetry, not synchronization.
pub struct IoTracker {
    bytes_read: AtomicU64,
    read_ops: AtomicU64,
}

impl IoTracker {
    pub fn new() -> Self {
        Self {
            bytes_read: AtomicU64::new(0),
            read_ops: AtomicU64::new(0),
        }
    }

    /// Record a completed read operation.
    pub fn record_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.read_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> IoSnapshot {
        IoSnapshot {
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            read_ops: self.read_ops.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.bytes_read.store(0, Ordering::Relaxed);
        self.read_ops.store(0, Ordering::Relaxed);
    }
}

impl Default for IoTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of I/O counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct IoSnapshot {
    pub bytes_read: u64,
    pub read_ops: u64,
}

/// Common interface for all storage backends.
pub trait StorageBackend: Send + Sync {
    fn open(&mut self, path: &Path) -> Result<(), RuntimeError>;
    fn close(&mut self) -> Result<(), RuntimeError>;

    /// Read a contiguous byte range from the file.
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, RuntimeError>;

    fn file_size(&self) -> u64;
    fn is_open(&self) -> bool;

    /// Returns the I/O tracker for this backend, if instrumented.
    fn io_tracker(&self) -> Option<&IoTracker> { None }
}

/// Configuration for the mmap-based storage backend.
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Maximum layers to prefetch ahead of the compute cursor.
    pub prefetch_window: usize,
    /// Use `madvise(MADV_SEQUENTIAL)` on layer blobs.
    pub advise_sequential: bool,
    /// Use `madvise(MADV_DONTNEED)` on released layers.
    pub release_with_dontneed: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            prefetch_window: 2,
            advise_sequential: true,
            release_with_dontneed: true,
        }
    }
}

/// The mmap-based storage backend.
///
/// Uses `mmap` for read-only mapping of the LBC file and relies on the
/// OS page cache. Implements windowed prefetch with explicit release
/// hints to mitigate the prefetch-release conflict.
pub trait MmapPageCacheBackend: StorageBackend {
    fn configure(&mut self, config: MmapConfig);

    /// Advise the OS to prefetch the byte range (`MADV_WILLNEED`).
    fn advise_willneed(&self, offset: u64, length: u64) -> Result<(), RuntimeError>;

    /// Advise the OS that the byte range is no longer needed (`MADV_DONTNEED`).
    fn advise_dontneed(&self, offset: u64, length: u64) -> Result<(), RuntimeError>;

    /// Estimate how many pages in the range are currently resident.
    fn pages_resident(&self, offset: u64, length: u64) -> Result<ResidencyInfo, RuntimeError>;

    /// Attempt to evict the entire file from the OS page cache.
    ///
    /// - Linux: `madvise(MADV_DONTNEED)` on the full mapping (immediate eviction).
    /// - macOS: `fcntl(F_NOCACHE, 1)` + `madvise(MADV_DONTNEED)` (best-effort).
    fn purge_page_cache(&self) -> Result<(), RuntimeError>;

    /// Returns `true` if residency is below `threshold` (0.0–1.0).
    fn is_cold(&self, threshold: f64) -> Result<bool, RuntimeError> {
        let info = self.pages_resident(0, self.file_size())?;
        Ok(info.residency_fraction() < threshold)
    }
}

/// Page residency information for a byte range.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidencyInfo {
    pub total_pages: usize,
    pub resident_pages: usize,
}

impl ResidencyInfo {
    pub fn residency_fraction(&self) -> f64 {
        if self.total_pages == 0 {
            0.0
        } else {
            self.resident_pages as f64 / self.total_pages as f64
        }
    }
}

/// Evict a file from the OS page cache without requiring an mmap.
///
/// - Linux: `posix_fadvise(POSIX_FADV_DONTNEED)` on the file descriptor.
/// - macOS: `fcntl(F_NOCACHE, 1)` then read/discard to force eviction.
///
/// Best-effort: some pages may remain resident.
#[cfg(unix)]
pub fn purge_file_cache(path: &Path) -> Result<(), RuntimeError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(|e| {
        RuntimeError::StorageIo(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
    })?;

    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY) };
    if fd < 0 {
        return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
    }

    // Get file size
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        unsafe { libc::close(fd); }
        return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
    }
    let len = stat.st_size as u64;

    #[cfg(target_os = "linux")]
    {
        unsafe {
            libc::posix_fadvise(fd, 0, len as i64, libc::POSIX_FADV_DONTNEED);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // F_NOCACHE disables unified buffer cache for future I/O on this fd.
        // Existing cached pages are not immediately evicted; we must read
        // through the file to force the UBC to replace them with uncached
        // reads, effectively purging the old cached copies.
        unsafe {
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }

        // Read through the file in 64KB chunks to trigger eviction.
        // Each read replaces cached pages with uncached ones (F_NOCACHE).
        const CHUNK: usize = 64 * 1024;
        let mut buf = [0u8; CHUNK];
        let mut remaining = len as usize;
        while remaining > 0 {
            let to_read = remaining.min(CHUNK);
            let n = unsafe {
                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, to_read)
            };
            if n <= 0 {
                break; // EOF or error -- best-effort
            }
            remaining -= n as usize;
        }
    }

    unsafe { libc::close(fd); }
    Ok(())
}

#[cfg(not(unix))]
pub fn purge_file_cache(_path: &Path) -> Result<(), RuntimeError> {
    Err(RuntimeError::Unsupported("purge_file_cache not available on this platform".into()))
}

