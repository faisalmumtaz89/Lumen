//! Mmap-based storage backend using OS page cache.
//!
//! Maps the entire LBC file read-only and uses `madvise` for prefetch/release
//! hints. Implements both `StorageBackend` and `MmapPageCacheBackend`.

use crate::error::RuntimeError;
use crate::storage::{IoTracker, MmapConfig, MmapPageCacheBackend, ResidencyInfo, StorageBackend};
use std::path::Path;
use std::ptr;

/// Mmap-based storage backend.
pub struct MmapStorageBackend {
    /// Pointer to the mmap region.
    ptr: *mut u8,
    /// Length of the mmap region.
    len: usize,
    /// File descriptor (kept open for the lifetime of the mapping).
    #[cfg(unix)]
    fd: i32,
    /// Mmap configuration.
    config: MmapConfig,
    /// Cached page size (avoids repeated sysconf syscalls).
    #[cfg(unix)]
    page_size: usize,
    /// Whether the mapping is active.
    is_open: bool,
    /// I/O tracking.
    io: IoTracker,
}

// SAFETY: The mmap region is read-only and immutable after creation.
unsafe impl Send for MmapStorageBackend {}
unsafe impl Sync for MmapStorageBackend {}

impl Default for MmapStorageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MmapStorageBackend {
    pub fn new() -> Self {
        Self {
            ptr: ptr::null_mut(),
            len: 0,
            #[cfg(unix)]
            fd: -1,
            config: MmapConfig::default(),
            #[cfg(unix)]
            page_size: unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize,
            is_open: false,
            io: IoTracker::new(),
        }
    }

    /// Returns the raw pointer and length of the mmap region (for advanced usage).
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is valid and len is correct after successful mmap.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Return a raw pointer and length into the mmap region (zero-copy).
    ///
    /// The returned pointer is valid as long as the mmap is open.
    /// Records the read in the I/O tracker for telemetry.
    pub fn slice_ref(&self, offset: u64, length: u64) -> Result<(*const u8, usize), RuntimeError> {
        let off = offset as usize;
        let len = length as usize;
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("slice_ref: offset={off} + length={len} overflows usize"),
            ))
        })?;
        if end > self.len {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("slice_ref: offset={off} + length={len} > file_size={}", self.len),
            )));
        }
        self.io.record_read(length);
        // SAFETY: ptr is valid and off..end is within bounds (checked above).
        Ok((unsafe { self.ptr.add(off) }, len))
    }
}

impl StorageBackend for MmapStorageBackend {
    #[cfg(unix)]
    fn open(&mut self, path: &Path) -> Result<(), RuntimeError> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|e| RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                e,
            )))?;

        // Open the file
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
        let len = stat.st_size as usize;

        if len == 0 {
            unsafe { libc::close(fd); }
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            )));
        }

        // mmap
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            unsafe { libc::close(fd); }
            return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
        }

        self.ptr = ptr as *mut u8;
        self.len = len;
        self.fd = fd;
        self.is_open = true;

        // Apply sequential advice if configured
        if self.config.advise_sequential {
            unsafe {
                libc::madvise(self.ptr as *mut libc::c_void, self.len, libc::MADV_SEQUENTIAL);
            }
        }

        Ok(())
    }

    #[cfg(not(unix))]
    fn open(&mut self, _path: &Path) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported("mmap not available on this platform".into()))
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        if !self.is_open {
            return Ok(());
        }

        #[cfg(unix)]
        {
            if !self.ptr.is_null() {
                unsafe {
                    libc::munmap(self.ptr as *mut libc::c_void, self.len);
                }
            }
            if self.fd >= 0 {
                unsafe { libc::close(self.fd); }
            }
        }

        self.ptr = ptr::null_mut();
        self.len = 0;
        #[cfg(unix)]
        { self.fd = -1; }
        self.is_open = false;
        Ok(())
    }

    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, RuntimeError> {
        let off = offset as usize;
        let len = length as usize;
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("read_range: offset={off} + length={len} overflows usize"),
            ))
        })?;
        if end > self.len {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("read_range: offset={off} + length={len} > file_size={}", self.len),
            )));
        }
        // Copy from mmap region (Phase 1 does a copy; zero-copy comes later)
        let slice = &self.as_slice()[off..end];
        self.io.record_read(length);
        Ok(slice.to_vec())
    }

    fn file_size(&self) -> u64 {
        self.len as u64
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn io_tracker(&self) -> Option<&IoTracker> {
        Some(&self.io)
    }
}

#[cfg(unix)]
impl MmapPageCacheBackend for MmapStorageBackend {
    fn configure(&mut self, config: MmapConfig) {
        self.config = config;
    }

    fn advise_willneed(&self, offset: u64, length: u64) -> Result<(), RuntimeError> {
        if !self.is_open || length == 0 {
            return Ok(());
        }
        let page_size = self.page_size;
        let off = offset as usize;
        let len = length as usize;
        let aligned_offset = off & !(page_size - 1);
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "advise_willneed: offset+length overflows",
            ))
        })?;
        let aligned_length = end - aligned_offset;

        let ret = unsafe {
            libc::madvise(
                self.ptr.add(aligned_offset) as *mut libc::c_void,
                aligned_length,
                libc::MADV_WILLNEED,
            )
        };
        if ret != 0 {
            return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
        }
        Ok(())
    }

    fn advise_dontneed(&self, offset: u64, length: u64) -> Result<(), RuntimeError> {
        if !self.is_open || length == 0 {
            return Ok(());
        }
        let page_size = self.page_size;
        let off = offset as usize;
        let len = length as usize;
        let aligned_offset = off & !(page_size - 1);
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "advise_dontneed: offset+length overflows",
            ))
        })?;
        let aligned_length = end - aligned_offset;

        // macOS uses MADV_FREE (lazy release), Linux uses MADV_DONTNEED (immediate)
        #[cfg(target_os = "macos")]
        let advice = libc::MADV_FREE;
        #[cfg(not(target_os = "macos"))]
        let advice = libc::MADV_DONTNEED;

        let ret = unsafe {
            libc::madvise(
                self.ptr.add(aligned_offset) as *mut libc::c_void,
                aligned_length,
                advice,
            )
        };
        if ret != 0 {
            return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
        }
        Ok(())
    }

    fn pages_resident(&self, offset: u64, length: u64) -> Result<ResidencyInfo, RuntimeError> {
        if !self.is_open || length == 0 {
            return Ok(ResidencyInfo::default());
        }
        let page_size = self.page_size;
        let off = offset as usize;
        let len = length as usize;
        let aligned_offset = off & !(page_size - 1);
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pages_resident: offset+length overflows",
            ))
        })?;
        let aligned_length = end - aligned_offset;
        let num_pages = aligned_length.div_ceil(page_size);

        // mincore returns per-page residency info
        // On macOS, the vec type is `c_char` (i8); on Linux it's `c_uchar` (u8)
        #[cfg(target_os = "macos")]
        let mut vec: Vec<libc::c_char> = vec![0; num_pages];
        #[cfg(not(target_os = "macos"))]
        let mut vec: Vec<libc::c_uchar> = vec![0; num_pages];

        let ret = unsafe {
            libc::mincore(
                self.ptr.add(aligned_offset) as *mut libc::c_void,
                aligned_length,
                vec.as_mut_ptr(),
            )
        };
        if ret != 0 {
            return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
        }

        let resident = vec.iter().filter(|&&v| (v as u8) & 1 != 0).count();

        Ok(ResidencyInfo {
            total_pages: num_pages,
            resident_pages: resident,
        })
    }

    fn purge_page_cache(&self) -> Result<(), RuntimeError> {
        if !self.is_open || self.len == 0 {
            return Ok(());
        }

        // macOS: set F_NOCACHE to disable unified buffer cache for this fd
        #[cfg(target_os = "macos")]
        {
            unsafe {
                libc::fcntl(self.fd, libc::F_NOCACHE, 1);
            }
        }

        // MADV_DONTNEED evicts pages immediately on Linux;
        // on macOS it's advisory but combined with F_NOCACHE it helps.
        let ret = unsafe {
            libc::madvise(
                self.ptr as *mut libc::c_void,
                self.len,
                libc::MADV_DONTNEED,
            )
        };
        if ret != 0 {
            return Err(RuntimeError::StorageIo(std::io::Error::last_os_error()));
        }

        Ok(())
    }
}

impl Drop for MmapStorageBackend {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MMAP_STORAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn create_test_file() -> (std::path::PathBuf, Vec<u8>) {
        let id = MMAP_STORAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_test_mmap_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_mmap.bin");
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }
        (path, data)
    }

    #[test]
    fn mmap_open_read_close() {
        let (path, expected) = create_test_file();
        let mut backend = MmapStorageBackend::new();
        backend.open(&path).unwrap();

        assert!(backend.is_open());
        assert_eq!(backend.file_size(), expected.len() as u64);

        let chunk = backend.read_range(10, 100).unwrap();
        assert_eq!(chunk, &expected[10..110]);

        backend.close().unwrap();
        assert!(!backend.is_open());
    }

    #[test]
    fn mmap_advise_willneed() {
        let (path, _) = create_test_file();
        let mut backend = MmapStorageBackend::new();
        backend.open(&path).unwrap();

        // Should not error
        backend.advise_willneed(0, 4096).unwrap();

        let info = backend.pages_resident(0, 4096).unwrap();
        assert!(info.total_pages > 0);
        // After WILLNEED, pages should be at least partially resident
        // (though this is advisory, not guaranteed)

        backend.close().unwrap();
    }

    #[test]
    fn mmap_read_out_of_bounds() {
        let (path, _) = create_test_file();
        let mut backend = MmapStorageBackend::new();
        backend.open(&path).unwrap();

        let result = backend.read_range(4090, 100);
        assert!(result.is_err());

        backend.close().unwrap();
    }
}
