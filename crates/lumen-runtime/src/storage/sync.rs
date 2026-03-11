//! Synchronous file-based storage backend.
//!
//! Uses `std::fs::File` with `pread` on Unix for `&self` reads (no mutex needed).

use crate::error::RuntimeError;
use crate::storage::{IoTracker, StorageBackend};
use std::path::Path;

pub struct SyncFileBackend {
    file: Option<std::fs::File>,
    file_size: u64,
    io: IoTracker,
}

impl Default for SyncFileBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncFileBackend {
    pub fn new() -> Self {
        Self {
            file: None,
            file_size: 0,
            io: IoTracker::new(),
        }
    }
}

impl StorageBackend for SyncFileBackend {
    fn open(&mut self, path: &Path) -> Result<(), RuntimeError> {
        let file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        self.file_size = metadata.len();
        self.file = Some(file);
        Ok(())
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        self.file = None;
        self.file_size = 0;
        Ok(())
    }

    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, RuntimeError> {
        let file = self.file.as_ref().ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "file not open",
            ))
        })?;

        let len = usize::try_from(length).map_err(|_| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("read_range: length {length} exceeds usize"),
            ))
        })?;
        let mut buf = vec![0u8; len];

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(&mut buf, offset)?;
        }

        #[cfg(not(unix))]
        {
            // Non-Unix: StorageBackend takes &self but std Seek/Read need &mut.
            // Compile error guides the user to add a Mutex wrapper for Windows.
            compile_error!("SyncFileBackend requires Unix pread; wrap File in Mutex for non-Unix");
        }

        self.io.record_read(length);
        Ok(buf)
    }

    fn file_size(&self) -> u64 {
        self.file_size
    }

    fn is_open(&self) -> bool {
        self.file.is_some()
    }

    fn io_tracker(&self) -> Option<&IoTracker> {
        Some(&self.io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_range_works() {
        let dir = std::env::temp_dir().join("lumen_test_sync_storage");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        }

        let mut backend = SyncFileBackend::new();
        backend.open(&path).unwrap();
        assert!(backend.is_open());
        assert_eq!(backend.file_size(), 8);

        let data = backend.read_range(2, 4).unwrap();
        assert_eq!(data, vec![2, 3, 4, 5]);

        backend.close().unwrap();
        assert!(!backend.is_open());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn io_tracker_counts_reads() {
        let dir = std::env::temp_dir().join("lumen_test_sync_io_tracker");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0u8; 64]).unwrap();
        }

        let mut backend = SyncFileBackend::new();
        backend.open(&path).unwrap();

        // Before any reads, counters should be zero
        let tracker = backend.io_tracker().unwrap();
        let snap = tracker.snapshot();
        assert_eq!(snap.bytes_read, 0);
        assert_eq!(snap.read_ops, 0);

        // Read 16 bytes
        backend.read_range(0, 16).unwrap();
        let snap = backend.io_tracker().unwrap().snapshot();
        assert_eq!(snap.bytes_read, 16);
        assert_eq!(snap.read_ops, 1);

        // Read 32 more bytes
        backend.read_range(16, 32).unwrap();
        let snap = backend.io_tracker().unwrap().snapshot();
        assert_eq!(snap.bytes_read, 48);
        assert_eq!(snap.read_ops, 2);

        backend.close().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
}
