//! Metal IO command queue for direct NVMe-to-GPU DMA (Metal 3, macOS 13+).
//!
//! Wraps `MTLIOCommandQueue` to load file byte ranges directly into Metal
//! buffers without going through CPU RAM. This eliminates the double-copy
//! path: `NVMe -> mmap page fault -> CPU RAM -> MTLBuffer(shared) -> blit`
//! and replaces it with: `NVMe -> MTLIOCommandQueue -> GPU buffer`.
//!
//! On Apple Silicon with NVMe SSDs, this should approach the raw NVMe
//! bandwidth (~5-6 GB/s on M3 Ultra) versus the ~2-3 GB/s achievable
//! through the mmap/pread path due to CPU memory pressure and TLB overhead.
//!
//! ## Requirements
//!
//! - macOS 13+ (Ventura)
//! - Metal 3 hardware (M2 or later)
//! - Destination buffer can be shared or private storage mode
//!
//! ## Fallback
//!
//! On pre-Metal 3 hardware or pre-macOS 13, `MetalIOQueue::new()` returns
//! `None` and callers fall back to the existing pread + blit copy path.

use crate::metal::ffi::{
    MetalBuffer, MetalDevice, MetalIOCommandQueue, MetalIOStatus,
};
use std::path::Path;

/// Safe wrapper around Metal's IO command queue for direct file-to-GPU DMA.
///
/// Holds a persistent `MTLIOCommandQueue` and creates per-operation command
/// buffers. The queue is concurrent, allowing multiple loads to be in flight
/// simultaneously.
pub struct MetalIOQueue {
    /// The underlying Metal IO command queue.
    queue: MetalIOCommandQueue,
}

impl MetalIOQueue {
    /// Create a new Metal IO queue on the given device.
    ///
    /// Returns `None` if the device does not support `MTLIOCommandQueue`
    /// (pre-Metal 3 or pre-macOS 13). Callers should fall back to pread.
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let queue = MetalIOCommandQueue::new(device)?;
        Some(Self { queue })
    }

    /// Load a byte range from a file directly into a Metal buffer.
    ///
    /// This is the primary API for streaming expert weights. It opens a
    /// Metal IO file handle, enqueues a load command, and waits for
    /// completion synchronously.
    ///
    /// - `device`: The Metal device (needed to open the file handle).
    /// - `dest_buffer`: Target Metal buffer (shared or private storage).
    /// - `dest_offset`: Byte offset into the destination buffer.
    /// - `src_path`: Path to the source file on disk.
    /// - `src_offset`: Byte offset into the source file.
    /// - `byte_count`: Number of bytes to transfer.
    ///
    /// Returns `Ok(())` on success, `Err(String)` on failure.
    pub fn load_sync(
        &self,
        device: &MetalDevice,
        dest_buffer: &MetalBuffer,
        dest_offset: u64,
        src_path: &Path,
        src_offset: u64,
        byte_count: u64,
    ) -> Result<(), String> {
        if byte_count == 0 {
            return Ok(());
        }

        let path_str = src_path.to_str().ok_or_else(|| {
            format!("Path contains non-UTF8 characters: {}", src_path.display())
        })?;

        // Open the file handle (retained for this operation).
        let file_handle = MetalIOCommandQueue::open_file(device, path_str)?;

        // Create and submit the IO command buffer.
        let cmd = self.queue.command_buffer().ok_or_else(|| {
            "Failed to create Metal IO command buffer".to_string()
        })?;

        cmd.load_buffer(dest_buffer, dest_offset, byte_count, &file_handle, src_offset);
        cmd.commit_and_wait();

        // Verify completion status.
        let status = cmd.status();
        if status != MetalIOStatus::Complete {
            return Err(format!(
                "Metal IO load failed with status {:?} (path={}, offset={}, size={})",
                status,
                src_path.display(),
                src_offset,
                byte_count,
            ));
        }

        Ok(())
    }

    /// Load multiple byte ranges from a file into a Metal buffer in a single
    /// IO command buffer. All loads are enqueued before commit, allowing the
    /// IO command queue to optimize scheduling.
    ///
    /// Each entry in `ranges` is `(dest_offset, src_offset, byte_count)`.
    pub fn load_ranges_sync(
        &self,
        device: &MetalDevice,
        dest_buffer: &MetalBuffer,
        src_path: &Path,
        ranges: &[(u64, u64, u64)],
    ) -> Result<(), String> {
        if ranges.is_empty() {
            return Ok(());
        }

        let path_str = src_path.to_str().ok_or_else(|| {
            format!("Path contains non-UTF8 characters: {}", src_path.display())
        })?;

        let file_handle = MetalIOCommandQueue::open_file(device, path_str)?;

        let cmd = self.queue.command_buffer().ok_or_else(|| {
            "Failed to create Metal IO command buffer".to_string()
        })?;

        for &(dest_offset, src_offset, byte_count) in ranges {
            if byte_count > 0 {
                cmd.load_buffer(dest_buffer, dest_offset, byte_count, &file_handle, src_offset);
            }
        }

        cmd.commit_and_wait();

        let status = cmd.status();
        if status != MetalIOStatus::Complete {
            return Err(format!(
                "Metal IO batch load failed with status {:?} (path={}, {} ranges)",
                status,
                src_path.display(),
                ranges.len(),
            ));
        }

        Ok(())
    }
}

// MetalIOQueue is Send+Sync because MetalIOCommandQueue is Send+Sync.
// The queue itself is thread-safe (Metal IO queues are concurrent).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_io_queue_creation() {
        let device = MetalDevice::system_default().expect("Metal device required");
        let io_queue = MetalIOQueue::new(&device);
        // On M2+ with macOS 13+, this should succeed.
        // On older hardware, it returns None (which is also a valid outcome).
        if io_queue.is_some() {
            eprintln!("MetalIOQueue: created successfully (Metal 3 / macOS 13+ confirmed)");
        } else {
            eprintln!("MetalIOQueue: not available (pre-Metal 3 or pre-macOS 13)");
        }
    }

    #[test]
    fn test_metal_io_load_small_file() {
        use std::io::Write;

        let device = MetalDevice::system_default().expect("Metal device required");
        let io_queue = match MetalIOQueue::new(&device) {
            Some(q) => q,
            None => {
                eprintln!("SKIP: MetalIOQueue not available on this hardware");
                return;
            }
        };

        // Create a temp file with known data.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lumen_metal_io_test_{}.bin", std::process::id()));
        let test_data: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&test_data).expect("write temp file");
            f.flush().expect("flush temp file");
        }

        // Create a shared Metal buffer (CPU-readable for verification).
        let buf = device
            .new_buffer(test_data.len())
            .expect("create Metal buffer");

        // Load via Metal IO.
        let result = io_queue.load_sync(
            &device,
            &buf,
            0,
            &path,
            0,
            test_data.len() as u64,
        );
        assert!(result.is_ok(), "Metal IO load failed: {:?}", result.err());

        // Verify contents.
        let mut readback = vec![0u8; test_data.len()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const u8,
                readback.as_mut_ptr(),
                test_data.len(),
            );
        }
        assert_eq!(
            readback, test_data,
            "Metal IO loaded data does not match source file"
        );

        // Test sub-range load.
        let buf2 = device.new_buffer(1024).expect("create Metal buffer 2");
        let result2 = io_queue.load_sync(&device, &buf2, 0, &path, 512, 1024);
        assert!(
            result2.is_ok(),
            "Metal IO sub-range load failed: {:?}",
            result2.err()
        );
        let mut readback2 = vec![0u8; 1024];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf2.contents() as *const u8,
                readback2.as_mut_ptr(),
                1024,
            );
        }
        assert_eq!(
            readback2,
            &test_data[512..1536],
            "Metal IO sub-range data does not match"
        );

        // Test batch load (load_ranges_sync).
        let buf3 = device.new_buffer(2048).expect("create Metal buffer 3");
        let ranges = vec![
            (0u64, 0u64, 1024u64),     // First 1KB from file start -> buffer start
            (1024u64, 2048u64, 1024u64), // 1KB from file offset 2048 -> buffer offset 1024
        ];
        let result3 = io_queue.load_ranges_sync(&device, &buf3, &path, &ranges);
        assert!(
            result3.is_ok(),
            "Metal IO batch load failed: {:?}",
            result3.err()
        );
        let mut readback3 = vec![0u8; 2048];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf3.contents() as *const u8,
                readback3.as_mut_ptr(),
                2048,
            );
        }
        assert_eq!(
            &readback3[..1024],
            &test_data[..1024],
            "Batch load range 1 mismatch"
        );
        assert_eq!(
            &readback3[1024..],
            &test_data[2048..3072],
            "Batch load range 2 mismatch"
        );

        // Cleanup.
        std::fs::remove_file(&path).ok();
        eprintln!("test_metal_io_load_small_file: PASS (all verifications correct)");
    }

    #[test]
    fn test_metal_io_zero_byte_load() {
        let device = MetalDevice::system_default().expect("Metal device required");
        let io_queue = match MetalIOQueue::new(&device) {
            Some(q) => q,
            None => {
                eprintln!("SKIP: MetalIOQueue not available");
                return;
            }
        };

        let buf = device.new_buffer(64).expect("create buffer");
        let path = std::path::Path::new("/dev/null");
        let result = io_queue.load_sync(&device, &buf, 0, path, 0, 0);
        assert!(result.is_ok(), "Zero-byte load should succeed");
    }

    #[test]
    fn test_metal_io_empty_ranges() {
        let device = MetalDevice::system_default().expect("Metal device required");
        let io_queue = match MetalIOQueue::new(&device) {
            Some(q) => q,
            None => {
                eprintln!("SKIP: MetalIOQueue not available");
                return;
            }
        };

        let buf = device.new_buffer(64).expect("create buffer");
        let path = std::path::Path::new("/dev/null");
        let result = io_queue.load_ranges_sync(&device, &buf, path, &[]);
        assert!(result.is_ok(), "Empty ranges should succeed");
    }

    /// Benchmark Metal IO vs pread for streaming expert loading.
    ///
    /// Requires: `/tmp/lumen-bench/mixtral-8x7b-v0.1.lbc`
    /// Run with: cargo test -p lumen-runtime -- --ignored test_metal_io_bandwidth
    #[test]
    #[ignore]
    fn test_metal_io_bandwidth() {
        use std::time::Instant;

        let lbc_path = std::path::Path::new("/tmp/lumen-bench/mixtral-8x7b-v0.1.lbc");
        if !lbc_path.exists() {
            eprintln!(
                "SKIP: benchmark file not found at {}",
                lbc_path.display()
            );
            return;
        }

        let device = MetalDevice::system_default().expect("Metal device required");
        let io_queue = match MetalIOQueue::new(&device) {
            Some(q) => q,
            None => {
                eprintln!("SKIP: MetalIOQueue not available on this hardware");
                return;
            }
        };

        // Read layer 0 size from LBC index.
        let lbc = lumen_format::reader::LbcFile::open(lbc_path).unwrap();
        if lbc.layer_indices.is_empty() {
            eprintln!("SKIP: no layers in LBC file");
            return;
        }

        let idx = &lbc.layer_indices[0];
        let offset = idx.layer_offset_bytes;
        let length = idx.layer_length_bytes as usize;

        eprintln!("=== Metal IO vs pread Bandwidth Benchmark ===");
        eprintln!(
            "File: {} ({:.1} GB)",
            lbc_path.display(),
            lbc_path.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1e9
        );
        eprintln!(
            "Layer 0: offset={}, length={:.1} MB",
            offset,
            length as f64 / 1e6
        );

        // Warm up: pread to prime SSD controller.
        let _ = crate::expert::reader::parallel_pread(lbc_path, offset, length, 4);

        // Benchmark pread (4 threads).
        let start = Instant::now();
        let _pread_data = crate::expert::reader::parallel_pread(lbc_path, offset, length, 4).unwrap();
        let elapsed_pread = start.elapsed();
        let bw_pread = length as f64 / 1e9 / elapsed_pread.as_secs_f64();
        eprintln!(
            "pread (4 threads): {:.1} MB in {:.2?} = {:.2} GB/s",
            length as f64 / 1e6,
            elapsed_pread,
            bw_pread
        );

        // Benchmark Metal IO (shared buffer).
        let shared_buf = device.new_buffer(length).expect("create shared buffer");
        let start = Instant::now();
        io_queue
            .load_sync(&device, &shared_buf, 0, lbc_path, offset, length as u64)
            .expect("Metal IO load failed");
        let elapsed_io = start.elapsed();
        let bw_io = length as f64 / 1e9 / elapsed_io.as_secs_f64();
        eprintln!(
            "Metal IO (shared): {:.1} MB in {:.2?} = {:.2} GB/s ({:.1}x vs pread)",
            length as f64 / 1e6,
            elapsed_io,
            bw_io,
            elapsed_pread.as_secs_f64() / elapsed_io.as_secs_f64()
        );

        // Benchmark Metal IO (private buffer).
        if let Some(private_buf) = device.new_buffer_private(length) {
            let start = Instant::now();
            io_queue
                .load_sync(&device, &private_buf, 0, lbc_path, offset, length as u64)
                .expect("Metal IO load to private buffer failed");
            let elapsed_priv = start.elapsed();
            let bw_priv = length as f64 / 1e9 / elapsed_priv.as_secs_f64();
            eprintln!(
                "Metal IO (private): {:.1} MB in {:.2?} = {:.2} GB/s ({:.1}x vs pread)",
                length as f64 / 1e6,
                elapsed_priv,
                bw_priv,
                elapsed_pread.as_secs_f64() / elapsed_priv.as_secs_f64()
            );
        }

        eprintln!("\n=== Benchmark PASS ===");
    }
}
