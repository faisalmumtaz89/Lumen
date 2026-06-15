//! Persistent on-disk cache for NVRTC-compiled CUDA kernel PTX.
//!
//! # Why this exists
//!
//! Lumen defers *all* GPU kernel compilation to runtime via NVRTC (no
//! build-time CUDA SDK dependency -- a deliberate architectural property).
//! That means every server launch recompiles ~252 NVRTC modules from source,
//! which dominates cold start. This module caches the NVRTC output to disk so
//! the second-and-later launches skip the NVRTC compile entirely and hand the
//! cached PTX straight to `cuModuleLoadData`.
//!
//! # What is cached
//!
//! The NVRTC `Image` bytes (the null-terminated PTX text NVRTC emits). Caching
//! PTX (rather than arch-specific cubin) preserves the per-host determinism
//! property: the driver JIT (PTX -> SASS) still runs on the actual host, and
//! the driver's own compute cache (`~/.nv/ComputeCache`) transparently caches
//! that second stage. The expensive, dominant stage we eliminate is the NVRTC
//! source->PTX compile.
//!
//! # Cache key
//!
//! `sha256(source) || arch || fast_math || cc_major.minor || nvrtc_version ||
//! driver_version`. Any change to a kernel's source, the target arch, the
//! compile flags, the device compute capability, the NVRTC toolkit, or the
//! driver invalidates that kernel's entry (cache miss -> recompile -> rewrite).
//! This mirrors the NVIDIA driver compute cache's own invalidation behavior.
//!
//! # Correctness
//!
//! The cache is a pure performance optimization: a cached entry, when loaded,
//! must produce byte-identical kernel behavior to a fresh NVRTC compile of the
//! same source under the same key. The driver loads the cached PTX through the
//! exact same `cuModuleLoadData` path it uses for a fresh compile, so the SASS
//! the driver JITs is identical. This is validated on-device by DET-001
//! byte-determinism and a cached-vs-fresh output-hash match (BIT-IDENTICAL
//! gate).
//!
//! # Robustness
//!
//! - Writes are atomic (temp file + `rename`), so a crash or a concurrent
//!   first-launch can never leave a half-written, corrupt cache entry.
//! - Any read / parse / validation error falls back to a fresh NVRTC compile;
//!   a bad cache file never crashes the server.
//! - The cache is default-ON. `LUMEN_CUDA_PTX_CACHE=0` disables it entirely
//!   (kill switch), matching `CUDA_CACHE_DISABLE` semantics.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide cache hit / miss counters, for the cold-vs-warm startup log.
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Snapshot the (hits, misses) counters. Used by the startup log to report how
/// many kernels were served from the warm cache vs freshly compiled.
pub(crate) fn stats() -> (u64, u64) {
    (
        CACHE_HITS.load(Ordering::Relaxed),
        CACHE_MISSES.load(Ordering::Relaxed),
    )
}

/// Record that a kernel was served from the warm cache (PTX loaded + accepted
/// by the driver without an NVRTC recompile).
pub(crate) fn record_hit() {
    CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Record that a kernel required a fresh NVRTC compile (cache miss, disabled,
/// or a rejected/corrupt entry that fell back to recompile).
pub(crate) fn record_miss() {
    CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

/// Magic header bytes identifying a Lumen PTX-cache file. A leading magic lets
/// the loader reject foreign / corrupt files before trusting the payload.
const CACHE_MAGIC: &[u8; 8] = b"LUMNPTX1";

/// Whether the PTX disk cache is enabled. Default ON; disabled only when
/// `LUMEN_CUDA_PTX_CACHE=0` (kill switch).
pub(crate) fn cache_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_PTX_CACHE") {
        Ok(v) => v != "0",
        Err(_) => true,
    }
}

/// Resolve the PTX cache directory: `$LUMEN_CACHE_DIR/ptx` (or, absent that
/// override, `$XDG_CACHE_HOME`/`~/.cache/lumen/ptx`). Mirrors the cli
/// `cache_dir()` priority without taking a dependency on the cli crate.
fn ptx_cache_dir() -> Option<PathBuf> {
    // Explicit per-process override for the whole cache path (debugging).
    if let Ok(v) = std::env::var("LUMEN_CUDA_PTX_CACHE_DIR") {
        if !v.is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    if let Ok(v) = std::env::var("LUMEN_CACHE_DIR") {
        if !v.is_empty() {
            return Some(PathBuf::from(v).join("ptx"));
        }
    }
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        if !v.is_empty() {
            return Some(PathBuf::from(v).join("lumen").join("ptx"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home).join(".cache").join("lumen").join("ptx"));
    }
    None
}

/// Components that uniquely identify a compiled kernel. Any difference here
/// must produce a different cache file (a cache miss).
pub(crate) struct CacheKey<'a> {
    /// Kernel source string (`.cu` contents). Borrowed only for hashing; never
    /// stored, so the lifetime is the caller's source slice.
    pub source: &'a str,
    /// Target arch passed to NVRTC, e.g. "compute_80" / "compute_61", or
    /// "default" when no explicit arch is set.
    pub arch: &'a str,
    /// Whether `--use_fast_math` (and its sub-flags) is enabled.
    pub fast_math: bool,
    /// Device compute capability (major, minor).
    pub cc: (i32, i32),
    /// NVRTC library version (major, minor).
    pub nvrtc_version: (i32, i32),
    /// CUDA driver version (integer, e.g. 12020).
    pub driver_version: i32,
}

impl<'a> CacheKey<'a> {
    /// The on-disk filename for this key: a hex SHA-256 over all components.
    /// The source hash dominates; the env components are appended so a
    /// driver/toolkit/arch change cleanly re-keys.
    fn digest_hex(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.source.as_bytes());
        // Domain-separate each component with a tag + length so that, e.g.,
        // arch "compute_8" + "0..." can never collide with "compute_80" + "...".
        h.update(b"|arch=");
        h.update(self.arch.as_bytes());
        h.update(b"|fm=");
        h.update(&[self.fast_math as u8]);
        h.update(b"|cc=");
        h.update(&self.cc.0.to_le_bytes());
        h.update(&self.cc.1.to_le_bytes());
        h.update(b"|nvrtc=");
        h.update(&self.nvrtc_version.0.to_le_bytes());
        h.update(&self.nvrtc_version.1.to_le_bytes());
        h.update(b"|drv=");
        h.update(&self.driver_version.to_le_bytes());
        hex(&h.finalize())
    }

    fn cache_path(&self) -> Option<PathBuf> {
        ptx_cache_dir().map(|d| d.join(format!("{}.ptxc", self.digest_hex())))
    }

    /// Path of the *driver-reject marker* for this key.
    ///
    /// A marker is written when the host driver rejects this key's PTX at
    /// `cuModuleLoadData` (the SASS JIT stage). The marker is keyed identically
    /// to the PTX entry (so it carries the same arch + cc + driver_version), and
    /// its existence lets a later warm launch skip both the doomed cached-PTX
    /// load *and* the doomed NVRTC recompile, going straight to the caller's
    /// fallback. See `mark_driver_reject` / `is_driver_rejected`.
    fn reject_path(&self) -> Option<PathBuf> {
        ptx_cache_dir().map(|d| d.join(format!("{}.ptxr", self.digest_hex())))
    }
}

/// Try to load cached NVRTC PTX bytes for `key`. Returns `None` on any miss,
/// read error, or validation failure -- the caller must then recompile. A bad
/// cache file is never fatal.
pub(crate) fn load(key: &CacheKey) -> Option<Vec<u8>> {
    if !cache_enabled() {
        return None;
    }
    let path = key.cache_path()?;
    let bytes = std::fs::read(&path).ok()?;
    parse_entry(&bytes)
}

/// Atomically write `ptx` (the NVRTC `Image` bytes, including the trailing NUL)
/// to the cache for `key`. Best-effort: any failure is silently ignored (the
/// kernel still loaded from the fresh compile; the cache is just not populated
/// this time). Uses temp-file + rename so a partial write can never be read.
pub(crate) fn store(key: &CacheKey, ptx: &[u8]) {
    if !cache_enabled() {
        return;
    }
    let Some(path) = key.cache_path() else { return };
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    // Unique temp name per (pid, key) so two concurrent first-launches writing
    // the *same* kernel don't clobber each other's temp file mid-write; the
    // final rename is atomic so whichever lands last wins with a complete file.
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        key.digest_hex(),
        std::process::id()
    ));
    let entry = serialize_entry(ptx);
    {
        let Ok(mut f) = std::fs::File::create(&tmp) else { return };
        if f.write_all(&entry).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        // Flush to the OS so the rename publishes a complete file.
        if f.flush().is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
    }
    // Atomic publish. If rename fails (e.g. cross-device), drop the temp.
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Record that the host driver *rejected* this key's PTX at `cuModuleLoadData`
/// (the PTX compiled fine under NVRTC but the device's SASS JIT refused it --
/// e.g. an sm_75 host loading `compute_80` PTX that uses `mma.sync`). Writes a
/// tiny marker file keyed identically to the PTX entry.
///
/// # Why this exists
///
/// Without the marker, every warm launch on such a host re-pays the full NVRTC
/// recompile for a kernel it can *never* load, then re-fails the driver JIT and
/// falls back -- pure wasted startup time (observed: an sm_75 T4 spending ~330s
/// re-compiling 76 `compute_80` kernels each launch that it always rejects).
/// With the marker, a warm launch detects the known rejection in one `stat()`
/// and goes straight to the caller's fallback path. The marker carries the
/// arch + cc + driver_version in its key, so a driver/toolkit upgrade that
/// *could* change the JIT outcome cleanly re-keys and retries from scratch.
///
/// Best-effort: a write failure just means the next launch retries (no harm).
pub(crate) fn mark_driver_reject(key: &CacheKey) {
    if !cache_enabled() {
        return;
    }
    let Some(path) = key.reject_path() else { return };
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    // A driver-rejected key's stale PTX entry is useless and only wastes a
    // read + a doomed load next launch; drop it so the marker is authoritative.
    if let Some(ptx_path) = key.cache_path() {
        let _ = std::fs::remove_file(&ptx_path);
    }
    let tmp = dir.join(format!(".{}.{}.rtmp", key.digest_hex(), std::process::id()));
    {
        let Ok(mut f) = std::fs::File::create(&tmp) else { return };
        // Content is irrelevant -- existence is the signal -- but a magic byte
        // makes the file self-describing if a human inspects the cache dir.
        if f.write_all(CACHE_MAGIC).is_err() || f.flush().is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
    }
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Whether this key was previously recorded as driver-rejected (see
/// [`mark_driver_reject`]). Returns `false` when caching is disabled so the
/// kill switch fully restores the un-cached compile-every-time behavior.
pub(crate) fn is_driver_rejected(key: &CacheKey) -> bool {
    if !cache_enabled() {
        return false;
    }
    key.reject_path()
        .map(|p| p.exists())
        .unwrap_or(false)
}

/// Serialize a cache entry: MAGIC || u32 len(payload) || payload || u32 crc(payload).
/// The length + CRC let the loader detect truncation / corruption.
fn serialize_entry(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    out.extend_from_slice(CACHE_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32(payload).to_le_bytes());
    out
}

/// Parse + validate a cache entry, returning the payload (PTX bytes) on
/// success. Returns `None` on any structural mismatch, so a corrupt or
/// truncated file is treated as a miss (-> recompile), never trusted.
fn parse_entry(bytes: &[u8]) -> Option<Vec<u8>> {
    // MAGIC(8) + len(4) + ... + crc(4) minimum.
    if bytes.len() < 16 {
        return None;
    }
    if &bytes[0..8] != CACHE_MAGIC {
        return None;
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    // payload occupies [12 .. 12+len), then a 4-byte CRC.
    let payload_end = 12usize.checked_add(len)?;
    let crc_end = payload_end.checked_add(4)?;
    if crc_end != bytes.len() {
        // Trailing-garbage or truncated: reject.
        return None;
    }
    let payload = &bytes[12..payload_end];
    let stored_crc = u32::from_le_bytes(bytes[payload_end..crc_end].try_into().ok()?);
    if crc32(payload) != stored_crc {
        return None;
    }
    // A valid PTX `Image` from NVRTC is NUL-terminated text. cuModuleLoadData
    // requires a NUL terminator; reject a payload that lost it.
    if payload.last() != Some(&0) {
        return None;
    }
    Some(payload.to_vec())
}

/// Remove a single cache directory recursively (test/utility helper).
#[allow(dead_code)]
pub(crate) fn clear() {
    if let Some(dir) = ptx_cache_dir() {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Count `.ptxc` files in the cache dir (diagnostic helper for the headline
/// gate: "cache files written ~= number of modules").
#[allow(dead_code)]
pub(crate) fn entry_count() -> usize {
    let Some(dir) = ptx_cache_dir() else { return 0 };
    count_ptxc(&dir)
}

#[allow(dead_code)]
fn count_ptxc(dir: &Path) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "ptxc")
                .unwrap_or(false)
        })
        .count()
}

// ---------------------------------------------------------------------------
// Self-contained SHA-256 (FIPS 180-4). Vendored to avoid adding a crate
// dependency to the lean lumen-runtime surface. Deterministic, no allocation
// beyond the streamed state.
// ---------------------------------------------------------------------------

struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.process(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.process(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        // Append 0x80 then zero-pad to 56 mod 64, then 8-byte big-endian length.
        let mut pad = [0u8; 72];
        pad[0] = 0x80;
        let pad_len = if self.buf_len < 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        self.update_no_count(&pad[..pad_len]);
        self.update_no_count(&bit_len.to_be_bytes());
        let mut out = [0u8; 32];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    /// Feed padding bytes without adding to the message-length counter.
    fn update_no_count(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.process(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.process(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn process(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// CRC-32 (IEEE 802.3) for cheap on-disk integrity. Not security-relevant
/// (the SHA-256 key already guards against source/env mismatch); this only
/// catches accidental truncation / bit-rot of the cache file itself.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate the process-global `LUMEN_CUDA_PTX_CACHE_DIR`
    /// env var. Each uses a unique temp dir, but the env var that points at it is
    /// process-wide, so without this lock parallel `cargo test` runs clobber each
    /// other's setting. Poison-tolerant: a panic in one test must not wedge the rest.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// SHA-256 of "abc" is the canonical FIPS 180-4 test vector.
    #[test]
    fn sha256_abc_vector() {
        let mut h = Sha256::new();
        h.update(b"abc");
        let got = hex(&h.finalize());
        assert_eq!(
            got,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// SHA-256 of the empty string.
    #[test]
    fn sha256_empty_vector() {
        let h = Sha256::new();
        assert_eq!(
            hex(&h.finalize()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// A long (>1 block) message to exercise the multi-block path.
    #[test]
    fn sha256_long_vector() {
        let mut h = Sha256::new();
        h.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&h.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// Streaming in odd-sized chunks must equal a single-shot hash.
    #[test]
    fn sha256_streaming_equiv() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let mut one = Sha256::new();
        one.update(&data);
        let a = hex(&one.finalize());
        let mut many = Sha256::new();
        for chunk in data.chunks(7) {
            many.update(chunk);
        }
        let b = hex(&many.finalize());
        assert_eq!(a, b);
    }

    /// A round-trip: serialize then parse returns the original payload.
    #[test]
    fn entry_roundtrip() {
        let payload = b"some ptx text\0";
        let ser = serialize_entry(payload);
        let got = parse_entry(&ser).expect("valid entry must parse");
        assert_eq!(got, payload);
    }

    /// A truncated entry must be rejected (treated as a miss).
    #[test]
    fn entry_truncated_rejected() {
        let payload = b"ptx\0";
        let mut ser = serialize_entry(payload);
        ser.truncate(ser.len() - 1);
        assert!(parse_entry(&ser).is_none());
    }

    /// A corrupted payload (CRC mismatch) must be rejected.
    #[test]
    fn entry_corrupt_payload_rejected() {
        let payload = b"ptx text\0";
        let mut ser = serialize_entry(payload);
        // Flip a byte inside the payload region (offset 12 = first payload byte).
        ser[12] ^= 0xFF;
        assert!(parse_entry(&ser).is_none());
    }

    /// Wrong magic must be rejected.
    #[test]
    fn entry_bad_magic_rejected() {
        let payload = b"ptx\0";
        let mut ser = serialize_entry(payload);
        ser[0] = b'X';
        assert!(parse_entry(&ser).is_none());
    }

    /// A payload lacking the NUL terminator must be rejected (cuModuleLoadData
    /// requires it).
    #[test]
    fn entry_missing_nul_rejected() {
        let payload = b"ptx no nul";
        let ser = serialize_entry(payload);
        assert!(parse_entry(&ser).is_none());
    }

    /// Differing any key component yields a different digest (no collision
    /// across source / arch / fast_math / cc / nvrtc / driver).
    #[test]
    fn key_components_change_digest() {
        let base = CacheKey {
            source: "kernel A",
            arch: "compute_80",
            fast_math: false,
            cc: (8, 0),
            nvrtc_version: (12, 2),
            driver_version: 12020,
        };
        let d0 = base.digest_hex();

        let variants = [
            CacheKey { source: "kernel B", ..clone_key(&base) },
            CacheKey { arch: "compute_61", ..clone_key(&base) },
            CacheKey { fast_math: true, ..clone_key(&base) },
            CacheKey { cc: (8, 6), ..clone_key(&base) },
            CacheKey { nvrtc_version: (12, 3), ..clone_key(&base) },
            CacheKey { driver_version: 12030, ..clone_key(&base) },
        ];
        for v in &variants {
            assert_ne!(d0, v.digest_hex(), "key component must change digest");
        }
    }

    fn clone_key<'a>(k: &CacheKey<'a>) -> CacheKey<'a> {
        CacheKey {
            source: k.source,
            arch: k.arch,
            fast_math: k.fast_math,
            cc: k.cc,
            nvrtc_version: k.nvrtc_version,
            driver_version: k.driver_version,
        }
    }

    /// The driver-reject marker is per-key: marking a `compute_80` key on cc 7.5
    /// must NOT make the same source's `compute_61` key (the working fallback)
    /// look rejected. This is the exact T4 scenario -- the sm_80 PTX is rejected
    /// while the sm_61 PTX loads fine -- so the marker must be arch/cc-scoped.
    #[test]
    fn driver_reject_marker_is_per_key() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Isolate the cache in a unique temp dir for this test.
        let dir = std::env::temp_dir().join(format!(
            "lumen-ptxc-test-{}-{}",
            std::process::id(),
            // a cheap unique-ish suffix without pulling in rand
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::env::set_var("LUMEN_CUDA_PTX_CACHE_DIR", &dir);
        // Ensure the kill switch isn't disabling the cache in this process.
        std::env::remove_var("LUMEN_CUDA_PTX_CACHE");

        let rejected = CacheKey {
            source: "dp4a kernel",
            arch: "compute_80",
            fast_math: false,
            cc: (7, 5),
            nvrtc_version: (12, 2),
            driver_version: 12020,
        };
        let fallback = CacheKey {
            arch: "compute_61",
            ..clone_key(&rejected)
        };

        assert!(!is_driver_rejected(&rejected), "fresh key must not be rejected");
        assert!(!is_driver_rejected(&fallback));

        mark_driver_reject(&rejected);

        assert!(
            is_driver_rejected(&rejected),
            "marked key must read as rejected"
        );
        assert!(
            !is_driver_rejected(&fallback),
            "the compute_61 fallback key must remain loadable (per-key marker)"
        );

        // A driver/toolkit bump re-keys -> retries from scratch (not rejected).
        let after_driver_bump = CacheKey {
            driver_version: 12030,
            ..clone_key(&rejected)
        };
        assert!(
            !is_driver_rejected(&after_driver_bump),
            "a driver-version change must clear the reject (re-key + retry)"
        );

        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("LUMEN_CUDA_PTX_CACHE_DIR");
    }

    /// Marking a key as driver-rejected drops any stale accepted-PTX entry for
    /// that key, so a later launch can't waste a read + doomed load on it.
    #[test]
    fn driver_reject_drops_stale_ptx_entry() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "lumen-ptxc-stale-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::env::set_var("LUMEN_CUDA_PTX_CACHE_DIR", &dir);
        std::env::remove_var("LUMEN_CUDA_PTX_CACHE");

        let key = CacheKey {
            source: "k",
            arch: "compute_80",
            fast_math: false,
            cc: (7, 5),
            nvrtc_version: (12, 2),
            driver_version: 12020,
        };
        store(&key, b"ptx body\0");
        assert!(load(&key).is_some(), "store then load must round-trip");

        mark_driver_reject(&key);
        assert!(load(&key).is_none(), "reject must drop the stale PTX entry");
        assert!(is_driver_rejected(&key));

        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("LUMEN_CUDA_PTX_CACHE_DIR");
    }
}
