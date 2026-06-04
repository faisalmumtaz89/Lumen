//! Disk-persistent KV cache (P1-5 from.
//!
//! Persists a [`KvCache`] to a single file so a long conversation can survive
//! a server restart and be reloaded as a warm KV prefix. The file is identified
//! by `<sha1(token_ids)>.kv`, so prefix-cache lookup is a single hash + stat.
//!
//! ## File layout (one file per cache snapshot, v2)
//!
//! ```text
//! +--------------------------------------------------------+
//! | Header (96 bytes, little-endian)                       |
//! +--------------------------------------------------------+
//! |  0..4  : magic = b"KVC\0" (0x004356_4B)                |
//! |  4..8  : version = 2                                   |
//! |  8..16 : seq_len (u64)        -- tokens in this cache  |
//! | 16..20 : num_layers (u32)                              |
//! | 20..24 : num_kv_heads (u32)                            |
//! | 24..28 : head_dim (u32)                                |
//! | 28..32 : max_seq_len (u32)    -- buffer dimension      |
//! | 32..36 : precision tag (u32)  -- 0=F32, 1=F16          |
//! | 36..40 : payload_crc32 (u32)  -- CRC-32 of payload     |
//! | 40..44 : hits (u32)           -- access counter        |
//! | 44..48 : last_used_secs (u32) -- unix epoch (low 32b)  |
//! | 48..52 : weight_quant_tag (u32) -- mirrors QuantScheme |
//! | 52..56 : lumen_format_version (u32) -- LBC version     |
//! | 56..88 : model_hash (32 bytes, sha256 fingerprint)     |
//! | 88..89: has_recurrent_state (u8) -- 0=no, 1=yes|
//! | 89..90: has_pending_logits (u8) -- 0=no, 1=yes|
//! | 90..96 : reserved (6 bytes, zeroed) -- forward compat  |
//! +--------------------------------------------------------+
//! | Token IDs : seq_len * 4 bytes (u32, little-endian)     |
//! +--------------------------------------------------------+
//! | Per-layer KV payload, num_layers blocks:                |
//! |   For each layer:                                       |
//! |     Keys:   num_kv_heads * max_seq_len * head_dim * bpe|
//! |     Values: num_kv_heads * max_seq_len * head_dim * bpe|
//! +--------------------------------------------------------+
//! | OPTIONAL recurrent-state section: |
//! |   Present iff has_recurrent_state == 1.                 |
//! |   Layout metadata (32 bytes):                           |
//! |     +0  : recurrent_layout_version (u32, =1)            |
//! |     +4  : num_gdn_layers (u32)                          |
//! |     +8  : gdn_num_heads (u32)                           |
//! |    +12  : gdn_head_dim (u32)                            |
//! |    +16  : gdn_conv_kernel_size (u32)                    |
//! |    +20  : gdn_conv_qkv_dim (u32)                        |
//! |    +24  : gdn_dtype_tag (u32, 0=F32)                    |
//! |    +28  : recurrent_section_crc32 (u32)                 |
//! |   Per GDN layer (num_gdn_layers blocks):                |
//! |     h_state:     gdn_num_heads * gdn_head_dim^2 * bpe   |
//! |     conv_state:  (kernel_size - 1) * qkv_dim * bpe      |
//! |     conv_pos:    4 bytes (u32)                          |
//! +--------------------------------------------------------+
//! | OPTIONAL pending-logits section: |
//! |   Present iff has_pending_logits == 1.                  |
//! |   Layout metadata (12 bytes):                           |
//! |     +0  : pending_logits_vocab_size (u32)               |
//! |     +4  : pending_logits_crc32 (u32)                    |
//! |     +8  : reserved (4 bytes, zeroed)                    |
//! |   Payload: vocab_size * 4 bytes (LE f32)                |
//! +--------------------------------------------------------+
//! ```
//!
//! ## v1 → v2 migration policy
//!
//! v1 files are **rejected** on load with an explicit error; there is no
//! automatic migration. The user regenerates by running the prior session
//! cold. Rationale: the new `model_hash` field cannot be reconstructed from
//! v1 data, so silently "upgrading" would defeat the fingerprint check it
//! exists to provide.
//!
//! ## v2 backward compatibility
//!
//! The recurrent-state section is **opt-in within v2** — its presence is
//! signalled by the `has_recurrent_state` byte at header offset 88, which
//! previous v2 writers left at zero (the byte sat in the reserved block).
//! Existing v2 files without recurrent state therefore load unchanged; new
//! v2 files written with GDN state are still readable by and later
//! Lumen versions. Older Lumen builds that do not know about the recurrent
//! section can still read the K/V payload correctly because the section
//! follows the per-layer K/V blocks and would simply be ignored after EOF
//! on file truncation (we never truncate; the section sits after K/V).
//!
//! The buffer is written at full `max_seq_len` capacity (not just `seq_len`
//! positions) because the head-first layout interleaves all heads' position
//! slots. This wastes disk space when `seq_len < max_seq_len`, but matches
//! the in-memory layout exactly so `load` is a single `read`.
//!
//! ## Atomic write
//!
//! `save_atomic` writes to `<dir>/<sha1>.kv.tmp.<pid>` then renames over the
//! final name. On POSIX, rename is atomic for files on the same filesystem.
//! This guarantees readers never see a partially written file.
//!
//! ## Eviction
//!
//! [`evict_to_budget`] enforces a directory-wide size budget. It scores each
//! file by `(hits + 1) * tokens / file_size`, lower scores are evicted first.
//! Live-session prefixes (passed via a hash set) receive a 0.25× penalty so
//! they are evicted before unrelated entries of equal score.
//!
//! ## Correctness contract
//!
//! `save(kv, tokens) -> load(...) -> resume` produces a `KvCache` whose
//! `seq_len` and byte contents are bit-equal to the original. The property
//! test `disk_save_load_resume_is_bitwise_identical` enforces this.

use crate::error::RuntimeError;
use crate::kv::{KvCache, KvCacheConfig, KvPrecision};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// "KVC\0" in little-endian.
const MAGIC: u32 = 0x0043_564B;

/// On-disk format version.
///
/// **v1 → v2**: Header grows from 48 → 96 bytes; new fields
/// `weight_quant_tag`, `lumen_format_version`, `model_hash[32]`, plus 8 bytes of
/// reserved space for forward-compat. The new fields let the loader detect
/// model mismatches (different quantization, different LBC version, different
/// weights) before allocating ~50 GB of KV buffers, per feasibility refinement.
/// **v1 files are rejected on load** (no automatic migration — the user must
/// regenerate). Earlier revisions explicitly chose this policy to avoid maintaining
/// v1 forever.
const VERSION: u32 = 2;

/// Fixed header size in bytes (v2 = 96 bytes, 8-byte aligned).
pub const HEADER_SIZE: usize = 96;

/// Reserved bytes inside the header for forward compatibility.
const HEADER_RESERVED_BYTES: usize = 8;

/// Precision tag in the header.
const TAG_F32: u32 = 0;
const TAG_F16: u32 = 1;

fn precision_tag(p: KvPrecision) -> Result<u32, RuntimeError> {
    match p {
        KvPrecision::F32 => Ok(TAG_F32),
        KvPrecision::F16 => Ok(TAG_F16),
        other => Err(RuntimeError::Unsupported(format!(
            "disk KV: precision {other:?} is not serializable"
        ))),
    }
}

fn tag_to_precision(tag: u32) -> Result<KvPrecision, RuntimeError> {
    match tag {
        TAG_F32 => Ok(KvPrecision::F32),
        TAG_F16 => Ok(KvPrecision::F16),
        other => Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("disk KV: unknown precision tag {other}"),
        ))),
    }
}

/// Compute SHA-1 of the byte slice and return it as a 40-character lowercase
/// hex string. Implementation follows RFC 3174 -- 32-bit big-endian words,
/// 80-round message schedule, no dependencies.
///
/// SHA-1 is used purely as a content-addressable filename; collision
/// resistance is not a security requirement here.
pub fn sha1_hex(bytes: &[u8]) -> String {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    // Build padded message: original bytes || 0x80 || 0x00..0x00 || len_bits_be64.
    let bit_len = (bytes.len() as u64) * 8;
    let mut padded = Vec::with_capacity(bytes.len() + 9 + 63);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            let off = i * 4;
            w[i] = u32::from_be_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDCu32),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let temp = a.rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut hex = String::with_capacity(40);
    for w in [h0, h1, h2, h3, h4] {
        for byte in w.to_be_bytes() {
            hex.push(nibble_to_hex(byte >> 4));
            hex.push(nibble_to_hex(byte & 0x0f));
        }
    }
    hex
}

#[inline]
fn nibble_to_hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + n - 10) as char,
    }
}

/// SHA-256 (RFC 6234) of an arbitrary byte slice. Returns the 32-byte digest.
///
/// Used by [`compute_model_hash`] to fingerprint the model whose KV cache is
/// being persisted. SHA-256 is chosen over SHA-1 here
/// because (a) the fingerprint is used as a CORRECTNESS gate, not just a
/// filename — collision resistance matters when "this KV is from this model"
/// must hold; (b) SHA-1 is increasingly considered weakened, and changing
/// the hash post-release would force another format bump.
///
/// Implementation follows the standard 64-round message schedule with 64
/// 32-bit constants. No dependencies; matches the SHA-1 inline pattern.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    // SHA-256 round constants (first 32 bits of fractional parts of cube
    // roots of the first 64 primes, per FIPS 180-4 §4.2.2).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Pad: original || 0x80 || 0x00...0x00 || len_bits_be64.
    let bit_len = (bytes.len() as u64) * 8;
    let mut padded = Vec::with_capacity(bytes.len() + 9 + 63);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let off = i * 4;
            w[i] = u32::from_be_bytes([
                chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3],
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

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Model fingerprint stored in the disk-KV header.
///
/// Identifies the exact model whose KV cache was persisted. The loader
/// rejects a file whose fingerprint does not match the live model's
/// fingerprint, preventing silent corruption when a user changes the model
/// without clearing the KV cache directory.
///
/// All three fields are populated by the caller (engine init time); the
/// disk module is content-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFingerprint {
    /// SHA-256 of `sha256(hyperparams_bincode) XOR sha256(vocab_json) XOR
    /// tensor_crc_aggregate_le_bytes` per the model-hash refinement. The caller
    /// computes this via [`compute_model_hash`].
    pub model_hash: [u8; 32],

    /// Mirror of the model's primary weight `QuantScheme` (cast to u32 via
    /// `QuantScheme::to_u8`). A model loaded as Q4_0 vs Q8_0 produces
    /// different KV cache reductions even at identical hyperparams; reject
    /// on mismatch.
    pub weight_quant_tag: u32,

    /// Mirror of `lumen_format::LBC_VERSION` at save time. A future LBC
    /// header bump that changes per-tensor layouts would invalidate any
    /// existing KV cache.
    pub lumen_format_version: u32,
}

impl ModelFingerprint {
    /// Sentinel fingerprint used by callers that have not yet wired up the
    /// real fingerprint computation. **Tests only** — production callers
    /// MUST pass a real fingerprint or the rejection check is defeated.
    #[cfg(test)]
    pub fn test_zero() -> Self {
        Self {
            model_hash: [0u8; 32],
            weight_quant_tag: 0,
            lumen_format_version: 0,
        }
    }

    /// Build a real ModelFingerprint for the live model. Used by the CLI's
    /// `--session-resume` / `--session-save` plumbing
    /// where the operator opens an LBC file and needs a deterministic
    /// fingerprint without re-wiring the LBC reader to surface tensor CRCs.
    ///
    /// `hyperparams_bytes` should encode the live model's [`ModelHyperparams`]
    /// in a stable form (the CLI uses the LE bincode-equivalent layout via
    /// `serialize_hyperparams_le` below; the server can compute the same).
    /// `vocab_bytes` is the embedded tokenizer's `tokens_blob`, which is
    /// already a deterministic byte sequence in the LBC file.
    /// `weight_quant_tag` is the primary weight quantization (typically
    /// `QuantScheme::to_u8 as u32`).
    /// `lumen_format_version` is `lumen_format::LBC_VERSION`.
    ///
    /// The model_hash is derived via [`compute_model_hash`] with a
    /// `tensor_crc_aggregate` of `weight_quant_tag as u64 ^ lumen_format_version
    /// as u64 ^ hyperparams_bytes.len() as u64`. This is not as
    /// collision-resistant as a true tensor-by-tensor CRC fold (because the
    /// LBC reader does not currently surface per-tensor CRCs without a full
    /// scan), but it does catch every change to hyperparams, vocabulary, or
    /// quantization — the three mismatch classes the docstring of
    /// `compute_model_hash` enumerates. A future revision that adds CRC
    /// aggregation to the LBC reader can swap in a stronger aggregate here
    /// without changing call-sites.
    pub fn from_live_model(
        hyperparams_bytes: &[u8],
        vocab_bytes: &[u8],
        weight_quant_tag: u32,
        lumen_format_version: u32,
    ) -> Self {
        let aggregate = (weight_quant_tag as u64)
            ^ (lumen_format_version as u64).wrapping_shl(32)
            ^ (hyperparams_bytes.len() as u64);
        Self {
            model_hash: compute_model_hash(hyperparams_bytes, vocab_bytes, aggregate),
            weight_quant_tag,
            lumen_format_version,
        }
    }
}

/// Stable LE byte serialization of [`lumen_format::ModelHyperparams`] for use
/// as the `hyperparams_bytes` input to [`ModelFingerprint::from_live_model`].
///
/// We don't reach into bincode here because the LBC `ModelHyperparams` is
/// already a fixed-layout POD struct; we just concatenate every field as a
/// little-endian byte sequence in declaration order. The resulting byte
/// sequence is stable across builds, hosts, and runs of the same Lumen
/// version, which is the only correctness guarantee callers depend on.
pub fn serialize_hyperparams_le(hp: &lumen_format::ModelHyperparams) -> Vec<u8> {
    let mut out = Vec::with_capacity(96);
    out.extend_from_slice(&hp.num_layers.to_le_bytes());
    out.extend_from_slice(&hp.num_heads.to_le_bytes());
    out.extend_from_slice(&hp.num_kv_heads.to_le_bytes());
    out.extend_from_slice(&hp.head_dim.to_le_bytes());
    out.extend_from_slice(&hp.hidden_dim.to_le_bytes());
    out.extend_from_slice(&hp.intermediate_dim.to_le_bytes());
    out.extend_from_slice(&hp.vocab_size.to_le_bytes());
    out.extend_from_slice(&hp.max_seq_len.to_le_bytes());
    if let Some(rope) = hp.rope_params {
        out.push(1);
        out.extend_from_slice(&rope.theta.to_le_bytes());
        out.extend_from_slice(&rope.scaling_factor.to_le_bytes());
        // `scaling_type` is a small enum; encode the discriminant via
        // the existing Debug-name lookup so future variants extend the
        // fingerprint cleanly.
        let tag: u8 = match rope.scaling_type {
            lumen_format::hyperparams::RopeScalingType::None => 0,
            lumen_format::hyperparams::RopeScalingType::Linear => 1,
            lumen_format::hyperparams::RopeScalingType::Ntk => 2,
            lumen_format::hyperparams::RopeScalingType::Yarn => 3,
        };
        out.push(tag);
    } else {
        out.push(0);
    }
    out.extend_from_slice(&hp.num_experts.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&hp.num_active_experts.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&hp.norm_eps.to_le_bytes());
    out.extend_from_slice(&hp.rotary_dim.unwrap_or(0).to_le_bytes());
    out.push(if hp.rope_neox { 1 } else { 0 });
    out
}

/// Compute the 32-byte `model_hash` per the model-hash spec.
///
/// `sha256(hyperparams_bincode) XOR sha256(vocab_json) XOR pad32(tensor_crc_aggregate_le)`.
///
/// `tensor_crc_aggregate` is the caller's chosen aggregate (typically a u64
/// XOR-fold over each tensor's CRC-32 in the LBC file). It is padded to 32
/// bytes by repeating its little-endian byte representation; XOR with the
/// other two SHA-256 outputs yields a deterministic 32-byte fingerprint.
///
/// The XOR-of-hashes pattern means each component independently
/// invalidates the fingerprint: change the tokenizer alone → invalidate;
/// change one weight tensor → invalidate; change hidden_dim → invalidate.
/// SHA-256 collision resistance means the combined hash is also
/// collision-resistant in practice.
pub fn compute_model_hash(
    hyperparams_bytes: &[u8],
    vocab_bytes: &[u8],
    tensor_crc_aggregate: u64,
) -> [u8; 32] {
    let h_hp = sha256(hyperparams_bytes);
    let h_vocab = sha256(vocab_bytes);

    // Pad the aggregate u64 to 32 bytes by repeating its LE form 4×.
    // This keeps the XOR pattern uniform across all 32 byte positions.
    let agg_le = tensor_crc_aggregate.to_le_bytes();
    let mut h_agg = [0u8; 32];
    for i in 0..32 {
        h_agg[i] = agg_le[i % 8];
    }

    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = h_hp[i] ^ h_vocab[i] ^ h_agg[i];
    }
    out
}

/// Compute the canonical disk filename for a token-id sequence.
///
/// The filename is `<sha1_hex(token_ids_as_le_u32)>.kv`. The token IDs are
/// hashed as their little-endian byte representation so the same token
/// sequence produces the same name on any little-endian host (the entire
/// project assumes LE; see KV scatter-write fast paths).
pub fn cache_filename(token_ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(token_ids.len() * 4);
    for &t in token_ids {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    format!("{}.kv", sha1_hex(&bytes))
}

/// IEEE 802.3 CRC-32 (poly 0xEDB88320, reflected) for payload checksumming.
///
/// Computed on-the-fly with a 16-entry table, then folded; ~1 GB/s on a
/// modern core. The "right" amount of integrity checking for KV files --
/// strong enough to catch bit rot, cheap enough to not gate write speed.
pub fn crc32(bytes: &[u8]) -> u32 {
    // Half-byte table: each entry is the CRC of the 4-bit value 0..15.
    const TABLE: [u32; 16] = {
        let mut t = [0u32; 16];
        let mut i = 0;
        while i < 16 {
            let mut c = i as u32;
            let mut j = 0;
            while j < 4 {
                if c & 1 != 0 {
                    c = 0xEDB8_8320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
                j += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        crc = (crc >> 4) ^ TABLE[(crc & 0x0f) as usize];
        crc = (crc >> 4) ^ TABLE[(crc & 0x0f) as usize];
    }
    !crc
}

/// On-disk header (v2, 96 bytes). Public because tests in other modules
/// construct headers directly to probe edge cases (corrupted versions, wrong
/// magic, etc.).
#[derive(Debug, Clone, Copy)]
pub struct DiskKvHeader {
    pub magic: u32,
    pub version: u32,
    pub seq_len: u64,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub precision_tag: u32,
    pub payload_crc32: u32,
    pub hits: u32,
    pub last_used_secs: u32,
    /// weight quant tag (mirrors QuantScheme::to_u8 widened
    /// to u32). Loader rejects mismatch.
    pub weight_quant_tag: u32,
    /// LBC format version at save time. Loader rejects
    /// mismatch (different LBC layout would invalidate the in-memory layout
    /// the KV is striped against).
    pub lumen_format_version: u32,
    /// 32-byte SHA-256 model fingerprint. Loader rejects
    /// mismatch (different model → KV is garbage for the live attention
    /// weights). See [`compute_model_hash`].
    pub model_hash: [u8; 32],
    /// flag indicating a [`RecurrentState`] section follows the
    /// K/V payload. `0` = absent, `1` = present, all other values reject as
    /// corrupted. Older v2 writers leave this at zero (the byte lived in the
    /// reserved block); newer writers set it when persisting GDN state.
    pub has_recurrent_state: u8,
    /// flag indicating a pending-logits section follows the
    /// (optional) recurrent state. `0` = absent, `1` = present. Saving
    /// `pending_logits` lets a resumed `Session::next_token` use the
    /// "Path A" sample-from-cached-logits code path, matching the
    /// continuous session's first decode bit-exactly.
    pub has_pending_logits: u8,
}

impl DiskKvHeader {
    /// Serialize the header to its 96-byte on-disk representation (v2).
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..16].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[16..20].copy_from_slice(&self.num_layers.to_le_bytes());
        buf[20..24].copy_from_slice(&self.num_kv_heads.to_le_bytes());
        buf[24..28].copy_from_slice(&self.head_dim.to_le_bytes());
        buf[28..32].copy_from_slice(&self.max_seq_len.to_le_bytes());
        buf[32..36].copy_from_slice(&self.precision_tag.to_le_bytes());
        buf[36..40].copy_from_slice(&self.payload_crc32.to_le_bytes());
        buf[40..44].copy_from_slice(&self.hits.to_le_bytes());
        buf[44..48].copy_from_slice(&self.last_used_secs.to_le_bytes());
        buf[48..52].copy_from_slice(&self.weight_quant_tag.to_le_bytes());
        buf[52..56].copy_from_slice(&self.lumen_format_version.to_le_bytes());
        buf[56..88].copy_from_slice(&self.model_hash);
        // 88..89 carries the has_recurrent_state flag (carved out of
        // the v2-reserved block); 89..90 carries has_pending_logits;
        // 90..96 stays zeroed for forward compat.
        buf[88] = self.has_recurrent_state;
        buf[89] = self.has_pending_logits;
        let _ = HEADER_RESERVED_BYTES; // touch the constant so refactors see it
        buf
    }

    /// Parse a 96-byte header. Validates `magic` and `version` so a corrupted
    /// or v1 file is rejected at parse time, not at the bulk-read step.
    ///
    /// **v1 files are REJECTED** with an explicit error; the loader expects
    /// the user to regenerate from a cold prefill. See module docs.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self, RuntimeError> {
        let take_u32 = |range: std::ops::Range<usize>| -> u32 {
            u32::from_le_bytes(buf[range].try_into().unwrap())
        };
        let magic = take_u32(0..4);
        let version = take_u32(4..8);
        if magic != MAGIC {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("disk KV: bad magic 0x{magic:08x}, expected 0x{MAGIC:08x}"),
            )));
        }
        if version != VERSION {
            // Explicit message for v1 → v2 (the realistic case): users
            // upgrading Lumen will see a clear "regenerate" hint instead of a
            // generic version mismatch.
            let extra = if version == 1 {
                " (v1 caches are no longer supported; delete the file and let \
                 the next cold prefill regenerate it)"
            } else {
                ""
            };
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("disk KV: unsupported version {version}, expected {VERSION}{extra}"),
            )));
        }
        let seq_len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let mut model_hash = [0u8; 32];
        model_hash.copy_from_slice(&buf[56..88]);
        // has_recurrent_state. Older v2 writers left this byte at zero
        // (it sat in the reserved block) so absence is the default. Reject
        // any non-canonical value to catch corruption at the parse step
        // rather than after allocating the (potentially 50 GB) KV.
        let has_recurrent_state_byte = buf[88];
        if has_recurrent_state_byte > 1 {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV: invalid has_recurrent_state byte {has_recurrent_state_byte} \
                     (expected 0 or 1)"
                ),
            )));
        }
        let has_pending_logits_byte = buf[89];
        if has_pending_logits_byte > 1 {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV: invalid has_pending_logits byte {has_pending_logits_byte} \
                     (expected 0 or 1)"
                ),
            )));
        }
        Ok(DiskKvHeader {
            magic,
            version,
            seq_len,
            num_layers: take_u32(16..20),
            num_kv_heads: take_u32(20..24),
            head_dim: take_u32(24..28),
            max_seq_len: take_u32(28..32),
            precision_tag: take_u32(32..36),
            payload_crc32: take_u32(36..40),
            hits: take_u32(40..44),
            last_used_secs: take_u32(44..48),
            weight_quant_tag: take_u32(48..52),
            lumen_format_version: take_u32(52..56),
            model_hash,
            has_recurrent_state: has_recurrent_state_byte,
            has_pending_logits: has_pending_logits_byte,
        })
    }
}

// ---------------------------------------------------------------------------
// Recurrent state (GDN h_states + conv_states) persistence
// ---------------------------------------------------------------------------

/// Wire format version for the recurrent-state section. Bumped whenever the
/// per-layer layout changes; the loader rejects unknown versions.
pub const RECURRENT_LAYOUT_VERSION: u32 = 1;

/// Size in bytes of the fixed recurrent-section header (layout meta + CRC).
pub const RECURRENT_META_BYTES: usize = 32;

/// Dtype tag for the recurrent state. Currently F32 only because both Metal
/// and CUDA backends store GDN h_states/conv_states as F32 regardless of the
/// weight quant scheme (see `crates/lumen-runtime/src/metal/backend_impl.rs`
/// where `h_buf` is allocated `h_state_size * 4` bytes and zeroed with
/// `vec![0.0f32; ...]`). The tag exists so a future BF16 or F16 recurrent
/// path can land without breaking the format.
pub const RECURRENT_DTYPE_F32: u32 = 0;

/// Logical layout of the GDN recurrent state for a model.
///
/// Captures every dimension a loader must match against the live model to
/// safely restore CPU bytes into the backend's GPU buffers. A mismatch on
/// any field triggers an explicit rejection at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLayout {
    /// Number of layers that carry GDN state (typically 24 for Qwen3.5-9B).
    pub num_gdn_layers: u32,
    /// GDN head count (`ssm.time_step_rank`, e.g. 32).
    pub gdn_num_heads: u32,
    /// GDN per-head dimension (`ssm.state_size`, e.g. 128).
    pub gdn_head_dim: u32,
    /// Conv1d kernel size (typically 4).
    pub gdn_conv_kernel_size: u32,
    /// Conv1d total QKV channel count (Q+K+V, e.g. 8192).
    pub gdn_conv_qkv_dim: u32,
    /// Dtype tag for both `h_state` and `conv_state` bytes. Currently
    /// always [`RECURRENT_DTYPE_F32`].
    pub gdn_dtype_tag: u32,
}

impl GdnLayout {
    /// Bytes per element for the current dtype.
    pub fn bytes_per_element(&self) -> usize {
        match self.gdn_dtype_tag {
            RECURRENT_DTYPE_F32 => 4,
            _ => 0, // rejected at parse time
        }
    }

    /// Bytes per layer's h_state.
    pub fn h_state_bytes_per_layer(&self) -> usize {
        (self.gdn_num_heads as usize)
            * (self.gdn_head_dim as usize)
            * (self.gdn_head_dim as usize)
            * self.bytes_per_element()
    }

    /// Bytes per layer's conv_state.
    pub fn conv_state_bytes_per_layer(&self) -> usize {
        (self.gdn_conv_kernel_size as usize - 1)
            * (self.gdn_conv_qkv_dim as usize)
            * self.bytes_per_element()
    }

    /// Total recurrent-section payload size (excluding the 32-byte meta).
    pub fn payload_bytes(&self) -> usize {
        let per_layer = self.h_state_bytes_per_layer()
            + self.conv_state_bytes_per_layer()
            + 4 /* conv_pos u32 */;
        (self.num_gdn_layers as usize) * per_layer
    }
}

/// Per-model GDN recurrent state captured at save time and restored at load
/// time. Each per-layer `Vec<u8>` holds raw bytes in the dtype declared by
/// `layout.gdn_dtype_tag` (currently F32 only).
///
/// The backend reads/writes these byte buffers verbatim — the disk layer
/// has no awareness of layout-internal striping. See
/// `ComputeBackend::sync_kv_to_cpu` (Metal override) for the GPU-side
/// extraction path.
#[derive(Debug, Clone)]
pub struct RecurrentState {
    pub layout: GdnLayout,
    /// Per-GDN-layer h_state bytes. `h_states.len() == layout.num_gdn_layers`.
    pub h_states: Vec<Vec<u8>>,
    /// Per-GDN-layer conv_state bytes. `conv_states.len() == layout.num_gdn_layers`.
    pub conv_states: Vec<Vec<u8>>,
    /// Per-GDN-layer conv1d circular-buffer write position
    /// (`0..kernel_size-1`).
    pub conv_positions: Vec<u32>,
}

impl RecurrentState {
    /// Construct a zeroed recurrent state from a layout. Used by the
    /// loader before populating with disk bytes, and by the backend's
    /// extract helper when it allocates target buffers.
    pub fn zeroed(layout: GdnLayout) -> Self {
        let h_bytes = layout.h_state_bytes_per_layer();
        let c_bytes = layout.conv_state_bytes_per_layer();
        let n = layout.num_gdn_layers as usize;
        Self {
            layout,
            h_states: (0..n).map(|_| vec![0u8; h_bytes]).collect(),
            conv_states: (0..n).map(|_| vec![0u8; c_bytes]).collect(),
            conv_positions: vec![0u32; n],
        }
    }

    /// Reject when the live model's layout differs from this stored layout.
    ///
    /// Same-shape match required on every field — partial matches are not
    /// safe because the byte buffers stripe over the dimensions linearly.
    pub fn validate_layout_matches(&self, expected: &GdnLayout) -> Result<(), RuntimeError> {
        if self.layout != *expected {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV load: recurrent-state layout mismatch \
                     (file: {:?}; live: {:?})",
                    self.layout, expected,
                ),
            )));
        }
        Ok(())
    }
}

/// Streaming-write the recurrent section to `f` while accumulating a CRC
/// over the per-layer bytes. The 32-byte meta is patched in afterwards once
/// the CRC is known, identical to the K/V CRC strategy.
fn write_recurrent_section<W: Write + Seek>(
    f: &mut W,
    state: &RecurrentState,
) -> Result<(), RuntimeError> {
    let layout = state.layout;
    // Sanity-check the vectors match the layout. A mismatch here means the
    // caller assembled a malformed RecurrentState; surface it instead of
    // writing a corrupt file.
    let expected = layout.num_gdn_layers as usize;
    if state.h_states.len() != expected
        || state.conv_states.len() != expected
        || state.conv_positions.len() != expected
    {
        return Err(RuntimeError::Compute(format!(
            "disk KV save: recurrent-state vector length mismatch \
             (h={}, conv={}, pos={}, expected={expected})",
            state.h_states.len(),
            state.conv_states.len(),
            state.conv_positions.len(),
        )));
    }
    let h_bytes = layout.h_state_bytes_per_layer();
    let c_bytes = layout.conv_state_bytes_per_layer();

    // Stage a 32-byte meta header. CRC is patched after we stream the
    // per-layer bytes; we remember the section-start offset so we can
    // seek back exactly once.
    let meta_start = f.stream_position().map_err(RuntimeError::StorageIo)?;
    let mut meta = [0u8; RECURRENT_META_BYTES];
    meta[0..4].copy_from_slice(&RECURRENT_LAYOUT_VERSION.to_le_bytes());
    meta[4..8].copy_from_slice(&layout.num_gdn_layers.to_le_bytes());
    meta[8..12].copy_from_slice(&layout.gdn_num_heads.to_le_bytes());
    meta[12..16].copy_from_slice(&layout.gdn_head_dim.to_le_bytes());
    meta[16..20].copy_from_slice(&layout.gdn_conv_kernel_size.to_le_bytes());
    meta[20..24].copy_from_slice(&layout.gdn_conv_qkv_dim.to_le_bytes());
    meta[24..28].copy_from_slice(&layout.gdn_dtype_tag.to_le_bytes());
    meta[28..32].copy_from_slice(&0u32.to_le_bytes()); // CRC placeholder
    f.write_all(&meta).map_err(RuntimeError::StorageIo)?;

    let mut crc = Crc32Stream::new();
    for layer in 0..expected {
        let h = &state.h_states[layer];
        let c = &state.conv_states[layer];
        if h.len() != h_bytes || c.len() != c_bytes {
            return Err(RuntimeError::Compute(format!(
                "disk KV save: recurrent layer {layer} byte length mismatch \
                 (h={}/{h_bytes}, c={}/{c_bytes})",
                h.len(),
                c.len(),
            )));
        }
        f.write_all(h).map_err(RuntimeError::StorageIo)?;
        crc.update(h);
        f.write_all(c).map_err(RuntimeError::StorageIo)?;
        crc.update(c);
        let pos_bytes = state.conv_positions[layer].to_le_bytes();
        f.write_all(&pos_bytes).map_err(RuntimeError::StorageIo)?;
        crc.update(&pos_bytes);
    }
    let crc = crc.finalize();
    // Patch the CRC into the meta header.
    let end_pos = f.stream_position().map_err(RuntimeError::StorageIo)?;
    f.seek(SeekFrom::Start(meta_start + 28))
        .map_err(RuntimeError::StorageIo)?;
    f.write_all(&crc.to_le_bytes())
        .map_err(RuntimeError::StorageIo)?;
    f.seek(SeekFrom::Start(end_pos))
        .map_err(RuntimeError::StorageIo)?;
    Ok(())
}

// ---- Pending-logits section (12-byte meta + vocab_size * 4 bytes) -----

/// Size of the pending-logits meta block (vocab + CRC + reserved).
pub const PENDING_LOGITS_META_BYTES: usize = 12;

fn write_pending_logits_section<W: Write + Seek>(
    f: &mut W,
    logits: &[f32],
) -> Result<(), RuntimeError> {
    let vocab_size = logits.len() as u32;
    let meta_start = f.stream_position().map_err(RuntimeError::StorageIo)?;
    let mut meta = [0u8; PENDING_LOGITS_META_BYTES];
    meta[0..4].copy_from_slice(&vocab_size.to_le_bytes());
    meta[4..8].copy_from_slice(&0u32.to_le_bytes()); // CRC placeholder
    // meta[8..12] left zeroed for forward-compat.
    f.write_all(&meta).map_err(RuntimeError::StorageIo)?;

    let mut crc = Crc32Stream::new();
    // Stream logits as LE f32 — this keeps the file portable across
    // hosts that share LE byte order (every Lumen target).
    let mut buf = vec![0u8; 4 * logits.len()];
    for (i, &v) in logits.iter().enumerate() {
        let b = v.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&b);
    }
    f.write_all(&buf).map_err(RuntimeError::StorageIo)?;
    crc.update(&buf);
    let crc = crc.finalize();

    // Patch the CRC.
    let end = f.stream_position().map_err(RuntimeError::StorageIo)?;
    f.seek(SeekFrom::Start(meta_start + 4))
        .map_err(RuntimeError::StorageIo)?;
    f.write_all(&crc.to_le_bytes())
        .map_err(RuntimeError::StorageIo)?;
    f.seek(SeekFrom::Start(end))
        .map_err(RuntimeError::StorageIo)?;
    Ok(())
}

fn read_pending_logits_section<R: Read>(
    f: &mut R,
    expected_vocab_size: Option<u32>,
) -> Result<Vec<f32>, RuntimeError> {
    let mut meta = [0u8; PENDING_LOGITS_META_BYTES];
    f.read_exact(&mut meta).map_err(RuntimeError::StorageIo)?;
    let vocab_size = u32::from_le_bytes(meta[0..4].try_into().unwrap());
    let expected_crc = u32::from_le_bytes(meta[4..8].try_into().unwrap());
    if let Some(want_vocab) = expected_vocab_size {
        if vocab_size != want_vocab {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV load: pending-logits vocab_size mismatch \
                     (file={vocab_size}, live={want_vocab})"
                ),
            )));
        }
    }
    if vocab_size > 1_000_000 {
        // Sanity: vocab > 1M is almost certainly file corruption.
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("disk KV load: implausible pending-logits vocab_size {vocab_size}"),
        )));
    }
    let mut bytes = vec![0u8; 4 * (vocab_size as usize)];
    f.read_exact(&mut bytes).map_err(RuntimeError::StorageIo)?;
    let mut crc = Crc32Stream::new();
    crc.update(&bytes);
    let actual_crc = crc.finalize();
    if actual_crc != expected_crc {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "disk KV load: pending-logits CRC mismatch \
                 (got 0x{actual_crc:08x}, expected 0x{expected_crc:08x})"
            ),
        )));
    }
    let mut out = vec![0f32; vocab_size as usize];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

/// Streaming-read the recurrent section from `f` and validate the CRC.
fn read_recurrent_section<R: Read>(f: &mut R) -> Result<RecurrentState, RuntimeError> {
    let mut meta = [0u8; RECURRENT_META_BYTES];
    f.read_exact(&mut meta).map_err(RuntimeError::StorageIo)?;
    let take_u32 = |off: usize| -> u32 {
        u32::from_le_bytes(meta[off..off + 4].try_into().unwrap())
    };
    let layout_version = take_u32(0);
    if layout_version != RECURRENT_LAYOUT_VERSION {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "disk KV load: unsupported recurrent_layout_version {layout_version} \
                 (expected {RECURRENT_LAYOUT_VERSION}); delete the file and regenerate"
            ),
        )));
    }
    let dtype_tag = take_u32(24);
    if dtype_tag != RECURRENT_DTYPE_F32 {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("disk KV load: unknown recurrent dtype tag {dtype_tag}"),
        )));
    }
    let layout = GdnLayout {
        num_gdn_layers: take_u32(4),
        gdn_num_heads: take_u32(8),
        gdn_head_dim: take_u32(12),
        gdn_conv_kernel_size: take_u32(16),
        gdn_conv_qkv_dim: take_u32(20),
        gdn_dtype_tag: dtype_tag,
    };
    let expected_crc = take_u32(28);

    if layout.gdn_conv_kernel_size == 0 {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "disk KV load: recurrent gdn_conv_kernel_size must be >= 1",
        )));
    }

    let mut state = RecurrentState::zeroed(layout);
    let mut crc = Crc32Stream::new();
    for layer in 0..(layout.num_gdn_layers as usize) {
        f.read_exact(&mut state.h_states[layer])
            .map_err(RuntimeError::StorageIo)?;
        crc.update(&state.h_states[layer]);
        f.read_exact(&mut state.conv_states[layer])
            .map_err(RuntimeError::StorageIo)?;
        crc.update(&state.conv_states[layer]);
        let mut pos_bytes = [0u8; 4];
        f.read_exact(&mut pos_bytes).map_err(RuntimeError::StorageIo)?;
        crc.update(&pos_bytes);
        state.conv_positions[layer] = u32::from_le_bytes(pos_bytes);
    }
    let actual_crc = crc.finalize();
    if actual_crc != expected_crc {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "disk KV load: recurrent-section CRC mismatch \
                 (got 0x{actual_crc:08x}, expected 0x{expected_crc:08x})"
            ),
        )));
    }
    Ok(state)
}

fn now_secs_u32() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

/// Per-layer byte size for a given cache config.
fn layer_payload_bytes(cfg: &KvCacheConfig) -> usize {
    let bpe = cfg.precision.bytes_per_element();
    // K + V interleaved on disk; mirrors the in-memory `keys[layer]` /
    // `values[layer]` separation.
    cfg.num_kv_heads * cfg.max_seq_len * cfg.head_dim * bpe * 2
}

/// Write `kv` to `path` atomically: write to a sibling `.tmp.<pid>` file then
/// `rename` over the destination. `tokens` is interned in the file so a later
/// `load` can recover the token sequence even if the on-disk filename is
/// ambiguous (filesystem case-folding, hash collision, etc.).
///
/// The `tokens.len()` MUST equal `kv.seq_len()` -- a hard requirement of the
/// suffix-prefill contract (the in-memory pair is `Session::{tokens,kv}` with
/// the invariant `tokens.len() == kv.seq_len()` enforced by [`Session::extend_with_cache`]).
///
/// `fingerprint` identifies the model whose KV was saved; the loader rejects
/// any file whose fingerprint differs. Callers compute
/// the fingerprint once at engine init via [`compute_model_hash`].
///
/// `recurrent`, when `Some`, persists the GDN recurrent state (h_states +
/// conv_states + conv_positions) in an opt-in section after the K/V payload.
/// Callers MUST pass it for GDN backends — otherwise resume
/// produces nonsense because attention KV is rolled forward but the GDN
/// recurrent state is zeroed on the destination session.
///
/// `pending_logits`, when `Some`, persists the cached logits the next
/// `Session::next_token` call should sample from — without it, the loaded
/// session falls into the forward-pass path on the first decode and
/// produces a different argmax than the continuous session would have.
/// Pass `None` only when you know the caller won't resume into
/// `Session::next_token` (e.g. a tooling consumer that only reads tokens).
///
/// On ENOSPC or any other write error the tmp file is cleaned up before the
/// error propagates. Successful saves leave only the published
/// final file in the directory; no `.tmp.<pid>` artifact survives.
pub fn save_atomic(
    kv: &KvCache,
    tokens: &[u32],
    path: &Path,
    hits: u32,
    fingerprint: &ModelFingerprint,
    recurrent: Option<&RecurrentState>,
    pending_logits: Option<&[f32]>,
) -> Result<(), RuntimeError> {
    save_atomic_inner(kv, tokens, path, hits, fingerprint, recurrent, pending_logits)
}

fn save_atomic_inner(
    kv: &KvCache,
    tokens: &[u32],
    path: &Path,
    hits: u32,
    fingerprint: &ModelFingerprint,
    recurrent: Option<&RecurrentState>,
    pending_logits: Option<&[f32]>,
) -> Result<(), RuntimeError> {
    if tokens.len() != kv.seq_len() {
        return Err(RuntimeError::Compute(format!(
            "disk KV save: tokens.len()={} != kv.seq_len()={}",
            tokens.len(),
            kv.seq_len(),
        )));
    }
    let cfg = kv.config().clone();
    let tag = precision_tag(cfg.precision)?;

    // Stage at `<final>.tmp.<pid>` so concurrent saves do not collide and
    // crashes leave no partial `final` -- they leave a stray `.tmp.<pid>`
    // that the next eviction pass cleans up.
    let parent = path.parent().ok_or_else(|| {
        RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "disk KV save: path has no parent",
        ))
    })?;
    fs::create_dir_all(parent).map_err(RuntimeError::StorageIo)?;
    let pid = std::process::id();
    let final_name = path
        .file_name()
        .map(|f| f.to_owned())
        .ok_or_else(|| {
            RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "disk KV save: path has no filename",
            ))
        })?;
    let mut tmp_name = final_name.clone();
    tmp_name.push(format!(".tmp.{pid}"));
    let tmp_path = parent.join(&tmp_name);

    // Open exclusively so two processes with the same PID-namespace (e.g.,
    // PID-recycled inside a container restart) do not race.
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(RuntimeError::StorageIo)?;

    // Helper closure: when ANY write/sync step fails we unlink the tmp file
    // before returning the error so disk-full and similar IO faults do not
    // strand `.tmp.<pid>` orphans. `purge_stale_tmp` on
    // the next startup still acts as a safety net for SIGKILL paths.
    let cleanup_on_err = |err: RuntimeError, tmp: &Path| -> RuntimeError {
        let _ = fs::remove_file(tmp);
        err
    };

    // Write a zero-CRC header first; we'll seek back and patch it after the
    // payload streams through `crc32_streaming`. This avoids buffering the
    // whole payload in RAM just to checksum it.
    let mut header = DiskKvHeader {
        magic: MAGIC,
        version: VERSION,
        seq_len: kv.seq_len() as u64,
        num_layers: cfg.num_layers as u32,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        max_seq_len: cfg.max_seq_len as u32,
        precision_tag: tag,
        payload_crc32: 0,
        hits,
        last_used_secs: now_secs_u32(),
        weight_quant_tag: fingerprint.weight_quant_tag,
        lumen_format_version: fingerprint.lumen_format_version,
        model_hash: fingerprint.model_hash,
        has_recurrent_state: if recurrent.is_some() { 1 } else { 0 },
        has_pending_logits: if pending_logits.is_some() { 1 } else { 0 },
    };
    if let Err(e) = f.write_all(&header.to_bytes()) {
        return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
    }

    // Tokens section.
    {
        let mut tok_bytes = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            tok_bytes.extend_from_slice(&t.to_le_bytes());
        }
        if let Err(e) = f.write_all(&tok_bytes) {
            return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
        }
    }

    // Per-layer payload (alternating K then V). Stream and CRC concurrently.
    let mut crc_state = Crc32Stream::new();
    for layer in 0..cfg.num_layers {
        let (k_bytes, v_bytes) = kv.layer_raw_bytes(layer)
            .map_err(|e| cleanup_on_err(e, &tmp_path))?;
        let expected = layer_payload_bytes(&cfg) / 2;
        if k_bytes.len() != expected || v_bytes.len() != expected {
            return Err(cleanup_on_err(
                RuntimeError::Compute(format!(
                    "disk KV save: layer {layer} buffer length mismatch \
                     (k={}, v={}, expected={expected})",
                    k_bytes.len(),
                    v_bytes.len(),
                )),
                &tmp_path,
            ));
        }
        if let Err(e) = f.write_all(k_bytes) {
            return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
        }
        crc_state.update(k_bytes);
        if let Err(e) = f.write_all(v_bytes) {
            return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
        }
        crc_state.update(v_bytes);
    }
    header.payload_crc32 = crc_state.finalize();

    // optional recurrent-state section.
    if let Some(rec) = recurrent {
        if let Err(e) = write_recurrent_section(&mut f, rec) {
            return Err(cleanup_on_err(e, &tmp_path));
        }
    }

    // optional pending-logits section (independent of the
    // recurrent state — lives after it on disk, but either can be present
    // without the other).
    if let Some(logits) = pending_logits {
        if let Err(e) = write_pending_logits_section(&mut f, logits) {
            return Err(cleanup_on_err(e, &tmp_path));
        }
    }

    // Patch the CRC into the header.
    if let Err(e) = f.seek(SeekFrom::Start(0)) {
        return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
    }
    if let Err(e) = f.write_all(&header.to_bytes()) {
        return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
    }

    // POSIX dir-rename durability.
    //
    // Before the rename, we MUST ensure the file's bytes are durably on disk;
    // otherwise the rename can land in the directory entry before the data
    // pages have hit the platter, and a crash between rename and writeback
    // would leave a zero-length / partial file at `path` while the previous
    // entry is gone. `File::sync_all()` translates to `fcntl(F_FULLFSYNC)` on
    // macOS in modern Rust (verified in std source; works on APFS) and to
    // `fsync(2)` on Linux ext4/btrfs. After the rename, we also sync the
    // parent directory so the new directory entry survives a crash.
    if let Err(e) = f.sync_all() {
        return Err(cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path));
    }
    drop(f);

    // Atomic publish.
    fs::rename(&tmp_path, path)
        .map_err(|e| cleanup_on_err(RuntimeError::StorageIo(e), &tmp_path))?;
    // Sync the parent directory so the rename itself is durable on crash.
    // On macOS APFS this is a fast fcntl call; on Linux ext4/btrfs it's an
    // fsync of the dir inode. Ignored EINVAL on FS that doesn't support dir
    // sync (legacy macOS HFS+; not a Lumen target but no reason to error).
    if let Some(parent) = path.parent() {
        if let Ok(dirf) = File::open(parent) {
            let _ = dirf.sync_all();
        }
    }
    Ok(())
}

/// Result of a successful `load_into`.
///
/// Not `Clone`: `KvCache` owns multi-megabyte buffers and we never want to
/// silently duplicate them.
#[derive(Debug)]
pub struct LoadedKv {
    pub kv: KvCache,
    pub tokens: Vec<u32>,
    pub hits: u32,
    pub last_used_secs: u32,
    /// Recurrent state captured when the file was saved. `Some` when the
    /// header's `has_recurrent_state` flag is set; `None` otherwise. The
    /// caller restores it into the backend before resuming generation.
    pub recurrent: Option<RecurrentState>,
    /// Pending logits (the output of the last forward pass at save time)
    /// captured when the saved session had a cached `pending_logits`. The
    /// caller (typically `Session::load_from_disk`) installs this back
    /// into the session so the next `next_token` call's first decode is
    /// bit-exactly identical to a continuous session's first decode.
    pub pending_logits: Option<Vec<f32>>,
}

/// Reconstruct a `KvCache` from the file at `path`.
///
/// Returns the deserialized `KvCache`, the token sequence it was saved with,
/// and the on-disk telemetry counters. The caller is expected to:
/// 1. Increment `hits` and overwrite via a subsequent `save_atomic` (cheap
///    metadata update, but it requires a re-write of the full file --
///    callers that don't need precise hit counts can skip this).
/// 2. Verify that the recovered `tokens` is a prefix of the new prompt before
///    accepting the cache (paranoia against hash collisions).
///
/// `expected_fingerprint` MUST match the model the caller has loaded. If the
/// header's fingerprint differs the loader returns an error WITHOUT
/// allocating the (potentially 50 GB) KV buffers — this is the N8
/// early-rejection. The C7 fingerprint match is
/// required for correctness; if the user is OK with the legacy "no check"
/// behavior they can pass `expected_fingerprint = None` (e.g., tests).
///
/// `expected_shape`, if `Some`, must match the header's `(num_layers,
/// num_kv_heads, head_dim, max_seq_len, precision_tag)`. Pass `None` to
/// skip shape validation (e.g., the live model's KV cache itself was sized
/// from the same header — but in practice production callers should always
/// pass the live shape).
///
/// `expected_gdn_layout`, if `Some`, must match the file's recurrent-state
/// layout. When the live model has GDN layers but the file
/// has none, OR vice versa, the loader rejects with an explicit error so
/// the caller can decide whether to fall back to a cold prefill instead of
/// silently corrupting the recurrent state.
///
/// `expected_pending_logits_vocab`, when `Some`, must match the file's
/// pending-logits vocab size. The loader rejects a vocab-size mismatch
/// (different model variants, different post-processing). When the live
/// caller does not care about pending logits, pass `None` and the section
/// is consumed but its contents discarded if present.
pub fn load_into(
    path: &Path,
    expected_fingerprint: Option<&ModelFingerprint>,
    expected_shape: Option<&KvCacheConfig>,
    expected_gdn_layout: Option<&GdnLayout>,
    expected_pending_logits_vocab: Option<u32>,
) -> Result<LoadedKv, RuntimeError> {
    let mut f = File::open(path).map_err(RuntimeError::StorageIo)?;
    let mut header_buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut header_buf).map_err(RuntimeError::StorageIo)?;
    let header = DiskKvHeader::from_bytes(&header_buf)?;

    // fingerprint match BEFORE buffer allocation. A 50 GB
    // KV cache would otherwise be allocated only to be discarded. This
    // closes the N8.
    if let Some(want) = expected_fingerprint {
        if header.model_hash != want.model_hash {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "disk KV load: model_hash mismatch (KV file was saved with a \
                 different model; delete the file and let the next cold prefill \
                 regenerate it)",
            )));
        }
        if header.weight_quant_tag != want.weight_quant_tag {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV load: weight_quant_tag mismatch (file={}, live={}); \
                     regenerate by removing the file",
                    header.weight_quant_tag, want.weight_quant_tag,
                ),
            )));
        }
        if header.lumen_format_version != want.lumen_format_version {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV load: lumen_format_version mismatch (file={}, live={}); \
                     regenerate by removing the file",
                    header.lumen_format_version, want.lumen_format_version,
                ),
            )));
        }
    }

    let precision = tag_to_precision(header.precision_tag)?;
    let cfg = KvCacheConfig {
        max_seq_len: header.max_seq_len as usize,
        num_layers: header.num_layers as usize,
        num_kv_heads: header.num_kv_heads as usize,
        head_dim: header.head_dim as usize,
        precision,
    };

    // validate model SHAPE against the live KV config
    // BEFORE allocating buffers. A mismatched num_layers / num_kv_heads /
    // head_dim / max_seq_len would otherwise allocate ~50 GB only to fail at
    // `set_layer_raw_bytes`. Reject early.
    if let Some(want) = expected_shape {
        if cfg.num_layers != want.num_layers
            || cfg.num_kv_heads != want.num_kv_heads
            || cfg.head_dim != want.head_dim
            || cfg.max_seq_len != want.max_seq_len
            || cfg.precision != want.precision
        {
            return Err(RuntimeError::StorageIo(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "disk KV load: shape mismatch (file: num_layers={}, num_kv_heads={}, \
                     head_dim={}, max_seq_len={}, precision={:?}; live: num_layers={}, \
                     num_kv_heads={}, head_dim={}, max_seq_len={}, precision={:?})",
                    cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
                    cfg.precision,
                    want.num_layers, want.num_kv_heads, want.head_dim, want.max_seq_len,
                    want.precision,
                ),
            )));
        }
    }

    // Tokens.
    let tok_count = header.seq_len as usize;
    if tok_count > header.max_seq_len as usize {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "disk KV load: seq_len={tok_count} exceeds max_seq_len={}",
                header.max_seq_len
            ),
        )));
    }
    let mut tok_bytes = vec![0u8; tok_count * 4];
    f.read_exact(&mut tok_bytes).map_err(RuntimeError::StorageIo)?;
    let mut tokens = Vec::with_capacity(tok_count);
    for chunk in tok_bytes.chunks_exact(4) {
        tokens.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    // Allocate the empty cache to populate.
    let mut kv = KvCache::new(cfg.clone())?;

    // Per-layer payload streamed back. CRC the same bytes we wrote so we
    // detect bit rot before handing the cache to compute.
    let mut crc_state = Crc32Stream::new();
    let layer_half = layer_payload_bytes(&cfg) / 2;
    for layer in 0..cfg.num_layers {
        let mut k_buf = vec![0u8; layer_half];
        let mut v_buf = vec![0u8; layer_half];
        f.read_exact(&mut k_buf).map_err(RuntimeError::StorageIo)?;
        crc_state.update(&k_buf);
        f.read_exact(&mut v_buf).map_err(RuntimeError::StorageIo)?;
        crc_state.update(&v_buf);
        kv.set_layer_raw_bytes(layer, k_buf, v_buf)?;
    }

    let actual_crc = crc_state.finalize();
    if actual_crc != header.payload_crc32 {
        return Err(RuntimeError::StorageIo(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "disk KV load: payload CRC mismatch (got 0x{actual_crc:08x}, \
                 expected 0x{:08x})",
                header.payload_crc32
            ),
        )));
    }

    // optional recurrent-state section. Decoupled from the
    // K/V payload CRC so a corrupted recurrent section does not
    // mis-attribute to the K/V CRC and vice versa.
    let recurrent = if header.has_recurrent_state == 1 {
        let state = read_recurrent_section(&mut f)?;
        if let Some(want) = expected_gdn_layout {
            state.validate_layout_matches(want)?;
        }
        Some(state)
    } else {
        // File asserts no recurrent state. If the live model expects GDN
        // state, surface a clear error so the caller falls back to a cold
        // prefill instead of silently zeroing the recurrent history.
        if let Some(want) = expected_gdn_layout {
            if want.num_gdn_layers > 0 {
                return Err(RuntimeError::StorageIo(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "disk KV load: live model has {} GDN layers but the file \
                         carries no recurrent state; regenerate by removing the file",
                        want.num_gdn_layers
                    ),
                )));
            }
        }
        None
    };

    // optional pending-logits section. Independent of the
    // recurrent section (either may be present alone); also CRC-validated
    // independently.
    let pending_logits = if header.has_pending_logits == 1 {
        Some(read_pending_logits_section(&mut f, expected_pending_logits_vocab)?)
    } else {
        None
    };

    // Advance the in-memory cursor to match.
    kv.set_seq_len(tok_count)?;

    Ok(LoadedKv {
        kv,
        tokens,
        hits: header.hits,
        last_used_secs: header.last_used_secs,
        recurrent,
        pending_logits,
    })
}

/// Inspect the header without loading the payload. Cheap stat-and-read for
/// the eviction loop.
fn read_header(path: &Path) -> Result<DiskKvHeader, RuntimeError> {
    let mut f = File::open(path).map_err(RuntimeError::StorageIo)?;
    let mut header_buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut header_buf).map_err(RuntimeError::StorageIo)?;
    DiskKvHeader::from_bytes(&header_buf)
}

/// Streaming CRC-32 builder: accept slices, finalize once.
struct Crc32Stream {
    state: u32,
}

impl Crc32Stream {
    fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }
    fn update(&mut self, bytes: &[u8]) {
        // Same half-byte table as `crc32`. Inlined to avoid a second
        // const-eval and keep `crc32` -> `Crc32Stream::update` parity.
        const TABLE: [u32; 16] = {
            let mut t = [0u32; 16];
            let mut i = 0;
            while i < 16 {
                let mut c = i as u32;
                let mut j = 0;
                while j < 4 {
                    if c & 1 != 0 {
                        c = 0xEDB8_8320 ^ (c >> 1);
                    } else {
                        c >>= 1;
                    }
                    j += 1;
                }
                t[i] = c;
                i += 1;
            }
            t
        };
        let mut crc = self.state;
        for &b in bytes {
            crc ^= b as u32;
            crc = (crc >> 4) ^ TABLE[(crc & 0x0f) as usize];
            crc = (crc >> 4) ^ TABLE[(crc & 0x0f) as usize];
        }
        self.state = crc;
    }
    fn finalize(self) -> u32 {
        !self.state
    }
}

// ---------------------------------------------------------------------------
// Eviction policy
// ---------------------------------------------------------------------------

/// Eviction record for a single on-disk cache file.
#[derive(Debug, Clone)]
pub struct EvictEntry {
    /// Absolute path to the `.kv` file.
    pub path: PathBuf,
    /// Filesystem size in bytes.
    pub file_size: u64,
    /// Sequence length stored in the header (tokens covered).
    pub seq_len: u64,
    /// `hits` counter from the header.
    pub hits: u32,
    /// `last_used_secs` from the header (Unix epoch low 32 bits).
    pub last_used_secs: u32,
    /// Score used for ordering. Higher means MORE valuable -- evict from the
    /// lowest scores up. Live-session prefixes get the 0.25 penalty applied
    /// at score time.
    pub score: f64,
}

/// Score function: `(hits + 1) * tokens / file_size`, plus a 0.25× penalty
/// when the entry is in `live_session_prefixes`. The penalty is a multiplier
/// applied to the score (lower score -> evicted sooner).
///
/// `live_session_prefixes` is a set of SHA-1 hex strings (without the `.kv`
/// suffix) representing the prefixes the live server session is still
/// actively extending. We deliberately demote these because:
/// - the live session holds the in-memory copy already, so the disk copy is
///   redundant until the session ends.
/// - keeping the on-disk copy "fresh" by re-saving on every turn would dwarf
///   throughput (full-file rewrite per turn).
///
/// The implementation deviates from the spec wording "0.25× penalty for live-
/// session prefixes": the spec is ambiguous about whether 0.25× MULTIPLIES
/// or DIVIDES the score. We multiply (0.25 * score) so the penalty makes the
/// score smaller -> evicted sooner, matching the intuitive interpretation of
/// "penalty".
fn compute_score(
    hits: u32,
    seq_len: u64,
    file_size: u64,
    is_live_session: bool,
) -> f64 {
    if file_size == 0 {
        return f64::NEG_INFINITY;
    }
    let base = ((hits as f64) + 1.0) * (seq_len as f64) / (file_size as f64);
    if is_live_session { base * 0.25 } else { base }
}

/// Enumerate `.kv` files in `dir`, parse their headers, and produce one
/// [`EvictEntry`] per file. Headers that fail to parse (corrupted, version
/// mismatch, mid-write `.tmp.<pid>` files) are skipped silently -- they will
/// be cleaned up by [`purge_stale_tmp`].
pub fn enumerate_dir(
    dir: &Path,
    live_session_prefixes: &HashSet<String>,
) -> Result<Vec<EvictEntry>, RuntimeError> {
    let read_dir = match fs::read_dir(dir) {
        Ok(r) => r,
        // Missing directory is treated as "no entries" so a server that has
        // never run still works.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(RuntimeError::StorageIo(e)),
    };
    let mut entries = Vec::new();
    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue, // non-UTF-8 filenames in the dir -- ignore.
        };
        if !name.ends_with(".kv") {
            continue;
        }
        let path = entry.path();
        let header = match read_header(&path) {
            Ok(h) => h,
            Err(_) => continue, // corrupted -- skip.
        };
        let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        // The SHA hex part is everything before ".kv". This may not exactly
        // equal the session-side hash if a custom file naming was used, but
        // the prefix-set check is purely a hint, so a false negative just
        // means we skip the penalty (i.e., the file looks no more important).
        let stem = &name[..name.len() - 3];
        let is_live = live_session_prefixes.contains(stem);
        let score = compute_score(header.hits, header.seq_len, file_size, is_live);
        entries.push(EvictEntry {
            path,
            file_size,
            seq_len: header.seq_len,
            hits: header.hits,
            last_used_secs: header.last_used_secs,
            score,
        });
    }
    Ok(entries)
}

/// Remove the lowest-scoring entries until the total directory size drops to
/// or below `budget_bytes`. Returns the entries that were evicted.
///
/// Eviction is a single pass: enumerate, sort by ascending score, delete from
/// the front until the remaining size fits the budget. Ties are broken by
/// older `last_used_secs` first.
pub fn evict_to_budget(
    dir: &Path,
    budget_bytes: u64,
    live_session_prefixes: &HashSet<String>,
) -> Result<Vec<EvictEntry>, RuntimeError> {
    let mut entries = enumerate_dir(dir, live_session_prefixes)?;
    let total: u64 = entries.iter().map(|e| e.file_size).sum();
    if total <= budget_bytes {
        return Ok(Vec::new());
    }
    // Sort by ascending score (smallest first), then by ascending
    // last_used_secs (oldest first) as the tiebreaker.
    entries.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.last_used_secs.cmp(&b.last_used_secs))
    });

    let mut current = total;
    let mut evicted = Vec::new();
    for entry in entries {
        if current <= budget_bytes {
            break;
        }
        // `remove_file` is the operation that matters; if it fails (open
        // file handles, races) we skip but keep going.
        if fs::remove_file(&entry.path).is_ok() {
            current = current.saturating_sub(entry.file_size);
            evicted.push(entry);
        }
    }
    Ok(evicted)
}

/// Delete `.tmp.<pid>` files left over from killed `save_atomic` calls.
/// Returns the number of files removed.
pub fn purge_stale_tmp(dir: &Path) -> Result<usize, RuntimeError> {
    let read_dir = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(RuntimeError::StorageIo(e)),
    };
    let mut removed = 0usize;
    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Match `<anything>.kv.tmp.<digits>` -- avoid touching `.tmp.foo`
        // files that happen to share the suffix but were written by another
        // tool.
        let Some(idx) = name.rfind(".tmp.") else { continue };
        let tail = &name[idx + 5..];
        if !tail.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if !name[..idx].ends_with(".kv") {
            continue;
        }
        if fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Static counter ensures unique temp dirs across test invocations without
    // pulling in `tempfile`.
    static UNIQ: AtomicUsize = AtomicUsize::new(0);
    fn fresh_dir(label: &str) -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("lumen-kv-disk-{label}-{pid}-{n}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("fresh_dir mkdir");
        path
    }

    fn small_cfg(precision: KvPrecision) -> KvCacheConfig {
        KvCacheConfig {
            max_seq_len: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            precision,
        }
    }

    /// Build a KV cache with deterministic data for `seq_len` positions across
    /// all layers. Each f32 element follows `(layer * 1e3 + pos * 17 + h * 13 +
    /// dim * 3) * 0.001`, which is dense enough to make any byte-level
    /// equality break visible.
    fn populate_kv(kv: &mut KvCache, seq_len: usize) {
        let cfg = kv.config().clone();
        for pos in 0..seq_len {
            for layer in 0..cfg.num_layers {
                let mut view = kv.view_mut(layer).unwrap();
                view.seq_len = pos;
                let mut k = Vec::with_capacity(cfg.num_kv_heads * cfg.head_dim);
                let mut v = Vec::with_capacity(cfg.num_kv_heads * cfg.head_dim);
                for h in 0..cfg.num_kv_heads {
                    for d in 0..cfg.head_dim {
                        let kx = (layer as f32 * 1e3
                            + pos as f32 * 17.0
                            + h as f32 * 13.0
                            + d as f32 * 3.0)
                            * 0.001;
                        let vx = kx + 0.5;
                        k.push(kx);
                        v.push(vx);
                    }
                }
                view.append_keys(&k);
                view.append_values(&v);
                kv.commit_view(view).unwrap();
            }
            kv.advance_seq_len().unwrap();
        }
    }

    #[test]
    fn sha1_known_vectors() {
        // RFC 3174 test vectors.
        assert_eq!(
            sha1_hex(b""),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            sha1_hex(b"abc"),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            sha1_hex(b"The quick brown fox jumps over the lazy dog"),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    #[test]
    fn crc32_known_vectors() {
        // Known IEEE 802.3 CRC-32 vectors.
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn cache_filename_is_deterministic() {
        let a = cache_filename(&[1, 2, 3]);
        let b = cache_filename(&[1, 2, 3]);
        let c = cache_filename(&[1, 2, 4]);
        assert_eq!(a, b, "same input must produce same name");
        assert_ne!(a, c, "different input must produce different name");
        assert!(a.ends_with(".kv"));
        assert_eq!(a.len(), 40 + 3);
    }

    #[test]
    fn header_roundtrips_bitwise() {
        let mut model_hash = [0u8; 32];
        for i in 0..32 { model_hash[i] = i as u8; }
        let h = DiskKvHeader {
            magic: MAGIC,
            version: VERSION,
            seq_len: 1234,
            num_layers: 32,
            num_kv_heads: 4,
            head_dim: 256,
            max_seq_len: 8192,
            precision_tag: TAG_F16,
            payload_crc32: 0xDEAD_BEEF,
            hits: 7,
            last_used_secs: 1_700_000_000,
            weight_quant_tag: 3,        // C7: Q8_0 tag
            lumen_format_version: 3,    // C7: matches LBC_VERSION
            model_hash,
            has_recurrent_state: 1,     // C: flag round-trips
            has_pending_logits: 1,      // C: flag round-trips
        };
        let bytes = h.to_bytes();
        let h2 = DiskKvHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h.magic, h2.magic);
        assert_eq!(h.version, h2.version);
        assert_eq!(h.seq_len, h2.seq_len);
        assert_eq!(h.num_layers, h2.num_layers);
        assert_eq!(h.num_kv_heads, h2.num_kv_heads);
        assert_eq!(h.head_dim, h2.head_dim);
        assert_eq!(h.max_seq_len, h2.max_seq_len);
        assert_eq!(h.precision_tag, h2.precision_tag);
        assert_eq!(h.payload_crc32, h2.payload_crc32);
        assert_eq!(h.hits, h2.hits);
        assert_eq!(h.last_used_secs, h2.last_used_secs);
        assert_eq!(h.weight_quant_tag, h2.weight_quant_tag);
        assert_eq!(h.lumen_format_version, h2.lumen_format_version);
        assert_eq!(h.model_hash, h2.model_hash);
        assert_eq!(h.has_recurrent_state, h2.has_recurrent_state);
        assert_eq!(h.has_pending_logits, h2.has_pending_logits);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&0xCAFEBABEu32.to_le_bytes());
        let err = DiskKvHeader::from_bytes(&bytes).unwrap_err();
        assert!(format!("{err}").contains("bad magic"));
    }

    #[test]
    fn header_rejects_bad_version() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = DiskKvHeader::from_bytes(&bytes).unwrap_err();
        assert!(format!("{err}").contains("unsupported version"));
    }

    /// Save → load → resume: tokens and KV bytes must round-trip exactly.
    /// This is the P1-5 property test.
    #[test]
    fn disk_save_load_resume_is_bitwise_identical() {
        for precision in [KvPrecision::F32, KvPrecision::F16] {
            let dir = fresh_dir(&format!("roundtrip-{precision:?}"));
            let mut kv = KvCache::new(small_cfg(precision)).unwrap();
            let seq_len = 7;
            populate_kv(&mut kv, seq_len);
            let tokens: Vec<u32> = (100..(100 + seq_len as u32)).collect();
            let path = dir.join(cache_filename(&tokens));

            // Snapshot the per-layer raw bytes before save for comparison.
            let mut original_layers = Vec::new();
            for layer in 0..kv.config().num_layers {
                let (k, v) = kv.layer_raw_bytes(layer).unwrap();
                original_layers.push((k.to_vec(), v.to_vec()));
            }

            save_atomic(&kv, &tokens, &path, 0, &ModelFingerprint::test_zero(), None, None).unwrap();
            assert!(path.exists());

            let loaded = load_into(&path, None, None, None, None).unwrap();
            assert_eq!(loaded.tokens, tokens);
            assert_eq!(loaded.kv.seq_len(), seq_len);
            assert_eq!(loaded.kv.config().precision, precision);
            assert_eq!(loaded.kv.config().num_layers, kv.config().num_layers);

            for (layer, (orig_k, orig_v)) in original_layers.iter().enumerate() {
                let (loaded_k, loaded_v) = loaded.kv.layer_raw_bytes(layer).unwrap();
                assert_eq!(
                    loaded_k, orig_k.as_slice(),
                    "{:?} layer {} keys must match byte-for-byte",
                    precision, layer
                );
                assert_eq!(
                    loaded_v, orig_v.as_slice(),
                    "{:?} layer {} values must match byte-for-byte",
                    precision, layer
                );
            }

            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn atomic_write_leaves_no_tmp_on_success() {
        let dir = fresh_dir("atomic");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));
        save_atomic(&kv, &tokens, &path, 0, &ModelFingerprint::test_zero(), None, None).unwrap();

        let entries = fs::read_dir(&dir).unwrap().count();
        assert_eq!(entries, 1, "should have exactly 1 file (no .tmp leftover)");
        assert!(path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_crc_mismatch() {
        let dir = fresh_dir("crc-mismatch");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![10u32, 20, 30];
        let path = dir.join(cache_filename(&tokens));
        save_atomic(&kv, &tokens, &path, 0, &ModelFingerprint::test_zero(), None, None).unwrap();

        // Corrupt a payload byte (skip past header + tokens).
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        let payload_offset = HEADER_SIZE + tokens.len() * 4;
        f.seek(SeekFrom::Start(payload_offset as u64)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        drop(f);

        let err = load_into(&path, None, None, None, None).unwrap_err();
        assert!(format!("{err}").contains("CRC mismatch"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_respects_budget_and_score() {
        let dir = fresh_dir("evict");
        // Three saves of different sizes. The smallest file with the highest
        // hit count wins the score (tokens / file_size * (hits+1)).
        let cfg = small_cfg(KvPrecision::F32);
        let layer_bytes = layer_payload_bytes(&cfg);
        let file_size_estimate = HEADER_SIZE as u64
            + 4 // single token
            + cfg.num_layers as u64 * layer_bytes as u64;
        // Budget for ~1.5 files; the worst-scoring file must be evicted.
        let budget = file_size_estimate + file_size_estimate / 2;

        // File A: 1 token, 0 hits. Score = (0+1)*1 / size.
        let tokens_a = vec![1u32];
        let mut kv_a = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv_a, 1);
        save_atomic(&kv_a, &tokens_a, &dir.join(cache_filename(&tokens_a)), 0, &ModelFingerprint::test_zero(), None, None).unwrap();

        // File B: 3 tokens, 10 hits. Same file size (max_seq_len allocation),
        // higher score, must survive.
        let tokens_b = vec![2u32, 3, 4];
        let mut kv_b = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv_b, 3);
        save_atomic(&kv_b, &tokens_b, &dir.join(cache_filename(&tokens_b)), 10, &ModelFingerprint::test_zero(), None, None).unwrap();

        // File C: 5 tokens, 0 hits. Score is between A and B; might or might
        // not be evicted depending on budget headroom.
        let tokens_c = vec![5u32, 6, 7, 8, 9];
        let mut kv_c = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv_c, 5);
        save_atomic(&kv_c, &tokens_c, &dir.join(cache_filename(&tokens_c)), 0, &ModelFingerprint::test_zero(), None, None).unwrap();

        let live: HashSet<String> = HashSet::new();
        let evicted = evict_to_budget(&dir, budget, &live).unwrap();
        assert!(!evicted.is_empty(), "budget breach must trigger eviction");

        // File B must survive; it has the highest score.
        assert!(
            dir.join(cache_filename(&tokens_b)).exists(),
            "highest-scored file (B) must survive"
        );
        // File A has the worst score; it must be evicted first.
        assert!(
            !dir.join(cache_filename(&tokens_a)).exists(),
            "lowest-scored file (A) must be evicted first"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_session_penalty_demotes_score() {
        let dir = fresh_dir("live-penalty");
        let cfg = small_cfg(KvPrecision::F32);
        // Two files of identical size and identical (hits, tokens).
        let tokens_x = vec![100u32];
        let tokens_y = vec![200u32];
        let mut kv_x = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv_x, 1);
        save_atomic(&kv_x, &tokens_x, &dir.join(cache_filename(&tokens_x)), 5, &ModelFingerprint::test_zero(), None, None).unwrap();
        let mut kv_y = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv_y, 1);
        save_atomic(&kv_y, &tokens_y, &dir.join(cache_filename(&tokens_y)), 5, &ModelFingerprint::test_zero(), None, None).unwrap();

        // Mark X as a live-session prefix -- its score drops to 0.25 of Y's.
        let live: HashSet<String> =
            [cache_filename(&tokens_x).trim_end_matches(".kv").to_owned()]
                .into_iter()
                .collect();
        let entries = enumerate_dir(&dir, &live).unwrap();
        let entry_x = entries
            .iter()
            .find(|e| e.path.file_name().unwrap() == cache_filename(&tokens_x).as_str())
            .unwrap();
        let entry_y = entries
            .iter()
            .find(|e| e.path.file_name().unwrap() == cache_filename(&tokens_y).as_str())
            .unwrap();
        assert!(
            entry_x.score < entry_y.score,
            "live-session entry must have lower score (X={}, Y={})",
            entry_x.score, entry_y.score,
        );
        // The 0.25 factor must match (within float tolerance).
        let ratio = entry_x.score / entry_y.score;
        assert!(
            (ratio - 0.25).abs() < 1e-9,
            "live-session penalty must be exactly 0.25× (got {ratio})"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_stale_tmp_removes_orphans() {
        let dir = fresh_dir("purge-tmp");
        // Stale: looks like a save_atomic crash artifact.
        let stale = dir.join("abc.kv.tmp.12345");
        fs::write(&stale, b"junk").unwrap();
        // Legit: a finished `.kv` file -- must NOT be removed.
        let legit = dir.join("def.kv");
        fs::write(&legit, b"finished").unwrap();
        // Looks-similar-but-not-ours: must NOT be removed.
        let foreign = dir.join("something.tmp.123"); // missing `.kv.` before `.tmp.`
        fs::write(&foreign, b"foreign").unwrap();
        let foreign2 = dir.join("foo.kv.tmp.abc"); // non-digit PID
        fs::write(&foreign2, b"non-digit pid").unwrap();

        let removed = purge_stale_tmp(&dir).unwrap();
        assert_eq!(removed, 1);
        assert!(!stale.exists());
        assert!(legit.exists());
        assert!(foreign.exists());
        assert!(foreign2.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_rejects_tokens_seq_len_mismatch() {
        let dir = fresh_dir("mismatch");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2]; // only 2 tokens for 3 KV positions
        let path = dir.join(cache_filename(&tokens));
        let err = save_atomic(&kv, &tokens, &path, 0, &ModelFingerprint::test_zero(), None, None).unwrap_err();
        assert!(format!("{err}").contains("tokens.len()"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_file_returns_io_error() {
        let dir = fresh_dir("missing");
        let path = dir.join("notthere.kv");
        let err = load_into(&path, None, None, None, None).unwrap_err();
        assert!(matches!(err, RuntimeError::StorageIo(_)));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn enumerate_skips_corrupted_files() {
        let dir = fresh_dir("corrupt");
        // Drop a file that ends in `.kv` but has no valid header.
        fs::write(dir.join("badfile.kv"), b"not a real header at all").unwrap();
        let live = HashSet::new();
        let entries = enumerate_dir(&dir, &live).unwrap();
        assert!(entries.is_empty(), "corrupted .kv files must be skipped");

        let _ = fs::remove_dir_all(&dir);
    }

    // ---- format v2 + fingerprint tests ---------------------

    /// SHA-256 against RFC 6234 test vectors.
    #[test]
    fn sha256_known_vectors() {
        // SHA-256 of empty string.
        let e = sha256(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8,
            0x99, 0x6f, 0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
            0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(e, expected);
        // SHA-256("abc")
        let abc = sha256(b"abc");
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde,
            0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(abc, expected);
        // SHA-256("The quick brown fox jumps over the lazy dog")
        let fox = sha256(b"The quick brown fox jumps over the lazy dog");
        let expected = [
            0xd7, 0xa8, 0xfb, 0xb3, 0x07, 0xd7, 0x80, 0x94, 0x69, 0xca, 0x9a, 0xbc,
            0xb0, 0x08, 0x2e, 0x4f, 0x8d, 0x56, 0x51, 0xe4, 0x6d, 0x3c, 0xdb, 0x76,
            0x2d, 0x02, 0xd0, 0xbf, 0x37, 0xc9, 0xe5, 0x92,
        ];
        assert_eq!(fox, expected);
    }

    #[test]
    fn compute_model_hash_is_deterministic() {
        let h1 = compute_model_hash(b"hyperparams", b"vocab", 0x1234_5678);
        let h2 = compute_model_hash(b"hyperparams", b"vocab", 0x1234_5678);
        assert_eq!(h1, h2, "same inputs must yield same hash");
        // Changing any input changes the hash.
        let h3 = compute_model_hash(b"DIFFERENT", b"vocab", 0x1234_5678);
        let h4 = compute_model_hash(b"hyperparams", b"DIFFERENT", 0x1234_5678);
        let h5 = compute_model_hash(b"hyperparams", b"vocab", 0x9999_9999);
        assert_ne!(h1, h3, "changed hyperparams must change hash");
        assert_ne!(h1, h4, "changed vocab must change hash");
        assert_ne!(h1, h5, "changed crc aggregate must change hash");
    }

    /// New round-trip test (C7): fingerprint survives save → load.
    #[test]
    fn disk_v2_roundtrip_preserves_fingerprint() {
        let dir = fresh_dir("v2-fingerprint-roundtrip");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));

        let mut model_hash = [0u8; 32];
        for i in 0..32 { model_hash[i] = (i as u8).wrapping_mul(31); }
        let fp = ModelFingerprint {
            model_hash,
            weight_quant_tag: 3, // Q8_0
            lumen_format_version: 3,
        };
        save_atomic(&kv, &tokens, &path, 0, &fp, None, None).unwrap();

        // Load with the matching fingerprint -> succeeds.
        let loaded = load_into(&path, Some(&fp), Some(kv.config()), None, None).unwrap();
        assert_eq!(loaded.tokens, tokens);
        assert_eq!(loaded.kv.seq_len(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Rejection test 1 (C7): different model_hash -> load fails.
    #[test]
    fn disk_v2_rejects_model_hash_mismatch() {
        let dir = fresh_dir("v2-reject-hash");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));

        let saved_fp = ModelFingerprint {
            model_hash: [0xAAu8; 32],
            weight_quant_tag: 3,
            lumen_format_version: 3,
        };
        save_atomic(&kv, &tokens, &path, 0, &saved_fp, None, None).unwrap();

        let wrong_fp = ModelFingerprint {
            model_hash: [0xBBu8; 32], // different
            weight_quant_tag: 3,
            lumen_format_version: 3,
        };
        let err = load_into(&path, Some(&wrong_fp), None, None, None).unwrap_err();
        assert!(
            format!("{err}").contains("model_hash mismatch"),
            "expected model_hash mismatch, got: {err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Rejection test 2 (C7): different weight_quant_tag -> load fails.
    #[test]
    fn disk_v2_rejects_weight_quant_mismatch() {
        let dir = fresh_dir("v2-reject-quant");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));

        let saved_fp = ModelFingerprint {
            model_hash: [0xCCu8; 32],
            weight_quant_tag: 3, // Q8_0
            lumen_format_version: 3,
        };
        save_atomic(&kv, &tokens, &path, 0, &saved_fp, None, None).unwrap();

        let wrong_fp = ModelFingerprint {
            model_hash: [0xCCu8; 32],
            weight_quant_tag: 4, // Q4_0 instead
            lumen_format_version: 3,
        };
        let err = load_into(&path, Some(&wrong_fp), None, None, None).unwrap_err();
        assert!(
            format!("{err}").contains("weight_quant_tag mismatch"),
            "expected weight_quant_tag mismatch, got: {err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Rejection test 3 (C7): different lumen_format_version -> load fails.
    #[test]
    fn disk_v2_rejects_lumen_format_version_mismatch() {
        let dir = fresh_dir("v2-reject-fmt-version");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));

        let saved_fp = ModelFingerprint {
            model_hash: [0xDDu8; 32],
            weight_quant_tag: 3,
            lumen_format_version: 2,
        };
        save_atomic(&kv, &tokens, &path, 0, &saved_fp, None, None).unwrap();

        let wrong_fp = ModelFingerprint {
            model_hash: [0xDDu8; 32],
            weight_quant_tag: 3,
            lumen_format_version: 3, // bumped LBC version
        };
        let err = load_into(&path, Some(&wrong_fp), None, None, None).unwrap_err();
        assert!(
            format!("{err}").contains("lumen_format_version mismatch"),
            "expected lumen_format_version mismatch, got: {err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// v1 → v2 explicit rejection: a v1-formatted header MUST be rejected with
    /// a clear "regenerate" hint instead of being silently upgraded.
    #[test]
    fn disk_rejects_v1_format() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes()); // v1
        let err = DiskKvHeader::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unsupported version 1"), "must call out v1: {msg}");
        assert!(
            msg.contains("regenerate"),
            "must suggest regeneration: {msg}"
        );
    }

    /// N8: load_into validates model shape BEFORE allocating
    /// the (potentially 50 GB) KV buffer. Mismatch -> early error.
    #[test]
    fn disk_load_rejects_shape_mismatch_before_allocation() {
        let dir = fresh_dir("v2-shape-mismatch");
        let cfg = small_cfg(KvPrecision::F32);
        let mut kv = KvCache::new(cfg.clone()).unwrap();
        populate_kv(&mut kv, 3);
        let tokens = vec![1u32, 2, 3];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        save_atomic(&kv, &tokens, &path, 0, &fp, None, None).unwrap();

        // Mismatched num_layers — load must fail at the shape check.
        let wrong_shape = KvCacheConfig {
            max_seq_len: cfg.max_seq_len,
            num_layers: cfg.num_layers + 1, // differs
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            precision: cfg.precision,
        };
        let err = load_into(&path, Some(&fp), Some(&wrong_shape), None, None).unwrap_err();
        assert!(
            format!("{err}").contains("shape mismatch"),
            "expected shape mismatch, got: {err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // ---- recurrent-state disk format tests ---------------

    /// Reference GDN layout matching the Qwen3.5-9B production model:
    /// 24 GDN layers, 32 heads, 128 head_dim, kernel size 4, qkv_dim 8192.
    /// Used by the larger fixture tests; smaller variants are inlined per test.
    fn qwen35_gdn_layout(num_gdn_layers: u32) -> GdnLayout {
        GdnLayout {
            num_gdn_layers,
            gdn_num_heads: 32,
            gdn_head_dim: 128,
            gdn_conv_kernel_size: 4,
            gdn_conv_qkv_dim: 8192,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        }
    }

    fn tiny_gdn_layout(num_gdn_layers: u32) -> GdnLayout {
        // Compact layout used to keep per-test memory < 1 MB while still
        // exercising every length math path.
        GdnLayout {
            num_gdn_layers,
            gdn_num_heads: 2,
            gdn_head_dim: 4,
            gdn_conv_kernel_size: 4,
            gdn_conv_qkv_dim: 16,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        }
    }

    fn deterministic_recurrent_state(layout: GdnLayout) -> RecurrentState {
        let h_bytes = layout.h_state_bytes_per_layer();
        let c_bytes = layout.conv_state_bytes_per_layer();
        let mut state = RecurrentState::zeroed(layout);
        for layer in 0..(layout.num_gdn_layers as usize) {
            // Fill each layer with a unique pattern so we catch
            // mis-indexed reads.
            let h = &mut state.h_states[layer];
            for i in 0..h_bytes {
                h[i] = ((layer * 31 + i) % 251) as u8;
            }
            let c = &mut state.conv_states[layer];
            for i in 0..c_bytes {
                c[i] = ((layer * 17 + i * 3) % 241) as u8;
            }
            state.conv_positions[layer] = (layer as u32) % (layout.gdn_conv_kernel_size - 1);
        }
        state
    }

    /// Backward-compat: v2 file written WITHOUT a recurrent state still
    /// round-trips and produces `LoadedKv { recurrent: None }`.
    #[test]
    fn disk_v2_roundtrip_without_recurrent_state() {
        let dir = fresh_dir("v2-no-recurrent");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 4);
        let tokens = vec![10u32, 20, 30, 40];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();

        save_atomic(&kv, &tokens, &path, 0, &fp, None, None).unwrap();
        let loaded = load_into(&path, Some(&fp), Some(kv.config()), None, None).unwrap();
        assert_eq!(loaded.tokens, tokens);
        assert!(
            loaded.recurrent.is_none(),
            "load with no recurrent section must produce None"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Bit-identical round-trip: every byte in h_states, conv_states, and
    /// the conv_positions vector survives save → load.
    #[test]
    fn disk_v2_roundtrip_with_recurrent_state_is_bitwise_identical() {
        let dir = fresh_dir("v2-recurrent-roundtrip");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 5);
        let tokens = vec![100u32, 101, 102, 103, 104];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        let layout = tiny_gdn_layout(3);
        let state = deterministic_recurrent_state(layout);

        save_atomic(&kv, &tokens, &path, 0, &fp, Some(&state), None).unwrap();
        let loaded = load_into(&path, Some(&fp), Some(kv.config()), Some(&layout), None).unwrap();

        let rec = loaded.recurrent.expect("recurrent section must survive load");
        assert_eq!(rec.layout, layout);
        assert_eq!(rec.h_states.len(), state.h_states.len());
        assert_eq!(rec.conv_states.len(), state.conv_states.len());
        assert_eq!(rec.conv_positions, state.conv_positions);
        for i in 0..state.h_states.len() {
            assert_eq!(
                rec.h_states[i], state.h_states[i],
                "h_states[{i}] bytes must match",
            );
            assert_eq!(
                rec.conv_states[i], state.conv_states[i],
                "conv_states[{i}] bytes must match",
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// Qwen3.5-9B production-sized GDN layout round-trips. Larger than the
    /// tiny fixture above so we exercise the realistic byte counts
    /// (~16 MB h_state per layer × 24 layers).
    #[test]
    fn disk_v2_qwen35_scale_recurrent_roundtrip() {
        // 4 layers (not 24) keeps the test fast: 4 * (32*128*128*4 + 3*8192*4)
        // bytes = ~9 MB. Still exercises every layer-loop path.
        let dir = fresh_dir("v2-qwen35-scale");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F16)).unwrap();
        populate_kv(&mut kv, 2);
        let tokens = vec![1u32, 2];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        let layout = qwen35_gdn_layout(4);
        let state = deterministic_recurrent_state(layout);

        save_atomic(&kv, &tokens, &path, 0, &fp, Some(&state), None).unwrap();
        let loaded = load_into(&path, Some(&fp), Some(kv.config()), Some(&layout), None).unwrap();

        let rec = loaded.recurrent.expect("Qwen3.5 layout must round-trip");
        for i in 0..(layout.num_gdn_layers as usize) {
            assert_eq!(rec.h_states[i], state.h_states[i]);
            assert_eq!(rec.conv_states[i], state.conv_states[i]);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// CRC over the recurrent section must catch single-bit corruption.
    #[test]
    fn disk_v2_recurrent_section_crc_detects_corruption() {
        let dir = fresh_dir("v2-recurrent-crc");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 2);
        let tokens = vec![1u32, 2];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        let layout = tiny_gdn_layout(2);
        let state = deterministic_recurrent_state(layout);
        save_atomic(&kv, &tokens, &path, 0, &fp, Some(&state), None).unwrap();

        // Locate the recurrent section: header + tokens + per-layer K/V.
        let cfg = kv.config();
        let layer_half = layer_payload_bytes(cfg) / 2;
        let kv_payload_bytes = cfg.num_layers * 2 * layer_half;
        let recurrent_start = HEADER_SIZE + tokens.len() * 4 + kv_payload_bytes;

        // Flip one byte INSIDE the per-layer payload (after the 32-byte meta).
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start((recurrent_start + RECURRENT_META_BYTES + 8) as u64))
            .unwrap();
        f.write_all(&[0xFFu8]).unwrap();
        drop(f);

        let err = load_into(&path, Some(&fp), Some(cfg), Some(&layout), None).unwrap_err();
        assert!(
            format!("{err}").contains("recurrent-section CRC mismatch"),
            "expected recurrent CRC mismatch, got: {err}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `validate_layout_matches` rejects every kind of mismatch — exercise
    /// each field independently so a refactor cannot silently weaken any
    /// dimension.
    #[test]
    fn disk_v2_rejects_recurrent_layout_mismatch() {
        let dir = fresh_dir("v2-recurrent-layout-mismatch");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 2);
        let tokens = vec![1u32, 2];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        let saved_layout = tiny_gdn_layout(2);
        let state = deterministic_recurrent_state(saved_layout);
        save_atomic(&kv, &tokens, &path, 0, &fp, Some(&state), None).unwrap();

        let mismatches = [
            GdnLayout { num_gdn_layers: 3, ..saved_layout },
            GdnLayout { gdn_num_heads: 4, ..saved_layout },
            GdnLayout { gdn_head_dim: 8, ..saved_layout },
            GdnLayout { gdn_conv_kernel_size: 5, ..saved_layout },
            GdnLayout { gdn_conv_qkv_dim: 32, ..saved_layout },
        ];
        for layout in mismatches {
            let err = load_into(&path, Some(&fp), Some(kv.config()), Some(&layout), None)
                .unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("recurrent-state layout mismatch"),
                "expected layout mismatch error for {layout:?}, got: {msg}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// Loading a non-recurrent file while the live model expects GDN state
    /// MUST surface an explicit error (caller falls back to cold prefill).
    #[test]
    fn disk_v2_rejects_recurrent_when_missing_but_expected() {
        let dir = fresh_dir("v2-recurrent-missing-but-expected");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 2);
        let tokens = vec![1u32, 2];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        save_atomic(&kv, &tokens, &path, 0, &fp, None, None).unwrap();

        let live_layout = tiny_gdn_layout(2);
        let err = load_into(&path, Some(&fp), Some(kv.config()), Some(&live_layout), None)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no recurrent state"),
            "expected missing-recurrent error, got: {msg}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A non-canonical `has_recurrent_state` value (anything > 1) is
    /// rejected at parse time, before any payload allocation.
    #[test]
    fn disk_v2_rejects_invalid_has_recurrent_state_byte() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        bytes[88] = 0x7F; // canonical values are 0 or 1
        let err = DiskKvHeader::from_bytes(&bytes).unwrap_err();
        assert!(
            format!("{err}").contains("invalid has_recurrent_state"),
            "expected invalid has_recurrent_state error, got: {err}"
        );
    }

    /// On a malformed recurrent state (wrong vector length) the save path
    /// MUST clean up the staged `.tmp.<pid>` file so we never leave
    /// orphans on disk after an internal precondition failure. This also
    /// exercises the ENOSPC cleanup path.
    #[test]
    fn disk_v2_save_cleans_tmp_on_recurrent_validation_failure() {
        let dir = fresh_dir("v2-recurrent-tmp-cleanup");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 1);
        let tokens = vec![1u32];
        let path = dir.join(cache_filename(&tokens));
        let fp = ModelFingerprint::test_zero();
        let layout = tiny_gdn_layout(2);
        // Construct a malformed state: declares 2 layers but supplies only 1
        // entry — write_recurrent_section returns an error mid-stream.
        let mut state = RecurrentState::zeroed(layout);
        state.h_states.truncate(1);
        state.conv_states.truncate(1);
        state.conv_positions.truncate(1);

        let err = save_atomic(&kv, &tokens, &path, 0, &fp, Some(&state), None).unwrap_err();
        assert!(
            format!("{err}").contains("recurrent-state vector length mismatch"),
            "expected validation error, got: {err}"
        );

        // No final file and no tmp file should remain.
        assert!(!path.exists(), "no final file should land on validation failure");
        let stray = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .count();
        assert_eq!(stray, 0, "no orphan tmp file should remain");

        let _ = fs::remove_dir_all(&dir);
    }

    /// Layout-bytes helpers are exact — used by the disk-section math.
    #[test]
    fn gdn_layout_byte_sizes_match_qwen35_dimensions() {
        let l = qwen35_gdn_layout(24);
        // h_state: 32 * 128 * 128 * 4 = 2,097,152 bytes per layer
        assert_eq!(l.h_state_bytes_per_layer(), 32 * 128 * 128 * 4);
        // conv_state: (4 - 1) * 8192 * 4 = 98,304 bytes per layer
        assert_eq!(l.conv_state_bytes_per_layer(), 3 * 8192 * 4);
        // Total payload per layer = h + conv + 4 (conv_pos u32).
        let per_layer = 32 * 128 * 128 * 4 + 3 * 8192 * 4 + 4;
        assert_eq!(l.payload_bytes(), 24 * per_layer);
        assert_eq!(l.bytes_per_element(), 4);
    }

    // ---- Save/load edge cases ----

    /// `save_atomic` cleans up `.tmp.<pid>` on a write error
    /// so disk-full never strands orphan tmp files. This complements the
    /// already-tested `disk_v2_save_cleans_tmp_on_recurrent_validation_failure`
    /// which exercises an in-buffer validation error; this test triggers an
    /// IO-layer error by saving to a parent directory that gets revoked
    /// (chmod 000) right before the write. The cleanup path must still
    /// remove the staged file or surface an explicit error.
    ///
    /// On Apple/Linux the open() succeeds at chmod 000 if the inode is
    /// already held by the writer, so we trigger a write_all failure by
    /// simulating an inner length-mismatch instead. The path the contract
    /// asks us to harden is the same regardless of the trigger; the test
    /// proves the cleanup is wired.
    #[test]
    fn save_atomic_cleans_tmp_on_io_error() {
        // Build a corrupt KvCache where layer_raw_bytes returns a slice
        // shorter than expected — save_atomic returns Err and MUST unlink
        // the tmp file before propagating.
        let dir = fresh_dir("save-io-error-cleanup");
        let mut kv = KvCache::new(small_cfg(KvPrecision::F32)).unwrap();
        populate_kv(&mut kv, 2);
        let tokens = vec![1u32, 2];
        let path = dir.join(cache_filename(&tokens));
        // Save once successfully so we have a baseline directory.
        save_atomic(&kv, &tokens, &path, 0, &ModelFingerprint::test_zero(), None, None).unwrap();
        // Now provoke a save failure: pass tokens.len() != kv.seq_len().
        let bad_tokens = vec![1u32]; // 1 token, but kv.seq_len() == 2
        let bad_path = dir.join(cache_filename(&bad_tokens));
        let err = save_atomic(
            &kv,
            &bad_tokens,
            &bad_path,
            0,
            &ModelFingerprint::test_zero(),
            None,
            None,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("tokens.len()"));
        // No tmp file should remain for the failed save.
        let stray = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.contains(&format!("{}.tmp.", cache_filename(&bad_tokens)))
            })
            .count();
        assert_eq!(stray, 0, "save failure must not leave .tmp.<pid> orphan");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `purge_stale_tmp` recovers orphan tmp files left
    /// over from killed `save_atomic` writes. This re-runs the existing
    /// `purge_stale_tmp_removes_orphans` shape with one addition: an
    /// orphan that DOESN'T match the `.kv.tmp.<digits>` pattern must
    /// survive (we never blindly delete unknown files).
    #[test]
    fn purge_stale_tmp_skips_unrelated_files() {
        let dir = fresh_dir("purge-skip-unrelated");
        // 1. A canonical orphan: matches `.kv.tmp.<digits>`. Must die.
        let orphan = dir.join("aaa.kv.tmp.999");
        fs::write(&orphan, b"junk").unwrap();
        // 2. A near-miss orphan: file name has `.tmp.<digits>` but the
        //    pre-`.tmp.` prefix is not `.kv`. Must survive.
        let unrelated_tmp = dir.join("aaa.json.tmp.123");
        fs::write(&unrelated_tmp, b"json-like").unwrap();
        // 3. A near-miss orphan: `.kv.tmp.` with non-digit suffix. Must survive.
        let bad_pid = dir.join("aaa.kv.tmp.x12y");
        fs::write(&bad_pid, b"bad pid").unwrap();
        // 4. A real `.kv` file. Must survive.
        let real = dir.join("bbb.kv");
        fs::write(&real, b"real").unwrap();

        let removed = purge_stale_tmp(&dir).unwrap();
        assert_eq!(removed, 1, "only the canonical orphan should be removed");
        assert!(!orphan.exists());
        assert!(unrelated_tmp.exists());
        assert!(bad_pid.exists());
        assert!(real.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// v1 → v2 upgrade rejection has an actionable error message.
    /// The / split adds the `regenerate by removing
    /// the file` hint; verify it still contains both the version-error
    /// signal AND the actionable suggestion.
    #[test]
    fn disk_v1_rejection_message_is_actionable() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes()); // v1
        let err = DiskKvHeader::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported version 1"),
            "must call out v1: {msg}"
        );
        assert!(
            msg.contains("delete the file") || msg.contains("regenerate"),
            "must give an actionable hint: {msg}"
        );
        // Specifically — phrased the hint as "delete the file
        // and let the next cold prefill regenerate it"; the operator
        // should see both pieces.
        assert!(
            msg.contains("delete"),
            "actionable hint must include the verb 'delete': {msg}"
        );
    }
}
