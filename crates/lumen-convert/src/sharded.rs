//! Multi-shard GGUF reader.
//!
//! HuggingFace publishes large GGUF models (>50 GB) as a set of `*-NNNNN-of-MMMMM.gguf`
//! shards. Each shard is itself a valid GGUF v3 file: it carries its own header,
//! its own metadata KV block, its own tensor-info section, and its own tensor-data
//! section. Tensors are NOT split across shard boundaries; each tensor lives
//! entirely in exactly one shard.
//!
//! The shards are tied together by three metadata keys (written by the
//! `gguf-split` tool):
//!   - `split.no`             -- u16, the 0-based shard index
//!   - `split.count`          -- u16, total number of shards (constant across siblings)
//!   - `split.tensors.count`  -- u64, total tensor count summed across all shards
//!
//! Reference: GGUF v3 multi-shard split spec (`split.no` / `split.count` /
//! `split.tensors.count` metadata keys).
//!
//! ## What this module does
//!
//! [`ShardedGguf`] discovers the sibling shards from a single shard's path,
//! parses each shard's header, validates consistency (matching `split.count`,
//! contiguous `split.no`, no tensor-name collisions across shards, matching
//! GGUF version + alignment), and exposes a [`ShardedGguf::view()`] that looks
//! like a single merged [`GgufFile`] to downstream consumers (metadata accessors
//! delegate to shard 0; the tensor list is the concatenation across shards).
//!
//! Tensor data is read on demand via [`ShardedGguf::read_tensor_data`], which
//! opens the appropriate shard file with [`std::io::BufReader`] and seeks to the
//! tensor's offset. Peak memory is O(1) per tensor read (matching the
//! single-shard streaming behaviour) -- the full 68 GB of BF16 weights are never
//! loaded simultaneously.
//!
//! ## Backward compatibility
//!
//! For single-shard GGUFs the discovery step is a no-op: [`ShardedGguf::open`]
//! returns a one-shard view that reads identically to the legacy single-file
//! path. This preserves byte-identical conversion for the existing Q8_0 / Q4_0
//! dense and MoE flows.

use crate::convert::ConvertError;
use crate::gguf::{GgufError, GgufFile, GgufTensorInfo, GgufValue};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Virtual address layout
// ---------------------------------------------------------------------------
//
// The merged [`GgufFile`] surfaced by [`ShardedGguf`] exposes per-shard tensor
// offsets through a *virtual* 64-bit address space. The high bits identify the
// shard; the low bits identify the position within that shard.
//
//   bits 63..56  = shard id (0..255)
//   bits 55..0   = absolute byte offset within the shard's file
//
// The merged `data_offset` is set to 0, so `tensor_data_offset(tensor) =
// 0 + tensor.offset = virtual_address` -- callers that go through
// `tensor_data_offset` followed by a `seek` automatically land in the right
// shard when paired with [`MultiShardReader`].
//
// We use 56 bits for the within-shard offset because individual GGUF shards
// in current HuggingFace publications stay well below 2^56 = 64 PiB
// (the largest practical GGUF shard is ~50 GiB before HF tooling complains).
// 8 bits for the shard id supports up to 255 shards -- comfortably above the
// largest current published split count.

/// Bit position of the shard id within a virtual address.
pub(crate) const SHARD_ID_SHIFT: u32 = 56;
/// Mask for the within-shard offset (bits 0..56).
pub(crate) const WITHIN_SHARD_MASK: u64 = (1u64 << SHARD_ID_SHIFT) - 1;
/// Maximum number of shards the virtual-address scheme can address.
pub(crate) const MAX_SHARDS: usize = 256;

/// Compose a virtual address from a shard id and an absolute byte offset
/// within that shard's on-disk file.
pub(crate) fn make_virtual_address(shard_id: u8, abs_offset: u64) -> u64 {
    debug_assert!(abs_offset <= WITHIN_SHARD_MASK,
        "abs_offset {abs_offset} exceeds 2^56 -- shard too large for virtual address scheme");
    ((shard_id as u64) << SHARD_ID_SHIFT) | abs_offset
}

/// Decompose a virtual address into `(shard_id, abs_offset)`.
pub(crate) fn split_virtual_address(addr: u64) -> (u8, u64) {
    let shard_id = (addr >> SHARD_ID_SHIFT) as u8;
    let abs_offset = addr & WITHIN_SHARD_MASK;
    (shard_id, abs_offset)
}

// ---------------------------------------------------------------------------
// Metadata keys -- GGUF v3 split convention
// ---------------------------------------------------------------------------

/// 0-based shard index within the split set.
pub const SPLIT_NO_KEY: &str = "split.no";
/// Total number of shards (constant across all siblings).
pub const SPLIT_COUNT_KEY: &str = "split.count";
/// Total tensor count summed across all shards (sanity-check value).
pub const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to multi-shard handling. Distinct from [`GgufError`] because
/// the failure modes (missing siblings, mismatched counts, name collisions) are
/// cross-shard concerns rather than single-file parse failures.
#[derive(Debug)]
pub enum ShardError {
    /// I/O failure opening or reading a shard file.
    Io(std::io::Error),
    /// A shard file did not parse as a valid GGUF.
    Parse(GgufError),
    /// The shard filename did not match the `*-NNNNN-of-MMMMM.gguf` pattern.
    InvalidShardName(String),
    /// `split.count` differs between shards (e.g. shard 1 says 2 but shard 2 says 3).
    SplitCountMismatch { shard: PathBuf, expected: u16, got: u16 },
    /// The set of `split.no` values is not contiguous 0..N or has duplicates.
    NonContiguousShards { found: Vec<u16>, expected_count: u16 },
    /// Fewer shards were located on disk than `split.count` requires.
    MissingShards { expected: u16, present: u16, looked_for: Vec<PathBuf> },
    /// Two shards declare the same tensor name.
    TensorNameCollision { name: String, first_shard: usize, second_shard: usize },
    /// Two shards declare different GGUF format versions.
    VersionMismatch { shard: PathBuf, expected: u32, got: u32 },
    /// Two shards declare different data-section alignments.
    AlignmentMismatch { shard: PathBuf, expected: u32, got: u32 },
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "shard I/O error: {e}"),
            Self::Parse(e) => write!(f, "shard parse error: {e}"),
            Self::InvalidShardName(n) => write!(
                f, "invalid shard filename {n:?}: expected pattern \"*-NNNNN-of-MMMMM.gguf\""
            ),
            Self::SplitCountMismatch { shard, expected, got } => write!(
                f, "shard {} declares split.count={got} but siblings declare {expected}",
                shard.display()
            ),
            Self::NonContiguousShards { found, expected_count } => write!(
                f, "shard set is non-contiguous: found split.no={:?}, expected 0..{expected_count}", found
            ),
            Self::MissingShards { expected, present, looked_for } => {
                writeln!(f, "expected {expected} shard(s) but only found {present} on disk")?;
                writeln!(f, "  looked for:")?;
                for p in looked_for {
                    writeln!(f, "    {}", p.display())?;
                }
                Ok(())
            }
            Self::TensorNameCollision { name, first_shard, second_shard } => write!(
                f, "tensor name {name:?} appears in both shard {first_shard} and shard {second_shard} \
                    (each tensor must live in exactly one shard)"
            ),
            Self::VersionMismatch { shard, expected, got } => write!(
                f, "shard {} has GGUF version {got} but sibling has {expected}",
                shard.display()
            ),
            Self::AlignmentMismatch { shard, expected, got } => write!(
                f, "shard {} has alignment {got} but sibling has {expected}",
                shard.display()
            ),
        }
    }
}

impl std::error::Error for ShardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<GgufError> for ShardError {
    fn from(e: GgufError) -> Self {
        Self::Parse(e)
    }
}

impl From<ShardError> for ConvertError {
    fn from(e: ShardError) -> Self {
        ConvertError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ShardSpec -- one entry per shard file
// ---------------------------------------------------------------------------

/// One shard file's metadata + its tensor descriptors. Owns the parsed
/// [`GgufFile`] header so each shard's `data_offset` is preserved verbatim
/// (alignment may vary between shards in principle, though the validator
/// rejects mismatches).
struct ShardSpec {
    /// On-disk path of this shard.
    path: PathBuf,
    /// 0-based shard index from `split.no`.
    shard_no: u16,
    /// Parsed header for this shard, including its own tensor info entries.
    file: GgufFile,
}

// ---------------------------------------------------------------------------
// ShardedGguf -- the public view
// ---------------------------------------------------------------------------

/// A multi-shard GGUF model presented as a single unified view.
///
/// Construct via [`ShardedGguf::open`] (auto-discovers siblings from any one
/// shard's path) or [`ShardedGguf::from_paths`] (explicit shard list).
///
/// Use [`Self::merged()`] to get a [`GgufFile`]-shaped view (canonical metadata
/// from shard 0, concatenated tensor list across all shards). Use
/// [`Self::read_tensor_data`] to read a tensor's bytes by name; the correct
/// shard file is opened on demand.
pub struct ShardedGguf {
    shards: Vec<ShardSpec>,
    /// The merged view: shard-0 metadata + concatenated tensor list.
    /// Each tensor's `offset` field is unchanged from its source shard's
    /// declaration (offset is per-shard-data-section).
    merged: GgufFile,
    /// `merged.tensors[i]` lives in shard `tensor_shard[i]`.
    tensor_shard: Vec<u16>,
    /// Quick lookup `name -> index_in_merged.tensors`.
    name_index: HashMap<String, usize>,
}

impl ShardedGguf {
    /// Open a sharded GGUF, given any one shard's path. Sibling shards are
    /// auto-discovered by parsing the `*-NNNNN-of-MMMMM.gguf` filename pattern
    /// and resolving siblings in the same directory.
    ///
    /// For a single-shard file (no `-NNNNN-of-MMMMM` suffix), returns a view
    /// containing only that file.
    pub fn open(any_shard_path: &Path) -> Result<Self, ShardError> {
        let paths = discover_sibling_shards(any_shard_path)?;
        Self::from_paths(&paths)
    }

    /// Open a sharded GGUF from an explicit list of shard paths. The paths can
    /// be in any order; they are sorted by `split.no` after parsing.
    pub fn from_paths(paths: &[PathBuf]) -> Result<Self, ShardError> {
        if paths.is_empty() {
            return Err(ShardError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "from_paths requires at least one shard",
            )));
        }

        // Parse every shard's header.
        let mut specs: Vec<ShardSpec> = Vec::with_capacity(paths.len());
        for p in paths {
            let f = GgufFile::open(p)?;
            // Single-shard files have no `split.no`; default to 0.
            let shard_no = read_u16_metadata(&f, SPLIT_NO_KEY).unwrap_or(0);
            specs.push(ShardSpec { path: p.clone(), shard_no, file: f });
        }

        // Sort by shard_no so we iterate in deterministic order.
        specs.sort_by_key(|s| s.shard_no);

        // Validate.
        validate_shards(&specs)?;

        // Build the merged view.
        let (merged, tensor_shard) = build_merged_view(&specs)?;
        let name_index = merged
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Self {
            shards: specs,
            merged,
            tensor_shard,
            name_index,
        })
    }

    /// The merged metadata + tensor view. Pass this to converter code that
    /// previously took `&GgufFile`.
    ///
    /// Note: the `data_offset` field on the returned [`GgufFile`] reflects
    /// shard-0's data offset. For multi-shard tensors, callers MUST go through
    /// [`Self::read_tensor_data`] / [`Self::tensor_data_absolute_offset`] which
    /// route to the correct shard. The single-shard happy path keeps using the
    /// reader-with-absolute-seek model unchanged.
    pub fn merged(&self) -> &GgufFile {
        &self.merged
    }

    /// Total number of shards in the set.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Total combined byte size of all shard files on disk.
    pub fn total_disk_size(&self) -> Result<u64, std::io::Error> {
        let mut total = 0u64;
        for s in &self.shards {
            total += std::fs::metadata(&s.path)?.len();
        }
        Ok(total)
    }

    /// Iterator over (shard_no, on-disk path) pairs in shard_no order.
    pub fn shard_paths(&self) -> impl Iterator<Item = (u16, &Path)> {
        self.shards.iter().map(|s| (s.shard_no, s.path.as_path()))
    }

    /// Look up a tensor by name in the merged view. Returns the tensor info
    /// plus the shard it lives in.
    pub fn find_tensor_with_shard(&self, name: &str) -> Option<(&GgufTensorInfo, u16)> {
        let idx = *self.name_index.get(name)?;
        Some((&self.merged.tensors[idx], self.tensor_shard[idx]))
    }

    /// Read the raw bytes for a tensor from its owning shard. Opens the shard
    /// file on demand. The tensor must be one of the [`GgufTensorInfo`] entries
    /// reachable via [`Self::merged`].
    pub fn read_tensor_data(&self, tensor: &GgufTensorInfo) -> Result<Vec<u8>, ConvertError> {
        let (shard_id, abs_offset) = split_virtual_address(tensor.offset);
        let shard_idx = shard_id as usize;
        if shard_idx >= self.shards.len() {
            return Err(ConvertError::MissingTensor(format!(
                "{}: virtual address points to shard {shard_idx}, only {} shard(s) loaded",
                tensor.name,
                self.shards.len()
            )));
        }
        let spec = &self.shards[shard_idx];
        let size = tensor.byte_size().unwrap_or(0) as usize;

        let f = std::fs::File::open(&spec.path)?;
        let mut reader = BufReader::new(f);
        reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; size];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Returns the absolute file offset of a tensor's data within its owning
    /// shard. Decoded from the virtual offset baked into the merged tensor info.
    pub fn tensor_data_absolute_offset(&self, tensor: &GgufTensorInfo) -> Result<u64, ConvertError> {
        let (shard_id, abs_offset) = split_virtual_address(tensor.offset);
        if (shard_id as usize) >= self.shards.len() {
            return Err(ConvertError::MissingTensor(tensor.name.clone()));
        }
        Ok(abs_offset)
    }

    /// On-disk path of the shard that owns this tensor.
    pub fn path_for_tensor(&self, tensor: &GgufTensorInfo) -> Result<&Path, ConvertError> {
        let (shard_id, _) = split_virtual_address(tensor.offset);
        let idx = shard_id as usize;
        if idx >= self.shards.len() {
            return Err(ConvertError::MissingTensor(tensor.name.clone()));
        }
        Ok(&self.shards[idx].path)
    }
}

impl std::fmt::Debug for ShardedGguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedGguf")
            .field("shard_count", &self.shards.len())
            .field("tensor_count", &self.merged.tensors.len())
            .field("version", &self.merged.version)
            .field("alignment", &self.merged.alignment)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Sibling discovery
// ---------------------------------------------------------------------------

/// Parse a shard filename of the form `<stem>-NNNNN-of-MMMMM.gguf` and return
/// `(stem, shard_no_1based, total_count)`. Returns `None` if the filename does
/// not match the pattern.
///
/// Shard numbering in filenames is 1-based by convention (00001-of-00002),
/// even though `split.no` in metadata is 0-based.
pub(crate) fn parse_shard_filename(filename: &str) -> Option<(String, u32, u32)> {
    // Strip the .gguf extension.
    let stripped = filename.strip_suffix(".gguf")?;

    // Find the last "-of-" marker.
    let of_pos = stripped.rfind("-of-")?;
    let (head, tail) = stripped.split_at(of_pos);
    let tail = &tail["-of-".len()..];

    // tail must be all digits (the total count).
    if tail.is_empty() || !tail.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let total: u32 = tail.parse().ok()?;

    // head ends with "-NNNNN" -- find the last '-' and check digits.
    let dash = head.rfind('-')?;
    let (stem, num_part) = head.split_at(dash);
    let num_part = &num_part[1..]; // drop the '-'
    if num_part.is_empty() || !num_part.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let shard_no_1based: u32 = num_part.parse().ok()?;

    if shard_no_1based == 0 || shard_no_1based > total {
        return None;
    }

    Some((stem.to_string(), shard_no_1based, total))
}

/// Build the filename of shard N (1-based) given a base stem and total count.
/// The width of the zero-padded numeric fields matches the HuggingFace
/// `gguf-split` convention: 5 digits for index and total.
pub(crate) fn make_shard_filename(stem: &str, shard_no_1based: u32, total: u32) -> String {
    format!("{stem}-{shard_no_1based:05}-of-{total:05}.gguf")
}

/// Given the path to one shard, locate every sibling shard in the same
/// directory.
///
/// Behaviour:
///   - Single-file GGUFs (filename does not match the `*-NNNNN-of-MMMMM.gguf`
///     pattern): returns `[input_path]`.
///   - Multi-shard: enumerates all `<stem>-NNNNN-of-MMMMM.gguf` siblings.
///     Returns an error if any shard 1..total is missing on disk.
pub(crate) fn discover_sibling_shards(path: &Path) -> Result<Vec<PathBuf>, ShardError> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| ShardError::InvalidShardName(path.display().to_string()))?;

    let (stem, _shard_no_1based, total) = match parse_shard_filename(filename) {
        Some(t) => t,
        None => {
            // Not a shard filename -- treat as single-file GGUF.
            return Ok(vec![path.to_path_buf()]);
        }
    };

    let parent = path.parent().unwrap_or_else(|| Path::new(""));

    // Enumerate expected sibling paths.
    let mut siblings: Vec<PathBuf> = (1..=total)
        .map(|i| parent.join(make_shard_filename(&stem, i, total)))
        .collect();

    // Check every expected sibling exists on disk.
    let present: u16 = siblings.iter().filter(|p| p.is_file()).count() as u16;
    if present as u32 != total {
        return Err(ShardError::MissingShards {
            expected: total as u16,
            present,
            looked_for: siblings,
        });
    }

    // Sort by filename for deterministic order (already in order from 1..=total
    // construction, but be safe in case future code shuffles).
    siblings.sort();
    Ok(siblings)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn read_u16_metadata(file: &GgufFile, key: &str) -> Option<u16> {
    let v = file.get_metadata(key)?;
    match v {
        GgufValue::U16(x) => Some(*x),
        GgufValue::U32(x) => u16::try_from(*x).ok(),
        GgufValue::U64(x) => u16::try_from(*x).ok(),
        GgufValue::I32(x) if *x >= 0 => u16::try_from(*x).ok(),
        _ => None,
    }
}

/// Verify the shard set is well-formed:
///   - all shards declare the same `split.count` (if any of them declare it),
///   - the `split.no` values are exactly {0, 1, ..., N-1},
///   - GGUF version is the same across shards,
///   - alignment is the same across shards.
fn validate_shards(specs: &[ShardSpec]) -> Result<(), ShardError> {
    if specs.is_empty() {
        return Ok(());
    }

    let first_version = specs[0].file.version;
    let first_alignment = specs[0].file.alignment;
    let declared_count = read_u16_metadata(&specs[0].file, SPLIT_COUNT_KEY);

    for spec in specs.iter().skip(1) {
        if spec.file.version != first_version {
            return Err(ShardError::VersionMismatch {
                shard: spec.path.clone(),
                expected: first_version,
                got: spec.file.version,
            });
        }
        if spec.file.alignment != first_alignment {
            return Err(ShardError::AlignmentMismatch {
                shard: spec.path.clone(),
                expected: first_alignment,
                got: spec.file.alignment,
            });
        }
        if let (Some(expected), Some(got)) = (declared_count, read_u16_metadata(&spec.file, SPLIT_COUNT_KEY)) {
            if expected != got {
                return Err(ShardError::SplitCountMismatch {
                    shard: spec.path.clone(),
                    expected,
                    got,
                });
            }
        }
    }

    // Contiguity of split.no values 0..N-1.
    // For single-shard sets (specs.len() == 1) this trivially passes.
    let n = specs.len() as u16;
    let declared = declared_count.unwrap_or(n);
    if declared != n {
        return Err(ShardError::MissingShards {
            expected: declared,
            present: n,
            looked_for: specs.iter().map(|s| s.path.clone()).collect(),
        });
    }

    let mut found: Vec<u16> = specs.iter().map(|s| s.shard_no).collect();
    let expected: Vec<u16> = (0..n).collect();
    if found != expected {
        // The set is already sorted by shard_no (we sort in from_paths), so any
        // mismatch means non-contiguous or duplicated values.
        found.sort();
        return Err(ShardError::NonContiguousShards {
            found,
            expected_count: n,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Merging
// ---------------------------------------------------------------------------

/// Build a merged [`GgufFile`] view from a validated, sorted shard list.
///
///   - Metadata is taken from shard 0 (canonical).
///   - The tensor list is the concatenation across shards in shard-0-first order.
///   - Tensor-name collisions across shards trigger [`ShardError::TensorNameCollision`].
///   - Each tensor's `offset` field is rewritten into a virtual address that
///     encodes the owning shard id (high 8 bits) and the absolute byte position
///     within that shard's on-disk file (low 56 bits). The merged
///     `data_offset` is set to `0` so that
///     `tensor_data_offset(t) == t.offset == virtual_address`. A
///     [`MultiShardReader`] interprets these virtual addresses transparently
///     during conversion.
fn build_merged_view(specs: &[ShardSpec]) -> Result<(GgufFile, Vec<u16>), ShardError> {
    let canonical = &specs[0].file;
    if specs.len() > MAX_SHARDS {
        return Err(ShardError::MissingShards {
            expected: specs.len() as u16,
            present: MAX_SHARDS as u16,
            looked_for: vec![],
        });
    }

    // Detect tensor-name collisions across shards.
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (s_idx, spec) in specs.iter().enumerate() {
        for t in &spec.file.tensors {
            if let Some(&prev) = seen.get(&t.name) {
                return Err(ShardError::TensorNameCollision {
                    name: t.name.clone(),
                    first_shard: prev,
                    second_shard: s_idx,
                });
            }
            seen.insert(t.name.clone(), s_idx);
        }
    }

    // Concatenate tensor lists, rewriting each tensor's `offset` into the
    // virtual-address space (see SHARD_ID_SHIFT above). The within-shard
    // absolute offset comes from the source GgufFile's `tensor_data_offset`
    // helper, which knows that shard's own header size + alignment padding.
    let total_tensors: usize = specs.iter().map(|s| s.file.tensors.len()).sum();
    let mut tensors: Vec<GgufTensorInfo> = Vec::with_capacity(total_tensors);
    let mut tensor_shard: Vec<u16> = Vec::with_capacity(total_tensors);
    for (s_idx, spec) in specs.iter().enumerate() {
        for t in &spec.file.tensors {
            let abs_offset = spec.file.tensor_data_offset(t);
            let virtual_offset = make_virtual_address(s_idx as u8, abs_offset);
            let mut t_remap = t.clone();
            t_remap.offset = virtual_offset;
            tensors.push(t_remap);
            tensor_shard.push(s_idx as u16);
        }
    }

    // The merged data_offset is 0: each tensor's `offset` field already encodes
    // its full virtual address. `tensor_data_offset(t)` becomes `0 + t.offset`.
    let merged = GgufFile {
        version: canonical.version,
        metadata: canonical.metadata.clone(),
        tensors,
        data_offset: 0,
        alignment: canonical.alignment,
    };

    Ok((merged, tensor_shard))
}

// ---------------------------------------------------------------------------
// MultiShardReader -- transparent Read + Seek over a sharded GGUF
// ---------------------------------------------------------------------------

/// A reader that implements [`Read`] + [`Seek`] over the union of every
/// shard's on-disk file, dispatching by the virtual addresses encoded in
/// merged tensor offsets.
///
/// Use case: existing converter code does
/// ```ignore
/// reader.seek(SeekFrom::Start(gguf.tensor_data_offset(tensor)))?;
/// reader.read_exact(&mut buf)?;
/// ```
/// When `gguf` is the merged view from [`ShardedGguf::merged`] and `reader`
/// is a [`MultiShardReader`], the virtual address is decoded into
/// `(shard_id, abs_offset)`, the appropriate shard's [`BufReader<File>`] is
/// selected, and the seek/read flow continues as if the model were a single
/// file. This means [`crate::tensor_io::read_tensor_data`] and the
/// architecture converters work unchanged across single-shard and multi-shard
/// inputs.
///
/// The reader holds one open file handle per shard for the duration of the
/// conversion. Each handle is wrapped in a [`BufReader`] with the same default
/// buffer size that `tensor_io` uses on the single-shard path -- there is no
/// new memory cost beyond one file handle per shard.
pub struct MultiShardReader {
    shards: Vec<BufReader<File>>,
    /// The shard currently positioned for the next read (set by `seek`).
    current_shard: u8,
}

impl MultiShardReader {
    /// Open file handles for every shard in `view`.
    pub fn open(view: &ShardedGguf) -> Result<Self, std::io::Error> {
        let mut shards = Vec::with_capacity(view.shards.len());
        for spec in &view.shards {
            let f = File::open(&spec.path)?;
            shards.push(BufReader::new(f));
        }
        Ok(Self { shards, current_shard: 0 })
    }
}

impl Read for MultiShardReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let idx = self.current_shard as usize;
        if idx >= self.shards.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("MultiShardReader: current shard id {idx} out of range"),
            ));
        }
        self.shards[idx].read(buf)
    }
}

impl Seek for MultiShardReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(virtual_addr) => {
                let (shard_id, abs_offset) = split_virtual_address(virtual_addr);
                let idx = shard_id as usize;
                if idx >= self.shards.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "MultiShardReader: virtual address 0x{virtual_addr:016x} targets shard {shard_id}, \
                             but only {} shard(s) are open",
                            self.shards.len()
                        ),
                    ));
                }
                self.current_shard = shard_id;
                self.shards[idx].seek(SeekFrom::Start(abs_offset))?;
                Ok(virtual_addr)
            }
            // SeekFrom::Current / SeekFrom::End are not used by the converter
            // (it always issues absolute seeks before each tensor read). Reject
            // them explicitly so a stray relative seek doesn't silently land
            // in the wrong shard.
            SeekFrom::Current(_) | SeekFrom::End(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "MultiShardReader supports only absolute (SeekFrom::Start) seeks",
            )),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, GgufBuilder};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lumen_sharded_test_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // -- parse_shard_filename --------------------------------------------------

    #[test]
    fn parse_canonical_shard_filename() {
        let (stem, no, total) =
            parse_shard_filename("Qwen_Qwen3.5-MoE-35B-A3B-BF16-00001-of-00002.gguf").unwrap();
        assert_eq!(stem, "Qwen_Qwen3.5-MoE-35B-A3B-BF16");
        assert_eq!(no, 1);
        assert_eq!(total, 2);
    }

    #[test]
    fn parse_shard_filename_second_of_two() {
        let (stem, no, total) =
            parse_shard_filename("model-00002-of-00002.gguf").unwrap();
        assert_eq!(stem, "model");
        assert_eq!(no, 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn parse_shard_filename_rejects_single_file() {
        // Pure single-file names without the -NNNNN-of-MMMMM suffix.
        assert!(parse_shard_filename("model.gguf").is_none());
        assert!(parse_shard_filename("model-Q8_0.gguf").is_none());
        assert!(parse_shard_filename("Qwen_Qwen3.5-9B-Q8_0.gguf").is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_non_gguf_extension() {
        assert!(parse_shard_filename("model-00001-of-00002.bin").is_none());
        assert!(parse_shard_filename("model-00001-of-00002").is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_zero_index() {
        // 1-based numbering: 0 is invalid.
        assert!(parse_shard_filename("model-00000-of-00002.gguf").is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_index_above_total() {
        // 00003-of-00002 is malformed.
        assert!(parse_shard_filename("model-00003-of-00002.gguf").is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_non_digits() {
        assert!(parse_shard_filename("model-abc-of-00002.gguf").is_none());
        assert!(parse_shard_filename("model-00001-of-xxx.gguf").is_none());
    }

    #[test]
    fn make_shard_filename_canonical_padding() {
        assert_eq!(
            make_shard_filename("Qwen_Qwen3.5-MoE-35B-A3B-BF16", 1, 2),
            "Qwen_Qwen3.5-MoE-35B-A3B-BF16-00001-of-00002.gguf"
        );
        assert_eq!(
            make_shard_filename("foo", 7, 12),
            "foo-00007-of-00012.gguf"
        );
    }

    #[test]
    fn shard_filename_roundtrip() {
        let stems = [
            "Qwen_Qwen3.5-MoE-35B-A3B-BF16",
            "Meta-Llama-3.1-70B-Q4_K_M",
            "foo",
            "model-with-dashes.in-stem",
        ];
        for stem in &stems {
            for total in [1u32, 2, 5, 12, 99] {
                for no in 1..=total {
                    let filename = make_shard_filename(stem, no, total);
                    let (parsed_stem, parsed_no, parsed_total) =
                        parse_shard_filename(&filename).expect("must roundtrip");
                    assert_eq!(parsed_stem, *stem, "stem mismatch for {filename}");
                    assert_eq!(parsed_no, no, "no mismatch for {filename}");
                    assert_eq!(parsed_total, total, "total mismatch for {filename}");
                }
            }
        }
    }

    // -- ShardSpec construction helpers --------------------------------------

    /// Build a synthetic 2-shard set with the given tensors split across shards.
    /// Returns the directory containing both shard files.
    fn write_2shard_set(
        dir: &Path,
        stem: &str,
        shard1_tensors: &[(&str, GgmlType, Vec<u64>, Vec<u8>)],
        shard2_tensors: &[(&str, GgmlType, Vec<u64>, Vec<u8>)],
    ) -> (PathBuf, PathBuf) {
        let total_tensors = shard1_tensors.len() + shard2_tensors.len();

        // Shard 1 carries the canonical metadata.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35moe");
        b1.add_u32("qwen35moe.block_count", 4);
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_u64("split.tensors.count", total_tensors as u64);
        for (name, ty, dims, data) in shard1_tensors {
            b1.add_tensor(name, *ty, dims, data.clone());
        }
        let p1 = dir.join(make_shard_filename(stem, 1, 2));
        std::fs::write(&p1, b1.build()).unwrap();

        // Shard 2: minimal metadata + its own tensors.
        let mut b2 = GgufBuilder::new();
        b2.add_string("general.architecture", "qwen35moe");
        b2.add_u16("split.no", 1);
        b2.add_u16("split.count", 2);
        b2.add_u64("split.tensors.count", total_tensors as u64);
        for (name, ty, dims, data) in shard2_tensors {
            b2.add_tensor(name, *ty, dims, data.clone());
        }
        let p2 = dir.join(make_shard_filename(stem, 2, 2));
        std::fs::write(&p2, b2.build()).unwrap();

        (p1, p2)
    }

    // -- Discovery -----------------------------------------------------------

    #[test]
    fn discover_single_file_returns_input() {
        let dir = temp_dir();
        let single = dir.join("model.gguf");
        std::fs::write(&single, [0u8; 8]).unwrap();
        let paths = discover_sibling_shards(&single).unwrap();
        assert_eq!(paths, vec![single]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_two_shard_set() {
        let dir = temp_dir();
        let (p1, p2) = write_2shard_set(
            &dir,
            "modelX",
            &[("token_embd.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64])],
            &[("blk.0.attn_q.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64])],
        );
        // Discover from either shard.
        let from_p1 = discover_sibling_shards(&p1).unwrap();
        let from_p2 = discover_sibling_shards(&p2).unwrap();
        assert_eq!(from_p1.len(), 2);
        assert_eq!(from_p2.len(), 2);
        assert!(from_p1.iter().any(|p| p == &p1));
        assert!(from_p1.iter().any(|p| p == &p2));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_missing_sibling_errors() {
        let dir = temp_dir();
        // Write only shard 1, claim total=2.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35moe");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        let p1 = dir.join("orphan-00001-of-00002.gguf");
        std::fs::write(&p1, b1.build()).unwrap();

        let err = discover_sibling_shards(&p1).unwrap_err();
        match err {
            ShardError::MissingShards { expected, present, .. } => {
                assert_eq!(expected, 2);
                assert_eq!(present, 1);
            }
            other => panic!("expected MissingShards, got {other}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Open + merge --------------------------------------------------------

    #[test]
    fn open_single_shard_backward_compat() {
        // A single-file GGUF (no shard suffix) must work identically to legacy.
        let dir = temp_dir();
        let mut b = GgufBuilder::new();
        b.add_string("general.architecture", "qwen35");
        b.add_u32("qwen35.block_count", 1);
        b.add_f32_tensor("token_embd.weight", &[4, 4], &[1.0; 16]);
        let p = dir.join("solo.gguf");
        std::fs::write(&p, b.build()).unwrap();

        let sharded = ShardedGguf::open(&p).unwrap();
        assert_eq!(sharded.shard_count(), 1);
        let m = sharded.merged();
        assert_eq!(m.tensors.len(), 1);
        assert_eq!(m.get_string("general.architecture"), Some("qwen35"));

        let t = m.find_tensor("token_embd.weight").unwrap().clone();
        let data = sharded.read_tensor_data(&t).unwrap();
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0; 16]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_two_shard_merged_view() {
        let dir = temp_dir();
        // Shard 1: token_embd + blk.0.attn_q
        // Shard 2: blk.1.attn_q + output.weight
        // u16 indices because tile_d's values exceed u8 range.
        let tile_a: Vec<u8> = (0u16..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let tile_b: Vec<u8> = (0u16..64).flat_map(|i| ((100 + i) as f32).to_le_bytes()).collect();
        let tile_c: Vec<u8> = (0u16..16).flat_map(|i| ((200 + i) as f32).to_le_bytes()).collect();
        let tile_d: Vec<u8> = (0u16..16).flat_map(|i| ((300 + i) as f32).to_le_bytes()).collect();

        let (p1, _p2) = write_2shard_set(
            &dir,
            "two-shard",
            &[
                ("token_embd.weight", GgmlType::F32, vec![16, 4], tile_a.clone()),
                ("blk.0.attn_q.weight", GgmlType::F32, vec![8, 8], tile_b.clone()),
            ],
            &[
                ("blk.1.attn_q.weight", GgmlType::F32, vec![4, 4], tile_c.clone()),
                ("output.weight", GgmlType::F32, vec![4, 4], tile_d.clone()),
            ],
        );

        let sharded = ShardedGguf::open(&p1).unwrap();
        assert_eq!(sharded.shard_count(), 2);
        let m = sharded.merged();
        assert_eq!(m.tensors.len(), 4);
        assert_eq!(m.get_string("general.architecture"), Some("qwen35moe"));
        assert_eq!(m.get_u32("qwen35moe.block_count"), Some(4));

        // Every tensor must be findable + reading must return the input bytes.
        for (name, expected) in [
            ("token_embd.weight", &tile_a),
            ("blk.0.attn_q.weight", &tile_b),
            ("blk.1.attn_q.weight", &tile_c),
            ("output.weight", &tile_d),
        ] {
            let t = m.find_tensor(name).expect(name).clone();
            let data = sharded.read_tensor_data(&t).unwrap();
            assert_eq!(&data, expected, "tensor {name} bytes mismatch");
        }

        // Shard attribution: first two go to shard 0, last two to shard 1.
        let (_t, s0a) = sharded.find_tensor_with_shard("token_embd.weight").unwrap();
        assert_eq!(s0a, 0);
        let (_t, s0b) = sharded.find_tensor_with_shard("blk.0.attn_q.weight").unwrap();
        assert_eq!(s0b, 0);
        let (_t, s1a) = sharded.find_tensor_with_shard("blk.1.attn_q.weight").unwrap();
        assert_eq!(s1a, 1);
        let (_t, s1b) = sharded.find_tensor_with_shard("output.weight").unwrap();
        assert_eq!(s1b, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Collision detection -------------------------------------------------

    #[test]
    fn duplicate_tensor_name_across_shards_errors() {
        let dir = temp_dir();
        let (p1, _p2) = write_2shard_set(
            &dir,
            "dup-set",
            &[("conflict.weight", GgmlType::F32, vec![4], vec![0u8; 16])],
            &[
                ("blk.0.attn_q.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64]),
                ("conflict.weight", GgmlType::F32, vec![4], vec![0u8; 16]),
            ],
        );

        let err = ShardedGguf::open(&p1).unwrap_err();
        match err {
            ShardError::TensorNameCollision { name, first_shard, second_shard } => {
                assert_eq!(name, "conflict.weight");
                assert_eq!(first_shard, 0);
                assert_eq!(second_shard, 1);
            }
            other => panic!("expected TensorNameCollision, got {other}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Validation ----------------------------------------------------------

    #[test]
    fn version_mismatch_across_shards_errors() {
        let dir = temp_dir();
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.version(3);
        b1.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
        let p1 = dir.join("vermix-00001-of-00002.gguf");
        std::fs::write(&p1, b1.build()).unwrap();

        let mut b2 = GgufBuilder::new();
        b2.add_string("general.architecture", "qwen35");
        b2.add_u16("split.no", 1);
        b2.add_u16("split.count", 2);
        b2.version(2);
        b2.add_f32_tensor("blk.0.attn_q.weight", &[4], &[0.0; 4]);
        let p2 = dir.join("vermix-00002-of-00002.gguf");
        std::fs::write(&p2, b2.build()).unwrap();

        let err = ShardedGguf::open(&p1).unwrap_err();
        match err {
            ShardError::VersionMismatch { expected, got, .. } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected VersionMismatch, got {other}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn split_count_mismatch_across_shards_errors() {
        let dir = temp_dir();
        // Shard 1 claims split.count=2; shard 2 claims split.count=3.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_f32_tensor("a.weight", &[4], &[0.0; 4]);
        let p1 = dir.join("mismatch-00001-of-00002.gguf");
        std::fs::write(&p1, b1.build()).unwrap();

        let mut b2 = GgufBuilder::new();
        b2.add_string("general.architecture", "qwen35");
        b2.add_u16("split.no", 1);
        b2.add_u16("split.count", 3);
        b2.add_f32_tensor("b.weight", &[4], &[0.0; 4]);
        let p2 = dir.join("mismatch-00002-of-00002.gguf");
        std::fs::write(&p2, b2.build()).unwrap();

        let err = ShardedGguf::open(&p1).unwrap_err();
        match err {
            ShardError::SplitCountMismatch { expected, got, .. } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            other => panic!("expected SplitCountMismatch, got {other}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shard_set_with_duplicate_split_no_errors() {
        // Two files both declare split.no=0, but they're both on disk pretending
        // to be a 2-shard set.
        let dir = temp_dir();
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35");
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_f32_tensor("a.weight", &[4], &[0.0; 4]);
        let p1 = dir.join("dup-00001-of-00002.gguf");
        std::fs::write(&p1, b1.build()).unwrap();

        let mut b2 = GgufBuilder::new();
        b2.add_string("general.architecture", "qwen35");
        b2.add_u16("split.no", 0); // duplicate
        b2.add_u16("split.count", 2);
        b2.add_f32_tensor("b.weight", &[4], &[0.0; 4]);
        let p2 = dir.join("dup-00002-of-00002.gguf");
        std::fs::write(&p2, b2.build()).unwrap();

        let err = ShardedGguf::open(&p1).unwrap_err();
        match err {
            ShardError::NonContiguousShards { found, expected_count } => {
                assert_eq!(expected_count, 2);
                assert_eq!(found, vec![0, 0]);
            }
            other => panic!("expected NonContiguousShards, got {other}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Public path/offset accessors ----------------------------------------

    #[test]
    fn shard_paths_and_total_disk_size() {
        let dir = temp_dir();
        let (p1, p2) = write_2shard_set(
            &dir,
            "sizecheck",
            &[("token_embd.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64])],
            &[("blk.0.attn_q.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64])],
        );

        let sharded = ShardedGguf::open(&p1).unwrap();
        let paths: Vec<_> = sharded.shard_paths().collect();
        assert_eq!(paths.len(), 2);
        let (sno0, _) = paths[0];
        let (sno1, _) = paths[1];
        assert_eq!(sno0, 0);
        assert_eq!(sno1, 1);

        let m1 = std::fs::metadata(&p1).unwrap().len();
        let m2 = std::fs::metadata(&p2).unwrap().len();
        let total = sharded.total_disk_size().unwrap();
        assert_eq!(total, m1 + m2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tensor_data_absolute_offset_lives_in_correct_shard() {
        let dir = temp_dir();
        let (p1, p2) = write_2shard_set(
            &dir,
            "offset-check",
            &[("token_embd.weight", GgmlType::F32, vec![4, 4], vec![0xAA; 64])],
            &[("blk.0.attn_q.weight", GgmlType::F32, vec![4, 4], vec![0xBB; 64])],
        );

        let sharded = ShardedGguf::open(&p1).unwrap();
        let m = sharded.merged();

        let t0 = m.find_tensor("token_embd.weight").unwrap().clone();
        let t1 = m.find_tensor("blk.0.attn_q.weight").unwrap().clone();

        // Reading via the public API must yield each tensor's actual byte
        // pattern (0xAA vs 0xBB) -- verifying the absolute-offset routing
        // picked the right shard file.
        let d0 = sharded.read_tensor_data(&t0).unwrap();
        let d1 = sharded.read_tensor_data(&t1).unwrap();
        assert!(d0.iter().all(|&b| b == 0xAA), "shard 0 tensor should be 0xAA pattern");
        assert!(d1.iter().all(|&b| b == 0xBB), "shard 1 tensor should be 0xBB pattern");

        // Spot-check the path-for-tensor accessor.
        assert_eq!(sharded.path_for_tensor(&t0).unwrap(), p1.as_path());
        assert_eq!(sharded.path_for_tensor(&t1).unwrap(), p2.as_path());

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- from_paths edge cases -----------------------------------------------

    /// Real-world shard 2 from the standard `gguf-split` tool carries
    /// minimal metadata: only `split.no`, `split.count`, and (sometimes)
    /// `split.tensors.count`. The architecture / hyperparams live in shard
    /// 1 only. The merge must pick shard 0's metadata; shard 2's missing
    /// metadata must not erase canonical fields.
    #[test]
    fn shard_2_minimal_metadata_preserved_from_shard_0() {
        let dir = temp_dir();
        let stem = "minimal-shard2";

        // Shard 1: full canonical metadata.
        let mut b1 = GgufBuilder::new();
        b1.add_string("general.architecture", "qwen35moe");
        b1.add_u32("qwen35moe.block_count", 32);
        b1.add_u32("qwen35moe.attention.head_count", 32);
        b1.add_u32("qwen35moe.embedding_length", 2048);
        b1.add_u32("qwen35moe.feed_forward_length", 768);
        b1.add_u16("split.no", 0);
        b1.add_u16("split.count", 2);
        b1.add_f32_tensor("token_embd.weight", &[8], &[0.0; 8]);
        let p1 = dir.join(make_shard_filename(stem, 1, 2));
        std::fs::write(&p1, b1.build()).unwrap();

        // Shard 2: ONLY split metadata, no architecture/hyperparams. This is
        // exactly what `gguf-split` produces for the second shard of a split.
        let mut b2 = GgufBuilder::new();
        b2.add_u16("split.no", 1);
        b2.add_u16("split.count", 2);
        b2.add_f32_tensor("blk.0.attn_q.weight", &[8], &[0.0; 8]);
        let p2 = dir.join(make_shard_filename(stem, 2, 2));
        std::fs::write(&p2, b2.build()).unwrap();

        let view = ShardedGguf::open(&p1).unwrap();
        let m = view.merged();

        // Canonical metadata is preserved from shard 0.
        assert_eq!(m.get_string("general.architecture"), Some("qwen35moe"));
        assert_eq!(m.get_u32("qwen35moe.block_count"), Some(32));
        assert_eq!(m.get_u32("qwen35moe.attention.head_count"), Some(32));
        assert_eq!(m.get_u32("qwen35moe.embedding_length"), Some(2048));
        assert_eq!(m.get_u32("qwen35moe.feed_forward_length"), Some(768));

        // Both tensors are present in the merged view.
        assert_eq!(m.tensors.len(), 2);
        assert!(m.find_tensor("token_embd.weight").is_some());
        assert!(m.find_tensor("blk.0.attn_q.weight").is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Tensors must read back from a 2-shard set even when the order in which
    /// shards are listed on disk does not match the `split.no` ordering. This
    /// simulates a filesystem returning siblings in an arbitrary order.
    #[test]
    fn tensor_reads_route_correctly_after_shard_resort() {
        let dir = temp_dir();
        let (p1, p2) = write_2shard_set(
            &dir,
            "resort",
            &[("alpha.weight", GgmlType::F32, vec![4], vec![0xAA; 16])],
            &[("beta.weight", GgmlType::F32, vec![4], vec![0xBB; 16])],
        );
        // Pass shard 2 first; the library must re-sort internally so the
        // merged view's shard-0 metadata still comes from p1 (split.no=0).
        let view = ShardedGguf::from_paths(&[p2.clone(), p1.clone()]).unwrap();
        let m = view.merged();
        let alpha = m.find_tensor("alpha.weight").unwrap().clone();
        let beta = m.find_tensor("beta.weight").unwrap().clone();
        let a = view.read_tensor_data(&alpha).unwrap();
        let b = view.read_tensor_data(&beta).unwrap();
        assert!(a.iter().all(|&v| v == 0xAA),
                "alpha (shard 0, split.no=0) must read 0xAA bytes regardless of input order");
        assert!(b.iter().all(|&v| v == 0xBB),
                "beta (shard 1, split.no=1) must read 0xBB bytes regardless of input order");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_paths_empty_rejects() {
        assert!(ShardedGguf::from_paths(&[]).is_err());
    }

    // -- MultiShardReader e2e tests ------------------------------------------

    /// Exercises the same `seek(SeekFrom::Start(gguf.tensor_data_offset(t))) + read_exact`
    /// flow that [`crate::tensor_io::read_tensor_data`] uses. Verifies that a
    /// merged view + MultiShardReader reads each tensor's exact bytes regardless
    /// of which shard it lives in.
    #[test]
    fn multi_shard_reader_e2e_matches_per_shard_reads() {
        let dir = temp_dir();
        // Each payload is 16 F32 elements = 64 bytes, matching the tensor's
        // declared byte_size (dims=[8,2] -> 16 elements * 4 bytes/F32 = 64).
        // Four distinct byte patterns so a misrouted seek cannot silently
        // succeed.
        let payload_a: Vec<u8> = (0u8..64).collect();
        let payload_b: Vec<u8> = (64u8..128).collect();
        let payload_c: Vec<u8> = (128u8..192).collect();
        let payload_d: Vec<u8> = (192u8..=255).collect();
        assert_eq!(payload_a.len(), 64);
        assert_eq!(payload_d.len(), 64);

        let (p1, _p2) = write_2shard_set(
            &dir,
            "e2e",
            &[
                ("alpha.weight", GgmlType::F32, vec![8, 2], payload_a.clone()),
                ("beta.weight", GgmlType::F32, vec![8, 2], payload_b.clone()),
            ],
            &[
                ("gamma.weight", GgmlType::F32, vec![8, 2], payload_c.clone()),
                ("delta.weight", GgmlType::F32, vec![8, 2], payload_d.clone()),
            ],
        );

        let view = ShardedGguf::open(&p1).unwrap();
        let merged = view.merged();
        let mut reader = MultiShardReader::open(&view).unwrap();

        // Replay the seek+read pattern that tensor_io uses.
        for (name, expected) in [
            ("alpha.weight", &payload_a),
            ("beta.weight", &payload_b),
            ("gamma.weight", &payload_c),
            ("delta.weight", &payload_d),
        ] {
            let t = merged.find_tensor(name).expect(name);
            let abs = merged.tensor_data_offset(t);
            reader.seek(SeekFrom::Start(abs)).unwrap();
            let size = t.byte_size().unwrap() as usize;
            let mut buf = vec![0u8; size];
            reader.read_exact(&mut buf).unwrap();
            assert_eq!(&buf, expected, "{name} bytes mismatch through MultiShardReader");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Single-shard byte-identity: a MultiShardReader over a 1-shard view
    /// reads exactly the same bytes as a vanilla BufReader<File> on the file.
    /// This is the regression guard for G3 (Q8_0 / Q4_0 single-shard
    /// byte-identical conversion).
    #[test]
    fn multi_shard_reader_single_shard_byte_identical() {
        let dir = temp_dir();
        let payload: Vec<u8> = (0u8..255).chain(std::iter::once(255u8)).collect();
        assert_eq!(payload.len(), 256);

        let mut b = GgufBuilder::new();
        b.add_string("general.architecture", "qwen35");
        b.add_tensor("token_embd.weight", GgmlType::F32, &[64], payload.clone());
        let p = dir.join("solo.gguf");
        std::fs::write(&p, b.build()).unwrap();

        let view = ShardedGguf::open(&p).unwrap();
        let merged = view.merged();
        let t = merged.find_tensor("token_embd.weight").unwrap().clone();
        let mut reader = MultiShardReader::open(&view).unwrap();
        let abs = merged.tensor_data_offset(&t);
        reader.seek(SeekFrom::Start(abs)).unwrap();
        let size = t.byte_size().unwrap() as usize;
        let mut buf = vec![0u8; size];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, payload, "single-shard MultiShardReader must be byte-identical");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Seeking outside the known shard range must return InvalidInput, not
    /// silently land in shard 0 or panic.
    #[test]
    fn multi_shard_reader_rejects_out_of_range_shard() {
        let dir = temp_dir();
        let mut b = GgufBuilder::new();
        b.add_string("general.architecture", "qwen35");
        b.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
        let p = dir.join("solo2.gguf");
        std::fs::write(&p, b.build()).unwrap();

        let view = ShardedGguf::open(&p).unwrap();
        let mut reader = MultiShardReader::open(&view).unwrap();

        // Forge a virtual address that targets shard 7 -- there's only shard 0.
        let bogus = make_virtual_address(7, 0);
        let err = reader.seek(SeekFrom::Start(bogus)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Relative seeks are rejected (we always issue absolute seeks before each
    /// tensor read; a relative seek would land in the wrong shard).
    #[test]
    fn multi_shard_reader_rejects_relative_seeks() {
        let dir = temp_dir();
        let mut b = GgufBuilder::new();
        b.add_string("general.architecture", "qwen35");
        b.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
        let p = dir.join("solo3.gguf");
        std::fs::write(&p, b.build()).unwrap();

        let view = ShardedGguf::open(&p).unwrap();
        let mut reader = MultiShardReader::open(&view).unwrap();

        let err = reader.seek(SeekFrom::Current(10)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        let err = reader.seek(SeekFrom::End(0)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Virtual address bit packing -----------------------------------------

    #[test]
    fn virtual_address_roundtrip() {
        for (sid, off) in [(0u8, 0u64), (1, 0x1234), (5, (1u64 << 40)), (255, WITHIN_SHARD_MASK)] {
            let v = make_virtual_address(sid, off);
            let (sid2, off2) = split_virtual_address(v);
            assert_eq!((sid, off), (sid2, off2),
                "roundtrip failed: shard={sid} off=0x{off:x} -> v=0x{v:016x} -> shard={sid2} off=0x{off2:x}");
        }
    }

    #[test]
    fn virtual_address_zero_shard_is_zero_high_bits() {
        // Shard 0 + offset X should equal X (compat with single-file callers
        // that don't yet route through the sharded path).
        for off in [0u64, 32, 1_048_576, WITHIN_SHARD_MASK] {
            assert_eq!(make_virtual_address(0, off), off);
        }
    }

    #[test]
    fn from_paths_out_of_order_resorts() {
        // Pass shard 2 before shard 1; result must still merge in shard-0-first order.
        let dir = temp_dir();
        let (p1, p2) = write_2shard_set(
            &dir,
            "ooo",
            &[("alpha.weight", GgmlType::F32, vec![4], vec![0xAA; 16])],
            &[("beta.weight", GgmlType::F32, vec![4], vec![0xBB; 16])],
        );
        let sharded = ShardedGguf::from_paths(&[p2.clone(), p1.clone()]).unwrap();
        let m = sharded.merged();
        // First tensor in merged.tensors must be the shard-0 entry (alpha).
        assert_eq!(m.tensors[0].name, "alpha.weight");
        assert_eq!(m.tensors[1].name, "beta.weight");
        let (_, s0) = sharded.find_tensor_with_shard("alpha.weight").unwrap();
        let (_, s1) = sharded.find_tensor_with_shard("beta.weight").unwrap();
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
