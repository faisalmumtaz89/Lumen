//! Expert-granular LBC file reader for MoE SSD streaming.
//!
//! Loads individual expert weights from an LBC file without reading the full
//! layer blob. This enables on-demand expert streaming where only the 3 FFN
//! matrices (gate, up, down) for a specific expert are read from disk.
//!
//! ## Seek Strategy
//!
//! Each expert's weight data lives at a known byte offset within the LBC file:
//!
//! ```text
//! absolute_offset = layer_index.layer_offset_bytes + expert_slice.{gate|up|down}.offset
//! ```
//!
//! The reader uses `pread`-style positioned reads (via `File::seek` + `File::read_exact`)
//! to fetch only the bytes needed for a single expert, avoiding full-layer reads.
//! For parallel loading, each thread opens its own file descriptor.

use lumen_format::index::{ExpertSlice, LayerIndex};
use lumen_format::reader::LbcFile;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Set F_NOCACHE on a file descriptor to bypass the OS page cache.
/// This prevents double-buffering for streaming workloads where each page
/// is read once and discarded. On non-macOS platforms this is a no-op.
#[cfg(target_os = "macos")]
fn set_no_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1i32) };
}

#[cfg(not(target_os = "macos"))]
fn set_no_cache(_file: &std::fs::File) {}

/// Errors specific to expert-granular reads.
#[derive(Debug, thiserror::Error)]
pub enum ExpertReaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("LBC format error: {0}")]
    Format(#[from] lumen_format::FormatError),

    #[error("layer {layer} has no MoE experts")]
    NotMoeLayer { layer: usize },

    #[error("layer {layer} expert {expert} out of range (num_experts={num_experts})")]
    ExpertOutOfRange {
        layer: usize,
        expert: u32,
        num_experts: usize,
    },

    #[error("layer {layer} has no router weight")]
    NoRouterWeight { layer: usize },

    #[error("layer {layer} out of range (num_layers={num_layers})")]
    LayerOutOfRange { layer: usize, num_layers: usize },
}

/// Loads individual expert weights from an LBC file without reading the full layer blob.
/// This enables on-demand expert streaming from SSD.
pub struct ExpertReader {
    /// Path to the LBC file (used for opening additional file descriptors for parallel reads).
    path: PathBuf,
    /// Primary file handle for sequential reads.
    file: std::fs::File,
    /// Parsed layer index table from the LBC header.
    layer_indices: Vec<LayerIndex>,
}

impl ExpertReader {
    /// Open an LBC file and parse its header/index for expert-granular access.
    pub fn open(path: &Path) -> Result<Self, ExpertReaderError> {
        let lbc = LbcFile::open(path)?;
        let file = std::fs::File::open(path)?;
        set_no_cache(&file);
        Ok(Self {
            path: path.to_path_buf(),
            file,
            layer_indices: lbc.layer_indices,
        })
    }

    /// Create an ExpertReader from pre-parsed LBC metadata.
    /// Useful for tests where the LBC is already in memory.
    pub fn from_lbc(path: &Path, layer_indices: Vec<LayerIndex>) -> Result<Self, ExpertReaderError> {
        let file = std::fs::File::open(path)?;
        set_no_cache(&file);
        Ok(Self {
            path: path.to_path_buf(),
            file,
            layer_indices,
        })
    }

    /// Number of layers in this model.
    pub fn num_layers(&self) -> usize {
        self.layer_indices.len()
    }

    /// Number of experts at the given layer, or None if not an MoE layer.
    pub fn num_experts(&self, layer: usize) -> Option<usize> {
        self.layer_indices
            .get(layer)
            .and_then(|idx| idx.subtensors.experts.as_ref())
            .map(|experts| experts.len())
    }

    /// Load raw bytes for a specific expert (gate + up + down, concatenated).
    ///
    /// Uses the ExpertSlice offsets to seek directly to the expert's data
    /// within the LBC file. Does NOT load the full layer blob -- only the
    /// 3 matrices for this expert.
    ///
    /// The returned bytes contain gate, up, and down projections concatenated
    /// in that order. Use the returned ExpertSlice to find the boundaries.
    pub fn load_expert(
        &mut self,
        layer: usize,
        expert: u32,
    ) -> Result<(Vec<u8>, ExpertSlice), ExpertReaderError> {
        // Build a read plan (borrows self immutably), then execute I/O separately.
        let plan = self.build_read_plan(layer, expert)?;
        Self::execute_read_plan(&mut self.file, &plan)
    }

    /// Load the router weight for a layer (always F32, typically small).
    pub fn load_router(&mut self, layer: usize) -> Result<Vec<u8>, ExpertReaderError> {
        // Extract needed metadata as owned values to avoid overlapping borrows.
        let layer_idx = self.validate_layer(layer)?;
        let router_slice = layer_idx
            .subtensors
            .router_weight
            .clone()
            .ok_or(ExpertReaderError::NoRouterWeight { layer })?;
        let blob_offset = layer_idx.layer_offset_bytes;
        self.read_slice(blob_offset, &router_slice)
    }

    /// Load multiple experts using a thread pool with separate file descriptors.
    ///
    /// Each request is (layer, expert_id). Returns results in the same order
    /// as the input requests. Uses `std::thread::scope` with one file handle
    /// per thread for parallel pread-style I/O.
    pub fn load_experts_parallel(
        &self,
        requests: &[(usize, u32)],
    ) -> Vec<Result<(Vec<u8>, ExpertSlice), ExpertReaderError>> {
        if requests.is_empty() {
            return Vec::new();
        }

        // Pre-validate and collect the read plans so threads only do I/O.
        let plans: Vec<_> = requests
            .iter()
            .map(|&(layer, expert)| self.build_read_plan(layer, expert))
            .collect();

        let path = &self.path;

        std::thread::scope(|scope| {
            let handles: Vec<_> = plans
                .into_iter()
                .map(|plan| {
                    scope.spawn(move || -> Result<(Vec<u8>, ExpertSlice), ExpertReaderError> {
                        let plan = plan?;
                        let mut file = std::fs::File::open(path)?;
                        set_no_cache(&file);
                        Self::execute_read_plan(&mut file, &plan)
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().expect("expert reader thread panicked"))
                .collect()
        })
    }

    /// Return the path to the underlying LBC file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the layer indices (for external parallel pread callers).
    pub fn layer_indices(&self) -> &[LayerIndex] {
        &self.layer_indices
    }

    // ---- Internal helpers ----

    /// Validate layer index and return a reference to it.
    fn validate_layer(&self, layer: usize) -> Result<&LayerIndex, ExpertReaderError> {
        self.layer_indices
            .get(layer)
            .ok_or(ExpertReaderError::LayerOutOfRange {
                layer,
                num_layers: self.layer_indices.len(),
            })
    }

    /// Validate layer+expert and return references to LayerIndex and ExpertSlice.
    fn validate_expert(
        &self,
        layer: usize,
        expert: u32,
    ) -> Result<(&LayerIndex, &ExpertSlice), ExpertReaderError> {
        let layer_idx = self.validate_layer(layer)?;
        let experts = layer_idx
            .subtensors
            .experts
            .as_ref()
            .ok_or(ExpertReaderError::NotMoeLayer { layer })?;
        let eid = expert as usize;
        if eid >= experts.len() {
            return Err(ExpertReaderError::ExpertOutOfRange {
                layer,
                expert,
                num_experts: experts.len(),
            });
        }
        Ok((layer_idx, &experts[eid]))
    }

    /// Read a TensorSlice from the file at blob_offset + slice.offset.
    fn read_slice(
        &mut self,
        blob_offset: u64,
        slice: &lumen_format::index::TensorSlice,
    ) -> Result<Vec<u8>, ExpertReaderError> {
        let abs_offset = blob_offset + slice.offset;
        let length = slice.length as usize;
        let mut buf = vec![0u8; length];
        self.file.seek(SeekFrom::Start(abs_offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Build a read plan for parallel execution (validates without doing I/O).
    fn build_read_plan(
        &self,
        layer: usize,
        expert: u32,
    ) -> Result<ReadPlan, ExpertReaderError> {
        let (layer_idx, expert_slice) = self.validate_expert(layer, expert)?;
        let blob_offset = layer_idx.layer_offset_bytes;
        Ok(ReadPlan {
            gate_offset: blob_offset + expert_slice.gate.offset,
            gate_length: expert_slice.gate.length as usize,
            gate_quant: expert_slice.gate.quant,
            up_offset: blob_offset + expert_slice.up.offset,
            up_length: expert_slice.up.length as usize,
            up_quant: expert_slice.up.quant,
            down_offset: blob_offset + expert_slice.down.offset,
            down_length: expert_slice.down.length as usize,
            down_quant: expert_slice.down.quant,
        })
    }

    /// Execute a read plan on a given file handle.
    fn execute_read_plan(
        file: &mut std::fs::File,
        plan: &ReadPlan,
    ) -> Result<(Vec<u8>, ExpertSlice), ExpertReaderError> {
        let total = plan.gate_length + plan.up_length + plan.down_length;
        let mut data = vec![0u8; total];

        // Read gate.
        file.seek(SeekFrom::Start(plan.gate_offset))?;
        file.read_exact(&mut data[..plan.gate_length])?;

        // Read up.
        file.seek(SeekFrom::Start(plan.up_offset))?;
        file.read_exact(&mut data[plan.gate_length..plan.gate_length + plan.up_length])?;

        // Read down.
        file.seek(SeekFrom::Start(plan.down_offset))?;
        file.read_exact(
            &mut data[plan.gate_length + plan.up_length..],
        )?;

        let local_slices = ExpertSlice {
            gate: lumen_format::index::TensorSlice {
                offset: 0,
                length: plan.gate_length as u64,
                quant: plan.gate_quant,
            },
            up: lumen_format::index::TensorSlice {
                offset: plan.gate_length as u64,
                length: plan.up_length as u64,
                quant: plan.up_quant,
            },
            down: lumen_format::index::TensorSlice {
                offset: (plan.gate_length + plan.up_length) as u64,
                length: plan.down_length as u64,
                quant: plan.down_quant,
            },
        };

        Ok((data, local_slices))
    }
}

/// Pre-validated I/O plan for a single expert read, decoupling validation from I/O.
struct ReadPlan {
    gate_offset: u64,
    gate_length: usize,
    gate_quant: lumen_format::quantization::QuantScheme,
    up_offset: u64,
    up_length: usize,
    up_quant: lumen_format::quantization::QuantScheme,
    down_offset: u64,
    down_length: usize,
    down_quant: lumen_format::quantization::QuantScheme,
}

/// Read `length` bytes starting at `offset` from the file at `path` using
/// parallel chunked reads. Spawns `num_threads` threads, each opening its
/// own F_NOCACHE file descriptor and reading a contiguous chunk via
/// seek + read_exact. This increases NVMe queue depth from 1 to
/// `num_threads`, saturating SSD bandwidth for large sequential reads.
///
/// Returns the assembled byte vector on success.
pub fn parallel_pread(
    path: &Path,
    offset: u64,
    length: usize,
    num_threads: usize,
) -> std::io::Result<Vec<u8>> {
    if length == 0 || num_threads == 0 {
        return Ok(Vec::new());
    }
    let threads = num_threads.min(length); // never more threads than bytes
    let chunk = (length + threads - 1) / threads;
    let mut result = vec![0u8; length];
    let path_buf = path.to_path_buf();

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for (i, chunk_slice) in result.chunks_mut(chunk).enumerate() {
            let chunk_offset = offset + (i * chunk) as u64;
            let p = &path_buf;
            handles.push(s.spawn(move || -> std::io::Result<()> {
                let mut f = std::fs::File::open(p)?;
                set_no_cache(&f);
                f.seek(SeekFrom::Start(chunk_offset))?;
                f.read_exact(chunk_slice)?;
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("parallel_pread thread panicked")?;
        }
        std::io::Result::Ok(())
    })?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_format::header::LbcHeader;
    use lumen_format::hyperparams::{ModelHyperparams, RopeParams};
    use lumen_format::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
    use lumen_format::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
    use lumen_format::writer::{write_lbc, GlobalTensors};
    use std::io::Write;

    /// Helper: create a TensorSlice at the given offset/length.
    fn make_slice(offset: u64, length: u64) -> TensorSlice {
        TensorSlice {
            offset,
            length,
            quant: QuantScheme::F32,
        }
    }

    /// Build a synthetic MoE LBC file on disk with known expert data.
    /// Returns (path, expected_expert_data) where expected_expert_data[layer][expert]
    /// contains the concatenated gate+up+down bytes for that expert.
    /// `test_name` ensures each test gets a unique file to avoid race conditions.
    fn create_moe_test_file(
        test_name: &str,
        num_layers: usize,
        num_experts: usize,
    ) -> (std::path::PathBuf, Vec<Vec<Vec<u8>>>) {
        let hp = ModelHyperparams {
            num_layers: num_layers as u32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_size: 32,
            max_seq_len: 64,
            rope_params: Some(RopeParams::default()),
            num_experts: Some(num_experts as u32),
            num_active_experts: Some(2),
            norm_eps: 1e-5,
            rotary_dim: None, rope_neox: false,
        };
        let qd = QuantizationDescriptor {
            scheme: QuantScheme::F32,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        };
        let header = LbcHeader::new(hp, qd);

        // Attention weights: wq=128, wk=128, wv=128, wo=128 (8*8*4=256 would be correct
        // for hidden_dim=8, head_dim=4, but we use small values for testing).
        // Norms: 32 each. Router: num_experts * hidden_dim * 4 bytes.
        // Each expert: gate=64, up=64, down=64 bytes.
        let attn_size = 128u64; // per attention weight
        let norm_size = 32u64;
        let router_size = (num_experts as u64) * 8 * 4; // num_experts * hidden_dim * f32
        let expert_tensor_size = 64u64; // gate, up, down each

        let mut expected: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut layer_blobs = Vec::new();
        let mut layer_indices = Vec::new();

        for layer in 0..num_layers {
            let mut blob = Vec::new();
            let mut offset = 0u64;

            // Attention weights (filled with layer-specific pattern).
            let attn_pattern = (layer as u8).wrapping_mul(17);
            for _ in 0..4 {
                blob.extend(vec![attn_pattern; attn_size as usize]);
            }
            let wq = make_slice(offset, attn_size);
            offset += attn_size;
            let wk = make_slice(offset, attn_size);
            offset += attn_size;
            let wv = make_slice(offset, attn_size);
            offset += attn_size;
            let wo = make_slice(offset, attn_size);
            offset += attn_size;

            // Norms.
            blob.extend(vec![0xAA; norm_size as usize]);
            let attn_norm = make_slice(offset, norm_size);
            offset += norm_size;
            blob.extend(vec![0xBB; norm_size as usize]);
            let ffn_norm = make_slice(offset, norm_size);
            offset += norm_size;

            // Router weight.
            blob.extend(vec![0xCC; router_size as usize]);
            let router = make_slice(offset, router_size);
            offset += router_size;

            // Expert weights with unique patterns per (layer, expert, tensor).
            let mut layer_experts = Vec::new();
            let mut expert_slices = Vec::new();
            for expert in 0..num_experts {
                let gate_pattern = ((layer * num_experts + expert) * 3) as u8;
                let up_pattern = ((layer * num_experts + expert) * 3 + 1) as u8;
                let down_pattern = ((layer * num_experts + expert) * 3 + 2) as u8;

                let gate_data = vec![gate_pattern; expert_tensor_size as usize];
                let up_data = vec![up_pattern; expert_tensor_size as usize];
                let down_data = vec![down_pattern; expert_tensor_size as usize];

                let gate_slice = make_slice(offset, expert_tensor_size);
                blob.extend_from_slice(&gate_data);
                offset += expert_tensor_size;

                let up_slice = make_slice(offset, expert_tensor_size);
                blob.extend_from_slice(&up_data);
                offset += expert_tensor_size;

                let down_slice = make_slice(offset, expert_tensor_size);
                blob.extend_from_slice(&down_data);
                offset += expert_tensor_size;

                expert_slices.push(ExpertSlice {
                    gate: gate_slice,
                    up: up_slice,
                    down: down_slice,
                });

                // Expected concatenated data for this expert.
                let mut concat = Vec::new();
                concat.extend_from_slice(&gate_data);
                concat.extend_from_slice(&up_data);
                concat.extend_from_slice(&down_data);
                layer_experts.push(concat);
            }
            expected.push(layer_experts);

            let subtensors = SubtensorOffsets {
                wq,
                wk,
                wv,
                wo,
                bq: None,
                bk: None,
                bv: None,
                w_gate: make_slice(0, 0),
                w_up: make_slice(0, 0),
                w_down: make_slice(0, 0),
                attn_norm,
                ffn_norm,
                router_weight: Some(router),
                experts: Some(expert_slices),
                shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
                attn_gate: None, attn_post_norm: None,
                ssm_a: None, ssm_conv1d: None, ssm_dt: None,
                ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
                attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
                layer_type: None,
            };
            layer_indices.push(LayerIndex {
                layer_offset_bytes: 0, // Fixed by writer.
                layer_length_bytes: blob.len() as u64,
                subtensors,
            });
            layer_blobs.push(blob);
        }

        let globals = GlobalTensors {
            embedding: vec![1u8; 32 * 8 * 4],
            final_norm: vec![2u8; 8 * 4],
            output_proj: vec![3u8; 32 * 8 * 4],
        };

        let blob_refs: Vec<&[u8]> = layer_blobs.iter().map(|b| b.as_slice()).collect();
        let mut out = Vec::new();
        write_lbc(&mut out, &header, &layer_indices, &globals, &blob_refs).unwrap();

        // Write to a temp file with unique name per test.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lumen_expert_reader_{}_{}.lbc",
            test_name,
            std::process::id()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&out).unwrap();
        f.flush().unwrap();

        (path, expected)
    }

    #[test]
    fn test_reader_open() {
        let (path, _expected) = create_moe_test_file("open", 2, 2);
        let reader = ExpertReader::open(&path).unwrap();
        assert_eq!(reader.num_layers(), 2);
        assert_eq!(reader.num_experts(0), Some(2));
        assert_eq!(reader.num_experts(1), Some(2));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_load_expert() {
        let (path, expected) = create_moe_test_file("load_expert", 2, 2);
        let mut reader = ExpertReader::open(&path).unwrap();

        // Load expert 0 from layer 0.
        let (data, slices) = reader.load_expert(0, 0).unwrap();
        assert_eq!(data, expected[0][0]);

        // Verify slices point to the correct regions.
        assert_eq!(slices.gate.offset, 0);
        assert_eq!(slices.gate.length, 64);
        assert_eq!(slices.up.offset, 64);
        assert_eq!(slices.up.length, 64);
        assert_eq!(slices.down.offset, 128);
        assert_eq!(slices.down.length, 64);

        // Load expert 1 from layer 0.
        let (data, _) = reader.load_expert(0, 1).unwrap();
        assert_eq!(data, expected[0][1]);

        // Load expert 0 from layer 1.
        let (data, _) = reader.load_expert(1, 0).unwrap();
        assert_eq!(data, expected[1][0]);

        // Load expert 1 from layer 1.
        let (data, _) = reader.load_expert(1, 1).unwrap();
        assert_eq!(data, expected[1][1]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_load_router() {
        let (path, _expected) = create_moe_test_file("load_router", 1, 2);
        let mut reader = ExpertReader::open(&path).unwrap();

        let router = reader.load_router(0).unwrap();
        // Router is num_experts * hidden_dim * sizeof(f32) = 2 * 8 * 4 = 64 bytes.
        assert_eq!(router.len(), 64);
        // All bytes should be 0xCC (our router fill pattern).
        assert!(router.iter().all(|&b| b == 0xCC));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_parallel() {
        let (path, expected) = create_moe_test_file("parallel", 2, 2);
        let reader = ExpertReader::open(&path).unwrap();

        // Load all 4 experts in parallel.
        let requests = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let results = reader.load_experts_parallel(&requests);

        assert_eq!(results.len(), 4);
        for (i, result) in results.into_iter().enumerate() {
            let (data, _slices) = result.unwrap();
            let (layer, expert) = requests[i];
            assert_eq!(
                data, expected[layer][expert as usize],
                "mismatch for layer={layer} expert={expert}"
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_layer_out_of_range() {
        let (path, _) = create_moe_test_file("layer_oor", 1, 2);
        let mut reader = ExpertReader::open(&path).unwrap();

        let result = reader.load_expert(5, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            ExpertReaderError::LayerOutOfRange { layer, num_layers } => {
                assert_eq!(layer, 5);
                assert_eq!(num_layers, 1);
            }
            other => panic!("expected LayerOutOfRange, got: {other}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_expert_out_of_range() {
        let (path, _) = create_moe_test_file("expert_oor", 1, 2);
        let mut reader = ExpertReader::open(&path).unwrap();

        let result = reader.load_expert(0, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            ExpertReaderError::ExpertOutOfRange {
                layer,
                expert,
                num_experts,
            } => {
                assert_eq!(layer, 0);
                assert_eq!(expert, 5);
                assert_eq!(num_experts, 2);
            }
            other => panic!("expected ExpertOutOfRange, got: {other}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_no_router() {
        // Create a dense (non-MoE) model file and try to load router.
        let hp = ModelHyperparams {
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_size: 32,
            max_seq_len: 64,
            rope_params: Some(RopeParams::default()),
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-5,
            rotary_dim: None, rope_neox: false,
        };
        let qd = QuantizationDescriptor {
            scheme: QuantScheme::F32,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        };
        let header = LbcHeader::new(hp, qd);
        let subtensors = SubtensorOffsets {
            wq: make_slice(0, 128),
            wk: make_slice(128, 128),
            wv: make_slice(256, 128),
            wo: make_slice(384, 128),
            bq: None,
            bk: None,
            bv: None,
            w_gate: make_slice(512, 256),
            w_up: make_slice(768, 256),
            w_down: make_slice(1024, 256),
            attn_norm: make_slice(1280, 32),
            ffn_norm: make_slice(1312, 32),
            router_weight: None,
            experts: None,
            shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
            attn_gate: None, attn_post_norm: None,
            ssm_a: None, ssm_conv1d: None, ssm_dt: None,
            ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
            attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
            layer_type: None,
        };
        let idx = LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: 1344,
            subtensors,
        };

        let globals = GlobalTensors {
            embedding: vec![1u8; 32 * 8 * 4],
            final_norm: vec![2u8; 8 * 4],
            output_proj: vec![3u8; 32 * 8 * 4],
        };
        let blob = vec![0u8; 1344];
        let mut out = Vec::new();
        write_lbc(&mut out, &header, &[idx], &globals, &[&blob]).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lumen_expert_reader_no_router_{}.lbc",
            std::process::id()
        ));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&out).unwrap();
        }

        let mut reader = ExpertReader::open(&path).unwrap();
        let result = reader.load_router(0);
        assert!(matches!(result, Err(ExpertReaderError::NoRouterWeight { .. })));

        let result = reader.load_expert(0, 0);
        assert!(matches!(result, Err(ExpertReaderError::NotMoeLayer { .. })));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_parallel_empty() {
        let (path, _) = create_moe_test_file("parallel_empty", 1, 2);
        let reader = ExpertReader::open(&path).unwrap();

        let results = reader.load_experts_parallel(&[]);
        assert!(results.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_reader_data_integrity() {
        // Verify that each expert's data uses the unique byte patterns we set.
        let (path, expected) = create_moe_test_file("integrity", 2, 3);
        let mut reader = ExpertReader::open(&path).unwrap();

        for layer in 0..2 {
            for expert in 0..3u32 {
                let (data, _) = reader.load_expert(layer, expert).unwrap();
                assert_eq!(
                    data,
                    expected[layer][expert as usize],
                    "data mismatch at layer={layer}, expert={expert}"
                );

                // Verify the byte patterns are unique per (layer, expert, tensor).
                let gate_pattern = ((layer * 3 + expert as usize) * 3) as u8;
                let up_pattern = ((layer * 3 + expert as usize) * 3 + 1) as u8;
                let down_pattern = ((layer * 3 + expert as usize) * 3 + 2) as u8;

                assert!(data[..64].iter().all(|&b| b == gate_pattern));
                assert!(data[64..128].iter().all(|&b| b == up_pattern));
                assert!(data[128..192].iter().all(|&b| b == down_pattern));
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parallel_pread_basic() {
        // Write a known-pattern file and read it back via parallel_pread.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lumen_expert_reader_ppread_{}.bin",
            std::process::id()
        ));
        let data: Vec<u8> = (0..8192u32).map(|i| (i % 256) as u8).collect();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }

        // Read full file with 4 threads.
        let result = parallel_pread(&path, 0, data.len(), 4).unwrap();
        assert_eq!(result, data, "full parallel_pread must match original data");

        // Read a sub-range (offset 1000, length 2000).
        let result2 = parallel_pread(&path, 1000, 2000, 3).unwrap();
        assert_eq!(result2, &data[1000..3000], "sub-range parallel_pread must match");

        // Edge case: single thread.
        let result3 = parallel_pread(&path, 0, data.len(), 1).unwrap();
        assert_eq!(result3, data, "single-thread parallel_pread must match");

        // Edge case: zero length.
        let result4 = parallel_pread(&path, 0, 0, 4).unwrap();
        assert!(result4.is_empty(), "zero-length must return empty vec");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parallel_pread_more_threads_than_bytes() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lumen_expert_reader_ppread_tiny_{}.bin",
            std::process::id()
        ));
        let data = vec![0xAB_u8; 10];
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }

        // 100 threads for 10 bytes -- should clamp to 10 threads.
        let result = parallel_pread(&path, 0, 10, 100).unwrap();
        assert_eq!(result, data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_f_nocache_applied_on_open() {
        // Verify ExpertReader::open succeeds (F_NOCACHE is best-effort).
        let (path, _) = create_moe_test_file("f_nocache", 1, 2);
        let reader = ExpertReader::open(&path).unwrap();
        assert_eq!(reader.num_layers(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// Benchmark sequential vs parallel pread throughput on a real LBC file.
    /// Measures bandwidth in GB/s for single-thread vs multi-thread reads.
    ///
    /// Requires: `/tmp/lumen-bench/mixtral-8x7b-v0.1.lbc`
    /// Run with: cargo test -p lumen-runtime -- --ignored test_parallel_pread_bandwidth
    #[test]
    #[ignore]
    fn test_parallel_pread_bandwidth() {
        use std::time::Instant;

        let lbc_path = std::path::Path::new("/tmp/lumen-bench/mixtral-8x7b-v0.1.lbc");
        if !lbc_path.exists() {
            eprintln!(
                "SKIP: benchmark file not found at {}",
                lbc_path.display()
            );
            return;
        }

        let lbc = lumen_format::reader::LbcFile::open(lbc_path).unwrap();
        if lbc.layer_indices.is_empty() {
            eprintln!("SKIP: no layers in LBC file");
            return;
        }

        let idx = &lbc.layer_indices[0];
        let offset = idx.layer_offset_bytes;
        let length = idx.layer_length_bytes as usize;

        eprintln!("=== Parallel pread Bandwidth Benchmark ===");
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

        // Warm up: sequential read to prime SSD controller.
        let _ = parallel_pread(lbc_path, offset, length, 1);

        // Sequential (1 thread).
        let start = Instant::now();
        let data_seq = parallel_pread(lbc_path, offset, length, 1).unwrap();
        let elapsed_seq = start.elapsed();
        let bw_seq = length as f64 / 1e9 / elapsed_seq.as_secs_f64();
        eprintln!(
            "Sequential (1 thread): {:.1} MB in {:.2?} = {:.2} GB/s",
            length as f64 / 1e6,
            elapsed_seq,
            bw_seq
        );

        // Parallel: 2, 4, 8 threads.
        for threads in [2, 4, 8] {
            let start = Instant::now();
            let data_par = parallel_pread(lbc_path, offset, length, threads).unwrap();
            let elapsed = start.elapsed();
            let bw = length as f64 / 1e9 / elapsed.as_secs_f64();
            assert_eq!(
                data_par, data_seq,
                "{}-thread read produced different data than sequential",
                threads
            );
            eprintln!(
                "Parallel ({} threads): {:.1} MB in {:.2?} = {:.2} GB/s ({:.1}x vs seq)",
                threads,
                length as f64 / 1e6,
                elapsed,
                bw,
                elapsed_seq.as_secs_f64() / elapsed.as_secs_f64()
            );
        }

        // Also benchmark ExpertReader parallel vs sequential.
        let reader = ExpertReader::open(lbc_path).unwrap();
        let num_experts = lbc.header.hyperparams.num_experts.unwrap_or(0) as usize;
        let top_k = lbc.header.hyperparams.num_active_experts.unwrap_or(2).min(num_experts as u32) as usize;

        if num_experts > 0 && top_k > 0 {
            eprintln!("\n--- ExpertReader: sequential vs parallel (layer 0, top-{}) ---", top_k);

            // Sequential.
            let mut seq_reader = ExpertReader::open(lbc_path).unwrap();
            let start = Instant::now();
            let mut seq_bytes = 0u64;
            for eid in 0..top_k as u32 {
                if let Ok((data, _)) = seq_reader.load_expert(0, eid) {
                    seq_bytes += data.len() as u64;
                }
            }
            let elapsed_seq_expert = start.elapsed();
            eprintln!(
                "Sequential: {} experts, {:.1} MB, {:.2?} ({:.2} GB/s)",
                top_k,
                seq_bytes as f64 / 1e6,
                elapsed_seq_expert,
                seq_bytes as f64 / 1e9 / elapsed_seq_expert.as_secs_f64()
            );

            // Parallel.
            let requests: Vec<(usize, u32)> = (0..top_k as u32).map(|e| (0, e)).collect();
            let start = Instant::now();
            let results = reader.load_experts_parallel(&requests);
            let elapsed_par_expert = start.elapsed();
            let par_bytes: u64 = results.iter()
                .filter_map(|r| r.as_ref().ok().map(|(d, _)| d.len() as u64))
                .sum();
            eprintln!(
                "Parallel:   {} experts, {:.1} MB, {:.2?} ({:.2} GB/s, {:.1}x speedup)",
                top_k,
                par_bytes as f64 / 1e6,
                elapsed_par_expert,
                par_bytes as f64 / 1e9 / elapsed_par_expert.as_secs_f64(),
                elapsed_seq_expert.as_secs_f64() / elapsed_par_expert.as_secs_f64()
            );
        }

        eprintln!("\n=== Benchmark PASS ===");
    }
}
