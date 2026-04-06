//! LBC binary format reader.
//!
//! Deserializes an LBC file into an [`LbcFile`] handle containing the header,
//! layer indices, and the file path for lazy layer reads.

use crate::crc::{crc32_finalize, crc32_update, CRC32_INIT};
use crate::header::{Endianness, GlobalTensorRange, LbcHeader, LBC_MAGIC, LBC_VERSION};
use crate::hyperparams::{ModelHyperparams, RopeParams, RopeScalingType};
use crate::index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
use crate::quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
use crate::FormatError;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// An opened LBC file handle.
///
/// Holds the parsed header and layer index. Actual layer blob data is read
/// lazily via the storage backend.
#[derive(Debug)]
pub struct LbcFile {
    pub header: LbcHeader,
    pub layer_indices: Vec<LayerIndex>,
    pub path: PathBuf,
}

/// The fixed-size header is at most this many bytes. This is a generous
/// upper bound; the actual header is smaller, but we read a bit extra to
/// be safe and avoid a second read in most cases.
const MAX_HEADER_BYTES: usize = 4096;

/// Maximum layer index entry size: layer_offset(8) + layer_length(8) +
/// 9 * TensorSlice(24) + bias_flags(8) + up to 3 bias TensorSlice(24) +
/// moe_header(8) + router TensorSlice(24) + 256 experts * 3 TensorSlice(24) +
/// ext_flags(8) + up to 12 TensorSlice(24) + layer_type(8) = ~19200 bytes.
/// We use a generous upper bound to accommodate models with many experts (e.g. 256).
const MAX_LAYER_INDEX_ENTRY_SIZE: usize = 20480;

impl LbcFile {
    /// Open and parse an LBC file from disk.
    ///
    /// Only reads the header and layer index sections (typically a few KB),
    /// NOT the entire file. This avoids loading multi-GB model data into
    /// memory just to parse metadata.
    pub fn open(path: &Path) -> Result<Self, FormatError> {
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();

        // Step 1: Read enough bytes for the header. The fixed header is
        // well under MAX_HEADER_BYTES, but cap at file size.
        let initial_read = MAX_HEADER_BYTES.min(file_len as usize);
        let mut header_buf = vec![0u8; initial_read];
        file.read_exact(&mut header_buf)?;

        // Step 2: Do a lightweight scan of the header to extract
        // layer_index_offset and num_layers so we know how much more to read.
        let (layer_index_offset, num_layers) = peek_header_offsets(&header_buf)?;

        let index_end = layer_index_offset as usize
            + (num_layers as usize) * MAX_LAYER_INDEX_ENTRY_SIZE;

        // Step 3: If the initial read already covers everything, parse from it.
        // Otherwise, read the additional bytes we need.
        let data = if index_end <= header_buf.len() {
            header_buf
        } else if index_end as u64 <= file_len {
            // Extend the buffer to include the layer index section
            header_buf.resize(index_end, 0);
            file.seek(SeekFrom::Start(initial_read as u64))?;
            file.read_exact(&mut header_buf[initial_read..])?;
            header_buf
        } else {
            // File is too small -- fall back to reading the whole thing and
            // let parse_lbc produce the appropriate truncation error.
            file.seek(SeekFrom::Start(0))?;
            let mut all = Vec::new();
            file.read_to_end(&mut all)?;
            all
        };

        let (header, layer_indices) = parse_lbc(&data)?;
        Ok(Self {
            header,
            layer_indices,
            path: path.to_path_buf(),
        })
    }

    /// Parse from an in-memory buffer (useful for tests).
    pub fn from_bytes(data: &[u8], path: PathBuf) -> Result<Self, FormatError> {
        let (header, layer_indices) = parse_lbc(data)?;
        Ok(Self {
            header,
            layer_indices,
            path,
        })
    }
}

/// Lightweight header peek: extract layer_index_offset and num_layers from
/// the header bytes without doing a full validation. This lets us know how
/// many additional bytes to read for the layer index section.
fn peek_header_offsets(data: &[u8]) -> Result<(u64, u32), FormatError> {
    let mut c = Cursor::new(data);

    // magic(4) + version(4) + endianness(1) + padding(3) + checksum(4) = 16
    c.skip(16)?;
    // hyperparams: 8 * u32(4) + has_rope(1) + rope_block(11) + 3 * u32(4) = 56
    c.skip(56)?;
    // quantization descriptor: 1+1+4+4+1+4 = 15
    c.skip(15)?;
    // alignment(8)
    c.skip(8)?;
    // num_layers
    let num_layers = c.read_u32()?;
    // has_expert_index(1) + padding(3)
    c.skip(4)?;
    // layer_index_offset
    let layer_index_offset = c.read_u64()?;

    Ok((layer_index_offset, num_layers))
}

/// Parse the header and layer indices from raw bytes.
fn parse_lbc(data: &[u8]) -> Result<(LbcHeader, Vec<LayerIndex>), FormatError> {
    let mut cursor = Cursor::new(data);

    // Header
    let magic = cursor.read_u32()?;
    if magic != LBC_MAGIC {
        return Err(FormatError::InvalidMagic {
            expected: LBC_MAGIC,
            found: magic,
        });
    }

    let version = cursor.read_u32()?;
    if version > LBC_VERSION {
        return Err(FormatError::UnsupportedVersion {
            version,
            max_supported: LBC_VERSION,
        });
    }

    let endianness_byte = cursor.read_u8()?;
    let endianness = match endianness_byte {
        0 => Endianness::Little,
        1 => Endianness::Big,
        _ => return Err(FormatError::InvalidEndianness(endianness_byte)),
    };
    cursor.skip(3)?; // padding

    let stored_checksum = cursor.read_u32()?;

    // Verify checksum: compute CRC32 over header bytes with checksum field zeroed
    // We'll verify checksum after parsing the full header

    let hyperparams = parse_hyperparams(&mut cursor)?;
    let quantization = parse_quant_desc(&mut cursor)?;

    let alignment = cursor.read_u64()?;
    let num_layers = cursor.read_u32()?;
    let has_expert_index = cursor.read_u8()? != 0;
    cursor.skip(3)?; // padding
    let layer_index_offset = cursor.read_u64()?;
    let expert_index_offset = cursor.read_u64()?;
    let payload_offset = cursor.read_u64()?;

    // Global tensor ranges (offset + length)
    let embed_offset = cursor.read_u64()?;
    let embed_length = cursor.read_u64()?;
    let norm_offset = cursor.read_u64()?;
    let norm_length = cursor.read_u64()?;
    let proj_offset = cursor.read_u64()?;
    let proj_length = cursor.read_u64()?;

    // v2: per-tensor quant metadata + weight_tying (8 bytes total)
    let (embed_quant, norm_quant, proj_quant, weight_tying) = if version >= 2 {
        let eq = QuantScheme::from_u8(cursor.read_u8()?).unwrap_or(QuantScheme::F32);
        let nq = QuantScheme::from_u8(cursor.read_u8()?).unwrap_or(QuantScheme::F32);
        let pq = QuantScheme::from_u8(cursor.read_u8()?).unwrap_or(QuantScheme::F32);
        let wt = cursor.read_u8()? != 0;
        cursor.skip(4)?; // padding
        (eq, nq, pq, wt)
    } else {
        (QuantScheme::F32, QuantScheme::F32, QuantScheme::F32, false)
    };

    let embedding = GlobalTensorRange { offset: embed_offset, length: embed_length, quant: embed_quant };
    let final_norm = GlobalTensorRange { offset: norm_offset, length: norm_length, quant: norm_quant };
    let output_proj = GlobalTensorRange { offset: proj_offset, length: proj_length, quant: proj_quant };

    let header_size = cursor.pos;

    // Verify CRC32 checksum (streaming — no heap allocation)
    let mut state = CRC32_INIT;
    state = crc32_update(state, &data[..12]);          // bytes before checksum field
    state = crc32_update(state, &[0u8; 4]);            // zeroed checksum field
    state = crc32_update(state, &data[16..header_size]); // bytes after checksum field
    let computed_checksum = crc32_finalize(state);
    if stored_checksum != computed_checksum {
        return Err(FormatError::ChecksumMismatch {
            expected: stored_checksum,
            computed: computed_checksum,
        });
    }

    let header = LbcHeader {
        magic,
        version,
        endianness,
        header_checksum: stored_checksum,
        hyperparams,
        quantization,
        alignment,
        num_layers,
        has_expert_index,
        layer_index_offset,
        expert_index_offset,
        payload_offset,
        embedding,
        final_norm,
        output_proj,
        weight_tying,
    };

    header.validate()?;

    // Parse layer indices — safe truncation check for 32-bit targets
    if layer_index_offset > usize::MAX as u64 {
        return Err(FormatError::UnexpectedEof {
            needed: layer_index_offset,
            available: data.len() as u64,
        });
    }
    cursor.pos = layer_index_offset as usize;
    let mut layer_indices = Vec::with_capacity(num_layers as usize);
    for _ in 0..num_layers {
        layer_indices.push(parse_layer_index(&mut cursor)?);
    }

    Ok((header, layer_indices))
}

fn parse_hyperparams(c: &mut Cursor<'_>) -> Result<ModelHyperparams, FormatError> {
    let num_layers = c.read_u32()?;
    let num_heads = c.read_u32()?;
    let num_kv_heads = c.read_u32()?;
    let head_dim = c.read_u32()?;
    let hidden_dim = c.read_u32()?;
    let intermediate_dim = c.read_u32()?;
    let vocab_size = c.read_u32()?;
    let max_seq_len = c.read_u32()?;

    let has_rope = c.read_u8()? != 0;
    let (rope_params, rope_neox, rotary_dim_raw) = if has_rope {
        let theta = c.read_f32()?;
        let scaling_factor = c.read_f32()?;
        let scaling_type = match c.read_u8()? {
            0 => RopeScalingType::None,
            1 => RopeScalingType::Linear,
            2 => RopeScalingType::Ntk,
            3 => RopeScalingType::Yarn,
            other => return Err(FormatError::InvalidRopeScalingType(other)),
        };
        // rope_neox: NeoX half-split RoPE for Qwen2/Qwen3.5.
        // Backward compatible: old files had 0x00 padding here (= false).
        let rope_neox = c.read_u8()? != 0;
        // rotary_dim as u8: 0 = full head_dim, N = partial.
        let rotary_dim_u8 = c.read_u8()?;
        (Some(RopeParams {
            theta,
            scaling_factor,
            scaling_type,
        }), rope_neox, rotary_dim_u8)
    } else {
        // Must match present-path size: theta(4) + scaling_factor(4) + type(1) + rope_neox(1) + rotary_dim(1) = 11
        c.skip(11)?;
        (None, false, 0u8)
    };

    let num_experts_raw = c.read_u32()?;
    let num_active_raw = c.read_u32()?;
    let norm_eps = c.read_f32()?;

    Ok(ModelHyperparams {
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        max_seq_len,
        rope_params,
        num_experts: if num_experts_raw > 0 { Some(num_experts_raw) } else { None },
        num_active_experts: if num_active_raw > 0 { Some(num_active_raw) } else { None },
        norm_eps,
        rotary_dim: if rotary_dim_raw > 0 { Some(rotary_dim_raw as u32) } else { None },
        rope_neox,
    })
}

fn parse_quant_desc(c: &mut Cursor<'_>) -> Result<QuantizationDescriptor, FormatError> {
    let scheme = QuantScheme::from_u8(c.read_u8()?)?;
    let group_tag = c.read_u8()?;
    let group_value = c.read_u32()?;
    let group_size = match group_tag {
        0 => QuantGroupSize::PerTensor,
        1 => QuantGroupSize::PerChannel,
        2 => QuantGroupSize::Group(group_value),
        _ => return Err(FormatError::UnsupportedQuantization(
            format!("unknown group size tag: {group_tag}"),
        )),
    };
    let block_byte_size = c.read_u32()?;
    let has_scale = c.read_u8()? != 0;
    let scale_value = c.read_u32()?;
    let scale_offset_in_block = if has_scale { Some(scale_value) } else { None };

    Ok(QuantizationDescriptor {
        scheme,
        group_size,
        block_byte_size,
        scale_offset_in_block,
    })
}

fn parse_tensor_slice(c: &mut Cursor<'_>) -> Result<TensorSlice, FormatError> {
    let offset = c.read_u64()?;
    let length = c.read_u64()?;
    let quant = QuantScheme::from_u8(c.read_u8()?)?;
    c.skip(7)?; // padding
    Ok(TensorSlice { offset, length, quant })
}

fn parse_layer_index(c: &mut Cursor<'_>) -> Result<LayerIndex, FormatError> {
    let layer_offset_bytes = c.read_u64()?;
    let layer_length_bytes = c.read_u64()?;

    let wq = parse_tensor_slice(c)?;
    let wk = parse_tensor_slice(c)?;
    let wv = parse_tensor_slice(c)?;
    let wo = parse_tensor_slice(c)?;
    let w_gate = parse_tensor_slice(c)?;
    let w_up = parse_tensor_slice(c)?;
    let w_down = parse_tensor_slice(c)?;
    let attn_norm = parse_tensor_slice(c)?;
    let ffn_norm = parse_tensor_slice(c)?;

    // Read optional bias flags + slices (added for Qwen2-family models).
    // For backward compatibility with pre-bias LBC files, check remaining data.
    let (bq, bk, bv) = if c.pos + 8 <= c.data.len() {
        let bias_flags = c.read_u8()?;
        c.skip(7)?; // padding to 8 bytes
        let bq = if bias_flags & 0x01 != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let bk = if bias_flags & 0x02 != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let bv = if bias_flags & 0x04 != 0 { Some(parse_tensor_slice(c)?) } else { None };
        (bq, bk, bv)
    } else {
        (None, None, None)
    };

    // Read optional MoE fields: num_experts(4) + has_router(1) + padding(3) = 8 bytes.
    // For backward compatibility with pre-MoE LBC files, check remaining data.
    let (router_weight, experts) = if c.pos + 8 <= c.data.len() {
        let num_experts = c.read_u32()?;
        let has_router = c.read_u8()? != 0;
        c.skip(3)?; // padding

        let router_weight = if has_router {
            Some(parse_tensor_slice(c)?)
        } else {
            None
        };

        let experts = if num_experts > 0 {
            let mut expert_vec = Vec::with_capacity(num_experts as usize);
            for _ in 0..num_experts {
                let gate = parse_tensor_slice(c)?;
                let up = parse_tensor_slice(c)?;
                let down = parse_tensor_slice(c)?;
                expert_vec.push(ExpertSlice { gate, up, down });
            }
            Some(expert_vec)
        } else {
            None
        };

        (router_weight, experts)
    } else {
        (None, None)
    };

    // Read optional extended fields: ext_flags(4) + padding(4) = 8 bytes.
    // For backward compatibility with pre-extended LBC files, check remaining data.
    let (shared_expert_gate, shared_expert_up, shared_expert_down,
         attn_gate, attn_post_norm,
         ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
         layer_type,
         attn_q_norm, attn_k_norm, ffn_gate_inp_shexp) = if c.pos + 8 <= c.data.len() {
        let ext_flags = c.read_u32()?;
        c.skip(4)?; // padding

        let shared_expert_gate = if ext_flags & (1 << 0) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let shared_expert_up = if ext_flags & (1 << 1) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let shared_expert_down = if ext_flags & (1 << 2) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let attn_gate = if ext_flags & (1 << 3) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let attn_post_norm = if ext_flags & (1 << 4) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_a = if ext_flags & (1 << 5) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_conv1d = if ext_flags & (1 << 6) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_dt = if ext_flags & (1 << 7) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_beta = if ext_flags & (1 << 8) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_alpha = if ext_flags & (1 << 9) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_norm = if ext_flags & (1 << 10) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ssm_out = if ext_flags & (1 << 11) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let layer_type = if ext_flags & (1 << 12) != 0 {
            let lt = c.read_u8()?;
            c.skip(7)?; // padding
            Some(lt)
        } else {
            None
        };
        let attn_q_norm = if ext_flags & (1 << 13) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let attn_k_norm = if ext_flags & (1 << 14) != 0 { Some(parse_tensor_slice(c)?) } else { None };
        let ffn_gate_inp_shexp = if ext_flags & (1 << 15) != 0 { Some(parse_tensor_slice(c)?) } else { None };

        (shared_expert_gate, shared_expert_up, shared_expert_down,
         attn_gate, attn_post_norm,
         ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
         layer_type,
         attn_q_norm, attn_k_norm, ffn_gate_inp_shexp)
    } else {
        (None, None, None, None, None, None, None, None, None, None, None, None, None,
         None, None, None)
    };

    Ok(LayerIndex {
        layer_offset_bytes,
        layer_length_bytes,
        subtensors: SubtensorOffsets {
            wq, wk, wv, wo,
            bq, bk, bv,
            w_gate, w_up, w_down,
            attn_norm, ffn_norm,
            router_weight,
            experts,
            shared_expert_gate, shared_expert_up, shared_expert_down,
            attn_gate, attn_post_norm,
            ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out,
            attn_q_norm, attn_k_norm, ffn_gate_inp_shexp,
            layer_type,
        },
    })
}

// ---------- Minimal cursor for zero-copy parsing ----------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<(), FormatError> {
        // Use subtraction instead of addition to avoid usize overflow
        let remaining = self.data.len() - self.pos; // safe: pos <= data.len()
        if n > remaining {
            Err(FormatError::UnexpectedEof {
                needed: self.pos as u64 + n as u64,
                available: self.data.len() as u64,
            })
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8, FormatError> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, FormatError> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, FormatError> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32, FormatError> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn skip(&mut self, n: usize) -> Result<(), FormatError> {
        self.ensure(n)?;
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_lbc, GlobalTensors};

    fn make_test_header() -> (LbcHeader, Vec<LayerIndex>) {
        let hp = ModelHyperparams {
            num_layers: 2,
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
            rotary_dim: None,
            rope_neox: false,
        };
        let qd = QuantizationDescriptor {
            scheme: QuantScheme::F32,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        };
        let header = LbcHeader::new(hp, qd);

        // Create simple layer indices (offsets will be fixed by writer)
        let make_slice = |off: u64, len: u64| TensorSlice {
            offset: off,
            length: len,
            quant: QuantScheme::F32,
        };
        let subtensors = SubtensorOffsets {
            wq: make_slice(0, 128),
            wk: make_slice(128, 128),
            wv: make_slice(256, 128),
            wo: make_slice(384, 128),
            bq: None, bk: None, bv: None,
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
            layer_offset_bytes: 0, // will be fixed by writer
            layer_length_bytes: 1344,
            subtensors,
        };
        let indices = vec![idx.clone(), idx];

        (header, indices)
    }

    #[test]
    fn roundtrip() {
        let (header, indices) = make_test_header();
        let embedding = vec![1u8; 32 * 8 * 4]; // vocab_size * hidden_dim * f32
        let final_norm = vec![2u8; 8 * 4]; // hidden_dim * f32
        let output_proj = vec![3u8; 32 * 8 * 4]; // vocab_size * hidden_dim * f32
        let globals = GlobalTensors {
            embedding: embedding.clone(),
            final_norm: final_norm.clone(),
            output_proj: output_proj.clone(),
        };

        let layer_blob = vec![42u8; 1344];
        let blobs: Vec<&[u8]> = vec![&layer_blob, &layer_blob];

        let mut out = Vec::new();
        write_lbc(&mut out, &header, &indices, &globals, &blobs).unwrap();

        // Parse it back
        let lbc = LbcFile::from_bytes(&out, PathBuf::from("test.lbc")).unwrap();
        assert_eq!(lbc.header.magic, LBC_MAGIC);
        assert_eq!(lbc.header.version, 2);
        assert_eq!(lbc.header.num_layers, 2);
        assert_eq!(lbc.header.hyperparams.hidden_dim, 8);
        assert_eq!(lbc.header.hyperparams.vocab_size, 32);
        assert_eq!(lbc.layer_indices.len(), 2);

        // Verify global tensor ranges
        assert_eq!(lbc.header.embedding.length, embedding.len() as u64);
        assert_eq!(lbc.header.final_norm.length, final_norm.len() as u64);
        assert_eq!(lbc.header.output_proj.length, output_proj.len() as u64);

        // Read global tensors back
        let emb_start = lbc.header.embedding.offset as usize;
        let emb_end = emb_start + lbc.header.embedding.length as usize;
        assert_eq!(&out[emb_start..emb_end], &embedding[..]);

        // Read layer blob back
        let l0 = &lbc.layer_indices[0];
        let blob_start = l0.layer_offset_bytes as usize;
        let blob_end = blob_start + l0.layer_length_bytes as usize;
        assert_eq!(&out[blob_start..blob_end], &layer_blob[..]);
    }

    #[test]
    fn bad_magic() {
        let data = vec![0u8; 256];
        let result = LbcFile::from_bytes(&data, PathBuf::from("bad.lbc"));
        assert!(result.is_err());
        match result.unwrap_err() {
            FormatError::InvalidMagic { .. } => {}
            other => panic!("expected InvalidMagic, got: {other}"),
        }
    }

    #[test]
    fn truncated_file() {
        let data = vec![0x4C, 0x42, 0x43, 0x01]; // valid magic, then nothing
        let result = LbcFile::from_bytes(&data, PathBuf::from("truncated.lbc"));
        assert!(result.is_err());
    }

    #[test]
    fn moe_roundtrip() {
        use crate::index::ExpertSlice;

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
            num_experts: Some(4),
            num_active_experts: Some(2),
            norm_eps: 1e-5,
            rotary_dim: None,
            rope_neox: false,
        };
        let qd = QuantizationDescriptor {
            scheme: QuantScheme::F32,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        };
        let header = LbcHeader::new(hp, qd);

        // Build a MoE layer with router + 4 experts
        // Attention: wq=128, wk=128, wv=128, wo=128
        // Norms: attn_norm=32, ffn_norm=32
        // Router: 4*8*4=128
        // 4 experts * (gate=512, up=512, down=512) = 4*1536 = 6144
        // Total: 512 + 64 + 128 + 6144 = 6848
        let make_slice = |off: u64, len: u64| TensorSlice {
            offset: off,
            length: len,
            quant: QuantScheme::F32,
        };

        let mut offset = 0u64;
        let wq = make_slice(offset, 128); offset += 128;
        let wk = make_slice(offset, 128); offset += 128;
        let wv = make_slice(offset, 128); offset += 128;
        let wo = make_slice(offset, 128); offset += 128;
        let attn_norm = make_slice(offset, 32); offset += 32;
        let ffn_norm = make_slice(offset, 32); offset += 32;
        let router = make_slice(offset, 128); offset += 128;

        let mut experts = Vec::new();
        for _ in 0..4 {
            let gate = make_slice(offset, 512); offset += 512;
            let up = make_slice(offset, 512); offset += 512;
            let down = make_slice(offset, 512); offset += 512;
            experts.push(ExpertSlice { gate, up, down });
        }

        let blob_size = offset;
        let subtensors = SubtensorOffsets {
            wq, wk, wv, wo,
            bq: None, bk: None, bv: None,
            w_gate: make_slice(0, 0),
            w_up: make_slice(0, 0),
            w_down: make_slice(0, 0),
            attn_norm, ffn_norm,
            router_weight: Some(router),
            experts: Some(experts),
            shared_expert_gate: None, shared_expert_up: None, shared_expert_down: None,
            attn_gate: None, attn_post_norm: None,
            ssm_a: None, ssm_conv1d: None, ssm_dt: None,
            ssm_beta: None, ssm_alpha: None, ssm_norm: None, ssm_out: None,
            attn_q_norm: None, attn_k_norm: None, ffn_gate_inp_shexp: None,
            layer_type: None,
        };
        let idx = LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors,
        };
        let indices = vec![idx];

        let globals = GlobalTensors {
            embedding: vec![1u8; 32 * 8 * 4],
            final_norm: vec![2u8; 8 * 4],
            output_proj: vec![3u8; 32 * 8 * 4],
        };

        let layer_blob = vec![42u8; blob_size as usize];
        let blobs: Vec<&[u8]> = vec![&layer_blob];

        let mut out = Vec::new();
        write_lbc(&mut out, &header, &indices, &globals, &blobs).unwrap();

        // Parse it back
        let lbc = LbcFile::from_bytes(&out, PathBuf::from("moe_test.lbc")).unwrap();
        assert_eq!(lbc.header.num_layers, 1);
        assert_eq!(lbc.header.hyperparams.num_experts, Some(4));
        assert_eq!(lbc.header.hyperparams.num_active_experts, Some(2));
        assert!(lbc.header.has_expert_index);
        assert_eq!(lbc.layer_indices.len(), 1);

        let parsed = &lbc.layer_indices[0];
        parsed.validate(0).unwrap();

        // Verify MoE fields round-trip correctly
        let st = &parsed.subtensors;
        assert!(st.router_weight.is_some());
        let parsed_router = st.router_weight.as_ref().unwrap();
        assert_eq!(parsed_router.offset, 576); // after wq+wk+wv+wo+attn_norm+ffn_norm
        assert_eq!(parsed_router.length, 128);

        let parsed_experts = st.experts.as_ref().unwrap();
        assert_eq!(parsed_experts.len(), 4);

        // Verify first expert offsets
        assert_eq!(parsed_experts[0].gate.offset, 704); // 576 + 128
        assert_eq!(parsed_experts[0].gate.length, 512);
        assert_eq!(parsed_experts[0].up.offset, 1216); // 704 + 512
        assert_eq!(parsed_experts[0].up.length, 512);
        assert_eq!(parsed_experts[0].down.offset, 1728); // 1216 + 512
        assert_eq!(parsed_experts[0].down.length, 512);

        // Verify last expert offsets
        assert_eq!(parsed_experts[3].gate.offset, 704 + 3 * 1536); // 3 experts * 1536 each
        assert_eq!(parsed_experts[3].down.offset, 704 + 3 * 1536 + 1024);

        // Dense FFN fields should be zero-length
        assert_eq!(st.w_gate.length, 0);
        assert_eq!(st.w_up.length, 0);
        assert_eq!(st.w_down.length, 0);
    }
}
