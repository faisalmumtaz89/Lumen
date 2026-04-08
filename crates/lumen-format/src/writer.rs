//! LBC binary format writer.
//!
//! Serializes an [`LbcHeader`], layer indices, and tensor blobs into the
//! LBC on-disk format. All multi-byte values are little-endian.
//!
//! ## Binary Layout
//!
//! ```text
//! [Header] [LayerIndex * L] [alignment padding]
//! [EmbeddingBlob] [FinalNormBlob] [OutputProjBlob]
//! [LayerBlob_0] ... [LayerBlob_{L-1}]
//! ```

use crate::crc::crc32;
use crate::header::{Endianness, GlobalTensorRange, LbcHeader};
use crate::hyperparams::{ModelHyperparams, RopeScalingType};
use crate::index::{LayerIndex, TensorSlice};
use crate::quantization::{QuantGroupSize, QuantizationDescriptor};
use crate::tokenizer::TokenizerSection;
use std::io::{self, Write};

/// Write a complete LBC file to any `Write` sink.
///
/// `layer_blobs` must have length equal to `header.num_layers`.
/// `global_tensors` is `(embedding, final_norm, output_proj)` raw bytes.
/// `tokenizer` is optional; if present, the tokenizer section is appended
/// after the last layer blob and the header is updated with its offset/CRC.
pub fn write_lbc<W: Write>(
    w: &mut W,
    header: &LbcHeader,
    layer_indices: &[LayerIndex],
    global_tensors: &GlobalTensors,
    layer_blobs: &[&[u8]],
    tokenizer: Option<&TokenizerSection>,
) -> io::Result<()> {
    if layer_indices.len() != header.num_layers as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "layer_indices.len()={} != header.num_layers={}",
                layer_indices.len(),
                header.num_layers
            ),
        ));
    }
    if layer_blobs.len() != header.num_layers as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "layer_blobs.len()={} != header.num_layers={}",
                layer_blobs.len(),
                header.num_layers
            ),
        ));
    }

    // --- Phase 1: compute layout ---
    let header_bytes = serialize_header(header);
    let index_bytes = serialize_layer_indices(layer_indices);

    let raw_prefix = header_bytes.len() + index_bytes.len();
    let alignment: usize = header.alignment.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("alignment {} exceeds platform pointer size", header.alignment),
        )
    })?;
    let padding = align_up(raw_prefix, alignment) - raw_prefix;

    // Global tensors come right after alignment padding
    let globals_start = raw_prefix + padding;
    let globals_total = global_tensors.embedding.len()
        + global_tensors.final_norm.len()
        + global_tensors.output_proj.len();

    // Layer blobs start after globals (aligned)
    let layers_start = align_up(globals_start + globals_total, alignment);

    // --- Phase 2: fix up header offsets ---
    let mut fixed_header = header.clone();
    fixed_header.layer_index_offset = header_bytes.len() as u64;
    fixed_header.payload_offset = globals_start as u64;

    // Global tensor ranges (preserve quant metadata from caller)
    let mut cursor = globals_start as u64;
    fixed_header.embedding = GlobalTensorRange {
        offset: cursor,
        length: global_tensors.embedding.len() as u64,
        quant: header.embedding.quant,
    };
    cursor += global_tensors.embedding.len() as u64;
    fixed_header.final_norm = GlobalTensorRange {
        offset: cursor,
        length: global_tensors.final_norm.len() as u64,
        quant: header.final_norm.quant,
    };
    cursor += global_tensors.final_norm.len() as u64;
    if fixed_header.weight_tying {
        // Weight tying: output_proj shares embedding storage
        fixed_header.output_proj = GlobalTensorRange {
            offset: fixed_header.embedding.offset,
            length: fixed_header.embedding.length,
            quant: fixed_header.embedding.quant,
        };
    } else {
        fixed_header.output_proj = GlobalTensorRange {
            offset: cursor,
            length: global_tensors.output_proj.len() as u64,
            quant: header.output_proj.quant,
        };
    }

    // Fix up layer index offsets relative to file
    let mut fixed_indices = layer_indices.to_vec();
    let mut layer_cursor = layers_start as u64;
    for (i, idx) in fixed_indices.iter_mut().enumerate() {
        idx.layer_offset_bytes = layer_cursor;
        idx.layer_length_bytes = layer_blobs[i].len() as u64;
        layer_cursor += layer_blobs[i].len() as u64;
        // Align each subsequent layer
        layer_cursor = align_up(layer_cursor as usize, alignment) as u64;
    }

    // Compute end of last layer (unaligned) for tokenizer placement
    let last_layer_end = if layer_blobs.is_empty() {
        layers_start as u64
    } else {
        let last = &fixed_indices[fixed_indices.len() - 1];
        last.layer_offset_bytes + last.layer_length_bytes
    };

    // Tokenizer section: serialize, compute CRC, set header offsets
    let tok_bytes = tokenizer.map(|t| t.serialize());
    if let Some(ref tok_data) = tok_bytes {
        let tok_start = align_up(last_layer_end as usize, alignment) as u64;
        fixed_header.tokenizer_section_offset = tok_start;
        fixed_header.tokenizer_section_length = tok_data.len() as u64;
        fixed_header.tokenizer_section_crc32 = crc32(tok_data);
    } else {
        fixed_header.tokenizer_section_offset = 0;
        fixed_header.tokenizer_section_length = 0;
        fixed_header.tokenizer_section_crc32 = 0;
    }

    // Re-serialize header with correct offsets, then compute checksum
    let mut header_bytes = serialize_header(&fixed_header);
    let checksum = crc32(&header_bytes);
    // Patch the checksum field (bytes 12..16 in the header)
    header_bytes[12..16].copy_from_slice(&checksum.to_le_bytes());

    let index_bytes = serialize_layer_indices(&fixed_indices);

    // --- Phase 3: write everything ---
    w.write_all(&header_bytes)?;
    w.write_all(&index_bytes)?;
    write_zeros(w, padding)?;

    // Global tensors
    w.write_all(&global_tensors.embedding)?;
    w.write_all(&global_tensors.final_norm)?;
    w.write_all(&global_tensors.output_proj)?;

    // Padding between globals and layers
    let globals_end = globals_start + globals_total;
    let layers_padding = layers_start - globals_end;
    write_zeros(w, layers_padding)?;

    // Layer blobs with alignment padding
    for (i, blob) in layer_blobs.iter().enumerate() {
        w.write_all(blob)?;
        if i + 1 < layer_blobs.len() {
            let cur = fixed_indices[i].layer_offset_bytes + fixed_indices[i].layer_length_bytes;
            let next = fixed_indices[i + 1].layer_offset_bytes;
            write_zeros(w, (next - cur) as usize)?;
        }
    }

    // Tokenizer section (after last layer, aligned)
    if let Some(ref tok_data) = tok_bytes {
        let tok_padding = fixed_header.tokenizer_section_offset as usize - last_layer_end as usize;
        write_zeros(w, tok_padding)?;
        w.write_all(tok_data)?;
    }

    Ok(())
}

/// Global tensors passed to the writer.
pub struct GlobalTensors {
    pub embedding: Vec<u8>,
    pub final_norm: Vec<u8>,
    pub output_proj: Vec<u8>,
}

/// Write `n` zero bytes using a fixed stack buffer (no heap allocation).
pub(crate) fn write_zeros<W: Write>(w: &mut W, mut n: usize) -> io::Result<()> {
    const ZERO_BUF: [u8; 128] = [0u8; 128];
    while n > 0 {
        let chunk = n.min(ZERO_BUF.len());
        w.write_all(&ZERO_BUF[..chunk])?;
        n -= chunk;
    }
    Ok(())
}

pub(crate) fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment > 0 && alignment.is_power_of_two(), "alignment must be a non-zero power of two");
    (value + alignment - 1) & !(alignment - 1)
}

// ---------- Serialization helpers ----------

/// Serialize the header to bytes (little-endian). Checksum field is initially 0.
pub(crate) fn serialize_header(h: &LbcHeader) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);

    buf.extend_from_slice(&h.magic.to_le_bytes());                  // 0..4
    buf.extend_from_slice(&h.version.to_le_bytes());                // 4..8
    buf.push(match h.endianness { Endianness::Little => 0, Endianness::Big => 1 }); // 8
    buf.extend_from_slice(&[0u8; 3]); // padding to align                           // 9..12
    buf.extend_from_slice(&h.header_checksum.to_le_bytes());        // 12..16

    // Hyperparams
    serialize_hyperparams(&mut buf, &h.hyperparams);

    // Quantization descriptor
    serialize_quant_desc(&mut buf, &h.quantization);

    buf.extend_from_slice(&h.alignment.to_le_bytes());
    buf.extend_from_slice(&h.num_layers.to_le_bytes());
    buf.push(h.has_expert_index as u8);
    buf.extend_from_slice(&[0u8; 3]); // padding
    buf.extend_from_slice(&h.layer_index_offset.to_le_bytes());
    buf.extend_from_slice(&h.expert_index_offset.to_le_bytes());
    buf.extend_from_slice(&h.payload_offset.to_le_bytes());

    // Global tensor ranges
    buf.extend_from_slice(&h.embedding.offset.to_le_bytes());
    buf.extend_from_slice(&h.embedding.length.to_le_bytes());
    buf.extend_from_slice(&h.final_norm.offset.to_le_bytes());
    buf.extend_from_slice(&h.final_norm.length.to_le_bytes());
    buf.extend_from_slice(&h.output_proj.offset.to_le_bytes());
    buf.extend_from_slice(&h.output_proj.length.to_le_bytes());

    // v2: per-tensor quant metadata (3 bytes) + weight_tying (1 byte) + padding (4 bytes)
    buf.push(h.embedding.quant.to_u8());
    buf.push(h.final_norm.quant.to_u8());
    buf.push(h.output_proj.quant.to_u8());
    buf.push(h.weight_tying as u8);
    buf.extend_from_slice(&[0u8; 4]); // padding to 8-byte alignment

    // v3: tokenizer section pointers (24 bytes)
    buf.extend_from_slice(&h.tokenizer_section_offset.to_le_bytes());
    buf.extend_from_slice(&h.tokenizer_section_length.to_le_bytes());
    buf.extend_from_slice(&h.tokenizer_section_crc32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]); // reserved

    buf
}

fn serialize_hyperparams(buf: &mut Vec<u8>, hp: &ModelHyperparams) {
    buf.extend_from_slice(&hp.num_layers.to_le_bytes());
    buf.extend_from_slice(&hp.num_heads.to_le_bytes());
    buf.extend_from_slice(&hp.num_kv_heads.to_le_bytes());
    buf.extend_from_slice(&hp.head_dim.to_le_bytes());
    buf.extend_from_slice(&hp.hidden_dim.to_le_bytes());
    buf.extend_from_slice(&hp.intermediate_dim.to_le_bytes());
    buf.extend_from_slice(&hp.vocab_size.to_le_bytes());
    buf.extend_from_slice(&hp.max_seq_len.to_le_bytes());

    // RoPE params: present flag + data
    match &hp.rope_params {
        Some(rp) => {
            buf.push(1);
            buf.extend_from_slice(&rp.theta.to_le_bytes());
            buf.extend_from_slice(&rp.scaling_factor.to_le_bytes());
            buf.push(match rp.scaling_type {
                RopeScalingType::None => 0,
                RopeScalingType::Linear => 1,
                RopeScalingType::Ntk => 2,
                RopeScalingType::Yarn => 3,
            });
            // rope_neox: NeoX half-split RoPE for Qwen2/Qwen3.5.
            buf.push(if hp.rope_neox { 1u8 } else { 0u8 });
            // rotary_dim as u8: 0 = full head_dim, N = partial.
            let rotary_dim_u8 = hp.rotary_dim.unwrap_or(0).min(255) as u8;
            buf.push(rotary_dim_u8);
        }
        None => {
            buf.push(0);
            // Must match present-path size: theta(4) + scaling_factor(4) + type(1) + rope_neox(1) + rotary_dim(1) = 11
            buf.extend_from_slice(&[0u8; 11]);
        }
    }

    // MoE fields
    buf.extend_from_slice(&hp.num_experts.unwrap_or(0).to_le_bytes());
    buf.extend_from_slice(&hp.num_active_experts.unwrap_or(0).to_le_bytes());
    buf.extend_from_slice(&hp.norm_eps.to_le_bytes());
}

fn serialize_quant_desc(buf: &mut Vec<u8>, qd: &QuantizationDescriptor) {
    buf.push(qd.scheme.to_u8());
    match qd.group_size {
        QuantGroupSize::PerTensor => {
            buf.push(0);
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        QuantGroupSize::PerChannel => {
            buf.push(1);
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        QuantGroupSize::Group(gs) => {
            buf.push(2);
            buf.extend_from_slice(&gs.to_le_bytes());
        }
    }
    buf.extend_from_slice(&qd.block_byte_size.to_le_bytes());
    match qd.scale_offset_in_block {
        Some(off) => {
            buf.push(1);
            buf.extend_from_slice(&off.to_le_bytes());
        }
        None => {
            buf.push(0);
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
    }
}

fn serialize_tensor_slice(buf: &mut Vec<u8>, ts: &TensorSlice) {
    buf.extend_from_slice(&ts.offset.to_le_bytes());
    buf.extend_from_slice(&ts.length.to_le_bytes());
    buf.push(ts.quant.to_u8());
    buf.extend_from_slice(&[0u8; 7]); // padding to 24 bytes per slice
}

pub(crate) fn serialize_layer_indices(indices: &[LayerIndex]) -> Vec<u8> {
    // Each index: offset(8) + length(8) + 9 tensor slices * 24 bytes + bias_flags(8)
    //   + up to 3 optional bias slices * 24 bytes
    //   + moe_flags(8) + optional router slice(24) + N * expert slices(3*24)
    let mut buf = Vec::with_capacity(indices.len() * 512);
    for idx in indices {
        buf.extend_from_slice(&idx.layer_offset_bytes.to_le_bytes());
        buf.extend_from_slice(&idx.layer_length_bytes.to_le_bytes());
        let st = &idx.subtensors;
        for slice in [
            &st.wq, &st.wk, &st.wv, &st.wo,
            &st.w_gate, &st.w_up, &st.w_down,
            &st.attn_norm, &st.ffn_norm,
        ] {
            serialize_tensor_slice(&mut buf, slice);
        }
        // Bias flags byte: bit 0 = bq, bit 1 = bk, bit 2 = bv
        let bias_flags: u8 =
            (st.bq.is_some() as u8)
            | ((st.bk.is_some() as u8) << 1)
            | ((st.bv.is_some() as u8) << 2);
        buf.push(bias_flags);
        buf.extend_from_slice(&[0u8; 7]); // padding to 8 bytes
        if let Some(ref s) = st.bq { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.bk { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.bv { serialize_tensor_slice(&mut buf, s); }

        // MoE fields: num_experts (u32) + has_router (u8) + padding (3 bytes) = 8 bytes
        // Followed by optional router slice and expert slices.
        let num_experts = st.experts.as_ref().map_or(0u32, |e| e.len() as u32);
        let has_router = st.router_weight.is_some();
        buf.extend_from_slice(&num_experts.to_le_bytes());
        buf.push(has_router as u8);
        buf.extend_from_slice(&[0u8; 3]); // padding
        if let Some(ref router) = st.router_weight {
            serialize_tensor_slice(&mut buf, router);
        }
        if let Some(ref experts) = st.experts {
            for expert in experts {
                serialize_tensor_slice(&mut buf, &expert.gate);
                serialize_tensor_slice(&mut buf, &expert.up);
                serialize_tensor_slice(&mut buf, &expert.down);
            }
        }

        // Extended fields (v2.1+): shared expert, attn gate/post_norm, SSM, layer_type.
        // Format: ext_flags(u32) + padding(4) = 8 bytes header, then optional slices.
        //   bit 0: shared_expert_gate
        //   bit 1: shared_expert_up
        //   bit 2: shared_expert_down
        //   bit 3: attn_gate
        //   bit 4: attn_post_norm
        //   bit 5: ssm_a
        //   bit 6: ssm_conv1d
        //   bit 7: ssm_dt
        //   bit 8: ssm_beta
        //   bit 9: ssm_alpha
        //   bit 10: ssm_norm
        //   bit 11: ssm_out
        //   bit 12: has_layer_type
        //   bit 13: attn_q_norm
        //   bit 14: attn_k_norm
        //   bit 15: ffn_gate_inp_shexp
        let ext_flags: u32 =
            (st.shared_expert_gate.is_some() as u32)
            | ((st.shared_expert_up.is_some() as u32) << 1)
            | ((st.shared_expert_down.is_some() as u32) << 2)
            | ((st.attn_gate.is_some() as u32) << 3)
            | ((st.attn_post_norm.is_some() as u32) << 4)
            | ((st.ssm_a.is_some() as u32) << 5)
            | ((st.ssm_conv1d.is_some() as u32) << 6)
            | ((st.ssm_dt.is_some() as u32) << 7)
            | ((st.ssm_beta.is_some() as u32) << 8)
            | ((st.ssm_alpha.is_some() as u32) << 9)
            | ((st.ssm_norm.is_some() as u32) << 10)
            | ((st.ssm_out.is_some() as u32) << 11)
            | ((st.layer_type.is_some() as u32) << 12)
            | ((st.attn_q_norm.is_some() as u32) << 13)
            | ((st.attn_k_norm.is_some() as u32) << 14)
            | ((st.ffn_gate_inp_shexp.is_some() as u32) << 15);
        buf.extend_from_slice(&ext_flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // padding to 8 bytes

        if let Some(ref s) = st.shared_expert_gate { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.shared_expert_up { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.shared_expert_down { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.attn_gate { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.attn_post_norm { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_a { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_conv1d { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_dt { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_beta { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_alpha { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_norm { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ssm_out { serialize_tensor_slice(&mut buf, s); }
        if let Some(lt) = st.layer_type {
            buf.push(lt);
            buf.extend_from_slice(&[0u8; 7]); // padding to 8 bytes
        }
        if let Some(ref s) = st.attn_q_norm { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.attn_k_norm { serialize_tensor_slice(&mut buf, s); }
        if let Some(ref s) = st.ffn_gate_inp_shexp { serialize_tensor_slice(&mut buf, s); }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperparams::RopeParams;
    use crate::quantization::QuantScheme;

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 128), 256);
    }

    #[test]
    #[should_panic(expected = "alignment must be a non-zero power of two")]
    fn align_up_zero_panics() {
        align_up(1, 0);
    }

    #[test]
    #[should_panic(expected = "alignment must be a non-zero power of two")]
    fn align_up_non_power_of_two_panics() {
        align_up(1, 3);
    }

    #[test]
    fn header_roundtrip_checksum() {
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
        let bytes = serialize_header(&header);
        // Checksum field should be 0 in initial serialization
        let cksum_bytes = &bytes[12..16];
        assert_eq!(cksum_bytes, &[0, 0, 0, 0]);
    }
}
