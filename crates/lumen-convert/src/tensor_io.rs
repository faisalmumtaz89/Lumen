//! Shared tensor I/O helpers for reading GGUF tensors and appending to LBC blobs.

use crate::convert::{ConvertError, ConvertTarget};
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use lumen_format::index::TensorSlice;
use lumen_format::quantization::QuantScheme;
use std::io::{Read, Seek, SeekFrom};

// ---------------------------------------------------------------------------
// Target-aware K-quant upcast (Metal compatibility)
// ---------------------------------------------------------------------------

/// Returns true if this GGML type is a K-quant family member
/// (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K).
///
/// Metal has no K-quant matmul or dequant kernels -- any layer tensor stored
/// as a K-quant type must be upcast to Q8_0 at convert time so Metal can
/// dispatch to its tiled Q8_0 matmul kernels at runtime. CUDA, by contrast,
/// dequantizes K-quant planes host-side at load
/// (lumen-runtime/src/cuda/gpu_buffers.rs).
pub(crate) fn is_k_quant(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
    )
}

/// Returns true if a Metal-target conversion must upcast this GGML type to
/// Q8_0: the K-quant family plus legacy Q5_0, none of which the Metal
/// backend has matmul or dequant kernels for (its resident loader accepts
/// only F32/F16/BF16/Q8_0/Q4_0 for dense layer slices). Every Metal upcast gate — here and in
/// the per-arch slice planners, which must mirror this byte layout — routes
/// through this predicate so the planner and the blob writer cannot diverge.
pub(crate) fn metal_needs_upcast(t: GgmlType) -> bool {
    is_k_quant(t) || t == GgmlType::Q5_0
}

// `effective_lbc_quant_for_target` and `effective_byte_size_for_target`
// helpers were considered but not needed: each layer-shape computation site
// (`compute_layer_shape_qwen35{,moe}` + `compute_stacked_slice`) handles the
// K-quant -> Q8_0 sizing inline. Keeping the size logic inline avoids
// double-dispatch and keeps the per-tensor branching explicit.

// ---------------------------------------------------------------------------
// Tensor name helpers
// ---------------------------------------------------------------------------

pub(crate) fn layer_tensor_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

/// Build per-expert tensor name: blk.{layer}.ffn_{kind}.{expert}.weight
pub(crate) fn expert_tensor_name(layer: usize, kind: &str, expert: usize) -> String {
    format!("blk.{layer}.ffn_{kind}.{expert}.weight")
}

// ---------------------------------------------------------------------------
// Tensor data reading
// ---------------------------------------------------------------------------

pub(crate) fn read_tensor_data<R: Read + Seek>(
    reader: &mut R,
    gguf: &GgufFile,
    tensor: &GgufTensorInfo,
) -> Result<Vec<u8>, ConvertError> {
    let offset = gguf.tensor_data_offset(tensor);
    let size = tensor.byte_size().unwrap_or(0);
    reader.seek(SeekFrom::Start(offset))?;
    read_exact_bounded(reader, size, &tensor.name)
}

/// Read exactly `size` bytes for a tensor. A malformed file can declare a
/// tensor far larger than it holds. A plain `vec![0u8; declared]` aborts on
/// the allocation; `try_reserve_exact` + `resize` still OOM-kills because
/// zeroing the pages faults them all in. This reserves the exact size ONCE
/// (fallibly — a size the allocator cannot provide returns a clean error,
/// and `reserve` touches no pages, so an over-commit does not fault the
/// declared size), then fills it with fixed 8 MiB `read_exact` chunks. The
/// single upfront reservation means the fill never reallocates (no
/// quadratic copy, no ~2x transient heap under any allocator), and physical
/// pages are backed only by bytes actually read — a short file hits EOF in
/// the first chunk and errors cleanly. Reader-agnostic: no `SeekFrom::End`
/// (the sharded reader rejects it).
pub(crate) fn read_exact_bounded<R: Read>(
    reader: &mut R,
    size: u64,
    name: &str,
) -> Result<Vec<u8>, ConvertError> {
    const CHUNK: usize = 8 << 20;
    let size_usize = usize::try_from(size).map_err(|_| {
        ConvertError::UnsupportedModel(format!(
            "tensor '{name}' declares {size} bytes, beyond addressable memory"
        ))
    })?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(size_usize).map_err(|_| {
        ConvertError::UnsupportedModel(format!(
            "tensor '{name}' declares {size} bytes, which cannot be allocated \
             (malformed file)"
        ))
    })?;
    let mut staging = vec![0u8; CHUNK.min(size_usize.max(1))];
    let mut remaining = size_usize;
    while remaining > 0 {
        let want = CHUNK.min(remaining);
        reader.read_exact(&mut staging[..want]).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                ConvertError::UnsupportedModel(format!(
                    "tensor '{name}' declares {size} bytes but fewer are \
                     available in the file (malformed file)"
                ))
            } else {
                ConvertError::Io(e)
            }
        })?;
        buf.extend_from_slice(&staging[..want]);
        remaining -= want;
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Blob writing helpers
// ---------------------------------------------------------------------------

/// Read and append a single tensor to the blob, optionally dequantizing to F32.
pub(crate) fn append_tensor_to_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    tensor_name: &str,
    dequantize: bool,
) -> Result<(), ConvertError> {
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        tensor_name,
        dequantize,
        None,
        ConvertTarget::Generic,
    )
}

/// Read and append a single tensor to the blob, with optional dequantization or requantization.
///
/// Defaults to the Generic (CUDA-compatible) target where K-quant layer
/// tensors are passed through unchanged. Call
/// [`append_tensor_to_blob_requant_with_target`] directly when target-aware
/// upcast (e.g. Metal K-quant → Q8_0) is needed.
pub(crate) fn append_tensor_to_blob_requant<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    tensor_name: &str,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
) -> Result<(), ConvertError> {
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        tensor_name,
        dequantize,
        requant_to,
        ConvertTarget::Generic,
    )
}

/// Target-aware version of [`append_tensor_to_blob_requant`].
///
/// When `target == ConvertTarget::Metal` and the source tensor is a K-quant
/// family member (Q2_K..Q6_K, Q8_K) or legacy Q5_0, the tensor is upcast to Q8_0 via the
/// existing dequant -> Q8_0 quantize pipeline. This avoids hitting the
/// Metal backend's missing K-quant kernels at runtime (the Q4 MoE-35B
/// regression that caused layer 3 NaN).
///
/// Norm tensors (anything whose name contains "norm") are exempted from the
/// upcast and are always written F32: the GPU backends read norm weights as
/// F32 (Metal rejects every non-F32 per-layer norm at load; CUDA rejects
/// the attention norms), so a non-F32 norm source is dequantized here
/// regardless of flags.
pub(crate) fn append_tensor_to_blob_requant_with_target<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    tensor_name: &str,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
    target: ConvertTarget,
) -> Result<(), ConvertError> {
    let tensor = gguf
        .find_tensor(tensor_name)
        .ok_or_else(|| ConvertError::MissingTensor(tensor_name.to_string()))?;
    let data = read_tensor_data(reader, gguf, tensor)?;
    let is_norm = tensor_name.contains("norm");
    let dequantize = dequantize || (is_norm && tensor.ggml_type != GgmlType::F32);

    // Metal target: K-quant or legacy-Q5_0 layer tensor (non-norm) -- upcast
    // to Q8_0. The round trip re-quantizes with new block scales; Q8_0's
    // 8-bit mantissa keeps the additional loss below the source quant's own
    // error. Skip when user already set --requant or --dequantize, those
    // flags take precedence.
    let needs_metal_upcast = target == ConvertTarget::Metal
        && !is_norm
        && !dequantize
        && requant_to.is_none()
        && metal_needs_upcast(tensor.ggml_type);
    if needs_metal_upcast {
        let f32_data =
            dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
        let n_elems = tensor.n_elements() as usize;
        let q8_data = quantize_f32_to_q8_0(&f32_data, n_elems);
        eprintln!(
            "    Metal upcast: {} ({:?} -> Q8_0, {} -> {} bytes)",
            tensor_name,
            tensor.ggml_type,
            data.len(),
            q8_data.len()
        );
        blob.extend_from_slice(&q8_data);
        return Ok(());
    }

    if let Some(target) = requant_to {
        if is_norm || dequantize {
            // Norm tensors always stay as F32
            let f32_data =
                dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
            blob.extend_from_slice(&f32_data);
        } else if tensor.ggml_type.to_lbc_quant() == Some(target) {
            // Already in target format
            blob.extend_from_slice(&data);
        } else {
            // Dequant -> F32 -> target
            let f32_data =
                dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
            let n_elems = tensor.n_elements() as usize;
            match target {
                QuantScheme::Q4_0 => {
                    let q_data = quantize_f32_to_q4_0(&f32_data, n_elems);
                    blob.extend_from_slice(&q_data);
                }
                QuantScheme::Q8_0 => {
                    let q_data = quantize_f32_to_q8_0(&f32_data, n_elems);
                    blob.extend_from_slice(&q_data);
                }
                _ => {
                    // Unsupported target: keep as F32
                    blob.extend_from_slice(&f32_data);
                }
            }
        }
    } else if dequantize {
        let f32_data =
            dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
        blob.extend_from_slice(&f32_data);
    } else if tensor.ggml_type == crate::gguf::GgmlType::Q4_1 {
        // SOURCE_FIDELITY: keep Q4_1 verbatim — the min term is part of the
        // source quantization and the CUDA runtime serves Q4_1 natively
        // (matvec_q4_1_q8_1). Must mirror the Q4_1 keep in the qwen35 plan.
        if target != ConvertTarget::Metal && crate::convert::source_fidelity() {
            eprintln!(
                "    Kept Q4_1 verbatim: {tensor_name} ({} bytes)",
                data.len()
            );
            blob.extend_from_slice(&data);
            return Ok(());
        }
        // Q4_1 has no dedicated GPU kernel (neither Metal nor CUDA dispatches
        // it outside the source-fidelity route).
        // Requantize to Q4_0: dequant Q4_1 -> F32 -> quantize Q4_0.
        let f32_data =
            dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
        let n_elems = tensor.n_elements() as usize;
        let q4_data = quantize_f32_to_q4_0(&f32_data, n_elems);
        eprintln!(
            "    Requantized Q4_1 -> Q4_0: {tensor_name} ({} -> {} bytes)",
            data.len(),
            q4_data.len()
        );
        blob.extend_from_slice(&q4_data);
    } else if tensor.ggml_type == crate::gguf::GgmlType::Q8_1 {
        // Q8_1 has no LBC QuantScheme and no dedicated GPU kernel.
        // Requantize to Q8_0: dequant Q8_1 -> F32 -> quantize Q8_0.
        let f32_data =
            dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
        let n_elems = tensor.n_elements() as usize;
        let q8_data = quantize_f32_to_q8_0(&f32_data, n_elems);
        eprintln!(
            "    Requantized Q8_1 -> Q8_0: {tensor_name} ({} -> {} bytes)",
            data.len(),
            q8_data.len()
        );
        blob.extend_from_slice(&q8_data);
    } else if tensor.ggml_type == crate::gguf::GgmlType::Q5_1 {
        // Q5_1 has no LBC QuantScheme and no dedicated GPU kernel.
        // Dequantize to F32.
        let f32_data =
            dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), tensor_name)?;
        eprintln!(
            "    Dequantized Q5_1 -> F32: {tensor_name} ({} -> {} bytes)",
            data.len(),
            f32_data.len()
        );
        blob.extend_from_slice(&f32_data);
    } else {
        blob.extend_from_slice(&data);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bias slice computation
// ---------------------------------------------------------------------------

/// Try to compute a TensorSlice for an optional bias tensor (e.g. attn_q.bias).
/// Returns None if the tensor doesn't exist in the GGUF file.
/// Bias tensors are always stored as F32.
pub(crate) fn try_compute_bias_slice(
    gguf: &GgufFile,
    layer: usize,
    suffix: &str,
    blob_offset: &mut u64,
) -> Option<TensorSlice> {
    let name = layer_tensor_name(layer, suffix);
    let tensor = gguf.find_tensor(&name)?;
    let n_elements = tensor.n_elements();
    let size = n_elements * 4; // F32 bias
    let slice = TensorSlice {
        offset: *blob_offset,
        length: size,
        quant: QuantScheme::F32,
    };
    *blob_offset = blob_offset.saturating_add(size);
    Some(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, GgufBuilder, GgufFile};
    use std::io::Cursor;

    /// A multi-chunk read (> the 8 MiB staging chunk), driven by a reader
    /// that yields one byte per call to exercise the loop maximally,
    /// returns exactly the requested bytes at EXACTLY the final capacity —
    /// no over-allocation (this catches a `read_to_end`-style ~2x growth
    /// regression). The stronger no-reallocation / no-quadratic-copy
    /// property under the production allocator (mimalloc) is pinned
    /// empirically by the retained real-conversion peak-RSS measurement
    /// (~2.6 GB on a 9 GB model), which a unit test cannot observe without
    /// a process-global allocator harness.
    #[test]
    fn bounded_read_multichunk_exact_no_overalloc() {
        // A reader that yields 1 byte per read call — maximises growth
        // opportunities within each staging chunk.
        struct DribbleReader<'a> {
            data: &'a [u8],
            pos: usize,
        }
        impl Read for DribbleReader<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.pos >= self.data.len() || buf.is_empty() {
                    return Ok(0);
                }
                buf[0] = self.data[self.pos];
                self.pos += 1;
                Ok(1)
            }
        }
        let size = (8 << 20) + 12345; // spans two staging chunks, non-aligned
        let src: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let mut r = DribbleReader { data: &src, pos: 0 };
        let out = read_exact_bounded(&mut r, size as u64, "t").unwrap();
        assert_eq!(out.len(), size);
        assert_eq!(out.capacity(), size, "capacity must be exact");
        assert_eq!(out, src);
    }

    /// A short stream errors cleanly as a truncation (never an abort), and
    /// a genuine non-EOF I/O fault is preserved as `Io`, not relabeled
    /// malformed.
    #[test]
    fn bounded_read_short_and_io_errors() {
        // Short stream -> UnexpectedEof -> the truncation message, NOT the
        // allocation-failure branch (distinguished by the message).
        let src = vec![0u8; 100];
        let mut cur = Cursor::new(&src);
        let err = read_exact_bounded(&mut cur, 1 << 20, "t").unwrap_err();
        assert!(
            format!("{err}").contains("available in the file"),
            "expected truncation message, got {err}"
        );

        // A genuine non-EOF I/O fault is preserved as ConvertError::Io.
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "denied",
                ))
            }
        }
        let mut fr = FailingReader;
        let err = read_exact_bounded(&mut fr, 64, "t").unwrap_err();
        assert!(
            matches!(err, ConvertError::Io(_)),
            "expected Io, got {err:?}"
        );
    }

    #[test]
    fn metal_target_upcasts_q5_0_layer_tensor_to_q8_0() {
        // 64-element Q5_0 tensor = 2 blocks x 22 bytes. Metal has no Q5_0
        // kernels (the runtime resident loader rejects the scheme), so the
        // Metal target must upcast it to Q8_0 (2 blocks x 34 bytes), exactly
        // like the K-quants.
        let mut q5 = vec![0u8; 2 * 22];
        for block in q5.chunks_exact_mut(22) {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale 1.0
            for (i, b) in block[6..22].iter_mut().enumerate() {
                *b = (i as u8) | (((15 - i) as u8) << 4);
            }
        }
        let mut builder = GgufBuilder::new();
        builder.add_tensor("blk.0.ffn_gate.weight", GgmlType::Q5_0, &[32, 2], q5);
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let mut reader = Cursor::new(&bytes);

        let mut blob = Vec::new();
        append_tensor_to_blob_requant_with_target(
            &mut blob,
            &mut reader,
            &gguf,
            "blk.0.ffn_gate.weight",
            false,
            None,
            ConvertTarget::Metal,
        )
        .unwrap();
        assert_eq!(
            blob.len(),
            2 * 34,
            "Metal target must emit Q8_0 blocks for a Q5_0 source, got {} bytes",
            blob.len()
        );

        // The Generic target keeps Q5_0 verbatim (CUDA dequantizes at load).
        let mut generic_blob = Vec::new();
        let mut reader2 = Cursor::new(&bytes);
        append_tensor_to_blob_requant_with_target(
            &mut generic_blob,
            &mut reader2,
            &gguf,
            "blk.0.ffn_gate.weight",
            false,
            None,
            ConvertTarget::Generic,
        )
        .unwrap();
        assert_eq!(generic_blob.len(), 2 * 22);
    }
}
