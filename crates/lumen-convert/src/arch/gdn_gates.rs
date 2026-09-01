//! Shared GDN (GatedDeltaNet) gate conversion logic for Qwen3.5 architectures.
//!
//! GGUF stores ssm_alpha/ssm_beta as F32. Metal's GDN gate kernels read them
//! as Q8_0 only, so default conversions force-requantize them; CUDA also
//! serves F32 gates (source-fidelity / `--dequantize` non-Metal artifacts).
//! This module centralises that logic so both the dense and MoE converters
//! handle it identically.

use crate::convert::{ConvertError, ConvertTarget};
use crate::gguf::{GgmlType, GgufFile};
use crate::tensor_io::*;
use crate::tensor_names::*;
use lumen_format::index::TensorSlice;
use lumen_format::quantization::QuantScheme;
use std::io::{Read, Seek};

/// SSM tensor suffixes that are never requantized to a user-specified target.
/// ssm_a/dt/conv1d are small F32 scalars read as `float*` in GPU kernels.
/// ssm_alpha/beta are Q8_0 gate matrices (force-requantized below).
/// ssm_norm is a norm tensor (F32).
const SSM_SUFFIXES: [&str; 6] = [SSM_A, SSM_CONV1D, SSM_DT, SSM_BETA, SSM_ALPHA, SSM_NORM];

/// How one SSM tensor is materialized in the layer blob.
enum SsmForm {
    /// Source bytes verbatim (size = source byte size, quant = source scheme).
    KeepSource,
    /// Dequantized to F32 (size = 4n, quant = F32).
    F32,
    /// Requantized via F32 to Q8_0 (size = 34n/32, quant = Q8_0).
    Q8,
}

/// Single decision point for how an SSM tensor is written: the planner sizes
/// from this and the writer produces bytes from it, so the layer-blob layout
/// cannot desync between them.
///
/// ssm_alpha/ssm_beta are Q8_0 wherever Metal must load the file (its GDN
/// gate pipelines read them as Q8_0 only); non-Metal targets serve F32 gates,
/// so `--dequantize` and SOURCE_FIDELITY may keep them F32 there. The SSM
/// scalars (a, conv1d, dt, norm) are always F32 — the runtime slots are typed
/// `f32`.
fn ssm_tensor_form(
    name: &str,
    suffix: &str,
    t: GgmlType,
    dequantize: bool,
    target: ConvertTarget,
) -> Result<SsmForm, ConvertError> {
    debug_assert!(
        SSM_SUFFIXES.contains(&suffix),
        "ssm_tensor_form governs only the six core SSM tensors, got {suffix}"
    );
    let is_gate = suffix == SSM_ALPHA || suffix == SSM_BETA;
    if is_gate {
        if dequantize && target != ConvertTarget::Metal {
            return Ok(SsmForm::F32);
        }
        if target != ConvertTarget::Metal && crate::convert::source_fidelity() && t == GgmlType::F32
        {
            return Ok(SsmForm::KeepSource);
        }
        if t == GgmlType::Q8_0 {
            return Ok(SsmForm::KeepSource);
        }
        if t == GgmlType::F32 || t.has_dequant_path() {
            return Ok(SsmForm::Q8);
        }
        return Err(ConvertError::UnsupportedTensorType {
            tensor: name.to_string(),
            ggml_type: format!("{t:?}"),
        });
    }
    if dequantize {
        return Ok(SsmForm::F32);
    }
    if t == GgmlType::F32 {
        return Ok(SsmForm::KeepSource);
    }
    if t.has_dequant_path() {
        return Ok(SsmForm::F32);
    }
    Err(ConvertError::UnsupportedTensorType {
        tensor: name.to_string(),
        ggml_type: format!("{t:?} (cannot force-dequant SSM scalar to F32)"),
    })
}

/// Compute a [`TensorSlice`] for a single SSM tensor, applying force-requant
/// to Q8_0 for ssm_alpha/ssm_beta when needed.
///
/// Returns `None` if the tensor is absent from the GGUF file.
pub(crate) fn compute_ssm_tensor_slice(
    gguf: &GgufFile,
    layer: usize,
    suffix: &str,
    blob_offset: &mut u64,
    dequantize: bool,
    target: ConvertTarget,
) -> Result<Option<TensorSlice>, ConvertError> {
    let name = layer_tensor_name(layer, suffix);
    let tensor = match gguf.find_tensor(&name) {
        Some(t) => t,
        None => return Ok(None),
    };
    let (length, quant) =
        match ssm_tensor_form(&name, suffix, tensor.ggml_type, dequantize, target)? {
            SsmForm::KeepSource => {
                let size =
                    tensor
                        .byte_size()
                        .ok_or_else(|| ConvertError::UnsupportedTensorType {
                            tensor: name.clone(),
                            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                        })?;
                let quant = tensor.ggml_type.to_lbc_quant().ok_or_else(|| {
                    ConvertError::UnsupportedTensorType {
                        tensor: name.clone(),
                        ggml_type: format!("{:?}", tensor.ggml_type),
                    }
                })?;
                (size, quant)
            }
            SsmForm::F32 => (tensor.n_elements() * 4, QuantScheme::F32),
            SsmForm::Q8 => {
                let n_elements = tensor.n_elements() as usize;
                assert!(
                    n_elements % 32 == 0,
                    "Q8_0 requires elements divisible by 32, got {n_elements} for {name}"
                );
                (((n_elements / 32) * 34) as u64, QuantScheme::Q8_0)
            }
        };
    let slice = TensorSlice {
        offset: *blob_offset,
        length,
        quant,
    };
    *blob_offset = blob_offset.saturating_add(length);
    Ok(Some(slice))
}

/// All SSM tensor slices needed by a GDN layer (shape computation).
pub(crate) struct SsmSlices {
    pub ssm_a: Option<TensorSlice>,
    pub ssm_conv1d: Option<TensorSlice>,
    pub ssm_dt: Option<TensorSlice>,
    pub ssm_beta: Option<TensorSlice>,
    pub ssm_alpha: Option<TensorSlice>,
    pub ssm_norm: Option<TensorSlice>,
}

/// Compute [`TensorSlice`]s for all six core SSM tensors (a, conv1d, dt, beta, alpha, norm).
///
/// ssm_alpha/ssm_beta are force-sized to Q8_0 when the source is not Q8_0.
/// ssm_out is NOT included -- its handling differs between dense (requant-aware)
/// and MoE (always-F32) converters.
pub(crate) fn compute_ssm_slices(
    gguf: &GgufFile,
    layer: usize,
    blob_offset: &mut u64,
    dequantize: bool,
    target: ConvertTarget,
) -> Result<SsmSlices, ConvertError> {
    Ok(SsmSlices {
        ssm_a: compute_ssm_tensor_slice(gguf, layer, SSM_A, blob_offset, dequantize, target)?,
        ssm_conv1d: compute_ssm_tensor_slice(
            gguf,
            layer,
            SSM_CONV1D,
            blob_offset,
            dequantize,
            target,
        )?,
        ssm_dt: compute_ssm_tensor_slice(gguf, layer, SSM_DT, blob_offset, dequantize, target)?,
        ssm_beta: compute_ssm_tensor_slice(gguf, layer, SSM_BETA, blob_offset, dequantize, target)?,
        ssm_alpha: compute_ssm_tensor_slice(
            gguf,
            layer,
            SSM_ALPHA,
            blob_offset,
            dequantize,
            target,
        )?,
        ssm_norm: compute_ssm_tensor_slice(gguf, layer, SSM_NORM, blob_offset, dequantize, target)?,
    })
}

/// Write the six core SSM tensors into a blob, force-requantizing ssm_alpha/beta
/// to Q8_0 when the source is F32/F16/BF16.
///
/// ssm_out is NOT included -- callers handle it separately (dense uses requant_to,
/// MoE always forces F32).
pub(crate) fn write_ssm_tensors<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    target: ConvertTarget,
) -> Result<(), ConvertError> {
    for suffix in &SSM_SUFFIXES {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            match ssm_tensor_form(&name, suffix, tensor.ggml_type, dequantize, target)? {
                SsmForm::KeepSource => {
                    append_tensor_to_blob_requant(blob, reader, gguf, &name, false, None)?;
                }
                SsmForm::F32 => {
                    append_tensor_to_blob_requant(blob, reader, gguf, &name, true, None)?;
                }
                SsmForm::Q8 => {
                    append_tensor_to_blob_requant(
                        blob,
                        reader,
                        gguf,
                        &name,
                        false,
                        Some(QuantScheme::Q8_0),
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Whether a GDN layer's `attn_qkv`/`attn_gate` pair must both be written
/// Q8_0 on the Metal target.
///
/// Metal's GDN decode path requires the pair to agree on whether it is Q8_0
/// (the loader rejects a split). The per-tensor K-quant upcast can
/// manufacture that split from a uniform source, and mixed-source GGUFs can
/// carry one natively — so under default flags, when the pair's post-upcast
/// schemes would split on the Q8_0 axis, both tensors are force-written
/// Q8_0. Single decision point for the dense and MoE planners AND writers.
/// Plan the Q8_0 slice for a tensor the GDN pair-force landed on Q8_0.
/// Q8_0 requires the element count to be a multiple of 32; a source whose
/// count is not (only reachable from a hand-crafted GGUF — real GDN dims
/// are always 32-aligned) is rejected here with a clear error instead of
/// panicking in the quantizer. Widening the pair-force to F16 and
/// F32-split sources broadened the reachability of this path, so the guard
/// lives at the plan step
/// (before any byte is written).
pub(crate) fn pair_forced_q8_slice(
    n_elements: usize,
    name: &str,
    blob_offset: &mut u64,
) -> Result<TensorSlice, ConvertError> {
    if n_elements % 32 != 0 {
        return Err(ConvertError::UnsupportedModel(format!(
            "{name}: the Metal GDN pair force lands this tensor on Q8_0, which \
             requires an element count divisible by 32, but it has {n_elements}. \
             Re-convert from a GGUF whose GDN projections are 32-aligned."
        )));
    }
    let size = ((n_elements / 32) * 34) as u64;
    let slice = TensorSlice {
        offset: *blob_offset,
        length: size,
        quant: QuantScheme::Q8_0,
    };
    *blob_offset = blob_offset.saturating_add(size);
    Ok(slice)
}

pub(crate) fn metal_gdn_pair_forces_q8(
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
    target: ConvertTarget,
) -> bool {
    if target != ConvertTarget::Metal || dequantize || requant_to.is_some() {
        return false;
    }
    let qkv = gguf.find_tensor(&layer_tensor_name(layer, ATTN_QKV));
    let gate = gguf.find_tensor(&layer_tensor_name(layer, ATTN_GATE_WEIGHT));
    let (Some(qkv), gate) = (qkv, gate) else {
        return false;
    };
    let Some(gate) = gate else {
        // Gate absent (malformed source — GDN serving requires it, and the
        // runtime errors cleanly at dispatch). Still force an F16 qkv to
        // Q8_0 so the LOAD does not reject what the dispatch would
        // diagnose more precisely.
        return qkv.ggml_type == GgmlType::F16;
    };
    // Q8_1 is unconditionally requantized to Q8_0 by both converters, so it
    // lands on the Q8 side of the pair.
    let is_q8_after_target =
        |t: GgmlType| t == GgmlType::Q8_0 || t == GgmlType::Q8_1 || metal_needs_upcast(t);
    // F16 forces the pair too: the Metal GDN prefill projections have no
    // F16 arm (the loader rejects F16 there), so an F16 source must land
    // Q8_0 — otherwise this converter emits an artifact its own loader
    // refuses. Bf16 stays: the prefill has Bf16 arms.
    let is_f16 = |t: GgmlType| t == GgmlType::F16;
    // An F32/non-F32 SPLIT forces the pair as well: the Metal decode
    // F32-gate fallback reads a norm buffer only the F32 QKV route writes,
    // so the loader rejects gate-F32 next to a non-F32 qkv (and vice
    // versa the qkv-F32 route pairs with a gate arm that exists, but
    // uniform Q8 is the safe landing for any split). F32/F32 stays: it is
    // loadable and `--dequantize` produces exactly that.
    let is_f32 = |t: GgmlType| t == GgmlType::F32;
    is_q8_after_target(qkv.ggml_type) != is_q8_after_target(gate.ggml_type)
        || is_f16(qkv.ggml_type)
        || is_f16(gate.ggml_type)
        || (is_f32(qkv.ggml_type) != is_f32(gate.ggml_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pair-force Q8_0 planner rejects a non-32-aligned element count
    /// (only reachable from a hand-crafted GGUF; the F16/F32-split
    /// widening broadened its reachability) with a clean error instead of
    /// the quantizer's panic, and plans the exact Q8_0 size otherwise.
    #[test]
    fn pair_forced_q8_slice_rejects_unaligned() {
        let mut off = 100u64;
        // 6144 = 192 blocks -> 192*34 bytes, offset advances.
        let ok = pair_forced_q8_slice(6144, "blk.0.attn_qkv.weight", &mut off).unwrap();
        assert_eq!(ok.quant, QuantScheme::Q8_0);
        assert_eq!(ok.length, 192 * 34);
        assert_eq!(ok.offset, 100);
        assert_eq!(off, 100 + 192 * 34);
        // Non-32-aligned -> clean error, offset untouched.
        let mut off2 = 0u64;
        let err = pair_forced_q8_slice(33, "blk.0.attn_qkv.weight", &mut off2).unwrap_err();
        assert!(
            matches!(err, ConvertError::UnsupportedModel(_))
                && format!("{err}").contains("divisible by 32"),
            "{err}"
        );
        assert_eq!(off2, 0, "offset must not advance on rejection");
    }
}
