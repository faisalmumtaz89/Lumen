//! Shared GDN (GatedDeltaNet) gate conversion logic for Qwen3.5 architectures.
//!
//! GGUF stores ssm_alpha/ssm_beta as F32, but the GDN runtime hardcodes Q8_0
//! matvec kernels for these tensors. This module centralises the force-requant
//! logic so both the dense and MoE converters handle it identically.

use crate::convert::ConvertError;
use crate::gguf::GgufFile;
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
) -> Result<Option<TensorSlice>, ConvertError> {
    let name = layer_tensor_name(layer, suffix);
    let tensor = match gguf.find_tensor(&name) {
        Some(t) => t,
        None => return Ok(None),
    };

    if dequantize {
        let n_elements = tensor.n_elements();
        let size = n_elements * 4;
        let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::F32 };
        *blob_offset += size;
        return Ok(Some(slice));
    }

    let quant = tensor.ggml_type.to_lbc_quant()
        .ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: name.to_string(),
            ggml_type: format!("{:?}", tensor.ggml_type),
        })?;

    // Force ssm_alpha/beta to Q8_0 if not already -- runtime hardcodes Q8_0 matvec.
    let is_alpha_or_beta = suffix == SSM_ALPHA || suffix == SSM_BETA;
    if is_alpha_or_beta && !matches!(quant, QuantScheme::Q8_0) {
        let n_elements = tensor.n_elements() as usize;
        assert!(n_elements % 32 == 0,
            "Q8_0 requires elements divisible by 32, got {n_elements} for {name}");
        let num_blocks = n_elements / 32;
        let size = (num_blocks * 34) as u64; // Q8_0: 34 bytes per 32 elements
        let slice = TensorSlice { offset: *blob_offset, length: size, quant: QuantScheme::Q8_0 };
        *blob_offset += size;
        Ok(Some(slice))
    } else {
        let size = tensor.byte_size().ok_or_else(|| ConvertError::UnsupportedTensorType {
            tensor: name.to_string(),
            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
        })?;
        let slice = TensorSlice { offset: *blob_offset, length: size, quant };
        *blob_offset += size;
        Ok(Some(slice))
    }
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
) -> Result<SsmSlices, ConvertError> {
    Ok(SsmSlices {
        ssm_a:      compute_ssm_tensor_slice(gguf, layer, SSM_A,      blob_offset, dequantize)?,
        ssm_conv1d: compute_ssm_tensor_slice(gguf, layer, SSM_CONV1D, blob_offset, dequantize)?,
        ssm_dt:     compute_ssm_tensor_slice(gguf, layer, SSM_DT,     blob_offset, dequantize)?,
        ssm_beta:   compute_ssm_tensor_slice(gguf, layer, SSM_BETA,   blob_offset, dequantize)?,
        ssm_alpha:  compute_ssm_tensor_slice(gguf, layer, SSM_ALPHA,  blob_offset, dequantize)?,
        ssm_norm:   compute_ssm_tensor_slice(gguf, layer, SSM_NORM,   blob_offset, dequantize)?,
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
) -> Result<(), ConvertError> {
    for suffix in &SSM_SUFFIXES {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let is_alpha_or_beta = *suffix == SSM_ALPHA || *suffix == SSM_BETA;
            let src_quant = tensor.ggml_type.to_lbc_quant();
            if is_alpha_or_beta && !matches!(src_quant, Some(QuantScheme::Q8_0)) {
                // Force-requantize to Q8_0 (dequant to F32 first, then quantize to Q8_0)
                append_tensor_to_blob_requant(blob, reader, gguf, &name, false, Some(QuantScheme::Q8_0))?;
            } else {
                append_tensor_to_blob_requant(blob, reader, gguf, &name, dequantize, /*requant_to=*/ None)?;
            }
        }
    }
    Ok(())
}
