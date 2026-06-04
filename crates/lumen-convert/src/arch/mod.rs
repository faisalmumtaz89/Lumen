//! Architecture-specific converters and dispatch.
//!
//! Lumen targets the Qwen3.5 family exclusively:
//!   - `qwen35`     -- dense Qwen3.5 with Gated Delta-Net layers
//!   - `qwen35moe`  -- Qwen3.5 MoE variant (preserved for future GPU support)
//!
//! GGUF architecture strings outside this set are rejected at
//! `hyperparams::extract_hyperparams`.

use crate::convert::{ConvertError, ConvertTarget};
use crate::gguf::GgufFile;
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

pub(crate) mod gdn_gates;
pub(crate) mod qwen35;
pub(crate) mod qwen35_moe;

/// Each supported model architecture implements this trait.
pub(crate) trait ArchConverter {
    /// Compute the LayerShape for a single layer.
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<LayerShape, ConvertError>;

    /// Write a single layer's tensor data into the blob buffer.
    fn write_layer_blob<R: Read + Seek>(
        &self,
        blob: &mut Vec<u8>,
        reader: &mut R,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<(), ConvertError>;

    /// Return a display name for progress logging.
    fn layer_kind_label(&self, layer: usize) -> String;
}

/// Enum dispatch wrapper that delegates to the concrete ArchConverter implementations.
/// Using an enum instead of `Box<dyn ArchConverter>` because the trait has a generic
/// method (`write_layer_blob<R>`) which makes it not dyn-compatible.
pub(crate) enum Converter {
    Qwen35(qwen35::Qwen35Converter),
    Qwen35Moe(qwen35_moe::Qwen35MoeConverter),
}

impl Converter {
    pub(crate) fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<LayerShape, ConvertError> {
        match self {
            Converter::Qwen35(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to, target),
            Converter::Qwen35Moe(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to, target),
        }
    }

    pub(crate) fn write_layer_blob<R: Read + Seek>(
        &self,
        blob: &mut Vec<u8>,
        reader: &mut R,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<(), ConvertError> {
        match self {
            Converter::Qwen35(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to, target),
            Converter::Qwen35Moe(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to, target),
        }
    }

    pub(crate) fn layer_kind_label(&self, layer: usize) -> String {
        match self {
            Converter::Qwen35(c) => c.layer_kind_label(layer),
            Converter::Qwen35Moe(c) => c.layer_kind_label(layer),
        }
    }
}

/// Select the appropriate Converter based on architecture string.
///
/// `num_experts` is forwarded to the MoE variant when the architecture is
/// `qwen35moe` (also accepts the alternate GGUF spellings
/// `qwen3_5_moe` and `qwen3.5_moe`). The dense `qwen35` path ignores it.
pub(crate) fn select_converter(
    arch: &str,
    num_experts: Option<u32>,
) -> Converter {
    if matches!(arch, "qwen35moe" | "qwen3_5_moe" | "qwen3.5_moe") {
        Converter::Qwen35Moe(qwen35_moe::Qwen35MoeConverter {
            num_experts: num_experts.unwrap_or(0),
        })
    } else {
        // `extract_hyperparams` has already rejected anything outside the
        // Qwen3.5 family, so the only remaining valid value here is "qwen35".
        Converter::Qwen35(qwen35::Qwen35Converter)
    }
}
