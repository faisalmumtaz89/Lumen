//! Architecture-specific converters and dispatch.

use crate::convert::ConvertError;
use crate::gguf::GgufFile;
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

pub(crate) mod dense;
pub(crate) mod gdn_gates;
pub(crate) mod moe;
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
    ) -> Result<(), ConvertError>;

    /// Return a display name for progress logging.
    fn layer_kind_label(&self, layer: usize) -> String;
}

/// Enum dispatch wrapper that delegates to the concrete ArchConverter implementations.
/// Using an enum instead of `Box<dyn ArchConverter>` because the trait has a generic
/// method (`write_layer_blob<R>`) which makes it not dyn-compatible.
pub(crate) enum Converter {
    Dense(dense::DenseConverter),
    Moe(moe::MoeConverter),
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
    ) -> Result<LayerShape, ConvertError> {
        match self {
            Converter::Dense(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to),
            Converter::Moe(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to),
            Converter::Qwen35(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to),
            Converter::Qwen35Moe(c) => c.compute_layer_shape(gguf, layer, dequantize, requant_to),
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
    ) -> Result<(), ConvertError> {
        match self {
            Converter::Dense(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to),
            Converter::Moe(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to),
            Converter::Qwen35(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to),
            Converter::Qwen35Moe(c) => c.write_layer_blob(blob, reader, gguf, layer, dequantize, requant_to),
        }
    }

    pub(crate) fn layer_kind_label(&self, layer: usize) -> String {
        match self {
            Converter::Dense(c) => c.layer_kind_label(layer),
            Converter::Moe(c) => c.layer_kind_label(layer),
            Converter::Qwen35(c) => c.layer_kind_label(layer),
            Converter::Qwen35Moe(c) => c.layer_kind_label(layer),
        }
    }
}

/// Select the appropriate Converter based on architecture string and model properties.
pub(crate) fn select_converter(
    arch: &str,
    num_experts: Option<u32>,
) -> Converter {
    let is_qwen35 = arch == "qwen35";
    let is_qwen35moe = matches!(arch, "qwen35moe" | "qwen3_5_moe" | "qwen3.5_moe");
    let is_moe = !is_qwen35moe && num_experts.map_or(false, |n| n > 0);

    if is_qwen35 {
        Converter::Qwen35(qwen35::Qwen35Converter)
    } else if is_qwen35moe {
        Converter::Qwen35Moe(qwen35_moe::Qwen35MoeConverter {
            num_experts: num_experts.unwrap_or(0),
        })
    } else if is_moe {
        Converter::Moe(moe::MoeConverter {
            num_experts: num_experts.unwrap_or(0),
        })
    } else {
        Converter::Dense(dense::DenseConverter)
    }
}
