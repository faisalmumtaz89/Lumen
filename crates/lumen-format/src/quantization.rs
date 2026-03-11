//! Quantization descriptors for weight storage.
//!
//! Weights in LBC are stored pre-quantized. The descriptor tells the compute
//! backend how to interpret the raw bytes.

/// Quantization scheme identifier.
///
/// The runtime uses this to dispatch the correct dequantization kernel.
// GGML-convention names like Q4_K are standard in the LLM quantization
// ecosystem. We preserve them for clarity and interoperability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantScheme {
    /// 32-bit IEEE 754 floats (unquantized).
    F32,
    /// 16-bit IEEE 754 half-precision (unquantized).
    F16,
    /// Brain floating point (16-bit, 8-bit exponent).
    Bf16,
    /// 8-bit with per-group scales and zero points.
    Q8_0,
    /// 4-bit with per-group scales and zero points.
    Q4_0,
    /// 4-bit with per-group scales and min (GGML Q4_1).
    Q4_1,
    /// 4-bit with 6-bit super-block scales (GGML Q4_K).
    Q4_K,
    /// 5-bit with per-group scales.
    Q5_0,
    /// 5-bit with per-group scales (GGML Q5_K).
    Q5_K,
    /// 6-bit with per-group scales (GGML Q6_K).
    Q6_K,
    /// 2-bit with per-group scales.
    Q2_K,
    /// 3-bit with per-group scales.
    Q3_K,
}

/// Number of elements sharing a scale/zero-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGroupSize {
    /// Single scale for the entire tensor.
    PerTensor,
    /// Per-channel/per-row.
    PerChannel,
    /// Block quantization with given group size.
    Group(u32),
}

/// Full quantization descriptor for a tensor or layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationDescriptor {
    pub scheme: QuantScheme,
    /// Group size for block-quantized formats.
    pub group_size: QuantGroupSize,
    /// Bytes per quantized block (data + scales + zeros).
    /// E.g., Q4_0 with group_size=32: 18 bytes (16 data + 2 scale).
    pub block_byte_size: u32,
    /// Byte offset of scale metadata within each block. `None` if scales
    /// are stored separately or the format has no per-block scales.
    pub scale_offset_in_block: Option<u32>,
}

impl QuantScheme {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::Bf16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q4_0 | Self::Q4_1 | Self::Q4_K => 4.0,
            Self::Q5_0 | Self::Q5_K => 5.0,
            Self::Q6_K => 6.0,
            Self::Q2_K => 2.0,
            Self::Q3_K => 3.0,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }

    /// Serialize to a single-byte tag for the LBC binary format.
    pub fn to_u8(&self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
            Self::Q8_0 => 3,
            Self::Q4_0 => 4,
            Self::Q4_1 => 5,
            Self::Q4_K => 6,
            Self::Q5_0 => 7,
            Self::Q5_K => 8,
            Self::Q6_K => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
        }
    }

    /// Deserialize from a single-byte tag.
    pub fn from_u8(tag: u8) -> Result<Self, crate::FormatError> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Q8_0),
            4 => Ok(Self::Q4_0),
            5 => Ok(Self::Q4_1),
            6 => Ok(Self::Q4_K),
            7 => Ok(Self::Q5_0),
            8 => Ok(Self::Q5_K),
            9 => Ok(Self::Q6_K),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            _ => Err(crate::FormatError::UnsupportedQuantization(
                format!("unknown quant scheme tag: {tag}"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SCHEMES: [QuantScheme; 12] = [
        QuantScheme::F32, QuantScheme::F16, QuantScheme::Bf16,
        QuantScheme::Q8_0, QuantScheme::Q4_0, QuantScheme::Q4_1,
        QuantScheme::Q4_K, QuantScheme::Q5_0, QuantScheme::Q5_K,
        QuantScheme::Q6_K, QuantScheme::Q2_K, QuantScheme::Q3_K,
    ];

    #[test]
    fn roundtrip_all_quant_schemes() {
        for scheme in ALL_SCHEMES {
            let tag = scheme.to_u8();
            let recovered = QuantScheme::from_u8(tag).unwrap();
            assert_eq!(scheme, recovered, "roundtrip failed for tag {tag}");
        }
    }

    #[test]
    fn invalid_tags_return_error() {
        assert!(QuantScheme::from_u8(12).is_err());
        assert!(QuantScheme::from_u8(255).is_err());
    }

    #[test]
    fn bits_per_weight_correctness() {
        let expected: [(QuantScheme, f32); 12] = [
            (QuantScheme::F32, 32.0), (QuantScheme::F16, 16.0), (QuantScheme::Bf16, 16.0),
            (QuantScheme::Q8_0, 8.0), (QuantScheme::Q4_0, 4.0), (QuantScheme::Q4_1, 4.0),
            (QuantScheme::Q4_K, 4.0), (QuantScheme::Q5_0, 5.0), (QuantScheme::Q5_K, 5.0),
            (QuantScheme::Q6_K, 6.0), (QuantScheme::Q2_K, 2.0), (QuantScheme::Q3_K, 3.0),
        ];
        for (scheme, bits) in expected {
            assert_eq!(scheme.bits_per_weight(), bits, "wrong bits for {:?}", scheme);
        }
    }

    #[test]
    fn is_quantized_classification() {
        // Unquantized: F32, F16, Bf16
        assert!(!QuantScheme::F32.is_quantized());
        assert!(!QuantScheme::F16.is_quantized());
        assert!(!QuantScheme::Bf16.is_quantized());
        // Quantized: all Q* variants
        assert!(QuantScheme::Q8_0.is_quantized());
        assert!(QuantScheme::Q4_0.is_quantized());
        assert!(QuantScheme::Q4_1.is_quantized());
        assert!(QuantScheme::Q4_K.is_quantized());
        assert!(QuantScheme::Q5_0.is_quantized());
        assert!(QuantScheme::Q5_K.is_quantized());
        assert!(QuantScheme::Q6_K.is_quantized());
        assert!(QuantScheme::Q2_K.is_quantized());
        assert!(QuantScheme::Q3_K.is_quantized());
    }
}

