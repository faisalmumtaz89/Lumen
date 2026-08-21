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
    /// compressed-tensors "pack-quantized" INT4, group-32, asymmetric
    /// (imported from HF safetensors checkpoints, e.g. AWQ/GPTQ-class
    /// releases compressed with this format).
    ///
    /// A tensor slice with this scheme holds the three source planes
    /// byte-for-byte, concatenated in fixed order with sizes derived from
    /// the logical shape `[n, k]` (`k % 32 == 0`, zero-points 4-bit packed
    /// along n):
    ///
    /// 1. `weight_packed`  — `n * k / 2` bytes (i32 words, 8 nibbles each,
    ///    little-nibble-first along k; values unsigned 0..=15)
    /// 2. `weight_scale`   — `n * (k / 32) * 2` bytes (BF16, one per group)
    /// 3. `weight_zero_point` — `ceil(n / 8) * (k / 32) * 4` bytes (i32
    ///    words packing 8 rows' 4-bit zero-points, little-nibble-first
    ///    along n; unsigned 0..=15)
    ///
    /// Dequantization: `w = (q - zp) * scale` per 32-element group along k.
    CtInt4G32,
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
            // 4 (packed) + 16/32 (BF16 scale) + 4/32 (packed zero-point).
            Self::CtInt4G32 => 4.625,
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
            Self::CtInt4G32 => 12,
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
            12 => Ok(Self::CtInt4G32),
            _ => Err(crate::FormatError::UnsupportedQuantization(format!(
                "unknown quant scheme tag: {tag}"
            ))),
        }
    }
}

/// Byte sizes of the three planes of a [`QuantScheme::CtInt4G32`] tensor
/// slice for logical shape `[n, k]`, in their fixed slice order. The single
/// source of truth for the plane derivation — the converter sizes writes
/// with it and the runtime locates planes with it.
///
/// Requires `k % 32 == 0` (the group size); `n` may be any positive value
/// (the zero-point plane rounds `n` up to a whole number of packed words).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtInt4G32Planes {
    pub qweight_bytes: u64,
    pub scale_bytes: u64,
    pub zero_point_bytes: u64,
}

impl CtInt4G32Planes {
    pub fn for_shape(n: u64, k: u64) -> Result<Self, crate::FormatError> {
        if n == 0 || k == 0 || k % 32 != 0 {
            return Err(crate::FormatError::UnsupportedQuantization(format!(
                "CtInt4G32 requires n > 0 and k % 32 == 0, got [{n}, {k}]"
            )));
        }
        let groups = k / 32;
        let overflow = || {
            crate::FormatError::UnsupportedQuantization(format!(
                "CtInt4G32 shape [{n}, {k}] overflows plane arithmetic"
            ))
        };
        Ok(Self {
            qweight_bytes: n.checked_mul(k).map(|v| v / 2).ok_or_else(overflow)?,
            scale_bytes: n
                .checked_mul(groups)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(overflow)?,
            zero_point_bytes: n
                .div_ceil(8)
                .checked_mul(groups)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(overflow)?,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        // Cannot overflow: each plane is at most n*k/2 bytes and for_shape
        // already rejected shapes whose products exceed u64.
        self.qweight_bytes + self.scale_bytes + self.zero_point_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SCHEMES: [QuantScheme; 13] = [
        QuantScheme::F32,
        QuantScheme::F16,
        QuantScheme::Bf16,
        QuantScheme::Q8_0,
        QuantScheme::Q4_0,
        QuantScheme::Q4_1,
        QuantScheme::Q4_K,
        QuantScheme::Q5_0,
        QuantScheme::Q5_K,
        QuantScheme::Q6_K,
        QuantScheme::Q2_K,
        QuantScheme::Q3_K,
        QuantScheme::CtInt4G32,
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
        assert!(QuantScheme::from_u8(13).is_err());
        assert!(QuantScheme::from_u8(255).is_err());
    }

    #[test]
    fn ct_int4_g32_planes_match_source_checkpoint_shapes() {
        // Plane sizes must equal the safetensors plane byte sizes of real
        // pack-quantized checkpoints (Qwen3.8-27B: down_proj [5120, 17408]
        // stores qweight i32[5120, 2176], scale bf16[5120, 544],
        // zero_point i32[640, 544]; gate_proj is the transpose case).
        let p = CtInt4G32Planes::for_shape(5120, 17408).unwrap();
        assert_eq!(p.qweight_bytes, 44_564_480);
        assert_eq!(p.scale_bytes, 5_570_560);
        assert_eq!(p.zero_point_bytes, 1_392_640);
        assert_eq!(p.total_bytes(), 44_564_480 + 5_570_560 + 1_392_640);

        let p = CtInt4G32Planes::for_shape(17408, 5120).unwrap();
        assert_eq!(p.qweight_bytes, 44_564_480);
        assert_eq!(p.scale_bytes, 5_570_560);
        assert_eq!(p.zero_point_bytes, 1_392_640);
    }

    #[test]
    fn ct_int4_g32_planes_round_up_zero_point_rows() {
        // n not divisible by 8: the zero-point plane packs 8 rows per i32
        // word, so 9 rows need 2 words per group.
        let p = CtInt4G32Planes::for_shape(9, 64).unwrap();
        assert_eq!(p.qweight_bytes, 9 * 64 / 2);
        assert_eq!(p.scale_bytes, 9 * 2 * 2);
        assert_eq!(p.zero_point_bytes, 2 * 2 * 4);
    }

    #[test]
    fn ct_int4_g32_planes_reject_bad_shapes() {
        assert!(CtInt4G32Planes::for_shape(0, 64).is_err());
        assert!(CtInt4G32Planes::for_shape(16, 0).is_err());
        assert!(CtInt4G32Planes::for_shape(16, 48).is_err());
    }

    #[test]
    fn bits_per_weight_correctness() {
        let expected: [(QuantScheme, f32); 13] = [
            (QuantScheme::F32, 32.0),
            (QuantScheme::F16, 16.0),
            (QuantScheme::Bf16, 16.0),
            (QuantScheme::Q8_0, 8.0),
            (QuantScheme::Q4_0, 4.0),
            (QuantScheme::Q4_1, 4.0),
            (QuantScheme::Q4_K, 4.0),
            (QuantScheme::Q5_0, 5.0),
            (QuantScheme::Q5_K, 5.0),
            (QuantScheme::Q6_K, 6.0),
            (QuantScheme::Q2_K, 2.0),
            (QuantScheme::Q3_K, 3.0),
            // Effective density including scale + zero-point metadata
            // (4 payload bits + 0.5 scale + 0.125 zero-point per weight).
            (QuantScheme::CtInt4G32, 4.625),
        ];
        for (scheme, bits) in expected {
            assert_eq!(
                scheme.bits_per_weight(),
                bits,
                "wrong bits for {:?}",
                scheme
            );
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
        assert!(QuantScheme::CtInt4G32.is_quantized());
    }
}
