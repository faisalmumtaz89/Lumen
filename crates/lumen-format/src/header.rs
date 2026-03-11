//! LBC file header.

use crate::hyperparams::ModelHyperparams;
use crate::quantization::{QuantScheme, QuantizationDescriptor};

/// Magic bytes identifying an LBC file: "LBC\x01" in little-endian u32.
pub const LBC_MAGIC: u32 = 0x01_43_42_4C; // 'L' 'B' 'C' 0x01

pub const LBC_VERSION: u32 = 2;

/// Default alignment for layer blobs (128 KiB).
pub const DEFAULT_ALIGNMENT: u64 = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Endianness {
    #[default]
    Little,
    Big,
}

/// A (file-level offset, length, quant) triple for a global tensor blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalTensorRange {
    pub offset: u64,
    pub length: u64,
    /// Quantization scheme of this tensor (v2+; defaults to F32).
    pub quant: QuantScheme,
}

impl Default for GlobalTensorRange {
    fn default() -> Self {
        Self { offset: 0, length: 0, quant: QuantScheme::F32 }
    }
}

/// The complete LBC file header.
#[derive(Debug, Clone)]
pub struct LbcHeader {
    pub magic: u32,
    pub version: u32,
    /// Byte order of the payload data.
    pub endianness: Endianness,
    /// CRC32 checksum of header bytes (excluding this field itself).
    pub header_checksum: u32,
    pub hyperparams: ModelHyperparams,
    /// Primary quantization descriptor; individual tensors may override
    /// via [`SubtensorOffsets`](crate::index::SubtensorOffsets).
    pub quantization: QuantizationDescriptor,
    /// All layer blob offsets must be multiples of this value.
    pub alignment: u64,
    pub num_layers: u32,
    /// `true` for MoE models.
    pub has_expert_index: bool,
    pub layer_index_offset: u64,
    /// 0 if absent.
    pub expert_index_offset: u64,
    pub payload_offset: u64,
    /// Token embedding table (file-level range).
    pub embedding: GlobalTensorRange,
    /// Final RMSNorm weights (file-level range).
    pub final_norm: GlobalTensorRange,
    /// Output projection / unembedding weights (file-level range).
    pub output_proj: GlobalTensorRange,
    /// Whether output_proj shares embedding storage (weight tying).
    pub weight_tying: bool,
}

impl LbcHeader {
    /// Validates header fields for internal consistency.
    pub fn validate(&self) -> Result<(), crate::FormatError> {
        if self.magic != LBC_MAGIC {
            return Err(crate::FormatError::InvalidMagic {
                expected: LBC_MAGIC,
                found: self.magic,
            });
        }
        if self.version > LBC_VERSION {
            return Err(crate::FormatError::UnsupportedVersion {
                version: self.version,
                max_supported: LBC_VERSION,
            });
        }
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(crate::FormatError::AlignmentViolation {
                offset: 0,
                required: self.alignment,
                actual: 0,
            });
        }
        if self.num_layers != self.hyperparams.num_layers {
            return Err(crate::FormatError::LayerCountMismatch {
                header_count: self.num_layers,
                hyperparams_count: self.hyperparams.num_layers,
            });
        }
        Ok(())
    }

    /// Creates a header with default offsets (caller fills them in after layout).
    pub fn new(hyperparams: ModelHyperparams, quantization: QuantizationDescriptor) -> Self {
        let num_layers = hyperparams.num_layers;
        let has_expert_index = hyperparams.is_moe();
        Self {
            magic: LBC_MAGIC,
            version: LBC_VERSION,
            endianness: Endianness::Little,
            header_checksum: 0, // computed during serialization
            hyperparams,
            quantization,
            alignment: DEFAULT_ALIGNMENT,
            num_layers,
            has_expert_index,
            layer_index_offset: 0,
            expert_index_offset: 0,
            payload_offset: 0,
            embedding: GlobalTensorRange::default(),
            final_norm: GlobalTensorRange::default(),
            output_proj: GlobalTensorRange::default(),
            weight_tying: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperparams::{ModelHyperparams, RopeParams, RopeScalingType};
    use crate::quantization::{QuantGroupSize, QuantScheme};

    fn test_hyperparams() -> ModelHyperparams {
        ModelHyperparams {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_size: 32,
            max_seq_len: 64,
            rope_params: Some(RopeParams {
                theta: 10000.0,
                scaling_factor: 1.0,
                scaling_type: RopeScalingType::None,
            }),
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-5,
        }
    }

    fn test_quant() -> QuantizationDescriptor {
        QuantizationDescriptor {
            scheme: QuantScheme::F32,
            group_size: QuantGroupSize::PerTensor,
            block_byte_size: 4,
            scale_offset_in_block: None,
        }
    }

    #[test]
    fn new_then_validate_passes() {
        let header = LbcHeader::new(test_hyperparams(), test_quant());
        header.validate().unwrap();
    }

    #[test]
    fn wrong_magic_fails() {
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.magic = 0xDEADBEEF;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, crate::FormatError::InvalidMagic { .. }));
    }

    #[test]
    fn version_validation() {
        // Version 0 and 1 pass (≤ LBC_VERSION which is 2)
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.version = 0;
        header.validate().unwrap();

        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.version = 1;
        header.validate().unwrap();

        // Version 3 fails (> LBC_VERSION)
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.version = 3;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, crate::FormatError::UnsupportedVersion { .. }));
    }

    #[test]
    fn alignment_validation() {
        // 0 fails
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.alignment = 0;
        assert!(header.validate().is_err());

        // 3 fails (not power of two)
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.alignment = 3;
        assert!(header.validate().is_err());

        // 1 passes
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.alignment = 1;
        header.validate().unwrap();

        // 128 passes
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.alignment = 128;
        header.validate().unwrap();
    }

    #[test]
    fn layer_count_mismatch_fails() {
        let mut header = LbcHeader::new(test_hyperparams(), test_quant());
        header.num_layers = 99;
        let err = header.validate().unwrap_err();
        assert!(matches!(err, crate::FormatError::LayerCountMismatch { .. }));
    }
}
