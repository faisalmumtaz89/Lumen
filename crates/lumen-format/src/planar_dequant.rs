//! Host dequantization for the planar weight schemes: the reference the
//! device kernels are held to, and the only place the decode is written.
//!
//! Both schemes decode as bit arithmetic on f32, never through a hardware
//! conversion instruction, so the result is identical on every host and
//! device. The two rules that are easy to get wrong, and that the tests pin:
//!
//! * **The scales are folded first.** NVFP4's value is
//!   `E2M1(nibble) * f32(E4M3(block_scale) * global_scale)`. f32 multiplication
//!   is not associative, so multiplying the nibble by the block scale first
//!   and the global scale second gives a different bit pattern on ~11 % of
//!   values.
//! * **The two formats disagree about signed zero.** E4M3 keeps it: `0x80`
//!   decodes to `-0.0`. E2M1 does not: code `0x8` decodes to `+0.0`.
//!
//! A plain exponent shift is wrong on the 2 E2M1 and 14 E4M3 nonzero
//! subnormal codes, which is why both decoders special-case a zero exponent.

use crate::FormatError;

/// Weights per block scale in NVFP4.
pub const NVFP4_GROUP: usize = 16;

/// Decode one E2M1 code (the low 4 bits) to f32.
///
/// 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit. Both zero codes
/// decode to `+0.0`.
pub fn e2m1_to_f32(code: u8) -> f32 {
    let c = u32::from(code & 0x0F);
    let sign = (c & 8) << 28;
    let exp = (c >> 1) & 3;
    let mantissa = c & 1;
    let magnitude = if exp == 0 {
        // Subnormal: 0.5 for code 0x1, zero otherwise.
        if mantissa != 0 {
            126 << 23
        } else {
            0
        }
    } else {
        ((exp + 126) << 23) | (mantissa << 22)
    };
    // A zero magnitude drops the sign: this format's own table has one zero.
    f32::from_bits(if magnitude == 0 { 0 } else { sign | magnitude })
}

/// Decode one E4M3 byte to f32.
///
/// 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. `0x7F` and `0xFF`
/// are the format's only two NaNs; there are no infinities, and the largest
/// finite value is 448.0. `-0.0` is preserved.
pub fn e4m3_to_f32(code: u8) -> f32 {
    let c = u32::from(code);
    let sign = (c & 0x80) << 24;
    let exp = (c >> 3) & 0xF;
    let mantissa = c & 7;
    if exp == 15 && mantissa == 7 {
        return f32::NAN;
    }
    let magnitude = if exp == 0 {
        // Subnormal: mantissa / 512, exact in f32 for every mantissa 0..=7.
        (mantissa as f32 / 512.0).to_bits()
    } else {
        ((exp + 120) << 23) | (mantissa << 20)
    };
    f32::from_bits(sign | magnitude)
}

/// Dequantize one NVFP4 matrix's planes, row-major, into `weights.len() * 2`
/// values.
///
/// `packed` holds two E2M1 nibbles per byte, the low nibble first;
/// `block_scales` holds one E4M3 code per [`NVFP4_GROUP`] weights, in the
/// same order.
pub fn dequantize_nvfp4(
    packed: &[u8],
    block_scales: &[u8],
    global_scale: f32,
) -> Result<Vec<f32>, FormatError> {
    let elements = packed.len() * 2;
    if elements % NVFP4_GROUP != 0 || block_scales.len() * NVFP4_GROUP != elements {
        return Err(FormatError::UnsupportedQuantization(format!(
            "NVFP4 planes disagree: {} packed bytes are {elements} weights, \
             but {} block scales cover {} weights",
            packed.len(),
            block_scales.len(),
            block_scales.len() * NVFP4_GROUP
        )));
    }
    let mut out = Vec::with_capacity(elements);
    for (block, &scale_code) in block_scales.iter().enumerate() {
        // The two scales are folded into one f32 before the nibble multiply.
        let scale = e4m3_to_f32(scale_code) * global_scale;
        for byte in &packed[block * (NVFP4_GROUP / 2)..(block + 1) * (NVFP4_GROUP / 2)] {
            out.push(e2m1_to_f32(byte & 0x0F) * scale);
            out.push(e2m1_to_f32(byte >> 4) * scale);
        }
    }
    Ok(out)
}

/// Dequantize one FP8 matrix's weight plane with its per-tensor scale.
pub fn dequantize_fp8(weights: &[u8], scale: f32) -> Vec<f32> {
    weights.iter().map(|&c| e4m3_to_f32(c) * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E2M1 from its arithmetic definition rather than its bit layout:
    /// 2 exponent bits with bias 1, 1 mantissa bit, subnormals at 2^-1.
    /// Independent of the decoder under test, which builds the bits directly.
    fn e2m1_by_arithmetic(code: u8) -> f32 {
        let exp = i32::from((code >> 1) & 3);
        let mantissa = f32::from(code & 1);
        let magnitude = if exp == 0 {
            mantissa * 0.5
        } else {
            (1.0 + mantissa * 0.5) * 2f32.powi(exp - 1)
        };
        // ModelOpt's table gives both zero codes +0.0.
        if magnitude == 0.0 {
            0.0
        } else if code & 8 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    /// E4M3 from its arithmetic definition: 4 exponent bits with bias 7,
    /// 3 mantissa bits, subnormals at 2^-6, signed zero preserved.
    fn e4m3_by_arithmetic(code: u8) -> f32 {
        let exp = i32::from((code >> 3) & 0xF);
        let mantissa = f32::from(code & 7);
        let magnitude = if exp == 0 {
            mantissa / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exp - 7)
        };
        if code & 0x80 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[test]
    fn e2m1_matches_the_format_arithmetic_on_all_16_codes() {
        for code in 0u8..16 {
            assert_eq!(
                e2m1_to_f32(code).to_bits(),
                e2m1_by_arithmetic(code).to_bits(),
                "E2M1 code {code:#04x}: {} vs {}",
                e2m1_to_f32(code),
                e2m1_by_arithmetic(code)
            );
        }
        // The values the two subnormal codes carry, which an exponent shift
        // alone gets wrong.
        assert_eq!(e2m1_to_f32(0x1), 0.5);
        assert_eq!(e2m1_to_f32(0x9), -0.5);
        // Both zeros are +0.0: this format's table drops the sign.
        assert_eq!(e2m1_to_f32(0x0).to_bits(), 0);
        assert_eq!(e2m1_to_f32(0x8).to_bits(), 0);
        // The endpoints.
        assert_eq!(e2m1_to_f32(0x7), 6.0);
        assert_eq!(e2m1_to_f32(0xF), -6.0);
    }

    #[test]
    fn e4m3_matches_the_format_arithmetic_on_all_254_finite_codes() {
        let mut finite = 0;
        let mut nan = 0;
        for code in 0u8..=255 {
            if code == 0x7F || code == 0xFF {
                assert!(e4m3_to_f32(code).is_nan(), "code {code:#04x} is a NaN");
                nan += 1;
                continue;
            }
            assert_eq!(
                e4m3_to_f32(code).to_bits(),
                e4m3_by_arithmetic(code).to_bits(),
                "E4M3 code {code:#04x}: {} vs {}",
                e4m3_to_f32(code),
                e4m3_by_arithmetic(code)
            );
            finite += 1;
        }
        assert_eq!((finite, nan), (254, 2));
        // Signed zero is preserved, unlike E2M1.
        assert_eq!(e4m3_to_f32(0x00).to_bits(), 0);
        assert_eq!(e4m3_to_f32(0x80).to_bits(), 0x8000_0000);
        // The smallest subnormal and the largest finite value.
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0xFE), -448.0);
    }

    #[test]
    fn nvfp4_folds_the_scales_before_the_nibble_multiply() {
        // A block whose two scales round differently depending on the order:
        // the fold is the reference's, so it is what the decode must do.
        let block_scale = 0x33u8; // E4M3, a non-power-of-two
        let global = 4.417_783e-4f32;
        let packed = [0x76u8; 8]; // nibbles 6 and 7 = 4.0 and 6.0
        let out = dequantize_nvfp4(&packed, &[block_scale], global).unwrap();
        let folded = e4m3_to_f32(block_scale) * global;
        assert_eq!(out.len(), 16);
        assert_eq!(out[0].to_bits(), (e2m1_to_f32(6) * folded).to_bits());
        assert_eq!(out[1].to_bits(), (e2m1_to_f32(7) * folded).to_bits());
    }

    #[test]
    fn nvfp4_rejects_planes_that_do_not_cover_each_other() {
        // 8 packed bytes are one 16-weight block; two block scales claim 32.
        assert!(dequantize_nvfp4(&[0u8; 8], &[0x38, 0x38], 1.0).is_err());
        assert!(dequantize_nvfp4(&[0u8; 4], &[0x38], 1.0).is_err());
    }

    /// Read a file of little-endian f32 values.
    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            bytes.len() % 4,
            0,
            "{}: not whole f32 values",
            path.display()
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Decode every tensor of a reference-vector directory and compare bit
    /// for bit with the values the producing toolchain's own dequantizer
    /// wrote. Reference vectors are large, so they live outside the repo;
    /// point `LUMEN_NVFP4_FIXTURE` at the directory to run this.
    ///
    /// Per tensor the directory holds, for NVFP4, `<tag>.packed.u8`,
    /// `<tag>.blockscale.u8`, `<tag>.globalscale.f32`, `<tag>.expected.f32`,
    /// and for FP8 `<tag>.e4m3.u8`, `<tag>.scale.f32`, `<tag>.expected.f32`.
    #[test]
    #[ignore = "needs LUMEN_NVFP4_FIXTURE pointing at a reference-vector directory"]
    fn decode_is_bit_identical_to_the_reference_vectors() {
        let Ok(dir) = std::env::var("LUMEN_NVFP4_FIXTURE") else {
            println!("SKIP: LUMEN_NVFP4_FIXTURE is not set");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();

        let (mut nvfp4_words, mut fp8_words, mut tensors) = (0usize, 0usize, 0usize);
        for name in &names {
            let (tag, decoded) = if let Some(tag) = name.strip_suffix(".packed.u8") {
                let packed = std::fs::read(dir.join(name)).unwrap();
                let scales = std::fs::read(dir.join(format!("{tag}.blockscale.u8"))).unwrap();
                let global = read_f32(&dir.join(format!("{tag}.globalscale.f32")));
                assert_eq!(global.len(), 1, "{tag}: global scale is not one value");
                let out = dequantize_nvfp4(&packed, &scales, global[0]).unwrap();
                nvfp4_words += out.len();
                (tag, out)
            } else if let Some(tag) = name.strip_suffix(".e4m3.u8") {
                let weights = std::fs::read(dir.join(name)).unwrap();
                let scale = read_f32(&dir.join(format!("{tag}.scale.f32")));
                assert_eq!(scale.len(), 1, "{tag}: scale is not one value");
                let out = dequantize_fp8(&weights, scale[0]);
                fp8_words += out.len();
                (tag, out)
            } else {
                continue;
            };
            let expected = read_f32(&dir.join(format!("{tag}.expected.f32")));
            assert_eq!(decoded.len(), expected.len(), "{tag}: value count");
            let mismatches: Vec<usize> = (0..decoded.len())
                .filter(|&i| decoded[i].to_bits() != expected[i].to_bits())
                .collect();
            assert!(
                mismatches.is_empty(),
                "{tag}: {} of {} values differ; first at {}: {:#010x} vs {:#010x}",
                mismatches.len(),
                decoded.len(),
                mismatches[0],
                decoded[mismatches[0]].to_bits(),
                expected[mismatches[0]].to_bits()
            );
            tensors += 1;
        }
        assert!(tensors > 0, "{}: no tensors found", dir.display());
        println!(
            "bit-identical: {tensors} tensors, {nvfp4_words} NVFP4 and {fp8_words} FP8 values, \
             0 mismatches"
        );
    }

    #[test]
    fn fp8_scales_every_code_by_the_tensor_scale() {
        let scale = 3.330_776e-3f32;
        let out = dequantize_fp8(&[0x00, 0x80, 0x7E, 0x01], scale);
        assert_eq!(out[0].to_bits(), (0.0f32 * scale).to_bits());
        assert_eq!(out[1].to_bits(), (-0.0f32 * scale).to_bits());
        assert_eq!(out[2].to_bits(), (448.0f32 * scale).to_bits());
        assert_eq!(out[3].to_bits(), (2f32.powi(-9) * scale).to_bits());
    }
}
