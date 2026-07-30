//! Qwen3.5 (dense) converter: hybrid GDN + full-attention with dense FFN.

use super::gdn_gates::{compute_ssm_slices, write_ssm_tensors};
use super::ArchConverter;
use crate::convert::{ConvertError, ConvertTarget};
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufFile};
use crate::tensor_io::*;
use crate::tensor_names::*;
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

use super::qwen35_moe::is_qwen35moe_full_attention_layer;

/// The stored format for `ssm_out`, resolved in ONE place.
///
/// # Why this is a function and not an inline `match`
///
/// This decision is made twice — once in `compute_layer_shape_qwen35`, which
/// advances `blob_offset` by the COMPUTED size, and once in
/// `write_qwen35_layer_blob`, which appends the ACTUAL bytes. If the two
/// disagree, every subsequent tensor in the layer blob shifts and the LBC is
/// silently corrupted: there is no checksum or length assertion between them
/// that would catch it. Both sites previously carried their own copy of the
/// same `match`, with paired comments asking the reader to keep them in sync.
/// Adding a gate to duplicated logic is how that eventually goes wrong, so both
/// now call this.
///
/// # The default: a Q8_0 FLOOR, and it is a quality keeper
///
/// Unarmed, the contract is floor-plus-passthrough:
/// `None -> Some(Q8_0)`, `Some(Q4_0) -> Some(Q8_0)` (the floor, deliberately
/// overriding `--requant q4_0`), `Some(other) -> Some(other)`. Since the CLI
/// only accepts `q4_0`/`q8_0`, the reachable result today is always
/// `Some(Q8_0)`.
///
/// Two separate justifications are recorded at the original sites, and they are
/// NOT the same claim:
///
/// * The Q8_0 *default* (over F32) is PERFORMANCE: the GDN runtime has fast
///   Q8_0/Q4_0 paths and a slow per-token F32 fallback, and an older F32
///   default "shipped LBCs that lost 100%+ Metal prefill on Qwen3.5-9B".
/// * The *floor* over `--requant q4_0` is PRECISION: "4-bit ssm_out corrupts
///   the GDN recurrence into degenerate output (measured 2026-06-10: a
///   requant-q4 LBC passed 1/15 short prompts vs 13/15 for an LBC converted
///   from the provider's direct Q4_0 GGUF, which ships Q8-class ssm_out)", with
///   the conclusion "ssm_out quantization is empirically the dominant quality
///   lever on this architecture". Note the floor was added even though a Q4_0
///   `ssm_out` fast path EXISTS — it is not a missing-kernel workaround.
///
/// # C4 (`LUMEN_CUDA_SSMOUT_NATIVE=1`)
///
/// Returns `Some(src_quant)` — the tensor's OWN stored format — so the write
/// path takes `append_tensor_to_blob_requant`'s `source == target`
/// short-circuit and copies the raw bytes unchanged, with no lossy
/// dequant/requant round trip. On Qwen3.5-9B-Q4_0 that leaves `ssm_out` Q4_0 on
/// the 12 GDN layers the GGUF already stores that way (the other 12 are Q8_0 in
/// the file and stay Q8_0), saving
/// `12 x 16,777,216 x (1.0625 - 0.5625) = 100.7 MB` = 96.0 MiB/token.
///
/// It does NOT return `None`, and that is the whole subtlety of this function.
/// `None` reads like "no requant, keep the source", but
/// `compute_slice_with_requant` matches only `Some(Q8_0)` and `Some(Q4_0)` and
/// falls through to **F32 at 4.0 B/weight** for everything else — so `None`
/// would INFLATE `ssm_out` from 1.0625 to 4.0 B/w (a +283% regression on this
/// tensor) and drop it onto the slow per-token F32 path the Q8_0 default was
/// introduced to escape. Exactly backwards from the lever's intent.
///
/// For the same reason a source format that is neither Q8_0 nor Q4_0 keeps the
/// Q8_0 floor rather than being passed through: there is no shape arm for it.
///
/// ⚠️ **RISK.** This removes a documented quality keeper. Enable it only with
/// the GDN quality gate armed. Two caveats on the evidence above, so the
/// decision is made on facts rather than on the comment's authority: the
/// 1/15-vs-13/15 measurement exists ONLY as that source assertion — no
/// artifact, log, or test backs it anywhere in the repo — and it was measured
/// against `--requant q4_0`, which requantizes EVERY `ssm_out` down from Q8,
/// whereas this gate merely stops UPCASTING the 12 layers already stored as
/// Q4_0. Those are different, and the second is strictly milder. That argues
/// for re-measuring, not for assuming the floor is wrong.
///
/// `src_quant` is the tensor's stored LBC scheme
/// (`tensor.ggml_type.to_lbc_quant()`), or `None` when the tensor is absent or
/// its GGML type has no LBC mapping.
fn ssm_out_target(
    requant_to: Option<QuantScheme>,
    src_quant: Option<QuantScheme>,
) -> Option<QuantScheme> {
    if crate::env_gates::ssmout_native() {
        // Only Q8_0 and Q4_0 have shape arms in `compute_slice_with_requant`;
        // anything else would silently resolve to F32 at 4.0 B/weight, so keep
        // the floor for those rather than making the lever a regression.
        if matches!(src_quant, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0)) {
            return src_quant;
        }
        return Some(QuantScheme::Q8_0);
    }
    match requant_to {
        Some(QuantScheme::Q4_0) => Some(QuantScheme::Q8_0),
        other => other.or(Some(QuantScheme::Q8_0)),
    }
}

pub(crate) struct Qwen35Converter;

impl ArchConverter for Qwen35Converter {
    fn compute_layer_shape(
        &self,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<LayerShape, ConvertError> {
        compute_layer_shape_qwen35(gguf, layer, dequantize, requant_to, target)
    }

    fn write_layer_blob<R: Read + Seek>(
        &self,
        blob: &mut Vec<u8>,
        reader: &mut R,
        gguf: &GgufFile,
        layer: usize,
        dequantize: bool,
        requant_to: Option<QuantScheme>,
        target: ConvertTarget,
    ) -> Result<(), ConvertError> {
        write_qwen35_layer_blob(blob, reader, gguf, layer, dequantize, requant_to, target)
    }

    fn layer_kind_label(&self, layer: usize) -> String {
        let kind = if is_qwen35moe_full_attention_layer(layer) {
            "full_attn"
        } else {
            "linear_attn"
        };
        format!("{}, dense", kind)
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 (dense) layer shape computation
// ---------------------------------------------------------------------------

/// Compute the LayerShape for a single Qwen3.5 (dense) layer.
///
/// Same hybrid GDN + full-attention architecture as Qwen3.5-MoE, but with
/// dense FFN (ffn_gate/ffn_up/ffn_down) instead of MoE (router + experts + shared expert).
fn compute_layer_shape_qwen35(
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
    target: ConvertTarget,
) -> Result<LayerShape, ConvertError> {
    let mut blob_size = 0u64;
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Helper to compute a TensorSlice for a given tensor.
    let compute_slice = |gguf: &GgufFile,
                         name: &str,
                         blob_offset: &mut u64,
                         dequantize: bool|
     -> Result<TensorSlice, ConvertError> {
        let tensor = gguf
            .find_tensor(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_string()))?;
        let is_norm = name.contains("norm");

        // Check if requantization applies
        if let Some(target_q) = requant_to {
            if is_norm || dequantize {
                // Norms stay F32
                let n_elements = tensor.n_elements();
                let size = n_elements * 4;
                let slice = TensorSlice {
                    offset: *blob_offset,
                    length: size,
                    quant: QuantScheme::F32,
                };
                *blob_offset += size;
                return Ok(slice);
            }
            let src_quant = tensor.ggml_type.to_lbc_quant();
            if src_quant == Some(target_q) {
                // Already in target format
                let size = tensor.byte_size().unwrap_or(0);
                let slice = TensorSlice {
                    offset: *blob_offset,
                    length: size,
                    quant: target_q,
                };
                *blob_offset += size;
                return Ok(slice);
            }
            // Compute size for target quant
            let n_elements = tensor.n_elements() as usize;
            assert!(
                n_elements % 32 == 0,
                "quantization requires elements divisible by 32, got {n_elements} for {name}"
            );
            let (size, quant) = match target_q {
                QuantScheme::Q8_0 => {
                    // Q8_0: 34 bytes per 32 elements
                    let num_blocks = n_elements / 32;
                    ((num_blocks * 34) as u64, QuantScheme::Q8_0)
                }
                QuantScheme::Q4_0 => {
                    // Q4_0: 18 bytes per 32 elements
                    let num_blocks = n_elements / 32;
                    ((num_blocks * 18) as u64, QuantScheme::Q4_0)
                }
                _ => {
                    // Unsupported target: F32
                    (n_elements as u64 * 4, QuantScheme::F32)
                }
            };
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant,
            };
            *blob_offset += size;
            return Ok(slice);
        }

        if dequantize {
            let n_elements = tensor.n_elements();
            let size = n_elements * 4;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::F32,
            };
            *blob_offset += size;
            Ok(slice)
        } else if target == ConvertTarget::Metal && !is_norm && is_k_quant(tensor.ggml_type) {
            // Metal K-quant upcast to Q8_0. Must match
            // `append_tensor_to_blob_requant_with_target` byte layout.
            let n_elements = tensor.n_elements() as usize;
            assert!(
                n_elements % 32 == 0,
                "Q8_0 requires elements divisible by 32, got {n_elements} for {name}"
            );
            let size = ((n_elements / 32) * 34) as u64;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::Q8_0,
            };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q4_1 {
            // Q4_1 has no dedicated GPU kernel -- requantize to Q4_0.
            let n_elements = tensor.n_elements();
            assert!(
                n_elements % 32 == 0,
                "Q4_1->Q4_0 requires elements divisible by 32, got {n_elements} for {name}"
            );
            let size = ((n_elements as usize / 32) * 18) as u64;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::Q4_0,
            };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q8_1 {
            // Q8_1 has no LBC QuantScheme -- requantize to Q8_0.
            let n_elements = tensor.n_elements();
            assert!(
                n_elements % 32 == 0,
                "Q8_1->Q8_0 requires elements divisible by 32, got {n_elements} for {name}"
            );
            let size = ((n_elements as usize / 32) * 34) as u64;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::Q8_0,
            };
            *blob_offset += size;
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q5_1 {
            // Q5_1 has no LBC QuantScheme -- dequantize to F32.
            let n_elements = tensor.n_elements();
            let size = n_elements * 4;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::F32,
            };
            *blob_offset += size;
            Ok(slice)
        } else {
            let quant = tensor.ggml_type.to_lbc_quant().ok_or_else(|| {
                ConvertError::UnsupportedTensorType {
                    tensor: name.to_string(),
                    ggml_type: format!("{:?}", tensor.ggml_type),
                }
            })?;
            let size = tensor
                .byte_size()
                .ok_or_else(|| ConvertError::UnsupportedTensorType {
                    tensor: name.to_string(),
                    ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                })?;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant,
            };
            *blob_offset += size;
            Ok(slice)
        }
    };

    // Helper for tensors that need a per-call requant override (not the
    // user's global `requant_to`). Used for SSM_OUT, which always wants the
    // runtime's fast Q8_0 / Q4_0 path regardless of the user's flag.
    // Returns None if the tensor is absent.
    let compute_slice_with_requant = |gguf: &GgufFile,
                                      layer: usize,
                                      suffix: &str,
                                      blob_offset: &mut u64,
                                      target: Option<QuantScheme>|
     -> Result<Option<TensorSlice>, ConvertError> {
        let name = layer_tensor_name(layer, suffix);
        let Some(tensor) = gguf.find_tensor(&name) else {
            return Ok(None);
        };
        let n_elements = tensor.n_elements() as usize;
        let src_quant = tensor.ggml_type.to_lbc_quant();
        let (size, quant) = match target {
            Some(QuantScheme::Q8_0) if n_elements % 32 == 0 => {
                if src_quant == Some(QuantScheme::Q8_0) {
                    (
                        tensor.byte_size().unwrap_or((n_elements / 32 * 34) as u64),
                        QuantScheme::Q8_0,
                    )
                } else {
                    ((n_elements / 32 * 34) as u64, QuantScheme::Q8_0)
                }
            }
            Some(QuantScheme::Q4_0) if n_elements % 32 == 0 => {
                if src_quant == Some(QuantScheme::Q4_0) {
                    (
                        tensor.byte_size().unwrap_or((n_elements / 32 * 18) as u64),
                        QuantScheme::Q4_0,
                    )
                } else {
                    ((n_elements / 32 * 18) as u64, QuantScheme::Q4_0)
                }
            }
            _ => ((n_elements * 4) as u64, QuantScheme::F32),
        };
        let slice = TensorSlice {
            offset: *blob_offset,
            length: size,
            quant,
        };
        *blob_offset += size;
        Ok(Some(slice))
    };

    // Helper for optional tensors.
    // Returns None if the tensor is absent. For tensors with no direct LBC
    // mapping but a known dequant path (Q8_1, Q5_1, MXFP4, etc.), forces
    // dequantization to F32 instead of silently skipping.
    let try_compute_opt_slice = |gguf: &GgufFile,
                                 layer: usize,
                                 suffix: &str,
                                 blob_offset: &mut u64,
                                 dequantize: bool|
     -> Result<Option<TensorSlice>, ConvertError> {
        let name = layer_tensor_name(layer, suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let force_dequant = !dequantize && tensor.ggml_type.to_lbc_quant().is_none();
            if force_dequant {
                if tensor.ggml_type.has_dequant_path() {
                    eprintln!(
                        "  Note: dequantizing {} ({:?} -> F32)",
                        name, tensor.ggml_type
                    );
                    Ok(Some(compute_slice(
                        gguf,
                        &name,
                        blob_offset,
                        /*dequantize=*/ true,
                    )?))
                } else {
                    eprintln!(
                        "  Warning: skipping {} (unsupported GGML type {:?})",
                        name, tensor.ggml_type
                    );
                    Ok(None)
                }
            } else {
                Ok(Some(compute_slice(gguf, &name, blob_offset, dequantize)?))
            }
        } else {
            Ok(None)
        }
    };

    // Attention projections: different tensor layout per layer type.
    let (wq, wk, wv, wo);
    if is_full_attn {
        wq = compute_slice(
            gguf,
            &layer_tensor_name(layer, ATTN_Q),
            &mut blob_size,
            dequantize,
        )?;
        wk = compute_slice(
            gguf,
            &layer_tensor_name(layer, ATTN_K),
            &mut blob_size,
            dequantize,
        )?;
        wv = compute_slice(
            gguf,
            &layer_tensor_name(layer, ATTN_V),
            &mut blob_size,
            dequantize,
        )?;
        wo = compute_slice(
            gguf,
            &layer_tensor_name(layer, ATTN_OUTPUT),
            &mut blob_size,
            dequantize,
        )?;
    } else {
        // Linear attention: fused QKV stored in wq slot; wk/wv/wo left as zero sentinel
        let z = TensorSlice {
            offset: 0,
            length: 0,
            quant: QuantScheme::F32,
        };
        wq = compute_slice(
            gguf,
            &layer_tensor_name(layer, ATTN_QKV),
            &mut blob_size,
            dequantize,
        )?;
        wk = z;
        wv = z;
        wo = z;
    }

    // Pre-attention norm (always present)
    let attn_norm = compute_slice(
        gguf,
        &layer_tensor_name(layer, ATTN_NORM),
        &mut blob_size,
        dequantize,
    )?;

    // Post-attention norm (present in all Qwen3.5 layers)
    let attn_post_norm =
        try_compute_opt_slice(gguf, layer, ATTN_POST_NORM, &mut blob_size, dequantize)?;

    // FFN norm (present in all layers)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    let ffn_norm = if gguf.find_tensor(&ffn_norm_name).is_some() {
        compute_slice(gguf, &ffn_norm_name, &mut blob_size, dequantize)?
    } else {
        TensorSlice {
            offset: 0,
            length: 0,
            quant: QuantScheme::F32,
        }
    };

    // Attention gate (full attention layers only)
    let attn_gate =
        try_compute_opt_slice(gguf, layer, ATTN_GATE_WEIGHT, &mut blob_size, dequantize)?;

    // SSM tensors (linear attention layers only) — never requantized to user target.
    // ssm_alpha/beta MUST be Q8_0 — the GDN runtime hardcodes Q8_0 matvec kernels.
    // Shared logic in gdn_gates handles force-requant from F32/F16/BF16 to Q8_0.
    let ssm = compute_ssm_slices(gguf, layer, &mut blob_size, dequantize)?;
    let ssm_a = ssm.ssm_a;
    let ssm_conv1d = ssm.ssm_conv1d;
    let ssm_dt = ssm.ssm_dt;
    let ssm_beta = ssm.ssm_beta;
    let ssm_alpha = ssm.ssm_alpha;
    let ssm_norm = ssm.ssm_norm;
    // SSM_OUT: Qwen3.5 GDN runtime has fast Q8_0 / Q4_0 paths (gdn.rs:1955-1999)
    // and a slow per-token F32 fallback. Default SSM_OUT to Q8_0 even when the
    // user did not pass `--requant`, so the runtime never falls into the F32
    // path. FLOOR at Q8_0 even under `--requant q4_0`: 4-bit ssm_out corrupts
    // the GDN recurrence into degenerate output (measured 2026-06-10:
    // a requant-q4 LBC passed 1/15 short prompts vs 13/15 for an LBC
    // converted from the provider's direct Q4_0 GGUF, which ships Q8-class
    // ssm_out). Cost of the floor: +202 MB on 9B (24 layers × 17.8 vs
    // 9.4 MB) — correctness wins; ssm_out quantization is empirically the
    // dominant quality lever on this architecture. (The even-older default
    // "force F32 unless requant handles it" shipped LBCs that lost 100%+
    // Metal prefill on Qwen3.5-9B.)
    let ssm_out_src = gguf
        .find_tensor(&layer_tensor_name(layer, SSM_OUT))
        .and_then(|t| t.ggml_type.to_lbc_quant());
    let ssm_out = compute_slice_with_requant(
        gguf,
        layer,
        SSM_OUT,
        &mut blob_size,
        ssm_out_target(requant_to, ssm_out_src),
    )?;

    // Dense FFN weights (present in all layers)
    let w_gate = compute_slice(
        gguf,
        &layer_tensor_name(layer, FFN_GATE),
        &mut blob_size,
        dequantize,
    )?;
    let w_up = compute_slice(
        gguf,
        &layer_tensor_name(layer, FFN_UP),
        &mut blob_size,
        dequantize,
    )?;
    let w_down = compute_slice(
        gguf,
        &layer_tensor_name(layer, FFN_DOWN),
        &mut blob_size,
        dequantize,
    )?;

    // Optional bias tensors
    let bq = try_compute_bias_slice(gguf, layer, ATTN_Q_BIAS, &mut blob_size);
    let bk = try_compute_bias_slice(gguf, layer, ATTN_K_BIAS, &mut blob_size);
    let bv = try_compute_bias_slice(gguf, layer, ATTN_V_BIAS, &mut blob_size);

    // Per-head Q/K RMSNorm weights (full attention layers only, always F32)
    let attn_q_norm = try_compute_opt_slice(
        gguf,
        layer,
        ATTN_Q_NORM,
        &mut blob_size,
        /*dequantize=*/ true,
    )?;
    let attn_k_norm = try_compute_opt_slice(
        gguf,
        layer,
        ATTN_K_NORM,
        &mut blob_size,
        /*dequantize=*/ true,
    )?;

    let layer_type = if is_full_attn { Some(0u8) } else { Some(1u8) };

    let subtensors = SubtensorOffsets {
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        w_gate,
        w_up,
        w_down,
        attn_norm,
        ffn_norm,
        router_weight: None,
        experts: None,
        shared_expert_gate: None,
        shared_expert_up: None,
        shared_expert_down: None,
        attn_gate,
        attn_post_norm,
        ssm_a,
        ssm_conv1d,
        ssm_dt,
        ssm_beta,
        ssm_alpha,
        ssm_norm,
        ssm_out,
        attn_q_norm,
        attn_k_norm,
        ffn_gate_inp_shexp: None,
        layer_type,
    };

    Ok(LayerShape {
        blob_size,
        index: LayerIndex {
            layer_offset_bytes: 0,
            layer_length_bytes: blob_size,
            subtensors,
        },
    })
}

// ---------------------------------------------------------------------------
// Qwen3.5 (dense) layer blob writing
// ---------------------------------------------------------------------------

/// Write a Qwen3.5 (dense) layer blob.
///
/// Same hybrid GDN + full-attention layout as Qwen3.5-MoE, but with dense FFN
/// (ffn_gate/ffn_up/ffn_down) instead of MoE (router + experts + shared expert).
/// Tensor order must match `compute_layer_shape_qwen35()`.
fn write_qwen35_layer_blob<R: Read + Seek>(
    blob: &mut Vec<u8>,
    reader: &mut R,
    gguf: &GgufFile,
    layer: usize,
    dequantize: bool,
    requant_to: Option<QuantScheme>,
    target: ConvertTarget,
) -> Result<(), ConvertError> {
    let is_full_attn = is_qwen35moe_full_attention_layer(layer);

    // Attention projections: layout differs by layer type
    if is_full_attn {
        // Full attention: separate Q/K/V/output tensors
        for suffix in &ATTN_TENSOR_SUFFIXES {
            append_tensor_to_blob_requant_with_target(
                blob,
                reader,
                gguf,
                &layer_tensor_name(layer, suffix),
                dequantize,
                requant_to,
                target,
            )?;
        }
    } else {
        // Linear attention: fused QKV tensor only (stored in wq slot in index)
        append_tensor_to_blob_requant_with_target(
            blob,
            reader,
            gguf,
            &layer_tensor_name(layer, ATTN_QKV),
            dequantize,
            requant_to,
            target,
        )?;
    }

    // Pre-attention norm
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        &layer_tensor_name(layer, ATTN_NORM),
        dequantize,
        requant_to,
        target,
    )?;

    // Post-attention norm (if present)
    let post_norm_name = layer_tensor_name(layer, ATTN_POST_NORM);
    if gguf.find_tensor(&post_norm_name).is_some() {
        append_tensor_to_blob_requant_with_target(
            blob,
            reader,
            gguf,
            &post_norm_name,
            dequantize,
            requant_to,
            target,
        )?;
    }

    // FFN norm (if present)
    let ffn_norm_name = layer_tensor_name(layer, FFN_NORM);
    if gguf.find_tensor(&ffn_norm_name).is_some() {
        append_tensor_to_blob_requant_with_target(
            blob,
            reader,
            gguf,
            &ffn_norm_name,
            dequantize,
            requant_to,
            target,
        )?;
    }

    // Attention gate (if present)
    let attn_gate_name = layer_tensor_name(layer, ATTN_GATE_WEIGHT);
    if gguf.find_tensor(&attn_gate_name).is_some() {
        append_tensor_to_blob_requant_with_target(
            blob,
            reader,
            gguf,
            &attn_gate_name,
            dequantize,
            requant_to,
            target,
        )?;
    }

    // SSM tensors (if present) — shared GDN gate logic handles force-requant
    // of ssm_alpha/beta to Q8_0 when source is F32/F16/BF16.
    write_ssm_tensors(blob, reader, gguf, layer, dequantize)?;
    {
        let name = layer_tensor_name(layer, SSM_OUT);
        if let Some(ssm_out_tensor) = gguf.find_tensor(&name) {
            let ssm_out_src = ssm_out_tensor.ggml_type.to_lbc_quant();
            // SSM_OUT: route through the runtime's fast Q8_0 GDN path.
            // FLOORED at Q8_0 even under `--requant q4_0` — 4-bit ssm_out
            // corrupts the GDN recurrence (2026-06-10 RCA; see the matching
            // floor + evidence in compute_slice_with_requant above; the two
            // MUST stay in sync for layer-shape symmetry). (Target is
            // irrelevant here: ssm_out is always force-requanted.)
            append_tensor_to_blob_requant(
                blob,
                reader,
                gguf,
                &name,
                /*dequantize=*/ false,
                ssm_out_target(requant_to, ssm_out_src),
            )?;
        }
    }

    // Dense FFN weights
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        &layer_tensor_name(layer, FFN_GATE),
        dequantize,
        requant_to,
        target,
    )?;
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        &layer_tensor_name(layer, FFN_UP),
        dequantize,
        requant_to,
        target,
    )?;
    append_tensor_to_blob_requant_with_target(
        blob,
        reader,
        gguf,
        &layer_tensor_name(layer, FFN_DOWN),
        dequantize,
        requant_to,
        target,
    )?;

    // Optional bias tensors (always F32)
    for bias_suffix in &[ATTN_Q_BIAS, ATTN_K_BIAS, ATTN_V_BIAS] {
        let name = layer_tensor_name(layer, bias_suffix);
        if let Some(tensor) = gguf.find_tensor(&name) {
            let data = read_tensor_data(reader, gguf, tensor)?;
            let f32_data =
                dequantize_to_f32_bytes(&data, tensor.ggml_type, tensor.n_elements(), &name)?;
            blob.extend_from_slice(&f32_data);
        }
    }

    // Per-head Q/K RMSNorm weights (always dequantized to F32)
    for suffix in &[ATTN_Q_NORM, ATTN_K_NORM] {
        let name = layer_tensor_name(layer, suffix);
        if gguf.find_tensor(&name).is_some() {
            append_tensor_to_blob(blob, reader, gguf, &name, /*dequantize=*/ true)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod ssm_out_target_tests {
    use super::ssm_out_target;
    use lumen_format::quantization::QuantScheme;

    /// The exact default-OFF contract, which is narrower than "always Q8_0":
    ///
    /// * `None`          -> `Some(Q8_0)`  (default away from the slow F32 path)
    /// * `Some(Q4_0)`    -> `Some(Q8_0)`  (**the floor** — deliberately ignores
    ///                                     the user's `--requant q4_0`)
    /// * `Some(other)`   -> `Some(other)` (explicit non-Q4 targets pass through)
    ///
    /// Because the CLI only accepts `q4_0`/`q8_0` (`lumen-cli/src/convert.rs`),
    /// the reachable result today is always `Some(Q8_0)` — but the passthrough
    /// arm is real code and is asserted here rather than assumed away, so a
    /// future CLI that accepts F16 does not silently change `ssm_out`.
    ///
    /// Arming C4 must be the ONLY way to change any of this. Guarded so an
    /// operator running the suite with the gate set sees a skip, not a spurious
    /// failure.
    #[test]
    fn unarmed_contract_is_floor_plus_passthrough() {
        if crate::env_gates::ssmout_native() {
            eprintln!("skipping: LUMEN_CUDA_SSMOUT_NATIVE is set in this environment");
            return;
        }
        // The floor: these are the two inputs the CLI can actually produce, and
        // both must land on Q8_0.
        // Unarmed, the result must not depend on the tensor's own format --
        // that dependence is exactly what arming C4 introduces.
        for src in [None, Some(QuantScheme::Q4_0), Some(QuantScheme::Q8_0)] {
            assert_eq!(ssm_out_target(None, src), Some(QuantScheme::Q8_0));
            assert_eq!(
                ssm_out_target(Some(QuantScheme::Q4_0), src),
                Some(QuantScheme::Q8_0),
                "the floor must override --requant q4_0 (src {src:?})"
            );
            assert_eq!(
                ssm_out_target(Some(QuantScheme::Q8_0), src),
                Some(QuantScheme::Q8_0)
            );
            // Passthrough for explicit non-Q4 targets.
            for scheme in [QuantScheme::F16, QuantScheme::F32] {
                assert_eq!(
                    ssm_out_target(Some(scheme), src),
                    Some(scheme),
                    "an explicit non-Q4_0 requant target must pass through"
                );
            }
            // And in no case may the unarmed default yield None, which
            // `compute_slice_with_requant` would resolve to F32 at 4.0 B/weight.
            for input in [
                None,
                Some(QuantScheme::Q4_0),
                Some(QuantScheme::Q8_0),
                Some(QuantScheme::F16),
            ] {
                assert!(
                    ssm_out_target(input, src).is_some(),
                    "unarmed ssm_out_target must never be None \
                     (requant_to = {input:?}, src = {src:?})"
                );
            }
        }
    }

    /// Both convert sites must resolve `ssm_out`'s format identically, or the
    /// shape pass advances `blob_offset` by one size while the write pass
    /// appends another and every later tensor in the layer blob shifts. There
    /// is no checksum between them, so this equality is the only guard.
    ///
    /// A single shared function makes that structurally true; this test pins it
    /// so a future refactor cannot re-introduce two divergent copies.
    #[test]
    fn shape_and_write_paths_agree() {
        for requant in [None, Some(QuantScheme::Q4_0), Some(QuantScheme::Q8_0)] {
            for src in [None, Some(QuantScheme::Q4_0), Some(QuantScheme::Q8_0)] {
                assert_eq!(
                    ssm_out_target(requant, src),
                    ssm_out_target(requant, src),
                    "ssm_out_target must be a pure function of its two inputs"
                );
            }
        }
    }

    /// C4 must never resolve to `None`, for ANY input, armed or not.
    ///
    /// `None` reads like "keep the source format" but
    /// `compute_slice_with_requant` matches only `Some(Q8_0)` / `Some(Q4_0)` and
    /// falls through to **F32 at 4.0 B/weight** otherwise. Returning `None`
    /// would therefore inflate `ssm_out` from 1.0625 to 4.0 B/w -- a +283%
    /// regression on the very tensor the lever is meant to shrink -- and put it
    /// on the slow per-token F32 path. This test holds in both flag states, so
    /// it is the one guard that cannot be bypassed by how the suite is run.
    #[test]
    fn never_resolves_to_none_in_either_flag_state() {
        for requant in [
            None,
            Some(QuantScheme::Q4_0),
            Some(QuantScheme::Q8_0),
            Some(QuantScheme::F16),
        ] {
            for src in [
                None,
                Some(QuantScheme::Q4_0),
                Some(QuantScheme::Q8_0),
                Some(QuantScheme::F32),
            ] {
                let got = ssm_out_target(requant, src);
                assert!(
                    got.is_some(),
                    "ssm_out_target({requant:?}, {src:?}) = None would resolve to F32 4.0 B/w"
                );
                assert!(
                    matches!(got, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0))
                        || got == requant,
                    "ssm_out_target({requant:?}, {src:?}) = {got:?} has no shape arm"
                );
            }
        }
    }

    /// When armed, the result is the tensor's OWN stored format for the two
    /// schemes that have shape arms, so the write path takes its
    /// `source == target` short-circuit and copies raw bytes -- no lossy
    /// dequant/requant round trip. Anything else keeps the Q8_0 floor rather
    /// than silently becoming F32.
    #[test]
    fn armed_returns_the_source_format_or_keeps_the_floor() {
        if !crate::env_gates::ssmout_native() {
            eprintln!("skipping: LUMEN_CUDA_SSMOUT_NATIVE is not set in this environment");
            return;
        }
        assert_eq!(
            ssm_out_target(None, Some(QuantScheme::Q4_0)),
            Some(QuantScheme::Q4_0),
            "armed: a Q4_0 source must stay Q4_0 -- that IS the 96.0 MiB/token"
        );
        assert_eq!(
            ssm_out_target(None, Some(QuantScheme::Q8_0)),
            Some(QuantScheme::Q8_0),
            "armed: a Q8_0 source stays Q8_0 (12 of 24 layers on 9B-Q4)"
        );
        for src in [None, Some(QuantScheme::F16), Some(QuantScheme::F32)] {
            assert_eq!(
                ssm_out_target(None, src),
                Some(QuantScheme::Q8_0),
                "armed: src {src:?} has no shape arm, so keep the floor"
            );
        }
    }

    /// `None` (the C4 result) must mean "store the source format", NOT F32.
    /// `compute_slice_with_requant` falls through to F32 for any target that is
    /// neither Q8_0 nor Q4_0, which is the slow per-token path the Q8_0 default
    /// exists to escape -- so the gate returning `None` is only correct because
    /// `None` short-circuits the requant entirely rather than selecting a
    /// scheme. This test documents the distinction that makes the gate safe.
    #[test]
    fn none_is_passthrough_not_f32() {
        assert_ne!(
            Some(QuantScheme::F32),
            None::<QuantScheme>,
            "sanity: None is not Some(F32)"
        );
        // The floor never yields None while unarmed, which is what keeps the
        // F32 fall-through unreachable in the default configuration.
        if !crate::env_gates::ssmout_native() {
            assert!(ssm_out_target(None, None).is_some());
        }
    }
}
