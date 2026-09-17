//! Qwen3.5 (dense) converter: hybrid GDN + full-attention with dense FFN.

use super::gdn_gates::{compute_ssm_slices, write_ssm_tensors};
use super::ArchConverter;
use crate::convert::{ConvertError, ConvertTarget};
use crate::dequant::*;
use crate::gguf::{GgmlType, GgufFile};
use crate::tensor_io::*;
use crate::tensor_names::*;
use lumen_format::hyperparams::GdnDims;
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use lumen_format::streaming_writer::LayerShape;
use std::io::{Read, Seek};

use super::qwen35_moe::is_qwen35moe_full_attention_layer;

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

/// Whether a layer's `ssm_out` keeps its source scheme. The layer plan and the
/// layer write both call this one function, on the same `gdn_v_dim` and the same
/// conversion flags, so the two cannot disagree. Under source fidelity — which a
/// K-quant source conversion takes by default — a Q5_K or Q8_0 `ssm_out` is kept, and
/// a K-quant source conversion also keeps a Q4_K or Q6_K one. A K-quant artifact
/// serves every kept K-quant `ssm_out` through the general K-quant kernels, Q5_K
/// included; outside one, a kept Q5_K `ssm_out` has its own dedicated kernel. A kept
/// Q8_0 one takes its split sibling either way. Only on a target that serves K-quant
/// planes: the Metal target requantises it.
///
/// What the DEFAULT keeps has to be servable, exactly as the head arm's default does
/// (`convert.rs`), and servable on the artifact the conversion is actually writing:
///
/// - Geometry: the kernels read this plane at `gdn_v_dim` and would truncate a row
///   that is not whole blocks, and the converter's own post-planning gate
///   (`serving_rules::validate_layer_plan`) refuses such a plan before a byte is
///   written, so a K-quant source whose `gdn_v_dim` is not whole blocks for the source
///   scheme takes the `ssm_out` 0.31.0 planned for it — requantised to Q8_0, or a
///   stored Q8_0 unchanged, which that gate refuses either way at a width that is not
///   whole 32-element blocks — rather than a kept plane the gate would refuse.
/// - Header: `--requant` and `--dequantize` stamp a Q8_0 / Q4_0 / F32 primary scheme,
///   and CUDA's Q4_K / Q5_K / Q6_K layer arms are scoped on a K-quant header
///   (`runtime_defaults::kquant_artifact`), so a K-quant plane kept under one of those
///   headers would be dequantised to F32 at load — 4 bytes per weight, against the 34
///   bytes per 32 weights of the Q8_0 `ssm_out` 0.31.0 wrote there, and against the
///   144 bytes per 256 weights the source stores a Q4_K one in. The default therefore
///   keeps nothing on those two routes: they write the `ssm_out` 0.31.0 wrote for
///   them. The embedding and a preserved Q6_K head are kept there because their arms
///   read the plane's own scheme whatever the header says.
///
/// The explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch is outside both rules: it keeps
/// the Q5_K and Q8_0 `ssm_out` 0.31.0 kept, at every width and under `--requant` /
/// `--dequantize` as well (both of those schemes are served whatever the header is),
/// and where that gate refuses the plan the conversion is refused with it, exactly as
/// in 0.31.0.
pub(crate) fn ssm_out_keeps_source(
    target: ConvertTarget,
    src: Option<GgmlType>,
    gdn_v_dim: usize,
    requant_to: Option<QuantScheme>,
    dequantize: bool,
) -> bool {
    if !crate::convert::target_serves_kquant(target) {
        return false;
    }
    // 0.31.0's answer under the explicit switch, unchanged at every width and flag.
    let explicit = crate::convert::source_fidelity_requested()
        && matches!(src, Some(GgmlType::Q5_K) | Some(GgmlType::Q8_0));
    // The K-quant source default: the same schemes plus Q4_K and Q6_K, only at a width
    // the kernels read whole blocks at, and only on the route that stamps the K-quant
    // header those two schemes' arms are scoped on.
    let default_keep = crate::convert::kquant_source()
        && requant_to.is_none()
        && !dequantize
        && matches!(
            src,
            Some(GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0)
        )
        && src.and_then(|t| t.to_lbc_quant()).is_some_and(|q| {
            lumen_format::serving_rules::validate_projection_row_width("ssm_out", q, gdn_v_dim)
                .is_ok()
        });
    explicit || default_keep
}

/// The width the loaders read a kept `ssm_out` at: the GDN V projection dimension,
/// `num_v_heads * head_dim`. Read from the source's SSM metadata through the same
/// keys, and with the same Qwen3.5-9B fallback, that `hyperparams::extract_hyperparams`
/// reads it with — `ssm.time_step_rank` is the presence signal for GDN dimensions, and
/// an undeclared shape resolves to [`GdnDims::QWEN35_9B`] — so this predicate and the
/// post-planning gate, which runs on `ModelHyperparams::gdn_dims()`, measure one width.
fn gdn_v_dim(gguf: &GgufFile) -> usize {
    let prefix = gguf.get_string("general.architecture").unwrap_or_default();
    let default = GdnDims::QWEN35_9B;
    match gguf.get_u32(&format!("{prefix}.ssm.time_step_rank")) {
        Some(num_v_heads) => {
            let head_dim = gguf
                .get_u32(&format!("{prefix}.ssm.state_size"))
                .unwrap_or(default.head_dim);
            num_v_heads as usize * head_dim as usize
        }
        None => default.v_dim() as usize,
    }
}

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
    let gdn_pair_q8 = !is_full_attn
        && super::gdn_gates::metal_gdn_pair_forces_q8(gguf, layer, dequantize, requant_to, target);

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
        // Norm tensors are always written F32 (mirrors the writer's forced
        // dequantize in `append_tensor_to_blob_requant_with_target` — every
        // backend reads norm weights as F32 and rejects anything else).
        if is_norm && tensor.ggml_type != GgmlType::F32 {
            let size = tensor.n_elements() * 4;
            let slice = TensorSlice {
                offset: *blob_offset,
                length: size,
                quant: QuantScheme::F32,
            };
            *blob_offset = blob_offset.saturating_add(size);
            return Ok(slice);
        }

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
                *blob_offset = blob_offset.saturating_add(size);
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
                *blob_offset = blob_offset.saturating_add(size);
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
            *blob_offset = blob_offset.saturating_add(size);
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
            *blob_offset = blob_offset.saturating_add(size);
            Ok(slice)
        } else if target == ConvertTarget::Metal && !is_norm && metal_needs_upcast(tensor.ggml_type)
        {
            // Metal K-quant / legacy-Q5_0 upcast to Q8_0. Must match
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
            *blob_offset = blob_offset.saturating_add(size);
            Ok(slice)
        } else if tensor.ggml_type == GgmlType::Q4_1 {
            if target != ConvertTarget::Metal && crate::convert::source_fidelity() {
                // SOURCE_FIDELITY: keep Q4_1 verbatim (the min term is part of
                // the source quantization; stripping it to Q4_0 is a quality
                // downcast the reference engine does not perform). CUDA-only:
                // Metal has no Q4_1 kernel, so the Metal target still
                // requantizes (mirrors append_tensor_to_blob_requant_with_target).
                let n_elements = tensor.n_elements();
                let size = ((n_elements as usize / 32) * 20) as u64;
                let slice = TensorSlice {
                    offset: *blob_offset,
                    length: size,
                    quant: QuantScheme::Q4_1,
                };
                *blob_offset = blob_offset.saturating_add(size);
                return Ok(slice);
            }
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
            *blob_offset = blob_offset.saturating_add(size);
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
            *blob_offset = blob_offset.saturating_add(size);
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
            *blob_offset = blob_offset.saturating_add(size);
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
            *blob_offset = blob_offset.saturating_add(size);
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
            None => {
                // SOURCE_FIDELITY passthrough: a None target keeps the source
                // scheme verbatim (the ssm_out caller only passes None for
                // sources the runtime serves natively — `ssm_out_keeps_source`:
                // Q5_K, Q8_0, and on a K-quant source conversion Q4_K and Q6_K).
                // Must mirror `write_layer_blob`'s None-target verbatim copy.
                let quant = src_quant.ok_or_else(|| ConvertError::UnsupportedTensorType {
                    tensor: name.clone(),
                    ggml_type: format!("{:?}", tensor.ggml_type),
                })?;
                let size =
                    tensor
                        .byte_size()
                        .ok_or_else(|| ConvertError::UnsupportedTensorType {
                            tensor: name.clone(),
                            ggml_type: format!("{:?} (unknown block geometry)", tensor.ggml_type),
                        })?;
                (size, quant)
            }
            _ => ((n_elements * 4) as u64, QuantScheme::F32),
        };
        let slice = TensorSlice {
            offset: *blob_offset,
            length: size,
            quant,
        };
        *blob_offset = blob_offset.saturating_add(size);
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
        let qkv_name = layer_tensor_name(layer, ATTN_QKV);
        wq = if gdn_pair_q8 {
            let t = gguf
                .find_tensor(&qkv_name)
                .ok_or_else(|| ConvertError::MissingTensor(qkv_name.clone()))?;
            super::gdn_gates::pair_forced_q8_slice(
                t.n_elements() as usize,
                &qkv_name,
                &mut blob_size,
            )?
        } else {
            compute_slice(gguf, &qkv_name, &mut blob_size, dequantize)?
        };
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

    // Attention gate — GDN layers on this architecture (full attention
    // fuses the gate into attn_q; the runtime still dispatches a separate
    // attn_gate on its own quant wherever one exists)
    let attn_gate = if gdn_pair_q8 {
        let name = layer_tensor_name(layer, ATTN_GATE_WEIGHT);
        match gguf.find_tensor(&name) {
            Some(t) => Some(super::gdn_gates::pair_forced_q8_slice(
                t.n_elements() as usize,
                &name,
                &mut blob_size,
            )?),
            None => None,
        }
    } else {
        try_compute_opt_slice(gguf, layer, ATTN_GATE_WEIGHT, &mut blob_size, dequantize)?
    };

    // SSM tensors (linear attention layers only) — never requantized to user target.
    // ssm_alpha/beta are Q8_0 on default paths (Metal's GDN kernels read only
    // Q8_0; CUDA also serves F32 gates). Shared logic in gdn_gates handles the
    // force-requant and the `--dequantize` / source-fidelity exceptions, and a
    // K-quant source's default non-Metal conversion, which keeps F32 gates of the
    // extent the projection reads.
    let ssm = compute_ssm_slices(gguf, layer, &mut blob_size, dequantize, target)?;
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
    // SOURCE_FIDELITY: keep ssm_out in its source format when the runtime can
    // serve it (`ssm_out_keeps_source`). The Q8_0 floor below guards the
    // historical hazard — REQUANTIZING ssm_out DOWN to 4-bit corrupts the
    // recurrence; serving the provider's own K-quant is the reference
    // engine's configuration, not a down-requant.
    let ssm_out_src = gguf
        .find_tensor(&layer_tensor_name(layer, SSM_OUT))
        .map(|t| t.ggml_type);
    let ssm_out_target =
        if ssm_out_keeps_source(target, ssm_out_src, gdn_v_dim(gguf), requant_to, dequantize) {
            None
        } else {
            match requant_to {
                Some(QuantScheme::Q4_0) => Some(QuantScheme::Q8_0),
                other => other.or(Some(QuantScheme::Q8_0)),
            }
        };
    let ssm_out = compute_slice_with_requant(gguf, layer, SSM_OUT, &mut blob_size, ssm_out_target)?;

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
    let gdn_pair_q8 = !is_full_attn
        && super::gdn_gates::metal_gdn_pair_forces_q8(gguf, layer, dequantize, requant_to, target);
    if gdn_pair_q8 {
        eprintln!(
            "    Metal GDN pair force: layer {layer} attn_qkv+attn_gate -> Q8_0 (mixed or F16 source)"
        );
    }

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
            if gdn_pair_q8 { false } else { dequantize },
            if gdn_pair_q8 {
                Some(QuantScheme::Q8_0)
            } else {
                requant_to
            },
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
            if gdn_pair_q8 { false } else { dequantize },
            if gdn_pair_q8 {
                Some(QuantScheme::Q8_0)
            } else {
                requant_to
            },
            target,
        )?;
    }

    // SSM tensors (if present) — shared GDN gate logic handles force-requant
    // of ssm_alpha/beta to Q8_0 when source is F32/F16/BF16.
    write_ssm_tensors(blob, reader, gguf, layer, dequantize, target)?;
    {
        let name = layer_tensor_name(layer, SSM_OUT);
        if gguf.find_tensor(&name).is_some() {
            // SSM_OUT: route through the runtime's fast Q8_0 GDN path.
            // FLOORED at Q8_0 even under `--requant q4_0` — 4-bit ssm_out
            // corrupts the GDN recurrence (2026-06-10 RCA; see the matching
            // floor + evidence in compute_slice_with_requant above; the two
            // MUST stay in sync for layer-shape symmetry). (Target is
            // irrelevant here: ssm_out is always force-requanted.)
            // SOURCE_FIDELITY: keep the source scheme verbatim (None target =
            // passthrough) — the same `ssm_out_keeps_source` the plan used.
            let src = gguf.find_tensor(&name).map(|t| t.ggml_type);
            let ssm_out_target =
                if ssm_out_keeps_source(target, src, gdn_v_dim(gguf), requant_to, dequantize) {
                    None
                } else {
                    match requant_to {
                        Some(QuantScheme::Q4_0) => Some(QuantScheme::Q8_0),
                        other => other.or(Some(QuantScheme::Q8_0)),
                    }
                };
            append_tensor_to_blob_requant(
                blob,
                reader,
                gguf,
                &name,
                /*dequantize=*/ false,
                ssm_out_target,
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
