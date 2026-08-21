//! HF checkpoint import: compressed-tensors pack-quantized INT4 g32 (the
//! [`QuantScheme::CtInt4G32`] dialect) → LBC, for dense Qwen3.5-family GDN
//! models.
//!
//! The HF checkpoint supplies ALL tensor data. A donor GGUF of the same
//! model supplies only tokenizer + hyperparameter metadata (the checkpoint
//! ships neither in a form the LBC pipeline consumes).
//!
//! Tensor-convention transforms applied while lowering HF tensors to the
//! LBC/GGUF layout the runtime expects (each pinned by exact byte comparison
//! of the base BF16 checkpoint against the reference GGUF conversion):
//!   - zero-centered RMSNorm weights (attn/post-attn/q/k/final): `w + 1`
//!   - GDN gated norm (`ssm_norm`): identity
//!   - `A_log` → `-exp(A_log)`
//!   - GDN v-head reorder `hf_head = ratio*(i % groups) + i/groups`
//!     (v-heads grouped by k-head in HF, round-robin in GGUF), applied to
//!     dt_bias, A_log, conv1d v-channels, `in_proj_qkv` v-rows, `in_proj_z`
//!     rows, `in_proj_a`/`in_proj_b` rows, and `out_proj` input columns
//!   - everything else: identity
//!
//! Quantized tensors keep their exact source planes (reindexed only where a
//! transform above demands it); floating-point tensors follow the LBC slot
//! conventions (norms/SSM scalars/alpha/beta as F32, projections as Bf16).
//!
//! [`QuantScheme::CtInt4G32`]: lumen_format::QuantScheme::CtInt4G32

use std::io::BufWriter;
use std::path::Path;

use crate::arch::qwen35_moe::is_qwen35moe_full_attention_layer;
use crate::convert::{ConvertError, ConvertStats};
use crate::ct_planes::{permute_k_blocks, permute_rows, permute_zero_point_rows};
use crate::dequant::convert_bf16_bytes_to_f32;
use crate::gguf::GgufFile;
use crate::hf_ct::{HfCtCheckpoint, HfDtype};
use crate::hyperparams::{extract_hyperparams, quant_descriptor_for};
use crate::sharded::ShardedGguf;
use lumen_format::index::{LayerIndex, SubtensorOffsets, TensorSlice};
use lumen_format::streaming_writer::{LayerShape, StreamingLbcWriter};
use lumen_format::tokenizer::TokenizerSection;
use lumen_format::writer::GlobalTensors;
use lumen_format::{CtInt4G32Planes, LbcHeader, QuantScheme};

/// GDN v-head reorder: output head `i` takes HF head `ratio*(i % groups) + i/groups`.
fn v_head_perm(num_v_heads: usize, num_k_heads: usize) -> Vec<usize> {
    let ratio = num_v_heads / num_k_heads;
    (0..num_v_heads)
        .map(|i| ratio * (i % num_k_heads) + i / num_k_heads)
        .collect()
}

/// Expand a per-head permutation to per-row (each head spans `rows_per_head`
/// consecutive rows), with `prefix_rows` identity rows in front.
fn expand_head_perm(head_perm: &[usize], rows_per_head: usize, prefix_rows: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..prefix_rows).collect();
    for &h in head_perm {
        let base = prefix_rows + h * rows_per_head;
        out.extend(base..base + rows_per_head);
    }
    out
}

struct Importer<'a> {
    ckpt: &'a HfCtCheckpoint,
    prefix: String,
    /// Per-head GDN reorder (v-heads).
    head_perm: Vec<usize>,
    num_v_heads: usize,
    num_k_heads: usize,
    gdn_head_dim: usize,
}

/// One lowered tensor: final LBC bytes + quant tag.
struct Lowered {
    bytes: Vec<u8>,
    quant: QuantScheme,
}

impl<'a> Importer<'a> {
    fn name(&self, layer: usize, suffix: &str) -> String {
        format!("{}layers.{layer}.{suffix}", self.prefix)
    }

    fn f32_le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Fetch an F32/BF16 tensor as f32 values, validating its logical shape
    /// exactly (an equal-element-count transposed tensor must not pass).
    fn fetch_f32(&self, name: &str, expect_shape: &[u64]) -> Result<Vec<f32>, ConvertError> {
        let info = self
            .ckpt
            .tensor_info(name)
            .ok_or_else(|| ConvertError::MissingTensor(name.to_owned()))?;
        if info.shape != expect_shape {
            return Err(ConvertError::TensorShapeMismatch {
                tensor: name.to_owned(),
                expected: format!("{expect_shape:?}"),
                got: format!("{:?}", info.shape),
            });
        }
        let bytes = self.ckpt.tensor_bytes(name)?;
        let f32_bytes = match info.dtype {
            HfDtype::F32 => bytes,
            HfDtype::Bf16 => convert_bf16_bytes_to_f32(&bytes),
            other => {
                return Err(ConvertError::UnsupportedTensorType {
                    tensor: name.to_owned(),
                    ggml_type: format!("{other:?} (expected F32/BF16)"),
                })
            }
        };
        Ok(f32_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Fetch the three CtInt4G32 planes of `{base}.weight_packed/…`,
    /// validated against the logical shape.
    fn fetch_planes(
        &self,
        base: &str,
    ) -> Result<Option<(Vec<u8>, Vec<u8>, Vec<u8>, usize, usize)>, ConvertError> {
        let packed_name = format!("{base}.weight_packed");
        if self.ckpt.tensor_info(&packed_name).is_none() {
            // No packed representation: the sibling planes must be absent
            // too, or the checkpoint is inconsistent (orphan planes would be
            // silently ignored otherwise).
            for suffix in ["weight_scale", "weight_zero_point", "weight_shape"] {
                let name = format!("{base}.{suffix}");
                if self.ckpt.tensor_info(&name).is_some() {
                    return Err(ConvertError::MissingTensor(format!(
                        "{packed_name} (checkpoint has {name} but no packed weight)"
                    )));
                }
            }
            return Ok(None);
        }
        // Exactly one representation per module: packed planes OR a plain
        // floating-point weight, never both.
        if self.ckpt.tensor_info(&format!("{base}.weight")).is_some() {
            return Err(ConvertError::UnsupportedTensorType {
                tensor: packed_name,
                ggml_type: "both packed and unpacked representations present".into(),
            });
        }
        let shape_name = format!("{base}.weight_shape");
        let shape_info = self
            .ckpt
            .tensor_info(&shape_name)
            .ok_or_else(|| ConvertError::MissingTensor(shape_name.clone()))?;
        if shape_info.dtype != HfDtype::I64 || shape_info.shape != [2] {
            return Err(ConvertError::UnsupportedTensorType {
                tensor: shape_name,
                ggml_type: format!(
                    "{:?} {:?} (expected I64 [2])",
                    shape_info.dtype, shape_info.shape
                ),
            });
        }
        let shape_bytes = self.ckpt.tensor_bytes(&shape_name)?;
        let dim = |i: usize| -> Result<usize, ConvertError> {
            let v = i64::from_le_bytes(shape_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            usize::try_from(v).ok().filter(|&d| d > 0).ok_or_else(|| {
                ConvertError::TensorShapeMismatch {
                    tensor: shape_name.clone(),
                    expected: "positive dimensions".into(),
                    got: format!("{v}"),
                }
            })
        };
        let (n, k) = (dim(0)?, dim(1)?);
        // Every plane must carry the exact dtype AND logical shape the layout
        // math assumes — a same-byte-count transposed or retyped plane would
        // otherwise be reinterpreted silently.
        let expect_plane =
            |suffix: &str, dtype: HfDtype, shape: [u64; 2]| -> Result<Vec<u8>, ConvertError> {
                let pname = format!("{base}.{suffix}");
                let info = self
                    .ckpt
                    .tensor_info(&pname)
                    .ok_or_else(|| ConvertError::MissingTensor(pname.clone()))?;
                if info.dtype != dtype || info.shape != shape {
                    return Err(ConvertError::UnsupportedTensorType {
                        tensor: pname,
                        ggml_type: format!(
                            "{:?} {:?} (expected {dtype:?} {shape:?})",
                            info.dtype, info.shape
                        ),
                    });
                }
                self.ckpt.tensor_bytes(&pname)
            };
        let groups = (k / 32) as u64;
        let planes =
            CtInt4G32Planes::for_shape(n as u64, k as u64).map_err(ConvertError::Format)?;
        let qweight = expect_plane("weight_packed", HfDtype::I32, [n as u64, k as u64 / 8])?;
        let scale = expect_plane("weight_scale", HfDtype::Bf16, [n as u64, groups])?;
        let zero = expect_plane(
            "weight_zero_point",
            HfDtype::I32,
            [(n as u64).div_ceil(8), groups],
        )?;
        for (plane, got, want) in [
            ("weight_packed", qweight.len() as u64, planes.qweight_bytes),
            ("weight_scale", scale.len() as u64, planes.scale_bytes),
            (
                "weight_zero_point",
                zero.len() as u64,
                planes.zero_point_bytes,
            ),
        ] {
            if got != want {
                return Err(ConvertError::TensorShapeMismatch {
                    tensor: format!("{base}.{plane}"),
                    expected: format!("{want} bytes for [{n}, {k}]"),
                    got: format!("{got} bytes"),
                });
            }
        }
        Ok(Some((qweight, scale, zero, n, k)))
    }

    /// Lower a Linear-module weight: CtInt4G32 planes when quantized, Bf16
    /// bytes otherwise. `row_perm` (logical output rows) is applied to
    /// either representation.
    fn lower_linear(
        &self,
        base: &str,
        expect_n: usize,
        expect_k: usize,
        row_perm: Option<&[usize]>,
    ) -> Result<Lowered, ConvertError> {
        if let Some((qweight, scale, zero, n, k)) = self.fetch_planes(base)? {
            if (n, k) != (expect_n, expect_k) {
                return Err(ConvertError::TensorShapeMismatch {
                    tensor: base.to_owned(),
                    expected: format!("[{expect_n}, {expect_k}]"),
                    got: format!("[{n}, {k}]"),
                });
            }
            let groups = k / 32;
            let (qweight, scale, zero) = match row_perm {
                Some(perm) => (
                    permute_rows(&qweight, k / 2, perm),
                    permute_rows(&scale, groups * 2, perm),
                    permute_zero_point_rows(&zero, n, groups, perm),
                ),
                None => (qweight, scale, zero),
            };
            let mut bytes = qweight;
            bytes.extend_from_slice(&scale);
            bytes.extend_from_slice(&zero);
            Ok(Lowered {
                bytes,
                quant: QuantScheme::CtInt4G32,
            })
        } else {
            let name = format!("{base}.weight");
            let info = self
                .ckpt
                .tensor_info(&name)
                .ok_or_else(|| ConvertError::MissingTensor(name.clone()))?;
            if info.dtype != HfDtype::Bf16 {
                return Err(ConvertError::UnsupportedTensorType {
                    tensor: name,
                    ggml_type: format!("{:?} (expected BF16 for unquantized Linear)", info.dtype),
                });
            }
            if info.shape != [expect_n as u64, expect_k as u64] {
                return Err(ConvertError::TensorShapeMismatch {
                    tensor: name,
                    expected: format!("[{expect_n}, {expect_k}]"),
                    got: format!("{:?}", info.shape),
                });
            }
            let bytes = self.ckpt.tensor_bytes(&name)?;
            let bytes = match row_perm {
                Some(perm) => permute_rows(&bytes, expect_k * 2, perm),
                None => bytes,
            };
            Ok(Lowered {
                bytes,
                quant: QuantScheme::Bf16,
            })
        }
    }

    /// Lower an F32 slot from an FP tensor, with optional value map and
    /// optional logical-row permutation (`row_len` f32 values per row).
    fn lower_f32(
        &self,
        name: &str,
        expect_shape: &[u64],
        map: impl Fn(f32) -> f32,
        row_perm: Option<(&[usize], usize)>,
    ) -> Result<Lowered, ConvertError> {
        let vals = self.fetch_f32(name, expect_shape)?;
        let vals: Vec<f32> = vals.into_iter().map(map).collect();
        let vals = match row_perm {
            Some((perm, row_len)) => {
                let mut out = Vec::with_capacity(vals.len());
                for &src in perm {
                    out.extend_from_slice(&vals[src * row_len..(src + 1) * row_len]);
                }
                out
            }
            None => vals,
        };
        Ok(Lowered {
            bytes: Self::f32_le(&vals),
            quant: QuantScheme::F32,
        })
    }

    /// Build one layer: subtensor slices in the exact `qwen35` blob order.
    /// With `blob = Some(..)` the tensor bytes are appended; with `None`
    /// only sizes are computed (the two passes share this single code path).
    fn build_layer(
        &self,
        layer: usize,
        hidden: usize,
        inter: usize,
        num_heads: usize,
        num_kv_heads: usize,
        attn_head_dim: usize,
        mut blob: Option<&mut Vec<u8>>,
    ) -> Result<LayerShape, ConvertError> {
        let is_full = is_qwen35moe_full_attention_layer(layer);
        let mut offset = 0u64;

        let mut push = |lowered: Lowered, blob: &mut Option<&mut Vec<u8>>| -> TensorSlice {
            let slice = TensorSlice {
                offset,
                length: lowered.bytes.len() as u64,
                quant: lowered.quant,
            };
            offset += slice.length;
            if let Some(b) = blob {
                b.extend_from_slice(&lowered.bytes);
            }
            slice
        };
        let zero = TensorSlice {
            offset: 0,
            length: 0,
            quant: QuantScheme::F32,
        };

        let qk_rows = self.num_k_heads * self.gdn_head_dim;
        let v_rows = self.num_v_heads * self.gdn_head_dim;
        let qkv_rows = 2 * qk_rows + v_rows;
        let qkv_row_perm = expand_head_perm(&self.head_perm, self.gdn_head_dim, 2 * qk_rows);
        let v_row_perm = expand_head_perm(&self.head_perm, self.gdn_head_dim, 0);

        // -- attention projections --
        let (wq, wk, wv, wo);
        if is_full {
            let q_rows = num_heads * attn_head_dim * 2; // Q+gate fused
            let kv_rows = num_kv_heads * attn_head_dim;
            let o_cols = num_heads * attn_head_dim;
            wq = push(
                self.lower_linear(&self.name(layer, "self_attn.q_proj"), q_rows, hidden, None)?,
                &mut blob,
            );
            wk = push(
                self.lower_linear(&self.name(layer, "self_attn.k_proj"), kv_rows, hidden, None)?,
                &mut blob,
            );
            wv = push(
                self.lower_linear(&self.name(layer, "self_attn.v_proj"), kv_rows, hidden, None)?,
                &mut blob,
            );
            wo = push(
                self.lower_linear(&self.name(layer, "self_attn.o_proj"), hidden, o_cols, None)?,
                &mut blob,
            );
        } else {
            wq = push(
                self.lower_linear(
                    &self.name(layer, "linear_attn.in_proj_qkv"),
                    qkv_rows,
                    hidden,
                    Some(&qkv_row_perm),
                )?,
                &mut blob,
            );
            wk = zero;
            wv = zero;
            wo = zero;
        }

        // -- norms (order: attn_norm, attn_post_norm; ffn_norm is absent) --
        let attn_norm = push(
            self.lower_f32(
                &self.name(layer, "input_layernorm.weight"),
                &[hidden as u64],
                |v| v + 1.0,
                None,
            )?,
            &mut blob,
        );
        let attn_post_norm = Some(push(
            self.lower_f32(
                &self.name(layer, "post_attention_layernorm.weight"),
                &[hidden as u64],
                |v| v + 1.0,
                None,
            )?,
            &mut blob,
        ));
        let ffn_norm = zero;

        // -- GDN tensors --
        let (attn_gate, ssm_a, ssm_conv1d, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out);
        if is_full {
            attn_gate = None;
            ssm_a = None;
            ssm_conv1d = None;
            ssm_dt = None;
            ssm_beta = None;
            ssm_alpha = None;
            ssm_norm = None;
            ssm_out = None;
        } else {
            attn_gate = Some(push(
                self.lower_linear(
                    &self.name(layer, "linear_attn.in_proj_z"),
                    v_rows,
                    hidden,
                    Some(&v_row_perm),
                )?,
                &mut blob,
            ));
            // order matches compute_ssm_slices: a, conv1d, dt, beta, alpha, norm
            ssm_a = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.A_log"),
                    &[self.num_v_heads as u64],
                    |v| -v.exp(),
                    Some((&self.head_perm, 1)),
                )?,
                &mut blob,
            ));
            ssm_conv1d = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.conv1d.weight"),
                    &[qkv_rows as u64, 1, 4],
                    |v| v,
                    Some((
                        &expand_head_perm(&self.head_perm, self.gdn_head_dim, 2 * qk_rows),
                        4,
                    )),
                )?,
                &mut blob,
            ));
            ssm_dt = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.dt_bias"),
                    &[self.num_v_heads as u64],
                    |v| v,
                    Some((&self.head_perm, 1)),
                )?,
                &mut blob,
            ));
            ssm_beta = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.in_proj_b.weight"),
                    &[self.num_v_heads as u64, hidden as u64],
                    |v| v,
                    Some((&self.head_perm, hidden)),
                )?,
                &mut blob,
            ));
            ssm_alpha = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.in_proj_a.weight"),
                    &[self.num_v_heads as u64, hidden as u64],
                    |v| v,
                    Some((&self.head_perm, hidden)),
                )?,
                &mut blob,
            ));
            ssm_norm = Some(push(
                self.lower_f32(
                    &self.name(layer, "linear_attn.norm.weight"),
                    &[self.gdn_head_dim as u64],
                    |v| v,
                    None,
                )?,
                &mut blob,
            ));
            // out_proj: [hidden, v_rows]; the v-head reorder permutes its
            // INPUT columns in head-sized blocks.
            let lowered = {
                let base = self.name(layer, "linear_attn.out_proj");
                match self.fetch_planes(&base)? {
                    Some((qweight, scale, zp, n, k)) => {
                        if (n, k) != (hidden, v_rows) {
                            return Err(ConvertError::TensorShapeMismatch {
                                tensor: base,
                                expected: format!("[{hidden}, {v_rows}]"),
                                got: format!("[{n}, {k}]"),
                            });
                        }
                        let (q, s, z) = permute_k_blocks(
                            &qweight,
                            &scale,
                            &zp,
                            n,
                            k,
                            self.gdn_head_dim,
                            &self.head_perm,
                        );
                        let mut bytes = q;
                        bytes.extend_from_slice(&s);
                        bytes.extend_from_slice(&z);
                        Lowered {
                            bytes,
                            quant: QuantScheme::CtInt4G32,
                        }
                    }
                    None => {
                        // Unquantized out_proj (e.g. layer 0): Bf16 with the
                        // column-block permutation applied per row.
                        let name = format!("{base}.weight");
                        let info = self
                            .ckpt
                            .tensor_info(&name)
                            .ok_or_else(|| ConvertError::MissingTensor(name.clone()))?;
                        if info.dtype != HfDtype::Bf16
                            || info.shape != [hidden as u64, v_rows as u64]
                        {
                            return Err(ConvertError::UnsupportedTensorType {
                                tensor: name,
                                ggml_type: format!(
                                    "{:?} {:?} (expected BF16 [{hidden}, {v_rows}])",
                                    info.dtype, info.shape
                                ),
                            });
                        }
                        let bytes = self.ckpt.tensor_bytes(&name)?;
                        let block = self.gdn_head_dim * 2;
                        let row = v_rows * 2;
                        let mut out = vec![0u8; bytes.len()];
                        for r in 0..hidden {
                            for (i, &src) in self.head_perm.iter().enumerate() {
                                out[r * row + i * block..r * row + (i + 1) * block]
                                    .copy_from_slice(
                                        &bytes[r * row + src * block..r * row + (src + 1) * block],
                                    );
                            }
                        }
                        Lowered {
                            bytes: out,
                            quant: QuantScheme::Bf16,
                        }
                    }
                }
            };
            ssm_out = Some(push(lowered, &mut blob));
        }

        // -- FFN --
        let w_gate = push(
            self.lower_linear(&self.name(layer, "mlp.gate_proj"), inter, hidden, None)?,
            &mut blob,
        );
        let w_up = push(
            self.lower_linear(&self.name(layer, "mlp.up_proj"), inter, hidden, None)?,
            &mut blob,
        );
        let w_down = push(
            self.lower_linear(&self.name(layer, "mlp.down_proj"), hidden, inter, None)?,
            &mut blob,
        );

        // -- per-head q/k norms (full attention only) --
        let (attn_q_norm, attn_k_norm) = if is_full {
            (
                Some(push(
                    self.lower_f32(
                        &self.name(layer, "self_attn.q_norm.weight"),
                        &[attn_head_dim as u64],
                        |v| v + 1.0,
                        None,
                    )?,
                    &mut blob,
                )),
                Some(push(
                    self.lower_f32(
                        &self.name(layer, "self_attn.k_norm.weight"),
                        &[attn_head_dim as u64],
                        |v| v + 1.0,
                        None,
                    )?,
                    &mut blob,
                )),
            )
        } else {
            (None, None)
        };

        let subtensors = SubtensorOffsets {
            wq,
            wk,
            wv,
            wo,
            bq: None,
            bk: None,
            bv: None,
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
            layer_type: Some(if is_full { 0u8 } else { 1u8 }),
        };
        Ok(LayerShape {
            blob_size: offset,
            index: LayerIndex {
                layer_offset_bytes: 0,
                layer_length_bytes: offset,
                subtensors,
            },
        })
    }
}

/// Convert an HF pack-quantized checkpoint directory to LBC, taking
/// tokenizer + hyperparameter metadata from `donor_gguf` (a GGUF of the
/// same model).
pub fn convert_hf_ct_to_lbc(
    hf_dir: &Path,
    donor_gguf: &Path,
    lbc_path: &Path,
) -> Result<ConvertStats, ConvertError> {
    let sharded = ShardedGguf::open(donor_gguf).map_err(|e| {
        ConvertError::UnsupportedArchitecture(format!("donor GGUF open failed: {e}"))
    })?;
    let gguf: &GgufFile = sharded.merged();
    let (hp, arch) = extract_hyperparams(gguf)?;
    if arch != "qwen35" || hp.num_experts.unwrap_or(0) > 0 {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "HF import supports dense qwen35 models only (donor arch: {arch})"
        )));
    }
    let gdn = hp.gdn.ok_or_else(|| {
        ConvertError::UnsupportedArchitecture("donor GGUF has no GDN metadata".into())
    })?;
    if gdn.head_dim == 0 || gdn.head_dim % 32 != 0 {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "GDN head_dim {} is not a positive multiple of the quantization group \
             size (32); the out_proj column reorder requires group-aligned heads",
            gdn.head_dim
        )));
    }
    // The v-head reorder is only a permutation for positive, divisible counts.
    if gdn.num_v_heads == 0 || gdn.num_k_heads == 0 || gdn.num_v_heads % gdn.num_k_heads != 0 {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "GDN head counts v={} k={} (need both > 0 and v divisible by k)",
            gdn.num_v_heads, gdn.num_k_heads
        )));
    }
    // The conv1d lowering fixes the kernel width at 4 taps; a donor declaring
    // anything else would make the runtime read taps the tensor doesn't have.
    if gdn.conv_kernel != 4 {
        return Err(ConvertError::UnsupportedArchitecture(format!(
            "GDN conv_kernel {} is unsupported (need 4)",
            gdn.conv_kernel
        )));
    }

    let ckpt = HfCtCheckpoint::open(hf_dir)?;
    // "Input" = the checkpoint carrying the weights, not the metadata donor.
    let input_size = ckpt.total_shard_bytes()?;
    let prefix = ["model.language_model.", "model."]
        .iter()
        .find(|p| {
            ckpt.tensor_info(&format!("{p}embed_tokens.weight"))
                .is_some()
        })
        .map(|p| (*p).to_owned())
        .ok_or_else(|| ConvertError::MissingTensor("embed_tokens.weight".into()))?;

    // Cross-check checkpoint dims against the donor's hyperparams.
    let tc = ckpt
        .config
        .get("text_config")
        .unwrap_or(&ckpt.config)
        .clone();
    for (key, want) in [
        ("hidden_size", hp.hidden_dim as u64),
        ("num_hidden_layers", hp.num_layers as u64),
        ("intermediate_size", hp.intermediate_dim as u64),
        ("vocab_size", hp.vocab_size as u64),
    ] {
        let got = tc.get(key).and_then(|v| v.as_u64());
        if got != Some(want) {
            return Err(ConvertError::UnsupportedArchitecture(format!(
                "checkpoint {key} = {got:?} does not match donor GGUF ({want})"
            )));
        }
    }
    // Secondary head-geometry keys: a mismatch is always fatal, but a key
    // absent from the checkpoint config is tolerated (naming varies across
    // config generations; the four required keys above pin the model family).
    // A key that is PRESENT with a non-integer value is rejected — treating
    // it as absent would fail open.
    for (key, want) in [
        ("num_attention_heads", u64::from(hp.num_heads)),
        ("num_key_value_heads", u64::from(hp.num_kv_heads)),
        ("linear_num_value_heads", u64::from(gdn.num_v_heads)),
        ("linear_num_key_heads", u64::from(gdn.num_k_heads)),
        ("linear_conv_kernel_dim", u64::from(gdn.conv_kernel)),
    ] {
        match tc.get(key) {
            None | Some(serde_json::Value::Null) => {}
            Some(v) => {
                let got = v.as_u64().ok_or_else(|| {
                    ConvertError::UnsupportedArchitecture(format!(
                        "checkpoint {key} = {v} is not an integer"
                    ))
                })?;
                if got != want {
                    return Err(ConvertError::UnsupportedArchitecture(format!(
                        "checkpoint {key} = {got} does not match donor GGUF ({want})"
                    )));
                }
            }
        }
    }
    // Numeric scalars that silently change model math if the donor is wrong:
    // compare when the checkpoint declares them (relative tolerance for
    // float representation differences).
    let donor_theta = hp.rope_params.map_or(10000.0, |rp| f64::from(rp.theta));
    for (key, want) in [
        ("rope_theta", donor_theta),
        ("rms_norm_eps", f64::from(hp.norm_eps)),
    ] {
        if let Some(got) = tc.get(key).and_then(|v| v.as_f64()) {
            let tol = want.abs().max(1e-12) * 1e-6;
            if (got - want).abs() > tol {
                return Err(ConvertError::UnsupportedArchitecture(format!(
                    "checkpoint {key} = {got} does not match donor GGUF ({want})"
                )));
            }
        }
    }

    let num_v_heads = gdn.num_v_heads as usize;
    let num_k_heads = gdn.num_k_heads as usize;
    if !ckpt.quant.ignore.is_empty() {
        eprintln!(
            "  Checkpoint keeps unquantized (compressed-tensors ignore): {}",
            ckpt.quant.ignore.join(", ")
        );
    }
    let importer = Importer {
        ckpt: &ckpt,
        prefix: prefix.clone(),
        head_perm: v_head_perm(num_v_heads, num_k_heads),
        num_v_heads,
        num_k_heads,
        gdn_head_dim: gdn.head_dim as usize,
    };

    let hidden = hp.hidden_dim as usize;
    let inter = hp.intermediate_dim as usize;
    let num_heads = hp.num_heads as usize;
    let num_kv_heads = hp.num_kv_heads as usize;
    // Full-attention head_dim comes from GGUF attention.key_length (may
    // differ from hidden/num_heads on this family).
    let attn_head_dim = hp.head_dim as usize;
    let num_layers = hp.num_layers as usize;

    // Pass 1: shapes.
    let mut layer_shapes = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        layer_shapes.push(importer.build_layer(
            layer,
            hidden,
            inter,
            num_heads,
            num_kv_heads,
            attn_head_dim,
            None,
        )?);
    }

    // Globals.
    let embedding_name = format!("{prefix}embed_tokens.weight");
    let einfo = ckpt
        .tensor_info(&embedding_name)
        .ok_or_else(|| ConvertError::MissingTensor(embedding_name.clone()))?;
    if einfo.dtype != HfDtype::Bf16 || einfo.shape != [hp.vocab_size as u64, hidden as u64] {
        return Err(ConvertError::UnsupportedTensorType {
            tensor: embedding_name.clone(),
            ggml_type: format!(
                "{:?} {:?} (expected BF16 [{}, {hidden}])",
                einfo.dtype, einfo.shape, hp.vocab_size
            ),
        });
    }
    let embedding = ckpt.tensor_bytes(&embedding_name)?;
    let final_norm_vals = importer.fetch_f32(&format!("{prefix}norm.weight"), &[hidden as u64])?;
    let final_norm = Importer::f32_le(&final_norm_vals.iter().map(|v| v + 1.0).collect::<Vec<_>>());
    let head = importer.lower_linear("lm_head", hp.vocab_size as usize, hidden, None)?;
    if head.quant != QuantScheme::Bf16 {
        // No runtime path serves a quantized global head; converting one
        // would produce an artifact that fails (or worse) at load time.
        return Err(ConvertError::UnsupportedArchitecture(
            "quantized lm_head is not supported — the checkpoint must keep \
             lm_head in BF16 (compressed-tensors `ignore` list)"
                .into(),
        ));
    }
    let (output_proj, output_proj_quant) = (head.bytes, head.quant);

    // The header's primary scheme advertises CtInt4G32; require that at
    // least one tensor actually carries it — an all-BF16 checkpoint that
    // merely retains a pack-quantized config must not be mislabeled.
    let has_ct4 = layer_shapes.iter().any(|ls| {
        let s = &ls.index.subtensors;
        [
            Some(&s.wq),
            Some(&s.wk),
            Some(&s.wv),
            Some(&s.wo),
            Some(&s.w_gate),
            Some(&s.w_up),
            Some(&s.w_down),
            s.attn_gate.as_ref(),
            s.ssm_out.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(|t| t.quant == QuantScheme::CtInt4G32)
    });
    if !has_ct4 {
        return Err(ConvertError::UnsupportedArchitecture(
            "checkpoint contains no pack-quantized tensors (nothing to import as CtInt4G32)".into(),
        ));
    }

    let qd = quant_descriptor_for(QuantScheme::CtInt4G32);
    let mut header = LbcHeader::new(hp, qd);
    header.embedding.quant = QuantScheme::Bf16;
    header.final_norm.quant = QuantScheme::F32;
    header.output_proj.quant = output_proj_quant;
    header.weight_tying = false;

    // The tokenizer is the donor's whole purpose — an LBC without one cannot
    // serve text prompts, so a donor we can't extract from is a hard error.
    let tokenizer_section = crate::tokenizer_data::extract_tokenizer(gguf).map(|td| {
        eprintln!(
            "  Tokenizer (donor GGUF): model={} vocab={} merges={}",
            td.model_type,
            td.tokens.len(),
            td.merges.len()
        );
        TokenizerSection {
            model_type: td.model_type,
            pre_tokenizer: td.pre_tokenizer,
            tokens: td.tokens,
            token_types: td.token_types,
            scores: td.scores,
            merges: td.merges,
            bos_token_id: td.bos_token_id,
            eos_token_id: td.eos_token_id,
            pad_token_id: td.pad_token_id,
            add_bos_token: td.add_bos_token,
            add_eos_token: td.add_eos_token,
            add_space_prefix: td.add_space_prefix,
            chat_template: td.chat_template,
        }
    });
    if tokenizer_section.is_none() {
        return Err(ConvertError::UnsupportedArchitecture(
            "donor GGUF has no extractable tokenizer (tokenizer.ggml.tokens missing)".into(),
        ));
    }

    // Write to a unique sibling temp file and rename into place at the end,
    // so a failed conversion never destroys an existing artifact and a
    // partial file never carries the final name. `create_new` refuses to
    // follow a pre-existing path (symlink or a concurrent conversion's
    // file); the guard removes the multi-GB partial on any error exit.
    struct TmpGuard(Option<std::path::PathBuf>);
    impl Drop for TmpGuard {
        fn drop(&mut self) {
            if let Some(p) = &self.0 {
                let _ = std::fs::remove_file(p);
            }
        }
    }
    let tmp_path = lbc_path.with_extension(format!("lbc.tmp.{}", std::process::id()));
    let output_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)?;
    let mut tmp_guard = TmpGuard(Some(tmp_path.clone()));
    let writer = BufWriter::with_capacity(8 * 1024 * 1024, output_file);
    let global_tensors = GlobalTensors {
        embedding,
        final_norm,
        output_proj,
    };
    let mut streaming = StreamingLbcWriter::begin(
        writer,
        &header,
        &layer_shapes,
        &global_tensors,
        tokenizer_section.as_ref(),
    )?;

    // Pass 2: layer data.
    let tensor_count = ckpt.tensor_names().count();
    for (layer, expected) in layer_shapes.iter().enumerate() {
        let mut blob = Vec::with_capacity(expected.blob_size as usize);
        let shape = importer.build_layer(
            layer,
            hidden,
            inter,
            num_heads,
            num_kv_heads,
            attn_head_dim,
            Some(&mut blob),
        )?;
        debug_assert_eq!(shape.blob_size, expected.blob_size);
        streaming.write_layer(&blob)?;
        eprintln!(
            "  Layer {}/{} ({:.1} MB, {})",
            layer + 1,
            num_layers,
            blob.len() as f64 / 1_048_576.0,
            if is_qwen35moe_full_attention_layer(layer) {
                "full-attn"
            } else {
                "linear-attn"
            },
        );
    }
    // Flush + sync explicitly: a drop-time flush error would otherwise be
    // silently ignored and the command could report success on a short file.
    let mut writer = streaming.finish()?;
    std::io::Write::flush(&mut writer)?;
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    std::fs::rename(&tmp_path, lbc_path)?;
    tmp_guard.0 = None;

    let output_size = std::fs::metadata(lbc_path)?.len();
    Ok(ConvertStats {
        input_size,
        output_size,
        num_layers: hp.num_layers,
        architecture: arch,
        tensor_count,
        quant_scheme: QuantScheme::CtInt4G32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ct_planes::{pack_nibbles, unpack_nibbles};
    use crate::gguf::GgufBuilder;
    use crate::hf_ct::test_fixture::{dialect_config, shard_bytes, write_checkpoint};
    use lumen_format::LbcFile;

    // Synthetic model: hidden 64, inter 96, vocab 48, 4 layers (layer 3 is
    // full attention). Attention: 4 heads / 2 kv, head_dim 8. GDN: 6 v-heads,
    // 2 k-heads, head_dim 32, conv 4 → qkv rows 2*64+192 = 320.
    const HID: usize = 64;
    const INTER: usize = 96;
    const VOCAB: usize = 48;
    const VH: usize = 6;
    const KH: usize = 2;
    const GHD: usize = 32;
    const QK_ROWS: usize = KH * GHD;
    const V_ROWS: usize = VH * GHD;
    const QKV_ROWS: usize = 2 * QK_ROWS + V_ROWS;

    fn rng(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn rand_bytes(len: usize, seed: &mut u64) -> Vec<u8> {
        (0..len).map(|_| rng(seed) as u8).collect()
    }

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        // Simple truncation is fine for fixtures — `rand_f32` only produces
        // values exactly representable in bf16, so no rounding ambiguity.
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn rand_f32(len: usize, seed: &mut u64) -> Vec<f32> {
        // Values exactly representable in bf16 (8-bit mantissa steps).
        (0..len)
            .map(|_| ((rng(seed) % 256) as f32 - 128.0) / 64.0)
            .collect()
    }

    struct Planes {
        q: Vec<u8>,
        s: Vec<u8>,
        z: Vec<u8>,
        n: usize,
        k: usize,
    }

    fn rand_planes(n: usize, k: usize, seed: &mut u64) -> Planes {
        let p = CtInt4G32Planes::for_shape(n as u64, k as u64).unwrap();
        Planes {
            q: rand_bytes(p.qweight_bytes as usize, seed),
            s: rand_bytes(p.scale_bytes as usize, seed),
            z: rand_bytes(p.zero_point_bytes as usize, seed),
            n,
            k,
        }
    }

    fn plane_entries<'a>(
        base: &str,
        p: &'a Planes,
        out: &mut Vec<(String, &'static str, Vec<u64>, &'a [u8])>,
    ) {
        let groups = (p.k / 32) as u64;
        out.push((
            format!("{base}.weight_packed"),
            "I32",
            vec![p.n as u64, (p.k / 8) as u64],
            &p.q,
        ));
        out.push((
            format!("{base}.weight_scale"),
            "BF16",
            vec![p.n as u64, groups],
            &p.s,
        ));
        out.push((
            format!("{base}.weight_zero_point"),
            "I32",
            vec![(p.n as u64).div_ceil(8), groups],
            &p.z,
        ));
        let shape: &'static [u8] = Box::leak(
            [p.n as u64, p.k as u64]
                .iter()
                .flat_map(|v| (*v as i64).to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        out.push((format!("{base}.weight_shape"), "I64", vec![2], shape));
    }

    /// Naive value-level row permutation of packed planes (independent of
    /// the production `ct_planes` implementations).
    fn naive_row_perm(p: &Planes, perm: &[usize]) -> Vec<u8> {
        let groups = p.k / 32;
        let qvals = unpack_nibbles(&p.q, p.n * p.k);
        let zvals: Vec<u8> = {
            // zp packed along n per group column: word row wr, col g
            let word_rows = p.n.div_ceil(8);
            let mut v = vec![0u8; p.n * groups];
            for wr in 0..word_rows {
                for g in 0..groups {
                    let off = (wr * groups + g) * 4;
                    let w =
                        u32::from_le_bytes([p.z[off], p.z[off + 1], p.z[off + 2], p.z[off + 3]]);
                    for j in 0..8 {
                        if wr * 8 + j < p.n {
                            v[(wr * 8 + j) * groups + g] = ((w >> (4 * j)) & 0xF) as u8;
                        }
                    }
                }
            }
            v
        };
        let mut out = Vec::new();
        // qweight rows
        let mut qperm = Vec::with_capacity(p.n * p.k);
        for &src in perm {
            qperm.extend_from_slice(&qvals[src * p.k..(src + 1) * p.k]);
        }
        out.extend_from_slice(&pack_nibbles(&qperm));
        // scale rows
        for &src in perm {
            out.extend_from_slice(&p.s[src * groups * 2..(src + 1) * groups * 2]);
        }
        // zp rows, re-packed
        let word_rows = p.n.div_ceil(8);
        for wr in 0..word_rows {
            for g in 0..groups {
                let mut w = 0u32;
                for j in 0..8 {
                    if wr * 8 + j < p.n {
                        w |= u32::from(zvals[perm[wr * 8 + j] * groups + g]) << (4 * j);
                    }
                }
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Naive value-level K-block permutation (blocks of `block` columns).
    fn naive_k_perm(p: &Planes, block: usize, perm: &[usize]) -> Vec<u8> {
        let groups = p.k / 32;
        let qvals = unpack_nibbles(&p.q, p.n * p.k);
        let mut qperm = Vec::with_capacity(p.n * p.k);
        for row in 0..p.n {
            for &src in perm {
                qperm.extend_from_slice(
                    &qvals[row * p.k + src * block..row * p.k + (src + 1) * block],
                );
            }
        }
        let mut out = pack_nibbles(&qperm);
        let gpb = block / 32; // groups per block
        for row in 0..p.n {
            for &src in perm {
                out.extend_from_slice(
                    &p.s[(row * groups + src * gpb) * 2..(row * groups + (src + 1) * gpb) * 2],
                );
            }
        }
        let word_rows = p.n.div_ceil(8);
        for wr in 0..word_rows {
            for &src in perm {
                out.extend_from_slice(
                    &p.z[(wr * groups + src * gpb) * 4..(wr * groups + (src + 1) * gpb) * 4],
                );
            }
        }
        out
    }

    fn f32_slice_of(file: &[u8], base: u64, s: &TensorSlice) -> Vec<f32> {
        let b = (base + s.offset) as usize;
        file[b..b + s.length as usize]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn raw_slice_of(file: &[u8], base: u64, s: &TensorSlice) -> Vec<u8> {
        let b = (base + s.offset) as usize;
        file[b..b + s.length as usize].to_vec()
    }

    #[test]
    fn end_to_end_synthetic_conversion_verified_bytewise() {
        let dir = std::env::temp_dir().join(format!("lumen-hfconv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // ---- donor GGUF (metadata + embedding for vocab discovery) ----
        let mut b = GgufBuilder::new();
        b.add_string("general.architecture", "qwen35");
        b.add_u32("qwen35.block_count", 4);
        b.add_u32("qwen35.attention.head_count", 4);
        b.add_u32("qwen35.attention.head_count_kv", 2);
        b.add_u32("qwen35.attention.key_length", 8);
        b.add_u32("qwen35.embedding_length", HID as u32);
        b.add_u32("qwen35.feed_forward_length", INTER as u32);
        b.add_u32("qwen35.context_length", 64);
        b.add_f32("qwen35.rope.freq_base", 10000.0);
        b.add_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-5);
        b.add_u32("qwen35.ssm.time_step_rank", VH as u32);
        b.add_u32("qwen35.ssm.group_count", KH as u32);
        b.add_u32("qwen35.ssm.state_size", GHD as u32);
        b.add_u32("qwen35.ssm.conv_kernel", 4);
        let token_names: Vec<String> = (0..VOCAB).map(|i| format!("t{i}")).collect();
        let token_refs: Vec<&str> = token_names.iter().map(String::as_str).collect();
        b.add_string_array("tokenizer.ggml.tokens", &token_refs);
        b.add_f32_tensor(
            "token_embd.weight",
            &[VOCAB as u64, HID as u64],
            &vec![0.0; VOCAB * HID],
        );
        let donor_path = dir.join("donor.gguf");
        std::fs::write(&donor_path, b.build()).unwrap();

        // ---- synthetic HF checkpoint ----
        let mut seed = 7u64;
        let pfx = "model.language_model.";
        let mut planes_store: Vec<(String, Planes)> = Vec::new();
        let mut fp_store: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
        let mut fp32_vals: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        let add_fp_bf16 =
            |name: String,
             shape: Vec<u64>,
             seed: &mut u64,
             fp_store: &mut Vec<(String, Vec<u64>, Vec<u8>)>,
             fp32_vals: &mut std::collections::HashMap<String, Vec<f32>>| {
                let n: usize = shape.iter().product::<u64>() as usize;
                let vals = rand_f32(n, seed);
                fp_store.push((name.clone(), shape, bf16_bytes(&vals)));
                fp32_vals.insert(name, vals);
            };

        // globals
        add_fp_bf16(
            format!("{pfx}embed_tokens.weight"),
            vec![VOCAB as u64, HID as u64],
            &mut seed,
            &mut fp_store,
            &mut fp32_vals,
        );
        add_fp_bf16(
            format!("{pfx}norm.weight"),
            vec![HID as u64],
            &mut seed,
            &mut fp_store,
            &mut fp32_vals,
        );
        add_fp_bf16(
            "lm_head.weight".to_string(),
            vec![VOCAB as u64, HID as u64],
            &mut seed,
            &mut fp_store,
            &mut fp32_vals,
        );

        for layer in 0..4usize {
            let l = |s: &str| format!("{pfx}layers.{layer}.{s}");
            add_fp_bf16(
                l("input_layernorm.weight"),
                vec![HID as u64],
                &mut seed,
                &mut fp_store,
                &mut fp32_vals,
            );
            add_fp_bf16(
                l("post_attention_layernorm.weight"),
                vec![HID as u64],
                &mut seed,
                &mut fp_store,
                &mut fp32_vals,
            );
            planes_store.push((l("mlp.gate_proj"), rand_planes(INTER, HID, &mut seed)));
            planes_store.push((l("mlp.up_proj"), rand_planes(INTER, HID, &mut seed)));
            planes_store.push((l("mlp.down_proj"), rand_planes(HID, INTER, &mut seed)));
            if layer == 3 {
                planes_store.push((
                    l("self_attn.q_proj"),
                    rand_planes(4 * 8 * 2, HID, &mut seed),
                ));
                planes_store.push((l("self_attn.k_proj"), rand_planes(2 * 8, HID, &mut seed)));
                planes_store.push((l("self_attn.v_proj"), rand_planes(2 * 8, HID, &mut seed)));
                planes_store.push((l("self_attn.o_proj"), rand_planes(HID, 4 * 8, &mut seed)));
                add_fp_bf16(
                    l("self_attn.q_norm.weight"),
                    vec![8],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("self_attn.k_norm.weight"),
                    vec![8],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
            } else {
                planes_store.push((
                    l("linear_attn.in_proj_qkv"),
                    rand_planes(QKV_ROWS, HID, &mut seed),
                ));
                planes_store.push((
                    l("linear_attn.in_proj_z"),
                    rand_planes(V_ROWS, HID, &mut seed),
                ));
                if layer == 0 {
                    add_fp_bf16(
                        l("linear_attn.out_proj.weight"),
                        vec![HID as u64, V_ROWS as u64],
                        &mut seed,
                        &mut fp_store,
                        &mut fp32_vals,
                    );
                } else {
                    planes_store.push((
                        l("linear_attn.out_proj"),
                        rand_planes(HID, V_ROWS, &mut seed),
                    ));
                }
                add_fp_bf16(
                    l("linear_attn.A_log"),
                    vec![VH as u64],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("linear_attn.dt_bias"),
                    vec![VH as u64],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("linear_attn.conv1d.weight"),
                    vec![QKV_ROWS as u64, 1, 4],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("linear_attn.in_proj_a.weight"),
                    vec![VH as u64, HID as u64],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("linear_attn.in_proj_b.weight"),
                    vec![VH as u64, HID as u64],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
                add_fp_bf16(
                    l("linear_attn.norm.weight"),
                    vec![GHD as u64],
                    &mut seed,
                    &mut fp_store,
                    &mut fp32_vals,
                );
            }
        }

        let mut entries: Vec<(String, &'static str, Vec<u64>, &[u8])> = Vec::new();
        for (base, p) in &planes_store {
            plane_entries(base, p, &mut entries);
        }
        for (name, shape, bytes) in &fp_store {
            entries.push((name.clone(), "BF16", shape.clone(), bytes));
        }
        let entry_refs: Vec<(&str, &str, &[u64], &[u8])> = entries
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), *b))
            .collect();
        let shard = shard_bytes(&entry_refs);
        let mut cfg = dialect_config();
        cfg["text_config"] = serde_json::json!({
            "hidden_size": HID, "num_hidden_layers": 4,
            "intermediate_size": INTER, "vocab_size": VOCAB,
        });
        let wm: Vec<(&str, &str)> = entries
            .iter()
            .map(|(n, _, _, _)| (n.as_str(), "model-00001.safetensors"))
            .collect();
        let hf_dir = dir.join("ckpt");
        write_checkpoint(&hf_dir, &cfg, &[("model-00001.safetensors", shard)], &wm);

        // ---- convert ----
        let lbc_path = dir.join("out.lbc");
        let stats = convert_hf_ct_to_lbc(&hf_dir, &donor_path, &lbc_path).unwrap();
        assert_eq!(stats.quant_scheme, QuantScheme::CtInt4G32);

        // ---- verify ----
        let lbc = LbcFile::open(&lbc_path).unwrap();
        let file = std::fs::read(&lbc_path).unwrap();
        let planes: std::collections::HashMap<&str, &Planes> =
            planes_store.iter().map(|(n, p)| (n.as_str(), p)).collect();
        let perm = v_head_perm(VH, KH);
        let qkv_row_perm = expand_head_perm(&perm, GHD, 2 * QK_ROWS);
        let v_row_perm = expand_head_perm(&perm, GHD, 0);

        assert_eq!(lbc.header.quantization.scheme, QuantScheme::CtInt4G32);
        assert_eq!(lbc.header.embedding.quant, QuantScheme::Bf16);
        assert_eq!(lbc.header.output_proj.quant, QuantScheme::Bf16); // lm_head ignored

        // globals: embedding identity, final_norm +1
        let emb = &file[lbc.header.embedding.offset as usize
            ..(lbc.header.embedding.offset + lbc.header.embedding.length) as usize];
        assert_eq!(
            emb,
            bf16_bytes(&fp32_vals[&format!("{pfx}embed_tokens.weight")])
        );
        let fnorm = &file[lbc.header.final_norm.offset as usize
            ..(lbc.header.final_norm.offset + lbc.header.final_norm.length) as usize];
        let expect_fnorm: Vec<u8> = fp32_vals[&format!("{pfx}norm.weight")]
            .iter()
            .flat_map(|v| (v + 1.0).to_le_bytes())
            .collect();
        assert_eq!(fnorm, expect_fnorm.as_slice());

        for layer in 0..4usize {
            let idx = &lbc.layer_indices[layer];
            let base = idx.layer_offset_bytes;
            let st = &idx.subtensors;
            let l = |s: &str| format!("{pfx}layers.{layer}.{s}");

            // norms: +1
            let an = f32_slice_of(&file, base, &st.attn_norm);
            let expect: Vec<f32> = fp32_vals[&l("input_layernorm.weight")]
                .iter()
                .map(|v| v + 1.0)
                .collect();
            assert_eq!(an, expect, "layer {layer} attn_norm");

            // FFN gate identity planes
            let g = raw_slice_of(&file, base, &st.w_gate);
            let p = planes[l("mlp.gate_proj").as_str()];
            let mut expect_g = p.q.clone();
            expect_g.extend_from_slice(&p.s);
            expect_g.extend_from_slice(&p.z);
            assert_eq!(g, expect_g, "layer {layer} w_gate");
            assert_eq!(st.w_gate.quant, QuantScheme::CtInt4G32);

            if layer == 3 {
                assert_eq!(st.layer_type, Some(0));
                let q = raw_slice_of(&file, base, &st.wq);
                let p = planes[l("self_attn.q_proj").as_str()];
                let mut expect_q = p.q.clone();
                expect_q.extend_from_slice(&p.s);
                expect_q.extend_from_slice(&p.z);
                assert_eq!(q, expect_q, "full-attn wq identity");
                let qn = f32_slice_of(&file, base, st.attn_q_norm.as_ref().unwrap());
                let expect: Vec<f32> = fp32_vals[&l("self_attn.q_norm.weight")]
                    .iter()
                    .map(|v| v + 1.0)
                    .collect();
                assert_eq!(qn, expect, "q_norm +1");
            } else {
                assert_eq!(st.layer_type, Some(1));
                // qkv: v-row permutation (naive expected)
                let q = raw_slice_of(&file, base, &st.wq);
                let p = planes[l("linear_attn.in_proj_qkv").as_str()];
                assert_eq!(
                    q,
                    naive_row_perm(p, &qkv_row_perm),
                    "layer {layer} qkv perm"
                );
                // z-gate rows
                let z = raw_slice_of(&file, base, st.attn_gate.as_ref().unwrap());
                let p = planes[l("linear_attn.in_proj_z").as_str()];
                assert_eq!(z, naive_row_perm(p, &v_row_perm), "layer {layer} z perm");
                // ssm_a: -exp + head perm
                let a = f32_slice_of(&file, base, st.ssm_a.as_ref().unwrap());
                let src = &fp32_vals[&l("linear_attn.A_log")];
                let expect: Vec<f32> = perm.iter().map(|&h| -src[h].exp()).collect();
                assert_eq!(a, expect, "layer {layer} ssm_a");
                // dt: head perm
                let dt = f32_slice_of(&file, base, st.ssm_dt.as_ref().unwrap());
                let src = &fp32_vals[&l("linear_attn.dt_bias")];
                let expect: Vec<f32> = perm.iter().map(|&h| src[h]).collect();
                assert_eq!(dt, expect, "layer {layer} ssm_dt");
                // conv: v-channel perm, rows of 4 taps
                let cv = f32_slice_of(&file, base, st.ssm_conv1d.as_ref().unwrap());
                let src = &fp32_vals[&l("linear_attn.conv1d.weight")];
                let ch_perm = expand_head_perm(&perm, GHD, 2 * QK_ROWS);
                let expect: Vec<f32> = ch_perm
                    .iter()
                    .flat_map(|&c| src[c * 4..(c + 1) * 4].to_vec())
                    .collect();
                assert_eq!(cv, expect, "layer {layer} conv perm");
                // alpha: head-row perm (rows of HID)
                let al = f32_slice_of(&file, base, st.ssm_alpha.as_ref().unwrap());
                let src = &fp32_vals[&l("linear_attn.in_proj_a.weight")];
                let expect: Vec<f32> = perm
                    .iter()
                    .flat_map(|&h| src[h * HID..(h + 1) * HID].to_vec())
                    .collect();
                assert_eq!(al, expect, "layer {layer} alpha perm");
                // ssm_norm identity
                let sn = f32_slice_of(&file, base, st.ssm_norm.as_ref().unwrap());
                assert_eq!(
                    sn,
                    fp32_vals[&l("linear_attn.norm.weight")],
                    "ssm_norm identity"
                );
                // out_proj: K-block perm (quantized on layers 1-2, Bf16 on 0)
                let op = st.ssm_out.as_ref().unwrap();
                let bytes = raw_slice_of(&file, base, op);
                if layer == 0 {
                    assert_eq!(op.quant, QuantScheme::Bf16);
                    let src = &fp32_vals[&l("linear_attn.out_proj.weight")];
                    let mut expect: Vec<f32> = Vec::new();
                    for r in 0..HID {
                        for &h in &perm {
                            expect.extend_from_slice(
                                &src[r * V_ROWS + h * GHD..r * V_ROWS + (h + 1) * GHD],
                            );
                        }
                    }
                    assert_eq!(bytes, bf16_bytes(&expect), "layer 0 out_proj bf16 col perm");
                } else {
                    assert_eq!(op.quant, QuantScheme::CtInt4G32);
                    let p = planes[l("linear_attn.out_proj").as_str()];
                    assert_eq!(
                        bytes,
                        naive_k_perm(p, GHD, &perm),
                        "layer {layer} out_proj K perm"
                    );
                }
            }
        }
    }
}
