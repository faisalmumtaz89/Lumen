//! The native NVFP4/FP8 prefill route: its exact structural admission and the conditions of its
//! publication.
//!
//! The route's kernels and GEMM plans are specialised to one verified model structure, so a model is
//! prefilled by it only when every condition below holds; otherwise the model is prefilled by the
//! F32 route, and for an artifact with NVFP4 or FP8 planes the load log names the first condition
//! that failed. The route is chosen once, when the weights are
//! loaded (`CudaBackend::preload_weights`); nothing falls back during a request.
//!
//! The structural conditions are pure host functions of the hyperparameters, the layer descriptors,
//! the slice schemes and lengths, and the few values read through [`SliceSource`] (the activation
//! scales and the `in_proj_a` / `in_proj_b` values):
//!
//! | Condition | Requirement |
//! |---|---|
//! | Q0 | Dense hybrid of 64 layers: full attention exactly at layers `l % 4 == 3`, GDN elsewhere; no MoE tensor |
//! | Q0.a | Hidden size 5120, MLP size 17408 |
//! | Q0.b | 24 query heads, 4 KV heads, head size 256; query and gate fused with per-head q/k norms; no q/k/v bias; RoPE over 64 dimensions, NeoX pairing, theta 1e7, no scaling |
//! | Q0.c | GDN: 48 value heads, 16 key heads, head size 128, conv kernel 4; its F32 tensors present at exact lengths |
//! | Q0.d | BF16 embedding, resident on the device; every layer norm F32 at its exact length |
//! | Q5 | Every MLP projection NVFP4, every attention and GDN projection FP8, each carrying an activation scale that is finite and positive, and each resident on the device as its stored planes; so every scale group (GDN qkv and z, attention q, k and v, MLP gate and up) is well formed, its scale being its members' maximum |
//! | Q6 | `in_proj_a` / `in_proj_b` hold only BF16 values, so their BF16 copy is exact |
//!
//! The runtime conditions are checked when the route is published:
//!
//! | Condition | Requirement |
//! |---|---|
//! | Q1 | The device has compute capability 12.0 |
//! | Q2 | NVRTC 12.8 or newer, listing target 120 (`compute_120a`) |
//! | Q3 | Every native kernel group compiles and reproduces its qualifying outputs, and the embedding gather of the vocabulary's highest id and of id 65537 (when the vocabulary has one) returns the resident embedding's stored rows |
//! | Q4 | cuBLASLt 12.8 or newer loads, and every GEMM plan has a measured, verified algorithm |
//! | Q7 | The KV store is F32 or BF16 |
//! | Q8 | The weight views, the scratch, the KV staging (an F32 store's only), the RoPE table and the shared GDN state allocate |
//! | Q9 | `LUMEN_CUDA_NATIVE_PREFILL` is not `0` and `LUMEN_CUDA_PREFILL_F32` is unset |

use crate::error::RuntimeError;
use crate::weight::cache::{LayerView, WeightProvider};
use lumen_format::hyperparams::{GdnDims, RopeParams, RopeScalingType};
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_format::{Fp8Planes, ModelHyperparams, Nvfp4Planes, QuantScheme};

pub const LAYERS: u32 = 64;
pub const HIDDEN: u32 = 5120;
pub const INTERMEDIATE: u32 = 17408;
pub const HEADS: u32 = 24;
pub const KV_HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 256;
pub const ROTARY_DIM: u32 = 64;
pub const ROPE: RopeParams = RopeParams {
    theta: 1.0e7,
    scaling_factor: 1.0,
    scaling_type: RopeScalingType::None,
};
pub const GDN: GdnDims = GdnDims {
    num_v_heads: 48,
    num_k_heads: 16,
    head_dim: 128,
    conv_kernel: 4,
};

/// Whether layer `l` of the admitted structure is a full-attention layer (the others are GDN).
pub fn is_attention_layer(l: usize) -> bool {
    l % 4 == 3
}

/// Reads bytes of a layer's slice: `len` bytes from `start` within the slice.
pub trait SliceSource {
    fn read(
        &self,
        layer: usize,
        slice: &TensorSlice,
        start: u64,
        len: u64,
    ) -> Result<Vec<u8>, RuntimeError>;
}

/// A [`SliceSource`] over a weight provider. It keeps the last layer it fetched, and only that one,
/// because a provider fetch reads the whole layer and admission and the views read each layer's
/// slices together.
pub struct ProviderSlices<'a> {
    provider: &'a dyn WeightProvider,
    layer: std::cell::RefCell<Option<LayerView>>,
}

impl<'a> ProviderSlices<'a> {
    pub fn new(provider: &'a dyn WeightProvider) -> Self {
        Self {
            provider,
            layer: std::cell::RefCell::new(None),
        }
    }
}

impl ProviderSlices<'_> {
    /// Run `f` on layer `layer`'s view, fetching it unless it is the one held.
    fn with_layer<R>(
        &self,
        layer: usize,
        f: impl FnOnce(&LayerView) -> Result<R, RuntimeError>,
    ) -> Result<R, RuntimeError> {
        let mut held = self.layer.borrow_mut();
        if held.as_ref().map_or(true, |v| v.layer_idx != layer) {
            // Release the held layer before fetching the next: one layer is held at a time.
            *held = None;
            *held = Some(self.provider.get_layer_raw(layer)?);
        }
        f(held.as_ref().expect("fetched above"))
    }

    /// Layer `layer`'s descriptor, from the fetch its slices are then read from.
    pub fn subtensors(&self, layer: usize) -> Result<SubtensorOffsets, RuntimeError> {
        self.with_layer(layer, |v| Ok(v.subtensors.clone()))
    }
}

impl SliceSource for ProviderSlices<'_> {
    fn read(
        &self,
        layer: usize,
        slice: &TensorSlice,
        start: u64,
        len: u64,
    ) -> Result<Vec<u8>, RuntimeError> {
        self.with_layer(layer, |view| {
            let bytes = view.subtensor_bytes(slice)?;
            let end = start
                .checked_add(len)
                .filter(|&e| e <= bytes.len() as u64)
                .ok_or_else(|| {
                    RuntimeError::Compute(format!(
                        "layer {layer}: bytes {start}+{len} lie outside a slice of {}",
                        bytes.len()
                    ))
                })?;
            Ok(bytes[start as usize..end as usize].to_vec())
        })
    }
}

/// The first admission condition a model fails, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub condition: &'static str,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.condition, self.reason)
    }
}

fn refuse(condition: &'static str, reason: String) -> Result<(), Refusal> {
    Err(Refusal { condition, reason })
}

/// What one slice field of a layer must be.
#[derive(Clone, Copy)]
enum Expect {
    /// An optional field `None`, or a mandatory one with length zero.
    Absent,
    /// F32, this many elements.
    F32(u64),
    /// Planar weights `[n][k]` of this scheme followed by an activation scale.
    Planar(QuantScheme, u64, u64),
}

/// The expected content and the condition it belongs to, for every slice field of a layer.
fn expected(field: &str, attention: bool) -> (Expect, &'static str) {
    let h = HIDDEN as u64;
    let i = INTERMEDIATE as u64;
    let gdn_v = (GDN.num_v_heads * GDN.head_dim) as u64;
    let gdn_qkv = 2 * (GDN.num_k_heads * GDN.head_dim) as u64 + gdn_v;
    let q_rows = 2 * (HEADS * HEAD_DIM) as u64;
    let kv_rows = (KV_HEADS * HEAD_DIM) as u64;
    let o_cols = (HEADS * HEAD_DIM) as u64;
    let v_heads = GDN.num_v_heads as u64;
    use Expect::*;
    use QuantScheme::{Fp8E4M3, Nvfp4};
    match (field, attention) {
        ("w_gate" | "w_up", _) => (Planar(Nvfp4, i, h), "Q5"),
        ("w_down", _) => (Planar(Nvfp4, h, i), "Q5"),
        ("attn_norm" | "attn_post_norm", _) => (F32(h), "Q0.d"),
        ("ffn_norm", _) => (Absent, "Q0.d"),
        ("bq" | "bk" | "bv", _) => (Absent, "Q0.b"),
        ("wq", true) => (Planar(Fp8E4M3, q_rows, h), "Q5"),
        ("wk" | "wv", true) => (Planar(Fp8E4M3, kv_rows, h), "Q5"),
        ("wo", true) => (Planar(Fp8E4M3, h, o_cols), "Q5"),
        ("attn_q_norm" | "attn_k_norm", true) => (F32(HEAD_DIM as u64), "Q0.b"),
        ("wq", false) => (Planar(Fp8E4M3, gdn_qkv, h), "Q5"),
        ("attn_gate", false) => (Planar(Fp8E4M3, gdn_v, h), "Q5"),
        ("ssm_out", false) => (Planar(Fp8E4M3, h, gdn_v), "Q5"),
        ("ssm_a" | "ssm_dt", false) => (F32(v_heads), "Q0.c"),
        ("ssm_conv1d", false) => (F32(gdn_qkv * GDN.conv_kernel as u64), "Q0.c"),
        ("ssm_alpha" | "ssm_beta", false) => (F32(v_heads * h), "Q0.c"),
        ("ssm_norm", false) => (F32(GDN.head_dim as u64), "Q0.c"),
        ("wk" | "wv" | "wo" | "attn_gate" | "ssm_out", _) => (Absent, "Q5"),
        ("attn_q_norm" | "attn_k_norm", _) => (Absent, "Q0.b"),
        (f, _) if f.starts_with("ssm_") => (Absent, "Q0.c"),
        // router_weight, shared_expert_*, ffn_gate_inp_shexp
        _ => (Absent, "Q0"),
    }
}

/// The conditions of [`admit`] on the hyperparameters and the embedding's scheme alone, which read no
/// layer: a model that fails them is refused before any layer is read.
pub fn admit_structure(hp: &ModelHyperparams, embedding: QuantScheme) -> Result<(), Refusal> {
    if hp.num_layers != LAYERS {
        return refuse(
            "Q0",
            format!("{} layers; the signature has {LAYERS}", hp.num_layers),
        );
    }
    if hp.num_experts.is_some() || hp.num_active_experts.is_some() {
        return refuse("Q0", "the model has experts".into());
    }
    if (hp.hidden_dim, hp.intermediate_dim) != (HIDDEN, INTERMEDIATE) {
        return refuse(
            "Q0.a",
            format!(
                "hidden {} and MLP {}; the signature has {HIDDEN} and {INTERMEDIATE}",
                hp.hidden_dim, hp.intermediate_dim
            ),
        );
    }
    if (hp.num_heads, hp.num_kv_heads, hp.head_dim) != (HEADS, KV_HEADS, HEAD_DIM) {
        return refuse(
            "Q0.b",
            format!(
                "{} query heads, {} KV heads, head size {}; the signature has {HEADS}, {KV_HEADS}, \
                 {HEAD_DIM}",
                hp.num_heads, hp.num_kv_heads, hp.head_dim
            ),
        );
    }
    if hp.rotary_dim != Some(ROTARY_DIM) || !hp.rope_neox || hp.rope_params != Some(ROPE) {
        return refuse(
            "Q0.b",
            format!(
                "RoPE over {:?} dimensions, NeoX {}, {:?}; the signature has Some({ROTARY_DIM}), \
                 true, {:?}",
                hp.rotary_dim,
                hp.rope_neox,
                hp.rope_params,
                Some(ROPE)
            ),
        );
    }
    if hp.gdn != Some(GDN) {
        return refuse(
            "Q0.c",
            format!("GDN dimensions {:?}; the signature has {:?}", hp.gdn, GDN),
        );
    }
    if embedding != QuantScheme::Bf16 {
        return refuse(
            "Q0.d",
            format!("the embedding is {embedding:?}; the signature has Bf16"),
        );
    }
    Ok(())
}

/// Admit the model to the native route, or name the first condition it fails.
pub fn admit(
    hp: &ModelHyperparams,
    embedding: QuantScheme,
    layers: &[SubtensorOffsets],
    src: &dyn SliceSource,
) -> Result<(), Refusal> {
    if layers.len() != hp.num_layers as usize {
        return refuse(
            "Q0",
            format!(
                "{} layers ({} described); the signature has {LAYERS}",
                hp.num_layers,
                layers.len()
            ),
        );
    }
    admit_each(hp, embedding, |l| Ok(layers[l].clone()), src).map(|_| ())
}

/// [`admit`] with each layer's descriptor taken from `layer` as its turn comes, so a model refused
/// at a layer is read no further; returns every layer's descriptor.
pub fn admit_each(
    hp: &ModelHyperparams,
    embedding: QuantScheme,
    mut layer: impl FnMut(usize) -> Result<SubtensorOffsets, RuntimeError>,
    src: &dyn SliceSource,
) -> Result<Vec<SubtensorOffsets>, Refusal> {
    admit_structure(hp, embedding)?;
    let mut layers = Vec::with_capacity(LAYERS as usize);
    for l in 0..LAYERS as usize {
        let offsets = layer(l).map_err(|e| Refusal {
            condition: "Q0",
            reason: format!("layer {l}: {e}"),
        })?;
        admit_layer(l, &offsets, src)?;
        for (name, slice) in [
            ("ssm_alpha", &offsets.ssm_alpha),
            ("ssm_beta", &offsets.ssm_beta),
        ] {
            let Some(slice) = slice else { continue };
            let bytes = src.read(l, slice, 0, slice.length).map_err(|e| Refusal {
                condition: "Q6",
                reason: format!("layer {l} {name}: {e}"),
            })?;
            if let Some(i) = first_non_bf16(&bytes) {
                return Err(Refusal {
                    condition: "Q6",
                    reason: format!("layer {l} {name} value {i} is not a BF16 value"),
                });
            }
        }
        layers.push(offsets);
    }
    Ok(layers)
}

/// The switches (Q9), checked before every other condition: `LUMEN_CUDA_NATIVE_PREFILL=0` selects the
/// F32 prefill, and so does `LUMEN_CUDA_PREFILL_F32`, which forces every model's F32 prefill path.
pub fn switches(native_prefill: bool, prefill_f32: bool) -> Result<(), Refusal> {
    if !native_prefill {
        return refuse("Q9", "LUMEN_CUDA_NATIVE_PREFILL=0".into());
    }
    if prefill_f32 {
        return refuse("Q9", "LUMEN_CUDA_PREFILL_F32 is set".into());
    }
    Ok(())
}

/// Whether a model whose planes are stored in `schemes` carries NVFP4 or FP8 planes: the models the
/// load log names the prefill route for, since the native route and its F32 alternative are theirs.
/// Any other model keeps its own prefill (by default the F16 GEMM), which a route line would misname.
pub fn carries_planar_planes(schemes: impl IntoIterator<Item = QuantScheme>) -> bool {
    schemes
        .into_iter()
        .any(|q| matches!(q, QuantScheme::Nvfp4 | QuantScheme::Fp8E4M3))
}

/// Refuse at a publication check point when a test injected a refusal of one of its `conditions`
/// (feature `test-fault-injection`); otherwise nothing.
pub(crate) fn injected(conditions: &[&'static str]) -> Result<(), Refusal> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(condition) = fault::refused(conditions) {
        return refuse(condition, "injected".into());
    }
    let _ = conditions;
    Ok(())
}

/// Test-only refusal and forward-failure injection for the route's publication and forward.
#[cfg(any(test, feature = "test-fault-injection"))]
pub mod fault {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static REFUSE: Mutex<Option<&'static str>> = Mutex::new(None);
    static FAIL_LAYER: AtomicUsize = AtomicUsize::new(usize::MAX);

    /// Make every later publication refuse `condition` at its check point, until `None` is set.
    pub fn refuse(condition: Option<&'static str>) {
        *REFUSE.lock().unwrap() = condition;
    }

    /// Make the next native forward that reaches layer `layer` fail there, once.
    pub fn fail_forward_at_layer(layer: usize) {
        FAIL_LAYER.store(layer, Ordering::SeqCst);
    }

    pub(crate) fn refused(conditions: &[&'static str]) -> Option<&'static str> {
        REFUSE.lock().unwrap().filter(|c| conditions.contains(c))
    }

    pub(crate) fn fails_at(layer: usize) -> bool {
        FAIL_LAYER
            .compare_exchange(layer, usize::MAX, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

/// Index of the first F32 in `bytes` whose low 16 bits are not zero, i.e. that BF16 cannot hold.
pub fn first_non_bf16(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(4)
        .position(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0xFFFF != 0)
}

fn admit_layer(l: usize, layer: &SubtensorOffsets, src: &dyn SliceSource) -> Result<(), Refusal> {
    let attention = is_attention_layer(l);
    let want_type = if attention { 0 } else { 1 };
    if layer.layer_type != Some(want_type) {
        return refuse(
            "Q0",
            format!(
                "layer {l} has type {:?}; the signature has {} there",
                layer.layer_type,
                if attention {
                    "full attention (0)"
                } else {
                    "GDN (1)"
                }
            ),
        );
    }
    let fields = layer.slice_fields();
    if fields.experts.is_some() {
        return refuse("Q0", format!("layer {l} has experts"));
    }
    let slices = fields
        .mandatory
        .iter()
        .map(|&(name, s)| (name, (s.length > 0).then_some(s)))
        .chain(fields.optional.iter().map(|&(name, s)| (name, s.as_ref())));
    for (name, slice) in slices {
        let (expect, condition) = expected(name, attention);
        admit_slice(l, name, slice, expect, src).map_err(|reason| Refusal { condition, reason })?;
    }
    Ok(())
}

fn admit_slice(
    l: usize,
    name: &str,
    slice: Option<&TensorSlice>,
    expect: Expect,
    src: &dyn SliceSource,
) -> Result<(), String> {
    let at = format!("layer {l} {name}");
    let slice = match (expect, slice) {
        (Expect::Absent, None) => return Ok(()),
        (Expect::Absent, Some(s)) => {
            return Err(format!(
                "{at} is present ({:?}, {} bytes); the signature has none",
                s.quant, s.length
            ))
        }
        (_, None) => return Err(format!("{at} is missing")),
        (_, Some(s)) => s,
    };
    let (scheme, base) = match expect {
        Expect::F32(n) => (QuantScheme::F32, n * 4),
        Expect::Planar(scheme, n, k) => {
            let base = match scheme {
                QuantScheme::Nvfp4 => Nvfp4Planes::for_shape(n, k).map(|p| p.total_bytes()),
                _ => Fp8Planes::for_shape(n, k).map(|p| p.total_bytes()),
            }
            .map_err(|e| format!("{at}: {e}"))?;
            (scheme, base)
        }
        Expect::Absent => unreachable!("handled above"),
    };
    if slice.quant != scheme {
        return Err(format!(
            "{at} is {:?}; the signature has {scheme:?}",
            slice.quant
        ));
    }
    if let Expect::F32(_) = expect {
        return if slice.length == base {
            Ok(())
        } else {
            Err(format!(
                "{at} is {} bytes; the signature has {base}",
                slice.length
            ))
        };
    }
    match lumen_format::planar_input_scale(base, slice.length) {
        Some(true) => {}
        Some(false) => return Err(format!("{at} carries no input_scale")),
        None => {
            return Err(format!(
                "{at} is {} bytes; the signature has {} (planes and input_scale)",
                slice.length,
                base + lumen_format::PLANAR_INPUT_SCALE_BYTES
            ))
        }
    }
    let scale = input_scale(src, l, slice, base).map_err(|e| format!("{at}: {e}"))?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "{at} input_scale {scale} is not finite and positive"
        ));
    }
    Ok(())
}

/// The activation scale stored after a slice's `planes_bytes` bytes of planes.
pub fn input_scale(
    src: &dyn SliceSource,
    layer: usize,
    slice: &TensorSlice,
    planes_bytes: u64,
) -> Result<f32, RuntimeError> {
    let b = src.read(
        layer,
        slice,
        planes_bytes,
        lumen_format::PLANAR_INPUT_SCALE_BYTES,
    )?;
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Every value a signature layer's reads return. Slices are told apart by offset: each one
    /// gets its own.
    #[derive(Default)]
    struct Values {
        /// Activation scales by (layer, slice offset); absent ones read 0.25.
        scales: HashMap<(usize, u64), f32>,
        /// (layer, slice offset, element) whose F32 is not a BF16 value.
        not_bf16: Option<(usize, u64, u64)>,
    }

    impl SliceSource for Values {
        fn read(
            &self,
            layer: usize,
            slice: &TensorSlice,
            start: u64,
            len: u64,
        ) -> Result<Vec<u8>, RuntimeError> {
            assert!(start + len <= slice.length, "read past the slice");
            if len == 4 && start > 0 {
                let s = self
                    .scales
                    .get(&(layer, slice.offset))
                    .copied()
                    .unwrap_or(0.25);
                return Ok(s.to_le_bytes().to_vec());
            }
            let mut out = Vec::with_capacity(len as usize);
            for i in 0..len / 4 {
                let mut bits = (((i * 2654435761) % 65521) as u32) << 16;
                if self.not_bf16 == Some((layer, slice.offset, i)) {
                    bits |= 1;
                }
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Ok(out)
        }
    }

    fn signature_hyperparams() -> ModelHyperparams {
        ModelHyperparams {
            num_layers: LAYERS,
            num_heads: HEADS,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            hidden_dim: HIDDEN,
            intermediate_dim: INTERMEDIATE,
            vocab_size: 248320,
            max_seq_len: 262144,
            rope_params: Some(ROPE),
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-6,
            rotary_dim: Some(ROTARY_DIM),
            rope_neox: true,
            gdn: Some(GDN),
        }
    }

    /// The converter's layer layout for the admitted checkpoint, written out independently of
    /// [`expected`]: full attention at layers 3, 7, ..., 63; FP8 projections and NVFP4 MLP, each with
    /// its activation scale; F32 norms and GDN tensors.
    fn signature_layer(l: usize) -> SubtensorOffsets {
        let attention = l % 4 == 3;
        let mut next = 0u64;
        let mut put = |quant: QuantScheme, length: u64| {
            let s = TensorSlice {
                offset: next,
                length,
                quant,
            };
            next += length;
            s
        };
        let fp8 = |n: u64, k: u64| n * k + 4 + 4;
        let nvfp4 = |n: u64, k: u64| n * k / 2 + n * k / 16 + 4 + 4;
        let f32s = |n: u64| n * 4;
        use QuantScheme::{Fp8E4M3, Nvfp4, F32};
        let zero = TensorSlice {
            offset: 0,
            length: 0,
            quant: F32,
        };
        let (wq, wk, wv, wo) = if attention {
            (
                put(Fp8E4M3, fp8(12288, 5120)),
                put(Fp8E4M3, fp8(1024, 5120)),
                put(Fp8E4M3, fp8(1024, 5120)),
                put(Fp8E4M3, fp8(5120, 6144)),
            )
        } else {
            (put(Fp8E4M3, fp8(10240, 5120)), zero, zero, zero)
        };
        let attn_norm = put(F32, f32s(5120));
        let attn_post_norm = Some(put(F32, f32s(5120)));
        let gdn = |put: &mut dyn FnMut(QuantScheme, u64) -> TensorSlice,
                   q: QuantScheme,
                   len: u64| { (!attention).then(|| put(q, len)) };
        let attn_gate = gdn(&mut put, Fp8E4M3, fp8(6144, 5120));
        let ssm_a = gdn(&mut put, F32, f32s(48));
        let ssm_conv1d = gdn(&mut put, F32, f32s(10240 * 4));
        let ssm_dt = gdn(&mut put, F32, f32s(48));
        let ssm_beta = gdn(&mut put, F32, f32s(48 * 5120));
        let ssm_alpha = gdn(&mut put, F32, f32s(48 * 5120));
        let ssm_norm = gdn(&mut put, F32, f32s(128));
        let ssm_out = gdn(&mut put, Fp8E4M3, fp8(5120, 6144));
        let w_gate = put(Nvfp4, nvfp4(17408, 5120));
        let w_up = put(Nvfp4, nvfp4(17408, 5120));
        let w_down = put(Nvfp4, nvfp4(5120, 17408));
        let (attn_q_norm, attn_k_norm) = if attention {
            (Some(put(F32, f32s(256))), Some(put(F32, f32s(256))))
        } else {
            (None, None)
        };
        SubtensorOffsets {
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            attn_norm,
            ffn_norm: zero,
            bq: None,
            bk: None,
            bv: None,
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
            layer_type: Some(if attention { 0 } else { 1 }),
        }
    }

    fn signature_layers() -> Vec<SubtensorOffsets> {
        (0..LAYERS as usize).map(signature_layer).collect()
    }

    /// Admit the signature after `edit`, returning the condition it fails (or `None`) and why.
    fn verdict(
        edit: impl FnOnce(
            &mut ModelHyperparams,
            &mut QuantScheme,
            &mut Vec<SubtensorOffsets>,
            &mut Values,
        ),
    ) -> Option<(&'static str, String)> {
        let mut hp = signature_hyperparams();
        let mut embedding = QuantScheme::Bf16;
        let mut layers = signature_layers();
        let mut values = Values::default();
        edit(&mut hp, &mut embedding, &mut layers, &mut values);
        admit(&hp, embedding, &layers, &values)
            .err()
            .map(|r| (r.condition, r.reason))
    }

    fn assert_refused(
        what: &str,
        condition: &str,
        needle: &str,
        edit: impl FnOnce(
            &mut ModelHyperparams,
            &mut QuantScheme,
            &mut Vec<SubtensorOffsets>,
            &mut Values,
        ),
    ) {
        match verdict(edit) {
            Some((c, reason)) => {
                assert_eq!(c, condition, "{what}: refused by {c}: {reason}");
                assert!(reason.contains(needle), "{what}: {reason}");
            }
            None => panic!("{what}: admitted"),
        }
    }

    #[test]
    fn the_signature_is_admitted() {
        assert_eq!(verdict(|_, _, _, _| {}), None);
        let layers = signature_layers();
        assert_eq!(
            layers.iter().filter(|l| l.layer_type == Some(0)).count(),
            16,
            "16 attention layers"
        );
    }

    #[test]
    fn attention_near_misses_are_refused_by_name() {
        assert_refused("32 query heads", "Q0.b", "32 query heads", |hp, _, _, _| {
            hp.num_heads = 32
        });
        assert_refused("8 KV heads", "Q0.b", "8 KV heads", |hp, _, _, _| {
            hp.num_kv_heads = 8
        });
        assert_refused("head size 128", "Q0.b", "head size 128", |hp, _, _, _| {
            hp.head_dim = 128
        });
        assert_refused("rotary_dim 128", "Q0.b", "Some(128)", |hp, _, _, _| {
            hp.rotary_dim = Some(128)
        });
        assert_refused(
            "no q norm",
            "Q0.b",
            "layer 3 attn_q_norm is missing",
            |_, _, l, _| l[3].attn_q_norm = None,
        );
        assert_refused("a q bias", "Q0.b", "layer 3 bq is present", |_, _, l, _| {
            l[3].bq = Some(TensorSlice {
                offset: 1 << 40,
                length: 12288 * 4,
                quant: QuantScheme::F32,
            })
        });
    }

    #[test]
    fn gdn_and_shape_near_misses_are_refused_by_name() {
        assert_refused("conv kernel 3", "Q0.c", "conv_kernel: 3", |hp, _, _, _| {
            hp.gdn = Some(GdnDims {
                conv_kernel: 3,
                ..GDN
            })
        });
        assert_refused("8 GDN k-heads", "Q0.c", "num_k_heads: 8", |hp, _, _, _| {
            hp.gdn = Some(GdnDims {
                num_k_heads: 8,
                ..GDN
            })
        });
        assert_refused("H = 4096", "Q0.a", "hidden 4096", |hp, _, _, _| {
            hp.hidden_dim = 4096
        });
        for e in [QuantScheme::F16, QuantScheme::Q8_0] {
            assert_refused("embedding", "Q0.d", &format!("{e:?}"), |_, emb, _, _| {
                *emb = e
            });
        }
    }

    #[test]
    fn layout_near_misses_are_refused_by_name() {
        assert_refused(
            "one MoE layer",
            "Q0",
            "layer 5 has experts",
            |_, _, l, _| l[5].experts = Some(Vec::new()),
        );
        assert_refused("a router", "Q0", "layer 5 router_weight", |_, _, l, _| {
            l[5].router_weight = Some(l[5].attn_norm)
        });
        assert_refused(
            "attention at layer 2",
            "Q0",
            "layer 2 has type Some(0)",
            |_, _, l, _| {
                let mut a = signature_layer(3);
                a.layer_type = Some(0);
                l[2] = a;
            },
        );
    }

    #[test]
    fn projection_near_misses_are_refused_by_name() {
        assert_refused(
            "BF16 out_proj",
            "Q5",
            "layer 0 ssm_out is Bf16",
            |_, _, l, _| {
                let s = l[0].ssm_out.as_mut().unwrap();
                s.quant = QuantScheme::Bf16;
                s.length = 5120 * 6144 * 2;
            },
        );
        assert_refused("NVFP4 k_proj", "Q5", "layer 3 wk is Nvfp4", |_, _, l, _| {
            l[3].wk.quant = QuantScheme::Nvfp4;
            l[3].wk.length = Nvfp4Planes::for_shape(1024, 5120).unwrap().total_bytes() + 4;
        });
        assert_refused(
            "no input_scale",
            "Q5",
            "layer 7 w_down carries no input_scale",
            |_, _, l, _| l[7].w_down.length -= 4,
        );
        assert_refused(
            "NaN input_scale",
            "Q5",
            "layer 9 wq input_scale NaN",
            |_, _, l, v| {
                v.scales.insert((9, l[9].wq.offset), f32::NAN);
            },
        );
        assert_refused(
            "zero input_scale",
            "Q5",
            "layer 11 w_up input_scale 0",
            |_, _, l, v| {
                v.scales.insert((11, l[11].w_up.offset), 0.0);
            },
        );
    }

    #[test]
    fn the_structure_alone_is_checked_before_any_layer_is_read() {
        let hp = signature_hyperparams();
        assert_eq!(admit_structure(&hp, QuantScheme::Bf16), Ok(()));
        let short = ModelHyperparams {
            num_layers: 32,
            ..hp
        };
        let r = admit_structure(&short, QuantScheme::Bf16).unwrap_err();
        assert_eq!(
            (r.condition, r.reason.as_str()),
            ("Q0", "32 layers; the signature has 64")
        );
        let r = admit_structure(&hp, QuantScheme::Q8_0).unwrap_err();
        assert_eq!(r.condition, "Q0.d");
        // The layer count must match the layers described.
        let r = admit(
            &hp,
            QuantScheme::Bf16,
            &signature_layers()[1..],
            &Values::default(),
        )
        .unwrap_err();
        assert_eq!(r.condition, "Q0");
        assert!(r.reason.contains("(63 described)"), "{}", r.reason);
    }

    #[test]
    fn admission_reads_no_layer_past_the_one_it_refuses() {
        let (hp, values) = (signature_hyperparams(), Values::default());
        let mut asked = Vec::new();
        let layers = admit_each(
            &hp,
            QuantScheme::Bf16,
            |l| {
                asked.push(l);
                Ok(signature_layer(l))
            },
            &values,
        )
        .unwrap();
        assert_eq!(
            asked,
            (0..64).collect::<Vec<_>>(),
            "each layer asked for once"
        );
        assert_eq!(layers.len(), 64);
        assert_eq!(layers[63].w_down.length, signature_layer(63).w_down.length);

        // An artifact without activation scales is refused at its first layer, which is all it reads.
        let mut asked = Vec::new();
        let r = admit_each(
            &hp,
            QuantScheme::Bf16,
            |l| {
                asked.push(l);
                let mut st = signature_layer(l);
                st.w_down.length -= 4;
                Ok(st)
            },
            &values,
        )
        .unwrap_err();
        assert_eq!(r.condition, "Q5");
        assert_eq!(asked, [0]);
    }

    /// A provider of `layers` tiny layers, counting its fetches.
    struct Counting {
        fetches: std::sync::Mutex<Vec<usize>>,
    }

    impl WeightProvider for Counting {
        fn prefetch_layer(
            &self,
            layer: usize,
            priority: crate::weight::cache::PrefetchPriority,
        ) -> Result<crate::weight::cache::PrefetchHandle, RuntimeError> {
            Ok(crate::weight::cache::PrefetchHandle::new(layer, priority))
        }
        fn get_layer_blocking(&self, layer: usize) -> Result<LayerView, RuntimeError> {
            self.fetches.lock().unwrap().push(layer);
            let mut st = signature_layer(3);
            st.attn_norm = TensorSlice {
                offset: 0,
                length: 8,
                quant: QuantScheme::F32,
            };
            Ok(LayerView::from_owned(layer, (0..16).collect(), st))
        }
        fn try_get_layer(&self, _: usize) -> Option<LayerView> {
            None
        }
        fn release_layer_hint(&self, _: usize) {}
        fn stats(&self) -> crate::weight::cache::CacheStats {
            Default::default()
        }
        fn num_layers(&self) -> usize {
            4
        }
    }

    #[test]
    fn provider_slices_fetch_a_layer_once_per_visit() {
        let provider = Counting {
            fetches: Default::default(),
        };
        let src = ProviderSlices::new(&provider);
        for l in [0, 0, 1, 1, 1, 0] {
            let norm = src.subtensors(l).unwrap().attn_norm;
            assert_eq!(src.read(l, &norm, 4, 4).unwrap(), [4, 5, 6, 7]);
        }
        assert_eq!(*provider.fetches.lock().unwrap(), [0, 1, 0]);
    }

    #[test]
    fn an_injected_refusal_stops_publication_at_its_own_check_point() {
        fault::refuse(Some("Q4"));
        assert_eq!(injected(&["Q1", "Q2", "Q3"]), Ok(()));
        let r = injected(&["Q4"]).unwrap_err();
        assert_eq!((r.condition, r.reason.as_str()), ("Q4", "injected"));
        fault::refuse(None);
        assert_eq!(injected(&["Q4"]), Ok(()));
        fault::fail_forward_at_layer(30);
        assert!(!fault::fails_at(29));
        assert!(fault::fails_at(30));
        assert!(!fault::fails_at(30), "one shot");
    }

    #[test]
    fn either_switch_selects_the_f32_route_by_name() {
        assert_eq!(switches(true, false), Ok(()));
        for (native_prefill, prefill_f32, reason) in [
            (false, false, "LUMEN_CUDA_NATIVE_PREFILL=0"),
            (false, true, "LUMEN_CUDA_NATIVE_PREFILL=0"),
            (true, true, "LUMEN_CUDA_PREFILL_F32 is set"),
        ] {
            let r = switches(native_prefill, prefill_f32).unwrap_err();
            assert_eq!((r.condition, r.reason.as_str()), ("Q9", reason));
        }
    }

    #[test]
    fn only_nvfp4_or_fp8_planes_name_the_prefill_route() {
        use QuantScheme::*;
        assert!(!carries_planar_planes([]));
        assert!(!carries_planar_planes([
            Q8_0, Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q2_K, Q3_K, CtInt4G32, F16, Bf16, F32
        ]));
        assert!(carries_planar_planes([Bf16, F32, Nvfp4]));
        assert!(carries_planar_planes([Q4_0, Fp8E4M3]));
    }

    #[test]
    fn ab_values_that_bf16_cannot_hold_are_refused() {
        assert_refused(
            "a/b not BF16",
            "Q6",
            "layer 4 ssm_beta value 777",
            |_, _, l, v| {
                v.not_bf16 = Some((4, l[4].ssm_beta.unwrap().offset, 777));
            },
        );
    }
}
