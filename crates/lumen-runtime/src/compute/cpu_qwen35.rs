//! Qwen3.5-family layers for the CPU backends: GatedDeltaNet linear attention,
//! gated full attention (interleaved Q+gate projection, per-head Q/K RMSNorm,
//! partial NeoX RoPE, sigmoid output gate), the dense SwiGLU FFN and the routed
//! MoE FFN with its gated shared expert.
//!
//! Both CPU backends run these layers through [`Qwen35`] and differ only in the
//! matrix-vector product they pass in ([`MatVec`]), so the naive backend is the
//! numerical reference for the SIMD one.
//!
//! A backend prepares the module once per model from `preload_weights`: every
//! layer is read and checked against the shapes and storage the compute reads,
//! and each layer's attention kind, FFN kind and GDN slot are recorded. Until
//! then every layer, recurrent-state save and restore is refused, so nothing
//! mutates KV or recurrent state before the whole model has been checked.
//!
//! The recurrent state belongs to the backend, as on Metal and CUDA: one live
//! hybrid sequence per backend. It is held in the order Metal's
//! `RecurrentState` writer uses, so a session saved by either backend restores
//! on the other: GDN layers in ascending order; h as `[value head][value
//! coordinate][key coordinate]` (float index `(h * D + v) * D + k`); the conv
//! ring as `[physical slot][Q | K | V channel]` holding the raw projection
//! inputs; the ring position is the next slot to overwrite.

use crate::compute::ActivationBuffer;
use crate::error::RuntimeError;
use crate::kv::disk::{GdnLayout, RecurrentState, RECURRENT_DTYPE_F32};
use crate::kv::KvCacheView;
use crate::sampling::softmax_inplace;
use crate::weight::cache::{LayerView, WeightProvider};
use crate::weight::kquant::{dequant_row_into, row_bytes};
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;

/// The matrix-vector product a backend supplies: `out[..rows] = W x` for the
/// row-major `rows x cols` matrix `w` stored as `scheme`. Short storage, input
/// or output is an error.
pub(crate) trait MatVec {
    fn matvec(
        &self,
        out: &mut [f32],
        w: &[u8],
        x: &[f32],
        rows: usize,
        cols: usize,
        scheme: QuantScheme,
    ) -> Result<(), RuntimeError>;
}

/// The attention block of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Attention {
    /// Llama-style attention, run by the backend's own path.
    Plain,
    /// Qwen3.5 full attention (`attn_q_norm` present).
    Gated,
    /// GatedDeltaNet linear attention (`layer_type == 1`).
    Gdn,
}

/// The FFN block of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ffn {
    Dense,
    Moe,
}

/// Model dimensions the layers read, fixed at preparation.
#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
    theta: f64,
    hv: usize,
    hk: usize,
    d: usize,
    kernel: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
}

impl Dims {
    fn new(hp: &ModelHyperparams) -> Self {
        let head_dim = hp.head_dim as usize;
        let rotary = match hp.rotary_dim {
            Some(r) if r > 0 && (r as usize) < head_dim => r as usize,
            _ => head_dim,
        };
        let g = hp.gdn_dims();
        Self {
            hidden: hp.hidden_dim as usize,
            eps: hp.norm_eps,
            heads: hp.num_heads as usize,
            kv_heads: hp.num_kv_heads as usize,
            head_dim,
            rotary,
            theta: hp.rope_params.map_or(10000.0, |r| r.theta as f64),
            hv: g.num_v_heads as usize,
            hk: g.num_k_heads as usize,
            d: g.head_dim as usize,
            kernel: g.conv_kernel as usize,
            inter: hp.intermediate_dim as usize,
            experts: hp.num_experts.unwrap_or(0) as usize,
            top_k: hp.num_active_experts.unwrap_or(0) as usize,
        }
    }

    fn qk(&self) -> usize {
        self.hk * self.d
    }

    fn v(&self) -> usize {
        self.hv * self.d
    }

    fn qkv(&self) -> usize {
        2 * self.qk() + self.v()
    }
}

/// One GDN layer's recurrent state.
struct GdnState {
    /// `[hv][d][d]`, value-major, key index fastest.
    h: Vec<f32>,
    /// `[kernel - 1][qkv]` raw projection inputs, stored physically.
    conv: Vec<f32>,
    /// The next ring slot to overwrite (also the oldest one).
    pos: usize,
}

impl GdnState {
    fn zeroed(d: &Dims) -> Self {
        Self {
            h: vec![0.0; d.hv * d.d * d.d],
            conv: vec![0.0; (d.kernel - 1) * d.qkv()],
            pos: 0,
        }
    }
}

/// What preparation recorded about the model.
struct Plan {
    dims: Dims,
    layers: Vec<(Attention, Ffn)>,
    /// `gdn_slot[layer]` is the layer's index among the GDN layers.
    gdn_slot: Vec<Option<usize>>,
    num_gdn: usize,
}

/// The Qwen3.5 layers and recurrent state of one CPU backend.
pub(crate) struct Qwen35 {
    plan: Option<Plan>,
    /// One entry per GDN layer, allocated on first use.
    state: Mutex<Vec<Option<GdnState>>>,
}

const NOT_PREPARED: &str =
    "the CPU backend has not read the model: call preload_weights() after init()";

impl Qwen35 {
    pub(crate) fn new() -> Self {
        Self {
            plan: None,
            state: Mutex::new(Vec::new()),
        }
    }

    /// Read every layer of `weights` and check it against what the compute
    /// reads; on success record the layer kinds and GDN slots. `check_plain`
    /// holds a plain dense layer to the backend's own path's storage rules. On
    /// any error nothing is recorded and the backend stays unprepared.
    pub(crate) fn prepare(
        &mut self,
        hp: &ModelHyperparams,
        weights: &dyn WeightProvider,
        check_plain: &dyn Fn(&SubtensorOffsets) -> Result<(), String>,
    ) -> Result<(), RuntimeError> {
        self.plan = None;
        let dims = Dims::new(hp);
        let num_layers = hp.num_layers as usize;
        if weights.num_layers() != num_layers {
            return Err(RuntimeError::Unsupported(format!(
                "the model has {} layers but its header says {num_layers}",
                weights.num_layers()
            )));
        }
        let mut layers = Vec::with_capacity(num_layers);
        let mut gdn_slot = Vec::with_capacity(num_layers);
        let mut num_gdn = 0;
        for layer in 0..num_layers {
            let view = weights.get_layer_raw(layer)?;
            let kind = check_layer(&dims, &view, check_plain)
                .map_err(|e| RuntimeError::Unsupported(format!("layer {layer}: {e}")))?;
            gdn_slot.push((kind.0 == Attention::Gdn).then(|| {
                num_gdn += 1;
                num_gdn - 1
            }));
            layers.push(kind);
        }
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) =
            (0..num_gdn).map(|_| None).collect();
        self.plan = Some(Plan {
            dims,
            layers,
            gdn_slot,
            num_gdn,
        });
        Ok(())
    }

    fn plan(&self) -> Result<&Plan, RuntimeError> {
        self.plan
            .as_ref()
            .ok_or_else(|| RuntimeError::Compute(NOT_PREPARED.into()))
    }

    /// Whether this module runs `layer`; `false` leaves it to the backend's
    /// plain dense path. An error before preparation.
    pub(crate) fn runs(&self, layer: usize) -> Result<bool, RuntimeError> {
        let plan = self.plan()?;
        let kind = plan.layers.get(layer).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "layer {layer} out of range ({} layers)",
                plan.layers.len()
            ))
        })?;
        Ok(*kind != (Attention::Plain, Ffn::Dense))
    }

    pub(crate) fn has_gdn(&self) -> bool {
        self.plan.as_ref().is_some_and(|p| p.num_gdn > 0)
    }

    pub(crate) fn has_moe(&self) -> bool {
        self.plan
            .as_ref()
            .is_some_and(|p| p.layers.iter().any(|l| l.1 == Ffn::Moe))
    }

    /// The recurrent-state layout of a prepared GDN model; `None` otherwise.
    pub(crate) fn gdn_layout(&self) -> Option<GdnLayout> {
        let plan = self.plan.as_ref().filter(|p| p.num_gdn > 0)?;
        let d = &plan.dims;
        Some(GdnLayout {
            num_gdn_layers: plan.num_gdn as u32,
            gdn_num_heads: d.hv as u32,
            gdn_head_dim: d.d as u32,
            gdn_conv_kernel_size: d.kernel as u32,
            gdn_conv_qkv_dim: d.qkv() as u32,
            gdn_dtype_tag: RECURRENT_DTYPE_F32,
        })
    }

    /// Zero every GDN layer's state.
    pub(crate) fn reset(&self) {
        for s in self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
            .flatten()
        {
            s.h.fill(0.0);
            s.conv.fill(0.0);
            s.pos = 0;
        }
    }

    /// The CPU backends' `sync_kv_to_cpu`: their KV already lives in the
    /// host cache, so only the recurrent state is copied, into `recurrent`
    /// (laid out by [`Self::gdn_layout`]). Refused before preparation.
    pub(crate) fn save_state(
        &self,
        recurrent: Option<&mut RecurrentState>,
    ) -> Result<(), RuntimeError> {
        self.plan()?;
        let Some(rec) = recurrent else {
            return Ok(());
        };
        self.check_layout(rec)?;
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for (i, s) in state.iter().enumerate() {
            let (h, conv, pos) = match s {
                Some(s) => (&s.h[..], &s.conv[..], s.pos),
                None => (&[][..], &[][..], 0),
            };
            write_f32s(&mut rec.h_states[i], h);
            write_f32s(&mut rec.conv_states[i], conv);
            rec.conv_positions[i] = pos as u32;
        }
        Ok(())
    }

    /// The CPU backends' `sync_kv_from_cpu`: replace the recurrent state with
    /// `recurrent`. Refused before preparation, for a layout other than this
    /// model's, and without a state for a model with GDN layers.
    pub(crate) fn restore_state(
        &self,
        recurrent: Option<&RecurrentState>,
    ) -> Result<(), RuntimeError> {
        self.plan()?;
        let Some(rec) = recurrent else {
            if self.has_gdn() {
                return Err(RuntimeError::Compute(
                    "a model with GDN layers restores only with its recurrent state".into(),
                ));
            }
            return Ok(());
        };
        self.check_layout(rec)?;
        let d = self.plan()?.dims;
        let slots = d.kernel - 1;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut restored = Vec::with_capacity(state.len());
        for i in 0..state.len() {
            let pos = rec.conv_positions[i] as usize;
            if pos >= slots {
                return Err(RuntimeError::Compute(format!(
                    "recurrent state: GDN layer {i} conv position {pos} is outside 0..{slots}"
                )));
            }
            let mut s = GdnState::zeroed(&d);
            read_f32s(&mut s.h, &rec.h_states[i]);
            read_f32s(&mut s.conv, &rec.conv_states[i]);
            s.pos = pos;
            restored.push(Some(s));
        }
        *state = restored;
        Ok(())
    }

    fn check_layout(&self, rec: &RecurrentState) -> Result<(), RuntimeError> {
        let want = self.gdn_layout().ok_or_else(|| {
            RuntimeError::Compute("recurrent state given for a model without GDN layers".into())
        })?;
        rec.validate_layout_matches(&want)?;
        let n = want.num_gdn_layers as usize;
        if rec.h_states.len() != n
            || rec.conv_states.len() != n
            || rec.conv_positions.len() != n
            || rec
                .h_states
                .iter()
                .any(|h| h.len() != want.h_state_bytes_per_layer())
            || rec
                .conv_states
                .iter()
                .any(|c| c.len() != want.conv_state_bytes_per_layer())
        {
            return Err(RuntimeError::Compute(
                "recurrent state buffers do not match their layout".into(),
            ));
        }
        Ok(())
    }

    /// Run `layer` (one this module [`runs`](Self::runs)) on the hidden state `x`.
    pub(crate) fn compute_layer(
        &self,
        layer: usize,
        x: &mut ActivationBuffer,
        w: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
        mv: &dyn MatVec,
    ) -> Result<(), RuntimeError> {
        let plan = self.plan()?;
        let d = &plan.dims;
        let (attn, ffn) = plan.layers[layer];
        let st = &w.subtensors;
        let mut h = x.as_f32_slice().to_vec();
        if h.len() != d.hidden {
            return Err(RuntimeError::Compute(format!(
                "layer {layer}: hidden state has {} elements, the model {}",
                h.len(),
                d.hidden
            )));
        }

        let normed = rmsnorm(&h, &vector(w, &st.attn_norm, d.hidden)?, d.eps);
        let attn_out = match attn {
            Attention::Gdn => {
                let slot = plan.gdn_slot[layer].expect("a GDN layer has a slot");
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let gs = state[slot].get_or_insert_with(|| GdnState::zeroed(d));
                gdn_attention(d, mv, w, &normed, gs)?
            }
            Attention::Gated => {
                let kv = kv.ok_or_else(|| {
                    RuntimeError::Compute("KV cache view required for attention".into())
                })?;
                gated_attention(d, mv, w, &normed, kv, seq_pos)?
            }
            Attention::Plain => {
                return Err(RuntimeError::Unsupported(format!(
                    "layer {layer}: plain attention runs on the backend's own path"
                )))
            }
        };
        add(&mut h, &attn_out);

        let ffn_norm = ffn_norm_slice(st).expect("checked at preparation");
        let normed = rmsnorm(&h, &vector(w, ffn_norm, d.hidden)?, d.eps);
        let ffn_out = match ffn {
            Ffn::Dense => swiglu(mv, w, &st.w_gate, &st.w_up, &st.w_down, &normed, d.inter, d)?,
            Ffn::Moe => moe(d, mv, w, &normed)?,
        };
        add(&mut h, &ffn_out);
        x.write_f32_from(&h);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer checks.

/// The FFN norm: `ffn_norm`, or `attn_post_norm` where `ffn_norm` is the
/// zero-length placeholder (as Metal reads it).
fn ffn_norm_slice(st: &SubtensorOffsets) -> Option<&TensorSlice> {
    if st.ffn_norm.length > 0 {
        Some(&st.ffn_norm)
    } else {
        st.attn_post_norm.as_ref()
    }
}

/// `slice` holds exactly a `rows x cols` matrix in a scheme the host decodes,
/// inside the layer blob.
fn check_matrix(
    view: &LayerView,
    name: &str,
    slice: &TensorSlice,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    let rb = row_bytes(slice.quant, cols).ok_or_else(|| {
        format!(
            "{name} is {:?}, which the CPU backends cannot read in rows of {cols}",
            slice.quant
        )
    })?;
    if slice.length as usize != rows * rb {
        return Err(format!(
            "{name} holds {} bytes; a {:?} [{rows} x {cols}] matrix is {}",
            slice.length,
            slice.quant,
            rows * rb
        ));
    }
    view.subtensor_bytes(slice)
        .map(|_| ())
        .map_err(|e| format!("{name}: {e}"))
}

fn required<'a>(name: &str, slice: &'a Option<TensorSlice>) -> Result<&'a TensorSlice, String> {
    slice.as_ref().ok_or_else(|| format!("{name} is missing"))
}

/// Classify a layer and check every tensor its compute reads.
fn check_layer(
    d: &Dims,
    view: &LayerView,
    check_plain: &dyn Fn(&SubtensorOffsets) -> Result<(), String>,
) -> Result<(Attention, Ffn), String> {
    let st = &view.subtensors;
    let attn = if st.layer_type == Some(1) {
        Attention::Gdn
    } else if st.attn_q_norm.is_some() {
        Attention::Gated
    } else {
        Attention::Plain
    };
    let ffn = if st.experts.is_some() || st.router_weight.is_some() {
        Ffn::Moe
    } else {
        Ffn::Dense
    };
    let hidden = d.hidden;
    let m = |name: &str, s: &TensorSlice, rows: usize, cols: usize| {
        check_matrix(view, name, s, rows, cols)
    };

    if (attn, ffn) == (Attention::Plain, Ffn::Dense) {
        let (qd, kvd) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
        m("attn_norm", &st.attn_norm, 1, hidden)?;
        m("wq", &st.wq, qd, hidden)?;
        m("wk", &st.wk, kvd, hidden)?;
        m("wv", &st.wv, kvd, hidden)?;
        m("wo", &st.wo, hidden, qd)?;
        m("ffn_norm", &st.ffn_norm, 1, hidden)?;
        m("w_gate", &st.w_gate, d.inter, hidden)?;
        m("w_up", &st.w_up, d.inter, hidden)?;
        m("w_down", &st.w_down, hidden, d.inter)?;
        check_plain(st)?;
        return Ok((attn, ffn));
    }
    if st.bq.is_some() || st.bk.is_some() || st.bv.is_some() {
        return Err("Q/K/V biases on a Qwen3.5 layer are not supported".into());
    }

    m("attn_norm", &st.attn_norm, 1, hidden)?;
    match attn {
        Attention::Gdn => {
            if d.kernel < 2 || d.hk == 0 || d.hv == 0 || d.d == 0 {
                return Err(format!(
                    "GDN dimensions {} V heads, {} K heads, head dim {}, conv kernel {} \
                     cannot run",
                    d.hv, d.hk, d.d, d.kernel
                ));
            }
            m("attn_qkv", &st.wq, d.qkv(), hidden)?;
            m(
                "attn_gate",
                required("attn_gate", &st.attn_gate)?,
                d.v(),
                hidden,
            )?;
            m(
                "ssm_alpha",
                required("ssm_alpha", &st.ssm_alpha)?,
                d.hv,
                hidden,
            )?;
            m(
                "ssm_beta",
                required("ssm_beta", &st.ssm_beta)?,
                d.hv,
                hidden,
            )?;
            m("ssm_a", required("ssm_a", &st.ssm_a)?, 1, d.hv)?;
            m("ssm_dt", required("ssm_dt", &st.ssm_dt)?, 1, d.hv)?;
            m(
                "ssm_conv1d",
                required("ssm_conv1d", &st.ssm_conv1d)?,
                1,
                d.qkv() * d.kernel,
            )?;
            m("ssm_norm", required("ssm_norm", &st.ssm_norm)?, 1, d.d)?;
            m("ssm_out", required("ssm_out", &st.ssm_out)?, hidden, d.v())?;
        }
        Attention::Gated => {
            let (qd, kvd) = (d.heads * d.head_dim, d.kv_heads * d.head_dim);
            if d.rotary % 2 != 0 {
                return Err(format!("rotary dimension {} is odd", d.rotary));
            }
            if st.attn_gate.is_some() {
                return Err(
                    "a full-attention layer with a separate attn_gate is not supported \
                            (the gate is interleaved into attn_q)"
                        .into(),
                );
            }
            m("attn_q", &st.wq, 2 * qd, hidden)?;
            m("attn_k", &st.wk, kvd, hidden)?;
            m("attn_v", &st.wv, kvd, hidden)?;
            m("attn_output", &st.wo, hidden, qd)?;
            m(
                "attn_q_norm",
                required("attn_q_norm", &st.attn_q_norm)?,
                1,
                d.head_dim,
            )?;
            m(
                "attn_k_norm",
                required("attn_k_norm", &st.attn_k_norm)?,
                1,
                d.head_dim,
            )?;
        }
        Attention::Plain => {
            return Err("plain attention with a MoE FFN is not supported on the CPU".into());
        }
    }

    let ffn_norm = ffn_norm_slice(st).ok_or("ffn_norm and attn_post_norm are both missing")?;
    m("ffn_norm", ffn_norm, 1, hidden)?;
    match ffn {
        Ffn::Dense => {
            m("ffn_gate", &st.w_gate, d.inter, hidden)?;
            m("ffn_up", &st.w_up, d.inter, hidden)?;
            m("ffn_down", &st.w_down, hidden, d.inter)?;
        }
        Ffn::Moe => {
            if d.experts == 0 || d.top_k == 0 || d.top_k > d.experts {
                return Err(format!(
                    "MoE layer with {} experts, {} active",
                    d.experts, d.top_k
                ));
            }
            m(
                "ffn_gate_inp",
                required("ffn_gate_inp", &st.router_weight)?,
                d.experts,
                hidden,
            )?;
            let experts = st.experts.as_ref().ok_or("experts are missing")?;
            if experts.len() != d.experts {
                return Err(format!(
                    "{} expert slices for {} experts",
                    experts.len(),
                    d.experts
                ));
            }
            for (i, e) in experts.iter().enumerate() {
                m(&format!("expert {i} gate"), &e.gate, d.inter, hidden)?;
                m(&format!("expert {i} up"), &e.up, d.inter, hidden)?;
                m(&format!("expert {i} down"), &e.down, hidden, d.inter)?;
            }
            match (
                &st.shared_expert_gate,
                &st.shared_expert_up,
                &st.shared_expert_down,
            ) {
                (Some(g), Some(u), Some(dn)) => {
                    m("shared expert gate", g, d.inter, hidden)?;
                    m("shared expert up", u, d.inter, hidden)?;
                    m("shared expert down", dn, hidden, d.inter)?;
                    if let Some(s) = &st.ffn_gate_inp_shexp {
                        m("ffn_gate_inp_shexp", s, 1, hidden)?;
                    }
                }
                (None, None, None) => {
                    if st.ffn_gate_inp_shexp.is_some() {
                        return Err("ffn_gate_inp_shexp without a shared expert".into());
                    }
                }
                _ => return Err("the shared expert is incomplete".into()),
            }
        }
    }
    Ok((attn, ffn))
}

// ---------------------------------------------------------------------------
// Compute.

fn bytes<'a>(w: &'a LayerView, s: &TensorSlice) -> Result<&'a [u8], RuntimeError> {
    w.subtensor_bytes(s)
}

/// A whole tensor of `n` elements decoded to f32.
fn vector(w: &LayerView, s: &TensorSlice, n: usize) -> Result<Vec<f32>, RuntimeError> {
    let mut out = vec![0.0f32; n];
    dequant_row_into(&mut out, bytes(w, s)?, 0, n, s.quant)?;
    Ok(out)
}

fn project(
    mv: &dyn MatVec,
    w: &LayerView,
    s: &TensorSlice,
    x: &[f32],
    rows: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let mut out = vec![0.0f32; rows];
    mv.matvec(&mut out, bytes(w, s)?, x, rows, x.len(), s.quant)?;
    Ok(out)
}

fn add(dst: &mut [f32], src: &[f32]) {
    for (a, b) in dst.iter_mut().zip(src) {
        *a += b;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rmsnorm_inplace(&mut out, w, eps);
    out
}

/// `x <- x / sqrt(mean(x^2) + eps) * w`.
fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for (v, &g) in x.iter_mut().zip(w) {
        *v = *v * scale * g;
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `down(silu(gate x) * up x)` with an `inter`-wide hidden layer.
#[allow(clippy::too_many_arguments)]
fn swiglu(
    mv: &dyn MatVec,
    w: &LayerView,
    gate: &TensorSlice,
    up: &TensorSlice,
    down: &TensorSlice,
    x: &[f32],
    inter: usize,
    d: &Dims,
) -> Result<Vec<f32>, RuntimeError> {
    let mut g = project(mv, w, gate, x, inter)?;
    let u = project(mv, w, up, x, inter)?;
    for (a, b) in g.iter_mut().zip(&u) {
        *a = silu(*a) * b;
    }
    project(mv, w, down, &g, d.hidden)
}

/// The GDN inputs of one token: the fused QKV projection, the per-head alpha
/// and beta projections, and the layer's conv, decay and dt parameters.
struct GdnStep<'a> {
    qkv: &'a [f32],
    a_raw: &'a [f32],
    b_raw: &'a [f32],
    conv_w: &'a [f32],
    ssm_a: &'a [f32],
    dt_bias: &'a [f32],
}

/// One token of the GatedDeltaNet core: the causal conv over the ring and the
/// current input, SiLU, per-head L2 norm of q and k, the gates
/// (`alpha = exp(ssm_a * softplus(a_raw + dt_bias))`, `beta = sigmoid(b_raw)`),
/// then the delta rule per value head `h`, which reads q/k head `h % hk`:
/// decay, retrieve, update, and output `S q / sqrt(d)`. Returns the `[hv][d]`
/// output before the output norm.
fn gdn_core(d: &Dims, gs: &mut GdnState, step: &GdnStep) -> Vec<f32> {
    let (qk, qkv, slots, dd) = (d.qk(), d.qkv(), d.kernel - 1, d.d);
    let mut c = vec![0.0f32; qkv];
    for (ch, out) in c.iter_mut().enumerate() {
        let w = &step.conv_w[ch * d.kernel..(ch + 1) * d.kernel];
        let mut s = 0.0f32;
        for (tap, &wt) in w[..slots].iter().enumerate() {
            s += wt * gs.conv[((gs.pos + tap) % slots) * qkv + ch];
        }
        s += w[slots] * step.qkv[ch];
        *out = silu(s);
    }
    gs.conv[gs.pos * qkv..(gs.pos + 1) * qkv].copy_from_slice(step.qkv);
    gs.pos = (gs.pos + 1) % slots;

    for head in c[..2 * qk].chunks_exact_mut(dd) {
        let norm = head.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if norm > 1e-12 { 1.0 / norm } else { 1e12 };
        for v in head.iter_mut() {
            *v *= scale;
        }
    }

    let q_scale = 1.0 / (dd as f32).sqrt();
    let mut out = vec![0.0f32; d.v()];
    for h in 0..d.hv {
        let kh = h % d.hk;
        let q = &c[kh * dd..(kh + 1) * dd];
        let k = &c[qk + kh * dd..qk + (kh + 1) * dd];
        let v = &c[2 * qk + h * dd..2 * qk + (h + 1) * dd];
        let alpha = (step.ssm_a[h] * softplus(step.a_raw[h] + step.dt_bias[h])).exp();
        let beta = sigmoid(step.b_raw[h]);
        let s = &mut gs.h[h * dd * dd..(h + 1) * dd * dd];
        for (vj, row) in s.chunks_exact_mut(dd).enumerate() {
            let mut retrieved = 0.0f32;
            for (r, &kk) in row.iter_mut().zip(k) {
                *r *= alpha;
                retrieved += *r * kk;
            }
            let delta = beta * (v[vj] - retrieved);
            let mut acc = 0.0f32;
            for ((r, &kk), &qq) in row.iter_mut().zip(k).zip(q) {
                *r += kk * delta;
                acc += *r * qq;
            }
            out[h * dd + vj] = acc * q_scale;
        }
    }
    out
}

/// The GDN output stage: each value head RMS-normalised with the shared
/// `[d]` weight, then gated by `silu(z)`.
fn gdn_output(d: &Dims, core: &[f32], z: &[f32], norm_w: &[f32]) -> Vec<f32> {
    let mut out = core.to_vec();
    for head in out.chunks_exact_mut(d.d) {
        rmsnorm_inplace(head, norm_w, d.eps);
    }
    for (o, &g) in out.iter_mut().zip(z) {
        *o *= silu(g);
    }
    out
}

fn gdn_attention(
    d: &Dims,
    mv: &dyn MatVec,
    w: &LayerView,
    normed: &[f32],
    gs: &mut GdnState,
) -> Result<Vec<f32>, RuntimeError> {
    let st = &w.subtensors;
    let opt = |s: &Option<TensorSlice>| s.expect("checked at preparation");
    let qkv = project(mv, w, &st.wq, normed, d.qkv())?;
    let z = project(mv, w, &opt(&st.attn_gate), normed, d.v())?;
    let a_raw = project(mv, w, &opt(&st.ssm_alpha), normed, d.hv)?;
    let b_raw = project(mv, w, &opt(&st.ssm_beta), normed, d.hv)?;
    let conv_w = vector(w, &opt(&st.ssm_conv1d), d.qkv() * d.kernel)?;
    let ssm_a = vector(w, &opt(&st.ssm_a), d.hv)?;
    let dt_bias = vector(w, &opt(&st.ssm_dt), d.hv)?;
    let norm_w = vector(w, &opt(&st.ssm_norm), d.d)?;
    let core = gdn_core(
        d,
        gs,
        &GdnStep {
            qkv: &qkv,
            a_raw: &a_raw,
            b_raw: &b_raw,
            conv_w: &conv_w,
            ssm_a: &ssm_a,
            dt_bias: &dt_bias,
        },
    );
    let gated = gdn_output(d, &core, &z, &norm_w);
    project(mv, w, &opt(&st.ssm_out), &gated, d.hidden)
}

/// NeoX RoPE over the first `d.rotary` dimensions of each `head_dim` head:
/// pairs `(i, i + rotary / 2)`, frequency `theta^(-2i / rotary)`.
fn rope_partial_neox(d: &Dims, x: &mut [f32], pos: usize) {
    let half = d.rotary / 2;
    let (cos, sin): (Vec<f32>, Vec<f32>) = (0..half)
        .map(|i| {
            let angle = pos as f64 / d.theta.powf((2 * i) as f64 / d.rotary as f64);
            (angle.cos() as f32, angle.sin() as f32)
        })
        .unzip();
    for head in x.chunks_exact_mut(d.head_dim) {
        for i in 0..half {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = a * sin[i] + b * cos[i];
        }
    }
}

/// Qwen3.5 full attention for one token at `pos`: the `[heads][q | gate]`
/// projection split, per-head RMSNorm of q and k, partial NeoX RoPE, K/V
/// appended to the cache, scaled-dot-product attention over the cache with
/// query head `h` reading KV head `h / (heads / kv_heads)`, the output gated by
/// `sigmoid(gate)`, then `attn_output`.
fn gated_attention(
    d: &Dims,
    mv: &dyn MatVec,
    w: &LayerView,
    normed: &[f32],
    kv: &mut KvCacheView,
    pos: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let st = &w.subtensors;
    let (hd, qd, kvd) = (d.head_dim, d.heads * d.head_dim, d.kv_heads * d.head_dim);
    let opt = |s: &Option<TensorSlice>| s.expect("checked at preparation");
    let qg = project(mv, w, &st.wq, normed, 2 * qd)?;
    let mut k = project(mv, w, &st.wk, normed, kvd)?;
    let v = project(mv, w, &st.wv, normed, kvd)?;
    let q_norm = vector(w, &opt(&st.attn_q_norm), hd)?;
    let k_norm = vector(w, &opt(&st.attn_k_norm), hd)?;

    let mut q = vec![0.0f32; qd];
    let mut gate = vec![0.0f32; qd];
    for h in 0..d.heads {
        q[h * hd..(h + 1) * hd].copy_from_slice(&qg[2 * h * hd..(2 * h + 1) * hd]);
        gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[(2 * h + 1) * hd..(2 * h + 2) * hd]);
        rmsnorm_inplace(&mut q[h * hd..(h + 1) * hd], &q_norm, d.eps);
    }
    for head in k.chunks_exact_mut(hd) {
        rmsnorm_inplace(head, &k_norm, d.eps);
    }
    rope_partial_neox(d, &mut q, pos);
    rope_partial_neox(d, &mut k, pos);

    kv.append_keys(&k);
    kv.append_values(&v);
    let n = (kv.seq_len + 1).min(kv.max_seq_len);
    let scale = 1.0 / (hd as f32).sqrt();
    let group = d.heads / d.kv_heads;
    let mut out = vec![0.0f32; qd];
    let mut scores = vec![0.0f32; n];
    let mut buf = vec![0.0f32; hd];
    for h in 0..d.heads {
        let base = (h / group) * kv.max_seq_len * hd;
        let qh = &q[h * hd..(h + 1) * hd];
        for (t, s) in scores.iter_mut().enumerate() {
            kv.read_keys_into(&mut buf, base + t * hd, hd);
            *s = qh.iter().zip(&buf).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        softmax_inplace(&mut scores);
        let oh = &mut out[h * hd..(h + 1) * hd];
        for (t, &p) in scores.iter().enumerate() {
            kv.read_values_into(&mut buf, base + t * hd, hd);
            for (o, &vv) in oh.iter_mut().zip(&buf) {
                *o += p * vv;
            }
        }
    }
    for (o, &g) in out.iter_mut().zip(&gate) {
        *o *= sigmoid(g);
    }
    project(mv, w, &st.wo, &out, d.hidden)
}

/// Experts chosen for `logits`: softmax, the `top_k` largest probabilities
/// (repeated argmax, the lowest index winning a tie), renormalised to sum to 1.
fn route(logits: &mut [f32], top_k: usize) -> Vec<(usize, f32)> {
    softmax_inplace(logits);
    let mut chosen = Vec::with_capacity(top_k);
    let mut sum = 0.0f32;
    for _ in 0..top_k {
        let mut best = (0usize, -1.0f32);
        for (i, &p) in logits.iter().enumerate() {
            if p > best.1 {
                best = (i, p);
            }
        }
        chosen.push(best);
        sum += best.1;
        logits[best.0] = -1.0;
    }
    if sum > 0.0 {
        for c in &mut chosen {
            c.1 /= sum;
        }
    }
    chosen
}

/// The routed experts weighted by [`route`], plus the shared expert scaled by
/// `sigmoid(ffn_gate_inp_shexp . x)` (by 1 without that gate).
fn moe(d: &Dims, mv: &dyn MatVec, w: &LayerView, x: &[f32]) -> Result<Vec<f32>, RuntimeError> {
    let st = &w.subtensors;
    let router = st.router_weight.expect("checked at preparation");
    let experts = st.experts.as_ref().expect("checked at preparation");
    let mut logits = project(mv, w, &router, x, d.experts)?;
    let mut out = vec![0.0f32; d.hidden];
    for (e, weight) in route(&mut logits, d.top_k) {
        let ex = &experts[e];
        let y = swiglu(mv, w, &ex.gate, &ex.up, &ex.down, x, d.inter, d)?;
        for (o, v) in out.iter_mut().zip(&y) {
            *o += weight * v;
        }
    }
    if let (Some(g), Some(u), Some(dn)) = (
        &st.shared_expert_gate,
        &st.shared_expert_up,
        &st.shared_expert_down,
    ) {
        let y = swiglu(mv, w, g, u, dn, x, d.inter, d)?;
        let gate = match &st.ffn_gate_inp_shexp {
            Some(s) => {
                let gw = vector(w, s, d.hidden)?;
                sigmoid(gw.iter().zip(x).map(|(a, b)| a * b).sum())
            }
            None => 1.0,
        };
        for (o, v) in out.iter_mut().zip(&y) {
            *o += gate * v;
        }
    }
    Ok(out)
}

fn write_f32s(dst: &mut [u8], src: &[f32]) {
    dst.fill(0);
    for (b, v) in dst.chunks_exact_mut(4).zip(src) {
        b.copy_from_slice(&v.to_le_bytes());
    }
}

fn read_f32s(dst: &mut [f32], src: &[u8]) {
    for (v, b) in dst.iter_mut().zip(src.chunks_exact(4)) {
        *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
}

#[cfg(test)]
#[path = "../../tests/native_oracle/gdn.rs"]
#[allow(dead_code)]
mod gdn_oracle;

#[cfg(test)]
mod tests {
    use super::gdn_oracle as orc;
    use super::*;
    use crate::compute::cpu_naive::NaiveF32Backend;
    use crate::kv::{KvCache, KvCacheConfig, KvPrecision};
    use lumen_format::hyperparams::{GdnDims, RopeParams};
    use lumen_format::index::ExpertSlice;

    /// Relative L2 distance of `got` from the f64 reference `want`.
    fn rel(got: &[f32], want: &[f64]) -> f64 {
        orc::err_of(got, want).rel_l2
    }

    /// The tolerance against an f64 reference for the naive (reference) path.
    const TIGHT: f64 = 1e-5;

    fn hp(gdn: GdnDims) -> ModelHyperparams {
        ModelHyperparams {
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 256,
            hidden_dim: 64,
            intermediate_dim: 32,
            vocab_size: 16,
            max_seq_len: 16,
            rope_params: Some(RopeParams {
                theta: 10_000_000.0,
                ..RopeParams::default()
            }),
            num_experts: Some(4),
            num_active_experts: Some(2),
            norm_eps: 1e-6,
            rotary_dim: Some(64),
            rope_neox: true,
            gdn: Some(gdn),
        }
    }

    fn rng_vec(g: &mut orc::Rng, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| (scale * g.normal()) as f32).collect()
    }

    /// A layer blob built tensor by tensor, every tensor F32.
    struct Blob(Vec<u8>);

    impl Blob {
        fn push(&mut self, v: &[f32]) -> TensorSlice {
            let offset = self.0.len() as u64;
            self.0.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            TensorSlice {
                offset,
                length: (v.len() * 4) as u64,
                quant: QuantScheme::F32,
            }
        }
    }

    fn offsets() -> SubtensorOffsets {
        let z = TensorSlice {
            offset: 0,
            length: 0,
            quant: QuantScheme::F32,
        };
        SubtensorOffsets {
            wq: z,
            wk: z,
            wv: z,
            wo: z,
            bq: None,
            bk: None,
            bv: None,
            w_gate: z,
            w_up: z,
            w_down: z,
            attn_norm: z,
            ffn_norm: z,
            router_weight: None,
            experts: None,
            shared_expert_gate: None,
            shared_expert_up: None,
            shared_expert_down: None,
            attn_gate: None,
            attn_post_norm: None,
            ssm_a: None,
            ssm_conv1d: None,
            ssm_dt: None,
            ssm_beta: None,
            ssm_alpha: None,
            ssm_norm: None,
            ssm_out: None,
            attn_q_norm: None,
            attn_k_norm: None,
            ffn_gate_inp_shexp: None,
            layer_type: None,
        }
    }

    /// `W x` in f64 for the row-major `rows x cols` matrix `w`.
    fn mv64(w: &[f32], x: &[f64], rows: usize) -> Vec<f64> {
        let cols = x.len();
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * x[c]).sum())
            .collect()
    }

    fn sigmoid64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn rms64(x: &mut [f64], w: &[f32], eps: f64) {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + eps).sqrt();
        for (v, &g) in x.iter_mut().zip(w) {
            *v *= s * g as f64;
        }
    }

    // -----------------------------------------------------------------------
    // T1: the GDN core against the f64 oracle at the 27B shape.

    /// Run `inp` through `gdn_core` token by token from the oracle's starting
    /// state and ring; returns the outputs `[t][hv][d]` and the final state.
    fn run_core(inp: &orc::Inputs) -> (Vec<f32>, GdnState) {
        let d = Dims {
            hv: orc::H,
            hk: orc::HK,
            d: orc::D,
            kernel: orc::SLOTS + 1,
            ..Dims::new(&hp(GdnDims::QWEN35_9B))
        };
        let mut gs = GdnState {
            h: inp.s0.clone(),
            conv: inp.ring.clone(),
            pos: inp.state_pos,
        };
        let mut out = Vec::with_capacity(inp.t * orc::H * orc::D);
        for t in 0..inp.t {
            let qkv: Vec<f32> = inp.qkv[t * orc::CONV..(t + 1) * orc::CONV]
                .iter()
                .map(|&b| orc::bf(b))
                .collect();
            let ab = &inp.ab[t * orc::AB..(t + 1) * orc::AB];
            out.extend(gdn_core(
                &d,
                &mut gs,
                &GdnStep {
                    qkv: &qkv,
                    a_raw: &ab[..orc::H],
                    b_raw: &ab[orc::H..],
                    conv_w: &inp.conv_w,
                    ssm_a: &inp.ssm_a,
                    dt_bias: &inp.dt_bias,
                },
            ));
        }
        (out, gs)
    }

    fn check_core(inp: &orc::Inputs, what: &str) {
        let want = orc::oracle(inp);
        let (out, gs) = run_core(inp);
        let per_token = orc::H * orc::D;
        for t in 0..inp.t {
            let e = rel(
                &out[t * per_token..(t + 1) * per_token],
                &want.out[t * per_token..(t + 1) * per_token],
            );
            assert!(e <= TIGHT, "{what}: token {t} output relL2 {e:.3e}");
        }
        let e = rel(&gs.h, &want.s);
        assert!(e <= TIGHT, "{what}: final state relL2 {e:.3e}");
        assert_eq!(orc::bits(&gs.conv), orc::bits(&want.ring), "{what}: ring");
        assert_eq!(gs.pos, want.state_pos, "{what}: ring position");
    }

    #[test]
    fn gdn_core_matches_the_f64_oracle_at_the_27b_shape() {
        check_core(&orc::Inputs::fresh(12, 3), "fresh");
        check_core(&orc::Inputs::continuation(9, 7, 5), "after 7 tokens");
        check_core(
            &orc::Inputs::continuation(9, 5, 8).ill_conditioned(13),
            "ill-conditioned",
        );
    }

    // -----------------------------------------------------------------------
    // Stage oracles.

    #[test]
    fn gdn_output_stage_matches_f64() {
        let mut g = orc::Rng::new(21);
        let d = Dims {
            hv: 3,
            d: 128,
            ..Dims::new(&hp(GdnDims::QWEN35_9B))
        };
        let core = rng_vec(&mut g, 3 * 128, 2.0);
        let z = rng_vec(&mut g, 3 * 128, 1.5);
        let norm_w: Vec<f32> = (0..128).map(|i| 0.5 + (i as f32) / 128.0).collect();
        let got = gdn_output(&d, &core, &z, &norm_w);
        let mut want: Vec<f64> = core.iter().map(|&v| v as f64).collect();
        for head in want.chunks_exact_mut(128) {
            rms64(head, &norm_w, d.eps as f64);
        }
        for (v, &zz) in want.iter_mut().zip(&z) {
            // SiLU, not the sigmoid of the full-attention gate
            *v *= zz as f64 * sigmoid64(zz as f64);
        }
        let e = rel(&got, &want);
        assert!(e <= TIGHT, "GDN output stage relL2 {e:.3e}");
    }

    /// Query head `h` reads KV head `h / 2`; every token's output is the f64
    /// attention over all keys so far.
    #[test]
    fn gated_attention_matches_f64() {
        let hp = hp(GdnDims::QWEN35_9B);
        let d = Dims::new(&hp);
        assert_eq!((d.head_dim, d.rotary), (256, 64));
        let (hd, qd, kvd, hidden) = (256, 4 * 256, 2 * 256, 64);
        let mut g = orc::Rng::new(5);
        let wq = rng_vec(&mut g, 2 * qd * hidden, 0.2);
        let wk = rng_vec(&mut g, kvd * hidden, 0.2);
        let wv = rng_vec(&mut g, kvd * hidden, 0.2);
        let wo = rng_vec(&mut g, hidden * qd, 0.05);
        let qn: Vec<f32> = (0..hd).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
        let kn: Vec<f32> = (0..hd).map(|i| 1.1 - (i % 5) as f32 * 0.04).collect();
        let mut blob = Blob(Vec::new());
        let mut st = offsets();
        st.wq = blob.push(&wq);
        st.wk = blob.push(&wk);
        st.wv = blob.push(&wv);
        st.wo = blob.push(&wo);
        st.attn_q_norm = Some(blob.push(&qn));
        st.attn_k_norm = Some(blob.push(&kn));
        let view = LayerView::from_owned(0, blob.0, st);
        let mut cache = KvCache::new(KvCacheConfig {
            max_seq_len: 8,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: hd,
            precision: KvPrecision::F32,
        })
        .unwrap();
        let mv = NaiveF32Backend::new();

        let (mut keys, mut vals): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
        for pos in 0..4 {
            let x = rng_vec(&mut g, hidden, 1.0);
            let mut view_kv = cache.view_mut(0).unwrap();
            let got = gated_attention(&d, &mv, &view, &x, &mut view_kv, pos).unwrap();
            cache.commit_view(view_kv).unwrap();
            cache.advance_seq_len().unwrap();

            let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let qg = mv64(&wq, &x64, 2 * qd);
            let mut k = mv64(&wk, &x64, kvd);
            let v = mv64(&wv, &x64, kvd);
            let rope = |head: &mut [f64]| {
                for i in 0..32 {
                    let a = pos as f64 / 10_000_000f64.powf(2.0 * i as f64 / 64.0);
                    let (x0, x1) = (head[i], head[i + 32]);
                    head[i] = x0 * a.cos() - x1 * a.sin();
                    head[i + 32] = x0 * a.sin() + x1 * a.cos();
                }
            };
            for head in k.chunks_exact_mut(hd) {
                rms64(head, &kn, 1e-6);
                rope(head);
            }
            keys.push(k);
            vals.push(v);
            let mut att = vec![0.0f64; qd];
            for h in 0..4 {
                let mut q = qg[2 * h * hd..(2 * h + 1) * hd].to_vec();
                let gate = &qg[(2 * h + 1) * hd..(2 * h + 2) * hd];
                rms64(&mut q, &qn, 1e-6);
                rope(&mut q);
                let kh = h / 2;
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        q.iter()
                            .zip(&k[kh * hd..(kh + 1) * hd])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                for (t, v) in vals.iter().enumerate() {
                    for j in 0..hd {
                        att[h * hd + j] += e[t] / sum * v[kh * hd + j];
                    }
                }
                for j in 0..hd {
                    att[h * hd + j] *= sigmoid64(gate[j]);
                }
            }
            let want = mv64(&wo, &att, hidden);
            let e = rel(&got, &want);
            assert!(e <= TIGHT, "position {pos}: gated attention relL2 {e:.3e}");
        }
    }

    /// Four experts with distinct scores, two active and renormalised, plus a
    /// shared expert scaled by a gate well away from 0 and 1.
    #[test]
    fn moe_matches_f64() {
        let hp = hp(GdnDims::QWEN35_9B);
        let d = Dims::new(&hp);
        let (hidden, inter) = (64, 32);
        let mut g = orc::Rng::new(9);
        let mut blob = Blob(Vec::new());
        let mut st = offsets();
        let router = rng_vec(&mut g, 4 * hidden, 0.3);
        st.router_weight = Some(blob.push(&router));
        let mut mats = Vec::new();
        let mut experts = Vec::new();
        for _ in 0..5 {
            let (gw, uw, dw) = (
                rng_vec(&mut g, inter * hidden, 0.2),
                rng_vec(&mut g, inter * hidden, 0.2),
                rng_vec(&mut g, hidden * inter, 0.2),
            );
            experts.push(ExpertSlice {
                gate: blob.push(&gw),
                up: blob.push(&uw),
                down: blob.push(&dw),
            });
            mats.push((gw, uw, dw));
        }
        let shared = experts.pop().unwrap();
        st.experts = Some(experts);
        st.shared_expert_gate = Some(shared.gate);
        st.shared_expert_up = Some(shared.up);
        st.shared_expert_down = Some(shared.down);
        let shexp = rng_vec(&mut g, hidden, 0.05);
        st.ffn_gate_inp_shexp = Some(blob.push(&shexp));
        let view = LayerView::from_owned(0, blob.0, st);
        let x = rng_vec(&mut g, hidden, 1.0);
        let got = moe(&d, &NaiveF32Backend::new(), &view, &x).unwrap();

        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let ffn = |(gw, uw, dw): &(Vec<f32>, Vec<f32>, Vec<f32>)| {
            let gg = mv64(gw, &x64, inter);
            let uu = mv64(uw, &x64, inter);
            let h: Vec<f64> = gg
                .iter()
                .zip(&uu)
                .map(|(a, b)| a * sigmoid64(*a) * b)
                .collect();
            mv64(dw, &h, hidden)
        };
        let logits = mv64(&router, &x64, 4);
        let m = logits.iter().cloned().fold(f64::MIN, f64::max);
        let p: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        let top = &order[..2];
        let norm: f64 = top.iter().map(|&e| p[e]).sum();
        let gaps: Vec<f64> = order.windows(2).map(|w| p[w[0]] - p[w[1]]).collect();
        assert!(gaps.iter().all(|&g| g > 1e-3), "scores too close: {p:?}");
        let mut want = vec![0.0f64; hidden];
        for &e in top {
            for (o, v) in want.iter_mut().zip(ffn(&mats[e])) {
                *o += p[e] / norm * v;
            }
        }
        let gate = sigmoid64(
            shexp
                .iter()
                .zip(&x64)
                .map(|(a, b)| *a as f64 * b)
                .sum::<f64>(),
        );
        assert!((0.05..0.95).contains(&gate), "shared gate {gate}");
        for (o, v) in want.iter_mut().zip(ffn(&mats[4])) {
            *o += gate * v;
        }
        let e = rel(&got, &want);
        assert!(e <= TIGHT, "MoE relL2 {e:.3e}");
    }
}
