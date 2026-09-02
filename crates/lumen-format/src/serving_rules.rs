//! Serving rules: the load-time validation the runtimes apply to a layer's
//! subtensor plan, hosted here so the CONVERTER can run the exact same
//! rules against the planned slices before writing an artifact — a single
//! source of truth instead of convert-side shadow mappings of loader
//! behavior (the round-7 structural fix). The runtime backends wrap these
//! with their own error types; lumen-convert refuses to emit any layer
//! plan these functions reject for the artifact's target.
//!
//! Every function is pure over `SubtensorOffsets` + dims and returns a
//! human-actionable message on rejection.

use crate::index::TensorSlice;
use crate::quantization::QuantScheme;

pub fn dense_slice_quant_supported(q: QuantScheme) -> bool {
    matches!(
        q,
        QuantScheme::F32
            | QuantScheme::F16
            | QuantScheme::Bf16
            | QuantScheme::Q8_0
            | QuantScheme::Q4_0
    )
}

/// Attention-geometry expectations derived from hyperparams at `init()`.
/// Plain values (no locks) so streaming load paths can validate without
/// touching the scratch mutex.
#[derive(Clone, Copy, Debug)]
pub struct AttnDims {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub hidden: usize,
    /// Attention head width: per-head Q/K norm weights are read at exactly
    /// this many F32 values per head.
    pub head_dim: usize,
    /// Fused GDN in-projection rows: (2*num_k_heads + num_v_heads)*head_dim
    /// from declared header dims, else the documented QWEN35_9B
    /// compatibility default — the same default the kernels dispatch on, so
    /// a headerless 9B-era artifact keeps loading and any mismatched
    /// artifact fails with observed-vs-expected named in the error.
    pub gdn_qkv_rows: usize,
    /// Whether the header actually declared GDN dims (false = the 9B
    /// compatibility default is in effect; mismatch errors say so).
    pub gdn_declared: bool,
    /// GDN value width num_v_heads*head_dim — the attn_gate output width
    /// the CUDA loader validates (equal to q_dim on every Qwen3.5 model
    /// by design, but derived independently here).
    pub gdn_v_dim: usize,
    /// GDN conv kernel size (declared-or-9B-default, like the rest).
    pub gdn_conv_kernel: usize,
}

/// GDN short-conv weight length: both backends index
/// `gdn_qkv_rows * conv_kernel` F32 values from `ssm_conv1d`
/// (cuda/shaders/gdn.cu, metal/gdn.rs) without consulting the slice
/// length — a short tensor is an out-of-bounds read on CUDA and a silent
/// read of the following layer's bytes on Metal. Universal exact-length
/// rule, called by the Metal loader, the CUDA loader, and the convert
/// gate.
pub fn validate_gdn_conv1d(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
    gdn_qkv_rows: usize,
    conv_kernel: usize,
) -> Result<(), String> {
    if st.layer_type != Some(1) {
        return Ok(());
    }
    if let Some(conv) = st.ssm_conv1d.as_ref() {
        if conv.length > 0 {
            let want = (gdn_qkv_rows as u64)
                .checked_mul(conv_kernel as u64)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| {
                    format!(
                        "layer {layer}: ssm_conv1d expectation overflows \
                         ({gdn_qkv_rows} x {conv_kernel} x 4 — malformed \
                         hyperparams)"
                    )
                })?;
            if conv.length != want {
                return Err(format!(
                    "layer {layer}: ssm_conv1d is {} bytes but the GDN \
                     kernels index qkv_rows x conv_kernel = {gdn_qkv_rows} \
                     x {conv_kernel} F32 values = {want} bytes. Re-convert \
                     with `lumen convert`.",
                    conv.length
                ));
            }
        }
    }
    Ok(())
}

/// Exact packed byte length of one weight row of `in_dim` input columns in
/// `quant`. Errors on schemes without a servable fixed layout and on
/// non-block-multiple widths (the dispatch kernels stride rows by
/// `in_dim/32` blocks).
fn attn_row_bytes(
    layer: usize,
    name: &str,
    quant: QuantScheme,
    in_dim: usize,
) -> Result<usize, String> {
    if in_dim == 0 {
        return Err(format!(
            "layer {layer}: {name} has a zero row width (malformed \
             hyperparams) — every expectation would be vacuously zero \
             bytes. Re-convert with `lumen convert`."
        ));
    }
    let row_overflow = || {
        format!(
            "layer {layer}: {name} row width {in_dim} overflows the row-byte \
             computation (malformed hyperparams)"
        )
    };
    match quant {
        QuantScheme::F32 => in_dim.checked_mul(4).ok_or_else(row_overflow),
        QuantScheme::F16 | QuantScheme::Bf16 => in_dim.checked_mul(2).ok_or_else(row_overflow),
        QuantScheme::Q8_0 | QuantScheme::Q4_0 => {
            if in_dim % 32 != 0 {
                return Err(format!(
                    "layer {layer}: {name} is {quant:?} but its row width \
                     {in_dim} is not a multiple of the 32-element block \
                     (malformed hyperparams)"
                ));
            }
            (in_dim / 32)
                .checked_mul(if quant == QuantScheme::Q8_0 { 34 } else { 18 })
                .ok_or_else(row_overflow)
        }
        other => Err(format!(
            "layer {layer}: {name} is {other:?}, which the Metal attention \
             kernels cannot serve. Re-convert with `lumen convert --target \
             metal`."
        )),
    }
}

/// Reject attention projections whose byte length disagrees with the row
/// count the dispatch derives from hyperparams. The routing predicate is
/// tensor PRESENCE (`attn_q_norm`), not architecture — the LBC carries no
/// architecture field — so a converted tensor with the wrong geometry
/// (e.g. a fused Q+gate wq on a layer routed to the single-launch path)
/// loads in-bounds and computes silently wrong output. Kernels take row
/// counts from hyperparams, never from the buffer, so byte length is the
/// one load-time observable that catches the whole class.
pub fn validate_attention_dims(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
    d: &AttnDims,
) -> Result<(), String> {
    validate_mandatory_presence(st).map_err(|e| format!("layer {layer}: {e}"))?;
    let expect = |name: &str,
                  slice: &crate::index::TensorSlice,
                  rows: usize,
                  in_dim: usize|
     -> Result<(), String> {
        let row = attn_row_bytes(layer, name, slice.quant, in_dim)?;
        let want = (rows as u64).checked_mul(row as u64).ok_or_else(|| {
            format!(
                "layer {layer}: {name} expectation overflows ({rows} rows x \
                 {row} bytes/row — malformed hyperparams)"
            )
        })?;
        if slice.length != want {
            return Err(format!(
                "layer {layer}: {name} is {} bytes but the dispatch expects \
                 {rows} rows x {row} bytes = {want} ({:?}, row width {in_dim}). \
                 The kernels derive dimensions from hyperparams, so this \
                 tensor would be read at the wrong geometry. Re-convert with \
                 `lumen convert`.",
                slice.length, slice.quant
            ));
        }
        Ok(())
    };
    if st.layer_type == Some(1) {
        // GDN: wq holds the fused in-projection; wk/wv are converter zero
        // sentinels — a non-empty wk/wv here is malformed.
        expect("attn_qkv (wq)", &st.wq, d.gdn_qkv_rows, d.hidden).map_err(|e| {
            if d.gdn_declared {
                e
            } else {
                format!(
                    "{e} NOTE: the model header declares no GDN dims, so the \
                     expectation is the Qwen3.5-9B compatibility default — \
                     if this is not a 9B-geometry model, re-convert from a \
                     GGUF carrying the ssm.* keys."
                )
            }
        })?;
        if st.wk.length != 0 || st.wv.length != 0 {
            return Err(format!(
                "layer {layer}: GDN layer carries non-empty wk/wv ({} / {} \
                 bytes); the fused in-projection leaves both empty. \
                 Re-convert with `lumen convert`.",
                st.wk.length, st.wv.length
            ));
        }
        validate_gdn_conv1d(layer, st, d.gdn_qkv_rows, d.gdn_conv_kernel)?;
        // ssm_out is the GDN output projection: the dispatch reads
        // hidden rows x gdn_v_dim columns from its offset, deriving both
        // from hyperparams — an index entry with any other length is read
        // at the wrong geometry (or past the declared slice).
        if let Some(out) = st.ssm_out.as_ref() {
            expect("ssm_out", out, d.hidden, d.gdn_v_dim)?;
        }
        return Ok(());
    }
    // Full attention: wq is 2*q_dim rows under Q+gate fusion (detected by
    // per-head Q-norm presence), q_dim rows otherwise; wk/wv are always
    // kv_dim rows — zero-length wk/wv on a full-attention layer is
    // malformed, not a sentinel.
    let wq_rows = if st.attn_q_norm.is_some() {
        2 * d.q_dim
    } else {
        d.q_dim
    };
    expect("wq", &st.wq, wq_rows, d.hidden)?;
    expect("wk", &st.wk, d.kv_dim, d.hidden)?;
    expect("wv", &st.wv, d.kv_dim, d.hidden)?;
    expect("wo", &st.wo, d.hidden, d.q_dim)?;
    validate_attn_vector_extents(layer, st, d.head_dim, d.q_dim, d.kv_dim)?;
    Ok(())
}

/// Reject layer tensors the Metal dispatch paths would misparse, before any
/// Metal path can execute over the blob. Called once per layer at GPU-resident
/// preload, and at zero-copy layer-buffer creation for the streaming /
/// non-resident paths (which never run preload).
pub fn validate_layer_quants(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
) -> Result<(), String> {
    // Reject quant schemes the Metal DENSE decode path has no dispatch
    // kernels for. Without this, a `--target generic` LBC (K-quant or
    // source-fidelity Q4_1 tensors intact, fine on CUDA) runs "successfully"
    // and generates garbage — the dispatch catch-alls feed those bytes to an
    // F32-reading pipeline. Fail loudly with the remedy instead.
    for (name, slice) in st.named_slices() {
        if slice.length == 0 {
            continue;
        }
        if !dense_slice_quant_supported(slice.quant) {
            return Err(format!(
                "layer {layer} tensor '{name}' is {:?}: the Metal \
                 backend has no dense DECODE dispatch kernels for this \
                 quant scheme. This LBC was converted for a different \
                 backend (`--target generic`); re-convert with \
                 `lumen convert --target metal` (K-quant and legacy \
                 Q5_0 layer tensors are upcast to Q8_0, Q4_1 is \
                 re-quantized to Q4_0).",
                slice.quant
            ));
        }
    }
    // The dense-FFN gate+up dispatch arms for Q4_0/F16/Bf16/F32
    // select on the gate's quant alone and bind w_up_off regardless,
    // and the fused shaders stride both pointers with one row_bytes
    // derived from the gate's scheme — a gate/up quant mismatch
    // (producible: the converter upcasts K-quant tensors per tensor)
    // would compute silently wrong output.
    if st.w_gate.length > 0 && st.w_up.length > 0 && st.w_gate.quant != st.w_up.quant {
        return Err(format!(
            "layer {layer}: ffn_gate is {:?} but ffn_up is {:?}: the \
             Metal fused FFN kernels require the pair to share one \
             quant scheme. Re-convert with a uniform quantization \
             (e.g. `lumen convert --target metal --requant q8_0`).",
            st.w_gate.quant, st.w_up.quant
        ));
    }
    // Full-attention layers WITHOUT per-head Q/K norms take the fused
    // single-launch QKV path: one kernel selected on wq's quant reads all
    // qkv_dim rows from wq's offset, so wq/wk/wv must share one quant scheme
    // and be contiguous. (Layers with attn_q_norm use the Q+gate path, which
    // projects K/V separately on their own quants and is exempt.)
    if st.layer_type != Some(1) && st.attn_q_norm.is_none() && st.wk.length > 0 && st.wv.length > 0
    {
        if st.wk.quant != st.wq.quant || st.wv.quant != st.wq.quant {
            return Err(format!(
                "layer {layer}: attn q/k/v quants differ ({:?}/{:?}/{:?}): \
                 the Metal fused QKV launch reads all rows at wq's scheme. \
                 Re-convert from a source GGUF with uniform Q/K/V \
                 quantization (`--requant q8_0` also works for dense models).",
                st.wq.quant, st.wk.quant, st.wv.quant
            ));
        }
        let wq_end = st.wq.offset.checked_add(st.wq.length);
        let wk_end = st.wk.offset.checked_add(st.wk.length);
        if wq_end != Some(st.wk.offset) || wk_end != Some(st.wv.offset) {
            return Err(format!(
                "layer {layer}: attn q/k/v tensors are not contiguous \
                 (malformed LBC): the Metal fused QKV launch reads them as \
                 one span. Re-convert with `lumen convert`."
            ));
        }
    }
    // On GDN layers the decode path pairs attn_gate with the qkv
    // route: the Q8_0 qkv+gate 2-stream kernel (default-on) decodes
    // the gate as Q8_0 without consulting its quant, and the non-Q8
    // routes' gate fallbacks read schemes they lack an arm for as
    // F32 — so a Q8_0/non-Q8_0 split between attn_qkv and
    // attn_gate in either direction computes silently wrong output.
    // Full-attention layers dispatch attn_gate on its own quant and
    // are exempt.
    if let (Some(1), Some(gate)) = (st.layer_type, st.attn_gate.as_ref()) {
        if gate.length > 0
            && ((st.wq.quant == QuantScheme::Q8_0) != (gate.quant == QuantScheme::Q8_0))
        {
            return Err(format!(
                "layer {layer}: attn_qkv is {:?} but attn_gate is {:?}: \
                 the Metal GDN decode path requires the pair to agree \
                 on whether it is Q8_0. Re-convert with \
                 `lumen convert --target metal` (it writes the pair \
                 uniformly).",
                st.wq.quant, gate.quant
            ));
        }
    }
    // On GDN layers an F32 attn_gate takes the decode fallback that reads
    // `normed_buf`, which only the F32 QKV route writes — the fused
    // Q4_0/F16/Bf16 QKV routes RMSNorm inline from `x_buf` and leave
    // `normed_buf` stale, so gate-F32 next to a fused QKV route computes
    // silently wrong output. (Q8_0 QKV + F32 gate is already rejected by
    // the Q8 pairing rule above.)
    if let (Some(1), Some(gate)) = (st.layer_type, st.attn_gate.as_ref()) {
        if gate.length > 0 && gate.quant == QuantScheme::F32 && st.wq.quant != QuantScheme::F32 {
            return Err(format!(
                "layer {layer}: attn_gate is F32 but attn_qkv is {:?}: the \
                 Metal GDN decode F32-gate fallback reads the separate \
                 RMSNorm buffer, which only the F32 QKV route populates. \
                 Re-convert with `lumen convert --target metal` (it writes \
                 the pair uniformly).",
                st.wq.quant
            ));
        }
    }
    validate_ffn_pre_norm(layer, st)?;
    // The GDN prefill projections (attn_qkv in-projection, attn_gate,
    // ssm_out) dispatch on Q8_0/Bf16/Q4_0 arms with a per-token F32
    // fallback loop — there is no F16 arm, so F16 weight bytes would be
    // read as f32, computing silently wrong prefill output (decode has
    // F16 arms; a real run prefills first, so the config is broken
    // end-to-end). Reject F16 on those tensors for GDN layers until the
    // prefill kernels gain F16 coverage.
    if st.layer_type == Some(1) {
        let f16_checked = [
            ("attn_qkv", Some(&st.wq)),
            ("attn_gate", st.attn_gate.as_ref()),
            ("ssm_out", st.ssm_out.as_ref()),
        ];
        for (name, slice) in f16_checked {
            if let Some(sl) = slice {
                if sl.length > 0 && sl.quant == QuantScheme::F16 {
                    return Err(format!(
                        "layer {layer}: {name} is F16: the Metal GDN prefill \
                         projections have no F16 kernel arm and would read \
                         the bytes as F32. Re-convert with `lumen convert \
                         --target metal` (it stores these tensors as Q8_0)."
                    ));
                }
            }
        }
    }
    // The GDN gate pipelines (decode and prefill) decode ssm_alpha/ssm_beta
    // as Q8_0 unconditionally (offsets are bound without consulting their
    // quant). The converter stores them as Q8_0 on every Metal-target path,
    // but source-fidelity conversions for other targets keep F32 gates, and
    // `--dequantize` writes F32 when the source stores them as Q8_0 —
    // these pipelines would parse those bytes as Q8_0 blocks, computing
    // silently wrong output.
    for (name, slice) in [("ssm_alpha", &st.ssm_alpha), ("ssm_beta", &st.ssm_beta)] {
        if let Some(s) = slice {
            if s.length > 0 && s.quant != QuantScheme::Q8_0 {
                return Err(format!(
                    "layer {layer}: {name} is {:?}: the Metal GDN gate \
                     pipelines read ssm_alpha/ssm_beta as Q8_0. Re-convert \
                     with `lumen convert --target metal` (it stores these \
                     tensors as Q8_0).",
                    s.quant
                ));
            }
        }
    }
    // The MoE expert dispatch has the same shape: it selects the
    // fused gate+up pipeline on the expert gate's quant alone
    // (CachedMoeMeta carries no up quant), so a per-expert gate/up
    // mismatch would also compute silently wrong output.
    if let Some(experts) = st.experts.as_ref() {
        for (i, e) in experts.iter().enumerate() {
            if e.gate.length > 0 && e.up.length > 0 && e.gate.quant != e.up.quant {
                return Err(format!(
                    "layer {layer} expert {i}: gate is {:?} but up is {:?}: \
                     the Metal fused expert FFN kernels require the pair \
                     to share one quant scheme. Re-convert with \
                     `lumen convert --target metal` from a source GGUF \
                     whose expert tensors share one quantization.",
                    e.gate.quant, e.up.quant
                ));
            }
        }
        // Expert dispatch takes its quant schemes from expert 0 alone
        // (CachedMoeMeta carries one scheme set for the whole bank), so a
        // layer whose experts differ from expert 0 would decode the others
        // at the wrong stride. The format stores per-expert schemes and does
        // not enforce uniformity; reject the divergence here.
        if let Some(first) = experts.first() {
            for (i, e) in experts.iter().enumerate().skip(1) {
                let pairs = [
                    ("gate", e.gate.quant, first.gate.quant),
                    ("up", e.up.quant, first.up.quant),
                    ("down", e.down.quant, first.down.quant),
                ];
                if let Some((name, got, want)) = pairs.iter().find(|(_, a, b)| a != b).copied() {
                    return Err(format!(
                        "layer {layer} expert {i}: {name} is {got:?} but \
                         expert 0's is {want:?}: the Metal expert dispatch \
                         applies expert 0's quant schemes to every expert. \
                         Re-convert from a source GGUF whose experts share \
                         one quantization.",
                    ));
                }
            }
        }
    }
    // A `Some` with length 0 passes every quant check above as "absent" and
    // would then be bound (or unwrapped) at dispatch.
    validate_layer_slices(layer, st)?;
    // The shared-expert FFN dispatch selects on the gate's quant alone and
    // binds up_off regardless (CachedMoeMeta carries no shared-expert up
    // quant): the Q8_0/Q4_0 arms select fused gate+up+SwiGLU shaders, and
    // the F16/Bf16/F32 arms take the separate gate/up matmul + barrier +
    // SwiGLU fallback. Both require the gate/up pair to share the quant the
    // dispatch selected; down projects independently on its own quant.
    let shexp_present = [
        st.shared_expert_gate.as_ref(),
        st.shared_expert_up.as_ref(),
        st.shared_expert_down.as_ref(),
    ]
    .map(|t| t.is_some_and(|s| s.length > 0));
    if shexp_present.iter().any(|&p| p) && !shexp_present.iter().all(|&p| p) {
        // The runtime uses the gate's presence as the shared-expert feature
        // flag and unwraps the other two tensors during dispatch.
        return Err(format!(
            "layer {layer}: incomplete shared-expert tensors (gate/up/down \
             present: {shexp_present:?}): a shared expert requires all \
             three. Re-convert with `lumen convert --target metal`."
        ));
    }
    if let (Some(gate), Some(up)) = (st.shared_expert_gate.as_ref(), st.shared_expert_up.as_ref()) {
        if gate.length > 0 && up.length > 0 {
            if gate.quant != up.quant {
                return Err(format!(
                    "layer {layer}: shared-expert gate is {:?} but up is {:?}: \
                     the Metal fused shared-expert kernels require the pair to \
                     share one quant scheme. Re-convert with \
                     `lumen convert --target metal`.",
                    gate.quant, up.quant
                ));
            }
            // The down projection dispatches independently on its own
            // quant (it has arms for every servable scheme), so only the
            // gate/up pair is constrained: uniform, and on a scheme the
            // fused kernels or the float fallback serve.
            if !matches!(
                gate.quant,
                QuantScheme::Q8_0
                    | QuantScheme::Q4_0
                    | QuantScheme::F16
                    | QuantScheme::Bf16
                    | QuantScheme::F32
            ) {
                return Err(format!(
                    "layer {layer}: shared-expert gate/up is {:?}: the Metal \
                     shared-expert FFN serves Q8_0/Q4_0 (fused) and \
                     F16/Bf16/F32 (separate matmuls). Re-convert with \
                     `lumen convert --target metal` (it quantizes the \
                     shared-expert tensors).",
                    gate.quant
                ));
            }
        }
    }
    let bias_present = [st.bq.is_some(), st.bk.is_some(), st.bv.is_some()];
    // The Q+gate attention path (layers with per-head Q/K norms) has no bias
    // handling at all — a bias there would be silently ignored, not dropped
    // partially.
    if bias_present.iter().any(|&p| p) && st.layer_type != Some(1) && st.attn_q_norm.is_some() {
        return Err(format!(
            "layer {layer}: QKV biases are present on a Q+gate attention \
             layer, which does not apply biases. Re-convert from a source \
             GGUF without attention biases."
        ));
    }
    // Every Metal shader reads norm tensors, the MoE routers, the SSM
    // scalar tensors, and the QKV biases as F32 without consulting their
    // quant (CUDA rejects non-F32 norms at load; Metal must too). The
    // converter writes them F32 on its forced paths, but a source GGUF
    // storing e.g. F16 norms passes the allowlist above and would be
    // misread.
    let f32_only: [(&str, Option<&TensorSlice>); 14] = [
        ("bq", st.bq.as_ref()),
        ("bk", st.bk.as_ref()),
        ("bv", st.bv.as_ref()),
        ("attn_norm", Some(&st.attn_norm)),
        ("ffn_norm", Some(&st.ffn_norm)),
        ("attn_post_norm", st.attn_post_norm.as_ref()),
        ("attn_q_norm", st.attn_q_norm.as_ref()),
        ("attn_k_norm", st.attn_k_norm.as_ref()),
        ("ssm_norm", st.ssm_norm.as_ref()),
        ("ssm_a", st.ssm_a.as_ref()),
        ("ssm_conv1d", st.ssm_conv1d.as_ref()),
        ("ssm_dt", st.ssm_dt.as_ref()),
        ("router_weight", st.router_weight.as_ref()),
        ("ffn_gate_inp_shexp", st.ffn_gate_inp_shexp.as_ref()),
    ];
    for (name, slice) in f32_only {
        if let Some(s) = slice {
            if s.length > 0 && s.quant != QuantScheme::F32 {
                return Err(format!(
                    "layer {layer}: {name} is {:?}: the Metal kernels read \
                     this tensor as F32. Re-convert from a source GGUF whose \
                     norm/router/SSM-scalar tensors are F32.",
                    s.quant
                ));
            }
        }
    }
    Ok(())
}

/// Enforce projection geometry for every fixed-layout scheme: the launchers
/// derive row counts from hyperparams, so a slice whose byte length decodes
/// to any other row count is read at the wrong geometry — in-bounds, silent
/// garbage. Zero-length slices are converter absence sentinels (wo and
/// wk/wv on GDN layers, dense FFN tensors on MoE layers) — presence is
/// validated where the role demands it; geometry only applies to tensors
/// that exist. Floats (F32/F16/Bf16) and every block scheme the CUDA
/// upload path accepts (Q8_0/Q4_0/Q4_1/Q5_0 and the K-quants, which reach
/// this path verbatim under `--target cuda`/generic — the Q8_0 upcast is
/// Metal-target-only) are covered here; CtInt4G32 is enforced in
/// `upload_projection_tensor`'s ct4 branch.
pub fn validate_projection_geometry(
    name: &str,
    slice: &crate::index::TensorSlice,
    in_dim: usize,
    allowed_out: &[usize],
) -> Result<(), String> {
    // (block elements, block bytes) per fixed-layout scheme. A width that
    // does not divide into whole blocks is malformed row geometry — the
    // kernels truncate to in_dim/block blocks per row — so it must FAIL,
    // never fall through to a skip (Metal's `attn_row_bytes` twin behaves
    // the same way).
    let block = match slice.quant {
        QuantScheme::F32 => Some((1usize, 4usize)),
        QuantScheme::F16 | QuantScheme::Bf16 => Some((1, 2)),
        QuantScheme::Q8_0 => Some((32, 34)),
        QuantScheme::Q4_0 => Some((32, 18)),
        QuantScheme::Q4_1 => Some((32, 20)),
        QuantScheme::Q5_0 => Some((32, 22)),
        QuantScheme::Q2_K => Some((256, 84)),
        QuantScheme::Q3_K => Some((256, 110)),
        QuantScheme::Q4_K => Some((256, 144)),
        QuantScheme::Q5_K => Some((256, 176)),
        QuantScheme::Q6_K => Some((256, 210)),
        _ => None,
    };
    let row_bytes = match block {
        Some((elems, bytes)) => {
            if in_dim == 0 {
                return Err(format!("{name} role has in_dim 0 (malformed hyperparams)."));
            }
            if in_dim % elems != 0 {
                return Err(format!(
                    "{name} is {:?} but in_dim {in_dim} is not a multiple \
                     of the {elems}-element block — the kernels would \
                     truncate the row (malformed hyperparams).",
                    slice.quant
                ));
            }
            Some(in_dim / elems * bytes)
        }
        None => None,
    };
    if let (Some(row), true) = (row_bytes, slice.length > 0) {
        let len = slice.length as usize;
        let ok = len % row == 0 && allowed_out.contains(&(len / row));
        if !ok {
            return Err(format!(
                "{name} is {len} bytes ({:?}, in_dim {in_dim}, {row} \
                 bytes/row) but this role requires out_dim in \
                 {allowed_out:?}. The kernels derive dimensions from \
                 hyperparams, so this tensor would be read at the wrong \
                 geometry. Re-convert with `lumen convert`.",
                slice.quant
            ));
        }
    }
    Ok(())
}

/// Exact byte extents for the F32 attention vectors the kernels read at
/// hyperparam-derived lengths: per-head Q/K norms at `head_dim` values
/// per head (CUDA's `rmsnorm_per_head_inplace` indexes
/// `weight[0..head_dim]` unconditionally — a short buffer is an
/// out-of-bounds read on CUDA's per-tensor allocation, and on Metal a
/// read of adjacent layer-blob bytes as weights), and QKV biases at
/// `q_dim`/`kv_dim` values.
/// Full-attention layers only; every tensor here is optional and
/// F32-enforced by the existing scheme rules.
pub fn validate_attn_vector_extents(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<(), String> {
    if st.layer_type == Some(1) {
        return Ok(());
    }
    let expect = |name: &str,
                  slice: &Option<crate::index::TensorSlice>,
                  elems: usize|
     -> Result<(), String> {
        if let Some(s) = slice.as_ref() {
            let want = (elems as u64).checked_mul(4).ok_or_else(|| {
                format!(
                    "layer {layer}: {name} expectation overflows ({elems} \
                     F32 values — malformed hyperparams)"
                )
            })?;
            if s.length != want {
                return Err(format!(
                    "layer {layer}: {name} is {} bytes but the dispatch \
                     reads {elems} F32 values = {want} bytes (malformed \
                     LBC). Re-convert with `lumen convert`.",
                    s.length
                ));
            }
        }
        Ok(())
    };
    expect("attn_q_norm", &st.attn_q_norm, head_dim)?;
    expect("attn_k_norm", &st.attn_k_norm, head_dim)?;
    expect("bq", &st.bq, q_dim)?;
    expect("bk", &st.bk, kv_dim)?;
    expect("bv", &st.bv, kv_dim)?;
    Ok(())
}

/// Zero-length mandatory tensors would suppress the geometry checks (zero
/// skips as an absence sentinel), so presence is enforced for every tensor
/// whose role demands it: wq always; wk/wv/wo on full attention (GDN's
/// fused in-projection legitimately leaves them empty); the dense FFN trio
/// on non-MoE layers (MoE layers route through experts and leave dense
/// sentinels).
pub fn validate_mandatory_presence(subs: &crate::index::SubtensorOffsets) -> Result<(), String> {
    if subs.wq.length == 0 {
        return Err("empty wq: every layer requires an attention \
             (or GDN in-) projection (malformed LBC). Re-convert with \
             `lumen convert`."
            .into());
    }
    if subs.layer_type != Some(1)
        && (subs.wk.length == 0 || subs.wv.length == 0 || subs.wo.length == 0)
    {
        return Err(format!(
            "full-attention layer carries empty wk/wv/wo ({} / {} / {} \
             bytes); the attention dispatch requires all three (malformed \
             LBC). Re-convert with `lumen convert`.",
            subs.wk.length, subs.wv.length, subs.wo.length
        ));
    }
    // The dispatch predicates diverge on a half pair: a lone Q-norm is
    // applied by CUDA and by Metal PREFILL but ignored by Metal decode
    // (which gates on both being present) — the same artifact silently
    // produces different output across backends AND between Metal's own
    // prefill and decode; a lone K-norm is ignored everywhere (CUDA
    // nests both norm blocks under Q-norm presence). No served
    // semantics exist for either half. One policy, fail closed (the N3
    // precedent).
    if subs.layer_type != Some(1) && subs.attn_q_norm.is_some() != subs.attn_k_norm.is_some() {
        return Err(format!(
            "half per-head Q/K norm pair (attn_q_norm {}, attn_k_norm {}); \
             the backends require both or neither (malformed LBC). \
             Re-convert with `lumen convert`.",
            if subs.attn_q_norm.is_some() {
                "present"
            } else {
                "absent"
            },
            if subs.attn_k_norm.is_some() {
                "present"
            } else {
                "absent"
            }
        ));
    }
    // Q+gate attention (per-head Q-norm present) applies no QKV biases on
    // either backend — Metal has rejected the combination since v0.17.0;
    // CUDA now agrees instead of silently taking a bias-applying unfused
    // path Metal cannot mirror (round-7 N3: one policy, both backends).
    if subs.layer_type != Some(1)
        && subs.attn_q_norm.is_some()
        && (subs.bq.is_some() || subs.bk.is_some() || subs.bv.is_some())
    {
        return Err(
            "QKV biases are present on a Q+gate attention layer (per-head \
             Q/K norms), which applies no biases (malformed LBC). \
             Re-convert with `lumen convert`."
                .into(),
        );
    }
    // An incomplete QKV bias set with quantized projections serves
    // divergently: CUDA applies each present bias independently while
    // Metal's fused bias arms apply biases only when all three exist —
    // and drops a partial set silently. With F32 projections both
    // backends apply each bias independently, so partial sets stay
    // valid there.
    let bias_present = [subs.bq.is_some(), subs.bk.is_some(), subs.bv.is_some()];
    let qkv_all_f32 = subs.wq.quant == QuantScheme::F32
        && subs.wk.quant == QuantScheme::F32
        && subs.wv.quant == QuantScheme::F32;
    if subs.layer_type != Some(1)
        && bias_present.iter().any(|&p| p)
        && !bias_present.iter().all(|&p| p)
        && !qkv_all_f32
    {
        return Err(format!(
            "incomplete QKV bias set (bq/bk/bv present: {bias_present:?}) \
             with non-F32 projections: the backends diverge on partial \
             sets (CUDA applies each present bias; Metal's fused decode \
             arms drop them), so this fails closed (malformed LBC). \
             Re-convert with `lumen convert`."
        ));
    }
    // A layer routes through experts only when BOTH the router and a
    // non-empty expert bank exist (the runtime's own MoE predicate);
    // router-without-experts or experts-without-router is malformed, and a
    // half-declared MoE layer must not exempt the dense FFN requirement.
    let has_experts = subs.experts.as_ref().is_some_and(|e| !e.is_empty());
    let is_moe = subs.router_weight.is_some() && has_experts;
    if subs.router_weight.is_some() != has_experts {
        return Err(format!(
            "layer declares router_weight={} but expert bank {} (malformed \
             LBC — MoE requires both). Re-convert with `lumen convert`.",
            subs.router_weight.is_some(),
            if has_experts {
                "present"
            } else {
                "absent/empty"
            }
        ));
    }
    if !is_moe && (subs.w_gate.length == 0 || subs.w_up.length == 0 || subs.w_down.length == 0) {
        return Err(format!(
            "non-MoE layer carries empty dense FFN tensors (gate/up/down = \
             {} / {} / {} bytes; malformed LBC). Re-convert with `lumen \
             convert`.",
            subs.w_gate.length, subs.w_up.length, subs.w_down.length
        ));
    }
    Ok(())
}

/// Reject present-but-zero-length optional tensors before any upload: the
/// upload and dispatch paths gate on presence alone, so a `Some` with
/// length 0 would become a "present" GPU buffer the kernels then read
/// past. The converter never emits one.
pub fn validate_layer_slices(
    layer: usize,
    subs: &crate::index::SubtensorOffsets,
) -> Result<(), String> {
    let fields = subs.slice_fields();
    for (name, slice) in fields.optional {
        if let Some(t) = slice {
            if t.length == 0 {
                return Err(format!(
                    "layer {layer}: {name} is present but zero-length \
                     (malformed LBC). Re-convert with `lumen convert`."
                ));
            }
        }
    }
    if let Some(experts) = fields.experts.as_ref() {
        for (i, e) in experts.iter().enumerate() {
            for (name, t) in [("gate", &e.gate), ("up", &e.up), ("down", &e.down)] {
                if t.length == 0 {
                    return Err(format!(
                        "layer {layer} expert {i}: {name} is zero-length \
                         (malformed LBC). Re-convert with `lumen convert`."
                    ));
                }
            }
        }
    }

    Ok(())
}

/// The FFN pre-norm must exist somewhere: ffn_norm is legitimately a zero
/// sentinel when attn_post_norm carries the pre-norm (GDN / MoE layers),
/// but with BOTH absent the loaders' fallbacks misbehave — Metal resolves
/// the norm offset to 0 and reads whatever tensor lives there as F32; CUDA
/// uploads a zero-length "present" norm buffer the rmsnorm kernel indexes
/// past. Universal: both loaders enforce it.
pub fn validate_ffn_pre_norm(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
) -> Result<(), String> {
    if st.ffn_norm.length == 0 && !st.attn_post_norm.as_ref().is_some_and(|s| s.length > 0) {
        return Err(format!(
            "layer {layer}: ffn_norm is empty and attn_post_norm is absent \
             — no FFN pre-norm exists (malformed LBC). Re-convert with \
             `lumen convert`."
        ));
    }
    Ok(())
}

/// Expert-bank uniformity, mirrored from the CUDA loader's
/// `build_moe_meta` (Metal enforces the same rules): the dispatch applies
/// expert 0's schemes to the whole bank and selects the fused gate+up
/// kernel from the gate's scheme alone. Universal.
pub fn validate_expert_bank(st: &crate::index::SubtensorOffsets) -> Result<(), String> {
    let Some(experts) = st.experts.as_ref() else {
        return Ok(());
    };
    let Some(first) = experts.first() else {
        return Ok(());
    };
    for (i, e) in experts.iter().enumerate() {
        if e.gate.quant != e.up.quant {
            return Err(format!(
                "expert {i}: gate is {:?} but up is {:?}: the fused expert \
                 kernels require the pair to share one quant scheme. \
                 Re-convert from a source GGUF whose expert tensors share \
                 one quantization.",
                e.gate.quant, e.up.quant
            ));
        }
        let pairs = [
            ("gate", e.gate.quant, first.gate.quant),
            ("up", e.up.quant, first.up.quant),
            ("down", e.down.quant, first.down.quant),
        ];
        if let Some((name, got, want)) = pairs.iter().find(|(_, a, b)| a != b).copied() {
            return Err(format!(
                "expert {i}: {name} is {got:?} but expert 0's is {want:?}: \
                 the expert dispatch applies expert 0's quant schemes to \
                 every expert. Re-convert from a source GGUF whose experts \
                 share one quantization.",
            ));
        }
    }
    Ok(())
}

/// A MoE layer's expert bank must hold exactly the header's declared
/// expert count. The runtime sizes GPU offset tables and dispatch grids
/// from the header count but fills only `min(header, bank.len())` entries
/// — a header claiming MORE experts than the bank leaves the surplus
/// pointing at offset 0, so those experts silently route to the first
/// tensor's bytes (wrong-weight output, no crash). Enforced at load on
/// both backends; non-MoE layers are exempt.
pub fn validate_expert_count(
    st: &crate::index::SubtensorOffsets,
    expected: usize,
) -> Result<(), String> {
    let has_experts = st.experts.as_ref().is_some_and(|e| !e.is_empty());
    if st.router_weight.is_none() || !has_experts {
        return Ok(());
    }
    let actual = st.experts.as_ref().map_or(0, |e| e.len());
    if actual != expected {
        return Err(format!(
            "MoE layer carries {actual} experts but the model header \
             declares {expected}; the dispatch grid is sized from the \
             header, so a mismatch routes surplus experts to offset 0 \
             (malformed LBC). Re-convert with `lumen convert`."
        ));
    }
    Ok(())
}

/// Composite layer-plan validation: everything the loaders will check at
/// load time, runnable over a PLANNED `SubtensorOffsets` before any byte
/// is written. `metal` adds the Metal-only rules (scheme allowlist, pair
/// rules, attention byte-geometry); presence and projection geometry
/// mirror the CUDA loader and apply to every target. This is the
/// converter's post-planning gate — the round-7 structural fix: the same
/// functions the loaders call, over the same written schemes, instead of
/// convert-side shadow mappings of source types.
pub fn validate_layer_plan(
    layer: usize,
    st: &crate::index::SubtensorOffsets,
    d: &AttnDims,
    inter: usize,
    num_experts: usize,
    metal: bool,
) -> Result<(), String> {
    validate_layer_slices(layer, st)?;
    validate_mandatory_presence(st).map_err(|e| format!("layer {layer}: {e}"))?;
    validate_attn_vector_extents(layer, st, d.head_dim, d.q_dim, d.kv_dim)?;
    validate_ffn_pre_norm(layer, st)?;
    validate_expert_bank(st)?;
    validate_expert_count(st, num_experts).map_err(|e| format!("layer {layer}: {e}"))?;
    validate_gdn_conv1d(layer, st, d.gdn_qkv_rows, d.gdn_conv_kernel)?;
    let is_gdn = st.layer_type == Some(1);
    let wq_rows = if is_gdn {
        d.gdn_qkv_rows
    } else if st.attn_q_norm.is_some() {
        2 * d.q_dim
    } else {
        d.q_dim
    };
    validate_projection_geometry("wq", &st.wq, d.hidden, &[wq_rows])?;
    validate_projection_geometry("wk", &st.wk, d.hidden, &[d.kv_dim])?;
    validate_projection_geometry("wv", &st.wv, d.hidden, &[d.kv_dim])?;
    validate_projection_geometry("wo", &st.wo, d.q_dim, &[d.hidden])?;
    // Dense FFN geometry applies when the tensors exist (zero-length MoE
    // sentinels skip inside validate_projection_geometry).
    validate_projection_geometry("w_gate", &st.w_gate, d.hidden, &[inter])?;
    validate_projection_geometry("w_up", &st.w_up, d.hidden, &[inter])?;
    validate_projection_geometry("w_down", &st.w_down, inter, &[d.hidden])?;
    // attn_gate: the CUDA loader validates [hidden -> gdn_v_dim]
    // unconditionally when present (v_dim == q_dim on every Qwen3.5
    // model by design). Universal.
    if let Some(gate) = st.attn_gate.as_ref() {
        validate_projection_geometry("attn_gate", gate, d.hidden, &[d.gdn_v_dim])?;
    }
    // ssm_out: the GDN output projection maps gdn_v_dim -> hidden (hidden
    // rows x gdn_v_dim width). Validate its geometry for EVERY target, not
    // only Metal: a `--target generic` conversion sizes ssm_out straight
    // from the source's element count, so a malformed-source GGUF would
    // otherwise emit a wrong-geometry ssm_out that no convert-time check
    // catches (the Metal load guard would, but CUDA/CPU would read it at the
    // wrong geometry). Well-formed sources are unaffected.
    if let Some(out) = st.ssm_out.as_ref() {
        validate_projection_geometry("ssm_out", out, d.gdn_v_dim, &[d.hidden])?;
    }
    if metal {
        validate_layer_quants(layer, st)?;
        validate_attention_dims(layer, st, d)?;
    }
    Ok(())
}
