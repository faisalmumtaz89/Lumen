//! The layer oracle: an independent host implementation of one whole native prefill layer at the
//! route's rounding points, fed a layer's dump (`native_prefill_dump`) and the real layer
//! weights. A dumped layer is checked two ways.
//!
//! **Step by step.** Each step is recomputed from the dump of its own inputs:
//! - the producers (norms, quantizers, SwiGLU, sigmoid gate) byte for byte, by the producers' host
//!   implementation (`producers`);
//! - the GEMMs as the f64 sum over the same codes and scales, scaled as the GEMM scales them, within
//!   `|D - R| <= rel max(|R|, |D|) + 2^-17 M` (M the sum of the products' magnitudes): rel = 2^-8 for
//!   a BF16 output, the bound every selected plan is verified to (one rounding, and a different F32
//!   summation order), and 2^-22 for the F32 output of the `a`/`b` GEMM (a cuBLAS `GemmEx`);
//! - the GDN chain by the GDN models' gate (`gdn::gate`): output and state against the f64 oracle and
//!   the rounded emulation, the conv output per element, the ring bit for bit; and the ring position;
//! - the last row handed to decode (`x_gpu`), when dumped, bit for bit.
//!
//! **Whole layer.** From the layer's input alone (its input rows, residual, ring and state) the layer
//! is evaluated twice at the same rounding points. E1 takes every GEMM's exact sum and solves the
//! in-chunk systems in f64. E2 moves every GEMM's exact sum by a pseudo-random amount within the
//! accumulation term of that bound (`2^-17 M`) before the output rounding, and solves in F32 as the
//! kernels do: one evaluation whose GEMMs meet their verified bound (a moved sum flips a BF16
//! rounding, which can flip a quantized code downstream). The dumped outputs (MLP
//! output, residual, ring, state, KV rows) must lie within 1.5x the E1-E2 distance of E1 in relative
//! L2, and never need to be closer than 2^-9, about the root-mean-square error of rounding a tensor
//! to BF16 once (between 2^-8 and 2^-7 relative per unit in the last place, divided by the square
//! root of 12). For the F32 state that floor admits a BF16-sized difference; its step checks bound
//! it far more tightly. On the real model's MLP output the E1-E2 distance is several percent (the
//! FP4 codes flip), so there the whole-layer check catches only gross errors; the step checks bound
//! each GEMM and quantizer tightly.
//!
//! Attention layers: the prep (q/k norm, RoPE, KV rows) and the attention are checked, and evaluated
//! for the whole layer, by an [`AttentionCore`]: [`Attention`], from the attention suite's host
//! models. Without one, the layer's `attention_core` and `layer` checks fail; every other step is
//! checked.
//!
//! [`handoff`] checks that a layer's inputs are the previous layer's outputs, and a slice's state
//! the previous slice's.

use super::attn;
use super::gdn;
use super::producers as p;
use lumen_format::index::{SubtensorOffsets, TensorSlice};
use lumen_runtime::cuda::native_prefill::{SliceSource, ROPE};
use lumen_runtime::cuda::native_prefill_dump::PrefillDump;
use lumen_runtime::cuda::native_prefill_weights::swizzled_offset;

pub const H: usize = p::H;
pub const I: usize = p::I;
/// GDN value heads and their rows of 128 in the gated norm.
pub const GDN_HEADS: usize = gdn::H;
/// Width of the attention output and its gate.
pub const ATTN_OUT: usize = 6144;

// ---------------------------------------------------------------------------------------------
// Weights.

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn e2m1_dec(c: u8) -> f32 {
    let v = p::E2M1[(c & 7) as usize];
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

/// An FP8 projection `[n][k]`: its planes as stored (codes, weight scale, activation scale) and the
/// decoded codes.
pub struct Fp8Weight {
    pub n: usize,
    pub k: usize,
    pub planes: Vec<u8>,
    pub scale: f32,
    pub input_scale: f32,
    dec: Vec<f32>,
}

impl Fp8Weight {
    fn load(src: &dyn SliceSource, l: usize, s: &TensorSlice, n: usize, k: usize) -> Self {
        let planes = src.read(l, s, 0, s.length).unwrap();
        assert_eq!(planes.len(), n * k + 8, "layer {l}: FP8 planes [{n}][{k}]");
        let tail = f32s(&planes[n * k..]);
        let dec = planes[..n * k].iter().map(|&c| p::e4m3_dec(c)).collect();
        Self {
            n,
            k,
            scale: tail[0],
            input_scale: tail[1],
            planes,
            dec,
        }
    }
}

/// An NVFP4 projection `[n][k]`: its planes as stored (codes, linear block scales, global scale,
/// activation scale) and the codes decoded with their block scales.
pub struct Fp4Weight {
    pub n: usize,
    pub k: usize,
    pub planes: Vec<u8>,
    pub global: f32,
    pub input_scale: f32,
    dec: Vec<f32>,
}

impl Fp4Weight {
    fn load(src: &dyn SliceSource, l: usize, s: &TensorSlice, n: usize, k: usize) -> Self {
        let planes = src.read(l, s, 0, s.length).unwrap();
        let (codes, blocks) = (n * k / 2, n * k / 16);
        assert_eq!(
            planes.len(),
            codes + blocks + 8,
            "layer {l}: NVFP4 planes [{n}][{k}]"
        );
        let tail = f32s(&planes[codes + blocks..]);
        let dec = (0..n * k)
            .map(|e| {
                let c = planes[e / 2] >> (4 * (e % 2)) & 15;
                e2m1_dec(c) * p::e4m3_dec(planes[codes + e / 16])
            })
            .collect();
        Self {
            n,
            k,
            global: tail[0],
            input_scale: tail[1],
            planes,
            dec,
        }
    }

    /// The linear block scales `[n][k / 16]`.
    pub fn block_scales(&self) -> &[u8] {
        let codes = self.n * self.k / 2;
        &self.planes[codes..codes + self.n * self.k / 16]
    }
}

/// A GDN layer's own tensors.
pub struct GdnWeights {
    /// `in_proj_a` then `in_proj_b` rows, BF16 `[96][5120]`.
    pub ab: Vec<u16>,
    pub conv_w: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub norm: Vec<f32>,
}

/// One layer's weights as the artifact stores them.
pub struct LayerWeights {
    pub layer: usize,
    pub attention: bool,
    /// The input norm and the post-attention norm, F32 `(w + 1)`.
    pub norm_in: Vec<f32>,
    pub norm_post: Vec<f32>,
    /// The FP8 input projections sharing the normed rows: GDN qkv and z, or attention q (with its
    /// gate), k and v.
    pub inputs: Vec<Fp8Weight>,
    /// The FP8 output projection (GDN out_proj or attention o_proj).
    pub out: Fp8Weight,
    pub gate: Fp4Weight,
    pub up: Fp4Weight,
    pub down: Fp4Weight,
    pub gdn: Option<GdnWeights>,
    /// An attention layer's per-head q and k norm weights, F32 `[256]`.
    pub qk_norm: Option<(Vec<f32>, Vec<f32>)>,
}

impl LayerWeights {
    /// Layer `l` of the admitted structure.
    pub fn load(src: &dyn SliceSource, l: usize, st: &SubtensorOffsets) -> Self {
        let attention = st.layer_type == Some(0);
        let whole = |s: &TensorSlice| f32s(&src.read(l, s, 0, s.length).unwrap());
        let (inputs, out) = if attention {
            (
                vec![
                    Fp8Weight::load(src, l, &st.wq, 12288, H),
                    Fp8Weight::load(src, l, &st.wk, 1024, H),
                    Fp8Weight::load(src, l, &st.wv, 1024, H),
                ],
                Fp8Weight::load(src, l, &st.wo, H, ATTN_OUT),
            )
        } else {
            (
                vec![
                    Fp8Weight::load(src, l, &st.wq, gdn::CONV, H),
                    Fp8Weight::load(src, l, st.attn_gate.as_ref().unwrap(), 6144, H),
                ],
                Fp8Weight::load(src, l, st.ssm_out.as_ref().unwrap(), H, 6144),
            )
        };
        let gdn = (!attention).then(|| {
            let bf16_rows = |s: &TensorSlice| -> Vec<u16> {
                whole(s)
                    .into_iter()
                    .map(|v| {
                        assert_eq!(v.to_bits() & 0xFFFF, 0, "layer {l}: a/b not BF16");
                        (v.to_bits() >> 16) as u16
                    })
                    .collect()
            };
            let mut ab = bf16_rows(st.ssm_alpha.as_ref().unwrap());
            ab.extend(bf16_rows(st.ssm_beta.as_ref().unwrap()));
            GdnWeights {
                ab,
                conv_w: whole(st.ssm_conv1d.as_ref().unwrap()),
                dt_bias: whole(st.ssm_dt.as_ref().unwrap()),
                ssm_a: whole(st.ssm_a.as_ref().unwrap()),
                norm: whole(st.ssm_norm.as_ref().unwrap()),
            }
        });
        let qk_norm = attention.then(|| {
            (
                whole(st.attn_q_norm.as_ref().unwrap()),
                whole(st.attn_k_norm.as_ref().unwrap()),
            )
        });
        Self {
            layer: l,
            attention,
            norm_in: whole(&st.attn_norm),
            norm_post: whole(st.attn_post_norm.as_ref().unwrap()),
            inputs,
            out,
            gate: Fp4Weight::load(src, l, &st.w_gate, I, H),
            up: Fp4Weight::load(src, l, &st.w_up, I, H),
            down: Fp4Weight::load(src, l, &st.w_down, H, I),
            gdn,
            qk_norm,
        }
    }

    /// The activation scale of the input projections: their members' maximum.
    pub fn proj_in_scale(&self) -> f32 {
        self.inputs
            .iter()
            .map(|w| w.input_scale)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// The activation scale of gate and up: their maximum.
    pub fn gate_up_scale(&self) -> f32 {
        self.gate.input_scale.max(self.up.input_scale)
    }
}

// ---------------------------------------------------------------------------------------------
// GEMMs.

/// Which evaluation a GEMM result belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Eval {
    /// The exact sum: the products are exact in F32 and summed in f64 (within (k - 1) 2^-53 of M).
    E1,
    /// The exact sum moved by up to `2^-17 M`, pseudo-randomly; `salt` tells GEMMs apart.
    E2 { salt: u64 },
}

/// A uniform value in [-1, 1) from an index and a salt.
fn unit(i: u64, salt: u64) -> f64 {
    let mut z = i
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// `x` `[t][k]` times `w` `[n][k]`^T: each element's sum in f64 and its magnitude `sum |x w|`, both
/// `[t][n]`. Every product of the formats used here is exact in F32.
fn gemm(x: &[f32], t: usize, w: &[f32], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    const CHUNK: usize = 64;
    let parts = gdn::par_map(n.div_ceil(CHUNK), |c| {
        let cols = c * CHUNK..(c * CHUNK + CHUNK).min(n);
        let mut out = Vec::with_capacity(cols.len() * t * 2);
        for j in cols {
            let wr = &w[j * k..(j + 1) * k];
            for i in 0..t {
                let xr = &x[i * k..(i + 1) * k];
                let (mut s, mut m) = (0.0f64, 0.0f64);
                for (&a, &b) in xr.iter().zip(wr) {
                    let prod = (a * b) as f64;
                    s += prod;
                    m += prod.abs();
                }
                out.push(s);
                out.push(m);
            }
        }
        out
    });
    let (mut r, mut mag) = (vec![0.0; t * n], vec![0.0; t * n]);
    for (c, part) in parts.into_iter().enumerate() {
        for (e, pair) in part.chunks_exact(2).enumerate() {
            let (j, i) = (c * CHUNK + e / t, e % t);
            r[i * n + j] = pair[0];
            mag[i * n + j] = pair[1];
        }
    }
    (r, mag)
}

/// A GEMM's result before its output rounding: the value `r` and the magnitude `m`, `[t][n]`.
pub struct GemmOut {
    pub r: Vec<f64>,
    pub m: Vec<f64>,
}

impl GemmOut {
    /// The sums scaled by `scale` (E2's moved within the accumulation term).
    fn new(r: Vec<f64>, m: Vec<f64>, scale: f64, eval: Eval) -> Self {
        let m: Vec<f64> = m.into_iter().map(|v| v * scale.abs()).collect();
        let r = r
            .into_iter()
            .zip(&m)
            .enumerate()
            .map(|(i, (v, &mi))| match eval {
                Eval::E1 => v * scale,
                Eval::E2 { salt } => v * scale + 2f64.powi(-17) * mi * unit(i as u64, salt),
            })
            .collect();
        Self { r, m }
    }

    /// The BF16 output: the value in F32, rounded to BF16.
    pub fn bf16(&self) -> Vec<u16> {
        self.r.iter().map(|&v| p::to_bf(v as f32)).collect()
    }
}

/// An FP8 GEMM: activation codes `[t][k]` at scale `x_scale` times `w`; the GEMM applies the product
/// of the two scales.
pub fn fp8_gemm(x8: &[u8], t: usize, w: &Fp8Weight, x_scale: f32, eval: Eval) -> GemmOut {
    let x: Vec<f32> = x8.iter().map(|&c| p::e4m3_dec(c)).collect();
    let (r, m) = gemm(&x, t, &w.dec, w.n, w.k);
    GemmOut::new(r, m, (w.scale * x_scale) as f64, eval)
}

/// An NVFP4 GEMM: activation codes `[t][k / 2]` with swizzled block scales `sf`, times `w`, with
/// alpha `group_scale * w.global` in F32.
pub fn fp4_gemm(
    x4: &[u8],
    sf: &[u8],
    t: usize,
    w: &Fp4Weight,
    group_scale: f32,
    eval: Eval,
) -> GemmOut {
    let k = w.k;
    let x: Vec<f32> = (0..t * k)
        .map(|e| {
            let (row, col) = (e / k, e % k);
            let c = x4[e / 2] >> (4 * (e % 2)) & 15;
            e2m1_dec(c) * p::e4m3_dec(sf[swizzled_offset(row, col / 16, k / 16)])
        })
        .collect();
    let (r, m) = gemm(&x, t, &w.dec, w.n, k);
    GemmOut::new(r, m, (group_scale * w.global) as f64, eval)
}

/// The BF16 `a`/`b` GEMM: normed rows `[t][5120]` times the `[96][5120]` rows, F32 output.
pub fn ab_gemm(normed: &[u16], t: usize, ab: &[u16], eval: Eval) -> GemmOut {
    let x: Vec<f32> = normed.iter().map(|&b| p::bf(b)).collect();
    let w: Vec<f32> = ab.iter().map(|&b| p::bf(b)).collect();
    let (r, m) = gemm(&x, t, &w, gdn::AB, H);
    GemmOut::new(r, m, 1.0, eval)
}

/// Elements of `got` outside the bound around `want` (a non-finite element always is), and the
/// worst error / bound. `rel` is the relative term: 2^-8 for a BF16 output, 2^-22 for an F32 one.
fn gemm_violations(got: &[f64], want: &GemmOut, rel: f64) -> (usize, f64) {
    let (mut bad, mut worst) = (0, 0.0f64);
    for ((&d, &r), &m) in got.iter().zip(&want.r).zip(&want.m) {
        let bound = rel * r.abs().max(d.abs()) + 2f64.powi(-17) * m;
        let err = (d - r).abs();
        if !d.is_finite() || err.is_nan() || err > bound {
            bad += 1;
        }
        if bound > 0.0 {
            worst = worst.max(err / bound);
        }
    }
    (bad, worst)
}

// ---------------------------------------------------------------------------------------------
// The dump of one layer, and the verdict.

/// One layer's tensors in a dump: its first `t` rows.
pub struct LayerDump<'a> {
    pub dump: &'a PrefillDump,
    pub layer: usize,
    pub t: usize,
}

impl LayerDump<'_> {
    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.dump.get(self.layer, name)
    }
    pub fn bf16(&self, name: &str) -> Option<Vec<u16>> {
        self.bytes(name).map(u16s)
    }
    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.bytes(name).map(f32s)
    }
    pub fn state_pos(&self, name: &str) -> Option<usize> {
        self.bytes(name)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
    }
}

pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

/// The checks of one layer.
#[derive(Default)]
pub struct LayerVerdict {
    pub checks: Vec<Check>,
}

impl LayerVerdict {
    pub fn push(&mut self, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
        self.checks.push(Check {
            name: name.into(),
            ok,
            detail: detail.into(),
        });
    }

    /// The names of the failed checks.
    pub fn failed(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter(|c| !c.ok)
            .map(|c| c.name.as_str())
            .collect()
    }

    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }

    /// One line per check.
    pub fn report(&self) -> String {
        self.checks
            .iter()
            .map(|c| {
                format!(
                    "  [{}] {} {}",
                    if c.ok { "ok" } else { "FAIL" },
                    c.name,
                    c.detail
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A tensor compared byte for byte.
    fn same(&mut self, name: &str, got: Option<&[u8]>, want: &[u8]) {
        match got {
            None => self.push(name, false, "not in the dump"),
            Some(g) if g.len() != want.len() => self.push(
                name,
                false,
                format!("{} bytes, expected {}", g.len(), want.len()),
            ),
            Some(g) => {
                let n = g.iter().zip(want).filter(|(a, b)| a != b).count();
                let first = g.iter().zip(want).position(|(a, b)| a != b);
                self.push(
                    name,
                    n == 0,
                    format!("{n} of {} bytes differ, first at {first:?}", g.len()),
                );
            }
        }
    }

    /// A GEMM output compared with its bound; `cols` selects the columns of a wider dumped tensor.
    fn gemm(
        &mut self,
        name: &str,
        got: Option<Vec<f64>>,
        ld: usize,
        cols: std::ops::Range<usize>,
        want: &GemmOut,
        rel: f64,
    ) {
        let Some(got) = got else {
            return self.push(name, false, "not in the dump");
        };
        let n = cols.len();
        if got.len() % ld != 0 || got.len() / ld * n != want.r.len() {
            return self.push(
                name,
                false,
                format!("{} values, expected rows of {ld}", got.len()),
            );
        }
        let got: Vec<f64> = got
            .chunks_exact(ld)
            .flat_map(|row| row[cols.clone()].to_vec())
            .collect();
        let (bad, worst) = gemm_violations(&got, want, rel);
        self.push(
            name,
            bad == 0,
            format!("{bad} of {} outside the bound, worst {worst:.3}", got.len()),
        );
    }
}

fn widen16(v: Option<Vec<u16>>) -> Option<Vec<f64>> {
    v.map(|v| v.into_iter().map(|b| p::bf(b) as f64).collect())
}

fn widen32(v: Option<Vec<f32>>) -> Option<Vec<f64>> {
    v.map(|v| v.into_iter().map(|x| x as f64).collect())
}

fn bytes16(v: &[u16]) -> Vec<u8> {
    p::bytes16(v)
}

// ---------------------------------------------------------------------------------------------
// The oracle.

/// An attention layer's prep and attention.
pub trait AttentionCore {
    /// Push the checks of the dumped prep outputs, KV cache and attention output, each from the dump
    /// of its own inputs.
    fn check(&self, w: &LayerWeights, d: &LayerDump, v: &mut LayerVerdict);
    /// For the whole-layer evaluation: the attention output and its gate from the projections `qg`,
    /// `k`, `v` that evaluation computed, and the KV rows the layer writes ([`kv_rows`]' layout), from
    /// the dumped KV cache before the layer. `e2` selects the second evaluation.
    fn evaluate(&self, w: &LayerWeights, d: &LayerDump, qkv: [&[u16]; 3], e2: bool) -> AttnEval;
}

/// An attention core's evaluation: `o` and `gate` BF16 `[t][6144]`, and the KV rows written.
pub struct AttnEval {
    pub o: Vec<u16>,
    pub gate: Vec<u16>,
    pub kv: Vec<f32>,
}

/// The rows `[p0, p0 + t)` of an attention layer's F32 KV cache `kv` (`[2][4][max_seq][256]`, K then
/// V), as `[2][4][t][256]`.
pub fn kv_rows(kv: &[f32], p0: usize, t: usize) -> Vec<f32> {
    let max_seq = kv.len() / (2 * attn::KVD);
    (0..2 * attn::HKV)
        .flat_map(|h| kv[(h * max_seq + p0) * attn::D..(h * max_seq + p0 + t) * attn::D].to_vec())
        .collect()
}

/// The attention core of the native kernels (`native_prefill_attn`), from the attention suite's host
/// models (`attn`). Its dump: `p0` (`u32`), `cs` the RoPE table rows `[p0, p0 + t)` F32 `[t][64]`,
/// `kv_in` and `kv` the F32 KV cache `[2][4][max_seq][256]` before and after the layer, `q` and `gate`
/// the prep's BF16 `[t][24][256]`, `o` the attention output.
///
/// - The prep byte for byte (`attention_core.q`, `.gate`), and the cache after the layer equal to the
///   cache before it with exactly the rows `[p0, p0 + t)` replaced by the prep's K and the V
///   projection widened from BF16 (`.kv`).
/// - The RoPE table rows (`.rope`) within `pos 2^-21 + 2^-21` of the f64 cosines and sines of
///   `pos theta^(-2i / 64)`: the table's F32 frequency (a `powf` of at most 4 units in the last place
///   and a division) and angle carry at most `pos 2^-22` of angle error, its cosines and sines two
///   units more.
/// - The attention (`.attention`) within 1.1x (relative L2) and 1.5x (largest) of the error the
///   rounded emulation makes against the exact f64 attention, the attention suite's gate.
/// - For the whole layer, E1 takes the rounded emulation's output and E2 the F32 model of the
///   kernel's algorithm.
pub struct Attention<'a> {
    pub hw: attn::Hw<'a>,
}

impl Attention<'_> {
    /// The first position, the RoPE table padded with zero rows below it, and the cache before the
    /// layer; `None` when one is missing or the table does not hold the layer's `t` rows.
    fn inputs(d: &LayerDump) -> Option<(usize, Vec<f32>, Vec<f32>)> {
        let p0 = d.state_pos("p0")?;
        let rows = d.f32("cs").filter(|r| r.len() == d.t * attn::ROT)?;
        let mut cs = vec![0.0; p0 * attn::ROT];
        cs.extend(rows);
        Some((p0, cs, d.f32("kv_in")?))
    }

    /// The prep's outputs and the cache after it, from the projections.
    fn prep(
        &self,
        w: &LayerWeights,
        qkv: [&[u16]; 3],
        t: usize,
        p0: usize,
        cs: &[f32],
        kv_in: &[f32],
    ) -> (attn::PrepOut, Vec<f32>) {
        let (q_w1, k_w1) = w.qk_norm.clone().unwrap();
        let inp = attn::PrepIn {
            t,
            qg: qkv[0].to_vec(),
            k: qkv[1].to_vec(),
            v: qkv[2].to_vec(),
            q_w1,
            k_w1,
        };
        let out = attn::prep_oracle(&self.hw, &inp, cs, p0, attn::Variant::Kernel);
        let max_seq = kv_in.len() / (2 * attn::KVD);
        let mut kv = kv_in.to_vec();
        for tt in 0..t {
            for hk in 0..attn::HKV {
                for c in 0..attn::D {
                    let from = (tt * attn::HKV + hk) * attn::D + c;
                    let at = (hk * max_seq + p0 + tt) * attn::D + c;
                    kv[at] = attn::bf(out.k[from]);
                    kv[attn::HKV * max_seq * attn::D + at] = attn::bf(inp.v[from]);
                }
            }
        }
        (out, kv)
    }

    fn case(q: &[u16], kv: &[f32], p0: usize) -> attn::AttnCase {
        let n = kv.len() / 2;
        attn::AttnCase {
            p0,
            max_seq: n / attn::KVD,
            q: q.iter().map(|&b| attn::bf(b)).collect(),
            k: kv[..n].to_vec(),
            v: kv[n..].to_vec(),
        }
    }
}

impl AttentionCore for Attention<'_> {
    fn check(&self, w: &LayerWeights, d: &LayerDump, v: &mut LayerVerdict) {
        let t = d.t;
        let (Some((p0, cs, kv_in)), Some(qg), Some(k), Some(vv)) =
            (Self::inputs(d), d.bf16("qg"), d.bf16("k"), d.bf16("v"))
        else {
            return v.push(
                "attention_core",
                false,
                "p0, cs (t rows), kv_in, qg, k or v is not in the dump",
            );
        };
        let (mut bad, mut worst) = (0usize, 0.0f64);
        for (i, row) in cs[p0 * attn::ROT..].chunks_exact(attn::ROT).enumerate() {
            let pos = (p0 + i) as f64;
            let tol = (pos + 1.0) * 2f64.powi(-21);
            for d in 0..attn::ROT / 2 {
                let angle = pos * (ROPE.theta as f64).powf(-((2 * d) as f64) / attn::ROT as f64);
                for (got, want) in [(row[d], angle.cos()), (row[attn::ROT / 2 + d], angle.sin())] {
                    let e = (got as f64 - want).abs();
                    bad += (e.is_nan() || e > tol) as usize;
                    worst = worst.max(e / tol);
                }
            }
        }
        v.push(
            "attention_core.rope",
            bad == 0,
            format!("{bad} table values outside the bound, worst {worst:.3}"),
        );
        let (want, kv) = self.prep(w, [&qg, &k, &vv], t, p0, &cs, &kv_in);
        v.same("attention_core.q", d.bytes("q"), &bytes16(&want.q));
        v.same("attention_core.gate", d.bytes("gate"), &bytes16(&want.gate));
        let kv_bytes: Vec<u8> = kv.iter().flat_map(|x| x.to_le_bytes()).collect();
        v.same("attention_core.kv", d.bytes("kv"), &kv_bytes);
        let (Some(q), Some(kv), Some(o)) = (d.bf16("q"), d.f32("kv"), d.bf16("o")) else {
            return v.push(
                "attention_core.attention",
                false,
                "q, kv or o is not in the dump",
            );
        };
        if o.len() != t * attn::QD || q.len() != t * attn::QD || kv.len() != kv_in.len() {
            return v.push(
                "attention_core.attention",
                false,
                "q, kv or o of another size",
            );
        }
        let case = Self::case(&q, &kv, p0);
        let rows: Vec<usize> = (0..t).collect();
        let (ok, msg) = attn::gate(
            &attn::gather(&o, &rows),
            &attn::attn_oracle(&case, &rows, false),
            &attn::attn_oracle(&case, &rows, true),
        );
        v.push("attention_core.attention", ok, msg);
    }

    fn evaluate(&self, w: &LayerWeights, d: &LayerDump, qkv: [&[u16]; 3], e2: bool) -> AttnEval {
        let t = d.t;
        let (p0, cs, kv_in) = Self::inputs(d).expect("p0, cs and kv_in in the dump");
        let (prep, kv) = self.prep(w, qkv, t, p0, &cs, &kv_in);
        let case = Self::case(&prep.q, &kv, p0);
        let rows: Vec<usize> = (0..t).collect();
        let o = if e2 {
            attn::attn_model_f32(&case, &rows, false)
        } else {
            attn::attn_oracle(&case, &rows, true)
        };
        AttnEval {
            o: o.iter().map(|&x| attn::to_bf(x as f32)).collect(),
            gate: prep.gate,
            kv: kv_rows(&kv, p0, t),
        }
    }
}

/// What the oracle runs with: the hardware probe and tables of the producers' approximate
/// instructions, and the device's multiprocessor count (it selects the GDN gated norm's reduction
/// shape). The norms' epsilon is the producers' [`p::EPS`].
pub struct Oracle<'a> {
    pub hw: &'a p::Hw<'a>,
    pub tab: &'a p::Tables,
    pub sm_count: u32,
}

/// A layer's whole-layer result.
pub struct LayerOut {
    pub mlp_out: Vec<u16>,
    pub resid: Vec<u16>,
    /// A GDN layer's ring and state.
    pub ring: Option<Vec<f32>>,
    pub state: Option<Vec<f64>>,
    /// An attention layer's KV rows, when its core writes them.
    pub kv: Option<Vec<f32>>,
}

fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        num += (g - w) * (g - w);
        den += w * w;
    }
    if !num.is_finite() {
        return f64::INFINITY;
    }
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    }
}

/// The smallest relative L2 limit of the whole-layer checks.
pub const WHOLE_LAYER_FLOOR: f64 = 1.0 / 512.0;
/// The whole-layer limit as a multiple of the distance between E1 and E2.
pub const WHOLE_LAYER_SPREAD: f64 = 1.5;

/// The checks that `next`'s inputs are `prev`'s outputs, byte for byte: for the next layer of one
/// prefill, its input rows and residual; for the same GDN layer's next slice, its ring, state and ring
/// position.
pub fn handoff(prev: &LayerDump, next: &LayerDump) -> LayerVerdict {
    let mut v = LayerVerdict::default();
    let pairs: &[(&str, &str)] = if next.layer == prev.layer + 1 {
        &[("mlp_out", "x"), ("resid_mlp", "resid_in")]
    } else {
        assert_eq!(next.layer, prev.layer, "handoff between unrelated layers");
        if prev.bytes("kv").is_some() {
            let (p, n) = (prev.state_pos("p0"), next.state_pos("p0"));
            v.push(
                "handoff.p0",
                p.is_some() && n == p.map(|p| p + prev.t),
                format!("{n:?} after {p:?} and {} tokens", prev.t),
            );
            &[("kv", "kv_in")]
        } else {
            &[
                ("ring", "ring_in"),
                ("state", "state_in"),
                ("state_pos", "state_pos_in"),
            ]
        }
    };
    for (out, inp) in pairs {
        match prev.bytes(out) {
            Some(want) => v.same(&format!("handoff.{inp}"), next.bytes(inp), want),
            None => v.push(
                format!("handoff.{inp}"),
                false,
                format!("{out} not in the dump"),
            ),
        }
    }
    v
}

impl Oracle<'_> {
    /// The input norm: the residual (the input rows themselves at layer 0) and the normed rows.
    fn norm_in(
        &self,
        w: &LayerWeights,
        x: &[u16],
        resid: Option<&[u16]>,
        t: usize,
    ) -> [Vec<u16>; 2] {
        let (r, normed) = p::gemma_norm(self.hw, x, resid, &w.norm_in, t, 0);
        [r, normed]
    }

    fn fp8_codes(&self, v: &[u16], scale: f32) -> Vec<u8> {
        p::fp8_all(v, p::fp8_inv(self.hw, scale))
    }

    /// The GDN gated norm's FP8 codes of `core` and `z`, in the reduction shape the route selects for
    /// its row count: one row per warp up to twice the multiprocessor count, two above.
    fn gated_norm(&self, w: &LayerWeights, core: &[u16], z: &[u16], t: usize) -> Vec<u8> {
        let rows = t * GDN_HEADS;
        let lanes = if rows as u64 <= 2 * self.sm_count as u64 {
            32
        } else {
            16
        };
        let g = w.gdn.as_ref().unwrap();
        let y = p::gated_norm(self.hw, self.tab, core, z, &g.norm, rows, lanes, 0);
        self.fp8_codes(&y, w.out.input_scale)
    }

    /// The attention output's gate to FP8 codes.
    fn sigmoid_gate(&self, w: &LayerWeights, o: &[u16], gate: &[u16]) -> Vec<u8> {
        self.fp8_codes(&p::sigmoid_bf16(self.tab, o, gate), w.out.input_scale)
    }

    /// The post-attention norm: (residual, NVFP4 codes, swizzled block scales).
    fn norm_post(
        &self,
        w: &LayerWeights,
        attn_out: &[u16],
        resid: &[u16],
        t: usize,
    ) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
        let (r, normed) = p::gemma_norm(self.hw, attn_out, Some(resid), &w.norm_post, t, 0);
        let (q, sf) = p::Fp4::new(self.hw, 1.0 / w.gate_up_scale()).quantize(&normed, t, H);
        (r, q, sf)
    }

    /// SwiGLU of `gu` to NVFP4: (codes, swizzled block scales).
    fn swiglu(&self, w: &LayerWeights, gu: &[u16], t: usize) -> (Vec<u8>, Vec<u8>) {
        let y = p::silu_bf16(self.tab, w.attention, gu, t);
        p::Fp4::new(self.hw, 1.0 / w.down.input_scale).quantize(&y, t, I)
    }

    /// The GDN models' inputs for a layer.
    fn gdn_inputs(
        w: &LayerWeights,
        t: usize,
        qkv: Vec<u16>,
        ab: Vec<f32>,
        ring: Vec<f32>,
        state: Vec<f32>,
        state_pos: usize,
    ) -> gdn::Inputs {
        let g = w.gdn.as_ref().unwrap();
        gdn::Inputs {
            t,
            state_pos,
            qkv,
            ring,
            conv_w: g.conv_w.clone(),
            ab,
            dt_bias: g.dt_bias.clone(),
            ssm_a: g.ssm_a.clone(),
            s0: state,
        }
    }

    /// Check a dumped layer step by step, then as a whole.
    pub fn check(
        &self,
        w: &LayerWeights,
        d: &LayerDump,
        core: Option<&dyn AttentionCore>,
    ) -> LayerVerdict {
        let mut v = LayerVerdict::default();
        let t = d.t;
        let Some(x) = d.bf16("x") else {
            v.push("x", false, "the layer input is not in the dump");
            return v;
        };
        let resid_in = d.bf16("resid_in");
        if (w.layer > 0) != resid_in.is_some() {
            v.push(
                "resid_in",
                false,
                "a residual input is dumped exactly after layer 0",
            );
            return v;
        }

        // The input norm.
        let [resid, normed] = self.norm_in(w, &x, resid_in.as_deref(), t);
        v.same("norm_in.resid", d.bytes("resid_attn"), &bytes16(&resid));
        v.same("norm_in.normed", d.bytes("normed"), &bytes16(&normed));
        let proj_in = w.proj_in_scale();
        if let Some(dn) = d.bf16("normed") {
            v.same("norm_in.x8", d.bytes("x8"), &self.fp8_codes(&dn, proj_in));
        }
        let x8 = d.bytes("x8").unwrap_or_default().to_vec();

        // The input projections.
        let names: &[&str] = if w.attention {
            &["qg", "k", "v"]
        } else {
            &["qkv", "z"]
        };
        for (name, pw) in names.iter().zip(&w.inputs) {
            if x8.len() == t * H {
                let want = fp8_gemm(&x8, t, pw, proj_in, Eval::E1);
                v.gemm(
                    name,
                    widen16(d.bf16(name)),
                    pw.n,
                    0..pw.n,
                    &want,
                    2f64.powi(-8),
                );
            } else {
                v.push(*name, false, "no x8 of the right size to recompute it from");
            }
        }

        // The mixer: the GDN chain, or the attention core.
        let out_codes = if w.attention {
            match core {
                Some(c) => c.check(w, d, &mut v),
                None => v.push(
                    "attention_core",
                    false,
                    "no attention oracle is plugged in: prep, KV rows and attention are unchecked",
                ),
            }
            match (d.bf16("o"), d.bf16("gate")) {
                (Some(o), Some(g)) => v.same(
                    "sigmoid_gate.o8",
                    d.bytes("o8"),
                    &self.sigmoid_gate(w, &o, &g),
                ),
                _ => v.push("sigmoid_gate.o8", false, "o or gate is not in the dump"),
            }
            d.bytes("o8")
        } else {
            self.check_gdn(w, d, d.bf16("normed").unwrap_or(normed), &mut v);
            match (d.bf16("core"), d.bf16("z")) {
                (Some(core), Some(z)) => v.same(
                    "gated_norm.y8",
                    d.bytes("y8"),
                    &self.gated_norm(w, &core, &z, t),
                ),
                _ => v.push("gated_norm.y8", false, "core or z is not in the dump"),
            }
            d.bytes("y8")
        };
        match out_codes {
            Some(c) if c.len() == t * w.out.k => {
                let want = fp8_gemm(c, t, &w.out, w.out.input_scale, Eval::E1);
                v.gemm(
                    "attn_out",
                    widen16(d.bf16("attn_out")),
                    H,
                    0..H,
                    &want,
                    2f64.powi(-8),
                );
            }
            _ => v.push("attn_out", false, "no output codes of the right size"),
        }

        // The MLP.
        match (d.bf16("attn_out"), d.bf16("resid_attn")) {
            (Some(ao), Some(r)) => {
                let (r2, q, sf) = self.norm_post(w, &ao, &r, t);
                v.same("norm_post.resid", d.bytes("resid_mlp"), &bytes16(&r2));
                v.same("norm_post.x4", d.bytes("x4"), &q);
                v.same("norm_post.x4sf", d.bytes("x4sf"), &sf);
            }
            _ => v.push(
                "norm_post",
                false,
                "attn_out or resid_attn is not in the dump",
            ),
        }
        match (d.bytes("x4"), d.bytes("x4sf")) {
            (Some(q), Some(sf)) if q.len() == t * H / 2 => {
                let gs = w.gate_up_scale();
                for (name, pw, cols) in [("gate", &w.gate, 0..I), ("up", &w.up, I..2 * I)] {
                    let want = fp4_gemm(q, sf, t, pw, gs, Eval::E1);
                    v.gemm(
                        name,
                        widen16(d.bf16("gu")),
                        2 * I,
                        cols,
                        &want,
                        2f64.powi(-8),
                    );
                }
            }
            _ => v.push("gate_up", false, "no x4 of the right size"),
        }
        if let Some(gu) = d.bf16("gu").filter(|g| g.len() == t * 2 * I) {
            let (q, sf) = self.swiglu(w, &gu, t);
            v.same("swiglu.d4", d.bytes("d4"), &q);
            v.same("swiglu.d4sf", d.bytes("d4sf"), &sf);
        } else {
            v.push("swiglu", false, "no gu of the right size");
        }
        match (d.bytes("d4"), d.bytes("d4sf")) {
            (Some(q), Some(sf)) if q.len() == t * I / 2 => {
                let want = fp4_gemm(q, sf, t, &w.down, w.down.input_scale, Eval::E1);
                v.gemm(
                    "down",
                    widen16(d.bf16("mlp_out")),
                    H,
                    0..H,
                    &want,
                    2f64.powi(-8),
                );
            }
            _ => v.push("down", false, "no d4 of the right size"),
        }

        // The row handed to decode: f32(mlp_out) + f32(residual) of the last token, unrounded.
        if let Some(got) = d.bytes("x_gpu") {
            match (d.bf16("mlp_out"), d.bf16("resid_mlp")) {
                (Some(mo), Some(r)) if mo.len() == t * H && r.len() == t * H => {
                    let want: Vec<u8> = (0..H)
                        .flat_map(|c| {
                            let i = (t - 1) * H + c;
                            (p::bf(mo[i]) + p::bf(r[i])).to_le_bytes()
                        })
                        .collect();
                    v.same("final_row", Some(got), &want);
                }
                _ => v.push("final_row", false, "mlp_out or resid_mlp missing"),
            }
        }

        // The whole layer.
        if w.attention && core.is_none() {
            v.push(
                "layer",
                false,
                "the whole-layer evaluation needs the attention oracle",
            );
        } else {
            self.check_whole(w, d, &x, resid_in, core, &mut v);
        }
        v
    }

    /// The GDN steps: the `a`/`b` GEMM, then conv, chunks and state by the GDN models' gate, and
    /// the ring position.
    fn check_gdn(&self, w: &LayerWeights, d: &LayerDump, normed: Vec<u16>, v: &mut LayerVerdict) {
        let t = d.t;
        let g = w.gdn.as_ref().unwrap();
        let want = ab_gemm(&normed, t, &g.ab, Eval::E1);
        v.gemm(
            "ab",
            widen32(d.f32("ab")),
            gdn::AB,
            0..gdn::AB,
            &want,
            2f64.powi(-22),
        );
        let (Some(qkv), Some(ab), Some(ring_in), Some(state_in), Some(pos_in)) = (
            d.bf16("qkv"),
            d.f32("ab"),
            d.f32("ring_in"),
            d.f32("state_in"),
            d.state_pos("state_pos_in"),
        ) else {
            return v.push(
                "gdn",
                false,
                "qkv, ab, ring_in, state_in or state_pos_in missing",
            );
        };
        let inp = Self::gdn_inputs(w, t, qkv, ab, ring_in, state_in, pos_in);
        let (Some(cv), Some(core), Some(state), Some(ring)) =
            (d.bf16("cv"), d.bf16("core"), d.f32("state"), d.f32("ring"))
        else {
            return v.push("gdn", false, "cv, core, state or ring missing");
        };
        let got = gdn::Got {
            cv: cv.into_iter().map(gdn::bf).collect(),
            gc: Vec::new(),
            u: Vec::new(),
            out: core.into_iter().map(gdn::bf).collect(),
            s: state,
            ring,
            state_pos: (pos_in + t) % gdn::SLOTS,
        };
        let verdict = gdn::gate(
            &inp,
            &got,
            &gdn::oracle(&inp),
            &gdn::emulate(&inp, true),
            false,
        );
        for name in [
            "finite",
            "out_oracle",
            "out_max",
            "out_emulation",
            "state_oracle",
            "state_max",
            "state_emulation",
            "conv",
            "ring",
        ] {
            let ok = !verdict.failed.contains(&name);
            v.push(
                format!("gdn.{name}"),
                ok,
                if ok { "" } else { verdict.msg.as_str() },
            );
        }
        let pos = d.state_pos("state_pos");
        v.push(
            "gdn.state_pos",
            pos == Some(got.state_pos),
            format!("{pos:?}, expected {}", got.state_pos),
        );
    }

    /// Evaluate layer `w` from its input rows `x`, residual and (GDN) the dumped ring and state
    /// before it: E1, or E2 when `e2` (each GEMM with its own salt).
    #[allow(clippy::too_many_arguments)]
    pub fn whole(
        &self,
        w: &LayerWeights,
        d: &LayerDump,
        x: &[u16],
        resid: Option<&[u16]>,
        core: Option<&dyn AttentionCore>,
        e2: bool,
    ) -> LayerOut {
        let t = d.t;
        let eval = |salt: u64| if e2 { Eval::E2 { salt } } else { Eval::E1 };
        let [resid, normed] = self.norm_in(w, x, resid, t);
        let proj_in = w.proj_in_scale();
        let x8 = self.fp8_codes(&normed, proj_in);
        let proj: Vec<Vec<u16>> = w
            .inputs
            .iter()
            .enumerate()
            .map(|(i, pw)| fp8_gemm(&x8, t, pw, proj_in, eval(i as u64)).bf16())
            .collect();
        let (mut ring, mut state, mut kv) = (None, None, None);
        let out_codes = if w.attention {
            let a = core.expect("an attention layer's core").evaluate(
                w,
                d,
                [&proj[0], &proj[1], &proj[2]],
                e2,
            );
            kv = Some(a.kv);
            self.sigmoid_gate(w, &a.o, &a.gate)
        } else {
            let g = w.gdn.as_ref().unwrap();
            let ab: Vec<f32> = ab_gemm(&normed, t, &g.ab, eval(3))
                .r
                .into_iter()
                .map(|v| v as f32)
                .collect();
            let inp = Self::gdn_inputs(
                w,
                t,
                proj[0].clone(),
                ab,
                d.f32("ring_in").unwrap(),
                d.f32("state_in").unwrap(),
                d.state_pos("state_pos_in").unwrap(),
            );
            let e = gdn::emulate_with(&inp, true, e2);
            let core: Vec<u16> = e.out.iter().map(|&o| gdn::to_bf(o as f32)).collect();
            (ring, state) = (Some(e.ring), Some(e.s));
            self.gated_norm(w, &core, &proj[1], t)
        };
        let attn_out = fp8_gemm(&out_codes, t, &w.out, w.out.input_scale, eval(4)).bf16();
        let (resid, x4, sf) = self.norm_post(w, &attn_out, &resid, t);
        let gs = w.gate_up_scale();
        let gate = fp4_gemm(&x4, &sf, t, &w.gate, gs, eval(5)).bf16();
        let up = fp4_gemm(&x4, &sf, t, &w.up, gs, eval(6)).bf16();
        let gu: Vec<u16> = (0..t)
            .flat_map(|r| [&gate[r * I..(r + 1) * I], &up[r * I..(r + 1) * I]].concat())
            .collect();
        let (d4, d4sf) = self.swiglu(w, &gu, t);
        let mlp_out = fp4_gemm(&d4, &d4sf, t, &w.down, w.down.input_scale, eval(7)).bf16();
        LayerOut {
            mlp_out,
            resid,
            ring,
            state,
            kv,
        }
    }

    /// The whole-layer check: the dumped outputs against E1, within [`WHOLE_LAYER_SPREAD`] times the
    /// E1-E2 distance and never below [`WHOLE_LAYER_FLOOR`].
    fn check_whole(
        &self,
        w: &LayerWeights,
        d: &LayerDump,
        x: &[u16],
        resid: Option<Vec<u16>>,
        core: Option<&dyn AttentionCore>,
        v: &mut LayerVerdict,
    ) {
        if !w.attention
            && (d.f32("ring_in").is_none()
                || d.f32("state_in").is_none()
                || d.state_pos("state_pos_in").is_none())
        {
            return v.push("layer", false, "ring_in, state_in or state_pos_in missing");
        }
        if w.attention && Attention::inputs(d).is_none() {
            return v.push("layer", false, "p0, cs (t rows) or kv_in missing");
        }
        let e1 = self.whole(w, d, x, resid.as_deref(), core, false);
        let e2 = self.whole(w, d, x, resid.as_deref(), core, true);
        let w16 = |v: &[u16]| -> Vec<f64> { v.iter().map(|&b| p::bf(b) as f64).collect() };
        let w32 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
        let mut pairs: Vec<(&str, Option<Vec<f64>>, Vec<f64>, Vec<f64>)> = vec![
            (
                "layer.mlp_out",
                widen16(d.bf16("mlp_out")),
                w16(&e1.mlp_out),
                w16(&e2.mlp_out),
            ),
            (
                "layer.resid",
                widen16(d.bf16("resid_mlp")),
                w16(&e1.resid),
                w16(&e2.resid),
            ),
        ];
        if let (Some(r1), Some(r2), Some(s1), Some(s2)) = (&e1.ring, &e2.ring, &e1.state, &e2.state)
        {
            pairs.push(("layer.ring", widen32(d.f32("ring")), w32(r1), w32(r2)));
            pairs.push((
                "layer.state",
                widen32(d.f32("state")),
                s1.clone(),
                s2.clone(),
            ));
        }
        if let (Some(k1), Some(k2), Some(p0)) = (&e1.kv, &e2.kv, d.state_pos("p0")) {
            let got = d.f32("kv").map(|kv| kv_rows(&kv, p0, d.t));
            pairs.push(("layer.kv", widen32(got), w32(k1), w32(k2)));
        }
        for (name, got, e1v, e2v) in pairs {
            let Some(got) = got.filter(|g| g.len() == e1v.len()) else {
                v.push(name, false, "not in the dump, or of another size");
                continue;
            };
            let spread = rel_l2(&e2v, &e1v);
            let limit = (WHOLE_LAYER_SPREAD * spread).max(WHOLE_LAYER_FLOOR);
            let dist = rel_l2(&got, &e1v);
            v.push(
                name,
                dist <= limit,
                format!("{dist:.3e} from E1, limit {limit:.3e} (E1-E2 {spread:.3e})"),
            );
        }
    }
}
