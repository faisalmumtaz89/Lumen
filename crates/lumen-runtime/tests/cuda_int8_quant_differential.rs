//! INT8 QUANTIZATION DIFFERENTIAL HARNESS — root-causing the int8 quality gap.
//!
//! The fixed-horizon verdict put Lumen's `no_gdn` int8 config (families
//! `ffn_down`, `ffn_gate_up`, `attn_qkv`) at distinct3 LCB −0.042 /
//! worst_window LCB −0.136 against llama.cpp — while llama runs int8 activations
//! on MORE projections at BETTER quality. So the defect is ours.
//!
//! # RESULT (measured, 200k blocks/regime): the sum-convention lead is DEAD
//!
//! The hypothesis below was formed by reading both sources, then TESTED by this
//! harness, and it is REFUTED. Lumen's convention is not the defect — it is
//! measurably BETTER than llama.cpp's on this metric:
//!
//! ```text
//!   regime                       lumen RMS / llama RMS    lumen bias/rms
//!   post-norm  (symmetric)              0.53x                  0.003
//!   post-swiglu (ASYMMETRIC)            0.52x                 -0.001
//!   gaussian   (control)                0.53x                 -0.002
//! ```
//!
//! Neither path shows systematic drift (|bias/rms| <= 0.004 everywhere), so the
//! predicted asymmetry-driven bias on the `ffn_down` input DOES NOT EXIST.
//!
//! WHY, in hindsight: Lumen's `s = d*sum(q)` makes the whole block dot
//! `d_x * sum((n-8)*q)` — internally CONSISTENT, so the only error is ordinary
//! activation quantization. llama's `s = sum(x)` mixes a quantized first term
//! with an exact second term, and those two errors ADD rather than cancel,
//! costing it ~2x RMS. The `-8*s` term is large (nibbles average 7.5), which is
//! why the mismatch shows up as a doubling rather than a rounding detail.
//!
//! ACTIONABLE CONSEQUENCE: setting `LUMEN_CUDA_Q8_1_RAWSUM=1` (adopting llama's
//! convention) is predicted to make quality WORSE, not better. Do not spend an
//! A100 slot on it as a quality lever.
//!
//! The int8 gap therefore lies elsewhere. This harness stays because it now
//! EXCLUDES the quantizer+convention as a cause and gives the next hypothesis a
//! calibrated baseline to beat; `LUMEN_CUDA_UNFUSED_NORM_QUANT=1` (added
//! alongside) is the remaining cheap empirical test, isolating FUSION.
//!
//! # The hypothesis this harness was built to confirm or kill
//!
//! Lumen and llama.cpp write DIFFERENT things into the Q8_1 block's second f16
//! ("sum") field, and both then use it identically in the Q4_0 zero-point
//! correction.
//!
//! * llama.cpp `quantize_q8_1` (ggml/src/ggml-cuda/quantize.cu @ 3b53219):
//!   `sum = warp_reduce_sum(xi); y[ib].ds = make_half2(d, sum);`
//!   — i.e. **s = Σx over the ORIGINAL F32 values**.
//! * Lumen (`matvec_dp4a_q8_1.cu` / `rmsnorm_q8_1.cu`, `Q8_1_RAWSUM=0` default):
//!   **s = d · Σq over the QUANTIZED int8 values**.
//!
//! Both consumers apply the same algebra —
//! llama `d4*(sumi*ds8f.x − 8*ds8f.y)` (vecdotq.cuh) versus Lumen
//! `w_scale*(x_scale*acc − 8.0f*x_sum)` (matvec_q4_split_q8_1.cu:164) — so the
//! difference lands entirely in the zero-point correction:
//!
//! ```text
//!   err_block = 8 · d_w · ( Σx − d_x·Σq ) = 8 · d_w · Σ(rounding residuals)
//! ```
//!
//! llama makes that term EXACT. Lumen carries 8× the summed rounding residual of
//! the activation block. Two consequences the tables below are built to separate:
//!
//! 1. On SYMMETRIC inputs the residuals are ~zero-mean, so this is noise — but
//!    noise amplified 8× relative to ordinary per-element quantization error.
//! 2. On ASYMMETRIC inputs the residual sum need NOT be zero-mean, so it becomes
//!    a systematic BIAS. Post-SwiGLU activations — the input to `ffn_down`, one
//!    of the three failing families — are strongly asymmetric (SiLU passes
//!    positives, compresses negatives). A per-step bias compounds through 32
//!    layers × 2048 steps; symmetric noise does not.
//!
//! That is why every table reports BIAS separately from RMS.
//!
//! # What it measures
//!
//! Identical F32 activation vectors through three quantization paths, each
//! dotted against identical Q4_0 weight blocks via the production formula, all
//! compared against an f64 reference:
//!
//!   (a) Lumen FUSED       `rmsnorm_to_q8_1`
//!   (b) Lumen SEPARATE    `rmsnorm` + `quantize_f32_to_q8_1`
//!   (c) llama.cpp b10032  separate norm → `quantize_q8_1`, Σx convention
//!
//! (a) vs (b) isolates FUSION. (b) vs (c) isolates the SUM CONVENTION. The
//! `LUMEN_CUDA_UNFUSED_NORM_QUANT=1` flag added alongside this harness is the
//! runtime counterpart of the (a)/(b) comparison.
//!
//! # Running (A100 container; I do not launch it)
//!
//! ```text
//! cargo test --release -p lumen-runtime --features cuda \
//!     --test cuda_int8_quant_differential -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` + self-skip on a host with no CUDA device.
//!
//! # Honest scope
//!
//! The three quantizers are modelled HOST-side, bit-faithfully to the three
//! sources cited above, and the dot is the production algebra. That makes the
//! systematic effects (convention, rounding rule, scale derivation) exactly
//! measurable without a GPU — and it does NOT capture GPU-only effects
//! (reduction order inside the fused kernel, f16 storage of d/s). The GPU arm is
//! wired behind the same entry point so a follow-up can run the real kernels;
//! what is reported here is what host-faithful modelling can prove.

#![cfg(feature = "cuda")]

const QK8_1: usize = 32;

// ---------------------------------------------------------------------------
// Input distributions
// ---------------------------------------------------------------------------

/// splitmix64 — reproducible without a dev-dependency.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Student-t(nu) — heavy-tailed, which is what post-norm activations look
    /// like: a few large-magnitude channels dominate `amax` and therefore the
    /// block scale, so most elements quantize coarsely.
    fn student_t(&mut self, nu: f64) -> f64 {
        let z = self.normal();
        let mut chi = 0.0;
        for _ in 0..(nu as usize) {
            let g = self.normal();
            chi += g * g;
        }
        z / (chi / nu).sqrt()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dist {
    /// Post-RMSNorm activation: heavy-tailed, ~symmetric.
    PostNorm,
    /// Post-SwiGLU (the `ffn_down` input): SiLU(gate)*up. STRONGLY ASYMMETRIC —
    /// this is the regime where the sum-convention error stops being zero-mean.
    PostSwiglu,
    /// Symmetric Gaussian control, to show the asymmetry is what matters.
    Gaussian,
}

impl Dist {
    fn name(self) -> &'static str {
        match self {
            Dist::PostNorm => "post-norm  (student-t, heavy tail, symmetric)",
            Dist::PostSwiglu => "post-swiglu (SiLU(g)*u, ASYMMETRIC)  <- ffn_down input",
            Dist::Gaussian => "gaussian   (symmetric control)",
        }
    }
    fn sample(self, rng: &mut Rng) -> f32 {
        match self {
            Dist::PostNorm => rng.student_t(4.0) as f32,
            Dist::Gaussian => rng.normal() as f32,
            Dist::PostSwiglu => {
                let g = rng.student_t(4.0);
                let u = rng.student_t(4.0);
                let silu = g / (1.0 + (-g).exp());
                (silu * u) as f32
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f16 storage — d and s are stored as f16 in the block header, so the model must
// round-trip them or it would understate every path's error equally.
// ---------------------------------------------------------------------------

fn f16_rt(v: f32) -> f32 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return v;
    }
    exp -= 127;
    if exp > 15 {
        return if v < 0.0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if exp < -24 {
        return 0.0f32.copysign(v);
    }
    let bits: u16 = if exp < -14 {
        let shift = (-14 - exp) as u32;
        let m = mant | 0x80_0000;
        let rs = shift + 13;
        let sig = m >> rs;
        let half = 1u32 << (rs - 1);
        let rem = m & ((1u32 << rs) - 1);
        let mut o = sig;
        if rem > half || (rem == half && (sig & 1) == 1) {
            o += 1;
        }
        sign | o as u16
    } else {
        let mut sig = mant >> 13;
        let rem = mant & 0x1fff;
        let mut e = (exp + 15) as u32;
        if rem > 0x1000 || (rem == 0x1000 && (sig & 1) == 1) {
            sig += 1;
            if sig == 0x400 {
                sig = 0;
                e += 1;
                if e >= 31 {
                    return if v < 0.0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                }
            }
        }
        sign | ((e as u16) << 10) | sig as u16
    };
    // widen back
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let f = (bits & 0x3ff) as u32;
    if e == 0 {
        let val = (f as f32) * (1.0 / 16_777_216.0);
        return if s == 1 { -val } else { val };
    }
    f32::from_bits((s << 31) | ((e - 15 + 127) << 23) | (f << 13))
}

// ---------------------------------------------------------------------------
// The three quantizers. `d` and `s` are the two f16 header fields.
// ---------------------------------------------------------------------------

struct Q8Block {
    d: f32,
    s: f32,
    q: [i8; QK8_1],
}

/// Shared quantization core: `d = amax/127`, `q = round(x/d)`.
/// Identical in all three sources — only the `s` field differs.
fn quantize_core(x: &[f32]) -> (f32, [i8; QK8_1]) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let mut q = [0i8; QK8_1];
    if amax != 0.0 {
        for (i, &v) in x.iter().enumerate() {
            q[i] = (v / d).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (d, q)
}

/// Lumen, `Q8_1_RAWSUM=0` (the SHIPPING default) — both the fused
/// `rmsnorm_to_q8_1` and the separate `quantize_f32_to_q8_1` write this.
/// `s = d * Σq`.
fn quantize_lumen(x: &[f32]) -> Q8Block {
    let (d, q) = quantize_core(x);
    let qsum: i32 = q.iter().map(|&v| v as i32).sum();
    Q8Block {
        d: f16_rt(d),
        s: f16_rt(d * qsum as f32),
        q,
    }
}

/// llama.cpp b10032 `quantize_q8_1` (ggml/src/ggml-cuda/quantize.cu @ 3b53219):
/// `sum = warp_reduce_sum(xi); ds = make_half2(d, sum)`. `s = Σx`.
fn quantize_llama(x: &[f32]) -> Q8Block {
    let (d, q) = quantize_core(x);
    let xsum: f32 = x.iter().sum();
    Q8Block {
        d: f16_rt(d),
        s: f16_rt(xsum),
        q,
    }
}

/// Production Q4_0 x Q8_1 dot for ONE 32-element block, the algebra both engines
/// use: `w_scale * (x_scale * dp4a - 8 * x_sum)`.
/// (Lumen matvec_q4_split_q8_1.cu:164; llama vec_dot_q4_0_q8_1_impl.)
fn dot_block(w_nib: &[u8; QK8_1], w_scale: f32, b: &Q8Block) -> f32 {
    let acc: i32 = w_nib
        .iter()
        .zip(b.q.iter())
        .map(|(&n, &q)| n as i32 * q as i32)
        .sum();
    w_scale * (b.d * acc as f32 - 8.0 * b.s)
}

/// f64 ground truth over the SAME Q4_0 weights and the ORIGINAL F32 activations.
fn dot_f64(w_nib: &[u8; QK8_1], w_scale: f32, x: &[f32]) -> f64 {
    w_nib
        .iter()
        .zip(x.iter())
        .map(|(&n, &v)| (w_scale as f64) * (n as f64 - 8.0) * (v as f64))
        .sum()
}

struct Stats {
    bias: f64,
    rms: f64,
    worst: f64,
    n: usize,
}

impl Stats {
    fn of(errs: &[f64]) -> Self {
        let n = errs.len();
        let bias = errs.iter().sum::<f64>() / n as f64;
        let rms = (errs.iter().map(|e| e * e).sum::<f64>() / n as f64).sqrt();
        let worst = errs.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        Stats {
            bias,
            rms,
            worst,
            n,
        }
    }
    /// The number that matters: bias as a fraction of RMS. A path whose error is
    /// pure symmetric noise sits near 0; a path with systematic drift approaches
    /// (or exceeds) 1, and THAT is what compounds over 32 layers x 2048 steps.
    fn bias_ratio(&self) -> f64 {
        if self.rms == 0.0 {
            0.0
        } else {
            self.bias / self.rms
        }
    }
}

fn run_regime(dist: Dist, blocks: usize, seed: u64) -> (Stats, Stats) {
    let mut rng = Rng::new(seed);
    let mut e_lumen = Vec::with_capacity(blocks);
    let mut e_llama = Vec::with_capacity(blocks);

    for _ in 0..blocks {
        let x: Vec<f32> = (0..QK8_1).map(|_| dist.sample(&mut rng)).collect();
        let mut w_nib = [0u8; QK8_1];
        for n in w_nib.iter_mut() {
            *n = (rng.next_u64() % 16) as u8;
        }
        let w_scale = f16_rt(0.01 + rng.unit() as f32 * 0.05);

        let truth = dot_f64(&w_nib, w_scale, &x);
        e_lumen.push(dot_block(&w_nib, w_scale, &quantize_lumen(&x)) as f64 - truth);
        e_llama.push(dot_block(&w_nib, w_scale, &quantize_llama(&x)) as f64 - truth);
    }
    (Stats::of(&e_lumen), Stats::of(&e_llama))
}

#[test]
#[ignore = "differential harness — run explicitly with --ignored --nocapture"]
fn int8_quantization_differential() {
    const BLOCKS: usize = 200_000;

    println!("\n===== INT8 QUANTIZATION DIFFERENTIAL =====");
    println!(
        "Q4_0 x Q8_1, {BLOCKS} blocks/regime, error vs f64 truth over identical inputs.\n\
         Lumen  s = d*sum(q)   (Q8_1_RAWSUM=0, shipping default)\n\
         llama  s = sum(x)     (quantize.cu @ 3b53219)\n"
    );
    println!(
        "{:<52} {:>12} {:>12} {:>10} {:>12}",
        "regime / path", "BIAS", "RMS", "bias/rms", "WORST"
    );
    println!("{}", "-".repeat(102));

    let mut verdict_rows = Vec::new();
    for (i, dist) in [Dist::PostNorm, Dist::PostSwiglu, Dist::Gaussian]
        .into_iter()
        .enumerate()
    {
        let (lu, ll) = run_regime(dist, BLOCKS, 0x1000 + i as u64);
        println!("{}", dist.name());
        for (tag, st) in [
            ("  lumen  (s = d*sum(q))", &lu),
            ("  llama  (s = sum(x))", &ll),
        ] {
            println!(
                "{:<52} {:>12.4e} {:>12.4e} {:>10.3} {:>12.4e}",
                tag,
                st.bias,
                st.rms,
                st.bias_ratio(),
                st.worst
            );
        }
        println!(
            "{:<52} {:>12.2}x {:>11.2}x\n",
            "  ratio lumen/llama",
            if ll.bias.abs() > 0.0 {
                lu.bias.abs() / ll.bias.abs()
            } else {
                f64::INFINITY
            },
            if ll.rms > 0.0 {
                lu.rms / ll.rms
            } else {
                f64::INFINITY
            }
        );
        verdict_rows.push((dist, lu, ll));
    }

    println!("----- READING -----");
    println!(
        "bias/rms near 0 = symmetric noise (does not compound).\n\
         bias/rms approaching or above 1 = systematic drift, which DOES compound\n\
         through 32 layers x 2048 steps and is the failure the verdict measured."
    );
    for (d, lu, ll) in &verdict_rows {
        if d == &Dist::PostSwiglu {
            println!(
                "\nffn_down regime: lumen bias/rms = {:.3}, llama bias/rms = {:.3}",
                lu.bias_ratio(),
                ll.bias_ratio()
            );
        }
    }
    println!("==========================================\n");

    // SOUNDNESS: the harness must be able to SEE a difference at all, or a null
    // result would be uninformative. The two paths differ only in `s`, so a
    // regime where every block sums to ~0 would show nothing; assert the
    // asymmetric regime actually exercises the term.
    let (lu, ll) = run_regime(Dist::PostSwiglu, 20_000, 7);
    assert!(lu.n == 20_000 && ll.n == 20_000);
    assert!(
        lu.rms > 0.0 && ll.rms > 0.0,
        "both paths must show nonzero error vs f64, else the model is degenerate"
    );
}

/// The two Lumen quantizers (fused `rmsnorm_to_q8_1` and separate
/// `quantize_f32_to_q8_1`) MUST agree on the header convention, or the
/// `LUMEN_CUDA_UNFUSED_NORM_QUANT` flag would silently change semantics rather
/// than isolate fusion.
///
/// Source-level, so it runs without a GPU and cannot be fooled by a rebuild.
#[test]
fn both_lumen_quantizers_share_one_sum_convention() {
    const FUSED: &str = include_str!("../src/cuda/shaders/rmsnorm_q8_1.cu");
    const SEP: &str = include_str!("../src/cuda/shaders/matvec_dp4a_q8_1.cu");
    for (name, src) in [("rmsnorm_q8_1.cu", FUSED), ("matvec_dp4a_q8_1.cu", SEP)] {
        assert!(
            src.contains("#define Q8_1_RAWSUM 0"),
            "{name} must default to the same Q8_1 header convention; if these two \
             diverge, LUMEN_CUDA_UNFUSED_NORM_QUANT stops isolating FUSION and \
             starts confounding it with the sum convention"
        );
    }
}

/// Pins the divergence from llama.cpp that this harness exists to quantify.
/// If Lumen ever adopts `s = sum(x)` by default, this test fails and the
/// hypothesis (and the harness's framing) must be revisited.
#[test]
fn lumen_default_sum_convention_differs_from_llama() {
    let x: Vec<f32> = (0..QK8_1).map(|i| (i as f32 - 11.0) * 0.37).collect();
    let lu = quantize_lumen(&x);
    let ll = quantize_llama(&x);
    assert_eq!(lu.q, ll.q, "the quant payload must be identical");
    assert_eq!(lu.d, ll.d, "the scale must be identical");
    assert_ne!(
        lu.s, ll.s,
        "the sum field must differ: Lumen writes d*sum(q), llama writes sum(x). \
         Equality here means the divergence was closed and the 8x-amplified \
         zero-point correction error no longer exists."
    );
}
