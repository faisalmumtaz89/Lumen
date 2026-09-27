//! f64 models of the native prefill's GDN layer (`native_prefill_gdn`) and the gate that compares a
//! run with them. Used by `cuda_native_gdn_test` and the layer oracle.
//!
//! - The oracle: the formulas of Lumen's F32 prefill (conv, SiLU, L2 norm of q and k, gates, then
//!   one delta-rule step per token), evaluated in f64 on the unrounded inputs.
//! - The emulation: the 64-token chunked form of the same recurrence, rounding to BF16 at the points
//!   the native kernels hold BF16 and computing everything else in f64. Its distance to the oracle is
//!   the error of that rounding.

pub const H: usize = 48;
pub const HK: usize = 16;
pub const D: usize = 128;
pub const QK: usize = HK * D;
pub const CONV: usize = 2 * QK + H * D;
pub const AB: usize = 2 * H;
pub const BT: usize = 64;
pub const SLOTS: usize = 3;

// ---------------------------------------------------------------------------------------------
// Numbers, randomness, threads.

pub fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// F32 -> BF16, round to nearest even.
pub fn to_bf(f: f32) -> u16 {
    let u = f.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// An f64 rounded to F32, then to BF16.
pub fn r(x: f64) -> f64 {
    bf(to_bf(x as f32)) as f64
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uni(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uni().max(1e-300);
        let u2 = self.uni();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `f(i)` for i in 0..n on every hardware thread, results in order.
pub fn par_map<R: Send, F: Fn(usize) -> R + Sync>(n: usize, f: F) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .min(n.max(1));
    let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                s.spawn(move || {
                    (w..n)
                        .step_by(threads)
                        .map(|i| (i, f(i)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, v) in h.join().unwrap() {
                out[i] = Some(v);
            }
        }
    });
    out.into_iter().map(Option::unwrap).collect()
}

pub struct Err {
    pub rel_l2: f64,
    pub max_abs: f64,
    pub finite: bool,
}

pub fn err_of<A: Copy + Into<f64>>(got: &[A], want: &[f64]) -> Err {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den, mut max_abs, mut finite) = (0.0f64, 0.0f64, 0.0f64, true);
    for (&g, &w) in got.iter().zip(want) {
        let g: f64 = g.into();
        finite &= g.is_finite();
        let d = g - w;
        num += d * d;
        den += w * w;
        max_abs = max_abs.max(d.abs());
    }
    Err {
        rel_l2: if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        },
        max_abs,
        finite,
    }
}

// ---------------------------------------------------------------------------------------------
// Layer inputs.

#[derive(Clone)]
pub struct Inputs {
    pub t: usize,
    pub state_pos: usize,
    /// [t][CONV] BF16 (the qkv GEMM output).
    pub qkv: Vec<u16>,
    /// [SLOTS][CONV] F32 ring before the layer.
    pub ring: Vec<f32>,
    /// [CONV][4].
    pub conv_w: Vec<f32>,
    /// [t][96] F32: a, then b.
    pub ab: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    /// [H][D][D] state before the layer, [head][value][key].
    pub s0: Vec<f32>,
}

impl Inputs {
    /// Synthetic layer weights: `ssm_a = -exp(A_log)` spans -0.036 .. -72 log-uniformly over the
    /// heads, so fast heads underflow their in-chunk decays and slow ones carry state across chunks.
    pub fn fresh(t: usize, seed: u64) -> Self {
        let mut g = Rng::new(11);
        let conv_w = (0..CONV * 4).map(|_| (0.5 * g.normal()) as f32).collect();
        let dt_bias = (0..H).map(|_| g.normal() as f32).collect();
        let ssm_a = (0..H)
            .map(|h| {
                let la = 0.036f64.ln() + (72.0f64.ln() - 0.036f64.ln()) * h as f64 / (H - 1) as f64;
                -la.exp() as f32
            })
            .collect();
        let mut inp = Inputs {
            t: 0,
            state_pos: 0,
            qkv: Vec::new(),
            ring: vec![0.0; SLOTS * CONV],
            conv_w,
            ab: Vec::new(),
            dt_bias,
            ssm_a,
            s0: vec![0.0; H * D * D],
        };
        inp.tokens(t, seed);
        inp
    }

    /// `t` tokens: qkv ~ N(0, 1) in BF16, a and b ~ N(0, 4) in F32 (not BF16 values).
    pub fn tokens(&mut self, t: usize, seed: u64) {
        let mut g = Rng::new(seed);
        self.t = t;
        self.qkv = (0..t * CONV).map(|_| to_bf(g.normal() as f32)).collect();
        self.ab = (0..t * AB).map(|_| (2.0 * g.normal()) as f32).collect();
    }

    /// The same weights and `t` new tokens after a `prefix`-token prompt: the oracle's state and
    /// ring after the prefix.
    pub fn continuation(t: usize, prefix: usize, seed: u64) -> Self {
        let p = Inputs::fresh(prefix, 7);
        let o = oracle(&p);
        let mut inp = p;
        inp.s0 = o.s.iter().map(|&v| v as f32).collect();
        inp.ring = o.ring;
        inp.state_pos = o.state_pos;
        inp.tokens(t, seed);
        inp
    }

    /// The same layer made ill-conditioned, as real activations are: each channel's inputs are
    /// 0.8 x a fixed value + 0.6 x noise (correlated keys), a is halved and shifted by -4 (slow decay)
    /// and b halved and shifted by +4 (beta near 1), so I + A is far from the identity.
    pub fn ill_conditioned(mut self, seed: u64) -> Self {
        let mut g = Rng::new(seed);
        let base: Vec<f64> = (0..CONV).map(|_| g.normal()).collect();
        for (i, q) in self.qkv.iter_mut().enumerate() {
            *q = to_bf((0.8 * base[i % CONV] + 0.6 * g.normal()) as f32);
        }
        for (i, v) in self.ab.iter_mut().enumerate() {
            *v = *v / 2.0 + if i % AB < H { -4.0 } else { 4.0 };
        }
        self
    }

    /// Tokens `[a, b)` as a layer of their own, with this layer's starting state.
    pub fn slice(&self, a: usize, b: usize) -> Self {
        let mut s = self.clone();
        s.t = b - a;
        s.qkv = self.qkv[a * CONV..b * CONV].to_vec();
        s.ab = self.ab[a * AB..b * AB].to_vec();
        s
    }

    pub fn x_at(&self, s: isize, c: usize) -> f64 {
        if s >= 0 {
            bf(self.qkv[s as usize * CONV + c]) as f64
        } else {
            let slot = (self.state_pos as isize + s + SLOTS as isize) as usize % SLOTS;
            self.ring[slot * CONV + c] as f64
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host models.

#[derive(Clone, Default)]
pub struct Model {
    /// [t][CONV] conv output, q/k normalized.
    pub cv: Vec<f64>,
    /// [t][H][D] layer output.
    pub out: Vec<f64>,
    /// [H][D][D] final state.
    pub s: Vec<f64>,
    /// [SLOTS][CONV] ring after the layer.
    pub ring: Vec<f32>,
    pub state_pos: usize,
    /// The output and final state of the same emulation with K K^T and the solve in F32, as the
    /// kernels do them (a rounded emulation only): how far two valid computations of the rounded form
    /// land apart on these inputs.
    pub f32_solve_out: Vec<f64>,
    pub f32_solve_s: Vec<f64>,
    /// [t][H] in-chunk gate sums (emulation only).
    pub g: Vec<f64>,
    /// [t][H][D] U (emulation only).
    pub u: Vec<f64>,
}

pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Conv + SiLU and the q/k L2 norm (scale 1 / max(norm, 1e-12)); with `rb`, the SiLU output is
/// rounded to BF16, normalized over the rounded values and rounded again. Also the next ring.
pub fn conv(inp: &Inputs, rb: bool, m: &mut Model) {
    let t = inp.t;
    let rr = |x: f64| if rb { r(x) } else { x };
    let cols: Vec<Vec<f64>> = par_map(CONV, |c| {
        (0..t)
            .map(|tt| {
                let mut s = 0.0;
                for tap in 0..4 {
                    s += inp.conv_w[c * 4 + tap] as f64
                        * inp.x_at(tt as isize - 3 + tap as isize, c);
                }
                rr(s / (1.0 + (-s).exp()))
            })
            .collect()
    });
    let mut cv = vec![0.0; t * CONV];
    for (c, col) in cols.iter().enumerate() {
        for (tt, &v) in col.iter().enumerate() {
            cv[tt * CONV + c] = v;
        }
    }
    for tt in 0..t {
        for head in 0..2 * HK {
            let v = &mut cv[tt * CONV + head * D..tt * CONV + (head + 1) * D];
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let scale = if norm > 1e-12 { 1.0 / norm } else { 1e12 };
            for x in v.iter_mut() {
                *x = rr(*x * scale);
            }
        }
    }
    m.cv = cv;
    m.ring = inp.ring.clone();
    for s in t.saturating_sub(SLOTS)..t {
        let slot = (inp.state_pos + s) % SLOTS;
        for c in 0..CONV {
            m.ring[slot * CONV + c] = bf(inp.qkv[s * CONV + c]);
        }
    }
    m.state_pos = (inp.state_pos + t) % SLOTS;
}

/// Gate and beta of token `tt`, head `h`; with `rb`, from BF16 a and b and beta rounded to BF16.
pub fn gate_beta(inp: &Inputs, tt: usize, h: usize, rb: bool) -> (f64, f64) {
    let rr = |x: f64| if rb { r(x) } else { x };
    let a = rr(inp.ab[tt * AB + h] as f64);
    let b = rr(inp.ab[tt * AB + H + h] as f64);
    let gate = inp.ssm_a[h] as f64 * softplus(a + inp.dt_bias[h] as f64);
    (gate, rr(1.0 / (1.0 + (-b).exp())))
}

/// Lumen's recurrence, one token at a time: decay, retrieve, delta, update, output.
pub fn oracle(inp: &Inputs) -> Model {
    let mut m = Model::default();
    conv(inp, false, &mut m);
    let t = inp.t;
    let qs = 1.0 / (D as f64).sqrt();
    let cv = &m.cv;
    let heads: Vec<(Vec<f64>, Vec<f64>)> = par_map(H, |h| {
        let kh = h % HK;
        let mut s: Vec<f64> = inp.s0[h * D * D..(h + 1) * D * D]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mut out = vec![0.0; t * D];
        for tt in 0..t {
            let row = &cv[tt * CONV..(tt + 1) * CONV];
            let (q, k, v) = (
                &row[kh * D..(kh + 1) * D],
                &row[QK + kh * D..QK + (kh + 1) * D],
                &row[2 * QK + h * D..2 * QK + (h + 1) * D],
            );
            let (gate, beta) = gate_beta(inp, tt, h, false);
            let a = gate.exp();
            for vj in 0..D {
                let sr = &mut s[vj * D..(vj + 1) * D];
                let mut ret = 0.0;
                for ki in 0..D {
                    sr[ki] *= a;
                    ret += sr[ki] * k[ki];
                }
                let delta = beta * (v[vj] - ret);
                let mut acc = 0.0;
                for ki in 0..D {
                    sr[ki] += k[ki] * delta;
                    acc += sr[ki] * q[ki];
                }
                out[tt * D + vj] = acc * qs;
            }
        }
        (s, out)
    });
    m.out = vec![0.0; t * H * D];
    m.s = vec![0.0; H * D * D];
    for (h, (s, out)) in heads.into_iter().enumerate() {
        m.s[h * D * D..(h + 1) * D * D].copy_from_slice(&s);
        for tt in 0..t {
            m.out[(tt * H + h) * D..(tt * H + h + 1) * D]
                .copy_from_slice(&out[tt * D..(tt + 1) * D]);
        }
    }
    m
}

/// The chunked form; with `rb`, rounded where the native kernels hold BF16, and then also computed
/// with K K^T and the solve in F32 (`f32_solve_*`).
pub fn emulate(inp: &Inputs, rb: bool) -> Model {
    let mut m = emulate_with(inp, rb, false);
    if rb {
        let f = emulate_with(inp, true, true);
        (m.f32_solve_out, m.f32_solve_s) = (f.out, f.s);
    }
    m
}

pub fn emulate_with(inp: &Inputs, rb: bool, f32_solve: bool) -> Model {
    let mut m = Model::default();
    conv(inp, rb, &mut m);
    let rr = |x: f64| if rb { r(x) } else { x };
    let t = inp.t;
    let qs = 1.0 / (D as f64).sqrt();
    let cv = &m.cv;
    let heads: Vec<_> = par_map(H, |h| {
        let kh = h % HK;
        let mut s: Vec<f64> = inp.s0[h * D * D..(h + 1) * D * D]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (mut out, mut gs, mut us) = (vec![0.0; t * D], vec![0.0; t], vec![0.0; t * D]);
        let q = |i: usize, d: usize| cv[i * CONV + kh * D + d];
        let k = |i: usize, d: usize| cv[i * CONV + QK + kh * D + d];
        let v = |i: usize, d: usize| cv[i * CONV + 2 * QK + h * D + d];
        for t0 in (0..t).step_by(BT) {
            let n = BT.min(t - t0);
            let (mut g, mut b) = (vec![0.0; n], vec![0.0; n]);
            let mut acc = 0.0;
            for i in 0..n {
                let (gate, beta) = gate_beta(inp, t0 + i, h, rb);
                acc += gate;
                g[i] = acc;
                b[i] = beta;
                gs[t0 + i] = acc;
            }
            let (mut a, mut aqk) = (vec![0.0; n * n], vec![0.0; n * n]);
            for i in 0..n {
                for j in 0..=i {
                    let (mut kk, mut qk, mut kk32) = (0.0, 0.0, 0.0f32);
                    for d in 0..D {
                        kk += k(t0 + i, d) * k(t0 + j, d);
                        kk32 = (k(t0 + i, d) as f32).mul_add(k(t0 + j, d) as f32, kk32);
                        qk += q(t0 + i, d) * k(t0 + j, d);
                    }
                    let e = (g[i] - g[j]).exp();
                    if j < i {
                        a[i * n + j] = if f32_solve {
                            (kk32 * e as f32 * b[i] as f32) as f64
                        } else {
                            kk * e * b[i]
                        };
                    }
                    aqk[i * n + j] = rr(qk * e);
                }
            }
            // X = (I + A)^-1, unit lower triangular, by forward substitution.
            let mut x = vec![0.0; n * n];
            for j in 0..n {
                x[j * n + j] = 1.0;
                for i in j + 1..n {
                    let s: f64 = if f32_solve {
                        (j..i).fold(0.0f32, |s, kk| {
                            (a[i * n + kk] as f32).mul_add(x[kk * n + j] as f32, s)
                        }) as f64
                    } else {
                        (j..i).map(|kk| a[i * n + kk] * x[kk * n + j]).sum()
                    };
                    x[i * n + j] = -s;
                }
            }
            let x: Vec<f64> = x.into_iter().map(rr).collect();
            let kb: Vec<f64> = (0..n * D)
                .map(|e| rr(k(t0 + e / D, e % D) * b[e / D] * g[e / D].exp()))
                .collect();
            let vb: Vec<f64> = (0..n * D)
                .map(|e| rr(v(t0 + e / D, e % D) * b[e / D]))
                .collect();
            let (mut w, mut u) = (vec![0.0; n * D], vec![0.0; n * D]);
            for i in 0..n {
                for d in 0..D {
                    let (mut sw, mut su) = (0.0, 0.0);
                    for j in 0..=i {
                        sw += x[i * n + j] * kb[j * D + d];
                        su += x[i * n + j] * vb[j * D + d];
                    }
                    w[i * D + d] = rr(sw);
                    u[i * D + d] = rr(su);
                    us[(t0 + i) * D + d] = u[i * D + d];
                }
            }
            let sb: Vec<f64> = s.iter().map(|&e| rr(e)).collect();
            let gl = g[n - 1];
            let (mut vn, mut vd, mut o) = (vec![0.0; n * D], vec![0.0; n * D], vec![0.0; n * D]);
            for i in 0..n {
                for vj in 0..D {
                    let sr = &sb[vj * D..(vj + 1) * D];
                    let (mut p, mut qo) = (0.0, 0.0);
                    for d in 0..D {
                        p += w[i * D + d] * sr[d];
                        qo += q(t0 + i, d) * sr[d];
                    }
                    let nv = u[i * D + vj] - p;
                    vn[i * D + vj] = rr(nv);
                    vd[i * D + vj] = rr(nv * (gl - g[i]).exp());
                    o[i * D + vj] = qo * g[i].exp();
                }
            }
            for i in 0..n {
                for vj in 0..D {
                    let mut acc = o[i * D + vj];
                    for j in 0..=i {
                        acc += aqk[i * n + j] * vn[j * D + vj];
                    }
                    out[(t0 + i) * D + vj] = rr(acc * qs);
                }
            }
            let eg = gl.exp();
            for vj in 0..D {
                for d in 0..D {
                    let mut acc = s[vj * D + d] * eg;
                    for i in 0..n {
                        acc += vd[i * D + vj] * k(t0 + i, d);
                    }
                    s[vj * D + d] = acc;
                }
            }
        }
        (s, out, gs, us)
    });
    m.out = vec![0.0; t * H * D];
    m.s = vec![0.0; H * D * D];
    m.g = vec![0.0; t * H];
    m.u = vec![0.0; t * H * D];
    for (h, (s, out, gs, us)) in heads.into_iter().enumerate() {
        m.s[h * D * D..(h + 1) * D * D].copy_from_slice(&s);
        for tt in 0..t {
            m.out[(tt * H + h) * D..(tt * H + h + 1) * D]
                .copy_from_slice(&out[tt * D..(tt + 1) * D]);
            m.u[(tt * H + h) * D..(tt * H + h + 1) * D].copy_from_slice(&us[tt * D..(tt + 1) * D]);
            m.g[tt * H + h] = gs[tt];
        }
    }
    m
}

/// The emulation of `inp` run as consecutive slices ending at `cuts`, each slice starting from the
/// previous one's F32 state and ring, as the kernels hand them over.
pub fn emulate_sliced(inp: &Inputs, cuts: &[usize]) -> Model {
    let mut m = emulate_sliced_with(inp, cuts, false);
    let f = emulate_sliced_with(inp, cuts, true);
    (m.f32_solve_out, m.f32_solve_s) = (f.out, f.s);
    m
}

pub fn emulate_sliced_with(inp: &Inputs, cuts: &[usize], f32_solve: bool) -> Model {
    let (mut all, mut a, mut cur) = (Model::default(), 0, inp.clone());
    for &b in cuts {
        let part = cur.slice(a, b);
        let e = emulate_with(&part, true, f32_solve);
        all.cv.extend_from_slice(&e.cv);
        all.out.extend_from_slice(&e.out);
        cur.s0 = e.s.iter().map(|&v| v as f32).collect();
        cur.ring = e.ring.clone();
        cur.state_pos = e.state_pos;
        all.s = e.s;
        all.ring = e.ring;
        all.state_pos = e.state_pos;
        a = b;
    }
    all
}

// ---------------------------------------------------------------------------------------------
// The gate.

/// What the kernels produced for one layer, widened.
#[derive(Clone)]
pub struct Got {
    pub cv: Vec<f32>,
    pub gc: Vec<f32>,
    pub u: Vec<f32>,
    pub out: Vec<f32>,
    pub s: Vec<f32>,
    pub ring: Vec<f32>,
    pub state_pos: usize,
}

/// Conv output elements outside their bound against the oracle, and the worst error / bound.
/// v: one BF16 rounding of an F32 conv, (2^-8 + 2^-22)|x| + 1.1 e_conv. q/k: the SiLU rounding, the
/// norm factor's error and the final rounding, (3 (2^-8 + 2^-22) + 1.1 |e_conv| / norm)|x| + 1.1
/// e_conv / norm. e_conv = 2^-21 sum_tap |w x| bounds the four-term F32 sum; 1.1 bounds SiLU's slope.
pub fn conv_violations(inp: &Inputs, orc: &Model, got: &[f32]) -> (usize, f64) {
    let t = inp.t;
    let rows: Vec<(Vec<f64>, Vec<f64>)> = par_map(t, |tt| {
        let mut raw = vec![0.0; 2 * QK];
        let mut ec = vec![0.0; CONV];
        for c in 0..CONV {
            let (mut s, mut a) = (0.0, 0.0);
            for tap in 0..4 {
                let p =
                    inp.conv_w[c * 4 + tap] as f64 * inp.x_at(tt as isize - 3 + tap as isize, c);
                s += p;
                a += p.abs();
            }
            ec[c] = a * 2f64.powi(-21);
            if c < 2 * QK {
                raw[c] = s / (1.0 + (-s).exp());
            }
        }
        (raw, ec)
    });
    let unit = 2f64.powi(-8) + 2f64.powi(-22);
    let (mut bad, mut worst) = (0, 0.0f64);
    for (tt, (raw, ec)) in rows.iter().enumerate() {
        for c in 0..CONV {
            let i = tt * CONV + c;
            let x = orc.cv[i];
            let e = (got[i] as f64 - x).abs();
            let tol = if c < 2 * QK {
                let head = c / D;
                let span = head * D..(head + 1) * D;
                let norm = raw[span.clone()]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                let ecn = ec[span].iter().map(|v| v * v).sum::<f64>().sqrt();
                (3.0 * unit + 1.1 * ecn / norm) * x.abs() + 1.1 * ec[c] / norm
            } else {
                unit * x.abs() + 1.1 * ec[c]
            };
            if e.is_nan() || e > tol {
                bad += 1;
            }
            worst = worst.max(e / tol);
        }
    }
    (bad, worst)
}

/// Relative L2 distance allowed between the kernels' G and U and the emulation's.
pub const G_TOL: f64 = 1e-5;
pub const U_TOL: f64 = 2e-4;

/// The bit patterns of F32 values, so +0 and -0 differ.
pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every value of a run as bits, field by field, so +0 and -0 differ.
pub fn got_bits(g: &Got) -> [Vec<u32>; 7] {
    [
        bits(&g.cv),
        bits(&g.gc),
        bits(&g.u),
        bits(&g.out),
        bits(&g.s),
        bits(&g.ring),
        vec![g.state_pos as u32],
    ]
}

/// A gate's result: the names of the checks that failed, and the measurements.
pub struct Verdict {
    pub failed: Vec<&'static str>,
    pub msg: String,
}

impl Verdict {
    pub fn ok(&self) -> bool {
        self.failed.is_empty()
    }
}

/// The gate of one layer against the oracle `orc` and the emulation `emu` (made with the run's own
/// chunk boundaries). Checks: `finite`; `out_oracle` / `state_oracle` (relative L2 within 1.1x the
/// emulation's error against the oracle); `out_max` / `state_max` (largest error within 1.5x the
/// emulation's); `out_emulation` / `state_emulation` (relative L2 to the emulation within 0.2x / 0.15x
/// its error); `conv` (the per-element bound); `ring` (bit-exact). `intermediates` adds `G` and `U`,
/// which only a one-shot run has.
pub fn gate(inp: &Inputs, got: &Got, orc: &Model, emu: &Model, intermediates: bool) -> Verdict {
    let eo = err_of(&got.out, &orc.out);
    let es = err_of(&got.s, &orc.s);
    let mo = err_of(&emu.out, &orc.out);
    let ms = err_of(&emu.s, &orc.s);
    let po = err_of(&got.out, &emu.out);
    let ps = err_of(&got.s, &emu.s);
    let (bad, worst) = conv_violations(inp, orc, &got.cv);
    // The distance limits never fall below 1.5x the distance between the two valid emulations.
    let spread_out = err_of(&emu.f32_solve_out, &emu.out).rel_l2 / mo.rel_l2;
    let spread_s = err_of(&emu.f32_solve_s, &emu.s).rel_l2 / ms.rel_l2;
    let (lim_out, lim_s) = (0.2f64.max(1.5 * spread_out), 0.15f64.max(1.5 * spread_s));
    let mut checks = vec![
        ("finite", eo.finite && es.finite),
        ("out_oracle", eo.rel_l2 <= 1.1 * mo.rel_l2 + 1e-7),
        ("out_max", eo.max_abs <= 1.5 * mo.max_abs + 1e-7),
        ("out_emulation", po.rel_l2 <= lim_out * mo.rel_l2 + 1e-7),
        ("state_oracle", es.rel_l2 <= 1.1 * ms.rel_l2 + 1e-7),
        ("state_max", es.max_abs <= 1.5 * ms.max_abs + 1e-7),
        ("state_emulation", ps.rel_l2 <= lim_s * ms.rel_l2 + 1e-7),
        ("conv", bad == 0),
        ("ring", bits(&got.ring) == bits(&orc.ring)),
    ];
    let mut msg = format!(
        "out relL2 {:.3e} ({:.3}x emulation {:.3e}) max {:.2}x to-emu {:.3}x | state relL2 {:.3e} \
         ({:.3}x emulation {:.3e}) max {:.2}x to-emu {:.3}x | limits {lim_out:.3}x / {lim_s:.3}x | conv \
         violations {bad} (worst {worst:.3})",
        eo.rel_l2,
        eo.rel_l2 / mo.rel_l2,
        mo.rel_l2,
        eo.max_abs / mo.max_abs,
        po.rel_l2 / mo.rel_l2,
        es.rel_l2,
        es.rel_l2 / ms.rel_l2,
        ms.rel_l2,
        es.max_abs / ms.max_abs,
        ps.rel_l2 / ms.rel_l2,
    );
    if intermediates {
        let eg = err_of(&got.gc, &emu.g);
        let eu = err_of(&got.u, &emu.u);
        msg += &format!(" | G to-emu {:.2e} | U to-emu {:.2e}", eg.rel_l2, eu.rel_l2);
        checks.push(("G", eg.finite && eg.rel_l2 <= G_TOL));
        checks.push(("U", eu.finite && eu.rel_l2 <= U_TOL));
    }
    let failed: Vec<&'static str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
    if !failed.is_empty() {
        msg += &format!(" | failed {failed:?}");
    }
    Verdict { failed, msg }
}
