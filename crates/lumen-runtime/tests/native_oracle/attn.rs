//! Host models of the native prefill's attention kernels (`native_prefill_attn`), used by
//! `cuda_native_attention_test` and the layer oracle: the prep (q/k norm, RoPE, gate split) byte for
//! byte, taking only the approximate hardware instructions (`rsqrt.approx.ftz`, `div.full`) from a
//! probe kernel run on exactly the operands computed here; an exact float64 attention; its emulation
//! rounding where the kernel's format rounds; the kernel's algorithm in F32; and the gate between
//! them.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill_attn::scale_log2;

pub const HQ: usize = 24;
pub const HKV: usize = 4;
pub const D: usize = 256;
pub const QD: usize = HQ * D;
pub const KVD: usize = HKV * D;
pub const ROT: usize = 64;
pub const TILE: usize = 64;
pub const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------------------------
// Number formats.

pub fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// F32 -> BF16, round to nearest even; NaN -> 0x7FFF.
pub fn to_bf(f: f32) -> u16 {
    let u = f.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fff;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

pub fn rbf(f: f32) -> f32 {
    bf(to_bf(f))
}

pub fn trunc_bf(f: f32) -> f32 {
    f32::from_bits(f.to_bits() & 0xffff_0000)
}

pub fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bytes32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------
// Approximate hardware instructions, taken from the device on the oracle's own operands.

pub const PROBE_SOURCE: &str = r#"
extern "C" __global__ void probe(const float* a, const float* b, float* out, unsigned int n, unsigned int op)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], r;
    if (op == 0) asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    else asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
    out[i] = r;
}
"#;

#[derive(Clone, Copy)]
pub enum Op {
    Rsqrt = 0,
    DivFull = 1,
}

pub struct Hw<'a> {
    pub dev: &'a CudaDevice,
    pub f: CudaFunction,
}

impl<'a> Hw<'a> {
    pub fn new(dev: &'a CudaDevice) -> Self {
        let m = dev.compile_and_load(PROBE_SOURCE).expect("probe module");
        Self {
            dev,
            f: m.load_function("probe").expect("probe"),
        }
    }

    pub fn run(&self, op: Op, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return Vec::new();
        }
        let da = self.dev.htod_copy(a).unwrap();
        let db = self.dev.htod_copy(b).unwrap();
        let mut out = self.dev.alloc_zeros::<f32>(a.len()).unwrap();
        let (n, opc) = (a.len() as u32, op as u32);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.dev
                .stream
                .launch_builder(&self.f)
                .arg(&da)
                .arg(&db)
                .arg(&mut out)
                .arg(&n)
                .arg(&opc)
                .launch(cfg)
                .unwrap();
        }
        self.dev.dtoh_copy(&out).unwrap()
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs.

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next() as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

pub fn normal_bf16(rng: &mut Rng, n: usize, sd: f64) -> Vec<u16> {
    (0..n).map(|_| to_bf((rng.normal() * sd) as f32)).collect()
}

/// A Gemma norm weight as the artifact stores it: F32 (w + 1) of a BF16 w, times `scale`.
pub fn gemma_weights(rng: &mut Rng, scale: f32) -> Vec<f32> {
    (0..D)
        .map(|_| (bf(to_bf((rng.normal() * 0.3) as f32)) + 1.0) * scale)
        .collect()
}

/// The projections of `t` tokens: `qg` [t][24][512], `k` and `v` [t][4][256] BF16, and the norm
/// weights.
#[derive(Clone)]
pub struct PrepIn {
    pub t: usize,
    pub qg: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub q_w1: Vec<f32>,
    pub k_w1: Vec<f32>,
}

impl PrepIn {
    /// Normal projections; tokens 5, 6 and 7 (when present) are the edge rows: zero, squares that
    /// overflow F32, and BF16 subnormals.
    pub fn realistic(t: usize, seed: u64, w_scale: f32) -> Self {
        let mut rng = Rng(seed);
        let mut p = Self {
            t,
            qg: normal_bf16(&mut rng, t * HQ * 2 * D, 2.0),
            k: normal_bf16(&mut rng, t * KVD, 2.0),
            v: normal_bf16(&mut rng, t * KVD, 1.0),
            q_w1: gemma_weights(&mut rng, w_scale),
            k_w1: gemma_weights(&mut rng, w_scale),
        };
        let edges: [fn(&mut Rng) -> u16; 3] = [
            |_| 0,
            |r| to_bf((r.normal() * 3e20) as f32),
            |r| (r.next() % 0x7f) as u16 | if r.next() % 2 == 0 { 0x8000 } else { 0 },
        ];
        for (i, e) in edges.iter().enumerate() {
            let tok = 5 + i;
            if tok < t {
                for x in &mut p.qg[tok * HQ * 2 * D..(tok + 1) * HQ * 2 * D] {
                    *x = e(&mut rng);
                }
                for x in &mut p.k[tok * KVD..(tok + 1) * KVD] {
                    *x = e(&mut rng);
                }
            }
        }
        p
    }

    pub fn prefix(&self, t: usize) -> Self {
        Self {
            t,
            qg: self.qg[..t * HQ * 2 * D].to_vec(),
            k: self.k[..t * KVD].to_vec(),
            v: self.v[..t * KVD].to_vec(),
            q_w1: self.q_w1.clone(),
            k_w1: self.k_w1.clone(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The prep oracle.

/// A head's sum of squares in the kernel's order: thread t (warp t / 32) takes fma(x0, x0, x1 * x1)
/// of its two columns, each warp a butterfly over offsets 16..1, then (w0 + w2) + (w1 + w3).
pub fn head_sum(x: &[f32]) -> f32 {
    let mut warps = [0.0f32; 4];
    for (w, sum) in warps.iter_mut().enumerate() {
        let mut lanes: Vec<f32> = (0..32)
            .map(|l| {
                let c = 2 * (w * 32 + l);
                x[c].mul_add(x[c], x[c + 1] * x[c + 1])
            })
            .collect();
        for o in [16, 8, 4, 2, 1] {
            let prev = lanes.clone();
            for (l, v) in lanes.iter_mut().enumerate() {
                *v = prev[l] + prev[l ^ o];
            }
        }
        *sum = lanes[0];
    }
    (warps[0] + warps[2]) + (warps[1] + warps[3])
}

#[derive(Clone, Copy, PartialEq)]
pub enum Variant {
    Kernel,
    RstdUlp,
    OtherContraction,
}

/// The prep's BF16 outputs: q [t][24][256], gate [t][24][256], k [t][4][256].
pub struct PrepOut {
    pub q: Vec<u16>,
    pub gate: Vec<u16>,
    pub k: Vec<u16>,
}

pub fn prep_oracle(hw: &Hw, inp: &PrepIn, cs: &[f32], p0: usize, variant: Variant) -> PrepOut {
    let heads = HQ + HKV;
    let head_in = |t: usize, h: usize| -> &[u16] {
        if h < HQ {
            &inp.qg[(t * HQ + h) * 2 * D..][..D]
        } else {
            &inp.k[(t * HKV + h - HQ) * D..][..D]
        }
    };
    let sums: Vec<f32> = (0..inp.t * heads)
        .map(|i| {
            let x: Vec<f32> = head_in(i / heads, i % heads)
                .iter()
                .map(|&b| bf(b))
                .collect();
            head_sum(&x)
        })
        .collect();
    let var = hw.run(Op::DivFull, &sums, &vec![256.0; sums.len()]);
    let arg: Vec<f32> = var.iter().map(|&v| v + EPS).collect();
    let mut rstd = hw.run(Op::Rsqrt, &arg, &vec![0.0; arg.len()]);
    if variant == Variant::RstdUlp {
        for r in &mut rstd {
            *r = f32::from_bits(r.to_bits() + 1);
        }
    }
    let mut out = PrepOut {
        q: vec![0; inp.t * QD],
        gate: vec![0; inp.t * QD],
        k: vec![0; inp.t * KVD],
    };
    for t in 0..inp.t {
        let row = &cs[(p0 + t) * ROT..][..ROT];
        for h in 0..heads {
            let r = rstd[t * heads + h];
            let w1 = if h < HQ { &inp.q_w1 } else { &inp.k_w1 };
            let mut n: Vec<u16> = head_in(t, h)
                .iter()
                .zip(w1)
                .map(|(&x, &w)| to_bf((r * bf(x)) * w))
                .collect();
            for d in 0..ROT / 2 {
                let (x1, x2) = (bf(n[d]), bf(n[d + ROT / 2]));
                let (c, s) = (row[d], row[ROT / 2 + d]);
                let (o1, o2) = match variant {
                    Variant::OtherContraction => ((-x2).mul_add(s, x1 * c), x2.mul_add(c, x1 * s)),
                    _ => (x1.mul_add(c, -(x2 * s)), x1.mul_add(s, x2 * c)),
                };
                n[d] = to_bf(o1);
                n[d + ROT / 2] = to_bf(o2);
            }
            if h < HQ {
                out.q[(t * HQ + h) * D..][..D].copy_from_slice(&n);
                out.gate[(t * HQ + h) * D..][..D]
                    .copy_from_slice(&inp.qg[(t * HQ + h) * 2 * D + D..][..D]);
            } else {
                out.k[(t * HKV + h - HQ) * D..][..D].copy_from_slice(&n);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The attention oracles.

/// One attention problem in the kernel's input values: q [T][24][256] and the caches
/// [4][max_seq][256] holding positions [0, p0 + t).
pub struct AttnCase {
    pub p0: usize,
    pub max_seq: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Rows checked against the oracles: all for T <= 256, otherwise the first and last 70, every row
/// within 2 of a 64-multiple, and every 11th row.
pub fn check_rows(t: usize) -> Vec<usize> {
    (0..t)
        .filter(|&r| {
            t <= 256 || r < 70 || r + 70 >= t || r % 64 <= 2 || r % 64 >= 62 || r % 11 == 0
        })
        .collect()
}

/// `f(row index, head)` for every checked row and head, on all cores; the results in row-major
/// [rows][24][256] order.
pub fn per_row_head(rows: &[usize], f: impl Fn(usize, usize) -> Vec<f64> + Sync) -> Vec<f64> {
    let jobs = rows.len() * HQ;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = jobs.div_ceil(threads).max(1);
    let mut out = vec![0.0; jobs * D];
    std::thread::scope(|s| {
        for (c, part) in out.chunks_mut(chunk * D).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, o) in part.chunks_mut(D).enumerate() {
                    let job = c * chunk + i;
                    o.copy_from_slice(&f(rows[job / HQ], job % HQ));
                }
            });
        }
    });
    out
}

/// Softmax attention in float64. `emulate` rounds the operands, the probabilities (the denominator
/// summed from the rounded ones) and the output to BF16, with the probabilities as 2^(s * log2 e / 16
/// - max); otherwise exact.
pub fn attn_oracle(c: &AttnCase, rows: &[usize], emulate: bool) -> Vec<f64> {
    let r = |x: f64| if emulate { rbf(x as f32) as f64 } else { x };
    per_row_head(rows, |t, h| {
        let hk = h / (HQ / HKV);
        let n = c.p0 + t + 1;
        let qv: Vec<f64> = (0..D).map(|d| r(c.q[t * QD + h * D + d] as f64)).collect();
        let mut s: Vec<f64> = (0..n)
            .map(|j| {
                let kr = &c.k[(hk * c.max_seq + j) * D..][..D];
                let dot: f64 = qv.iter().zip(kr).map(|(a, &b)| a * r(b as f64)).sum();
                dot / 16.0
                    * if emulate {
                        std::f64::consts::LOG2_E
                    } else {
                        1.0
                    }
            })
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0.0;
        for x in &mut s {
            *x = if emulate {
                r((*x - m).exp2())
            } else {
                (*x - m).exp()
            };
            l += *x;
        }
        let mut o = vec![0.0; D];
        for (j, p) in s.iter().enumerate() {
            let vr = &c.v[(hk * c.max_seq + j) * D..][..D];
            for (a, &b) in o.iter_mut().zip(vr) {
                *a += p * r(b as f64);
            }
        }
        o.iter().map(|&x| r(x / l)).collect()
    })
}

/// The kernel's algorithm in F32: 64-key tiles, a running maximum of the raw scores, probabilities
/// exp2(fma(s, c, -(m * c))) rounded to BF16 per tile, the denominator summed from them. `trunc_k`
/// stages the keys by truncation instead of rounding to nearest.
pub fn attn_model_f32(c: &AttnCase, rows: &[usize], trunc_k: bool) -> Vec<f64> {
    let sl2 = scale_log2();
    per_row_head(rows, |t, h| {
        let hk = h / (HQ / HKV);
        let qp = c.p0 + t;
        let qv: Vec<f32> = (0..D).map(|d| rbf(c.q[t * QD + h * D + d])).collect();
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        let mut acc = vec![0.0f32; D];
        let mut sc = vec![0.0f32; TILE];
        for kv0 in (0..=qp).step_by(TILE) {
            let mut mx = m;
            for (j, s) in sc.iter_mut().enumerate() {
                let key = kv0 + j;
                *s = if key > qp {
                    f32::NEG_INFINITY
                } else {
                    let kr = &c.k[(hk * c.max_seq + key) * D..][..D];
                    qv.iter()
                        .zip(kr)
                        .map(|(a, &b)| a * if trunc_k { trunc_bf(b) } else { rbf(b) })
                        .sum()
                };
                mx = mx.max(*s);
            }
            let mc = -(mx * sl2);
            let alpha = m.mul_add(sl2, mc).exp2();
            m = mx;
            let mut ls = 0.0f32;
            for a in &mut acc {
                *a *= alpha;
            }
            for (j, &s) in sc.iter().enumerate() {
                let p = rbf(s.mul_add(sl2, mc).exp2());
                ls += p;
                if p != 0.0 {
                    let vr = &c.v[(hk * c.max_seq + kv0 + j) * D..][..D];
                    for (a, &b) in acc.iter_mut().zip(vr) {
                        *a += p * rbf(b);
                    }
                }
            }
            l = l * alpha + ls;
        }
        acc.iter().map(|&a| rbf(a * (1.0 / l)) as f64).collect()
    })
}

pub struct Err {
    pub rel_l2: f64,
    pub max_abs: f64,
    pub finite: bool,
}

pub fn err_of(got: &[f64], exact: &[f64]) -> Err {
    let (mut num, mut den, mut max_abs, mut finite) = (0.0, 0.0, 0.0f64, true);
    for (&g, &r) in got.iter().zip(exact) {
        finite &= g.is_finite();
        num += (g - r) * (g - r);
        den += r * r;
        max_abs = max_abs.max((g - r).abs());
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

/// The gate: relative L2 within 1.1x and the largest error within 1.5x of the emulation's own error
/// against the exact oracle.
pub fn gate(got: &[f64], exact: &[f64], emu: &[f64]) -> (bool, String) {
    let (e, m) = (err_of(got, exact), err_of(emu, exact));
    let ok = e.finite && e.rel_l2 <= 1.1 * m.rel_l2 + 1e-6 && e.max_abs <= 1.5 * m.max_abs + 1e-6;
    (
        ok,
        format!(
            "relL2 {:.3e} ({:.3}x emulation {:.3e}), max {:.3e} ({:.2}x emulation {:.3e})",
            e.rel_l2,
            e.rel_l2 / m.rel_l2,
            m.rel_l2,
            e.max_abs,
            e.max_abs / m.max_abs,
            m.max_abs
        ),
    )
}

pub fn gather(out: &[u16], rows: &[usize]) -> Vec<f64> {
    rows.iter()
        .flat_map(|&t| out[t * QD..(t + 1) * QD].iter().map(|&b| bf(b) as f64))
        .collect()
}
