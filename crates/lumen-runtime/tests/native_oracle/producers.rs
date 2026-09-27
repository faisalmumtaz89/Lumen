//! The host implementation of the native prefill's producer kernels' arithmetic
//! (`native_prefill_kernels`), byte for byte: every IEEE step computed here, the approximate hardware
//! instructions (rcp/rsqrt/ex2 approximations, `div.full`, `div.approx`) taken from a probe kernel
//! run on exactly the operands computed here. Used by `cuda_native_producers_test` and the layer
//! oracle.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill_kernels::fp4_scale_bytes;
use lumen_runtime::cuda::native_prefill_weights::swizzled_offset;

pub const H: usize = 5120;
pub const I: usize = 17408;
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

pub fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

pub fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}

pub fn add_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) + ftz(b))
}

pub fn e4m3_value(c: u8) -> f32 {
    let (e, m) = ((c >> 3) & 15, c & 7);
    let v = if e == 0 {
        m as f32 / 512.0
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if c & 0x80 != 0 {
        -v
    } else {
        v
    }
}

pub fn e4m3_dec(c: u8) -> f32 {
    if c & 0x7f == 0x7f {
        f32::NAN
    } else {
        e4m3_value(c)
    }
}

/// Nearest entry of an ascending table (exact comparison in f64), ties to the even code.
pub fn nearest_even(table: &[f32], a: f32) -> usize {
    let hi = table.partition_point(|&t| t < a);
    if hi == 0 {
        return 0;
    }
    if hi == table.len() {
        return hi - 1;
    }
    let lo = hi - 1;
    let (dl, dh) = (a as f64 - table[lo] as f64, table[hi] as f64 - a as f64);
    if dl < dh || (dl == dh && lo % 2 == 0) {
        lo
    } else {
        hi
    }
}

/// cvt.rn.satfinite.e4m3: nearest even, |v| >= 448 -> +-448, NaN -> 0x7F.
pub fn e4m3_enc(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs();
    if a >= 448.0 {
        return s | 0x7e;
    }
    s | nearest_even(e4m3_table(), a) as u8
}

/// The 127 non-negative finite E4M3 values, ascending by code.
pub fn e4m3_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..127u8).map(e4m3_value).collect())
}

pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// cvt.rn.satfinite.e2m1: nearest even, |v| >= 6 -> +-6, NaN -> +6.
pub fn e2m1_enc(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7;
    }
    let s = if v.is_sign_negative() { 0x8 } else { 0 };
    if v.abs() >= 6.0 {
        return s | 0x7;
    }
    s | nearest_even(&E2M1, v.abs()) as u8
}

/// Whether `v` lies exactly halfway between two neighbours of `table` (a rounding tie).
pub fn is_tie(table: &[f32], v: f32) -> bool {
    let a = v.abs();
    table
        .windows(2)
        .any(|w| a > w[0] && a < w[1] && (a as f64 - w[0] as f64) == (w[1] as f64 - a as f64))
}

// ---------------------------------------------------------------------------------------------
// Hardware probe of the approximate instructions.

pub const PROBE_SOURCE: &str = r#"
extern "C" __global__ void probe(const float* a, const float* b, float* out, unsigned int n, unsigned int op)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], r;
    if (op == 0) asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    else if (op == 1) asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    else if (op == 2) asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    else if (op == 3) asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    else if (op == 4) asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
    else asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
    out[i] = r;
}
"#;

#[derive(Clone, Copy)]
pub enum Op {
    Rcp = 0,
    Rsqrt = 1,
    Ex2Ftz = 2,
    Ex2 = 3,
    DivFull = 4,
    DivApproxFtz = 5,
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

    pub fn unary(&self, op: Op, a: &[f32]) -> Vec<f32> {
        self.run(op, a, &vec![0.0; a.len()])
    }

    pub fn one(&self, op: Op, a: f32, b: f32) -> f32 {
        self.run(op, &[a], &[b])[0]
    }
}

// ---------------------------------------------------------------------------------------------
// The oracle.

pub const LOG2E: u32 = 0x3FB8AA3B;
pub const NEG_LOG2E: u32 = 0xBFB8AA3B;

/// Tables over all 65536 BF16 inputs of the element-wise approximate functions.
pub struct Tables {
    /// Sigmoid of the element-wise gates: 1 / (1 + ex2.approx((0 - z) * log2 e)) with div.full.
    pub sigmoid: Vec<f32>,
    /// Fast-math SwiGLU: silu(g) = div.approx.ftz(g, 1 + ex2.approx.ftz(g * -log2 e)).
    pub silu_fast: Vec<f32>,
    /// IEEE SwiGLU: the divisor 1 + expf(-g) of the standard library's expf with its fused + 1.
    pub silu_precise_den: Vec<f32>,
}

impl Tables {
    pub fn new(hw: &Hw) -> Self {
        let g: Vec<f32> = (0..=u16::MAX).map(bf).collect();
        let t: Vec<f32> = g
            .iter()
            .map(|&z| (0.0f32 - z) * f32::from_bits(LOG2E))
            .collect();
        let e = hw.unary(Op::Ex2, &t);
        let a: Vec<f32> = e.iter().map(|&e| e + 1.0).collect();
        let sigmoid = hw.run(Op::DivFull, &vec![1.0; a.len()], &a);

        let t: Vec<f32> = g
            .iter()
            .map(|&x| mul_ftz(x, f32::from_bits(NEG_LOG2E)))
            .collect();
        let e = hw.unary(Op::Ex2Ftz, &t);
        let d: Vec<f32> = e.iter().map(|&e| add_ftz(e, 1.0)).collect();
        let silu_fast = hw.run(Op::DivApproxFtz, &g, &d);

        let mut js = Vec::with_capacity(g.len());
        let f: Vec<f32> = g
            .iter()
            .map(|&x| {
                let a = x.mul_add(f32::from_bits(0xBBBB989D), 0.5);
                let c = if a.is_nan() { 0.0 } else { a.clamp(0.0, 1.0) };
                // fma.rm(c, 252, 12582913): the F32 grid is the integers here, so rounding down is
                // the floor of the exact value.
                let j = (12582913.0f64 + (c as f64 * 252.0).floor()) as f32;
                let k = j + f32::from_bits(0xCB40007F);
                js.push(j);
                x.mul_add(
                    f32::from_bits(0xB2A57060),
                    x.mul_add(f32::from_bits(0xBFB8AA3B), -k),
                )
            })
            .collect();
        let e = hw.unary(Op::Ex2Ftz, &f);
        let silu_precise_den = e
            .iter()
            .zip(&js)
            .map(|(&e, &j)| e.mul_add(f32::from_bits(j.to_bits() << 23), 1.0))
            .collect();
        Self {
            sigmoid,
            silu_fast,
            silu_precise_den,
        }
    }

    pub fn silu_mul(&self, precise: bool, g: u16, u: u16) -> f32 {
        if precise {
            bf(u) * (bf(g) / self.silu_precise_den[g as usize])
        } else {
            mul_ftz(bf(u), self.silu_fast[g as usize])
        }
    }
}

/// The FP8 static quantizer's reciprocal scale, div.full(1, input_scale).
pub fn fp8_inv(hw: &Hw, input_scale: f32) -> f32 {
    hw.one(Op::DivFull, 1.0, input_scale)
}

pub fn fp8(x: f32, inv: f32) -> u8 {
    e4m3_enc((x * inv).max(-448.0).min(448.0))
}

/// The NVFP4 quantizer at global scale `s`.
pub struct Fp4 {
    pub s: f32,
    pub r6: f32,
    /// rcp(f32(SF) * rcp(S)) for every scale code SF.
    pub scale_of: Vec<f32>,
    /// Write the block scales row-major instead of swizzled (a corrupted oracle).
    pub linear: bool,
}

impl Fp4 {
    pub fn new(hw: &Hw, s: f32) -> Self {
        let r6 = hw.one(Op::Rcp, 6.0, 0.0);
        let rs = hw.one(Op::Rcp, s, 0.0);
        let args: Vec<f32> = (0..=255u8).map(|c| mul_ftz(e4m3_dec(c), rs)).collect();
        Self {
            s,
            r6,
            scale_of: hw.unary(Op::Rcp, &args),
            linear: false,
        }
    }

    /// Codes [m][k / 2] and scales (fp4_scale_bytes(m, k), rows [m, pad128(m)) zero) of x [m][k] BF16.
    pub fn quantize(&self, x: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let nb = k / 16;
        let mut q = vec![0u8; m * k / 2];
        let mut sf = vec![0u8; fp4_scale_bytes(m as u32, k as u32)];
        for row in 0..m {
            for b in 0..nb {
                let v = &x[row * k + b * 16..row * k + b * 16 + 16];
                // Lane-wise NaN-ignoring maxima of the even and odd elements, then (x > y) ? x : y.
                let lane = |start: usize| {
                    let mut acc = bf(v[start]).abs();
                    for i in 1..8 {
                        let a = bf(v[start + 2 * i]).abs();
                        acc = if acc.is_nan() {
                            a
                        } else if a.is_nan() {
                            acc
                        } else {
                            acc.max(a)
                        };
                    }
                    acc
                };
                let (lx, ly) = (lane(0), lane(1));
                let vec_max = if lx > ly { lx } else { ly };
                let code = e4m3_enc(mul_ftz(self.s, mul_ftz(vec_max, self.r6)));
                let nonzero = ftz(vec_max) != 0.0;
                let scale = if nonzero {
                    self.scale_of[code as usize]
                } else {
                    0.0
                };
                let off = if self.linear {
                    row * nb + b
                } else {
                    swizzled_offset(row, b, nb)
                };
                sf[off] = code;
                for i in 0..8 {
                    q[row * k / 2 + b * 8 + i] = e2m1_enc(mul_ftz(bf(v[2 * i]), scale))
                        | (e2m1_enc(mul_ftz(bf(v[2 * i + 1]), scale)) << 4);
                }
            }
        }
        (q, sf)
    }
}

/// The hidden-size Gemma RMSNorm: (new residual, BF16 output). `rstd_ulps` moves every rstd by that
/// many ulps (a corrupted oracle).
pub fn gemma_norm(
    hw: &Hw,
    x: &[u16],
    resid: Option<&[u16]>,
    w1: &[f32],
    m: usize,
    rstd_ulps: i32,
) -> (Vec<u16>, Vec<u16>) {
    let mut h = vec![0f32; m * H];
    for (i, v) in h.iter_mut().enumerate() {
        *v = match resid {
            Some(r) => bf(x[i]) + bf(r[i]),
            None => bf(x[i]),
        };
    }
    let new_resid: Vec<u16> = h.iter().map(|&v| to_bf(v)).collect();
    let args: Vec<f32> = (0..m)
        .map(|row| {
            let hr = &h[row * H..row * H + H];
            let lane: Vec<f32> = (0..64)
                .map(|t| {
                    let mut s = 0f32;
                    for v1 in 0..10 {
                        for v0 in 0..8 {
                            let c = v1 * 512 + t * 8 + v0;
                            s = hr[c].mul_add(hr[c], s);
                        }
                    }
                    s
                })
                .collect();
            let warp = |w: usize| butterfly(&lane[w * 32..w * 32 + 32], &[1, 2, 4, 8, 16]);
            let total = warp(0) + warp(1);
            EPS + total / H as f32
        })
        .collect();
    let rstd = hw.unary(Op::Rsqrt, &args);
    let mut out = vec![0u16; m * H];
    for row in 0..m {
        let r = f32::from_bits((rstd[row].to_bits() as i64 + rstd_ulps as i64) as u32);
        for c in 0..H {
            out[row * H + c] = to_bf((h[row * H + c] * r) * w1[c]);
        }
    }
    (new_resid, out)
}

/// A shuffle-xor butterfly over `lanes` with the given offsets; every lane ends equal, lane 0's value.
pub fn butterfly(lanes: &[f32], offsets: &[usize]) -> f32 {
    let mut v = lanes.to_vec();
    for &o in offsets {
        v = (0..v.len()).map(|i| v[i] + v[i ^ o]).collect();
    }
    v[0]
}

/// The GDN gated norm's BF16 output for `rows` rows of 128, reduction shape `lanes` (32 or 16).
#[allow(clippy::too_many_arguments)]
pub fn gated_norm(
    hw: &Hw,
    tab: &Tables,
    x: &[u16],
    z: &[u16],
    w: &[f32],
    rows: usize,
    lanes: usize,
    rstd_ulps: i32,
) -> Vec<u16> {
    let n = 128 / lanes;
    let offsets: Vec<usize> =
        std::iter::successors(Some(lanes / 2), |&o| (o > 1).then_some(o / 2)).collect();
    let sums: Vec<f32> = (0..rows)
        .map(|row| {
            let partial: Vec<f32> = (0..lanes)
                .map(|l| {
                    let v: Vec<f32> = (0..n).map(|i| bf(x[row * 128 + l * n + i])).collect();
                    let mut s = v[1] * v[1];
                    s = v[0].mul_add(v[0], s);
                    for &vi in &v[2..] {
                        s = vi.mul_add(vi, s);
                    }
                    s
                })
                .collect();
            butterfly(&partial, &offsets)
        })
        .collect();
    let mean = hw.run(Op::DivFull, &sums, &vec![128.0; rows]);
    let args: Vec<f32> = mean.iter().map(|&v| EPS + v).collect();
    let rstd = hw.unary(Op::Rsqrt, &args);
    let mut out = vec![0u16; rows * 128];
    for row in 0..rows {
        let r = f32::from_bits((rstd[row].to_bits() as i64 + rstd_ulps as i64) as u32);
        for c in 0..128 {
            let i = row * 128 + c;
            let zi = z[i];
            let gate = tab.sigmoid[zi as usize] * bf(zi);
            out[i] = to_bf(((r * bf(x[i])) * w[c]) * gate);
        }
    }
    out
}

pub fn silu_bf16(tab: &Tables, precise: bool, gu: &[u16], m: usize) -> Vec<u16> {
    let mut out = vec![0u16; m * I];
    for row in 0..m {
        for c in 0..I {
            let (g, u) = (gu[row * 2 * I + c], gu[row * 2 * I + I + c]);
            out[row * I + c] = to_bf(tab.silu_mul(precise, g, u));
        }
    }
    out
}

pub fn sigmoid_bf16(tab: &Tables, o: &[u16], gate: &[u16]) -> Vec<u16> {
    o.iter()
        .zip(gate)
        .map(|(&o, &g)| to_bf(tab.sigmoid[g as usize] * bf(o)))
        .collect()
}

pub fn fp8_all(x: &[u16], inv: f32) -> Vec<u8> {
    x.iter().map(|&v| fp8(bf(v), inv)).collect()
}

pub fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
