//! The native prefill's layer oracle (`native_oracle::layer`), qualified before the route is
//! assembled: one layer at a time is run through the native components (the producer kernels, the
//! cuBLASLt plans, the GDN and attention kernels, the prefill weight views) as the route will chain
//! them, its tensors recorded by the dump hook (`native_prefill_dump`), and the oracle must pass the
//! layer as run and catch it changed at one point. The runner reads its weights and scales as the route does
//! (`PrefillWeightViews`), not through the oracle's loader.
//!
//! Layers 0 to 2 (GDN) and 3 (attention) of the real model, on the first tokens of a real prompt: its
//! embedded rows at layer 0, each layer's output feeding the next. The attention layer runs the
//! attention kernels (`native_prefill_attn`) on its own KV cache, and the oracle checks it with the
//! native attention core (`Attention`).
//!
//! - `layer_oracle_is_qualified`: every check passes on 16 tokens from zero state and on the next 16
//!   (ring position 1, attention from position 16), with every layer's inputs the previous layer's
//!   outputs and every layer's second-slice state its first slice's; without an attention core,
//!   exactly `attention_core` and `layer` fail on the attention layer. Run changed at one point, the
//!   second slice must fail the checks named for the change: in layer 1, one FP4 code of the
//!   post-attention norm flipped; the gate/up activation scale taken as its members' minimum (the up
//!   projection's scale raised by a quarter, read by both sides, so minimum and maximum differ); the
//!   residual stored before the add; the conv ring written as if its position were 0; in layer 3,
//!   the KV rows written for position p0 + 1. Each kind of check must reject a good dump with one
//!   tensor changed past its bound, and a changed handoff must be caught.
//! - `layer_oracle_runtime`: layers 0 to 3 at 7 (the gated norm's one-row-per-warp shape on a GPU of
//!   168 or more SMs), 16, 64 and 128 tokens pass, with the oracle's time per layer.
//!
//! Needs a GPU of compute capability 12.0 with NVRTC and cuBLASLt 12.8 or newer, the real artifact
//! (`LUMEN_NATIVE_MODEL`), a file of the prompt's token ids separated by white space
//! (`LUMEN_NATIVE_IDS`, at least 128), and a `LUMEN_CACHE_DIR`:
//!
//!   cargo test --release -p lumen-runtime --features test-prefill-dump \
//!     --test cuda_native_prefill_test -- --ignored --test-threads=1 --nocapture
#![cfg(feature = "test-prefill-dump")]

mod native_oracle;

use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{CudaSlice, DevicePtr};
use lumen_format::index::SubtensorOffsets;
use lumen_format::Nvfp4Planes;
use lumen_runtime::cuda::cublaslt::LtOperands;
use lumen_runtime::cuda::cublaslt_algo_cache::WeightOperand;
use lumen_runtime::cuda::ffi::CudaDevice;
use lumen_runtime::cuda::native_prefill::{
    input_scale, is_attention_layer, ProviderSlices, SliceSource, ROPE,
};
use lumen_runtime::cuda::native_prefill_attn::NativeAttnKernels;
use lumen_runtime::cuda::native_prefill_dump::PrefillDump;
use lumen_runtime::cuda::native_prefill_gdn::{next_conv_position, NativeGdnKernels};
use lumen_runtime::cuda::native_prefill_gemm::{
    bucket, bucket_rows, NativeGemm, TableSource, ATTN_KV, ATTN_Q, DOWN, GATE_UP, GDN_QKV, GDN_Z,
    OUT,
};
use lumen_runtime::cuda::native_prefill_kernels::{
    fp4_scale_bytes, gdn_norm_lanes_per_row, NativePrefillKernels,
};
use lumen_runtime::cuda::native_prefill_weights::{LayerScales, PrefillWeightViews};
use lumen_runtime::cuda::shaders::{
    NATIVE_PREFILL_ATTN_KERNEL_SOURCE, NATIVE_PREFILL_GDN_KERNEL_SOURCE,
};
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use native_oracle::layer::{
    handoff, Attention, AttentionCore, LayerDump, LayerVerdict, LayerWeights, Oracle, ATTN_OUT, H,
    I,
};
use native_oracle::{attn, gdn, producers as p};
use std::time::Instant;

/// The layers this suite runs.
const LAYERS: usize = 4;

// ---------------------------------------------------------------------------------------------
// The layer as the route will run it.

/// A change at one point, for the negative controls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Defect {
    None,
    /// One FP4 code of the post-attention norm's output flipped before the gate/up GEMMs.
    FlipX4Code,
    /// The input norm's residual left as it was before the add.
    ResidBeforeAdd,
}

/// GDN or attention kernels to run in place of the loaded ones (altered, for the controls).
#[derive(Clone, Copy)]
enum Alt<'a> {
    None,
    Gdn(&'a NativeGdnKernels),
    Attn(&'a NativeAttnKernels),
}

/// A layer's weights on the device as the route holds them: the planes as stored (codes, then
/// scales), the norms, and a GDN layer's conv weights, dt bias, `ssm_a` and gated-norm weight.
struct DevLayer {
    inputs: Vec<CudaSlice<u8>>,
    out: CudaSlice<u8>,
    /// Gate, up and down planes.
    mlp: [CudaSlice<u8>; 3],
    norm_in: CudaSlice<f32>,
    norm_post: CudaSlice<f32>,
    gdn: Option<[CudaSlice<f32>; 4]>,
    /// An attention layer's q and k norm weights, F32 (w + 1) `[256]`.
    qk: Option<[CudaSlice<f32>; 2]>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl DevLayer {
    fn upload(dev: &CudaDevice, src: &dyn SliceSource, l: usize, st: &SubtensorOffsets) -> Self {
        let bytes = |s: &lumen_format::index::TensorSlice| src.read(l, s, 0, s.length).unwrap();
        let up8 = |s: &lumen_format::index::TensorSlice| dev.htod_copy(&bytes(s)).unwrap();
        let up32 = |s: &lumen_format::index::TensorSlice| dev.htod_copy(&f32s(&bytes(s))).unwrap();
        let (inputs, out) = if is_attention_layer(l) {
            (vec![up8(&st.wq), up8(&st.wk), up8(&st.wv)], up8(&st.wo))
        } else {
            (
                vec![up8(&st.wq), up8(st.attn_gate.as_ref().unwrap())],
                up8(st.ssm_out.as_ref().unwrap()),
            )
        };
        Self {
            inputs,
            out,
            mlp: [up8(&st.w_gate), up8(&st.w_up), up8(&st.w_down)],
            norm_in: up32(&st.attn_norm),
            norm_post: up32(st.attn_post_norm.as_ref().unwrap()),
            gdn: (!is_attention_layer(l)).then(|| {
                [&st.ssm_conv1d, &st.ssm_dt, &st.ssm_a, &st.ssm_norm]
                    .map(|s| up32(s.as_ref().unwrap()))
            }),
            qk: is_attention_layer(l)
                .then(|| [&st.attn_q_norm, &st.attn_k_norm].map(|s| up32(s.as_ref().unwrap()))),
        }
    }
}

/// Positions the attention layers' KV caches hold.
const MAX_SEQ: usize = 256;

/// A layer's state across slices: a GDN layer's ring, state and ring position, or an attention
/// layer's F32 KV cache `[2][4][MAX_SEQ][256]` (K, then V) and the next position.
#[derive(Clone)]
enum State {
    Gdn {
        ring: Vec<f32>,
        state: Vec<f32>,
        pos: u32,
    },
    Attn {
        kv: Vec<f32>,
        p0: u32,
    },
}

impl State {
    fn zero(l: usize) -> Self {
        if is_attention_layer(l) {
            State::Attn {
                kv: vec![0.0; 2 * attn::KVD * MAX_SEQ],
                p0: 0,
            }
        } else {
            State::Gdn {
                ring: vec![0.0; gdn::SLOTS * gdn::CONV],
                state: vec![0.0; gdn::H * gdn::D * gdn::D],
                pos: 0,
            }
        }
    }
}

/// A layer's outputs: the MLP output and the residual, `[t][5120]`.
struct Out {
    mlp_out: Vec<u16>,
    resid: Vec<u16>,
}

/// The native components a layer runs through.
struct Ctx {
    k: NativePrefillKernels,
    gk: NativeGdnKernels,
    ak: NativeAttnKernels,
    /// The RoPE table, `MAX_SEQ` rows.
    cs: CudaSlice<f32>,
    gemm: NativeGemm,
    views: PrefillWeightViews,
    /// Each layer's NVFP4 global scales (gate, up, down).
    globals: Vec<[f32; 3]>,
    eps: f32,
}

fn ptr<T>(dev: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&dev.stream).0
}

/// `v` (`[t][cols]`) followed by zero rows up to `m` rows.
fn padded(v: &[u16], m: usize, cols: usize) -> Vec<u16> {
    let mut out = v.to_vec();
    out.resize(m * cols, 0);
    out
}

/// Run layer `l` on `t` tokens: `x` its input rows, `resid` the residual entering it (`None` at layer
/// 0, whose residual is `x`), `st` its state (updated), `scales` its activation scales, `alt` GDN or
/// attention kernels in place of the loaded ones. Every tensor of the layer is recorded in `dump`.
#[allow(clippy::too_many_arguments)]
fn run_layer(
    dev: &CudaDevice,
    c: &mut Ctx,
    l: usize,
    dl: &DevLayer,
    x: &[u16],
    resid: Option<&[u16]>,
    st: &mut State,
    t: usize,
    scales: LayerScales,
    defect: Defect,
    alt: Alt,
    dump: &mut PrefillDump,
) -> Out {
    let (k, eps) = (&c.k, c.eps);
    let gk = match alt {
        Alt::Gdn(g) => g,
        _ => &c.gk,
    };
    let ak = match alt {
        Alt::Attn(a) => a,
        _ => &c.ak,
    };
    let cs = &c.cs;
    let views = &c.views;
    let [g_gate, g_up, g_down] = c.globals[l];
    let plans = &mut c.gemm;
    let attention = is_attention_layer(l);
    let m = bucket_rows(bucket(t).unwrap());
    let tu = t as u32;
    let z16 = |n: usize| dev.alloc_zeros::<u16>(n).unwrap();
    let z8 = |n: usize| dev.alloc_zeros::<u8>(n).unwrap();
    let p = |s: &dyn DevicePtrOf| s.at(dev);
    let rec = |dump: &mut PrefillDump, name: &'static str, at: u64, bytes: usize| unsafe {
        dump.record(dev, l, name, at, bytes).unwrap()
    };
    let mut gemm =
        |shape: usize, alpha: f32, w: u64, w_scale: u64, x: u64, x_scale: u64, d: u64| unsafe {
            plans
                .run(
                    shape,
                    t,
                    alpha,
                    &LtOperands {
                        w,
                        w_scale,
                        x,
                        x_scale,
                        d,
                    },
                )
                .unwrap()
        };
    // An FP8 weight's planes and the address of its weight scale after its `n * k` codes.
    let fp8_w = |s: &CudaSlice<u8>, n: usize, kk: usize| (p(s), p(s) + (n * kk) as u64);

    // The FP8 GEMMs read their activation scales from the device: the layer's pair in the views'
    // table, unless the scales were changed for this run.
    let changed = scales != views.scales[l];
    let own_scales = dev.htod_copy(&[scales.proj_in, scales.proj_out]).unwrap();
    let base = if changed {
        p(&own_scales)
    } else {
        p(&views.fp8_scales) + (8 * l) as u64
    };
    let (s_in, s_out) = (base, base + 4);

    let xd = dev.htod_copy(&padded(x, m, H)).unwrap();
    let mut rd = dev.htod_copy(&padded(resid.unwrap_or(x), m, H)).unwrap();
    rec(dump, "x", p(&xd), t * H * 2);
    if resid.is_some() {
        rec(dump, "resid_in", p(&rd), t * H * 2);
    }

    // The input norm (at layer 0 the residual is the input and is not written).
    let x8 = z8(m * H);
    let normed = z16(m * H);
    let before = dev.dtoh_copy(&rd).unwrap();
    unsafe {
        if resid.is_none() {
            k.rmsnorm_fp8(
                dev,
                p(&rd),
                p(&dl.norm_in),
                eps,
                tu,
                scales.proj_in,
                p(&x8),
                p(&normed),
            )
        } else {
            k.add_rmsnorm_fp8(
                dev,
                p(&xd),
                p(&rd),
                p(&dl.norm_in),
                eps,
                tu,
                scales.proj_in,
                p(&x8),
                p(&normed),
            )
        }
        .unwrap();
    }
    if defect == Defect::ResidBeforeAdd {
        dev.htod_copy_into(&before, &mut rd).unwrap();
    }
    rec(dump, "resid_attn", p(&rd), t * H * 2);
    rec(dump, "normed", p(&normed), t * H * 2);
    rec(dump, "x8", p(&x8), t * H);

    let codes_out = z8(m * ATTN_OUT);
    if !attention {
        let State::Gdn {
            ring: ring_h,
            state: state_h,
            pos,
        } = st
        else {
            panic!("layer {l} is a GDN layer")
        };
        let [conv_w, dt_bias, ssm_a, gnorm] = dl.gdn.as_ref().unwrap();
        let ab_w = views.ab[l].as_ref().unwrap();
        let ring = dev.htod_copy(ring_h).unwrap();
        let state = dev.htod_copy(state_h).unwrap();
        rec(dump, "ring_in", p(&ring), ring_h.len() * 4);
        rec(dump, "state_in", p(&state), state_h.len() * 4);
        dump.record_host(l, "state_pos_in", pos.to_le_bytes().to_vec());

        let qkv = z16(m * gdn::CONV);
        let z = z16(m * 6144);
        let (wp, ws) = fp8_w(&dl.inputs[0], gdn::CONV, H);
        gemm(GDN_QKV, 1.0, wp, ws, p(&x8), s_in, p(&qkv));
        let (wp, ws) = fp8_w(&dl.inputs[1], 6144, H);
        gemm(GDN_Z, 1.0, wp, ws, p(&x8), s_in, p(&z));
        // a and b: the BF16 rows [96][5120] times the normed rows, F32 out, as the route's cuBLAS
        // GEMM computes them.
        let ab = dev.alloc_zeros::<f32>(m * gdn::AB).unwrap();
        let (one, zero) = (1.0f32, 0.0f32);
        let st = unsafe {
            cublas_sys::cublasGemmEx(
                *dev.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                gdn::AB as i32,
                t as i32,
                H as i32,
                &one as *const f32 as *const std::ffi::c_void,
                p(ab_w) as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                H as i32,
                p(&normed) as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                H as i32,
                &zero as *const f32 as *const std::ffi::c_void,
                p(&ab) as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                gdn::AB as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        assert_eq!(
            st,
            cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "a/b GEMM"
        );
        rec(dump, "qkv", p(&qkv), t * gdn::CONV * 2);
        rec(dump, "z", p(&z), t * 6144 * 2);
        rec(dump, "ab", p(&ab), t * gdn::AB * 4);

        let hd = gdn::H * gdn::D;
        let cv = z16(t * gdn::CONV);
        let gc = dev.alloc_zeros::<f32>(t * gdn::H).unwrap();
        let (wb, ub, aqk) = (z16(t * hd), z16(t * hd), z16(t * gdn::H * gdn::BT));
        let core = z16(m * hd);
        unsafe {
            gk.conv(dev, p(&qkv), p(&ring), p(conv_w), p(&cv), tu, *pos)
                .unwrap();
            gk.chunk_intra(
                dev,
                p(&cv),
                p(&ab),
                p(dt_bias),
                p(ssm_a),
                p(&gc),
                p(&wb),
                p(&ub),
                p(&aqk),
                tu,
            )
            .unwrap();
            gk.chunk_state(
                dev,
                p(&cv),
                p(&gc),
                p(&wb),
                p(&ub),
                p(&aqk),
                p(&state),
                p(&core),
                tu,
            )
            .unwrap();
        }
        *pos = next_conv_position(*pos, tu);
        rec(dump, "cv", p(&cv), t * gdn::CONV * 2);
        rec(dump, "ring", p(&ring), ring_h.len() * 4);
        rec(dump, "core", p(&core), t * hd * 2);
        rec(dump, "state", p(&state), state_h.len() * 4);
        dump.record_host(l, "state_pos", pos.to_le_bytes().to_vec());
        *ring_h = dev.dtoh_copy(&ring).unwrap();
        *state_h = dev.dtoh_copy(&state).unwrap();

        let rows = tu * gdn::H as u32;
        let lanes = gdn_norm_lanes_per_row(rows, k.sm_count());
        unsafe {
            k.gdn_norm_gate_fp8(
                dev,
                p(&core),
                p(&z),
                p(gnorm),
                eps,
                rows,
                lanes,
                scales.proj_out,
                p(&codes_out),
            )
            .unwrap();
        }
        rec(dump, "y8", p(&codes_out), t * ATTN_OUT);
    } else {
        let qg = z16(m * 12288);
        let (kk, vv) = (z16(m * 1024), z16(m * 1024));
        for (i, (shape, n, d)) in [
            (ATTN_Q, 12288, &qg),
            (ATTN_KV, 1024, &kk),
            (ATTN_KV, 1024, &vv),
        ]
        .into_iter()
        .enumerate()
        {
            let (wp, ws) = fp8_w(&dl.inputs[i], n, H);
            gemm(shape, 1.0, wp, ws, p(&x8), s_in, p(d));
        }
        rec(dump, "qg", p(&qg), t * 12288 * 2);
        rec(dump, "k", p(&kk), t * 1024 * 2);
        rec(dump, "v", p(&vv), t * 1024 * 2);
        let State::Attn { kv: kv_h, p0 } = st else {
            panic!("layer {l} is an attention layer")
        };
        let [q_w1, k_w1] = dl.qk.as_ref().unwrap();
        let n = attn::KVD * MAX_SEQ;
        let kv = dev.htod_copy(kv_h).unwrap();
        let stage = z16(2 * n);
        let (kc, vc, ks, vs) = (
            p(&kv),
            p(&kv) + 4 * n as u64,
            p(&stage),
            p(&stage) + 2 * n as u64,
        );
        dump.record_host(l, "p0", p0.to_le_bytes().to_vec());
        rec(
            dump,
            "cs",
            p(cs) + (*p0 as usize * attn::ROT * 4) as u64,
            t * attn::ROT * 4,
        );
        rec(dump, "kv_in", kc, 2 * n * 4);
        let (q, gate, o) = (z16(m * ATTN_OUT), z16(m * ATTN_OUT), z16(m * ATTN_OUT));
        let ms = MAX_SEQ as u32;
        unsafe {
            ak.prep(
                dev,
                p(&qg),
                p(&kk),
                p(&vv),
                p(q_w1),
                p(k_w1),
                p(cs),
                eps,
                tu,
                *p0,
                ms,
                p(&q),
                p(&gate),
                kc,
                vc,
                ks,
                vs,
            )
            .unwrap();
            ak.kv_to_bf16(dev, kc, vc, ks, vs, *p0, ms).unwrap();
            ak.attention(dev, p(&q), ks, vs, p(&o), tu, *p0, ms)
                .unwrap();
        }
        rec(dump, "q", p(&q), t * ATTN_OUT * 2);
        rec(dump, "gate", p(&gate), t * ATTN_OUT * 2);
        rec(dump, "kv", kc, 2 * n * 4);
        rec(dump, "o", p(&o), t * ATTN_OUT * 2);
        *kv_h = dev.dtoh_copy(&kv).unwrap();
        *p0 += tu;
        unsafe {
            k.sigmoid_gate_fp8(dev, p(&o), p(&gate), tu, scales.proj_out, p(&codes_out))
                .unwrap();
        }
        rec(dump, "o8", p(&codes_out), t * ATTN_OUT);
    }

    let attn_out = z16(m * H);
    let (wp, ws) = fp8_w(&dl.out, H, ATTN_OUT);
    gemm(OUT, 1.0, wp, ws, p(&codes_out), s_out, p(&attn_out));
    rec(dump, "attn_out", p(&attn_out), t * H * 2);

    // The MLP.
    let x4 = z8(m * H / 2);
    let x4sf = z8(fp4_scale_bytes(tu, H as u32));
    unsafe {
        k.add_rmsnorm_fp4(
            dev,
            p(&attn_out),
            p(&rd),
            p(&dl.norm_post),
            eps,
            tu,
            1.0 / scales.gate_up,
            p(&x4),
            p(&x4sf),
        )
        .unwrap();
    }
    if defect == Defect::FlipX4Code {
        // Bit 1 of a code's magnitude: a different E2M1 value whatever the code.
        let at = p(&x4) + ((3 % t) * H / 2 + 100) as u64;
        let mut b = [0u8; 1];
        dev.synchronize().unwrap();
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut b, at).unwrap();
            b[0] ^= 0x02;
            cudarc::driver::result::memcpy_htod_sync(at, &b).unwrap();
        }
    }
    rec(dump, "resid_mlp", p(&rd), t * H * 2);
    rec(dump, "x4", p(&x4), t * H / 2);
    rec(dump, "x4sf", p(&x4sf), fp4_scale_bytes(tu, H as u32));

    let gu = z16(m * 2 * I);
    let [gate_sf, up_sf, down_sf] = [0, 1, 2].map(|i| p(&views.mlp_scales[l][i]));
    let [gate_w, up_w, down_w] = [0, 1, 2].map(|i| p(&dl.mlp[i]));
    let gs_ = scales.gate_up;
    gemm(
        GATE_UP,
        gs_ * g_gate,
        gate_w,
        gate_sf,
        p(&x4),
        p(&x4sf),
        p(&gu),
    );
    let gu_up = p(&gu) + (I * 2) as u64;
    gemm(GATE_UP, gs_ * g_up, up_w, up_sf, p(&x4), p(&x4sf), gu_up);
    rec(dump, "gu", p(&gu), t * 2 * I * 2);
    let d4 = z8(m * I / 2);
    let d4sf = z8(fp4_scale_bytes(tu, I as u32));
    unsafe {
        k.silu_mul_fp4(
            dev,
            attention,
            p(&gu),
            tu,
            1.0 / scales.down,
            p(&d4),
            p(&d4sf),
        )
        .unwrap();
    }
    rec(dump, "d4", p(&d4), t * I / 2);
    rec(dump, "d4sf", p(&d4sf), fp4_scale_bytes(tu, I as u32));
    let mlp_out = z16(m * H);
    let alpha = scales.down * g_down;
    gemm(DOWN, alpha, down_w, down_sf, p(&d4), p(&d4sf), p(&mlp_out));
    rec(dump, "mlp_out", p(&mlp_out), t * H * 2);
    // The row the route hands to decode after its last layer.
    let x_gpu = dev.alloc_zeros::<f32>(H).unwrap();
    unsafe {
        k.final_row_f32(dev, p(&mlp_out), p(&rd), tu - 1, p(&x_gpu))
            .unwrap();
    }
    rec(dump, "x_gpu", p(&x_gpu), H * 4);
    Out {
        mlp_out: dev.dtoh_copy(&mlp_out).unwrap()[..t * H].to_vec(),
        resid: dev.dtoh_copy(&rd).unwrap()[..t * H].to_vec(),
    }
}

/// A device buffer's address.
trait DevicePtrOf {
    fn at(&self, dev: &CudaDevice) -> u64;
}

impl<T> DevicePtrOf for CudaSlice<T> {
    fn at(&self, dev: &CudaDevice) -> u64 {
        self.device_ptr(&dev.stream).0
    }
}

// ---------------------------------------------------------------------------------------------
// Set-up.

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

struct Model {
    /// The oracle's reading of the layers.
    weights: Vec<LayerWeights>,
    dev_layers: Vec<DevLayer>,
    /// The embedded rows of the prompt, BF16 `[ids][5120]`.
    embedded: Vec<u16>,
    /// Layer 1's gate and up activation scales, read as the route reads them.
    gate_up_members: (f32, f32),
}

/// The real model's layers 0 to 3, the prompt's embedded rows and the native components.
fn setup(tokens: usize) -> (CudaDevice, Ctx, Model) {
    let provider =
        SyncWeightProvider::open(std::path::Path::new(&env("LUMEN_NATIVE_MODEL"))).unwrap();
    let lbc = provider.lbc();
    let eps = lbc.header.hyperparams.norm_eps;
    assert_eq!(eps, p::EPS, "the host norms' epsilon");
    let ids: Vec<usize> = std::fs::read_to_string(env("LUMEN_NATIVE_IDS"))
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .take(tokens)
        .collect();
    assert_eq!(
        ids.len(),
        tokens,
        "LUMEN_NATIVE_IDS holds fewer than {tokens} ids"
    );
    assert_eq!(
        provider.embedding_raw.len(),
        lbc.header.hyperparams.vocab_size as usize * H * 2,
        "a BF16 embedding"
    );
    let embedded: Vec<u16> = ids
        .iter()
        .flat_map(|&id| {
            provider.embedding_raw[id * H * 2..(id + 1) * H * 2]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
        })
        .collect();
    let offsets: Vec<SubtensorOffsets> = lbc.layer_indices[..LAYERS]
        .iter()
        .map(|l| l.subtensors.clone())
        .collect();
    let src = ProviderSlices::new(&provider);
    let src: &dyn SliceSource = &src;
    let weights: Vec<LayerWeights> = (0..LAYERS)
        .map(|l| LayerWeights::load(src, l, &offsets[l]))
        .collect();
    let dev = CudaDevice::new(0).unwrap();
    let dev_layers: Vec<DevLayer> = (0..LAYERS)
        .map(|l| DevLayer::upload(&dev, src, l, &offsets[l]))
        .collect();
    // The route's views of the first layers (the views take the layer index from the position).
    let views = PrefillWeightViews::build(&dev, H, I, &offsets, src).unwrap();
    // Each NVFP4 plane's global scale, stored after its codes and block scales.
    let global = |l: usize, s: &lumen_format::index::TensorSlice, n: usize, k: usize| {
        let b = src.read(l, s, (n * k / 2 + n * k / 16) as u64, 4).unwrap();
        f32::from_le_bytes(b.try_into().unwrap())
    };
    let globals: Vec<[f32; 3]> = offsets
        .iter()
        .enumerate()
        .map(|(l, st)| {
            [
                global(l, &st.w_gate, I, H),
                global(l, &st.w_up, I, H),
                global(l, &st.w_down, H, I),
            ]
        })
        .collect();
    let member = |s: &lumen_format::index::TensorSlice| {
        let planes = Nvfp4Planes::for_shape(I as u64, H as u64).unwrap();
        input_scale(src, 1, s, planes.total_bytes()).unwrap()
    };
    let gate_up_members = (member(&offsets[1].w_gate), member(&offsets[1].w_up));

    // The plans' weight operands, read only when the cache holds no table for this library.
    let fp8 = |s: &CudaSlice<u8>, n: usize, k: usize| WeightOperand {
        plane: ptr(&dev, s),
        scale: ptr(&dev, s) + (n * k) as u64,
        unit_alpha: 1.0,
    };
    let fp4 = |l: usize, i: usize| WeightOperand {
        plane: ptr(&dev, &dev_layers[l].mlp[i]),
        scale: ptr(&dev, &views.mlp_scales[l][i]),
        unit_alpha: globals[l][i],
    };
    let (g, a) = (&dev_layers[0], &dev_layers[3]);
    let operands = [
        vec![fp4(0, 0)],
        vec![fp4(0, 2)],
        vec![fp8(&g.inputs[0], gdn::CONV, H)],
        vec![fp8(&g.inputs[1], 6144, H)],
        vec![fp8(&g.out, H, ATTN_OUT)],
        vec![fp8(&a.inputs[0], 12288, H)],
        vec![fp8(&a.inputs[1], 1024, H)],
    ];
    let (gemm, report) = unsafe { NativeGemm::build(&dev, &operands) }.unwrap();
    let source = match report.source {
        TableSource::Cached(path) => format!("read from {}", path.display()),
        TableSource::Selected { reason, .. } => format!("selected ({reason})"),
    };
    println!(
        "plans {source} in {:.1} s, {}",
        report.seconds, report.library
    );
    let k = NativePrefillKernels::load(&dev).unwrap_or_else(|e| panic!("{e}"));
    let gk = NativeGdnKernels::load(&dev).unwrap_or_else(|e| panic!("{e}"));
    let ak = NativeAttnKernels::load(&dev).unwrap_or_else(|e| panic!("{e}"));
    let cs = dev.alloc_zeros::<f32>(MAX_SEQ * attn::ROT).unwrap();
    unsafe { ak.rope_table(&dev, ptr(&dev, &cs), MAX_SEQ as u32, ROPE.theta) }.unwrap();
    (
        dev,
        Ctx {
            k,
            gk,
            ak,
            cs,
            gemm,
            views,
            globals,
            eps,
        },
        Model {
            weights,
            dev_layers,
            embedded,
            gate_up_members,
        },
    )
}

/// Collects the checks of a test; the test fails at the end if any failed.
#[derive(Default)]
struct Checks {
    failed: Vec<String>,
    passed: usize,
}

impl Checks {
    fn check(&mut self, ok: bool, what: &str, detail: &str) {
        println!("[{}] {what}\n{detail}", if ok { "PASS" } else { "FAIL" });
        if ok {
            self.passed += 1;
        } else {
            self.failed.push(what.to_string());
        }
    }

    fn finish(self) {
        println!("SUMMARY pass={} fail={}", self.passed, self.failed.len());
        assert!(self.failed.is_empty(), "failed: {:?}", self.failed);
        assert!(self.passed > 0);
    }
}

/// The oracle's checks of layer `w` of `dump`, and the seconds they took.
fn verdict(
    o: &Oracle,
    w: &LayerWeights,
    dump: &PrefillDump,
    t: usize,
    core: Option<&dyn AttentionCore>,
) -> (LayerVerdict, f64) {
    let start = Instant::now();
    let v = o.check(
        w,
        &LayerDump {
            dump,
            layer: w.layer,
            t,
        },
        core,
    );
    (v, start.elapsed().as_secs_f64())
}

fn sm_count(dev: &CudaDevice) -> u32 {
    dev.ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap() as u32
}

fn at(dump: &PrefillDump, layer: usize, t: usize) -> LayerDump<'_> {
    LayerDump { dump, layer, t }
}

/// A copy of layer `l` of `dump` with tensor `name` changed by `change`.
fn changed(dump: &PrefillDump, l: usize, name: &str, change: &dyn Fn(&mut Vec<u8>)) -> PrefillDump {
    let mut d = PrefillDump::new([l]);
    for t in dump.tensors().iter().filter(|t| t.layer == l) {
        let mut b = t.bytes.clone();
        if t.name == name {
            change(&mut b);
            assert_ne!(b, t.bytes, "{name} unchanged");
        }
        d.record_host(l, t.name, b);
    }
    d
}

// ---------------------------------------------------------------------------------------------
// Tests.

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8, the real artifact and a prompt"]
fn layer_oracle_is_qualified() {
    const T: usize = 16;
    let (dev, mut c, mut model) = setup(2 * T);
    let hw = p::Hw::new(&dev);
    let tab = p::Tables::new(&hw);
    let o = Oracle {
        hw: &hw,
        tab: &tab,
        sm_count: sm_count(&dev),
    };
    let core = Attention {
        hw: attn::Hw::new(&dev),
    };
    let mut l = Checks::default();
    let mut states: Vec<State> = (0..LAYERS).map(State::zero).collect();
    // Layers 1 and 3's inputs in the second slice, and each slice's dump.
    let (mut control_in, mut dumps) = (vec![None; LAYERS], Vec::new());
    for slice in 0..2 {
        let rows = &model.embedded[slice * T * H..(slice + 1) * T * H];
        let mut dump = PrefillDump::new(0..LAYERS);
        let (mut x, mut resid) = (rows.to_vec(), None::<Vec<u16>>);
        for layer in 0..LAYERS {
            let w = &model.weights[layer];
            if slice == 1 {
                control_in[layer] = Some((x.clone(), resid.clone(), states[layer].clone()));
            }
            let scales = c.views.scales[layer];
            let out = run_layer(
                &dev,
                &mut c,
                layer,
                &model.dev_layers[layer],
                &x,
                resid.as_deref(),
                &mut states[layer],
                T,
                scales,
                Defect::None,
                Alt::None,
                &mut dump,
            );
            let at_pos = if w.attention {
                format!("from position {}", slice * T)
            } else {
                format!("ring position {}", slice * T % 3)
            };
            let what = format!("slice {slice} ({at_pos}) layer {layer}, {T} tokens");
            if w.attention {
                let (v, _) = verdict(&o, w, &dump, T, None);
                l.check(
                    v.failed() == ["attention_core", "layer"],
                    &format!("{what}, no attention core: exactly attention_core and layer fail"),
                    &v.report(),
                );
                let (v, secs) = verdict(&o, w, &dump, T, Some(&core));
                l.check(v.ok(), &format!("{what} ({secs:.1} s)"), &v.report());
            } else {
                let (v, secs) = verdict(&o, w, &dump, T, None);
                l.check(v.ok(), &format!("{what} ({secs:.1} s)"), &v.report());
            }
            if layer > 0 {
                let v = handoff(&at(&dump, layer - 1, T), &at(&dump, layer, T));
                l.check(
                    v.ok(),
                    &format!("slice {slice} layer {} -> {layer} handoff", layer - 1),
                    &v.report(),
                );
            }
            (x, resid) = (out.mlp_out, Some(out.resid));
        }
        dumps.push(dump);
    }
    for layer in 0..LAYERS {
        let v = handoff(&at(&dumps[0], layer, T), &at(&dumps[1], layer, T));
        l.check(
            v.ok(),
            &format!("layer {layer} slice 0 -> 1 handoff"),
            &v.report(),
        );
    }

    // Each kind of check rejects a good dump of layer 1 (second slice) changed at one point past its
    // bound.
    let kept = &dumps[1];
    let w1 = &model.weights[1];
    let bf16_at = |i: usize, k: f32| {
        move |b: &mut Vec<u8>| {
            let v = p::bf(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
            b[2 * i..2 * i + 2].copy_from_slice(&p::to_bf(v * k).to_le_bytes());
        }
    };
    let f32_at = |i: usize, k: f32| {
        move |b: &mut Vec<u8>| {
            let v = f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
            b[4 * i..4 * i + 4].copy_from_slice(&(v * k).to_le_bytes());
        }
    };
    let bf16_all = |k: f32| {
        move |b: &mut Vec<u8>| {
            for c in b.chunks_exact_mut(2) {
                let v = p::bf(u16::from_le_bytes([c[0], c[1]]));
                c.copy_from_slice(&p::to_bf(v * k).to_le_bytes());
            }
        }
    };
    let f32_all = |k: f32| {
        move |b: &mut Vec<u8>| {
            for c in b.chunks_exact_mut(4) {
                let v = f32::from_le_bytes(c.try_into().unwrap());
                c.copy_from_slice(&(v * k).to_le_bytes());
            }
        }
    };
    let byte_xor = |i: usize, m: u8| move |b: &mut Vec<u8>| b[i] ^= m;
    type Change = Box<dyn Fn(&mut Vec<u8>)>;
    let dump_controls: Vec<(&str, &str, Change)> = vec![
        ("normed", "norm_in.normed", Box::new(byte_xor(2 * 77, 1))),
        ("qkv", "qkv", Box::new(bf16_at(5 * gdn::CONV + 9, 1.25))),
        ("z", "z", Box::new(bf16_at(7 * 6144 + 11, 1.25))),
        ("ab", "ab", Box::new(f32_at(3 * gdn::AB + 50, 1.1))),
        (
            "cv",
            "gdn.conv",
            Box::new(bf16_at(2 * gdn::CONV + 700, 1.25)),
        ),
        ("core", "gdn.out_oracle", Box::new(bf16_all(1.25))),
        ("y8", "gated_norm.y8", Box::new(byte_xor(1234, 1))),
        ("attn_out", "attn_out", Box::new(bf16_at(2 * H + 13, 1.25))),
        ("gu", "gate", Box::new(bf16_at(4 * 2 * I + 17, 1.25))),
        ("gu", "up", Box::new(bf16_at(4 * 2 * I + I + 17, 1.25))),
        ("d4sf", "swiglu.d4sf", Box::new(byte_xor(19, 1))),
        ("mlp_out", "down", Box::new(bf16_at(6 * H + 21, 1.25))),
        ("x_gpu", "final_row", Box::new(f32_at(99, 1.1))),
        ("state_pos", "gdn.state_pos", Box::new(byte_xor(0, 1))),
        ("mlp_out", "layer.mlp_out", Box::new(bf16_all(1.25))),
        ("state", "layer.state", Box::new(f32_all(1.25))),
    ];
    for (name, check, change) in dump_controls {
        let d = changed(kept, 1, name, &*change);
        let (v, _) = verdict(&o, w1, &d, T, None);
        l.check(
            v.failed().contains(&check),
            &format!("dump control: {name} changed, rejected by {check}"),
            &format!("failed {:?}", v.failed()),
        );
    }
    let attn_controls: Vec<(&str, &str, Change)> = vec![
        ("cs", "attention_core.rope", Box::new(f32_all(0.0))),
        ("q", "attention_core.q", Box::new(byte_xor(2 * 300, 1))),
        (
            "gate",
            "attention_core.gate",
            Box::new(byte_xor(2 * 301, 1)),
        ),
        ("kv", "attention_core.kv", Box::new(f32_at(5, 1.1))),
        ("o", "attention_core.attention", Box::new(bf16_all(1.25))),
        ("o8", "sigmoid_gate.o8", Box::new(byte_xor(1234, 1))),
        ("kv", "layer.kv", Box::new(f32_all(1.25))),
    ];
    for (name, check, change) in attn_controls {
        let d = changed(kept, 3, name, &*change);
        let (v, _) = verdict(&o, &model.weights[3], &d, T, Some(&core));
        l.check(
            v.failed().contains(&check),
            &format!("dump control: layer 3 {name} changed, rejected by {check}"),
            &format!("failed {:?}", v.failed()),
        );
    }
    // A dump missing an input fails the checks that need it; it does not stop the suite.
    for (layer, name, trips) in [
        (1, "state_in", &["gdn", "layer"][..]),
        (3, "cs", &["attention_core", "layer"][..]),
        (3, "kv_in", &["attention_core", "layer"][..]),
    ] {
        let mut d = PrefillDump::new([layer]);
        for t in kept
            .tensors()
            .iter()
            .filter(|t| t.layer == layer && t.name != name)
        {
            d.record_host(layer, t.name, t.bytes.clone());
        }
        let (v, _) = verdict(&o, &model.weights[layer], &d, T, Some(&core));
        let failed = v.failed();
        l.check(
            trips.iter().all(|t| failed.contains(t)),
            &format!("dump control: layer {layer} without {name}, rejected by {trips:?}"),
            &format!("failed {failed:?}"),
        );
    }
    let short = changed(kept, 3, "cs", &|b: &mut Vec<u8>| {
        b.truncate(b.len() - 4 * attn::ROT)
    });
    let (v, _) = verdict(&o, &model.weights[3], &short, T, Some(&core));
    l.check(
        v.failed().contains(&"attention_core") && v.failed().contains(&"layer"),
        "dump control: layer 3 with a RoPE table one row short, rejected by attention_core and layer",
        &format!("failed {:?}", v.failed()),
    );
    let d = changed(kept, 2, "x", &bf16_at(3, 1.25));
    let v = handoff(&at(kept, 1, T), &at(&d, 2, T));
    l.check(
        v.failed() == ["handoff.x"],
        "dump control: layer 2's input not layer 1's output, rejected by handoff.x",
        &v.report(),
    );
    let d = changed(&dumps[1], 3, "kv_in", &f32_at(7, 1.25));
    let v = handoff(&at(&dumps[0], 3, T), &at(&d, 3, T));
    l.check(
        v.failed() == ["handoff.kv_in"],
        "dump control: layer 3's second-slice cache not its first slice's, rejected by handoff.kv_in",
        &v.report(),
    );

    // Controls on layer 1 of the second slice, run changed at one point.
    let (x, resid, state) = control_in[1].clone().unwrap();
    let altered = NATIVE_PREFILL_GDN_KERNEL_SOURCE.replace(
        "ring[((state_pos + s) % 3) * NATIVE_GDN_CONV + c] =",
        "ring[(s % 3) * NATIVE_GDN_CONV + c] =",
    );
    assert_ne!(
        altered, NATIVE_PREFILL_GDN_KERNEL_SOURCE,
        "the ring write was found"
    );
    // SAFETY: the altered write's slot `s % 3` (s >= 0) is still one of the ring's three.
    let ring_at_zero = unsafe { NativeGdnKernels::compile_source(&dev, &altered) }
        .unwrap_or_else(|e| panic!("{e}"));
    // Up's activation scale raised by a quarter (the real checkpoint's gate and up scales are equal), in
    // the scales the runner reads and in the oracle's weights.
    let (gate_s, up_s) = model.gate_up_members;
    let raised = up_s * 1.25;
    let with = |gate_up: f32| LayerScales {
        gate_up,
        ..c.views.scales[1]
    };
    let controls: [(&str, Defect, LayerScales, bool, &[&str]); 5] = [
        (
            "unchanged, up scale raised a quarter",
            Defect::None,
            with(gate_s.max(raised)),
            true,
            &[],
        ),
        (
            "one FP4 code flipped",
            Defect::FlipX4Code,
            c.views.scales[1],
            false,
            &["norm_post.x4"],
        ),
        (
            "gate/up scale the minimum",
            Defect::None,
            with(gate_s.min(raised)),
            true,
            &["norm_post.x4"],
        ),
        (
            "residual stored before the add",
            Defect::ResidBeforeAdd,
            c.views.scales[1],
            false,
            &["norm_in.resid", "layer.resid"],
        ),
        (
            "ring written as if its position were 0",
            Defect::None,
            c.views.scales[1],
            false,
            &["gdn.ring", "layer.ring"],
        ),
    ];
    let stored = model.weights[1].up.input_scale;
    for (i, (what, defect, scales, raise, trips)) in controls.into_iter().enumerate() {
        model.weights[1].up.input_scale = if raise { raised } else { stored };
        let mut s = state.clone();
        let mut dump = PrefillDump::new([1]);
        let alt = if i == 4 {
            Alt::Gdn(&ring_at_zero)
        } else {
            Alt::None
        };
        run_layer(
            &dev,
            &mut c,
            1,
            &model.dev_layers[1],
            &x,
            resid.as_deref(),
            &mut s,
            T,
            scales,
            defect,
            alt,
            &mut dump,
        );
        let (v, _) = verdict(&o, &model.weights[1], &dump, T, None);
        let failed = v.failed();
        let ok = if trips.is_empty() {
            v.ok()
        } else {
            trips.iter().all(|t| failed.contains(t))
        };
        let expect = if trips.is_empty() {
            "passes".to_string()
        } else {
            format!("caught by {trips:?}")
        };
        l.check(
            ok,
            &format!("control {what}: {expect}"),
            &format!("failed {failed:?}\n{}", v.report()),
        );
    }

    // The attention layer's KV rows written one position late (the attention suite's alteration),
    // on layer 3 of the second slice (p0 = 16), and the same run unchanged.
    let (x, resid, state) = control_in[3].clone().unwrap();
    let (from, to) = (
        "const unsigned long long kv_row = (unsigned long long)hk * max_seq + p0 + t;",
        "const unsigned long long kv_row = (unsigned long long)hk * max_seq + p0 + t + 1;",
    );
    assert_eq!(NATIVE_PREFILL_ATTN_KERNEL_SOURCE.matches(from).count(), 1);
    let source = NATIVE_PREFILL_ATTN_KERNEL_SOURCE.replacen(from, to, 1);
    // SAFETY: the late row p0 + T (32) stays inside the cache's MAX_SEQ (256) rows.
    let late = unsafe { NativeAttnKernels::compile_source(&dev, &source) }
        .unwrap_or_else(|e| panic!("{e}"));
    let scales3 = c.views.scales[3];
    for (what, alt, trips) in [
        ("unchanged", Alt::None, &[][..]),
        (
            "KV written for position p0 + 1",
            Alt::Attn(&late),
            &["attention_core.kv"][..],
        ),
    ] {
        let mut s = state.clone();
        let mut dump = PrefillDump::new([3]);
        run_layer(
            &dev,
            &mut c,
            3,
            &model.dev_layers[3],
            &x,
            resid.as_deref(),
            &mut s,
            T,
            scales3,
            Defect::None,
            alt,
            &mut dump,
        );
        let (v, _) = verdict(&o, &model.weights[3], &dump, T, Some(&core));
        let failed = v.failed();
        let ok = if trips.is_empty() {
            v.ok()
        } else {
            trips.iter().all(|t| failed.contains(t))
        };
        l.check(
            ok,
            &format!(
                "control layer 3, {what}: {}",
                if trips.is_empty() {
                    "passes".to_string()
                } else {
                    format!("caught by {trips:?}")
                }
            ),
            &format!("failed {failed:?}\n{}", v.report()),
        );
    }
    l.finish();
}

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8, the real artifact and a prompt"]
fn layer_oracle_runtime() {
    let (dev, mut c, model) = setup(128);
    let hw = p::Hw::new(&dev);
    let tab = p::Tables::new(&hw);
    let o = Oracle {
        hw: &hw,
        tab: &tab,
        sm_count: sm_count(&dev),
    };
    let mut l = Checks::default();
    let core = Attention {
        hw: attn::Hw::new(&dev),
    };
    for t in [7usize, 16, 64, 128] {
        let mut dump = PrefillDump::new(0..LAYERS);
        let (mut x, mut resid) = (model.embedded[..t * H].to_vec(), None::<Vec<u16>>);
        for layer in 0..LAYERS {
            let w = &model.weights[layer];
            let mut s = State::zero(layer);
            let scales = c.views.scales[layer];
            let out = run_layer(
                &dev,
                &mut c,
                layer,
                &model.dev_layers[layer],
                &x,
                resid.as_deref(),
                &mut s,
                t,
                scales,
                Defect::None,
                Alt::None,
                &mut dump,
            );
            let (v, secs) = verdict(&o, w, &dump, t, Some(&core));
            println!("TIME oracle layer {layer} T={t}: {secs:.2} s");
            l.check(v.ok(), &format!("layer {layer}, {t} tokens"), &v.report());
            (x, resid) = (out.mlp_out, Some(out.resid));
        }
    }
    l.finish();
}
