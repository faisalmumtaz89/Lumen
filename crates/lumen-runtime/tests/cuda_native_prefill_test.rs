//! The native prefill's layer oracle (`native_oracle::layer`), qualified on its own: one layer at a
//! time is run through the native components (the producer kernels, the cuBLASLt plans, the GDN and
//! attention kernels, the prefill weight views) as the route chains them, its tensors recorded by the
//! dump hook (`native_prefill_dump`), and the oracle must pass the layer as run and catch it changed
//! at one point. The runner reads its weights and scales as the route does
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
//!   cargo test --release -p lumen-runtime \
//!     --features test-prefill-dump,test-state-snapshot,test-fault-injection \
//!     --test cuda_native_prefill_test -- --ignored --test-threads=1 --nocapture
//!
//! The route as the backend publishes and runs it (a backend built as `lumen-server` builds it: F32
//! KV, the raw global planes, the context capped at 4096), on the real artifact (`LUMEN_NATIVE_MODEL`)
//! and prompts from a JSON file of token-id arrays under the keys P128, P512 and P2048
//! (`LUMEN_NATIVE_IDS_JSON`):
//!
//! - `route_is_announced_and_counted`: the route is published and runs one forward per slice of at most
//!   2048 tokens; with `LUMEN_CUDA_NATIVE_PREFILL=0` or `LUMEN_CUDA_PREFILL_F32` set it is refused as
//!   Q9, naming the switch, and prompts take the F32 route. With `test-fault-injection`, a refusal injected at each condition's check point leaves the
//!   F32 route, naming that condition, and a prompt still prefills.
//! - `native_prefill_is_the_first_operation_after_load` (`test-state-snapshot`): a native prefill right
//!   after load works, and the same prompt after a reset leaves the same state and row, bit for bit,
//!   twice.
//! - `layer_oracle_on_the_assembled_route`: the layer oracle on the route's own dumps: every layer at
//!   16 and 128 tokens with every handoff between layers, sampled layers at 2048 tokens, the second
//!   slice of a 2049-token prompt, a prompt in 64-token calls with the handoffs between calls, and 131
//!   tokens. `LUMEN_NATIVE_ORACLE_SCOPE` (a comma list of t16, t128, t2048, t2049, slices, t131)
//!   selects parts.
//! - `continuation_across_routes` (`test-state-snapshot`): prefill, four teacher-forced decode steps and
//!   a second prefill and two more decode steps, for every order of the native (N) and F32 (F) routes
//!   and a set of lengths; the state the routes leave after each prefill agrees in layout, and their
//!   logits agree; a native prefill after decode passes the layer oracle
//!   and reads the state decode left; an artifact without activation scales
//!   (`LUMEN_NATIVE_OLD_MODEL`) is refused the native route as Q5, and its F32 prefill matches the F32
//!   route on the extended artifact bit for bit.
//! - `reset_and_reuse` (`test-state-snapshot`): prompts that are not prefixes of each other; a prompt
//!   after another prompt and a reset, a shorter one after a longer one, and (with
//!   `test-fault-injection`) one after a failed native forward and a reset, each leave the state and
//!   logits of their first run; the same prompt without a reset differs.
//! - `vram_and_load_time`: load time and device memory; no device memory left allocated after a native
//!   forward; with `test-fault-injection`, an injected Q8 leaves the F32 route.
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
// The layer as the route runs it.

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

// ---------------------------------------------------------------------------------------------
// The route as the backend publishes and runs it.

use lumen_format::quantization::QuantScheme;
use lumen_runtime::compute::ComputeBackend;
#[cfg(feature = "test-state-snapshot")]
use lumen_runtime::compute::{ActivationBuffer, ComputeDtype};
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::{KvCache, KvCacheConfig, KvPrecision};

/// The KV capacity of the route's backend, as `lumen-server --context-len 4096`.
const CONTEXT: usize = 4096;
/// Layers of the admitted model.
const MODEL_LAYERS: usize = 64;

fn model(var: &str) -> SyncWeightProvider {
    SyncWeightProvider::open(std::path::Path::new(&env(var))).unwrap()
}

/// The backend as `lumen-server` builds it for CUDA: F32 KV, the raw global planes, the context
/// capped at [`CONTEXT`], every weight preloaded (which publishes the prefill route).
fn route_backend(provider: &SyncWeightProvider) -> CudaBackend {
    route_backend_at(provider, KvPrecision::F32)
}

/// [`route_backend`] with the KV cache stored at `kv`.
fn route_backend_at(provider: &SyncWeightProvider, kv: KvPrecision) -> CudaBackend {
    let mut hp = provider.lbc().header.hyperparams;
    hp.max_seq_len = hp.max_seq_len.min(CONTEXT as u32);
    let mut cuda = CudaBackend::new(0).expect("CUDA device 0");
    cuda.set_kv_precision(kv).expect("KV precision");
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    if matches!(provider.embedding_quant, QuantScheme::Bf16) && !provider.embedding_raw.is_empty() {
        cuda.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
    }
    if matches!(
        provider.output_proj_quant,
        QuantScheme::Nvfp4 | QuantScheme::Bf16 | QuantScheme::Q8_0
    ) && !provider.output_proj_raw.is_empty()
    {
        cuda.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
    }
    if provider.weight_tying {
        cuda.set_weight_tying(true);
    }
    cuda.init(&hp).expect("init");
    cuda.preload_weights(provider).expect("preload");
    cuda
}

fn route_kv(provider: &SyncWeightProvider) -> KvCache {
    route_kv_at(provider, KvPrecision::F32)
}

fn route_kv_at(provider: &SyncWeightProvider, precision: KvPrecision) -> KvCache {
    let hp = provider.lbc().header.hyperparams;
    KvCache::new(KvCacheConfig {
        max_seq_len: CONTEXT,
        num_layers: hp.num_layers as usize,
        num_kv_heads: hp.num_kv_heads as usize,
        head_dim: hp.head_dim as usize,
        precision,
    })
    .unwrap()
}

/// The token ids of prompt `case` of `LUMEN_NATIVE_IDS_JSON`.
fn case_ids(case: &str) -> Vec<u32> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(env("LUMEN_NATIVE_IDS_JSON")).unwrap())
            .unwrap();
    v[case]
        .as_array()
        .unwrap_or_else(|| panic!("{case} is not in LUMEN_NATIVE_IDS_JSON"))
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

/// P2048's ids followed by P512's: prompts up to 2560 tokens.
fn long_ids() -> Vec<u32> {
    let mut ids = case_ids("P2048");
    ids.extend(case_ids("P512"));
    ids
}

#[cfg(feature = "test-state-snapshot")]
fn logits_of(cuda: &CudaBackend, row: &[f32]) -> Vec<f32> {
    let mut buf = ActivationBuffer::zeros(row.len(), ComputeDtype::F32);
    buf.write_f32_from(row);
    cuda.compute_final(&buf).expect("compute_final").data
}

#[cfg(feature = "test-state-snapshot")]
fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

#[cfg(feature = "test-state-snapshot")]
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "test-state-snapshot")]
/// Relative L2 distance of `got` from `want`; infinite for different lengths or a non-finite sum.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
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

/// Reset the backend and prefill `prompts` one after another into a fresh cache, recording `layers`
/// during each call in `record`. Returns the last call's row, one dump per recorded call in order,
/// and the cache.
fn run_calls(
    cuda: &CudaBackend,
    provider: &SyncWeightProvider,
    prompts: &[&[u32]],
    record: &[usize],
    layers: &[usize],
) -> (Vec<f32>, Vec<PrefillDump>, KvCache) {
    cuda.reset_recurrent_state();
    let mut kv = route_kv(provider);
    let mut row = Vec::new();
    let mut dumps = Vec::new();
    for (i, ids) in prompts.iter().enumerate() {
        let recording = record.contains(&i);
        if recording {
            cuda.set_native_prefill_dump(Some(layers.to_vec()));
        }
        row = cuda.prefill(ids, provider, &mut kv).expect("prefill");
        if recording {
            dumps.push(cuda.take_native_prefill_dump().expect("a dump"));
        }
    }
    (row, dumps, kv)
}

/// The oracle's inputs for the real model's layers.
struct RouteOracle<'a> {
    offsets: Vec<SubtensorOffsets>,
    src: ProviderSlices<'a>,
}

impl<'a> RouteOracle<'a> {
    fn new(provider: &'a SyncWeightProvider) -> Self {
        Self {
            offsets: provider
                .lbc()
                .layer_indices
                .iter()
                .map(|l| l.subtensors.clone())
                .collect(),
            src: ProviderSlices::new(provider),
        }
    }

    fn weights(&self, l: usize) -> LayerWeights {
        LayerWeights::load(&self.src, l, &self.offsets[l])
    }
}

/// Check each of `layers` of `dump` (a call of `t` tokens) with the layer oracle, and every handoff
/// between two consecutive layers of the list.
#[allow(clippy::too_many_arguments)]
fn check_layers(
    l: &mut Checks,
    o: &Oracle,
    core: &Attention,
    ro: &RouteOracle,
    dump: &PrefillDump,
    layers: &[usize],
    t: usize,
    what: &str,
) {
    for (i, &layer) in layers.iter().enumerate() {
        let w = ro.weights(layer);
        let c: Option<&dyn AttentionCore> = if w.attention { Some(core) } else { None };
        let (v, secs) = verdict(o, &w, dump, t, c);
        l.check(
            v.ok(),
            &format!("{what}: layer {layer} ({secs:.1} s)"),
            &v.report(),
        );
        if i > 0 && layers[i - 1] + 1 == layer {
            let v = handoff(&at(dump, layer - 1, t), &at(dump, layer, t));
            l.check(
                v.ok(),
                &format!("{what}: layer {} -> {layer} handoff", layer - 1),
                &v.report(),
            );
        }
    }
}

fn oracle_scope(part: &str) -> bool {
    std::env::var("LUMEN_NATIVE_ORACLE_SCOPE")
        .map(|s| s.split(',').any(|p| p.trim() == part))
        .unwrap_or(true)
}

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn route_is_announced_and_counted() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let ids = long_ids();
    let mut l = Checks::default();
    let cuda = route_backend(&provider);
    // Either switch selects the F32 route, refused as Q9 naming it.
    let switch = if std::env::var("LUMEN_CUDA_NATIVE_PREFILL").as_deref() == Ok("0") {
        Some("LUMEN_CUDA_NATIVE_PREFILL=0")
    } else if std::env::var("LUMEN_CUDA_PREFILL_F32").is_ok() {
        Some("LUMEN_CUDA_PREFILL_F32 is set")
    } else {
        None
    };
    let switched_off = switch.is_some();
    let refusal = cuda.native_prefill_refusal();
    if let Some(reason) = switch {
        l.check(
            refusal.as_ref().map(|r| (r.condition, r.reason.as_str())) == Some(("Q9", reason))
                && cuda.native_prefill_forwards().is_none(),
            &format!("{reason}: the F32 route, refused as Q9"),
            &format!("{refusal:?}"),
        );
    } else {
        l.check(
            refusal.is_none() && cuda.native_prefill_forwards() == Some(0),
            "the native route is published",
            &format!("{refusal:?}"),
        );
    }
    let before = cuda.native_prefill_forwards();
    let (row, _, _) = run_calls(&cuda, &provider, &[&ids[..128]], &[], &[]);
    let after_128 = cuda.native_prefill_forwards();
    let (row2049, _, _) = run_calls(&cuda, &provider, &[&ids[..2049]], &[], &[]);
    let after_2049 = cuda.native_prefill_forwards();
    let finite = row.iter().chain(&row2049).all(|v| v.is_finite());
    let counted = if switched_off {
        [before, after_128, after_2049] == [None, None, None]
    } else {
        [before, after_128, after_2049] == [Some(0), Some(1), Some(3)]
    };
    l.check(
        counted && finite,
        "forwards: one per slice of at most 2048 tokens (128 tokens, then 2049), rows finite",
        &format!("{before:?} {after_128:?} {after_2049:?}, finite {finite}"),
    );
    drop(cuda);

    #[cfg(feature = "test-fault-injection")]
    if !switched_off {
        use lumen_runtime::cuda::native_prefill::fault;
        for condition in ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"] {
            fault::refuse(Some(condition));
            let cuda = route_backend(&provider);
            fault::refuse(None);
            let refusal = cuda.native_prefill_refusal();
            let (row, _, _) = run_calls(&cuda, &provider, &[&ids[..16]], &[], &[]);
            l.check(
                refusal.as_ref().map(|r| (r.condition, r.reason.as_str()))
                    == Some((condition, "injected"))
                    && cuda.native_prefill_forwards().is_none()
                    && row.iter().all(|v| v.is_finite()),
                &format!("{condition} injected: the F32 route, naming {condition}, prefills"),
                &format!("{refusal:?}"),
            );
        }
    }
    l.finish();
}

/// FNV-1a over the bits of `v`: equal hashes stand for bit-equal values.
#[cfg(feature = "test-state-snapshot")]
fn fnv(v: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in v.iter().flat_map(|x| x.to_bits().to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Every value of two snapshots bit for bit.
#[cfg(feature = "test-state-snapshot")]
fn same_state(
    a: &lumen_runtime::cuda::StateSnapshot,
    b: &lumen_runtime::cuda::StateSnapshot,
) -> bool {
    a.seq_len == b.seq_len
        && a.conv_positions == b.conv_positions
        && a.decode_token_count == b.decode_token_count
        && a.kv.len() == b.kv.len()
        && a.kv
            .iter()
            .zip(&b.kv)
            .all(|(x, y)| same_bits(&x.0, &y.0) && same_bits(&x.1, &y.1))
        && a.h_states.len() == b.h_states.len()
        && a.h_states
            .iter()
            .zip(&b.h_states)
            .all(|(x, y)| same_bits(x, y))
        && a.conv_states.len() == b.conv_states.len()
        && a.conv_states
            .iter()
            .zip(&b.conv_states)
            .all(|(x, y)| same_bits(x, y))
        && same_bits(&a.x, &b.x)
}

#[cfg(feature = "test-state-snapshot")]
#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn native_prefill_is_the_first_operation_after_load() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let ids = case_ids("P128");
    let mut l = Checks::default();
    let cuda = route_backend(&provider);
    // No reset, decode or other prefill before this one; its layers 0, 3 and 63 are recorded.
    let dumped = [0usize, 3, 63];
    cuda.set_native_prefill_dump(Some(dumped.to_vec()));
    let mut kv = route_kv(&provider);
    let row = cuda
        .prefill(&ids, &provider, &mut kv)
        .expect("first prefill");
    let dump = cuda.take_native_prefill_dump().expect("a dump");
    let first = cuda.snapshot_state(&kv).unwrap();
    let logits = logits_of(&cuda, &row);
    let state_bits: Vec<f32> = first
        .kv
        .iter()
        .flat_map(|(k, v)| k.iter().chain(v))
        .chain(first.h_states.iter().flatten())
        .chain(first.conv_states.iter().flatten())
        .chain(&first.x)
        .copied()
        .chain(first.conv_positions.iter().map(|&p| p as f32))
        .collect();
    println!(
        "HASH row={} logits={} state={}",
        fnv(&row),
        fnv(&logits),
        fnv(&state_bits)
    );
    let dev = CudaDevice::new(0).unwrap();
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
    let ro = RouteOracle::new(&provider);
    check_layers(
        &mut l,
        &o,
        &core,
        &ro,
        &dump,
        &dumped,
        ids.len(),
        "the first prefill after load",
    );
    // Controls on the same dump: layer 3's KV cache and layer 0's GDN state each changed.
    let scale = |b: &mut Vec<u8>| {
        for c in b.chunks_exact_mut(4) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            c.copy_from_slice(&(v * 1.25 + 0.01).to_le_bytes());
        }
    };
    let t = ids.len();
    for (layer, name, check) in [(3, "kv", "attention_core.kv"), (0, "state", "gdn.")] {
        let bad = changed(&dump, layer, name, &scale);
        let w = ro.weights(layer);
        let c: Option<&dyn AttentionCore> = if w.attention { Some(&core) } else { None };
        let (v, _) = verdict(&o, &w, &bad, t, c);
        let failed = v.failed();
        l.check(
            failed.iter().any(|f| f.starts_with(check)),
            &format!("control: layer {layer}'s {name} changed, rejected by {check}*"),
            &format!("failed {failed:?}"),
        );
    }
    l.check(
        cuda.native_prefill_forwards() == Some(1) && row.iter().all(|v| v.is_finite()),
        "the first operation after load is a native prefill",
        &format!(
            "forwards {:?}, first token {}",
            cuda.native_prefill_forwards(),
            argmax(&logits)
        ),
    );
    for again in 1..=2 {
        let (r, _, kv) = run_calls(&cuda, &provider, &[&ids], &[], &[]);
        let snap = cuda.snapshot_state(&kv).unwrap();
        let lg = logits_of(&cuda, &r);
        l.check(
            same_bits(&r, &row) && same_state(&snap, &first) && same_bits(&lg, &logits),
            &format!("after a reset, run {again}: row, state and logits bit-identical"),
            &format!("row rel L2 {:.3e}", rel_l2(&r, &row)),
        );
    }
    l.finish();
}

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn layer_oracle_on_the_assembled_route() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let cuda = route_backend(&provider);
    let dev = CudaDevice::new(0).unwrap();
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
    let ro = RouteOracle::new(&provider);
    let ids = long_ids();
    let mut l = Checks::default();
    assert_eq!(cuda.native_prefill_refusal(), None, "the native route");

    // Every layer, in batches that overlap by one layer so every handoff is checked.
    for t in [16usize, 128] {
        if !oracle_scope(&format!("t{t}")) {
            continue;
        }
        for start in (0..MODEL_LAYERS).step_by(16) {
            let layers: Vec<usize> = (start.saturating_sub(1)..(start + 16)).collect();
            let (_, mut dump, _) = run_calls(&cuda, &provider, &[&ids[..t]], &[0], &layers);
            let dump = dump.remove(0);
            check_layers(
                &mut l,
                &o,
                &core,
                &ro,
                &dump,
                &layers,
                t,
                &format!("{t} tokens"),
            );
        }
    }
    if oracle_scope("t2048") {
        let layers = vec![0, 1, 2, 3, 31, 62, 63];
        let (_, mut dump, _) = run_calls(&cuda, &provider, &[&ids[..2048]], &[0], &layers);
        check_layers(
            &mut l,
            &o,
            &core,
            &ro,
            &dump.remove(0),
            &layers,
            2048,
            "2048 tokens",
        );
    }
    if oracle_scope("t2049") {
        // The recorder keeps the later slice: 1 token at position 2048.
        let layers = vec![0, 1, 2, 3, 62, 63];
        let (_, mut dump, _) = run_calls(&cuda, &provider, &[&ids[..2049]], &[0], &layers);
        let dump = dump.remove(0);
        let p0 = at(&dump, 3, 1).state_pos("p0");
        l.check(
            p0 == Some(2048),
            "2049 tokens: the recorded slice is the second, at position 2048",
            &format!("p0 {p0:?}"),
        );
        check_layers(
            &mut l,
            &o,
            &core,
            &ro,
            &dump,
            &layers,
            1,
            "2049 tokens, second slice",
        );
    }
    if oracle_scope("slices") {
        // 256 tokens in four calls of 64: the third call checked, and its inputs are the second's
        // outputs.
        let layers = vec![0, 1, 2, 3, 4, 63];
        let calls: Vec<&[u32]> = ids[..256].chunks(64).collect();
        // Both calls' dumps from one run: a reset keeps each cache's rows past its length, so another
        // run's rows there would differ.
        let (_, mut dumps, _) = run_calls(&cuda, &provider, &calls, &[1, 2], &layers);
        let (third, second) = (dumps.pop().unwrap(), dumps.pop().unwrap());
        check_layers(
            &mut l,
            &o,
            &core,
            &ro,
            &third,
            &layers,
            64,
            "64-token calls, the third",
        );
        for &layer in &layers {
            let v = handoff(&at(&second, layer, 64), &at(&third, layer, 64));
            l.check(
                v.ok(),
                &format!("64-token calls: layer {layer}, second -> third call handoff"),
                &v.report(),
            );
        }
    }
    if oracle_scope("t131") {
        let layers = vec![0, 1, 2, 3, 4, 63];
        let (_, mut dump, _) = run_calls(&cuda, &provider, &[&ids[..131]], &[0], &layers);
        check_layers(
            &mut l,
            &o,
            &core,
            &ro,
            &dump.remove(0),
            &layers,
            131,
            "131 tokens",
        );
    }
    l.finish();
}

/// One order of routes: a prefill of `l1` tokens, `STEPS` teacher-forced decode steps, a prefill of
/// `l2` tokens; the second prefill's layers `dump` recorded when it is native.
#[cfg(feature = "test-state-snapshot")]
struct Continuation {
    /// State after the first prefill, after the decode steps, and after the second prefill.
    after_first: lumen_runtime::cuda::StateSnapshot,
    after_decode: lumen_runtime::cuda::StateSnapshot,
    after_second: lumen_runtime::cuda::StateSnapshot,
    /// Logits of each decode step, of the second prefill's last position, then of [`AFTER`]
    /// teacher-forced decode steps after it.
    logits: Vec<Vec<f32>>,
    dump: Option<PrefillDump>,
}

#[cfg(feature = "test-state-snapshot")]
const STEPS: usize = 4;
/// Teacher-forced decode steps after the second prefill.
#[cfg(feature = "test-state-snapshot")]
const AFTER: usize = 2;

#[cfg(feature = "test-state-snapshot")]
fn continuation(
    cuda: &CudaBackend,
    provider: &SyncWeightProvider,
    ids: &[u32],
    (l1, l2): (usize, usize),
    native: (bool, bool),
    dump: &[usize],
) -> Continuation {
    cuda.reset_recurrent_state();
    let mut kv = route_kv(provider);
    cuda.set_native_prefill_suspended(!native.0);
    cuda.prefill(&ids[..l1], provider, &mut kv)
        .expect("first prefill");
    let after_first = cuda.snapshot_state(&kv).unwrap();
    let mut logits = Vec::new();
    for &id in &ids[l1..l1 + STEPS] {
        logits.push(
            cuda.decode_token(id, provider, &mut kv)
                .expect("decode")
                .data,
        );
    }
    let after_decode = cuda.snapshot_state(&kv).unwrap();
    cuda.set_native_prefill_suspended(!native.1);
    if native.1 {
        cuda.set_native_prefill_dump(Some(dump.to_vec()));
    }
    let from = l1 + STEPS;
    let row = cuda
        .prefill(&ids[from..from + l2], provider, &mut kv)
        .expect("second prefill");
    let dump = if native.1 {
        cuda.take_native_prefill_dump()
    } else {
        None
    };
    cuda.set_native_prefill_suspended(false);
    logits.push(logits_of(cuda, &row));
    let after_second = cuda.snapshot_state(&kv).unwrap();
    for &id in &ids[from + l2..from + l2 + AFTER] {
        logits.push(
            cuda.decode_token(id, provider, &mut kv)
                .expect("decode after the second prefill")
                .data,
        );
    }
    Continuation {
        after_first,
        after_decode,
        after_second,
        logits,
        dump,
    }
}

/// The largest relative L2 distance between two routes' state after the same prompt, per kind of
/// tensor, with the exact parts (lengths and ring positions) compared exactly.
#[cfg(feature = "test-state-snapshot")]
fn state_distance(
    a: &lumen_runtime::cuda::StateSnapshot,
    b: &lumen_runtime::cuda::StateSnapshot,
) -> (bool, [f64; 4]) {
    let worst = |pairs: Vec<f64>| pairs.into_iter().fold(0.0f64, f64::max);
    let exact = a.seq_len == b.seq_len
        && a.conv_positions == b.conv_positions
        && a.kv.len() == b.kv.len()
        && a.h_states.len() == b.h_states.len();
    (
        exact,
        [
            worst(
                a.kv.iter()
                    .zip(&b.kv)
                    .flat_map(|(x, y)| [rel_l2(&x.0, &y.0), rel_l2(&x.1, &y.1)])
                    .collect(),
            ),
            worst(
                a.h_states
                    .iter()
                    .zip(&b.h_states)
                    .map(|(x, y)| rel_l2(x, y))
                    .collect(),
            ),
            worst(
                a.conv_states
                    .iter()
                    .zip(&b.conv_states)
                    .map(|(x, y)| rel_l2(x, y))
                    .collect(),
            ),
            rel_l2(&a.x, &b.x),
        ],
    )
}

/// `snap` with one GDN state transposed, and with one KV cache's rows moved a position.
#[cfg(feature = "test-state-snapshot")]
fn state_controls(
    snap: &lumen_runtime::cuda::StateSnapshot,
) -> [lumen_runtime::cuda::StateSnapshot; 2] {
    let mut transposed = snap.clone();
    let h = &mut transposed.h_states[0];
    let orig = h.clone();
    for head in 0..48 {
        for i in 0..128 {
            for j in 0..128 {
                h[(head * 128 + i) * 128 + j] = orig[(head * 128 + j) * 128 + i];
            }
        }
    }
    let mut moved = snap.clone();
    moved.kv[0].0.rotate_left(256);
    [transposed, moved]
}

/// The limit on the relative L2 distance between the two routes' state and logits.
#[cfg(feature = "test-state-snapshot")]
const ROUTE_DISTANCE: f64 = 0.25;

#[cfg(feature = "test-state-snapshot")]
#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and both real artifacts"]
fn continuation_across_routes() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let cuda = route_backend(&provider);
    assert_eq!(cuda.native_prefill_refusal(), None, "the native route");
    let dev = CudaDevice::new(0).unwrap();
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
    let ro = RouteOracle::new(&provider);
    let ids = long_ids();
    let mut l = Checks::default();
    let lengths = [
        (63, 64),
        (64, 65),
        (65, 63),
        (127, 128),
        (128, 131),
        (131, 127),
        (2047, 2),
        (2048, 1),
        (2049, 64),
        (63, 2049),
    ];
    let dumped = [0usize, 3, 63];
    let mut f32_runs = Vec::new();
    for &(l1, l2) in &lengths {
        let what = format!("{l1} + {STEPS} decode + {l2}");
        let run = |native| continuation(&cuda, &provider, &ids, (l1, l2), native, &dumped);
        let (ndn, fdn, ndf, fdf) = (
            run((true, true)),
            run((false, true)),
            run((true, false)),
            run((false, false)),
        );
        // The routes leave the same state, up to their arithmetic, after the first prefill and after
        // the second; the controls (a transposed GDN state, KV moved a position) exceed the limit.
        let after_first = [("N-d-N", &ndn)];
        let after_second = [("N-d-N", &ndn), ("F-d-N", &fdn), ("N-d-F", &ndf)];
        for (when, runs) in [("first", &after_first[..]), ("second", &after_second[..])] {
            let snap = |c: &Continuation| {
                if when == "first" {
                    c.after_first.clone()
                } else {
                    c.after_second.clone()
                }
            };
            let base = snap(&fdf);
            for (name, run) in runs {
                let got = snap(run);
                let (exact, d) = state_distance(&got, &base);
                l.check(
                    exact && d.iter().all(|&x| x <= ROUTE_DISTANCE),
                    &format!("{what}: state after the {when} prefill, {name} vs F-d-F, within {ROUTE_DISTANCE}"),
                    &format!("lengths and ring positions equal {exact}; KV {:.3e}, GDN state {:.3e}, ring {:.3e}, row {:.3e}", d[0], d[1], d[2], d[3]),
                );
                let [transposed, moved] = state_controls(&base);
                let (_, dt) = state_distance(&got, &transposed);
                let (_, dm) = state_distance(&got, &moved);
                l.check(
                    dt[1] > ROUTE_DISTANCE && dm[0] > ROUTE_DISTANCE,
                    &format!("{what}: after the {when} prefill, {name}: controls exceed the limit (a transposed GDN state, KV moved a position)"),
                    &format!("GDN state {:.3e}, KV {:.3e}", dt[1], dm[0]),
                );
            }
        }
        // Decode continues either route's state alike; the second prefill reads decode's state.
        for (name, run, base) in [
            ("N-d-N", &ndn, &fdf),
            ("F-d-N", &fdn, &fdf),
            ("N-d-F", &ndf, &fdf),
        ] {
            let dists: Vec<f64> = run
                .logits
                .iter()
                .zip(&base.logits)
                .map(|(a, b)| rel_l2(a, b))
                .collect();
            let agree = run
                .logits
                .iter()
                .zip(&base.logits)
                .filter(|(a, b)| argmax(a) == argmax(b))
                .count();
            l.check(
                dists.iter().all(|&x| x <= ROUTE_DISTANCE),
                &format!("{what}: {name} logits ({STEPS} decode steps, the second prefill, {AFTER} decode steps) vs F-d-F within {ROUTE_DISTANCE}"),
                &format!(
                    "rel L2 {}; argmax agrees at {agree} of {}",
                    dists.iter().map(|d| format!("{d:.3e}")).collect::<Vec<_>>().join(" "),
                    dists.len()
                ),
            );
            // Control: the logits of another position.
            let n = run.logits.len();
            let other = rel_l2(&run.logits[n - 2], &base.logits[n - 1]);
            l.check(
                other > ROUTE_DISTANCE,
                &format!("{what}: {name}: control, the logits one position apart exceed the limit"),
                &format!("rel L2 {other:.3e}"),
            );
        }
        for (name, run) in [("N-d-N", &ndn), ("F-d-N", &fdn)] {
            let dump = run.dump.as_ref().expect("the native second prefill's dump");
            let snap = &run.after_decode;
            // A second prefill past 2048 tokens is recorded in its last slice, which starts 2048
            // positions later from the state its first slice left; its KV rows before that slice are
            // still the ones decode left.
            let whole = l2 <= 2048;
            let td = if whole { l2 } else { l2 - 2048 };
            let first_pos = snap.seq_len + l2 - td;
            // The second prefill read the state decode left.
            let g_state = |layer: usize| {
                let f = |n: &str| {
                    at(dump, layer, td)
                        .f32(n)
                        .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                };
                // The GDN layers before it: the layer less the attention layers before it.
                let g = layer - (layer + 1) / 4;
                f("state_in") == Some(snap.h_states[g].iter().map(|x| x.to_bits()).collect())
                    && f("ring_in")
                        == Some(snap.conv_states[g].iter().map(|x| x.to_bits()).collect())
                    && at(dump, layer, td).state_pos("state_pos_in")
                        == Some(snap.conv_positions[g] as usize)
            };
            let kv_in = at(dump, 3, td).f32("kv_in").unwrap_or_default();
            let max_seq = kv_in.len() / (2 * 4 * 256);
            let seq = snap.seq_len;
            let live = |half: usize| -> Vec<u32> {
                (0..4)
                    .flat_map(|hk| {
                        let at = (half * 4 + hk) * max_seq * 256;
                        kv_in[at..at + seq * 256].iter().map(|x| x.to_bits())
                    })
                    .collect()
            };
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let kv_ok = max_seq >= seq
                && live(0) == bits(&snap.kv[0].0)
                && live(1) == bits(&snap.kv[0].1)
                && at(dump, 3, td).state_pos("p0") == Some(first_pos);
            let gdn_ok = !whole || g_state(0);
            l.check(
                gdn_ok && kv_ok,
                &format!("{what}: {name}'s second prefill reads the state decode left (layer 0 state, ring and position; layer 3 KV and position)"),
                &format!("GDN {gdn_ok} (checked: {whole}) KV {kv_ok}"),
            );
            check_layers(
                &mut l,
                &o,
                &core,
                &ro,
                dump,
                &dumped,
                td,
                &format!("{what}: {name}, second prefill ({td} tokens from {first_pos})"),
            );
        }
        f32_runs.push(fdf.logits);
    }
    drop(cuda);

    // An artifact without activation scales: the F32 route, the same weights, the same logits.
    let old = model("LUMEN_NATIVE_OLD_MODEL");
    let cuda = route_backend(&old);
    let refusal = cuda.native_prefill_refusal();
    l.check(
        refusal.as_ref().map(|r| r.condition) == Some("Q5"),
        "the artifact without activation scales: the F32 route, refused as Q5",
        &format!("{refusal:?}"),
    );
    for (&(l1, l2), want) in lengths.iter().zip(&f32_runs) {
        let run = continuation(&cuda, &old, &ids, (l1, l2), (false, false), &[]);
        let same = run.logits.len() == want.len()
            && run.logits.iter().zip(want).all(|(a, b)| same_bits(a, b));
        l.check(
            same,
            &format!("{l1} + {STEPS} decode + {l2}: F-d-F on the artifact without scales equals it on the extended one, bit for bit"),
            "",
        );
    }
    l.finish();
}

#[cfg(feature = "test-state-snapshot")]
#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn reset_and_reuse() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let cuda = route_backend(&provider);
    assert_eq!(cuda.native_prefill_refusal(), None, "the native route");
    let mut l = Checks::default();
    let state_of = |ids: &[u32]| {
        let (row, _, kv) = run_calls(&cuda, &provider, &[ids], &[], &[]);
        (cuda.snapshot_state(&kv).unwrap(), logits_of(&cuda, &row))
    };
    // No prompt is a prefix of another, so the KV rows and state a reset leaves behind are not the
    // ones the next prompt writes.
    let long = case_ids("P2048");
    let a = &long[1500..1628];
    let b = &long[300..812];
    // 131 tokens run at 144 rows: rows 131 to 143 of every GEMM input are padding, which a longer
    // prompt before them leaves written.
    let short = &long[1000..1131];
    let (short_first, short_logits) = state_of(short);
    let (first, first_logits) = state_of(b);
    run_calls(&cuda, &provider, &[a], &[], &[]);
    let (again, again_logits) = state_of(b);
    l.check(
        same_state(&again, &first) && same_bits(&again_logits, &first_logits),
        "512 tokens after another prompt and a reset: state and logits bit-identical to their first run",
        &format!("first token {}", argmax(&first_logits)),
    );
    run_calls(&cuda, &provider, &[&long], &[], &[]);
    let (short_again, short_again_logits) = state_of(short);
    l.check(
        same_state(&short_again, &short_first) && same_bits(&short_again_logits, &short_logits),
        "131 tokens after 2048 and a reset: state and logits bit-identical to the first prompt after load",
        &format!("first token {}", argmax(&short_logits)),
    );
    // Control: the same 131 tokens again without a reset run at positions 131 to 261, after the
    // first copy, and must differ.
    let (carried_row, _, _) = run_calls(&cuda, &provider, &[short, short], &[], &[]);
    let carried = logits_of(&cuda, &carried_row);
    l.check(
        !same_bits(&carried, &short_logits),
        "control: the 131 tokens again without a reset (continuing at position 131) differ",
        &format!("logits rel L2 {:.3e}", rel_l2(&carried, &short_logits)),
    );
    #[cfg(feature = "test-fault-injection")]
    {
        use lumen_runtime::cuda::native_prefill::fault;
        cuda.reset_recurrent_state();
        let mut kv = route_kv(&provider);
        fault::fail_forward_at_layer(30);
        let failed = cuda.prefill(a, &provider, &mut kv);
        let (after, after_logits) = state_of(b);
        l.check(
            failed
                .as_ref()
                .is_err_and(|e| e.to_string().contains("layer 30: injected"))
                && same_state(&after, &first)
                && same_bits(&after_logits, &first_logits),
            "512 tokens after a native forward failed at layer 30 and a reset: bit-identical to their first run",
            &format!("{:?}", failed.err()),
        );
    }
    l.finish();
}

#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn vram_and_load_time() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let dev = CudaDevice::new(0).unwrap();
    let mut l = Checks::default();
    let free0 = dev.free_memory().unwrap();
    let start = Instant::now();
    let cuda = route_backend(&provider);
    let load = start.elapsed().as_secs_f64();
    let free1 = dev.free_memory().unwrap();
    let ids = case_ids("P2048");
    run_calls(&cuda, &provider, &[&ids[..16]], &[], &[]);
    let free2 = dev.free_memory().unwrap();
    let (row, _, _) = run_calls(&cuda, &provider, &[&ids], &[], &[]);
    let free3 = dev.free_memory().unwrap();
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    println!(
        "LOAD {load:.1} s; device memory: {:.2} GiB free before, {:.2} GiB after load ({:.2} GiB taken), {:.2} GiB after a 16-token and {:.2} GiB after a 2048-token native prefill",
        gib(free0),
        gib(free1),
        gib(free0.saturating_sub(free1)),
        gib(free2),
        gib(free3)
    );
    l.check(
        cuda.native_prefill_refusal().is_none()
            && free3 == free2
            && row.iter().all(|v| v.is_finite()),
        "no device memory is left allocated after a forward (free memory after the 16- and 2048-token prefills equal; a peak inside the forward is not visible here)",
        &format!("{free2} vs {free3} bytes free"),
    );
    drop(cuda);
    #[cfg(feature = "test-fault-injection")]
    {
        use lumen_runtime::cuda::native_prefill::fault;
        fault::refuse(Some("Q8"));
        let cuda = route_backend(&provider);
        fault::refuse(None);
        let refusal = cuda.native_prefill_refusal();
        let (row, _, _) = run_calls(&cuda, &provider, &[&ids[..16]], &[], &[]);
        l.check(
            refusal.as_ref().map(|r| r.condition) == Some("Q8")
                && row.iter().all(|v| v.is_finite()),
            "an allocation refused (Q8): the F32 route, named",
            &format!("{refusal:?}"),
        );
    }
    l.finish();
}

/// A BF16 KV store changes nothing the native prefill computes: the route attends to BF16 keys and
/// values either way (an F32 store's rows restaged, or the BF16 store's own), so every prompt's last
/// row is bit for bit the F32 store's. Prompts of 128, 512 and 2048 tokens, 2560 tokens in two
/// slices, and 2560 tokens in two calls of 64 and 2496 (older rows read back from the store). The
/// route is published at both stores and runs one forward per slice.
#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn bf16_store_prefills_bit_for_bit_like_the_f32_store() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let ids = long_ids();
    // Two slices whole, and a second call that crosses a slice boundary.
    assert_eq!(ids.len(), 2560, "P2048 then P512");
    let prompts: Vec<Vec<&[u32]>> = vec![
        vec![&ids[..128]],
        vec![&ids[..512]],
        vec![&ids[..2048]],
        vec![&ids[..]],
        vec![&ids[..64], &ids[64..]],
    ];
    let rows = |kv: KvPrecision| -> Vec<Vec<f32>> {
        let cuda = route_backend_at(&provider, kv);
        assert!(
            cuda.native_prefill_refusal().is_none() && cuda.native_prefill_forwards() == Some(0),
            "{kv:?}: the native route is published: {:?}",
            cuda.native_prefill_refusal()
        );
        let rows = prompts
            .iter()
            .map(|calls| {
                cuda.reset_recurrent_state();
                let mut cache = route_kv_at(&provider, kv);
                let mut row = Vec::new();
                for ids in calls {
                    row = cuda.prefill(ids, &provider, &mut cache).expect("prefill");
                }
                row
            })
            .collect();
        let slices: u64 = prompts
            .iter()
            .flatten()
            .map(|ids| ids.len().div_ceil(2048) as u64)
            .sum();
        assert_eq!(
            cuda.native_prefill_forwards(),
            Some(slices),
            "{kv:?}: forwards"
        );
        rows
    };
    let f32_rows = rows(KvPrecision::F32);
    let bf16_rows = rows(KvPrecision::Bf16);
    for (i, (a, b)) in f32_rows.iter().zip(&bf16_rows).enumerate() {
        assert!(
            !a.is_empty() && a.iter().all(|x| x.is_finite()),
            "prompt {i}: row"
        );
        assert!(
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "prompt {i}: the BF16 store's row differs from the F32 store's"
        );
    }
}

/// A 16-bit KV store the native route refuses prefills on the F32 route, which reads the store
/// through the widening pair released while the native route was tried and made again when it was
/// refused: an F16 store (refused at Q7), and with `test-fault-injection` a BF16 store refused at
/// Q4, after the route's own buffers allocated. With `test-state-snapshot` too, each store's logits
/// on the F32 route are the F32 store's to within its rounding: the same argmax and a small
/// relative L2 distance, which a key/value mix-up in the 16-bit writers or the widening would not
/// keep.
#[test]
#[ignore = "needs a GPU, cuBLASLt >= 12.8 and the real artifact"]
fn a_refused_16_bit_store_prefills_on_the_f32_route() {
    let provider = model("LUMEN_NATIVE_MODEL");
    let ids = long_ids();
    let cuda = route_backend_at(&provider, KvPrecision::F16);
    let refusal = cuda.native_prefill_refusal();
    assert_eq!(
        refusal.as_ref().map(|r| r.condition),
        Some("Q7"),
        "{refusal:?}"
    );
    cuda.reset_recurrent_state();
    let mut cache = route_kv_at(&provider, KvPrecision::F16);
    let row = cuda
        .prefill(&ids[..512], &provider, &mut cache)
        .expect("an F16 store prefills on the F32 route");
    assert!(!row.is_empty() && row.iter().all(|x| x.is_finite()));
    assert_eq!(cuda.native_prefill_forwards(), None);
    drop(cuda);

    #[cfg(all(feature = "test-fault-injection", feature = "test-state-snapshot"))]
    {
        use lumen_runtime::cuda::native_prefill::fault;
        let f32_route_logits = |kv: KvPrecision, inject: Option<&'static str>, refused: &str| {
            fault::refuse(inject);
            let cuda = route_backend_at(&provider, kv);
            fault::refuse(None);
            let refusal = cuda.native_prefill_refusal();
            assert_eq!(
                refusal.as_ref().map(|r| r.condition),
                Some(refused),
                "{kv:?}: {refusal:?}"
            );
            cuda.reset_recurrent_state();
            let mut cache = route_kv_at(&provider, kv);
            let row = cuda
                .prefill(&ids[..512], &provider, &mut cache)
                .unwrap_or_else(|e| panic!("{kv:?} on the F32 route: {e}"));
            assert!(!row.is_empty() && row.iter().all(|x| x.is_finite()));
            assert_eq!(cuda.native_prefill_forwards(), None);
            logits_of(&cuda, &row)
        };
        let want = f32_route_logits(KvPrecision::F32, Some("Q4"), "Q4");
        for (kv, inject, refused) in [
            (KvPrecision::F16, None, "Q7"),
            (KvPrecision::Bf16, Some("Q4"), "Q4"),
        ] {
            let got = f32_route_logits(kv, inject, refused);
            let d = rel_l2(&got, &want);
            eprintln!("{kv:?} store, F32 route: logits relative L2 {d:.3e} from the F32 store's");
            assert_eq!(argmax(&got), argmax(&want), "{kv:?}: argmax");
            assert!(d < 0.05, "{kv:?}: relative L2 {d:.3e}");
        }
    }
}
