//! The native NVFP4/FP8 prefill route as the backend serves it: its publication at load, and its
//! forward.
//!
//! Publication checks the route's conditions in a fixed order (`native_prefill`'s module doc names
//! them) and either returns the route, with every buffer it needs allocated and every GEMM plan
//! built, or the first condition that fails. The shared GDN state (`ensure_gdn_scratch`) exists before
//! the route does, so a native prefill can be the first operation after load.
//!
//! The forward runs a prompt in slices of at most [`MAX_ROWS`] tokens. Per slice: the embedded rows,
//! then every layer as its components define it (`native_prefill_kernels`, `native_prefill_gdn`,
//! `native_prefill_attn`, the cuBLASLt plans of `native_prefill_gemm`), with BF16 activations and a
//! BF16 residual. It writes the state decode reads, where and as decode keeps it: each attention
//! layer's KV rows (F32 or BF16, the store's type) and length, each GDN layer's recurrent state, conv ring and ring position, and
//! the last position's hidden row (`scratch.x_gpu`, F32).

use super::{
    CudaBackend, GdnScratchGpu, GpuWeightBuf, KvCacheGpu, KvStore, LayerWeightsGpu, MutableState,
};
use crate::cuda::cublaslt::{library_report, LtOperands};
use crate::cuda::cublaslt_algo_cache::WeightOperand;
use crate::cuda::ffi::CudaDevice;
use crate::cuda::native_prefill::{
    self as route, admit_each, injected, is_attention_layer, ProviderSlices, Refusal, HIDDEN,
    INTERMEDIATE, ROPE, ROTARY_DIM,
};
use crate::cuda::native_prefill_attn::NativeAttnKernels;
#[cfg(feature = "test-prefill-dump")]
use crate::cuda::native_prefill_dump::PrefillDump;
use crate::cuda::native_prefill_gdn::{next_conv_position, NativeGdnKernels, CHUNK, CONV_DIM};
use crate::cuda::native_prefill_gemm::{
    bucket, bucket_rows, NativeGemm, TableSource, ATTN_KV, ATTN_Q, DOWN, GATE_UP, GDN_QKV, GDN_Z,
    MAX_ROWS, OUT,
};
use crate::cuda::native_prefill_kernels::{
    fp4_scale_bytes, gdn_norm_lanes_per_row, NativePrefillKernels, ATTN_OUT,
    GDN_NORM_ROWS_PER_TOKEN,
};
use crate::cuda::native_prefill_weights::PrefillWeightViews;
use crate::error::RuntimeError;
use crate::kv::KvPrecision;
use crate::weight::cache::WeightProvider;
use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};

const H: usize = HIDDEN as usize;
const I: usize = INTERMEDIATE as usize;
/// GDN value width: 48 heads of 128, the gated norm's and the out projection's width.
const GDN_V: usize = 6144;
/// GDN `a` and `b` columns: 48 each.
const GDN_AB: usize = 96;
const GDN_HEADS: usize = GDN_NORM_ROWS_PER_TOKEN as usize;
/// Query and gate rows of an attention layer's fused projection.
const ATTN_QG: usize = 12288;
/// Rows of the K and V projections: 4 KV heads of 256.
const ATTN_KVD: usize = 1024;
const AO: usize = ATTN_OUT as usize;
const CONV: usize = CONV_DIM as usize;
const ROT: usize = ROTARY_DIM as usize;

fn dp<T>(device: &CudaDevice, s: &CudaSlice<T>) -> u64 {
    s.device_ptr(&device.stream).0
}

/// Buffers the forward reuses across prompts, each sized for [`MAX_ROWS`] rows.
struct NativeScratch {
    ids: CudaSlice<u32>,
    /// The residual, BF16 `[rows][5120]`.
    resid: CudaSlice<u16>,
    /// Each layer's input and its MLP output, BF16 `[rows][5120]`.
    hidden: CudaSlice<u16>,
    normed: CudaSlice<u16>,
    x8: CudaSlice<u8>,
    qkv: CudaSlice<u16>,
    z: CudaSlice<u16>,
    ab: CudaSlice<f32>,
    cv: CudaSlice<u16>,
    gc: CudaSlice<f32>,
    w: CudaSlice<u16>,
    u: CudaSlice<u16>,
    aqk: CudaSlice<u16>,
    core: CudaSlice<u16>,
    /// FP8 codes of the out projection's input: the GDN gated norm or the gated attention output.
    codes_out: CudaSlice<u8>,
    qg: CudaSlice<u16>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    q: CudaSlice<u16>,
    gate: CudaSlice<u16>,
    o: CudaSlice<u16>,
    attn_out: CudaSlice<u16>,
    x4: CudaSlice<u8>,
    x4sf: CudaSlice<u8>,
    gu: CudaSlice<u16>,
    d4: CudaSlice<u8>,
    d4sf: CudaSlice<u8>,
}

/// Allocates zeroed device buffers and counts their bytes.
struct Alloc<'a> {
    device: &'a CudaDevice,
    bytes: usize,
}

impl Alloc<'_> {
    fn zeros<T: DeviceRepr + ValidAsZeroBits>(
        &mut self,
        n: usize,
    ) -> Result<CudaSlice<T>, RuntimeError> {
        let s = self.device.alloc_zeros::<T>(n)?;
        self.bytes += n * std::mem::size_of::<T>();
        Ok(s)
    }
}

impl NativeScratch {
    fn new(a: &mut Alloc) -> Result<Self, RuntimeError> {
        let r = MAX_ROWS;
        let hd = GDN_HEADS * 128;
        Ok(Self {
            ids: a.zeros(r)?,
            resid: a.zeros(r * H)?,
            hidden: a.zeros(r * H)?,
            normed: a.zeros(r * H)?,
            x8: a.zeros(r * H)?,
            qkv: a.zeros(r * CONV)?,
            z: a.zeros(r * GDN_V)?,
            ab: a.zeros(r * GDN_AB)?,
            cv: a.zeros(r * CONV)?,
            gc: a.zeros(r * GDN_HEADS)?,
            w: a.zeros(r * hd)?,
            u: a.zeros(r * hd)?,
            aqk: a.zeros(r * GDN_HEADS * CHUNK as usize)?,
            core: a.zeros(r * hd)?,
            codes_out: a.zeros(r * AO)?,
            qg: a.zeros(r * ATTN_QG)?,
            k: a.zeros(r * ATTN_KVD)?,
            v: a.zeros(r * ATTN_KVD)?,
            q: a.zeros(r * AO)?,
            gate: a.zeros(r * AO)?,
            o: a.zeros(r * AO)?,
            attn_out: a.zeros(r * H)?,
            x4: a.zeros(r * H / 2)?,
            x4sf: a.zeros(fp4_scale_bytes(r as u32, HIDDEN))?,
            gu: a.zeros(r * 2 * I)?,
            d4: a.zeros(r * I / 2)?,
            d4sf: a.zeros(fp4_scale_bytes(r as u32, INTERMEDIATE))?,
        })
    }
}

/// The device addresses one layer's forward reads: its planes as stored (codes, then scales), its
/// norms, and a GDN layer's conv weights, dt bias, `ssm_a` and gated-norm weight.
struct LayerPlanes {
    /// GDN: qkv and z. Attention: q (with its gate), k and v.
    inputs: Vec<u64>,
    out: u64,
    mlp: [u64; 3],
    norm_in: u64,
    norm_post: u64,
    /// Conv weights, dt bias, `ssm_a`, gated-norm weight (the first 128 values of the tiled copy).
    gdn: Option<[u64; 4]>,
    /// The q and k norm weights.
    qk: Option<[u64; 2]>,
}

fn planes(device: &CudaDevice, w: &GpuWeightBuf, fp4: bool) -> Option<u64> {
    match (w, fp4) {
        (GpuWeightBuf::Fp8Raw(b), false) | (GpuWeightBuf::Nvfp4Raw(b), true) => Some(dp(device, b)),
        _ => None,
    }
}

/// The addresses of layer `l`'s weights, or which one is not resident as the route reads it.
fn layer_planes(
    device: &CudaDevice,
    l: usize,
    lw: &LayerWeightsGpu,
) -> Result<LayerPlanes, String> {
    let fp8 = |name: &str, w: Option<&GpuWeightBuf>| {
        w.and_then(|w| planes(device, w, false))
            .ok_or_else(|| format!("layer {l} {name} is not resident as its FP8 planes"))
    };
    let f32s = |name: &str, s: Option<&CudaSlice<f32>>| {
        s.map(|s| dp(device, s))
            .ok_or_else(|| format!("layer {l} {name} is not resident"))
    };
    let mut mlp = [0u64; 3];
    for (i, (name, w)) in [
        ("w_gate", &lw.w_gate),
        ("w_up", &lw.w_up),
        ("w_down", &lw.w_down),
    ]
    .into_iter()
    .enumerate()
    {
        mlp[i] = planes(device, w, true)
            .ok_or_else(|| format!("layer {l} {name} is not resident as its NVFP4 planes"))?;
    }
    let attention = is_attention_layer(l);
    let (inputs, out) = if attention {
        (
            vec![
                fp8("wq", Some(&lw.wq))?,
                fp8("wk", Some(&lw.wk))?,
                fp8("wv", Some(&lw.wv))?,
            ],
            fp8("wo", Some(&lw.wo))?,
        )
    } else {
        (
            vec![
                fp8("wq", Some(&lw.wq))?,
                fp8("attn_gate", lw.attn_gate.as_ref())?,
            ],
            fp8("ssm_out", lw.ssm_out.as_ref())?,
        )
    };
    Ok(LayerPlanes {
        inputs,
        out,
        mlp,
        norm_in: dp(device, &lw.attn_norm),
        norm_post: dp(device, &lw.ffn_norm),
        gdn: if attention {
            None
        } else {
            Some([
                f32s("ssm_conv1d", lw.ssm_conv1d.as_ref())?,
                f32s("ssm_dt", lw.ssm_dt_bias.as_ref())?,
                f32s("ssm_a", lw.ssm_a.as_ref())?,
                f32s("ssm_norm", lw.ssm_norm_tiled.as_ref())?,
            ])
        },
        qk: if attention {
            Some([
                f32s("attn_q_norm", lw.attn_q_norm.as_ref())?,
                f32s("attn_k_norm", lw.attn_k_norm.as_ref())?,
            ])
        } else {
            None
        },
    })
}

/// The published route.
pub(super) struct NativePrefill {
    k: NativePrefillKernels,
    gk: NativeGdnKernels,
    ak: NativeAttnKernels,
    gemm: NativeGemm,
    views: PrefillWeightViews,
    /// RoPE table rows [0, max_seq).
    rope: CudaSlice<f32>,
    /// BF16 staging of one attention layer's K and V, `[4][max_seq][256]` each, which the attention
    /// reads: allocated for an F32 KV store only, since a BF16 store is already that layout and is
    /// read in place.
    stage: Option<(CudaSlice<u16>, CudaSlice<u16>)>,
    max_seq: usize,
    scratch: NativeScratch,
    eps: f32,
    sm_count: u32,
    forwards: u64,
    #[cfg(feature = "test-prefill-dump")]
    dump: Option<PrefillDump>,
    /// The row a dumped layer would hand to decode, one device row kept for the route's life so no
    /// recorded buffer is freed and reused while the dump reads it.
    #[cfg(feature = "test-prefill-dump")]
    dump_row: CudaSlice<f32>,
    #[cfg(feature = "test-prefill-dump")]
    suspended: bool,
}

impl Drop for NativePrefill {
    fn drop(&mut self) {
        // No enqueued kernel or GEMM may still read the buffers when they are freed.
        let _ = self.scratch.ids.stream().synchronize();
    }
}

/// Publish the route for the weights `weights` just loaded into `st`, or name the first condition
/// it fails. On success the second value describes the route for the load log.
pub(super) fn publish(
    backend: &CudaBackend,
    st: &mut MutableState,
    weights: &dyn WeightProvider,
) -> Result<(NativePrefill, String), Refusal> {
    let device = &backend.device;
    let refusal = |condition: &'static str| {
        move |e: RuntimeError| Refusal {
            condition,
            reason: e.to_string(),
        }
    };
    injected(&["Q9"])?;
    route::switches(
        crate::runtime_defaults::native_prefill_enabled(),
        std::env::var("LUMEN_CUDA_PREFILL_F32").is_ok(),
    )?;
    injected(&["Q7"])?;
    if !matches!(st.kv_precision, KvPrecision::F32 | KvPrecision::Bf16) {
        return Err(Refusal {
            condition: "Q7",
            reason: format!(
                "the KV store is {:?}; the route writes F32 or BF16",
                st.kv_precision
            ),
        });
    }

    // Structure (Q0-Q6).
    injected(&["Q0", "Q0.a", "Q0.b", "Q0.c", "Q0.d", "Q5", "Q6"])?;
    let hp = *backend.hp().map_err(refusal("Q0"))?;
    // Every layer is read twice at most: here, its descriptor and the values admission checks from one
    // fetch per layer (a model refused at a layer is read no further), and once more below for the
    // weight views.
    let src = ProviderSlices::new(weights);
    let layers = admit_each(&hp, backend.embedding_quant, |l| src.subtensors(l), &src)?;
    if st.globals.embedding_bf16.is_none() {
        return Err(Refusal {
            condition: "Q0.d",
            reason: "the BF16 embedding is not resident on the device".into(),
        });
    }
    let planes = st
        .layer_weights_cache
        .iter()
        .enumerate()
        .map(|(l, lw)| layer_planes(device, l, lw))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|reason| Refusal {
            condition: "Q5",
            reason,
        })?;
    if planes.len() != hp.num_layers as usize {
        return Err(Refusal {
            condition: "Q5",
            reason: format!("{} of {} layers resident", planes.len(), hp.num_layers),
        });
    }

    // Kernels (Q1-Q3).
    injected(&["Q1", "Q2", "Q3"])?;
    let k = NativePrefillKernels::load(device)?;
    let gk = NativeGdnKernels::load(device)?;
    let ak = NativeAttnKernels::load(device)?;
    wide_ids_gather(device, &k, st, hp.vocab_size as usize)?;

    // Memory (Q8): the shared GDN state first, then the route's own buffers.
    injected(&["Q8"])?;
    let free_before = device.free_memory().unwrap_or(0);
    backend.ensure_gdn_scratch(st).map_err(refusal("Q8"))?;
    let views = PrefillWeightViews::build(device, H, I, &layers, &src).map_err(refusal("Q8"))?;
    let max_seq = st.kv_max_seq_len;
    let mut a = Alloc { device, bytes: 0 };
    let scratch = NativeScratch::new(&mut a).map_err(refusal("Q8"))?;
    let scratch_bytes = a.bytes;
    let stage = if st.kv_precision == KvPrecision::F32 {
        let n_stage = (route::KV_HEADS * route::HEAD_DIM) as usize * max_seq;
        Some((
            a.zeros::<u16>(n_stage).map_err(refusal("Q8"))?,
            a.zeros::<u16>(n_stage).map_err(refusal("Q8"))?,
        ))
    } else {
        None
    };
    let rope = a.zeros::<f32>(max_seq * ROT).map_err(refusal("Q8"))?;
    // SAFETY: `rope` holds `max_seq` rows of the table.
    unsafe { ak.rope_table(device, dp(device, &rope), max_seq as u32, ROPE.theta) }
        .map_err(refusal("Q8"))?;
    let staging_bytes = a.bytes - scratch_bytes;

    // GEMM plans (Q4). Selection, when the plan cache cannot be used, times and verifies each shape
    // on a real layer's weights: layer 0 (GDN) and layer 3 (attention).
    injected(&["Q4"])?;
    let fp8 = |plane: u64, n: usize, k: usize| WeightOperand {
        plane,
        scale: plane + (n * k) as u64,
        unit_alpha: 1.0,
    };
    let fp4 = |i: usize| WeightOperand {
        plane: planes[0].mlp[i],
        scale: dp(device, &views.mlp_scales[0][i]),
        unit_alpha: views.globals[0][i],
    };
    let (g, at) = (&planes[0], &planes[3]);
    let operands = [
        vec![fp4(0)],
        vec![fp4(2)],
        vec![fp8(g.inputs[0], CONV, H)],
        vec![fp8(g.inputs[1], GDN_V, H)],
        vec![fp8(g.out, H, AO)],
        vec![fp8(at.inputs[0], ATTN_QG, H)],
        vec![fp8(at.inputs[1], ATTN_KVD, H)],
    ];
    // SAFETY: every operand is a resident layer plane or view of its shape.
    let (gemm, report) = unsafe { NativeGemm::build(device, &operands) }.map_err(refusal("Q4"))?;
    let free_after = device.free_memory().unwrap_or(0);
    let plans = match &report.source {
        TableSource::Cached(path) => format!("read from {}", path.display()),
        TableSource::Selected { reason, stored, .. } => format!(
            "selected in {:.1} s ({reason}){}",
            report.seconds,
            stored
                .as_ref()
                .map(|p| format!(", stored at {}", p.display()))
                .unwrap_or_default()
        ),
    };
    let mb = |b: usize| b as f64 / (1u64 << 20) as f64;
    let line = format!(
        "{}; plans {plans}; views {:.0} MiB, scratch {:.0} MiB, KV staging and RoPE table {:.0} MiB \
         for {max_seq} positions; {:.0} MiB of device memory taken with the GDN state",
        library_report(),
        mb(views.device_bytes()),
        mb(scratch_bytes),
        mb(staging_bytes),
        mb(free_before.saturating_sub(free_after)),
    );
    let sm_count = k.sm_count();
    Ok((
        NativePrefill {
            k,
            gk,
            ak,
            gemm,
            views,
            rope,
            stage,
            max_seq,
            scratch,
            eps: hp.norm_eps,
            sm_count,
            forwards: 0,
            #[cfg(feature = "test-prefill-dump")]
            dump: None,
            #[cfg(feature = "test-prefill-dump")]
            dump_row: device.alloc_zeros::<f32>(H).map_err(refusal("Q8"))?,
            #[cfg(feature = "test-prefill-dump")]
            suspended: false,
        },
        line,
    ))
}

/// Gather two rows of the resident embedding with the route's gather, one of them at an id above
/// 65535 (the vocabulary's last when it is smaller), and compare them with the rows as stored: a
/// gather that read its ids narrower than 32 bits would return other rows. Refused as Q3.
fn wide_ids_gather(
    device: &CudaDevice,
    k: &NativePrefillKernels,
    st: &MutableState,
    vocab: usize,
) -> Result<(), Refusal> {
    let q3 = |reason: String| Refusal {
        condition: "Q3",
        reason: format!("native_embed_gather_bf16 on the resident embedding: {reason}"),
    };
    let err = |e: RuntimeError| q3(e.to_string());
    let table = st
        .globals
        .embedding_bf16
        .as_ref()
        .ok_or_else(|| q3("no BF16 embedding".into()))?;
    let ids = [65537.min(vocab - 1) as u32, (vocab - 1) as u32];
    let row = 2 * H;
    let d_ids = device.htod_copy(&ids).map_err(err)?;
    let out = device.alloc_zeros::<u16>(2 * H).map_err(err)?;
    // SAFETY: both ids are rows of the table; `out` holds two rows.
    unsafe {
        k.embed_gather_bf16(
            device,
            dp(device, table),
            dp(device, &d_ids),
            2,
            dp(device, &out),
        )
    }
    .map_err(err)?;
    let got: Vec<u8> = device
        .dtoh_copy(&out)
        .map_err(err)?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    for (i, &id) in ids.iter().enumerate() {
        let at = id as usize * row;
        let want = device
            .dtoh_copy_view(&table.slice(at..at + row))
            .map_err(err)?;
        if got[i * row..(i + 1) * row] != want[..] {
            return Err(q3(format!("row {id} differs from the stored row")));
        }
    }
    Ok(())
}

/// What a forward writes besides its own buffers.
pub(super) struct DecodeState<'a> {
    pub layers: &'a [LayerWeightsGpu],
    pub gdn: &'a mut GdnScratchGpu,
    pub kv: &'a mut [Option<KvCacheGpu>],
    pub embedding: &'a CudaSlice<u8>,
    pub x_gpu: &'a mut CudaSlice<f32>,
}

impl NativePrefill {
    /// Whether prompts are prefilled by this route (a test can suspend it between prompts).
    pub(super) fn serves(&self) -> bool {
        #[cfg(feature = "test-prefill-dump")]
        if self.suspended {
            return false;
        }
        true
    }

    /// Prefill `tokens` at positions from `pos_start`, in slices of at most [`MAX_ROWS`] tokens, and
    /// leave the last position's hidden row in `state.x_gpu`. An error leaves the state partly
    /// written, as the F32 route's does.
    pub(super) fn run(
        &mut self,
        device: &CudaDevice,
        state: DecodeState,
        tokens: &[u32],
        pos_start: usize,
    ) -> Result<(), RuntimeError> {
        let planes = state
            .layers
            .iter()
            .enumerate()
            .map(|(l, lw)| layer_planes(device, l, lw))
            .collect::<Result<Vec<_>, _>>()
            .map_err(RuntimeError::Compute)?;
        let DecodeState {
            gdn,
            kv,
            embedding,
            x_gpu,
            ..
        } = state;
        let mut last = 0;
        for (i, slice) in tokens.chunks(MAX_ROWS).enumerate() {
            self.slice(
                device,
                &planes,
                gdn,
                kv,
                dp(device, embedding),
                slice,
                pos_start + i * MAX_ROWS,
            )?;
            last = slice.len();
        }
        let s = &self.scratch;
        // SAFETY: the rows were written by the last slice; `x_gpu` holds 5120 values.
        unsafe {
            self.k.final_row_f32(
                device,
                dp(device, &s.hidden),
                dp(device, &s.resid),
                last as u32 - 1,
                dp(device, x_gpu),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn slice(
        &mut self,
        device: &CudaDevice,
        planes: &[LayerPlanes],
        gdn: &mut GdnScratchGpu,
        kv: &mut [Option<KvCacheGpu>],
        embedding: u64,
        tokens: &[u32],
        p0: usize,
    ) -> Result<(), RuntimeError> {
        let t = tokens.len();
        let m = bucket(t)
            .map(bucket_rows)
            .ok_or_else(|| RuntimeError::Compute(format!("a native slice of {t} tokens")))?;
        let s = &mut self.scratch;
        device
            .stream
            .memcpy_htod(tokens, &mut s.ids.slice_mut(0..t))
            .map_err(|e| RuntimeError::Compute(format!("native prefill token ids: {e}")))?;
        // Each GEMM runs at the bucket's m rows and reads its activation rows [t, m) too. The producers
        // write rows [0, t) only, so after a longer earlier prompt those rows hold its values; zeroed
        // here, every GEMM input is zero past the prompt, as the plans' verification assumed.
        if m > t {
            for (buf, cols) in [
                (dp(device, &s.x8), H),
                (dp(device, &s.codes_out), AO),
                (dp(device, &s.x4), H / 2),
                (dp(device, &s.d4), I / 2),
            ] {
                // SAFETY: rows [t, m) of a buffer of MAX_ROWS rows of `cols` bytes.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        buf + (t * cols) as u64,
                        0,
                        (m - t) * cols,
                        device.stream.cu_stream(),
                    )
                }
                .map_err(|e| RuntimeError::Compute(format!("native prefill padding: {e}")))?;
            }
        }
        // SAFETY: the ids were checked against the vocabulary by `prefill`; `resid` holds t rows.
        unsafe {
            self.k.embed_gather_bf16(
                device,
                embedding,
                dp(device, &s.ids),
                t as u32,
                dp(device, &s.resid),
            )?;
        }
        for (l, lp) in planes.iter().enumerate() {
            #[cfg(any(test, feature = "test-fault-injection"))]
            if route::fault::fails_at(l) {
                return Err(RuntimeError::Compute(format!(
                    "native prefill layer {l}: injected failure"
                )));
            }
            self.layer(device, l, lp, gdn, kv, t, p0)?;
        }
        self.forwards += 1;
        Ok(())
    }

    /// Record `bytes` bytes at `ptr` as `name` of layer `l` when a dump wants it.
    #[cfg(feature = "test-prefill-dump")]
    fn rec(
        &mut self,
        device: &CudaDevice,
        l: usize,
        name: &'static str,
        ptr: u64,
        bytes: usize,
    ) -> Result<(), RuntimeError> {
        match self.dump.as_mut() {
            // SAFETY: every recorded range lies inside a live buffer of the route or the backend.
            Some(d) => unsafe { d.record(device, l, name, ptr, bytes) },
            None => Ok(()),
        }
    }

    #[cfg(not(feature = "test-prefill-dump"))]
    #[inline(always)]
    fn rec(
        &mut self,
        _device: &CudaDevice,
        _l: usize,
        _name: &'static str,
        _ptr: u64,
        _bytes: usize,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// Record host bytes as `name` of layer `l` when a dump wants it.
    #[allow(unused_variables)]
    fn rec_host(&mut self, l: usize, name: &'static str, bytes: impl FnOnce() -> Vec<u8>) {
        #[cfg(feature = "test-prefill-dump")]
        if let Some(d) = self.dump.as_mut() {
            d.record_host(l, name, bytes());
        }
    }

    /// Record an attention layer's whole K cache followed by its whole V cache, `bytes` each, as
    /// `name`.
    #[allow(unused_variables)]
    fn rec_kv(
        &mut self,
        device: &CudaDevice,
        l: usize,
        name: &'static str,
        k: u64,
        v: u64,
        bytes: usize,
    ) -> Result<(), RuntimeError> {
        #[cfg(feature = "test-prefill-dump")]
        if self.dump.as_ref().is_some_and(|d| d.wants(l)) {
            device.synchronize()?;
            let mut host = vec![0u8; 2 * bytes];
            for (i, at) in [k, v].into_iter().enumerate() {
                // SAFETY: each cache holds `bytes` bytes.
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_sync(
                        &mut host[i * bytes..(i + 1) * bytes],
                        at,
                    )
                }
                .map_err(|e| RuntimeError::Compute(format!("dump layer {l} {name}: {e}")))?;
            }
            self.rec_host(l, name, || host);
        }
        Ok(())
    }

    /// One layer of a slice of `t` tokens at positions from `p0`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        device: &CudaDevice,
        l: usize,
        lp: &LayerPlanes,
        gdn: &mut GdnScratchGpu,
        kv: &mut [Option<KvCacheGpu>],
        t: usize,
        p0: usize,
    ) -> Result<(), RuntimeError> {
        let d = device;
        let tu = t as u32;
        let eps = self.eps;
        let attention = is_attention_layer(l);
        let scales = self.views.scales[l];
        let [g_gate, g_up, g_down] = self.views.globals[l];
        // The FP8 GEMMs read their activation scales from the device table: [proj_in, proj_out].
        let s_in = dp(d, &self.views.fp8_scales) + (8 * l) as u64;
        let s_out = s_in + 4;
        let p = |s: &CudaSlice<u16>| dp(d, s);
        let p8 = |s: &CudaSlice<u8>| dp(d, s);
        let s = &self.scratch;
        let (resid, hidden, normed, x8) = (p(&s.resid), p(&s.hidden), p(&s.normed), p8(&s.x8));
        let codes_out = p8(&s.codes_out);
        let attn_out = p(&s.attn_out);

        self.rec(d, l, "x", if l == 0 { resid } else { hidden }, t * H * 2)?;
        if l > 0 {
            self.rec(d, l, "resid_in", resid, t * H * 2)?;
        }

        // The input norm (at layer 0 the residual is the embedded rows and is not written).
        // SAFETY (this and every launch below): each address is a buffer sized for its launch (the
        // slice's bucket rows for activations, the model's shapes for weights, `max_seq` rows for the
        // KV caches, staging and RoPE table, the conv ring and GDN state) and freed only in stream
        // order; inputs are written earlier on the device stream and no output overlaps another
        // argument; the prefill's capacity check keeps `p0 + t <= max_seq`, and
        // `next_conv_position` keeps `state_pos < 3`.
        unsafe {
            if l == 0 {
                self.k
                    .rmsnorm_fp8(d, resid, lp.norm_in, eps, tu, scales.proj_in, x8, normed)?;
            } else {
                self.k.add_rmsnorm_fp8(
                    d,
                    hidden,
                    resid,
                    lp.norm_in,
                    eps,
                    tu,
                    scales.proj_in,
                    x8,
                    normed,
                )?;
            }
        }
        self.rec(d, l, "resid_attn", resid, t * H * 2)?;
        self.rec(d, l, "normed", normed, t * H * 2)?;
        self.rec(d, l, "x8", x8, t * H)?;

        if !attention {
            let g = gdn.gdn_layer_map[l]
                .ok_or_else(|| RuntimeError::Compute(format!("layer {l} has no GDN state")))?;
            let [conv_w, dt_bias, ssm_a, gnorm] = lp
                .gdn
                .ok_or_else(|| RuntimeError::Compute(format!("layer {l} GDN weights")))?;
            let ring = dp(d, &gdn.conv_states[g]);
            let state = dp(d, &gdn.h_states[g]);
            let pos = gdn.conv_positions[g];
            let ring_bytes = gdn.conv_states[g].len() * 4;
            let state_bytes = gdn.h_states[g].len() * 4;
            self.rec(d, l, "ring_in", ring, ring_bytes)?;
            self.rec(d, l, "state_in", state, state_bytes)?;
            self.rec_host(l, "state_pos_in", || pos.to_le_bytes().to_vec());

            let s = &self.scratch;
            let (qkv, z, ab) = (p(&s.qkv), p(&s.z), dp(d, &s.ab));
            self.gemm(GDN_QKV, t, 1.0, lp.inputs[0], CONV, x8, s_in, qkv)?;
            self.gemm(GDN_Z, t, 1.0, lp.inputs[1], GDN_V, x8, s_in, z)?;
            let ab_w = self.views.ab[l]
                .as_ref()
                .map(|w| dp(d, w))
                .ok_or_else(|| RuntimeError::Compute(format!("layer {l} a/b view")))?;
            ab_gemm(d, ab_w, normed, ab, t)?;
            self.rec(d, l, "qkv", qkv, t * CONV * 2)?;
            self.rec(d, l, "z", z, t * GDN_V * 2)?;
            self.rec(d, l, "ab", ab, t * GDN_AB * 4)?;

            let s = &self.scratch;
            let (cv, gc, w, u, aqk, core) = (
                p(&s.cv),
                dp(d, &s.gc),
                p(&s.w),
                p(&s.u),
                p(&s.aqk),
                p(&s.core),
            );
            unsafe {
                self.gk.conv(d, qkv, ring, conv_w, cv, tu, pos)?;
                self.gk
                    .chunk_intra(d, cv, ab, dt_bias, ssm_a, gc, w, u, aqk, tu)?;
                self.gk.chunk_state(d, cv, gc, w, u, aqk, state, core, tu)?;
            }
            let pos = next_conv_position(pos, tu);
            gdn.conv_positions[g] = pos;
            self.rec(d, l, "cv", cv, t * CONV * 2)?;
            self.rec(d, l, "ring", ring, ring_bytes)?;
            self.rec(d, l, "core", core, t * GDN_V * 2)?;
            self.rec(d, l, "state", state, state_bytes)?;
            self.rec_host(l, "state_pos", || pos.to_le_bytes().to_vec());

            let rows = tu * GDN_HEADS as u32;
            let lanes = gdn_norm_lanes_per_row(rows, self.sm_count);
            unsafe {
                self.k.gdn_norm_gate_fp8(
                    d,
                    core,
                    z,
                    gnorm,
                    eps,
                    rows,
                    lanes,
                    scales.proj_out,
                    codes_out,
                )?;
            }
            self.rec(d, l, "y8", codes_out, t * AO)?;
        } else {
            let s = &self.scratch;
            let (qg, kk, vv) = (p(&s.qg), p(&s.k), p(&s.v));
            self.gemm(ATTN_Q, t, 1.0, lp.inputs[0], ATTN_QG, x8, s_in, qg)?;
            self.gemm(ATTN_KV, t, 1.0, lp.inputs[1], ATTN_KVD, x8, s_in, kk)?;
            self.gemm(ATTN_KV, t, 1.0, lp.inputs[2], ATTN_KVD, x8, s_in, vv)?;
            self.rec(d, l, "qg", qg, t * ATTN_QG * 2)?;
            self.rec(d, l, "k", kk, t * ATTN_KVD * 2)?;
            self.rec(d, l, "v", vv, t * ATTN_KVD * 2)?;

            let cache = kv
                .get_mut(l)
                .and_then(Option::as_mut)
                .ok_or_else(|| RuntimeError::Compute(format!("layer {l} has no KV cache")))?;
            if cache.seq_len() != p0 || cache.max_seq_len != self.max_seq {
                return Err(RuntimeError::KvCache(format!(
                    "layer {l}: the KV cache holds {} of {} positions; the native prefill starts at \
                     {p0} with staging for {}",
                    cache.seq_len(),
                    cache.max_seq_len,
                    self.max_seq
                )));
            }
            // The rows the attention reads, BF16: the staging beside an F32 store (the prep writes
            // both, and older rows are restaged from the store), or a BF16 store itself (the prep
            // writes it alone, null F32 caches). `(kr, vr, bytes)` is the store as a dump records it.
            let (kc, vc, ks, vs, (kr, vr, bytes)) = match (&cache.store, &self.stage) {
                (KvStore::F32 { k, v }, Some((k_stage, v_stage))) => {
                    let (kc, vc) = (dp(d, k), dp(d, v));
                    (kc, vc, p(k_stage), p(v_stage), (kc, vc, 4))
                }
                (KvStore::Bf16 { k, v }, None) => {
                    let (ks, vs) = (p(k), p(v));
                    (0, 0, ks, vs, (ks, vs, 2))
                }
                _ => {
                    return Err(RuntimeError::KvCache(format!(
                        "layer {l}: the native prefill writes an F32 or BF16 KV store, the one it \
                         was published for"
                    )))
                }
            };
            let n = (route::KV_HEADS * route::HEAD_DIM) as usize * self.max_seq;
            let cs = dp(d, &self.rope);
            let ms = self.max_seq as u32;
            let [q_w1, k_w1] = lp
                .qk
                .ok_or_else(|| RuntimeError::Compute(format!("layer {l} q/k norms")))?;
            self.rec_host(l, "p0", || (p0 as u32).to_le_bytes().to_vec());
            self.rec(d, l, "cs", cs + (p0 * ROT * 4) as u64, t * ROT * 4)?;
            self.rec_kv(d, l, "kv_in", kr, vr, n * bytes)?;
            let s = &self.scratch;
            let (q, gate, o) = (p(&s.q), p(&s.gate), p(&s.o));
            let p0u = p0 as u32;
            unsafe {
                self.ak.prep(
                    d, qg, kk, vv, q_w1, k_w1, cs, eps, tu, p0u, ms, q, gate, kc, vc, ks, vs,
                )?;
                if kc != 0 {
                    self.ak.kv_to_bf16(d, kc, vc, ks, vs, p0u, ms)?;
                }
                self.ak.attention(d, q, ks, vs, o, tu, p0u, ms)?;
            }
            self.rec(d, l, "q", q, t * AO * 2)?;
            self.rec(d, l, "gate", gate, t * AO * 2)?;
            self.rec_kv(d, l, "kv", kr, vr, n * bytes)?;
            self.rec(d, l, "o", o, t * AO * 2)?;
            cache.advance_seq_len_by(t);
            unsafe {
                self.k
                    .sigmoid_gate_fp8(d, o, gate, tu, scales.proj_out, codes_out)?;
            }
            self.rec(d, l, "o8", codes_out, t * AO)?;
        }

        self.gemm(OUT, t, 1.0, lp.out, H, codes_out, s_out, attn_out)?;
        self.rec(d, l, "attn_out", attn_out, t * H * 2)?;

        // The MLP.
        let s = &self.scratch;
        let (x4, x4sf, gu, d4, d4sf) = (p8(&s.x4), p8(&s.x4sf), p(&s.gu), p8(&s.d4), p8(&s.d4sf));
        unsafe {
            self.k.add_rmsnorm_fp4(
                d,
                attn_out,
                resid,
                lp.norm_post,
                eps,
                tu,
                1.0 / scales.gate_up,
                x4,
                x4sf,
            )?;
        }
        self.rec(d, l, "resid_mlp", resid, t * H * 2)?;
        self.rec(d, l, "x4", x4, t * H / 2)?;
        self.rec(d, l, "x4sf", x4sf, fp4_scale_bytes(tu, HIDDEN))?;
        let [gate_sf, up_sf, down_sf] = [0, 1, 2].map(|i| dp(d, &self.views.mlp_scales[l][i]));
        let [gate_w, up_w, down_w] = lp.mlp;
        let gs = scales.gate_up;
        self.gemm_fp4(GATE_UP, t, gs * g_gate, gate_w, gate_sf, x4, x4sf, gu)?;
        self.gemm_fp4(
            GATE_UP,
            t,
            gs * g_up,
            up_w,
            up_sf,
            x4,
            x4sf,
            gu + (I * 2) as u64,
        )?;
        self.rec(d, l, "gu", gu, t * 2 * I * 2)?;
        unsafe {
            self.k
                .silu_mul_fp4(d, attention, gu, tu, 1.0 / scales.down, d4, d4sf)?;
        }
        self.rec(d, l, "d4", d4, t * I / 2)?;
        self.rec(d, l, "d4sf", d4sf, fp4_scale_bytes(tu, INTERMEDIATE))?;
        let alpha = scales.down * g_down;
        self.gemm_fp4(DOWN, t, alpha, down_w, down_sf, d4, d4sf, hidden)?;
        self.rec(d, l, "mlp_out", hidden, t * H * 2)?;
        #[cfg(feature = "test-prefill-dump")]
        if self.dump.as_ref().is_some_and(|dump| dump.wants(l)) {
            // The row this layer would hand to decode were it the last.
            let row = dp(d, &self.dump_row);
            unsafe {
                self.k.final_row_f32(d, hidden, resid, tu - 1, row)?;
            }
            self.rec(d, l, "x_gpu", row, H * 4)?;
        }
        Ok(())
    }

    /// An FP8 GEMM: `w` the planes of an `n x k` weight (codes, then the weight scale at `n * k`).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        shape: usize,
        t: usize,
        alpha: f32,
        w: u64,
        n: usize,
        x: u64,
        x_scale: u64,
        out: u64,
    ) -> Result<(), RuntimeError> {
        let k = crate::cuda::native_prefill_gemm::SHAPES[shape].k;
        self.gemm_fp4(shape, t, alpha, w, w + (n * k) as u64, x, x_scale, out)
    }

    /// A GEMM of `t` rows with explicit weight and activation scale addresses.
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp4(
        &mut self,
        shape: usize,
        t: usize,
        alpha: f32,
        w: u64,
        w_scale: u64,
        x: u64,
        x_scale: u64,
        d: u64,
    ) -> Result<(), RuntimeError> {
        // SAFETY: the operands are live device buffers sized for the bucket's rows, their rows past
        // `t` zero (see `slice`), enqueued on the device stream.
        unsafe {
            self.gemm.run(
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
        }
    }
}

/// `ab[t][96] = normed[t][5120] * w[96][5120]^T`, BF16 in, F32 accumulate and out: the cuBLAS call
/// of the F32 route's BF16 GEMM launcher, on addresses.
fn ab_gemm(
    device: &CudaDevice,
    w: u64,
    normed: u64,
    ab: u64,
    t: usize,
) -> Result<(), RuntimeError> {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: `w` holds [96][5120] BF16, `normed` [t][5120] BF16, `ab` [t][96] F32; the handle runs
    // on the device stream.
    let status = unsafe {
        cublas_sys::cublasGemmEx(
            *device.blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            GDN_AB as i32,
            t as i32,
            H as i32,
            &alpha as *const f32 as *const std::ffi::c_void,
            w as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            H as i32,
            normed as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            H as i32,
            &beta as *const f32 as *const std::ffi::c_void,
            ab as *mut std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            GDN_AB as i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "native prefill a/b GEMM: status={status:?}"
        )));
    }
    Ok(())
}

/// The route's test-only surface (feature `test-prefill-dump`).
#[cfg(feature = "test-prefill-dump")]
impl CudaBackend {
    fn with_route<R>(&self, f: impl FnOnce(&mut MutableState) -> R) -> R {
        let mut guard = self.state.lock().unwrap();
        f(guard.as_mut().expect("CUDA backend not initialized"))
    }

    /// Record the tensors of `layers` in every later native forward (a new, empty dump), or stop
    /// recording with `None`. No effect on the F32 route.
    pub fn set_native_prefill_dump(&self, layers: Option<Vec<usize>>) {
        self.with_route(|st| {
            if let Some(np) = st.native_prefill.as_mut() {
                np.dump = layers.map(PrefillDump::new);
            }
        })
    }

    /// The dump recorded since [`Self::set_native_prefill_dump`], which stops recording.
    pub fn take_native_prefill_dump(&self) -> Option<PrefillDump> {
        self.with_route(|st| st.native_prefill.as_mut().and_then(|np| np.dump.take()))
    }

    /// Native forwards (one per slice) since load, or `None` when the route is not published.
    pub fn native_prefill_forwards(&self) -> Option<u64> {
        self.with_route(|st| st.native_prefill.as_ref().map(|np| np.forwards))
    }

    /// The condition publication refused, or `None` when the native route is published.
    pub fn native_prefill_refusal(&self) -> Option<Refusal> {
        self.with_route(|st| st.native_prefill_refusal.clone())
    }

    /// Prefill with the F32 route (`true`) or the published native route (`false`) from the next
    /// prompt on. The F32 route reads a 16-bit KV store through widening buffers, which exist only
    /// when the native route was refused, so over a published route the KV store must be F32.
    pub fn set_native_prefill_suspended(&self, suspended: bool) {
        self.with_route(|st| {
            if let Some(np) = st.native_prefill.as_mut() {
                np.suspended = suspended;
            }
        })
    }
}
