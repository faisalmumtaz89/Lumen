//! Batched prefill pipeline for the CUDA backend.
//!
//! Replaces token-at-a-time prefill (128 * 32 * 15 = 61,440 kernel launches
//! for pp128 on 8B) with batched operations (~700 launches). F32 projections
//! use cuBLAS SGEMM; Q8_0 weights use cublasGemmEx HGEMM via pre-dequanted
//! F16 caches (tensor core path); native F16 weights use cublasGemmEx HGEMM
//! directly (no dequant needed -- already in the right format). Q4_0 falls
//! back to per-row matvec. Attention remains sequential per token since each
//! token's causal mask differs.
//!
//! cuBLAS SGEMM for row-major matrices:
//!
//! Our weight W is [out_dim, in_dim] row-major. Activation A is [batch, in_dim]
//! row-major. Output C is [batch, out_dim] row-major. We want C = A * W^T.
//!
//! cuBLAS is column-major. Row-major M[R,C] looks like col-major M_cm[C,R].
//! So W_cm is [in_dim, out_dim], A_cm is [in_dim, batch], C_cm is [out_dim, batch].
//!
//! C_cm = W_cm^T * A_cm => transa=T, transb=N
//! m=out_dim, n=batch, k=in_dim
//! A(cublas)=W, lda=in_dim, B(cublas)=A, ldb=in_dim, C(cublas)=C, ldc=out_dim

use cudarc::cublas::{sys as cublas_sys, Gemm, GemmConfig};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig as CudarcLaunchConfig,
    PushKernelArg,
};

use crate::error::RuntimeError;

use super::decode::{matvec_block_size, rmsnorm_block_size, rmsnorm_shared_bytes, KernelSet};
use super::ffi::CudaDevice;
use super::gpu_buffers::GpuWeightBuf;
use super::kv_cache::KvView;
use super::types::LaunchConfig;
use super::workspace::{Rest, Role, Workspace};
use lumen_format::quantization::QuantScheme;

/// Element counts of the GDN prefill's workspace values for `batch` rows: the
/// QKV projection and conv output (`qkv`), the four per-head gate values
/// (`heads`) and the three value-side outputs (`value`), all F32. The
/// workspace planner places exactly these, and the load-time memory plan
/// counts what it places, so the two cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnPrefillScratchLayout {
    pub qkv: usize,
    pub heads: usize,
    pub value: usize,
}

impl GdnPrefillScratchLayout {
    pub(crate) fn new(batch: usize, qkv_dim: usize, num_heads: usize, value_dim: usize) -> Self {
        Self {
            qkv: batch * qkv_dim,
            heads: batch * num_heads,
            value: batch * value_dim,
        }
    }
}

/// Element counts of the prefill's workspace values for `batch` rows of one
/// model's geometry (`activation_f16` and `dequant_f16` in bytes). The
/// workspace planner places these, the conversion scratch at most these
/// ([`ConversionScratch::plan`]), and the load-time memory plan counts what
/// it places, so the two cannot disagree.
///
/// `dequant_f32` holds the largest projection weight matrix in F32 across ALL
/// kernels that share this scratch: standard attention (wq/wk/wv/wo), FFN
/// (w_gate/w_up/w_down), AND -- when present -- GDN fused QKV (out_dim =
/// gdn_qkv_dim, in_dim = hidden_dim) and the GDN SSM output projection
/// (out_dim = hidden_dim, in_dim = gdn_value_dim). The GDN dimensions are
/// `Some` for models with GDN layers (Qwen3.5 family) and `None` for
/// pure-attention models; Qwen3.5-35B-A3B's `qkv_dim * hidden_dim = 8192 *
/// 2048` exceeds its attention/FFN maximum, so they cannot be left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillScratchLayout {
    pub hidden: usize,
    pub q: usize,
    pub kv: usize,
    pub inter: usize,
    pub q_gate: Option<(usize, usize)>,
    pub token_ids: usize,
    pub q_single: usize,
    pub dequant_f32: usize,
    pub activation_f16: usize,
    pub dequant_f16: usize,
}

impl PrefillScratchLayout {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        batch: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        gdn_qkv_dim: Option<usize>,
        gdn_value_dim: Option<usize>,
        qgate_fused: bool,
    ) -> Self {
        let gdn_qkv = gdn_qkv_dim.unwrap_or(0);
        let gdn_value = gdn_value_dim.unwrap_or(0);
        let max_weight_elems = [
            q_dim * hidden_dim, // wq
            // wq with Q+gate fusion projects out = q_dim * 2 through this scratch.
            if qgate_fused {
                2 * q_dim * hidden_dim
            } else {
                0
            },
            kv_dim * hidden_dim,    // wk, wv
            hidden_dim * q_dim,     // wo
            inter_dim * hidden_dim, // w_gate, w_up
            hidden_dim * inter_dim, // w_down
            gdn_qkv * hidden_dim,   // GDN fused QKV projection
            hidden_dim * gdn_value, // GDN SSM output projection
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        // The F16 activation buffer covers the widest activation row any
        // kernel converts before cublasGemmEx HGEMM, GDN value-side inputs
        // (ssm_out reads rows of `value_dim`) included.
        let max_in_dim = [hidden_dim, inter_dim, q_dim, gdn_value]
            .into_iter()
            .max()
            .unwrap_or(hidden_dim);
        Self {
            hidden: batch * hidden_dim,
            q: batch * q_dim,
            kv: batch * kv_dim,
            inter: batch * inter_dim,
            q_gate: qgate_fused.then_some((batch * q_dim * 2, batch * q_dim)),
            token_ids: batch,
            q_single: q_dim,
            dequant_f32: max_weight_elems,
            activation_f16: batch * max_in_dim * 2,
            dequant_f16: max_weight_elems * 2,
        }
    }
}

/// The paths of a prefill slice a load can take, beyond those the layouts'
/// optional parts record (the fused Q+gate attention, the tiled attention,
/// the GDN layers).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefillRoutes {
    /// An attention layer without the fused Q+gate projection.
    pub plain_attention: bool,
    /// The GDN layers' fused batched state update; otherwise their per-token
    /// loop over the decode kernels.
    pub gdn_fused: bool,
    /// A layer with a dense FFN.
    pub dense_ffn: bool,
    /// A layer with a MoE FFN.
    pub moe_ffn: bool,
}

/// What [`alloc_prefill_workspace`] allocates: the scratch layouts for the
/// longest slice, when the tiled attention is the route the score block's
/// element count for the whole context, when a MoE layer takes the grouped
/// route its scratch for the longest slice, the paths a slice can take and
/// the conversion scratch its GEMMs take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillWorkspaceLayout {
    pub scratch: PrefillScratchLayout,
    pub gdn: Option<GdnPrefillScratchLayout>,
    pub score_elems: Option<usize>,
    pub grouped: Option<super::moe::PrefillGroupedLayout>,
    pub routes: PrefillRoutes,
    pub conversion: ConversionScratch,
    /// Place every value in its own region (the unshared layout) instead of
    /// sharing bytes between values never live together, so a test can
    /// compare the two.
    #[cfg(feature = "test-state-snapshot")]
    pub unshared: bool,
}

impl PrefillWorkspaceLayout {
    /// Where each value lies in the workspace's arena.
    pub(crate) fn regions(&self) -> super::workspace::RegionTable {
        #[cfg(feature = "test-state-snapshot")]
        if self.unshared {
            return super::workspace::unshared(self);
        }
        super::workspace::compact(self, &super::workspace::liveness(self))
    }

    /// Device bytes of the whole workspace: the arena its regions lie in.
    pub(crate) fn bytes(&self) -> u64 {
        self.regions().bytes() as u64
    }

    /// The layout of these parts with the compact planner.
    #[cfg(test)]
    pub(crate) fn of(
        scratch: PrefillScratchLayout,
        gdn: Option<GdnPrefillScratchLayout>,
        score_elems: Option<usize>,
        grouped: Option<super::moe::PrefillGroupedLayout>,
        routes: PrefillRoutes,
        conversion: ConversionScratch,
    ) -> Self {
        Self {
            scratch,
            gdn,
            score_elems,
            grouped,
            routes,
            conversion,
            #[cfg(feature = "test-state-snapshot")]
            unshared: false,
        }
    }
}

/// The served models' prefill geometry at the 2048-row slice and a 32768-token
/// context, for the workspace's unit tests.
#[cfg(test)]
pub(crate) mod test_models {
    use super::*;

    /// A model's geometry: hidden width, query and KV heads of `head_dim`,
    /// FFN width, GDN `(qkv_dim, num_heads, value_dim)`, and whether its FFN
    /// is MoE (no dense FFN).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Model {
        pub name: &'static str,
        pub hidden: usize,
        pub heads: usize,
        pub kv_heads: usize,
        pub head_dim: usize,
        pub inter: usize,
        pub gdn: (usize, usize, usize),
        pub moe: bool,
    }

    /// Qwen3.8-27B: 24 query / 4 KV heads of 256, FFN 17408, GDN 10240 / 48 / 6144.
    pub(crate) const Q27B: Model = Model {
        name: "27B",
        hidden: 5120,
        heads: 24,
        kv_heads: 4,
        head_dim: 256,
        inter: 17408,
        gdn: (10240, 48, 6144),
        moe: false,
    };
    /// Qwen3.5-9B: 16 query / 4 KV heads of 256, FFN 12288, GDN 8192 / 32 / 4096.
    pub(crate) const Q9B: Model = Model {
        name: "9B",
        hidden: 4096,
        heads: 16,
        kv_heads: 4,
        head_dim: 256,
        inter: 12288,
        gdn: (8192, 32, 4096),
        moe: false,
    };
    /// Qwen3.5-35B-A3B: 16 query / 2 KV heads of 256, expert FFN 512, GDN
    /// 8192 / 32 / 4096, every layer MoE.
    pub(crate) const MOE: Model = Model {
        name: "35B-A3B",
        hidden: 2048,
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        inter: 512,
        gdn: (8192, 32, 4096),
        moe: true,
    };

    pub(crate) const ROWS: usize = 2048;
    pub(crate) const CONTEXT: usize = 32768;

    impl Model {
        pub(crate) fn q_dim(&self) -> usize {
            self.heads * self.head_dim
        }

        pub(crate) fn scratch(&self) -> PrefillScratchLayout {
            PrefillScratchLayout::new(
                ROWS,
                self.hidden,
                self.q_dim(),
                self.kv_heads * self.head_dim,
                self.inter,
                Some(self.gdn.0),
                Some(self.gdn.2),
                true,
            )
        }

        /// The GEMMs of a fused Q+gate attention layer and a GDN layer, each
        /// with its FFN, every weight `weight` but the GDN alpha/beta pair,
        /// `alpha_beta` (passed an F16 cache when `ab_cache`).
        pub(crate) fn gemms(
            &self,
            weight: WeightClass,
            alpha_beta: WeightClass,
            ab_cache: bool,
        ) -> Vec<PrefillGemm> {
            let (h, q, kv, i) = (
                self.hidden,
                self.q_dim(),
                self.kv_heads * self.head_dim,
                self.inter,
            );
            let (qkv, heads, value) = self.gdn;
            let g = |out_dim, in_dim| PrefillGemm {
                weight,
                out_dim,
                in_dim,
                residual: false,
                gdn_alpha_beta: false,
                f16_cache: false,
            };
            let ab = PrefillGemm {
                weight: alpha_beta,
                gdn_alpha_beta: true,
                f16_cache: ab_cache,
                ..g(heads, h)
            };
            let mut gemms = vec![
                g(2 * q, h),
                g(kv, h),
                g(kv, h),
                PrefillGemm {
                    residual: true,
                    ..g(h, q)
                },
                g(qkv, h),
                g(value, h),
                ab,
                ab,
                PrefillGemm {
                    residual: true,
                    ..g(h, value)
                },
            ];
            if !self.moe {
                for _ in 0..2 {
                    gemms.extend([g(i, h), g(i, h), g(h, i)]);
                }
            }
            gemms
        }

        /// The workspace layout of this model with conversion scratch
        /// `conversion` and the fused GDN path.
        pub(crate) fn layout(&self, conversion: ConversionScratch) -> PrefillWorkspaceLayout {
            PrefillWorkspaceLayout::of(
                self.scratch(),
                Some(GdnPrefillScratchLayout::new(
                    ROWS, self.gdn.0, self.gdn.1, self.gdn.2,
                )),
                attn_score_block_elems(ROWS, self.heads, self.kv_heads, CONTEXT - ROWS),
                // 256 experts, 8 routed per token, the W10 gate+up on.
                self.moe.then(|| super::super::moe::PrefillGroupedLayout {
                    w10: true,
                    ..super::super::moe::PrefillGroupedLayout::new(
                        ROWS,
                        8,
                        256,
                        self.hidden,
                        self.inter,
                    )
                }),
                PrefillRoutes {
                    plain_attention: false,
                    gdn_fused: true,
                    dense_ffn: !self.moe,
                    moe_ffn: self.moe,
                },
                conversion,
            )
        }
    }

    /// The default switches of a dense model: projections through HGEMM, the
    /// Q8_0 split layout releasing the raw planes.
    pub(crate) fn dense_switches() -> GemmSwitches {
        GemmSwitches {
            mmq_kernel: true,
            mmq_residual_kernel: true,
            q8_split: true,
            ..GemmSwitches::default()
        }
    }

    /// The default switches of a MoE model: Q8_0 projections through MMQ,
    /// BF16 weights past the F16 caches, `bf16` alpha/beta through F32.
    pub(crate) fn moe_switches(bf16: bool) -> GemmSwitches {
        GemmSwitches {
            q8_proj_mmq: true,
            gdn_ab_f32: bf16,
            moe_bf16_native: true,
            mmq_kernel: true,
            mmq_residual_kernel: true,
            ..GemmSwitches::default()
        }
    }
}

/// Allocate the workspace `layout` describes: one arena with its values'
/// regions. The score block is part of the selected route, so failing to
/// allocate it fails the load.
pub(crate) fn alloc_prefill_workspace(
    device: &CudaDevice,
    layout: &PrefillWorkspaceLayout,
) -> Result<Workspace, RuntimeError> {
    Workspace::alloc(device, layout.regions())
}

/// Tokens per prefill slice. A prompt longer than this runs through the
/// layers in slices, each through one workspace sized for the slice, so the
/// workspace is bounded by this and not by the prompt: on the 27B Q8_0 model
/// it takes 668,991,488 bytes with a 32768-token context. The tiled
/// attention's score block, which shares the workspace, grows with the
/// context (`group × min(slice, 512) × keys` F32: ~384 MiB at 32k keys on the
/// 27B). Prompts up to this length run as one slice, exactly as before.
pub(crate) const PREFILL_SLICE_TOKENS: usize = 2048;

/// Element count of the tiled prefill attention's score block for one
/// prefill: `group × min(batch, ATTN_PREFILL_SGEMM_ROWS) × (pos_start +
/// batch)` F32. The block is reused across every KV head and every layer, so
/// this is the path's whole footprint. `None` when the geometry cannot host
/// the path (no KV heads, or a query-head count that is not a multiple of
/// them -- `launch_flash_attention_sgemm` rejects both) or when the product
/// overflows `usize`.
pub(crate) fn attn_score_block_elems(
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    pos_start: usize,
) -> Option<usize> {
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return None;
    }
    let group = num_heads / num_kv_heads;
    let rows_max = batch.min(ATTN_PREFILL_SGEMM_ROWS);
    let kv_total = pos_start.checked_add(batch)?;
    group.checked_mul(rows_max)?.checked_mul(kv_total)
}

/// Decode a planar plane into the f32 dequant scratch (`dequant_nvfp4_to_f32` / `dequant_fp8_to_f32`).
///
/// The kernels take the plane and the element count, deriving every internal offset from n — including the
/// F32 scale after the weight planes — so nothing else is passed. The NVFP4 kernel writes four elements per
/// thread and the FP8 kernel one, so the grid covers `n / 4` or `n` threads accordingly.
///
/// # Safety
///
/// `plane` must hold `n_elements` values of `quant`'s scheme (its planes, then the F32 scale), decoded
/// by `func`, and `out` must have `n_elements` elements.
unsafe fn launch_dequant_plane_to_f32(
    device: &CudaDevice,
    func: &CudaFunction,
    quant: QuantScheme,
    plane: &CudaSlice<u8>,
    out: &mut CudaViewMut<'_, f32>,
    n_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let block = 256u32;
    let n_u32 = n_elements as u32;
    let threads = match quant {
        QuantScheme::Nvfp4 => n_u32 / 4,
        _ => n_u32,
    };
    let grid = threads.div_ceil(block);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = device.stream.launch_builder(func);
    b.arg(plane).arg(out).arg(&n_u32);
    b.launch(cfg)
        .map_err(|e| RuntimeError::Compute(format!("{label} planar prefill dequant: {e}")))?;
    Ok(())
}

/// Decode a planar (NVFP4 or FP8) weight of `num_elements` into the f32 dequant scratch, the operand the
/// prefill SGEMM reads in place of a resident F16 image. Refuses by name when the scheme's decode kernel
/// is not loaded or the scratch is too small.
///
/// # Safety
///
/// A planar `weight` must hold `num_elements` values of its scheme.
unsafe fn dequant_planar_into_scratch(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    dequant_scratch: &mut CudaViewMut<'_, f32>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let (quant, plane, decode) = match weight {
        GpuWeightBuf::Nvfp4Raw(plane) => (
            QuantScheme::Nvfp4,
            plane,
            kernels.planar.nvfp4_prefill.as_ref(),
        ),
        GpuWeightBuf::Fp8Raw(plane) => (
            QuantScheme::Fp8E4M3,
            plane,
            kernels.planar.fp8_prefill.as_ref(),
        ),
        _ => unreachable!("only a planar weight is decoded into the dequant scratch"),
    };
    let decode = decode.ok_or_else(|| {
        RuntimeError::Compute(format!(
            "{quant:?} {label}: prefill decode kernel is not loaded, so prefill cannot serve it"
        ))
    })?;
    if dequant_scratch.len() < num_elements {
        return Err(RuntimeError::Compute(format!(
            "sgemm {label}: dequant scratch too small: have {} elements, need {num_elements}",
            dequant_scratch.len(),
        )));
    }
    launch_dequant_plane_to_f32(
        device,
        decode,
        quant,
        plane,
        dequant_scratch,
        num_elements,
        label,
    )
}

/// Batch embed tokens into [batch, hidden_dim] on GPU.
///
/// # Safety
///
/// All token IDs must be < vocab_size. `token_ids_gpu` must have `batch` elements.
/// `output` must have `batch * hidden_dim` elements. Embedding buffers must be valid.
pub(crate) unsafe fn launch_embed_batch(
    device: &CudaDevice,
    kernels: &KernelSet,
    embedding_f32: &CudaSlice<f32>,
    embedding_q8: Option<&CudaSlice<u8>>,
    embedding_f16: Option<&CudaSlice<u8>>,
    embedding_bf16: Option<&CudaSlice<u8>>,
    embedding_q4: Option<&CudaSlice<u8>>,
    embedding_kquant: Option<(QuantScheme, &CudaSlice<u8>)>,
    token_ids_gpu: &CudaView<'_, u32>,
    output: &mut CudaViewMut<'_, f32>,
    batch: usize,
    hidden_dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * hidden_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_u32 = batch as u32;
    let hd = hidden_dim as u32;

    // Dispatch priority: BF16 > F16 > K-quant (Q4_K/Q5_K/Q6_K) > Q4_0 > Q8_0 > F32 (same order as
    // embed_token_gpu)
    if let Some(emb_bf16) = embedding_bf16 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_bf16)
            .arg(emb_bf16)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_bf16 launch: {e}")))?;
    } else if let Some(emb_f16) = embedding_f16 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_f16)
            .arg(emb_f16)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_f16 launch: {e}")))?;
    } else if let Some((scheme, emb_kq)) = embedding_kquant {
        if !scheme.is_kquant_superblock() {
            return Err(RuntimeError::Compute(format!(
                "embed_batch: {scheme:?} is not a K-quant embedding scheme"
            )));
        }
        let group = super::decode::kquant_kernels(kernels, scheme);
        let tag = super::decode::kquant_tag(scheme);
        let group = group.ok_or_else(|| {
            RuntimeError::Compute(format!("embed_batch_{tag}: kernels not loaded"))
        })?;
        device
            .stream
            .launch_builder(&group.embed_batch)
            .arg(emb_kq)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_{tag} launch: {e}")))?;
        static SEEN: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        super::decode::announce_route_once(&SEEN, || {
            format!("[CUDA] embed_batch_{tag}: ACTIVE (token_embd batch, in={hidden_dim})")
        });
    } else if let Some(emb_q4) = embedding_q4 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_q4_0)
            .arg(emb_q4)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_q4_0 launch: {e}")))?;
    } else if let Some(emb_q8) = embedding_q8 {
        device
            .stream
            .launch_builder(&kernels.embed_batch_q8_0)
            .arg(emb_q8)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_q8_0 launch: {e}")))?;
    } else {
        device
            .stream
            .launch_builder(&kernels.embed_batch_f32)
            .arg(embedding_f32)
            .arg(token_ids_gpu)
            .arg(output)
            .arg(&batch_u32)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| RuntimeError::Compute(format!("embed_batch_f32 launch: {e}")))?;
    }
    Ok(())
}

/// Batched RMSNorm: normalize each row of [batch, dim] independently.
///
/// # Safety
///
/// `x` and `out` must have `batch * dim` elements. `weight` must have `dim` elements.
pub(crate) unsafe fn launch_rmsnorm_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &CudaView<'_, f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaViewMut<'_, f32>,
    eps: f32,
    batch: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let block_size = rmsnorm_block_size(dim);
    let shared_bytes = rmsnorm_shared_bytes(block_size);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.rmsnorm_batched)
        .arg(x)
        .arg(weight)
        .arg(out)
        .arg(&eps)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("rmsnorm_batched launch: {e}")))?;
    Ok(())
}

/// MoE BF16 fidelity routing: is this projection a native-BF16 weight on a MoE
/// model that should bypass the F16-cache fast path?
///
/// Empirical (2026-06-09, GQ-001 MoE-bf16 precision sweep): the F16-cache path
/// (`launch_cublas_hgemm`: F16 weights + `FAST_16F` reduced-precision
/// accumulation) makes the MoE BF16 model emit WRONG arithmetic — e.g. it reads
/// the input "1000 - 37" as "100 - 37" = 63, and degenerates into repetition —
/// whereas the native BF16 arm (`launch_cublas_gemm_bf16`: BF16 weights + full
/// `CUBLAS_COMPUTE_32F` accumulation) yields the correct answer (963). Routing
/// MoE BF16 to the native arm moves GQ-001 from 13/15 FAIL to 14/15 PASS (the
/// residual is a cosmetic near-tie intermediate slip inherent to the 7-bit BF16
/// mantissa; final answers are correct). AND-gated on `model_is_moe()` so DENSE
/// BF16 models keep the F16-cache fast path BYTE-IDENTICAL. The env override
/// `LUMEN_CUDA_MOE_BF16_NATIVE=0` restores the F16-cache path for MoE BF16 too.
fn moe_bf16_native_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("LUMEN_CUDA_MOE_BF16_NATIVE").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE")
        )
    })
}

fn moe_bf16_native_path(weight: &GpuWeightBuf) -> bool {
    matches!(weight, GpuWeightBuf::Bf16Raw(_))
        && crate::runtime_defaults::model_is_moe()
        && moe_bf16_native_enabled()
}

/// `LUMEN_CUDA_PREFILL_F32` present: the prefill GEMMs take their F32 SGEMM
/// fallback where the weight has one.
fn prefill_f32_forced() -> bool {
    std::env::var("LUMEN_CUDA_PREFILL_F32").is_ok()
}

/// `LUMEN_CUDA_Q8_PROJ_MMQ` present, whatever its value: a raw Q8_0 weight
/// prefers MMQ over an F16 cache, and the residual GEMM takes MMQ.
fn q8_proj_mmq_set() -> bool {
    std::env::var("LUMEN_CUDA_Q8_PROJ_MMQ").is_ok()
}

/// Whether the GDN alpha/beta projections take the F32 SGEMM, from the value
/// of `LUMEN_CUDA_GDN_AB_F32` (`0`/`false` off, any other value on; unset: on
/// for a MoE BF16 model).
fn gdn_ab_f32_on(env: Option<&str>) -> bool {
    match env {
        Some("0") | Some("false") | Some("FALSE") => false,
        Some(_) => true,
        None => crate::runtime_defaults::model_is_moe_bf16(),
    }
}

/// A prefill GEMM's weight as [`launch_gemm_projection`] and
/// [`launch_gemm_residual`] dispatch on it: the variant that picks the arm,
/// and with it the conversion scratch the arm takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightClass {
    F32,
    F16Raw,
    Bf16Raw,
    Q8Raw,
    Q8Split,
    Q4Raw,
    KQuant,
    Ct4Raw,
    /// NVFP4 or FP8 planes.
    Planar,
    /// `Q8Aligned` (per-row matvec) or `Q4Aligned` (refused): no scratch.
    Aligned,
}

impl WeightClass {
    pub(crate) fn of(weight: &GpuWeightBuf) -> Self {
        match weight {
            GpuWeightBuf::F32(_) => Self::F32,
            GpuWeightBuf::F16Raw(_) => Self::F16Raw,
            GpuWeightBuf::Bf16Raw(_) => Self::Bf16Raw,
            GpuWeightBuf::Q8Raw(_) => Self::Q8Raw,
            GpuWeightBuf::Q8Split(_) => Self::Q8Split,
            GpuWeightBuf::Q4Raw(_) => Self::Q4Raw,
            GpuWeightBuf::Q4KRaw(_) | GpuWeightBuf::Q5KRaw(_) | GpuWeightBuf::Q6KRaw(_) => {
                Self::KQuant
            }
            GpuWeightBuf::Ct4Raw(_) => Self::Ct4Raw,
            GpuWeightBuf::Nvfp4Raw(_) | GpuWeightBuf::Fp8Raw(_) => Self::Planar,
            GpuWeightBuf::Q8Aligned(_) | GpuWeightBuf::Q4Aligned(_) => Self::Aligned,
        }
    }
}

/// One GEMM a prefill slice runs, as far as the conversion scratch its route
/// takes depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillGemm {
    pub weight: WeightClass,
    pub out_dim: usize,
    pub in_dim: usize,
    /// Run by [`launch_gemm_residual`], not [`launch_gemm_projection`].
    pub residual: bool,
    /// The `gdn_alpha` / `gdn_beta` projection.
    pub gdn_alpha_beta: bool,
    /// Passed an F16 cache the load builds.
    pub f16_cache: bool,
}

/// The switches the prefill GEMMs' routes read, read as the GEMMs read them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GemmSwitches {
    /// [`prefill_f32_forced`].
    pub force_f32: bool,
    /// [`q8_proj_mmq_set`].
    pub q8_proj_mmq_set: bool,
    /// `q8_proj_mmq_enabled()`: the projection GEMM's MMQ route.
    pub q8_proj_mmq: bool,
    /// [`gdn_ab_f32_on`].
    pub gdn_ab_f32: bool,
    /// A BF16 weight bypasses an F16 cache ([`moe_bf16_native_path`]).
    pub moe_bf16_native: bool,
    pub mmq_kernel: bool,
    pub mmq_residual_kernel: bool,
    /// A raw Q8_0 plane may be resident only in the split layout by the time a
    /// request runs (`q8_split_release_raw_enabled()`).
    pub q8_split: bool,
}

impl GemmSwitches {
    pub(crate) fn read(kernels: &KernelSet) -> Self {
        Self {
            force_f32: prefill_f32_forced(),
            q8_proj_mmq_set: q8_proj_mmq_set(),
            q8_proj_mmq: crate::runtime_defaults::q8_proj_mmq_enabled(),
            gdn_ab_f32: gdn_ab_f32_on(std::env::var("LUMEN_CUDA_GDN_AB_F32").ok().as_deref()),
            moe_bf16_native: crate::runtime_defaults::model_is_moe() && moe_bf16_native_enabled(),
            mmq_kernel: kernels.mmq_q8_0_batched.is_some(),
            mmq_residual_kernel: kernels.mmq_q8_0_batched_residual.is_some(),
            q8_split: crate::runtime_defaults::q8_split_release_raw_enabled(),
        }
    }
}

/// Lengths of the conversion scratch regions (`dequant_f32` in F32
/// elements, the others in bytes); `None` for one no GEMM's route takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionScratch {
    pub dequant_f32: Option<usize>,
    pub activation_f16: Option<usize>,
    pub dequant_f16: Option<usize>,
}

impl ConversionScratch {
    /// The scratch `gemms` take over `rows` rows under `switches`: each region
    /// is the largest need, capped at `bound`, of the GEMMs whose route can
    /// reach an arm that takes it. `bound` is the scratch sized to the model's
    /// largest weight and widest row, so every length comparison a GEMM makes
    /// on a region decides as it does on `bound`: a need within `bound` is
    /// within the region, and a need over `bound` is over the region, which
    /// never exceeds `bound`. The arms follow [`launch_gemm_projection`] and
    /// [`launch_gemm_residual`]; an arm counts whenever a GEMM can reach it:
    /// an F16-cache arm and the arm the GEMM takes without the cache, and a
    /// raw Q8_0 weight's split-layout arm when its raw copy can be released.
    pub(crate) fn plan(
        gemms: &[PrefillGemm],
        switches: &GemmSwitches,
        rows: usize,
        bound: &PrefillScratchLayout,
    ) -> Self {
        let mut scratch = Self::default();
        for gemm in gemms {
            scratch.add(gemm, gemm.weight, switches, rows, bound);
            if gemm.weight == WeightClass::Q8Raw && switches.q8_split {
                scratch.add(gemm, WeightClass::Q8Split, switches, rows, bound);
            }
        }
        scratch
    }

    fn add(
        &mut self,
        gemm: &PrefillGemm,
        weight: WeightClass,
        sw: &GemmSwitches,
        rows: usize,
        bound: &PrefillScratchLayout,
    ) {
        fn grow(region: &mut Option<usize>, need: usize, bound: usize) {
            *region = Some(region.unwrap_or(0).max(need.min(bound)));
        }
        let (out_dim, in_dim) = (gemm.out_dim, gemm.in_dim);
        let dequant_f16_need = 2 * out_dim * in_dim;
        let dequant_f16_short = dequant_f16_need > bound.dequant_f16;
        let dequant_f32 =
            |s: &mut Self| grow(&mut s.dequant_f32, out_dim * in_dim, bound.dequant_f32);
        let activation = |s: &mut Self| {
            grow(
                &mut s.activation_f16,
                rows * in_dim * 2,
                bound.activation_f16,
            )
        };
        let f16_pair = |s: &mut Self| {
            grow(&mut s.dequant_f16, dequant_f16_need, bound.dequant_f16);
            activation(s);
        };
        let q8raw = weight == WeightClass::Q8Raw;
        let mmq_kernel = if gemm.residual {
            sw.mmq_residual_kernel
        } else {
            sw.mmq_kernel
        };
        let prefer_mmq_over_f16cache =
            sw.q8_proj_mmq_set && q8raw && in_dim % 32 == 0 && mmq_kernel;
        let force_alpha_beta_f32 = !gemm.residual && sw.gdn_ab_f32 && gemm.gdn_alpha_beta && q8raw;
        let moe_bf16_native = weight == WeightClass::Bf16Raw && sw.moe_bf16_native;
        if gemm.f16_cache
            && !sw.force_f32
            && !prefer_mmq_over_f16cache
            && !force_alpha_beta_f32
            && !moe_bf16_native
        {
            activation(self);
        }
        match weight {
            WeightClass::Planar => dequant_f32(self),
            WeightClass::F32 | WeightClass::Aligned => {}
            WeightClass::F16Raw | WeightClass::Bf16Raw => activation(self),
            WeightClass::Q8Raw => {
                let use_mmq = if gemm.residual {
                    sw.q8_proj_mmq_set
                } else {
                    sw.q8_proj_mmq && !force_alpha_beta_f32
                } && mmq_kernel
                    && in_dim % 32 == 0;
                if use_mmq {
                    // MMQ reads the raw plane and takes no scratch.
                } else if sw.force_f32 || force_alpha_beta_f32 || dequant_f16_short {
                    dequant_f32(self);
                } else {
                    f16_pair(self);
                }
            }
            WeightClass::Q4Raw | WeightClass::Q8Split => {
                if sw.force_f32 || dequant_f16_short {
                    dequant_f32(self);
                } else {
                    f16_pair(self);
                }
            }
            WeightClass::KQuant | WeightClass::Ct4Raw => {
                if !(sw.force_f32 || dequant_f16_short) {
                    f16_pair(self);
                }
            }
        }
    }
}

/// One line per process for the Q8_0 dequant -> F16 -> HGEMM prefill route on a raw AoS plane.
fn announce_q8_hgemm_once() {
    static Q8_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if super::decode::cuda_verbose()
        && !Q8_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        eprintln!("[CUDA] Prefill Q8_0 HGEMM: ACTIVE (dequant->F16->tensor core path)");
    }
}

/// The same route on a plane resident only in the split layout (raw copy released after the
/// split clone): its own line, so a route receipt shows which dequant fed the HGEMM.
fn announce_q8_split_hgemm_once() {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if super::decode::cuda_verbose() && !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[CUDA] Prefill Q8_0 split-layout HGEMM: ACTIVE (dequant split->F16->tensor core path)"
        );
    }
}

/// Batched GEMM projection: out = input * weight^T.
///
/// For F32 weights, uses cuBLAS SGEMM directly. For Q8_0 weights with a
/// pre-dequanted F16 cache, uses cublasGemmEx HGEMM (tensor cores). For Q8_0
/// without F16 cache, dequantizes to the F32 scratch buffer then calls cuBLAS
/// SGEMM. For native F16 weights (`F16Raw`), uses cublasGemmEx HGEMM directly
/// (no dequant needed -- weights are already F16). For Q4_0, falls back to
/// per-row matvec. For a K-quant plane, dequantizes the whole plane to the F16
/// scratch with the scheme's kernel, then HGEMM.
///
/// cuBLAS column-major mapping for row-major data:
/// Row-major W[out_dim, in_dim] = col-major W_cm[in_dim, out_dim]
/// Row-major A[batch, in_dim] = col-major A_cm[in_dim, batch]
/// Row-major C[batch, out_dim] = col-major C_cm[out_dim, batch]
/// C_cm = W_cm^T * A_cm
/// cublasSgemm(T, N, out_dim, batch, in_dim, 1.0, W, in_dim, A, in_dim, 0.0, C, out_dim)
///
/// # Safety
///
/// `input` must be [batch, in_dim]. `weight` must be [out_dim, in_dim].
/// `output` must be [batch, out_dim]. `conversion` lends the conversion
/// scratch the weight's route takes: an `ActivationF16` region the route
/// borrows must hold `batch * in_dim` 16-bit elements (the conversion does
/// not check); the dequant regions' lengths are checked.
pub(crate) unsafe fn launch_gemm_projection<const N: usize>(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    input: &CudaView<'_, f32>,
    output: &mut CudaViewMut<'_, f32>,
    conversion: &mut Rest<'_, N>,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // Validate buffer sizes before launching GPU work.
    let input_needed = batch * in_dim;
    let output_needed = batch * out_dim;
    if input.len() < input_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm {label}: input buffer too small: have {} elements, \
             need {} (batch={batch}, in_dim={in_dim})",
            input.len(),
            input_needed,
        )));
    }
    if output.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm {label}: output buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            output.len(),
            output_needed,
        )));
    }

    // diagnostic: when LUMEN_CUDA_PREFILL_F32=1, bypass the HGEMM-F16
    // tensor-core fast path AND the F16-cache Q8/Q4 paths; force the SGEMM-F32
    // fallback paths in the `match weight` block. Used to test whether HGEMM-F16
    // rounding accounts for the L0 drift on the Q8 projection.
    let force_f32 = prefill_f32_forced();

    // when LUMEN_CUDA_Q8_PROJ_MMQ=1, also bypass the F16-cache
    // fast path so that Q8Raw weights are routed to mmq_q8_0_batched (MMQ-style
    // INT8xINT8->INT32->F32-scale dp4a math). The F16 cache would otherwise
    // short-circuit before the match arm.
    let q8_proj_mmq = q8_proj_mmq_set();
    let weight_is_q8raw = matches!(weight, GpuWeightBuf::Q8Raw(_));
    let prefer_mmq_over_f16cache =
        q8_proj_mmq && weight_is_q8raw && in_dim % 32 == 0 && kernels.mmq_q8_0_batched.is_some();

    // `ssm_alpha` and `ssm_beta` weights are stored as F32 in the
    // GGUF source and an F32 SGEMM dispatch is the canonical reference path.
    // Default conversions force-requantize them to Q8_0 at LBC creation
    // time (see `crates/lumen-convert/src/arch/gdn_gates.rs`; a K-quant source's
    // default non-Metal conversion at the extent the projection reads,
    // source-fidelity, HF-import, and `--dequantize` non-Metal artifacts keep
    // F32), so the default runtime path runs them through HGEMM-F16 /
    // MMQ-Q8 — both of which introduce ~0.4-3.4% per-element rounding
    // noise from the requant step. When the `gdn_alpha`/`gdn_beta`
    // projection is routed through F32 SGEMM, it matches that canonical F32
    // path (modulo SGEMM accumulator order).
    //
    // Default-ON for MoE BF16 (`model_is_moe_bf16()`): empirically this F32
    // alpha/beta path is what kills the BF16 GQ-001 arith-05 repetition that the
    // `moe_bf16_native` main-projection routing alone leaves — taking MoE bf16 to
    // 14/15 PASS. It is gated to BF16 models because the same lever REGRESSES MoE
    // q8 (adds a DD-REP). Env override: `LUMEN_CUDA_GDN_AB_F32=0` forces OFF, any
    // other value forces ON.
    let gdn_ab_f32_env = std::env::var("LUMEN_CUDA_GDN_AB_F32").ok();
    let gdn_ab_f32 =
        gdn_ab_f32_on(gdn_ab_f32_env.as_deref()) && (label == "gdn_alpha" || label == "gdn_beta");
    let force_alpha_beta_f32 = gdn_ab_f32 && weight_is_q8raw;

    // Fast path: HGEMM with pre-dequanted F16 weights (tensor core, 312 TFLOPS on A100).
    // Converts F32 activations to F16 on the fly, uses cublasGemmEx with F16 inputs
    // and F32 compute/accumulate for numerical stability.
    let f16_cache_active = !force_f32
        && !prefer_mmq_over_f16cache
        && !force_alpha_beta_f32
        && !moe_bf16_native_path(weight)
        && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        let [mut activation] = conversion.regions([Role::ActivationF16]);
        let activation_f16 = &mut activation.u8();
        static HGEMM_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if super::decode::cuda_verbose()
            && !HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("[CUDA] Prefill HGEMM: ACTIVE (tensor core path)");
        }

        // Step 1: Convert F32 activation to F16 via vectorized kernel (4 elems/thread).
        launch_f32_to_f16_fast(
            device,
            kernels,
            input,
            activation_f16,
            batch * in_dim,
            label,
        )?;

        // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
        launch_cublas_hgemm(
            device,
            &w_f16.as_view(),
            &activation_f16.as_view(),
            output,
            out_dim,
            batch,
            in_dim,
            0.0,
            label,
        )?;
        return Ok(());
    }

    match weight {
        // The planar schemes prefill through the shared F32 SGEMM: decode the plane into the dequant
        // scratch, run the GEMM, discard the scratch. No resident F16 image, because an F16 copy takes
        // several times the planes' bytes and a model sized for the planes cannot hold both. This is the
        // route the Q8_0 fallback already takes, so prefill keeps one GEMM rather than gaining a second.
        GpuWeightBuf::Nvfp4Raw(_) | GpuWeightBuf::Fp8Raw(_) => {
            let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
            let dequant_scratch = &mut dequant_f32.f32();
            dequant_planar_into_scratch(
                device,
                kernels,
                weight,
                dequant_scratch,
                out_dim * in_dim,
                label,
            )?;
            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 0.0f32,
                ldc: out_dim as i32,
            };
            device
                .blas
                .gemm(cfg, &*dequant_scratch, input, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!("cuBLAS SGEMM (planar dequant) {label}: {e}"))
                })?;
            return Ok(());
        }
        GpuWeightBuf::F32(w_f32) => {
            let weight_needed = out_dim * in_dim;
            if w_f32.len() < weight_needed {
                return Err(RuntimeError::Compute(format!(
                    "sgemm {label}: weight buffer too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    w_f32.len(),
                    weight_needed,
                )));
            }
            // cuBLAS SGEMM: C_cm[out_dim, batch] = W_cm^T[out_dim, in_dim] * A_cm[in_dim, batch]
            // transa = T (transpose W_cm to get [out_dim, in_dim])
            // transb = N (A_cm is already [in_dim, batch])
            // m = out_dim, n = batch, k = in_dim
            // lda = in_dim (leading dim of W_cm[in_dim, out_dim])
            // ldb = in_dim (leading dim of A_cm[in_dim, batch])
            // ldc = out_dim (leading dim of C_cm[out_dim, batch])
            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 0.0f32,
                ldc: out_dim as i32,
            };
            device
                .blas
                .gemm(cfg, w_f32, input, output)
                .map_err(|e| RuntimeError::Compute(format!("cuBLAS SGEMM {label}: {e}")))?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // when LUMEN_CUDA_Q8_PROJ_MMQ=1, route Q8 projection
            // through the MMQ INT8 dp4a math (INT8xINT8->INT32->F32-scale).
            // This is the path required to close the qkv_pre_conv 5.85e-2
            // max-abs drift that HGEMM-F16 and SGEMM-F32 paths could not close.
            // Default OFF preserves byte-identical behaviour vs main; ENV-ON
            // routes to mmq_q8_0_batched (modulo dp4a/MMA microarchitecture
            // differences).
            // when LUMEN_CUDA_GDN_AB_F32=1, bypass MMQ
            // for `gdn_alpha` / `gdn_beta` projections specifically. This
            // forces the Q8 -> F32-dequant -> SGEMM-F32 path, restoring the
            // F32 SGEMM data path for these tensors (the GGUF source weight
            // is F32).
            // MoE models DEFAULT to the MMQ (INT8 dp4a) Q8 projection path.
            // llama.cpp computes Q8 matmuls in INT8 dp4a; lumen's default
            // dequant->F16->HGEMM path drifts ~5.85e-2 max-abs from that, which
            // is harmless for dense models but the 256-expert top-K router
            // AMPLIFIES it into flipped expert selection -> degenerate math on
            // Qwen3.5-MoE-35B-A3B (q8 writes "17x20=140" instead of 340). MMQ
            // restores llama-matching INT8 numerics so the products are correct.
            // Dense q8 keeps the faster HGEMM path (no router to amplify drift).
            // Env `LUMEN_CUDA_Q8_PROJ_MMQ=0|1` overrides the per-model default.
            let mmq_enabled = crate::runtime_defaults::q8_proj_mmq_enabled();
            let use_mmq = mmq_enabled
                && !force_alpha_beta_f32
                && kernels.mmq_q8_0_batched.is_some()
                && in_dim % 32 == 0;
            if use_mmq {
                static MMQ_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if super::decode::cuda_verbose()
                    && !MMQ_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!(
                        "[CUDA]: Prefill Q8 MMQ: ACTIVE ({label}, batch={batch}, \
                         out_dim={out_dim}, in_dim={in_dim})"
                    );
                }
                launch_mmq_q8_0_batched(
                    device, kernels, w_q8, input, output, out_dim, in_dim, batch, label,
                )?;
                return Ok(());
            }
            // Dequantize Q8_0 weights to F16 scratch, then cuBLAS HGEMM (tensor cores).
            //
            // This replaces the old F32 SGEMM path (19.5 TFLOPS) with:
            // 1. One dequant_q8_0_to_f16 kernel (num_elements threads)
            // 2. One f32_to_f16 conversion of activations
            // 3. One cublasGemmEx HGEMM call (312 TFLOPS on A100)
            //
            // Critical for GDN layers where F16 weight caches are skipped (OOM concern)
            // but HGEMM tensor cores provide 16x higher throughput than SGEMM.
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: when LUMEN_CUDA_PREFILL_F32 is set, take the
            // SGEMM-F32 fallback even if the F16 scratch is large enough. This
            // tests whether HGEMM-F16 rounding accounts for the L0 drift on
            // the Q8 projection (qkv_pre_conv) path.
            // also force F32 SGEMM for `gdn_alpha`/`gdn_beta` to
            // restore the F32-GGUF path for these projections.
            if force_f32
                || force_alpha_beta_f32
                || conversion.len(Role::DequantF16) < f16_bytes_needed
            {
                if force_alpha_beta_f32 {
                    static AB_F32_LOGGED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if (gdn_ab_f32_env.is_some() || super::decode::cuda_verbose())
                        && !AB_F32_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
                    {
                        eprintln!(
                            "[CUDA]: GDN alpha/beta forced through F32 SGEMM \
                             (F32 SGEMM on dequantized GDN alpha/beta; bypasses MMQ/HGEMM)"
                        );
                    }
                }
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q8_0_to_f32(
                    device,
                    kernels,
                    w_q8,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 0.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM fallback (dequant Q8_0) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                announce_q8_hgemm_once();

                // Step 1: Dequantize Q8_0 -> F16 in scratch buffer.
                launch_dequant_q8_0_to_f16(
                    device,
                    kernels,
                    w_q8,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    0.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            let [mut activation] = conversion.regions([Role::ActivationF16]);
            let activation_f16 = &mut activation.u8();
            // Native F16 weights: cublasGemmEx HGEMM directly (no dequant needed).
            // Convert F32 activations to F16, then HGEMM with F16 weights.
            // note: when LUMEN_CUDA_PREFILL_F32=1, this branch CANNOT
            // take an F32 fallback because the source weight is already F16.
            if force_f32 {
                static F16_FORCED_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !F16_FORCED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: LUMEN_CUDA_PREFILL_F32=1 but weight is F16Raw \
                         ({label}); HGEMM-F16 path still taken (weight class is F16)."
                    );
                }
            }
            static F16_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if super::decode::cuda_verbose()
                && !F16_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!("[CUDA] Prefill HGEMM F16Raw: ACTIVE (native F16 tensor core path)");
            }

            // Step 1: Convert F32 activation to F16 via vectorized kernel.
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 2: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
            launch_cublas_hgemm(
                device,
                &w_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            // Dequantize Q4_0 weights to F16 scratch, then cuBLAS HGEMM (tensor cores).
            //
            // Same pattern as Q8Raw HGEMM: replaces `batch` sequential matvec launches with:
            // 1. One dequant_q4_0_to_f16 kernel (num_elements threads)
            // 2. One f32_to_f16 conversion of activations
            // 3. One cublasGemmEx HGEMM call (312 TFLOPS on A100)
            //
            // Critical for GDN layers where F16 caches are skipped to save GPU memory.
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q4_0_to_f32(
                    device,
                    kernels,
                    w_q4,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 0.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM fallback (dequant Q4_0) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                static Q4_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if super::decode::cuda_verbose()
                    && !Q4_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!("[CUDA] Prefill Q4_0 HGEMM: ACTIVE (dequant->F16->tensor core path)");
                }

                // Step 1: Dequantize Q4_0 -> F16 in scratch buffer.
                launch_dequant_q4_0_to_f16(
                    device,
                    kernels,
                    w_q4,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 2: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 3: cublasGemmEx HGEMM (F16 weight + F16 activation -> F32 output).
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    0.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::Q4KRaw(_) | GpuWeightBuf::Q5KRaw(_) | GpuWeightBuf::Q6KRaw(_) => {
            // K-quant prefill: dequantize the whole plane to the F16 scratch
            // with the scheme's kernel, then HGEMM — the Ct4Raw pattern below.
            let (scheme, w_kq) =
                super::gpu_buffers::kquant_weight(weight).expect("matched a K-quant variant");
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: {scheme:?} requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    conversion.len(Role::DequantF16),
                )));
            }
            let [mut weight_f16, mut activation] =
                conversion.regions([Role::DequantF16, Role::ActivationF16]);
            let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
            launch_dequant_kquant_to_f16(
                device,
                kernels,
                scheme,
                w_kq,
                dequant_f16,
                num_elements,
                label,
            )?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                &dequant_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        GpuWeightBuf::Ct4Raw(w_ct4) => {
            // CtInt4G32 prefill: dequantize to F16 scratch, then HGEMM —
            // the same pattern as the Q4Raw arm above. No F32 fallback
            // exists for this format (its dequant kernel emits F16 only).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: CtInt4G32 requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    conversion.len(Role::DequantF16),
                )));
            }
            let [mut weight_f16, mut activation] =
                conversion.regions([Role::DequantF16, Role::ActivationF16]);
            let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
            launch_dequant_ct4_to_f16(device, kernels, w_ct4, dequant_f16, num_elements, label)?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                &dequant_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill: fall back to per-row matvec
            // (aligned format is optimized for single-token decode, not GEMM).
            for row in 0..batch {
                let in_offset = row * in_dim;
                let out_offset = row * out_dim;
                launch_matvec_slice(
                    device, kernels, weight, input, output, in_offset, out_offset, out_dim, in_dim,
                    label,
                )?;
            }
        }
        GpuWeightBuf::Q4Aligned(_) => {
            // Q4Aligned should not appear in prefill -- aligned repack is skipped
            // for GDN models, and non-GDN models use F16 HGEMM (routed above).
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight in batched prefill GEMM {label} -- unexpected"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // BF16 prefill: cuBLAS `cublasGemmEx` with CUDA_R_16BF inputs and
            // F32 output (CUBLAS_COMPUTE_32F). Tensor-core BF16 mma.sync path
            // on SM_80+ (A100 312 TFLOPS). Replaces the previous per-row
            // matvec_bf16 fallback which was ~42x slower than llama.cpp on
            // batched prefill.
            //
            // BF16 has the same 2 B/elem footprint as F16 — we reuse the F16
            // activation scratch buffer (`activation_f16`) for the BF16
            // activation conversion. The buffer holds bytes (`u8`), so the
            // interpretation is determined by the cuBLAS `cudaDataType_t`.
            let activation_needed = batch * in_dim * 2;
            if conversion.len(Role::ActivationF16) < activation_needed {
                return Err(RuntimeError::Compute(format!(
                    "bf16_gemm {label}: activation scratch too small: have {} bytes, \
                     need {} (batch={batch}, in_dim={in_dim})",
                    conversion.len(Role::ActivationF16),
                    activation_needed,
                )));
            }
            static BF16_HGEMM_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if super::decode::cuda_verbose()
                && !BF16_HGEMM_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!("[CUDA] Prefill BF16 GemmEx: ACTIVE (native BF16 tensor core path)");
            }

            // Step 1: Convert F32 activation to BF16 via vectorized kernel.
            let [mut activation] = conversion.regions([Role::ActivationF16]);
            let activation_f16 = &mut activation.u8();
            launch_f32_to_bf16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 2: cublasGemmEx (BF16 weight + BF16 activation -> F32 output).
            launch_cublas_gemm_bf16(
                device,
                &w_bf16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                0.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Split(w_split) => {
            // A Q8_0 plane resident only in the split layout (its raw copy was released
            // after the split clone): the same dequant -> F16 -> HGEMM route as `Q8Raw`,
            // reading the split layout, and the same F32 SGEMM fallback conditions (the
            // alpha/beta F32 override is `Q8Raw`-only, so it is not one of them here). The
            // MMQ route needs the AoS bytes, so a model that prefills Q8 through MMQ
            // keeps its raw planes and never reaches this arm.
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q8_split_to_f32(
                    device,
                    kernels,
                    w_split,
                    dequant_scratch,
                    num_elements,
                    in_dim,
                    label,
                )?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 0.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM fallback (dequant Q8_0 split) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                announce_q8_split_hgemm_once();
                launch_dequant_q8_split_to_f16(
                    device,
                    kernels,
                    w_split,
                    dequant_f16,
                    num_elements,
                    in_dim,
                    label,
                )?;
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    0.0,
                    label,
                )?;
            }
        }
    }
    Ok(())
}

/// Batched GEMM with fused residual: out = input * weight^T + residual.
///
/// For F32 weights, copies the residual into the output buffer first, then
/// calls cuBLAS SGEMM with beta=1.0 to accumulate C = alpha*A*B^T + beta*C.
/// For Q8_0 weights with F16 cache, uses cublasGemmEx HGEMM with beta=1.0.
/// For Q8_0 without F16 cache, dequantizes to F32 scratch then SGEMM.
/// For native F16 weights (`F16Raw`), uses cublasGemmEx HGEMM directly with
/// beta=1.0 (no dequant needed). For Q4_0, falls back to per-row matvec +
/// residual. For a K-quant plane, dequantizes to the F16 scratch, copies the
/// residual into `output`, then HGEMM with beta=1.0.
///
/// # Safety
///
/// `input` must be [batch, in_dim]. `weight` must be [out_dim, in_dim].
/// `residual` and `output` must be [batch, out_dim]. `conversion` lends the
/// conversion scratch as for [`launch_gemm_projection`].
pub(crate) unsafe fn launch_gemm_residual<const N: usize>(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    weight_f16_cache: Option<&CudaSlice<u8>>,
    input: &CudaView<'_, f32>,
    residual: &CudaView<'_, f32>,
    output: &mut CudaViewMut<'_, f32>,
    conversion: &mut Rest<'_, N>,
    batch: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // Validate buffer sizes before launching GPU work.
    let input_needed = batch * in_dim;
    let output_needed = batch * out_dim;
    if input.len() < input_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: input buffer too small: have {} elements, \
             need {} (batch={batch}, in_dim={in_dim})",
            input.len(),
            input_needed,
        )));
    }
    if residual.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: residual buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            residual.len(),
            output_needed,
        )));
    }
    if output.len() < output_needed {
        return Err(RuntimeError::Compute(format!(
            "sgemm_residual {label}: output buffer too small: have {} elements, \
             need {} (batch={batch}, out_dim={out_dim})",
            output.len(),
            output_needed,
        )));
    }

    // diagnostic: when LUMEN_CUDA_PREFILL_F32=1, bypass HGEMM-F16 paths
    // and use the SGEMM-F32 fallbacks. Used to test the HGEMM-F16-precision
    // hypothesis on the linear_attn_out (ssm_out residual) GEMM.
    let force_f32 = prefill_f32_forced();

    // same env gate as's projection MMQ (LUMEN_CUDA_Q8_PROJ_MMQ=1)
    // also covers the residual GEMM site (ssm_out). When the env is set AND the
    // weight is Q8Raw AND the kernel loaded, prefer the MMQ residual path over
    // the F16-cache fast path so the MMQ INT8 dp4a math is taken on the GDN-
    // block exit projection (`launch_gemm_residual` is called with
    // label="gdn_ssm_out" from backend_impl.rs:5012).
    //
    // closed `qkv_pre_conv` drift 7700x; closes the residual
    // `linear_attn_out` drift (~0.226 max-abs persists with alone) by
    // extending the same MMQ math to this second Q8 projection site.
    let q8_proj_mmq = q8_proj_mmq_set();
    let q8_residual_mmq = q8_proj_mmq;
    let weight_is_q8raw = matches!(weight, GpuWeightBuf::Q8Raw(_));
    let prefer_mmq_over_f16cache = q8_residual_mmq
        && weight_is_q8raw
        && in_dim % 32 == 0
        && kernels.mmq_q8_0_batched_residual.is_some();

    // diagnostic: log weight variant + gate decision on FIRST entry to
    // each label, to confirm MMQ residual dispatch actually fires on
    // `gdn_ssm_out` at L0 (not just `wo` at dense-attn layers). Strictly
    // diagnostic; remove or env-gate after release sign-off.
    if q8_proj_mmq {
        static GEMM_RES_TRACE: std::sync::Once = std::sync::Once::new();
        GEMM_RES_TRACE.call_once(|| {
            let variant = match weight {
                GpuWeightBuf::F32(_) => "F32",
                GpuWeightBuf::Q8Raw(_) => "Q8Raw",
                GpuWeightBuf::Q4Raw(_) => "Q4Raw",
                GpuWeightBuf::F16Raw(_) => "F16Raw",
                _ => "OTHER",
            };
            eprintln!(
                "[CUDA] trace[1st call]: launch_gemm_residual label={label} weight={variant} \
                 in_dim={in_dim} out_dim={out_dim} f16_cache={} prefer_mmq={} q8_residual_mmq={}",
                weight_f16_cache.is_some(),
                prefer_mmq_over_f16cache,
                q8_residual_mmq,
            );
        });
    }

    // Fast path: HGEMM residual with pre-dequanted F16 weights.
    // Copy residual to output first, then HGEMM with beta=1.0.
    let f16_cache_active = !force_f32
        && !prefer_mmq_over_f16cache
        && !moe_bf16_native_path(weight)
        && weight_f16_cache.is_some();
    if let Some(w_f16) = weight_f16_cache.filter(|_| f16_cache_active) {
        let [mut activation] = conversion.regions([Role::ActivationF16]);
        let activation_f16 = &mut activation.u8();
        // Copy residual -> output for beta=1.0 accumulation.
        device
            .stream
            .memcpy_dtod(
                &residual.slice(..output_needed),
                &mut output.slice_mut(..output_needed),
            )
            .map_err(|e| RuntimeError::Compute(format!("dtod residual copy {label}: {e}")))?;

        // Convert F32 activation to F16 via vectorized kernel.
        launch_f32_to_f16_fast(
            device,
            kernels,
            input,
            activation_f16,
            batch * in_dim,
            label,
        )?;

        // cublasGemmEx with beta=1.0 for residual accumulation.
        launch_cublas_hgemm(
            device,
            &w_f16.as_view(),
            &activation_f16.as_view(),
            output,
            out_dim,
            batch,
            in_dim,
            1.0,
            label,
        )?;
        return Ok(());
    }

    match weight {
        // The planar schemes prefill through the shared F32 SGEMM: decode the plane into the dequant
        // scratch, run the GEMM, discard the scratch. No resident F16 image, because an F16 copy takes
        // several times the planes' bytes and a model sized for the planes cannot hold both. This is the
        // route the Q8_0 fallback already takes, so prefill keeps one GEMM rather than gaining a second.
        GpuWeightBuf::Nvfp4Raw(_) | GpuWeightBuf::Fp8Raw(_) => {
            let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
            let dequant_scratch = &mut dequant_f32.f32();
            dequant_planar_into_scratch(
                device,
                kernels,
                weight,
                dequant_scratch,
                out_dim * in_dim,
                label,
            )?;
            // Copy residual -> output so SGEMM can accumulate with beta=1.0. Without this the caller
            // gets `W*x` where it asked for `W*x + y`: this function's contract, and every other arm
            // here, adds the residual.
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!(
                        "dtod residual copy (planar dequant) {label}: {e}"
                    ))
                })?;
            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 1.0f32,
                ldc: out_dim as i32,
            };
            device
                .blas
                .gemm(cfg, &*dequant_scratch, input, output)
                .map_err(|e| {
                    RuntimeError::Compute(format!("cuBLAS SGEMM (planar dequant) {label}: {e}"))
                })?;
            return Ok(());
        }
        GpuWeightBuf::F32(w_f32) => {
            let weight_needed = out_dim * in_dim;
            if w_f32.len() < weight_needed {
                return Err(RuntimeError::Compute(format!(
                    "sgemm_residual {label}: weight buffer too small: have {} elements, \
                     need {} (out_dim={out_dim}, in_dim={in_dim})",
                    w_f32.len(),
                    weight_needed,
                )));
            }
            // Copy residual -> output so SGEMM can accumulate with beta=1.0.
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy for {label}: {e}"))
                })?;

            let cfg = GemmConfig {
                transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: out_dim as i32,
                n: batch as i32,
                k: in_dim as i32,
                alpha: 1.0f32,
                lda: in_dim as i32,
                ldb: in_dim as i32,
                beta: 1.0f32,
                ldc: out_dim as i32,
            };
            device.blas.gemm(cfg, w_f32, input, output).map_err(|e| {
                RuntimeError::Compute(format!("cuBLAS SGEMM+residual {label}: {e}"))
            })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // when LUMEN_CUDA_Q8_PROJ_MMQ=1 (and the sub-gate to disable
            // ONLY this site is NOT set), route Q8 residual GEMM through the
            // MMQ kernel (INT8xINT8->INT32->F32-scale math) with fused residual
            // add. Default OFF preserves byte-identical behavior vs main;
            // ENV-ON closes the `linear_attn_out` ~0.226 max-abs drift that
            // survives's projection-only fix.
            let use_mmq =
                q8_residual_mmq && kernels.mmq_q8_0_batched_residual.is_some() && in_dim % 32 == 0;
            if use_mmq {
                static MMQ_RES_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if super::decode::cuda_verbose()
                    && !MMQ_RES_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!(
                        "[CUDA]: Prefill Q8 MMQ+residual: ACTIVE ({label}, batch={batch}, \
                         out_dim={out_dim}, in_dim={in_dim})"
                    );
                }
                launch_mmq_q8_0_batched_residual(
                    device, kernels, w_q8, input, residual, output, out_dim, in_dim, batch, label,
                )?;
                return Ok(());
            }
            // Dequantize Q8_0 -> F16, then cuBLAS HGEMM with fused residual (tensor cores).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q8_0_to_f32(
                    device,
                    kernels,
                    w_q8,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0 fallback) {label}: {e}"
                        ))
                    })?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 1.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM+residual fallback (dequant Q8_0) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                // Step 1: Copy residual -> output for beta=1.0 accumulation.
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0) {label}: {e}"
                        ))
                    })?;

                // Step 2: Dequantize Q8_0 -> F16 in scratch buffer.
                launch_dequant_q8_0_to_f16(
                    device,
                    kernels,
                    w_q8,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    1.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            let [mut activation] = conversion.regions([Role::ActivationF16]);
            let activation_f16 = &mut activation.u8();
            // Native F16 weights: cublasGemmEx HGEMM with residual (no dequant needed).
            // Copy residual -> output, convert F32 activation to F16, then HGEMM beta=1.0.
            // note: when LUMEN_CUDA_PREFILL_F32=1, this branch CANNOT
            // take an F32 fallback because the source weight is already F16.
            if force_f32 {
                static F16_FORCED_RES_LOGGED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !F16_FORCED_RES_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[CUDA]: LUMEN_CUDA_PREFILL_F32=1 but residual weight is F16Raw \
                         ({label}); HGEMM-F16 path still taken."
                    );
                }
            }

            // Step 1: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy F16Raw {label}: {e}"))
                })?;

            // Step 2: Convert F32 activation to F16 via vectorized kernel.
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 3: cublasGemmEx with beta=1.0 for residual accumulation.
            launch_cublas_hgemm(
                device,
                &w_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            // Dequantize Q4_0 -> F16, then cuBLAS HGEMM with fused residual (tensor cores).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            // diagnostic: force SGEMM-F32 fallback when LUMEN_CUDA_PREFILL_F32=1.
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                // Fallback to F32 SGEMM if dequant_f16 buffer is too small.
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q4_0_to_f32(
                    device,
                    kernels,
                    w_q4,
                    dequant_scratch,
                    num_elements,
                    label,
                )?;
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q4_0 fallback) {label}: {e}"
                        ))
                    })?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 1.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM+residual fallback (dequant Q4_0) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                // Step 1: Copy residual -> output for beta=1.0 accumulation.
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q4_0) {label}: {e}"
                        ))
                    })?;

                // Step 2: Dequantize Q4_0 -> F16 in scratch buffer.
                launch_dequant_q4_0_to_f16(
                    device,
                    kernels,
                    w_q4,
                    dequant_f16,
                    num_elements,
                    label,
                )?;

                // Step 3: Convert F32 activation to F16.
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;

                // Step 4: cublasGemmEx HGEMM with beta=1.0 for residual accumulation.
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    1.0,
                    label,
                )?;
            }
        }
        GpuWeightBuf::Q4KRaw(_) | GpuWeightBuf::Q5KRaw(_) | GpuWeightBuf::Q6KRaw(_) => {
            // K-quant prefill with residual: dequantize to the F16 scratch,
            // copy the residual into `output`, HGEMM with beta = 1.0.
            let (scheme, w_kq) =
                super::gpu_buffers::kquant_weight(weight).expect("matched a K-quant variant");
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: {scheme:?} requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    conversion.len(Role::DequantF16),
                )));
            }
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy ({scheme:?}) {label}: {e}"))
                })?;
            let [mut weight_f16, mut activation] =
                conversion.regions([Role::DequantF16, Role::ActivationF16]);
            let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
            launch_dequant_kquant_to_f16(
                device,
                kernels,
                scheme,
                w_kq,
                dequant_f16,
                num_elements,
                label,
            )?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                &dequant_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        GpuWeightBuf::Ct4Raw(w_ct4) => {
            // CtInt4G32 prefill: dequantize to F16 scratch, then HGEMM —
            // the same pattern as the Q4Raw arm above. No F32 fallback
            // exists for this format (its dequant kernel emits F16 only).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                return Err(RuntimeError::Compute(format!(
                    "gemm {label}: CtInt4G32 requires the F16 HGEMM path \
                     (dequant_f16 has {} bytes, need {f16_bytes_needed}; \
                     LUMEN_CUDA_PREFILL_F32 is unsupported for this format)",
                    conversion.len(Role::DequantF16),
                )));
            }
            // beta=1.0 accumulates into `output`, so it must hold the residual first.
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy (CtInt4G32) {label}: {e}"))
                })?;
            let [mut weight_f16, mut activation] =
                conversion.regions([Role::DequantF16, Role::ActivationF16]);
            let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
            launch_dequant_ct4_to_f16(device, kernels, w_ct4, dequant_f16, num_elements, label)?;
            launch_f32_to_f16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;
            launch_cublas_hgemm(
                device,
                &dequant_f16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Aligned(_) => {
            // Q8Aligned in batched prefill residual: fall back to per-row matvec.
            for row in 0..batch {
                let in_offset = row * in_dim;
                let res_offset = row * out_dim;
                let out_offset = row * out_dim;
                launch_matvec_residual_slice(
                    device, kernels, weight, input, residual, output, in_offset, res_offset,
                    out_offset, out_dim, in_dim, label,
                )?;
            }
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned weight in batched prefill GEMM residual {label} -- unexpected"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            // BF16 prefill + residual: cuBLAS `cublasGemmEx` with beta=1.0.
            // Copies residual -> output, converts F32 activation -> BF16, then
            // invokes the tensor-core BF16 GemmEx (CUDA_R_16BF, F32 acc) which
            // accumulates W * A + residual into output.
            // replaces the per-row matvec_bf16_residual fallback (the 42x gap).
            let activation_needed = batch * in_dim * 2;
            if conversion.len(Role::ActivationF16) < activation_needed {
                return Err(RuntimeError::Compute(format!(
                    "bf16_gemm_residual {label}: activation scratch too small: have {} bytes, \
                     need {} (batch={batch}, in_dim={in_dim})",
                    conversion.len(Role::ActivationF16),
                    activation_needed,
                )));
            }

            // Step 1: Copy residual -> output for beta=1.0 accumulation.
            device
                .stream
                .memcpy_dtod(
                    &residual.slice(..output_needed),
                    &mut output.slice_mut(..output_needed),
                )
                .map_err(|e| {
                    RuntimeError::Compute(format!("dtod residual copy Bf16Raw {label}: {e}"))
                })?;

            // Step 2: Convert F32 activation to BF16 via vectorized kernel.
            let [mut activation] = conversion.regions([Role::ActivationF16]);
            let activation_f16 = &mut activation.u8();
            launch_f32_to_bf16_fast(
                device,
                kernels,
                input,
                activation_f16,
                batch * in_dim,
                label,
            )?;

            // Step 3: cublasGemmEx BF16 with beta=1.0 for residual accumulation.
            launch_cublas_gemm_bf16(
                device,
                &w_bf16.as_view(),
                &activation_f16.as_view(),
                output,
                out_dim,
                batch,
                in_dim,
                1.0,
                label,
            )?;
        }
        GpuWeightBuf::Q8Split(w_split) => {
            // Split-layout-only Q8_0 plane (raw copy released): the `Q8Raw` residual
            // route on the split layout — residual copied into the output, then the
            // dequant -> F16 -> HGEMM accumulate (beta = 1), or the F32 SGEMM fallback
            // under the same conditions. MMQ models keep their raw planes (see the
            // projection arm).
            let num_elements = out_dim * in_dim;
            let f16_bytes_needed = num_elements * 2;
            if force_f32 || conversion.len(Role::DequantF16) < f16_bytes_needed {
                if conversion.len(Role::DequantF32) < num_elements {
                    return Err(RuntimeError::Compute(format!(
                        "sgemm_residual {label}: dequant scratch too small: have {} elements, \
                         need {} (out_dim={out_dim}, in_dim={in_dim})",
                        conversion.len(Role::DequantF32),
                        num_elements,
                    )));
                }
                let [mut dequant_f32] = conversion.regions([Role::DequantF32]);
                let dequant_scratch = &mut dequant_f32.f32();
                launch_dequant_q8_split_to_f32(
                    device,
                    kernels,
                    w_split,
                    dequant_scratch,
                    num_elements,
                    in_dim,
                    label,
                )?;
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0 split fallback) {label}: {e}"
                        ))
                    })?;
                let cfg = GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: out_dim as i32,
                    n: batch as i32,
                    k: in_dim as i32,
                    alpha: 1.0f32,
                    lda: in_dim as i32,
                    ldb: in_dim as i32,
                    beta: 1.0f32,
                    ldc: out_dim as i32,
                };
                device
                    .blas
                    .gemm(cfg, &*dequant_scratch, input, output)
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "cuBLAS SGEMM+residual fallback (dequant Q8_0 split) {label}: {e}"
                        ))
                    })?;
            } else {
                let [mut weight_f16, mut activation] =
                    conversion.regions([Role::DequantF16, Role::ActivationF16]);
                let (dequant_f16, activation_f16) = (&mut weight_f16.u8(), &mut activation.u8());
                device
                    .stream
                    .memcpy_dtod(
                        &residual.slice(..output_needed),
                        &mut output.slice_mut(..output_needed),
                    )
                    .map_err(|e| {
                        RuntimeError::Compute(format!(
                            "dtod residual copy (dequant Q8_0 split) {label}: {e}"
                        ))
                    })?;
                launch_dequant_q8_split_to_f16(
                    device,
                    kernels,
                    w_split,
                    dequant_f16,
                    num_elements,
                    in_dim,
                    label,
                )?;
                launch_f32_to_f16_fast(
                    device,
                    kernels,
                    input,
                    activation_f16,
                    batch * in_dim,
                    label,
                )?;
                launch_cublas_hgemm(
                    device,
                    &dequant_f16.as_view(),
                    &activation_f16.as_view(),
                    output,
                    out_dim,
                    batch,
                    in_dim,
                    1.0,
                    label,
                )?;
            }
        }
    }
    Ok(())
}

/// Launch batched RoPE for Q and K matrices.
///
/// # Safety
///
/// `q` must be [batch, q_dim]. `k` must be [batch, kv_dim].
pub(crate) unsafe fn launch_rope_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &mut CudaViewMut<'_, f32>,
    k: &mut CudaViewMut<'_, f32>,
    pos_start: usize,
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
    rope_neox: bool,
    rotary_dim: u32,
) -> Result<(), RuntimeError> {
    // For partial RoPE (rotary_dim > 0), only rotate first rotary_dim dims per head.
    let actual_rot = if rotary_dim > 0 && (rotary_dim as usize) < head_dim {
        rotary_dim as usize
    } else {
        head_dim
    };
    let half_rot = actual_rot / 2;
    let total_q_pairs = num_q_heads * half_rot;
    let total_work = batch * total_q_pairs;
    let config = LaunchConfig::for_elements(total_work);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let pos_start_u32 = pos_start as u32;
    let batch_u32 = batch as u32;
    let nqh = num_q_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;

    let rope_fn = if rope_neox {
        &kernels.rope_apply_batched_neox
    } else {
        &kernels.rope_apply_batched
    };
    device
        .stream
        .launch_builder(rope_fn)
        .arg(&mut *q)
        .arg(&mut *k)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nqh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&theta)
        .arg(&rotary_dim)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("rope_apply_batched launch: {e}")))?;
    Ok(())
}

/// Write a batch of K/V data to the GPU KV cache at positions pos_start..pos_start+batch-1.
///
/// # Safety
///
/// `data` must be [batch, num_kv_heads * head_dim]. Cache must have capacity.
pub(crate) unsafe fn launch_kv_cache_write_batch(
    device: &CudaDevice,
    kernels: &KernelSet,
    cache: &mut CudaSlice<f32>,
    data: &CudaView<'_, f32>,
    pos_start: usize,
    batch: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<(), RuntimeError> {
    let kv_dim = num_kv_heads * head_dim;
    let total = batch * kv_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let pos_start_u32 = pos_start as u32;
    let batch_u32 = batch as u32;
    let nkvh = num_kv_heads as u32;
    let msl = max_seq_len as u32;
    let hd = head_dim as u32;
    device
        .stream
        .launch_builder(&kernels.kv_cache_write_batch)
        .arg(cache)
        .arg(data)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nkvh)
        .arg(&msl)
        .arg(&hd)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("kv_cache_write_batch launch: {e}")))?;
    Ok(())
}

/// Batched SwiGLU: gate = silu(gate) * up for [batch, inter_dim].
///
/// # Safety
///
/// `gate` and `up` must have `batch * inter_dim` elements.
pub(crate) unsafe fn launch_swiglu_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    gate: &mut CudaViewMut<'_, f32>,
    up: &CudaView<'_, f32>,
    batch: usize,
    inter_dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * inter_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_u32 = total as u32;
    device
        .stream
        .launch_builder(&kernels.swiglu_batched)
        .arg(gate)
        .arg(up)
        .arg(&total_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("swiglu_batched launch: {e}")))?;
    Ok(())
}

/// Batched residual add: x += residual for [batch, dim].
///
/// # Safety
///
/// `x` and `residual` must have `batch * dim` elements.
pub(crate) unsafe fn launch_residual_add_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    x: &mut CudaViewMut<'_, f32>,
    residual: &CudaView<'_, f32>,
    batch: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let total = batch * dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_u32 = total as u32;
    device
        .stream
        .launch_builder(&kernels.residual_add_batched)
        .arg(x)
        .arg(residual)
        .arg(&total_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("residual_add_batched launch: {e}")))?;
    Ok(())
}

/// Extract a single row from a [batch, dim] matrix into a [dim] vector.
///
/// # Safety
///
/// `matrix` must have at least `(row_idx + 1) * dim` elements.
/// `output` must have `dim` elements.
pub(crate) unsafe fn launch_extract_row(
    device: &CudaDevice,
    kernels: &KernelSet,
    matrix: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    row_idx: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(dim);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_u32 = row_idx as u32;
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.extract_row)
        .arg(matrix)
        .arg(output)
        .arg(&row_u32)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("extract_row launch: {e}")))?;
    Ok(())
}

/// Write a [dim] vector into row `row_idx` of a [batch, dim] matrix on GPU.
///
/// # Safety
///
/// `matrix` must have at least `(row_idx + 1) * dim` elements.
/// `input` must have `dim` elements.
#[allow(dead_code)]
pub(crate) unsafe fn launch_scatter_row(
    device: &CudaDevice,
    kernels: &KernelSet,
    matrix: &mut CudaSlice<f32>,
    input: &CudaSlice<f32>,
    row_idx: usize,
    dim: usize,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(dim);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_u32 = row_idx as u32;
    let dim_u32 = dim as u32;
    device
        .stream
        .launch_builder(&kernels.scatter_row)
        .arg(matrix)
        .arg(input)
        .arg(&row_u32)
        .arg(&dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("scatter_row launch: {e}")))?;
    Ok(())
}

// Catches an `ALL` that has fallen behind the enum's last variant. It cannot
// catch a variant appended after that one — `latch`'s exhaustive match is what
// closes that case.

/// Launch the Q8_0 -> F32 dequantization kernel.
///
/// Dequantizes `num_elements` from the Q8_0 raw buffer into a contiguous F32 buffer.
/// Grid: ceil(num_elements / 256), Block: 256. Each thread dequantizes one element.
///
/// # Safety
///
/// `q8_data` must contain enough Q8_0 blocks for `num_elements` (i.e.,
/// ceil(num_elements / 32) * 34 bytes). `f32_out` must have `num_elements` elements.
/// launch the MMQ-style Q8_0 batched matmul kernel.
///
/// Computes `out[t, r] = sum_k(dequant(weight_q8[r, k]) * x[t, k])` via the
/// MMQ path: per-token Q8_1 activation quantization, INT8 dp4a inner products,
/// F32 scale at K-block granularity. INT32-exact intra-block sums, F32 scale
/// applied only at sum-time.
///
/// `in_dim` must be a multiple of 32 (Q8_0 block size).
///
/// # Safety
///
/// `weight_q8` must contain `out_dim * (in_dim/32) * 34` bytes of Q8_0 data.
/// `x` must have `batch * in_dim` F32 elements. `out` must have `batch *
/// out_dim` F32 elements.
pub(crate) unsafe fn launch_mmq_q8_0_batched(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight_q8: &CudaSlice<u8>,
    x: &CudaView<'_, f32>,
    out: &mut CudaViewMut<'_, f32>,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "mmq_q8_0_batched {label}: in_dim ({in_dim}) must be a multiple of 32"
        )));
    }
    let f = kernels.mmq_q8_0_batched.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!(
            "mmq_q8_0_batched {label}: kernel not loaded (requires SM 6.1+ for __dp4a)"
        ))
    })?;
    const NR: usize = 2;
    let grid_x = (out_dim + NR - 1) / NR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_x as u32, batch as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let batch_u32 = batch as u32;
    device
        .stream
        .launch_builder(f)
        .arg(weight_q8)
        .arg(x)
        .arg(out)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .arg(&batch_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("mmq_q8_0_batched {label}: {e}")))?;
    Ok(())
}

/// MMQ Q8_0 batched matmul WITH RESIDUAL ADD: `out = residual + W @ x`.
///
/// Same MMQ INT8 dp4a path as `launch_mmq_q8_0_batched`, except the kernel
/// fuses the residual add into the final store site. Used by
/// `launch_gemm_residual`'s Q8Raw path when `LUMEN_CUDA_Q8_PROJ_MMQ=1`, to
/// route the `ssm_out` (4096 -> 2048) GEMM through the MMQ INT8 path.
/// Closes the residual `linear_attn_out` drift (~0.226 max-abs) that survives
/// 's projection-only MMQ fix.
///
/// `in_dim` must be a multiple of 32 (Q8_0 block size).
///
/// # Safety
///
/// `weight_q8` must contain `out_dim * (in_dim/32) * 34` bytes of Q8_0 data.
/// `x` must have `batch * in_dim` F32 elements. `residual` and `out` must
/// have `batch * out_dim` F32 elements. `out` and `residual` may NOT alias
/// (the kernel reads `residual[idx]` and writes `out[idx]` in distinct
/// warps; aliasing would create a read-after-write hazard).
pub(crate) unsafe fn launch_mmq_q8_0_batched_residual(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight_q8: &CudaSlice<u8>,
    x: &CudaView<'_, f32>,
    residual: &CudaView<'_, f32>,
    out: &mut CudaViewMut<'_, f32>,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if in_dim % 32 != 0 {
        return Err(RuntimeError::Compute(format!(
            "mmq_q8_0_batched_residual {label}: in_dim ({in_dim}) must be a multiple of 32"
        )));
    }
    let f = kernels.mmq_q8_0_batched_residual.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!(
            "mmq_q8_0_batched_residual {label}: kernel not loaded \
             (requires SM 6.1+ for dp4a + the MMQ residual variant)"
        ))
    })?;
    const NR: usize = 2;
    let grid_x = (out_dim + NR - 1) / NR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (grid_x as u32, batch as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    let batch_u32 = batch as u32;
    device
        .stream
        .launch_builder(f)
        .arg(weight_q8)
        .arg(x)
        .arg(residual)
        .arg(out)
        .arg(&out_dim_u32)
        .arg(&in_dim_u32)
        .arg(&batch_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("mmq_q8_0_batched_residual {label}: {e}")))?;
    Ok(())
}

unsafe fn launch_dequant_q8_0_to_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    q8_data: &CudaSlice<u8>,
    f32_out: &mut CudaViewMut<'_, f32>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_0_to_f32)
        .arg(q8_data)
        .arg(f32_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_0_to_f32 {label}: {e}")))?;
    Ok(())
}

/// Dequantize Q4_0 weights to F32 scratch buffer for cuBLAS SGEMM.
///
/// Each thread dequantizes one element: reads the block's F16 scale and the
/// element's 4-bit nibble, computes `scale * (nibble - 8)`.
///
/// # Safety
///
/// `q4_data` must contain valid Q4_0 blocks. `f32_out` must have at least
/// `num_elements` elements.
unsafe fn launch_dequant_q4_0_to_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_data: &CudaSlice<u8>,
    f32_out: &mut CudaViewMut<'_, f32>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q4_0_to_f32)
        .arg(q4_data)
        .arg(f32_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q4_0_to_f32 {label}: {e}")))?;
    Ok(())
}

/// Dequantize Q8_0 weights to F16 scratch buffer for cuBLAS HGEMM.
///
/// Each thread dequantizes one element from Q8_0 format to F16.
/// Enables the tensor core HGEMM path (312 TFLOPS on A100) for Q8_0 weights
/// that lack a persistent F16 cache (e.g., GDN layers).
///
/// # Safety
///
/// `q8_data` must contain valid Q8_0 blocks. `f16_out` must have at least
/// `num_elements * 2` bytes.
unsafe fn launch_dequant_q8_0_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    q8_data: &CudaSlice<u8>,
    f16_out: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_0_to_f16)
        .arg(q8_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_0_to_f16 {label}: {e}")))?;
    Ok(())
}

/// `dequant_q8_split_to_f32`: the F32 tile of a Q8_0 plane resident in the split layout.
unsafe fn launch_dequant_q8_split_to_f32(
    device: &CudaDevice,
    kernels: &KernelSet,
    split: &CudaSlice<u8>,
    f32_out: &mut CudaViewMut<'_, f32>,
    num_elements: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // 32 elements per 34-byte block, nb * 34 bytes per row: the kernel indexes the row base
    // from the element id and divides by `in_dim`, so a short plane reads out of bounds. `<`
    // rather than the K-quant sibling's `!=`: the plane is exact today (`repack_q8_raw_to_split`
    // allocates out_dim * nb * 34), and only shortness is unsafe.
    let expected_w = num_elements / 32 * 34;
    if in_dim == 0
        || in_dim % 32 != 0
        || num_elements % in_dim != 0
        || split.len() < expected_w
        || f32_out.len() < num_elements
        || u32::try_from(num_elements).is_err()
        || u32::try_from(in_dim).is_err()
    {
        return Err(RuntimeError::Compute(format!(
            "dequant_q8_split_to_f32 {label}: {num_elements} elements x {in_dim} do not fit the plane ({} bytes) \
             or the output, or exceed the kernel's u32 extents",
            split.len()
        )));
    }
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    let in_dim_u32 = in_dim as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_split_to_f32)
        .arg(split)
        .arg(f32_out)
        .arg(&n)
        .arg(&in_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_split_to_f32 {label}: {e}")))?;
    Ok(())
}

/// `dequant_q8_split_to_f16`: the F16 tile of a Q8_0 plane resident in the split layout
/// (bit-identical to `dequant_q8_0_to_f16` on the raw plane).
unsafe fn launch_dequant_q8_split_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    split: &CudaSlice<u8>,
    f16_out: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // 32 elements per 34-byte block, nb * 34 bytes per row: the kernel indexes the row base
    // from the element id and divides by `in_dim`, so a short plane reads out of bounds. `<`
    // rather than the K-quant sibling's `!=`: the plane is exact today (`repack_q8_raw_to_split`
    // allocates out_dim * nb * 34), and only shortness is unsafe.
    let expected_w = num_elements / 32 * 34;
    if in_dim == 0
        || in_dim % 32 != 0
        || num_elements % in_dim != 0
        || split.len() < expected_w
        || f16_out.len() < num_elements * 2
        || u32::try_from(num_elements).is_err()
        || u32::try_from(in_dim).is_err()
    {
        return Err(RuntimeError::Compute(format!(
            "dequant_q8_split_to_f16 {label}: {num_elements} elements x {in_dim} do not fit the plane ({} bytes) \
             or the output, or exceed the kernel's u32 extents",
            split.len()
        )));
    }
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    let in_dim_u32 = in_dim as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q8_split_to_f16)
        .arg(split)
        .arg(f16_out)
        .arg(&n)
        .arg(&in_dim_u32)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q8_split_to_f16 {label}: {e}")))?;
    Ok(())
}

/// Dequantize Q4_0 weights to F16 scratch buffer for cuBLAS HGEMM.
///
/// Each thread dequantizes one element from Q4_0 format to F16: reads the
/// block's F16 scale and the element's 4-bit nibble, computes
/// `f16(scale * (nibble - 8))`.
///
/// # Safety
///
/// `q4_data` must contain valid Q4_0 blocks. `f16_out` must have at least
/// `num_elements * 2` bytes.
unsafe fn launch_dequant_q4_0_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    q4_data: &CudaSlice<u8>,
    f16_out: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&kernels.dequant_q4_0_to_f16)
        .arg(q4_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_q4_0_to_f16 {label}: {e}")))?;
    Ok(())
}

/// Dequantize a whole K-quant plane into the F16 scratch with the scheme's
/// `dequant_<tag>_to_f16` kernel (one thread per element). The kernel indexes
/// superblocks from the element id, so the byte length is checked first.
unsafe fn launch_dequant_kquant_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    scheme: QuantScheme,
    data: &CudaSlice<u8>,
    f16_out: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    if !scheme.is_kquant_superblock() {
        return Err(RuntimeError::Compute(format!(
            "dequant to F16 {label}: {scheme:?} is not a K-quant scheme"
        )));
    }
    let group = super::decode::kquant_kernels(kernels, scheme);
    let (block_bytes, tag) = super::decode::kquant_layout(scheme);
    let group = group.ok_or_else(|| {
        RuntimeError::Compute(format!("dequant_{tag}_to_f16 {label}: kernels not loaded"))
    })?;
    let expected_w = num_elements / 256 * block_bytes;
    let needed_out = num_elements * 2;
    if num_elements % 256 != 0
        || data.len() != expected_w
        || f16_out.len() < needed_out
        || u32::try_from(num_elements).is_err()
    {
        return Err(RuntimeError::Compute(format!(
            "dequant_{tag}_to_f16 {label}: shape mismatch: {num_elements} elements, \
             weight {} bytes (expected {expected_w}), out {} bytes (need {needed_out})",
            data.len(),
            f16_out.len(),
        )));
    }
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(&group.dequant_to_f16)
        .arg(data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_{tag}_to_f16 {label}: {e}")))?;
    Ok(())
}

/// Launch `dequant_ct4_to_f16`: CtInt4G32 decode blocks -> F16 image.
///
/// # Safety
///
/// Same contract as `launch_dequant_q4_0_to_f16`.
unsafe fn launch_dequant_ct4_to_f16(
    device: &CudaDevice,
    kernels: &KernelSet,
    ct4_data: &CudaSlice<u8>,
    f16_out: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let dequant_fn = kernels.dequant_ct4_to_f16.as_ref().ok_or_else(|| {
        RuntimeError::Compute(format!("dequant_ct4_to_f16 {label}: kernel unavailable"))
    })?;
    // 32 elements per 20-byte block; the kernel indexes blocks straight from
    // the element id, so a short weight buffer would read out of bounds.
    let expected_w = num_elements / 32 * 20;
    let needed_out = num_elements * 2;
    if num_elements % 32 != 0
        || ct4_data.len() != expected_w
        || f16_out.len() < needed_out
        || u32::try_from(num_elements).is_err()
    {
        return Err(RuntimeError::Compute(format!(
            "dequant_ct4_to_f16 {label}: shape mismatch: {num_elements} elements, \
             weight {} bytes (expected {expected_w}), out {} bytes (need {needed_out})",
            ct4_data.len(),
            f16_out.len(),
        )));
    }
    let config = LaunchConfig::for_elements(num_elements);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = num_elements as u32;
    device
        .stream
        .launch_builder(dequant_fn)
        .arg(ct4_data)
        .arg(f16_out)
        .arg(&n)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("dequant_ct4_to_f16 {label}: {e}")))?;
    Ok(())
}

/// Per-row matvec dispatch for non-F32 fallback path.
///
/// Dispatches the appropriate matvec kernel for a single row within a batched matrix,
/// using element offsets to address the correct row of input and output.
///
/// # Safety
///
/// Input and output slices must be large enough for the given offsets + dimensions.
unsafe fn launch_matvec_slice(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaView<'_, f32>,
    output: &mut CudaViewMut<'_, f32>,
    _in_offset: usize,
    _out_offset: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    // For the non-F32 fallback, we use try_slice to get views at the correct offsets.
    let in_view = input
        .try_slice(_in_offset.._in_offset + in_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_slice input slice out of bounds: offset={_in_offset} dim={in_dim}",
            ))
        })?;
    let mut out_view = output
        .try_slice_mut(_out_offset.._out_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_slice output slice out of bounds: offset={_out_offset} dim={out_dim}",
            ))
        })?;

    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::F32(_) => unreachable!("F32 uses cuBLAS SGEMM path"),
        GpuWeightBuf::Ct4Raw(_)
        | GpuWeightBuf::Q4KRaw(_)
        | GpuWeightBuf::Q5KRaw(_)
        | GpuWeightBuf::Q6KRaw(_)
        | GpuWeightBuf::Nvfp4Raw(_)
        | GpuWeightBuf::Fp8Raw(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_slice {label}: CtInt4G32, K-quant, NVFP4 and FP8 planes are served by the \
                 prefill GEMM paths, not the per-row fallback"
            )));
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_f16)
                .arg(w_f16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!("matvec F16 {label} prefill: {e}")))?;
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels
                .matvec_q8_0_aligned
                .as_ref()
                .or(kernels.matvec_q8_0_dp4a.as_ref())
                .unwrap_or(&kernels.matvec_q8_0);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(q8a_fn)
                .arg(w_q8a)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec Q8_0 aligned {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // Q8_0 prefill: dp4a -> v1 fallback.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8_fn = kernels
                .matvec_q8_0_dp4a
                .as_ref()
                .unwrap_or(&kernels.matvec_q8_0);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let shmem = 0u32;
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(q8_fn)
                .arg(w_q8)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| RuntimeError::Compute(format!("matvec Q8_0 {label} prefill: {e}")))?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0)
                .arg(w_q4)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!("matvec Q4_0 {label} prefill: {e}")))?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned in per-row matvec prefill {label} -- should route through F16 HGEMM"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_bf16)
                .arg(w_bf16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| RuntimeError::Compute(format!("matvec BF16 {label} prefill: {e}")))?;
        }
        // split-layout: the per-row prefill matvec does not serve the sibling (the
        // batched GEMM serves Q8Split)
        GpuWeightBuf::Q8Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_slice {label}: a split-layout plane reached the per-row prefill \
                 matvec; the batched prefill serves Q8Split through its dequant -> GEMM arm",
            )));
        }
    }
    Ok(())
}

/// Launch Flash Attention v2 (Br=1) for all tokens in a prefill batch.
///
/// Processes all query tokens against the KV cache in a SINGLE kernel launch
/// with causal masking. Each thread block handles one (head, token) pair.
///
/// Grid: (num_heads, batch, 1) -- one block per (head, query_token)
/// Block: (128, 1, 1) -- 128 threads (4 warps)
///
/// Replaces the sequential extract_row -> attention_decode -> scatter_row loop
/// that required 3 * batch kernel launches per layer.
///
/// # Arguments
///
/// * `q_batch` - Batched Q vectors `[batch, num_heads * head_dim]`
/// * `kv_cache` - GPU KV cache with data for positions 0..pos_start+batch-1
/// * `attn_out` - Output buffer `[batch, num_heads * head_dim]`
/// * `batch` - Number of query tokens
/// * `num_heads` - Number of Q attention heads
/// * `num_kv_heads` - Number of KV attention heads (for GQA)
/// * `head_dim` - Dimension per attention head
/// * `pos_start` - Position of first query token in the sequence
///
/// # Safety
///
/// * `q_batch` must have `batch * num_heads * head_dim` elements
/// * `attn_out` must have `batch * num_heads * head_dim` elements
/// * KV cache must contain valid data for `pos_start + batch` positions
#[allow(dead_code)]
pub(crate) unsafe fn launch_flash_attention_v2(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaSlice<f32>,
    kv: &KvView<'_>,
    attn_out: &mut CudaSlice<f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{flash_attention_v2_block_size, flash_attention_v2_shared_bytes};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_v2: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(),
            needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_v2: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(),
            needed,
        )));
    }

    let block_size = flash_attention_v2_block_size();
    let shared_bytes = flash_attention_v2_shared_bytes(head_dim as u32);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, batch as u32, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv.seq_stride as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_v2)
        .arg(q_batch)
        .arg(kv.k)
        .arg(kv.v)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_v2 launch: {e}")))?;

    Ok(())
}

/// Query rows per block of the tiled SGEMM prefill attention: bounds the score
/// scratch to `group * 512 * kv_len` floats.
pub(crate) const ATTN_PREFILL_SGEMM_ROWS: usize = 512;

/// Threads per block of `attn_softmax_causal_rows`. The kernel folds the row
/// max and the row sum across exactly `SMX_THREADS / 32` warps in a fixed
/// order and sizes its shared `part[]` from that count, so this value and
/// `SMX_THREADS` in `attn_softmax_causal.cu` must agree; the kernel carries a
/// `static_assert` on the warp count.
pub(crate) const ATTN_SOFTMAX_CAUSAL_THREADS: u32 = 128;

/// Query rows of the block starting at `qb` in a `batch`-row prefill: the
/// block size the tiled SGEMM attention walks with, `ATTN_PREFILL_SGEMM_ROWS`
/// except for the last (partial) block. `qb < batch` is the caller's loop
/// condition; `qb >= batch` yields 0, which would not terminate the walk.
pub(crate) fn attn_sgemm_block_rows(batch: usize, qb: usize) -> usize {
    batch.saturating_sub(qb).min(ATTN_PREFILL_SGEMM_ROWS)
}

/// Tiled prefill attention in exact F32: for each block of query rows and
/// each KV head, S = Q·Kᵀ over that head's query group by one strided-batched
/// SGEMM (the same K for every head of the group), a causal row softmax, then
/// O = P·V by a second strided-batched SGEMM. Same inputs and outputs as
/// `launch_flash_attention_br4`.
///
/// # Exact F32
///
/// Both products run as `cublasSgemmStridedBatched` with F32 operands, F32
/// scalars and an F32 accumulator, on the handle's default math mode
/// (`CUBLAS_DEFAULT_MATH`, which cuBLAS documents as using "compute and
/// intermediate storage precisions with at least the same number of mantissa
/// and exponent bits as requested"). TF32 carries ten mantissa bits against
/// F32's twenty-three, so it is outside what that mode may select for an F32
/// request; cuBLAS documents TF32 acceleration of single-precision routines as
/// what the `CUBLAS_TF32_TENSOR_OP_MATH` math mode enables. Nothing in the
/// engine calls `cublasSetMathMode`, and the only documented environment lever,
/// `NVIDIA_TF32_OVERRIDE=0`, can only take TF32 away. **Calling
/// `cublasSetMathMode` with `CUBLAS_TF32_TENSOR_OP_MATH` (or building this
/// through `cublasGemmEx` with a `_FAST_TF32` compute type) would silently
/// break the exact-F32 contract these two calls stand on, and with it the
/// exact-F32 prefill attention.**
///
/// # Evaluation order
///
/// The arithmetic is exact F32 but not the scalar kernel's order: the softmax
/// row is normalised before P·V and reduced over the whole row, where the
/// scalar kernel accumulates unnormalised weighted values online and
/// normalises afterwards. Greedy output can therefore differ from the scalar
/// kernel where two candidates sit within rounding of each other, and cuBLAS
/// guarantees bit-wise reproducibility only within a toolkit version on a
/// given architecture and SM count.
///
/// Dispatched on the fused Q+gate prefill path only (`attn_q_norm` present —
/// every artifact today's converter produces: qwen35 / qwen35moe). A layer
/// without per-head q/k norms takes the plain prefill dispatch, which runs the
/// scalar kernel.
///
/// `scores` is the block `alloc_prefill_workspace` sized for this prefill; it
/// is never allocated here (see the check below for why).
///
/// Both GEMMs batch over the query group with `stride_a = 0`, which points
/// every instance at the one K (or V) that group shares. cuBLAS's reference
/// for `cublas<t>gemmStridedBatched` states the requirement on the batch
/// offsets as "The unit for the offset is number of elements and must not be
/// zero", while NVIDIA's own strided-batched example broadcasts an input with
/// a zero stride; the sentence reads as a constraint on the output offsets,
/// which the same page spells out ("Matrices C[i] should not overlap;
/// otherwise, undefined behavior is expected") and which this call honours:
/// `stride_c` is `rows * s_ld` for Q·Kᵀ and `head_dim` for P·V, so no two
/// instances write the same element. A and B are read-only, so their
/// instances may alias. Measured correct on CUDA 13.1 / sm_120 (md5-identical
/// greedy output against the scalar kernel). If a future cuBLAS rejects a
/// zero input stride, the replacement is one GEMM per group head — `group`
/// times the launches per block and KV head, no extra memory.
///
/// # Safety
///
/// `q` holds `batch * num_heads * head_dim` floats, `attn_out` at least as
/// many, and the KV cache holds positions `0..pos_start + batch`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_flash_attention_sgemm(
    device: &CudaDevice,
    kernels: &KernelSet,
    q: &CudaView<'_, f32>,
    kv: &KvView<'_>,
    attn_out: &mut CudaViewMut<'_, f32>,
    scores: Option<&mut CudaViewMut<'_, f32>>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use cudarc::cublas::StridedBatchedConfig;
    // Buffer sizes first, before any allocation or launch (as in
    // `launch_flash_attention_br4`).
    let needed_io = batch * num_heads * head_dim;
    if q.len() < needed_io {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_sgemm: q too small: have {} elements, \
             need {needed_io} (batch={batch}, q_dim={})",
            q.len(),
            num_heads * head_dim,
        )));
    }
    if attn_out.len() < needed_io {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_sgemm: attn_out too small: have {} elements, \
             need {needed_io} (batch={batch}, q_dim={})",
            attn_out.len(),
            num_heads * head_dim,
        )));
    }
    let softmax_fn = kernels.attn_softmax_causal.as_ref().ok_or_else(|| {
        RuntimeError::Compute("attention prefill sgemm: softmax kernel unavailable".into())
    })?;
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: {num_heads} query heads over {num_kv_heads} KV heads"
        )));
    }
    let group = num_heads / num_kv_heads;
    let q_dim = num_heads * head_dim;
    let max_seq = kv.seq_stride;
    let kv_total = pos_start + batch;
    if kv_total > max_seq {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: {kv_total} positions exceed the KV cache's {max_seq}"
        )));
    }
    let needed =
        attn_score_block_elems(batch, num_heads, num_kv_heads, pos_start).ok_or_else(|| {
            RuntimeError::Compute("attention prefill sgemm: score block size overflows".into())
        })?;
    // Never allocated here. The layer loop writes each layer's KV and
    // advances that cache's length BEFORE it reaches this dispatch, while the
    // host-side KV length only advances after the last layer; an allocation
    // that failed inside the loop would abort the prefill with the two out of
    // step. `preload_weights` therefore reserves the block for the whole
    // context, and the dispatcher only routes here when it is present. This check is the guard on that contract, not a fallback.
    let have_block = match scores.as_ref() {
        Some(s) => s.len() >= needed,
        None => false,
    };
    if !have_block {
        return Err(RuntimeError::Compute(format!(
            "attention prefill sgemm: no score block for batch {batch} at position \
             {pos_start} ({needed} floats needed); the block is sized before the \
             layer loop and this route runs only when it exists"
        )));
    }
    let s_buf = scores.expect("score block presence checked immediately above");
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut qb = 0usize;
    while qb < batch {
        let rows = attn_sgemm_block_rows(batch, qb);
        let kv_len = pos_start + qb + rows; // causal bound for this block
        let s_ld = kv_len;
        for kv_h in 0..num_kv_heads {
            let h0 = kv_h * group;
            let k_view = kv.k.slice(kv_h * max_seq * head_dim..);
            let v_view = kv.v.slice(kv_h * max_seq * head_dim..);
            let q_view = q.slice(qb * q_dim + h0 * head_dim..);
            // S[g][i][j] = sum_d Q[i][h0+g][d] * K[j][d]: in cuBLAS's column-major
            // terms C(kv_len x rows) = K(head_dim x kv_len)^T * Q(head_dim x rows).
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: kv_len as i32,
                    n: rows as i32,
                    k: head_dim as i32,
                    alpha: 1.0f32,
                    lda: head_dim as i32,
                    ldb: q_dim as i32,
                    beta: 0.0f32,
                    ldc: s_ld as i32,
                },
                batch_size: group as i32,
                stride_a: 0,
                stride_b: head_dim as i64,
                stride_c: (rows * s_ld) as i64,
            };
            device
                .blas
                .gemm_strided_batched(cfg, &k_view, &q_view, s_buf)
                .map_err(|e| RuntimeError::Compute(format!("attention prefill sgemm QK: {e}")))?;
            let rows_u32 = rows as u32;
            let kv_u32 = kv_len as u32;
            let ld_u32 = s_ld as u32;
            let p0 = (pos_start + qb) as u32;
            device
                .stream
                .launch_builder(softmax_fn)
                .arg(&mut *s_buf)
                .arg(&rows_u32)
                .arg(&kv_u32)
                .arg(&ld_u32)
                .arg(&p0)
                .arg(&scale)
                .launch(CudarcLaunchConfig {
                    grid_dim: (rows as u32, group as u32, 1),
                    block_dim: (ATTN_SOFTMAX_CAUSAL_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| RuntimeError::Compute(format!("attn_softmax_causal_rows: {e}")))?;
            // O[i][h0+g][d] = sum_j P[g][i][j] * V[j][d]: column-major
            // C(head_dim x rows) = V(head_dim x kv_len) * P(kv_len x rows).
            let mut o_view = attn_out.slice_mut(qb * q_dim + h0 * head_dim..);
            let cfg = StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    m: head_dim as i32,
                    n: rows as i32,
                    k: kv_len as i32,
                    alpha: 1.0f32,
                    lda: head_dim as i32,
                    ldb: s_ld as i32,
                    beta: 0.0f32,
                    ldc: q_dim as i32,
                },
                batch_size: group as i32,
                stride_a: 0,
                stride_b: (rows * s_ld) as i64,
                stride_c: head_dim as i64,
            };
            device
                .blas
                .gemm_strided_batched(cfg, &v_view, &*s_buf, &mut o_view)
                .map_err(|e| RuntimeError::Compute(format!("attention prefill sgemm PV: {e}")))?;
        }
        qb += rows;
    }
    Ok(())
}

/// Launch Flash Attention Br=4 for all tokens in a prefill batch.
///
/// Processes 4 query tokens per thread block using warp-level parallelism.
/// Each of the 4 warps independently handles one query row, avoiding
/// block-level syncs between queries for higher throughput.
///
/// Grid: (num_heads, ceil(batch / 4), 1)
/// Block: (128, 1, 1) -- 4 warps of 32 threads
///
/// Preferred over `launch_flash_attention_v2` when batch >= 4, as it
/// processes 4x more queries per block with the same thread count.
///
/// # Arguments
///
/// Same as `launch_flash_attention_v2`.
///
/// # Safety
///
/// Same as `launch_flash_attention_v2`.
pub(crate) unsafe fn launch_flash_attention_br4(
    device: &CudaDevice,
    kernels: &KernelSet,
    q_batch: &CudaView<'_, f32>,
    kv: &KvView<'_>,
    attn_out: &mut CudaViewMut<'_, f32>,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos_start: usize,
) -> Result<(), RuntimeError> {
    use super::decode::{flash_attention_br4_block_size, flash_attention_br4_shared_bytes, FA_BR};

    let q_dim = num_heads * head_dim;
    let needed = batch * q_dim;
    if q_batch.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: q_batch too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            q_batch.len(),
            needed,
        )));
    }
    if attn_out.len() < needed {
        return Err(RuntimeError::Compute(format!(
            "flash_attention_br4: attn_out too small: have {} elements, \
             need {} (batch={batch}, q_dim={q_dim})",
            attn_out.len(),
            needed,
        )));
    }

    let block_size = flash_attention_br4_block_size();
    let shared_bytes = flash_attention_br4_shared_bytes(head_dim as u32);
    let q_tiles = (batch as u32 + FA_BR - 1) / FA_BR;
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (num_heads as u32, q_tiles, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let batch_u32 = batch as u32;
    let nh = num_heads as u32;
    let nkvh = num_kv_heads as u32;
    let hd = head_dim as u32;
    let ps = pos_start as u32;
    let msl = kv.seq_stride as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    device
        .stream
        .launch_builder(&kernels.flash_attention_br4)
        .arg(q_batch)
        .arg(kv.k)
        .arg(kv.v)
        .arg(attn_out)
        .arg(&batch_u32)
        .arg(&nh)
        .arg(&nkvh)
        .arg(&hd)
        .arg(&ps)
        .arg(&msl)
        .arg(&scale)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("flash_attention_br4 launch: {e}")))?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Per-row matvec + residual dispatch for non-F32 fallback path.
///
/// # Safety
///
/// Same constraints as `launch_matvec_slice`, plus residual slice validity.
unsafe fn launch_matvec_residual_slice(
    device: &CudaDevice,
    kernels: &KernelSet,
    weight: &GpuWeightBuf,
    input: &CudaView<'_, f32>,
    residual: &CudaView<'_, f32>,
    output: &mut CudaViewMut<'_, f32>,
    in_offset: usize,
    res_offset: usize,
    out_offset: usize,
    out_dim: usize,
    in_dim: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let in_view = input
        .try_slice(in_offset..in_offset + in_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res input slice out of bounds: offset={in_offset} dim={in_dim}",
            ))
        })?;
    let res_view = residual
        .try_slice(res_offset..res_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res residual slice out of bounds: offset={res_offset} dim={out_dim}",
            ))
        })?;
    let mut out_view = output
        .try_slice_mut(out_offset..out_offset + out_dim)
        .ok_or_else(|| {
            RuntimeError::Compute(format!(
                "matvec_res output slice out of bounds: offset={out_offset} dim={out_dim}",
            ))
        })?;

    let mv_block = matvec_block_size();
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (out_dim as u32, 1, 1),
        block_dim: (mv_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    match weight {
        GpuWeightBuf::F32(_) => unreachable!("F32 uses cuBLAS SGEMM path"),
        GpuWeightBuf::Ct4Raw(_)
        | GpuWeightBuf::Q4KRaw(_)
        | GpuWeightBuf::Q5KRaw(_)
        | GpuWeightBuf::Q6KRaw(_)
        | GpuWeightBuf::Nvfp4Raw(_)
        | GpuWeightBuf::Fp8Raw(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_res {label}: CtInt4G32, K-quant, NVFP4 and FP8 planes are served by the \
                 prefill GEMM paths, not the per-row fallback"
            )));
        }
        GpuWeightBuf::Q8Aligned(w_q8a) => {
            // Q8_0 aligned residual prefill: dp4a with native int* loads.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8a_fn = kernels
                .matvec_q8_0_aligned_residual
                .as_ref()
                .or(kernels.matvec_q8_0_dp4a_residual.as_ref())
                .unwrap_or(&kernels.matvec_q8_0_residual);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            device
                .stream
                .launch_builder(q8a_fn)
                .arg(w_q8a)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q8_0 aligned {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q8Raw(w_q8) => {
            // Q8_0 residual prefill: dp4a -> v1 fallback.
            use super::decode::{matvec_q8_0_grid, Q8_0_BLOCK_DIM};
            let q8_fn = kernels
                .matvec_q8_0_dp4a_residual
                .as_ref()
                .unwrap_or(&kernels.matvec_q8_0_residual);
            let q8_grid = matvec_q8_0_grid(out_dim as u32);
            let shmem = 0u32;
            let q8_launch = CudarcLaunchConfig {
                grid_dim: (q8_grid, 1, 1),
                block_dim: (Q8_0_BLOCK_DIM, 1, 1),
                shared_mem_bytes: shmem,
            };
            device
                .stream
                .launch_builder(q8_fn)
                .arg(w_q8)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(q8_launch)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q8_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::F16Raw(w_f16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_f16_residual)
                .arg(w_f16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&res_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res F16 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Raw(w_q4) => {
            device
                .stream
                .launch_builder(&kernels.matvec_q4_0_residual)
                .arg(w_q4)
                .arg(&in_view)
                .arg(&res_view)
                .arg(&mut out_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res Q4_0 {label} prefill: {e}"))
                })?;
        }
        GpuWeightBuf::Q4Aligned(_) => {
            return Err(RuntimeError::Compute(format!(
                "Q4Aligned in per-row matvec+res prefill {label} -- should route through F16 HGEMM"
            )));
        }
        GpuWeightBuf::Bf16Raw(w_bf16) => {
            device
                .stream
                .launch_builder(&kernels.matvec_bf16_residual)
                .arg(w_bf16)
                .arg(&in_view)
                .arg(&mut out_view)
                .arg(&res_view)
                .arg(&out_dim_u32)
                .arg(&in_dim_u32)
                .launch(launch_cfg)
                .map_err(|e| {
                    RuntimeError::Compute(format!("matvec+res BF16 {label} prefill: {e}"))
                })?;
        }
        // split-layout: the per-row prefill matvec does not serve the sibling (the
        // batched GEMM serves Q8Split)
        GpuWeightBuf::Q8Split(_) => {
            return Err(RuntimeError::Compute(format!(
                "matvec_res {label}: a split-layout plane reached the per-row prefill \
                 matvec; the batched prefill serves Q8Split through its dequant -> GEMM arm",
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Shared helpers: F32->F16 fast conversion and cuBLAS HGEMM wrapper.
// ============================================================================

/// Fast F32->F16 conversion using vectorized kernel (4 elems/thread) when available,
/// falling back to scalar kernel (1 elem/thread) otherwise.
///
/// # Safety
///
/// `src` must have at least `num_elements` F32 values. `dst` must have at least
/// `num_elements * 2` bytes (F16 = 2 bytes each).
pub(crate) unsafe fn launch_f32_to_f16_fast(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let n = num_elements as u32;

    // Prefer vectorized kernel (4 elems/thread): 4x fewer threads, coalesced loads.
    if let Some(ref vec4_fn) = kernels.f32_to_f16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!("f32_to_f16_vec4 {label}: {e}",)))?;
    } else {
        // Fallback: scalar kernel (still uses hardware PTX cvt.rn.f16.f32).
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_f16_vec)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!("f32_to_f16_vec {label}: {e}",)))?;
    }
    Ok(())
}

/// Wrapper for cublasGemmEx HGEMM: F16 weight + F16 activation -> F32 output.
///
/// Computes C = alpha * W^T * A + beta * C where W is F16 [out_dim, in_dim],
/// A is F16 [batch, in_dim], and C is F32 [batch, out_dim].
///
/// Uses `CUBLAS_COMPUTE_32F_FAST_16F` for maximum tensor core throughput (312 TFLOPS A100).
///
/// # Safety
///
/// `w_f16` must be [out_dim * in_dim] F16 elements. `a_f16` must be [batch * in_dim] F16 elements.
/// `output` must be [batch * out_dim] F32 elements.
unsafe fn launch_cublas_hgemm(
    device: &CudaDevice,
    w_f16: &CudaView<'_, u8>,
    a_f16: &CudaView<'_, u8>,
    output: &mut CudaViewMut<'_, f32>,
    out_dim: usize,
    batch: usize,
    in_dim: usize,
    beta_val: f32,
    label: &str,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = beta_val;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_f16.device_ptr(&device.stream);
    let (a_ptr, _) = a_f16.device_ptr(&device.stream);
    let (c_ptr, _) = output.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,
        batch as i32,
        in_dim as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16F,
        in_dim as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx HGEMM {label}: status={status:?}",
        )));
    }
    Ok(())
}

/// F32 -> BF16 bulk conversion. Prefers vectorized (4 elems/thread) variant;
/// falls back to scalar. Used to convert activations before
/// `launch_cublas_gemm_bf16`.
///
/// # Safety
///
/// `src` must have at least `num_elements` F32 values. `dst` must have at
/// least `num_elements * 2` bytes (BF16 = 2 bytes each).
pub(crate) unsafe fn launch_f32_to_bf16_fast(
    device: &CudaDevice,
    kernels: &super::decode::KernelSet,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, u8>,
    num_elements: usize,
    label: &str,
) -> Result<(), RuntimeError> {
    let n = num_elements as u32;

    if let Some(ref vec4_fn) = kernels.f32_to_bf16_vec4 {
        let block_size = 256u32;
        let elems_per_block = block_size * 4;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(vec4_fn)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!("f32_to_bf16_vec4 {label}: {e}",)))?;
    } else {
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = CudarcLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .stream
            .launch_builder(&kernels.f32_to_bf16_vec)
            .arg(src)
            .arg(dst)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| RuntimeError::Compute(format!("f32_to_bf16_vec {label}: {e}",)))?;
    }
    Ok(())
}

/// Wrapper for `cublasGemmEx` BF16: BF16 weight + BF16 activation -> F32 output.
///
/// Computes C = alpha * W^T * A + beta * C where W is BF16 [out_dim, in_dim],
/// A is BF16 [batch, in_dim], and C is F32 [batch, out_dim].
///
/// Uses `CUBLAS_COMPUTE_32F` with `CUDA_R_16BF` inputs. cuBLAS automatically
/// selects the tensor-core BF16 path on SM_80+ (A100 312 TFLOPS via mma.sync
/// bf16.bf16.f32). On older GPUs it falls back to software BF16; this path is
/// only invoked when the LBC header declares BF16 weights, which the runtime
/// already gates on Ampere+ capability via the matvec_bf16 SM_80 PTX path.
///
/// F32 accumulator preserves numerical equivalence with the per-row
/// matvec_bf16 fallback this replaces — both accumulate dot products in F32.
///
/// # Safety
///
/// `w_bf16` must be `out_dim * in_dim` BF16 elements (2 bytes each).
/// `a_bf16` must be `batch * in_dim` BF16 elements.
/// `output` must be `batch * out_dim` F32 elements.
pub(crate) unsafe fn launch_cublas_gemm_bf16(
    device: &CudaDevice,
    w_bf16: &CudaView<'_, u8>,
    a_bf16: &CudaView<'_, u8>,
    output: &mut CudaViewMut<'_, f32>,
    out_dim: usize,
    batch: usize,
    in_dim: usize,
    beta_val: f32,
    label: &str,
) -> Result<(), RuntimeError> {
    let alpha: f32 = 1.0;
    let beta: f32 = beta_val;

    use cudarc::driver::DevicePtr;
    let (w_ptr, _) = w_bf16.device_ptr(&device.stream);
    let (a_ptr, _) = a_bf16.device_ptr(&device.stream);
    let (c_ptr, _) = output.device_ptr(&device.stream);

    let status = cublas_sys::cublasGemmEx(
        *device.blas.handle(),
        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        out_dim as i32,
        batch as i32,
        in_dim as i32,
        &alpha as *const f32 as *const std::ffi::c_void,
        w_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,
        a_ptr as *const std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_16BF,
        in_dim as i32,
        &beta as *const f32 as *const std::ffi::c_void,
        c_ptr as *mut std::ffi::c_void,
        cublas_sys::cudaDataType_t::CUDA_R_32F,
        out_dim as i32,
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(RuntimeError::Compute(format!(
            "cublasGemmEx BF16 GEMM {label}: status={status:?}",
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The 16-bit store's readers and writers. Every function here takes `u16`
// buffers and the kernels in `KernelSet::kv16`, which exist only when the
// backend was built for a 16-bit store and are of that store's format; nothing
// here can be reached with F32 bytes.
// ---------------------------------------------------------------------------

/// `kv_cache_write_batch_{f16,bf16}`: `batch` tokens of F32 K or V into a
/// 16-bit store, rounding on the way in and counting what is stored as ±Inf or
/// NaN.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_kv_cache_write_batch_16(
    device: &CudaDevice,
    kv16: &super::decode::Kv16Kernels,
    cache: &mut CudaSlice<u16>,
    data: &CudaView<'_, f32>,
    overflow: &mut CudaSlice<u32>,
    pos_start: usize,
    batch: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<(), RuntimeError> {
    let kv_dim = num_kv_heads * head_dim;
    let total = batch * kv_dim;
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let pos_start_u32 = pos_start as u32;
    let batch_u32 = batch as u32;
    let nkvh = num_kv_heads as u32;
    let msl = max_seq_len as u32;
    let hd = head_dim as u32;
    device
        .stream
        .launch_builder(&kv16.write_batch)
        .arg(cache)
        .arg(data)
        .arg(overflow)
        .arg(&pos_start_u32)
        .arg(&batch_u32)
        .arg(&nkvh)
        .arg(&msl)
        .arg(&hd)
        .launch(launch_cfg)
        .map_err(|e| RuntimeError::Compute(format!("16-bit kv_cache_write_batch launch: {e}")))?;
    Ok(())
}

/// `kv_cache_widen_{f16,bf16}` for K and V: positions `0..count` of every head,
/// widened into `[num_kv_heads, count, head_dim]` F32 buffers — the F32 cache
/// layout with a position stride of `count`, which is what the returned view
/// says.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_kv_widen_16<'a>(
    device: &CudaDevice,
    kv16: &super::decode::Kv16Kernels,
    k_cache: &CudaSlice<u16>,
    v_cache: &CudaSlice<u16>,
    out: &'a mut (CudaSlice<f32>, CudaSlice<f32>),
    num_kv_heads: usize,
    count: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Result<KvView<'a>, RuntimeError> {
    let total = num_kv_heads * count * head_dim;
    if out.0.len() < total || out.1.len() < total {
        return Err(RuntimeError::Compute(format!(
            "kv widen: {total} floats needed, buffers hold {} and {}",
            out.0.len(),
            out.1.len()
        )));
    }
    let config = LaunchConfig::for_elements(total);
    let launch_cfg = CudarcLaunchConfig {
        grid_dim: (config.grid_dim, 1, 1),
        block_dim: (config.block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkvh = num_kv_heads as u32;
    let cnt = count as u32;
    let msl = max_seq_len as u32;
    let hd = head_dim as u32;
    for (cache, dst, which) in [(k_cache, &mut out.0, "K"), (v_cache, &mut out.1, "V")] {
        device
            .stream
            .launch_builder(&kv16.widen)
            .arg(cache)
            .arg(dst)
            .arg(&nkvh)
            .arg(&cnt)
            .arg(&msl)
            .arg(&hd)
            .launch(launch_cfg)
            .map_err(|e| {
                RuntimeError::Compute(format!("16-bit kv_cache_widen {which} launch: {e}"))
            })?;
    }
    Ok(KvView {
        k: &out.0,
        v: &out.1,
        seq_stride: count,
    })
}

#[cfg(test)]
mod tests {
    use super::super::workspace::RegionTable;
    use super::test_models::{dense_switches, moe_switches, Model, MOE, Q27B, Q9B, ROWS};
    use super::*;
    use lumen_format::planar_dequant::{dequantize_fp8, dequantize_nvfp4};

    /// Qwen3.8-27B (hidden 5120, 24 query / 4 KV heads of 256, FFN 17408,
    /// GDN qkv 10240 and value 6144 over 48 heads, fused Q+gate) at the
    /// 2048-row slice and a 32768-token context: the bytes the load reserves
    /// for its prefill, element by element from the layouts' arithmetic.
    #[test]
    fn the_27b_prefill_workspace_is_what_its_buffers_take() {
        let scratch = PrefillScratchLayout::new(
            2048,
            5120,
            24 * 256,
            4 * 256,
            17408,
            Some(10240),
            Some(6144),
            true,
        );
        // dequant_f32 is the FFN matrix, activation_f16 the FFN row width.
        assert_eq!(scratch.dequant_f32, 17408 * 5120);
        assert_eq!(scratch.activation_f16, 2048 * 17408 * 2);
        let gdn = GdnPrefillScratchLayout::new(2048, 10240, 48, 6144);
        let scores = attn_score_block_elems(2048, 24, 4, 32768 - 2048).unwrap();
        // group 6 x min(rows, 512) x 32768 keys.
        assert_eq!(scores, 6 * 512 * 32768);
        let ws = PrefillWorkspaceLayout::of(
            scratch,
            Some(gdn),
            Some(scores),
            None,
            PrefillRoutes::default(),
            ConversionScratch::default(),
        );
        // One buffer per value: the scratch values take 1_327_554_560 bytes,
        // the GDN values 4 * (2 * 2048 * 10240 + 4 * 2048 * 48 + 3 * 2048 *
        // 6144) and the score block 4 * 6 * 512 * 32768. Every value's bytes
        // are a multiple of the 256-byte region alignment at 2048 rows, so the
        // arena holds no padding between them: 0 bytes.
        let unshared = super::super::workspace::unshared(&ws).bytes() as u64;
        assert_eq!(
            unshared,
            1_327_554_560
                + 4 * (2 * 2048 * 10240 + 4 * 2048 * 48 + 3 * 2048 * 6144)
                + 4 * scores as u64
        );
        assert_eq!(unshared, 2_050_547_712);
    }

    fn plan(model: &Model, gemms: &[PrefillGemm], switches: &GemmSwitches) -> ConversionScratch {
        ConversionScratch::plan(gemms, switches, ROWS, &model.scratch())
    }

    fn sizes(
        f32: Option<usize>,
        activation: Option<usize>,
        f16: Option<usize>,
    ) -> ConversionScratch {
        ConversionScratch {
            dequant_f32: f32,
            activation_f16: activation,
            dequant_f16: f16,
        }
    }

    /// A dense model whose projections dequantize to F16 (Q8_0, its split
    /// layout, Q4_0, the K-quants, INT4) takes the F16 pair at the largest
    /// weight (the FFN matrix, 2 bytes an element) and the widest row (the
    /// down projection's FFN width), and no F32 dequant scratch: no GEMM's
    /// F16 need exceeds the one-size bound, so none falls back.
    #[test]
    fn the_f16_dequant_schemes_take_the_f16_pair_at_the_largest_weight() {
        for model in [Q27B, Q9B] {
            for weight in [
                WeightClass::Q8Raw,
                WeightClass::Q8Split,
                WeightClass::Q4Raw,
                WeightClass::KQuant,
                WeightClass::Ct4Raw,
            ] {
                let got = plan(
                    &model,
                    &model.gemms(weight, WeightClass::Q8Raw, false),
                    &dense_switches(),
                );
                let ffn = model.inter * model.hidden;
                assert_eq!(
                    got,
                    sizes(None, Some(ROWS * model.inter * 2), Some(2 * ffn)),
                    "{} {weight:?}",
                    model.name
                );
                assert_eq!(got.dequant_f16, Some(model.scratch().dequant_f16));
            }
        }
    }

    /// BF16 weights convert only their activations; the Q8_0 alpha/beta pair
    /// still dequantizes to F16 at its own size (heads x hidden).
    #[test]
    fn bf16_weights_take_the_activation_scratch_and_q8_gates_their_own_f16() {
        for model in [Q27B, Q9B] {
            let got = plan(
                &model,
                &model.gemms(WeightClass::Bf16Raw, WeightClass::Q8Raw, false),
                &dense_switches(),
            );
            let gates = model.gdn.1 * model.hidden;
            assert_eq!(
                got,
                sizes(None, Some(ROWS * model.inter * 2), Some(2 * gates)),
                "{}",
                model.name
            );
        }
    }

    /// Planar weights decode into the F32 scratch only, at the largest
    /// weight; F32 weights take no scratch at all.
    #[test]
    fn planar_weights_take_the_f32_scratch_and_f32_weights_none() {
        for model in [Q27B, Q9B] {
            let planar = plan(
                &model,
                &model.gemms(WeightClass::Planar, WeightClass::F32, false),
                &dense_switches(),
            );
            let f32 = plan(
                &model,
                &model.gemms(WeightClass::F32, WeightClass::F32, false),
                &dense_switches(),
            );
            assert_eq!(
                planar,
                sizes(Some(model.inter * model.hidden), None, None),
                "{}",
                model.name
            );
            assert_eq!(f32, ConversionScratch::default(), "{}", model.name);
        }
    }

    /// A mixed import (planar projections, a BF16 GDN output projection)
    /// takes the F32 scratch for the planes and the activation scratch at the
    /// BF16 GEMM's row.
    #[test]
    fn a_mixed_import_takes_the_scratch_of_each_of_its_routes() {
        let model = Q27B;
        let mut gemms = model.gemms(WeightClass::Planar, WeightClass::F32, false);
        let ssm_out = gemms
            .iter_mut()
            .find(|g| g.residual && g.in_dim == model.gdn.2)
            .unwrap();
        ssm_out.weight = WeightClass::Bf16Raw;
        let got = plan(&model, &gemms, &dense_switches());
        assert_eq!(
            got,
            sizes(
                Some(model.inter * model.hidden),
                Some(ROWS * model.gdn.2 * 2),
                None
            )
        );
    }

    /// `LUMEN_CUDA_PREFILL_F32` sends the Q8_0 and Q4_0 projections to the F32
    /// fallback (and leaves the K-quants refused, needing nothing).
    #[test]
    fn the_f32_switch_takes_the_f32_scratch() {
        let model = Q27B;
        let switches = GemmSwitches {
            force_f32: true,
            ..dense_switches()
        };
        for weight in [WeightClass::Q8Raw, WeightClass::Q4Raw] {
            let got = plan(&model, &model.gemms(weight, weight, false), &switches);
            assert_eq!(
                got,
                sizes(Some(model.inter * model.hidden), None, None),
                "{weight:?}"
            );
        }
        let kquant = plan(
            &model,
            &model.gemms(WeightClass::KQuant, WeightClass::KQuant, false),
            &switches,
        );
        assert_eq!(kquant, ConversionScratch::default());
    }

    /// The MoE model's Q8_0 projections take MMQ (no scratch); its residual
    /// GEMMs (`wo`, `ssm_out`, 2048 x 4096) dequantize to F16, and the cached
    /// alpha/beta pair converts its hidden-wide rows.
    #[test]
    fn moe_q8_takes_the_f16_pair_of_its_residual_gemms_only() {
        let model = MOE;
        let got = plan(
            &model,
            &model.gemms(WeightClass::Q8Raw, WeightClass::Q8Raw, true),
            &moe_switches(false),
        );
        assert_eq!(
            got,
            sizes(None, Some(ROWS * 4096 * 2), Some(2 * 2048 * 4096))
        );
        // With the MMQ switch set, the residual GEMMs take MMQ too, and only
        // the cache arm of the alpha/beta pair converts anything.
        let mmq_everywhere = GemmSwitches {
            q8_proj_mmq_set: true,
            ..moe_switches(false)
        };
        let got = plan(
            &model,
            &model.gemms(WeightClass::Q8Raw, WeightClass::Q8Raw, true),
            &mmq_everywhere,
        );
        assert_eq!(got, ConversionScratch::default());
    }

    /// MoE BF16: BF16 GEMMs convert their activations (widest row 4096), the
    /// Q8_0 alpha/beta pair takes the F32 SGEMM at heads x hidden.
    #[test]
    fn moe_bf16_takes_the_activation_scratch_and_f32_gates() {
        let model = MOE;
        let got = plan(
            &model,
            &model.gemms(WeightClass::Bf16Raw, WeightClass::Q8Raw, true),
            &moe_switches(true),
        );
        assert_eq!(got, sizes(Some(32 * 2048), Some(ROWS * 4096 * 2), None));
    }

    /// A GEMM whose F16 need exceeds the one-size bound takes the F32
    /// fallback as with the bound, and that fallback's own length check
    /// fails as with the bound: the regions are capped at the bound.
    #[test]
    fn a_need_over_the_bound_is_planned_as_the_bound_decides_it() {
        let model = Q27B;
        let bound = model.scratch();
        let over = PrefillGemm {
            weight: WeightClass::Q8Raw,
            out_dim: model.inter + 1,
            in_dim: model.hidden,
            residual: false,
            gdn_alpha_beta: false,
            f16_cache: false,
        };
        let got = plan(&model, &[over], &dense_switches());
        assert_eq!(got, sizes(Some(bound.dequant_f32), None, None));
        assert!(got.dequant_f32.unwrap() < over.out_dim * over.in_dim);
    }

    /// The production planar prefill launchers on NVFP4 and FP8 planes of many groups:
    /// `dequant_planar_into_scratch` (through `launch_dequant_plane_to_f32`) decodes each plane bit for bit
    /// as the host decoder does, and the planar branch of `launch_gemm_residual` returns `W*x + residual`
    /// within the prefill accuracy bound (`rel_l2 <= 1e-4`) of an f64 host reference. The shape leaves a
    /// partial last thread block. Needs a CUDA device; run with
    /// `cargo test --release -p lumen-runtime --features cuda planar_prefill_launchers -- --ignored`.
    #[test]
    #[ignore = "needs a CUDA device; run with --ignored"]
    fn planar_prefill_launchers_match_the_host_reference() {
        let device = CudaDevice::new(0).expect("this test needs a CUDA device");
        let spec = super::super::attention_decode::DecodeAttentionSpec::for_shape(2, 2, 128)
            .expect("a test shape inside the kernel's domain");
        let kernels =
            super::super::decode::compile_all_kernels(&device, crate::kv::KvPrecision::F32, spec)
                .expect("kernels compile");
        // 1632 NVFP4 groups; 6528 decode threads for NVFP4, not a multiple of the 256-thread block.
        let (out_dim, in_dim) = (96usize, 272usize);
        let n = out_dim * in_dim;
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        // Any E4M3 code but the two NaNs.
        let finite_e4m3 = |b: u8| if b == 0x7F || b == 0xFF { 0x38 } else { b };

        let packed: Vec<u8> = (0..n / 2).map(|_| next()).collect();
        let block_scales: Vec<u8> = (0..n / 16).map(|_| finite_e4m3(next())).collect();
        let nvfp4_global = 3.5e-4f32;
        let mut nvfp4_plane = packed.clone();
        nvfp4_plane.extend_from_slice(&block_scales);
        nvfp4_plane.extend_from_slice(&nvfp4_global.to_le_bytes());

        let fp8_weights: Vec<u8> = (0..n).map(|_| finite_e4m3(next())).collect();
        let fp8_scale = 3.3e-3f32;
        let mut fp8_plane = fp8_weights.clone();
        fp8_plane.extend_from_slice(&fp8_scale.to_le_bytes());

        let cases = [
            (
                "nvfp4",
                GpuWeightBuf::Nvfp4Raw(device.htod_copy(&nvfp4_plane).unwrap()),
                dequantize_nvfp4(&packed, &block_scales, nvfp4_global).expect("host decode"),
            ),
            (
                "fp8",
                GpuWeightBuf::Fp8Raw(device.htod_copy(&fp8_plane).unwrap()),
                dequantize_fp8(&fp8_weights, fp8_scale),
            ),
        ];
        for (tag, weight, w) in &cases {
            let mut scratch = device.alloc_zeros::<f32>(n).unwrap();
            unsafe {
                dequant_planar_into_scratch(
                    &device,
                    &kernels,
                    weight,
                    &mut scratch.as_view_mut(),
                    n,
                    tag,
                )
                .unwrap();
            }
            let decoded = device.dtoh_copy(&scratch).unwrap();
            if let Some(i) = (0..n).find(|&i| decoded[i].to_bits() != w[i].to_bits()) {
                panic!(
                    "{tag}: element {i}: device {:#010x} vs host {:#010x}",
                    decoded[i].to_bits(),
                    w[i].to_bits()
                );
            }

            for batch in [1usize, 33] {
                let x: Vec<f32> = (0..batch * in_dim)
                    .map(|k| (k % 211) as f32 * 0.004 - 0.42)
                    .collect();
                let residual: Vec<f32> = (0..batch * out_dim)
                    .map(|k| (k % 13) as f32 * 0.5 - 3.0)
                    .collect();
                let x_gpu = device.htod_copy(&x).unwrap();
                let residual_gpu = device.htod_copy(&residual).unwrap();
                let mut out = device.alloc_zeros::<f32>(batch * out_dim).unwrap();
                // The planar arm borrows the F32 dequant scratch only.
                let mut ws =
                    Workspace::alloc(&device, RegionTable::of(&[(Role::DequantF32, n)])).unwrap();
                let ([], mut conversion) = ws.lend([]);
                unsafe {
                    launch_gemm_residual(
                        &device,
                        &kernels,
                        weight,
                        None,
                        &x_gpu.as_view(),
                        &residual_gpu.as_view(),
                        &mut out.as_view_mut(),
                        &mut conversion,
                        batch,
                        out_dim,
                        in_dim,
                        tag,
                    )
                    .unwrap();
                }
                let got = device.dtoh_copy(&out).unwrap();

                // out[b, r] = residual[b, r] + sum_k W[r, k] * x[b, k], in f64.
                let mut want = vec![0.0f64; batch * out_dim];
                let mut wx_sq = 0.0f64;
                for b in 0..batch {
                    for r in 0..out_dim {
                        let dot: f64 = (0..in_dim)
                            .map(|k| w[r * in_dim + k] as f64 * x[b * in_dim + k] as f64)
                            .sum();
                        wx_sq += dot * dot;
                        want[b * out_dim + r] = residual[b * out_dim + r] as f64 + dot;
                    }
                }
                let want_sq: f64 = want.iter().map(|v| v * v).sum();
                let err_sq: f64 = got
                    .iter()
                    .zip(&want)
                    .map(|(g, v)| (*g as f64 - v).powi(2))
                    .sum();
                let rel_l2 = (err_sq / want_sq).sqrt();
                // The residual is O(1) by construction; W*x must be large enough too, so that dropping
                // either term moves the result far past the bound.
                assert!(
                    (wx_sq / want_sq).sqrt() > 1e-2,
                    "{tag} batch {batch}: W*x is too small against the residual to be checked"
                );
                assert!(
                    rel_l2 <= 1e-4,
                    "{tag} batch {batch}: rel_l2 {rel_l2:e} > 1e-4"
                );
            }
        }
    }
}
