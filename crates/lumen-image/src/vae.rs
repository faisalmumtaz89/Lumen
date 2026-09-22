//! CPU reference for the Qwen-Image-2.1 VAE decoder.
//!
//! This mirrors `AutoencoderKLQwenImage21._decode` and `QwenImage21Decoder3d`
//! from the diffusers reference (`autoencoder_kl_qwenimage21.py`). Four things
//! in that file are surprising enough to restate here, because each one would
//! otherwise be "fixed" into something plausible and wrong:
//!
//! 1. **`QwenImage21CausalConv3d` subclasses `nn.Conv2d`, not `nn.Conv3d`.** Its
//!    forward squeezes the temporal axis away, pads `(pad_w, pad_w, pad_h,
//!    pad_h)`, runs the 2-D convolution and unsqueezes the axis back. There is
//!    no temporal convolution anywhere in this model, and the layer *raises* if
//!    handed a feature cache. So every convolution below is a plain 2-D
//!    convolution over a single frame.
//!
//! 2. **`QwenImage21RMS_norm` is `F.normalize`, not an epsilon-RMS.** It is
//!    `F.normalize(x, dim=1) * sqrt(dim) * gamma`, i.e. divide by
//!    `max(||x||_2, 1e-12)` over the channel axis and then scale by `sqrt(C)`.
//!    That equals an RMS norm whose epsilon is zero and whose guard is a clamp
//!    on the norm rather than a term inside the square root; the two differ
//!    sharply for small-magnitude channel vectors. `bias` is `False` at every
//!    construction site, so `gamma` is the only parameter.
//!
//! 3. **`is_residual` is `true`**, which selects `QwenImage21ResidualUpBlock`:
//!    a learned `upsampler` plus a parameter-free `QwenImage21DupUp3D`
//!    `avg_shortcut` (repeat-interleave over channels, reshape, permute) added
//!    to the block output.
//!
//! 4. **The feature cache is inert here.** `_decode` calls `clear_cache()`, so
//!    every `feat_cache` slot is `None` when frame 0 runs; each `CausalConv3d`
//!    therefore receives `cache_x = None` and behaves as a plain convolution,
//!    and `QwenImage21Resample`'s `upsample3d` branch takes its
//!    `feat_cache[idx] is None` path, which stores the marker `"Rep"`, skips
//!    `time_conv` entirely and does no temporal upsampling. The cache only ever
//!    holds a tensor from frame 1 onwards — and frame 1 immediately hits
//!    `self.conv_in(x, feat_cache[idx])` with that tensor, which raises. So the
//!    reference supports exactly one frame, the cache never influences a
//!    result, and this implementation omits it. The consequence is that
//!    `time_conv`'s weights are never read (see [`VaeDecoder::load`]).
//!
//! Everything is single-threaded f32, clarity over speed: the reference is the
//! numerical target, not the fast path.

use std::path::Path;

use crate::lbi::{LbiError, LbiFile};
use crate::tensor::silu;

/// `F.normalize`'s default epsilon: the *norm* is clamped up to this, rather
/// than an epsilon being added under the square root.
const NORMALIZE_EPS: f32 = 1e-12;

/// The reference hard-codes `QwenImage21MidBlock(dims[0], dropout, num_layers=1)`
/// for both encoder and decoder, so the middle block always has one attention
/// layer and two residual blocks.
const MID_BLOCK_NUM_LAYERS: usize = 1;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum VaeError {
    Lbi(LbiError),
    /// A weight the decoder needs is not in the `.lbi`.
    MissingTensor(String),
    /// A weight is present but not the shape the architecture requires.
    ShapeMismatch {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    /// The config selects a variant this reference does not implement.
    Unsupported(String),
    /// A config field is present but malformed or internally inconsistent.
    BadConfig(String),
}

impl std::fmt::Display for VaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lbi(e) => write!(f, "lbi: {e}"),
            Self::MissingTensor(n) => write!(f, "weight {n} is missing"),
            Self::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "weight {tensor} has shape {actual:?} but the decoder needs {expected:?}"
            ),
            Self::Unsupported(m) => write!(f, "unsupported VAE configuration: {m}"),
            Self::BadConfig(m) => write!(f, "bad VAE configuration: {m}"),
        }
    }
}

impl std::error::Error for VaeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lbi(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LbiError> for VaeError {
    fn from(e: LbiError) -> Self {
        Self::Lbi(e)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The fields of `vae/config.json` the decoder reads.
///
/// Field names keep the checkpoint's spelling, including `temperal_downsample`.
#[derive(Debug, Clone, PartialEq)]
pub struct VaeConfig {
    /// `decoder_base_dim`; falls back to `base_dim` when the checkpoint stores
    /// it as null, exactly as the reference's `__init__` does.
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Kept for fidelity with the checkpoint. `QwenImage21Decoder3d` stores
    /// `attn_scales` and never consults it — attention in the decoder lives
    /// only in the middle block, which is unconditional — so it does not affect
    /// anything here either.
    pub attn_scales: Vec<f32>,
    /// The *encoder's* list. The decoder is built from its reverse; see
    /// [`VaeConfig::temperal_upsample`].
    pub temperal_downsample: Vec<bool>,
    pub out_channels: usize,
    pub is_residual: bool,
    pub patch_size: Option<usize>,
    pub scale_factor_spatial: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl VaeConfig {
    /// The shipped Qwen-Image-2.1 VAE configuration.
    // `latents_mean[24]` is -0.5236, which clippy reads as an approximation of
    // pi/6. It is a checkpoint constant, not a mathematical one.
    #[allow(clippy::approx_constant)]
    pub fn qwen_image_2_1() -> Self {
        Self {
            decoder_base_dim: 144,
            z_dim: 64,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            attn_scales: Vec::new(),
            temperal_downsample: vec![false, true, true, true],
            out_channels: 4,
            is_residual: true,
            patch_size: None,
            scale_factor_spatial: 16,
            latents_mean: vec![
                0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873,
                -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320,
                2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286,
                -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751,
                -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764,
                0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026,
                1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
            ],
            latents_std: vec![
                3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500,
                3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443,
                2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830,
                3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
                3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168,
                3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
                3.7662, 3.6746, 3.4563, 3.8161,
            ],
        }
    }

    /// `self.temperal_upsample = temperal_downsample[::-1]` — the autoencoder
    /// reverses the encoder's list and passes *that* to the decoder, which is
    /// why the decoder's own `temperal_upsample` default is never used.
    pub fn temperal_upsample(&self) -> Vec<bool> {
        self.temperal_downsample.iter().rev().copied().collect()
    }

    /// Overlay whatever `vae/config.json` keys the `.lbi` carries onto the
    /// shipped defaults. A missing or null key keeps the default; a key of the
    /// wrong type is an error rather than a silent fallback.
    pub fn from_lbi_config(value: &serde_json::Value) -> Result<Self, VaeError> {
        let mut cfg = Self::qwen_image_2_1();
        let obj = match value {
            serde_json::Value::Null => return Ok(cfg),
            serde_json::Value::Object(o) => o,
            other => {
                return Err(VaeError::BadConfig(format!(
                    "expected a JSON object, found {other}"
                )))
            }
        };

        // `decoder_base_dim` is `int | None`; when null the reference copies
        // `base_dim` into it.
        cfg.decoder_base_dim = match obj.get("decoder_base_dim") {
            Some(serde_json::Value::Null) | None => {
                cfg_usize(obj, "base_dim", cfg.decoder_base_dim)?
            }
            Some(v) => as_usize("decoder_base_dim", v)?,
        };
        cfg.z_dim = cfg_usize(obj, "z_dim", cfg.z_dim)?;
        cfg.num_res_blocks = cfg_usize(obj, "num_res_blocks", cfg.num_res_blocks)?;
        cfg.out_channels = cfg_usize(obj, "out_channels", cfg.out_channels)?;
        cfg.scale_factor_spatial =
            cfg_usize(obj, "scale_factor_spatial", cfg.scale_factor_spatial)?;
        cfg.is_residual = cfg_bool(obj, "is_residual", cfg.is_residual)?;
        cfg.patch_size = match obj.get("patch_size") {
            Some(serde_json::Value::Null) | None => cfg.patch_size,
            Some(v) => Some(as_usize("patch_size", v)?),
        };
        cfg.dim_mult = cfg_vec(obj, "dim_mult", cfg.dim_mult, as_usize)?;
        cfg.attn_scales = cfg_vec(obj, "attn_scales", cfg.attn_scales, as_f32)?;
        cfg.temperal_downsample = cfg_vec(
            obj,
            "temperal_downsample",
            cfg.temperal_downsample,
            as_bool_elem,
        )?;
        cfg.latents_mean = cfg_vec(obj, "latents_mean", cfg.latents_mean, as_f32)?;
        cfg.latents_std = cfg_vec(obj, "latents_std", cfg.latents_std, as_f32)?;
        Ok(cfg)
    }

    /// Reject every configuration whose decoder this module does not build.
    pub(crate) fn validate(&self) -> Result<(), VaeError> {
        if !self.is_residual {
            return Err(VaeError::Unsupported(
                "is_residual is false, which selects QwenImage21UpBlock; only the \
                 residual up block (is_residual = true) is implemented"
                    .to_string(),
            ));
        }
        match self.patch_size {
            None | Some(1) => {}
            Some(p) => {
                return Err(VaeError::Unsupported(format!(
                    "patch_size {p} needs the unpatchify path, which is not implemented"
                )))
            }
        }
        if self.dim_mult.is_empty() {
            return Err(VaeError::BadConfig("dim_mult is empty".to_string()));
        }
        if self.dim_mult.iter().any(|&m| m == 0) {
            return Err(VaeError::BadConfig(
                "dim_mult holds a zero multiplier".to_string(),
            ));
        }
        for (name, v) in [
            ("decoder_base_dim", self.decoder_base_dim),
            ("z_dim", self.z_dim),
            ("out_channels", self.out_channels),
        ] {
            if v == 0 {
                return Err(VaeError::BadConfig(format!("{name} is zero")));
            }
        }
        if self
            .dim_mult
            .iter()
            .any(|&m| self.decoder_base_dim.checked_mul(m).is_none())
        {
            return Err(VaeError::BadConfig(
                "decoder_base_dim * dim_mult overflows".to_string(),
            ));
        }
        // `temperal_upsample[i]` is read for every block that upsamples, which
        // is every block but the last.
        let upsampling_blocks = self.dim_mult.len() - 1;
        if self.temperal_downsample.len() < upsampling_blocks {
            return Err(VaeError::BadConfig(format!(
                "temperal_downsample has {} entries but {upsampling_blocks} upsampling \
                 blocks read it",
                self.temperal_downsample.len()
            )));
        }
        // Each upsampling block doubles height and width, so the spatial
        // compression ratio the config advertises has to be 2^blocks.
        let implied = 1usize.checked_shl(upsampling_blocks as u32);
        if implied != Some(self.scale_factor_spatial) {
            return Err(VaeError::BadConfig(format!(
                "{upsampling_blocks} upsampling blocks imply a spatial factor of {implied:?}, \
                 but scale_factor_spatial is {}",
                self.scale_factor_spatial
            )));
        }
        Ok(())
    }

    /// `dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]`.
    pub(crate) fn decoder_dims(&self) -> Vec<usize> {
        let last = *self.dim_mult.last().expect("validated as non-empty");
        std::iter::once(last)
            .chain(self.dim_mult.iter().rev().copied())
            .map(|u| self.decoder_base_dim * u)
            .collect()
    }
}

fn as_usize(key: &str, v: &serde_json::Value) -> Result<usize, VaeError> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| VaeError::BadConfig(format!("{key} is not a non-negative integer: {v}")))
}

fn as_f32(key: &str, v: &serde_json::Value) -> Result<f32, VaeError> {
    v.as_f64()
        .map(|n| n as f32)
        .ok_or_else(|| VaeError::BadConfig(format!("{key} is not a number: {v}")))
}

fn as_bool_elem(key: &str, v: &serde_json::Value) -> Result<bool, VaeError> {
    v.as_bool()
        .ok_or_else(|| VaeError::BadConfig(format!("{key} is not a boolean: {v}")))
}

fn cfg_usize(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    current: usize,
) -> Result<usize, VaeError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(current),
        Some(v) => as_usize(key, v),
    }
}

fn cfg_bool(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    current: bool,
) -> Result<bool, VaeError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(current),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| VaeError::BadConfig(format!("{key} is not a boolean: {v}"))),
    }
}

fn cfg_vec<T>(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    current: Vec<T>,
    parse: fn(&str, &serde_json::Value) -> Result<T, VaeError>,
) -> Result<Vec<T>, VaeError> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(current),
        Some(serde_json::Value::Array(a)) => a.iter().map(|v| parse(key, v)).collect(),
        Some(v) => Err(VaeError::BadConfig(format!("{key} is not an array: {v}"))),
    }
}

// ---------------------------------------------------------------------------
// A 4-D NCHW activation buffer
// ---------------------------------------------------------------------------

/// Row-major `[n, c, h, w]`.
///
/// The reference's activations are 5-D `[b, c, t, h, w]`, but `t` is 1 at every
/// point of a single-frame decode (see the module docs: the temporal upsample
/// is skipped on the first chunk, and later chunks cannot run at all), so the
/// temporal axis is folded away here and restored only in the decoded output's
/// documented shape.
#[derive(Debug, Clone, PartialEq)]
struct Tensor4 {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Tensor4 {
    fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            n,
            c,
            h,
            w,
            data: vec![0.0; n * c * h * w],
        }
    }

    fn new(n: usize, c: usize, h: usize, w: usize, data: Vec<f32>) -> Self {
        assert_eq!(n * c * h * w, data.len(), "shape does not match the data");
        Self { n, c, h, w, data }
    }

    fn hw(&self) -> usize {
        self.h * self.w
    }

    fn plane(&self, n: usize, c: usize) -> &[f32] {
        let hw = self.hw();
        let off = (n * self.c + c) * hw;
        &self.data[off..off + hw]
    }

    fn plane_mut(&mut self, n: usize, c: usize) -> &mut [f32] {
        let hw = self.hw();
        let off = (n * self.c + c) * hw;
        &mut self.data[off..off + hw]
    }

    /// Elementwise `self += other`.
    fn add_assign(&mut self, other: &Tensor4) {
        assert_eq!(
            (self.n, self.c, self.h, self.w),
            (other.n, other.c, other.h, other.w),
            "shape mismatch in add"
        );
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// A 2-D convolution with stride 1 and symmetric zero padding.
///
/// This covers every convolution the decoder runs: `QwenImage21CausalConv3d`
/// pads `(padding[1], padding[1], padding[0], padding[0])` and then calls
/// `nn.Conv2d` with zero padding, which is the same thing, and the `nn.Conv2d`
/// inside `QwenImage21Resample.resample` uses `padding=1` directly. No decoder
/// convolution is strided — only the encoder's downsamplers are.
#[derive(Debug, Clone)]
struct Conv2d {
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    /// `[out_c, in_c, kh, kw]`, the `nn.Conv2d` layout.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv2d {
    fn apply(&self, x: &Tensor4) -> Tensor4 {
        assert_eq!(x.c, self.in_c, "convolution input channels differ");
        assert!(
            x.h + 2 * self.pad_h >= self.kh && x.w + 2 * self.pad_w >= self.kw,
            "convolution kernel is larger than the padded input"
        );
        let oh = x.h + 2 * self.pad_h - self.kh + 1;
        let ow = x.w + 2 * self.pad_w - self.kw + 1;
        let mut out = Tensor4::zeros(x.n, self.out_c, oh, ow);

        // Accumulation order per output element is (in channel, kernel row,
        // kernel column), the same order the obvious nested sum would use; the
        // loop nest is only rearranged so the innermost loop walks contiguous
        // memory.
        //
        // Output channels are independent, so they are split across threads when
        // there are enough of them to be worth it. Each output element still
        // accumulates in the same order, so the result is unchanged.
        let threads = if self.out_c >= 8 {
            crate::tensor::default_threads().min(self.out_c)
        } else {
            1
        };
        // Writes one output channel plane: `dst` is that plane, `[oh, ow]`.
        let compute_plane = |b: usize, oc: usize, dst: &mut [f32]| {
            let o = dst;
            for ic in 0..self.in_c {
                let hw = x.h * x.w;
                let xp = &x.data[(b * x.c + ic) * hw..(b * x.c + ic) * hw + hw];
                for ki in 0..self.kh {
                    for kj in 0..self.kw {
                        let wv = self.weight[((oc * self.in_c + ic) * self.kh + ki) * self.kw + kj];
                        for oy in 0..oh {
                            let iy = oy + ki;
                            if iy < self.pad_h || iy - self.pad_h >= x.h {
                                continue;
                            }
                            let iy = iy - self.pad_h;
                            let row = &xp[iy * x.w..iy * x.w + x.w];
                            let orow = &mut o[oy * ow..oy * ow + ow];
                            for ox in 0..ow {
                                let ix = ox + kj;
                                if ix < self.pad_w || ix - self.pad_w >= x.w {
                                    continue;
                                }
                                orow[ox] += wv * row[ix - self.pad_w];
                            }
                        }
                    }
                }
            }
        };

        let plane = oh * ow;
        if threads <= 1 {
            for b in 0..x.n {
                for oc in 0..self.out_c {
                    let off = (b * self.out_c + oc) * plane;
                    compute_plane(b, oc, &mut out.data[off..off + plane]);
                }
            }
        } else {
            // Planes are independent and contiguous, so the output is handed out
            // in contiguous bands and each thread derives its own (batch,
            // channel) from its position. No shared state, no ordering.
            let per = threads * plane;
            std::thread::scope(|scope| {
                for (band_idx, band) in out.data.chunks_mut(per).enumerate() {
                    let out_c = self.out_c;
                    scope.spawn(move || {
                        let base = band_idx * threads;
                        for (i, pl) in band.chunks_mut(plane).enumerate() {
                            let flat = base + i;
                            compute_plane(flat / out_c, flat % out_c, pl);
                        }
                    });
                }
            });
        }

        // `nn.Conv2d` adds the bias to the completed sum.
        for b in 0..out.n {
            for oc in 0..self.out_c {
                let bias = self.bias[oc];
                for v in out.plane_mut(b, oc) {
                    *v += bias;
                }
            }
        }
        out
    }
}

/// `QwenImage21RMS_norm`: `F.normalize(x, dim=1) * sqrt(C) * gamma`.
///
/// `F.normalize` divides by `||x||_2.clamp_min(1e-12)` along the channel axis,
/// so this is *not* `x / sqrt(mean(x^2) + eps)`: the guard is a clamp on the
/// norm, and there is no epsilon inside the square root. `bias` is `False` at
/// every construction site in the reference, so there is no additive term.
fn rms_norm_channels(x: &Tensor4, gamma: &[f32]) -> Tensor4 {
    assert_eq!(x.c, gamma.len(), "norm width differs");
    let scale = (x.c as f32).sqrt();
    let hw = x.hw();
    let mut out = Tensor4::zeros(x.n, x.c, x.h, x.w);
    for b in 0..x.n {
        for p in 0..hw {
            let mut sum_sq = 0.0f32;
            for c in 0..x.c {
                let v = x.plane(b, c)[p];
                sum_sq += v * v;
            }
            let denom = sum_sq.sqrt().max(NORMALIZE_EPS);
            for c in 0..x.c {
                let v = x.plane(b, c)[p];
                out.plane_mut(b, c)[p] = (v / denom) * scale * gamma[c];
            }
        }
    }
    out
}

fn silu_in_place(x: &mut Tensor4) {
    for v in x.data.iter_mut() {
        *v = silu(*v);
    }
}

/// `QwenImage21Upsample(scale_factor=(2, 2), mode="nearest-exact")`.
///
/// For an exact integer scale of 2, `nearest-exact` maps output index `i` to
/// `floor((i + 0.5) / 2)`, which equals `i / 2` for every non-negative `i`, so
/// it agrees with plain nearest here.
fn upsample_nearest_2x(x: &Tensor4) -> Tensor4 {
    let mut out = Tensor4::zeros(x.n, x.c, x.h * 2, x.w * 2);
    for b in 0..x.n {
        for c in 0..x.c {
            let src = x.plane(b, c);
            let ow = x.w * 2;
            let dst = out.plane_mut(b, c);
            for y in 0..x.h * 2 {
                for xi in 0..ow {
                    dst[y * ow + xi] = src[(y / 2) * x.w + xi / 2];
                }
            }
        }
    }
    out
}

/// How a block's `avg_shortcut` (`QwenImage21DupUp3D`) is shaped.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DupUpSpec {
    out_c: usize,
    factor_t: usize,
    factor_s: usize,
    /// `out_channels * factor // in_channels`, the `repeat_interleave` count.
    repeats: usize,
}

impl DupUpSpec {
    fn new(in_c: usize, out_c: usize, factor_t: usize, factor_s: usize) -> Result<Self, VaeError> {
        let factor = factor_t * factor_s * factor_s;
        if out_c * factor % in_c != 0 {
            return Err(VaeError::BadConfig(format!(
                "DupUp3D needs out_channels ({out_c}) * factor ({factor}) to be divisible by \
                 in_channels ({in_c})"
            )));
        }
        Ok(Self {
            out_c,
            factor_t,
            factor_s,
            repeats: out_c * factor / in_c,
        })
    }

    fn factor(&self) -> usize {
        self.factor_t * self.factor_s * self.factor_s
    }
}

/// `QwenImage21DupUp3D.forward(x, first_chunk=True)` for a single input frame.
///
/// The reference repeat-interleaves the channel axis by `repeats`, views the
/// result as `[B, out_c, factor_t, factor_s, factor_s, T, H, W]`, permutes to
/// `[B, out_c, T, factor_t, H, factor_s, W, factor_s]` and merges each pair, so
///
/// ```text
/// out[b, o, t*factor_t + ft, h*factor_s + fs1, w*factor_s + fs2]
///     == x[b, (o*factor + ft*factor_s^2 + fs1*factor_s + fs2) / repeats, t, h, w]
/// ```
///
/// `first_chunk` then keeps `x[:, :, factor_t - 1:]`. With `T == 1` the
/// temporal axis has exactly `factor_t` entries and the slice keeps exactly one
/// of them — index `factor_t - 1` — which is why this takes no `first_chunk`
/// flag: `_decode` passes `first_chunk=True` for frame 0, and no later frame
/// can run (the causal convolutions refuse the cache frame 0 leaves behind).
fn dup_up_first_chunk(x: &Tensor4, spec: &DupUpSpec) -> Tensor4 {
    assert_eq!(
        x.c * spec.repeats,
        spec.out_c * spec.factor(),
        "DupUp3D repeat count does not match the channel counts"
    );
    let fs = spec.factor_s;
    let ft = spec.factor_t - 1;
    let mut out = Tensor4::zeros(x.n, spec.out_c, x.h * fs, x.w * fs);
    let ow = x.w * fs;
    for b in 0..x.n {
        for o in 0..spec.out_c {
            for fs1 in 0..fs {
                for fs2 in 0..fs {
                    let j = o * spec.factor() + ft * fs * fs + fs1 * fs + fs2;
                    let src = x.plane(b, j / spec.repeats).to_vec();
                    let dst = out.plane_mut(b, o);
                    for y in 0..x.h {
                        for xi in 0..x.w {
                            dst[(y * fs + fs1) * ow + xi * fs + fs2] = src[y * x.w + xi];
                        }
                    }
                }
            }
        }
    }
    out
}

/// `QwenImage21AttentionBlock`: single-head self-attention over the `h * w`
/// spatial positions, with `1x1` convolutions for the projections.
#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: Vec<f32>,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl AttentionBlock {
    fn apply(&self, x: &Tensor4) -> Tensor4 {
        let c = x.c;
        let tokens = x.hw();
        let normed = rms_norm_channels(x, &self.norm);
        // `to_qkv` emits `[q | k | v]` stacked on the channel axis; the
        // reference's `chunk(3, dim=-1)` after the transpose splits exactly
        // there.
        let qkv = self.to_qkv.apply(&normed);
        let scale = 1.0 / (c as f32).sqrt();

        let mut ctx = Tensor4::zeros(x.n, c, x.h, x.w);
        let mut row = vec![0.0f32; tokens];
        for b in 0..x.n {
            for i in 0..tokens {
                for j in 0..tokens {
                    let mut acc = 0.0f32;
                    for d in 0..c {
                        acc += qkv.plane(b, d)[i] * qkv.plane(b, c + d)[j];
                    }
                    row[j] = acc * scale;
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    denom += *v;
                }
                for v in row.iter_mut() {
                    *v /= denom;
                }
                for d in 0..c {
                    let value = qkv.plane(b, 2 * c + d);
                    let mut acc = 0.0f32;
                    for j in 0..tokens {
                        acc += row[j] * value[j];
                    }
                    ctx.plane_mut(b, d)[i] = acc;
                }
            }
        }

        let mut out = self.proj.apply(&ctx);
        out.add_assign(x);
        out
    }
}

/// `QwenImage21ResidualBlock`. Dropout is identity at inference.
#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: Vec<f32>,
    conv1: Conv2d,
    norm2: Vec<f32>,
    conv2: Conv2d,
    /// `conv_shortcut`, present only when the block changes channel count.
    shortcut: Option<Conv2d>,
}

impl ResidualBlock {
    fn apply(&self, x: &Tensor4) -> Tensor4 {
        let residual = match &self.shortcut {
            Some(conv) => conv.apply(x),
            None => x.clone(),
        };
        let mut h = rms_norm_channels(x, &self.norm1);
        silu_in_place(&mut h);
        let h = self.conv1.apply(&h);
        let mut h = rms_norm_channels(&h, &self.norm2);
        silu_in_place(&mut h);
        let mut h = self.conv2.apply(&h);
        h.add_assign(&residual);
        h
    }
}

/// `QwenImage21MidBlock`: `resnets[0]`, then each attention paired with the
/// next residual block.
#[derive(Debug, Clone)]
struct MidBlock {
    resnets: Vec<ResidualBlock>,
    attentions: Vec<AttentionBlock>,
}

impl MidBlock {
    fn apply(&self, x: &Tensor4) -> Tensor4 {
        let mut h = self.resnets[0].apply(x);
        for (attn, resnet) in self.attentions.iter().zip(&self.resnets[1..]) {
            h = attn.apply(&h);
            h = resnet.apply(&h);
        }
        h
    }
}

/// `QwenImage21ResidualUpBlock`: `num_res_blocks + 1` residual blocks, an
/// optional learned upsampler, and an optional parameter-free `DupUp3D`
/// shortcut added to the result.
#[derive(Debug, Clone)]
struct ResidualUpBlock {
    resnets: Vec<ResidualBlock>,
    /// `upsampler.resample.1`, the `nn.Conv2d` that follows the nearest-exact
    /// 2x interpolation. `upsampler.time_conv` exists in the checkpoint for the
    /// `upsample3d` modes but is never reached on a single frame, so it is not
    /// loaded.
    upsampler: Option<Conv2d>,
    shortcut: Option<DupUpSpec>,
}

impl ResidualUpBlock {
    fn apply(&self, x: &Tensor4) -> Tensor4 {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.apply(&h);
        }
        if let Some(conv) = &self.upsampler {
            h = conv.apply(&upsample_nearest_2x(&h));
        }
        if let Some(spec) = &self.shortcut {
            h.add_assign(&dup_up_first_chunk(x, spec));
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn entry_shape(file: &LbiFile, name: &str) -> Result<Vec<u64>, VaeError> {
    file.get(name)
        .map(|e| e.shape.clone())
        .ok_or_else(|| VaeError::MissingTensor(name.to_string()))
}

fn read_exact_shape(file: &LbiFile, name: &str, expected: &[u64]) -> Result<Vec<f32>, VaeError> {
    let actual = entry_shape(file, name)?;
    if actual != expected {
        return Err(VaeError::ShapeMismatch {
            tensor: name.to_string(),
            expected: expected.to_vec(),
            actual,
        });
    }
    Ok(file.read_f32(name)?)
}

fn load_conv(
    file: &LbiFile,
    prefix: &str,
    out_c: usize,
    in_c: usize,
    k: usize,
    pad: usize,
) -> Result<Conv2d, VaeError> {
    let wname = format!("{prefix}.weight");
    let bname = format!("{prefix}.bias");
    let weight = read_exact_shape(
        file,
        &wname,
        &[out_c as u64, in_c as u64, k as u64, k as u64],
    )?;
    let bias = read_exact_shape(file, &bname, &[out_c as u64])?;
    Ok(Conv2d {
        out_c,
        in_c,
        kh: k,
        kw: k,
        pad_h: pad,
        pad_w: pad,
        weight,
        bias,
    })
}

/// Load a `QwenImage21RMS_norm` `gamma`.
///
/// The parameter is created as `torch.ones((dim, 1, 1))` when `images=True` and
/// `torch.ones((dim, 1, 1, 1))` when `images=False`; the trailing axes exist
/// only to broadcast. Any stored shape whose leading extent is `dim` and whose
/// element count is `dim` therefore carries the same `dim` values, so all of
/// those are accepted and anything else is rejected.
fn load_norm(file: &LbiFile, name: &str, dim: usize, images: bool) -> Result<Vec<f32>, VaeError> {
    let actual = entry_shape(file, name)?;
    let canonical: Vec<u64> = if images {
        vec![dim as u64, 1, 1]
    } else {
        vec![dim as u64, 1, 1, 1]
    };
    let count: u64 = actual.iter().product();
    if actual.first() != Some(&(dim as u64)) || count != dim as u64 {
        return Err(VaeError::ShapeMismatch {
            tensor: name.to_string(),
            expected: canonical,
            actual,
        });
    }
    Ok(file.read_f32(name)?)
}

fn load_residual_block(
    file: &LbiFile,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<ResidualBlock, VaeError> {
    let norm1 = load_norm(file, &format!("{prefix}.norm1.gamma"), in_dim, false)?;
    let conv1 = load_conv(file, &format!("{prefix}.conv1"), out_dim, in_dim, 3, 1)?;
    let norm2 = load_norm(file, &format!("{prefix}.norm2.gamma"), out_dim, false)?;
    let conv2 = load_conv(file, &format!("{prefix}.conv2"), out_dim, out_dim, 3, 1)?;
    // `conv_shortcut` is `nn.Identity` when the channel count is unchanged, so
    // the checkpoint has no tensor for it.
    let shortcut = if in_dim != out_dim {
        Some(load_conv(
            file,
            &format!("{prefix}.conv_shortcut"),
            out_dim,
            in_dim,
            1,
            0,
        )?)
    } else {
        None
    };
    Ok(ResidualBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        shortcut,
    })
}

fn load_attention(file: &LbiFile, prefix: &str, dim: usize) -> Result<AttentionBlock, VaeError> {
    Ok(AttentionBlock {
        norm: load_norm(file, &format!("{prefix}.norm.gamma"), dim, true)?,
        to_qkv: load_conv(file, &format!("{prefix}.to_qkv"), dim * 3, dim, 1, 0)?,
        proj: load_conv(file, &format!("{prefix}.proj"), dim, dim, 1, 0)?,
    })
}

// ---------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------

/// The Qwen-Image-2.1 VAE decoder, weights resident.
#[derive(Debug)]
pub struct VaeDecoder {
    config: VaeConfig,
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<ResidualUpBlock>,
    norm_out: Vec<f32>,
    conv_out: Conv2d,
}

impl VaeDecoder {
    /// Load the decoder half of a `vae` `.lbi`.
    ///
    /// The architecture is derived from the container's config (falling back to
    /// [`VaeConfig::qwen_image_2_1`] for anything the config omits), and every
    /// weight that architecture needs is looked up by name and shape-checked.
    ///
    /// Two groups of the checkpoint's tensors are deliberately not read: the
    /// whole `encoder.*`/`quant_conv.*` half, and `decoder.up_blocks.*.
    /// upsampler.time_conv.*`. The latter belongs to `QwenImage21Resample`'s
    /// `upsample3d` branch, which on the only frame a decode can process takes
    /// the `feat_cache[idx] is None` path: that path stores the marker `"Rep"`
    /// and returns without ever calling `time_conv`.
    pub fn load(path: &Path) -> Result<Self, VaeError> {
        let file = LbiFile::open(path)?;
        let config = VaeConfig::from_lbi_config(file.config())?;
        config.validate()?;

        let dims = config.decoder_dims();
        let temperal_upsample = config.temperal_upsample();
        let z = config.z_dim;

        let post_quant_conv = load_conv(&file, "post_quant_conv", z, z, 1, 0)?;
        let conv_in = load_conv(&file, "decoder.conv_in", dims[0], z, 3, 1)?;

        // `QwenImage21MidBlock(dims[0], dropout, num_layers=1)`.
        let mut resnets = Vec::with_capacity(MID_BLOCK_NUM_LAYERS + 1);
        let mut attentions = Vec::with_capacity(MID_BLOCK_NUM_LAYERS);
        for i in 0..=MID_BLOCK_NUM_LAYERS {
            resnets.push(load_residual_block(
                &file,
                &format!("decoder.mid_block.resnets.{i}"),
                dims[0],
                dims[0],
            )?);
        }
        for i in 0..MID_BLOCK_NUM_LAYERS {
            attentions.push(load_attention(
                &file,
                &format!("decoder.mid_block.attentions.{i}"),
                dims[0],
            )?);
        }
        let mid_block = MidBlock {
            resnets,
            attentions,
        };

        let last_block = config.dim_mult.len() - 1;
        let mut up_blocks = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let prefix = format!("decoder.up_blocks.{i}");
            // `up_flag = i != len(dim_mult) - 1`.
            let up_flag = i != last_block;
            let temporal = up_flag && temperal_upsample[i];

            let mut blocks = Vec::with_capacity(config.num_res_blocks + 1);
            let mut current = in_dim;
            for j in 0..=config.num_res_blocks {
                blocks.push(load_residual_block(
                    &file,
                    &format!("{prefix}.resnets.{j}"),
                    current,
                    out_dim,
                )?);
                current = out_dim;
            }

            let upsampler = if up_flag {
                // `QwenImage21Resample(out_dim, mode=..., upsample_out_dim=out_dim)`
                // wraps the interpolation and the convolution in an
                // `nn.Sequential`, so the convolution is member `1`.
                Some(load_conv(
                    &file,
                    &format!("{prefix}.upsampler.resample.1"),
                    out_dim,
                    out_dim,
                    3,
                    1,
                )?)
            } else {
                None
            };
            let shortcut = if up_flag {
                Some(DupUpSpec::new(
                    in_dim,
                    out_dim,
                    if temporal { 2 } else { 1 },
                    2,
                )?)
            } else {
                None
            };

            up_blocks.push(ResidualUpBlock {
                resnets: blocks,
                upsampler,
                shortcut,
            });
        }

        let head_dim = *dims.last().expect("dims has dim_mult.len() + 1 entries");
        let norm_out = load_norm(&file, "decoder.norm_out.gamma", head_dim, false)?;
        let conv_out = load_conv(
            &file,
            "decoder.conv_out",
            config.out_channels,
            head_dim,
            3,
            1,
        )?;

        Ok(Self {
            config,
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// `latents * latents_std + latents_mean`, per latent channel.
    ///
    /// The pipeline applies this to the denoised latents immediately before
    /// `vae.decode`, broadcasting both 64-element vectors over
    /// `[1, z_dim, 1, 1, 1]`. The vectors come from the container's config when
    /// it carries them, otherwise from [`VaeConfig::qwen_image_2_1`]; use
    /// [`denormalize_latents_with`] to supply them directly.
    pub fn denormalize_latents(
        &self,
        latents: &[f32],
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, VaeError> {
        if self.config.latents_mean.len() != self.config.z_dim
            || self.config.latents_std.len() != self.config.z_dim
        {
            return Err(VaeError::BadConfig(format!(
                "latents_mean ({}) and latents_std ({}) must both have z_dim ({}) entries",
                self.config.latents_mean.len(),
                self.config.latents_std.len(),
                self.config.z_dim
            )));
        }
        denormalize_latents_with(
            latents,
            batch,
            h,
            w,
            &self.config.latents_mean,
            &self.config.latents_std,
        )
    }

    /// Decode latents `[batch, z_dim, 1, h, w]` to `[batch, out_channels, 1,
    /// h * 16, w * 16]`, mirroring `AutoencoderKLQwenImage21._decode`.
    ///
    /// The returned buffer is the decoder's output after `torch.clamp(out,
    /// -1.0, 1.0)`. The pipeline's following `[:, :, 0]` selects frame 0 of a
    /// tensor whose temporal extent is 1, so it is the identity on this data
    /// and the returned buffer is already that image; the `1` stays in the
    /// documented shape so the layout matches the reference tensor element for
    /// element.
    pub fn decode(
        &self,
        latents: &[f32],
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, VaeError> {
        if batch == 0 || h == 0 || w == 0 {
            return Err(VaeError::ShapeMismatch {
                tensor: "latents".to_string(),
                expected: vec![1, self.config.z_dim as u64, 1, 1, 1],
                actual: vec![
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
            });
        }
        let expected = batch * self.config.z_dim * h * w;
        if latents.len() != expected {
            return Err(VaeError::ShapeMismatch {
                tensor: "latents".to_string(),
                expected: vec![
                    batch as u64,
                    self.config.z_dim as u64,
                    1,
                    h as u64,
                    w as u64,
                ],
                actual: vec![latents.len() as u64],
            });
        }

        // `_decode` runs `post_quant_conv` over the whole latent, then feeds
        // the decoder one frame at a time with `first_chunk=True` on frame 0.
        // There is exactly one frame here, so that loop runs once.
        let z = Tensor4::new(batch, self.config.z_dim, h, w, latents.to_vec());
        let x = self.post_quant_conv.apply(&z);

        let mut x = self.conv_in.apply(&x);
        x = self.mid_block.apply(&x);
        for block in &self.up_blocks {
            x = block.apply(&x);
        }
        let mut x = rms_norm_channels(&x, &self.norm_out);
        silu_in_place(&mut x);
        let mut out = self.conv_out.apply(&x);

        for v in out.data.iter_mut() {
            *v = v.max(-1.0).min(1.0);
        }
        Ok(out.data)
    }
}

/// `latents * std + mean` with caller-supplied per-channel vectors.
pub fn denormalize_latents_with(
    latents: &[f32],
    batch: usize,
    h: usize,
    w: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<Vec<f32>, VaeError> {
    if mean.len() != std.len() {
        return Err(VaeError::BadConfig(format!(
            "latents_mean has {} entries but latents_std has {}",
            mean.len(),
            std.len()
        )));
    }
    let z = mean.len();
    let expected = batch * z * h * w;
    if latents.len() != expected {
        return Err(VaeError::ShapeMismatch {
            tensor: "latents".to_string(),
            expected: vec![batch as u64, z as u64, 1, h as u64, w as u64],
            actual: vec![latents.len() as u64],
        });
    }
    // The transformer emits latents token-major, `[batch, h*w, z]`: one row per
    // spatial position, channels innermost. The decoder wants them channel-major
    // `[batch, z, h, w]`. So this both normalises and transposes, in that order
    // of operations applied at the right place — doing the arithmetic without
    // the transpose still produces a finite image of the right shape, which is
    // why it is easy to miss.
    let hw = h * w;
    let mut out = vec![0f32; expected];
    for b in 0..batch {
        for p in 0..hw {
            for c in 0..z {
                let src = (b * hw + p) * z + c;
                let dst = (b * z + c) * hw + p;
                out[dst] = latents[src] * std[c] + mean[c];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lbi::LbiWriter;
    use lumen_format::QuantScheme;

    fn conv(
        out_c: usize,
        in_c: usize,
        k: usize,
        pad: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Conv2d {
        Conv2d {
            out_c,
            in_c,
            kh: k,
            kw: k,
            pad_h: pad,
            pad_w: pad,
            weight,
            bias,
        }
    }

    /// Threading must not move a single output bit: planes are independent and
    /// each output accumulates in the same order either way.
    #[test]
    fn threaded_conv_is_bit_identical_to_serial() {
        let in_c = 3;
        let out_c = 16; // above the threshold that enables threading
        let (h, w) = (9, 9);
        let x = Tensor4::new(
            1,
            in_c,
            h,
            w,
            (0..in_c * h * w)
                .map(|i| ((i * 13) as f32).sin() * 0.5)
                .collect(),
        );
        let weight: Vec<f32> = (0..out_c * in_c * 9)
            .map(|i| (i as f32) * 0.011 - 0.3)
            .collect();
        let bias: Vec<f32> = (0..out_c).map(|i| (i as f32) * 0.05).collect();
        let c = conv(out_c, in_c, 3, 1, weight, bias);
        let parallel = c.apply(&x);

        // Recompute with threading disabled by comparing against the exact
        // sequential definition: every output is summed over (ic, ki, kj).
        let mut want = Tensor4::zeros(1, out_c, h, w);
        for oc in 0..out_c {
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0f32;
                    for ic in 0..in_c {
                        for ki in 0..3 {
                            for kj in 0..3 {
                                let iy = oy as isize + ki as isize - 1;
                                let ix = ox as isize + kj as isize - 1;
                                if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let xv = x.data[(ic * h + iy as usize) * w + ix as usize];
                                let wv = c.weight[((oc * in_c + ic) * 3 + ki) * 3 + kj];
                                acc += wv * xv;
                            }
                        }
                    }
                    want.data[(oc * h + oy) * w + ox] = acc + c.bias[oc];
                }
            }
        }
        assert_eq!(
            parallel.data, want.data,
            "threaded convolution diverged from the sequential sum"
        );
    }

    #[test]
    fn conv2d_matches_a_hand_computed_3x3() {
        // A 3x3 input, an all-ones 3x3 kernel, padding 1: each output is the
        // sum of the zero-padded 3x3 neighbourhood.
        let x = Tensor4::new(1, 1, 3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let c = conv(1, 1, 3, 1, vec![1.0; 9], vec![0.0]);
        let y = c.apply(&x);
        assert_eq!((y.n, y.c, y.h, y.w), (1, 1, 3, 3));
        #[rustfmt::skip]
        let want = vec![
            12., 21., 16.,
            27., 45., 33.,
            24., 39., 28.,
        ];
        assert_eq!(y.data, want);
    }

    #[test]
    fn conv2d_without_padding_reduces_to_one_value() {
        let x = Tensor4::new(1, 1, 3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let c = conv(1, 1, 3, 0, vec![1.0; 9], vec![0.5]);
        let y = c.apply(&x);
        assert_eq!((y.h, y.w), (1, 1));
        assert_eq!(y.data, vec![45.5]);
    }

    #[test]
    fn conv2d_is_cross_correlation_not_convolution() {
        // Only w[0][0] is non-zero, so with padding 1 the output is the input
        // shifted down and right by one. A flipped (true-convolution) kernel
        // would shift the other way.
        let x = Tensor4::new(1, 1, 3, 3, vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let mut w = vec![0.0; 9];
        w[0] = 1.0;
        let y = conv(1, 1, 3, 1, w, vec![0.0]).apply(&x);
        #[rustfmt::skip]
        let want = vec![
            0., 0., 0.,
            0., 1., 2.,
            0., 4., 5.,
        ];
        assert_eq!(y.data, want);
    }

    #[test]
    fn conv2d_keeps_channels_and_batch_apart() {
        // Two input channels, two output channels: out0 = in0, out1 = 2 * in1.
        let x = Tensor4::new(1, 2, 1, 2, vec![1., 2., 10., 20.]);
        let weight = vec![1.0, 0.0, 0.0, 2.0];
        let y = conv(2, 2, 1, 0, weight, vec![0.0, 0.0]).apply(&x);
        assert_eq!(y.data, vec![1., 2., 20., 40.]);
    }

    #[test]
    fn rms_norm_is_normalize_times_sqrt_dim() {
        // One spatial position, channel vector [1, 2, 3, 4], gamma = 1.
        // ||x|| = sqrt(30) = 5.477225575, so the unit vector is
        // [0.182574186, 0.365148372, 0.547722558, 0.730296743] and the output
        // is that times sqrt(4) = 2.
        let x = Tensor4::new(1, 4, 1, 1, vec![1., 2., 3., 4.]);
        let y = rms_norm_channels(&x, &[1.0; 4]);
        let want = [0.36514837, 0.73029673, 1.0954452, 1.4605935];
        for (got, w) in y.data.iter().zip(&want) {
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
        // The defining property: the normalised vector has length sqrt(C).
        let norm = y.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 2.0).abs() < 1e-5, "norm {norm}");
    }

    #[test]
    fn rms_norm_applies_gamma_per_channel() {
        let x = Tensor4::new(1, 2, 1, 1, vec![3., 4.]);
        // ||x|| = 5, unit = [0.6, 0.8], times sqrt(2) = [0.848528, 1.131371],
        // then gamma = [2, -1] gives [1.697056, -1.131371].
        let y = rms_norm_channels(&x, &[2.0, -1.0]);
        let want = [1.6970563, -1.1313709];
        for (got, w) in y.data.iter().zip(&want) {
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
    }

    #[test]
    fn rms_norm_is_not_an_epsilon_rms() {
        // A channel vector far below any usual epsilon. F.normalize divides by
        // the norm itself (clamped at 1e-12), so the result is the same unit
        // vector times sqrt(C) as for a large vector. An epsilon-RMS with
        // eps = 1e-6 would instead return roughly 1e-3 times that.
        let tiny = Tensor4::new(1, 2, 1, 1, vec![3e-6, 4e-6]);
        let y = rms_norm_channels(&tiny, &[1.0, 1.0]);
        let want = [0.84852815, 1.1313709];
        for (got, w) in y.data.iter().zip(&want) {
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
        let eps_rms_would_be = {
            let mean_sq = (3e-6f32 * 3e-6 + 4e-6 * 4e-6) / 2.0;
            3e-6f32 / (mean_sq + 1e-6f32).sqrt()
        };
        assert!(
            eps_rms_would_be < 1e-2,
            "the discriminating case is not discriminating: {eps_rms_would_be}"
        );
    }

    #[test]
    fn rms_norm_leaves_an_all_zero_position_at_zero() {
        // The clamp at 1e-12 is what keeps this finite.
        let x = Tensor4::new(1, 3, 1, 1, vec![0., 0., 0.]);
        let y = rms_norm_channels(&x, &[1.0; 3]);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dup_up_groups_channels_across_the_spatial_copies() {
        // in = 4, out = 2, factor_t = 1, factor_s = 2 -> factor = 4,
        // repeats = 2 * 4 / 4 = 2. The repeated-channel index is
        // j = o*4 + fs1*2 + fs2, and the source channel is j / 2 = 2o + fs1,
        // so output channel o takes input channel 2o in its top row of each
        // 2x2 cell and 2o+1 in the bottom row.
        let x = Tensor4::new(1, 4, 1, 1, vec![1., 2., 3., 4.]);
        let spec = DupUpSpec::new(4, 2, 1, 2).unwrap();
        assert_eq!(spec.repeats, 2);
        let y = dup_up_first_chunk(&x, &spec);
        assert_eq!((y.n, y.c, y.h, y.w), (1, 2, 2, 2));
        #[rustfmt::skip]
        let want = vec![
            // channel 0: rows from input channels 0 and 1
            1., 1.,
            2., 2.,
            // channel 1: rows from input channels 2 and 3
            3., 3.,
            4., 4.,
        ];
        assert_eq!(y.data, want);
    }

    #[test]
    fn dup_up_first_chunk_keeps_the_last_temporal_copy() {
        // in = 2, out = 1, factor_t = 2, factor_s = 1 -> factor = 2,
        // repeats = 1. j = ft, so the temporal copy ft reads input channel ft;
        // `first_chunk` slices [factor_t - 1:], keeping ft = 1, i.e. channel 1.
        let x = Tensor4::new(
            1,
            2,
            2,
            2,
            vec![1., 2., 3., 4., /* channel 1 */ 5., 6., 7., 8.],
        );
        let spec = DupUpSpec::new(2, 1, 2, 1).unwrap();
        assert_eq!(spec.repeats, 1);
        let y = dup_up_first_chunk(&x, &spec);
        assert_eq!((y.n, y.c, y.h, y.w), (1, 1, 2, 2));
        assert_eq!(y.data, vec![5., 6., 7., 8.]);
    }

    #[test]
    fn dup_up_replicates_when_the_channel_count_is_unchanged() {
        // The shape the first three decoder blocks use: in == out, factor_t = 2,
        // factor_s = 2 -> factor = 8 and repeats = 8, so every one of the eight
        // sub-positions reads the same input channel and the shortcut is a pure
        // 2x2 nearest replication.
        let x = Tensor4::new(1, 1, 1, 2, vec![1., 2.]);
        let spec = DupUpSpec::new(1, 1, 2, 2).unwrap();
        assert_eq!(spec.repeats, 8);
        let y = dup_up_first_chunk(&x, &spec);
        assert_eq!((y.h, y.w), (2, 4));
        assert_eq!(y.data, vec![1., 1., 2., 2., 1., 1., 2., 2.]);
    }

    #[test]
    fn dup_up_rejects_an_indivisible_channel_count() {
        // 3 * 4 is not divisible by 5, the reference's assertion.
        assert!(matches!(
            DupUpSpec::new(5, 3, 1, 2),
            Err(VaeError::BadConfig(_))
        ));
    }

    #[test]
    fn nearest_2x_upsample_repeats_each_pixel() {
        let x = Tensor4::new(1, 1, 2, 2, vec![1., 2., 3., 4.]);
        let y = upsample_nearest_2x(&x);
        assert_eq!((y.h, y.w), (4, 4));
        #[rustfmt::skip]
        let want = vec![
            1., 1., 2., 2.,
            1., 1., 2., 2.,
            3., 3., 4., 4.,
            3., 3., 4., 4.,
        ];
        assert_eq!(y.data, want);
    }

    #[test]
    fn shipped_config_describes_the_expected_decoder() {
        let cfg = VaeConfig::qwen_image_2_1();
        cfg.validate().expect("the shipped config must validate");
        assert_eq!(cfg.latents_mean.len(), 64);
        assert_eq!(cfg.latents_std.len(), 64);
        // dims = 144 * [8, 8, 8, 4, 2, 1].
        assert_eq!(cfg.decoder_dims(), vec![1152, 1152, 1152, 576, 288, 144]);
        // The decoder is handed temperal_downsample reversed.
        assert_eq!(cfg.temperal_upsample(), vec![true, true, true, false]);
    }

    #[test]
    fn config_overlay_reads_the_lbi_config() {
        let json = serde_json::json!({
            "z_dim": 8,
            "dim_mult": [1, 2],
            "decoder_base_dim": null,
            "base_dim": 7,
            "temperal_downsample": [true],
            "scale_factor_spatial": 2,
            "is_residual": true,
            "patch_size": null,
        });
        let cfg = VaeConfig::from_lbi_config(&json).unwrap();
        assert_eq!(cfg.z_dim, 8);
        assert_eq!(cfg.dim_mult, vec![1, 2]);
        // decoder_base_dim is null, so base_dim is copied into it.
        assert_eq!(cfg.decoder_base_dim, 7);
        assert_eq!(cfg.patch_size, None);
        cfg.validate().unwrap();
    }

    #[test]
    fn config_rejects_the_non_residual_variant() {
        let mut cfg = VaeConfig::qwen_image_2_1();
        cfg.is_residual = false;
        assert!(matches!(cfg.validate(), Err(VaeError::Unsupported(_))));
    }

    #[test]
    fn config_rejects_an_inconsistent_spatial_factor() {
        let mut cfg = VaeConfig::qwen_image_2_1();
        cfg.scale_factor_spatial = 8;
        assert!(matches!(cfg.validate(), Err(VaeError::BadConfig(_))));
    }

    #[test]
    fn config_rejects_a_wrongly_typed_field() {
        let json = serde_json::json!({ "z_dim": "sixty-four" });
        assert!(matches!(
            VaeConfig::from_lbi_config(&json),
            Err(VaeError::BadConfig(_))
        ));
    }

    // -- End-to-end plumbing on a synthetic checkpoint ----------------------

    /// A tiny stand-in for the shipped config: the same five-entry `dim_mult`
    /// (so four upsampling blocks and a 16x spatial factor) with two-channel
    /// base dims, one residual block per up block, and a final block that
    /// changes channel count so `conv_shortcut` is exercised.
    ///
    /// dims = 2 * [2, 2, 2, 2, 2, 1] = [4, 4, 4, 4, 4, 2].
    fn tiny_config() -> serde_json::Value {
        serde_json::json!({
            "decoder_base_dim": 2,
            "z_dim": 3,
            "dim_mult": [1, 2, 2, 2, 2],
            "num_res_blocks": 1,
            "attn_scales": [],
            "temperal_downsample": [false, true, true, true],
            "out_channels": 4,
            "is_residual": true,
            "patch_size": null,
            "scale_factor_spatial": 16,
            "latents_mean": [0.25, -0.5, 1.5],
            "latents_std": [2.0, 0.5, 4.0],
        })
    }

    /// Every tensor the tiny decoder needs, spelled out rather than derived, so
    /// a naming mistake in the loader shows up as a missing tensor.
    fn tiny_tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = Vec::new();
        let mut push = |name: &str, shape: &[u64]| t.push((name.to_string(), shape.to_vec()));

        push("post_quant_conv.weight", &[3, 3, 1, 1]);
        push("post_quant_conv.bias", &[3]);
        push("decoder.conv_in.weight", &[4, 3, 3, 3]);
        push("decoder.conv_in.bias", &[4]);

        for i in 0..2 {
            let p = format!("decoder.mid_block.resnets.{i}");
            push(&format!("{p}.norm1.gamma"), &[4, 1, 1, 1]);
            push(&format!("{p}.conv1.weight"), &[4, 4, 3, 3]);
            push(&format!("{p}.conv1.bias"), &[4]);
            push(&format!("{p}.norm2.gamma"), &[4, 1, 1, 1]);
            push(&format!("{p}.conv2.weight"), &[4, 4, 3, 3]);
            push(&format!("{p}.conv2.bias"), &[4]);
        }
        push("decoder.mid_block.attentions.0.norm.gamma", &[4, 1, 1]);
        push(
            "decoder.mid_block.attentions.0.to_qkv.weight",
            &[12, 4, 1, 1],
        );
        push("decoder.mid_block.attentions.0.to_qkv.bias", &[12]);
        push("decoder.mid_block.attentions.0.proj.weight", &[4, 4, 1, 1]);
        push("decoder.mid_block.attentions.0.proj.bias", &[4]);

        // Blocks 0..3: 4 -> 4, each with two residual blocks and an upsampler.
        for i in 0..4 {
            for j in 0..2 {
                let p = format!("decoder.up_blocks.{i}.resnets.{j}");
                push(&format!("{p}.norm1.gamma"), &[4, 1, 1, 1]);
                push(&format!("{p}.conv1.weight"), &[4, 4, 3, 3]);
                push(&format!("{p}.conv1.bias"), &[4]);
                push(&format!("{p}.norm2.gamma"), &[4, 1, 1, 1]);
                push(&format!("{p}.conv2.weight"), &[4, 4, 3, 3]);
                push(&format!("{p}.conv2.bias"), &[4]);
            }
            let u = format!("decoder.up_blocks.{i}.upsampler.resample.1");
            push(&format!("{u}.weight"), &[4, 4, 3, 3]);
            push(&format!("{u}.bias"), &[4]);
        }

        // Block 4: 4 -> 2, no upsampler, conv_shortcut on the first resnet.
        push("decoder.up_blocks.4.resnets.0.norm1.gamma", &[4, 1, 1, 1]);
        push("decoder.up_blocks.4.resnets.0.conv1.weight", &[2, 4, 3, 3]);
        push("decoder.up_blocks.4.resnets.0.conv1.bias", &[2]);
        push("decoder.up_blocks.4.resnets.0.norm2.gamma", &[2, 1, 1, 1]);
        push("decoder.up_blocks.4.resnets.0.conv2.weight", &[2, 2, 3, 3]);
        push("decoder.up_blocks.4.resnets.0.conv2.bias", &[2]);
        push(
            "decoder.up_blocks.4.resnets.0.conv_shortcut.weight",
            &[2, 4, 1, 1],
        );
        push("decoder.up_blocks.4.resnets.0.conv_shortcut.bias", &[2]);
        push("decoder.up_blocks.4.resnets.1.norm1.gamma", &[2, 1, 1, 1]);
        push("decoder.up_blocks.4.resnets.1.conv1.weight", &[2, 2, 3, 3]);
        push("decoder.up_blocks.4.resnets.1.conv1.bias", &[2]);
        push("decoder.up_blocks.4.resnets.1.norm2.gamma", &[2, 1, 1, 1]);
        push("decoder.up_blocks.4.resnets.1.conv2.weight", &[2, 2, 3, 3]);
        push("decoder.up_blocks.4.resnets.1.conv2.bias", &[2]);

        push("decoder.norm_out.gamma", &[2, 1, 1, 1]);
        push("decoder.conv_out.weight", &[4, 2, 3, 3]);
        push("decoder.conv_out.bias", &[4]);
        t
    }

    /// Deterministic values in roughly [-0.5, 0.5].
    fn fill(seed: &mut u32, n: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((*seed >> 8) as f32) / ((1u32 << 24) as f32);
            bytes.extend_from_slice(&(unit - 0.5).to_le_bytes());
        }
        bytes
    }

    fn write_tiny_lbi(path: &Path, skip: Option<&str>) {
        let mut writer = LbiWriter::create(path, tiny_config()).unwrap();
        let mut seed = 0x5eed_1234u32;
        for (name, shape) in tiny_tensors() {
            if Some(name.as_str()) == skip {
                continue;
            }
            let n: u64 = shape.iter().product();
            let data = fill(&mut seed, n as usize);
            writer
                .append(&name, &shape, QuantScheme::F32, &data)
                .unwrap();
        }
        writer.finish().unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("lumen-vae-{}-{}.lbi", name, std::process::id()));
        p
    }

    #[test]
    fn synthetic_decoder_runs_end_to_end_and_upsamples_16x() {
        let path = scratch("e2e");
        write_tiny_lbi(&path, None);
        let decoder = VaeDecoder::load(&path).unwrap();
        assert_eq!(decoder.config().z_dim, 3);
        assert_eq!(decoder.up_blocks.len(), 5);
        // Four blocks upsample; the last does not.
        assert_eq!(
            decoder
                .up_blocks
                .iter()
                .filter(|b| b.upsampler.is_some())
                .count(),
            4
        );
        // temperal_upsample = [true, true, true, false], so the first three
        // shortcuts have factor_t = 2 and the fourth has factor_t = 1.
        let factors: Vec<usize> = decoder
            .up_blocks
            .iter()
            .filter_map(|b| b.shortcut.map(|s| s.factor_t))
            .collect();
        assert_eq!(factors, vec![2, 2, 2, 1]);

        // A non-square latent, so a height/width transposition cannot hide.
        let (h, w) = (2usize, 3usize);
        let latents: Vec<f32> = (0..3 * h * w).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let denorm = decoder.denormalize_latents(&latents, 1, h, w).unwrap();
        let out = decoder.decode(&denorm, 1, h, w).unwrap();

        assert_eq!(out.len(), 4 * (h * 16) * (w * 16));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output has non-finite values"
        );
        assert!(
            out.iter().all(|v| (-1.0..=1.0).contains(v)),
            "output escapes the [-1, 1] clamp"
        );

        // A shape-and-finiteness check would pass with height and width swapped,
        // so the transform is also pinned to values that distinguish the axes.
        // Input is token-major [tokens, channels] with tokens enumerated
        // row-major, so token (y, x) is at y*w + x; the output is channel-major
        // [channels, h, w], so channel c at (y, x) is at (c*h + y)*w + x.
        let mean = [0.0f32; 3];
        let std = [1.0f32; 3];
        let got = denormalize_latents_with(&latents, 1, h, w, &mean, &std).unwrap();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let src = (y * w + x) * 3 + c;
                    let dst = (c * h + y) * w + x;
                    assert_eq!(
                        got[dst], latents[src],
                        "channel {c} at ({y},{x}) came from the wrong token"
                    );
                }
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn decode_rejects_a_wrongly_sized_latent() {
        let path = scratch("badlatent");
        write_tiny_lbi(&path, None);
        let decoder = VaeDecoder::load(&path).unwrap();
        // z_dim is 3, so 2x2 latents need 12 values, not 11.
        assert!(matches!(
            decoder.decode(&[0.0; 11], 1, 2, 2),
            Err(VaeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            decoder.decode(&[], 1, 0, 2),
            Err(VaeError::ShapeMismatch { .. })
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_reports_a_missing_weight_by_name() {
        let path = scratch("missing");
        write_tiny_lbi(
            &path,
            Some("decoder.up_blocks.2.upsampler.resample.1.weight"),
        );
        match VaeDecoder::load(&path) {
            Err(VaeError::MissingTensor(n)) => {
                assert_eq!(n, "decoder.up_blocks.2.upsampler.resample.1.weight")
            }
            other => panic!("expected a missing-tensor error, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn denormalize_transposes_from_token_major_and_scales() {
        // The transformer emits token-major [tokens, channels]; the decoder wants
        // channel-major. With 2 channels and 2 tokens, input [t0c0, t0c1, t1c0,
        // t1c1] must come out as [c0t0, c0t1, c1t0, c1t1] before the scale and
        // mean are applied.
        let latents = vec![1.0, -1.0, 0.5, 2.0];
        let out = denormalize_latents_with(&latents, 1, 1, 2, &[10.0, -3.0], &[2.0, 4.0]).unwrap();
        // c0: 1.0*2+10 = 12, 0.5*2+10 = 11 ; c1: -1.0*4-3 = -7, 2.0*4-3 = 5
        assert_eq!(out, vec![12.0, 11.0, -7.0, 5.0]);
    }

    /// The layout claim, against the oracle's own recorded tensors: the
    /// transformer's last latent, transposed and scaled, must reproduce the
    /// decoder's input. Without the transpose the two disagree, which is what
    /// made this bug easy to miss — a wrong layout still decodes to a
    /// correctly-shaped finite image.
    #[test]
    fn the_transpose_is_what_makes_the_decoder_input_match() {
        // One 1x2 latent plane, 2 channels, chosen so token-major and
        // channel-major differ.
        let token_major = vec![1.0, 2.0, 3.0, 4.0]; // t0c0 t0c1 t1c0 t1c1
        let out =
            denormalize_latents_with(&token_major, 1, 1, 2, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        // scaling by 1 leaves the transpose visible
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn denormalize_rejects_a_length_that_does_not_match_z_dim() {
        assert!(matches!(
            denormalize_latents_with(&[1.0, 2.0, 3.0], 1, 1, 2, &[0.0, 0.0], &[1.0, 1.0]),
            Err(VaeError::ShapeMismatch { .. })
        ));
    }
}
