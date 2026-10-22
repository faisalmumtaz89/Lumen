//! The text-to-image pipeline: prompt in, image out.
//!
//! The sequence mirrors `QwenImage21Pipeline.__call__` for the text-to-image
//! case, where `true_cfg_scale` is 1 so there is one conditional pass per step
//! and no negative prompt:
//!
//! 1. render the prompt template and encode it
//! 2. run the text encoder over those ids, dropping the leading system tokens
//! 3. build the joint `img_mask`: the encoder's own image slots (none for
//!    text-to-image) plus one per 2x2 group of target latents
//! 4. run the denoising loop, feeding the DiT its own output each step
//! 5. denormalise the final latent and decode it
//!
//! The three components are loaded one at a time and dropped between uses: the
//! text encoder's language tower puts 12.9 GiB of BF16 weights on the device and
//! the transformer 13.3 GiB, so a generation never holds them together.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use crate::cuda::text_gpu::ResidentLayers;
use crate::dit::{Dit, DitConfig, DitForwardArgs};
use crate::lbi::LbiFile;
use crate::png::Rgba;
use crate::scheduler::{SchedulerConfig, SigmaSchedule};
use crate::tensor::Matrix;
use crate::text_encoder::{TextEncoder, SYSTEM_PREFIX_TOKENS};
use crate::tokenizer::Tokenizer;
use crate::vae::{VaeConfig, VaeDecoder};

/// Where the three converted components and the tokenizer live.
#[derive(Debug, Clone)]
pub struct PipelinePaths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub text_encoder: PathBuf,
    pub vocab: PathBuf,
    pub merges: PathBuf,
    pub added_tokens: Option<PathBuf>,
}

impl PipelinePaths {
    /// The default layout for a converted checkpoint beside its source.
    pub fn from_roots(lbi_dir: &Path, checkpoint_dir: &Path) -> Self {
        Self {
            transformer: lbi_dir.join("transformer.lbi"),
            vae: lbi_dir.join("vae.lbi"),
            text_encoder: lbi_dir.join("text_encoder.lbi"),
            vocab: checkpoint_dir.join("processor/vocab.json"),
            merges: checkpoint_dir.join("processor/merges.txt"),
            added_tokens: Some(checkpoint_dir.join("processor/added_tokens.json")),
        }
    }

    /// The part of [`check`](Self::check) that concerns the three converted
    /// components: whether a conversion is usable, the tokenizer files aside.
    pub fn check_components(&self, gpu: bool) -> Result<(), PipelineError> {
        let named = |path: &Path, e: &dyn std::fmt::Display| {
            PipelineError::Unsupported(format!("{}: {e}", path.display()))
        };
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let text =
            TextEncoder::load(&self.text_encoder).map_err(|e| named(&self.text_encoder, &e))?;
        #[cfg(feature = "cuda")]
        if gpu {
            let file =
                LbiFile::open(&self.text_encoder).map_err(|e| named(&self.text_encoder, &e))?;
            crate::cuda::text_gpu::TextGpu::check(&file, text.config())
                .map_err(|e| named(&self.text_encoder, &e))?;
        }

        let dit = DitConfig::qwen_image_2_1();
        let hidden = dit.inner_dim();
        let last = dit.num_layers - 1;
        let file = LbiFile::open(&self.transformer).map_err(|e| named(&self.transformer, &e))?;
        // `img_in` is a widening weight on the device and takes any float
        // storage; the block projections and `proj_out` are bf16 GEMM operands.
        for (name, shape, projection) in [
            (
                "img_in.weight".to_string(),
                [hidden, dit.in_channels],
                false,
            ),
            (
                "transformer_blocks.0.attn.to_q.weight".to_string(),
                [hidden, hidden],
                true,
            ),
            (
                format!("transformer_blocks.{last}.img_mlp.out.weight"),
                [hidden, dit.mlp_hidden()],
                true,
            ),
            (
                "proj_out.weight".to_string(),
                [dit.out_channels, hidden],
                true,
            ),
        ] {
            expect_tensor(&file, &name, &shape, gpu && projection)
                .map_err(|e| named(&self.transformer, &e))?;
        }

        let file = LbiFile::open(&self.vae).map_err(|e| named(&self.vae, &e))?;
        let vae = VaeConfig::from_lbi_config(file.config())
            .and_then(|c| c.validate().map(|()| c))
            .map_err(|e| named(&self.vae, &e))?;
        let z = vae.z_dim;
        let first = vae.decoder_dims()[0];
        for (name, shape) in [
            ("post_quant_conv.weight", [z, z, 1, 1]),
            ("decoder.conv_in.weight", [first, z, 3, 3]),
        ] {
            expect_tensor(&file, name, &shape, false).map_err(|e| named(&self.vae, &e))?;
        }

        Ok(())
    }

    /// Open every required file the way a generation will, without reading
    /// weights: the text encoder's whole tensor manifest (and, when `gpu`,
    /// everything the device text tower would refuse to load), the
    /// transformer's and the VAE's configuration plus the tensors that
    /// identify each container as that component at the expected shapes
    /// (stored as bf16 when `gpu`, which is what the device transformer
    /// multiplies), and the tokenizer with the chat template's markers as
    /// whole tokens. A checkpoint that fails here would fail every request.
    pub fn check(&self, gpu: bool) -> Result<(), PipelineError> {
        self.check_components(gpu)?;
        let tokenizer = Tokenizer::from_files_with_added(
            &self.vocab,
            &self.merges,
            self.added_tokens.as_deref(),
        )
        .map_err(|e| {
            PipelineError::Unsupported(format!(
                "tokenizer files under {}: {e}",
                self.vocab.parent().unwrap_or(&self.vocab).display()
            ))
        })?;
        for marker in ["<|im_start|>", "<|im_end|>"] {
            if tokenizer.encode(marker)?.len() != 1 {
                return Err(PipelineError::Unsupported(format!(
                    "{marker} is not a single token of the tokenizer under {}",
                    self.vocab.parent().unwrap_or(&self.vocab).display()
                )));
            }
        }
        #[cfg(feature = "cuda")]
        if gpu {
            check_device()?;
        }
        Ok(())
    }
}

/// The tensor `name` is present with exactly `shape`, stored as bf16 when
/// `bf16` (the device transformer's projections accept nothing else).
fn expect_tensor(file: &LbiFile, name: &str, shape: &[usize], bf16: bool) -> Result<(), String> {
    let entry = file
        .get(name)
        .ok_or_else(|| format!("tensor {name} is missing"))?;
    let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    if entry.shape != want {
        return Err(format!(
            "tensor {name} has shape {:?}, expected {want:?}",
            entry.shape
        ));
    }
    if bf16 && entry.quant != lumen_format::QuantScheme::Bf16 {
        return Err(format!(
            "tensor {name} is stored as {:?}, but the device transformer needs Bf16",
            entry.quant
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub enum PipelineError {
    Tokenizer(crate::tokenizer::TokenizerError),
    Text(crate::text_encoder::TextEncoderError),
    Dit(crate::dit::DitError),
    Vae(crate::vae::VaeError),
    Lbi(crate::lbi::LbiError),
    /// A request outside what this pipeline implements.
    Unsupported(String),
    /// The progress callback asked to stop.
    Cancelled,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizer(e) => write!(f, "tokenizer: {e}"),
            Self::Text(e) => write!(f, "text encoder: {e}"),
            Self::Dit(e) => write!(f, "transformer: {e}"),
            Self::Vae(e) => write!(f, "vae: {e}"),
            Self::Lbi(e) => write!(f, "lbi: {e}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<crate::tokenizer::TokenizerError> for PipelineError {
    fn from(e: crate::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(e)
    }
}
impl From<crate::text_encoder::TextEncoderError> for PipelineError {
    fn from(e: crate::text_encoder::TextEncoderError) -> Self {
        Self::Text(e)
    }
}
impl From<crate::dit::DitError> for PipelineError {
    fn from(e: crate::dit::DitError) -> Self {
        Self::Dit(e)
    }
}
impl From<crate::vae::VaeError> for PipelineError {
    fn from(e: crate::vae::VaeError) -> Self {
        Self::Vae(e)
    }
}
impl From<crate::lbi::LbiError> for PipelineError {
    fn from(e: crate::lbi::LbiError) -> Self {
        Self::Lbi(e)
    }
}

/// The system message the reference's template opens with.
pub const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// The shortest side a request may ask for: one latent tile.
pub const MIN_SIDE: usize = 32;

/// The longest side a request may ask for.
///
/// This is a memory bound, not an aesthetic one: the pipeline allocates a
/// `latents * channels` tensor per denoising step and a `latents * hidden`
/// activation inside the transformer, so the cost is quadratic in the side. At
/// 4096 the transformer's largest activation is 1.5 GiB and the decode runs in
/// bands when the whole image does not fit. Without a cap, one request can ask
/// for a tensor in the hundreds of gigabytes and the process dies rather than
/// answering.
pub const MAX_SIDE: usize = 4096;

/// The fewest denoising steps a request may ask for: with one step the
/// schedule's terminal stretch divides zero by zero (the reference's own
/// one-step schedule is `[NaN, 0]`), so no image is defined.
pub const MIN_STEPS: usize = 2;

/// The most denoising steps a request may ask for.
///
/// Each step is a full forward pass, so this bounds the request's duration
/// rather than its correctness; 200 is already generous beside the 40-50 the
/// model is used at.
pub const MAX_STEPS: usize = 200;

/// Parse `"WxH"` into `(width, height)`, each side within
/// [`MIN_SIDE`]..=[`MAX_SIDE`]. Sides are rounded down to a multiple of 32 by
/// the generation itself.
pub fn parse_size(size: &str) -> Result<(usize, usize), String> {
    let (w, h) = size
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("size {size:?} is not WxH"))?;
    let w: usize = w
        .trim()
        .parse()
        .map_err(|_| format!("size width {w:?} is not a number"))?;
    let h: usize = h
        .trim()
        .parse()
        .map_err(|_| format!("size height {h:?} is not a number"))?;
    if w < MIN_SIDE || h < MIN_SIDE {
        return Err(format!(
            "size {w}x{h} is below the {MIN_SIDE} pixel minimum side (one latent tile)"
        ));
    }
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!(
            "size {w}x{h} exceeds the {MAX_SIDE} pixel maximum side"
        ));
    }
    Ok((w, h))
}

/// Whether a step count is within [`MIN_STEPS`]..=[`MAX_STEPS`].
pub fn check_steps(steps: usize) -> Result<(), String> {
    if steps < MIN_STEPS {
        return Err(format!(
            "num_inference_steps must be at least {MIN_STEPS}; a one-step schedule is undefined"
        ));
    }
    if steps > MAX_STEPS {
        return Err(format!(
            "num_inference_steps {steps} exceeds the {MAX_STEPS} maximum"
        ));
    }
    Ok(())
}

/// Latent side length for a requested image side, given the VAE's 16x spatial
/// compression: the pipeline rounds to a multiple of 32 pixels and then divides.
pub fn latent_side(pixels: usize) -> usize {
    let multiple_of = 16 * 2;
    (pixels / multiple_of * multiple_of) / 16
}

/// Render the prompt template the checkpoint was trained on.
///
/// Built as a raw string rather than through the processor's chat template: the
/// two tokenize differently and the checkpoint expects this one.
pub fn render_prompt(prompt: &str) -> String {
    let p = if prompt.is_empty() { " " } else { prompt };
    format!(
        "<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Everything a generation needs.
pub struct GenerationRequest<'a> {
    pub prompt: &'a str,
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub seed: u64,
    /// When set, the initial latents instead of the ones `seed` would draw.
    ///
    /// This exists so a run can start from the reference's own noise: the two
    /// generators differ (ours is SplitMix64, the reference's is torch), so a
    /// shared seed gives two different pictures and no way to tell a numerical
    /// disagreement from a different random draw. Feeding the reference's noise
    /// removes that confound and leaves only the arithmetic.
    pub init_latents: Option<&'a [f32]>,
}

/// The latents denoising starts from: the caller's, checked for size, or
/// noise drawn from the seed.
fn starting_latents(
    req: &GenerationRequest<'_>,
    seq: usize,
    channels: usize,
) -> Result<Vec<f32>, PipelineError> {
    match req.init_latents {
        Some(given) => {
            if given.len() != seq * channels {
                return Err(PipelineError::Unsupported(format!(
                    "init_latents has {} values, expected {}",
                    given.len(),
                    seq * channels
                )));
            }
            Ok(given.to_vec())
        }
        None => Ok(initial_noise(seq, channels, req.seed)),
    }
}

/// The initial latent noise, drawn from the seed.
///
/// Its own generator rather than `rand`: one algorithm, pinned here, so the same
/// seed gives the same tensor on every platform and the oracle comparison is
/// meaningful.
pub fn initial_noise(seq: usize, channels: usize, seed: u64) -> Vec<f32> {
    // SplitMix64, then two uniforms per f32 via the Box-Muller transform.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(seq * channels);
    while out.len() < seq * channels {
        // Uniform in (0, 1], so the log is finite.
        let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((next() >> 11) as f64) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(seq * channels);
    out
}

/// Generate one image on the CPU.
///
/// Components are loaded and dropped in sequence so peak memory is the largest
/// single component rather than their sum.
///
/// `progress` is called with `(steps done, steps total)` once the prompt is
/// encoded and after every denoising step; `Break` stops the generation
/// there with [`PipelineError::Cancelled`].
pub fn generate_cpu(
    paths: &PipelinePaths,
    req: &GenerationRequest<'_>,
    progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<Rgba, PipelineError> {
    check_steps(req.steps).map_err(PipelineError::Unsupported)?;
    let lat_h = latent_side(req.height);
    let lat_w = latent_side(req.width);
    let seq = lat_h * lat_w;

    // 1-2. Encode the prompt.
    let tokenizer = Tokenizer::from_files_with_added(
        &paths.vocab,
        &paths.merges,
        paths.added_tokens.as_deref(),
    )?;
    let ids = tokenizer.encode(&render_prompt(req.prompt))?;
    let encoder = TextEncoder::load(&paths.text_encoder)?;
    let hidden = encoder.forward(&ids)?;
    drop(encoder);
    if hidden.rows < SYSTEM_PREFIX_TOKENS {
        return Err(PipelineError::Unsupported(format!(
            "the prompt encoded to {} tokens, fewer than the {} the template strips",
            hidden.rows, SYSTEM_PREFIX_TOKENS
        )));
    }
    let text_seq = hidden.rows - SYSTEM_PREFIX_TOKENS;
    let text = Matrix::new(
        text_seq,
        hidden.cols,
        hidden.data[SYSTEM_PREFIX_TOKENS * hidden.cols..].to_vec(),
    );

    // 3. The joint mask: no condition images for text-to-image, so the
    // encoder's slots are all text, and one slot is appended per 2x2 group of
    // target latents.
    let mut img_mask = vec![false; text_seq];
    img_mask.resize(text_seq + seq / 4, true);

    // 4-5. Denoise.
    if progress(0, req.steps).is_break() {
        return Err(PipelineError::Cancelled);
    }
    let dit = Dit::load(&paths.transformer)?;
    let cfg = DitConfig::qwen_image_2_1();
    let sched = SigmaSchedule::new(req.steps, seq, &SchedulerConfig::qwen_image_2_1());
    let mut latents = starting_latents(req, seq, cfg.in_channels)?;
    let shapes = [(1u64, lat_h as u64, lat_w as u64)];
    for step in 0..req.steps {
        let hidden_states = Matrix::new(seq, cfg.in_channels, latents.clone());
        let out = dit.forward(DitForwardArgs {
            hidden_states: &hidden_states,
            encoder_hidden_states: &text,
            timestep: sched.sigmas[step],
            img_shapes: &shapes,
            img_mask: &img_mask,
        })?;
        // Step 0 returns the joint sequence; later steps only the target.
        let pred = if out.rows > seq {
            out.data[(out.rows - seq) * out.cols..].to_vec()
        } else {
            out.data.clone()
        };
        latents = sched.step(step, &latents, &pred, false);
        if progress(step + 1, req.steps).is_break() {
            return Err(PipelineError::Cancelled);
        }
    }
    drop(dit);

    // 6. Decode.
    let vae = VaeDecoder::load(&paths.vae)?;
    let denorm = vae.denormalize_latents(&latents, 1, lat_h, lat_w)?;
    let decoded = vae.decode(&denorm, 1, lat_h, lat_w)?;
    Ok(Rgba::from_planar_rgba(lat_w * 16, lat_h * 16, &decoded))
}

// ---------------------------------------------------------------------------
// The GPU path
// ---------------------------------------------------------------------------

/// The CUDA device ordinal the GPU pipeline runs on.
pub const GPU_DEVICE: usize = 0;

/// The most device memory a generation needs, in bytes: 24,142 MiB, measured
/// as the device's used memory while the image server denoises a 4096x4096
/// image, the largest size a request may ask for, with an 8,192-byte prompt,
/// the longest it may send, on an RTX 5090 with a 10 ms `nvidia-smi` sampler.
/// The server keeps the VAE loaded while it denoises; [`generate_gpu`], which
/// loads one component at a time, peaks at 23,666 MiB there (21,672 MiB of live
/// allocations at once and 1,437 MiB of context, kernels and cuBLAS workspaces
/// outside the memory pool; the rest is pool memory a generation keeps for
/// reuse), and with a short prompt at 22,610 MiB for 4096x4096, 16,402 MiB for
/// 2048x2048 and 14,930 MiB for 1024x1024. The server's prompt encoding and
/// decoding go higher beside the resident transformer, but release it and run
/// again when they run out; denoising cannot, so a device with less total
/// memory could fail such a request, and the startup check refuses it instead.
pub const PEAK_DEVICE_BYTES: u64 = 24_142 << 20;

/// The compute capability the GPU pipeline needs: the text tower's flash
/// attention and the VAE's convolutions are built for it.
pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (8, 0);

/// The NVRTC target of the kernels built for [`MIN_COMPUTE_CAPABILITY`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const MIN_COMPUTE_TARGET: &str = "compute_80";

/// Refuse a device of compute capability `cc` below [`MIN_COMPUTE_CAPABILITY`],
/// or whose total memory is below [`PEAK_DEVICE_BYTES`]: one requirement for
/// every request size, which the server serves any of.
pub fn check_device_facts(cc: (i32, i32), total_bytes: u64) -> Result<(), PipelineError> {
    let (need_major, need_minor) = MIN_COMPUTE_CAPABILITY;
    if cc < MIN_COMPUTE_CAPABILITY {
        return Err(PipelineError::Unsupported(format!(
            "image generation needs a CUDA GPU of compute capability {need_major}.{need_minor} \
             or newer and the device has compute capability {}.{}",
            cc.0, cc.1
        )));
    }
    if total_bytes < PEAK_DEVICE_BYTES {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        return Err(PipelineError::Unsupported(format!(
            "image generation needs a CUDA device with at least {:.1} GiB of memory \
             ({PEAK_DEVICE_BYTES} bytes; the startup minimum, from the measured 4096x4096 peak) \
             and the device has {:.1} GiB ({total_bytes} bytes)",
            gib(PEAK_DEVICE_BYTES),
            gib(total_bytes),
        )));
    }
    Ok(())
}

/// Refuse a machine without CUDA device [`GPU_DEVICE`], or whose device
/// [`check_device_facts`] refuses.
#[cfg(feature = "cuda")]
pub fn check_device() -> Result<(), PipelineError> {
    let dev = lumen_runtime::cuda::ffi::CudaDevice::new(GPU_DEVICE)
        .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;
    let cc = dev
        .compute_capability()
        .map_err(|e| PipelineError::Unsupported(format!("device compute capability query: {e}")))?;
    let total = dev
        .total_memory()
        .map_err(|e| PipelineError::Unsupported(format!("device memory query: {e}")))?;
    check_device_facts(cc, total as u64)
}

/// Generate one image on the GPU.
///
/// Same sequence as [`generate_cpu`], with each component's device
/// implementation. The three components still load and free one at a time: the
/// text encoder's language tower puts 12.9 GiB of BF16 weights on the device and
/// the transformer 13.3 GiB, and the encoder's output is the only thing the
/// transformer consumes, so neither needs the other on the device.
#[cfg(feature = "cuda")]
pub fn generate_gpu(
    sources: &GpuSources,
    req: &GenerationRequest<'_>,
    progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<Rgba, PipelineError> {
    use lumen_runtime::cuda::ffi::CudaDevice;

    check_steps(req.steps).map_err(PipelineError::Unsupported)?;
    let dev = CudaDevice::new(GPU_DEVICE)
        .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;

    let ids = sources.tokenizer.encode(&render_prompt(req.prompt))?;
    let (text, _) = encode_prompt_gpu(sources, &dev, &ids, ResidentLayers::default(), None)?;
    let dit = sources.load_dit(&dev)?;
    let pool = PoolReuse::start(&dev);
    let latents = denoise_gpu(&dit, req, &text, progress);
    // The transformer's weights are freed before the pool is trimmed, on
    // every exit.
    drop(dit);
    let latents = latents?;
    // The VAE's kernels and cuBLAS handle load outside the pool.
    pool.release_unused()?;
    let vae = crate::cuda::vae_gpu::VaeGpu::load_with(&sources.vae, &dev, None)
        .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
    // A decode that does not fit is retried in twice as many bands.
    let most = decode_band_limit(req);
    let mut bands = 1;
    let image = loop {
        match decode_gpu(&vae, req, &latents, bands) {
            Ok(image) => break image,
            Err(e) if out_of_memory(&e) && bands < most => {
                bands = (bands * 2).min(most);
                pool.release_unused()?;
            }
            Err(e) => return Err(PipelineError::Unsupported(e)),
        }
    };
    drop(vae);
    pool.finish()?;
    Ok(image)
}

/// The CUDA pipeline's host side, opened once and kept for every generation:
/// the tokenizer and the three containers' mappings. A component loaded from
/// a mapping this process has already read reuses its page tables, so after
/// the first generation an upload runs at copy speed instead of faulting
/// every page in again. A container replaced on disk afterwards (a conversion
/// renames its new file into place) is not seen until the sources are opened
/// again.
///
/// [`GpuSources::pin_text_encoder`] adds page-locked host copies of the text
/// encoder, the one component every generation loads.
#[cfg(feature = "cuda")]
pub struct GpuSources {
    tokenizer: Tokenizer,
    text_encoder: LbiFile,
    text_pinned: Option<crate::cuda::text_gpu::PinnedMatrices>,
    transformer: LbiFile,
    vae: LbiFile,
}

#[cfg(feature = "cuda")]
impl GpuSources {
    pub fn open(paths: &PipelinePaths) -> Result<Self, PipelineError> {
        let open = |path: &Path| {
            LbiFile::open(path)
                .map_err(|e| PipelineError::Unsupported(format!("{}: {e}", path.display())))
        };
        Ok(Self {
            tokenizer: Tokenizer::from_files_with_added(
                &paths.vocab,
                &paths.merges,
                paths.added_tokens.as_deref(),
            )?,
            text_encoder: open(&paths.text_encoder)?,
            text_pinned: None,
            transformer: open(&paths.transformer)?,
            vae: open(&paths.vae)?,
        })
    }

    /// Copy the text encoder's layer matrices into page-locked host memory
    /// (12.9 GiB for Qwen-Image-2.1), from which every later load uploads them at the
    /// link's full rate. Page-locked memory is held for the sources' lifetime
    /// and is not bounded by the host's memlock or cgroup memory limits, so
    /// whether the host can spare it is the caller's decision.
    pub fn pin_text_encoder(&mut self) -> Result<(), PipelineError> {
        let config =
            crate::text_encoder::TextEncoderConfig::from_lbi_config(self.text_encoder.config())?;
        let ctx = cudarc::driver::CudaContext::new(GPU_DEVICE)
            .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;
        let pinned = crate::cuda::text_gpu::PinnedMatrices::copy(&self.text_encoder, &config, &ctx)
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
        self.text_pinned = Some(pinned);
        Ok(())
    }

    fn load_dit(
        &self,
        dev: &lumen_runtime::cuda::ffi::CudaDevice,
    ) -> Result<crate::cuda::dit_gpu::DitGpu, PipelineError> {
        crate::cuda::dit_gpu::DitGpu::load_with(&self.transformer, dev, DitConfig::qwen_image_2_1())
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))
    }
}

/// The transformer and the VAE held on the device between generations, as the
/// image-only server runs them (no text model shares its process).
///
/// The text encoder loads for each generation, but its first layers can stay:
/// how many fit beside the transformer, the VAE and a generation's activations
/// depends on the size (on a 32 GiB card all 36 stay beside 1024x1024
/// generations, fewer beside larger ones). The first
/// generation of a size keeps no text layers, and the device memory its
/// denoising and decoding use above that point is recorded (the peak of the
/// memory pool they allocate from). Later generations of that size keep as
/// many layers from the first as leave that much free plus
/// [`RESIDENT_TEXT_MARGIN`], and load only the rest; a prompt longer than the
/// recorded one reserves that much more in proportion to its text rows
/// ([`room_for`]) and records the size again, keeping the larger figure. A
/// denoising or decoding step that still runs out of memory drops the
/// kept layers and that size's figure and runs again, before anything else is
/// released. Encoding and decoding each run beside the
/// resident transformer when they fit; one that runs out of device memory
/// there, as any encode does on a card much smaller than 32 GiB, is retried
/// once with the transformer
/// released, and the next generation loads it again. The smallest prompt and
/// the smallest image that ran out are remembered, so work at least that large
/// releases the transformer first instead of failing again.
#[cfg(feature = "cuda")]
pub struct GpuResident {
    dev: lumen_runtime::cuda::ffi::CudaDevice,
    sources: GpuSources,
    dit: Option<crate::cuda::dit_gpu::DitGpu>,
    vae: crate::cuda::vae_gpu::VaeGpu,
    /// Tokens of the smallest prompt whose encoding ran out of memory beside
    /// the transformer.
    encode_needs_room: Option<usize>,
    /// Pixels of the smallest image whose decode ran out of memory beside the
    /// transformer.
    decode_needs_room: Option<usize>,
    /// The text encoder's first layers, kept between generations.
    text_layers: ResidentLayers,
    /// Per image size, the longest prompt measured and the most device bytes
    /// the denoising and decoding of that prompt or a shorter one measured
    /// before it took above what was allocated when they began.
    room: HashMap<(usize, usize), (usize, usize)>,
    /// Per image size, the bands its decode last needed to fit.
    decode_bands: HashMap<(usize, usize), usize>,
}

#[cfg(feature = "cuda")]
impl GpuResident {
    /// Load the transformer and the VAE onto [`GPU_DEVICE`].
    pub fn load(sources: GpuSources) -> Result<Self, PipelineError> {
        let dev = lumen_runtime::cuda::ffi::CudaDevice::new(GPU_DEVICE)
            .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;
        let dit = sources.load_dit(&dev)?;
        let vae = crate::cuda::vae_gpu::VaeGpu::load_with(&sources.vae, &dev, None)
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
        Ok(Self {
            dev,
            sources,
            dit: Some(dit),
            vae,
            encode_needs_room: None,
            decode_needs_room: None,
            text_layers: ResidentLayers::default(),
            room: HashMap::new(),
            decode_bands: HashMap::new(),
        })
    }

    /// Generate one image with the resident components; the same sequence and
    /// arithmetic as [`generate_gpu`]. A denoising pass retried after running
    /// out of memory beside kept text layers reports its progress from 0 again.
    pub fn generate(
        &mut self,
        req: &GenerationRequest<'_>,
        progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Rgba, PipelineError> {
        check_steps(req.steps).map_err(PipelineError::Unsupported)?;
        let ids = self.sources.tokenizer.encode(&render_prompt(req.prompt))?;
        if self.encode_needs_room.is_some_and(|t| ids.len() >= t) {
            self.release_transformer();
        }
        let size = (req.width, req.height);
        // Layers are kept only beside a resident transformer: the room was
        // measured with it loaded, and the free memory of a device without it
        // would count the space its reload needs.
        let recorded = self.room.get(&size).copied();
        let room = recorded
            .filter(|_| self.dit.is_some())
            .and_then(|(tokens, bytes)| room_for(bytes, tokens, ids.len()));
        let resident = std::mem::take(&mut self.text_layers);
        let (text, kept) = match encode_prompt_gpu(&self.sources, &self.dev, &ids, resident, room) {
            Ok(encoded) => encoded,
            Err(e) if self.dit.is_some() && is_out_of_memory(&e) => {
                self.release_transformer();
                self.encode_needs_room = Some(smallest(self.encode_needs_room, ids.len()));
                encode_prompt_gpu(
                    &self.sources,
                    &self.dev,
                    &ids,
                    ResidentLayers::default(),
                    None,
                )?
            }
            Err(e) => return Err(e),
        };
        self.text_layers = kept;

        // A size not yet measured runs with no text layers kept, and a prompt
        // longer than the measured one with the scaled room; either records the
        // size's room, from a run with denoising steps that keeps the
        // transformer and its text layers throughout, so the figure is what the
        // same run needs next time. The pool's growth above the start does not
        // depend on the layers kept. The measurement starts once every free
        // queued so far has taken effect.
        // A device whose pool cannot be read is never measured, so it keeps
        // no text layers.
        let longer = recorded.map_or(true, |(tokens, _)| ids.len() > tokens);
        let mut measure_from = match &self.dit {
            Some(_) if req.steps > 0 && longer => {
                synchronize(&self.dev)?;
                memory_pool_reset(&self.dev).ok()
            }
            _ => None,
        };
        let dit = match self.dit.take() {
            Some(dit) => dit,
            None => self.sources.load_dit(&self.dev)?,
        };
        let pool = PoolReuse::start(&self.dev);
        let mut latents = denoise_gpu(&dit, req, &text, progress);
        // Kept layers are only ever an estimate's worth: a step that runs out
        // beside them drops them and the estimate, and runs again.
        if matches!(&latents, Err(e) if is_out_of_memory(e)) && !self.text_layers.is_empty() {
            measure_from = None;
            if let Err(e) = self.forget_room(size).and_then(|()| pool.release_unused()) {
                self.dit = Some(dit);
                return Err(e);
            }
            latents = denoise_gpu(&dit, req, &text, progress);
        }
        self.dit = Some(dit);
        let latents = latents?;
        let pixels = req.width * req.height;
        if self.decode_needs_room.is_some_and(|p| pixels >= p) {
            self.release_transformer();
            pool.release_unused()?;
        }
        // A decode that does not fit drops the kept text layers first, then
        // releases the transformer, and only then runs in twice as many bands:
        // a whole-image decode plus reloading the transformer for the next
        // generation takes less time than a banded decode beside it, so bands
        // are for images that do not fit even on their own.
        let most = decode_band_limit(req);
        let mut bands = self.decode_bands.get(&size).copied().unwrap_or(1);
        let image = loop {
            match decode_gpu(&self.vae, req, &latents, bands) {
                Ok(image) => break image,
                Err(e) if out_of_memory(&e) && !self.text_layers.is_empty() => {
                    measure_from = None;
                    self.forget_room(size)?;
                }
                Err(e) if out_of_memory(&e) && self.dit.is_some() => {
                    self.release_transformer();
                    self.decode_needs_room = Some(smallest(self.decode_needs_room, pixels));
                }
                Err(e) if out_of_memory(&e) && bands < most => {
                    bands = (bands * 2).min(most);
                    self.decode_bands.insert(size, bands);
                }
                Err(e) => return Err(PipelineError::Unsupported(e)),
            }
            // What the failed attempt and the release freed goes back to the
            // driver before the retry.
            pool.release_unused()?;
        };
        if let (Some(start), Some(_)) = (measure_from, &self.dit) {
            // A peak that cannot be read leaves the size unmeasured; the image
            // stands either way.
            if let Ok(peak) = memory_pool_peak(&self.dev) {
                let bytes = peak.saturating_sub(start);
                let bytes = recorded.map_or(bytes, |(_, earlier)| bytes.max(earlier));
                self.room.insert(size, (ids.len(), bytes));
            }
        }
        pool.finish()?;
        Ok(image)
    }

    /// Drop the kept text layers and the room figure that kept them, once
    /// their frees have taken effect.
    fn forget_room(&mut self, size: (usize, usize)) -> Result<(), PipelineError> {
        self.text_layers = ResidentLayers::default();
        self.room.remove(&size);
        synchronize(&self.dev)
    }

    /// Free the transformer, and the text layers kept beside it, for work that
    /// does not fit beside them.
    fn release_transformer(&mut self) {
        self.dit = None;
        self.text_layers = ResidentLayers::default();
    }
}

/// How the driver and cuBLAS report a device allocation they could not make;
/// the errors reach here as text.
#[cfg(feature = "cuda")]
const OUT_OF_MEMORY: [&str; 2] = ["CUDA_ERROR_OUT_OF_MEMORY", "CUBLAS_STATUS_ALLOC_FAILED"];

#[cfg(feature = "cuda")]
fn out_of_memory(text: &str) -> bool {
    OUT_OF_MEMORY.iter().any(|m| text.contains(m))
}

#[cfg(feature = "cuda")]
fn is_out_of_memory(e: &PipelineError) -> bool {
    out_of_memory(&format!("{e}"))
}

/// The room a generation of `tokens` prompt ids needs at a size where one of
/// `measured` ids took `bytes`: the same for a prompt no longer, otherwise
/// scaled by the ratio of their text rows (the ids after the stripped system
/// prefix), rounded up. Every allocation of a generation at a fixed size grows
/// at most linearly in its text rows, so the ratio bounds the longer prompt's
/// need; a measurement without text rows bounds no longer prompt.
#[cfg(feature = "cuda")]
fn room_for(bytes: usize, measured: usize, tokens: usize) -> Option<usize> {
    let text_rows = |ids: usize| ids.saturating_sub(SYSTEM_PREFIX_TOKENS) as u128;
    let (measured, tokens) = (text_rows(measured), text_rows(tokens));
    if tokens <= measured {
        return Some(bytes);
    }
    if measured == 0 {
        return None;
    }
    usize::try_from((bytes as u128 * tokens).div_ceil(measured)).ok()
}

/// The smaller of a remembered threshold and a new one.
#[cfg(feature = "cuda")]
fn smallest(known: Option<usize>, new: usize) -> usize {
    known.map_or(new, |k| k.min(new))
}

/// The rendered prompt's hidden states, with the template's system prefix
/// stripped. The encoder loads for the call on top of `resident`, its layers
/// kept from an earlier call. Given `room`, the bytes the rest of the
/// generation needs free, it keeps as many layers from the first as leave that
/// much plus [`RESIDENT_TEXT_MARGIN`] and returns them; otherwise it frees them
/// all.
#[cfg(feature = "cuda")]
fn encode_prompt_gpu(
    sources: &GpuSources,
    dev: &lumen_runtime::cuda::ffi::CudaDevice,
    ids: &[u32],
    resident: ResidentLayers,
    room: Option<usize>,
) -> Result<(Matrix, ResidentLayers), PipelineError> {
    let unsupported = |e: String| PipelineError::Unsupported(e);
    let config =
        crate::text_encoder::TextEncoderConfig::from_lbi_config(sources.text_encoder.config())?;
    let encoder = crate::cuda::text_gpu::TextGpu::load_with(
        &sources.text_encoder,
        sources.text_pinned.as_ref(),
        dev,
        config,
        resident,
    )
    .map_err(|e| unsupported(format!("{e}")))?;
    let hidden = match encoder.forward(ids) {
        Ok(hidden) => hidden,
        Err(e) => {
            // Layers kept from an earlier load are freed on the stream that
            // uploaded them, which does not wait for this one: let its queued
            // work finish before the encoder drops them.
            synchronize(dev)?;
            return Err(unsupported(format!("{e}")));
        }
    };
    let kept = match room {
        Some(room) => {
            // Every transient of the forward is freed once the device is idle,
            // so what is free then plus the layers is what keeping none leaves.
            synchronize(dev)?;
            let free = dev.free_memory().map_err(|e| unsupported(format!("{e}")))?;
            let budget = (free + encoder.layer_bytes()).saturating_sub(room + RESIDENT_TEXT_MARGIN);
            let kept = encoder.into_resident(budget);
            // The layers not kept are freed on the encoder's stream; let those
            // frees land before the denoising steps allocate.
            synchronize(dev)?;
            kept
        }
        None => ResidentLayers::default(),
    };
    if hidden.rows < SYSTEM_PREFIX_TOKENS {
        return Err(PipelineError::Unsupported(format!(
            "the prompt encoded to {} tokens, fewer than the {} the template strips",
            hidden.rows, SYSTEM_PREFIX_TOKENS
        )));
    }
    let text = Matrix::new(
        hidden.rows - SYSTEM_PREFIX_TOKENS,
        hidden.cols,
        hidden.data[SYSTEM_PREFIX_TOKENS * hidden.cols..].to_vec(),
    );
    Ok((text, kept))
}

/// Device memory left unplanned beside the text layers a generation keeps: the
/// pool's rounding and allocations made outside it.
#[cfg(feature = "cuda")]
pub const RESIDENT_TEXT_MARGIN: usize = 512 << 20;

/// Wait for every stream on the device, so queued work and frees have taken
/// effect.
#[cfg(feature = "cuda")]
fn synchronize(dev: &lumen_runtime::cuda::ffi::CudaDevice) -> Result<(), PipelineError> {
    dev.ctx
        .synchronize()
        .map_err(|e| PipelineError::Unsupported(format!("synchronize: {e}")))
}

/// Reset the device memory pool's high-water mark to what is allocated now,
/// and return that amount.
#[cfg(feature = "cuda")]
fn memory_pool_reset(dev: &lumen_runtime::cuda::ffi::CudaDevice) -> Result<usize, PipelineError> {
    use cudarc::driver::sys::CUmemPool_attribute as Attr;
    let pool = memory_pool(dev)?;
    let zero: u64 = 0;
    // Safety: the pool is the device's current pool, and the attribute takes a
    // `cuuint64_t`.
    unsafe {
        cudarc::driver::sys::cuMemPoolSetAttribute(
            pool,
            Attr::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            &zero as *const u64 as *mut std::ffi::c_void,
        )
        .result()
        .map_err(|e| PipelineError::Unsupported(format!("memory pool: {e}")))?;
    }
    memory_pool_attr(pool, Attr::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
}

/// The most the device memory pool has had allocated since its last reset.
#[cfg(feature = "cuda")]
fn memory_pool_peak(dev: &lumen_runtime::cuda::ffi::CudaDevice) -> Result<usize, PipelineError> {
    let pool = memory_pool(dev)?;
    memory_pool_attr(
        pool,
        cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    )
}

/// The device memory pool kept whole while a generation runs.
///
/// The pool's default release threshold is zero: at every synchronize it hands
/// the memory freed since back to the driver, and the next allocations map it
/// again. A denoising step synchronizes several times and the decode allocates
/// gigabytes, so from just before the denoising steps to the end of the decode
/// the threshold is raised and that memory is reused instead. Afterwards the
/// threshold is restored and the pool trimmed, handing its unused memory back
/// to the driver before the next prompt encode reads free memory.
///
/// The pool belongs to the device, not to one pipeline: this assumes one
/// generation at a time on the device, as the image server runs them. On a
/// device without memory pools, or whose pool cannot be read, nothing is
/// kept.
#[cfg(feature = "cuda")]
struct PoolReuse {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    /// `None` on a device without memory pools, or once finished.
    pool: Option<cudarc::driver::sys::CUmemoryPool>,
}

#[cfg(feature = "cuda")]
impl PoolReuse {
    fn start(dev: &lumen_runtime::cuda::ffi::CudaDevice) -> Self {
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        let supported = dev
            .ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
            .is_ok_and(|v| v != 0);
        let pool = supported
            .then(|| memory_pool(dev).ok())
            .flatten()
            .filter(|&pool| set_release_threshold(pool, u64::MAX).is_ok());
        Self {
            ctx: dev.ctx.clone(),
            pool,
        }
    }

    /// Hand the pool's unused memory back to the driver once every queued
    /// free has taken effect, before a retry that needs what was released.
    fn release_unused(&self) -> Result<(), PipelineError> {
        let Some(pool) = self.pool else {
            return Ok(());
        };
        self.ctx
            .synchronize()
            .map_err(|e| PipelineError::Unsupported(format!("synchronize: {e}")))?;
        // Safety: the pool is the device's current pool, alive with `ctx`.
        unsafe { cudarc::driver::sys::cuMemPoolTrimTo(pool, 0) }
            .result()
            .map_err(|e| PipelineError::Unsupported(format!("memory pool: {e}")))
    }

    /// Restore the release threshold and trim the pool, once the generation's
    /// device buffers are dropped. A failure leaves the guard armed, so
    /// dropping it tries again.
    fn finish(mut self) -> Result<(), PipelineError> {
        let result = self.restore();
        if result.is_ok() {
            self.pool = None;
        }
        result
    }

    /// Both steps are attempted; the first failure is reported.
    fn restore(&self) -> Result<(), PipelineError> {
        let threshold = self
            .pool
            .map_or(Ok(()), |pool| set_release_threshold(pool, 0));
        let trim = self.release_unused();
        threshold.and(trim)
    }
}

#[cfg(feature = "cuda")]
impl Drop for PoolReuse {
    /// A generation that ends early (an error, a cancellation, a panic)
    /// restores the pool all the same; there is nothing to report a failure to.
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(feature = "cuda")]
fn set_release_threshold(
    pool: cudarc::driver::sys::CUmemoryPool,
    bytes: u64,
) -> Result<(), PipelineError> {
    // Safety: the attribute takes a `cuuint64_t`.
    unsafe {
        cudarc::driver::sys::cuMemPoolSetAttribute(
            pool,
            cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &bytes as *const u64 as *mut std::ffi::c_void,
        )
    }
    .result()
    .map_err(|e| PipelineError::Unsupported(format!("memory pool: {e}")))
}

/// The device's current memory pool, the one every stream-ordered allocation
/// on it draws from.
#[cfg(feature = "cuda")]
fn memory_pool(
    dev: &lumen_runtime::cuda::ffi::CudaDevice,
) -> Result<cudarc::driver::sys::CUmemoryPool, PipelineError> {
    let mut pool = std::ptr::null_mut();
    // Safety: `pool` is written by the call.
    unsafe { cudarc::driver::sys::cuDeviceGetMemPool(&mut pool, dev.ctx.cu_device()) }
        .result()
        .map_err(|e| PipelineError::Unsupported(format!("memory pool: {e}")))?;
    Ok(pool)
}

#[cfg(feature = "cuda")]
fn memory_pool_attr(
    pool: cudarc::driver::sys::CUmemoryPool,
    attr: cudarc::driver::sys::CUmemPool_attribute,
) -> Result<usize, PipelineError> {
    let mut value: u64 = 0;
    // Safety: the attribute is a `cuuint64_t`, written by the call.
    unsafe {
        cudarc::driver::sys::cuMemPoolGetAttribute(
            pool,
            attr,
            &mut value as *mut u64 as *mut std::ffi::c_void,
        )
    }
    .result()
    .map_err(|e| PipelineError::Unsupported(format!("memory pool: {e}")))?;
    Ok(value as usize)
}

/// The denoising loop: the final latents for `req`, conditioned on `text`.
///
/// The latents are bf16, as the reference keeps them: the starting latents are
/// rounded to bf16, and each scheduler step rounds as `SigmaSchedule::step`
/// documents for a bf16 model output. The transformer takes the timestep the
/// reference hands it, `SigmaSchedule::model_timestep`.
#[cfg(feature = "cuda")]
fn denoise_gpu(
    dit: &crate::cuda::dit_gpu::DitGpu,
    req: &GenerationRequest<'_>,
    text: &Matrix,
    progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<Vec<f32>, PipelineError> {
    let lat_h = latent_side(req.height);
    let lat_w = latent_side(req.width);
    let seq = lat_h * lat_w;
    // The joint mask: the text rows, then one slot per 2x2 latent patch.
    let mut img_mask = vec![false; text.rows];
    img_mask.resize(text.rows + seq / 4, true);
    if progress(0, req.steps).is_break() {
        return Err(PipelineError::Cancelled);
    }
    let cfg = DitConfig::qwen_image_2_1();
    let sched = SigmaSchedule::new(req.steps, seq, &SchedulerConfig::qwen_image_2_1());
    let mut latents: Vec<f32> = starting_latents(req, seq, cfg.in_channels)?
        .into_iter()
        .map(crate::tensor::bf16_round)
        .collect();
    let shapes = [(1u64, lat_h as u64, lat_w as u64)];
    for step in 0..req.steps {
        let hidden_states = Matrix::new(seq, cfg.in_channels, latents.clone());
        let out = dit
            .forward(DitForwardArgs {
                hidden_states: &hidden_states,
                encoder_hidden_states: text,
                timestep: sched.model_timestep(step),
                img_shapes: &shapes,
                img_mask: &img_mask,
            })
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
        let pred = if out.rows > seq {
            out.data[(out.rows - seq) * out.cols..].to_vec()
        } else {
            out.data.clone()
        };
        latents = sched.step(step, &latents, &pred, true);
        if progress(step + 1, req.steps).is_break() {
            return Err(PipelineError::Cancelled);
        }
    }
    Ok(latents)
}

/// The most bands a decode is split into: bands of at least 32 latent rows
/// (512 image rows), which the shipped decoder's context rows on either side
/// ([`crate::cuda::vae_gpu::VaeGpu::halo_rows`]) at most double.
#[cfg(feature = "cuda")]
fn decode_band_limit(req: &GenerationRequest<'_>) -> usize {
    (latent_side(req.height) / 32).max(1)
}

/// The image the final latents decode to.
///
/// The decoder runs in `bands` horizontal bands, with the same result as one
/// ([`crate::cuda::vae_gpu::VaeGpu::decode_in_bands`]).
#[cfg(feature = "cuda")]
fn decode_gpu(
    vae: &crate::cuda::vae_gpu::VaeGpu,
    req: &GenerationRequest<'_>,
    latents: &[f32],
    bands: usize,
) -> Result<Rgba, String> {
    let lat_h = latent_side(req.height);
    let lat_w = latent_side(req.width);
    let denorm = crate::vae::denormalize_latents_bf16(
        latents,
        1,
        lat_h,
        lat_w,
        &vae.config().latents_mean,
        &vae.config().latents_std,
    )
    .map_err(|e| format!("{e}"))?;
    let decoded = vae
        .decode_in_bands(&denorm, 1, lat_h, lat_w, bands)
        .map_err(|e| format!("{e}"))?;
    Ok(Rgba::from_planar_rgba(lat_w * 16, lat_h * 16, &decoded))
}

#[cfg(test)]
mod request_limit_tests {
    use super::{check_steps, parse_size};

    #[test]
    fn a_request_size_is_bounded_on_both_sides_and_steps_are_bounded() {
        assert_eq!(parse_size("1024x1024"), Ok((1024, 1024)));
        assert_eq!(parse_size(" 1536 X 1024 "), Ok((1536, 1024)));
        assert_eq!(parse_size("32x32"), Ok((32, 32)));
        assert_eq!(parse_size("4096x4096"), Ok((4096, 4096)));
        assert!(parse_size("31x1024").unwrap_err().contains("minimum"));
        assert!(parse_size("4097x100").unwrap_err().contains("maximum"));
        assert!(parse_size("1024").unwrap_err().contains("not WxH"));
        assert!(parse_size("ax1024").unwrap_err().contains("not a number"));
        assert!(parse_size("1024x").unwrap_err().contains("not a number"));
        assert!(parse_size("-1024x1024")
            .unwrap_err()
            .contains("not a number"));
        assert_eq!(check_steps(2), Ok(()));
        assert_eq!(check_steps(200), Ok(()));
        assert!(check_steps(1).unwrap_err().contains("at least 2"));
        assert!(check_steps(0).unwrap_err().contains("at least 2"));
    }

    /// The generation entry points refuse a step count outside the limits
    /// before they touch any file, whoever calls them.
    #[test]
    fn generation_refuses_a_step_count_outside_the_limits() {
        let missing = std::path::PathBuf::from("/nonexistent");
        let paths = super::PipelinePaths {
            transformer: missing.clone(),
            vae: missing.clone(),
            text_encoder: missing.clone(),
            vocab: missing.clone(),
            merges: missing,
            added_tokens: None,
        };
        for steps in [0, 1, 201] {
            let req = super::GenerationRequest {
                prompt: "x",
                height: 256,
                width: 256,
                steps,
                seed: 0,
                init_latents: None,
            };
            let err =
                super::generate_cpu(
                    &paths,
                    &req,
                    &mut |_, _| std::ops::ControlFlow::Continue(()),
                )
                .unwrap_err()
                .to_string();
            assert!(err.contains("num_inference_steps"), "{steps}: {err}");
        }
        assert!(check_steps(201).unwrap_err().contains("maximum"));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod room_tests {
    use super::{room_for, SYSTEM_PREFIX_TOKENS};

    /// A prompt no longer than the measured one needs the measured room; a
    /// longer one that room times the ratio of text rows, rounded up; a
    /// measurement without text rows covers no longer prompt.
    #[test]
    fn a_longer_prompt_needs_the_measured_room_scaled_by_its_text_rows() {
        let p = SYSTEM_PREFIX_TOKENS;
        assert_eq!(room_for(1000, p + 26, p + 26), Some(1000));
        assert_eq!(room_for(1000, p + 26, p + 3), Some(1000));
        assert_eq!(room_for(1000, p + 26, p + 28), Some(1077));
        assert_eq!(room_for(1000, p + 20, p + 40), Some(2000));
        assert_eq!(
            room_for(3 << 30, p + 26, p + 300),
            Some(((3u128 << 30) * 300).div_ceil(26) as usize)
        );
        assert_eq!(room_for(1000, p, p), Some(1000));
        assert_eq!(room_for(1000, p, p + 1), None);
        assert_eq!(room_for(1000, p - 2, p), Some(1000));
        assert_eq!(room_for(usize::MAX, p + 1, p + 2), None);
    }
}

#[cfg(test)]
mod device_facts_tests {
    use super::{check_device_facts, PEAK_DEVICE_BYTES};

    /// A device below the measured peak is refused with both amounts in the
    /// message, and no request size in it; one at or above it passes.
    #[test]
    fn a_small_device_is_refused_with_both_amounts_named() {
        let err = check_device_facts((8, 0), 12 << 30)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("12.0 GiB") && err.contains(&(12u64 << 30).to_string()),
            "{err}"
        );
        assert!(
            err.contains("23.6 GiB") && err.contains(&PEAK_DEVICE_BYTES.to_string()),
            "{err}"
        );
        assert!(
            err.contains("at least 23.6 GiB") && err.contains("the device has 12.0 GiB"),
            "{err}"
        );
        assert!(err.contains("startup minimum"), "{err}");
        assert!(!err.contains("a generation at"), "{err}");
        assert!(check_device_facts((8, 0), PEAK_DEVICE_BYTES - 1).is_err());
        assert!(check_device_facts((8, 0), PEAK_DEVICE_BYTES).is_ok());
        assert!(check_device_facts((12, 0), 32 << 30).is_ok());
    }

    #[test]
    fn the_kernel_target_is_the_minimum_compute_capability() {
        let (major, minor) = super::MIN_COMPUTE_CAPABILITY;
        assert_eq!(super::MIN_COMPUTE_TARGET, format!("compute_{major}{minor}"));
    }

    /// Below compute capability 8.0 a device is refused on that first, with
    /// both capabilities named, whatever its memory.
    #[test]
    fn a_device_below_compute_capability_8_is_refused_on_it_first() {
        for total in [15 << 30, 48u64 << 30] {
            let err = check_device_facts((7, 5), total).unwrap_err().to_string();
            assert!(
                err.contains("compute capability 8.0 or newer")
                    && err.contains("compute capability 7.5"),
                "{err}"
            );
            assert!(!err.contains("GiB"), "{err}");
        }
    }
}
