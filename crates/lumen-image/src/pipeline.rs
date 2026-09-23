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
//! text encoder's language tower is 14.1 GiB of BF16 weights and the
//! transformer 13.3 GiB, so they do not fit on the card together.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

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

    /// Open every required file the way a generation will, without reading
    /// weights: the text encoder's whole tensor manifest (and, when `gpu`,
    /// everything the device text tower would refuse to load), the
    /// transformer's and the VAE's configuration plus the tensors that
    /// identify each container as that component at the expected shapes
    /// (stored as bf16 when `gpu`, which is what the device transformer
    /// multiplies), and the tokenizer with the chat template's markers as
    /// whole tokens. A checkpoint that fails here would fail every request
    /// after the text model had been evicted for it.
    pub fn check(&self, gpu: bool) -> Result<(), PipelineError> {
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
            let dev = lumen_runtime::cuda::ffi::CudaDevice::new(GPU_DEVICE)
                .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;
            let total = dev
                .total_memory()
                .map_err(|e| PipelineError::Unsupported(format!("device memory query: {e}")))?;
            check_device_memory(total as u64)?;
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

/// The most device memory a generation uses, in bytes: 21,491 MiB, measured
/// as the device's used memory at the peak of a 2048x2048 generation (the
/// VAE decode; a 1024x1024 generation peaks at 15,923 MiB in the text-encoder
/// phase) on an RTX 5090 with a 10 ms `nvidia-smi` sampler. A device with
/// less total memory would fail a request at that size after the text model
/// had been evicted for it, so the startup check refuses it instead.
pub const PEAK_DEVICE_BYTES: u64 = 21_491 << 20;

/// Refuse a device whose total memory is below [`PEAK_DEVICE_BYTES`].
pub fn check_device_memory(total_bytes: u64) -> Result<(), PipelineError> {
    if total_bytes < PEAK_DEVICE_BYTES {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        return Err(PipelineError::Unsupported(format!(
            "the device has {:.1} GiB of memory ({total_bytes} bytes) and a generation at \
             2048x2048 needs {:.1} GiB ({PEAK_DEVICE_BYTES} bytes)",
            gib(total_bytes),
            gib(PEAK_DEVICE_BYTES),
        )));
    }
    Ok(())
}

/// Generate one image on the GPU.
///
/// Same sequence as [`generate_cpu`], with each component's device
/// implementation. The three components still load and free one at a time: the
/// text encoder's language tower is 14.1 GiB of BF16 weights and the transformer
/// 13.3 GiB, so they do not fit on a 32 GiB card together and do not need to —
/// the encoder's output is the only thing the transformer consumes.
#[cfg(feature = "cuda")]
pub fn generate_gpu(
    sources: &GpuSources,
    req: &GenerationRequest<'_>,
    progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<Rgba, PipelineError> {
    use lumen_runtime::cuda::ffi::CudaDevice;

    let dev = CudaDevice::new(GPU_DEVICE)
        .map_err(|e| PipelineError::Unsupported(format!("no CUDA device: {e}")))?;

    let ids = sources.tokenizer.encode(&render_prompt(req.prompt))?;
    let text = encode_prompt_gpu(sources, &dev, &ids)?;
    let latents = {
        let dit = sources.load_dit(&dev)?;
        denoise_gpu(&dit, req, &text, progress)?
    };
    let vae = crate::cuda::vae_gpu::VaeGpu::load_with(&sources.vae, &dev, None)
        .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
    decode_gpu(&vae, req, &latents).map_err(PipelineError::Unsupported)
}

/// The CUDA pipeline's host side, opened once and kept for every generation:
/// the tokenizer and the three containers' mappings. A component loaded from
/// a mapping this process has already read reuses its page tables, so after
/// the first generation an upload runs at copy speed instead of faulting
/// every page in again. A container replaced on disk afterwards (a conversion
/// renames its new file into place) is not seen until the sources are opened
/// again.
#[cfg(feature = "cuda")]
pub struct GpuSources {
    tokenizer: Tokenizer,
    text_encoder: LbiFile,
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
            transformer: open(&paths.transformer)?,
            vae: open(&paths.vae)?,
        })
    }

    fn load_dit(
        &self,
        dev: &lumen_runtime::cuda::ffi::CudaDevice,
    ) -> Result<crate::cuda::dit_gpu::DitGpu, PipelineError> {
        crate::cuda::dit_gpu::DitGpu::load_with(&self.transformer, dev, DitConfig::qwen_image_2_1())
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))
    }
}

/// The transformer and the VAE held on the device between generations, for a
/// device nothing else needs in between (the text model on the CPU or on
/// another card).
///
/// The text encoder still loads for each generation: the three do not fit on a
/// 32 GiB card together with a decode's activations (1.45 GiB is left with all
/// three resident, and a 1024x1024 decode needs more), and it is the component
/// needed first and only briefly. Encoding and decoding each run beside the
/// resident transformer when they fit; one that runs out of device memory
/// there — a 2048x2048 decode, a prompt of thousands of tokens, any encode on
/// a card much smaller than 32 GiB — is retried once with the transformer
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
        })
    }

    /// Generate one image with the resident components; the same sequence and
    /// arithmetic as [`generate_gpu`].
    pub fn generate(
        &mut self,
        req: &GenerationRequest<'_>,
        progress: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Rgba, PipelineError> {
        let ids = self.sources.tokenizer.encode(&render_prompt(req.prompt))?;
        if self.encode_needs_room.is_some_and(|t| ids.len() >= t) {
            self.dit = None;
        }
        let text = match encode_prompt_gpu(&self.sources, &self.dev, &ids) {
            Ok(text) => text,
            Err(e) if self.dit.is_some() && is_out_of_memory(&e) => {
                self.dit = None;
                self.encode_needs_room = Some(smallest(self.encode_needs_room, ids.len()));
                encode_prompt_gpu(&self.sources, &self.dev, &ids)?
            }
            Err(e) => return Err(e),
        };
        let dit = match self.dit.take() {
            Some(dit) => dit,
            None => self.sources.load_dit(&self.dev)?,
        };
        let latents = denoise_gpu(&dit, req, &text, progress);
        self.dit = Some(dit);
        let latents = latents?;
        let pixels = req.width * req.height;
        if self.decode_needs_room.is_some_and(|p| pixels >= p) {
            self.dit = None;
        }
        match decode_gpu(&self.vae, req, &latents) {
            Ok(image) => Ok(image),
            Err(e) if self.dit.is_some() && e.contains(OUT_OF_MEMORY) => {
                self.dit = None;
                self.decode_needs_room = Some(smallest(self.decode_needs_room, pixels));
                decode_gpu(&self.vae, req, &latents).map_err(PipelineError::Unsupported)
            }
            Err(e) => Err(PipelineError::Unsupported(e)),
        }
    }
}

/// How the device's driver reports an allocation it could not make; the errors
/// reach here as text.
#[cfg(feature = "cuda")]
const OUT_OF_MEMORY: &str = "CUDA_ERROR_OUT_OF_MEMORY";

#[cfg(feature = "cuda")]
fn is_out_of_memory(e: &PipelineError) -> bool {
    format!("{e}").contains(OUT_OF_MEMORY)
}

/// The smaller of a remembered threshold and a new one.
#[cfg(feature = "cuda")]
fn smallest(known: Option<usize>, new: usize) -> usize {
    known.map_or(new, |k| k.min(new))
}

/// The rendered prompt's hidden states, with the template's system prefix
/// stripped. The encoder is loaded for the call and freed when it returns.
#[cfg(feature = "cuda")]
fn encode_prompt_gpu(
    sources: &GpuSources,
    dev: &lumen_runtime::cuda::ffi::CudaDevice,
    ids: &[u32],
) -> Result<Matrix, PipelineError> {
    let hidden = {
        let config =
            crate::text_encoder::TextEncoderConfig::from_lbi_config(sources.text_encoder.config())?;
        let encoder = crate::cuda::text_gpu::TextGpu::load_with(&sources.text_encoder, dev, config)
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
        encoder
            .forward(ids)
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?
    };
    if hidden.rows < SYSTEM_PREFIX_TOKENS {
        return Err(PipelineError::Unsupported(format!(
            "the prompt encoded to {} tokens, fewer than the {} the template strips",
            hidden.rows, SYSTEM_PREFIX_TOKENS
        )));
    }
    Ok(Matrix::new(
        hidden.rows - SYSTEM_PREFIX_TOKENS,
        hidden.cols,
        hidden.data[SYSTEM_PREFIX_TOKENS * hidden.cols..].to_vec(),
    ))
}

/// The denoising loop: the final latents for `req`, conditioned on `text`.
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
    let mut latents = starting_latents(req, seq, cfg.in_channels)?;
    let shapes = [(1u64, lat_h as u64, lat_w as u64)];
    for step in 0..req.steps {
        let hidden_states = Matrix::new(seq, cfg.in_channels, latents.clone());
        let out = dit
            .forward(DitForwardArgs {
                hidden_states: &hidden_states,
                encoder_hidden_states: text,
                timestep: sched.sigmas[step],
                img_shapes: &shapes,
                img_mask: &img_mask,
            })
            .map_err(|e| PipelineError::Unsupported(format!("{e}")))?;
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
    Ok(latents)
}

/// The image the final latents decode to.
#[cfg(feature = "cuda")]
fn decode_gpu(
    vae: &crate::cuda::vae_gpu::VaeGpu,
    req: &GenerationRequest<'_>,
    latents: &[f32],
) -> Result<Rgba, String> {
    let lat_h = latent_side(req.height);
    let lat_w = latent_side(req.width);
    let denorm = crate::vae::denormalize_latents_with(
        latents,
        1,
        lat_h,
        lat_w,
        &vae.config().latents_mean,
        &vae.config().latents_std,
    )
    .map_err(|e| format!("{e}"))?;
    let decoded = vae
        .decode(&denorm, 1, lat_h, lat_w)
        .map_err(|e| format!("{e}"))?;
    Ok(Rgba::from_planar_rgba(lat_w * 16, lat_h * 16, &decoded))
}

#[cfg(test)]
mod device_memory_tests {
    use super::{check_device_memory, PEAK_DEVICE_BYTES};

    /// A device below the measured peak is refused with both amounts in the
    /// message; one at or above it passes.
    #[test]
    fn a_small_device_is_refused_with_both_amounts_named() {
        let err = check_device_memory(16 << 30).unwrap_err().to_string();
        assert!(
            err.contains("16.0 GiB") && err.contains(&(16u64 << 30).to_string()),
            "{err}"
        );
        assert!(
            err.contains("21.0 GiB") && err.contains(&PEAK_DEVICE_BYTES.to_string()),
            "{err}"
        );
        assert!(check_device_memory(PEAK_DEVICE_BYTES).is_ok());
        assert!(check_device_memory(32 << 30).is_ok());
    }
}
