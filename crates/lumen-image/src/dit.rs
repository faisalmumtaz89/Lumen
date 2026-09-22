//! The Qwen-Image-2.1 diffusion transformer, on the CPU.
//!
//! Mirrors `QwenImage21Transformer2DModel`: a stack of single-stream blocks over
//! one joint text/image sequence, block-causal attention, three-axis complex
//! RoPE, and a single modulation projection shared by every block. Operand order
//! follows the reference rather than what is convenient, because this is the
//! implementation a GPU port is judged against.
//!
//! Batch size is fixed at one. The reference carries a batch axis mainly so the
//! `t = 0` modulation row can ride along with the sampled timesteps, and one
//! sample needs two rows rather than a batch.
//!
//! Not implemented here, deliberately: the prefix KV cache (a decoding-speed
//! optimisation whose output the prefill path already defines), the text padding
//! mask (`encoder_hidden_states_mask`; callers pass unpadded prompts), and
//! `patch_size`, which 2.1 fixes at 1.

use std::path::Path;

use crate::lbi::{LbiError, LbiFile};
use crate::tensor::{
    gelu_tanh, layer_norm_rows, rms_norm_rows, silu, zero_center_rms_norm_rows, Matrix,
};

/// Each vision-language image slot stands for a 2x2 group of latent tokens.
const IMG_TOKENS_PER_SLOT: usize = 4;

/// `QwenImage21TimestepProjEmbeddings` fixes the sinusoidal width at 256.
const TIMESTEP_DIM: usize = 256;
const TIMESTEP_MAX_PERIOD: f64 = 10000.0;
const TIMESTEP_TIME_FACTOR: f32 = 1000.0;

/// Rows of the RoPE frequency table. The reference builds it as
/// `cat(rope_params(arange(8192)), rope_params(flip(arange(1024)) * -1 - 1))`,
/// so rows `[8192, 9216)` hold positions `[-1024, -1)` and a negative position
/// lands on its own row through Python's negative indexing. Reproducing the
/// table rather than evaluating a position directly also reproduces the
/// aliasing: a position below `-1024` wraps onto a positive row instead of
/// failing, exactly as it does in torch.
const ROPE_POS_ROWS: i64 = 8192;
const ROPE_NEG_ROWS: i64 = 1024;
const ROPE_ROWS: i64 = ROPE_POS_ROWS + ROPE_NEG_ROWS;

#[derive(Debug)]
pub enum DitError {
    Lbi(LbiError),
    /// The checkpoint does not carry a tensor the model needs.
    MissingTensor(String),
    /// A tensor, or a caller-supplied input, has the wrong extent.
    ShapeMismatch {
        what: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
}

impl std::fmt::Display for DitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DitError::Lbi(e) => write!(f, "lbi: {e}"),
            DitError::MissingTensor(name) => write!(f, "checkpoint has no tensor `{name}`"),
            DitError::ShapeMismatch {
                what,
                expected,
                actual,
            } => write!(f, "{what}: expected {expected:?}, got {actual:?}"),
        }
    }
}

impl std::error::Error for DitError {}

impl From<LbiError> for DitError {
    fn from(e: LbiError) -> Self {
        DitError::Lbi(e)
    }
}

/// The transformer's architecture constants.
#[derive(Debug, Clone)]
pub struct DitConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub context_in_dim: usize,
    pub mlp_ratio: usize,
    /// Rotary dims for the frame, height and width axes. Their halves must sum
    /// to `attention_head_dim / 2`, which is what makes one position's
    /// frequencies exactly as wide as a head.
    pub axes_dims_rope: [usize; 3],
    pub eps: f32,
    pub rope_theta: f32,
    /// Modulate text and condition-image tokens from `t = 0`.
    pub causal_condition: bool,
}

impl DitConfig {
    /// The values in the shipped `transformer/config.json`.
    pub fn qwen_image_2_1() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            num_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 32,
            context_in_dim: 4096,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
            rope_theta: 10000.0,
            causal_condition: true,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        self.inner_dim() * self.mlp_ratio
    }
}

/// One single-stream block. Modulation lives on the parent, not here.
#[derive(Debug)]
struct DitBlock {
    to_q: Matrix,
    to_k: Matrix,
    to_v: Matrix,
    to_out: Matrix,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    mlp_gate: Matrix,
    mlp_proj: Matrix,
    mlp_out: Matrix,
}

/// One rotary axis: `ROPE_ROWS` positions of `half` complex frequencies.
#[derive(Debug)]
struct RopeAxis {
    half: usize,
    data: Vec<(f32, f32)>,
}

impl RopeAxis {
    fn build(dim: usize, theta: f32) -> Self {
        // `range(0, dim, 2)`, so an odd `dim` rounds up — the reference derives
        // the width from the step range, not from `dim / 2`.
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|j| 1.0 / theta.powf(j as f32 / dim as f32))
            .collect();
        let half = inv.len();
        let mut data = Vec::with_capacity(ROPE_ROWS as usize * half);
        for row in 0..ROPE_ROWS {
            let index = if row < ROPE_POS_ROWS {
                row
            } else {
                row - ROPE_ROWS
            };
            for &f in &inv {
                let angle = index as f32 * f;
                data.push((angle.cos(), angle.sin()));
            }
        }
        Self { half, data }
    }

    fn row(&self, index: i64) -> Result<&[(f32, f32)], DitError> {
        let row = if index < 0 { ROPE_ROWS + index } else { index };
        if !(0..ROPE_ROWS).contains(&row) {
            return Err(DitError::ShapeMismatch {
                what: format!("rope position {index} is outside the frequency table"),
                expected: vec![ROPE_ROWS as u64],
                actual: vec![index.unsigned_abs()],
            });
        }
        let start = row as usize * self.half;
        Ok(&self.data[start..start + self.half])
    }
}

/// The loaded transformer.
#[derive(Debug)]
pub struct Dit {
    config: DitConfig,
    img_in: Matrix,
    text_norm: Vec<f32>,
    txt_in: Matrix,
    txt_out: Matrix,
    time_linear_1: Matrix,
    time_linear_2: Matrix,
    modulation: Matrix,
    blocks: Vec<DitBlock>,
    norm_out: Matrix,
    proj_out: Matrix,
    rope: [RopeAxis; 3],
}

/// One forward pass's inputs.
#[derive(Debug, Clone, Copy)]
pub struct DitForwardArgs<'a> {
    /// Packed latents, condition images first and the target image last.
    pub hidden_states: &'a Matrix,
    /// The vision-language encoder's hidden states over its whole sequence.
    pub encoder_hidden_states: &'a Matrix,
    /// The denoising step scaled to `[0, 1]`; the sinusoidal embedding scales it
    /// back up by 1000 itself.
    pub timestep: f32,
    /// Per-image `(frame, height, width)` in latent tokens, target image last.
    pub img_shapes: &'a [(u64, u64, u64)],
    /// `true` at the vision-language encoder's image slots, with one slot per
    /// 2x2 group of target latents appended.
    pub img_mask: &'a [bool],
}

impl Dit {
    /// Load every weight from a converted `.lbi` file, assuming the shipped
    /// Qwen-Image-2.1 architecture.
    pub fn load(path: &Path) -> Result<Self, DitError> {
        Self::load_with(path, DitConfig::qwen_image_2_1())
    }

    /// Load against an explicit architecture.
    pub fn load_with(path: &Path, config: DitConfig) -> Result<Self, DitError> {
        let file = LbiFile::open(path)?;
        let hidden = config.inner_dim();
        let mlp = config.mlp_hidden();

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("transformer_blocks.{i}");
            blocks.push(DitBlock {
                to_q: matrix(&file, &format!("{p}.attn.to_q.weight"), hidden, hidden)?,
                to_k: matrix(&file, &format!("{p}.attn.to_k.weight"), hidden, hidden)?,
                to_v: matrix(&file, &format!("{p}.attn.to_v.weight"), hidden, hidden)?,
                to_out: matrix(&file, &format!("{p}.attn.to_out.0.weight"), hidden, hidden)?,
                norm_q: vector(
                    &file,
                    &format!("{p}.attn.norm_q.weight"),
                    config.attention_head_dim,
                )?,
                norm_k: vector(
                    &file,
                    &format!("{p}.attn.norm_k.weight"),
                    config.attention_head_dim,
                )?,
                mlp_gate: matrix(
                    &file,
                    &format!("{p}.img_mlp.gate_layer.weight"),
                    mlp,
                    hidden,
                )?,
                mlp_proj: matrix(&file, &format!("{p}.img_mlp.proj.weight"), mlp, hidden)?,
                mlp_out: matrix(&file, &format!("{p}.img_mlp.out.weight"), hidden, mlp)?,
            });
        }

        let rope = [
            RopeAxis::build(config.axes_dims_rope[0], config.rope_theta),
            RopeAxis::build(config.axes_dims_rope[1], config.rope_theta),
            RopeAxis::build(config.axes_dims_rope[2], config.rope_theta),
        ];
        let rope_half: usize = rope.iter().map(|a| a.half).sum();
        if rope_half != config.attention_head_dim / 2 {
            return Err(DitError::ShapeMismatch {
                what: "rope axis dims do not cover a head".to_string(),
                expected: vec![(config.attention_head_dim / 2) as u64],
                actual: vec![rope_half as u64],
            });
        }

        Ok(Self {
            img_in: matrix(&file, "img_in.weight", hidden, config.in_channels)?,
            text_norm: vector(&file, "txt_in.text_norm.weight", config.context_in_dim)?,
            txt_in: matrix(
                &file,
                "txt_in.in_layer.weight",
                hidden,
                config.context_in_dim,
            )?,
            txt_out: matrix(&file, "txt_in.out_layer.weight", hidden, hidden)?,
            time_linear_1: matrix(
                &file,
                "time_text_embed.timestep_embedder.linear_1.weight",
                hidden,
                TIMESTEP_DIM,
            )?,
            time_linear_2: matrix(
                &file,
                "time_text_embed.timestep_embedder.linear_2.weight",
                hidden,
                hidden,
            )?,
            modulation: matrix(&file, "modulation.1.weight", 4 * hidden, hidden)?,
            norm_out: matrix(&file, "norm_out.linear.weight", hidden, hidden)?,
            proj_out: matrix(&file, "proj_out.weight", config.out_channels, hidden)?,
            blocks,
            rope,
            config,
        })
    }

    /// One forward pass over the whole joint sequence.
    ///
    /// The result has one row per joint token, text included; the caller takes
    /// the trailing `target_tokens` rows, as the pipeline's
    /// `noise_pred[:, -latents.size(1):]` does.
    pub fn forward(&self, args: DitForwardArgs<'_>) -> Result<Matrix, DitError> {
        let cfg = &self.config;
        let hidden = cfg.inner_dim();

        if args.hidden_states.cols != cfg.in_channels {
            return Err(mismatch(
                "hidden_states width",
                cfg.in_channels,
                args.hidden_states.cols,
            ));
        }
        if args.encoder_hidden_states.cols != cfg.context_in_dim {
            return Err(mismatch(
                "encoder_hidden_states width",
                cfg.context_in_dim,
                args.encoder_hidden_states.cols,
            ));
        }
        let Some(&(tf, th, tw)) = args.img_shapes.last() else {
            return Err(mismatch("img_shapes length", 1, 0));
        };
        let target_tokens = (tf * th * tw) as usize;
        if target_tokens % IMG_TOKENS_PER_SLOT != 0 {
            return Err(mismatch(
                "target image tokens per 2x2 slot",
                0,
                target_tokens % IMG_TOKENS_PER_SLOT,
            ));
        }
        let text_seq = args.encoder_hidden_states.rows;
        let want_slots = text_seq + target_tokens / IMG_TOKENS_PER_SLOT;
        if args.img_mask.len() != want_slots {
            return Err(mismatch("img_mask length", want_slots, args.img_mask.len()));
        }

        let img = args.hidden_states.linear(&self.img_in, None);
        let txt = self.text_projection(args.encoder_hidden_states);

        // Each image slot stands for four latent tokens, so expand it four-fold
        // and drop the projected latents into the expanded positions. The rows
        // the reference concatenates for the target slots are zeros, and they
        // are overwritten in full, so they only ever set the sequence length.
        let mut image_pad_mask = Vec::with_capacity(args.img_mask.len());
        let mut joint = Matrix::zeros(
            args.img_mask
                .iter()
                .map(|&s| if s { IMG_TOKENS_PER_SLOT } else { 1 })
                .sum(),
            hidden,
        );
        let mut token = 0;
        for (slot, &is_image) in args.img_mask.iter().enumerate() {
            let repeats = if is_image { IMG_TOKENS_PER_SLOT } else { 1 };
            for _ in 0..repeats {
                if slot < text_seq {
                    joint.row_mut(token).copy_from_slice(txt.row(slot));
                }
                image_pad_mask.push(is_image);
                token += 1;
            }
        }
        let image_tokens = image_pad_mask.iter().filter(|&&b| b).count();
        if img.rows != image_tokens {
            return Err(mismatch("packed latent rows", image_tokens, img.rows));
        }
        let mut next = 0;
        for (token, &is_image) in image_pad_mask.iter().enumerate() {
            if is_image {
                joint.row_mut(token).copy_from_slice(img.row(next));
                next += 1;
            }
        }
        let seq = joint.rows;

        let (frame_index, height_index, width_index) =
            rope_indices(args.img_shapes, &image_pad_mask)?;
        let freqs = self.rope_freqs(&frame_index, &height_index, &width_index)?;
        let (image_ids, target_token_mask) = token_metadata(&image_pad_mask, args.img_shapes)?;

        // With `causal_condition` the modulation carries an extra `t = 0` row,
        // which every token outside the target image reads.
        let timesteps: &[f32] = if cfg.causal_condition {
            &[args.timestep, 0.0]
        } else {
            std::slice::from_ref(&args.timestep)
        };
        let temb = self.timestep_embedding(timesteps);
        let modulation = apply_silu(&temb).linear(&self.modulation, None);
        let mod_row: Vec<usize> = if cfg.causal_condition {
            target_token_mask
                .iter()
                .map(|&target| if target { 0 } else { 1 })
                .collect()
        } else {
            vec![0; seq]
        };

        let mut x = joint;
        for block in &self.blocks {
            let normed = scale_rows(&layer_norm_rows(&x, cfg.eps), &modulation, 0, &mod_row);
            let attn = self.attention(block, &normed, &freqs, &image_ids);
            x = add_gated(&x, &attn, &modulation, hidden, &mod_row);

            let normed = scale_rows(
                &layer_norm_rows(&x, cfg.eps),
                &modulation,
                2 * hidden,
                &mod_row,
            );
            let mlp = feed_forward(block, &normed);
            x = add_gated(&x, &mlp, &modulation, 3 * hidden, &mod_row);
        }

        // `QwenImage21AdaLayerNormContinuous`: scale only, read from `temb`
        // rather than from the shared modulation.
        let scale = apply_silu(&temb).linear(&self.norm_out, None);
        let out = scale_rows(&layer_norm_rows(&x, cfg.eps), &scale, 0, &mod_row);
        Ok(out.linear(&self.proj_out, None))
    }

    /// `QwenImage21TextProjection`: zero-centred RMSNorm, linear, GELU, linear.
    fn text_projection(&self, encoder_hidden_states: &Matrix) -> Matrix {
        let normed =
            zero_center_rms_norm_rows(encoder_hidden_states, &self.text_norm, self.config.eps);
        let mut h = normed.linear(&self.txt_in, None);
        for v in h.data.iter_mut() {
            *v = gelu_tanh(*v);
        }
        h.linear(&self.txt_out, None)
    }

    /// The sinusoidal timestep embedding followed by `TimestepEmbedding`, whose
    /// `forward` puts the activation between the two linears.
    fn timestep_embedding(&self, timesteps: &[f32]) -> Matrix {
        let mut proj = Matrix::zeros(timesteps.len(), TIMESTEP_DIM);
        for (r, &t) in timesteps.iter().enumerate() {
            temporal_timesteps(t, proj.row_mut(r));
        }
        let h = apply_silu(&proj.linear(&self.time_linear_1, None));
        h.linear(&self.time_linear_2, None)
    }

    /// One position's frequencies: the frame, height and width axes
    /// concatenated, half as wide as a head because they are complex.
    fn rope_freqs(
        &self,
        frame: &[i64],
        height: &[i64],
        width: &[i64],
    ) -> Result<Vec<Vec<(f32, f32)>>, DitError> {
        let mut out = Vec::with_capacity(frame.len());
        for t in 0..frame.len() {
            let mut row = Vec::with_capacity(self.config.attention_head_dim / 2);
            row.extend_from_slice(self.rope[0].row(frame[t])?);
            row.extend_from_slice(self.rope[1].row(height[t])?);
            row.extend_from_slice(self.rope[2].row(width[t])?);
            out.push(row);
        }
        Ok(out)
    }

    fn attention(
        &self,
        block: &DitBlock,
        x: &Matrix,
        freqs: &[Vec<(f32, f32)>],
        image_ids: &[i64],
    ) -> Matrix {
        let heads = self.config.num_attention_heads;
        let head_dim = self.config.attention_head_dim;
        let seq = x.rows;

        // `[seq, heads * head_dim]` and `[seq * heads, head_dim]` are the same
        // row-major buffer, so the per-head norm and rotation are plain row
        // operations and the existing primitives apply unchanged.
        let q = Matrix::new(seq * heads, head_dim, x.linear(&block.to_q, None).data);
        let k = Matrix::new(seq * heads, head_dim, x.linear(&block.to_k, None).data);
        let v = Matrix::new(seq * heads, head_dim, x.linear(&block.to_v, None).data);
        let mut q = rms_norm_rows(&q, &block.norm_q, self.config.eps);
        let mut k = rms_norm_rows(&k, &block.norm_k, self.config.eps);
        apply_rope(&mut q, freqs, heads);
        apply_rope(&mut k, freqs, heads);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = Matrix::zeros(seq, heads * head_dim);
        let mut scores = vec![0f32; seq];
        for h in 0..heads {
            for qi in 0..seq {
                let qrow = q.row(qi * heads + h);
                let mut max = f32::NEG_INFINITY;
                for kj in 0..seq {
                    if !attends(image_ids, qi, kj) {
                        scores[kj] = f32::NEG_INFINITY;
                        continue;
                    }
                    let krow = k.row(kj * heads + h);
                    let mut acc = 0f32;
                    for d in 0..head_dim {
                        acc += qrow[d] * krow[d];
                    }
                    scores[kj] = acc * scale;
                    if scores[kj] > max {
                        max = scores[kj];
                    }
                }
                // Every query attends to at least itself, so the row is never
                // fully masked and the sum is never zero.
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = if s.is_finite() { (*s - max).exp() } else { 0.0 };
                    sum += *s;
                }
                let orow = &mut out.row_mut(qi)[h * head_dim..(h + 1) * head_dim];
                for kj in 0..seq {
                    if scores[kj] == 0.0 {
                        continue;
                    }
                    let p = scores[kj] / sum;
                    let vrow = v.row(kj * heads + h);
                    for d in 0..head_dim {
                        orow[d] += p * vrow[d];
                    }
                }
            }
        }
        out.linear(&block.to_out, None)
    }
}

/// `QwenImage21SwiGLUFeedForward`: the gate branch goes through SiLU, the
/// projection branch does not.
fn feed_forward(block: &DitBlock, x: &Matrix) -> Matrix {
    let mut gate = x.linear(&block.mlp_gate, None);
    let proj = x.linear(&block.mlp_proj, None);
    for (g, &p) in gate.data.iter_mut().zip(&proj.data) {
        *g = silu(*g) * p;
    }
    gate.linear(&block.mlp_out, None)
}

fn apply_silu(m: &Matrix) -> Matrix {
    let mut out = m.clone();
    for v in out.data.iter_mut() {
        *v = silu(*v);
    }
    out
}

/// `hidden * (1 + scale)`, each token reading the modulation row it was
/// assigned. `col_off` picks one of the four chunks of a shared modulation.
fn scale_rows(x: &Matrix, params: &Matrix, col_off: usize, rows: &[usize]) -> Matrix {
    let cols = x.cols;
    let mut out = x.clone();
    for t in 0..out.rows {
        let scale = &params.row(rows[t])[col_off..col_off + cols];
        for (v, &s) in out.row_mut(t).iter_mut().zip(scale) {
            *v *= 1.0 + s;
        }
    }
    out
}

/// `x + tanh(gate) * y`.
fn add_gated(x: &Matrix, y: &Matrix, params: &Matrix, col_off: usize, rows: &[usize]) -> Matrix {
    let cols = x.cols;
    let mut out = x.clone();
    for t in 0..out.rows {
        let gate = &params.row(rows[t])[col_off..col_off + cols];
        for ((v, &yv), &g) in out.row_mut(t).iter_mut().zip(y.row(t)).zip(gate) {
            *v += g.tanh() * yv;
        }
    }
    out
}

/// `apply_rotary_emb_qwen(..., use_real=False)`: each channel pair is one
/// complex number, multiplied by the position's frequency.
fn apply_rope(m: &mut Matrix, freqs: &[Vec<(f32, f32)>], heads: usize) {
    let head_dim = m.cols;
    for r in 0..m.rows {
        let f = &freqs[r / heads];
        let row = m.row_mut(r);
        for j in 0..head_dim / 2 {
            let (re, im) = (row[2 * j], row[2 * j + 1]);
            let (cos, sin) = f[j];
            row[2 * j] = re * cos - im * sin;
            row[2 * j + 1] = re * sin + im * cos;
        }
    }
}

/// `QwenImage21TemporalTimesteps`: cos in the first half, sin in the second.
pub fn temporal_timesteps(timestep: f32, out: &mut [f32]) {
    let half = TIMESTEP_DIM / 2;
    let t = TIMESTEP_TIME_FACTOR * timestep;
    // `math.log(max_period)` is a Python float, which torch narrows against the
    // f32 arange, so the exponent is an f32 computation.
    let neg_log = -(TIMESTEP_MAX_PERIOD.ln() as f32);
    for i in 0..half {
        let arg = t * (neg_log * i as f32 / half as f32).exp();
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
}

/// The block-causal rule `(q_idx >= kv_idx) or same_image_block`.
///
/// For text-to-image, where the only image block is the target, this reduces to
/// a causal triangle over the text and full visibility for the image tokens.
pub fn attends(image_ids: &[i64], q: usize, kv: usize) -> bool {
    q >= kv || (image_ids[q] == image_ids[kv] && image_ids[q] >= 0)
}

/// Label every joint token with the image block it belongs to, and mark the
/// target image's tokens.
///
/// Boundaries come from `img_shapes`, not from runs of `true`: two adjacent
/// condition images form one run but must stay separate blocks.
pub fn token_metadata(
    image_pad_mask: &[bool],
    img_shapes: &[(u64, u64, u64)],
) -> Result<(Vec<i64>, Vec<bool>), DitError> {
    let lengths: Vec<usize> = img_shapes
        .iter()
        .map(|&(f, h, w)| (f * h * w) as usize)
        .collect();
    let Some(&target_len) = lengths.last() else {
        return Err(mismatch("img_shapes length", 1, 0));
    };
    let positions: Vec<usize> = image_pad_mask
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let total: usize = lengths.iter().sum();
    if total != positions.len() {
        return Err(mismatch("image tokens", total, positions.len()));
    }

    let mut image_ids = vec![-1i64; image_pad_mask.len()];
    let mut next = 0;
    for (block, &len) in lengths.iter().enumerate() {
        for _ in 0..len {
            image_ids[positions[next]] = block as i64;
            next += 1;
        }
    }
    let mut target_token_mask = vec![false; image_pad_mask.len()];
    for &p in &positions[positions.len() - target_len..] {
        target_token_mask[p] = true;
    }
    Ok((image_ids, target_token_mask))
}

/// The frame, height and width position of every joint token, mirroring
/// `QwenImage21Rope.forward`.
///
/// Text advances one shared position on all three axes. An image block freezes
/// the frame axis at the position the preceding text reached and lays its tokens
/// on a height/width grid centred on zero, then advances the shared position by
/// `max(height, width)` — not by the token count.
pub fn rope_indices(
    img_shapes: &[(u64, u64, u64)],
    image_pad_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), DitError> {
    let total_len = image_pad_mask.len();
    let mut frame = Vec::with_capacity(total_len);
    let mut grid_height = Vec::new();
    let mut grid_width = Vec::new();
    let mut cursor = 0usize;
    let mut position = 0i64;

    for (block, &(_, height, width)) in img_shapes.iter().enumerate() {
        let (h, w) = (height as usize, width as usize);
        let found = image_pad_mask[cursor..].iter().position(|&b| b);
        let Some(offset) = found else {
            return Err(DitError::ShapeMismatch {
                what: format!("no image token at or after position {cursor} for image block"),
                expected: vec![img_shapes.len() as u64],
                actual: vec![block as u64],
            });
        };
        let block_start = cursor + offset;

        let text_len = block_start - cursor;
        for k in 0..text_len {
            frame.push(position + k as i64);
        }
        position += text_len as i64;

        cursor = block_start + h * w;
        for _ in 0..h * w {
            frame.push(position);
        }
        position += h.max(w) as i64;

        for row in -((h - h / 2) as i64)..(h / 2) as i64 {
            for _ in 0..w {
                grid_height.push(row);
            }
        }
        for _ in 0..h {
            for col in -((w - w / 2) as i64)..(w / 2) as i64 {
                grid_width.push(col);
            }
        }
    }
    if cursor < total_len {
        for k in 0..total_len - cursor {
            frame.push(position + k as i64);
        }
    }
    if frame.len() != total_len {
        return Err(mismatch("rope positions", total_len, frame.len()));
    }

    let mut height_index = frame.clone();
    let mut width_index = frame.clone();
    let mut next = 0;
    for (t, &is_image) in image_pad_mask.iter().enumerate() {
        if is_image {
            height_index[t] = grid_height[next];
            width_index[t] = grid_width[next];
            next += 1;
        }
    }
    Ok((frame, height_index, width_index))
}

fn mismatch(what: &str, expected: usize, actual: usize) -> DitError {
    DitError::ShapeMismatch {
        what: what.to_string(),
        expected: vec![expected as u64],
        actual: vec![actual as u64],
    }
}

fn matrix(file: &LbiFile, name: &str, rows: usize, cols: usize) -> Result<Matrix, DitError> {
    let entry = file
        .get(name)
        .ok_or_else(|| DitError::MissingTensor(name.to_string()))?;
    let want = vec![rows as u64, cols as u64];
    if entry.shape != want {
        return Err(DitError::ShapeMismatch {
            what: name.to_string(),
            expected: want,
            actual: entry.shape.clone(),
        });
    }
    Ok(Matrix::new(rows, cols, file.read_f32(name)?))
}

fn vector(file: &LbiFile, name: &str, len: usize) -> Result<Vec<f32>, DitError> {
    let entry = file
        .get(name)
        .ok_or_else(|| DitError::MissingTensor(name.to_string()))?;
    let want = vec![len as u64];
    if entry.shape != want {
        return Err(DitError::ShapeMismatch {
            what: name.to_string(),
            expected: want,
            actual: entry.shape.clone(),
        });
    }
    Ok(file.read_f32(name)?)
}

/// The reference's shape and position policy, in the reference's words.
///
/// [`crate::cuda::dit_gpu`] carries the same functions because the GPU forward
/// needs them; this re-export is what lets a check compare the two copies
/// directly instead of by inspection. Nothing else should call in here: the
/// entry point for a forward pass is [`Dit::forward`].
pub mod policy {
    pub use super::{attends, rope_indices, temporal_timesteps, token_metadata};

    /// [`token_metadata`] and [`rope_indices`] in one call, as the forward pass
    /// uses them.
    pub fn metadata_and_rope(
        img_shapes: &[(u64, u64, u64)],
        image_pad_mask: &[bool],
    ) -> Result<(Vec<i64>, Vec<bool>, Vec<i64>, Vec<i64>, Vec<i64>), super::DitError> {
        let (ids, target) = token_metadata(image_pad_mask, img_shapes)?;
        let (frame, height, width) = rope_indices(img_shapes, image_pad_mask)?;
        Ok((ids, target, frame, height, width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_format::QuantScheme;

    /// A config small enough to hold in a test, with the same shape relations as
    /// the real one: the rotary axis halves (1 + 1 + 2) cover half a head.
    fn tiny_config() -> DitConfig {
        DitConfig {
            in_channels: 4,
            out_channels: 4,
            num_layers: 2,
            attention_head_dim: 8,
            num_attention_heads: 2,
            context_in_dim: 6,
            mlp_ratio: 2,
            axes_dims_rope: [2, 2, 4],
            eps: 1e-6,
            rope_theta: 10000.0,
            causal_condition: true,
        }
    }

    /// Every tensor the loader must find, written out by hand so the test is a
    /// statement about the checkpoint's names rather than a copy of the loader.
    fn tiny_manifest() -> Vec<(String, Vec<u64>)> {
        let mut m = vec![
            ("img_in.weight".to_string(), vec![16, 4]),
            ("txt_in.text_norm.weight".to_string(), vec![6]),
            ("txt_in.in_layer.weight".to_string(), vec![16, 6]),
            ("txt_in.out_layer.weight".to_string(), vec![16, 16]),
            (
                "time_text_embed.timestep_embedder.linear_1.weight".to_string(),
                vec![16, 256],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.weight".to_string(),
                vec![16, 16],
            ),
            ("modulation.1.weight".to_string(), vec![64, 16]),
            ("norm_out.linear.weight".to_string(), vec![16, 16]),
            ("proj_out.weight".to_string(), vec![4, 16]),
        ];
        for i in 0..2 {
            for (suffix, shape) in [
                ("attn.to_q.weight", vec![16, 16]),
                ("attn.to_k.weight", vec![16, 16]),
                ("attn.to_v.weight", vec![16, 16]),
                ("attn.to_out.0.weight", vec![16, 16]),
                ("attn.norm_q.weight", vec![8]),
                ("attn.norm_k.weight", vec![8]),
                ("img_mlp.gate_layer.weight", vec![32, 16]),
                ("img_mlp.proj.weight", vec![32, 16]),
                ("img_mlp.out.weight", vec![16, 32]),
            ] {
                m.push((format!("transformer_blocks.{i}.{suffix}"), shape));
            }
        }
        m
    }

    /// Small, distinct, deterministic values: a weight of zero would hide an
    /// operand-order mistake behind a zero output.
    fn fill(seed: u64, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 4);
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32) / (1u32 << 24) as f32;
            out.extend_from_slice(&(unit * 0.2 - 0.1).to_le_bytes());
        }
        out
    }

    fn write_tiny(name: &str, skip: Option<&str>) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("dit-test-{}-{name}.lbi", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut w = crate::lbi::LbiWriter::create(&path, serde_json::json!({"probe": true}))
            .expect("create");
        for (seed, (tensor, shape)) in tiny_manifest().into_iter().enumerate() {
            if Some(tensor.as_str()) == skip {
                continue;
            }
            let n: u64 = shape.iter().product();
            w.append(
                &tensor,
                &shape,
                QuantScheme::F32,
                &fill(seed as u64 + 1, n as usize),
            )
            .expect("append");
        }
        w.finish().expect("finish");
        path
    }

    #[test]
    fn rope_indices_match_the_python_for_one_2x3_image() {
        // Two text tokens then a 2x3 image block, so `image_pad_mask` is
        // [F, F, T, T, T, T, T, T] and `img_shapes` is [(1, 2, 3)].
        //
        // Python, step by step:
        //   block_start = 2, text_len = 2 -> frame_index = [0, 1], position = 2
        //   cursor = 2 + 6 = 8, frame_index += [2] * 6, position += max(2, 3)
        //   image_height_index = [h for h in range(-1, 1) for _ in range(3)]
        //                      = [-1, -1, -1, 0, 0, 0]
        //   image_width_index  = [w for _ in range(2) for w in range(-2, 1)]
        //                      = [-2, -1, 0, -2, -1, 0]
        //   height/width start as a copy of frame_index and are overwritten at
        //   the image positions.
        let mask = vec![false, false, true, true, true, true, true, true];
        let (frame, height, width) = rope_indices(&[(1, 2, 3)], &mask).unwrap();
        assert_eq!(frame, vec![0, 1, 2, 2, 2, 2, 2, 2]);
        assert_eq!(height, vec![0, 1, -1, -1, -1, 0, 0, 0]);
        assert_eq!(width, vec![0, 1, -2, -1, 0, -2, -1, 0]);
    }

    #[test]
    fn rope_indices_advance_the_frame_axis_by_max_height_width() {
        // A 2x3 image followed by two more text tokens: the text after the block
        // resumes at 2 + max(2, 3) = 5, not at 2 + 6.
        let mut mask = vec![false, false];
        mask.extend(std::iter::repeat_n(true, 6));
        mask.extend([false, false]);
        let (frame, _, _) = rope_indices(&[(1, 2, 3)], &mask).unwrap();
        assert_eq!(frame, vec![0, 1, 2, 2, 2, 2, 2, 2, 5, 6]);
    }

    #[test]
    fn text_to_image_mask_is_text_causal_and_image_open() {
        // Three text tokens then one 2x3 target image: the only block is the
        // target, so `image_ids` is [-1, -1, -1, 0, 0, 0, 0, 0, 0].
        let mut mask = vec![false; 3];
        mask.extend(std::iter::repeat_n(true, 6));
        let (image_ids, target) = token_metadata(&mask, &[(1, 2, 3)]).unwrap();
        assert_eq!(image_ids, vec![-1, -1, -1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(target, mask);

        for q in 0..3 {
            for kv in 0..9 {
                assert_eq!(attends(&image_ids, q, kv), kv <= q, "text q={q} kv={kv}");
            }
        }
        for q in 3..9 {
            for kv in 0..9 {
                assert!(attends(&image_ids, q, kv), "image q={q} kv={kv}");
            }
        }
    }

    #[test]
    fn two_image_blocks_stay_separate_even_when_adjacent() {
        // A condition image and the target image with no text between them form
        // one run of `true` but two blocks: the earlier block must not see the
        // later one.
        let mask = vec![true, true, true, true];
        let (image_ids, target) = token_metadata(&mask, &[(1, 1, 2), (1, 1, 2)]).unwrap();
        assert_eq!(image_ids, vec![0, 0, 1, 1]);
        assert_eq!(target, vec![false, false, true, true]);
        assert!(
            !attends(&image_ids, 0, 2),
            "condition must not see the target"
        );
        assert!(attends(&image_ids, 2, 0), "the target sees the condition");
        assert!(attends(&image_ids, 0, 1), "a block is bidirectional");
    }

    #[test]
    fn forward_returns_one_row_per_joint_token() {
        let path = write_tiny("forward", None);
        let dit = Dit::load_with(&path, tiny_config()).expect("load");

        // One 2x4 target image: 8 latent tokens, so two vision-language slots
        // on top of two text tokens.
        let hidden_states = Matrix::new(8, 4, (0..32).map(|i| i as f32 * 0.01).collect());
        let encoder_hidden_states = Matrix::new(2, 6, (0..12).map(|i| i as f32 * 0.02).collect());
        let img_mask = [false, false, true, true];
        let out = dit
            .forward(DitForwardArgs {
                hidden_states: &hidden_states,
                encoder_hidden_states: &encoder_hidden_states,
                timestep: 0.7,
                img_shapes: &[(1, 2, 4)],
                img_mask: &img_mask,
            })
            .expect("forward");

        assert_eq!((out.rows, out.cols), (10, 4));
        assert!(out.data.iter().all(|v| v.is_finite()), "non-finite output");
        assert!(out.data.iter().any(|&v| v != 0.0), "output is all zeros");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn forward_rejects_an_img_mask_that_does_not_cover_the_target() {
        let path = write_tiny("badmask", None);
        let dit = Dit::load_with(&path, tiny_config()).expect("load");
        let hidden_states = Matrix::zeros(8, 4);
        let encoder_hidden_states = Matrix::zeros(2, 6);
        // One slot short: 8 target tokens need two.
        let img_mask = [false, false, true];
        let err = dit
            .forward(DitForwardArgs {
                hidden_states: &hidden_states,
                encoder_hidden_states: &encoder_hidden_states,
                timestep: 0.7,
                img_shapes: &[(1, 2, 4)],
                img_mask: &img_mask,
            })
            .unwrap_err();
        assert!(
            matches!(err, DitError::ShapeMismatch { ref what, .. } if what == "img_mask length"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_reads_exactly_the_checkpoint_names() {
        let path = write_tiny("names", None);
        Dit::load_with(&path, tiny_config()).expect("the manifest names must load");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_names_a_tensor_the_checkpoint_is_missing() {
        // Drop each name in turn: a loader that read a different name would
        // still succeed here, so this pins the whole list, not just its length.
        for (tensor, _) in tiny_manifest() {
            let path = write_tiny("missing", Some(&tensor));
            let err = Dit::load_with(&path, tiny_config()).unwrap_err();
            match err {
                DitError::MissingTensor(name) => assert_eq!(name, tensor),
                other => panic!("dropping `{tensor}` gave {other}"),
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn rope_table_reproduces_negative_positions() {
        // Position -1 must land on the table's last row, which the reference
        // builds from `flip(arange(1024)) * -1 - 1`. The expected values are
        // `QwenImage21Rope(theta=10000, axes_dim=[..., 4]).freqs[-1][-1]` read
        // out of torch, so this is not a restatement of the formula below.
        let axis = RopeAxis::build(4, 10000.0);
        let want = [
            (0.540_302_3f32, -0.841_470_96f32),
            (0.999_95f32, -0.009_999_833f32),
        ];
        let got = axis.row(-1).unwrap();
        assert_eq!(got.len(), want.len());
        for (j, (&(cos, sin), &(wc, ws))) in got.iter().zip(&want).enumerate() {
            assert!((cos - wc).abs() < 1e-7, "cos at {j}: {cos} vs {wc}");
            assert!((sin - ws).abs() < 1e-7, "sin at {j}: {sin} vs {ws}");
        }
        assert!(axis.row(ROPE_ROWS).is_err(), "past the table must fail");
    }

    #[test]
    fn temporal_timesteps_puts_cos_first_and_scales_by_1000() {
        let mut out = vec![0f32; TIMESTEP_DIM];
        temporal_timesteps(0.0, &mut out);
        assert!(out[..TIMESTEP_DIM / 2].iter().all(|&v| v == 1.0));
        assert!(out[TIMESTEP_DIM / 2..].iter().all(|&v| v == 0.0));

        // Channel 0 has frequency 1, so the argument is `1000 * timestep`.
        temporal_timesteps(0.001, &mut out);
        assert!((out[0] - 1.0f32.cos()).abs() < 1e-6, "cos {}", out[0]);
        assert!(
            (out[TIMESTEP_DIM / 2] - 1.0f32.sin()).abs() < 1e-6,
            "sin {}",
            out[TIMESTEP_DIM / 2]
        );
    }
}
