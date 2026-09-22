//! The Qwen3-VL text tower: the prompt encoder Qwen-Image-2.1 conditions on.
//!
//! # Text path only
//!
//! The checkpoint is a `Qwen3VLForConditionalGeneration`, so it also carries a
//! vision tower (`model.visual.*`) and an `lm_head`. Text-to-image runs
//! neither. `QwenImage21Pipeline._get_qwen_prompt_embeds` hands the model a
//! text-only template with no `pixel_values` and no `image_grid_thw`, so
//! `Qwen3VLModel.compute_3d_position_ids` finds `has_multimodal == False`,
//! cannot compute M-RoPE, and returns `None` position ids; the text model then
//! builds `arange(seq)` and broadcasts it to all three rotary rows. Nothing in
//! that path touches the vision tower, and the pipeline reads hidden states
//! rather than logits, so `lm_head` is unused too. Implementing either here
//! would be code no caller can exercise, so this module stops at
//! `model.language_model.*`.
//!
//! # The final RMS norm is deliberately skipped
//!
//! `Qwen3VLTextModel.forward` ends with `self.norm(hidden_states)`, but the
//! pipeline installs `text_model.norm.register_forward_hook(lambda module,
//! args, output: args[0])` around the call, which makes that module return its
//! own input. The transformer was trained on the *pre-norm* last-layer output,
//! and the hook is how the pipeline gets it on both transformers 4.x and 5.x.
//! So [`TextEncoder::forward`] returns the last decoder layer's output
//! unnormalised. `model.language_model.norm.weight` is still in the manifest
//! [`load`] checks, so a checkpoint that omits it, or carries it at the wrong
//! shape or in a scheme this reference cannot decode, is rejected at
//! [`load`]: the manifest is this module's statement of what a Qwen3-VL text
//! tower contains, and that tensor is part of one. Its bytes are never
//! decoded.
//!
//! # Precision
//!
//! Everything here is f32, weights included. That is deliberate — an f32
//! reference is what a backend should be compared against — but it is not what
//! the shipped model runs, and one place the difference is visible is worth
//! stating up front.
//!
//! `Qwen3VLTextRotaryEmbedding.forward` computes its tables under
//! `maybe_autocast(..., enabled=False)`, so the arithmetic really is float32,
//! but it ends with `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)` —
//! and the shipped text encoder runs in bfloat16 (`oracle-bundle`'s
//! `environment.json` records `"component_dtypes": {"text_encoder":
//! ["torch.bfloat16"]}`). The real run therefore rounds both tables to bf16
//! and [`rope_tables`] does not. Measured over the shipped config at 38
//! positions, f32 here against bf16 there differs by at most 1.95e-3 on `cos`,
//! mean 3.53e-4. Anyone gating this module against a bf16 oracle should set
//! the tolerance from that floor, not from f32 epsilon.
//!
//! # Memory
//!
//! An f32 copy of the whole tower is 7.57 B parameters, 28.2 GiB, which must
//! not be resident.
//! The mmap-backed [`LbiFile`] stays open inside the encoder; [`load`] checks
//! every tensor's name, shape and dtype from the index alone, and each weight
//! is decoded to f32 only for the one projection that uses it and dropped
//! immediately after. Peak resident weight is therefore one MLP matrix
//! (12288x4096 f32, 192 MiB). The embedding is never decoded whole: only the
//! prompt's rows are gathered out of the mapping.
//!
//! [`load`]: TextEncoder::load

use std::path::Path;

use lumen_format::QuantScheme;
use serde_json::Value;

use crate::lbi::{half_to_f32, LbiError, LbiFile};
use crate::tensor::{add, rms_norm_rows, silu, Matrix};

/// Every text-tower tensor sits under this prefix.
///
/// `Qwen3VLForConditionalGeneration.base_model_prefix` is `"model"` and holds
/// `self.model = Qwen3VLModel`, whose `self.language_model` is the
/// `Qwen3VLTextModel` this file implements.
const PREFIX: &str = "model.language_model.";

/// Qwen3-VL's `<|image_pad|>` id, used when the config does not carry
/// `image_token_id`.
///
/// The pipeline derives the id from the tokenizer, but `.lbi` stores only the
/// component's `config.json`, which for this checkpoint carries the id the
/// model itself compares against (`Qwen3VLModel` matches `input_ids ==
/// self.config.image_token_id`). This constant is the documented Qwen3-VL
/// value, used only when that key is absent.
pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 151_655;

/// Rows of the encoder output the pipeline drops before handing them to the
/// transformer: the tokenised system message.
///
/// `QwenImage21Pipeline.__init__` computes this as
/// `len(apply_chat_template(system_message))` rather than hardcoding it. For
/// the shipped system prompt that is the 14 ids of
/// `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n`
/// — confirmed against the reference bundle, whose 38-token prompt yields 24
/// encoder rows. It is exposed here because this crate has no tokenizer;
/// a caller that does have one should prefer its own count.
pub const SYSTEM_PREFIX_TOKENS: usize = 14;

/// What went wrong loading or running the text encoder.
#[derive(Debug)]
pub enum TextEncoderError {
    Lbi(LbiError),
    /// The checkpoint does not carry a tensor the architecture needs.
    MissingTensor {
        name: String,
    },
    /// A tensor is present but not the shape the config implies.
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    /// The config JSON is absent, unparseable, or internally inconsistent.
    BadConfig(String),
    /// A weight is stored in a scheme this reference cannot decode.
    ///
    /// Unreachable through [`TextEncoder`] as the container stands today:
    /// `TensorEntry::expected_length` has no storage rule for a quantized
    /// scheme, so `LbiFile::from_bytes` rejects such an entry while opening the
    /// file and `LbiWriter::append` refuses to write one. Both sites below
    /// raise it from a `match` that has to be exhaustive, not from a path a
    /// caller can take, which is why neither has a test.
    UnsupportedScheme {
        name: String,
        scheme: QuantScheme,
    },
    /// A token id the embedding table cannot address.
    TokenOutOfRange {
        id: u32,
        vocab: usize,
    },
    /// The encoder has nothing to encode.
    EmptyInput,
    /// The three M-RoPE position rows must describe the same sequence.
    PositionRowsDiffer {
        lengths: [usize; 3],
    },
}

impl std::fmt::Display for TextEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lbi(e) => write!(f, "{e}"),
            Self::MissingTensor { name } => write!(f, "the checkpoint has no tensor {name}"),
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "tensor {name} is {found:?} but the config implies {expected:?}"
            ),
            Self::BadConfig(why) => write!(f, "text encoder config: {why}"),
            Self::UnsupportedScheme { name, scheme } => {
                write!(
                    f,
                    "tensor {name} is stored as {scheme:?}, which cannot be decoded here"
                )
            }
            Self::TokenOutOfRange { id, vocab } => {
                write!(f, "token id {id} is outside the {vocab}-entry vocabulary")
            }
            Self::EmptyInput => write!(f, "the token sequence is empty"),
            Self::PositionRowsDiffer { lengths } => write!(
                f,
                "the three position rows have lengths {lengths:?} but must agree"
            ),
        }
    }
}

impl std::error::Error for TextEncoderError {}

impl From<LbiError> for TextEncoderError {
    fn from(e: LbiError) -> Self {
        Self::Lbi(e)
    }
}

/// Ceilings [`TextEncoderConfig::validate`] holds every structural dimension
/// to.
///
/// A `.lbi` config is a file this module did not write, so its numbers get the
/// same treatment [`crate::lbi`] already gives offsets and lengths: bounded
/// before anything is computed from them. Two things go wrong without that.
///
/// *Arithmetic.* No product this module forms from two of these fields exceeds
/// 2^40, and the three that size an allocation — `num_layers * 11` for the
/// manifest, `num_attention_heads * head_dim` for the query width,
/// `mrope_section[dim] * 3` for the rotary recomposition — stay under 2^25. So
/// none of them can wrap a comparison or panic on a hostile config.
///
/// *Time.* [`TextEncoder::load`] builds the whole manifest before it looks up
/// a single tensor, so an unbounded layer count is a slow refusal rather than
/// a fast one: 200000 layers spends 370 ms on 2.2 M strings to report one
/// missing tensor.
///
/// Each ceiling is at least 27 times the corresponding value in the config
/// this module targets ([`TextEncoderConfig::qwen_image_21`]: 36 layers,
/// hidden 4096, 32 heads of 128, intermediate 12288, vocabulary 151936), so a
/// checkpoint of this family cannot reach one and a config that does is
/// describing something else.
const MAX_LAYERS: usize = 1024;
const MAX_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 4096;
/// Bound on `hidden_size` and `intermediate_size` alike.
const MAX_WIDTH: usize = 1 << 20;
const MAX_VOCAB: usize = 1 << 22;

/// The hyperparameters the text tower needs.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEncoderConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// How many of the `head_dim / 2` rotary slots each of the three M-RoPE
    /// position rows (T, H, W) owns.
    pub mrope_section: [usize; 3],
}

impl TextEncoderConfig {
    /// Qwen-Image-2.1's text encoder: the language-model half of Qwen3-VL-8B.
    ///
    /// Named for the model it is, not the pipeline slot: 36 layers of width
    /// 4096 with a 12288 MLP and an untied 151936-entry vocabulary is the 8B
    /// checkpoint.
    pub fn qwen_image_21() -> Self {
        Self {
            num_layers: 36,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 5e6,
            mrope_section: [24, 20, 20],
        }
    }

    /// Parse the `.lbi` config JSON.
    ///
    /// `lbi-convert` stores the component's `config.json` verbatim, so the
    /// fields may sit under `text_config` (a composite `Qwen3VL` config) or at
    /// the top level (a bare text config). The fallback is applied **per key**,
    /// not all or nothing: a `text_config` that carries most of the fields and
    /// leaves, say, `rope_theta` at the root resolves both. A `text_config`
    /// that is present but is not an object is a corrupt file rather than a
    /// config to fall back from, and is rejected.
    ///
    /// The rotary settings have moved between `rope_parameters` and
    /// `rope_scaling` across transformers releases, so both spellings are
    /// accepted, as is a flat `rope_theta`. Anything ambiguous or inconsistent
    /// is rejected rather than guessed.
    pub fn from_lbi_config(v: &Value) -> Result<Self, TextEncoderError> {
        let view = ConfigView::new(v)?;

        let num_layers = need_usize(&view, "num_hidden_layers")?;
        let hidden_size = need_usize(&view, "hidden_size")?;
        let num_attention_heads = need_usize(&view, "num_attention_heads")?;
        let num_key_value_heads = need_usize(&view, "num_key_value_heads")?;
        let intermediate_size = need_usize(&view, "intermediate_size")?;
        let vocab_size = need_usize(&view, "vocab_size")?;

        if num_attention_heads == 0 {
            return Err(TextEncoderError::BadConfig(
                "`num_attention_heads` is zero".to_string(),
            ));
        }
        // transformers resolves `head_dim` the same way: the explicit field if
        // present, else the even split of the hidden size.
        let head_dim = opt_usize(&view, "head_dim")?.unwrap_or(hidden_size / num_attention_heads);

        // Defaulted, not required: this is `Qwen3VLTextConfig`'s own default and
        // a checkpoint that disagrees will say so explicitly.
        let rms_norm_eps = match opt_f64(&view, "rms_norm_eps")? {
            None => 1e-6f32,
            Some(raw) => narrow_to_positive_f32("rms_norm_eps", raw)?,
        };

        if let Some(act) = view.get("hidden_act").and_then(Value::as_str) {
            if act != "silu" {
                return Err(TextEncoderError::BadConfig(format!(
                    "`hidden_act` is {act:?}, but only SwiGLU with SiLU is implemented"
                )));
            }
        }
        if view.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return Err(TextEncoderError::BadConfig(
                "`attention_bias` is true, but the projections are loaded without biases"
                    .to_string(),
            ));
        }

        let rope = view
            .get("rope_parameters")
            .or_else(|| view.get("rope_scaling"))
            .filter(|r| r.is_object());
        if let Some(kind) = rope
            .and_then(|r| r.get("rope_type").or_else(|| r.get("type")))
            .and_then(Value::as_str)
        {
            // `"default"` is what this checkpoint is expected to carry, and the
            // default inverse-frequency schedule is what this implements.
            //
            // `"mrope"` is accepted purely defensively, as a legacy alias.
            // transformers 5.2.0 would *not* accept it: `ROPE_INIT_FUNCTIONS`
            // holds only `dynamic`, `linear`, `llama3`, `longrope` and `yarn`,
            // `standardize_rope_params` copies `type` to `rope_type` without
            // rewriting it, and building a `Qwen3VLTextRotaryEmbedding` with
            // `rope_type='mrope'` raises `KeyError: 'mrope'`. Since the real
            // checkpoint's `config.json` cannot be read from here, being more
            // permissive than transformers is the safe direction: the spelling
            // selects the schedule this file already implements either way.
            //
            // A genuine scaling scheme is not implemented.
            if kind != "default" && kind != "mrope" {
                return Err(TextEncoderError::BadConfig(format!(
                    "rope type {kind:?} is not implemented"
                )));
            }
        }
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| view.get("rope_theta"))
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                TextEncoderError::BadConfig("`rope_theta` is missing or not a number".to_string())
            })?;
        let rope_theta = narrow_to_positive_f32("rope_theta", rope_theta)?;

        let mrope_section = match rope
            .and_then(|r| r.get("mrope_section"))
            .or_else(|| view.get("mrope_section"))
        {
            Some(raw) => parse_mrope_section(raw)?,
            // `Qwen3VLTextRotaryEmbedding` itself falls back to this when the
            // key is absent; the sum check below still guards it.
            None => [24, 20, 20],
        };

        let cfg = Self {
            num_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            mrope_section,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject a config whose fields cannot describe this architecture.
    ///
    /// `num_attention_heads * head_dim` is deliberately *not* required to equal
    /// `hidden_size`. `Qwen3VLTextAttention` reads `head_dim` from the config
    /// independently of the hidden size and sizes `q_proj` as `hidden ->
    /// heads * head_dim` with `o_proj` as `heads * head_dim -> hidden`, so the
    /// two widths are decoupled by construction — `Qwen3NextConfig` ships
    /// hidden 2048 with 16 heads of 256. A parse that picked up the wrong
    /// sub-config is caught by [`TextEncoder::validate_tensors`] instead, which
    /// pins `q_proj`, `k_proj`, `o_proj` and `embed_tokens` against the shapes
    /// actually in the file: ground truth, and strictly stronger than an
    /// identity that need not hold.
    ///
    /// Every dimension is also held to [`MAX_LAYERS`] and friends, so nothing
    /// computed from a config — here or downstream — can overflow.
    pub fn validate(&self) -> Result<(), TextEncoderError> {
        let bad = |why: String| Err(TextEncoderError::BadConfig(why));
        // Ceilings before anything else: the checks below, and every caller of
        // a validated config, form products of these fields.
        for (key, value, ceiling) in [
            ("num_hidden_layers", self.num_layers, MAX_LAYERS),
            ("num_attention_heads", self.num_attention_heads, MAX_HEADS),
            ("num_key_value_heads", self.num_key_value_heads, MAX_HEADS),
            ("head_dim", self.head_dim, MAX_HEAD_DIM),
            ("hidden_size", self.hidden_size, MAX_WIDTH),
            ("intermediate_size", self.intermediate_size, MAX_WIDTH),
            ("vocab_size", self.vocab_size, MAX_VOCAB),
        ] {
            if value > ceiling {
                return bad(format!(
                    "`{key}` is {value}, above the {ceiling} this reference accepts"
                ));
            }
        }
        if self.num_layers == 0 {
            return bad("no layers".to_string());
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 || self.vocab_size == 0 {
            return bad("hidden, intermediate and vocab sizes must all be positive".to_string());
        }
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return bad(format!(
                "head_dim {} must be positive and even for rotary embeddings",
                self.head_dim
            ));
        }
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return bad("head counts must be positive".to_string());
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return bad(format!(
                "{} query heads do not split evenly over {} key/value heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        let half = self.head_dim / 2;
        // Bound each row's share before summing. `sum == half` over
        // non-negative entries already implies every entry is at most `half`,
        // so this refuses nothing a correct config could offer — but without
        // it a hostile pair such as `[2^63, 2^63, 64]` wraps a `usize` add
        // back onto `half` in release and sails past the check below.
        if let Some(&over) = self.mrope_section.iter().find(|&&n| n > half) {
            return bad(format!(
                "mrope_section {:?} gives one row {over} of the {half} rotary slots",
                self.mrope_section
            ));
        }
        let sum: usize = self.mrope_section.iter().sum();
        if sum != half {
            return bad(format!(
                "mrope_section {:?} sums to {sum}, but there are {half} rotary slots",
                self.mrope_section
            ));
        }
        // These two see only the stored f32, so they can only report the stored
        // f32. A file that spelled `1e-60` is caught earlier, by
        // `narrow_to_positive_f32`, which still has the f64 to quote; this is
        // the backstop for a config built in code.
        //
        // A NaN fails `is_finite`, so it is rejected before a comparison it
        // would otherwise slip past.
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return bad(format!("rope_theta {} must be positive", self.rope_theta));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return bad(format!(
                "rms_norm_eps {} must be positive",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }

    /// Query heads per key/value head.
    ///
    /// Assumes a config that has passed [`validate`](Self::validate), which is
    /// every config this module produces; a zero `num_key_value_heads` divides
    /// by zero.
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Rotary slots, which is half the head dimension.
    pub fn rope_half(&self) -> usize {
        self.head_dim / 2
    }
}

/// Looks a key up in the text sub-config first, then at the top level.
///
/// The fallback is per key rather than all or nothing, which is what lets a
/// composite config that leaves one rotary field at the root resolve.
struct ConfigView<'a> {
    text: Option<&'a Value>,
    root: &'a Value,
}

impl<'a> ConfigView<'a> {
    /// Build a view, rejecting a `text_config` that is present but unusable.
    ///
    /// Falling back to the root for every key of a `text_config` that is a
    /// string or an array would parse a corrupt file into a plausible config
    /// and say nothing. An absent key and an explicit JSON `null` both mean
    /// "not carried" here, as they do for every scalar below.
    fn new(root: &'a Value) -> Result<Self, TextEncoderError> {
        let text = match root.get("text_config") {
            None | Some(Value::Null) => None,
            Some(v) if v.is_object() => Some(v),
            Some(v) => {
                return Err(TextEncoderError::BadConfig(format!(
                    "`text_config` is {}, not an object",
                    json_kind(v)
                )))
            }
        };
        Ok(Self { text, root })
    }

    fn get(&self, key: &str) -> Option<&'a Value> {
        self.text
            .and_then(|t| t.get(key))
            .or_else(|| self.root.get(key))
    }
}

/// What a value is, for a message that says what the file held.
fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Narrow a config scalar to the f32 the forward pass runs in, reporting what
/// the file said rather than what the cast produced.
///
/// `1e-60` is a finite, positive f64 that becomes exactly `0.0` in f32, and
/// `1e60` becomes `inf`. Quoting the narrowed value misreports the file —
/// "rms_norm_eps 0 must be positive" for a file that plainly said `1e-60` —
/// so the check runs on both and the message quotes the f64.
fn narrow_to_positive_f32(key: &str, raw: f64) -> Result<f32, TextEncoderError> {
    let narrowed = raw as f32;
    if !raw.is_finite() || raw <= 0.0 || !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(TextEncoderError::BadConfig(format!(
            "`{key}` is {raw:e}, which is not a positive f32"
        )));
    }
    Ok(narrowed)
}

fn opt_usize(view: &ConfigView<'_>, key: &str) -> Result<Option<usize>, TextEncoderError> {
    match view.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| TextEncoderError::BadConfig(format!("`{key}` is not a whole number"))),
    }
}

fn need_usize(view: &ConfigView<'_>, key: &str) -> Result<usize, TextEncoderError> {
    opt_usize(view, key)?.ok_or_else(|| TextEncoderError::BadConfig(format!("`{key}` is missing")))
}

fn opt_f64(view: &ConfigView<'_>, key: &str) -> Result<Option<f64>, TextEncoderError> {
    match view.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| TextEncoderError::BadConfig(format!("`{key}` is not a number"))),
    }
}

fn parse_mrope_section(raw: &Value) -> Result<[usize; 3], TextEncoderError> {
    let items = raw.as_array().ok_or_else(|| {
        TextEncoderError::BadConfig("`mrope_section` is not an array".to_string())
    })?;
    if items.len() != 3 {
        return Err(TextEncoderError::BadConfig(format!(
            "`mrope_section` has {} entries, but there are three position rows (T, H, W)",
            items.len()
        )));
    }
    let mut out = [0usize; 3];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item.as_u64().ok_or_else(|| {
            TextEncoderError::BadConfig("`mrope_section` holds a non-integer".to_string())
        })? as usize;
    }
    Ok(out)
}

/// `<|image_pad|>` id from the config, or the documented Qwen3-VL default.
fn image_token_id_from_config(v: &Value) -> u32 {
    // `load` parses the hyperparameters first, so a malformed `text_config` has
    // already been reported by the time this runs; `ok()` keeps this helper
    // total rather than duplicating that diagnostic.
    ConfigView::new(v)
        .ok()
        .and_then(|view| view.get("image_token_id"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID)
}

/// Which of the three position rows supplies each rotary slot.
///
/// `recomposition_frequencies` starts from row T and overwrites
/// `slice(offset, mrope_section[dim] * 3, 3)` with row `dim`, for H at offset 1
/// and W at offset 2. Offsets 1 and 2 never collide under a stride of 3, so the
/// order of the two overwrites is irrelevant and the result is just an owner
/// per slot. A `stop` past the last slot clips, as Python slicing does.
fn mrope_slot_rows(cfg: &TextEncoderConfig) -> Vec<usize> {
    let half = cfg.rope_half();
    let mut rows = vec![0usize; half];
    for dim in 1..3 {
        let stop = (cfg.mrope_section[dim] * 3).min(half);
        let mut idx = dim;
        while idx < stop {
            rows[idx] = dim;
            idx += 3;
        }
    }
    rows
}

/// The `cos` and `sin` tables for one sequence, each `[seq, head_dim]`.
///
/// `positions` is the T, H and W position row, in that order; they must all
/// describe the same sequence. Each table's second half repeats its first,
/// which is what `torch.cat((freqs_thw, freqs_thw), dim=-1)` produces and what
/// makes the rotate-half form below a plain pairwise rotation.
///
/// The arithmetic is f32 throughout because the reference forces float32 here
/// (`maybe_autocast(..., enabled=False)`) and computes `base ** exponent` in
/// float32 too, so widening would drift from it rather than towards it.
///
/// It stops one line short of the reference, which ends
/// `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)` and so hands the
/// shipped bf16 run bf16 tables. These stay f32. The module's "Precision"
/// section carries the measured consequence: up to 1.95e-3 on `cos` against a
/// bf16 oracle, which is the floor for any comparison with one.
pub fn rope_tables(
    cfg: &TextEncoderConfig,
    positions: [&[f32]; 3],
) -> Result<(Matrix, Matrix), TextEncoderError> {
    cfg.validate()?;
    let seq = positions[0].len();
    if positions[1].len() != seq || positions[2].len() != seq {
        return Err(TextEncoderError::PositionRowsDiffer {
            lengths: [positions[0].len(), positions[1].len(), positions[2].len()],
        });
    }
    if seq == 0 {
        return Err(TextEncoderError::EmptyInput);
    }

    let head_dim = cfg.head_dim;
    let half = cfg.rope_half();
    let rows = mrope_slot_rows(cfg);
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0 / cfg.rope_theta.powf((2 * j) as f32 / head_dim as f32))
        .collect();

    let mut cos = Matrix::zeros(seq, head_dim);
    let mut sin = Matrix::zeros(seq, head_dim);
    for s in 0..seq {
        for j in 0..half {
            let angle = inv_freq[j] * positions[rows[j]][s];
            let (c, n) = (angle.cos(), angle.sin());
            let at = s * head_dim + j;
            cos.data[at] = c;
            cos.data[at + half] = c;
            sin.data[at] = n;
            sin.data[at + half] = n;
        }
    }
    Ok((cos, sin))
}

/// `x * cos + rotate_half(x) * sin`, applied to every head of a
/// `[seq, heads * head_dim]` matrix.
fn apply_rope(x: &Matrix, cos: &Matrix, sin: &Matrix, head_dim: usize) -> Matrix {
    let half = head_dim / 2;
    let heads = x.cols / head_dim;
    let mut out = Matrix::zeros(x.rows, x.cols);
    for s in 0..x.rows {
        let trig = s * head_dim;
        for h in 0..heads {
            let base = s * x.cols + h * head_dim;
            for j in 0..half {
                // `rotate_half(x)` is `[-x[half..], x[..half]]`, so the low half
                // pairs with a negated high element and the high half with a
                // plain low one.
                let lo = x.data[base + j];
                let hi = x.data[base + j + half];
                out.data[base + j] = lo * cos.data[trig + j] - hi * sin.data[trig + j];
                out.data[base + j + half] =
                    hi * cos.data[trig + j + half] + lo * sin.data[trig + j + half];
            }
        }
    }
    out
}

/// RMS normalise each head slice of a `[seq, heads * head_dim]` matrix.
///
/// `Qwen3VLTextAttention` builds its `q_norm`/`k_norm` over `head_dim` and
/// applies them after the head split ("only on the head dim!"), so the head
/// slices are exactly the rows of a `[seq * heads, head_dim]` view.
fn norm_per_head(m: Matrix, weight: &[f32], head_dim: usize, eps: f32) -> Matrix {
    let (rows, cols) = (m.rows, m.cols);
    let flat = Matrix::new(m.data.len() / head_dim, head_dim, m.data);
    let normed = rms_norm_rows(&flat, weight, eps);
    Matrix::new(rows, cols, normed.data)
}

/// `true` wherever a token is the image placeholder.
///
/// The pipeline builds this from `input_ids == self._img_token_id` and carries
/// it alongside the hidden states so the transformer knows which encoder rows
/// describe a conditioning image.
///
/// This is the mask for the **whole** prompt, template included, and the
/// pipeline slices `[self._drop_idx:]` off it exactly as it does off the
/// hidden states (`pipeline_qwenimage21.py:317-321`). A caller that drops the
/// leading [`SYSTEM_PREFIX_TOKENS`] rows of [`TextEncoder::forward`]'s output
/// must drop the same rows here, or the mask will be 14 entries longer than
/// the states it describes.
pub fn image_pad_mask(token_ids: &[u32], image_pad_id: u32) -> Vec<bool> {
    token_ids.iter().map(|&t| t == image_pad_id).collect()
}

/// Name of one of a layer's tensors.
fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("{PREFIX}layers.{layer}.{suffix}")
}

/// Name of a top-level tensor.
fn tower_tensor(suffix: &str) -> String {
    format!("{PREFIX}{suffix}")
}

/// The CPU reference for the text tower.
#[derive(Debug)]
pub struct TextEncoder {
    file: LbiFile,
    config: TextEncoderConfig,
    image_pad_token_id: u32,
}

impl TextEncoder {
    /// Open a text-encoder `.lbi` and check it against its own config.
    ///
    /// Every expected tensor's name, shape and dtype is verified from the index
    /// alone, so a checkpoint that cannot complete a forward pass fails here
    /// rather than halfway through one. No weight bytes are decoded.
    pub fn load(path: &Path) -> Result<Self, TextEncoderError> {
        let file = LbiFile::open(path)?;
        let config = TextEncoderConfig::from_lbi_config(file.config())?;
        let image_pad_token_id = image_token_id_from_config(file.config());
        let encoder = Self {
            file,
            config,
            image_pad_token_id,
        };
        encoder.validate_tensors()?;
        Ok(encoder)
    }

    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// The id the prompt uses for an image placeholder, for
    /// [`image_pad_mask`].
    pub fn image_pad_token_id(&self) -> u32 {
        self.image_pad_token_id
    }

    /// Every tensor the forward pass reads, plus the unused final norm.
    fn manifest(&self) -> Vec<(String, Vec<u64>)> {
        let c = &self.config;
        let hidden = c.hidden_size as u64;
        let inter = c.intermediate_size as u64;
        let q_width = (c.num_attention_heads * c.head_dim) as u64;
        let kv_width = (c.num_key_value_heads * c.head_dim) as u64;
        let head = c.head_dim as u64;

        // `validate` held `num_layers` to `MAX_LAYERS`, so this is at most
        // 11266 and the product cannot overflow.
        let mut out = Vec::with_capacity(2 + c.num_layers * 11);
        out.push((
            tower_tensor("embed_tokens.weight"),
            vec![c.vocab_size as u64, hidden],
        ));
        out.push((tower_tensor("norm.weight"), vec![hidden]));
        for i in 0..c.num_layers {
            out.push((layer_tensor(i, "input_layernorm.weight"), vec![hidden]));
            out.push((
                layer_tensor(i, "self_attn.q_proj.weight"),
                vec![q_width, hidden],
            ));
            out.push((
                layer_tensor(i, "self_attn.k_proj.weight"),
                vec![kv_width, hidden],
            ));
            out.push((
                layer_tensor(i, "self_attn.v_proj.weight"),
                vec![kv_width, hidden],
            ));
            out.push((
                layer_tensor(i, "self_attn.o_proj.weight"),
                vec![hidden, q_width],
            ));
            out.push((layer_tensor(i, "self_attn.q_norm.weight"), vec![head]));
            out.push((layer_tensor(i, "self_attn.k_norm.weight"), vec![head]));
            out.push((
                layer_tensor(i, "post_attention_layernorm.weight"),
                vec![hidden],
            ));
            out.push((layer_tensor(i, "mlp.gate_proj.weight"), vec![inter, hidden]));
            out.push((layer_tensor(i, "mlp.up_proj.weight"), vec![inter, hidden]));
            out.push((layer_tensor(i, "mlp.down_proj.weight"), vec![hidden, inter]));
        }
        out
    }

    fn validate_tensors(&self) -> Result<(), TextEncoderError> {
        for (name, shape) in self.manifest() {
            let entry = self
                .file
                .get(&name)
                .ok_or_else(|| TextEncoderError::MissingTensor { name: name.clone() })?;
            if entry.shape != shape {
                return Err(TextEncoderError::ShapeMismatch {
                    name,
                    expected: shape,
                    found: entry.shape.clone(),
                });
            }
            // Unreachable as the container stands: `LbiFile::open` has already
            // rejected a quantized entry, having no storage rule to size one
            // with. Kept so this module states the schemes it decodes rather
            // than inheriting them. See `TextEncoderError::UnsupportedScheme`.
            if !matches!(
                entry.quant,
                QuantScheme::F32 | QuantScheme::F16 | QuantScheme::Bf16
            ) {
                return Err(TextEncoderError::UnsupportedScheme {
                    name,
                    scheme: entry.quant,
                });
            }
        }
        Ok(())
    }

    /// Decode a whole tensor to f32.
    fn read(&self, name: &str) -> Result<Vec<f32>, TextEncoderError> {
        if self.file.get(name).is_none() {
            return Err(TextEncoderError::MissingTensor {
                name: name.to_string(),
            });
        }
        Ok(self.file.read_f32(name)?)
    }

    /// A norm's learned scale.
    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, TextEncoderError> {
        let data = self.read(name)?;
        if data.len() != len {
            return Err(self.shape_error(name, vec![len as u64]));
        }
        Ok(data)
    }

    /// Apply one `nn.Linear`, holding its f32 weight only for this call.
    fn project(&self, x: &Matrix, name: &str, out_dim: usize) -> Result<Matrix, TextEncoderError> {
        let data = self.read(name)?;
        if data.len() != out_dim * x.cols {
            return Err(self.shape_error(name, vec![out_dim as u64, x.cols as u64]));
        }
        let weight = Matrix::new(out_dim, x.cols, data);
        Ok(x.linear(&weight, None))
    }

    fn shape_error(&self, name: &str, expected: Vec<u64>) -> TextEncoderError {
        TextEncoderError::ShapeMismatch {
            name: name.to_string(),
            expected,
            found: self
                .file
                .get(name)
                .map(|e| e.shape.clone())
                .unwrap_or_default(),
        }
    }

    /// Gather the prompt's embedding rows straight out of the mapping.
    fn embed(&self, token_ids: &[u32]) -> Result<Matrix, TextEncoderError> {
        let name = tower_tensor("embed_tokens.weight");
        let entry = self
            .file
            .get(&name)
            .ok_or_else(|| TextEncoderError::MissingTensor { name: name.clone() })?;
        let quant = entry.quant;
        let bytes = self
            .file
            .tensor_bytes(&name)
            .ok_or_else(|| TextEncoderError::MissingTensor { name: name.clone() })?;
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        // `validate_tensors` pinned the shape to [vocab, hidden] and the
        // container pinned the byte length to that shape, so the stride divides
        // exactly and an in-range row always lies inside the mapping.
        let stride = bytes.len() / vocab;

        let mut out = Matrix::zeros(token_ids.len(), hidden);
        for (r, &id) in token_ids.iter().enumerate() {
            let row = id as usize;
            if row >= vocab {
                return Err(TextEncoderError::TokenOutOfRange { id, vocab });
            }
            let start = row * stride;
            decode_row(&name, quant, &bytes[start..start + stride], out.row_mut(r))?;
        }
        Ok(out)
    }

    /// One attention block: per-head norms, rotary, causal GQA softmax, output
    /// projection.
    fn attention(
        &self,
        x: &Matrix,
        layer: usize,
        cos: &Matrix,
        sin: &Matrix,
    ) -> Result<Matrix, TextEncoderError> {
        let c = &self.config;
        let (hd, seq) = (c.head_dim, x.rows);
        let (nq, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let groups = c.num_key_value_groups();
        let (q_width, kv_width) = (nq * hd, nkv * hd);

        let q = self.project(x, &layer_tensor(layer, "self_attn.q_proj.weight"), q_width)?;
        let q_scale = self.vector(&layer_tensor(layer, "self_attn.q_norm.weight"), hd)?;
        let q = apply_rope(
            &norm_per_head(q, &q_scale, hd, c.rms_norm_eps),
            cos,
            sin,
            hd,
        );

        let k = self.project(x, &layer_tensor(layer, "self_attn.k_proj.weight"), kv_width)?;
        let k_scale = self.vector(&layer_tensor(layer, "self_attn.k_norm.weight"), hd)?;
        let k = apply_rope(
            &norm_per_head(k, &k_scale, hd, c.rms_norm_eps),
            cos,
            sin,
            hd,
        );

        // `v` carries no norm and no rotary.
        let v = self.project(x, &layer_tensor(layer, "self_attn.v_proj.weight"), kv_width)?;

        let scaling = (hd as f32).powf(-0.5);
        let mut context = Matrix::zeros(seq, q_width);
        let mut weights = vec![0f32; seq];
        for h in 0..nq {
            // `repeat_kv` is a repeat_interleave over the head axis, so a block
            // of `groups` consecutive query heads shares one key/value head.
            let kv = h / groups;
            for i in 0..seq {
                let q_at = i * q_width + h * hd;
                // Causality by construction: the reference adds -inf to the
                // future scores and softmaxes them to zero, which is the same
                // as never forming them.
                let mut peak = f32::NEG_INFINITY;
                for j in 0..=i {
                    let k_at = j * kv_width + kv * hd;
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += q.data[q_at + d] * k.data[k_at + d];
                    }
                    let score = dot * scaling;
                    weights[j] = score;
                    if score > peak {
                        peak = score;
                    }
                }
                let mut total = 0f32;
                for w in weights.iter_mut().take(i + 1) {
                    *w = (*w - peak).exp();
                    total += *w;
                }
                let inv_total = 1.0 / total;
                for j in 0..=i {
                    let p = weights[j] * inv_total;
                    let v_at = j * kv_width + kv * hd;
                    for d in 0..hd {
                        context.data[q_at + d] += p * v.data[v_at + d];
                    }
                }
            }
        }

        self.project(
            &context,
            &layer_tensor(layer, "self_attn.o_proj.weight"),
            c.hidden_size,
        )
    }

    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    fn mlp(&self, x: &Matrix, layer: usize) -> Result<Matrix, TextEncoderError> {
        let inter = self.config.intermediate_size;
        let mut gate = self.project(x, &layer_tensor(layer, "mlp.gate_proj.weight"), inter)?;
        let up = self.project(x, &layer_tensor(layer, "mlp.up_proj.weight"), inter)?;
        for (g, &u) in gate.data.iter_mut().zip(&up.data) {
            *g = silu(*g) * u;
        }
        self.project(
            &gate,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            self.config.hidden_size,
        )
    }

    /// One decoder layer: pre-norm attention and pre-norm MLP, each residual.
    fn decoder_layer(
        &self,
        hidden: &Matrix,
        layer: usize,
        cos: &Matrix,
        sin: &Matrix,
    ) -> Result<Matrix, TextEncoderError> {
        let eps = self.config.rms_norm_eps;
        let width = self.config.hidden_size;

        let scale = self.vector(&layer_tensor(layer, "input_layernorm.weight"), width)?;
        let attended = self.attention(&rms_norm_rows(hidden, &scale, eps), layer, cos, sin)?;
        let hidden = add(hidden, &attended);

        let scale = self.vector(
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            width,
        )?;
        let expanded = self.mlp(&rms_norm_rows(&hidden, &scale, eps), layer)?;
        Ok(add(&hidden, &expanded))
    }

    /// Encode a prompt to `[seq, hidden]`, **without** the final RMS norm.
    ///
    /// All three M-RoPE rows are the plain `0..seq-1`: for text-to-image the
    /// model is handed no grid, so it falls back to `arange` broadcast across
    /// T, H and W, and the recomposition below has nothing to interleave.
    ///
    /// The caller gets every row, including the template's. The pipeline drops
    /// the leading system-message rows ([`SYSTEM_PREFIX_TOKENS`] of them for
    /// the shipped prompt) before conditioning the transformer.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Matrix, TextEncoderError> {
        if token_ids.is_empty() {
            return Err(TextEncoderError::EmptyInput);
        }
        let positions: Vec<f32> = (0..token_ids.len()).map(|i| i as f32).collect();
        let (cos, sin) = rope_tables(&self.config, [&positions, &positions, &positions])?;

        let mut hidden = self.embed(token_ids)?;
        for layer in 0..self.config.num_layers {
            hidden = self.decoder_layer(&hidden, layer, &cos, &sin)?;
        }
        Ok(hidden)
    }
}

/// Decode one stored row into f32.
///
/// `bytes` must hold exactly as many values as `out` has slots. The three
/// gates upstream — the container's own length check, `validate_tensors`
/// pinning the embedding to `[vocab, hidden]`, and the stride `embed` derives
/// from those — make a disagreement unreachable from [`TextEncoder`] today.
/// It is still an error rather than a `zip` that stops at the shorter side,
/// because a short row would otherwise leave the tail of `out` at whatever it
/// held before and report success, and because this decoder is a second copy
/// of `LbiFile::read_f32`'s and the two can drift apart.
fn decode_row(
    name: &str,
    quant: QuantScheme,
    bytes: &[u8],
    out: &mut [f32],
) -> Result<(), TextEncoderError> {
    let width = match quant {
        QuantScheme::F32 => 4,
        QuantScheme::F16 | QuantScheme::Bf16 => 2,
        // Unreachable: see `TextEncoderError::UnsupportedScheme`. The arm is
        // here because the match must be exhaustive.
        scheme => {
            return Err(TextEncoderError::UnsupportedScheme {
                name: name.to_string(),
                scheme,
            })
        }
    };
    // `out.len()` is a hidden size, which `validate` held under `MAX_WIDTH`, so
    // this product is at most 2^22.
    if bytes.len() != out.len() * width {
        return Err(TextEncoderError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![out.len() as u64],
            found: vec![(bytes.len() / width) as u64],
        });
    }
    match quant {
        QuantScheme::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        QuantScheme::Bf16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes(c.try_into().unwrap());
                *o = f32::from_bits((bits as u32) << 16);
            }
        }
        QuantScheme::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = half_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
            }
        }
        // Rejected above, before any byte was read.
        scheme => {
            return Err(TextEncoderError::UnsupportedScheme {
                name: name.to_string(),
                scheme,
            })
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lbi::LbiWriter;
    use std::path::PathBuf;

    /// A config small enough to hand-check, with the three rotary rows given
    /// two slots each.
    fn hand_config() -> TextEncoderConfig {
        TextEncoderConfig {
            num_layers: 1,
            hidden_size: 12,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 12,
            intermediate_size: 4,
            vocab_size: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            mrope_section: [2, 2, 2],
        }
    }

    /// The fixture used for the multi-head tests: 2 layers, 4 query heads over
    /// 2 key/value heads, head_dim 2.
    fn small_config(num_layers: usize) -> TextEncoderConfig {
        TextEncoderConfig {
            num_layers,
            hidden_size: 8,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 2,
            intermediate_size: 6,
            vocab_size: 6,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [1, 0, 0],
        }
    }

    /// One query head of width 2 over a hidden size of 2: the smallest shape
    /// in which a projection can be asymmetric, so a transposed `[out, in]`
    /// read is a different matrix rather than the same one.
    ///
    /// `head_dim` 2 leaves a single rotary slot whose inverse frequency is
    /// `theta^0 = 1`, so the rotation angle *is* the position.
    fn single_head_config(rms_norm_eps: f32) -> TextEncoderConfig {
        TextEncoderConfig {
            num_layers: 1,
            hidden_size: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: 2,
            rms_norm_eps,
            rope_theta: 10.0,
            mrope_section: [1, 0, 0],
        }
    }

    /// Layer 0's attention weights for [`single_head_config`].
    ///
    /// Every projection is asymmetric, so reading one as `[in, out]` selects a
    /// different column; `o_proj`'s rows differ from each other, so its output
    /// cannot be mistaken for its input; and no matrix has a zero column, so
    /// no input lane is discarded.
    fn attention_weights(q_norm: [f32; 2], k_norm: [f32; 2]) -> impl Fn(&str, usize) -> Vec<f32> {
        move |name: &str, n: usize| match name {
            "model.language_model.layers.0.self_attn.q_proj.weight" => vec![1.0, 2.0, 4.0, 3.0],
            "model.language_model.layers.0.self_attn.k_proj.weight" => vec![2.0, 5.0, 1.0, 3.0],
            "model.language_model.layers.0.self_attn.v_proj.weight" => vec![1.0, 2.0, 3.0, -1.0],
            "model.language_model.layers.0.self_attn.o_proj.weight" => vec![1.0, 2.0, 3.0, 5.0],
            "model.language_model.layers.0.self_attn.q_norm.weight" => q_norm.to_vec(),
            "model.language_model.layers.0.self_attn.k_norm.weight" => k_norm.to_vec(),
            _ => vec![0f32; n],
        }
    }

    /// The shipped config spelled flat, the way `lbi-convert` stores a bare
    /// text config, using the older `rope_scaling` key.
    fn flat_config_json() -> Value {
        serde_json::json!({
            "num_hidden_layers": 36,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 12288,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000.0,
            "rope_scaling": {"type": "default", "mrope_section": [24, 20, 20]},
        })
    }

    fn config_json(cfg: &TextEncoderConfig, image_token_id: Option<u32>) -> Value {
        let mut root = serde_json::json!({
            "text_config": {
                "num_hidden_layers": cfg.num_layers,
                "hidden_size": cfg.hidden_size,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "intermediate_size": cfg.intermediate_size,
                "vocab_size": cfg.vocab_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "hidden_act": "silu",
                "attention_bias": false,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": cfg.rope_theta,
                    "mrope_section": cfg.mrope_section,
                },
            }
        });
        if let Some(id) = image_token_id {
            root["image_token_id"] = serde_json::json!(id);
        }
        root
    }

    /// The checkpoint's tensor names, spelled out rather than borrowed from the
    /// implementation, so this fixture pins the strings `load` looks for.
    fn manifest(cfg: &TextEncoderConfig) -> Vec<(String, Vec<u64>)> {
        let hidden = cfg.hidden_size as u64;
        let inter = cfg.intermediate_size as u64;
        let q_width = (cfg.num_attention_heads * cfg.head_dim) as u64;
        let kv_width = (cfg.num_key_value_heads * cfg.head_dim) as u64;
        let head = cfg.head_dim as u64;
        let mut out = vec![
            (
                "model.language_model.embed_tokens.weight".to_string(),
                vec![cfg.vocab_size as u64, hidden],
            ),
            ("model.language_model.norm.weight".to_string(), vec![hidden]),
        ];
        for i in 0..cfg.num_layers {
            out.extend([
                (
                    format!("model.language_model.layers.{i}.input_layernorm.weight"),
                    vec![hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.q_proj.weight"),
                    vec![q_width, hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.k_proj.weight"),
                    vec![kv_width, hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.v_proj.weight"),
                    vec![kv_width, hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.o_proj.weight"),
                    vec![hidden, q_width],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.q_norm.weight"),
                    vec![head],
                ),
                (
                    format!("model.language_model.layers.{i}.self_attn.k_norm.weight"),
                    vec![head],
                ),
                (
                    format!("model.language_model.layers.{i}.post_attention_layernorm.weight"),
                    vec![hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.mlp.gate_proj.weight"),
                    vec![inter, hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.mlp.up_proj.weight"),
                    vec![inter, hidden],
                ),
                (
                    format!("model.language_model.layers.{i}.mlp.down_proj.weight"),
                    vec![hidden, inter],
                ),
            ]);
        }
        out
    }

    /// A deterministic LCG, so fixtures need no `rand`.
    fn pseudo(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32) / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn name_seed(name: &str) -> u64 {
        name.bytes().fold(1_469_598_103_934_665_603u64, |h, b| {
            (h ^ b as u64).wrapping_mul(1_099_511_628_211)
        })
    }

    fn random_fill(name: &str, n: usize) -> Vec<f32> {
        pseudo(name_seed(name), n)
    }

    fn write_model(
        tag: &str,
        entries: &[(String, Vec<u64>)],
        config: Value,
        fill: &dyn Fn(&str, usize) -> Vec<f32>,
    ) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "text-encoder-test-{}-{tag}.lbi",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut w = LbiWriter::create(&path, config).expect("create");
        for (name, shape) in entries {
            let n: u64 = shape.iter().product();
            let values = fill(name, n as usize);
            assert_eq!(
                values.len(),
                n as usize,
                "fill produced {name} at a bad size"
            );
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in &values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            w.append(name, shape, QuantScheme::F32, &bytes)
                .expect("append");
        }
        w.finish().expect("finish");
        path
    }

    fn load_random(tag: &str, cfg: &TextEncoderConfig) -> (TextEncoder, PathBuf) {
        let path = write_model(tag, &manifest(cfg), config_json(cfg, None), &random_fill);
        let enc = TextEncoder::load(&path).expect("load");
        (enc, path)
    }

    // ---------------------------------------------------------------- rope

    /// Hand-check the M-RoPE recomposition with three different position rows.
    ///
    /// With `mrope_section = [2, 2, 2]` over six slots, H takes
    /// `slice(1, 6, 3)` = {1, 4} and W takes `slice(2, 6, 3)` = {2, 5}, leaving
    /// T with {0, 3}. Each slot's angle must be that row's position times that
    /// slot's inverse frequency.
    #[test]
    fn rope_recomposition_interleaves_the_three_position_rows() {
        let cfg = hand_config();
        let (t, h, w) = ([1.0f32], [2.0f32], [3.0f32]);
        let (cos, sin) = rope_tables(&cfg, [&t, &h, &w]).expect("tables");
        assert_eq!((cos.rows, cos.cols), (1, 12));

        let half = 6;
        let owner_position = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
        for j in 0..half {
            let inv_freq = 1.0f32 / 100.0f32.powf((2 * j) as f32 / 12.0);
            let angle = inv_freq * owner_position[j];
            assert!(
                (cos.data[j] - angle.cos()).abs() < 1e-6,
                "cos slot {j} is {}, want {}",
                cos.data[j],
                angle.cos()
            );
            assert!(
                (sin.data[j] - angle.sin()).abs() < 1e-6,
                "sin slot {j} is {}, want {}",
                sin.data[j],
                angle.sin()
            );
            // `torch.cat((freqs_thw, freqs_thw))`: the halves are identical.
            assert_eq!(cos.data[j], cos.data[j + half], "cos half {j}");
            assert_eq!(sin.data[j], sin.data[j + half], "sin half {j}");
        }

        // Skipping the recomposition would leave every slot on row T, so the
        // H- and W-owned slots must differ from the all-T table while the
        // T-owned ones match it.
        let (all_t, _) = rope_tables(&cfg, [&t, &t, &t]).expect("tables");
        for j in [0usize, 3] {
            assert_eq!(cos.data[j], all_t.data[j], "slot {j} should belong to T");
        }
        for j in [1usize, 2, 4, 5] {
            assert!(
                (cos.data[j] - all_t.data[j]).abs() > 1e-6,
                "slot {j} should not come from T"
            );
        }
    }

    /// A `stop` past the last slot must clip, as Python slicing does.
    ///
    /// The shipped `[24, 20, 20]` stops H and W at 60 of 64, inside the owner
    /// vector, so it never exercises the clip; `[4, 30, 30]` asks for `30 * 3
    /// = 90` slots and does.
    #[test]
    fn mrope_slot_owners_clip_like_python_slicing() {
        let clipping = TextEncoderConfig {
            mrope_section: [4, 30, 30],
            ..TextEncoderConfig::qwen_image_21()
        };
        clipping
            .validate()
            .expect("a section summing to the slot count is valid");
        // Both overwrites run the whole way, so every slot falls to `j % 3`.
        let owners = mrope_slot_rows(&clipping);
        assert_eq!(owners, (0..64).map(|j| j % 3).collect::<Vec<_>>());

        // The shipped section leaves a genuine four-slot T tail: 61 and 62
        // would be H's and W's under an unclipped stride, and are T's here.
        let shipped = mrope_slot_rows(&TextEncoderConfig::qwen_image_21());
        assert_eq!(shipped.len(), 64);
        for (j, &owner) in shipped.iter().enumerate().take(60) {
            assert_eq!(owner, j % 3, "slot {j}");
        }
        assert_eq!(&shipped[60..], &[0usize, 0, 0, 0][..]);
    }

    /// Position 0 must rotate by nothing at all, which is what fixes the
    /// schedule's origin: `cos` is exactly one and `sin` exactly zero there.
    #[test]
    fn rope_position_zero_is_the_identity_rotation() {
        let cfg = hand_config();
        let zero = [0.0f32];
        let (cos, sin) = rope_tables(&cfg, [&zero, &zero, &zero]).expect("tables");
        assert!(cos.data.iter().all(|&c| c == 1.0), "cos {:?}", cos.data);
        assert!(sin.data.iter().all(|&s| s == 0.0), "sin {:?}", sin.data);

        // And the same holds for the first row of a longer sequence, so a
        // schedule numbered from one would show up here.
        let seq = [0.0f32, 1.0, 2.0];
        let (cos, sin) = rope_tables(&cfg, [&seq, &seq, &seq]).expect("tables");
        assert!(cos.row(0).iter().all(|&c| c == 1.0), "cos {:?}", cos.row(0));
        assert!(sin.row(0).iter().all(|&s| s == 0.0), "sin {:?}", sin.row(0));
        assert!(sin.row(1).iter().any(|&s| s != 0.0), "row 1 did not rotate");
    }

    #[test]
    fn rope_rejects_ragged_and_empty_position_rows() {
        let cfg = hand_config();
        let two = [0.0f32, 1.0];
        let one = [0.0f32];
        assert!(matches!(
            rope_tables(&cfg, [&two, &one, &two]),
            Err(TextEncoderError::PositionRowsDiffer { lengths: [2, 1, 2] })
        ));
        let none: [f32; 0] = [];
        assert!(matches!(
            rope_tables(&cfg, [&none, &none, &none]),
            Err(TextEncoderError::EmptyInput)
        ));
    }

    /// The rotation is the rotate-half form, not a pairwise-interleaved one.
    #[test]
    fn apply_rope_rotates_the_two_halves() {
        // head_dim 4, one head, one row; cos/sin are built with both halves
        // equal, as `rope_tables` does.
        let (c0, c1) = (0.6f32, 0.8);
        let (s0, s1) = (0.8f32, 0.6);
        let cos = Matrix::new(1, 4, vec![c0, c1, c0, c1]);
        let sin = Matrix::new(1, 4, vec![s0, s1, s0, s1]);
        let x = Matrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let out = apply_rope(&x, &cos, &sin, 4);
        let want = [
            1.0 * c0 - 3.0 * s0,
            2.0 * c1 - 4.0 * s1,
            3.0 * c0 + 1.0 * s0,
            4.0 * c1 + 2.0 * s1,
        ];
        for (got, want) in out.data.iter().zip(&want) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    // ------------------------------------------------------------- config

    #[test]
    fn shipped_config_is_self_consistent() {
        let cfg = TextEncoderConfig::qwen_image_21();
        cfg.validate().expect("the shipped config must validate");
        assert_eq!(cfg.num_key_value_groups(), 4);
        assert_eq!(cfg.rope_half(), 64);
        assert_eq!(cfg.mrope_section.iter().sum::<usize>(), 64);
    }

    #[test]
    fn config_parses_nested_and_flat_spellings_alike() {
        let cfg = TextEncoderConfig::qwen_image_21();
        let nested = config_json(&cfg, None);
        assert_eq!(TextEncoderConfig::from_lbi_config(&nested).unwrap(), cfg);

        // A bare text config using the older `rope_scaling` spelling.
        assert_eq!(
            TextEncoderConfig::from_lbi_config(&flat_config_json()).unwrap(),
            cfg
        );
    }

    /// `rope_type: "mrope"` is accepted as a legacy alias, defensively.
    ///
    /// This pins *our* tolerance, not transformers' behaviour: transformers
    /// 5.2.0 rejects the spelling outright, since `ROPE_INIT_FUNCTIONS` holds
    /// only `dynamic`, `linear`, `llama3`, `longrope` and `yarn`, and
    /// `Qwen3VLTextRotaryEmbedding` raises `KeyError: 'mrope'` for anything
    /// else. `"default"` is what this checkpoint is expected to carry; both
    /// spellings select the schedule this module implements.
    #[test]
    fn mrope_rope_type_is_accepted_as_a_legacy_alias() {
        let cfg = TextEncoderConfig::qwen_image_21();
        for spelling in ["default", "mrope"] {
            let mut v = flat_config_json();
            v["rope_scaling"]["type"] = serde_json::json!(spelling);
            assert_eq!(
                TextEncoderConfig::from_lbi_config(&v).unwrap(),
                cfg,
                "{spelling} should be accepted"
            );
        }
    }

    /// A `text_config` that is present but is not an object is a corrupt file,
    /// not a config to fall back from.
    #[test]
    fn config_rejects_a_non_object_text_config() {
        let cfg = TextEncoderConfig::qwen_image_21();
        // The root alone parses, so only `text_config` is at issue below.
        assert_eq!(
            TextEncoderConfig::from_lbi_config(&flat_config_json()).unwrap(),
            cfg
        );

        for junk in [
            serde_json::json!("oops"),
            serde_json::json!([1, 2, 3]),
            serde_json::json!(7),
            serde_json::json!(true),
        ] {
            let mut v = flat_config_json();
            v["text_config"] = junk.clone();
            match TextEncoderConfig::from_lbi_config(&v) {
                Err(TextEncoderError::BadConfig(why)) => {
                    assert!(why.contains("text_config"), "{why}")
                }
                other => panic!("text_config {junk} should be rejected, got {other:?}"),
            }
        }

        // An explicit null means "not carried", as it does for every scalar.
        let mut null = flat_config_json();
        null["text_config"] = Value::Null;
        assert_eq!(TextEncoderConfig::from_lbi_config(&null).unwrap(), cfg);
    }

    /// The root fallback is per key, not all or nothing.
    #[test]
    fn config_falls_back_to_the_root_one_key_at_a_time() {
        let cfg = TextEncoderConfig::qwen_image_21();
        let flat = flat_config_json();
        let mut root = flat.as_object().unwrap().clone();
        // Two keys move into an otherwise sparse `text_config`; everything
        // else must still resolve at the root.
        let moved = serde_json::json!({
            "hidden_size": root.remove("hidden_size").unwrap(),
            "num_hidden_layers": root.remove("num_hidden_layers").unwrap(),
        });
        root.insert("text_config".to_string(), moved);
        assert_eq!(
            TextEncoderConfig::from_lbi_config(&Value::Object(root)).unwrap(),
            cfg
        );
    }

    /// No config value, however hostile, may wrap a comparison or panic.
    ///
    /// Each of these once reached arithmetic that overflowed: `num_layers * 11`
    /// sizing the manifest, `heads * head_dim` in the old width check, and a
    /// `mrope_section` whose sum wrapped back onto the slot count in release
    /// and so slipped past the check meant to catch it.
    #[test]
    fn config_rejects_dimensions_beyond_the_ceilings() {
        let bad = |v: Value| match TextEncoderConfig::from_lbi_config(&v) {
            Err(TextEncoderError::BadConfig(why)) => why,
            other => panic!("expected BadConfig, got {other:?}"),
        };
        for key in [
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
        ] {
            for value in [u64::MAX, 1u64 << 62] {
                let mut v = flat_config_json();
                v[key] = serde_json::json!(value);
                let why = bad(v);
                assert!(why.contains(key), "{key} = {value}: {why}");
            }
        }

        let mut wrapping = flat_config_json();
        let half_of_usize = 1u64 << 63;
        wrapping["rope_scaling"]["mrope_section"] =
            serde_json::json!([half_of_usize, half_of_usize, 64]);
        let why = bad(wrapping);
        assert!(why.contains("mrope_section"), "{why}");

        // The shipped config is nowhere near any ceiling.
        TextEncoderConfig::qwen_image_21().validate().unwrap();
    }

    /// A scalar that cannot be narrowed must be reported as the file spelled
    /// it, not as the narrowing left it.
    #[test]
    fn config_reports_a_scalar_as_the_file_spelled_it() {
        let bad = |key: &str, value: Value| {
            let mut v = flat_config_json();
            v[key] = value;
            match TextEncoderConfig::from_lbi_config(&v) {
                Err(TextEncoderError::BadConfig(why)) => why,
                other => panic!("expected BadConfig, got {other:?}"),
            }
        };
        // 1e-60 is a perfectly good positive f64 that narrows to exactly 0.0,
        // and 1e60 narrows to infinity. Quoting the narrowed value would
        // report a number the file never held.
        for key in ["rms_norm_eps", "rope_theta"] {
            let why = bad(key, serde_json::json!(1e-60));
            assert!(why.contains("1e-60"), "{key}: {why}");
            let why = bad(key, serde_json::json!(1e60));
            assert!(why.contains("1e60"), "{key}: {why}");
            let why = bad(key, serde_json::json!(-1.0));
            assert!(why.contains(key), "{key}: {why}");
        }
    }

    /// The count of template rows the pipeline drops, which no other test
    /// pins.
    #[test]
    fn system_prefix_matches_the_shipped_template() {
        assert_eq!(SYSTEM_PREFIX_TOKENS, 14);
        // Where the 14 comes from: the reference bundle's 38-token prompt
        // yields 24 encoder rows.
        assert_eq!(38 - SYSTEM_PREFIX_TOKENS, 24);
    }

    #[test]
    fn config_rejects_what_it_cannot_run() {
        let bad = |v: Value| match TextEncoderConfig::from_lbi_config(&v) {
            Err(TextEncoderError::BadConfig(why)) => why,
            other => panic!("expected BadConfig, got {other:?}"),
        };
        let base = config_json(&TextEncoderConfig::qwen_image_21(), None);

        let mut missing = base.clone();
        missing["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("hidden_size");
        assert!(bad(missing).contains("hidden_size"));

        let mut ragged = base.clone();
        ragged["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([24, 20, 19]);
        assert!(bad(ragged).contains("sums to 63"));

        let mut scaled = base.clone();
        scaled["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        assert!(bad(scaled).contains("yarn"));

        let mut gelu = base.clone();
        gelu["text_config"]["hidden_act"] = serde_json::json!("gelu");
        assert!(bad(gelu).contains("gelu"));

        let mut biased = base;
        biased["text_config"]["attention_bias"] = serde_json::json!(true);
        assert!(bad(biased).contains("attention_bias"));
    }

    #[test]
    fn image_pad_id_comes_from_the_config_then_the_default() {
        let cfg = small_config(1);
        let with_id = write_model(
            "img-id",
            &manifest(&cfg),
            config_json(&cfg, Some(4242)),
            &random_fill,
        );
        assert_eq!(
            TextEncoder::load(&with_id).unwrap().image_pad_token_id(),
            4242
        );
        let _ = std::fs::remove_file(&with_id);

        let without = write_model(
            "no-img-id",
            &manifest(&cfg),
            config_json(&cfg, None),
            &random_fill,
        );
        assert_eq!(
            TextEncoder::load(&without).unwrap().image_pad_token_id(),
            DEFAULT_IMAGE_TOKEN_ID
        );
        assert_eq!(DEFAULT_IMAGE_TOKEN_ID, 151_655);
        let _ = std::fs::remove_file(&without);
    }

    #[test]
    fn image_pad_mask_marks_exactly_the_placeholder() {
        let ids = [151_644u32, 151_655, 9, 151_655];
        assert_eq!(
            image_pad_mask(&ids, DEFAULT_IMAGE_TOKEN_ID),
            vec![false, true, false, true]
        );
        assert!(image_pad_mask(&[], DEFAULT_IMAGE_TOKEN_ID).is_empty());
    }

    // -------------------------------------------------------------- loading

    #[test]
    fn load_accepts_exactly_the_checkpoint_names() {
        let cfg = small_config(2);
        let (enc, path) = load_random("load-ok", &cfg);
        assert_eq!(enc.config(), &cfg);
        // The final norm is in the manifest even though `forward` never reads it.
        assert!(enc
            .manifest()
            .iter()
            .any(|(n, _)| n == "model.language_model.norm.weight"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_names_the_tensor_a_checkpoint_is_missing() {
        let cfg = small_config(2);
        let mut entries = manifest(&cfg);
        let dropped = "model.language_model.layers.1.self_attn.k_norm.weight";
        entries.retain(|(n, _)| n != dropped);
        let path = write_model(
            "load-missing",
            &entries,
            config_json(&cfg, None),
            &random_fill,
        );
        match TextEncoder::load(&path) {
            Err(TextEncoderError::MissingTensor { name }) => assert_eq!(name, dropped),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_reports_a_tensor_stored_at_the_wrong_shape() {
        let cfg = small_config(2);
        let mut entries = manifest(&cfg);
        let target = "model.language_model.layers.0.mlp.down_proj.weight";
        for (name, shape) in entries.iter_mut() {
            if name == target {
                // Same element count, transposed: the container accepts it, the
                // architecture does not.
                shape.swap(0, 1);
            }
        }
        let path = write_model(
            "load-shape",
            &entries,
            config_json(&cfg, None),
            &random_fill,
        );
        match TextEncoder::load(&path) {
            Err(TextEncoderError::ShapeMismatch {
                name,
                expected,
                found,
            }) => {
                assert_eq!(name, target);
                assert_eq!(expected, vec![8, 6]);
                assert_eq!(found, vec![6, 8]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// `num_attention_heads * head_dim` need not equal `hidden_size`.
    ///
    /// `Qwen3VLTextAttention` reads `head_dim` from the config and sizes
    /// `q_proj` as `hidden -> heads * head_dim` with `o_proj` as
    /// `heads * head_dim -> hidden`, so the two widths are decoupled;
    /// `Qwen3NextConfig` ships hidden 2048 with 16 heads of 256. `validate`
    /// used to require them equal, which is stricter than the model.
    ///
    /// Every other fixture here happens to satisfy the identity, which leaves
    /// the manifest's `[out, in]` orientation unpinned — a projection declared
    /// the other way round is invisible while its matrix is square. With the
    /// widths apart it is not.
    #[test]
    fn head_width_may_differ_from_the_hidden_size() {
        let cfg = TextEncoderConfig {
            num_layers: 1,
            hidden_size: 12,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 6,
            intermediate_size: 8,
            vocab_size: 5,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [1, 1, 1],
        };
        assert_ne!(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);
        cfg.validate()
            .expect("24 lanes of query over a hidden size of 12 is legal");

        let (enc, path) = load_random("asymmetric", &cfg);
        let out = enc.forward(&[0, 4, 2]).expect("forward");
        assert_eq!((out.rows, out.cols), (3, cfg.hidden_size));
        assert!(out.data.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_file(&path);

        // A checkpoint storing either projection transposed is rejected, and
        // so is a manifest that expects one transposed.
        for (target, transposed) in [
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                vec![12, 24],
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                vec![24, 12],
            ),
        ] {
            let mut entries = manifest(&cfg);
            for (name, shape) in entries.iter_mut() {
                if name == target {
                    *shape = transposed.clone();
                }
            }
            let path = write_model("asym-flip", &entries, config_json(&cfg, None), &random_fill);
            match TextEncoder::load(&path) {
                Err(TextEncoderError::ShapeMismatch { name, found, .. }) => {
                    assert_eq!(name, target);
                    assert_eq!(found, transposed);
                }
                other => panic!("transposed {target} should be rejected, got {other:?}"),
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    /// `load` must refuse an absurd layer count instead of sizing a manifest
    /// from it.
    ///
    /// `u64::MAX` overflowed the manifest's capacity — a debug panic and a
    /// release `capacity overflow` — and a merely large 200000 spent 370 ms
    /// building 2.2 M strings before reporting the first missing tensor. The
    /// timing bound below has a margin of some thousands, so it is a check
    /// that the manifest is not built at all rather than a benchmark.
    #[test]
    fn load_refuses_an_absurd_layer_count_before_building_a_manifest() {
        let cfg = small_config(1);
        let entries = manifest(&cfg);
        for claimed in [u64::MAX, 200_000] {
            let mut config = config_json(&cfg, None);
            config["text_config"]["num_hidden_layers"] = serde_json::json!(claimed);
            let path = write_model("absurd-layers", &entries, config, &random_fill);
            let started = std::time::Instant::now();
            let outcome = TextEncoder::load(&path);
            let took = started.elapsed();
            let _ = std::fs::remove_file(&path);
            match outcome {
                Err(TextEncoderError::BadConfig(why)) => {
                    assert!(why.contains("num_hidden_layers"), "{why}")
                }
                other => panic!("expected BadConfig for {claimed} layers, got {other:?}"),
            }
            assert!(took.as_millis() < 100, "{claimed} layers took {took:?}");
        }
    }

    /// A stored row that does not hold exactly one value per output slot is an
    /// error, not a partial fill.
    ///
    /// Three gates upstream make this unreachable through [`TextEncoder`], but
    /// the decoder is a second copy of `LbiFile::read_f32`'s and a silent
    /// partial fill is the wrong thing for it to do if they ever diverge.
    #[test]
    fn decode_row_rejects_a_row_that_is_not_the_slot_width() {
        // Short: four slots, one f32 of data. A bare `zip` fills the first
        // slot and leaves the other three at whatever they held.
        let mut out = [9.0f32; 4];
        match decode_row("t", QuantScheme::F32, &[0u8; 4], &mut out) {
            Err(TextEncoderError::ShapeMismatch {
                expected, found, ..
            }) => assert_eq!((expected, found), (vec![4], vec![1])),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        assert_eq!(out, [9.0; 4], "a rejected row must not have been written");

        // Long: one slot, four f32 of data, three of them silently dropped.
        let mut out = [9.0f32; 1];
        assert!(matches!(
            decode_row("t", QuantScheme::F32, &[0u8; 16], &mut out),
            Err(TextEncoderError::ShapeMismatch { .. })
        ));

        // A half-width scheme is measured at its own width, not at four bytes.
        let mut out = [9.0f32; 2];
        decode_row("t", QuantScheme::Bf16, &[0u8; 4], &mut out).expect("two bf16 fill two slots");
        assert_eq!(out, [0.0; 2]);
        assert!(matches!(
            decode_row("t", QuantScheme::F16, &[0u8; 6], &mut out),
            Err(TextEncoderError::ShapeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------- forward

    #[test]
    fn sub_steps_have_the_shapes_the_next_one_needs() {
        let cfg = small_config(2);
        let (enc, path) = load_random("shapes", &cfg);
        let ids = [0u32, 3, 5];
        let positions: Vec<f32> = (0..ids.len()).map(|i| i as f32).collect();
        let (cos, sin) = rope_tables(&cfg, [&positions, &positions, &positions]).unwrap();

        let embedded = enc.embed(&ids).unwrap();
        assert_eq!((embedded.rows, embedded.cols), (3, cfg.hidden_size));
        assert_eq!((cos.rows, cos.cols), (3, cfg.head_dim));

        let attended = enc.attention(&embedded, 0, &cos, &sin).unwrap();
        assert_eq!((attended.rows, attended.cols), (3, cfg.hidden_size));

        let expanded = enc.mlp(&embedded, 0).unwrap();
        assert_eq!((expanded.rows, expanded.cols), (3, cfg.hidden_size));

        let layered = enc.decoder_layer(&embedded, 1, &cos, &sin).unwrap();
        assert_eq!((layered.rows, layered.cols), (3, cfg.hidden_size));

        let out = enc.forward(&ids).unwrap();
        assert_eq!((out.rows, out.cols), (3, cfg.hidden_size));
        assert!(out.data.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn forward_rejects_an_empty_prompt_and_an_out_of_range_token() {
        let cfg = small_config(1);
        let (enc, path) = load_random("bad-input", &cfg);
        assert!(matches!(
            enc.forward(&[]),
            Err(TextEncoderError::EmptyInput)
        ));
        match enc.forward(&[0, 6]) {
            Err(TextEncoderError::TokenOutOfRange { id, vocab }) => {
                assert_eq!((id, vocab), (6, 6));
            }
            other => panic!("expected TokenOutOfRange, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A later token must not change any earlier row, and must change its own.
    #[test]
    fn attention_is_causal() {
        let cfg = small_config(2);
        let (enc, path) = load_random("causal", &cfg);
        let base = enc.forward(&[1, 2, 3, 4, 0]).unwrap();

        let tail_changed = enc.forward(&[1, 2, 3, 4, 5]).unwrap();
        for r in 0..4 {
            assert_eq!(base.row(r), tail_changed.row(r), "row {r} moved");
        }
        assert_ne!(base.row(4), tail_changed.row(4), "the changed row is stale");

        let middle_changed = enc.forward(&[1, 2, 0, 4, 0]).unwrap();
        for r in 0..2 {
            assert_eq!(base.row(r), middle_changed.row(r), "row {r} moved");
        }
        assert_ne!(base.row(2), middle_changed.row(2));
        assert_ne!(base.row(4), middle_changed.row(4));
        let _ = std::fs::remove_file(&path);
    }

    /// Token id `n` must read table row `n`.
    #[test]
    fn embed_gathers_the_row_the_token_id_names() {
        let cfg = small_config(1);
        let hidden = cfg.hidden_size;
        let fill = |name: &str, n: usize| -> Vec<f32> {
            if name == "model.language_model.embed_tokens.weight" {
                // Row r lane c holds `r * 10 + c`, so both the row and the
                // lane are identifiable and an off-by-one row is visible.
                (0..n)
                    .map(|i| ((i / hidden) * 10 + i % hidden) as f32)
                    .collect()
            } else {
                vec![0f32; n]
            }
        };
        let path = write_model("embed", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        // The last id is in the list on purpose: row `id + 1` is off the table.
        let ids = [2u32, 0, 5];
        let got = enc.embed(&ids).unwrap();
        assert_eq!((got.rows, got.cols), (3, hidden));
        for (r, &id) in ids.iter().enumerate() {
            let want: Vec<f32> = (0..hidden).map(|c| (id as usize * 10 + c) as f32).collect();
            assert_eq!(got.row(r), want.as_slice(), "row {r} did not read row {id}");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// One head over two tokens, computed by hand.
    ///
    /// [`query_heads_read_the_interleaved_key_value_head`] isolates the GQA
    /// mapping by neutralising everything around it, which leaves five
    /// separate properties unobservable: that `q_norm` is applied at all, that
    /// `q_norm` and `k_norm` are not interchanged, that the rotary comes
    /// *after* the per-head norm, that the norm uses its weight and the
    /// configured epsilon, and that a weight is read `[out, in]`. Nothing else
    /// pins the `head_dim^-0.5` factor either. This fixture observes all of
    /// them at once: both per-head norms are distinct and away from one, every
    /// projection is asymmetric, and `rms_norm_eps` is 0.5 rather than 1e-6 so
    /// the epsilon is a term rather than a rounding detail.
    ///
    /// The expected rows were derived independently, in Python, from
    /// `Qwen3VLTextAttention.forward`, not captured from this implementation.
    #[test]
    fn attention_matches_a_hand_computed_single_head() {
        let cfg = single_head_config(0.5);
        let fill = attention_weights([0.5, 2.0], [3.0, 0.25]);
        let path = write_model("attn", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        // The two input rows are the identity, so row `s` selects column `s`
        // of each projection — a transposed read would select row `s`.
        let x = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let positions = [0.0f32, 1.0];
        let (cos, sin) = rope_tables(&cfg, [&positions, &positions, &positions]).unwrap();
        let out = enc.attention(&x, 0, &cos, &sin).unwrap();

        // Row 0 is closed form: at position 0 it attends only to itself, so the
        // softmax is 1 and the context is v_proj's first column [1, 3] whatever
        // the queries, keys, norms and scaling do. o_proj then gives
        // [1*1 + 2*3, 3*1 + 5*3]. Reading o_proj as [in, out] gives [10, 17].
        assert_eq!(out.row(0), [7.0f32, 18.0].as_slice());

        // Row 1 carries the rest:
        //   q_proj  -> [2, 3]        k_proj  -> [5, 3]
        //   rms(0.5)-> [0.3779645,   rms(0.5)-> [3.5856857,
        //               2.267787]                0.1792843]
        //   rope(1) -> [-1.704062,   rope(1) -> [1.7864916,
        //               1.5433366]               3.1141183]
        //   scores  = [-4.0165658, 1.2458092]   (x head_dim^-0.5 = 0.70710677)
        //   softmax = [0.005156256, 0.99484372]
        //   context = 0.005156256*[1, 3] + 0.99484372*[2, -1]
        //           = [1.9948437, -0.97937495]
        //   o_proj  -> [1*1.9948437 + 2*-0.97937495,
        //               3*1.9948437 + 5*-0.97937495]
        let want = [0.036_093_83f32, 1.087_656_5];
        for (lane, (&got, &want)) in out.row(1).iter().zip(&want).enumerate() {
            assert!(
                (got - want).abs() < 2e-6,
                "row 1 lane {lane} is {got}, want {want}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The softmax must subtract the row maximum before exponentiating.
    ///
    /// The per-head norm fixes each head's magnitude, so extreme logits have
    /// to come from the norm weights: at 1e3 the scores reach 1e6, where a
    /// bare `exp` is `inf` and the whole row becomes `inf / inf`.
    #[test]
    fn softmax_subtracts_the_row_maximum() {
        let cfg = single_head_config(1e-6);
        let fill = attention_weights([1e3, 1e3], [1e3, 1e3]);
        let path = write_model("softmax", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        let x = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let positions = [0.0f32, 1.0];
        let (cos, sin) = rope_tables(&cfg, [&positions, &positions, &positions]).unwrap();
        let out = enc.attention(&x, 0, &cos, &sin).unwrap();

        assert!(
            out.data.iter().all(|v| v.is_finite()),
            "the row exponentiated to infinity: {:?}",
            out.data
        );
        // Row 0 attends only to itself: context is v_proj's first column
        // [1, 3], and o_proj gives [7, 18]. Row 1's scores are 7.3e4 and
        // 1.28e6, so the softmax is exactly [0, 1], the context is the second
        // column [2, -1], and o_proj gives [1*2 + 2*-1, 3*2 + 5*-1].
        assert_eq!(out.row(0), [7.0f32, 18.0].as_slice());
        assert_eq!(out.row(1), [0.0f32, 1.0].as_slice());
        let _ = std::fs::remove_file(&path);
    }

    /// Query head `h` must read key/value head `h / groups`, which
    /// `repeat_interleave` gives, not `h % num_key_value_heads`.
    ///
    /// This fixture is deliberately degenerate so that only the mapping is
    /// observable: zeroed queries and keys make every score equal, unit norms
    /// make the per-head norms invisible, and an identity `o_proj` mixes
    /// nothing. Those properties are pinned by
    /// [`attention_matches_a_hand_computed_single_head`] instead.
    #[test]
    fn query_heads_read_the_interleaved_key_value_head() {
        let cfg = small_config(1);
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        // Zero queries and keys make every score equal, so the output of head
        // `h` is exactly its key/value head's `v`; `v` head `g` is the constant
        // `g + 1`; `o_proj` is the identity, so nothing is mixed afterwards.
        let fill = move |name: &str, n: usize| -> Vec<f32> {
            if name.ends_with("self_attn.v_proj.weight") {
                let mut w = vec![0f32; n];
                for g in 0..cfg.num_key_value_heads {
                    for d in 0..head_dim {
                        w[(g * head_dim + d) * hidden] = (g + 1) as f32;
                    }
                }
                w
            } else if name.ends_with("self_attn.o_proj.weight") {
                let mut w = vec![0f32; n];
                for i in 0..hidden {
                    w[i * hidden + i] = 1.0;
                }
                w
            } else if name.ends_with("norm.weight") {
                vec![1.0; n]
            } else {
                vec![0f32; n]
            }
        };
        let cfg = small_config(1);
        let path = write_model("gqa", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        // Two identical rows, so the causal average is the same at both.
        let mut x = Matrix::zeros(2, cfg.hidden_size);
        x.data[0] = 1.0;
        x.data[cfg.hidden_size] = 1.0;
        let positions = [0.0f32, 1.0];
        let (cos, sin) = rope_tables(&cfg, [&positions, &positions, &positions]).unwrap();
        let out = enc.attention(&x, 0, &cos, &sin).unwrap();

        let groups = cfg.num_key_value_groups();
        for row in 0..2 {
            for h in 0..cfg.num_attention_heads {
                let want = (h / groups + 1) as f32;
                let alternative = (h % cfg.num_key_value_heads + 1) as f32;
                for d in 0..head_dim {
                    let got = out.data[row * cfg.hidden_size + h * head_dim + d];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "row {row} head {h} lane {d} read {got}, want {want} \
                         (a modulo mapping would give {alternative})"
                    );
                }
            }
        }
        // The two mappings must actually disagree, or this proves nothing.
        assert_ne!(1 / groups, 1 % cfg.num_key_value_heads);
        let _ = std::fs::remove_file(&path);
    }

    /// `down_proj(silu(gate_proj(x)) * up_proj(x))` on a one-token fixture.
    #[test]
    fn mlp_matches_a_hand_computed_swiglu() {
        let cfg = TextEncoderConfig {
            num_layers: 1,
            hidden_size: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [1, 0, 0],
        };
        let fill = |name: &str, n: usize| -> Vec<f32> {
            match name {
                // [inter=2, hidden=2], row-major.
                "model.language_model.layers.0.mlp.gate_proj.weight" => vec![1.0, 0.0, 0.0, 1.0],
                "model.language_model.layers.0.mlp.up_proj.weight" => vec![2.0, 0.0, 0.0, 3.0],
                // [hidden=2, inter=2].
                "model.language_model.layers.0.mlp.down_proj.weight" => vec![1.0, 1.0, 1.0, -1.0],
                _ => vec![0f32; n],
            }
        };
        let path = write_model("swiglu", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        let x = Matrix::new(1, 2, vec![1.0, 0.0]);
        let out = enc.mlp(&x, 0).unwrap();
        // gate = [1, 0]; up = [2, 0]; silu(1) * 2 = 1.4621172, silu(0) * 0 = 0.
        let inner = silu(1.0) * 2.0;
        let want = [inner, inner];
        assert_eq!(out.cols, 2);
        for (got, want) in out.data.iter().zip(&want) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        assert!(
            (inner - 1.462_117_2).abs() < 1e-6,
            "silu path drifted: {inner}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// With both sub-blocks zeroed, a decoder layer must pass its input
    /// through untouched — which only holds if both residuals are present.
    #[test]
    fn decoder_layer_keeps_both_residuals() {
        let cfg = small_config(1);
        let fill = |name: &str, n: usize| -> Vec<f32> {
            if name.ends_with("layernorm.weight") || name.ends_with("norm.weight") {
                vec![1.0; n]
            } else {
                vec![0f32; n]
            }
        };
        let path = write_model("residual", &manifest(&cfg), config_json(&cfg, None), &fill);
        let enc = TextEncoder::load(&path).expect("load");

        let hidden = Matrix::new(2, 8, (0..16).map(|i| i as f32 * 0.25 - 2.0).collect());
        let positions = [0.0f32, 1.0];
        let (cos, sin) = rope_tables(&cfg, [&positions, &positions, &positions]).unwrap();
        let out = enc.decoder_layer(&hidden, 0, &cos, &sin).unwrap();
        assert_eq!(out.data, hidden.data);
        let _ = std::fs::remove_file(&path);
    }

    /// The pipeline neutralises `Qwen3VLTextModel.norm` with a forward hook, so
    /// `forward` must hand back the last decoder layer's output untouched.
    ///
    /// Every block is zeroed here, which makes the tower the identity on the
    /// embedding; applying the loaded `norm.weight` would visibly change it.
    #[test]
    fn forward_returns_the_pre_norm_hidden_state() {
        let cfg = small_config(1);
        let fill = |name: &str, n: usize| -> Vec<f32> {
            if name == "model.language_model.embed_tokens.weight" {
                // Rows far from unit RMS, so the final norm could not be a
                // no-op if it were applied.
                (0..n).map(|i| (i % 5) as f32 + 0.5).collect()
            } else if name == "model.language_model.norm.weight" {
                vec![7.0; n]
            } else if name.ends_with("norm.weight") {
                vec![1.0; n]
            } else {
                vec![0f32; n]
            }
        };
        let path = write_model(
            "final-norm",
            &manifest(&cfg),
            config_json(&cfg, None),
            &fill,
        );
        let enc = TextEncoder::load(&path).expect("load");

        let ids = [1u32, 4];
        let embedded = enc.embed(&ids).unwrap();
        assert_eq!(enc.forward(&ids).unwrap().data, embedded.data);

        let scale = vec![7.0f32; cfg.hidden_size];
        let normed = rms_norm_rows(&embedded, &scale, cfg.rms_norm_eps);
        assert_ne!(
            normed.data, embedded.data,
            "the fixture cannot tell the two apart"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `forward` is `embed` followed by every layer over `arange(seq)`.
    ///
    /// Note what this can and cannot pin. Rotating the query at `i` and the
    /// key at `j` by the same schedule leaves the score a function of `j - i`,
    /// so *adding a constant to every position changes nothing*: measured on
    /// this fixture, a schedule of `1..=seq` moves the output by 4.8e-7, one
    /// ulp of rounding, while moving a single position by four moves it by
    /// 7.4e-2. A test cannot honestly claim to catch a uniform shift, and this
    /// one does not try. What it pins is the composition — the schedule is
    /// `arange`, the layers run in order, and nothing else is applied — plus,
    /// below, that the fixture responds to the schedule at all, so the
    /// equality above is not vacuous. The origin itself is pinned by
    /// [`rope_position_zero_is_the_identity_rotation`].
    #[test]
    fn forward_numbers_positions_from_zero() {
        // Four rotary slots per head and four query heads over two key/value
        // heads, so the schedule meets a real inverse-frequency ladder rather
        // than the single slot a `head_dim` of 2 would give.
        let cfg = TextEncoderConfig {
            num_layers: 2,
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 16,
            vocab_size: 10,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [2, 1, 1],
        };
        let (enc, path) = load_random("positions", &cfg);
        let ids = [0u32, 3, 5, 9, 2, 7];
        let embedded = enc.embed(&ids).unwrap();

        let chain = |positions: &[f32]| {
            let (cos, sin) = rope_tables(&cfg, [positions, positions, positions]).unwrap();
            let mut hidden = embedded.clone();
            for layer in 0..cfg.num_layers {
                hidden = enc.decoder_layer(&hidden, layer, &cos, &sin).unwrap();
            }
            hidden
        };

        let arange: Vec<f32> = (0..ids.len()).map(|i| i as f32).collect();
        assert_eq!(
            enc.forward(&ids).unwrap().data,
            chain(&arange).data,
            "forward must be embed then every layer over arange(seq)"
        );

        // The chain does depend on the schedule, so the equality above is a
        // statement about the positions and not about a fixture that ignores
        // them: moving one position moves the output far past rounding.
        let mut moved = arange.clone();
        moved[5] = 9.0;
        let spread = chain(&arange)
            .data
            .iter()
            .zip(&chain(&moved).data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(spread > 1e-3, "the fixture ignores its positions: {spread}");
        let _ = std::fs::remove_file(&path);
    }

    /// The per-head norm must normalise each head slice on its own, not the
    /// whole row: two heads of very different magnitude come out the same.
    #[test]
    fn head_norm_normalises_each_head_slice() {
        // Two heads of width 2, four orders of magnitude apart. Both must come
        // out at 1; normalising the whole row instead would leave head 0 near
        // zero and head 1 near sqrt(2).
        let x = Matrix::new(1, 4, vec![0.1, 0.1, 1e3, 1e3]);
        let out = norm_per_head(x, &[1.0, 1.0], 2, 1e-6);
        for v in &out.data {
            assert!((v - 1.0).abs() < 1e-3, "head slice not normalised: {v}");
        }
    }
}
