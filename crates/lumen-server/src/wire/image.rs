//! The image generation endpoint's wire types.
//!
//! The shape follows the OpenAI images API: the request carries `prompt`,
//! `size` and `num_inference_steps`, and the response is
//! `{"created": <ts>, "data": [{"b64_json": "..."}]}`.

use serde::{Deserialize, Serialize};

/// A generation request.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    /// Optional: the server's configured image model is used when absent.
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    /// `"WxH"`, e.g. `"1024x1024"`.
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_steps")]
    pub num_inference_steps: usize,
    /// Flow-matching guidance. The model is trained without it, so 1.0 means
    /// one conditional pass per step; anything above 1 additionally requires a
    /// negative prompt, which this endpoint does not take.
    #[serde(default = "default_guidance")]
    pub true_cfg_scale: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Images per request. One is generated; any other count is refused
    /// rather than answered with fewer images than asked for.
    #[serde(default = "default_n")]
    pub n: usize,
    /// `"b64_json"` (default) or `"url"`.
    #[serde(default = "default_response_format")]
    pub response_format: String,
    /// Only `"png"` is produced.
    #[serde(default = "default_output_format")]
    pub output_format: String,
}

fn default_size() -> String {
    "1024x1024".to_string()
}
fn default_steps() -> usize {
    40
}
fn default_guidance() -> f32 {
    1.0
}
fn default_n() -> usize {
    1
}
fn default_response_format() -> String {
    "b64_json".to_string()
}
fn default_output_format() -> String {
    "png".to_string()
}

/// The largest side a request may ask for.
///
/// This is a memory bound, not an aesthetic one: the pipeline allocates a
/// `latents * channels` tensor per denoising step and a `latents * hidden`
/// activation inside the transformer, so the cost is quadratic in the side. At
/// 2048 the transformer activation is already well past a gigabyte. Without a
/// cap, one request can ask for a tensor in the hundreds of gigabytes and the
/// process dies rather than answering.
pub const MAX_SIDE: usize = 2048;

/// The most denoising steps a request may ask for.
///
/// Each step is a full forward pass, so this bounds the request's duration
/// rather than its correctness; 200 is already generous beside the 40-50 the
/// model is used at.
pub const MAX_STEPS: usize = 200;

impl ImageGenerationRequest {
    /// Parse `"WxH"` into `(width, height)`, bounded.
    pub fn dimensions(&self) -> Result<(usize, usize), String> {
        let (w, h) = self
            .size
            .split_once(['x', 'X'])
            .ok_or_else(|| format!("size {:?} is not WxH", self.size))?;
        let w: usize = w
            .trim()
            .parse()
            .map_err(|_| format!("size width {w:?} is not a number"))?;
        let h: usize = h
            .trim()
            .parse()
            .map_err(|_| format!("size height {h:?} is not a number"))?;
        if w < 32 || h < 32 {
            return Err(format!(
                "size {w}x{h} is below the 32 pixel minimum side (one latent tile)"
            ));
        }
        if w > MAX_SIDE || h > MAX_SIDE {
            return Err(format!(
                "size {w}x{h} exceeds the {MAX_SIDE} pixel maximum side"
            ));
        }
        Ok((w, h))
    }

    /// Validate the step count.
    pub fn check_steps(&self) -> Result<(), String> {
        if self.num_inference_steps == 0 {
            return Err("num_inference_steps must be at least 1".to_string());
        }
        if self.num_inference_steps > MAX_STEPS {
            return Err(format!(
                "num_inference_steps {} exceeds the {MAX_STEPS} maximum",
                self.num_inference_steps
            ));
        }
        Ok(())
    }

    /// Whether the request asks for the one image a generation produces.
    pub fn check_n(&self) -> Result<(), String> {
        if self.n != 1 {
            return Err(format!(
                "n is {}: one image is generated per request",
                self.n
            ));
        }
        Ok(())
    }

    /// Whether the guidance scale is a usable number.
    pub fn check_guidance(&self) -> Result<(), String> {
        if self.true_cfg_scale != 1.0 {
            return Err(
                "true_cfg_scale must be 1.0: the endpoint runs one conditional pass per step"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// One generated image.
#[derive(Debug, Clone, Serialize)]
pub struct ImageDatum {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// The response body.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageDatum>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_reference_endpoint() {
        let r: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt": "a cat"}"#).expect("parse");
        assert_eq!(r.size, "1024x1024");
        assert_eq!(r.num_inference_steps, 40);
        assert_eq!(r.true_cfg_scale, 1.0);
        assert_eq!(r.response_format, "b64_json");
        assert_eq!(r.output_format, "png");
        assert_eq!(r.dimensions().unwrap(), (1024, 1024));
    }

    /// Every bound is a memory or time bound, so each must actually fire.
    #[test]
    fn bounds_reject_what_would_exhaust_the_machine() {
        let mk = |size: &str, steps: usize| -> ImageGenerationRequest {
            serde_json::from_str(&format!(
                r#"{{"prompt":"x","size":"{size}","num_inference_steps":{steps}}}"#
            ))
            .unwrap()
        };
        assert!(
            mk("100000x100000", 8).dimensions().is_err(),
            "no upper bound"
        );
        assert!(
            mk("2049x2048", 8).dimensions().is_err(),
            "just past the cap"
        );
        assert!(mk("2048x2048", 8).dimensions().is_ok(), "at the cap");
        assert!(
            mk("31x1024", 8).dimensions().is_err(),
            "below one latent tile"
        );
        assert!(mk("32x32", 8).dimensions().is_ok(), "one latent tile");
        assert!(mk("512x512", 0).check_steps().is_err(), "zero steps");
        assert!(
            mk("512x512", 201).check_steps().is_err(),
            "past the step cap"
        );
        assert!(mk("512x512", 200).check_steps().is_ok(), "at the step cap");
    }

    /// Any count but one is refused: fewer images than asked for is a wrong
    /// answer, not a partial one.
    #[test]
    fn n_must_be_one() {
        let parse = |body: &str| serde_json::from_str::<ImageGenerationRequest>(body).unwrap();
        assert!(parse(r#"{"prompt":"x"}"#).check_n().is_ok());
        assert!(parse(r#"{"prompt":"x","n":1}"#).check_n().is_ok());
        for n in [0, 2, 4] {
            let err = parse(&format!(r#"{{"prompt":"x","n":{n}}}"#))
                .check_n()
                .unwrap_err();
            assert!(err.contains(&format!("n is {n}")), "{err}");
        }
    }

    /// A non-finite guidance scale would silently compare false against every
    /// threshold, so it is rejected rather than coerced.
    #[test]
    fn guidance_must_be_exactly_one() {
        let r: ImageGenerationRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert!(r.check_guidance().is_ok());
        let mut nan = r.clone();
        nan.true_cfg_scale = f32::NAN;
        assert!(nan.check_guidance().is_err(), "NaN must not pass");
        let mut inf = r.clone();
        inf.true_cfg_scale = f32::INFINITY;
        assert!(inf.check_guidance().is_err(), "inf must not pass");
        let mut off = r;
        off.true_cfg_scale = 0.5;
        assert!(
            off.check_guidance().is_err(),
            "a value the pass ignores must not pass"
        );
    }

    #[test]
    fn size_parses_both_separators_and_rejects_junk() {
        let mut r: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt": "x", "size": "512X768"}"#).unwrap();
        assert_eq!(r.dimensions().unwrap(), (512, 768));
        r.size = "512".into();
        assert!(r.dimensions().is_err());
        r.size = "0x512".into();
        assert!(r.dimensions().is_err());
        r.size = "axb".into();
        assert!(r.dimensions().is_err());
    }

    /// Only one of `b64_json` / `url` is emitted, so a client sees the field it
    /// asked for rather than an empty companion.
    #[test]
    fn an_unused_datum_field_is_omitted() {
        let d = ImageDatum {
            b64_json: Some("abc".into()),
            url: None,
        };
        let s = serde_json::to_string(&d).unwrap();
        assert_eq!(s, r#"{"b64_json":"abc"}"#);
    }
}
