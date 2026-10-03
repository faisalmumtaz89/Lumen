//! `POST /v1/images/generations`.
//!
//! Registered only when an image model is configured, so a text-only deployment
//! never sees the route and never loads the image crate's device path.
//!
//! The handler is deliberately blocking on a worker thread: the pipeline is
//! CPU- or GPU-bound for tens of seconds and would otherwise stall the async
//! reactor that serves the text endpoints.

use axum::response::Response;

use crate::error::ServerError;

#[cfg(feature = "image")]
use {
    crate::wire::image::{ImageDatum, ImageGenerationRequest, ImageGenerationResponse},
    axum::response::IntoResponse,
    axum::Json,
};

/// The longest prompt the endpoint accepts, in bytes; it bounds a request's
/// tokenizer and text-encoder cost.
#[cfg(feature = "image")]
const MAX_PROMPT_BYTES: usize = 8 * 1024;

/// Held for the whole of a generation, taken in the async handler so a
/// request waiting its turn holds no thread and simply goes away with its
/// connection; see the handler.
#[cfg(feature = "image")]
static GENERATION: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Set when the request's handler is dropped, which is how a client that
/// disconnected becomes visible to the blocking task: a started blocking task
/// cannot be aborted, so it reads the flag before the pipeline and at every
/// step boundary and stops there.
#[cfg(feature = "image")]
struct CancelOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(feature = "image")]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Set once for the process by [`request_shutdown`]; a running generation
/// stops at its next step boundary instead of holding the shutdown for the
/// rest of its steps.
#[cfg(feature = "image")]
static SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Stop every running and waiting generation: the server is shutting down.
#[cfg(feature = "image")]
pub fn request_shutdown() {
    SHUTDOWN.store(true, std::sync::atomic::Ordering::Release);
}

#[cfg(feature = "image")]
fn shutting_down() -> bool {
    SHUTDOWN.load(std::sync::atomic::Ordering::Acquire)
}

/// Where a generation runs and how long it may take.
#[derive(Debug, Clone)]
pub struct ImageConfig {
    /// Directory holding `transformer.lbi`, `vae.lbi`, `text_encoder.lbi`.
    pub lbi_dir: std::path::PathBuf,
    /// The checkpoint directory, for the tokenizer files.
    pub checkpoint_dir: std::path::PathBuf,
    /// The model id this endpoint reports and accepts.
    pub model_id: String,
    /// Run on the GPU. `false` uses the CPU reference, which is correct but
    /// takes minutes per image.
    pub use_gpu: bool,
    /// Keep a page-locked host copy of the text encoder (CUDA only), from
    /// which each generation loads it faster.
    pub pin_text_encoder: bool,
}

/// The generation endpoint's state.
pub struct ImageState {
    pub config: ImageConfig,
    /// The transformer and VAE kept resident on the device between generations;
    /// `None` on the CPU (`config.use_gpu` false), where each generation runs on
    /// the host.
    #[cfg(feature = "image")]
    pub resident: Option<std::sync::Mutex<lumen_image::pipeline::GpuResident>>,
}

/// Whether the client asked for the PNG itself rather than the JSON body: its
/// `Accept` header weighs `image/png` above `application/json`, each weighed by
/// the most specific media range that names it (RFC 9110 §12.5.1). No header,
/// or one that names neither, keeps the JSON default.
#[cfg(feature = "image")]
fn prefers_png(headers: &axum::http::HeaderMap) -> bool {
    // Weights by specificity: the exact type, `type/*`, then `*/*`.
    let mut png = [None::<f32>; 3];
    let mut json = [None::<f32>; 3];
    let note = |slot: &mut Option<f32>, q: f32| *slot = Some(slot.map_or(q, |s| s.max(q)));
    for value in headers.get_all(axum::http::header::ACCEPT) {
        let Ok(value) = value.to_str() else { continue };
        for range in value.split(',') {
            let mut parts = range.split(';');
            let media = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
            let q = parts.find_map(|param| {
                let (name, value) = param.split_once('=')?;
                name.trim().eq_ignore_ascii_case("q").then(|| value.trim())
            });
            let Some(q) = q.map_or(Some(1.0), |q| q.parse::<f32>().ok()) else {
                continue;
            };
            match media.as_str() {
                "image/png" => note(&mut png[0], q),
                "image/*" => note(&mut png[1], q),
                "application/json" => note(&mut json[0], q),
                "application/*" => note(&mut json[1], q),
                "*/*" => {
                    note(&mut png[2], q);
                    note(&mut json[2], q);
                }
                _ => {}
            }
        }
    }
    let weight = |w: [Option<f32>; 3]| w.into_iter().flatten().next().unwrap_or(0.0);
    weight(png) > weight(json)
}

/// Base64, so the response carries a PNG without a separate file store.
#[cfg(feature = "image")]
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `GET /v1/models` on an image-only server: reports the single image model it
/// serves, in the same shape as the text server's listing, so a client — and the
/// deployment's guard — probes readiness and discovers the model uniformly across
/// both server kinds. Mounted only when no text engine is present (see
/// [`crate::router::build_router_with_images`]); otherwise the text listing is served.
#[cfg(feature = "image")]
pub async fn list_image_models(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<ImageState>>,
) -> impl axum::response::IntoResponse {
    axum::Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.config.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "lumen",
        }]
    }))
}

#[cfg(feature = "image")]
pub async fn generate_image(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<ImageState>>,
    headers: axum::http::HeaderMap,
    crate::router::OpenAiJson(req): crate::router::OpenAiJson<ImageGenerationRequest>,
) -> Result<Response, ServerError> {
    let png_body = prefers_png(&headers);
    if let Some(model) = &req.model {
        if model != &state.config.model_id {
            return Err(ServerError::bad_request_field(
                format!(
                    "unknown model {model:?}; this server serves {:?}",
                    state.config.model_id
                ),
                "model",
                "unknown_model",
            ));
        }
    }
    if req.output_format != "png" {
        return Err(ServerError::bad_request_field(
            format!(
                "output_format {:?} is not supported; only \"png\"",
                req.output_format
            ),
            "output_format",
            "unsupported_value",
        ));
    }
    if !matches!(req.response_format.as_str(), "b64_json" | "url") {
        return Err(ServerError::bad_request_field(
            format!(
                "response_format {:?} is not supported; use \"b64_json\" or \"url\"",
                req.response_format
            ),
            "response_format",
            "unsupported_value",
        ));
    }
    req.check_steps()
        .map_err(|m| ServerError::bad_request_field(m, "num_inference_steps", "invalid_value"))?;
    req.check_guidance()
        .map_err(|m| ServerError::bad_request_field(m, "true_cfg_scale", "invalid_value"))?;
    req.check_n()
        .map_err(|m| ServerError::bad_request_field(m, "n", "invalid_value"))?;
    let (width, height) = req
        .dimensions()
        .map_err(|m| ServerError::bad_request_field(m, "size", "invalid_value"))?;
    // Tokenizing and encoding grow with the prompt, so an unbounded prompt is
    // a cheap way to pin a worker; the text endpoints already bound theirs.
    if req.prompt.len() > MAX_PROMPT_BYTES {
        return Err(ServerError::bad_request_field(
            format!(
                "prompt is {} bytes, above the {MAX_PROMPT_BYTES} maximum",
                req.prompt.len()
            ),
            "prompt",
            "invalid_value",
        ));
    }

    let cfg = state.config.clone();
    let resident_state = std::sync::Arc::clone(&state);
    let prompt = req.prompt.clone();
    let steps = req.num_inference_steps;
    let seed = req.seed.unwrap_or(42);
    let out_format = req.response_format.clone();

    // One generation at a time: a generation fills the device (or, on the CPU
    // path, the host) on its own, so a second running alongside would fail both.
    let one_at_a_time = GENERATION.lock().await;
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(std::sync::Arc::clone(&cancelled));

    // The pipeline blocks for tens of seconds; keep it off the async reactor.
    let image = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ServerError> {
        let _one_at_a_time = one_at_a_time;
        let paths =
            lumen_image::pipeline::PipelinePaths::from_roots(&cfg.lbi_dir, &cfg.checkpoint_dir);
        let gen_req = lumen_image::pipeline::GenerationRequest {
            prompt: &prompt,
            height,
            width,
            steps,
            seed,
            init_latents: None,
        };
        // A client that left, or a shutdown that began, has no reader for the
        // image, so the generation is stopped: checked before it starts, so work
        // that is already unwanted never begins, and at every step of a running
        // generation, which stops there. The response carries only the finished
        // image, so the step counts have no reader.
        let stop = || cancelled.load(std::sync::atomic::Ordering::Acquire) || shutting_down();
        let stopped = || {
            if shutting_down() {
                ServerError::EngineUnavailable("the server is shutting down".to_string())
            } else {
                ServerError::Internal("the client disconnected".to_string())
            }
        };
        if stop() {
            return Err(stopped());
        }
        let mut progress = |_, _| {
            if stop() {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        };
        let failed = |e: lumen_image::pipeline::PipelineError| match e {
            lumen_image::pipeline::PipelineError::Cancelled => stopped(),
            e => ServerError::Runtime(format!("generation failed: {e}")),
        };
        let rgba = if let Some(resident) = &resident_state.resident {
            // One generation at a time already: `GENERATION` is held.
            // A panic inside a generation leaves the pipeline usable: a
            // transformer it held is reloaded by the next one.
            resident
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .generate(&gen_req, &mut progress)
                .map_err(failed)?
        } else {
            lumen_image::pipeline::generate_cpu(&paths, &gen_req, &mut progress).map_err(failed)?
        };
        Ok(lumen_image::png::encode(&rgba))
    })
    .await
    .map_err(|e| ServerError::Internal(format!("generation task failed: {e}")))??;

    if png_body {
        return Ok(([(axum::http::header::CONTENT_TYPE, "image/png")], image).into_response());
    }
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let datum = if out_format == "url" {
        // No file store is configured, so a url response carries the same bytes
        // as a data URL rather than a link that would 404.
        ImageDatum {
            b64_json: None,
            url: Some(format!("data:image/png;base64,{}", base64_encode(&image))),
        }
    } else {
        ImageDatum {
            b64_json: Some(base64_encode(&image)),
            url: None,
        }
    };
    Ok(Json(ImageGenerationResponse {
        created,
        data: vec![datum],
    })
    .into_response())
}

#[cfg(not(feature = "image"))]
pub async fn generate_image() -> Result<Response, ServerError> {
    Err(ServerError::bad_request(
        "this server was built without the image feature",
    ))
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::base64_encode;

    /// The padding cases are where a hand-written encoder goes wrong, so all
    /// three lengths are pinned against the RFC 4648 vectors.
    #[test]
    fn base64_matches_the_rfc_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_handles_high_bytes() {
        assert_eq!(base64_encode(&[0xFF, 0xFF, 0xFF]), "////");
        assert_eq!(base64_encode(&[0x00, 0x00, 0x00]), "AAAA");
    }

    /// `prefers_png` over a request carrying one `Accept` line per value.
    fn png_for(accept: &[&str]) -> bool {
        let mut headers = axum::http::HeaderMap::new();
        for value in accept {
            headers.append(axum::http::header::ACCEPT, value.parse().unwrap());
        }
        super::prefers_png(&headers)
    }

    #[test]
    fn json_stays_the_default() {
        assert!(!png_for(&[]));
        assert!(!png_for(&["*/*"]));
        assert!(!png_for(&["application/json"]));
        assert!(!png_for(&["text/html"]));
        assert!(!png_for(&["image/png, application/json"]));
    }

    #[test]
    fn a_client_that_names_png_gets_png() {
        assert!(png_for(&["image/png"]));
        assert!(png_for(&["IMAGE/PNG"]));
        assert!(png_for(&["image/*"]));
        assert!(png_for(&["image/png, */*;q=0.8"]));
        assert!(png_for(&["application/json;q=0.5, image/png"]));
        assert!(png_for(&["text/html", "image/png"]));
    }

    #[test]
    fn the_most_specific_range_sets_the_weight() {
        assert!(!png_for(&["image/png;q=0, image/*"]));
        assert!(!png_for(&["image/*;q=0.2, */*"]));
        assert!(!png_for(&["image/png;q=0.4, application/json;q=0.5"]));
        assert!(png_for(&["image/png;q=0.6, application/json;q=0.5"]));
        assert!(png_for(&["image/png; Q=0.9, */*;q=0.1"]));
        assert!(png_for(&[
            "image/png ; q = 0.9 , application/json ; q = 0.8"
        ]));
    }

    #[test]
    fn a_range_with_an_unreadable_weight_is_ignored() {
        assert!(!png_for(&["image/png;q=high"]));
        assert!(png_for(&["image/png;q=high, image/*"]));
    }
}
