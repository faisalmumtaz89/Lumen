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
    /// The text engine, so a generation can take the device exclusively and
    /// release it when it finishes.
    pub engine: crate::engine::EngineHandle,
    /// The transformer and VAE kept on the device between generations, when
    /// the text engine is not on it; `None` when it is (each generation then
    /// loads its components under the lease) and on the CPU.
    #[cfg(feature = "image")]
    pub resident: Option<std::sync::Mutex<lumen_image::pipeline::GpuResident>>,
    /// The tokenizer and the containers' mappings, kept open for generations
    /// that load their components under the lease; `None` when `resident`
    /// holds them and on the CPU. With `config.use_gpu`, exactly one of the two
    /// is set: a generation with neither runs on the CPU.
    #[cfg(feature = "image")]
    pub sources: Option<lumen_image::pipeline::GpuSources>,
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

#[cfg(feature = "image")]
pub async fn generate_image(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<ImageState>>,
    crate::router::OpenAiJson(req): crate::router::OpenAiJson<ImageGenerationRequest>,
) -> Result<Response, ServerError> {
    let engine = state.engine.clone();
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

    // One generation at a time: each loads a component set that fills the
    // card (or, on the CPU path, the host) on its own, so a second one
    // running alongside would fail both. The text engine's lease serialises
    // only the generations that evict it; this covers the rest.
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
        // image. Checked before the lease, so a request that is already over
        // never evicts the text model; again after it, since the grant can
        // wait behind a running text request, so the guard's drop restores
        // the model straight away instead of after a generation nobody
        // collects; and at every step of a running generation, which stops
        // there. The response carries only the finished image, so the step
        // counts have no reader.
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
        // Take the device exclusively when the text model is on it: the model
        // is evicted for the duration and restored when the guard is dropped
        // below, before the PNG is encoded. Without it the two do not fit the card
        // together. A text engine off that device (CPU, another card) needs
        // no eviction.
        let lease = if cfg.use_gpu && engine.holds_device(lumen_image::pipeline::GPU_DEVICE) {
            Some(engine.try_exclusive()?)
        } else {
            None
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
        } else if let Some(sources) = &resident_state.sources {
            lumen_image::pipeline::generate_gpu(sources, &gen_req, &mut progress).map_err(failed)?
        } else {
            lumen_image::pipeline::generate_cpu(&paths, &gen_req, &mut progress).map_err(failed)?
        };
        drop(lease);
        Ok(lumen_image::png::encode(&rgba))
    })
    .await
    .map_err(|e| ServerError::Internal(format!("generation task failed: {e}")))??;

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
}
