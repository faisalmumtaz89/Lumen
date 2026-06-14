use crate::help::print_run_usage;

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::compute::cpu_naive::NaiveF32Backend;
use lumen_runtime::compute::cpu_simd::SimdF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::metal::MetalF32Backend;
#[cfg(target_os = "macos")]
use lumen_runtime::AccelerateBatchBackend;
#[cfg(feature = "cuda")]
use lumen_runtime::CudaBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::engine::{GenerationResult, InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::kv::disk::{ModelFingerprint, serialize_hyperparams_le};
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::pipeline::PipelineMode;
use lumen_runtime::session::Session;
use lumen_runtime::storage::MmapConfig;
use lumen_runtime::weight::provider_async::AsyncWeightProvider;
use lumen_runtime::weight::provider_mmap::MmapWeightProvider;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use lumen_runtime::WeightProvider;
use lumen_format::quantization::QuantScheme;

use std::path::Path;

/// Set to `true` when the operator passes `--repeat-penalty`/`--repetition-penalty`
/// on the CLI. When `false` (operator did not override), `effective_sampling`
/// substitutes the model-aware `repetition_penalty` default once the LBC has
/// been opened and `runtime_defaults::set_model_is_moe` recorded the MoE flag
/// (MoE → 1.03, dense → 1.05). Process-global to avoid threading a bool through
/// every `run_with_*` signature; mirrors the `runtime_defaults` set-once-in-main
/// atomic pattern and is written exactly once during arg parsing.
static REPETITION_PENALTY_EXPLICIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// F4: caller-supplied textual stop sequences (`--stop`), recorded once during
/// arg parsing. Process-global (same set-once rationale as
/// `REPETITION_PENALTY_EXPLICIT`) so `print_generated_text` can truncate the
/// answer at the first matched stop string WITHOUT threading a `Vec<String>`
/// through every `run_with_*` / `run_engine` signature. Empty / unset → the
/// answer prints verbatim (byte-identical to pre-F4).
static STOP_SEQUENCES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Returns the sampling params to actually run with. Identical to `base` except
/// that, when the operator did NOT pass `--repeat-penalty` explicitly, the
/// `repetition_penalty` is resolved model-aware via
/// `runtime_defaults::repetition_penalty_default()` (MoE → 1.03, all quants;
/// dense → 1.05, unchanged). The MoE value is capped at 1.03 because the GDN
/// decode parity stack now lands the math near-tie at greedy, and a penalty of
/// 1.05+ corrupts MoE arithmetic ("17 x 20 = … = 39"); see
/// `repetition_penalty_default` for the source-of-truth rationale.
///
/// Must be called AFTER `runtime_defaults::set_model_is_moe(..)` (i.e. after the
/// LBC is opened) so the MoE flag is populated. A pure-greedy override
/// (`--repeat-penalty 1.0` → `repetition_penalty = None`) is preserved because
/// it sets the explicit flag and is therefore not substituted.
fn effective_sampling(base: &SamplingParams) -> SamplingParams {
    // The greedy anti-degeneration guard is resolved from the MoE flag (+
    // LUMEN_ANTI_RESTATE override) regardless of whether the operator pinned an
    // explicit repetition penalty — it is an orthogonal, backend-agnostic veto
    // and a `--repeat-penalty 1.0` pure-greedy run still wants the doubling
    // gone.
    let anti_restate = lumen_runtime::runtime_defaults::anti_restate_default();
    if REPETITION_PENALTY_EXPLICIT.load(std::sync::atomic::Ordering::Relaxed) {
        return SamplingParams { anti_restate, ..base.clone() };
    }
    SamplingParams {
        repetition_penalty: Some(lumen_runtime::runtime_defaults::repetition_penalty_default()),
        anti_restate,
        ..base.clone()
    }
}

/// Resolves the CLI-effective `frequency_penalty`. When the operator passed
/// `--frequency-penalty` (`explicit == true`) the parsed value wins (an
/// explicit flag always overrides the env). Otherwise the shared
/// `runtime_defaults::frequency_penalty_resolved()` is consulted so the CLI
/// honours `LUMEN_FREQUENCY_PENALTY` IDENTICALLY to the server wire
/// (`diag_frequency_penalty`). The resolved value is zero-normalized to `None`
/// (CLI convention; `Some(0.0)` and `None` are byte-identical for
/// `penalties_active` / `apply_penalties`), so an unset env stays a no-op.
fn cli_resolve_frequency_penalty(parsed: Option<f32>, explicit: bool) -> Option<f32> {
    if explicit {
        return parsed;
    }
    match lumen_runtime::runtime_defaults::frequency_penalty_resolved() {
        v if v == 0.0 => None,
        v => Some(v),
    }
}

/// Resolves the CLI-effective `repeat_last_n`. When the operator passed
/// `--repeat-last-n` (`explicit == true`) the parsed value wins. Otherwise the
/// shared `runtime_defaults::repeat_last_n_resolved()` is consulted so the CLI
/// honours `LUMEN_REPEAT_LAST_N` IDENTICALLY to the server wire
/// (`diag_repeat_last_n`); an unset env yields `None` (full-history window).
fn cli_resolve_repeat_last_n(parsed: Option<usize>, explicit: bool) -> Option<usize> {
    if explicit {
        return parsed;
    }
    lumen_runtime::runtime_defaults::repeat_last_n_resolved()
}

pub(crate) fn parse_arg(args: &[String], i: usize, name: &str) -> String {
    args.get(i).unwrap_or_else(|| {
        eprintln!("Error: {name} requires a value");
        std::process::exit(1);
    }).clone()
}

pub(crate) fn parse_arg_num<T: std::str::FromStr>(args: &[String], i: usize, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let val = parse_arg(args, i, name);
    val.parse().unwrap_or_else(|e| {
        eprintln!("Error: {name} must be a valid number, got '{val}': {e}");
        std::process::exit(1);
    })
}

/// Levenshtein edit distance between two strings (standard 2-row DP).
fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }

    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0usize; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost)
                .min(prev[j + 1] + 1)
                .min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Suggest similar model names for a mistyped input using edit distance and
/// prefix matching against the registry's canonical keys and aliases.
fn suggest_models(input: &str, registry: &crate::registry::Registry) -> Vec<String> {
    let input_lower = input.to_lowercase();
    let mut candidates: Vec<(usize, String)> = Vec::new();

    // Check canonical keys.
    for entry in registry.list() {
        let key_lower = entry.key.to_lowercase();
        let dist = levenshtein(&input_lower, &key_lower);
        if dist <= 3 || key_lower.starts_with(&input_lower) || input_lower.starts_with(&key_lower) {
            let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            candidates.push((dist, format!("  {:<20} {} ({})", entry.key, entry.display_name, quants.join(", "))));
        }
    }

    // Check aliases.
    for alias in registry.alias_keys() {
        let alias_lower = alias.to_lowercase();
        let dist = levenshtein(&input_lower, &alias_lower);
        if dist <= 3 || alias_lower.starts_with(&input_lower) || input_lower.starts_with(&alias_lower) {
            // Resolve to display the canonical entry info.
            if let Some(entry) = registry.resolve(alias) {
                let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                let line = format!("  {:<20} {} ({})", alias, entry.display_name, quants.join(", "));
                // Avoid duplicates (alias might resolve to same model already added).
                if !candidates.iter().any(|(_, l)| l == &line) {
                    candidates.push((dist, line));
                }
            }
        }
    }

    // Sort by edit distance (closest first).
    candidates.sort_by_key(|(dist, _)| *dist);
    candidates.into_iter().map(|(_, line)| line).collect()
}

pub(crate) fn run_inference(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut tokens_str: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut system_str: Option<String> = None;
    // Per-invocation reasoning toggle. `None` = not passed on the CLI → defer
    // to the shared resolver (env `LUMEN_CHAT_ENABLE_THINKING`, then the
    // default `false`). `--think` sets it to `Some(true)`. Resolved AFTER arg
    // parsing via `resolve_enable_thinking`, keeping CLI/server identical.
    let mut think_flag: Option<bool> = None;
    let mut max_tokens: usize = usize::MAX; // unlimited by default, stops at EOS
    // F4: caller-supplied textual stop sequences (`--stop`), repeatable /
    // comma-list. Empty by default → no textual stop (the EOS/max-tokens
    // behaviour is byte-identical to pre-F4).
    let mut stop_text: Vec<String> = Vec::new();
    // No-temperature default sourced from the SINGLE canonical
    // `runtime_defaults::default_temperature()` (0.7) so the CLI matches both
    // wire surfaces; an explicit `--temperature` still overrides below.
    let mut temperature: f32 = lumen_runtime::runtime_defaults::default_temperature();
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    // Sampler defaults. PURE-greedy (`repeat_penalty=1.0`) is known to
    // degenerate into repetition on every quant at generation lengths
    // >= 512 on long-form prompts, so the default for `--repeat-penalty`
    // is `1.05` so out-of-the-box long-form serving stays coherent.
    // PURE-greedy comparisons still reproduce via explicit
    // `--repeat-penalty 1.0`, which the parser at line ~285 converts to
    // `None` (disable). All other penalty sampler-chain flags remain OFF
    // by default.
    //
    // server defaults aligned to match (wire/openai.rs +
    // wire/anthropic.rs both default repeat_penalty=1.05 server-internal
    // when the client omits the field, preserving the OpenAI/Anthropic
    // API surface).
    let mut top_k: Option<usize> = None;
    let mut top_p: Option<f32> = None;
    let mut min_p: Option<f32> = None;
    // `repetition_penalty` default is resolved model-aware AFTER the LBC is
    // opened (see `run_engine`): dense → 1.05, MoE → 1.03 (all quants). We start
    // from the dense default here; if the operator does NOT pass
    // `--repeat-penalty`, `run_engine` substitutes the MoE default once
    // `model_is_moe()` is known. (The GDN decode parity stack lands the MoE math
    // near-tie at greedy; the MoE value is capped at 1.03 because 1.05+ corrupts
    // MoE arithmetic to "= 39" — mirrors the server wire default.) An explicit
    // `--repeat-penalty` flips the process-global REPETITION_PENALTY_EXPLICIT
    // (mirroring the `runtime_defaults` set-once-in-main atomic pattern) so the
    // model-aware substitution is suppressed.
    let mut repetition_penalty: Option<f32> = Some(1.05);
    let mut presence_penalty: Option<f32> = None;
    let mut frequency_penalty: Option<f32> = None;
    let mut repeat_last_n: Option<usize> = None;
    // Track whether the operator passed the flag at all (an explicit
    // `--frequency-penalty 0` / `--repeat-last-n 0` both normalize to `None`,
    // so `None` alone cannot distinguish "unset" from "explicitly disabled").
    // When the flag is NOT passed, the build site below falls back to the
    // shared env resolvers (`LUMEN_FREQUENCY_PENALTY` / `LUMEN_REPEAT_LAST_N`)
    // so the CLI honours the same env as the server; an explicit flag wins.
    let mut frequency_penalty_explicit = false;
    let mut repeat_last_n_explicit = false;
    let mut use_sync = false;
    let mut use_async = false;
    let mut use_simd = false;
    let mut use_metal = false;
    let mut use_accelerate = false;
    let mut use_cuda = false;
    let mut explicitly_chose_backend = false;
    let mut cuda_device: usize = 0;
    let mut gpu_resident = true;  // Default: GPU-resident when --metal
    let mut option_a = false;
    let mut threads: usize = 0;
    let mut profile = false;
    let mut context_len: Option<usize> = None;
    let mut verbose = false;
    let mut verbose_routing = false;
    let mut routing_bias: Option<f32> = None;
    // disk-persistent KV cache flags.
    let mut kv_disk_dir: Option<String> = None;
    let mut kv_disk_space_mb: Option<u64> = None;
    // `--session-resume <path>` opts a CLI invocation into
    // restoring a previously-persisted Session from `<path>` (Metal-only,
    // default OFF — see Session::load_from_disk for the safety contract).
    let mut session_resume_path: Option<String> = None;
    // `--session-save <path>` opts the invocation into
    // persisting the final Session state at `<path>` after generation
    // completes (default OFF — explicit user opt-in).
    let mut session_save_path: Option<String> = None;
    // explicit KV cache precision. None = backend-appropriate default
    // (Metal → F16, CUDA/CPU → F32). Honor `LUMEN_KV_PRECISION` env override
    // when the CLI flag is not supplied.
    let mut kv_precision_override: Option<KvPrecision> = None;
    let mut positional_args: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --model requires a path");
                    std::process::exit(1);
                }).clone());
            }
            "--tokens" => {
                i += 1;
                tokens_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --tokens requires token IDs");
                    std::process::exit(1);
                }).clone());
            }
            "--prompt" => {
                i += 1;
                prompt_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --prompt requires text");
                    std::process::exit(1);
                }).clone());
            }
            "--system" => {
                i += 1;
                system_str = Some(args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --system requires text");
                    std::process::exit(1);
                }).clone());
            }
            "--max-tokens" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --max-tokens requires a number");
                    std::process::exit(1);
                });
                max_tokens = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --max-tokens must be a positive integer, got: {val}");
                    std::process::exit(1);
                });
            }
            "--stop" => {
                // F4 CLI parity with the server `stop` / `stop_sequences`. The
                // flag is REPEATABLE and also accepts a comma-separated list, so
                // `--stop A --stop B` and `--stop A,B` are equivalent. The
                // truncation is applied at the print layer (see
                // `print_generated_text`): the decoded answer is cut at the first
                // matched stop string, matching the server's user-visible
                // semantics. Empty entries are ignored.
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --stop requires a stop string");
                    std::process::exit(1);
                });
                for part in val.split(',') {
                    if !part.is_empty() {
                        stop_text.push(part.to_string());
                    }
                }
            }
            "--temperature" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --temperature requires a number");
                    std::process::exit(1);
                });
                temperature = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --temperature must be a number, got: {val}");
                    std::process::exit(1);
                });
            }
            "--seed" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --seed requires a number");
                    std::process::exit(1);
                });
                seed = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --seed must be an integer, got: {val}");
                    std::process::exit(1);
                });
            }
            // sampler-chain flags (top-k / top-p / min-p) + anti-degeneration:
            // anti-degeneration penalties (repeat-penalty / repeat-last-n /
            // presence / frequency). Standard short names used by most LLM
            // CLIs.
            "--top-k" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --top-k requires a number"); std::process::exit(1);
                });
                let k: usize = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --top-k must be a non-negative integer, got: {val}");
                    std::process::exit(1);
                });
                top_k = Some(k);
            }
            "--top-p" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --top-p requires a number"); std::process::exit(1);
                });
                top_p = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --top-p must be a float, got: {val}");
                    std::process::exit(1);
                }));
            }
            "--min-p" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --min-p requires a number"); std::process::exit(1);
                });
                min_p = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --min-p must be a float, got: {val}");
                    std::process::exit(1);
                }));
            }
            "--repeat-penalty" | "--repetition-penalty" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --repeat-penalty requires a number"); std::process::exit(1);
                });
                let p: f32 = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --repeat-penalty must be a float, got: {val}");
                    std::process::exit(1);
                });
                repetition_penalty = if (p - 1.0).abs() < f32::EPSILON { None } else { Some(p) };
                REPETITION_PENALTY_EXPLICIT.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "--repeat-last-n" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --repeat-last-n requires a number"); std::process::exit(1);
                });
                let n: i64 = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --repeat-last-n must be an integer, got: {val}");
                    std::process::exit(1);
                });
                // 0 or negative = disable windowing (use full history); -1 also disables.
                repeat_last_n = if n <= 0 { None } else { Some(n as usize) };
                repeat_last_n_explicit = true;
            }
            "--presence-penalty" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --presence-penalty requires a number"); std::process::exit(1);
                });
                let p: f32 = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --presence-penalty must be a float, got: {val}");
                    std::process::exit(1);
                });
                presence_penalty = if p == 0.0 { None } else { Some(p) };
            }
            "--frequency-penalty" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --frequency-penalty requires a number"); std::process::exit(1);
                });
                let p: f32 = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --frequency-penalty must be a float, got: {val}");
                    std::process::exit(1);
                });
                frequency_penalty = if p == 0.0 { None } else { Some(p) };
                frequency_penalty_explicit = true;
            }
            "--threads" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --threads requires a number");
                    std::process::exit(1);
                });
                threads = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --threads must be a non-negative integer, got: {val}");
                    std::process::exit(1);
                });
            }
            "--sync" => {
                use_sync = true;
            }
            "--async" => {
                use_async = true;
            }
            "--simd" => {
                use_simd = true;
                explicitly_chose_backend = true;
            }
            "--metal" => {
                use_metal = true;
                explicitly_chose_backend = true;
            }
            "--accelerate" => {
                use_accelerate = true;
                explicitly_chose_backend = true;
            }
            "--cuda" => {
                use_cuda = true;
                explicitly_chose_backend = true;
            }
            "--cuda-device" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --cuda-device requires a device ordinal (e.g. 0, 1)");
                    std::process::exit(1);
                });
                cuda_device = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --cuda-device must be a non-negative integer, got: {val}");
                    std::process::exit(1);
                });
                use_cuda = true; // --cuda-device implies --cuda
                explicitly_chose_backend = true;
            }
            "--gpu-resident" => {
                gpu_resident = true;
            }
            "--no-gpu-resident" | "--streaming" => {
                gpu_resident = false;
            }
            "--option-a" => {
                option_a = true;
            }
            "--think" => {
                // Opt into the Qwen3.5 reasoning trace: opens a `<think>`
                // block in the prompt; the trace is routed to stderr (the
                // answer stays on stdout) via the shared ReasoningExtractor.
                think_flag = Some(true);
            }
            "--no-think" => {
                // Explicit opt-out (overrides LUMEN_CHAT_ENABLE_THINKING).
                think_flag = Some(false);
            }
            "--profile" => {
                profile = true;
                verbose = true; // profiling implies verbose diagnostics
            }
            "--verbose-routing" => {
                verbose_routing = true;
            }
            "--routing-bias" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --routing-bias requires a float value");
                    std::process::exit(1);
                });
                routing_bias = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --routing-bias must be a number, got: {val}");
                    std::process::exit(1);
                }));
            }
            "--context-len" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --context-len requires a number");
                    std::process::exit(1);
                });
                // Parse `--context-len <N>` and reject
                // `0` (and any unsigned-overflow input) up front. The previous
                // behaviour parsed `0` successfully and let the value flow into
                // `effective_max_seq_len` → KV cache allocation, where it
                // silently hung the CUDA backend during init (no error).
                // The minimum practical context for any Lumen model is 64
                // tokens (one decode step needs prompt + at least one new
                // token + headroom).
                let parsed: usize = val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: --context-len must be a positive integer, got: {val}");
                    std::process::exit(1);
                });
                if parsed < 64 {
                    eprintln!(
                        "Error: --context-len must be >= 64 (minimum practical KV cache size), got: {parsed}"
                    );
                    std::process::exit(1);
                }
                context_len = Some(parsed);
            }
            "--kv-disk-dir" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --kv-disk-dir requires a directory path");
                    std::process::exit(1);
                });
                kv_disk_dir = Some(val.clone());
            }
            "--kv-disk-space-mb" => {
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --kv-disk-space-mb requires a positive integer");
                    std::process::exit(1);
                });
                kv_disk_space_mb = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!(
                        "Error: --kv-disk-space-mb must be a positive integer, got: {val}"
                    );
                    std::process::exit(1);
                }));
            }
            "--session-resume" => {
                // opt-in restore of a previously-persisted
                // Session. The path must point at a file produced by an
                // earlier `--session-save` (or programmatic
                // `Session::save_to_disk`) run on the SAME model. Metal-only
                // — CUDA returns an error from `Session::load_from_disk`
                // via the backend's stubbed `sync_kv_from_cpu`.
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --session-resume requires a file path");
                    std::process::exit(1);
                });
                session_resume_path = Some(val.clone());
            }
            "--session-save" => {
                // opt-in save of the final Session state to a
                // file after generation completes. Pair with `--session-resume`
                // on the next invocation to continue the conversation.
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --session-save requires a file path");
                    std::process::exit(1);
                });
                session_save_path = Some(val.clone());
            }
            "--kv-precision" => {
                // explicit KV cache precision.
                // Metal backend stores KV in F16 (validated by
                // backend.validate_kv_precision); CUDA backend stores KV in
                // F32. Passing the wrong precision now produces a clean error
                // instead of silent corruption.
                i += 1;
                let val = args.get(i).unwrap_or_else(|| {
                    eprintln!("Error: --kv-precision requires a value (f16 | f32)");
                    std::process::exit(1);
                });
                kv_precision_override = Some(parse_kv_precision(val).unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }));
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--help" | "-h" => {
                print_run_usage();
                return;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option: {other}");
                    print_run_usage();
                    std::process::exit(1);
                }
                // Positional argument -- collect it.
                positional_args.push(other.to_string());
            }
        }
        i += 1;
    }

    // Resolve positional args: [MODEL] [PROMPT...]
    // Flags take precedence; conflicts between flags and positionals are errors.
    if !positional_args.is_empty() {
        if model_path.is_none() {
            model_path = Some(positional_args.remove(0));
        } else if !positional_args.is_empty() {
            // --model was set via flag. Check if first positional looks like a model name.
            let first = &positional_args[0];
            let reg = crate::registry::load_registry();
            if reg.resolve(first).is_some() {
                eprintln!("Error: model specified twice — --model {} and '{}'", model_path.as_ref().unwrap(), first);
                eprintln!("Use one or the other, not both.");
                std::process::exit(1);
            }
        }
        // Remaining positionals become the prompt.
        if !positional_args.is_empty() {
            if prompt_str.is_some() {
                eprintln!("Error: --prompt flag and positional prompt are mutually exclusive.");
                eprintln!("Use either:  lumen run <model> \"prompt\"");
                eprintln!("         or: lumen run --model <model> --prompt \"prompt\"");
                std::process::exit(1);
            }
            prompt_str = Some(positional_args.join(" "));
        }
    }

    // Determine the model string for error messages before resolution.
    let model_str = model_path.clone().unwrap_or_default();

    let model_path = match model_path {
        Some(p) => p,
        None => {
            // If --prompt is provided without a model, use the default from registry.
            if prompt_str.is_some() {
                let registry = crate::registry::load_registry();
                let default = registry.default_model();
                eprintln!("No model specified. Using default: {} ({})",
                    default.display_name, default.key);
                default.key.clone()
            } else {
                eprintln!("Model name is required.\n");
                let registry = crate::registry::load_registry();
                eprintln!("Available models:");
                for entry in registry.list() {
                    let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                    quants.sort();
                    let tags: Vec<String> = quants.iter().map(|q| format!("{}:{}", entry.key, q.to_lowercase())).collect();
                    eprintln!("  {}", tags.join(", "));
                }
                eprintln!("\nUsage: lumen run <model>:<quant> \"your prompt\"");
                eprintln!("Example: lumen run qwen3.5-9b:q8_0 \"What is 2+2?\"");
                std::process::exit(1);
            }
        }
    };

    // Mutual exclusivity: --tokens and --prompt cannot both be provided.
    if tokens_str.is_some() && prompt_str.is_some() {
        eprintln!("Error: --prompt and --tokens are mutually exclusive");
        std::process::exit(1);
    }

    // Resolve the reasoning toggle ONCE via the single shared resolver, so the
    // CLI honours `--think` / `--no-think` (explicit) then
    // `LUMEN_CHAT_ENABLE_THINKING` (env) then the default — identically to the
    // server. Used both for the chat-template `<think>` tail and to drive the
    // shared ReasoningExtractor in `print_generated_text`.
    let enable_thinking =
        lumen_runtime::runtime_defaults::resolve_enable_thinking(think_flag);

    // F4: record the parsed `--stop` list process-globally so the answer printer
    // can truncate at the first match. Set exactly once; ignored if already set.
    let _ = STOP_SEQUENCES.set(stop_text);

    // Validate `--cuda-device <N>` upfront against
    // `cudaGetDeviceCount()` before any model loading or CUDA backend
    // initialisation. Passing `--cuda-device 99` (or any
    // out-of-range ordinal) used to hang silently for >=30 s inside
    // `CudaContext::new` with no stderr output. Validating up front means the
    // user sees a clear error in <1 s without spinning up the rest of the
    // pipeline. This check is a no-op when the CUDA feature is not compiled
    // in (the CLI never reaches the CUDA backend in that build).
    #[cfg(feature = "cuda")]
    {
        if use_cuda {
            match lumen_runtime::cuda::ffi::device_count() {
                Ok(count) => {
                    if count == 0 {
                        eprintln!(
                            "Error: --cuda-device {cuda_device} requested but no CUDA devices are available (is the NVIDIA driver installed?)"
                        );
                        std::process::exit(1);
                    }
                    if cuda_device >= count {
                        eprintln!(
                            "Error: --cuda-device {cuda_device} out of range; CUDA reports {count} device(s) available (valid ordinals: 0..{})",
                            count - 1
                        );
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error: failed to query CUDA device count: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    // On a non-CUDA build, suppress the now-possibly-unused `cuda_device`
    // binding when the user passed `--cuda-device` against a CPU-only build.
    #[cfg(not(feature = "cuda"))]
    {
        let _ = cuda_device;
    }

    // disk-persistent KV cache directory housekeeping.
    //
    // Two responsibilities at startup, both idempotent:
    //   1. Sweep `.tmp.<pid>` stragglers left behind by killed save_atomic
    //      writers (this run inherits an empty PID namespace from the user's
    //      shell, so any `.tmp.<digits>` files are orphans of crashed
    //      writers).
    //   2. If a byte budget was provided, evict the lowest-scoring entries
    //      to fit. No live-session set is passed because a one-shot `lumen
    //      run` does not hold a persistent in-memory cache that needs the
    //      0.25x penalty; the server uses the `evict_to_budget` call from
    //      its own engine loop.
    if let Some(ref dir) = kv_disk_dir {
        let path = std::path::PathBuf::from(dir);
        match lumen_runtime::kv::disk::purge_stale_tmp(&path) {
            Ok(n) if n > 0 && verbose => {
                eprintln!("[kv-disk] purged {n} stale .tmp.<pid> file(s) from {dir}");
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[kv-disk] purge_stale_tmp({dir}) failed: {e}");
            }
        }
        if let Some(mb) = kv_disk_space_mb {
            let budget = mb.saturating_mul(1024 * 1024);
            match lumen_runtime::kv::disk::evict_to_budget(
                &path,
                budget,
                &std::collections::HashSet::new(),
            ) {
                Ok(removed) if !removed.is_empty() && verbose => {
                    eprintln!(
                        "[kv-disk] eviction removed {} file(s) to fit budget {} MB",
                        removed.len(),
                        mb
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[kv-disk] evict_to_budget({dir}, {mb} MB) failed: {e}");
                }
            }
        }
    }
    // Suppress unused-variable warnings for the path where the flag is set
    // but no work followed (e.g., `--kv-disk-space-mb` with no dir).
    let _ = &kv_disk_dir;
    let _ = &kv_disk_space_mb;

    // `--session-resume` / `--session-save` are wired into the
    // `lumen run` engine paths. The flags are validated here (paths
    // exist for resume; parent dirs exist for save) and the actual
    // load/save calls happen inside the `run_with_*` engine builders.
    //
    // Auto-on suffix cache UX:
    //   * If `--session-resume <P>` is set and `--session-save` is NOT,
    //     implicitly set `--session-save = <P>` so the same path round-trips
    //     between invocations (first run warm-saves, second hits the cache).
    //   * If `--session-resume <P>` is set but the file does not yet exist,
    //     treat it as a fresh session (cold prefill, then save) instead of
    //     erroring. The previous behaviour required the operator to manually
    //     bootstrap the file with a separate `--session-save` invocation.
    if session_resume_path.is_some() && session_save_path.is_none() {
        session_save_path = session_resume_path.clone();
        if verbose {
            if let Some(ref p) = session_save_path {
                eprintln!(
                    "[session] --session-resume {p} implies --session-save {p} \
                     (suffix-prefill cache auto-on; pass --session-save to override)"
                );
            }
        }
    }
    if let Some(ref p) = session_resume_path {
        let pp = std::path::Path::new(p);
        if pp.exists() {
            if !pp.is_file() {
                eprintln!("Error: --session-resume path is not a regular file: {p}");
                std::process::exit(2);
            }
        } else {
            // File missing — bootstrap as a fresh session. The save path
            // (auto-promoted above, or explicitly set) will create the file
            // when generation completes.
            if verbose {
                eprintln!(
                    "[session] --session-resume {p} not found; starting fresh \
                     (will be created on this run's --session-save)"
                );
            }
            session_resume_path = None;
        }
    }
    if let Some(ref p) = session_save_path {
        let pp = std::path::Path::new(p);
        if let Some(parent) = pp.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                eprintln!(
                    "Error: --session-save parent directory does not exist: {}",
                    parent.display()
                );
                std::process::exit(2);
            }
        }
    }

    // Dual-mode --model: if the value looks like a file path (contains / or \,
    // or ends with .lbc/.gguf), treat it as a direct path. Otherwise, try to
    // resolve it as a preset name from the model registry.
    let model_path = resolve_model_path(&model_path, verbose);

    let path = Path::new(&model_path);
    if !path.exists() {
        eprintln!("Error: model file not found: {model_path}");
        std::process::exit(1);
    }

    // Resolve prompt tokens: either from --tokens (raw IDs) or --prompt (text -> tokenizer).
    let (prompt_tokens, tokenizer) = if let Some(ref tokens) = tokens_str {
        // --tokens mode: parse integer IDs (existing behavior, unchanged).
        let ids: Vec<u32> = tokens
            .split_whitespace()
            .map(|s| {
                s.parse::<u32>().unwrap_or_else(|_| {
                    eprintln!("Error: token IDs must be non-negative integers, got: {s}");
                    std::process::exit(1);
                })
            })
            .collect();
        if ids.is_empty() {
            eprintln!("Error: --tokens must contain at least one token ID");
            std::process::exit(1);
        }
        (ids, None) // No tokenizer needed for --tokens mode
    } else if let Some(ref prompt) = prompt_str {
        if prompt.is_empty() {
            eprintln!("Error: --prompt must not be empty");
            std::process::exit(1);
        }
        // Load tokenizer from LBC file header (targeted seek, not full-file read).
        let lbc = lumen_format::reader::LbcFile::open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error parsing model file: {e}");
                std::process::exit(1);
            });
        let tok_section = lbc.tokenizer.unwrap_or_else(|| {
            eprintln!("Error: This model has no embedded tokenizer (LBC v2).");
            eprintln!("Re-convert with: lumen convert --input model.gguf --output model.lbc");
            std::process::exit(1);
        });
        // Build TokenizerData from TokenizerSection (move fields, no clone).
        let tok_data = lumen_convert::tokenizer_data::TokenizerData {
            model_type: tok_section.model_type,
            pre_tokenizer: tok_section.pre_tokenizer,
            tokens: tok_section.tokens,
            token_types: tok_section.token_types,
            scores: tok_section.scores,
            merges: tok_section.merges,
            bos_token_id: tok_section.bos_token_id,
            eos_token_id: tok_section.eos_token_id,
            pad_token_id: tok_section.pad_token_id,
            add_bos_token: tok_section.add_bos_token,
            add_eos_token: tok_section.add_eos_token,
            add_space_prefix: tok_section.add_space_prefix,
            chat_template: tok_section.chat_template,
        };
        let tokenizer = crate::tokenize::BpeTokenizer::from_tokenizer_data(&tok_data);
        let templated = tokenizer.apply_chat_template_with_system(
            prompt,
            system_str.as_deref(),
            enable_thinking,
        );
        let ids = tokenizer.encode(&templated);
        if ids.is_empty() {
            eprintln!("Error: prompt produced no tokens after tokenization");
            std::process::exit(1);
        }
        if verbose { eprintln!("Tokenized prompt: {} tokens", ids.len()); }
        // [XCHK] Echo the exact chat-templated prompt token IDs (env LUMEN_XCHK=1,
        // default OFF). Lets the cross-backend forensic diff verify Metal and CUDA
        // ingest byte-identical inputs before comparing per-op sumsq trajectories.
        if std::env::var("LUMEN_XCHK").as_deref() == Ok("1") {
            eprintln!("[XCHK] prompt_tokens count={} ids={ids:?}", ids.len());
        }
        (ids, Some(tokenizer))
    } else {
        eprintln!("No prompt provided.\n");
        eprintln!("Usage:");
        eprintln!("  lumen run {} \"What is the capital of France?\"", model_str);
        eprintln!("  lumen run --model {} --prompt \"Hello\"", model_str);
        std::process::exit(1);
    };

    // Backend auto-detection: if no backend flag was provided, pick the best
    // available backend for this platform.
    //
    // `--sync` selects the weight *I/O provider* (sync pread vs mmap), NOT the
    // compute backend. It must NOT suppress compute-backend auto-detection:
    // doing so left `--sync` (with no explicit backend) falling through to the
    // CPU `NaiveF32Backend`, which cannot run this model's GatedDeltaNet layers
    // (zero-length wq sentinel -> matmul panic). The provider choice is honoured
    // separately by the `run_with_sync` / `run_with_mmap` dispatch below, and
    // `run_with_sync` performs the GPU-resident preload that GDN requires.
    //
    // `--async` is intentionally STILL excluded here: `run_with_async` does not
    // perform the GPU-resident preload, so auto-selecting Metal for it would
    // panic in the streaming GDN batched-prefill path (`gdn_h_states` unallocated).
    // Wiring the async provider's GPU-resident path is out of scope for this fix;
    // `--async` behaviour is left exactly as it was on the prior baseline.
    if !explicitly_chose_backend && !use_async {
        #[cfg(target_os = "macos")]
        {
            use_metal = true;
            if verbose { eprintln!("Auto-detected backend: Metal (Apple Silicon GPU)"); }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Check for NVIDIA GPU presence.
            if std::path::Path::new("/dev/nvidia0").exists() {
                #[cfg(feature = "cuda")]
                {
                    use_cuda = true;
                    if verbose { eprintln!("Auto-detected backend: CUDA (NVIDIA GPU)"); }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    use_simd = true;
                    if verbose {
                        eprintln!("NVIDIA GPU detected but CUDA support not compiled in.");
                        eprintln!("Falling back to CPU (SIMD) backend.");
                        eprintln!("Rebuild with: cargo build --release --features cuda");
                    }
                }
            } else {
                use_simd = true;
                if verbose {
                    eprintln!("No GPU detected. Using CPU (SIMD) backend.");
                    eprintln!("For faster inference:");
                    eprintln!("  macOS:  rebuild and use --metal");
                    eprintln!("  NVIDIA: rebuild with --features cuda");
                    eprintln!("  Modal:  modal run examples/cuda_inference.py");
                }
            }
        }
    }

    // When the operator did NOT pass the flag, honour the shared env resolvers
    // so the CLI matches the server wire (`diag_frequency_penalty` /
    // `diag_repeat_last_n`, which delegate to the same resolvers). An explicit
    // flag wins (the `_explicit` guard). Centralised in pure helpers so the
    // precedence is unit-testable and identical for both fields.
    let frequency_penalty =
        cli_resolve_frequency_penalty(frequency_penalty, frequency_penalty_explicit);
    let repeat_last_n = cli_resolve_repeat_last_n(repeat_last_n, repeat_last_n_explicit);

    let sampling = SamplingParams {
        temperature,
        seed: Some(seed),
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        repeat_last_n,
        // Resolved later in `effective_sampling` from the MoE flag +
        // LUMEN_ANTI_RESTATE; this base literal just satisfies the field.
        anti_restate: false,
    };
    let stop = if let Some(ref tok) = tokenizer {
        let eos_ids = tok.stop_token_ids.clone();
        if eos_ids.is_empty() {
            StopCondition::MaxTokens(max_tokens)
        } else {
            StopCondition::MaxTokensOrEos { max_tokens, eos_tokens: eos_ids }
        }
    } else {
        StopCondition::MaxTokens(max_tokens)
    };

    // GPU-resident mode for Metal and CUDA backends.
    if !use_metal && !use_cuda {
        gpu_resident = false;
    }

    // Determine backend display name for the banner.
    let backend_name = if use_metal {
        "Metal"
    } else if use_cuda {
        "CUDA"
    } else if use_accelerate {
        "Accelerate"
    } else if use_simd {
        "SIMD"
    } else {
        "CPU"
    };

    // Extract short model name from file path for banner display.
    let model_display = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&model_path)
        .to_string();

    // resolve KV precision once at the dispatch site so all three
    // run_with_* paths see the same value and any --kv-precision / env / default
    // resolution logic lives in one place.
    let kv_precision = resolve_kv_precision(kv_precision_override, use_metal, use_cuda, verbose);

    // Bundle session flags into a single SessionFlags struct so
    // the run_with_* signatures stay readable.
    let session_flags = SessionFlags {
        resume_path: session_resume_path.clone(),
        save_path: session_save_path.clone(),
    };

    if use_async {
        run_with_async(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, threads, profile, verbose, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name, kv_precision, &session_flags, enable_thinking);
    } else if use_sync {
        run_with_sync(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, threads, profile, verbose, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name, kv_precision, &session_flags, enable_thinking);
    } else {
        run_with_mmap(path, use_simd, use_metal, use_cuda, cuda_device, use_accelerate, gpu_resident, option_a, threads, profile, verbose, verbose_routing, routing_bias, &prompt_tokens, &stop, &sampling, context_len, tokenizer.as_ref(), &model_display, backend_name, kv_precision, &session_flags, enable_thinking);
    }
}

/// CLI-side bundle of `--session-resume` / `--session-save` flags.
///
/// Wraps the two `Option<String>` paths so the `run_with_*` dispatchers can
/// take a single reference and forward it into the engine path. Both fields
/// default to `None` — the existing CLI default of "no session disk
/// interaction" — so adding `SessionFlags::default()` to a `run_with_*`
/// call site leaves behaviour unchanged.
#[derive(Debug, Clone, Default)]
pub(crate) struct SessionFlags {
    pub resume_path: Option<String>,
    pub save_path: Option<String>,
}

impl SessionFlags {
    /// True if either resume OR save is set — the engine path then needs
    /// to take the session-driven branch (`engine.generate_with_session`)
    /// instead of the legacy `engine.generate` direct-KV path.
    pub fn is_active(&self) -> bool {
        self.resume_path.is_some() || self.save_path.is_some()
    }
}

/// Build a `ModelFingerprint` for the live LBC file. Used by the
/// `--session-resume` / `--session-save` flow to validate that the file
/// being loaded matches the model we're about to run inference with.
///
/// Inputs:
/// - `hyperparams`: the live model's hyperparams from `provider.lbc().header`.
/// - `vocab_blob`: the tokenizer's `tokens_blob` if present (deterministic
///   bytes), else empty (the fingerprint still discriminates by hyperparams).
/// - `weight_quant_tag`: `output_proj_quant.to_u8() as u32`. Q4_0 vs Q8_0
///   produces different KV cache reductions so this MUST be part of the
///   fingerprint.
fn build_live_fingerprint(
    hyperparams: &lumen_format::ModelHyperparams,
    vocab_blob: &[u8],
    weight_quant_tag: u32,
) -> ModelFingerprint {
    let hp_bytes = serialize_hyperparams_le(hyperparams);
    ModelFingerprint::from_live_model(
        &hp_bytes,
        vocab_blob,
        weight_quant_tag,
        lumen_format::LBC_VERSION,
    )
}

/// Extract the tokenizer's `tokens` blob from the LBC file for use as the
/// `vocab_bytes` input to [`build_live_fingerprint`]. Returns empty `Vec`
/// for models without an embedded tokenizer (the fingerprint still
/// discriminates on hyperparams + quant).
fn tokenizer_vocab_blob(lbc: &lumen_format::reader::LbcFile) -> Vec<u8> {
    match &lbc.tokenizer {
        Some(tok) => {
            // The `tokens` field is a `Vec<String>` in `TokenizerSection`;
            // concatenate the token bytes with a NUL separator for a
            // deterministic byte sequence.
            let mut out = Vec::with_capacity(tok.tokens.iter().map(|t| t.len() + 1).sum());
            for t in &tok.tokens {
                out.extend_from_slice(t.as_bytes());
                out.push(0);
            }
            out
        }
        None => Vec::new(),
    }
}

/// Capture the LBC contents the [`build_live_fingerprint`] / `run_generation`
/// caller needs before passing the weight provider into the engine. This is
/// the small bundle a CLI dispatch site populates once (right after
/// `provider.open`) so the `run_with_*` functions don't have to re-import
/// the concrete provider type just to read `provider.lbc()`.
#[derive(Clone)]
pub(crate) struct LiveModel {
    pub hyperparams: lumen_format::ModelHyperparams,
    pub vocab_blob: Vec<u8>,
    pub weight_quant: QuantScheme,
}

impl LiveModel {
    pub fn from_lbc(
        lbc: &lumen_format::reader::LbcFile,
        weight_quant: QuantScheme,
    ) -> Self {
        Self {
            hyperparams: lbc.header.hyperparams,
            vocab_blob: tokenizer_vocab_blob(lbc),
            weight_quant,
        }
    }

    pub fn fingerprint(&self) -> ModelFingerprint {
        build_live_fingerprint(
            &self.hyperparams,
            &self.vocab_blob,
            self.weight_quant.to_u8() as u32,
        )
    }
}

/// Compute effective max_seq_len: user override, or right-sized for the actual
/// prompt + generation length to avoid GPU memory pressure from oversized KV caches.
///
/// Models with large context windows (e.g. 128K) allocate KV caches proportional
/// to max_seq_len. On GPU, this causes severe performance degradation due to memory
/// Parse a `--kv-precision` value into a `KvPrecision` variant.
///
/// Accepts `f16` / `f32` (case-insensitive). Other values return an error.
/// Int8 / Int4 are reserved for future quantized KV; not exposed via the CLI.
fn parse_kv_precision(value: &str) -> Result<KvPrecision, String> {
    match value.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "half" => Ok(KvPrecision::F16),
        "f32" | "fp32" | "float" => Ok(KvPrecision::F32),
        other => Err(format!(
            "--kv-precision must be one of: f16, f32 (got '{other}')"
        )),
    }
}

/// Resolve the effective KV precision in priority order: CLI flag override >
/// `LUMEN_KV_PRECISION` env var > backend-appropriate default.
///
/// Backend defaults match each backend's hardcoded storage:
/// - Metal: F16 (gpu_k_cache/gpu_v_cache are F16 only)
/// - CUDA: F32 (KvCacheGpu is F32 only)
/// - CPU (naive/SIMD): F32 (default of RuntimeConfig)
///
/// Returns `None` to mean "use the existing config default" so callers do not
/// have to know about backend-specific defaults.
fn resolve_kv_precision(
    cli_override: Option<KvPrecision>,
    use_metal: bool,
    use_cuda: bool,
    verbose: bool,
) -> KvPrecision {
    // CLI flag wins.
    if let Some(p) = cli_override {
        if verbose {
            eprintln!("[kv-precision] using {p:?} from --kv-precision");
        }
        return p;
    }
    // Env var second.
    if let Ok(env_val) = std::env::var("LUMEN_KV_PRECISION") {
        match parse_kv_precision(&env_val) {
            Ok(p) => {
                if verbose {
                    eprintln!("[kv-precision] using {p:?} from LUMEN_KV_PRECISION");
                }
                return p;
            }
            Err(e) => {
                eprintln!(
                    "[kv-precision] warning: LUMEN_KV_PRECISION='{env_val}' invalid \
                     ({e}); falling back to backend default",
                );
            }
        }
    }
    // Backend-appropriate default. Metal pins KV to F16; CUDA and CPU both
    // use F32. `use_cuda` is matched explicitly here even though it shares
    // F32 with the CPU fallback, so a future CUDA-F16 path slots in cleanly
    // (just change the CUDA arm rather than adding a new branch).
    let default = if use_metal {
        KvPrecision::F16
    } else {
        let _ = use_cuda; // reserved for future CUDA-F16 path; same default today
        KvPrecision::F32
    };
    if verbose {
        eprintln!("[kv-precision] using {default:?} (backend default)");
    }
    default
}

/// pressure / TLB thrashing (empirically measured: Llama 3.2 at context=8192 is 28%
/// slower than context=256 for pp128, because KV cache grows from 8 MB to 537 MB).
///
/// Default: min(model_max, max(prompt_len + max_gen + 256 headroom, 512)).
/// This ensures KV cache is right-sized for actual usage while leaving headroom
/// for multi-turn conversations. Use --context-len to override.
fn effective_max_seq_len(model_max: usize, user_override: Option<usize>, prompt_len: usize, max_gen: usize, verbose: bool) -> usize {
    let effective = match user_override {
        Some(n) => n.min(model_max),
        None => {
            // Right-size: actual usage + 256 headroom, minimum 512
            let needed = prompt_len.saturating_add(max_gen).saturating_add(256);
            let capped = needed.max(512).min(model_max);
            capped
        }
    };
    if verbose && effective < model_max {
        eprintln!("  Context length: {} (model supports {}, use --context-len {} to increase)",
            effective, model_max, model_max);
    }
    effective
}

/// Resolve a `--model` value to a file path.
///
/// Supports three forms:
/// - File path: `/path/to/model.lbc` or `model.gguf` (returned as-is)
/// - Preset with quant tag: `qwen3.5-9b:q4_0` (downloads specific quant)
/// - Preset without tag: `qwen3.5-9b` (errors with available quants if multiple exist,
///   or auto-selects if only one quant is available)
fn resolve_model_path(value: &str, verbose: bool) -> String {
    // Heuristic: looks like a file path if it contains path separators or
    // ends with a known model file extension.
    let looks_like_path = value.contains('/')
        || value.contains('\\')
        || value.ends_with(".lbc")
        || value.ends_with(".gguf");

    if looks_like_path {
        return value.to_owned();
    }

    // Parse model:quant tag syntax (e.g., "qwen3.5-9b:q4_0").
    let (model_name, explicit_quant) = if let Some(colon_pos) = value.rfind(':') {
        let name = &value[..colon_pos];
        let tag = &value[colon_pos + 1..];
        if tag.is_empty() {
            (value, None) // trailing colon, treat as no tag
        } else {
            (name, Some(tag.to_uppercase()))
        }
    } else {
        (value, None)
    };

    // Try resolving as a preset name.
    let reg = crate::registry::load_registry();
    let entry = match reg.resolve(model_name) {
        Some(e) => e.clone(),
        None => {
            // Not a known preset -- show helpful error with suggestions.
            let suggestions = suggest_models(model_name, &reg);
            if !suggestions.is_empty() {
                eprintln!("Error: unknown model '{}'\n", model_name);
                eprintln!("Did you mean?");
                for s in &suggestions {
                    eprintln!("{s}");
                }
            } else {
                eprintln!("Error: unknown model '{}'\n", model_name);
                eprintln!("Available models:");
                for entry in reg.list() {
                    let quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
                    eprintln!("  {:<20} {} ({})", entry.key, entry.display_name, quants.join(", "));
                }
            }
            eprintln!("\nRun 'lumen models' to see all available models.");
            std::process::exit(1);
        }
    };

    // Determine quantization.
    let quant = if let Some(ref q) = explicit_quant {
        // User specified an explicit quant tag. Two paths:
        //   1. Registry-known quant -> trust it, may download if uncached.
        //   2. Registry-unknown quant BUT a cached LBC exists at the
        //      canonical path (`<cache>/<key>-<QUANT>.lbc`) -> trust the
        //      cache. Common case: locally-converted Q4_0 / BF16 that
        //      isn't published as a downloadable GGUF.
        // Only reject if NEITHER registry NOR cache knows the quant.
        let in_registry = entry.gguf_files.contains_key(q.as_str());
        let in_cache = crate::cache::cached_lbc(&entry.key, q.as_str()).is_some();
        if !in_registry && !in_cache {
            let available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            eprintln!("Error: quantization '{}' not available for {}\n", q, entry.display_name);
            eprintln!("Available (downloadable):");
            for a in &available {
                eprintln!("  {}:{}", model_name, a.to_lowercase());
            }
            eprintln!();
            eprintln!("Tip: if you have a locally-converted LBC, pass --model <path.lbc> directly,");
            eprintln!("     or place it at: {}", crate::cache::lbc_path(&entry.key, q.as_str()).display());
            std::process::exit(1);
        }
        q.as_str().to_owned()
    } else if entry.gguf_files.len() == 1 {
        // Only one quant available — auto-select it.
        entry.gguf_files.keys().next().unwrap().clone()
    } else {
        // Multiple quants — require explicit choice.
        eprintln!("Multiple quantizations available for {}:\n", entry.display_name);
        let mut quants: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
        quants.sort();
        for q in &quants {
            eprintln!("  {}:{}", model_name, q.to_lowercase());
        }
        eprintln!("\nSpecify one: lumen run {}:<quant> \"your prompt\"", model_name);
        std::process::exit(1);
    };
    let quant = quant.as_str();

    // Check cache first.
    if let Some(cached) = crate::cache::cached_lbc(&entry.key, quant) {
        if verbose { eprintln!("Using cached model: {}", cached.display()); }
        return cached.to_string_lossy().into_owned();
    }

    // Not cached -- attempt download + convert.
    let gguf_source = match entry.gguf_files.get(quant) {
        Some(src) => src.clone(),
        None => {
            let available: Vec<&str> = entry.gguf_files.keys().map(|s| s.as_str()).collect();
            eprintln!("Error: no {quant} GGUF available for {}", entry.display_name);
            eprintln!("Available quantizations: {}", available.join(", "));
            std::process::exit(1);
        }
    };

    #[cfg(feature = "download")]
    {
        if verbose { eprintln!("Model '{}' not cached. Downloading {}...", value, entry.display_name); }
        let gguf_path = resolve_download_gguf_shards(&gguf_source, verbose);
        let lbc_out = crate::cache::lbc_path(&entry.key, quant);
        resolve_convert_to_lbc(&gguf_path, &lbc_out, verbose);
        if verbose { eprintln!("Ready: {}", lbc_out.display()); }
        return lbc_out.to_string_lossy().into_owned();
    }

    #[cfg(not(feature = "download"))]
    {
        let _ = gguf_source; // suppress unused warning
        eprintln!("Error: model '{}' is not cached.", value);
        eprintln!("Download it first with: lumen pull {}", value);
        eprintln!("Or pass a direct file path: --model /path/to/model.lbc");
        std::process::exit(1);
    }
}

/// Fetch every shard for a GgufSource, returning the path to the primary
/// (first) shard. The converter is pointed at the primary shard; multi-shard
/// auto-discovery resolves the siblings.
#[cfg(feature = "download")]
fn resolve_download_gguf_shards(src: &crate::registry::GgufSource, verbose: bool) -> std::path::PathBuf {
    if src.is_multi_shard() && verbose {
        eprintln!(
            "Multi-shard model: {} shard(s) to fetch from {}",
            src.files.len(),
            src.repo
        );
    }
    let mut primary: Option<std::path::PathBuf> = None;
    for (idx, filename) in src.files.iter().enumerate() {
        let path = resolve_download_gguf(&src.repo, filename, verbose);
        if idx == 0 {
            primary = Some(path);
        }
    }
    primary.unwrap_or_else(|| {
        eprintln!("Internal error: GgufSource had no shard files");
        std::process::exit(1);
    })
}

#[cfg(feature = "download")]
fn resolve_download_gguf(repo: &str, filename: &str, verbose: bool) -> std::path::PathBuf {
    if let Some(existing) = crate::cache::cached_gguf(filename) {
        if verbose { eprintln!("GGUF already downloaded: {}", existing.display()); }
        return existing;
    }

    crate::cache::ensure_cache_dir().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    // Auto-download without confirmation in --model resolution (interactive
    // users who want a confirmation prompt should use `lumen pull` instead).
    crate::download::download_gguf(repo, filename, &crate::cache::cache_dir(), true)
        .unwrap_or_else(|e| {
            eprintln!("Download failed: {e}");
            std::process::exit(1);
        })
}

#[cfg(feature = "download")]
fn resolve_convert_to_lbc(gguf_path: &std::path::Path, lbc_out: &std::path::Path, verbose: bool) {
    use lumen_convert::convert::{convert_gguf_to_lbc, ConvertOptions};

    let opts = ConvertOptions {
        alignment: 128 * 1024,
        dequantize_to_f32: false,
        requant_to: None,
        target: crate::convert::default_target_for_host(),
    };

    if verbose { eprintln!("Converting to LBC: {} -> {}", gguf_path.display(), lbc_out.display()); }
    match convert_gguf_to_lbc(gguf_path, lbc_out, &opts) {
        Ok(stats) => { if verbose { eprintln!("{stats}"); } }
        Err(e) => {
            eprintln!("Conversion failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Decode and print generated tokens as text when a tokenizer is available.
/// Only the generated ANSWER goes to stdout (for piping). When
/// `enable_thinking` is set, the model's `<think>...</think>` reasoning trace
/// is split off via the SAME shared [`lumen_runtime::tooling::ReasoningExtractor`]
/// the server uses and routed to stderr under a `[reasoning]` label, so it
/// never pollutes the piped answer. No headers or decoration on the answer.
///
/// With `enable_thinking == false` (the default) the extractor is a pure
/// passthrough, so this is byte-identical to the prior `println!("{text}")`.
fn print_generated_text(
    tokens: &[u32],
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    enable_thinking: bool,
) {
    if let Some(tok) = tokenizer {
        let stop_ids = &tok.stop_token_ids;
        let clean: Vec<u32> = tokens.iter().copied().filter(|t| !stop_ids.contains(t)).collect();
        // Diagnostic (default OFF): dump raw generated token ids plus the
        // per-id decoded byte string, so a model-decode degenerate token
        // (e.g. an extra "lication" id) is distinguishable from any
        // detokenizer effect. The streaming detok is a pure per-id byte
        // concat, so a doubled sub-word here means the model emitted it.
        if std::env::var("LUMEN_DUMP_TOKEN_IDS").is_ok() {
            eprintln!("[DUMP_TOKEN_IDS] count={}", clean.len());
            eprintln!("[DUMP_TOKEN_IDS] ids={clean:?}");
            for &id in &clean {
                let frag = String::from_utf8_lossy(&tok.decode_bytes(&[id])).to_string();
                eprintln!("[DUMP_TOKEN_IDS] id={id} frag={frag:?}");
            }
        }
        let text = tok.decode(&clean);
        if !enable_thinking {
            // F8: strip `<tool_call>...</tool_call>` blocks from stdout via the
            // SAME shared parser both server surfaces use, so the CLI answer
            // matches the server `content`. BYTE-IDENTITY: `parse_final` on text
            // with no `<tool_call>` returns `content == input` verbatim, so plain
            // (non-tool) output is identical to the historical `println!("{text}")`.
            let parsed = lumen_runtime::tooling::parse_final(&text);
            print_parsed_tool_calls(&parsed.tool_calls);
            // F4: truncate the answer at the first `--stop` match (no-op when no
            // `--stop` was passed → byte-identical).
            println!("{}", apply_stop_truncation(&parsed.content));
            return;
        }
        // Split reasoning from answer with the shared extractor (batch decode,
        // so one feed + finish reconstructs the full split).
        let mut extractor = lumen_runtime::tooling::ReasoningExtractor::new(true);
        let mut delta = extractor.feed(&text);
        let tail = extractor.finish();
        delta.reasoning.push_str(&tail.reasoning);
        delta.content.push_str(&tail.content);
        if !delta.reasoning.is_empty() {
            eprintln!("[reasoning] {}", delta.reasoning);
        }
        // F8: strip tool-call blocks from the ANSWER content too (the reasoning
        // trace already went to stderr above). Same byte-identity guarantee for
        // tool-free content.
        let parsed = lumen_runtime::tooling::parse_final(&delta.content);
        print_parsed_tool_calls(&parsed.tool_calls);
        // F4: stop truncation applies to the answer, not the reasoning trace.
        println!("{}", apply_stop_truncation(&parsed.content));
    }
}

/// F4: truncate `content` at the earliest occurrence of any process-global
/// `--stop` string (the matched bytes and everything after are dropped, exactly
/// like the server's "emit up-to-but-excluding the stop" rule). Because the CLI
/// prints the FULL generated answer in one batch, a simple earliest-substring
/// search is sufficient (no cross-chunk straddle to handle). Returns `content`
/// UNCHANGED when no `--stop` was passed, so non-stop output is byte-identical.
fn apply_stop_truncation(content: &str) -> &str {
    match STOP_SEQUENCES.get() {
        Some(stops) => truncate_at_first_stop(content, stops),
        None => content,
    }
}

/// Pure earliest-stop truncation over an explicit stop list, factored out so it
/// is unit-testable without the process-global `STOP_SEQUENCES` OnceLock. Empty
/// list (or no match) → `content` returned verbatim (byte-identity guarantee).
fn truncate_at_first_stop<'a>(content: &'a str, stops: &[String]) -> &'a str {
    let mut cut: Option<usize> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = content.find(s.as_str()) {
            cut = Some(cut.map_or(pos, |c: usize| c.min(pos)));
        }
    }
    match cut {
        Some(pos) => &content[..pos],
        None => content,
    }
}

/// F8: surface parsed tool calls to STDERR (mirroring the `[reasoning]`
/// channel), so stdout carries only the assistant's plain-text answer — the
/// same split both server wire surfaces apply. No-op when there are no calls,
/// so tool-free output is byte-identical.
fn print_parsed_tool_calls(calls: &[lumen_runtime::tooling::ParsedToolCall]) {
    for call in calls {
        eprintln!("[tool_call] {}({})", call.name, call.arguments_json);
    }
}

/// Print the Lumen ASCII banner with inference metrics.
///
/// Only prints when stderr is a TTY (not piped or redirected), so scripted
/// usage and benchmarks get clean output.
fn print_banner(model_name: &str, backend: &str, metrics_summary: &str) {
    // Check if stderr is a TTY. We use stderr because all banner output goes
    // there (eprintln), leaving stdout clean for token output.
    #[cfg(unix)]
    {
        extern "C" { fn isatty(fd: std::ffi::c_int) -> std::ffi::c_int; }
        // stderr = fd 2
        if unsafe { isatty(2) } == 0 {
            return;
        }
    }

    let sep = "\u{2500}".repeat(46);
    eprintln!();
    eprintln!(" _   _   _ __  __ ___ _  _ ");
    eprintln!("| | | | | |  \\/  | __| \\| |");
    eprintln!("| |_| |_| | |\\/| | _|| .` |");
    eprintln!("|____\\___/|_|  |_|___|_|\\_|");
    eprintln!();
    eprintln!(" Rust LLM Inference Engine");
    eprintln!();
    eprintln!("{sep}");
    eprintln!("  Source    github.com/faisalmumtaz89/Lumen");
    eprintln!("  Engine    Lumen v{} (Rust + {backend})", env!("CARGO_PKG_VERSION"));
    eprintln!("  Model     {model_name}");
    // Indent each line of the metrics summary for alignment within the banner.
    for line in metrics_summary.lines() {
        eprintln!("  {line}");
    }
    eprintln!("{sep}");
}

/// Select CPU or Metal backend based on platform and flags.
///
/// Extracted so that both `#[cfg(feature = "cuda")]` and `#[cfg(not(feature = "cuda"))]`
/// branches can call it without duplicating the platform-specific Metal logic.
// `use_metal`/`verbose` are used only on the macOS/Metal build path.
#[allow(unused_variables)]
fn create_cpu_or_metal_backend(
    use_metal: bool,
    use_simd: bool,
    threads: usize,
    verbose: bool,
) -> Box<dyn ComputeBackend> {
    #[cfg(target_os = "macos")]
    if use_metal {
        let metal = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        if verbose { eprintln!("Metal GPU backend: {}", metal.device_name()); }
        return Box::new(metal);
    }

    #[cfg(not(target_os = "macos"))]
    if use_metal {
        eprintln!("Error: --metal is only supported on macOS");
        std::process::exit(1);
    }

    if use_simd {
        Box::new(SimdF32Backend::with_threads(threads))
    } else {
        Box::new(NaiveF32Backend::new())
    }
}

/// Create and initialize the appropriate compute backend.
///
/// All backends are configured through the `ComputeBackend` trait after boxing,
/// so the setup logic (global tensors, quant data, init, profile) is shared.
///
/// `threads` controls the SIMD backend thread pool size:
///   0 = auto-detect (all available cores), N = use exactly N threads.
#[allow(clippy::too_many_arguments)]
fn create_backend(
    use_simd: bool,
    #[allow(unused_variables)]
    use_metal: bool,
    #[allow(unused_variables)]
    use_cuda: bool,
    #[allow(unused_variables)]
    cuda_device: usize,
    threads: usize,
    profile: bool,
    verbose: bool,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    hyperparams: &lumen_format::hyperparams::ModelHyperparams,
    #[allow(unused_variables)]
    output_proj_raw: Vec<u8>,
    #[allow(unused_variables)]
    output_proj_quant: QuantScheme,
    #[allow(unused_variables)]
    embedding_raw: Vec<u8>,
    #[allow(unused_variables)]
    embedding_quant: QuantScheme,
    #[allow(unused_variables)]
    weight_tying: bool,
) -> Box<dyn ComputeBackend> {
    // Construct the concrete backend and box it. All subsequent setup goes
    // through the ComputeBackend trait, keeping the logic backend-agnostic.
    #[allow(unused_mut)]
    let mut backend: Box<dyn ComputeBackend> = {
        // CUDA backend (cross-platform, requires --features cuda at build time).
        #[cfg(feature = "cuda")]
        if use_cuda {
            let cuda = CudaBackend::new(cuda_device).unwrap_or_else(|e| {
                eprintln!("Error: CUDA backend unavailable: {e}");
                std::process::exit(1);
            });
            Box::new(cuda)
        } else {
            create_cpu_or_metal_backend(use_metal, use_simd, threads, verbose)
        }

        #[cfg(not(feature = "cuda"))]
        {
            if use_cuda {
                eprintln!("Error: --cuda requires building with --features cuda");
                std::process::exit(1);
            }
            create_cpu_or_metal_backend(use_metal, use_simd, threads, verbose)
        }
    };

    // Backend-agnostic setup via trait methods.
    backend.set_global_tensors(embedding, final_norm, output_proj);

    // BF16 added to the raw-weight allow-list. Without this, BF16 LBC models silently route
    // through the F32 CPU-dequant fallback and the BF16 raw-weight path
    // is skipped. Load-bearing for the BF16 path's prefill kernel.
    if matches!(output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16) && !output_proj_raw.is_empty() {
        if verbose {
            eprintln!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                output_proj_quant, output_proj_raw.len(), output_proj_raw.len() as f64 / 1048576.0);
        }
        backend.set_output_proj_raw(output_proj_raw, output_proj_quant);
    }
    if matches!(embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !embedding_raw.is_empty() {
        if verbose {
            eprintln!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                embedding_quant, embedding_raw.len(), embedding_raw.len() as f64 / 1048576.0);
        }
        backend.set_embedding_raw(embedding_raw, embedding_quant);
    }
    if weight_tying {
        if verbose { eprintln!("  weight_tying: output_proj shares embedding storage"); }
        backend.set_weight_tying(true);
    }

    backend.init(hyperparams).unwrap_or_else(|e| {
        eprintln!("Error: backend initialization failed: {e}");
        std::process::exit(1);
    });
    if profile { backend.set_profile(true); }

    backend
}

#[allow(clippy::too_many_arguments)]
fn run_with_async(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
    kv_precision: KvPrecision,
    session_flags: &SessionFlags,
    enable_thinking: bool,
) {
    let provider = AsyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    // feed the LBC-resolved dense quant into the runtime
    // defaults registry BEFORE the backend is constructed so the cached
    // `LUMEN_CUDA_BF16_GEMMEX` / `LUMEN_CUDA_DECODE_GRAPH*` resolvers
    // observe the model-aware default on their first read. CLI path stays
    // on `set_path_is_server(false)` (set in `main`) so the decode-delay
    // default remains 0 µs — CLI is fork-deterministic.
    lumen_runtime::runtime_defaults::set_model_dense_quant(provider.output_proj_quant);
    // PRIMARY (bulk) quant — body attn/FFN scheme, NOT output_proj (which is
    // Q8_0 for both 27B-q4 and 27B-q8). Drives the per-quant attention-precision
    // default (27B q4/q8/bf16 all -> pvf32; quant unset -> legacy WMMA).
    lumen_runtime::runtime_defaults::set_model_primary_quant(
        provider.lbc().header.quantization.scheme,
    );
    lumen_runtime::runtime_defaults::set_model_block_count(
        provider.lbc().header.hyperparams.num_layers,
    );
    // Feed the MoE flag into the runtime defaults so the Q8-only
    // resolvers (Q8_SPLIT / OUTPUT_PROJ_SPLIT / Q8_SCALE_HW /
    // OUTPUT_PROJ_NR / FFN_FUSED_GLU_SKIP) stay OFF on MoE 30B-A3B.
    // Without this gate, `q8_split_default()` returned `true` for Q8 MoE
    // and the SPLIT clone pass corrupted MoE decode into 1 valid token
    // followed by 159 `[PAD248319]` per prompt.
    lumen_runtime::runtime_defaults::set_model_is_moe(
        provider.lbc().header.hyperparams.num_experts.unwrap_or(0) > 0,
    );

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

    // pass the right-sized max_seq_len through to the backend via
    // hyperparams so the CUDA KV cache is sized exactly for the requested
    // context (no silent ceiling). Mirrors the mmap path's `hyperparams_capped`.
    let mut hyperparams_capped = provider.lbc().header.hyperparams;
    hyperparams_capped.max_seq_len = max_seq_len as u32;

    let backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hyperparams_capped,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 2,
        kv_precision,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let mut engine = InferenceEngine::new(config, hyperparams_capped);
    // Install the anti-restate guard's per-token-id byte decoder (powers the
    // sub-word-doubling rule). No-op for the id-only n-gram restate rule and
    // when the guard is disabled (dense / LUMEN_ANTI_RESTATE=0).
    if let Some(tok) = tokenizer {
        engine.set_token_decoder(tok.byte_decoder_fn());
    }
    let live = LiveModel::from_lbc(provider.lbc(), provider.output_proj_quant);
    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate,
        &hyperparams_capped,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
        session_flags, &live, enable_thinking,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_with_sync(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
    kv_precision: KvPrecision,
    session_flags: &SessionFlags,
    enable_thinking: bool,
) {
    let provider = SyncWeightProvider::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    // see the AsyncWeightProvider branch for rationale. The
    // sync provider follows the same pattern — set the model-aware
    // defaults BEFORE any backend constructor runs.
    lumen_runtime::runtime_defaults::set_model_dense_quant(provider.output_proj_quant);
    // PRIMARY (bulk) quant — body attn/FFN scheme, NOT output_proj (which is
    // Q8_0 for both 27B-q4 and 27B-q8). Drives the per-quant attention-precision
    // default (27B q4/q8/bf16 all -> pvf32; quant unset -> legacy WMMA).
    lumen_runtime::runtime_defaults::set_model_primary_quant(
        provider.lbc().header.quantization.scheme,
    );
    lumen_runtime::runtime_defaults::set_model_block_count(
        provider.lbc().header.hyperparams.num_layers,
    );
    // Feed the MoE flag alongside the dense-quant hint.
    lumen_runtime::runtime_defaults::set_model_is_moe(
        provider.lbc().header.hyperparams.num_experts.unwrap_or(0) > 0,
    );

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

    // pass the right-sized max_seq_len through to the backend via
    // hyperparams so the CUDA KV cache is sized exactly for the requested
    // context (no silent ceiling). Mirrors the mmap path's `hyperparams_capped`.
    let mut hyperparams_capped = provider.lbc().header.hyperparams;
    hyperparams_capped.max_seq_len = max_seq_len as u32;

    // `mut` + `#[allow(unused_mut)]`: the backend is mutated by the
    // `preload_weights` call below, but only on builds/platforms where a
    // GPU-resident backend is selected (Metal on macOS, CUDA with --features
    // cuda). On a pure CPU build neither branch fires, so the binding would be
    // flagged unused-mut without this allow. Mirrors `run_with_async`.
    #[allow(unused_mut)]
    let mut backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hyperparams_capped,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    // GPU-resident weight upload for the Metal AND CUDA backends.
    //
    // Mirrors `lumen-server` (which calls `preload_weights` after `init`),
    // `run_with_async` (which preloads for `gpu_resident && use_cuda`), and the
    // mmap path's `preload_weights_gpu_resident`. This is REQUIRED, not an
    // optimization: GatedDeltaNet layers allocate their recurrent state buffers
    // (`gdn_h_states` / `gdn_conv_states`) inside the GPU-resident preload, and
    // the batched-prefill GDN kernel indexes them unconditionally. Without the
    // preload the sync path errors ("GDN layers require GPU-resident weights" on
    // CUDA; `gdn.rs` panic on Metal) on the first GDN layer. `preload_weights`
    // is a no-op on the CPU/SIMD backends, so it is only consequential when a
    // GPU backend (Metal on macOS, CUDA with --features cuda) was selected.
    if use_metal || use_cuda {
        backend.preload_weights(&provider).unwrap_or_else(|e| {
            eprintln!("Error: GPU-resident preload failed: {e}");
            std::process::exit(1);
        });
    }

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 1,
        kv_precision,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let mut engine = InferenceEngine::new(config, hyperparams_capped);
    // Install the anti-restate guard's per-token-id byte decoder (powers the
    // sub-word-doubling rule). No-op for the id-only n-gram restate rule and
    // when the guard is disabled (dense / LUMEN_ANTI_RESTATE=0).
    if let Some(tok) = tokenizer {
        engine.set_token_decoder(tok.byte_decoder_fn());
    }
    let live = LiveModel::from_lbc(provider.lbc(), provider.output_proj_quant);
    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate,
        &hyperparams_capped,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
        session_flags, &live, enable_thinking,
    );
}

#[allow(clippy::too_many_arguments)]
// option_a / verbose_routing / routing_bias / verbose are used on the
// Metal / non-CUDA decode paths; unused on the CUDA decode path.
#[allow(unused_variables)]
fn run_with_mmap(
    path: &Path,
    use_simd: bool,
    use_metal: bool,
    use_cuda: bool,
    cuda_device: usize,
    use_accelerate: bool,
    gpu_resident: bool,
    option_a: bool,
    threads: usize,
    profile: bool,
    verbose: bool,
    verbose_routing: bool,
    routing_bias: Option<f32>,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    context_len: Option<usize>,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    model_display: &str,
    backend_name: &str,
    kv_precision: KvPrecision,
    session_flags: &SessionFlags,
    enable_thinking: bool,
) {
    let mmap_config = MmapConfig {
        prefetch_window: 2,
        advise_sequential: true,
        release_with_dontneed: true,
    };
    let provider = MmapWeightProvider::open(path, mmap_config).unwrap_or_else(|e| {
        eprintln!("Error opening model: {e}");
        std::process::exit(1);
    });

    // see the AsyncWeightProvider branch for rationale.
    lumen_runtime::runtime_defaults::set_model_dense_quant(provider.output_proj_quant);
    // PRIMARY (bulk) quant — body attn/FFN scheme, NOT output_proj (which is
    // Q8_0 for both 27B-q4 and 27B-q8). Drives the per-quant attention-precision
    // default (27B q4/q8/bf16 all -> pvf32; quant unset -> legacy WMMA).
    lumen_runtime::runtime_defaults::set_model_primary_quant(
        provider.lbc().header.quantization.scheme,
    );
    lumen_runtime::runtime_defaults::set_model_block_count(
        provider.lbc().header.hyperparams.num_layers,
    );
    // Feed the MoE flag alongside the dense-quant hint.
    lumen_runtime::runtime_defaults::set_model_is_moe(
        provider.lbc().header.hyperparams.num_experts.unwrap_or(0) > 0,
    );

    let model_max = provider.lbc().header.hyperparams.max_seq_len as usize;
    let max_seq_len = effective_max_seq_len(model_max, context_len, prompt_tokens.len(), match stop { StopCondition::MaxTokens(n) => *n, StopCondition::MaxTokensOrEos { max_tokens, .. } => *max_tokens, StopCondition::EosTokens(_) => model_max }, verbose);

    // Create hyperparams with capped max_seq_len for Metal init.
    // This ensures KV cache, RoPE tables, and scratch buffers use the capped value.
    let mut hyperparams_capped = provider.lbc().header.hyperparams;
    hyperparams_capped.max_seq_len = max_seq_len as u32;

    let config = RuntimeConfig {
        pipeline_mode: PipelineMode::MinMem,
        prefetch_distance: 2,
        kv_precision,
        max_seq_len,
        collect_per_layer_timings: profile,
    };

    let mut engine = InferenceEngine::new(config, hyperparams_capped);
    // Install the anti-restate guard's per-token-id byte decoder (powers the
    // sub-word-doubling rule). No-op for the id-only n-gram restate rule and
    // when the guard is disabled (dense / LUMEN_ANTI_RESTATE=0).
    if let Some(tok) = tokenizer {
        engine.set_token_decoder(tok.byte_decoder_fn());
    }

    // Metal path: use concrete MetalF32Backend for both batched prefill + decode
    #[cfg(target_os = "macos")]
    if use_metal {
        if verbose {
            eprintln!("Prompt tokens: {prompt_tokens:?}");
            eprintln!("Running inference with Metal GPU batched prefill...\n");
        }

        let mut metal = MetalF32Backend::new().unwrap_or_else(|e| {
            eprintln!("Error: Metal backend unavailable: {e}");
            std::process::exit(1);
        });
        if verbose { eprintln!("Metal GPU backend: {}", metal.device_name()); }
        metal.set_global_tensors(
            provider.embedding.clone(),
            provider.final_norm.clone(),
            provider.output_proj.clone(),
        );
        if matches!(provider.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.output_proj_raw.is_empty() {
            if verbose {
                eprintln!("  output_proj: {:?} ({} bytes, ~{:.1} MB)",
                    provider.output_proj_quant, provider.output_proj_raw.len(), provider.output_proj_raw.len() as f64 / 1048576.0);
            }
            metal.set_output_proj_raw(provider.output_proj_raw.clone(), provider.output_proj_quant);
        }
        if matches!(provider.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) && !provider.embedding_raw.is_empty() {
            if verbose {
                eprintln!("  embedding: {:?} ({} bytes, ~{:.1} MB)",
                    provider.embedding_quant, provider.embedding_raw.len(), provider.embedding_raw.len() as f64 / 1048576.0);
            }
            metal.set_embedding_raw(provider.embedding_raw.clone(), provider.embedding_quant);
        }
        if provider.weight_tying {
            metal.set_weight_tying(true);
        }
        // Configure MoE expert caching for streaming mode (non-GPU-resident).
        // Default capacity: 4 experts per MoE layer (covers ~40-50% activations
        // based on heavy-tailed expert distribution research).
        if hyperparams_capped.num_experts.unwrap_or(0) > 0 && !gpu_resident {
            let num_layers = hyperparams_capped.num_layers as usize;
            let cache_capacity = num_layers * 4; // 4 hot experts/layer
            metal.configure_expert_cache(path, cache_capacity);
            if verbose {
                eprintln!(
                    "MoE expert cache: {} experts/layer x {} layers = {} capacity",
                    4, num_layers, cache_capacity,
                );
            }
        }
        // Enable router diagnostics before init so per-layer buffers are allocated.
        if verbose_routing {
            metal.configure_router_debug(true);
            if verbose { eprintln!("Router diagnostics: enabled (--verbose-routing)"); }
        }
        // Configure cache-conditional routing bias.
        if let Some(lambda) = routing_bias {
            metal.configure_routing_bias(lambda);
            if verbose { eprintln!("Routing bias \u{03bb}={lambda} (cache-conditional routing active)"); }
        }
        metal.init(&hyperparams_capped).unwrap_or_else(|e| {
            eprintln!("Error: Metal initialization failed: {e}");
            std::process::exit(1);
        });
        if profile { metal.set_profile(true); }

        // Configure MoE Option A dispatch (streaming + GPU-resident).
        // When enabled, only top-K experts are dispatched per token instead of all 8.
        // Option A now works in GPU-resident mode via two-CB split per MoE layer.
        if option_a {
            metal.configure_option_a(true);
            let top_k = hyperparams_capped.num_active_experts.unwrap_or(2) as usize;
            if !gpu_resident {
                // Streaming mode: warmup profiler to populate expert cache.
                metal.configure_expert_warmup(8, top_k);
                if verbose { eprintln!("Option A enabled: top-{} expert dispatch (streaming mode)", top_k); }
            } else {
                if verbose { eprintln!("Option A enabled: top-{} expert dispatch (GPU-resident mode)", top_k); }
            }
        }

        // Pre-load all layer weights into GPU-resident Metal buffers.
        // Eliminates TLB misses and page table walks from mmap access.
        if gpu_resident {
            metal.preload_weights_gpu_resident(&provider).unwrap_or_else(|e| {
                eprintln!("Error: GPU-resident preload failed: {e}");
                std::process::exit(1);
            });
        }

        // Route through the session-driven path when
        // --session-resume / --session-save is set; otherwise use the
        // legacy direct-KV `engine.generate` path. Both produce a
        // `GenerationResult` of the same shape.
        // This Metal batched-prefill path bypasses `run_engine`, so resolve
        // the model-aware `repetition_penalty` default here too (MoE → 1.03
        // when the operator did not pass `--repeat-penalty`).
        let resolved_sampling = effective_sampling(sampling);
        let live = LiveModel::from_lbc(provider.lbc(), provider.output_proj_quant);
        let gen_result = run_generation(
            &engine,
            &provider,
            &metal as &dyn ComputeBackend,
            prompt_tokens,
            stop,
            &resolved_sampling,
            session_flags,
            &live,
            verbose,
        );
        match gen_result {
            Ok(result) => {
                print_generated_text(&result.tokens, tokenizer, enable_thinking);
                if verbose {
                    eprintln!("Generated tokens: {:?}", result.tokens);
                    eprintln!("\n--- Metrics ---");
                    eprintln!("{}", result.metrics.summary());
                    metal.print_profile();
                    // Print MoE expert caching stats.
                    if let Some(summary) = metal.expert_profiler_summary() {
                        eprintln!("\n--- MoE Expert Profiler ---");
                        eprintln!("Total record() calls: {}", summary.total_tokens);
                        for (i, entropy) in summary.per_layer_entropy.iter().enumerate() {
                            eprintln!("  Layer {i}: entropy={entropy:.3}");
                        }
                        if !summary.global_top_experts.is_empty() {
                            eprintln!("Top 10 global experts:");
                            for &(layer, eid, freq) in summary.global_top_experts.iter().take(10) {
                                eprintln!("  layer={layer} expert={eid} freq={freq:.3}");
                            }
                        }
                    }
                    if let Some(stats) = metal.expert_cache_stats() {
                        eprintln!("\n--- MoE Expert Cache ---");
                        eprintln!("  cached: {}/{} experts ({} bytes)",
                            stats.cached_experts, stats.capacity, stats.cached_bytes);
                        eprintln!("  hits: {}, misses: {}, hit_rate: {:.1}%",
                            stats.total_hits, stats.total_misses, stats.hit_rate * 100.0);
                    }
                    // MoE expert I/O statistics.
                    let (disk, cache, blob) = metal.expert_io_stats();
                    if disk + cache + blob > 0 {
                        let total = disk + cache + blob;
                        eprintln!("\n--- MoE Expert I/O ---");
                        eprintln!("  from_cache: {} bytes ({:.1}%)",
                            cache, cache as f64 / total as f64 * 100.0);
                        eprintln!("  from_disk:  {} bytes ({:.1}%)",
                            disk, disk as f64 / total as f64 * 100.0);
                        eprintln!("  from_blob:  {} bytes ({:.1}%)",
                            blob, blob as f64 / total as f64 * 100.0);
                        eprintln!("  total:      {} bytes", total);
                    }
                    // Router diagnostics summary.
                    if let Some(summary) = metal.router_debug_summary() {
                        eprintln!("\n{summary}");
                    }
                    print_banner(model_display, backend_name, &result.metrics.summary());
                }
            }
            Err(e) => {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    #[allow(unused_mut)]
    let mut backend = create_backend(
        use_simd,
        use_metal,
        use_cuda,
        cuda_device,
        threads,
        profile,
        verbose,
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
        &hyperparams_capped,
        provider.output_proj_raw.clone(),
        provider.output_proj_quant,
        provider.embedding_raw.clone(),
        provider.embedding_quant,
        provider.weight_tying,
    );

    // GPU-resident preload for CUDA: loads all layer weights to GPU memory.
    // Enables native quantized kernel paths (dp4a, HGEMV) and batched prefill.
    if gpu_resident && use_cuda {
        backend.as_mut().preload_weights(&provider).unwrap_or_else(|e| {
            eprintln!("Error: CUDA GPU-resident preload failed: {e}");
            std::process::exit(1);
        });
    }

    let live = LiveModel::from_lbc(provider.lbc(), provider.output_proj_quant);
    run_engine(
        &engine, &provider, backend.as_ref(), use_accelerate,
        &hyperparams_capped,
        &provider.embedding, &provider.final_norm,
        prompt_tokens, stop, sampling, tokenizer,
        verbose, model_display, backend_name,
        session_flags, &live, enable_thinking,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    engine: &InferenceEngine,
    weights: &dyn WeightProvider,
    backend: &dyn ComputeBackend,
    #[allow(unused_variables)]
    use_accelerate: bool,
    #[allow(unused_variables)]
    hyperparams: &lumen_format::hyperparams::ModelHyperparams,
    #[allow(unused_variables)]
    embedding: &[f32],
    #[allow(unused_variables)]
    final_norm: &[f32],
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    verbose: bool,
    model_display: &str,
    backend_name: &str,
    session_flags: &SessionFlags,
    live: &LiveModel,
    enable_thinking: bool,
) {
    if verbose { eprintln!("Prompt tokens: {prompt_tokens:?}"); }

    // Resolve the model-aware `repetition_penalty` default now that the LBC is
    // open and `runtime_defaults::set_model_is_moe` has run (called by every
    // `run_with_*` before reaching here). When the operator did not pass
    // `--repeat-penalty`, MoE models get 1.03 (the GDN parity stack lands the
    // math at greedy; 1.05+ corrupts MoE arithmetic); dense and explicit-override
    // paths are unchanged.
    let resolved_sampling = effective_sampling(sampling);
    let sampling = &resolved_sampling;

    // Metal batched prefill is handled in the run_with_* functions directly,
    // not here, to avoid creating a second Metal backend.

    #[cfg(target_os = "macos")]
    if use_accelerate {
        if verbose { eprintln!("Running inference with Accelerate batched prefill...\n"); }
        if session_flags.is_active() {
            // The Accelerate path uses a separate prefill backend; the
            // session-driven path is not yet plumbed through it because
            // `Session::extend_with_prefill_backend` exists but the
            // `generate_with_session` API only accepts a single
            // `ComputeBackend`. Fail explicitly so the operator knows the
            // combination is intentionally unsupported.
            eprintln!(
                "Error: --session-resume / --session-save is not supported \
                 with --accelerate (the Accelerate batched-prefill path \
                 uses a distinct backend; resume requires the unified \
                 compute backend). Re-run without --accelerate."
            );
            std::process::exit(2);
        }
        let mut accel = AccelerateBatchBackend::new(
            hyperparams,
            prompt_tokens.len(),
            embedding.to_vec(),
            final_norm.to_vec(),
        );
        match engine.generate_with_prefill(prompt_tokens, weights, backend, &mut accel, stop, sampling) {
            Ok(result) => {
                print_generated_text(&result.tokens, tokenizer, enable_thinking);
                if verbose {
                    eprintln!("Generated tokens: {:?}", result.tokens);
                    eprintln!("\n--- Metrics ---");
                    eprintln!("{}", result.metrics.summary());
                    backend.print_profile();
                    print_banner(model_display, backend_name, &result.metrics.summary());
                }
            }
            Err(e) => {
                eprintln!("Inference error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    #[cfg(not(target_os = "macos"))]
    if use_accelerate {
        eprintln!("Error: --accelerate is only supported on macOS");
        std::process::exit(1);
    }

    if verbose { eprintln!("Running inference...\n"); }

    let gen_result = run_generation(
        engine,
        weights,
        backend,
        prompt_tokens,
        stop,
        sampling,
        session_flags,
        live,
        verbose,
    );
    match gen_result {
        Ok(result) => {
            print_generated_text(&result.tokens, tokenizer, enable_thinking);
            if verbose {
                eprintln!("Generated tokens: {:?}", result.tokens);
                eprintln!("\n--- Metrics ---");
                eprintln!("{}", result.metrics.summary());
                backend.print_profile();
                print_banner(model_display, backend_name, &result.metrics.summary());
            }
        }
        Err(e) => {
            eprintln!("Inference error: {e}");
            std::process::exit(1);
        }
    }
}

/// Route a single generation request through either the
/// session-driven path (when `--session-resume` / `--session-save` is set)
/// or the legacy direct-KV path (otherwise).
///
/// Both branches produce a [`GenerationResult`] of the same shape; callers
/// can pipe the output the same way regardless of which branch ran.
///
/// Session lifecycle for the session-driven branch:
/// 1. If `--session-resume` is set, [`Session::load_from_disk`] is called
///    with the live model fingerprint. A fingerprint mismatch surfaces an
///    actionable error and aborts (exit 2). Tokens, KV bytes, and any GDN
///    recurrent state are restored bit-identically.
/// 2. Otherwise a fresh [`Session`] is constructed with the same
///    `RuntimeConfig` + `ModelHyperparams` + `SamplingParams` the legacy
///    path uses.
/// 3. The engine runs [`InferenceEngine::generate_with_session`], which
///    re-uses any cached prefix on a resumed session and falls back to a
///    cold prefill on a fresh session (Path 5 of `extend_with_cache`).
/// 4. If `--session-save` is set, [`Session::save_to_disk`] is called with
///    the resolved fingerprint. The atomic-rename publishing means a
///    concurrent loader never sees a partial write.
#[allow(clippy::too_many_arguments)]
fn run_generation(
    engine: &InferenceEngine,
    weights: &dyn WeightProvider,
    backend: &dyn ComputeBackend,
    prompt_tokens: &[u32],
    stop: &StopCondition,
    sampling: &SamplingParams,
    session_flags: &SessionFlags,
    live: &LiveModel,
    verbose: bool,
) -> Result<GenerationResult, lumen_runtime::error::RuntimeError> {
    if !session_flags.is_active() {
        // Legacy path — bit-exact when disabled.
        return engine.generate(prompt_tokens, weights, backend, stop, sampling);
    }

    // Session-driven path: compute the fingerprint once, then construct
    // the Session via either `load_from_disk` or `new`.
    let fingerprint = live.fingerprint();

    let mut session = if let Some(ref p) = session_flags.resume_path {
        if verbose {
            eprintln!("[session] resuming from {p}");
        }
        let path = std::path::Path::new(p);
        Session::load_from_disk(
            path,
            engine.config().clone(),
            *engine.hyperparams(),
            sampling.clone(),
            backend,
            &fingerprint,
        )
        .map_err(|e| {
            // Wrap with a session-resume hint so the operator immediately
            // sees what failed.
            lumen_runtime::error::RuntimeError::Compute(format!(
                "--session-resume {p}: {e}"
            ))
        })?
    } else {
        Session::new(
            engine.config().clone(),
            *engine.hyperparams(),
            sampling.clone(),
        )?
    };

    let result =
        engine.generate_with_session(&mut session, prompt_tokens, weights, backend, stop)?;

    if let Some(ref p) = session_flags.save_path {
        if verbose {
            eprintln!("[session] saving to {p}");
        }
        let path = std::path::Path::new(p);
        if let Err(e) = session.save_to_disk(path, backend, &fingerprint) {
            return Err(lumen_runtime::error::RuntimeError::Compute(format!(
                "--session-save {p}: {e}"
            )));
        }
    }

    Ok(result)
}

// ---- --kv-precision parsing tests ------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kv_precision_accepts_f16_variants() {
        assert_eq!(parse_kv_precision("f16").unwrap(), KvPrecision::F16);
        assert_eq!(parse_kv_precision("F16").unwrap(), KvPrecision::F16);
        assert_eq!(parse_kv_precision("fp16").unwrap(), KvPrecision::F16);
        assert_eq!(parse_kv_precision("half").unwrap(), KvPrecision::F16);
    }

    #[test]
    fn parse_kv_precision_accepts_f32_variants() {
        assert_eq!(parse_kv_precision("f32").unwrap(), KvPrecision::F32);
        assert_eq!(parse_kv_precision("F32").unwrap(), KvPrecision::F32);
        assert_eq!(parse_kv_precision("fp32").unwrap(), KvPrecision::F32);
        assert_eq!(parse_kv_precision("float").unwrap(), KvPrecision::F32);
    }

    #[test]
    fn parse_kv_precision_rejects_unknown() {
        assert!(parse_kv_precision("int8").is_err());
        assert!(parse_kv_precision("q4_0").is_err());
        assert!(parse_kv_precision("").is_err());
        assert!(parse_kv_precision("garbage").is_err());
    }

    #[test]
    fn resolve_kv_precision_honors_cli_override() {
        let p = resolve_kv_precision(Some(KvPrecision::F16), false, false, false);
        assert_eq!(p, KvPrecision::F16);
        let p = resolve_kv_precision(Some(KvPrecision::F32), true, false, false);
        assert_eq!(p, KvPrecision::F32, "CLI flag must override Metal default");
    }

    #[test]
    fn resolve_kv_precision_metal_defaults_to_f16() {
        // Clear env to ensure backend default takes effect. Serialise on the
        // shared ENV_LOCK and restore the prior value before asserting so this
        // mutation never races (or leaks to) a sibling env-reading test.
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_KV_PRECISION").ok();
        std::env::remove_var("LUMEN_KV_PRECISION");
        let p = resolve_kv_precision(None, true, false, false);
        if let Some(v) = saved {
            std::env::set_var("LUMEN_KV_PRECISION", v);
        }
        assert_eq!(p, KvPrecision::F16);
    }

    #[test]
    fn resolve_kv_precision_cuda_defaults_to_f32() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_KV_PRECISION").ok();
        std::env::remove_var("LUMEN_KV_PRECISION");
        let p = resolve_kv_precision(None, false, true, false);
        if let Some(v) = saved {
            std::env::set_var("LUMEN_KV_PRECISION", v);
        }
        assert_eq!(p, KvPrecision::F32);
    }

    #[test]
    fn resolve_kv_precision_cpu_defaults_to_f32() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_KV_PRECISION").ok();
        std::env::remove_var("LUMEN_KV_PRECISION");
        let p = resolve_kv_precision(None, false, false, false);
        if let Some(v) = saved {
            std::env::set_var("LUMEN_KV_PRECISION", v);
        }
        assert_eq!(p, KvPrecision::F32);
    }

    // --- F1: CLI honours LUMEN_FREQUENCY_PENALTY / LUMEN_REPEAT_LAST_N when
    // the flag is unset, and an explicit flag overrides the env. These mutate
    // process env, so serialise on a module-local lock to avoid cross-test
    // interference.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn cli_frequency_penalty_honors_env_when_flag_unset_but_flag_overrides() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_FREQUENCY_PENALTY").ok();

        // Flag unset (explicit=false) + env set → CLI adopts the env value,
        // matching the server wire.
        std::env::set_var("LUMEN_FREQUENCY_PENALTY", "0.4");
        let env_set = cli_resolve_frequency_penalty(None, false);

        // Flag passed (explicit=true) → the parsed value wins over the env.
        // (--frequency-penalty 0 normalizes to None at the parser, i.e. an
        // explicit disable that must NOT be re-enabled by the env.)
        let flag_disable = cli_resolve_frequency_penalty(None, true);
        let flag_value = cli_resolve_frequency_penalty(Some(0.2), true);

        // Flag unset + env unset → no-op (None), byte-identical to today.
        std::env::remove_var("LUMEN_FREQUENCY_PENALTY");
        let env_unset = cli_resolve_frequency_penalty(None, false);

        // Restore BEFORE asserting so a failed assert cannot leak env state.
        match saved {
            Some(v) => std::env::set_var("LUMEN_FREQUENCY_PENALTY", v),
            None => std::env::remove_var("LUMEN_FREQUENCY_PENALTY"),
        }

        assert_eq!(env_set, Some(0.4));
        assert_eq!(flag_disable, None);
        assert_eq!(flag_value, Some(0.2));
        assert_eq!(env_unset, None);
    }

    #[test]
    fn cli_repeat_last_n_honors_env_when_flag_unset_but_flag_overrides() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_REPEAT_LAST_N").ok();

        // Flag unset + env set → CLI adopts the env window.
        std::env::set_var("LUMEN_REPEAT_LAST_N", "64");
        let env_set = cli_resolve_repeat_last_n(None, false);

        // Flag passed wins over env (an explicit `--repeat-last-n 0` → None
        // disable must not be re-windowed by the env).
        let flag_disable = cli_resolve_repeat_last_n(None, true);
        let flag_value = cli_resolve_repeat_last_n(Some(32), true);

        // Flag unset + env unset → None (full-history), byte-identical.
        std::env::remove_var("LUMEN_REPEAT_LAST_N");
        let env_unset = cli_resolve_repeat_last_n(None, false);

        // Restore BEFORE asserting so a failed assert cannot leak env state.
        match saved {
            Some(v) => std::env::set_var("LUMEN_REPEAT_LAST_N", v),
            None => std::env::remove_var("LUMEN_REPEAT_LAST_N"),
        }

        assert_eq!(env_set, Some(64));
        assert_eq!(flag_disable, None);
        assert_eq!(flag_value, Some(32));
        assert_eq!(env_unset, None);
    }

    // --- F8: CLI strips <tool_call> blocks from stdout via the shared
    // `parse_final`, byte-identical for tool-free text. These assert the exact
    // text the CLI prints (`parse_final(...).content`) on the default
    // (thinking-off) answer path.

    #[test]
    fn f8_parse_final_plain_text_is_byte_identical() {
        // The load-bearing F8 invariant: text with NO <tool_call> round-trips
        // verbatim, so the historical `println!("{text}")` output is unchanged.
        for s in [
            "Hello, world.",
            "multi\nline\noutput",
            "trailing newline\n",
            "ends with an angle bracket <",
            "less-than in math: a < b and c > d",
            "",
        ] {
            let parsed = lumen_runtime::tooling::parse_final(s);
            assert_eq!(parsed.content, s, "parse_final must echo tool-free text verbatim: {s:?}");
            assert!(parsed.tool_calls.is_empty());
        }
    }

    #[test]
    fn f8_parse_final_strips_tool_call_block_from_content() {
        // A <tool_call> block is removed from the printed content and surfaced
        // as a structured call (mirrors both server surfaces).
        let text = "Sure.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call> Done.";
        let parsed = lumen_runtime::tooling::parse_final(text);
        assert!(!parsed.content.contains("<tool_call>"), "marker must be stripped");
        assert!(!parsed.content.contains("get_weather"), "tool JSON must not leak to stdout");
        assert!(parsed.content.contains("Sure."));
        assert!(parsed.content.contains("Done."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
    }

    // --- F4: CLI `--stop` truncation (batch). Tested through the pure
    // `truncate_at_first_stop` helper so the process-global OnceLock is not
    // disturbed.

    #[test]
    fn f4_cli_empty_stop_list_is_byte_identical() {
        // No --stop (empty list) → content returned verbatim, byte-identical.
        let full = "alpha STOP beta";
        assert_eq!(truncate_at_first_stop(full, &[]), full);
        assert_eq!(truncate_at_first_stop(full, &["".to_string()]), full, "empty stop entries are ignored");
        assert_eq!(truncate_at_first_stop(full, &["ZZZ".to_string()]), full, "no match → verbatim");
    }

    #[test]
    fn f4_cli_stop_truncates_at_first_match() {
        let full = "keep this STOP drop this";
        assert_eq!(truncate_at_first_stop(full, &["STOP".to_string()]), "keep this ");
        // Earliest of multiple stops wins.
        let two = "aaa END bbb STOP ccc";
        assert_eq!(
            truncate_at_first_stop(two, &["STOP".to_string(), "END".to_string()]),
            "aaa ",
            "the earliest-starting stop string must win"
        );
    }
}
