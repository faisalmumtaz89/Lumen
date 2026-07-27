//! Audited fixed-decode benchmark mode (BENCHMARK ARTIFACT — not shipped).
//!
//! # Why this exists
//!
//! The shipping CLI's `Decode: X tok/s` is **not** a valid engine-vs-engine
//! number:
//!
//! * The decode timer (`engine.rs`, `decode_start`) is taken *before* the first
//!   token is sampled from the **prefill** logits. A 256-token generation
//!   therefore performs only **255** backend decode calls, but the reported
//!   figure divides **256** by that interval — a systematic ~+0.4% overstatement
//!   versus `llama-bench tg256`, which performs 256 real decode forwards.
//! * `StopCondition::MaxTokensOrEos` lets a run stop early on EOS, so a
//!   "256-token" battery can silently be a 40-token battery.
//!
//! # What this mode does instead
//!
//! It runs the **same** engine, backend, weights and prompt the CLI just
//! configured — this hook sits at the top of `run_engine`, *after* every
//! `run_with_*` has opened the provider, installed the four model-aware
//! setters (`set_model_dense_quant` / `set_model_primary_quant` /
//! `set_model_block_count` / `set_model_is_moe`) and fully configured the
//! backend. Setup divergence is therefore impossible by construction: that
//! class of bug (a bespoke harness that skipped the setters) once understated a
//! whole 9-cell board by up to 55%.
//!
//! Per measured sequence it asks for `steps + 1` tokens under a **pure
//! `MaxTokens`** stop condition (no EOS branch → EOS cannot shorten a run), so
//! exactly `steps` backend decode calls execute after the prefill-produced
//! token. Throughput is then
//!
//! ```text
//! tok/s = steps / metrics.decode_time
//! ```
//!
//! `generate_with_prefill` builds a fresh `KvCache` per call, so recurrent/KV
//! state is reset between sequences. One warm weight-resident process runs
//! `warmup` discarded sequences followed by `runs` measured ones.
//!
//! Activated only when `LUMEN_DECODE_BENCH=1`; unset, this file changes nothing.
//!
//! Configuration (env, so the CLI's argument surface is untouched):
//!   LUMEN_DECODE_BENCH=1            enable
//!   LUMEN_DECODE_BENCH_STEPS=256    decode calls per sequence
//!   LUMEN_DECODE_BENCH_RUNS=5       measured sequences
//!   LUMEN_DECODE_BENCH_WARMUP=1     discarded sequences
//!   LUMEN_DECODE_BENCH_CELL=9b-q4   label recorded in the JSON
//!   LUMEN_DECODE_BENCH_JSON=<path>  write the result JSON here

use lumen_runtime::engine::{InferenceEngine, SamplingParams, StopCondition};
use lumen_runtime::weight::cache::WeightProvider;
use lumen_runtime::ComputeBackend;
use std::time::Instant;

pub(crate) struct BenchCfg {
    pub steps: usize,
    pub runs: usize,
    pub warmup: usize,
    pub cell: String,
    pub json: Option<String>,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// `Some(cfg)` when `LUMEN_DECODE_BENCH` is truthy.
pub(crate) fn from_env() -> Option<BenchCfg> {
    let on = matches!(
        std::env::var("LUMEN_DECODE_BENCH").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    if !on {
        return None;
    }
    Some(BenchCfg {
        steps: env_usize("LUMEN_DECODE_BENCH_STEPS", 256),
        runs: env_usize("LUMEN_DECODE_BENCH_RUNS", 5),
        warmup: env_usize("LUMEN_DECODE_BENCH_WARMUP", 1),
        cell: std::env::var("LUMEN_DECODE_BENCH_CELL").unwrap_or_else(|_| "unknown".into()),
        json: std::env::var("LUMEN_DECODE_BENCH_JSON").ok(),
    })
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Run the fixed-decode battery. Returns the JSON string.
///
/// `sampling` MUST already be the greedy penalty-off configuration the caller
/// passed on the command line; the routing assertion below fails the run if it
/// is not, because a penalty would route decode off the shipping GPU-argmax
/// path and measure different kernels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cfg: &BenchCfg,
    engine: &InferenceEngine,
    weights: &dyn WeightProvider,
    backend: &dyn ComputeBackend,
    prompt_tokens: &[u32],
    sampling: &SamplingParams,
    tokenizer: Option<&crate::tokenize::BpeTokenizer>,
    backend_name: &str,
    model_display: &str,
) -> String {
    // ---- route assertions: measure the shipping decode path or fail loudly ----
    let caps = backend.caps();
    let setters = lumen_runtime::runtime_defaults::model_setter_snapshot();
    let gpu_greedy = lumen_runtime::engine::use_gpu_greedy_predicate(
        sampling,
        caps.gpu_resident,
        caps.gpu_argmax,
    );
    let mut asserts: Vec<(String, String)> = vec![
        ("backend".into(), backend_name.to_string()),
        ("gpu_argmax_cap".into(), caps.gpu_argmax.to_string()),
        ("gpu_resident_cap".into(), caps.gpu_resident.to_string()),
        ("use_gpu_greedy".into(), gpu_greedy.to_string()),
        ("temperature".into(), format!("{:?}", sampling.temperature)),
        (
            "repetition_penalty".into(),
            format!("{:?}", sampling.repetition_penalty),
        ),
        ("repeat_last_n".into(), format!("{:?}", sampling.repeat_last_n)),
        ("kv_precision".into(), format!("{:?}", engine.config().kv_precision)),
        ("max_seq_len".into(), engine.config().max_seq_len.to_string()),
        ("num_layers".into(), engine.hyperparams().num_layers.to_string()),
        (
            "num_experts".into(),
            engine.hyperparams().num_experts.unwrap_or(0).to_string(),
        ),
        // The FOUR model-aware setters, read back from the runtime registry the
        // backend actually consulted. Recording the *resolved* values (not just
        // trusting that the CLI called the setters) is what makes the
        // missing-setter artifact — which once understated a 9-cell board by up
        // to 55% — impossible to reproduce unnoticed.
        ("setter_dense_quant_hint".into(), setters.0.to_string()),
        ("setter_model_primary_quant".into(), format!("{:?}", setters.1)),
        ("setter_model_block_count".into(), setters.2.to_string()),
        ("setter_model_is_moe".into(), setters.3.to_string()),
    ];

    // hard-check the resolved setter state: unset hints mean LEGACY kernel
    // dispatch, which is exactly the artifact this board exists to avoid.
    if setters.0 == "unset" || setters.1.is_none() || setters.2 == 0 {
        eprintln!(
            "[decode-bench] FATAL: model-aware setters not resolved \
             (dense_hint={} primary_quant={:?} block_count={} is_moe={}). \
             Unset hints route to LEGACY kernels — refusing to measure.",
            setters.0, setters.1, setters.2, setters.3
        );
        std::process::exit(26);
    }
    if setters.3 != (engine.hyperparams().num_experts.unwrap_or(0) > 0) {
        eprintln!(
            "[decode-bench] FATAL: is_moe setter ({}) disagrees with the model \
             ({} experts) — the backend is configured for a different model class.",
            setters.3,
            engine.hyperparams().num_experts.unwrap_or(0)
        );
        std::process::exit(27);
    }
    if !gpu_greedy {
        eprintln!(
            "[decode-bench] FATAL: decode would NOT take the shipping GPU-argmax greedy path \
             (gpu_argmax={} gpu_resident={} temp={:?} rep_penalty={:?}). Refusing to publish a \
             number measured on different kernels.",
            caps.gpu_argmax, caps.gpu_resident, sampling.temperature, sampling.repetition_penalty
        );
        std::process::exit(21);
    }

    // record the LUMEN_* environment actually in force (including notable unsets)
    let mut envs: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("LUMEN_"))
        .collect();
    envs.sort();
    for k in ["LUMEN_METAL_MMAP_ONLY", "LUMEN_CUDA_PTX_CACHE"] {
        if std::env::var(k).is_err() {
            envs.push((k.to_string(), "<unset>".into()));
        }
    }

    // EOS is deliberately absent: a pure MaxTokens stop cannot be shortened.
    let stop = StopCondition::MaxTokens(cfg.steps + 1);

    let mut samples: Vec<f64> = Vec::with_capacity(cfg.runs);
    let mut durations_ns: Vec<u128> = Vec::with_capacity(cfg.runs);
    let mut step_counts: Vec<usize> = Vec::with_capacity(cfg.runs);
    let mut first_text = String::new();
    let mut first_ids: Vec<u32> = Vec::new();
    // Determinism check: under greedy decoding with a fresh KV/recurrent state,
    // every sequence must emit the SAME token IDs. A mismatch means state leaked
    // between sequences (or the run is non-deterministic) — either way the
    // battery is not measuring what it claims and must not be published.
    let mut token_hashes: Vec<u64> = Vec::new();
    let hash_tokens = |ids: &[u32]| -> u64 {
        // FNV-1a over the token id stream
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in ids {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    };

    let total = cfg.warmup + cfg.runs;
    for i in 0..total {
        let measured = i >= cfg.warmup;
        let t_seq = Instant::now();
        let result = match engine.generate(prompt_tokens, weights, backend, &stop, sampling) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[decode-bench] sequence {i} failed: {e}");
                std::process::exit(22);
            }
        };
        // tokens = 1 (from prefill logits) + N decode calls
        let decode_calls = result.tokens.len().saturating_sub(1);
        // `decode_loop_time` covers EXACTLY the N backend decode calls. The
        // legacy `decode_time` also contains the first CPU-side full-vocabulary
        // sample off the prefill logits — a cost the shipping GPU-argmax path
        // never pays per token and that `llama-bench tgN` never pays at all, so
        // including it would not be the same unit as the comparator.
        let secs = result.metrics.decode_loop_time.as_secs_f64();
        if secs <= 0.0 {
            eprintln!(
                "[decode-bench] FATAL: decode_loop_time is zero — this generation path \
                 does not instrument the decode loop; refusing to publish a number."
            );
            std::process::exit(24);
        }
        let tps = if secs > 0.0 { decode_calls as f64 / secs } else { 0.0 };
        eprintln!(
            "[decode-bench] {} seq{} {}: decode_calls={} decode_time={:.4}s tps={:.3} (seq wall {:.1}s)",
            cfg.cell,
            i,
            if measured { "MEASURED" } else { "warmup " },
            decode_calls,
            secs,
            tps,
            t_seq.elapsed().as_secs_f64()
        );
        if decode_calls != cfg.steps {
            eprintln!(
                "[decode-bench] FATAL: expected exactly {} decode calls, got {} \
                 (tokens={}). A short sequence is not a datum.",
                cfg.steps,
                decode_calls,
                result.tokens.len()
            );
            std::process::exit(23);
        }
        if measured {
            samples.push(tps);
            durations_ns.push(result.metrics.decode_loop_time.as_nanos());
            step_counts.push(decode_calls);
            token_hashes.push(hash_tokens(&result.tokens));
            if samples.len() == 1 {
                first_ids = result.tokens.clone();
                if let Some(tok) = tokenizer {
                    first_text = tok.decode(&result.tokens);
                }
            }
        }
    }

    // ROUTE VERIFICATION: refuse to report timings for a branch that never ran.
    // Two earlier sweeps produced plausible numbers from unreachable code — a
    // string oracle that missed the GDN label spellings, and an F16 branch
    // placed after an early return. Timing is meaningless if the requested
    // representation was not the one dispatched.
    // Print the census ALWAYS, not only when a zone plan is being verified.
    // Rounds 24-25 measured the int8 Q4 paths (aligned dp4a, split mmvq) as
    // flat, and "flat" is indistinguishable from "never dispatched" without
    // this — the exact trap that made three earlier rounds worthless.
    {
        let census = lumen_runtime::runtime_defaults::route_census_summary();
        if census.is_empty() {
            eprintln!("[Q4ROUTE] (census empty — no Q4 dispatch recorded)");
        }
        for (fam, path, n) in census {
            eprintln!("[Q4ROUTE] {fam} -> {path} x{n}");
        }
    }
    if lumen_runtime::runtime_defaults::route_census_enabled() {
        let plan = lumen_runtime::runtime_defaults::q4_act_plan();
        match lumen_runtime::runtime_defaults::route_census_verify(plan) {
            Ok(m) => {
                eprintln!("[decode-bench] {m}");
                asserts.push(("route_census".into(), m));
            }
            Err(e) => {
                eprintln!("[decode-bench] FATAL: requested route was not taken.\n{e}");
                std::process::exit(31);
            }
        }
    }

    // Every measured sequence must be token-identical (greedy + fresh state).
    //
    // EXCEPT under LUMEN_DECODE_ABLATE: an ablation deliberately skips a phase,
    // so a skipped state update legitimately makes sequences differ. The assert
    // exists to catch state LEAKAGE in real arms; applying it to a
    // timing-only ablation just refuses the measurement (rc=25) and leaves the
    // phase unsplit — which is what happened to the gdn_recur and head arms in
    // rounds 29b and 30. Ablation output is garbage by construction and is
    // never admissible as a result; only its wall time is read.
    let ablating = std::env::var("LUMEN_DECODE_ABLATE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    if ablating && token_hashes.windows(2).any(|w| w[0] != w[1]) {
        eprintln!(
            "[decode-bench] NOTE: sequences differ under LUMEN_DECODE_ABLATE \
             (expected — a skipped phase changes output); timing still valid."
        );
    }
    if !ablating && token_hashes.windows(2).any(|w| w[0] != w[1]) {
        eprintln!(
            "[decode-bench] FATAL: measured sequences are NOT token-identical \
             (hashes {token_hashes:?}). Greedy decode from a reset state must be \
             deterministic; differing output means state leaked across sequences."
        );
        std::process::exit(25);
    }

    // ---- statistics: mean headline, sample SD (n-1), CV vs mean ----
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let sd = if n > 1 {
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let mut srt = samples.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 1 {
        srt[n / 2]
    } else {
        (srt[n / 2 - 1] + srt[n / 2]) / 2.0
    };

    let json = format!(
        concat!(
            "{{\n",
            " \"engine\": \"lumen\",\n",
            " \"harness\": \"decode_bench (fixed-N decode, EOS-proof, shipping setup path)\",\n",
            " \"cell\": \"{cell}\",\n",
            " \"model\": \"{model}\",\n",
            " \"backend\": \"{backend}\",\n",
            " \"lumen_version\": \"{ver}\",\n",
            " \"decode_steps\": {steps},\n",
            " \"warmup\": {warmup},\n",
            " \"runs\": {runs},\n",
            " \"samples\": [{samples}],\n",
            " \"decode_ns\": [{durns}],\n",
            " \"decode_calls_per_seq\": [{steplist}],\n",
            " \"mean\": {mean:.4},\n",
            " \"median\": {median:.4},\n",
            " \"min\": {min:.4},\n",
            " \"max\": {max:.4},\n",
            " \"sd_sample\": {sd:.4},\n",
            " \"cv_pct\": {cv:.4},\n",
            " \"prompt_token_count\": {ptc},\n",
            " \"prompt_token_ids\": [{ptids}],\n",
            " \"route_asserts\": {{{asserts}}},\n",
            " \"lumen_env\": {{{envs}}},\n",
            " \"generated_token_ids_run1\": [{gids}],\n",
            " \"text_run1\": \"{text}\"\n",
            "}}"
        ),
        cell = json_escape(&cfg.cell),
        model = json_escape(model_display),
        backend = json_escape(backend_name),
        ver = option_env!("LUMEN_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION")),
        steps = cfg.steps,
        warmup = cfg.warmup,
        runs = cfg.runs,
        samples = samples
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(", "),
        durns = durations_ns
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        steplist = step_counts
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        mean = mean,
        median = median,
        min = srt[0],
        max = srt[n - 1],
        sd = sd,
        cv = if mean > 0.0 { 100.0 * sd / mean } else { 0.0 },
        ptc = prompt_tokens.len(),
        ptids = prompt_tokens
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        asserts = asserts
            .drain(..)
            .map(|(k, v)| format!("\"{}\": \"{}\"", json_escape(&k), json_escape(&v)))
            .collect::<Vec<_>>()
            .join(", "),
        envs = envs
            .iter()
            .map(|(k, v)| format!("\"{}\": \"{}\"", json_escape(k), json_escape(v)))
            .collect::<Vec<_>>()
            .join(", "),
        gids = first_ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        text = json_escape(&first_text),
    );

    // Transactional publish: write to a sibling temp file, fsync-by-close, then
    // atomically rename. A benchmark that "succeeds" while failing to persist
    // its evidence is worse than one that fails loudly, so any error here is
    // fatal rather than a warning.
    if let Some(path) = &cfg.json {
        let tmp = format!("{path}.tmp.{}", std::process::id());
        if let Err(e) = std::fs::write(&tmp, &json) {
            eprintln!("[decode-bench] FATAL: could not write {tmp}: {e}");
            std::process::exit(28);
        }
        match std::fs::read_to_string(&tmp) {
            Ok(back) if back.len() == json.len() => {}
            Ok(_) => {
                eprintln!("[decode-bench] FATAL: readback of {tmp} does not match");
                std::process::exit(28);
            }
            Err(e) => {
                eprintln!("[decode-bench] FATAL: could not read back {tmp}: {e}");
                std::process::exit(28);
            }
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            eprintln!("[decode-bench] FATAL: could not publish {path}: {e}");
            std::process::exit(28);
        }
    }
    println!("{json}");
    json
}
