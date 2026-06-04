#!/usr/bin/env python3
"""Native bench harness for the CUDA MoE + dense validation on a remote A100 box.

Validates the CUDA MoE runtime path (per-expert and batched dispatch)
on Qwen3.5-35B-A3B + Qwen3.5-9B without Modal.

Run on the remote A100 box only.  Expects:
  - `<LUMEN_CACHE_ROOT>/lbc/qwen_qwen3.5-35b-a3b-q8_0.lbc`
  - `<LUMEN_CACHE_ROOT>/lbc/qwen_qwen3.5-9b-q8_0.lbc`
  - Lumen workspace at `~/lumen/` (or `LUMEN_ROOT` override)
  - cargo + nvcc on PATH (or via rustup env / /usr/local/cuda)

It generates `crates/lumen-runtime/tests/cuda_moe_validate_test.rs`,
runs the four bench cells (A1, A2, D.1, D.2, E), parses
`NATIVE-BENCH-*-METRIC` lines, persists results to
`<LUMEN_CACHE_ROOT>/artifacts/native-bench-results.json`, prints a
summary table, and cleans up the auto-gen test file.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LUMEN_ROOT = Path(os.environ.get("LUMEN_ROOT", str(Path.home() / "lumen")))
CACHE_ROOT = Path(os.environ.get("LUMEN_CACHE_ROOT", "/mnt/nvme0/lumen-cache"))

MOE_LBC_PATH = str(CACHE_ROOT / "lbc" / "qwen_qwen3.5-35b-a3b-q8_0.lbc")
DENSE_LBC_PATH = str(CACHE_ROOT / "lbc" / "qwen_qwen3.5-9b-q8_0.lbc")

# MoE equivalence check: 10 prompts x 32 tokens per arm.
CELL_A_PROMPTS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    [100, 200, 300, 400, 500, 600, 700, 800],
    [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    [5000, 5050, 5100, 5150, 5200, 5250, 5300, 5350],
    [6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000],
    [1, 100, 1000, 10000, 50000, 100000, 5000, 500],
    [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
]
CELL_A_DECODE_TOKENS = 32
CELL_A_MATCH_THRESHOLD = 30  # >=30/32 tokens must match per prompt

# MoE / dense decode acceptance windows.
CELL_D1_BW_FLOOR_TPS = 340.0
CELL_D1_TOLERANCE_PCT = 15.0
CELL_D2_BASELINE_TPS = 52.572   # dense Q8 baseline at seq_len=4096
CELL_D2_TOLERANCE_PCT = 2.0

# Long-context seq_lens.
CELL_E_SEQ_LENS = [4096, 16384, 32768]
CELL_E_DECODE_TOKENS = 16

# Bench-trial config.
BENCH_WARMUP_TOKENS = 3
BENCH_MEASURE_TRIALS = 5

# Timeouts.
PER_CELL_TIMEOUT_SEC = 1800
BUILD_TIMEOUT_SEC = 1800

# Output paths.
RUN_ID = "native-bench"
DEFAULT_RESULTS_PATH = CACHE_ROOT / "artifacts" / f"{RUN_ID}-results.json"
DEFAULT_LOG_PATH = CACHE_ROOT / "logs" / f"{RUN_ID}.log"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _build_env() -> dict:
    """Return a subprocess env with rustup + CUDA on PATH."""
    env = os.environ.copy()
    extra_paths = [
        str(Path.home() / ".cargo" / "bin"),
        "/usr/local/cuda/bin",
    ]
    cur_path = env.get("PATH", "")
    parts = cur_path.split(":") if cur_path else []
    for p in extra_paths:
        if p and p not in parts:
            parts.insert(0, p)
    env["PATH"] = ":".join(parts)
    return env


def _gpu_info() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _gpu_memory_used_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return -1.0


# ---------------------------------------------------------------------------
# Test-file generation
# ---------------------------------------------------------------------------

def _gen_validation_test_file(
    moe_lbc_path: str,
    dense_lbc_path: str,
) -> str:
    """Generate the cells-A/D/E Rust test file."""
    prompts_rust = ",\n            ".join(
        "vec![" + ", ".join(f"{t}u32" for t in prompt) + "]"
        for prompt in CELL_A_PROMPTS
    )
    cell_e_seq_lens = ", ".join(str(s) for s in CELL_E_SEQ_LENS)

    test_content = textwrap.dedent(f"""\
        //! Native MoE validation harness (AUTO-GENERATED -- DO NOT COMMIT).
        //!
        //! Written by `scripts/native_bench.py` per bench run.
        //! Deleted before the run completes.
        //!
        //! Cells:
        //!   - bench_cell_a_correctness  (Path A1 = A2 equivalence)
        //!   - bench_cell_d1_moe_decode_35b   (Qwen3.5-35B-A3B Q8 decode TPS)
        //!   - bench_cell_d2_dense_decode_9b  (Qwen3.5-9B dense Q8 decode TPS)
        //!   - bench_cell_e_long_context      (MoE long-context token streams)

        #![cfg(feature = "cuda")]

        use cudarc::driver::CudaContext;
        use lumen_format::quantization::QuantScheme;
        use lumen_runtime::compute::ComputeBackend;
        use lumen_runtime::cuda::CudaBackend;
        use lumen_runtime::kv::{{KvCache, KvCacheConfig, KvPrecision}};
        use lumen_runtime::weight::provider_sync::SyncWeightProvider;
        use std::time::Instant;

        const MOE_LBC_PATH: &str = "{moe_lbc_path}";
        const DENSE_LBC_PATH: &str = "{dense_lbc_path}";
        const CELL_A_DECODE_TOKENS: usize = {CELL_A_DECODE_TOKENS};
        const CELL_E_DECODE_TOKENS: usize = {CELL_E_DECODE_TOKENS};
        const BENCH_WARMUP_TOKENS: usize = {BENCH_WARMUP_TOKENS};
        const BENCH_MEASURE_TRIALS: usize = {BENCH_MEASURE_TRIALS};

        fn setup_real_model_backend(
            lbc_path: &str,
        ) -> Result<(SyncWeightProvider, CudaBackend), String> {{
            let path = std::path::Path::new(lbc_path);
            if !path.exists() {{
                return Err(format!("LBC not found: {{lbc_path}}"));
            }}
            let _ctx = CudaContext::new(0).map_err(|e| format!("cuInit: {{e}}"))?;
            let provider = SyncWeightProvider::open(path)
                .map_err(|e| format!("open LBC: {{e}}"))?;
            let hp = provider.lbc().header.hyperparams;
            eprintln!(
                "  Model: {{}} layers, hidden={{}}, heads={{}}, kv_heads={{}}, head_dim={{}}, max_seq={{}}, num_experts={{:?}}",
                hp.num_layers, hp.hidden_dim, hp.num_heads, hp.num_kv_heads,
                hp.head_dim, hp.max_seq_len, hp.num_experts,
            );
            let mut cuda = CudaBackend::new(0).map_err(|e| format!("CudaBackend::new: {{e}}"))?;
            cuda.set_global_tensors(
                provider.embedding.clone(),
                provider.final_norm.clone(),
                provider.output_proj.clone(),
            );
            if matches!(provider.output_proj_quant,
                        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                && !provider.output_proj_raw.is_empty()
            {{
                cuda.set_output_proj_raw(
                    provider.output_proj_raw.clone(),
                    provider.output_proj_quant,
                );
            }}
            if matches!(provider.embedding_quant,
                        QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                && !provider.embedding_raw.is_empty()
            {{
                cuda.set_embedding_raw(
                    provider.embedding_raw.clone(),
                    provider.embedding_quant,
                );
            }}
            if provider.weight_tying {{
                cuda.set_weight_tying(true);
            }}
            cuda.init(&hp).map_err(|e| format!("init: {{e}}"))?;
            Ok((provider, cuda))
        }}

        fn alloc_kv(hp: &lumen_format::ModelHyperparams, max_seq: usize)
            -> Result<KvCache, String>
        {{
            let kv_config = KvCacheConfig {{
                max_seq_len: max_seq,
                num_layers: hp.num_layers as usize,
                num_kv_heads: hp.num_kv_heads as usize,
                head_dim: hp.head_dim as usize,
                precision: KvPrecision::F32,
            }};
            KvCache::new(kv_config).map_err(|e| format!("KvCache::new: {{e}}"))
        }}

        fn prefill_then_decode_n(
            backend: &mut CudaBackend,
            provider: &SyncWeightProvider,
            kv: &mut KvCache,
            prompt: &[u32],
            n_decode: usize,
        ) -> Result<Vec<u32>, String> {{
            backend.prefill(prompt, provider, kv).map_err(|e| format!("prefill: {{e}}"))?;
            let mut decoded: Vec<u32> = Vec::with_capacity(n_decode);
            let mut next_token: u32 = *prompt.last().unwrap_or(&1u32);
            for _ in 0..n_decode {{
                let logits = backend.decode_token(next_token, provider, kv)
                    .map_err(|e| format!("decode_token: {{e}}"))?;
                next_token = logits.argmax() as u32;
                decoded.push(next_token);
            }}
            Ok(decoded)
        }}

        // -------------------------------------------------------------------
        // MoE equivalence -- Path A1 (per-expert) vs Path A2 (batched)
        // -------------------------------------------------------------------

        #[test]
        #[ignore]
        fn bench_cell_a_correctness() {{
            let prompts: Vec<Vec<u32>> = vec![
                {prompts_rust}
            ];

            let arm = std::env::var("LUMEN_CUDA_MOE_BATCHED").unwrap_or_default();
            let arm_label = match arm.as_str() {{
                "1" | "true" | "yes" => "A2",
                _ => "A1",
            }};
            eprintln!("=== MoE equivalence arm={{arm_label}} (LUMEN_CUDA_MOE_BATCHED={{arm}}) ===");

            let (provider, mut backend) = match setup_real_model_backend(MOE_LBC_PATH) {{
                Ok(v) => v,
                Err(e) => {{ eprintln!("  SKIP_SETUP: {{e}}"); return; }}
            }};
            let hp = provider.lbc().header.hyperparams;

            eprintln!("  Preloading weights...");
            let t_pre = Instant::now();
            if let Err(e) = backend.preload_weights(&provider) {{
                eprintln!("  PRELOAD_FAILED: {{e}}");
                return;
            }}
            eprintln!("  preload: OK ({{:.1}} ms)", t_pre.elapsed().as_secs_f64() * 1000.0);

            let caps = backend.caps();
            eprintln!("  caps().moe = {{}}", caps.moe);
            eprintln!("  NATIVE-BENCH-A-METRIC caps_moe={{}}", caps.moe);

            for (pi, prompt) in prompts.iter().enumerate() {{
                let max_seq = prompt.len() + CELL_A_DECODE_TOKENS + 16;
                let mut kv = match alloc_kv(&hp, max_seq) {{
                    Ok(k) => k,
                    Err(e) => {{ eprintln!("  KV_ALLOC_FAILED prompt={{pi}}: {{e}}"); continue; }}
                }};
                let result = prefill_then_decode_n(
                    &mut backend, &provider, &mut kv, prompt, CELL_A_DECODE_TOKENS,
                );
                match result {{
                    Ok(decoded) => {{
                        let toks_json: Vec<String> = decoded.iter()
                            .map(|t| t.to_string()).collect();
                        eprintln!("  NATIVE-BENCH-A-METRIC arm={{arm_label}} prompt={{pi}} tokens=[{{}}]",
                                  toks_json.join(","));
                    }}
                    Err(e) => {{
                        eprintln!("  NATIVE-BENCH-A-METRIC arm={{}} prompt={{}} FAIL={{:?}}",
                                  arm_label, pi, e);
                    }}
                }}
            }}

            eprintln!("=== MoE equivalence arm={{arm_label}} END ===");
        }}

        // -------------------------------------------------------------------
        // MoE decode bench (Qwen3.5-35B-A3B Q8_0)
        // -------------------------------------------------------------------

        #[test]
        #[ignore]
        fn bench_cell_d1_moe_decode_35b() {{
            eprintln!("=== MoE 35B-A3B Q8 decode bench ===");
            let (provider, mut backend) = match setup_real_model_backend(MOE_LBC_PATH) {{
                Ok(v) => v,
                Err(e) => {{ eprintln!("  SKIP_SETUP: {{e}}"); return; }}
            }};
            let hp = provider.lbc().header.hyperparams;
            eprintln!("  Preloading weights...");
            let t_pre = Instant::now();
            if let Err(e) = backend.preload_weights(&provider) {{
                eprintln!("  PRELOAD_FAILED: {{e}}");
                return;
            }}
            eprintln!("  preload: OK ({{:.1}} ms)", t_pre.elapsed().as_secs_f64() * 1000.0);

            let prompt: Vec<u32> = (1u32..=16u32).collect();
            let max_seq = 64;
            let mut kv = match alloc_kv(&hp, max_seq) {{
                Ok(k) => k,
                Err(e) => {{ eprintln!("  KV_ALLOC_FAILED: {{e}}"); return; }}
            }};

            if let Err(e) = backend.prefill(&prompt, &provider, &mut kv) {{
                eprintln!("  PREFILL_FAILED: {{e}}"); return;
            }}

            let mut tok: u32 = 1;
            for _ in 0..BENCH_WARMUP_TOKENS {{
                let logits = match backend.decode_token(tok, &provider, &mut kv) {{
                    Ok(l) => l,
                    Err(e) => {{ eprintln!("  WARMUP_FAILED: {{e}}"); return; }}
                }};
                tok = logits.argmax() as u32;
            }}

            let mut us_per_tok: Vec<f64> = Vec::with_capacity(BENCH_MEASURE_TRIALS);
            for _ in 0..BENCH_MEASURE_TRIALS {{
                let t = Instant::now();
                let logits = match backend.decode_token(tok, &provider, &mut kv) {{
                    Ok(l) => l,
                    Err(e) => {{ eprintln!("  DECODE_FAILED: {{e}}"); return; }}
                }};
                us_per_tok.push(t.elapsed().as_secs_f64() * 1_000_000.0);
                tok = logits.argmax() as u32;
            }}
            us_per_tok.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = us_per_tok.len();
            let median_us = if n % 2 == 0 {{
                (us_per_tok[n / 2 - 1] + us_per_tok[n / 2]) / 2.0
            }} else {{ us_per_tok[n / 2] }};
            let tps = 1_000_000.0 / median_us;
            eprintln!("  NATIVE-BENCH-D-METRIC cell=D1 decode_us_median={{:.2}} decode_tps_median={{:.4}} trials={{}}",
                      median_us, tps, n);
        }}

        // -------------------------------------------------------------------
        // Dense decode bench (Qwen3.5-9B Q8_0)
        // -------------------------------------------------------------------

        #[test]
        #[ignore]
        fn bench_cell_d2_dense_decode_9b() {{
            eprintln!("=== Qwen3.5-9B dense Q8 decode bench ===");
            let (provider, mut backend) = match setup_real_model_backend(DENSE_LBC_PATH) {{
                Ok(v) => v,
                Err(e) => {{ eprintln!("  SKIP_SETUP: {{e}}"); return; }}
            }};
            let hp = provider.lbc().header.hyperparams;
            eprintln!("  Preloading weights...");
            let t_pre = Instant::now();
            if let Err(e) = backend.preload_weights(&provider) {{
                eprintln!("  PRELOAD_FAILED: {{e}}");
                return;
            }}
            eprintln!("  preload: OK ({{:.1}} ms)", t_pre.elapsed().as_secs_f64() * 1000.0);

            let caps = backend.caps();
            eprintln!("  caps().moe (dense) = {{}}", caps.moe);
            eprintln!("  NATIVE-BENCH-D-METRIC cell=D2 caps_moe_dense={{}}", caps.moe);

            let prefill_len = 4095usize;
            let prompt: Vec<u32> = (0..prefill_len).map(|i| ((i % 50) as u32) + 1u32).collect();
            let max_seq = prefill_len + BENCH_WARMUP_TOKENS + BENCH_MEASURE_TRIALS + 16;
            let mut kv = match alloc_kv(&hp, max_seq) {{
                Ok(k) => k,
                Err(e) => {{ eprintln!("  KV_ALLOC_FAILED: {{e}}"); return; }}
            }};

            if let Err(e) = backend.prefill(&prompt, &provider, &mut kv) {{
                eprintln!("  PREFILL_FAILED: {{e}}"); return;
            }}

            let mut tok: u32 = 1;
            for _ in 0..BENCH_WARMUP_TOKENS {{
                let logits = match backend.decode_token(tok, &provider, &mut kv) {{
                    Ok(l) => l,
                    Err(e) => {{ eprintln!("  WARMUP_FAILED: {{e}}"); return; }}
                }};
                tok = logits.argmax() as u32;
            }}

            let mut us_per_tok: Vec<f64> = Vec::with_capacity(BENCH_MEASURE_TRIALS);
            for _ in 0..BENCH_MEASURE_TRIALS {{
                let t = Instant::now();
                let logits = match backend.decode_token(tok, &provider, &mut kv) {{
                    Ok(l) => l,
                    Err(e) => {{ eprintln!("  DECODE_FAILED: {{e}}"); return; }}
                }};
                us_per_tok.push(t.elapsed().as_secs_f64() * 1_000_000.0);
                tok = logits.argmax() as u32;
            }}
            us_per_tok.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = us_per_tok.len();
            let median_us = if n % 2 == 0 {{
                (us_per_tok[n / 2 - 1] + us_per_tok[n / 2]) / 2.0
            }} else {{ us_per_tok[n / 2] }};
            let tps = 1_000_000.0 / median_us;
            eprintln!("  NATIVE-BENCH-D-METRIC cell=D2 decode_us_median={{:.2}} decode_tps_median={{:.4}} trials={{}}",
                      median_us, tps, n);
        }}

        // -------------------------------------------------------------------
        // Long-context MoE decode (4K / 16K / 32K)
        // -------------------------------------------------------------------

        #[test]
        #[ignore]
        fn bench_cell_e_long_context() {{
            eprintln!("=== MoE 35B-A3B long-context decode ===");
            let (provider, mut backend) = match setup_real_model_backend(MOE_LBC_PATH) {{
                Ok(v) => v,
                Err(e) => {{ eprintln!("  SKIP_SETUP: {{e}}"); return; }}
            }};
            let hp = provider.lbc().header.hyperparams;
            eprintln!("  Preloading weights...");
            if let Err(e) = backend.preload_weights(&provider) {{
                eprintln!("  PRELOAD_FAILED: {{e}}"); return;
            }}

            let seq_lens: Vec<usize> = vec![{cell_e_seq_lens}];
            for sl in seq_lens {{
                eprintln!("  -- seq_len={{}} --", sl);
                let prefill_len = sl.saturating_sub(1);
                let prompt: Vec<u32> = (0..prefill_len)
                    .map(|i| ((i % 50) as u32) + 1u32).collect();
                let max_seq = sl + CELL_E_DECODE_TOKENS + 16;
                let mut kv = match alloc_kv(&hp, max_seq) {{
                    Ok(k) => k,
                    Err(e) => {{ eprintln!("    KV_ALLOC_FAILED sl={{}}: {{}}", sl, e); continue; }}
                }};
                if let Err(e) = backend.prefill(&prompt, &provider, &mut kv) {{
                    eprintln!("    PREFILL_FAILED sl={{}}: {{}}", sl, e); continue;
                }}
                let mut tok: u32 = 1;
                let mut decoded: Vec<u32> = Vec::with_capacity(CELL_E_DECODE_TOKENS);
                for _ in 0..CELL_E_DECODE_TOKENS {{
                    let logits = match backend.decode_token(tok, &provider, &mut kv) {{
                        Ok(l) => l,
                        Err(e) => {{ eprintln!("    DECODE_FAILED sl={{}}: {{}}", sl, e); break; }}
                    }};
                    tok = logits.argmax() as u32;
                    decoded.push(tok);
                }}
                let toks_json: Vec<String> = decoded.iter().map(|t| t.to_string()).collect();
                eprintln!("  NATIVE-BENCH-E-METRIC seq_len={{}} tokens=[{{}}]",
                          sl, toks_json.join(","));
            }}
        }}
    """)
    return test_content


# ---------------------------------------------------------------------------
# Cargo execution
# ---------------------------------------------------------------------------

def _cargo_build_tests(env: dict) -> tuple[bool, str]:
    """`cargo test --no-run` so the bench cells share one compiled binary."""
    print("\n=== Building lumen-runtime tests (CUDA, release) ===", flush=True)
    cmd = (
        "cargo test --release -p lumen-runtime --features cuda "
        "--test cuda_moe_validate_test --no-run"
    )
    print(f"  cmd: {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(
        cmd, shell=True, cwd=str(LUMEN_ROOT),
        env=env, capture_output=True, text=True,
        timeout=BUILD_TIMEOUT_SEC,
    )
    wall = round(time.time() - t, 1)
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-3500:]
        print(f"BUILD FAILED ({wall}s):\n{tail}", flush=True)
        return False, tail
    print(f"  build: OK ({wall}s)", flush=True)
    return True, ""


def _run_cargo_test(test_fn: str, arm_env: dict, timeout_sec: int = PER_CELL_TIMEOUT_SEC) -> dict:
    """Run one bench cell."""
    cmd = (
        "cargo test --release -p lumen-runtime --features cuda "
        f"--test cuda_moe_validate_test -- --nocapture --ignored {test_fn}"
    )
    print(f"\n--- Running: {test_fn} ---", flush=True)
    print(f"  cmd: {cmd}", flush=True)
    env_subset = {k: v for k, v in arm_env.items()
                  if k.startswith("LUMEN_") or k.startswith("RUST_")}
    print(f"  env (LUMEN_/RUST_): {env_subset}", flush=True)
    print(f"  GPU MB used before: {_gpu_memory_used_mb():.0f}", flush=True)
    t_start = time.time()
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=str(LUMEN_ROOT),
            env=arm_env, capture_output=True, text=True,
            timeout=timeout_sec,
        )
        full_out = (r.stdout or "") + "\n" + (r.stderr or "")
        print(full_out[-4500:], flush=True)
        wall = round(time.time() - t_start, 2)
        print(f"  GPU MB used after: {_gpu_memory_used_mb():.0f}", flush=True)
        return {
            "test_fn": test_fn,
            "exit_code": r.returncode,
            "wall_time_sec": wall,
            "stdout_tail": full_out[-8000:],
            "stdout_full": full_out,
        }
    except subprocess.TimeoutExpired:
        return {
            "test_fn": test_fn,
            "exit_code": -1,
            "wall_time_sec": round(time.time() - t_start, 2),
            "error": f"timeout after {timeout_sec}s",
            "stdout_tail": "",
            "stdout_full": "",
        }
    except Exception as e:
        return {
            "test_fn": test_fn,
            "exit_code": -2,
            "wall_time_sec": round(time.time() - t_start, 2),
            "error": f"{type(e).__name__}: {e}",
            "stdout_tail": "",
            "stdout_full": "",
        }


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def _parse_cell_a_metrics(stdout: str) -> dict:
    out: dict = {}
    for raw in stdout.split("\n"):
        line = raw.strip()
        if "NATIVE-BENCH-A-METRIC" not in line or "tokens=" not in line:
            continue
        idx = line.find("NATIVE-BENCH-A-METRIC ")
        rest = line[idx + len("NATIVE-BENCH-A-METRIC "):]
        kvs: dict = {}
        for tok in rest.split(" "):
            if "=" in tok:
                k, _, v = tok.partition("=")
                kvs[k] = v
        arm = kvs.get("arm")
        try:
            prompt_i = int(kvs.get("prompt", "-1"))
        except ValueError:
            continue
        toks_raw = kvs.get("tokens", "")
        if toks_raw.startswith("[") and toks_raw.endswith("]"):
            inner = toks_raw[1:-1]
            try:
                toks = [int(x) for x in inner.split(",") if x.strip()]
            except ValueError:
                continue
            out.setdefault(arm, {})[prompt_i] = toks
    return out


def _parse_cell_de_metrics(stdout: str) -> dict:
    out: dict = {"D": [], "E": [], "caps": {}}
    for raw in stdout.split("\n"):
        line = raw.strip()
        if "NATIVE-BENCH-D-METRIC" in line:
            idx = line.find("NATIVE-BENCH-D-METRIC ")
            rest = line[idx + len("NATIVE-BENCH-D-METRIC "):]
            kvs: dict = {}
            for tok in rest.split(" "):
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kvs[k] = v
            for k in ("decode_us_median", "decode_tps_median"):
                if k in kvs:
                    try:
                        kvs[k] = float(kvs[k])
                    except ValueError:
                        pass
            for k in ("trials",):
                if k in kvs:
                    try:
                        kvs[k] = int(kvs[k])
                    except ValueError:
                        pass
            out["D"].append(kvs)
        elif "NATIVE-BENCH-E-METRIC" in line:
            idx = line.find("NATIVE-BENCH-E-METRIC ")
            rest = line[idx + len("NATIVE-BENCH-E-METRIC "):]
            kvs: dict = {}
            for tok in rest.split(" "):
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kvs[k] = v
            if "seq_len" in kvs:
                try:
                    kvs["seq_len"] = int(kvs["seq_len"])
                except ValueError:
                    pass
            if "tokens" in kvs:
                tr = kvs["tokens"]
                if tr.startswith("[") and tr.endswith("]"):
                    try:
                        kvs["tokens"] = [int(x) for x in tr[1:-1].split(",") if x.strip()]
                    except ValueError:
                        pass
            out["E"].append(kvs)
        elif "NATIVE-BENCH-A-METRIC caps_moe=" in line:
            try:
                v = line.split("caps_moe=", 1)[1].strip()
                out["caps"]["moe_lbc"] = (v.lower() == "true")
            except Exception:
                pass
    return out


def _compute_cell_a_verdict(metrics_a1: dict, metrics_a2: dict) -> dict:
    per_prompt = []
    n_pass = 0
    n_total = 0
    for prompt_i in sorted(set(metrics_a1.keys()) | set(metrics_a2.keys())):
        a1 = metrics_a1.get(prompt_i)
        a2 = metrics_a2.get(prompt_i)
        rec: dict = {"prompt_idx": prompt_i,
                     "a1_len": len(a1) if a1 else 0,
                     "a2_len": len(a2) if a2 else 0}
        if not a1 or not a2:
            rec["status"] = "missing"
            per_prompt.append(rec)
            n_total += 1
            continue
        if len(a1) != len(a2):
            rec["status"] = "length_mismatch"
            per_prompt.append(rec)
            n_total += 1
            continue
        matches = sum(1 for x, y in zip(a1, a2) if x == y)
        rec["matches"] = matches
        rec["total"] = len(a1)
        rec["match_pct"] = 100.0 * matches / len(a1) if a1 else 0.0
        rec["status"] = "pass" if matches >= CELL_A_MATCH_THRESHOLD else "fail"
        rec["a1_tokens"] = a1
        rec["a2_tokens"] = a2
        per_prompt.append(rec)
        n_total += 1
        if rec["status"] == "pass":
            n_pass += 1
    return {
        "per_prompt": per_prompt,
        "n_prompts_total": n_total,
        "n_prompts_pass": n_pass,
        "threshold_per_prompt": f">={CELL_A_MATCH_THRESHOLD}/{CELL_A_DECODE_TOKENS} tokens",
        "overall_pass": n_pass == n_total and n_total >= 10,
    }


def _compute_cell_d_verdict(d_metrics: list) -> dict:
    by_cell = {m.get("cell"): m for m in d_metrics if isinstance(m, dict)}
    out: dict = {"by_cell": {}}

    d1 = by_cell.get("D1")
    if d1 and "decode_tps_median" in d1:
        tps = d1["decode_tps_median"]
        lo = CELL_D1_BW_FLOOR_TPS * (1.0 - CELL_D1_TOLERANCE_PCT / 100.0)
        hi = CELL_D1_BW_FLOOR_TPS * (1.0 + CELL_D1_TOLERANCE_PCT / 100.0)
        out["by_cell"]["D1"] = {
            "measured_tps": tps,
            "bw_floor_tps": CELL_D1_BW_FLOOR_TPS,
            "tolerance_pct": CELL_D1_TOLERANCE_PCT,
            "accept_range": [lo, hi],
            "status": "pass" if lo <= tps <= hi else (
                "above_target" if tps > hi else "below_target"
            ),
        }
    else:
        out["by_cell"]["D1"] = {"status": "missing"}

    d2 = by_cell.get("D2")
    if d2 and "decode_tps_median" in d2:
        tps = d2["decode_tps_median"]
        lo = CELL_D2_BASELINE_TPS * (1.0 - CELL_D2_TOLERANCE_PCT / 100.0)
        hi = CELL_D2_BASELINE_TPS * (1.0 + CELL_D2_TOLERANCE_PCT / 100.0)
        out["by_cell"]["D2"] = {
            "measured_tps": tps,
            "baseline_tps": CELL_D2_BASELINE_TPS,
            "tolerance_pct": CELL_D2_TOLERANCE_PCT,
            "accept_range": [lo, hi],
            "status": "pass" if lo <= tps <= hi else (
                "above_baseline" if tps > hi else "below_baseline"
            ),
        }
    else:
        out["by_cell"]["D2"] = {"status": "missing"}

    out["overall_pass"] = (
        out["by_cell"].get("D1", {}).get("status") == "pass"
        and out["by_cell"].get("D2", {}).get("status") in ("pass", "above_baseline")
    )
    return out


def _compute_cell_e_verdict(e_metrics: list) -> dict:
    by_sl: dict = {}
    for m in e_metrics:
        sl = m.get("seq_len")
        if sl is None:
            continue
        toks = m.get("tokens", [])
        by_sl[sl] = {
            "n_decoded": len(toks),
            "tokens": toks,
            "status": "pass" if len(toks) == CELL_E_DECODE_TOKENS else "fail",
        }
    overall = (
        all(v["status"] == "pass" for v in by_sl.values())
        and len(by_sl) == len(CELL_E_SEQ_LENS)
    )
    return {"by_seq_len": by_sl, "overall_pass": overall}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_results(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults persisted to {path}", flush=True)


def _print_summary_table(summary: dict) -> None:
    a = summary.get("cell_a_verdict", {})
    d = summary.get("cell_d_verdict", {}).get("by_cell", {})
    e = summary.get("cell_e_verdict", {}).get("by_seq_len", {})

    print("\n" + "=" * 70, flush=True)
    print("NATIVE BENCH SUMMARY TABLE", flush=True)
    print("=" * 70, flush=True)
    print(f"  Overall pass        : {summary.get('overall_pass')}", flush=True)
    print(f"  MoE equivalence     : {a.get('overall_pass')} "
          f"({a.get('n_prompts_pass', '?')}/{a.get('n_prompts_total', '?')} prompts)",
          flush=True)
    d1 = d.get("D1", {})
    d2 = d.get("D2", {})
    print(f"  MoE 35B-A3B decode  : status={d1.get('status')}  "
          f"tps={d1.get('measured_tps', '?')}  "
          f"target={CELL_D1_BW_FLOOR_TPS} +-{CELL_D1_TOLERANCE_PCT}%",
          flush=True)
    print(f"  Dense 9B decode     : status={d2.get('status')}  "
          f"tps={d2.get('measured_tps', '?')}  "
          f"baseline={CELL_D2_BASELINE_TPS} +-{CELL_D2_TOLERANCE_PCT}%",
          flush=True)
    for sl in sorted(e.keys()):
        rec = e[sl]
        print(f"  Long-ctx seq_len={sl:>6}: {rec.get('status')}  "
              f"decoded={rec.get('n_decoded')}/{CELL_E_DECODE_TOKENS}",
              flush=True)
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_PATH),
        help=f"Output JSON path (default: {DEFAULT_RESULTS_PATH})",
    )
    ap.add_argument(
        "--moe-lbc",
        default=MOE_LBC_PATH,
        help=f"Path to MoE LBC (default: {MOE_LBC_PATH})",
    )
    ap.add_argument(
        "--dense-lbc",
        default=DENSE_LBC_PATH,
        help=f"Path to dense LBC (default: {DENSE_LBC_PATH})",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `cargo test --no-run` (assume binary exists)",
    )
    ap.add_argument(
        "--cells",
        default="A,D1,D2,E",
        help="Comma-separated cells to run (subset of A,D1,D2,E)",
    )
    ap.add_argument(
        "--keep-test-file",
        action="store_true",
        help="Do not delete the auto-gen test file after the run",
    )
    args = ap.parse_args()

    cells_to_run = {c.strip().upper() for c in args.cells.split(",") if c.strip()}
    results_path = Path(args.results)

    print("=" * 70, flush=True)
    print("NATIVE BENCH (Qwen3.5-35B-A3B + 9B on remote A100)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Date (UTC) : {datetime.now(tz=timezone.utc).isoformat()}", flush=True)
    print(f"  GPU        : {_gpu_info()}", flush=True)
    print(f"  LUMEN_ROOT : {LUMEN_ROOT}", flush=True)
    print(f"  MoE LBC    : {args.moe_lbc}", flush=True)
    print(f"  Dense LBC  : {args.dense_lbc}", flush=True)
    print(f"  Results -> : {results_path}", flush=True)
    print(f"  Sections   : {sorted(cells_to_run)}", flush=True)

    if not LUMEN_ROOT.exists():
        print(f"\nERROR: LUMEN_ROOT not found: {LUMEN_ROOT}", flush=True)
        return 2
    if not Path(args.moe_lbc).exists():
        print(f"\nERROR: MoE LBC not found: {args.moe_lbc}", flush=True)
        return 2
    if not Path(args.dense_lbc).exists():
        print(f"  WARN: dense LBC missing -- D.2 will SKIP: {args.dense_lbc}",
              flush=True)

    env = _build_env()

    summary: dict = {
        "run_id": RUN_ID,
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "gpu": _gpu_info(),
        "moe_lbc_path": args.moe_lbc,
        "dense_lbc_path": args.dense_lbc,
        "cells_requested": sorted(cells_to_run),
        "cell_a_prompts": len(CELL_A_PROMPTS),
        "cell_a_decode_tokens": CELL_A_DECODE_TOKENS,
        "cell_a_match_threshold": CELL_A_MATCH_THRESHOLD,
        "cell_d1_bw_floor_tps": CELL_D1_BW_FLOOR_TPS,
        "cell_d1_tolerance_pct": CELL_D1_TOLERANCE_PCT,
        "cell_d2_baseline_tps": CELL_D2_BASELINE_TPS,
        "cell_d2_tolerance_pct": CELL_D2_TOLERANCE_PCT,
        "cell_e_seq_lens": CELL_E_SEQ_LENS,
        "cell_e_decode_tokens": CELL_E_DECODE_TOKENS,
    }

    # ---- Phase A: write auto-gen test file ----
    print("\n=== Generating MoE validation test file ===", flush=True)
    test_content = _gen_validation_test_file(
        moe_lbc_path=args.moe_lbc,
        dense_lbc_path=args.dense_lbc,
    )
    test_path = LUMEN_ROOT / "crates/lumen-runtime/tests/cuda_moe_validate_test.rs"
    test_path.write_text(test_content)
    print(f"  Wrote: {test_path} ({len(test_content)} bytes)", flush=True)
    summary["test_file_path"] = str(test_path)
    summary["test_file_size_bytes"] = len(test_content)

    # ---- Phase B: build tests once ----
    if args.skip_build:
        print("\n  --skip-build set; assuming compiled binary already exists", flush=True)
        summary["build_lumen_runtime_tests"] = {"status": "skipped"}
    else:
        ok, err_tail = _cargo_build_tests(env)
        if not ok:
            summary["build_lumen_runtime_tests"] = {
                "status": "fail",
                "tail": err_tail[-2000:],
            }
            summary["overall_pass"] = False
            try:
                if not args.keep_test_file:
                    test_path.unlink()
            except OSError:
                pass
            _persist_results(summary, results_path)
            _print_summary_table(summary)
            return 1
        summary["build_lumen_runtime_tests"] = {"status": "ok"}

    # ---- Phase C: run cells ----
    base_env = dict(env)
    base_env["LUMEN_CUDA_BF16_GEMMEX"] = "1"  # match runtime defaults
    base_env["LUMEN_CUDA_MAX_SEQ_LEN"] = str(CELL_E_SEQ_LENS[-1] + 64)

    # MoE equivalence arm A1 (per-expert).
    if "A" in cells_to_run:
        print("\n\n>>>>>> CELL A -- Path A1 (per-expert dispatch) <<<<<<", flush=True)
        env_a1 = dict(base_env)
        env_a1["LUMEN_CUDA_MOE_BATCHED"] = "0"
        res_a1 = _run_cargo_test("bench_cell_a_correctness", env_a1)
        summary["cell_a_arm1_raw"] = {k: v for k, v in res_a1.items()
                                       if k != "stdout_full"}

        print("\n\n>>>>>> CELL A -- Path A2 (batched dispatch) <<<<<<", flush=True)
        env_a2 = dict(base_env)
        env_a2["LUMEN_CUDA_MOE_BATCHED"] = "1"
        res_a2 = _run_cargo_test("bench_cell_a_correctness", env_a2)
        summary["cell_a_arm2_raw"] = {k: v for k, v in res_a2.items()
                                       if k != "stdout_full"}

        a1_m = _parse_cell_a_metrics(res_a1.get("stdout_full", ""))
        a2_m = _parse_cell_a_metrics(res_a2.get("stdout_full", ""))
        summary["cell_a_a1_per_prompt_tokens"] = a1_m
        summary["cell_a_a2_per_prompt_tokens"] = a2_m
        summary["cell_a_verdict"] = _compute_cell_a_verdict(
            a1_m.get("A1", {}), a2_m.get("A2", {})
        )

    # MoE 35B-A3B batched decode.
    d1_m = {"D": [], "E": [], "caps": {}}
    if "D1" in cells_to_run:
        print("\n\n>>>>>> CELL D.1 -- Qwen3.5-35B-A3B Q8 decode (batched MoE) <<<<<<",
              flush=True)
        env_d1 = dict(base_env)
        env_d1["LUMEN_CUDA_MOE_BATCHED"] = "1"
        res_d1 = _run_cargo_test("bench_cell_d1_moe_decode_35b", env_d1)
        summary["cell_d1_raw"] = {k: v for k, v in res_d1.items()
                                   if k != "stdout_full"}
        d1_m = _parse_cell_de_metrics(res_d1.get("stdout_full", ""))

    # Dense 9B decode regression.
    d2_m = {"D": [], "E": [], "caps": {}}
    if "D2" in cells_to_run and Path(args.dense_lbc).exists():
        print("\n\n>>>>>> CELL D.2 -- Qwen3.5-9B dense Q8 decode <<<<<<", flush=True)
        res_d2 = _run_cargo_test("bench_cell_d2_dense_decode_9b", dict(base_env))
        summary["cell_d2_raw"] = {k: v for k, v in res_d2.items()
                                   if k != "stdout_full"}
        d2_m = _parse_cell_de_metrics(res_d2.get("stdout_full", ""))
    elif "D2" in cells_to_run:
        summary["cell_d2_raw"] = {"status": "skipped_no_dense_lbc"}

    d_combined = d1_m["D"] + d2_m["D"]
    summary["cell_d_raw_metrics"] = d_combined
    summary["cell_d_verdict"] = _compute_cell_d_verdict(d_combined)

    # Long-context decode.
    if "E" in cells_to_run:
        print("\n\n>>>>>> CELL E -- long-context decode (4K/16K/32K, batched MoE) <<<<<<",
              flush=True)
        env_e = dict(base_env)
        env_e["LUMEN_CUDA_MOE_BATCHED"] = "1"
        res_e = _run_cargo_test("bench_cell_e_long_context", env_e)
        summary["cell_e_raw"] = {k: v for k, v in res_e.items()
                                  if k != "stdout_full"}
        e_m = _parse_cell_de_metrics(res_e.get("stdout_full", ""))
        summary["cell_e_raw_metrics"] = e_m["E"]
        summary["cell_e_verdict"] = _compute_cell_e_verdict(e_m["E"])

    # ---- Phase D: aggregate verdict ----
    cell_a_pass = summary.get("cell_a_verdict", {}).get("overall_pass", False) \
        if "A" in cells_to_run else True
    cell_d_pass = summary.get("cell_d_verdict", {}).get("overall_pass", False) \
        if ("D1" in cells_to_run or "D2" in cells_to_run) else True
    cell_e_pass = summary.get("cell_e_verdict", {}).get("overall_pass", False) \
        if "E" in cells_to_run else True

    summary["overall_pass"] = cell_a_pass and cell_d_pass and cell_e_pass
    summary["binding_gate_verdicts"] = {
        "G3_cell_a_token_stream_exact": cell_a_pass,
        "G5_cell_d_no_regression":       cell_d_pass,
        "G6_cell_e_long_context":        cell_e_pass,
    }
    summary["completed_utc"] = datetime.now(tz=timezone.utc).isoformat()

    # ---- Phase E: cleanup ----
    if not args.keep_test_file:
        try:
            test_path.unlink()
            print(f"\n  Removed auto-gen test file: {test_path}", flush=True)
        except OSError:
            pass

    _persist_results(summary, results_path)
    _print_summary_table(summary)

    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
