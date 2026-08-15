# Getting Started with Lumen

Lumen runs **LLM inference in Rust, for Apple Silicon and NVIDIA CUDA**. Built from scratch with zero ML dependencies — no PyTorch, no ONNX, no Python runtime. v1 (current) ships verified support for **Qwen3.5-9B**, **Qwen3.6-27B**, **Qwen3.8-27B**, and **Qwen3.5-MoE 35B-A3B**; additional model families are planned.

This page is the 5-minute path: install, pull a model, run inference. The Qwen3.5 examples below reflect what is currently shipped end-to-end — more model families coming.

## 1. Install

**Fastest — pre-built binary (no toolchain).** Detects your platform (macOS → Metal, Linux x86_64 + NVIDIA → CUDA), installs `lumen` + `lumen-server`, and helps you set up a model:

```bash
curl -fsSL https://servelumen.com/install.sh | bash
```

**Or build from source** (needs the Rust toolchain):

```bash
# Apple Silicon macOS (Metal backend — auto-detected)
cargo install --path crates/lumen-cli

# NVIDIA Linux (CUDA backend)
cargo install --path crates/lumen-cli --features cuda
```

Requirements (source build):

- Rust (pinned via `rust-toolchain.toml`; `rustup` installs it automatically)
- CUDA path: an NVIDIA GPU + driver. No build-time CUDA SDK needed — kernels compile at runtime via NVRTC.
- Metal path: macOS on Apple Silicon. No extra deps.

Reference: top-level [`README.md`](../README.md) § Build.

## 2. Pull a model

The registry currently lists the v1 cells (Qwen3.5 family plus the Qwen3.6-27B and Qwen3.8-27B dense models). Future model families will appear here as they ship.

```bash
# Default: Qwen3.5-9B at Q8_0 (~10 GB GGUF download, converts to LBC on first use)
lumen pull qwen3.5-9b:q8_0

# MoE — Q8_0, Q4_0, BF16 all in the registry
lumen pull qwen3.5-moe-35b-a3b:q8_0
lumen pull qwen3.5-moe-35b-a3b:q4_0
```

`lumen models` lists everything cached and everything available. The model registry source-of-truth is `model_registry.toml`. Unsupported `(model, quant)` combinations are rejected with a clear error listing available alternatives.

All three `qwen3.5-9b` quants (Q8_0, Q4_0, BF16) are in the registry and auto-download. To re-quantize a custom GGUF/LBC for a different target, use `lumen convert --input <gguf> --output <lbc> --requant q4_0`. See [`docs/cli.md`](cli.md) for convert options.

## 3. Run inference

```bash
# Default sampling: temperature 0.7 with a random seed — output varies each run.
# For reproducible output add `--seed <n>`; for greedy/argmax use `--temperature 0`.
lumen run qwen3.5-9b:q8_0 "Write a haiku about Rust"

# Longer answer with an explicit temperature
lumen run qwen3.5-9b:q8_0 "Explain quantum computing" \
  --max-tokens 200 --temperature 0.7

# MoE — Q8_0 and Q4_0 auto-download from the registry
lumen run qwen3.5-moe-35b-a3b:q4_0 "What is the meaning of life?" --max-tokens 200
```

(MoE and 27B BF16 are the highest-quality builds; they ship as large nested 2-shard GGUFs (~55–70 GB) that `lumen pull <model>:bf16` — and the installer's BF16 menu option — auto-download and convert. The download is large and not resumable, so make sure you have ample free disk.)

The backend is auto-detected: macOS → Metal, Linux with a CUDA device → CUDA, else SIMD CPU fallback. Force a backend with `--cuda`, `--metal`, or `--simd`. Pick a CUDA device with `--cuda-device <n>`.

## 4. What you should see

A coherent text stream, token by token, on your GPU (Metal on Apple Silicon, CUDA on NVIDIA). Decode throughput is competitive with llama.cpp on the benchmarked configs — Lumen meets or beats it on several. Full per-config numbers and methodology are in [`bench/RESULTS.md`](../bench/RESULTS.md).

## 5. Before you deploy

Read [`docs/production.md`](production.md) before putting Lumen on real traffic — it covers serving-mode choice (`lumen-server` vs `lumen run`), per-quant GPU sizing, and known limitations. One gotcha worth knowing on day one: PURE-greedy (`--temperature 0`, no penalty) deterministically loops on long generations (≥512 tokens) — use `--temperature 0.7` OR, on DENSE models, `--repetition-penalty 1.05 --repeat-last-n 64`. (When the flag is omitted the server/CLI apply a model-aware penalty: 1.05 dense / 1.03 MoE; MoE must stay ≤ 1.03.)

## Next steps

- [CLI reference](cli.md) — full flag list (also `lumen run --help`)
- [Model support](support.md) — what is verified-against-llama.cpp
- [HTTP server](server.md) — OpenAI + Anthropic wire formats
- [Environment variables](environment-variables.md) — all `LUMEN_*` flags
- [Troubleshooting](troubleshooting.md) — common failure modes
- [Architecture](architecture.md) — for contributors
