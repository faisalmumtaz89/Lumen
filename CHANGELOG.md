# Changelog

All notable changes to Lumen are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
`0.1.0` is published.

## [Unreleased] — pre-`0.1.0`

### Production-ready (2026-06-02)

CUDA (NVIDIA, compute capability 8.0+ — Ampere / Hopper; benchmarked on A100-80GB):

- Qwen3.5-9B dense at Q8_0 (**0.91× llama.cpp** decode), Q4_0 (0.64× llama.cpp), BF16 (**0.93–0.94× llama.cpp** decode)
- Qwen3.5-MoE-35B-A3B (architecture-truthful active params: 30B-A3B) at Q8_0 (0.584× llama.cpp), Q4_0 (0.674× llama.cpp), and **BF16 (0.902× llama.cpp, production-recommended)**
- Validated end-to-end on the full models × quants matrix against llama.cpp

Metal (Apple Silicon, M-series; benchmarked on M3 Ultra):

- Qwen3.5-9B dense at Q8_0 (**0.98× llama.cpp** decode), Q4_0 (**1.17× llama.cpp** decode — beats llama.cpp), BF16 (0.83× llama.cpp decode)
- Qwen3.5-MoE-35B-A3B at Q8_0 + Q4_0 functional (sole provider on Apple Silicon — llama.cpp build 8680 cannot load this architecture; requires `LUMEN_METAL_MMAP_ONLY=1`)
- Validated end-to-end on the full models × quants matrix against llama.cpp

### Features

- Hybrid GDN linear-attention + dense FFN forward pass (L=32 layers, 24 GDN + 8 full-attention)
- MoE forward pass with top-K expert dispatch
- Flash Attention 2 prefill (CUDA, `LUMEN_CUDA_FA2_BLOCKSKIP=1`)
- Long-context decode beyond 40,950 tokens via tiled streaming-softmax (default-on as of `ATTN_DECODE_TILED_DEFAULT_THRESHOLD=0`)
- F16 KV cache (Metal); F32 KV cache (CUDA, CPU)
- Disk-persistent KV cache with eviction policy (`--kv-disk-dir`, `--kv-disk-space-mb`)
- Session save/resume with suffix-prefill cache (a cache-hit turn skips reprocessing the shared prefix; cache-reuse throughput is not part of the published benchmark suite)
- HTTP server (`lumen-server`) with OpenAI + Anthropic wire formats, SSE streaming, template-driven tool-call parser (v1 ships the Qwen3.5 `<tool_call>` marker pattern)
- Per-request reasoning / extended-thinking control (default OFF): OpenAI `enable_thinking` (+ vLLM `chat_template_kwargs.enable_thinking`) with `delta.reasoning_content`, Anthropic `thinking.type` with a `thinking` content block, CLI `--think`, and a separate `reasoning_budget` (distinct from `max_tokens`); `LUMEN_CHAT_ENABLE_THINKING` overrides the default (see `docs/server.md` and `.artifacts/REASONING-CONTROL-DESIGN.md`)
- BPE tokenizer embedded in LBC v3 (no Python at runtime)
- GGUF → LBC converter supporting K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 via dequant on import
- `Configuration precedence: CLI flag > env var > built-in default` documented end-to-end

### Known limitations

- Concurrent CLI bursts (≥4) per GPU are unsupported by design — use `lumen-server`
- Q8 / Q4 prefill × llama.cpp ratios are structurally below 1.0 on the current NVRTC compute_61 stack
- PURE-greedy long-form generation (≥512 tokens) deterministically loops — use `--temperature 0.7` or, on DENSE models, `--repetition-penalty 1.05 --repeat-last-n 64` (when omitted the server/CLI apply a model-aware penalty: 1.05 dense / 1.03 MoE — MoE must stay ≤ 1.03 or arithmetic corrupts)
- BF16 MoE-30B-A3B requires a dedicated 80 GB+ GPU (peak VRAM 72.4 GB)
- `lumen-server` Authorization / CORS / per-request timeout are not implemented; deploy behind a reverse proxy
- BF16-dense / Q8-MoE / Q4-MoE on Metal require `LUMEN_METAL_MMAP_ONLY=1` (M3 Ultra 96 GB residency budget)

### History

For pre-`0.1.0` commit-level history see the git log. Notable cumulative work:

- Extensive optimization across the CUDA and Metal kernel paths
- Removed 1385 LoC of non-Qwen3.5 architecture support to focus the v1 surface
- Ten-configuration CUDA gold-standard validation against llama.cpp
- Production-readiness audit across 10 operational dimensions (2026-05-29)
- CUDA final validation (2026-06-02): full models × quants matrix verified on A100-80GB
- Metal final validation (2026-06-02): full models × quants matrix verified on M3 Ultra, head-to-head vs llama-bench build 8680

### Documentation

- Documentation pass (2026-06-02): added the `docs/` tree, `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`; fixed README hero numbers and the vLLM prefill ratio (2.29× → 2.62×).

[Unreleased]: https://github.com/faisalmumtaz89/Lumen/compare/main...HEAD
