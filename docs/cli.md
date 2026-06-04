# CLI Reference

The canonical, always-up-to-date reference is `lumen run --help` (printed by [`crates/lumen-cli/src/help.rs`](../crates/lumen-cli/src/help.rs)).

## Subcommands

| Command | Purpose |
|---------|---------|
| `lumen run <model:quant> "<prompt>"` | Pull (if needed), convert (if needed), run inference, print text |
| `lumen pull <model:quant>` | Download GGUF, convert to LBC, cache; do not run |
| `lumen models` | List all registry entries and disk-cached LBCs |
| `lumen convert --input <gguf> --output <lbc> [--requant <q>]` | Manually convert a GGUF to LBC (optionally re-quantize) |
| `lumen --help` / `lumen run --help` / `lumen convert --help` | Full reference |

## Common flags (excerpt)

| Flag | Description |
|------|-------------|
| `--system <text>` | System prompt |
| `--max-tokens <n>` | Tokens to generate (default: unlimited, stops at EOS) |
| `--temperature <f>` | Sampling temperature (0 = greedy, default 0.8) |
| `--top-p` / `--top-k` / `--min-p` | Nucleus / top-K / min-prob cutoffs |
| `--repetition-penalty` / `--presence-penalty` / `--frequency-penalty` | Sampling penalties |
| `--seed <n>` | Random seed for reproducibility |
| `--cuda` / `--metal` / `--simd` | Force a backend |
| `--cuda-device <n>` | CUDA device ordinal (default 0) |
| `--context-len <n>` | KV cache size (auto-sized by default) |
| `--kv-precision f16\|f32` | KV cache precision (per-backend default: Metal f16, CUDA / CPU f32) |
| `--kv-disk-dir <path>` | Directory for disk-persistent KV cache |
| `--kv-disk-space-mb <n>` | KV cache space budget on disk |
| `--session-save <p>` / `--session-resume <p>` | Persist / restore a Session across runs (Metal today) |
| `--no-gpu-resident` | Stream weights from disk instead of GPU memory |
| `--gpu-resident` | Force GPU-resident weights |
| `--verbose` | Show diagnostics and metrics |
| `--profile` | Per-operation timing breakdown (implies `--verbose`) |
| `--tokens "<t1 t2 ...>"` | Raw token mode (skip BPE tokenizer) |

For the complete flag list run `lumen run --help`. The flag set may include additional fields (`--accelerate`, `--option-a`, `--routing-bias`, `--threads`, `--verbose-routing`, `--sync`, `--async`, …) that are not listed here.

**Configuration precedence:** CLI flag > environment variable > built-in default. For example, `--kv-precision f32` overrides `LUMEN_KV_PRECISION=f16`; with neither set the per-backend default applies.
