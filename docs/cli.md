# CLI Reference

The canonical, always-up-to-date reference is `lumen run --help` (printed by [`crates/lumen-cli/src/help.rs`](../crates/lumen-cli/src/help.rs)).

## Subcommands

| Command | Purpose |
|---------|---------|
| `lumen run <model:quant> "<prompt>"` | Pull (if needed), convert (if needed), run inference, print text. On a CUDA build, a text model the GPU the run would use (`--cuda-device`, else device 0) cannot run is refused before its first byte, naming the model's quants this does not rule out: a K-quant (Q4_K_M, Q5_K_M) on a GPU below compute capability 8.0, a model whose weights alone, judged from its download size, are bigger than the GPU's memory, and a BF16 model whose weights, held as F32 below compute capability 9.0, cannot fit (judged from a download whose size is known and is the one measured for the model, so not through a proxy, which reports no size); `--simd`, `--metal`, or `--async` without `--cuda` skips the check. A run on a CPU backend is refused the same way when the weights alone are bigger than the memory this machine has available |
| `lumen run <model> "<prompt>"` | Bare name: the model's default quant (Q8_0 for `qwen3.5-9b`, Q4_0 for `qwen3.8-27b` and `qwen3.5-moe`; `lumen models` marks it), downloaded on first use like any `model:quant`. When another quant of the model is already downloaded, it says so before downloading the default |
| `lumen pull <model:quant>` | Download GGUF, convert to LBC, cache; do not run. On a CUDA build, a text model CUDA device 0 (or `--cuda-device <n>`) cannot run is refused before its first byte, by the same rules as `lumen run`. With `--cpu`, or outside macOS with no NVIDIA GPU or on a build without CUDA, the check is made against the memory this machine has available |
| `lumen pull qwen-image` | Download the Qwen-Image-2.1 checkpoint (30.8 GiB, pinned to one Hugging Face commit, every file checked by size and SHA-256), convert it to `.lbi` for `lumen-server qwen-image` (about 78 GiB of disk at the peak, the download included), and remove the checkpoint's other files from the cache; the model's directory in the cache must be yours alone; CUDA builds only, refused before any download unless the first visible CUDA device has compute capability 8.0+ and at least 21.0 GiB. An interrupted pull's downloads resume where they stopped, and an unfinished conversion is made again |
| `lumen image "<prompt>" [-o file.png] [--size WxH] [--steps n] [--seed n]` | Make one picture with Qwen-Image-2.1 and write a PNG (default `image-<seed>.png`, never overwriting; 1024x1024, 40 steps, random seed); the first run downloads and converts the model as `lumen pull qwen-image` does, without asking, as `lumen run` fetches a text model, when the first visible CUDA device has compute capability 8.0+ and at least 21.0 GiB (otherwise it is refused before the download); CUDA builds only |
| `lumen models` | List all registry entries, the disk-cached LBCs and a cached image model |
| `lumen convert --input <gguf> --output <lbc> [--requant <q>]` | Manually convert a GGUF to LBC (optionally re-quantize; `--requant` is refused for MoE models, whose expert tensors are carried in their source quantization) |
| `lumen convert --input <donor-gguf> --from-hf <dir> --output <lbc>` | Import an HF compressed-tensors INT4 checkpoint (CtInt4G32) or an NVIDIA ModelOpt mixed-precision checkpoint whose attention and MLP projections and GDN `in_proj_qkv` / `in_proj_z` are each NVFP4, FP8 or BF16 (the GDN output projection FP8 or BF16, the output head NVFP4 or BF16; the embedding BF16, the GDN `in_proj_a` / `in_proj_b` unquantized); CUDA runtime only (see `docs/lbc-format.md`) |
| `lumen --help` / `lumen run --help` / `lumen convert --help` | Full reference |

## Common flags (excerpt)

| Flag | Description |
|------|-------------|
| `--system <text>` | System prompt |
| `--max-tokens <n>` | Tokens to generate (default: unlimited, stops at EOS; 8192 on CUDA and the CPU when `--context-len` is not given) |
| `--temperature <f>` | Sampling temperature (0 = greedy, default 0.7) |
| `--top-p` / `--top-k` / `--min-p` | Nucleus / top-K / min-prob cutoffs |
| `--repetition-penalty` / `--presence-penalty` / `--frequency-penalty` | Sampling penalties |
| `--seed <n>` | Sampling seed (default: random each run; set a fixed value for reproducible output) |
| `--cuda` / `--metal` / `--simd` | Force a backend |
| `--cuda-device <n>` | CUDA device ordinal (default 0) |
| `--context-len <n>` | KV cache size (auto-sized by default) |
| `--kv-precision f16\|bf16\|f32` | KV cache storage (per-backend default: Metal f16, CUDA / CPU f32). On CUDA `f16` and `bf16` halve the cache's bytes and its decode-attention reads; the model's keys and values are rounded to half or to bfloat16 on the way in. `bf16` is CUDA only (see `docs/environment-variables.md`, `LUMEN_KV_PRECISION`) |
| `--kv-disk-dir <path>` | Directory for disk-persistent KV cache |
| `--kv-disk-space-mb <n>` | KV cache space budget on disk |
| `--session-save <p>` / `--session-resume <p>` | Persist / restore a Session across runs (Metal and the CPU backends). The file holds the full history and the next token's logits, so a resumed run on the same backend whose `--tokens` start with the saved history continues token for token as if it had never stopped (with `--temperature 0` and the same sampling flags; a sampled run restarts its random numbers from its seed). A resumed session keeps the cache size it was saved with, and a `--context-len` of another size is refused; a file an earlier version saved after generation is refused |
| `--no-gpu-resident` | Stream weights from disk instead of GPU memory (CUDA only: Metal decode requires resident weights) |
| `--gpu-resident` | Force GPU-resident weights |
| `--verbose` | Show diagnostics and metrics |
| `--profile` | Per-operation timing breakdown (implies `--verbose`) |
| `--tokens "<t1 t2 ...>"` | Raw token mode (skip BPE tokenizer) |

For the complete flag list run `lumen run --help`. The flag set may include additional fields (`--accelerate`, `--option-a`, `--routing-bias`, `--threads`, `--verbose-routing`, `--sync`, `--async`, …) that are not listed here.

**Configuration precedence:** CLI flag > environment variable > built-in default. For example, `--kv-precision f32` overrides `LUMEN_KV_PRECISION=f16`; with neither set the per-backend default applies.
