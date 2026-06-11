pub(crate) fn print_usage() {
    println!(
        "\
lumen - GPU-resident LLM inference engine

USAGE:
    lumen <COMMAND> [OPTIONS]

COMMANDS:
    run                   Run inference on a model (single-process; cold-loads every invocation)
    pull                  Download and convert a model from the registry
    models                List cached and available models
    convert               Convert a GGUF model to LBC format
    generate-test-model   Generate a synthetic model (LBC file)
    bench                 Run benchmarks (I/O, throughput, cold/warm)
    purge                 Evict a model file from the OS page cache
    help                  Print this help message

FOR CONCURRENT-CLIENT SERVING:
    Use 'lumen-server' (long-running HTTP server with OpenAI/Anthropic wire formats).
    'lumen run' is per-process: cold-loads weights every invocation (60-120s for MoE).
    16-client concurrent burst against 'lumen run' fails by design.

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version"
    );
}

pub(crate) fn print_pull_usage() {
    println!(
        "\
USAGE:
    lumen pull <model-name> [OPTIONS]

ARGUMENTS:
    <model-name>          Model name or alias from the registry (currently: qwen3-5-9b, qwen3.5-9b)

OPTIONS:
    --quant <scheme>      Quantization format (default: Q8_0). Available: Q8_0, Q4_0, BF16, F16
    --yes, -y             Skip download confirmation prompt
    -h, --help            Print this help message

EXAMPLES:
    lumen pull qwen3-5-9b                 Download Qwen3.5 9B (Q8_0)
    lumen pull qwen3.5-9b --quant Q4_0    Download Qwen3.5 9B (Q4_0)
    lumen pull qwen3-5-9b --yes           Download Qwen3.5 9B without confirmation"
    );
}

pub(crate) fn print_run_usage() {
    println!(
        "\
USAGE:
    lumen run <model>:<quant> \"your prompt\" [OPTIONS]
    lumen run --model <name-or-path> --prompt \"Hello\" [OPTIONS]
    lumen run --model <path.lbc> --tokens \"0 1 2\" [OPTIONS]

CONFIGURATION PRECEDENCE:
    CLI flag > environment variable > built-in default.
    Example: --kv-precision f32 overrides LUMEN_KV_PRECISION=f16; with
    neither set, the per-backend default (Metal f16, CUDA/CPU f32) applies.

EXAMPLES:
    lumen run qwen3.5-9b:q8_0 \"What is the meaning of life?\"
    lumen run qwen3.5-9b:q4_0 \"Write a haiku about Rust\"
    lumen run qwen3.5-9b:bf16 \"Explain quantum computing\" --max-tokens 200
    lumen run --model qwen3-5-9b --prompt \"hello\" --temperature 0.7

MODELS:
    Run 'lumen models' to see available models.
    Quantization tag: q8_0 (best quality / production default), q4_0 (smaller),
                      bf16 (full precision, fastest prefill on supported GPUs)

OPTIONS:
    --model <name|path>   Model name from registry (e.g. qwen3-5-9b:q8_0) or path to .lbc/.gguf file
    --prompt <text>       Text prompt (requires LBC with embedded tokenizer)
    --system <text>       System prompt (optional, overrides default)
    --think               Enable the Qwen3.5 reasoning trace: opens a <think> block; the
                          reasoning is printed to stderr ([reasoning] ...) and the answer to
                          stdout. Default OFF (closed empty-think tail). --no-think forces OFF
                          (overrides LUMEN_CHAT_ENABLE_THINKING).
    --tokens <ids>        Space-separated token IDs (--prompt and --tokens are mutually exclusive)
    --max-tokens <n>      Max tokens to generate (default: unlimited, stops at EOS)
                          PRODUCTION: pass --max-tokens 512 (minimum) for multilingual prompts;
                          the Qwen3.5 chat template opens a <think>...</think> block that may
                          consume the budget before producing the answer in the target language.
    --stop <text>         Textual stop sequence (mirrors the server OpenAI `stop` /
                          Anthropic `stop_sequences`). The answer is cut at the first matched
                          stop string; the match and everything after it are dropped. Repeatable
                          and accepts a comma-separated list, so `--stop A --stop B` and
                          `--stop A,B` are equivalent. Default: none (answer prints to EOS).
    --temperature <f>     Sampling temperature (default: 0.7, 0=greedy)
                          PRODUCTION: PURE-greedy (--temperature 0 + no penalty) deterministically
                          loops on long-form generation (>=512 tokens). Use sampling (0.7); for a
                          greedy long-form penalty use --repetition-penalty 1.05 --repeat-last-n 64
                          on DENSE models only (MoE must stay <= 1.03 — leave the flag unset so the
                          model-aware default applies).
    --top-p <f>           Nucleus sampling cutoff (default: 1.0 = disabled)
    --top-k <n>           Top-K sampling cutoff (default: 0 = disabled)
    --min-p <f>           Min-probability sampling cutoff (default: 0.0 = disabled)
    --repetition-penalty <f>
                          Multiplicative penalty for tokens in the recent window. When this flag
                          is OMITTED the default is MODEL-AWARE (resolved by
                          runtime_defaults::repetition_penalty_default): 1.05 dense / 1.03 MoE.
                          --repeat-penalty is an accepted alias.
                          PRODUCTION recommendation for long-form greedy on DENSE models: 1.05.
                          MoE (Qwen3.5-MoE class) MUST stay <= 1.03: 1.05 corrupts MoE arithmetic
                          (matrix-proven '17 x 20 = ... = 39') — leave the flag unset so the
                          model-aware 1.03 default applies.
    --repeat-last-n <n>   Window size for --repetition-penalty (default: 0 = disabled).
                          PRODUCTION recommendation paired with --repetition-penalty 1.05 on
                          DENSE models: 64. (Does not apply to the MoE 1.03 default.)
    --presence-penalty <f>
                          Additive penalty for tokens already in the context (default: 0.0).
    --frequency-penalty <f>
                          Additive penalty proportional to token frequency (default: 0.0).
    --seed <n>            Random seed (default: random, set for reproducibility)
    --sync                Use sync file backend instead of mmap
    --async               Use async I/O backend (background prefetch thread)
    --simd                Use SIMD-accelerated compute backend
    --threads <n>         Thread count for SIMD backend (default: 0 = auto-detect)
    --metal               Use Metal GPU compute backend (macOS only)
    --cuda                Use CUDA GPU compute backend (NVIDIA, requires --features cuda)

    If no backend flag (--metal, --cuda, --simd) is given, auto-detects:
      macOS -> Metal,  Linux with /dev/nvidia0 -> CUDA,  else -> SIMD (CPU)
    --cuda-device <n>     Select CUDA device ordinal (default: 0, implies --cuda)
    --accelerate          Use Accelerate AMX batched prefill (macOS only, use with --simd)
    --gpu-resident        Pre-load all weights into GPU Metal buffers (DEFAULT with --metal)
    --no-gpu-resident     Disable GPU-resident mode, use SSD-streaming (alias: --streaming)
    --option-a            MoE: dispatch only top-K experts per token (streaming + GPU-resident)
    --routing-bias <f>    MoE: cache-conditional routing bias lambda (default: 0.0 = disabled)
    --context-len <n>     Max context length for KV cache (default: auto-sized to prompt + generation + headroom).
                          Memory: roughly `context_len * num_layers * num_kv_heads * head_dim * 4 bytes * 2`
                          per session (F32 KV, K+V doubled; F16 halves the CPU mirror).
                          Qwen3.5-9B example: 32768 * 32 * 8 * 128 * 4 * 2 = ~8.6 GB at full context.
                          PRODUCTION (BF16 only): pin --context-len to a single value (e.g. 8192)
                          per deployment. The BF16 mmvf kernel produces different first-token
                          argmax at different KV-cache layout sizes; pinning eliminates
                          the cross-deployment non-determinism. Q8 / Q4 are unaffected.
    --kv-precision <p>    KV cache storage precision: f16 | f32. Default: backend-appropriate
                          (Metal -> f16, CUDA -> f32, CPU -> f32). The Metal backend pins KV
                          to f16; passing --kv-precision f32 on Metal is rejected with an
                          explicit error. CUDA likewise requires f32 in the current release.
                          Honors `LUMEN_KV_PRECISION=f16|f32` env override.
    --kv-disk-dir <dir>   Directory for the disk-persistent KV cache.
                          When set, the runtime purges stale .tmp.<pid> writes at
                          startup and (if --kv-disk-space-mb is also set) evicts
                          older entries to fit the budget.
    --kv-disk-space-mb <n>
                          Soft budget (megabytes) for the disk KV directory. Lowest
                          scoring entries -- `(hits+1)*tokens/file_size`, 0.25x
                          penalty on live-session prefixes -- are evicted first.
    --session-resume <p>  Restore a previously persisted Session from <p>. The file MUST
                          have been written by a prior `--session-save <p>` against the
                          SAME model (a fingerprint mismatch is rejected with a clear
                          error). Reuses any cached prefix with the new prompt; falls back
                          to a cold prefill on divergence. Metal-only today; CUDA returns
                          an error (the CPU↔GPU sync path is not yet implemented).
    --session-save <p>    Persist the live Session to <p> after generation completes.
                          Atomically published via `.tmp.<pid>` -> rename so concurrent
                          readers never see a partial write. Captures K/V bytes, the
                          GDN recurrent state, the token history, and the pending logits
                          so a subsequent `--session-resume <p>` is argmax-identical to a
                          continuous session.
                          The override `LUMEN_SUFFIX_THRESHOLD=<n>` (positive integer)
                          tunes the per-call cached-prefix prefill/decode hand-off cutoff.
    --profile             Print per-operation timing breakdown after inference (implies --verbose)
    --verbose, -v         Show diagnostics, metrics, and banner (default: quiet, text only)
    --verbose-routing     Print per-layer MoE router diagnostics (entropy, expert selection)

ENVIRONMENT VARIABLES (CUDA backend):
    LUMEN_CUDA_DECODE_DELAY_US=<N>
                          Per-decode-step CPU sleep in microseconds, applied AFTER
                          `device.synchronize()` in the CUDA decode paths. Default
                          `0` (OFF) is bit-exact. Set `=50` to mitigate a CUDA-
                          scheduler timing race seen under heavy MoE Q4 server
                          concurrency. CLI is deterministic without this knob.
                          Cost <=1% TPOT."
    );
}

pub(crate) fn print_bench_usage() {
    println!(
        "\
USAGE:
    lumen bench [OPTIONS]

OPTIONS:
    --suite <name>       Preset suite: minimal | async-comparison | ssd-hypothesis | custom (default: minimal)
    --size <spec>        Model size: 256mb | 1gb | 4gb | 7b (default: 1gb)
    --backend <name>     Storage backend: sync | mmap | async (default: mmap)
    --prefetch <n>       Prefetch distance (default: 2)
    --mode <name>        Pipeline mode: minmem | perf (default: minmem)
    --cold-start         Purge page cache before each iteration
    --iters <n>          Measured iterations (default: 3)
    --prompt-len <n>     Prompt token count (default: 128)
    --gen-len <n>        Tokens to generate (default: 32)
    --output-dir <path>  Directory for models and results
    --json               Output as JSON
    --simd               Use SIMD-accelerated compute backend"
    );
}

pub(crate) fn print_convert_usage() {
    println!(
        "\
USAGE:
    lumen convert --input <model.gguf> [--output <model.lbc>] [OPTIONS]

OPTIONS:
    --input <path>       Path to input GGUF model file (required)
    --output <path>      Path to output LBC file (default: input with .lbc extension)
    --dequantize         Dequantize all tensors to F32 (larger but compatible)
    --requant <scheme>   Requantize weights to target scheme during conversion
                         Supported: q4_0, q8_0
    --target <backend>   Runtime backend the LBC is being prepared for.
                         metal:   upcast K-quant layer tensors (Q2..Q6_K) to Q8_0
                                  (Metal has no K-quant dispatch kernels).
                         generic: keep K-quant layer tensors as-is (CUDA host).
                         Default: metal on macOS, generic elsewhere."
    );
}
