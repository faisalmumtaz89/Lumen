pub(crate) fn print_usage() {
    println!(
        "\
lumen - GPU-resident LLM inference engine

USAGE:
    lumen <COMMAND> [OPTIONS]

COMMANDS:
    run                   Run inference on a model
    pull                  Download and convert a model from the registry
    models                List cached and available models
    convert               Convert a GGUF model to LBC format
    generate-test-model   Generate a synthetic model (LBC file)
    bench                 Run benchmarks (I/O, throughput, cold/warm)
    purge                 Evict a model file from the OS page cache
    help                  Print this help message

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
    <model-name>          Model name or alias from the registry (e.g. qwen2-5-3b, llama-8b)

OPTIONS:
    --quant <scheme>      Quantization format (default: Q8_0). Available: Q8_0, Q4_0, F16
    --yes, -y             Skip download confirmation prompt
    -h, --help            Print this help message

EXAMPLES:
    lumen pull qwen2-5-3b                 Download Qwen2.5 3B (Q8_0)
    lumen pull llama-8b --quant Q4_0      Download Llama 3.1 8B (Q4_0)
    lumen pull tinyllama --yes             Download TinyLlama without confirmation"
    );
}

pub(crate) fn print_run_usage() {
    println!(
        "\
USAGE:
    lumen run <model>:<quant> \"your prompt\" [OPTIONS]
    lumen run --model <name-or-path> --prompt \"Hello\" [OPTIONS]
    lumen run --model <path.lbc> --tokens \"0 1 2\" [OPTIONS]

EXAMPLES:
    lumen run qwen2.5-3b:q8_0 \"What is the meaning of life?\"
    lumen run tinyllama:q4_0 \"Write a haiku about Rust\"
    lumen run llama-8b:q8_0 \"Explain quantum computing\" --max-tokens 200
    lumen run --model qwen2-5-3b --prompt \"hello\" --temperature 0.7

MODELS:
    Run 'lumen models' to see available models.
    Quantization tag: q8_0 (best quality), q4_0 (smaller/faster), f16 (full precision)

OPTIONS:
    --model <name|path>   Model name from registry (e.g. qwen2-5-3b:q8_0) or path to .lbc/.gguf file
    --prompt <text>       Text prompt (requires LBC with embedded tokenizer)
    --system <text>       System prompt (optional, overrides default)
    --tokens <ids>        Space-separated token IDs (--prompt and --tokens are mutually exclusive)
    --max-tokens <n>      Max tokens to generate (default: unlimited, stops at EOS)
    --temperature <f>     Sampling temperature (default: 0.8, 0=greedy)
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
    --context-len <n>     Max context length for KV cache (default: auto-sized to prompt + generation + headroom)
    --profile             Print per-operation timing breakdown after inference (implies --verbose)
    --verbose, -v         Show diagnostics, metrics, and banner (default: quiet, text only)
    --verbose-routing     Print per-layer MoE router diagnostics (entropy, expert selection)"
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
                         Supported: q4_0, q8_0"
    );
}
