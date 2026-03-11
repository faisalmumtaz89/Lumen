#!/usr/bin/env bash
#
# Lumen Benchmark Suite
# ---------------------
# Automated, reproducible benchmarking for Lumen inference engine vs MLX.
# Compatible with macOS bash 3.2 (no associative arrays).
#
# Usage:
#   ./bench/run_bench.sh                          # Full suite, all models
#   ./bench/run_bench.sh --quick                   # pp128+gen128 only, 3 runs
#   ./bench/run_bench.sh --lumen-only              # Skip MLX comparison
#   ./bench/run_bench.sh --models "llama-8b"       # Filter to matching models
#   ./bench/run_bench.sh --runs 10                 # 10 measured runs per config
#   ./bench/run_bench.sh --prompt-lengths "128 512" # Custom prompt lengths
#   ./bench/run_bench.sh --gen-lengths "32 128"     # Custom generation lengths
#   ./bench/run_bench.sh --cooldown 10              # 10s cooldown between models
#   ./bench/run_bench.sh --no-build                 # Skip cargo build step
#
# Output:
#   bench/results/YYYY-MM-DD_HHMMSS/results.md    # Markdown summary
#   bench/results/YYYY-MM-DD_HHMMSS/results.json  # Machine-parseable JSON
#   bench/results/YYYY-MM-DD_HHMMSS/raw/          # Raw output from each run

set -euo pipefail

# ============================================================================
# Configuration defaults
# ============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="/tmp/lumen-bench"
LUMEN_BIN="${REPO_ROOT}/target/release/lumen"
MLX_VENV="${HOME}/.venvs/mlx-bench"
LLAMA_BENCH_BIN="/opt/homebrew/bin/llama-bench"

PROMPT_LENGTHS=(32 128 512 1024)
GEN_LENGTHS=(32 128 256)
RUNS=5
WARMUP_RUNS=1
COOLDOWN_SECS=30
CONFIG_COOLDOWN_SECS=5
MODEL_FILTER=""
LUMEN_ONLY=0
QUICK_MODE=0
SKIP_BUILD=0
SKIP_LLAMACPP=0
RUNS_EXPLICIT=0

# Auto-detect hardware
if command -v sysctl >/dev/null 2>&1; then
    HARDWARE_NAME="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
    GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Total Number of Cores' | awk '{print $NF}' || echo '')"
    TOTAL_MEM_GB="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
    if [ -n "$GPU_CORES" ] && [ "$TOTAL_MEM_GB" -gt 0 ]; then
        HARDWARE_DESC="${HARDWARE_NAME} (${GPU_CORES} GPU cores, ${TOTAL_MEM_GB} GB)"
    elif [ "$TOTAL_MEM_GB" -gt 0 ]; then
        HARDWARE_DESC="${HARDWARE_NAME} (${TOTAL_MEM_GB} GB)"
    else
        HARDWARE_DESC="${HARDWARE_NAME}"
    fi
else
    HARDWARE_DESC="Unknown"
fi

# MLX model mapping: parallel arrays (bash 3.2 compatible, no declare -A)
MLX_KEYS=()
MLX_PATHS=()

# llama.cpp model mapping: parallel arrays (bash 3.2 compatible)
LLAMA_KEYS=()
LLAMA_PATHS=()
LLAMA_UNSUPPORTED=()

# ============================================================================
# Argument parsing
# ============================================================================

usage() {
    cat <<'USAGE'
Lumen Benchmark Suite

USAGE:
  ./bench/run_bench.sh [OPTIONS]

OPTIONS:
  --quick                  Quick mode: pp128+gen128 only, 3 runs
  --lumen-only             Skip MLX and llama.cpp comparison benchmarks
  --skip-llamacpp          Skip llama.cpp benchmarks only
  --no-build               Skip cargo build step
  --models FILTER          Comma-separated substrings to match model filenames
                           e.g. --models "llama-8b,tinyllama"
  --runs N                 Number of measured runs per config (default: 5)
  --warmup N               Number of warmup runs (default: 1)
  --cooldown N             Seconds to pause between model switches (default: 30)
  --config-cooldown N      Seconds to pause between configs within a model (default: 5)
  --prompt-lengths "L..."  Space-separated prompt lengths (default: "32 128 512 1024")
  --gen-lengths "L..."     Space-separated generation lengths (default: "32 128 256")
  --bench-dir DIR          Directory containing .lbc models (default: /tmp/lumen-bench)
  --help                   Show this help message

EXAMPLES:
  ./bench/run_bench.sh --quick --lumen-only
  ./bench/run_bench.sh --models "llama-8b" --runs 10
  ./bench/run_bench.sh --prompt-lengths "128 512" --gen-lengths "128"
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --lumen-only)
            LUMEN_ONLY=1
            shift
            ;;
        --skip-llamacpp)
            SKIP_LLAMACPP=1
            shift
            ;;
        --no-build)
            SKIP_BUILD=1
            shift
            ;;
        --models)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            RUNS_EXPLICIT=1
            shift 2
            ;;
        --warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --cooldown)
            COOLDOWN_SECS="$2"
            shift 2
            ;;
        --config-cooldown)
            CONFIG_COOLDOWN_SECS="$2"
            shift 2
            ;;
        --prompt-lengths)
            IFS=' ' read -ra PROMPT_LENGTHS <<< "$2"
            shift 2
            ;;
        --gen-lengths)
            IFS=' ' read -ra GEN_LENGTHS <<< "$2"
            shift 2
            ;;
        --bench-dir)
            BENCH_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

if [[ "$QUICK_MODE" -eq 1 ]]; then
    PROMPT_LENGTHS=(128)
    GEN_LENGTHS=(128)
    # Only override RUNS if user didn't explicitly set it
    if [[ "$RUNS_EXPLICIT" -eq 0 ]]; then
        RUNS=3
    fi
fi

# ============================================================================
# Output directory setup
# ============================================================================

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/bench/results/${TIMESTAMP}"
RAW_DIR="${RESULTS_DIR}/raw"
mkdir -p "$RAW_DIR"

# ============================================================================
# Logging helpers
# ============================================================================

log() {
    echo "[$(date +%H:%M:%S)] $*" >&2
}

log_section() {
    echo "" >&2
    echo "================================================================" >&2
    echo "  $*" >&2
    echo "================================================================" >&2
    echo "" >&2
}

# ============================================================================
# Thermal monitoring helpers
# ============================================================================

THERMAL_EXTRA_COOLDOWN=30

# log_thermal: log current thermal pressure level to stderr and optionally to a file
# Args: $1=raw_output_file (optional, thermal state appended if provided)
# Returns: 0 if thermal OK, 1 if high thermal pressure detected
log_thermal() {
    local raw_file="${1:-}"
    local thermal_line
    thermal_line=$(pmset -g therm 2>/dev/null | grep -iE 'thermal|pressure|warning' | head -1 || echo "")

    if [[ -z "$thermal_line" ]]; then
        thermal_line="thermal: nominal (no warnings)"
    fi

    log "  Thermal: ${thermal_line}"

    if [[ -n "$raw_file" ]]; then
        echo "# THERMAL STATE: ${thermal_line}" >> "$raw_file"
    fi

    # Detect high thermal pressure: "heavy" or "critical" or "serious" or "tripped"
    if echo "$thermal_line" | grep -qiE 'heavy|critical|serious|tripped'; then
        return 1
    fi
    return 0
}

# check_thermal_and_cool: check thermal state, warn and add extra cooldown if hot
# Args: $1=raw_output_file (optional)
check_thermal_and_cool() {
    local raw_file="${1:-}"
    if ! log_thermal "$raw_file"; then
        log "  WARNING: High thermal pressure detected! Adding ${THERMAL_EXTRA_COOLDOWN}s extra cooldown..."
        if [[ -n "$raw_file" ]]; then
            echo "# THERMAL WARNING: Extra ${THERMAL_EXTRA_COOLDOWN}s cooldown applied" >> "$raw_file"
        fi
        sleep "$THERMAL_EXTRA_COOLDOWN"
    fi
}

# ============================================================================
# Math helpers (inline python for median/stddev)
# ============================================================================

# compute_stats: reads newline-separated floats from stdin, outputs "median stddev"
compute_stats() {
    python3 -c "
import sys, statistics
vals = [float(line.strip()) for line in sys.stdin if line.strip()]
if not vals:
    print('0.0 0.0')
elif len(vals) == 1:
    print(f'{vals[0]:.1f} 0.0')
else:
    med = statistics.median(vals)
    sd = statistics.stdev(vals)
    print(f'{med:.1f} {sd:.1f}')
"
}

# ============================================================================
# MLX model lookup (bash 3.2 compatible -- no associative arrays)
# ============================================================================

mlx_add() {
    local key="$1"
    local path="$2"
    MLX_KEYS+=("$key")
    MLX_PATHS+=("$path")
}

# Returns the MLX path for a given LBC basename, or empty string if not found
mlx_lookup() {
    local key="$1"
    local i
    for ((i = 0; i < ${#MLX_KEYS[@]}; i++)); do
        if [[ "${MLX_KEYS[$i]}" == "$key" ]]; then
            echo "${MLX_PATHS[$i]}"
            return 0
        fi
    done
    echo ""
    return 1
}

# ============================================================================
# llama.cpp model lookup (bash 3.2 compatible -- no associative arrays)
# ============================================================================

llama_add() {
    local key="$1"
    local path="$2"
    LLAMA_KEYS+=("$key")
    LLAMA_PATHS+=("$path")
}

llama_lookup() {
    local key="$1"
    local i
    for ((i = 0; i < ${#LLAMA_KEYS[@]}; i++)); do
        if [[ "${LLAMA_KEYS[$i]}" == "$key" ]]; then
            echo "${LLAMA_PATHS[$i]}"
            return 0
        fi
    done
    echo ""
    return 1
}

llama_is_unsupported() {
    local key="$1"
    local u
    for u in "${LLAMA_UNSUPPORTED[@]}"; do
        if [[ "$u" == "$key" ]]; then
            return 0
        fi
    done
    return 1
}

# ============================================================================
# Model discovery
# ============================================================================

discover_lbc_models() {
    local models=()
    for f in "${BENCH_DIR}"/*.lbc; do
        [[ -f "$f" ]] || continue
        local bn
        bn="$(basename "$f" .lbc)"

        # Apply model filter if specified
        if [[ -n "$MODEL_FILTER" ]]; then
            local matched=0
            IFS=',' read -ra filters <<< "$MODEL_FILTER"
            for filt in "${filters[@]}"; do
                filt="$(echo "$filt" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                if [[ "$bn" == *"$filt"* ]]; then
                    matched=1
                    break
                fi
            done
            [[ "$matched" -eq 1 ]] || continue
        fi

        models+=("$f")
    done
    echo "${models[@]}"
}

discover_mlx_models() {
    # --- TinyLlama ---
    # Q8_0 (HF cache or local)
    local tinyllama_dir
    tinyllama_dir="$(ls -d "${HOME}/.cache/huggingface/hub/models--steamdroid--TinyLlama-1.1B-Chat-v1.0-mlx-8Bit/snapshots"/*/ 2>/dev/null | head -1)"
    if [[ -n "$tinyllama_dir" && -d "$tinyllama_dir" ]]; then
        mlx_add "tinyllama-1.1b-q8_0" "$tinyllama_dir"
    elif [[ -d "${BENCH_DIR}/tinyllama-1.1b-mlx-8bit" ]]; then
        mlx_add "tinyllama-1.1b-q8_0" "${BENCH_DIR}/tinyllama-1.1b-mlx-8bit"
    fi
    # F16
    if [[ -d "${BENCH_DIR}/tinyllama-1.1b-mlx-f16" ]]; then
        mlx_add "tinyllama-1.1b-f16" "${BENCH_DIR}/tinyllama-1.1b-mlx-f16"
    fi
    # Q4_0 (MLX 4-bit)
    if [[ -d "${BENCH_DIR}/tinyllama-1.1b-mlx-4bit" ]]; then
        mlx_add "tinyllama-1.1b-q4_0" "${BENCH_DIR}/tinyllama-1.1b-mlx-4bit"
    fi

    # --- Llama 3.1 8B ---
    # Q8_0 (HF cache or local)
    local llama8b_dir
    llama8b_dir="$(ls -d "${HOME}/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-8bit/snapshots"/*/ 2>/dev/null | head -1)"
    if [[ -n "$llama8b_dir" && -d "$llama8b_dir" ]]; then
        mlx_add "llama-3.1-8b-q8_0" "$llama8b_dir"
    elif [[ -d "${BENCH_DIR}/llama-3.1-8b-mlx-8bit" ]]; then
        mlx_add "llama-3.1-8b-q8_0" "${BENCH_DIR}/llama-3.1-8b-mlx-8bit"
    fi
    # F16
    if [[ -d "${BENCH_DIR}/llama-3.1-8b-mlx-f16" ]]; then
        # Only add if model weights are complete (at least 4 safetensors shards)
        local shard_count
        shard_count="$(find "${BENCH_DIR}/llama-3.1-8b-mlx-f16" -name '*.safetensors' -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$shard_count" -ge 4 ]]; then
            mlx_add "llama-3.1-8b-f16" "${BENCH_DIR}/llama-3.1-8b-mlx-f16"
        fi
    fi
    # Q4_0 (MLX 4-bit)
    if [[ -d "${BENCH_DIR}/llama-3.1-8b-mlx-4bit" ]]; then
        local shard_count
        shard_count="$(find "${BENCH_DIR}/llama-3.1-8b-mlx-4bit" -name '*.safetensors' -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$shard_count" -ge 1 ]]; then
            mlx_add "llama-3.1-8b-q4_0" "${BENCH_DIR}/llama-3.1-8b-mlx-4bit"
        fi
    fi

    # --- Qwen3.5-9B ---
    # Q8_0
    if [[ -d "${BENCH_DIR}/qwen35-9b-mlx-8bit" ]]; then
        mlx_add "qwen35-9b-q8_0" "${BENCH_DIR}/qwen35-9b-mlx-8bit"
    fi
    # F16
    if [[ -d "${BENCH_DIR}/qwen35-9b-mlx-f16" ]]; then
        mlx_add "qwen35-9b-f16" "${BENCH_DIR}/qwen35-9b-mlx-f16"
    fi
    # Q4_0 (MLX 4-bit)
    if [[ -d "${BENCH_DIR}/qwen35-9b-mlx-4bit" ]]; then
        mlx_add "qwen35-9b-q4_0" "${BENCH_DIR}/qwen35-9b-mlx-4bit"
    fi
}

discover_llamacpp_models() {
    # Only discover if llama-bench binary exists
    if [[ ! -x "$LLAMA_BENCH_BIN" ]]; then
        log "  llama-bench not found at ${LLAMA_BENCH_BIN}, skipping llama.cpp"
        SKIP_LLAMACPP=1
        return
    fi

    # --- TinyLlama ---
    local f
    # Q8_0
    f="${BENCH_DIR}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    [[ -f "$f" ]] && llama_add "tinyllama-1.1b-q8_0" "$f"
    # F16
    f="${BENCH_DIR}/TinyLlama-1.1B-Chat-v1.0-f16.gguf"
    [[ -f "$f" ]] && llama_add "tinyllama-1.1b-f16" "$f"
    # Q4_0
    f="${BENCH_DIR}/TinyLlama-1.1B-Chat-v1.0-Q4_0.gguf"
    [[ -f "$f" ]] && llama_add "tinyllama-1.1b-q4_0" "$f"

    # --- Llama 3.1 8B ---
    # Q8_0
    f="${BENCH_DIR}/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"
    [[ -f "$f" ]] && llama_add "llama-3.1-8b-q8_0" "$f"
    # F16
    f="${BENCH_DIR}/Llama-3.1-8B-Instruct-f16.gguf"
    [[ -f "$f" ]] && llama_add "llama-3.1-8b-f16" "$f"
    # Q4_0
    f="${BENCH_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf"
    [[ -f "$f" ]] && llama_add "llama-3.1-8b-q4_0" "$f"

    # Qwen3.5-9B: UNSUPPORTED -- GDN layers fall back to CPU (no Metal kernel)
    LLAMA_UNSUPPORTED+=("qwen35-9b-q8_0")
    LLAMA_UNSUPPORTED+=("qwen35-9b-q4_0")
    LLAMA_UNSUPPORTED+=("qwen35-9b-f16")
}

# Extract human-readable model name and quant from LBC filename
# e.g. "tinyllama-1.1b-q8_0" -> "TinyLlama 1.1B Q8_0"
# Output: two fields separated by tab -- "model_name\tquant"
parse_model_name() {
    local bn="$1"
    local name quant

    # Extract quant suffix (last -q..._... or -f16 segment)
    if echo "$bn" | grep -qE -- '-[qQ][0-9]+_[0-9a-zA-Z]+$'; then
        quant="$(echo "$bn" | grep -oE -- '[qQ][0-9]+_[0-9a-zA-Z]+$')"
        name="${bn%-"$quant"}"
    elif echo "$bn" | grep -qE -- '-[fF]16$'; then
        quant="F16"
        name="${bn%-[fF]16}"
    else
        quant="unknown"
        name="$bn"
    fi

    # Prettify common model names
    case "$name" in
        tinyllama-1.1b)  name="TinyLlama 1.1B" ;;
        llama-3.1-8b)    name="Llama 3.1 8B" ;;
        llama-3.2-1b)    name="Llama 3.2 1B" ;;
        qwen25-0.5b)     name="Qwen2.5 0.5B" ;;
        qwen35-9b)       name="Qwen3.5 9B" ;;
        qwen35-35b-a3b)  name="Qwen3.5 35B-A3B" ;;
        *)               ;; # keep raw name
    esac

    quant="$(echo "$quant" | tr '[:lower:]' '[:upper:]')"
    printf '%s\t%s' "$name" "$quant"
}

# ============================================================================
# Generate synthetic token IDs for a given prompt length
# ============================================================================

generate_tokens() {
    local count="$1"
    # Use seq for efficiency (avoids bash loop for large counts)
    seq -s ' ' 1 "$count"
}

# ============================================================================
# Lumen benchmark runner
# ============================================================================

# run_lumen_single: runs one Lumen inference and extracts prefill/decode tok/s
# Args: $1=lbc_path $2=prompt_len $3=gen_len $4=raw_output_file
# Outputs: writes "prefill_tps decode_tps" to stdout, raw output to file
run_lumen_single() {
    local lbc_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local raw_file="$4"

    local tokens
    tokens="$(generate_tokens "$prompt_len")"

    # Context length: prompt + gen + headroom
    local context_len=$(( prompt_len + gen_len + 256 ))

    local output
    if ! output=$("$LUMEN_BIN" run \
        --model "$lbc_path" \
        --metal \
        --tokens "$tokens" \
        --max-tokens "$gen_len" \
        --temperature 0 \
        --context-len "$context_len" \
        2>&1); then
        echo "$output" > "$raw_file"
        echo "FAIL FAIL"
        return 1
    fi

    echo "$output" > "$raw_file"

    # Parse: "Prefill: 4618.3 tok/s (27.7ms)"
    local prefill_tps
    prefill_tps="$(echo "$output" | grep -oE 'Prefill:[[:space:]]+[0-9]+(\.[0-9]+)? tok/s' | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"

    # Parse: "Decode:  291.4 tok/s, TPOT: 3.4ms"
    local decode_tps
    decode_tps="$(echo "$output" | grep -oE 'Decode:[[:space:]]+[0-9]+(\.[0-9]+)? tok/s' | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"

    if [[ -z "$prefill_tps" || -z "$decode_tps" ]]; then
        echo "FAIL FAIL"
        return 1
    fi

    echo "$prefill_tps $decode_tps"
}

# run_lumen_bench: runs warmup + measured runs for one (model, pp, gen) config
# Args: $1=lbc_path $2=prompt_len $3=gen_len $4=basename
# Outputs: "prefill_median prefill_stddev decode_median decode_stddev"
run_lumen_bench() {
    local lbc_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local bn="$4"

    local config_label="pp${prompt_len}_gen${gen_len}"
    local prefill_vals=()
    local decode_vals=()

    # Warmup runs (discarded)
    local w
    for ((w = 1; w <= WARMUP_RUNS; w++)); do
        log "    [warmup $w/$WARMUP_RUNS] ${config_label}"
        local raw_file="${RAW_DIR}/${bn}_${config_label}_warmup${w}.txt"
        run_lumen_single "$lbc_path" "$prompt_len" "$gen_len" "$raw_file" > /dev/null 2>&1 || true
    done

    # Measured runs
    local failures=0
    local r
    for ((r = 1; r <= RUNS; r++)); do
        log "    [run $r/$RUNS] ${config_label}"
        local raw_file="${RAW_DIR}/${bn}_${config_label}_run${r}.txt"
        local result
        result="$(run_lumen_single "$lbc_path" "$prompt_len" "$gen_len" "$raw_file" 2>/dev/null)" || true

        local pp_val dec_val
        pp_val="$(echo "$result" | awk '{print $1}')"
        dec_val="$(echo "$result" | awk '{print $2}')"

        if [[ "$pp_val" == "FAIL" || -z "$pp_val" ]]; then
            log "      WARN: run $r failed, skipping"
            failures=$((failures + 1))
            continue
        fi

        prefill_vals+=("$pp_val")
        decode_vals+=("$dec_val")
    done

    # Need at least 1 successful run
    if [[ ${#prefill_vals[@]} -eq 0 ]]; then
        echo "FAIL 0 FAIL 0"
        return 1
    fi

    # Compute median and stddev
    local pp_stats dec_stats
    pp_stats="$(printf '%s\n' "${prefill_vals[@]}" | compute_stats)"
    dec_stats="$(printf '%s\n' "${decode_vals[@]}" | compute_stats)"

    local pp_median pp_stddev dec_median dec_stddev
    pp_median="$(echo "$pp_stats" | awk '{print $1}')"
    pp_stddev="$(echo "$pp_stats" | awk '{print $2}')"
    dec_median="$(echo "$dec_stats" | awk '{print $1}')"
    dec_stddev="$(echo "$dec_stats" | awk '{print $2}')"

    if [[ "$failures" -gt 0 ]]; then
        log "      NOTE: $failures/$RUNS runs failed, stats from ${#prefill_vals[@]} runs"
    fi

    echo "$pp_median $pp_stddev $dec_median $dec_stddev"
}

# ============================================================================
# MLX benchmark runner
# ============================================================================

# run_mlx_single: runs one MLX benchmark and extracts prefill/decode tok/s
# Args: $1=mlx_path $2=prompt_len $3=gen_len $4=raw_output_file
run_mlx_single() {
    local mlx_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local raw_file="$4"

    local output
    # Use -n 5 to let MLX warm up internally (1 warmup + 5 measured trials).
    # We take the Averages line which gives the mean of 5 measured trials.
    if ! output=$(source "${MLX_VENV}/bin/activate" && python -m mlx_lm.benchmark \
        --model "$mlx_path" \
        -p "$prompt_len" \
        -g "$gen_len" \
        -n 5 \
        2>&1); then
        echo "$output" > "$raw_file"
        echo "FAIL FAIL"
        return 1
    fi

    echo "$output" > "$raw_file"

    # Parse: "prompt_tps=887.748, generation_tps=76.946"
    # or line: "Averages: prompt_tps=887.748, generation_tps=76.946, peak_memory=8.698"
    local prefill_tps
    prefill_tps="$(echo "$output" | grep -oE 'prompt_tps=[0-9]+(\.[0-9]+)?' | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)"

    local decode_tps
    decode_tps="$(echo "$output" | grep -oE 'generation_tps=[0-9]+(\.[0-9]+)?' | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)"

    if [[ -z "$prefill_tps" || -z "$decode_tps" ]]; then
        echo "FAIL FAIL"
        return 1
    fi

    echo "$prefill_tps $decode_tps"
}

# run_mlx_bench: runs warmup + measured runs for one (model, pp, gen) config
# Args: $1=mlx_path $2=prompt_len $3=gen_len $4=basename
# Outputs: "prefill_median prefill_stddev decode_median decode_stddev"
run_mlx_bench() {
    local mlx_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local bn="$4"

    local config_label="pp${prompt_len}_gen${gen_len}"
    local prefill_vals=()
    local decode_vals=()

    # Warmup runs (discarded)
    local w
    for ((w = 1; w <= WARMUP_RUNS; w++)); do
        log "    [warmup $w/$WARMUP_RUNS] MLX ${config_label}"
        local raw_file="${RAW_DIR}/mlx_${bn}_${config_label}_warmup${w}.txt"
        run_mlx_single "$mlx_path" "$prompt_len" "$gen_len" "$raw_file" > /dev/null 2>&1 || true
    done

    # Measured runs
    local failures=0
    local r
    for ((r = 1; r <= RUNS; r++)); do
        log "    [run $r/$RUNS] MLX ${config_label}"
        local raw_file="${RAW_DIR}/mlx_${bn}_${config_label}_run${r}.txt"
        local result
        result="$(run_mlx_single "$mlx_path" "$prompt_len" "$gen_len" "$raw_file" 2>/dev/null)" || true

        local pp_val dec_val
        pp_val="$(echo "$result" | awk '{print $1}')"
        dec_val="$(echo "$result" | awk '{print $2}')"

        if [[ "$pp_val" == "FAIL" || -z "$pp_val" ]]; then
            log "      WARN: MLX run $r failed, skipping"
            failures=$((failures + 1))
            continue
        fi

        prefill_vals+=("$pp_val")
        decode_vals+=("$dec_val")
    done

    if [[ ${#prefill_vals[@]} -eq 0 ]]; then
        echo "FAIL 0 FAIL 0"
        return 1
    fi

    local pp_stats dec_stats
    pp_stats="$(printf '%s\n' "${prefill_vals[@]}" | compute_stats)"
    dec_stats="$(printf '%s\n' "${decode_vals[@]}" | compute_stats)"

    local pp_median pp_stddev dec_median dec_stddev
    pp_median="$(echo "$pp_stats" | awk '{print $1}')"
    pp_stddev="$(echo "$pp_stats" | awk '{print $2}')"
    dec_median="$(echo "$dec_stats" | awk '{print $1}')"
    dec_stddev="$(echo "$dec_stats" | awk '{print $2}')"

    if [[ "$failures" -gt 0 ]]; then
        log "      NOTE: $failures/$RUNS MLX runs failed, stats from ${#prefill_vals[@]} runs"
    fi

    echo "$pp_median $pp_stddev $dec_median $dec_stddev"
}

# ============================================================================
# llama.cpp benchmark runner
# ============================================================================

# run_llamacpp_single: runs one llama-bench invocation and extracts prefill/decode tok/s
# Args: $1=gguf_path $2=prompt_len $3=gen_len $4=raw_output_file
run_llamacpp_single() {
    local gguf_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local raw_file="$4"

    local output
    if ! output=$("$LLAMA_BENCH_BIN" \
        -m "$gguf_path" \
        -p "$prompt_len" \
        -n "$gen_len" \
        -ngl 99 \
        -t 1 \
        -r 1 \
        -o md \
        2>/dev/null); then
        echo "$output" > "$raw_file"
        echo "FAIL FAIL"
        return 1
    fi

    echo "$output" > "$raw_file"

    # Parse llama-bench md output:
    # | model | size | params | backend | threads | test | t/s |
    # | llama 8B Q8_0 | 7.95 GiB | 8.03 B | MTL,BLAS | 1 | pp128 | 971.62 ± 0.00 |
    local prefill_tps
    prefill_tps="$(echo "$output" | grep "pp${prompt_len}" | awk -F'|' '{print $8}' | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"

    local decode_tps
    decode_tps="$(echo "$output" | grep "tg${gen_len}" | awk -F'|' '{print $8}' | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)"

    if [[ -z "$prefill_tps" || -z "$decode_tps" ]]; then
        echo "FAIL FAIL"
        return 1
    fi

    echo "$prefill_tps $decode_tps"
}

# run_llamacpp_bench: runs warmup + measured runs for one (model, pp, gen) config
# Args: $1=gguf_path $2=prompt_len $3=gen_len $4=basename
# Outputs: "prefill_median prefill_stddev decode_median decode_stddev"
run_llamacpp_bench() {
    local gguf_path="$1"
    local prompt_len="$2"
    local gen_len="$3"
    local bn="$4"

    local config_label="pp${prompt_len}_gen${gen_len}"
    local prefill_vals=()
    local decode_vals=()

    # Warmup runs (discarded)
    local w
    for ((w = 1; w <= WARMUP_RUNS; w++)); do
        log "    [warmup $w/$WARMUP_RUNS] llama.cpp ${config_label}"
        local raw_file="${RAW_DIR}/llamacpp_${bn}_${config_label}_warmup${w}.txt"
        run_llamacpp_single "$gguf_path" "$prompt_len" "$gen_len" "$raw_file" > /dev/null 2>&1 || true
    done

    # Measured runs
    local failures=0
    local r
    for ((r = 1; r <= RUNS; r++)); do
        log "    [run $r/$RUNS] llama.cpp ${config_label}"
        local raw_file="${RAW_DIR}/llamacpp_${bn}_${config_label}_run${r}.txt"
        local result
        result="$(run_llamacpp_single "$gguf_path" "$prompt_len" "$gen_len" "$raw_file" 2>/dev/null)" || true

        local pp_val dec_val
        pp_val="$(echo "$result" | awk '{print $1}')"
        dec_val="$(echo "$result" | awk '{print $2}')"

        if [[ "$pp_val" == "FAIL" || -z "$pp_val" ]]; then
            log "      WARN: llama.cpp run $r failed, skipping"
            failures=$((failures + 1))
            continue
        fi

        prefill_vals+=("$pp_val")
        decode_vals+=("$dec_val")
    done

    if [[ ${#prefill_vals[@]} -eq 0 ]]; then
        echo "FAIL 0 FAIL 0"
        return 1
    fi

    local pp_stats dec_stats
    pp_stats="$(printf '%s\n' "${prefill_vals[@]}" | compute_stats)"
    dec_stats="$(printf '%s\n' "${decode_vals[@]}" | compute_stats)"

    local pp_median pp_stddev dec_median dec_stddev
    pp_median="$(echo "$pp_stats" | awk '{print $1}')"
    pp_stddev="$(echo "$pp_stats" | awk '{print $2}')"
    dec_median="$(echo "$dec_stats" | awk '{print $1}')"
    dec_stddev="$(echo "$dec_stats" | awk '{print $2}')"

    if [[ "$failures" -gt 0 ]]; then
        log "      NOTE: $failures/$RUNS llama.cpp runs failed, stats from ${#prefill_vals[@]} runs"
    fi

    echo "$pp_median $pp_stddev $dec_median $dec_stddev"
}

# ============================================================================
# JSON output generator
# ============================================================================

generate_json() {
    local json_file="$1"
    local results_tmp="$2"

    python3 << PYEOF
import json

results = []
with open('${results_tmp}', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 10:
            continue
        entry = {
            'engine': parts[0],
            'model_file': parts[1],
            'model_name': parts[2],
            'quant': parts[3],
            'prompt_len': int(parts[4]),
            'gen_len': int(parts[5]),
            'prefill_tps_median': float(parts[6]) if parts[6] not in ('FAIL', 'UNSUPPORTED') else None,
            'prefill_tps_stddev': float(parts[7]) if parts[6] not in ('FAIL', 'UNSUPPORTED') else None,
            'decode_tps_median': float(parts[8]) if parts[8] not in ('FAIL', 'UNSUPPORTED') else None,
            'decode_tps_stddev': float(parts[9]) if parts[8] not in ('FAIL', 'UNSUPPORTED') else None,
            'unsupported': parts[6] == 'UNSUPPORTED',
        }
        results.append(entry)

output = {
    'timestamp': '${TIMESTAMP}',
    'hardware': '${HARDWARE_DESC}',
    'runs_per_config': ${RUNS},
    'warmup_runs': ${WARMUP_RUNS},
    'prompt_lengths': [$(printf '%s,' "${PROMPT_LENGTHS[@]}" | sed 's/,$//')],
    'gen_lengths': [$(printf '%s,' "${GEN_LENGTHS[@]}" | sed 's/,$//')],
    'results': results,
}

with open('${json_file}', 'w') as f:
    json.dump(output, f, indent=2)
import sys as _sys
print('JSON written to ${json_file}', file=_sys.stderr)
PYEOF
}

# ============================================================================
# Markdown table generator
# ============================================================================

generate_markdown() {
    local md_file="$1"
    local results_tmp="$2"

    python3 << PYEOF
import sys
from collections import defaultdict

md_file = "${md_file}"
results_tmp = "${results_tmp}"
prompt_lengths = [$(printf '%s,' "${PROMPT_LENGTHS[@]}" | sed 's/,$//')]
gen_lengths = [$(printf '%s,' "${GEN_LENGTHS[@]}" | sed 's/,$//')]
runs = ${RUNS}
warmup = ${WARMUP_RUNS}
timestamp = "${TIMESTAMP}"
lumen_only = (${LUMEN_ONLY} == 1)
skip_llamacpp = (${SKIP_LLAMACPP} == 1)

# Parse results
# Format: engine|model_file|model_name|quant|prompt_len|gen_len|pp_med|pp_sd|dec_med|dec_sd
lumen_data = {}      # (model_name, quant) -> {(pp, gen) -> (pp_med, pp_sd, dec_med, dec_sd)}
mlx_data = {}        # (model_name, quant) -> {(pp, gen) -> (pp_med, pp_sd, dec_med, dec_sd)}
llamacpp_data = {}   # (model_name, quant) -> {(pp, gen) -> (pp_med, pp_sd, dec_med, dec_sd)}

with open(results_tmp, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 10:
            continue
        engine = parts[0]
        model_name = parts[2]
        quant = parts[3]
        pl = int(parts[4])
        gl = int(parts[5])
        pp_med = parts[6]
        pp_sd = parts[7]
        dec_med = parts[8]
        dec_sd = parts[9]

        key = (model_name, quant)
        config = (pl, gl)

        if engine == "lumen":
            store = lumen_data
        elif engine == "mlx":
            store = mlx_data
        elif engine == "llamacpp":
            store = llamacpp_data
        else:
            continue
        if key not in store:
            store[key] = {}
        store[key][config] = (pp_med, pp_sd, dec_med, dec_sd)

def fmt_val(med, sd):
    """Format as 'median+-stddev' or '--'."""
    if med == "FAIL":
        return "--"
    med_f = float(med)
    sd_f = float(sd)
    if med_f >= 100:
        return "{:.0f}+-{:.0f}".format(med_f, sd_f)
    else:
        return "{:.1f}+-{:.1f}".format(med_f, sd_f)

def fmt_med(med):
    """Format median only, no stddev."""
    if med == "FAIL":
        return "--"
    if med == "UNSUPPORTED":
        return "n/a"
    med_f = float(med)
    if med_f >= 100:
        return "{:.0f}".format(med_f)
    else:
        return "{:.1f}".format(med_f)

def fmt_ratio(lumen_med, other_med):
    """Format Lumen/other ratio. >1 = Lumen faster, <1 = Lumen slower."""
    if lumen_med == "UNSUPPORTED" or other_med == "UNSUPPORTED":
        return "n/a"
    if lumen_med == "FAIL" or other_med == "FAIL":
        return "--"
    l = float(lumen_med)
    m = float(other_med)
    if m == 0:
        return "--"
    ratio = l / m
    if ratio >= 1.0:
        return "**{:.2f}x**".format(ratio)
    else:
        return "{:.2f}x".format(ratio)

# Apples-to-apples: only compare same (model_name, quant) across engines.
# mlx_data and llamacpp_data are already keyed by (model_name, quant).

has_mlx = bool(mlx_data) and not lumen_only
has_llamacpp = bool(llamacpp_data) and not lumen_only and not skip_llamacpp

def has_real_baseline(mn, q):
    """True if at least one baseline engine has real (non-UNSUPPORTED) data for same quant."""
    if (mn, q) in mlx_data:
        return True
    if (mn, q) in llamacpp_data:
        cfgs = llamacpp_data[(mn, q)]
        if any(v[0] != "UNSUPPORTED" for v in cfgs.values()):
            return True
    return False

lines = []
engine_names = ["Lumen"]
if has_mlx:
    engine_names.append("MLX")
if has_llamacpp:
    engine_names.append("llama.cpp")
title_engines = " vs ".join(engine_names)

lines.append("# {} Benchmark ({})".format(title_engines, timestamp))
lines.append("")
lines.append("**Hardware**: ${HARDWARE_DESC}")
method_parts = ["{} measured runs + {} warmup per config".format(runs, warmup)]
if has_mlx:
    method_parts.append("MLX runs first (cold GPU), MLX uses 5 internal trials")
lines.append("**Methodology**: {}".format(", ".join(method_parts)))
lines.append("**Configs**: prompt lengths {} x generation lengths {}".format(
    ', '.join(str(p) for p in prompt_lengths),
    ', '.join(str(g) for g in gen_lengths)))
lines.append("")
lines.append("> Ratio = Lumen / baseline. Values >1.00 mean Lumen is faster (bold). Values <1.00 mean Lumen is slower.")
if has_llamacpp:
    lines.append("> n/a = model not supported by that engine on Metal GPU.")
lines.append("")

if not lumen_only and (has_mlx or has_llamacpp):
    # ================================================================
    # HEADLINE: Summary table (pp128+gen128, the canonical config)
    # ================================================================
    lines.append("## Summary (pp128 + gen128)")
    lines.append("")

    # Build header dynamically based on available engines
    hdr = "| Model | Quant | Lumen Decode"
    sep_h = "|-------|:-----:|-------------:"
    if has_mlx:
        hdr += " | MLX Decode | vs MLX"
        sep_h += "|-----------:|:-----:"
    if has_llamacpp:
        hdr += " | LC Decode | vs LC"
        sep_h += "|----------:|:----:"
    hdr += " | Lumen Prefill"
    sep_h += "|--------------:"
    if has_mlx:
        hdr += " | MLX Prefill | vs MLX"
        sep_h += "|------------:|:-----:"
    if has_llamacpp:
        hdr += " | LC Prefill | vs LC"
        sep_h += "|-----------:|:----:"
    hdr += " |"
    sep_h += "|"
    lines.append(hdr)
    lines.append(sep_h)

    canonical = (128, 128)
    for (model_name, quant) in sorted(lumen_data.keys()):
        lumen_configs = lumen_data[(model_name, quant)]
        if canonical not in lumen_configs:
            continue
        # Apples-to-apples: only match same (model_name, quant) in baselines
        if not has_real_baseline(model_name, quant):
            continue
        mlx_configs = mlx_data.get((model_name, quant), {})
        lc_configs = llamacpp_data.get((model_name, quant), {})

        l_pp, _, l_dec, _ = lumen_configs[canonical]
        row = "| {} | {} | {}".format(model_name, quant, fmt_med(l_dec))

        if has_mlx:
            if canonical in mlx_configs:
                _, _, m_dec, _ = mlx_configs[canonical]
                row += " | {} | {}".format(fmt_med(m_dec), fmt_ratio(l_dec, m_dec))
            else:
                row += " | -- | --"
        if has_llamacpp:
            if canonical in lc_configs:
                _, _, lc_dec, _ = lc_configs[canonical]
                row += " | {} | {}".format(fmt_med(lc_dec), fmt_ratio(l_dec, lc_dec))
            else:
                row += " | n/a | n/a"

        row += " | {}".format(fmt_med(l_pp))
        if has_mlx:
            if canonical in mlx_configs:
                m_pp, _, _, _ = mlx_configs[canonical]
                row += " | {} | {}".format(fmt_med(m_pp), fmt_ratio(l_pp, m_pp))
            else:
                row += " | -- | --"
        if has_llamacpp:
            if canonical in lc_configs:
                lc_pp, _, _, _ = lc_configs[canonical]
                row += " | {} | {}".format(fmt_med(lc_pp), fmt_ratio(l_pp, lc_pp))
            else:
                row += " | n/a | n/a"

        row += " |"
        lines.append(row)

    lines.append("")

    # ================================================================
    # DECODE: Detail across all configs, per model
    # ================================================================
    lines.append("## Decode: {} (tok/s)".format(title_engines))
    lines.append("")

    config_labels = []
    for pl in prompt_lengths:
        for gl in gen_lengths:
            config_labels.append("pp{}+g{}".format(pl, gl))

    header = "| Model | Quant |"
    sep = "|-------|:-----:|"
    for label in config_labels:
        header += " {} |".format(label)
        sep += "------:|"
    lines.append(header)
    lines.append(sep)

    for (model_name, quant) in sorted(lumen_data.keys()):
        # Apples-to-apples: skip rows with no same-quant baseline
        if not has_real_baseline(model_name, quant):
            continue
        lumen_configs = lumen_data[(model_name, quant)]
        mlx_configs = mlx_data.get((model_name, quant), {})
        lc_configs = llamacpp_data.get((model_name, quant), {})

        # Lumen row
        row_l = "| {} | {} |".format(model_name, quant)
        for pl in prompt_lengths:
            for gl in gen_lengths:
                cfg = (pl, gl)
                if cfg in lumen_configs:
                    _, _, l_dec, _ = lumen_configs[cfg]
                    row_l += " {} |".format(fmt_med(l_dec))
                else:
                    row_l += " -- |"
        lines.append(row_l)

        # MLX sub-row (same quant)
        if has_mlx and (model_name, quant) in mlx_data:
            row_m = "| ^(MLX {}) | |".format(quant)
            row_r = "| ^(vs MLX) | |"
            for pl in prompt_lengths:
                for gl in gen_lengths:
                    cfg = (pl, gl)
                    if cfg in mlx_configs:
                        _, _, m_dec, _ = mlx_configs[cfg]
                        row_m += " {} |".format(fmt_med(m_dec))
                        if cfg in lumen_configs:
                            _, _, l_dec, _ = lumen_configs[cfg]
                            row_r += " {} |".format(fmt_ratio(l_dec, m_dec))
                        else:
                            row_r += " -- |"
                    else:
                        row_m += " -- |"
                        row_r += " -- |"
            lines.append(row_m)
            lines.append(row_r)

        # llama.cpp sub-row (same quant)
        if has_llamacpp and (model_name, quant) in llamacpp_data:
            is_unsup = any(v[2] == "UNSUPPORTED" for v in lc_configs.values())
            if is_unsup:
                row_lc = "| ^(llama.cpp) | |"
                row_lr = "| ^(vs LC) | |"
                for _ in config_labels:
                    row_lc += " n/a |"
                    row_lr += " n/a |"
            else:
                row_lc = "| ^(llama.cpp {}) | |".format(quant)
                row_lr = "| ^(vs LC) | |"
                for pl in prompt_lengths:
                    for gl in gen_lengths:
                        cfg = (pl, gl)
                        if cfg in lc_configs:
                            _, _, lc_dec, _ = lc_configs[cfg]
                            row_lc += " {} |".format(fmt_med(lc_dec))
                            if cfg in lumen_configs:
                                _, _, l_dec, _ = lumen_configs[cfg]
                                row_lr += " {} |".format(fmt_ratio(l_dec, lc_dec))
                            else:
                                row_lr += " -- |"
                        else:
                            row_lc += " -- |"
                            row_lr += " -- |"
            lines.append(row_lc)
            lines.append(row_lr)

        lines.append("|  |  |" + " |" * len(config_labels))

    lines.append("")

    # ================================================================
    # PREFILL: Detail across all configs, per model
    # ================================================================
    lines.append("## Prefill: {} (tok/s)".format(title_engines))
    lines.append("")

    lines.append(header)
    lines.append(sep)

    for (model_name, quant) in sorted(lumen_data.keys()):
        # Apples-to-apples: skip rows with no same-quant baseline
        if not has_real_baseline(model_name, quant):
            continue
        lumen_configs = lumen_data[(model_name, quant)]
        mlx_configs = mlx_data.get((model_name, quant), {})
        lc_configs = llamacpp_data.get((model_name, quant), {})

        # Lumen row
        row_l = "| {} | {} |".format(model_name, quant)
        for pl in prompt_lengths:
            for gl in gen_lengths:
                cfg = (pl, gl)
                if cfg in lumen_configs:
                    l_pp, _, _, _ = lumen_configs[cfg]
                    row_l += " {} |".format(fmt_med(l_pp))
                else:
                    row_l += " -- |"
        lines.append(row_l)

        # MLX sub-row (same quant)
        if has_mlx and (model_name, quant) in mlx_data:
            row_m = "| ^(MLX {}) | |".format(quant)
            row_r = "| ^(vs MLX) | |"
            for pl in prompt_lengths:
                for gl in gen_lengths:
                    cfg = (pl, gl)
                    if cfg in mlx_configs:
                        m_pp, _, _, _ = mlx_configs[cfg]
                        row_m += " {} |".format(fmt_med(m_pp))
                        if cfg in lumen_configs:
                            l_pp, _, _, _ = lumen_configs[cfg]
                            row_r += " {} |".format(fmt_ratio(l_pp, m_pp))
                        else:
                            row_r += " -- |"
                    else:
                        row_m += " -- |"
                        row_r += " -- |"
            lines.append(row_m)
            lines.append(row_r)

        # llama.cpp sub-row (same quant)
        if has_llamacpp and (model_name, quant) in llamacpp_data:
            is_unsup = any(v[0] == "UNSUPPORTED" for v in lc_configs.values())
            if is_unsup:
                row_lc = "| ^(llama.cpp) | |"
                row_lr = "| ^(vs LC) | |"
                for _ in config_labels:
                    row_lc += " n/a |"
                    row_lr += " n/a |"
            else:
                row_lc = "| ^(llama.cpp {}) | |".format(quant)
                row_lr = "| ^(vs LC) | |"
                for pl in prompt_lengths:
                    for gl in gen_lengths:
                        cfg = (pl, gl)
                        if cfg in lc_configs:
                            lc_pp, _, _, _ = lc_configs[cfg]
                            row_lc += " {} |".format(fmt_med(lc_pp))
                            if cfg in lumen_configs:
                                l_pp, _, _, _ = lumen_configs[cfg]
                                row_lr += " {} |".format(fmt_ratio(l_pp, lc_pp))
                            else:
                                row_lr += " -- |"
                        else:
                            row_lc += " -- |"
                            row_lr += " -- |"
            lines.append(row_lc)
            lines.append(row_lr)

        lines.append("|  |  |" + " |" * len(config_labels))

    lines.append("")

elif lumen_only:
    # ================================================================
    # Lumen-only mode: show all results without MLX comparison
    # ================================================================
    lines.append("## Decode (tok/s)")
    lines.append("")

    config_labels = []
    for pl in prompt_lengths:
        for gl in gen_lengths:
            config_labels.append("pp{}+g{}".format(pl, gl))

    header = "| Model | Quant |"
    sep = "|-------|:-----:|"
    for label in config_labels:
        header += " {} |".format(label)
        sep += "------:|"
    lines.append(header)
    lines.append(sep)

    for (model_name, quant) in sorted(lumen_data.keys()):
        lumen_configs = lumen_data[(model_name, quant)]
        row = "| {} | {} |".format(model_name, quant)
        for pl in prompt_lengths:
            for gl in gen_lengths:
                cfg = (pl, gl)
                val = "--"
                if cfg in lumen_configs:
                    _, _, dec_med, dec_sd = lumen_configs[cfg]
                    val = fmt_val(dec_med, dec_sd)
                row += " {} |".format(val)
        lines.append(row)

    lines.append("")
    lines.append("## Prefill (tok/s)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for (model_name, quant) in sorted(lumen_data.keys()):
        lumen_configs = lumen_data[(model_name, quant)]
        row = "| {} | {} |".format(model_name, quant)
        for pl in prompt_lengths:
            for gl in gen_lengths:
                cfg = (pl, gl)
                val = "--"
                if cfg in lumen_configs:
                    pp_med, pp_sd, _, _ = lumen_configs[cfg]
                    val = fmt_val(pp_med, pp_sd)
                row += " {} |".format(val)
        lines.append(row)

    lines.append("")

# ================================================================
# Models without any baseline comparison (Lumen-only entries)
# ================================================================
lumen_only_models = [(mn, q) for (mn, q) in sorted(lumen_data.keys()) if not has_real_baseline(mn, q)]
if lumen_only_models and not lumen_only:
    lines.append("## Lumen Only (no baseline)")
    lines.append("")
    lines.append("| Model | Quant | Config | Prefill (tok/s) | Decode (tok/s) |")
    lines.append("|-------|:-----:|--------|----------------:|---------------:|")
    for (model_name, quant) in lumen_only_models:
        configs = lumen_data[(model_name, quant)]
        for (pl, gl) in sorted(configs.keys()):
            pp_med, pp_sd, dec_med, dec_sd = configs[(pl, gl)]
            lines.append("| {} | {} | pp{}+gen{} | {} | {} |".format(
                model_name, quant, pl, gl, fmt_val(pp_med, pp_sd), fmt_val(dec_med, dec_sd)))
    lines.append("")

output = "\n".join(lines) + "\n"

with open(md_file, 'w') as f:
    f.write(output)

# Print to stderr so it appears on the terminal (stdout is for piping)
import sys as _sys
print(output, file=_sys.stderr)
PYEOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_section "Lumen Benchmark Suite"
    log "Timestamp:        ${TIMESTAMP}"
    log "Results dir:      ${RESULTS_DIR}"
    log "Bench dir:        ${BENCH_DIR}"
    log "Prompt lengths:   ${PROMPT_LENGTHS[*]}"
    log "Gen lengths:      ${GEN_LENGTHS[*]}"
    log "Runs per config:  ${RUNS}"
    log "Warmup runs:      ${WARMUP_RUNS}"
    log "Cooldown (secs):  ${COOLDOWN_SECS} (model), ${CONFIG_COOLDOWN_SECS} (config)"
    log "Lumen only:       ${LUMEN_ONLY}"
    log "Quick mode:       ${QUICK_MODE}"
    echo "" >&2

    # ------------------------------------------------------------------
    # Step 1: Build Lumen
    # ------------------------------------------------------------------
    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        log_section "Building Lumen (cargo build --release)"
        if ! (cd "$REPO_ROOT" && cargo build --release -p lumen-cli 2>&1); then
            echo "ERROR: cargo build failed" >&2
            exit 1
        fi
        log "Build complete: ${LUMEN_BIN}"
    else
        log "Skipping build (--no-build)"
    fi

    if [[ ! -x "$LUMEN_BIN" ]]; then
        echo "ERROR: Lumen binary not found at ${LUMEN_BIN}" >&2
        echo "Run without --no-build or build manually." >&2
        exit 1
    fi

    # ------------------------------------------------------------------
    # Step 2: Discover models
    # ------------------------------------------------------------------
    log_section "Discovering Models"

    local lbc_models_str
    lbc_models_str="$(discover_lbc_models)"
    if [[ -z "$lbc_models_str" ]]; then
        echo "ERROR: No .lbc models found in ${BENCH_DIR}" >&2
        if [[ -n "$MODEL_FILTER" ]]; then
            echo "  (filter: ${MODEL_FILTER})" >&2
        fi
        exit 1
    fi

    IFS=' ' read -ra LBC_MODELS <<< "$lbc_models_str"
    log "Found ${#LBC_MODELS[@]} LBC model(s):"
    local m
    for m in "${LBC_MODELS[@]}"; do
        local bn
        bn="$(basename "$m" .lbc)"
        local name_quant
        name_quant="$(parse_model_name "$bn")"
        local display_name display_quant
        display_name="$(printf '%s' "$name_quant" | cut -f1)"
        display_quant="$(printf '%s' "$name_quant" | cut -f2)"
        log "  - ${bn}  (${display_name} ${display_quant})"
    done

    if [[ "$LUMEN_ONLY" -eq 0 ]]; then
        discover_mlx_models
        log "Found ${#MLX_KEYS[@]} MLX model(s):"
        local i
        for ((i = 0; i < ${#MLX_KEYS[@]}; i++)); do
            log "  - ${MLX_KEYS[$i]} -> ${MLX_PATHS[$i]}"
        done

        if [[ "$SKIP_LLAMACPP" -eq 0 ]]; then
            discover_llamacpp_models
            if [[ "$SKIP_LLAMACPP" -eq 0 ]]; then
                log "Found ${#LLAMA_KEYS[@]} llama.cpp model(s):"
                for ((i = 0; i < ${#LLAMA_KEYS[@]}; i++)); do
                    log "  - ${LLAMA_KEYS[$i]} -> ${LLAMA_PATHS[$i]}"
                done
                if [[ ${#LLAMA_UNSUPPORTED[@]} -gt 0 ]]; then
                    log "Unsupported in llama.cpp: ${LLAMA_UNSUPPORTED[*]}"
                fi
            fi
        fi
    fi

    # ------------------------------------------------------------------
    # Step 3: Compute total benchmark count
    # ------------------------------------------------------------------
    local total_configs=$(( ${#LBC_MODELS[@]} * ${#PROMPT_LENGTHS[@]} * ${#GEN_LENGTHS[@]} ))
    local total_runs=$(( total_configs * (WARMUP_RUNS + RUNS) ))
    log ""
    log "Total Lumen configs: ${total_configs}"
    log "Total Lumen runs:    ${total_runs} (incl. warmup)"

    if [[ "$LUMEN_ONLY" -eq 0 ]]; then
        local mlx_configs=0
        for m in "${LBC_MODELS[@]}"; do
            local bn
            bn="$(basename "$m" .lbc)"
            local mlx_path
            mlx_path="$(mlx_lookup "$bn")" || true
            if [[ -n "$mlx_path" ]]; then
                mlx_configs=$(( mlx_configs + ${#PROMPT_LENGTHS[@]} * ${#GEN_LENGTHS[@]} ))
            fi
        done
        local mlx_total_runs=$(( mlx_configs * (WARMUP_RUNS + RUNS) ))
        log "Total MLX configs:   ${mlx_configs}"
        log "Total MLX runs:      ${mlx_total_runs} (x5 internal trials each)"

        if [[ "$SKIP_LLAMACPP" -eq 0 ]]; then
            local lc_configs=0
            for m in "${LBC_MODELS[@]}"; do
                local bn
                bn="$(basename "$m" .lbc)"
                local gguf_path
                gguf_path="$(llama_lookup "$bn")" || true
                if [[ -n "$gguf_path" ]]; then
                    lc_configs=$(( lc_configs + ${#PROMPT_LENGTHS[@]} * ${#GEN_LENGTHS[@]} ))
                fi
            done
            local lc_total_runs=$(( lc_configs * (WARMUP_RUNS + RUNS) ))
            log "Total llama.cpp configs: ${lc_configs}"
            log "Total llama.cpp runs:    ${lc_total_runs}"
        fi

        log "NOTE: MLX runs FIRST (cold GPU), then llama.cpp, then Lumen"
    fi

    # ------------------------------------------------------------------
    # Step 4: Run benchmarks
    # ------------------------------------------------------------------

    # Temporary file to collect all results in pipe-delimited format
    local results_tmp="${RESULTS_DIR}/results_raw.txt"
    : > "$results_tmp"

    # ------------------------------------------------------------------
    # Phase 1: Run MLX benchmarks FIRST (cold GPU, most accurate)
    # ------------------------------------------------------------------
    if [[ "$LUMEN_ONLY" -eq 0 ]]; then
        local mlx_model_idx=0
        for lbc_path in "${LBC_MODELS[@]}"; do
            local bn
            bn="$(basename "$lbc_path" .lbc)"

            local name_quant
            name_quant="$(parse_model_name "$bn")"
            local model_name quant
            model_name="$(printf '%s' "$name_quant" | cut -f1)"
            quant="$(printf '%s' "$name_quant" | cut -f2)"

            local mlx_path
            mlx_path="$(mlx_lookup "$bn")" || true
            if [[ -z "$mlx_path" ]]; then
                continue
            fi

            if [[ "$mlx_model_idx" -gt 0 && "$COOLDOWN_SECS" -gt 0 ]]; then
                log ""
                log "Cooling down for ${COOLDOWN_SECS}s between MLX models..."
                sleep "$COOLDOWN_SECS"
            fi

            log_section "MLX: ${model_name} ${quant} (cold GPU)"

            local prompt_len gen_len
            local mlx_config_idx=0
            for prompt_len in "${PROMPT_LENGTHS[@]}"; do
                for gen_len in "${GEN_LENGTHS[@]}"; do
                    # Inter-config cooldown (skip first config)
                    if [[ "$mlx_config_idx" -gt 0 && "$CONFIG_COOLDOWN_SECS" -gt 0 ]]; then
                        sleep "$CONFIG_COOLDOWN_SECS"
                    fi
                    mlx_config_idx=$((mlx_config_idx + 1))

                    log "  Config: pp${prompt_len}+gen${gen_len}"

                    # Thermal check before each benchmark
                    local raw_thermal="${RAW_DIR}/mlx_${bn}_pp${prompt_len}_gen${gen_len}_thermal.txt"
                    check_thermal_and_cool "$raw_thermal"

                    local stats
                    stats="$(run_mlx_bench "$mlx_path" "$prompt_len" "$gen_len" "$bn")" || true

                    local pp_med pp_sd dec_med dec_sd
                    pp_med="$(echo "$stats" | awk '{print $1}')"
                    pp_sd="$(echo "$stats" | awk '{print $2}')"
                    dec_med="$(echo "$stats" | awk '{print $3}')"
                    dec_sd="$(echo "$stats" | awk '{print $4}')"

                    log "    Result: prefill=${pp_med}+-${pp_sd} tok/s, decode=${dec_med}+-${dec_sd} tok/s"

                    echo "mlx|${bn}|${model_name}|${quant}|${prompt_len}|${gen_len}|${pp_med}|${pp_sd}|${dec_med}|${dec_sd}" >> "$results_tmp"
                done
            done

            mlx_model_idx=$((mlx_model_idx + 1))
        done

        if [[ "$mlx_model_idx" -gt 0 && "$COOLDOWN_SECS" -gt 0 ]]; then
            log ""
            log "Cooling down for ${COOLDOWN_SECS}s after MLX phase..."
            sleep "$COOLDOWN_SECS"
        fi
    fi

    # ------------------------------------------------------------------
    # Phase 1.5: Run llama.cpp benchmarks (after MLX, before Lumen)
    # ------------------------------------------------------------------
    if [[ "$LUMEN_ONLY" -eq 0 && "$SKIP_LLAMACPP" -eq 0 ]]; then
        local lc_model_idx=0
        for lbc_path in "${LBC_MODELS[@]}"; do
            local bn
            bn="$(basename "$lbc_path" .lbc)"
            local name_quant
            name_quant="$(parse_model_name "$bn")"
            local model_name quant
            model_name="$(printf '%s' "$name_quant" | cut -f1)"
            quant="$(printf '%s' "$name_quant" | cut -f2)"

            # Check if unsupported
            if llama_is_unsupported "$bn"; then
                log "  llama.cpp: ${model_name} ${quant} -- UNSUPPORTED (GDN on Metal)"
                for prompt_len in "${PROMPT_LENGTHS[@]}"; do
                    for gen_len in "${GEN_LENGTHS[@]}"; do
                        echo "llamacpp|${bn}|${model_name}|${quant}|${prompt_len}|${gen_len}|UNSUPPORTED|0|UNSUPPORTED|0" >> "$results_tmp"
                    done
                done
                continue
            fi

            # Lookup GGUF path
            local gguf_path
            gguf_path="$(llama_lookup "$bn")" || true
            if [[ -z "$gguf_path" ]]; then
                continue
            fi

            if [[ "$lc_model_idx" -gt 0 && "$COOLDOWN_SECS" -gt 0 ]]; then
                log ""
                log "Cooling down for ${COOLDOWN_SECS}s between llama.cpp models..."
                sleep "$COOLDOWN_SECS"
            fi

            log_section "llama.cpp: ${model_name} ${quant}"

            local prompt_len gen_len
            local lc_config_idx=0
            for prompt_len in "${PROMPT_LENGTHS[@]}"; do
                for gen_len in "${GEN_LENGTHS[@]}"; do
                    if [[ "$lc_config_idx" -gt 0 && "$CONFIG_COOLDOWN_SECS" -gt 0 ]]; then
                        sleep "$CONFIG_COOLDOWN_SECS"
                    fi
                    lc_config_idx=$((lc_config_idx + 1))

                    log "  Config: pp${prompt_len}+gen${gen_len}"

                    local raw_thermal="${RAW_DIR}/llamacpp_${bn}_pp${prompt_len}_gen${gen_len}_thermal.txt"
                    check_thermal_and_cool "$raw_thermal"

                    local stats
                    stats="$(run_llamacpp_bench "$gguf_path" "$prompt_len" "$gen_len" "$bn")" || true

                    local pp_med pp_sd dec_med dec_sd
                    pp_med="$(echo "$stats" | awk '{print $1}')"
                    pp_sd="$(echo "$stats" | awk '{print $2}')"
                    dec_med="$(echo "$stats" | awk '{print $3}')"
                    dec_sd="$(echo "$stats" | awk '{print $4}')"

                    log "    Result: prefill=${pp_med}+-${pp_sd} tok/s, decode=${dec_med}+-${dec_sd} tok/s"

                    echo "llamacpp|${bn}|${model_name}|${quant}|${prompt_len}|${gen_len}|${pp_med}|${pp_sd}|${dec_med}|${dec_sd}" >> "$results_tmp"
                done
            done

            lc_model_idx=$((lc_model_idx + 1))
        done

        if [[ "$lc_model_idx" -gt 0 && "$COOLDOWN_SECS" -gt 0 ]]; then
            log ""
            log "Cooling down for ${COOLDOWN_SECS}s after llama.cpp phase before Lumen phase..."
            sleep "$COOLDOWN_SECS"
        fi
    fi

    # ------------------------------------------------------------------
    # Phase 2: Run Lumen benchmarks
    # ------------------------------------------------------------------
    local config_idx=0
    local model_idx=0

    for lbc_path in "${LBC_MODELS[@]}"; do
        local bn
        bn="$(basename "$lbc_path" .lbc)"

        local name_quant
        name_quant="$(parse_model_name "$bn")"
        local model_name quant
        model_name="$(printf '%s' "$name_quant" | cut -f1)"
        quant="$(printf '%s' "$name_quant" | cut -f2)"

        # Cooldown between models (skip first)
        if [[ "$model_idx" -gt 0 && "$COOLDOWN_SECS" -gt 0 ]]; then
            log ""
            log "Cooling down for ${COOLDOWN_SECS}s between models..."
            sleep "$COOLDOWN_SECS"
        fi

        log_section "Lumen: ${model_name} ${quant} (${bn})"

        local prompt_len gen_len
        local lumen_config_idx=0
        for prompt_len in "${PROMPT_LENGTHS[@]}"; do
            for gen_len in "${GEN_LENGTHS[@]}"; do
                # Inter-config cooldown (skip first config)
                if [[ "$lumen_config_idx" -gt 0 && "$CONFIG_COOLDOWN_SECS" -gt 0 ]]; then
                    sleep "$CONFIG_COOLDOWN_SECS"
                fi
                lumen_config_idx=$((lumen_config_idx + 1))

                config_idx=$((config_idx + 1))
                log "  Config ${config_idx}/${total_configs}: pp${prompt_len}+gen${gen_len}"

                # Thermal check before each benchmark
                local raw_thermal="${RAW_DIR}/${bn}_pp${prompt_len}_gen${gen_len}_thermal.txt"
                check_thermal_and_cool "$raw_thermal"

                local stats
                stats="$(run_lumen_bench "$lbc_path" "$prompt_len" "$gen_len" "$bn")" || true

                local pp_med pp_sd dec_med dec_sd
                pp_med="$(echo "$stats" | awk '{print $1}')"
                pp_sd="$(echo "$stats" | awk '{print $2}')"
                dec_med="$(echo "$stats" | awk '{print $3}')"
                dec_sd="$(echo "$stats" | awk '{print $4}')"

                log "    Result: prefill=${pp_med}+-${pp_sd} tok/s, decode=${dec_med}+-${dec_sd} tok/s"

                echo "lumen|${bn}|${model_name}|${quant}|${prompt_len}|${gen_len}|${pp_med}|${pp_sd}|${dec_med}|${dec_sd}" >> "$results_tmp"
            done
        done

        model_idx=$((model_idx + 1))
    done

    # ------------------------------------------------------------------
    # Step 5: Generate output
    # ------------------------------------------------------------------
    log_section "Generating Reports"

    local md_file="${RESULTS_DIR}/results.md"
    local json_file="${RESULTS_DIR}/results.json"

    generate_markdown "$md_file" "$results_tmp"
    generate_json "$json_file" "$results_tmp"

    log ""
    log "Results saved to:"
    log "  Markdown: ${md_file}"
    log "  JSON:     ${json_file}"
    log "  Raw logs: ${RAW_DIR}/"

    # Also create a symlink to latest results
    local latest_link="${REPO_ROOT}/bench/results/latest"
    rm -f "$latest_link"
    ln -sf "${RESULTS_DIR}" "$latest_link"
    log "  Latest:   ${latest_link}"
}

# ============================================================================
# Trap: print partial results on interrupt
# ============================================================================

cleanup() {
    echo ""
    echo "INTERRUPTED -- generating partial results..."
    local results_tmp="${RESULTS_DIR}/results_raw.txt"
    if [[ -f "$results_tmp" && -s "$results_tmp" ]]; then
        local md_file="${RESULTS_DIR}/results.md"
        local json_file="${RESULTS_DIR}/results.json"
        generate_markdown "$md_file" "$results_tmp" 2>/dev/null || true
        generate_json "$json_file" "$results_tmp" 2>/dev/null || true
        echo "Partial results saved to: ${RESULTS_DIR}/"
    else
        echo "No results collected yet."
    fi
    exit 130
}

trap cleanup INT TERM

# ============================================================================
# Entry point
# ============================================================================

main "$@"
