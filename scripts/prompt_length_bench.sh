#!/usr/bin/env bash
# Prompt-length x output-length bucket bench.
#
# Populates 3 measurement buckets:
#
#   B2: short prompt (M<=128) x long output (G>=256)  -- Q4 cumulative near-tie check
#   B3: medium prompt (M in [512, 2048]) x short output (G<=32)  -- pf scaling gap
#   B7: long prompt (M in [4K, 16K]) x long output (G>=512)  -- stationarity
#
# Per-bucket capture:
#   * decode tok/s (5-trial median)
#   * prefill tok/s (single shot)
#   * top-1 first 16 generated tokens (sensible-text gate)
#   * per-token decode tok/s timeline for B7 (via incremental --max-tokens runs)
#   * 2x same-seed determinism check
#
# Pinned to A100 PCIe Q8_0 LBC; runs on a specific GPU passed via env CUDA_VISIBLE_DEVICES.
#
# Invokes the locally built lumen binary at "$REPO/target/release/lumen".
# Override REPO to point at your local checkout (defaults to $HOME/lumen).

set -uo pipefail

REPO=${REPO:-$HOME/lumen}
LBC_DIR=${LBC_DIR:-/mnt/nvme0/lumen-cache/lbc}
LOG_DIR=${LOG_DIR:-$HOME/lumen-cache/logs/prompt-length-bench}
LUMEN=$REPO/target/release/lumen
DENSE_LBC=$LBC_DIR/qwen_qwen3.5-9b-q8_0.lbc

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p "$LOG_DIR"
RESULTS_JSON=$LOG_DIR/results.json
SUMMARY=$LOG_DIR/summary.txt
: > "$SUMMARY"
: > "$RESULTS_JSON"
echo "{" > "$RESULTS_JSON"

# Pin the CUDA opt-in stack so envelopes match the published baselines.
export LUMEN_CUDA_BF16_GEMMEX=${LUMEN_CUDA_BF16_GEMMEX:-1}
export LUMEN_CUDA_DECODE_TILED_THRESHOLD=${LUMEN_CUDA_DECODE_TILED_THRESHOLD:-0}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

extract_tps() {
  grep -oE 'Decode:[[:space:]]+[0-9]+\.[0-9]+ tok/s' "$1" 2>/dev/null \
    | tail -n1 | grep -oE '[0-9]+\.[0-9]+'
}
extract_prefill_tps() {
  grep -oE 'Prefill:[[:space:]]+[0-9]+\.[0-9]+ tok/s' "$1" 2>/dev/null \
    | tail -n1 | grep -oE '[0-9]+\.[0-9]+'
}
extract_prefill_ms() {
  grep -oE 'Prefill:[[:space:]]+[0-9]+\.[0-9]+ tok/s \([0-9]+\.[0-9]+ms\)' "$1" 2>/dev/null \
    | tail -n1 | grep -oE '\([0-9]+\.[0-9]+ms\)' | tr -d '()ms'
}
extract_gen_tokens() {
  grep -aE '^Generated tokens:' "$1" 2>/dev/null | tail -n1 \
    | sed -E 's/^Generated tokens: \[//; s/\]$//'
}
extract_prompt_tokens_count() {
  # "Prompt: 11 tok, Generated: 10 tok"
  grep -aE 'Prompt:[[:space:]]+[0-9]+ tok' "$1" 2>/dev/null | tail -n1 \
    | grep -oE 'Prompt:[[:space:]]+[0-9]+' | grep -oE '[0-9]+'
}
extract_gen_tokens_count() {
  grep -aE 'Generated:[[:space:]]+[0-9]+ tok' "$1" 2>/dev/null | tail -n1 \
    | grep -oE 'Generated:[[:space:]]+[0-9]+' | grep -oE '[0-9]+'
}

# median of stdin (one float per line) using python
median() {
  python3 -c "
import sys, statistics
vals = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: vals.append(float(line))
    except ValueError: pass
if vals: print(f'{statistics.median(vals):.4f}')
else: print('n/a')
"
}

stddev_pct() {
  # stdin: floats. prints stdev/mean * 100.
  python3 -c "
import sys, statistics
vals = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: vals.append(float(line))
    except ValueError: pass
if len(vals) >= 2:
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    print(f'{(sd/m)*100:.2f}' if m else '0.00')
else:
    print('n/a')
"
}

run_lumen() {
  # $1 = label, $2 = prompt-text, $3 = max-tokens, $4 = ctx-len (or 0 = auto), $5 = seed (or "")
  local label="$1" prompt="$2" maxtok="$3" ctxlen="$4" seed="$5"
  local logf="$LOG_DIR/$label.log"
  local args=( run --model "$DENSE_LBC" --prompt "$prompt" \
                  --max-tokens "$maxtok" --temperature 0 --cuda --verbose )
  if [[ "$ctxlen" -gt 0 ]]; then args+=(--context-len "$ctxlen"); fi
  if [[ -n "$seed" ]]; then args+=(--seed "$seed"); fi
  echo "==> [$label] $(date -Is)" > "$logf"
  echo "    cmd: $LUMEN ${args[*]:0:20} ... (prompt truncated)" >> "$logf"
  timeout 1800 "$LUMEN" "${args[@]}" >> "$logf" 2>&1
  return $?
}

# Build prompts of approx M tokens by repeating a small fixed-token chunk.
# We rely on the lumen "Prompt: N tok" output to record the real measured M.
build_short_prompt() {
  # ~14 tokens "Write one sentence explaining transformers."
  echo "Write one sentence explaining transformers."
}

build_essay_prompt() {
  # ~18 tokens.
  echo "Write a 256-word essay on the Industrial Revolution."
}

build_medium_prompt() {
  # Target M tokens for B3. We repeat a ~30-char/~6-7-token sentence ~N times.
  local target_tokens="$1"
  python3 -c "
target = $target_tokens
# 'The quick brown fox jumps over the lazy dog. ' ~= 9 tokens.
chunk = 'The quick brown fox jumps over the lazy dog. '
n = max(1, target // 9)
import sys
sys.stdout.write((chunk * n).rstrip())
sys.stdout.write('\nQUESTION: continue.\n')
"
}

build_long_prompt() {
  local target_tokens="$1"
  python3 -c "
target = $target_tokens
para = ('The history of computing spans many decades, beginning with mechanical '
        'calculators in the 17th century. Early electronic computers in the 1940s, '
        'such as ENIAC and Colossus, paved the way for modern digital systems. '
        'The invention of the transistor in 1947 and the integrated circuit in '
        '1958 enabled Moores Law: the doubling of computational density roughly '
        'every two years. ')
# ~70 tokens / paragraph.
n = max(1, target // 70)
import sys
sys.stdout.write((para * n).rstrip())
sys.stdout.write('\n\nQUESTION: Summarize in 80 words.\n')
"
}

# extract first-16 generated token-ids as space-joined CSV
first16_tokens() {
  local toks_line="$1"
  python3 -c "
import sys
line = '''$toks_line'''
parts = [t.strip() for t in line.split(',') if t.strip()]
print(' '.join(parts[:16]))
"
}

# --- record helpers ---
log() { echo "$@" | tee -a "$SUMMARY"; }

# -------------------------------------------------------------------
# B2: M<=128, G=512.  Q4-quality / Q8-quality / stationarity-lite.
# -------------------------------------------------------------------
echo | tee -a "$SUMMARY"
log "=== BUCKET B2: short prompt (M<=128) x long output (G=512) ==="

B2_PROMPT="$(build_essay_prompt)"
B2_MAX=512
# Auto-size context: prompt + max-gen + 256 headroom => well within 1024.
B2_CTX=1024

# 5 trials
B2_TPS_VALS=""
B2_PF_VALS=""
B2_LABELS=""

for trial in 1 2 3 4 5; do
  label="b2_t${trial}"
  run_lumen "$label" "$B2_PROMPT" "$B2_MAX" "$B2_CTX" ""
  rc=$?
  tps=$(extract_tps "$LOG_DIR/$label.log")
  pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
  gen=$(extract_gen_tokens_count "$LOG_DIR/$label.log")
  prompt_n=$(extract_prompt_tokens_count "$LOG_DIR/$label.log")
  log "[B2 t$trial] rc=$rc M=$prompt_n G=$gen prefill=${pf:-n/a} decode=${tps:-n/a}"
  if [[ -n "$tps" ]]; then
    B2_TPS_VALS="$B2_TPS_VALS$tps"$'\n'
    B2_PF_VALS="$B2_PF_VALS$pf"$'\n'
  fi
  B2_LABELS="$B2_LABELS $label"
done

B2_TPS_MED=$(printf '%s' "$B2_TPS_VALS" | median)
B2_TPS_VAR=$(printf '%s' "$B2_TPS_VALS" | stddev_pct)
B2_PF_MED=$(printf '%s' "$B2_PF_VALS" | median)
log "[B2] decode tok/s median=$B2_TPS_MED (cv%=$B2_TPS_VAR) | prefill tok/s median=$B2_PF_MED"

# --- B2: Q4-quality determinism check ---
# Determinism: run twice with same fixed seed, compare generated tokens.
run_lumen "b2_det1" "$B2_PROMPT" 64 "$B2_CTX" 42
run_lumen "b2_det2" "$B2_PROMPT" 64 "$B2_CTX" 42
B2_DET_T1=$(extract_gen_tokens "$LOG_DIR/b2_det1.log")
B2_DET_T2=$(extract_gen_tokens "$LOG_DIR/b2_det2.log")
if [[ "$B2_DET_T1" == "$B2_DET_T2" ]] && [[ -n "$B2_DET_T1" ]]; then
  B2_DET="MATCH"
else
  B2_DET="DIVERGE"
fi
log "[B2] determinism: $B2_DET"

# --- B2: per-segment decode rate (cumulative-exposure check) ---
# Run with G=32, G=128, G=256, G=512 and compare decode tok/s.
# This is the "Q4 near-tie cumulative" proxy: if drift accumulates, rate degrades.
B2_SEG_RATES=""
for G in 32 128 256 512; do
  label="b2_seg_g${G}"
  run_lumen "$label" "$B2_PROMPT" "$G" "$B2_CTX" 1
  tps=$(extract_tps "$LOG_DIR/$label.log")
  toks=$(extract_gen_tokens "$LOG_DIR/$label.log")
  first16=$(first16_tokens "$toks")
  log "[B2 seg G=$G] decode=${tps:-n/a} first16='$first16'"
  if [[ -n "$tps" ]]; then
    B2_SEG_RATES="$B2_SEG_RATES G=$G:$tps"
  fi
done

# Grab a decoded text sample for the 512-token run.
B2_TEXT_SAMPLE=$(grep -aE '^(Generated|Thinking|[A-Z])' "$LOG_DIR/b2_t1.log" 2>/dev/null \
  | head -c 240 | tr '\n' ' ' || echo "")

cat >> "$RESULTS_JSON" <<EOF
  "B2": {
    "M_target": 128,
    "M_measured": "$(extract_prompt_tokens_count "$LOG_DIR/b2_t1.log")",
    "G": 512,
    "decode_tps_median": "$B2_TPS_MED",
    "decode_tps_cv_pct": "$B2_TPS_VAR",
    "prefill_tps_median": "$B2_PF_MED",
    "determinism": "$B2_DET",
    "per_segment_tps": "$B2_SEG_RATES",
    "decoded_sample": "$(echo "$B2_TEXT_SAMPLE" | sed 's/"/\\"/g' | head -c 200)"
  },
EOF

# -------------------------------------------------------------------
# B3: M in [512, 2048], G<=32.  Prefill scaling gap.
# -------------------------------------------------------------------
echo | tee -a "$SUMMARY"
log "=== BUCKET B3: medium prompt (M in [512, 2048]) x short output (G=32) ==="

declare -A B3_DECODE_MED
declare -A B3_PREFILL_MED
declare -A B3_FIRST16
declare -A B3_DET

for TARGET_M in 512 2048; do
  log "--- B3 M=$TARGET_M ---"
  PROMPT="$(build_medium_prompt "$TARGET_M")"
  # ctx = M + 32 + 256 headroom -> round to next 1024 boundary
  CTX_LEN=$(python3 -c "print(((($TARGET_M + 32 + 256) // 1024) + 1) * 1024)")

  TPS_VALS=""
  PF_VALS=""
  for trial in 1 2 3 4 5; do
    label="b3_m${TARGET_M}_t${trial}"
    run_lumen "$label" "$PROMPT" 32 "$CTX_LEN" ""
    rc=$?
    tps=$(extract_tps "$LOG_DIR/$label.log")
    pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
    prompt_n=$(extract_prompt_tokens_count "$LOG_DIR/$label.log")
    log "[B3 M~$TARGET_M t$trial] rc=$rc M_actual=$prompt_n prefill=${pf:-n/a} decode=${tps:-n/a}"
    if [[ -n "$tps" ]]; then TPS_VALS="$TPS_VALS$tps"$'\n'; fi
    if [[ -n "$pf" ]]; then PF_VALS="$PF_VALS$pf"$'\n'; fi
  done
  B3_DECODE_MED[$TARGET_M]=$(printf '%s' "$TPS_VALS" | median)
  B3_PREFILL_MED[$TARGET_M]=$(printf '%s' "$PF_VALS" | median)
  log "[B3 M=$TARGET_M] decode median=${B3_DECODE_MED[$TARGET_M]} prefill median=${B3_PREFILL_MED[$TARGET_M]}"

  TOKS=$(extract_gen_tokens "$LOG_DIR/b3_m${TARGET_M}_t1.log")
  B3_FIRST16[$TARGET_M]=$(first16_tokens "$TOKS")
  log "[B3 M=$TARGET_M] first16='${B3_FIRST16[$TARGET_M]}'"

  # determinism
  run_lumen "b3_m${TARGET_M}_det1" "$PROMPT" 16 "$CTX_LEN" 7
  run_lumen "b3_m${TARGET_M}_det2" "$PROMPT" 16 "$CTX_LEN" 7
  T1=$(extract_gen_tokens "$LOG_DIR/b3_m${TARGET_M}_det1.log")
  T2=$(extract_gen_tokens "$LOG_DIR/b3_m${TARGET_M}_det2.log")
  if [[ "$T1" == "$T2" ]] && [[ -n "$T1" ]]; then B3_DET[$TARGET_M]="MATCH"; else B3_DET[$TARGET_M]="DIVERGE"; fi
done

# Scaling check: pf(2048) / pf(512) should be ~ <1 if M^2 prefill dominates; or ~1 if HBM-bound.
B3_SCALE=$(python3 -c "
p512 = ${B3_PREFILL_MED[512]:-0}
p2048 = ${B3_PREFILL_MED[2048]:-0}
if p512 > 0 and p2048 > 0:
    print(f'{p2048/p512:.3f}')
else:
    print('n/a')
")
log "[B3] prefill scaling ratio pf(M=2048)/pf(M=512) = $B3_SCALE"

cat >> "$RESULTS_JSON" <<EOF
  "B3": {
    "M512": {
      "decode_tps_median": "${B3_DECODE_MED[512]}",
      "prefill_tps_median": "${B3_PREFILL_MED[512]}",
      "first16": "${B3_FIRST16[512]}",
      "determinism": "${B3_DET[512]}"
    },
    "M2048": {
      "decode_tps_median": "${B3_DECODE_MED[2048]}",
      "prefill_tps_median": "${B3_PREFILL_MED[2048]}",
      "first16": "${B3_FIRST16[2048]}",
      "determinism": "${B3_DET[2048]}"
    },
    "scaling_ratio_pf2048_over_pf512": "$B3_SCALE"
  },
EOF

# -------------------------------------------------------------------
# B7: long prompt (M~8192) x long output (G=512). Stationarity test.
# -------------------------------------------------------------------
echo | tee -a "$SUMMARY"
log "=== BUCKET B7: long prompt (M~8192) x long output (G=512) -- stationarity ==="

B7_M=8192
B7_PROMPT="$(build_long_prompt "$B7_M")"
B7_CTX=$(python3 -c "print(((($B7_M + 512 + 256) // 1024) + 1) * 1024)")

# Trials: G=32, G=128, G=256, G=512 with same prompt + same seed for stationarity baseline.
# decode tok/s at G=32 is "early" decode rate; G=512 is "late" decode rate.
# Stationarity gate per A2: rate at G/2 within +/- 5% of rate at G=32.
B7_SEG_TPS=()
for G in 32 128 256 512; do
  label="b7_g${G}"
  run_lumen "$label" "$B7_PROMPT" "$G" "$B7_CTX" ""
  rc=$?
  tps=$(extract_tps "$LOG_DIR/$label.log")
  pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
  M_actual=$(extract_prompt_tokens_count "$LOG_DIR/$label.log")
  G_actual=$(extract_gen_tokens_count "$LOG_DIR/$label.log")
  log "[B7 G=$G] rc=$rc M_actual=$M_actual G_actual=$G_actual prefill=${pf:-n/a} decode=${tps:-n/a}"
  B7_SEG_TPS+=("$G:$tps")
done

# Stationarity computation:
#   for i in {1..4}: delta = (tps_i - tps_1) / tps_1 * 100
B7_STATIONARITY=$(python3 -c "
segs = '''${B7_SEG_TPS[@]}'''.split()
data = {}
for s in segs:
    if ':' not in s: continue
    g, t = s.split(':', 1)
    try:
        data[int(g)] = float(t)
    except ValueError:
        pass
if 32 in data:
    base = data[32]
    out = []
    for g in (128, 256, 512):
        if g in data:
            delta = (data[g] - base) / base * 100
            out.append(f'G={g}:{data[g]:.2f} tps (delta_vs_G32={delta:+.2f}%)')
    max_abs = max([abs((data[g]-base)/base*100) for g in (128,256,512) if g in data] + [0.0])
    print(f'BASELINE G=32: {base:.2f} tps')
    for line in out: print(line)
    verdict = 'STATIONARY' if max_abs <= 10.0 else 'DRIFT_DETECTED'
    print(f'MAX_ABS_DRIFT_PCT={max_abs:.2f} VERDICT={verdict}')
else:
    print('VERDICT=NO_BASELINE')
")
log "[B7 stationarity]"
echo "$B7_STATIONARITY" | tee -a "$SUMMARY"

# 5-trial median at G=512 for the headline cell.
B7_TPS_VALS=""
B7_PF_VALS=""
for trial in 1 2 3 4 5; do
  label="b7_med_t${trial}"
  run_lumen "$label" "$B7_PROMPT" 512 "$B7_CTX" ""
  tps=$(extract_tps "$LOG_DIR/$label.log")
  pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
  log "[B7 t$trial] decode=${tps:-n/a} prefill=${pf:-n/a}"
  if [[ -n "$tps" ]]; then B7_TPS_VALS="$B7_TPS_VALS$tps"$'\n'; fi
  if [[ -n "$pf" ]]; then B7_PF_VALS="$B7_PF_VALS$pf"$'\n'; fi
done
B7_TPS_MED=$(printf '%s' "$B7_TPS_VALS" | median)
B7_PF_MED=$(printf '%s' "$B7_PF_VALS" | median)
log "[B7] decode tok/s median=$B7_TPS_MED  | prefill tok/s median=$B7_PF_MED"

B7_TOKS=$(extract_gen_tokens "$LOG_DIR/b7_g32.log")
B7_FIRST16=$(first16_tokens "$B7_TOKS")
log "[B7] first16='$B7_FIRST16'"

# Determinism
run_lumen "b7_det1" "$B7_PROMPT" 32 "$B7_CTX" 99
run_lumen "b7_det2" "$B7_PROMPT" 32 "$B7_CTX" 99
T1=$(extract_gen_tokens "$LOG_DIR/b7_det1.log")
T2=$(extract_gen_tokens "$LOG_DIR/b7_det2.log")
if [[ "$T1" == "$T2" ]] && [[ -n "$T1" ]]; then B7_DET="MATCH"; else B7_DET="DIVERGE"; fi
log "[B7] determinism: $B7_DET"

# Append B7 JSON.  Use python to safely escape the multi-line stationarity report.
python3 - <<PYEOF >> "$RESULTS_JSON"
import json
report = """${B7_STATIONARITY}"""
first16 = """${B7_FIRST16}"""
out = {
    "M_target": ${B7_M},
    "G": 512,
    "decode_tps_median_G512": "${B7_TPS_MED}",
    "prefill_tps_median": "${B7_PF_MED}",
    "stationarity_report": report,
    "first16": first16,
    "determinism": "${B7_DET}",
    "segment_tps_raw": " ".join("""${B7_SEG_TPS[@]}""".split())
}
print('  "B7":', json.dumps(out) + ',')
PYEOF

# Close JSON.
echo '  "_meta": { "gpu": "'"$CUDA_VISIBLE_DEVICES"'", "lbc": "'"$DENSE_LBC"'", "ts": "'"$(date -Is)"'" }' >> "$RESULTS_JSON"
echo "}" >> "$RESULTS_JSON"

echo | tee -a "$SUMMARY"
log "=== SUMMARY (prompt-length bench) ==="
log "B2 decode median (M<=128, G=512): $B2_TPS_MED tok/s, cv=$B2_TPS_VAR%; determinism=$B2_DET"
log "B2 segment-tps timeline: $B2_SEG_RATES"
log "B3 M=512  decode=${B3_DECODE_MED[512]} prefill=${B3_PREFILL_MED[512]} det=${B3_DET[512]}"
log "B3 M=2048 decode=${B3_DECODE_MED[2048]} prefill=${B3_PREFILL_MED[2048]} det=${B3_DET[2048]}"
log "B3 pf scaling ratio (M=2048 / M=512): $B3_SCALE"
log "B7 decode median (M~8K, G=512): $B7_TPS_MED tok/s; prefill: $B7_PF_MED"
log "B7 stationarity report:"
echo "$B7_STATIONARITY" | tee -a "$SUMMARY"
log "B7 determinism: $B7_DET"

log "Results JSON: $RESULTS_JSON"
log "Summary:      $SUMMARY"
