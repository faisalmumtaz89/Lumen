#!/usr/bin/env bash
# Prompt-length bench B7 ONLY rerun.
#
# Recovers B7 stationarity test after the original prompt_length_bench.sh
# under-sized --context-len: 8192-token target paragraph repetition tokenized to
# 10201 actual tokens, hitting the 9216-token KV cap.
#
# Fix: use context-len 16384, target ~5500 paragraph-tokens (~6900 actual),
# leaving ~9000 tokens headroom for KV + decode.
#
# Single-purpose: ONLY B7. Appends to the existing summary.txt so the report
# carries B2+B3 from the original run plus B7 from this rerun.

set -uo pipefail

REPO=${REPO:-$HOME/lumen}
LBC_DIR=${LBC_DIR:-/mnt/nvme0/lumen-cache/lbc}
LOG_DIR=${LOG_DIR:-$HOME/lumen-cache/logs/prompt-length-bench}
LUMEN=$REPO/target/release/lumen
DENSE_LBC=$LBC_DIR/qwen_qwen3.5-9b-q8_0.lbc
SUMMARY=$LOG_DIR/summary.txt
RESULTS_JSON=$LOG_DIR/results.json

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export LUMEN_CUDA_BF16_GEMMEX=${LUMEN_CUDA_BF16_GEMMEX:-1}
export LUMEN_CUDA_DECODE_TILED_THRESHOLD=${LUMEN_CUDA_DECODE_TILED_THRESHOLD:-0}

mkdir -p "$LOG_DIR"

# Helpers
extract_tps() {
  grep -oE 'Decode:[[:space:]]+[0-9]+\.[0-9]+ tok/s' "$1" 2>/dev/null \
    | tail -n1 | grep -oE '[0-9]+\.[0-9]+'
}
extract_prefill_tps() {
  grep -oE 'Prefill:[[:space:]]+[0-9]+\.[0-9]+ tok/s' "$1" 2>/dev/null \
    | tail -n1 | grep -oE '[0-9]+\.[0-9]+'
}
extract_gen_tokens() {
  grep -aE '^Generated tokens:' "$1" 2>/dev/null | tail -n1 \
    | sed -E 's/^Generated tokens: \[//; s/\]$//'
}
extract_prompt_tokens_count() {
  grep -aE 'Prompt:[[:space:]]+[0-9]+ tok' "$1" 2>/dev/null | tail -n1 \
    | grep -oE 'Prompt:[[:space:]]+[0-9]+' | grep -oE '[0-9]+'
}
extract_gen_tokens_count() {
  grep -aE 'Generated:[[:space:]]+[0-9]+ tok' "$1" 2>/dev/null | tail -n1 \
    | grep -oE 'Generated:[[:space:]]+[0-9]+' | grep -oE '[0-9]+'
}

median() {
  python3 -c "
import sys, statistics
vals=[]
for l in sys.stdin:
    l=l.strip()
    if l:
        try: vals.append(float(l))
        except ValueError: pass
print(f'{statistics.median(vals):.4f}' if vals else 'n/a')
"
}

first16_tokens() {
  python3 -c "
line = '''$1'''
parts = [t.strip() for t in line.split(',') if t.strip()]
print(' '.join(parts[:16]))
"
}

log() { echo "$@" | tee -a "$SUMMARY"; }

run_lumen() {
  local label="$1" prompt="$2" maxtok="$3" ctxlen="$4" seed="$5"
  local logf="$LOG_DIR/$label.log"
  local args=( run --model "$DENSE_LBC" --prompt "$prompt" \
                  --max-tokens "$maxtok" --temperature 0 --cuda --verbose \
                  --context-len "$ctxlen" )
  if [[ -n "$seed" ]]; then args+=(--seed "$seed"); fi
  echo "==> [$label] $(date -Is)" > "$logf"
  echo "    cmd: $LUMEN ${args[*]:0:18} ..." >> "$logf"
  timeout 1800 "$LUMEN" "${args[@]}" >> "$logf" 2>&1
  return $?
}

# B7: M~8192 (actual), G up to 512. Build prompt targeting ~5500 paragraph-tokens
# which empirically tokenizes to ~7000-7500 actual. CTX=16384.
B7_CTX=16384

build_long_prompt() {
  # paragraph ~ 70 tokens; we want ~7000 actual so target=5500
  python3 -c "
target = 5500
para = ('The history of computing spans many decades, beginning with mechanical '
        'calculators in the 17th century. Early electronic computers in the 1940s, '
        'such as ENIAC and Colossus, paved the way for modern digital systems. '
        'The invention of the transistor in 1947 and the integrated circuit in '
        '1958 enabled Moores Law: the doubling of computational density roughly '
        'every two years. ')
n = max(1, target // 70)
import sys
sys.stdout.write((para * n).rstrip())
sys.stdout.write('\n\nQUESTION: Summarize in 80 words.\n')
"
}

B7_PROMPT="$(build_long_prompt)"

echo | tee -a "$SUMMARY"
log "=== BUCKET B7 (RERUN with ctx=16384): long prompt (M~7000) x long output (G up to 512) -- stationarity ==="

# 4 segment runs at G in {32, 128, 256, 512}; SAME seed-less greedy (temp=0).
B7_SEG_TPS=()
for G in 32 128 256 512; do
  label="b7r_g${G}"
  run_lumen "$label" "$B7_PROMPT" "$G" "$B7_CTX" ""
  rc=$?
  tps=$(extract_tps "$LOG_DIR/$label.log")
  pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
  M_actual=$(extract_prompt_tokens_count "$LOG_DIR/$label.log")
  G_actual=$(extract_gen_tokens_count "$LOG_DIR/$label.log")
  log "[B7r G=$G] rc=$rc M_actual=${M_actual:-?} G_actual=${G_actual:-?} prefill=${pf:-n/a} decode=${tps:-n/a}"
  B7_SEG_TPS+=("$G:${tps:-NA}")
done

# Stationarity computation: relative drift vs G=32
B7_STATIONARITY=$(python3 -c "
segs = '''${B7_SEG_TPS[@]}'''.split()
data = {}
for s in segs:
    if ':' not in s: continue
    g, t = s.split(':', 1)
    if t == 'NA': continue
    try:
        data[int(g)] = float(t)
    except ValueError:
        pass
if 32 in data:
    base = data[32]
    out = [f'BASELINE G=32: {base:.2f} tps']
    deltas = []
    for g in (128, 256, 512):
        if g in data:
            delta = (data[g] - base) / base * 100
            deltas.append(abs(delta))
            out.append(f'G={g}:{data[g]:.2f} tps (delta_vs_G32={delta:+.2f}%)')
    max_abs = max(deltas) if deltas else 0.0
    verdict = 'STATIONARY' if max_abs <= 10.0 else 'DRIFT_DETECTED'
    out.append(f'MAX_ABS_DRIFT_PCT={max_abs:.2f} VERDICT={verdict}')
    print('\n'.join(out))
else:
    print('VERDICT=NO_BASELINE')
")
log "[B7r stationarity]"
echo "$B7_STATIONARITY" | tee -a "$SUMMARY"

# 5-trial median at G=512
B7_TPS_VALS=""
B7_PF_VALS=""
for trial in 1 2 3 4 5; do
  label="b7r_med_t${trial}"
  run_lumen "$label" "$B7_PROMPT" 512 "$B7_CTX" ""
  tps=$(extract_tps "$LOG_DIR/$label.log")
  pf=$(extract_prefill_tps "$LOG_DIR/$label.log")
  M_actual=$(extract_prompt_tokens_count "$LOG_DIR/$label.log")
  log "[B7r t$trial] M=${M_actual:-?} decode=${tps:-n/a} prefill=${pf:-n/a}"
  if [[ -n "$tps" ]]; then B7_TPS_VALS="$B7_TPS_VALS$tps"$'\n'; fi
  if [[ -n "$pf" ]]; then B7_PF_VALS="$B7_PF_VALS$pf"$'\n'; fi
done

B7_TPS_MED=$(printf '%s' "$B7_TPS_VALS" | median)
B7_PF_MED=$(printf '%s' "$B7_PF_VALS" | median)
log "[B7r] decode tok/s median (5-trial @ G=512): $B7_TPS_MED tok/s"
log "[B7r] prefill tok/s median (5-trial): $B7_PF_MED tok/s"

# Sample first-16 from the G=32 run (smallest)
B7_TOKS=$(extract_gen_tokens "$LOG_DIR/b7r_g32.log")
B7_FIRST16=$(first16_tokens "$B7_TOKS")
log "[B7r] first16 (G=32): '$B7_FIRST16'"

# Determinism
run_lumen "b7r_det1" "$B7_PROMPT" 32 "$B7_CTX" 99
run_lumen "b7r_det2" "$B7_PROMPT" 32 "$B7_CTX" 99
T1=$(extract_gen_tokens "$LOG_DIR/b7r_det1.log")
T2=$(extract_gen_tokens "$LOG_DIR/b7r_det2.log")
if [[ "$T1" == "$T2" ]] && [[ -n "$T1" ]]; then B7_DET="MATCH"; else B7_DET="DIVERGE"; fi
log "[B7r] determinism: $B7_DET"

# Sample decoded text snippet (G=512 run)
B7_SAMPLE=$(grep -aE '^([A-Z]|Generated|>)' "$LOG_DIR/b7r_med_t1.log" 2>/dev/null \
  | head -c 240 | tr '\n' ' ' || echo "")
log "[B7r] decoded sample (first 240 chars of t1): $B7_SAMPLE"

# Final RERUN-only summary
log ""
log "=== B7-RERUN SUMMARY ==="
log "ctx-len:                $B7_CTX"
log "decode tok/s @G=512:    $B7_TPS_MED (5-trial median)"
log "prefill tok/s:          $B7_PF_MED (5-trial median)"
log "stationarity report:"
echo "$B7_STATIONARITY" | tee -a "$SUMMARY"
log "determinism:            $B7_DET"
log "first16:                $B7_FIRST16"
