#!/usr/bin/env bash
# Run the self-contained quality gates across all local Metal cells, accumulate a scorecard.
set +e
cd "$(dirname "$0")/.."
CACHE="$HOME/Library/Caches/lumen"
OUT=/tmp/qsuite
mkdir -p "$OUT"
SCORE="$OUT/metal_scorecard.txt"
: > "$SCORE"
GATES="${GATES:-GQ-001,GQ-002}"
PORT=8500

# cell-label  model-file
CELLS=(
  "9b-bf16-metal    qwen3-5-9b-BF16.lbc"
  "9b-q8-metal      qwen3-5-9b-Q8_0.lbc"
  "9b-q4-metal      qwen3-5-9b-Q4_0.lbc"
  "27b-bf16-metal   qwen36-BF16.lbc"
  "27b-q4-metal     qwen36-Q4_0.lbc"
  "moe-q8-metal     qwen3-5-moe-35b-a3b-Q8_0.lbc"
  "moe-q4-metal     qwen3-5-moe-35b-a3b-Q4_0.lbc"
)

for entry in "${CELLS[@]}"; do
  set -- $entry
  cell="$1"; mf="$2"
  PORT=$((PORT+1))
  if [ ! -f "$CACHE/$mf" ]; then
    echo "$cell  SKIP (model $mf not found)" | tee -a "$SCORE"
    continue
  fi
  pkill -9 -f "lumen-server.*$PORT" 2>/dev/null; sleep 1
  echo ">>> $cell ($mf) gates=$GATES  $(date +%H:%M:%S)" | tee -a "$SCORE"
  python3 quality/run_suite.py --model "$CACHE/$mf" --backend metal \
      --cell "$cell" --port "$PORT" --gates "$GATES" --out "$OUT" 2>&1 \
    | grep -E "^\| GQ-|CELL " | tee -a "$SCORE"
  echo "" | tee -a "$SCORE"
done
echo "=== METAL MATRIX DONE $(date +%H:%M:%S) ===" | tee -a "$SCORE"
echo "ALL_DONE" > "$OUT/metal_matrix.done"
