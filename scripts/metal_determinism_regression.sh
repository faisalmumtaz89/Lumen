#!/usr/bin/env bash
# Metal greedy-decode DETERMINISM regression gate (DET-001).
#
# WHY THIS EXISTS
# ---------------
# DET-001 was a set of three intra-kernel cross-threadgroup races in the Metal
# Qwen3.5 path that made greedy decode (temperature=0) non-deterministic at a low,
# scheduler-timing-dependent rate (~5-10% of runs diverged). All three are fixed
# bit-for-bit (no env toggle, default behavior):
#   1. ssm_conv1d_state_update split  (gdn_advanced.msl / gdn.rs)  -- prefill conv_state
#   2. fused_rope_kv_mha: no in-place k_vec write-back (attention.msl) -- decode GQA K
#   3. deinterleave_norm_assemble de-aliased (gdn_core.msl) -- decode Q/gate vs K/V
#
# These races only surface with the REAL model's GQA ratio + GDN recurrence + batched
# prefill at scale, so a synthetic-tiny-model unit test cannot exercise them. This
# script is therefore the AUTHORITATIVE determinism regression: it loads the real model
# in lumen-server ONCE and fires N identical greedy completions, asserting that every
# decoded byte-stream is identical (1 distinct md5 / N). It is the permanent replacement
# for the ~13 LUMEN_METAL_DET_* investigation toggles that have since been removed.
#
# A divergence (distinct md5 > 1) is a DET-001 REGRESSION and must fail the gate.
#
# USAGE
#   scripts/metal_determinism_regression.sh [MODEL_LBC] [N_TRIALS] [PROMPT]
# Defaults:
#   MODEL_LBC = $LUMEN_DET_MODEL or ~/Library/Caches/lumen/qwen3-5-9b-Q8_0.lbc
#   N_TRIALS  = 50
#   PROMPT    = "Explain in detail how photosynthesis works in green plants, step by step."
#
# Exit 0 => deterministic (1 distinct md5 / N). Exit 1 => REGRESSION or infra failure.
#
# RESOURCE DISCIPLINE: exactly one lumen process at a time; the script kills stale
# lumen, launches its own server, and tears it down on exit. Runs at decode-delay=0
# (the production default, bit-exact). It does NOT set any LUMEN_METAL_DET_* var --
# none exist anymore; determinism must hold from the kernels alone.
set -u
LANG=C; LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/target/release/lumen-server"
PORT="${LUMEN_DET_PORT:-8166}"

MODEL="${1:-${LUMEN_DET_MODEL:-$HOME/Library/Caches/lumen/qwen3-5-9b-Q8_0.lbc}}"
N="${2:-50}"
PROMPT="${3:-Explain in detail how photosynthesis works in green plants, step by step.}"
MAXTOK="${LUMEN_DET_MAXTOK:-48}"

if [ ! -x "$SERVER" ]; then
  echo "[FAIL] lumen-server not built. Run: cargo build --release -p lumen-server --features bin"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "[SKIP] model not found: $MODEL (set LUMEN_DET_MODEL or pass as arg 1)"
  exit 0
fi

# Production default: decode-delay 0 (bit-exact). Do NOT set any DET toggle.
unset LUMEN_METAL_DECODE_DELAY_US 2>/dev/null || true

# One-process discipline.
pkill -f "target/release/lumen" 2>/dev/null || true
for _ in $(seq 1 10); do pgrep -f "target/release/lumen" >/dev/null || break; sleep 1; done
if pgrep -f "target/release/lumen" >/dev/null; then echo "[FAIL] a lumen process is still alive"; exit 1; fi

LOG="$(mktemp -t lumen_det_server.XXXXXX.log)"
"$SERVER" --model "$MODEL" --backend metal --port "$PORT" --log-level warn > "$LOG" 2>&1 &
SRV_PID=$!
cleanup() {
  kill -TERM "$SRV_PID" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SRV_PID" 2>/dev/null || true
  rm -f "$LOG"
}
trap cleanup EXIT

# Wait for readiness (no /health route; GET /v1/models). Up to 240s for big loads.
ready=0
for i in $(seq 1 240); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[FAIL] server died during load; tail:"; tail -20 "$LOG"; exit 1; fi
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { ready=1; break; }
  sleep 1
done
[ "$ready" = "1" ] || { echo "[FAIL] server not ready in 240s; tail:"; tail -20 "$LOG"; exit 1; }

echo "[run] model=$(basename "$MODEL") N=$N max_tokens=$MAXTOK temp=0 seed=42 delay=0(default)"
PROMPT_JSON="$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')"
RESULTS="$(mktemp -t lumen_det_md5s.XXXXXX)"
: > "$RESULTS"
ok=0
for t in $(seq 1 "$N"); do
  body=$(printf '{"model":"m","prompt":%s,"max_tokens":%s,"temperature":0,"seed":42,"stream":false}' "$PROMPT_JSON" "$MAXTOK")
  resp=$(curl -s --max-time 40 -H 'Content-Type: application/json' -H 'Connection: close' -d "$body" "http://127.0.0.1:$PORT/v1/completions" 2>/dev/null)
  [ -z "$resp" ] && { echo "trial $t: EMPTY/timeout"; echo "TIMEOUT" >> "$RESULTS"; continue; }
  text=$(printf '%s' "$resp" | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["text"], end="")
except Exception as e: sys.stderr.write("parse-err:%s\n"%e)')
  m=$(printf '%s' "$text" | md5 -q 2>/dev/null || printf '%s' "$text" | md5sum | awk '{print $1}')
  echo "$m" >> "$RESULTS"
  ok=$((ok+1))
done

distinct=$(sort -u "$RESULTS" | grep -vc TIMEOUT)
timeouts=$(grep -c TIMEOUT "$RESULTS" || true)
echo "=== completed=$ok/$N distinct_md5=$distinct timeouts=$timeouts ==="
sort "$RESULTS" | uniq -c | sort -rn
rm -f "$RESULTS"

if [ "$ok" -lt "$N" ]; then echo "[FAIL] only $ok/$N requests completed"; exit 1; fi
if [ "$distinct" -ne 1 ]; then echo "[FAIL] DET-001 REGRESSION: $distinct distinct outputs across $N identical greedy requests"; exit 1; fi
echo "[PASS] greedy decode is byte-deterministic ($N/$N identical)"
exit 0
