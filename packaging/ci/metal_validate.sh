#!/usr/bin/env bash
# Metal functional validation for the self-hosted Apple-Silicon CI runner.
# Builds nothing (the workflow already built); starts the release lumen-server on a
# small model and checks: server comes up, DET-001 byte-determinism (N=20 greedy ->
# 1 distinct), and a coherence smoke. Writes results under $RUNNER_TEMP/metal-val/.
#
# Model: set LUMEN_TEST_MODEL to a cached .lbc path on the runner to avoid a 9 GB
# pull every run; otherwise this falls back to `lumen pull qwen3.5-9b:q8_0`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Default to the freshly-built binary, but allow an override so release.yml can
# validate the EXACT binary extracted from the shipping tarball (artifact promotion).
SERVER="${LUMEN_SERVER_BIN:-$ROOT/target/release/lumen-server}"
LUMEN="${LUMEN_BIN:-$ROOT/target/release/lumen}"
OUT="${RUNNER_TEMP:-/tmp}/metal-val"; mkdir -p "$OUT"
PORT=8533
PROMPT='What is the capital of France?'

[ -x "$SERVER" ] || { echo "::error::$SERVER not built"; exit 1; }

# Resolve the model.
MODEL="${LUMEN_TEST_MODEL:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "LUMEN_TEST_MODEL unset/missing -> pulling qwen3.5-9b:q8_0"
  "$LUMEN" pull qwen3.5-9b:q8_0 --yes > "$OUT/pull.log" 2>&1 || true
  MODEL="$(grep -oE '/[^ ]+\.lbc' "$OUT/pull.log" | tail -1)"
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "::error::no model LBC available"; exit 1; }
echo "model: $MODEL"

# One server, torn down on exit.
"$SERVER" --model "$MODEL" --port "$PORT" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "::error::server died on startup"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done

gen() {  # deterministic greedy generation -> stdout
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":48,\"temperature\":0}" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

# DET-001: N=20 greedy must collapse to one distinct hash.
echo "== DET-001 (N=20) =="
: > "$OUT/det.hashes"
for _ in $(seq 1 20); do gen "$PROMPT" | shasum -a 256 | awk '{print $1}' >> "$OUT/det.hashes"; done
DISTINCT=$(sort -u "$OUT/det.hashes" | wc -l | tr -d ' ')
echo "distinct=$DISTINCT (want 1)" | tee "$OUT/det.result"
[ "$DISTINCT" = "1" ] || { echo "::error::Metal DET-001 non-deterministic ($DISTINCT distinct)"; exit 1; }

# Coherence smoke.
echo "== coherence =="
FR="$(gen "$PROMPT")"; echo "france -> $FR" | tee -a "$OUT/coherence.txt"
echo "$FR" | grep -qi paris || { echo "::error::coherence: 'Paris' not in answer"; exit 1; }
MATH="$(gen 'What is 17 times 23? Reply with only the number.')"; echo "17x23 -> $MATH" | tee -a "$OUT/coherence.txt"

echo "METAL VALIDATION PASS (DET-001 1-distinct, coherent)"
