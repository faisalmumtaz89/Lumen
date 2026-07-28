#!/usr/bin/env python3
"""run_suite.py — drive the Model Quality Acceptance Suite against one cell.

Spins (or connects to) a lumen server, runs the self-contained GQ-* gates
(GQ-001 short / GQ-002 medium / GQ-003 long / GQ-004 very-long), applies the
DD-* detectors + answer keys, and emits the per-cell Required-Output-Format
results table + a cell-<id>.json manifest.

Fidelity gates (GQ-005/006 vs llama) and the coherence
judge (GQ-013) are run by companion drivers; this driver marks them DEFERRED so
the scorecard is explicit (AH-7: never a silent omission).

Usage:
  run_suite.py --model PATH --backend metal --cell 9b-q8-metal --port 8401
  run_suite.py --server http://127.0.0.1:8401 --cell 9b-q8-metal
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as dd  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

# gate -> (corpus file, pass fraction, kind)
GATES = {
    "GQ-001": ("short.jsonl", 14 / 15, "short"),
    "GQ-002": ("medium.jsonl", 9 / 10, "generation"),
    "GQ-004": ("verylong.jsonl", 5 / 5, "verylong"),
    # GQ-014 (multi-turn conversation fidelity): stateless chat protocol — each
    # turn re-POSTs the FULL alternating history. A conversation passes only if
    # EVERY turn passes (DD-* + answer/anchor + the per-turn cross-turn check);
    # the gate passes only if ALL conversations pass (threshold 1.0, like GQ-004).
    # It exercises chat-history templating, growing-prefix KV reuse, and
    # cross-turn context (anaphora, state tracking, recall, running constraints,
    # big-prefix recall) that the single-turn gates never touch.
    # NOTE: numbered GQ-014, NOT GQ-005 — GQ-005 is the pre-existing FOUNDATIONAL
    # reference-greedy-fidelity-vs-llama gate (Section 10), referenced throughout
    # the suite docs and the MoE near-tie analysis; reusing 005 would collide.
    "GQ-014": ("multiturn.jsonl", 1.0, "multiturn"),
    # GQ-008 structured-output adherence and GQ-007 tool calling: both are
    # PARSE-OR-FAIL gates, so the threshold is 1.0. A JSON answer that is
    # "nearly" valid is not valid, and a tool call with a mangled argument is a
    # failure at the call site, not a near miss.
    "GQ-008": ("structured.jsonl", 1.0, "structured"),
    "GQ-007": ("structured.jsonl", 1.0, "structured"),
    # GQ-009/010/012 share one corpus: long-context retrieval, multilingual
    # instruction following, near-tie sampling and degeneration traps. These
    # exist to stress ACTIVATION PRECISION specifically — near-tie prompts sit
    # where argmax margins are smallest, which is where a quantised activation
    # changes a token first.
    "GQ-010": ("stress.jsonl", 1.0, "stress"),
}


def load_corpus(fname: str) -> list[dict]:
    path = CORPUS / fname
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def wait_ready(base: str, timeout_s: int | None = None) -> bool:
    # Large models (MoE q8 ~35 GB, bf16 ~70 GB) can take >>300 s to load from
    # disk; default generous, override via QSUITE_READY_TIMEOUT.
    if timeout_s is None:
        timeout_s = int(os.environ.get("QSUITE_READY_TIMEOUT", "3600"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=4) as r:
                if b"list" in r.read():
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def spin_server(model: str, backend: str, port: int, ctx: int = 4096, server_bin: str | None = None):
    srv = Path(server_bin) if server_bin else (HERE.parent / "target" / "release" / "lumen-server")
    if not srv.exists():
        sys.exit(f"lumen-server not found at {srv}")
    env = dict(os.environ)
    if backend == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log = open(f"/tmp/qsuite-server-{port}.log", "wb")
    proc = subprocess.Popen(
        [str(srv), "--model", model, "--backend", backend, "--port", str(port),
         "--context-len", str(ctx)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    return proc


def query(base: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> tuple[str, str]:
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason", "")


def query_tools(base: str, prompt: str, tools: list[dict], max_tokens: int,
                temperature: float = 0.0) -> tuple[str, str, list[dict]]:
    """GQ-007 client: POST with an OpenAI-shaped `tools` array.

    Returns (content, finish_reason, tool_calls). `tool_calls` is normalised to
    a list of {"name", "arguments"} so the scorer does not have to care whether
    the server nests them under `function` (OpenAI shape) or emits them flat.
    An empty list means the model answered in prose, which is the correct
    outcome for a negative item and a failure for a positive one.
    """
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": t} for t in tools],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    calls = []
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function", c)
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return msg.get("content") or "", choice.get("finish_reason", ""), calls


def query_messages(base: str, messages: list[dict], max_tokens: int,
                   temperature: float = 0.0) -> tuple[str, str]:
    """Multi-turn variant: POST the FULL alternating user/assistant history.

    The protocol is intentionally STATELESS at the HTTP layer — the client owns
    the transcript and re-sends it every turn (no server-side session id). This
    is the standard OpenAI chat-completions shape and is exactly what stresses
    chat-history templating + growing-prefix KV-cache reuse: turn N re-prefills
    turns 1..N-1 as the prefix and decodes only the new assistant reply.
    """
    body = json.dumps({
        "model": "x",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason", "")


# ---------------------------------------------------------------------------
# answer-key + anchor checks
# ---------------------------------------------------------------------------

def answer_ok(text: str, item: dict) -> bool:
    answers = item.get("answers")
    if not answers:
        return True
    low = text.lower()
    if item.get("answers_all"):
        return all(a.lower() in low for a in answers)
    # default: any acceptable alternative present (word-boundary for short numerics)
    for a in answers:
        al = a.lower()
        if re.search(rf"(?<!\d){re.escape(al)}(?!\d)", low) if al.isdigit() else (al in low):
            return True
    return False


def anchors_ok(text: str, item: dict) -> bool:
    anchors = item.get("anchors")
    if not anchors:
        return True
    low = text.lower()
    return any(a.lower() in low for a in anchors)  # >=1 anchor = on-topic


def window_detectors(text: str, win_words: int = 256) -> dd.QualityVerdict:
    """Sliding-window DD over a long output to catch late-onset degeneration."""
    words = text.split()
    if len(words) <= win_words:
        return dd.evaluate(text)
    worst = None
    for i in range(0, len(words), win_words // 2):
        chunk = " ".join(words[i : i + win_words])
        v = dd.evaluate(chunk)
        if not v.passed:
            return v
        worst = v
    return worst or dd.evaluate(text)


# ---------------------------------------------------------------------------
# gate runners
# ---------------------------------------------------------------------------

def run_prompt(base: str, item: dict, kind: str) -> dict:
    text, finish = query(base, item["prompt"], item.get("max_tokens", 256))
    # DD-TERM is NOT applied to open-ended generation: hitting max_tokens on an
    # explanation prompt is expected (truncation != degeneration). It is only
    # meaningful as an opt-in per-prompt flag ("require_eos": true) for prompts
    # that genuinely must self-terminate.
    v = dd.evaluate(
        text,
        is_math=item.get("is_math", False),
        finish_reason=finish,
        expect_eos=True,
        check_term=item.get("require_eos", False),
    )
    if kind == "verylong" or (kind == "stress" and item.get("kind") == "degeneration"):
        wv = window_detectors(text)
        if not wv.passed:
            v = wv
    correct = answer_ok(text, item)
    ontopic = anchors_ok(text, item)
    ok = v.passed and correct and ontopic
    return {
        "id": item["id"],
        "passed": ok,
        "dd_passed": v.passed,
        "dd_fired": v.failed_names(),
        "answer_correct": correct,
        "on_topic": ontopic,
        "finish_reason": finish,
        "snippet": text[:240].replace("\n", " "),
    }


def _extract_json(text: str):
    """Pull a JSON value out of a reply, tolerating code fences but NOT prose.

    Returns (value, error). A reply that needs prose stripped has already
    disobeyed "return ONLY JSON", so the caller records it as a failure even
    when the embedded JSON parses — adherence is the gate, not parseability.
    """
    t = text.strip()
    fenced = False
    if t.startswith("```"):
        fenced = True
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"```\s*$", "", t).strip()
    try:
        return json.loads(t), ("code-fenced" if fenced else None)
    except Exception:
        pass
    m = re.search(r"[\[{].*[\]}]", t, re.S)
    if not m:
        return None, "no JSON found"
    try:
        return json.loads(m.group(0)), "JSON embedded in prose"
    except Exception as e:
        return None, f"unparseable: {type(e).__name__}"


def run_structured(base: str, item: dict) -> dict:
    """GQ-007 / GQ-008. Tool items go through the chat endpoint with `tools`."""
    kind = item.get("kind")
    notes = []
    if kind in ("tool", "tool_negative"):
        text, finish, calls = query_tools(
            base, item["prompt"], item.get("tools", []), item.get("max_tokens", 256))
        want = item.get("expect_tool")
        if want is None:
            # A tool was offered that the request does not need. Calling it is
            # the failure this item exists to catch.
            ok = not calls
            if calls:
                notes.append(f"called {calls[0].get('name')} when no call was needed")
        elif not calls:
            ok, _ = False, notes.append("no tool call emitted")
        else:
            got = calls[0]
            ok = got.get("name") == want
            if not ok:
                notes.append(f"called {got.get('name')!r}, wanted {want!r}")
            args = got.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args, _ = {}, notes.append("arguments not valid JSON")
            for k, v in (item.get("expect_args") or {}).items():
                got_v = (args or {}).get(k)
                same = (str(got_v).strip().lower() == str(v).strip().lower()
                        if not isinstance(v, (int, float)) else got_v == v)
                if not same:
                    ok = False
                    notes.append(f"arg {k}={got_v!r} wanted {v!r}")
        return {"id": item["id"], "passed": ok, "dd_passed": True, "dd_fired": [],
                "answer_correct": ok, "on_topic": True, "finish_reason": finish,
                "notes": notes, "snippet": text[:240].replace("\n", " ")}

    text, finish = query(base, item["prompt"], item.get("max_tokens", 256))
    if kind == "json":
        val, err = _extract_json(text)
        ok = val is not None and err is None
        if err:
            notes.append(err)
        if val is not None:
            need = item.get("require_keys") or []
            n = item.get("require_array_len")
            if n is not None:
                if not isinstance(val, list) or len(val) != n:
                    ok = False
                    notes.append(f"wanted array of {n}, got {type(val).__name__}"
                                 f"{'' if not isinstance(val, list) else f' len {len(val)}'}")
                objs = val if isinstance(val, list) else []
            else:
                objs = [val]
            for o in objs:
                if not isinstance(o, dict):
                    ok = False
                    notes.append(f"element is {type(o).__name__}, not object")
                    continue
                missing = [k for k in need if k not in o]
                if missing:
                    ok = False
                    notes.append(f"missing keys {missing}")
    elif kind == "markdown_table":
        rows = [l for l in text.splitlines()
                if l.strip().startswith("|") and set(l.strip()) - set("|-: ")]
        body = [r for r in rows[1:]] if rows else []
        ok = len(body) == item.get("require_rows", 0)
        if not ok:
            notes.append(f"{len(body)} data rows, wanted {item.get('require_rows')}")
    else:
        ok, _ = False, notes.append(f"unknown structured kind {kind!r}")
    return {"id": item["id"], "passed": ok, "dd_passed": True, "dd_fired": [],
            "answer_correct": ok, "on_topic": True, "finish_reason": finish,
            "notes": notes, "snippet": text[:240].replace("\n", " ")}


def run_gate(base: str, gate: str) -> dict:
    fname, pass_frac, kind = GATES[gate]
    items = load_corpus(fname)
    if not items:
        return {"gate": gate, "status": "DEFERRED", "evidence": f"no corpus {fname}"}
    if gate in ("GQ-007", "GQ-008"):
        # One corpus, two gates: tool items belong to GQ-007, the rest to
        # GQ-008. Scoring both over the whole file would double-count.
        tool_kinds = ("tool", "tool_negative")
        items = [it for it in items
                 if (it.get("kind") in tool_kinds) == (gate == "GQ-007")]
        if not items:
            return {"gate": gate, "status": "DEFERRED",
                    "evidence": f"no {gate} items in {fname}"}
        results = [run_structured(base, it) for it in items]
    else:
        results = [run_prompt(base, it, kind) for it in items]
    npass = sum(1 for r in results if r["passed"])
    n = len(results)
    passed = npass >= max(1, round(pass_frac * n)) and npass == n if pass_frac >= 1.0 else npass >= round(pass_frac * n)
    fails = [r for r in results if not r["passed"]]
    status = "PASS" if passed else "FAIL"
    ev = f"{npass}/{n} prompts pass (threshold {pass_frac:.2f}). "
    if fails:
        ev += "Fails: " + "; ".join(
            f"{r['id']}[dd:{','.join(r['dd_fired']) or '-'}"
            f"{',ans✗' if not r['answer_correct'] else ''}"
            f"{',topic✗' if not r['on_topic'] else ''}]" for r in fails[:6]
        )
    return {"gate": gate, "status": status, "evidence": ev, "results": results}


# ---------------------------------------------------------------------------
# GQ-014 — multi-turn conversation fidelity
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Constraint-checking word count: strip markdown emphasis/punctuation so
    "**Blue**." and "Blue" both count as 1, matching how a human reads a
    'three words' answer. Used only by the expect_words cross-turn check."""
    cleaned = re.sub(r"[*_`#>~]", " ", text)
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned)  # drop standalone punctuation
    return len([w for w in cleaned.split() if any(c.isalnum() for c in w)])


def score_turn(text: str, finish: str, turn: dict) -> dict:
    """Score one assistant reply: DD-* (windowed) + answer-key + anchors + the
    optional per-turn cross-turn check (expect_words). Mirrors run_prompt's
    semantics so the multi-turn gate is as strict as the single-turn gates —
    a degenerate loop or a wrong cross-turn answer FAILs the turn (and thus the
    whole conversation). DD-TERM is opt-in (require_eos) since short
    constrained answers legitimately may not emit EOS within the cap."""
    v = window_detectors(text)
    if v.passed and turn.get("require_eos", False):
        tv = dd.dd_term(finish, expect_eos=True)
        if not tv.passed:
            v = dd.QualityVerdict(passed=False, results=v.results + [tv])
    correct = answer_ok(text, turn)
    ontopic = anchors_ok(text, turn)
    # cross-turn brevity constraint: a running "answer tersely" instruction set in
    # an EARLIER turn must still be honored on THIS turn. We gate on a CEILING
    # (max_words), not exact equality, because exact word-count adherence is a
    # model-quality property that even the llama.cpp reference does not hit ("exactly
    # three words" -> both Lumen and LC produced 1-6 words on the SAME conversation,
    # 9b-q8, GQ-014 control 2026-06-12). The DEFECT the gate must catch is the model
    # FORGETTING the brevity constraint and rambling (constraint dropped across the
    # growing prefix) — a ceiling catches that robustly without false-firing on terse
    # variance both engines share. `expect_words` (exact target) is recorded for
    # diagnostics but does NOT gate.
    constraint_ok = True
    constraint_detail = None
    n = _word_count(text)
    mw = turn.get("max_words")
    ew = turn.get("expect_words")
    if mw is not None:
        constraint_ok = (0 < n <= mw)
        constraint_detail = f"words={n} (ceiling {mw}" + (f", target {ew}" if ew else "") + ")"
    elif ew is not None:  # diagnostic-only target: record, never gate
        constraint_detail = f"words={n} (target {ew}, advisory)"
    ok = v.passed and correct and ontopic and constraint_ok
    return {
        "passed": ok,
        "dd_passed": v.passed,
        "dd_fired": v.failed_names(),
        "answer_correct": correct,
        "on_topic": ontopic,
        "constraint_ok": constraint_ok,
        "constraint_detail": constraint_detail,
        "finish_reason": finish,
        "text": text,
        "snippet": text[:160].replace("\n", " "),
    }


def run_conversation(base: str, convo: dict) -> dict:
    """Drive one scripted conversation turn-by-turn over the stateless protocol.
    Returns per-turn results, the full transcript (for determinism hashing /
    cache-equivalence), and a conversation-level pass = ALL turns pass."""
    messages: list[dict] = []
    turn_results = []
    for ti, turn in enumerate(convo["turns"]):
        messages.append({"role": "user", "content": turn["user"]})
        text, finish = query_messages(base, messages, turn.get("max_tokens", 256))
        messages.append({"role": "assistant", "content": text})
        r = score_turn(text, finish, turn)
        r["turn"] = ti
        turn_results.append(r)
    passed = all(r["passed"] for r in turn_results)
    transcript = [r["text"] for r in turn_results]
    return {
        "id": convo["id"],
        "passed": passed,
        "turns": turn_results,
        "transcript": transcript,
        "final_messages": messages,
    }


def _transcript_hash(transcript: list[str]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in transcript:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # turn separator (avoids boundary-merge collisions)
    return h.hexdigest()


def run_gq005(base: str, convos: list[dict], variants: set[str],
              respin=None, teardown_warm=None) -> dict:
    """GQ-014 driver.

    Base run: every conversation, every turn scored; conversation passes iff
    all turns pass; gate passes iff all conversations pass.

    variant 'determinism': re-run each conversation a SECOND time; the two
    transcripts must be byte-identical (sha256). Non-determinism across an
    identical stateless replay is a real defect (links DET-001).

    variant 'cache-equiv': re-issue the FINAL turn of each conversation against
    a FRESHLY RESTARTED server (cold prefill of the SAME full history). The
    final answer must be token-identical to the warm-cache base run. This is the
    multi-turn analog of CORR-010 KV-cache equivalence: a divergence between a
    warm growing-prefix cache and a cold full re-prefill of the same context is
    a KV / cache-management bug (high severity). `respin` is a zero-arg callable
    returning a fresh base URL (server restarted); None disables the variant.
    """
    base_runs = [run_conversation(base, c) for c in convos]
    npass = sum(1 for r in base_runs if r["passed"])
    n = len(base_runs)

    det = None
    if "determinism" in variants:
        det = []
        for c, br in zip(convos, base_runs):
            r2 = run_conversation(base, c)
            h1 = _transcript_hash(br["transcript"])
            h2 = _transcript_hash(r2["transcript"])
            det.append({"id": c["id"], "identical": h1 == h2, "h1": h1[:12], "h2": h2[:12]})

    cache = None
    if "cache-equiv" in variants and respin is not None:
        # Tear the WARM server down BEFORE spawning the cold one. The warm
        # transcripts are already captured in base_runs, so we no longer need the
        # warm process — and for large models (e.g. 27B bf16 ~52 GB) a second
        # full-precision server cannot co-reside with the warm one on a single
        # 80 GB GPU (the cold respawn OOMs). Freeing the warm server first makes
        # the cold cold-prefill memory-safe on one GPU.
        if teardown_warm is not None:
            teardown_warm()
        cold_base = respin()
        cache = []
        for c, br in zip(convos, base_runs):
            msgs = br["final_messages"][:-1]  # drop the warm assistant reply
            final_turn = c["turns"][-1]
            cold_text, _ = query_messages(cold_base, msgs, final_turn.get("max_tokens", 256))
            warm_text = br["transcript"][-1]
            cache.append({
                "id": c["id"],
                "identical": cold_text == warm_text,
                "warm": warm_text[:100].replace("\n", " "),
                "cold": cold_text[:100].replace("\n", " "),
            })

    base_ok = npass == n
    det_ok = det is None or all(d["identical"] for d in det)
    cache_ok = cache is None or all(c["identical"] for c in cache)
    passed = base_ok and det_ok and cache_ok
    status = "PASS" if passed else "FAIL"

    ev = f"{npass}/{n} conversations pass (all-turns, threshold 1.00). "
    fails = [r for r in base_runs if not r["passed"]]
    if fails:
        parts = []
        for r in fails[:6]:
            bad = [f"t{t['turn']}[" +
                   ",".join(filter(None, [
                       ("dd:" + ",".join(t["dd_fired"])) if t["dd_fired"] else "",
                       "ans✗" if not t["answer_correct"] else "",
                       "topic✗" if not t["on_topic"] else "",
                       ("constr✗ " + (t["constraint_detail"] or "")) if not t["constraint_ok"] else "",
                   ])) + "]"
                   for t in r["turns"] if not t["passed"]]
            parts.append(f"{r['id']}: " + " ".join(bad))
        ev += "Fails: " + " | ".join(parts)
    if det is not None:
        ndet = sum(1 for d in det if d["identical"])
        ev += f" | determinism {ndet}/{len(det)} byte-identical"
        if ndet != len(det):
            ev += " [" + ",".join(d["id"] for d in det if not d["identical"]) + "]"
    if cache is not None:
        ncache = sum(1 for c in cache if c["identical"])
        ev += f" | cache-equiv {ncache}/{len(cache)} final-turn token-identical"
        if ncache != len(cache):
            ev += " [" + ",".join(c["id"] for c in cache if not c["identical"]) + "]"

    return {
        "gate": "GQ-014", "status": status, "evidence": ev,
        "results": base_runs, "determinism": det, "cache_equiv": cache,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--server")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--gates", default="GQ-001,GQ-002,GQ-004")
    ap.add_argument("--out", default="/tmp/qsuite")
    ap.add_argument("--server-bin", default=None)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="server context length (GQ-014 long-context turn needs >=4096)")
    ap.add_argument("--mt-variants", default="determinism,cache-equiv",
                    help="GQ-014 extra variants (comma): determinism, cache-equiv. "
                         "cache-equiv needs a respawnable server (--model, not --server).")
    args = ap.parse_args()

    proc_holder = [None]  # warm server process (None when --server / torn down)
    base = args.server
    try:
        if not base:
            if not args.model:
                sys.exit("need --server or --model")
            proc_holder[0] = spin_server(args.model, args.backend, args.port, ctx=args.ctx,
                                         server_bin=args.server_bin)
            base = f"http://127.0.0.1:{args.port}"
            if not wait_ready(base):
                print(f"[{args.cell}] SERVER_NOT_READY (see /tmp/qsuite-server-{args.port}.log)")
                return 2

        # cache-equiv variant needs to restart THIS server on a fresh port; only
        # possible when we own the process (--model spin, not an external --server).
        mt_variants = {v.strip() for v in args.mt_variants.split(",") if v.strip()}
        respin = None
        teardown_warm = None
        cold = {"proc": None}
        if "cache-equiv" in mt_variants and proc_holder[0] is not None and args.model:
            def _respin():
                cold_port = args.port + 100
                if cold["proc"] is not None:
                    try:
                        os.killpg(os.getpgid(cold["proc"].pid), signal.SIGKILL)
                    except Exception:
                        pass
                cold["proc"] = spin_server(args.model, args.backend, cold_port,
                                           ctx=args.ctx, server_bin=args.server_bin)
                cold_base = f"http://127.0.0.1:{cold_port}"
                if not wait_ready(cold_base):
                    raise RuntimeError("cold respawn server not ready")
                return cold_base
            respin = _respin

            def _teardown_warm():
                # Free the warm server's GPU memory before the cold respawn so a
                # large model's cold-prefill server can fit on the same GPU
                # (27B bf16 ~52 GB cannot co-reside twice on one 80 GB GPU).
                if proc_holder[0] is not None:
                    try:
                        os.killpg(os.getpgid(proc_holder[0].pid), signal.SIGKILL)
                    except Exception:
                        pass
                    proc_holder[0] = None  # also prevents double-kill in finally
            teardown_warm = _teardown_warm
        elif "cache-equiv" in mt_variants:
            print(f"[{args.cell}] NOTE: cache-equiv variant disabled (needs --model spin, not external --server)")
            mt_variants.discard("cache-equiv")

        rows = []
        for gate in args.gates.split(","):
            gate = gate.strip()
            if gate not in GATES:
                rows.append({"gate": gate, "status": "DEFERRED", "evidence": "not a self-contained gate (needs llama / tools / judge driver)"})
                continue
            print(f"[{args.cell}] running {gate} ...", flush=True)
            if gate == "GQ-014":
                convos = load_corpus(GATES["GQ-014"][0])
                if not convos:
                    rows.append({"gate": gate, "status": "DEFERRED", "evidence": "no corpus multiturn.jsonl"})
                    continue
                rows.append(run_gq005(base, convos, mt_variants, respin=respin,
                                      teardown_warm=teardown_warm))
            else:
                rows.append(run_gate(base, gate))
        if cold["proc"] is not None:
            try:
                os.killpg(os.getpgid(cold["proc"].pid), signal.SIGKILL)
            except Exception:
                pass

        # results table
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"cell-{args.cell}.json").write_text(json.dumps(rows, indent=1))
        npass = sum(1 for r in rows if r["status"] == "PASS")
        nfail = sum(1 for r in rows if r["status"] == "FAIL")
        print()
        print(f"=== CELL {args.cell} — self-contained quality gates ===")
        print(f"| Gate | Status | Evidence |")
        print(f"|------|--------|----------|")
        for r in rows:
            sym = {"PASS": "✓", "FAIL": "✗", "DEFERRED": "·"}[r["status"]]
            print(f"| {r['gate']} | {sym} {r['status']} | {r['evidence']} |")
        verdict = "PRISTINE (self-contained subset)" if nfail == 0 else "NOT PRISTINE"
        print(f"\nCELL {args.cell}: {npass} ✓ / {nfail} ✗  → {verdict}")
        return 0 if nfail == 0 else 1
    finally:
        if proc_holder[0] is not None:
            try:
                os.killpg(os.getpgid(proc_holder[0].pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
