#!/usr/bin/env python3
"""DD-* degeneration detectors for the Model Quality Acceptance Suite (Section 10).

Every generative GQ-* gate runs decoded text (+ optional token-id stream +
finish_reason) through these mechanical, length-aware detectors (AH-1). A gate
FAILs if ANY applicable detector fires. Detectors are intentionally strict: the
reference engine (llama.cpp, same quant) proves the clean continuation is
reachable, so a Lumen degeneration is a defect, never "a different valid path"
(AH-15).

Run `python3 detectors.py --selftest` to validate against captured real defects.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from dataclasses import dataclass, field


@dataclass
class DetectorResult:
    name: str
    passed: bool
    detail: str


@dataclass
class QualityVerdict:
    passed: bool
    results: list[DetectorResult] = field(default_factory=list)

    def failed_names(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _ngrams(seq: list, n: int) -> list[tuple]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


# ---------------------------------------------------------------------------
# DD-REP — repetition: distinct-n-gram ratio + max n-gram count (length-aware)
# ---------------------------------------------------------------------------

def dd_rep(text: str, token_ids: list[int] | None = None) -> DetectorResult:
    units = token_ids if token_ids is not None else _words(text)
    n = 4
    grams = _ngrams(units, n)
    if len(grams) < n:
        return DetectorResult("DD-REP", True, "too short to assess")
    total = len(grams)
    distinct = len(set(grams))
    ratio = distinct / total
    floor = 0.65 if len(units) < 512 else 0.55
    # max allowed count of any single n-gram
    counts: dict[tuple, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    # Markdown-structural guard (mirrors dd_loop's): a 4-column table's
    # alignment row "| :--- | :--- | :--- | :--- |" repeats the structural
    # 4-gram ('|', ':---', '|', ':---') once per table — 3 tables in a
    # 256-word window exceeds the windowed max_allowed of 2 and false-fired
    # on a clean, complete, finish=stop coffee guide (2026-06-11 fidelity
    # calibration, N=3 byte-identical capture). Only CONTENT-carrying 4-grams
    # (any unit with an alphanumeric char) count toward the top-count gate;
    # real degenerate loops repeat words/numbers, and pure punctuation spam
    # is still caught by DD-CHARSPAM. The distinct-ratio floor below is
    # unaffected (it sees all grams).
    def _is_structural(g: tuple) -> bool:
        return all(not re.search(r"[A-Za-z0-9]", str(u)) for u in g)
    content_counts = {g: c for g, c in counts.items() if not _is_structural(g)}
    if not content_counts:
        content_counts = counts
    # Shared-stem guard (2026-06-11 fidelity calibration): a real degenerate
    # loop repeats the stem AND its continuation ("17 times 20 = 17 times 20
    # = ..."); natural technical prose repeats a stem with DISTINCT correct
    # continuations ("The browser sends a" -> TCP SYN / TCP ACK / TLS Client
    # Hello — full-text distinct-4gram ratio 0.995, finish=stop, verified
    # gold-standard output that false-fired). Only count a 4-gram toward the
    # top-count gate if its occurrences share continuations (i.e. the
    # follow-up unit is NOT distinct each time). Identical-continuation loops
    # still fire; the distinct-ratio floor below is unaffected.
    #
    # n=4 WINDOW-CLIP fix (2026-06-12 fidelity calibration): the diverging
    # content word can sit ONE token PAST the immediate continuation, so a
    # single-offset look-ahead mis-classifies natural enumeration as a loop.
    # short-arith-05 (27b @ AP=2, finish=stop, answer 963 CORRECT) emits three
    # correct borrowing lines:
    #     * The $0$ in the hundreds place becomes $9$ ...
    #     * The $0$ in the tens     place becomes $9$ ...
    #     * The $0$ in the ones     place becomes $10$.
    # The 4-gram ('*','The','$0$','in') repeats 3x and its IMMEDIATE next word
    # is 'the' all three times (single-offset diversity 0.33 -> looks looped),
    # but the genuinely diverging content word (hundreds / tens / ones) is the
    # NEXT token, i.e. offset +2 from the gram. Computing diversity over a SHORT
    # look-ahead WINDOW (offsets +1..+WIN) and taking the MAX recovers the real
    # divergence: a true loop repeats the SAME sequence at every offset (window
    # diversity stays ~0 -> still fires), while natural enumeration diverges
    # within a couple tokens (window diversity >= 0.75 -> suppressed). The
    # distinct-ratio floor below is unaffected and a genuinely high repeat count
    # over a SHORT output is still caught by that floor.
    _CONT_WINDOW = 2

    def _continuation_diversity(g: tuple) -> float:
        starts = [i for i in range(len(units) - n)
                  if tuple(units[i:i + n]) == g]
        if len(starts) <= 1:
            return 1.0
        best = 0.0
        for off in range(1, _CONT_WINDOW + 1):
            nexts = [units[i + n + off - 1] for i in starts
                     if i + n + off - 1 < len(units)]
            if len(nexts) <= 1:
                # not enough tail to judge this offset; treat as diverse so we
                # never UPGRADE a near-end stem into a loop on missing context.
                best = max(best, 1.0)
                continue
            best = max(best, len(set(nexts)) / len(nexts))
        return best
    top_gram, top_count = max(content_counts.items(), key=lambda kv: kv[1])
    max_allowed = max(2, len(units) // 200)
    if top_count > max_allowed and _continuation_diversity(top_gram) >= 0.75:
        # natural shared-stem phrasing, not a loop: pick the worst gram whose
        # continuations DO repeat (if any) for the gate instead.
        looping = {g: c for g, c in content_counts.items()
                   if c > max_allowed and _continuation_diversity(g) < 0.75}
        if looping:
            top_gram, top_count = max(looping.items(), key=lambda kv: kv[1])
        else:
            top_count = 0  # no loop-like gram exceeds the gate
    if ratio < floor:
        return DetectorResult(
            "DD-REP", False,
            f"distinct-4gram ratio {ratio:.3f} < floor {floor} (len={len(units)})",
        )
    if top_count > max_allowed:
        return DetectorResult(
            "DD-REP", False,
            f"4-gram {top_gram!r} repeats {top_count}x > max {max_allowed}",
        )
    return DetectorResult("DD-REP", True, f"ratio {ratio:.3f}, top-4gram {top_count}x")


# ---------------------------------------------------------------------------
# DD-LOOP — no period-p (p<=32) token cycle repeated >=3x anywhere
# ---------------------------------------------------------------------------

def dd_loop(text: str, token_ids: list[int] | None = None) -> DetectorResult:
    units = token_ids if token_ids is not None else _words(text)
    m = len(units)
    for p in range(1, 33):
        if m < 3 * p:
            break
        # slide a window; check any position where unit[i:i+p] repeats >=3x consecutively
        i = 0
        while i + 3 * p <= m:
            block = units[i : i + p]
            reps = 1
            j = i + p
            while j + p <= m and units[j : j + p] == block:
                reps += 1
                j += p
            if reps >= 3:
                # Skip markdown/punctuation-STRUCTURAL cycles: a markdown table
                # alignment row "| :--- | :--- | :--- |" tokenizes to a period-2
                # block ['|', ':---'] repeated per column — legitimate formatting,
                # not degeneration. Likewise "--- --- ---" dividers. Only fire when
                # the repeating cycle carries CONTENT (a unit with an alphanumeric
                # char); real degenerate loops repeat words/numbers, and char-level
                # punctuation spam is still caught by DD-CHARSPAM. Verified against a
                # clean, complete coffee-guide capture that DD-LOOP false-fired on.
                if any(re.search(r"[A-Za-z0-9]", str(u)) for u in block):
                    return DetectorResult(
                        "DD-LOOP", False,
                        f"period-{p} block repeated {reps}x at unit {i}: {block!r}",
                    )
            i += 1
    return DetectorResult("DD-LOOP", True, "no cycle")


# ---------------------------------------------------------------------------
# DD-SUBWORD — zero suffix-re-emit within a word ("multiplicationlication")
# ---------------------------------------------------------------------------

_ALPHA_RUN = re.compile(r"[A-Za-z]{8,}")


def dd_subword(text: str) -> DetectorResult:
    for w in _ALPHA_RUN.findall(text):
        lw = w.lower()
        n = len(lw)
        # split w = a + b, b a suffix of a, |b|>=4, |a|>=4
        for i in range(4, n - 3):
            a, b = lw[:i], lw[i:]
            if len(b) >= 4 and a.endswith(b):
                return DetectorResult(
                    "DD-SUBWORD", False,
                    f"sub-word doubling in {w!r}: {a!r}+{b!r} (b is suffix of a)",
                )
    return DetectorResult("DD-SUBWORD", True, "no sub-word doubling")


# ---------------------------------------------------------------------------
# DD-SPAM — no single token/word > 20% of a >50-unit output
# ---------------------------------------------------------------------------

def dd_spam(text: str, token_ids: list[int] | None = None) -> DetectorResult:
    units = token_ids if token_ids is not None else [w.lower() for w in _words(text)]
    if len(units) <= 50:
        return DetectorResult("DD-SPAM", True, "too short to assess")
    counts: dict = {}
    for u in units:
        counts[u] = counts.get(u, 0) + 1
    top, c = max(counts.items(), key=lambda kv: kv[1])
    frac = c / len(units)
    if frac > 0.20:
        return DetectorResult("DD-SPAM", False, f"unit {top!r} is {frac:.1%} of output")
    return DetectorResult("DD-SPAM", True, f"max unit {frac:.1%}")


# ---------------------------------------------------------------------------
# DD-CHARSPAM — char-level periodic spam (catches space-less degenerate output
# like "[PAD248319][PAD248319]..." that word-level detectors miss)
# ---------------------------------------------------------------------------

def dd_charspam(text: str) -> DetectorResult:
    s = re.sub(r"\s+", "", text)
    if len(s) < 60:
        return DetectorResult("DD-CHARSPAM", True, "too short to assess")
    n = 8
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    ratio = len(set(grams)) / len(grams)
    if ratio < 0.15:
        return DetectorResult("DD-CHARSPAM", False, f"char-8gram distinct ratio {ratio:.3f} < 0.15 (periodic spam)")
    return DetectorResult("DD-CHARSPAM", True, f"char-8gram ratio {ratio:.3f}")


# ---------------------------------------------------------------------------
# DD-ARITH — every stated equality chain must have all sides equal
# ---------------------------------------------------------------------------

# Marks a stripped `\text{}` unit inside a normalized expression. Chosen because
# it never occurs in real arithmetic and `_EXPR` accepts it (so it does not split
# an expression); dd_arith skips any chain containing it.
_UNIT_FLAG = "¤"


def _normalize_math(text: str) -> str:
    s = text
    # Replace LaTeX text/unit blocks with the UNIT FLAG (`_UNIT_FLAG`, U+00A4 ¤).
    # Rationale: a `\text{}` unit anywhere in an equation means the model is doing
    # dimensional math, not a pure numeric identity — "30 \text{min} = 0.5 \text{h}"
    # is physically true yet "30 = 0.5" is a false NUMERIC equation, and
    # "60 \text{km} * 2 = 120 \text{km}" is true (60*2=120) but stripping the unit
    # to a SPACE leaves "60 * 2 = 120" (ok) while stripping it to a breaking sentinel
    # ORPHANS "2 = 120" (false fire). The flag is a char that `_EXPR` ACCEPTS (so it
    # never splits an expression and orphans a fragment); dd_arith then SKIPS any
    # chain that contains it (a unit-bearing relation is not a numeric identity).
    # Genuine degenerations ("491 * 23 = 391", "120 + 39 = 169") carry no `\text{}`
    # and are still evaluated and fired.
    # No surrounding spaces on the flag: extra spaces would create a 2+ space run
    # that dd_arith's display-equation split breaks on, orphaning the equation from
    # its flag. The flag is in `_EXPR`, so it attaches to the adjacent number.
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{[^{}]*\}", _UNIT_FLAG, s)
    s = re.sub(r"\\(?:,|;|:|!|quad|qquad|;|\s)", " ", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("\\left", "").replace("\\right", "")
    # Replace math delimiters with a SPACE (not empty) so "$$a=b$$ $$c=d$$" leaves a
    # 2+ space gap that dd_arith's display-equation split keys on (a lone "$$ $$"
    # collapses to a single space if delimiters vanish, merging the two equations).
    s = s.replace("$", " ").replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    # exponent a^b -> a**b (so "10^2 = 100" is one valid expression, not "2 = 100")
    s = re.sub(r"\^", "**", s)
    # "17 x 23" / "17X23" (digit x digit) -> multiply
    s = re.sub(r"(\d)\s*[xX]\s*(\d)", r"\1*\2", s)
    return s


# a single arithmetic expression: digits, + - * / ( ) . spaces, and the unit flag
# (accepted so a `\text{}` unit mid-expression does not orphan a fragment; chains
# containing the flag are skipped in dd_arith), >=1 char
_EXPR = r"[0-9][0-9+\-*/(). ¤]*"
_CHAIN_RE = re.compile(rf"({_EXPR})(?:=\s*({_EXPR}))+")
_EQ_SPLIT = re.compile(r"=")
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
    ast.Num if hasattr(ast, "Num") else ast.Constant,
)


def _safe_eval(expr: str):
    expr = expr.strip().strip("=").rstrip("+-*/.( ").lstrip(")+*/. ").strip()
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    for sub in ast.walk(node):
        if not isinstance(sub, _SAFE_NODES):
            return None
        # bound exponentiation to avoid pathological values / slow big-ints
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            exp = sub.right
            if isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) and abs(exp.value) > 64:
                return None
    try:
        return eval(compile(node, "<arith>", "eval"))  # safe: whitelisted AST
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def dd_arith(text: str) -> DetectorResult:
    s = _normalize_math(text)
    # Split on display-equation boundaries (runs of 2+ spaces / newlines, which is
    # what "$$a=b$$ $$c=d$$" collapses to after $-stripping) so adjacent equations
    # are evaluated INDEPENDENTLY. A single greedy scan across the gap merges them
    # into one super-chain whose middle side ("13*13  13*10") fails to parse, which
    # silently drops the WHOLE chain — a missed fire. Real model output usually has
    # prose between displays that breaks the merge, but back-to-back displays do not.
    for seg in re.split(r"\s{2,}|\n+", s):
        for m in re.finditer(rf"{_EXPR}(?:=\s*{_EXPR})+", seg):
            chain = m.group(0)
            if _UNIT_FLAG in chain:
                # unit-bearing relation ("30 min = 0.5 h", "60 km * 2 = 120 km"):
                # the bare numbers carry units, so it is not a numeric identity.
                continue
            sides = [p for p in _EQ_SPLIT.split(chain)]
            vals = []
            for side in sides:
                v = _safe_eval(side)
                if v is None:
                    vals = []
                    break
                vals.append(v)
            if len(vals) >= 2:
                first = vals[0]
                for v in vals[1:]:
                    if not math.isclose(v, first, rel_tol=1e-6, abs_tol=1e-6):
                        return DetectorResult(
                            "DD-ARITH", False,
                            f"false equation: {chain.strip()!r} -> sides {vals}",
                        )
    return DetectorResult("DD-ARITH", True, "all stated equations true")


# ---------------------------------------------------------------------------
# DD-TERM — clean EOS expected (task-specific; very-long inverts)
# ---------------------------------------------------------------------------

def dd_term(finish_reason: str | None, expect_eos: bool = True) -> DetectorResult:
    if finish_reason is None:
        return DetectorResult("DD-TERM", True, "finish_reason not provided")
    if expect_eos and finish_reason != "stop":
        return DetectorResult("DD-TERM", False, f"finish_reason={finish_reason}, expected stop/EOS")
    return DetectorResult("DD-TERM", True, f"finish_reason={finish_reason}")


# ---------------------------------------------------------------------------
# evaluate — run the applicable detector set
# ---------------------------------------------------------------------------

def evaluate(
    text: str,
    token_ids: list[int] | None = None,
    finish_reason: str | None = None,
    is_math: bool = False,
    expect_eos: bool = True,
    check_term: bool = False,
) -> QualityVerdict:
    results = [
        dd_rep(text, token_ids),
        dd_loop(text, token_ids),
        dd_subword(text),
        dd_spam(text, token_ids),
        dd_charspam(text),
    ]
    if is_math:
        results.append(dd_arith(text))
    if check_term:
        results.append(dd_term(finish_reason, expect_eos))
    return QualityVerdict(passed=all(r.passed for r in results), results=results)


# ---------------------------------------------------------------------------
# self-test against captured REAL defect outputs (no model needed)
# ---------------------------------------------------------------------------

_SELFTEST = [
    # (label, text, is_math, expect_fail, expect_detectors_firing)
    (
        "q8_MoE_doubling (REAL capture)",
        r"To compute $17 \times 23$, we can break down the multiplicationlication into "
        r"simpler parts using the distributive property. $$17 \times 23 = 17 \times (20 + 3)$$ "
        r"$$= (17 \times 20) + (17 \times 3)$$ $$= 17 \times 20 = 17 \times 20 = 340$$ "
        r"$$= 17 \times 3 = 51$$ $$= 340 + 51 = 391$$",
        # q8's arithmetic is individually CORRECT (reaches 391 via true steps);
        # its ONLY defect is the cosmetic sub-word doubling -> DD-SUBWORD alone.
        True, True, {"DD-SUBWORD"},
    ),
    (
        "q4_MoE_garble (REAL capture)",
        r"To compute $17 \times 23$, we can break down the calculation: "
        r"$$17 \times 23 = 491 \times 23 = 391$$ Let's verify using the distributive property: "
        r"$$17 \times 23 = 17 \times (20 + 3) = (17 \times 20) + (17 \times 3) = 340 + 51 = 391$$",
        True, True, {"DD-ARITH"},
    ),
    (
        "bf16_MoE_clean (REAL capture, should PASS)",
        r"To compute $17 \times 23$, use the distributive property: "
        r"$$17 \times 23 = 17 \times (20 + 3) = 17 \times 20 + 17 \times 3 = 340 + 51 = 391$$ "
        r"The final numeric answer is 391.",
        True, False, set(),
    ),
    (
        "hard_loop (synthetic, rep collapse)",
        "17 times 20 = 17 times 20 = 17 times 20 = 17 times 20 = 17 times 20 = 17 times 20 = ",
        True, True, {"DD-REP", "DD-LOOP"},
    ),
    (
        "clean_prose (should PASS)",
        "The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, "
        "and the atmosphere scatters shorter blue wavelengths more strongly than longer red ones, "
        "so the daytime sky looks predominantly blue to an observer on the ground.",
        False, False, set(),
    ),
    (
        "legit_long_words (false-positive guard, should PASS)",
        "The implementation of the representation requires understanding the characteristics and "
        "responsibilities of international organizations and their administrative infrastructure.",
        False, False, set(),
    ),
    (
        # REAL capture (q8 reason-01, ans 120 km/h CORRECT, finish=stop): a LaTeX
        # unit conversion "30 min = 0.5 h" must NOT false-fire DD-ARITH. The bare
        # numbers 30 and 0.5 differ ONLY because the units differ — a physical
        # identity, not a false arithmetic claim. Regression guard for the
        # _normalize_math unit-sentinel fix.
        "unit_conversion (REAL q8 reason-01 false-fire guard, should PASS)",
        r"To find the speed of the train in kilometers per hour (km/h): "
        r"Distance: $60 \text{ km}$. Time: $30 \text{ minutes}$. "
        r"Convert the time to hours: $30 \text{ min} = 0.5 \text{ h}$. "
        r"Then the speed is $\frac{60 \text{ km}}{0.5 \text{ h}} = 120 \text{ km/h}$.",
        True, False, set(),
    ),
    (
        # REAL capture (bf16 arith-04, 13^2): the intermediate "120 + 39 = 169" is a
        # GENUINE false numeric equation (120+39=159, model misread 130->120) and
        # MUST still fire DD-ARITH even though the FINAL answer 169 is correct. Locks
        # in that the unit-sentinel fix does NOT mask real number-misread defects.
        "number_misread (REAL bf16 arith-04 genuine defect, should FAIL DD-ARITH)",
        r"To find 13 squared, multiply 13 by itself: $$13^2 = 13 \times 13$$ "
        r"$$13 \times 10 = 130$$ $$13 \times 3 = 39$$ $$120 + 39 = 169$$",
        True, True, {"DD-ARITH"},
    ),
    (
        # REAL capture (q8 vlong-guide, a CLEAN+COMPLETE coffee guide): DD-LOOP
        # false-fired on the markdown TABLE alignment row "| :--- | :--- | :--- |"
        # (period-2 ['|', ':---'] x3 — legitimate formatting). With enough prose
        # around it that DD-SPAM on '|' stays under 20% (as in the real 1206-word
        # output). Must PASS — structural markdown is not degeneration.
        "markdown_table (REAL q8 vlong DD-LOOP false-fire guard, should PASS)",
        "Brewing great coffee comes down to a few controllable variables: grind size, "
        "water temperature, brew time, and the coffee-to-water ratio. Start with a 1:16 "
        "ratio and adjust to taste. Use water just off the boil, and aim for a total brew "
        "time appropriate to your chosen method. The table below summarizes the most "
        "common problems and how to correct each one as you dial in your technique.\n\n"
        "| Symptom | Likely Cause | Suggested Fix |\n"
        "| :--- | :--- | :--- |\n"
        "| Sour and sharp | Under-extraction | Grind finer or use hotter water. |\n"
        "| Bitter and dry | Over-extraction | Grind coarser or use cooler water. |\n"
        "| Weak and watery | Ratio too low | Use more coffee relative to the water. |\n\n"
        "Keep a simple log of every brew and change only one variable at a time. With a "
        "little patient practice you will develop a reliable instinct for the small "
        "adjustments that matter most. Happy brewing!",
        False, False, set(),
    ),
    (
        # GQ-005 multi-turn shape: a single reply under a running "answer in
        # exactly three words" constraint. Detectors run PER TURN, so each reply
        # is scored in isolation — a legitimately terse 3-word answer must NOT
        # false-fire any DD. (The word-count constraint itself is enforced by the
        # driver's expect_words check, not by the detectors.) Brevity is not
        # degeneration; this guards the constraint conversations (mt-constraint-3words).
        "multiturn_3word_answer (GQ-005 brevity guard, should PASS)",
        "The capital is Paris.",
        False, False, set(),
    ),
    (
        # GQ-005 multi-turn shape: a real WITHIN-REPLY degenerate loop must still
        # fire even though the reply is short. A constrained conversation that
        # collapses into "blue blue blue ..." is a genuine defect, not brevity.
        # Locks in that the brevity guard above did not soften strictness.
        "multiturn_short_loop (GQ-005 real-loop must fire, should FAIL)",
        "blue blue blue blue blue blue blue blue blue blue blue blue",
        False, True, {"DD-REP", "DD-LOOP"},
    ),
    (
        # REAL capture (27b-q8/q4 short-arith-05 "Compute 1000 minus 37" @ AP=2,
        # finish=stop, answer 963 CORRECT — 2026-06-12 fidelity calibration).
        # The 4-gram ('*','The','$0$','in') repeats 3x across three CORRECT
        # borrowing lines; the diverging content word (hundreds/tens/ones) sits
        # at look-ahead offset +2, so the single-offset shared-stem guard
        # mis-fired DD-REP. The window-diversity fix (offsets +1..+2) suppresses
        # it. Pure correct math pedagogy — must PASS.
        "short_arith_borrowing (REAL 27b short-arith-05 DD-REP FP, should PASS)",
        r"To compute 1000 minus 37, I will use subtraction with borrowing. "
        r"Since 1000 has zeros, we borrow across the places: "
        r"* The $0$ in the hundreds place becomes $9$ after borrowing. "
        r"* The $0$ in the tens place becomes $9$ after borrowing. "
        r"* The $0$ in the ones place becomes $10$ before subtracting. "
        r"Now subtract: $10 - 7 = 3$ in the ones, $9 - 3 = 6$ in the tens, "
        r"$9 - 0 = 9$ in the hundreds, and $0$ thousands. So the answer is $963$.",
        False, False, set(),
    ),
    (
        # REAL capture (27b vlong-explain-01 "what happens when you type a URL"
        # @ AP=2, finish=stop, coherent — 2026-06-12 fidelity calibration). The DNS
        # recursion chain repeats the stem "the resolver queries the" with
        # DISTINCT correct continuations (root / TLD / authoritative servers).
        # The shared-stem continuation-diversity guard already passes this at
        # WORD granularity (the suite gate); this positive control LOCKS that in
        # so the window-diversity change does not regress it.
        "dns_recursion_stem (REAL 27b vlong-explain-01 DD-REP FP, should PASS)",
        "When you type a URL, the computer must resolve the domain name to an IP "
        "address. First, the resolver queries the root servers to find the "
        "responsible TLD. Then, the resolver queries the TLD servers to find the "
        "authoritative name server. Finally, the resolver queries the "
        "authoritative servers to obtain the final IP address. This recursive "
        "process resolves the hostname quickly thanks to layered caching.",
        False, False, set(),
    ),
    (
        # POSITIVE CONTROL: a GENUINE
        # degenerate loop must NOT slip through the relaxed window-diversity
        # guard. Here a stem with an IDENTICAL full continuation sequence is
        # looped 4x amid all-unique padding so the distinct-ratio stays HIGH
        # (the ratio floor does NOT catch it) — the ONLY thing that can catch it
        # is the top-count gate. A true loop repeats the same sequence at EVERY
        # look-ahead offset, so window-diversity stays ~0 and the gate fires.
        # This proves the window relaxation did not open a hole for real loops.
        "identical_loop (positive control: real loop must still fire, should FAIL)",
        " ".join(f"uniqueword{i}" for i in range(300))
        + " " + ("the cache stores the value forever and " * 4),
        False, True, {"DD-REP"},
    ),
    (
        # Positive control #1 (2026-06-12): a degenerate loop whose ONLY
        # variation is a throwaway COUNTER token at look-ahead offset +2 — the
        # exact inverse of the short-arith-05 FP (which diverges in content at +2).
        # This is the case most likely to slip the window-MAX relaxation: if the
        # guard wrongly rescued any gram with high diversity at ONE offset, the
        # "...batch N of the queue..." stutter would pass. It must FAIL: the tail
        # 4-gram after the counter re-converges identically, so a real loop always
        # leaves a caught gram. Proves the relaxation cannot be defeated by a
        # single varying token inside an otherwise-identical repeated phrase.
        "counter_loop (positive control: varying-token loop must fire, should FAIL)",
        "System busy. " + " ".join(f"x{i}" for i in range(60)) + " "
        + " ".join(f"please wait while processing batch {i} of the queue now"
                   for i in range(1, 6)),
        False, True, {"DD-REP"},
    ),
    (
        # Positive control #2 (2026-06-12): a genuine loop buried in a LONG
        # all-unique prefix so the distinct-4gram ratio stays well above the floor
        # (the ratio floor canNOT catch it) AND the loop count sits just past the
        # max_allowed boundary (max(2, len//200)). Only the per-gram top-count gate
        # can catch it. Locks in that a high-ratio output is not a free pass for a
        # real repeated degenerate phrase.
        "highratio_buried_loop (positive control: buried real loop must fire, should FAIL)",
        " ".join(f"token{i}" for i in range(400)) + " "
        + ("error code seven occurred during the read and " * 4),
        False, True, {"DD-REP"},
    ),
    (
        # Positive control #3 (2026-06-12): a near-boundary loop — exactly
        # 3 repeats of an identical-continuation 4-gram in a medium output, sitting
        # right at max_allowed=2 (3 > 2). Confirms the gate fires at the MINIMUM
        # loop count the relaxation could have softened, not only on egregious 4-5x
        # loops. Proves the window-diversity change did not raise the effective
        # firing threshold for genuine short loops.
        "near_boundary_loop (positive control: 3x identical loop must fire, should FAIL)",
        "Here is the analysis of the dataset over several dimensions and metrics. "
        + " ".join(f"w{i}" for i in range(40)) + " "
        + ("the value diverges to infinity rapidly " * 3)
        + " ".join(f"v{i}" for i in range(10)),
        False, True, {"DD-REP"},
    ),
]


def _selftest() -> int:
    ok = True
    for label, text, is_math, expect_fail, expect_firing in _SELFTEST:
        v = evaluate(text, is_math=is_math)
        fired = set(v.failed_names())
        passed_as_expected = (not v.passed) == expect_fail
        firing_ok = expect_firing.issubset(fired) if expect_fail else (len(fired) == 0)
        status = "OK" if (passed_as_expected and firing_ok) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"[{status}] {label}")
        print(f"        verdict={'PASS' if v.passed else 'FAIL'} (expected {'FAIL' if expect_fail else 'PASS'}); "
              f"fired={sorted(fired) or '∅'}; expected-fired⊇{sorted(expect_firing) or '∅'}")
        for r in v.results:
            if not r.passed:
                print(f"          ✗ {r.name}: {r.detail}")
    print()
    print("SELF-TEST:", "ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("usage: detectors.py --selftest", file=sys.stderr)
    sys.exit(2)
