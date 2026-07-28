#!/usr/bin/env python3
"""Graded degeneration scoring for long-form generations.

The binary DD-REP / DD-SPAM detectors could not separate activation
configurations: a per-family sweep ranked arms non-monotonically, with an arm
carrying MORE int8 scoring better than the all-F32 control. Threshold detectors
answer "did this cross the line", and on 2000-token generations individual
items sit close enough to the line that small numeric perturbations flip them
either way. Fifteen coin flips cannot measure a small effect.

These are continuous instead. Each is computed per generation, then compared
PAIRWISE across arms on the same prompt, so prompt difficulty cancels.

  distinct3     unique trigrams / total trigrams over the whole text.
                Falls as text repeats itself. The primary measure.
  worst_window  the same ratio over the WORST 200-word window. A generation
                that is fine for 1500 words and then loops scores badly here
                while `distinct3` barely moves — and looping late is exactly
                the failure mode long-form degeneration takes.
  max_run       longest run of an immediately repeated 5-gram. Catches hard
                loops that ratio measures dilute.
  tail_ratio    distinct3 of the last third over distinct3 of the first third.
                Below 1.0 means quality decays as generation proceeds.

Higher is better for distinct3, worst_window and tail_ratio; lower for
max_run. None has a pass threshold — the verdict comes from the paired
difference and its confidence interval, not from any single value.
"""
import re


def _words(text):
    return re.sub(r"\s+", " ", text.strip().lower()).split(" ")


def _distinct_n(words, n=3):
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _worst_window(words, win=200, n=3):
    if len(words) <= win:
        return _distinct_n(words, n)
    worst = 1.0
    for i in range(0, len(words) - win, max(1, win // 4)):
        worst = min(worst, _distinct_n(words[i:i + win], n))
    return worst


def _max_repeat_run(words, n=5):
    """Longest chain of back-to-back identical n-grams."""
    if len(words) < 2 * n:
        return 0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    best = run = 0
    for i in range(1, len(grams)):
        if grams[i] == grams[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def score(text):
    w = _words(text)
    third = max(1, len(w) // 3)
    head, tail = w[:third], w[-third:]
    d_head = _distinct_n(head)
    return {
        "words": len(w),
        "distinct3": round(_distinct_n(w), 5),
        "worst_window": round(_worst_window(w), 5),
        "max_run": _max_repeat_run(w),
        "tail_ratio": round(_distinct_n(tail) / d_head, 5) if d_head else 0.0,
    }


def paired_delta(cand_scores, ref_scores, key):
    """Mean paired difference (candidate - reference) with a bootstrap 95% CI.

    Paired on prompt id, so prompt difficulty cancels. The CI is what decides:
    an interval straddling zero means the corpus cannot resolve the effect,
    which is a real answer and the one the old gate could not give.
    """
    import random
    ids = sorted(set(cand_scores) & set(ref_scores))
    d = [cand_scores[i][key] - ref_scores[i][key] for i in ids]
    if not d:
        return None
    mean = sum(d) / len(d)
    rng = random.Random(42)
    boots = []
    for _ in range(4000):
        s = [d[rng.randrange(len(d))] for _ in range(len(d))]
        boots.append(sum(s) / len(s))
    boots.sort()
    return {"n": len(d), "mean": round(mean, 5),
            "ci_lo": round(boots[100], 5), "ci_hi": round(boots[3899], 5)}


if __name__ == "__main__":
    import json, sys
    print(json.dumps(score(sys.stdin.read()), indent=2))
