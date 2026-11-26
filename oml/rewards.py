"""Verifiable math reward: extract the last \\boxed{...} and check it against the gold answer."""
from __future__ import annotations

import re


def last_boxed(text: str):
    """Return the content of the last \\boxed{...} (brace-balanced), or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


def _norm(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\!", "").replace("\\,", "").replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("\\%", "")
    return s.rstrip(".")


def is_correct(completion: str, gold: str) -> bool:
    pred = last_boxed(completion)
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    try:  # symbolic equivalence when math-verify is installed
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


def has_box(completion: str) -> bool:
    return last_boxed(completion) is not None


def _text(c) -> str:
    # TRL passes conversational completions as [{"role": "assistant", "content": "..."}]
    return c[0]["content"] if isinstance(c, list) else c


def correctness_reward(prompts, completions, answer, **kwargs):
    return [1.0 if is_correct(_text(c), a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(prompts, completions, **kwargs):
    """Small shaping reward for emitting a boxed answer (ablate with format_weight=0)."""
    return [1.0 if has_box(_text(c)) else 0.0 for c in completions]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod
