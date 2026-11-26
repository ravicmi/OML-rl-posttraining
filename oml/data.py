"""Dataset loaders. All return HF Datasets with columns: prompt (chat messages), answer, level, uid.

HF dataset ids below are from memory -- verify on the cluster (`python -m oml.data` prints a sample of each).
"""
from __future__ import annotations

import re

from datasets import Dataset, load_dataset

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def _chat(q: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]


def _boxed_from_solution(sol: str):
    from .rewards import last_boxed

    return last_boxed(sol)


def load_math_train(max_samples: int | None = None) -> Dataset:
    """MATH train (7.5k) with Level 1-5 -- free difficulty prior."""
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    rows = []
    for i, ex in enumerate(ds):
        ans = _boxed_from_solution(ex["solution"])
        if ans is None:
            continue
        lvl = ex.get("level", "Level ?")
        m = re.search(r"(\d)", lvl)
        rows.append({"prompt": _chat(ex["problem"]), "answer": ans, "level": int(m.group(1)) if m else 0, "uid": f"math-{i}"})
    ds = Dataset.from_list(rows)
    return ds.select(range(min(max_samples, len(ds)))) if max_samples else ds


def load_gsm8k_train(max_samples: int | None = None) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = [
        {"prompt": _chat(ex["question"]), "answer": ex["answer"].split("####")[-1].strip().replace(",", ""), "level": 0, "uid": f"gsm-{i}"}
        for i, ex in enumerate(ds)
    ]
    ds = Dataset.from_list(rows)
    return ds.select(range(min(max_samples, len(ds)))) if max_samples else ds


def load_eval(name: str) -> Dataset:
    """Eval sets: math500, gsm8k, aime24."""
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"prompt": _chat(e["problem"]), "answer": e["answer"], "level": e.get("level", 0), "uid": f"m500-{i}"} for i, e in enumerate(ds)]
    elif name == "math_rest":
        # the 4,500 MATH test problems that are NOT in MATH-500: a larger, independent held-out set (lower-variance than MATH-500)
        m500 = {e["problem"] for e in load_dataset("HuggingFaceH4/MATH-500", split="test")}
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
        rows = []
        for i, e in enumerate(ds):
            ans = _boxed_from_solution(e["solution"])
            if ans is None or e["problem"] in m500:
                continue
            m = re.search(r"(\d)", e.get("level", ""))
            rows.append({"prompt": _chat(e["problem"]), "answer": ans, "level": int(m.group(1)) if m else 0, "uid": f"mtest-{i}"})
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"prompt": _chat(e["question"]), "answer": e["answer"].split("####")[-1].strip().replace(",", ""), "level": 0, "uid": f"gsm-t{i}"} for i, e in enumerate(ds)]
    elif name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        rows = [{"prompt": _chat(e["problem"]), "answer": str(e["answer"]), "level": 5, "uid": f"aime24-{i}"} for i, e in enumerate(ds)]
    else:
        raise ValueError(f"unknown eval set {name}")
    return Dataset.from_list(rows)


def load_train(name: str, max_samples: int | None = None) -> Dataset:
    return {"math": load_math_train, "gsm8k": load_gsm8k_train}[name](max_samples)


if __name__ == "__main__":
    for n in ["math500", "gsm8k", "aime24"]:
        d = load_eval(n)
        print(n, len(d), d[0]["answer"])
    d = load_math_train()
    print("math train", len(d), d[0]["answer"], "levels:", sorted(set(d["level"])))
