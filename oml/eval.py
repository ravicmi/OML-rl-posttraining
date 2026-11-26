"""Evaluate a checkpoint with vLLM: accuracy / pass@k / mean length / boxed-rate on math benchmarks.

  python -m oml.eval --model Qwen/Qwen3-0.6B --sets math500,gsm8k,aime24 --n 8 --out outputs/eval/base.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import load_eval
from .rewards import has_box, is_correct, pass_at_k


def evaluate(model: str, sets: list[str], n: int = 8, max_tokens: int = 4096, temperature: float = 0.7, top_p: float = 0.95,
             gpu_util: float = 0.85, enable_thinking: bool = False, max_samples: int | None = None) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_tokens + 1024, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    res = {"model": model, "n": n, "temperature": temperature, "max_tokens": max_tokens, "sets": {}}
    for name in sets:
        ds = load_eval(name)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        prompts = [tok.apply_chat_template(e["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) for e in ds]
        outs = llm.generate(prompts, sp)
        correct, lens, boxed, trunc = [], [], [], []
        for e, o in zip(ds, outs):
            cs = [is_correct(c.text, e["answer"]) for c in o.outputs]
            correct.append(sum(cs))
            lens += [len(c.token_ids) for c in o.outputs]
            boxed += [has_box(c.text) for c in o.outputs]
            trunc += [c.finish_reason == "length" for c in o.outputs]
        ks = sorted({1, min(n, 4), n})
        r = {"num_problems": len(ds), "mean_acc": float(np.mean(correct) / n), "mean_len": float(np.mean(lens)),
             "boxed_rate": float(np.mean(boxed)), "truncated_rate": float(np.mean(trunc))}
        for k in ks:
            r[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for c in correct]))
        res["sets"][name] = r
        print(name, json.dumps(r))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", default="math500,gsm8k,aime24")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = evaluate(a.model, a.sets.split(","), a.n, a.max_tokens, a.temperature, enable_thinking=a.thinking, max_samples=a.max_samples)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=2)
