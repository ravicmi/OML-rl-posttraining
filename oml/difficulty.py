"""Score every training problem by the starting policy's pass rate (used by curricula / filtering).

  python -m oml.difficulty --model <sft ckpt> --train math --n 8 --out outputs/difficulty/math.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from .data import load_train
from .rewards import is_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="math")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    ds = load_train(a.train, a.max_samples)
    ds = ds.select(range(a.shard, len(ds), a.nshards))  # sharding across GPUs
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=a.max_tokens + 1024, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(e["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in ds]
    outs = llm.generate(prompts, SamplingParams(n=a.n, temperature=a.temperature, max_tokens=a.max_tokens))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for e, o in zip(ds, outs):
            c = sum(is_correct(x.text, e["answer"]) for x in o.outputs)
            f.write(json.dumps({"uid": e["uid"], "level": e["level"], "pass_rate": c / a.n}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
