"""Queue re-evaluation of every finished model (and the baseline) on the held-out MATH-rest set (4,500 problems not in MATH-500).
   python3 scripts/make_queue_fulleval.py        (run ON the cluster)
Results: outputs/eval_full/<run>.json. 4 samples/problem, 1536 max tokens, temperature 0.7 (same decoding as the other evals).
"""
import glob
import os

ROOT = "/home/ravi.sah/oml"
Q = f"{ROOT}/queue/pending"
OUT = f"{ROOT}/outputs"
runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{OUT}/runs/*/final_path.txt"))
runs = [r for r in runs if r[:3] in ("pr_",) or "_s" in r and r.rsplit("_s", 1)[1].isdigit()]
EV = "python -m oml.eval --sets math_rest --n 4 --max_tokens 1536"
items = [("fe_base", f"{EV} --model Qwen/Qwen3-0.6B --out {OUT}/eval_full/base.json")]
for r in runs:
    items.append((f"fe_{r}", f"{EV} --model $(cat {OUT}/runs/{r}/final_path.txt) --out {OUT}/eval_full/{r}.json"))
os.makedirs(Q, exist_ok=True)
for k, (name, cmd) in enumerate(items):
    open(f"{Q}/{200 + k:03d}_{name}", "w").write(f"{name} shell {cmd}\n")
print("queued", len(items), "evals;", len(runs), "runs")
