"""Enqueue the runs that use N-sample pass rates (N = 16 or 32): filter (0<p<1) and mid band (3/8..5/8), Dr.GRPO, 500 steps, 4 seeds.
   python3 scripts/make_queue_pr_n.py N     (called by after_difficulty.sh once scoring has finished)
"""
import json
import sys

N = int(sys.argv[1])
ROOT = "/home/ravi.sah/oml"
PR = f"{ROOT}/outputs/difficulty/math{N}.jsonl"
Q = f"{ROOT}/queue/pending"
COMMON = "algo=dr_grpo max_steps=500 save_steps=100 grpo.save_only_model=true grpo.save_total_limit=10"
P = f"curriculum.difficulty=passrate curriculum.passrate_path={PR}"
ARMS = {f"pr{N}_filter": f"curriculum.mode=filter {P}",
        f"pr{N}_mid": f"curriculum.mode=band curriculum.keep_min=0.375 curriculum.keep_max=0.625 {P}"}
p = [json.loads(l)["pass_rate"] for l in open(PR)]
print(f"N={N}: {len(p)} problems; filter keeps {sum(0 < x < 1 for x in p)}; mid band keeps {sum(0.375 - 1e-9 <= x <= 0.625 + 1e-9 for x in p)}")
base = 310 if N == 16 else 330
k = base
for seed in range(4):
    for arm, ov in ARMS.items():
        open(f"{Q}/{k:03d}_{arm}_s{seed}", "w").write(f"{arm}_s{seed} train {ov} seed={seed} {COMMON}\n")
        k += 1
print("queued", k - base, "runs")
