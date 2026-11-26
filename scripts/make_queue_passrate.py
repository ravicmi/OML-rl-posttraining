"""Queue the pass-rate / curriculum study (Dr.GRPO, 500 steps, 8 arms x 4 seeds = 32 runs). Run ON the cluster:
     python3 scripts/make_queue_passrate.py [n_seeds]
Seed-major, so whichever GPUs free up first always pull the next run and partial progress stays balanced across arms.
Pass rate k/8 comes from the starting policy (outputs/difficulty/math.jsonl). Band sizes (of 7498): hard k=1-2: 1238, mid k=3-5: 1427,
easy k=6-7: 1328, drop-only-0: 5156, drop-only-8: 6335, filter 0<p<1: 3993.
"""
import os
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Q = "/home/ravi.sah/oml/queue/pending"
PR = "/home/ravi.sah/oml/outputs/difficulty/math.jsonl"
COMMON = "algo=dr_grpo max_steps=500 save_steps=100 grpo.save_only_model=true grpo.save_total_limit=10"
P = f"curriculum.difficulty=passrate curriculum.passrate_path={PR}"
ARMS = {
    "pr_rand":  "curriculum.mode=randsub curriculum.subset_size=3993",                      # size-matched random control for the filter
    "pr_drop0": f"curriculum.mode=band curriculum.keep_min=0.125 {P}",                       # drop only never-solved (0/8)
    "pr_drop8": f"curriculum.mode=band curriculum.keep_max=0.875 {P}",                       # drop only always-solved (8/8)
    "pr_hard":  f"curriculum.mode=band curriculum.keep_min=0.125 curriculum.keep_max=0.25 {P}",
    "pr_mid":   f"curriculum.mode=band curriculum.keep_min=0.375 curriculum.keep_max=0.625 {P}",
    "pr_easy":  f"curriculum.mode=band curriculum.keep_min=0.75 curriculum.keep_max=0.875 {P}",
    "pr_fe2h":  f"curriculum.mode=filter_easy2hard curriculum.n_stages=3 {P}",              # filter + easy->hard
    "pr_e2h":   f"curriculum.mode=easy2hard curriculum.n_stages=3 {P}",                     # easy->hard, no filter
}
os.makedirs(Q, exist_ok=True)
k = 100
for seed in range(N):
    for arm, ov in ARMS.items():
        open(f"{Q}/{k:03d}_{arm}_s{seed}", "w").write(f"{arm}_s{seed} train {ov} seed={seed} {COMMON}\n")
        k += 1
print("queued", k - 100, "runs")
