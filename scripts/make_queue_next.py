"""Queue: (4) 16- and 32-sample difficulty scoring (training runs are enqueued automatically when scoring ends),
(1) completion-cap pilot, (2) learning-rate sweep.   python3 scripts/make_queue_next.py     (run ON the cluster)
"""
import os

ROOT = "/home/ravi.sah/oml"
Q = f"{ROOT}/queue/pending"
OUT = f"{ROOT}/outputs"
PR8 = f"{OUT}/difficulty/math.jsonl"
os.makedirs(Q, exist_ok=True)
items = []
# --- (4) scoring: 4 shards each for N=16 and N=32 (starting policy, temperature 1.0, 1024 tokens: same as the 8-sample scoring)
k = 300
for N in (16, 32):
    for i in range(4):
        cmd = (f"python -m oml.difficulty --model Qwen/Qwen3-0.6B --train math --n {N} --max_tokens 1024 --shard {i} --nshards 4 "
               f"--out {OUT}/difficulty/math{N}.shard{i}.jsonl && touch {OUT}/difficulty/n{N}_shard{i}.done && bash {ROOT}/src/scripts/after_difficulty.sh {N}")
        items.append((k, f"diff{N}_shard{i}", "shell", cmd)); k += 1
COMMON = "algo=dr_grpo max_steps=500 save_steps=100 grpo.save_only_model=true grpo.save_total_limit=10"
F = f"curriculum.mode=filter curriculum.difficulty=passrate curriculum.passrate_path={PR8}"
M = f"curriculum.mode=band curriculum.keep_min=0.375 curriculum.keep_max=0.625 curriculum.difficulty=passrate curriculum.passrate_path={PR8}"
ARM = {"drgrpo": "curriculum.mode=none", "filter": F, "mid": M}
# --- (1) completion cap 1536 (training cap; eval cap is unchanged at 1536)
k = 400
for seed in range(4):
    for arm, ov in ARM.items():
        items.append((k, f"cap_{arm}_s{seed}", "train", f"{ov} seed={seed} grpo.max_completion_length=1536 {COMMON}")); k += 1
# --- (2) learning rate sweep: 1e-6, 4e-6, 8e-6 on Dr.GRPO and filter, 2 seeds (lr 2e-6 results exist already)
k = 500
for seed in range(2):
    for lr in ("1e-6", "4e-6", "8e-6"):
        for arm in ("drgrpo", "filter"):
            val = lr.replace("e", ".0e")   # PyYAML reads "1e-6" as a string but "1.0e-6" as a float
            items.append((k, f"lr{lr}_{arm}_s{seed}", "train", f"{ARM[arm]} seed={seed} lr={val} {COMMON}")); k += 1
for k, name, kind, args in items:
    open(f"{Q}/{k:03d}_{name}", "w").write(f"{name} {kind} {args}\n")
print("queued", len(items), "items")
