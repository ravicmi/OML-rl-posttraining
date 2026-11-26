"""Fill the shared work queue ($ROOT/queue/pending) for the 10-seed round. Run ON the cluster (login node):
     python3 scripts/make_queue.py [n_seeds]
Order: difficulty shards + n=8 baseline first, then runs seed-major (so partial progress stays balanced across arms).
"""
import os
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
Q = "/home/ravi.sah/oml/queue/pending"
OUT = "/home/ravi.sah/oml/outputs"
PR = f"{OUT}/difficulty/math.jsonl"
os.makedirs(Q, exist_ok=True)
COMMON = "max_steps=500 save_steps=100 grpo.save_only_model=true grpo.save_total_limit=10"
ARMS = {
    "drgrpo": "algo=dr_grpo curriculum.mode=none",
    "gspo": "algo=gspo curriculum.mode=none",
    "dapo": "algo=dapo curriculum.mode=none",
    "drgrpo_filter": f"algo=dr_grpo curriculum.mode=filter curriculum.difficulty=passrate curriculum.passrate_path={PR}",
}
items = []
for i in range(4):
    cmd = (f"python -m oml.difficulty --model Qwen/Qwen3-0.6B --train math --n 8 --max_tokens 1024 --shard {i} --nshards 4 "
           f"--out {OUT}/difficulty/math.shard{i}.jsonl && touch {OUT}/difficulty/shard{i}.done")
    items.append((f"diff_shard{i}", "shell", cmd))
items.append(("base8_qwen3_0p6b", "eval", "Qwen/Qwen3-0.6B"))
for seed in range(N):
    for arm, ov in ARMS.items():
        items.append((f"{arm}_s{seed}", "train", f"{ov} seed={seed} {COMMON}"))
for k, (name, kind, args) in enumerate(items):
    open(f"{Q}/{k:03d}_{name}", "w").write(f"{name} {kind} {args}\n")
print("queued", len(items), "items in", Q)
