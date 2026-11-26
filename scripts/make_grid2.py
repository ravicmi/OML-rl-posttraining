"""Write job files for the 10-seed confirmation round: 4 arms x N seeds, 500 steps, 8 runs per node.
   python scripts/make_grid2.py [n_seeds]   ->  slurm/grid2_node{1..}.jobs
Each node gets 2 seeds x 4 arms so a node failure removes seeds evenly across arms.
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PR = "/home/ravi.sah/oml/outputs/difficulty/math.jsonl"
COMMON = "max_steps=500 save_steps=100 grpo.save_only_model=true grpo.save_total_limit=10"
ARMS = {
    "drgrpo": "algo=dr_grpo curriculum.mode=none",
    "gspo": "algo=gspo curriculum.mode=none",
    "dapo": "algo=dapo curriculum.mode=none",
    "drgrpo_filter": f"algo=dr_grpo curriculum.mode=filter curriculum.difficulty=passrate curriculum.passrate_path={PR}",
}
per_node = 2
for node in range(N // per_node):
    with open(f"slurm/grid2_node{node + 1}.jobs", "w") as f:
        for seed in range(node * per_node, (node + 1) * per_node):
            for arm, ov in ARMS.items():
                f.write(f"{arm}_s{seed} train {ov} seed={seed} {COMMON}\n")
print("wrote", N // per_node, "job files,", N * len(ARMS), "runs")
