"""Print (or run with --submit) the sbatch commands for the ablation grid.

  python scripts/make_sweep.py            # dry run
  python scripts/make_sweep.py --submit   # on the cluster login node
Grid = algos x curricula x seeds. Edit below. Keep the first pass small (1 seed) to catch bugs before burning GPUs.
"""
import argparse
import itertools
import subprocess

ALGOS = ["grpo", "dr_grpo", "dapo", "gspo"]
CURRICULA = ["none", "easy2hard", "hard2easy", "filter"]  # hard2easy = anti-curriculum control
SEEDS = [0]

ap = argparse.ArgumentParser()
ap.add_argument("--submit", action="store_true")
ap.add_argument("--base", default="configs/base.yaml")
ap.add_argument("--extra", nargs="*", default=[], help="extra key=value overrides, e.g. model=/path/to/sft")
a = ap.parse_args()

for algo, cur, seed in itertools.product(ALGOS, CURRICULA, SEEDS):
    name = f"{algo}_{cur}_s{seed}"
    cmd = ["sbatch", f"--job-name={name}", "slurm/rl_run.sbatch", a.base, f"algo={algo}", f"curriculum.mode={cur}", f"seed={seed}", f"run_name={name}", *a.extra]
    print(" ".join(cmd))
    if a.submit:
        subprocess.run(cmd, check=True)
