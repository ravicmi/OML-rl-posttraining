"""Collate every finished experiment into one markdown report (tables only; figures come from analyze_round2.py).
   python scripts/collate.py <outputs_dir> > RESULTS.md        (run on the cluster; uses whatever has finished)
"""
import glob
import json
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ravi.sah/oml/outputs"
E = lambda n: json.load(open(f"{root}/eval/{n}.json"))["sets"]
base = E("base8_qwen3_0p6b")
B = ("math500", "gsm8k", "aime24")
pct = lambda v: f"{100 * v:.1f}"


def history(run):
    """Concatenate the training log across stages (each stage is its own trainer)."""
    h = []
    for sd in sorted(glob.glob(f"{root}/runs/{run}/stage*")):
        cks = sorted(glob.glob(f"{sd}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
        if cks and os.path.exists(f"{cks[-1]}/trainer_state.json"):
            h += [x for x in json.load(open(f"{cks[-1]}/trainer_state.json"))["log_history"] if "reward" in x]
    return h


def diag(run, last=50):
    h = history(run)[-last:]
    g = lambda k: np.mean([x[k] for x in h if k in x]) if any(k in x for x in h) else float("nan")
    return g("reward"), g("frac_reward_zero_std"), g("completions/clipped_ratio")


def seeds_of(arm):
    return sorted(int(re.search(rf"^{re.escape(arm)}_s(\d+)\.json$", os.path.basename(f)).group(1)) for f in glob.glob(f"{root}/eval/{arm}_s*.json"))


def arm_table(title, arms, labels):
    print(f"\n### {title}\n")
    print("| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Train reward (last 50) | Zero-signal groups | Truncated |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| Baseline (no RL, n=8) | - | {pct(base['math500']['pass@1'])} | {pct(base['math500']['pass@8'])} | {pct(base['gsm8k']['pass@1'])} | {pct(base['aime24']['pass@1'])} | - | - | - |")
    for a in arms:
        ss = seeds_of(a)
        if not ss:
            continue
        ev = [E(f"{a}_s{s}") for s in ss]
        cell = lambda b, k="pass@1": (f"{pct(np.mean([e[b][k] for e in ev]))} ± {pct(np.std([e[b][k] for e in ev], ddof=1))}" if len(ev) > 1 else pct(ev[0][b][k]))
        d = np.array([diag(f"{a}_s{s}") for s in ss])
        m = np.nanmean(d, axis=0)
        print(f"| {labels[a]} | {len(ss)} | {cell('math500')} | {cell('math500', 'pass@8')} | {cell('gsm8k')} | {cell('aime24')} | {m[0]:.2f} | {100 * m[1]:.0f}% | {100 * m[2]:.0f}% |")


def paired(arms, ref, b="math500"):
    print(f"\nPaired same-seed difference vs `{ref}` ({b} pass@1, percentage points):\n")
    print("| Arm | n seeds | Mean diff | s.e. | t | Seeds won |")
    print("|---|---|---|---|---|---|")
    r = {s: E(f"{ref}_s{s}")[b]["pass@1"] for s in seeds_of(ref)}
    for a in arms:
        x = {s: E(f"{a}_s{s}")[b]["pass@1"] for s in seeds_of(a)}
        c = sorted(set(x) & set(r))
        if len(c) < 2:
            continue
        d = np.array([x[s] - r[s] for s in c]) * 100
        se = d.std(ddof=1) / len(d) ** 0.5
        print(f"| {a} | {len(c)} | {d.mean():+.2f} | {se:.2f} | {d.mean() / se:.1f} | {int((d > 0).sum())}/{len(d)} |")


def per_seed(arms):
    print("\n#### Per-seed MATH-500 pass@1 (%)\n")
    ss = sorted({s for a in arms for s in seeds_of(a)})
    print("| Arm | " + " | ".join(f"s{s}" for s in ss) + " |")
    print("|---|" + "---|" * len(ss))
    for a in arms:
        v = {s: E(f"{a}_s{s}")["math500"]["pass@1"] for s in seeds_of(a)}
        print(f"| {a} | " + " | ".join(pct(v[s]) if s in v else "-" for s in ss) + " |")


def curve(arms):
    print("\n#### MATH-500 pass@1 (%) at checkpoints (mean over seeds; n=4 samples at checkpoints, n=8 at step 500)\n")
    print("| Arm | 100 | 200 | 300 | 400 | 500 |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        row = []
        for st in (100, 200, 300, 400):
            v = [json.load(open(f))["sets"]["math500"]["pass@1"] for f in glob.glob(f"{root}/eval_ckpt/{a}_s*_checkpoint-{st}.json")]
            row.append(pct(np.mean(v)) if v else "-")
        v = [E(f"{a}_s{s}")["math500"]["pass@1"] for s in seeds_of(a)]
        row.append(pct(np.mean(v)) if v else "-")
        print(f"| {a} | " + " | ".join(row) + " |")


print("# Qwen3-0.6B RL post-training ablations: collated results\n")
print("Setup: Qwen3-0.6B, MATH train (7,498 problems), verifiable boxed-answer reward, 16 prompts x 8 samples per step, lr 2e-6, 1024-token completion cap,"
      " TRL GRPOTrainer + vLLM. Eval: temperature 0.7, 1536 tokens, 8 samples/problem. pass@1 in %, mean ± std across seeds.\n")

# ---- 1. 100-step screen
print("## 1. 100-step screening grid (1 seed, 4 samples/problem)\n")
sj = json.load(open(f"{root}/summary.json"))
print("| Run | MATH-500 p@1 | GSM8K p@1 | AIME24 p@1 | Truncated (MATH-500) | Train reward first→last |")
print("|---|---|---|---|---|---|")
for r in sj:
    if r["run"] == "smoke":
        continue
    tr = f"{r['train_reward_first']:.2f}→{r['train_reward_last']:.2f}" if "train_reward_first" in r else "-"
    print(f"| {r['run']} | {pct(r['math500_p@1'])} | {pct(r['gsm8k_p@1'])} | {pct(r['aime24_p@1'])} | {pct(r['math500_trunc'])}% | {tr} |")

# ---- 2. 500-step round
ARMS2 = ["drgrpo", "gspo", "dapo", "drgrpo_filter"]
L2 = {"drgrpo": "Dr.GRPO", "gspo": "GSPO", "dapo": "DAPO", "drgrpo_filter": "Dr.GRPO + pass-rate filter (0<p<1)"}
print("\n## 2. 500-step comparison: algorithm and filter (4 arms x up to 10 seeds)")
arm_table("Final results at step 500", ARMS2, L2)
paired(["gspo", "dapo", "drgrpo_filter"], "drgrpo")
paired(["gspo", "dapo"], "drgrpo_filter")
paired(["gspo", "dapo", "drgrpo_filter"], "drgrpo", "gsm8k")
per_seed(ARMS2)
curve(ARMS2)

# ---- 3. pass-rate study (whatever has finished)
PR = ["drgrpo", "drgrpo_filter", "pr_rand", "pr_drop0", "pr_drop8", "pr_hard", "pr_mid", "pr_easy", "pr_fe2h", "pr_e2h"]
L3 = {"drgrpo": "no filter (7,498)", "drgrpo_filter": "filter 0<p<1 (3,993)", "pr_rand": "random subset (3,993)", "pr_drop0": "drop never-solved (5,156)",
      "pr_drop8": "drop always-solved (6,335)", "pr_hard": "hard band p=1-2/8 (1,238)", "pr_mid": "mid band p=3-5/8 (1,427)", "pr_easy": "easy band p=6-7/8 (1,328)",
      "pr_fe2h": "filter + easy->hard", "pr_e2h": "easy->hard, no filter"}
done3 = [a for a in PR if a.startswith("pr_") and seeds_of(a)]
print("\n## 3. Pass-rate / curriculum study (Dr.GRPO, 500 steps)")
if done3:
    arm_table("Final results at step 500 (only arms with finished runs are listed)", [a for a in PR if a in done3 or a in ("drgrpo", "drgrpo_filter")], L3)
    paired([a for a in done3], "drgrpo_filter")
    paired([a for a in done3], "drgrpo")
    # multi-stage arms (pr_fe2h, pr_e2h) are left out of the curve: their per-stage checkpoint names collide in eval_ckpt/
    curve([a for a in PR if (a in done3 or a in ("drgrpo", "drgrpo_filter")) and a not in ("pr_fe2h", "pr_e2h")])
else:
    print("\nNo run of this study has finished yet.")

# ---- 4. status
print("\n## 4. Not yet run\n\nRLOO, PPO, OPSD, SDPO (need custom trainers); learning-rate and completion-cap pilot; runs longer than 500 steps.")
