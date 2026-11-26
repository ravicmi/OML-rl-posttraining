"""Plot training curves + final eval for the RL ablation runs.
   python scripts/plot_runs.py <outputs_dir> <out_dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ravi.sah/oml/outputs"
out = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(out, exist_ok=True)

# fixed colour per entity (validated categorical order: blue, orange, aqua, yellow, magenta, violet)
COL = {"grpo_none": "#2a78d6", "drgrpo_none": "#eb6834", "dapo_none": "#1baf7a", "gspo_none": "#d99200",
       "drgrpo_e2h": "#e87ba4", "drgrpo_h2e": "#6250d6", "base_qwen3_0p6b": "#8a8984"}
LAB = {"grpo_none": "GRPO", "drgrpo_none": "Dr.GRPO", "dapo_none": "DAPO", "gspo_none": "GSPO",
       "drgrpo_e2h": "Dr.GRPO easy→hard", "drgrpo_h2e": "Dr.GRPO hard→easy", "base_qwen3_0p6b": "Baseline"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})


def series(run, key):
    xs, ys, off = [], [], 0
    for sd in sorted(glob.glob(f"{root}/runs/{run}/stage*")):
        ts = glob.glob(f"{sd}/checkpoint-*/trainer_state.json")
        if not ts:
            continue
        h = [x for x in json.load(open(ts[0]))["log_history"] if key in x]
        for i, x in enumerate(h):
            xs.append(off + i + 1); ys.append(x[key])
        off += len(h)
    return np.array(xs), np.array(ys)


def smooth(y, w=10):
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def panel(ax, runs, key, title, ylab, pct=False, stages=False):
    for r in runs:
        x, y = series(r, key)
        if len(x) == 0:
            continue
        ax.plot(x, smooth(y), color=COL[r], lw=2, label=LAB[r])
        ax.text(x[-1] + 1, smooth(y)[-1], LAB[r], color=MUTED, fontsize=8, va="center")
    if stages:
        for s in (33, 66):
            ax.axvline(s, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_xlim(0, 125)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


ALG = ["grpo_none", "drgrpo_none", "dapo_none", "gspo_none"]
CUR = ["drgrpo_none", "drgrpo_e2h", "drgrpo_h2e"]
fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
panel(axs[0, 0], ALG, "reward", "A. Training reward, 4 algorithms (10-step average)", "mean reward (fraction correct)")
panel(axs[0, 1], CUR, "reward", "B. Training reward, curricula (stage edges dotted)\nnot comparable: each stage has different difficulty", "mean reward", stages=True)
panel(axs[1, 0], ALG, "frac_reward_zero_std", "C. Groups with NO learning signal\n(all 8 samples right, or all wrong)", "fraction of groups", pct=True)
panel(axs[1, 1], ALG, "completions/clipped_ratio", "D. Samples cut off at the 1024-token training cap", "fraction truncated", pct=True)
fig.suptitle("Qwen3-0.6B on MATH train, 100 RL steps, 1 seed (seed 0): training dynamics", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{out}/training_curves.png", dpi=150)

# ---- eval bars
runs = ["base_qwen3_0p6b"] + ALG + ["drgrpo_e2h", "drgrpo_h2e"]
ev = {r: json.load(open(f"{root}/eval/{r}.json"))["sets"] for r in runs}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (s, n, ttl) in zip(axs, [("math500", 500, "MATH-500 pass@1"), ("gsm8k", 1319, "GSM8K pass@1")]):
    vals = [ev[r][s]["pass@1"] for r in runs]
    se = [np.sqrt(v * (1 - v) / n) for v in vals]
    ax.bar(range(len(runs)), vals, yerr=se, color=[COL[r] for r in runs], width=0.62, ecolor=MUTED, capsize=3)
    ax.axhline(vals[0], color=MUTED, lw=1, ls="--")
    for i, v in enumerate(vals):
        ax.text(i, v + se[i] + 0.006, f"{v:.1%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([LAB[r].replace(" ", "\n", 1) for r in runs], fontsize=8)
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.06); ax.yaxis.set_major_locator(MultipleLocator(0.02)); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("note: y-axis does not start at 0", color=MUTED, fontsize=8)
    ax.set_title(ttl, loc="left", color=INK, fontsize=11, fontweight="bold"); ax.grid(axis="x", visible=False)
fig.suptitle("Final evaluation, 4 samples/problem, 1 seed. Error bars = ±1 binomial SE (approximate; ignores sampling repeats)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{out}/eval_bars.png", dpi=150)
print("saved")
