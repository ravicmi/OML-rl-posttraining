"""Figure for the pass-rate / curriculum study.  python scripts/plot_prstudy.py <outputs_dir> <out_dir>"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
ARMS = [("drgrpo", "No filter\n7,498", "#8a8984"), ("pr_rand", "Random\n3,993", "#8a8984"), ("pr_drop8", "Drop\n8/8\n6,335", "#d99200"),
        ("pr_drop0", "Drop\n0/8\n5,156", "#d99200"), ("drgrpo_filter", "Filter\n0<p<1\n3,993", "#eb6834"), ("pr_hard", "Hard\n1-2/8\n1,238", "#1baf7a"),
        ("pr_mid", "Mid\n3-5/8\n1,427", "#1baf7a"), ("pr_easy", "Easy\n6-7/8\n1,328", "#1baf7a"), ("pr_fe2h", "Filter +\neasy→hard", "#e87ba4"), ("pr_e2h", "Easy→hard\nno filter", "#e87ba4")]
base = json.load(open(f"{root}/eval/base8_qwen3_0p6b.json"))["sets"]["math500"]["pass@1"]


def seeds(a):
    return sorted(int(re.search(r"_s(\d+)\.json$", f).group(1)) for f in glob.glob(f"{root}/eval/{a}_s*.json"))


def diag(run, key, last=50):
    h = []
    for sd in sorted(glob.glob(f"{root}/runs/{run}/stage*")):
        cks = sorted(glob.glob(f"{sd}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
        if cks and os.path.exists(f"{cks[-1]}/trainer_state.json"):
            h += [x for x in json.load(open(f"{cks[-1]}/trainer_state.json"))["log_history"] if key in x]
    return np.mean([x[key] for x in h[-last:]])


fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax = axs[0]
for i, (a, lab, col) in enumerate(ARMS):
    v = [json.load(open(f"{root}/eval/{a}_s{s}.json"))["sets"]["math500"]["pass@1"] for s in seeds(a)]
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=col, s=26, zorder=3, alpha=0.8)
    ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, lw=2, zorder=4)
    ax.text(i, np.mean(v) + 0.004 + (0.0035 if i % 2 else 0), f"{np.mean(v):.1%}", ha="center", fontsize=8, color=INK)
ax.axhline(base, color=MUTED, ls="--", lw=1); ax.text(len(ARMS) - 0.5, base + 0.001, "baseline", color=MUTED, fontsize=8, ha="right")
ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=7.5)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.grid(axis="x", visible=False)
ax.set_title("A. Final MATH-500 pass@1 (dots = seeds, bar = mean)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
for ax, key, ttl in ((axs[1], "frac_reward_zero_std", "B. Groups with no learning signal (last 50 steps)"), (axs[2], "completions/clipped_ratio", "C. Samples truncated at 1024 tokens")):
    m = [np.mean([diag(f"{a}_s{s}", key) for s in seeds(a)]) for a, _, _ in ARMS]
    ax.barh(range(len(ARMS)), m, color=[c for _, _, c in ARMS], height=0.65)
    for i, v in enumerate(m):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([l.replace("\n", " ") for _, l, _ in ARMS], fontsize=7.5); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.grid(axis="y", visible=False); ax.set_title(ttl, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig.suptitle("Pass-rate study: Dr.GRPO, 500 steps, 4 seeds per new arm (10 for the references). Grey = controls, amber = drop-only, orange = filter, green = single bands, pink = ordering", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{out}/prstudy.png", dpi=150); print("saved")
