"""Analyse the multi-seed round: results table + plots from whatever runs have finished.
   python scripts/analyze_round2.py <outputs_dir> <out_dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ravi.sah/oml/outputs"
out = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(out, exist_ok=True)
ARMS = ["drgrpo", "gspo", "dapo", "drgrpo_filter"]
LAB = {"drgrpo": "Dr.GRPO", "gspo": "GSPO", "dapo": "DAPO", "drgrpo_filter": "Dr.GRPO + pass-rate filter"}
COL = {"drgrpo": "#eb6834", "gspo": "#d99200", "dapo": "#1baf7a", "drgrpo_filter": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})

base = json.load(open(f"{root}/eval/base8_qwen3_0p6b.json"))["sets"]


def seeds(arm):
    return sorted(int(f.split("_s")[-1][:-5]) for f in glob.glob(f"{root}/eval/{arm}_s*.json"))


def final(arm, s, bench, key="pass@1"):
    return json.load(open(f"{root}/eval/{arm}_s{s}.json"))["sets"][bench][key]


def history(arm, s):
    f = f"{root}/runs/{arm}_s{s}/stage0/checkpoint-500/trainer_state.json"
    return [h for h in json.load(open(f))["log_history"] if "reward" in h] if os.path.exists(f) else []


def smooth(y, w=20):
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


# ---------- table
rows = []
print("arm | seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1")
print(f"baseline (n=8) | - | {base['math500']['pass@1']:.1%} | {base['math500']['pass@8']:.1%} | {base['gsm8k']['pass@1']:.1%} | {base['aime24']['pass@1']:.1%}")
for a in ARMS:
    ss = seeds(a)
    if not ss:
        continue
    cell = lambda b, k="pass@1": (lambda v: f"{np.mean(v):.1%} ±{np.std(v, ddof=1):.1%}" if len(v) > 1 else f"{v[0]:.1%}")([final(a, s, b, k) for s in ss])
    print(f"{LAB[a]} | {len(ss)} ({ss}) | {cell('math500')} | {cell('math500', 'pass@8')} | {cell('gsm8k')} | {cell('aime24')}")
    rows.append({"arm": a, "seeds": ss, **{f"{b}_p1": [final(a, s, b) for s in ss] for b in ("math500", "gsm8k", "aime24")},
                 "math500_p8": [final(a, s, "math500", "pass@8") for s in ss]})
json.dump({"baseline": base, "arms": rows}, open(f"{root}/summary_round2.json", "w"), indent=2)

# ---------- training curves
fig, axs = plt.subplots(2, 2, figsize=(13, 8.4))
for ax, (key, ttl, pct) in zip(axs.flat, [("reward", "A. Training reward (20-step avg; thick = mean of seeds)", False),
                                           ("frac_reward_zero_std", "B. Groups with NO learning signal", True),
                                           ("completions/clipped_ratio", "C. Samples truncated at the 1024-token cap", True),
                                           (None, "D. MATH-500 pass@1 at checkpoints (dots = seeds, line = mean)", True)]):
    for a in ARMS:
        ss = seeds(a)
        if key:
            ys = [np.array([h[key] for h in history(a, s) if key in h]) for s in ss]
            ys = [y for y in ys if len(y) > 100]
            if not ys:
                continue
            n = min(len(y) for y in ys); sm = np.array([smooth(y[:n]) for y in ys])
            for r in sm:
                ax.plot(range(1, n + 1), r, color=COL[a], lw=0.8, alpha=0.35)
            ax.plot(range(1, n + 1), sm.mean(0), color=COL[a], lw=2.4, label=f"{LAB[a]} (n={len(ys)})")
        else:
            pts = {0: [base["math500"]["pass@1"]]}
            for s in ss:
                for st in (100, 200, 300, 400):
                    f = f"{root}/eval_ckpt/{a}_s{s}_checkpoint-{st}.json"
                    if os.path.exists(f):
                        pts.setdefault(st, []).append(json.load(open(f))["sets"]["math500"]["pass@1"])
                pts.setdefault(500, []).append(final(a, s, "math500"))
            xs = sorted(pts)
            for x in xs[1:]:
                ax.scatter([x] * len(pts[x]), pts[x], color=COL[a], s=14, alpha=0.5)
            ax.plot(xs, [np.mean(pts[x]) for x in xs], color=COL[a], lw=2.4, marker="o", ms=5, label=LAB[a])
    if key is None:
        ax.axhline(base["math500"]["pass@1"], color=MUTED, ls="--", lw=1)
        ax.set_xlabel("training step (step 0 = baseline)")
    else:
        ax.set_xlabel("training step")
    ax.set_title(ttl, loc="left", color=INK, fontsize=10.5, fontweight="bold")
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("Qwen3-0.6B on MATH train, 500 RL steps: completed seeds only (in progress)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{out}/round2_curves.png", dpi=150)

# ---------- final eval strip plot
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, (b, ttl) in zip(axs, [("math500", "MATH-500 pass@1"), ("gsm8k", "GSM8K pass@1"), ("aime24", "AIME24 pass@1 (30 problems)")]):
    for i, a in enumerate(ARMS):
        ss = seeds(a)
        if not ss:
            continue
        v = [final(a, s, b) for s in ss]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[a], s=34, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(v)] * 2, color=INK, lw=2, zorder=4)
        ax.text(i, np.mean(v) + 0.004 if b != "aime24" else np.mean(v) + 0.006, f"{np.mean(v):.1%}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(base[b]["pass@1"], color=MUTED, ls="--", lw=1)
    ax.text(3.45, base[b]["pass@1"], "baseline", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([LAB[a].replace(" + ", "\n+ ").replace("Dr.GRPO", "Dr.GRPO", 1) for a in ARMS], fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}" if b == "aime24" else f"{v:.0%}"); ax.set_title(ttl, loc="left", color=INK, fontsize=11, fontweight="bold", pad=14); ax.grid(axis="x", visible=False)
fig.suptitle("Final eval at step 500, 8 samples/problem. Dots = individual seeds, bar = mean; dashed = baseline", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{out}/round2_final.png", dpi=150)
print("saved")
