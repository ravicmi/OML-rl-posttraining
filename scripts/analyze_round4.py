"""Results of the follow-up experiments: 16/32-sample pass rates, 1536-token cap, learning-rate sweep.
   python3 scripts/analyze_round4.py [outputs_dir]   (run ON the cluster; needs numpy)  -> markdown on stdout
"""
import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ravi.sah/oml/outputs"


def seeds(arm):
    return sorted(int(os.path.basename(f).rsplit("_s", 1)[1][:-5]) for f in glob.glob(f"{root}/eval/{arm}_s*.json"))


def get(arm, s, bench="math500", key="pass@1"):
    return 100 * json.load(open(f"{root}/eval/{arm}_s{s}.json"))["sets"][bench][key]


def train_stat(arm, s, key, last=50):
    f = f"{root}/runs/{arm}_s{s}/stage0/checkpoint-500/trainer_state.json"
    if not os.path.exists(f):
        return None
    v = [h[key] for h in json.load(open(f))["log_history"] if key in h]
    return float(np.mean(v[-last:])) if v else None


def row(arm, label=None, ref=None):
    ss = seeds(arm)
    if not ss:
        return
    m = lambda b, k="pass@1": [get(arm, s, b, k) for s in ss]
    f = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
    z = [train_stat(arm, s, "frac_reward_zero_std") for s in ss]
    z = [x for x in z if x is not None]
    tr = [100 * get(arm, s, "math500", "truncated_rate") / 100 for s in ss]
    print(f"| {label or arm} | {len(ss)} | {f(m('math500'))} | {f(m('math500', 'pass@8'))} | {f(m('gsm8k'))} | {f(m('aime24'))} | "
          f"{np.mean(tr):.0f}% | {('%.0f%%' % (100 * np.mean(z))) if z else '-'} |")


def paired(a, b, label=None):
    sa, sb = seeds(a), seeds(b)
    ss = sorted(set(sa) & set(sb))
    if len(ss) < 2:
        return
    d = np.array([get(a, s) - get(b, s) for s in ss])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"| {label or a + ' - ' + b} | {len(ss)} | {d.mean():+.2f} | {se:.2f} | {d.mean() / se:+.1f} | {int((d > 0).sum())}/{len(d)} |")


HDR = "| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Truncated | Zero-signal |\n|---|---|---|---|---|---|---|---|"
PH = "| Comparison | n seeds | Mean diff (pt) | s.e. | t | Seeds won |\n|---|---|---|---|---|---|"

print("## Reference arms (earlier rounds)\n\n" + HDR)
for a in ("drgrpo", "drgrpo_filter", "pr_mid"):
    row(a)
print("\n## A. 16- and 32-sample pass rates (Dr.GRPO, 500 steps)\n\n" + HDR)
for a in ("drgrpo_filter", "pr16_filter", "pr32_filter", "pr_mid", "pr16_mid", "pr32_mid"):
    row(a, {"drgrpo_filter": "filter, 8-sample labels", "pr16_filter": "filter, 16-sample labels", "pr32_filter": "filter, 32-sample labels",
            "pr_mid": "mid band, 8-sample labels", "pr16_mid": "mid band, 16-sample labels", "pr32_mid": "mid band, 32-sample labels"}[a])
print("\n" + PH)
for a, b in (("pr16_filter", "drgrpo_filter"), ("pr32_filter", "drgrpo_filter"), ("pr16_mid", "pr_mid"), ("pr32_mid", "pr_mid"),
             ("pr16_filter", "drgrpo"), ("pr32_filter", "drgrpo"), ("pr16_mid", "drgrpo"), ("pr32_mid", "drgrpo"), ("pr32_filter", "pr16_filter"), ("pr32_mid", "pr16_mid")):
    paired(a, b)
print("\n## B. Completion cap 1536 (train) vs 1024\n\n" + HDR)
for a in ("drgrpo", "cap_drgrpo", "drgrpo_filter", "cap_filter", "pr_mid", "cap_mid"):
    row(a)
print("\n" + PH)
for a, b in (("cap_drgrpo", "drgrpo"), ("cap_filter", "drgrpo_filter"), ("cap_mid", "pr_mid"), ("cap_filter", "cap_drgrpo"), ("cap_mid", "cap_drgrpo")):
    paired(a, b)
print("\n## C. Learning-rate sweep (2e-6 = default; same seeds 0,1 for the paired rows)\n\n" + HDR)
for arm in ("drgrpo", "filter"):
    for lr in ("1e-6", "2e-6", "4e-6", "8e-6"):
        a = ("drgrpo" if arm == "drgrpo" else "drgrpo_filter") if lr == "2e-6" else f"lr{lr}_{arm}"
        row(a, f"{arm} lr={lr}")
print("\n" + PH)
for arm in ("drgrpo", "filter"):
    ref = "drgrpo" if arm == "drgrpo" else "drgrpo_filter"
    for lr in ("1e-6", "4e-6", "8e-6"):
        paired(f"lr{lr}_{arm}", ref, f"{arm}: lr {lr} - 2e-6")
for lr in ("1e-6", "4e-6", "8e-6"):
    paired(f"lr{lr}_filter", f"lr{lr}_drgrpo", f"filter - drgrpo at lr {lr}")
print("\n## Per-seed MATH-500 pass@1\n\n| Arm | seeds: values |\n|---|---|")
for a in ("pr16_filter", "pr32_filter", "pr16_mid", "pr32_mid", "cap_drgrpo", "cap_filter", "cap_mid") + tuple(f"lr{l}_{x}" for l in ("1e-6", "4e-6", "8e-6") for x in ("drgrpo", "filter")):
    ss = seeds(a)
    if ss:
        print(f"| {a} | " + ", ".join(f"s{s}={get(a, s):.1f}" for s in ss) + " |")
