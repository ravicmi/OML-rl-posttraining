"""Summarise runs: final eval metrics + training reward curve stats.   python scripts/collect.py [outputs_dir]"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/home/ravi.sah/oml/outputs"
rows = []
for f in sorted(glob.glob(f"{root}/eval/*.json")):
    name = os.path.basename(f)[:-5]
    ev = json.load(open(f))["sets"]
    row = {"run": name}
    for s in ("math500", "gsm8k", "aime24"):
        if s in ev:
            row[f"{s}_p@1"] = ev[s]["pass@1"]
            row[f"{s}_p@4"] = ev[s]["pass@4"]
    row["math500_len"] = ev["math500"]["mean_len"]
    row["math500_trunc"] = ev["math500"]["truncated_rate"]
    # training curve from trainer_state.json (all stages)
    rewards, lens = [], []
    for ts in sorted(glob.glob(f"{root}/runs/{name}/stage*/checkpoint-*/trainer_state.json")):
        for h in json.load(open(ts))["log_history"]:
            r = next((h[k] for k in h if k.startswith("reward") and not k.endswith("std")), None)
            if r is not None:
                rewards.append(r)
            if "completions/mean_length" in h:
                lens.append(h["completions/mean_length"])
    if rewards:
        k = max(1, len(rewards) // 5)
        row["train_reward_first"] = sum(rewards[:k]) / k
        row["train_reward_last"] = sum(rewards[-k:]) / k
        row["steps_logged"] = len(rewards)
    rows.append(row)

cols = ["run", "math500_p@1", "math500_p@4", "gsm8k_p@1", "aime24_p@1", "math500_len", "math500_trunc", "train_reward_first", "train_reward_last"]
print(" | ".join(cols))
for r in rows:
    print(" | ".join(r["run"] if c == "run" else (f"{r[c]:.3f}" if c in r else "-") for c in cols))
json.dump(rows, open(f"{root}/summary.json", "w"), indent=2)
