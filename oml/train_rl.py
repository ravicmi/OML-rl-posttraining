"""Config-driven RL post-training (TRL GRPOTrainer + vLLM colocate) for algorithm / curriculum ablations.

  python -m oml.train_rl --config configs/base.yaml algo=dr_grpo curriculum.mode=easy2hard seed=1 run_name=drgrpo_e2h_s1

Overrides are dotted key=value pairs applied on top of the YAML. Algorithm presets (ALGOS) set the
TRL GRPOConfig knobs that differ between variants; anything under `grpo:` in the YAML wins over presets.

NOT YET RUN: TRL's GRPOConfig fields drift between releases, so unknown keys are dropped with a warning.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os

import yaml

from .curriculum import build_stages
from .data import load_train
from .rewards import correctness_reward, format_reward

# What differs between algorithms, expressed as GRPOConfig kwargs.
ALGOS = {
    # Original GRPO: per-sequence length norm, std-scaled advantages, KL to reference.
    "grpo": dict(loss_type="grpo", scale_rewards=True, beta=0.04),
    # Dr.GRPO: no length normalisation, no std scaling, no KL.
    "dr_grpo": dict(loss_type="dr_grpo", scale_rewards=False, beta=0.0),
    # DAPO: token-level loss, clip-higher, mask truncated samples, no KL.
    "dapo": dict(loss_type="dapo", scale_rewards=True, beta=0.0, epsilon_high=0.28, mask_truncated_completions=True),
    # GSPO: sequence-level importance ratios.
    "gspo": dict(loss_type="grpo", scale_rewards=True, beta=0.0, importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4),
    # TODO (needs custom trainers, see README): rloo, ppo, opsd, sdpo
}


def set_dotted(d: dict, key: str, val):
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val


def load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        k, v = o.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)

    from datasets import Dataset  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    algo = cfg["algo"]
    if algo not in ALGOS:
        raise SystemExit(f"algo {algo!r} not implemented; have {list(ALGOS)}")
    out_root = os.path.join(cfg["output_dir"], cfg["run_name"])
    os.makedirs(out_root, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_root, "config.json"), "w"), indent=2)

    train = load_train(cfg["data"]["train"], cfg["data"].get("max_samples"))
    cur = cfg["curriculum"]
    stages = build_stages(train, cur["mode"], cur.get("n_stages", 3), cur.get("difficulty", "passrate"), cur.get("passrate_path"), cfg["seed"],
                          keep_min=cur.get("keep_min"), keep_max=cur.get("keep_max"), subset_size=cur.get("subset_size"))
    steps_per_stage = max(1, cfg["max_steps"] // len(stages))

    fw = cfg.get("format_weight", 0.0)
    reward_funcs, reward_weights = [correctness_reward], [1.0]
    if fw > 0:
        reward_funcs.append(format_reward)
        reward_weights.append(fw)

    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    model = cfg["model"]
    for si, ds in enumerate(stages):
        kw = dict(
            output_dir=os.path.join(out_root, f"stage{si}"),
            seed=cfg["seed"],
            max_steps=steps_per_stage,
            learning_rate=cfg["lr"],
            bf16=True,
            gradient_checkpointing=True,
            save_strategy="steps",
            save_steps=cfg.get("save_steps", steps_per_stage),
            save_total_limit=1,
            logging_steps=1,
            report_to=cfg.get("report_to", "none"),
            reward_weights=reward_weights,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=cfg.get("vllm_gpu_util", 0.3),
            chat_template_kwargs={"enable_thinking": False},
        )
        kw.update(ALGOS[algo])
        kw.update(cfg.get("grpo", {}))
        dropped = [k for k in kw if k not in valid]
        if dropped:
            print(f"[warn] GRPOConfig has no fields {dropped}; dropping (check TRL version)")
        args = GRPOConfig(**{k: v for k, v in kw.items() if k in valid})
        print(f"=== stage {si + 1}/{len(stages)}: {len(ds)} problems, {steps_per_stage} steps, algo={algo}, curriculum={cur['mode']} ===")
        trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=args, train_dataset=ds)
        trainer.train()
        stage_dir = os.path.join(out_root, f"stage{si}", "final")
        trainer.save_model(stage_dir)
        model = stage_dir  # next stage continues from this policy (optimizer state is reset per stage)
        del trainer
    open(os.path.join(out_root, "final_path.txt"), "w").write(model)
    print("final model:", model)


if __name__ == "__main__":
    main()
