# OML: post-SFT RL ablations on a 0.6B model (BharatGen H100 cluster)

Goal: compare RL post-training algorithms and curriculum designs on math reasoning, starting from an SFT checkpoint
(default placeholder: `Qwen/Qwen3-0.6B`; set `model:` to your SFT checkpoint).

## Layout
- `cluster/` — local→cluster tooling (SSM access, rsync, remote env setup). `source cluster/env.sh` first.
- `oml/` — `data` (MATH/GSM8K train; MATH-500/GSM8K/AIME24 eval), `rewards` (boxed-answer verifier, pass@k),
  `curriculum` (none / easy2hard / hard2easy / filter), `difficulty` (pass-rate scoring), `eval` (vLLM), `train_rl` (TRL GRPO variants).
- `configs/base.yaml` — one run's settings; `scripts/make_sweep.py` — algo × curriculum × seed grid; `slurm/` — sbatch templates.

## Cluster facts (checked 2026-11-23)
- AWS profile `bharatgen`, account 992382472282, region ap-south-1 only (explicit deny elsewhere).
- H100s: SageMaker HyperPod Slurm clusters of `ml.p5.48xlarge` (8×H100, 96 vCPU, 2 TB RAM). Use **bgen-cluster**
  (login node reached via SSM; your Linux user exists there). `bgen-cluster-c` has no `ravi.sah` account yet.
- Shared home `/home/ravi.sah` = `/fsx/ravi.sah` (OpenZFS, 464 TB free). Lustre `/fsx` is 99% full — don't write there.
- Compute nodes have internet (HF + PyPI reachable). No conda/uv; system Python 3.10; docker + enroot exist.

## Workflow
1. `source cluster/env.sh`; log in: `aws ssm start-session --target $OML_TARGET` (needs a real terminal).
2. One-time: authorise your SSH key for rsync (`cluster/ssh_config_snippet.txt`), then `cluster/sync.sh`.
3. On the cluster: `bash /home/ravi.sah/oml/src/cluster/setup_env.sh`; then `python -m oml.data` to verify datasets.
4. Baseline: `sbatch slurm/eval.sbatch <sft_model> /home/ravi.sah/oml/outputs/eval/sft.json`.
5. Smoke test: `sbatch slurm/rl_run.sbatch configs/base.yaml max_steps=5 run_name=smoke`.
6. Optional pass-rate curriculum: `python -m oml.difficulty --model <sft> --out .../difficulty/math.jsonl`, then `curriculum.difficulty=passrate curriculum.passrate_path=...`.
7. Grid: `python scripts/make_sweep.py --submit`.

## Status
Written but **not yet executed on the cluster** (TRL/vLLM versions unverified; HF dataset ids from memory).
Not implemented (need custom trainers): RLOO, PPO, OPSD, SDPO.
