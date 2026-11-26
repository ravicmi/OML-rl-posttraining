#!/bin/bash
# One RL ablation run on 1 GPU (a 0.6B model needs little; pack up to 8 runs per p5 node).
# Submit:  sbatch --job-name=NAME slurm/rl_run.sbatch configs/base.yaml algo=dr_grpo run_name=drgrpo_s0 seed=0
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ravi.sah/oml/logs/%x-%j.out
set -euo pipefail
ROOT=/home/ravi.sah/oml
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache
export PYTHONPATH=$ROOT/src
export TOKENIZERS_PARALLELISM=false
cd "$ROOT/src"
python -m oml.train_rl --config "$@"
