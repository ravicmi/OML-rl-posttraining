#!/bin/bash
# Evaluate a model/checkpoint.  sbatch --job-name=eval slurm/eval.sbatch <model_path> <out.json> [extra oml.eval args]
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ravi.sah/oml/logs/%x-%j.out
set -euo pipefail
ROOT=/home/ravi.sah/oml
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache PYTHONPATH=$ROOT/src
cd "$ROOT/src"
MODEL=$1; OUT=$2; shift 2
python -m oml.eval --model "$MODEL" --out "$OUT" "$@"
