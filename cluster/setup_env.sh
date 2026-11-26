#!/usr/bin/env bash
# Run ON the cluster (login node): creates a venv on the shared FSx home and installs the stack.
#   bash /home/ravi.sah/oml/src/cluster/setup_env.sh
# Python 3.10 is the system python on the nodes; no conda/uv installed.
set -euo pipefail
ROOT=/home/ravi.sah/oml
mkdir -p "$ROOT"/{venv_parent,outputs,logs,hf_cache}
rm -rf "$ROOT/venv"; python3 -m virtualenv "$ROOT/venv"  # python3-venv is not installed on the nodes
source "$ROOT/venv/bin/activate"
pip install -U pip wheel
# vLLM pins its own torch/CUDA build; install it first so everything else matches.
pip install vllm
pip install -r "$ROOT/src/requirements.txt"
python - <<'EOF'
import torch, trl, vllm, transformers
print("torch", torch.__version__, "| trl", trl.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
EOF
echo "OK. Activate with: source $ROOT/venv/bin/activate"
