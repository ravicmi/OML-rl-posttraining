#!/usr/bin/env bash
# Push src/ to the cluster (requires the `bgen` ssh host from ssh_config_snippet.txt).
#   src/cluster/sync.sh          # push code
#   src/cluster/sync.sh pull runs   # pull results (outputs/runs -> src/outputs/runs)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/home/ravi.sah/oml
if [[ "${1:-push}" == "pull" ]]; then
  mkdir -p "$HERE/outputs/${2:-runs}"
  rsync -avz --exclude='checkpoint-*' --exclude='*.safetensors' "bgen:$REMOTE/outputs/${2:-runs}/" "$HERE/outputs/${2:-runs}/"
else
  ssh bgen "mkdir -p $REMOTE/src $REMOTE/outputs $REMOTE/logs"
  rsync -avz --delete --exclude='outputs/' --exclude='__pycache__/' --exclude='.venv/' "$HERE/" "bgen:$REMOTE/src/"
fi
