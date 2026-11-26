#!/usr/bin/env bash
# Run one shell command on the cluster login node over SSM (needs a real TTY: run from a terminal).
#   src/cluster/ssm_run.sh 'squeue -u $USER'
set -euo pipefail
source "$(dirname "$0")/env.sh"
CMD="${1:?usage: ssm_run.sh '<command>'}"
# AWS-StartNonInteractiveCommand does not use a shell, so wrap in bash -lc.
ESC=${CMD//\\/\\\\}; ESC=${ESC//\"/\\\"}
PARAMS=$(python3 -c 'import json,sys; print(json.dumps({"command":[sys.argv[1]]}))' "bash -lc \"${ESC}\"")
exec aws ssm start-session --target "$OML_TARGET" \
  --document-name AWS-StartNonInteractiveCommand --parameters "$PARAMS"
