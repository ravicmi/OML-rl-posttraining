#!/usr/bin/env bash
# One-time: append your local public key to ~/.ssh/authorized_keys on the cluster (idempotent). Run in a real terminal.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PUB="$(cat "${1:-$HOME/.ssh/id_ed25519.pub}")"
"$HERE/ssm_run.sh" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && (grep -qF '$PUB' ~/.ssh/authorized_keys || echo '$PUB' >> ~/.ssh/authorized_keys) && chmod 600 ~/.ssh/authorized_keys && echo KEY_AUTHORIZED && wc -l ~/.ssh/authorized_keys"
