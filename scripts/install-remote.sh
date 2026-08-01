#!/usr/bin/env bash
set -euo pipefail

# Sync claude-code-acp to a remote SSH host, build, and install.
# Usage:
#   ./scripts/install-remote.sh
#   ACP_SSH_HOST=workstation ./scripts/install-remote.sh
#
# Default host: workstation (same as vscode-acp deploy).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SSH_HOST="${ACP_SSH_HOST:-workstation}"
REMOTE_DIR="${ACP_REMOTE_DIR:-projects/claude-code-acp}"

echo "📤 Syncing claude-code-acp → ${SSH_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude node_modules \
  --exclude .git \
  --exclude dist \
  --exclude '*.log' \
  "$REPO_DIR/" "${SSH_HOST}:${REMOTE_DIR}/"

echo "📦 Building + installing on ${SSH_HOST}..."
ssh "$SSH_HOST" "set -e
cd ~/${REMOTE_DIR}
export PATH=\"\$HOME/.npm-global/bin:\$HOME/.local/bin:\$PATH\"
if [[ ! -d node_modules ]]; then
  npm install
fi
bash scripts/install.sh
"

echo "✅ Remote install complete on ${SSH_HOST}"
echo "   Reload the remote VS Code window to pick up claude-agent-acp."
