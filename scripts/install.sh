#!/usr/bin/env bash
set -euo pipefail

# Build claude-code-acp and install to ~/.local/bin via symlink.
# Usage: ./scripts/install.sh
#
# Installs both names used by vscode-acp settings:
#   - claude-code-acp  (package / bin name)
#   - claude-agent-acp (vscode-acp "Claude Code" agent command)
#
# Symlinks point at bin/claude-code-acp.js which imports ../dist/index.js
# relative to the repo, so a bare copy of the bin would break.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"

echo "📦 Building claude-code-acp from $REPO_DIR..."
cd "$REPO_DIR"

if [[ ! -d node_modules ]]; then
  echo "   Installing npm dependencies..."
  npm install
fi

npm run build

BUILT_JS="$REPO_DIR/dist/index.js"
BIN_WRAPPER="$REPO_DIR/bin/claude-code-acp.js"
if [[ ! -f "$BUILT_JS" ]]; then
  echo "❌ Build output not found: $BUILT_JS"
  exit 1
fi
if [[ ! -f "$BIN_WRAPPER" ]]; then
  echo "❌ Bin wrapper not found: $BIN_WRAPPER"
  exit 1
fi
chmod +x "$BIN_WRAPPER"

VERSION=$(node -p "require('$REPO_DIR/package.json').version")
echo "   Built version: $VERSION"
echo "   Output: $BUILT_JS"

mkdir -p "$LOCAL_BIN"
ln -sf "$BIN_WRAPPER" "$LOCAL_BIN/claude-code-acp"
ln -sf "$BIN_WRAPPER" "$LOCAL_BIN/claude-agent-acp"

# Prefer local build over Homebrew/npm-global if those dirs exist
BREW_GLOBAL="/opt/homebrew/lib/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
if [[ -e "$BREW_GLOBAL" || -L "$BREW_GLOBAL" ]]; then
  ln -sf "$BUILT_JS" "$BREW_GLOBAL"
  echo "   Also symlinked brew package: $BREW_GLOBAL"
fi

NPM_GLOBAL="$HOME/.npm-global/lib/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js"
if [[ -e "$NPM_GLOBAL" || -L "$NPM_GLOBAL" ]]; then
  ln -sf "$BUILT_JS" "$NPM_GLOBAL"
  echo "   Also symlinked npm-global: $NPM_GLOBAL"
fi

echo "✅ Installed claude-code-acp v$VERSION"
echo "   $LOCAL_BIN/claude-code-acp -> $BIN_WRAPPER"
echo "   $LOCAL_BIN/claude-agent-acp -> $BIN_WRAPPER"
echo ""
echo "   Verify: $(command -v claude-agent-acp 2>/dev/null || echo 'claude-agent-acp not on PATH')"
echo ""
echo "✅ Done! Reload VS Code to use the new claude-agent-acp."
