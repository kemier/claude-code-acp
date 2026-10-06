#!/usr/bin/env bash
set -euo pipefail

# POSIX wrapper around the cross-platform Node installer.
# Usage: ./scripts/install.sh
exec node "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.mjs" "$@"
