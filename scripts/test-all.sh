#!/usr/bin/env bash
set -euo pipefail

# scripts/test-all.sh
#
# Run tests with the full feature set (excluding future GPU).
#
# Usage:
#   ./scripts/test-all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALL_FEATURES="bin-scribble-cli,bin-model-downloader,bin-scribble-server"

cd "$REPO_ROOT"

echo "==> Running tests (features: $ALL_FEATURES)"
cargo test --features "$ALL_FEATURES"
