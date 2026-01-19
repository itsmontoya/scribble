#!/usr/bin/env bash
set -euo pipefail

# scripts/build-all.sh
#
# Build all Scribble binaries with the full feature set (excluding future GPU).
#
# Usage:
#   ./scripts/build-all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALL_FEATURES="bin-scribble-cli,bin-model-downloader,bin-scribble-server"

cd "$REPO_ROOT"

echo "==> Building release binaries (features: $ALL_FEATURES)"
cargo build --release --bins --features "$ALL_FEATURES"
