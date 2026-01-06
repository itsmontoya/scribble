#!/usr/bin/env bash
set -euo pipefail

# scripts/build.sh
#
# Build a single Scribble binary and copy it to an output directory.
#
# Usage:
#   ./scripts/build.sh <binary> [output_dir]
#
# Examples:
#   ./scripts/build.sh scribble-cli
#   ./scripts/build.sh scribble-server ./dist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Build a single Scribble binary and copy it to an output directory.

Usage:
  ./scripts/build.sh <binary> [output_dir]

Examples:
  ./scripts/build.sh scribble-cli
  ./scripts/build.sh scribble-server ./dist
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

BIN="${1:-}"
OUT_DIR="${2:-.}"

case "$BIN" in
  ""|-h|--help)
    usage
    exit 0
    ;;
esac

need_cmd cargo

FEATURES=""
case "$BIN" in
  scribble-cli) FEATURES="bin-scribble-cli" ;;
  scribble-server) FEATURES="bin-scribble-server" ;;
  model-downloader) FEATURES="bin-model-downloader" ;;
  *)
    die "unknown binary: $BIN (expected: scribble-cli, scribble-server, model-downloader)"
    ;;
esac

cd "$REPO_ROOT"

echo "==> Building: $BIN (features: $FEATURES)"
cargo build --release --features "$FEATURES" --bin "$BIN"

SRC="$REPO_ROOT/target/release/$BIN"
[ -x "$SRC" ] || die "build output not found or not executable: $SRC"

mkdir -p "$OUT_DIR"
DEST="$OUT_DIR/$BIN"
cp "$SRC" "$DEST"

echo "==> Wrote: $DEST"
