#!/usr/bin/env bash
set -euo pipefail

# scripts/build-macos-release.sh
#
# Builds macOS release binaries (arm64 by default), generates SHA256SUMS.txt,
# and uploads them to the matching GitHub Release tag via `gh`.
#
# Usage:
#   ./scripts/build-macos-release.sh
#   ./scripts/build-macos-release.sh --tag v0.0.3
#   ./scripts/build-macos-release.sh --bin scribble-cli --bin model-downloader
#   ./scripts/build-macos-release.sh --target aarch64-apple-darwin
#   ./scripts/build-macos-release.sh --no-upload
#
# Notes:
# - Requires: rustup, cargo, (python OR cargo metadata), gh, shasum
# - Intended for Apple Silicon Macs (aarch64-apple-darwin). Intel target is possible but
#   may fail if native deps don't cross-compile cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TAG=""
TARGET="aarch64-apple-darwin"
OS_LABEL="macOS"
ARCH_LABEL="arm64"
NO_UPLOAD="false"
CLEAN="false"
BINS=()

usage() {
  sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)
      shift
      TAG="${1:-}"
      [ -n "$TAG" ] || die "--tag requires a value"
      ;;
    --target)
      shift
      TARGET="${1:-}"
      [ -n "$TARGET" ] || die "--target requires a value"
      ;;
    --os-label)
      shift
      OS_LABEL="${1:-}"
      [ -n "$OS_LABEL" ] || die "--os-label requires a value"
      ;;
    --arch-label)
      shift
      ARCH_LABEL="${1:-}"
      [ -n "$ARCH_LABEL" ] || die "--arch-label requires a value"
      ;;
    --bin)
      shift
      bin="${1:-}"
      [ -n "$bin" ] || die "--bin requires a value"
      BINS+=("$bin")
      ;;
    --no-upload)
      NO_UPLOAD="true"
      ;;
    --clean)
      CLEAN="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown arg: $1 (use --help)"
      ;;
  esac
  shift || true
done

need_cmd cargo
need_cmd rustup
need_cmd shasum
# gh is only required if uploading
if [ "$NO_UPLOAD" != "true" ]; then
  need_cmd gh
fi

cd "$REPO_ROOT"

if [ "$CLEAN" = "true" ]; then
  echo "==> Cleaning build artifacts"
  cargo clean
fi

echo "==> Ensuring Rust target installed: $TARGET"
rustup target add "$TARGET" >/dev/null

# Determine version from Cargo metadata (prefer python; fallback to rust-only parsing)
VERSION=""
if command -v python >/dev/null 2>&1; then
  VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')"
else
  # Fallback: parse Cargo.toml version (best-effort; assumes first package version line)
  VERSION="$(awk -F\" '/^version = "/ {print $2; exit}' Cargo.toml || true)"
fi
[ -n "$VERSION" ] || die "could not determine version"

if [ -z "$TAG" ]; then
  TAG="v$VERSION"
fi

echo "==> Version: $VERSION"
echo "==> Tag:     $TAG"
echo "==> Target:  $TARGET"

# Map target -> labels (optional convenience)
case "$TARGET" in
  aarch64-apple-darwin)
    ARCH_LABEL="${ARCH_LABEL:-arm64}"
    ;;
  x86_64-apple-darwin)
    ARCH_LABEL="${ARCH_LABEL:-x86_64}"
    ;;
esac

echo "==> Building release binaries"
if [ "${#BINS[@]}" -gt 0 ]; then
  for b in "${BINS[@]}"; do
    echo "    - cargo build --release --all-features --bin $b --target $TARGET"
    cargo build --release --all-features --bin "$b" --target "$TARGET"
  done
else
  echo "    - cargo build --release --all-features --bins --target $TARGET"
  cargo build --release --all-features --bins --target "$TARGET"
fi

DIST_DIR="$REPO_ROOT/dist-macos"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "==> Collecting binaries"
BIN_DIR="$REPO_ROOT/target/$TARGET/release"
[ -d "$BIN_DIR" ] || die "expected build output dir not found: $BIN_DIR"

collect_one() {
  local src="$1"
  local name
  name="$(basename "$src")"

  # Skip non-executables
  [ -x "$src" ] || return 0

  local out="${DIST_DIR}/${name}-v${VERSION}-${OS_LABEL}-${ARCH_LABEL}"
  cp "$src" "$out"
  echo "    + $(basename "$out")"
}

if [ "${#BINS[@]}" -gt 0 ]; then
  for b in "${BINS[@]}"; do
    [ -f "${BIN_DIR}/${b}" ] || die "binary not found: ${BIN_DIR}/${b} (did the build succeed?)"
    collect_one "${BIN_DIR}/${b}"
  done
else
  shopt -s nullglob
  for f in "${BIN_DIR}/"*; do
    [ -f "$f" ] || continue
    # Skip common non-bin artifacts
    case "$(basename "$f")" in
      *.d|*.rlib|*.rmeta|*.a) continue ;;
      build|deps|examples|incremental) continue ;;
    esac
    collect_one "$f"
  done
  shopt -u nullglob
fi

echo "==> Generating SHA256SUMS.txt"
(
  cd "$DIST_DIR"
  shasum -a 256 * > SHA256SUMS.txt
)
echo "    + SHA256SUMS.txt"

echo "==> Dist directory: $DIST_DIR"
ls -lah "$DIST_DIR" | sed '1d' | sed 's/^/    /'

if [ "$NO_UPLOAD" = "true" ]; then
  echo "==> --no-upload set; done."
  exit 0
fi

echo "==> Uploading assets to GitHub Release: $TAG"
# Ensure you’re authenticated: gh auth status
gh release upload "$TAG" "$DIST_DIR"/* --clobber

echo "==> Done. Uploaded macOS assets for $TAG."
