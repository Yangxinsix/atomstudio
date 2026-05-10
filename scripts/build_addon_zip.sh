#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
ADDON_SRC="$ROOT_DIR/addon/atomstudio_addon"
VERSION="0.1.0"
ZIP_PATH="$DIST_DIR/atomstudio_addon_v${VERSION}.zip"

mkdir -p "$DIST_DIR"
rm -f "$ZIP_PATH"

(
  cd "$ROOT_DIR/addon"
  zip -r "$ZIP_PATH" "atomstudio_addon" >/dev/null
)

echo "Built $ZIP_PATH"
