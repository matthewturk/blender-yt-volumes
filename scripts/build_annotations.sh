#!/usr/bin/env bash
# Build the bundled annotation library (src/yt_volumes/assets/annotations.blend)
# and its menu index from the nodebpy sources in assets/annotations/.
#   scripts/build_annotations.sh [--force]
# Rebuilds only when the sources changed since the last build, unless --force.
# The built files aren't in git: run this after checkout, before using the add-on
# from src/ (the tests run it themselves).
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/..")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# Isolated from user extensions, which may bundle another nodebpy.
run() { (cd "$ROOT" && PYTHONPATH= BLENDER_USER_EXTENSIONS="$TMP/no-extensions" uv run -q --isolated --only-group assets "$@"); }
mkdir -p "$ROOT/src/yt_volumes/assets"
if [[ ${1:-} == --force ]]; then run nodebpy build; else run nodebpy ensure; fi
run python scripts/annotation_assets.py index
