#!/usr/bin/env bash
# Export annotation node groups and materials from a working .blend into
# assets/annotations/, then rebuild the bundled library and its menu index.
#   scripts/export_annotations.sh my_work.blend ["Asset Name" ...]
# Exports the named assets, or every asset in the file. Tag a node group
# "Annotation Scene" or "3D Scene" to list it in the Add yt Annotation menu;
# untagged groups ship as helpers. Commit what changes under assets/ and
# src/yt_volumes/assets/.
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/..")
BLEND=$(realpath "$1"); shift
NO_EXTENSIONS=$(mktemp -d); trap 'rm -rf "$NO_EXTENSIONS"' EXIT
# Isolated from user extensions, which may bundle another nodebpy.
run() { (cd "$ROOT" && PYTHONPATH= BLENDER_USER_EXTENSIONS="$NO_EXTENSIONS" uv run -q --isolated --only-group assets "$@"); }

if (($#)); then NAMES=("$@"); else
  LISTED=$(run python scripts/annotation_assets.py names "$BLEND"); mapfile -t NAMES <<< "$LISTED"
fi
# --names, even for all of them: a dump without it deletes other assets' sources.
run nodebpy dump "$BLEND" --names "${NAMES[@]}"
run nodebpy build
run nodebpy check
run python scripts/annotation_assets.py index
git -C "$ROOT" status --short -- assets src/yt_volumes/assets
