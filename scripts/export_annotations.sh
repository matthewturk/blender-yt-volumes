#!/usr/bin/env bash
# Export annotation node groups and materials from a working .blend into
# assets/annotations/, then rebuild the bundled library and its menu index.
#   scripts/export_annotations.sh [--dry-run] my_work.blend ["Asset Name" ...]
# Exports the named assets, or every asset in the file. Tag a node group
# "Annotation Scene" or "3D Scene" to list it in the Add yt Annotation menu;
# untagged ones ship as helpers for other groups. --dry-run shows what would
# change in assets/annotations/ and writes nothing. Commit what changes under
# assets/ and src/yt_volumes/assets/.
set -euo pipefail
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if (($# < 1)); then sed -n '2,9s/^# \{0,1\}//p' "$0" >&2; exit 2; fi
ROOT=$(realpath "$(dirname "$0")/..")
BLEND=$(realpath "$1"); shift
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# Isolated from user extensions, which may bundle another nodebpy.
run() { (cd "$ROOT" && PYTHONPATH= BLENDER_USER_EXTENSIONS="$TMP/no-extensions" uv run -q --isolated --only-group assets "$@"); }

# Always run, for its warnings, even when names are given.
LISTED=$(run python scripts/annotation_assets.py names "$BLEND")
mapfile -t FOUND <<< "$LISTED"
if (($#)); then
  NAMES=()
  for name in "$@"; do
    if printf '%s\n' "${FOUND[@]}" | grep -qxF -- "$name"; then NAMES+=("$name")
    else echo "warning: no asset named '$name' in $(basename "$BLEND"); skipping it" >&2; fi
  done
  ((${#NAMES[@]})) || { echo "Nothing to export." >&2; exit 1; }
else NAMES=("${FOUND[@]}"); fi
echo "Exporting: $(printf "'%s', " "${NAMES[@]}" | sed 's/, $//')" >&2

if ((DRY_RUN)); then
  cp -r "$ROOT/assets/annotations" "$TMP/annotations"
  run nodebpy dump "$BLEND" "$TMP/annotations" --names "${NAMES[@]}" > /dev/null
  echo "Dry run: the export would change assets/annotations/ like this (nothing written):"
  if diff -ruN -x __pycache__ "$ROOT/assets/annotations" "$TMP/annotations" > "$TMP/changes.diff"; then
    echo "  no changes"
  else
    grep '^+++ ' "$TMP/changes.diff" | sed "s|^+++ $TMP/|  |; s|\t.*||"
    echo; cat "$TMP/changes.diff"
  fi
  exit 0
fi

# --names, even for all of them: a dump without it deletes other assets' sources.
run nodebpy dump "$BLEND" --names "${NAMES[@]}"
run nodebpy build
run nodebpy check
run python scripts/annotation_assets.py index
git -C "$ROOT" status --short -- assets src/yt_volumes/assets
