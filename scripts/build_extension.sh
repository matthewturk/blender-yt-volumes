#!/usr/bin/env bash
# Build per-platform yt-volumes extension zips with their Python dependencies bundled,
# plus the Deep Zoom extension (pure Python, one zip).
#   scripts/build_extension.sh <blender> <out_dir>
# PLATFORMS="linux-x64 ..." limits the platforms; WHEELS=<dir> adds local wheels (e.g. a yt_idv build).
set -euo pipefail
BLENDER=$1; mkdir -p "$2"; OUT=$(realpath "$2"); STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
cp -r "$(dirname "$0")/../src/yt_volumes" "$STAGE/" && mkdir "$STAGE/yt_volumes/wheels"
# Blender platform -> uv platform, then pip's wheel tags for it.
declare -A TAGS=([linux-x64]="x86_64-manylinux_2_28 manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64 linux_x86_64"
                 [windows-x64]="x86_64-pc-windows-msvc win_amd64"
                 [macos-arm64]="aarch64-apple-darwin macosx_11_0_arm64 macosx_10_9_universal2")
PLATS=(${PLATFORMS:-${!TAGS[@]}})
for bp in "${PLATS[@]}"; do
  read -r uvplat tags <<< "${TAGS[$bp]}"
  # numpy comes from Blender.
  uv pip compile -q --python-version 3.13 --python-platform "$uvplat" --only-binary :all: --no-emit-package numpy \
      ${WHEELS:+--find-links "$WHEELS"} --no-header --no-annotate - -o "$STAGE/$bp.txt" <<< $'yt_idv\nyt\npooch\npandas\nh5py'
  uv run -q --no-project --python 3.13 --with pip -- pip download -q --no-deps --only-binary :all: --python-version 3.13 \
      $(printf -- '--platform %s ' $tags) ${WHEELS:+--find-links "$WHEELS"} -r "$STAGE/$bp.txt" -d "$STAGE/yt_volumes/wheels"
done
# Prepended: top-level TOML keys can't follow the manifest's tables.
M=$STAGE/yt_volumes/blender_manifest.toml
{ printf 'platforms = [%s]\n' "$(printf '"%s", ' "${PLATS[@]}")"
  printf 'wheels = [\n%s]\n' "$(cd "$STAGE/yt_volumes" && ls wheels/*.whl | sed 's|.*|  "./&",|')"
  cat "$M"; } > "$M.new" && mv "$M.new" "$M"
"$BLENDER" --command extension build --split-platforms --source-dir "$STAGE/yt_volumes" --output-dir "$OUT"
"$BLENDER" --command extension build --source-dir "$(dirname "$0")/../extensions/deep_zoom" --output-dir "$OUT"
