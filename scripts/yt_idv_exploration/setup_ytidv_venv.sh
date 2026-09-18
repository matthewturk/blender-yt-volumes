#!/usr/bin/env bash
# HISTORICAL -- kept for the record, no longer the recommended path. This
# separate venv existed to work around yt_idv's now-fixed unconditional
# `imgui` dependency (no cp313 wheel, fails to build against Python 3.13's
# changed C API): yt_idv's own supported range topped out at 3.12, which
# doesn't match this repo's own bpy==5.2.0/Python-3.13 dev venv (see
# pyproject.toml), so there was no single Python version that could hold
# both. Fixed upstream 2026-09-15 (yt-project/yt_idv#268, "imgui-optional")
# -- imgui is now an opt-in extra, so a plain install no longer needs it,
# and `bpy` + real `yt_idv` now coexist in ONE Python-3.13 venv. Prefer
# `uv sync --group ytidv` in this repo's own venv instead -- see README
# "Development & testing" -> "Headless yt_idv testing, in this same venv".
#
# Original rationale, for context on why this script and the subprocess
# bridge it enabled (ytidv_worker.py) exist at all:
#
# Usage:
#   ./scripts/yt_idv_exploration/setup_ytidv_venv.sh /path/to/venv
#   source /path/to/venv/bin/activate
#   python scripts/yt_idv_exploration/explore_basic_render.py

set -euo pipefail

VENV_DIR="${1:?usage: $0 <venv-dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YT_IDV_CHECKOUT="$HERE/../../yt_idv"

if [ ! -d "$YT_IDV_CHECKOUT" ]; then
    echo "Expected a yt_idv checkout at $YT_IDV_CHECKOUT -- not found." >&2
    exit 1
fi

python3.12 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$YT_IDV_CHECKOUT" -q
# yt's sample-data loader (yt.load_sample) and the Enzo frontend IsolatedGalaxy
# uses both lazy-import these; not part of yt_idv's own install_requires.
"$VENV_DIR/bin/pip" install pooch pandas h5py -q

echo "Done. Activate with: source $VENV_DIR/bin/activate"
