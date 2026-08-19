#!/usr/bin/env bash
# Sets up a THROWAWAY venv for exploring yt_idv, separate from this repo's
# own bpy==5.2.0/Python-3.13 dev venv (see pyproject.toml). Why separate:
# bpy==5.2.0 simply isn't published for Python 3.12 (confirmed -- `pip
# install bpy==5.2.0` fails there), and yt_idv's own supported range tops
# out at 3.12 with a heavy compiled/OpenGL dependency stack that doesn't
# belong in the lightweight bpy-testing venv anyway -- so there is currently
# no single Python this repo's own venv could use to run both. Real
# Blender's OWN embedded Python (NOT bpy-as-a-pip-wheel) doesn't have this
# conflict -- see README "yt_idv integration".
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
