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
# Uses `uv` rather than plain `python -m venv`/`pip` -- two real, concrete
# advantages, not just a style preference: (1) it fetches an isolated
# Python build for you (`uv python install`), so this no longer requires a
# matching `python3.1X` to already be on PATH; (2) `uv pip install` is
# dramatically faster, which matters here because getting the Python
# version WRONG means a slow, doomed source-compile of `imgui` (yt_idv's
# GUI/OpenGL dependency) instead of an instant prebuilt-wheel install.
#
# DEFAULT_PYTHON=3.11, not 3.12, confirmed empirically (not just "yt_idv's
# docs top out at 3.12"): `imgui` 2.0.0 (the version yt_idv 0.5.4 currently
# pulls in) publishes prebuilt wheels only up to cp311 -- NOTHING for
# cp312/cp313 on PyPI, for any platform, checked directly against PyPI's
# own JSON API. 3.12 MIGHT still work by compiling `imgui` from source (a
# C++ compiler + Python dev headers, both needed) -- this project's own
# earlier exploration did exactly that, successfully -- but 3.13 does NOT,
# even with a fully header-equipped, uv-managed CPython 3.13 supplying the
# missing headers: `imgui`'s own vendored/pre-generated Cython C++ source
# calls CPython private APIs (`_PyLong_AsByteArray`, `_PyGen_
# SetStopIterationValue`) whose signatures changed/disappeared in 3.13 --
# a genuine upstream incompatibility, not an environment problem, and not
# fixable by anything on the installing side (uv included) short of a
# newer `imgui` release. 3.11 is the newest version confirmed to need NO
# compilation at all -- every dependency in the full yt_idv/yt stack
# (`imgui`, `numpy`, `matplotlib`, `pandas`, `h5py`, ...) has a prebuilt
# wheel there. See README "Installing yt_idv into your Blender's Python"
# for what this means for your REAL Blender specifically -- the version
# that matters is BLENDER'S bundled Python, not this venv's, if you intend
# to `sys.path`-attach this venv's site-packages directly into Blender's
# own process (only safe when the two match EXACTLY -- compiled extensions
# can't load across a Python version mismatch, full stop, regardless of
# uv); if Blender's own bundled Python is newer than 3.11 (or otherwise
# doesn't have wheels here), run yt_idv in THIS venv as a separate
# subprocess instead and bridge to it (see `ytidv_worker.py` for the
# pattern already used elsewhere in this exploration for exactly this
# reason) rather than trying to import it directly inside Blender.
#
# Usage:
#   ./scripts/yt_idv_exploration/setup_ytidv_venv.sh /path/to/venv [python-version]
#   source /path/to/venv/bin/activate
#   python scripts/yt_idv_exploration/explore_basic_render.py

set -euo pipefail

VENV_DIR="${1:?usage: $0 <venv-dir> [python-version]}"
PYTHON_VERSION="${2:-3.11}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YT_IDV_CHECKOUT="$HERE/../../yt_idv"

if [ ! -d "$YT_IDV_CHECKOUT" ]; then
    echo "Expected a yt_idv checkout at $YT_IDV_CHECKOUT -- not found." >&2
    exit 1
fi

uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -e "$YT_IDV_CHECKOUT"
# yt's sample-data loader (yt.load_sample) and the Enzo frontend IsolatedGalaxy
# uses both lazy-import these; not part of yt_idv's own install_requires.
uv pip install --python "$VENV_DIR/bin/python" pooch pandas h5py

echo "Done. Activate with: source $VENV_DIR/bin/activate"
