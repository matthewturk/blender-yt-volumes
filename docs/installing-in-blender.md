# Installing yt_idv and yt-volumes in a stock Blender

How to get from a fresh Blender download to F12 renders with yt_idv, without
this repo's dev venv. Tested on Linux x86-64 with the official
`blender-5.2.2-linux-x64.tar.xz` and yt_idv `main`; both routes below render
the README's quick-start frame identically. Not yet tried on macOS or Windows
(yt_idv's offscreen rendering uses EGL, which macOS doesn't have).

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), a C
compiler (yt_idv `main` has Cython extensions and no prebuilt wheel), and an
NVIDIA/AMD/Intel GPU driver with EGL.

## What you have to match

- **Python version.** Blender 5.x embeds CPython 3.13, so every compiled
  package (yt_idv, yt, h5py, ...) must be a `cp313` build. Check with
  `blender -b --python-expr "import sys; print(sys.version)"`.
- **No `Python.h` in Blender.** Blender's bundled Python ships without
  headers, so `pip install "yt_idv @ git+..."` *into Blender's Python*
  fails with `fatal error: Python.h: No such file or directory`. Build
  yt_idv with a uv-managed Python 3.13 instead, which has headers; the
  wheel it produces loads fine in Blender's 3.13.
- **numpy.** Blender bundles its own numpy (2.3.4 in 5.2). Pin the same
  version so installing yt doesn't replace or shadow it:
  ```
  BLENDER_PY=/path/to/blender-5.2.2-linux-x64/5.2/python/bin/python3.13
  NUMPY=$($BLENDER_PY -c "import numpy; print(numpy.__version__)")
  ```

Below, `$BLENDER` is the `blender` executable and `$BLENDER_PY` its bundled
Python as above. Run the commands with `PYTHONPATH` unset.

## Route 1 (recommended): a uv venv that Blender adds to its path

The packages live in an ordinary venv outside Blender, so a Blender upgrade
(or a Steam update) doesn't wipe them, and the venv can be shared between
Blender versions with the same Python minor version.

```
uv venv --python 3.13 ~/blender-ytenv
uv pip install --python ~/blender-ytenv/bin/python \
    "yt_idv @ git+https://github.com/yt-project/yt_idv.git" \
    pooch pandas h5py "numpy==$NUMPY"
```

Then tell Blender about it with a startup script. Blender runs everything in
`<user scripts>/startup/` at launch, including from a desktop icon, so no
environment variables or launch flags are needed:

```
mkdir -p ~/.config/blender/5.2/scripts/startup
cat > ~/.config/blender/5.2/scripts/startup/yt_venv.py <<'EOF'
import os, site
site.addsitedir(os.path.expanduser("~/blender-ytenv/lib/python3.13/site-packages"))
def register(): pass  # Blender warns about startup scripts without one
EOF
```

`site.addsitedir` appends, so Blender's own bundled packages (numpy,
requests, ...) still win over the venv's copies.

Alternative to the startup script: launch with
`PYTHONPATH=~/blender-ytenv/lib/python3.13/site-packages blender --python-use-system-env`.
Blender ignores `PYTHONPATH` without that flag. Here the venv comes *first*
on the path, so the numpy pin matters more.

## Route 2: install into Blender's bundled Python, from a tarball

Self-contained (one directory holds Blender and everything it needs), at the
cost of being undone by a Blender reinstall. Build the yt_idv wheel first,
from a source tarball or a checkout:

```
curl -L -o yt_idv-main.tar.gz \
    https://github.com/yt-project/yt_idv/archive/refs/heads/main.tar.gz
uv build --wheel --python 3.13 --out-dir wheels yt_idv-main.tar.gz
```

then install it and the rest (all of which have `cp313` wheels on PyPI)
straight into Blender's Python:

```
uv pip install --python $BLENDER_PY wheels/yt_idv-*-cp313-*.whl \
    pooch pandas h5py "numpy==$NUMPY"
```

`uv pip install --python` writes into a non-venv interpreter without
needing `ensurepip`. The Blender directory has to be writable by you; for a
system or Steam install, use route 1.

## Route 3: the extension zip with everything bundled

`scripts/build_extension.sh` builds yt-volumes as a Blender extension with
yt, yt_idv and their dependencies inside, one zip per platform, plus a
`deep_zoom` zip for all platforms. Installing them (Preferences > Get
Extensions > Install from Disk) is the whole setup: no uv, no compiler, no
startup script. numpy is left out; Blender's is used. Deep Zoom stays a
separate extension so it can author camera paths without yt_idv.

```
scripts/build_extension.sh $BLENDER dist/
```

It needs uv and downloads wheels for Linux x64, Windows x64 and macOS arm64
from this one machine. Until yt_idv publishes wheels on PyPI, build one as in
route 2 and point the script at it, for your own platform only:

```
PLATFORMS=linux-x64 WHEELS=wheels/ scripts/build_extension.sh $BLENDER dist/
```

Installed this way the module is `bl_ext.user_default.yt_volumes`, not
`yt_volumes`, and its preferences are separate from a classic install's.
Don't enable both at once.

## Installing the add-on

```
cd src && zip -r ../yt_volumes.zip yt_volumes -x '*__pycache__*'
```

Preferences > Add-ons > Install from Disk, pick `yt_volumes.zip`, enable
**yt-volumes**. Or headless:

```
$BLENDER -b --factory-startup --python-expr "
import bpy
bpy.ops.preferences.addon_install(filepath='yt_volumes.zip')
bpy.ops.preferences.addon_enable(module='yt_volumes')
bpy.ops.wm.save_userpref()"
```

For deep camera paths, also install Deep Zoom: the `deep_zoom` zip from
route 3, or zip `extensions/deep_zoom/` yourself, and use Preferences > Get
Extensions > Install from Disk.

In the GUI, set Preferences > System > GPU Backend to **Vulkan** and restart.
Under OpenGL, NVIDIA's EGL refuses to make yt_idv's context current next to
Blender's own (`EGL_BAD_ACCESS` on F12). Batch Cycles renders (`blender -b`)
work under either backend; EEVEE needs Vulkan in both.

## Checking it

```
$BLENDER -b --python-expr "import yt, yt_idv, numpy, addon_utils; \
print(yt.__version__, yt_idv.__file__, numpy.__file__, addon_utils.check('yt_volumes'))"
```

`numpy.__file__` should be under Blender's own `5.2/python`, and the last
value `(True, True)`. Then a real frame, using the files from `examples/`
(`make_simple_blend.py` builds `simple.blend`; the `IsolatedGalaxy` sample
downloads on first use):

```
$BLENDER -b --python-expr "import sys; sys.path.insert(0, '.'); \
import make_simple_blend; make_simple_blend.build('simple.blend')"
$BLENDER -b --python-expr "import sys; from yt_volumes.render_volume import main; sys.exit(main())" \
    -- render_volume.toml --sample IsolatedGalaxy --frames 1 \
    --set blender.file='"simple.blend"' --set blender.device='"OPTIX"'
```

Run both from `examples/` (Blender doesn't put the current directory on
`sys.path`, hence the insert), and pick `OPTIX`, `CUDA`, `HIP` or `CPU` for
`blender.device`. The frame shows the galaxy in its box with the sphere
cutting into it.

To try any of this without touching your real Blender settings, set
`BLENDER_USER_RESOURCES=/some/scratch/dir`; Blender then keeps add-ons,
startup scripts and preferences there instead of `~/.config/blender/5.2`.
