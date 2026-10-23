"""Headless driver: open an existing `.blend` file and render a range of
frames through the add-on (`bpy.ops.render.volume_override`) and yt_idv,
using the pip-installed `bpy` module in this repo's dev venv.

Run with:

    uv run --group ytidv python \\
        scripts/headless/render_headless.py

Dataset, field, renderer mode, and colormap live in `scene_setup.py`; edit
that file for routine scene changes, not this one. The look is applied to
the volume from `scene_setup.configure_volume` on every run.

The `ERROR Invalid operator call 'RENDER_OT_view_show'` line in the log is
harmless (there is no Image Editor to show the result in headlessly).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, ADDON_SRC_PARENT)
sys.path.insert(0, HERE)  # so SCENE_SETUP_MODULE resolves as an importable module

import bpy  # noqa: E402
from yt_volumes import operators  # noqa: E402

BLEND_PATH = os.path.join(HERE, "test-cycles-override.blend")
OUT_DIR = os.path.join(HERE, "_headless_render_out")
# RENDER_START_FRAME/RENDER_END_FRAME env vars (inclusive, default 1-250)
# render part of the animation, e.g. to resume an interrupted run.
FRAMES = range(
    int(os.environ.get("RENDER_START_FRAME", 1)),
    int(os.environ.get("RENDER_END_FRAME", 250)) + 1,
)
SCENE_SETUP_MODULE = "scene_setup"  # module with configure_volume/setup_volume

scene_setup = __import__(SCENE_SETUP_MODULE)

os.makedirs(OUT_DIR, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)
bpy.ops.preferences.addon_enable(module="yt_volumes")

scene = bpy.context.scene
prefs = bpy.context.preferences.addons["yt_volumes"].preferences
prefs.verbose_logging = True

# Flags the volume object here because the test file doesn't; drop this if
# your .blend already has it flagged. Flagged volume objects must be an
# Empty with Display Type 'Cube', not a mesh.
VOLUME_OBJECT_NAME = "Volume 1"
cube = bpy.data.objects[VOLUME_OBJECT_NAME]
if cube.type != "EMPTY" or cube.empty_display_type != "CUBE":
    raise RuntimeError(
        f"{BLEND_PATH}'s {VOLUME_OBJECT_NAME!r} object is a {cube.type}, "
        "not an Empty-Cube -- flagged volume objects are required to be "
        "one (see operators.build_volume_params's own docstring). Replace "
        "it in the .blend file (Add > Empty > Cube, matching the desired "
        "transform, Object Properties > Viewport Display > Cube) and "
        "re-save before running this script."
    )
cube.volume_override.is_volume_override = True
scene_setup.configure_volume(cube.volume_override)
cube.volume_override.yt_idv_volume_provider = f"{SCENE_SETUP_MODULE}:setup_volume"

scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = 1920  # yt_idv renders square at max(x, y), then crops
scene.render.resolution_y = 1080
scene.render.image_settings.file_format = "PNG"

# A fresh headless session has compute_device_type "NONE" whatever the
# .blend says, so Cycles silently uses the CPU unless this is set.
# get_devices() populates .devices; only OPTIX is enabled so work isn't split
# with the much slower CPU.
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = "OPTIX"
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    device.use = device.type == "OPTIX"
    print(f"[render_headless] cycles device {device.name!r} ({device.type}): use={device.use}")
scene.cycles.device = "GPU"
scene.cycles.denoiser = "OPTIX"  # GPU denoiser instead of the OpenImageDenoise default
if hasattr(scene.cycles, "denoising_use_gpu"):
    scene.cycles.denoising_use_gpu = True

FRAME_PREFIX = "frame5_"  # output filename prefix

# render_session applies the add-on's scene settings once for the whole run;
# without it they are forced and restored every frame, which invalidates
# Cycles caches and is very slow.
try:
    with operators.render_session(bpy.context):
        for frame in FRAMES:
            scene.frame_set(frame)
            # Must be set per frame: the add-on's output is not auto-numbered.
            scene.render.filepath = os.path.join(OUT_DIR, f"{FRAME_PREFIX}{frame:04d}")
            result = bpy.ops.render.volume_override(animation=False)
            print(f"[render_headless] frame {frame}: {result} -> {scene.render.filepath}.png")
except KeyboardInterrupt:
    # Frames finished so far are already on disk; nothing to clean up.
    print(f"[render_headless] stopped early at frame {frame} (KeyboardInterrupt)")
else:
    print("[render_headless] done, wrote:", sorted(os.listdir(OUT_DIR)))
