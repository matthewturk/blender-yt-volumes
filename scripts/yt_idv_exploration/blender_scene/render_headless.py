"""Headless driver: open a real, artist-authored `.blend` file (built in
whatever Blender you normally use -- yt_idv doesn't need to be installed
there at all, only the add-on itself, since flagging/configuring objects is
just ordinary custom-property data) and render a handful of frames through
the REAL add-on pipeline (`bpy.ops.render.volume_override`) and REAL yt_idv,
entirely inside THIS REPO's own `uv`-managed dev venv -- no separate
Blender binary, no `--background` flag needed at all, since `bpy` (the
pip-installable Blender-as-a-Python-module wheel) is itself a full Blender
that can open arbitrary `.blend` files directly via `bpy.ops.wm.
open_mainfile`.

Run with:

    uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/render_headless.py

Two real things found only by actually running this against a real,
externally-authored file, not this repo's own from-scratch demo scenes:

  1. `bpy.ops.preferences.addon_enable(module="cycles_volume_override")`
     works with nothing more than `sys.path.insert(0, ".../src")` -- no
     need for the `_get_prefs` stub-monkeypatch this directory's other demo
     scripts (`render_real_demo.py`, `verify_overlap.py`) use. That
     monkeypatch exists to dodge a DIFFERENT problem (building a synthetic
     scene from scratch, no real add-on install anywhere) -- it was never
     required just to reach real Preferences/the real operator headlessly.
  2. **Per-frame output filenames are NOT automatic.** This add-on's pass2
     write (`BLENDER_WORKBENCH` engine + compositor graph, `write_still=
     True` -- see `operators.render_current_frame`) does not append the
     frame number to `scene.render.filepath` the way a plain single-image
     render does. Looping frames without resetting `filepath` each time
     confirmed silently overwriting the SAME output file every frame.
     Fixed below by setting it explicitly inside the loop.

Also worth knowing, not a bug: `renderer_mode` (and every other yt_idv
Preferences-panel knob) lives in `AddonPreferences`, NOT the `.blend` file
-- it won't travel with a file authored/saved in a different Blender
profile. `scene_setup.py` (in this same directory) is where that -- and
which dataset/field yt_idv actually renders -- lives instead: EDIT THAT
FILE, not this one, for a routine "different scene" change. This script
just imports it by name (`SCENE_SETUP_MODULE` below), applies its
`configure_preferences(prefs)`, and points the flagged object's
`yt_idv_volume_provider` at its `setup_volume` -- see yt_idv_renderer.py's
"Custom volume setup" for the underlying mechanism. Per-object state
(`is_volume_override`, density/absorption/color, `dataset_path`/`dataset_
kind`, camera animation) DOES travel with the .blend file, since it's real
Object/Scene data -- this script only sets `is_volume_override`/
`density_scale` explicitly because the specific test file it was verified
against (`test-cycles-override.blend`, a plain animated-camera-orbiting-a-
cube scene) hadn't actually flagged its cube.

One harmless, expected headless artifact you'll see in the log and can
ignore: `ERROR Invalid operator call 'RENDER_OT_view_show'` --
`operators._show_render_result` tries to pop the Render Result into an
Image Editor area, which doesn't exist in this headless session; every
frame still reports `{'FINISHED'}` regardless.

Verified end to end against a real 1920x1080, 250-frame, EEVEE-by-default,
Blender-5.3-saved file, rendered here at a deliberately small 480x270/32
samples for speed: 3 sample frames across the camera's zoom (first pays
yt_idv's one-time dataset-load cost, ~10s; later frames ~2s each once the
dataset/GL-context caches are warm -- see yt_idv_renderer.py's module
docstring on those caches), each producing genuinely different, non-blank
output (confirmed via mean pixel value, not just "didn't crash").
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, ADDON_SRC_PARENT)
sys.path.insert(0, HERE)  # so SCENE_SETUP_MODULE resolves as an importable module

import bpy  # noqa: E402

BLEND_PATH = os.path.join(REPO_ROOT, "test-cycles-override.blend")
OUT_DIR = os.path.join(HERE, "_headless_render_out")
# RENDER_START_FRAME/RENDER_END_FRAME (env vars, both inclusive, default the
# full 1-250 range): lets a shell loop invoke this script repeatedly across
# successive frame ranges, each a FRESH process/GPU context -- needed
# because Cycles/OptiX was found to leak GPU memory across ~180 consecutive
# bpy.ops.render.render() calls in one long-lived process (real crash:
# "System is out of GPU memory" at frame 183 of a real HD run). A single
# process safely handles a few hundred frames at low-res but far fewer at
# full HD -- keep chunks well under ~150 frames at HD to leave margin.
# scene_setup.py's FIXED_CMAP_MIN/MAX (not this script) is what keeps the
# color mapping consistent across chunk boundaries -- see its own comment.
FRAMES = range(
    int(os.environ.get("RENDER_START_FRAME", 1)),
    int(os.environ.get("RENDER_END_FRAME", 250)) + 1,
)
SCENE_SETUP_MODULE = "scene_setup"  # edit THAT file for dataset/field/look changes, not this one

scene_setup = __import__(SCENE_SETUP_MODULE)

os.makedirs(OUT_DIR, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)
bpy.ops.preferences.addon_enable(module="cycles_volume_override")

scene = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
prefs.verbose_logging = True
scene_setup.configure_preferences(prefs)

# This test file's own cube wasn't actually flagged -- see module docstring.
# Skip/adjust this block once you're pointing BLEND_PATH at a file where
# the volume object(s) were already flagged (and tuned) in the GUI.
cube = bpy.data.objects["Cube"]
cube.volume_override.is_volume_override = True
cube.volume_override.density_scale = 3.0
cube.volume_override.yt_idv_volume_provider = f"{SCENE_SETUP_MODULE}:setup_volume"

scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.render.resolution_x = 1920  # full HD -- yt_idv renders SQUARE internally at
scene.render.resolution_y = 1080  # max(x, y) then crops, so this is a 1920x1920 yt_idv render per frame
scene.render.image_settings.file_format = "PNG"

# GPU/OptiX -- real bug found while profiling the first HD run: the .blend
# file itself already has scene.cycles.device == "GPU" (saved from whatever
# desktop Blender authored it), but that alone does nothing -- Cycles also
# needs a compute BACKEND selected at the Preferences level
# (compute_device_type), which a fresh headless bpy session starts at "NONE"
# regardless of what the .blend says. Confirmed empirically: without this
# block, pass1's Cycles render silently fell back to the CPU (an old Xeon
# X5650) and took ~32s/frame at 1920x1080/32 samples -- over 90% of total
# per-frame time, dwarfing yt_idv's own ~1.6s and the EXR read/write's
# ~0.5s combined. `get_devices()` must be called before `.devices` is
# populated; enabling ONLY the OPTIX entry (not also CUDA/CPU) avoids
# Cycles splitting work across a fast GPU and a much slower CPU device.
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = "OPTIX"
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    device.use = device.type == "OPTIX"
    print(f"[render_headless] cycles device {device.name!r} ({device.type}): use={device.use}")
scene.cycles.device = "GPU"
scene.cycles.denoiser = "OPTIX"  # AI/GPU denoiser, not the CPU-only OpenImageDenoise default
if hasattr(scene.cycles, "denoising_use_gpu"):
    scene.cycles.denoising_use_gpu = True

FRAME_PREFIX = "frame2_"  # distinct from the earlier 480x270 "frame_" run -- doesn't overwrite it

try:
    for frame in FRAMES:
        scene.frame_set(frame)
        # Per-frame, NOT hoisted above the loop -- see module docstring, finding 2.
        scene.render.filepath = os.path.join(OUT_DIR, f"{FRAME_PREFIX}{frame:04d}")
        result = bpy.ops.render.volume_override(animation=False)
        print(f"[render_headless] frame {frame}: {result} -> {scene.render.filepath}.png")
except KeyboardInterrupt:
    # Ctrl+C mid-run -- every frame rendered so far already finished and
    # was written to disk (each iteration is fully synchronous, see
    # `_render_with_yt_idv`), so there's nothing to clean up; just stop
    # instead of dumping a traceback for what is a completely normal way
    # to end a long/exploratory headless run early.
    print(f"[render_headless] stopped early at frame {frame} (KeyboardInterrupt)")
else:
    print("[render_headless] done, wrote:", sorted(os.listdir(OUT_DIR)))
