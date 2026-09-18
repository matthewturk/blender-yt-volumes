"""Real, end-to-end demonstration of the whole pipeline: a genuine Cycles
render of a sphere embedded inside the flagged volume box, composited with
real yt_idv volume content via the add-on's ACTUAL two-pass pipeline
(`operators.render_current_frame`) -- not a hand-rolled numpy composite,
not a fabricated RenderRequest. Both the sphere and the volume end up in
one final image, with real occlusion in BOTH directions visible at once:
genuine volume in front of the sphere veils part of it (ordinary Alpha
Over, no depth clip involved), and the real depth clip (driven by the
sphere's own real Cycles Z pass) removes the volume behind it, so the
sphere also visibly occludes something. Background alpha is handled
correctly throughout too.

Runs entirely in-process now, via this repo's own `uv`-managed dev venv --
NOT real Blender, and no subprocess bridge either:

    env -u LD_PRELOAD uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/render_real_demo.py

This used to need a genuine yt_idv subprocess bridge (a throwaway
Python-3.12 venv, `ytidv_worker.py`) because PyPI's yt_idv release (0.5.4)
lists `imgui` as an unconditional install dependency, which has no cp313
wheel and fails to build from source against 3.13's changed C API -- and
this repo's own dev venv is pinned to Python 3.13 to match the pip `bpy`
wheel below. Fixed upstream 2026-09-15 (yt-project/yt_idv#268,
"imgui-optional"): imgui moved to an opt-in `gui` extra (only needed for the
interactive pyglet+imgui control panel -- never for this add-on's own
`egl`-backend, no-GUI rendering path), so a plain `pip install yt_idv` no
longer needs it, and `bpy` + real `yt_idv` now coexist in ONE Python 3.13
interpreter -- confirmed empirically (`import bpy; import yt_idv` together,
a real offscreen EGL render, and this whole script end to end, all in this
repo's own venv). Not yet in a PyPI release as of this writing (0.5.4
predates it), hence the git dependency in `pyproject.toml`'s `ytidv` group
-- see README "yt_idv integration" for the full story.

`operators.RENDERER_MODES` is used completely unmodified -- no monkeypatch
needed any more. Everything, including the real yt_idv render itself, is
`operators.render_current_frame()` run against a real scene: Pass 1's real
Cycles beauty+Z render, `read_depth_buffer`, per-object `RenderRequest`
construction (satisfied by the real `yt_idv_*_renderer` functions in
`yt_idv_renderer.py`), `write_render_result`, Pass 2's real compositor
Alpha-Over graph and final render.

Black-background handling, addressed three ways:
  1. `scene.render.film_transparent = True` -- Pass 1's own Cycles beauty
     EXR has alpha=0 wherever there's no real geometry (not opaque black),
     so the Alpha-Over chain composites the yt_idv layer over a genuinely
     transparent background rather than a black rectangle.
  2. yt_idv's own OpenGL clear color is (0, 0, 0, 0) (confirmed earlier in
     this exploration) -- background pixels in the volume layer are also
     truly transparent, not opaque black, so the final Alpha-Over result is
     transparent wherever BOTH layers are empty.
  3. Verified below, not just asserted: the final composited Render Result
     is read back and a background-corner pixel's alpha is checked directly
     (see `_verify_background_alpha`), and a checkerboard-composited PNG is
     written alongside the plain output specifically so transparency is
     visually obvious (a black background would show as a black rectangle
     over the checkerboard; a correctly transparent one lets the checker
     pattern show through).
"""

import os
import sys

import bpy
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
IMAGES_DIR = os.path.normpath(os.path.join(HERE, "..", "images"))

WORK_DIR = os.path.join(HERE, "_real_demo_work")

RENDERER_MODE = os.environ.get("VOLUME_OVERRIDE_MODE", "projection")  # max_intensity | transfer_function | projection
MODE_KEY_BY_NAME = {
    "max_intensity": "YT_IDV_MAX_INTENSITY",
    "transfer_function": "YT_IDV_TRANSFER_FUNCTION",
    "projection": "YT_IDV_PROJECTION",
}

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)

from cycles_volume_override import operators, properties  # noqa: E402


class _StubPrefs:
    renderer_mode = MODE_KEY_BY_NAME[RENDERER_MODE]
    work_dir = WORK_DIR
    verbose_logging = True


def build_scene():
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # The flagged "volume box": edge length 2 (spans -1..1 on each axis).
    # yt_idv_renderer's registration transform uses this object's own
    # bounding_box_local/matrix_world directly as the Blender-world <->
    # yt_idv-unitary mapping (see its module docstring, "Cycles Z-buffer
    # wiring") -- no special setup needed beyond flagging it.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    box = bpy.context.active_object
    box.name = "FlaggedVolumeBox"
    box.volume_override.is_volume_override = True

    # Sphere embedded INSIDE the box, off toward the camera but not poking
    # out past the box's own near face -- chosen for BOTH occlusion
    # directions to be visible at once, not just one:
    #   - genuine volume in FRONT of the sphere (between the box's near
    #     face and the sphere's own near surface) draws OVER the sphere via
    #     the compositor's Alpha Over (foreground=volume, background=beauty
    #     -- see compositor_build.py's module docstring), so part of the
    #     sphere should read as veiled/occluded by gas, not fully opaque.
    #   - genuine volume BEHIND the sphere gets removed by the real depth
    #     clip, so the sphere still visibly occludes something too.
    # This used to sit at a box CORNER with radius = half the box's edge
    # (an earlier iteration): first at the corner FARTHEST from the camera,
    # which left ~nothing behind the sphere for the clip to remove at all
    # (confirmed by verify_overlap.py: 0.0% alpha drop in every mode); then
    # at the NEAREST corner, which fixed the clip but, combined with a
    # sphere half as wide as the box, ended up occluding the box's entire
    # visible silhouette from camera -- no volume showed at all, and there
    # was no near-side volume left to occlude the sphere either (the sphere
    # sat right at the box's own near face, nothing in front of it).
    # Smaller and more central fixes both: room for the box's own
    # silhouette to show around it, AND room for real volume on both sides
    # of it along camera rays.
    box_edge_length = 2.0
    sphere_radius = box_edge_length * 0.28
    sphere_center = (-0.55, -0.55, 0.55)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=sphere_radius, location=sphere_center, segments=64, ring_count=32,
    )
    sphere = bpy.context.active_object
    sphere.name = "EmbeddedSphere"

    mat = bpy.data.materials.new("EmbeddedSphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.15, 0.05, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35
    sphere.data.materials.append(mat)

    # Camera: pulled back on the diagonal so the box's own silhouette (where
    # the volume shows through around the sphere) and the embedded sphere
    # (partly veiled by near-side volume, partly occluding far-side volume)
    # are both clearly visible together.
    bpy.ops.object.camera_add(location=(-3.4, -4.0, 2.6))
    cam = bpy.context.active_object
    cam.name = "DemoCamera"
    target = mathutils.Vector((0.35, 0.35, 0.35))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 40.0
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(4.0, -3.0, 6.0))
    sun = bpy.context.active_object
    # Was 3.0 -- blew the specular highlight's linear R channel out past 1.0
    # (measured ~1.5) while G/B stayed under it, so this script's own
    # tonemap-free `_srgb_encode` clipped R first and left a warm/pink-white
    # highlight instead of a clean white one (see conversation: confirmed via
    # direct EXR inspection that the yt_idv volume layer was genuinely
    # (0,0,0,0) there -- this was a real Cycles specular highlight, not a
    # compositing artifact). Lowered so the highlight's linear peak stays
    # under 1.0 and all three channels clip together (or not at all).
    sun.data.energy = 1.2

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512  # square -- see yt_idv_renderer's own
    #                                   documented non-square-aspect caveat
    scene.render.film_transparent = True  # see module docstring, point 1
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = os.path.join(WORK_DIR, "final_output_")

    return box, sphere, cam


def _srgb_encode(linear):
    linear = np.clip(linear, 0.0, None)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _write_png(path, rgba_float):
    """Write an (H, W, 4) float array straight to PNG via Blender's own
    image API (no external PNG library needed) -- caller is responsible for
    already having values in a sane 0..1-ish display range.
    """
    h, w = rgba_float.shape[:2]
    img = bpy.data.images.new("tmp_png_out", width=w, height=h, alpha=True)
    img.pixels.foreach_set(np.clip(rgba_float, 0.0, 1.0).astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _verify_and_export(scene):
    """Read pass 2's real final EXR back from disk (NOT the in-memory
    "Render Result" datablock -- in `--background` mode its `.pixels`
    buffer reads back as size (0, 0) after `write_still=True`, a background-
    mode quirk unrelated to whether the render itself is correct; the file
    Pass 2 actually wrote to disk is the authoritative output), verify the
    background is genuinely transparent (not opaque black), and write both
    a plain display-ready PNG and a checkerboard-composited PNG that makes
    transparency visually obvious.
    """
    final_path = scene.render.frame_path()
    if not os.path.exists(final_path):
        # Real Blender didn't append a frame-number suffix for this single-
        # frame still (frame_path()'s prediction and the file `render.render`
        # actually wrote disagreed by that suffix alone) -- fall back to
        # whatever actually landed on disk next to it.
        import glob
        candidates = sorted(glob.glob(scene.render.filepath + "*"))
        if not candidates:
            raise RuntimeError(f"no final output found near {scene.render.filepath!r} (predicted {final_path!r})")
        final_path = candidates[-1]
    img = bpy.data.images.load(final_path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
    finally:
        bpy.data.images.remove(img)

    # Background-alpha check: a corner pixel should have neither the
    # sphere nor the box's volume silhouette anywhere near it.
    corner_alpha = float(pixels[2, 2, 3])
    print(f"[render_real_demo] background corner pixel alpha = {corner_alpha:.4f} "
          f"(expect ~0.0 -- see module docstring point 3)")
    if corner_alpha > 0.02:
        print("[render_real_demo] WARNING: background does not look transparent -- "
              "film_transparent or the yt_idv layer's own alpha may not be wired correctly")

    # Plain display-ready PNG: sRGB-encode the premultiplied RGB, keep alpha.
    display = pixels.copy()
    display[..., :3] = _srgb_encode(display[..., :3])
    plain_path = os.path.join(IMAGES_DIR, "24_real_demo_sphere_and_volume.png")
    _write_png(plain_path, display)
    print(f"[render_real_demo] wrote {plain_path}")

    # Checkerboard-composited PNG: proves the alpha is really being used --
    # premultiplied-alpha-over onto a two-tone checker background. If the
    # background were actually opaque black, the checker would NOT show
    # through here.
    square = 32
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // square) + (xx // square)) % 2).astype(np.float32)
    checker_rgb = 0.25 + 0.35 * checker[..., None]  # two mid-grays
    composited_rgb = display[..., :3] + checker_rgb * (1.0 - display[..., 3:4])
    composited = np.concatenate([composited_rgb, np.ones_like(display[..., 3:4])], axis=-1)
    checker_path = os.path.join(IMAGES_DIR, "25_real_demo_over_checkerboard.png")
    _write_png(checker_path, composited)
    print(f"[render_real_demo] wrote {checker_path}")


def main():
    properties.register()
    operators._get_prefs = lambda context: _StubPrefs()

    box, sphere, cam = build_scene()
    print(f"[render_real_demo] scene built: box={box.name} sphere={sphere.name} "
          f"cam={cam.name} mode={RENDERER_MODE}")

    operators.render_current_frame(bpy.context)
    print(f"[render_real_demo] real two-pass pipeline finished; "
          f"final EXR at {bpy.context.scene.render.frame_path()!r}")

    _verify_and_export(bpy.context.scene)


if __name__ == "__main__":
    main()
