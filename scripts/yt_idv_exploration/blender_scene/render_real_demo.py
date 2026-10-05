"""End-to-end demo: a Cycles sphere embedded in the flagged volume box,
composited with yt_idv volume content through the add-on's two-pass
pipeline (`operators.render_current_frame`). Volume in front of the sphere
veils it (Alpha Over) and the depth clip removes volume behind it.

Runs in-process in this repo's dev venv (bpy and yt_idv together):

    env -u LD_PRELOAD uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/render_real_demo.py

Set VOLUME_OVERRIDE_MODE to max_intensity, transfer_function, or projection.

Background transparency:
  1. `film_transparent = True` gives pass 1 alpha=0 where there's no geometry.
  2. yt_idv clears to (0, 0, 0, 0), so the volume layer is transparent too.
  3. `_verify_and_export` checks a corner pixel's alpha and also writes a
     checkerboard composite to make transparency visible.
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

from yt_volumes import operators, properties  # noqa: E402


class _StubPrefs:
    renderer_mode = MODE_KEY_BY_NAME[RENDERER_MODE]
    work_dir = WORK_DIR
    verbose_logging = True


def build_scene():
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Flagged volume box spanning -1..1; its bounds map to yt_idv's domain.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    box = bpy.context.active_object
    box.name = "FlaggedVolumeBox"
    box.volume_override.is_volume_override = True

    # Sphere inside the box, toward the camera, with volume both in front of
    # it (veils the sphere) and behind it (removed by the depth clip).
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

    # Camera on the diagonal so both the box silhouette and sphere are visible.
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
    # Kept low so the specular highlight stays under 1.0 and doesn't clip to
    # pink in the tonemap-free `_srgb_encode`.
    sun.data.energy = 1.2

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
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
    """Write an (H, W, 4) float array (display range 0..1) to PNG via bpy."""
    h, w = rgba_float.shape[:2]
    img = bpy.data.images.new("tmp_png_out", width=w, height=h, alpha=True)
    img.pixels.foreach_set(np.clip(rgba_float, 0.0, 1.0).astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _verify_and_export(scene):
    """Read pass 2's final EXR, check background alpha, and write PNGs.

    Reads the file on disk: headless, the "Render Result" image's `.pixels`
    is empty after `write_still=True`.
    """
    final_path = scene.render.frame_path()
    if not os.path.exists(final_path):
        # frame_path() may predict a frame-number suffix that wasn't written.
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

    # A corner pixel should be empty background.
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

    # Premultiplied alpha-over onto a checkerboard, to show transparency.
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
