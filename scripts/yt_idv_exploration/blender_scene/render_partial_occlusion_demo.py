"""Demo of the volume occluding part of a Cycles sphere (the opposite of
`render_real_demo.py`). No depth clip is involved: the add-on's Alpha Over
draws the volume layer over the beauty pass wherever the volume has alpha.

The sphere sits behind the box relative to the camera and offset sideways,
so part of it is hidden under the volume's screen footprint and part peeks
out beside it. (A sphere placed inside the box is hidden completely, since
the colormap's low-density opacity floor makes the interior near-opaque.)

Run with real Blender (needs a real Cycles render) and the yt_idv subprocess
bridge (`ytidv_worker.py`, venv from YTIDV_VENV_PYTHON):

    /data/SteamLibrary/steamapps/common/Blender/blender --background \\
        --python scripts/yt_idv_exploration/blender_scene/render_partial_occlusion_demo.py

`_verify_and_export` checks background alpha, then compares beauty vs.
final color at a "behind the box" pixel (should differ a lot) and a
"peeking out" pixel (should stay close).
"""

import json
import os
import subprocess
import sys

import bpy
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
IMAGES_DIR = os.path.normpath(os.path.join(HERE, "..", "images"))

YTIDV_VENV_PYTHON = os.environ.get(
    "YTIDV_VENV_PYTHON",
    "/tmp/claude-1000/-home-mturk-yt-cycles-yt/db045906-ff8b-426c-bb6d-8949e7ae6506/scratchpad/ytidv_venv/bin/python",
)
WORKER_SCRIPT = os.path.join(HERE, "ytidv_worker.py")
WORK_DIR = os.path.join(HERE, "_partial_occlusion_demo_work")

RENDERER_MODE = os.environ.get("VOLUME_OVERRIDE_MODE", "max_intensity")  # max_intensity | transfer_function | projection
# max_intensity gives a more sharply bounded blob than projection.
MODE_KEY_BY_NAME = {
    "max_intensity": "YT_IDV_MAX_INTENSITY",
    "transfer_function": "YT_IDV_TRANSFER_FUNCTION",
    "projection": "YT_IDV_PROJECTION",
}
# (plain, checkerboard) output basenames per mode; max_intensity's are
# referenced by the READMEs.
OUTPUT_IMAGE_NAMES = {
    "max_intensity": ("28_partial_occlusion_sphere_and_volume.png", "29_partial_occlusion_over_checkerboard.png"),
}

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)

from yt_volumes import external_render, operators, properties  # noqa: E402


def _bridge_render(request):
    """Stand-in for a yt_idv_*_renderer that shells out to the yt_idv venv."""
    req_json = {
        "camera": {
            "matrix_world": request.camera.matrix_world,
            "projection": request.camera.projection,
            "lens_mm": request.camera.lens_mm,
            "sensor_width_mm": request.camera.sensor_width_mm,
            "sensor_height_mm": request.camera.sensor_height_mm,
            "sensor_fit": request.camera.sensor_fit,
            "shift_x": request.camera.shift_x,
            "shift_y": request.camera.shift_y,
            "ortho_scale": request.camera.ortho_scale,
            "clip_start": request.camera.clip_start,
            "clip_end": request.camera.clip_end,
            "resolution_x": request.camera.resolution_x,
            "resolution_y": request.camera.resolution_y,
            "pixel_aspect_x": request.camera.pixel_aspect_x,
            "pixel_aspect_y": request.camera.pixel_aspect_y,
            "projection_matrix": request.camera.projection_matrix,
        },
        "depth": {
            "depth_convention": request.depth.depth_convention,
            "no_hit_value": request.depth.no_hit_value,
            "row_order": request.depth.row_order,
            "units": request.depth.units,
        },
        "volume": {
            "matrix_world": request.volume.matrix_world,
            "bounding_box_local": request.volume.bounding_box_local,
        },
        "frame_number": request.frame.frame_number,
    }
    req_path = os.path.join(WORK_DIR, "bridge_request.json")
    depth_path = os.path.join(WORK_DIR, "bridge_depth.npy")
    out_path = os.path.join(WORK_DIR, "bridge_result.npy")
    with open(req_path, "w") as f:
        json.dump(req_json, f)
    np.save(depth_path, np.asarray(request.depth.data, dtype=np.float32))

    subprocess.run(
        [
            YTIDV_VENV_PYTHON, WORKER_SCRIPT, "render_with_depth",
            "--request", req_path, "--depth-npy", depth_path,
            "--mode", RENDERER_MODE, "--out-npy", out_path,
        ],
        check=True, cwd=HERE,
    )
    rgba = np.load(out_path)
    return external_render.RenderResult(rgba=rgba, alpha_mode="PREMULTIPLIED", color_space="LINEAR")


class _StubPrefs:
    renderer_mode = MODE_KEY_BY_NAME[RENDERER_MODE]
    work_dir = WORK_DIR
    verbose_logging = True


SPHERE_RADIUS = 0.8
# Distance of the sphere's center past the box's far boundary, along the view.
SPHERE_MARGIN_PAST_BOX_EXIT = 0.6
# Sideways offset so part of the sphere sticks out past the box on screen;
# too small and it is fully hidden.
SPHERE_SIDEWAYS_OFFSET = 2.2

CAMERA_LOCATION = (-3.4, -4.0, 2.6)


def build_scene():
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Flagged volume box spanning -1..1; its bounds map to yt_idv's domain.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    box = bpy.context.active_object
    box.name = "FlaggedVolumeBox"
    box.volume_override.is_volume_override = True

    box_center = mathutils.Vector((0.0, 0.0, 0.0))
    cam_loc = mathutils.Vector(CAMERA_LOCATION)
    view_dir = (box_center - cam_loc).normalized()  # camera -> box center
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    sideways = view_dir.cross(world_up).normalized()

    # Ray parameter where the view ray from the center exits the unit box.
    box_exit_t = min(1.0 / abs(c) for c in view_dir)

    sphere_center = (
        box_center
        + view_dir * (box_exit_t + SPHERE_MARGIN_PAST_BOX_EXIT)
        + sideways * SPHERE_SIDEWAYS_OFFSET
    )

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=SPHERE_RADIUS, location=sphere_center, segments=64, ring_count=32,
    )
    sphere = bpy.context.active_object
    sphere.name = "PartiallyHiddenSphere"

    mat = bpy.data.materials.new("PartiallyHiddenSphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.15, 0.05, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35
    sphere.data.materials.append(mat)

    # Camera aimed at the box center.
    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    cam = bpy.context.active_object
    cam.name = "DemoCamera"
    target = mathutils.Vector((0.0, 0.0, 0.0))
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
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = os.path.join(WORK_DIR, "final_output_")

    # Box's matrix_world is identity, so these are already world-space.
    return box, sphere, cam, tuple(sphere_center), sideways, view_dir


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


def _world_to_pixel(scene, cam, world_point, width, height):
    """Project a world point to (row, col), row 0 = bottom like bpy image pixels."""
    import bpy_extras

    ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam, mathutils.Vector(world_point))
    col = int(ndc.x * width)
    row = int(ndc.y * height)  # NDC y=0 is the bottom row
    return max(0, min(height - 1, row)), max(0, min(width - 1, col))


def _verify_and_export(scene, cam, sphere_center_world, sideways, view_dir, sphere_radius, work_dir):
    """Read pass 2's final EXR, check alpha and partial occlusion, write PNGs.

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
    print(f"[render_partial_occlusion_demo] background corner pixel alpha = {corner_alpha:.4f} "
          f"(expect ~0.0 -- see module docstring point 3)")
    if corner_alpha > 0.02:
        print("[render_partial_occlusion_demo] WARNING: background does not look transparent -- "
              "film_transparent or the yt_idv layer's own alpha may not be wired correctly")

    # --- Partial-occlusion check against the beauty pass ---
    beauty_path = os.path.join(work_dir, f"vo_beauty_{scene.frame_current:04d}.exr")
    beauty_img = bpy.data.images.load(beauty_path, check_existing=False)
    try:
        beauty = np.array(beauty_img.pixels[:], dtype=np.float32).reshape(h, w, beauty_img.channels)
    finally:
        bpy.data.images.remove(beauty_img)

    # Two camera-facing points on the sphere: one toward the box (occluded)
    # and one away from it (unoccluded).
    center = mathutils.Vector(sphere_center_world)
    facing = -view_dir
    toward_box_dir = (facing * 0.4 + (-sideways) * 0.9).normalized()
    away_from_box_dir = (facing * 0.4 + sideways * 0.9).normalized()
    behind_point = center + toward_box_dir * sphere_radius
    peek_point = center + away_from_box_dir * sphere_radius

    behind_row, behind_col = _world_to_pixel(scene, cam, behind_point, w, h)
    peek_row, peek_col = _world_to_pixel(scene, cam, peek_point, w, h)

    print(f"[render_partial_occlusion_demo] 'behind the box' pixel (row={behind_row}, col={behind_col}): "
          f"beauty(sphere-only)={beauty[behind_row, behind_col, :3]} "
          f"final(composited)={pixels[behind_row, behind_col, :3]}")
    print(f"[render_partial_occlusion_demo] 'peeking out beside the box' pixel (row={peek_row}, col={peek_col}): "
          f"beauty(sphere-only)={beauty[peek_row, peek_col, :3]} "
          f"final(composited)={pixels[peek_row, peek_col, :3]}")

    behind_diff = float(np.linalg.norm(pixels[behind_row, behind_col, :3] - beauty[behind_row, behind_col, :3]))
    peek_diff = float(np.linalg.norm(pixels[peek_row, peek_col, :3] - beauty[peek_row, peek_col, :3]))
    print(f"[render_partial_occlusion_demo] |final - beauty| at 'behind the box' = {behind_diff:.4f} "
          f"(expect LARGE -- volume should visibly draw over the sphere here)")
    print(f"[render_partial_occlusion_demo] |final - beauty| at 'peeking out' = {peek_diff:.4f} "
          f"(expect SMALL -- this part of the sphere should render close to normal)")
    if not (behind_diff > 0.05 and behind_diff > peek_diff * 2.0):
        print("[render_partial_occlusion_demo] WARNING: partial-occlusion claim not clearly "
              "confirmed by these two sample pixels -- geometry may need retuning (see "
              "SPHERE_SIDEWAYS_OFFSET/SPHERE_MARGIN_PAST_BOX_EXIT)")

    # Plain display-ready PNG: sRGB-encode the premultiplied RGB, keep alpha.
    display = pixels.copy()
    display[..., :3] = _srgb_encode(display[..., :3])
    plain_path = os.path.join(IMAGES_DIR, OUTPUT_IMAGE_NAMES[RENDERER_MODE][0])
    _write_png(plain_path, display)
    print(f"[render_partial_occlusion_demo] wrote {plain_path}")

    # Premultiplied alpha-over onto a checkerboard, to show transparency.
    square = 32
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // square) + (xx // square)) % 2).astype(np.float32)
    checker_rgb = 0.25 + 0.35 * checker[..., None]  # two mid-grays
    composited_rgb = display[..., :3] + checker_rgb * (1.0 - display[..., 3:4])
    composited = np.concatenate([composited_rgb, np.ones_like(display[..., 3:4])], axis=-1)
    checker_path = os.path.join(IMAGES_DIR, OUTPUT_IMAGE_NAMES[RENDERER_MODE][1])
    _write_png(checker_path, composited)
    print(f"[render_partial_occlusion_demo] wrote {checker_path}")


def main():
    properties.register()
    operators.RENDERER_MODES[MODE_KEY_BY_NAME[RENDERER_MODE]] = _bridge_render
    operators._get_prefs = lambda context: _StubPrefs()

    box, sphere, cam, sphere_center_world, sideways, view_dir = build_scene()
    print(f"[render_partial_occlusion_demo] scene built: box={box.name} sphere={sphere.name} "
          f"cam={cam.name} mode={RENDERER_MODE} sphere_center_world={sphere_center_world}")

    operators.render_current_frame(bpy.context)
    print(f"[render_partial_occlusion_demo] real two-pass pipeline finished; "
          f"final EXR at {bpy.context.scene.render.frame_path()!r}")

    _verify_and_export(bpy.context.scene, cam, sphere_center_world, sideways, view_dir, SPHERE_RADIUS, WORK_DIR)


if __name__ == "__main__":
    main()
