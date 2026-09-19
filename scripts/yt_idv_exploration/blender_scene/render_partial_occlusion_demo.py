"""Real, end-to-end demonstration of the OPPOSITE compositing direction from
`render_real_demo.py`: there, a solid sphere occluded the volume (via the
external depth clip stopping yt_idv's ray march at the sphere's own Cycles
Z-buffer); here, the VOLUME occludes part of a solid sphere instead -- no
depth-clip involved at all for this direction, just the add-on's existing
Alpha-Over compositing (`compositor_build.build_or_update_graph`) drawing
the (real, per-pixel-alpha) volume layer OVER the Cycles beauty pass. That
mechanism was already fully general -- it draws the volume result over
whatever Cycles rendered underneath using the volume layer's own alpha,
unconditionally -- so demonstrating this direction needed a scene geometry
change, not a code change: a sphere positioned so that real, bright volume
content sits BETWEEN the camera and part of the sphere's own surface.

Geometry -- first attempt and why it didn't work, then the fix, both worth
recording:

First attempt: center the sphere on the dataset's own density PEAK
(`IsolatedGalaxy`'s peak sits at unitary (0.504, 0.500, 0.500) -- basically
the domain's own center, confirmed via `ds.all_data().quantities.
max_location("density")`), shifted slightly past it along the camera's view
axis so the peak would sit just in front of the sphere. Result: the volume
occluded the ENTIRE sphere, not just the part near the peak -- this
project's own already-documented finding that the current colormap gives
even very LOW densities a nonzero opacity "floor" (not true transparency)
turns out to mean the WHOLE box interior reads as close to fully opaque once
a ray has traveled even a modest distance into the domain, not just near
the peak. Any sphere placed meaningfully INSIDE the box's interior sits
behind enough accumulated "floor" opacity to be hidden completely --
there's no shallow-vs-deep gradient to exploit for a partial effect this
way.

The fix: this add-on's own compositing doesn't care about 3D "inside the
box" at all -- Cycles never renders the box (it's hidden, flagged for
override), so the ONLY thing that determines whether the volume covers a
given sphere pixel is whether that SAME screen pixel also has meaningful
alpha in the volume's own 2D render. So instead of relying on the box's
interior opacity gradient, this positions the sphere BEHIND the box
relative to the camera (well past the box's own far boundary along the
camera's view direction) and OFFSET SIDEWAYS, so its silhouette partially
overlaps the box's screen footprint (occluded, hidden under real volume
color) while the rest sticks out past the box's silhouette on one side
(fully visible, zero volume alpha there since nothing of the domain
projects to those pixels at all) -- a ball peeking out from behind an
opaque-looking curtain, not a ball buried inside translucent haze.

Run with REAL Blender itself, not the pip bpy wheel used for this repo's own
tests -- this needs a genuine Cycles render:

    /data/SteamLibrary/steamapps/common/Blender/blender --background \\
        --python scripts/yt_idv_exploration/blender_scene/render_partial_occlusion_demo.py

Same subprocess-bridge substitution as `render_real_demo.py` (yt_idv still
cannot be imported into Blender 5.3's own bundled Python 3.13 -- see that
script's docstring for why) -- everything else is the real, unmodified
`operators.render_current_frame()` pipeline against a real scene.

Verification, mirroring `render_real_demo.py`'s background-alpha checks:
`_verify_and_export` samples a pixel expected to be "sphere behind the box"
(should be hidden under real volume color) and a pixel expected to be
"sphere peeking out beside the box" (should render close to its real,
unoccluded Cycles color), and prints both the real Cycles beauty color and
the final composited color at each -- if the pipeline is doing what this
docstring claims, the "behind the box" pixel's final color should differ
substantially from its own beauty-only sphere color (the volume genuinely
drew over it), while the "peeking out" pixel's final color should stay
close to its beauty-only sphere color (that part of the sphere is NOT
wrongly occluded).
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
# max_intensity (not projection, unlike render_real_demo.py's default) --
# a single brightest-sample-per-ray gives the peak a harder, more clearly
# bounded blob than projection's additive/diffuse integration, which matters
# here since the blob's own SHAPE is what has to visibly stand out against
# the sphere, not just its rough presence.
MODE_KEY_BY_NAME = {
    "max_intensity": "YT_IDV_MAX_INTENSITY",
    "transfer_function": "YT_IDV_TRANSFER_FUNCTION",
    "projection": "YT_IDV_PROJECTION",
    "threshold_fog": "YT_IDV_THRESHOLD_FOG",
}
# (plain, checkerboard) output basenames per mode -- max_intensity keeps the
# original 28/29 numbering already referenced by the READMEs; other modes
# get their own numbers so re-running with a different mode doesn't clobber
# the reference images other docs point to.
OUTPUT_IMAGE_NAMES = {
    "max_intensity": ("28_partial_occlusion_sphere_and_volume.png", "29_partial_occlusion_over_checkerboard.png"),
    "threshold_fog": ("30_partial_occlusion_fog_sphere_and_volume.png", "31_partial_occlusion_fog_over_checkerboard.png"),
}

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)

from cycles_volume_override import external_render, operators, properties  # noqa: E402


def _bridge_render(request):
    """Drop-in replacement for a yt_idv_renderer.yt_idv_*_renderer function
    -- same RenderRequest -> RenderResult signature `operators.
    RENDERER_MODES` expects -- that shells out to the yt_idv/Python-3.12 venv
    instead of calling yt_idv in-process. See module docstring for why.
    """
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
# How far PAST the box's own far boundary (along the camera's straight-
# through view direction, from the box's center) the sphere's center sits --
# see module docstring's "Geometry" section for why "behind the box", not
# "inside" it, is what actually produces a partial (not total) occlusion.
SPHERE_MARGIN_PAST_BOX_EXIT = 0.6
# How far SIDEWAYS (perpendicular to the view direction) the sphere is
# offset from dead-center -- this is what lets part of its silhouette stick
# out past the box's own screen footprint. Tuned empirically (using
# `bpy_extras.object_utils.world_to_camera_view` to check the sphere's own
# projected screen bbox against the box's projected screen bbox -- box
# spans screen columns ~111-407 at 512x512 for this exact camera/box, see
# module docstring's "Geometry" section): 1.15 left the sphere's own column
# range (295-382) entirely INSIDE the box's (fully hidden, no peek-out at
# all); 2.2 puts it at columns ~356-473 -- ~51px genuinely overlapping the
# box (occluded) and ~66px past its edge (visible), a real partial split.
SPHERE_SIDEWAYS_OFFSET = 2.2

CAMERA_LOCATION = (-3.4, -4.0, 2.6)


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

    box_center = mathutils.Vector((0.0, 0.0, 0.0))
    cam_loc = mathutils.Vector(CAMERA_LOCATION)
    view_dir = (box_center - cam_loc).normalized()  # camera -> box center
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    sideways = view_dir.cross(world_up).normalized()

    # Where this ray exits the box (half-extent 1, axis-aligned, centered at
    # the origin) -- the box boundary is reached as soon as ANY axis
    # component of `box_center + t*view_dir` hits +-1, i.e. at the SMALLEST
    # such t across the three axes.
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

    # Camera: aimed at the box's own center (not a corner -- see
    # render_real_demo.py for that variant), so the box's silhouette and the
    # sphere peeking out from behind it are both framed together.
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

    # Box's matrix_world is identity (created at the origin, never
    # Box's matrix_world is identity (created at the origin, never
    # transformed), so `sphere_center`/`sideways` are already world-space --
    # returned as plain tuples/vectors rather than re-deriving them from the
    # box's transform, which would be needed if the box were ever moved.
    return box, sphere, cam, tuple(sphere_center), sideways, view_dir


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


def _world_to_pixel(scene, cam, world_point, width, height):
    """Project a world-space point to (row, col) pixel indices in the SAME
    BOTTOM_LEFT_ORIGIN convention `bpy.data.images.load(...).pixels` always
    uses (row 0 = bottom -- see `operators.DEPTH_ROW_ORDER`'s docstring for
    why that's this project's established convention), via Blender's own
    camera projection (not a hand-rolled one) -- used only to pick pixels to
    SAMPLE for the verification printout below, not anywhere in the actual
    render path.
    """
    import bpy_extras

    ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam, mathutils.Vector(world_point))
    col = int(ndc.x * width)
    row = int(ndc.y * height)  # NDC y=0 at bottom -- matches row 0 = bottom directly
    return max(0, min(height - 1, row)), max(0, min(width - 1, col))


def _verify_and_export(scene, cam, sphere_center_world, sideways, view_dir, sphere_radius, work_dir):
    """Read pass 2's real final EXR back from disk (NOT the in-memory
    "Render Result" datablock -- in `--background` mode its `.pixels`
    buffer reads back as size (0, 0) after `write_still=True`, a background-
    mode quirk unrelated to whether the render itself is correct; the file
    Pass 2 actually wrote to disk is the authoritative output), verify the
    background is genuinely transparent (not opaque black), write both a
    plain display-ready PNG and a checkerboard-composited PNG, AND check the
    actual partial-occlusion claim directly against real pixel data (not
    just "the picture looks right") -- see module docstring's "Verification"
    section.
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
    print(f"[render_partial_occlusion_demo] background corner pixel alpha = {corner_alpha:.4f} "
          f"(expect ~0.0 -- see module docstring point 3)")
    if corner_alpha > 0.02:
        print("[render_partial_occlusion_demo] WARNING: background does not look transparent -- "
              "film_transparent or the yt_idv layer's own alpha may not be wired correctly")

    # --- The actual partial-occlusion claim, checked against real pixel
    # data -- see module docstring's "Verification" section. ---
    beauty_path = os.path.join(work_dir, f"vo_beauty_{scene.frame_current:04d}.exr")
    beauty_img = bpy.data.images.load(beauty_path, check_existing=False)
    try:
        beauty = np.array(beauty_img.pixels[:], dtype=np.float32).reshape(h, w, beauty_img.channels)
    finally:
        bpy.data.images.remove(beauty_img)

    # Two points on the sphere's own surface (camera-facing hemisphere,
    # biased slightly toward the camera along -view_dir): one toward the
    # box (-sideways -- should land within the box's screen footprint,
    # hence occluded by real volume color) and one away from the box
    # (+sideways -- should land past the box's silhouette, hence
    # unoccluded).
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
