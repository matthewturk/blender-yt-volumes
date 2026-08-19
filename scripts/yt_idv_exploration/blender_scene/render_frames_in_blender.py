"""Blender-side half of the yt_idv camera-motion exploration: build a real
Blender scene (camera + a flagged cube + reference AMR grid-box wireframes
+ a bold marker sphere), orbit the camera across several frames, get a real
`yt_idv_max_intensity_renderer()` render via the subprocess bridge to the
throwaway yt_idv venv (`ytidv_worker.py` -- see that file's docstring and
../README.md for why a subprocess bridge is needed here at all: bpy and
yt_idv can't share one Python in this sandbox), and COMPOSITE the sphere
(rendered separately, transparent background) on top of the yt_idv frame --
so there's one image per frame where you can SEE the sphere sitting in
front of real yt_idv content, not two disconnected PNGs you have to
mentally line up yourself. The grid-box wireframes turned out too thin to
read at a glance (see ../README.md); the sphere is the obvious, unmissable
marker requested instead.

Run with THIS REPO's OWN dev venv (has bpy, not yt_idv):

    env -u LD_PRELOAD uv run python \
        scripts/yt_idv_exploration/blender_scene/render_frames_in_blender.py \
        /path/to/ytidv-venv/bin/python

Uses `operators.build_camera_params()` -- the add-on's REAL function, not a
hand-rolled stand-in -- to build each frame's `CameraParams`.

Grid-box mapping (ARBITRARY, for visual reference only -- there is no real
dataset-to-object spatial registration yet, see yt_idv_renderer.py's module
docstring): the dataset's own [0, 1]^3 "unitary" domain is linearly mapped
onto the flagged cube's local bounding box.

The composite is ILLUSTRATIVE, not the real add-on's compositing pipeline:
it's a plain alpha-over of the sphere (Blender's own render, real alpha)
onto the yt_idv frame, done here with bpy image loads + numpy, so the
sphere always wins wherever it's opaque -- there's no real per-pixel depth
comparison against yt_idv content (which doesn't have its own per-pixel
depth to compare against -- see yt_idv_renderer.py's module docstring on
why this add-on's yt_idv integration doesn't depth-clip at all yet).
"""

import json
import math
import os
import subprocess
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
from cycles_volume_override import operators, properties  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "ytidv_worker.py")
OUTPUT_DIR = os.path.join(HERE, "output")

N_FRAMES = 6
ORBIT_RADIUS_BLENDER = 6.0
ORBIT_HEIGHT_BLENDER = 2.0
RESOLUTION = (320, 240)
MAX_GRID_BOXES = 15  # keep the scene legible -- IsolatedGalaxy has 173 grids total

SPHERE_RADIUS = 0.35  # deliberately SMALLER than the cube's own half-extent (1.0)
#                        -- an earlier 0.7 nearly matched the cube's whole
#                        projected width (see blender_view_0001.png) and
#                        swallowed the entire visible blob, making the
#                        composite look like a flat sticker with no gas
#                        left showing anywhere. Small enough here to only
#                        partly cover it, with SPHERE_SIDE_OFFSET below.
SPHERE_FRACTION_TOWARD_CAMERA = 0.45  # 0 = at cube center, 1 = at the camera
SPHERE_SIDE_OFFSET = 0.5  # shifted along the camera's own local +X ("right")
#                            so it sits off-center over the blob rather than
#                            dead-centered on top of it -- "partly in front
#                            of it", not "directly covering all of it"
SPHERE_COLOR = (0.05, 0.9, 0.15, 1.0)  # bold green -- unmistakable against the yt_idv render's reds/blues

# Demo-only choice, NOT a claim about the production add-on's row-order
# handling (that's separately tested in tests/test_yt_idv_renderer.py):
# these PNGs are written by yt.write_bitmap() and read back by
# bpy.data.images.load(), both of which use the ordinary "row 0 = top of
# image" PNG convention -- TOP_LEFT_ORIGIN is what makes the sphere render
# and the yt_idv render line up pixel-for-pixel when composited below.
# BOTTOM_LEFT_ORIGIN (what operators.read_depth_buffer actually wants) would
# vertically flip the yt_idv frame relative to Blender's own PNG here.
ROW_ORDER_FOR_THIS_DEMO = "TOP_LEFT_ORIGIN"


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.engine = "BLENDER_WORKBENCH"
    # Flat, unlit object colors -- makes SPHERE_COLOR render exactly as
    # specified regardless of the light rig, so it reads as an unmistakable
    # solid marker rather than a shaded gray blob.
    scene.display.shading.color_type = "OBJECT"

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = "FlaggedVolumeCube"
    # hide_render (NOT display_type='WIRE' -- tried that first, confirmed it
    # does NOT affect bpy.ops.render.render() output for Workbench in this
    # Blender, only the viewport, despite looking like a render-affecting
    # per-object property) so it doesn't visually occlude the grid-box
    # wireframes nested inside it. Also matches the real add-on's own
    # semantics: a flagged object gets hidden from the main render pass --
    # see scene_sync.hidden_from_main_render.
    cube.hide_render = True
    properties.register()
    cube.volume_override.is_volume_override = True
    cube.volume_override.dataset_kind = "yt"
    cube.volume_override.dataset_path = "yt.load_sample('IsolatedGalaxy')"  # documentation only, not wired up

    bpy.ops.object.light_add(type="SUN", location=(4, -4, 6))

    cam_data = bpy.data.cameras.new("OrbitCam")
    cam_data.lens = 35.0
    cam_obj = bpy.data.objects.new("OrbitCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    return scene, cube, cam_obj


def add_marker_sphere():
    """A bold, solid, unmistakably-colored sphere -- the "does this
    actually line up with yt_idv content" marker requested after the
    AMR grid-box wireframes turned out too thin to read at a glance. Its
    position gets updated every frame in the main loop (see
    `position_sphere_toward_camera`) to sit partway between the cube's
    center and the camera, so it visibly overlaps the yt_idv content in
    every frame regardless of orbit angle.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "MarkerSphere"
    sphere.color = SPHERE_COLOR
    return sphere


def position_sphere_toward_camera(sphere, cube, cam_obj):
    """Move `sphere` to `SPHERE_FRACTION_TOWARD_CAMERA` of the way from the
    cube's center to the camera (so it sits between the camera and the
    yt_idv content, whose apparent center is roughly the cube's own
    location -- see yt_idv_renderer.py's module docstring on the orbit
    mapping), then nudge it sideways along the camera's own local +X ("right")
    axis by `SPHERE_SIDE_OFFSET` -- so from the camera's point of view it
    reads as PARTLY in front of the blob, not squarely on top of all of it.
    Using the camera's OWN right axis (not a fixed world direction) keeps
    the offset visually consistent -- always "to the right of center" from
    the camera's perspective -- as it orbits.
    """
    toward_camera = cube.location.lerp(cam_obj.location, SPHERE_FRACTION_TOWARD_CAMERA)
    camera_right = cam_obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
    sphere.location = toward_camera + camera_right.normalized() * SPHERE_SIDE_OFFSET


def add_grid_box_wireframes(cube, grids_info):
    """One wireframe box per grid, mapped from the dataset's unitary
    [0, 1]^3 domain onto `cube`'s own local bounding box. Purely a visual
    reference -- see module docstring.

    Deliberately mesh + Wireframe modifier, NOT an Empty: Empties
    (`empty_display_type='CUBE'`) are viewport-only gizmos with no
    geometry -- they never appear in an actual render (Cycles, Workbench,
    doesn't matter), confirmed the hard way (first attempt used Empties;
    `blender_view_*.png` showed only the plain flagged cube, no grid boxes
    at all, in every frame). A cube mesh with a Wireframe modifier turns its
    edges into real renderable geometry instead -- though even this turned
    out subtle enough at normal viewing size that a bolder marker (the
    sphere above) was requested to actually confirm alignment at a glance.
    """
    half_extent = max(cube.dimensions) / 2.0
    center = cube.location.copy()

    def to_blender(unitary_xyz):
        return [
            center[i] + (unitary_xyz[i] - 0.5) * 2.0 * half_extent
            for i in range(3)
        ]

    created = []
    for i, grid in enumerate(grids_info["grids"][:MAX_GRID_BOXES]):
        lo = to_blender(grid["left_edge"])
        hi = to_blender(grid["right_edge"])
        box_center = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
        box_size = max((hi[k] - lo[k]) for k in range(3)) or 0.01

        bpy.ops.mesh.primitive_cube_add(size=box_size, location=box_center)
        box = bpy.context.active_object
        box.name = f"AMRGrid_L{grid['level']}_{i:03d}"
        wire_mod = box.modifiers.new(name="Wireframe", type="WIREFRAME")
        wire_mod.thickness = max(0.01, box_size * 0.01)
        created.append(box)
    print(f"added {len(created)} grid-box wireframes (of {len(grids_info['grids'])} fetched, "
          f"{grids_info.get('total_grids', '?')} total in dataset)")
    return created


def orbit_camera_to_frame(cam_obj, cube, frame_index, n_frames):
    angle = 2 * math.pi * frame_index / n_frames
    x = cube.location.x + ORBIT_RADIUS_BLENDER * math.cos(angle)
    y = cube.location.y + ORBIT_RADIUS_BLENDER * math.sin(angle)
    z = cube.location.z + ORBIT_HEIGHT_BLENDER
    cam_obj.location = (x, y, z)

    direction = (cube.location - cam_obj.location)
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_sphere_only_transparent(scene, sphere, everything_else, path):
    """A second render pass: JUST the marker sphere, transparent background
    -- used as the foreground layer for `composite_with_yt_idv_frame`. Not
    how the real add-on composites (see module docstring); this is a plain
    illustrative alpha-over for this demo only.
    """
    prev_hide = {obj.name: obj.hide_render for obj in everything_else}
    prev_transparent = scene.render.film_transparent
    prev_color_mode = scene.render.image_settings.color_mode
    try:
        for obj in everything_else:
            obj.hide_render = True
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
    finally:
        for obj in everything_else:
            obj.hide_render = prev_hide[obj.name]
        scene.render.film_transparent = prev_transparent
        scene.render.image_settings.color_mode = prev_color_mode


def _load_png_as_array(path):
    import numpy as np

    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
    finally:
        bpy.data.images.remove(img)
    return arr


def _save_array_as_png(array, path):
    h, w = array.shape[:2]
    img = bpy.data.images.new("composite_tmp", width=w, height=h, alpha=True)
    img.pixels.foreach_set(array.astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def composite_with_yt_idv_frame(yt_idv_frame_path, sphere_only_path, out_path):
    """Plain straight-alpha "over": sphere (opaque where it is, transparent
    everywhere else) drawn on top of the yt_idv frame. See module
    docstring's "ILLUSTRATIVE, not the real add-on's compositing pipeline".
    """
    import numpy as np

    background = _load_png_as_array(yt_idv_frame_path)
    foreground = _load_png_as_array(sphere_only_path)
    fg_rgb, fg_a = foreground[..., :3], foreground[..., 3:4]
    out_rgb = fg_rgb * fg_a + background[..., :3] * (1.0 - fg_a)
    out_a = fg_a[..., 0] + background[..., 3] * (1.0 - fg_a[..., 0])
    out = np.concatenate([out_rgb, out_a[..., None]], axis=-1)
    _save_array_as_png(out, out_path)


def camera_params_to_json(camera_params, row_order, resolution_x, resolution_y, frame_number):
    return {
        "camera": {
            "matrix_world": camera_params.matrix_world,
            "projection": camera_params.projection,
            "lens_mm": camera_params.lens_mm,
            "sensor_width_mm": camera_params.sensor_width_mm,
            "sensor_height_mm": camera_params.sensor_height_mm,
            "sensor_fit": camera_params.sensor_fit,
            "shift_x": camera_params.shift_x,
            "shift_y": camera_params.shift_y,
            "ortho_scale": camera_params.ortho_scale,
            "clip_start": camera_params.clip_start,
            "clip_end": camera_params.clip_end,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "pixel_aspect_x": camera_params.pixel_aspect_x,
            "pixel_aspect_y": camera_params.pixel_aspect_y,
        },
        "row_order": row_order,
        "frame_number": frame_number,
    }


def main(ytidv_python):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene, cube, cam_obj = build_scene()
    sphere = add_marker_sphere()

    grids_path = os.path.join(OUTPUT_DIR, "grids.json")
    subprocess.run(
        [ytidv_python, WORKER, "describe_grids", "--out", grids_path, "--max-grids", str(MAX_GRID_BOXES)],
        check=True,
    )
    with open(grids_path) as f:
        grids_info = json.load(f)
    grid_boxes = add_grid_box_wireframes(cube, grids_info)

    requests = []
    for i in range(N_FRAMES):
        frame_number = i + 1
        scene.frame_set(frame_number)
        orbit_camera_to_frame(cam_obj, cube, i, N_FRAMES)
        position_sphere_toward_camera(sphere, cube, cam_obj)
        bpy.context.view_layer.update()

        camera_params = operators.build_camera_params(scene, cam_obj)
        requests.append(camera_params_to_json(
            camera_params, row_order=ROW_ORDER_FOR_THIS_DEMO,
            resolution_x=RESOLUTION[0], resolution_y=RESOLUTION[1],
            frame_number=frame_number,
        ))

        # Full-scene Blender view (opaque background, grid boxes + sphere
        # visible) for reference.
        blender_view_path = os.path.join(OUTPUT_DIR, f"blender_view_{frame_number:04d}.png")
        scene.render.filepath = blender_view_path
        bpy.ops.render.render(write_still=True)

        # Sphere-only, transparent background -- the foreground layer for
        # the composite below.
        sphere_only_path = os.path.join(OUTPUT_DIR, f"sphere_only_{frame_number:04d}.png")
        render_sphere_only_transparent(scene, sphere, [cube] + grid_boxes, sphere_only_path)

        print(f"frame {frame_number}: camera at {tuple(round(c, 2) for c in cam_obj.location)}, "
              f"sphere at {tuple(round(c, 2) for c in sphere.location)}")

    requests_path = os.path.join(OUTPUT_DIR, "requests.json")
    with open(requests_path, "w") as f:
        json.dump(requests, f, indent=2)

    subprocess.run(
        [ytidv_python, WORKER, "render_many", "--requests", requests_path, "--out-dir", OUTPUT_DIR],
        check=True,
    )

    for i in range(N_FRAMES):
        frame_number = i + 1
        yt_idv_frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number:04d}.png")
        sphere_only_path = os.path.join(OUTPUT_DIR, f"sphere_only_{frame_number:04d}.png")
        composite_path = os.path.join(OUTPUT_DIR, f"composite_{frame_number:04d}.png")
        composite_with_yt_idv_frame(yt_idv_frame_path, sphere_only_path, composite_path)
        print(f"frame {frame_number}: composite -> {composite_path}")

    blend_path = os.path.join(OUTPUT_DIR, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"saved scene to {blend_path}")
    print(f"all output in {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} /path/to/ytidv-venv/bin/python", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
