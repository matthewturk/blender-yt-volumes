"""Camera-motion exploration: orbit a camera around a flagged cube over
several frames, render each frame with yt_idv via the subprocess bridge
(`ytidv_worker.py`; bpy and yt_idv can't share one Python in this sandbox),
and alpha-over a separately rendered marker sphere on top so alignment is
visible in one image per frame.

Run with this repo's dev venv (has bpy, not yt_idv):

    env -u LD_PRELOAD uv run python \
        scripts/yt_idv_exploration/blender_scene/render_frames_in_blender.py \
        /path/to/ytidv-venv/bin/python

The AMR grid-box wireframes map the dataset's [0, 1]^3 domain onto the
cube's bounding box for visual reference only. The composite is a plain
alpha-over (the sphere always wins where opaque), not the add-on's real
depth-aware compositing pipeline.
"""

import json
import math
import os
import subprocess
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
from cycles_yt import operators, properties  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "ytidv_worker.py")
OUTPUT_DIR = os.path.join(HERE, "output")

N_FRAMES = 6
ORBIT_RADIUS_BLENDER = 6.0
ORBIT_HEIGHT_BLENDER = 2.0
RESOLUTION = (320, 240)
MAX_GRID_BOXES = 15  # IsolatedGalaxy has 173 grids; keep the scene legible

SPHERE_RADIUS = 0.35  # small enough to only partly cover the volume
SPHERE_FRACTION_TOWARD_CAMERA = 0.45  # 0 = at cube center, 1 = at the camera
SPHERE_SIDE_OFFSET = 0.5  # offset along the camera's local +X, so it sits off-center
SPHERE_COLOR = (0.05, 0.9, 0.15, 1.0)

# Demo-only: these PNGs use row 0 = top, so TOP_LEFT_ORIGIN lines the two
# renders up. (The add-on itself uses BOTTOM_LEFT_ORIGIN for depth buffers.)
ROW_ORDER_FOR_THIS_DEMO = "TOP_LEFT_ORIGIN"


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.engine = "BLENDER_WORKBENCH"
    # Flat, unlit object colors, so SPHERE_COLOR renders as given.
    scene.display.shading.color_type = "OBJECT"

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = "FlaggedVolumeCube"
    # hide_render, not display_type='WIRE' (which only affects the viewport),
    # so the cube doesn't occlude the grid-box wireframes inside it.
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
    """Add the solid-colored marker sphere; repositioned every frame."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_RADIUS, location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "MarkerSphere"
    sphere.color = SPHERE_COLOR
    return sphere


def position_sphere_toward_camera(sphere, cube, cam_obj):
    """Place `sphere` between the cube and camera, offset to the camera's right."""
    toward_camera = cube.location.lerp(cam_obj.location, SPHERE_FRACTION_TOWARD_CAMERA)
    camera_right = cam_obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
    sphere.location = toward_camera + camera_right.normalized() * SPHERE_SIDE_OFFSET


def add_grid_box_wireframes(cube, grids_info):
    """One wireframe box per grid, mapped from [0, 1]^3 onto `cube`'s bounds.

    Uses mesh + Wireframe modifier, since Empties never appear in renders.
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
    """Render only the marker sphere on a transparent background."""
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
    """Straight-alpha "over" of the sphere render onto the yt_idv frame."""
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

        # Full-scene Blender view, for reference.
        blender_view_path = os.path.join(OUTPUT_DIR, f"blender_view_{frame_number:04d}.png")
        scene.render.filepath = blender_view_path
        bpy.ops.render.render(write_still=True)

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
