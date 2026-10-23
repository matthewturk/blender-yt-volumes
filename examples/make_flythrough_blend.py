"""Turn flythrough_densest.py's camera path into a .blend and a render_volume config.

flythrough_densest.py, a yt_idv flythrough planner, sits next to this script.

    python examples/make_flythrough_blend.py SCENE.zip OUT_PREFIX [flythrough options]

Writes OUT_PREFIX.blend and OUT_PREFIX.toml; render with

    python -m yt_volumes.render_volume OUT_PREFIX.toml

The path, target and colormap plan come from flythrough_densest.py's own
functions, run once here against the saved scene. In Blender the volume box is
a 2x2x2 Empty, so world = 2 * box - 1 + its location. Two ways to place the
camera:

- default: the box sits at the origin and the camera follows the path
  literally until it is --floor box widths from the target; past that it holds
  the floor distance and the rest of the approach becomes the volume's zoom
  level.
- --fly: the box is offset so the target is the world origin, where float32
  keeps its relative precision, and the camera follows the whole path
  literally. The worker rescales yt_idv to the camera's distance each frame.
  Cycles objects near the target stay registered with the data at any depth,
  but Blender clips them closer than 1e-6 world units to the camera.
- --deep-zoom: a Deep Zoom extension scene, for editing in Blender. One shot
  anchored at the target, the camera keyed along the path's direction at
  --radius world units, and the approach in the zoom curve, so the images are
  the same. The volume's Dataset Path is the scene, so F12 works in the GUI;
  --grids adds the grid map.
"""

import argparse
import importlib.util
import math
import os
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

RENDER_MODES = {"max_intensity": "YT_IDV_MAX_INTENSITY", "projection": "YT_IDV_PROJECTION"}


def _load_flythrough(path):
    if not os.path.isfile(path):
        raise SystemExit(
            f"no flythrough_densest.py at {path}: pass its location with --flythrough"
        )
    spec = importlib.util.spec_from_file_location("flythrough_densest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("scene")
    p.add_argument("out", help="output prefix for the .blend and .toml")
    p.add_argument("--flythrough", default=os.path.join(here, "flythrough_densest.py"))
    p.add_argument("--distance", type=float, default=0.2)
    p.add_argument("--orbit-distance", type=float)
    p.add_argument("--deep-distance", type=float)
    p.add_argument("--deep-frames-per-decade", type=float, default=692.0)
    p.add_argument("--deep-ease-frames", type=int, default=90)
    p.add_argument("--zoom-frames", type=int, default=400)
    p.add_argument("--orbit-frames", type=int, default=600)
    p.add_argument("--orbit-degrees", type=float, default=360.0)
    p.add_argument("--orbit-ease-frames", type=int, default=90)
    p.add_argument("--near-fraction", type=float, default=0.01)
    p.add_argument("--near-min", type=float, default=1e-6)
    p.add_argument("--recenter-factor", type=float, default=10.0)
    p.add_argument(
        "--floor",
        type=float,
        help="closest the Blender camera gets, in box widths; default --distance",
    )
    p.add_argument("--fly", action="store_true", help="fly the camera all the way in; see above")
    p.add_argument("--deep-zoom", action="store_true", help="write a Deep Zoom extension scene; see above")
    p.add_argument("--radius", type=float, default=3.0, help="--deep-zoom: the camera's distance from the anchor, world units")
    p.add_argument("--grids", help="--deep-zoom: grid structure .npz for the grid map")
    p.add_argument("--target", type=float, nargs=3, help="in the scaled box; default the densest cell")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--cmap-decades", type=float, default=6.0)
    p.add_argument("--cmap-percentile", type=float, default=99.9)
    p.add_argument("--cmap-sample-every", type=int, default=5)
    p.add_argument("--cmap-smooth-frames", type=float, default=60.0)
    p.add_argument("--cmap-every", type=int, default=5, help="frames between colormap curve keys")
    return p.parse_args()


def plan(args, fly):
    """Path, target and colormap plan, computed with yt_idv offscreen.

    Returns (positions, distances, target_box, target_unitary, cmap_low,
    cmap_high, render_method, colormap); positions and targets are in the
    scaled box, [0, 1] across the data source's bounding box.
    """
    import yt
    import yt_idv
    import yt_idv.scene_graph

    yt.set_log_level(40)
    rc = yt_idv.render_context("egl", width=args.width, height=args.height)
    t = time.perf_counter()
    rc.scene = yt_idv.scene_graph.SceneGraph.load(args.scene)
    print(f"loaded {args.scene} in {time.perf_counter() - t:.1f} s", flush=True)
    component = rc.scene.components[0]
    data = component.data
    fly.scale_scene(data)
    if args.target is not None:
        target = np.array(args.target, dtype="f8")
    else:
        value, target = fly.densest_cell(data)
        print(f"densest cell (normalized value {value:.4g}) at {target}")
    target_unitary = target * data.applied_scale_ratio + np.asarray(data.applied_scale_offset)

    camera = rc.scene.camera
    camera.update(aspect_ratio=args.width / args.height)
    start = np.array(camera.position, dtype="f8")
    if args.deep_distance is not None:
        positions, distances = fly.deep_zoom_path(
            start, target, args.distance, args.zoom_frames, args.deep_distance,
            args.deep_frames_per_decade, args.deep_ease_frames, args.orbit_frames,
            args.orbit_degrees, ease_frames=args.orbit_ease_frames,
        )
        near_planes = fly.Recentering(
            rc.scene, distances, args.recenter_factor, args.near_fraction, args.near_min
        )
    else:
        positions = fly.camera_path(
            start, target, args.distance, args.zoom_frames, args.orbit_frames,
            args.orbit_degrees, orbit_distance=args.orbit_distance,
            ease_frames=args.orbit_ease_frames,
        )
        distances = np.linalg.norm(np.array(positions) - target, axis=1)
        near_planes = np.maximum(args.near_fraction * distances, args.near_min)

    rc.scene.render()
    low, high = fly.plan_cmap_bounds(
        rc, positions, target, args.cmap_sample_every, args.cmap_percentile,
        args.cmap_decades, args.cmap_smooth_frames, near_planes=near_planes,
    )
    return (
        np.array(positions), np.asarray(distances), target, target_unitary,
        low, high, component.render_method, component.colormap.colormap_name, data.applied_scale_ratio,
    )


def build_blend(path, positions, distances, target, floor, width, height, look, fly=False, fov_deg=45.0):
    """The scene: the volume Empty, a focus Empty at the target, and a camera
    keyed every frame. Returns the zoom level per frame.
    """
    import bpy
    import mathutils

    from yt_volumes import properties, render_volume

    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(positions)
    scene.render.engine = "CYCLES"  # yt_idv modes need it
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    volume.empty_display_size = 1.0
    scene.collection.objects.link(volume)
    render_volume.apply_render_settings(volume.volume_override, look)
    # world position of the target, and of the box's center
    origin = np.zeros(3) if fly else 2.0 * target - 1.0
    volume.location = mathutils.Vector(origin - (2.0 * target - 1.0))
    focus = bpy.data.objects.new("Focus", None)
    focus.location = mathutils.Vector(origin)
    scene.collection.objects.link(focus)

    camera_data = bpy.data.cameras.new("Camera")
    # yt_idv's fov is vertical
    camera_data.sensor_fit = "VERTICAL"
    camera_data.angle_y = math.radians(fov_deg)
    camera_data.clip_start = 1e-6 if fly else 2.0 * floor * 1e-3
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"  # world z stays up, as in the flythrough

    levels = []
    for k, (position, distance) in enumerate(zip(positions, distances)):
        held = distance if fly else max(distance, floor)
        levels.append(math.log10(held / distance) if held > distance else 0.0)
        # relative to the target first, so a target at the origin keeps the offset's precision
        camera.location = mathutils.Vector(origin + 2.0 * (position - target) * (held / distance))
        camera.keyframe_insert("location", frame=k + 1)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return levels


def build_deep_zoom_blend(path, args, positions, distances, target, target_unitary, box_ratio, look, fov_deg=45.0):
    """A Deep Zoom shot-space scene with the same images as the path."""
    import bpy
    import mathutils

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions"))
    import deep_zoom
    from deep_zoom import authoring, shot_space

    from yt_volumes import properties, render_volume

    bpy.ops.wm.read_factory_settings(use_empty=True)
    deep_zoom.register()
    properties.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(positions)
    scene.render.engine = "CYCLES"  # yt_idv modes need it
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    # in shot space the volume's transform plays no part
    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    scene.collection.objects.link(volume)
    volume.volume_override.is_volume_override = True
    volume.volume_override.dataset_path = os.path.abspath(args.scene)
    render_volume.apply_render_settings(volume.volume_override, look)

    settings = scene.deep_zoom
    settings.enabled = True
    if args.grids:
        settings.grid_file = os.path.abspath(args.grids)
    authoring.add_shot(scene, target_unitary, 1, "Flythrough")

    focus = bpy.data.objects.new("Focus", None)
    scene.collection.objects.link(focus)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.sensor_fit = "VERTICAL"  # yt_idv's fov is vertical
    camera_data.angle_y = math.radians(fov_deg)
    camera_data.clip_start = 0.01
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"  # world z stays up, as in the flythrough

    for k, (position, distance) in enumerate(zip(positions, distances)):
        # the path's distance, in unitary lengths, becomes the zoom
        settings.zoom = math.log10(args.radius / (settings.world_scale * distance * box_ratio))
        scene.keyframe_insert("deep_zoom.zoom", frame=k + 1)
        camera.location = mathutils.Vector((position - target) / distance * args.radius)
        camera.keyframe_insert("location", frame=k + 1)
    report = authoring.check_path(scene, camera)
    flagged = sum(1 for kinds in report.values() if kinds)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"deep zoom scene: zoom {authoring.zoom_at(scene, 1):.2f} to "
          f"{authoring.zoom_at(scene, len(positions)):.2f}, check_path flags {flagged} frames")


def _toml_curve(rows, indent="  "):
    return "[\n" + ",\n".join(f"{indent}[{', '.join(repr(float(v)) if i else str(int(v)) for i, v in enumerate(r))}]" for r in rows) + ",\n]"


def render_mode(render_method):
    mode = RENDER_MODES.get(render_method)
    if mode is None:
        raise SystemExit(f"no render_volume mode for render_method {render_method!r}")
    return mode


def write_config(path, args, blend, target_unitary, levels, low, high, render_method, colormap):
    n = len(low)
    keys = sorted(set(range(0, n, args.cmap_every)) | {n - 1})
    cmap_rows = [[k + 1, low[k], high[k]] for k in keys]
    if levels is None:
        # a Deep Zoom scene carries its own zoom
        extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions", "deep_zoom")
        extensions = f'\nextensions = ["{os.path.normpath(extensions_dir)}"]'
        zoom = ""
    else:
        extensions = ""
        zoom_rows = [[k + 1, levels[k]] for k in range(n)]
        zoom = f"\n[zoom]\ntarget = [{', '.join(repr(float(v)) for v in target_unitary)}]\nlevel = {_toml_curve(zoom_rows)}\n"
    mode = render_mode(render_method)
    with open(path, "w") as f:
        f.write(f"""# Written by examples/make_flythrough_blend.py from {os.path.basename(args.scene)}.
#   python -m yt_volumes.render_volume {os.path.basename(path)}

[data]
scene = "{os.path.abspath(args.scene)}"

[blender]
file = "{os.path.basename(blend)}"
volume = "Volume"{extensions}
frames = "1-{n}"
output = "{os.path.basename(path)[:-5]}_frames/frame_"
resolution = [{args.width}, {args.height}]
samples = {args.samples}
device = "OPTIX"

[render]
mode = "{mode}"
colormap = "{colormap}"
cmap_interpolation = "log"
cmap_range = {_toml_curve(cmap_rows)}
{zoom}""")


def main():
    args = parse_args()
    fly = _load_flythrough(args.flythrough)
    floor = args.floor if args.floor is not None else args.distance
    positions, distances, target, target_unitary, low, high, render_method, colormap, box_ratio = plan(args, fly)
    blend, config = f"{args.out}.blend", f"{args.out}.toml"
    look = {"mode": render_mode(render_method), "colormap": colormap}
    if args.deep_zoom:
        build_deep_zoom_blend(os.path.abspath(blend), args, positions, distances, target, target_unitary, box_ratio, look)
        levels = None
    else:
        levels = build_blend(
            os.path.abspath(blend), positions, distances, target, floor, args.width, args.height, look, fly=args.fly,
        )
    write_config(config, args, blend, target_unitary, levels, low, high, render_method, colormap)
    print(f"wrote {blend} and {config}: {len(positions)} frames, distance "
          f"{distances[0]:.3g} to {distances[-1]:.3g} box widths")


if __name__ == "__main__":
    sys.exit(main())
