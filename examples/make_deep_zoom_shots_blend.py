"""Author a two-shot dive into DeeplyNestedZoom with the Deep Zoom extension.

    python examples/make_deep_zoom_shots_blend.py OUT.blend GRIDS.npz

GRIDS.npz comes from extensions/deep_zoom/grid_structure.py. The script does
what an author would in the Deep Zoom panel: anchor a first shot at a coarse
guess and dive 5 decades; spot where the finer grids cluster on the grid map
and re-anchor there; glide the orbit onto the new anchor and dive 7 more. Then
it checks the path and exports it next to OUT.blend.
"""

import argparse
import math
import os
import sys

import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions"))
import deep_zoom  # noqa: E402
from deep_zoom import authoring, grid_structure, shot_space  # noqa: E402

FRAMES = 240
CUT = 100
RADIUS = 3.0


def _orbit(frame, center):
    angle = math.radians(30.0 + 300.0 * (frame - 1) / (FRAMES - 1))
    return center + RADIUS * np.array([math.cos(angle) * math.cos(0.35), math.sin(angle) * math.cos(0.35), math.sin(0.35)])


def _key(obj, frame, location):
    obj.location = mathutils.Vector(location)
    obj.keyframe_insert("location", frame=frame)


def build(path, grids_path, grid_map=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    deep_zoom.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    # the flagged volume; in shot space its transform plays no part
    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    scene.collection.objects.link(volume)

    settings = scene.deep_zoom
    settings.enabled = True
    settings.grid_file = os.path.abspath(grids_path)
    settings.map_max_size = 8.0
    settings.map_max_distance = 30.0
    settings.map_thickness = 0.006
    settings.show_grid_map = grid_map

    # where the finest grids are (an author would find this on the grid map)
    grids = grid_structure.load(grids_path)
    deepest = np.argmax(grids["level"])
    target = (grids["left"][deepest] + grids["right"][deepest]) / 2

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = 35.0
    camera.data.clip_start = 0.01
    scene.collection.objects.link(camera)
    scene.camera = camera
    focus = bpy.data.objects.new("Focus", None)
    scene.collection.objects.link(focus)
    track = camera.constraints.new("TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # shot 1: a coarse guess at the peak, and a 5-decade dive
    authoring.add_shot(scene, np.round(target, 5), 1, "Approach")
    authoring.add_dive(scene, 1, CUT - 1, 5.0)
    for frame in range(1, CUT, 3):
        _key(camera, frame, _orbit(frame, np.zeros(3)))
    _key(focus, 1, (0.0, 0.0, 0.0))

    # shot 2: the finer grids cluster about a unit off-centre; re-anchor there
    scene.frame_set(CUT)
    cluster = shot_space.to_world(target, shot_space.parse_anchor(scene.deep_zoom.shots[0].anchor), authoring.zoom_at(scene, CUT))
    authoring.add_shot_at(scene, cluster, CUT, camera, "Core")
    # glide the orbit and the aim onto the new anchor, then dive 7 more decades
    for frame in range(CUT, FRAMES + 1, 3):
        glide = min(1.0, (frame - CUT) / 40.0)
        glide = glide * glide * (3.0 - 2.0 * glide)
        center = -cluster * (1.0 - glide)
        _key(camera, frame, _orbit(frame, center))
        _key(focus, frame, center)
    authoring.add_dive(scene, CUT, FRAMES - CUT, 7.0)

    report = authoring.check_path(scene, camera)
    flagged = sorted(f for f, kinds in report.items() if kinds)
    print(f"check_path: {len(flagged)} frames need attention" + (f" (first {flagged[0]})" if flagged else ""))
    authoring.export_camera_path(scene, os.path.splitext(path)[0] + "_path.json")
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    print(f"wrote {path}: shots {[s.name for s in settings.shots]}, target {shot_space.format_anchor(target)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("grids")
    parser.add_argument("--no-grid-map", action="store_true", help="leave the grid map out of the render")
    args = parser.parse_args()
    build(args.out, args.grids, grid_map=not args.no_grid_map)


if __name__ == "__main__":
    main()
