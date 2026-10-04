"""Author a two-shot deep dive with the Deep Zoom extension.

    python examples/make_deep_zoom_shots_blend.py OUT.blend GRIDS.npz [options]

GRIDS.npz comes from extensions/deep_zoom/grid_structure.py. The script does
what an author would in the Deep Zoom panel: anchor a first shot (by default a
coarse guess at the finest grid, rounded to 5 decimals) and dive; re-anchor on
the finest grid, which by then sits off-centre on the grid map; glide the
orbit onto the new anchor and dive on. Then it checks the path and exports it
next to OUT.blend. The defaults are the DeeplyNestedZoom dive; for PopII_prime:

    --dataset /data/PopII_prime/biggest_scene.zip --first-anchor bbox \\
    --zooms 1.3 8 9 --cut 150
"""

import argparse
import io
import json
import math
import os
import sys
import zipfile

import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions"))
import deep_zoom  # noqa: E402
from deep_zoom import authoring, grid_structure, shot_space  # noqa: E402

from cycles_volume_override import properties  # noqa: E402

RADIUS = 3.0


def _orbit(frame, frames, center):
    angle = math.radians(30.0 + 300.0 * (frame - 1) / (frames - 1))
    return center + RADIUS * np.array([math.cos(angle) * math.cos(0.35), math.sin(angle) * math.cos(0.35), math.sin(0.35)])


def _key(obj, frame, location):
    obj.location = mathutils.Vector(location)
    obj.keyframe_insert("location", frame=frame)


def _scene_bbox_center(path):
    """Center of a saved yt_idv scene's data-source bbox, unitary float64."""
    with zipfile.ZipFile(path) as archive:
        doc = json.loads(archive.read("scene.json"))
        attrs = next(d["attributes"] for d in doc["data_objects"] if "_bbox" in d.get("attributes", {}))
        return np.load(io.BytesIO(archive.read(attrs["_bbox"]["__array__"]))).mean(axis=0)


def build(path, grids_path, grid_map=True, dataset=None, first_anchor=None, zooms=(0.0, 5.0, 12.0),
          cut=100, frames=240):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    deep_zoom.register()
    properties.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.engine = "CYCLES"  # yt_idv modes need it
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
    volume.volume_override.is_volume_override = True
    if dataset:
        # for F12 in the GUI; render_volume configs name the dataset themselves
        volume.volume_override.dataset_path = os.path.abspath(dataset)

    settings = scene.deep_zoom
    settings.enabled = True
    settings.grid_file = os.path.abspath(grids_path)
    settings.map_thickness = 0.006
    settings.show_grid_map = grid_map

    # where the finest grids are (an author finds this with Anchor at Finest Grid)
    grids = grid_structure.load(grids_path)
    deepest = np.argmax(grids["level"])
    target = (grids["left"][deepest] + grids["right"][deepest]) / 2
    if first_anchor == "bbox":
        anchor = _scene_bbox_center(dataset)
    elif first_anchor:
        anchor = shot_space.parse_anchor(first_anchor)
    else:
        anchor = np.round(target, 5)

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

    zoom_start, zoom_cut, zoom_end = zooms
    authoring.add_shot(scene, anchor, 1, "Approach")
    settings.zoom = zoom_start
    authoring.add_dive(scene, 1, cut - 1, zoom_cut - zoom_start)
    for frame in range(1, cut, 3):
        _key(camera, frame, _orbit(frame, frames, np.zeros(3)))
    _key(focus, 1, (0.0, 0.0, 0.0))

    # re-anchor on the finest grid, wherever it sits on the grid map by now
    scene.frame_set(cut)
    point = shot_space.to_world(target, anchor, authoring.zoom_at(scene, cut))
    authoring.add_shot_at(scene, point, cut, camera, "Core")
    # glide the orbit and the aim onto the new anchor, then dive on
    for frame in range(cut, frames + 1, 3):
        glide = min(1.0, (frame - cut) / 40.0)
        glide = glide * glide * (3.0 - 2.0 * glide)
        center = -point * (1.0 - glide)
        _key(camera, frame, _orbit(frame, frames, center))
        _key(focus, frame, center)
    authoring.add_dive(scene, cut, frames - cut, zoom_end - zoom_cut)

    report = authoring.check_path(scene, camera)
    flagged = sorted(f for f, kinds in report.items() if kinds)
    print(f"check_path: {len(flagged)} frames need attention" + (f" (first {flagged[0]})" if flagged else ""))
    authoring.export_camera_path(scene, os.path.splitext(path)[0] + "_path.json")
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    print(f"wrote {path}: shots {[s.name for s in settings.shots]}, first anchor "
          f"{shot_space.format_anchor(anchor)}, target {shot_space.format_anchor(target)}, "
          f"re-anchor offset at the cut {np.linalg.norm(point):.3g} world units")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("grids")
    parser.add_argument("--no-grid-map", action="store_true", help="leave the grid map out of the render")
    parser.add_argument("--dataset", help="set as the volume's Dataset Path, for F12 in the GUI")
    parser.add_argument("--first-anchor", help="'x, y, z', or 'bbox' for a saved scene's bbox center; "
                        "default the finest grid rounded to 5 decimals")
    parser.add_argument("--zooms", type=float, nargs=3, default=(0.0, 5.0, 12.0), metavar=("START", "CUT", "END"))
    parser.add_argument("--cut", type=int, default=100, help="frame of the re-anchor")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()
    build(args.out, args.grids, grid_map=not args.no_grid_map, dataset=args.dataset,
          first_anchor=args.first_anchor, zooms=tuple(args.zooms), cut=args.cut, frames=args.frames)


if __name__ == "__main__":
    main()
