"""Points of interest that follow the zoom.

A points object is a mesh whose vertices Deep Zoom moves into shot space every
frame, from float64 unitary positions kept on the object (custom properties are
doubles), so they stay exact at any depth. Read them by index in Geometry Nodes
(Object Info, Sample Index), by name through each point's vertex group (Copy
Location, Child Of, Damped Track), or with `world_positions`. Every point
carries `world_per_unitary`, the current scale, for sizing things by it.
"""

import os

import bpy
import numpy as np

from . import shot_space

UNITARY_KEY = "deep_zoom_unitary"  # flat float64 x, y, z per point
NAMES_KEY = "deep_zoom_names"
SCALE_ATTRIBUTE = "world_per_unitary"


def is_points(obj):
    return obj.type == "MESH" and UNITARY_KEY in obj


def unitary_positions(obj):
    """`(n, 3)` float64 unitary positions of a points object."""
    return np.array(obj[UNITARY_KEY], dtype=np.float64).reshape(-1, 3)


def names(obj):
    return list(obj.get(NAMES_KEY, []))


def _anchor(scene):
    from . import authoring

    shot = authoring.active_shot(scene) if scene.deep_zoom.enabled else None
    anchor = None if shot is None else shot_space.parse_anchor(shot.anchor)
    return None if anchor is None or isinstance(anchor, str) else anchor


def world_positions(obj, scene=None):
    """`(n, 3)` float64 shot-space positions of a points object now, or None
    when no shot with explicit coordinates is playing.
    """
    scene = scene or bpy.context.scene
    anchor = _anchor(scene)
    if anchor is None:
        return None
    settings = scene.deep_zoom
    return shot_space.to_world(unitary_positions(obj), anchor, settings.zoom, settings.world_scale)


def create_points(scene, unitary, name="DeepZoomPoints", point_names=None, attributes=None):
    """A points object at float64 `unitary` `(n, 3)` positions, linked to
    `scene`. `point_names` name each point's vertex group (default `point_<i>`);
    `attributes` maps names to `(n,)` arrays stored as float point attributes.
    """
    unitary = np.asarray(unitary, dtype=np.float64).reshape(-1, 3)
    count = len(unitary)
    point_names = [str(n) for n in point_names] if point_names is not None else [f"point_{i}" for i in range(count)]
    if len(point_names) != count:
        raise ValueError(f"{len(point_names)} names for {count} points")
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(count)
    for key, values in (attributes or {}).items():
        mesh.attributes.new(key, "FLOAT", "POINT").data.foreach_set("value", np.asarray(values, dtype=np.float32))
    mesh.attributes.new(SCALE_ATTRIBUTE, "FLOAT", "POINT")
    obj = bpy.data.objects.new(name, mesh)
    obj[UNITARY_KEY] = unitary.ravel().tolist()
    obj[NAMES_KEY] = point_names
    for index, point_name in enumerate(point_names):
        obj.vertex_groups.new(name=point_name).add([index], 1.0, "REPLACE")
    scene.collection.objects.link(obj)
    update_object(obj, scene)
    return obj


def read_points_file(path):
    """`(unitary (n, 3), attributes)` from a .npy array (columns x, y, z, then
    any extras as `column_<i>`) or a CSV/whitespace text file, whose optional
    header names the columns.
    """
    if path.lower().endswith(".npy"):
        data = np.atleast_2d(np.load(path))
        headers = [f"column_{i}" for i in range(data.shape[1])]
    else:
        with open(path) as f:
            first = f.readline()
        delimiter = "," if "," in first else None
        try:
            [float(v) for v in first.replace(",", " ").split()]
            header = None
        except ValueError:
            header = [h.strip() for h in (first.split(",") if delimiter else first.split())]
        data = np.atleast_2d(np.loadtxt(path, delimiter=delimiter, skiprows=1 if header else 0, dtype=np.float64))
        headers = header or [f"column_{i}" for i in range(data.shape[1])]
    if data.shape[1] < 3:
        raise ValueError(f"{os.path.basename(path)}: need x, y, z columns, got {data.shape[1]}")
    return data[:, :3], {h: data[:, i] for i, h in enumerate(headers) if i >= 3}


def update_object(obj, scene):
    positions = world_positions(obj, scene)
    obj.hide_viewport = obj.hide_render = positions is None
    if positions is None:
        return
    mesh = obj.data
    mesh.vertices.foreach_set("co", positions.astype(np.float32).ravel())
    scale = shot_space.world_per_unitary(scene.deep_zoom.zoom, scene.deep_zoom.world_scale)
    mesh.attributes[SCALE_ATTRIBUTE].data.foreach_set("value", np.full(len(mesh.vertices), scale, dtype=np.float32))
    mesh.update()


def update_points(scene):
    for obj in scene.objects:
        if is_points(obj):
            update_object(obj, scene)


class DEEPZOOM_OT_add_points(bpy.types.Operator):
    """Add points of interest that follow the zoom, from unitary x, y, z in a .npy, .csv or .txt file"""

    bl_idname = "deep_zoom.add_points"
    bl_label = "Add Points from File"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.npy;*.csv;*.txt;*.dat", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            unitary, attributes = read_points_file(bpy.path.abspath(self.filepath))
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:200])
            return {"CANCELLED"}
        name = os.path.splitext(os.path.basename(self.filepath))[0] or "DeepZoomPoints"
        obj = create_points(context.scene, unitary, name=name, attributes=attributes)
        if obj.hide_viewport:
            self.report({"WARNING"}, "Points are hidden until a shot with explicit anchor coordinates plays")
        return {"FINISHED"}
