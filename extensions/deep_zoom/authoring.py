"""Shots in shot space, the grid map, the Dive helper, path export and the UI.

The scene's `deep_zoom` settings hold an animatable zoom (log10) and a list
of shots, each a float64 anchor (unitary "x, y, z", or "max") starting at a
frame. During a shot the Blender world is shot space (see `shot_space`): the
anchor is the origin, and the camera is animated at ordinary sizes while the
zoom curve does the diving. Flagged volume objects render in shot space, and
the grid map draws the dataset's grid boxes near the current scale so there is
something to steer by at any depth. Export writes every frame's camera pose in
unitary float64, independent of Blender's float32.
"""

import json
import math
import os

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import grid_structure, shot_space

GRID_MAP_NAME = "DeepZoomGridMap"
_grid_cache = {}  # absolute path -> grid_structure dict


def _settings(scene):
    return scene.deep_zoom


def _redraw(self, context):
    update_grid_map(context.scene)


class DeepZoomShot(bpy.types.PropertyGroup):
    """One shot: a float64 anchor from a start frame on."""

    anchor: bpy.props.StringProperty(
        name="Anchor",
        description="The shot's origin: unitary 'x, y, z' (kept as text for float64), or 'max'",
        default="0.5, 0.5, 0.5",
        update=_redraw,
    )
    frame_start: bpy.props.IntProperty(name="Start", default=1, update=_redraw)


class DeepZoomSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Shot Space",
        description="Render flagged volumes in shot space: the active shot's anchor at the origin, scaled by the zoom",
        default=False,
        update=_redraw,
    )
    zoom: bpy.props.FloatProperty(
        name="Zoom",
        description="Magnification as a power of 10; animate it to dive",
        default=0.0,
        update=_redraw,
    )
    world_scale: bpy.props.FloatProperty(
        name="World Scale",
        description="World units per unitary length at zoom 0",
        default=shot_space.DEFAULT_WORLD_SCALE,
        min=1e-6,
        update=_redraw,
    )
    shots: bpy.props.CollectionProperty(type=DeepZoomShot)
    active_shot_index: bpy.props.IntProperty(default=0)
    grid_file: bpy.props.StringProperty(
        name="Grid Structure",
        description=".npz written by cycles_volume_override.grid_structure",
        subtype="FILE_PATH",
        update=_redraw,
    )
    show_grid_map: bpy.props.BoolProperty(name="Grid Map", default=True, update=_redraw)
    map_min_size: bpy.props.FloatProperty(name="Smallest", default=0.1, min=0.0, update=_redraw)
    map_max_size: bpy.props.FloatProperty(name="Largest", default=6.0, min=0.0, update=_redraw)
    map_max_distance: bpy.props.FloatProperty(name="Reach", default=50.0, min=0.0, update=_redraw)
    map_max_count: bpy.props.IntProperty(
        name="Max Boxes", default=600, min=1, update=_redraw,
        description="Draw at most this many boxes, the nearest the anchor first",
    )
    map_thickness: bpy.props.FloatProperty(name="Line Width", default=0.004, min=0.0, update=_redraw)


# --- shots and zoom -------------------------------------------------------------


def active_shot(scene, frame=None):
    """The shot playing at `frame` (default the current one), or None."""
    shots = _settings(scene).shots
    index = shot_space.active_shot([s.frame_start for s in shots], scene.frame_current if frame is None else frame)
    return None if index is None else shots[index]


def _fcurve(owner, data_path, index=0):
    anim = owner.animation_data
    if anim is None or anim.action is None:
        return None
    from bpy_extras import anim_utils

    channelbag = anim_utils.animdata_get_channelbag_for_assigned_slot(anim)
    return None if channelbag is None else channelbag.fcurves.find(data_path, index=index)


def zoom_at(scene, frame):
    """The zoom at `frame`, from its F-curve if it is animated."""
    curve = _fcurve(scene, "deep_zoom.zoom")
    return _settings(scene).zoom if curve is None else curve.evaluate(frame)


def volume_custom_params(scene):
    """custom_params that put flagged volumes in shot space, or {} when off."""
    settings = _settings(scene)
    shot = active_shot(scene) if settings.enabled else None
    if shot is None:
        return {}
    return {
        "yt_idv_shot_space": settings.world_scale,
        "yt_idv_zoom_target": shot.anchor,
        "yt_idv_zoom_level": settings.zoom,
    }


def add_shot(scene, anchor, frame, name=None):
    shot = _settings(scene).shots.add()
    shot.name = name or f"Shot {len(_settings(scene).shots)}"
    shot.anchor = anchor if isinstance(anchor, str) else shot_space.format_anchor(anchor)
    shot.frame_start = frame
    return shot


def _bezier_split(p0, p1, p2, p3, x):
    """De Casteljau split of a 2D cubic at the parameter where its x is `x`
    (x increases along an F-curve segment). Returns the five new control points.
    """
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        u = 1.0 - t
        if u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0] < x:
            lo = t
        else:
            hi = t
    lerp = lambda a, b: a + (b - a) * t  # noqa: E731
    p01, p12, p23 = lerp(p0, p1), lerp(p1, p2), lerp(p2, p3)
    p012, p123 = lerp(p01, p12), lerp(p12, p23)
    return p01, p012, lerp(p012, p123), p123, p23


def _split(curve, frame):
    """Add a key at `frame` without changing the curve's shape. Handles must
    already be FREE, so no neighbour recomputes its own.
    """
    points = curve.keyframe_points
    if any(k.co.x == frame for k in points):
        return
    value = curve.evaluate(frame)
    i0 = max((i for i, k in enumerate(points) if k.co.x < frame), default=None)
    i1 = i0 + 1 if i0 is not None else 0
    has_next = i1 < len(points)
    bezier = i0 is not None and has_next and points[i0].interpolation == "BEZIER"
    if bezier:
        vec = lambda v: np.array([v[0], v[1]], dtype=np.float64)  # noqa: E731
        k0, k1 = points[i0], points[i1]
        p01, p012, mid, p123, p23 = _bezier_split(vec(k0.co), vec(k0.handle_right), vec(k1.handle_left), vec(k1.co), frame)
        interpolation = "BEZIER"
    else:
        interpolation = points[i0].interpolation if i0 is not None else "BEZIER"
    # insert() reallocates the keys: look them up again afterwards
    points.insert(frame, value, options={"FAST"})
    key = next(k for k in points if k.co.x == frame)
    key.handle_left_type = key.handle_right_type = "FREE"
    key.interpolation = interpolation
    if not bezier:
        key.handle_left = (frame - 1.0, value)
        key.handle_right = (frame + 1.0, value)
        return
    points[i0].handle_right = p01
    key.co = mid
    key.handle_left = p012
    key.handle_right = p123
    points[i0 + 2].handle_left = p23


def _followers(camera):
    """The objects whose keys a cut must move: the camera and what it tracks."""
    if camera is None:
        return []
    objects = [camera]
    target = _track_target(camera)
    if target is not None and target not in objects:
        objects.append(target)
    return objects


def _location_curves(obj):
    return [(_fcurve(obj, "location", axis), axis) for axis in range(3)]


def _cut(curves, frame, transform):
    """Map the curves from `frame` on through `transform(value, axis)`,
    leaving every earlier frame exactly as it was. `curves` is
    [(fcurve or None, axis)]; `transform` must be affine in the value, so that
    applying it to keys and handles maps the frames between keys too.
    """
    for curve, axis in curves:
        if curve is None:
            continue
        for key in curve.keyframe_points:
            key.handle_left_type = key.handle_right_type = "FREE"
        _split(curve, frame - 1)
        _split(curve, frame)
        for key in curve.keyframe_points:
            if key.co.x < frame:
                continue
            key.co.y = transform(key.co.y, axis)
            key.handle_left.y = transform(key.handle_left.y, axis)
            key.handle_right.y = transform(key.handle_right.y, axis)
        curve.update()


def add_shot_at(scene, point, frame, camera=None, name=None):
    """A new shot from `frame`, anchored at `point` (world units in the shot
    playing then). From `frame` on, the location keys of `camera` and of what
    it tracks move by -`point`: the view doesn't jump at the cut, the camera's
    motion is unchanged in shot space, and since the zoom magnifies about the
    anchor, the rest of the dive converges on the new point.
    """
    current = active_shot(scene, frame)
    if current is None:
        raise ValueError("there's no shot to re-anchor from; add a first shot")
    old = shot_space.parse_anchor(current.anchor)
    if isinstance(old, str):
        raise ValueError(f"shot {current.name!r} is anchored at 'max'; give it explicit coordinates first")
    settings = _settings(scene)
    zoom = zoom_at(scene, frame)
    point = np.asarray(point, dtype=np.float64)
    anchor = shot_space.to_unitary(point, old, zoom, settings.world_scale)
    for obj in _followers(camera):
        _cut(_location_curves(obj), frame, lambda value, axis: value - point[axis])
    shot = add_shot(scene, anchor, frame, name)
    update_grid_map(scene)
    return shot


# --- when to re-anchor or rescale -------------------------------------------------

FLOAT32_EPS = 2.0 ** -24
JITTER_LIMIT_PX = 0.25  # re-anchor when float32 moves things more than this
SCALE_LIMITS = (0.01, 100.0)  # rescale when the focus is nearer or farther, in world units


def _track_target(camera):
    for constraint in camera.constraints:
        if constraint.type in ("TRACK_TO", "DAMPED_TRACK", "LOCKED_TRACK") and constraint.target is not None:
            return constraint.target
    return None


def focus_point(camera):
    """What the camera looks at: its tracking target, else the anchor (the origin)."""
    target = _track_target(camera)
    return np.zeros(3) if target is None else np.array(target.matrix_world.translation, dtype=np.float64)


def _vertical_fov(scene, camera):
    data = camera.data
    aspect = scene.render.resolution_x / scene.render.resolution_y
    fit = data.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if aspect >= 1.0 else "VERTICAL"
        sensor = data.sensor_width
    else:
        sensor = data.sensor_height if fit == "VERTICAL" else data.sensor_width
    half = sensor / 2.0 / data.lens
    return 2.0 * math.atan(half / aspect if fit == "HORIZONTAL" else half)


def frame_health(scene, camera=None):
    """How the current frame sits in shot space.

    Returns a dict: `distance` (camera to focus, world units), `jitter_px`
    (float32 error of the coordinates in play, in pixels at the focus), and
    `advice`, a list of ("reanchor" | "rescale", message) pairs.
    """
    camera = camera or scene.camera
    position = np.array(camera.matrix_world.translation, dtype=np.float64)
    focus = focus_point(camera)
    distance = float(np.linalg.norm(position - focus))
    pixel = 2.0 * distance * math.tan(_vertical_fov(scene, camera) / 2.0) / scene.render.resolution_y
    extent = max(np.abs(position).max(), np.abs(focus).max())
    jitter = FLOAT32_EPS * extent / pixel if pixel > 0 else math.inf
    advice = []
    if jitter > JITTER_LIMIT_PX:
        advice.append(("reanchor", f"float32 jitter ~{jitter:.2g} px: re-anchor at the focus"))
    if distance > 0 and not SCALE_LIMITS[0] <= distance <= SCALE_LIMITS[1]:
        advice.append(("rescale", f"focus {distance:.3g} units away: rescale by {-math.log10(distance):+.2f}"))
    return {"distance": distance, "jitter_px": jitter, "advice": advice}


MARKER_PREFIX = "DZ "


def check_path(scene, camera=None, frame_start=None, frame_end=None, mark=True):
    """frame_health for every frame; with `mark`, timeline markers ("DZ
    re-anchor", "DZ rescale") at the first frame of each run needing one.
    Returns {frame: advice kinds}.
    """
    camera = camera or scene.camera
    frame_start = scene.frame_start if frame_start is None else frame_start
    frame_end = scene.frame_end if frame_end is None else frame_end
    saved = scene.frame_current
    report = {}
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            report[frame] = [kind for kind, _ in frame_health(scene, camera)["advice"]]
    finally:
        scene.frame_set(saved)
    if mark:
        for marker in [m for m in scene.timeline_markers if m.name.startswith(MARKER_PREFIX)]:
            scene.timeline_markers.remove(marker)
        for kind, label in (("reanchor", "re-anchor"), ("rescale", "rescale")):
            previous = False
            for frame in range(frame_start, frame_end + 1):
                now = kind in report[frame]
                if now and not previous:
                    scene.timeline_markers.new(MARKER_PREFIX + label, frame=frame)
                previous = now
    return report


def rescale_at(scene, frame, decades, camera=None):
    """Change units from `frame` on without changing the view: the zoom rises
    by `decades` and the camera (and what it tracks) moves 10**decades
    farther from the anchor.
    """
    settings = _settings(scene)
    factor = 10.0 ** decades
    zoom_curve = _fcurve(scene, "deep_zoom.zoom")
    if zoom_curve is None:
        # an unanimated zoom becomes a step
        start = settings.zoom
        for f, value in ((frame - 1, start), (frame, start + decades)):
            settings.zoom = value
            scene.keyframe_insert("deep_zoom.zoom", frame=f)
        zoom_curve = _fcurve(scene, "deep_zoom.zoom")
        for key in zoom_curve.keyframe_points:
            key.interpolation = "CONSTANT"
    else:
        _cut([(zoom_curve, 0)], frame, lambda value, axis: value + decades)
    for obj in _followers(camera or scene.camera):
        curves = _location_curves(obj)
        if all(curve is None for curve, _ in curves):
            continue
        _cut(curves, frame, lambda value, axis: value * factor)
    scene.frame_set(scene.frame_current)
    update_grid_map(scene)


# --- grid map --------------------------------------------------------------------


def _grids(path):
    path = bpy.path.abspath(path)
    if not path or not os.path.isfile(path):
        return None
    if path not in _grid_cache:
        _grid_cache[path] = grid_structure.load(path)
    return _grid_cache[path]


def _grid_map_material():
    material = bpy.data.materials.get(GRID_MAP_NAME)
    if material is not None:
        return material
    material = bpy.data.materials.new(GRID_MAP_NAME)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    level = nodes.new("ShaderNodeAttribute")
    level.attribute_name = "level_fraction"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.2, 0.45, 1.0, 1.0)
    ramp.color_ramp.elements[1].color = (0.6, 1.0, 0.85, 1.0)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.5
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(level.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs[0])
    return material


def _grid_map_object(scene):
    obj = bpy.data.objects.get(GRID_MAP_NAME)
    if obj is None:
        mesh = bpy.data.meshes.new(GRID_MAP_NAME)
        obj = bpy.data.objects.new(GRID_MAP_NAME, mesh)
        scene.collection.objects.link(obj)
        wire = obj.modifiers.new("Lines", "WIREFRAME")
        wire.use_replace = True
        wire.use_even_offset = False
        mesh.materials.append(_grid_map_material())
    return obj


def update_grid_map(scene):
    """Rebuild the grid map for the scene's current shot and zoom."""
    settings = _settings(scene)
    obj = bpy.data.objects.get(GRID_MAP_NAME)
    shot = active_shot(scene)
    grids = _grids(settings.grid_file)
    anchor = None if shot is None else shot_space.parse_anchor(shot.anchor)
    if not (settings.enabled and settings.show_grid_map and grids is not None) or anchor is None or isinstance(anchor, str):
        if obj is not None:
            obj.hide_viewport = obj.hide_render = True
        return
    obj = _grid_map_object(scene)
    obj.hide_viewport = obj.hide_render = False
    obj.modifiers["Lines"].thickness = settings.map_thickness
    left, right, index = shot_space.visible_boxes(
        grids["left"], grids["right"], anchor, settings.zoom, settings.world_scale,
        min_size=settings.map_min_size, max_size=settings.map_max_size,
        max_distance=settings.map_max_distance, max_count=settings.map_max_count,
    )
    verts, faces = shot_space.box_faces(left, right)
    mesh = obj.data
    mesh.clear_geometry()
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, dtype=np.int32))
    levels = grids["level"][index].astype(np.float32)
    fraction = levels / max(float(grids["level"].max()), 1.0)
    attribute = mesh.attributes.new("level_fraction", "FLOAT", "POINT")
    attribute.data.foreach_set("value", np.repeat(fraction, 8))
    mesh.update()


@persistent
def _on_frame_change(scene, depsgraph=None):
    if _settings(scene).enabled:
        update_grid_map(scene)


# --- export ----------------------------------------------------------------------


def camera_path(scene, camera=None, frame_start=None, frame_end=None):
    """Every frame's camera pose in unitary float64, as a JSON-ready dict."""
    camera = camera or scene.camera
    settings = _settings(scene)
    frame_start = scene.frame_start if frame_start is None else frame_start
    frame_end = scene.frame_end if frame_end is None else frame_end
    saved = scene.frame_current
    frames = []
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            shot = active_shot(scene)
            anchor = shot_space.parse_anchor(shot.anchor)
            if isinstance(anchor, str):
                raise ValueError(f"shot {shot.name!r} is anchored at 'max'; export needs explicit coordinates")
            matrix = np.array(camera.matrix_world, dtype=np.float64)
            frames.append({
                "frame": frame,
                "shot": shot.name,
                "zoom": settings.zoom,
                "position": shot_space.to_unitary(matrix[:3, 3], anchor, settings.zoom, settings.world_scale).tolist(),
                # shot space is a uniform scale of unitary space, so directions carry over
                "forward": (-matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])).tolist(),
                "up": (matrix[:3, 1] / np.linalg.norm(matrix[:3, 1])).tolist(),
                # the camera's distance to the anchor, in unitary lengths
                "distance": float(np.linalg.norm(matrix[:3, 3]) / shot_space.world_per_unitary(settings.zoom, settings.world_scale)),
            })
    finally:
        scene.frame_set(saved)
    data = camera.data
    return {
        "format": "cycles-volume-override deep-zoom path 1",
        "coordinates": "unitary",
        "world_scale": settings.world_scale,
        "shots": [{"name": s.name, "anchor": s.anchor, "frame_start": s.frame_start} for s in settings.shots],
        "camera": {
            "lens_mm": data.lens, "sensor_width_mm": data.sensor_width, "sensor_height_mm": data.sensor_height,
            "sensor_fit": data.sensor_fit, "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "angle_y_deg": math.degrees(data.angle_y),
        },
        "frames": frames,
    }


def export_camera_path(scene, path, camera=None):
    with open(path, "w") as f:
        json.dump(camera_path(scene, camera), f, indent=1)


# --- operators and UI ------------------------------------------------------------


class DEEPZOOM_OT_add_shot(bpy.types.Operator):
    """Start a new shot here, anchored at the 3D cursor"""

    bl_idname = "deep_zoom.add_shot"
    bl_label = "New Shot at Cursor"
    bl_options = {"REGISTER", "UNDO"}

    keep_view: bpy.props.BoolProperty(
        name="Keep View", default=True,
        description="Shift the camera's later location keys so the data doesn't jump",
    )

    def execute(self, context):
        scene = context.scene
        settings = _settings(scene)
        try:
            if not settings.shots:
                add_shot(scene, "0.5, 0.5, 0.5", scene.frame_current, "Shot 1")
            else:
                add_shot_at(scene, scene.cursor.location, scene.frame_current,
                            scene.camera if self.keep_view else None)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        settings.active_shot_index = len(settings.shots) - 1
        scene.cursor.location = (0.0, 0.0, 0.0)  # the new anchor
        update_grid_map(scene)
        return {"FINISHED"}


def finest_box_center(scene):
    """Center of the finest box in the grid structure, float64 unitary, or None."""
    grids = _grids(_settings(scene).grid_file)
    if grids is None:
        return None
    finest = np.argmax(grids["level"])
    return (grids["left"][finest] + grids["right"][finest]) / 2.0


class DEEPZOOM_OT_anchor_finest(bpy.types.Operator):
    """Anchor the selected shot at the center of the finest grid box"""

    bl_idname = "deep_zoom.anchor_finest"
    bl_label = "Anchor at Finest Grid"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = _settings(context.scene)
        center = finest_box_center(context.scene)
        if center is None or not settings.shots:
            self.report({"ERROR"}, "needs a grid structure file and a shot")
            return {"CANCELLED"}
        settings.shots[settings.active_shot_index].anchor = shot_space.format_anchor(center)
        return {"FINISHED"}


class DEEPZOOM_OT_remove_shot(bpy.types.Operator):
    """Remove the selected shot"""

    bl_idname = "deep_zoom.remove_shot"
    bl_label = "Remove Shot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = _settings(context.scene)
        if settings.shots:
            settings.shots.remove(settings.active_shot_index)
            settings.active_shot_index = max(0, min(settings.active_shot_index, len(settings.shots) - 1))
            update_grid_map(context.scene)
        return {"FINISHED"}


class DEEPZOOM_OT_jump_to_shot(bpy.types.Operator):
    """Go to the selected shot's first frame"""

    bl_idname = "deep_zoom.jump_to_shot"
    bl_label = "Jump to Shot"

    def execute(self, context):
        settings = _settings(context.scene)
        if settings.shots:
            context.scene.frame_set(settings.shots[settings.active_shot_index].frame_start)
        return {"FINISHED"}


def add_dive(scene, frame, frames, decades):
    """Keyframe the zoom from `frame` to `frame + frames`, `decades` deeper,
    linearly: a steady exponential dive.
    """
    settings = _settings(scene)
    start = zoom_at(scene, frame)
    for f, value in ((frame, start), (frame + frames, start + decades)):
        settings.zoom = value
        scene.keyframe_insert("deep_zoom.zoom", frame=f)
    curve = _fcurve(scene, "deep_zoom.zoom")
    for key in curve.keyframe_points:
        if frame <= key.co.x <= frame + frames:
            key.interpolation = "LINEAR"
    curve.update()
    scene.frame_set(scene.frame_current)


class DEEPZOOM_OT_dive(bpy.types.Operator):
    """Keyframe a steady dive: the zoom rises linearly from this frame"""

    bl_idname = "deep_zoom.dive"
    bl_label = "Dive"
    bl_options = {"REGISTER", "UNDO"}

    decades: bpy.props.FloatProperty(name="Decades", default=3.0, description="How much deeper, as a power of 10")
    frames: bpy.props.IntProperty(name="Frames", default=120, min=1)

    def execute(self, context):
        add_dive(context.scene, context.scene.frame_current, self.frames, self.decades)
        return {"FINISHED"}


class DEEPZOOM_OT_check_path(bpy.types.Operator):
    """Check every frame and mark where to re-anchor or rescale"""

    bl_idname = "deep_zoom.check_path"
    bl_label = "Check Path"

    def execute(self, context):
        if context.scene.camera is None:
            self.report({"ERROR"}, "the scene has no camera")
            return {"CANCELLED"}
        report = check_path(context.scene)
        bad = {kind for kinds in report.values() for kind in kinds}
        count = sum(1 for kinds in report.values() if kinds)
        if count:
            self.report({"WARNING"}, f"{count} frames need attention ({', '.join(sorted(bad))}); see the DZ markers")
        else:
            self.report({"INFO"}, "every frame is fine")
        return {"FINISHED"}


class DEEPZOOM_OT_reanchor_focus(bpy.types.Operator):
    """Start a new shot here, anchored at what the camera is looking at"""

    bl_idname = "deep_zoom.reanchor_focus"
    bl_label = "Re-anchor at Focus"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        try:
            add_shot_at(scene, focus_point(scene.camera), scene.frame_current, scene.camera)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _settings(scene).active_shot_index = len(_settings(scene).shots) - 1
        scene.frame_set(scene.frame_current)
        return {"FINISHED"}


class DEEPZOOM_OT_rescale(bpy.types.Operator):
    """Change units from here on without changing the view"""

    bl_idname = "deep_zoom.rescale"
    bl_label = "Rescale"
    bl_options = {"REGISTER", "UNDO"}

    decades: bpy.props.FloatProperty(name="Decades", description="Default: bring the focus to 1 unit away")

    def invoke(self, context, event):
        distance = frame_health(context.scene)["distance"]
        self.decades = -math.log10(distance) if distance > 0 else 0.0
        return self.execute(context)

    def execute(self, context):
        rescale_at(context.scene, context.scene.frame_current, self.decades)
        return {"FINISHED"}


class DEEPZOOM_OT_export_path(bpy.types.Operator):
    """Write every frame's camera pose in unitary float64 coordinates"""

    bl_idname = "deep_zoom.export_path"
    bl_label = "Export Camera Path"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="//camera_path.json")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            export_camera_path(context.scene, bpy.path.abspath(self.filepath))
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"wrote {self.filepath}")
        return {"FINISHED"}


class DEEPZOOM_UL_shots(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False)
        row.prop(item, "frame_start", text="")


class DEEPZOOM_PT_panel(bpy.types.Panel):
    bl_idname = "DEEPZOOM_PT_panel"
    bl_label = "Deep Zoom"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Deep Zoom"

    def draw(self, context):
        scene = context.scene
        settings = _settings(scene)
        layout = self.layout
        layout.prop(settings, "enabled")
        col = layout.column()
        col.enabled = settings.enabled
        row = col.row(align=True)
        row.prop(settings, "zoom")
        row.operator(DEEPZOOM_OT_dive.bl_idname, text="", icon="SORT_DESC")
        col.prop(settings, "world_scale")
        unit = 1.0 / shot_space.world_per_unitary(settings.zoom, settings.world_scale)
        col.label(text=f"1 unit = {unit:.3g} of the domain")

        col.label(text="Shots")
        row = col.row()
        row.template_list("DEEPZOOM_UL_shots", "", settings, "shots", settings, "active_shot_index", rows=3)
        side = row.column(align=True)
        side.operator(DEEPZOOM_OT_add_shot.bl_idname, icon="ADD", text="")
        side.operator(DEEPZOOM_OT_remove_shot.bl_idname, icon="REMOVE", text="")
        side.operator(DEEPZOOM_OT_jump_to_shot.bl_idname, icon="PLAY", text="")
        if settings.shots:
            row = col.row(align=True)
            row.prop(settings.shots[settings.active_shot_index], "anchor")
            row.operator(DEEPZOOM_OT_anchor_finest.bl_idname, text="", icon="SNAP_VOLUME")
        playing = active_shot(scene)
        if playing is not None:
            col.label(text=f"Playing: {playing.name}")

        col.separator()
        col.prop(settings, "grid_file")
        col.prop(settings, "show_grid_map")
        sub = col.column(align=True)
        sub.enabled = settings.show_grid_map
        sub.prop(settings, "map_min_size")
        sub.prop(settings, "map_max_size")
        sub.prop(settings, "map_max_distance")
        sub.prop(settings, "map_max_count")
        sub.prop(settings, "map_thickness")

        if scene.camera is not None and active_shot(scene) is not None:
            col.separator()
            health = frame_health(scene)
            box = col.box()
            box.label(text=f"Focus {health['distance']:.3g} units, jitter {health['jitter_px']:.2g} px",
                      icon="ERROR" if health["advice"] else "CHECKMARK")
            for _, message in health["advice"]:
                box.label(text=message)
            row = box.row(align=True)
            row.operator(DEEPZOOM_OT_reanchor_focus.bl_idname, icon="PIVOT_CURSOR")
            row.operator(DEEPZOOM_OT_rescale.bl_idname, icon="FULLSCREEN_ENTER")
            box.operator(DEEPZOOM_OT_check_path.bl_idname, icon="MARKER_HLT")

        col.separator()
        col.operator(DEEPZOOM_OT_export_path.bl_idname, icon="EXPORT")


CLASSES = (
    DeepZoomShot,
    DeepZoomSettings,
    DEEPZOOM_OT_add_shot,
    DEEPZOOM_OT_anchor_finest,
    DEEPZOOM_OT_remove_shot,
    DEEPZOOM_OT_jump_to_shot,
    DEEPZOOM_OT_dive,
    DEEPZOOM_OT_check_path,
    DEEPZOOM_OT_reanchor_focus,
    DEEPZOOM_OT_rescale,
    DEEPZOOM_OT_export_path,
    DEEPZOOM_UL_shots,
    DEEPZOOM_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.deep_zoom = bpy.props.PointerProperty(type=DeepZoomSettings)
    if _on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change)


def unregister():
    if _on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change)
    del bpy.types.Scene.deep_zoom
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
