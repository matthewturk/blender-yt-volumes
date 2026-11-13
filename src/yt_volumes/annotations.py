"""The annotation scene: a flat scene laid over the finished frame, holding
colorbars that read volumes' transfer functions, scale bars, and anything
built with the "yt Annotation View" group, all through Geometry Nodes.

Its orthographic camera puts x from 0 to 1 across the frame and y from 0 to
height/width up it, origin at the bottom left; build annotations on z = 0.
Pass 2 renders it with Cycles through a Render Layers node, at the main
scene's frame and resolution, and lays it over the volumes.
"""

import re

import bpy

from . import annotation_library, units

SCENE_PROPERTY = "yt_annotation_scene"
COLORBAR_GROUP = "yt Colorbar"
FOV_GROUP = "yt Field of View"
VIEW_GROUP = "yt Annotation View"
TEMPLATE_GROUP = "yt Annotation"
MATERIAL_NAME = "yt Annotation"
COLOR_ATTRIBUTE = "yt_annotation_color"  # scene-linear; the material emits it
TICK_ATTRIBUTE = "yt_tick_value"
CAMERA_HEIGHT = 10.0
ASSET_LIBRARY = annotation_library.BLEND


def aspect(scene):
    """Frame height over width, in the annotation scene's y units."""
    render = scene.render
    return (render.resolution_y * render.pixel_aspect_y) / (render.resolution_x * render.pixel_aspect_x)


def annotation_scene(scene):
    return getattr(scene, SCENE_PROPERTY, None)


def _assign(owner, name, value):
    # Same-value assignments still tag the scene for re-sync.
    if getattr(owner, name) != value:
        setattr(owner, name, value)


def sync(scene, annotations):
    """Match `annotations`' frame, resolution and camera framing to `scene`'s."""
    for name in ("resolution_x", "resolution_y", "resolution_percentage", "pixel_aspect_x", "pixel_aspect_y"):
        _assign(annotations.render, name, getattr(scene.render, name))
    _assign(annotations, "frame_current", scene.frame_current)
    camera = annotations.camera
    if camera is not None:
        location = (0.5, aspect(scene) / 2.0, CAMERA_HEIGHT)
        if tuple(camera.location) != location:
            camera.location = location


def add_annotation_scene(scene):
    """`scene`'s annotation scene, made if it has none."""
    view_node_group()  # in Add > Group for building annotations
    existing = annotation_scene(scene)
    if existing is not None:
        return existing
    annotations = bpy.data.scenes.new(f"{scene.name} Annotations")
    # Needs the GPU compositor in the main scene; see operators.compatibility_warnings.
    annotations.render.engine = "BLENDER_EEVEE"
    annotations.eevee.taa_render_samples = 16
    annotations.cycles.samples = 16
    annotations.cycles.use_denoising = False
    annotations.cycles.max_bounces = 0
    annotations.render.film_transparent = True
    annotations.view_settings.view_transform = "Standard"
    data = bpy.data.cameras.new(annotations.name)
    data.type, data.ortho_scale, data.sensor_fit = "ORTHO", 1.0, "HORIZONTAL"
    data.clip_start, data.clip_end = 0.1, 2.0 * CAMERA_HEIGHT
    camera = bpy.data.objects.new(annotations.name, data)
    annotations.collection.objects.link(camera)
    annotations.camera = camera
    setattr(scene, SCENE_PROPERTY, annotations)
    sync(scene, annotations)
    return annotations


def _append(kind, name):
    """Append `name` from the bundled library into `bpy.data.<kind>`, using the
    file's own copies of anything it depends on that the file already has.
    """
    collections = ("node_groups", "materials", "fonts")
    before = {c: set(getattr(bpy.data, c).keys()) for c in collections}
    with bpy.data.libraries.load(ASSET_LIBRARY, link=False) as (_, dst):
        setattr(dst, kind, [name])
    appended = getattr(dst, kind)[0]
    for c in collections:
        data = getattr(bpy.data, c)
        for new_name in set(data.keys()) - before[c]:
            idb = data[new_name]
            idb.asset_clear()  # assets in the library, plain data in the user's file
            # A dependency the file already had arrives renamed, e.g. "yt Annotation View.001".
            original = re.sub(r"\.\d{3}$", "", new_name)
            if idb != appended and original in before[c]:
                idb.user_remap(data[original])
                data.remove(idb)
    return appended


def _get_or_append(kind, name):
    return getattr(bpy.data, kind).get(name) or _append(kind, name)


def annotation_material():
    """An emission material showing each point's `yt_annotation_color`."""
    return _get_or_append("materials", MATERIAL_NAME)


def colorbar_node_group():
    """The "yt Colorbar" group, appended if the file has none: a bar of the
    volume's transfer function colors from the object's origin to +x, ticks
    and log10 labels under it, a title, and its opacity above it.
    """
    return _get_or_append("node_groups", COLORBAR_GROUP)


def view_node_group():
    """The "yt Annotation View" group, appended if the file has none: where a
    main-scene point lands on the frame, and how long one Unit is there.
    """
    return _get_or_append("node_groups", VIEW_GROUP)


def fov_node_group():
    """The "yt Field of View" group, appended if the file has none: a labeled
    scale bar from the object's origin to +x, sized at the volume's zoom
    target or middle; also an example of "yt Annotation View".
    """
    return _get_or_append("node_groups", FOV_GROUP)


def annotation_template():
    """A new Geometry Nodes tree to build an annotation from: a dot Size
    across at the volume's zoom target or middle, through "yt Annotation View".
    """
    return _append("node_groups", TEMPLATE_GROUP)


def _set_input(modifier, name, value):
    item = next(i for i in modifier.node_group.interface.items_tree
                if i.item_type == "SOCKET" and i.in_out == "INPUT" and i.name == name)
    getattr(modifier.properties.inputs, item.identifier).value = value


def default_title(volume):
    """`log₁₀ <field> (<units>)` for a volume, as far as it's known."""
    from . import yt_idv_renderer

    params = volume.volume_override.to_custom_params()
    field = params.get("yt_idv_field") or ""
    field = field.split(",")[-1].strip() if field else "field"
    fit = yt_idv_renderer.gn_tf_fit(volume.volume_override.dataset_uuid)
    units = f" ({fit['units']})" if fit and fit.get("units") else ""
    return f"log₁₀ {field}{units}"


def _annotations_with(annotations, group_name):
    return [o for o in annotations.objects if any(
        m.type == "NODES" and m.node_group is not None and m.node_group.name.startswith(group_name) for m in o.modifiers
    )]


def colorbars(annotations):
    return _annotations_with(annotations, COLORBAR_GROUP)


def default_unit(volume):
    """The first of `volume`'s Data Units its units object knows, else unitary."""
    group = volume.get(units.KEY, {})
    return next((u for u in units.data_units(volume) if group.get(f"scene_per_{u}")), "unitary")


# Inputs filled from the volume, once, when the object is added: only inputs
# with exactly these names and socket types, so "Unit Fraction" or a float "Unit" are left alone.
_VOLUME_INPUTS = {
    "Volume": ("NodeSocketObject", lambda volume: volume),
    "Units Object": ("NodeSocketObject", lambda volume: volume.volume_override.units_object),
    "Unit": ("NodeSocketString", default_unit),
    "Title": ("NodeSocketString", default_title),
}


def _volume_inputs(inputs):
    """The names in `inputs` ({socket name: socket type}) filled from the volume."""
    return {name for name, kind in inputs.items() if _VOLUME_INPUTS.get(name, (None,))[0] == kind}
# Where new annotation-scene objects go, by group; others go at the frame's origin.
_PLACES = {
    COLORBAR_GROUP: lambda annotations: (0.2, 0.08 + 0.15 * len(colorbars(annotations)), 0.0),
    FOV_GROUP: lambda annotations: (0.7, 0.1 + 0.12 * len(_annotations_with(annotations, FOV_GROUP)), 0.0),
}


def main_scene(scene):
    """The scene whose annotation scene `scene` is, or `scene` itself."""
    return next((s for s in bpy.data.scenes if annotation_scene(s) == scene), scene)


def active_volume(context):
    """The active object if it's a volume, else the scene's only volume, else None."""
    def is_volume(obj):
        return getattr(obj, "volume_override", None) is not None and obj.volume_override.is_volume_override

    if is_volume(context.object):
        return context.object
    volumes = [o for o in main_scene(context.scene).objects if is_volume(o)]
    return volumes[0] if len(volumes) == 1 else None


def unavailable(name, volume):
    """Why library annotation `name` can't be added for `volume` (may be None), or None."""
    inputs = _volume_inputs(annotation_library.read_index()[name]["inputs"])
    if volume is None and ("Volume" in inputs or "Units Object" in inputs):
        return "Select a volume"
    if "Units Object" in inputs and volume.volume_override.units_object is None:
        return f"Add a units object to {volume.name!r} first (Units for Geometry Nodes)"
    return None


def add_from_library(scene, name, volume=None, collection=None):
    """A new object running library annotation `name`, with its inputs filled
    from `volume`: in `scene`'s annotation scene (made if needed), or in
    `collection` (default the main scene's) at the 3D cursor. The starter
    "yt Annotation" gets its own copy of its tree; other groups are shared.
    """
    reason = unavailable(name, volume)
    if reason is not None:
        raise ValueError(f"Can't add {name!r}: {reason}")
    scene = main_scene(scene)
    label = name.removeprefix("yt ")
    obj = bpy.data.objects.new(f"{volume.name} {label}" if volume else label, bpy.data.meshes.new(label))
    if annotation_library.read_index()[name]["scene"] == annotation_library.ANNOTATION_SCENE:
        annotations = add_annotation_scene(scene)
        obj.location = _PLACES.get(name, lambda _: (0.0, 0.0, 0.0))(annotations)
        annotations.collection.objects.link(obj)
    else:
        obj.location = scene.cursor.location
        (collection or scene.collection).objects.link(obj)
    modifier = obj.modifiers.new(name, "NODES")
    modifier.node_group = annotation_template() if name == TEMPLATE_GROUP else _get_or_append("node_groups", name)
    if volume is not None:
        for item in modifier.node_group.interface.items_tree:
            if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name in _volume_inputs({item.name: item.socket_type}):
                _set_input(modifier, item.name, _VOLUME_INPUTS[item.name][1](volume))
    return obj


def add_colorbar(scene, volume):
    """A colorbar of `volume`'s transfer function in `scene`'s annotation
    scene (made if needed), below any others there.
    """
    return add_from_library(scene, COLORBAR_GROUP, volume)


def add_field_of_view(scene, volume):
    """A field-of-view scale bar for `volume` in `scene`'s annotation scene
    (made if needed), at the bottom right. `volume` needs a units object.
    """
    return add_from_library(scene, FOV_GROUP, volume)


def add_annotation(scene, volume):
    """An annotation object in `scene`'s annotation scene (made if needed),
    with its own copy of the starter tree to edit. `volume` needs a units object.
    """
    return add_from_library(scene, TEMPLATE_GROUP, volume)


class VOLUMEOVERRIDE_OT_add_colorbar(bpy.types.Operator):
    """Add a colorbar of this volume's transfer function to the scene's annotation scene"""

    bl_idname = "volume_override.add_colorbar"
    bl_label = "Add Colorbar"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and getattr(obj, "volume_override", None) is not None and obj.volume_override.is_volume_override

    def execute(self, context):
        obj = add_colorbar(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; open that scene to place it")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_field_of_view(bpy.types.Operator):
    """Add a scale bar estimating this volume's field of view to the scene's annotation scene"""

    bl_idname = "volume_override.add_field_of_view"
    bl_label = "Add Field of View"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        props = getattr(obj, "volume_override", None) if obj is not None else None
        if props is None or not props.is_volume_override:
            return False
        if props.units_object is None:
            cls.poll_message_set("Add a units object first (Units for Geometry Nodes)")
            return False
        return True

    def execute(self, context):
        obj = add_field_of_view(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; open that scene to place it")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_annotation(bpy.types.Operator):
    """Add an annotation with its own Geometry Nodes tree to edit, wired to this volume's units object"""

    bl_idname = "volume_override.add_annotation"
    bl_label = "Add Annotation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return VOLUMEOVERRIDE_OT_add_field_of_view.poll(context)

    def execute(self, context):
        obj = add_annotation(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; edit its yt Annotation nodes there")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_library_annotation(bpy.types.Operator):
    """Add an annotation from the yt-volumes library"""

    bl_idname = "volume_override.add_library_annotation"
    bl_label = "Add yt Annotation"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Annotation", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        entry = annotation_library.read_index().get(properties.name)
        if entry is None:
            return cls.__doc__
        reason = unavailable(properties.name, active_volume(context))
        return entry["description"] + (f"\n\nUnavailable: {reason}" if reason else "")

    def execute(self, context):
        if self.name not in annotation_library.read_index():
            self.report({"ERROR"}, f"No annotation {self.name!r} in the library")
            return {"CANCELLED"}
        volume = active_volume(context)
        reason = unavailable(self.name, volume)
        if reason is not None:
            self.report({"ERROR"}, reason)
            return {"CANCELLED"}
        in_main = main_scene(context.scene) == context.scene
        obj = add_from_library(context.scene, self.name, volume, context.collection if in_main else None)
        if obj.name in context.view_layer.objects:  # added to the scene being looked at
            for selected in context.selected_objects:
                selected.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        self.report({"INFO"}, f"Added {obj.name!r} to {obj.users_scene[0].name!r}")
        return {"FINISHED"}


class VOLUMEOVERRIDE_MT_add_annotation(bpy.types.Menu):
    """Annotations from the yt-volumes library, for the annotation scene or the 3D scene"""

    bl_idname = "VOLUMEOVERRIDE_MT_add_annotation"
    bl_label = "yt Annotation"

    def draw(self, context):
        layout = self.layout
        entries = annotation_library.read_index().values()
        volume = active_volume(context)
        for scene, heading in ((annotation_library.ANNOTATION_SCENE, "Annotation Scene"), (annotation_library.MAIN_SCENE, "3D Scene")):
            names = [e["name"] for e in entries if e["scene"] == scene]
            if not names:
                continue
            layout.label(text=heading)
            for name in names:
                row = layout.row()
                row.enabled = unavailable(name, volume) is None
                row.operator(VOLUMEOVERRIDE_OT_add_library_annotation.bl_idname, text=name).name = name


def _draw_add_menu(self, context):
    self.layout.menu(VOLUMEOVERRIDE_MT_add_annotation.bl_idname, icon="GEOMETRY_NODES")


_CLASSES = (
    VOLUMEOVERRIDE_OT_add_colorbar, VOLUMEOVERRIDE_OT_add_field_of_view, VOLUMEOVERRIDE_OT_add_annotation,
    VOLUMEOVERRIDE_OT_add_library_annotation, VOLUMEOVERRIDE_MT_add_annotation,
)


def register():
    bpy.types.Scene.yt_annotation_scene = bpy.props.PointerProperty(
        type=bpy.types.Scene, name="Annotation Scene",
        description="A flat scene laid over the finished frame (x 0-1 across it); holds colorbars and scale bars",
    )
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, SCENE_PROPERTY):
        del bpy.types.Scene.yt_annotation_scene
