"""Probes: Sphere Empties that run a yt analysis on the data inside them and
put the result in a float Image, shown in the annotation scene by "yt Image
Panel", and named scalars in `obj["yt_values"]`. The pipeline updates them
after the volumes render, before pass 2, so a driver in the main scene sees
the previous run's values unless they're baked (Bake Values).
"""

import json

import bpy
import numpy as np

from . import annotations, probe_analysis
from .properties import VolumeOverrideCustomParam

PANEL_GROUP = "yt Image Panel"
LINE_GROUP = "yt Line Panel"
LABEL_GROUP = "yt Probe Label"
DEFAULT_ANALYSIS = "yt_volumes.probe_analysis:phase_plot"


def _is_volume(_self, obj):
    return obj.volume_override.is_volume_override


class YtProbeProperties(bpy.types.PropertyGroup):
    is_probe: bpy.props.BoolProperty(
        name="Probe", description="Run an analysis on the data in this Empty's sphere and put the result in an Image",
    )
    volume: bpy.props.PointerProperty(
        type=bpy.types.Object, poll=_is_volume, name="Volume", description="The volume whose dataset to analyse",
    )
    analysis: bpy.props.StringProperty(
        name="Analysis", default=DEFAULT_ANALYSIS,
        description="'module:function' taking a ProbeContext and returning a ProbeResult",
    )
    image: bpy.props.PointerProperty(
        type=bpy.types.Image, name="Image", description="Float Image the result goes in; made if empty",
    )
    update: bpy.props.EnumProperty(
        name="Update",
        items=(
            ("ON_CHANGE", "On Change", "When the sphere, dataset, analysis or parameters change"),
            ("EVERY_FRAME", "Every Frame", "Every rendered frame, even if nothing changed"),
            ("MANUAL", "Manual", "Only from the Update Probe button"),
        ),
        default="ON_CHANGE",
    )
    params: bpy.props.CollectionProperty(type=VolumeOverrideCustomParam)
    params_active_index: bpy.props.IntProperty(default=0)
    state: bpy.props.StringProperty(
        name="State", description="JSON the analysis keeps between runs, such as fitted ranges",
    )
    probe_id: bpy.props.StringProperty(name="Probe ID")


def probes(scene):
    return [obj for obj in scene.objects if obj.yt_probe.is_probe]


def sphere(obj):
    """World-space `(center, radius)` of a Sphere Empty; scale is averaged
    over axes, so a non-uniformly scaled Empty still gets a sphere.
    """
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    scale = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    return matrix[:3, 3], obj.empty_display_size * scale


def _probe_id(obj):
    props = obj.yt_probe
    if not props.probe_id:
        import uuid

        props.probe_id = uuid.uuid4().hex
    return props.probe_id


def write_image(obj, pixels):
    """Put `(h, w, 4)` bottom-row-first `pixels` in the probe's Image."""
    props = obj.yt_probe
    height, width = pixels.shape[:2]
    image = props.image
    if image is None:
        image = props.image = new_image(f"yt Probe {obj.name}", width, height)
    elif not image.is_float:
        raise ValueError(f"probe {obj.name!r}: Image {image.name!r} must be a float Image")
    elif tuple(image.size) != (width, height):
        image.scale(width, height)
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    return image


def new_image(name, width=64, height=64):
    image = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    image.alpha_mode = "STRAIGHT"
    return image


def _inputs(modifier):
    return {item.name: item for item in modifier.node_group.interface.items_tree
            if item.item_type == "SOCKET" and item.in_out == "INPUT"}


def _value(modifier, item):
    return getattr(modifier.properties.inputs, item.identifier).value


def panels(image=None, probe=None):
    """`(object, modifier)` pairs whose Geometry Nodes Image input is `image`
    or whose Probe input is `probe`.
    """
    found = []
    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            inputs = _inputs(modifier)
            for name, socket_type, target in (("Image", "NodeSocketImage", image), ("Probe", "NodeSocketObject", probe)):
                item = inputs.get(name)
                if (target is not None and item is not None and item.socket_type == socket_type
                        and _value(modifier, item) == target):
                    found.append((obj, modifier))
                    break
    return found


def _sync_meta(obj, image, meta):
    """Copy `meta` onto `image` (if any) and onto same-named inputs of the
    annotations showing the probe. An annotation with a "Quantity" input also
    gets that scalar in "Value" and its units in "Units".
    """
    if image is not None:
        for key, value in meta.items():
            if image.get(key) != value:
                image[key] = value
    for _obj, modifier in panels(image, obj):
        inputs = _inputs(modifier)
        values = dict(meta)
        quantity = inputs.get("Quantity")
        if quantity is not None and quantity.socket_type == "NodeSocketString":
            name = _value(modifier, quantity)
            if name in meta:
                values["Value"] = meta[name]
                values["Units"] = meta.get(f"{name} Units", "")
        for name, value in values.items():
            item = inputs.get(name)
            if item is None or item.name in ("Image", "Probe", "Quantity"):
                continue
            socket = getattr(modifier.properties.inputs, item.identifier)
            try:
                if socket.value != value:
                    socket.value = value
            except TypeError:
                pass  # a same-named input of another type


VALUES_KEY = "yt_values"


def write_values(obj, scalars):
    """Put `scalars` in `obj["yt_values"]`, for drivers; tags `obj` if any changed."""
    if VALUES_KEY not in obj:
        obj[VALUES_KEY] = {}
    group = obj[VALUES_KEY]
    changed = False
    for name, value in scalars.items():
        if group.get(name) != float(value):
            group[name] = float(value)
            changed = True
    if changed:
        obj.update_tag()


def update_probe(scene, obj, frame=None, force=False):
    """Bring `obj`'s Image up to date; True if it was rewritten.

    Needs the probe's volume rendered once this session, which loads its
    dataset and fixes its registration.
    """
    from . import operators, yt_idv_renderer

    props = obj.yt_probe
    if props.volume is None:
        raise ValueError(f"probe {obj.name!r} has no volume")
    volume = operators.build_volume_params(props.volume)
    ds = yt_idv_renderer.loaded_dataset(volume.dataset.dataset_id)
    if ds is None:
        raise LookupError(f"probe {obj.name!r}: render {props.volume.name!r} once first, to load its dataset")
    A, b = yt_idv_renderer.world_to_unitary(volume)
    center, radius = sphere(obj)
    center_unitary = A @ center + b
    radius_unitary = radius * float(np.cbrt(abs(np.linalg.det(A))))
    frame = frame or operators.build_frame_params(scene)
    params = {item.key: item.value for item in props.params if item.key}
    probe_id = _probe_id(obj)
    key = probe_analysis.probe_key(
        volume.dataset.version.hash, center_unitary, radius_unitary, props.analysis, params,
        frame.frame_number if props.update == "EVERY_FRAME" else None,
        loaded=f"{id(ds)}:{getattr(ds, 'unique_identifier', '')}",
    )
    if force or (props.image is None and not obj.get(VALUES_KEY)):
        probe_analysis.forget(probe_id)

    contexts = []

    def make_context():
        center_code = ds.arr(center_unitary, "unitary").to("code_length")
        radius_code = ds.quan(radius_unitary, "unitary").to("code_length")
        ctx = probe_analysis.ProbeContext(
            ds=ds, sphere=ds.sphere(center_code, radius_code), center=center_code, radius=radius_code,
            frame=frame, params=params, state=json.loads(props.state or "{}"),
        )
        contexts.append(ctx)
        return ctx

    result, fresh = probe_analysis.run(probe_id, key, make_context, props.analysis)
    if contexts:
        state = json.dumps(contexts[0].state, sort_keys=True)
        if state != props.state:
            props.state = state
    if not fresh:
        return False
    image = write_image(obj, result.values) if result.values is not None else props.image
    write_values(obj, result.scalars)
    _sync_meta(obj, image, {**result.meta, **result.scalars, "Probe Radius": float(radius)})
    return True


def update_probes(scene, rendered, frame, log=print):
    """Update the scene's probes on `rendered` volumes, except Manual ones."""
    for obj in probes(scene):
        props = obj.yt_probe
        if props.update == "MANUAL" or props.volume is None:
            continue
        if props.volume not in rendered:
            log(f"probe {obj.name!r}: skipped, {props.volume.name!r} isn't rendered")
            continue
        if update_probe(scene, obj, frame):
            log(f"probe {obj.name!r}: updated" + (f" {props.image.name!r}" if props.image is not None else ""))


def bake(scene, obj, frames):
    """Run the probe on each of `frames` and keyframe its values there, so
    drivers on them have each frame's value before that frame renders.
    Returns the keyframed names.
    """
    current = scene.frame_current
    names = set()
    try:
        for frame in frames:
            scene.frame_set(frame)
            update_probe(scene, obj)
            for name in obj.get(VALUES_KEY, {}).keys():
                obj.keyframe_insert(f'["{VALUES_KEY}"]["{name}"]', frame=frame, group="yt Probe")
                names.add(name)
    finally:
        scene.frame_set(current)
    return sorted(names)


def reset(obj):
    """Drop what the analysis fitted, so the next run fits again."""
    obj.yt_probe.state = ""
    probe_analysis.clear_cache()
    probe_analysis.forget(_probe_id(obj))


# kind -> (label, analysis, panel group, params, update)
KINDS = {
    "PHASE": ("Phase Diagram", DEFAULT_ANALYSIS, PANEL_GROUP, {}, "ON_CHANGE"),
    "HISTORY": ("Time History", "yt_volumes.probe_analysis:history", LINE_GROUP,
                {"fields": "gas, temperature", "x": "frame"}, "EVERY_FRAME"),
    "VALUES": ("Values", "yt_volumes.probe_analysis:quantities", LABEL_GROUP,
               {"fields": "gas, temperature"}, "ON_CHANGE"),
}


def add_probe(context, volume, kind="PHASE"):
    """A Sphere Empty probing `volume`, at the 3D cursor, and an annotation
    showing it in the annotation scene ("yt Image Panel" for a phase diagram,
    "yt Line Panel" for a time history, "yt Probe Label" for values).
    Returns the probe.
    """
    label, analysis, group, params, update = KINDS[kind]
    scene = annotations.main_scene(context.scene)
    obj = bpy.data.objects.new(f"{volume.name} {label} Probe", None)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = 0.1 * max(volume.empty_display_size * max(volume.matrix_world.to_scale()), 1e-6)
    obj.location = scene.cursor.location
    scene.collection.objects.link(obj)
    props = obj.yt_probe
    props.is_probe = True
    props.volume = volume
    props.analysis = analysis
    props.update = update
    if kind == "HISTORY":
        params = {**params, "x_range": f"{scene.frame_start}, {scene.frame_end}"}
    for key, value in params.items():
        item = props.params.add()
        item.key, item.value = key, value
    panel = annotations.add_from_library(scene, group, volume, probe=obj)
    panel.name = f"{obj.name} {group.removeprefix('yt ')}"
    if group != LABEL_GROUP:
        props.image = new_image(f"yt Probe {obj.name}")
        annotations._set_input(panel.modifiers[0], "Image", props.image)
    return obj


def active_probe(context):
    obj = context.object
    return obj if obj is not None and obj.yt_probe.is_probe else None


class VOLUMEOVERRIDE_OT_add_probe(bpy.types.Operator):
    """Add a sphere that analyses the active volume's data, shown in the annotation scene"""

    bl_idname = "volume_override.add_probe"
    bl_label = "Add Probe"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(
        name="Kind",
        items=(
            ("PHASE", "Phase Diagram", "Gas mass over density and temperature, as a colormapped plot"),
            ("HISTORY", "Time History", "Mass-weighted mean temperature over the frames, as a line plot"),
            ("VALUES", "Values", "Mass-weighted mean temperature, as a label and a value for drivers"),
        ),
    )

    @classmethod
    def poll(cls, context):
        return annotations.active_volume(context) is not None

    def execute(self, context):
        obj = add_probe(context, annotations.active_volume(context), self.kind)
        for selected in context.selected_objects:
            selected.select_set(False)
        if obj.name in context.view_layer.objects:
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_update_probe(bpy.types.Operator):
    """Run the active probe's analysis now and rewrite its Image"""

    bl_idname = "volume_override.update_probe"
    bl_label = "Update Probe"

    @classmethod
    def poll(cls, context):
        return active_probe(context) is not None

    def execute(self, context):
        try:
            update_probe(annotations.main_scene(context.scene), active_probe(context), force=True)
        except (LookupError, ValueError, NotImplementedError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_bake_probe(bpy.types.Operator):
    """Run the active probe on every frame of the scene's range and keyframe its values"""

    bl_idname = "volume_override.bake_probe"
    bl_label = "Bake Values"

    @classmethod
    def poll(cls, context):
        return active_probe(context) is not None

    def execute(self, context):
        scene = annotations.main_scene(context.scene)
        try:
            names = bake(scene, active_probe(context), range(scene.frame_start, scene.frame_end + 1))
        except (LookupError, ValueError, NotImplementedError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not names:
            self.report({"WARNING"}, "The analysis returned no values to keyframe")
        else:
            self.report({"INFO"}, f"Keyframed {', '.join(names)} over frames {scene.frame_start}-{scene.frame_end}")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_reset_probe(bpy.types.Operator):
    """Forget the ranges the active probe's analysis fitted; the next run fits them again"""

    bl_idname = "volume_override.reset_probe"
    bl_label = "Reset Fitted Ranges"

    @classmethod
    def poll(cls, context):
        return active_probe(context) is not None

    def execute(self, context):
        reset(active_probe(context))
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_probe_param(bpy.types.Operator):
    bl_idname = "volume_override.add_probe_param"
    bl_label = "Add Probe Parameter"

    def execute(self, context):
        context.object.yt_probe.params.add()
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_remove_probe_param(bpy.types.Operator):
    bl_idname = "volume_override.remove_probe_param"
    bl_label = "Remove Probe Parameter"

    def execute(self, context):
        props = context.object.yt_probe
        if 0 <= props.params_active_index < len(props.params):
            props.params.remove(props.params_active_index)
        return {"FINISHED"}


class VOLUMEOVERRIDE_PT_probe(bpy.types.Panel):
    bl_idname = "VOLUMEOVERRIDE_PT_probe"
    bl_label = "yt Probe"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "EMPTY"

    def draw_header(self, context):
        self.layout.prop(context.object.yt_probe, "is_probe", text="")

    def draw(self, context):
        obj = context.object
        props = obj.yt_probe
        layout = self.layout
        layout.enabled = props.is_probe
        if obj.empty_display_type != "SPHERE":
            layout.label(text="Set Display As to Sphere to see the probed region", icon="INFO")
        layout.prop(props, "volume")
        layout.prop(props, "analysis")
        layout.prop(props, "update")
        layout.template_ID(props, "image")
        layout.label(text="Parameters:")
        row = layout.row()
        row.template_list("VOLUMEOVERRIDE_UL_custom_params", "probe", props, "params", props, "params_active_index", rows=3)
        sub = row.column(align=True)
        sub.operator(VOLUMEOVERRIDE_OT_add_probe_param.bl_idname, icon="ADD", text="")
        sub.operator(VOLUMEOVERRIDE_OT_remove_probe_param.bl_idname, icon="REMOVE", text="")
        row = layout.row(align=True)
        row.operator(VOLUMEOVERRIDE_OT_update_probe.bl_idname, icon="FILE_REFRESH")
        row.operator(VOLUMEOVERRIDE_OT_bake_probe.bl_idname, icon="KEYINGSET")
        row.operator(VOLUMEOVERRIDE_OT_reset_probe.bl_idname, icon="LOOP_BACK")
        values = obj.get(VALUES_KEY)
        if values:
            box = layout.box()
            box.label(text='Values (drivers: ["yt_values"]["name"]):')
            for name in values.keys():
                box.prop(obj, f'["{VALUES_KEY}"]["{name}"]', text=name)


def _draw_add_menu(self, context):
    for kind, (label, *_rest) in KINDS.items():
        self.layout.operator(VOLUMEOVERRIDE_OT_add_probe.bl_idname, text=f"Probe: {label}", icon="SPHERE").kind = kind


_CLASSES = (
    YtProbeProperties, VOLUMEOVERRIDE_OT_add_probe, VOLUMEOVERRIDE_OT_update_probe, VOLUMEOVERRIDE_OT_bake_probe,
    VOLUMEOVERRIDE_OT_reset_probe,
    VOLUMEOVERRIDE_OT_add_probe_param, VOLUMEOVERRIDE_OT_remove_probe_param, VOLUMEOVERRIDE_PT_probe,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.yt_probe = bpy.props.PointerProperty(type=YtProbeProperties)
    annotations.VOLUMEOVERRIDE_MT_add_annotation.append(_draw_add_menu)


def unregister():
    annotations.VOLUMEOVERRIDE_MT_add_annotation.remove(_draw_add_menu)
    del bpy.types.Object.yt_probe
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
