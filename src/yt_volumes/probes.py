"""Probes: Sphere Empties that run a yt analysis on the data inside them and
put the result in a float Image, shown in the annotation scene by "yt Image
Panel". The pipeline updates them after the volumes render, before pass 2.
"""

import json

import bpy
import numpy as np

from . import annotations, probe_analysis
from .properties import VolumeOverrideCustomParam

PANEL_GROUP = "yt Image Panel"
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


def panels(image):
    """`(object, modifier)` pairs whose Geometry Nodes Image input is `image`."""
    found = []
    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            for item in modifier.node_group.interface.items_tree:
                if (item.item_type == "SOCKET" and item.in_out == "INPUT" and item.socket_type == "NodeSocketImage"
                        and getattr(modifier.properties.inputs, item.identifier).value == image):
                    found.append((obj, modifier))
                    break
    return found


def _sync_meta(image, meta):
    """Copy `meta` onto `image` and onto same-named inputs of panels showing it."""
    for key, value in meta.items():
        if image.get(key) != value:
            image[key] = value
    for _obj, modifier in panels(image):
        for item in modifier.node_group.interface.items_tree:
            if item.item_type != "SOCKET" or item.in_out != "INPUT" or item.name not in meta:
                continue
            socket = getattr(modifier.properties.inputs, item.identifier)
            if socket.value != meta[item.name]:
                socket.value = meta[item.name]


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
    )
    if force or props.image is None:
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
    _sync_meta(write_image(obj, result.values), result.meta)
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
            log(f"probe {obj.name!r}: wrote {props.image.name!r}")


def reset(obj):
    """Drop what the analysis fitted, so the next run fits again."""
    obj.yt_probe.state = ""
    probe_analysis.clear_cache()
    probe_analysis.forget(_probe_id(obj))


def add_probe(context, volume):
    """A Sphere Empty probing `volume`, at the 3D cursor, and a "yt Image
    Panel" showing its Image in the annotation scene. Returns the probe.
    """
    scene = annotations.main_scene(context.scene)
    obj = bpy.data.objects.new(f"{volume.name} Probe", None)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = 0.1 * max(volume.empty_display_size * max(volume.matrix_world.to_scale()), 1e-6)
    obj.location = scene.cursor.location
    scene.collection.objects.link(obj)
    props = obj.yt_probe
    props.is_probe = True
    props.volume = volume
    props.image = new_image(f"yt Probe {obj.name}")
    panel = annotations.add_from_library(scene, PANEL_GROUP)
    panel.name = f"{obj.name} Panel"
    annotations._set_input(panel.modifiers[0], "Image", props.image)
    return obj


def active_probe(context):
    obj = context.object
    return obj if obj is not None and obj.yt_probe.is_probe else None


class VOLUMEOVERRIDE_OT_add_probe(bpy.types.Operator):
    """Add a sphere that runs a phase diagram on the active volume's data, shown in the annotation scene"""

    bl_idname = "volume_override.add_probe"
    bl_label = "Add Probe"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return annotations.active_volume(context) is not None

    def execute(self, context):
        obj = add_probe(context, annotations.active_volume(context))
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
        row.operator(VOLUMEOVERRIDE_OT_reset_probe.bl_idname, icon="LOOP_BACK")


def _draw_add_menu(self, context):
    self.layout.operator(VOLUMEOVERRIDE_OT_add_probe.bl_idname, icon="SPHERE")


_CLASSES = (
    YtProbeProperties, VOLUMEOVERRIDE_OT_add_probe, VOLUMEOVERRIDE_OT_update_probe, VOLUMEOVERRIDE_OT_reset_probe,
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
