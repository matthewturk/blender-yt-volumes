"""Marks objects for external volume rendering and holds the parameters that
get passed straight through as `VolumeObjectParams.custom_params`.

A handful of common volume parameters are exposed as first-class fields (so
they get a normal, typed UI) plus a free-form string key/value table for
anything the real renderer needs that we haven't anticipated -- both feed
into the same `custom_params` dict, so growing the renderer's parameter set
never requires touching this add-on's Python.
"""

import bpy

CUSTOM_PARAM_KEY_MAXLEN = 63  # bpy.props.StringProperty has no built-in limit;
#                                keep collection items sane-sized for display.


class VolumeOverrideCustomParam(bpy.types.PropertyGroup):
    """One free-form key/value pair, merged into custom_params by name."""

    key: bpy.props.StringProperty(name="Key", maxlen=CUSTOM_PARAM_KEY_MAXLEN)
    value: bpy.props.StringProperty(name="Value")


class VolumeOverrideProperties(bpy.types.PropertyGroup):
    """Attached to every Object as `obj.volume_override`."""

    is_volume_override: bpy.props.BoolProperty(
        name="Render with External Volume Renderer",
        description=(
            "Hide this object from the Cycles pass and render it instead "
            "via the external volume renderer, composited back in via "
            "depth-clipped Alpha Over"
        ),
        default=False,
    )

    density_scale: bpy.props.FloatProperty(
        name="Density Scale",
        description="Passed through as custom_params['density_scale']",
        default=1.0,
        min=0.0,
    )
    absorption: bpy.props.FloatVectorProperty(
        name="Absorption Coefficient",
        description="Passed through as custom_params['absorption'] (RGB)",
        subtype="COLOR",
        size=3,
        default=(0.1, 0.1, 0.1),
        min=0.0,
    )
    scattering: bpy.props.FloatVectorProperty(
        name="Scattering Coefficient",
        description="Passed through as custom_params['scattering'] (RGB)",
        subtype="COLOR",
        size=3,
        default=(0.5, 0.5, 0.5),
        min=0.0,
    )
    color: bpy.props.FloatVectorProperty(
        name="Color",
        description="Passed through as custom_params['color'] (RGB)",
        subtype="COLOR",
        size=3,
        default=(0.4, 0.6, 1.0),
        min=0.0,
        max=1.0,
    )
    vdb_filepath: bpy.props.StringProperty(
        name="VDB / Grid File",
        description="Passed through as custom_params['vdb_filepath']",
        subtype="FILE_PATH",
        default="",
    )

    custom_params_extra: bpy.props.CollectionProperty(type=VolumeOverrideCustomParam)
    custom_params_extra_active_index: bpy.props.IntProperty(default=0)

    def to_custom_params(self) -> dict:
        """Build the `VolumeObjectParams.custom_params` dict for this object."""
        params = {
            "density_scale": self.density_scale,
            "absorption": tuple(self.absorption),
            "scattering": tuple(self.scattering),
            "color": tuple(self.color),
            "vdb_filepath": self.vdb_filepath,
        }
        for item in self.custom_params_extra:
            if item.key:
                params[item.key] = item.value
        return params


class VOLUMEOVERRIDE_UL_custom_params(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "key", text="", emboss=False)
        row.prop(item, "value", text="", emboss=False)


class VOLUMEOVERRIDE_OT_add_custom_param(bpy.types.Operator):
    bl_idname = "volume_override.add_custom_param"
    bl_label = "Add Custom Parameter"

    def execute(self, context):
        context.object.volume_override.custom_params_extra.add()
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_remove_custom_param(bpy.types.Operator):
    bl_idname = "volume_override.remove_custom_param"
    bl_label = "Remove Custom Parameter"

    def execute(self, context):
        props = context.object.volume_override
        idx = props.custom_params_extra_active_index
        if 0 <= idx < len(props.custom_params_extra):
            props.custom_params_extra.remove(idx)
        return {"FINISHED"}


class VOLUMEOVERRIDE_PT_object_panel(bpy.types.Panel):
    bl_idname = "VOLUMEOVERRIDE_PT_object_panel"
    bl_label = "Volume Override"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def draw(self, context):
        layout = self.layout
        props = context.object.volume_override
        layout.prop(props, "is_volume_override")

        col = layout.column()
        col.enabled = props.is_volume_override
        col.prop(props, "density_scale")
        col.prop(props, "absorption")
        col.prop(props, "scattering")
        col.prop(props, "color")
        col.prop(props, "vdb_filepath")

        col.label(text="Extra custom_params:")
        row = col.row()
        row.template_list(
            "VOLUMEOVERRIDE_UL_custom_params", "",
            props, "custom_params_extra",
            props, "custom_params_extra_active_index",
            rows=3,
        )
        sub = row.column(align=True)
        sub.operator("volume_override.add_custom_param", icon="ADD", text="")
        sub.operator("volume_override.remove_custom_param", icon="REMOVE", text="")


CLASSES = (
    VolumeOverrideCustomParam,
    VolumeOverrideProperties,
    VOLUMEOVERRIDE_UL_custom_params,
    VOLUMEOVERRIDE_OT_add_custom_param,
    VOLUMEOVERRIDE_OT_remove_custom_param,
    VOLUMEOVERRIDE_PT_object_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.volume_override = bpy.props.PointerProperty(type=VolumeOverrideProperties)


def unregister():
    del bpy.types.Object.volume_override
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def flagged_objects(scene):
    """All objects in `scene` flagged for external volume rendering."""
    return [obj for obj in scene.objects if getattr(obj, "volume_override", None) and obj.volume_override.is_volume_override]
