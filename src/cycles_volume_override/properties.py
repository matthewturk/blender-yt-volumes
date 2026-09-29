"""Marks objects for external volume rendering and holds the parameters that
get passed straight through as `VolumeObjectParams.custom_params`, plus the
dataset identity fields consumed by `operators.build_dataset_params()` (see
change_tracking.py and README "Dataset / change tracking").

A handful of common volume parameters are exposed as first-class fields (so
they get a normal, typed UI) plus a free-form string key/value table for
anything the real renderer needs that we haven't anticipated -- both feed
into the same `custom_params` dict, so growing the renderer's parameter set
never requires touching this add-on's Python.

`dataset_path`/`dataset_kind` are deliberately generic, not VDB-specific:
the end goal is arbitrary scientific datasets (e.g. a yt-loadable simulation
output), not just Blender-native volumetric formats. `dataset_kind` is a
free-form label (`"yt"`, `"vdb"`, `"hdf5"`, whatever your renderer's loaders
recognize) -- Blender doesn't interpret it, it's just carried through to
`DatasetParams.dataset_kind` for the renderer to dispatch on. Anything a
specific loader needs beyond a path (a unit system override, an HDF5
dataset key) goes in `custom_params_extra` below, the same free-form table
as any other renderer parameter.

`yt_idv_volume_provider` is the one exception that DOES get its own typed
field despite that "use custom_params_extra" rule -- see its own docstring
and `yt_idv_renderer.py`'s "Custom volume setup": a plain path/kind string
can't express real yt dataset setup (field selection, derived fields, cut
regions), so this points at a Python function you write instead. Still
just another `custom_params` entry once built (`to_custom_params()` below),
same as every other field here -- the typed field is purely for UI
discoverability given how central it is for real yt_idv use.
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
            "Render this object's region via the external volume renderer "
            "instead, composited back in via depth-clipped Alpha Over. "
            "Requires an Empty with Display Type 'Cube' -- see "
            "operators.build_volume_params's docstring for why (an Empty "
            "never renders in Cycles at all, so there's no need to hide it "
            "for pass 1, and a real mesh's own bound_box would double as "
            "the volume's registered extent even though that's not what "
            "you'd want)"
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
        # Loud, obviously-synthetic red by default -- deliberately NOT a
        # plausible "real" volume color -- so a flagged object's placeholder
        # output (DEBUG_FLAT / SAMPLE_CONSTANT renderer modes; see
        # operators.RENDERER_MODES) is unmistakable against Cycles' own
        # render while debugging the pipeline. Change it once you're past
        # "is this wired up at all" and into "does this look right".
        default=(1.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
    )
    opacity: bpy.props.FloatProperty(
        name="Opacity",
        description=(
            "Overall opacity multiplier applied to this object's rendered "
            "result AFTER the external renderer runs, regardless of which "
            "renderer_mode is active -- NOT passed through custom_params "
            "for a renderer to interpret itself (unlike density_scale/"
            "absorption/scattering/color above); see "
            "operators.render_current_frame's own comment for why applying "
            "it uniformly here, once, covers every renderer (including "
            "future ones) rather than needing each renderer to remember to "
            "read a custom_params key. Lives on the OBJECT (a real "
            "keyframable ID), unlike the yt_idv Preferences-panel alpha "
            "controls (Fog/MIP Alpha Max, Ramp Power, the curve widget) -- "
            "those live on AddonPreferences, which has no animation_data "
            "at all and categorically can't be keyframed (confirmed: "
            "`id_data` is None there). This is the one you keyframe for a "
            "fade in/out"
        ),
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    dataset_path: bpy.props.StringProperty(
        name="Dataset Path",
        description=(
            "Path to the actual scientific dataset this object represents -- "
            "a yt-loadable simulation output, an HDF5/VDB file, a directory "
            "of timestep files, whatever your renderer's loader expects. "
            "See DatasetParams.dataset_kind for how the renderer is told "
            "what KIND of thing this path points to"
        ),
        subtype="FILE_PATH",
        default="",
    )
    dataset_kind: bpy.props.StringProperty(
        name="Dataset Kind",
        description=(
            "Free-form label for which loader the renderer should use for "
            "dataset_path (e.g. 'yt', 'vdb', 'hdf5') -- not interpreted by "
            "Blender at all, just carried through to DatasetParams.dataset_kind"
        ),
        default="",
    )
    yt_idv_volume_provider: bpy.props.StringProperty(
        name="yt_idv Volume Provider",
        description=(
            "Optional, yt_idv renderer_modes only: 'module.submodule:"
            "function_name' pointing at a Python function YOU write, called "
            "once per frame, that sets up (or confirms/reuses) the yt_idv "
            "Scene for THIS object -- lets you load a real dataset, pick a "
            "field, build derived fields/cut regions, anything yt can do, "
            "instead of the hardcoded IsolatedGalaxy sample every yt_idv "
            "render otherwise uses. See yt_idv_renderer.py's module "
            "docstring ('Custom volume setup') and its VolumeSetupContext/"
            "VolumeSetupResult for the exact contract. Passed through as "
            "custom_params['yt_idv_volume_provider']; ignored by every "
            "non-yt_idv renderer_mode, and by yt_idv modes too if left "
            "empty (falls back to the hardcoded sample)"
        ),
        default="",
    )
    dataset_uuid: bpy.props.StringProperty(
        name="Dataset ID",
        description=(
            "Stable per-object identifier for change-tracking cache keys, "
            "auto-generated once and stored in the .blend file (survives "
            "renaming the object, unlike using its name as the key). "
            "Read-only -- see change_tracking.py"
        ),
        default="",
    )

    # --- Blender frame -> dataset time mapping ---
    # See operators.resolve_requested_time() and README "Dataset / change
    # tracking" -> "Time mapping". Blender only resolves a single requested
    # scalar and hands it over as DatasetParams.requested_time -- snapping
    # that to an actual timestep the dataset has is the renderer's job (it's
    # the one that knows what timesteps exist), not Blender's.
    mapping_policy: bpy.props.EnumProperty(
        name="Time Mapping",
        description="How the current Blender frame maps to DatasetParams.requested_time",
        items=[
            ("DIRECT", "Direct", "requested_time = current Blender frame number, unchanged"),
            ("AFFINE", "Affine", "requested_time = frame * Time Scale + Time Offset"),
            ("MANUAL", "Manual", "requested_time = Manual Time below -- keyframe or drive it yourself"),
        ],
        default="DIRECT",
    )
    time_units: bpy.props.EnumProperty(
        name="Time Units",
        description="What DatasetParams.requested_time actually means, for the renderer to interpret correctly",
        items=[
            ("FRAME_INDEX", "Frame Index", "An integer-ish index into the dataset's own timestep list"),
            ("SECONDS", "Seconds", "Physical/simulation time in seconds"),
            ("REDSHIFT", "Redshift", "Cosmological redshift"),
            ("CUSTOM", "Custom", "Some other convention your renderer/loader defines"),
        ],
        default="FRAME_INDEX",
    )
    time_scale: bpy.props.FloatProperty(
        name="Time Scale",
        description="AFFINE policy only: requested_time = frame * time_scale + time_offset",
        default=1.0,
    )
    time_offset: bpy.props.FloatProperty(
        name="Time Offset",
        description="AFFINE policy only: requested_time = frame * time_scale + time_offset",
        default=0.0,
    )
    manual_time: bpy.props.FloatProperty(
        name="Manual Time",
        description=(
            "MANUAL policy only: requested_time is read directly from this "
            "property. Free to animate however you like -- keyframe it "
            "directly (simplest, most robust) or drive it with a Driver. If "
            "you use a SCRIPTED driver expression here, note Blender "
            "disables SCRIPTED drivers entirely when 'Auto Run Python "
            "Scripts' is off, which can silently break this for anyone "
            "else opening the file; a non-scripted driver type (Average/"
            "Sum/Min/Max) or plain keyframes don't have that risk"
        ),
        default=0.0,
    )

    custom_params_extra: bpy.props.CollectionProperty(type=VolumeOverrideCustomParam)
    custom_params_extra_active_index: bpy.props.IntProperty(default=0)

    def to_custom_params(self) -> dict:
        """Build the `VolumeObjectParams.custom_params` dict for this object.

        Deliberately does NOT include dataset_path/dataset_kind -- those are
        dataset *identity*, handled separately by
        `operators.build_dataset_params()` -> `DatasetParams`, with their own
        change-tracking hash. Everything here is a "cheap to recompute,
        doesn't require reloading the dataset" shading knob.
        """
        params = {
            "density_scale": self.density_scale,
            "absorption": tuple(self.absorption),
            "scattering": tuple(self.scattering),
            "color": tuple(self.color),
        }
        if self.yt_idv_volume_provider:
            params["yt_idv_volume_provider"] = self.yt_idv_volume_provider
        for item in self.custom_params_extra:
            if item.key:
                params[item.key] = item.value
        return params


def ensure_dataset_uuid(obj) -> str:
    """Generate and persist `dataset_uuid` on first use, then always return
    it. Called from `operators.build_dataset_params()` -- not eagerly on
    flagging, so toggling the flag on/off/on before ever rendering doesn't
    burn through UUIDs for no reason.
    """
    props = obj.volume_override
    if not props.dataset_uuid:
        import uuid
        props.dataset_uuid = uuid.uuid4().hex
    return props.dataset_uuid


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
        obj = context.object
        props = obj.volume_override
        layout.prop(props, "is_volume_override")

        if props.is_volume_override and (obj.type != "EMPTY" or obj.empty_display_type != "CUBE"):
            box = layout.box()
            box.alert = True
            box.label(text="Must be an Empty with Display Type 'Cube'", icon="ERROR")
            box.label(text=f"(this is a {obj.type.title()}) -- rendering will raise an error")

        col = layout.column()
        col.enabled = props.is_volume_override
        col.prop(props, "density_scale")
        col.prop(props, "absorption")
        col.prop(props, "scattering")
        col.prop(props, "color")
        col.prop(props, "opacity")
        col.separator()
        col.prop(props, "dataset_path")
        col.prop(props, "dataset_kind")
        col.prop(props, "yt_idv_volume_provider")
        if props.dataset_uuid:
            row = col.row()
            row.enabled = False
            row.prop(props, "dataset_uuid", text="Dataset ID")

        col.separator()
        col.prop(props, "mapping_policy")
        col.prop(props, "time_units")
        if props.mapping_policy == "AFFINE":
            col.prop(props, "time_scale")
            col.prop(props, "time_offset")
        elif props.mapping_policy == "MANUAL":
            col.prop(props, "manual_time")

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

        col.operator("volume_override.reset_change_tracking", icon="FILE_REFRESH")


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
