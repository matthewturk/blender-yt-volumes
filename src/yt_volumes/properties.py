"""Per-object properties: the external-volume flag, the render look (renderer
mode and its settings, passed through as `VolumeObjectParams.custom_params`),
and dataset identity used by `operators.build_dataset_params()`.

Typed fields plus the free-form key/value table all end up in the same
`custom_params` dict. `dataset_kind` is a free-form label for the renderer
to dispatch on; Blender doesn't interpret it.
"""

import bpy
import numpy as np

from . import transfer_function_nodes

CUSTOM_PARAM_KEY_MAXLEN = 63


class VolumeOverrideCustomParam(bpy.types.PropertyGroup):
    """One free-form key/value pair, merged into custom_params by name."""

    key: bpy.props.StringProperty(name="Key", maxlen=CUSTOM_PARAM_KEY_MAXLEN)
    value: bpy.props.StringProperty(name="Value")


# Settings that set the look, in custom_params as `yt_idv_<name>` (render_volume's
# [render.yt_idv] table may set any of them).
LOOK_SETTINGS = (
    "backend", "samples_per_cell", "colormap", "rescale_colormap",
    "tf_fit_range", "tf_percentile_low", "tf_percentile_high", "tf_log_min", "tf_log_max",
    "tf_n_layers", "tf_layer_width", "tf_alpha_first", "tf_alpha_last", "tf_colormap",
    "fog_value_min", "fog_value_max", "fog_colormap", "fog_alpha_max", "fog_ramp_power",
    "mip_alpha_colormap", "mip_alpha_max", "mip_alpha_ramp_power",
    "yt_tf", "yt_tf_sample_factor", "yt_tf_ray_length",
    "gn_tf_fit_range", "gn_tf_percentile_low", "gn_tf_percentile_high",
    "gn_tf_value_min", "gn_tf_value_max", "gn_tf_opacity_scale",
    "alpha_source", "color_source",
)
ALPHA_CURVE_NODE_NAME = "AlphaCurve"


def _forget_colormap_range(self, context):
    # A range fitted in one mode (e.g. Projection's large sums) is wrong for another.
    self.colormap_range_set = False


def _ensure_alpha_curve(self, context):
    if self.alpha_source != "CURVE" or self.alpha_curve is not None:
        return
    tree = bpy.data.node_groups.new("VolumeOverride Alpha Curve", "ShaderNodeTree")
    node = tree.nodes.new("ShaderNodeFloatCurve")
    node.name = ALPHA_CURVE_NODE_NAME
    node.mapping.initialize()
    self.alpha_curve = tree


def _viewport_preview_toggled(self, context):
    from . import viewport_preview

    viewport_preview.restart()


def _redraw_viewports(self, context):
    for area in context.screen.areas if context.screen else ():
        if area.type == "VIEW_3D":
            area.tag_redraw()


class VolumeOverrideProperties(bpy.types.PropertyGroup):
    """Attached to every Object as `obj.volume_override`."""

    is_volume_override: bpy.props.BoolProperty(
        name="Render with External Volume Renderer",
        description=(
            "Render this object's region via the external volume renderer "
            "instead, composited back in via depth-clipped Alpha Over. "
            "Requires an Empty with Display Type 'Cube' ('Sphere' for "
            "spherical data) -- see "
            "operators.build_volume_params's docstring for why (an Empty "
            "never renders at all, so there's no need to hide it "
            "for pass 1, and a real mesh's own bound_box would double as "
            "the volume's registered extent even though that's not what "
            "you'd want)"
        ),
        default=False,
    )

    show_in_viewport: bpy.props.BoolProperty(
        name="Show in Viewport",
        description=(
            "Draw this volume over the 3D viewport with yt_idv, from each viewport's view, occluded by "
            "the scene; the first draw loads the data unless a render already did"
        ),
        default=False,
        update=_viewport_preview_toggled,
    )
    show_grids: bpy.props.BoolProperty(
        name="Show Grids in Viewport",
        description="Draw the dataset's grid boxes in the 3D viewport, once it has been rendered this session",
        default=False,
        update=_redraw_viewports,
    )
    grid_level_min: bpy.props.IntProperty(
        name="Min Level", description="Coarsest grid level drawn", default=0, min=0, update=_redraw_viewports,
    )
    grid_level_max: bpy.props.IntProperty(
        name="Max Level", description="Finest grid level drawn", default=99, min=0, update=_redraw_viewports,
    )
    grid_consolidate: bpy.props.BoolProperty(
        name="Consolidate",
        description="Draw a few bounding boxes around where the selected levels are, instead of every grid",
        default=False,
        update=_redraw_viewports,
    )
    grid_clusters: bpy.props.IntProperty(
        name="Boxes", description="Most bounding boxes to consolidate into", default=8, min=1, max=256,
        update=_redraw_viewports,
    )

    # --- Look: renderer mode and its settings (all end up in custom_params) ---
    renderer_mode: bpy.props.EnumProperty(
        name="Renderer",
        description="How yt_idv renders this volume",
        # .blend files store the number, so keep each item's number when reordering.
        items=[
            ("YT_IDV_MAX_INTENSITY", "Maximum Intensity Projection", "Keep the brightest sample along each ray", "NONE", 0),
            ("YT_IDV_MAX_INTENSITY_ALPHA", "Maximum Intensity, Value-Driven Alpha", "Maximum Intensity Projection with alpha driven by the value itself: transparent at the low end of what is visible, opaque at the high end", "NONE", 1),
            ("YT_IDV_TRANSFER_FUNCTION", "yt Layers", "Evenly spaced layers in log10 of the field, as yt's ColorTransferFunction.add_layers", "NONE", 2),
            ("YT_IDV_PROJECTION", "Projective Integration", "Unweighted line-of-sight integration: adds samples along each ray", "NONE", 3),
            ("YT_IDV_YT_TRANSFER_FUNCTION", "yt Transfer Function", "A yt ColorTransferFunction you build, converted to yt_idv's so it renders as yt would", "NONE", 4),
            ("YT_IDV_THRESHOLD_FOG", "Threshold Fog", "A continuous transfer function: transparent below a threshold, alpha ramping up above it", "NONE", 5),
            ("YT_IDV_GN_TRANSFER_FUNCTION", "Geometry Nodes Transfer Function", "A transfer function drawn with Geometry Nodes on this object: a Color Ramp and an opacity curve, or any nodes", "NONE", 7),
            ("SAMPLE_CONSTANT", "Debug: Depth x Constant", "No yt_idv: clipped path length through the box, for fast pipeline tests", "NONE", 6),
        ],
        default="YT_IDV_MAX_INTENSITY",
        update=_forget_colormap_range,
    )
    backend: bpy.props.EnumProperty(
        name="Backend",
        description="Offscreen OpenGL backend",
        items=[
            ("egl", "EGL", "Needs a working GPU + driver stack"),
            ("osmesa", "OSMesa (buggy)", "Software rendering; corrupts non-square renders"),
        ],
        default="egl",
    )
    samples_per_cell: bpy.props.FloatProperty(
        name="Samples per Cell",
        description=(
            "Ray samples per cell width. Too few shows as wood grain and chevrons where cells "
            "are large on screen, as in deep zooms; 8-16 for final frames"
        ),
        default=4.0, min=0.1, soft_max=32.0,
    )
    colormap: bpy.props.StringProperty(
        name="Colormap",
        description="Maximum Intensity and Projection: any yt/matplotlib colormap name. Empty keeps yt_idv's default",
        default="",
    )
    rescale_colormap: bpy.props.BoolProperty(
        name="Rescale Colormap Every Render",
        description="Re-fit the colormap range to each render's data; off keeps the remembered range",
        default=False,
    )
    colormap_range_set: bpy.props.BoolProperty(
        name="Remember Colormap Range",
        description="Render with Colormap Range below. Set by the first render after it's cleared; uncheck to re-fit",
        default=False,
    )
    # float64 would be nicer, but a float32 range is plenty for a colormap.
    colormap_range: bpy.props.FloatVectorProperty(
        name="Colormap Range",
        description="Colormap (min, max) used while Remember Colormap Range is on",
        size=2,
        default=(0.0, 1.0),
    )

    # yt layers (YT_IDV_TRANSFER_FUNCTION): ColorTransferFunction.add_layers
    tf_fit_range: bpy.props.BoolProperty(
        name="Fit to Data",
        description="Span the layers' transfer function, in log space, over the data between the Low and High percentiles by volume; off uses Min and Max (log10)",
        default=True,
    )
    tf_percentile_low: bpy.props.FloatProperty(
        name="Low %",
        description="Percentile of the data, by volume, at the transfer function's low end",
        default=50.0, min=0.0, max=100.0,
    )
    tf_percentile_high: bpy.props.FloatProperty(
        name="High %",
        description="Percentile of the data, by volume, at the transfer function's high end; 100 keeps the densest cell (the top layer sits a little inside it)",
        default=100.0, min=0.0, max=100.0,
    )
    tf_log_min: bpy.props.FloatProperty(
        name="Min (log10)",
        description="log10 of the field, in its own units, at the transfer function's low end; used with Fit to Data off",
        default=-28.0, precision=3,
    )
    tf_log_max: bpy.props.FloatProperty(
        name="Max (log10)",
        description="log10 of the field, in its own units, at the transfer function's high end; used with Fit to Data off",
        default=-22.0, precision=3,
    )
    tf_n_layers: bpy.props.IntProperty(
        name="Layers",
        description="Number of layers, evenly spaced in log10 just inside the transfer function's range",
        default=8, min=1, max=64,
    )
    tf_layer_width: bpy.props.FloatProperty(
        name="Layer Width",
        description="Each layer's Gaussian width, add_layers' w, in (log10 of the field) squared; 0 uses yt's default, which is very thin",
        default=0.003, min=0.0, precision=4, step=0.1,
    )
    tf_alpha_first: bpy.props.FloatProperty(
        name="Alpha First",
        description="Opacity of the lowest layer; the layers between are spaced geometrically up to Alpha Last. Lower keeps the outer layers from hiding the inner ones",
        default=1.0, min=1e-4, max=1.0,
    )
    tf_alpha_last: bpy.props.FloatProperty(
        name="Alpha Last",
        description="Opacity of the highest layer",
        default=1.0, min=1e-4, max=1.0,
    )
    tf_colormap: bpy.props.StringProperty(
        name="Layers Colormap",
        description="Any yt/matplotlib colormap name, spanning the transfer function's range",
        default="arbre",
    )

    # Threshold fog (YT_IDV_THRESHOLD_FOG)
    fog_value_min: bpy.props.FloatProperty(
        name="Fog Threshold (Value Min)",
        description="Fully transparent below this value (yt_idv's [0, 1]-normalized value space)",
        default=1e-9, precision=10, min=0.0,
    )
    fog_value_max: bpy.props.FloatProperty(
        name="Fog Value Max",
        description="Alpha reaches Fog Alpha Max at this value",
        default=1e-2, precision=6, min=0.0,
    )
    fog_colormap: bpy.props.StringProperty(
        name="Fog Colormap",
        description="Any yt/matplotlib colormap name",
        default="dusk",
    )
    fog_alpha_max: bpy.props.FloatProperty(
        name="Fog Alpha Max",
        description="Alpha at Fog Value Max; below 1 keeps the densest gas soft",
        default=0.85, min=0.0, max=1.0,
    )
    fog_ramp_power: bpy.props.FloatProperty(
        name="Fog Ramp Power",
        description="Shape of the alpha ramp in log value: 1 is linear, below 1 rises fast past the threshold, above 1 stays thin longer",
        default=0.15, min=0.01, max=10.0,
    )

    # Maximum intensity with value-driven alpha (YT_IDV_MAX_INTENSITY_ALPHA)
    mip_alpha_colormap: bpy.props.StringProperty(
        name="MIP Alpha Colormap",
        description="Any yt/matplotlib colormap name",
        default="dusk",
    )
    mip_alpha_max: bpy.props.FloatProperty(
        name="MIP Alpha Max",
        description="Alpha at the high end of the visible range; below 1 keeps objects behind the brightest material visible",
        default=0.7, min=0.0, max=1.0,
    )
    mip_alpha_ramp_power: bpy.props.FloatProperty(
        name="MIP Alpha Ramp Power",
        description="Shape of the alpha ramp across the visible range, as for Fog Ramp Power",
        default=0.15, min=0.01, max=10.0,
    )

    # yt transfer function (YT_IDV_YT_TRANSFER_FUNCTION)
    yt_tf: bpy.props.StringProperty(
        name="yt Transfer Function",
        description="'module:function' or 'file.py:function' returning a yt ColorTransferFunction (grey opacity, log10 of the field)",
        default="",
    )
    yt_tf_sample_factor: bpy.props.FloatProperty(
        name="yt TF Sample Factor",
        description="Ray samples per cell for the yt modes; thin layers need more than the other modes. 0 uses Samples per Cell",
        default=16.0, min=0.0,
    )
    yt_tf_ray_length: bpy.props.FloatProperty(
        name="yt TF Ray Length",
        description="Divides the transfer function's emission and opacity, as a yt camera's ray length in unitary units; larger is fainter and more transparent",
        default=0.02, min=1e-9,
    )

    # Geometry Nodes transfer function (YT_IDV_GN_TRANSFER_FUNCTION)
    gn_tf_fit_range: bpy.props.BoolProperty(
        name="Fit to Data",
        description="Span the transfer function, in log space, over the data between the Low and High percentiles by volume; off uses TF Value Min and Max",
        default=True,
    )
    gn_tf_percentile_low: bpy.props.FloatProperty(
        name="Low %",
        description="Percentile of the data, by volume, at the transfer function's low end; a little above 0 drops the near-empty tail",
        default=1.0, min=0.0, max=100.0,
    )
    gn_tf_percentile_high: bpy.props.FloatProperty(
        name="High %",
        description="Percentile of the data, by volume, at the transfer function's high end; 100 keeps the densest cell",
        default=100.0, min=0.0, max=100.0,
    )
    gn_tf_opacity_scale: bpy.props.FloatProperty(
        name="Opacity Scale",
        description="Multiplies the transfer function's emission and opacity, per box width of path; raise it until small or thin structures show",
        default=100.0, min=0.0, soft_max=10000.0,
    )
    gn_tf_value_min: bpy.props.FloatProperty(
        name="TF Min (log10)",
        description="log10 of the field, in its own units, at the transfer function's low end; used with Fit to Data off",
        default=-28.0, precision=3,
    )
    gn_tf_value_max: bpy.props.FloatProperty(
        name="TF Max (log10)",
        description="log10 of the field, in its own units, at the transfer function's high end; used with Fit to Data off",
        default=-10.0, precision=3,
    )

    # Alpha curve / color image, shared by Threshold Fog and MIP Alpha
    alpha_source: bpy.props.EnumProperty(
        name="Alpha Source",
        description="How the alpha ramp for Threshold Fog / MIP Alpha is shaped",
        items=[
            ("RAMP_POWER", "Ramp Power", "The Ramp Power setting: a single-exponent curve"),
            ("CURVE", "Custom Curve", "A hand-drawn curve"),
        ],
        default="RAMP_POWER",
        update=_ensure_alpha_curve,
    )
    # A CurveMapping can't be a property itself, so it lives on a node in this group.
    alpha_curve: bpy.props.PointerProperty(name="Alpha Curve", type=bpy.types.NodeTree)
    color_source: bpy.props.EnumProperty(
        name="Color Source",
        description="Where Threshold Fog / MIP Alpha get their RGB",
        items=[
            ("NAMED", "Named Colormap", "The Fog/MIP Alpha Colormap setting"),
            ("IMAGE", "Image", "A Blender Image, averaged across rows into a 1D gradient"),
        ],
        default="NAMED",
    )
    color_image: bpy.props.PointerProperty(
        name="Color Image",
        description="Image used as the colormap when Color Source is Image",
        type=bpy.types.Image,
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
    # A string so the target keeps float64 precision; FloatVectorProperty is float32.
    zoom_target: bpy.props.StringProperty(
        name="Zoom Target",
        description=(
            "yt_idv modes: the point to zoom into, as 'x, y, z' in the "
            "dataset's unitary coordinates or 'max' for the field's maximum. "
            "Empty turns zooming off"
        ),
        default="",
    )
    zoom_level: bpy.props.FloatProperty(
        name="Zoom Level",
        description=(
            "Magnification about the Zoom Target as a power of 10 (3 = 1000x). "
            "Animate it linearly for a steady fly-in"
        ),
        default=0.0,
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

    custom_params_extra: bpy.props.CollectionProperty(type=VolumeOverrideCustomParam)
    custom_params_extra_active_index: bpy.props.IntProperty(default=0)

    def to_custom_params(self) -> dict:
        """Build the `VolumeObjectParams.custom_params` dict for this object.

        Excludes dataset_path/dataset_kind, which are tracked separately in
        `DatasetParams`, and the alpha curve / color image, which
        `operators.build_volume_params` samples.
        """
        params = {"renderer_mode": self.renderer_mode}
        params.update({f"yt_idv_{name}": getattr(self, name) for name in LOOK_SETTINGS})
        if self.colormap_range_set:
            params["yt_idv_cmap_range"] = tuple(self.colormap_range)
        if self.yt_idv_volume_provider:
            params["yt_idv_volume_provider"] = self.yt_idv_volume_provider
        if self.zoom_target:
            params["yt_idv_zoom_target"] = self.zoom_target
            params["yt_idv_zoom_level"] = self.zoom_level
        for item in self.custom_params_extra:
            if item.key:
                params[item.key] = item.value
        return params


def ensure_dataset_uuid(obj) -> str:
    """Return `dataset_uuid`, generating and storing it on first use."""
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


_MODE_SETTINGS = {
    "YT_IDV_MAX_INTENSITY": ("colormap",),
    "YT_IDV_PROJECTION": ("colormap",),
    "YT_IDV_TRANSFER_FUNCTION": (
        "tf_n_layers", "tf_layer_width", "tf_alpha_first", "tf_alpha_last", "tf_colormap",
        "tf_fit_range", "tf_percentile_low", "tf_percentile_high", "tf_log_min", "tf_log_max",
        "yt_tf_ray_length", "yt_tf_sample_factor",
    ),
    "YT_IDV_THRESHOLD_FOG": ("fog_value_min", "fog_value_max", "fog_colormap", "fog_alpha_max", "fog_ramp_power"),
    "YT_IDV_MAX_INTENSITY_ALPHA": ("mip_alpha_colormap", "mip_alpha_max", "mip_alpha_ramp_power"),
    "YT_IDV_YT_TRANSFER_FUNCTION": ("yt_tf", "yt_tf_ray_length", "yt_tf_sample_factor"),
    "YT_IDV_GN_TRANSFER_FUNCTION": (
        "gn_tf_opacity_scale", "gn_tf_fit_range", "gn_tf_percentile_low", "gn_tf_percentile_high",
        "gn_tf_value_min", "gn_tf_value_max",
    ),
}


def _draw_gn_transfer_function(layout, obj):
    from . import yt_idv_renderer

    fit = yt_idv_renderer.gn_tf_fit(obj.volume_override.dataset_uuid)
    if fit is not None:
        low, high = np.log10(fit["physical_range"])
        layout.label(text=f"log10 range: {low:.2f} to {high:.2f} {fit['units']}")
    ramp, curve = transfer_function_nodes.find_nodes(obj)
    if ramp is None and curve is None:
        layout.operator("volume_override.add_transfer_function", icon="ADD")
        return
    # The starter group's nodes, editable here; anything else in Geometry Nodes.
    if ramp is not None:
        layout.label(text="Color")
        layout.template_color_ramp(ramp, "color_ramp", expand=True)
    if curve is not None:
        layout.label(text="Opacity")
        layout.template_curve_mapping(curve, "mapping")


def _draw_look(layout, props):
    layout.prop(props, "renderer_mode")
    if props.renderer_mode == "SAMPLE_CONSTANT":
        return
    box = layout.box()
    box.prop(props, "backend")
    box.prop(props, "samples_per_cell")
    col = box.column(align=True)
    for name in _MODE_SETTINGS[props.renderer_mode]:
        row = col.row()
        if name in ("gn_tf_value_min", "gn_tf_value_max"):
            row.enabled = not props.gn_tf_fit_range
        elif name in ("gn_tf_percentile_low", "gn_tf_percentile_high"):
            row.enabled = props.gn_tf_fit_range
        elif name in ("tf_log_min", "tf_log_max"):
            row.enabled = not props.tf_fit_range
        elif name in ("tf_percentile_low", "tf_percentile_high"):
            row.enabled = props.tf_fit_range
        row.prop(props, name)
    box.prop(props, "rescale_colormap")
    row = box.row(align=True)
    row.enabled = not props.rescale_colormap
    row.prop(props, "colormap_range_set", text="")
    sub = row.row(align=True)
    sub.enabled = props.colormap_range_set
    sub.prop(props, "colormap_range", text="")
    if props.renderer_mode in ("YT_IDV_THRESHOLD_FOG", "YT_IDV_MAX_INTENSITY_ALPHA"):
        sub = box.box()
        sub.prop(props, "alpha_source", expand=True)
        if props.alpha_source == "CURVE" and props.alpha_curve is not None:
            node = props.alpha_curve.nodes.get(ALPHA_CURVE_NODE_NAME)
            if node is not None:
                sub.template_curve_mapping(node, "mapping")
        sub.prop(props, "color_source", expand=True)
        if props.color_source == "IMAGE":
            sub.prop(props, "color_image")
    if props.renderer_mode == "YT_IDV_GN_TRANSFER_FUNCTION":
        _draw_gn_transfer_function(box.box(), props.id_data)
    if props.renderer_mode == "YT_IDV_TRANSFER_FUNCTION":
        from . import yt_idv_renderer

        bounds = yt_idv_renderer.layer_bounds(props.dataset_uuid)
        if bounds is not None:
            box.label(text=f"log10 range: {bounds[0]:.2f} to {bounds[1]:.2f}")


def _draw_viewport_preview(layout, obj, scene):
    props = obj.volume_override
    # An operator, not the property: turning it on may ask before loading the data.
    icon = "CHECKBOX_HLT" if props.show_in_viewport else "CHECKBOX_DEHLT"
    layout.operator("volume_override.toggle_viewport", text="Show in Viewport", icon=icon, emboss=False)
    if props.show_in_viewport:
        layout.prop(scene, "yt_volumes_preview_scale")


def _draw_grids(layout, obj):
    props = obj.volume_override
    layout.prop(props, "show_grids")
    if not props.show_grids:
        return
    from . import grid_overlay

    box = layout.box()
    levels = grid_overlay.loaded_levels(obj)
    box.label(text="Render once to load the grids" if levels is None else f"Levels {levels[0]}-{levels[1]}")
    row = box.row(align=True)
    row.prop(props, "grid_level_min")
    row.prop(props, "grid_level_max")
    row = box.row(align=True)
    row.prop(props, "grid_consolidate")
    sub = row.row(align=True)
    sub.enabled = props.grid_consolidate
    sub.prop(props, "grid_clusters")


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

        if props.is_volume_override and (
            obj.type != "EMPTY" or obj.empty_display_type not in ("CUBE", "SPHERE")
        ):
            box = layout.box()
            box.alert = True
            box.label(text="Must be an Empty with Display Type 'Cube' or 'Sphere'", icon="ERROR")
            box.label(text=f"(this is a {obj.type.title()}) -- rendering will raise an error")

        col = layout.column()
        col.enabled = props.is_volume_override
        _draw_look(col, props)
        if props.renderer_mode.startswith("YT_IDV_"):
            from . import viewer_link

            col.operator("volume_override.open_viewer", icon="WINDOW")
            if viewer_link.viewer_open(obj):
                col.operator("volume_override.send_camera_to_viewer", icon="CAMERA_DATA")
                col.label(text="Enter in the window sends its camera here")
        col.separator()
        col.prop(props, "dataset_path")
        col.prop(props, "dataset_kind")
        col.prop(props, "yt_idv_volume_provider")
        if props.renderer_mode.startswith("YT_IDV_"):
            _draw_viewport_preview(col, obj, context.scene)
        _draw_grids(col, obj)
        col.prop(props, "zoom_target")
        row = col.row()
        row.enabled = bool(props.zoom_target)
        row.prop(props, "zoom_level")
        if props.dataset_uuid:
            row = col.row()
            row.enabled = False
            row.prop(props, "dataset_uuid", text="Dataset ID")


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


class VOLUMEOVERRIDE_OT_add_volume(bpy.types.Operator):
    """Add an Empty-Cube volume rendered by yt_idv, with a Geometry Nodes transfer function"""

    bl_idname = "volume_override.add_volume"
    bl_label = "yt Volume"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = bpy.data.objects.new("yt Volume", None)
        obj.empty_display_type = "CUBE"
        obj.location = context.scene.cursor.location
        context.collection.objects.link(obj)
        props = obj.volume_override
        props.is_volume_override = True
        props.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
        transfer_function_nodes.add_transfer_function_modifier(obj)
        for selected in context.selected_objects:
            selected.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


def _draw_add_menu(self, context):
    self.layout.operator(VOLUMEOVERRIDE_OT_add_volume.bl_idname, icon="VOLUME_DATA")


CLASSES = (
    *transfer_function_nodes.CLASSES,
    VOLUMEOVERRIDE_OT_add_volume,
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
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    del bpy.types.Object.volume_override
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def flagged_objects(scene):
    """All objects in `scene` flagged for external volume rendering."""
    return [obj for obj in scene.objects if getattr(obj, "volume_override", None) and obj.volume_override.is_volume_override]
