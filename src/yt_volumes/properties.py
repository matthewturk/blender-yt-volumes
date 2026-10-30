"""Per-object properties: the external-volume flag, the render look (renderer
mode and its settings, passed through as `VolumeObjectParams.custom_params`),
and dataset identity used by `operators.build_dataset_params()`.

Typed fields plus the free-form key/value table all end up in the same
`custom_params` dict. `dataset_kind` is a free-form label for the renderer
to dispatch on; Blender doesn't interpret it.
"""

import bpy
import numpy as np

from . import tf_plot, transfer_function_nodes

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
    "tf_n_layers", "tf_layer_width", "tf_alpha_first", "tf_alpha_last", "tf_brightness_ramp",
    "tf_camera_fade", "tf_camera_fade_decades", "tf_opaque", "tf_opaque_above", "tf_opaque_edge", "auto_exposure",
    "mip_alpha_max", "mip_alpha_ramp_power",
    "yt_tf", "yt_tf_ray_length", "gn_tf_opacity_scale", "opacity_per_view",
    "alpha_source", "color_source",
)
ALPHA_CURVE_NODE_NAME = "AlphaCurve"


_YT_MODES = ("YT_IDV_TRANSFER_FUNCTION", "YT_IDV_YT_TRANSFER_FUNCTION")


def _renderer_mode_changed(self, context):
    # A range fitted in one mode (e.g. Projection's large sums) is wrong for another.
    self.colormap_range_set = False
    # Thin yt layers need finer steps; a value set by hand is kept.
    defaults = (_yt_idv_renderer().SAMPLES_PER_CELL, _yt_idv_renderer().YT_SAMPLES_PER_CELL)
    if self.samples_per_cell in defaults:
        self.samples_per_cell = defaults[self.renderer_mode in _YT_MODES]


def _yt_idv_renderer():
    from . import yt_idv_renderer

    return yt_idv_renderer


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
        update=lambda self, context: _note_flagged(self.is_volume_override),
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
            ("YT_IDV_GN_TRANSFER_FUNCTION", "Geometry Nodes Transfer Function", "A transfer function drawn with Geometry Nodes on this object: a Color Ramp and an opacity curve, or any nodes", "NONE", 7),
            ("SAMPLE_CONSTANT", "Debug: Depth x Constant", "No yt_idv: clipped path length through the box, for fast pipeline tests", "NONE", 6),
        ],
        default="YT_IDV_MAX_INTENSITY",
        update=_renderer_mode_changed,
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
            "Ray samples per cell width; switching mode sets 16 for the yt modes and 4 for the others, "
            "unless changed by hand. Too few shows as wood grain and chevrons where cells are large on "
            "screen, as in deep zooms"
        ),
        default=4.0, min=0.1, soft_max=32.0,
    )
    colormap: bpy.props.StringProperty(
        name="Colormap",
        description="Any yt/matplotlib colormap name. Empty uses the mode's default",
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

    # The transfer function's range: yt Layers and Geometry Nodes Transfer Function
    tf_fit_range: bpy.props.BoolProperty(
        name="Fit to Data",
        description="Span the transfer function, in log space, over the data between the Low and High percentiles by volume; off uses Min and Max (log10)",
        default=True,
    )
    tf_percentile_low: bpy.props.FloatProperty(
        name="Low %",
        description="Percentile of the data, by volume, at the transfer function's low end",
        default=10.0, min=0.0, max=100.0,
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
        default=-20.0, precision=3,
    )

    # yt Layers (YT_IDV_TRANSFER_FUNCTION): ColorTransferFunction.add_layers
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
    tf_brightness_ramp: bpy.props.FloatProperty(
        name="Brightness Ramp",
        description="Brightness of the highest layers relative to the lowest, rising geometrically with log10 of the field; opacity is unchanged. Above 1 makes dense, small cores glow",
        default=1.0, min=1e-3, soft_max=1000.0,
    )
    tf_camera_fade: bpy.props.BoolProperty(
        name="Fade Around Camera",
        description="Fade out layers the camera has passed: those sparser than the field it has reached, over Fade Decades. With a zoom target, that field is the highest along the way in, so the fade only moves forward",
        default=False,
    )
    tf_camera_fade_decades: bpy.props.FloatProperty(
        name="Fade Decades",
        description="How many decades sparser than the field the camera has reached a layer is when it has faded out completely",
        default=1.0, min=0.01, soft_max=10.0,
    )
    tf_opaque: bpy.props.BoolProperty(
        name="Opaque End",
        description="Make the field above Opaque Above fully opaque, in the colormap's color there, after the layers; Fade Around Camera doesn't fade it",
        default=False,
    )
    tf_opaque_above: bpy.props.FloatProperty(
        name="Opaque Above (log10)",
        description="log10 of the field from which the transfer function is fully opaque",
        default=-12.0,
    )
    tf_opaque_edge: bpy.props.FloatProperty(
        name="Opaque Edge",
        description="Decades below Opaque Above over which opacity rises, geometrically, from Alpha Last to solid; 0 for a hard edge",
        default=1.0, min=0.0, soft_max=4.0,
    )

    # Maximum intensity with value-driven alpha (YT_IDV_MAX_INTENSITY_ALPHA)
    mip_alpha_max: bpy.props.FloatProperty(
        name="MIP Alpha Max",
        description="Alpha at the high end of the visible range; below 1 keeps objects behind the brightest material visible",
        default=0.7, min=0.0, max=1.0,
    )
    mip_alpha_ramp_power: bpy.props.FloatProperty(
        name="MIP Alpha Ramp Power",
        description="Shape of the alpha ramp across the visible range: 1 is linear, below 1 rises fast from the low end, above 1 stays thin longer",
        default=0.15, min=0.01, max=10.0,
    )

    tf_active_slot: bpy.props.IntProperty(
        name="Transfer Function Slot",
        description="The Geometry Nodes transfer function slot shown for editing (0-based)",
        default=0, min=0,
    )
    tf_slot: bpy.props.EnumProperty(
        name="Slot",
        description="The transfer function slot to edit; Blend chooses which ones render",
        items=transfer_function_nodes.slot_enum_items,
        get=lambda self: self.tf_active_slot,
        set=lambda self, value: transfer_function_nodes.select_slot(self.id_data, value),
    )
    tf_show_stops: bpy.props.BoolProperty(
        name="Edit Stops",
        description="Show the Color Ramp's stops for editing by hand",
        default=False,
    )
    show_tf_plot: bpy.props.BoolProperty(
        name="Show Transfer Function",
        description="Plot the transfer function as yt does (ColorTransferFunction.show): color over the value range, with the opacity curve; redrawn when the look changes",
        default=False,
    )
    opacity_per_view: bpy.props.BoolProperty(
        name="Opacity Per View",
        description=(
            "While zoomed, Ray Length and Opacity Scale apply per unit of the view, as for a yt camera, so a "
            "zoomed-in look matches the unzoomed one. Off, they apply per length of data, and deep zooms "
            "need far smaller Ray Lengths"
        ),
        default=True,
    )
    auto_exposure: bpy.props.BoolProperty(
        name="Auto Exposure",
        description=(
            "Transfer function modes: scale the transfer function's brightness (not its opacity) so the "
            "brightest structure sits near white with little clipping. Each render starts from the last "
            "one's exposure, so animations change smoothly"
        ),
        default=True,
    )

    # yt transfer function (YT_IDV_YT_TRANSFER_FUNCTION)
    yt_tf: bpy.props.StringProperty(
        name="yt Transfer Function",
        description="'module:function' or 'file.py:function' returning a yt ColorTransferFunction (grey opacity, log10 of the field)",
        default="",
    )
    yt_tf_ray_length: bpy.props.FloatProperty(
        name="yt TF Ray Length",
        description="Divides the transfer function's emission and opacity, as a yt camera's ray length in unitary units; larger is fainter and more transparent",
        default=0.02, min=1e-9,
    )

    # Geometry Nodes transfer function (YT_IDV_GN_TRANSFER_FUNCTION)
    gn_tf_opacity_scale: bpy.props.FloatProperty(
        name="Opacity Scale",
        description="Multiplies the transfer function's emission and opacity, per box width of path; raise it until small or thin structures show",
        default=100.0, min=0.0, soft_max=10000.0,
    )
    # Alpha curve / color image for MIP Alpha
    alpha_source: bpy.props.EnumProperty(
        name="Alpha Source",
        description="How the MIP Alpha ramp is shaped",
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
        description="Where MIP Alpha gets its RGB",
        items=[
            ("NAMED", "Named Colormap", "The Colormap setting"),
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
    data_units: bpy.props.StringProperty(
        name="Data Units",
        description=(
            "Lengths to convert to scene units, comma-separated yt units (e.g. 'kpc, pc, Mpc/h, kpccm'); "
            "read them from the object's yt_units custom properties, e.g. in a driver"
        ),
        default="kpc",
    )
    units_object: bpy.props.PointerProperty(
        name="Units Object",
        description="One-vertex mesh whose point attributes are this volume's yt_units, kept current for Geometry Nodes",
        type=bpy.types.Object,
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
    zoom_limit_depth: bpy.props.BoolProperty(
        name="Limit Zoom Depth",
        description=(
            "While zoomed, end rays at Zoom Depth times the camera's distance to the zoom "
            "target, so the gas far behind it, mostly sparse, doesn't haze the view"
        ),
        default=True,
    )
    zoom_depth: bpy.props.FloatProperty(
        name="Zoom Depth",
        description="How far rays reach while zoomed, in multiples of the camera's distance to the zoom target; above 1 reaches past it",
        default=2.0, min=1.0, soft_max=100.0,
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
        if self.zoom_limit_depth:  # also for a zoom from Deep Zoom
            params["yt_idv_zoom_depth"] = self.zoom_depth
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


_TF_RANGE = ("tf_fit_range", "tf_percentile_low", "tf_percentile_high", "tf_log_min", "tf_log_max")
_MODE_SETTINGS = {
    "YT_IDV_MAX_INTENSITY": ("colormap",),
    "YT_IDV_PROJECTION": ("colormap",),
    "YT_IDV_TRANSFER_FUNCTION": (
        "tf_n_layers", "tf_layer_width", "tf_alpha_first", "tf_alpha_last", "colormap", "tf_brightness_ramp",
        "tf_camera_fade", "tf_camera_fade_decades", "tf_opaque", "tf_opaque_above", "tf_opaque_edge",
        *_TF_RANGE, "yt_tf_ray_length", "opacity_per_view", "auto_exposure",
    ),
    "YT_IDV_MAX_INTENSITY_ALPHA": ("colormap", "mip_alpha_max", "mip_alpha_ramp_power"),
    "YT_IDV_YT_TRANSFER_FUNCTION": ("yt_tf", "yt_tf_ray_length", "opacity_per_view", "auto_exposure"),
    "YT_IDV_GN_TRANSFER_FUNCTION": ("gn_tf_opacity_scale", "opacity_per_view", *_TF_RANGE, "auto_exposure"),
}
# Modes that color with a colormap over a fitted range (Rescale Colormap, Colormap Range).
_COLORMAP_RANGE_MODES = ("YT_IDV_MAX_INTENSITY", "YT_IDV_MAX_INTENSITY_ALPHA", "YT_IDV_PROJECTION")


def _draw_gn_transfer_function(layout, obj):
    from . import yt_idv_renderer

    props = obj.volume_override
    fit = yt_idv_renderer.gn_tf_fit(props.dataset_uuid)
    if fit is not None:
        low, high = np.log10(fit["physical_range"])
        layout.label(text=f"log10 range: {low:.2f} to {high:.2f} {fit['units']}")
    elif not props.tf_fit_range:
        low, high = props.tf_log_min, props.tf_log_max
    else:
        low = high = None
    ramp, curve = transfer_function_nodes.find_nodes(obj)
    if ramp is None and curve is None:
        layout.operator("volume_override.add_transfer_function", icon="ADD")
        return
    # The starter group's nodes, editable here; anything else in Geometry Nodes.
    tree = transfer_function_nodes.transfer_function_tree(obj)
    slots = transfer_function_nodes.slot_nodes(tree) if tree is not None else []
    if len(slots) > 1:
        _draw_tf_blend(layout.box(), obj, slots)
    row = layout.row(align=True)
    row.label(text="Edit Slot" if len(slots) > 1 else "Slot")
    row.prop(props, "tf_slot", text="")
    if slots:
        row.prop(transfer_function_nodes.active_slot(obj), "label", text="", placeholder="Name")
    row.operator("volume_override.add_tf_slot", text="", icon="ADD")
    row.operator("volume_override.remove_tf_slot", text="", icon="REMOVE")
    if ramp is not None:
        row = layout.row(align=True)
        row.label(text="Colormap")
        name = transfer_function_nodes.colormap_name(obj) or "Custom"
        row.operator("volume_override.load_tf_colormap", text=name, icon="COLOR")
        row.prop(props, "tf_show_stops", text="", icon="IPO_LINEAR")
        if props.tf_show_stops:
            layout.template_color_ramp(ramp, "color_ramp", expand=True)
    if curve is not None:
        decades_node = transfer_function_nodes.find_node(obj, transfer_function_nodes.DECADES_NODE_NAME)
        decades = decades_node.outputs[0].default_value if decades_node is not None else 0.0
        row = layout.row()
        row.label(text="Opacity")
        scale = f"Log, {decades:g} decades" if decades > 0 else "Linear"
        row.operator("volume_override.set_opacity_curve_scale", text=scale, icon="PREFERENCES")
        layout.template_curve_mapping(curve, "mapping")
        if decades > 0:
            layout.label(text=f"Height 0.5 = {10 ** (-decades / 2):.2g}, near 0 = {10 ** -decades:.2g}, 0 = off")
    row = layout.row(align=True)
    row.operator("volume_override.add_colorbar", icon="COLOR")
    if getattr(bpy.context.scene, "yt_annotation_scene", None) is not None:
        row.prop(bpy.context.scene, "yt_annotation_scene", text="")
    if low is not None:
        # The ramp and curve run 0-1 across the range; this places a zoomed-in view.
        layout.label(text=f"0 = {low:.2f}, 0.5 = {(low + high) / 2:.2f}, 1 = {high:.2f} (log10)")
        layout.label(text=f"0.01 along = {(high - low) / 100:.3g} decades")


def _slot_name(slots, index):
    return f"{index + 1}: {slots[index].label or 'Slot'}"


def _draw_tf_blend(layout, obj, slots):
    """The Blend slider, what it shows now, and its keyframes as a list."""
    blend = transfer_function_nodes.blend_value(obj)
    position = min(max(blend, 1.0), len(slots)) - 1.0
    low = min(int(position), len(slots) - 2)
    along = position - low
    if along < 1e-3 or along > 1 - 1e-3:
        showing = _slot_name(slots, round(position))
    else:
        showing = f"{_slot_name(slots, low)} → {_slot_name(slots, low + 1)}, {along:.0%}"
    layout.label(text=f"Showing {showing}", icon="HIDE_OFF")
    found = transfer_function_nodes._blend_input(obj)
    if found is None:  # made before Blend was a modifier input; adding a slot updates it
        layout.label(text="Add a slot to update this transfer function", icon="INFO")
        return
    layout.prop(found[0], "value", text="Blend", slider=True)
    fcurve = transfer_function_nodes.blend_fcurve(obj)
    if fcurve is not None and fcurve.keyframe_points:
        col = layout.column(align=True)
        for point in sorted(fcurve.keyframe_points, key=lambda p: p.co[0]):
            frame, value = point.co
            row = col.row(align=True)
            index = int(round(min(max(value, 1.0), len(slots)))) - 1
            label = _slot_name(slots, index) if abs(value - (index + 1)) < 1e-3 else f"blend {value:.2f}"
            go = row.operator("volume_override.goto_tf_slot_key", text=f"Frame {frame:g}", icon="KEYFRAME_HLT", emboss=False)
            go.frame = int(round(frame))
            row.label(text=label)
            row.operator("volume_override.remove_tf_slot_key", text="", icon="X", emboss=False).frame = frame
        row = layout.row(align=True)
        current = {p.interpolation for p in fcurve.keyframe_points}
        for easing, label, _ in transfer_function_nodes.BLEND_EASINGS:
            row.operator("volume_override.set_tf_blend_easing", text=label, depress=current == {easing}).easing = easing
    frame = bpy.context.scene.frame_current
    layout.operator("volume_override.key_tf_slot", text=f"Keyframe Edited Slot at Frame {frame}", icon="KEY_HLT")


TF_PLOT_SCALE = 18.0  # icon size in UI units; the plot keeps its aspect inside it


def _draw_tf_plot(layout, obj):
    if not tf_plot.available():
        return
    row = layout.row()
    row.prop(obj.volume_override, "show_tf_plot")
    row.operator("volume_override.show_tf_images", icon="IMAGE_DATA")
    if not obj.volume_override.show_tf_plot:
        return
    icon = tf_plot.icon(obj)
    message = tf_plot.error(obj)
    if message:
        layout.label(text=message, icon="ERROR")
    elif icon:
        layout.template_icon(icon_value=icon, scale=TF_PLOT_SCALE)
    else:
        layout.label(text="Plotting the transfer function...")


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
        if name in ("tf_log_min", "tf_log_max"):
            row.enabled = not props.tf_fit_range
        elif name in ("tf_percentile_low", "tf_percentile_high"):
            row.enabled = props.tf_fit_range
        elif name == "tf_camera_fade_decades":
            row.enabled = props.tf_camera_fade
        elif name in ("tf_opaque_above", "tf_opaque_edge"):
            row.enabled = props.tf_opaque
        row.prop(props, name)
        if name == "auto_exposure" and props.auto_exposure:
            from . import yt_idv_renderer

            gain = yt_idv_renderer.exposure_gain(props.dataset_uuid)
            if gain is not None:
                col.label(text=f"Exposure: x{gain:.3g} on the transfer function")
    if props.renderer_mode in _COLORMAP_RANGE_MODES:
        box.prop(props, "rescale_colormap")
        row = box.row(align=True)
        row.enabled = not props.rescale_colormap
        row.prop(props, "colormap_range_set", text="")
        sub = row.row(align=True)
        sub.enabled = props.colormap_range_set
        sub.prop(props, "colormap_range", text="")
    if props.renderer_mode == "YT_IDV_MAX_INTENSITY_ALPHA":
        sub = box.box()
        sub.prop(props, "alpha_source", expand=True)
        if props.alpha_source == "CURVE" and props.alpha_curve is not None:
            node = props.alpha_curve.nodes.get(ALPHA_CURVE_NODE_NAME)
            if node is not None:
                sub.template_curve_mapping(node, "mapping")
        sub.prop(props, "color_source", expand=True)
        if props.color_source == "IMAGE":
            sub.prop(props, "color_image")
    if props.renderer_mode in tf_plot.MODES:
        _draw_tf_plot(box, props.id_data)
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
        layout.prop(scene, "yt_volumes_preview_samples")


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
        row = col.row(align=True)
        row.prop(props, "zoom_limit_depth")
        sub = row.row(align=True)
        sub.enabled = props.zoom_limit_depth
        sub.prop(props, "zoom_depth", text="")
        _draw_units(col, obj)
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
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        handlers.append(_rescan_flags)
    _flag_scan["objects"] = -1


def unregister():
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _rescan_flags in handlers:
            handlers.remove(_rescan_flags)
    bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    del bpy.types.Object.volume_override
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


def _draw_units(layout, obj):
    row = layout.row(align=True)
    row.prop(obj.volume_override, "data_units")
    row.operator("volume_override.read_units", text="", icon="FILE_REFRESH")
    if obj.volume_override.units_object is None:
        layout.operator("volume_override.add_units_object", icon="GEOMETRY_NODES")
    else:
        row = layout.row(align=True)
        row.prop(obj.volume_override, "units_object")
        row.operator("volume_override.add_field_of_view", text="", icon="DRIVER_DISTANCE")
        row.operator("volume_override.add_annotation", text="", icon="GEOMETRY_NODES")
    group = obj.get("yt_units", {})
    for unit in (u.strip() for u in obj.volume_override.data_units.split(",") if u.strip()):
        value = group.get(f"scene_per_{unit}")
        if value:
            layout.label(text=f"1 {unit} = {value:.4g} scene units")
        elif "registry" in group:
            layout.label(text=f"{unit}: not a length this dataset's units know", icon="ERROR")
        else:
            layout.label(text=f"{unit}: not known yet (render, or Read Units)")


# Whether any object in bpy.data is flagged, rescanned whenever the object count changes.
_flag_scan = {"objects": -1, "any": False}


def _note_flagged(flagged):
    if flagged:
        _flag_scan["any"] = True


def any_flagged():
    """Whether any object in the file may be flagged; cheap enough for handlers that run on every update."""
    count = len(bpy.data.objects)
    if count != _flag_scan["objects"]:
        _flag_scan["objects"] = count
        _flag_scan["any"] = any(getattr(obj, "volume_override", None) and obj.volume_override.is_volume_override for obj in bpy.data.objects)
    return _flag_scan["any"]


@bpy.app.handlers.persistent
def _rescan_flags(*args):
    # A loaded file or an undo step can bring flags back without changing the object count.
    _flag_scan["objects"] = -1


def flagged_objects(scene):
    """All objects in `scene` flagged for external volume rendering."""
    if not any_flagged():
        return []
    return [obj for obj in scene.objects if getattr(obj, "volume_override", None) and obj.volume_override.is_volume_override]


def _renderable(layer_collection):
    """Objects a render includes from `layer_collection` and its children."""
    if layer_collection.exclude or layer_collection.collection.hide_render:
        return set()
    found = set(layer_collection.collection.objects)
    for child in layer_collection.children:
        found |= _renderable(child)
    return found


def rendered_volumes(scene, view_layer=None):
    """Flagged objects a render of `view_layer` (default the active one)
    includes: Disable in Renders off, collection neither excluded nor
    disabled in renders.
    """
    if view_layer is None:
        view_layer = bpy.context.view_layer if bpy.context.scene == scene else scene.view_layers[0]
    included = _renderable(view_layer.layer_collection)
    return [obj for obj in flagged_objects(scene) if not obj.hide_render and obj in included]
