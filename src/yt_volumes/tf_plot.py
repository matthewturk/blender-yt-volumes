"""yt's own picture of a volume's transfer function (`ColorTransferFunction.show`),
in a preview icon the Volume Override panel shows and redraws when the look changes.

Needs yt and matplotlib; without them the option isn't offered.
"""

import importlib.util
import inspect

import bpy
import numpy as np
from bpy.app.handlers import persistent

WIDTH, HEIGHT = 480, 360  # pixels of the plot image
PLOT_BINS = 512  # yt's show() draws an nbins x nbins image, so tables are resampled first
DEBOUNCE = 0.2  # seconds after a change before replotting
IMAGE_DPI = 300  # the plot image's resolution, for zooming in the Image Editor
VALUE_ROWS, TABLE_ROWS = 16, 48  # rows of the table image: log10 values above, the table below
MODES = ("YT_IDV_TRANSFER_FUNCTION", "YT_IDV_YT_TRANSFER_FUNCTION", "YT_IDV_GN_TRANSFER_FUNCTION")

_shown = {}  # object name -> signature of the plot in its icon, or an error message
_pending = set()  # object names with a replot scheduled
_previews = None  # bpy.utils.previews collection; not saved in the .blend
_with_images = set()  # object names whose plot and table are also kept as Images


def available():
    return all(importlib.util.find_spec(name) is not None for name in ("yt", "matplotlib"))


def _preview(obj, create=False):
    global _previews
    import bpy.utils.previews

    if _previews is None:
        if not create:
            return None
        _previews = bpy.utils.previews.new()
    name = obj.name
    return _previews.get(name) or (_previews.new(name) if create else None)


def register():
    bpy.utils.register_class(VOLUMEOVERRIDE_OT_show_tf_images)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.frame_change_post.append(_on_depsgraph_update)  # keyframed look settings


def unregister():
    global _previews
    for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
        if _on_depsgraph_update in handlers:
            handlers.remove(_on_depsgraph_update)
    if _previews is not None:
        bpy.utils.previews.remove(_previews)
    _previews = None
    _shown.clear()
    _with_images.clear()
    if VOLUMEOVERRIDE_OT_show_tf_images.is_registered:
        bpy.utils.unregister_class(VOLUMEOVERRIDE_OT_show_tf_images)


def transfer_function(obj):
    """The yt ColorTransferFunction `obj` renders with, or None for modes
    without one. yt Layers and the Geometry Nodes mode span the range Fit to
    Data found at the last render, or Min/Max before one; Fade Around Camera,
    which depends on the view, isn't applied.
    """
    from . import transfer_function_nodes
    from . import yt_idv_renderer as yir

    props = obj.volume_override
    mode = props.renderer_mode
    if mode == "YT_IDV_TRANSFER_FUNCTION":
        bounds = (props.tf_fit_range and yir.layer_bounds(props.dataset_uuid)) or (props.tf_log_min, props.tf_log_max)
        params = dict(props.to_custom_params(), yt_idv_tf_camera_fade=False)
        return yir.layers_transfer_function(bounds, *yir._layers_settings(bounds, params))
    if mode == "YT_IDV_YT_TRANSFER_FUNCTION":
        if not props.yt_tf:
            return None
        make = yir._resolve_callable(props.yt_tf)
        return make({}) if inspect.signature(make).parameters else make()
    if mode == "YT_IDV_GN_TRANSFER_FUNCTION":
        import yt

        rgba = transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get())
        fit = props.tf_fit_range and yir.gn_tf_fit(props.dataset_uuid)
        bounds = tuple(np.log10(fit["physical_range"])) if fit else transfer_function_nodes.blended_bounds(obj)
        tf = yt.ColorTransferFunction(bounds, nbins=len(rgba), grey_opacity=True)
        for channel, values in zip(tf.funcs, np.asarray(rgba).T):
            channel.y[:] = values
        return tf
    return None


def opacity_decades(obj):
    """Decades of `obj`'s log opacity scale, 0 if linear or not a Geometry Nodes TF."""
    from . import transfer_function_nodes

    if obj.volume_override.renderer_mode != "YT_IDV_GN_TRANSFER_FUNCTION":
        return 0.0
    return transfer_function_nodes.opacity_decades(obj)


def _for_display(tf, decades=0.0):
    """`tf` resampled to PLOT_BINS bins, keeping each bin's maximum so thin
    layers stay visible, with colors brighter than 1 (Brightness Ramp) scaled
    into range. With `decades` > 0, opacity becomes the log curve's height.
    """
    import yt

    from . import transfer_function_nodes

    low, high = tf.x_bounds
    shown = yt.ColorTransferFunction((low, high), nbins=PLOT_BINS, grey_opacity=True)
    edges = np.searchsorted(tf.funcs[0].x, np.linspace(low, high, PLOT_BINS + 1))
    edges[-1] = len(tf.funcs[0].x)
    for source, target in zip(tf.funcs, shown.funcs):
        target.y[:] = np.maximum.reduceat(source.y, np.minimum(edges[:-1], len(source.y) - 1))
    shown.funcs[3].y = np.minimum(shown.funcs[3].y, 1.0)  # an opaque end's opacity is far above 1
    shown.funcs[3].y = transfer_function_nodes._opacity_to_height(shown.funcs[3].y, decades)
    brightest = max(float(channel.y.max()) for channel in shown.funcs[:3])
    if brightest > 1.0:
        for channel in shown.funcs[:3]:
            channel.y /= brightest
    return shown


def _opacity_ticks(axes, decades):
    """Label the opacity axis, whose image rows run 0 to PLOT_BINS, in opacity."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    if decades <= 0:
        axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / PLOT_BINS:.2f}"))
        return
    powers = np.arange(int(np.floor(decades)) + 1)
    powers = powers[:: max(1, int(np.ceil(len(powers) / 6)))]
    axes.yaxis.set_major_locator(FixedLocator(PLOT_BINS * (1.0 - powers / decades)))
    axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{10.0 ** (decades * (y / PLOT_BINS - 1.0)):g}"))
    axes.set_ylabel("opacity (log)", fontsize=8)


def render(tf, dpi=100, decades=0.0):
    """`(HEIGHT, WIDTH, 4)` float RGBA (times `dpi / 100`) of yt's `show()` plot
    of `tf`, top row first; opacity on a log axis over `decades` if > 0.
    """
    import matplotlib

    matplotlib.use("Agg")  # show() draws on pyplot's current figure; never open a window
    from matplotlib import pyplot

    figure = pyplot.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=dpi)
    try:
        _for_display(tf, decades).show()
        axes = figure.axes[0]
        axes.set_aspect("auto")
        _opacity_ticks(axes, decades)  # show() labels the opacity axis in image rows
        low, high = tf.x_bounds
        axes.set_title(f"log10 {low:.4f} to {high:.4f}", fontsize=9)
        figure.tight_layout()
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba(), dtype=np.float32) / 255.0
    finally:
        pyplot.close(figure)


def plot(obj):
    """Draw `obj`'s transfer function into its preview icon; None, or why not."""
    tf = transfer_function(obj)
    if tf is None:
        return "no transfer function to show (set the yt Transfer Function)"
    decades = opacity_decades(obj)
    pixels = render(tf, decades=decades)
    preview = _preview(obj, create=True)
    preview.image_size = (WIDTH, HEIGHT)
    preview.image_pixels_float.foreach_set(pixels[::-1].ravel())  # bottom row first
    if obj.name in _with_images:
        _write_images(obj, tf, decades)
    return None


def plot_image_name(obj):
    return f"yt TF plot {obj.name}"


def table_image_name(obj):
    return f"yt TF table {obj.name}"


def table_pixels(tf):
    """`(VALUE_ROWS + TABLE_ROWS, nbins, 4)` float rows, top first: each
    column's log10 field value (in R, G and B) above its RGBA table entry.
    """
    x = tf.funcs[3].x
    rgba = np.stack([channel.y for channel in tf.funcs], axis=-1)
    values = np.stack([x, x, x, np.ones_like(x)], axis=-1)
    return np.concatenate([np.repeat(values[None], VALUE_ROWS, 0), np.repeat(rgba[None], TABLE_ROWS, 0)]).astype(np.float32)


def _set_image(name, pixels):
    """Fill the float Image `name` with top-first `pixels`, as Non-Color so the
    Image Editor reads out the values themselves.
    """
    height, width = pixels.shape[:2]
    image = bpy.data.images.get(name)
    if image is None or tuple(image.size) != (width, height):
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
        image.colorspace_settings.name = "Non-Color"
        image.alpha_mode = "STRAIGHT"
    image.pixels.foreach_set(pixels[::-1].ravel())  # Blender stores the bottom row first
    image.update()
    return image


def _write_images(obj, tf, decades):
    _set_image(plot_image_name(obj), render(tf, dpi=IMAGE_DPI, decades=decades))
    _set_image(table_image_name(obj), table_pixels(tf))


def show_images(obj):
    """Keep `obj`'s plot and table as Images from now on; the table Image."""
    _with_images.add(obj.name)
    message = plot(obj)
    if message:
        raise ValueError(message)
    return bpy.data.images[table_image_name(obj)]


class VOLUMEOVERRIDE_OT_show_tf_images(bpy.types.Operator):
    """Open the transfer function's table (log10 values above their RGBA) and a sharp plot in an Image Editor"""

    bl_idname = "volume_override.show_tf_images"
    bl_label = "Open in Image Editor"

    def execute(self, context):
        obj = context.object
        try:
            image = show_images(obj)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:200])
            return {"CANCELLED"}
        area = next((a for a in context.screen.areas if a.type == "IMAGE_EDITOR"), None)
        if area is None:
            bpy.ops.wm.window_new()
            area = context.window_manager.windows[-1].screen.areas[0]
            area.type = "IMAGE_EDITOR"
        area.spaces.active.image = image
        area.spaces.active.display_channels = "COLOR"  # clicking still reads out alpha
        self.report({"INFO"}, f"Also in the Image Editor: {plot_image_name(obj)!r}")
        return {"FINISHED"}


def _signature(obj):
    """What the plot depends on: the look, and the range the last render fit."""
    from . import yt_idv_renderer as yir

    props = obj.volume_override
    look = sorted((k, repr(v)) for k, v in props.to_custom_params().items())
    fitted = (yir.layer_bounds(props.dataset_uuid), (yir.gn_tf_fit(props.dataset_uuid) or {}).get("physical_range"))
    table = None
    if props.renderer_mode == "YT_IDV_GN_TRANSFER_FUNCTION":
        from . import transfer_function_nodes

        try:
            table = hash(transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get()).tobytes())
        except ValueError:
            table = "unsampled"
    # Changing the scale keeps each point's opacity, so the table alone may not change.
    return repr((look, fitted, table, opacity_decades(obj)))


def icon(obj):
    """The icon id of `obj`'s plot (0 before the first one), scheduling a
    replot if the look changed since the last one.
    """
    signature = _signature(obj)
    if _shown.get(obj.name, (None,))[0] != signature and obj.name not in _pending:
        _pending.add(obj.name)
        bpy.app.timers.register(lambda name=obj.name, sig=signature: _replot(name, sig), first_interval=DEBOUNCE)
    preview = _preview(obj)
    return preview.icon_id if preview is not None else 0


def error(obj):
    return _shown.get(obj.name, (None, None))[1]


def _replot(name, signature):
    _pending.discard(name)
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None
    try:
        message = plot(obj)
    except Exception as exc:  # e.g. a yt Transfer Function file with an error; shown in the panel
        message = f"{type(exc).__name__}: {exc}"[:200]
    _shown[name] = (signature, message)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "PROPERTIES":
                area.tag_redraw()
    return None


@persistent
def _on_depsgraph_update(scene, depsgraph):
    # Edits made outside the panel (e.g. the node editor) don't redraw it, so check here.
    from . import properties

    for obj in properties.flagged_objects(scene):
        if obj.volume_override.show_tf_plot and obj.volume_override.renderer_mode in MODES:
            icon(obj)
