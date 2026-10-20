"""yt's own picture of a volume's transfer function (`ColorTransferFunction.show`),
in an image the Volume Override panel shows and redraws when the look changes.

Needs yt and matplotlib; without them the option isn't offered.
"""

import importlib.util
import inspect

import bpy
import numpy as np

WIDTH, HEIGHT = 480, 260  # pixels of the plot image
PLOT_BINS = 512  # yt's show() draws an nbins x nbins image, so tables are resampled first
DEBOUNCE = 0.2  # seconds after a change before replotting

_shown = {}  # object name -> signature of the plot in its image, or an error message
_pending = set()  # object names with a replot scheduled


def available():
    return all(importlib.util.find_spec(name) is not None for name in ("yt", "matplotlib"))


def image_name(obj):
    return f"yt TF {obj.name}"


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
        bounds = tuple(np.log10(fit["physical_range"])) if fit else (props.tf_log_min, props.tf_log_max)
        tf = yt.ColorTransferFunction(bounds, nbins=len(rgba), grey_opacity=True)
        for channel, values in zip(tf.funcs, np.asarray(rgba).T):
            channel.y[:] = values
        return tf
    return None


def _for_display(tf):
    """`tf` resampled to PLOT_BINS bins, keeping each bin's maximum so thin
    layers stay visible, with colors brighter than 1 (Brightness Ramp) scaled
    into range.
    """
    import yt

    low, high = tf.x_bounds
    shown = yt.ColorTransferFunction((low, high), nbins=PLOT_BINS, grey_opacity=True)
    edges = np.searchsorted(tf.funcs[0].x, np.linspace(low, high, PLOT_BINS + 1))
    edges[-1] = len(tf.funcs[0].x)
    for source, target in zip(tf.funcs, shown.funcs):
        target.y[:] = np.maximum.reduceat(source.y, np.minimum(edges[:-1], len(source.y) - 1))
    brightest = max(float(channel.y.max()) for channel in shown.funcs[:3])
    if brightest > 1.0:
        for channel in shown.funcs[:3]:
            channel.y /= brightest
    return shown


def render(tf):
    """`(HEIGHT, WIDTH, 4)` float RGBA of yt's `show()` plot of `tf`, top row first."""
    import matplotlib

    matplotlib.use("Agg")  # show() draws on pyplot's current figure; never open a window
    from matplotlib import pyplot
    from matplotlib.ticker import FuncFormatter

    figure = pyplot.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    try:
        _for_display(tf).show()
        axes = figure.axes[0]
        axes.set_aspect("auto")
        # show() labels the opacity axis in image rows rather than 0 to 1.
        axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / PLOT_BINS:.2f}"))
        figure.tight_layout()
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba(), dtype=np.float32) / 255.0
    finally:
        pyplot.close(figure)


def _image(obj):
    name = image_name(obj)
    image = bpy.data.images.get(name)
    if image is None or tuple(image.size) != (WIDTH, HEIGHT):
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, WIDTH, HEIGHT, alpha=True)
    texture = bpy.data.textures.get(name) or bpy.data.textures.new(name, "IMAGE")
    texture.image = image
    return image, texture


def plot(obj):
    """Draw `obj`'s transfer function into its image; None, or why not."""
    tf = transfer_function(obj)
    if tf is None:
        return "no transfer function to show (set the yt Transfer Function)"
    pixels = render(tf)
    image, texture = _image(obj)
    image.pixels.foreach_set(pixels[::-1].ravel())  # Blender stores the bottom row first
    image.update()
    texture.update_tag()
    return None


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
    return repr((look, fitted, table))


def texture(obj):
    """The texture showing `obj`'s plot, scheduling a replot if the look
    changed since the last one (panels can't change data while drawing).
    """
    signature = _signature(obj)
    if _shown.get(obj.name, (None,))[0] != signature and obj.name not in _pending:
        _pending.add(obj.name)
        bpy.app.timers.register(lambda name=obj.name, sig=signature: _replot(name, sig), first_interval=DEBOUNCE)
    return bpy.data.textures.get(image_name(obj))


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
