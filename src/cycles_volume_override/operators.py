"""Render orchestration: the two-pass-per-frame pipeline this add-on's F12
override runs, plus the request-building glue between Blender data and the
external_render.py contract.

Why two full render passes per frame (not obvious from the spec's lettered
steps alone -- worth stating explicitly, like the two open questions in
prompt.md):

Blender's compositor executes as part of ONE `render.render()` call, right
after the 3D engine renders and BEFORE that call returns. That means by the
time any `render_post` handler could react to "the Z pass is ready," the
compositor for THIS SAME render has already run -- using whatever was in
the Image nodes beforehand (stale). There is no handler hook that fires
between "Z pass exists" and "compositor already consumed it." So a single
`render.render()` call cannot both produce this frame's Z pass AND composite
this frame's volume-renderer result from it.

Fix: two passes, explicitly triggered by our own operator (the spec's "or an
overridden render operator" option) rather than handlers alone:

  Pass 1 (real Cycles): main scene, flagged objects hidden, Z pass on,
  compositor set to JUST write Combined + Z to full-float EXR via File
  Output (compositor_build.build_pass1_capture_graph). We throw away
  whatever pass 1's own Composite output is -- it's built from stale/no
  volume layers -- we only want the two EXRs it wrote to disk.

  Between passes: read the Z EXR back into a DepthBuffer, call
  render_volume() per flagged object, write each RenderResult to its own
  EXR, then rebuild the compositor as Image(beauty EXR) -> chained Alpha
  Over per volume -> Composite (compositor_build.build_or_update_graph).

  Pass 2 (cheap, engine swapped to Workbench): the compositor graph built
  above reads EVERYTHING from EXRs via Image nodes -- it has no live
  Render Layers dependency -- so pass 2's own 3D render is thrown away too;
  it exists purely to make Blender execute the compositor and write the
  real final output file. Workbench is a rasterizer and essentially free
  compared to a second Cycles pass, which is the whole point.
"""

import math
import os
import shutil
import tempfile
import time

import bpy

from . import change_tracking, compositor_build, external_render, properties, sample_renderers, scene_sync, yt_idv_renderer

PASS2_ENGINE = "BLENDER_WORKBENCH"

# --- Debug logging -----------------------------------------------------------
#
# Added while diagnosing a reported hang: prints a timestamped line to
# Blender's system console (Window > Toggle System Console on Windows; just
# the terminal you launched Blender from on Linux/macOS) before/after every
# step of the two-pass pipeline that can plausibly block -- both
# `bpy.ops.render.render()` calls, every per-object `render_volume()` call,
# and the EXR read/write round-trips -- so a hang shows up as "the last line
# printed" instead of a silent freeze with no way to tell which step it's in.
# Gated on `VolumeOverridePreferences.verbose_logging` (default on) so it can
# be switched off once the pipeline is known-good; see draw() below.
_LOG_PREFIX = "[VolumeOverride]"
_log_start_time = None


def _log(msg):
    global _log_start_time
    now = time.monotonic()
    if _log_start_time is None:
        _log_start_time = now
    print(f"{_LOG_PREFIX} [+{now - _log_start_time:7.2f}s] {msg}", flush=True)


class _StepTimer:
    """`with _StepTimer("pass 1 render"): ...` -- logs entry immediately (so
    you see it start even if it never finishes) and exit with elapsed time.
    """

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        _log(f"START {self.label}")
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.monotonic() - self._t0
        if exc_type is None:
            _log(f"DONE  {self.label} ({elapsed:.2f}s)")
        else:
            _log(f"FAILED {self.label} ({elapsed:.2f}s): {exc_type.__name__}: {exc}")
        return False


class _NullStepTimer:
    """No-op stand-in for `_StepTimer` when `verbose_logging` is off."""

    def __init__(self, label):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

class YtIdvRenderSettings(bpy.types.PropertyGroup):
    """Live, UI-editable overrides for `yt_idv_renderer.py`'s own hardcoded
    module constants -- one field per constant, defaults matching those
    constants exactly, so leaving everything at its default reproduces the
    old hardcoded behavior precisely. Deliberately does NOT expose dataset
    selection (`SAMPLE_DATASET_NAME`/`SAMPLE_FIELD` stay hardcoded) --
    dataset loading/creation is a separate, not-yet-designed problem.

    Reaches the renderer via `VolumeObjectParams.custom_params` (see
    `_yt_idv_custom_params`/`build_volume_params` below), NOT by
    `yt_idv_renderer.py` importing `bpy`/reading this directly -- keeps that
    module's own "no bpy dependency, testable via a hand-built RenderRequest
    alone" property intact. Values here only ever apply to `YT_IDV_*`
    `renderer_mode`s (see `VolumeOverridePreferences.draw`'s conditional
    section); harmless if set while a different renderer_mode is active,
    since `custom_params` is a passthrough dict every renderer is free to
    ignore.
    """

    backend: bpy.props.EnumProperty(
        name="Backend",
        description="Offscreen OpenGL backend -- see yt_idv_renderer.py module docstring, 'yt_idv bug 2': OSMesa corrupts non-square renders, kept here for completeness/experimentation only",
        items=[
            ("egl", "EGL", "Needs a working GPU + driver stack; the only backend confirmed clean at non-square resolutions"),
            ("osmesa", "OSMesa (buggy)", "Software rendering, no GPU needed -- but corrupts non-square renders (yt_idv bug 2, root cause not identified)"),
        ],
        default="egl",
    )

    # --- Isocontour transfer function (YT_IDV_TRANSFER_FUNCTION) --------
    tf_value_min: bpy.props.FloatProperty(
        name="Isocontour Value Min",
        description="Lower bound of the isocontour layers' density range, in yt_idv's own [0, 1]-normalized value space (NOT raw physical units) -- see yt_idv_renderer.py TF_VALUE_MIN",
        default=1e-6, precision=8, min=0.0,
    )
    tf_value_max: bpy.props.FloatProperty(
        name="Isocontour Value Max",
        description="Upper bound of the isocontour layers' density range, same normalized space as Value Min -- see yt_idv_renderer.py TF_VALUE_MAX",
        default=2e-3, precision=6, min=0.0,
    )
    tf_n_layers: bpy.props.IntProperty(
        name="Isocontour Layers",
        description="How many isocontour layers yt.ColorTransferFunction.add_layers() spaces evenly (in log space) between Value Min/Max -- see yt_idv_renderer.py TF_N_LAYERS",
        default=5, min=1, max=64,
    )
    tf_colormap: bpy.props.StringProperty(
        name="Isocontour Colormap",
        description="Any yt/matplotlib colormap name (e.g. viridis, dusk, arbre, inferno) -- see yt_idv_renderer.py TF_COLORMAP",
        default="viridis",
    )

    # --- Threshold fog transfer function (YT_IDV_THRESHOLD_FOG) ---------
    fog_value_min: bpy.props.FloatProperty(
        name="Fog Threshold (Value Min)",
        description="Fully transparent below this density (yt_idv's own [0, 1]-normalized value space, NOT raw physical units) -- see yt_idv_renderer.py TF_FOG_VALUE_MIN",
        default=1e-9, precision=10, min=0.0,
    )
    fog_value_max: bpy.props.FloatProperty(
        name="Fog Value Max",
        description="Alpha reaches Fog Alpha Max at this density, same normalized space as Fog Threshold -- see yt_idv_renderer.py TF_FOG_VALUE_MAX",
        default=1e-2, precision=6, min=0.0,
    )
    fog_colormap: bpy.props.StringProperty(
        name="Fog Colormap",
        description="Any yt/matplotlib colormap name -- see yt_idv_renderer.py TF_FOG_COLORMAP",
        default="dusk",
    )
    fog_alpha_max: bpy.props.FloatProperty(
        name="Fog Alpha Max",
        description="Alpha at Fog Value Max -- below 1.0 keeps even the densest gas looking soft/foggy rather than a hard solid wall -- see yt_idv_renderer.py TF_FOG_ALPHA_MAX",
        default=0.85, min=0.0, max=1.0,
    )
    fog_ramp_power: bpy.props.FloatProperty(
        name="Fog Ramp Power",
        description="Shape of the alpha ramp between the threshold and Value Max -- 1.0 is linear (in log-density space); below 1.0 ramps up fast just past the threshold, thickening more gradually toward Value Max (looks hazier); above 1.0 stays thin longer before thickening quickly -- see yt_idv_renderer.py TF_FOG_RAMP_POWER",
        default=0.15, min=0.01, max=10.0,
    )

    # --- Maximum Intensity, value-driven alpha (YT_IDV_MAX_INTENSITY_ALPHA)
    mip_alpha_colormap: bpy.props.StringProperty(
        name="MIP Alpha Colormap",
        description="Any yt/matplotlib colormap name -- see yt_idv_renderer.py MIP_ALPHA_COLORMAP",
        default="dusk",
    )
    mip_alpha_max: bpy.props.FloatProperty(
        name="MIP Alpha Max",
        description="Alpha at the high end of whatever's currently visible (MIP's own per-view auto-scaled range) -- below 1.0 keeps something visible behind even the brightest material via Alpha-Over, which otherwise can fully hide objects embedded in dense gas -- see yt_idv_renderer.py MIP_ALPHA_MAX",
        default=0.7, min=0.0, max=1.0,
    )
    mip_alpha_ramp_power: bpy.props.FloatProperty(
        name="MIP Alpha Ramp Power",
        description="Shape of the alpha ramp across the visible value range -- same meaning as Fog Ramp Power -- see yt_idv_renderer.py MIP_ALPHA_RAMP_POWER",
        default=0.15, min=0.01, max=10.0,
    )

    # --- Curve-driven alpha / image-driven color (Threshold Fog + MIP
    # Alpha only -- see yt_idv_renderer.py module docstring's "Curve-driven
    # alpha / image-driven color" section). Shared across both modes
    # deliberately -- one curve/image setting, not four independent ones --
    # see `_yt_idv_custom_params`'s own comment for why.
    alpha_source: bpy.props.EnumProperty(
        name="Alpha Source",
        description="How the alpha-vs-position ramp for Threshold Fog / MIP Alpha is shaped",
        items=[
            ("RAMP_POWER", "Ramp Power", "The Fog/MIP Ramp Power field above -- a single-exponent curve"),
            ("CURVE", "Custom Curve", "A hand-drawn curve, edited below via Blender's own curve widget"),
        ],
        default="RAMP_POWER",
    )
    color_source: bpy.props.EnumProperty(
        name="Color Source",
        description="Where the RGB color for Threshold Fog / MIP Alpha comes from",
        items=[
            ("NAMED", "Named Colormap", "The Fog/MIP Alpha Colormap field above -- any yt/matplotlib colormap name"),
            ("IMAGE", "Image", "A Blender Image, read back and treated as a 1D horizontal gradient (averaged across rows) -- selected below"),
        ],
        default="NAMED",
    )
    # A NAME, not `PointerProperty(type=bpy.types.Image)` -- confirmed
    # directly that AddonPreferences (and anything nested inside one, even
    # a plain PropertyGroup that merely CONTAINS an ID pointer somewhere in
    # its tree) categorically rejects ID/data-block PointerProperty fields:
    # `ValueError: ... PointerProperty could not register because this
    # type doesn't support data-block properties`. Same restriction that
    # ruled out `PointerProperty(type=bpy.types.CurveMapping)` above, for a
    # different underlying reason (CurveMapping isn't an ID type at all;
    # Image IS, but AddonPreferences still can't hold one). Standard
    # Blender workaround: store the name, look it up in `bpy.data.images`
    # when needed (`_yt_idv_custom_params`), and use `layout.prop_search`
    # (not `layout.prop`) for a proper picker UI -- see `draw()`.
    color_image_name: bpy.props.StringProperty(
        name="Color Image",
        description="Any Blender Image -- averaged across its rows then resampled horizontally into the colormap, so any image works regardless of its native size/aspect (see _sample_image_as_1d_rgba)",
        default="",
    )


_ALPHA_CURVE_NODE_GROUP_NAME = "VolumeOverride yt_idv Alpha Curve (internal)"
_ALPHA_CURVE_NODE_NAME = "AlphaCurve"


def _get_or_create_alpha_curve_node():
    """A `ShaderNodeFloatCurve`, tucked away in its own node group purely as
    a data container for its `.mapping` (a real `bpy.types.CurveMapping`)
    -- NEVER connected to any real shader output or used for actual
    shading, never added to any Material. This is the standard workaround
    for "I want a native curve-editing widget somewhere that isn't a node
    editor": `CurveMapping` can't be a `PointerProperty` on a plain
    PropertyGroup (confirmed directly: `TypeError: PointerProperty(...)
    expected an RNA type derived from ID or ID Property Group`) -- it can
    only exist as a field Blender's OWN C-defined structs already carry
    (nodes, brushes, color management settings, ...). A node's own
    `.mapping` is the most convenient of those to piggyback on:
    `UILayout.template_curve_mapping(node, "mapping")` draws the SAME
    widget a Shader Editor's N-panel would show, directly inside this
    add-on's own Preferences panel -- see `VolumeOverridePreferences.draw`.

    The node group lives in `bpy.data.node_groups` (persisted with the
    .blend file, same as any other node group) and is created lazily, once,
    the first time it's needed -- `get`-or-`new` on both the group and the
    node itself, so calling this repeatedly (once per `draw()` call, once
    per render) is cheap and always returns the SAME node/mapping, not a
    fresh one each time.
    """
    tree = bpy.data.node_groups.get(_ALPHA_CURVE_NODE_GROUP_NAME)
    if tree is None:
        tree = bpy.data.node_groups.new(_ALPHA_CURVE_NODE_GROUP_NAME, "ShaderNodeTree")
    node = tree.nodes.get(_ALPHA_CURVE_NODE_NAME)
    if node is None:
        node = tree.nodes.new("ShaderNodeFloatCurve")
        node.name = _ALPHA_CURVE_NODE_NAME
        node.mapping.initialize()
    return node


def _evaluate_alpha_curve(nbins=256):
    """The hidden alpha curve (see `_get_or_create_alpha_curve_node`),
    sampled at `nbins` evenly-spaced positions across its own `[0, 1]`
    domain -- the plain list of floats that actually crosses `custom_
    params` as `"yt_idv_alpha_curve"` (see `_yt_idv_custom_params`),
    resampled again on the `yt_idv_renderer.py` side to whatever bin count
    the LUT actually needs (`_resample_1d`) -- so `nbins` here doesn't need
    to match that exactly, it's just "how finely to sample the curve
    itself", 256 being a reasonable default resolution for a hand-drawn
    curve either way.

    `mapping.initialize()` is called every time (confirmed safe/idempotent
    on an already-initialized, already-edited mapping -- doesn't reset
    existing points) rather than assumed to have already happened, since
    `CurveMapping.evaluate()` needs it to have run at least once since the
    curve's own points were last changed.
    """
    node = _get_or_create_alpha_curve_node()
    mapping = node.mapping
    mapping.initialize()
    curve = mapping.curves[0]
    return [mapping.evaluate(curve, i / (nbins - 1)) for i in range(nbins)]


def _sample_image_as_1d_rgba(image, nbins=256):
    """Read `image` (a `bpy.types.Image`) back as an `(nbins, 4)` RGBA list
    of lists, "functionally 1D" per the user's own framing: averaged across
    ALL rows (collapsing whatever vertical variation exists) then resampled
    horizontally to `nbins` evenly-spaced positions -- so any image works
    as a colormap source regardless of its own native resolution/aspect
    ratio, not just a literal 1-pixel-tall gradient strip. Missing channels
    (RGB-only or grayscale images) are padded with alpha=1.0 (unused
    anyway -- alpha always comes from the alpha curve/ramp, never this).

    Raises `ValueError` for a 0x0 image (e.g. one whose source file isn't
    packed/available) rather than silently producing a degenerate
    colormap -- a clear, immediate error beats a mysteriously blank/wrong
    render.
    """
    w, h = image.size[0], image.size[1]
    if w == 0 or h == 0:
        raise ValueError(
            f"image {image.name!r} has no pixel data (size {w}x{h}) -- "
            "pack or reload it before using it as a yt_idv color source"
        )
    import numpy as np

    pixels = np.array(image.pixels[:], dtype=np.float64).reshape(h, w, image.channels)
    if image.channels < 4:
        padded = np.ones((h, w, 4))
        padded[..., : image.channels] = pixels
        pixels = padded
    row_avg = pixels.mean(axis=0)  # (w, 4) -- collapse vertical variation, "functionally 1D"
    src_x = np.linspace(0.0, 1.0, w)
    dst_x = np.linspace(0.0, 1.0, nbins)
    out = np.empty((nbins, 4))
    for c in range(4):
        out[:, c] = np.interp(dst_x, src_x, row_avg[:, c])
    return out.tolist()


# Keyed by VolumeOverridePreferences.renderer_mode -- see that enum's items
# for what each option is for.
RENDERER_MODES = {
    "STUB": external_render.render_volume,
    "DEBUG_FLAT": external_render.debug_fake_render_volume,
    "SAMPLE_CONSTANT": sample_renderers.constant_density_renderer,
    "YT_IDV_MAX_INTENSITY": yt_idv_renderer.yt_idv_max_intensity_renderer,
    "YT_IDV_MAX_INTENSITY_ALPHA": yt_idv_renderer.yt_idv_max_intensity_alpha_renderer,
    "YT_IDV_TRANSFER_FUNCTION": yt_idv_renderer.yt_idv_transfer_function_renderer,
    "YT_IDV_PROJECTION": yt_idv_renderer.yt_idv_projection_renderer,
    "YT_IDV_THRESHOLD_FOG": yt_idv_renderer.yt_idv_threshold_fog_renderer,
}

# Renderer modes that create their own offscreen EGL context (yt_idv) --
# see render_current_frame()'s `compositor_device` override for why this
# set specifically, not all of RENDERER_MODES, needs it.
YT_IDV_RENDERER_MODES = {
    "YT_IDV_MAX_INTENSITY", "YT_IDV_TRANSFER_FUNCTION", "YT_IDV_PROJECTION",
}


class VolumeOverridePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    renderer_mode: bpy.props.EnumProperty(
        name="Renderer",
        description="Which render_volume() implementation the two-pass pipeline actually calls",
        items=[
            ("STUB", "Real renderer (stub)", "external_render.render_volume() -- raises NotImplementedError until wired in"),
            ("DEBUG_FLAT", "Debug: flat placeholder", "A flat semi-transparent color filling the whole frame, ignoring geometry -- see external_render.debug_fake_render_volume()"),
            ("SAMPLE_CONSTANT", "Sample: depth x constant", "A real (if non-physical) demo: ray/volume-box intersection, depth-clipped, output = path length x a constant -- see sample_renderers.constant_density_renderer()"),
            ("YT_IDV_MAX_INTENSITY", "yt_idv: Maximum Intensity Projection (exploratory)", "Real yt_idv offscreen volume renderer -- hardcoded IsolatedGalaxy sample dataset, depth-clipped against Cycles' own Z pass via the flagged object's bounding box as registration. Needs yt_idv installed in Blender's own Python -- see README 'yt_idv integration'. See yt_idv_renderer.py module docstring for exactly what is/isn't real here."),
            ("YT_IDV_MAX_INTENSITY_ALPHA", "yt_idv: Maximum Intensity, value-driven alpha (exploratory)", "Same Maximum Intensity Projection as above, but alpha is driven by the selected value itself (transparent at the low end of what's visible, opaque at the high end) instead of a binary presence flag. Same hardcoded dataset/depth-clip/camera registration -- see yt_idv_renderer.py module docstring."),
            ("YT_IDV_TRANSFER_FUNCTION", "yt_idv: Transfer Function (exploratory)", "Real yt_idv offscreen volume renderer with an actual transfer function -- isocontour layers at fixed densities in log space, yt.ColorTransferFunction.add_layers(). Same hardcoded dataset/depth-clip/camera registration as Maximum Intensity Projection -- see yt_idv_renderer.py module docstring."),
            ("YT_IDV_PROJECTION", "yt_idv: Projective Integration (exploratory)", "Real yt_idv offscreen volume renderer using unweighted line-of-sight integration (additive, not max) -- tends to look smoother/less speckled than Maximum Intensity Projection. Same hardcoded dataset/depth-clip/camera registration -- see yt_idv_renderer.py module docstring."),
            ("YT_IDV_THRESHOLD_FOG", "yt_idv: Threshold Fog (exploratory)", "Real yt_idv offscreen volume renderer with a continuous transfer function -- fully transparent below a density threshold, alpha ramps up above it for a hazy/foggy look, instead of isocontour lines. Same hardcoded dataset/depth-clip/camera registration as the other yt_idv modes -- see yt_idv_renderer.py module docstring."),
        ],
        default="STUB",
    )
    work_dir: bpy.props.StringProperty(
        name="Working Directory",
        description="Where per-frame EXR intermediates are written. Empty = system temp dir",
        subtype="DIR_PATH",
        default="",
    )
    verbose_logging: bpy.props.BoolProperty(
        name="Verbose Logging",
        description=(
            "Print timestamped progress for every step of the two-pass render "
            "pipeline to Blender's system console (Window > Toggle System "
            "Console on Windows; the terminal you launched Blender from on "
            "Linux/macOS). Turn on if a render seems stuck or fails -- the "
            "last line printed tells you which step it's in"
        ),
        default=True,
    )
    yt_idv: bpy.props.PointerProperty(type=YtIdvRenderSettings)

    def draw(self, context):
        self.layout.prop(self, "renderer_mode")
        self.layout.prop(self, "work_dir")
        if self.renderer_mode.startswith("YT_IDV_"):
            # Only shown for yt_idv renderer_modes -- see YtIdvRenderSettings'
            # own docstring for why these live in a separate PropertyGroup
            # rather than flat fields here (keeps this panel uncluttered for
            # every other renderer_mode, current or future).
            box = self.layout.box()
            box.label(text="yt_idv Renderer Parameters (exploratory)", icon="EXPERIMENTAL")
            box.prop(self.yt_idv, "backend")
            if self.renderer_mode == "YT_IDV_TRANSFER_FUNCTION":
                col = box.column(align=True)
                col.label(text="Isocontour Transfer Function")
                col.prop(self.yt_idv, "tf_value_min")
                col.prop(self.yt_idv, "tf_value_max")
                col.prop(self.yt_idv, "tf_n_layers")
                col.prop(self.yt_idv, "tf_colormap")
            elif self.renderer_mode == "YT_IDV_THRESHOLD_FOG":
                col = box.column(align=True)
                col.label(text="Threshold Fog Transfer Function")
                col.prop(self.yt_idv, "fog_value_min")
                col.prop(self.yt_idv, "fog_value_max")
                col.prop(self.yt_idv, "fog_colormap")
                col.prop(self.yt_idv, "fog_alpha_max")
                col.prop(self.yt_idv, "fog_ramp_power")
            elif self.renderer_mode == "YT_IDV_MAX_INTENSITY_ALPHA":
                col = box.column(align=True)
                col.label(text="Maximum Intensity, Value-Driven Alpha")
                col.prop(self.yt_idv, "mip_alpha_colormap")
                col.prop(self.yt_idv, "mip_alpha_max")
                col.prop(self.yt_idv, "mip_alpha_ramp_power")
            if self.renderer_mode in ("YT_IDV_THRESHOLD_FOG", "YT_IDV_MAX_INTENSITY_ALPHA"):
                # Curve-driven alpha / image-driven color -- shared by both
                # continuous-ramp modes (one setting, not two independent
                # copies); NOT offered for YT_IDV_TRANSFER_FUNCTION, whose
                # isocontour-layers shape doesn't reduce to a single
                # alpha-vs-position curve the same way -- see
                # yt_idv_renderer.py module docstring's "Curve-driven alpha
                # / image-driven color" section.
                curve_box = box.box()
                curve_box.label(text="Alpha Source")
                curve_box.prop(self.yt_idv, "alpha_source", expand=True)
                if self.yt_idv.alpha_source == "CURVE":
                    curve_box.template_curve_mapping(_get_or_create_alpha_curve_node(), "mapping")
                curve_box.label(text="Color Source")
                curve_box.prop(self.yt_idv, "color_source", expand=True)
                if self.yt_idv.color_source == "IMAGE":
                    curve_box.prop_search(self.yt_idv, "color_image_name", bpy.data, "images")
        self.layout.prop(self, "verbose_logging")
        self.layout.operator("volume_override.reset_change_tracking")


def _get_prefs(context):
    return context.preferences.addons[__package__].preferences


def _work_dir(context):
    prefs = _get_prefs(context)
    d = prefs.work_dir or os.path.join(tempfile.gettempdir(), "cycles_volume_override")
    os.makedirs(d, exist_ok=True)
    return d


_FRAME_DIR_PREFIX = "frame_"


def _sweep_stale_frame_dirs(work_dir):
    """Best-effort cleanup of per-frame temp directories left behind by a
    run that didn't exit cleanly.

    `render_current_frame`'s own `finally` block (below) reliably cleans up
    its OWN frame directory for an ordinary Python exception -- but a hard
    crash bypasses `finally` entirely: confirmed real during the GPU-leak
    investigation, an X11 `BadAccess` GLX error took the whole process down
    without ever unwinding through Python (Xlib's default error handler
    calls `exit()` directly at the C level). `kill -9` and a plain segfault
    have the same problem. Rather than trying to catch every possible hard-
    crash mode (a losing battle), this instead makes the NEXT run clean up
    after a previous one: called once at the start of every
    `render_current_frame`, before that frame's own directory is created,
    sweeping any leftover `frame_*` directory already in `work_dir`.

    Safe to sweep unconditionally: this add-on's pipeline is always
    strictly one-frame-at-a-time within a single process, so by the time a
    NEW `render_current_frame` call starts, any `frame_*` directory already
    present MUST be from an earlier frame that either already finished
    (and would have removed its own directory in `finally`) or crashed
    hard (and left it behind) -- never one still legitimately in use.
    Matches only our own `frame_*` prefix, so a user's configured `work_dir`
    (see `AddonPreferences.work_dir`) never loses anything else it holds.
    """
    try:
        entries = os.listdir(work_dir)
    except OSError:
        return
    for name in entries:
        if not name.startswith(_FRAME_DIR_PREFIX):
            continue
        path = os.path.join(work_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# RenderRequest construction
# ---------------------------------------------------------------------------

def _matrix_to_list(matrix):
    return [list(row) for row in matrix]


def build_camera_params(scene, camera_obj):
    cam = camera_obj.data
    render = scene.render

    projection_matrix = None
    if cam.type in ("PERSP", "ORTHO"):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        matrix = camera_obj.calc_matrix_camera(
            depsgraph,
            x=render.resolution_x, y=render.resolution_y,
            scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y,
        )
        projection_matrix = _matrix_to_list(matrix)

    panorama = build_panorama_params(cam) if cam.type == "PANO" else None

    return external_render.CameraParams(
        matrix_world=_matrix_to_list(camera_obj.matrix_world),
        projection=cam.type,  # "PERSP" | "ORTHO" | "PANO"
        lens_mm=cam.lens if cam.type == "PERSP" else None,
        sensor_width_mm=cam.sensor_width,
        sensor_height_mm=cam.sensor_height,
        sensor_fit=cam.sensor_fit,
        shift_x=cam.shift_x,
        shift_y=cam.shift_y,
        ortho_scale=cam.ortho_scale if cam.type == "ORTHO" else None,
        clip_start=cam.clip_start,
        clip_end=cam.clip_end,
        resolution_x=render.resolution_x,
        resolution_y=render.resolution_y,
        pixel_aspect_x=render.pixel_aspect_x,
        pixel_aspect_y=render.pixel_aspect_y,
        projection_matrix=projection_matrix,
        panorama=panorama,
        dof=build_dof_params(camera_obj, cam),
    )


def _panorama_settings_source(cam):
    """Blender < 5.0 (probable, per the project's own compositor idiom --
    see README "Blender 5.0+ compositor"): panorama/fisheye fields live under
    `camera.data.cycles`, a PropertyGroup the Cycles addon registers onto
    Camera. On the pip `bpy==5.2.0` used to develop this, they were found to
    be core `Camera` attributes directly (`hasattr(cam, 'cycles')` is False
    there) -- apparently merged in Blender 5.0's camera/engine unification.
    Not independently confirmed on a real <5.0 Blender; if this guesses
    wrong for your version, `hasattr` will just be False when it should be
    True (or vice versa) and the AttributeError below will say so plainly.
    """
    return cam.cycles if hasattr(cam, "cycles") else cam


def build_panorama_params(cam):
    src = _panorama_settings_source(cam)
    return external_render.PanoramicLensParams(
        panorama_type=src.panorama_type,
        fisheye_lens_mm=src.fisheye_lens,
        fisheye_fov_rad=src.fisheye_fov,
        fisheye_polynomial_k0=src.fisheye_polynomial_k0,
        fisheye_polynomial_k1=src.fisheye_polynomial_k1,
        fisheye_polynomial_k2=src.fisheye_polynomial_k2,
        fisheye_polynomial_k3=src.fisheye_polynomial_k3,
        fisheye_polynomial_k4=src.fisheye_polynomial_k4,
        latitude_min_rad=src.latitude_min,
        latitude_max_rad=src.latitude_max,
        longitude_min_rad=src.longitude_min,
        longitude_max_rad=src.longitude_max,
        central_cylindrical_radius=getattr(src, "central_cylindrical_radius", 0.0),
        central_cylindrical_range_u_min=getattr(src, "central_cylindrical_range_u_min", 0.0),
        central_cylindrical_range_u_max=getattr(src, "central_cylindrical_range_u_max", 0.0),
        central_cylindrical_range_v_min=getattr(src, "central_cylindrical_range_v_min", 0.0),
        central_cylindrical_range_v_max=getattr(src, "central_cylindrical_range_v_max", 0.0),
    )


def build_dof_params(camera_obj, cam):
    dof = cam.dof
    focus_distance = dof.focus_distance
    target = dof.focus_object
    if target is not None:
        if dof.focus_subtarget and target.type == "ARMATURE" and target.pose:
            bone = target.pose.bones.get(dof.focus_subtarget)
            target_loc = (target.matrix_world @ bone.matrix).translation if bone else target.matrix_world.translation
        else:
            target_loc = target.matrix_world.translation
        focus_distance = (target_loc - camera_obj.matrix_world.translation).length
    return external_render.DepthOfFieldParams(
        use_dof=dof.use_dof,
        focus_distance_m=focus_distance,
        aperture_fstop=dof.aperture_fstop,
        aperture_blades=dof.aperture_blades,
        aperture_ratio=dof.aperture_ratio,
        aperture_rotation_rad=dof.aperture_rotation,
    )


def _yt_idv_custom_params():
    """yt_idv-specific rendering knobs from the add-on's own Preferences
    panel (`VolumeOverridePreferences.yt_idv`, a `YtIdvRenderSettings`),
    merged into EVERY `VolumeObjectParams.custom_params` dict regardless of
    which `renderer_mode` is actually active -- harmless for non-yt_idv
    renderers, which simply never look at `yt_idv_*`-prefixed keys, same as
    any other `custom_params` entry they don't recognize (see
    external_render.py's own docstring on why `custom_params` is the
    correct place to add renderer-specific knobs without touching the
    frozen dataclass contract). See `yt_idv_renderer.py`'s `_configure_*`
    functions for what each key controls and its own hardcoded fallback
    default -- the DEFAULTS on `YtIdvRenderSettings`' fields just need to
    stay in agreement with those, they're not the single source of truth.

    Falls back to `{}` (not an exception) when the add-on's own preferences
    aren't registered/reachable -- true in this repo's own lightweight unit
    tests, which register `properties` but not `operators`/`preferences`
    (see tests/test_operators.py's `clean_scene_with_props` fixture) --
    so `build_volume_params(obj)` keeps working unchanged there.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
    except KeyError:
        return {}
    s = prefs.yt_idv
    params = {
        "yt_idv_backend": s.backend,
        "yt_idv_tf_value_min": s.tf_value_min,
        "yt_idv_tf_value_max": s.tf_value_max,
        "yt_idv_tf_n_layers": s.tf_n_layers,
        "yt_idv_tf_colormap": s.tf_colormap,
        "yt_idv_fog_value_min": s.fog_value_min,
        "yt_idv_fog_value_max": s.fog_value_max,
        "yt_idv_fog_colormap": s.fog_colormap,
        "yt_idv_fog_alpha_max": s.fog_alpha_max,
        "yt_idv_fog_ramp_power": s.fog_ramp_power,
        "yt_idv_mip_alpha_colormap": s.mip_alpha_colormap,
        "yt_idv_mip_alpha_max": s.mip_alpha_max,
        "yt_idv_mip_alpha_ramp_power": s.mip_alpha_ramp_power,
    }
    # Curve-driven alpha / image-driven color -- only computed (and only
    # added to custom_params) when actually selected, so the RAMP_POWER/
    # NAMED-colormap defaults above are completely unaffected by this
    # feature existing at all -- yt_idv_renderer.py's `_configure_threshold_
    # fog`/`_configure_max_intensity_alpha` just don't find these keys and
    # fall back to ramp_power/named-colormap exactly as before. One shared
    # curve/image, not four independent copies -- see YtIdvRenderSettings'
    # own `alpha_source`/`color_source` fields' docstrings for why sharing
    # across Fog/MIP-alpha is the right amount of complexity here.
    if s.alpha_source == "CURVE":
        params["yt_idv_alpha_curve"] = _evaluate_alpha_curve()
    if s.color_source == "IMAGE" and s.color_image_name:
        image = bpy.data.images.get(s.color_image_name)
        if image is not None:  # stale name (image renamed/deleted) -- fall back to NAMED silently rather than raise mid-render
            params["yt_idv_color_image_rgba"] = _sample_image_as_1d_rgba(image)
    return params


def build_volume_params(obj):
    custom_params = obj.volume_override.to_custom_params()
    custom_params.update(_yt_idv_custom_params())
    cache_key = properties.ensure_dataset_uuid(obj)
    return external_render.VolumeObjectParams(
        matrix_world=_matrix_to_list(obj.matrix_world),
        bounding_box_local=[list(corner) for corner in obj.bound_box],
        custom_params=custom_params,
        params_version=change_tracking.version_for(cache_key, "params", custom_params),
        dataset=build_dataset_params(obj, cache_key),
    )


def _safe_mtime(filepath):
    return os.path.getmtime(filepath) if filepath and os.path.exists(filepath) else None


def resolve_sequence_frame(frame_start, frame_duration, frame_offset, sequence_mode, scene_frame):
    """Best-effort replica of Blender's Volume-sequence frame remapping
    (start/duration/offset + clip/extend/repeat/ping-pong), used to populate
    `DatasetParams.resolved_sequence_frame`.

    NOT independently empirically verified against Blender's actual internal
    resolution the way the Z-pass/camera-projection findings were -- this
    follows the documented semantics of frame_start/frame_duration/
    frame_offset/sequence_mode, but hasn't been checked against a real VDB
    sequence's actual per-frame file selection. Re-verify (same marker-style
    technique used elsewhere in this repo, adapted: render a sequence with
    known frame contents and confirm which frame shows at which scene frame)
    before trusting the exact number for anything precision-sensitive.
    """
    if not frame_duration:
        return None
    local = scene_frame - frame_start + frame_offset
    n = frame_duration
    if sequence_mode == "CLIP":
        return local if 0 <= local < n else None
    if sequence_mode == "EXTEND":
        return max(0, min(n - 1, local))
    if sequence_mode == "REPEAT":
        return local % n
    if sequence_mode == "PING_PONG":
        period = 2 * n
        local = local % period
        return local if local < n else period - 1 - local
    return local


def build_dataset_params(obj, cache_key):
    """Dataset identity for `obj`, generic-first (see external_render.py's
    `DatasetParams` docstring): primarily whatever the user typed into
    `properties.py`'s `dataset_path`/`dataset_kind` fields, which works for
    ANY dataset your renderer's loaders understand (a yt-loadable simulation
    output, an HDF5 file, ...) regardless of what kind of Blender object is
    carrying the flag.

    If `obj` happens to be a Blender `Volume` object (native VDB support),
    override path/sequence fields with what Blender itself already knows
    about it instead of trusting the user to duplicate that by hand.
    """
    props = obj.volume_override
    dataset_path = bpy.path.abspath(props.dataset_path) if props.dataset_path else None
    dataset_kind = props.dataset_kind
    is_sequence = False
    frame_start = frame_duration = frame_offset = None
    sequence_mode = None
    resolved_frame = None
    grid_names = []

    if obj.type == "VOLUME":
        vol = obj.data
        if vol.filepath:
            dataset_path = bpy.path.abspath(vol.filepath)
            dataset_kind = dataset_kind or "vdb"
        is_sequence = vol.is_sequence
        grid_names = [g.name for g in vol.grids]
        if is_sequence:
            frame_start, frame_duration, frame_offset = vol.frame_start, vol.frame_duration, vol.frame_offset
            sequence_mode = vol.sequence_mode
            resolved_frame = resolve_sequence_frame(
                frame_start, frame_duration, frame_offset, sequence_mode,
                bpy.context.scene.frame_current,
            )

    # NOTE: `resolved_frame` (and `requested_time` below) are deliberately
    # NOT part of `identity` -- they change every frame by design, and
    # hashing them in would make `version.changed` report True every frame
    # regardless of whether the renderer would actually need to reload
    # anything. `identity` covers only what's genuinely expensive to change:
    # which dataset this object points to, not which moment of it.
    identity = {
        "dataset_path": dataset_path,
        "dataset_kind": dataset_kind,
        "mtime": _safe_mtime(dataset_path),
        "is_sequence": is_sequence,
    }
    return external_render.DatasetParams(
        dataset_id=cache_key,
        dataset_kind=dataset_kind,
        dataset_path=dataset_path,
        requested_time=resolve_requested_time(obj),
        time_units=props.time_units,
        mapping_policy=props.mapping_policy,
        is_blender_volume_sequence=is_sequence,
        sequence_frame_start=frame_start,
        sequence_frame_duration=frame_duration,
        sequence_frame_offset=frame_offset,
        sequence_mode=sequence_mode,
        resolved_sequence_frame=resolved_frame,
        grid_names=grid_names,
        version=change_tracking.version_for(cache_key, "dataset", identity),
    )


def resolve_requested_time(obj):
    """DIRECT/AFFINE/MANUAL policy resolution -- see properties.py's
    `mapping_policy` field and README "Dataset / change tracking" -> "Time
    mapping" for the full design discussion (in particular, why this is a
    plain Python function and not a Blender Driver: a driver alone can
    produce this same scalar, but has no way to know what timesteps the
    dataset actually has -- that snapping step still has to happen
    somewhere, and it happens on the renderer side, not here).

    MANUAL reads `manual_time` directly -- it doesn't care whether that
    property got its value from a plain keyframe or a Driver, both work
    identically from this function's point of view (it just reads the
    already-evaluated property; by the time this runs, the depsgraph has
    already been updated for this frame, same requirement `matrix_world`
    already relies on elsewhere in this module).
    """
    props = obj.volume_override
    frame = bpy.context.scene.frame_current
    if props.mapping_policy == "DIRECT":
        return float(frame)
    if props.mapping_policy == "AFFINE":
        return frame * props.time_scale + props.time_offset
    if props.mapping_policy == "MANUAL":
        return props.manual_time
    raise ValueError(f"unknown mapping_policy {props.mapping_policy!r}")


def build_frame_params(scene):
    fps = scene.render.fps / scene.render.fps_base
    return external_render.FrameParams(
        frame_number=scene.frame_current,
        scene_time_seconds=scene.frame_current / fps,
    )


# --- Depth read-back --------------------------------------------------------
#
# Findings baked in here come from scripts/verify_zpass.py (see README
# "Z-pass verification findings"):
#   depth_convention = "PLANAR"   -- empirically measured on Cycles, NOT
#                                    Euclidean ray distance as commonly
#                                    assumed; center and off-axis pixels on a
#                                    plane perpendicular to view axis read
#                                    the SAME value.
#   no_hit_value     = 1e10       -- exact sentinel Cycles wrote for
#                                    background/no-geometry pixels.
#   row_order        = "BOTTOM_LEFT_ORIGIN" -- this is Blender's documented
#                                    Image.pixels convention (row 0 =
#                                    bottom). NOT independently closed-loop
#                                    verified in this environment because
#                                    the pip `bpy` wheel used for the
#                                    verification script has a broken EXR
#                                    decoder (bpy.data.images.load()
#                                    returns a 0x0 image); the raw on-disk
#                                    EXR scanline order WAS confirmed
#                                    top-first via a direct OpenEXR-library
#                                    read, which is consistent with (not
#                                    proof of) this claim about a real
#                                    Blender's in-memory buffer. Re-run
#                                    scripts/verify_zpass.py against a real
#                                    Blender to close this loop -- see
#                                    README "Verification caveats".

DEPTH_CONVENTION = "PLANAR"
DEPTH_NO_HIT_VALUE = 1e10
DEPTH_ROW_ORDER = "BOTTOM_LEFT_ORIGIN"


def read_depth_buffer(scene, path):
    import numpy as np

    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        channels = img.channels
        if w == 0 or h == 0 or channels == 0:
            # Real bug hit once (Blender 5.x, real install, not just the pip
            # test module): bpy.data.images.load() + .pixels silently
            # decoded a genuinely-written, non-empty EXR as 0x0/0-channel
            # instead of raising -- traced to compositor_build.py writing
            # this file as a NAMED multi-part EXR, which that Python-facing
            # pixel-buffer path doesn't reliably support (see
            # _configure_single_file_output's docstring for the fix -- item
            # names are empty now). Failing loudly here with the actual
            # on-disk facts is the safety net if that regresses, or if some
            # other Blender version hits the same failure mode differently.
            on_disk = os.path.exists(path)
            size_bytes = os.path.getsize(path) if on_disk else None
            raise RuntimeError(
                f"read_depth_buffer: {path!r} loaded as size={img.size[:]} "
                f"channels={channels} (expected a real image) -- "
                f"file exists on disk: {on_disk}, size: {size_bytes} bytes, "
                f"img.file_format={img.file_format!r}. See "
                "_configure_single_file_output's docstring in compositor_build.py."
            )
        data = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, channels)
        data = data[..., 0]  # single-channel Z; ignore any padding channels
    finally:
        bpy.data.images.remove(img)

    scale_length = scene.unit_settings.scale_length
    return external_render.DepthBuffer(
        data=data * scale_length,
        depth_convention=DEPTH_CONVENTION,
        no_hit_value=DEPTH_NO_HIT_VALUE * scale_length,
        row_order=DEPTH_ROW_ORDER,
        units="meters",
    )


def write_render_result(result, path):
    """Write a RenderResult.rgba (H, W, 4) float32 array to a full-float EXR.

    Assumes `result.rgba` uses the SAME row order as the DepthBuffer it was
    given (BOTTOM_LEFT_ORIGIN, per above) -- the renderer received that
    convention via DepthBuffer.row_order and is expected to echo it, since
    Image.pixels (used both to read depth back and here to write the
    result) only ever works in that one convention.
    """
    h, w = result.rgba.shape[:2]
    img = bpy.data.images.new(f"vo_result_tmp", width=w, height=h, alpha=True, float_buffer=True)
    compositor_build.set_linear_colorspace(img)
    img.pixels.foreach_set(result.rgba.astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)


# --- Per-frame pipeline ------------------------------------------------------

def _sorted_far_to_near(scene, flagged):
    cam_loc = scene.camera.matrix_world.translation
    def distance(obj):
        return (obj.matrix_world.translation - cam_loc).length
    return sorted(flagged, key=distance, reverse=True)


def _apply_opacity(result, opacity):
    """`VolumeOverrideProperties.opacity` (a real, keyframable Object-level
    property -- see that field's own docstring for why it lives there
    rather than alongside the yt_idv-specific alpha controls in
    AddonPreferences, which can't be keyframed at all: confirmed directly,
    `AddonPreferences` instances have no `animation_data`/`id_data`,
    `keyframe_insert` raises outright) applied uniformly to EVERY
    renderer's output, here, once -- rather than each `RENDERER_MODES`
    entry needing to remember to read a `custom_params['opacity']` key
    itself (which future/third-party renderers could easily forget).

    Multiplying the WHOLE `(H, W, 4)` premultiplied-alpha RGBA array by a
    single scalar is exact, not an approximation: for premultiplied alpha,
    `rgb = color * alpha`, so scaling alpha by `f` while keeping the same
    underlying `color` means the new premultiplied rgb is `color * (alpha *
    f) = (color * alpha) * f = rgb * f` -- i.e. exactly the same scalar `f`
    applies to the alpha channel and the premultiplied color channels
    alike. Relies on `result.alpha_mode == "PREMULTIPLIED"`, already
    asserted on every `RenderResult` right before this is called.

    `opacity == 1.0` (the default) is a no-op that skips the array copy
    entirely -- zero cost for every object that never touches this field.
    """
    if opacity == 1.0:
        return result
    return external_render.RenderResult(
        # .astype(...) rather than a bare `*` -- keeps float32 regardless of
        # numpy's scalar-promotion rules for `float32_array * python_float`
        # (no module-level numpy import needed here either way; `result.rgba`
        # is already an ndarray).
        rgba=(result.rgba * opacity).astype(result.rgba.dtype),
        alpha_mode=result.alpha_mode,
        color_space=result.color_space,
    )


def render_current_frame(context):
    """Runs the full two-pass pipeline for `scene.frame_current`. Restores
    object visibility even if any step raises.
    """
    scene = context.scene
    if scene.camera is None:
        raise RuntimeError("Scene has no active camera")

    prefs = _get_prefs(context)
    verbose = prefs.verbose_logging
    log = _log if verbose else (lambda *a, **k: None)
    step = _StepTimer if verbose else _NullStepTimer

    log(f"render_current_frame: frame={scene.frame_current} engine={scene.render.engine} "
        f"resolution={scene.render.resolution_x}x{scene.render.resolution_y}")

    work_dir = _work_dir(context)
    _sweep_stale_frame_dirs(work_dir)
    frame_dir = tempfile.mkdtemp(prefix=_FRAME_DIR_PREFIX, dir=work_dir)
    flagged = properties.flagged_objects(scene)
    log(f"flagged objects: {[o.name for o in flagged]}, frame_dir={frame_dir}")
    scene_sync.sync_scene_state(scene, flagged)

    view_layer = context.view_layer
    prev_use_pass_z = view_layer.use_pass_z
    view_layer.use_pass_z = True

    prev_engine = scene.render.engine
    prev_compositor_device = scene.render.compositor_device
    prev_view_transform = scene.view_settings.view_transform
    if prefs.renderer_mode in YT_IDV_RENDERER_MODES:
        if prev_view_transform != "Standard":
            # Real, reproducible finding, not a style preference: Blender's
            # `AgX` view transform (the default on every new file/scene
            # since 4.0 -- i.e. this is what a "pretty much exactly
            # default" scene has, not an unusual choice) applies a filmic
            # highlight rolloff meant for photographic renders. Applied to
            # an already-colormapped scientific visualization, it crushes
            # the real signal into a flat, low-contrast plateau instead of
            # preserving it. Confirmed directly, same frame/data/camera,
            # nothing else changed: AgX topped out at 197/255 with ~95% of
            # pixels packed into ONE narrow brightness band; `Standard`
            # reached 254/255 with real gradation across the volume's own
            # visible range. Erring on the side of correct-looking output
            # by default here -- restored to whatever it was afterward
            # (below), and only for yt_idv modes (the other renderer modes
            # don't go through this add-on's own colormapping at all, so
            # whatever view transform you picked for them is left alone).
            _log(f"WARNING: scene view_transform was {prev_view_transform!r} -- forcing "
                 f"'Standard' for this yt_idv render (restored to {prev_view_transform!r} "
                 f"afterward). {prev_view_transform!r} applies a filmic highlight rolloff "
                 f"that crushes an already-colormapped scientific visualization into a flat, "
                 f"low-contrast image -- see operators.render_current_frame's own comment for "
                 f"the direct A/B measurement. Set your scene's View Transform to 'Standard' "
                 f"(or 'Raw') yourself to silence this.")
            scene.view_settings.view_transform = "Standard"
        # Real, reproducible finding: Blender's own realtime compositor
        # (the default `compositor_device == "GPU"`) grabs a GPU/EGL context
        # as soon as ANY render runs through a scene's compositor node tree
        # -- even a trivial one with no GPU-only nodes at all (confirmed:
        # reproduces with nothing but a Render Layers -> File Output graph,
        # i.e. exactly pass 1's own graph below). Once that's happened in
        # THIS process, yt_idv's own subsequent `yt_idv.render_context("egl",
        # ...)` call fails with `EGLError(err=EGL_BAD_ACCESS, baseOperation=
        # eglMakeCurrent, ...)` -- confirmed via a minimal repro isolating
        # exactly this (Cycles render alone: fine; Cycles render through ANY
        # compositor tree, then a fresh yt_idv EGL context: fails every
        # time). Forcing the compositor to `"CPU"` for the duration of a
        # yt_idv-mode render avoids the GPU/EGL grab entirely and was
        # confirmed to fix it end to end (see scripts/yt_idv_exploration/
        # blender_scene/render_real_demo.py, now runs fully in-process with
        # real bpy + real yt_idv in one interpreter). Scoped to yt_idv modes
        # only -- the other renderer modes never touch EGL/GL, so there's no
        # reason to override this scene setting for them.
        scene.render.compositor_device = "CPU"
    # This frame's own full-float EXR intermediates (beauty/depth/per-object
    # volume results) all live in `frame_dir`, a fresh per-frame directory
    # (see `_sweep_stale_frame_dirs`'s own docstring for the real bug this
    # replaced: nothing ever deleted these before, so a long run silently
    # accumulated one full set per frame in `work_dir` forever -- at HD
    # resolution over a few hundred frames that's easily multiple GB, and it
    # crashed a real render outright with ENOSPC). One `shutil.rmtree` in
    # `finally` below cleans up everything this frame created, success or
    # exception alike -- simpler than tracking individual file paths, and
    # automatically covers any future file this pipeline starts writing
    # without needing a matching update here.
    try:
        # --- Pass 1: real Cycles beauty + Z, flagged objects hidden -----
        with step("pass1 build compositor graph"):
            beauty_path, depth_path = compositor_build.build_pass1_capture_graph(
                scene, frame_dir, scene.frame_current,
            )
        log(f"pass1 will write beauty={beauty_path!r} depth={depth_path!r}")
        with scene_sync.hidden_from_main_render(flagged):
            with step(f"pass1 render (bpy.ops.render.render), engine={scene.render.engine}"):
                bpy.ops.render.render(write_still=False)
        log(f"pass1 wrote beauty exists={os.path.exists(beauty_path)} depth exists={os.path.exists(depth_path)}")

        with step("read_depth_buffer"):
            depth_buffer = read_depth_buffer(scene, depth_path)
        camera_params = build_camera_params(scene, scene.camera)
        frame_params = build_frame_params(scene)

        render_fn = RENDERER_MODES[prefs.renderer_mode]
        log(f"renderer_mode={prefs.renderer_mode!r} ({render_fn.__module__}.{render_fn.__qualname__})")

        volume_layers = []
        for obj in _sorted_far_to_near(scene, flagged):
            request = external_render.RenderRequest(
                camera=camera_params,
                depth=depth_buffer,
                volume=build_volume_params(obj),
                frame=frame_params,
            )
            with step(f"render_volume for {obj.name!r}"):
                result = render_fn(request)
            assert result.alpha_mode == "PREMULTIPLIED"
            assert result.color_space == "LINEAR"
            result = _apply_opacity(result, obj.volume_override.opacity)
            result_path = os.path.join(
                frame_dir, f"vo_result_{obj.name}_{scene.frame_current:04d}.exr",
            )
            with step(f"write_render_result for {obj.name!r} -> {result_path!r}"):
                write_render_result(result, result_path)
            volume_layers.append((obj.name, result_path))

        # --- Pass 2: compositor-only (cheap engine), writes the real output
        with step("pass2 build/update compositor graph"):
            compositor_build.build_or_update_graph(scene, beauty_path, volume_layers)
        scene.render.engine = PASS2_ENGINE
        with step(f"pass2 render (bpy.ops.render.render, write_still=True), engine={PASS2_ENGINE}"):
            bpy.ops.render.render(write_still=True)
        log(f"pass2 done; final output at {scene.render.frame_path()!r} "
            f"(Render Result image also holds it -- see _show_render_result)")
    finally:
        scene.render.engine = prev_engine
        view_layer.use_pass_z = prev_use_pass_z
        scene.render.compositor_device = prev_compositor_device
        scene.view_settings.view_transform = prev_view_transform
        shutil.rmtree(frame_dir, ignore_errors=True)


def _show_render_result(context):
    """Pop Blender's own Render Result viewer, same as a normal F12 press.

    Our two-pass pipeline calls `bpy.ops.render.render()` in plain
    EXEC_DEFAULT (no `'INVOKE_DEFAULT'`) so it stays fully synchronous --
    pass 2 must not start until pass 1 and the per-object renders are done,
    and the render-engine swap around it (see module docstring) would race
    against an async invoke. The tradeoff: EXEC_DEFAULT never pops the
    Render Result window or updates an open Image Editor the way a normal
    F12 does, even though `bpy.data.images["Render Result"]` genuinely holds
    our correct composited output afterward (confirmed -- pass 2's own
    render.render() call always updates it) -- reported as "I don't see it
    anywhere" even though real EXR files were written correctly.
    `render.view_show()` only displays whatever's already there; it doesn't
    re-render, so calling it after the fact is safe regardless of how the
    pipeline above ran.

    Skipped entirely in `--background`/headless runs (`bpy.app.background`)
    -- there's no window capable of showing an image there, and calling it
    anyway doesn't raise a catchable Python exception (confirmed directly):
    `RENDER_OT_view_show`'s own poll failure is reported straight to
    Blender's console/log as an `ERROR`, bypassing the `try/except
    RuntimeError` below entirely, so that guard alone never actually
    silenced it. The render itself already succeeded and wrote real output
    files either way -- this is purely about not printing a scary-looking
    (but harmless) ERROR line on every single headless frame.
    """
    if bpy.app.background:
        _log("background mode -- not attempting to show Render Result (output was still written to disk/Render Result)")
        return
    try:
        bpy.ops.render.view_show("INVOKE_DEFAULT")
    except RuntimeError as exc:
        # Not fatal -- the render itself already succeeded and wrote real
        # output files; failing to pop the viewer (e.g. no valid window
        # context) shouldn't be reported as a render failure.
        _log(f"could not show Render Result ({exc}) -- output was still written to disk/Render Result")


class VOLUMEOVERRIDE_OT_render(bpy.types.Operator):
    """Single F12 press: Cycles+Z pass -> external volume renderer -> depth-
    clipped Alpha Over composite -> saved output. See operators.py module
    docstring for why this is two internal render passes.
    """

    bl_idname = "render.volume_override"
    bl_label = "Render (with Volume Override)"
    bl_options = {"REGISTER"}

    animation: bpy.props.BoolProperty(name="Animation", default=False)
    _timer = None
    _frame_iter = None

    def execute(self, context):
        if not properties.flagged_objects(context.scene):
            # Nothing in this scene opts into the override -- delegate
            # straight to Blender's own render operator instead of running
            # the two-pass pipeline (which forces the Z pass on, rebuilds
            # the compositor graph, and swaps the engine to Workbench for
            # the final write). Without this, enabling the add-on breaks
            # ordinary Cycles rendering for every scene, even ones that
            # never use the override -- F12/Ctrl+F12 are unconditionally
            # remapped to this operator (see handlers.py), so this check is
            # what keeps that remap a no-op until a volume is actually
            # flagged.
            return bpy.ops.render.render("INVOKE_DEFAULT", animation=self.animation)

        if not self.animation:
            try:
                render_current_frame(context)
            except Exception as exc:
                self.report({"ERROR"}, f"Volume override render failed: {exc}")
                return {"CANCELLED"}
            _show_render_result(context)
            return {"FINISHED"}

        scene = context.scene
        self._frame_iter = iter(range(scene.frame_start, scene.frame_end + 1))
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._finish(context)
            self.report({"WARNING"}, "Volume override animation render cancelled")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        frame = next(self._frame_iter, None)
        if frame is None:
            self._finish(context)
            _show_render_result(context)  # show the LAST frame rendered, like F12 would
            return {"FINISHED"}

        context.scene.frame_set(frame)
        try:
            render_current_frame(context)
        except Exception as exc:
            self._finish(context)
            self.report({"ERROR"}, f"Volume override render failed at frame {frame}: {exc}")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        self._finish(context)


def draw_render_menu(self, context):
    self.layout.operator(VOLUMEOVERRIDE_OT_render.bl_idname, text="Render (Volume Override)")
    self.layout.operator(
        VOLUMEOVERRIDE_OT_render.bl_idname, text="Render Animation (Volume Override)",
    ).animation = True


class VOLUMEOVERRIDE_OT_reset_change_tracking(bpy.types.Operator):
    """Force every flagged object's next RenderRequest to report
    params_version.changed / dataset.version.changed = True, regardless of
    whether anything actually changed. Use this when you know the
    renderer's own cache was invalidated independently of Blender (e.g. you
    restarted a subprocess renderer, or manually cleared its cache) and want
    Blender's convenience flag to stop reporting False for things it thinks
    are unchanged but the renderer no longer has cached.

    Does NOT affect `.hash` -- that's a pure content hash of the current
    state either way. Only resets the "last hash I sent" bookkeeping this
    flag is compared against.
    """

    bl_idname = "volume_override.reset_change_tracking"
    bl_label = "Reset Change Tracking"

    def execute(self, context):
        change_tracking.reset()
        self.report({"INFO"}, "Change tracking reset -- next render reports everything as changed")
        return {"FINISHED"}


CLASSES = (
    YtIdvRenderSettings,  # must precede VolumeOverridePreferences -- its PointerProperty target
    VolumeOverridePreferences,
    VOLUMEOVERRIDE_OT_render,
    VOLUMEOVERRIDE_OT_reset_change_tracking,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_render.append(draw_render_menu)
    change_tracking.reset()  # fresh registration -- see change_tracking.reset()'s docstring


def unregister():
    bpy.types.TOPBAR_MT_render.remove(draw_render_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
