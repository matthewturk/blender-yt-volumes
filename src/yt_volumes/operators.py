"""Render orchestration: the two-pass-per-frame pipeline behind the F12
override, plus the glue that builds RenderRequests from Blender data.

Two passes are needed because Blender's compositor runs inside the same
`render.render()` call that produces the Z pass, so one call can't both
produce this frame's depth and composite a volume render made from it.

  Pass 1: the scene's engine (Cycles, or EEVEE under the Vulkan GPU
  backend) renders beauty + Z, written to EXR by a File Output graph.
  Between passes: read Z back, call the renderer per flagged object, write
  each result to EXR, and build an Image -> Alpha Over -> Composite graph.
  Pass 2: renders again (same engine) purely to run that compositor graph
  and write the final output.
"""

import math
import os
import shutil
import tempfile
import time

import bpy

from . import change_tracking, compositor_build, external_render, properties, sample_renderers, scene_sync, yt_idv_renderer

# --- Debug logging -----------------------------------------------------------
# Timestamped per-step console logging, gated on the verbose_logging preference.
_LOG_PREFIX = "[VolumeOverride]"
_log_start_time = None


def _log(msg):
    global _log_start_time
    now = time.monotonic()
    if _log_start_time is None:
        _log_start_time = now
    print(f"{_LOG_PREFIX} [+{now - _log_start_time:7.2f}s] {msg}", flush=True)


class _StepTimer:
    """Context manager logging a step's start and its elapsed time on exit."""

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


def _evaluate_alpha_curve(tree, nbins=256):
    """Sample the alpha curve held in `tree` at `nbins` points over [0, 1].

    `initialize()` is required before `evaluate()` after edits; it doesn't
    reset existing points.
    """
    mapping = tree.nodes[properties.ALPHA_CURVE_NODE_NAME].mapping
    mapping.initialize()
    curve = mapping.curves[0]
    return [mapping.evaluate(curve, i / (nbins - 1)) for i in range(nbins)]


def _sample_image_as_1d_rgba(image, nbins=256):
    """Read `image` as an `(nbins, 4)` RGBA list: averaged across rows, then
    resampled horizontally. Missing channels are padded with 1.0.

    Raises `ValueError` for a 0x0 image (e.g. source file not available).
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
    row_avg = pixels.mean(axis=0)  # (w, 4)
    src_x = np.linspace(0.0, 1.0, w)
    dst_x = np.linspace(0.0, 1.0, nbins)
    out = np.empty((nbins, 4))
    for c in range(4):
        out[:, c] = np.interp(dst_x, src_x, row_avg[:, c])
    return out.tolist()


# Keyed by VolumeOverrideProperties.renderer_mode.
RENDERER_MODES = {
    "SAMPLE_CONSTANT": sample_renderers.constant_density_renderer,
    "YT_IDV_MAX_INTENSITY": yt_idv_renderer.yt_idv_max_intensity_renderer,
    "YT_IDV_MAX_INTENSITY_ALPHA": yt_idv_renderer.yt_idv_max_intensity_alpha_renderer,
    "YT_IDV_TRANSFER_FUNCTION": yt_idv_renderer.yt_idv_transfer_function_renderer,
    "YT_IDV_PROJECTION": yt_idv_renderer.yt_idv_projection_renderer,
    "YT_IDV_THRESHOLD_FOG": yt_idv_renderer.yt_idv_threshold_fog_renderer,
    "YT_IDV_YT_TRANSFER_FUNCTION": yt_idv_renderer.yt_idv_yt_transfer_function_renderer,
}

# Modes that create their own offscreen EGL context (see compositor_device in
# render_current_frame).
YT_IDV_RENDERER_MODES = {mode for mode in RENDERER_MODES if mode.startswith("YT_IDV_")}


class VolumeOverridePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

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

    def draw(self, context):
        self.layout.prop(self, "work_dir")
        self.layout.prop(self, "verbose_logging")
        self.layout.operator("volume_override.reset_change_tracking")


def _get_prefs(context):
    return context.preferences.addons[__package__].preferences


def _work_dir(context):
    prefs = _get_prefs(context)
    d = prefs.work_dir or os.path.join(tempfile.gettempdir(), "yt_volumes")
    os.makedirs(d, exist_ok=True)
    return d


_FRAME_DIR_PREFIX = "frame_"


def _sweep_stale_frame_dirs(work_dir):
    """Remove leftover `frame_*` directories in `work_dir` from a previous
    run that crashed hard (e.g. an X11/GLX error exits without unwinding
    `finally`). Safe because frames render strictly one at a time.
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
    """Where panorama/fisheye settings live: `cam.cycles` on older Blender,
    core `Camera` attributes on 5.x. (The <5.0 case is untested.)
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


def _yt_idv_custom_params(obj):
    """custom_params sampled from the object's alpha curve and color image.
    Absent unless selected; the renderer then uses the ramp/named colormap.
    """
    props = obj.volume_override
    params = {}
    if props.alpha_source == "CURVE" and props.alpha_curve is not None:
        params["yt_idv_alpha_curve"] = _evaluate_alpha_curve(props.alpha_curve)
    if props.color_source == "IMAGE" and props.color_image is not None:
        params["yt_idv_color_image_rgba"] = _sample_image_as_1d_rgba(props.color_image)
    return params


VOLUME_EMPTY_DISPLAY_TYPES = ("CUBE", "SPHERE")


def _empty_cube_bounding_box_local(obj):
    """The 8 local-space corners `[-size, +size]^3` of a flagged Empty-Cube or
    Empty-Sphere, `size` being its `empty_display_size`.

    Flagged objects must be such Empties: Empties never appear in a render,
    so they needn't be hidden for pass 1 (toggling `hide_render`
    per frame invalidates Cycles' persistent-data cache). A sphere stands in
    for spherical data, its radius the data's outer radius. Raises otherwise.
    """
    if obj.type != "EMPTY" or obj.empty_display_type not in VOLUME_EMPTY_DISPLAY_TYPES:
        raise RuntimeError(
            f"{obj.name!r} is flagged as a volume override but is not an "
            f"Empty-Cube or Empty-Sphere (type={obj.type!r}, empty_display_type="
            f"{getattr(obj, 'empty_display_type', None)!r}) -- flagged "
            "objects must be an Empty with Display Type 'Cube', or 'Sphere' "
            "for spherical data (Object Properties > Viewport Display, or "
            "Add > Empty). See build_volume_params's own docstring for why."
        )
    size = obj.empty_display_size
    return [
        [x, y, z]
        for x in (-size, size) for y in (-size, size) for z in (-size, size)
    ]


def _deep_zoom_params(scene):
    """Shot-space custom_params from the Deep Zoom extension's scene settings,
    or {} when it isn't installed or is off. Mirrors its volume_custom_params.
    """
    settings = getattr(scene, "deep_zoom", None)
    if settings is None or not settings.enabled or not settings.shots:
        return {}
    frame = scene.frame_current
    started = [s for s in settings.shots if s.frame_start <= frame]
    shot = max(started, key=lambda s: s.frame_start) if started else min(settings.shots, key=lambda s: s.frame_start)
    return {
        "yt_idv_shot_space": settings.world_scale,
        "yt_idv_zoom_target": shot.anchor,
        "yt_idv_zoom_level": settings.zoom,
    }


def build_volume_params(obj):
    custom_params = obj.volume_override.to_custom_params()
    custom_params.update(_yt_idv_custom_params(obj))
    custom_params.update(_deep_zoom_params(bpy.context.scene))
    cache_key = properties.ensure_dataset_uuid(obj)
    return external_render.VolumeObjectParams(
        matrix_world=_matrix_to_list(obj.matrix_world),
        bounding_box_local=_empty_cube_bounding_box_local(obj),
        custom_params=custom_params,
        params_version=change_tracking.version_for(cache_key, "params", custom_params),
        dataset=build_dataset_params(obj, cache_key),
    )


def _safe_mtime(filepath):
    return os.path.getmtime(filepath) if filepath and os.path.exists(filepath) else None


def resolve_sequence_frame(frame_start, frame_duration, frame_offset, sequence_mode, scene_frame):
    """Best-effort replica of Blender's Volume-sequence frame remapping
    (start/duration/offset + clip/extend/repeat/ping-pong), used to populate
    `DatasetParams.resolved_sequence_frame`. Not verified against a real
    VDB sequence.
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
    """Dataset identity for `obj`, from its `dataset_path`/`dataset_kind`
    properties; for a Blender `Volume` object, path/sequence fields come
    from the Volume itself.
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

    # Per-frame values (resolved_frame) are excluded so
    # version.changed only reflects which dataset is referenced.
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
        is_blender_volume_sequence=is_sequence,
        sequence_frame_start=frame_start,
        sequence_frame_duration=frame_duration,
        sequence_frame_offset=frame_offset,
        sequence_mode=sequence_mode,
        resolved_sequence_frame=resolved_frame,
        grid_names=grid_names,
        version=change_tracking.version_for(cache_key, "dataset", identity),
    )


def build_frame_params(scene):
    fps = scene.render.fps / scene.render.fps_base
    return external_render.FrameParams(
        frame_number=scene.frame_current,
        scene_time_seconds=scene.frame_current / fps,
    )


# --- Depth read-back --------------------------------------------------------
# Z is planar (distance along the view axis, not ray length) in Cycles and
# EEVEE alike; 1e10 marks no-hit pixels; Image.pixels rows start at the bottom.

DEPTH_CONVENTION = "PLANAR"
DEPTH_NO_HIT_VALUE = 1e10
DEPTH_ROW_ORDER = "BOTTOM_LEFT_ORIGIN"

_DEPTH_IMAGE_NAME = "vo_depth_tmp"


def read_depth_buffer(scene, path):
    """Read `path` (this frame's Z-pass EXR) into a DepthBuffer, in meters.

    Reuses one Image datablock: load()+remove() every frame invalidates
    Cycles' persistent-data cache.
    """
    import numpy as np

    img = bpy.data.images.get(_DEPTH_IMAGE_NAME)
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = _DEPTH_IMAGE_NAME
    else:
        img.filepath = path
        img.reload()
    w, h = img.size[0], img.size[1]
    channels = img.channels
    if w == 0 or h == 0 or channels == 0:
        # Blender silently loads a named multi-part EXR as 0x0; fail loudly.
        on_disk = os.path.exists(path)
        size_bytes = os.path.getsize(path) if on_disk else None
        raise RuntimeError(
            f"read_depth_buffer: {path!r} loaded as size={img.size[:]} "
            f"channels={channels} (expected a real image) -- "
            f"file exists on disk: {on_disk}, size: {size_bytes} bytes, "
            f"img.file_format={img.file_format!r}. See "
            "_configure_single_file_output's docstring in compositor_build.py."
        )
    # foreach_get, not pixels[:], which is far slower.
    buf = np.empty(h * w * channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    data = buf.reshape(h, w, channels)
    data = data[..., 0]  # single-channel Z; ignore any padding channels

    scale_length = scene.unit_settings.scale_length
    return external_render.DepthBuffer(
        data=data * scale_length,
        depth_convention=DEPTH_CONVENTION,
        no_hit_value=DEPTH_NO_HIT_VALUE * scale_length,
        row_order=DEPTH_ROW_ORDER,
        units="meters",
    )


_RESULT_IMAGE_NAME = "vo_result_tmp"
_EXR_WRITER_SCENE_NAME = "__VolumeOverrideExrWriter"


def _exr_writer_scene():
    """Never-rendered Scene holding 32-bit EXR `image_settings` for
    `Image.save_render(..., scene=)`.
    """
    scene = bpy.data.scenes.get(_EXR_WRITER_SCENE_NAME)
    if scene is None:
        scene = bpy.data.scenes.new(_EXR_WRITER_SCENE_NAME)
    settings = scene.render.image_settings
    if settings.file_format != "OPEN_EXR":
        settings.file_format = "OPEN_EXR"
    if settings.color_depth != "32":
        settings.color_depth = "32"
    return scene


def write_render_result(result, path):
    """Write a RenderResult.rgba (H, W, 4) float32 array to a full-float EXR.

    `result.rgba` must use the DepthBuffer's row order (BOTTOM_LEFT_ORIGIN).

    Reuses one Image datablock (recreated only on resolution change), for
    the same persistent-data cache reason as `read_depth_buffer`. Uses
    `save_render()`, not `save()`: `save()` makes Cycles' next render
    re-sync all objects and rebuild the light tree, and flips the image to
    a FILE source that breaks reuse.
    """
    h, w = result.rgba.shape[:2]
    img = bpy.data.images.get(_RESULT_IMAGE_NAME)
    if img is not None and (img.size[0], img.size[1]) != (w, h):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(_RESULT_IMAGE_NAME, width=w, height=h, alpha=True, float_buffer=True)
    compositor_build.set_linear_colorspace(img)
    img.pixels.foreach_set(result.rgba.astype("float32").ravel())
    img.save_render(path, scene=_exr_writer_scene())


# --- Per-frame pipeline ------------------------------------------------------

def _sorted_far_to_near(scene, flagged):
    cam_loc = scene.camera.matrix_world.translation
    def distance(obj):
        return (obj.matrix_world.translation - cam_loc).length
    return sorted(flagged, key=distance, reverse=True)


_active_render_session = None  # set by render_session.__enter__/__exit__ below


class render_session:
    """Context manager for multi-frame renders: captures the scene settings
    `render_current_frame` forces (use_pass_z, use_persistent_data,
    compositor_device, view_transform) and restores them once at the end
    instead of after every frame. Restoring and re-forcing them each frame,
    even to the same values, invalidates Cycles' persistent-data cache.

    Without a session, `render_current_frame` restores after each call.
    Not reentrant.

    Usage::

        with operators.render_session(context):
            for frame in frames:
                scene.frame_set(frame)
                bpy.ops.render.volume_override(animation=False)
    """

    def __init__(self, context):
        self._context = context

    def __enter__(self):
        scene = self._context.scene
        view_layer = self._context.view_layer
        self._saved = {
            "use_pass_z": view_layer.use_pass_z,
            "use_persistent_data": scene.render.use_persistent_data,
            "compositor_device": scene.render.compositor_device,
            "view_transform": scene.view_settings.view_transform,
        }
        global _active_render_session
        _active_render_session = self
        # Sweep once per run; KeyError means the add-on isn't registered.
        try:
            _sweep_stale_frame_dirs(_work_dir(self._context))
        except KeyError:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active_render_session
        _active_render_session = None
        scene = self._context.scene
        view_layer = self._context.view_layer
        view_layer.use_pass_z = self._saved["use_pass_z"]
        scene.render.use_persistent_data = self._saved["use_persistent_data"]
        scene.render.compositor_device = self._saved["compositor_device"]
        scene.view_settings.view_transform = self._saved["view_transform"]
        return False  # never swallow exceptions


def _gpu_backend():
    """The GPU backend Blender is running on ("OPENGL", "VULKAN", ...), not the
    preference, which only applies after a restart.
    """
    import gpu

    try:
        return gpu.platform.backend_type_get()
    except SystemError:  # background mode before the GPU module initializes
        return bpy.context.preferences.system.gpu_backend


def _engine_warning(scene):
    """A warning for engine/camera combinations whose depth is only approximate, or None."""
    if scene.render.engine != "CYCLES" and scene.camera.data.type == "PANO":
        return (
            f"{scene.render.engine} with a panoramic camera: its Z pass differs from the "
            "panorama's exact depth by up to a few percent, so the volume's depth clip "
            "can be off near occluders. Use Cycles for panoramic renders."
        )
    return None


def render_current_frame(context):
    """Runs the full two-pass pipeline for `scene.frame_current`. Restores
    scene settings (unless a `render_session` is active) even if a step raises.
    """
    scene = context.scene
    if scene.camera is None:
        raise RuntimeError("Scene has no active camera")

    prefs = _get_prefs(context)
    flagged = properties.flagged_objects(scene)
    uses_yt_idv = any(obj.volume_override.renderer_mode in YT_IDV_RENDERER_MODES for obj in flagged)
    if uses_yt_idv and scene.render.engine != "CYCLES" and _gpu_backend() != "VULKAN":
        # Under the OpenGL backend EEVEE holds a GL context, and yt_idv's EGL
        # context then fails with EGL_BAD_ACCESS (or EEVEE dies with X BadAccess).
        raise RuntimeError(
            f"yt_idv renderer modes with {scene.render.engine} need Blender's Vulkan GPU "
            f"backend (Preferences > System > GPU Backend, then restart); otherwise "
            f"switch Render Properties > Render Engine to Cycles"
        )
    warning = _engine_warning(scene)
    if warning:
        _log(f"WARNING: {warning}")
    verbose = prefs.verbose_logging
    log = _log if verbose else (lambda *a, **k: None)
    step = _StepTimer if verbose else _NullStepTimer

    log(f"render_current_frame: frame={scene.frame_current} engine={scene.render.engine} "
        f"resolution={scene.render.resolution_x}x{scene.render.resolution_y}")

    work_dir = _work_dir(context)
    if _active_render_session is None:
        _sweep_stale_frame_dirs(work_dir)
    frame_dir = tempfile.mkdtemp(prefix=_FRAME_DIR_PREFIX, dir=work_dir)
    log(f"flagged objects: {[o.name for o in flagged]}, frame_dir={frame_dir}")
    scene_sync.sync_scene_state(scene, flagged)

    # Settings are only assigned when they differ: even same-value
    # reassignment invalidates Cycles' persistent-data cache.
    view_layer = context.view_layer
    prev_use_pass_z = view_layer.use_pass_z
    if not prev_use_pass_z:
        view_layer.use_pass_z = True

    # Keeps Cycles' BVH/geometry cache between the two passes and across frames.
    prev_use_persistent_data = scene.render.use_persistent_data
    if not prev_use_persistent_data:
        scene.render.use_persistent_data = True

    prev_compositor_device = scene.render.compositor_device
    prev_view_transform = scene.view_settings.view_transform
    if uses_yt_idv:
        if prev_view_transform != "Standard":
            # The default AgX transform flattens already-colormapped output.
            _log(f"WARNING: scene view_transform was {prev_view_transform!r} -- forcing "
                 f"'Standard' for this yt_idv render (restored to {prev_view_transform!r} "
                 f"afterward). {prev_view_transform!r} applies a filmic highlight rolloff "
                 f"that crushes an already-colormapped scientific visualization into a flat, "
                 f"low-contrast image -- see operators.render_current_frame's own comment for "
                 f"the direct A/B measurement. Set your scene's View Transform to 'Standard' "
                 f"(or 'Raw') yourself to silence this.")
            scene.view_settings.view_transform = "Standard"
        # The GPU compositor grabs an EGL context, after which yt_idv's own
        # EGL context fails with EGL_BAD_ACCESS; use the CPU compositor.
        if prev_compositor_device != "CPU":
            scene.render.compositor_device = "CPU"
    # All of this frame's EXR intermediates live in frame_dir, removed in
    # `finally` (they otherwise fill the disk on long runs).
    try:
        # --- Pass 1: beauty + Z from the scene's engine ----------------------
        with step("pass1 build compositor graph"):
            beauty_path, depth_path = compositor_build.build_pass1_capture_graph(
                scene, frame_dir, scene.frame_current,
            )
        log(f"pass1 will write beauty={beauty_path!r} depth={depth_path!r}")
        with step(f"pass1 render (bpy.ops.render.render), engine={scene.render.engine}"):
            bpy.ops.render.render(write_still=False)
        log(f"pass1 wrote beauty exists={os.path.exists(beauty_path)} depth exists={os.path.exists(depth_path)}")

        with step("read_depth_buffer"):
            depth_buffer = read_depth_buffer(scene, depth_path)
        camera_params = build_camera_params(scene, scene.camera)
        frame_params = build_frame_params(scene)

        volume_layers = []
        for obj in _sorted_far_to_near(scene, flagged):
            props = obj.volume_override
            render_fn = RENDERER_MODES[props.renderer_mode]
            log(f"{obj.name!r}: renderer_mode={props.renderer_mode!r}")
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
            if result.cmap_range is not None and not props.rescale_colormap and not props.colormap_range_set:
                props.colormap_range = result.cmap_range
                props.colormap_range_set = True
            result_path = os.path.join(
                frame_dir, f"vo_result_{obj.name}_{scene.frame_current:04d}.exr",
            )
            with step(f"write_render_result for {obj.name!r} -> {result_path!r}"):
                write_render_result(result, result_path)
            volume_layers.append((obj.name, result_path))

        # --- Pass 2: compositor-only, writes the real output ---------------
        # Same engine as pass 1: rendering with any other engine in between
        # invalidates Cycles' persistent-data cache. Nothing is path-traced,
        # since the graph has no Render Layers node.
        with step("pass2 build/update compositor graph"):
            compositor_build.build_or_update_graph(scene, beauty_path, volume_layers)
        with step(f"pass2 render (bpy.ops.render.render, write_still=True), engine={scene.render.engine}"):
            bpy.ops.render.render(write_still=True)
        log(f"pass2 done; final output at {scene.render.frame_path()!r} "
            f"(Render Result image also holds it -- see _show_render_result)")
    finally:
        # An active render_session restores these once at the end instead.
        if _active_render_session is None:
            view_layer.use_pass_z = prev_use_pass_z
            scene.render.compositor_device = prev_compositor_device
            scene.view_settings.view_transform = prev_view_transform
            scene.render.use_persistent_data = prev_use_persistent_data
        shutil.rmtree(frame_dir, ignore_errors=True)


def _show_render_result(context):
    """Pop Blender's Render Result viewer, same as a normal F12 press.

    Needed because the pipeline's synchronous (EXEC_DEFAULT) renders never
    open it. Skipped in background mode, where the poll failure logs an
    ERROR that the try/except below can't catch.
    """
    if bpy.app.background:
        _log("background mode -- not attempting to show Render Result (output was still written to disk/Render Result)")
        return
    try:
        bpy.ops.render.view_show("INVOKE_DEFAULT")
    except RuntimeError as exc:
        # Not fatal: the render already succeeded.
        _log(f"could not show Render Result ({exc}) -- output was still written to disk/Render Result")


class VOLUMEOVERRIDE_OT_render(bpy.types.Operator):
    """Render with the scene's engine plus yt_idv volumes, depth-clipped and composited."""

    bl_idname = "render.volume_override"
    bl_label = "Render (with Volume Override)"
    bl_options = {"REGISTER"}

    animation: bpy.props.BoolProperty(name="Animation", default=False)
    _timer = None
    _frame_iter = None
    _session = None

    def execute(self, context):
        if not properties.flagged_objects(context.scene):
            # F12 is always remapped here; fall back to a normal render.
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
        # Exited in _finish, however the loop ends.
        self._session = render_session(context)
        self._session.__enter__()
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
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None

    def cancel(self, context):
        self._finish(context)


def draw_render_menu(self, context):
    self.layout.operator(VOLUMEOVERRIDE_OT_render.bl_idname, text="Render (Volume Override)")
    self.layout.operator(
        VOLUMEOVERRIDE_OT_render.bl_idname, text="Render Animation (Volume Override)",
    ).animation = True


class VOLUMEOVERRIDE_OT_reset_change_tracking(bpy.types.Operator):
    """Make every flagged object's next render report params/dataset as
    changed, e.g. after the renderer's own cache was cleared. Content
    hashes are unaffected.
    """

    bl_idname = "volume_override.reset_change_tracking"
    bl_label = "Reset Change Tracking"

    def execute(self, context):
        change_tracking.reset()
        self.report({"INFO"}, "Change tracking reset -- next render reports everything as changed")
        return {"FINISHED"}


CLASSES = (
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
