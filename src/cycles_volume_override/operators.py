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
import tempfile
import time

import bpy

from . import change_tracking, compositor_build, external_render, properties, sample_renderers, scene_sync

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

# Keyed by VolumeOverridePreferences.renderer_mode -- see that enum's items
# for what each option is for.
RENDERER_MODES = {
    "STUB": external_render.render_volume,
    "DEBUG_FLAT": external_render.debug_fake_render_volume,
    "SAMPLE_CONSTANT": sample_renderers.constant_density_renderer,
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

    def draw(self, context):
        self.layout.prop(self, "renderer_mode")
        self.layout.prop(self, "work_dir")
        self.layout.prop(self, "verbose_logging")
        self.layout.operator("volume_override.reset_change_tracking")


def _get_prefs(context):
    return context.preferences.addons[__package__].preferences


def _work_dir(context):
    prefs = _get_prefs(context)
    d = prefs.work_dir or os.path.join(tempfile.gettempdir(), "cycles_volume_override")
    os.makedirs(d, exist_ok=True)
    return d


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


def build_volume_params(obj):
    custom_params = obj.volume_override.to_custom_params()
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


def render_current_frame(context):
    """Runs the full two-pass pipeline for `scene.frame_current`. Restores
    object visibility even if any step raises.
    """
    scene = context.scene
    if scene.camera is None:
        raise RuntimeError("Scene has no active camera")

    verbose = _get_prefs(context).verbose_logging
    log = _log if verbose else (lambda *a, **k: None)
    step = _StepTimer if verbose else _NullStepTimer

    log(f"render_current_frame: frame={scene.frame_current} engine={scene.render.engine} "
        f"resolution={scene.render.resolution_x}x{scene.render.resolution_y}")

    work_dir = _work_dir(context)
    flagged = properties.flagged_objects(scene)
    log(f"flagged objects: {[o.name for o in flagged]}, work_dir={work_dir}")
    scene_sync.sync_scene_state(scene, flagged)

    view_layer = context.view_layer
    prev_use_pass_z = view_layer.use_pass_z
    view_layer.use_pass_z = True

    prev_engine = scene.render.engine
    try:
        # --- Pass 1: real Cycles beauty + Z, flagged objects hidden -----
        with step("pass1 build compositor graph"):
            beauty_path, depth_path = compositor_build.build_pass1_capture_graph(
                scene, work_dir, scene.frame_current,
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

        prefs = _get_prefs(context)
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
            result_path = os.path.join(
                work_dir, f"vo_result_{obj.name}_{scene.frame_current:04d}.exr",
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
    """
    try:
        bpy.ops.render.view_show("INVOKE_DEFAULT")
    except RuntimeError as exc:
        # Not fatal -- the render itself already succeeded and wrote real
        # output files; failing to pop the viewer (e.g. no valid window
        # context, such as a background/headless call) shouldn't be reported
        # as a render failure.
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
