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

import bpy

from . import compositor_build, external_render, properties, scene_sync

PASS2_ENGINE = "BLENDER_WORKBENCH"


class VolumeOverridePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug_fake_volume_renderer: bpy.props.BoolProperty(
        name="Use debug fake volume renderer",
        description=(
            "render_volume() is a stub (NotImplementedError) until the real "
            "external renderer is wired in. Enable this to exercise the "
            "rest of the pipeline (depth clipping, two-pass compositing) "
            "with a flat placeholder color instead of crashing"
        ),
        default=False,
    )
    work_dir: bpy.props.StringProperty(
        name="Working Directory",
        description="Where per-frame EXR intermediates are written. Empty = system temp dir",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        self.layout.prop(self, "debug_fake_volume_renderer")
        self.layout.prop(self, "work_dir")


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
    return external_render.VolumeObjectParams(
        matrix_world=_matrix_to_list(obj.matrix_world),
        bounding_box_local=[list(corner) for corner in obj.bound_box],
        custom_params=obj.volume_override.to_custom_params(),
    )


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
    img.colorspace_settings.name = "Linear"
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

    work_dir = _work_dir(context)
    flagged = properties.flagged_objects(scene)
    scene_sync.sync_scene_state(scene, flagged)

    view_layer = context.view_layer
    prev_use_pass_z = view_layer.use_pass_z
    view_layer.use_pass_z = True

    prev_engine = scene.render.engine
    try:
        # --- Pass 1: real Cycles beauty + Z, flagged objects hidden -----
        beauty_path, depth_path = compositor_build.build_pass1_capture_graph(
            scene, work_dir, scene.frame_current,
        )
        with scene_sync.hidden_from_main_render(flagged):
            bpy.ops.render.render(write_still=False)

        depth_buffer = read_depth_buffer(scene, depth_path)
        camera_params = build_camera_params(scene, scene.camera)
        frame_params = build_frame_params(scene)

        prefs = _get_prefs(context)
        render_fn = (
            external_render.debug_fake_render_volume
            if prefs.debug_fake_volume_renderer
            else external_render.render_volume
        )

        volume_layers = []
        for obj in _sorted_far_to_near(scene, flagged):
            request = external_render.RenderRequest(
                camera=camera_params,
                depth=depth_buffer,
                volume=build_volume_params(obj),
                frame=frame_params,
            )
            result = render_fn(request)
            assert result.alpha_mode == "PREMULTIPLIED"
            assert result.color_space == "LINEAR"
            result_path = os.path.join(
                work_dir, f"vo_result_{obj.name}_{scene.frame_current:04d}.exr",
            )
            write_render_result(result, result_path)
            volume_layers.append((obj.name, result_path))

        # --- Pass 2: compositor-only (cheap engine), writes the real output
        compositor_build.build_or_update_graph(scene, beauty_path, volume_layers)
        scene.render.engine = PASS2_ENGINE
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.engine = prev_engine
        view_layer.use_pass_z = prev_use_pass_z


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


CLASSES = (
    VolumeOverridePreferences,
    VOLUMEOVERRIDE_OT_render,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_render.append(draw_render_menu)


def unregister():
    bpy.types.TOPBAR_MT_render.remove(draw_render_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
