"""Flagged volumes drawn over the 3D viewport, rendered with yt_idv from each
viewport's own view.

A POST_VIEW handler notices when a viewport's view or the scene changed and
reads its depth buffer; a timer re-renders the volumes once things settle,
clipped against that depth as F12 clips against the Z pass; a POST_PIXEL
handler draws the latest image. Renders share F12's render context and loaded
data (see `yt_idv_renderer._use_render_size`).
"""

import time

import bpy
import numpy as np

from . import external_render, operators, properties

SETTLE = 0.15  # seconds without changes before a viewport re-renders

_regions = {}  # region pointer -> {"signature", "view", "texture", "shown", "error"}
_last_change = [0.0]
_generation = [0]  # bumped when a volume, its nodes or the scene changes
_draw_handles = []


def camera_params(view_matrix, window_matrix, width, height, clip_start, clip_end):
    """`CameraParams` for a perspective viewport: `view_matrix` world-to-view,
    `window_matrix` its projection, at a `width` x `height` image.
    """
    window = np.asarray(window_matrix, dtype=np.float64)
    # With a sensor 2 wide fit horizontally, a lens of P[0][0] gives tan(hfov / 2) = 1 / P[0][0].
    return external_render.CameraParams(
        matrix_world=np.linalg.inv(np.asarray(view_matrix, dtype=np.float64)).tolist(),
        projection="PERSP",
        lens_mm=float(window[0, 0]),
        sensor_width_mm=2.0,
        sensor_height_mm=2.0,
        sensor_fit="HORIZONTAL",
        shift_x=0.0,
        shift_y=0.0,
        ortho_scale=None,
        clip_start=clip_start,
        clip_end=clip_end,
        resolution_x=width,
        resolution_y=height,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
        projection_matrix=window.tolist(),
        panorama=None,
        dof=None,
    )


def planar_depth(window_depth, window_matrix, scale_length=1.0):
    """The viewport's window-space depths (0..1, bottom row first) as F12's Z
    pass holds them: planar distance in scene units times `scale_length`, with
    the far plane (nothing drawn) as no hit.
    """
    depth = np.asarray(window_depth, dtype=np.float64)
    projection = np.asarray(window_matrix, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Inverts z_ndc = (P22 z + P23) / -z for the eye-space z, with z_ndc = 2 d - 1.
        distance = projection[2, 3] / ((2.0 * depth - 1.0) + projection[2, 2])
    no_hit = operators.DEPTH_NO_HIT_VALUE * scale_length
    return np.where(depth >= 1.0, no_hit, distance * scale_length).astype(np.float32)


def _resample(depth, width, height):
    """Nearest-pixel resample of a `(rows, columns)` depth array."""
    rows = ((np.arange(height) + 0.5) * depth.shape[0] / height).astype(int)
    columns = ((np.arange(width) + 0.5) * depth.shape[1] / width).astype(int)
    return depth[np.ix_(rows, columns)]


def render_view(scene, camera, depth=None):
    """`(height, width, 4)` premultiplied RGBA of the flagged yt_idv volumes
    seen by `camera`, bottom row first, composited far to near; `depth`
    (planar, as `planar_depth`, any size) clips them, else nothing does.
    """
    width, height = camera.resolution_x, camera.resolution_y
    scale_length = scene.unit_settings.scale_length
    if depth is None:
        data = np.full((height, width), operators.DEPTH_NO_HIT_VALUE * scale_length, dtype="f4")
    else:
        data = _resample(np.asarray(depth, dtype="f4"), width, height)
    depth = external_render.DepthBuffer(
        data=data,
        depth_convention=operators.DEPTH_CONVENTION,
        no_hit_value=operators.DEPTH_NO_HIT_VALUE * scale_length,
        row_order=operators.DEPTH_ROW_ORDER,
        units="meters",
    )
    frame = operators.build_frame_params(scene)
    position = np.asarray(camera.matrix_world)[:3, 3]
    volumes = [o for o in properties.flagged_objects(scene) if o.volume_override.renderer_mode in operators.YT_IDV_RENDERER_MODES]
    volumes.sort(key=lambda o: -np.linalg.norm(np.asarray(o.matrix_world.translation) - position))
    image = np.zeros((height, width, 4), dtype=np.float32)
    for obj in volumes:
        request = external_render.RenderRequest(
            camera=camera, depth=depth, volume=operators.build_volume_params(obj), frame=frame,
        )
        layer = operators.RENDERER_MODES[obj.volume_override.renderer_mode](request).rgba
        image = layer + image * (1.0 - layer[..., 3:4])
    return image


def _signature(region, rv3d, scale):
    return (
        tuple(map(tuple, rv3d.view_matrix)), tuple(map(tuple, rv3d.window_matrix)),
        region.width, region.height, scale, _generation[0],
    )


def _capture():
    """POST_VIEW: on a change, record the view and read the scene's depth."""
    context = bpy.context
    scene, region, rv3d = context.scene, context.region, context.region_data
    if not scene.yt_volumes_preview or rv3d is None:
        return
    state = _regions.setdefault(region.as_pointer(), {"signature": None, "texture": None, "shown": None, "error": None})
    signature = _signature(region, rv3d, scene.yt_volumes_preview_scale)
    if signature == state["signature"]:
        return
    state["signature"] = signature
    state["view"] = (
        [list(r) for r in rv3d.view_matrix], [list(r) for r in rv3d.window_matrix],
        region.width, region.height, rv3d.is_perspective, context.space_data.clip_start, context.space_data.clip_end,
    )
    try:
        import gpu

        buffer = gpu.state.active_framebuffer_get().read_depth(0, 0, region.width, region.height)
        try:
            window_depth = np.array(buffer, dtype=np.float32)
        except (TypeError, ValueError):  # a Buffer without the buffer protocol: slower
            window_depth = np.array(buffer.to_list(), dtype=np.float32)
        window_depth = window_depth.reshape(region.height, region.width)
        state["depth"] = planar_depth(window_depth, rv3d.window_matrix, scene.unit_settings.scale_length)
    except Exception as exc:  # then the volume is drawn unclipped
        state["depth"], state["depth_error"] = None, str(exc)[:200]
    _last_change[0] = time.monotonic()
    if not bpy.app.timers.is_registered(_render_pending):
        bpy.app.timers.register(_render_pending, first_interval=SETTLE)


def _draw():
    """POST_PIXEL: draw the viewport's latest image."""
    context = bpy.context
    if not context.scene.yt_volumes_preview or context.region_data is None:
        return
    state = _regions.get(context.region.as_pointer())
    if state is None:
        return
    if state["texture"] is not None:
        import gpu
        from gpu_extras.presets import draw_texture_2d

        gpu.state.blend_set("ALPHA_PREMULT")
        draw_texture_2d(state["texture"], (0, 0), context.region.width, context.region.height)
        gpu.state.blend_set("NONE")
    message = state["error"] or (state.get("depth_error") and f"no occlusion: {state['depth_error']}")
    if message:
        import blf

        blf.size(0, 12)
        blf.position(0, 12, 12, 0)
        blf.draw(0, f"yt_idv preview: {message}")


def _render_pending():
    """Timer: re-render viewports whose view changed, once changes settle."""
    waited = time.monotonic() - _last_change[0]
    if waited < SETTLE:
        return SETTLE - waited
    scene = bpy.context.scene
    for state in _regions.values():
        if state["shown"] == state["signature"] or "view" not in state:
            continue
        state["shown"] = state["signature"]
        view, window, width, height, perspective, clip_start, clip_end = state["view"]
        if not perspective:
            state["texture"], state["error"] = None, "orthographic views aren't supported"
            continue
        scale = scene.yt_volumes_preview_scale
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        try:
            image = render_view(scene, camera_params(view, window, *size, clip_start, clip_end), state.get("depth"))
        except Exception as exc:  # shown in the viewport; F12 raises the same error with its details
            state["texture"], state["error"] = None, str(exc)[:200]
            continue
        import gpu

        buffer = gpu.types.Buffer("FLOAT", image.size, image.ravel())
        state["texture"], state["error"] = gpu.types.GPUTexture(size, format="RGBA16F", data=buffer), None
    _redraw_viewports()
    return None


def _redraw_viewports():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _on_depsgraph_update(scene, depsgraph):
    # Anything may change the image now: a volume's look, its nodes, or geometry in front of it.
    if scene.yt_volumes_preview and len(depsgraph.updates):
        _generation[0] += 1
        _redraw_viewports()


def _toggle(self, context):
    _regions.clear()
    _redraw_viewports()


class VOLUMEOVERRIDE_PT_viewport_preview(bpy.types.Panel):
    """Draw flagged volumes in the 3D viewport with yt_idv"""

    bl_label = "yt-volumes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "yt-volumes"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "yt_volumes_preview")
        row = layout.row()
        row.enabled = context.scene.yt_volumes_preview
        row.prop(context.scene, "yt_volumes_preview_scale")


def register():
    bpy.types.Scene.yt_volumes_preview = bpy.props.BoolProperty(
        name="Show Volumes",
        description="Draw flagged volumes over the viewport with yt_idv, from each viewport's view (the first draw loads the data)",
        default=False, update=_toggle,
    )
    bpy.types.Scene.yt_volumes_preview_scale = bpy.props.FloatProperty(
        name="Resolution", description="Preview image size as a fraction of the viewport's",
        default=0.5, min=0.1, max=1.0, subtype="FACTOR",
    )
    bpy.utils.register_class(VOLUMEOVERRIDE_PT_viewport_preview)
    _draw_handles[:] = [
        bpy.types.SpaceView3D.draw_handler_add(_capture, (), "WINDOW", "POST_VIEW"),
        bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL"),
    ]
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
    _draw_handles.clear()
    if bpy.app.timers.is_registered(_render_pending):
        bpy.app.timers.unregister(_render_pending)
    _regions.clear()
    bpy.utils.unregister_class(VOLUMEOVERRIDE_PT_viewport_preview)
    del bpy.types.Scene.yt_volumes_preview_scale
    del bpy.types.Scene.yt_volumes_preview
