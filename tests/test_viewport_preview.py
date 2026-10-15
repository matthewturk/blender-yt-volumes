import math

import bpy
import numpy as np
import pytest

from yt_volumes import change_tracking, operators, properties, viewport_preview
from yt_volumes import yt_idv_renderer as yir


@pytest.fixture
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    change_tracking.reset()
    yield bpy.context.scene
    properties.unregister()


def _camera(scene, lens=35.0):
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = lens
    camera.location = (0.6, -2.0, 0.4)
    camera.rotation_euler = (1.4, 0.0, 0.3)
    scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera


@pytest.mark.parametrize("width, height", [(320, 180), (180, 320)])
def test_a_viewports_matrices_give_the_cameras_view(scene, width, height):
    camera = _camera(scene)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    window = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)

    params = viewport_preview.camera_params(camera.matrix_world.inverted(), window, width, height, 0.1, 100.0)

    np.testing.assert_allclose(params.matrix_world, np.array(camera.matrix_world), atol=1e-5)
    expected = operators.build_camera_params(scene, camera)
    aspect = width / height
    assert yir._approx_vertical_fov_deg(params, aspect) == pytest.approx(yir._approx_vertical_fov_deg(expected, aspect), rel=1e-5)


def _flag(scene, name, location, mode="YT_IDV_MAX_INTENSITY"):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "CUBE"
    obj.location = location
    obj.volume_override.is_volume_override = True
    obj.volume_override.renderer_mode = mode
    scene.collection.objects.link(obj)
    return obj


def test_volumes_are_composited_far_to_near(scene, monkeypatch):
    camera = _camera(scene)
    _flag(scene, "Near", (0.6, -1.0, 0.4))
    _flag(scene, "Far", (0.6, 5.0, 0.4))
    _flag(scene, "Debug", (0.0, 0.0, 0.0), mode="SAMPLE_CONSTANT")  # not a yt_idv mode: skipped
    bpy.context.view_layer.update()
    colors = {"Near": (0.5, 0.0, 0.0, 0.5), "Far": (0.0, 0.0, 1.0, 1.0)}
    order = []

    def fake_render(request):
        name = next(o.name for o in scene.objects if o.volume_override.dataset_uuid == request.volume.dataset.dataset_id)
        order.append(name)
        rgba = np.broadcast_to(np.array(colors[name], dtype=np.float32), (4, 6, 4)).copy()
        return yir.RenderResult(rgba=rgba, alpha_mode="PREMULTIPLIED", color_space="LINEAR")

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_render)
    params = viewport_preview.camera_params(camera.matrix_world.inverted(), np.eye(4) * 2, 6, 4, 0.1, 100.0)

    image = viewport_preview.render_view(scene, params)

    assert order == ["Far", "Near"]
    np.testing.assert_allclose(image[0, 0], [0.5, 0.0, 0.5, 1.0])  # near over far, premultiplied
    assert math.isclose(float(image[..., 3].min()), 1.0)


def test_window_depth_converts_to_planar_distance(scene):
    camera = _camera(scene)
    camera.data.clip_start, camera.data.clip_end = 0.1, 100.0
    projection = np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=320, y=180))
    distances = np.array([[0.1, 0.5, 2.0], [7.5, 40.0, 99.0]])
    # what the GPU stores: z_ndc = (P22 z + P23) / -z at eye-space z = -distance, mapped to 0..1
    z = -distances
    window_depth = 0.5 * ((projection[2, 2] * z + projection[2, 3]) / -z + 1.0)
    window_depth[1, 2] = 1.0  # nothing drawn here: the far plane

    planar = viewport_preview.planar_depth(window_depth, projection, scale_length=2.0)

    np.testing.assert_allclose(planar[0], 2.0 * distances[0], rtol=1e-4)
    np.testing.assert_allclose(planar[1, :2], 2.0 * distances[1, :2], rtol=1e-3)
    assert planar[1, 2] == pytest.approx(operators.DEPTH_NO_HIT_VALUE * 2.0)
