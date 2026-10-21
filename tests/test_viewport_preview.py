import dataclasses
import math

import bpy
import numpy as np
import pytest

from yt_volumes import change_tracking, external_render, operators, properties, viewport_preview
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
@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.1, -0.05)])
def test_a_viewports_matrices_give_the_cameras_view(scene, width, height, shift):
    camera = _camera(scene)
    camera.data.shift_x, camera.data.shift_y = shift
    scene.render.resolution_x, scene.render.resolution_y = width, height
    window = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)

    params = viewport_preview.camera_params(camera.matrix_world.inverted(), window, width, height, 0.1, 100.0)

    np.testing.assert_allclose(params.matrix_world, np.array(camera.matrix_world), atol=1e-5)
    expected = operators.build_camera_params(scene, camera)
    aspect = width / height
    assert yir._approx_vertical_fov_deg(params, aspect) == pytest.approx(yir._approx_vertical_fov_deg(expected, aspect), rel=1e-5)
    assert yir._shift_offset(params, height, width) == yir._shift_offset(expected, height, width)


def _flag(scene, name, location, mode="YT_IDV_MAX_INTENSITY"):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "CUBE"
    obj.location = location
    obj.volume_override.is_volume_override = True
    obj.volume_override.renderer_mode = mode
    obj.volume_override.show_in_viewport = True
    scene.collection.objects.link(obj)
    return obj


def test_volumes_are_composited_far_to_near(scene, monkeypatch):
    camera = _camera(scene)
    _flag(scene, "Near", (0.6, -1.0, 0.4))
    _flag(scene, "Far", (0.6, 5.0, 0.4))
    _flag(scene, "Debug", (0.0, 0.0, 0.0), mode="SAMPLE_CONSTANT")  # not a yt_idv mode: skipped
    _flag(scene, "Hidden", (0.6, 0.0, 0.4)).volume_override.show_in_viewport = False  # not shown: skipped
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

    viewport_preview.register()  # the viewport's scene settings
    try:
        image = viewport_preview.render_view(scene, params)
    finally:
        viewport_preview.unregister()

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

    scene.unit_settings.scale_length = 2.0  # changes neither depth nor matrix_world
    planar = viewport_preview.planar_depth(window_depth, projection)

    np.testing.assert_allclose(planar[0], distances[0], rtol=1e-4)
    np.testing.assert_allclose(planar[1, :2], distances[1, :2], rtol=1e-3)
    assert planar[1, 2] == pytest.approx(operators.DEPTH_NO_HIT_VALUE)


def _loaded(obj, source):
    """Pretend the renderer has `obj`'s data loaded from `source`."""
    import types

    scene_ = types.SimpleNamespace(_yt_volumes_source=source)
    yir._volume_scene_cache[obj.volume_override.dataset_uuid] = (scene_, None, obj.volume_override.backend)


def test_pending_loads_lists_volumes_until_their_data_is_loaded(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(yir, "_volume_scene_cache", {})
    data = tmp_path / "scene.zip"
    data.write_bytes(b"x" * 2_500_000)
    volume = _flag(scene, "Volume", (0, 0, 0))
    volume.volume_override.dataset_path = str(data)
    _flag(scene, "Debug", (0, 0, 0), mode="SAMPLE_CONSTANT")

    pending = operators.pending_loads(properties.flagged_objects(scene))
    assert [(obj.name, path) for obj, path in pending] == [("Volume", str(data))]
    assert "scene.zip (2 MB) for 'Volume'" in operators.load_message(pending)

    _loaded(volume, str(data))
    assert operators.pending_loads([volume]) == []


def test_the_sample_dataset_is_named_in_the_message(scene, monkeypatch):
    monkeypatch.setattr(yir, "_volume_scene_cache", {})
    volume = _flag(scene, "Volume", (0, 0, 0))
    assert f"the {yir.SAMPLE_DATASET_NAME} sample for 'Volume'" in operators.load_message(operators.pending_loads([volume]))


def test_only_confirmed_loads_happen_in_the_viewport(scene, monkeypatch):
    monkeypatch.setattr(yir, "_volume_scene_cache", {})
    monkeypatch.setattr(viewport_preview, "_confirmed", set())
    monkeypatch.setattr(viewport_preview, "restart", lambda: None)
    saved_on = _flag(scene, "Saved On", (0, 0, 0))  # e.g. a file saved with Show in Viewport on
    assert viewport_preview._hide_unconfirmed(scene)
    assert not saved_on.volume_override.show_in_viewport

    bpy.utils.register_class(viewport_preview.VOLUMEOVERRIDE_OT_toggle_viewport)
    try:
        with bpy.context.temp_override(object=saved_on):
            bpy.ops.volume_override.toggle_viewport()  # as after Load and Show
    finally:
        bpy.utils.unregister_class(viewport_preview.VOLUMEOVERRIDE_OT_toggle_viewport)
    assert saved_on.volume_override.show_in_viewport
    assert not viewport_preview._hide_unconfirmed(scene)  # confirmed: it stays on


def test_changing_frames_rerenders_the_preview(scene, tmp_path):
    viewport_preview.register()
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / "saved.blend"))
        bpy.ops.wm.open_mainfile(filepath=str(tmp_path / "saved.blend"))  # drops handlers that aren't persistent
        scene = bpy.context.scene
        _flag(scene, "Shown", (0, 0, 0))
        before = viewport_preview._generation[0]
        scene.frame_set(scene.frame_current + 1)  # runs frame_change_post, not depsgraph_update_post
        assert viewport_preview._generation[0] > before
    finally:
        viewport_preview.unregister()


def test_the_preview_waits_out_a_volume_override_render(scene, monkeypatch):
    rendered = []
    monkeypatch.setattr(viewport_preview, "render_view", lambda *a, **k: rendered.append(a))
    monkeypatch.setattr(viewport_preview, "_last_change", [0.0])
    monkeypatch.setattr(operators, "_active_render_session", object())

    assert viewport_preview._render_pending() == pytest.approx(0.5)  # try again later
    assert rendered == []


@pytest.mark.parametrize("own, viewport, expected", [(16.0, 4.0, 4.0), (2.0, 4.0, 2.0)])
def test_the_viewport_samples_at_most_its_own_samples_per_cell(scene, monkeypatch, own, viewport, expected):
    viewport_preview.register()
    try:
        camera = _camera(scene)
        volume = _flag(scene, "Shown", (0, 0, 0))
        volume.volume_override.samples_per_cell = own
        scene.yt_volumes_preview_samples = viewport
        seen = []

        def renderer(request):
            seen.append(request.volume.custom_params["yt_idv_samples_per_cell"])
            h, w = request.depth.data.shape
            return external_render.RenderResult(rgba=np.zeros((h, w, 4), "f4"), alpha_mode="PREMULTIPLIED", color_space="LINEAR")

        monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", renderer)
        monkeypatch.setattr(viewport_preview.units, "record", lambda *a: None)
        params = operators.build_camera_params(scene, camera)
        viewport_preview.render_view(scene, dataclasses.replace(params, resolution_x=8, resolution_y=6))

        assert seen == [expected]
        assert volume.volume_override.samples_per_cell == own  # F12's own setting is untouched
    finally:
        viewport_preview.unregister()
