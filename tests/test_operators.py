import os
from types import SimpleNamespace

import bpy
import numpy as np
import pytest
from mathutils import Vector

from yt_volumes import change_tracking, operators, properties


def _fake_obj(name, location):
    return SimpleNamespace(name=name, matrix_world=SimpleNamespace(translation=Vector(location)))


def test_sorted_far_to_near_orders_by_camera_distance():
    scene = SimpleNamespace(camera=_fake_obj("Cam", (0, 0, 0)))
    near = _fake_obj("Near", (0, 0, 2))
    far = _fake_obj("Far", (0, 0, 10))
    mid = _fake_obj("Mid", (0, 0, 5))

    ordered = operators._sorted_far_to_near(scene, [near, far, mid])
    assert [o.name for o in ordered] == ["Far", "Mid", "Near"]


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 64
    scene.render.resolution_y = 48
    yield scene


def _add_camera(scene, cam_type, **kwargs):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = cam_type
    for k, v in kwargs.items():
        setattr(cam_data, k, v)
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def test_build_camera_params_persp_has_projection_matrix_no_panorama(clean_scene):
    cam_obj = _add_camera(clean_scene, "PERSP", lens=50.0)
    params = operators.build_camera_params(clean_scene, cam_obj)
    assert params.projection == "PERSP"
    assert params.projection_matrix is not None
    assert len(params.projection_matrix) == 4
    assert params.panorama is None
    assert params.lens_mm == 50.0


def test_build_camera_params_ortho_has_projection_matrix(clean_scene):
    cam_obj = _add_camera(clean_scene, "ORTHO", ortho_scale=8.0)
    params = operators.build_camera_params(clean_scene, cam_obj)
    assert params.projection_matrix is not None
    assert params.lens_mm is None
    assert params.ortho_scale == 8.0


def test_build_camera_params_pano_has_no_projection_matrix_but_has_panorama(clean_scene):
    cam_obj = _add_camera(clean_scene, "PANO", panorama_type="EQUIRECTANGULAR")
    params = operators.build_camera_params(clean_scene, cam_obj)
    assert params.projection_matrix is None
    assert params.panorama is not None
    assert params.panorama.panorama_type == "EQUIRECTANGULAR"


def test_build_dof_params_plain_focus_distance(clean_scene):
    cam_obj = _add_camera(clean_scene, "PERSP")
    cam_obj.data.dof.use_dof = True
    cam_obj.data.dof.focus_distance = 12.5
    dof = operators.build_dof_params(cam_obj, cam_obj.data)
    assert dof.use_dof is True
    assert dof.focus_distance_m == pytest.approx(12.5)


def test_build_dof_params_resolves_focus_object_to_euclidean_distance(clean_scene):
    cam_obj = _add_camera(clean_scene, "PERSP")
    cam_obj.location = (0, 0, 0)
    cam_obj.data.dof.use_dof = True
    target = bpy.data.objects.new("Target", None)
    target.location = (3, 4, 0)  # distance 5 from origin
    clean_scene.collection.objects.link(target)
    cam_obj.data.dof.focus_object = target
    bpy.context.view_layer.update()

    dof = operators.build_dof_params(cam_obj, cam_obj.data)
    assert dof.focus_distance_m == pytest.approx(5.0)


# --- Dataset / change tracking ----------------------------------------------

@pytest.fixture
def clean_scene_with_props():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    change_tracking.reset()
    yield bpy.context.scene
    properties.unregister()


def test_build_dataset_params_generic_path_not_a_volume_object(clean_scene_with_props):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    obj.volume_override.dataset_kind = "yt"

    cache_key = properties.ensure_dataset_uuid(obj)
    dataset = operators.build_dataset_params(obj, cache_key)

    assert dataset.dataset_kind == "yt"
    assert dataset.dataset_path.endswith("sim_001")
    assert dataset.is_blender_volume_sequence is False
    assert dataset.dataset_id == cache_key
    assert dataset.version.changed is True  # first time seen


def test_build_dataset_params_repeat_call_same_value_reports_unchanged(clean_scene_with_props):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    cache_key = properties.ensure_dataset_uuid(obj)

    operators.build_dataset_params(obj, cache_key)
    second = operators.build_dataset_params(obj, cache_key)
    assert second.version.changed is False


def test_build_dataset_params_path_change_reports_changed(clean_scene_with_props):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    cache_key = properties.ensure_dataset_uuid(obj)

    operators.build_dataset_params(obj, cache_key)
    obj.volume_override.dataset_path = "/data/sim_002"
    second = operators.build_dataset_params(obj, cache_key)
    assert second.version.changed is True


def test_build_dataset_params_volume_object_overrides_path_from_blender_data(clean_scene_with_props):
    vol_data = bpy.data.volumes.new("V")
    vol_data.filepath = "/data/smoke.vdb"
    obj = bpy.data.objects.new("VolObj", vol_data)
    clean_scene_with_props.collection.objects.link(obj)
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "this should be overridden"

    cache_key = properties.ensure_dataset_uuid(obj)
    dataset = operators.build_dataset_params(obj, cache_key)

    assert dataset.dataset_path.endswith("smoke.vdb")
    assert dataset.dataset_kind == "vdb"


def test_build_volume_params_dataset_and_params_version_are_independent(clean_scene_with_props):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    obj.volume_override.mip_alpha_max = 0.5

    operators.build_volume_params(obj)  # first call, establishes history

    obj.volume_override.mip_alpha_max = 0.6  # cheap param changes, dataset doesn't
    second = operators.build_volume_params(obj)
    assert second.params_version.changed is True
    assert second.dataset.version.changed is False


def test_build_volume_params_bounding_box_comes_from_empty_display_size(clean_scene_with_props):
    # bound_box is all zeros for an Empty, so the extent comes from empty_display_size.
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.empty_display_size = 2.5
    obj.volume_override.is_volume_override = True

    params = operators.build_volume_params(obj)

    corners = params.bounding_box_local
    assert len(corners) == 8
    xs = {c[0] for c in corners}
    ys = {c[1] for c in corners}
    zs = {c[2] for c in corners}
    assert xs == ys == zs == {-2.5, 2.5}


def test_build_volume_params_rejects_a_mesh_object(clean_scene_with_props):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    with pytest.raises(RuntimeError, match="Empty-Cube"):
        operators.build_volume_params(obj)


def test_build_volume_params_rejects_a_non_cube_empty(clean_scene_with_props):
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    with pytest.raises(RuntimeError, match="Empty-Cube or Empty-Sphere"):
        operators.build_volume_params(obj)


def test_build_volume_params_accepts_a_sphere_empty_with_its_bounding_cube(clean_scene_with_props):
    bpy.ops.object.empty_add(type='SPHERE')
    obj = bpy.context.active_object
    obj.empty_display_size = 2.0
    obj.volume_override.is_volume_override = True

    corners = np.array(operators.build_volume_params(obj).bounding_box_local)
    np.testing.assert_allclose(corners.min(axis=0), -2.0)
    np.testing.assert_allclose(corners.max(axis=0), 2.0)


@pytest.fixture
def addon_enabled():
    # register_class alone doesn't populate preferences.addons[...]; a real enable does.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="yt_volumes")
    change_tracking.reset()
    yield bpy.context.scene
    bpy.ops.preferences.addon_disable(module="yt_volumes")


def _flagged_cube():
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    return obj


def test_build_volume_params_includes_the_objects_look_settings(clean_scene_with_props):
    obj = _flagged_cube()
    obj.volume_override.renderer_mode = "YT_IDV_MAX_INTENSITY_ALPHA"
    obj.volume_override.colormap = "inferno"
    obj.volume_override.mip_alpha_ramp_power = 0.42

    params = operators.build_volume_params(obj).custom_params

    assert params["renderer_mode"] == "YT_IDV_MAX_INTENSITY_ALPHA"
    assert params["yt_idv_colormap"] == "inferno"
    assert params["yt_idv_mip_alpha_ramp_power"] == pytest.approx(0.42)
    # Untouched fields default to yt_idv_renderer's module constants.
    from yt_volumes import yt_idv_renderer as yir
    assert params["yt_idv_mip_alpha_max"] == pytest.approx(yir.MIP_ALPHA_MAX)
    assert "yt_idv_cmap_range" not in params


def test_look_change_changes_the_params_hash(clean_scene_with_props):
    obj = _flagged_cube()
    first = operators.build_volume_params(obj).params_version.hash
    obj.volume_override.rescale_colormap = True
    assert operators.build_volume_params(obj).params_version.hash != first


def test_remembered_colormap_range_is_passed_and_cleared_by_a_mode_change(clean_scene_with_props):
    obj = _flagged_cube()
    props = obj.volume_override
    props.colormap_range = (0.25, 4.0)
    props.colormap_range_set = True
    assert operators.build_volume_params(obj).custom_params["yt_idv_cmap_range"] == pytest.approx((0.25, 4.0))

    props.renderer_mode = "YT_IDV_PROJECTION"

    assert props.colormap_range_set is False
    assert "yt_idv_cmap_range" not in operators.build_volume_params(obj).custom_params


def test_yt_idv_custom_params_omits_curve_and_image_keys_by_default(clean_scene_with_props):
    params = operators._yt_idv_custom_params(_flagged_cube())
    assert "yt_idv_alpha_curve" not in params
    assert "yt_idv_color_image_rgba" not in params


def test_selecting_curve_alpha_creates_a_per_object_curve(clean_scene_with_props):
    first, second = _flagged_cube(), _flagged_cube()
    first.volume_override.alpha_source = "CURVE"
    second.volume_override.alpha_source = "CURVE"

    assert first.volume_override.alpha_curve is not None
    assert first.volume_override.alpha_curve != second.volume_override.alpha_curve


def test_evaluate_alpha_curve_default_is_linear(clean_scene_with_props):
    obj = _flagged_cube()
    obj.volume_override.alpha_source = "CURVE"
    values = operators._evaluate_alpha_curve(obj.volume_override.alpha_curve, nbins=5)
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-3)


def test_yt_idv_custom_params_includes_alpha_curve_when_selected(clean_scene_with_props):
    obj = _flagged_cube()
    obj.volume_override.alpha_source = "CURVE"

    params = operators._yt_idv_custom_params(obj)

    assert len(params["yt_idv_alpha_curve"]) == 256
    assert params["yt_idv_alpha_curve"][0] == pytest.approx(0.0, abs=1e-3)
    assert params["yt_idv_alpha_curve"][-1] == pytest.approx(1.0, abs=1e-3)


def test_sample_image_as_1d_rgba_averages_rows_and_resamples_width(clean_scene_with_props):
    import numpy as np

    img = bpy.data.images.new("vo_test_gradient", width=4, height=2, alpha=True, float_buffer=True)
    data = np.zeros((2, 4, 4), dtype=np.float32)
    data[..., 0] = np.linspace(0.0, 1.0, 4)[None, :]  # identical red gradient in both rows
    data[..., 1] = 0.5
    data[..., 3] = 1.0
    img.pixels.foreach_set(data.ravel())

    try:
        rgba = operators._sample_image_as_1d_rgba(img, nbins=4)
        assert len(rgba) == 4
        assert rgba[0][0] == pytest.approx(0.0, abs=1e-4)
        assert rgba[-1][0] == pytest.approx(1.0, abs=1e-4)
        assert all(row[1] == pytest.approx(0.5, abs=1e-4) for row in rgba)
    finally:
        bpy.data.images.remove(img)


def test_sample_image_as_1d_rgba_raises_on_zero_sized_image():
    # images.new() clamps to 1x1, so fake the 0x0 a missing source file gives.
    class _FakeZeroSizedImage:
        name = "broken"
        size = (0, 0)

    with pytest.raises(ValueError):
        operators._sample_image_as_1d_rgba(_FakeZeroSizedImage())


def test_yt_idv_custom_params_includes_color_image_rgba_when_selected(clean_scene_with_props):
    import numpy as np

    img = bpy.data.images.new("vo_test_gradient2", width=4, height=1, alpha=True, float_buffer=True)
    data = np.zeros((1, 4, 4), dtype=np.float32)
    data[..., 2] = np.linspace(0.0, 1.0, 4)[None, :]
    data[..., 3] = 1.0
    img.pixels.foreach_set(data.ravel())
    obj = _flagged_cube()
    obj.volume_override.color_source = "IMAGE"
    obj.volume_override.color_image = img

    try:
        params = operators._yt_idv_custom_params(obj)
        assert len(params["yt_idv_color_image_rgba"]) == 256
        assert params["yt_idv_color_image_rgba"][0][2] == pytest.approx(0.0, abs=1e-3)
        assert params["yt_idv_color_image_rgba"][-1][2] == pytest.approx(1.0, abs=1e-3)
    finally:
        bpy.data.images.remove(img)


@pytest.mark.parametrize(
    "sequence_mode,scene_frame,expected",
    [
        ("CLIP", 5, 5),
        ("CLIP", 20, None),  # past frame_duration, no wrap -> no data
        ("CLIP", -3, None),  # before frame_start -> no data
        ("EXTEND", 20, 9),  # clamps to last frame (duration=10 -> indices 0..9)
        ("EXTEND", -3, 0),  # clamps to first frame
        ("REPEAT", 12, 2),  # wraps around
        ("PING_PONG", 12, 7),  # bounces back: 10,9,8,7 for local=10,11,12... see below
    ],
)
def test_resolve_sequence_frame(sequence_mode, scene_frame, expected):
    # frame_start=0, frame_duration=10, frame_offset=0
    result = operators.resolve_sequence_frame(0, 10, 0, sequence_mode, scene_frame)
    assert result == expected


def test_resolve_sequence_frame_no_duration_returns_none():
    assert operators.resolve_sequence_frame(0, 0, 0, "CLIP", 5) is None


def test_advancing_frames_does_not_flip_dataset_version_changed(clean_scene_with_props):
    """Advancing frames must not look like a dataset change (forcing a reload)."""
    obj = _flagged_cube()
    obj.volume_override.dataset_path = "/data/sim_001"
    cache_key = properties.ensure_dataset_uuid(obj)

    clean_scene_with_props.frame_set(1)
    first = operators.build_dataset_params(obj, cache_key)
    clean_scene_with_props.frame_set(2)
    second = operators.build_dataset_params(obj, cache_key)

    assert second.version.changed is False
    assert second.version.hash == first.version.hash


# --- F12 pass-through when nothing is flagged --------------------------------

@pytest.fixture
def registered_render_operator():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    operators.register()
    yield
    operators.unregister()
    properties.unregister()


def test_render_operator_delegates_to_plain_render_when_nothing_flagged(registered_render_operator, monkeypatch):
    """F12 is remapped unconditionally, so unflagged scenes must render normally."""
    scene = bpy.context.scene
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "BLENDER_WORKBENCH"
    bpy.ops.object.camera_add(location=(0, 0, 5))
    scene.camera = bpy.context.active_object
    assert properties.flagged_objects(scene) == []

    called = []
    monkeypatch.setattr(operators, "render_current_frame", lambda *a, **k: called.append(True))

    result = bpy.ops.render.volume_override(animation=False)

    assert result == {"FINISHED"}
    assert called == []  # the two-pass pipeline must not run


def test_render_operator_runs_pipeline_when_something_flagged(registered_render_operator, monkeypatch):
    scene = bpy.context.scene
    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    called = []
    monkeypatch.setattr(operators, "render_current_frame", lambda *a, **k: called.append(True))
    monkeypatch.setattr(operators, "_show_render_result", lambda *a, **k: None)

    result = bpy.ops.render.volume_override(animation=False)

    assert result == {"FINISHED"}
    assert called == [True]


# --- view_transform forced to Standard for yt_idv modes (AgX washout fix) ---
# AgX (the default view transform) washes out colormapped output. These run a
# real tiny Cycles pass 1 with RENDERER_MODES monkeypatched (no yt_idv needed).

def _fake_render_result(request):
    import numpy as np

    from yt_volumes import external_render

    h, w = request.depth.data.shape
    return external_render.RenderResult(
        rgba=np.zeros((h, w, 4), dtype="float32"),
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )


def test_render_current_frame_forces_standard_view_transform_for_yt_idv_mode(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "AgX"
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)

    seen_view_transform = []

    def fake_renderer(request):
        seen_view_transform.append(scene.view_settings.view_transform)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_view_transform == ["Standard"]  # forced during the render
    assert scene.view_settings.view_transform == "AgX"  # restored afterward, untouched


def test_render_current_frame_remembers_the_colormap_range_once(addon_enabled, monkeypatch, tmp_path):
    import dataclasses

    scene = addon_enabled
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)
    props = _flagged_cube().volume_override
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(tmp_path)
    ranges = iter([(1.0, 2.0), (5.0, 6.0)])

    def fake_renderer(request):
        return dataclasses.replace(_fake_render_result(request), cmap_range=next(ranges))

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)
    operators.render_current_frame(bpy.context)

    assert props.colormap_range_set
    assert tuple(props.colormap_range) == pytest.approx((1.0, 2.0))


def test_render_current_frame_leaves_standard_view_transform_alone(addon_enabled, monkeypatch, tmp_path, capsys):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "Standard"
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)
    prefs.verbose_logging = False

    seen_view_transform = []

    def fake_renderer(request):
        seen_view_transform.append(scene.view_settings.view_transform)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)
    capsys.readouterr()  # discard anything printed before this point

    operators.render_current_frame(bpy.context)

    assert seen_view_transform == ["Standard"]
    assert scene.view_settings.view_transform == "Standard"
    assert "view_transform" not in capsys.readouterr().out  # no warning needed -- nothing changed


def test_render_current_frame_view_transform_warning_ignores_verbose_logging(addon_enabled, monkeypatch, tmp_path, capsys):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "AgX"
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)
    prefs.verbose_logging = False

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)
    capsys.readouterr()

    operators.render_current_frame(bpy.context)

    out = capsys.readouterr().out
    assert "WARNING" in out and "AgX" in out and "Standard" in out


def test_render_current_frame_does_not_touch_view_transform_for_non_yt_idv_mode(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "AgX"
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True
    box.volume_override.renderer_mode = "SAMPLE_CONSTANT"

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)

    operators.render_current_frame(bpy.context)

    assert scene.view_settings.view_transform == "AgX"  # left alone


# --- pass2 engine / persistent_data perf fixes --------------------------------
# Switching pass 2 to Workbench invalidated Cycles' persistent-data cache every
# frame; pass 2 doesn't path-trace anyway, since it never uses Render Layers.

def test_render_current_frame_does_not_switch_engine_for_pass2(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)

    seen_engine_during_render = []

    def fake_renderer(request):
        seen_engine_during_render.append(scene.render.engine)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_engine_during_render == ["CYCLES"]
    assert scene.render.engine == "CYCLES"  # unchanged afterward too -- nothing to even restore


def test_render_current_frame_enables_and_restores_use_persistent_data(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.use_persistent_data = False
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)

    seen_persistent_data = []

    def fake_renderer(request):
        seen_persistent_data.append(scene.render.use_persistent_data)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_persistent_data == [True]  # enabled during the render
    assert scene.render.use_persistent_data is False  # restored to whatever it was before


def test_render_current_frame_never_touches_hide_render_on_the_flagged_object(addon_enabled, monkeypatch, tmp_path):
    # Regression (perf): toggling hide_render per frame invalidated Cycles'
    # persistent-data cache; Empties never render anyway.
    scene = addon_enabled
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True
    box.hide_render = False

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)

    seen_hide_render = []

    def fake_renderer(request):
        seen_hide_render.append(box.hide_render)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_hide_render == [False]  # never set to True at any point
    assert box.hide_render is False


def test_only_volumes_enabled_for_renders_are_rendered(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(tmp_path)

    shown = _flagged_cube()
    camera_off = _flagged_cube()
    camera_off.hide_render = True
    hidden = bpy.data.collections.new("NotRendered")
    scene.collection.children.link(hidden)
    hidden.hide_render = True
    in_hidden = _flagged_cube()
    for collection in in_hidden.users_collection:
        collection.objects.unlink(in_hidden)
    hidden.objects.link(in_hidden)
    excluded = bpy.data.collections.new("Excluded")
    scene.collection.children.link(excluded)
    bpy.context.view_layer.layer_collection.children["Excluded"].exclude = True
    in_excluded = _flagged_cube()
    for collection in in_excluded.users_collection:
        collection.objects.unlink(in_excluded)
    excluded.objects.link(in_excluded)
    eye_off = _flagged_cube()
    eye_off.hide_set(True)  # viewport only: still renders

    rendered = []

    def fake_renderer(request):
        rendered.append(next(o.name for o in scene.objects if getattr(o, "volume_override", None)
                             and o.volume_override.dataset_uuid == request.volume.dataset.dataset_id))
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)
    operators.render_current_frame(bpy.context)

    assert sorted(rendered) == sorted([shown.name, eye_off.name])
    assert camera_off.hide_render  # left as the user set it


# --- render_session (batch-scoped restore) --------------------------------
# Restoring/re-forcing scene settings around every frame invalidated Cycles'
# cache; render_session restores once at the end. Bare calls still restore.

def _setup_yt_idv_scene(scene, tmp_path, view_transform="AgX"):
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = view_transform
    _add_camera(scene, "PERSP", lens=50.0)
    _flagged_cube()

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = str(tmp_path)
    return prefs


def test_render_current_frame_without_a_session_restores_every_call(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    _setup_yt_idv_scene(scene, tmp_path)
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)

    operators.render_current_frame(bpy.context)
    assert scene.view_settings.view_transform == "AgX"  # restored after THIS call

    operators.render_current_frame(bpy.context)
    assert scene.view_settings.view_transform == "AgX"  # restored again -- no session active


def test_render_current_frame_inside_a_session_only_restores_at_session_exit(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    _setup_yt_idv_scene(scene, tmp_path)
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)

    with operators.render_session(bpy.context):
        operators.render_current_frame(bpy.context)
        assert scene.view_settings.view_transform == "Standard"  # NOT restored mid-session

        operators.render_current_frame(bpy.context)
        assert scene.view_settings.view_transform == "Standard"  # still not restored

    assert scene.view_settings.view_transform == "AgX"  # restored once, at session exit


def test_render_current_frame_sweeps_once_per_session_not_once_per_frame(addon_enabled, monkeypatch, tmp_path):
    # Regression (perf): sweeping every frame slowed the render.
    scene = addon_enabled
    _setup_yt_idv_scene(scene, tmp_path)
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)

    calls = []
    real_sweep = operators._sweep_stale_frame_dirs

    def counting_sweep(work_dir):
        calls.append(work_dir)
        return real_sweep(work_dir)

    monkeypatch.setattr(operators, "_sweep_stale_frame_dirs", counting_sweep)

    operators.render_current_frame(bpy.context)
    operators.render_current_frame(bpy.context)
    assert len(calls) == 2, "a bare call (no session) sweeps every single time"

    calls.clear()
    with operators.render_session(bpy.context):
        assert len(calls) == 1, "render_session.__enter__ itself sweeps once, up front"
        operators.render_current_frame(bpy.context)
        operators.render_current_frame(bpy.context)
        operators.render_current_frame(bpy.context)
    assert len(calls) == 1, "no per-frame call inside an active session -- only the one from __enter__"


def test_render_session_restores_original_values_even_if_never_forced(clean_scene):
    scene = clean_scene
    view_layer = bpy.context.view_layer
    scene.view_settings.view_transform = "Filmic"
    scene.render.compositor_device = "GPU"
    scene.render.use_persistent_data = False
    view_layer.use_pass_z = False

    with operators.render_session(bpy.context):
        scene.view_settings.view_transform = "Standard"
        scene.render.compositor_device = "CPU"
        scene.render.use_persistent_data = True
        view_layer.use_pass_z = True

    assert scene.view_settings.view_transform == "Filmic"
    assert scene.render.compositor_device == "GPU"
    assert scene.render.use_persistent_data is False
    assert view_layer.use_pass_z is False


# --- per-frame temp directory cleanup -----------------------------------------
# Regression: per-frame EXRs were never deleted, so long runs hit ENOSPC. Hard
# crashes (e.g. X11 BadAccess) skip `finally`, hence the startup sweep.

def _render_one_frame(scene, work_dir):
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)

    bpy.ops.object.empty_add(type='CUBE')
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    prefs = bpy.context.preferences.addons["yt_volumes"].preferences
    prefs.work_dir = work_dir
    return prefs


def _frame_dirs(work_dir):
    return [
        name for name in os.listdir(work_dir)
        if name.startswith(operators._FRAME_DIR_PREFIX) and os.path.isdir(os.path.join(work_dir, name))
    ]


def test_render_current_frame_cleans_up_its_own_frame_dir_after_success(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    _render_one_frame(scene, str(tmp_path))
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)

    operators.render_current_frame(bpy.context)

    assert _frame_dirs(str(tmp_path)) == [], (
        f"expected the frame directory to be removed, work_dir has {os.listdir(tmp_path)}"
    )


def test_render_current_frame_cleans_up_frame_dir_even_when_pipeline_raises(addon_enabled, monkeypatch, tmp_path):
    scene = addon_enabled
    _render_one_frame(scene, str(tmp_path))

    def failing_renderer(request):
        raise RuntimeError("simulated render_volume failure")

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", failing_renderer)

    with pytest.raises(RuntimeError, match="simulated render_volume failure"):
        operators.render_current_frame(bpy.context)

    assert _frame_dirs(str(tmp_path)) == [], (
        f"expected the frame directory to be removed even after an exception, "
        f"work_dir has {os.listdir(tmp_path)}"
    )


def test_render_current_frame_sweeps_a_frame_dir_left_by_a_previous_hard_crash(addon_enabled, monkeypatch, tmp_path):
    stale_dir = tmp_path / f"{operators._FRAME_DIR_PREFIX}stale123"
    stale_dir.mkdir()
    (stale_dir / "vo_beauty_0001.exr").write_bytes(b"leftover")
    non_managed_dir = tmp_path / "not_ours"
    non_managed_dir.mkdir()

    scene = addon_enabled
    _render_one_frame(scene, str(tmp_path))
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)

    operators.render_current_frame(bpy.context)

    assert not stale_dir.exists(), "the stale frame_* directory from a 'previous crash' should have been swept"
    assert non_managed_dir.exists(), "a directory that doesn't match our own frame_* prefix must be left alone"


def test_sweep_stale_frame_dirs_only_removes_frame_prefixed_directories(tmp_path):
    (tmp_path / f"{operators._FRAME_DIR_PREFIX}abc").mkdir()
    (tmp_path / f"{operators._FRAME_DIR_PREFIX}def").mkdir()
    (tmp_path / "unrelated_dir").mkdir()
    (tmp_path / f"{operators._FRAME_DIR_PREFIX}looks_like_a_dir_but_is_a_file").write_bytes(b"")

    operators._sweep_stale_frame_dirs(str(tmp_path))

    remaining = set(os.listdir(tmp_path))
    assert remaining == {"unrelated_dir", f"{operators._FRAME_DIR_PREFIX}looks_like_a_dir_but_is_a_file"}, (
        f"expected only frame_*-prefixed DIRECTORIES removed, got {remaining}"
    )


def _write_exr(path, value, shape=(2, 2)):
    import numpy as np

    from yt_volumes import compositor_build

    img = bpy.data.images.new("test_exr", width=shape[1], height=shape[0], alpha=True, float_buffer=True)
    compositor_build.set_linear_colorspace(img)  # the default tag bakes an sRGB encode into the file; set before the pixels
    img.pixels.foreach_set(np.full(shape[0] * shape[1] * 4, value, dtype="float32"))
    img.filepath_raw, img.file_format, img.use_half_precision = path, "OPEN_EXR", False
    img.save()
    bpy.data.images.remove(img)


def test_read_depth_buffer_reuses_the_same_image_datablock_across_calls(tmp_path):
    import numpy as np

    from yt_volumes import external_render

    path_a = str(tmp_path / "depth_a.exr")
    path_b = str(tmp_path / "depth_b.exr")
    _write_exr(path_a, 1.5)
    _write_exr(path_b, 3.0)

    scene = bpy.context.scene
    depth_a = operators.read_depth_buffer(scene, path_a)
    img_after_first = bpy.data.images.get(operators._DEPTH_IMAGE_NAME)
    assert img_after_first is not None
    pointer_after_first = img_after_first.as_pointer()

    depth_b = operators.read_depth_buffer(scene, path_b)
    img_after_second = bpy.data.images.get(operators._DEPTH_IMAGE_NAME)

    assert img_after_second is not None
    assert img_after_second.as_pointer() == pointer_after_first
    assert np.allclose(depth_a.data, 1.5, atol=1e-4)
    assert np.allclose(depth_b.data, 3.0, atol=1e-4)


def test_every_yt_idv_renderer_mode_gets_the_cpu_compositor():
    # A yt_idv mode missing from this set leaves the GPU compositor on, and
    # yt_idv's eglMakeCurrent then fails with EGL_BAD_ACCESS.
    yt_idv_modes = {m for m, fn in operators.RENDERER_MODES.items() if fn.__module__.endswith("yt_idv_renderer")}
    assert yt_idv_modes == operators.YT_IDV_RENDERER_MODES
    assert {"YT_IDV_MAX_INTENSITY_ALPHA", "YT_IDV_GN_TRANSFER_FUNCTION"} <= yt_idv_modes


def test_yt_idv_modes_refuse_a_non_cycles_engine(monkeypatch, clean_scene_with_props):
    scene = clean_scene_with_props
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    _flagged_cube()
    monkeypatch.setattr(operators, "_get_prefs", lambda context: SimpleNamespace(verbose_logging=False))
    monkeypatch.setattr(operators, "_gpu_backend", lambda: "OPENGL")
    with pytest.raises(RuntimeError, match="Vulkan GPU backend"):
        operators.render_current_frame(bpy.context)


@pytest.mark.parametrize("backend, during", [("OPENGL", "CPU"), ("VULKAN", "GPU")])
def test_only_opengl_gets_the_cpu_compositor(addon_enabled, monkeypatch, tmp_path, backend, during):
    # Under Vulkan the CPU compositor doesn't restore the window's GPU context
    # after a nested EEVEE render (the annotation scene), and Blender crashes on the next redraw.
    scene = addon_enabled
    _setup_yt_idv_scene(scene, tmp_path)
    scene.render.compositor_device = "GPU"
    monkeypatch.setattr(operators, "_gpu_backend", lambda: backend)
    seen = []

    def fake_renderer(request):
        seen.append(scene.render.compositor_device)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)
    operators.render_current_frame(bpy.context)
    assert seen == [during]
    assert scene.render.compositor_device == "GPU"


# --- EEVEE --------------------------------------------------------------------

def _depth_seen_by_renderer(scene, engine, monkeypatch, tmp_path):
    """Run the real two-pass pipeline with `engine` and return the depth the renderer got."""
    scene.render.engine = engine
    seen = []

    def fake_renderer(request):
        seen.append(request.depth)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)
    operators.render_current_frame(bpy.context)
    return seen[0]


def test_eevee_and_cycles_hand_the_renderer_the_same_depth(addon_enabled, monkeypatch, tmp_path):
    # The renderer assumes planar depth with 1e10 for no hit, whichever engine made it.
    scene = addon_enabled
    _setup_yt_idv_scene(scene, tmp_path)
    scene.camera.location = (0.0, 0.0, 10.0)
    bpy.ops.mesh.primitive_plane_add(size=6, location=(3.0, 0.0, 0.0))  # right half of the frame
    monkeypatch.setattr(operators, "_gpu_backend", lambda: "VULKAN")
    scene.unit_settings.scale_length = 0.01  # Z stays in Blender units, like the matrices

    cycles = _depth_seen_by_renderer(scene, "CYCLES", monkeypatch, tmp_path)
    eevee = _depth_seen_by_renderer(scene, "BLENDER_EEVEE", monkeypatch, tmp_path)

    assert eevee.depth_convention == cycles.depth_convention == "PLANAR"
    hit = cycles.data < 1e9
    assert hit.any() and not hit.all()
    np.testing.assert_array_equal(eevee.data < 1e9, hit)
    np.testing.assert_allclose(eevee.data[hit], cycles.data[hit], rtol=1e-4)
    np.testing.assert_allclose(cycles.data[hit], 10.0, rtol=1e-4)
    assert (eevee.data[~hit] == cycles.data[~hit]).all()


def test_gpu_backend_falls_back_to_the_preference_before_gpu_init(monkeypatch):
    import gpu

    def not_initialized():
        raise SystemError("GPU functions for drawing requires the gpu module to be initialized.")

    monkeypatch.setattr(gpu.platform, "backend_type_get", not_initialized)
    assert operators._gpu_backend() == bpy.context.preferences.system.gpu_backend


@pytest.mark.parametrize("engine, camera_type, warns", [
    ("BLENDER_EEVEE", "PANO", True),
    ("BLENDER_EEVEE", "PERSP", False),
    ("CYCLES", "PANO", False),
])
def test_engine_warning_flags_eevee_panoramas(clean_scene, engine, camera_type, warns):
    scene = clean_scene
    scene.render.engine = engine
    _add_camera(scene, camera_type)
    assert (operators._engine_warning(scene) is not None) == warns


def test_loading_notice_is_a_no_op_in_background_mode():
    entered = []
    with operators._loading_notice(bpy.context, "Loading x"):
        entered.append(True)
    assert entered == [True]


def test_gn_transfer_function_mode_samples_the_starter_nodes(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    obj.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    transfer_function_nodes.add_transfer_function_modifier(obj)

    rgba = np.asarray(operators._yt_idv_custom_params(obj)["yt_idv_gn_tf_rgba"])

    assert rgba.shape == (transfer_function_nodes.NBINS, 4)
    # The ramp's sRGB stops come back sRGB-encoded; opacity is the curve's 0 -> 0.8.
    assert rgba[0] == pytest.approx([0.001, 0.0, 0.014, 0.0], abs=2e-3)
    assert rgba[-1] == pytest.approx([0.988, 0.998, 0.645, 0.8], abs=2e-3)


def test_gn_transfer_function_follows_an_edited_ramp_and_curve(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    obj.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    transfer_function_nodes.add_transfer_function_modifier(obj)
    ramp, curve = transfer_function_nodes.find_nodes(obj)
    for element in ramp.color_ramp.elements:
        element.color = (1.0, 0.0, 0.0, 1.0)
    curve.mapping.curves[0].points[-1].location = (1.0, 0.25)
    curve.mapping.update()
    obj.update_tag()

    rgba = np.asarray(operators._yt_idv_custom_params(obj)["yt_idv_gn_tf_rgba"])

    assert rgba[100] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=2e-3)
    assert rgba[-1] == pytest.approx([1.0, 0.0, 0.0, 0.25], abs=2e-3)


def test_gn_transfer_function_mode_without_nodes_explains_how_to_add_them(clean_scene_with_props):
    obj = _flagged_cube()
    obj.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"

    with pytest.raises(ValueError, match="add a transfer function"):
        operators._yt_idv_custom_params(obj)


def _outline(obj):
    bpy.context.view_layer.update()
    geometry = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
    mesh = geometry.mesh
    xs = [v.co.x for v in mesh.vertices]
    return len(mesh.edges), len(mesh.polygons), min(xs), max(xs)


def test_transfer_function_modifier_outlines_the_empty_with_edges_that_follow_it(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    obj.empty_display_size = 1.5
    transfer_function_nodes.add_transfer_function_modifier(obj)
    assert _outline(obj) == pytest.approx((12, 0, -1.5, 1.5))

    obj.empty_display_size = 0.5
    assert _outline(obj) == pytest.approx((12, 0, -0.5, 0.5))

    obj.empty_display_type = "SPHERE"
    edges, faces, low, high = _outline(obj)
    assert (edges, faces) == (96, 0)  # three 32-segment circles
    assert (low, high) == pytest.approx((-0.5, 0.5))


def test_distribution_lands_on_the_transfer_function_points_and_is_packed(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    transfer_function_nodes.add_transfer_function_modifier(obj)
    n = transfer_function_nodes.NBINS
    points = {
        "volume_fraction": np.full(n, 1.0 / n),
        "mass_fraction": np.linspace(0.0, 2.0 / n, n),
        "log10_value": np.linspace(-27.3, -9.8, n),  # PNG packing would clamp these to [0, 1]
    }
    bpy.context.view_layer.update()  # evaluated once already, as after a render
    transfer_function_nodes.update_distribution(obj, points)
    bpy.context.view_layer.update()

    geometry = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
    instances = geometry.instances_pointcloud()
    for name, key in (("yt_tf_volume_fraction", "volume_fraction"), ("yt_tf_mass_fraction", "mass_fraction"),
                      ("yt_tf_log10_value", "log10_value")):
        values = np.empty(n, dtype=np.float32)
        instances.attributes[name].data.foreach_get("value", values)
        np.testing.assert_allclose(values, points[key], rtol=1e-6, atol=1e-9)
    image = obj.modifiers[0].node_group.nodes[transfer_function_nodes.DISTRIBUTION_NODE_NAME].inputs["Image"].default_value
    assert image.packed_file is not None and image.file_format == "OPEN_EXR"


def _stereo(scene, mode="OFFAXIS", pivot="CENTER"):
    scene.render.use_multiview = True
    scene.render.views_format = "STEREO_3D"
    scene.render.image_settings.views_format = "INDIVIDUAL"
    stereo = scene.camera.data.stereo
    stereo.convergence_mode, stereo.pivot = mode, pivot
    stereo.interocular_distance, stereo.convergence_distance = 0.4, 5.0


def test_an_eyes_projection_is_blenders_with_the_eyes_shift(clean_scene):
    scene = clean_scene
    scene.render.resolution_x, scene.render.resolution_y = 320, 200
    camera = _add_camera(scene, "PERSP", lens=40.0)
    camera.data.shift_x = 0.03
    _stereo(scene)

    params = operators.build_camera_params(scene, camera, "left")
    _, shift_x = operators.stereo_eye(camera, "left")
    camera.data.shift_x = shift_x
    bpy.context.view_layer.update()
    expected = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=320, y=200)

    assert params.shift_x == pytest.approx(shift_x)
    np.testing.assert_allclose(params.projection_matrix, np.array(expected), atol=1e-6)


@pytest.mark.parametrize("mode", ["OFFAXIS", "PARALLEL", "TOE"])
def test_eye_cameras_see_what_cycles_draws_in_each_eye(clean_scene, tmp_path, mode):
    from yt_volumes import compositor_build

    scene = clean_scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.resolution_x, scene.render.resolution_y = 160, 100
    camera = _add_camera(scene, "PERSP", lens=50.0)
    camera.location, camera.rotation_euler = (0.3, -6.0, 0.5), (1.52, 0.02, 0.1)
    _stereo(scene, mode)
    center = np.array([0.8, 0.5, -0.4])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=center)
    scene.view_layers[0].use_pass_z = True
    _, depth_path = compositor_build.build_pass1_capture_graph(scene, str(tmp_path), scene.frame_current)
    bpy.ops.render.render()

    for view, suffix in operators.render_views(scene):
        depth = operators.read_depth_buffer(scene, operators._with_suffix(depth_path, suffix)).data  # bottom row first
        rows, cols = np.nonzero(depth < 1e9)
        params = operators.build_camera_params(scene, camera, view)
        clip = np.array(params.projection_matrix) @ np.linalg.inv(np.array(params.matrix_world)) @ np.append(center, 1.0)
        x, y = (clip[:2] / clip[3] + 1.0) / 2.0 * (160, 100)
        assert (cols + 0.5).mean() == pytest.approx(x, abs=0.5), view
        assert (rows + 0.5).mean() == pytest.approx(y, abs=0.5), view


def test_a_stereo_render_renders_each_eye_and_composites_it(addon_enabled, monkeypatch, tmp_path):
    import dataclasses

    scene = addon_enabled
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "Standard"
    _add_camera(scene, "PERSP", lens=50.0)
    _stereo(scene)
    _flagged_cube()
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(tmp_path / "work")
    scene.render.filepath = str(tmp_path / "out_")
    seen = []

    def fake_renderer(request):
        seen.append((request.frame.view_index, request.camera.matrix_world[0][3]))
        rgba = np.zeros(request.depth.data.shape + (4,), dtype="float32")
        rgba[..., request.frame.view_index * 2] = rgba[..., 3] = 1.0  # left red, right blue
        return dataclasses.replace(_fake_render_result(request), rgba=rgba)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert [index for index, _ in seen] == [0, 1]
    assert seen[0][1] == pytest.approx(-0.2) and seen[1][1] == pytest.approx(0.2)
    for suffix, channel in (("_L", 0), ("_R", 2)):
        image = bpy.data.images.load(str(tmp_path / f"out_{suffix}.png"))
        pixel = np.array(image.pixels[:4])
        assert pixel[channel] > 0.9 and pixel[2 - channel] < 0.1, suffix


def test_a_plain_render_warns_once_per_render(addon_enabled, monkeypatch, tmp_path, capsys):
    from yt_volumes import handlers

    scene = addon_enabled
    scene.render.resolution_x = scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.frame_start, scene.frame_end = 1, 3
    scene.render.filepath = str(tmp_path / "plain_")
    _add_camera(scene, "PERSP", lens=50.0)
    _flagged_cube()
    popups = []
    monkeypatch.setattr(handlers, "_interactive", lambda: True)
    monkeypatch.setattr(bpy.app.timers, "register", lambda fn, **kw: popups.append(fn))

    bpy.ops.render.render(animation=True)  # Blender's own, not the pipeline
    first = capsys.readouterr().out.count("leaves the yt_idv volumes out")
    bpy.ops.render.render()

    assert first == 1  # not once per frame
    assert capsys.readouterr().out.count("leaves the yt_idv volumes out") == 1
    assert popups == [handlers._show_plain_render_warning] * 2


@pytest.mark.parametrize("output, expected", [("out_", "out_0005.png"), ("frame_###_x", "frame_005_x.png")])
def test_volume_override_renders_are_named_by_frame(addon_enabled, monkeypatch, tmp_path, output, expected):
    scene = addon_enabled
    scene.render.resolution_x = scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    _add_camera(scene, "PERSP", lens=50.0)
    _flagged_cube()
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(tmp_path / "work")
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", _fake_render_result)
    scene.render.filepath = str(tmp_path / output)
    scene.frame_set(5)

    operators._render_numbered_frame(bpy.context)  # what F12 and each animation frame run

    assert (tmp_path / expected).exists()
    assert scene.render.filepath == str(tmp_path / output)


def test_loading_a_colormap_fills_the_gn_ramp(clean_scene_with_props):
    import matplotlib

    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    obj.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    transfer_function_nodes.add_transfer_function_modifier(obj)
    with bpy.context.temp_override(object=obj):
        assert bpy.ops.volume_override.load_tf_colormap(colormap="viridis") == {"FINISHED"}

    rgba = np.asarray(operators._yt_idv_custom_params(obj)["yt_idv_gn_tf_rgba"])

    viridis = matplotlib.colormaps["viridis"]
    for i in (0, len(rgba) // 3, len(rgba) - 1):
        assert rgba[i, :3] == pytest.approx(viridis(i / (len(rgba) - 1))[:3], abs=1e-2)


def test_a_qualitative_colormap_loads_as_constant_steps(clean_scene_with_props):
    import matplotlib

    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    transfer_function_nodes.add_transfer_function_modifier(obj)
    ramp, _ = transfer_function_nodes.find_nodes(obj)
    transfer_function_nodes.load_colormap(ramp, "tab10")

    assert ramp.color_ramp.interpolation == "CONSTANT"
    assert len(ramp.color_ramp.elements) == 10
    rgba = transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get())
    assert rgba[len(rgba) // 2 + 20, :3] == pytest.approx(matplotlib.colormaps["tab10"].colors[5], abs=2e-3)


def test_log_opacity_keeps_the_curve_points_opacities(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes

    obj = _flagged_cube()
    transfer_function_nodes.add_transfer_function_modifier(obj)
    _, curve = transfer_function_nodes.find_nodes(obj)
    points = curve.mapping.curves[0].points
    points[-1].location = (1.0, 0.01)
    curve.mapping.update()
    tree = curve.id_data

    transfer_function_nodes.set_opacity_decades(tree, 4.0)
    obj.update_tag()

    assert points[-1].location[1] == pytest.approx(0.5)  # 1e-2 is half of 4 decades
    rgba = transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get())
    assert rgba[-1, 3] == pytest.approx(0.01, rel=1e-3)
    assert rgba[0, 3] == 0.0  # the curve's 0 stays transparent

    transfer_function_nodes.set_opacity_decades(tree, 0.0)
    obj.update_tag()
    assert points[-1].location[1] == pytest.approx(0.01)
    rgba = transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get())
    assert rgba[-1, 3] == pytest.approx(0.01, rel=1e-3)


def _pre_slot_transfer_function(obj, with_scale):
    """A transfer function as 0.1.0 (no opacity scale) or 0.1.1 made it: the
    ramp and curve right in the modifier's group.
    """
    from yt_volumes import transfer_function_nodes as tfn

    tfn.add_transfer_function_modifier(obj)
    tree = obj.modifiers[0].node_group
    for node in [n for n in tree.nodes if n.name.startswith((tfn.SLOT_PREFIX, tfn.MIX_PREFIX))]:
        if node.bl_idname == "GeometryNodeGroup":
            bpy.data.node_groups.remove(node.node_tree)
        tree.nodes.remove(node)
    tree.interface.remove(tree.interface.items_tree[tfn.BLEND_INPUT])
    value = tree.nodes[tfn.VALUE_NODE_NAME]
    value.name = "Separate XYZ"
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.name = tfn.COLOR_NODE_NAME
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 1.0, 1.0)
    curve = tree.nodes.new("ShaderNodeFloatCurve")
    curve.name = tfn.ALPHA_NODE_NAME
    curve.mapping.curves[0].points[1].location = (1.0, 0.5)
    curve.mapping.update()
    tree.links.new(value.outputs["X"], ramp.inputs["Fac"])
    tree.links.new(value.outputs["X"], curve.inputs["Value"])
    tree.links.new(ramp.outputs["Color"], tfn._store_node(tree, tfn.COLOR_ATTRIBUTE).inputs["Value"])
    tree.links.new(curve.outputs["Value"], tfn._store_node(tree, tfn.ALPHA_ATTRIBUTE).inputs["Value"])
    if with_scale:
        tfn.set_opacity_decades(tree, 2.0)
    obj.update_tag()
    return tree


def test_opacity_scale_upgrades_a_pre_scale_group(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tree = _pre_slot_transfer_function(obj, with_scale=False)

    with bpy.context.temp_override(object=obj):
        assert bpy.ops.volume_override.set_opacity_curve_scale(decades=3.0) == {"FINISHED"}

    assert tree.nodes[tfn.DECADES_NODE_NAME].outputs[0].default_value == 3.0
    rgba = tfn.sample(obj, bpy.context.evaluated_depsgraph_get())
    assert rgba[-1, 3] == pytest.approx(0.5, rel=1e-3)


def _sample(obj):
    from yt_volumes import transfer_function_nodes as tfn

    obj.update_tag()
    return tfn.sample(obj, bpy.context.evaluated_depsgraph_get())


def _two_slots(obj):
    """Slot 1 the starter; slot 2 all red, opacity a flat 0.2. Returns the tree."""
    from yt_volumes import transfer_function_nodes as tfn

    obj.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    tfn.add_transfer_function_modifier(obj)
    with bpy.context.temp_override(object=obj):
        assert bpy.ops.volume_override.add_tf_slot() == {"FINISHED"}
    assert obj.volume_override.tf_active_slot == 1
    ramp, curve = tfn.find_nodes(obj)
    for element in ramp.color_ramp.elements:
        element.color = (1.0, 0.0, 0.0, 1.0)
    for point in curve.mapping.curves[0].points:
        point.location = (point.location[0], 0.2)
    curve.mapping.update()
    return tfn.transfer_function_tree(obj)


def test_blend_cross_fades_between_slots(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    _two_slots(obj)
    tfn.set_blend(obj, 1.0)
    first = _sample(obj)

    tfn.set_blend(obj, 2.0)
    second = _sample(obj)
    tfn.set_blend(obj, 1.5)
    half = _sample(obj)

    assert second[:, 3] == pytest.approx(0.2, abs=2e-3)
    assert second[-1, :3] == pytest.approx([1.0, 0.0, 0.0], abs=2e-3)
    assert half[:, 3] == pytest.approx((first[:, 3] + second[:, 3]) / 2, abs=2e-3)
    # Where slot 1 is transparent, only slot 2's red shows.
    assert first[100, 3] == 0.0 and half[100, :3] == pytest.approx([1.0, 0.0, 0.0], abs=2e-3)
    tfn.set_blend(obj, 7.0)  # clamped to the last slot
    assert _sample(obj) == pytest.approx(second, abs=1e-6)


def test_log_slots_blend_in_log_opacity(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tree = _two_slots(obj)
    slots = tfn.slot_nodes(tree)
    # Slot 1 at 1e-4, slot 2 at 1e-2 (opacities kept as the scale changes).
    curve = slots[0].node_tree.nodes[tfn.ALPHA_NODE_NAME]
    for point in curve.mapping.curves[0].points:
        point.location = (point.location[0], 1e-4)
    curve.mapping.update()
    tfn.set_opacity_decades(slots[0].node_tree, 4.0)
    for point in slots[1].node_tree.nodes[tfn.ALPHA_NODE_NAME].mapping.curves[0].points:
        point.location = (point.location[0], 0.01)
    tfn.set_opacity_decades(slots[1].node_tree, 4.0)
    tfn.set_blend(obj, 1.5)

    rgba = _sample(obj)

    assert rgba[2000, 3] == pytest.approx(1e-3, rel=1e-2)  # the geometric mean, not 5e-3


def test_blend_can_be_keyframed(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tree = _two_slots(obj)
    scene = bpy.context.scene
    with bpy.context.temp_override(object=obj):
        for frame, slot in ((1, 0), (11, 1)):
            scene.frame_set(frame)
            obj.volume_override.tf_slot = str(slot)
            assert bpy.ops.volume_override.key_tf_slot() == {"FINISHED"}
        assert bpy.ops.volume_override.set_tf_blend_easing(easing="LINEAR") == {"FINISHED"}
    # Once Blend is animated, choosing a slot to edit leaves it alone.
    obj.volume_override.tf_slot = "0"
    assert tfn.blend_is_animated(obj) and tfn.blend_value(obj) == 2.0

    alphas = []
    for frame in (1, 6, 11):
        bpy.context.scene.frame_set(frame)
        alphas.append(_sample(obj)[-1, 3])

    assert alphas == pytest.approx([0.8, 0.5, 0.2], abs=2e-3)


def test_removing_a_slot_keeps_the_rest(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tree = _two_slots(obj)
    second = tfn.slot_nodes(tree)[1].node_tree
    obj.volume_override.tf_active_slot = 0
    with bpy.context.temp_override(object=obj):
        assert bpy.ops.volume_override.remove_tf_slot() == {"FINISHED"}
        assert bpy.ops.volume_override.remove_tf_slot.poll() is False

    assert [n.node_tree for n in tfn.slot_nodes(tree)] == [second]
    assert tfn.slot_nodes(tree)[0].name == f"{tfn.SLOT_PREFIX}1"
    assert _sample(obj)[:, 3] == pytest.approx(0.2, abs=2e-3)


@pytest.mark.parametrize("with_scale", [False, True])
def test_adding_a_slot_converts_an_older_group(clean_scene_with_props, with_scale):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tree = _pre_slot_transfer_function(obj, with_scale)
    before = _sample(obj)

    with bpy.context.temp_override(object=obj):
        assert bpy.ops.volume_override.add_tf_slot() == {"FINISHED"}

    assert tfn.COLOR_NODE_NAME not in tree.nodes and len(tfn.slot_nodes(tree)) == 2
    assert not [n for n in tree.nodes if n.bl_idname == "ShaderNodeFloatCurve" or getattr(n, "input_type", "") == "FLOAT"]
    assert _sample(obj) == pytest.approx(before, abs=2e-3)
    assert tfn.blend_value(obj) == 2.0  # the new slot is shown
    assert _sample(obj) == pytest.approx(before, abs=2e-3)  # the copy matches


def test_colormap_reload_with_fewer_stops_keeps_them_in_order(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tfn.add_transfer_function_modifier(obj)
    ramp, _ = tfn.find_nodes(obj)
    tfn.load_colormap(ramp, "viridis")
    tfn.load_colormap(ramp, "tab10")

    assert [round(e.position, 4) for e in ramp.color_ramp.elements] == [i / 10 for i in range(10)]


def test_a_colormap_loads_with_stops_where_it_bends(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    tfn.add_transfer_function_modifier(obj)
    ramp, _ = tfn.find_nodes(obj)
    tfn.load_colormap(ramp, "viridis", max_stops=8)
    positions = [e.position for e in ramp.color_ramp.elements]
    assert len(positions) == 8
    assert positions[0] == 0.0 and positions[-1] == 1.0
    tfn.load_colormap(ramp, "gray")
    assert len(ramp.color_ramp.elements) < 20  # stops once the fit is close
    assert tfn.colormap_name(obj) == "gray"


def test_slot_keyframes_can_be_removed(clean_scene_with_props):
    from yt_volumes import transfer_function_nodes as tfn

    obj = _flagged_cube()
    _two_slots(obj)
    with bpy.context.temp_override(object=obj):
        for frame in (1, 11):
            bpy.context.scene.frame_set(frame)
            bpy.ops.volume_override.key_tf_slot()
        bpy.ops.volume_override.remove_tf_slot_key(frame=1.0)
        assert [p.co[0] for p in tfn.blend_fcurve(obj).keyframe_points] == [11.0]
        bpy.ops.volume_override.remove_tf_slot_key(frame=11.0)
    assert tfn.blend_fcurve(obj) is None and not tfn.blend_is_animated(obj)


def test_the_annotation_scene_is_laid_over_the_render(addon_enabled, monkeypatch, tmp_path):
    from yt_volumes import annotations, compositor_build, transfer_function_nodes as tfn

    scene = addon_enabled
    scene.render.resolution_x, scene.render.resolution_y = 200, 112
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.view_settings.view_transform = "Standard"
    _add_camera(scene, "PERSP", lens=50.0)
    volume = _flagged_cube()
    volume.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    tfn.add_transfer_function_modifier(volume)
    n = tfn.NBINS
    tfn.update_distribution(volume, {"volume_fraction": np.zeros(n), "mass_fraction": np.zeros(n), "log10_value": np.linspace(-2.3, 1.7, n)})
    colorbar = annotations.add_colorbar(scene, volume)
    colorbar.location = (0.2, 0.12, 0.0)
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(tmp_path / "work")
    scene.render.filepath = str(tmp_path / "out_")
    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_GN_TRANSFER_FUNCTION", _fake_render_result)
    pass1_overlay_muted = []
    build_pass1 = compositor_build.build_pass1_capture_graph

    def recording_build_pass1(scene, *args):
        paths = build_pass1(scene, *args)
        node = compositor_build.ensure_compositor_tree(scene).nodes.get(compositor_build._node_name("Overlay"))
        pass1_overlay_muted.append(node and node.mute)
        return paths

    monkeypatch.setattr(compositor_build, "build_pass1_capture_graph", recording_build_pass1)
    with operators.render_session(bpy.context):
        for frame in (1, 2):  # the second reuses the built graph
            scene.frame_set(frame)
            operators.render_current_frame(bpy.context)
    # An unmuted Render Layers node would render the annotation scene in pass 1 too.
    assert pass1_overlay_muted == [None, True]

    image = bpy.data.images.load(str(tmp_path / "out_.png"))
    pixels = np.array(image.pixels[:]).reshape(112, 200, 4)  # bottom row first
    row = int(0.135 / annotations.aspect(scene) * 112)  # the bar's middle, in frame y
    # The starter ramp's middle stop, sRGB, at the bar's middle.
    assert pixels[row, 100, :3] == pytest.approx([0.735, 0.216, 0.330], abs=0.06)
    assert pixels[100, 100, :3] == pytest.approx(pixels[100, 10, :3], abs=1e-3)  # nothing drawn up there


def test_a_colorbar_ticks_whole_decades_of_the_fitted_range(clean_scene_with_props):
    from yt_volumes import annotations, transfer_function_nodes as tfn

    annotations.register()
    try:
        scene = bpy.context.scene
        volume = _flagged_cube()
        tfn.add_transfer_function_modifier(volume)
        n = tfn.NBINS
        tfn.update_distribution(volume, {"volume_fraction": np.zeros(n), "mass_fraction": np.zeros(n), "log10_value": np.linspace(-2.3, 1.7, n)})
        colorbar = annotations.add_colorbar(scene, volume)
        overlay = scene.yt_annotation_scene
        scene.frame_set(7)
        annotations.sync(scene, overlay)
        assert overlay.frame_current == 7 and overlay.camera.data.type == "ORTHO"
        assert annotations.add_annotation_scene(scene) == overlay  # one per scene
        bpy.data.node_groups.remove(bpy.data.node_groups[annotations.VIEW_GROUP])
        annotations.add_colorbar(scene, volume)
        assert annotations.VIEW_GROUP in bpy.data.node_groups  # made again for a scene that already exists

        with bpy.context.temp_override(scene=overlay, view_layer=overlay.view_layers[0]):
            depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh = colorbar.evaluated_get(depsgraph).data
        tick_bottom = -0.4 * 0.03  # ticks hang 0.4 bar heights below the bar
        xs = np.array([v.co.x for v in mesh.vertices if abs(v.co.y - tick_bottom) < 1e-5])
        centers = sorted({round(float(x), 4) for x in (xs[::2] + xs[1::2]) / 2})  # each tick's two bottom corners
        assert centers == pytest.approx([0.6 * (d + 2.3) / 4.0 for d in (-2, -1, 0, 1)], abs=1e-4)
    finally:
        annotations.unregister()


@pytest.mark.parametrize("engine, device, backend, mode, warns", [
    ("BLENDER_EEVEE", "CPU", "VULKAN", "SAMPLE_CONSTANT", True),
    ("BLENDER_EEVEE", "GPU", "VULKAN", "YT_IDV_MAX_INTENSITY", False),
    ("BLENDER_EEVEE", "GPU", "OPENGL", "YT_IDV_MAX_INTENSITY", True),  # forced onto the CPU compositor
    ("BLENDER_EEVEE", "GPU", "OPENGL", "SAMPLE_CONSTANT", False),
    ("CYCLES", "CPU", "VULKAN", "SAMPLE_CONSTANT", False),
])
def test_an_eevee_annotation_scene_beside_the_cpu_compositor_warns(addon_enabled, monkeypatch, engine, device, backend, mode, warns):
    from yt_volumes import annotations, transfer_function_nodes as tfn

    scene = addon_enabled
    volume = _flagged_cube()
    volume.volume_override.renderer_mode = mode
    tfn.add_transfer_function_modifier(volume)
    annotations.add_colorbar(scene, volume)
    scene.yt_annotation_scene.render.engine = engine
    scene.render.compositor_device = device
    monkeypatch.setattr(operators, "_gpu_backend", lambda: backend)

    assert bool(operators.compatibility_warnings(scene, bpy.context.view_layer)) == warns


def test_an_annotation_scene_with_only_its_camera_never_warns(addon_enabled):
    from yt_volumes import annotations

    scene = addon_enabled
    _flagged_cube()
    annotations.add_annotation_scene(scene).render.engine = "BLENDER_EEVEE"
    scene.render.compositor_device = "CPU"
    assert operators.compatibility_warnings(scene, bpy.context.view_layer) == []
