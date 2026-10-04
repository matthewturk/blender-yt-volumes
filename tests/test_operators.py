import os
from types import SimpleNamespace

import bpy
import numpy as np
import pytest
from mathutils import Vector

from cycles_volume_override import change_tracking, operators, properties


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
    obj.volume_override.density_scale = 1.0

    operators.build_volume_params(obj)  # first call, establishes history

    obj.volume_override.density_scale = 2.0  # cheap param changes, dataset doesn't
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


def test_build_volume_params_has_no_yt_idv_keys_when_addon_not_enabled(clean_scene_with_props):
    # Only `properties` is registered, so preferences.addons[...] raises KeyError.
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    params = operators.build_volume_params(obj)

    assert not any(k.startswith("yt_idv_") for k in params.custom_params)


@pytest.fixture
def addon_enabled():
    # register_class alone doesn't populate preferences.addons[...]; a real enable does.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="cycles_volume_override")
    change_tracking.reset()
    yield bpy.context.scene
    bpy.ops.preferences.addon_disable(module="cycles_volume_override")


def test_build_volume_params_includes_yt_idv_settings_when_addon_enabled(addon_enabled):
    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.yt_idv.fog_colormap = "inferno"
    prefs.yt_idv.fog_ramp_power = 0.42

    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    params = operators.build_volume_params(obj)

    assert params.custom_params["yt_idv_fog_colormap"] == "inferno"
    assert params.custom_params["yt_idv_fog_ramp_power"] == pytest.approx(0.42)
    # Untouched fields default to yt_idv_renderer's module constants.
    from cycles_volume_override import yt_idv_renderer as yir
    assert params.custom_params["yt_idv_mip_alpha_max"] == pytest.approx(yir.MIP_ALPHA_MAX)


def test_yt_idv_custom_params_omits_curve_and_image_keys_by_default(addon_enabled):
    params = operators._yt_idv_custom_params()
    assert "yt_idv_alpha_curve" not in params
    assert "yt_idv_color_image_rgba" not in params


def test_get_or_create_alpha_curve_node_is_stable_across_calls(addon_enabled):
    node1 = operators._get_or_create_alpha_curve_node()
    node1.mapping.initialize()
    node1.mapping.curves[0].points.new(0.5, 0.75)
    node1.mapping.update()

    node2 = operators._get_or_create_alpha_curve_node()

    # Compare .as_pointer(): bpy wrappers aren't cached, so `is` is unreliable.
    assert node2.name == node1.name
    assert node2.as_pointer() == node1.as_pointer()
    assert len(node2.mapping.curves[0].points) == 3


def test_evaluate_alpha_curve_default_is_linear(addon_enabled):
    values = operators._evaluate_alpha_curve(nbins=5)
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-3)


def test_yt_idv_custom_params_includes_alpha_curve_when_selected(addon_enabled):
    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.yt_idv.alpha_source = "CURVE"

    params = operators._yt_idv_custom_params()

    assert "yt_idv_alpha_curve" in params
    assert len(params["yt_idv_alpha_curve"]) == 256
    assert params["yt_idv_alpha_curve"][0] == pytest.approx(0.0, abs=1e-3)
    assert params["yt_idv_alpha_curve"][-1] == pytest.approx(1.0, abs=1e-3)


def test_sample_image_as_1d_rgba_averages_rows_and_resamples_width(addon_enabled):
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


def test_yt_idv_custom_params_includes_color_image_rgba_when_selected(addon_enabled):
    import numpy as np

    img = bpy.data.images.new("vo_test_gradient2", width=4, height=1, alpha=True, float_buffer=True)
    data = np.zeros((1, 4, 4), dtype=np.float32)
    data[..., 2] = np.linspace(0.0, 1.0, 4)[None, :]
    data[..., 3] = 1.0
    img.pixels.foreach_set(data.ravel())
    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.yt_idv.color_source = "IMAGE"
    prefs.yt_idv.color_image_name = img.name

    try:
        params = operators._yt_idv_custom_params()
        assert "yt_idv_color_image_rgba" in params
        assert len(params["yt_idv_color_image_rgba"]) == 256
        assert params["yt_idv_color_image_rgba"][0][2] == pytest.approx(0.0, abs=1e-3)
        assert params["yt_idv_color_image_rgba"][-1][2] == pytest.approx(1.0, abs=1e-3)
    finally:
        bpy.data.images.remove(img)


def test_yt_idv_custom_params_ignores_stale_color_image_name(addon_enabled):
    # Image renamed/deleted after selection: fall back silently, don't raise mid-render.
    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.yt_idv.color_source = "IMAGE"
    prefs.yt_idv.color_image_name = "does_not_exist_anywhere"

    params = operators._yt_idv_custom_params()

    assert "yt_idv_color_image_rgba" not in params


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


# --- Time mapping (requested_time) ------------------------------------------

def _flagged_cube(scene):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    return obj


def test_resolve_requested_time_direct_matches_scene_frame(clean_scene_with_props):
    obj = _flagged_cube(clean_scene_with_props)
    obj.volume_override.mapping_policy = "DIRECT"
    clean_scene_with_props.frame_set(7)
    assert operators.resolve_requested_time(obj) == 7.0


def test_resolve_requested_time_affine_applies_scale_and_offset(clean_scene_with_props):
    obj = _flagged_cube(clean_scene_with_props)
    obj.volume_override.mapping_policy = "AFFINE"
    obj.volume_override.time_scale = 2.0
    obj.volume_override.time_offset = 100.0
    clean_scene_with_props.frame_set(7)
    assert operators.resolve_requested_time(obj) == pytest.approx(114.0)  # 7*2+100


def test_resolve_requested_time_manual_reads_property_directly(clean_scene_with_props):
    obj = _flagged_cube(clean_scene_with_props)
    obj.volume_override.mapping_policy = "MANUAL"
    obj.volume_override.manual_time = 42.5
    clean_scene_with_props.frame_set(999)  # irrelevant under MANUAL
    assert operators.resolve_requested_time(obj) == pytest.approx(42.5)


def test_requested_time_change_does_not_flip_dataset_version_changed(clean_scene_with_props):
    """Advancing frames must not look like a dataset change (forcing a reload)."""
    obj = _flagged_cube(clean_scene_with_props)
    obj.volume_override.dataset_path = "/data/sim_001"
    obj.volume_override.mapping_policy = "DIRECT"
    cache_key = properties.ensure_dataset_uuid(obj)

    clean_scene_with_props.frame_set(1)
    first = operators.build_dataset_params(obj, cache_key)
    assert first.requested_time == 1.0

    clean_scene_with_props.frame_set(2)
    second = operators.build_dataset_params(obj, cache_key)
    assert second.requested_time == 2.0
    assert second.version.changed is False  # only requested_time moved
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

    from cycles_volume_override import external_render

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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
    prefs.work_dir = str(tmp_path)

    seen_view_transform = []

    def fake_renderer(request):
        seen_view_transform.append(scene.view_settings.view_transform)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_view_transform == ["Standard"]  # forced during the render
    assert scene.view_settings.view_transform == "AgX"  # restored afterward, untouched


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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "SAMPLE_CONSTANT"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
    prefs.work_dir = str(tmp_path)

    seen_hide_render = []

    def fake_renderer(request):
        seen_hide_render.append(box.hide_render)
        return _fake_render_result(request)

    monkeypatch.setitem(operators.RENDERER_MODES, "YT_IDV_MAX_INTENSITY", fake_renderer)

    operators.render_current_frame(bpy.context)

    assert seen_hide_render == [False]  # never set to True at any point
    assert box.hide_render is False


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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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

    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.renderer_mode = "YT_IDV_MAX_INTENSITY"
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


# --- opacity field / _apply_opacity -------------------------------------------

def test_apply_opacity_is_a_noop_at_default_1_0():
    import numpy as np
    from cycles_volume_override import external_render

    result = external_render.RenderResult(
        rgba=np.ones((2, 2, 4), dtype="float32") * 0.8,
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )

    out = operators._apply_opacity(result, 1.0)

    assert out is result  # exact no-op, not just numerically equal


def test_apply_opacity_scales_premultiplied_rgba_uniformly():
    import numpy as np
    from cycles_volume_override import external_render

    result = external_render.RenderResult(
        rgba=np.ones((2, 2, 4), dtype="float32") * 0.8,
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )

    out = operators._apply_opacity(result, 0.5)

    assert np.allclose(out.rgba, 0.4)
    assert out.rgba.dtype == np.dtype("float32")
    assert out.alpha_mode == "PREMULTIPLIED"
    assert out.color_space == "LINEAR"


def test_write_render_result_reuses_the_same_image_datablock_across_calls(tmp_path):
    # Regression (perf): images.new() + remove() per call invalidated
    # Cycles' persistent-data cache.
    import numpy as np

    from cycles_volume_override import external_render

    path_a = str(tmp_path / "a.exr")
    path_b = str(tmp_path / "b.exr")
    result_a = external_render.RenderResult(
        rgba=np.full((2, 2, 4), 0.25, dtype="float32"),
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )
    result_b = external_render.RenderResult(
        rgba=np.full((2, 2, 4), 0.75, dtype="float32"),
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )

    operators.write_render_result(result_a, path_a)
    img_after_first = bpy.data.images.get(operators._RESULT_IMAGE_NAME)
    assert img_after_first is not None
    pointer_after_first = img_after_first.as_pointer()

    operators.write_render_result(result_b, path_b)
    img_after_second = bpy.data.images.get(operators._RESULT_IMAGE_NAME)

    assert img_after_second is not None
    assert img_after_second.as_pointer() == pointer_after_first
    assert os.path.exists(path_a)
    assert os.path.exists(path_b)

    reread_a = operators.read_depth_buffer(bpy.context.scene, path_a)
    reread_b = operators.read_depth_buffer(bpy.context.scene, path_b)
    assert np.allclose(reread_a.data, 0.25, atol=1e-5)
    assert np.allclose(reread_b.data, 0.75, atol=1e-5)


def test_write_render_result_reuses_datablock_after_previous_file_is_deleted(tmp_path):
    # Regression: Image.save() left the datablock backed by the (since
    # deleted) file, reading back as (0, 0), so reuse never held per-frame.
    import numpy as np

    from cycles_volume_override import external_render

    def result(value):
        return external_render.RenderResult(
            rgba=np.full((3, 4, 4), value, dtype="float32"),
            alpha_mode="PREMULTIPLIED", color_space="LINEAR",
        )

    path_a = str(tmp_path / "a.exr")
    operators.write_render_result(result(0.25), path_a)
    img = bpy.data.images[operators._RESULT_IMAGE_NAME]
    pointer_after_first = img.as_pointer()
    assert img.source == "GENERATED"
    os.remove(path_a)
    img.buffers_free()

    path_b = str(tmp_path / "b.exr")
    operators.write_render_result(result(0.75), path_b)
    img = bpy.data.images[operators._RESULT_IMAGE_NAME]
    assert img.as_pointer() == pointer_after_first
    assert tuple(img.size) == (4, 3)
    assert np.allclose(operators.read_depth_buffer(bpy.context.scene, path_b).data, 0.75, atol=1e-5)


def test_read_depth_buffer_reuses_the_same_image_datablock_across_calls(tmp_path):
    import numpy as np

    from cycles_volume_override import external_render

    path_a = str(tmp_path / "depth_a.exr")
    path_b = str(tmp_path / "depth_b.exr")
    operators.write_render_result(
        external_render.RenderResult(
            rgba=np.full((2, 2, 4), 1.5, dtype="float32"),
            alpha_mode="PREMULTIPLIED", color_space="LINEAR",
        ),
        path_a,
    )
    operators.write_render_result(
        external_render.RenderResult(
            rgba=np.full((2, 2, 4), 3.0, dtype="float32"),
            alpha_mode="PREMULTIPLIED", color_space="LINEAR",
        ),
        path_b,
    )

    scene = bpy.context.scene
    depth_a = operators.read_depth_buffer(scene, path_a)
    img_after_first = bpy.data.images.get(operators._DEPTH_IMAGE_NAME)
    assert img_after_first is not None
    pointer_after_first = img_after_first.as_pointer()

    depth_b = operators.read_depth_buffer(scene, path_b)
    img_after_second = bpy.data.images.get(operators._DEPTH_IMAGE_NAME)

    assert img_after_second is not None
    assert img_after_second.as_pointer() == pointer_after_first
    scale_length = scene.unit_settings.scale_length
    assert np.allclose(depth_a.data, 1.5 * scale_length, atol=1e-4)
    assert np.allclose(depth_b.data, 3.0 * scale_length, atol=1e-4)


def test_volume_override_opacity_field_defaults_to_1_0_and_is_keyframable(clean_scene_with_props):
    # Lives on the Object, not AddonPreferences, because only the former is keyframable.
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    assert obj.volume_override.opacity == pytest.approx(1.0)

    obj.volume_override.opacity = 0.0
    obj.volume_override.keyframe_insert(data_path="opacity", frame=1)
    obj.volume_override.opacity = 1.0
    obj.volume_override.keyframe_insert(data_path="opacity", frame=21)

    assert obj.animation_data is not None and obj.animation_data.action is not None

    clean_scene_with_props.frame_set(1)
    assert obj.volume_override.opacity == pytest.approx(0.0)
    clean_scene_with_props.frame_set(21)
    assert obj.volume_override.opacity == pytest.approx(1.0)
    clean_scene_with_props.frame_set(11)  # midpoint, linear interpolation
    assert obj.volume_override.opacity == pytest.approx(0.5, abs=1e-3)


def test_every_yt_idv_renderer_mode_gets_the_cpu_compositor():
    # A yt_idv mode missing from this set leaves the GPU compositor on, and
    # yt_idv's eglMakeCurrent then fails with EGL_BAD_ACCESS.
    yt_idv_modes = {m for m, fn in operators.RENDERER_MODES.items() if fn.__module__.endswith("yt_idv_renderer")}
    assert yt_idv_modes == operators.YT_IDV_RENDERER_MODES
    assert {"YT_IDV_MAX_INTENSITY_ALPHA", "YT_IDV_THRESHOLD_FOG"} <= yt_idv_modes


def test_yt_idv_modes_refuse_a_non_cycles_engine(monkeypatch, clean_scene_with_props):
    scene = clean_scene_with_props
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    monkeypatch.setattr(operators, "_get_prefs", lambda context: SimpleNamespace(renderer_mode="YT_IDV_MAX_INTENSITY", verbose_logging=False))
    with pytest.raises(RuntimeError, match="need the Cycles render engine"):
        operators.render_current_frame(bpy.context)
