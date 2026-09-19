from types import SimpleNamespace

import bpy
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
    bpy.ops.mesh.primitive_cube_add()
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
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    cache_key = properties.ensure_dataset_uuid(obj)

    operators.build_dataset_params(obj, cache_key)
    second = operators.build_dataset_params(obj, cache_key)
    assert second.version.changed is False


def test_build_dataset_params_path_change_reports_changed(clean_scene_with_props):
    bpy.ops.mesh.primitive_cube_add()
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
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.dataset_path = "/data/sim_001"
    obj.volume_override.density_scale = 1.0

    operators.build_volume_params(obj)  # first call, establishes history

    obj.volume_override.density_scale = 2.0  # cheap param changes, dataset doesn't
    second = operators.build_volume_params(obj)
    assert second.params_version.changed is True
    assert second.dataset.version.changed is False


def test_build_volume_params_has_no_yt_idv_keys_when_addon_not_enabled(clean_scene_with_props):
    # `clean_scene_with_props` registers `properties` only, not the whole
    # add-on -- `bpy.context.preferences.addons[__package__]` genuinely
    # doesn't exist in that state (confirmed: raises KeyError), which is
    # exactly the case `_yt_idv_custom_params`'s own try/except is for. Pins
    # the fallback down explicitly, not just "the other test still passes".
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    params = operators.build_volume_params(obj)

    assert not any(k.startswith("yt_idv_") for k in params.custom_params)


@pytest.fixture
def addon_enabled():
    # Real add-on-enable flow (not just bpy.utils.register_class) -- that
    # alone does NOT populate bpy.context.preferences.addons[__package__]
    # (confirmed: raises KeyError), which _yt_idv_custom_params reads.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="cycles_volume_override")
    change_tracking.reset()
    yield bpy.context.scene
    bpy.ops.preferences.addon_disable(module="cycles_volume_override")


def test_build_volume_params_includes_yt_idv_settings_when_addon_enabled(addon_enabled):
    prefs = bpy.context.preferences.addons["cycles_volume_override"].preferences
    prefs.yt_idv.fog_colormap = "inferno"
    prefs.yt_idv.fog_ramp_power = 0.42

    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True

    params = operators.build_volume_params(obj)

    assert params.custom_params["yt_idv_fog_colormap"] == "inferno"
    assert params.custom_params["yt_idv_fog_ramp_power"] == pytest.approx(0.42)
    # A field left at its default should match yt_idv_renderer.py's own
    # module constant of the same name -- the whole point of the defaults
    # being "the old hardcoded behavior, unchanged, until you touch a slider".
    from cycles_volume_override import yt_idv_renderer as yir
    assert params.custom_params["yt_idv_mip_alpha_max"] == pytest.approx(yir.MIP_ALPHA_MAX)


def test_yt_idv_custom_params_omits_curve_and_image_keys_by_default(addon_enabled):
    # alpha_source/color_source default to RAMP_POWER/NAMED -- the new
    # curve/image mechanism (see yt_idv_renderer.py module docstring's
    # "Curve-driven alpha / image-driven color" section) must be a
    # completely inert no-op until explicitly opted into.
    params = operators._yt_idv_custom_params()
    assert "yt_idv_alpha_curve" not in params
    assert "yt_idv_color_image_rgba" not in params


def test_get_or_create_alpha_curve_node_is_stable_across_calls(addon_enabled):
    node1 = operators._get_or_create_alpha_curve_node()
    node1.mapping.initialize()
    node1.mapping.curves[0].points.new(0.5, 0.75)
    node1.mapping.update()

    node2 = operators._get_or_create_alpha_curve_node()

    # NOT `is` -- repeated bpy_struct attribute access doesn't reliably
    # return the identical Python wrapper object even for the same
    # underlying RNA data; compare by name/identity-of-data instead.
    assert node2.name == node1.name
    assert node2.as_pointer() == node1.as_pointer()
    assert len(node2.mapping.curves[0].points) == 3


def test_evaluate_alpha_curve_default_is_linear(addon_enabled):
    # Untouched CurveMapping default is the identity ramp (points at (0,0)
    # and (1,1)) -- see module investigation confirming this via a direct
    # ShaderNodeFloatCurve test.
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
    # A genuinely 0x0 image can't be produced via bpy.data.images.new()
    # itself (confirmed: Blender clamps width/height to a 1x1 minimum) --
    # this is the same "broken/unavailable source file" failure mode
    # documented in operators.read_depth_buffer's own comment, reproduced
    # here with a minimal stand-in rather than a real (unbreakable) Image.
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
    # Image renamed/deleted after being selected -- see
    # `_yt_idv_custom_params`'s own comment on why this falls back silently
    # (to the NAMED-colormap mechanism) rather than raising mid-render.
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
    bpy.ops.mesh.primitive_cube_add()
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
    """The whole point of excluding requested_time from the identity hash:
    advancing frames (the normal case) must NOT look like "the dataset
    changed" to the renderer, or every frame would force a reload.
    """
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
    """Enabling this add-on remaps F12/Ctrl+F12 to render.volume_override
    unconditionally (see handlers.py) -- so for a scene with no flagged
    volumes, this operator must behave exactly like Blender's own
    render.render, not run the two-pass pipeline (which forces the Z pass
    on, rebuilds the compositor graph, and swaps the engine to Workbench).
    Otherwise installing the add-on breaks ordinary rendering everywhere,
    even in scenes that never use the override.
    """
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
    bpy.ops.mesh.primitive_cube_add()
    box = bpy.context.active_object
    box.volume_override.is_volume_override = True

    called = []
    monkeypatch.setattr(operators, "render_current_frame", lambda *a, **k: called.append(True))
    monkeypatch.setattr(operators, "_show_render_result", lambda *a, **k: None)

    result = bpy.ops.render.volume_override(animation=False)

    assert result == {"FINISHED"}
    assert called == [True]


# --- opacity field / _apply_opacity -------------------------------------------

def test_apply_opacity_is_a_noop_at_default_1_0():
    import numpy as np
    from cycles_volume_override import external_render

    result = external_render.RenderResult(
        rgba=np.ones((2, 2, 4), dtype="float32") * 0.8,
        alpha_mode="PREMULTIPLIED", color_space="LINEAR",
    )

    out = operators._apply_opacity(result, 1.0)

    assert out is result  # exact no-op, not just numerically equal -- see _apply_opacity's own docstring


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


def test_volume_override_opacity_field_defaults_to_1_0_and_is_keyframable(clean_scene_with_props):
    # The whole point of putting `opacity` on obj.volume_override (a real
    # Object-level PropertyGroup) rather than alongside the yt_idv alpha
    # controls in AddonPreferences: Object supports animation_data, so
    # standard keyframe_insert/frame_set actually animates it -- confirmed
    # directly (AddonPreferences instances have no animation_data at all,
    # id_data is None, keyframe_insert raises outright).
    bpy.ops.mesh.primitive_cube_add()
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
