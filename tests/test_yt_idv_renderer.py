"""Unit tests for yt_idv_renderer.py's pure-Python helpers ONLY -- the parts
that don't need `yt`/`yt_idv` importable at all. The actual yt_idv-dependent
render path (`_load_sample_source`, `_get_render_context`,
`yt_idv_max_intensity_renderer` end to end) is instead verified by actually
running the real scripts in `scripts/yt_idv_exploration/blender_scene/`
(`render_real_demo.py`, `verify_overlap.py`) -- see yt_idv_renderer.py's
module docstring and README "yt_idv integration" for the full story.

This suite's own venv (this repo's `dev` dependency group: `bpy`/`pytest`)
does NOT need `yt`/`matplotlib`/`yt_idv` to run -- and historically COULDN'T
have them, since PyPI's `yt_idv` pulled in `imgui`, which had no Python
3.13 wheel. That's fixed upstream now (yt-project/yt_idv#268), so `bpy` and
real `yt_idv` genuinely coexist in one Python 3.13 venv (the optional
`ytidv` group, on top of `dev`, in the SAME `.venv` this suite runs in) --
meaning `yt`/`matplotlib` MAY be importable here too, depending on whether
that group has also been synced. Tests below that rely on the colormap
path failing (`import yt`/`import matplotlib`) force that failure via
`sys.modules` rather than assuming the ambient venv lacks them.
"""

import math
import sys
import types

import bpy
import numpy as np
import pytest

from cycles_volume_override import change_tracking, operators
from cycles_volume_override import yt_idv_renderer as yir
from cycles_volume_override.external_render import (
    CameraParams,
    DatasetParams,
    DepthBuffer,
    DepthOfFieldParams,
    FrameParams,
    RenderRequest,
    VersionInfo,
    VolumeObjectParams,
)


def _persp_camera(**overrides):
    defaults = dict(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        projection="PERSP",
        lens_mm=50.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_fit="AUTO",
        shift_x=0.0,
        shift_y=0.0,
        ortho_scale=None,
        clip_start=0.1,
        clip_end=1000.0,
        resolution_x=1920,
        resolution_y=1080,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
        projection_matrix=None,
        panorama=None,
        dof=DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    defaults.update(overrides)
    return CameraParams(**defaults)


class _FakeDataset:
    """Trivial stand-in for a real yt Dataset -- just enough surface for
    _camera_update_kwargs/_domain_array_unitary (`.domain_center`/
    `.domain_width` as plain arrays, no `.in_units`, per
    _domain_array_unitary's docstring on why that's supported).
    """

    domain_center = np.array([0.5, 0.5, 0.5])
    domain_width = np.array([1.0, 1.0, 1.0])


class _FakeYTArray(np.ndarray):
    """Stand-in for a real yt YTArray -- has `.in_units(...)` returning
    something with a `.d` attribute, like the real thing, but in
    non-unitary units so _domain_array_unitary's conversion is actually
    exercised (not just a no-op pass-through).
    """

    def in_units(self, name):
        assert name == "unitary"
        # Simulate a dataset whose domain is 2 code_length wide -- "unitary"
        # (fraction of domain width) is this array's own values / 2.
        converted = (np.asarray(self) / 2.0).view(np.ndarray)
        return type("Converted", (), {"d": converted})()


def test_approx_vertical_fov_deg_horizontal_sensor_fit_matches_manual_formula():
    camera = _persp_camera(sensor_fit="AUTO", sensor_width_mm=36.0, lens_mm=50.0)
    aspect = 1920 / 1080
    got = yir._approx_vertical_fov_deg(camera, aspect)

    horiz_fov = 2 * math.atan(18.0 / 50.0)
    expected = math.degrees(2 * math.atan(math.tan(horiz_fov / 2) / aspect))
    assert got == pytest.approx(expected)


def test_approx_vertical_fov_deg_vertical_sensor_fit_ignores_aspect():
    camera = _persp_camera(sensor_fit="VERTICAL", sensor_height_mm=24.0, lens_mm=50.0)
    expected = math.degrees(2 * math.atan(12.0 / 50.0))
    assert yir._approx_vertical_fov_deg(camera, aspect_ratio=2.5) == pytest.approx(expected)
    assert yir._approx_vertical_fov_deg(camera, aspect_ratio=0.5) == pytest.approx(expected)


def _cube_bbox(half_extent=1.0):
    return [
        [x, y, z]
        for x in (-half_extent, half_extent)
        for y in (-half_extent, half_extent)
        for z in (-half_extent, half_extent)
    ]


def _identity_volume(bounding_box_local=None, matrix_world=None):
    """A flagged object registered to yt_idv's unitary domain -- default is
    a 2x2x2 cube at the origin (Blender's own default cube), identity
    transform.
    """
    version = VersionInfo(hash="x", changed=True)
    dataset = DatasetParams(
        dataset_id="obj", dataset_kind="", dataset_path=None,
        requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
        is_blender_volume_sequence=False, sequence_frame_start=None,
        sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
        version=version,
    )
    return VolumeObjectParams(
        matrix_world=matrix_world or [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bounding_box_local=bounding_box_local or _cube_bbox(),
        custom_params={}, params_version=version, dataset=dataset,
    )


def test_bbox_local_min_extent_computes_min_and_extent():
    bbox = [[x, y, z] for x in (-1, 1) for y in (-2, 2) for z in (0, 4)]
    local_min, extent = yir._bbox_local_min_extent(bbox)
    np.testing.assert_allclose(local_min, [-1, -2, 0])
    np.testing.assert_allclose(extent, [2, 4, 4])


def test_bbox_local_min_extent_guards_degenerate_axis_to_avoid_divide_by_zero():
    bbox = [[0.0, 0.0, 0.0]] * 8  # fully degenerate object bounding box
    _, extent = yir._bbox_local_min_extent(bbox)
    np.testing.assert_allclose(extent, [1.0, 1.0, 1.0])  # guarded, not 0


def test_registered_camera_kwargs_rejects_non_persp():
    camera = _persp_camera(projection="ORTHO")
    with pytest.raises(NotImplementedError, match="perspective-only"):
        yir._registered_camera_kwargs(camera, _identity_volume(), _FakeDataset(), 100, 100)


def test_registered_camera_kwargs_maps_position_through_object_bbox_registration():
    # Camera at world (0, 0, 5), identity rotation (looks down -Z, up +Y).
    # Object: default 2x2x2 cube at the world origin -- unitary (0.5, 0.5,
    # 0.5) is the object's own center, by construction.
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    volume = _identity_volume()
    ds = _FakeDataset()  # domain_center == (0.5, 0.5, 0.5), matches the object's own unitary center

    kwargs = yir._registered_camera_kwargs(camera, volume, ds, width=200, height=100)

    # position = (cam_pos_local - bbox_min) / bbox_extent = ((0,0,5)-(-1,-1,-1))/(2,2,2)
    np.testing.assert_allclose(kwargs["position"], [0.5, 0.5, 3.0], atol=1e-6)
    # Camera sits exactly on the ray from the object's center along -Z, so
    # looking through the (real, transformed) forward direction lands
    # exactly back on the domain center -- not a coincidence of this
    # particular symmetric test case, but confirms the two aren't
    # independently wrong in a way that happens to cancel out either.
    np.testing.assert_allclose(kwargs["focus"], [0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(kwargs["up"], [0, 1, 0], atol=1e-6)
    assert kwargs["aspect_ratio"] == pytest.approx(2.0)


def test_registered_camera_kwargs_position_scales_with_object_size():
    # SAME camera/object placement as above, but the object is HALF the
    # size (bbox half_extent=0.5 instead of 1.0) -- the object now occupies
    # a smaller fraction of unitary space, so the same real-world distance
    # from camera to object maps to a LARGER unitary distance.
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    volume = _identity_volume(bounding_box_local=_cube_bbox(half_extent=0.5))
    ds = _FakeDataset()

    kwargs = yir._registered_camera_kwargs(camera, volume, ds, width=100, height=100)

    # ((0,0,5) - (-0.5,-0.5,-0.5)) / (1,1,1) = (0.5, 0.5, 5.5)
    np.testing.assert_allclose(kwargs["position"], [0.5, 0.5, 5.5], atol=1e-6)


def test_near_far_planes_bounds_the_unit_cube_with_margin():
    # Camera well outside the unit cube on the diagonal -- nearest corner
    # (1,1,1), farthest corner (0,0,0).
    position = np.array([2.0, 2.0, 2.0])
    planes = yir._near_far_planes(position, margin=0.1)
    nearest = np.linalg.norm(np.array([1.0, 1.0, 1.0]) - position)
    farthest = np.linalg.norm(np.array([0.0, 0.0, 0.0]) - position)
    assert planes["near_plane"] == pytest.approx(nearest - 0.1)
    assert planes["far_plane"] == pytest.approx(farthest + 0.1)


def test_near_far_planes_floors_near_plane_when_camera_is_inside_the_cube():
    # Camera at the cube's own center -- nearest corner distance minus
    # margin would go negative; must floor to the documented 1e-5, not a
    # negative or zero near plane (real yt_idv cameras reject non-positive
    # near planes).
    position = np.array([0.5, 0.5, 0.5])
    planes = yir._near_far_planes(position, margin=1.0)
    assert planes["near_plane"] == pytest.approx(1e-5)


def test_near_far_planes_ratio_does_not_blow_up_with_camera_distance():
    # The bug this replaced (`near = 0.01 * look_ahead`, `far = look_ahead *
    # 4.0`) had a CONSTANT 400:1 ratio no matter how far the camera sat from
    # the domain -- the whole point of the fix is that the ratio now tracks
    # the domain's own (small, fixed) angular size instead of blowing up
    # with distance. A camera far away should see a near/far ratio close to
    # 1.0 (near and far corners of a small cube look almost equidistant from
    # very far away), nowhere near the old formula's fixed 400.
    position = np.array([100.0, 100.0, 100.0])
    planes = yir._near_far_planes(position, margin=0.1)
    ratio = planes["far_plane"] / planes["near_plane"]
    assert ratio < 2.0


def test_domain_array_unitary_converts_real_ytarray_like_objects():
    value = np.asarray([4.0, 4.0, 4.0]).view(_FakeYTArray)
    got = yir._domain_array_unitary(value)
    np.testing.assert_allclose(got, [2.0, 2.0, 2.0])


def test_domain_array_unitary_passes_through_plain_arrays():
    got = yir._domain_array_unitary((1.0, 2.0, 3.0))
    np.testing.assert_allclose(got, [1.0, 2.0, 3.0])


# --- Custom volume setup harness (yt_idv_volume_provider) -------------------
#
# `_resolve_volume_provider`/`_default_volume_provider`/`_get_volume_scene`/
# `_render_signature*` are all pure Python (or take fakes standing in for a
# real yt_idv render_context/scene) -- no real yt/yt_idv/GL context needed,
# same testability discipline as everything else in this file. The actual
# `_render_with_yt_idv` skip-render behavior IS covered here too (with a
# fully faked render_context/scene/camera) since it's pure orchestration
# logic, not a real GL render -- only `rc.run()`/`set_external_depth_clip`'s
# OWN internals stay verified manually against real yt_idv (see module
# docstring "Custom volume setup").

@pytest.fixture
def _clean_yir_state():
    """Every test below shares yt_idv_renderer.py's module-level caches AND
    change_tracking's global `_last_hashes` (the same store operators.py's
    own params/dataset version tracking uses, just a different category) --
    without resetting both before AND after, an earlier test's leftover
    state (e.g. a `dataset_id="obj"` entry) would make a later test's
    "first call" no longer actually be first.
    """
    def _clear():
        yir._volume_scene_cache.clear()
        yir._render_result_cache.clear()
        yir._render_context_cache.clear()
        yir._dataset_cache.clear()
        change_tracking.reset()
    _clear()
    yield
    _clear()


class _FakeRenderContext:
    """Stand-in for a real yt_idv render context -- just enough surface for
    `_default_volume_provider`/a custom provider to call `.add_scene(...)`
    and for tests to confirm how many times/with what arguments it was.
    """

    def __init__(self):
        self.add_scene_calls = []

    def add_scene(self, data_source, field, **kwargs):
        self.add_scene_calls.append((data_source, field, kwargs))
        return f"scene[{field}]#{len(self.add_scene_calls)}"


def _request_with(custom_params=None, dataset_id="obj", camera_overrides=None, render_size=4):
    """A full, real `RenderRequest` -- real (small) numpy depth data, a real
    `CameraParams`/`VolumeObjectParams` -- for exercising `_render_signature`/
    `_render_signature_changed`/`_render_with_yt_idv` end to end without any
    real yt/yt_idv/bpy involved.
    """
    volume = _identity_volume()
    volume.dataset.dataset_id = dataset_id
    volume.custom_params = custom_params if custom_params is not None else {}
    return RenderRequest(
        camera=_persp_camera(**(camera_overrides or {})),
        depth=DepthBuffer(
            data=np.zeros((render_size, render_size), dtype=np.float32),
            depth_convention="EUCLIDEAN", no_hit_value=1e10,
            row_order="TOP_LEFT_ORIGIN", units="meters",
        ),
        volume=volume,
        frame=FrameParams(frame_number=1, scene_time_seconds=0.0),
    )


def test_resolve_volume_provider_imports_module_and_returns_named_function(monkeypatch):
    fake_module = types.ModuleType("_test_yir_provider_mod_1")

    def setup_fn(ctx):
        return "sentinel"

    fake_module.setup_fn = setup_fn
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_1", fake_module)

    resolved = yir._resolve_volume_provider("_test_yir_provider_mod_1:setup_fn")

    assert resolved is setup_fn


def test_resolve_volume_provider_rejects_string_without_colon():
    with pytest.raises(ValueError, match="module.submodule:function_name"):
        yir._resolve_volume_provider("no_colon_here")


def test_resolve_volume_provider_raises_attributeerror_for_missing_function(monkeypatch):
    fake_module = types.ModuleType("_test_yir_provider_mod_2")
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_2", fake_module)

    with pytest.raises(AttributeError, match="missing_fn"):
        yir._resolve_volume_provider("_test_yir_provider_mod_2:missing_fn")


def test_default_volume_provider_builds_once_then_reuses_previous_scene(monkeypatch):
    monkeypatch.setattr(yir, "_load_sample_source", lambda: ("fake-ds", "fake-data-source"))
    rc = _FakeRenderContext()

    first = yir._default_volume_provider(yir.VolumeSetupContext(
        render_context=rc, dataset=None, custom_params={}, frame=None,
        previous_scene=None, previous_ds=None,
    ))
    assert first.changed is True
    assert first.ds == "fake-ds"
    assert len(rc.add_scene_calls) == 1  # real work happened

    second = yir._default_volume_provider(yir.VolumeSetupContext(
        render_context=rc, dataset=None, custom_params={}, frame=None,
        previous_scene=first.scene, previous_ds=first.ds,
    ))
    assert second.changed is False
    assert second.scene == first.scene
    assert len(rc.add_scene_calls) == 1  # NOT called again -- reused, not rebuilt


def test_get_volume_scene_uses_default_provider_when_custom_params_unset(monkeypatch, _clean_yir_state):
    fake_rc = _FakeRenderContext()
    monkeypatch.setattr(yir, "_get_render_context", lambda backend, w, h: fake_rc)
    monkeypatch.setattr(yir, "_load_sample_source", lambda: ("ds-obj", "data-source-obj"))

    rc, scene, ds, changed = yir._get_volume_scene(_request_with(custom_params={}), render_size=64)

    assert rc is fake_rc
    assert ds == "ds-obj"
    assert changed is True
    assert len(fake_rc.add_scene_calls) == 1


def test_get_volume_scene_dispatches_to_named_provider(monkeypatch, _clean_yir_state):
    calls = []

    def my_provider(ctx):
        calls.append(ctx)
        return yir.VolumeSetupResult(scene="my-scene", ds="my-ds", changed=True)

    fake_module = types.ModuleType("_test_yir_provider_mod_3")
    fake_module.my_provider = my_provider
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_3", fake_module)
    monkeypatch.setattr(yir, "_get_render_context", lambda backend, w, h: _FakeRenderContext())

    request = _request_with(custom_params={"yt_idv_volume_provider": "_test_yir_provider_mod_3:my_provider"})
    rc, scene, ds, changed = yir._get_volume_scene(request, render_size=64)

    assert (scene, ds, changed) == ("my-scene", "my-ds", True)
    assert len(calls) == 1
    assert calls[0].previous_scene is None  # first call for this dataset_id
    assert calls[0].custom_params == request.volume.custom_params
    assert calls[0].dataset is request.volume.dataset


def test_get_volume_scene_passes_previous_scene_through_on_unchanged_context(monkeypatch, _clean_yir_state):
    seen_previous_scenes = []

    def my_provider(ctx):
        seen_previous_scenes.append(ctx.previous_scene)
        if ctx.previous_scene is not None:
            return yir.VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)
        return yir.VolumeSetupResult(scene="scene-1", ds="ds-1", changed=True)

    fake_module = types.ModuleType("_test_yir_provider_mod_4")
    fake_module.my_provider = my_provider
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_4", fake_module)
    monkeypatch.setattr(yir, "_get_render_context", lambda backend, w, h: _FakeRenderContext())

    request = _request_with(custom_params={"yt_idv_volume_provider": "_test_yir_provider_mod_4:my_provider"})
    yir._get_volume_scene(request, render_size=64)
    yir._get_volume_scene(request, render_size=64)  # same object, same render_size

    assert seen_previous_scenes == [None, "scene-1"]


def test_get_volume_scene_invalidates_previous_scene_when_render_size_changes(monkeypatch, _clean_yir_state):
    # A resolution change means a DIFFERENT render context (yt_idv contexts
    # are fixed-size at creation) -- any scene built against the old one is
    # unusable regardless of whether the provider's own dataset state is
    # still perfectly fine, so previous_scene must come back None even
    # though this is still the same dataset_id as the first call.
    seen_previous_scenes = []

    def my_provider(ctx):
        seen_previous_scenes.append(ctx.previous_scene)
        return yir.VolumeSetupResult(scene="scene-x", ds="ds-x", changed=True)

    fake_module = types.ModuleType("_test_yir_provider_mod_5")
    fake_module.my_provider = my_provider
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_5", fake_module)
    monkeypatch.setattr(yir, "_get_render_context", lambda backend, w, h: _FakeRenderContext())

    request = _request_with(custom_params={"yt_idv_volume_provider": "_test_yir_provider_mod_5:my_provider"})
    yir._get_volume_scene(request, render_size=64)
    yir._get_volume_scene(request, render_size=128)

    assert seen_previous_scenes == [None, None]


def test_render_signature_changed_true_on_first_call(_clean_yir_state):
    assert yir._render_signature_changed("obj", _request_with(), render_size=4, volume_changed=False) is True


def test_render_signature_changed_false_when_nothing_changed(_clean_yir_state):
    request = _request_with()
    yir._render_signature_changed("obj", request, render_size=4, volume_changed=False)  # records the baseline
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=False) is False


def test_render_signature_changed_true_when_camera_moves(_clean_yir_state):
    request = _request_with()
    yir._render_signature_changed("obj", request, render_size=4, volume_changed=False)
    moved = _request_with(camera_overrides={
        "matrix_world": [[1, 0, 0, 9], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    })
    assert yir._render_signature_changed("obj", moved, render_size=4, volume_changed=False) is True


def test_render_signature_changed_true_when_depth_buffer_content_differs(_clean_yir_state):
    # Same camera, different depth CONTENT -- e.g. an unrelated occluder
    # moved elsewhere in the Cycles scene -- must still count as "changed",
    # not just camera motion (see _render_signature's own docstring).
    request = _request_with()
    yir._render_signature_changed("obj", request, render_size=4, volume_changed=False)
    request.depth.data[0, 0] = 1.0
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=False) is True


def test_render_signature_changed_forced_true_when_volume_provider_reports_changed(_clean_yir_state):
    # Identical signature both times, but volume_changed=True on the second
    # call must still force a re-render -- the provider's own data changing
    # is independent of (and not visible in) camera/depth/params.
    request = _request_with()
    yir._render_signature_changed("obj", request, render_size=4, volume_changed=False)
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=True) is True


class _FakeYtIdvCamera:
    def __init__(self):
        self.update_calls = []
        self.update_matrices_calls = 0

    def update(self, **kwargs):
        self.update_calls.append(kwargs)

    def _update_matrices(self):
        self.update_matrices_calls += 1


class _FakeYtIdvScene:
    def __init__(self):
        self.components = []
        self.camera = _FakeYtIdvCamera()


def test_render_with_yt_idv_skips_rc_run_when_nothing_changed_but_rerenders_on_camera_move(monkeypatch, _clean_yir_state):
    render_size = 4
    fake_scene = _FakeYtIdvScene()
    call_count = {"n": 0}

    class _FakeRc:
        def __init__(self):
            self.run_calls = 0

        def run(self):
            self.run_calls += 1
            # A distinct image each real render, so accidentally-identical
            # pixel data can't hide a real vs. cached-result mixup.
            return np.full((render_size, render_size, 4), fill_value=float(self.run_calls), dtype=np.float32)

    fake_rc = _FakeRc()

    def fake_get_volume_scene(request, rsize):
        call_count["n"] += 1
        # Only the very first call reports real data-side change; every
        # later call says "nothing changed on the volume side" -- so any
        # re-render that DOES happen after that must be because
        # _render_signature_changed's own camera/depth signal fired, not
        # because the (fake) provider asked for one.
        return fake_rc, fake_scene, _FakeDataset(), call_count["n"] == 1

    monkeypatch.setattr(yir, "_get_volume_scene", fake_get_volume_scene)
    monkeypatch.setattr(yir, "_cycles_depth_to_yt_idv_window_depth", lambda request, camera: np.zeros((render_size, render_size), dtype=np.float32))
    monkeypatch.setattr(yir, "set_external_depth_clip", lambda scene, depth: None)

    configure_calls = []
    configure = lambda component, custom_params: configure_calls.append(component)

    request = _request_with(render_size=render_size)
    result1 = yir._render_with_yt_idv(request, configure)
    assert fake_rc.run_calls == 1

    result2 = yir._render_with_yt_idv(request, configure)  # identical request
    assert fake_rc.run_calls == 1  # skipped -- no second GPU render
    assert result2 is result1  # the exact cached RenderResult, not a re-derived equal one

    moved = _request_with(render_size=render_size, camera_overrides={
        "matrix_world": [[1, 0, 0, 9], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    })
    result3 = yir._render_with_yt_idv(moved, configure)
    assert fake_rc.run_calls == 2  # camera moved -- real re-render happened
    assert result3 is not result1


def test_render_with_yt_idv_reasserts_rc_scene_when_objects_share_a_context(monkeypatch, _clean_yir_state):
    # Real yt_idv detail this pins down (see module docstring "Custom volume
    # setup"): a render_context only ever tracks ONE "current" scene
    # (`rc.scene`, set by `add_scene`) -- `rc.run()` renders THAT, not
    # whatever `scene` reference this module happens to be holding. Two
    # objects sharing one `rc` (the common case -- resolution is scene-wide)
    # must each force `rc.scene` back to their OWN scene before rendering,
    # even when neither object's own cached scene needed rebuilding.
    render_size = 4

    class _FakeRcWithScene:
        def __init__(self):
            self.scene = None
            self.rendered_scenes = []

        def run(self):
            self.rendered_scenes.append(self.scene)
            return np.zeros((render_size, render_size, 4), dtype=np.float32)

    shared_rc = _FakeRcWithScene()
    scene_a, scene_b = _FakeYtIdvScene(), _FakeYtIdvScene()

    def fake_get_volume_scene(request, rsize):
        scene = scene_a if request.volume.dataset.dataset_id == "a" else scene_b
        return shared_rc, scene, _FakeDataset(), True  # always "changed" -- force a real render every call

    monkeypatch.setattr(yir, "_get_volume_scene", fake_get_volume_scene)
    monkeypatch.setattr(yir, "_cycles_depth_to_yt_idv_window_depth", lambda request, camera: np.zeros((render_size, render_size), dtype=np.float32))
    monkeypatch.setattr(yir, "set_external_depth_clip", lambda scene, depth: None)
    configure = lambda component, custom_params: None

    yir._render_with_yt_idv(_request_with(dataset_id="a", render_size=render_size), configure)
    yir._render_with_yt_idv(_request_with(dataset_id="b", render_size=render_size), configure)
    yir._render_with_yt_idv(_request_with(dataset_id="a", render_size=render_size), configure)

    assert shared_rc.rendered_scenes == [scene_a, scene_b, scene_a]


def test_set_external_depth_clip_reuses_existing_texture_instead_of_leaking_gpu_memory(monkeypatch):
    # Real GPU-memory leak found running a full HD animation: a fresh
    # Texture2D(data=...) every frame (this function runs once per frame
    # from _render_with_yt_idv) allocates a NEW OpenGL texture via
    # glGenTextures every time (yt_idv.opengl_support.Texture has no
    # __del__/glDeleteTextures), so the previous frame's GPU texture is
    # never freed. Confirmed via an isolated repro (nothing but this
    # function + rc.run() in a loop): a rock-solid ~20MB/iteration linear
    # climb -- and this is exactly what crashed a real 250-frame HD render
    # with "System is out of GPU memory" at frame 183. Fix: reuse the same
    # Texture2D instance and reassign its `.data` trait instead of
    # constructing a new one every call -- verified here via a fake
    # Texture2D that just counts how many times it's actually constructed,
    # no real GL context needed.
    created = []

    class FakeTexture2D:
        def __init__(self, data):
            self.data = data
            created.append(self)

    fake_module = types.ModuleType("yt_idv.opengl_support")
    fake_module.Texture2D = FakeTexture2D
    monkeypatch.setitem(sys.modules, "yt_idv.opengl_support", fake_module)

    component = types.SimpleNamespace(use_external_depth_clip=False, external_depth_texture=None)
    scene = types.SimpleNamespace(components=[component])

    yir.set_external_depth_clip(scene, np.zeros((4, 4), dtype="f4"))
    assert len(created) == 1
    first_texture = component.external_depth_texture
    assert isinstance(first_texture, FakeTexture2D)

    yir.set_external_depth_clip(scene, np.full((4, 4), 5.0, dtype="f4"))
    assert len(created) == 1, (
        "second call constructed a NEW Texture2D instead of reusing the "
        "existing one -- this leaks one GPU texture's worth of VRAM per "
        "call (see this test's own comment for the real crash it caused)"
    )
    assert component.external_depth_texture is first_texture


def test_fix_yt_idv_axis_order_passes_through_already_correct_shape():
    image = np.zeros((100, 200, 4), dtype=np.float32)
    fixed = yir._fix_yt_idv_axis_order(image, expected_height=100, expected_width=200)
    assert fixed.shape == (100, 200, 4)
    assert fixed is image


def test_fix_yt_idv_axis_order_transposes_yt_idv_bug_1_shape():
    # Pins yt_idv bug #1 (see module docstring): a non-square render comes
    # back as (width, height, 4) instead of (height, width, 4).
    image = np.arange(100 * 200 * 4, dtype=np.float32).reshape(200, 100, 4)
    fixed = yir._fix_yt_idv_axis_order(image, expected_height=100, expected_width=200)
    assert fixed.shape == (100, 200, 4)
    np.testing.assert_array_equal(fixed, image.swapaxes(0, 1))


def test_fix_yt_idv_axis_order_raises_on_genuinely_unexpected_shape():
    image = np.zeros((7, 11, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="matching neither"):
        yir._fix_yt_idv_axis_order(image, expected_height=100, expected_width=200)


def test_to_row_order_top_left_is_passthrough():
    image = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert yir._to_row_order(image, "TOP_LEFT_ORIGIN") is image


def test_to_row_order_bottom_left_flips_rows():
    image = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    flipped = yir._to_row_order(image, "BOTTOM_LEFT_ORIGIN")
    np.testing.assert_array_equal(flipped, image[::-1, :, :])


def test_to_row_order_rejects_unknown_convention():
    image = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="unknown row_order"):
        yir._to_row_order(image, "DIAGONAL_ORIGIN")


def test_center_crop_recovers_requested_non_square_shape_from_a_square_render():
    # Pins the workaround for yt_idv bug 3 (see module docstring): render
    # square, then crop back down to the actually-requested shape.
    square = np.arange(10 * 10 * 4, dtype=np.float32).reshape(10, 10, 4)
    cropped = yir._center_crop(square, expected_height=4, expected_width=8)
    assert cropped.shape == (4, 8, 4)
    # Centered: 1 row of margin top/bottom (10-4=6, 6//2=3... use exact offsets)
    y0 = (10 - 4) // 2
    x0 = (10 - 8) // 2
    np.testing.assert_array_equal(cropped, square[y0:y0 + 4, x0:x0 + 8, :])


class _FakeTransferFunctionTexture:
    def __init__(self):
        self.data = None


class _FakeComponent:
    """Stand-in for a real yt_idv `block_rendering` component -- just
    enough surface for `_configure_max_intensity`/`_configure_transfer_function`
    to exercise without needing yt_idv importable.
    """

    def __init__(self):
        self.render_method = None
        self.tf_min = None
        self.tf_max = None
        self.tf_log = None
        self.transfer_function = _FakeTransferFunctionTexture()


def test_configure_max_intensity_only_sets_render_method():
    component = _FakeComponent()
    yir._configure_max_intensity(component, custom_params={})
    assert component.render_method == "max_intensity"
    assert component.tf_min is None  # untouched


def test_mip_alpha_constants_produce_a_transparent_to_opaque_ramp():
    # `_configure_max_intensity_alpha` itself needs yt/matplotlib to actually
    # build the colormap via `_get_mip_alpha_colormap_data`/`_build_mip_alpha_
    # colormap_data` -- rather than depend on whether those happen to be
    # importable in this venv, this instead pins the actual MIP_ALPHA_MAX/
    # RAMP_POWER constants that function feeds into the already-tested
    # `_threshold_fog_alpha_ramp`, so a change to either constant that breaks
    # the "transparent low, opaque high" guarantee shows up here without
    # needing yt at all.
    ramp = yir._threshold_fog_alpha_ramp(yir.MIP_ALPHA_NBINS, yir.MIP_ALPHA_MAX, yir.MIP_ALPHA_RAMP_POWER)
    assert ramp[0] == pytest.approx(0.0)
    assert ramp[-1] == pytest.approx(yir.MIP_ALPHA_MAX)
    assert np.all(np.diff(ramp) >= 0.0)


# --- Preferences-panel override mechanism (custom_params -> _configure_*) --
#
# The actual `_get_*_data`/`_build_*_data` functions these `_configure_*`
# functions call still need yt/matplotlib -- rather than depend on whether
# those happen to be importable in this venv, these tests monkeypatch that
# ONE call out with a spy, so what's actually under test is the "does
# custom_params correctly override the module constant, and does an EMPTY
# custom_params correctly fall back to it" logic, which is this add-on's
# own code, not yt_idv/matplotlib's.

def test_configure_transfer_function_empty_custom_params_uses_module_constants(monkeypatch):
    component = _FakeComponent()
    calls = []
    monkeypatch.setattr(yir, "_get_isocontour_transfer_function_data", lambda *a: (calls.append(a), np.zeros((256, 1, 4), dtype="u1"))[1])

    yir._configure_transfer_function(component, custom_params={})

    assert calls == [(yir.TF_VALUE_MIN, yir.TF_VALUE_MAX, yir.TF_N_LAYERS, yir.TF_COLORMAP)]
    assert component.tf_min == yir.TF_VALUE_MIN
    assert component.tf_max == yir.TF_VALUE_MAX


def test_configure_transfer_function_custom_params_override_module_constants(monkeypatch):
    component = _FakeComponent()
    calls = []
    monkeypatch.setattr(yir, "_get_isocontour_transfer_function_data", lambda *a: (calls.append(a), np.zeros((256, 1, 4), dtype="u1"))[1])

    yir._configure_transfer_function(component, custom_params={
        "yt_idv_tf_value_min": 5e-4, "yt_idv_tf_value_max": 0.5,
        "yt_idv_tf_n_layers": 9, "yt_idv_tf_colormap": "inferno",
    })

    assert calls == [(5e-4, 0.5, 9, "inferno")]
    assert component.tf_min == 5e-4
    assert component.tf_max == 0.5


def test_configure_threshold_fog_custom_params_override_module_constants(monkeypatch):
    component = _FakeComponent()
    calls = []
    monkeypatch.setattr(
        yir, "_get_threshold_fog_transfer_function_data",
        lambda *a, **k: (calls.append((a, k)), np.zeros((256, 1, 4), dtype="u1"))[1],
    )

    yir._configure_threshold_fog(component, custom_params={
        "yt_idv_fog_value_min": 1e-7, "yt_idv_fog_value_max": 0.05,
        "yt_idv_fog_colormap": "algae", "yt_idv_fog_alpha_max": 0.5, "yt_idv_fog_ramp_power": 2.0,
    })

    assert calls == [((1e-7, 0.05, "algae", 0.5, 2.0), {"alpha_curve": None, "color_rgba": None})]
    assert component.tf_min == 1e-7
    assert component.tf_max == 0.05


def test_configure_threshold_fog_passes_through_alpha_curve_and_color_rgba(monkeypatch):
    # Curve-driven alpha / image-driven color -- see yt_idv_renderer.py
    # module docstring's "Curve-driven alpha / image-driven color" section:
    # operators.py evaluates these into plain lists and passes them through
    # custom_params under these exact keys; _configure_threshold_fog just
    # needs to forward them unchanged.
    component = _FakeComponent()
    calls = []
    monkeypatch.setattr(
        yir, "_get_threshold_fog_transfer_function_data",
        lambda *a, **k: (calls.append((a, k)), np.zeros((256, 1, 4), dtype="u1"))[1],
    )
    alpha_curve = [0.0, 0.5, 1.0]
    color_rgba = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

    yir._configure_threshold_fog(component, custom_params={
        "yt_idv_alpha_curve": alpha_curve, "yt_idv_color_image_rgba": color_rgba,
    })

    assert calls == [(
        (yir.TF_FOG_VALUE_MIN, yir.TF_FOG_VALUE_MAX, yir.TF_FOG_COLORMAP, yir.TF_FOG_ALPHA_MAX, yir.TF_FOG_RAMP_POWER),
        {"alpha_curve": alpha_curve, "color_rgba": color_rgba},
    )]


def test_configure_max_intensity_alpha_custom_params_override_module_constants(monkeypatch):
    component = _FakeComponent()
    component.colormap = _FakeTransferFunctionTexture()  # just needs a settable .data
    calls = []
    monkeypatch.setattr(
        yir, "_get_mip_alpha_colormap_data",
        lambda *a, **k: (calls.append((a, k)), np.zeros((256, 4), dtype="f4"))[1],
    )

    yir._configure_max_intensity_alpha(component, custom_params={
        "yt_idv_mip_alpha_colormap": "arbre", "yt_idv_mip_alpha_max": 0.9, "yt_idv_mip_alpha_ramp_power": 0.5,
    })

    assert calls == [(("arbre", 0.9, 0.5), {"alpha_curve": None, "color_rgba": None})]
    assert component.render_method == "max_intensity_alpha"


def test_configure_max_intensity_alpha_passes_through_alpha_curve_and_color_rgba(monkeypatch):
    component = _FakeComponent()
    component.colormap = _FakeTransferFunctionTexture()
    calls = []
    monkeypatch.setattr(
        yir, "_get_mip_alpha_colormap_data",
        lambda *a, **k: (calls.append((a, k)), np.zeros((256, 4), dtype="f4"))[1],
    )
    alpha_curve = [0.0, 1.0]
    color_rgba = [[0.1, 0.2, 0.3, 1.0]]

    yir._configure_max_intensity_alpha(component, custom_params={
        "yt_idv_alpha_curve": alpha_curve, "yt_idv_color_image_rgba": color_rgba,
    })

    assert calls == [(
        (yir.MIP_ALPHA_COLORMAP, yir.MIP_ALPHA_MAX, yir.MIP_ALPHA_RAMP_POWER),
        {"alpha_curve": alpha_curve, "color_rgba": color_rgba},
    )]


def test_apply_transfer_function_settings_sets_tf_fields_not_render_method():
    # The pure-Python half of _configure_transfer_function -- see that
    # function's docstring for why it's split out (the other half needs
    # `yt` importable to actually build tf_data, this doesn't).
    component = _FakeComponent()
    fake_tf_data = np.zeros((256, 1, 4), dtype="u1")

    yir._apply_transfer_function_settings(component, tf_min=1e-5, tf_max=3e-3, tf_data=fake_tf_data)

    assert component.tf_min == 1e-5
    assert component.tf_max == 3e-3
    assert component.tf_log is True
    assert component.transfer_function.data is fake_tf_data
    assert component.render_method is None  # not this function's job


def test_render_method_mode_switch_is_idempotent_on_the_same_component():
    # Pins that _get_context's caching (real context reused across calls) is
    # safe to combine with EITHER renderer mode -- each call reconfigures the
    # component fresh rather than relying on whatever was set last time.
    component = _FakeComponent()
    yir._apply_transfer_function_settings(component, tf_min=1e-5, tf_max=3e-3, tf_data=np.zeros((256, 1, 4), dtype="u1"))
    component.render_method = "transfer_function"
    assert component.render_method == "transfer_function"

    yir._configure_max_intensity(component, custom_params={})
    assert component.render_method == "max_intensity"
    # tf_min/tf_max/tf_log/transfer_function.data are left as whatever
    # transfer_function mode set them to -- harmless, since max_intensity's
    # own shader combo doesn't read them at all (see shaders/shaderlist.yaml).
    assert component.tf_min == 1e-5


def test_threshold_fog_alpha_ramp_starts_at_zero_and_ends_at_alpha_max():
    ramp = yir._threshold_fog_alpha_ramp(nbins=256, alpha_max=0.85, ramp_power=2.0)
    assert ramp.shape == (256,)
    assert ramp[0] == pytest.approx(0.0)
    assert ramp[-1] == pytest.approx(0.85)


def test_threshold_fog_alpha_ramp_is_monotonically_nondecreasing():
    ramp = yir._threshold_fog_alpha_ramp(nbins=256, alpha_max=1.0, ramp_power=2.0)
    assert np.all(np.diff(ramp) >= 0.0)


def test_threshold_fog_alpha_ramp_power_above_one_stays_below_the_linear_ramp():
    # power > 1.0 is documented as "ramps up slowly at first" -- pin that
    # down numerically, not just by description: at the ramp's own midpoint,
    # a power=2.0 ramp must sit strictly below what a plain linear
    # (power=1.0) ramp would give at that same point.
    linear = yir._threshold_fog_alpha_ramp(nbins=101, alpha_max=1.0, ramp_power=1.0)
    slow = yir._threshold_fog_alpha_ramp(nbins=101, alpha_max=1.0, ramp_power=2.0)
    midpoint = 50
    assert slow[midpoint] < linear[midpoint]


def test_resample_1d_returns_input_unchanged_when_already_the_right_length():
    values = np.array([0.1, 0.4, 0.9])
    assert np.array_equal(yir._resample_1d(values, nbins=3), values)


def test_resample_1d_linearly_interpolates_to_a_different_length():
    # 3 samples spanning [0, 1] -> 5 evenly-spaced positions should hit the
    # exact original values at the endpoints and midpoint, with linear
    # interpolation in between.
    resampled = yir._resample_1d([0.0, 0.5, 1.0], nbins=5)
    assert resampled == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_resample_rgb_interpolates_each_channel_independently():
    rgb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # red -> green, 2 samples
    resampled = yir._resample_rgb(rgb, nbins=3)
    assert resampled.shape == (3, 3)
    assert resampled[0] == pytest.approx([1.0, 0.0, 0.0])
    assert resampled[1] == pytest.approx([0.5, 0.5, 0.0])
    assert resampled[2] == pytest.approx([0.0, 1.0, 0.0])


def test_build_threshold_fog_transfer_function_data_with_curve_and_color_overrides():
    # Both `alpha_curve`/`color_rgba` given -- takes neither the named-
    # colormap nor `yt`/matplotlib code path at all, so this is exercisable
    # in this bpy-only venv (see module docstring's "NOT wired into this
    # repo's own pytest suite" section on why `yt`/matplotlib aren't
    # installed here).
    alpha_curve = [0.0, 1.0]  # a plain linear ramp 0 -> 1
    color_rgba = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]  # red -> blue

    data = yir._build_threshold_fog_transfer_function_data(
        value_min=1e-9, value_max=1e-2, colormap="unused", alpha_max=0.5, ramp_power=1.0,
        nbins=4, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )

    assert data.shape == (4, 1, 4)
    assert data.dtype == np.dtype("u1")
    # alpha: curve resampled to 4 bins (0, 1/3, 2/3, 1) * alpha_max=0.5
    assert list(data[:, 0, 3]) == pytest.approx([0, 42, 85, 127], abs=1)
    # color: red at bin 0, blue at bin 3 (resampled color_rgba), alpha
    # column from color_rgba is ignored (it's all 1.0 there, alpha above
    # came from alpha_curve*alpha_max instead, which is NOT 255 anywhere)
    assert int(data[0, 0, 0]) == 255 and int(data[0, 0, 2]) == 0
    assert int(data[-1, 0, 0]) == 0 and int(data[-1, 0, 2]) == 255


def test_build_threshold_fog_transfer_function_data_alpha_curve_only_keeps_named_color_path(monkeypatch):
    # alpha_curve given, color_rgba NOT -- still needs the colormap path
    # (`import yt`/`import matplotlib`), confirming the two overrides are
    # independent switches, by checking the failure mode itself (ImportError
    # from the colormap path, not e.g. a TypeError from mixing up the two).
    # Forces the import to fail via sys.modules rather than relying on
    # yt/matplotlib being genuinely absent from this venv -- they're an
    # *optional* `ytidv` dependency group on top of this suite's own `dev`
    # group (see README "yt_idv integration"), so whether they're actually
    # importable here depends on what else has been `uv sync`'d into the
    # same venv, not on anything this test controls.
    monkeypatch.setitem(sys.modules, "yt", None)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError):
        yir._build_threshold_fog_transfer_function_data(
            value_min=1e-9, value_max=1e-2, colormap="dusk", alpha_max=0.5, ramp_power=1.0,
            alpha_curve=[0.0, 1.0],
        )


def test_get_threshold_fog_transfer_function_data_caches_by_curve_and_color_content():
    alpha_curve = [0.0, 1.0]
    color_rgba = [[1.0, 0.0, 0.0, 1.0]]
    first = yir._get_threshold_fog_transfer_function_data(
        1e-9, 1e-2, "unused", 0.5, 1.0, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )
    second = yir._get_threshold_fog_transfer_function_data(
        1e-9, 1e-2, "unused", 0.5, 1.0, alpha_curve=list(alpha_curve), color_rgba=[list(row) for row in color_rgba],
    )
    assert first is second  # same content, different list objects -- still a cache hit
    different = yir._get_threshold_fog_transfer_function_data(
        1e-9, 1e-2, "unused", 0.5, 1.0, alpha_curve=[0.0, 0.5, 1.0], color_rgba=color_rgba,
    )
    assert different is not first


def test_build_mip_alpha_colormap_data_with_curve_and_color_overrides():
    alpha_curve = [0.0, 1.0]
    color_rgba = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

    data = yir._build_mip_alpha_colormap_data(
        colormap="unused", alpha_max=0.5, ramp_power=1.0, nbins=4,
        alpha_curve=alpha_curve, color_rgba=color_rgba,
    )

    assert data.shape == (4, 4)
    assert data.dtype == np.dtype("f4")
    assert list(data[:, 3]) == pytest.approx([0.0, 1 / 6, 1 / 3, 0.5], abs=1e-5)
    assert data[0, 0] == pytest.approx(1.0) and data[0, 2] == pytest.approx(0.0)
    assert data[-1, 0] == pytest.approx(0.0) and data[-1, 2] == pytest.approx(1.0)


def test_apply_transfer_function_settings_works_with_fog_bounds_and_a_fake_ramp():
    # `_configure_threshold_fog` itself isn't called here -- it needs `yt`/
    # matplotlib (not installed in this bpy-only test venv, see module
    # docstring's "NOT wired into this repo's own pytest suite" section) to
    # actually build the fog LUT via `_build_threshold_fog_transfer_
    # function_data`. This instead exercises the exact same pure-Python
    # `_apply_transfer_function_settings` call `_configure_threshold_fog`
    # makes, with TF_FOG_VALUE_MIN/MAX and a hand-built LUT whose bin 0 is
    # fully transparent -- pinning down the "fully transparent below
    # threshold" guarantee without needing yt.
    component = _FakeComponent()
    fake_fog_data = np.zeros((256, 1, 4), dtype="u1")
    fake_fog_data[:, 0, 3] = np.linspace(0, 255, 256).astype("u1")  # alpha ramps 0 -> 255

    yir._apply_transfer_function_settings(
        component, tf_min=yir.TF_FOG_VALUE_MIN, tf_max=yir.TF_FOG_VALUE_MAX, tf_data=fake_fog_data,
    )

    assert component.tf_min == yir.TF_FOG_VALUE_MIN
    assert component.tf_max == yir.TF_FOG_VALUE_MAX
    assert component.tf_log is True
    assert component.transfer_function.data[0, 0, 3] == 0
    assert component.transfer_function.data[-1, 0, 3] == 255


def test_pad_to_square_centers_the_array_and_fills_the_margin():
    depth = np.zeros((4, 8), dtype="f4")
    padded = yir._pad_to_square(depth, render_size=10, fill_value=1.0)
    assert padded.shape == (10, 10)
    y0, x0 = (10 - 4) // 2, (10 - 8) // 2
    np.testing.assert_array_equal(padded[y0 : y0 + 4, x0 : x0 + 8], depth)
    assert padded[0, 0] == 1.0
    assert padded[-1, -1] == 1.0


# --- _cycles_depth_to_yt_idv_window_depth ------------------------------------
#
# Uses a REAL bpy camera (real projection_matrix, not hand-rolled) --
# reusing the exact fixture pattern tests/test_sample_renderers.py already
# established, plus the SAME box (2x2x2 at world (0, 0, 5)) that project
# already uses for its own depth-clip tests, so this is checked against
# real, working Blender camera math, not just internally-consistent numbers.

RES_X, RES_Y = 33, 25  # odd dims -> a pixel sits exactly on the optical axis
CENTER_COL, CENTER_ROW = 16, 12


@pytest.fixture
def persp_camera_params():
    """Real PERSP CameraParams (real projection_matrix from bpy), camera at
    world (0, 0, 10), identity rotation -> looks straight down -Z.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    cam_data = bpy.data.cameras.new("C")
    cam_data.type = "PERSP"
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("C", cam_data)
    cam_obj.location = (0.0, 0.0, 10.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    bpy.context.view_layer.update()
    return operators.build_camera_params(scene, cam_obj)


def _boxed_volume_at_5():
    """2x2x2 box centered at world (0, 0, 5) -- along the camera's
    straight-down-the-axis ray (camera at world z=10), the box spans
    PLANAR depth [4, 6]: near face at world z=6 -> depth 4, far face at
    world z=4 -> depth 6, center at world z=5 -> depth 5.
    """
    return _identity_volume(
        bounding_box_local=_cube_bbox(),
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]],
    )


def _uniform_depth_buffer(fill_value, no_hit_mask=None, row_order="BOTTOM_LEFT_ORIGIN"):
    data = np.full((RES_Y, RES_X), fill_value, dtype=np.float32)
    if no_hit_mask is not None:
        data[no_hit_mask] = 1e10
    return DepthBuffer(
        data=data, depth_convention="PLANAR", no_hit_value=1e10,
        row_order=row_order, units="meters",
    )


class _IdentityYtIdvCamera:
    """Stand-in for yt_idv's own (already-positioned) camera -- an identity
    projection*view means the sampled window_depth directly reflects the
    reconstructed point's own unitary Z coordinate (`0.5*z + 0.5`), which
    makes the expected values in these tests trivial to predict by hand
    rather than needing to replicate yt_idv's actual perspective matrix.
    """

    projection_matrix = np.eye(4)
    view_matrix = np.eye(4)


def test_cycles_depth_to_yt_idv_window_depth_is_uniform_across_a_planar_hit(persp_camera_params):
    # PLANAR depth is view-space Z -- constant across the whole
    # perpendicular-to-view plane regardless of x/y offset (the
    # empirically-confirmed convention this whole add-on relies on, see
    # README "Z-pass verification findings"). A uniform depth=5.0 fill
    # should therefore reconstruct to world Z=5 EVERYWHERE, which is
    # unitary Z=0.5 for this box (local Z=(5-5)=0, bbox -1..1 -> unitary
    # 0.5) -- so window_depth should be uniformly 0.5*0.5+0.5=0.75 across
    # the ENTIRE array, not just the center pixel.
    request = RenderRequest(
        camera=persp_camera_params,
        depth=_uniform_depth_buffer(5.0),
        volume=_boxed_volume_at_5(),
        frame=FrameParams(frame_number=1, scene_time_seconds=0.0),
    )

    window_depth = yir._cycles_depth_to_yt_idv_window_depth(request, _IdentityYtIdvCamera())

    assert window_depth.shape == (RES_Y, RES_X)
    np.testing.assert_allclose(window_depth, 0.75, atol=1e-5)


def test_cycles_depth_to_yt_idv_window_depth_masks_no_hit_pixels_to_far(persp_camera_params):
    # TOP_LEFT_ORIGIN here specifically, so array position [0, 0] maps
    # directly (no row-order flip to account for) -- keeps the assertion
    # simple; the flip itself is separately covered by the uniform-fill
    # test above (flip-invariant, since every value is identical) and by
    # test_to_row_order_* elsewhere in this file.
    no_hit_mask = np.zeros((RES_Y, RES_X), dtype=bool)
    no_hit_mask[0, 0] = True  # one corner pixel is background/sky
    request = RenderRequest(
        camera=persp_camera_params,
        depth=_uniform_depth_buffer(5.0, no_hit_mask=no_hit_mask, row_order="TOP_LEFT_ORIGIN"),
        volume=_boxed_volume_at_5(),
        frame=FrameParams(frame_number=1, scene_time_seconds=0.0),
    )
    window_depth = yir._cycles_depth_to_yt_idv_window_depth(request, _IdentityYtIdvCamera())
    assert window_depth[0, 0] == pytest.approx(1.0)  # no-hit -> no clip
    assert window_depth[CENTER_ROW, CENTER_COL] == pytest.approx(0.75)  # everywhere else, unaffected
