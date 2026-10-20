"""Unit tests for yt_idv_renderer.py's pure-Python helpers.

These don't need yt/yt_idv; the real render path is exercised by the scripts in
scripts/yt_idv_exploration/blender_scene/. yt/matplotlib may or may not be
installed (optional `ytidv` group), so tests needing them absent force it via
sys.modules.
"""

import math
import sys
import types

import bpy
import numpy as np
import pytest

from yt_volumes import change_tracking, operators
from yt_volumes import yt_idv_renderer as yir
from yt_volumes.external_render import (
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
    """Stand-in for a yt Dataset with plain-array domain_center/domain_width."""

    domain_center = np.array([0.5, 0.5, 0.5])
    domain_width = np.array([1.0, 1.0, 1.0])


class _FakeYTArray(np.ndarray):
    """Stand-in for a YTArray whose in_units("unitary") actually converts."""

    def in_units(self, name):
        assert name == "unitary"
        # Domain is 2 code_length wide.
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
    assert yir._approx_vertical_fov_deg(camera, 2.5) == pytest.approx(expected)
    assert yir._approx_vertical_fov_deg(camera, 0.5) == pytest.approx(expected)


def _square_canvas_spans(camera, frame_aspect):
    """tan of half the square canvas's vertical FOV."""
    return math.tan(math.radians(yir._approx_vertical_fov_deg(camera, frame_aspect, 1.0)) / 2)


@pytest.mark.parametrize("fit, frame_aspect, expected_half_tan", [
    # landscape: the square spans the frame's width
    ("AUTO", 16 / 9, 18.0 / 50.0),
    ("HORIZONTAL", 16 / 9, 18.0 / 50.0),
    ("VERTICAL", 16 / 9, 12.0 / 50.0 * 16 / 9),
    # portrait: the square spans the frame's height; AUTO puts sensor_width there
    ("AUTO", 9 / 16, 18.0 / 50.0),
    ("HORIZONTAL", 9 / 16, 18.0 / 50.0 * 16 / 9),
    ("VERTICAL", 9 / 16, 12.0 / 50.0),
])
def test_square_canvas_fov_covers_the_frames_longer_side(fit, frame_aspect, expected_half_tan):
    # Regression: VERTICAL fit used the frame's vertical FOV, so the center
    # crop of a landscape frame came out zoomed in by the aspect ratio.
    camera = _persp_camera(sensor_fit=fit, sensor_width_mm=36.0, sensor_height_mm=24.0, lens_mm=50.0)
    assert _square_canvas_spans(camera, frame_aspect) == pytest.approx(expected_half_tan)


def _cube_bbox(half_extent=1.0):
    return [
        [x, y, z]
        for x in (-half_extent, half_extent)
        for y in (-half_extent, half_extent)
        for z in (-half_extent, half_extent)
    ]


def _identity_volume(bounding_box_local=None, matrix_world=None):
    """A flagged object; defaults to a 2x2x2 cube at the origin."""
    version = VersionInfo(hash="x", changed=True)
    dataset = DatasetParams(
        dataset_id="obj", dataset_kind="", dataset_path=None,
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
    # Camera at world (0, 0, 5) looking down -Z; 2x2x2 cube at the origin.
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    volume = _identity_volume()
    ds = _FakeDataset()  # domain_center == (0.5, 0.5, 0.5), matches the object's own unitary center

    kwargs = yir._registered_camera_kwargs(camera, volume, ds, width=200, height=100)

    # position = (cam_pos_local - bbox_min) / bbox_extent = ((0,0,5)-(-1,-1,-1))/(2,2,2)
    np.testing.assert_allclose(kwargs["position"], [0.5, 0.5, 3.0], atol=1e-6)
    np.testing.assert_allclose(kwargs["focus"], [0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(kwargs["up"], [0, 1, 0], atol=1e-6)
    assert kwargs["aspect_ratio"] == pytest.approx(2.0)


def test_registered_camera_kwargs_position_scales_with_object_size():
    # Half-size object: same world distance maps to a larger unitary distance.
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    volume = _identity_volume(bounding_box_local=_cube_bbox(half_extent=0.5))
    ds = _FakeDataset()

    kwargs = yir._registered_camera_kwargs(camera, volume, ds, width=100, height=100)

    # ((0,0,5) - (-0.5,-0.5,-0.5)) / (1,1,1) = (0.5, 0.5, 5.5)
    np.testing.assert_allclose(kwargs["position"], [0.5, 0.5, 5.5], atol=1e-6)


def test_near_far_planes_bounds_the_unit_cube_with_margin():
    position = np.array([2.0, 2.0, 2.0])
    planes = yir._near_far_planes(position, margin=0.1)
    nearest = np.linalg.norm(np.array([1.0, 1.0, 1.0]) - position)
    farthest = np.linalg.norm(np.array([0.0, 0.0, 0.0]) - position)
    assert planes["near_plane"] == pytest.approx(nearest - 0.1)
    assert planes["far_plane"] == pytest.approx(farthest + 0.1)


def test_near_far_planes_floors_near_plane_when_camera_is_inside_the_cube():
    # yt_idv cameras reject non-positive near planes.
    position = np.array([0.5, 0.5, 0.5])
    planes = yir._near_far_planes(position, margin=1.0)
    assert planes["near_plane"] == pytest.approx(1e-5)


def test_near_far_planes_floors_near_plane_inside_the_cube_with_default_margin():
    # Regression: the nearest corner is ~0.87 away, which used to clip everything closer.
    planes = yir._near_far_planes(np.array([0.5, 0.5, 0.5]))
    assert planes["near_plane"] == pytest.approx(1e-5)


def test_near_far_planes_measure_to_the_nearest_face_not_corner():
    planes = yir._near_far_planes(np.array([0.5, 0.5, 3.0]), margin=0.1)
    assert planes["near_plane"] == pytest.approx(1.9)


def test_near_far_planes_ratio_does_not_blow_up_with_camera_distance():
    # Regression: the old formula had a fixed 400:1 far/near ratio at any distance.
    position = np.array([100.0, 100.0, 100.0])
    planes = yir._near_far_planes(position, margin=0.1)
    ratio = planes["far_plane"] / planes["near_plane"]
    assert ratio < 2.0


class _FakeBlockCollection:
    """Stand-in for a yt_idv BlockCollection with #287's scale transform:
    `scale = True` maps the data source's bbox onto [0, 1] along its longest axis.
    """

    def __init__(self, left_edges, right_edges, bbox=None, has_scaling=True):
        self._unscaled = (np.asarray(left_edges, dtype="f8"), np.asarray(right_edges, dtype="f8"))
        self.bbox = bbox
        self._scale = False
        self.applied_scale_offset = (0.0, 0.0, 0.0)
        self.applied_scale_ratio = 1.0
        if has_scaling:
            self.apply_scale = self._apply_scale
        self.vertex_array = types.SimpleNamespace(attributes=[
            types.SimpleNamespace(name="in_left_edge", data=None),
            types.SimpleNamespace(name="in_right_edge", data=None),
        ])
        self._set_scale(np.zeros(3), 1.0)

    def _set_scale(self, offset, ratio):
        self.applied_scale_offset = tuple(offset)
        self.applied_scale_ratio = ratio
        for attribute, edges in zip(self.vertex_array.attributes, self._unscaled):
            attribute.data = ((edges - offset) / ratio).astype("f4")

    def _apply_scale(self, offset, ratio):
        current = self.applied_scale_ratio
        self._set_scale(np.asarray(self.applied_scale_offset) + current * np.asarray(offset), current * ratio)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        if value:
            low, high = (np.asarray(e, dtype="f8") for e in self.bbox)
            self._set_scale(low, (high - low).max())
        else:
            self._set_scale(np.zeros(3), 1.0)


def _block_scene(*collections):
    components = [types.SimpleNamespace(name="block_rendering", data=c) for c in collections]
    components.append(types.SimpleNamespace(name="block_outline", data=None))  # ignored
    return types.SimpleNamespace(components=components)


def _subset_collection(**kwargs):
    # A region x in [0.4, 0.5], y in [0.45, 0.5], z in [0.45, 0.475] (unitary),
    # inside coarser blocks that extend past it.
    return _FakeBlockCollection(
        left_edges=[[0.3, 0.4, 0.4], [0.45, 0.4, 0.4]],
        right_edges=[[0.45, 0.6, 0.6], [0.6, 0.6, 0.6]],
        bbox=([0.4, 0.45, 0.45], [0.5, 0.5, 0.475]),
        **kwargs,
    )


def _full_domain_collection(**kwargs):
    return _FakeBlockCollection(
        left_edges=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        right_edges=[[0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
        bbox=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        **kwargs,
    )


def test_scene_model_bounds_spans_all_block_collections():
    scene = _block_scene(_subset_collection(), _full_domain_collection())
    low, high = yir._scene_model_bounds(scene)
    np.testing.assert_allclose(low, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(high, [1.0, 1.0, 1.0])


def test_scene_model_bounds_clips_blocks_to_data_source_bbox():
    low, high = yir._scene_model_bounds(_block_scene(_subset_collection()))
    np.testing.assert_allclose(low, [0.4, 0.45, 0.45], atol=1e-6)
    np.testing.assert_allclose(high, [0.5, 0.5, 0.475], atol=1e-6)


def test_scene_model_bounds_uses_blocks_without_bbox():
    collection = _subset_collection()
    collection.bbox = None
    low, high = yir._scene_model_bounds(_block_scene(collection))
    np.testing.assert_allclose(low, [0.3, 0.4, 0.4], atol=1e-6)
    np.testing.assert_allclose(high, [0.6, 0.6, 0.6], atol=1e-6)


def test_scene_model_bounds_is_none_without_blocks():
    assert yir._scene_model_bounds(types.SimpleNamespace(components=[])) is None


def test_scale_scene_to_subset_fills_unit_cube_along_longest_axis():
    collection = _subset_collection()
    yir.scale_scene_to_subset(_block_scene(collection))
    assert collection.scale is True
    low, high = yir._scene_model_bounds(_block_scene(collection))
    np.testing.assert_allclose(low, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(high, [1.0, 0.5, 0.25], atol=1e-6)  # aspect preserved


def test_scale_scene_to_subset_leaves_full_domain_unscaled():
    collection = _full_domain_collection()
    yir.scale_scene_to_subset(_block_scene(collection))
    assert collection.scale is False


def test_scale_scene_to_subset_is_idempotent():
    collection = _subset_collection()
    scene = _block_scene(collection)
    yir.scale_scene_to_subset(scene)
    yir.scale_scene_to_subset(scene)
    np.testing.assert_allclose(yir._scene_model_bounds(scene)[1], [1.0, 0.5, 0.25], atol=1e-6)


def test_scale_scene_to_subset_raises_without_yt_idv_scene_transforms():
    with pytest.raises(RuntimeError, match="#287"):
        yir.scale_scene_to_subset(_block_scene(_subset_collection(has_scaling=False)))


def test_near_far_planes_bracket_model_bounds_when_given():
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    volume = _identity_volume()
    unit = yir._registered_camera_kwargs(camera, volume, _FakeDataset(), 100, 100)
    bounded = yir._registered_camera_kwargs(
        camera, volume, _FakeDataset(), 100, 100,
        model_bounds=(np.zeros(3), np.array([1.0, 0.5, 0.25])),
    )
    position = unit["position"].astype(np.float64)
    corners = yir._box_corners(np.zeros(3), [1.0, 0.5, 0.25])
    dists = np.linalg.norm(corners - position, axis=1)
    gap = np.maximum(np.maximum(-position, position - [1.0, 0.5, 0.25]), 0.0)
    assert bounded["near_plane"] == pytest.approx(np.linalg.norm(gap) - 0.1)
    assert bounded["far_plane"] == pytest.approx(dists.max() + 0.1)
    assert bounded["near_plane"] > unit["near_plane"]  # z extent is only 0.25
    np.testing.assert_allclose(bounded["position"], unit["position"])  # registration unchanged


def test_near_far_planes_for_full_domain_bounds_match_unit_cube():
    position = np.array([2.0, 2.0, 2.0])
    corners = yir._box_corners(np.zeros(3), np.ones(3))
    assert yir._near_far_planes(position, corners=corners) == yir._near_far_planes(position)


def test_domain_array_unitary_converts_real_ytarray_like_objects():
    value = np.asarray([4.0, 4.0, 4.0]).view(_FakeYTArray)
    got = yir._domain_array_unitary(value)
    np.testing.assert_allclose(got, [2.0, 2.0, 2.0])


def test_domain_array_unitary_passes_through_plain_arrays():
    got = yir._domain_array_unitary((1.0, 2.0, 3.0))
    np.testing.assert_allclose(got, [1.0, 2.0, 3.0])


# --- Custom volume setup harness (yt_idv_volume_provider) -------------------
# Uses fake render contexts/scenes; no GL context needed.

@pytest.fixture
def _clean_yir_state(monkeypatch):
    """Reset yt_idv_renderer's module caches and change_tracking around each
    test; render sizes need no real framebuffers.
    """
    monkeypatch.setattr(yir, "_use_render_size", lambda scene, size: None)

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
    """Stand-in for a yt_idv render context that records add_scene calls."""

    def __init__(self):
        self.add_scene_calls = []

    def add_scene(self, data_source, field, **kwargs):
        self.add_scene_calls.append((data_source, field, kwargs))
        return types.SimpleNamespace(label=f"scene[{field}]#{len(self.add_scene_calls)}", components=[])


def _request_with(custom_params=None, dataset_id="obj", camera_overrides=None, render_size=4):
    """A small but complete RenderRequest, no yt/yt_idv/bpy needed."""
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


@pytest.fixture
def fake_yt_samples(monkeypatch, tmp_path):
    """A fake yt whose sample data dir is tmp_path; returns the names load_sample got."""
    loaded = []
    api = types.SimpleNamespace(_get_test_data_dir_path=lambda: tmp_path)
    fake_yt = types.SimpleNamespace(load_sample=lambda name, **kw: loaded.append(name) or "ds")
    monkeypatch.setitem(sys.modules, "yt", fake_yt)
    monkeypatch.setitem(sys.modules, "yt.sample_data", types.SimpleNamespace(api=api))
    monkeypatch.setitem(sys.modules, "yt.sample_data.api", api)
    return loaded


@pytest.fixture
def online_access():
    system = bpy.context.preferences.system
    saved = system.use_online_access
    yield lambda on: setattr(system, "use_online_access", on)
    system.use_online_access = saved


def test_load_sample_refuses_to_download_when_offline(fake_yt_samples, online_access):
    online_access(False)
    with pytest.raises(RuntimeError, match="--online"):
        yir.load_sample("IsolatedGalaxy")
    assert fake_yt_samples == []


def test_load_sample_uses_downloaded_data_when_offline(fake_yt_samples, online_access, tmp_path):
    online_access(False)
    (tmp_path / "IsolatedGalaxy").mkdir()
    assert yir.load_sample("IsolatedGalaxy/galaxy0030/galaxy0030") == "ds"
    assert fake_yt_samples == ["IsolatedGalaxy/galaxy0030/galaxy0030"]


def test_load_sample_downloads_when_online(fake_yt_samples, online_access):
    online_access(True)
    assert yir.load_sample("IsolatedGalaxy") == "ds"


def test_default_volume_provider_builds_once_then_reuses_previous_scene(monkeypatch):
    monkeypatch.setattr(yir, "_load_sample_source", lambda online_access=None: ("fake-ds", "fake-data-source"))
    rc = _FakeRenderContext()

    no_path = types.SimpleNamespace(dataset_path=None, dataset_kind="")
    first = yir._default_volume_provider(yir.VolumeSetupContext(
        render_context=rc, dataset=no_path, custom_params={}, frame=None,
        previous_scene=None, previous_ds=None,
    ))
    assert first.changed is True
    assert first.ds == "fake-ds"
    assert len(rc.add_scene_calls) == 1  # real work happened

    second = yir._default_volume_provider(yir.VolumeSetupContext(
        render_context=rc, dataset=no_path, custom_params={}, frame=None,
        previous_scene=first.scene, previous_ds=first.ds,
    ))
    assert second.changed is False
    assert second.scene == first.scene
    assert len(rc.add_scene_calls) == 1  # NOT called again -- reused, not rebuilt


def test_default_volume_provider_loads_the_dataset_path_and_reloads_when_it_changes(monkeypatch):
    loaded = []
    fake_yt = types.ModuleType("yt")
    fake_yt.load = lambda path: loaded.append(path) or types.SimpleNamespace(all_data=lambda: f"all[{path}]")
    monkeypatch.setitem(sys.modules, "yt", fake_yt)
    rc = _FakeRenderContext()

    def provide(path, previous=None, params=None):
        return yir._default_volume_provider(yir.VolumeSetupContext(
            render_context=rc, dataset=types.SimpleNamespace(dataset_path=path, dataset_kind=""),
            custom_params=params or {}, frame=None,
            previous_scene=previous and previous.scene, previous_ds=previous and previous.ds,
        ))

    first = provide("/data/a", params={"yt_idv_field": "gas, temperature"})
    assert loaded == ["/data/a"]
    assert rc.add_scene_calls[-1][:2] == ("all[/data/a]", ("gas", "temperature"))
    assert provide("/data/a", first).changed is False
    second = provide("/data/b", first)
    assert second.changed is True and loaded == ["/data/a", "/data/b"]


def test_get_volume_scene_uses_default_provider_when_custom_params_unset(monkeypatch, _clean_yir_state):
    fake_rc = _FakeRenderContext()
    monkeypatch.setattr(yir, "_get_render_context", lambda backend: fake_rc)
    monkeypatch.setattr(yir, "_load_sample_source", lambda online_access=None: ("ds-obj", "data-source-obj"))

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
    monkeypatch.setattr(yir, "_get_render_context", lambda backend: _FakeRenderContext())

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
    monkeypatch.setattr(yir, "_get_render_context", lambda backend: _FakeRenderContext())

    request = _request_with(custom_params={"yt_idv_volume_provider": "_test_yir_provider_mod_4:my_provider"})
    yir._get_volume_scene(request, render_size=64)
    yir._get_volume_scene(request, render_size=64)  # same object, same render_size

    assert seen_previous_scenes == [None, "scene-1"]


def test_get_volume_scene_keeps_the_scene_across_render_sizes(monkeypatch, _clean_yir_state):
    # One context serves every size (see _use_render_size), so a new size reuses the loaded scene.
    seen_previous_scenes = []

    def my_provider(ctx):
        seen_previous_scenes.append(ctx.previous_scene)
        return yir.VolumeSetupResult(scene="scene-x", ds="ds-x", changed=True)

    fake_module = types.ModuleType("_test_yir_provider_mod_5")
    fake_module.my_provider = my_provider
    monkeypatch.setitem(sys.modules, "_test_yir_provider_mod_5", fake_module)
    monkeypatch.setattr(yir, "_get_render_context", lambda backend: _FakeRenderContext())

    request = _request_with(custom_params={"yt_idv_volume_provider": "_test_yir_provider_mod_5:my_provider"})
    yir._get_volume_scene(request, render_size=64)
    yir._get_volume_scene(request, render_size=128)

    assert seen_previous_scenes == [None, "scene-x"]


def test_render_signature_changed_true_on_first_call(_clean_yir_state):
    assert yir._render_signature_changed("obj", _request_with(), render_size=4, volume_changed=False) is True


def test_render_signature_changed_false_when_nothing_changed(_clean_yir_state):
    request = _request_with()
    yir._record_render_signature("obj", request, render_size=4)
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=False) is False


def test_render_signature_changed_does_not_record(_clean_yir_state):
    request = _request_with()
    yir._render_signature_changed("obj", request, render_size=4, volume_changed=False)
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=False) is True


def test_render_signature_changed_true_when_camera_moves(_clean_yir_state):
    request = _request_with()
    yir._record_render_signature("obj", request, render_size=4)
    moved = _request_with(camera_overrides={
        "matrix_world": [[1, 0, 0, 9], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    })
    assert yir._render_signature_changed("obj", moved, render_size=4, volume_changed=False) is True


def test_render_signature_changed_true_when_depth_buffer_content_differs(_clean_yir_state):
    # e.g. an occluder moved while the camera stayed put.
    request = _request_with()
    yir._record_render_signature("obj", request, render_size=4)
    request.depth.data[0, 0] = 1.0
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=False) is True


def test_render_signature_changed_forced_true_when_volume_provider_reports_changed(_clean_yir_state):
    request = _request_with()
    yir._record_render_signature("obj", request, render_size=4)
    assert yir._render_signature_changed("obj", request, render_size=4, volume_changed=True) is True


def test_render_signature_changed_when_render_mode_changes(_clean_yir_state):
    request = _request_with()
    yir._record_render_signature("obj", request, 4, render_mode="_configure_max_intensity")
    assert yir._render_signature_changed("obj", request, 4, False, render_mode="_configure_max_intensity") is False
    assert yir._render_signature_changed("obj", request, 4, False, render_mode="_configure_max_intensity_alpha") is True


class _FakeYtIdvCamera:
    def __init__(self):
        self.update_calls = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)


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
            # Distinct image per real render, so cached vs. fresh is detectable.
            return np.full((render_size, render_size, 4), fill_value=float(self.run_calls), dtype=np.float32)

    fake_rc = _FakeRc()

    def fake_get_volume_scene(request, rsize):
        call_count["n"] += 1
        # Only the first call reports a volume-side change.
        return fake_rc, fake_scene, _FakeDataset(), call_count["n"] == 1

    monkeypatch.setattr(yir, "_get_volume_scene", fake_get_volume_scene)
    monkeypatch.setattr(yir, "_cycles_depth_to_yt_idv_window_depth", lambda request, camera, model_shift=None: np.zeros((render_size, render_size), dtype=np.float32))
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


def test_render_with_yt_idv_retries_after_a_failed_render(monkeypatch, _clean_yir_state):
    render_size = 4

    class _FlakyRc:
        run_calls = 0

        def run(self):
            self.run_calls += 1
            if self.run_calls == 1:
                raise RuntimeError("GL error")
            return np.ones((render_size, render_size, 4), dtype=np.float32)

    rc = _FlakyRc()
    monkeypatch.setattr(yir, "_get_volume_scene", lambda request, rsize: (rc, _FakeYtIdvScene(), _FakeDataset(), False))
    monkeypatch.setattr(yir, "_cycles_depth_to_yt_idv_window_depth", lambda request, camera, model_shift=None: np.zeros((render_size, render_size), dtype=np.float32))
    monkeypatch.setattr(yir, "set_external_depth_clip", lambda scene, depth: None)
    request = _request_with(render_size=render_size)

    with pytest.raises(RuntimeError):
        yir._render_with_yt_idv(request, lambda component, custom_params: None)
    result = yir._render_with_yt_idv(request, lambda component, custom_params: None)

    assert rc.run_calls == 2
    assert result.rgba.max() == 1.0


@pytest.mark.parametrize("params, invalid, bounds", [
    ({}, True, (None, None)),  # nothing remembered: fit this render
    ({"yt_idv_cmap_range": (1.0, 2.0)}, False, (1.0, 2.0)),
    ({"yt_idv_cmap_range": (1.0, 2.0), "yt_idv_rescale_colormap": True}, True, (None, None)),
])
def test_apply_colormap_range(params, invalid, bounds):
    component = types.SimpleNamespace(fixed_cmap_min=None, fixed_cmap_max=None, cmap_min=None,
                                      cmap_max=None, _cmap_bounds_invalid=None)
    yir._apply_colormap_range(component, params)
    assert component._cmap_bounds_invalid is invalid
    assert (component.cmap_min, component.cmap_max) == bounds


def test_render_with_yt_idv_reasserts_rc_scene_when_objects_share_a_context(monkeypatch, _clean_yir_state):
    # rc.run() renders rc.scene (the last add_scene), so objects sharing a
    # context must each reassert their own scene.
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
    monkeypatch.setattr(yir, "_cycles_depth_to_yt_idv_window_depth", lambda request, camera, model_shift=None: np.zeros((render_size, render_size), dtype=np.float32))
    monkeypatch.setattr(yir, "set_external_depth_clip", lambda scene, depth: None)
    configure = lambda component, custom_params: None

    yir._render_with_yt_idv(_request_with(dataset_id="a", render_size=render_size), configure)
    yir._render_with_yt_idv(_request_with(dataset_id="b", render_size=render_size), configure)
    yir._render_with_yt_idv(_request_with(dataset_id="a", render_size=render_size), configure)

    assert shared_rc.rendered_scenes == [scene_a, scene_b, scene_a]


def test_set_external_depth_clip_reuses_existing_texture_instead_of_leaking_gpu_memory(monkeypatch):
    # Regression: yt_idv Textures are never freed, so a new Texture2D per
    # frame leaked GPU memory until long renders crashed.
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


def test_a_new_depth_clip_size_frees_the_old_texture(monkeypatch):
    released = []

    class FakeTexture2D:
        def __init__(self, data):
            self.data = data

        def release(self):
            released.append(self)

    fake_module = types.ModuleType("yt_idv.opengl_support")
    fake_module.Texture2D = FakeTexture2D
    monkeypatch.setitem(sys.modules, "yt_idv.opengl_support", fake_module)
    monkeypatch.setattr(yir, "_depth_textures", {})
    component = types.SimpleNamespace(use_external_depth_clip=False, external_depth_texture=None)
    scene = types.SimpleNamespace(components=[component])

    yir.set_external_depth_clip(scene, np.zeros((8, 8), dtype="f4"))  # F12's size
    full = component.external_depth_texture
    yir.set_external_depth_clip(scene, np.zeros((4, 4), dtype="f4"))  # a viewport's
    yir.set_external_depth_clip(scene, np.zeros((8, 8), dtype="f4"))

    assert released[0] is full and len(released) == 2
    assert len(yir._depth_textures) == 1


def test_fix_yt_idv_axis_order_passes_through_already_correct_shape():
    image = np.zeros((100, 200, 4), dtype=np.float32)
    fixed = yir._fix_yt_idv_axis_order(image, expected_height=100, expected_width=200)
    assert fixed.shape == (100, 200, 4)
    assert fixed is image


def test_fix_yt_idv_axis_order_transposes_yt_idv_bug_1_shape():
    # yt_idv bug 1: non-square renders come back as (width, height, 4).
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
    # Workaround for yt_idv bug 3: render square, then crop.
    square = np.arange(10 * 10 * 4, dtype=np.float32).reshape(10, 10, 4)
    cropped = yir._center_crop(square, expected_height=4, expected_width=8)
    assert cropped.shape == (4, 8, 4)
    y0 = (10 - 4) // 2
    x0 = (10 - 8) // 2
    np.testing.assert_array_equal(cropped, square[y0:y0 + 4, x0:x0 + 8, :])


class _FakeTransferFunctionTexture:
    def __init__(self):
        self.data = None


class _FakeComponent:
    """Stand-in for a yt_idv block_rendering component."""

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
    ramp = yir._alpha_ramp(yir.MIP_ALPHA_NBINS, yir.MIP_ALPHA_MAX, yir.MIP_ALPHA_RAMP_POWER)
    assert ramp[0] == pytest.approx(0.0)
    assert ramp[-1] == pytest.approx(yir.MIP_ALPHA_MAX)
    assert np.all(np.diff(ramp) >= 0.0)


# --- Preferences-panel override mechanism (custom_params -> _configure_*) --
# The yt/matplotlib-dependent _get_*_data calls are replaced with spies.

def test_tf_bounds_fit_percentiles_of_the_data_or_use_fixed_log10_values(monkeypatch):
    collection = types.SimpleNamespace(_textures_are_normalized=False)
    edges = np.linspace(-30.0, -20.0, 11)  # uniform by volume
    monkeypatch.setattr(yir, "_value_distribution", lambda c: (edges, np.ones(10), np.ones(10)))

    assert yir._tf_bounds(collection, {}) == pytest.approx((-29.0, -20.0))  # 10% to 100%
    assert yir._tf_bounds(collection, {"yt_idv_tf_percentile_low": 10.0, "yt_idv_tf_percentile_high": 90.0}) == pytest.approx((-29.0, -21.0))
    fixed = {"yt_idv_tf_fit_range": False, "yt_idv_tf_log_min": -27.0, "yt_idv_tf_log_max": -23.0}
    assert yir._tf_bounds(collection, fixed) == (-27.0, -23.0)


def test_brightness_ramp_scales_colors_with_log_value_but_not_opacity():
    import yt

    tf = yt.ColorTransferFunction((-20.0, -10.0), nbins=11, grey_opacity=True)
    for channel in tf.funcs:
        channel.y[:] = 0.5

    yir._ramp_brightness(tf, 100.0)

    np.testing.assert_allclose(tf.funcs[0].y, 0.5 * 100.0 ** np.linspace(0, 1, 11))
    np.testing.assert_allclose(tf.funcs[3].y, 0.5)


def test_layer_values_match_yt_add_layers():
    import yt

    tf = yt.ColorTransferFunction((-28.0, -22.0), nbins=6001, grey_opacity=True)
    tf.add_layers(4, w=1e-4)
    alpha = tf.funcs[3].y
    peaks = tf.funcs[3].x[np.nonzero((alpha[1:-1] > alpha[:-2]) & (alpha[1:-1] >= alpha[2:]))[0] + 1]
    np.testing.assert_allclose(yir._layer_values((-28.0, -22.0), 4), peaks, atol=2e-3)


def test_camera_fade_dims_only_layers_the_camera_has_passed():
    layers = np.array([-27.0, -26.0, -25.5, -25.0, -22.0])
    np.testing.assert_allclose(yir._camera_fade(layers, 1e-25, 1.0), [0.0, 0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(yir._camera_fade(layers, None, 1.0), 1.0)  # outside the data


def test_approach_value_is_the_running_maximum_along_the_way_in(monkeypatch):
    collection = types.SimpleNamespace(_yt_volumes_zoom_max=np.zeros(3))
    monkeypatch.setattr(yir, "_block_collections", lambda scene: [collection])
    # denser inward, with a dip at r = 0.1 and a spike at r = 1e-3
    def field(scene, point):
        r = np.linalg.norm(point)
        bump = 100.0 if 9e-4 < r < 1.1e-3 else (0.01 if 0.05 < r < 0.2 else 1.0)
        return bump / max(r, 1e-12)
    monkeypatch.setattr(yir, "_field_value_at", field)

    at = lambda r: yir._approach_value(None, np.array([r, 0.0, 0.0]), "max")
    assert at(1.0) == pytest.approx(1.0, rel=0.05)
    assert at(0.1) == pytest.approx(5.0, rel=0.05)  # held at its value going into the dip (r = 0.2)
    assert at(5e-4) == pytest.approx(100.0 / 9e-4, rel=0.05)  # the spike holds until passed
    assert at(1e-6) == pytest.approx(1e6, rel=0.05)


def test_smoothed_running_maximum_stays_monotonic_without_jumps():
    staircase = np.where(np.arange(yir._APPROACH_RADII.size) < 1000, 3.0, 1.0)  # a 2-decade jump
    staircase[-50:] = np.nan  # outside the data

    smooth = yir._smooth_monotonic(staircase)

    inside = smooth[:-50]
    assert np.all(np.diff(inside) <= 1e-12)  # still non-increasing outward
    assert np.abs(np.diff(inside)).max() < 0.05  # the jump is spread over ~0.3 decades
    assert np.isnan(smooth[-50:]).all()


def test_percentile_ends_are_the_outermost_occupied_bins():
    edges = np.linspace(-10.0, 0.0, 11)
    volume = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-30, 0.0])  # the last occupied bin vanishes in the sum

    low, high = yir._percentile_range((edges, volume, volume), 0.0, 100.0)

    assert np.log10([low, high]) == pytest.approx([-9.0, -1.0])


def test_configure_transfer_function_builds_yt_add_layers(monkeypatch):
    component = _FakeComponent()
    component.data = types.SimpleNamespace()
    applied = []
    monkeypatch.setattr(yir, "_tf_bounds", lambda collection, params: (-28.0, -22.0))
    monkeypatch.setattr(yir, "_apply_yt_transfer_function", lambda c, params, key, make: applied.append((key, make)))

    yir._configure_transfer_function(component, {
        "yt_idv_tf_n_layers": 4, "yt_idv_tf_layer_width": 0.01, "yt_idv_colormap": "viridis",
        "yt_idv_tf_alpha_first": 0.1, "yt_idv_tf_alpha_last": 1.0,
    })

    (key, make), = applied
    assert key[1:4] == ((-28.0, -22.0), 4, 0.01) and key[-2:] == ("viridis", 1.0)
    tf = make()
    assert tf.grey_opacity and tuple(tf.x_bounds) == (-28.0, -22.0)
    # four peaks, alpha rising geometrically from 0.1 to 1
    alpha = tf.funcs[3].y
    peaks = np.nonzero((alpha[1:-1] > alpha[:-2]) & (alpha[1:-1] >= alpha[2:]))[0] + 1
    assert len(peaks) == 4
    np.testing.assert_allclose(alpha[peaks], np.geomspace(0.1, 1.0, 4), rtol=0.02)


def test_configure_max_intensity_alpha_custom_params_override_module_constants(monkeypatch):
    component = _FakeComponent()
    component.colormap = _FakeTransferFunctionTexture()  # just needs a settable .data
    calls = []
    monkeypatch.setattr(
        yir, "_get_mip_alpha_colormap_data",
        lambda *a, **k: (calls.append((a, k)), np.zeros((256, 4), dtype="f4"))[1],
    )

    yir._configure_max_intensity_alpha(component, custom_params={
        "yt_idv_colormap": "arbre", "yt_idv_mip_alpha_max": 0.9, "yt_idv_mip_alpha_ramp_power": 0.5,
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
    component = _FakeComponent()
    fake_tf_data = np.zeros((256, 1, 4), dtype="u1")

    yir._apply_transfer_function_settings(component, tf_min=1e-5, tf_max=3e-3, tf_data=fake_tf_data)

    assert component.tf_min == 1e-5
    assert component.tf_max == 3e-3
    assert component.tf_log is True
    assert component.transfer_function.data is fake_tf_data
    assert component.render_method is None  # not this function's job


def test_render_method_mode_switch_is_idempotent_on_the_same_component():
    # Cached contexts are reused across modes, so each mode must reconfigure.
    component = _FakeComponent()
    yir._apply_transfer_function_settings(component, tf_min=1e-5, tf_max=3e-3, tf_data=np.zeros((256, 1, 4), dtype="u1"))
    component.render_method = "transfer_function"
    assert component.render_method == "transfer_function"

    yir._configure_max_intensity(component, custom_params={})
    assert component.render_method == "max_intensity"
    # Leftover tf_* fields are harmless; max_intensity doesn't read them.
    assert component.tf_min == 1e-5


def test_alpha_ramp_starts_at_zero_and_ends_at_alpha_max():
    ramp = yir._alpha_ramp(nbins=256, alpha_max=0.85, ramp_power=2.0)
    assert ramp.shape == (256,)
    assert ramp[0] == pytest.approx(0.0)
    assert ramp[-1] == pytest.approx(0.85)


def test_alpha_ramp_is_monotonically_nondecreasing():
    ramp = yir._alpha_ramp(nbins=256, alpha_max=1.0, ramp_power=2.0)
    assert np.all(np.diff(ramp) >= 0.0)


def test_alpha_ramp_power_above_one_stays_below_the_linear_ramp():
    linear = yir._alpha_ramp(nbins=101, alpha_max=1.0, ramp_power=1.0)
    slow = yir._alpha_ramp(nbins=101, alpha_max=1.0, ramp_power=2.0)
    midpoint = 50
    assert slow[midpoint] < linear[midpoint]


def test_resample_1d_returns_input_unchanged_when_already_the_right_length():
    values = np.array([0.1, 0.4, 0.9])
    assert np.array_equal(yir._resample_1d(values, nbins=3), values)


def test_resample_1d_linearly_interpolates_to_a_different_length():
    resampled = yir._resample_1d([0.0, 0.5, 1.0], nbins=5)
    assert resampled == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_resample_rgb_interpolates_each_channel_independently():
    rgb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # red -> green, 2 samples
    resampled = yir._resample_rgb(rgb, nbins=3)
    assert resampled.shape == (3, 3)
    assert resampled[0] == pytest.approx([1.0, 0.0, 0.0])
    assert resampled[1] == pytest.approx([0.5, 0.5, 0.0])
    assert resampled[2] == pytest.approx([0.0, 1.0, 0.0])


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


def test_pad_to_square_centers_the_array_and_fills_the_margin():
    depth = np.zeros((4, 8), dtype="f4")
    padded = yir._pad_to_square(depth, render_size=10, fill_value=1.0)
    assert padded.shape == (10, 10)
    y0, x0 = (10 - 4) // 2, (10 - 8) // 2
    np.testing.assert_array_equal(padded[y0 : y0 + 4, x0 : x0 + 8], depth)
    assert padded[0, 0] == 1.0
    assert padded[-1, -1] == 1.0


# --- _cycles_depth_to_yt_idv_window_depth ------------------------------------
# Uses a real bpy camera, same setup as tests/test_sample_renderers.py.

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
    """2x2x2 box at world (0, 0, 5): planar depth [4, 6] from the camera at z=10."""
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
    """Identity-matrix yt_idv camera, so window_depth = 0.5 * unitary_z + 0.5."""

    projection_matrix = np.eye(4)
    view_matrix = np.eye(4)


def test_cycles_depth_to_yt_idv_window_depth_is_uniform_across_a_planar_hit(persp_camera_params):
    # PLANAR depth is view-space Z, so depth 5 is world z=5 (unitary 0.5)
    # everywhere -> window_depth 0.75 across the whole array.
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
    # TOP_LEFT_ORIGIN so [0, 0] needs no row flip.
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


# --- zoom ---------------------------------------------------------------------


def test_zoom_target_parses_float64_coordinates():
    target = yir._zoom_target_unitary("0.4915073296178, 0.505260534, 0.49058056", None)
    assert target.dtype == np.float64
    assert target[0] == 0.4915073296178


def test_zoom_target_rejects_malformed_spec():
    with pytest.raises(ValueError, match="zoom target"):
        yir._zoom_target_unitary("0.5, 0.5", None)


def test_apply_zoom_without_target_leaves_transform_alone():
    collection = _subset_collection()
    yir.scale_scene_to_subset(_block_scene(collection))
    before = (collection.applied_scale_offset, collection.applied_scale_ratio)
    assert yir.apply_zoom(_block_scene(collection), {}) is None
    assert (collection.applied_scale_offset, collection.applied_scale_ratio) == before


def test_apply_zoom_puts_target_at_model_origin_magnified():
    collection = _full_domain_collection()
    scene = _block_scene(collection)
    target = np.array([0.25, 0.5, 0.75])
    anchor, scale = yir.apply_zoom(scene, {"yt_idv_zoom_target": "0.25 0.5 0.75", "yt_idv_zoom_level": 3.0})
    np.testing.assert_allclose(anchor, target)  # full domain: cube coords are unitary
    assert scale == 1.0  # no camera given
    np.testing.assert_allclose(collection.applied_scale_offset, target)
    assert collection.applied_scale_ratio == pytest.approx(1e-3)
    # the block corner at the origin lands 1000x farther from the target
    np.testing.assert_allclose(collection.vertex_array.attributes[0].data[0], -target * 1e3, rtol=1e-6)


def test_apply_zoom_is_absolute_across_frames_and_reverts_when_cleared():
    collection = _full_domain_collection()
    scene = _block_scene(collection)
    for level in (1.0, 5.0, 2.0):
        yir.apply_zoom(scene, {"yt_idv_zoom_target": "0.5 0.5 0.5", "yt_idv_zoom_level": level})
    assert collection.applied_scale_ratio == pytest.approx(1e-2)
    yir.apply_zoom(scene, {})
    np.testing.assert_allclose(collection.applied_scale_offset, 0.0)
    assert collection.applied_scale_ratio == pytest.approx(1.0)


def test_apply_zoom_composes_with_subset_scaling():
    collection = _subset_collection()
    scene = _block_scene(collection)
    yir.scale_scene_to_subset(scene)
    # region low corner (0.4, 0.45, 0.45), longest axis 0.1: unitary 0.45 -> cube 0.5
    anchor, _ = yir.apply_zoom(scene, {"yt_idv_zoom_target": "0.45 0.475 0.4625", "yt_idv_zoom_level": 2.0})
    np.testing.assert_allclose(anchor, [0.5, 0.25, 0.125], atol=1e-12)
    assert collection.applied_scale_ratio == pytest.approx(0.1 / 100.0)


def test_registered_camera_kwargs_apply_model_frame():
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
    plain = yir._registered_camera_kwargs(camera, _identity_volume(), _FakeDataset(), 100, 100)
    anchor = np.array([0.25, 0.5, 0.75])
    framed = yir._registered_camera_kwargs(
        camera, _identity_volume(), _FakeDataset(), 100, 100, model_frame=(anchor, 10.0),
    )
    np.testing.assert_allclose(framed["position"], (plain["position"] - anchor) * 10.0, rtol=1e-6)
    view = lambda k: (k["focus"] - k["position"]) / np.linalg.norm(k["focus"] - k["position"])
    np.testing.assert_allclose(view(framed), view(plain), atol=1e-6)


def test_cycles_depth_to_yt_idv_window_depth_applies_model_frame(persp_camera_params):
    request = RenderRequest(
        camera=persp_camera_params,
        depth=_uniform_depth_buffer(5.0),
        volume=_boxed_volume_at_5(),
        frame=FrameParams(frame_number=1, scene_time_seconds=0.0),
    )
    window_depth = yir._cycles_depth_to_yt_idv_window_depth(
        request, _IdentityYtIdvCamera(), (np.array([0.0, 0.0, 0.25]), 2.0),
    )
    # unitary z 0.5 -> (0.5 - 0.25) * 2 = 0.5 -> 0.5 * 0.5 + 0.5
    np.testing.assert_allclose(window_depth, 0.75, atol=1e-5)


def test_zoom_target_max_without_dataset_finds_densest_unmasked_cell():
    # two 2x2x2-cell blocks; the hottest corner of block 1 is masked off
    hot = np.zeros((3, 3, 3), dtype="f4")
    hot[2, 2, 2] = 8.0  # only cell (1, 1, 1) touches it: value 1.0
    masked = np.full((3, 3, 3), 5.0, dtype="f4")
    collection = _FakeBlockCollection(
        left_edges=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        right_edges=[[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
    )
    collection.vertex_array.attributes.append(
        types.SimpleNamespace(name="in_dx", data=np.full((2, 3), 0.125, dtype="f4"))  # model units
    )
    collection.data_source = None
    collection.block_data = {0: hot, 1: masked}
    collection.block_bitmaps = {
        0: np.full((2, 2, 2), 255, dtype="u1"),
        1: np.zeros((2, 2, 2), dtype="u1"),
    }
    collection._set_scale(np.array([0.5, 0.0, 0.0]), 2.0)  # unitary = model * 2 + offset
    target = yir._zoom_target_unitary("max", collection)
    # cell (1, 1, 1) of block 0 spans unitary [0.25, 0.5]^3
    np.testing.assert_allclose(target, [0.375, 0.375, 0.375], atol=1e-6)


def test_apply_zoom_scales_so_the_camera_is_one_unit_from_the_target():
    # 2x2x2 cube at the origin: unitary 0.5 is world 0; the camera is 1e-6 cube widths away
    collection = _full_domain_collection()
    anchor, scale = yir.apply_zoom(
        _block_scene(collection), {"yt_idv_zoom_target": "0.5 0.5 0.5", "yt_idv_zoom_level": 2.0},
        _identity_volume(), np.array([0.0, 0.0, 2e-6]),
    )
    assert scale == pytest.approx(1e6)
    # the data is magnified 100x on top of the numeric scale
    assert collection.applied_scale_ratio == pytest.approx(1.0 / (100.0 * 1e6))


def test_apply_zoom_leaves_scale_alone_for_a_distant_camera():
    _, scale = yir.apply_zoom(
        _block_scene(_full_domain_collection()), {"yt_idv_zoom_target": "0.5 0.5 0.5"},
        _identity_volume(), np.array([0.0, 0.0, 5.0]),
    )
    assert scale == 1.0


def test_world_to_model_keeps_precision_near_a_target_at_the_world_origin():
    # Regression: going through unitary coordinates (~0.5) lost 1e-16 absolute,
    # i.e. 1e-4 of a 1e-12 offset.
    target = np.array([0.5396208033853327, 0.5348483302877867, 0.5151424408177263])
    offset = (-(2 * target - 1)).astype("f4").astype(np.float64)  # Blender stores float32
    volume = _identity_volume(matrix_world=[[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]])
    A, b = yir._world_to_model(volume, (target, 1.0))
    camera = b + np.array([0.0, 0.0, 2e-12])  # 1e-12 cube widths above the target
    np.testing.assert_allclose(A @ (camera - b), [0.0, 0.0, 1e-12], rtol=1e-9, atol=1e-24)


# --- spherical data -------------------------------------------------------------


def _spherical_collection():
    """A shell's blocks: native (r, theta, phi) edges plus their cartesian boxes."""
    attrs = {
        "in_left_edge": [[0.5, 0.0, 0.0]], "in_right_edge": [[1.0, np.pi, 2 * np.pi]],
        "le_cart": [[-1.0, -1.0, -1.0]], "re_cart": [[1.0, 1.0, 1.0]],
    }
    return types.SimpleNamespace(
        _yt_geom_str="spherical",
        scale=False,
        vertex_array=types.SimpleNamespace(attributes=[
            types.SimpleNamespace(name=k, data=np.asarray(v, dtype="f4")) for k, v in attrs.items()
        ]),
    )


def test_spherical_bounds_come_from_the_cartesian_boxes():
    low, high = yir._scene_model_bounds(_block_scene(_spherical_collection()))
    np.testing.assert_allclose(low, -1.0)
    np.testing.assert_allclose(high, 1.0)


def test_spherical_scenes_get_the_unit_sphere_frame_and_no_subset_scaling():
    scene = _block_scene(_spherical_collection())
    yir.scale_scene_to_subset(scene)  # would fail on the fake if it tried
    assert yir.apply_zoom(scene, {}) is yir.SPHERICAL_MODEL_FRAME


def test_spherical_scenes_refuse_to_zoom():
    with pytest.raises(NotImplementedError, match="spherical"):
        yir.apply_zoom(_block_scene(_spherical_collection()), {"yt_idv_zoom_target": "max"})


def test_spherical_registration_maps_the_inscribed_sphere_to_the_unit_sphere():
    # a radius-1 Empty-Sphere at the origin; the camera 3 radii above the north pole
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
    volume = _identity_volume(bounding_box_local=_cube_bbox())
    kwargs = yir._registered_camera_kwargs(
        camera, volume, _FakeDataset(), 100, 100, model_frame=yir.SPHERICAL_MODEL_FRAME,
    )
    np.testing.assert_allclose(kwargs["position"], [0.0, 0.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(kwargs["focus"], [0.0, 0.0, 0.0], atol=1e-6)  # looking at the center


# --- shot space -----------------------------------------------------------------


def _deep_zoom_shot_space():
    """The Deep Zoom extension's pure-math module, loaded from extensions/."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "extensions" / "deep_zoom" / "shot_space.py"
    spec = importlib.util.spec_from_file_location("deep_zoom_shot_space", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shot_space_volume_maps_world_to_the_shot_space_formula():
    shot_space = _deep_zoom_shot_space()

    collection = _full_domain_collection()
    scene = _block_scene(collection)
    anchor = np.array([0.491507330198, 0.505260534, 0.490580561])
    params = {"yt_idv_shot_space": 2.0, "yt_idv_zoom_target": shot_space.format_anchor(anchor)}
    volume = yir._shot_space_volume(scene, _identity_volume(bounding_box_local=_cube_bbox()), params)
    # at zoom 0 a unitary point lands at (point - anchor) * 2; check through the worker's own mapping
    point = anchor + np.array([1e-3, -2e-3, 5e-4])
    A, b = yir._world_to_model(volume)  # world -> unitary (full domain: cube = unitary)
    world = shot_space.to_world(point, anchor, 0.0)
    np.testing.assert_allclose(A @ (world - b), point, rtol=0, atol=1e-15)
    # and the anchor is the world origin
    np.testing.assert_allclose(A @ (np.zeros(3) - b), anchor, rtol=0, atol=1e-15)


def test_shot_space_volume_leaves_ordinary_scenes_alone():
    volume = _identity_volume()
    assert yir._shot_space_volume(_block_scene(_full_domain_collection()), volume, {}) is volume


def test_shot_space_registers_data_and_camera_together_at_any_zoom():
    shot_space = _deep_zoom_shot_space()

    anchor = np.array([0.491507330198, 0.505260534, 0.490580561])
    point = anchor + np.array([3e-10, -1e-10, 2e-10])  # a feature 1e-10 from the anchor
    for zoom in (0.0, 4.0, 9.5):
        collection = _full_domain_collection()
        scene = _block_scene(collection)
        params = {
            "yt_idv_shot_space": 2.0,
            "yt_idv_zoom_target": shot_space.format_anchor(anchor),
            "yt_idv_zoom_level": zoom,
        }
        volume = yir._shot_space_volume(scene, _identity_volume(bounding_box_local=_cube_bbox()), params)
        camera_world = shot_space.to_world(point, anchor, zoom)  # a camera sitting on the feature
        frame = yir.apply_zoom(scene, params, volume, camera_world)
        A, b = yir._world_to_model(volume, frame)
        camera_model = A @ (camera_world - b)
        # where yt_idv draws the feature: model = (unitary - offset) / ratio
        data_model = (point - np.asarray(collection.applied_scale_offset)) / collection.applied_scale_ratio
        np.testing.assert_allclose(camera_model, data_model, rtol=1e-9)


def test_planes_are_bounded_relative_to_the_target_in_a_zoom_frame():
    # Regression: deep in a zoom the far plane reached the whole domain, ~1e9
    # model units, and float32 ray reconstruction broke down.
    camera = _persp_camera(matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3e-9], [0, 0, 0, 1]])
    anchor = np.full(3, 0.5)
    kwargs = yir._registered_camera_kwargs(
        camera, _identity_volume(), _FakeDataset(), 100, 100,
        model_bounds=(np.full(3, -1e9), np.full(3, 1e9)), model_frame=(anchor, 1e9),
    )
    to_target = np.linalg.norm(kwargs["position"])
    assert to_target == pytest.approx(1.5, rel=1e-6)
    assert kwargs["far_plane"] == pytest.approx(yir.FAR_REACH * to_target)
    assert kwargs["near_plane"] == pytest.approx(yir.NEAR_FRACTION * to_target)


def test_zoom_depth_ends_rays_past_the_target():
    planes = yir.camera_planes(np.array([0.0, 0.0, 1.5]), (np.full(3, -1e9), np.full(3, 1e9)), (np.zeros(3), 1.0), zoom_depth=2.0)
    assert planes["far_plane"] == pytest.approx(3.0)
    unlimited = yir.camera_planes(np.array([0.0, 0.0, 1.5]), (np.full(3, -1e9), np.full(3, 1e9)), (np.zeros(3), 1.0))
    assert unlimited["far_plane"] == pytest.approx(yir.FAR_REACH * 1.5)
    unzoomed = yir.camera_planes(np.array([0.5, 0.5, 3.0]), zoom_depth=2.0)  # no zoom frame: no limit
    assert unzoomed["far_plane"] > 3.0


def test_field_value_at_reads_the_nearest_texel_of_the_containing_block():
    # one block over [0, 1]^3 with 2 cells per side, so 3 vertex texels per side
    data = np.arange(27, dtype="f4").reshape(3, 3, 3) / 26.0
    collection = types.SimpleNamespace(
        _yt_geom_str="cartesian",
        _get_unscaled_edges=lambda: (np.zeros((1, 3)), np.ones((1, 3)), np.full((1, 3), 0.5)),
        block_data={0: data},
        _textures_are_normalized=True, val_range=26.0, min_val=100.0,
    )
    scene = _block_scene(collection)
    # (0.9, 0.1, 0.6) is nearest vertex (2, 0, 1): flat index 19
    assert yir._field_value_at(scene, np.array([0.9, 0.1, 0.6])) == pytest.approx(119.0)
    assert yir._field_value_at(scene, np.array([1.5, 0.5, 0.5])) is None


def test_needs_load_until_the_scene_is_cached_for_this_dataset(_clean_yir_state):
    request = _request_with(render_size=4)
    assert yir.needs_load(request)
    scene = types.SimpleNamespace(_yt_volumes_source=yir.SAMPLE_DATASET_NAME)
    yir._volume_scene_cache["obj"] = (scene, None, "egl")
    assert not yir.needs_load(request)
    assert not yir.needs_load(_request_with(render_size=8))  # any size reuses it
    request.volume.dataset.dataset_path = "/data/other.zip"
    assert yir.needs_load(request)


def test_samples_per_cell_sets_cartesian_and_spherical_sample_factor():
    cartesian = types.SimpleNamespace(sample_factor=1.0, data=types.SimpleNamespace(_yt_geom_str="cartesian"))
    spherical = types.SimpleNamespace(sample_factor=0.0, data=types.SimpleNamespace(_yt_geom_str="spherical"))
    annotation = types.SimpleNamespace(data=None)
    for component in (cartesian, spherical, annotation):
        yir._apply_samples_per_cell(component, {"yt_idv_samples_per_cell": 10.0})
    assert cartesian.sample_factor == 10.0
    assert spherical.sample_factor == pytest.approx(-1.0)  # a tenth of the element's width
    assert not hasattr(annotation, "sample_factor")
    yir._apply_samples_per_cell(cartesian, {})
    assert cartesian.sample_factor == yir.SAMPLES_PER_CELL  # unset: the default


def _gn_component(monkeypatch):
    """A fake component whose float32 table texture already exists (making one needs GL)."""
    component = _FakeComponent()
    component.transfer_function.data = np.zeros((256, 1, 4), dtype="f4")
    monkeypatch.setitem(yir._float_tf_textures, id(component), component.transfer_function)
    return component


def test_configure_gn_transfer_function_uploads_the_scaled_rgba_over_its_value_range(monkeypatch):
    component = _gn_component(monkeypatch)
    component.data = _two_block_data()
    table = component.transfer_function
    rgba = np.linspace(0.0, 1.0, 256)[:, None].repeat(4, axis=1).tolist()

    yir._configure_gn_transfer_function(component, custom_params={
        "yt_idv_gn_tf_rgba": rgba, "yt_idv_tf_fit_range": False, "yt_idv_gn_tf_opacity_scale": 50.0,
        # log10 of the field in its own units; the fake normalizes over [1, 11]
        "yt_idv_tf_log_min": np.log10(1.1), "yt_idv_tf_log_max": np.log10(2.0),
    })

    assert component.render_method == "transfer_function"
    assert component.tf_min == pytest.approx(0.01) and component.tf_max == pytest.approx(0.1)
    assert component.tf_log
    assert component.transfer_function is table  # updated in place, not a new texture
    data = table.data
    assert data.shape == (256, 1, 4) and data.dtype == np.float32
    assert data[0, 0].tolist() == [0, 0, 0, 0] and data[-1, 0].tolist() == [50, 50, 50, 50]


def test_configure_gn_transfer_function_without_samples_raises():
    with pytest.raises(ValueError, match="wasn't sampled"):
        yir._configure_gn_transfer_function(_FakeComponent(), custom_params={})


def _two_block_data():
    """A big block of thin gas (each texel 0.125 volume) and a tiny dense one (1e-6)."""
    attribute = types.SimpleNamespace(name="in_dx", data=np.array([[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]]))
    return types.SimpleNamespace(
        block_data={0: np.full((2, 2, 2), 1e-10), 1: np.full((2, 2, 2), 1e-3)},
        vertex_array=types.SimpleNamespace(attributes=[attribute]),
        _textures_are_normalized=True, min_val=1.0, max_val=11.0, field_units="g/cm**3",
    )


def test_gn_transfer_function_fits_percentiles_of_the_data_by_volume(monkeypatch):
    component = _gn_component(monkeypatch)
    component.data = _two_block_data()

    yir._configure_gn_transfer_function(component, custom_params={
        "yt_idv_gn_tf_rgba": np.ones((256, 4)).tolist(),
        "yt_idv_tf_log_min": -30.0, "yt_idv_tf_log_max": 0.0,  # ignored while fitting
    })

    # 10% by volume falls in the thin gas; 100% is the densest texel
    assert component.tf_min == pytest.approx(1e-10, rel=0.02)
    assert component.tf_max == pytest.approx(1e-3, rel=1e-5) and component.tf_max > 1e-3
    fit = component.data._yt_volumes_gn_tf_fit
    assert fit["units"] == "g/cm**3"
    assert fit["physical_range"][1] == pytest.approx(1e-3 * 10.0 + 1.0, rel=1e-5)
    points = fit["points"]
    # Almost all the volume is thin gas at the low end; the mass splits by value * volume.
    assert points["volume_fraction"].sum() == pytest.approx(1.0, abs=0.02)
    assert points["volume_fraction"][-1] == pytest.approx(1e-6 / (0.125 + 1e-6), rel=0.01)
    thin_mass, dense_mass = (1e-10 * 10 + 1) * 0.125, (1e-3 * 10 + 1) * 1e-6  # physical value times volume
    assert points["mass_fraction"][-1] == pytest.approx(dense_mass / (thin_mass + dense_mass), rel=0.01)
    assert points["log10_value"][-1] == pytest.approx(np.log10(1e-3 * 10.0 + 1.0), rel=1e-5)


def test_gn_transfer_function_high_percentile_can_cut_the_dense_tail(monkeypatch):
    component = _gn_component(monkeypatch)
    component.data = _two_block_data()

    yir._configure_gn_transfer_function(component, custom_params={
        "yt_idv_gn_tf_rgba": np.ones((256, 4)).tolist(), "yt_idv_tf_percentile_high": 99.0,
    })

    assert component.tf_max == pytest.approx(1e-10, rel=0.02)


def test_yt_transfer_function_reattaches_its_table_after_another_mode_swapped_it(monkeypatch):
    component = _FakeComponent()
    converted = {"transfer_function": object(), "tf_min": 0.1, "tf_max": 0.9, "tf_log": True}
    component.data = object()
    key = (id(component.data), "tf.py:make", 0.02)
    monkeypatch.setitem(yir._yt_tf_applied, id(component), (key, converted))
    monkeypatch.setattr(yir, "_resolve_callable", lambda spec: lambda: pytest.fail("rebuilt the table"))
    component.transfer_function = object()  # e.g. the GN mode's table

    yir._configure_yt_transfer_function(component, {"yt_idv_yt_tf": "tf.py:make"})

    assert component.transfer_function is converted["transfer_function"]
    assert (component.tf_min, component.tf_max) == (0.1, 0.9)


def test_yt_transfer_function_opacity_is_per_view_or_per_data_length(monkeypatch):
    import yt_idv.tf_conversion as tfc

    component = _FakeComponent()
    component.data = types.SimpleNamespace(_yt_volumes_base_transform=(np.zeros(3), 0.5), applied_scale_ratio=0.5e-4)
    lengths = []
    monkeypatch.setattr(tfc, "convert_yt_transfer_function", lambda tf, data, ray_length, nbins: lengths.append(ray_length) or {})
    monkeypatch.setattr(tfc, "apply_converted_transfer_function", lambda c, converted: None)
    monkeypatch.setitem(yir._yt_tf_applied, id(component), None)

    yir._apply_yt_transfer_function(component, {"yt_idv_yt_tf_ray_length": 0.03, "yt_idv_opacity_per_view": False}, None, lambda: None)
    monkeypatch.setitem(yir._yt_tf_applied, id(component), None)
    yir._apply_yt_transfer_function(component, {"yt_idv_yt_tf_ray_length": 0.03}, None, lambda: None)

    # per data length, undoing model units 1e4 times longer; then per view, as set
    assert lengths == [pytest.approx(0.03 * 1e4), pytest.approx(0.03)]


def test_exposure_factor_aims_the_brightest_pixels_near_white_without_clipping():
    image = np.zeros((100, 100, 4), dtype=np.float32)
    image[:10, :10, 0] = 0.45  # 1% of pixels at 0.45
    assert yir._exposure_factor(image) == pytest.approx(2.0)
    image[:10, :10, 0] = 0.88
    assert yir._exposure_factor(image) == 1.0  # within 10% of the target
    image[:10, :10, 0] = 1.0  # 1% clipped
    assert yir._exposure_factor(image) == 0.5
    assert yir._exposure_factor(np.zeros((4, 4, 4))) == 1.0  # nothing visible: leave the gain alone
    image[:] = 0.0
    image[0, 0, 0] = 0.3  # one lit pixel, under 0.1% of the frame
    assert yir._exposure_factor(image) == pytest.approx(3.0)


def test_auto_exposure_gain_stays_bounded(monkeypatch):
    monkeypatch.setattr(yir, "_exposure_gains", {"d": 6.12e44})  # a gain an older version wound up
    table = np.full((8, 1, 4), 0.1, dtype=np.float32)
    component = types.SimpleNamespace(render_method="transfer_function", transfer_function=types.SimpleNamespace(data=table))
    scene = types.SimpleNamespace(components=[component])
    dim = lambda: np.full((8, 8, 4), 1e-9, dtype=np.float32)  # never bright enough

    for _ in range(20):
        yir._auto_exposed(scene, "d", dim)
        assert yir.AUTO_EXPOSURE_GAINS[0] <= yir.exposure_gain("d") <= yir.AUTO_EXPOSURE_GAINS[1]
    yir._auto_exposed(scene, "d", lambda: np.zeros((8, 8, 4), dtype=np.float32))
    assert yir.exposure_gain("d") == yir.AUTO_EXPOSURE_GAINS[1]  # an empty frame doesn't move it


def test_auto_exposure_converges_and_leaves_the_cached_tables_unscaled(monkeypatch):
    monkeypatch.setattr(yir, "_exposure_gains", {})
    table = np.full((8, 1, 4), 0.1, dtype=np.float32)
    component = types.SimpleNamespace(render_method="transfer_function", transfer_function=types.SimpleNamespace(data=table))
    scene = types.SimpleNamespace(components=[component])

    def run():  # the image's brightest pixel follows the table, as emission would
        image = np.zeros((10, 10, 4), dtype=np.float32)
        image[:5, :1, 0] = min(float(component.transfer_function.data.max()) * 3.0, 1.0)  # 5% of pixels
        return image

    image = yir._auto_exposed(scene, "ds", run)

    assert float(image[..., 0].max()) == pytest.approx(yir.AUTO_EXPOSURE_TARGET, rel=0.1)
    assert yir.exposure_gain("ds") == pytest.approx(3.0, rel=0.1)
    np.testing.assert_array_equal(component.transfer_function.data, table)  # unscaled again


def test_render_sizes_use_a_16_bit_normalized_scene_framebuffer(monkeypatch):
    # Projection adds up alpha per brick crossed; a float target kept it above 1 (blown-out, blocky
    # renders), and an 8-bit one rounds away faint transfer-function emission (bands, lost gas).
    from OpenGL import GL
    import yt_idv.opengl_support as ogl

    class FakeFramebuffer:
        released = False

        def __init__(self, fb_tex=None):
            self.fb_tex = fb_tex

        def release(self):
            self.released = True

    class FakeScene:
        def __init__(self):
            self.fb = None
            self.components = [types.SimpleNamespace(fb=None)]

        def __iter__(self):
            return iter(self.components)

    monkeypatch.setattr(GL, "glViewport", lambda *args: None)
    monkeypatch.setattr(ogl, "Framebuffer", FakeFramebuffer)
    monkeypatch.setattr(yir, "_unorm16_texture", lambda size: ("RGBA16", size))
    scene = FakeScene()

    yir._use_render_size(scene, 64)
    first = scene.fb
    assert first.fb_tex == ("RGBA16", 64)
    yir._use_render_size(scene, 64)
    assert scene.fb is first  # same size: kept
    yir._use_render_size(scene, 32)
    assert first.released and scene.fb.fb_tex == ("RGBA16", 32)
    assert scene.components[0].fb._yt_volumes_size == 32


@pytest.mark.parametrize("offset", [(0, 0), (7, 0), (-5, 3)])
def test_a_shifted_crop_takes_back_the_padded_depth(offset):
    depth = np.arange(6 * 10, dtype="f4").reshape(6, 10)
    padded = yir._pad_to_square(depth, render_size=30, offset=offset)
    np.testing.assert_array_equal(yir._center_crop(padded[..., None], 6, 10, offset)[..., 0], depth)


def test_camera_shift_is_whole_pixels_along_the_sensor_fit():
    camera = types.SimpleNamespace(sensor_fit="AUTO", shift_x=0.1, shift_y=-0.05)
    assert yir._shift_offset(camera, height=100, width=200) == (20, -10)
    assert yir._shift_offset(camera, height=200, width=100) == (20, -10)  # AUTO fits the longer side


def test_auto_exposure_can_keep_the_last_gain(monkeypatch):
    monkeypatch.setattr(yir, "_exposure_gains", {})
    table = np.full((256, 1, 4), 0.5, dtype=np.float32)
    component = types.SimpleNamespace(render_method="transfer_function", transfer_function=types.SimpleNamespace(data=table))
    scene = types.SimpleNamespace(components=[component])
    yir._exposure_gains["d"] = 3.0
    dim = lambda: np.full((8, 8, 4), 0.01, dtype=np.float32)  # would raise the gain

    yir._auto_exposed(scene, "d", dim, adapt=False)
    assert yir._exposure_gains["d"] == 3.0
    yir._auto_exposed(scene, "d", dim)
    assert yir._exposure_gains["d"] > 3.0
