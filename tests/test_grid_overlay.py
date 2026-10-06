"""Tests for grid_overlay's box placement; drawing needs a GPU and isn't tested."""

import math
import types

import numpy as np
import pytest

from yt_volumes import grid_overlay
from yt_volumes import yt_idv_renderer as yir
from yt_volumes.external_render import VolumeObjectParams

from test_yt_idv_renderer import _block_scene, _FakeBlockCollection

LEFT = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.3, 0.35, 0.4]]
RIGHT = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.31, 0.36, 0.41]]


def _scene(bbox=None):
    collection = _FakeBlockCollection(LEFT, RIGHT, bbox=bbox)
    dx = (np.asarray(RIGHT) - np.asarray(LEFT)) / 8
    collection.vertex_array.attributes.append(types.SimpleNamespace(name="in_dx", data=dx.astype("f4")))
    return _block_scene(collection)


def _volume(custom_params):
    c, s = math.cos(0.4), math.sin(0.4)
    matrix = [[2 * c, -2 * s, 0, 1.5], [2 * s, 2 * c, 0, -3.0], [0, 0, 2, 0.5], [0, 0, 0, 1]]
    corners = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return VolumeObjectParams(matrix, corners, custom_params, None, None)


@pytest.fixture(autouse=True)
def _clear_cache():
    grid_overlay._unitary_cache.clear()


@pytest.mark.parametrize("custom_params", [
    {},
    {"yt_idv_zoom_target": "0.3, 0.35, 0.4", "yt_idv_zoom_level": 1.5},
    {"yt_idv_zoom_target": "0.3, 0.35, 0.4", "yt_idv_zoom_level": 1.0, "yt_idv_shot_space": 10.0},
])
@pytest.mark.parametrize("bbox", [None, ([0.2, 0.2, 0.2], [0.6, 0.7, 0.6])])
def test_boxes_land_where_the_renderer_puts_the_blocks(custom_params, bbox):
    scene = _scene(bbox)
    if bbox is not None:
        yir.scale_scene_to_subset(scene)
    volume = _volume(custom_params)
    corners, _ = grid_overlay.world_boxes(scene, types.SimpleNamespace(), volume, custom_params)

    # what _render_with_yt_idv does to place the scene
    render_volume = yir._shot_space_volume(scene, volume, custom_params)
    model_frame = yir.apply_zoom(scene, custom_params, render_volume)
    A, b = yir._world_to_model(render_volume, model_frame)
    collection = yir._block_collections(scene)[0]
    lows = yir._vertex_attribute(collection, "in_left_edge")
    highs = yir._vertex_attribute(collection, "in_right_edge")
    # the volume cube in model space; drawn boxes are the blocks clipped to it
    matrix = np.array(render_volume.matrix_world)
    local_min, local_extent = yir._bbox_local_min_extent(render_volume.bounding_box_local)
    cube_low, cube_high = (
        ((local_min + c * local_extent) @ matrix[:3, :3].T + matrix[:3, 3] - b) @ A.T for c in (0.0, 1.0)
    )
    lows, highs = np.clip(lows, cube_low, cube_high), np.clip(highs, cube_low, cube_high)
    assert len(corners)
    for box in corners:
        low = (box[0] - b) @ A.T
        i = int(np.argmin(np.abs(lows - low).max(axis=1)))
        np.testing.assert_allclose(low, lows[i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose((box[7] - b) @ A.T, highs[i], rtol=1e-5, atol=1e-6)


def test_boxes_are_clipped_to_the_cube():
    collection = _FakeBlockCollection(
        [[-0.5, 0.2, 0.2], [1.5, 1.5, 1.5], [-1.0, -1.0, -1.0]],
        [[0.5, 0.4, 0.4], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
    )
    collection.vertex_array.attributes.append(
        types.SimpleNamespace(name="in_dx", data=np.array([[0.1] * 3, [0.05] * 3, [0.2] * 3], dtype="f4"))
    )
    corners = [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    volume = VolumeObjectParams(np.eye(4).tolist(), corners, {}, None, None)
    # the second box is outside, the third covers the whole cube
    boxes, level = grid_overlay.world_boxes(_block_scene(collection), types.SimpleNamespace(), volume, {})
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0, 0], [0.0, 0.2, 0.2], atol=1e-6)
    np.testing.assert_allclose(boxes[0, 7], [0.5, 0.4, 0.4], atol=1e-6)


def test_dataset_grids_are_used_when_the_dataset_has_them():
    def unitary(values):
        return types.SimpleNamespace(to=lambda unit: types.SimpleNamespace(d=np.asarray(values, dtype="f8")))

    index = types.SimpleNamespace(
        grid_left_edge=unitary([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        grid_right_edge=unitary([[0.5, 1.0, 1.0], [0.75, 0.75, 0.75]]),
        grid_levels=np.array([[1], [0]]),
    )
    volume = _volume({})
    corners, level = grid_overlay.world_boxes(_scene(), types.SimpleNamespace(index=index), volume, {})
    assert list(level) == [0, 1]
    matrix = np.array(volume.matrix_world)
    expected = (-1.0 + 2 * np.array([0.5, 0.5, 0.5])) @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(corners[0, 0], expected)


def test_cap_is_shared_across_levels():
    level = np.array([0] * 10 + [1] * 10 + [5] * 3)
    picked = grid_overlay._spread_across_levels(level, 9)
    assert sorted(np.bincount(level[picked], minlength=6)[[0, 1, 5]]) == [3, 3, 3]


def test_level_range_selects_levels():
    _, level = grid_overlay.world_boxes(_scene(), types.SimpleNamespace(), _volume({}), {}, levels=(1, None))
    assert len(level) and level.min() >= 1
    _, level = grid_overlay.world_boxes(_scene(), types.SimpleNamespace(), _volume({}), {}, levels=(0, 2))
    assert list(level) == [2]  # the level-0 box is the whole cube


def test_consolidate_finds_separate_clumps():
    rng = np.random.default_rng(0)
    centers = np.repeat([[0.1, 0.1, 0.1], [0.8, 0.2, 0.5], [0.5, 0.9, 0.9]], 50, axis=0)
    left = centers + rng.uniform(-0.02, 0.02, centers.shape)
    right = left + 0.01
    groups = grid_overlay.consolidate(left, right, 3)
    assert sorted(len(g) for g in groups) == [50, 50, 50]
    assert all(len(set(g // 50)) == 1 for g in groups)


def test_consolidated_boxes_bound_their_members():
    corners, level = grid_overlay.world_boxes(
        _scene(), types.SimpleNamespace(), _volume({}), {}, levels=(1, None), clusters=1,
    )
    assert len(corners) == 1
    np.testing.assert_allclose(corners[0], grid_overlay.world_boxes(
        _scene(), types.SimpleNamespace(), _volume({}), {}, levels=(2, 2),
    )[0][0])  # level 2 is the 0.25-0.5 box, which contains the finer one


def test_spherical_data_has_no_boxes():
    from test_yt_idv_renderer import _spherical_collection

    scene = _block_scene(_spherical_collection())
    assert grid_overlay.world_boxes(scene, types.SimpleNamespace(), _volume({}), {}) is None
