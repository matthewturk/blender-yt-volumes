import numpy as np
import pytest

from yt_volumes import edge_blend

FAR = 1e10  # no hit
OBJECT, OTHER = 1.5, 2.5  # Cryptomatte ids


def _layers(shape, *ranks):
    """(ids, coverage) with `ranks` of (id, coverage array) per pixel."""
    ids = np.zeros(shape + (len(ranks),), np.float32)
    coverage = np.zeros_like(ids)
    for i, (id_, cov) in enumerate(ranks):
        ids[..., i] = np.where(np.asarray(cov) > 0, id_, 0.0)
        coverage[..., i] = cov
    return ids, coverage


def _volume(depth, front=0.2, back=1.0):
    """A volume layer clipped as the renderer would: dimmer where a surface cuts it."""
    value = np.where(depth < FAR, front, back).astype(np.float32)
    return np.repeat(value[..., None], 4, axis=-1)


def test_an_outline_pixel_blends_by_how_much_the_surface_covers():
    depth = np.full((12, 12), FAR, np.float32)
    depth[:, :6] = 3.0
    coverage = np.where(np.arange(12) < 6, 1.0, 0.0) * np.ones((12, 1))
    coverage[:, 5] = 0.5  # the outline: half covered, its one depth sample on the surface
    out = edge_blend.blend(_volume(depth), depth, *_layers((12, 12), (OBJECT, coverage)))

    assert out[6, 5, 0] == pytest.approx(0.6, rel=1e-6)
    np.testing.assert_array_equal(out[:, :5], np.float32(0.2))  # covered: left alone
    np.testing.assert_array_equal(out[:, 6:], 1.0)


def test_dithered_transparency_averages_out():
    rng = np.random.default_rng(1)
    depth = np.where(rng.random((40, 40)) < 0.75, 3.0, FAR).astype(np.float32)  # depth written 75% of the time
    coverage = np.full((40, 40), 0.75)
    out = edge_blend.blend(_volume(depth), depth, *_layers((40, 40), (OBJECT, coverage)))

    inner = out[5:-5, 5:-5, 0]
    assert inner.mean() == pytest.approx(0.75 * 0.2 + 0.25 * 1.0, abs=0.03)
    assert inner.std() < 0.05 < _volume(depth)[5:-5, 5:-5, 0].std()


def test_a_transparent_surface_that_always_writes_depth_stays_clipped_inside():
    depth = np.full((20, 20), FAR, np.float32)
    depth[:, :10] = 3.0
    coverage = np.where(np.arange(20) < 10, 0.75, 0.0) * np.ones((20, 1))  # opacity 0.75, as Cycles reports
    out = edge_blend.blend(_volume(depth), depth, *_layers((20, 20), (OBJECT, coverage)))

    np.testing.assert_array_equal(out[:, :10], np.float32(0.2))  # no band inside the outline


def test_only_the_nearer_objects_coverage_counts():
    depth = np.full((12, 12), 8.0, np.float32)  # a wall behind everything
    depth[:, :6] = 3.0
    front = np.where(np.arange(12) < 6, 1.0, 0.0) * np.ones((12, 1))
    front[:, 5] = 0.25
    wall = 1.0 - front
    volume = np.where(depth < 5, 0.2, 0.6).astype(np.float32)[..., None].repeat(4, -1)
    out = edge_blend.blend(volume, depth, *_layers((12, 12), (OBJECT, front), (OTHER, wall)))

    assert out[6, 5, 0] == pytest.approx(0.25 * 0.2 + 0.75 * 0.6, rel=1e-6)


def test_no_edges_no_change():
    depth = np.full((8, 8), FAR, np.float32)
    rgba = np.random.default_rng(0).random((8, 8, 4)).astype(np.float32)
    out = edge_blend.blend(rgba, depth, *_layers((8, 8), (OBJECT, np.zeros((8, 8)))))
    np.testing.assert_array_equal(out, rgba)


def test_a_faint_object_is_found_under_eevees_world_entry():
    depth = np.full((12, 12), FAR, np.float32)
    depth[:, :6] = 3.0  # an object 30% opaque that always writes depth
    faint = np.where(np.arange(12) < 6, 0.3, 0.0) * np.ones((12, 1))
    faint[:, 5] = 0.15
    world = 1.0 - faint  # id 0, ranked first wherever the object is faint
    ids, coverage = _layers((12, 12), (OBJECT, faint))
    ids = np.concatenate([np.zeros_like(ids), ids], -1)
    coverage = np.concatenate([world[..., None], coverage], -1)
    out = edge_blend.blend(_volume(depth), depth, ids, coverage)

    assert out[6, 5, 0] == pytest.approx(0.5 * 0.2 + 0.5 * 1.0, rel=1e-6)  # half the object's footprint
