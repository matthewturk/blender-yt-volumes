"""Smooth where surfaces cut a volume layer, using Cryptomatte coverage.

The Z pass holds one sample per pixel, so the volume's depth clip is all or
nothing in each pixel: jagged along surface outlines, and speckled across
Dithered transparent materials, whose depth is written for a random share of
pixels. Cryptomatte coverage is accumulated over every sample, so where a
pixel's neighbourhood spans two depths, the volume layer is blended between
its value in front of the nearer surface and behind it, weighted by how much
of the pixel the nearer surface covers.
"""

import os

import bpy
import numpy as np

SCENE_PROPERTY = "yt_volumes_smooth_edges"
RADIUS = 3  # pixels each side of an edge pixel to take the front and back volume from
DEPTH_RATIO = 1.01  # a neighbourhood spans an edge when its depths differ by more than this factor
COVERAGE_EPSILON = 0.005  # pixels this close to fully covered or uncovered are left alone
CHUNK = 20000  # edge pixels per vectorized batch, bounding memory
_IMAGE_PREFIX = "vo_crypto_tmp"


def register():
    setattr(bpy.types.Scene, SCENE_PROPERTY, bpy.props.BoolProperty(
        name="Smooth Surface Edges",
        description=(
            "Blend volumes where surfaces cut them by how much of each pixel the surface covers, "
            "smoothing jagged outlines and the speckle behind Dithered transparent materials. "
            "Adds a Cryptomatte pass to the render"
        ),
        default=False,
    ))


def unregister():
    if hasattr(bpy.types.Scene, SCENE_PROPERTY):
        delattr(bpy.types.Scene, SCENE_PROPERTY)


def enabled(scene):
    return bool(getattr(scene, SCENE_PROPERTY, False))


def layer_count(view_layer):
    """Cryptomatte EXR layers (CryptoObject00, 01, ...): two ranks each."""
    return (view_layer.pass_cryptomatte_depth + 1) // 2


def read_cryptomatte(paths):
    """`(ids, coverage)`, each `(height, width, ranks)` float32, from the
    Cryptomatte layer EXRs at `paths`, rows bottom first as Blender loads them.
    """
    layers = []
    for i, path in enumerate(paths):
        name = f"{_IMAGE_PREFIX}{i:02d}"
        image = bpy.data.images.get(name)
        if image is None:
            image = bpy.data.images.load(path, check_existing=False)
            image.name = name
        else:
            image.filepath = path
            image.reload()
        # ids are float bit patterns: no color conversion, no alpha association
        image.colorspace_settings.name = "Non-Color"
        image.alpha_mode = "CHANNEL_PACKED"
        width, height = image.size
        if not width or not image.channels:
            raise RuntimeError(f"Cryptomatte layer {os.path.basename(path)!r} loaded empty")
        pixels = np.empty(width * height * image.channels, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        layers.append(pixels.reshape(height, width, image.channels)[..., :4])
    stacked = np.concatenate(layers, axis=-1)  # id, coverage, id, coverage, ...
    return stacked[..., 0::2], stacked[..., 1::2]


def _window_extreme(a, radius, combine):
    """`a`'s minimum or maximum (`combine` np.minimum or np.maximum) over each pixel's window."""
    k = 2 * radius + 1
    padded = np.pad(a, radius, mode="edge")
    rows = padded[:, : padded.shape[1] - k + 1].copy()
    for i in range(1, k):
        combine(rows, padded[:, i : i + rows.shape[1]], out=rows)
    out = rows[: rows.shape[0] - k + 1].copy()
    for i in range(1, k):
        combine(out, rows[i : i + out.shape[0]], out=out)
    return out


def blend(rgba, depth, ids, coverage, radius=RADIUS, ratio=DEPTH_RATIO):
    """`rgba` (premultiplied, same rows as `depth`) blended across depth edges.

    Where a pixel's neighbourhood holds depths more than `ratio` apart, splits
    it at their geometric mean into front and back pixels, and gives the pixel
    `w * front + (1 - w) * back` (each the mean volume over those pixels).
    `w` is the pixel's coverage by the nearest object, divided by that
    object's opacity if it writes depth wherever it is (a Dithered material
    writes it for a random share of pixels instead, so needs no division).
    """
    rgba = np.asarray(rgba, dtype=np.float32)
    near_min = _window_extreme(depth, radius, np.minimum)
    far_max = _window_extreme(depth, radius, np.maximum)
    ys, xs = np.nonzero(far_max > near_min * ratio)
    if not len(ys):
        return rgba
    k = 2 * radius + 1
    width = depth.shape[1] + 2 * radius
    offsets = (np.arange(k)[:, None] * width + np.arange(k)).ravel()  # window, as flat padded indices
    edge_pad = ((radius, radius), (radius, radius), (0, 0))
    depth_pad = np.pad(depth, radius, mode="edge").ravel()
    rgba_pad = np.pad(rgba, edge_pad, mode="edge").reshape(-1, 4)
    # The window statistics use the top two ranks, which hold any object that covers much of a pixel.
    if ids.shape[-1] < 2:
        ids, coverage = (np.concatenate([a, np.zeros_like(a)], -1) for a in (ids, coverage))
    top_ids = np.pad(ids[..., :2], edge_pad, mode="edge").reshape(-1, 2)
    top_coverage = np.pad(coverage[..., :2], edge_pad, mode="edge").reshape(-1, 2)
    out = rgba.copy()
    for start in range(0, len(ys), CHUNK):
        y, x = ys[start:start + CHUNK], xs[start:start + CHUNK]
        window = (y * width + x)[:, None] + offsets  # (n, k*k): the window's top-left is the pixel minus radius
        window_depth = np.take(depth_pad, window)
        split = np.sqrt(near_min[y, x] * far_max[y, x])[:, None]
        near = window_depth < split
        volume = np.take(rgba_pad, window, axis=0)  # (n, k*k, 4)
        near_count = np.count_nonzero(near, axis=1)
        weights = np.ones((len(y), 2, k * k), np.float32)
        weights[:, 0] = near
        front_sum, total = np.moveaxis(weights @ volume, 1, 0)
        front = front_sum / near_count[:, None]
        back = (total - front_sum) / (k * k - near_count)[:, None]
        # the nearest object: the main one where the window is nearest (id 0 is
        # EEVEE's world, which outranks objects less than half opaque)
        rows = np.arange(len(y))
        nearest = window[rows, window_depth.argmin(1)]
        nearest_ids, nearest_coverage = top_ids[nearest], top_coverage[nearest]
        front_id = nearest_ids[rows, np.where(nearest_ids != 0, nearest_coverage, -1.0).argmax(1)]
        window_ids, window_cov = np.take(top_ids, window, axis=0), np.take(top_coverage, window, axis=0)
        front_column = front_id[:, None]
        window_coverage = (window_cov[..., 0] * (window_ids[..., 0] == front_column)
                           + window_cov[..., 1] * (window_ids[..., 1] == front_column))
        c = (coverage[y, x] * (ids[y, x] == front_id[:, None])).sum(-1)  # every rank, for the pixel itself
        # Inside the object (outline pixels excluded), a dithered one leaves depth
        # unwritten for about (1 - opacity) of pixels; one that writes it, for none.
        inside = window_coverage >= 0.9 * window_coverage.max(1, keepdims=True)
        inside_count = np.count_nonzero(inside, axis=1)
        opacity = np.where(inside, window_coverage, 0.0).sum(1) / inside_count
        unwritten = np.count_nonzero(~near & inside, axis=1) / inside_count
        dithered = unwritten > (1.0 - opacity) / 4
        w = np.clip(np.where(dithered, c, c / np.maximum(opacity, 1e-6)), 0.0, 1.0)
        partial = (w > COVERAGE_EPSILON) & (w < 1.0 - COVERAGE_EPSILON)
        w = w[partial, None]
        out[y[partial], x[partial]] = w * front[partial] + (1.0 - w) * back[partial]
    return out
