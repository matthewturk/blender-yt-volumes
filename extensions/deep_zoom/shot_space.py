"""Shot space: the Blender world of a deep-zoom authoring scene.

A shot has a float64 anchor in the dataset's unitary coordinates, and the scene
an animatable zoom (log10). During a shot,

    world = (unitary - anchor) * 10**zoom * world_scale

so the anchor is always the world origin, where float32 keeps its relative
precision, and the camera moves at ordinary sizes however deep the zoom. Pure
Python, so the renderer, the add-on and tests can share it.
"""

import numpy as np

DEFAULT_WORLD_SCALE = 2.0  # world units per box width at zoom 0, like a 2x2x2 Empty-Cube


def parse_anchor(spec):
    """`"x, y, z"` (unitary) as float64; "max" passes through for the renderer."""
    if spec.strip().lower() == "max":
        return "max"
    values = spec.replace(",", " ").split()
    if len(values) != 3:
        raise ValueError(f"shot anchor {spec!r} is not 'max' or 'x, y, z'")
    return np.array([float(v) for v in values])


def format_anchor(anchor):
    """The inverse of parse_anchor; repr keeps every float64 digit."""
    return ", ".join(repr(float(v)) for v in anchor)


def world_per_unitary(zoom, world_scale=DEFAULT_WORLD_SCALE):
    return 10.0 ** zoom * world_scale


def to_world(points, anchor, zoom, world_scale=DEFAULT_WORLD_SCALE):
    """Unitary points (..., 3) as shot-space world coordinates, in float64."""
    return (np.asarray(points, dtype=np.float64) - anchor) * world_per_unitary(zoom, world_scale)


def to_unitary(points, anchor, zoom, world_scale=DEFAULT_WORLD_SCALE):
    return np.asarray(anchor, dtype=np.float64) + np.asarray(points, dtype=np.float64) / world_per_unitary(zoom, world_scale)


def active_shot(shots, frame):
    """Index of the shot playing at `frame`: the last one starting at or before
    it (the first shot if none has started). `shots` is a list of start frames.
    """
    if not shots:
        return None
    started = [i for i, start in enumerate(shots) if start <= frame]
    if not started:
        return min(range(len(shots)), key=lambda i: shots[i])
    return max(started, key=lambda i: shots[i])


def visible_boxes(left, right, anchor, zoom, world_scale=DEFAULT_WORLD_SCALE,
                  min_size=0.02, max_size=50.0, max_distance=50.0, max_count=4000):
    """Grid boxes worth drawing in shot space: between `min_size` and
    `max_size` world units across, and within `max_distance` of the origin.
    Keeps the `max_count` largest. Returns world-space (left, right, index).
    """
    left = to_world(left, anchor, zoom, world_scale)
    right = to_world(right, anchor, zoom, world_scale)
    size = (right - left).max(axis=1)
    # distance from the origin to the nearest point of each box
    gap = np.maximum(np.maximum(left, -right), 0.0)
    keep = (size >= min_size) & (size <= max_size) & (np.linalg.norm(gap, axis=1) <= max_distance)
    index = np.nonzero(keep)[0]
    if index.size > max_count:
        index = index[np.argsort(-size[index])[:max_count]]
    return left[index], right[index], index


def box_edges(left, right):
    """Vertices (8n, 3) and edges (12n, 2) of axis-aligned boxes."""
    n = len(left)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    verts = left[:, None, :] + corners[None, :, :] * (right - left)[:, None, :]
    # corner index = 4x + 2y + z; edges join corners differing in one bit
    pairs = np.array([(a, a | bit) for a in range(8) for bit in (1, 2, 4) if not a & bit])
    edges = pairs[None, :, :] + 8 * np.arange(n)[:, None, None]
    return verts.reshape(-1, 3), edges.reshape(-1, 2)


def box_faces(left, right):
    """Vertices (8n, 3) and quads (6n, 4) of axis-aligned boxes."""
    verts, _ = box_edges(left, right)
    # corner index = 4x + 2y + z, wound outward
    quads = np.array([(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    faces = quads[None, :, :] + 8 * np.arange(len(left))[:, None, None]
    return verts, faces.reshape(-1, 4)
