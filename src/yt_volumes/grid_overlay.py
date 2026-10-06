"""Viewport overlay of a volume's grid boxes.

Draws the AMR grids of the dataset the yt_idv renderer has loaded (or, for a
saved scene or non-grid dataset, yt_idv's blocks), placed with the same
registration and zoom as the render. Nothing is loaded here: an object shows
grids once it has been rendered in this session.
"""

import bpy
import numpy as np

from . import operators, properties
from .external_render import VolumeObjectParams
from . import yt_idv_renderer as yir

MAX_GRID_BOXES = 5000  # shared evenly across levels; more is unreadable well before it is slow
COARSE_COLOR = (0.2, 0.45, 1.0, 0.8)
FINE_COLOR = (0.6, 1.0, 0.85, 0.8)
LINE_WIDTH = 1.5

_unitary_cache = {}  # (id(scene), id(ds)) -> (left, right, level), unitary float64
_batch_cache = {}  # object name -> (key, batch)
_errors = {}  # object name -> last error logged
_handle = None
# corner index = 4x + 2y + z; edges join corners differing in one bit
_CORNERS = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
_EDGE_PAIRS = np.array([(a, a | bit) for a in range(8) for bit in (1, 2, 4) if not a & bit])


def _amr_levels(dx, root_dx):
    """AMR level, log2(root dx / dx), per row of `dx`."""
    return np.round(np.log2(root_dx / dx).mean(axis=1)).astype(np.int32)


def _unitary_boxes(scene, ds):
    """`(left, right, level, collection)` in unitary coordinates, with
    `collection` the index of each box's block collection: the dataset's grids
    if it has them, else the scene's blocks. None for spherical data.

    Block levels need yt_idv's `root_dx`; without it they count from the
    coarsest block.
    """
    key = (id(scene), id(ds))
    cached = _unitary_cache.get(key)
    # ids are reused once an object is freed, so check it's the same scene
    if cached is not None and cached[0] is scene and cached[1] is ds:
        return cached[2]
    collections = yir._block_collections(scene)
    boxes = None
    if collections and not any(yir._is_spherical(c) for c in collections):
        try:
            index = ds.index
            left = index.grid_left_edge.to("unitary").d.astype(np.float64)
            boxes = (
                left,
                index.grid_right_edge.to("unitary").d.astype(np.float64),
                np.asarray(index.grid_levels, dtype=np.int32).ravel(),
                np.zeros(len(left), dtype=np.int32),
            )
        except AttributeError:
            lefts, rights, dxs, owners = [], [], [], []
            for i, c in enumerate(collections):
                if hasattr(c, "_get_unscaled_edges"):
                    left, right, dx = (np.asarray(e, dtype=np.float64) for e in c._get_unscaled_edges())
                else:
                    offset = np.asarray(c.applied_scale_offset, dtype=np.float64)
                    ratio = c.applied_scale_ratio
                    left = yir._vertex_attribute(c, "in_left_edge") * ratio + offset
                    right = yir._vertex_attribute(c, "in_right_edge") * ratio + offset
                    dx = yir._vertex_attribute(c, "in_dx") * ratio
                lefts.append(left)
                rights.append(right)
                dxs.append(dx)
                owners.append(np.full(len(left), i, dtype=np.int32))
            dx = np.concatenate(dxs)
            root_dx = next((c.root_dx for c in collections if getattr(c, "root_dx", None) is not None), None)
            if root_dx is None:
                root_dx = dx[np.argmax(dx.max(axis=1))]
            level = _amr_levels(dx, np.asarray(root_dx, dtype=np.float64))
            boxes = (np.concatenate(lefts), np.concatenate(rights), level, np.concatenate(owners))
    _unitary_cache[key] = (scene, ds, boxes)
    return boxes


def loaded_boxes(obj):
    """`{"left", "right", "level"}` (unitary float64) of the boxes loaded for
    `obj`, as `grid_structure.load` returns them; None before its first render.
    """
    cached = yir._volume_scene_cache.get(obj.volume_override.dataset_uuid)
    boxes = None if cached is None else _unitary_boxes(cached[0], cached[1])
    if boxes is None or not len(boxes[2]):
        return None
    return dict(zip(("left", "right", "level"), boxes[:3]))


def loaded_levels(obj):
    """`(min, max)` level of the boxes loaded for `obj`, or None before its
    first render.
    """
    boxes = loaded_boxes(obj)
    return None if boxes is None else (int(boxes["level"].min()), int(boxes["level"].max()))


def _spread_across_levels(level, max_boxes):
    """Indices of at most `max_boxes` boxes, taking one from each level in
    turn so the finest levels aren't crowded out by the coarse ones.
    """
    order = np.argsort(level, kind="stable")
    sorted_level = level[order]
    rank = np.arange(len(order)) - np.searchsorted(sorted_level, sorted_level)
    return order[np.lexsort((sorted_level, rank))[:max_boxes]]


def consolidate(left, right, max_boxes):
    """Indices grouping the boxes into at most `max_boxes` tight clusters.

    Repeatedly splits the cluster with the longest bounding box along that
    axis, where the two halves' bounding boxes have the least total
    volume. Returns a list of index arrays.
    """
    clusters = [np.arange(len(left))]
    extents = [right.max(axis=0) - left.min(axis=0)]
    while len(clusters) < max_boxes:
        splittable = [i for i, c in enumerate(clusters) if len(c) > 1]
        if not splittable:
            break
        i = max(splittable, key=lambda i: float(extents[i].max()))
        members = clusters[i]
        axis = int(np.argmax(extents[i]))
        members = members[np.argsort(left[members, axis] + right[members, axis], kind="stable")]
        low, high = left[members], right[members]
        # bounding boxes of members[:k] and members[k:], k = 1..n-1
        head = np.maximum.accumulate(high)[:-1] - np.minimum.accumulate(low)[:-1]
        tail = np.maximum.accumulate(high[::-1])[::-1][1:] - np.minimum.accumulate(low[::-1])[::-1][1:]
        k = int(np.argmin(head.prod(axis=1) + tail.prod(axis=1))) + 1
        clusters[i:i + 1] = [members[:k], members[k:]]
        extents[i:i + 1] = [head[k - 1], tail[k - 1]]
    return clusters


def world_boxes(scene, ds, volume, custom_params, levels=(0, None), clusters=0, max_boxes=MAX_GRID_BOXES):
    """World-space `(corners, level)` of the boxes to draw: corners is
    `(n, 8, 3)`, corner index 4x + 2y + z.

    Only boxes with `levels[0] <= level <= levels[1]` (None: no upper bound),
    clipped to the volume's cube.
    With `clusters`, they are merged into at most that many bounding boxes,
    each with its finest level. `volume` is a `VolumeObjectParams`; only its
    placement is used.
    """
    boxes = _unitary_boxes(scene, ds)
    if boxes is None:
        return None
    left, right, level, owner = boxes
    # Unitary -> volume-cube coordinates, matching apply_zoom: each collection
    # has its own base transform, the zoom target stays put and everything
    # else spreads out from it by the magnification.
    collections = yir._block_collections(scene)
    bases = [yir._base_transform(c) for c in collections]
    spec = custom_params.get("yt_idv_zoom_target")
    ratio = np.array([b[1] for b in bases])[owner][:, None]
    if spec:
        targets = np.array([yir._zoom_target_unitary(spec, c) for c in collections])
        magnification = 10.0 ** float(custom_params.get("yt_idv_zoom_level", 0.0))
        anchor = (targets[0] - bases[0][0]) / bases[0][1]
        origin = targets[owner]
    else:
        magnification, anchor = 1.0, 0.0
        origin = np.array([b[0] for b in bases])[owner]

    level_min, level_max = levels
    keep = level >= level_min
    if level_max is not None:
        keep &= level <= level_max
    keep = np.nonzero(keep)[0]
    scale = magnification / ratio[keep]
    cube_left = anchor + (left[keep] - origin[keep]) * scale
    cube_right = anchor + (right[keep] - origin[keep]) * scale
    # Clip to the volume Empty; a box covering all of it would only redraw its outline.
    inside = np.all(cube_right > 0.0, axis=1) & np.all(cube_left < 1.0, axis=1)
    inside &= ~(np.all(cube_left <= 0.0, axis=1) & np.all(cube_right >= 1.0, axis=1))
    cube_left = np.clip(cube_left[inside], 0.0, 1.0)
    cube_right = np.clip(cube_right[inside], 0.0, 1.0)
    level = level[keep][inside]
    keep = keep[inside]
    if clusters:
        groups = consolidate(cube_left, cube_right, clusters) if len(keep) else []
        cube_left = np.array([cube_left[g].min(axis=0) for g in groups]).reshape(-1, 3)
        cube_right = np.array([cube_right[g].max(axis=0) for g in groups]).reshape(-1, 3)
        level = np.array([level[g].max() for g in groups], dtype=np.int32)

    keep = _spread_across_levels(level, max_boxes)

    volume = yir._shot_space_volume(scene, volume, custom_params)
    matrix = np.array(volume.matrix_world, dtype=np.float64)
    local_min, local_extent = yir._bbox_local_min_extent(volume.bounding_box_local)

    low, high = cube_left[keep], cube_right[keep]
    cube = low[:, None, :] + _CORNERS[None, :, :] * (high - low)[:, None, :]
    corners = (local_min + cube * local_extent) @ matrix[:3, :3].T + matrix[:3, 3]
    return corners, level[keep]


def _line_batch(corners, level, top_level):
    import gpu
    from gpu_extras.batch import batch_for_shader

    verts = corners.reshape(-1, 3)
    edges = (_EDGE_PAIRS[None, :, :] + 8 * np.arange(len(corners))[:, None, None]).reshape(-1)
    fraction = level / max(top_level, 1)
    colors = np.repeat(
        np.array(COARSE_COLOR) + fraction[:, None] * (np.array(FINE_COLOR) - np.array(COARSE_COLOR)), 8, axis=0
    )
    shader = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
    batch = batch_for_shader(
        shader, "LINES",
        {"pos": verts[edges].astype(np.float32), "color": colors[edges].astype(np.float32)},
    )
    return shader, batch


def _object_batch(obj, blender_scene):
    """The cached line batch for `obj`, rebuilt when its placement changes."""
    cached = yir._volume_scene_cache.get(obj.volume_override.dataset_uuid)
    if cached is None:
        return None
    scene, ds, _ = cached
    custom_params = obj.volume_override.to_custom_params()
    custom_params.update(operators._deep_zoom_params(blender_scene))
    # Not build_volume_params: that records change-tracking state.
    volume = VolumeObjectParams(
        matrix_world=operators._matrix_to_list(obj.matrix_world),
        bounding_box_local=operators._empty_cube_bounding_box_local(obj),
        custom_params=custom_params, params_version=None, dataset=None,
    )
    key = (
        id(scene), id(ds), str(volume.matrix_world), str(volume.bounding_box_local),
        custom_params.get("yt_idv_zoom_target"), custom_params.get("yt_idv_zoom_level"),
        custom_params.get("yt_idv_shot_space"),
    )
    props = obj.volume_override
    levels = (props.grid_level_min, props.grid_level_max)
    clusters = props.grid_clusters if props.grid_consolidate else 0
    key += (levels, clusters)
    entry = _batch_cache.get(obj.name)
    if entry is None or entry[0] != key:
        boxes = world_boxes(scene, ds, volume, custom_params, levels, clusters)
        drawable = boxes is not None and len(boxes[1])
        entry = (key, _line_batch(*boxes, loaded_levels(obj)[1]) if drawable else None)
        _batch_cache[obj.name] = entry
    return entry[1]


def _draw():
    import gpu

    context = bpy.context
    blender_scene = context.scene
    batches = []
    for obj in properties.flagged_objects(blender_scene):
        if not obj.volume_override.show_grids or not obj.visible_get():
            continue
        try:
            batch = _object_batch(obj, blender_scene)
        except Exception as exc:  # never break the viewport
            if _errors.get(obj.name) != repr(exc):
                _errors[obj.name] = repr(exc)
                operators._log(f"grid overlay for {obj.name!r}: {exc!r}")
            continue
        if batch is not None:
            batches.append(batch)
    if not batches:
        return
    region = context.region
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    for shader, batch in batches:
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", LINE_WIDTH)
        batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def register():
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
    _batch_cache.clear()
    _unitary_cache.clear()
